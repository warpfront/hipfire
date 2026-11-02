#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma-2 (arch 18) end-to-end parity check — the oracle leg.

Drives the daemon's `embed` verb over a safetensors dir and compares the
served 768-d embeddings (and their Matryoshka truncations) against a
float32 CPU `transformers` reference produced by the committed
`scripts/make_embeddinggemma2_reference.py` (google/embeddinggemma-2,
L2), plus the edge-case contract: image input refused, empty text refused,
oversized and non-integer `dims` refused, prompt-exceeds-limit refused, bad
kind refused, and >8192-token input truncated to exactly 8192.

Expect hipfire (f16 GEMM weights, f32 activations) at ~1e-3 relative-L2
against the f32 reference; the band is 2e-2 (two orders of margin, still
tight enough that a misread convention — wrong norm order, missing PLE,
wrong window — fails by O(1)).

Usage:
    python3 scripts/check_embeddinggemma2_parity.py \
        --model <dir-with-model.safetensors> --ref <ref.npz> \
        [--daemon target/release/daemon]

Vision legs (default on, --no-vision to skip): require the reference to carry
the `img_*` keys the extended generator writes, resolve/emit the `.vl` sidecar
next to the model (see ensure_vision_sidecar), and drive the expanded image
requests through the same `embed` verb. The tower-less image refusal is
asserted against a sidecar-less symlink farm so both behaviors are covered in
one run.

Audio and video legs ride the same `vision_on` guard and are skipped
individually when the reference predates them. Video goes over the wire as
`videos_base64`: an array of VIDEOS, each a list of base64 frame images in
display order (one `<|video|>` marker per video in the text), mirroring the
frame-per-index pixels of the generator.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np

Q = "task: search result | query: "
D = "title: none | text: "

# (ref.npz key, kind, prompt+text) — texts byte-identical to make_reference.py.
CASES = [
    ("short_query", "query", "what is hipfire"),
    ("short_doc", "document", "hipfire is an inference engine for AMD GPUs."),
    ("code_doc", "document", "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    return quicksort([x for x in a[1:] if x < a[0]]) + [a[0]] + quicksort([x for x in a[1:] if x >= a[0]])"),
    ("unicode", "document", "Straße café naïve 日本語のテキストです。🚀 ünïcödé"),
    ("emoji", "query", "fire rocket 🚀🔥 lol"),
    ("one_char", "document", "x"),
    ("medium_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 40).strip()),
    # Past the 512-radius window and NOT 64-aligned: the eager path runs with
    # both the sliding band mask and padding rows in play.
    ("window_pad_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 90).strip()),
    ("long_doc", "document", ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 1200).strip()),
]

MRL = [512, 256, 128]
BAND = 2e-2

# (ref.npz key, kind, prompt+text, png bytes) — keep in lockstep with
# make_embeddinggemma2_reference.py IMAGE_CASES (same keys, same pixels).
# The wire field is the contract §C `images_base64` (the v0 refusal leg's
# singular `image_base64` shape is gone once towers land).
IMAGE_CASES = [
    ("img_square_256", "query", "what is in this image <|image|>", (1024, 1024, "grad")),
    ("img_wide_264tok", "document", "<|image|> a wide screenshot", (270, 480, "checker")),
    ("img_tiny", "document", "tiny <|image|>", (48, 48, "solid")),
    ("img_extreme_aspect", "document", "extremely <|image|> wide", (8, 8000, "grad")),
    ("img_nonmultiple_pool", "query", "non <|image|> multiple of the pool grid", (222, 384, "noise")),
    ("img_with_marker", "query", "<|image|> describe in detail", (336, 240, "checker")),
]

# (ref.npz key, kind, prompt+text, duration_s) — lockstep with the generator's
# AUDIO_CASES. Wire field: `audios_base64` (WAV bytes), mirroring images.
# M is the uncapped mask-subsample count (15 s → 375 > 280 — no cap exists).
AUD_TOK = "<" + "|audio|>"  # the marker literal, kept split to survive transport
AUDIO_CASES = [
    ("aud_short_3", "query", "what does this sound like <|audio|>", 0.1),
    ("aud_half_13", "document", "<|audio|> half a second of tone", 0.5),
    ("aud_1s_25", "query", "one second <|audio|> of sine", 1.0),
    ("aud_2s_50", "document", "two seconds of <|audio|> sine", 2.0),
    ("aud_cap_280", "query", "at the <|audio|> cap", 11.2),
    ("aud_15s_375", "document", "fifteen seconds of " + AUD_TOK + " tone", 15.0),
]

# Video cases: (key, kind, prompt+text, (num_frames, height, width, pattern)).
# ONE `<|video|>` marker per VIDEO (id 258884). The processor expands it into
# one `boi + <|video|>×S + eoi` block PER FRAME (S ≤ 140 soft tokens/frame).
# Frames differ by index, and the sampler is exercised by a ≤4-frame clip, an
# exactly-32-frame clip and a 77-frame clip (uniform linspace down to
# max_frames=32).
VIDEO_CASES = [
    ("vid_4_frames", "query", "what is in this clip <|video|>", (4, 240, 336, "checker")),
    ("vid_32_frames", "document", "<|video|> one full sampling window", (32, 192, 192, "grad")),
    ("vid_77_sampled", "query", "a longer clip <|video|> sampled down", (77, 168, 288, "noise")),
    ("vid_tall_aspect", "document", "<|video|> a tall portrait clip", (5, 480, 144, "grad")),
    ("vid_marker_mid", "query", "clip starts <|video|> clip ends", (3, 96, 96, "solid")),
]


def _wav_b64(seconds):
    # 440 Hz sine at 16 kHz, float32 — byte-identical samples to the
    # generator's _sine.
    import base64
    import io

    import soundfile as sf

    sr = 16000
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    wav = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_bytes(h, w, kind):
    # Byte-identical pixels to the generator's _pattern/_image_bytes.
    import base64
    import io

    from PIL import Image

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w]
    if kind == "grad":
        base = ((x * 255) // max(w - 1, 1) + (y * 255) // max(h - 1, 1)) // 2
        arr = np.stack([base, (base * 3) // 4, 255 - base], -1).astype(np.uint8)
    elif kind == "checker":
        block = max(min(h, w) // 8, 2)
        c = (((x // block) + (y // block)) % 2) * 200 + 40
        arr = np.stack([c, 255 - c, c], -1).astype(np.uint8)
    elif kind == "solid":
        arr = np.full((h, w, 3), 137, np.uint8)
    elif kind == "noise":
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        raise ValueError(kind)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _video_frame(h, w, kind, t):
    """Frame `t` of a synthetic clip: a deterministic function of its index,
    so frame selection and frame order are both exercised."""
    rng = np.random.default_rng(t)
    y, x = np.mgrid[0:h, 0:w]
    if kind == "grad":
        base = ((x * 255) // max(w - 1, 1) + (y * 255) // max(h - 1, 1) + 17 * t) % 256
        return np.stack([base, (base * 3) // 4, 255 - base], -1).astype(np.uint8)
    if kind == "checker":
        block = max(min(h, w) // 8, 2)
        c = (((x // block) + (y // block) + t) % 2) * 200 + 40
        return np.stack([c, 255 - c, c], -1).astype(np.uint8)
    if kind == "solid":
        return np.full((h, w, 3), (137 + 13 * t) % 256, np.uint8)
    if kind == "noise":
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    raise ValueError(kind)


def _png_frame_b64(h, w, kind, t):
    # Byte-identical frame pixels to the generator's _video_frame/_video_bytes.
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_video_frame(h, w, kind, t)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ensure_vision_sidecar(model_dir, quantizer_bin):
    """Return a model dir whose load discovers the arch-18 `.vl` pack.

    Discovery for a dir source probes `<dir>.vl` (a SIBLING of the directory —
    `hipfire_runtime::sidecar::sidecar_candidates` with no stem to strip), so
    the helper: (a) accepts an existing sibling pack, else (b) builds a farm —
    `farm/<dirname>/` symlinked to the model's entries plus the pack emitted by
    the quantizer at `farm/<dirname>.vl` — and hands back the farm copy. The
    user's model dir is never written; the quantizer's trunk artifact lands
    beside the pack as a sibling, never inside the loaded dir.
    """
    import glob as globmod
    import os as osmod
    import subprocess as sp
    import tempfile as tf

    name = osmod.path.basename(osmod.path.normpath(model_dir))
    if osmod.path.exists(osmod.path.join(osmod.path.dirname(osmod.path.normpath(model_dir)), f"{name}.vl")):
        return model_dir
    if globmod.glob(osmod.path.join(model_dir, "*.vl")):
        return model_dir
    if not quantizer_bin or not osmod.path.exists(quantizer_bin):
        raise SystemExit(
            "vision legs requested but no `<dir>.vl` sibling pack for the model; "
            "emit one with the quantizer (pass --quantizer) or rerun with --no-vision"
        )
    farm = tf.mkdtemp(prefix="eg2-parity-vl-")
    copy = osmod.path.join(farm, name)
    osmod.mkdir(copy)
    for entry in osmod.listdir(model_dir):
        osmod.symlink(osmod.path.join(model_dir, entry), osmod.path.join(copy, entry))
    r = sp.run([quantizer_bin, "--input", model_dir,
                "--output", osmod.path.join(farm, f"{name}.hfq")],
               capture_output=True, text=True)
    pack = osmod.path.join(farm, f"{name}.vl")
    if r.returncode != 0 or not osmod.path.exists(pack):
        raise SystemExit(
            f"sidecar emission failed ({r.returncode}):\n{r.stdout[-800:]}\n{r.stderr[-800:]}"
        )
    return copy


class Daemon:
    def __init__(self, binary, model, params=None):
        self.proc = subprocess.Popen(
            [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._stderr_buf = []
        threading.Thread(target=self._drain, daemon=True).start()
        self._wlock = threading.Lock()
        self._pending = {}
        self._plock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()
        self.send({"type": "load", "model": model, "params": params or {}})
        # The `loaded` reply to a load carries no id — match on None.
        self.loaded = self._wait("loaded", None)

    def _drain(self):
        for line in self.proc.stderr:
            self._stderr_buf.append(line)

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._plock:
                q = self._pending.get(d.get("id"))
            if q is not None:
                q.put(d)

    def send(self, obj):
        with self._wlock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _wait(self, want, rid, timeout=900.0):
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            d = q.get(timeout=timeout)
        except queue.Empty:
            # A second daemon cannot load while another holds the GPU lock, so
            # "no reply" usually means the child died — surface its stderr
            # rather than a bare timeout.
            tail = "".join(self._stderr_buf[-6:])
            raise RuntimeError(
                f"daemon gave no {want!r} reply in {timeout:.0f}s — stderr tail:\n{tail}"
            )
        finally:
            with self._plock:
                self._pending.pop(rid, None)
        if d.get("type") == "error":
            raise RuntimeError(f"daemon error: {d.get('message')}")
        return d

    def embed(self, rid, kind, input_obj, dims=None, expect_error=False):
        req = {"type": "embed", "id": str(rid), "kind": kind, "input": input_obj}
        if dims is not None:
            req["dims"] = dims
        d = self._wait_embed(req, str(rid))
        if expect_error:
            if d.get("type") != "error":
                raise AssertionError(f"expected an error, got {d.get('type')}: {str(d)[:200]}")
            return d.get("message", "")
        if d.get("type") == "error":
            raise RuntimeError(f"embed {rid}: {d.get('message')}")
        return d

    def _wait_embed(self, req, rid):
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            self.send(req)
            return q.get(timeout=900.0)
        finally:
            with self._plock:
                self._pending.pop(rid, None)

    def stop(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def rel_l2(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--no-vision", action="store_true",
                    help="skip the media legs (v0 text-only contract)")
    ap.add_argument("--quantizer", default="target/release/hipfire-quantize",
                    help="quantizer binary used to emit a missing .vl sidecar")
    args = ap.parse_args()

    ref = np.load(args.ref)
    have_vision_ref = f"ids_{IMAGE_CASES[0][0]}" in ref
    vision_on = have_vision_ref and not args.no_vision
    model_dir = args.model
    if vision_on:
        model_dir = ensure_vision_sidecar(args.model, args.quantizer)
    d = Daemon(args.daemon, model_dir,
               params={"vision_mode": "auto" if vision_on else "off"})
    print(f"loaded: arch={d.loaded.get('arch')} vision={'on' if vision_on else 'off'}")

    failures = 0
    rid = 0
    for key, kind, text in CASES:
        rid += 1
        r = d.embed(rid, kind, {"text": text})
        ref_ids = ref[f"ids_{key}"]
        if r["n_input_tokens"] != len(ref_ids):
            print(f"FAIL {key}: n_input_tokens {r['n_input_tokens']} != ref {len(ref_ids)}")
            failures += 1
            continue
        got = np.array(r["embeddings"], dtype=np.float32)
        e = rel_l2(got, ref[f"emb_{key}"])
        ok = e <= BAND
        print(f"{'ok  ' if ok else 'FAIL'} {key}: n={r['n_input_tokens']} dim={r['dim']} rel-L2 {e:.2e}")
        failures += 0 if ok else 1
        # Matryoshka truncations must reproduce the reference's truncated
        # re-normalized vectors.
        for m in MRL:
            rid += 1
            r2 = d.embed(rid, kind, {"text": text}, dims=m)
            got2 = np.array(r2["embeddings"], dtype=np.float32)
            e2 = rel_l2(got2, ref[f"emb_{key}_{m}"])
            ok2 = e2 <= BAND and r2["dim"] == m and len(got2) == m
            print(f"{'ok  ' if ok2 else 'FAIL'}   mrl {m}: rel-L2 {e2:.2e} dim {r2['dim']}")
            failures += 0 if ok2 else 1

    # ── vision legs ───────────────────────────────────────────────────
    if vision_on:
        for key, kind, text, (h, w, pat) in IMAGE_CASES:
            rid += 1
            img_b64 = _png_bytes(h, w, pat)
            r = d.embed(rid, kind, {"text": text, "images_base64": [img_b64]})
            ref_ids = ref[f"ids_{key}"]
            if r["n_input_tokens"] != len(ref_ids):
                print(f"FAIL {key}: n_input_tokens {r['n_input_tokens']} != ref {len(ref_ids)}")
                failures += 1
                continue
            got = np.array(r["embeddings"], dtype=np.float32)
            e = rel_l2(got, ref[f"emb_{key}"])
            ok = e <= BAND
            n_soft = int(ref[f"nsoft_{key}"])
            print(f"{'ok  ' if ok else 'FAIL'} {key}: n={r['n_input_tokens']} ({n_soft} soft) rel-L2 {e:.2e}")
            failures += 0 if ok else 1
            # One MRL width per image case bounds the run; the text cases
            # already cover the full truncation ladder.
            rid += 1
            r2 = d.embed(rid, kind, {"text": text, "images_base64": [img_b64]}, dims=512)
            got2 = np.array(r2["embeddings"], dtype=np.float32)
            e2 = rel_l2(got2, ref[f"emb_{key}_512"])
            ok2 = e2 <= BAND and r2["dim"] == 512 and len(got2) == 512
            print(f"{'ok  ' if ok2 else 'FAIL'}   mrl 512: rel-L2 {e2:.2e}")
            failures += 0 if ok2 else 1

        # Marker-count mismatch: two markers, one image → pointing error
        # (HF validate_inputs parity, mirrored by the wire layer).
        rid += 1
        msg = d.embed(rid, "query",
                      {"text": "<|image|> and <|image|>",
                       "images_base64": [_png_bytes(48, 48, "solid")]},
                      expect_error=True)
        ok = "marker" in msg
        print(f"{'ok  ' if ok else 'FAIL'} marker-count mismatch refused: {msg[:90]}")
        failures += 0 if ok else 1

    # ── audio legs ────────────────────────────────────────────────────
    if vision_on and f"ids_{AUDIO_CASES[0][0]}" in ref:
        for key, kind, text, seconds in AUDIO_CASES:
            rid += 1
            wav_b64 = _wav_b64(seconds)
            r = d.embed(rid, kind, {"text": text, "audios_base64": [wav_b64]})
            ref_ids = ref[f"ids_{key}"]
            if r["n_input_tokens"] != len(ref_ids):
                print(f"FAIL {key}: n_input_tokens {r['n_input_tokens']} != ref {len(ref_ids)}")
                failures += 1
                continue
            got = np.array(r["embeddings"], dtype=np.float32)
            e = rel_l2(got, ref[f"emb_{key}"])
            ok = e <= BAND
            n_soft = int(ref[f"nsoft_{key}"])
            print(f"{'ok  ' if ok else 'FAIL'} {key}: n={r['n_input_tokens']} ({n_soft} soft) rel-L2 {e:.2e}")
            failures += 0 if ok else 1
            rid += 1
            r2 = d.embed(rid, kind, {"text": text, "audios_base64": [wav_b64]}, dims=512)
            got2 = np.array(r2["embeddings"], dtype=np.float32)
            e2 = rel_l2(got2, ref[f"emb_{key}_512"])
            ok2 = e2 <= BAND and r2["dim"] == 512 and len(got2) == 512
            print(f"{'ok  ' if ok2 else 'FAIL'}   mrl 512: rel-L2 {e2:.2e}")
            failures += 0 if ok2 else 1

        # Audio marker-count mismatch: two markers, one clip.
        rid += 1
        msg = d.embed(rid, "query",
                      {"text": "<|audio|> and <|audio|>",
                       "audios_base64": [_wav_b64(0.5)]},
                      expect_error=True)
        ok = "marker" in msg
        print(f"{'ok  ' if ok else 'FAIL'} audio marker-count mismatch refused: {msg[:90]}")
        failures += 0 if ok else 1
    elif vision_on:
        print("SKIP audio legs: reference predates the audio cases")

    # ── video legs ────────────────────────────────────────────────────
    if vision_on and f"ids_{VIDEO_CASES[0][0]}" in ref:
        for key, kind, text, (n, h, w, pat) in VIDEO_CASES:
            rid += 1
            frames = [_png_frame_b64(h, w, pat, t) for t in range(n)]
            r = d.embed(rid, kind, {"text": text, "videos_base64": [frames]})
            ref_ids = ref[f"ids_{key}"]
            if r["n_input_tokens"] != len(ref_ids):
                print(f"FAIL {key}: n_input_tokens {r['n_input_tokens']} != ref {len(ref_ids)}")
                failures += 1
                continue
            got = np.array(r["embeddings"], dtype=np.float32)
            e = rel_l2(got, ref[f"emb_{key}"])
            ok = e <= BAND
            n_soft = int(ref[f"nsoft_{key}"])
            n_frames = int(ref[f"nframes_{key}"])
            print(f"{'ok  ' if ok else 'FAIL'} {key}: n={r['n_input_tokens']} "
                  f"({n_soft} soft / {n_frames} frames) rel-L2 {e:.2e}")
            failures += 0 if ok else 1
            rid += 1
            r2 = d.embed(rid, kind, {"text": text, "videos_base64": [frames]}, dims=512)
            got2 = np.array(r2["embeddings"], dtype=np.float32)
            e2 = rel_l2(got2, ref[f"emb_{key}_512"])
            ok2 = e2 <= BAND and r2["dim"] == 512 and len(got2) == 512
            print(f"{'ok  ' if ok2 else 'FAIL'}   mrl 512: rel-L2 {e2:.2e}")
            failures += 0 if ok2 else 1

        # Video marker-count mismatch: two markers, one video.
        rid += 1
        msg = d.embed(rid, "query",
                      {"text": "<|video|> and <|video|>",
                       "videos_base64": [[_png_frame_b64(48, 48, "solid", 0)]]},
                      expect_error=True)
        ok = "marker" in msg
        print(f"{'ok  ' if ok else 'FAIL'} video marker-count mismatch refused: {msg[:90]}")
        failures += 0 if ok else 1

        # A frame that is not PNG/JPEG is refused with the extraction recipe
        # to run, never handed to the decoder.
        import base64 as b64mod
        rid += 1
        msg = d.embed(rid, "query",
                      {"text": "clip <|video|>",
                       "videos_base64": [[b64mod.b64encode(b"not an image at all").decode("ascii")]]},
                      expect_error=True)
        ok = "ffmpeg" in msg
        print(f"{'ok  ' if ok else 'FAIL'} non-image video frame refused: {msg}")
        failures += 0 if ok else 1
    elif vision_on:
        print("SKIP video legs: reference predates the video cases")

    # ── edge cases ────────────────────────────────────────────────────
    if not vision_on:
        # Tower-less load (or --no-vision): image input must be refused with a
        # pointing error naming the modality and a remedy (the daemon gate says
        # "no vision encoder loaded ... vision_mode is off", the arch const says
        # "no vision tower is resident ... put the sidecar"), never silently
        # ignored.
        rid += 1
        msg = d.embed(rid, "query", {"text": "a cat", "images_base64": ["iVBORw0KGgo="]}, expect_error=True)
        ok = "vision" in msg and ("vision_mode" in msg or "sidecar" in msg or "tower" in msg)
        print(f"{'ok  ' if ok else 'FAIL'} image input refused: {msg[:90]}")
        failures += 0 if ok else 1
        rid += 1
        msg = d.embed(rid, "query", {"text": "a cat", "audios_base64": [_wav_b64(0.5)]}, expect_error=True)
        ok = "audio" in msg and ("vision_mode" in msg or "sidecar" in msg or "tower" in msg)
        print(f"{'ok  ' if ok else 'FAIL'} audio input refused: {msg[:90]}")
        failures += 0 if ok else 1
        rid += 1
        msg = d.embed(rid, "query",
                      {"text": "a clip", "videos_base64": [[_png_frame_b64(48, 48, "solid", 0)]]},
                      expect_error=True)
        ok = "vision" in msg and ("vision_mode" in msg or "sidecar" in msg or "tower" in msg)
        print(f"{'ok  ' if ok else 'FAIL'} video input refused: {msg[:90]}")
        failures += 0 if ok else 1

    rid += 1
    msg = d.embed(rid, "query", {"text": ""}, expect_error=True)
    ok = bool(msg)
    print(f"{'ok  ' if ok else 'FAIL'} empty text refused: {msg[:90]}")
    failures += 0 if ok else 1

    rid += 1
    msg = d.embed(rid, "query", {"text": "hello"}, dims=999, expect_error=True)
    ok = "999" in msg
    print(f"{'ok  ' if ok else 'FAIL'} oversized dims refused: {msg[:90]}")
    failures += 0 if ok else 1

    # kind is validated by parse_kind — drive it through a raw request:
    d.send({"type": "embed", "id": "badkind", "kind": "poem", "input": "x"})
    try:
        resp = d._wait(None, "badkind")
        ok = False
        detail = str(resp)[:90]
    except RuntimeError as e:
        ok = True
        detail = str(e)[:90]
    print(f"{'ok  ' if ok else 'FAIL'} bad kind refused: {detail}")
    failures += 0 if ok else 1

    # Long doc truncation: the reference was tokenized with HF truncation at
    # 8192; hipfire must land on exactly 8192 input tokens.
    rid += 1
    r = d.embed(rid, "document", {"text": CASES[-1][2]})
    ok = r["n_input_tokens"] == len(ref["ids_long_doc"]) == 8192
    print(f"{'ok  ' if ok else 'FAIL'} long doc truncated to {r['n_input_tokens']} (ref {len(ref['ids_long_doc'])})")
    failures += 0 if ok else 1

    # dims must be a strictly non-negative integer: a float/string silently
    # meaning "full width" is a protocol trap, not a convenience.
    rid += 1
    msg = d.embed(rid, "query", {"text": "hello"}, dims=12.5, expect_error=True)
    ok = "non-negative integer" in msg
    print(f"{'ok  ' if ok else 'FAIL'} non-integer dims refused: {msg[:90]}")
    failures += 0 if ok else 1

    d.stop()

    # Prompt-exceeds-limit: truncation makes any *body* fit, so the guard runs
    # on the kind prompt's own tokenization. The shipped checkpoint declares no
    # length keys, so the cheapest honest fixture is a symlink farm over the
    # same weights with a 4-token query limit. (`d` is stopped first: a second
    # daemon cannot load while another holds the GPU lock.)
    with tempfile.TemporaryDirectory() as tiny:
        for entry in os.listdir(args.model):
            if entry == "config_sentence_transformers.json":
                continue  # written fresh below (a symlink would write through)
            os.symlink(os.path.join(args.model, entry), os.path.join(tiny, entry))
        with open(os.path.join(tiny, "config_sentence_transformers.json"), "w") as fh:
            json.dump(
                {"prompts": {"query": Q, "document": D},
                 "query_length": 4, "document_length": 8192},
                fh,
            )
        tiny_d = Daemon(args.daemon, tiny)
        msg = tiny_d.embed("tinyq", "query", {"text": "hello world"}, expect_error=True)
        ok = "leaving no room" in msg
        print(f"{'ok  ' if ok else 'FAIL'} prompt-exceeds-limit refused: {msg[:90]}")
        failures += 0 if ok else 1
        r = tiny_d.embed("tinyd", "document", {"text": "hello world"})
        ok = r.get("type") == "embedded"
        print(f"{'ok  ' if ok else 'FAIL'} document under its own 8192 limit still encodes")
        failures += 0 if ok else 1
        tiny_d.stop()

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
