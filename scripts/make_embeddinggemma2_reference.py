#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma-2 (arch 18) parity reference generator — the `ref.npz`
`check_embeddinggemma2_parity.py` compares against.

float32 CPU `transformers`: `AutoModel` (google/embeddinggemma-2) →
last_hidden_state (the checkpoint's own 768-d `embedding_projection`)
→ sentence-transformers mean pooling over every token (the prompts are
pooled too, `1_Pooling` `include_prompt: true`) → L2 → Matryoshka
truncations 512/256/128 re-normalized. Texts and kind prompts are
byte-identical to the parity script's CASES; HF `tokenizers` truncation
at 8192 (special-token-aware, bos/eos frame survives) matches
`assemble_input`'s body-only cut, so `ids_<key>` asserts the wire path's
token count bit-for-bit.

Usage:
    python3 scripts/make_embeddinggemma2_reference.py \
        --model <dir-with-model.safetensors> --out ref.npz

Adds the vision legs (IMAGE_CASES below): the text carries one `<|image|>`
marker per image; the HF `EmbeddingGemma2Processor` expands each marker into
`boi + <|image|>×N + eoi` (N dynamic ≤ 280, `num_soft_tokens_per_image`), the
tower output is spliced at those positions, and the SAME mean-pool/L2/MRL head
produces the reference vector. `ids_<img key>` is the expanded token sequence
the daemon must reproduce bit-for-bit; `nsoft_<key>` is N.

Adds the audio legs (AUDIO_CASES below): one `<|audio|>` (id 258881) per clip,
expanded into `boa + <|audio|>×M + eoa` with M the uncapped mask-subsample
count; the clip is a deterministic 440 Hz sine at 16 kHz.

Adds the video legs (VIDEO_CASES below): one `<|video|>` (id 258884) per VIDEO,
expanded by `replace_video_token` into one `boi + <|video|>×S + eoi` block PER
FRAME (S ≤ 140 per-frame soft tokens, the checkpoint's `max_soft_tokens`), so
`nsoft_<key>` is the TOTAL soft slots (`S × n_frames`) while `nframes_<key>` is
how many frames the sampler kept (max_frames=32, `overflow_strategy="uniform"`).
Frames are a deterministic function of their index, so a frame-selection or
frame-order bug cannot pass.

Needs torch (CPU suffices) + transformers >= 5.18 (`embedding_gemma2`
architecture) + numpy + pillow.
"""
import argparse
import glob
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

Q = "task: search result | query: "
D = "title: none | text: "

# Byte-identical to check_embeddinggemma2_parity.py CASES — keep the two in
# lockstep; the parity script indexes this file by key.
CASES = [
    ("short_query", "query", "what is hipfire"),
    ("short_doc", "document", "hipfire is an inference engine for AMD GPUs."),
    ("code_doc", "document", "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    return quicksort([x for x in a[1:] if x < a[0]]) + [a[0]] + quicksort([x for x in a[1:] if x >= a[0]])"),
    ("unicode", "document", "Straße café naïve 日本語のテキストです。🚀 ünïcödé"),
    ("emoji", "query", "fire rocket 🚀🔥 lol"),
    ("one_char", "document", "x"),
    ("medium_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 40).strip()),
    ("window_pad_doc", "document", ("The quick brown fox jumps over the lazy dog. " * 90).strip()),
    ("long_doc", "document", ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. " * 1200).strip()),
]

MRL = [512, 256, 128]

# Vision cases: (key, kind, text, (H, W, pattern)). The text MUST carry the
# `<|image|>` marker exactly once per image (the processor validates the count)
# — the processor expands each marker into `boi + <|image|>×N + eoi` with N
# dynamic ≤ 280. Patterns are deterministic functions of pixel coordinates so
# the npz is reproducible bit-for-bit. Sizes exercise the processor's edges:
# floor-to-48 resize, non-square aspects, the target-side==0 clamp (8×8000),
# patch grids that are and are not multiples of the 3×3 pool, and marker
# positions (start / middle / end).
IMAGE_CASES = [
    ("img_square_256", "query", "what is in this image <|image|>", (1024, 1024, "grad")),
    ("img_wide_264tok", "document", "<|image|> a wide screenshot", (270, 480, "checker")),
    ("img_tiny", "document", "tiny <|image|>", (48, 48, "solid")),
    ("img_extreme_aspect", "document", "extremely <|image|> wide", (8, 8000, "grad")),
    ("img_nonmultiple_pool", "query", "non <|image|> multiple of the pool grid", (222, 384, "noise")),
    ("img_with_marker", "query", "<|image|> describe in detail", (336, 240, "checker")),
]


def _pattern(h, w, kind, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    if kind == "grad":
        base = ((x * 255) // max(w - 1, 1) + (y * 255) // max(h - 1, 1)) // 2
        return np.stack([base, (base * 3) // 4, 255 - base], -1).astype(np.uint8)
    if kind == "checker":
        block = max(min(h, w) // 8, 2)
        c = (((x // block) + (y // block)) % 2) * 200 + 40
        return np.stack([c, 255 - c, c], -1).astype(np.uint8)
    if kind == "solid":
        return np.full((h, w, 3), 137, np.uint8)
    if kind == "noise":
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    raise ValueError(kind)


def _image_bytes(h, w, kind):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_pattern(h, w, kind)).save(buf, format="PNG")
    return buf.getvalue()


def run_image_cases(model, processor, out):
    import io

    import torch
    from PIL import Image

    for key, kind, text, (h, w, pat) in IMAGE_CASES:
        prompt = Q if kind == "query" else D
        img = Image.open(io.BytesIO(_image_bytes(h, w, pat)))
        inputs = processor(text=[prompt + text], images=[img],
                           return_tensors="pt")
        n_soft = int((inputs["input_ids"][0] == 258880).sum())
        ids = inputs["input_ids"][0]
        out[f"ids_{key}"] = ids.numpy()
        out[f"nsoft_{key}"] = np.asarray(n_soft)
        hs = model(**inputs).last_hidden_state[0]
        pooled = hs.mean(dim=0)
        pooled = pooled / pooled.norm()
        out[f"emb_{key}"] = pooled.numpy()
        for m in MRL:
            t = pooled[:m]
            out[f"emb_{key}_{m}"] = (t / t.norm()).numpy()
        print(f"{key}: {len(ids)} tokens ({n_soft} soft)")


# Audio cases: (key, kind, text, duration_s). The text carries `<|audio|>`
# (id 258881) once per clip; M = the feature-mask subsample result (frontend
# pad_to_multiple_of=128, then two stride-2 subsamples), UNCAPPED — verified
# against the float32 reference: 0.1 s → 3, 0.5 s → 13, 1.0 s → 25, 2.0 s →
# 50, 11.2 s → 280, 15 s → 375. The checkpoint's audio_seq_length=280
# processor_config key is NOT enforced as a cap in the audio path.
AUDIO_CASES = [
    ("aud_short_3", "query", "what does this sound like <|audio|>", 0.1),
    ("aud_half_13", "document", "<|audio|> half a second of tone", 0.5),
    ("aud_1s_25", "query", "one second <|audio|> of sine", 1.0),
    ("aud_2s_50", "document", "two seconds of <|audio|> sine", 2.0),
    ("aud_cap_280", "query", "at the <|audio|> cap", 11.2),
    ("aud_15s_375", "document", "fifteen seconds of <|audio|> tone", 15.0),
]


def _sine(seconds):
    sr = 16000
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def run_audio_cases(model, processor, out):
    import torch

    for key, kind, text, seconds in AUDIO_CASES:
        prompt = Q if kind == "query" else D
        wav = _sine(seconds)
        inputs = processor(text=[prompt + text], audio=[wav],
                           sampling_rate=16000, return_tensors="pt")
        n_soft = int((inputs["input_ids"][0] == 258881).sum())
        ids = inputs["input_ids"][0]
        out[f"ids_{key}"] = ids.numpy()
        out[f"nsoft_{key}"] = np.asarray(n_soft)
        hs = model(**inputs).last_hidden_state[0]
        pooled = hs.mean(dim=0)
        pooled = pooled / pooled.norm()
        out[f"emb_{key}"] = pooled.numpy()
        for m in MRL:
            t = pooled[:m]
            out[f"emb_{key}_{m}"] = (t / t.norm()).numpy()
        print(f"{key}: {len(ids)} tokens ({n_soft} soft)")


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


def _video_bytes(h, w, kind, t):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_video_frame(h, w, kind, t)).save(buf, format="PNG")
    return buf.getvalue()


def run_video_cases(model, processor, out):
    import io

    import torch
    from PIL import Image

    for key, kind, text, (n, h, w, pat) in VIDEO_CASES:
        prompt = Q if kind == "query" else D
        frames = [Image.open(io.BytesIO(_video_bytes(h, w, pat, t)))
                  for t in range(n)]
        inputs = processor(text=[prompt + text], videos=[frames],
                           return_tensors="pt")
        n_soft = int((inputs["input_ids"][0] == 258884).sum())
        n_frames = int(inputs["pixel_values_videos"].shape[0])
        ids = inputs["input_ids"][0]
        out[f"ids_{key}"] = ids.numpy()
        out[f"nsoft_{key}"] = np.asarray(n_soft)
        out[f"nframes_{key}"] = np.asarray(n_frames)
        hs = model(**inputs).last_hidden_state[0]
        pooled = hs.mean(dim=0)
        pooled = pooled / pooled.norm()
        out[f"emb_{key}"] = pooled.numpy()
        t512 = pooled[:512]
        out[f"emb_{key}_512"] = (t512 / t512.norm()).numpy()
        print(f"{key}: {len(ids)} tokens, {n_frames} frames ({n_soft} soft)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="dir with model.safetensors (HF snapshot or a glob of it)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--skip-images", action="store_true",
                    help="text cases only (the v0 text-only npz contract)")
    args = ap.parse_args()
    model_dir = args.model
    if os.path.isdir(model_dir) and not os.path.exists(
            os.path.join(model_dir, "model.safetensors")):
        hits = sorted(glob.glob(os.path.join(model_dir, "*", "model.safetensors")))
        if hits:
            model_dir = os.path.dirname(hits[0])

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    out = {}
    with torch.no_grad():
        for key, kind, text in CASES:
            prompt = Q if kind == "query" else D
            enc = tok(prompt + text, return_tensors="pt",
                      truncation=True, max_length=8192)
            ids = enc["input_ids"][0]
            out[f"ids_{key}"] = ids.numpy()
            hs = model(**enc).last_hidden_state[0]  # [n, 768], already projected
            pooled = hs.mean(dim=0)
            pooled = pooled / pooled.norm()
            out[f"emb_{key}"] = pooled.numpy()
            for m in MRL:
                t = pooled[:m]
                out[f"emb_{key}_{m}"] = (t / t.norm()).numpy()
            print(f"{key}: {len(ids)} tokens")

        if not args.skip_images:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_dir)
            run_image_cases(model, processor, out)
            run_audio_cases(model, processor, out)
            run_video_cases(model, processor, out)

    np.savez(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
