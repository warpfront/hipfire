#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma 2 (arch 18) multi-slot route check — the arch-17
`check_embed_multislot.py` shape.

Encodes the same inputs through the ordinary daemon (sequential) and through
`experimental_multi_slot` (the arch-18 slot engine's batched forward) and
requires them **bit-identical** — the batched forward is deterministic
(nothing carries state across segments), so any difference is a segment
cross-read or a batch-layout bug, exactly the failure class this route can
have. Also diffs the slot route against the HF f32 reference (rel-L2 band
2e-2) so the batch path is anchored to ground truth, not just self-agreement.
"""
import queue
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_embeddinggemma2_parity import (  # noqa: E402
    CASES,
    AUDIO_CASES,
    IMAGE_CASES,
    VIDEO_CASES,
    Daemon,
    _png_bytes,
    _png_frame_b64,
    _wav_b64,
    ensure_vision_sidecar,
    rel_l2,
)


def embed_concurrent(d, tag, kind, text, out):
    out[tag] = d.embed(tag, kind, {"text": text})


def embed_concurrent_image(d, tag, kind, text, img_b64, out):
    out[tag] = d.embed(tag, kind, {"text": text, "images_base64": [img_b64]})


def embed_concurrent_audio(d, tag, kind, text, wav_b64, out):
    out[tag] = d.embed(tag, kind, {"text": text, "audios_base64": [wav_b64]})


def embed_concurrent_video(d, tag, kind, text, frames_b64, out):
    out[tag] = d.embed(tag, kind, {"text": text, "videos_base64": [frames_b64]})


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--quantizer", default="target/release/hipfire-quantize")
    args = ap.parse_args()

    ref = np.load(args.ref)
    bodies = [(k, kind, text) for k, kind, text in CASES if k != "long_doc"]

    vision_on = (not args.no_vision) and f"ids_{IMAGE_CASES[0][0]}" in ref
    model_dir = ensure_vision_sidecar(args.model, args.quantizer) if vision_on else args.model

    seq = Daemon(args.daemon, model_dir,
                 params={"vision_mode": "auto" if vision_on else "off"})
    print(f"sequential daemon: arch {seq.loaded.get('arch')}")
    img_key, img_kind, img_text, img_dims = IMAGE_CASES[0]
    img_b64 = _png_bytes(*img_dims) if vision_on else None
    aud_key, aud_kind, aud_text, aud_secs = AUDIO_CASES[0]
    aud_b64 = _wav_b64(aud_secs) if vision_on else None
    vid_key, vid_kind, vid_text, (vid_n, vid_h, vid_w, vid_pat) = VIDEO_CASES[0]
    vid_frames = (
        [_png_frame_b64(vid_h, vid_w, vid_pat, t) for t in range(vid_n)]
        if vision_on and f"ids_{vid_key}" in ref else None
    )
    seq_out = {}
    for i, (key, kind, text) in enumerate(bodies):
        seq_out[key] = seq.embed(f"seq{i}", kind, {"text": text})
    if vision_on:
        seq_out["slot_img"] = seq.embed("seq_img", img_kind, {"text": img_text, "images_base64": [img_b64]})
        if f"ids_{aud_key}" in ref:
            seq_out["slot_aud"] = seq.embed("seq_aud", aud_kind, {"text": aud_text, "audios_base64": [aud_b64]})
        if vid_frames is not None:
            seq_out["slot_vid"] = seq.embed(
                "seq_vid", vid_kind, {"text": vid_text, "videos_base64": [vid_frames]}
            )
    seq.stop()

    slot_params = {
        "experimental_multi_slot": True,
        "experimental_multi_slot_slots": args.slots,
        "max_batch_tokens": 8192,
        "max_seq": 8192,
    }
    slot_params["vision_mode"] = "auto" if vision_on else "off"
    slot = Daemon(args.daemon, model_dir, params=slot_params)
    if not slot.loaded.get("experimental_multi_slot"):
        print("FAIL: daemon did not enter experimental_multi_slot")
        return 1
    print(f"slot daemon: arch {slot.loaded.get('arch')} (experimental_multi_slot, {args.slots} slots)")

    failures = 0
    # All requests in flight at once — the collector must merge them into one
    # row block (mixed lengths force per-segment attention and per-segment
    # position restarts inside one forward).
    threads, slot_out, errs = [], {}, {}
    for i, (key, kind, text) in enumerate(bodies):
        t = threading.Thread(
            target=embed_concurrent, args=(slot, f"slot{i}", kind, text, slot_out), daemon=True
        )
        t.start()
        threads.append(t)
    if vision_on:
        # One image request in flight WITH the text batch: mixed media in one
        # batched forward is exactly the segment cross-read class this route
        # can have.
        t = threading.Thread(
            target=embed_concurrent_image, args=(slot, "slot_img", img_kind, img_text, img_b64, slot_out),
            daemon=True,
        )
        t.start()
        threads.append(t)
        if f"ids_{aud_key}" in ref:
            t = threading.Thread(
                target=embed_concurrent_audio, args=(slot, "slot_aud", aud_kind, aud_text, aud_b64, slot_out),
                daemon=True,
            )
            t.start()
            threads.append(t)
        if vid_frames is not None:
            # The video leg rides the same batched forward: a multi-frame
            # segment beside text/image/audio is where a per-frame row block
            # mis-slotted over the flat frame axis would show up.
            t = threading.Thread(
                target=embed_concurrent_video,
                args=(slot, "slot_vid", vid_kind, vid_text, vid_frames, slot_out),
                daemon=True,
            )
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    for i, (key, kind, text) in enumerate(bodies):
        try:
            r = slot_out[f"slot{i}"]
        except RuntimeError as e:
            print(f"FAIL {key}: slot route error: {e}")
            failures += 1
            continue
        s = seq_out[key]
        same = np.array_equal(
            np.array(r["embeddings"], dtype=np.float32),
            np.array(s["embeddings"], dtype=np.float32),
        )
        e = rel_l2(r["embeddings"], ref[f"emb_{key}"])
        ok = same and e <= 2e-2 and r["n_input_tokens"] == s["n_input_tokens"]
        print(
            f"{'ok  ' if ok else 'FAIL'} {key}: bit-identical={same} slot-vs-ref rel-L2 {e:.2e}"
        )
        failures += 0 if ok else 1
    if vision_on:
        try:
            r = slot_out["slot_img"]
        except RuntimeError as e:
            print(f"FAIL {img_key}: slot route error: {e}")
            failures += 1
        else:
            s = seq_out["slot_img"]
            same = np.array_equal(
                np.array(r["embeddings"], dtype=np.float32),
                np.array(s["embeddings"], dtype=np.float32),
            )
            e = rel_l2(r["embeddings"], ref[f"emb_{img_key}"])
            n_ok = r["n_input_tokens"] == s["n_input_tokens"] == len(ref[f"ids_{img_key}"])
            ok = same and e <= 2e-2 and n_ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {img_key} (in-batch): bit-identical={same} "
                f"n={r['n_input_tokens']} slot-vs-ref rel-L2 {e:.2e}"
            )
            failures += 0 if ok else 1
    if vision_on and f"ids_{aud_key}" in ref:
        try:
            r = slot_out["slot_aud"]
        except RuntimeError as e:
            print(f"FAIL {aud_key}: slot route error: {e}")
            failures += 1
        else:
            s = seq_out["slot_aud"]
            same = np.array_equal(
                np.array(r["embeddings"], dtype=np.float32),
                np.array(s["embeddings"], dtype=np.float32),
            )
            e = rel_l2(r["embeddings"], ref[f"emb_{aud_key}"])
            n_ok = r["n_input_tokens"] == s["n_input_tokens"] == len(ref[f"ids_{aud_key}"])
            ok = same and e <= 2e-2 and n_ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {aud_key} (in-batch): bit-identical={same} "
                f"n={r['n_input_tokens']} slot-vs-ref rel-L2 {e:.2e}"
            )
            failures += 0 if ok else 1
    if vid_frames is not None:
        try:
            r = slot_out["slot_vid"]
        except RuntimeError as e:
            print(f"FAIL {vid_key}: slot route error: {e}")
            failures += 1
        else:
            s = seq_out["slot_vid"]
            same = np.array_equal(
                np.array(r["embeddings"], dtype=np.float32),
                np.array(s["embeddings"], dtype=np.float32),
            )
            e = rel_l2(r["embeddings"], ref[f"emb_{vid_key}"])
            n_ok = r["n_input_tokens"] == s["n_input_tokens"] == len(ref[f"ids_{vid_key}"])
            ok = same and e <= 2e-2 and n_ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {vid_key} (in-batch): bit-identical={same} "
                f"n={r['n_input_tokens']} slot-vs-ref rel-L2 {e:.2e}"
            )
            failures += 0 if ok else 1
    slot.stop()

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
