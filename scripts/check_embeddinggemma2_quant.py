#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""EmbeddingGemma 2 (arch 18) quantization matrix — the arch-17
`check_embed_quant.py` shape.

Loads every artifact the CLI can write (dir, .q8, .q8f16, .mq6, .hf6, .mq4,
.hf4), encodes the parity fixtures through the daemon `embed` verb, and
asserts each artifact's embeddings sit inside its bit-width band against the
HF f32 reference (q8 3e-2, 6-bit 8e-2, 4-bit 2.5e-1 — an order under the O(1)
a broken dequant arm produces. The band is chosen by the artifact's own
dominant quant type (via the `astrea` HFQ parser), never its filename.

An artifact whose format is not in the dense HFQ decode set is skipped with a
reason, and a run where *every* artifact was skipped exits non-zero: a gate
that encoded nothing must not report a pass.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from check_embeddinggemma2_parity import (  # noqa: E402
    CASES,
    IMAGE_CASES,
    Daemon,
    _png_bytes,
    rel_l2,
)


def artifact_format(path):
    """The artifact's real quant format from its HFQ tensor index (the
    arch-17 `check_embed_quant.py` derivation), never its filename."""
    if os.path.isdir(path):
        return "dir"
    try:
        import astrea
        summary, _ = astrea.read_hfq_index(path, max_tensors=0)
        counts = summary.get("quant_type_counts", {})
        compressed = {k: v for k, v in counts.items() if k not in ("F16", "F32")}
        dom = compressed or counts
        return max(dom, key=dom.get) if dom else None
    except Exception:
        return None

# Bands calibrated on THIS fixture (google/embeddinggemma-2, gfx1101, f32
# HF reference via ref.npz): measured worst rel-L2 across the fixtures was
# dir 1.0e-3, q8 2.0e-2, 6-bit 1.0e-1, 4-bit 3.4e-1 — the arch-18 trunk is a
# narrow (512-d) 24-layer encoder whose single-vector mean pool does not
# cancel systematic quant bias, so 6/4-bit run nearer their bit-width ceiling
# than the arch-17 multi-vector encoder's bands (8e-2 / 2.5e-1). Each band
# keeps ~50% headroom over the measured ladder and stays an order below the
# O(1) a broken dequant arm produces.
BANDS = {
    "MQ4G256": 5.0e-1, "MQ4G256V2": 5.0e-1, "HFQ4G256": 5.0e-1, "HFQ4G128": 5.0e-1,
    "MQ6G256": 1.5e-1, "MQ6G256V2": 1.5e-1, "HFQ6G256": 1.5e-1,
    "Q8_0": 3.0e-2, "Q8F16": 3.0e-2, "MQ8G256": 3.0e-2,
    "dir": 2.0e-3,
}

# Formats that are not in the dense HFQ decode set at all: an artifact built
# with one cannot be loaded, so it has no band. `hipfire-quantize --format q4k`
# will still emit them for this arch (`quantize_q4k` exists for the arches whose
# kernels decode Q4K), and the arch-18 loader refuses such a checkpoint by name
# — `weights::refuse_unsupported_quant_types` — instead of reaching the shared
# host decoder's `panic!` for the missing arm.
NON_LOADABLE = {
    "Q4_K": "GGUF/Maple-side k-quant, not in the dense HFQ decode set",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("artifacts", nargs="+")
    args = ap.parse_args()

    ref = np.load(args.ref)
    fixtures = [(k, kind, text) for k, kind, text in CASES if k in
                ("short_query", "short_doc", "code_doc", "unicode", "medium_doc")]
    vision_ref = f"ids_{IMAGE_CASES[1][0]}" in ref
    # One image fixture per artifact: the towers ride F16 in the sidecar for
    # every tier, so the trunk's own band governs the image legs too — a
    # separate (looser) image band would only hide a splice bug.
    img_key, img_kind, img_text, img_dims = IMAGE_CASES[1]
    img_b64 = _png_bytes(*img_dims)

    def sibling_vl(path):
        import glob as globmod

        if os.path.isdir(path):
            hits = globmod.glob(os.path.join(path, "*.vl"))
            return hits[0] if hits else None
        stem = os.path.basename(path).rsplit(".", 1)[0]
        cand = os.path.join(os.path.dirname(path) or ".", f"{stem}.vl")
        return cand if os.path.exists(cand) else None

    failures = 0
    measured = 0
    for path in args.artifacts:
        fmt = artifact_format(path) if not os.path.isdir(path) else "dir"
        band = BANDS.get(fmt)
        name = os.path.basename(path)
        if fmt in NON_LOADABLE:
            print(f"SKIP {name}: [{fmt}] is not loadable — {NON_LOADABLE[fmt]}")
            continue
        if band is None:
            print(f"SKIP {name}: unknown format {fmt}")
            continue
        measured += 1
        vl = sibling_vl(path)
        d = Daemon(args.daemon, path,
                   params={"vision_mode": "auto"} if vl else None)
        worst, worst_key = 0.0, ""
        for key, kind, text in fixtures:
            r = d.embed(key, kind, {"text": text})
            e = rel_l2(r["embeddings"], ref[f"emb_{key}"])
            if e > worst:
                worst, worst_key = e, key
            if r["n_input_tokens"] != len(ref[f"ids_{key}"]):
                print(f"FAIL {name} {key}: token count {r['n_input_tokens']} != ref")
                failures += 1
        img_note = ""
        if vision_ref and vl:
            r = d.embed("img", img_kind, {"text": img_text, "images_base64": [img_b64]})
            e = rel_l2(r["embeddings"], ref[f"emb_{img_key}"])
            if e > worst:
                worst, worst_key = e, img_key
            if r["n_input_tokens"] != len(ref[f"ids_{img_key}"]):
                print(f"FAIL {name} {img_key}: token count {r['n_input_tokens']} != ref")
                failures += 1
            img_note = " +img"
        elif vision_ref and not vl:
            print(f"SKIP img legs for {name}: no sibling .vl sidecar")
        d.stop()
        ok = worst <= band
        print(f"{'ok  ' if ok else 'FAIL'} {name:14s} [{fmt:10s}] worst rel-L2 {worst:.2e} on {worst_key}{img_note} (band {band:.1e})")
        failures += 0 if ok else 1

    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    if not measured:
        # Every artifact was skipped: a gate that ran no leg must not report a
        # pass (a mistyped filename or a batch built only from NON_LOADABLE
        # formats otherwise "passes" without encoding anything).
        print(
            f"\nNOTHING MEASURED: all {len(args.artifacts)} artifact(s) were skipped "
            "(unknown format, or every one a non-loadable tier) — no leg ran"
        )
        return 1
    print(f"\nALL PASS ({measured} artifact(s) measured, {len(args.artifacts) - measured} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
