#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
"""Per-layer cosine between hipfire's Maple residuals and a reference dump.

hipfire side (the probe is `lab`-gated, so build it first):
  cargo build --release -p hipfire-arch-maple --example maple_coherence --features lab
  HIPFIRE_MAPLE_DUMP_HIDDEN=ours.bin \\
    target/release/examples/maple_coherence --model maple.hfq --max-tokens 1

Reference side (HF, CPU, trust_remote_code) — dump `hidden_states` after each
decoder layer in the same record format:
  [u32 layer][u32 hidden][hidden f32 LE]  repeated, appended in order.

Then:
  maple_compare_hidden.py --ours ours.bin --ref ref.bin [--pos -1]

A cosine cliff at layer n localises the bug to layer n's block. On this arch the
two silent-wrong-answer risks to check first are RoPE-applied-on-a-NoPE-layer
and QK-norm ordering; both leave every earlier layer near 1.0.
"""
import argparse
import struct
import sys

import numpy as np


def read_dump(path):
    """-> dict[layer] -> list of vectors, one per position, in file order."""
    out = {}
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    n = len(blob)
    while off < n:
        if off + 8 > n:
            raise SystemExit(f"{path}: truncated record header at byte {off}")
        layer, hidden = struct.unpack_from("<II", blob, off)
        off += 8
        need = hidden * 4
        if off + need > n:
            raise SystemExit(
                f"{path}: truncated payload for layer {layer} at byte {off} "
                f"(need {need}, have {n - off})"
            )
        vec = np.frombuffer(blob, dtype=np.float32, count=hidden, offset=off)
        off += need
        out.setdefault(layer, []).append(vec)
    return out


def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument(
        "--pos",
        type=int,
        default=-1,
        help="which recorded position to compare (default: last)",
    )
    ap.add_argument("--threshold", type=float, default=0.9999)
    args = ap.parse_args()

    ours = read_dump(args.ours)
    ref = read_dump(args.ref)

    common = sorted(set(ours) & set(ref))
    if not common:
        raise SystemExit(
            f"no layers in common (ours: {sorted(ours)[:5]}..., ref: {sorted(ref)[:5]}...)"
        )
    if set(ours) != set(ref):
        print(
            f"WARNING: layer sets differ — ours {len(ours)}, ref {len(ref)}; "
            f"comparing the {len(common)} in common",
            file=sys.stderr,
        )

    print(f"{'layer':>5}  {'cosine':>10}  {'|ours|':>10}  {'|ref|':>10}")
    first_bad = None
    for l in common:
        a = ours[l][args.pos]
        b = ref[l][args.pos]
        if a.shape != b.shape:
            raise SystemExit(f"layer {l}: hidden {a.shape} vs {b.shape}")
        c = cosine(a, b)
        flag = "" if c >= args.threshold else "   <-- CLIFF"
        if c < args.threshold and first_bad is None:
            first_bad = l
        print(
            f"{l:>5}  {c:>10.6f}  {np.linalg.norm(a):>10.3f}  "
            f"{np.linalg.norm(b):>10.3f}{flag}"
        )

    print()
    if first_bad is None:
        print(f"PASS: every layer >= {args.threshold}")
        return 0
    print(
        f"FAIL: first layer below {args.threshold} is {first_bad}. "
        f"Layers < {first_bad} agree, so the defect is in layer {first_bad}'s block."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
