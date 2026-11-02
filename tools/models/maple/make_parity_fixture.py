#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
"""Generate the Maple qt=51 cross-implementation parity fixture.

Reads a REAL tensor out of a Maple-Preview safetensors shard and writes:

  <out>.f32  — the source weights as f32 LE   (MAPLE_FIXTURE_IN)
  <out>.bin  — the expected MQ2G256LloydU bytes (MAPLE_FIXTURE_EXPECTED)

Then:

  MAPLE_FIXTURE_IN=<out>.f32 MAPLE_FIXTURE_EXPECTED=<out>.bin \\
    cargo test -p hipfire-quantize maple_fixture_parity -- --nocapture

WHY THIS EXISTS. Every other test of the packer round-trips through OUR OWN
dequantizer, so a shared misunderstanding of the 72 B/group layout would
round-trip perfectly and still feed the kernel garbage. The packer below is
written from the LAYOUT SPEC, independently of the Rust implementation, so a
byte-for-byte match is real evidence. It has already earned its keep once: it
caught the -0.0/+0.0 non-determinism that made the Rust packer emit two
different byte strings for the same tensor.

The fixtures are large and uncommitted; THIS GENERATOR is committed so they can
always be rebuilt. A previous run's fixtures were lost and the env-gated test
then reported "ok" while doing nothing.

Layout being asserted (72 bytes per group of 256 weights):
  [0..8)   four fp16 codebook entries, ASCENDING, slot 3 duplicates slot 2
  [8..72)  64 bytes of 2-bit indices, 4 per byte, LSB-first
"""
import argparse
import json
import struct
import sys

import numpy as np

GROUP = 256
BLOCK_BYTES = 72


def read_tensor(shard_path, name):
    """Minimal safetensors reader — header length, header JSON, then the slice."""
    with open(shard_path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        if name not in header:
            raise SystemExit(
                f"{name!r} not in {shard_path}\nfirst few: "
                + ", ".join(list(header)[:5])
            )
        meta = header[name]
        if meta["dtype"] != "BF16":
            raise SystemExit(f"{name}: expected BF16, got {meta['dtype']}")
        start, end = meta["data_offsets"]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    bf16 = np.frombuffer(raw, dtype=np.uint16)
    # BF16 -> f32 is a 16-bit left shift; exact, no rounding.
    f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(meta["shape"])


def pack_group(vals):
    """Pack one 256-weight group into 72 bytes, from the layout spec."""
    # Distinct levels. +0.0 and -0.0 must collapse to ONE level and emit +0.0:
    # np.unique alone keeps them distinct, which is the exact trap that made the
    # first Rust packer non-deterministic.
    cleaned = np.where(vals == 0.0, 0.0, vals)
    levels = sorted(set(float(v) for v in cleaned))
    if len(levels) > 3:
        raise SystemExit(f"block is not ternary: {len(levels)} distinct values")

    # Pad to 4 entries by duplicating the TOP level; slot 3 is never indexed.
    cb = [levels[min(i, len(levels) - 1)] for i in range(4)]
    cb16 = np.array(cb, dtype=np.float16)
    # The GPU decodes fp16; refuse anything that is not exactly representable.
    for i, lv in enumerate(levels):
        if float(cb16[i]) != lv:
            raise SystemExit(f"level {lv} is not exactly representable in fp16")

    out = bytearray(BLOCK_BYTES)
    out[0:8] = cb16.tobytes()

    index_of = {lv: i for i, lv in enumerate(levels)}
    for i in range(64):
        byte = 0
        for j in range(4):
            v = float(cleaned[4 * i + j])
            byte |= (index_of[v] & 0x3) << (2 * j)
        out[8 + i] = byte
    return bytes(out)


def pack(vals):
    if vals.size % GROUP != 0:
        raise SystemExit(f"length {vals.size} is not a multiple of {GROUP}")
    return b"".join(
        pack_group(vals[i : i + GROUP]) for i in range(0, vals.size, GROUP)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="model-0000N-of-00009.safetensors")
    ap.add_argument(
        "--tensor",
        default="model.layers.0.mlp.experts.0.gate_proj.weight",
        help="tensor name (default: an expert gate_proj)",
    )
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument(
        "--rows",
        type=int,
        default=0,
        help="limit to the first N rows (0 = all); keeps the fixture small",
    )
    args = ap.parse_args()

    t = read_tensor(args.shard, args.tensor)
    if args.rows:
        t = t[: args.rows]
    vals = np.ascontiguousarray(t, dtype=np.float32).ravel()
    k = t.shape[-1]
    if k % GROUP != 0:
        raise SystemExit(f"K={k} must be a multiple of {GROUP}")

    packed = pack(vals)

    with open(args.out + ".f32", "wb") as f:
        f.write(vals.tobytes())
    with open(args.out + ".bin", "wb") as f:
        f.write(packed)

    nz = int(np.count_nonzero(vals))
    print(f"tensor  : {args.tensor}  shape={list(t.shape)}  K={k}")
    print(f"weights : {vals.size}  nonzero={nz} ({nz / vals.size:.1%})")
    print(f"distinct: {sorted(set(float(v) for v in np.where(vals == 0.0, 0.0, vals)))[:6]}")
    print(f"packed  : {len(packed)} bytes ({len(packed) * 8 / vals.size:.4f} bpw)")
    print(f"wrote   : {args.out}.f32  {args.out}.bin")
    print()
    print("Now run:")
    print(
        f"  MAPLE_FIXTURE_IN={args.out}.f32 MAPLE_FIXTURE_EXPECTED={args.out}.bin \\"
    )
    print("    cargo test -p hipfire-quantize maple_fixture_parity -- --nocapture")


if __name__ == "__main__":
    sys.exit(main())
