#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Golden fixture for `rdna_compute::fp8_e4m3` (FP8 E4M3B128 / E4M3PT decode).

Writes `golden.bin` next to this script: a sequence of little-endian records

    u16 scale_bits                      bf16 bits of the (positive) scale s_inv
    u16 bf16[254]                       teacher BF16 bits for the 254 finite e4m3fn
                                        codes in ascending byte order
                                        (0x7F and 0xFF, the NaN codes, are skipped)

Reference values come from `subq4/calib2/fp8_teacher.py:dequant_e4m3_block` (the
spec of record; imported by path, never reimplemented), run as a single-row
`[1, 254]` block problem with one scale per 128-wide block (both blocks get the
same scale). A scale is kept only if the teacher accepts the whole sweep (finite,
F32-exact product, no bf16 overflow).

Usage:  python3 gen_golden.py [--teacher PATH]
Deterministic: fixed seed, no clocks. Prints the sha256 of the produced file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

DEFAULT_TEACHER = "/home/kaden/qcal/release-0.4.2/subq4/calib2/fp8_teacher.py"
SEED = 0x6F5E4D3C
RANDOM_SCALES = 32


def load_teacher(path: str):
    spec = importlib.util.spec_from_file_location("fp8_teacher", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fp8_teacher"] = mod
    spec.loader.exec_module(mod)
    return mod


def bf16_bits(x: float) -> int:
    """Round a python float to bf16 bits with the F32 -> bf16 RNE formula (finite only)."""
    u = int(np.float32(x).view(np.uint32))
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def scale_list(rng: np.random.Generator) -> list[int]:
    fixed = [
        0.0,            # zero scale: all outputs are signed zeros
        1.0,
        2.0 ** -20,
        2.0 ** -9,
        0.0017,
        2.0 ** -100,    # smallest scale whose products stay F32-normal-exact with 2^-9 codes
        2.0 ** -30,
        2.0 ** -3,
        0.0123,
        0.1,
        0.75,
        3.0,
        448.0 ** -1,    # 1/448: the usual amax->e4m3 inverse
        2.0 ** 10,
        1234.5,
        2.0 ** 20,
        2.0 ** 100,     # largest scale: 448 * 2^100 stays finite, and so does 2 * s
    ]
    bits = [bf16_bits(v) for v in fixed]
    # 32 pseudo-random positive bf16 scales: biased exponent 27..227 (2^-100 .. 2^100), random mantissa.
    while len(bits) < len(fixed) + RANDOM_SCALES:
        e = int(rng.integers(27, 228))
        m = int(rng.integers(0, 128))
        b = (e << 7) | m
        if b not in bits:
            bits.append(b)
    return bits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    args = ap.parse_args()
    t = load_teacher(args.teacher)

    codes = np.array([b for b in range(256) if not np.isnan(t.E4M3_F32[b])], dtype=np.uint8)
    assert len(codes) == 254 and 0x7F not in codes and 0xFF not in codes
    q = codes.reshape(1, 254)

    out = bytearray()
    kept = 0
    rng = np.random.default_rng(SEED)
    for s in scale_list(rng):
        s_inv = np.array([[s, s]], dtype=np.uint16)  # 254 columns span two 128-wide blocks
        try:
            got = t.dequant_e4m3_block(q, s_inv)
        except t.Fp8TeacherError as e:
            raise SystemExit(f"teacher refused scale {s:#06x}: {e}")
        # The per-element teacher formula, recomputed independently from E4M3_F32 / f32_to_bf16_rne.
        s32 = t.bf16_bits_to_f32(np.array([s], dtype=np.uint16))[0]
        ref = t.f32_to_bf16_rne(t.E4M3_F32[codes] * s32)
        assert np.array_equal(got[0], ref), f"scale {s:#06x}: block vs elementwise teacher disagree"
        out += struct.pack("<H", s) + got[0].astype("<u2").tobytes()
        kept += 1

    dest = Path(__file__).with_name("golden.bin")
    dest.write_bytes(bytes(out))
    print(f"wrote {dest} records={kept} bytes={len(out)} sha256={hashlib.sha256(out).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
