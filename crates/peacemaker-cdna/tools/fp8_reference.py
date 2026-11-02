#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Independent reference for `peacemaker_cdna::float`, written as
`tests/fixtures/fp8_reference.json`.

Exact rational arithmetic (`fractions.Fraction`) defines every value and every
rounding; numpy supplies the f32 bit patterns (and nextafter) and cross-checks
F16 RNE; `ml_dtypes`, when importable, cross-checks FP8 decode and in-range
RNE (it is never required). Install it only into a throwaway venv:

    python3 -m venv --system-site-packages /tmp/pmc-float-venv
    /tmp/pmc-float-venv/bin/pip install ml_dtypes
    /tmp/pmc-float-venv/bin/python crates/peacemaker-cdna/tools/fp8_reference.py

(run from the repository root; `python3 …` works too, without ml_dtypes).

Policies (shared with float.rs; CDNA3 ISA §7.2 p.50-51, the hardware-side
choices are [UNVERIFIED: confirm on MI300X] there):
- FP16_OVFL applies to the source magnitude before rounding: |x| > max →
  ovfl=1: ±max; ovfl=0: E4M3 NaN, E5M2 ±Inf (FNUZ: NaN 0x80). ±Inf likewise.
- NaN → FNUZ 0x80, OCP sign|0x7f.
- FNUZ zero results (incl. -0.0 and negative underflow) → 0x00; OCP keeps sign.
- Decoded NaN: FNUZ 0x7fc00000, OCP sign|0x7fc00000.

Fixture layout (all hex, lower case):
- "decode_<enc>_<kind>": 256 f32 bit patterns, index = code.
- "f8_rne": "XXXXXXXX:" + 8 codes, order fnuz_e4m3, fnuz_e5m2, ocp_e4m3,
  ocp_e5m2 with FP16_OVFL=0, then the same four with FP16_OVFL=1.
- "f16": "XXXXXXXX:RRRR:TTTT:BBBB" = f16 RNE, f16 RTZ, bf16 RNE.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from fractions import Fraction

import numpy as np

try:
    import ml_dtypes  # optional cross-check
except ImportError:  # pragma: no cover
    ml_dtypes = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "fixtures", "fp8_reference.json")

QNAN = 0x7FC00000


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bits_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits_frac(b: int) -> Fraction:
    """Exact value of a finite f32 bit pattern."""
    sign = -1 if b >> 31 else 1
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    assert e != 0xFF
    if e == 0:
        v = Fraction(m, 1 << 149)
    else:
        v = Fraction((1 << 23) | m) * Fraction(2) ** (e - 150)
    return sign * v


def frac_f32_bits(v: Fraction) -> int:
    """f32 bits of an exactly representable non-negative value (or zero)."""
    out = f32_bits(float(v))
    assert bits_frac(out) == v, v
    return out


class Fmt:
    """A small binary float format: grid of finite non-negative values."""

    def __init__(self, name, ebits, mbits, bias, max_code, enc):
        self.name, self.ebits, self.mbits, self.bias = name, ebits, mbits, bias
        self.max_code, self.enc = max_code, enc

    def mag_value(self, mag: int) -> Fraction:
        e, m = mag >> self.mbits, mag & ((1 << self.mbits) - 1)
        if e == 0:
            return Fraction(m) * Fraction(2) ** (1 - self.bias - self.mbits)
        return (1 + Fraction(m, 1 << self.mbits)) * Fraction(2) ** (e - self.bias)

    @property
    def max(self) -> Fraction:
        return self.mag_value(self.max_code)

    def grid(self):
        return [self.mag_value(c) for c in range(self.max_code + 1)]


F8_FMTS = [
    Fmt("fnuz_e4m3", 4, 3, 8, 0x7F, "fnuz"),
    Fmt("fnuz_e5m2", 5, 2, 16, 0x7F, "fnuz"),
    Fmt("ocp_e4m3", 4, 3, 7, 0x7E, "ocp"),
    Fmt("ocp_e5m2", 5, 2, 15, 0x7B, "ocp"),
]


def decode(f: Fmt, code: int) -> int:
    sign = code >> 7
    mag = code & 0x7F
    if f.enc == "fnuz":
        if code == 0x80:
            return QNAN
    elif mag > f.max_code:
        if f.ebits == 5 and mag == 0x7C:
            return (sign << 31) | 0x7F800000
        return (sign << 31) | QNAN
    return (sign << 31) | frac_f32_bits(f.mag_value(mag))


def rne_index(grid, v: Fraction) -> int:
    """Index of the nearest grid point to v >= 0, ties to the even index
    (index parity == mantissa LSB parity on a binary grid)."""
    lo, hi = 0, len(grid) - 1
    if v >= grid[hi]:
        return hi
    while hi - lo > 1:  # invariant grid[lo] <= v < grid[hi]
        mid = (lo + hi) // 2
        if grid[mid] <= v:
            lo = mid
        else:
            hi = mid
    dl, dh = v - grid[lo], grid[hi] - v
    if dl < dh:
        return lo
    if dh < dl:
        return hi
    return lo if lo % 2 == 0 else hi


def to_f8(f: Fmt, xb: int, ovfl: bool) -> int:
    sign = xb >> 31
    e = (xb >> 23) & 0xFF
    nan = 0x80 if f.enc == "fnuz" else (sign << 7) | 0x7F
    if e == 0xFF and (xb & 0x7FFFFF):
        return nan
    over = e == 0xFF or abs(bits_frac(xb)) > f.max
    if over:
        if ovfl:
            return (sign << 7) | f.max_code
        if f.ebits == 4 or f.enc == "fnuz":
            return nan
        return (sign << 7) | 0x7C
    idx = rne_index(f.grid(), abs(bits_frac(xb)))
    if idx == 0 and f.enc == "fnuz":
        return 0
    return (sign << 7) | idx


# ---- F16 / BF16 (exact, independent of numpy) -------------------------------

F16 = Fmt("f16", 5, 10, 15, 0x7BFF, "ieee")
BF16 = Fmt("bf16", 8, 7, 127, 0x7F7F, "ieee")


def ieee_round(f: Fmt, xb: int, rtz: bool) -> int:
    sign = xb >> 31
    e = (xb >> 23) & 0xFF
    m = xb & 0x7FFFFF
    sbit = sign << (f.ebits + f.mbits)
    inf = ((1 << f.ebits) - 1) << f.mbits
    if e == 0xFF:
        if m == 0:
            return sbit | inf
        # quiet NaN, top payload bits kept
        shift = 23 - f.mbits
        return sbit | inf | (1 << (f.mbits - 1)) | (m >> shift)
    v = abs(bits_frac(xb))
    # exact grid arithmetic: LSB of the binade (or subnormal step)
    emin = 1 - f.bias
    ex = emin if v == 0 else max(emin, math.floor(math.log2(v)))
    # correct possible log2 float error
    while ex > emin and Fraction(2) ** ex > v:
        ex -= 1
    while Fraction(2) ** (ex + 1) <= v:
        ex += 1
    ulp = Fraction(2) ** (ex - f.mbits)
    q = v / ulp
    n = math.floor(q)
    if not rtz:
        r = q - n
        if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
            n += 1
    val = n * ulp
    if val > f.max:
        return sbit | (f.max_code if rtz else inf)
    # encode val
    if val == 0:
        return sbit
    if val < Fraction(2) ** emin:
        return sbit | int(val / Fraction(2) ** (emin - f.mbits))
    ev = ex if val < Fraction(2) ** (ex + 1) else ex + 1
    mant = int((val / Fraction(2) ** ev - 1) * (1 << f.mbits))
    return sbit | ((ev + f.bias) << f.mbits) | mant


# ---- sweeps -----------------------------------------------------------------


def nextup(b: int) -> int:
    return f32_bits(float(np.nextafter(np.float32(bits_f32(b)), np.float32(np.inf))))


def nextdown(b: int) -> int:
    return f32_bits(float(np.nextafter(np.float32(bits_f32(b)), np.float32(-np.inf))))


def around(points, out):
    """Every point p (non-negative f32 bits), p±1 ulp."""
    for p in points:
        out.add(p)
        if p != 0:
            out.add(nextdown(p))
        if p < 0x7F800000:
            out.add(nextup(p))


def f8_sweep():
    pts = set()
    for f in F8_FMTS:
        g = f.grid()
        vals = list(g)
        vals += [(a + b) / 2 for a, b in zip(g, g[1:])]
        # boundaries: half the smallest subnormal, the subnormal/normal edge,
        # beyond max: max + half ulp (IEEE tie), max + ulp, 2*max
        step = g[1]
        top_ulp = g[-1] - g[-2]
        vals += [step / 2, step / 4, step * 3 / 4, Fraction(2) ** (1 - f.bias)]
        vals += [g[-1] + top_ulp / 2, g[-1] + top_ulp, 2 * g[-1], 4 * g[-1]]
        around([frac_f32_bits(v) for v in vals], pts)
    # f32 specials: 0, smallest/largest f32 subnormal, smallest normal, max, Inf
    around([0, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000], pts)
    pos = sorted(pts)
    xs = pos + [p | 0x80000000 for p in pos]
    xs += [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF]  # NaNs, both signs
    return xs


def f16_sweep():
    pts = set()
    for f in (F16, BF16):
        # binade edges of every exponent, subnormal steps, max, overflow ties
        emin = 1 - f.bias
        for ex in range(emin - f.mbits - 2, f.bias + 2):
            if not -149 <= ex <= 127:
                continue
            # every binade of f16; bf16: both ends plus every 8th binade
            edge = ex < emin + 3 or ex > f.bias - 3
            if f is BF16 and not edge and ex % 8:
                continue
            p = Fraction(2) ** ex
            ulp = Fraction(2) ** (max(ex, emin) - f.mbits)
            for v in (p, p + ulp / 2, p + ulp, p + 3 * ulp / 2, p - ulp / 4):
                if 0 < v < Fraction(2) ** 128 and v >= Fraction(1, 1 << 149):
                    try:
                        pts.add(frac_f32_bits(v))
                    except (AssertionError, OverflowError):
                        pass
        top_ulp = Fraction(2) ** (f.bias - f.mbits)
        for v in (f.max, f.max + top_ulp / 2, f.max + top_ulp / 4):
            if v < Fraction(2) ** 128:
                pts.add(frac_f32_bits(v))
    near = set()
    around(sorted(pts), near)
    pts |= near
    # f32 specials
    around([0, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000], pts)
    # deterministic pseudo-random f32 patterns (LCG), finite only
    s = 0x2545F491
    for _ in range(600):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        b = (s >> 32) & 0x7FFFFFFF
        if b < 0x7F800000:
            pts.add(b)
    pos = sorted(pts)
    xs = pos + [p | 0x80000000 for p in pos]
    xs += [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFBFFFFF, 0x7FFFFFFF]
    return xs


# ---- cross-checks -----------------------------------------------------------


def cross_check(xs8, xs16, rne8, f16rows):
    # numpy f16 RNE (IEEE) on non-NaN inputs
    for b, (r, _t, _bf) in zip(xs16, f16rows):
        if ((b >> 23) & 0xFF) == 0xFF and (b & 0x7FFFFF):
            continue
        n = int(np.array([bits_f32(b)], dtype=np.float32).astype(np.float16).view(np.uint16)[0])
        assert n == r, (hex(b), hex(n), hex(r))
    if ml_dtypes is None:
        print("ml_dtypes not importable: skipped FP8/BF16 cross-check", file=sys.stderr)
        return
    md = {
        "fnuz_e4m3": ml_dtypes.float8_e4m3fnuz,
        "fnuz_e5m2": ml_dtypes.float8_e5m2fnuz,
        "ocp_e4m3": ml_dtypes.float8_e4m3fn,
        "ocp_e5m2": ml_dtypes.float8_e5m2,
    }
    for i, f in enumerate(F8_FMTS):
        t = md[f.name]
        codes = np.arange(256, dtype=np.uint8).view(t).astype(np.float32)
        for c in range(256):
            ours = bits_f32(decode(f, c))
            if math.isnan(ours):
                assert math.isnan(float(codes[c])), (f.name, c)
            else:
                assert f32_bits(float(codes[c])) == decode(f, c), (f.name, c)
        # in-range RNE (policies differ only for NaN / |x| > max)
        for b, row in zip(xs8, rne8):
            v = bits_f32(b)
            if not math.isfinite(v) or abs(Fraction(v)) > f.max:
                continue
            got = int(np.array([v], dtype=np.float32).astype(t).view(np.uint8)[0])
            assert got == row[i], (f.name, hex(b), hex(got), hex(row[i]))
    for b, (_r, _t, bf) in zip(xs16, f16rows):
        if ((b >> 23) & 0xFF) == 0xFF and (b & 0x7FFFFF):
            continue
        got = int(np.array([bits_f32(b)], dtype=np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)[0])
        assert got == bf, (hex(b), hex(got), hex(bf))
    print("ml_dtypes cross-check passed", file=sys.stderr)


def main():
    out = {
        "generator": "crates/peacemaker-cdna/tools/fp8_reference.py",
        "regenerate": "python3 crates/peacemaker-cdna/tools/fp8_reference.py (from the repo root; see its docstring for the optional ml_dtypes venv)",
    }
    for f in F8_FMTS:
        out["decode_" + f.name] = ["%08x" % decode(f, c) for c in range(256)]
    xs8 = f8_sweep()
    rne8 = []
    lines = []
    for b in xs8:
        row = [to_f8(f, b, ovfl) for ovfl in (False, True) for f in F8_FMTS]
        rne8.append(row)
        lines.append("%08x:" % b + "".join("%02x" % c for c in row))
    out["f8_rne"] = lines
    xs16 = f16_sweep()
    rows = []
    lines = []
    for b in xs16:
        r = (ieee_round(F16, b, False), ieee_round(F16, b, True), ieee_round(BF16, b, False))
        rows.append(r)
        lines.append("%08x:%04x:%04x:%04x" % (b, *r))
    out["f16"] = lines
    cross_check(xs8, xs16, rne8, rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("{\n")
        items = list(out.items())
        for k, (key, val) in enumerate(items):
            sep = "," if k + 1 < len(items) else ""
            if isinstance(val, list):
                fh.write(" %s: [\n" % json.dumps(key))
                for j in range(0, len(val), 8):
                    chunk = ", ".join(json.dumps(s) for s in val[j : j + 8])
                    tail = "," if j + 8 < len(val) else ""
                    fh.write("  %s%s\n" % (chunk, tail))
                fh.write(" ]%s\n" % sep)
            else:
                fh.write(" %s: %s%s\n" % (json.dumps(key), json.dumps(val), sep))
        fh.write("}\n")
    print("wrote %s: %d f8 rows, %d f16 rows" % (os.path.normpath(OUT), len(xs8), len(xs16)), file=sys.stderr)


if __name__ == "__main__":
    main()
