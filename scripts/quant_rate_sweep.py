#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
"""Per-tensor rate-distortion curves D_i(b) -- the input an allocator consumes.

Dequantizes an existing artifact to serve as pseudo-source, then re-quantizes
each sampled group at candidate rates (uniform min/max and Lloyd-Max) and
measures relative MSE. Reports the DISTRIBUTION across tensors, because a
graded/non-uniform allocation only pays if D_i(b) has spread across i.

Pseudo-source caveat: the reference is itself quantized, so absolute SNR is
optimistic and Lloyd levels can snap toward the source grid. Relative spread
and per-rate slope -- what the allocator uses -- are the intended outputs.
"""
import argparse, sys, re
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from quant_distortion_bench import parse_hfq, dequant_groups, SUPPORTED, G


def quant_uniform(w, bits):
    """w: (N, G) -> reconstructed, min/max affine grid (matches the encoder)."""
    lo = w.min(axis=1, keepdims=True)
    hi = w.max(axis=1, keepdims=True)
    n = (1 << bits) - 1
    sc = (hi - lo) / n
    sc = np.where(sc > 0, sc, 1.0)
    q = np.rint((w - lo) / sc)
    np.clip(q, 0, n, out=q)
    return q * sc + lo


def quant_lloyd(w, bits, iters=20):
    """Per-row Lloyd-Max, k = 2**bits levels, quantile init."""
    k = 1 << bits
    qs = (np.arange(k) + 0.5) / k
    lv = np.quantile(w, qs, axis=1).T.copy()            # (N, k)
    for _ in range(iters):
        d = np.abs(w[:, :, None] - lv[:, None, :])      # (N, G, k)
        a = d.argmin(axis=2)                            # (N, G)
        for j in range(k):
            m = (a == j)
            cnt = m.sum(axis=1)
            s = (w * m).sum(axis=1)
            lv[:, j] = np.where(cnt > 0, s / np.maximum(cnt, 1), lv[:, j])
        lv.sort(axis=1)
    d = np.abs(w[:, :, None] - lv[:, None, :])
    a = d.argmin(axis=2)
    return np.take_along_axis(lv, a, axis=1)


# E8 lattice quantizer (Conway-Sloane), matching crates/hipfire-quantize/src/e8.rs.
# E8 = D8 u (D8 + 1/2), D8 = {x in Z^8 : sum(x) even}. Per-32 block normalized by
# max/6; coords live in the biased box [-bias, 2^bits-1-bias].
E8_STEP = {4: 0.88, 3: 1.8, 2: 3.8}
E8_BIAS = {4: 7, 3: 4, 2: 2}


def _nearest_d8(u):
    a = np.rint(u)
    d = u - a
    rows = np.arange(u.shape[0])
    j = np.abs(d).argmax(axis=1)
    a_fix = a.copy()
    a_fix[rows, j] = a[rows, j] + np.where(d[rows, j] >= 0, 1.0, -1.0)
    odd = (a.sum(axis=1).astype(np.int64) % 2 != 0)
    return np.where(odd[:, None], a_fix, a)


def quant_e8(w, bits):
    """w: (N, 256) -> E8-quantized reconstruction at `bits` per coordinate."""
    step, bias = E8_STEP[bits], E8_BIAS[bits]
    lo, hi = -float(bias), float((1 << bits) - 1 - bias)
    N = w.shape[0]
    blk = w.reshape(N, 8, 32)
    mx = np.abs(blk).max(axis=2, keepdims=True)
    bs = np.where(mx > 0, mx / 6.0, 1.0)
    u = (blk / bs / step).reshape(-1, 8)
    A = np.clip(_nearest_d8(u), lo, hi)
    B = np.clip(_nearest_d8(u - 0.5) + 0.5, lo + 0.5, hi + 0.5)
    dA = ((u - A) ** 2).sum(axis=1)
    dB = ((u - B) ** 2).sum(axis=1)
    p = np.where((dA <= dB)[:, None], A, B)
    return (p.reshape(N, 8, 32) * step * bs).reshape(N, G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--groups", type=int, default=16, help="groups sampled per tensor")
    ap.add_argument("--rates", default="2,3", help="candidate bit widths")
    ap.add_argument("--filter", default=r"experts\.\d+\.w[123]\.weight",
                    help="regex selecting tensors to sweep")
    ap.add_argument("--group-by", default=r"experts\.(\d+)\.",
                    help="regex whose group(1) aggregates tensors into units")
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rates = [int(x) for x in a.rates.split(",")]
    idx = parse_hfq(a.artifact)
    mm = np.memmap(a.artifact, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(a.seed)
    sel = re.compile(a.filter)
    grp = re.compile(a.group_by)

    units = {}
    for name, t in idx.items():
        if t["qt"] not in SUPPORTED or not sel.search(name):
            continue
        if len(t["shape"]) < 2 or t["shape"][1] % G:
            continue
        m = grp.search(name)
        key = m.group(0) if m else name
        units.setdefault(key, []).append((name, t))
    keys = sorted(units)
    if a.max_units:
        keys = keys[:a.max_units]
    print(f"{a.artifact.split('/')[-1]}: {len(keys)} units, "
          f"{sum(len(units[k]) for k in keys)} tensors, "
          f"{a.groups} groups/tensor, rates={rates}\n")

    per_unit = {r: [] for r in rates}
    per_unit_l = {r: [] for r in rates}
    per_unit_e = {r: [] for r in rates}
    for key in keys:
        num = {r: 0.0 for r in rates}
        numl = {r: 0.0 for r in rates}
        nume = {r: 0.0 for r in rates}
        den = 0.0
        for name, t in units[key]:
            M, K = t["shape"][0], t["shape"][1]
            ngrp = (M * K) // G
            k = min(a.groups, ngrp)
            gidx = rng.choice(ngrp, size=k, replace=False)
            W = dequant_groups(mm, t, gidx).astype(np.float64)
            den += float(np.sum(W ** 2))
            for r in rates:
                num[r] += float(np.sum((quant_uniform(W, r) - W) ** 2))
                numl[r] += float(np.sum((quant_lloyd(W, r) - W) ** 2))
                if r in E8_STEP:
                    nume[r] += float(np.sum((quant_e8(W, r) - W) ** 2))
        for r in rates:
            per_unit[r].append(num[r] / den)
            per_unit_l[r].append(numl[r] / den)
            if r in E8_STEP:
                per_unit_e[r].append(nume[r] / den)

    def db(x):
        x = np.asarray(x)
        return -10.0 * np.log10(np.maximum(x, 1e-12))

    print(f"{'rate':<6} {'codebook':<10} {'med SNR':>8} {'min':>8} {'max':>8} "
          f"{'spread':>8} {'p10':>8} {'p90':>8}")
    print("-" * 74)
    for r in rates:
        arms = [("uniform", per_unit[r]), ("lloyd", per_unit_l[r])]
        if per_unit_e[r]:
            arms.append(("E8", per_unit_e[r]))
        for lbl, arr in arms:
            s = db(arr)
            print(f"{r}b{'':<4} {lbl:<10} {np.median(s):>8.2f} {s.min():>8.2f} "
                  f"{s.max():>8.2f} {s.max()-s.min():>8.2f} "
                  f"{np.percentile(s,10):>8.2f} {np.percentile(s,90):>8.2f}")


if __name__ == "__main__":
    main()
