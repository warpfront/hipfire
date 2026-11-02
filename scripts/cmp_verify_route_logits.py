#!/usr/bin/env python3
"""Compare two raw LE f32 logit dumps: cmp_verify_route_logits.py base.f32 cand.f32 VOCAB"""
import sys

import numpy as np


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    base_p, cand_p, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
    A = np.memmap(base_p, dtype="<f4", mode="r").reshape(-1, vocab)
    B = np.memmap(cand_p, dtype="<f4", mode="r").reshape(-1, vocab)
    if A.shape != B.shape:
        print(f"shape mismatch {A.shape} vs {B.shape}")
        sys.exit(1)
    n = A.shape[0]
    kl = np.empty(n)
    mx = np.empty(n)
    rel = np.empty(n)
    eq = np.empty(n, dtype=bool)
    margin = np.empty(n)
    for i in range(n):
        a = np.asarray(A[i], dtype=np.float64)
        b = np.asarray(B[i], dtype=np.float64)
        la, lb = a - lse(a), b - lse(b)
        kl[i] = float((np.exp(la) * (la - lb)).sum())
        d = a - b
        mx[i] = np.abs(d).max()
        na = np.linalg.norm(a)
        rel[i] = np.linalg.norm(d) / na if na > 0 else float("nan")
        ia, ib = int(a.argmax()), int(b.argmax())
        eq[i] = ia == ib
        top2 = np.partition(a, -2)[-2:]
        margin[i] = top2[1] - top2[0]
    print(f"rows={n} vocab={vocab}")
    print(f"KL(base||cand) nats: mean={kl.mean():.6e} p99={np.percentile(kl, 99):.6e} max={kl.max():.6e}")
    print(f"max|dlogit|: mean={mx.mean():.6e} max={mx.max():.6e}")
    print(f"rel_rms: mean={np.nanmean(rel):.6e} max={np.nanmax(rel):.6e}")
    print(f"argmax agreement: {100.0 * eq.mean():.4f}% ({int(eq.sum())}/{n})")
    bad = np.nonzero(~eq)[0]
    if len(bad):
        print("argmax disagreements (row, base top1-top2 margin):")
        for i in bad:
            print(f"  row {i}: margin={margin[i]:.6f} KL={kl[i]:.6e}")


if __name__ == "__main__":
    main()
