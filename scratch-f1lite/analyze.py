#!/usr/bin/env python3
"""Summarize F1Lite timing logs: per (card, N, arm) the mean of the per-process
medians with min/max, and the paired per-process A/C ratio."""
import collections
import glob
import re
import statistics
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "logs"
pat = re.compile(r"RESULT arm=(\S+) stem=(\S+) N=(\d+) total_ms=([\d.]+) gemm_ms=([\d.]+) "
                 r"producer_ms=([\d.]+)")
for card in ("xtx", "halo"):
    files = sorted(glob.glob(f"{root}/{card}-time-*.log"))
    if not files:
        continue
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in files:
        for line in open(f):
            m = pat.search(line)
            if m:
                arm, _, n, tot, g, p = m.groups()
                rows[int(n)][arm][f] = (float(tot), float(g), float(p))
    print(f"## {card} ({len(files)} processes)")
    for n, arms in sorted(rows.items()):
        print(f"N={n}")
        for arm, per in arms.items():
            t = [v[0] for v in per.values()]
            print(f"  {arm:10s} total {statistics.mean(t):8.4f} [{min(t):.4f},{max(t):.4f}]"
                  f" gemm {statistics.mean(v[1] for v in per.values()):8.4f}"
                  f" producer {statistics.mean(v[2] for v in per.values()):7.4f}")
        a, c = arms.get("A_pair", {}), arms.get("C_f1lite", {})
        r = [a[f][0] / c[f][0] for f in a if f in c]
        if r:
            print(f"  A/C {statistics.mean(r):.4f} [{min(r):.4f},{max(r):.4f}]"
                  f"  saved/call {statistics.mean(a[f][0] - c[f][0] for f in a if f in c):.4f} ms")
