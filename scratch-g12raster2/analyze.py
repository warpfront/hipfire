#!/usr/bin/env python3
"""Weighted full-shape summary of host TIME logs.
usage: analyze.py log [log ...]
Per log: per-shape median ms per arm, paired speedup (median of per-round ratios),
and the pp8192 call-weighted total per arm vs control."""
import re
import sys
from collections import defaultdict

pat = re.compile(r"TIME order=(\S+) N=(\d+) mode=(\S+) shape=(\S+) calls=(\d+) arm=(\S+) median_ms=([\d.]+) .*speedup_vs_control_med=([\d.]+) sclk_med_mhz=(-?\d+)")


def main():
    agg = defaultdict(lambda: defaultdict(dict))  # arm -> log -> shape -> (ms, spd)
    calls = {}
    for path in sys.argv[1:]:
        for line in open(path):
            m = pat.search(line)
            if not m:
                continue
            order, n, mode, shape, c, arm, ms, spd, clk = m.groups()
            calls[shape] = int(c)
            agg[arm][path][shape] = (float(ms), float(spd))
    logs = sys.argv[1:]
    shapes = list(calls)
    for path in logs:
        print(f"== {path}")
        ctrl = agg[next(iter(agg))][path]
        tot_c = sum(calls[s] * ctrl[s][0] for s in shapes if s in ctrl)
        print(f"{'arm':12s} " + " ".join(f"{s.split('_')[0]+'_'+s.split('_')[1]:>11s}" for s in shapes) + "   weighted_ms  saving%  paired_wsave%  worst_shape%")
        for arm in agg:
            d = agg[arm].get(path)
            if not d:
                continue
            tot = sum(calls[s] * d[s][0] for s in shapes if s in d)
            # paired weighted saving: control ms scaled by per-shape paired speedup
            tot_p = sum(calls[s] * ctrl[s][0] / d[s][1] for s in shapes if s in d)
            worst = min((d[s][1] - 1) * 100 for s in shapes if s in d)
            print(f"{arm:12s} " + " ".join(f"{(d[s][1]-1)*100:+11.2f}" for s in shapes if s in d)
                  + f"   {tot:10.2f}  {100*(1-tot/tot_c):+6.2f}   {100*(1-tot_p/tot_c):+6.2f}   {worst:+6.2f}")


main()
