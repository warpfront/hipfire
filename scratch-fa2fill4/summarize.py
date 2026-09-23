#!/usr/bin/env python3
"""Summarize the traced (second) prefill of run_profile.py outputs, off vs on.

usage: summarize.py <trace_dir_on> <trace_dir_off>
Only dispatches that start inside the 'trace' request's monotonic window are
counted (rocprofv3 timestamps are CLOCK_MONOTONIC ns)."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(d):
    d = Path(d)
    r = json.loads((d / 'trace-response.json').read_text())
    lo, hi = r['start_monotonic_ns'], r['end_monotonic_ns']
    per = defaultdict(lambda: [0, 0.0, set()])
    total, first, last = 0.0, None, None
    with (d / 'daemon_kernel_trace.csv').open() as f:
        for row in csv.DictReader(f):
            s, e = int(row['Start_Timestamp']), int(row['End_Timestamp'])
            if not lo <= s <= hi:
                continue
            ms = (e - s) / 1e6
            k = row['Kernel_Name']
            per[k][0] += 1
            per[k][1] += ms
            per[k][2].add(row['VGPR_Count'])
            total += ms
            first = s if first is None else min(first, s)
            last = e if last is None else max(last, e)
    return r['wall_ms'], total, (last - first) / 1e6, per


on, off = load(sys.argv[1]), load(sys.argv[2])
for name, (wall, gpu, span, per) in (('on', on), ('off', off)):
    print(f'{name}: request wall {wall:.2f} ms, summed GPU {gpu:.2f} ms, GPU span {span:.2f} ms, dispatches {sum(v[0] for v in per.values())}')
print(f'delta summed GPU {on[1] - off[1]:+.2f} ms ({(on[1] / off[1] - 1) * 100:+.2f}%), wall {on[0] - off[0]:+.2f} ms')
keys = set(on[3]) | set(off[3])
rows = sorted(keys, key=lambda k: -abs(on[3].get(k, [0, 0.0])[1] - off[3].get(k, [0, 0.0])[1]))
print(f'{"kernel":60s} {"n_off":>6s} {"ms_off":>10s} {"n_on":>6s} {"ms_on":>10s} {"delta":>9s} vgpr_off/on')
for k in rows[:12]:
    a, b = off[3].get(k, [0, 0.0, set()]), on[3].get(k, [0, 0.0, set()])
    print(f'{k[:60]:60s} {a[0]:6d} {a[1]:10.2f} {b[0]:6d} {b[1]:10.2f} {b[1] - a[1]:+9.2f} {sorted(a[2])}/{sorted(b[2])}')
