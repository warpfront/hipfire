#!/usr/bin/env python3
"""F1Lite pp8192 trace delta: second-request kernels of trace dirs A (baseline)
and B (F1-lite), grouped by (kernel, grid). Prints the targeted FFN rows
(gate/up SET, F1 gate_up_silu, SwiGLU producer, h-producer), their totals, and
the whole-trace GPU/wall times. Usage: trace_delta.py <A dir> <B dir>"""
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys


def load(root):
    root = Path(root)
    trace = json.loads((root / 'trace-response.json').read_text())
    rows = [r for r in csv.DictReader((root / 'daemon_kernel_trace.csv').open())
            if trace['start_monotonic_ns'] <= int(r['Start_Timestamp'])
            and int(r['End_Timestamp']) <= trace['end_monotonic_ns']]
    if not rows:
        raise RuntimeError(f'{root}: no second-request kernels')
    sums = defaultdict(lambda: [0, 0.0])
    for r in rows:
        key = (r['Kernel_Name'], f"{r['Grid_Size_X']}x{r['Grid_Size_Y']}")
        sums[key][0] += 1
        sums[key][1] += (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e6
    gpu = sum(v[1] for v in sums.values())
    return sums, gpu, trace['wall_ms'], len(rows)


def targeted(name):
    return ('_set_gfx11' in name or 'gate_up_silu' in name
            or name.startswith('fused_silu_mul_mq_rotate_awq_i4'))


a, ga, wa, na = load(sys.argv[1])
b, gb, wb, nb = load(sys.argv[2])
print(f"{'kernel':58s} {'grid':>10s} {'A calls':>7s} {'A ms':>9s} {'B calls':>7s} {'B ms':>9s}")
ta = tb = 0.0
for key in sorted(set(a) | set(b)):
    if not targeted(key[0]):
        continue
    ca, ma = a.get(key, [0, 0.0])
    cb, mb = b.get(key, [0, 0.0])
    ta, tb = ta + ma, tb + mb
    print(f'{key[0][:58]:58s} {key[1]:>10s} {ca:7d} {ma:9.2f} {cb:7d} {mb:9.2f}')
print(f"{'targeted total (all SET + F1 + producers)':69s} {ta:17.2f} {tb:17.2f}  delta {tb - ta:+.2f} ms")
print(f'kernels A {na} B {nb}; GPU A {ga:.2f} B {gb:.2f} ms (delta {gb - ga:+.2f}); '
      f'request wall A {wa:.2f} B {wb:.2f} ms (delta {wb - wa:+.2f})')
