#!/usr/bin/env python3
"""Per-shape IU4 GEMM time from rocprof traces, OFF vs ON.
usage: trace_shapes.py off1,off2 on1,on2
Shape key = (SET|ADD, row tiles*128 = padded M, token tiles) from the dispatch grid."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'trace'
NAMES = {(False, 136): 'gate_up SET M17408', (False, 80): 'LA QKV SET M10240', (False, 48): 'LA Z SET M6144',
         (False, 1): 'LA beta/alpha SET M48', (False, 96): 'FA Q SET M12288', (False, 8): 'FA K/V SET M1024',
         (True, 40): 'ADD M5120 (K by call class)'}


def shapes(label):
    resp = json.loads((ROOT / label / 'trace-response.json').read_text())
    t0, t1 = resp['start_monotonic_ns'], resp['end_monotonic_ns']
    out = defaultdict(lambda: [0, 0.0])
    for r in csv.DictReader((ROOT / label / 'daemon_kernel_trace.csv').open()):
        if 'mmq_iu4' not in r['Kernel_Name']:
            continue
        s, e = int(r['Start_Timestamp']), int(r['End_Timestamp'])
        if s < t0 or e > t1:
            continue
        add = 'full_add' in r['Kernel_Name']
        rt = int(r['Grid_Size_X']) // int(r['Workgroup_Size_X'])
        tt = int(r['Grid_Size_Y']) // int(r['Workgroup_Size_Y'])
        key = (add, rt, tt, 'silu' if 'silu' in r['Kernel_Name'] else 0)
        # ADD K is not in the grid: the two ADD classes alternate LA/FA out (K6144)
        # and down (K17408); split them by duration below.
        out[key][0] += 1
        out[key][1] += (e - s) / 1e6
        out[key].append((e - s) / 1e6)
    return out


def split_add(d):
    res = {}
    for key, v in d.items():
        add, rt, tt, vg = key
        if not add:
            res[(add, rt, tt, 'SILU' if vg == 'silu' else 'K5120')] = (v[0], v[1])
            continue
        durs = sorted(v[2:])
        # bimodal: K6144 (~2 ms) vs K17408 (~6 ms)
        lo = [x for x in durs if x < 4.0]
        hi = [x for x in durs if x >= 4.0]
        res[(add, rt, tt, 'K6144')] = (len(lo), sum(lo))
        res[(add, rt, tt, 'K17408')] = (len(hi), sum(hi))
    return res


def mean_runs(labels):
    runs = [split_add(shapes(l)) for l in labels]
    keys = set().union(*runs)
    return {k: (runs[0][k][0], sum(r[k][1] for r in runs) / len(runs)) for k in keys}


off = mean_runs(sys.argv[1].split(','))
on = mean_runs(sys.argv[2].split(','))
print(f"{'shape':34s} {'tok tiles':>9s} {'calls':>6s} {'off ms':>10s} {'on ms':>10s} {'delta ms':>9s} {'delta%':>7s} {'off ms/call':>11s} {'on ms/call':>10s}")
to = tn = 0.0
for k in sorted(off, key=lambda k: -off[k][1]):
    add, rt, tt, kk = k
    name = ('gate_up SILU M17408 (F1)' if kk == 'SILU' else NAMES.get((add, rt), f"{'ADD' if add else 'SET'} rows{rt*128}")) + ('' if not add else ' ' + kk)
    c, o = off[k]
    n = on.get(k, (0, 0.0))[1]
    to += o
    tn += n
    print(f'{name:34s} {tt:9d} {c:6d} {o:10.3f} {n:10.3f} {n - o:+9.3f} {100 * (n / o - 1):+7.2f} {o / c:11.4f} {n / c:10.4f}')
print(f"{'IU4 total':34s} {'':9s} {'':6s} {to:10.3f} {tn:10.3f} {tn - to:+9.3f} {100 * (tn / to - 1):+7.2f}")
