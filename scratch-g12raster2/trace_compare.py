#!/usr/bin/env python3
"""Compare rocprof pp8192 traces (trace/<label>/symbols.tsv) OFF vs ON.
usage: trace_compare.py off1,off2 on1,on2
IU4 GEMM symbols (`mmq_iu4`: SET/ADD and the F1 gate/up SILU) are compared per SET/ADD entry with the
`_g12r` suffix stripped; everything else is reported as one 'other' class."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'trace'


def load(label):
    d = {}
    for line in (ROOT / label / 'symbols.tsv').read_text().splitlines():
        name, calls, ms = line.split('\t')
        d[name] = (int(calls), float(ms))
    s = json.loads((ROOT / label / 'summary.json').read_text())
    return d, s


def classify(sym):
    if 'mmq_iu4' in sym:
        base = sym.replace('_g12r', '')
        return ('iu4 ' + base.split('(')[0], sym.endswith('_g12r') or '_g12r' in sym)
    return ('other', False)


def agg(labels):
    per = defaultdict(list)
    gpu, tok, g12r_seen = [], [], set()
    for lab in labels:
        d, s = load(lab)
        cls = defaultdict(lambda: [0, 0.0])
        for sym, (c, ms) in d.items():
            k, is_g = classify(sym)
            cls[k][0] += c
            cls[k][1] += ms
            if k != 'other':
                g12r_seen.add(is_g)
        for k, v in cls.items():
            per[k].append(v)
        gpu.append(s['gpu_ms'])
        tok.append((s['timings'] or {}).get('prefill_tok_s'))
    return per, gpu, tok, g12r_seen


off_l, on_l = sys.argv[1].split(','), sys.argv[2].split(',')
off, gpu_off, tok_off, g_off = agg(off_l)
on, gpu_on, tok_on, g_on = agg(on_l)
print(f'OFF {off_l}: g12r symbols seen={sorted(g_off)}  ON {on_l}: g12r symbols seen={sorted(g_on)}')
mean = lambda xs: sum(xs) / len(xs)
print(f"{'class':70s} {'calls':>6s} {'off ms':>10s} {'on ms':>10s} {'delta':>9s} {'delta%':>7s}")
tot_off = tot_on = 0.0
for k in sorted(off, key=lambda k: -mean([v[1] for v in off[k]])):
    o = mean([v[1] for v in off[k]])
    n = mean([v[1] for v in on.get(k, [[0, 0.0]])])
    calls = off[k][0][0]
    if k != 'other':
        tot_off += o
        tot_on += n
    print(f'{k:70s} {calls:6d} {o:10.3f} {n:10.3f} {n - o:+9.3f} {100 * (n / o - 1):+7.2f}')
print(f"{'IU4 GEMM total':70s} {'':6s} {tot_off:10.3f} {tot_on:10.3f} {tot_on - tot_off:+9.3f} {100 * (tot_on / tot_off - 1):+7.2f}")
print(f"{'GPU total (per run: off ' + str(gpu_off) + ' on ' + str(gpu_on) + ')':70s} {'':6s} {mean(gpu_off):10.3f} {mean(gpu_on):10.3f} {mean(gpu_on) - mean(gpu_off):+9.3f} {100 * (mean(gpu_on) / mean(gpu_off) - 1):+7.2f}")
print(f'prefill tok/s off {tok_off} mean {mean(tok_off):.1f}  on {tok_on} mean {mean(tok_on):.1f}  {100 * (mean(tok_on) / mean(tok_off) - 1):+.2f}%')
