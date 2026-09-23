#!/usr/bin/env python3
"""Per-run paired speedup (%) per shape and call-weighted saving for one or more arms vs control.
usage: summarize_arm.py <arm[,arm]> log [log ...]"""
import re
import sys

pat = re.compile(r"TIME order=(\S+) N=(\d+) mode=(\S+) shape=(\S+) calls=(\d+) arm=(\S+) median_ms=([\d.]+) .*speedup_vs_control_med=([\d.]+)")
ORDER = ['gate_up_SET_M17408_K5120', 'gate_up_SILU_M17408_K5120', 'qkvza_QKV_SET_M10240_K5120', 'qkvza_Z_SET_M6144_K5120',
         'qkvza_BA_SET_M48_K5120', 'qkv_Q_SET_M12288_K5120', 'qkv_KV_SET_M1024_K5120',
         'res_ADD_M5120_K6144', 'res_ADD_M5120_K17408']
SHORT = ['gateSET', 'gateSILU', 'QKV', 'Z', 'BA', 'FA_Q', 'FA_KV', 'ADD6144', 'ADD17408']
arms = sys.argv[1].split(',')
print(f"{'run':28s} {'arm':12s} " + ' '.join(f'{s:>8s}' for s in SHORT) + '  weighted')
for path in sys.argv[2:]:
    d = {}
    ref = None
    for line in open(path):
        m = pat.search(line)
        if not m:
            continue
        o, n, mode, shape, c, arm, ms, sp = m.groups()
        ref = ref or arm  # first arm printed per shape is the reference
        d.setdefault(shape, {})[arm] = (float(ms), float(sp), int(c))
    for arm in arms:
        if not all(arm in v for v in d.values()):
            continue
        tc = sum(v[ref][0] * v[ref][2] for v in d.values())
        ta = sum(v[arm][0] * v[arm][2] for v in d.values())
        cells = [f'{(d[s][arm][1] - 1) * 100:+8.2f}' if s in d else f"{'-':>8s}" for s in ORDER]
        print(f"{path.split('/')[-1][:-4]:28s} {arm:12s} " + ' '.join(cells) + f'  {100 * (1 - ta / tc):+6.2f}%')
