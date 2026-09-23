#!/usr/bin/env python3
"""pp8192 rocprof A (HIPFIRE_V2B_ADDEPI=0) vs B (default): totals, per-symbol
deltas, and the V2B/norm split by call population. usage: trace_delta.py [dir]"""
import csv
import json
from pathlib import Path
import statistics as st
import sys

base = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'trace')
inv = {arm: json.loads((base / f'gfx1151-8192-{arm}' / 'inventory.json').read_text()) for arm in 'AB'}
a, b = inv['A'], inv['B']
lines = [f"gpu_ms A {a['gpu_ms']:.3f} B {b['gpu_ms']:.3f} delta {b['gpu_ms'] - a['gpu_ms']:+.3f}"
         f" ({100 * (b['gpu_ms'] - a['gpu_ms']) / a['gpu_ms']:+.2f}%)",
         f"wall_ms A {a['wall_ms']:.3f} B {b['wall_ms']:.3f} delta {b['wall_ms'] - a['wall_ms']:+.3f}"
         f" ({100 * (b['wall_ms'] - a['wall_ms']) / a['wall_ms']:+.2f}%)",
         f"pp8192 t/s (wall) A {8192e3 / a['wall_ms']:.1f} B {8192e3 / b['wall_ms']:.1f}",
         'per-symbol (B - A), |delta| >= 0.5 ms:']
sa = {r['name']: r for r in a['sum_by_symbol']}
sb = {r['name']: r for r in b['sum_by_symbol']}
rows = []
for n in set(sa) | set(sb):
    x = sa.get(n, {'calls': 0, 'ms': 0.0})
    y = sb.get(n, {'calls': 0, 'ms': 0.0})
    rows.append((y['ms'] - x['ms'], n, x['calls'], x['ms'], y['calls'], y['ms']))
for r in sorted(rows, key=lambda r: -abs(r[0])):
    if abs(r[0]) >= 0.5:
        lines.append('  %+9.3f %-52s A %4d %9.3f  B %4d %9.3f' % r)
lines.append('V2B / IU4 RMSNorm populations (K17408 ADD > 12 ms):')
for arm in 'AB':
    rows = list(csv.DictReader((base / f'gfx1151-8192-{arm}' / 'trace-only-kernels.csv').open()))
    d = {}
    for r in rows:
        n = r['Kernel_Name']
        if 'v2b' in n or 'rmsnorm_mq_rotate_awq_i4' in n:
            t = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e6
            if 'residual_iu4_v2b' in n:
                n += ' big' if t > 12 else ' small'
            d.setdefault(n, []).append(t)
    for k, v in sorted(d.items()):
        lines.append(f'  {arm} {k:<58} n={len(v):3d} sum={sum(v):9.3f} mean={st.mean(v):7.3f}')
text = '\n'.join(lines) + '\n'
(base / 'gfx1151-8192-delta.txt').write_text(text)
print(text, end='')
