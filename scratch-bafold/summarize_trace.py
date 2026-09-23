#!/usr/bin/env python3
"""Summarize only the timed, uncached pp8192 request from rocprof CSV."""
import collections
import csv
import json
from pathlib import Path
import sys

arch, arm = sys.argv[1:3]
root = Path('/home/kaden/hipfire-bafold/scratch-bafold') / arch / 'trace' / arm
window = json.loads((root / 'trace-response.json').read_text())
with (root / 'daemon_kernel_trace.csv').open() as trace:
    all_rows = list(csv.DictReader(trace))
rows = [r for r in all_rows if window['start_monotonic_ns'] <= int(r['Start_Timestamp'])
        and int(r['End_Timestamp']) <= window['end_monotonic_ns']]
patterns = ('residual_iu4_v2c_set', 'residual_iu4_v2b_set',
            'residual_mmq_iu4_full_set', 'residual_wmma_gfx11_mw4_lds',
            'convert_f32_to_f16', 'split_mq4v2_z_betaalpha')
agg = collections.defaultdict(lambda: {'calls': 0, 'ms': 0.0, 'grid': collections.Counter()})
for r in rows:
    name = r['Kernel_Name']
    if not any(p in name for p in patterns):
        continue
    item = agg[name]
    item['calls'] += 1
    item['ms'] += (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e6
    grid = tuple(int(r[f'Grid_Size_{axis}']) // int(r[f'Workgroup_Size_{axis}'])
                 for axis in 'XYZ')
    item['grid'][str(grid)] += 1
summary = {'arch': arch, 'arm': arm, 'timed_kernels': len(rows),
           'request_wall_ms': window['wall_ms'], 'kernels': {
               k: {'calls': v['calls'], 'ms': round(v['ms'], 4), 'grid': v['grid']}
               for k, v in sorted(agg.items())}}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
