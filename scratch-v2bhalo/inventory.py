#!/usr/bin/env python3
import csv
from collections import defaultdict
import json
from pathlib import Path

BASE = Path('/home/kaden/hipfire-v2bhalo/scratch-v2bhalo/trace')
for arch in ('gfx1151',):
    for length in ('8192', '512'):
        root = BASE / f'{arch}-{length}'
        if not (root / 'daemon_kernel_trace.csv').exists():
            continue
        trace = json.loads((root / 'trace-response.json').read_text())
        warm = json.loads((root / 'warm-response.json').read_text())
        rows = list(csv.DictReader((root / 'daemon_kernel_trace.csv').open()))
        selected = [r for r in rows if trace['start_monotonic_ns'] <= int(r['Start_Timestamp'])
                    and int(r['End_Timestamp']) <= trace['end_monotonic_ns']]
        warmrows = [r for r in rows if warm['start_monotonic_ns'] <= int(r['Start_Timestamp'])
                    and int(r['End_Timestamp']) <= warm['end_monotonic_ns']]
        if not selected:
            raise RuntimeError(f'{arch}-{length} has no second-request kernels')
        sums = defaultdict(lambda: [0, 0])
        for row in selected:
            item = sums[row['Kernel_Name']]
            item[0] += 1
            item[1] += int(row['End_Timestamp']) - int(row['Start_Timestamp'])
        ns = sum(item[1] for item in sums.values())
        info = {'arch': arch, 'length': length, 'kernel_count': len(selected),
                'warm_count': len(warmrows), 'total_trace_count': len(rows),
                'gpu_ms': ns / 1e6, 'wall_ms': trace['wall_ms'], 'warm_ms': warm['wall_ms'],
                'first_after_start_ms': (min(int(r['Start_Timestamp']) for r in selected) - trace['start_monotonic_ns']) / 1e6,
                'last_before_end_ms': (trace['end_monotonic_ns'] - max(int(r['End_Timestamp']) for r in selected)) / 1e6,
                'sum_by_symbol': [{'name': name, 'calls': value[0], 'ms': value[1] / 1e6,
                                   'pct': 100 * value[1] / ns}
                                  for name, value in sorted(sums.items(), key=lambda v: -v[1][1])],
                'usage': trace['usage']}
        (root / 'inventory.json').write_text(json.dumps(info, indent=2) + '\n')
        with (root / 'trace-only-kernels.csv').open('w') as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(selected)
        print(arch, length, 'selected', len(selected), 'warm', len(warmrows),
              'all', len(rows), 'gpu_ms', round(ns / 1e6, 3), 'wall_ms', round(trace['wall_ms'], 3),
              'first_ms', round(info['first_after_start_ms'], 3),
              'last_ms', round(info['last_before_end_ms'], 3))
