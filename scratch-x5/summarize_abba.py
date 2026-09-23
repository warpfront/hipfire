#!/usr/bin/env python3
"""Evaluate two fresh-process ABBA cycles from effective Q8/VMM reports."""
import json
from pathlib import Path
import statistics
import sys
arch = sys.argv[1]
root = Path('/home/kaden/hipfire-x5/scratch-x5') / arch / 'abba'
reports = [(i, arm, json.loads((root / f'{i:02d}-{arm}.json').read_text()))
           for i, arm in enumerate('ABBAABBA', 1)]
summary = {'arch': arch, 'rows': {}}
for family, key in [('prefill', 'tokens'), ('decode', 'context')]:
    lengths = sorted({row[key] for _, _, r in reports for row in r[family]})
    for size in lengths:
        vals = {arm: [next(row['stats']['median'] for row in r[family] if row[key] == size)
                      for _, a, r in reports if a == arm] for arm in 'AB'}
        ratio = statistics.mean(vals['B']) / statistics.mean(vals['A'])
        summary['rows'][f'{family}:{size}'] = {
            'baseline_tok_s': vals['A'], 'x5_tok_s': vals['B'], 'speedup': ratio,
            'cycle_speedup': [statistics.mean(vals['B'][i:i+2]) / statistics.mean(vals['A'][i:i+2])
                              for i in (0, 2)]}
summary['pass'] = (any(summary['rows'][f'prefill:{n}']['speedup'] >= 1.015 for n in (512, 8192))
                   and all(row['speedup'] >= 0.99 for row in summary['rows'].values()))
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
if not summary['pass']:
    sys.exit(1)
