#!/usr/bin/env python3
"""Run Prof040's identical warm/trace protocol against the FA2 candidate."""
import csv
import importlib.util
import json
from pathlib import Path

root = Path('/home/kaden/hipfire-fa2q16')
spec = importlib.util.spec_from_file_location(
    'prof040', '/home/kaden/hipfire-prof040/scratch-prof040/run_profile.py')
prof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prof)
prof.ROOT = root
prof.OUT = root / 'scratch-fa2q16' / 'trace-default'
prompt = root / 'benchmarks/prompts/pp8192.txt'
linked = not prompt.exists()
if linked:
    prompt.symlink_to('/home/kaden/hipfire-prof040/benchmarks/prompts/pp8192.txt')
try:
    print('FA2_TRACE_START gfx1151 pp8192', flush=True)
    prof.run('gfx1151', '8192')
finally:
    if linked:
        prompt.unlink()
dest = prof.OUT / 'gfx1151-8192'
window = json.loads((dest / 'trace-response.json').read_text())
with (dest / 'daemon_kernel_trace.csv').open() as source:
    rows = [row for row in csv.DictReader(source)
            if window['start_monotonic_ns'] <= int(row['Start_Timestamp'])
            and int(row['End_Timestamp']) <= window['end_monotonic_ns']]
attention = [row for row in rows if row['Kernel_Name'] == 'attention_q8_0_fa2_gqa_gfx11']
duration = sum(int(row['End_Timestamp']) - int(row['Start_Timestamp'])
               for row in attention) / 1e6
grid = sorted({(row['Grid_Size_X'], row['Grid_Size_Y'], row['Workgroup_Size_X'])
               for row in attention})
summary = {'fa2_symbol': 'attention_q8_0_fa2_gqa_gfx11',
           'calls': len(attention), 'fa2_ms': duration,
           'grid_workitems_x_y_block_x': grid,
           'kernel_window_count': len(rows),
           'trace_wall_ms': window['wall_ms']}
(dest / 'fa2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('FA2_TRACE', summary, flush=True)
