#!/usr/bin/env python3
"""Two fresh-process ABBA matrix cycles: pinned mq4-lloyd (A) vs gfx11-v2b-halo (B)."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ARCH = sys.argv[1]
DEVICE = {'gfx1100': '0', 'gfx1151': '1'}[ARCH]
ROOT = Path('/home/kaden/hipfire-v2bhalo')
OUT = ROOT / 'scratch-v2bhalo' / ARCH / 'abba'
OUT.mkdir(parents=True, exist_ok=True)
# A: byte copies of the mq4-lloyd (84d2992a, code == land-042 776358d6d)
# release binaries, pinned so a concurrent rebuild cannot change the baseline.
BASELINE = ROOT / 'scratch-v2bhalo' / 'baseline-mq4lloyd'
TREES = {'A': BASELINE, 'B': ROOT}
MODEL = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
COMMAND = ['bench', MODEL, '--matrix', '--pp', '512,8192', '--ctx', '128',
           '--tg', '128', '--spec', 'off', '--runs', '3', '--warmups', '1',
           '--kv-mode', 'q8', '--kv-backend', 'vmm', '--json']
# Ship gate: pp512 or pp8192 >= +1.5%, no row more than 1% slower, decode
# within 1%.
SHIP_RATIO = 1.015
ROW_FLOOR = 0.99
reports = []
for index, arm in enumerate('ABBAABBA', 1):
    tree = TREES[arm]
    stem = OUT / f'{index:02d}-{arm}'
    home = OUT / f'{index:02d}-{arm}-home'
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith('HIPFIRE_')}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=DEVICE, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_DAEMON_BIN=str(tree / 'target/release/daemon'))
    command = [str(tree / 'target/release/hipfire'), *COMMAND]
    stem.with_suffix('.command.json').write_text(json.dumps(command) + '\n')
    proc = subprocess.Popen(command, cwd=tree, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        expected_rocr = f'ROCR_VISIBLE_DEVICES={DEVICE}'.encode()
        for _ in range(100):
            process_env = Path(f'/proc/{proc.pid}/environ').read_bytes().split(b'\0')
            if expected_rocr in process_env and b'HIP_VISIBLE_DEVICES=0' in process_env:
                break
            time.sleep(0.01)  # Popen returns before the child has exec'd.
        else:
            raise RuntimeError(f'{ARCH} run {index} {arm}: process visibility not attested')
        stem.with_suffix('.env').write_text(f'pid={proc.pid}\n' +
            '\n'.join(item.decode(errors='replace') for item in process_env if item) + '\n')
        log, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    stem.with_suffix('.log').write_text(log)
    if proc.returncode:
        raise RuntimeError(f'{ARCH} run {index} {arm} exited {proc.returncode}; see {stem}.log')
    if f'GPU dev 0: {ARCH}' not in log or 'KV cache: Q8 vmm (' not in log:
        raise RuntimeError(f'{ARCH} run {index} {arm}: wrong device or KV backend; see log')
    decoder = json.JSONDecoder()
    report = None
    for offset, char in enumerate(log):
        if char != '{':
            continue
        try:
            candidate, _ = decoder.raw_decode(log[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get('protocol') == 'synthetic-pp-tg-matrix-v1':
            report = candidate
            break
    if report is None or report.get('kv_mode') != 'q8' or report.get('kv_backend') != 'vmm':
        raise RuntimeError(f'{ARCH} run {index} {arm}: effective Q8/VMM report missing')
    if report.get('gpu', {}).get('arch') != ARCH:
        raise RuntimeError(f'{ARCH} run {index} {arm}: report arch mismatch')
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = {f'pp{x["tokens"]}': x['stats']['median'] for x in report['prefill']}
    rows.update({f'tg{x["tokens"]}@{x["context"]}': x['stats']['median']
                 for x in report['decode']})
    reports.append((arm, rows))
    print(f'{ARCH} {index}/8 {arm}: {rows}', flush=True)
summary = {'arch': ARCH, 'rows': {}}
for metric in ('pp512', 'pp8192', 'tg128@128'):
    values = {arm: [rows[metric] for a, rows in reports if a == arm] for arm in 'AB'}
    ratio = statistics.mean(values['B']) / statistics.mean(values['A'])
    cycles = [statistics.mean(values['B'][i:i+2]) / statistics.mean(values['A'][i:i+2])
              for i in (0, 2)]
    summary['rows'][metric] = {'mq4lloyd_tok_s': values['A'], 'v2b_tok_s': values['B'],
                               'mq4lloyd_mean': statistics.mean(values['A']),
                               'v2b_mean': statistics.mean(values['B']),
                               'speedup': ratio, 'cycle_speedup': cycles}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
speed = {metric: row['speedup'] for metric, row in summary['rows'].items()}
if max(speed['pp512'], speed['pp8192']) < SHIP_RATIO:
    raise RuntimeError(f'{ARCH} neither pp512 nor pp8192 reaches {SHIP_RATIO}')
for metric, ratio in speed.items():
    if ratio < ROW_FLOOR:
        raise RuntimeError(f'{ARCH} {metric} more than 1% slower ({ratio:.4f})')
if abs(speed['tg128@128'] - 1) > 0.01:
    raise RuntimeError(f'{ARCH} decode outside 1%')
print(f'{ARCH} ABBA gate PASS', flush=True)
