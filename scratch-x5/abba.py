#!/usr/bin/env python3
"""Fresh-process pinned-baseline vs X5 matrix gate; one architecture per invocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ARCH = sys.argv[1]
VISIBLE = {'gfx1151': '1', 'gfx1100': '0'}[ARCH]
MODEL = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
OUT = Path('/home/kaden/hipfire-x5/scratch-x5') / ARCH / 'abba'
OUT.mkdir(parents=True, exist_ok=True)
ROOTS = {'A': Path('/home/kaden/hipfire-prof040'), 'B': Path('/home/kaden/hipfire-x5')}
for index, arm in enumerate('ABBAABBA', 1):
    root = ROOTS[arm]
    home = OUT / f'{index:02d}-{arm}-home'
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith('HIPFIRE_')}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=VISIBLE, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_DAEMON_BIN=str(root / 'target/release/daemon'))
    cmd = [str(root / 'target/release/hipfire'), 'bench', MODEL, '--matrix',
           '--pp', '512,8192', '--ctx', '128', '--tg', '128', '--spec', 'off',
           '--runs', '3', '--warmups', '1', '--kv-mode', 'q8',
           '--kv-backend', 'vmm', '--json']
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (OUT / f'{index:02d}-{arm}.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{ARCH} run {index} {arm}: exit {result.returncode}; see log')
    report = None
    decoder = json.JSONDecoder()
    for offset, char in enumerate(result.stdout):
        if char == '{':
            try:
                candidate, _ = decoder.raw_decode(result.stdout[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get('protocol') == 'synthetic-pp-tg-matrix-v1':
                report = candidate
                break
    if report is None or report.get('kv_mode') != 'q8' or report.get('kv_backend') != 'vmm':
        raise RuntimeError(f'{ARCH} run {index} {arm}: missing effective Q8 VMM JSON')
    if report.get('gpu', {}).get('arch') != ARCH:
        raise RuntimeError(f'{ARCH} run {index} {arm}: device arch mismatch')
    (OUT / f'{index:02d}-{arm}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{ARCH} run {index}/8 arm={arm} complete', flush=True)
print('ABBA complete', flush=True)
