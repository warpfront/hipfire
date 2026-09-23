#!/usr/bin/env python3
"""Fresh-process gfx11 WT2 and matrix ABBA, pinned to Q8 VMM."""
import json
import os
from pathlib import Path
import subprocess
import sys

arch, mode = sys.argv[1:3]
visible = {'gfx1100': '0', 'gfx1151': '1'}[arch]
root = Path('/home/kaden/hipfire-bafold/scratch-bafold') / arch / mode
root.mkdir(parents=True, exist_ok=True)
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
reference = '/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin'
binaries = {'A': Path('/home/kaden/hipfire-bafold-base/target/release'),
            'B': Path('/home/kaden/hipfire-bafold/target/release')}

for idx, arm in enumerate('ABBAABBA' if mode == 'abba' else ('A' if mode == 'wt2a' else 'B'), 1):
    home = root / f'{idx:02d}-{arm}-home'
    home.mkdir(exist_ok=True)
    env = {key: val for key, val in os.environ.items() if not key.startswith('HIPFIRE_')}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_DAEMON_BIN=str(binaries[arm] / 'daemon'))
    if mode == 'abba':
        cmd = [str(binaries[arm] / 'hipfire'), 'bench', model, '--matrix',
               '--pp', '512,8192', '--ctx', '128', '--tg', '128', '--spec', 'off',
               '--runs', '3', '--warmups', '1', '--kv-mode', 'q8', '--kv-backend', 'vmm', '--json']
    else:
        cmd = [str(binaries[arm] / 'examples/eval_hipfire'), '--model', model,
               '--ref', reference, '--output', str(root / 'fold-q8.kldseq'),
               '--max-chunks', '24', '--kv-mode', 'q8', '--kv-v', 'q8']
    result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (root / f'{idx:02d}-{arm}.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{arch} {mode} {arm}: exit {result.returncode}; see {root}')
    if 'KV cache: Q8 vmm (' not in result.stdout:
        raise RuntimeError(f'{arch} {mode} {arm}: missing Q8 VMM assertion; see {root}')
    if mode == 'abba':
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
        if not report or report.get('kv_mode') != 'q8' or report.get('kv_backend') != 'vmm' or report.get('gpu', {}).get('arch') != arch:
            raise RuntimeError(f'{arch} {mode} {arm}: effective protocol mismatch')
        (root / f'{idx:02d}-{arm}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{arch} {mode} {idx} {arm} complete', flush=True)
