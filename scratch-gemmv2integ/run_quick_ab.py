#!/usr/bin/env python3
"""One fresh-process matrix run per arm, A (8a2f42ef8 binaries, 33,280 B LDS
request) then B (LDS-fix binaries, 32,768 B), on one leased gfx11 card."""
import json
import os
from pathlib import Path
import subprocess
import sys

ARCH = sys.argv[1]
DEVICE = {'gfx1100': '0', 'gfx1151': '1'}[ARCH]
ROOT = Path('/home/kaden/hipfire-gemmv2')
OUT = ROOT / 'scratch-gemmv2integ' / ARCH / 'lds-fix'
OUT.mkdir(parents=True, exist_ok=True)
TREES = {'A': ROOT, 'B': ROOT / 'scratch-gemmv2integ' / 'fix-lds'}
MODEL = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
COMMAND = ['bench', MODEL, '--matrix', '--pp', '512,8192', '--ctx', '128',
           '--tg', '128', '--spec', 'off', '--runs', '3', '--warmups', '1',
           '--kv-mode', 'q8', '--kv-backend', 'vmm', '--json']
rows = {}
for index, arm in enumerate('AB', 1):
    tree = TREES[arm]
    home = OUT / f'{index:02d}-{arm}-home'
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=DEVICE, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_DAEMON_BIN=str(tree / 'target/release/daemon'))
    proc = subprocess.run([str(tree / 'target/release/hipfire'), *COMMAND], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = proc.stdout
    (OUT / f'{index:02d}-{arm}.log').write_text(log)
    if proc.returncode or f'GPU dev 0: {ARCH}' not in log or 'KV cache: Q8 vmm (' not in log:
        raise RuntimeError(f'run {index} {arm}: failed or wrong device/KV; see log')
    dec = json.JSONDecoder()
    report = None
    for off, ch in enumerate(log):
        if ch != '{':
            continue
        try:
            cand, _ = dec.raw_decode(log[off:])
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and cand.get('protocol') == 'synthetic-pp-tg-matrix-v1':
            report = cand
            break
    if report is None or report.get('kv_mode') != 'q8' or report.get('kv_backend') != 'vmm':
        raise RuntimeError(f'run {index} {arm}: Q8/VMM report missing')
    (OUT / f'{index:02d}-{arm}.json').write_text(json.dumps(report, indent=2) + '\n')
    r = {f'pp{x["tokens"]}': x['stats']['median'] for x in report['prefill']}
    r.update({f'tg{x["tokens"]}@{x["context"]}': x['stats']['median'] for x in report['decode']})
    rows[arm] = r
    print(f'{ARCH} {arm}: {r}', flush=True)
summary = {m: {'A': rows['A'][m], 'B': rows['B'][m], 'ratio': rows['B'][m] / rows['A'][m]}
           for m in rows['A']}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
