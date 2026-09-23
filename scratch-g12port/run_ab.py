#!/usr/bin/env python3
"""Fresh-process pp8192 A/B on one local gfx1201 card (ABBA order).

Usage: run_ab.py <label> <card> <arm-spec> <arm-spec> [--pairs P]
arm-spec: NAME:BINDIR[:KEY=VALUE,...]. Each process runs
`hipfire bench <model> --matrix --pp 8192 --ctx 128 --tg 128 --spec off --runs 3
--warmups 1 --kv-mode fp8 --json` (the guard's command at pp8192 only) with HIPFIRE_DAEMON_BIN pinned to the arm's daemon, after
one discarded warmup process. Writes scratch-g12port/ab/<label>/summary.json.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

CARDS = {'A': 'GPU-9eb7aeda51c88ffd', 'C': 'GPU-085289909a86cc63',
         'D': 'GPU-6109a4cb5f833235', 'E': 'GPU-05f92432f2312a0e'}
ROOT = Path(__file__).resolve().parent
MODEL = '/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'

args = sys.argv[1:]
pairs = 1
tg = '--tg' in args
if tg:
    args.remove('--tg')
if '--pairs' in args:
    i = args.index('--pairs')
    pairs = int(args[i + 1])
    del args[i:i + 2]
label, card, *specs = args
arms = {}
order = []
for spec in specs:
    parts = spec.split(':', 2)
    name, bindir = parts[0], Path(parts[1]).resolve()
    extra = dict(kv.split('=', 1) for kv in parts[2].split(',')) if len(parts) > 2 and parts[2] else {}
    arms[name] = (bindir, extra)
    order.append(name)
a, b = order
out = ROOT / 'ab' / label
out.mkdir(parents=True, exist_ok=True)
home = Path(f'/home/kaden/.hipfire-homes/g12port-{card}')
home.mkdir(exist_ok=True)


def run(name, tag):
    bindir, extra = arms[name]
    (home / '.hipfire' / 'daemon.pid').unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
    env.update(HOME=str(home), ROCR_VISIBLE_DEVICES=CARDS[card], HIP_VISIBLE_DEVICES=CARDS[card],
               HIPFIRE_KERNEL_CACHE=str(home / '.hipfire_kernels'),
               HIPFIRE_MODELS_DIR='/home/kaden/.hipfire/models', HIPFIRE_GRAPH='1',
               HIPFIRE_DAEMON_BIN=str(bindir / 'daemon'))
    env.update(extra)
    cmd = [str(bindir / 'hipfire'), 'bench', MODEL, '--matrix', '--pp', '8192', '--ctx', '128',
           '--tg', '128', '--spec', 'off', '--runs', '3', '--warmups', '1', '--kv-mode', 'fp8', '--json']
    with (out / f'{tag}.daemon.log').open('w') as log:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=log, text=True)
    (out / f'{tag}.json').write_text(proc.stdout)
    text = (out / f'{tag}.daemon.log').read_text(errors='replace')
    if proc.returncode or 'KV cache: Fp8 vmm (' not in text or 'gfx1201' not in text:
        raise RuntimeError(f'{tag}: rc={proc.returncode} or fp8 VMM/gfx1201 not asserted')
    rep = json.loads(proc.stdout)
    row = dict(tag=tag, arm=name)
    pp = [r for r in rep['prefill'] if r['tokens'] == 8192][0]['samples']
    row['pp8192_samples'] = pp
    row['pp8192_median'] = statistics.median(pp)
    if rep.get('decode'):
        row['tg128_samples'] = rep['decode'][0]['samples']
        row['tg128_median'] = statistics.median(row['tg128_samples'])
    print(json.dumps(row), flush=True)
    return row


run(a, 'warmup')
rows = []
seq = []
for p in range(pairs):
    seq += [a, b, b, a] if p % 2 == 0 else [b, a, a, b]
for i, name in enumerate(seq):
    rows.append(run(name, f'p{i}-{name}'))
summary = dict(label=label, card=card, arms={k: dict(bindir=str(v[0]), env=v[1]) for k, v in arms.items()},
               rows=rows)
for name in order:
    meds = [r['pp8192_median'] for r in rows if r['arm'] == name]
    summary[f'{name}_pp8192'] = dict(process_medians=meds, mean=statistics.mean(meds))
    if True:
        tgs = [r['tg128_median'] for r in rows if r['arm'] == name]
        summary[f'{name}_tg128'] = dict(process_medians=tgs, mean=statistics.mean(tgs))
summary['delta_pct'] = 100 * (summary[f'{b}_pp8192']['mean'] / summary[f'{a}_pp8192']['mean'] - 1)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'arms')}, indent=2))
