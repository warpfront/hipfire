#!/usr/bin/env python3
"""WT2 c24 (gfx1201 default fp8 KV) for one arm on one local card.

Usage: run_wt2.py <label> <bin-dir> <card> [KEY=VALUE ...]
card: A|C|D|E. Writes scratch-a4c2/wt2/<label>.{kldseq,log,env}.
"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

CARDS = {'A': 'GPU-9eb7aeda51c88ffd', 'C': 'GPU-085289909a86cc63',
         'D': 'GPU-6109a4cb5f833235', 'E': 'GPU-05f92432f2312a0e'}
ROOT = Path(__file__).resolve().parent
MODEL = '/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
REF = '/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin'

label, bindir, card = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
extra = dict(kv.split('=', 1) for kv in sys.argv[4:])
uuid = CARDS[card]
out = ROOT / 'wt2'
out.mkdir(exist_ok=True)
home = Path('/home/kaden/.hipfire-homes/a4c2')
home.mkdir(exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
env.update(HOME=str(home), ROCR_VISIBLE_DEVICES=uuid, HIP_VISIBLE_DEVICES=uuid,
           HIPFIRE_KERNEL_CACHE=str(home / '.hipfire_kernels'),
           HIPFIRE_MODELS_DIR='/home/kaden/.hipfire/models', HIPFIRE_GRAPH='1')
env.update(extra)
seq = out / f'{label}.kldseq'
cmd = [str(bindir / 'eval_hipfire'), '--model', MODEL, '--ref', REF, '--kv-mode', 'fp8',
       '--kv-v', 'q8', '--scoring-mode', 'prefill', '--max-chunks', '24', '--output', str(seq)]
(out / f'{label}.env').write_text(' '.join(f'{k}={v}' for k, v in sorted(env.items())
                                           if k.startswith(('HIPFIRE_', 'ROCR', 'HIP_', 'HOME')))
                                  + '\n' + ' '.join(cmd) + '\n')
proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(out / f'{label}.log').write_text(proc.stdout)
if proc.returncode:
    sys.exit(f'{label}: eval exited {proc.returncode}')
if 'gfx1201' not in proc.stdout:
    sys.exit(f'{label}: gfx1201 not asserted in log')
m = re.search(r'slice-mean KLD = ([0-9.]+)', proc.stdout)
if not m:
    sys.exit(f'{label}: no slice-mean KLD')
digest = hashlib.sha256(seq.read_bytes()).hexdigest()
print(f'{label} card-{card} WT2 c24 fp8 KLD={m.group(1)} sha256={digest}')
