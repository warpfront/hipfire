#!/usr/bin/env python3
"""WT2 c24 Q8/Q8 for gfx11 exact GDN preparation and gfx1100 KKT."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

arch, tag = sys.argv[1:3]
assert arch in ('gfx1100', 'gfx1151')
assert tag in ('baseline', 'default')
root = Path('/home/kaden/hipfire-gdnstack')
out = root / 'scratch-gdn' / 'quality' / arch / tag
out.mkdir(parents=True, exist_ok=True)
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
ref = '/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin'
visible = {'gfx1100': '0', 'gfx1151': '1'}[arch]
env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
env.update(HOME=str(out/'home'), XDG_CONFIG_HOME=str(out/'home'/'.config'),
           ROCR_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES='0',
           HIPFIRE_GRAPH='0', HIPFIRE_NORMALIZE_PROMPT='0')
if tag == 'baseline':
    env.update(HIPFIRE_GDN_PREP_GFX11='0', HIPFIRE_GDN_KKT_GFX1100='0')
Path(env['HOME']).mkdir(parents=True, exist_ok=True)
command = [str(root/'target/release/examples/eval_hipfire'), '--model', model,
           '--ref', ref, '--kv-mode', 'q8', '--kv-v', 'q8',
           '--scoring-mode', 'prefill', '--max-chunks', '24', '--output', str(out/'score.kldseq')]
proc = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True)
(out/'eval.log').write_text(proc.stdout)
(out/'config.txt').write_text('\n'.join(f'{k}={v}' for k, v in sorted(env.items()) if k.startswith(('HIPFIRE_', 'ROCR_', 'HIP_'))) + '\n')
if proc.returncode:
    raise RuntimeError(f'{tag} eval exit {proc.returncode}: {proc.stdout[-3000:]}')
if f'GPU dev 0: {arch}' not in proc.stdout:
    raise RuntimeError(f'{tag} device assertion failed')
m = re.search(r'slice-mean KLD = ([0-9.]+)', proc.stdout)
if not m:
    raise RuntimeError(f'{tag} missing WT2 slice mean: {proc.stdout[-3000:]}')
score = float(m.group(1))
reference = {'gfx1100': 0.076879, 'gfx1151': 0.076901}[arch]
digest = hashlib.sha256((out/'score.kldseq').read_bytes()).hexdigest()
print(f'{arch} {tag} WT2 c24 score={score:.6f} baseline={reference:.6f} delta={score-reference:+.6f} sha256={digest}', flush=True)
if tag == 'default':
    pinned = root / 'scratch-gdn' / 'quality' / arch / 'baseline' / 'score.kldseq'
    if not pinned.exists() or pinned.read_bytes() != (out/'score.kldseq').read_bytes():
        raise RuntimeError(f'{tag} exact WT2 sequence differs from stack baseline')
    print(f'{arch} default byte-identical to stack baseline', flush=True)
