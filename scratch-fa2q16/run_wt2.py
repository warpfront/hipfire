import os
from pathlib import Path
import re
import subprocess
import sys

root = Path('/home/kaden/hipfire-fa2q16')
base = Path('/home/kaden/hipfire-land040')
out = root / 'scratch-fa2q16'
arch = sys.argv[1]
assert arch in ('gfx1100', 'gfx1151')
ordinal = {'gfx1100': '0', 'gfx1151': '1'}[arch]
home = out / f'home-default-{arch}'
env = dict(os.environ, HOME=str(home), ROCR_VISIBLE_DEVICES=ordinal,
           HIP_VISIBLE_DEVICES='0', HIPFIRE_GRAPH='0', HIPFIRE_NORMALIZE_PROMPT='0')
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
ref = '/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin'
scores = {}
print('WT2_START', arch, flush=True)
for arm, tree in [('A', base), ('B', root)]:
    output = out / f'{arch}-wt2-{arm}.kldseq'
    cmd = [str(tree / 'target/release/examples/eval_hipfire'), '--model', model,
           '--ref', ref, '--kv-mode', 'q8', '--kv-v', 'q8', '--scoring-mode', 'prefill',
           '--max-chunks', '24', '--output', str(output)]
    run = subprocess.run(cmd, cwd=tree, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / f'{arch}-wt2-{arm}.log').write_text(run.stdout)
    print(arch, arm, 'exit', run.returncode, flush=True)
    if run.returncode: raise RuntimeError(f'WT2 {arm} failed: {run.stdout[-1000:]}')
    result = re.search(r'slice-mean KLD = ([0-9.]+)', run.stdout)
    if not result: raise RuntimeError(f'WT2 {arm}: no KLD')
    scores[arm] = float(result.group(1))
    print(arch, arm, 'KLD', scores[arm], flush=True)
difference = abs(scores['A'] - scores['B'])
print(arch, 'WT2 difference', difference, flush=True)
if scores['B'] > 0.10 or difference > 0.0005:
    raise RuntimeError(f'{arch}: WT2 quality gate failed')
