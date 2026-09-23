import json
import os
from pathlib import Path
import subprocess
import sys
import time

# land-041 composite gate: one fresh-process candidate matrix run per card vs
# one same-day landed arm. Asserts `KV cache: Q8 vmm (` + arch line per process
# from /proc env + log, no row >1% below landed, and pp8192 at/above the FA2
# branch's recorded B values within noise.
root = Path('/home/kaden/hipfire-land041')
base = Path('/home/kaden/hipfire-prof040')
out = root / 'scratch-land041'
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
arch = sys.argv[1]
assert arch in ('gfx1100', 'gfx1151')
ordinal = {'gfx1100': '0', 'gfx1151': '1'}[arch]
# FA2 branch recorded candidate (B) means, knob-free defaults, same protocol.
fa2_b = {'gfx1100': {'pp512': 2130.35, 'pp8192': 2224.30},
         'gfx1151': {'pp512': 797.25, 'pp8192': 751.45}}[arch]
home = out / f'home-matrix-{arch}'
home.mkdir(exist_ok=True)
env = dict(os.environ, HOME=str(home), ROCR_VISIBLE_DEVICES=ordinal,
           HIP_VISIBLE_DEVICES='0', HIPFIRE_GRAPH='1')
command = ['bench', model, '--matrix', '--pp', '512,8192', '--ctx', '128', '--tg', '128',
           '--spec', 'off', '--runs', '3', '--warmups', '1', '--kv-mode', 'q8',
           '--kv-backend', 'vmm', '--json']
order = [('A1', base), ('B1', root)]
results = {}
print('MATRIX_START', arch, 'A=e5f944a3a B=land-041, built-in defaults', flush=True)
for label, worktree in order:
    exe = worktree / 'target/release/hipfire'
    stem = out / f'{arch}-matrix-{label}'
    (stem.with_suffix('.pre-pids')).write_text(subprocess.run(
        ['rocm-smi', '--showpids'], text=True, capture_output=True, check=True).stdout)
    run = subprocess.Popen([str(exe), *command], cwd=worktree, env=env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(0.1)
    process_env = Path(f'/proc/{run.pid}/environ').read_bytes().replace(b'\0', b'\n').decode()
    (stem.with_suffix('.env')).write_text(f'pid={run.pid}\n' + process_env)
    assert f'ROCR_VISIBLE_DEVICES={ordinal}\n' in process_env
    assert 'HIP_VISIBLE_DEVICES=0\n' in process_env
    log, _ = run.communicate()
    stem.with_suffix('.log').write_text(log)
    (stem.with_suffix('.post-pids')).write_text(subprocess.run(
        ['rocm-smi', '--showpids'], text=True, capture_output=True, check=True).stdout)
    print(arch, label, 'exit', run.returncode, flush=True)
    if run.returncode:
        raise RuntimeError(f'{label} failed: {log[-2500:]}')
    if f'GPU dev 0: {arch}' not in log or 'KV cache: Q8 vmm (' not in log:
        raise RuntimeError(f'{label}: GPU arch or VMM attestation missing')
    marker = '\n{\n  "protocol": "synthetic-pp-tg-matrix-v1"'
    index = log.rfind(marker)
    if index < 0:
        raise RuntimeError(f'{label}: matrix JSON missing')
    data = json.loads(log[index + 1:])
    stem.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
    values = {f'pp{x["tokens"]}': x['stats']['median'] for x in data['prefill']}
    values.update({f'tg{x["tokens"]}@{x["context"]}': x['stats']['median'] for x in data['decode']})
    results[label] = values
    print(arch, label, values, flush=True)
for metric in ('pp512', 'pp8192', 'tg128@128'):
    pct = (results['B1'][metric] / results['A1'][metric] - 1) * 100
    print(arch, metric, 'A1', results['A1'][metric], 'B1', results['B1'][metric],
          'gain_pct', pct, flush=True)
    if pct < -1:
        raise RuntimeError(f'{metric}: greater than 1% below landed')
for metric in ('pp512', 'pp8192'):
    rel = results['B1'][metric] / fa2_b[metric] - 1
    print(arch, metric, 'B1 vs FA2-B', rel * 100, flush=True)
    if rel < -0.02:
        raise RuntimeError(f'{metric}: more than 2% below FA2 branch value')
print(arch, 'MATRIX GATE PASS', flush=True)
