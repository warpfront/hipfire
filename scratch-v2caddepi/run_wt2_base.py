#!/usr/bin/env python3
"""WT2 c24 Q8/Q8 of the stack port, lever off (A: HIPFIRE_V2C_ADDEPI=0) vs on
(B: default) in the SAME binaries; the lever is exact, so the two kldseq files
must be byte-identical. usage: run_wt2.py <arch>"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

arch = sys.argv[1]
ordinal = {'gfx1100': '0', 'gfx1151': '1'}[arch]
root = Path('/home/kaden/hipfire-v2caddepi')
out = root / "scratch-v2caddepi" / arch / "wt2-base"
out.mkdir(parents=True, exist_ok=True)
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
ref = '/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin'
scores = {}
files = {}
for arm in ('A', 'B'):
    home = out / f'{arm}-home'
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith('HIPFIRE_')}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=ordinal, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_GRAPH='0', HIPFIRE_NORMALIZE_PROMPT='0')
    if arm == 'A':
        env['HIPFIRE_V2C_ADDEPI'] = '0'
    output = out / f'{arm}.kldseq'
    cmd = [str(root / "target-base/eval_hipfire"), '--model', model,
           '--ref', ref, '--kv-mode', 'q8', '--kv-v', 'q8', '--scoring-mode', 'prefill',
           '--max-chunks', '24', '--output', str(output)]
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        expected_rocr = f'ROCR_VISIBLE_DEVICES={ordinal}'.encode()
        for _ in range(100):
            process_env = Path(f'/proc/{proc.pid}/environ').read_bytes().split(b'\0')
            if expected_rocr in process_env and b'HIP_VISIBLE_DEVICES=0' in process_env:
                break
            time.sleep(0.01)  # Popen returns before the child has exec'd.
        else:
            raise RuntimeError(f'{arch} WT2 {arm}: process visibility not attested')
        (out / f'{arm}.env').write_text(f'pid={proc.pid}\n' +
            '\n'.join(v.decode(errors='replace') for v in process_env if v) + '\n')
        log, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    (out / f'{arm}.log').write_text(log)
    if proc.returncode:
        raise RuntimeError(f'{arch} WT2 {arm} exited {proc.returncode}')
    if f'GPU dev 0: {arch}' not in log:
        raise RuntimeError(f'{arch} WT2 {arm}: wrong arch')
    match = re.search(r'slice-mean KLD = ([0-9.]+)', log)
    if not match:
        raise RuntimeError(f'{arch} WT2 {arm}: no slice mean')
    scores[arm] = float(match.group(1))
    files[arm] = output
    print(f'{arch} WT2 {arm}: {scores[arm]:.6f} sha256 '
          f'{hashlib.sha256(output.read_bytes()).hexdigest()}', flush=True)
same = files['A'].read_bytes() == files['B'].read_bytes()
print(f'{arch} WT2 c24 Q8/Q8 off {scores["A"]:.6f} on {scores["B"]:.6f} '
      f'{"PASS: byte-identical" if same else "FAIL: kldseq differs"}', flush=True)
if not same:
    sys.exit(1)
