#!/usr/bin/env python3
"""Manual knob-free server on one gfx11 card, then the 5-prompt battery
(scripts/serve_harness.py --no-spawn) and the 511-token partial-N reply."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

arch = sys.argv[1]
visible = {'gfx1100': '0', 'gfx1151': '1'}[arch]
port = {'gfx1100': 11977, 'gfx1151': 11978}[arch]
root = Path('/home/kaden/hipfire-v2bhalo')
out = root / 'scratch-v2bhalo' / arch
out.mkdir(parents=True, exist_ok=True)
home = out / 'serve-home'
home.mkdir(exist_ok=True)
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
           ROCR_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES='0',
           HIPFIRE_DAEMON_BIN=str(root / 'target/release/daemon'))


def proc_env(pid):
    items = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    return {i.split(b'=', 1)[0].decode(): i.split(b'=', 1)[1].decode() for i in items if b'=' in i}


log_path = out / 'serve.log'
with log_path.open('w') as log:
    server = subprocess.Popen([str(root / 'target/release/hipfire'), 'serve', '--model', model,
                               f'127.0.0.1:{port}'], cwd=root, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('serve exited; see serve.log')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('serve not ready')
        text = log_path.read_text()
        if f'GPU dev 0: {arch}' not in text or 'KV cache: Q8 vmm (' not in text:
            raise RuntimeError('wrong device or KV backend in serve.log')
        daemons = subprocess.run(['pgrep', '-P', str(server.pid)], text=True,
                                 stdout=subprocess.PIPE).stdout.split()
        with (out / 'serve-env.txt').open('w') as f:
            for pid in [str(server.pid), *daemons]:
                e = proc_env(pid)
                f.write(f'pid={pid} ROCR_VISIBLE_DEVICES={e.get("ROCR_VISIBLE_DEVICES")} '
                        f'HIP_VISIBLE_DEVICES={e.get("HIP_VISIBLE_DEVICES")} HOME={e.get("HOME")}\n')
                assert e.get('ROCR_VISIBLE_DEVICES') == visible and e.get('HIP_VISIBLE_DEVICES') == '0'
        subprocess.run([sys.executable, str(root / 'scripts/serve_harness.py'), '--no-spawn',
                        '--port', str(port), '--model', model, '--mode', 'battery',
                        '--max-think-tokens', '1', '--max-tokens', '512',
                        '--out', str(out / 'battery.json')],
                       cwd=root, env=env, check=True,
                       stdout=(out / 'battery.log').open('w'), stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, str(root / 'scratch-v2bhalo/odd_fill.py'), arch,
                        str(port)], cwd=root, check=True)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            server.wait(timeout=60)
