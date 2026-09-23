#!/usr/bin/env python3
"""Profile one uncached pp8192 request under the fold service binary."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

arch, arm = sys.argv[1:3]
root = Path('/home/kaden/hipfire-bafold')
out = root / 'scratch-bafold' / arch / 'trace' / arm
out.mkdir(parents=True, exist_ok=True)
home = out / 'home'
home.mkdir(exist_ok=True)
visible = {'gfx1100': '0', 'gfx1151': '1'}[arch]
port = {'gfx1100': 11941, 'gfx1151': 11942}[arch]
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
baseline = Path('/home/kaden/hipfire-v2bhalo' if arch == 'gfx1151' else '/home/kaden/hipfire-bafold-base')
bin_dir = (baseline if arm == 'A' else root) / 'target/release'
env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
           ROCR_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES='0',
           HIPFIRE_DAEMON_BIN='/home/kaden/hipfire-gfx11a4/scratch-gfx11a4/rocprof-a4-wrap.sh',
           HIPFIRE_ROCPROF_DAEMON_TARGET=str(bin_dir / 'daemon'),
           HIPFIRE_ROCPROF_OUTPUT_DIR=str(out))
for old in out.glob('daemon*.csv'):
    old.unlink()
for old in out.glob('*-response.json'):
    old.unlink()
subprocess.run(['rocm-smi', '--showpids'], env=env, stdout=(out / 'gpu-pre.txt').open('w'), check=True)
with (out / 'serve.log').open('w') as log:
    server = subprocess.Popen([str(bin_dir / 'hipfire'), 'serve', '--model', model,
                               f'127.0.0.1:{port}'], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError(f'service exited {server.returncode}: {out / "serve.log"}')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError(f'service not ready: {out / "serve.log"}')
        serve_log = (out / 'serve.log').read_text()
        if f'GPU dev 0: {arch}' not in serve_log or 'KV cache: Q8 vmm (' not in serve_log:
            raise RuntimeError(f'arch or KV mismatch: {out / "serve.log"}')
        subprocess.run(['python3', str(root / 'scratch-bafold/trace_request.py'), arch, arm],
                       env=env, check=True)
    finally:
        stopped = subprocess.run([str(bin_dir / 'hipfire'), 'stop'], env=env,
                                 capture_output=True, text=True, timeout=60)
        (out / 'stop.log').write_text(stopped.stdout + stopped.stderr)
        if stopped.returncode:
            print(f'hipfire stop returned {stopped.returncode}; see {out / "stop.log"}', file=sys.stderr)
        server.wait(timeout=60)
        # The CLI stops the server but can leave its rocprof-wrapped daemon.
        # Signal only our exact child binary/isolated HOME so its CSV flushes.
        pid_file = home / '.hipfire/daemon.pid'
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            proc = Path('/proc') / str(pid)
            if proc.exists():
                exe = os.readlink(proc / 'exe')
                daemon_env = (proc / 'environ').read_bytes()
                if exe != str(bin_dir / 'daemon') or f'HOME={home}'.encode() not in daemon_env.split(b'\0'):
                    raise RuntimeError(f'not signalling unverified daemon {pid}: {exe}')
                os.kill(pid, signal.SIGTERM)
                for _ in range(60):
                    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                        break
                    time.sleep(.5)
                else:
                    os.kill(pid, signal.SIGKILL)
                    raise RuntimeError(f'daemon {pid} did not stop after SIGTERM; aborted trace')
        subprocess.run(['rocm-smi', '--showpids'], env=env,
                       stdout=(out / 'gpu-post.txt').open('w'), check=True)
if not (out / 'daemon_kernel_trace.csv').exists():
    raise RuntimeError(f'profiler CSV missing: {out}')
subprocess.run(['python3', str(root / 'scratch-bafold/summarize_trace.py'), arch, arm], check=True)
