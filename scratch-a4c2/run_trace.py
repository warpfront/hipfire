#!/usr/bin/env python3
"""rocprof one warmed, uncached pp8192 request on one local gfx1201 card.

Usage: run_trace.py <label> <bin-dir> <card> [KEY=VALUE ...]
Serves the local model with the fp8 VMM default (explicit --kv-mode fp8, as
the guard does for a local path), warms with a different-prefix 8192-token
request, then traces the pp8192 fixture. Stops with `hipfire stop` and, if
needed, SIGTERMs the verified wrapped daemon so rocprof flushes. Writes
scratch-a4c2/trace/<label>/{summary.json,symbols.tsv,...}.
"""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

CARDS = {'A': 'GPU-9eb7aeda51c88ffd', 'C': 'GPU-085289909a86cc63',
         'D': 'GPU-6109a4cb5f833235', 'E': 'GPU-05f92432f2312a0e'}
PORTS = {'A': 11961, 'C': 11962, 'D': 11963, 'E': 11964}
ROOT = Path(__file__).resolve().parent
MODEL = '/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
FIXTURE = Path('/home/kaden/ClaudeCode/warpfront/wt-g12prof/scratch-g12prof/pp8192.txt')
WRAP = Path('/home/kaden/ClaudeCode/warpfront/wt-g12prof/scratch-g12prof/rocprof-wrap.sh')

label, bindir, card = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
extra = dict(kv.split('=', 1) for kv in sys.argv[4:])
out = ROOT / 'trace' / label
out.mkdir(parents=True, exist_ok=True)
for old in list(out.glob('daemon*.csv')) + list(out.glob('*-response.json')):
    old.unlink()
home = Path('/home/kaden/.hipfire-homes/a4c2')
home.mkdir(exist_ok=True)
(home / '.hipfire' / 'daemon.pid').unlink(missing_ok=True)
port = PORTS[card]
env = {k: v for k, v in os.environ.items() if not k.startswith('HIPFIRE_')}
env.update(HOME=str(home), ROCR_VISIBLE_DEVICES=CARDS[card], HIP_VISIBLE_DEVICES=CARDS[card],
           HIPFIRE_KERNEL_CACHE=str(home / '.hipfire_kernels'),
           HIPFIRE_MODELS_DIR='/home/kaden/.hipfire/models', HIPFIRE_GRAPH='1',
           HIPFIRE_DAEMON_BIN=str(WRAP), HIPFIRE_ROCPROF_DAEMON_TARGET=str(bindir / 'daemon'),
           HIPFIRE_ROCPROF_OUTPUT_DIR=str(out))
env.update(extra)
(out / 'env.txt').write_text('\n'.join(f'{k}={v}' for k, v in sorted(env.items())
                                       if k.startswith(('HIPFIRE_', 'ROCR', 'HIP_', 'HOME'))) + '\n')
prompt = FIXTURE.read_text()
(out / 'prompt.md5').write_text(hashlib.md5(FIXTURE.read_bytes()).hexdigest() + '\n')


def request(tag, content):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': 1, 'temperature': 0, 'top_p': 1, 'top_k': 1,
            'reasoning_effort': 'none', 'max_think_tokens': 1,
            'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 json.dumps(body).encode(), {'Content-Type': 'application/json'})
    start = time.monotonic_ns()
    chunks = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                chunks.append(json.loads(line[6:]))
    end = time.monotonic_ns()
    usage = next((c['usage'] for c in reversed(chunks) if c.get('usage')), None)
    timings = next((c['timings'] for c in reversed(chunks) if c.get('timings')), None)
    text = ''.join(ch.get('delta', {}).get('content') or '' for c in chunks for ch in c.get('choices', []))
    res = dict(start_monotonic_ns=start, end_monotonic_ns=end, wall_ms=(end - start) / 1e6,
               usage=usage, timings=timings, text=text)
    (out / f'{tag}-response.json').write_text(json.dumps(res, indent=2) + '\n')
    if not usage or usage.get('prompt_tokens') != 8192 or usage.get('prompt_tokens_details', {}).get('cached_tokens') != 0:
        raise RuntimeError(f'{tag}: bad usage {usage}')
    return res


subprocess.run(['rocm-smi', '--showpids'], stdout=(out / 'gpu-pre.txt').open('w'), check=True)
with (out / 'serve.log').open('w') as log:
    server = subprocess.Popen([str(bindir / 'hipfire'), 'serve', '--model', MODEL, '--kv-mode', 'fp8',
                               '--kv-backend', 'vmm', f'127.0.0.1:{port}'],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError(f'serve exited {server.returncode}')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError('serve not ready')
        request('warm', 'A swift orange fox' + prompt[len('The quick brown fox'):])
        time.sleep(1)
        trace = request('trace', prompt)
    finally:
        stop = subprocess.run([str(bindir / 'hipfire'), 'stop'], env=env, capture_output=True, text=True, timeout=120)
        (out / 'stop.log').write_text(stop.stdout + stop.stderr)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=30)
        pid_file = home / '.hipfire' / 'daemon.pid'
        if pid_file.exists():
            pid = int(pid_file.read_text().strip() or 0)
            proc = Path('/proc') / str(pid)
            if pid and proc.exists() and f'HOME={home}'.encode() in (proc / 'environ').read_bytes().split(b'\0'):
                os.kill(pid, signal.SIGTERM)
                for _ in range(120):
                    if not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z':
                        break
                    time.sleep(0.5)
            pid_file.unlink(missing_ok=True)
        subprocess.run(['rocm-smi', '--showpids'], stdout=(out / 'gpu-post.txt').open('w'), check=True)

log = (out / 'serve.log').read_text(errors='replace')
if 'gfx1201' not in log or 'KV cache: Fp8 vmm (' not in log:
    raise RuntimeError('gfx1201 / fp8 VMM not asserted in serve.log')
for _ in range(60):
    if (out / 'daemon_kernel_trace.csv').exists():
        break
    time.sleep(1)
rows = list(csv.DictReader((out / 'daemon_kernel_trace.csv').open()))
t0, t1 = trace['start_monotonic_ns'], trace['end_monotonic_ns']
sel = [r for r in rows if int(r['Start_Timestamp']) >= t0 and int(r['End_Timestamp']) <= t1]
agg = collections.defaultdict(lambda: [0, 0.0])
for r in sel:
    a = agg[r['Kernel_Name']]
    a[0] += 1
    a[1] += (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e6
with (out / 'symbols.tsv').open('w') as f:
    for name, (calls, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        f.write(f'{name}\t{calls}\t{ms:.3f}\n')
summary = dict(label=label, card=card, bindir=str(bindir), extra_env=extra,
               daemon_sha256=hashlib.sha256((bindir / 'daemon').read_bytes()).hexdigest(),
               dispatches=len(sel), gpu_ms=round(sum(v[1] for v in agg.values()), 3),
               wall_ms=trace['wall_ms'], timings=trace['timings'], text=trace['text'])
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('label', 'dispatches', 'gpu_ms', 'wall_ms')}),
      'prefill_tok_s', (trace['timings'] or {}).get('prefill_tok_s'))
