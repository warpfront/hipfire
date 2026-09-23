#!/usr/bin/env python3
"""Profile one independently warmed, uncached prefill (AddEpilogue).

usage: run_profile.py <arch> <length> <A|B>; A = HIPFIRE_V2B_ADDEPI=0, B = default."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/kaden/hipfire-addepi-stack')
OUT = ROOT / 'scratch-addepi' / 'trace'
MODEL = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
WRAPPER = '/home/kaden/hipfire-gfx11a4/scratch-gfx11a4/rocprof-a4-wrap.sh'


def run(arch, length, arm):
    dest = OUT / f'{arch}-{length}-{arm}'
    dest.mkdir(parents=True, exist_ok=True)
    home = dest / 'home'
    home.mkdir(exist_ok=True)
    port = {'gfx1151': 11942, 'gfx1100': 11941}[arch]
    visible = {'gfx1151': '1', 'gfx1100': '0'}[arch]
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / '.config'),
               ROCR_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES='0',
               HIPFIRE_DAEMON_BIN=WRAPPER,
               HIPFIRE_ROCPROF_DAEMON_TARGET=str(ROOT / 'target/release/daemon'),
               HIPFIRE_ROCPROF_OUTPUT_DIR=str(dest))
    # No HIPFIRE_HOME, KV, graph, or kernel tuning overrides.
    for name in list(env):
        if name.startswith('HIPFIRE_') and name not in (
            'HIPFIRE_DAEMON_BIN', 'HIPFIRE_ROCPROF_DAEMON_TARGET', 'HIPFIRE_ROCPROF_OUTPUT_DIR'):
            del env[name]
    if arm == 'A':
        env['HIPFIRE_V2B_ADDEPI'] = '0'
    prompt_path = Path('/home/kaden/hipfire-prof040/benchmarks/prompts') / f'pp{length}.txt'
    prompt = prompt_path.read_text()
    (dest / 'prompt.md5').write_text(f'{hashlib.md5(prompt_path.read_bytes()).hexdigest()}  {prompt_path}\n')
    def capture(cmd, name):
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (dest / name).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f'{name}: {result.returncode}: {result.stdout}')
    capture(['/home/kaden/hipfire-gfx11stack-profile/scratch-gfx11profile/assert_arch', arch], 'arch-assertion.txt')
    capture(['rocm-smi', '--showuse', '--showmemuse', '--showpids'], 'gpu-pre.txt')
    def request(label, content):
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': content}],
                'max_tokens': 1, 'temperature': 0, 'top_p': 1, 'top_k': 1,
                'reasoning_effort': 'none', 'max_think_tokens': 1,
                'stream': True, 'stream_options': {'include_usage': True}}
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json'})
        start = time.monotonic_ns()
        chunks = []
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                    chunks.append(json.loads(line[6:]))
        end = time.monotonic_ns()
        usage = next((c['usage'] for c in reversed(chunks) if c.get('usage')), None)
        result = {'start_monotonic_ns': start, 'end_monotonic_ns': end,
                  'wall_ms': (end-start)/1e6, 'usage': usage, 'chunks': chunks}
        (dest / f'{label}-response.json').write_text(json.dumps(result, indent=2) + '\n')
        if not usage or usage.get('prompt_tokens') != int(length) or usage.get('completion_tokens') != 1:
            raise RuntimeError(f'{label}: unexpected token counts: {usage}')
        if usage.get('prompt_tokens_details', {}).get('cached_tokens') != 0:
            raise RuntimeError(f'{label}: cached tokens: {usage}')
        return result
    server = None
    with (dest / 'serve.log').open('w') as log:
        try:
            server = subprocess.Popen([str(ROOT / 'target/release/hipfire'), 'serve',
                                       '--model', MODEL, f'127.0.0.1:{port}'],
                                      env=env, stdout=log, stderr=subprocess.STDOUT)
            ready = False
            for _ in range(240):
                if server.poll() is not None:
                    break
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1).read()
                    ready = True
                    break
                except Exception:
                    time.sleep(1)
            if not ready:
                raise RuntimeError('serve did not become ready; see serve.log')
            server_log = (dest / 'serve.log').read_text()
            if f'GPU dev 0: {arch}' not in server_log:
                raise RuntimeError('daemon arch mismatch')
            if 'KV cache: Q8 vmm (' not in server_log:
                raise RuntimeError('daemon KV backend mismatch')
            def process_env(pid):
                values = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
                return {v.split(b'=', 1)[0].decode(): v.split(b'=', 1)[1].decode()
                        for v in values if b'=' in v}
            with (dest / 'gpu-process-env.txt').open('w') as assertion:
                parent = process_env(server.pid)
                assertion.write(f'parent pid={server.pid} ROCR_VISIBLE_DEVICES={parent.get("ROCR_VISIBLE_DEVICES")} HIP_VISIBLE_DEVICES={parent.get("HIP_VISIBLE_DEVICES")} HOME={parent.get("HOME")}\n')
                assert parent.get('ROCR_VISIBLE_DEVICES') == visible and parent.get('HIP_VISIBLE_DEVICES') == '0'
                capture(['rocm-smi', '--showpids'], 'pids-ready.txt')
                pids = []
                for row in (dest / 'pids-ready.txt').read_text().splitlines():
                    fields = row.split()
                    if len(fields) > 1 and fields[0].isdigit() and fields[1] != 'gpusentry':
                        # Other agents' daemons on the other card are not ours.
                        try:
                            if process_env(int(fields[0])).get('HOME') == str(home):
                                pids.append(int(fields[0]))
                        except (FileNotFoundError, PermissionError):
                            pass
                for pid in pids:
                    child = process_env(pid)
                    assertion.write(f'daemon pid={pid} ROCR_VISIBLE_DEVICES={child.get("ROCR_VISIBLE_DEVICES")} HIP_VISIBLE_DEVICES={child.get("HIP_VISIBLE_DEVICES")}\n')
                    assert child.get('ROCR_VISIBLE_DEVICES') == visible
                    assert child.get('HIP_VISIBLE_DEVICES') == '0'
                if len(pids) != 1:
                    raise RuntimeError(f'expected one GPU daemon, got {pids}')
            warm = 'A swift orange fox' + prompt[len('The quick brown fox'):]
            request('warm', warm)
            time.sleep(1)
            result = request('trace', prompt)
            print(f'{arch} {length}: prompt={result["usage"]["prompt_tokens"]} '
                  f'cached={result["usage"]["prompt_tokens_details"]} wall_ms={result["wall_ms"]:.3f}', flush=True)
        finally:
            if server and server.poll() is None:
                # Graceful daemon stop first so rocprof flushes its CSVs.
                subprocess.run([str(ROOT / 'target/release/hipfire'), 'stop', str(port)], env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=120)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.send_signal(signal.SIGTERM)
                    server.wait(timeout=40)
            time.sleep(2)
            capture(['rocm-smi', '--showuse', '--showmemuse', '--showpids'], 'gpu-post.txt')
    if not (dest / 'daemon_kernel_trace.csv').exists():
        raise RuntimeError('rocprof did not flush kernel trace')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])
