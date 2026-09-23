#!/usr/bin/env python3
"""Warm one uncached fixture, then timestamp one traced pp8192 request."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

arch, arm = sys.argv[1:3]
root = Path('/home/kaden/hipfire-bafold/scratch-bafold') / arch / 'trace' / arm
root.mkdir(parents=True, exist_ok=True)
port = {'gfx1100': 11941, 'gfx1151': 11942}[arch]
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
fixture = Path('/home/kaden/hipfire-prof040/benchmarks/prompts/pp8192.txt')
prompt = fixture.read_text()
(root / 'prompt.md5').write_text(f'{hashlib.md5(fixture.read_bytes()).hexdigest()}  {fixture}\n')
if os.environ.get('ROCR_VISIBLE_DEVICES') != {'gfx1100': '0', 'gfx1151': '1'}[arch] or os.environ.get('HIP_VISIBLE_DEVICES') != '0':
    raise RuntimeError('physical/logical GPU visibility mismatch')

def request(label, content):
    body = {'model': model, 'messages': [{'role': 'user', 'content': content}],
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
    usage = next((chunk['usage'] for chunk in reversed(chunks) if chunk.get('usage')), None)
    result = {'start_monotonic_ns': start, 'end_monotonic_ns': end,
              'wall_ms': (end - start) / 1e6, 'usage': usage}
    (root / f'{label}-response.json').write_text(json.dumps(result, indent=2) + '\n')
    if not usage or usage.get('prompt_tokens') != 8192 or usage.get('completion_tokens') != 1:
        raise RuntimeError(f'{label}: wrong token count: {usage}')
    if usage.get('prompt_tokens_details', {}).get('cached_tokens') != 0:
        raise RuntimeError(f'{label}: cached tokens: {usage}')
    print(f'{arch} {label}: wall_ms={result["wall_ms"]:.2f}', flush=True)

with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
    response.read()
request('warm', 'A swift orange fox' + prompt[len('The quick brown fox'):])
time.sleep(1)
request('trace', prompt)
