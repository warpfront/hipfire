#!/usr/bin/env python3
"""Submit the canonical 511-token prompt to an existing serve process."""
import json
from pathlib import Path
import sys
import urllib.request
arch = sys.argv[1]
port = int(sys.argv[2])
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
prompt = Path('/home/kaden/hipfire-prof040/benchmarks/prompts/ttft_511.txt').read_text()
body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 512, 'temperature': 0, 'top_p': 1, 'top_k': 1,
        'reasoning_effort': 'none', 'max_think_tokens': 1, 'stream': False}
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                             json.dumps(body).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=600) as reply:
    result = json.load(reply)
out = Path('/home/kaden/hipfire-x5/scratch-x5') / f'{arch}-ttft511.json'
out.write_text(json.dumps(result, indent=2) + '\n')
usage = result.get('usage', {})
content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
print(f'{arch} prompt_tokens={usage.get("prompt_tokens")} finish={result.get("choices", [{}])[0].get("finish_reason")} content={content[:500]!r}')
if usage.get('prompt_tokens') != 511 or not content.strip():
    raise RuntimeError('odd-fill fallback produced missing or empty response')
