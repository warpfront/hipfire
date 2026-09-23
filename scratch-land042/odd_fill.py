#!/usr/bin/env python3
"""Exercise the 511-token partial-N prefill route against a manual server."""
import json
from pathlib import Path
import sys
import urllib.request

arch = sys.argv[1]
port = int(sys.argv[2])
model = '/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq'
prompt = Path('/home/kaden/hipfire-land042/benchmarks/prompts/ttft_511.txt').read_text()
body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 512, 'temperature': 0, 'top_p': 1, 'top_k': 1,
        'reasoning_effort': 'none', 'max_think_tokens': 1, 'stream': False}
request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
    json.dumps(body).encode(), {'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
out = Path('/home/kaden/hipfire-land042/scratch-land042') / arch / 'ttft511.json'
out.write_text(json.dumps(result, indent=2) + '\n')
choice = result.get('choices', [{}])[0]
content = choice.get('message', {}).get('content', '')
print(f'{arch} prompt_tokens={result.get("usage", {}).get("prompt_tokens")} '
      f'finish={choice.get("finish_reason")} content={content[:500]!r}')
if result.get('usage', {}).get('prompt_tokens') != 511 or choice.get('finish_reason') != 'stop' or not content.strip():
    raise RuntimeError(f'{arch}: incomplete TTFT-511 reply')
