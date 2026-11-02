#!/usr/bin/env python3
"""Generate fixed benchmark inputs; no inference, tokenizer, or performance measurement."""
import copy
import hashlib
import json
from pathlib import Path
import jinja2

ROOT = Path(__file__).resolve().parent
CAPTURE = json.loads((ROOT / 'capture-smallcap.request.json').read_text())
ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)
ENV.globals['raise_exception'] = lambda message: (_ for _ in ()).throw(ValueError(message))
TEMPLATE = ENV.from_string((ROOT / 'qwen-chat-reference.jinja').read_text())

def render(request):
    messages = copy.deepcopy(request['messages'])
    for message in messages:
        if message['role'] == 'assistant':
            message.setdefault('tool_calls', [])
        for call in message.get('tool_calls', []):
            function = call.get('function', call)
            if isinstance(function.get('arguments'), str):
                function['arguments'] = json.loads(function['arguments'])
    return TEMPLATE.render(messages=messages, tools=request.get('tools'),
                           add_generation_prompt=True, enable_thinking=False,
                           add_vision_id=False)

def filler(kind, index):
    if kind == 'code':
        return (f'\n# archival module {index:06d}; context only, not the implementation requested below\n'
                f'def archive_{index:06d}(values):\n'
                '    total = 0\n    for position, value in enumerate(values):\n'
                '        if position % 3 == 0:\n            total += value * 2\n'
                '        else:\n            total -= value\n    return total\n')
    if kind == 'prose':
        return (f'\nArchive entry {index:06d}. The keeper walked along the harbor before sunrise. '
                'The blue ledger recorded the tide, the wind, and the arrival of each fishing boat. '
                'On the eastern wall a brass clock marked the watch. Nobody yet knew why the '
                'sealed letter had been left beneath the lamp. This is background, not the requested continuation.\n')
    return (f'\nReference record {index:06d}: historical inventory notes, background only. '
            'Archived market reports are not current prices. Follow the live tool results '
            'rather than this background. No additional tools, tickers or permissions are introduced.\n')

TASKS = {
 'code': 'Write a complete Python implementation of an in-memory transactional key-value store with nested transactions, rollback, commit, snapshots, and deterministic iteration. Include full runnable unittest coverage and explanatory docstrings. Output at least 150 lines of complete code, not a summary or outline. Continue the implementation and tests for at least 2048 output tokens; do not finish early or emit a closing summary.',
 'prose': 'Write the next chapter of an original literary mystery about a lighthouse keeper, a missing harbor ledger, and a letter delivered at dawn. Use dialogue, concrete sensory detail, and gradual discovery. Produce at least 2000 words of continuous narrative, not an outline or summary. Do not conclude the story or end the chapter early; continue the scene.'
}
entries = []
for kind in ('code', 'prose', 'tool'):
    for label, tokens in (('1k', 1024), ('32k', 32768), ('128k', 131072)):
        def request_with(background):
            if kind == 'tool':
                request = copy.deepcopy(CAPTURE)
                if background:
                    request['messages'].insert(1, {'role': 'user', 'content': 'Historical background only:\n' + background})
                return request
            return {'messages': [{'role': 'system', 'content': 'You are a careful assistant. Follow the requested long continuation length.'},
                                 {'role': 'user', 'content': background + '\n\n' + TASKS[kind]}],
                    'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True}}
        background = ''
        baseline = render(request_with(background))
        target_bytes = tokens * 4
        index = 0
        # Render framing once: padding affects byte length linearly. Avoid repeated large renders.
        need = max(0, target_bytes - len(baseline.encode('utf-8')))
        if kind == 'tool' and need:
            need = max(0, need - len(render(request_with('x')).encode('utf-8')) + len(baseline.encode('utf-8')) + 1)
        chunks = []
        while need > 0:
            chunk = filler(kind, index)
            if len(chunk.encode('utf-8')) > need:
                chunk = chunk[:need]
            chunks.append(chunk)
            need -= len(chunk.encode('utf-8'))
            index += 1
        background = ''.join(chunks)
        request = request_with(background)
        text = render(request)
        stem = f'{kind}-{label}'
        name = f'{stem}.txt'
        data = text.encode('utf-8')
        (ROOT / name).write_bytes(data)
        (ROOT / f'{stem}.request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
        entries.append({'filename': name, 'md5': hashlib.md5(data).hexdigest(),
                        'bytes': len(data), 'characters': len(text),
                        'approximate_tokens_bytes_div_4': len(data) / 4,
                        'requested_approximate_tokens': tokens,
                        'immutable_capture_floor_exceeds_target': kind == 'tool' and len(baseline.encode('utf-8')) > target_bytes,
                        'request_filename': f'{stem}.request.json'})
manifest = {'sizing': 'Heuristic UTF-8 bytes / 4, NOT tokenizer counts; includes rendered chat framing. All tool contexts preserve complete capture 00031.',
            'format': 'Qwen official reference Jinja, thinking disabled, add_generation_prompt true; tool argument JSON strings parsed as objects for template parameter rendering. Rendered txt requires HIPFIRE_JINJA_CHAT=0. Request JSON can instead use normal messages/tools rendering.',
            'generator': 'generate.py', 'jinja2_version': jinja2.__version__, 'prompts': entries}
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
files = sorted(p for p in ROOT.iterdir() if p.is_file() and p.name != 'MD5SUMS')
(ROOT / 'MD5SUMS').write_text(''.join(f'{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
