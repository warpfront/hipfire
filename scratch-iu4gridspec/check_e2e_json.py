#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) not in (3, 4):
    raise SystemExit(f"usage: {sys.argv[0]} FILE EXPECT_ARCH [EXPECT_PROMPT_TOKENS]")
path = Path(sys.argv[1])
expected_arch = sys.argv[2]
data = json.loads(path.read_text())

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)

if not any(expected_arch in value for value in strings(data)):
    raise SystemExit(f"ARCH_ASSERT expected={expected_arch} json={path}")
if len(sys.argv) == 4:
    expected_tokens = int(sys.argv[3])
    actual_tokens = data.get("prompt_tokens")
    if actual_tokens != expected_tokens:
        raise SystemExit(
            f"PROMPT_TOKENS_ASSERT expected={expected_tokens} actual={actual_tokens} json={path}"
        )
print(f"validated {path.name}: arch={expected_arch}" +
      (f" prompt_tokens={data['prompt_tokens']}" if len(sys.argv) == 4 else ""))
