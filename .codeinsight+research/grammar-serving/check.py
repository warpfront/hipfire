#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# Throwaway smoke verifier for baseline constrained Qwen3.8 MQ4XT grammar serve.
# Own only this path under .codeinsight+research/grammar-serving/.

"""HTTP smoke check: tools bash+read, force a bash `echo Hello` tool call.

Usage:
  python3 check.py --base-url http://127.0.0.1:11520 --model /path/to/qwen3.8-27b.mq4-xt
  python3 check.py http://127.0.0.1:11520 /path/to/qwen3.8-27b.mq4-xt

Exit 0 on pass, nonzero on fail. Always writes full response JSON to --out
(default: sibling response.json) on both success and failure when a body was
received.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
]

USER_CONTENT = "Use the bash tool to run: echo Hello"

DEFAULT_BODY = {
    "messages": [{"role": "user", "content": USER_CONTENT}],
    "tools": TOOLS,
    "stream": False,
    "max_tokens": 256,
    "temperature": 0,
    "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-check constrained grammar serve tool_calls path."
    )
    p.add_argument(
        "positional",
        nargs="*",
        help="optional: BASE_URL MODEL (alternative to --base-url/--model)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="serve base URL, e.g. http://127.0.0.1:11520",
    )
    p.add_argument(
        "--model",
        default=None,
        help="model path or tag sent as request body model field",
    )
    p.add_argument(
        "--out",
        default=None,
        help="path to write full response JSON (default: ./response.json beside this script)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout seconds (default 180)",
    )
    args = p.parse_args(argv)

    pos = list(args.positional or [])
    if args.base_url is None and pos:
        args.base_url = pos.pop(0)
    if args.model is None and pos:
        args.model = pos.pop(0)
    if pos:
        p.error(f"unexpected positional args: {pos}")
    if not args.base_url:
        p.error("--base-url (or positional BASE_URL) is required")
    if not args.model:
        p.error("--model (or positional MODEL) is required")
    if args.out is None:
        args.out = str(Path(__file__).resolve().parent / "response.json")
    return args


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fail(msg: str, *, out_path: str, retained: Any, code: int = 1) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    try:
        _write_json(out_path, retained)
        print(f"retained response JSON -> {out_path}", file=sys.stderr)
    except OSError as exc:
        print(f"WARN: could not write retained JSON to {out_path}: {exc}", file=sys.stderr)
    return code


def _extract_choice(resp: dict[str, Any]) -> dict[str, Any] | None:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    ch0 = choices[0]
    return ch0 if isinstance(ch0, dict) else None


def _validate(resp: dict[str, Any]) -> list[str]:
    """Return list of failure reasons (empty = pass)."""
    errors: list[str] = []
    ch = _extract_choice(resp)
    if ch is None:
        return ["missing choices[0]"]

    fr = ch.get("finish_reason")
    if fr != "tool_calls":
        errors.append(f"finish_reason={fr!r} expected 'tool_calls'")

    msg = ch.get("message")
    if not isinstance(msg, dict):
        errors.append("choices[0].message missing or not object")
        return errors

    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or len(tcs) < 1:
        errors.append(f"tool_calls empty or missing (got {tcs!r})")
        return errors

    tc0 = tcs[0]
    if not isinstance(tc0, dict):
        errors.append("tool_calls[0] not object")
        return errors

    fn = tc0.get("function")
    if not isinstance(fn, dict):
        errors.append("tool_calls[0].function missing")
        return errors

    name = fn.get("name")
    if name != "bash":
        errors.append(f"tool name={name!r} expected 'bash'")

    raw_args = fn.get("arguments")
    if not isinstance(raw_args, str):
        errors.append(f"arguments not a string: {type(raw_args).__name__}")
        return errors

    try:
        args_obj = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        errors.append(f"arguments not valid JSON: {raw_args!r} ({exc})")
        return errors

    if not isinstance(args_obj, dict):
        errors.append(f"arguments JSON not object: {args_obj!r}")
        return errors

    cmd = args_obj.get("command")
    if not isinstance(cmd, str):
        errors.append(f"arguments.command missing/not string: {cmd!r}")
        return errors

    # Accept exact echo Hello or trivial whitespace/quote variants that still
    # run the same command intent (model sometimes adds quotes).
    normalized = " ".join(cmd.strip().split())
    ok_cmds = {
        "echo Hello",
        "echo 'Hello'",
        'echo "Hello"',
        "echo Hello;",
    }
    if normalized not in ok_cmds and not (
        normalized.startswith("echo ") and "Hello" in normalized and "&&" not in normalized
    ):
        errors.append(f"arguments.command={cmd!r} does not look like echo Hello")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    url = _endpoint(args.base_url)
    body = dict(DEFAULT_BODY)
    body["model"] = args.model
    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    retained: Any = {
        "request": {"url": url, "body": body},
        "error": None,
        "response": None,
        "http_status": None,
    }

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read()
            retained["http_status"] = getattr(resp, "status", None) or resp.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        retained["http_status"] = exc.code
        retained["error"] = f"HTTPError {exc.code}: {exc.reason}"
        text = raw.decode("utf-8", "replace")
        try:
            retained["response"] = json.loads(text) if text else None
        except json.JSONDecodeError:
            retained["response"] = {"_raw": text}
        return _fail(retained["error"], out_path=args.out, retained=retained)
    except urllib.error.URLError as exc:
        retained["error"] = f"URLError: {exc.reason}"
        return _fail(retained["error"], out_path=args.out, retained=retained, code=2)
    except TimeoutError as exc:
        retained["error"] = f"timeout: {exc}"
        return _fail(retained["error"], out_path=args.out, retained=retained, code=2)

    text = raw.decode("utf-8", "replace")
    try:
        resp_json = json.loads(text)
    except json.JSONDecodeError as exc:
        retained["response"] = {"_raw": text}
        retained["error"] = f"response not JSON: {exc}"
        return _fail(retained["error"], out_path=args.out, retained=retained)

    retained["response"] = resp_json
    errors = _validate(resp_json)
    if errors:
        retained["error"] = "; ".join(errors)
        return _fail(retained["error"], out_path=args.out, retained=retained)

    _write_json(args.out, retained)
    ch = _extract_choice(resp_json) or {}
    msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
    tcs = msg.get("tool_calls") if isinstance(msg, dict) else []
    tc0 = tcs[0] if isinstance(tcs, list) and tcs else {}
    fn = tc0.get("function") if isinstance(tc0, dict) else {}
    print(
        "PASS: finish_reason=tool_calls "
        f"name={fn.get('name')!r} args={fn.get('arguments')!r}"
    )
    print(f"retained response JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
