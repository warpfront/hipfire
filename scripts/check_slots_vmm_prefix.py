#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU check: cross-session prefix reuse on the VMM multi-slot route.

Two requests share a long system prompt and differ only in the final user
turn. The first request captures the 512-token prefill boundary in the VMM
host-snapshot prefix cache; the second must reuse it (`cached_tokens > 0`,
`vmm prefix cache HIT` in the trace) and produce exactly the same greedy text
as a cold server with the cache disabled. A third request with a different
system prompt must miss (`cached_tokens == 0`).

Run on a GPU host (or the runtime container):

  python3 scripts/check_slots_vmm_prefix.py --model <4b.mq4>
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(port: int, model: str, messages: list[dict], timeout: float) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 24,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def answer(reply: dict) -> str:
    return (((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


def cached_tokens(reply: dict) -> int:
    usage = reply.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


def health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - not ready yet
        return None


def run_server(
    binary: Path,
    daemon: Path,
    model: str,
    port: int,
    prefix_cache: bool,
    max_bytes: int,
    ctx: int,
    repo: Path,
    log: Path,
    deadline_s: float,
):
    log_handle = log.open("w")
    env = os.environ.copy()
    env.update(
        {
            "HIPFIRE_DAEMON_BIN": str(daemon),
            "HIPFIRE_SERVE_MULTI_SLOT": "1",
            "HIPFIRE_SERVE_MULTI_SLOT_SLOTS": "2",
            "HIPFIRE_SERVE_MULTI_SLOT_CTX": str(ctx),
            "HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK": "512",
            "HIPFIRE_SLOT_TRACE": "1",
            "HIPFIRE_SERVE_PREFIX_CACHE": "1" if prefix_cache else "0",
            "HIPFIRE_SERVE_PREFIX_CACHE_MAX_BYTES": str(max_bytes),
        }
    )
    proc = subprocess.Popen(
        [
            str(binary),
            "serve",
            "--model",
            model,
            "--kv-backend",
            "vmm",
            "--kv-mode",
            "q8",
            "--spec",
            "off",
            "--idle-timeout",
            "0",
            f"127.0.0.1:{port}",
        ],
        env=env,
        cwd=repo,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_handle.flush()
            raise RuntimeError(f"serve exited {proc.returncode}\n{log.read_text(errors='replace')[-3000:]}")
        if health(port):
            return proc, log_handle
        time.sleep(0.5)
    raise RuntimeError("serve did not become ready")


def stop(proc, log_handle) -> None:
    log_handle.close()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=11592)
    ap.add_argument("--cold-port", type=int, default=11593)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-cache-bytes", type=int, default=256 << 20)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("HIPFIRE_BIN", repo / "target/release/hipfire"))
    daemon = Path(os.environ.get("HIPFIRE_DAEMON_BIN", binary.parent / "daemon"))
    if not binary.is_file() or not daemon.is_file():
        print(f"FAIL: missing binary ({binary}) or daemon ({daemon})", file=sys.stderr)
        return 1

    system = (
        "You are a precise assistant. The following reference material is "
        "deliberately long so the prompt crosses one prefill chunk. "
        + (
            "Rule: answer the final user request with only the requested word. "
            "Background: rivers flow downhill; the sky is blue; lists are ordered; "
            "the prefix cache must preserve every token of this preamble. "
        )
        * 24
    )
    suffix_a = "Reply with the word: first"
    suffix_b = "Reply with the word: second"
    messages_a = [
        {"role": "system", "content": system},
        {"role": "user", "content": suffix_a},
    ]
    messages_b = [
        {"role": "system", "content": system},
        {"role": "user", "content": suffix_b},
    ]
    messages_c = [
        {"role": "system", "content": "Different preamble. " + system[20:]},
        {"role": "user", "content": suffix_b},
    ]

    failures: list[str] = []
    warm_text: str | None = None
    log_on = Path(tempfile.mkstemp(prefix="slots-vmm-prefix-on-", suffix=".log")[1])
    log_off = Path(tempfile.mkstemp(prefix="slots-vmm-prefix-off-", suffix=".log")[1])
    proc_on = proc_off = None
    held_on = held_off = None
    try:
        proc_on, held_on = run_server(
            binary,
            daemon,
            args.model,
            args.port,
            True,
            args.max_cache_bytes,
            args.ctx,
            repo,
            log_on,
            args.timeout,
        )
        # First request: cold, captures the 512-token boundary.
        first = post(args.port, args.model, messages_a, args.timeout)
        print(f"[check] first cold: {answer(first)!r} prompt={first.get('usage', {}).get('prompt_tokens')}")
        if cached_tokens(first) != 0:
            failures.append(f"first request unexpectedly reused {cached_tokens(first)} tokens")
        # Second request: same prefix, different suffix → hit.
        warm = post(args.port, args.model, messages_b, args.timeout)
        warm_text = answer(warm)
        warm_cached = cached_tokens(warm)
        print(f"[check] second (warm): {warm_text!r} cached={warm_cached}")
        if warm_cached <= 0:
            failures.append("second request did not reuse the cached prefix")
        # Different system prompt → miss.
        miss = post(args.port, args.model, messages_c, args.timeout)
        print(f"[check] negative (different prefix): {answer(miss)!r} cached={cached_tokens(miss)}")
        if cached_tokens(miss) != 0:
            failures.append("different-prefix request falsely hit the cache")
        held_on.flush()
        trace = log_on.read_text(errors="replace")
        if "vmm prefix cache HIT" not in trace:
            failures.append("no 'vmm prefix cache HIT' in the trace")
        if "vmm prefix cache STORE" not in trace:
            failures.append("no 'vmm prefix cache STORE' in the trace")
        if "vmm prefix capture failed" in trace or "hit at" in trace and "failed" in trace:
            failures.append("prefix cache reported a capture/restore failure")
    except Exception as exc:  # noqa: BLE001 - summary below
        failures.append(str(exc))
    finally:
        if proc_on is not None and held_on is not None:
            stop(proc_on, held_on)
        # The daemon child can outlive the CLI by a few seconds while it
        # drains; its GPU lock must be gone before the cold server starts.
        time.sleep(2)

    # Cold control: identical prompt on a cache-disabled server. The suffix
    # prefill starts at the same 512-token chunk boundary, so greedy text must
    # match the warm arm byte for byte.
    if not failures:
        try:
            last_exc = None
            for attempt in range(15):
                try:
                    proc_off, held_off = run_server(
                        binary,
                        daemon,
                        args.model,
                        args.cold_port,
                        False,
                        0,
                        args.ctx,
                        repo,
                        log_off,
                        args.timeout,
                    )
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001 - retry lock drain
                    last_exc = exc
                    if "already reserved" not in str(exc):
                        raise
                    time.sleep(2)
            if last_exc is not None:
                raise last_exc
            cold = post(args.cold_port, args.model, messages_b, args.timeout)
            cold_text = answer(cold)
            print(f"[check] cold control: {cold_text!r}")
            if cached_tokens(cold) != 0:
                failures.append("cache-disabled server reported cached tokens")
            if warm_text is None or warm_text != cold_text:
                failures.append(
                    f"warm output {warm_text!r} != cold control {cold_text!r}"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
        finally:
            if proc_off is not None and held_off is not None:
                stop(proc_off, held_off)

    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nlogs: {log_on} {log_off}", file=sys.stderr)
        return 1
    print(
        "\nPASS: VMM prefix cache hit, negative miss, and warm text equals the "
        f"cold control (logs {log_on}, {log_off})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
