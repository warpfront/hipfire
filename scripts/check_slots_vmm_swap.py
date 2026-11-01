#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU check: VMM multi-slot eviction → swap-in → continuation reuse.

Two slots, three one-turn conversations. The third admission evicts the LRU
idle session to host swap. Following up on the evicted conversation must:

  * restore its KV/DeltaNet state (`swap-in` in the slot trace),
  * reuse the restored prefix (`continuation HIT ... reused N of M` with
    N > 0, not a full cold re-prefill), and
  * answer from the prior turn (`ALPHA`).

That last assertion is the regression guard for a VMM restore that wrote the
owner bytes but never published `snap.seq_len` into the pool descriptor:
`begin_turn` then read `held = 0` and silently re-prefilled the whole
conversation after every swap-in.

Run on a GPU host (or the runtime container):

  python3 scripts/check_slots_vmm_swap.py --model <4b.mq4>
"""

import argparse
import json
import os
import re
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


def health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - not ready yet
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=11584)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument(
        "--paged-vmm",
        action="store_true",
        help="test the shared paged pool on the VMM backend instead of per-request VMM owners",
    )
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("HIPFIRE_BIN", repo / "target/release/hipfire"))
    daemon = Path(os.environ.get("HIPFIRE_DAEMON_BIN", binary.parent / "daemon"))
    if not binary.is_file() or not daemon.is_file():
        print(f"FAIL: missing binary ({binary}) or daemon ({daemon})", file=sys.stderr)
        return 1

    log = Path(tempfile.mkstemp(prefix="slots-vmm-swap-", suffix=".log")[1])
    log_handle = log.open("w")
    env = os.environ.copy()
    env.update(
        {
            "HIPFIRE_DAEMON_BIN": str(daemon),
            "HIPFIRE_SERVE_MULTI_SLOT": "1",
            "HIPFIRE_SERVE_MULTI_SLOT_SLOTS": "2",
            "HIPFIRE_SERVE_MULTI_SLOT_CTX": str(args.ctx),
            "HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK": "512",
            "HIPFIRE_SLOT_TRACE": "1",
        }
    )
    serve_args = [
        str(binary),
        "serve",
        "--model",
        args.model,
        "--kv-mode",
        "q8",
        "--spec",
        "off",
        "--idle-timeout",
        "0",
    ]
    if args.paged_vmm:
        env["HIPFIRE_SERVE_MULTI_SLOT_PAGED"] = "1"
        env["HIPFIRE_SLOTS_PAGED_BACKEND"] = "vmm"
    else:
        serve_args.extend(["--kv-backend", "vmm"])
    serve_args.append(f"127.0.0.1:{args.port}")
    proc = subprocess.Popen(
        serve_args,
        env=env,
        cwd=repo,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    failures: list[str] = []
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"serve exited {proc.returncode}")
            h = health(args.port)
            if h and h.get("model"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("serve did not become ready")

        system = {"role": "system", "content": "Answer concisely."}
        convs: dict[str, tuple[list[dict], str]] = {}
        for name in ("ALPHA", "BETA", "GAMMA"):
            msgs = [
                system,
                {"role": "user", "content": f"Remember the name {name}. Reply ok."},
            ]
            ans = answer(post(args.port, args.model, msgs, args.timeout))
            convs[name] = (msgs, ans)
            print(f"[check] {name} first turn: {ans!r}")
            if not ans:
                failures.append(f"{name} first turn was empty")

        # GAMMA's admission must have evicted the LRU idle session (ALPHA).
        msgs, ans = convs["ALPHA"]
        follow = msgs + [
            {"role": "assistant", "content": ans},
            {
                "role": "user",
                "content": "What name did I ask you to remember? Answer with one word.",
            },
        ]
        got = answer(post(args.port, args.model, follow, args.timeout))
        print(f"[check] ALPHA after eviction: {got!r}")
        if "ALPHA" not in got.upper():
            failures.append(f"restored conversation did not recall ALPHA: {got!r}")
        log_handle.flush()
        text = log.read_text(errors="replace")
        if "swap-out" not in text:
            failures.append("no swap-out in the slot trace (eviction did not park a session)")
        if "swap-in" not in text:
            failures.append("no swap-in in the slot trace (the session was not restored)")
        if "swap-in FAILED" in text:
            failures.append("swap-in failed")
        hits = re.findall(r"continuation HIT -- session \d+ reused (\d+) of (\d+) tokens", text)
        if not hits:
            failures.append("no continuation HIT after the swap-in")
        else:
            reused, total = map(int, hits[-1])
            print(f"[check] continuation reuse after swap-in: {reused}/{total}")
            if reused == 0:
                failures.append(
                    "swap-in restored the owner but reused 0 tokens "
                    "(pool seq_len not published; full cold re-prefill)"
                )
    except Exception as exc:  # noqa: BLE001 - one summary, not a traceback
        failures.append(str(exc))
    finally:
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

    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nserve log: {log}", file=sys.stderr)
        return 1
    print(f"\nPASS: eviction → swap-in → prefix reuse (log {log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
