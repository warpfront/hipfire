#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU coherence check: post-image text turns on the multi-slot VMM route.

The VMM slot engine runs image requests on the sequential per-token M-RoPE
path and text requests on the batched `_vmm` path. A *text continuation* of an
image conversation needs M-RoPE phases too (`pos + rope_delta`); the `_vmm`
kernels have no M-RoPE writer. This check drives the feature end to end:

  1. image turn + question through a multi-slot `--kv-backend vmm` serve;
  2. a text follow-up on the same conversation (the post-image continuation);
  3. a plain text request (no image) on the same engine;
  4. a prompt longer than `serve.multi_slot_prefill_chunk` with a fact at the
     end, so the chunk boundary is exercised and the answer must recall it.

It fails on an HTTP error, an empty answer, a missing expected fact, or the
old `VMM slot forward: VL/M-RoPE rows are not admitted` error in the log.

Run on a GPU host (or in the runtime container):

  python3 scripts/check_slots_vmm_vision.py \
      --model ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4 \
      --vision ~/models/hipfire/Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.vl \
      --image benchmarks/vision/images/doge.jpeg

Exit status 0 = PASS, 1 = FAIL. The decoded text is printed for eyeballing.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_chat(port: int, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body)


def health(port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        return None


def answer_text(reply: dict) -> str:
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def wait_ready(proc: subprocess.Popen, port: int, deadline: float, log_path: Path) -> dict:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"serve exited with {proc.returncode}\n{tail}")
        h = health(port)
        if h and h.get("status") == "ok" and h.get("model"):
            return h
        time.sleep(0.5)
    tail = log_path.read_text(errors="replace")[-4000:]
    raise RuntimeError(f"serve did not become ready in time\n{tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--image", default="benchmarks/vision/images/doge.jpeg")
    ap.add_argument(
        "--image2",
        default="benchmarks/vision/images/scene_1.jpg",
        help="second image for the multi-image check (skipped when absent)",
    )
    ap.add_argument("--port", type=int, default=11577)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--spec", default="off", choices=["off", "mtp", "dflash"])
    ap.add_argument(
        "--paged-vmm",
        action="store_true",
        help="test batched vision on the shared paged pool over VMM",
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("HIPFIRE_BIN", repo / "target/release/hipfire"))
    daemon = Path(os.environ.get("HIPFIRE_DAEMON_BIN", binary.parent / "daemon"))
    if not binary.is_file():
        print(f"FAIL: hipfire binary not found: {binary}", file=sys.stderr)
        return 1
    if not daemon.is_file():
        print(f"FAIL: daemon binary not found: {daemon}", file=sys.stderr)
        return 1
    image_bytes = Path(args.image).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode()
    image2_b64 = None
    if Path(args.image2).is_file():
        image2_b64 = base64.b64encode(Path(args.image2).read_bytes()).decode()

    log = Path(tempfile.mkstemp(prefix="slots-vmm-vision-", suffix=".log")[1])
    log_handle = log.open("w")
    env = os.environ.copy()
    env.update(
        {
            "HIPFIRE_DAEMON_BIN": str(daemon),
            "HIPFIRE_SERVE_MULTI_SLOT": "1",
            "HIPFIRE_SERVE_MULTI_SLOT_SLOTS": "2",
            "HIPFIRE_SERVE_MULTI_SLOT_CTX": str(args.ctx),
            "HIPFIRE_SERVE_MULTI_SLOT_PREFILL_CHUNK": str(args.chunk),
            "HIPFIRE_VISION_MODE": "on",
            # The post-image follow-up must be a session continuation (KV
            # reuse), not a cold re-prefill; the trace makes that checkable.
            "HIPFIRE_SLOT_TRACE": "1",
        }
    )
    cmd = [
        str(binary),
        "serve",
        "--model",
        args.model,
        "--vision",
        args.vision,
        "--kv-mode",
        "q8",
        "--spec",
        args.spec,
        "--idle-timeout",
        "0",
    ]
    if args.paged_vmm:
        env["HIPFIRE_SERVE_MULTI_SLOT_PAGED"] = "1"
        env["HIPFIRE_SLOTS_PAGED_BACKEND"] = "vmm"
    else:
        cmd.extend(["--kv-backend", "vmm"])
    cmd.append(f"127.0.0.1:{args.port}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=repo,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    failures: list[str] = []
    try:
        h = wait_ready(proc, args.port, time.monotonic() + args.timeout, log)
        print(
            f"[check] ready: mode={h.get('mode')} multi_slot={h.get('multi_slot')} "
            f"backend={h.get('kv_backend') or h.get('capabilities', {}).get('kv_backend')}"
        )

        system = {"role": "system", "content": "Answer concisely and literally."}
        image_turn = {
            "model": args.model,
            "messages": [
                system,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one short sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
        }
        t0 = time.monotonic()
        first = post_chat(args.port, image_turn, args.timeout)
        image_answer = answer_text(first)
        print(f"[check] image turn ({time.monotonic() - t0:.1f}s): {image_answer!r}")
        if not image_answer:
            failures.append("image turn returned an empty answer")

        if image2_b64 is not None:
            multi = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What animal is in the first image? Answer with one word.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image2_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 24,
                "temperature": 0.0,
                "thinking_budget": "off",
                "stream": False,
            }
            multi_answer = answer_text(post_chat(args.port, multi, args.timeout))
            print(
                f"[check] multi-image first-image answer "
                f"({time.monotonic() - t0:.1f}s): {multi_answer!r}"
            )
            if "dog" not in multi_answer.lower():
                failures.append(f"multi-image check did not identify the dog: {multi_answer!r}")

        # Follow-up: the same image conversation, text only. Its KV carries the
        # image turn's compressed-grid phases, so the request needs the
        # sequential M-RoPE route. Before the fix the engine rejected the
        # batched step with "VL/M-RoPE rows are not admitted".
        # The previous user turn is history only: keep its text shape but no
        # image part, so the gateway does not re-encode the image. The
        # rendered conversation identity stays the same and the engine must
        # take the continuation-HIT path (asserted from the trace log below).
        previous_user_text = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one short sentence."}
            ],
        }
        followup = {
            "model": args.model,
            "messages": [
                system,
                previous_user_text,
                {"role": "assistant", "content": image_answer},
                {
                    "role": "user",
                    "content": "What animal is in the image? Answer with one word.",
                },
            ],
            "temperature": 0,
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
        }
        t0 = time.monotonic()
        second = post_chat(args.port, followup, args.timeout)
        follow = answer_text(second)
        print(f"[check] post-image text follow-up ({time.monotonic() - t0:.1f}s): {follow!r}")
        if not follow:
            failures.append("post-image follow-up returned an empty answer")
        elif not re.search(r"dog|shiba|inu|animal", follow, re.IGNORECASE):
            failures.append(f"post-image follow-up did not identify the dog: {follow!r}")

        # Plain text on the same engine (no image anywhere in this session).
        plain = {
            "model": args.model,
            "messages": [
                system,
                {"role": "user", "content": "Say the single word: pong"},
            ],
            "temperature": 0,
            "max_tokens": 16,
            "thinking": {"type": "disabled"},
        }
        plain_answer = answer_text(post_chat(args.port, plain, args.timeout))
        print(f"[check] plain text turn: {plain_answer!r}")
        if not plain_answer:
            failures.append("plain text turn returned an empty answer")

        # Chunk boundary: the fact is at the end of a prompt longer than one
        # prefill chunk. A lost prompt suffix or a stale KV row shows up as a
        # wrong/missing answer.
        filler = (
            "The following notes are background material. "
            "Rivers flow downhill, the sky is blue, and lists are ordered. "
        )
        need_chars = max(args.chunk * 8, 2400)
        long_prompt = (filler * ((need_chars // len(filler)) + 1))[
            :need_chars
        ] + "\n\nThe secret access code is ZEBRA-42. What is the secret access code?"
        chunk_turn = {
            "model": args.model,
            "messages": [system, {"role": "user", "content": long_prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
        }
        t0 = time.monotonic()
        chunk_reply = post_chat(args.port, chunk_turn, args.timeout)
        chunk_answer = answer_text(chunk_reply)
        prompt_tokens = (chunk_reply.get("usage") or {}).get("prompt_tokens")
        print(
            f"[check] long prompt ({time.monotonic() - t0:.1f}s, "
            f"prompt_tokens={prompt_tokens}): {chunk_answer!r}"
        )
        if "ZEBRA-42" not in chunk_answer:
            failures.append(
                "long prompt did not recall the fact across the prefill chunk: "
                f"{chunk_answer!r}"
            )

        # OpenAI logprobs: the slot engine forces AR and returns the sampled
        # row's log-probability plus top-K candidates.
        lp_reply = post_chat(
            args.port,
            {
                "model": args.model,
                "messages": [system, {"role": "user", "content": "Say the word: pong"}],
                "temperature": 0,
                "max_tokens": 8,
                "thinking": {"type": "disabled"},
                "logprobs": True,
                "top_logprobs": 5,
            },
            args.timeout,
        )
        lp_choice = (lp_reply.get("choices") or [{}])[0]
        lp_content = (lp_choice.get("logprobs") or {}).get("content") or []
        print(f"[check] logprobs entries: {len(lp_content)}")
        if not lp_content:
            failures.append("logprobs request returned no token logprobs")
        else:
            for entry in lp_content:
                lp = entry.get("logprob")
                if not isinstance(lp, (int, float)) or lp > 1e-6:
                    failures.append(f"invalid token logprob: {entry!r}")
                    break
                top = entry.get("top_logprobs")
                if not isinstance(top, list) or len(top) > 5:
                    failures.append(f"invalid top_logprobs: {entry!r}")
                    break
        try:
            post_chat(
                args.port,
                {
                    "model": args.model,
                    "messages": [system, {"role": "user", "content": "hi"}],
                    "top_logprobs": 5,
                    "max_tokens": 4,
                },
                args.timeout,
            )
            failures.append("top_logprobs without logprobs was accepted")
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                failures.append(
                    f"top_logprobs without logprobs returned HTTP {exc.code}, expected 400"
                )

        # User stop sequences: the matcher withholds the stop bytes and ends
        # the turn as `stop`; the session is closed so the next turn cannot
        # extend KV past text the client never saw.
        stop_request = {
            "model": args.model,
            "messages": [
                system,
                {"role": "user", "content": "Repeat exactly this code: ZEBRA-42"},
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stop": ["ZEBRA-42"],
            "thinking": {"type": "disabled"},
        }
        stop_reply = post_chat(args.port, stop_request, args.timeout)
        stop_choice = (stop_reply.get("choices") or [{}])[0]
        stop_text = answer_text(stop_reply)
        stop_reason = stop_choice.get("finish_reason")
        print(f"[check] stop sequence: finish={stop_reason} content={stop_text!r}")
        if stop_reason != "stop":
            failures.append(f"stop request finish_reason={stop_reason!r}, expected 'stop'")
        if "ZEBRA-42" in stop_text:
            failures.append("stop text leaked into the client content")
        # Streaming mode must apply the same matcher: no stop bytes may cross
        # the wire, and the terminal chunk must carry finish_reason=stop.
        stream_req = dict(stop_request)
        stream_req["stream"] = True
        data = json.dumps(stream_req).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        stream_text = ""
        stream_finish = None
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    frame = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in frame.get("choices") or []:
                    stream_text += (choice.get("delta") or {}).get("content") or ""
                    if choice.get("finish_reason"):
                        stream_finish = choice["finish_reason"]
        print(f"[check] stop stream: finish={stream_finish} content={stream_text!r}")
        if stream_finish != "stop":
            failures.append(f"streaming stop finish_reason={stream_finish!r}, expected 'stop'")
        if "ZEBRA-42" in stream_text:
            failures.append("streaming stop text leaked into the client content")
        # The engine's slot must be released after a stop: a new request must
        # still be served.
        after_stop = answer_text(
            post_chat(
                args.port,
                {
                    "model": args.model,
                    "messages": [system, {"role": "user", "content": "Say the word: pong"}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "thinking": {"type": "disabled"},
                },
                args.timeout,
            )
        )
        print(f"[check] after stop: {after_stop!r}")
        if not after_stop:
            failures.append("a request after a stop sequence returned empty")
        # Malformed stop shapes are a 400, not a silently ignored field.
        try:
            post_chat(
                args.port,
                {
                    "model": args.model,
                    "messages": [system, {"role": "user", "content": "hi"}],
                    "stop": [1],
                    "max_tokens": 8,
                },
                args.timeout,
            )
            failures.append("malformed stop field was accepted instead of 400")
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                failures.append(f"malformed stop returned HTTP {exc.code}, expected 400")

        # Permissive JSON mode: response_format json_object is enforced as
        # `{"type":"object"}`, so the answer must parse as a JSON object.
        json_reply = post_chat(
            args.port,
            {
                "model": args.model,
                "messages": [
                    system,
                    {
                        "role": "user",
                        "content": "Return a JSON object with keys a and b and integer values.",
                    },
                ],
                "temperature": 0,
                "max_tokens": 48,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
            args.timeout,
        )
        json_text = answer_text(json_reply)
        print(f"[check] json_object: {json_text!r}")
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            failures.append(f"json_object did not return a JSON object: {json_text!r}")

        # Two concurrent pure-text requests must both complete (request
        # isolation / no cross-slot contamination).
        results: list[tuple[str, str]] = []
        lock = threading.Lock()

        def one(tag: str, text: str) -> None:
            try:
                r = post_chat(
                    args.port,
                    {
                        "model": args.model,
                        "messages": [system, {"role": "user", "content": text}],
                        "temperature": 0,
                        "max_tokens": 24,
                        "thinking": {"type": "disabled"},
                    },
                    args.timeout,
                )
                with lock:
                    results.append((tag, answer_text(r)))
            except Exception as exc:  # noqa: BLE001 - reported as a failure
                with lock:
                    failures.append(f"concurrent {tag} failed: {exc}")

        threads = [
            threading.Thread(target=one, args=("a", "Reply with the word: alpha")),
            threading.Thread(target=one, args=("b", "Reply with the word: beta")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"[check] concurrent replies: {results}")
        if len(results) != 2 or any(not text for _, text in results):
            failures.append(f"concurrent requests did not both answer: {results}")

        log_handle.flush()
        log_text = log.read_text(errors="replace")
        if "VMM slot forward: VL/M-RoPE rows are not admitted" in log_text:
            failures.append("log contains the old VL/M-RoPE rejection on the VMM route")
        if "continuation HIT" not in log_text:
            failures.append(
                "post-image follow-up was not a session continuation "
                "(no 'continuation HIT' in the slot trace)"
            )
        if "[slot-trace] internal" in log_text or "engine poisoned" in log_text:
            failures.append("log reports an internal engine fault")
    except Exception as exc:  # noqa: BLE001 - one failure summary, not a traceback
        failures.append(str(exc))
    finally:
        log_handle.close()
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    if failures:
        print("\nFAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nserve log: {log}", file=sys.stderr)
        return 1
    print(
        "\nPASS: post-image continuation, plain text, chunk recall, stop sequences, "
        f"concurrency (log {log})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
