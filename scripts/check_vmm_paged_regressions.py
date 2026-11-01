#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU regression check for the paged q8 VMM KV slot engine.

Drives a running `hipfire serve` (multi-slot, `--kv-mode q8`, VMM paged KV)
over HTTP and checks the behaviours that page-table bugs break first:

  1. edges:     greedy output of every prompt length in --edges (default the
                128-token page edges 127/128/129, 255/256/257, 1151/1152/1153)
                is stable solo, and identical when the same prompt runs
                concurrently with others (--rounds rounds of --lanes lanes).
  2. saturate:  --bursts bursts of --lanes concurrent --saturate-len prompts,
                sized so the page pool runs out. Every request must finish or
                be refused with an HTTP error; the connection must not drop.
  3. teardown:  --lanes streaming requests are cut off by the client after a
                few tokens (abandoned mid-decode), repeated --aborts times.
  4. after both 2 and 3, the solo and concurrent outputs from step 1 must be
     bit-identical again (pages, granules and slot state were torn down).

Prompt lengths are counted in prompt tokens as reported by the server
(`usage.prompt_tokens`, chat template included); the filler is calibrated per
model with a binary search of 1-token requests.

Example:

  hipfire serve --model ~/models/.../model.mq4r --kv-mode q8 --idle-timeout 0 &
  python3 scripts/check_vmm_paged_regressions.py --port 11435 --model ~/models/.../model.mq4r

Exit status 0 = PASS, 1 = FAIL.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

WORDS = ["alpha", "river", "stone", "maple", "cloud", "orbit",
         "lemon", "piano", "tiger", "ocean", "pixel", "quartz"]
HEAD = "Read the list, then write a short story that uses the last five items.\n"


def filler_prompt(n_words):
    return HEAD + " ".join(f"{WORDS[i % 12]}{i}" for i in range(n_words))


class Client:
    def __init__(self, port, model, timeout):
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.model = model
        self.timeout = timeout

    def body(self, prompt, max_tokens, stream=False):
        return json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()

    def chat(self, prompt, max_tokens):
        """Returns (status, sha-or-error, usage). status -1 = transport failure."""
        req = urllib.request.Request(self.url, data=self.body(prompt, max_tokens),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:200], None
        except Exception as e:  # connection reset, timeout, bad JSON
            return -1, repr(e)[:200], None
        msg = out["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\u241e" + (msg.get("content") or "")
        return 200, hashlib.sha1(text.encode()).hexdigest()[:12], out.get("usage") or {}

    def abandon_stream(self, prompt, max_tokens, after_chunks):
        """Starts a streaming request and drops the connection mid-decode."""
        req = urllib.request.Request(self.url, data=self.body(prompt, max_tokens, True),
                                     headers={"Content-Type": "application/json"})
        seen = 0
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for line in r:
                    if line.startswith(b"data:") and b"[DONE]" not in line:
                        seen += 1
                        if seen >= after_chunks:
                            break
        except Exception as e:
            return f"error before abort: {e!r}"[:200]
        return None if seen >= after_chunks else f"stream ended after {seen} chunks"


def calibrate(client, target):
    """Smallest filler word count whose prompt is exactly `target` tokens."""
    def tokens(n):
        status, info, usage = client.chat(filler_prompt(n), 1)
        if status != 200:
            raise SystemExit(f"calibration request failed: {status} {info}")
        return usage["prompt_tokens"]
    lo, hi = 0, target
    while lo < hi:
        mid = (lo + hi) // 2
        if tokens(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    got = tokens(lo)
    if got != target:
        # Word-level filler cannot hit every count; pad with single letters.
        base = filler_prompt(max(lo - 1, 0))
        for pad in range(1, 64):
            prompt = base + " x" * pad
            status, info, usage = client.chat(prompt, 1)
            if status == 200 and usage["prompt_tokens"] == target:
                return prompt
        raise SystemExit(f"cannot build a {target}-token prompt (closest {got})")
    return filler_prompt(lo)


def concurrent(client, prompts, gen):
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        return list(ex.map(lambda p: client.chat(p, gen), prompts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--model", required=True, help="model path as passed to serve")
    ap.add_argument("--edges", default="127,128,129,255,256,257,1151,1152,1153")
    ap.add_argument("--gen", type=int, default=32, help="greedy tokens per edge request")
    ap.add_argument("--lanes", type=int, default=4, help="concurrent requests (<= serve slots)")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--saturate-len", type=int, default=3841)
    ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--aborts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    client = Client(args.port, args.model, args.timeout)
    failures = []

    def fail(msg):
        failures.append(msg)
        print("FAIL", msg, flush=True)

    edges = [int(x) for x in args.edges.split(",")]
    t0 = time.time()
    prompts = {n: calibrate(client, n) for n in sorted(set(edges + [args.saturate_len]))}
    print(f"calibrated {len(prompts)} prompt lengths in {time.time() - t0:.1f}s", flush=True)

    solo = {}
    for n in edges:
        a, b = client.chat(prompts[n], args.gen), client.chat(prompts[n], args.gen)
        if a[0] != 200 or b[0] != 200:
            fail(f"solo {n}: {a[:2]} {b[:2]}")
        elif a[1] != b[1]:
            fail(f"solo {n} not deterministic: {a[1]} vs {b[1]}")
        solo[n] = a[1]
        print("solo", n, a[1], flush=True)

    def edge_rounds(tag):
        for r in range(args.rounds):
            lanes = [edges[(r * args.lanes + i) % len(edges)] for i in range(args.lanes)]
            for n, (status, sha, _) in zip(lanes, concurrent(client, [prompts[n] for n in lanes], args.gen)):
                if status != 200:
                    fail(f"{tag} round {r} len {n}: HTTP {status} {sha}")
                elif sha != solo[n]:
                    fail(f"{tag} round {r} len {n} lanes {lanes}: {sha} != solo {solo[n]}")
        print(tag, "done", flush=True)

    def solo_again(tag):
        for n in edges:
            status, sha, _ = client.chat(prompts[n], args.gen)
            if status != 200 or sha != solo[n]:
                fail(f"{tag} solo {n}: HTTP {status} {sha} != {solo[n]}")
        print(tag, "solo recheck done", flush=True)

    edge_rounds("concurrent")

    for b in range(args.bursts):
        outs = concurrent(client, [prompts[args.saturate_len]] * args.lanes, 48)
        print(f"saturate burst {b}:", [(s, i) for s, i, _ in outs], flush=True)
        for status, info, _ in outs:
            if status == -1:
                fail(f"saturate burst {b}: transport failure {info}")
    solo_again("after-saturate")

    for a in range(args.aborts):
        with cf.ThreadPoolExecutor(args.lanes) as ex:
            errs = list(ex.map(lambda i: client.abandon_stream(
                prompts[edges[(a + i) % len(edges)]], 512, 4 + 3 * i), range(args.lanes)))
        for e in errs:
            if e:
                fail(f"abort round {a}: {e}")
        time.sleep(1.0)
    solo_again("after-abort")
    edge_rounds("concurrent-after-abort")

    print(f"{'PASS' if not failures else 'FAIL'}: {len(failures)} failure(s), "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
