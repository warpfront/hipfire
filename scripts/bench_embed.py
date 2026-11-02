#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Encode benchmark for the arch-17 qwen3.5 embedding model.

Drives the daemon over its JSONL protocol (the same path a request takes),
loads once, then times N encode requests per fixture. Reports the
prefill-equivalent throughput (`pp`: one bidirectional pass over the whole
sequence) and latency.

There is no `tg`: the model has no LM head (`tie_word_embeddings: true`, no
`lm_head` tensor), so it cannot generate tokens autoregressively. The encoder
processes the whole sequence in one pass; there is no decode loop to time.

Two transports, one measurement:
  --daemon <bin> --model <path>   drives the daemon's JSONL protocol directly
  --http-url <url>                POSTs /v1/embeddings to a running serve
                                  (this is the user-facing path; use it to
                                  measure a container image)

Usage:
  scripts/bench_embed.py --daemon target/release/daemon --model <path> \
      [--runs 5] [--warmups 2] [--json]
  scripts/bench_embed.py --http-url http://127.0.0.1:8421 --runs 5
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

PROMPTS = ["embed_short", "embed_medium", "embed_long", "embed_max"]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--http-url", help="e.g. http://127.0.0.1:8421")
    ap.add_argument("--model", help="checkpoint path (required with --daemon)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--prompts-dir", default="benchmarks/prompts")
    ap.add_argument("--prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--kind", default="document")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    proc = None
    if args.http_url:
        import urllib.request

        url = args.http_url.rstrip("/") + "/v1/embeddings"

        def embed(text):
            body = json.dumps({"input": text, "kind": args.kind}).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                d = json.loads(resp.read())
            if "error" in d:
                raise SystemExit(f"serve error: {d['error']}")
            e = d["data"][0]
            return {"n_input_tokens": d["usage"]["prompt_tokens"], "n_tokens": e["n_tokens"]}

        # Confirm the endpoint answers before timing anything.
        embed("warmup")
    else:
        if not args.model:
            raise SystemExit("--model is required with --daemon")
        proc = subprocess.Popen(
            [args.daemon],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        def send(obj):
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()

        def recv_until(want):
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise SystemExit("daemon closed the pipe")
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "error":
                    raise SystemExit(f"daemon error: {d.get('message')}")
                if d.get("type") == want:
                    return d

        send({"type": "load", "model": args.model, "params": {}})
        recv_until("loaded")

        def embed(text):
            send({"type": "embed", "id": "bench", "kind": args.kind, "input": {"text": text}})
            d = recv_until("embedded")
            return {"n_input_tokens": d["n_input_tokens"], "n_tokens": d["n_tokens"]}

    results = []
    for name in args.prompts:
        path = os.path.join(args.prompts_dir, f"{name}.txt")
        text = open(path).read()
        # warmups (kernel JIT / pool growth), then timed runs
        for _ in range(args.warmups):
            embed(text)
        times, last = [], None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            last = embed(text)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        n_in = last["n_input_tokens"]
        n_tok = last["n_tokens"]
        results.append(
            {
                "fixture": name,
                "prompt_md5": md5(path),
                "prompt_tokens": n_in,
                "kept_tokens": n_tok,
                "runs": args.runs,
                "median_ms": round(med * 1e3, 3),
                "min_ms": round(min(times) * 1e3, 3),
                "max_ms": round(max(times) * 1e3, 3),
                "pp_tokens_per_s": round(n_in / med, 1),
            }
        )

    if proc is not None:
        send({"type": "unload"})
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    if args.json:
        print(json.dumps({"runs": results}, indent=2))
    else:
        print(f"{'fixture':14s} {'tokens':>7s} {'kept':>6s} {'median ms':>10s} {'pp tok/s':>9s}")
        for r in results:
            print(
                f"{r['fixture']:14s} {r['prompt_tokens']:7d} {r['kept_tokens']:6d} "
                f"{r['median_ms']:10.2f} {r['pp_tokens_per_s']:9.1f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
