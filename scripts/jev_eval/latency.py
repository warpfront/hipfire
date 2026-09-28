"""Decide vs generate latency on identical jev-bench examples, against a running serve.

For each example, time (a) one `POST /v1/systemone` decide and (b) the same model
answering the same question through `/v1/chat/completions` (greedy, thinking off,
short max_tokens), and report p50/p95 wall-clock per path. Requests are sequential;
serve admission serialises them anyway.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True, help="model tag or path sent to serve")
ap.add_argument("--task", action="append", required=True, help="jev-bench task name, e.g. ag-news:200")
ap.add_argument("--out", required=True, help="JSON file for raw timings + summary")
a = ap.parse_args()

bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
sys.path.insert(0, str(bench))
import jevbench as jb  # noqa: E402

URL = f"http://127.0.0.1:{a.port}"


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return (time.perf_counter() - t) * 1e3


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


result = {"model": a.model, "tasks": {}}
for spec in a.task:
    name, _, n = spec.partition(":")
    n = int(n or 200)
    ex, q = jb.TASKS[name][0](500)
    ex = ex[:n]
    crit = q.get("criteria") or {}
    if isinstance(crit, dict):
        opts = "\n".join(f"{k}: {v}" for k, v in crit.items())
    else:
        opts = "\n".join(f"{i}: {v}" for i, v in enumerate(crit))
    post("/v1/systemone", {"model": a.model, "state": ex[0]["state"], "questions": {"q": q}})  # warm-up
    dec, gen = [], []
    for e in ex:
        state = e["state"] if isinstance(e["state"], str) else json.dumps(e["state"], indent=2)
        dec.append(post("/v1/systemone", {"model": a.model, "state": e["state"], "questions": {"q": q}}))
        gen.append(post("/v1/chat/completions", {
            "model": a.model, "max_tokens": 12, "temperature": 0,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content":
                          f"{state}\n\n{q['instructions']}\nOptions:\n{opts}\nAnswer with the option name only."}]}))
    summ = {k: {"p50_ms": round(pct(v, .5), 1), "p95_ms": round(pct(v, .95), 1)}
            for k, v in (("decide", dec), ("generate", gen))}
    result["tasks"][name] = {"n": n, **summ, "raw": {"decide": dec, "generate": gen}}
    print(f"{name} n={n}: decide p50 {summ['decide']['p50_ms']} ms / p95 {summ['decide']['p95_ms']} ms; "
          f"generate p50 {summ['generate']['p50_ms']} ms / p95 {summ['generate']['p95_ms']} ms", flush=True)
Path(a.out).write_text(json.dumps(result, indent=1))
