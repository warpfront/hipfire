#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Drive the MTP parity fixture against a running `hipfire serve`.

One cell = one fixture cell (`<prompt>-C<n>-<t1|greedy>`): C streams started
together, each sending the prompt `--rounds` times back to back. Per cell the
driver writes `<out>/<cell>-r<rep>.json` with the per-request terminal
`timings` (tau, cycles, mtp_depth_* on lane serves, ttft_ms, latency_ms),
`/stats` before and after, and `request_index_start` (the ordinal of the
cell's first request in this server process, for slicing a
`HIPFIRE_MTP_TRACE=1` log). `check-mtp-parity-ratchet.py from-driver` turns
those files into cell records.

Request bytes follow benchmarks/mtp-parity/fixture.json: the prompt file is the
single user message (agentic: the file's own `messages` + `tools`),
max_tokens 256, thinking off (`chat_template_kwargs.enable_thinking=false` and
`reasoning_effort=none`), t1 = temperature 1.0 / top_p 0.95 / top_k 20 with
seed `1000*rep + 100*round + stream`, greedy = temperature 0. No nonce line.

Usage:
  mtp-parity-drive.py --base http://127.0.0.1:8080 --out DIR [--cells all|c1|lanes|a,b]
                      [--rep 0] [--rounds 3] [--warm 2]
  mtp-parity-drive.py --self-test
"""
import argparse
import http.server
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "benchmarks/mtp-parity/fixture.json"


def request_body(fixture, cell, rep, rnd, stream, model):
    prompt = fixture["prompts"][cell["prompt"]]
    data = (ROOT / prompt["path"]).read_bytes()
    req = fixture["request"]
    body = {
        "model": model,
        "max_tokens": req["max_tokens"],
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }
    if cell["prompt"] == "agentic":
        spec = json.loads(data)
        body["messages"] = spec["messages"]
        body["tools"] = spec["tools"]
    elif prompt.get("kind") == "mix":
        # W2: stream i sends entry i of the pinned list (wrapping), each its own prompt (and tools)
        mix = json.loads(data)["requests"]
        entry = mix[stream % len(mix)]
        body["messages"] = [{"role": "user", "content": entry["text"]}]
        if entry.get("tools"):
            body["tools"] = entry["tools"]
    else:
        body["messages"] = [{"role": "user", "content": data.decode("utf-8")}]
    mode = fixture["modes"][cell["mode"]]
    body["temperature"] = mode["temperature"]
    if mode["temperature"] > 0:
        body["top_p"] = mode["top_p"]
        body["top_k"] = mode["top_k"]
        body["seed"] = 1000 * rep + 100 * rnd + stream
    return body


def one_request(base, body, timeout):
    rq = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    out = {"error": None, "timings": None, "usage": None, "finish": None, "chunks": 0, "ttft_ms_client": None}
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if obj.get("timings"):
                    out["timings"] = obj["timings"]
                if obj.get("usage"):
                    out["usage"] = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        out["chunks"] += 1
                        if out["ttft_ms_client"] is None:
                            out["ttft_ms_client"] = (time.monotonic() - t0) * 1000.0
                    if ch.get("finish_reason"):
                        out["finish"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}"
    except Exception as e:  # recorded, never raised: a failed request must show in the cell
        out["error"] = repr(e)
    out["wall_ms"] = (time.monotonic() - t0) * 1000.0
    return out


def percentile(xs, q):
    """Nearest-rank percentile of a non-empty list."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def summarize_cell(streams, wall_s):
    """TTFT and steady decode reported separately from the end-to-end figure.

    * `reported_tok_s`: all completion tokens / the cell's wall, i.e. what a loadgen that times the
      whole request (prefill, first token, and the straggler tail after the early streams finish)
      reports.
    * `steady_tok_s`: sum over streams of (tokens after the first) / (decode seconds after the first
      token), the aggregate decode rate with TTFT and the straggler tail excluded.
    * `ttft_ms`: the client's time to the first streamed content chunk, p50 / p95 / max.
    Token counts are the server's `usage.completion_tokens`; a request without it is counted as one
    token per chunk and the summary says so (`tokens_source`)."""
    reqs = [r for st in streams for r in st if r and not r.get("error")]
    out = {"requests": len(reqs)}
    if not reqs:
        return out
    toks, src = [], "usage"
    for r in reqs:
        n = (r.get("usage") or {}).get("completion_tokens")
        if n is None:
            n, src = r.get("chunks", 0), "chunks"
        toks.append(n)
    out["tokens"] = sum(toks)
    out["tokens_source"] = src
    out["reported_tok_s"] = sum(toks) / wall_s if wall_s > 0 else None
    ttfts = [r["ttft_ms_client"] for r in reqs if r.get("ttft_ms_client") is not None]
    if ttfts:
        out["ttft_ms"] = {"p50": percentile(ttfts, 0.5), "p95": percentile(ttfts, 0.95), "max": max(ttfts)}
    steady = 0.0
    counted = 0
    for st in streams:
        after = secs = 0.0
        for r in st:
            if not r or r.get("error") or r.get("ttft_ms_client") is None:
                continue
            n = (r.get("usage") or {}).get("completion_tokens")
            n = r.get("chunks", 0) if n is None else n
            dec = (r["wall_ms"] - r["ttft_ms_client"]) / 1000.0
            if n > 1 and dec > 0:
                after += n - 1
                secs += dec
        if secs > 0:
            steady += after / secs
            counted += 1
    out["steady_tok_s"] = steady if counted else None
    out["steady_streams"] = counted
    return out


def served_model(base, wait_s=1500.0):
    """The `model` string the server reports once the model is resident (requests must name it).

    `/health` answers while the model still loads: `model` is null and `loading_model` is set until
    it is resident, so poll until it is."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                h = json.load(r)
            if h.get("model") and not h.get("loading_model"):
                return h["model"]
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"model not resident after {wait_s:.0f} s")
        time.sleep(2)


def get_stats(base):
    try:
        with urllib.request.urlopen(base + "/stats", timeout=5) as r:
            return json.load(r)
    except Exception as e:
        return {"error": repr(e)}


def run_cell(base, fixture, cell, rep, rounds, counter, timeout, model):
    streams = cell["concurrency"]
    results = [[None] * rounds for _ in range(streams)]
    start_index = counter["n"]
    before = get_stats(base)

    def worker(s):
        for rnd in range(rounds):
            results[s][rnd] = one_request(base, request_body(fixture, cell, rep, rnd, s, model), timeout)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(s,)) for s in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    counter["n"] += streams * rounds
    wall = time.monotonic() - t0
    return {
        "schema": "mtp-parity-driver-cell/1", "cell": cell["cell"], "rep": rep, "rounds": rounds,
        "concurrency": streams, "request_index_start": start_index, "requests": streams * rounds,
        "wall_s": wall, "stats_before": before, "stats_after": get_stats(base),
        "summary": summarize_cell(results, wall),
        "streams": results,
    }


def select(fixture, spec):
    cells = fixture["cells"]
    if spec == "all":
        return cells
    if spec == "c1":
        return [c for c in cells if c["concurrency"] == 1]
    if spec == "lanes":
        return [c for c in cells if c["concurrency"] > 1]
    names = set(spec.split(","))
    chosen = [c for c in cells if c["cell"] in names]
    if len(chosen) != len(names):
        raise SystemExit(f"unknown cells: {sorted(names - {c['cell'] for c in chosen})}")
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warm", type=int, default=2, help="unrecorded warmup requests (counted in request_index_start)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--model", help="model string to send (default: /health model)")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not (args.base and args.out):
        ap.error("--base and --out are required")
    fixture = json.load(open(FIXTURE))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}
    model = args.model or served_model(args.base)
    warm_cell = next(c for c in fixture["cells"] if c["concurrency"] == 1 and c["mode"] == "greedy")
    for i in range(args.warm):
        one_request(args.base, request_body(fixture, warm_cell, args.rep, 0, i, model), args.timeout)
        counter["n"] += 1
    for cell in select(fixture, args.cells):
        res = run_cell(args.base, fixture, cell, args.rep, args.rounds, counter, args.timeout, model)
        (out / f"{cell['cell']}-r{args.rep}.json").write_text(json.dumps(res, indent=1))
        errs = sum(1 for s in res["streams"] for r in s if r["error"])
        sm = res["summary"]
        tt = sm.get("ttft_ms") or {}
        print(f"{cell['cell']} rep {args.rep}: {res['requests']} requests, {errs} errors, {res['wall_s']:.1f}s; "
              f"ttft p50 {tt.get('p50', float('nan')):.0f} ms; reported {sm.get('reported_tok_s') or 0:.1f} tok/s, "
              f"steady {sm.get('steady_tok_s') or 0:.1f} tok/s", flush=True)
    return 0


def self_test():
    """A mock chat server: one SSE content chunk, a terminal `timings`, /stats."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        health_calls = 0

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                H.health_calls += 1
                body = json.dumps({"model": None, "loading_model": "x"} if H.health_calls < 3 else {"model": "mock-model", "loading_model": None}).encode()
            else:
                body = json.dumps({"draft_positions": [], "spec_cycles": len(seen)}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not req.get("model"):
                msg = b'{"error":"model is required"}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            seen.append(req)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i, obj in enumerate(({"choices": [{"delta": {"content": "hi"}}]},
                                     {"choices": [{"delta": {}, "finish_reason": "length"}],
                                      "usage": {"completion_tokens": 21},
                                      "timings": {"tau": 1.5, "cycles": 4, "ttft_ms": 5.0, "latency_ms": 45.0}})):
                if i == 1:
                    time.sleep(0.05)  # decode time after the first token
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    fixture = json.load(open(FIXTURE))
    counter = {"n": 0}
    t1 = next(c for c in fixture["cells"] if c["cell"] == "story-C4-t1")
    assert served_model(base) == "mock-model"
    res = run_cell(base, fixture, t1, rep=2, rounds=2, counter=counter, timeout=10, model="mock-model")
    bad = one_request(base, {"messages": []}, 5)
    assert bad["error"].startswith("HTTP 400") and "model is required" in bad["error"], bad["error"]
    assert res["requests"] == 8 and counter["n"] == 8 and res["request_index_start"] == 0
    assert all(r["error"] is None and r["timings"]["cycles"] == 4 for s in res["streams"] for r in s)
    seeds = sorted(r["seed"] for r in seen)
    assert seeds == sorted(1000 * 2 + 100 * rnd + s for s in range(4) for rnd in range(2)), seeds
    assert all(r["model"] == "mock-model" for r in seen)
    assert all(r["temperature"] == 1.0 and r["top_p"] == 0.95 and r["top_k"] == 20 for r in seen)
    assert all(r["max_tokens"] == 256 and r["chat_template_kwargs"] == {"enable_thinking": False} for r in seen)
    prompt = (ROOT / fixture["prompts"]["story"]["path"]).read_bytes().decode()
    assert all(r["messages"] == [{"role": "user", "content": prompt}] for r in seen)
    seen.clear()
    g = next(c for c in fixture["cells"] if c["cell"] == "agentic-C1-greedy")
    run_cell(base, fixture, g, rep=0, rounds=1, counter=counter, timeout=10, model="mock-model")
    assert counter["n"] == 9 and "seed" not in seen[0] and seen[0]["temperature"] == 0.0
    assert seen[0]["tools"] and seen[0]["messages"][-1]["role"] == "user"
    assert len(select(fixture, "c1")) == 12 and len(select(fixture, "all")) == 32
    assert len(select(fixture, "lanes")) == 20
    # TTFT and steady decode are reported separately: 8 requests x 21 tokens, 50 ms of decode each
    sm = res["summary"]
    assert sm["requests"] == 8 and sm["tokens"] == 8 * 21 and sm["tokens_source"] == "usage", sm
    assert sm["ttft_ms"]["p50"] <= sm["ttft_ms"]["p95"] <= sm["ttft_ms"]["max"]
    # 4 streams, each 2 requests x 20 tokens after the first in ~0.05 s of decode each: steady ~ 4 x 400 tok/s
    assert sm["steady_streams"] == 4 and 700 < sm["steady_tok_s"] < 1700, sm
    # W2 mix: stream i sends diverse prompt i % 8
    seen.clear()
    w2 = next(c for c in fixture["cells"] if c["cell"] == "w2mix-C8-t1")
    run_cell(base, fixture, w2, rep=0, rounds=1, counter=counter, timeout=10, model="mock-model")
    mix = json.loads((ROOT / fixture["prompts"]["w2mix"]["path"]).read_bytes())["requests"]
    got = sorted(r["messages"][0]["content"] for r in seen)
    assert got == sorted(e["text"] for e in mix) and len(set(got)) == 8
    assert sum(1 for r in seen if r.get("tools")) == 1  # only tool_agentic carries tools
    # summarize_cell on hand-made streams: counts, units and failure handling
    mk = lambda n, ttft, wall: {"error": None, "usage": {"completion_tokens": n}, "chunks": 3,
                                "ttft_ms_client": ttft, "wall_ms": wall}
    out = summarize_cell([[mk(101, 100.0, 1100.0)], [mk(201, 300.0, 1300.0), {"error": "x"}]], 2.0)
    assert out["requests"] == 2 and out["tokens"] == 302 and abs(out["reported_tok_s"] - 151.0) < 1e-9, out
    assert abs(out["steady_tok_s"] - (100 / 1.0 + 200 / 1.0)) < 1e-9, out
    assert out["ttft_ms"]["p50"] in (100.0, 300.0) and out["ttft_ms"]["max"] == 300.0
    assert summarize_cell([[{"error": "x"}]], 1.0) == {"requests": 0}
    srv.shutdown()
    print("mtp-parity-drive: self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
