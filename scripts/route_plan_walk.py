#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Record the kernel closure of one model route by driving the real dispatch.

One daemon, one model load, then a request walk over the route's axes (prompt
rows, decode context, greedy and sampled generates, session resumes, n-gram
copies, long context), with the developer-only HIPFIRE_MODULE_LOAD_AUDIT log.
`route_plan_build.py` turns one or more walk directories into the checked-in
route plan (crates/rdna-compute/src/route_plans/). Never used by product loading.

usage: route_plan_walk.py --run DIR --daemon BIN --model M --route ar|mtp
                          [--prompt-dir DIR] [--max-seq N] [--env K=V ...]
                          [--profile full|xtx-fn-bounded] [--load-param K=JSON ...]
The environment (device selection, HIPFIRE_PLE_FORMAT, ...) is inherited; the
run gets its own HOME/HIPFIRE_HOME, lock dir and an empty kernel cache (`--env
HIPFIRE_KERNEL_CACHE=DIR` points it at a warm one). Walks of Qwen3.5
(27B, A3B) need a daemon whose loader logs `route plan axes:` (HIPFIRE_MODULE_LOAD_AUDIT=1);
Flash-Next walks only need the older `qwen4 state:` line.
`--sealed` replays the checked-in plan sealed (HIPFIRE_ROUTE_PLAN_SEAL=1) for the builder.
`--profile xtx-fn-bounded` is the gfx1100 Flash-Next walk: one fabric job's worth of requests
(every generate decodes at most 8 tokens; prompts from short rows to the 8K / 16.9K / 32K depth
prompts in `--prompt-dir`), instead of the full walk's ~640 requests and 128K prompt.
`--load-param K=JSON` adds a key to the `load` request's params (e.g. kv_mode='"q8"').
The daemon runs in its own process group and is killed when `--timeout` expires or the walk exits
abnormally, so a hung or aborted walk never leaves a daemon holding the card.
"""
import atexit
import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

BOUNDED_MAX_TOKENS = 8
# XTX card-drop rule (observed 2026-10-28): the first long-prefill request of a process that decodes past ~4 tokens
# never drops the card, later long decodes drop it at token 13-24. Every long-prefill request of the bounded walk
# (>= BOUNDED_LONG_WORDS words, and the depth prompts) therefore decodes at most BOUNDED_LONG_TOKENS tokens.
BOUNDED_LONG_TOKENS = 4
BOUNDED_LONG_WORDS = 512
# Prompt-length sweep of the bounded profile, in FILLER words (the chat template adds ~12 rows): every
# row count to 8, a coarse stride through the row classes below one 768-row S1 tile, the tile edge,
# and multi-tile prompts of ~1K..4K words.
BOUNDED_WORDS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 384, 512, 768, 1600, 4100)
# Depth prompts of the XtxGap PF screens (8,204 / 16,896 / 32,780 framed rows): one partial S1 window, exactly
# one 16,896-row window, one full window plus a 15,884-row second window and tail.
BOUNDED_DEPTH_PROMPTS = ("depth_8192.txt", "depth_16884.txt", "depth_32768.txt")

SAMPLING = [
    {"temperature": 0},
    {"temperature": 0.7, "top_p": 0.9},
    {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
    {"temperature": 0.8, "top_k": 40, "seed": 7},
    {"temperature": 0.8, "min_p": 0.05},
    {"temperature": 0.6, "presence_penalty": 0.5, "frequency_penalty": 0.3},
    {"temperature": 0, "repeat_penalty": 1.2},
    {"temperature": 0, "top_logprobs": 5},
    {"temperature": 0.7, "logit_bias": {"1000": -5.0}},
]

PROMPTS = [
    "glimmer_coding_prompt.txt", "fiction_lighthouse.txt", "bare_factual.txt",
    "humaneval_0_has_close_elements.txt", "lru_cache_pep8_strict.txt", "code_edit_rewrite_copy.txt",
    "agentic_user_read.txt", "glimmer_prefill_1024.txt", "adaptive_kv_long_prefill.txt",
    "cohere2moe_long.txt",
]
FILLER = ("The lighthouse keeper logged the tide, the wind, the passing ships and the colour of the "
          "evening sky in a ledger that had outlived three keepers before him. ")
# 125K-130K-token request: the walker measures tokens per FILLER repeat with a probe and sizes the
# prompt to the window midpoint; FIT_GUESS_REPS only seeds the recorded plan.
FIT_LO, FIT_HI = 125000, 130000
FIT_GUESS_REPS = 3700
FIT_PROBE_REPS = 1000


def prefill_rows():
    """Every row count to 300, then a stride with every 128-multiple +-2 to 4224, then the
    chunk, staging and context tails past it."""
    rows = set(range(1, 301))
    rows.update(range(301, 1101, 7))
    for base in range(128, 4225, 128):
        rows.update(base + d for d in (-2, -1, 0, 1, 2))
    rows.update([2047, 2048, 2049, 3000, 5000, 6143, 6144, 6145, 8191, 8192, 8193, 8200, 12289, 16385, 33000,
                 125000, 131000])
    return sorted(r for r in rows if r >= 1)


def filler_words(n):
    words = FILLER.split()
    return " ".join((words * (n // len(words) + 1))[:n])


def bounded_plan(prompt_dir):
    """The gfx1100 Flash-Next walk (`--profile xtx-fn-bounded`): the same plan for the AR and the MTP route,
    every generate decoding at most BOUNDED_MAX_TOKENS tokens. Sized to the XTX's measured speeds (rw1: prefill
    of a sub-512-row prompt is 2.6-28 s, 8 decoded tokens ~2 s, load ~105 s): ~37 requests, ~320 s of requests.
    Sampled generates on a short prompt (sampler + MTP verify/accept kernels), thinking-on generates (a
    4-token think budget), a prompt-length sweep (short prefill row classes, the 768-row S1 tile), a session
    resume, then the 8K / 16.9K / 32K depth prompts (S1 windows, the window tail, deep decode context). Long
    prefills (>= BOUNDED_LONG_WORDS words, the depth prompts) decode BOUNDED_LONG_TOKENS tokens, the rest 8."""
    cap = BOUNDED_MAX_TOKENS
    plan = [{"type": "generate", "prompt": "Hi", "max_tokens": cap, "temperature": 0}]
    short = "Name three rivers in Europe."
    for s in SAMPLING[1:]:     # SAMPLING[0] is plain greedy: the warmup and the sweep already are
        plan.append(dict({"type": "generate", "prompt": short, "max_tokens": cap}, **s))
    # Thinking-on generates (the typed flag the daemon treats as authoritative), tiny think budget.
    for s in ({"temperature": 0}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}):
        plan.append(dict({"type": "generate", "prompt": "Is 391 prime? Reason step by step, then answer.",
                          "max_tokens": cap, "thinking_enabled": True, "max_think_tokens": 4,
                          "_name": "think-short"}, **s))
    for n in BOUNDED_WORDS:
        plan.append({"type": "generate", "prompt": filler_words(n),
                     "max_tokens": BOUNDED_LONG_TOKENS if n >= BOUNDED_LONG_WORDS else cap, "temperature": 0,
                     "_words": n})
    plan.append({"type": "generate", "prompt": filler_words(40), "max_tokens": cap, "temperature": 0.8,
                 "top_k": 40, "seed": 40, "_words": 40})
    # Session resumes (append prefill onto a cached prefix), greedy then sampled.
    plan.append({"type": "generate", "prompt": short, "max_tokens": cap, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Now name two more.", "max_tokens": cap, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Which is longest?", "max_tokens": cap, "temperature": 0.7,
                 "top_p": 0.9})
    for name in BOUNDED_DEPTH_PROMPTS:
        plan.append({"type": "generate", "prompt": (prompt_dir / name).read_text(),
                     "max_tokens": BOUNDED_LONG_TOKENS, "temperature": 0, "_name": name})
    assert all(m["max_tokens"] <= cap for m in plan if m["type"] == "generate")
    return plan


def walk_plan(route, prompt_dir, max_seq):
    plan = [{"type": "generate", "prompt": "Hi", "max_tokens": 16, "temperature": 0}]
    medium = (prompt_dir / "glimmer_coding_prompt.txt").read_text()
    for s in SAMPLING:
        plan.append(dict({"type": "generate", "prompt": medium, "max_tokens": 24}, **s))
    for name in PROMPTS:
        text = (prompt_dir / name).read_text()
        plan.append({"type": "generate", "prompt": text, "max_tokens": 96, "temperature": 0, "_name": name})
    for name in ("code_edit_rewrite_copy.txt", "glimmer_coding_prompt.txt", "glimmer_prefill_1024.txt"):
        text = (prompt_dir / name).read_text()
        plan.append({"type": "generate", "prompt": text, "max_tokens": 96, "temperature": 0.7, "top_p": 0.9,
                     "_name": name})
    # Thinking-on generates (greedy and sampled, uncapped and capped think budget). The runner
    # defaults every generate to max_think_tokens=1 (thinking off); these entries carry their own
    # max_think_tokens, which the runner's setdefault leaves alone, plus the typed thinking flag
    # the daemon treats as authoritative.
    think_short = "Is 391 prime? Reason step by step, then answer."
    for prompt, tag in ((think_short, "think-short"), (medium, "think-medium")):
        for cap, max_tokens in ((0, 128), (24, 96)):
            for s in ({"temperature": 0}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}):
                plan.append(dict({"type": "generate", "prompt": prompt, "max_tokens": max_tokens,
                                  "thinking_enabled": True, "max_think_tokens": cap,
                                  "_name": f"{tag}-cap{cap}"}, **s))
    # n-gram takeover: long verbatim copies, greedy and sampled.
    copy = "Repeat the following text exactly, twice:\n\n" + FILLER * 12
    plan.append({"type": "generate", "prompt": copy, "max_tokens": 192, "temperature": 0})
    plan.append({"type": "generate", "prompt": copy, "max_tokens": 192, "temperature": 0.6, "top_p": 0.95})
    # Session resumes (append prefill onto a cached prefix), greedy then sampled.
    plan.append({"type": "generate", "prompt": "Name three rivers in Europe.", "max_tokens": 40, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Now name two more.", "max_tokens": 40, "temperature": 0})
    plan.append({"type": "generate", "chain_user": "Which is longest?", "max_tokens": 40, "temperature": 0.7,
                 "top_p": 0.9})
    # Long context (QSA selection past the dense window): ~20.7K-token code and ~31K synthetic.
    if max_seq > 34000:
        code = (prompt_dir / "qwen38_issue693_longcode_20676.txt").read_text()
        plan.append({"type": "generate", "prompt": code, "max_tokens": 32, "temperature": 0, "_name": "20k"})
        plan.append({"type": "generate", "prompt": code, "max_tokens": 32, "temperature": 0.7, "top_p": 0.9,
                     "_name": "20k-sampled"})
        plan.append({"type": "generate", "prompt": "Summarize this ledger.\n\n" + FILLER * 900,
                     "max_tokens": 32, "temperature": 0, "_name": "31k"})
    if max_seq > 110000:
        plan.append({"type": "generate", "prompt": "Summarize this ledger.\n\n" + FILLER * 2900,
                     "max_tokens": 24, "temperature": 0.7, "top_p": 0.9, "_name": "100k-sampled"})
    if max_seq >= 131072:
        # 125K-130K-token prompt: the walker calibrates the filler repeat count against the
        # daemon's own prompt-token count (done.prompt_tokens) and fails the request if the
        # measured prompt lands outside the window. Prefill plus a short decode at full context.
        plan.append({"type": "generate", "prompt": "Summarize this ledger.\n\n" + FILLER * FIT_GUESS_REPS,
                     "max_tokens": 16, "temperature": 0, "_name": "128k",
                     "_fit": {"lo": FIT_LO, "hi": FIT_HI, "head": "Summarize this ledger.\n\n"}})
    if route == "ar":
        for rows in prefill_rows():
            if rows < max_seq:
                plan.append({"type": "bench_prefill", "tokens": rows})
        for ctx in (1, 2, 127, 128, 1000, 2047, 2048, 4095, 4096, 8192, 9000, 16400, 33000, 65536, 120000,
                    125000, 131000):
            if ctx + 8 < max_seq:
                plan.append({"type": "bench_decode", "context_tokens": ctx, "iterations": 3})
    else:
        # Native-MTP prompt fill runs on generate only: sweep prompt lengths (rows include the
        # chat template) with short decodes, plus the chunk tails.
        words = FILLER.split()
        for n in list(range(1, 65)) + list(range(66, 1100, 9)) + [2040, 2050, 4090, 4100, 4200, 8190, 8200]:
            text = " ".join((words * (n // len(words) + 1))[:n])
            plan.append({"type": "generate", "prompt": text, "max_tokens": 6, "temperature": 0, "_words": n})
        for n in (40, 300, 900):
            text = " ".join((words * (n // len(words) + 1))[:n])
            plan.append({"type": "generate", "prompt": text, "max_tokens": 12, "temperature": 0.8,
                         "top_k": 40, "seed": n, "_words": n})
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--route", choices=["ar", "mtp"], required=True)
    ap.add_argument("--prompt-dir", default=str(pathlib.Path(__file__).resolve().parent.parent
                                                / "benchmarks" / "prompts"))
    ap.add_argument("--max-seq", type=int, default=40960)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--profile", choices=["full", "xtx-fn-bounded"], default="full",
                    help="full: the ~640-request walk of the gfx1201 plans; xtx-fn-bounded: the gfx1100 "
                         "Flash-Next walk (at most 8 decoded tokens per request, depth prompts from --prompt-dir)")
    ap.add_argument("--load-param", action="append", default=[], metavar="K=JSON",
                    help="extra key of the load request's params, value as JSON (e.g. kv_mode='\"q8\"')")
    ap.add_argument("--timeout", type=float, default=3000)
    ap.add_argument("--request-timeout", type=float, default=0,
                    help="seconds one request may take before the daemon is killed and the partial walk is "
                         "recorded (0: only --timeout applies)")
    ap.add_argument("--sealed", action="store_true",
                    help="validate a checked-in plan: keep route plans on, so any kernel the sealed "
                         "route did not plan fails its request")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    run.mkdir(parents=True, exist_ok=False)
    for name in ("home/.hipfire", "lockdir", "kcache"):
        (run / name).mkdir(parents=True)
    plan = (bounded_plan(pathlib.Path(a.prompt_dir)) if a.profile == "xtx-fn-bounded"
            else walk_plan(a.route, pathlib.Path(a.prompt_dir), a.max_seq))
    env = os.environ.copy()
    env.update(HOME=str(run / "home"), HIPFIRE_HOME=str(run / "home" / ".hipfire"),
               HIPFIRE_LOCK_DIR=str(run / "lockdir"), HIPFIRE_KERNEL_CACHE=str(run / "kcache"),
               HIPFIRE_MODULE_LOAD_AUDIT="1")
    # An inherited switch of either mode must not leak into this run.
    env.pop("HIPFIRE_ROUTE_PLAN", None)
    env.pop("HIPFIRE_ROUTE_PLAN_SEAL", None)
    if a.sealed:
        # Replay of a checked-in plan: seal it (no predicted extras), so any kernel the plan
        # did not name fails its request. route_plan_build.py --sealed-replay checks this.
        env["HIPFIRE_ROUTE_PLAN_SEAL"] = "1"
    else:
        # Observe the lazy closure, not a checked-in plan's preload.
        env["HIPFIRE_ROUTE_PLAN"] = "0"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    load = {"type": "load", "model": a.model,
            "params": {"max_seq": a.max_seq, "mtp_mode": "on" if a.route == "mtp" else "off"}}
    for kv in a.load_param:
        k, v = kv.split("=", 1)
        load["params"][k] = json.loads(v)
    daemon = pathlib.Path(a.daemon)
    (run / "command.json").write_text(json.dumps({
        "daemon": str(daemon), "daemon_sha256": hashlib.sha256(daemon.read_bytes()).hexdigest(),
        "route": a.route, "env": {k: v for k, v in env.items() if k.startswith(("HIP", "ROCR")) or k == "HOME"},
        "load": load, "requests": len(plan)}, indent=1))
    (run / "plan.json").write_text(json.dumps(plan, indent=1))

    err = (run / "stderr.log").open("w")
    out = (run / "stdout.log").open("w")
    proc = subprocess.Popen([str(daemon)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err, text=True, bufsize=1, start_new_session=True)

    def kill_daemon():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    atexit.register(kill_daemon)
    # A SIGTERM (fabric max-secs expiry) must unwind through atexit, or the daemon would keep the card.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    watchdog = threading.Timer(a.timeout, kill_daemon)
    watchdog.daemon = True
    watchdog.start()
    deadline = time.monotonic() + a.timeout
    last_text = ""

    def abort(reason):
        """The walk ended early (daemon exit, request or walk timeout): keep what it recorded. The audit stream
        of the completed requests stays a valid, smaller walk; `route_plan_build.py --accept-partial` takes it."""
        kill_daemon()
        (run / "result.json").write_text(json.dumps({"rc": None, "ok": False, "partial": True, "reason": reason,
                                                     "errors": errors, "results": results}, indent=1))
        print(json.dumps({"partial": True, "reason": reason, "requests": len(results) - 1}), flush=True)
        raise SystemExit(f"walk ended early: {reason}")

    def send(msg, done_types):
        nonlocal last_text
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()
        t = time.monotonic()
        last_text = ""
        timed_out = []

        def request_expired():
            timed_out.append(True)
            kill_daemon()

        timer = threading.Timer(a.request_timeout, request_expired) if a.request_timeout > 0 else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    abort(f"request timeout {a.request_timeout}s during {msg.get('type')} {msg.get('id', '')}"
                          if timed_out else f"daemon exited during {msg.get('type')} {msg.get('id', '')}")
                out.write(line)
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("type") == "token":
                    last_text += obj.get("text", "")
                if obj.get("type") == "commit_ready":
                    proc.stdin.write(json.dumps({"type": "commit", "id": obj["id"],
                                                 "attempt_id": obj["attempt_id"]}) + "\n")
                    proc.stdin.flush()
                if obj.get("type") in done_types:
                    return obj, time.monotonic() - t
        finally:
            if timer:
                timer.cancel()
        abort("walk timeout")

    def prompt_tokens_of(done_obj):
        """Prompt length the daemon reports on `done` (prompt_tokens, else prefill + cached)."""
        if done_obj.get("prompt_tokens") is not None:
            return done_obj["prompt_tokens"]
        if done_obj.get("prefill_tokens") is None and done_obj.get("cached_tokens") is None:
            return None
        return (done_obj.get("prefill_tokens") or 0) + (done_obj.get("cached_tokens") or 0)

    results, errors = [], 0
    obj, dt = send(load, ("loaded", "error"))
    results.append({"req": "load", "resp": obj, "s": round(dt, 2)})
    print(json.dumps({"load": obj.get("type"), "s": round(dt, 2)}), flush=True)
    if obj.get("type") == "loaded":
        history = []
        for i, msg in enumerate(plan):
            fit = msg.get("_fit")
            msg = {k: v for k, v in msg.items() if not k.startswith("_")}
            if fit is not None:
                # Size the prompt from the daemon's own token count: probe a fixed repeat count,
                # take tokens per repeat, aim at the window midpoint.
                probe = {"type": "generate", "id": f"fit{i}", "attempt_id": 1000000 + i,
                        "prompt": fit["head"] + FILLER * FIT_PROBE_REPS, "max_tokens": 1,
                        "temperature": 0, "max_think_tokens": 1}
                print(f"ROUTE_WALK_REQ\t{i}\t{json.dumps({k: v for k, v in probe.items() if k != 'prompt'})}",
                      file=err, flush=True)
                pobj, pdt = send(probe, ("done", "error"))
                probe_tokens = prompt_tokens_of(pobj)
                if pobj.get("type") == "error" or not probe_tokens:
                    errors += 1
                    print(json.dumps({"i": i, "fit_probe_error": pobj})[:400], flush=True)
                else:
                    reps = round((fit["lo"] + fit["hi"]) / 2 * FIT_PROBE_REPS / probe_tokens)
                    msg["prompt"] = fit["head"] + FILLER * reps
                    print(json.dumps({"i": i, "fit_probe_tokens": probe_tokens, "reps": reps}), flush=True)
            if msg["type"] == "generate":
                msg.setdefault("id", f"g{i}")
                msg.setdefault("attempt_id", i + 1)
                msg.setdefault("max_think_tokens", 1)
                if "chain_user" in msg:
                    history = history + [{"role": "assistant", "content": last_text},
                                         {"role": "user", "content": msg.pop("chain_user")}]
                    msg["messages"] = history
                else:
                    history = [{"role": "user", "content": msg["prompt"]}]
                done = ("done", "error")
            elif msg["type"] == "bench_prefill":
                done = ("prefill_result", "error")
            else:
                done = ("decode_result", "bench_decode_result", "error", "done")
            # Request boundary in the audit stream.
            print(f"ROUTE_WALK_REQ\t{i}\t{json.dumps({k: v for k, v in msg.items() if k not in ('prompt', 'messages')})}",
                  file=err, flush=True)
            obj, dt = send(msg, done)
            res = {"i": i, "type": msg["type"], "resp_type": obj.get("type"), "s": round(dt, 3),
                   "error": obj if obj.get("type") == "error" else None}
            if msg["type"] == "generate":
                res["prompt_tokens"] = prompt_tokens_of(obj)
            results.append(res)
            if obj.get("type") == "error":
                errors += 1
                print(json.dumps({"i": i, "error": obj})[:400], flush=True)
            elif fit is not None and not fit["lo"] <= (res["prompt_tokens"] or 0) <= fit["hi"]:
                errors += 1
                res["error"] = {"type": "error", "message": f"prompt tokens {res['prompt_tokens']} outside "
                                f"[{fit['lo']}, {fit['hi']}]"}
                print(json.dumps({"i": i, "error": res["error"]}), flush=True)
    try:
        proc.stdin.write(json.dumps({"type": "unload"}) + "\n")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait(timeout=180)
    ok = obj.get("type") != "error" and rc == 0 and errors == 0 and len(results) == len(plan) + 1
    (run / "result.json").write_text(json.dumps({"rc": rc, "ok": ok, "errors": errors, "results": results}, indent=1))
    print(json.dumps({"rc": rc, "ok": ok, "errors": errors, "requests": len(results) - 1}), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
