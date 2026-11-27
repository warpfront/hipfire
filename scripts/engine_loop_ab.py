#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""A/B parity harness for the engine-loop flag (G12, HIPF-468).

Runs ONE daemon binary as two processes, `HIPFIRE_ENGINE_LOOP=legacy` and
`HIPFIRE_ENGINE_LOOP=engine`, drives the same scenario matrix through both and
diffs what they produce. Usage, the route file format, the scenarios and the
intended-diff whitelist: docs/engine-loop-ab.md.

  flock ~/.halo-gpu.lock python3 scripts/engine_loop_ab.py \\
      --daemon target/release/daemon --route ROUTE.json --out DIR [--perf]

Exit 0 = every scenario equal (or an intended diff), 1 = a diff, 2 = harness or
daemon failure. Writes DIR/<route>.json (receipt) and appends rows to
DIR/EVIDENCE.md.
"""

import argparse
import fcntl
import hashlib
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ARMS = ("legacy", "engine")
# Done-envelope keys that are wall-clock measurements, never compared (at any depth).
TIMING = re.compile(r"(_ms|_s|tok_s|_us|_ns|wall|time|timings?|elapsed)$")
# Speculation bookkeeping. Compared exactly unless the route sets "spec_noise": true
# (Qwen4 MTP: these counters differ between two legacy processes with identical
# tokens, HIPF-474); then judged by G-spec only. A field present on one arm only
# is always a diff.
SPEC = re.compile(r"^(tau|cycles|rounds|accepted|mtp_windows|floor_ar_windows|ar_windows|mtp_retired"
                  r"|ngram_mod_.*|dflash_profit_ratio|dflash_retired|dflash_retire_cycle)$")
# Greedy scenarios whose done envelopes feed G-spec (aborted requests excluded).
GREEDY = ("cold", "multi_turn", "interleaved", "abort_prefill", "abort_decode", "eos_budget", "isolation")
COLD = "Write one sentence about the sea."
# Route key `chat_system`: this system message opens every multi-turn, interleaved
# and batch conversation, so each prompt is longer than the 256-row prefill chunk
# (a Gemma 4 follow-up reuses the previous prompt rounded down to the chunk).
CHAT_SYSTEM = "You are a concise assistant. Reference notes, not to be quoted:\n" + "".join(
    f"Note {i}: the {w} sensor on rack {i % 7} reported {i * 13 % 97} units. "
    for i, w in enumerate(["thermal", "power", "fan", "humidity"] * 6))


def chat_start(route):
    """The messages a conversation starts with (the route's `chat_system`)."""
    return [{"role": "system", "content": CHAT_SYSTEM}] if route.get("chat_system") else []
COUNT = "List the numbers from 1 to 100, one per line, as digits."  # decodes long: aborts land mid-decode
LONG = (
    "Summarize the following notes in two sentences.\n"
    + "".join(f"Note {i}: the {w} sensor on rack {i % 7} reported {i * 13 % 97} units. "
              for i, w in enumerate(["thermal", "power", "fan", "humidity"] * 40))
)


class Daemon:
    """One daemon process speaking the stdin/stdout JSONL protocol."""

    def __init__(self, binary, env, log_prefix, timeout_s):
        self.timeout_s = timeout_s
        self.log = open(f"{log_prefix}.jsonl", "w")
        self.p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f"{log_prefix}.stderr", "w"), env=env,
                                  text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.q.get(timeout=self.timeout_s)
        if line is None:
            raise RuntimeError(f"daemon exited rc={self.p.poll()}")
        self.log.write(line)
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"type": "nonjson", "raw": line}

    def load(self, model, params):
        self.send({"type": "load", "model": model, "params": params})
        while (e := self.recv()).get("type") not in ("loaded", "error"):
            pass
        if e["type"] != "loaded":
            raise RuntimeError(f"load failed: {e}")
        return e

    def generate(self, rid, req, abort_at_token=None, abort_now=False):
        """Run one request to its terminal event. abort_now sends the abort right
        behind the request (lands before or during prefill); abort_at_token sends
        it after that many token or reasoning events (mid-decode)."""
        return self.generate_many([(rid, req, "now" if abort_now else abort_at_token)])[rid]

    def generate_many(self, specs, then=None):
        """Run several requests at once to their terminal events; returns {rid: result}.
        specs: [(rid, req, abort)]; abort is None, "now" (sent right behind the
        request), "start" (sent on its `gen_start`: admitted, so it lands in the
        prompt) or N (sent after N token or reasoning events: mid-decode).
        then: {rid: (rid2, req2)} sends rid2's `generate` right behind rid's abort."""
        live = {}

        def generate(rid, req, abort):
            self.send({"type": "generate", "id": rid, "attempt_id": 1, **req})
            live[rid] = {"req": req, "abort": abort, "events": [], "ids": []}
            if abort == "now":
                abort_(rid)

        def abort_(rid):
            self.send({"type": "abort", "id": rid, "attempt_id": 1})
            if rid in (then or {}):
                generate(*then[rid], None)

        for spec in specs:
            generate(*spec)
        out = {}
        while len(out) < len(live):
            e = self.recv()
            rid, t = e.get("id"), e.get("type")
            s = live.get(rid)
            if s is None or rid in out:
                continue
            if t == "gen_start" and s["abort"] == "start":
                abort_(rid)
            elif t in ("token", "reasoning"):
                s["events"].append([t[0], e.get("text", "")])
                if isinstance(s["abort"], int) and len(s["events"]) == s["abort"]:
                    abort_(rid)
            elif t == "committed":  # HIPFIRE_EMIT_TOKEN_IDS=1: exact token ids
                s["ids"].append([e.get("tok_id"), e.get("pos")])
            elif t == "commit_ready":
                if s["abort"] == "ready":  # the turn finished; the client leaves it
                    abort_(rid)
                else:
                    self.send({"type": "commit", "id": rid, "attempt_id": 1})
            elif t in ("done", "error"):
                out[rid] = result(t, s["events"], s["ids"], s["req"].get("max_tokens"),
                                  {k: v for k, v in e.items() if k not in ("type", "id", "attempt_id")})
        return out

    def close(self):
        try:
            self.send({"type": "unload"})
            while self.recv().get("type") != "unloaded":
                pass
            self.p.stdin.close()
        except (BrokenPipeError, OSError, RuntimeError, queue.Empty):
            pass
        try:
            self.p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()
        return self.p.returncode


def result(typ, events, ids, max_tokens, done):
    text = "".join(x[1] for x in events)
    return {"type": typ, "events": events, "ids": ids, "text": text, "max_tokens": max_tokens,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "done": done}


def greedy(prompt, max_tokens, **kw):
    return {"prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens, **kw}


def arm_env(route, arm):
    # committed token-id events make G-tok exact on ids, not only on decoded text
    return {**os.environ, "HIPFIRE_EMIT_TOKEN_IDS": "1", **route.get("env", {}), "HIPFIRE_ENGINE_LOOP": arm}


def run_arm(args, route, arm, out, eos_n=None):
    """All scenarios on one fresh daemon. Returns {scenario: [result, ...]}.
    eos_n: the eos_budget length to rerun with (default: this arm's own probe)."""
    env = arm_env(route, arm)
    d = Daemon(args.daemon, env, f"{out}/{route['name']}.{arm}", args.timeout)
    res = {}
    mt = route.get("max_tokens", 64)
    try:
        res["load"] = [{"done": d.load(route["model"], route.get("load", {}))}]
        # cold start: first request on a fresh load
        res["cold"] = [d.generate("cold", greedy(COLD, mt))]
        # multi-turn reuse: each turn resends the history; cached_tokens must match
        hist, turns = chat_start(route), []
        for k, user in enumerate(["Name three primes.", "Now add them up.",
                                  "Is the sum prime?", "Answer in one word."]):
            hist.append({"role": "user", "content": user})
            # thinking off: prior reasoning is dropped from the re-rendered history,
            # which would break the prefix match and leave cached_tokens at 0
            r = d.generate(f"mt{k}", greedy(user, mt, messages=list(hist), thinking_enabled=False))
            hist.append({"role": "assistant", "content": r["text"]})
            turns.append(r)
        res["multi_turn"] = turns
        # two conversations interleaved A1 B1 A2 B2: a turn must reuse its own
        # conversation's prefix with the other one's state live in between (N records)
        convs = {c: chat_start(route) for c in "AB"}
        res["interleaved"] = []
        for t, user in enumerate(["Name a {} fruit.", "Why is it {}?"]):
            for c, color in (("A", "red"), ("B", "yellow")):
                convs[c].append({"role": "user", "content": user.format(color)})
                r = d.generate(f"il{c}{t}", greedy(convs[c][-1]["content"], mt, messages=list(convs[c]),
                                                    thinking_enabled=False))
                convs[c].append({"role": "assistant", "content": r["text"]})
                res["interleaved"].append(r)
        # abort mid-prefill (long prompt, abort queued behind the request), then the
        # same prompt again: a partially prefilled prefix must not be reused wrongly
        res["abort_prefill"] = [d.generate("ap", greedy(LONG, mt), abort_now=True),
                                d.generate("ap-next", greedy(LONG, mt))]
        # abort mid-decode after 8 tokens, then the same prompt again (prefix reuse)
        res["abort_decode"] = [d.generate("ad", greedy(COUNT, 4 * mt), abort_at_token=8),
                               d.generate("ad-next", greedy(COUNT, mt))]
        # stop on the last budget token: probe the natural length, then rerun with
        # the budget ending exactly at the EOS (both tokens-counting conventions).
        # Both arms rerun at the legacy n: the engine does not count EOS, so its own
        # probe is one shorter and would pair different budgets.
        p = "Reply with exactly the word: yes"
        nothink = {"thinking_enabled": False}  # a thinking model would not stop inside the budget
        probe = d.generate("eos-probe", greedy(p, mt, **nothink))
        n = probe["done"].get("tokens", 0) if eos_n is None else eos_n
        res["eos_budget"] = [probe] + [d.generate(f"eos-{b}", greedy(p, b, **nothink)) for b in (n, n + 1) if b > 0]
        # seeded sampled: same seed twice (reproducibility), then a seed sweep for
        # the cross-arm distribution check
        sp = dict(temperature=0.8, top_p=0.95, max_tokens=mt)
        q = "Write a four-line poem about a city at night."
        res["seeded_repro"] = [d.generate(f"s{k}", {"prompt": q, "seed": 1234, **sp}) for k in range(2)]
        res["seeded_sweep"] = [d.generate(f"sw{s}", {"prompt": q, "seed": s, **sp})
                               for s in range(args.seeds)]
        # state leak: the cold prompt again after every scenario above (aborts,
        # interleaved records, sampled requests) must give the cold tokens
        res["isolation"] = [d.generate("iso", greedy(COLD, mt))]
    finally:
        res["daemon_rc"] = d.close()
    return res


# ── Batch-path cache scenario (`--batch`, HIPF-472) ─────────────────────────
# Plain `generate` requests run the singleton route, which keeps one session
# record; the N records live behind the continuous-batch route (`serve.vmm_batch`).
BATCH_LOAD = {"continuous_batch_size": 4, "serve_vmm_batch": {}}
# Twin conversation pairs (identical messages): P aborts its second turn in the
# prompt, D in decode; the twin's second turn is the control the retry must match.
# D's third turn is aborted in decode on the resident state, P's third turn right behind it.
BATCH_TURNS = {"P": ("Explain how a hash map resolves collisions.", LONG, COLD),
               "D": ("Name five rivers in Europe.", COUNT, COUNT)}


def batch_req(messages, max_tokens):
    """A greedy chat turn as `hipfire bench --backend batch` sends it
    (bench_concurrency.rs build_request): neutral penalties keep it batch-eligible."""
    return {"messages": messages, "prompt": messages[-1]["content"], "temperature": 0.0, "top_p": 1.0,
            "max_tokens": max_tokens, "repeat_penalty": 1.0, "presence_penalty": 0.0, "min_p": 0.0,
            "thinking_enabled": False, "assistant_prefix": "closed_think", "serve_continuous_batch": True}


def run_batch(args, route, arm, out):
    """One fresh daemon on the continuous-batch route: turn 0 of P, P~, D, D~ at once;
    then their turn 1 at once, P's aborted on `gen_start` (in the prompt), D's after
    8 tokens; then P's and D's turn 1 again; then D's turn 2 alone (resident), aborted
    after 8 tokens with P's turn 2 sent right behind the abort; then D's and D~'s turn 2.
    Returns {request: result}."""
    prefix = f"{out}/{route['name']}.batch.{arm}"
    wait_gpu_released()  # a `--conc` bench's daemon can still hold the reservation (HIPF-472 r4)
    d = Daemon(args.daemon, arm_env(route, arm), prefix, args.timeout)
    mt = route.get("max_tokens", 64)
    res = {}
    try:
        res["loaded"] = d.load(route["model"], {**route.get("load", {}), **route.get("batch_load", BATCH_LOAD)})
        hist = {c: chat_start(route) + [{"role": "user", "content": BATCH_TURNS[c[0]][0]}]
                for c in ("P", "P~", "D", "D~")}
        res.update(d.generate_many([(c + "0", batch_req(hist[c], mt), None) for c in hist]))
        for c in hist:
            hist[c] += [{"role": "assistant", "content": res[c + "0"]["text"]},
                        {"role": "user", "content": BATCH_TURNS[c[0]][1]}]
        res.update(d.generate_many([("P1", batch_req(hist["P"], mt), "start"), ("P~1", batch_req(hist["P~"], mt), None),
                                    ("D1", batch_req(hist["D"], 4 * mt), 8), ("D~1", batch_req(hist["D~"], mt), None)]))
        res.update(d.generate_many([("P1r", batch_req(hist["P"], mt), None), ("D1r", batch_req(hist["D"], mt), None)]))
        for c, turn in (("P", "P1r"), ("D", "D1r"), ("D~", "D~1")):
            hist[c] += [{"role": "assistant", "content": res[turn]["text"]},
                        {"role": "user", "content": BATCH_TURNS[c[0]][2]}]
        res.update(d.generate_many([("D2", batch_req(hist["D"], 4 * mt), 8)],
                                   then={"D2": ("P2", batch_req(hist["P"], mt))}))
        res.update(d.generate_many([("D2r", batch_req(hist["D"], mt), None), ("D~2", batch_req(hist["D~"], mt), None)]))
        for c, turn in (("P", "P2"), ("D", "D2r"), ("D~", "D~2")):
            hist[c] += [{"role": "assistant", "content": res[turn]["text"]},
                        {"role": "user", "content": BATCH_TURNS[c[0]][2]}]
        res.update(d.generate_many([("D3", batch_req(hist["D"], mt), "ready")],
                                   then={"D3": ("P3", batch_req(hist["P"], mt))}))
        res.update(d.generate_many([("D3r", batch_req(hist["D"], mt), None), ("D~3", batch_req(hist["D~"], mt), None)]))
    finally:
        res["daemon_rc"] = d.close()
    with open(f"{prefix}.jsonl") as f:
        res["refused"] = refused_records(f, Path(f"{prefix}.stderr").read_text(errors="replace"))
    return res


def hist_len(r):
    """Tokens a finished turn commits to its conversation: the prompt plus every
    committed id (EOS included; legacy `tokens` counts it, the engine's does not).
    A follow-up that reuses the whole turn reports exactly this as `cached_tokens`."""
    d = r["done"]
    n = len(r["ids"]) if r["ids"] else (d.get("tokens") or 0) + (d.get("finish_reason") == "stop")
    return (d.get("cached_tokens") or 0) + (d.get("prefill_tokens") or 0) + n


def batch_checks(rows, expect_engine, reuse_mode="turn", ref=None):
    """The `--batch` gates on one arm's rows (the run gates the engine arm only;
    legacy fn-lanes keep no records). Pure: --self-test drives it with fakes.
    - every request ends in `done`; the aborted ones `aborted`, the others batched
    - the twins' turn 0 are token-exact (identical prompts at once)
    - reuse_mode "turn": every turn 1 that finished (P~1, D~1, and the retries P1r,
      D1r after the aborts) reuses at least its conversation's committed turn 0;
      P2, D~2 and D2r (after D2's resident abort and P2's turn) their committed
      turn 1; P3, D~3, D3r their committed turn 2
    - reuse_mode "prompt" (a template that re-renders earlier replies, Gemma 4): the
      reply is never reusable, so every follow-up reuses more than 0 and at least
      the `ref` arm's (legacy) reuse of the same request
    - a retry reuses at least its control (the abort lost nothing), and with the
      control's reuse gives the control's tokens and ids"""
    fails, unex, notes = [], [], []
    cached = lambda r: r["done"].get("cached_tokens") or 0
    same = lambda a, b: (a["events"], a["ids"]) == (b["events"], b["ids"])
    for name in ("P0", "P~0", "D0", "D~0", "P1", "P~1", "D1", "D~1", "P1r", "D1r", "D2", "P2", "D2r", "D~2",
                 "D3", "P3", "D3r", "D~3"):
        r, aborted = rows[name], name in ("P1", "D1", "D2", "D3")
        if r["type"] != "done":
            fails.append(f"{name} error: {r['done'].get('message')}")
            continue
        if expect_engine and r["done"].get("loop") != "engine":
            fails.append(f"{name} loop {r['done'].get('loop')!r}")
        if aborted and r["done"].get("finish_reason") != "aborted":
            unex.append(f"{name}: the abort did not land (finish_reason {r['done'].get('finish_reason')})")
        if not aborted and not (r["done"].get("continuous_batch") or {}).get("executed"):
            unex.append(f"{name} ran on the singleton route, not batched")
    if rows["P1"]["events"]:
        unex.append(f"P1: the abort landed after {len(rows['P1']['events'])} decode events, not in the prompt")
    for name in ("D1", "D2"):
        if len(rows[name]["events"]) > rows[name]["max_tokens"] // 2:
            unex.append(f"{name}: the abort landed at {len(rows[name]['events'])}/{rows[name]['max_tokens']} events")
    for a, b in (("P0", "P~0"), ("D0", "D~0")):
        if not same(rows[a], rows[b]):
            fails.append(f"twins {a}/{b} differ at turn 0")
    reuse = {}
    for turn, prev, n in (("P~1", "P~0", 0), ("D~1", "D~0", 0), ("P1r", "P0", 0), ("D1r", "D0", 0),
                          ("P2", "P1r", 1), ("D~2", "D~1", 1), ("D2r", "D1r", 1),
                          ("P3", "P2", 2), ("D~3", "D~2", 2), ("D3r", "D2r", 2)):
        reuse[turn] = [cached(rows[turn]), hist_len(rows[prev])]
        if reuse_mode == "turn" and reuse[turn][0] < reuse[turn][1]:
            fails.append(f"{turn} reused {reuse[turn][0]} < {reuse[turn][1]} (its committed turn {n})")
        elif reuse_mode == "prompt":
            floor = cached(ref[turn]) if ref else 0
            reuse[turn].append(floor)
            if not reuse[turn][0]:
                fails.append(f"{turn} reused no prefix")
            elif reuse[turn][0] < floor:
                fails.append(f"{turn} reused {reuse[turn][0]} < legacy {floor}")
    for retry, ctrl in (("P1r", "P~1"), ("D1r", "D~1"), ("D2r", "D~2"), ("D3r", "D~3")):
        if cached(rows[retry]) < cached(rows[ctrl]):
            fails.append(f"{retry} reused {cached(rows[retry])} < control {ctrl} {cached(rows[ctrl])} (the abort lost reuse)")
        elif cached(rows[retry]) != cached(rows[ctrl]):
            notes.append(f"{retry} reused {cached(rows[retry])}, control {ctrl} {cached(rows[ctrl])}: tokens not compared")
        elif not same(rows[retry], rows[ctrl]):
            fails.append(f"{retry} tokens differ from {ctrl} at the same reuse")
    if rows.get("daemon_rc") != 0:
        fails.append(f"daemon rc {rows.get('daemon_rc')}")
    if rows.get("refused"):  # the engine's own reason for a cold turn (e.g. a retired MTP lane)
        notes.append(f"lane records not continued (engine log): {', '.join(rows['refused'])}")
    return {"ok": not fails, "fails": fails, "unexercised": unex, "notes": notes, "reuse": reuse}


def refused_records(lines, log_text):
    """The requests whose lane record the engine logged as not continued: tag N is
    the N-th `gen_start` on the daemon's stdout (see refused_followups)."""
    starts = [e.get("id") for e in map(json.loads, (ln for ln in lines if '"gen_start"' in ln))
              if e.get("type") == "gen_start"]
    return [starts[t - 1] if 0 < t <= len(starts) else f"tag {t}" for t in map(int, REFUSAL.findall(log_text))]


def strip_timing(v):
    if isinstance(v, dict):
        return {k: strip_timing(x) for k, x in sorted(v.items()) if not TIMING.search(k)}
    if isinstance(v, list):
        return [strip_timing(x) for x in v]
    return v


def done_view(done):
    return {k: v for k, v in strip_timing(done).items() if k != "loop"}


# Note kinds. Only DIFF fails a scenario; UNEXERCISED marks a scenario that did not
# test what its name says (shown in the summary, does not fail it).
DIFF, INTENDED, SPEC_STAT, UNEXERCISED = "diff", "intended", "spec-stat", "unexercised"
# Prompt-side done fields, compared on sampled requests too.
PROMPT_SIDE = ("cached_tokens", "prefill_tokens")
# Scenario -> index of its first follow-up that should reuse a prefix the legacy
# loop drops (another conversation in between, an abort before it).
MORE_REUSE = {"interleaved": 2, "abort_prefill": 1, "abort_decode": 1}


def toks(r):
    """A result's committed token ids, without their positions."""
    return [t for t, _ in r.get("ids", [])]


def compare(sc, a, b, spec_noise=False, positions=False):
    """Diff two result lists (a = legacy, b = engine). Returns (status, notes);
    notes are [kind, text]; status is diff|unexercised|intended|equal."""
    notes = []

    def note(kind, text):
        notes.append([kind, text])
    if len(a) != len(b):
        note(DIFF, f"{len(a)} vs {len(b)} requests")
    for i, (x, y) in enumerate(zip(a, b)):
        # an error outside an abort is a broken scenario, even when both arms agree
        for name, r in (("legacy", x), ("engine", y)):
            if sc != "load" and r["type"] == "error" and not (sc.startswith("abort") and i == 0):
                note(DIFF, f"[{i}] {name} error: {r['done'].get('message')}")
        dx, dy = done_view(x["done"]), done_view(y["done"])
        if sc == "load":
            if dx != dy:
                note(DIFF, "loaded envelope differs")
            continue
        if sc.startswith("seeded"):
            for key in PROMPT_SIDE:
                if dx.get(key) != dy.get(key):
                    note(DIFF, f"[{i}] {key}: {dx.get(key)!r} vs {dy.get(key)!r}")
            continue
        # coverage, judged on the legacy arm
        if sc == "eos_budget" and i == 0 and dx.get("finish_reason") != "stop":
            note(UNEXERCISED, "the probe hit its budget before EOS")
        if sc == "multi_turn" and i > 0 and not dx.get("cached_tokens"):
            note(UNEXERCISED, f"[{i}] turn reused no prefix (cached_tokens {dx.get('cached_tokens')!r})")
        # reuse the legacy loop may lack (one live session record): judged on the
        # engine arm; more reuse than legacy over the same prompt is intended
        if sc in MORE_REUSE and i >= MORE_REUSE[sc]:
            cx, cy = dx.get("cached_tokens") or 0, dy.get("cached_tokens") or 0
            if not cy:
                note(UNEXERCISED, f"[{i}] the engine reused no prefix")
            if cy > cx and cx + (dx.get("prefill_tokens") or 0) == cy + (dy.get("prefill_tokens") or 0):
                note(INTENDED, f"[{i}] engine reused {cy} cached tokens, legacy {cx}")
                dx, dy = ({k: v for k, v in d.items() if k not in PROMPT_SIDE} for d in (dx, dy))
        if sc == "abort_prefill" and i == 0 and x["events"]:
            note(UNEXERCISED, f"abort landed after {len(x['events'])} decode events, not in prefill")
        if sc == "abort_decode" and i == 0:
            if x.get("max_tokens") and len(x["events"]) > x["max_tokens"] // 2:
                note(UNEXERCISED, f"abort landed at {len(x['events'])}/{x['max_tokens']} events")
            if dx.get("finish_reason") != "aborted":
                note(DIFF, f"abort did not land (finish_reason {dx.get('finish_reason')})")
            if dx.get("finish_reason") != dy.get("finish_reason"):
                note(DIFF, f"finish_reason {dx.get('finish_reason')} vs {dy.get('finish_reason')}")
            # the abort lands asynchronously: only the common prefix is compared
            for f, xs, ys in (("events", x["events"], y["events"]), ("ids", toks(x), toks(y))):
                if xs[:len(ys)] != ys[:len(xs)]:
                    note(DIFF, f"{f} diverge before the abort")
            continue
        if x["events"] != y["events"]:
            note(DIFF, f"[{i}] tokens differ ({x['sha']} vs {y['sha']})")
        if toks(x) != toks(y):
            note(DIFF, f"[{i}] token ids differ")
        elif x.get("ids", []) != y.get("ids", []):
            # Intended diff 6 (`positions`, Gemma 4 EAGLE only): the engine
            # stamps every committed token with its conversation position;
            # legacy Gemma 4 EAGLE counts from the first generated token.
            note(INTENDED if positions else DIFF, f"[{i}] committed positions differ, token ids equal")
        for key in sorted(set(dx) | set(dy)):
            vx, vy = dx.get(key), dy.get(key)
            if vx == vy:
                continue
            # Intended diff 2: EOS is not counted in `tokens` on the engine.
            if key == "tokens" and isinstance(vx, int) and vy == vx - 1 and dx.get("finish_reason") == "stop":
                note(INTENDED, f"[{i}] tokens {vx}->{vy} (EOS not counted)")
            # Intended diff 3: a stop on the last budget token is `stop`.
            elif key == "finish_reason" and sc == "eos_budget" and (vx, vy) == ("length", "stop"):
                note(INTENDED, f"[{i}] finish_reason length->stop on last budget token")
            elif SPEC.match(key) and spec_noise and key in dx and key in dy:
                note(SPEC_STAT, f"[{i}] {key}: {vx!r} vs {vy!r}")
            else:
                note(DIFF, f"[{i}] {key}: {vx!r} vs {vy!r}")
    kinds = {k for k, _ in notes}
    status = next((k for k in (DIFF, UNEXERCISED, INTENDED) if k in kinds), "equal")
    return status, notes


def first_tokens(rows, k=8):
    hist = {}
    for r in rows:
        for e in r["events"][:k]:
            hist[e[1]] = hist.get(e[1], 0) + 1
    tot = sum(hist.values()) or 1
    return {t: c / tot for t, c in hist.items()}


def sampled_checks(res):
    """Intended diff 1 (per-request RNG): sampled output is not compared bitwise
    across arms. Each arm must reproduce itself, must honour the seed (as many
    distinct sweep outputs as legacy), and the arms' early-token distributions
    must stay close (total variation distance)."""
    out = {}
    for arm in ARMS:
        a, b = res[arm]["seeded_repro"]
        out[f"{arm}_reproducible"] = a["events"] == b["events"]
        out[f"{arm}_distinct"] = len({r["sha"] for r in res[arm]["seeded_sweep"]})
    p, q = first_tokens(res["legacy"]["seeded_sweep"]), first_tokens(res["engine"]["seeded_sweep"])
    out["tvd_first8"] = round(0.5 * sum(abs(p.get(t, 0) - q.get(t, 0)) for t in set(p) | set(q)), 4)
    out["bitwise_equal_across_arms"] = all(
        x["events"] == y["events"] for x, y in zip(res["legacy"]["seeded_sweep"], res["engine"]["seeded_sweep"]))
    return out


def loop_checks(res, expect_engine):
    """Every done envelope carries `loop` once an arm emits it at all; the value is
    `legacy` or the arm's own; with expect_engine, every engine-arm done is `engine`
    (catches an engine arm that silently fell back to legacy)."""
    out, ok = {}, True
    for arm in ARMS:
        vals = [r["done"].get("loop") for sc, rows in res[arm].items() if sc not in ("load", "daemon_rc")
                for r in rows if r["type"] == "done"]
        out[arm] = {str(v): vals.count(v) for v in set(vals)}
        bad = [v for v in vals if v not in (None, "legacy", arm)]
        missing = None in vals and any(v is not None for v in vals)
        fell_back = arm == "engine" and expect_engine and any(v != "engine" for v in vals)
        ok = ok and not bad and not missing and not fell_back
    return ok, out


def judge(res, route, tvd_max, perf=None):
    """All gates over both arms' results. Pure: --self-test drives it with fakes."""
    scenarios = {}
    for sc in res["legacy"]:
        if sc == "daemon_rc":
            continue
        status, notes = compare(sc, res["legacy"][sc], res["engine"][sc], route.get("spec_noise", False),
                                "drafter" in route.get("load", {}))
        scenarios[sc] = {"status": status, "notes": notes}
    sampled = sampled_checks(res)
    sampled["ok"] = (sampled["legacy_reproducible"] and sampled["engine_reproducible"]
                     and sampled["engine_distinct"] >= sampled["legacy_distinct"]
                     and sampled["tvd_first8"] <= tvd_max)
    # G-spec: engine mean tau >= legacy - 1 %; an engine that drops tau fails. On a spec_noise
    # route ~150 windows spread ~1 % sd run to run (HIPF-615), so G-spec is report-only there
    # and the conc k=1 cells (500+ windows each) gate tau.
    taus = {arm: [r["done"]["tau"] for sc in GREEDY if sc in res[arm]
                  for r in res[arm][sc][int(sc.startswith("abort")):] if "tau" in r["done"]] for arm in ARMS}
    g_spec = None
    if taus["legacy"]:
        g_spec = (len(taus["engine"]) == len(taus["legacy"])
                  and statistics.mean(taus["engine"]) >= 0.99 * statistics.mean(taus["legacy"]))
    loop_ok, loops = loop_checks(res, route.get("expect_engine", False))
    isolation = {arm: isolation_check(res[arm]) for arm in ARMS if "isolation" in res[arm]}
    g_perf = perf["median"]["g_perf"] if perf else None
    ok = (all(s["status"] != DIFF for s in scenarios.values()) and sampled["ok"]
          and (g_spec is not False or route.get("spec_noise", False))
          and loop_ok and False not in isolation.values() and g_perf is not False
          and res["legacy"]["daemon_rc"] == res["engine"]["daemon_rc"] == 0)
    return {"ok": ok, "scenarios": scenarios, "sampled": sampled, "tau": taus, "g_spec": g_spec,
            "loop_ok": loop_ok, "loop": loops, "isolation": isolation, "g_perf": g_perf}


def isolation_check(rows):
    """No state leak: the cold prompt rerun last gives the cold tokens and ids. Judged
    when the rerun prefilled cold or restored the whole prompt; a partial prefix reuse
    is not token-exact with a cold prefill, so it is `unexercised`."""
    c, i = rows["cold"][0], rows["isolation"][0]
    prompt = (c["done"].get("cached_tokens") or 0) + (c["done"].get("prefill_tokens") or 0)
    if 0 < (i["done"].get("cached_tokens") or 0) < prompt:
        return UNEXERCISED
    return (c["events"], c.get("ids", [])) == (i["events"], i.get("ids", []))


def self_test():
    """GPU-free check of the diff rules and gates (`--self-test`)."""
    def r(text, typ="done", **done):
        return {"type": typ, "events": [["t", c] for c in text], "sha": text, "done": done, "max_tokens": 64}

    def st(sc, a, b, **kw):
        return compare(sc, a, b, **kw)[0]
    same = [r("ab", tokens=3, finish_reason="stop", prefill_ms=1.0)]
    assert st("cold", same, [r("ab", tokens=3, finish_reason="stop", prefill_ms=9.0, loop="engine")]) == "equal"
    assert st("cold", [r("ab", floor={"x_ms": 1, "k": 2})], [r("ab", floor={"x_ms": 5, "k": 2})]) == "equal"
    assert st("cold", [r("ab", floor={"x_ms": 1, "k": 2})], [r("ab", floor={"x_ms": 1, "k": 3})]) == "diff"
    assert st("cold", [r("ab", w=[{"t_ms": 1}])], [r("ab", w=[{"t_ms": 2}])]) == "equal"
    assert st("cold", same, [r("ac", tokens=3, finish_reason="stop")]) == "diff"
    assert st("cold", same, [r("ab", tokens=2, finish_reason="stop")]) == "intended"
    assert st("cold", same, [r("ab", tokens=1, finish_reason="stop")]) == "diff"
    assert st("cold", [r("ab", tokens=3, finish_reason="length")],
              [r("ab", tokens=2, finish_reason="length")]) == "diff"
    assert st("cold", same, [r("ab", tokens=3, finish_reason="stop", cached_tokens=4)]) == "diff"
    assert st("cold", same, same + same) == "diff"
    probe = r("ab", tokens=2, finish_reason="stop")
    budget = [probe, r("ab", tokens=2, finish_reason="length")]
    assert st("eos_budget", budget, [probe, r("ab", tokens=2, finish_reason="stop")]) == "intended"
    assert st("cold", budget, [probe, r("ab", tokens=2, finish_reason="stop")]) == "diff"
    assert st("eos_budget", [probe, probe], [probe, r("ab", tokens=2, finish_reason="length")]) == "diff"
    assert st("eos_budget", budget[1:], budget[1:]) == "unexercised"
    err = [r("", "error", message="ctx")]
    assert st("cold", err, err) == "diff"
    assert st("load", [{"done": {"arch": "a", "load_ms": 1}}], [{"done": {"arch": "a", "load_ms": 2}}]) == "equal"
    assert st("load", [{"done": {"arch": "a"}}], [{"done": {"arch": "b"}}]) == "diff"
    # spec counters: exact by default, loose with spec_noise, a dropped field always a diff
    sx, sy = [r("ab", tau=1.2, cycles=4)], [r("ab", tau=0.9, cycles=5)]
    assert st("cold", sx, sy) == "diff"
    assert st("cold", sx, sy, spec_noise=True) == "equal"
    assert st("cold", sx, [r("ab", cycles=5)], spec_noise=True) == "diff"
    # aborts
    fx = r("x", cached_tokens=3)  # a follow-up that reused the prompt prefix
    ab = [r("abcd", finish_reason="aborted"), fx]
    assert st("abort_decode", ab, [r("abc", finish_reason="aborted"), fx]) == "equal"
    assert st("abort_decode", ab, [r("abX", finish_reason="aborted"), fx]) == "diff"
    assert st("abort_decode", ab, [r("abcd", finish_reason="stop"), fx]) == "diff"
    assert st("abort_decode", ab, [r("abcd", finish_reason="aborted"), r("y", cached_tokens=3)]) == "diff"
    assert st("abort_decode", [r("ab", finish_reason="stop"), fx],
              [r("ab", finish_reason="stop"), fx]) == "diff"
    assert st("abort_decode", [r("a" * 40, finish_reason="aborted"), fx],
              [r("a" * 40, finish_reason="aborted"), fx]) == "unexercised"
    assert st("abort_decode", [ab[0], r("x")], [ab[0], r("x")]) == "unexercised"  # no reuse after the abort
    assert st("abort_prefill", [r("", "error", message="m"), r("x")], [r("", "error", message="m"), r("x")]) != "diff"
    assert st("abort_decode", ab, [ab[0], r("", "error", message="m")]) == "diff"
    assert st("abort_decode", [ab[0], r("", "error", message="m")], [ab[0], r("", "error", message="m")]) == "diff"
    assert st("abort_prefill", [r("", finish_reason="aborted"), fx],
              [r("", finish_reason="aborted"), fx]) == "equal"
    assert st("abort_prefill", [r("", finish_reason="aborted"), r("x")],
              [r("", finish_reason="aborted"), r("x")]) == "unexercised"
    assert st("abort_prefill", [r("ab", finish_reason="aborted"), fx],
              [r("ab", finish_reason="aborted"), fx]) == "unexercised"
    mt = [r("a", cached_tokens=0), r("b", cached_tokens=0)]
    assert st("multi_turn", mt, mt) == "unexercised"
    assert st("multi_turn", mt, [mt[0], r("b", cached_tokens=0, tokens=-1)]) == "diff"
    mt2 = [r("a", cached_tokens=0), r("b", cached_tokens=0, tokens=3, finish_reason="stop")]
    assert st("multi_turn", mt2, [mt2[0], r("b", cached_tokens=0, tokens=2, finish_reason="stop")]) == "unexercised"
    # sampled requests: tokens free, prompt-side fields exact
    assert st("seeded_sweep", [r("ab", cached_tokens=0)], [r("xy", cached_tokens=0)]) == "equal"
    assert st("seeded_sweep", [r("ab", cached_tokens=0)], [r("xy", cached_tokens=5)]) == "diff"
    # token ids: exact, aborts on the common prefix
    ids = lambda x, i: {**x, "ids": i}
    assert st("cold", [ids(r("ab"), [[1, 0]])], [ids(r("ab"), [[2, 0]])]) == "diff"
    # positions: intended only on a Gemma 4 EAGLE route
    assert st("cold", [ids(r("ab"), [[1, 0]])], [ids(r("ab"), [[1, 5]])]) == "diff"
    assert st("cold", [ids(r("ab"), [[1, 0]])], [ids(r("ab"), [[1, 5]])], positions=True) == "intended"
    assert st("abort_decode", [ids(ab[0], [[1, 0], [2, 1]]), r("x", cached_tokens=3)],
              [ids(ab[0], [[1, 0]]), r("x", cached_tokens=3)]) == "equal"
    assert st("abort_decode", [ids(ab[0], [[1, 0], [2, 1]]), r("x", cached_tokens=3)],
              [ids(ab[0], [[1, 0], [3, 1]]), r("x", cached_tokens=3)]) == "diff"
    il = [r("a", cached_tokens=0), r("b", cached_tokens=0), r("c", cached_tokens=5), r("d", cached_tokens=0)]
    assert st("interleaved", il, il) == "unexercised"
    assert st("interleaved", il[:3], il[:3]) == "equal"
    assert st("interleaved", il[:3], il[:2] + [r("c", cached_tokens=4)]) == "diff"
    # the engine's N live records reuse what one legacy record dropped: intended, if the prompt is the same
    lx = il[:2] + [r("c", cached_tokens=0, prefill_tokens=10)]
    assert st("interleaved", lx, il[:2] + [r("c", cached_tokens=6, prefill_tokens=4)]) == "intended"
    assert st("interleaved", lx, il[:2] + [r("c", cached_tokens=6, prefill_tokens=5)]) == "diff"
    assert st("abort_decode", [ab[0], r("x", cached_tokens=0, prefill_tokens=9)],
              [ab[0], r("x", cached_tokens=8, prefill_tokens=1)]) == "intended"

    # gates
    def arm(loop=None, tau=1.0, sweep=("abc", "bcd", "cde", "def"), repro=("p", "p"), rc=0, iso="ab"):
        def d(text):
            g = r(text, finish_reason="stop", tokens=len(text), cached_tokens=0, tau=tau, loop=loop)
            g["done"] = {k: v for k, v in g["done"].items() if v is not None}
            return g
        return {"load": [{"done": {"arch": "x"}}], "cold": [d("ab")], "isolation": [d(iso)],
                "seeded_repro": [d(repro[0]), d(repro[1])], "seeded_sweep": [d(s) for s in sweep], "daemon_rc": rc}

    def ok(eng, leg=None, route=None, perf=None):
        return judge({"legacy": leg or arm(), "engine": eng}, route or {}, 0.35, perf)["ok"]
    assert ok(arm())
    assert ok(arm(loop="engine"), route={"expect_engine": True})
    assert not ok(arm(rc=1))
    assert not ok(arm(repro=("p", "q")))
    assert not ok(arm(sweep=("xyz", "yzx", "zxy", "xzy")))  # distribution moved
    assert not ok(arm(sweep=("abc",) * 4))  # seed ignored
    assert not ok(arm(sweep=("abc",) * 4), leg=arm(sweep=("abc", "abd", "abe", "abf")))  # seed ignored, tvd 0.25
    assert not ok(arm(tau=0.5))  # exact spec counters
    assert not ok(arm(tau=None), route={"spec_noise": True})
    assert ok(arm(tau=0.5), route={"spec_noise": True})  # report-only; conc k=1 gates tau (HIPF-615)
    assert judge({"legacy": arm(), "engine": arm(tau=0.985)}, {}, 0.35)["g_spec"] is False  # 1 % A/B bound
    assert judge({"legacy": arm(), "engine": arm(tau=0.995)}, {}, 0.35)["g_spec"] is True
    assert not ok(arm(loop="legacy"), route={"expect_engine": True})  # silent fallback
    assert not ok(arm(loop="bogus"))
    assert not ok(arm(), leg=arm(loop="engine"))
    mixed = arm(loop="engine")
    mixed["cold"][0]["done"].pop("loop")
    assert not ok(mixed)
    assert not ok(arm(), perf={"median": {"g_perf": False}})
    assert ok(arm(), perf={"median": {"g_perf": True}})
    assert perf_ok({"prefill_tok_s_delta_pct": -1.9, "decode_tok_s_delta_pct": 1.9})
    assert not perf_ok({"prefill_tok_s_delta_pct": -2.1, "decode_tok_s_delta_pct": 0.0})
    assert not perf_ok({"prefill_tok_s_delta_pct": 0.0, "decode_tok_s_delta_pct": 2.1})
    assert not perf_ok({"prefill_tok_s_delta_pct": None, "decode_tok_s_delta_pct": 0.0})
    assert ok(arm(), perf={"median": {"g_perf": "intended"}})
    gain = {"prefill_tok_s_delta_pct": 0.0, "decode_tok_s_delta_pct": 2.3}
    assert perf_ok(gain, decode_gain=True) == "intended" and not perf_ok(gain)
    assert not perf_ok({"prefill_tok_s_delta_pct": 0.0, "decode_tok_s_delta_pct": -2.1}, decode_gain=True)
    assert not perf_ok({"prefill_tok_s_delta_pct": -2.1, "decode_tok_s_delta_pct": 2.3}, decode_gain=True)
    assert not ok(arm(iso="ax"), leg=arm(iso="ax"))  # state leaked into a later request
    part = arm(iso="ax")
    part["cold"][0]["done"]["prefill_tokens"] = 5
    part["isolation"][0]["done"]["cached_tokens"] = 1  # partial reuse: not judged
    assert isolation_check(part) == UNEXERCISED and ok(part, leg=part)
    part["isolation"][0]["done"]["cached_tokens"] = 5  # the whole prompt restored: judged
    assert isolation_check(part) is False
    conc_self_test()
    print("self-test ok")


def conc_self_test():
    # wire parsing: warmups skipped, k from the sample number, turn from attempt_id
    order = bench_order(["stateless", "multiturn"], [1, 4], 2)
    assert order[:3] == [("stateless", 1), ("stateless", 4), ("stateless", 1)] and order[4] == ("multiturn", 1)
    w = [json.dumps(x) for x in (
        {"type": "token", "id": "bench-noslots-2-W1-0", "text": "w"},
        {"type": "done", "id": "bench-noslots-2-W1-0", "attempt_id": 1},
        {"type": "token", "id": "bench-noslots-2-Q1-3", "text": "a"},
        {"type": "committed", "id": "bench-noslots-2-Q1-3", "tok_id": 7, "pos": 9, "t_ms": 1},
        {"type": "done", "id": "bench-noslots-2-Q1-3", "attempt_id": 1, "cached_tokens": 0},
        {"type": "token", "id": "bench-noslots-5-Q1-3", "text": "b"},
        {"type": "done", "id": "bench-noslots-5-Q1-3", "attempt_id": 2, "cached_tokens": 4})] + ["junk"]
    got = wire_requests(w, order)
    assert sorted(got) == ["multiturn/1/1/3/1", "stateless/4/1/3/0"], got
    assert got["stateless/4/1/3/0"]["ids"] == [[7, 9]] and got["stateless/4/1/3/0"]["text"] == "a"
    # metrics: tok/s over the sample wall, TTFT from t_send (queue wait included)
    s = lambda t_send, t_first, t_end, tokens: {"t_send": t_send, "t_first": t_first, "t_end": t_end,
                                                 "tokens": tokens, "ok": True}
    m = cell_metrics({"status": "ok", "rejected_samples": [], "samples": [
        {"overlap": {}, "streams": [s(0, 1, 3, 11), s(0, 2, 4, 11)]}]})
    assert (m["out_tok_s"], m["ttft_p50_ms"], m["tpot_p50_ms"]) == (5.5, 1500.0, 200.0), m
    assert abs(pctl([1, 2, 3, 4], 0.99) - 3.97) < 1e-9

    # gates
    def req(text="ab", cached=0, tau=None, loop="engine", typ="done"):
        done = {k: v for k, v in (("cached_tokens", cached), ("tau", tau), ("loop", loop)) if v is not None}
        return result(typ, [["t", c] for c in text], [[ord(c), i] for i, c in enumerate(text)], None, done)

    def conc_arm(loop, backend, tok_s=30.0, ttft=1000.0, k4=None, cached=5, tau=None, rc=0, stamp=None, mt4=None):
        reqs, stamp = {}, stamp or loop
        for wl in ("stateless", "multiturn"):
            for k in (1, 4):
                reqs[f"{wl}/{k}/1/0/0"] = req(tau=tau, loop=stamp)
                if wl == "multiturn":
                    reqs[f"{wl}/{k}/1/0/1"] = req("cd", cached=cached, tau=tau, loop=stamp)
        if k4:
            reqs["stateless/4/1/0/0"] = k4
        if mt4:
            reqs["multiturn/4/1/0/1"] = mt4
        cells = {f"{wl}/{k}": {"out_tok_s": tok_s, "ttft_p50_ms": ttft, "ttft_p99_ms": ttft}
                 for wl in ("stateless", "multiturn") for k in (1, 4)}
        return {"loop": loop, "backend": backend, "rc": rc, "cells": cells, "reqs": reqs}

    def cj(eng=None, leg=None, batch=None, c=None, expect=True):
        arms = {"legacy-noslots": leg or conc_arm("legacy", "noslots"),
                "engine-noslots": eng or conc_arm("engine", "noslots"),
                # fn-lanes are not row-exact: a legacy arm's own C1 != CN is never gated
                "legacy-batch": batch or conc_arm("legacy", "batch", tok_s=60.0, ttft=900.0,
                                                  k4=req("XY", loop="legacy"))}
        return judge_conc(arms, {"target": ["legacy", "batch"]} if c is None else c, expect)
    good = conc_arm("engine", "noslots", tok_s=61.0, ttft=900.0)
    assert cj(eng=good)["ok"], cj(eng=good)
    assert not cj()["ok"]  # serialized engine: below the fn-lanes target
    assert cj(c={})["ok"]  # no target: correctness gates only
    c1 = cj(eng=conc_arm("engine", "noslots", k4=req("ax")), leg=conc_arm("legacy", "noslots", k4=req("ax")), c={})
    assert c1["fails"] == ["engine-noslots: C1 != CN 1"], c1["fails"]
    assert not cj(eng=conc_arm("engine", "noslots", k4=req("ax")), c={})["ok"]  # != ref
    bad_ids = req()
    bad_ids["ids"] = [[0, 0]]
    assert not cj(eng=conc_arm("engine", "noslots", k4=bad_ids), c={})["ok"]
    assert not cj(eng=conc_arm("engine", "noslots", stamp="legacy"), c={})["ok"]  # silent fallback
    assert cj(eng=conc_arm("engine", "noslots", stamp="legacy"), c={}, expect=False)["ok"]
    assert not cj(eng=conc_arm("engine", "noslots", k4=req(typ="error")), c={})["ok"]
    assert not cj(eng=conc_arm("engine", "noslots", cached=4), c={})["ok"]  # reuses less than legacy
    assert cj(eng=conc_arm("engine", "noslots", cached=9), c={})["ok"]
    unex = cj(eng=conc_arm("engine", "noslots", cached=0), leg=conc_arm("legacy", "noslots", cached=0), c={})
    assert unex["ok"] and unex["unexercised"]
    assert not cj(eng=conc_arm("engine", "noslots", tau=2.9), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    assert cj(eng=conc_arm("engine", "noslots", tau=2.98), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    # k=1 bound is 2 % (HIPF-619 S1)
    assert not cj(eng=conc_arm("engine", "noslots", tau=2.93), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    assert cj(eng=conc_arm("engine", "noslots", tau=2.95), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    # report-only A/B G-spec needs a gating conc k=1 cell (HIPF-619 B1, HIPF-624 S1)
    sn, jab = {"spec_noise": True}, {"g_spec": True}
    gated = cj(eng=conc_arm("engine", "noslots", tau=3.0), leg=conc_arm("legacy", "noslots", tau=3.0), c={})
    no_k1 = {"engine": {"engine-noslots": {"g_spec": {}}}}
    assert spec_unexercised(sn, jab, None) == ["G-spec: report-only and no conc k=1 cell"]
    assert spec_unexercised(sn, jab, gated) == []
    assert spec_unexercised(sn, jab, no_k1) == ["G-spec: report-only and no conc k=1 cell"]
    assert spec_unexercised({}, jab, None) == [] and spec_unexercised(sn, {"g_spec": None}, None) == []

    # k>1 lane tau vs a stored engine baseline (HIPF-604): bound -15 %
    def runs3(loop, tau4, runs=3, tau2=None):
        a = conc_arm(loop, "noslots", tau=3.0)
        for key in [f"{wl}/{k}/{n}/0/0" for wl in ("stateless", "multiturn") for k in (1, 2, 4)
                    for n in range(1, runs + 1)]:
            if key not in a["reqs"] and (tau2 is not None or key.split("/")[1] != "2"):
                a["reqs"][key] = req(tau=3.0, loop=loop)
        if tau2 is not None:
            a["cells"].update({f"{wl}/2": a["cells"][f"{wl}/4"] for wl in ("stateless", "multiturn")})
        for key, x in a["reqs"].items():
            if loop == "engine" and key.split("/")[1] in ("2", "4"):
                x["done"]["tau"] = tau4 if key.split("/")[1] == "4" else tau2
        return a
    lt = {"lane_tau": {"bench": bench_params({"runs": 3}),
                       "arms": {"engine-noslots": {"stateless/4": {"mean": 2.0}, "multiturn/4": {"mean": 2.0}}}}}
    lane = lambda eng, c=lt: cj(eng=eng, leg=runs3("legacy", 3.0), c={"runs": 3, **c})
    assert lane(runs3("engine", 1.71))["ok"]
    assert lane(runs3("engine", 1.69))["fails"] == ["engine-noslots: lane tau 2"]
    assert lane(runs3("engine", 2.0))["engine"]["engine-noslots"]["lane_tau"]["multiturn/4"]["n"] == [4, 4]
    no_tau = runs3("engine", 2.0)
    no_tau["reqs"]["multiturn/4/2/0/0"]["done"].pop("tau")
    assert lane(no_tau)["fails"] == ["engine-noslots: lane tau 1"]  # every lane request must report tau
    short = cj(eng=runs3("engine", 1.0, runs=2), leg=runs3("legacy", 3.0, runs=2), c={**lt, "runs": 2})
    assert short["ok"] and short["unexercised"] == ["engine-noslots: lane tau baseline bench params differ"], short
    for diff in ({"args": ["--max-tokens", "64"]}, {"workloads": ["stateless"]}):
        assert lane(runs3("engine", 1.0), {**lt, **diff})["unexercised"] == [
            "engine-noslots: lane tau baseline bench params differ"], diff
    lt2 = {"lane_tau": {**lt["lane_tau"], "arms": {"engine-noslots": {
        **lt["lane_tau"]["arms"]["engine-noslots"], "stateless/2": {"mean": 2.0}, "multiturn/2": {"mean": 2.0}}}}}
    k2 = lambda t: cj(eng=runs3("engine", 2.0, tau2=t), leg=runs3("legacy", 3.0, tau2=3.0), c={"runs": 3, **lt2})
    assert k2(1.71)["ok"] and k2(1.69)["fails"] == ["engine-noslots: lane tau 2"]  # k=2 cells gate too
    one = lane(runs3("engine", 1.0), {"lane_tau": {**lt["lane_tau"], "arms": {"engine-noslots": {
        "stateless/4": {"mean": 1.0}}}}})
    assert one["ok"] and one["unexercised"] == ["engine-noslots: lane tau multiturn/4 not in the baseline"], one
    assert lane(runs3("engine", 1.0), {"lane_tau": {**lt["lane_tau"], "arms": {}}})["unexercised"] == [
        "engine-noslots: no lane tau baseline"]
    assert lane(runs3("engine", 1.0), {**lt, "k": "1,4"})["unexercised"] == [
        "engine-noslots: lane tau baseline bench params differ"]
    assert not lane(runs3("engine", 1.0), {})["fails"]  # no lane_tau: k>1 tau report-only
    assert not cj(eng=conc_arm("engine", "noslots", tau=None), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    # k>1 tau is report-only: lanes accept fewer drafts per window than the singleton ref (HIPF-604/615)
    lanes4 = conc_arm("engine", "noslots", tau=2.9)
    for wl in ("stateless", "multiturn"):
        for k, x in lanes4["reqs"].items():
            if k.startswith(f"{wl}/4/"):
                x["done"]["tau"] = 2.2
    k4 = cj(eng=lanes4, leg=conc_arm("legacy", "noslots", tau=2.9), c={})
    assert k4["ok"] and k4["engine"]["engine-noslots"]["tau"]["stateless/4"]["pass"] is False, k4
    assert not cj(eng=conc_arm("engine", "noslots", tok_s=58.0, ttft=900.0))["ok"]  # throughput below target
    assert not cj(eng=conc_arm("engine", "noslots", tok_s=61.0, ttft=950.0))["ok"]  # TTFT above target
    assert not cj(eng=good, batch=conc_arm("legacy", "batch", rc=1))["ok"]  # an arm that did not run
    assert cj(eng=good)["engine"]["engine-noslots"]["target"] == {"stateless/4": True, "multiturn/4": True}
    # legacy drops the follow-up's prefix at k=4 and prefills it cold (other tokens); an
    # engine that keeps it is compared with the legacy request that reused as much (k=1)
    cold4 = conc_arm("legacy", "noslots", mt4=req("zz", cached=0, loop="legacy"))
    kept = cj(leg=cold4, c={})
    assert kept["ok"] and not kept["unexercised"] and kept["reference_c1_ne_cn"]["legacy-noslots"] == 1, kept
    lost = cj(eng=conc_arm("engine", "noslots", mt4=req("zz", cached=0)), leg=cold4, c={})
    assert lost["fails"] == ["engine-noslots: C1 != CN 1", "engine-noslots: cached_tokens < C1 1"], lost["fails"]
    odd = cj(eng=conc_arm("engine", "noslots", mt4=req("cd", cached=7)), leg=cold4, c={})
    assert odd["ok"] and odd["engine"]["engine-noslots"]["ref_reuse_unmatched"] == ["multiturn/4/1/0/1"]
    # a lost record: the k=4 follow-up reuses less than at k=1 (the legacy ref reused 0 there),
    # unless the engine logged that it refused the record
    shorter = conc_arm("engine", "noslots", mt4=req("cd", cached=3))
    lost_rec = cj(eng=shorter, leg=cold4, c={})
    assert lost_rec["fails"] == ["engine-noslots: cached_tokens < C1 1"], lost_rec["fails"]
    shorter["refused"] = ["multiturn/4/1/0/1"]
    exempt = cj(eng=shorter, leg=cold4, c={})
    assert exempt["ok"] and exempt["engine"]["engine-noslots"]["refused"] == ["multiturn/4/1/0/1"]
    below_ref = conc_arm("engine", "noslots", cached=0)  # legacy-noslots reused 5 on every follow-up
    assert not cj(eng=below_ref, c={})["ok"]
    below_ref["refused"] = [k for k in below_ref["reqs"] if k.startswith("multiturn") and k.endswith("/1")]
    assert not [f for f in cj(eng=below_ref, c={})["fails"] if "< ref" in f]  # refused: exempt vs ref too
    # C1 != CN exemptions: a refused follow-up (cold), or an earlier turn that reused more than at k=1
    cold_ref = conc_arm("engine", "noslots", mt4=req("zz", cached=0))
    cold_ref["refused"] = ["multiturn/4/1/0/1"]
    ex = cj(eng=cold_ref, leg=cold4, c={})
    assert ex["ok"] and ex["engine"]["engine-noslots"]["c1_ne_cn_exempt"] == {"multiturn/4/1/0/1": "refused"}, ex
    shared = conc_arm("engine", "noslots", mt4=req("zz", cached=5))
    shared["reqs"]["multiturn/4/1/0/0"] = req(cached=3)  # turn 0 reused a shared prefix C1 ran cold
    sx = cj(eng=shared, leg=cold4, c={})["engine"]["engine-noslots"]
    assert sx["c1_ne_cn"] == [] and "multiturn/4/1/0/0 reused 3 > C1 0" in sx["c1_ne_cn_exempt"]["multiturn/4/1/0/1"]

    # refusal tags: tag N = the N-th gen_start; the refused request is the record's next turn
    ev = lambda t, rid, att=1, **kw: json.dumps({"type": t, "id": rid, "attempt_id": att, **kw})
    bat = {"continuous_batch": {"executed": True}}
    wire = [ev("gen_start", "bench-batch-1-W1-0"), ev("done", "bench-batch-1-W1-0", **bat),
            ev("gen_start", "bench-batch-2-Q1-3"), ev("gen_start", "bench-batch-2-Q1-5"),  # tags 2, 3
            ev("done", "bench-batch-2-Q1-3", **bat), ev("gen_start", "bench-batch-2-Q1-3", 2),
            ev("done", "bench-batch-2-Q1-5", **bat), ev("done", "bench-batch-2-Q1-3", 2, **bat)]
    log = ("[engine] lane record 2 not continued: lane record 2: the MTP head is at 7, the trunk at 9\n"
           "[engine] lane record 3 not continued: x\n[engine] lane record 1 not continued: x\n"
           "[engine] lane record 99 not continued: x\n")
    order = bench_order(["multiturn"], [1, 4], 1)
    assert refused_followups(wire, log, order) == (["multiturn/4/1/3/1"], [3, 99])
    singleton = wire[:-1] + [ev("done", "bench-batch-2-Q1-3", 2, execution_mode="sequential")]
    assert refused_followups(singleton, log, order) == ([], [2, 3, 1, 99])  # no tag order to trust

    batch_self_test()


def batch_self_test():
    def row(text="ab", typ="done", cached=0, prefill=5, finish="stop", batched=True, loop="engine", mt=64):
        done = {"cached_tokens": cached, "prefill_tokens": prefill, "finish_reason": finish, "loop": loop,
                "continuous_batch": {"executed": batched}}
        return result(typ, [["t", c] for c in text], [[ord(c), i] for i, c in enumerate(text)], mt, done)
    assert hist_len(row("abc", cached=2, prefill=5)) == 10  # prompt 7 + 3 committed ids
    assert hist_len({"ids": [], "done": {"prefill_tokens": 5, "tokens": 3, "finish_reason": "stop"}}) == 9

    def rows(**over):
        base = {"P0": row(), "P~0": row(), "D0": row(), "D~0": row(),  # turn 0 commits 5 + 2 = 7
                "P1": row("", finish="aborted"), "D1": row("x" * 8, finish="aborted", mt=256),
                "P~1": row("cd", cached=7), "D~1": row("ef", cached=7),
                "P1r": row("cd", cached=7), "D1r": row("ef", cached=7),  # turn 1 commits 7 + 5 + 2 = 14
                "D2": row("x" * 8, finish="aborted", mt=256), "P2": row("gh", cached=14),
                "D2r": row("ij", cached=14), "D~2": row("ij", cached=14),  # turn 2 commits 14 + 5 + 2 = 21
                "D3": row("kl", finish="aborted"), "P3": row("mn", cached=21),
                "D3r": row("kl", cached=21), "D~3": row("kl", cached=21), "daemon_rc": 0}
        return {**base, **over}
    good = batch_checks(rows(), True)
    assert good["ok"] and not good["unexercised"] and not good["notes"], good
    assert good["reuse"]["P1r"] == [7, 7]
    assert batch_checks(rows(P1r=row("cd", cached=0)), True)["fails"] == [
        "P1r reused 0 < 7 (its committed turn 0)", "P1r reused 0 < control P~1 7 (the abort lost reuse)"]
    assert batch_checks(rows(**{"D~1": row("ef", cached=3)}), True)["fails"] == ["D~1 reused 3 < 7 (its committed turn 0)"]
    assert batch_checks(rows(D1r=row("eX", cached=7)), True)["fails"] == ["D1r tokens differ from D~1 at the same reuse"]
    assert batch_checks(rows(D2r=row("ij", cached=0)), True)["fails"] == [
        "D2r reused 0 < 14 (its committed turn 1)", "D2r reused 0 < control D~2 14 (the abort lost reuse)"]
    # a retry below its control fails even where the turn floor holds (a control that reused more)
    assert batch_checks(rows(**{"D~3": row("kl", cached=30)}), True)["fails"] == [
        "D3r reused 21 < control D~3 30 (the abort lost reuse)"]
    # "prompt" mode (Gemma 4): any reuse > 0 and >= legacy's; the committed turn is not required
    gem = rows(**{"P~1": row("cd", cached=3), "P1r": row("cd", cached=3)})
    leg = rows(**{"P~1": row("cd", cached=3), "D~1": row("ef", cached=2), "P1r": row("cd", cached=3)})
    assert batch_checks(gem, True)["fails"] == ["P~1 reused 3 < 7 (its committed turn 0)",
                                                "P1r reused 3 < 7 (its committed turn 0)"]
    ok_g = batch_checks(gem, True, "prompt", leg)
    assert ok_g["ok"] and ok_g["reuse"]["D~1"] == [7, 7, 2], ok_g
    assert batch_checks(rows(**{"D~1": row("ef", cached=1), "D1r": row("ef", cached=1)}), True, "prompt", leg
                        )["fails"] == ["D~1 reused 1 < legacy 2", "D1r reused 1 < legacy 7"]
    assert batch_checks(rows(**{"P~1": row("cd", cached=0)}), True, "prompt")["fails"] == ["P~1 reused no prefix"]
    assert batch_checks(rows(D2r=row("iX", cached=14)), True)["fails"] == ["D2r tokens differ from D~2 at the same reuse"]
    more = batch_checks(rows(D1r=row("eX", cached=9), D2r=row("ij", cached=16), D3r=row("kl", cached=23)),
                        True)  # more than the control: not compared
    assert more["ok"] and more["notes"]
    assert batch_checks(rows(**{"P~0": row("aX")}), True)["fails"] == ["twins P0/P~0 differ at turn 0"]
    assert not batch_checks(rows(P0=row(loop="legacy")), True)["ok"]
    assert batch_checks(rows(P0=row(loop="legacy")), False)["ok"]
    assert not batch_checks(rows(D1r=row(typ="error")), True)["ok"]
    assert not batch_checks(rows(daemon_rc=1), True)["ok"]
    late = batch_checks(rows(P1=row("ab", finish="aborted"), D1=row("x" * 200, finish="aborted", mt=256)), True)
    assert late["ok"] and len(late["unexercised"]) == 2
    assert batch_checks(rows(D1=row("x" * 9, finish="length", mt=256)), True)["unexercised"]
    assert batch_checks(rows(P1r=row("cd", cached=7, batched=False)), True)["unexercised"]
    noted = batch_checks(rows(refused=["P0", "P~0"]), True)
    assert noted["ok"] and noted["notes"] == ["lane records not continued (engine log): P0, P~0"]
    starts = [json.dumps({"type": "gen_start", "id": i}) for i in ("P0", "P~0", "D0")] + ['{"type":"token","id":"P0"}']
    assert refused_records(starts, "[engine] lane record 2 not continued: x\n[engine] lane record 7 not continued: y\n"
                           ) == ["P~0", "tag 7"]


def host_state(own_pid=None):
    kfd = []
    for pdir in Path("/proc").glob("[0-9]*"):
        try:
            if pdir.name != str(own_pid) and any(
                    os.readlink(f).startswith("/dev/kfd") for f in (pdir / "fd").iterdir()):
                kfd.append(f"{pdir.name}:{(pdir / 'comm').read_text().strip()}")
        except OSError:
            pass
    sclk = []
    for f in Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"):
        sclk += [ln.split(":")[1].strip(" *") for ln in f.read_text().splitlines() if ln.endswith("*")]
    return {"loadavg": os.getloadavg(), "kfd": kfd, "gfxclk": sclk}


def wait_gpu_released(timeout_s=120):
    """Wait until no hipfire GPU reservation lock is held: the previous arm's daemon can still
    hold it after its bench returned (`FATAL ... already reserved`, Loki HIPF-557 N3)."""
    deadline = time.time() + timeout_s
    for path in [*Path("/run/lock/hipfire").glob("gpu-*.lock"), *Path("/tmp/hipfire-locks").glob("gpu-*.lock")]:
        try:
            f = open(path)
        except OSError:
            continue
        with f:
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        return
                    time.sleep(0.5)


def perf_ok(m, decode_gain=False):
    """G-perf: both median tok/s deltas within +-2 %. With the route's
    `perf_decode_gain` (intended diff 7) a decode gain above +2 % is "intended"."""
    p, d = m.get("prefill_tok_s_delta_pct"), m.get("decode_tok_s_delta_pct")
    if p is None or d is None or abs(p) > 2.0 or d < -2.0:
        return False
    return True if d <= 2.0 else ("intended" if decode_gain else False)


def run_perf(args, route, out):
    """Interleaved fresh loads, median of --perf-reps per arm."""
    rows = {arm: [] for arm in ARMS}
    for rep in range(args.perf_reps):
        for arm in ARMS:
            env = dict(os.environ, **route.get("env", {}), HIPFIRE_ENGINE_LOOP=arm)
            d = Daemon(args.daemon, env, f"{out}/{route['name']}.perf.{arm}.{rep}", args.timeout)
            try:
                d.load(route["model"], route.get("load", {}))
                d.generate("warm", greedy("Hello.", 8))
                host = host_state(d.p.pid)
                # no ignore_eos on the wire: the counting tail keeps decode going to the budget
                r = d.generate("perf", greedy(LONG + "\nThen count from 1 to 500, comma separated.",
                                              route.get("perf_decode_tokens", 256)))
            finally:
                d.close()
            dn = r["done"]
            # Routes without the *_tok_s fields: same prompt on both arms, so a rate
            # from prefill_ms (and tokens/decode_ms) gives the same relative delta.
            pre = dn.get("prefill_tok_s") or (1000 / dn["prefill_ms"] if dn.get("prefill_ms") else None)
            dec = dn.get("decode_tok_s") or (1000 * dn["tokens"] / dn["decode_ms"] if dn.get("decode_ms") else None)
            rows[arm].append({"prefill_tok_s": pre, "decode_tok_s": dec, "host": host,
                              "prefill_field": "prefill_tok_s" if dn.get("prefill_tok_s") else "1000/prefill_ms"})
    summary = {}
    for arm in ARMS:
        summary[arm] = {k: statistics.median(x[k] for x in rows[arm] if x[k] is not None)
                        if any(x[k] is not None for x in rows[arm]) else None
                        for k in ("prefill_tok_s", "decode_tok_s")}
    for k in ("prefill_tok_s", "decode_tok_s"):
        lg, en = summary["legacy"][k], summary["engine"][k]
        summary[f"{k}_delta_pct"] = round(100 * (en - lg) / lg, 2) if lg and en else None
    summary["g_perf"] = perf_ok(summary, route.get("perf_decode_gain", False))
    return {"reps": rows, "median": summary}


# ── Concurrency matrix (`--conc`, HIPF-472) ─────────────────────────────────
# `hipfire bench --concurrency` per (loop, backend) arm, the daemon behind a wire
# tap: the bench JSON gives the timing, the tapped wire every request's tokens,
# token ids and done envelope.
TAP = '#!/usr/bin/env bash\n"$TAP_DAEMON" "$@" | tee -a "$TAP_OUT"\nexit "${PIPESTATUS[0]}"\n'
WORKLOADS = ("stateless", "multiturn")  # bench's order for `--workload both`


def bench_order(workloads, points, runs):
    """(workload, k) of bench sample n at index n-1: one backend per bench process,
    workloads in turn, k interleaved within each round (bench_concurrency.rs
    sweep_backend / sweep_order)."""
    return [(wl, k) for wl in workloads for _ in range(runs) for k in points]


def wire_requests(lines, order):
    """Measured requests from a tapped wire, keyed `workload/k/round/conv/turn`.
    Bench ids are `bench-<backend>-<sample>-<nonce>-<conv>`, attempt = turn + 1."""
    live, out = {}, {}
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = e.get("id")
        if not isinstance(rid, str) or not rid.startswith("bench-"):
            continue
        sample, nonce, conv = rid.split("-")[-3:]
        if not nonce.startswith("Q"):  # W = warmup
            continue
        t = e.get("type")
        cur = live.setdefault(rid, ([], []))
        if t in ("token", "reasoning"):
            cur[0].append([t[0], e.get("text", "")])
        elif t == "committed":
            cur[1].append([e.get("tok_id"), e.get("pos")])
        elif t in ("done", "error"):
            del live[rid]
            wl, k = order[int(sample) - 1]
            done = {x: v for x, v in e.items() if x not in ("type", "id", "attempt_id")}
            out[f"{wl}/{k}/{nonce[1:]}/{conv}/{e.get('attempt_id', 1) - 1}"] = result(t, *cur, None, done)
    return out


REFUSAL = re.compile(r"\[engine\] lane record (\d+) not continued")


def refused_followups(lines, log_text, order):
    """Measured follow-ups whose lane record the engine logged as not continued
    (`[engine] lane record <tag> not continued`). A tag is the engine's admission
    number (engine/serve.rs `NEXT_TAG`, from 1, taken right before the request's
    `gen_start`), so tag N is the N-th `gen_start` on the wire: the turn that left
    the record. The refused request is the first later one with the same round
    nonce and conversation and the next turn. Returns (keys as wire_requests,
    unmapped tags); every tag is unmapped unless every `done` on the wire ran
    batched, because a singleton request takes no tag."""
    starts, all_batched = [], True
    for line in lines:
        if '"gen_start"' not in line and '"done"' not in line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "gen_start":
            starts.append((str(e.get("id")), e.get("attempt_id", 1)))
        elif e.get("type") == "done":
            all_batched = all_batched and bool((e.get("continuous_batch") or {}).get("executed"))
    tags = [int(t) for t in REFUSAL.findall(log_text)]
    if not all_batched:
        return [], tags
    keys, unmapped = set(), []
    for tag in tags:
        if not 0 < tag <= len(starts):
            unmapped.append(tag)
            continue
        rid, att = starts[tag - 1]
        nonce, conv = rid.split("-")[-2:]
        if not nonce.startswith("Q"):  # a warmup's record: no measured follow-up
            continue
        nxt = next((r for r, a in starts[tag:] if r.split("-")[-2:] == [nonce, conv] and a == att + 1), None)
        if nxt is None:
            unmapped.append(tag)
            continue
        wl, k = order[int(nxt.split("-")[-3]) - 1]
        keys.add(f"{wl}/{k}/{nonce[1:]}/{conv}/{att}")
    return sorted(keys), unmapped


def pctl(xs, p):
    """Linear-interpolated percentile (numpy's default, as bench_concurrency.rs)."""
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def cell_metrics(cell):
    """Median over a bench cell's samples of output tok/s and per-sample TTFT/TPOT
    p50/p99 (ms). Overlap-rejected samples count too: a loop that serializes is
    measured as it is. TTFT runs from t_send, so a client-side queue wait counts."""
    rows = []
    for s in cell["samples"] + cell["rejected_samples"]:
        st = [x for x in s["streams"] if x["ok"] and x["t_first"] is not None]
        if not st:
            continue
        ttft = [x["t_first"] - x["t_send"] for x in st]
        tpot = [(x["t_end"] - x["t_first"]) / (x["tokens"] - 1) for x in st if x["tokens"] > 1]
        wall = max(x["t_end"] for x in s["streams"]) - min(x["t_send"] for x in s["streams"])
        row = {"out_tok_s": sum(x["tokens"] for x in st) / wall if wall > 0 else None,
               "decode_concurrency": s["overlap"].get("decode_concurrency")}
        for name, xs in (("ttft", ttft), ("tpot", tpot)):
            for p in (50, 99):
                v = pctl(xs, p / 100)
                row[f"{name}_p{p}_ms"] = None if v is None else 1000 * v
        rows.append(row)
    m = {"status": cell["status"], "samples": len(rows)}
    for key in ("out_tok_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "decode_concurrency"):
        xs = [r[key] for r in rows if r[key] is not None]
        m[key] = round(statistics.median(xs), 2) if xs else None
    return m


def run_conc(args, route, out):
    """One `hipfire bench --concurrency` process per `conc.arms` entry."""
    c = route["conc"]
    points = sorted({int(x) for x in str(c.get("k", "1,2,4")).split(",")})
    workloads = [w for w in WORKLOADS if w in c.get("workloads", WORKLOADS)]
    runs = c.get("runs", 3)
    tap = out / "daemon-tap.sh"
    tap.write_text(TAP)
    tap.chmod(0o755)
    arms = {}
    for loop, backend in c.get("arms", [["legacy", "noslots"], ["engine", "noslots"]]):
        base = out / f"{route['name']}.conc.{loop}-{backend}"
        wire = Path(f"{base}.wire.jsonl")
        wire.write_text("")
        env = {**arm_env(route, loop), "HIPFIRE_DAEMON_BIN": str(tap),
               "TAP_DAEMON": os.path.realpath(args.daemon), "TAP_OUT": str(wire)}
        cmd = [args.hipfire, "bench", route["model"], "--concurrency", ",".join(map(str, points)),
               "--backend", backend, "--workload", workloads[0] if len(workloads) == 1 else "both",
               "--runs", str(runs), "-j", *c.get("args", [])]
        wait_gpu_released()
        host = host_state()
        with open(f"{base}.json", "w") as fo, open(f"{base}.log", "w") as fe:
            rc = subprocess.run(cmd, env=env, stdout=fo, stderr=fe).returncode
        try:
            cells = json.loads(Path(f"{base}.json").read_text())["cells"]
        except (json.JSONDecodeError, KeyError):
            cells = []
        with wire.open() as f:
            reqs = wire_requests(f, bench_order(workloads, points, runs))
        with wire.open() as f:
            refused, unmapped = refused_followups(f, Path(f"{base}.log").read_text(errors="replace"),
                                                  bench_order(workloads, points, runs))
        arms[f"{loop}-{backend}"] = {
            "loop": loop, "backend": backend, "rc": rc, "cmd": " ".join(cmd), "host": host,
            "cells": {f"{x['workload']}/{x['k']}": cell_metrics(x) for x in cells}, "reqs": reqs,
            "refused": refused, "refusals_unmapped": unmapped}
    return arms


def judge_conc(arms, c, expect_engine):
    """Concurrency gates on every engine arm; legacy arms are reference rows
    (legacy fn-lanes are not row-exact by default, HIPF-469). Pure: --self-test
    drives it with fakes. Per engine arm:
    - every request ends in `done`; `loop` stamps as in loop_checks
    - C1 == CN: each k>1 request's tokens and ids equal the same request at k=1, except
      a known reuse difference (`c1_ne_cn_exempt`): the request's lane record was
      refused, or an earlier turn of its conversation reused more at k>1 than at k=1
      (a prefix shared across conversations, Loki HIPF-597 N1); a cold prefill is not
      token-exact with a reused prefix
    - G-tok: tokens and ids equal the `ref` arm's (default legacy-noslots)
    - cache: each multi-turn follow-up reuses at least the ref arm's `cached_tokens`,
      and at k>1 at least the same request at k=1 (C1), unless the engine logged
      that it refused to continue its lane record (`refused`, refused_followups)
    - G-spec: mean tau per k=1 cell >= ref - 2 %. k>1 cells are not judged against
      ref: lane tau measures the lane window policy, which differs between engine
      lanes and legacy fn-lanes (HIPF-615)
    - lane tau (k>1, with `lane_tau`, a stored engine baseline): every request of the
      cell reports tau, and the mean over its runs is >= the baseline mean - 15 %
    - target (k>1): out tok/s >= target - tol, TTFT p50 and p99 <= target + tol"""
    tol = c.get("tol_pct", 2.0) / 100
    ref = arms.get("-".join(c.get("ref", ["legacy", "noslots"])))
    target = arms.get("-".join(c["target"])) if c.get("target") else None
    fails, unexercised, report = [], [], {}
    if ref is None or (c.get("target") and target is None):
        fails.append("ref or target arm missing")
    for tag, a in arms.items():
        if a["rc"] != 0 or not a["cells"] or not a["reqs"]:
            fails.append(f"{tag}: bench rc {a['rc']}, {len(a['cells'])} cells, {len(a['reqs'])} requests")
    same = lambda x, y: x is not None and y is not None and (x["events"], x["ids"]) == (y["events"], y["ids"])
    cached = lambda x: (x or {"done": {}})["done"].get("cached_tokens") or 0
    at = lambda key, k: "/".join([key.split("/")[0], str(k), *key.split("/")[2:]])
    c1_ne_cn = lambda reqs: sorted(k for k, x in reqs.items()
                                   if k.split("/")[1] != "1" and not same(x, reqs.get(at(k, 1))))
    # every arm's C1 != CN count; legacy rows are reference only (a legacy multi-turn
    # follow-up that loses its prefix at k>1 prefills cold, which is not token-exact
    # with the reused prefix, HIPF-472)
    reference = {tag: len(c1_ne_cn(a["reqs"])) for tag, a in arms.items() if a["loop"] != "engine"}
    for tag, a in arms.items():
        if a["loop"] != "engine":
            continue
        reqs, rr = a["reqs"], (ref or {}).get("reqs", {})
        ks = sorted({x.split("/")[1] for x in rr}, key=int)
        r = report[tag] = {}
        r["errors"] = sorted(k for k, x in reqs.items() if x["type"] != "done")
        loops = [x["done"].get("loop") for x in reqs.values() if x["type"] == "done"]
        r["loop"] = {str(v): loops.count(v) for v in set(loops)}
        loop_bad = (any(v not in (None, "legacy", "engine") for v in loops)
                    or (None in loops and any(v is not None for v in loops))
                    or (expect_engine and any(v != "engine" for v in loops)))
        refused = set(a.get("refused", []))
        r["c1_ne_cn"], r["c1_ne_cn_exempt"] = [], {}
        for k in c1_ne_cn(reqs):
            wl, kk, rnd, conv, turn = k.split("/")
            earlier = [f"{wl}/{kk}/{rnd}/{conv}/{t}" for t in range(int(turn))]
            why = ("refused" if k in refused else
                   next((f"{e} reused {cached(reqs.get(e))} > C1 {cached(reqs.get(at(e, 1)))}" for e in earlier
                         if cached(reqs.get(e)) > cached(reqs.get(at(e, 1)))), None))
            if why:
                r["c1_ne_cn_exempt"][k] = why
            else:
                r["c1_ne_cn"].append(k)
        # G-tok: against the ref request with the same prefix reuse (same k first),
        # since a reused prefix and a cold prefill are not token-exact
        refs = {k: next((y for y in (rr.get(at(k, kk)) for kk in [k.split("/")[1], *ks])
                         if y is not None and cached(y) == cached(x)), None) for k, x in reqs.items()}
        r["ne_ref"] = sorted(k for k, x in reqs.items() if k not in rr or (refs[k] and not same(x, refs[k])))
        r["ref_reuse_unmatched"] = sorted(k for k in reqs if k in rr and refs[k] is None)
        if r["ref_reuse_unmatched"]:
            unexercised.append(f"{tag}: {len(r['ref_reuse_unmatched'])} requests reused a prefix length no ref "
                               f"request did (tokens not compared)")
        follow = [k for k in reqs if k.startswith("multiturn/") and not k.endswith("/0")]
        r["cache_below_ref"] = sorted(k for k in follow if k not in refused and cached(reqs[k]) < cached(rr.get(k)))
        # vs C1: legacy-noslots reuses nothing at k>1, so only C1 sees a lost record (Loki HIPF-586 S3)
        r["cache_below_c1"] = sorted(k for k in follow if k.split("/")[1] != "1" and k not in refused
                                     and cached(reqs[k]) < cached(reqs.get(at(k, 1))))
        r["refused"] = sorted(k for k in follow if k in refused)
        r["refusals_unmapped"] = a.get("refusals_unmapped", [])
        r["cache_hits"] = [sum(cached(reqs[k]) > 0 for k in follow), sum(cached(rr.get(k)) > 0 for k in follow),
                           len(follow)]
        if follow and not r["cache_hits"][0]:
            unexercised.append(f"{tag}: no multi-turn follow-up reused a prefix")
        r["g_spec"], r["target"] = {}, {}
        r["tau"], r["lane_tau"] = {}, {}
        lt = c.get("lane_tau")
        base = (lt or {}).get("arms", {}).get(tag)
        if lt and base is None:
            unexercised.append(f"{tag}: no lane tau baseline")
        elif lt and lt.get("bench") != bench_params(c):  # the cell sequence moves lane tau (HIPF-604)
            unexercised.append(f"{tag}: lane tau baseline bench params differ")
            base = None
        mean = lambda t: round(statistics.mean(t), 4) if t else None
        for cell, m in a["cells"].items():
            taus = [[x["done"]["tau"] for k, x in rs.items() if k.startswith(cell + "/") and "tau" in x["done"]]
                    for rs in (reqs, rr)]
            if base is not None and int(cell.split("/")[1]) > 1:
                keys = [k for k in reqs if k.startswith(cell + "/")]
                if cell not in base:
                    unexercised.append(f"{tag}: lane tau {cell} not in the baseline")
                else:
                    # 15 %: clean runs-3 cell means spread up to 3.4 % sd between receipts
                    # (multiturn/2), ~4.2 % sd against a 2-receipt baseline (HIPF-604)
                    r["lane_tau"][cell] = {
                        "mean": mean(taus[0]), "baseline": base[cell]["mean"], "n": [len(taus[0]), len(keys)],
                        "pass": bool(taus[0]) and len(taus[0]) == len(keys)
                        and statistics.mean(taus[0]) >= 0.85 * base[cell]["mean"]}
            if taus[1]:
                ok_ = (len(taus[0]) == len(taus[1])
                       # 2 %: a clean multiturn/1 cell spreads ~2 % per bench run (HIPF-619 S1)
                       and statistics.mean(taus[0]) >= 0.98 * statistics.mean(taus[1]))
                r["tau"][cell] = {"engine_mean": mean(taus[0]), "ref_mean": mean(taus[1]),
                                  "n": [len(taus[0]), len(taus[1])], "pass": ok_}
                if cell.split("/")[1] == "1":
                    r["g_spec"][cell] = ok_
            t = (target or {}).get("cells", {}).get(cell)
            if target and int(cell.split("/")[1]) > 1:
                r["target"][cell] = bool(t) and all(m[x] is not None and t[x] is not None for x in (
                    "out_tok_s", "ttft_p50_ms", "ttft_p99_ms")) and (
                    m["out_tok_s"] >= t["out_tok_s"] * (1 - tol)
                    and m["ttft_p50_ms"] <= t["ttft_p50_ms"] * (1 + tol)
                    and m["ttft_p99_ms"] <= t["ttft_p99_ms"] * (1 + tol))
        for name, bad in (("errors", r["errors"]), ("loop", loop_bad), ("C1 != CN", r["c1_ne_cn"]),
                          ("tokens != ref", r["ne_ref"]), ("cached_tokens < ref", r["cache_below_ref"]),
                          ("cached_tokens < C1", r["cache_below_c1"]),
                          ("G-spec", not all(r["g_spec"].values())),
                          ("lane tau", [k for k, v in r["lane_tau"].items() if not v["pass"]]),
                          ("target", [k for k, v in r["target"].items() if not v])):
            if bad:
                fails.append(f"{tag}: {name}" + (f" {len(bad)}" if isinstance(bad, list) else ""))
    return {"ok": not fails, "fails": fails, "unexercised": unexercised, "engine": report,
            "reference_c1_ne_cn": reference, "cells": {tag: a["cells"] for tag, a in arms.items()}}


def bench_params(c):
    """The conc settings that move a cell's lane tau: cell order, run count (in multiturn
    cells the first run drafts highest), prompts and lengths (HIPF-604)."""
    return {x: c.get(x) for x in ("k", "runs", "workloads", "args")}


def lane_tau_baseline(out, source, receipts):
    """Write the k>1 lane tau baseline (`conc.lane_tau`, HIPF-604) from conc receipts
    of one accepted build: per engine arm and k>1 cell, the mean tau over every
    request of every run, with the receipts' provenance."""
    taus, src, bench = {}, [], set()
    for p in receipts:
        x = json.loads(Path(p).read_text())
        if x["route"]["conc"].get("runs", 3) < 3:
            raise SystemExit(f"{p}: lane tau baseline needs runs >= 3")
        bench.add(json.dumps(bench_params(x["route"]["conc"])))
        src.append({"receipt": os.path.realpath(p), "daemon_md5": x["daemon"]["md5"],
                    "hipfire_md5": x["hipfire"]["md5"], "fixture_sha256": x["fixture_sha256"],
                    "harness_sha256": x["harness"]["sha256"], "verdict": x["verdict"]})
        for tag, a in x["conc"]["arms"].items():
            for key, q in a["reqs"].items() if a["loop"] == "engine" else ():
                wl, k, run = key.split("/")[:3]
                if int(k) > 1:
                    if "tau" not in q["done"]:
                        raise SystemExit(f"{p}: {tag} {key} reports no tau")
                    taus.setdefault(tag, {}).setdefault(f"{wl}/{k}", {}).setdefault((p, run), []).append(
                        q["done"]["tau"])
    if len(bench) != 1 or len({(s["daemon_md5"], s["fixture_sha256"]) for s in src}) != 1:
        raise SystemExit("lane tau baseline: receipts differ in bench params, daemon or fixture")
    arms = {tag: {cell: {"mean": round(statistics.mean(t for v in rs.values() for t in v), 4),
                         "n": sum(map(len, rs.values())), "runs": len(rs),
                         "run_means": [round(statistics.mean(v), 4) for v in rs.values()]}
                  for cell, rs in sorted(cells.items())} for tag, cells in taus.items()}
    Path(out).write_text(json.dumps({"source": source, "bench": json.loads(bench.pop()), "receipts": src,
                                     "arms": arms}, indent=1) + "\n")


def spec_unexercised(route, j, conc):
    """A report-only A/B G-spec (spec_noise) must not pass silently: some conc k=1
    cell has to gate tau (Loki HIPF-619 B1). conc is None without --conc."""
    if route.get("spec_noise") and j["g_spec"] is not None and not (
            conc and any(c["g_spec"] for c in conc["engine"].values())):
        return ["G-spec: report-only and no conc k=1 cell"]
    return []


def sha256_cached(path, out):
    """Fixture sha256, cached on (path, size, mtime) because the files are large."""
    st = os.stat(path)
    cache_f = Path(out) / "sha256-cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    key = f"{os.path.realpath(path)}:{st.st_size}:{st.st_mtime_ns}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while b := f.read(1 << 24):
                h.update(b)
        cache[key] = h.hexdigest()
        cache_f.write_text(json.dumps(cache, indent=1))
    return cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="check the diff rules, no GPU")
    ap.add_argument("--daemon")
    ap.add_argument("--route", help="route JSON file (docs/engine-loop-ab.md)")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, default=8, help="seed sweep size for the distribution check")
    ap.add_argument("--tvd-max", type=float, default=0.35)
    ap.add_argument("--perf", action="store_true", help="also run the perf arm")
    ap.add_argument("--perf-reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800, help="per-event timeout, seconds")
    ap.add_argument("--conc", action="store_true", help="also run the route's concurrency matrix (`conc`)")
    ap.add_argument("--conc-only", action="store_true", help="skip the A/B scenarios; run the concurrency matrix")
    ap.add_argument("--batch", action="store_true", help="also run the batch-path cache scenario")
    ap.add_argument("--batch-only", action="store_true", help="skip the A/B scenarios; run the batch-path scenario")
    ap.add_argument("--hipfire", help="hipfire CLI for --conc (default: next to --daemon)")
    ap.add_argument("--lane-tau-baseline", nargs="+", metavar="ARG",
                    help="OUT SOURCE_SHA CONC_RECEIPT...: write a k>1 lane tau baseline and exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.lane_tau_baseline:
        if len(args.lane_tau_baseline) < 3:
            ap.error("--lane-tau-baseline needs OUT SOURCE_SHA CONC_RECEIPT...")
        return lane_tau_baseline(*args.lane_tau_baseline[:2], args.lane_tau_baseline[2:])
    if not (args.daemon and args.route and args.out):
        ap.error("--daemon, --route and --out are required")
    route = json.loads(Path(args.route).read_text())
    route.setdefault("name", Path(args.route).stem)
    if isinstance((route.get("conc") or {}).get("lane_tau"), str):  # relative to the route file
        route["conc"]["lane_tau"] = json.loads(
            (Path(args.route).parent / os.path.expanduser(route["conc"]["lane_tau"])).read_text())
    conc_on, batch_on = args.conc or args.conc_only, args.batch or args.batch_only
    ab_on = not (args.conc_only or args.batch_only)
    if conc_on and "conc" not in route:
        ap.error("--conc needs a `conc` block in the route file")
    args.hipfire = args.hipfire or str(Path(args.daemon).with_name("hipfire"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = os.path.expanduser(route["model"])
    route["model"] = model

    # identify the harness when it starts: a run takes hours and the file may change under it
    me = Path(__file__).resolve()
    git = lambda *a: subprocess.run(["git", "-C", str(me.parent), *a], capture_output=True, text=True).stdout.strip()
    harness = {"sha256": hashlib.sha256(me.read_bytes()).hexdigest(), "git_head": git("rev-parse", "HEAD"),
               "dirty": bool(git("status", "--porcelain", "--", str(me)))}
    res, t0, j, perf, conc, unex, summary = {}, time.time(), {"ok": True}, None, None, [], []
    if ab_on:
        for arm in ARMS:
            res[arm] = run_arm(args, route, arm, out,
                               None if arm == "legacy" else res["legacy"]["eos_budget"][0]["done"].get("tokens", 0))
        perf = run_perf(args, route, out) if args.perf else None
        j = judge(res, route, args.tvd_max, perf)
        # a pass with unexercised scenarios must not be citable as covering them
        unex = [k for k, v in j["scenarios"].items() if v["status"] == UNEXERCISED]
        unex += [f"isolation ({arm})" for arm, v in j["isolation"].items() if v == UNEXERCISED]
        sampled = j["sampled"]
        summary.append(", ".join(f"{k}={v['status']}" for k, v in j["scenarios"].items()))
        summary.append(f"sampled tvd={sampled['tvd_first8']} repro={sampled['legacy_reproducible']}/"
                       f"{sampled['engine_reproducible']} distinct={sampled['legacy_distinct']}/{sampled['engine_distinct']}")
        summary.append(f"loop={j['loop']['engine']}" + ("" if j["loop_ok"] else " (FAIL)"))
        summary.append(f"isolation={j['isolation']}")
        if j["g_spec"] is not None:
            summary.append(f"G-spec={j['g_spec']}" + (" (report-only)" if route.get("spec_noise") else ""))
        if perf:
            m = perf["median"]
            summary.append(f"G-perf={m['g_perf']} prefill {m['prefill_tok_s_delta_pct']}% "
                           f"decode {m['decode_tok_s_delta_pct']}%")
    if conc_on:
        arms = run_conc(args, route, out)
        conc = judge_conc(arms, route["conc"], route.get("expect_engine", False))
        unex += conc["unexercised"]
        summary.append("conc=" + ("pass" if conc["ok"] else "FAIL: " + "; ".join(conc["fails"])))
    if ab_on:
        unex += spec_unexercised(route, j, conc)
    batch = None
    if batch_on:
        brows = {arm: run_batch(args, route, arm, out) for arm in ARMS}
        # gated on the engine arm; legacy fn-lanes keep no lane records (reference row)
        mode = route.get("batch_reuse", "turn")
        batch = {"engine": batch_checks(brows["engine"], route.get("expect_engine", False), mode, brows["legacy"]),
                 "legacy_reference": batch_checks(brows["legacy"], False, mode), "arms": brows}
        unex += [f"batch: {u}" for u in batch["engine"]["unexercised"]]
        summary.append(f"batch reuse={batch['engine']['reuse']} "
                       + ("pass" if batch["engine"]["ok"] else "FAIL: " + "; ".join(batch["engine"]["fails"])))
    ok = j["ok"] and (conc is None or conc["ok"]) and (batch is None or batch["engine"]["ok"])
    verdict = ("pass-partial" if unex else "pass") if ok else "fail"
    summary = ", ".join(summary)

    receipt = {
        "route": route, "verdict": verdict, "unexercised": unex, **{k: v for k, v in j.items() if k != "ok"},
        "daemon": {"path": os.path.realpath(args.daemon),
                   "md5": hashlib.md5(Path(args.daemon).read_bytes()).hexdigest()},
        "fixture_sha256": sha256_cached(model, out),
        # identifies the harness bytes even when run from a copy outside the repo
        "harness": harness,
        "host": host_state(), "wall_s": round(time.time() - t0, 1), "arms": res,
    }
    if perf:
        receipt["perf"] = perf
    if conc:
        receipt["hipfire"] = {"path": os.path.realpath(args.hipfire),
                              "md5": hashlib.md5(Path(args.hipfire).read_bytes()).hexdigest()}
        receipt["conc"] = {**conc, "arms": arms}
    if batch:
        receipt["batch"] = batch
    parts = [p for p, on in (("conc", conc_on), ("batch", batch_on)) if on]
    rpath = out / f"{route['name']}{'' if ab_on else '.' + '-'.join(parts)}.json"
    rpath.write_text(json.dumps(receipt, indent=1))

    ev = out / "EVIDENCE.md"
    if not ev.exists():
        ev.write_text("| route | harness sha256 | daemon md5 | fixture sha256 | command | receipt | result | disposition |\n"
                      "|---|---|---|---|---|---|---|---|\n")
    with ev.open("a") as f:
        f.write(f"| {route['name']} | {receipt['harness']['sha256'][:12]} | {receipt['daemon']['md5'][:8]} | "
                f"{receipt['fixture_sha256'][:16]} | `{' '.join(sys.argv)}` | {rpath} | {summary} | "
                f"{('current' + (' (unexercised: ' + ', '.join(unex) + ')' if unex else '')) if ok else 'failed'} |\n")
    print(json.dumps({"route": route["name"], "verdict": receipt["verdict"], "summary": summary}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, queue.Empty) as e:
        print(f"engine_loop_ab: {e}", file=sys.stderr)
        sys.exit(2)
