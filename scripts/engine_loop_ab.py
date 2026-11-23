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
        msg = {"type": "generate", "id": rid, "attempt_id": 1, **req}
        self.send(msg)
        if abort_now:
            self.send({"type": "abort", "id": rid, "attempt_id": 1})
        events, ids, n = [], [], 0
        while True:
            e = self.recv()
            if e.get("id") != rid:
                continue
            t = e.get("type")
            if t in ("token", "reasoning"):
                events.append([t[0], e.get("text", "")])
                n += 1
                if abort_at_token is not None and n == abort_at_token:
                    self.send({"type": "abort", "id": rid, "attempt_id": 1})
            elif t == "committed":  # HIPFIRE_EMIT_TOKEN_IDS=1: exact token ids
                ids.append([e.get("tok_id"), e.get("pos")])
            elif t == "commit_ready":
                self.send({"type": "commit", "id": rid, "attempt_id": 1})
            elif t in ("done", "error"):
                return result(t, events, ids, req.get("max_tokens"),
                              {k: v for k, v in e.items() if k not in ("type", "id", "attempt_id")})

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
        hist, turns = [], []
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
        convs = {c: [] for c in "AB"}
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
        cnt = "List the numbers from 1 to 100, one per line, as digits."
        res["abort_decode"] = [d.generate("ad", greedy(cnt, 4 * mt), abort_at_token=8),
                               d.generate("ad-next", greedy(cnt, mt))]
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


def compare(sc, a, b, spec_noise=False):
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
        if sc in ("multi_turn", "interleaved") and i >= (1 if sc == "multi_turn" else 2) and not dx.get("cached_tokens"):
            note(UNEXERCISED, f"[{i}] turn reused no prefix (cached_tokens {dx.get('cached_tokens')!r})")
        if sc == "abort_decode" and i == 1 and not dx.get("cached_tokens"):
            note(UNEXERCISED, "the follow-up after the abort reused no prefix")
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
            for f in ("events", "ids"):
                xs, ys = x.get(f, []), y.get(f, [])
                if xs[:len(ys)] != ys[:len(xs)]:
                    note(DIFF, f"{f} diverge before the abort")
            continue
        if x["events"] != y["events"]:
            note(DIFF, f"[{i}] tokens differ ({x['sha']} vs {y['sha']})")
        if x.get("ids", []) != y.get("ids", []):
            note(DIFF, f"[{i}] token ids differ")
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
        status, notes = compare(sc, res["legacy"][sc], res["engine"][sc], route.get("spec_noise", False))
        scenarios[sc] = {"status": status, "notes": notes}
    sampled = sampled_checks(res)
    sampled["ok"] = (sampled["legacy_reproducible"] and sampled["engine_reproducible"]
                     and sampled["engine_distinct"] >= sampled["legacy_distinct"]
                     and sampled["tvd_first8"] <= tvd_max)
    # G-spec: engine mean tau >= legacy - 1 %; an engine that drops tau fails.
    taus = {arm: [r["done"]["tau"] for sc in GREEDY if sc in res[arm]
                  for r in res[arm][sc][int(sc.startswith("abort")):] if "tau" in r["done"]] for arm in ARMS}
    g_spec = None
    if taus["legacy"]:
        g_spec = (len(taus["engine"]) == len(taus["legacy"])
                  and statistics.mean(taus["engine"]) >= 0.99 * statistics.mean(taus["legacy"]))
    loop_ok, loops = loop_checks(res, route.get("expect_engine", False))
    # no state leak: within each arm the cold prompt rerun last gives the cold tokens and ids
    isolation = {arm: [(r["events"], r.get("ids", [])) for r in (res[arm]["cold"][0], res[arm]["isolation"][0])]
                 for arm in ARMS if "isolation" in res[arm]}
    isolation = {arm: a == b for arm, (a, b) in isolation.items()}
    g_perf = perf["median"]["g_perf"] if perf else None
    ok = (all(s["status"] != DIFF for s in scenarios.values()) and sampled["ok"] and g_spec is not False
          and loop_ok and all(isolation.values()) and g_perf is not False
          and res["legacy"]["daemon_rc"] == res["engine"]["daemon_rc"] == 0)
    return {"ok": ok, "scenarios": scenarios, "sampled": sampled, "tau": taus, "g_spec": g_spec,
            "loop_ok": loop_ok, "loop": loops, "isolation": isolation, "g_perf": g_perf}


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
    assert st("abort_prefill", [r("", finish_reason="aborted"), r("x")],
              [r("", finish_reason="aborted"), r("x")]) == "equal"
    assert st("abort_prefill", [r("ab", finish_reason="aborted"), r("x")],
              [r("ab", finish_reason="aborted"), r("x")]) == "unexercised"
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
    assert st("abort_decode", [ids(ab[0], [[1, 0], [2, 1]]), r("x", cached_tokens=3)],
              [ids(ab[0], [[1, 0]]), r("x", cached_tokens=3)]) == "equal"
    assert st("abort_decode", [ids(ab[0], [[1, 0], [2, 1]]), r("x", cached_tokens=3)],
              [ids(ab[0], [[1, 0], [3, 1]]), r("x", cached_tokens=3)]) == "diff"
    il = [r("a", cached_tokens=0), r("b", cached_tokens=0), r("c", cached_tokens=5), r("d", cached_tokens=0)]
    assert st("interleaved", il, il) == "unexercised"
    assert st("interleaved", il[:3], il[:3]) == "equal"
    assert st("interleaved", il[:3], il[:2] + [r("c", cached_tokens=4)]) == "diff"

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
    assert not ok(arm(tau=0.5), route={"spec_noise": True})
    assert not ok(arm(tau=0.985), route={"spec_noise": True})  # G-spec bound is 1 %
    assert ok(arm(tau=0.995), route={"spec_noise": True})
    assert not ok(arm(tau=None), route={"spec_noise": True})
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
    assert not ok(arm(iso="ax"), leg=arm(iso="ax"))  # state leaked into a later request
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

    def conc_arm(loop, backend, tok_s=30.0, ttft=1000.0, k4=None, cached=5, tau=None, rc=0, stamp=None):
        reqs, stamp = {}, stamp or loop
        for wl in ("stateless", "multiturn"):
            for k in (1, 4):
                reqs[f"{wl}/{k}/1/0/0"] = req(tau=tau, loop=stamp)
                if wl == "multiturn":
                    reqs[f"{wl}/{k}/1/0/1"] = req("cd", cached=cached, tau=tau, loop=stamp)
        if k4:
            reqs["stateless/4/1/0/0"] = k4
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
    assert not cj(eng=conc_arm("engine", "noslots", tau=None), leg=conc_arm("legacy", "noslots", tau=3.0), c={})["ok"]
    assert not cj(eng=conc_arm("engine", "noslots", tok_s=58.0, ttft=900.0))["ok"]  # throughput below target
    assert not cj(eng=conc_arm("engine", "noslots", tok_s=61.0, ttft=950.0))["ok"]  # TTFT above target
    assert not cj(eng=good, batch=conc_arm("legacy", "batch", rc=1))["ok"]  # an arm that did not run
    assert cj(eng=good)["engine"]["engine-noslots"]["target"] == {"stateless/4": True, "multiturn/4": True}


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


def perf_ok(m):
    """G-perf: both median tok/s deltas within +-2 %."""
    return all(m.get(k) is not None and abs(m[k]) <= 2.0
               for k in ("prefill_tok_s_delta_pct", "decode_tok_s_delta_pct"))


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
    summary["g_perf"] = perf_ok(summary)
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
        host = host_state()
        with open(f"{base}.json", "w") as fo, open(f"{base}.log", "w") as fe:
            rc = subprocess.run(cmd, env=env, stdout=fo, stderr=fe).returncode
        try:
            cells = json.loads(Path(f"{base}.json").read_text())["cells"]
        except (json.JSONDecodeError, KeyError):
            cells = []
        with wire.open() as f:
            reqs = wire_requests(f, bench_order(workloads, points, runs))
        arms[f"{loop}-{backend}"] = {
            "loop": loop, "backend": backend, "rc": rc, "cmd": " ".join(cmd), "host": host,
            "cells": {f"{x['workload']}/{x['k']}": cell_metrics(x) for x in cells}, "reqs": reqs}
    return arms


def judge_conc(arms, c, expect_engine):
    """Concurrency gates on every engine arm; legacy arms are reference rows
    (legacy fn-lanes are not row-exact by default, HIPF-469). Pure: --self-test
    drives it with fakes. Per engine arm:
    - every request ends in `done`; `loop` stamps as in loop_checks
    - C1 == CN: each k>1 request's tokens and ids equal the same request at k=1
    - G-tok: tokens and ids equal the `ref` arm's (default legacy-noslots)
    - cache: each multi-turn follow-up reuses at least the ref arm's `cached_tokens`
    - G-spec: mean tau per cell >= ref - 1 %
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
    for tag, a in arms.items():
        if a["loop"] != "engine":
            continue
        reqs, rr = a["reqs"], (ref or {}).get("reqs", {})
        r = report[tag] = {}
        r["errors"] = sorted(k for k, x in reqs.items() if x["type"] != "done")
        loops = [x["done"].get("loop") for x in reqs.values() if x["type"] == "done"]
        r["loop"] = {str(v): loops.count(v) for v in set(loops)}
        loop_bad = (any(v not in (None, "legacy", "engine") for v in loops)
                    or (None in loops and any(v is not None for v in loops))
                    or (expect_engine and any(v != "engine" for v in loops)))
        r["c1_ne_cn"] = sorted(k for k, x in reqs.items() if k.split("/")[1] != "1" and not same(
            x, reqs.get("/".join([k.split("/")[0], "1", *k.split("/")[2:]]))))
        r["ne_ref"] = sorted(k for k, x in reqs.items() if not same(x, rr.get(k)))
        follow = [k for k in reqs if k.startswith("multiturn/") and not k.endswith("/0")]
        cached = lambda x: (x or {"done": {}})["done"].get("cached_tokens") or 0
        r["cache_below_ref"] = sorted(k for k in follow if cached(reqs[k]) < cached(rr.get(k)))
        r["cache_hits"] = [sum(cached(reqs[k]) > 0 for k in follow), sum(cached(rr.get(k)) > 0 for k in follow),
                           len(follow)]
        if follow and not r["cache_hits"][0]:
            unexercised.append(f"{tag}: no multi-turn follow-up reused a prefix")
        r["g_spec"], r["target"] = {}, {}
        for cell, m in a["cells"].items():
            taus = [[x["done"]["tau"] for k, x in rs.items() if k.startswith(cell + "/") and "tau" in x["done"]]
                    for rs in (reqs, rr)]
            if taus[1]:
                r["g_spec"][cell] = (len(taus[0]) == len(taus[1])
                                     and statistics.mean(taus[0]) >= 0.99 * statistics.mean(taus[1]))
            t = (target or {}).get("cells", {}).get(cell)
            if target and int(cell.split("/")[1]) > 1:
                r["target"][cell] = bool(t) and all(m[x] is not None and t[x] is not None for x in (
                    "out_tok_s", "ttft_p50_ms", "ttft_p99_ms")) and (
                    m["out_tok_s"] >= t["out_tok_s"] * (1 - tol)
                    and m["ttft_p50_ms"] <= t["ttft_p50_ms"] * (1 + tol)
                    and m["ttft_p99_ms"] <= t["ttft_p99_ms"] * (1 + tol))
        for name, bad in (("errors", r["errors"]), ("loop", loop_bad), ("C1 != CN", r["c1_ne_cn"]),
                          ("tokens != ref", r["ne_ref"]), ("cached_tokens < ref", r["cache_below_ref"]),
                          ("G-spec", not all(r["g_spec"].values())),
                          ("target", [k for k, v in r["target"].items() if not v])):
            if bad:
                fails.append(f"{tag}: {name}" + (f" {len(bad)}" if isinstance(bad, list) else ""))
    return {"ok": not fails, "fails": fails, "unexercised": unexercised, "engine": report,
            "cells": {tag: a["cells"] for tag, a in arms.items()}}


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
    ap.add_argument("--conc-only", action="store_true", help="run only the concurrency matrix")
    ap.add_argument("--hipfire", help="hipfire CLI for --conc (default: next to --daemon)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.daemon and args.route and args.out):
        ap.error("--daemon, --route and --out are required")
    route = json.loads(Path(args.route).read_text())
    route.setdefault("name", Path(args.route).stem)
    conc_on = args.conc or args.conc_only
    if conc_on and "conc" not in route:
        ap.error("--conc needs a `conc` block in the route file")
    args.hipfire = args.hipfire or str(Path(args.daemon).with_name("hipfire"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = os.path.expanduser(route["model"])
    route["model"] = model

    res, t0, j, perf, conc, unex, summary = {}, time.time(), {"ok": True}, None, None, [], []
    if not args.conc_only:
        for arm in ARMS:
            res[arm] = run_arm(args, route, arm, out,
                               None if arm == "legacy" else res["legacy"]["eos_budget"][0]["done"].get("tokens", 0))
        perf = run_perf(args, route, out) if args.perf else None
        j = judge(res, route, args.tvd_max, perf)
        # a pass with unexercised scenarios must not be citable as covering them
        unex = [k for k, v in j["scenarios"].items() if v["status"] == UNEXERCISED]
        sampled = j["sampled"]
        summary.append(", ".join(f"{k}={v['status']}" for k, v in j["scenarios"].items()))
        summary.append(f"sampled tvd={sampled['tvd_first8']} repro={sampled['legacy_reproducible']}/"
                       f"{sampled['engine_reproducible']} distinct={sampled['legacy_distinct']}/{sampled['engine_distinct']}")
        summary.append(f"loop={j['loop']['engine']}" + ("" if j["loop_ok"] else " (FAIL)"))
        summary.append(f"isolation={j['isolation']}")
        if j["g_spec"] is not None:
            summary.append(f"G-spec={j['g_spec']}")
        if perf:
            m = perf["median"]
            summary.append(f"G-perf={m['g_perf']} prefill {m['prefill_tok_s_delta_pct']}% "
                           f"decode {m['decode_tok_s_delta_pct']}%")
    if conc_on:
        arms = run_conc(args, route, out)
        conc = judge_conc(arms, route["conc"], route.get("expect_engine", False))
        unex += conc["unexercised"]
        summary.append("conc=" + ("pass" if conc["ok"] else "FAIL: " + "; ".join(conc["fails"])))
    ok = j["ok"] and (conc is None or conc["ok"])
    verdict = ("pass-partial" if unex else "pass") if ok else "fail"
    summary = ", ".join(summary)

    me = Path(__file__).resolve()
    git = lambda *a: subprocess.run(["git", "-C", str(me.parent), *a], capture_output=True, text=True).stdout.strip()
    receipt = {
        "route": route, "verdict": verdict, "unexercised": unex, **{k: v for k, v in j.items() if k != "ok"},
        "daemon": {"path": os.path.realpath(args.daemon),
                   "md5": hashlib.md5(Path(args.daemon).read_bytes()).hexdigest()},
        "fixture_sha256": sha256_cached(model, out),
        # identifies the harness bytes even when run from a copy outside the repo
        "harness": {"sha256": hashlib.sha256(me.read_bytes()).hexdigest(), "git_head": git("rev-parse", "HEAD"),
                    "dirty": bool(git("status", "--porcelain", "--", str(me)))},
        "host": host_state(), "wall_s": round(time.time() - t0, 1), "arms": res,
    }
    if perf:
        receipt["perf"] = perf
    if conc:
        receipt["hipfire"] = {"path": os.path.realpath(args.hipfire),
                              "md5": hashlib.md5(Path(args.hipfire).read_bytes()).hexdigest()}
        receipt["conc"] = {**conc, "arms": arms}
    rpath = out / f"{route['name']}{'.conc' if args.conc_only else ''}.json"
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
