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
GREEDY = ("cold", "multi_turn", "abort_prefill", "abort_decode", "eos_budget")
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
        events, n = [], 0
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
            elif t == "commit_ready":
                self.send({"type": "commit", "id": rid, "attempt_id": 1})
            elif t in ("done", "error"):
                done = {k: v for k, v in e.items() if k not in ("type", "id", "attempt_id")}
                text = "".join(x[1] for x in events)
                return {"type": t, "events": events, "text": text, "max_tokens": req.get("max_tokens"),
                        "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "done": done}

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


def greedy(prompt, max_tokens, **kw):
    return {"prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens, **kw}


def run_arm(args, route, arm, out, eos_n=None):
    """All scenarios on one fresh daemon. Returns {scenario: [result, ...]}.
    eos_n: the eos_budget length to rerun with (default: this arm's own probe)."""
    env = dict(os.environ, **route.get("env", {}), HIPFIRE_ENGINE_LOOP=arm)
    d = Daemon(args.daemon, env, f"{out}/{route['name']}.{arm}", args.timeout)
    res = {}
    mt = route.get("max_tokens", 64)
    try:
        res["load"] = [{"done": d.load(route["model"], route.get("load", {}))}]
        # cold start: first request on a fresh load
        res["cold"] = [d.generate("cold", greedy("Write one sentence about the sea.", mt))]
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
        if sc == "multi_turn" and i > 0 and not dx.get("cached_tokens"):
            note(UNEXERCISED, f"[{i}] turn reused no prefix (cached_tokens {dx.get('cached_tokens')!r})")
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
            k = min(len(x["events"]), len(y["events"]))
            if x["events"][:k] != y["events"][:k]:
                note(DIFF, "streams diverge before the abort")
            continue
        if x["events"] != y["events"]:
            note(DIFF, f"[{i}] tokens differ ({x['sha']} vs {y['sha']})")
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
    g_perf = perf["median"]["g_perf"] if perf else None
    ok = (all(s["status"] != DIFF for s in scenarios.values()) and sampled["ok"] and g_spec is not False
          and loop_ok and g_perf is not False and res["legacy"]["daemon_rc"] == res["engine"]["daemon_rc"] == 0)
    return {"ok": ok, "scenarios": scenarios, "sampled": sampled, "tau": taus, "g_spec": g_spec,
            "loop_ok": loop_ok, "loop": loops, "g_perf": g_perf}


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
    ab = [r("abcd", finish_reason="aborted"), r("x")]
    assert st("abort_decode", ab, [r("abc", finish_reason="aborted"), r("x")]) == "equal"
    assert st("abort_decode", ab, [r("abX", finish_reason="aborted"), r("x")]) == "diff"
    assert st("abort_decode", ab, [r("abcd", finish_reason="stop"), r("x")]) == "diff"
    assert st("abort_decode", ab, [r("abcd", finish_reason="aborted"), r("y")]) == "diff"
    assert st("abort_decode", [r("ab", finish_reason="stop"), r("x")],
              [r("ab", finish_reason="stop"), r("x")]) == "diff"
    assert st("abort_decode", [r("a" * 40, finish_reason="aborted"), r("x")],
              [r("a" * 40, finish_reason="aborted"), r("x")]) == "unexercised"
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

    # gates
    def arm(loop=None, tau=1.0, sweep=("abc", "bcd", "cde", "def"), repro=("p", "p"), rc=0):
        def d(text):
            g = r(text, finish_reason="stop", tokens=len(text), cached_tokens=0, tau=tau, loop=loop)
            g["done"] = {k: v for k, v in g["done"].items() if v is not None}
            return g
        return {"load": [{"done": {"arch": "x"}}], "cold": [d("ab")],
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
    print("self-test ok")


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
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.daemon and args.route and args.out):
        ap.error("--daemon, --route and --out are required")
    route = json.loads(Path(args.route).read_text())
    route.setdefault("name", Path(args.route).stem)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = os.path.expanduser(route["model"])
    route["model"] = model

    res, t0 = {}, time.time()
    for arm in ARMS:
        res[arm] = run_arm(args, route, arm, out,
                           None if arm == "legacy" else res["legacy"]["eos_budget"][0]["done"].get("tokens", 0))
    perf = run_perf(args, route, out) if args.perf else None
    j = judge(res, route, args.tvd_max, perf)
    ok = j["ok"]
    # a pass with unexercised scenarios must not be citable as covering them
    unex = [k for k, v in j["scenarios"].items() if v["status"] == UNEXERCISED]
    verdict = ("pass-partial" if unex else "pass") if ok else "fail"

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
    rpath = out / f"{route['name']}.json"
    rpath.write_text(json.dumps(receipt, indent=1))

    ev = out / "EVIDENCE.md"
    if not ev.exists():
        ev.write_text("| route | harness sha256 | daemon md5 | fixture sha256 | command | receipt | result | disposition |\n"
                      "|---|---|---|---|---|---|---|---|\n")
    sampled = j["sampled"]
    summary = ", ".join(f"{k}={v['status']}" for k, v in j["scenarios"].items())
    summary += (f", sampled tvd={sampled['tvd_first8']} repro={sampled['legacy_reproducible']}/"
                f"{sampled['engine_reproducible']} distinct={sampled['legacy_distinct']}/{sampled['engine_distinct']}")
    summary += f", loop={j['loop']['engine']}" + ("" if j["loop_ok"] else " (FAIL)")
    if j["g_spec"] is not None:
        summary += f", G-spec={j['g_spec']}"
    if perf:
        m = perf["median"]
        summary += f", G-perf={m['g_perf']} prefill {m['prefill_tok_s_delta_pct']}% decode {m['decode_tok_s_delta_pct']}%"
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
