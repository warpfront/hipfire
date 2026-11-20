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
# Done-envelope keys that are wall-clock measurements, never compared.
TIMING = re.compile(r"(_ms|_s|tok_s|_us|_ns|wall|time|elapsed)$")
# Speculation bookkeeping: judged by G-spec (mean tau), never exactly. Qwen4 MTP
# acceptance differs between two legacy processes with identical tokens (HIPF-468).
SPEC = re.compile(r"^(tau|cycles|accepted|mtp_windows|floor_ar_windows|ar_windows|mtp_retired|ngram_mod_.*)$")
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
                return {"type": t, "events": events, "text": text,
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
        # abort mid-prefill (long prompt, abort queued behind the request), then a
        # follow-up that must not see corrupt state
        res["abort_prefill"] = [d.generate("ap", greedy(LONG, mt), abort_now=True),
                                d.generate("ap-next", greedy("Write one sentence about the sea.", mt))]
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


def done_view(done):
    return {k: v for k, v in sorted(done.items()) if not TIMING.search(k) and k != "loop"}


def compare(sc, a, b, a_name="legacy", b_name="engine"):
    """Diff two result lists. Returns (status, notes); status equal|intended|diff."""
    notes, intended = [], False
    if len(a) != len(b):
        return "diff", [f"{len(a)} vs {len(b)} requests"]
    for i, (x, y) in enumerate(zip(a, b)):
        # an error outside an abort is a broken scenario, even when both arms agree
        for name, r in ((a_name, x), (b_name, y)):
            if sc != "load" and r["type"] == "error" and not (sc.startswith("abort") and i == 0):
                notes.append(f"[{i}] {name} error: {r['done'].get('message')}")
        if sc.startswith("seeded"):
            continue
        if sc == "load":
            if done_view(x["done"]) != done_view(y["done"]):
                notes.append("loaded envelope differs")
            continue
        dx, dy = done_view(x["done"]), done_view(y["done"])
        if sc == "eos_budget" and i == 0 and dx.get("finish_reason") != "stop":
            notes.append("eos_budget: not exercised (the probe hit its budget before EOS)")
        if sc == "abort_decode" and i == 0:
            if dx.get("finish_reason") != "aborted":
                notes.append(f"abort_decode: abort did not land (finish_reason {dx.get('finish_reason')})")
            # abort lands asynchronously: the shorter stream must be a prefix
            # of the longer one, and both must end aborted.
            k = min(len(x["events"]), len(y["events"]))
            if x["events"][:k] != y["events"][:k]:
                notes.append("abort_decode: streams diverge before the abort")
            if dx.get("finish_reason") != dy.get("finish_reason"):
                notes.append(f"abort_decode finish_reason {dx.get('finish_reason')} vs {dy.get('finish_reason')}")
            continue
        if x["events"] != y["events"]:
            notes.append(f"[{i}] tokens differ ({x['sha']} vs {y['sha']})")
        for key in sorted(set(dx) | set(dy)):
            vx, vy = dx.get(key), dy.get(key)
            if vx == vy:
                continue
            # Intended diff 2: EOS is not counted in `tokens` on the engine.
            if key == "tokens" and isinstance(vx, int) and vy == vx - 1 and dx.get("finish_reason") == "stop":
                intended = True
                notes.append(f"[{i}] intended: tokens {vx}->{vy} (EOS not counted)")
            # Intended diff 3: a stop on the last budget token is `stop`.
            elif key == "finish_reason" and sc == "eos_budget" and (vx, vy) == ("length", "stop"):
                intended = True
                notes.append(f"[{i}] intended: finish_reason length->stop on last budget token")
            elif SPEC.match(key):
                notes.append(f"[{i}] spec-stat: {key}: {vx!r} vs {vy!r}")
            else:
                notes.append(f"[{i}] {key}: {vx!r} vs {vy!r}")
    real = [n for n in notes if "intended:" not in n and "spec-stat:" not in n]
    return ("diff" if real else "intended" if intended else "equal"), notes


def self_test():
    """GPU-free check of the diff rules (`--self-test`)."""
    def r(text, typ="done", **done):
        return {"type": typ, "events": [["t", c] for c in text], "sha": text, "done": done}
    same = [r("ab", tokens=3, finish_reason="stop", prefill_ms=1.0)]
    assert compare("cold", same, [r("ab", tokens=3, finish_reason="stop", prefill_ms=9.0, loop="engine")])[0] == "equal"
    assert compare("cold", same, [r("ac", tokens=3, finish_reason="stop")])[0] == "diff"
    assert compare("cold", same, [r("ab", tokens=2, finish_reason="stop")])[0] == "intended"
    assert compare("cold", same, [r("ab", tokens=3, finish_reason="stop", cached_tokens=4)])[0] == "diff"
    probe = r("ab", tokens=2, finish_reason="stop")
    budget = [probe, r("ab", tokens=2, finish_reason="length")]
    assert compare("eos_budget", budget, [probe, r("ab", tokens=2, finish_reason="stop")])[0] == "intended"
    assert compare("cold", budget, [probe, r("ab", tokens=2, finish_reason="stop")])[0] == "diff"
    assert compare("eos_budget", budget[1:], budget[1:])[0] == "diff"  # probe never reached EOS
    err = [r("", "error", message="ctx")]
    assert compare("cold", err, err)[0] == "diff"
    assert compare("cold", [r("ab", tau=1.2, cycles=4)], [r("ab", tau=0.9, cycles=5)])[0] == "equal"
    ab = [r("abcd", finish_reason="aborted"), r("x")]
    assert compare("abort_decode", ab, [r("abc", finish_reason="aborted"), r("x")])[0] == "equal"
    assert compare("abort_decode", ab, [r("abX", finish_reason="aborted"), r("x")])[0] == "diff"
    assert compare("abort_decode", [r("ab", finish_reason="stop"), r("x")],
                   [r("ab", finish_reason="stop"), r("x")])[0] == "diff"
    print("self-test ok")


def first_tokens(rows, k=8):
    hist = {}
    for r in rows:
        for e in r["events"][:k]:
            hist[e[1]] = hist.get(e[1], 0) + 1
    tot = sum(hist.values()) or 1
    return {t: c / tot for t, c in hist.items()}


def sampled_checks(res):
    """Intended diff 1 (per-request RNG): sampled output is not compared bitwise
    across arms. Each arm must reproduce itself; the arms' early-token
    distributions must stay close (total variation distance)."""
    out = {}
    for arm in ARMS:
        a, b = res[arm]["seeded_repro"]
        out[f"{arm}_reproducible"] = a["events"] == b["events"]
    p, q = first_tokens(res["legacy"]["seeded_sweep"]), first_tokens(res["engine"]["seeded_sweep"])
    out["tvd_first8"] = round(0.5 * sum(abs(p.get(t, 0) - q.get(t, 0)) for t in set(p) | set(q)), 4)
    out["bitwise_equal_across_arms"] = all(
        x["events"] == y["events"] for x, y in zip(res["legacy"]["seeded_sweep"], res["engine"]["seeded_sweep"]))
    return out


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
    summary["g_perf"] = all(v is not None and abs(v) <= 2.0 for k, v in summary.items() if k.endswith("_pct"))
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
    scenarios = {}
    for sc in res["legacy"]:
        if sc == "daemon_rc":
            continue
        status, notes = compare(sc, res["legacy"][sc], res["engine"][sc])
        scenarios[sc] = {"status": status, "notes": notes,
                         "loop": {arm: [r["done"].get("loop") for r in res[arm][sc]] for arm in ARMS}}
    sampled = sampled_checks(res)
    sampled_ok = (sampled["legacy_reproducible"] and sampled["engine_reproducible"]
                  and sampled["tvd_first8"] <= args.tvd_max)
    # A spec route reports tau; G-spec: engine tau >= legacy tau - 1 %.
    taus = {arm: [r["done"]["tau"] for sc in ("cold", "multi_turn", "abort_prefill", "abort_decode", "eos_budget")
                  for r in res[arm][sc][int(sc.startswith("abort")):] if "tau" in r["done"]]
            for arm in ARMS}
    g_spec = None
    if taus["legacy"] and taus["engine"]:
        g_spec = statistics.mean(taus["engine"]) >= 0.99 * statistics.mean(taus["legacy"])
    # Loop marker: absent (flag unknown, pre-engine binary) or the value the arm ran.
    loop_ok = all(v in (None, "legacy", arm) for sc in scenarios.values()
                  for arm, vs in sc["loop"].items() for v in vs)
    ok = (all(s["status"] != "diff" for s in scenarios.values()) and sampled_ok
          and g_spec is not False and loop_ok and res["legacy"]["daemon_rc"] == res["engine"]["daemon_rc"] == 0)

    receipt = {
        "route": route, "verdict": "pass" if ok else "fail", "scenarios": scenarios,
        "sampled": dict(sampled, ok=sampled_ok), "tau": taus, "g_spec": g_spec, "loop_ok": loop_ok,
        "daemon": {"path": os.path.realpath(args.daemon),
                   "md5": hashlib.md5(Path(args.daemon).read_bytes()).hexdigest()},
        "fixture_sha256": sha256_cached(model, out),
        "head": subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
        "host": host_state(), "wall_s": None, "arms": res,
    }
    if args.perf:
        receipt["perf"] = run_perf(args, route, out)
    receipt["wall_s"] = round(time.time() - t0, 1)
    rpath = out / f"{route['name']}.json"
    rpath.write_text(json.dumps(receipt, indent=1))

    ev = out / "EVIDENCE.md"
    if not ev.exists():
        ev.write_text("| route | head | daemon md5 | fixture sha256 | command | receipt | result | disposition |\n"
                      "|---|---|---|---|---|---|---|---|\n")
    summary = ", ".join(f"{k}={v['status']}" for k, v in scenarios.items())
    summary += f", sampled tvd={sampled['tvd_first8']} repro={sampled['legacy_reproducible']}/{sampled['engine_reproducible']}"
    if g_spec is not None:
        summary += f", G-spec={g_spec}"
    if args.perf:
        m = receipt["perf"]["median"]
        summary += f", perf prefill {m['prefill_tok_s_delta_pct']}% decode {m['decode_tok_s_delta_pct']}%"
    with ev.open("a") as f:
        f.write(f"| {route['name']} | {receipt['head'][:9]} | {receipt['daemon']['md5'][:8]} | "
                f"{receipt['fixture_sha256'][:16]} | `{' '.join(sys.argv)}` | {rpath} | {summary} | "
                f"{'current' if ok else 'failed'} |\n")
    print(json.dumps({"route": route["name"], "verdict": receipt["verdict"], "summary": summary}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, queue.Empty) as e:
        print(f"engine_loop_ab: {e}", file=sys.stderr)
        sys.exit(2)
