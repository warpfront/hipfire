"""Decide GPU correctness gates (spec §10). Exit 0 iff no gate FAILs.

Each gate prints PASS, FAIL or INCONCLUSIVE; INCONCLUSIVE does not fail the
run but is printed prominently.

Gates:
  1a  exact mode: a second daemon with the batch-composition-dependent kernel
      routes pinned to their row-invariant alternatives (EXACT_ENV). There,
      snapshot decide == `_debug_no_snapshot` decide bit for bit (Δ == 0.0),
      and each question of the 3-question request == a single-question
      decide of it alone (Δ == 0.0: restore leaves no residue). A
      single-question decide is one plain full prefill (prefix_tokens == 0,
      no snapshot), which is checked too. Also checks the template closed
      thinking (no 400).
  1b  default mode, per question: |Δlogp| snapshot vs full prefill
      (drift_q) <= 2 x floor_q, where floor_q is that question's path-noise
      floor = one-call prefill vs token-by-token prefill (A vs C) of the same
      prompt, measured per run by the `split_prefill_probe` example in
      PROBE_DECIDE_REQUEST mode. The probe's A must reproduce the daemon's
      full-prefill answer (Δ <= 1e-6), else the floor is not comparable: FAIL.
      1a proves exactness; 1b only guards against gross regressions of the
      default kernels, and the floor is a different noise source (per-token
      path) than the drift (split point), hence the factor 2.
  1c  restore order-invariance (default mode): each question's answer is
      bit-identical (Δ == 0) whether it runs first (no restore) or after
      restores, at a fixed split point.
  2   question isolation, baseline-relative: PASS iff
      |P(code | code in sibling) - P(code | no code)| <= 0.05 and
      P(code | code in state) - P(code | no code) >= 0.3. If only the
      sensitivity half fails, the model cannot do the probe: INCONCLUSIVE.
  3   model left clean: greedy generate identical before/after a decide.
  4   no device-memory leak over N decides: growth of the daemon's own
      amdgpu GTT + VRAM (/proc/<pid>/fdinfo) < 64 MiB; the device-wide
      sysfs gtt + vram sum over all cards is printed alongside and is the
      fallback when fdinfo is unreadable (FAIL if neither is readable).
  5   error replies end to end: 422 + required_max_seq for an over-long
      state, 422 for a malformed request, 422 naming Jev's per-question
      (32000) and request (64000) token limits, without required_max_seq; a
      normal decide succeeds after each. With --cask the over-long reply is
      the eviction-limit 422 instead (no required_max_seq; the message names
      the limit).
  0   (--cask only, per daemon) CASK is really configured: an over-limit
      decide gets the eviction-limit 422 (only reachable with eviction on),
      then a normal decide succeeds.

--cask loads every daemon with the load params `hipfire serve` sends when
config has cask on (cask, cask_sidecar, cask_budget, cask_beta, cask_core_frac,
cask_fold_m), defaults read from ~/.hipfire/config.json, so every gate
(including 1a exact-mode equality) runs with eviction configured. The
floor probe (1b) loads without CASK.

Why two modes for gate 1: one-call vs split prefill differs by an inherent,
deterministic precision property of kernels whose route or rounding depends
on the batch composition, not by a decide defect
(.superpowers/sdd/split-prefill-investigation.md):
  - HFQ4G128 GEMM switches to int8-activation MMQ on 16-aligned batches
    (gfx1151), off with HIPFIRE_HFQ4G128_MMQ=0;
  - the generic MQ4 MMQ route (int8 activations) engages at batch >= 128,
    off with HIPFIRE_MMQ=0 (found by this gate: the 129-token queue prompt
    of qwen3.5-4b hit it; the investigation's N<=120 probe did not);
  - the Q8 GatedDeltaNet state is requantised once per launch, per token
    with HIPFIRE_DN_REQUANT_PER_TOKEN=1;
  - the f16 WMMA flash-prefill attention is not bit-row-invariant at hd=128,
    legacy f32 kernel with HIPFIRE_FLASH_PREFILL=0.
Suffixes of 1-3 tokens take the per-token path (different kernels); a
single-question decide never splits (it is one full prefill), so it is the
exact-mode single-question reference directly.

EXACT_ENV pins today's batch-composition-dependent routes. When PR #768's
default-on gfx1151 routes land, EXACT_ENV must be extended with their
disable switches or gate 1a's Δ == 0 checks will fail on those routes.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daemon_client import REPO, Daemon  # noqa: E402

EXACT_ENV = {"HIPFIRE_HFQ4G128_MMQ": "0", "HIPFIRE_MMQ": "0",
             "HIPFIRE_DN_REQUANT_PER_TOKEN": "1", "HIPFIRE_FLASH_PREFILL": "0"}
PROBE = REPO / "target/release/examples/split_prefill_probe"

STATE = {"channel": "email", "subject": "Where is my order?",
         "body": "Order #3527 was placed 21 days ago and tracking has not updated."}
QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which support queue should handle this ticket?",
              "criteria": {"billing": "Payments, refunds", "shipping": "Delivery status, lost parcels",
                           "technical": "App bugs, login", "general": "Anything else"}},
    "priority": {"type": "score", "instructions": "How urgent is this ticket?",
                 "criteria": ["Low", "Normal", "High", "Critical"]},
    "angry": {"type": "noul", "instructions": "The customer sounds angry."},
}


def dist(a):
    """Answer -> {key: probability}, keyed so comparisons never depend on
    dict/value order."""
    if a["type"] == "noul":
        return {"true": a["noul"], "false": 1 - a["noul"]}
    return dict(a["probabilities"])


def dlogp(a, b):
    """max |Δ log p| between two answers, aligned by key."""
    da, db = dist(a), dist(b)
    if set(da) != set(db):
        raise AssertionError(f"answer keys differ: {sorted(da)} vs {sorted(db)}")
    worst = 0.0
    for k in sorted(da):
        worst = max(worst, abs(math.log(max(da[k], 1e-12)) - math.log(max(db[k], 1e-12))))
    return worst


def need_answers(r, what):
    if "answers" not in r:
        raise AssertionError(f"{what}: no answers in reply: {r}")
    return r["answers"]


def fmt_answer(a):
    if a["type"] == "noul":
        return f"noul={a['noul']:.4f}"
    top = max(a["probabilities"], key=a["probabilities"].get)
    return f"{a['type']} top={top} p={a['probabilities'][top]:.4f}"


def gate_exact_mode(d):
    snap = d.decide(STATE, QUESTIONS)
    err = snap.get("error") or {}
    if err.get("status") == 400 and "thinking" in err.get("message", ""):
        return False, f"template cannot close thinking: {err}"
    sa = need_answers(snap, "snapshot")
    fa = need_answers(d.decide(STATE, QUESTIONS, _debug_no_snapshot=True), "full-prefill")
    snap_d = {n: dlogp(sa[n], fa[n]) for n in QUESTIONS}
    single_d, single_prefix = {}, set()
    for name, q in QUESTIONS.items():
        # A single-question decide is one plain full prefill (no snapshot).
        r1 = d.decide(STATE, {name: q})
        s1 = need_answers(r1, f"single {name}")
        single_prefix.add(r1["timing"]["prefix_tokens"])
        single_d[name] = dlogp(sa[name], s1[name])
    ok = (max(snap_d.values()) == 0.0 and max(single_d.values()) == 0.0
          and snap["timing"]["prefix_tokens"] > 0 and single_prefix == {0})
    f = lambda m: ", ".join(f"{k}={v:.3g}" for k, v in m.items())  # noqa: E731
    return ok, (f"exact env {EXACT_ENV}: snapshot vs full-prefill Δ [{f(snap_d)}] (==0), "
                f"multi-question vs single-question (plain full prefill) Δ [{f(single_d)}] (==0), "
                f"prefix_tokens multi={snap['timing']['prefix_tokens']} (>0) "
                f"single={sorted(single_prefix)} (==[0]), closed-think template OK (no 400)")


def measure_floor(model, log_path):
    """A (one call) vs C (token by token) on the decide prompts, default env."""
    if not PROBE.exists():
        raise RuntimeError(f"{PROBE} missing: cargo build --release -p hipfire-generate "
                           "--features lab --example split_prefill_probe")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"state": STATE, "questions": QUESTIONS}, fh)
        req = fh.name
    try:
        with open(log_path or os.devnull, "w") as log:
            out = subprocess.run([str(PROBE), str(model)], env={**os.environ, "PROBE_DECIDE_REQUEST": req},
                                 stdout=subprocess.PIPE, stderr=log, text=True, check=True).stdout
    finally:
        os.unlink(req)
    for line in out.splitlines():
        if line.startswith("DECIDE_FLOOR "):
            return json.loads(line[len("DECIDE_FLOOR "):])
    raise RuntimeError(f"probe printed no DECIDE_FLOOR line:\n{out}")


FLOOR_FACTOR = 2.0   # human ruling: per question, drift_q <= 2 x floor_q
FLOOR_CONSISTENCY = 1e-6


def gate_noise_floor(d, floor):
    snap = need_answers(d.decide(STATE, QUESTIONS), "snapshot")
    full = need_answers(d.decide(STATE, QUESTIONS, _debug_no_snapshot=True), "full-prefill")
    pq = floor["per_question"]
    # The probe's one-call A must reproduce the daemon's full-prefill answer:
    # otherwise the floor was measured on a different prompt or kernel route.
    same = max(dlogp(pq[n]["a_answer"], full[n]) for n in QUESTIONS)
    rows, ok = [], same <= FLOOR_CONSISTENCY
    for n in QUESTIONS:
        drift = dlogp(snap[n], full[n])
        fl = dlogp(pq[n]["a_answer"], pq[n]["c_answer"])
        q_ok = drift <= FLOOR_FACTOR * fl
        ok &= q_ok
        rows.append(f"{n}: drift={drift:.4f} floor={fl:.4f} 2xfloor={FLOOR_FACTOR * fl:.4f} "
                    f"{'ok' if q_ok else 'OVER'}")
    cons = (f"probe A vs daemon full-prefill Δ = {same:.3g} (<= {FLOOR_CONSISTENCY:g})"
            + ("" if same <= FLOOR_CONSISTENCY else
               " — FLOOR MEASURED ON A DIFFERENT PROMPT/ROUTE, not comparable"))
    return ok, (f"per question, snapshot vs full-prefill drift <= {FLOOR_FACTOR:g} x floor "
                f"(one-call vs token-by-token, split_prefill_probe, this run): [{'; '.join(rows)}]; {cons}")


def gate_restore_order(d):
    # Clean restore test with the split point held fixed: every question
    # ordering shares the same prefix, so a question answered first (no
    # restore, straight from the snapshot point) must match the same question
    # answered after one or two restores. Any Δ here is restore residue, not
    # chunk-boundary numerics.
    import itertools
    names = list(QUESTIONS)
    ref, worst, prefixes = None, 0.0, set()
    for perm in itertools.permutations(names):
        r = d.decide(STATE, {n: QUESTIONS[n] for n in perm})
        a = need_answers(r, f"order {perm}")
        prefixes.add(r["timing"]["prefix_tokens"])
        if ref is None:
            ref = a
        for n in names:
            worst = max(worst, dlogp(ref[n], a[n]))
    ok = worst == 0.0 and len(prefixes) == 1
    return ok, (f"max |Δlogp| across all {math.factorial(len(names))} question orderings = {worst:.3g} (==0), "
                f"prefix_tokens={sorted(prefixes)}")


def gate_isolation(d):
    q = {"type": "noul", "instructions": "Does the text contain the secret code word PELICAN?"}
    none = need_answers(d.decide("A short note about the weather.", {"probe": q}),
                        "no-code")["probe"]["noul"]
    sib = need_answers(d.decide("A short note about the weather.", {
        "probe": q,
        "leak": {"type": "noul", "instructions": "The secret code word is PELICAN. Is it sunny?"},
    }), "sibling")["probe"]["noul"]
    in_state = need_answers(d.decide("A short note about the weather. The secret code word is PELICAN.",
                                     {"probe": q}), "in-state")["probe"]["noul"]
    leak_ok = abs(sib - none) <= 0.05
    sens_ok = in_state - none >= 0.3
    detail = (f"P(code): no code anywhere = {none:.3f}, code in sibling question = {sib:.3f} "
              f"(|Δ|={abs(sib - none):.3f} <=0.05: {leak_ok}), code in state = {in_state:.3f} "
              f"(Δ={in_state - none:.3f} >=0.3: {sens_ok})")
    if not leak_ok:
        return False, detail
    if not sens_ok:
        return INCONCLUSIVE, detail + " — model cannot do the probe"
    return True, detail


def gate_clean(d):
    prompt = "Where is my order? Order #3527 was placed 21 days ago. Reply in one sentence."
    d.reset()
    after_reset = d.generate_greedy(prompt)
    d.reset()
    need_answers(d.decide(STATE, QUESTIONS), "decide")
    after_decide = d.generate_greedy(prompt)
    d.reset()
    unrelated = "Name three primary colours."
    base2 = d.generate_greedy(unrelated)
    d.reset()
    need_answers(d.decide(STATE, QUESTIONS), "decide")
    after2 = d.generate_greedy(unrelated)
    d.reset()
    ok = after_reset == after_decide and base2 == after2 and len(after_reset) > 0 and len(base2) > 0
    return ok, (f"shared-prefix prompt identical={after_reset == after_decide}, "
                f"unrelated identical={base2 == after2}; "
                f"texts: {after_reset[:60]!r} / {base2[:60]!r}")


def device_mem():
    """(gtt_used, vram_used) summed over every /sys/class/drm/card*/device
    node, each None if no node exposes it."""
    tot = {"gtt": None, "vram": None}
    for kind in tot:
        for p in sorted(Path("/sys/class/drm").glob(f"card*/device/mem_info_{kind}_used")):
            try:
                v = int(p.read_text())
            except (OSError, ValueError):
                continue
            tot[kind] = (tot[kind] or 0) + v
    return tot["gtt"], tot["vram"]


def process_mem(pid):
    """(gtt, vram) bytes held by this process's amdgpu DRM clients, from
    /proc/<pid>/fdinfo `drm-memory-*`; None if no amdgpu fd is visible."""
    unit = {"": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
    tot, seen = {"gtt": 0, "vram": 0}, False
    for f in Path(f"/proc/{pid}/fdinfo").glob("*"):
        try:
            t = f.read_text()
        except OSError:
            continue
        if "drm-driver:\tamdgpu" not in t:
            continue
        seen = True
        for kind, val, u in re.findall(r"drm-memory-(gtt|vram):\s+(\d+)\s*([KMG]iB)?", t):
            tot[kind] += int(val) * unit[u or ""]
    return (tot["gtt"], tot["vram"]) if seen else None


def gate_leak(d, n):
    need_answers(d.decide(STATE, QUESTIONS), "warmup")
    g0, v0 = device_mem()
    p0 = process_mem(d.p.pid)
    for i in range(n):
        need_answers(d.decide(STATE, QUESTIONS), f"leak iter {i}")
    g1, v1 = device_mem()
    p1 = process_mem(d.p.pid)
    mib = lambda x: f"{x / 2**20:.1f}"  # noqa: E731
    if g0 is None and v0 is None:
        return False, "no mem_info_gtt_used / mem_info_vram_used sysfs node readable — cannot measure"
    dg = (g1 - g0) if g0 is not None and g1 is not None else 0
    dv = (v1 - v0) if v0 is not None and v1 is not None else 0
    sys_s = (f"system-wide (sysfs, all cards) {mib(dg + dv)} MiB "
             f"[gtt {mib(dg) if g0 is not None else 'n/a'} + vram {mib(dv) if v0 is not None else 'n/a'}]")
    # The sysfs counters are device-wide: on a shared workstation other
    # processes move them by tens of MiB over a 1000-decide run (observed
    # -101.6 and +64.3 MiB while the daemon's own fdinfo total moved 2 MiB).
    # When the daemon's own DRM fdinfo is readable it is the criterion; the
    # device-wide sum is printed alongside and is the fallback.
    if p0 is not None and p1 is not None:
        pg, pv = p1[0] - p0[0], p1[1] - p0[1]
        grew = pg + pv
        basis = f"daemon process (fdinfo) {mib(grew)} MiB [gtt {mib(pg)} + vram {mib(pv)}]"
    else:
        grew = dg + dv
        basis = "daemon fdinfo unreadable, using system-wide"
    ok = grew < 64 * 1024 * 1024
    return ok, f"device memory growth over {n} decides: {basis} (<64); {sys_s}"


def cask_limit(cask):
    """Decide's eviction-safe limit: min(cask_budget, physical_cap). The load
    requires max_seq >= budget + beta + 4 and physical_cap is clamped to at
    least that, so the budget always binds."""
    return cask["cask_budget"]


LOREM = "lorem ipsum dolor sit amet "   # 5-7 tokens per repetition
JEV_QUESTION_LIMIT = 32000
JEV_REQUEST_LIMIT = 64000


def check_cask_over_limit(d, cask):
    """Over the eviction limit: 422, no required_max_seq, message names the limit."""
    limit = cask_limit(cask)
    if limit >= JEV_QUESTION_LIMIT * 4 // 7:
        raise AssertionError(f"cask_budget {limit} too large: an over-limit state would hit "
                             f"Jev's {JEV_QUESTION_LIMIT}-token per-question limit first")
    # limit/4 reps x 5-7 tokens = 1.25-1.75 x limit: over the eviction limit,
    # under Jev's per-question limit (checked before it).
    r = d.decide(LOREM * (limit // 4), QUESTIONS)
    e = r.get("error") or {}
    ok = (e.get("status") == 422 and "required_max_seq" not in e
          and f"limited to {limit} tokens" in e.get("message", ""))
    return ok, (f"over eviction limit: status={e.get('status')} "
                f"required_max_seq={e.get('required_max_seq')} (absent) "
                f"message={e.get('message')!r} (names limit {limit}) -> {ok}")


def gate_cask_configured(d, cask):
    ok, note = check_cask_over_limit(d, cask)
    r = d.decide(STATE, QUESTIONS)
    after = "answers" in r
    return ok and after, (f"{note}; normal decide succeeds={after}"
                          + ("" if after else f" reply={r}"))


def gate_errors(d, max_seq, cask=None):
    notes, ok = [], True
    if cask:
        # (i) under CASK: over the eviction limit -> 422, no required_max_seq.
        ok_i, note = check_cask_over_limit(d, cask)
        notes.append(note)
    else:
        # (i) state longer than the loaded max_seq -> 422 + required_max_seq.
        # max_seq/4 reps x 5-7 tokens: over max_seq, under Jev's limits.
        long_state = LOREM * (max_seq // 4)
        r = d.decide(long_state, QUESTIONS)
        e = r.get("error") or {}
        req = e.get("required_max_seq")
        ok_i = (e.get("status") == 422 and isinstance(req, int) and not isinstance(req, bool)
                and req > max_seq)
        notes.append(f"over-long: status={e.get('status')} required_max_seq={req} (> {max_seq}) -> {ok_i}")
    ok &= ok_i
    after_i = "answers" in d.decide(STATE, QUESTIONS)
    notes.append(f"decide after={after_i}")
    ok &= after_i
    # (ii) malformed request -> 422.
    r = d.decide(STATE, {})
    e2 = r.get("error") or {}
    ok_ii = e2.get("status") == 422
    notes.append(f"questions={{}}: status={e2.get('status')} -> {ok_ii}")
    ok &= ok_ii
    after_ii = "answers" in d.decide(STATE, QUESTIONS)
    notes.append(f"decide after={after_ii}")
    ok &= after_ii
    # (iii) Jev's per-question limit: state alone >= 40000 tokens.
    r = d.decide(LOREM * 8000, QUESTIONS)
    e3 = r.get("error") or {}
    ok_iii = (e3.get("status") == 422 and "required_max_seq" not in e3
              and f"per-question limit of {JEV_QUESTION_LIMIT}" in e3.get("message", ""))
    notes.append(f"state >32000 tokens: status={e3.get('status')} message={e3.get('message')!r} -> {ok_iii}")
    ok &= ok_iii
    # (iv) Jev's request limit: 4 questions of 20000-28000 tokens each (each
    # under 32000, together over 64000 with a short shared state).
    long_q = {f"q{i}": {"type": "noul", "instructions": f"{i} " + LOREM * 4000} for i in range(4)}
    r = d.decide("s", long_q)
    e4 = r.get("error") or {}
    ok_iv = (e4.get("status") == 422 and "required_max_seq" not in e4
             and f"request limit of {JEV_REQUEST_LIMIT}" in e4.get("message", ""))
    notes.append(f"questions >64000 tokens: status={e4.get('status')} message={e4.get('message')!r} -> {ok_iv}")
    ok &= ok_iv
    after_iv = "answers" in d.decide(STATE, QUESTIONS)
    notes.append(f"decide after={after_iv}")
    ok &= after_iv
    return ok, "; ".join(notes)


INCONCLUSIVE = "INCONCLUSIVE"
LABEL = {True: "PASS", False: "FAIL", INCONCLUSIVE: "INCONCLUSIVE"}


def run(name, fn, *args):
    try:
        res = fn(*args)
    except Exception as ex:  # noqa: BLE001 - a gate that raises is a FAIL
        traceback.print_exc()
        res = (False, f"raised {type(ex).__name__}: {ex}")
    ok, detail = res
    print(f"{LABEL[ok]:<12} gate {name}: {detail}", flush=True)
    return name, res


def with_daemon(model, max_seq, log, env, body, cask=None):
    d = Daemon(stderr=log, env=env)
    try:
        info = d.load(model, max_seq=max_seq, params=cask)
        print(f"loaded {model}: arch={info.get('arch')} max_seq={max_seq}"
              f"{' env=' + json.dumps(env) if env else ''}"
              f"{' cask=' + json.dumps(cask) if cask else ''}", flush=True)
        pre = [run("0 CASK configured", gate_cask_configured, d, cask)] if cask else []
        return pre + body(d)
    finally:
        d.close()


def serve_cask_params(args):
    """The cask* load params `hipfire serve` sends (crates/hipfire-cli/src/main.rs
    load params), defaults from ~/.hipfire/config.json."""
    cfg_path = Path.home() / ".hipfire/config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    sidecar = args.cask_sidecar or cfg.get("cask_sidecar", "")
    if not sidecar or not Path(sidecar).is_file():
        raise SystemExit(f"--cask needs a sidecar file (got {sidecar!r}); pass --cask-sidecar")
    return {"cask": True, "cask_sidecar": sidecar,
            "cask_budget": int(cfg.get("cask_budget", 16384)),
            "cask_beta": int(cfg.get("cask_beta", 128)),
            "cask_core_frac": float(cfg.get("cask_core_frac", 0.5)),
            "cask_fold_m": int(cfg.get("cask_fold_m", 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-seq", type=int, default=None,
                    help="default 8192; with --cask the serve config max_seq (131072)")
    ap.add_argument("--cask", action="store_true",
                    help="load every daemon with CASK eviction configured as serve does")
    ap.add_argument("--cask-sidecar", help="override config.json cask_sidecar")
    ap.add_argument("--leak-n", type=int, default=1000)
    ap.add_argument("--daemon-log", help="redirect daemon/probe stderr to this file prefix")
    a = ap.parse_args()
    cask = serve_cask_params(a) if a.cask else None
    if a.max_seq is None:
        a.max_seq = 131072 if cask else 8192
    logs = []

    def log(suffix):
        if not a.daemon_log:
            return None
        f = open(f"{a.daemon_log}.{suffix}", "w")
        logs.append(f)
        return f

    results = []
    try:
        # Sequential: probe, then exact daemon, then default daemon (one GPU
        # model process at a time).
        floor = None
        try:
            floor = measure_floor(a.model, f"{a.daemon_log}.probe" if a.daemon_log else None)
            print(f"noise floor measured (one-call vs token-by-token, default env): max={floor['max']:.4f}",
                  flush=True)
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            results.append(("1b default-mode noise floor", (False, f"floor probe failed: {ex}")))
            print(f"FAIL         gate 1b default-mode noise floor: floor probe failed: {ex}", flush=True)

        results += with_daemon(a.model, a.max_seq, log("exact"), EXACT_ENV, lambda d: [
            run("1a exact-mode bit-exactness", gate_exact_mode, d)], cask)

        def default_body(d):
            r = []
            if floor is not None:
                r.append(run("1b default-mode noise floor", gate_noise_floor, d, floor))
            r += [run("1c restore order-invariance", gate_restore_order, d),
                  run("2 question isolation", gate_isolation, d),
                  run("3 model left clean", gate_clean, d),
                  run("4 no leak", gate_leak, d, a.leak_n),
                  run("5 error replies", gate_errors, d, a.max_seq, cask)]
            return r
        results += with_daemon(a.model, a.max_seq, log("default"), None, default_body, cask)
    finally:
        for f in logs:
            f.close()
    counts = {lab: sum(LABEL[ok] == lab for _, (ok, _) in results) for lab in LABEL.values()}
    print(f"summary: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['INCONCLUSIVE']} INCONCLUSIVE "
          f"(INCONCLUSIVE does not fail the run)", flush=True)
    sys.exit(1 if counts["FAIL"] or not results else 0)


if __name__ == "__main__":
    main()
