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
  2   question isolation, baseline-relative, exact mode (a sibling question
      shifts the default-mode split point, and split-point drift alone can
      move answers by up to the noise floor -- see 1b -- which would swamp
      the 0.05 threshold): PASS iff
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
  S1  session cache reuse: a session decide between two chat turns leaves
      the chat prompt cache usable. (a) no delta, default env: turn 2 text
      bit-identical to a run without the decide, decide start=extend with 0
      conversation prefill, turn 2 cached_tokens == decide prefix_tokens.
      (b) delta (decide includes the next user turn), exact env: same text,
      conversation prefill > 0, turn 2 cached_tokens == E. INCONCLUSIVE when
      the no-decide baseline gets no cache hit (no chat cache: Llama
      carrier, --cask).
  S2  session exactness (exact env), split like gate 1 (human ruling): vs a
      one-call full prefill of the same session tokens (`_debug_no_snapshot`,
      run last: it ends reset and clears the assistant-turn cache).
      (a) prefill-built conversations (a cold start with another
      conversation cached; an extend over the conversation that decide
      committed): Δ == 0. (b) decode-built (a warm extend right after the
      chat turn, whose reply the model generated token by token): per
      question drift <= 2 x the one-call vs token-by-token floor, measured
      per run by split_prefill_probe in the exact env on a v1 decide over
      CONV_C's text with the session questions (the probe loads with
      max_seq 512, so S4's long conversation reuses this floor).
  S3  session no leak: N session decides cycling three conversations (two
      share a ~2.5k-token system prefix; a chat turn per cycle re-creates the
      prefill checkpoints a cold start drops), so extend, resume and cold are
      all exercised and counted; fdinfo growth < 64 MiB. INCONCLUSIVE if the
      memory criterion passes but resume never ran (on an arch with prefill
      checkpoints; see below).
  S4  session stale cache (exact env): with an unrelated conversation
      cached (cold) and with one sharing a long prefix cached (resume), the
      answers equal a one-call full prefill (Δ == 0, so resume == cold
      exactly: both prefill-built); the warm answers W (extend right after
      the chat turn: decode-built) are within 2 x the floor per question, as
      S2 (b); and the next chat turn reuses the decide's conversation. INCONCLUSIVE if resume was not exercised;
      then S4r reruns it with a shared prefix longer than the first prefill
      chunk + 2048 tokens, so a checkpoint lies inside it.
  S5  session error replies: both / neither / empty / non-list messages ->
      422; an over-long conversation (over max_seq, or over the eviction
      limit with --cask) with 128 questions -> 422 (+ required_max_seq >
      max_seq without --cask) in <= 4x the time of the same refusal with 1
      question + 0.25 s (refused after the probe and one question render,
      not 128); then a session decide still extends with zero delta.
  S1-S5 are INCONCLUSIVE when the model has no Jinja chat template (session
  mode's spec §12.5 400). On an arch that is not `cache_capable` (the Llama
  carrier) the chat path keeps no conversation: S1 is INCONCLUSIVE, S2 and
  S4 compare answers only (as under --cask), and S3/S5 still expect the
  decide's own extend. An arch without prefill checkpoints (not in
  CHECKPOINT_ARCHES: the Llama carrier) cannot resume, so there S3 passes on
  extend + cold and S4 is PASS/FAIL on the answers, never INCONCLUSIVE for
  a missing resume.
  S6  (--draft only: a daemon loaded with a DFlash drafter; runs S6 alone,
      with HIPFIRE_QWEN_CACHE_TRACE=1) session decide with a speculator is
      read-only. (a) on the active conversation: timing committed=false,
      start=extend reusing the whole cached conversation, and the next chat
      turn's greedy text and cached_tokens equal a no-decide baseline (the
      text is weak evidence, spec verification being lossless; the
      cached_tokens equality is the check; MTP is not gated). (b) on an unrelated
      conversation (start=cold, rollback): the next chat turn over a history
      holding the earlier assistant turn still splices it (every
      `jinja lookup` trace line in daemon stderr says hit=true). (c) no leak
      over N session decides (extend and cold, a chat turn re-establishing
      the cache after each cold).

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
import time
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


def measure_floor(model, log_path, request=None, env=None):
    """A (one call) vs C (token by token) on the decide prompts of `request`
    (default: the v1 STATE/QUESTIONS), default env unless `env` is given."""
    if not PROBE.exists():
        raise RuntimeError(f"{PROBE} missing: cargo build --release -p hipfire-generate "
                           "--features lab --example split_prefill_probe")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(request or {"state": STATE, "questions": QUESTIONS}, fh)
        req = fh.name
    try:
        with open(log_path or os.devnull, "w") as log:
            out = subprocess.run([str(PROBE), str(model)],
                                 env={**os.environ, **(env or {}), "PROBE_DECIDE_REQUEST": req},
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
    detail = (f"exact env {EXACT_ENV}: P(code): no code anywhere = {none:.3f}, "
              f"code in sibling question = {sib:.3f} "
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


def measure_leak(d, n, step, what):
    """Run step(i) n times. Growth of the daemon's own amdgpu GTT + VRAM
    (fdinfo) must stay < 64 MiB; the device-wide sysfs sum is printed
    alongside and is the fallback when fdinfo is unreadable."""
    g0, v0 = device_mem()
    p0 = process_mem(d.p.pid)
    for i in range(n):
        step(i)
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
    return ok, f"device memory growth over {n} {what}: {basis} (<64); {sys_s}"


def gate_leak(d, n):
    need_answers(d.decide(STATE, QUESTIONS), "warmup")
    return measure_leak(
        d, n, lambda i: need_answers(d.decide(STATE, QUESTIONS), f"leak iter {i}"), "decides")


def cask_limit(cask):
    """Decide's eviction-safe limit: min(cask_budget, physical_cap). The load
    requires max_seq >= budget + beta + 4 and physical_cap is clamped to at
    least that, so the budget always binds."""
    return cask["cask_budget"]


LOREM = "lorem ipsum dolor sit amet "   # 5-7 tokens per repetition
JEV_QUESTION_LIMIT = 32000
JEV_REQUEST_LIMIT = 64000

# ---- session mode (spec §12.8) ---------------------------------------------
SESSION_SYS = "You are a customer support agent for an online shop."
CONV_C = [{"role": "system", "content": SESSION_SYS},
          {"role": "user", "content": "Order #3527 was placed 21 days ago and tracking has not "
                                      "updated. Reply in one short sentence."}]
CONV_U = [{"role": "system", "content": "You are a poet."},
          {"role": "user", "content": "Write one short line about rain."}]
X_USER = {"role": "user", "content": "Name three primary colours in one short sentence."}
# A 2-3k-token system turn shared by CONV_L and CONV_X: the chat prefill of
# one leaves a DeltaNet checkpoint inside the prefix the other shares (the
# first checkpoint lands at the first prefill chunk, ar.rs:3880).
LONG_SYS = SESSION_SYS + " Policy notes: " + LOREM * 400
# S4r: a shared prefix longer than the first prefill chunk (256-512 tokens)
# + ckpt_interval (2048), for a rerun when S4 found no checkpoint inside it.
LONG_SYS_RERUN = SESSION_SYS + " Policy notes: " + LOREM * 900
CONV_L = [{"role": "system", "content": LONG_SYS}, CONV_C[1]]
CONV_X = [{"role": "system", "content": LONG_SYS}, X_USER]
FOLLOW = {"role": "user", "content": "Thanks. What should I do next? Reply in one short sentence."}
SESSION_Q = {
    "escalate": {"type": "noul",
                 "instructions": "Should this conversation be escalated to a human agent?"},
    "topic": {"type": "choice", "instructions": "What is the conversation about?",
              "criteria": {"shipping": "Delivery, tracking, lost parcels",
                           "billing": "Payments, refunds", "other": "Anything else"}},
    "mood": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Furious"]},
}


def chat_turn(d, messages, max_tokens=1024):
    """One greedy chat turn; returns the conversation extended by the reply.
    The chat path does not store a length-capped turn in its prompt cache,
    which would make the reuse gates vacuous, so that fails the setup. The
    Llama-carrier generate route reports no `finish_reason` (nor
    `cached_tokens`); there a turn that stopped short of `max_tokens` ended
    on EOS."""
    text, done = d.chat(messages, max_tokens=max_tokens)
    reason = done.get("finish_reason")
    ended = reason == "stop" if reason is not None else done.get("tokens", max_tokens) < max_tokens
    if not ended:
        raise AssertionError(f"setup: chat turn finish_reason={reason!r} tokens="
                             f"{done.get('tokens')} (need 'stop', or < {max_tokens} tokens)")
    return messages + [{"role": "assistant", "content": text}]


def session(d, messages, what, **extra):
    r = d.session_decide(messages, SESSION_Q, **extra)
    return need_answers(r, what), r.get("timing", {})


def session_unavailable(d):
    """The spec §12.5 400 for a model without a Jinja chat template (e.g. an
    HFQ file with none embedded and no override): session mode cannot run,
    so the session gates are INCONCLUSIVE rather than FAIL. Returns the
    message, or None when session mode is available."""
    d.reset()
    r = d.session_decide(CONV_C, {"escalate": SESSION_Q["escalate"]})
    e = r.get("error") or {}
    if e.get("status") == 400 and "chat template" in e.get("message", ""):
        return e["message"]
    return None


# Archs whose chat path takes DeltaNet prefill checkpoints (Qwen35Carrier,
# arch 5/6), the only ones a session decide can resume from (spec §12.2).
# The Llama carrier keeps none: its starts are extend or cold only.
CHECKPOINT_ARCHES = {"qwen3_5", "qwen3_5_moe"}


def resumable(d):
    return d.info.get("arch") in CHECKPOINT_ARCHES


def run_session_gates(d, gates):
    """Run [(name, fn, *args)] session gates, or report each INCONCLUSIVE
    when the loaded model has no chat template for session mode."""
    why = session_unavailable(d)
    if why is None:
        return [run(name, fn, *args) for name, fn, *args in gates]
    res = []
    for name, *_ in gates:
        detail = f"session mode unavailable on this model: 400 {why!r}"
        print(f"{LABEL[INCONCLUSIVE]:<12} gate {name}: {detail}", flush=True)
        res.append((name, (INCONCLUSIVE, detail)))
    return res


def max_dlogp(a, b):
    return max(dlogp(a[n], b[n]) for n in a)


def gate_session_reuse(d, with_delta, reuse=True):
    d.reset()
    conv = chat_turn(d, CONV_C)
    nxt = conv + [FOLLOW]
    base_text, base_done = d.chat(nxt)
    base_cached = base_done.get("cached_tokens", 0)
    if not reuse or base_cached == 0:
        return INCONCLUSIVE, (f"no-decide baseline cached_tokens={base_cached}: no chat prompt "
                              f"cache to preserve on this arch/config")
    d.reset()
    if chat_turn(d, CONV_C) != conv:
        return False, "setup: greedy turn 1 is not reproducible after reset"
    _, t = session(d, nxt if with_delta else conv, "session decide")
    text, done = d.chat(nxt)
    cached = done.get("cached_tokens", 0)
    cpt = t.get("conversation_prefill_tokens")
    delta_ok = (cpt or 0) > 0 if with_delta else cpt == 0
    ok = (text == base_text and len(text) > 0 and t.get("start") == "extend" and delta_ok
          and cached == t.get("prefix_tokens") and cached >= base_cached)
    return ok, (f"{'delta' if with_delta else 'no delta'}: next-turn text identical="
                f"{text == base_text}; decide start={t.get('start')} (extend), "
                f"conversation_prefill_tokens={cpt} ({'>0' if with_delta else '==0'}); "
                f"next turn cached_tokens={cached} (== decide prefix_tokens "
                f"{t.get('prefix_tokens')}, >= no-decide {base_cached}); "
                f"texts {base_text[:40]!r} / {text[:40]!r}")


def conv_text(messages):
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def session_floor_request(messages):
    """The S2/S4 floor probe request: a v1 decide over the conversation's text
    with the session questions, so the floor is per SESSION_Q question."""
    return {"state": conv_text(messages), "questions": SESSION_Q}


def drift_vs_floor(ans, ref, floor, source):
    """Human ruling (spec §12.8): a conversation the model partly generated by
    token-by-token decode is not bit-identical to a prefill of the same
    tokens (the precision property of gate 1b), so per question its drift
    vs the one-call reference must be <= FLOOR_FACTOR x the measured
    one-call vs token-by-token floor."""
    if floor is None:
        return False, f"decode-built leg: no floor ({source} failed)"
    rows, ok = [], True
    for n in ans:
        drift = dlogp(ans[n], ref[n])
        pq = floor["per_question"][n]
        fl = dlogp(pq["a_answer"], pq["c_answer"])
        q_ok = drift <= FLOOR_FACTOR * fl
        ok &= q_ok
        rows.append(f"{n}: drift={drift:.4f} 2xfloor={FLOOR_FACTOR * fl:.4f} {'ok' if q_ok else 'OVER'}")
    return ok, f"[{'; '.join(rows)}] (floor: {source})"


def gate_session_exact(d, reuse=True, floor=None, floor_src="n/a"):
    """(a) prefill-built conversations: Δ == 0 vs the one-call full prefill
    (cold start with another conversation cached; extend over the
    conversation that cold decide committed). (b) decode-built: a warm
    extend right after the chat turn, drift <= 2 x floor per question.
    Without a chat prompt cache (--cask, Llama carrier) the warm leg starts
    cold, is prefill-built and must be exact too."""
    d.reset()
    conv = chat_turn(d, CONV_C)
    warm, tw = session(d, conv, "warm session")
    # Cache a different conversation (the chat path's cold start keeps the
    # assistant-turn cache, so conv's reply still splices).
    chat_turn(d, CONV_U)
    other, to = session(d, conv, "session with another conversation cached")
    # Extend over the conversation the cold decide above committed: KV and
    # DeltaNet built by prefill rather than by the chat turn's decode steps.
    warm_p, tp = session(d, conv, "warm session over a decide-committed conversation")
    # The reference last: it ends reset (v1 semantics) and clears the
    # assistant-turn cache, so it must not precede a warm-session answer.
    ref, _ = session(d, conv, "full-prefill reference", _debug_no_snapshot=True)
    do, dp = max_dlogp(other, ref), max_dlogp(warm_p, ref)
    starts_ok = (tw.get("start") == "extend" and to.get("start") != "extend"
                 and tp.get("start") == "extend") if reuse else True
    prefill_ok = do == 0.0 and dp == 0.0
    if reuse:
        warm_ok, warm_s = drift_vs_floor(warm, ref, floor, floor_src)
        warm_s = f"(b) decode-built: warm extend after the chat turn (start={tw.get('start')}) {warm_s}"
    else:
        dw = max_dlogp(warm, ref)
        warm_ok = dw == 0.0
        warm_s = f"warm (start={tw.get('start')}, prefill-built without a chat cache) Δ={dw:.3g} (==0)"
    ok = prefill_ok and warm_ok and starts_ok
    return ok, (f"exact env, max |Δlogp| vs one-call full prefill of the same tokens (ran last): "
                f"(a) prefill-built: another conversation cached (start={to.get('start')}) Δ={do:.3g}, "
                f"extend over the decide-committed conversation (start={tp.get('start')}, "
                f"conversation_prefill_tokens={tp.get('conversation_prefill_tokens')}) Δ={dp:.3g} "
                f"(both ==0); {warm_s}")


def gate_session_stale(d, reuse=True, long_sys=LONG_SYS, decide_reuse=True, floor=None,
                       floor_src="n/a", resumable=True):
    """W = answers with the conversation's own chat turn cached (decode-built
    when there is a chat cache). (a) prefill-built: the cold start (unrelated
    conversation cached) and the resume (shared-prefix conversation cached)
    each Δ == 0 vs the one-call full prefill, so resume == cold exactly.
    (b) W vs that reference: drift <= 2 x floor per question (Δ == 0 when W
    started cold). The reference runs last. On an arch without prefill
    checkpoints (`resumable` false: the Llama carrier) resume cannot run, so
    the answers are compared only and the verdict is PASS/FAIL."""
    conv_l = [{"role": "system", "content": long_sys}, CONV_C[1]]
    conv_x = [{"role": "system", "content": long_sys}, X_USER]
    d.reset()
    conv = chat_turn(d, conv_l)
    w, tw = session(d, conv, "own conversation cached")
    chat_turn(d, CONV_U)
    cold, tc = session(d, conv, "unrelated conversation cached")
    chat_turn(d, conv_x)
    res, tr = session(d, conv, "shared-prefix conversation cached")
    _, done = d.chat(conv + [FOLLOW], max_tokens=16)
    ref, _ = session(d, conv, "full-prefill reference", _debug_no_snapshot=True)
    dc, dr, drc = max_dlogp(cold, ref), max_dlogp(res, ref), max_dlogp(res, cold)
    ok = dc == 0.0 and dr == 0.0 and drc == 0.0
    if tw.get("start") == "extend":
        w_ok, w_s = drift_vs_floor(w, ref, floor, floor_src)
        w_s = f"(b) decode-built W (start=extend after the chat turn) {w_s}"
    else:
        dw = max_dlogp(w, ref)
        w_ok = dw == 0.0
        w_s = f"W (start={tw.get('start')}, prefill-built) Δ={dw:.3g} (==0)"
    ok = ok and w_ok
    if reuse:
        ok = (ok and tw.get("start") == "extend" and tc.get("start") == "cold"
              and done.get("cached_tokens") == tr.get("prefix_tokens"))
    detail = (f"exact env, E={tw.get('prefix_tokens')} tokens, vs one-call full prefill (ran last): "
              f"(a) prefill-built: unrelated cache start={tc.get('start')} Δ={dc:.3g}; shared-prefix "
              f"cache start={tr.get('start')} (cached_tokens={tr.get('cached_tokens')}) Δ={dr:.3g}; "
              f"resume vs cold Δ={drc:.3g} (all ==0); {w_s}; "
              f"next chat cached_tokens={done.get('cached_tokens')} (== {tr.get('prefix_tokens')})")
    if ok and reuse and tr.get("start") != "resume":
        return INCONCLUSIVE, detail + " — no checkpoint inside the shared prefix: resume not exercised"
    if ok and decide_reuse and resumable and not reuse and tr.get("start") != "resume":
        return INCONCLUSIVE, detail + (" — no chat prompt cache on this arch (answers compared only): "
                                       "resume not exercised")
    return ok, detail


S3_CYCLE = 6   # session decides per S3 cycle


def gate_session_leak(d, n, reuse=True, resumable=True):
    """Cycles of: a chat turn on CONV_X (re-creates the prefill checkpoints
    in the shared LONG_SYS prefix that a cold start drops), then decides on
    X (extend), L (resume from a checkpoint in the shared prefix), L
    (extend), X (resume), U (cold), U (extend). A cycle ends in the state it
    started in, so whole cycles are measured after one warmup cycle. On an
    arch without prefill checkpoints (`resumable` false: the Llama carrier)
    resume cannot run, so extend and cold suffice."""
    d.reset()
    conv_l = chat_turn(d, CONV_L)
    conv_u = chat_turn(d, CONV_U)
    starts = {}

    def cycle(i, count=True):
        conv_x = chat_turn(d, CONV_X)
        for j, c in enumerate([conv_x, conv_l, conv_l, conv_x, conv_u, conv_u]):
            _, t = session(d, c, f"leak cycle {i} decide {j}")
            if count:
                starts[t.get("start")] = starts.get(t.get("start"), 0) + 1

    cycle(-1, count=False)
    cycles = max(1, -(-n // S3_CYCLE))
    ok, detail = measure_leak(d, cycles, cycle,
                              f"cycles ({cycles * S3_CYCLE} session decides + {cycles} chat turns)")
    detail = f"{detail}; starts={starts}"
    if not reuse:
        return ok and set(starts) == {"cold"}, detail + " (need only cold)"
    if not resumable:
        return ok and {"extend", "cold"} <= set(starts), detail + (
            " (need extend and cold; no prefill checkpoints on this arch, so no resume)")
    if not ok or not {"extend", "cold"} <= set(starts):
        return False, detail + " (need extend, cold and resume)"
    if "resume" not in starts:
        return INCONCLUSIVE, detail + " — resume not exercised"
    return True, detail + " (need extend, cold and resume)"


def session_oversized(d, max_seq, cask):
    """An over-long conversation (over max_seq; with --cask over the eviction
    limit) is refused after the probe and R_0 only, however many questions
    (spec §12.5): the 128-question refusal must take about as long as the
    1-question one, well below 128 whole-conversation renders."""
    limit = cask_limit(cask) if cask else max_seq
    big = [{"role": "system", "content": SESSION_SYS + " Policy notes: " + LOREM * (limit // 4)},
           CONV_C[1]]
    many = {f"q{i:03}": SESSION_Q["escalate"] for i in range(128)}
    times, errs = [], []
    for qs in ({"q000": SESSION_Q["escalate"]}, many):
        t0 = time.monotonic()
        r = d.session_decide(big, qs)
        times.append(time.monotonic() - t0)
        errs.append(r.get("error") or {})
    e1, e = errs
    if cask:
        shape = (e.get("status") == 422 and "required_max_seq" not in e
                 and f"limited to {limit} tokens" in e.get("message", ""))
    else:
        req = e.get("required_max_seq")
        shape = (e.get("status") == 422 and isinstance(req, int) and not isinstance(req, bool)
                 and req > max_seq)
    shape = shape and e1.get("status") == 422
    fast = times[1] <= FAST_REFUSAL_FACTOR * times[0] + 0.25
    return shape and fast, (f"over-long conversation x128 questions: status={e.get('status')} "
                            f"required_max_seq={e.get('required_max_seq')} "
                            f"({'absent' if cask else f'> {max_seq}'}) message={e.get('message')!r}; "
                            f"{times[1]:.3f}s vs 1 question {times[0]:.3f}s "
                            f"(<= {FAST_REFUSAL_FACTOR}x + 0.25s) -> {shape and fast}")


FAST_REFUSAL_FACTOR = 4


def gate_session_errors(d, reuse=True, max_seq=8192, cask=None):
    d.reset()
    conv = chat_turn(d, CONV_C)
    session(d, conv, "commit")
    notes, ok = [], True
    ok_o, note = session_oversized(d, max_seq, cask)
    notes.append(note)
    ok &= ok_o
    for label, fields in [("state and messages", {"state": "s", "messages": conv}),
                          ("neither", {}),
                          ("empty messages", {"messages": []}),
                          ("messages not a list", {"messages": "hi"})]:
        d.send({"type": "decide", "id": "e", "questions": SESSION_Q, **fields})
        e = d.recv_until({"decided", "error"}).get("error") or {}
        ok_i = e.get("status") == 422
        notes.append(f"{label}: status={e.get('status')} message={e.get('message')!r} -> {ok_i}")
        ok &= ok_i
    _, t = session(d, conv, "after refusals")
    want = "extend" if reuse else "cold"
    after_ok = t.get("start") == want and (not reuse or t.get("conversation_prefill_tokens") == 0)
    notes.append(f"session decide after: start={t.get('start')} (=={want}), "
                 f"conversation_prefill_tokens={t.get('conversation_prefill_tokens')}")
    return ok and after_ok, "; ".join(notes)


# ---- S6: session decide with a DFlash speculator loaded (read-only) --------
def gate_spec_active(d):
    """Read-only session decide with a DFlash speculator loaded. The
    next-turn text identity is weak evidence: speculative decoding is
    lossless (every drafted token is verified by the target model), so the
    greedy text would match even if the decide had disturbed the drafter's
    context. The meaningful check is `cached_tokens` equality: the decide's
    reuse and the next turn's cache hit both equal the no-decide baseline,
    so the target cache was left exactly as found. MTP speculators are not
    gated (only a DFlash drafter is loaded here)."""
    d.reset()
    conv = chat_turn(d, CONV_C)
    nxt = conv + [FOLLOW]
    base_text, base_done = d.chat(nxt)
    base_cached = base_done.get("cached_tokens", 0)
    d.reset()
    if chat_turn(d, CONV_C) != conv:
        return False, "setup: greedy turn 1 is not reproducible after reset"
    _, t = session(d, conv, "session decide, speculator loaded")
    text, done = d.chat(nxt)
    cached = done.get("cached_tokens", 0)
    # The decide reuses exactly the cached conversation (its `cached_tokens`
    # == the baseline's). Its delta need not be 0: the DFlash chat path's
    # cache ends at `<|im_end|>` without the `\n` trailer the render has,
    # so E is one token past it (qwen-cache ids trace).
    ok = (t.get("committed") is False and t.get("start") == "extend"
          and t.get("cached_tokens") == base_cached
          and text == base_text and len(text) > 0 and cached == base_cached and cached > 0)
    return ok, (f"decide committed={t.get('committed')} (false), start={t.get('start')} (extend), "
                f"decide cached_tokens={t.get('cached_tokens')} (== no-decide next-turn "
                f"{base_cached}), conversation_prefill_tokens={t.get('conversation_prefill_tokens')}, "
                f"prefix_tokens={t.get('prefix_tokens')}; next turn text identical="
                f"{text == base_text}, cached_tokens={cached} (== no-decide {base_cached}, >0); "
                f"texts {base_text[:40]!r} / {text[:40]!r}")


LOOKUP_RE = re.compile(r"\[qwen-cache jinja lookup[^\]]*\].*?\bhit=(true|false)")


def gate_spec_cold_keeps_turns(d, log_path):
    if not log_path:
        return False, "needs the daemon stderr log (HIPFIRE_QWEN_CACHE_TRACE lines)"
    d.reset()
    conv = chat_turn(d, CONV_C)
    _, t = session(d, CONV_U, "session decide on an unrelated conversation")
    off = os.path.getsize(log_path)
    text, done = d.chat(conv + [FOLLOW])
    with open(log_path, errors="replace") as fh:
        fh.seek(off)
        hits = LOOKUP_RE.findall(fh.read())
    ok = (t.get("start") == "cold" and t.get("committed") is False and len(hits) > 0
          and all(h == "true" for h in hits) and len(text) > 0)
    return ok, (f"unrelated decide start={t.get('start')} (cold), committed={t.get('committed')}; "
                f"next chat turn jinja lookups hit={hits} (all true, >=1), "
                f"cached_tokens={done.get('cached_tokens')}, text {text[:40]!r}")


def gate_spec_leak(d, n):
    """Pairs of decides on the cached conversation (extend, read-only) and
    one on an unrelated conversation (cold: the rollback empties the cache),
    then a chat turn re-establishes the cache. 3 decides per cycle."""
    d.reset()
    conv_c = chat_turn(d, CONV_C)
    starts = {}

    def cycle(i, count=True):
        for j, c in enumerate([conv_c, conv_c, CONV_U]):
            _, t = session(d, c, f"leak cycle {i} decide {j}")
            if count:
                starts[t.get("start")] = starts.get(t.get("start"), 0) + 1
        if chat_turn(d, CONV_C) != conv_c:
            raise AssertionError(f"leak cycle {i}: re-established turn 1 differs")

    cycle(-1, count=False)
    cycles = max(1, -(-n // 3))
    ok, detail = measure_leak(d, cycles, cycle,
                              f"cycles ({cycles * 3} session decides + {cycles} chat turns)")
    covered = {"extend", "cold"} <= set(starts)
    return ok and covered, f"{detail}; starts={starts} (need extend and cold)"


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
        # max_seq/4 reps x 5-7 tokens: over max_seq, under Jev's per-question
        # limit (32000). Guard against exceeding the per-question limit.
        reps = max_seq // 4
        long_state = LOREM * reps
        # Rough estimate: 5-7 tokens per LOREM rep; verify not exceeding 32k.
        # If it would, skip the test (clear sign max_seq is too large).
        estimated_tokens = reps * 7 + len(QUESTIONS) * 100  # rough full-prompt estimate
        if estimated_tokens >= JEV_QUESTION_LIMIT:
            raise AssertionError(f"max_seq {max_seq} too large: over-length test would hit "
                                 f"Jev's per-question limit ({JEV_QUESTION_LIMIT}) instead of "
                                 f"required_max_seq, making the gate ineffective")
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


def with_daemon(model, max_seq, log, env, body, cask=None, params=None):
    d = Daemon(stderr=log, env=env)
    try:
        info = d.load(model, max_seq=max_seq, params={**(cask or {}), **(params or {})})
        d.info = info
        print(f"loaded {model}: arch={info.get('arch')} max_seq={max_seq}"
              f"{' env=' + json.dumps(env) if env else ''}"
              f"{' cask=' + json.dumps(cask) if cask else ''}"
              f"{' params=' + json.dumps(params) if params else ''}"
              f" cache_capable={info.get('cache_capable')}", flush=True)
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
    ap.add_argument("--draft", help="DFlash drafter path: run only gate S6 on a daemon loaded "
                                    "with it (load param `draft`)")
    a = ap.parse_args()
    cask = serve_cask_params(a) if a.cask else None
    if a.max_seq is None:
        a.max_seq = 131072 if cask else 8192
    if a.draft:
        return main_draft(a)
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

        # Session floor for the decode-built legs of S2/S4 (human ruling):
        # one-call vs token-by-token, exact env (the env S2/S4 run in), on a
        # v1 decide over CONV_C's text with the session questions. The probe
        # loads with max_seq 512, so S4's ~2k-token conversation cannot be
        # probed; S4 uses the same per-question floor (stated in its detail).
        # Not needed under --cask (no chat cache, so no decode-built leg).
        fl_short = (None, "not measured (--cask: no decode-built leg)")
        if cask is None:
            src = "exact-env split_prefill_probe one-call vs token-by-token, CONV_C text"
            try:
                f = measure_floor(a.model, f"{a.daemon_log}.probe-session" if a.daemon_log else None,
                                  session_floor_request(CONV_C), EXACT_ENV)
                fl_short = (f, src)
                print(f"session floor ({src}): " + ", ".join(
                    f"{n}={dlogp(v['a_answer'], v['c_answer']):.4f}"
                    for n, v in f["per_question"].items()), flush=True)
            except Exception as ex:  # noqa: BLE001 - the decode-built legs then FAIL
                traceback.print_exc()
                fl_short = (None, f"{src}: probe failed: {ex}")
        fl_long = (fl_short[0], fl_short[1] + " (the probe's max_seq 512 cannot host the "
                                              "~2k-token S4 conversation)")

        # Two notions of reuse (spec §12.2, §12.5): the decide's own (off under
        # eviction: every start cold), and the chat path's prompt cache (also
        # needs a `cache_capable` arch; the Llama carrier's chat path keeps no
        # conversation, so there a chat turn is never extended and the gates
        # that need one compare answers only, as under --cask).
        decide_reuse = cask is None

        def chat_cache(d):
            return decide_reuse and bool(d.info.get("cache_capable"))

        def exact_body(d):
            cc = chat_cache(d)
            r = [run("1a exact-mode bit-exactness", gate_exact_mode, d),
                 run("2 question isolation", gate_isolation, d)]
            r += run_session_gates(d, [
                ("S1b session reuse with delta (exact)", gate_session_reuse, d, True, cc),
                ("S2 session exactness (exact)", gate_session_exact, d, cc, *fl_short),
                ("S4 session stale cache (exact)", gate_session_stale, d, cc, LONG_SYS,
                 decide_reuse, *fl_long, resumable(d))])
            if cc and r[-1][1][0] == INCONCLUSIVE and "resume not exercised" in r[-1][1][1]:
                # fl_long is the CONV_C floor (the probe cannot host the long
                # conversation; stated in the detail), reused for the rerun.
                r.append(run("S4r session stale cache, longer shared prefix (exact)",
                              gate_session_stale, d, cc, LONG_SYS_RERUN, decide_reuse, *fl_long,
                              resumable(d)))
            return r
        results += with_daemon(a.model, a.max_seq, log("exact"), EXACT_ENV, exact_body, cask)

        def default_body(d):
            r = []
            if floor is not None:
                r.append(run("1b default-mode noise floor", gate_noise_floor, d, floor))
            r += [run("1c restore order-invariance", gate_restore_order, d),
                  run("3 model left clean", gate_clean, d),
                  run("4 no leak", gate_leak, d, a.leak_n),
                  run("5 error replies", gate_errors, d, a.max_seq, cask)]
            r += run_session_gates(d, [
                ("S1a session reuse, no delta", gate_session_reuse, d, False, chat_cache(d)),
                ("S3 session no leak", gate_session_leak, d, a.leak_n, decide_reuse,
                 resumable(d)),
                ("S5 session error replies", gate_session_errors, d, decide_reuse, a.max_seq,
                 cask)])
            return r
        results += with_daemon(a.model, a.max_seq, log("default"), None, default_body, cask)
    finally:
        for f in logs:
            f.close()
    return summarize(results)


def summarize(results):
    counts = {lab: sum(LABEL[ok] == lab for _, (ok, _) in results) for lab in LABEL.values()}
    print(f"summary: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['INCONCLUSIVE']} INCONCLUSIVE "
          f"(INCONCLUSIVE does not fail the run)", flush=True)
    return 1 if counts["FAIL"] or not results else 0


def main_draft(a):
    """Gate S6 only: one default-env daemon with the DFlash drafter loaded and
    the prompt-cache trace on (S6b reads it from the daemon stderr log)."""
    if a.cask:
        raise SystemExit("--draft does not combine with --cask")
    if a.daemon_log:
        log_path = f"{a.daemon_log}.draft"
    else:
        fd, log_path = tempfile.mkstemp(prefix="jev-gates-draft-", suffix=".log")
        os.close(fd)
    with open(log_path, "w") as log:
        results = with_daemon(a.model, a.max_seq, log, {"HIPFIRE_QWEN_CACHE_TRACE": "1"}, lambda d: [
            run("S6a speculator: session decide on the active conversation", gate_spec_active, d),
            run("S6b speculator: cold session decide keeps assistant turns",
                gate_spec_cold_keeps_turns, d, log_path),
            run("S6c speculator: session no leak", gate_spec_leak, d, a.leak_n)],
            params={"draft": a.draft})
    print(f"daemon stderr: {log_path}", flush=True)
    return summarize(results)


if __name__ == "__main__":
    sys.exit(main())
