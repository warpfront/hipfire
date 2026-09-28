"""Decide GPU correctness gates (spec §10). Exit 0 iff all pass.

Gates:
  1   snapshot exactness: 3-question snapshot decide vs _debug_no_snapshot
      (full prefill per question); also checks the template closed thinking
      (no 400 "cannot disable thinking").
  1b  per-question isolation from restore: each question in the 3-question
      snapshot request vs a single-question decide of that question alone.
  1c  restore order-invariance: each question's answer is identical whether
      it runs first (no restore) or after restores, at a fixed split point.
  2   question isolation: sibling question text does not move a probe.
  3   model left clean: greedy generate identical before/after a decide.
  4   no device-memory leak over N decides.
  5   error replies end to end: 422 + required_max_seq for an over-long
      state, 422 for a malformed request; a normal decide succeeds after each.
"""
import argparse
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daemon_client import Daemon  # noqa: E402

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
TOL = 0.05  # max |Δ log p|, shared by gates 1 and 1b


def dist(a):
    if a["type"] == "noul":
        return [a["noul"], 1 - a["noul"]]
    return list(a["probabilities"].values())


def dlogp(a, b):
    worst = 0.0
    for p, q in zip(dist(a), dist(b)):
        worst = max(worst, abs(math.log(max(p, 1e-12)) - math.log(max(q, 1e-12))))
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


def gate_exact(d):
    snap = d.decide(STATE, QUESTIONS)
    err = snap.get("error") or {}
    if err.get("status") == 400 and "thinking" in err.get("message", ""):
        return False, f"template cannot close thinking: {err}"
    full = d.decide(STATE, QUESTIONS, _debug_no_snapshot=True)
    sa = need_answers(snap, "snapshot")
    fa = need_answers(full, "full-prefill")
    per = {name: dlogp(sa[name], fa[name]) for name in QUESTIONS}
    worst = max(per.values())
    ok = worst < TOL and snap["timing"]["prefix_tokens"] > 0
    per_s = ", ".join(f"{k}={v:.4f}" for k, v in per.items())
    ans_s = "; ".join(f"{k}: {fmt_answer(sa[k])}" for k in QUESTIONS)
    return ok, (f"max |Δlogp| snapshot vs full-prefill = {worst:.4f} (<{TOL}) [{per_s}], "
                f"prefix_tokens={snap['timing']['prefix_tokens']}, "
                f"closed-think template OK (no 400); answers: {ans_s}")


def gate_restore_isolation(d):
    multi = need_answers(d.decide(STATE, QUESTIONS), "multi")
    per, diag = {}, {}
    for name, q in QUESTIONS.items():
        single = need_answers(d.decide(STATE, {name: q}), f"single {name}")
        per[name] = dlogp(multi[name], single[name])
        # Diagnostic: a single-question decide has no restore, only the
        # prefix/suffix split, so its Δ vs full prefill is pure chunk-boundary
        # numerics.
        full = need_answers(d.decide(STATE, {name: q}, _debug_no_snapshot=True), f"full {name}")
        diag[name] = dlogp(single[name], full[name])
    worst = max(per.values())
    ok = worst < TOL
    per_s = ", ".join(f"{k}={v:.4f}" for k, v in per.items())
    diag_s = ", ".join(f"{k}={v:.4f}" for k, v in diag.items())
    return ok, (f"max |Δlogp| multi-question vs single-question = {worst:.4f} (<{TOL}) [{per_s}]; "
                f"diag single-q split vs full-prefill [{diag_s}]")


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
    ok = worst < TOL and len(prefixes) == 1
    return ok, (f"max |Δlogp| across all {math.factorial(len(names))} question orderings = {worst:.6f} (<{TOL}), "
                f"prefix_tokens={sorted(prefixes)}")


def gate_isolation(d):
    q = {"type": "noul", "instructions": "Does the text contain the secret code word PELICAN?"}
    none = need_answers(d.decide("A short note about the weather.", {"probe": q}),
                        "no-code")["probe"]["noul"]
    base = need_answers(d.decide("A short note about the weather.", {
        "probe": q,
        "leak": {"type": "noul", "instructions": "The secret code word is PELICAN. Is it sunny?"},
    }), "sibling")["probe"]["noul"]
    in_state = need_answers(d.decide("A short note about the weather. The secret code word is PELICAN.",
                                     {"probe": q}), "in-state")["probe"]["noul"]
    ok = base < 0.2 and in_state > 0.8
    return ok, (f"P(code) with code in sibling question = {base:.3f} (<0.2), in state = {in_state:.3f} (>0.8), "
                f"baseline no code anywhere = {none:.3f}")


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


def gtt_used():
    for p in Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"):
        return int(p.read_text())
    for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"):
        return int(p.read_text())
    return None


def gate_leak(d, n):
    need_answers(d.decide(STATE, QUESTIONS), "warmup")
    before = gtt_used()
    for i in range(n):
        need_answers(d.decide(STATE, QUESTIONS), f"leak iter {i}")
    after = gtt_used()
    if before is None:
        return False, "no mem_info_gtt_used / mem_info_vram_used sysfs node — cannot measure"
    grew = after - before
    ok = grew < 64 * 1024 * 1024
    return ok, f"device memory growth over {n} decides = {grew / 2**20:.1f} MiB (<64)"


def gate_errors(d, max_seq):
    notes, ok = [], True
    # (i) state longer than the loaded max_seq -> 422 + required_max_seq.
    long_state = "lorem ipsum dolor sit amet " * (max_seq // 2)
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
    return ok, "; ".join(notes)


def run(name, fn, *args):
    try:
        res = fn(*args)
    except Exception as ex:  # noqa: BLE001 - a gate that raises is a FAIL
        traceback.print_exc()
        res = (False, f"raised {type(ex).__name__}: {ex}")
    ok, detail = res
    print(f"{'PASS' if ok else 'FAIL'}  gate {name}: {detail}", flush=True)
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--leak-n", type=int, default=1000)
    ap.add_argument("--daemon-log", help="redirect daemon stderr to this file")
    a = ap.parse_args()
    log = open(a.daemon_log, "w") if a.daemon_log else None
    d = Daemon(stderr=log)
    results = []
    try:
        info = d.load(a.model, max_seq=a.max_seq)
        print(f"loaded {a.model}: arch={info.get('arch')} max_seq={a.max_seq}", flush=True)
        results = [run("1 snapshot exactness", gate_exact, d),
                   run("1b restore isolation", gate_restore_isolation, d),
                   run("1c restore order-invariance", gate_restore_order, d),
                   run("2 question isolation", gate_isolation, d),
                   run("3 model left clean", gate_clean, d),
                   run("4 no leak", gate_leak, d, a.leak_n),
                   run("5 error replies", gate_errors, d, a.max_seq)]
    finally:
        d.close()
        if log:
            log.close()
    failed = sum(not ok for _, (ok, _) in results)
    print(f"summary: {len(results) - failed}/{len(results)} gates passed", flush=True)
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
