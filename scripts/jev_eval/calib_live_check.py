# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""GPU check L1 (spec §13.7): with the repo daemon driven directly, a
calibrated decide equals the offline apply_t of the raw decide, an explicit
T = 1 is bit-identical to no calibration, the timing echoes the temperatures,
and a malformed calibration is a 422. One model, held-out examples only.

Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE — either the raw readout does not repeat
bit for bit (two runs cannot be compared), or the session-mode precondition
(cold, then extend, extend) did not hold. INCONCLUSIVE is never reported as
PASS.

Only licence-clear sources are asked (licences.py, human decision)."""
import argparse
import json
import sys

sys.dont_write_bytecode = True
import calibrate as cb  # noqa: E402
import heldout  # noqa: E402
import licences  # noqa: E402
from daemon_client import Daemon  # noqa: E402

CAL = {"choice": 1.7, "score": 0.6, "noul": 2.3}
ONE = {"choice": 1.0, "score": 1.0, "noul": 1.0}
TOL = 1e-9


def probs_of(ans, keys):
    return [ans["noul"], 1 - ans["noul"]] if ans["type"] == "noul" else [ans["probabilities"][k] for k in keys]


def check_pair(raw, cal, keys, t):
    """(problems, max |dp|) for one question's raw and calibrated answers."""
    bad = []
    rp, cp = probs_of(raw, keys), probs_of(cal, keys)
    d = max(abs(x - y) for x, y in zip(cb.apply_t(rp, t), cp))
    if d > TOL:
        bad.append(f"probabilities differ from apply_t(raw) by {d:.3g}")
    if raw["type"] == "choice":
        k = len(cp)
        if cal["choice"] != raw["choice"]:
            bad.append(f"choice moved {raw['choice']} -> {cal['choice']}")
        if abs(cal["confidence"] - (max(cp) - 1 / k) / (1 - 1 / k)) > 1e-12:
            bad.append("choice confidence is not (p_max - 1/K)/(1 - 1/K) of the calibrated p")
    if raw["type"] == "score":
        if abs(cal["score"] - sum(i * x for i, x in enumerate(cp))) > 1e-9:
            bad.append("score is not the mean of the calibrated p")
        if abs(cal["confidence"] - max(cp)) > 1e-12:
            bad.append("score confidence is not the calibrated p_max")
    return bad, d


def run(d, picked):
    fails, worst = [], 0.0
    session_inconclusive = False
    name, r = picked[0]
    qs = {"q": r["question"]}
    if d.decide(r["state"], qs).get("answers") != d.decide(r["state"], qs).get("answers"):
        print(f"INCONCLUSIVE: a repeated raw decide ({name}) is not bit-identical")
        return None, worst, session_inconclusive
    for name, r in picked:
        qs = {"q": r["question"]}
        raw = d.decide(r["state"], qs)
        cal = d.decide(r["state"], qs, calibration=CAL)
        one = d.decide(r["state"], qs, calibration=ONE)
        if any("error" in v for v in (raw, cal, one)):
            fails.append(f"{name}: daemon error {[v.get('error') for v in (raw, cal, one)]}")
            continue
        bad, delta = check_pair(raw["answers"]["q"], cal["answers"]["q"], r["keys"], CAL[r["question"]["type"]])
        worst = max(worst, delta)
        fails += [f"{name}: {b}" for b in bad]
        if one["answers"] != raw["answers"]:
            fails.append(f"{name}: explicit T = 1 is not bit-identical to no calibration")
        if cal["timing"].get("calibration") != CAL:
            fails.append(f"{name}: timing.calibration = {cal['timing'].get('calibration')}")
        if "calibration" in raw["timing"] or "calibration" in one["timing"]:
            fails.append(f"{name}: an identity calibration was echoed in the timing")

    # Session mode: cold, extend, calibrated extend (the last two share the split point).
    name, r = next((n, x) for n, x in picked if x["question"]["type"] == "choice")
    text = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}]
    qs = {"q": r["question"]}
    replies = [d.session_decide(msgs, qs), d.session_decide(msgs, qs), d.session_decide(msgs, qs, calibration=CAL)]
    if any("error" in v for v in replies):
        fails.append(f"session: daemon error {[v.get('error') for v in replies]}")
    else:
        starts = [v["timing"].get("start") for v in replies]
        if starts[1:] != ["extend", "extend"]:
            print(f"INCONCLUSIVE (session part): starts {starts}, expected [cold, extend, extend]")
            session_inconclusive = True
        else:
            bad, delta = check_pair(replies[1]["answers"]["q"], replies[2]["answers"]["q"], r["keys"], CAL["choice"])
            worst = max(worst, delta)
            fails += [f"session: {b}" for b in bad]

    for bad_cal in ({"choice": 0}, {"bogus": 1.0}, "hot"):
        v = d.decide(r["state"], qs, calibration=bad_cal)
        err = v.get("error") or {}
        if err.get("status") != 422 or "calibration" not in err.get("message", ""):
            fails.append(f"malformed calibration {bad_cal!r}: {v}")
    if "error" in d.decide(r["state"], qs):
        fails.append("a normal decide after the 422s failed")
    return fails, worst, session_inconclusive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--max-seq", type=int, default=16384)
    a = ap.parse_args()
    manifest = json.loads((heldout.OUT / "manifest.json").read_text())
    picked = []
    for name in manifest:
        if licences.is_restricted(name):
            continue  # licence-clear held-out sources only (licences.py)
        rows = heldout.read_jsonl(heldout.WORK / "examples" / f"{name}.jsonl")
        picked += [(name, r) for r in rows[: a.per_source]]
    d = Daemon()
    try:
        d.load(a.model, max_seq=a.max_seq)
        fails, worst, session_inconclusive = run(d, picked)
    finally:
        d.close()
    if fails is None:
        return 2
    print(f"L1 live equivalence: {len(picked)} examples, worst |dp| {worst:.3g} (tolerance {TOL})")
    for f in fails:
        print("FAIL", f)
    if fails:
        print(f"FAIL ({len(fails)})")
        return 1
    if session_inconclusive:
        # Not PASS: the session part above already printed which precondition
        # failed. A distinct code (2, same as the full-run INCONCLUSIVE path)
        # keeps a CI step from reading this as green.
        print("INCONCLUSIVE (session part) — not PASS")
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
