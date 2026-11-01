#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""MTP acceptance + window-cost ratchet over the committed parity fixture.

Why this exists: speculative-decode wins are easy to lose silently. A head
retrain, a draft-path change, or a lane-policy change can drop per-position
acceptance on one workload class (agentic, math, non-English) while a pinned
pair of prompts still looks fine. This tool pins the prompt bytes, turns any
run into a per-position acceptance record, and compares it against a
committed baseline with thresholds that are pre-registered from measured
run-to-run spread (see benchmarks/mtp-parity/ratchet.json `rule`).

Definitions (identical to MtpTrain's `a_d` and to the Radiance /stats
`draft_positions`):
  offered_d  windows whose draft offered at least d rows
  reached_d  windows with offered >= d AND the first d-1 drafts accepted
  accepted_d windows with the first d drafts accepted
  a_d        accepted_d / reached_d   (conditional per-position acceptance)
  yield      committed tokens / lane-window (a lane-window is one user step)
  window_ms  wall time of one step (engine step; all lanes verify together)

Subcommands (all CPU-only; the card job that produces records is separate):
  fixture                         verify committed prompt bytes (md5 + size)
  extract-trace  LOG --cell ID    QWEN4_MTP_TRACE log -> cell record (singleton)
  from-radiance  FILE [--cell ID] Radiance /stats cell export -> cell record
  register       RECORDS...       derive baseline + tolerances from >=3 reps
  check          RECORDS...       compare against the ratchet; exit 1 on regress
  guard --base REF                fail if the ratchet was loosened vs REF
                                  without a `RATCHET-RAISE: <cell>` declaration
  --self-test                     unit tests (no files, no git)

Exit: 0 pass, 1 regression/violation, 2 usage/IO error, 3 unregistered cell,
4 partial class coverage, 5 underpowered cell.  3/4/5 fail closed unless the
matching --allow-* flag is given.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "benchmarks/mtp-parity/fixture.json"
RATCHET = ROOT / "benchmarks/mtp-parity/ratchet.json"
CELL_SCHEMA = "mtp-parity-cell/1"
RATCHET_SCHEMA = "mtp-parity-ratchet/1"

EXIT_OK, EXIT_REGRESS, EXIT_USAGE, EXIT_UNREG, EXIT_PARTIAL, EXIT_POWER = 0, 1, 2, 3, 4, 5


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def new_record(cell, depths, steps, committed, window_ms=None, window_src=None,
               engine="", build="", rep=0, extra=None):
    rec = {
        "schema": CELL_SCHEMA,
        "cell": cell,
        "engine": engine,
        "build": build,
        "rep": rep,
        "depths": depths,
        "steps": steps,
        "committed": committed,
        "window_ms": window_ms,
        "window_ms_source": window_src,
    }
    if extra:
        rec["extra"] = extra
    return rec


def depth_row(depth, offered, reached, accepted):
    return {"depth": depth, "offered": offered, "reached": reached, "accepted": accepted}


def validate_record(rec):
    errs = []
    if rec.get("schema") != CELL_SCHEMA:
        errs.append(f"schema != {CELL_SCHEMA}")
    if not rec.get("cell"):
        errs.append("missing cell")
    prev = None
    for row in rec.get("depths", []):
        d, o, r, a = (row.get(k) for k in ("depth", "offered", "reached", "accepted"))
        if not all(isinstance(x, int) and x >= 0 for x in (d, o, r, a)):
            errs.append(f"depth row not non-negative ints: {row}")
            continue
        if not (a <= r <= o):
            errs.append(f"depth {d}: need accepted <= reached <= offered, got {a}/{r}/{o}")
        if prev is not None and r > prev["accepted"]:
            errs.append(f"depth {d}: reached {r} > accepted at depth {d - 1} ({prev['accepted']})")
        prev = row
    if not isinstance(rec.get("steps"), int) or rec["steps"] <= 0:
        errs.append("steps must be a positive int")
    if not isinstance(rec.get("committed"), int) or rec["committed"] < 0:
        errs.append("committed must be a non-negative int")
    return errs


def rate(row):
    return row["accepted"] / row["reached"] if row["reached"] else None


def yield_of(rec):
    return rec["committed"] / rec["steps"]


def pool(recs):
    """Sum counts over reps of one cell. window_ms: mean of the reps that have it."""
    depths = {}
    for rec in recs:
        for row in rec["depths"]:
            acc = depths.setdefault(row["depth"], [0, 0, 0])
            acc[0] += row["offered"]
            acc[1] += row["reached"]
            acc[2] += row["accepted"]
    wms = [r["window_ms"] for r in recs if r.get("window_ms") is not None]
    return {
        "depths": [depth_row(d, *v) for d, v in sorted(depths.items())],
        "steps": sum(r["steps"] for r in recs),
        "committed": sum(r["committed"] for r in recs),
        "window_ms": sum(wms) / len(wms) if wms else None,
        "reps": len(recs),
    }


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------
TRACE_PREFIX = "QWEN4_MTP_TRACE "


def parse_trace(lines):
    """-> list of requests, each {'req': event, 'windows': [window events]}."""
    reqs, cur = [], None
    for line in lines:
        i = line.find(TRACE_PREFIX)
        if i < 0:
            continue
        try:
            ev = json.loads(line[i + len(TRACE_PREFIX):])
        except ValueError:
            continue
        if ev.get("event") == "request":
            cur = {"req": ev, "windows": []}
            reqs.append(cur)
        elif ev.get("event") == "window" and cur is not None:
            cur["windows"].append(ev)
    return reqs


def trace_to_counts(windows, max_depth=3, source="mtp"):
    """Per-position counts over native-head windows; steps/committed over all windows."""
    offered = [0] * max_depth
    reached = [0] * max_depth
    accepted = [0] * max_depth
    steps = committed = 0
    for w in windows:
        steps += 1
        committed += len(w.get("committed", []))
        if w.get("source") != source:
            continue
        for d, row in enumerate(w.get("rows", [])[:max_depth]):
            offered[d] += 1
        for d, row in enumerate(w.get("rows", [])[:max_depth]):
            reached[d] += 1
            if row.get("match"):
                accepted[d] += 1
            else:
                break
    return offered, reached, accepted, steps, committed


def extract_trace(log_path, cell, request_slice=None, max_depth=3, build="", rep=0):
    with open(log_path, errors="replace") as f:
        reqs = parse_trace(f)
    if request_slice is not None:
        reqs = reqs[request_slice[0]:request_slice[1]]
    windows = [w for r in reqs for w in r["windows"]]
    if not windows:
        raise ValueError("no QWEN4_MTP_TRACE window events selected (run with HIPFIRE_MTP_TRACE=1)")
    off, rea, acc, steps, committed = trace_to_counts(windows, max_depth)
    depths = [depth_row(d + 1, off[d], rea[d], acc[d]) for d in range(max_depth)]
    sources = {}
    for w in windows:
        sources[w.get("source", "?")] = sources.get(w.get("source", "?"), 0) + 1
    return new_record(cell, depths, steps, committed, build=build, rep=rep,
                      engine="hipfire singleton (QWEN4_MTP_TRACE)",
                      extra={"windows_by_source": sources, "requests": len(reqs),
                             "note": "reached counts only offered drafts: adaptive K / margin prune make deeper depths a selected subset"})


def from_radiance(path, cell=None):
    d = json.load(open(path))
    pos = d["draft_positions_delta"]
    depths = [depth_row(p["depth"], p["reached"], p["reached"], p["accepted"]) for p in pos]
    per_user = d.get("tokens_per_step_per_user")
    committed = int(d["tokens_committed"])
    steps = round(committed / per_user) if per_user else int(d["steps_decode"])
    name = cell or f"{d['workload']}-C{d['users']}-{'t1' if d.get('temperature', 0) > 0 else 'greedy'}"
    return new_record(name, depths, steps, committed, window_ms=d.get("step_ms_decode"),
                      window_src="radiance step_ms_decode", engine=d.get("engine", "radiance"),
                      rep=d.get("rep", 0),
                      extra={"max_tokens": d.get("max_tokens"), "engine_steps_decode": d.get("steps_decode"),
                             "offered_note": "Radiance exports reached/accepted only; offered := reached"})


# --------------------------------------------------------------------------
# thresholds (pre-registered rule) and register
# --------------------------------------------------------------------------
def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cv(xs):
    m = sum(xs) / len(xs) if xs else 0.0
    return stdev(xs) / m if m else 0.0


def derive_baseline(recs, rule):
    """Baseline + tolerances for one cell from its repeat records."""
    if len(recs) < rule["min_reps"]:
        raise ValueError(f"need >= {rule['min_reps']} fresh-process reps, got {len(recs)}")
    p = pool(recs)
    depth_out = []
    for row in p["depths"]:
        per_rep = []
        for rec in recs:
            r = next(x for x in rec["depths"] if x["depth"] == row["depth"])
            if r["reached"]:
                per_rep.append(r["accepted"] / r["reached"])
        depth_out.append({
            "depth": row["depth"], "reached": row["reached"], "accepted": row["accepted"],
            "rate": rate(row), "rate_sd_rep": stdev(per_rep),
        })
    thin = [d["depth"] for d in depth_out if d["reached"] < rule["min_trials_per_depth"]]
    if thin:
        raise ValueError(f"depths {thin} have < {rule['min_trials_per_depth']} pooled reached trials: cell underpowered "
                         "(a prompt that ends early, e.g. a short tool call, cannot carry a baseline)")
    yields = [yield_of(r) for r in recs]
    wms = [r["window_ms"] for r in recs if r.get("window_ms") is not None]
    cv_y = cv(yields)
    if cv_y > rule["max_cv_register"]:
        raise ValueError(f"yield cv across reps {cv_y:.3f} > {rule['max_cv_register']}: cell too noisy, add samples")
    out = {
        "depths": depth_out,
        "yield": p["committed"] / p["steps"], "yield_cv_rep": cv_y,
        "window_ms": (sum(wms) / len(wms)) if wms else None,
        "window_ms_cv_rep": cv(wms) if wms else None,
        "window_ms_source": next((r.get("window_ms_source") for r in recs if r.get("window_ms") is not None), None),
        "reps": len(recs), "steps": p["steps"],
        "build": recs[0].get("build", ""), "engine": recs[0].get("engine", ""),
    }
    return out


def depth_tolerance(base_row, new_row, rule):
    """Allowed drop in a_d: z * max(binomial SE of the difference, between-rep sd), floored."""
    pb, nb = base_row["rate"], base_row["reached"]
    pn, nn = rate(new_row), new_row["reached"]
    var = pb * (1 - pb) / nb + pn * (1 - pn) / nn
    return max(rule["abs_floor"], rule["z"] * math.sqrt(var), rule["z"] * base_row.get("rate_sd_rep", 0.0))


def compare_cell(base, newp, rule):
    """-> (verdict, rows). verdict in PASS|FAIL|INSUFFICIENT."""
    rows, verdict = [], "PASS"
    for brow in base["depths"]:
        nrow = next((r for r in newp["depths"] if r["depth"] == brow["depth"]), None)
        if nrow is None or nrow["reached"] < rule["min_trials_per_depth"]:
            n = 0 if nrow is None else nrow["reached"]
            if brow["reached"] >= rule["min_trials_per_depth"]:
                rows.append((f"a_{brow['depth']}", brow["rate"], None, None, f"INSUFFICIENT (reached {n} < {rule['min_trials_per_depth']})"))
                verdict = "INSUFFICIENT" if verdict == "PASS" else verdict
            continue
        tol = depth_tolerance(brow, nrow, rule)
        pn = rate(nrow)
        bad = pn < brow["rate"] - tol
        rows.append((f"a_{brow['depth']}", brow["rate"], pn, tol, "FAIL" if bad else "ok"))
        if bad:
            verdict = "FAIL"
    yb = base["yield"]
    yn = newp["committed"] / newp["steps"]
    ytol = max(rule["yield_floor_rel"], rule["z"] * base.get("yield_cv_rep", 0.0)) * yb
    ybad = yn < yb - ytol
    rows.append(("yield", yb, yn, ytol, "FAIL" if ybad else "ok"))
    if ybad:
        verdict = "FAIL"
    if base.get("window_ms") is not None:
        if newp.get("window_ms") is None:
            rows.append(("window_ms", base["window_ms"], None, None, "INSUFFICIENT (no window_ms in record)"))
            if verdict == "PASS":
                verdict = "INSUFFICIENT"
        else:
            wtol = max(rule["window_floor_rel"], rule["z"] * (base.get("window_ms_cv_rep") or 0.0)) * base["window_ms"]
            wbad = newp["window_ms"] > base["window_ms"] + wtol
            rows.append(("window_ms", base["window_ms"], newp["window_ms"], wtol, "FAIL" if wbad else "ok"))
            if wbad:
                verdict = "FAIL"
    return verdict, rows


# --------------------------------------------------------------------------
# fixture / ratchet io
# --------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def check_fixture(fixture, root=ROOT):
    errs = []
    for name, p in fixture["prompts"].items():
        path = root / p["path"]
        try:
            data = path.read_bytes()
        except OSError as e:
            errs.append(f"{name}: cannot read {p['path']}: {e}")
            continue
        md5 = hashlib.md5(data).hexdigest()
        if md5 != p["md5"]:
            errs.append(f"{name}: {p['path']} md5 {md5} != pinned {p['md5']}")
        if len(data) != p["bytes"]:
            errs.append(f"{name}: {p['path']} is {len(data)} bytes, pinned {p['bytes']}")
    classes = {c["class"] for c in fixture["cells"]}
    for c in fixture["cells"]:
        if c["prompt"] not in fixture["prompts"]:
            errs.append(f"cell {c['cell']}: unknown prompt {c['prompt']}")
    for name, p in fixture["prompts"].items():
        if p["class"] not in classes:
            errs.append(f"prompt {name}: class {p['class']} has no cell")
    return errs


def load_records(paths):
    recs = []
    for p in paths:
        path = Path(p)
        files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for f in files:
            d = load_json(f)
            items = d if isinstance(d, list) else [d]
            for it in items:
                if it.get("schema") != CELL_SCHEMA:
                    continue
                errs = validate_record(it)
                if errs:
                    raise ValueError(f"{f}: invalid record: {'; '.join(errs)}")
                recs.append(it)
    return recs


def group(recs):
    g = {}
    for r in recs:
        g.setdefault(r["cell"], []).append(r)
    return g


def class_table(fixture, cellmap, results):
    """Per-class pooled a_1..a_3 and yield across the cells seen."""
    cls_of = {c["cell"]: c["class"] for c in fixture["cells"]}
    by = {}
    for cell, recs in cellmap.items():
        by.setdefault(cls_of.get(cell, "?"), []).extend(recs)
    lines = []
    for cname in sorted(by):
        p = pool(by[cname])
        rs = " ".join(f"a{r['depth']}={rate(r):.3f}" if r["reached"] else f"a{r['depth']}=-" for r in p["depths"])
        lines.append(f"  class {cname:<12} cells={len({r['cell'] for r in by[cname]}):<2} {rs}  yield={p['committed'] / p['steps']:.3f}")
    return lines


def cmd_check(args):
    fixture, ratchet = load_json(FIXTURE), load_json(RATCHET)
    rule = ratchet["rule"]
    recs = load_records(args.records)
    if not recs:
        print("check: no cell records found", file=sys.stderr)
        return EXIT_USAGE
    cellmap = group(recs)
    gated = [c["cell"] for c in fixture["cells"] if ratchet["cells"].get(c["cell"], {}).get("gate", True)]
    missing = [c for c in gated if c not in cellmap]
    worst = EXIT_OK
    for cell, rs in sorted(cellmap.items()):
        entry = ratchet["cells"].get(cell)
        if entry is None:
            print(f"[{cell}] not in fixture")
            worst = max(worst, EXIT_USAGE)
            continue
        if entry.get("baseline") is None:
            print(f"[{cell}] UNREGISTERED (no baseline; run a registration campaign)")
            if not args.allow_unregistered:
                worst = max(worst, EXIT_UNREG) if worst != EXIT_REGRESS else worst
            continue
        verdict, rows = compare_cell(entry["baseline"], pool(rs), rule)
        print(f"[{cell}] {verdict} (reps {len(rs)})")
        for name, b, n, tol, v in rows:
            ns = "-" if n is None else f"{n:.4f}"
            ts = "-" if tol is None else f"{tol:.4f}"
            print(f"    {name:<10} base={b:.4f} new={ns} tol={ts}  {v}")
        if verdict == "FAIL":
            worst = EXIT_REGRESS
        elif verdict == "INSUFFICIENT" and not args.allow_underpowered and worst != EXIT_REGRESS:
            worst = max(worst, EXIT_POWER)
    print("per-class (pooled over the supplied cells):")
    for line in class_table(fixture, cellmap, None):
        print(line)
    if missing:
        print(f"missing cells ({len(missing)}): {', '.join(missing)}")
        if not args.allow_partial and worst != EXIT_REGRESS:
            print("PARTIAL: a pass requires every gated cell of every class (durability = breadth)")
            worst = max(worst, EXIT_PARTIAL)
    print({EXIT_OK: "RATCHET PASS", EXIT_REGRESS: "RATCHET FAIL (regression)", EXIT_UNREG: "RATCHET: unregistered cells (fail closed)",
           EXIT_PARTIAL: "RATCHET: partial coverage (fail closed)", EXIT_POWER: "RATCHET: underpowered cells (fail closed)",
           EXIT_USAGE: "RATCHET: usage error"}[worst])
    return worst


def lowers(old, new):
    """True if `new` baseline is weaker than `old` (rate/yield down, window up, tolerance wider)."""
    if old is None:
        return False
    if new is None:
        return True
    for od in old["depths"]:
        nd = next((d for d in new["depths"] if d["depth"] == od["depth"]), None)
        if nd is None or nd["rate"] < od["rate"]:
            return True
    if new["yield"] < old["yield"]:
        return True
    if old.get("window_ms") is not None and (new.get("window_ms") is None or new["window_ms"] > old["window_ms"]):
        return True
    return False


def rule_widened(old, new):
    out = []
    for k in ("z", "abs_floor", "yield_floor_rel", "window_floor_rel"):
        if new.get(k, 0) > old.get(k, 0):
            out.append(f"rule.{k} {old.get(k)} -> {new.get(k)}")
    for k in ("min_reps", "min_trials_per_depth"):
        if new.get(k, 0) < old.get(k, 0):
            out.append(f"rule.{k} {old.get(k)} -> {new.get(k)}")
    return out


def guard_violations(old_ratchet, new_ratchet, declared):
    v = []
    for line in rule_widened(old_ratchet["rule"], new_ratchet["rule"]):
        if "rule" not in declared:
            v.append(f"{line} (declare with `RATCHET-RAISE: rule ...`)")
    for cell, oc in old_ratchet["cells"].items():
        nc = new_ratchet["cells"].get(cell)
        if nc is None:
            if cell not in declared:
                v.append(f"cell {cell} removed (declare with `RATCHET-RAISE: {cell} ...`)")
            continue
        if oc.get("gate", True) and not nc.get("gate", True) and cell not in declared:
            v.append(f"cell {cell} un-gated (declare with `RATCHET-RAISE: {cell} ...`)")
        if lowers(oc.get("baseline"), nc.get("baseline")) and cell not in declared:
            v.append(f"cell {cell} baseline weakened (declare with `RATCHET-RAISE: {cell} ...`)")
    return v


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)


def cmd_guard(args):
    base = git("show", f"{args.base}:benchmarks/mtp-parity/ratchet.json")
    if base.returncode != 0:
        print(f"guard: {args.base} has no ratchet.json yet; nothing to compare")
        return EXIT_OK
    old, new = json.loads(base.stdout), load_json(RATCHET)
    log = git("log", f"{args.base}..HEAD", "--format=%B")
    declared = set(re.findall(r"^RATCHET-RAISE:\s*(\S+)", log.stdout, re.M))
    viol = guard_violations(old, new, declared)
    for line in viol:
        print(f"guard: {line}")
    print("guard: " + ("FAIL" if viol else "ok"))
    return EXIT_REGRESS if viol else EXIT_OK


def cmd_register(args):
    fixture, ratchet = load_json(FIXTURE), load_json(RATCHET)
    cellmap = group(load_records(args.records))
    known = {c["cell"] for c in fixture["cells"]}
    failed = weaker = 0
    for cell, rs in sorted(cellmap.items()):
        if cell not in known:
            print(f"register: {cell} is not a fixture cell", file=sys.stderr)
            failed += 1
            continue
        try:
            base = derive_baseline(rs, ratchet["rule"])
        except ValueError as e:
            print(f"register: {cell}: {e}", file=sys.stderr)
            failed += 1
            continue
        old = ratchet["cells"].setdefault(cell, {"gate": True, "baseline": None}).get("baseline")
        if lowers(old, base) and not args.declare_raise:
            print(f"register: {cell}: new baseline is weaker than the committed one; pass --declare-raise "
                  f"and put `RATCHET-RAISE: {cell} <reason>` in the commit message", file=sys.stderr)
            weaker += 1
            continue
        ratchet["cells"][cell]["baseline"] = base
        print(f"register: {cell}: {len(rs)} reps, yield {base['yield']:.3f}, "
              + ", ".join(f"a{d['depth']}={d['rate']:.3f}" for d in base["depths"] if d["rate"] is not None))
    if failed or weaker:
        print(f"register: {failed} cell(s) refused, {weaker} weaker than committed; nothing written", file=sys.stderr)
        return EXIT_REGRESS if weaker and not failed else EXIT_USAGE
    if not args.dry_run:
        RATCHET.write_text(json.dumps(ratchet, indent=1) + "\n")
    return EXIT_OK


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def self_test():
    import copy
    import tempfile
    rule = {"z": 3.0, "abs_floor": 0.02, "yield_floor_rel": 0.03, "window_floor_rel": 0.05,
            "min_reps": 3, "min_trials_per_depth": 50, "max_cv_register": 0.15}

    def rec(rep, acc=(0.8, 0.7, 0.6), reached=(400, 320, 224), steps=400, yld=3.0, wms=40.0):
        depths = [depth_row(d + 1, reached[d], reached[d], round(reached[d] * acc[d])) for d in range(3)]
        depths[1]["reached"] = depths[0]["accepted"]
        depths[2]["reached"] = depths[1]["accepted"]
        for i in (1, 2):
            depths[i]["offered"] = max(depths[i]["offered"], depths[i]["reached"])
            depths[i]["accepted"] = min(depths[i]["accepted"], depths[i]["reached"])
        return new_record("story-C1-t1", depths, steps, int(steps * yld), window_ms=wms, window_src="test", rep=rep)

    reps = [rec(i, wms=40.0 + 0.2 * i, yld=3.0 + 0.02 * i) for i in range(3)]
    for r in reps:
        assert not validate_record(r), validate_record(r)
    base = derive_baseline(reps, rule)
    # identical rerun passes
    v, _ = compare_cell(base, pool(reps), rule)
    assert v == "PASS", v
    # acceptance collapse at depth 2 fails
    bad = [rec(i, acc=(0.8, 0.45, 0.6), wms=40.0) for i in range(3)]
    v, rows = compare_cell(base, pool(bad), rule)
    assert v == "FAIL" and any(n == "a_2" and s == "FAIL" for n, _, _, _, s in rows), rows
    # small acceptance wobble inside tolerance passes
    ok = [rec(i, acc=(0.79, 0.69, 0.6), wms=40.0) for i in range(3)]
    v, _ = compare_cell(base, pool(ok), rule)
    assert v == "PASS", v
    # yield drop fails
    low = [rec(i, yld=2.5, wms=40.0) for i in range(3)]
    v, rows = compare_cell(base, pool(low), rule)
    assert v == "FAIL" and any(n == "yield" and s == "FAIL" for n, _, _, _, s in rows)
    # window cost rise fails; within floor passes
    slow = [rec(i, wms=45.0) for i in range(3)]
    v, rows = compare_cell(base, pool(slow), rule)
    assert v == "FAIL" and any(n == "window_ms" and s == "FAIL" for n, _, _, _, s in rows)
    fine = [rec(i, wms=41.5) for i in range(3)]
    v, _ = compare_cell(base, pool(fine), rule)
    assert v == "PASS", v
    # missing window_ms when baseline has one is INSUFFICIENT, not a pass
    nowin = [rec(i, wms=None) for i in range(3)]
    v, _ = compare_cell(base, pool(nowin), rule)
    assert v == "INSUFFICIENT", v
    # underpowered depth is INSUFFICIENT
    tiny = [rec(i, reached=(10, 8, 6), steps=10, wms=40.0) for i in range(3)]
    v, _ = compare_cell(base, pool(tiny), rule)
    assert v == "INSUFFICIENT", v
    # registration needs min reps and bounded noise
    try:
        derive_baseline(reps[:2], rule)
        raise AssertionError("2 reps accepted")
    except ValueError:
        pass
    try:
        derive_baseline([rec(i, reached=(10, 8, 6), steps=10, wms=40.0) for i in range(3)], rule)
        raise AssertionError("underpowered cell accepted")
    except ValueError:
        pass
    noisy = [rec(i, yld=y) for i, y in enumerate((2.0, 3.0, 4.0))]
    try:
        derive_baseline(noisy, rule)
        raise AssertionError("noisy cell accepted")
    except ValueError:
        pass
    # record validation
    broken = copy.deepcopy(reps[0])
    broken["depths"][1]["reached"] = broken["depths"][0]["accepted"] + 5
    assert validate_record(broken)
    # trace extraction: a window with rows [match, miss, match] reaches depth 1,2 only
    w = lambda rows, committed: {"event": "window", "source": "mtp", "rows": [{"match": m} for m in rows], "committed": committed}
    off, rea, acc, steps, committed = trace_to_counts(
        [w([True, False, True], [1, 2, 3]), w([True, True, True], [1, 2, 3, 4]), w([False], [9]),
         {"event": "window", "source": "ngram", "rows": [{"match": True}], "committed": [1, 2]}], 3)
    assert (off, rea, acc) == ([3, 2, 2], [3, 2, 1], [2, 1, 1]), (off, rea, acc)
    assert (steps, committed) == (4, 10), (steps, committed)
    # guard: weakening is caught unless declared
    old = {"rule": dict(rule), "cells": {"c": {"gate": True, "baseline": base}}}
    weaker = copy.deepcopy(old)
    weaker["cells"]["c"]["baseline"]["depths"][0]["rate"] -= 0.05
    assert guard_violations(old, weaker, set())
    assert not guard_violations(old, weaker, {"c"})
    wider = copy.deepcopy(old)
    wider["rule"]["z"] = 4.0
    assert guard_violations(old, wider, set()) and not guard_violations(old, wider, {"rule"})
    raised = copy.deepcopy(old)
    raised["cells"]["c"]["baseline"]["yield"] += 0.5
    assert not guard_violations(old, raised, set())
    removed = {"rule": dict(rule), "cells": {}}
    assert guard_violations(old, removed, set())
    # fixture check on a temp tree
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "p.txt").write_bytes(b"hello")
        fx = {"prompts": {"a": {"path": "p.txt", "md5": hashlib.md5(b"hello").hexdigest(), "bytes": 5, "class": "k"}},
              "cells": [{"cell": "a-C1-t1", "prompt": "a", "class": "k"}]}
        assert not check_fixture(fx, t)
        (t / "p.txt").write_bytes(b"hellO")
        assert check_fixture(fx, t)
    print("check-mtp-parity-ratchet: self-test ok")
    return EXIT_OK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("fixture")
    p = sub.add_parser("extract-trace")
    p.add_argument("log")
    p.add_argument("--cell", required=True)
    p.add_argument("--requests", help="request-event slice START:END (default all)")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--build", default="")
    p.add_argument("--rep", type=int, default=0)
    p = sub.add_parser("from-radiance")
    p.add_argument("file")
    p.add_argument("--cell")
    p = sub.add_parser("register")
    p.add_argument("records", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--declare-raise", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("records", nargs="+")
    p.add_argument("--allow-unregistered", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--allow-underpowered", action="store_true")
    p = sub.add_parser("guard")
    p.add_argument("--base", default="origin/beta")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        if args.cmd == "fixture":
            errs = check_fixture(load_json(FIXTURE))
            for e in errs:
                print(f"fixture: {e}")
            print("fixture: " + ("FAIL" if errs else "ok (prompt bytes match the pinned md5 and size)"))
            return EXIT_REGRESS if errs else EXIT_OK
        if args.cmd == "extract-trace":
            sl = None
            if args.requests:
                a, b = args.requests.split(":")
                sl = (int(a) if a else None, int(b) if b else None)
            print(json.dumps(extract_trace(args.log, args.cell, sl, args.max_depth, args.build, args.rep), indent=1))
            return EXIT_OK
        if args.cmd == "from-radiance":
            print(json.dumps(from_radiance(args.file, args.cell), indent=1))
            return EXIT_OK
        if args.cmd == "register":
            return cmd_register(args)
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "guard":
            return cmd_guard(args)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    ap.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
