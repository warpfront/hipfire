#!/usr/bin/env python3
"""CPU-only S1 combined-lever quality scorer (stdlib only).

Inputs are native `qwen4_slowlink_quality --phase score` outputs (`kl.jsonl`, `kl-summary.json`) of one arm
(OFF / PLACE / IU4 / QSA / STACK / PLACE_IU4 historical definitions only; the qualified campaign runs
OFF + PLACE_IU4 fresh), each scored against the SAME immutable beta (old Q1a `baseline`) reference.
Only the two primary long-context pairs participate: `primary-ctx16k` and `primary-ctx4k` (512 teacher-forced
rows each, KL(beta||candidate) in f64 nats from the native scorer).

GATE (Main ruling, every qualified arm incl. OFF and PLACE_IU4; point means of the native KL):
    mean KL(beta||arm) at 16K <= 0.0395 nats   and   mean KL(beta||arm) at 4K <= 0.0273 nats
(the beta's own pair CI95 upper bounds), plus same-route determinism and coherent decoded text (checked
elsewhere). All flags are state-q8. The pooled mean is reported, not gated.

Intervals use the S1Integrate/Q1 method transplanted UNCHANGED from q1_judge.py: deterministic stratified
moving-block bootstrap (block 32, 10000 replicates, splitmix64 seeded from the fixed label
`hipfire.q1.moving-block-bootstrap.v1`, linear-interpolated percentiles). They are DESCRIPTIVE ONLY: they
never gate (the gate uses the point means), they are not iid CIs and say nothing about generalisation.

Optional DIAGNOSTIC: per-row paired delta KL(beta||on)-KL(beta||off) with the same bootstrap machinery (the
same block draws serve the KL delta and the top-1 delta). It carries no pass/fail verdict.

`check_reference_identity` binds the native `kl-summary.json` reference stream hashes to the immutable beta
phase summary (kl.jsonl alone cannot authenticate the shared reference).

CLI: quality_scorer.py --absolute ARM/kl.jsonl [--off OFF/kl.jsonl]   Exit 0 gate pass, 1 gate fail, 2 invalid.
"""
import argparse
import hashlib
import json
import math
import sys

PRIMARY_PAIRS = ("primary-ctx16k", "primary-ctx4k")
CONTEXT_OF = {"primary-ctx16k": "ctx16k", "primary-ctx4k": "ctx4k"}
CELL_OF = {"primary-ctx16k": "primary-ctx16k-teacher", "primary-ctx4k": "primary-ctx4k-teacher"}
EXPECTED_ROWS = 512
# beta own-pair CI95 upper bounds: exact values sealed in holds/Q1-PAIR-EXECUTION-RESULT.json
# (sha256 762704be28edc99fbd29ce05dff5fddc7f477a4fdc72bb1247b88fc655cc25a1,
# parts.scale.summary.scale.contexts.<ctx>.kl_mean_ci95_upper) and the declared rounded values.
GATE_BOUNDS_EXACT = {"ctx16k": 0.03948458079446915, "ctx4k": 0.027284316982763658}
GATE_BOUNDS_DECLARED = {"ctx16k": 0.0395, "ctx4k": 0.0273}
BOUNDS_FILE_SHA256 = "762704be28edc99fbd29ce05dff5fddc7f477a4fdc72bb1247b88fc655cc25a1"
OLD_Q1_POINT_GATE = 0.010                          # historical, informational only
BOOT_BLOCK = 32
BOOT_REPS = 10000
BOOT_LABEL = "hipfire.q1.moving-block-bootstrap.v1"
BOOT_CONFIDENCE = 0.95
KL_NEG_TOLERANCE = 1e-9
SUM_REL_TOLERANCE = 1e-9
SUMMARY_SCHEMA = "hipfire.qwen4.slowlink_quality.kl_summary.v1"
BOOT_NOTE = ("DESCRIPTIVE ONLY: stratified moving-block bootstrap (block 32, 10000 reps, splitmix64 from the fixed "
             "label), the S1Integrate/Q1 method; not an iid CI, never gates.")


class Invalid(Exception):
    pass


# ----------------------------------------- bootstrap (transplanted unchanged from q1_judge.py)
MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, m):
        limit = (1 << 64) - ((1 << 64) % m)
        while True:
            x = self.next()
            if x < limit:
                return x % m


def seed_from_label(label):
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def percentile(sorted_vals, p):
    pos = (len(sorted_vals) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def stratified_block_bootstrap(strata, label=BOOT_LABEL, block=BOOT_BLOCK, reps=BOOT_REPS, confidence=BOOT_CONFIDENCE):
    """strata: {name: [(kl, top1_match 0/1), ...]} in row order. Every replicate resamples each stratum independently with overlapping moving
    blocks of `block` consecutive positions (start uniform in [0, n-block]; the last block of a stratum is truncated to n); the SAME block draws
    serve the KL mean and the top-1 rate. Returns {name|'pooled': {'kl': (lo, hi), 'top1': (lo, hi)}} (pooled = equal weight over all rows)."""
    names = list(strata)
    prefix = {}
    for nm in names:
        pk, pt = [0.0], [0.0]
        for kl, t in strata[nm]:
            pk.append(pk[-1] + kl)
            pt.append(pt[-1] + t)
        prefix[nm] = (pk, pt, len(strata[nm]))
    rng = SplitMix64(seed_from_label(label))
    sums = {nm: ([], []) for nm in names}
    pooled_k, pooled_t = [], []
    total_n = sum(prefix[nm][2] for nm in names)
    for _ in range(reps):
        tk = tt = 0.0
        for nm in names:
            pk, pt, n = prefix[nm]
            if n < block:
                raise ValueError(f"stratum {nm} has {n} < block {block} positions")
            sk = st = 0.0
            for b in range(-(-n // block)):
                s = rng.below(n - block + 1)
                take = min(block, n - b * block)
                sk += pk[s + take] - pk[s]
                st += pt[s + take] - pt[s]
            sums[nm][0].append(sk / n)
            sums[nm][1].append(st / n)
            tk += sk
            tt += st
        pooled_k.append(tk / total_n)
        pooled_t.append(tt / total_n)
    alpha = (1.0 - confidence) / 2.0
    out = {}
    for nm in names:
        a, b = sorted(sums[nm][0]), sorted(sums[nm][1])
        out[nm] = {"kl": (percentile(a, alpha), percentile(a, 1 - alpha)), "top1": (percentile(b, alpha), percentile(b, 1 - alpha))}
    a, b = sorted(pooled_k), sorted(pooled_t)
    out["pooled"] = {"kl": (percentile(a, alpha), percentile(a, 1 - alpha)), "top1": (percentile(b, alpha), percentile(b, 1 - alpha))}
    return out


def _boot(strata, reps=BOOT_REPS):
    try:
        return stratified_block_bootstrap(strata, reps=reps)
    except ValueError as e:
        raise Invalid(str(e))


def _ci(pair):
    return {"lo": pair[0] + 0.0, "hi": pair[1] + 0.0, "conf": BOOT_CONFIDENCE, "method": "stratified-moving-block-bootstrap",
            "block": BOOT_BLOCK, "reps": BOOT_REPS, "seed_label": BOOT_LABEL, "descriptive_only": True}


# ------------------------------------------------------------------- loading

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def load_jsonl(path):
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as e:
                    raise Invalid("%s:%d: bad JSON: %s" % (path, ln, e))
                if not isinstance(rec, dict):
                    raise Invalid("%s:%d: record is not an object" % (path, ln))
                out.append(rec)
    except OSError as e:
        raise Invalid("cannot read %s: %s" % (path, e))
    return out


def select_primary(records, label):
    """{native pair: {row: rec}} of the two primary pairs, fully validated; (groups, ignored names, schemas)."""
    groups, ignored, schemas = {}, set(), set()
    for i, rec in enumerate(records):
        where = "%s record #%d" % (label, i)
        name = rec.get("pair")
        if not isinstance(name, str) or not name:
            raise Invalid("%s: record lacks pair name" % where)
        if name not in PRIMARY_PAIRS:
            ignored.add(name)
            continue
        row = rec.get("row")
        if not _is_int(row) or row < 0:
            raise Invalid("%s (%s): bad row index %r" % (where, name, row))
        for k in ("context_position", "next_token", "reference_top1_id", "candidate_top1_id"):
            if k not in rec:
                raise Invalid("%s (%s row %d): %s missing" % (where, name, row, k))
            v = rec[k]
            if not _is_int(v) or v < 0:
                raise Invalid("%s (%s row %d): %s must be a nonnegative integer" % (where, name, row, k))
        if rec.get("kl_status") != "ok":
            raise Invalid("%s (%s row %d): kl_status=%r (must be 'ok')" % (where, name, row, rec.get("kl_status")))
        kl = rec.get("kl_nats")
        if isinstance(kl, bool) or not isinstance(kl, (int, float)) or not math.isfinite(kl):
            raise Invalid("%s (%s row %d): kl_nats missing/nonfinite: %r" % (where, name, row, kl))
        if kl < -KL_NEG_TOLERANCE:
            raise Invalid("%s (%s row %d): materially negative kl_nats %r" % (where, name, row, kl))
        tm = rec.get("top1_match")
        if not isinstance(tm, bool):
            raise Invalid("%s (%s row %d): top1_match not a bool" % (where, name, row))
        if tm != (rec["reference_top1_id"] == rec["candidate_top1_id"]):
            raise Invalid("%s (%s row %d): top1_match inconsistent with top1 ids" % (where, name, row))
        sch = rec.get("schema")
        if not isinstance(sch, str) or not sch:
            raise Invalid("%s (%s row %d): schema missing" % (where, name, row))
        schemas.add(sch)
        g = groups.setdefault(name, {})
        if row in g:
            raise Invalid("%s: duplicate row %d in %s" % (label, row, name))
        g[row] = rec
    for pair in PRIMARY_PAIRS:
        g = groups.get(pair)
        if g is None:
            raise Invalid("%s: required pair %s is absent" % (label, pair))
        if sorted(g) != list(range(EXPECTED_ROWS)):
            raise Invalid("%s: %s has %d rows (indices must be exactly 0..%d)" % (label, pair, len(g), EXPECTED_ROWS - 1))
    if len(schemas) > 1:
        raise Invalid("%s: mixed schemas %s" % (label, sorted(schemas)))
    return groups, ignored, schemas


def _rate(matches):
    return math.fsum(1.0 if m else 0.0 for m in matches) / len(matches)


# ------------------------------------------------------------------ analysis

def analyze_absolute(recs, reps=BOOT_REPS):
    """KL(beta||arm): raw point means and the descriptive bootstrap intervals, per context and pooled."""
    g, ign, _sch = select_primary(recs, "arm")
    strata, contexts, all_kl, all_tm = {}, {}, [], []
    for pair in PRIMARY_PAIRS:
        ctx = CONTEXT_OF[pair]
        kl = [float(g[pair][r]["kl_nats"]) for r in range(EXPECTED_ROWS)]
        tm = [g[pair][r]["top1_match"] for r in range(EXPECTED_ROWS)]
        strata[ctx] = [(k, 1.0 if t else 0.0) for k, t in zip(kl, tm)]
        contexts[ctx] = {"native_pair": pair, "cell": CELL_OF[pair], "rows": EXPECTED_ROWS,
                         "raw_mean_kl_nats": math.fsum(kl) / EXPECTED_ROWS + 0.0, "max_kl_nats": max(kl),
                         "top1_agree": _rate(tm)}
        all_kl += kl
        all_tm += tm
    ci = _boot(strata, reps)
    for ctx in contexts:
        contexts[ctx]["ci_kl"] = _ci(ci[ctx]["kl"])
        contexts[ctx]["ci_top1"] = _ci(ci[ctx]["top1"])
    pooled = {"rows": len(all_kl), "raw_mean_kl_nats": math.fsum(all_kl) / len(all_kl) + 0.0,
              "top1_agree": _rate(all_tm), "ci_kl": _ci(ci["pooled"]["kl"]), "ci_top1": _ci(ci["pooled"]["top1"]),
              "gated": False}
    return {"status": "ok", "ci_note": BOOT_NOTE, "contexts": contexts, "pooled": pooled,
            "ignored_pairs": sorted(ign)}


def evaluate_gate(absolute):
    """Main ruling: point mean KL(beta||arm) <= the beta's own pair CI95 upper bound, inclusive, per context.
    `pass` uses the exact sealed bounds (Q1-PAIR-EXECUTION-RESULT.json); the declared rounded 0.0395/0.0273
    outcome is reported alongside as `pass_declared_rounded` (rounded is the looser of the two)."""
    per = {}
    for ctx, bound in GATE_BOUNDS_EXACT.items():
        mean = absolute["contexts"][ctx]["raw_mean_kl_nats"]
        rounded = GATE_BOUNDS_DECLARED[ctx]
        per[ctx] = {"mean_kl_nats": mean, "bound_nats": bound, "pass": mean <= bound, "margin_nats": bound - mean,
                    "declared_rounded_bound_nats": rounded, "pass_declared_rounded": mean <= rounded,
                    "ci_kl_descriptive": absolute["contexts"][ctx]["ci_kl"]}
    old = {ctx: {"mean_kl_nats": absolute["contexts"][ctx]["raw_mean_kl_nats"], "bound_nats": OLD_Q1_POINT_GATE,
                 "pass": absolute["contexts"][ctx]["raw_mean_kl_nats"] <= OLD_Q1_POINT_GATE} for ctx in GATE_BOUNDS_EXACT}
    pooled_mean = absolute["pooled"]["raw_mean_kl_nats"]
    old["pooled"] = {"mean_kl_nats": pooled_mean, "bound_nats": OLD_Q1_POINT_GATE, "pass": pooled_mean <= OLD_Q1_POINT_GATE}
    return {"rule": "mean KL(beta||arm) <= beta own-pair CI95 upper (exact): 16K 0.03948458079446915, 4K "
                    "0.027284316982763658; declared rounded 0.0395/0.0273 also reported (point means; CIs descriptive)",
            "direction_note": ("the bounds are the scale-pair CI95 uppers of KL(beta2048||beta4096); the candidate "
                               "statistic is KL(beta4096||candidate) with the Q1a baseline as beta: a different "
                               "direction, applied as the explicit Main calibration"),
            "per_context": per, "pass": all(v["pass"] for v in per.values()),
            "pass_declared_rounded": all(v["pass_declared_rounded"] for v in per.values()),
            "raw_old_q1_010_diagnostic": {"note": ("historical Q1 raw 0.010 point-mean verdict, preserved SEPARATELY; "
                                                    "not the acceptance gate"), "per_context_and_pooled": old,
                                          "pass": all(v["pass"] for v in old.values())},
            "pooled_mean_kl_nats": pooled_mean, "pooled_gated": False}


def analyze_delta_diagnostic(off_recs, on_recs, reps=BOOT_REPS):
    """Optional diagnostic: paired per-row delta KL(beta||on)-KL(beta||off), same bootstrap machinery. NO verdict."""
    off_g, _i0, off_sch = select_primary(off_recs, "off")
    on_g, _i1, on_sch = select_primary(on_recs, "on")
    if off_sch != on_sch:
        raise Invalid("off/on schemas differ: %s vs %s" % (sorted(off_sch), sorted(on_sch)))
    strata, contexts, all_d = {}, {}, []
    for pair in PRIMARY_PAIRS:
        ctx = CONTEXT_OF[pair]
        rows = []
        for row in range(EXPECTED_ROWS):
            o, n = off_g[pair][row], on_g[pair][row]
            for k in ("context_position", "next_token", "reference_top1_id"):
                if o[k] != n[k]:
                    raise Invalid("%s row %d: %s mismatch off=%r on=%r" % (pair, row, k, o[k], n[k]))
            rows.append((float(n["kl_nats"]) - float(o["kl_nats"]),
                         (1.0 if n["top1_match"] else 0.0) - (1.0 if o["top1_match"] else 0.0)))
        strata[ctx] = rows
        all_d += [d for d, _t in rows]
        contexts[ctx] = {"native_pair": pair, "rows": EXPECTED_ROWS,
                         "delta_mean_kl_nats": math.fsum(d for d, _t in rows) / EXPECTED_ROWS + 0.0,
                         "delta_top1_rate": math.fsum(t for _d, t in rows) / EXPECTED_ROWS + 0.0}
    ci = _boot(strata, reps)
    for ctx in contexts:
        contexts[ctx]["ci_delta_kl"] = _ci(ci[ctx]["kl"])
        contexts[ctx]["ci_delta_top1"] = _ci(ci[ctx]["top1"])
    return {"status": "ok", "diagnostic_only": True, "verdict": None, "ci_note": BOOT_NOTE, "contexts": contexts,
            "pooled": {"rows": len(all_d), "delta_mean_kl_nats": math.fsum(all_d) / len(all_d) + 0.0,
                       "ci_delta_kl": _ci(ci["pooled"]["kl"]), "ci_delta_top1": _ci(ci["pooled"]["top1"])}}


# ------------------------------------------------------------ identity checks

def _summary_pairs(summary, label):
    if not isinstance(summary, dict) or not isinstance(summary.get("pairs"), list):
        raise Invalid("%s: kl-summary has no pairs list" % label)
    out = {}
    for p in summary["pairs"]:
        if not isinstance(p, dict) or not isinstance(p.get("pair"), str):
            raise Invalid("%s: malformed kl-summary pair entry" % label)
        if p["pair"] in out:
            raise Invalid("%s: duplicate kl-summary pair %s" % (label, p["pair"]))
        out[p["pair"]] = p
    return out


def check_reference_identity(beta_cells, summaries, candidate_cells=None):
    """Bind each arm's native kl-summary to the immutable beta reference.

    beta_cells: {cell: stream_sha256} from the sealed beta phase-summary. summaries: {arm: kl-summary dict}.
    candidate_cells: {arm: {cell: stream_sha256}} from that arm's s1a phase summary (the scored candidate
    stream must be the arm's). Raises Invalid on any mismatch (a reference differing between arms is fatal)."""
    evidence = {}
    for pair in PRIMARY_PAIRS:
        want = beta_cells.get(CELL_OF[pair])
        if not isinstance(want, str) or len(want) != 64:
            raise Invalid("beta phase summary has no stream hash for %s" % CELL_OF[pair])
        evidence[pair] = {"beta_reference_stream_sha256": want, "arms": {}}
    for arm, summ in summaries.items():
        pairs = _summary_pairs(summ, arm)
        for pair in PRIMARY_PAIRS:
            p = pairs.get(pair)
            if p is None:
                raise Invalid("%s: kl-summary lacks pair %s" % (arm, pair))
            ref = p.get("reference") or {}
            if ref.get("cell") != CELL_OF[pair]:
                raise Invalid("%s %s: reference cell %r != %s" % (arm, pair, ref.get("cell"), CELL_OF[pair]))
            if ref.get("stream_sha256") != evidence[pair]["beta_reference_stream_sha256"]:
                raise Invalid("%s %s: reference stream hash %r differs from the sealed beta reference %s"
                              % (arm, pair, ref.get("stream_sha256"), evidence[pair]["beta_reference_stream_sha256"]))
            if (p.get("rows") != EXPECTED_ROWS or p.get("expected_rows") != EXPECTED_ROWS
                    or p.get("complete") is not True or p.get("infinite_rows") != 0):
                raise Invalid("%s %s: pair not complete/finite 512 rows (rows=%r expected=%r complete=%r infinite=%r)"
                              % (arm, pair, p.get("rows"), p.get("expected_rows"), p.get("complete"),
                                 p.get("infinite_rows")))
            cand = p.get("candidate") or {}
            if cand.get("cell") != CELL_OF[pair]:
                raise Invalid("%s %s: candidate cell %r" % (arm, pair, cand.get("cell")))
            if candidate_cells is not None:
                cc = candidate_cells.get(arm, {}).get(CELL_OF[pair])
                if cc != cand.get("stream_sha256"):
                    raise Invalid("%s %s: scored candidate stream %r != the arm s1a cell stream %r"
                                  % (arm, pair, cand.get("stream_sha256"), cc))
            evidence[pair]["arms"][arm] = {
                "reference_stream_sha256": ref["stream_sha256"], "candidate_stream_sha256": cand.get("stream_sha256"),
                "native_kl_mean_nats": p.get("kl_mean_nats"), "native_top1_agree_rows": p.get("top1_agree_rows")}
    return evidence


def check_summary_matches_rows(summary, recs, label):
    """The native summary's sums/top-1 counts must agree with the kl.jsonl rows actually scored."""
    pairs = _summary_pairs(summary, label)
    g, _ign, _sch = select_primary(recs, label)
    for pair in PRIMARY_PAIRS:
        rows = [g[pair][r] for r in range(EXPECTED_ROWS)]
        s = math.fsum(float(r["kl_nats"]) for r in rows)
        want = pairs[pair].get("kl_sum_nats")
        if not isinstance(want, (int, float)) or abs(s - float(want)) > 1e-6 * max(1.0, abs(s)):
            raise Invalid("%s %s: kl.jsonl sum %r != kl-summary kl_sum_nats %r" % (label, pair, s, want))
        agree = sum(1 for r in rows if r["top1_match"])
        if agree != pairs[pair].get("top1_agree_rows"):
            raise Invalid("%s %s: kl.jsonl top1 agree %d != kl-summary %r" % (label, pair, agree, pairs[pair].get("top1_agree_rows")))


# ----------------------------------------------------------------------- CLI

def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--absolute", required=True, help="kl.jsonl of one arm")
    ap.add_argument("--off", help="OFF kl.jsonl: also print the optional paired delta diagnostic")
    args = ap.parse_args(argv)
    try:
        arm = load_jsonl(args.absolute)
        absolute = analyze_absolute(arm)
        res = {"absolute": absolute, "gate": evaluate_gate(absolute)}
        if args.off:
            res["delta_diagnostic"] = analyze_delta_diagnostic(load_jsonl(args.off), arm)
    except Invalid as e:
        print(dumps({"status": "invalid", "error": str(e), "exit_code": 2}))
        return 2
    code = 0 if res["gate"]["pass"] else 1
    res["exit_code"] = code
    print(dumps(res))
    return code


if __name__ == "__main__":
    sys.exit(main())
