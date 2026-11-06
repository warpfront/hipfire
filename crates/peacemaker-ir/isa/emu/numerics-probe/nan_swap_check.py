#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Offline receipt checker for the nan_swap hardware probe (standard library only, no GPU, no network).

usage: nan_swap_check.py RECEIPT_DIR BUILD_DIR

RECEIPT_DIR holds what `nan_swap OUTDIR` wrote: nan_swap_summary.json / .tsv, nan_swap_modes.tsv,
nan_swap_samples.tsv, nan_swap_edges.tsv (+ identity.host.txt from run.sh when present).
BUILD_DIR holds the output of nan_swap_build.sh: build.json, nan_swap_gfx1201 and the disassembly/objects.

Checks (all must hold; the process exits 1 otherwise, 2 for missing/unparseable inputs):
  1. summary.json: complete, gfx1201, modes 0..15, all suites, exact case counts per (mode, suite, op):
     nan = 16,777,214 NaNs x 16 partners, tri = 64,000, rand = 4,194,304 (and >= 1,048,576); 6 outputs per case.
  2. every selection / dual-vs-native / x-vs-y / safe-criterion / unexpected-NaN counter is zero; MODE
     control and readback counters are zero and the observed controls equal the expected ones for all modes.
  3. unsafe inputs (both NaN, quiet(a) != quiet(b)): swap unequal in every case; safe inputs: swap equal in every case.
  4. invalid results are entirely 0xffc00000 (no 0x7fc00000, no other value) and invalid coverage is nonzero.
  5. summary.tsv agrees with summary.json counter by counter; modes.tsv agrees with the control data.
  6. raw edge rows (all 16 modes) and raw sample rows are parsed independently and all six outputs of every row
     are compared with a first-source NaN / invalid oracle (input denormal flush by mode, derived here, not copied
     from the probe), the known-input safe condition, and the recorded oracle kinds. Rows whose oracle expects no
     NaN are only checked to be non-NaN and mutually consistent; no finite numeric model is claimed.
  7. build.json source hash equals the sha256 of nan_swap.hip beside this script (when present); binary / object /
     disassembly hashes equal the files in BUILD_DIR (when present); opcode checks ok.
Writes RECEIPT_DIR/nan_swap_check.json and prints a summary.
"""
import hashlib
import json
import os
import sys

NAN_ITEMS = 16777214
PARTNERS = 16
TRI = 64000
RAND = 4194304
RAND_MIN = 1048576
MODES = list(range(16))
SUITES = ["nan", "tri", "rand"]
OPS = ["add", "mul", "fmac"]
COUNTERS = [
    "cases", "nan_expected", "invalid_expected", "unknown_unchecked", "sel_native_orig", "sel_native_swap", "sel_dual_x_orig",
    "sel_dual_y_orig", "sel_dual_x_swap", "sel_dual_y_swap", "sel_any", "dual_vs_native_x_orig", "dual_vs_native_y_orig",
    "dual_vs_native_x_swap", "dual_vs_native_y_swap", "dual_vs_native_any", "x_vs_y_orig", "x_vs_y_swap",
    "swap_unequal_native", "swap_unequal_dual_x", "swap_unequal_dual_y", "swap_unequal_any", "safe_cases", "safe_swap_mismatch",
    "unsafe_cases", "unsafe_swap_unequal", "both_nan_cases", "unexpected_nan", "invalid_result_7fc00000", "invalid_result_ffc00000",
    "invalid_result_other", "native_orig_result_nan"]
ZERO_COUNTERS = [
    "sel_native_orig", "sel_native_swap", "sel_dual_x_orig", "sel_dual_y_orig", "sel_dual_x_swap", "sel_dual_y_swap", "sel_any",
    "dual_vs_native_x_orig", "dual_vs_native_y_orig", "dual_vs_native_x_swap", "dual_vs_native_y_swap", "dual_vs_native_any",
    "x_vs_y_orig", "x_vs_y_swap", "safe_swap_mismatch", "unexpected_nan"]
CLASSES = ["sel_mismatch", "dual_mismatch", "safe_mismatch", "swap_unequal_unsafe", "sel_ok_invalid_or_c", "sel_ok_single_nan",
           "unexpected_nan", "safe_both_nan"]
FAIL_CLASSES = {"sel_mismatch", "dual_mismatch", "safe_mismatch", "unexpected_nan"}
K_NONE, K_A, K_B, K_INVALID, K_C, K_INFCONF, K_UNKNOWN = range(7)
KIND_NAMES = ["none", "a", "b", "invalid", "c", "infconf", "unknown"]
DEFAULT_NAN = 0xFFC00000
SLOTS = 32
WAVES_PER_LAUNCH = 2048 * 8  # grid 2048 blocks x 256 threads (wave32)

failures = []
notes = []


def check(cond, msg):
    if not cond:
        if len(failures) < 200:
            failures.append(msg)
        elif len(failures) == 200:
            failures.append("... further failures suppressed")
    return bool(cond)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def die(msg):
    print("nan_swap_check: " + msg, file=sys.stderr)
    sys.exit(2)


def read_tsv(path):
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except OSError as e:
        die("cannot read %s: %s" % (path, e))
    lines = [l for l in lines if l != ""]
    if not lines:
        die("empty tsv %s" % path)
    hdr = lines[0].split("\t")
    rows = []
    for n, l in enumerate(lines[1:], 2):
        v = l.split("\t")
        if len(v) != len(hdr):
            die("%s line %d: %d fields, expected %d" % (path, n, len(v), len(hdr)))
        rows.append(dict(zip(hdr, v)))
    return hdr, rows


def hx(s):
    return int(s, 16)


# ------------------------------------------------------------------ independent oracle
def isnan(x):
    return (x & 0x7FFFFFFF) > 0x7F800000


def isinf(x):
    return (x & 0x7FFFFFFF) == 0x7F800000


def flushed_zero(x, flush_in):
    return (x & 0x7FFFFFFF) == 0 or (flush_in and (x & 0x7F800000) == 0)


def oracle(op, a, b, c, flush_in):
    """-> (kind, value). First NaN of (a, b) wins (quieted, sign/payload kept); add: inf + -inf invalid;
    mul/fmac: inf * (flushed) zero invalid; fmac: then NaN c; then infinite product vs opposite infinite c."""
    if isnan(a):
        return K_A, a | 0x400000
    if isnan(b):
        return K_B, b | 0x400000
    if op == "add":
        if isinf(a) and isinf(b) and ((a ^ b) >> 31):
            return K_INVALID, DEFAULT_NAN
        return K_NONE, 0
    az, bz = flushed_zero(a, flush_in), flushed_zero(b, flush_in)
    ia, ib = isinf(a), isinf(b)
    if (ia and bz) or (ib and az):
        return K_INVALID, DEFAULT_NAN
    if op == "mul":
        return K_NONE, 0
    if isnan(c):
        return K_C, c | 0x400000
    if isinf(c):
        if ia or ib:
            if ((a ^ b) >> 31) != (c >> 31):
                return K_INFCONF, DEFAULT_NAN
            return K_NONE, 0
        # finite product against infinite c: overflow is rounding-dependent, not judged
        if not az and not bz and (((a >> 23) & 0xFF) + ((b >> 23) & 0xFF)) >= 380:
            return K_UNKNOWN, 0
    return K_NONE, 0


def safe_input(a, b):
    return (not (isnan(a) and isnan(b))) or ((a | 0x400000) == (b | 0x400000))


class RowStats:
    def __init__(self):
        self.rows = 0
        self.judged = {k: 0 for k in KIND_NAMES}
        self.dir = {"orig": {k: 0 for k in KIND_NAMES}, "swap": {k: 0 for k in KIND_NAMES}}
        self.safe_rows = 0
        self.unsafe_rows = 0
        self.invalid_rows = 0


def check_row(tag, mode, op, a, b, c, rec, st):
    """rec: dict of recorded hex ints. Compares all six outputs with the independent oracle."""
    flush_in = (mode & 1) == 0
    st.rows += 1
    ko, vo = oracle(op, a, b, c, flush_in)
    ks, vs = oracle(op, b, a, c, flush_in)
    ok = True
    ok &= check(rec["kind_orig"] == ko, "%s: recorded orig kind %s != oracle %s" % (tag, KIND_NAMES[rec["kind_orig"]] if rec["kind_orig"] < 7 else rec["kind_orig"], KIND_NAMES[ko]))
    ok &= check(rec["kind_swap"] == ks, "%s: recorded swap kind %s != oracle %s" % (tag, KIND_NAMES[rec["kind_swap"]] if rec["kind_swap"] < 7 else rec["kind_swap"], KIND_NAMES[ks]))
    if ko in (K_A, K_B, K_C):
        ok &= check(rec["exp_orig"] == vo, "%s: recorded expected-orig %08x != oracle %08x" % (tag, rec["exp_orig"], vo))
    if ks in (K_A, K_B, K_C):
        ok &= check(rec["exp_swap"] == vs, "%s: recorded expected-swap %08x != oracle %08x" % (tag, rec["exp_swap"], vs))
    for name, kind, val in (("native_orig", ko, vo), ("dualx_orig", ko, vo), ("dualy_orig", ko, vo),
                            ("native_swap", ks, vs), ("dualx_swap", ks, vs), ("dualy_swap", ks, vs)):
        r = rec[name]
        if kind in (K_A, K_B, K_C):
            ok &= check(r == val, "%s: %s=%08x expected first-source NaN %08x (%s)" % (tag, name, r, val, KIND_NAMES[kind]))
        elif kind in (K_INVALID, K_INFCONF):
            ok &= check(r == DEFAULT_NAN, "%s: %s=%08x expected default NaN ffc00000 (%s)" % (tag, name, r, KIND_NAMES[kind]))
        elif kind == K_NONE:
            ok &= check(not isnan(r), "%s: %s=%08x is NaN where the oracle expects none" % (tag, name, r))
    # dual halves equal native in the same order (no finite value model claimed)
    ok &= check(rec["dualx_orig"] == rec["native_orig"] and rec["dualy_orig"] == rec["native_orig"],
                "%s: dual orig X/Y %08x/%08x != native %08x" % (tag, rec["dualx_orig"], rec["dualy_orig"], rec["native_orig"]))
    ok &= check(rec["dualx_swap"] == rec["native_swap"] and rec["dualy_swap"] == rec["native_swap"],
                "%s: dual swap X/Y %08x/%08x != native %08x" % (tag, rec["dualx_swap"], rec["dualy_swap"], rec["native_swap"]))
    sf = safe_input(a, b)
    ok &= check(rec["safe"] == (1 if sf else 0), "%s: recorded safe=%d but known-input condition is %d" % (tag, rec["safe"], 1 if sf else 0))
    unequal = (rec["native_orig"] != rec["native_swap"]) or (rec["dualx_orig"] != rec["dualx_swap"]) or (rec["dualy_orig"] != rec["dualy_swap"])
    if sf:
        st.safe_rows += 1
        ok &= check(not unequal, "%s: safe input but swapped outputs differ" % tag)
    else:
        st.unsafe_rows += 1
        ok &= check(unequal, "%s: unsafe input but swapped outputs all equal" % tag)
    if ko in (K_INVALID, K_INFCONF):
        st.invalid_rows += 1
    st.judged[KIND_NAMES[ko]] += 1
    st.dir["orig"][KIND_NAMES[ko]] += 1
    st.dir["swap"][KIND_NAMES[ks]] += 1
    return ok


def rec_from(row, mode_key=None):
    return {
        "exp_orig": hx(row["exp_orig"]), "exp_swap": hx(row["exp_swap"]),
        "native_orig": hx(row["native_orig"]), "native_swap": hx(row["native_swap"]),
        "dualx_orig": hx(row["dualx_orig"]), "dualy_orig": hx(row["dualy_orig"]),
        "dualx_swap": hx(row["dualx_swap"]), "dualy_swap": hx(row["dualy_swap"]),
        "kind_orig": int(row["kind_orig"]), "kind_swap": int(row["kind_swap"]), "safe": int(row["safe"]),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    rdir, bdir = sys.argv[1], sys.argv[2]
    path = lambda n: os.path.join(rdir, n)
    for n in ("nan_swap_summary.json", "nan_swap_summary.tsv", "nan_swap_modes.tsv", "nan_swap_samples.tsv", "nan_swap_edges.tsv"):
        if not os.path.isfile(path(n)):
            die("missing receipt file %s" % path(n))
    if not os.path.isfile(os.path.join(bdir, "build.json")):
        die("missing %s" % os.path.join(bdir, "build.json"))
    with open(path("nan_swap_summary.json")) as f:
        S = json.load(f)
    with open(os.path.join(bdir, "build.json")) as f:
        B = json.load(f)

    # ---------------------------------------------------------------- header / config
    check(S.get("probe") == "nan_swap", "probe name")
    check(S.get("complete") is True, "summary.complete is not true")
    check(str(S.get("arch", "")).startswith("gfx1201"), "arch %r is not gfx1201" % S.get("arch"))
    check(S.get("modes_run") == MODES, "modes_run %r != 0..15" % S.get("modes_run"))
    sa = str(S.get("suites_arg", ""))
    sel = set(sa.split(","))
    check("all" in sel or {"nan", "tri", "rand", "edges"} <= sel, "suites_arg %r does not select every suite" % sa)
    check(S.get("nan_items") == NAN_ITEMS, "nan_items %r != %d" % (S.get("nan_items"), NAN_ITEMS))
    check(S.get("nan_partners") == PARTNERS, "nan_partners %r" % S.get("nan_partners"))
    check(S.get("tri_triples") == TRI, "tri_triples %r != %d" % (S.get("tri_triples"), TRI))
    check(S.get("random_triples_per_mode_op") == RAND, "random_triples_per_mode_op %r != %d" % (S.get("random_triples_per_mode_op"), RAND))
    check(isinstance(S.get("random_triples_per_mode_op"), int) and S["random_triples_per_mode_op"] >= RAND_MIN, "random triples below minimum %d" % RAND_MIN)
    seed = str(S.get("seed", ""))
    try:
        int(seed, 16)
    except ValueError:
        check(False, "seed %r not hex" % seed)
    chunk = S.get("chunk_items") or 0
    check(chunk > 0, "chunk_items missing")
    expect = {"nan": NAN_ITEMS * PARTNERS, "tri": TRI, "rand": RAND}
    launches_exp = {"nan": -(-NAN_ITEMS // chunk) if chunk else 0, "tri": 1, "rand": 1}
    edges_meta = S.get("edges", {})
    rows_per_mode = edges_meta.get("rows_per_mode", 0)
    check(rows_per_mode > 0 and rows_per_mode == 3 * edges_meta.get("triples", -1), "edges metadata inconsistent: %r" % edges_meta)

    # ---------------------------------------------------------------- cells
    cells = {}
    for c in S.get("cells", []):
        key = (c["mode"], c["suite"])
        check(key not in cells, "duplicate cell %r" % (key,))
        cells[key] = c
    check(len(cells) == 48, "%d cells, expected 48 (16 modes x 3 suites)" % len(cells))
    tot = {k: 0 for k in COUNTERS}
    inv_cov = {}
    unknown_total = 0
    for m in MODES:
        den, rnd = m & 3, m >> 2
        exp_ctl = [0x00800000 if den & 1 else 0, 0x00400000 if den & 2 else 0,
                   0x3F800001 if rnd <= 1 else 0x3F800000, 0xBF800001 if rnd in (0, 2) else 0xBF800000]
        for s in SUITES:
            c = cells.get((m, s))
            if not check(c is not None, "missing cell mode %d suite %s" % (m, s)):
                continue
            tag = "mode %d %s" % (m, s)
            check(c.get("done") is True, tag + ": not done")
            check(c.get("denorm") == den and c.get("round") == rnd, tag + ": mode decode")
            check(c.get("expected_cases_per_op") == expect[s], tag + ": expected_cases_per_op %r != %d" % (c.get("expected_cases_per_op"), expect[s]))
            check(c.get("launches") == launches_exp[s], tag + ": launches %r != %d" % (c.get("launches"), launches_exp[s]))
            ctl = c.get("control", {})
            check(ctl.get("ctl_bad_waves") == 0, tag + ": control bad waves %r" % ctl.get("ctl_bad_waves"))
            check(ctl.get("mode_bad_waves") == 0, tag + ": MODE readback bad waves %r" % ctl.get("mode_bad_waves"))
            check(ctl.get("waves") == WAVES_PER_LAUNCH * launches_exp[s], tag + ": waves checked %r != %d" % (ctl.get("waves"), WAVES_PER_LAUNCH * launches_exp[s]))
            try:
                mo, ma = int(ctl["mode_or"], 16), int(ctl["mode_and"], 16)
                want = rnd | (den << 4)
                check((mo & 0x33) == want and (ma & 0x33) == want, tag + ": MODE readback %08x/%08x masked != %02x" % (mo, ma, want))
                obs = [int(x, 16) for x in ctl["obs"]]
                exp = [int(x, 16) for x in ctl["exp"]]
                check(exp == exp_ctl, tag + ": expected controls %r differ from independent %r" % (exp, exp_ctl))
                check(obs == exp_ctl, tag + ": observed controls %r != %r" % (obs, exp_ctl))
            except (KeyError, ValueError) as e:
                check(False, tag + ": control data unparseable (%s)" % e)
            ops = c.get("ops", {})
            for o in OPS:
                ot = tag + " " + o
                d = ops.get(o)
                if not check(d is not None, ot + ": missing op"):
                    continue
                for k in COUNTERS:
                    if not check(k in d and isinstance(d[k], int), ot + ": counter %s missing" % k):
                        d[k] = 0
                for k in COUNTERS:
                    tot[k] += d[k]
                check(d["cases"] == expect[s], ot + ": cases %d != %d" % (d["cases"], expect[s]))
                check(d.get("outputs") == 6 * expect[s], ot + ": outputs %r != %d" % (d.get("outputs"), 6 * expect[s]))
                check(d.get("count_ok") is True, ot + ": count_ok not true")
                for k in ZERO_COUNTERS:
                    check(d[k] == 0, ot + ": %s = %d (must be 0)" % (k, d[k]))
                check(d["safe_cases"] + d["unsafe_cases"] == d["cases"], ot + ": safe+unsafe != cases")
                check(d["unsafe_swap_unequal"] == d["unsafe_cases"], ot + ": unsafe %d but swap-unequal %d (all unsafe must differ)" % (d["unsafe_cases"], d["unsafe_swap_unequal"]))
                check(d["swap_unequal_any"] == d["unsafe_swap_unequal"], ot + ": swap_unequal_any %d != unsafe_swap_unequal %d (safe inputs must be equal)" % (d["swap_unequal_any"], d["unsafe_swap_unequal"]))
                check(d["unsafe_cases"] > 0, ot + ": no unsafe inputs exercised")
                check(d["both_nan_cases"] >= d["unsafe_cases"], ot + ": both_nan < unsafe")
                check(d["invalid_result_7fc00000"] == 0 and d["invalid_result_other"] == 0, ot + ": invalid results not entirely ffc00000 (7fc=%d other=%d)" % (d["invalid_result_7fc00000"], d["invalid_result_other"]))
                check(d["invalid_result_ffc00000"] == d["invalid_expected"], ot + ": invalid ffc00000 %d != invalid_expected %d" % (d["invalid_result_ffc00000"], d["invalid_expected"]))
                if s == "nan":
                    check(d["nan_expected"] == d["cases"] and d["invalid_expected"] == 0 and d["unknown_unchecked"] == 0, ot + ": NaN-sweep oracle coverage")
                    check(d["native_orig_result_nan"] == d["cases"], ot + ": NaN sweep native results not all NaN")
                else:
                    inv_cov[(m, s, o)] = d["invalid_expected"]
                    check(d["invalid_expected"] > 0, ot + ": no invalid coverage")
                unknown_total += d["unknown_unchecked"]
    check(sum(inv_cov.values()) > 0, "no invalid coverage at all")
    for k in ("selection_mismatch_cases", "native_dual_mismatch_cases", "safe_criterion_mismatch_cases", "unexpected_nan_cases",
              "control_bad_waves", "mode_readback_bad_waves", "case_count_mismatch_cells"):
        check(S.get("totals", {}).get(k) == 0, "totals.%s = %r (must be 0)" % (k, S.get("totals", {}).get(k)))
    exp_cases = 3 * 16 * sum(expect.values())
    check(S.get("totals", {}).get("cases") == exp_cases, "totals.cases %r != %d" % (S.get("totals", {}).get("cases"), exp_cases))
    check(S.get("totals", {}).get("outputs") == 6 * exp_cases, "totals.outputs %r != %d" % (S.get("totals", {}).get("outputs"), 6 * exp_cases))
    check(tot["cases"] == exp_cases, "summed cases %d != %d" % (tot["cases"], exp_cases))
    check(S.get("totals", {}).get("swap_unequal_cases_informational") == tot["swap_unequal_any"], "totals swap-unequal != summed counters")

    # ---------------------------------------------------------------- summary.tsv / modes.tsv cross-check
    _, srows = read_tsv(path("nan_swap_summary.tsv"))
    seen = set()
    for r in srows:
        key = (int(r["mode"]), r["suite"], r["op"])
        check(key not in seen, "summary.tsv duplicate %r" % (key,))
        seen.add(key)
        c = cells.get((key[0], key[1]))
        d = c["ops"].get(key[2]) if c else None
        if not check(d is not None, "summary.tsv row %r has no json cell" % (key,)):
            continue
        check(int(r["expected_cases"]) == expect[key[1]], "summary.tsv %r expected_cases" % (key,))
        check(int(r["count_ok"]) == 1, "summary.tsv %r count_ok" % (key,))
        check(int(r["outputs"]) == 6 * d["cases"], "summary.tsv %r outputs" % (key,))
        for k in COUNTERS:
            check(int(r[k]) == d[k], "summary.tsv %r counter %s %s != json %d" % (key, k, r[k], d[k]))
    check(len(seen) == 48 * 3 // 3 * 3, "summary.tsv has %d (mode,suite,op) rows, expected 144" % len(seen))

    _, mrows = read_tsv(path("nan_swap_modes.tsv"))
    msuite = {}
    for r in mrows:
        key = (int(r["mode"]), r["suite"])
        check(key not in msuite, "modes.tsv duplicate %r" % (key,))
        msuite[key] = r
    for m in MODES:
        den, rnd = m & 3, m >> 2
        exp_ctl = [0x00800000 if den & 1 else 0, 0x00400000 if den & 2 else 0,
                   0x3F800001 if rnd <= 1 else 0x3F800000, 0xBF800001 if rnd in (0, 2) else 0xBF800000]
        for s in SUITES + ["edges"]:
            r = msuite.get((m, s))
            if not check(r is not None, "modes.tsv missing mode %d suite %s" % (m, s)):
                continue
            tag = "modes.tsv mode %d %s" % (m, s)
            check(int(r["ctl_bad_waves"]) == 0 and int(r["mode_bad_waves"]) == 0, tag + ": nonzero control/readback failures")
            check(int(r["waves"]) > 0, tag + ": no waves checked")
            if s == "edges":
                check(int(r["waves"]) == (-(-rows_per_mode // 256)) * 8, tag + ": waves %s" % r["waves"])
            want = rnd | (den << 4)
            check((hx(r["mode_readback_or"]) & 0x33) == want and (hx(r["mode_readback_and"]) & 0x33) == want, tag + ": MODE readback")
            obs = [hx(r["ctl_in"]), hx(r["ctl_out"]), hx(r["ctl_round_up"]), hx(r["ctl_round_dn"])]
            check(obs == exp_ctl, tag + ": observed controls %r != %r" % (obs, exp_ctl))
            exps = [hx(r["exp_ctl_in"]), hx(r["exp_ctl_out"]), hx(r["exp_ctl_round_up"]), hx(r["exp_ctl_round_dn"])]
            check(exps == exp_ctl, tag + ": recorded expected controls differ from independent")
            if s != "edges":
                c = cells.get((m, s))
                if c:
                    check(int(r["waves"]) == c["control"]["waves"], tag + ": waves differ from json")

    # ---------------------------------------------------------------- raw edges
    _, erows = read_tsv(path("nan_swap_edges.tsv"))
    est = RowStats()
    emodes = {}
    layout = {}
    for r in erows:
        m = int(r["mode"])
        emodes.setdefault(m, []).append(r)
    check(sorted(emodes) == MODES, "edge rows modes %r != 0..15" % sorted(emodes))
    for m in MODES:
        rows = emodes.get(m, [])
        check(len(rows) == rows_per_mode, "edge mode %d: %d rows != %d" % (m, len(rows), rows_per_mode))
        seq = tuple((r["op"], r["a"], r["b"], r["c"], r["triple_idx"]) for r in rows)
        if m == MODES[0]:
            layout = seq
        else:
            check(seq == layout, "edge mode %d: input rows differ from mode 0" % m)
        for i, r in enumerate(rows):
            op = r["op"]
            if not check(op in OPS, "edge mode %d row %d: op %r" % (m, i, op)):
                continue
            check_row("edge mode %d %s %s,%s,%s" % (m, op, r["a"], r["b"], r["c"]), m, op, hx(r["a"]), hx(r["b"]), hx(r["c"]), rec_from(r), est)
    # The NaN-neighbour edges put the NaN in a; swapping supplies the b-only case.
    # Require selection-kind coverage across the two measured orders together.
    for k in ("a", "b", "invalid", "c", "infconf"):
        check(est.dir["orig"][k] + est.dir["swap"][k] > 0,
              "edge rows: no oracle kind %s coverage in either order" % k)
    check(est.safe_rows > 0 and est.unsafe_rows > 0, "edge rows: need both safe and unsafe inputs")
    check(est.invalid_rows > 0, "edge rows: no invalid coverage")

    # ---------------------------------------------------------------- raw samples
    _, prows = read_tsv(path("nan_swap_samples.tsv"))
    pst = RowStats()
    per_class = {}
    per_group = {}
    for r in prows:
        m, s, o, cl = int(r["mode"]), r["suite"], r["op"], r["class"]
        if not check(s in SUITES and o in OPS and cl in CLASSES and m in MODES, "samples: bad key %r" % ((m, s, o, cl),)):
            continue
        per_class[cl] = per_class.get(cl, 0) + 1
        g = (s, m, o, cl)
        per_group[g] = per_group.get(g, 0) + 1
        a, b, c = hx(r["a"]), hx(r["b"]), hx(r["c"])
        rec = rec_from(r)
        check(cl not in FAIL_CLASSES, "samples: failure-class row present %r" % (g,))
        check_row("sample %s mode %d %s %s %s,%s,%s" % (s, m, o, cl, r["a"], r["b"], r["c"]), m, o, a, b, c, rec, pst)
        sf = safe_input(a, b)
        unequal = rec["native_orig"] != rec["native_swap"] or rec["dualx_orig"] != rec["dualx_swap"] or rec["dualy_orig"] != rec["dualy_swap"]
        if cl == "swap_unequal_unsafe":
            check((not sf) and unequal, "samples: swap_unequal_unsafe row not unsafe/unequal %s,%s,%s" % (r["a"], r["b"], r["c"]))
        if cl == "safe_both_nan":
            check(sf and isnan(a) and isnan(b), "samples: safe_both_nan row not safe both-NaN")
        if cl in ("sel_ok_invalid_or_c", "sel_ok_single_nan"):
            ko = rec["kind_orig"]
            check(ko != K_NONE and ko != K_UNKNOWN, "samples: %s row without NaN/invalid oracle" % cl)
        check(int(r["slot"]) < SLOTS, "samples: slot >= 32")
    for g, n in per_group.items():
        check(n <= SLOTS, "samples: group %r has %d > %d rows" % (g, n, SLOTS))
    for m in MODES:
        for s in SUITES:
            for o in OPS:
                check(per_group.get((s, m, o, "swap_unequal_unsafe"), 0) > 0, "samples: no swap_unequal_unsafe evidence for %s mode %d %s" % (s, m, o))
    check(sorted({int(r["mode"]) for r in prows}) == MODES, "samples do not cover all 16 modes")
    for d in ("orig", "swap"):
        for k in ("a", "b", "invalid", "c"):
            check(pst.dir[d][k] > 0, "samples: no %s-order oracle kind %s coverage" % (d, k))

    # ---------------------------------------------------------------- build / hashes
    check(B.get("arch") == "gfx1201", "build.json arch %r" % B.get("arch"))
    check(B.get("opcode_checks_ok") is True, "build.json opcode_checks_ok is %r" % B.get("opcode_checks_ok"))
    if "uuid_source" in S:  # corrected-source receipts: the label must be the canonical ROCr id, not hex-encoded ASCII
        check(str(S.get("uuid", "")).startswith("GPU-") and S.get("uuid_source") == "hip_ascii_hex", "uuid %r (source %r) is not a normalised GPU- id" % (S.get("uuid"), S.get("uuid_source")))
        if S.get("uuid_rocr_visible_devices"):
            check(S.get("uuid_matches_rocr_visible_devices") is True, "uuid %r != ROCR_VISIBLE_DEVICES %r" % (S.get("uuid"), S.get("uuid_rocr_visible_devices")))
    else:
        notes.append("receipt predates uuid normalisation fields (uuid label may be hex-encoded ASCII)")
    hashes = {"receipt": {}, "build": {}}
    for n in sorted(os.listdir(rdir)):
        p = path(n)
        if os.path.isfile(p) and n != "nan_swap_check.json":
            hashes["receipt"][n] = sha256_file(p)
    bj = os.path.join(bdir, "build.json")
    hashes["build"]["build.json"] = sha256_file(bj)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nan_swap.hip")
    src_status = "absent"
    if os.path.isfile(src):
        sh = sha256_file(src)
        hashes["build"]["nan_swap.hip(beside checker)"] = sh
        src_status = "match" if sh == B.get("source_sha256") else "MISMATCH"
        check(sh == B.get("source_sha256"), "nan_swap.hip sha256 %s != build.json source_sha256 %s" % (sh, B.get("source_sha256")))
    binary = os.path.join(bdir, "nan_swap_gfx1201")
    for fname, key in (("nan_swap_gfx1201", "binary_sha256"), (B.get("device_obj") or "", "device_obj_sha256"),
                       (B.get("host_obj") or "", "host_obj_sha256"), ("nan_swap_gfx1201.disasm.txt", "disasm_sha256")):
        if not fname:
            continue
        p = os.path.join(bdir, fname)
        if os.path.isfile(p):
            h = sha256_file(p)
            hashes["build"][fname] = h
            check(h == B.get(key), "%s sha256 %s != build.json %s %s" % (fname, h, key, B.get(key)))
    if os.path.isfile(binary):
        ident = path("identity.host.txt")
        if os.path.isfile(ident):
            md5 = md5_file(binary)
            with open(ident) as f:
                txt = f.read()
            lines = [l.split()[0] for l in txt.splitlines() if len(l.split()) == 2 and len(l.split()[0]) == 32]
            if lines:
                check(md5 in lines, "identity.host.txt md5 %r does not include binary md5 %s" % (lines, md5))
            else:
                notes.append("identity.host.txt has no md5 line")
    else:
        notes.append("binary not present in BUILD_DIR; binary hash taken from build.json only")

    # ---------------------------------------------------------------- identity / report
    ident_txt = None
    if os.path.isfile(path("identity.host.txt")):
        with open(path("identity.host.txt")) as f:
            ident_txt = f.read().splitlines()[:40]
    env = S.get("env", {})
    job_env = {k: v for k, v in env.items() if k.startswith(("FABRIC", "HIP_", "ROCR_", "CARD", "NS_", "NP_"))}
    ok = not failures
    report = {
        "check": "nan_swap_check", "pass": ok, "failure_count": len(failures), "failures": failures[:50], "notes": notes,
        "receipt_dir": os.path.abspath(rdir), "build_dir": os.path.abspath(bdir),
        "source_hash_status": src_status, "build_source_sha256": B.get("source_sha256"), "build_binary_sha256": B.get("binary_sha256"),
        "hashes": hashes,
        "job": {"arch": S.get("arch"), "device": S.get("device"), "uuid": S.get("uuid"), "pci": S.get("pci"), "host": S.get("host"),
                "uuid_source": S.get("uuid_source"), "uuid_raw_bytes_hex": S.get("uuid_raw_bytes_hex"),
                "uuid_rocr_visible_devices": S.get("uuid_rocr_visible_devices"), "uuid_matches_rocr_visible_devices": S.get("uuid_matches_rocr_visible_devices"),
                "argv": S.get("argv"), "seed": S.get("seed"), "env": job_env, "total_secs": S.get("total_secs"), "identity_host_txt": ident_txt},
        "counts": {
            "cells": len(cells), "modes": len(MODES), "nan_items": NAN_ITEMS, "partners": PARTNERS, "tri": TRI, "rand": RAND,
            "total_cases": tot["cases"], "total_outputs": 6 * tot["cases"], "expected_cases": exp_cases,
            "swap_unequal": tot["swap_unequal_any"], "unsafe_cases": tot["unsafe_cases"], "safe_cases": tot["safe_cases"],
            "both_nan_cases": tot["both_nan_cases"], "invalid_expected": tot["invalid_expected"],
            "invalid_result_ffc00000": tot["invalid_result_ffc00000"], "invalid_result_7fc00000": tot["invalid_result_7fc00000"],
            "invalid_result_other": tot["invalid_result_other"], "unknown_unchecked": unknown_total,
            "edge_rows": len(erows), "edge_rows_per_mode": rows_per_mode, "sample_rows": len(prows),
            "edge_oracle_kinds": est.judged, "edge_direction_kinds": est.dir, "sample_oracle_kinds": pst.judged, "sample_direction_kinds": pst.dir,
            "edge_safe_rows": est.safe_rows, "edge_unsafe_rows": est.unsafe_rows, "sample_classes": per_class,
        },
    }
    with open(path("nan_swap_check.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    c = report["counts"]
    print("nan_swap_check: %s" % ("PASS" if ok else "FAIL (%d failures)" % len(failures)))
    print("  job: %s %s uuid=%s pci=%s host=%s" % (S.get("arch"), S.get("device"), S.get("uuid"), S.get("pci"), S.get("host")))
    print("  cells=%d cases=%d outputs=%d swap_unequal=%d unsafe=%d safe=%d" % (c["cells"], c["total_cases"], c["total_outputs"], c["swap_unequal"], c["unsafe_cases"], c["safe_cases"]))
    print("  invalid expected=%d ffc00000=%d 7fc00000=%d other=%d unknown_unchecked=%d" % (c["invalid_expected"], c["invalid_result_ffc00000"], c["invalid_result_7fc00000"], c["invalid_result_other"], unknown_total))
    print("  edge rows=%d (%d/mode) sample rows=%d source_hash=%s" % (c["edge_rows"], rows_per_mode, c["sample_rows"], src_status))
    for fmsg in failures[:20]:
        print("  FAIL: " + fmsg)
    print("  wrote %s" % path("nan_swap_check.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
