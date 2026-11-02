#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1 byte-exact driver of the gfx1100 K3 producer `mq_i4_a4_r128_k640_gfx1100` (standalone qualification only).

Imports the certified `mq_i4_a4_oracle.py` that sits next to this file (build_cases / write_cases / expected_bytes /
describe_mismatch / slots / to_bf16_bits / gen_fwht_signs / selftest; none of its algorithms are copied) and the pinned
`pilot_solve.a4_emulate` (SHA256 below; a different file is refused).  The expected bytes are CPU-derived only and are
never taken from the GPU.  Runner: example `mq_i4_a4_gfx1100_g1_oracle` (raw HIP load, same ABI as the gfx1201
`mq_i4_a4_byte_oracle`, symbol `mq_i4_a4_r128_k640_gfx1100`).  See docs/MQ_I4_A4_GFX1100_G1_ORACLE.md.

Case set = every original case family and the 64-row batch-independence check of the base harness, plus
  * `codescale`: BF16-rounded rows built as signed code (-8..7) * f16 positive scale
    (0x0000, 0x0001 subnormal, 0x03ff, 0x0400, 0x3c00, 0x7bff maximum).  K3 itself consumes BF16 activations, not weight
    scales; these are only deliberately structured BF16 inputs (code * scale; scale 0 with a negative code is -0.0).
  * `real` (optional, never part of the synthetic verdict): `--real-manifest` (real-weight-derived inputs, ctl3-hot
    gate/up over deterministic A4) and/or `--real-h/--real-sorted/--real-l`.  Absent -> reported `pending`, not passed.
  * negative launches, each run in its own directory with deliberately mutated host inputs: `neg_signs/` (wrong sign
    tables) and `neg_sorted/` (wrong sorted slot vector).  Their device output is compared with the ORIGINAL case's CPU
    expected bytes and must differ (mismatch bytes > 0).

Work-directory layout (W = --work):
  W/base/ W/neg_signs/ W/neg_sorted/   runner case dirs (manifest.json + N.{h,sorted,signs1,signs2}.bin, runner writes N.xq.out.bin)
  W/expected/N.xq.expected.bin         CPU expected bytes of the base cases (optional on the compare host; recomputed if absent)
  W/prepare.json                       case metadata, input hashes, expected hashes, negative mutation records, real inputs
  W/run_runner.sh                      `sh W/run_runner.sh RUNNER CODEOBJECT`: runs the runner on the 3 dirs, writes
                                       W/receipts/{base,neg_signs,neg_sorted}.{cmd,rc,stderr,runner.json} and
                                       W/receipts/{runner,codeobject}.sha256

Modes (one is chosen; without a flag = prepare + run + compare in one process):
  --selftest       CPU only: base rotation twin selftest + codescale generator checks.
  --prepare-only   CPU: write W as above (and a receipt); no GPU.
  --compare-only   CPU: read W (+ the runner receipts produced by run_runner.sh), recompute / verify expected bytes,
                   compare outputs, write the final receipt.
  (default)        prepare, run `[--runner-prefix ...] sh W/run_runner.sh --runner --codeobject`, compare.

Exit status: 0 for verdict PASS or PASS_SYNTHETIC_REAL_PENDING, 1 otherwise.  The JSON receipt is written on every
path, including runner failure; failures are never suppressed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE_HARNESS = os.path.join(HERE, "mq_i4_a4_oracle.py")
RUNNER_SOURCE = os.path.join(REPO, "crates", "hipfire-runtime", "examples", "mq_i4_a4_gfx1100_g1_oracle.rs")
PILOT_SHA256 = "53965f2328e32d86c7b87460a02573a065424e220383891da210441dc9f6c0df"
SCHEMA = "mq-i4-a4-gfx1100-g1-oracle-v1"
SYMBOL = "mq_i4_a4_r128_k640_gfx1100"
ARCH = "gfx1100"
EXPECT_UUID = "GPU-43390a851e296ee5"
EXPECT_BDF = "0000:67:00.0"
LINK_CAVEAT = ("GPU-43390a851e296ee5 (0000:67:00.0) is an x16 endpoint behind a PCIe 4.0 x4 root-port link; every timing and "
               "bandwidth figure in this receipt carries that limitation.")
TIMING_SCOPE = ("per-iteration host wall clock around launch_kernel_blob + device_synchronize of the K3 kernel alone "
                "(launch + sync included, no memset or copy in the window), synchronized per loop; not an end-to-end claim")
REAL_MANIFEST_LABEL = "real-weight-derived inputs (ctl3-hot gate/up over deterministic A4)"
REAL_PLAIN_LABEL = "real BF16 grouped down input (caller supplied)"
ROLES = ("base", "neg_signs", "neg_sorted")
F16_SCALES = (0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x7BFF)
K, G, REC = 640, 5, 72
N_SOLO = 64


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ codescale inputs
def f16_value(bits):
    return np.array([bits], np.uint16).view(np.float16).astype(np.float32)[0]


def code_scale_rows(base, rng):
    """([(f16 bits, bf16 rows [n, K])], mixed-scale bf16 rows [64, K]): signed code -8..7 times a positive f16 scale."""
    ar = np.arange(K)
    per_scale = []
    for bits in F16_SCALES:
        sc = f16_value(bits)
        rows = [((ar + sh) % 16) - 8 for sh in range(16)]  # cyclic codes, every phase
        rows += [np.full(K, code) for code in range(-8, 8)]  # a constant code
        rows += [((ar // 8 + sh) % 16) - 8 for sh in (0, 5, 11)]  # blocks of 8 equal codes
        rows += [rng.integers(-8, 8, K) for _ in range(8)]
        per_scale.append((bits, base.to_bf16_bits(np.array(rows, np.float32) * sc)))
    mixed = []
    for r in range(64):  # each K128 group of a row on a different scale
        codes = rng.integers(-8, 8, (G, 128)).astype(np.float32)
        sc = np.array([f16_value(F16_SCALES[(r + g) % len(F16_SCALES)]) for g in range(G)], np.float32)[:, None]
        mixed.append((codes * sc).reshape(K))
    return per_scale, base.to_bf16_bits(np.array(mixed, np.float32))


def code_scale_cases(base, seed, s1, s2):
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x5CA1E5))  # separate stream: the base cases stay untouched
    per_scale, mixed = code_scale_rows(base, rng)
    cases = []
    for bits, rows in per_scale:
        p = len(rows)
        cases.append(base.Case(f"codescale_f16_{bits:04x}", "codescale", rows, rng.permutation(p).astype(np.int32), p, s1, s2))
    cases.append(base.Case("codescale_mixed_scales", "codescale", mixed, rng.permutation(len(mixed)).astype(np.int32), len(mixed), s1, s2))
    allrows = np.concatenate([r for _, r in per_scale] + [mixed], axis=0)
    p = len(allrows)
    cases.append(base.Case("codescale_all_sparse", "codescale", allrows, base.slots(rng, p, p + 11, 0.1, force_valid=(0, p - 1)), p + 11, s1, s2))
    return cases


def selftest_codescale(base):
    per_scale, mixed = code_scale_rows(base, np.random.Generator(np.random.PCG64(1)))
    ar = np.arange(K)
    for bits, rows in per_scale:
        assert rows.dtype == np.uint16 and rows.shape == (43, K), bits
        vals = (rows.astype(np.uint32) << 16).view(np.float32)
        if bits == 0x3C00:  # scale 1.0: every code is exactly representable
            assert np.array_equal(vals[:16], np.stack([(((ar + sh) % 16) - 8) for sh in range(16)]).astype(np.float32))
        if bits == 0x0000:  # scale 0: magnitude zero, negative codes give -0.0
            assert not np.any(rows & 0x7FFF) and np.any(rows == 0x8000)
        else:
            assert np.any(rows & 0x7FFF)
    assert mixed.shape == (64, K) and mixed.dtype == np.uint16
    return {"scales": [f"0x{b:04x}" for b in F16_SCALES], "rows_per_scale": 43, "mixed_rows": 64}


# ------------------------------------------------------------------ real inputs
def real_cases(base, a, s1, s2):
    """([Case], {name: meta}, meta-of-run) for `--real-h/--real-sorted/--real-l` and `--real-manifest`."""
    cases, meta, run = [], {}, {}
    if a.real_h or a.real_sorted or a.real_l is not None:
        if not (a.real_h and a.real_sorted and a.real_l is not None):
            raise SystemExit("--real-h, --real-sorted and --real-l must be given together")
        h = np.fromfile(a.real_h, "<u2").astype(np.uint16)
        if len(h) == 0 or len(h) % K:
            raise SystemExit(f"--real-h: {len(h)} u16 values is not a positive multiple of {K}")
        h = h.reshape(-1, K)
        srt = np.fromfile(a.real_sorted, "<i4").astype(np.int32)
        if len(srt) != len(h):
            raise SystemExit(f"--real-sorted: {len(srt)} entries for {len(h)} rows")
        valid = srt[srt >= 0]
        if len(np.unique(valid)) != len(valid) or (len(valid) and (valid.max() >= a.real_l)):
            raise SystemExit("--real-sorted: valid slots must be unique and < --real-l")
        cases.append(base.Case("real_down_input", "real", h, srt, a.real_l, s1, s2))
        meta["real_down_input"] = {"label": REAL_PLAIN_LABEL, "h_path": os.path.abspath(a.real_h), "h_file_sha256": base.sha256_file(a.real_h),
                                   "sorted_path": os.path.abspath(a.real_sorted), "sorted_file_sha256": base.sha256_file(a.real_sorted), "l": a.real_l,
                                   "nonfinite_bf16_values": int(np.count_nonzero(((h >> 7) & 0xFF) == 0xFF))}
    if a.real_manifest:
        mpath = os.path.abspath(a.real_manifest)
        with open(mpath, "rb") as f:
            raw = f.read()
        man = json.loads(raw)
        run = {"label": REAL_MANIFEST_LABEL, "manifest": mpath, "manifest_sha256": sha256_bytes(raw), "artifact_sha256": man.get("artifact_sha256"),
               "layers": man.get("layers"), "source_mode": man.get("source_mode"),
               "note": "not an XTS-route or real-distribution measurement; deterministic identity slots [0..rows), L=rows, fixed signs"}
        for c in man["cases"]:
            hp = c["h_path"] if os.path.isabs(c["h_path"]) else os.path.join(os.path.dirname(mpath), c["h_path"])
            h = np.fromfile(hp, "<u2").astype(np.uint16)
            rows = int(c["rows"])
            if rows < 1 or len(h) != rows * K:
                raise SystemExit(f"real manifest case {c['name']}: {hp} holds {len(h)} u16 values, expected {rows}*{K}")
            name = "real_" + str(c["name"])
            cases.append(base.Case(name, "real", h.reshape(rows, K), np.arange(rows, dtype=np.int32), rows, s1, s2))
            meta[name] = {"label": REAL_MANIFEST_LABEL, "manifest_case": c["name"], "layer": c.get("layer"), "expert": c.get("expert"), "rows": rows,
                          "h_path": hp, "h_file_sha256": base.sha256_file(hp), "manifest_sha256": run["manifest_sha256"], "artifact_sha256": man.get("artifact_sha256"),
                          "nonfinite_bf16_values": int(np.count_nonzero(((h >> 7) & 0xFF) == 0xFF))}
    return cases, meta, run


# ------------------------------------------------------------------ negatives
def mut_negate_s1(base, c):
    return base.Case("neg_signs_negate_s1", "negative", c.h, c.sorted, c.l, -c.s1, c.s2), {"signs1"}


def mut_swap_signs(base, c):
    return base.Case("neg_signs_swap_s1_s2", "negative", c.h, c.sorted, c.l, c.s2, c.s1), {"signs1", "signs2"}


def mut_rotate_slots(base, c):
    s = np.where(c.sorted >= 0, (c.sorted + 1) % c.l, -1).astype(np.int32)
    return base.Case("neg_sorted_rotate_slots", "negative", c.h, s, c.l, c.s1, c.s2), {"sorted"}


def mut_swap_pair(base, c):
    s = c.sorted.copy()
    valid = np.nonzero(s >= 0)[0]
    for i in range(len(valid) - 1):  # first two valid rows with different data, so the swap changes the bytes
        a, b = valid[i], valid[i + 1]
        if not np.array_equal(c.h[a], c.h[b]):
            s[a], s[b] = s[b], s[a]
            return base.Case("neg_sorted_swap_pair", "negative", c.h, s, c.l, c.s1, c.s2), {"sorted"}
    raise SystemExit("neg_sorted_swap_pair: no two valid rows with different data")


def mut_drop_valid(base, c):
    s = c.sorted.copy()
    s[int(np.nonzero(s >= 0)[0][0])] = -1
    return base.Case("neg_sorted_drop_valid", "negative", c.h, s, c.l, c.s1, c.s2), {"sorted"}


NEGATIVES = (  # (dir, original case, mutation, description)
    ("neg_signs", "rand10k", mut_negate_s1, "signs1 negated"),
    ("neg_signs", "rand10k", mut_swap_signs, "signs1 and signs2 swapped"),
    ("neg_sorted", "rand10k", mut_rotate_slots, "every valid slot s -> (s+1) mod L"),
    ("neg_sorted", "rand10k_sparse", mut_swap_pair, "slots of the first two valid rows swapped"),
    ("neg_sorted", "chunk4096_edges_valid", mut_drop_valid, "first valid row turned into a pad row (-1)"),
)
FILES = ("h", "sorted", "signs1", "signs2")


def case_hashes(base, d, name):
    return {f: base.sha256_file(os.path.join(d, f"{name}.{f}.bin")) for f in FILES}


def load_cases(base, d, metas):
    out = []
    for m in metas:
        n = m["name"]
        h = np.fromfile(os.path.join(d, f"{n}.h.bin"), "<u2").astype(np.uint16).reshape(-1, K)
        srt = np.fromfile(os.path.join(d, f"{n}.sorted.bin"), "<i4").astype(np.int32)
        s1 = np.fromfile(os.path.join(d, f"{n}.signs1.bin"), "<f4").astype(np.float32)
        s2 = np.fromfile(os.path.join(d, f"{n}.signs2.bin"), "<f4").astype(np.float32)
        out.append(base.Case(n, m["family"], h, srt, m["l"], s1, s2))
    return out


# ------------------------------------------------------------------ prepare
def runner_script(a):
    timing = f"--iterations {int(a.iterations)} --warmup {int(a.warmup)}" if a.iterations else ""
    return f"""#!/bin/sh
# Generated by mq_i4_a4_gfx1100_oracle.py --prepare-only. usage: sh run_runner.sh RUNNER CODEOBJECT
# Runs the K3 gfx1100 runner on base/, neg_signs/, neg_sorted/ (identity + timing flags baked in). Run it as ONE fabric job
# on the expected card; no CU-mask or device-visibility variable is set here.
W=$(cd "$(dirname "$0")" && pwd)
RUNNER=$1
CODEOBJECT=$2
[ -n "$RUNNER" ] && [ -n "$CODEOBJECT" ] || {{ echo "usage: sh run_runner.sh RUNNER CODEOBJECT" >&2; exit 2; }}
mkdir -p "$W/receipts"
rm -f "$W"/base/*.xq.out.bin "$W"/neg_signs/*.xq.out.bin "$W"/neg_sorted/*.xq.out.bin
rm -f "$W"/receipts/*
sha256sum "$RUNNER" > "$W/receipts/runner.sha256"
sha256sum "$CODEOBJECT" > "$W/receipts/codeobject.sha256"
overall=0
run() {{
  role=$1
  shift
  set -- "$RUNNER" --codeobject "$CODEOBJECT" --cases "$W/$role" --expect-uuid {shlex.quote(a.expect_uuid)} --expect-bdf {shlex.quote(a.expect_bdf)} "$@"
  printf '%s ' "$@" > "$W/receipts/$role.cmd"
  "$@" > "$W/receipts/$role.runner.json" 2> "$W/receipts/$role.stderr"
  rc=$?
  echo "$rc" > "$W/receipts/$role.rc"
  [ "$rc" -eq 0 ] || overall=1
}}
run base {timing}
run neg_signs
run neg_sorted
exit $overall
"""


def prepare(a, receipt, base, W):
    ps = base.load_pilot(a.pilot_solve)
    dirs = {r: os.path.join(W, r) for r in ROLES}
    for d in list(dirs.values()) + [os.path.join(W, "expected"), os.path.join(W, "receipts")]:
        os.makedirs(d, exist_ok=True)
    cases = base.build_cases(a.seed, a.rows)
    s1, s2 = cases[0].s1, cases[0].s2
    cases += code_scale_cases(base, a.seed, s1, s2)
    rc_, rmeta, rrun = real_cases(base, a, s1, s2)
    cases += rc_
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate case names")
    by = {c.name: c for c in cases}
    base.write_cases(dirs["base"], cases)
    meta = []
    for c in cases:
        want = base.expected_bytes(ps, c)  # CPU only
        want.tofile(os.path.join(W, "expected", f"{c.name}.xq.expected.bin"))
        meta.append({"name": c.name, "family": c.family, "p": int(len(c.h)), "l": int(c.l), "valid_rows": int((c.sorted >= 0).sum()),
                     "inputs": case_hashes(base, dirs["base"], c.name), "expected_sha256": sha256_bytes(want.tobytes()), "real": rmeta.get(c.name)})
    negs, by_dir = [], {}
    for d, of, fn, desc in NEGATIVES:
        nc, expect_changed = fn(base, by[of])
        by_dir.setdefault(d, []).append(nc)
        negs.append({"name": nc.name, "dir": d, "of": of, "mutation": desc, "expected_mutated_inputs": sorted(expect_changed), "p": int(len(nc.h)), "l": int(nc.l)})
    for d, lst in by_dir.items():
        base.write_cases(dirs[d], lst)
    for n in negs:
        n["inputs"] = case_hashes(base, dirs[n["dir"]], n["name"])
        orig = next(m for m in meta if m["name"] == n["of"])["inputs"]
        n["original_inputs"] = orig
        n["mutated_inputs"] = sorted(f for f in FILES if n["inputs"][f] != orig[f])
        n["mutation_ok"] = n["mutated_inputs"] == n["expected_mutated_inputs"]
    prep = {"schema": SCHEMA + "-prepare", "created": utc(), "seed": a.seed, "rows": a.rows, "expect_uuid": a.expect_uuid, "expect_bdf": a.expect_bdf,
            "iterations": a.iterations, "warmup": a.warmup, "hashes": receipt["hashes"], "cases": meta, "negatives": negs, "real_run": rrun,
            "layout": {r: os.path.relpath(d, W) for r, d in dirs.items()}}
    with open(os.path.join(W, "prepare.json"), "w") as f:
        json.dump(prep, f, indent=1)
    script = os.path.join(W, "run_runner.sh")
    with open(script, "w") as f:
        f.write(runner_script(a))
    os.chmod(script, 0o755)
    receipt.update(cases_prepared=len(cases), negatives_prepared=len(negs), real_inputs_prepared=len(rmeta), work=W, run_script=script,
                   input_hashes={m["name"]: m["inputs"] for m in meta}, negative_input_hashes={n["name"]: n["inputs"] for n in negs})
    return prep


# ------------------------------------------------------------------ compare
def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def read_role(W, role, base):
    r = {"role": role}
    d = os.path.join(W, "receipts")
    rc = read_text(os.path.join(d, f"{role}.rc"))
    r["cmd"] = (read_text(os.path.join(d, f"{role}.cmd")) or "").strip() or None
    r["rc"] = int(rc.strip()) if rc and rc.strip().lstrip("-").isdigit() else None
    r["stderr_tail"] = (read_text(os.path.join(d, f"{role}.stderr")) or "").strip()[-2000:]
    out = read_text(os.path.join(d, f"{role}.runner.json")) or ""
    r["runner_receipt"] = None
    if out.strip():
        try:
            r["runner_receipt"] = json.loads(out.strip().splitlines()[-1])
        except json.JSONDecodeError:
            r["runner_stdout_tail"] = out[-2000:]
    rr = r["runner_receipt"] or {}
    dev = rr.get("device") or {}
    with open(os.path.join(W, role, "manifest.json")) as f:
        want = sorted(c["name"] for c in json.load(f)["cases"])
    got = sorted(x.get("case") for x in rr.get("launches", []))
    r["identity"] = {"uuid": dev.get("uuid"), "bdf": dev.get("bdf"), "arch": dev.get("arch"), "identity_match": dev.get("identity_match")}
    r["identity_ok"] = bool(dev.get("identity_match")) and dev.get("arch") == ARCH
    r["launched_cases_ok"] = got == want
    r["ok"] = r["rc"] == 0 and r["identity_ok"] and r["launched_cases_ok"] and rr.get("symbol") == SYMBOL
    return r


def read_sha_file(path):
    t = read_text(path)
    return t.split()[0] if t and t.split() else None


def provenance_failures(receipt, pre_hash, bins):
    """Everything that ties this comparison to the prepared inputs and the measured binaries; any entry fails runner_ok."""
    fails = []
    for k in ("driver_sha256", "base_harness_sha256", "pilot_solve_sha256"):
        if not receipt["hashes"].get(k) or receipt["hashes"].get(k) != pre_hash.get(k):
            fails.append(f"{k} differs from the hash recorded by prepare")
    for key in ("runner", "codeobject"):
        if not bins[f"{key}_sha256_remote"]:
            fails.append(f"{key}: missing remote sha256 (W/receipts/{key}.sha256)")
        if f"{key}_local_missing" in bins:
            fails.append(f"{key}: provided local path {bins[f'{key}_local_missing']} does not exist")
        if bins.get(f"{key}_remote_matches_local") is False:
            fails.append(f"{key}: local sha256 differs from (or remote missing for) the remote-measured sha256")
    return fails


def compare(a, receipt, base, W, prep):
    pre_hash = prep["hashes"]
    for k in ("driver_sha256", "base_harness_sha256", "pilot_solve_sha256"):
        receipt["hashes"][k + "_matches_prepare"] = receipt["hashes"].get(k) == pre_hash.get(k)
    roles = {r: read_role(W, r, base) for r in ROLES}
    receipt["runner_roles"] = roles
    runner_ok = all(r["ok"] for r in roles.values())
    devs = {(r["identity"]["uuid"], r["identity"]["bdf"]) for r in roles.values()}
    identity = {"uuid": next(iter(devs))[0] if len(devs) == 1 else None, "bdf": next(iter(devs))[1] if len(devs) == 1 else None, "arch": roles["base"]["identity"]["arch"]}
    identity["pinned_match"] = identity["uuid"] == a.expect_uuid and identity["bdf"] == a.expect_bdf and identity["arch"] == ARCH and len(devs) == 1
    receipt["device"] = identity
    runner_ok = runner_ok and identity["pinned_match"]
    # binary / codeobject hashes, as measured on the machine that ran the job (and locally when the paths were given)
    bins = {"runner_sha256_remote": read_sha_file(os.path.join(W, "receipts", "runner.sha256")),
            "codeobject_sha256_remote": read_sha_file(os.path.join(W, "receipts", "codeobject.sha256"))}
    for key, path in (("runner", a.runner), ("codeobject", a.codeobject)):
        if path and not os.path.exists(path):
            bins[f"{key}_local_missing"] = os.path.abspath(path)  # a local path was provided but cannot be hashed: gating failure
        elif path:
            bins[f"{key}_path"], bins[f"{key}_sha256_local"] = os.path.abspath(path), base.sha256_file(path)
            bins[f"{key}_remote_matches_local"] = bool(bins[f"{key}_sha256_remote"]) and bins[f"{key}_sha256_remote"] == bins[f"{key}_sha256_local"]
    receipt["binaries"] = bins
    receipt["hashes"]["runner_binary_sha256"] = bins.get("runner_sha256_local") or bins["runner_sha256_remote"]
    receipt["hashes"]["codeobject_sha256"] = bins.get("codeobject_sha256_local") or bins["codeobject_sha256_remote"]
    # expected bytes: CPU only (files written by prepare, hash-checked; recomputed when absent or --recompute-expected)
    ps = None
    cases = load_cases(base, os.path.join(W, "base"), prep["cases"])
    by = {c.name: c for c in cases}
    expected = {}
    for m in prep["cases"]:
        c = by[m["name"]]
        path = os.path.join(W, "expected", f"{c.name}.xq.expected.bin")
        if a.recompute_expected or not os.path.exists(path):
            ps = ps or base.load_pilot(a.pilot_solve)
            want = base.expected_bytes(ps, c)
        else:
            want = np.fromfile(path, np.uint8)
        if sha256_bytes(want.tobytes()) != m["expected_sha256"]:
            raise SystemExit(f"{c.name}: expected bytes disagree with the hash recorded by prepare")
        expected[c.name] = want
    # inputs on disk must still be what prepare hashed
    inputs_ok = all(case_hashes(base, os.path.join(W, "base"), m["name"]) == m["inputs"] for m in prep["cases"])
    receipt["inputs_unchanged_since_prepare"] = inputs_ok
    prov_fail = provenance_failures(receipt, pre_hash, bins)
    receipt["provenance_failures"] = prov_fail
    runner_ok = runner_ok and not prov_fail

    def output(role, d, name):
        if not roles[role]["ok"]:
            return None  # a failed runner job leaves no trustworthy outputs
        p = os.path.join(W, d, f"{name}.xq.out.bin")
        return np.fromfile(p, np.uint8) if os.path.exists(p) else None

    results, outs = [], {}
    for m in prep["cases"]:
        c = by[m["name"]]
        want = expected[c.name]
        got = output("base", "base", c.name)
        row = {"case": c.name, "family": c.family, "p": m["p"], "l": m["l"], "valid_rows": m["valid_rows"], "bytes": int(len(want)),
               "expected_sha256": m["expected_sha256"], "input_sha256": m["inputs"]}
        if got is None:
            row.update(pass_=False, error="no output (runner failed or output missing)")
        else:
            outs[c.name] = got
            row["output_sha256"] = sha256_bytes(got.tobytes())
            if got.shape != want.shape:
                row.update(pass_=False, error=f"output length {got.size} != {want.size}")
            else:
                bad = int(np.count_nonzero(got != want))
                row.update(mismatch_bytes=bad, pass_=bad == 0)
                if bad:
                    row["first_mismatches"] = base.describe_mismatch(c, got, want)
        results.append(row)
    # batch independence (solo launch bytes == the same row's bytes inside the batch launch)
    big, indep = by["rand10k"], []
    for i in range(N_SOLO):
        solo, batch = outs.get(f"solo_{i}"), outs.get("rand10k")
        if solo is None or batch is None:
            indep.append(False)
            continue
        r = (i * 157 + 3) % len(big.h)
        indep.append(bool(np.array_equal(solo.reshape(G, 1, REC)[:, 0], batch.reshape(G, big.l, REC)[:, int(big.sorted[r])])))
    synthetic = [r for r in results if r["family"] != "real"]
    families = {}
    for r in synthetic:
        f = families.setdefault(r["family"], {"cases": 0, "rows": 0, "pass": True})
        f["cases"] += 1
        f["rows"] += r["valid_rows"]
        f["pass"] &= bool(r["pass_"])
    synthetic_pass = all(bool(r["pass_"]) for r in synthetic) and all(indep) and inputs_ok
    # negatives
    neg_rows = []
    orig_inputs = {m["name"]: m["inputs"] for m in prep["cases"]}
    for n in prep["negatives"]:
        # the negative inputs on disk are rehashed here: they must be exactly what prepare wrote, differ from the original
        # in exactly the intended files, and the original must be the one prepare hashed
        disk = case_hashes(base, os.path.join(W, n["dir"]), n["name"])
        live_mutated = sorted(f for f in FILES if disk[f] != orig_inputs[n["of"]][f])
        live_ok = (n["mutation_ok"] and disk == n["inputs"] and n["original_inputs"] == orig_inputs[n["of"]]
                   and live_mutated == n["expected_mutated_inputs"] and live_mutated == n["mutated_inputs"])
        want = expected[n["of"]]
        got = output(n["dir"], n["dir"], n["name"])
        row = {"case": n["name"], "dir": n["dir"], "of": n["of"], "mutation": n["mutation"], "mutated_inputs": live_mutated,
               "expected_mutated_inputs": n["expected_mutated_inputs"], "mutation_ok": live_ok, "mutation_ok_at_prepare": n["mutation_ok"],
               "inputs_sha256": n["inputs"], "original_inputs_sha256": n["original_inputs"], "expected_original_sha256": sha256_bytes(want.tobytes())}
        if got is None:
            row.update(pass_=False, error="no output (runner failed or output missing)")
        else:
            row["output_sha256"] = sha256_bytes(got.tobytes())
            row["differs_from_positive_output"] = bool(n["of"] in outs and not np.array_equal(outs[n["of"]], got))
            if got.shape != want.shape:
                row.update(pass_=False, error=f"output length {got.size} != {want.size}")
            else:
                bad = int(np.count_nonzero(got != want))
                row.update(mismatch_bytes=bad, pass_=bad > 0 and live_ok)
        neg_rows.append(row)
    negative_pass = bool(neg_rows) and all(bool(r["pass_"]) for r in neg_rows)
    # real inputs
    real_rows = [r for r in results if r["family"] == "real"]
    real = {"label": None, "cases": []}
    if not real_rows:
        real.update(status="pending", reason="no real input supplied (--real-manifest or --real-h/--real-sorted/--real-l); not passed")
    else:
        real["status"] = "passed" if all(bool(r["pass_"]) for r in real_rows) else "failed"
        real.update({k: v for k, v in (prep.get("real_run") or {}).items() if k != "note"})
        real["note"] = (prep.get("real_run") or {}).get("note")
        for r in real_rows:
            real["cases"].append({**{k: v for k, v in r.items() if k != "pass_"}, "pass": bool(r["pass_"]),
                                  "source": next(m["real"] for m in prep["cases"] if m["name"] == r["case"])})
        real["label"] = (real["cases"][0]["source"] or {}).get("label")
    if not runner_ok or not synthetic_pass or not negative_pass or real["status"] == "failed":
        verdict = "FAIL"
    elif real["status"] == "pending":
        verdict = "PASS_SYNTHETIC_REAL_PENDING"
    else:
        verdict = "PASS"
    receipt.update(cases=[{**{k: v for k, v in r.items() if k != "pass_"}, "pass": bool(r["pass_"])} for r in results], families=families,
                   total_valid_rows=sum(r["valid_rows"] for r in synthetic), batch_independence={"rows": len(indep), "pass": all(indep)},
                   negatives=[{**{k: v for k, v in r.items() if k != "pass_"}, "pass": bool(r["pass_"])} for r in neg_rows],
                   negative_counts={"cases": len(neg_rows), "passed": sum(bool(r["pass_"]) for r in neg_rows),
                                    "mismatch_bytes": {r["case"]: r.get("mismatch_bytes") for r in neg_rows}},
                   mismatch_counts={"synthetic_total_bytes": sum(r.get("mismatch_bytes", 0) for r in synthetic),
                                    "failed_cases": [r["case"] for r in results if not r["pass_"]]},
                   real=real, runner_ok=runner_ok, synthetic_pass=synthetic_pass, negative_pass=negative_pass, verdict=verdict, pass_=verdict == "PASS")
    receipt["pass"] = receipt.pop("pass_")
    receipt["timing"] = timing_block(prep, roles["base"])
    return verdict


def timing_block(prep, role):
    if not prep["iterations"]:
        return {"requested": None, "link_caveat": LINK_CAVEAT}
    per = {x["case"]: x["timing"] for x in (role["runner_receipt"] or {}).get("launches", []) if "timing" in x}
    out = {"requested": {"iterations": prep["iterations"], "warmup": prep["warmup"]}, "scope": TIMING_SCOPE, "link_caveat": LINK_CAVEAT,
           "headline_case": "rand10k", "headline": per.get("rand10k"), "per_case": per}
    if prep["iterations"] < 10:
        out["note"] = "fewer than 10 iterations (e.g. the default 1) is a single-launch sample, not a timing measurement; run timing in a separate timing-class fabric job with its own --work"
    return out


# ------------------------------------------------------------------ main
def finish(a, receipt):
    if not a.receipt:
        return
    path = os.path.abspath(a.receipt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(receipt, f, indent=1, default=str)
    os.replace(tmp, path)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--compare-only", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--receipt")
    ap.add_argument("--runner", help="runner binary (default mode; compare-only: optional local hash cross-check)")
    ap.add_argument("--codeobject", help="mq_i4_a4_pm_gfx1100.hxaco (default mode; compare-only: optional local hash cross-check)")
    ap.add_argument("--runner-prefix", default="", help="shell-split prefix of the job, e.g. 'fabric-run.sh --class correct --card xtx ... --'")
    ap.add_argument("--iterations", type=int, default=1, help="timed launches after the correctness launch; default 1 (single sample in a correctness job); 0 disables; use a separate --work and a fabric timing-class job for real timing")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--real-h")
    ap.add_argument("--real-sorted")
    ap.add_argument("--real-l", type=int)
    ap.add_argument("--real-manifest")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0x4B31A401)
    ap.add_argument("--pilot-solve")
    ap.add_argument("--base-harness", default=BASE_HARNESS)
    ap.add_argument("--expect-base-sha256", help="refuse a base harness with a different SHA256")
    ap.add_argument("--expect-uuid", default=EXPECT_UUID)
    ap.add_argument("--expect-bdf", default=EXPECT_BDF)
    ap.add_argument("--emitter-source", action="append", default=[], help="kernel emitter source file(s) to hash into the receipt")
    ap.add_argument("--recompute-expected", action="store_true")
    a = ap.parse_args()
    if a.iterations < 0 or a.warmup < 0 or (a.warmup and not a.iterations):
        raise SystemExit("--iterations/--warmup: non-negative, and --warmup needs --iterations")
    return a


def run(a, receipt):
    bad_env = sorted(k for k in os.environ if "CU_MASK" in k.upper())
    if bad_env:
        raise SystemExit(f"refusing to run: CU-mask environment variable(s) set: {bad_env}")
    base = load_module("mq_i4_a4_oracle", a.base_harness)
    h = receipt["hashes"]
    h["driver_sha256"] = base.sha256_file(os.path.abspath(__file__))
    h["base_harness_sha256"] = base.sha256_file(a.base_harness)
    h["runner_source_sha256"] = base.sha256_file(RUNNER_SOURCE) if os.path.exists(RUNNER_SOURCE) else None
    h["emitter_sources"] = {p: base.sha256_file(p) for p in a.emitter_source}
    if a.expect_base_sha256 and a.expect_base_sha256 != h["base_harness_sha256"]:
        raise SystemExit(f"base harness SHA256 {h['base_harness_sha256']} != expected {a.expect_base_sha256}")
    a.pilot_solve = a.pilot_solve or base.PILOT
    h["pilot_solve"], h["pilot_solve_sha256"] = a.pilot_solve, base.sha256_file(a.pilot_solve)
    h["pilot_solve_pinned_sha256"] = PILOT_SHA256
    if h["pilot_solve_sha256"] != PILOT_SHA256:
        raise SystemExit(f"pilot_solve SHA256 {h['pilot_solve_sha256']} != pinned {PILOT_SHA256}")
    with redirect_stdout(io.StringIO()):  # the base selftest prints its own JSON line
        base.selftest()  # SystemExit on a rotation-twin mismatch
    receipt["base_selftest"] = "pass"
    if a.selftest:
        receipt["codescale_selftest"] = selftest_codescale(base)
        receipt["verdict"], receipt["pass"] = "SELFTEST_PASS", True
        return 0
    if a.rows < 10000:
        raise SystemExit("--rows must be at least 10000 (the acceptance population)")
    if not a.work or not a.receipt:
        raise SystemExit("--work and --receipt are required")
    W = os.path.abspath(a.work)
    receipt["work"] = W
    prep = None
    if not a.compare_only:
        prep = prepare(a, receipt, base, W)
        if a.prepare_only:
            receipt["verdict"], receipt["pass"] = "PREPARED", None
            return 0
        if not (a.runner and a.codeobject):
            raise SystemExit("--runner and --codeobject are required to launch")
        for p in (a.runner, a.codeobject):
            if not os.path.exists(p):
                raise SystemExit(f"{p}: not found")
        cmd = shlex.split(a.runner_prefix) + ["sh", os.path.join(W, "run_runner.sh"), os.path.abspath(a.runner), os.path.abspath(a.codeobject)]
        job = subprocess.run(cmd, capture_output=True, text=True)
        receipt["runner_job"] = {"cmd": cmd, "rc": job.returncode, "stdout_tail": job.stdout[-2000:], "stderr_tail": job.stderr[-2000:]}
    else:
        with open(os.path.join(W, "prepare.json")) as f:
            prep = json.load(f)
        a.iterations, a.warmup = prep["iterations"], prep["warmup"]
        a.seed, a.rows = prep["seed"], prep["rows"]
    receipt.update(seed=prep["seed"], rows=prep["rows"], iterations=prep["iterations"], warmup=prep["warmup"])
    verdict = compare(a, receipt, base, W, prep)
    return 0 if verdict in ("PASS", "PASS_SYNTHETIC_REAL_PENDING") else 1


def main():
    a = parse()
    receipt = {"schema": SCHEMA, "symbol": SYMBOL, "expected_arch": ARCH, "expected_uuid": a.expect_uuid, "expected_bdf": a.expect_bdf,
               "link_caveat": LINK_CAVEAT, "timing_scope": TIMING_SCOPE, "started": utc(), "driver_argv": sys.argv, "cwd": os.getcwd(),
               "mode": "selftest" if a.selftest else "prepare-only" if a.prepare_only else "compare-only" if a.compare_only else "full",
               "env": {k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")},
               "numpy": np.__version__, "hashes": {}, "pass": False, "verdict": "FAIL"}
    try:
        import torch
        receipt["torch"] = torch.__version__
    except ImportError:
        receipt["torch"] = None
    rc = 1
    try:
        rc = run(a, receipt)
    except SystemExit as e:
        if e.code not in (0, None):
            receipt["error"] = str(e.code)
            print(e.code, file=sys.stderr)
        rc = 0 if e.code in (0, None) else 1
    except Exception as e:  # noqa: BLE001  recorded in the receipt and re-signalled through rc
        receipt["error"] = f"{type(e).__name__}: {e}"
        receipt["traceback"] = traceback.format_exc()[-4000:]
        print(traceback.format_exc(), file=sys.stderr)
    receipt["finished"], receipt["rc"] = utc(), rc
    finish(a, receipt)
    keys = ("mode", "verdict", "pass", "total_valid_rows", "families", "batch_independence", "negative_counts", "error")
    print(json.dumps({k: receipt[k] for k in keys if k in receipt} | {"real_status": receipt.get("real", {}).get("status")}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
