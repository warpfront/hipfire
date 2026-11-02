#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact acceptance harness of the gfx1201 K3 producer `mq_i4_a4_r128_k640`.

The kernel (crates/peacemaker-isa/src/kernels/mq_i4_a4.rs) takes grouped BF16 SwiGLU rows h [P][640], the flat-slot
vector `sorted` (-1 = pad row), the R128 sign tables and writes the down sidecar xq [5][L][72 B] (`block_i4_128`
records, element j in byte j/2, low nibble first), record (g, sorted[p]) from row p group g, nothing for pad rows.

The expected bytes are the CPU twin: the R128 rotation of `mq_rotate_x_128_v2` / `qwen4_moe_rotate128_i4.hip`
(widen bf16, * signs1, in-lane butterflies 1,2 then lane butterflies, (* 0.0883883476f) * signs2; every step one f32
rounding, stage order 1,2,4,8,16,32,64) followed by `pilot_solve.a4_emulate(candidates=2)` on the rotated rows, placed
at record g*L + sorted[p] in an xq buffer pre-filled with 0xA5 (so a missing, extra or misplaced record, or a write
for a pad row, differs). Every byte must match; there is no tolerance. The runtime `mq_i4_reference` has no A4
producer (only the `A4Block` consumer), so `a4_emulate` is the only CPU oracle.

Phases (one invocation does all unless limited):
  * `--selftest`      CPU only: the vectorised rotation equals a lane-for-lane transcription of the HIP kernel on
                      several rows; exits before any case generation or launch.
  * (default)         generate cases into --work, run the GPU runner (`mq_i4_a4_byte_oracle`, loads --codeobject), compare.
  * `--prepare-only`  generate cases and the expected sidecars, do not launch.
  * `--compare-only`  compare the runner's existing `N.xq.out.bin` files with the expected bytes.

Case families (all gating): rand10k (10,000 rows, all slots valid, mixed distributions), rand10k_sparse (10,240 rows,
~20% pads, L > valid slots), extremes (every single-spike position, Walsh rows whose rotation is one spike and exact
zeros, ties, huge / tiny / subnormal bf16, mixed scales, outliers), zeros (+0 / -0 rows, zero groups inside live
rows), shapes P = 1, 2, 7, 8, 15, 16, 17 (sparse slots, first / last slot, all-pad), chunk boundaries P = 4095, 4096,
4097, 8191, 8192, 8193 (valid / pad rows at the 4096 multiples), random sign tables, and `solo_*` single-row launches
whose bytes must equal the same row's bytes inside the 10k batch (batch independence).  `bigl_*` are 256 rows over L = 1M and 1024 rows over L = 13M flat slots (record indices past 2^32/72^2 and 2^32/72).

Run it under the card scheduler, e.g. (from the worktree root):
  /home/kaden/qcal/locks/fabric-sched/fabric-run.sh --class correct --card A --mem-gb 2 --priority P1 \
    --owner <owner> --max-secs 1800 -- python3 crates/peacemaker-isa/tools/mq_i4_a4_oracle.py \
    --codeobject /abs/mq_i4_a4_pm_gfx1201.co --runner <target>/release/examples/mq_i4_a4_byte_oracle \
    --work /abs/work/dir --receipt /abs/rcpt/r4.k3.oracle.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

K, G, REC = 640, 5, 72
SCALE = np.frombuffer(np.uint32(0x3DB504F3).tobytes(), np.float32)[0]  # 0.0883883476f, 1/sqrt(128) as f32
PILOT = "/home/kaden/qcal/release-0.4.2/subq4/calib2/pilot_solve.py"
BF16_CLIP = float(2.0 ** 120)  # |x| <= 2^120: the rotation (<= 128 * 0.0884 * max) stays finite in f32
FILL = 0xA5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gen_fwht_signs(seed, n):
    """`gen_fwht_signs` of the engine / quantizer (LCG, bit 16)."""
    state, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        state = ((state * 1103515245) + 12345) & 0x7FFFFFFF
        out.append(1.0 if (state >> 16) & 1 else -1.0)
    return np.array(out, np.float32)


def to_bf16_bits(x):
    """Round-to-nearest-even f32 -> bf16 bits (finite inputs, clipped to +-2^120)."""
    x = np.clip(np.asarray(x, np.float32), -BF16_CLIP, BF16_CLIP).astype(np.float32)
    u = x.view(np.uint32)
    r = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (r >> 16).astype(np.uint16)


def rotate(h_u16, s1, s2):
    """Rotated f32 rows [P, 5, 128] of bf16 rows [P, 640]: the vectorised twin of qwen4_moe_rotate128_i4.hip."""
    with np.errstate(all="ignore"):
        x = (h_u16.astype(np.uint32) << 16).view(np.float32).reshape(-1, G, 128)
        v = (x * s1.astype(np.float32)).astype(np.float32)
        for stride in (1, 2, 4, 8, 16, 32, 64):
            v = v.reshape(-1, G, 128 // (2 * stride), 2, stride)
            lo, hi = v[..., 0, :], v[..., 1, :]
            v = np.stack(((lo + hi).astype(np.float32), (lo - hi).astype(np.float32)), axis=-2)
        v = v.reshape(-1, G, 128)
        return ((v * SCALE).astype(np.float32) * s2.astype(np.float32)).astype(np.float32)


def rotate_lanes(row_u16, group, s1, s2):
    """Lane-for-lane transcription of the HIP kernel (32 lanes x 4 elements, ds_swizzle xor partners) for one group."""
    f = np.float32
    with np.errstate(all="ignore"):
        raw = row_u16[group * 128:(group + 1) * 128]
        v = [[f(0.0)] * 4 for _ in range(32)]
        for lane in range(32):
            d0 = lane * 4
            vals = [(np.uint32(raw[d0 + i]) << np.uint32(16)) for i in range(4)]
            vals = [np.frombuffer(np.uint32(u).tobytes(), np.float32)[0] for u in vals]
            for i in range(4):
                v[lane][i] = f(vals[i] * f(s1[d0 + i]))
            a = v[lane]
            t = a[0]; a[0] = f(a[0] + a[1]); a[1] = f(t - a[1])
            t = a[2]; a[2] = f(a[2] + a[3]); a[3] = f(t - a[3])
            t = a[0]; a[0] = f(a[0] + a[2]); a[2] = f(t - a[2])
            t = a[1]; a[1] = f(a[1] + a[3]); a[3] = f(t - a[3])
        for stride in (1, 2, 4, 8, 16):
            prior = [list(x) for x in v]
            for lane in range(32):
                for i in range(4):
                    p = prior[lane ^ stride][i]
                    cur = prior[lane][i]
                    v[lane][i] = f(p - cur) if lane & stride else f(cur + p)
        out = np.zeros(128, np.float32)
        for lane in range(32):
            for i in range(4):
                out[lane * 4 + i] = f(f(v[lane][i] * SCALE) * f(s2[lane * 4 + i]))
        return out


def selftest():
    rng = np.random.Generator(np.random.PCG64(7))
    s1, s2 = gen_fwht_signs(43, 128), gen_fwht_signs(1043, 128)
    h = to_bf16_bits(rng.standard_normal((4, K)) * 3.0)
    h[1, :] = to_bf16_bits(np.where(np.arange(K) % 5 == 0, 1.0, -0.375))
    fast = rotate(h, s1, s2)
    for r in range(h.shape[0]):
        for g in range(G):
            slow = rotate_lanes(h[r], g, s1, s2)
            if slow.tobytes() != fast[r, g].tobytes():
                raise SystemExit(f"selftest: rotation twin differs at row {r} group {g}")
    print(json.dumps({"selftest": "pass", "rows": int(h.shape[0]), "groups": G}))


def load_pilot(path):
    spec = importlib.util.spec_from_file_location("pilot_solve", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pilot_solve"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ cases
class Case:
    def __init__(self, name, family, h, sorted_, l, s1, s2):
        assert h.dtype == np.uint16 and h.shape[1] == K and sorted_.dtype == np.int32 and len(sorted_) == len(h)
        valid = sorted_[sorted_ >= 0]
        assert len(np.unique(valid)) == len(valid) and (len(valid) == 0 or valid.max() < l)
        self.name, self.family, self.h, self.sorted, self.l, self.s1, self.s2 = name, family, h, sorted_, l, s1, s2


def slots(rng, p, l, pad_frac, force_valid=(), force_pad=()):
    valid = rng.random(p) >= pad_frac
    for i in force_valid:
        if 0 <= i < p:
            valid[i] = True
    for i in force_pad:
        if 0 <= i < p:
            valid[i] = False
    n = int(valid.sum())
    assert n <= l
    out = np.full(p, -1, np.int32)
    out[valid] = rng.permutation(l)[:n].astype(np.int32)
    return out


def mixed_rows(rng, p, tag=0):
    """p rows of mixed distributions (bf16 bits)."""
    rows = np.empty((p, K), np.float32)
    kind = (np.arange(p) + tag) % 9
    for i in range(p):
        k = kind[i]
        if k == 0:
            r = rng.standard_normal(K)
        elif k == 1:
            r = rng.standard_normal(K) * 2.0 ** rng.uniform(-20, 20)
        elif k == 2:
            r = rng.uniform(-1, 1, K)
        elif k == 3:
            r = np.clip(rng.standard_t(2, K), -1e4, 1e4)
        elif k == 4:
            r = np.where(rng.random(K) < 0.1, rng.standard_normal(K), 0.0)
        elif k == 5:
            r = rng.standard_normal(K)
            r[rng.integers(K)] *= 2.0 ** rng.uniform(5, 60)
        elif k == 6:
            r = rng.standard_normal((G, 128)) * (2.0 ** rng.integers(-30, 30, (G, 1)))
            r = r.reshape(K)
        elif k == 7:
            r = rng.standard_normal(K) * (2.0 ** (-90 if (i // 9) % 2 == 0 else 100))
        else:
            r = np.round(rng.standard_normal(K) * 4.0) / 4.0  # coarse values: many exact butterfly sums and ties
        rows[i] = r
    return to_bf16_bits(rows)


def walsh_rows(s1, count):
    """Rows whose rotation is a single spike plus exact zeros: x = s1 * W_j (W_j the Walsh row of the lane bits)."""
    out = np.empty((count, K), np.float32)
    idx = np.arange(128)
    for r in range(count):
        for g in range(G):
            j = (r * 7 + g * 29 + 3) % 128
            w = np.array([-1.0 if bin(i & j).count("1") & 1 else 1.0 for i in idx], np.float32)
            out[r, g * 128:(g + 1) * 128] = s1 * w * np.float32(2.0 ** (((r * 5 + g * 3) % 9) - 4))
    return to_bf16_bits(out)


def extreme_rows(rng, s1):
    rows = []
    for pos in range(K):  # single nonzero at every position: all rotated magnitudes equal (an amax tie in every lane)
        r = np.zeros(K, np.float32)
        r[pos] = 1.0 if pos % 3 else -2.0 ** ((pos % 7) - 3)
        rows.append(r)
    rows.append(np.full(K, 1.0, np.float32))
    rows.append(np.where(np.arange(K) % 2, -1.0, 1.0).astype(np.float32))
    rows.append(np.where(np.arange(K) % 128 < 64, 1.0, -1.0).astype(np.float32))
    for e in (120, 100, 60, 0, -60, -90, -100, -110):  # huge .. tiny uniform magnitudes with random signs
        rows.append((np.where(rng.random(K) < 0.5, -1.0, 1.0) * 2.0 ** e * rng.uniform(1, 2, K)).astype(np.float32))
    for e in (-120, -125, -128, -131, -133):  # subnormal bf16 / f32 rotations (rotated amax below 2^-100)
        rows.append((np.where(rng.random(K) < 0.5, -1.0, 1.0) * 2.0 ** e * rng.uniform(1, 2, K)).astype(np.float32))
    for _ in range(64):  # one huge outlier over tiny neighbours
        r = rng.standard_normal(K).astype(np.float32) * np.float32(2.0 ** -60)
        r[rng.integers(K)] = 2.0 ** 100
        rows.append(r)
    for _ in range(64):  # per-group magnitudes 2^-100 .. 2^100
        rows.append((rng.standard_normal((G, 128)) * (2.0 ** rng.integers(-100, 100, (G, 1)))).reshape(K).astype(np.float32))
    for step in (0.0625, 0.25, 1.0):  # multiples of a power of two: exact sums, rotated values on a grid (ties)
        rows.append((np.round(rng.standard_normal(K) / step) * step).astype(np.float32))
    base = to_bf16_bits(np.array(rows, np.float32))
    return np.concatenate([base, walsh_rows(s1, 256)], axis=0)


def zero_rows(rng):
    pz = np.zeros(K, np.uint16)
    nz = np.full(K, 0x8000, np.uint16)
    rows = [pz, nz, np.where(np.arange(K) % 2, pz, nz).astype(np.uint16)]
    live = mixed_rows(rng, 61, 1)
    for i in range(live.shape[0]):  # zero a random subset of K128 groups (all five for some rows)
        mask = rng.random(G) < (0.4 if i % 5 else 1.0)
        for g in range(G):
            if mask[g]:
                live[i, g * 128:(g + 1) * 128] = 0
    t = np.zeros(K, np.uint16)  # a single bf16 subnormal in an otherwise zero group
    t[200] = 0x0001
    rows.append(t)
    return np.concatenate([np.array(rows), live], axis=0)


def build_cases(seed, total_rows):
    rng = np.random.Generator(np.random.PCG64(seed))
    s1, s2 = gen_fwht_signs(43, 128), gen_fwht_signs(1043, 128)
    cases = []

    def add(name, family, h, sorted_, l, a=s1, b=s2):
        cases.append(Case(name, family, h, sorted_, l, a, b))

    h = mixed_rows(rng, total_rows)
    add("rand10k", "rand10k", h, rng.permutation(total_rows).astype(np.int32), total_rows)
    p = total_rows + 240
    add("rand10k_sparse", "rand10k_sparse", mixed_rows(rng, p, 4), slots(rng, p, p + 2048, 0.2, force_valid=(0, p - 1)), p + 2048)
    ext = extreme_rows(rng, s1)
    add("extremes", "extremes", ext, slots(rng, len(ext), len(ext) + 13, 0.2, force_valid=(0, len(ext) - 1)), len(ext) + 13)
    z = zero_rows(rng)
    add("zeros", "zeros", z, slots(rng, len(z), len(z) + 5, 0.1), len(z) + 5)
    for p in (1, 2, 7, 8, 15, 16, 17):
        l = p + 5
        s = slots(rng, p, l, 0.0 if p == 1 else 0.3, force_valid=(0, p - 1))
        if p == 1:
            s[0] = l - 1  # the last slot
        add(f"rows{p}", "shapes", mixed_rows(rng, p, p), s, l)
    add("rows8_allpad", "shapes", mixed_rows(rng, 8), np.full(8, -1, np.int32), 8)
    add("rows16_first_last_slot", "shapes", mixed_rows(rng, 16, 2), np.concatenate([[0], np.full(14, -1), [20]]).astype(np.int32), 21)
    for p in (4095, 4096, 4097, 8191, 8192, 8193):
        edges = [0, 4095, 4096, 4097, 8191, 8192, p - 1]
        add(f"chunk{p}_edges_valid", "chunks", mixed_rows(rng, p, p), slots(rng, p, p + 101, 0.25, force_valid=edges), p + 101)
        add(f"chunk{p}_edges_pad", "chunks", mixed_rows(rng, p, p + 1), slots(rng, p, p + 101, 0.25, force_pad=edges), p + 101)
    add("rand_signs", "signs", mixed_rows(rng, 512, 3), slots(rng, 512, 600, 0.1), 600,
        np.where(rng.random(128) < 0.5, -1.0, 1.0).astype(np.float32), np.where(rng.random(128) < 0.5, -1.0, 1.0).astype(np.float32))
    add("signs_all_plus", "signs", mixed_rows(rng, 64, 5), slots(rng, 64, 70, 0.1), 70, np.ones(128, np.float32), np.ones(128, np.float32))
    big = cases[0]
    for i in range(64):  # solo launches of rows of the 10k batch
        r = (i * 157 + 3) % total_rows
        add(f"solo_{i}", "solo", big.h[r:r + 1].copy(), np.array([0], np.int32), 1)
    # Large flat-slot counts: record index g*L + slot reaches past 2^32/72^2 (~828K; the 64-bit address product once
    # read a clobbered low word) and, at L = 13M, past 2^32/72 (~59.6M; the 32-bit low product itself wraps).
    for p, l in ((256, 1_000_000), (1024, 13_000_000)):
        add(f"bigl_p{p}_l{l}", "bigl", mixed_rows(rng, p, p), slots(rng, p, l, 0.1, force_valid=(0, p - 1)), l)
    return cases


def write_cases(work, cases):
    os.makedirs(work, exist_ok=True)
    manifest = []
    for c in cases:
        c.h.astype("<u2").tofile(os.path.join(work, f"{c.name}.h.bin"))
        c.sorted.astype("<i4").tofile(os.path.join(work, f"{c.name}.sorted.bin"))
        c.s1.astype("<f4").tofile(os.path.join(work, f"{c.name}.signs1.bin"))
        c.s2.astype("<f4").tofile(os.path.join(work, f"{c.name}.signs2.bin"))
        manifest.append({"name": c.name, "p": int(len(c.h)), "l": int(c.l)})
    with open(os.path.join(work, "manifest.json"), "w") as f:
        json.dump({"cases": manifest}, f)


def expected_bytes(ps, c):
    """The xq buffer a correct producer leaves: records of every valid row, 0xA5 everywhere else."""
    buf = np.full((G, c.l, REC), FILL, np.uint8)
    valid = np.nonzero(c.sorted >= 0)[0]
    if len(valid):
        x = rotate(c.h[valid], c.s1, c.s2).reshape(len(valid), K)
        recs = np.frombuffer(ps.a4_emulate(x, 2), np.uint8).reshape(len(valid), G, REC)
        for g in range(G):
            buf[g, c.sorted[valid]] = recs[:, g]
    return buf.reshape(-1)


def describe_mismatch(c, got, want):
    """First mismatching records as (row, group, slot, field)."""
    diff = np.nonzero(got.reshape(G, c.l, REC) != want.reshape(G, c.l, REC))
    owner = {int(s): int(p) for p, s in enumerate(c.sorted) if s >= 0}
    seen, out = set(), []
    for g, slot, byte in zip(*diff):
        key = (int(g), int(slot))
        if key in seen:
            continue
        seen.add(key)
        field = "d" if byte < 4 else "s" if byte < 8 else f"qs[{int(byte) - 8}]"
        out.append({"row": owner.get(int(slot), -1), "group": int(g), "slot": int(slot), "first_byte": int(byte), "field": field})
        if len(out) == 3:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--codeobject")
    ap.add_argument("--runner")
    ap.add_argument("--work")
    ap.add_argument("--receipt")
    ap.add_argument("--pilot-solve", default=PILOT)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0x4B31A401)
    ap.add_argument("--expect-uuid")
    ap.add_argument("--expect-bdf")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    if not a.work:
        raise SystemExit("--work is required")
    if a.rows < 10000:
        raise SystemExit("--rows must be at least 10000 (the acceptance population)")
    ps = load_pilot(a.pilot_solve)
    cases = build_cases(a.seed, a.rows)
    receipt = {"schema": "mq-i4-a4-oracle-v1", "symbol": "mq_i4_a4_r128_k640", "seed": a.seed, "rows": a.rows,
               "pilot_solve": a.pilot_solve, "pilot_solve_sha256": sha256_file(a.pilot_solve), "harness_sha256": sha256_file(__file__),
               "numpy": np.__version__, "torch": __import__("torch").__version__}
    if not a.compare_only:
        write_cases(a.work, cases)
    expected = {c.name: expected_bytes(ps, c) for c in cases}
    receipt["cases_prepared"] = len(cases)
    if a.prepare_only:
        print(json.dumps(receipt))
        return
    if not a.compare_only:
        if not (a.codeobject and a.runner):
            raise SystemExit("--codeobject and --runner are required to launch")
        cmd = [a.runner, "--codeobject", a.codeobject, "--cases", a.work]
        for flag, value in (("--expect-uuid", a.expect_uuid), ("--expect-bdf", a.expect_bdf)):
            if value:
                cmd += [flag, value]
        run = subprocess.run(cmd, capture_output=True, text=True)
        receipt["runner_rc"], receipt["runner_stderr"] = run.returncode, run.stderr.strip()[-2000:]
        receipt["codeobject"], receipt["codeobject_sha256"] = a.codeobject, sha256_file(a.codeobject)
        receipt["runner"], receipt["runner_sha256"] = a.runner, sha256_file(a.runner)
        if run.stdout.strip():
            try:
                receipt["runner_receipt"] = json.loads(run.stdout.strip().splitlines()[-1])
                receipt["device"] = receipt["runner_receipt"].get("device")
            except json.JSONDecodeError:
                receipt["runner_stdout"] = run.stdout[-2000:]
        if run.returncode != 0:
            receipt["pass"] = False
            finish(a, receipt)
            raise SystemExit(f"runner failed rc={run.returncode}: {run.stderr.strip()[-500:]}")
    results, outs = [], {}
    for c in cases:
        got = np.fromfile(os.path.join(a.work, f"{c.name}.xq.out.bin"), np.uint8)
        outs[c.name] = got
        want = expected[c.name]
        row = {"case": c.name, "family": c.family, "p": int(len(c.h)), "l": int(c.l), "valid_rows": int((c.sorted >= 0).sum()),
               "bytes": int(len(want)), "output_sha256": hashlib.sha256(got.tobytes()).hexdigest(),
               "expected_sha256": hashlib.sha256(want.tobytes()).hexdigest()}
        if got.shape != want.shape:
            row.update(pass_=False, error=f"output length {got.size} != {want.size}")
        else:
            bad = int(np.count_nonzero(got != want))
            row["mismatch_bytes"] = bad
            row["pass_"] = bad == 0
            if bad:
                row["first_mismatches"] = describe_mismatch(c, got, want)
        results.append(row)
    # Batch independence: a row launched alone produces exactly the bytes it has inside the 10k batch.
    big, indep = cases[0], []
    for i in range(64):
        r = (i * 157 + 3) % a.rows
        solo = outs[f"solo_{i}"].reshape(G, 1, REC)[:, 0]
        slot = int(big.sorted[r])
        batch = outs["rand10k"].reshape(G, big.l, REC)[:, slot]
        indep.append(bool(np.array_equal(solo, batch)))
    families = {}
    for r in results:
        f = families.setdefault(r["family"], {"cases": 0, "rows": 0, "pass": True})
        f["cases"] += 1
        f["rows"] += r["valid_rows"]
        f["pass"] &= bool(r["pass_"])
    receipt.update(cases=[{**{k: v for k, v in r.items() if k != "pass_"}, "pass": bool(r["pass_"])} for r in results], families=families,
                   total_valid_rows=sum(r["valid_rows"] for r in results), batch_independence={"rows": len(indep), "pass": all(indep)},
                   pass_=all(bool(r["pass_"]) for r in results) and all(indep))
    receipt["pass"] = receipt.pop("pass_")
    finish(a, receipt)
    print(json.dumps({k: receipt[k] for k in ("pass", "total_valid_rows", "families", "batch_independence")}))
    if not receipt["pass"]:
        raise SystemExit(1)


def finish(a, receipt):
    if a.receipt:
        os.makedirs(os.path.dirname(os.path.abspath(a.receipt)), exist_ok=True)
        with open(a.receipt, "w") as f:
            json.dump(receipt, f, indent=1)


if __name__ == "__main__":
    main()
