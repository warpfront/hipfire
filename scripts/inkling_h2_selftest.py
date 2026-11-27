#!/usr/bin/env python3
"""Self-tests for inkling_h2.py (CPU only, no GPU, no network).

  inkling_h2_selftest.py [--threads T] [--golden-bin h2-golden] [--work DIR]
                         [--template B.hfq --src-dir X/calib/src] [--e2e]

--golden-bin  Rust golden generator (built from the real mq_symmetric.rs / float16.rs; see the receipt) used
              to prove bit parity of the f16 truncation and of the full symmetric qt44 encode.
--template    B.hfq (Rust-encoded symmetric qt44, S=1) + --src-dir: python RTN must reproduce its bytes.
--e2e         synthetic one-layer HFHS + source dump through `solve` and `heldout`.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling_h2 as H  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(name + (": " + detail if detail else ""))


def skip(name, why):
    RESULTS.append((name, None, why))
    print(f"[SKIP] {name} -- {why}", flush=True)


# ------------------------------------------------------------------------------------------------
def test_rotation():
    r = H.rotation().numpy()
    err = np.abs(r @ r.T - np.eye(256)).max()
    check("rotation orthogonality (R R^T = I, f64)", err < 1e-13, f"max|RR^T-I|={err:.2e}")
    # R must be the matrix of the f32 FWHT: fwht(e_j) == R[:, j]
    eye = torch.from_numpy(np.eye(256, dtype=np.float32))
    cols = H.fwht256(eye).numpy()  # row j = fwht(e_j)
    err = np.abs(cols - r.T).max()
    check("fwht256(f32) equals R (columns)", err < 1e-6, f"max err {err:.2e}")
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((4, 512)).astype(np.float32))
    back = H.fwht256(H.fwht256(x), inverse=True)
    err = float((back - x).abs().max())
    check("inverse fwht round trip", err < 1e-5, f"max err {err:.2e}")


def test_awq_law():
    import math as m
    rng = np.random.default_rng(2)
    v = np.exp(rng.normal(0, 3, 500)).astype(np.float32)
    v[3], v[7], v[9] = 0.0, np.nan, np.inf
    for alpha in (0.0, 0.1, 0.35, 0.55):
        # verbatim rx_quant.compute_awq_scales (glibc math) reference
        half_alpha = float(np.float32(alpha)) * 0.5
        logs, total = [], 0.0
        for x in v.tolist():
            vc = min(max(float(x), 1e-12), 1e30) if not m.isnan(x) else 1e-12
            ls = half_alpha * m.log(vc)
            logs.append(ls)
            total += ls
        mean = total / len(logs)
        ref = np.clip(np.array([m.exp(l - mean) for l in logs], dtype=np.float64).astype(np.float32),
                      np.float32(1e-2), np.float32(1e2)).astype(np.float32)
        got = H.awq_scales(v, alpha)
        diff = np.abs(got.astype(np.float64) - ref) / ref
        check(f"AWQ law alpha={alpha}", diff.max() < 2e-7, f"max rel diff {diff.max():.2e} (numpy vs glibc exp/log)")
    check("alpha=0 gives S == 1 exactly", np.all(H.awq_scales(v, 0.0) == 1.0))
    bits = H.f32_to_f16_trunc_np(np.array([1.0, 1.0004882, 0.99999, 1e-3, 3.14159, 100.0], dtype=np.float32))
    ok = bits[0] == 0x3C00 and bits[1] == 0x3C00  # truncation, not rounding (RNE would give 0x3C01)
    check("S16 serializer truncates (hipfire f32_to_f16)", ok, f"bits {[hex(b) for b in bits]}")


def adversarial_weights(m, k, seed):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((m, k)) * 0.02).astype(np.float32)
    w[1] = (rng.laplace(0, 0.01, k)).astype(np.float32)
    w[2] = 0.0                                                         # all-zero row (zero halves)
    w[3] = (rng.standard_normal(k) * 1e-9).astype(np.float32)          # underflowing scales
    w[4] = (rng.standard_normal(k) * 3e2).astype(np.float32)           # large
    w[5, :] = 0.0625                                                   # constant row
    w[6] = np.round(rng.standard_normal(k) * 4) * 0.125                # exact grid ties
    w[7, k - 128:] = 0.0                                               # DN-like zero tail
    return w


def test_f16_and_rtn_vs_rust(golden_bin, work):
    if not golden_bin:
        skip("f16 trunc + symmetric qt44 encode bit parity vs Rust", "no --golden-bin")
        return
    gdir = Path(work) / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    bits = np.concatenate((rng.integers(0, 2 ** 32, 200000, dtype=np.uint64).astype(np.uint32),
                           np.array([0, 0x80000000, 0x3F800000, 0x7F800000, 0x7FC00000, 0x477FE000, 0x38800000,
                                     0x33800000, 0x33000000, 0x00000001], dtype=np.uint32)))
    bits.astype("<u4").tofile(gdir / "f32_bits.u32")
    cases = {"gauss_8_1280": 11, "adv_8_1280": 12, "adv_8_2304": 13, "adv_16_256": 14, "gauss_256_2304": 15,
             "gauss_128_1280": 16}
    for name, seed in cases.items():
        _, m, k = name.split("_")
        w = adversarial_weights(int(m), int(k), seed) if name.startswith("adv") else \
            (np.random.default_rng(seed).standard_normal((int(m), int(k))) * 0.02).astype(np.float32)
        w.astype("<f4").tofile(gdir / f"{name}.w.f32")
    subprocess.run([str(golden_bin), str(gdir)], check=True)
    got = H.f32_to_f16_trunc_np(bits.view(np.float32))
    ref = np.fromfile(gdir / "f32_to_f16_trunc.u16", dtype="<u2")
    nan = np.isnan(bits.view(np.float32))
    check("f32_to_f16 truncation == Rust float16.rs (200k random bit patterns + edge cases)",
          np.array_equal(got[~nan], ref[~nan]) and np.array_equal(got[nan] >> 10, ref[nan] >> 10),
          f"{int((got != ref).sum())} mismatches")
    for name in cases:
        _, m, k = name.split("_")
        m, k = int(m), int(k)
        w = np.fromfile(gdir / f"{name}.w.f32", dtype="<f4").reshape(m, k)
        wrot = H.fwht256(torch.from_numpy(w))
        d, z, c = H.rtn_symmetric(wrot.numpy())
        mine = H.pack_groups(d, z, c)
        rust = np.fromfile(gdir / f"{name}.groups.u8", dtype=np.uint8).reshape(-1, 136)
        check(f"symmetric qt44 RTN bit-identical to quantize_mq4g256v2_symmetric ({name})",
              np.array_equal(mine, rust), f"{int((mine != rust).any(axis=1).sum())}/{len(mine)} groups differ")


def test_rtn_vs_template(template, src_dir, source_st):
    """Python S=1 symmetric RTN must reproduce the Rust-encoded bytes of B.hfq: attention from the source
    safetensors (bf16 -> f32, as the quantizer ingests it) and expert GU/DN from the srcdump."""
    if not template or not (src_dir or source_st):
        skip("RTN parity vs Rust-encoded groups in B.hfq", "no --template with --source-st/--src-dir")
        return
    arch, meta, tindex = H.hfq_index(template)
    check("template is the symmetric padded layout", meta.get("hipfire_inkling2_mq4_layout") == "qt44-padded-awq-v1"
          and meta.get("mq4v2.symmetric") in (1, "1", True),
          f"layout={meta.get('hipfire_inkling2_mq4_layout')} symmetric={meta.get('mq4v2.symmetric')}")
    picks = []  # (label, name, loader)
    if source_st:
        from safetensors import safe_open
        handle = safe_open(source_st, "pt")
        for layer in (5, 6, 20):
            for slot in ("wq_du", "wk_dv", "wv_dv", "wo_ud", "wr_du", "gate_attn"):
                name = H.tname(layer, slot)
                picks.append((name, (lambda n=name: handle.get_tensor(n).float().numpy())))
    if src_dir:
        for layer in (5, 6, 20):
            try:
                src = H.SrcDump(src_dir, layer)
            except FileNotFoundError:
                continue
            for e in (0, 63, 127):
                for proj in ("gate_up_proj", "down_proj"):
                    name = H.ename(layer, e, proj)
                    picks.append((name, (lambda s=src, n=name: s.get(n))))
    check("template parity candidates found", len(picks) > 0)
    for name, load in picks:
        if name not in tindex:
            check(f"template contains {name}", False)
        w = load()
        kp = tindex[name][2][1]
        _, groups = H.b_variant_original(w, kp)
        raw = H.hfq_tensor_bytes(template, tindex[name]).reshape(-1, 136)
        check(f"RTN(S=1) bytes == template bytes: {name}", np.array_equal(groups, raw),
              f"{int((groups != raw).any(axis=1).sum())}/{len(raw)} groups differ; shape {w.shape} -> K_phys {kp}")


def test_rtn_numpy_reference():
    """Independent per-group python/numpy-float32 transcription of quantize_symmetric_half."""
    f32 = np.float32
    mults = [f32(1.0), np.frombuffer(np.uint32(0x3FA49249).tobytes(), dtype=np.float32)[0],
             np.frombuffer(np.uint32(0x3FDB6DB7).tobytes(), dtype=np.float32)[0], f32(2.0)]

    def half(values):
        amax = f32(0)
        for v in values:
            amax = max(amax, abs(v))
        if amax == 0:
            return 0, 0x8000, np.full(128, 8, np.uint8)
        base = (amax / f32(7.5)) * f32(0.5)
        best = (None, None, np.inf)
        for mu in mults:
            sb = int(H.f32_to_f16_trunc_np(np.array([base * mu], dtype=f32))[0])
            sc = H.f16_bits_to_f32(np.array([sb], dtype=np.uint16))[0]
            if sc == 0:
                continue
            zb = int(H.f32_to_f16_trunc_np(np.array([f32(-8) * sc], dtype=f32))[0])
            zr = H.f16_bits_to_f32(np.array([zb], dtype=np.uint16))[0]
            inv = f32(1) / sc
            mse = 0.0
            for v in values:
                c = np.floor((v - zr) * inv + f32(0.5))
                c = min(max(c, f32(0)), f32(15))
                fm = f32(float(c) * float(sc) + float(zr))
                e = v - fm
                mse += float(e) * float(e)
            if mse < best[2]:
                best = (sb, zb, mse)
        if best[0] is None or best[0] == 0:
            return 0, 0x8000, np.full(128, 8, np.uint8)
        sc = H.f16_bits_to_f32(np.array([best[0]], dtype=np.uint16))[0]
        zr = H.f16_bits_to_f32(np.array([best[1]], dtype=np.uint16))[0]
        inv = f32(1) / sc
        cs = np.array([min(max(np.floor((v - zr) * inv + f32(0.5)), f32(0)), f32(15)) for v in values], dtype=np.uint8)
        return best[0], best[1], cs

    w = adversarial_weights(8, 512, 21)
    wrot = H.fwht256(torch.from_numpy(w)).numpy()
    d, z, c = H.rtn_symmetric(wrot)
    bad = 0
    for r in range(8):
        for h in range(4):
            sb, zb, cs = half(wrot[r, h * 128:(h + 1) * 128])
            if sb != d[r, h] or zb != z[r, h] or not np.array_equal(cs, c[r, h * 128:(h + 1) * 128]):
                bad += 1
    check("numba RTN == independent numpy-f32 transcription (adversarial rows)", bad == 0, f"{bad} halves differ")
    sym = np.all((H.f16_bits_to_f32(z) == -8 * H.f16_bits_to_f32(d)) | (d == 0))
    zero_ok = np.all(z[d == 0] == 0x8000) and np.all(c.reshape(8, 4, 128)[d == 0] == 8)
    check("symmetric header: z == f16(-8d), d==0 -> z=0x8000 and code 8", sym and zero_ok and (d == 0).any())


def planted_problem(seed=5, m=32, k=512, n=300):
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((k, k // 4))
    x = rng.standard_normal((n, k // 4)) @ mix.T + 0.1 * rng.standard_normal((n, k))
    h = (x.T @ x) / n
    w = (rng.standard_normal((m, k)) * 0.05).astype(np.float32)
    return w, h


def test_gptq(factor_impls=("fast", "ref")):
    w, h = planted_problem()
    wt = torch.from_numpy(w)
    d, z, c = H.rtn_symmetric(w)
    hrot = torch.from_numpy(h).double()
    e_rtn = H.row_errors(wt, hrot, d, z, c).sum()
    outs = {}
    for impl in factor_impls:
        u, perm, damp, cond = H.factor_inverse_cholesky(hrot, 1.0, impl)
        outs[impl] = (u, perm, damp)
        codes = H.gptq(wt, d, z, u, perm)
        e_gptq = H.row_errors(wt, hrot, d, z, codes).sum()
        check(f"GPTQ lowers undamped E vs RTN on the planted problem [{impl}]", e_gptq < 0.85 * e_rtn,
              f"E_rtn={e_rtn:.4g} E_gptq={e_gptq:.4g} ratio={e_gptq / e_rtn:.3f}")
        check(f"GPTQ codes within 0..15 [{impl}]", codes.min() >= 0 and codes.max() <= 15)
    if len(outs) == 2:
        du = float((outs["fast"][0] - outs["ref"][0]).abs().max())
        check("fast factor (flipped Cholesky) == reference chain U", du < 1e-9 and
              np.array_equal(outs["fast"][1], outs["ref"][1]), f"max|dU|={du:.2e}")
    # identity H reduces to RTN codes
    eye = torch.eye(k := w.shape[1], dtype=torch.float64)
    u, perm, damp, _ = H.factor_inverse_cholesky(eye, 1.0, "fast")
    codes_i = H.gptq(wt, d, z, u, perm)
    check("identity H reduces GPTQ to RTN codes exactly", np.array_equal(codes_i, c),
          f"{int((codes_i != c).sum())} codes differ")
    # damping fallback: a rank-1 H needs damping; mean-diag damping must succeed
    v = torch.from_numpy(np.random.default_rng(6).standard_normal(k)).double()
    u, perm, damp, cond = H.factor_inverse_cholesky(torch.outer(v, v), 1.0, "fast")
    check("rank-1 H solves with d1.0 damping", math.isfinite(damp) and damp > 0, f"damp={damp:.4g}")
    # fail closed: all-zero H
    try:
        H.factor_inverse_cholesky(torch.zeros(k, k, dtype=torch.float64), 1.0, "fast")
        check("all-zero H fails closed", False)
    except H.FactorError:
        check("all-zero H fails closed", True)


def test_refit():
    rng = np.random.default_rng(421)
    m, k = 3, 256
    codes = rng.integers(0, 16, (m, k), dtype=np.uint8)
    codes[0, :128] = 8
    d = np.full((m, 2), .08, dtype=np.float32)
    d[0, 0] = 0
    d_bits = H.f32_to_f16_trunc_np(d)
    z_bits = H.f32_to_f16_trunc_np(-8 * d)
    z_bits[d_bits == 0] = 0x8000
    w = ((codes.astype(np.float64) - 8.) * .137 + rng.normal(0, .008, (m, k))).astype(np.float32)
    a = torch.from_numpy(rng.normal(size=(k, k)))
    h = a.T @ a + torch.eye(k, dtype=torch.float64) * 10
    nd, nz = H.half_refit(torch.from_numpy(w), h, d_bits, z_bits, codes)
    check("refit: zero half keeps d=0, z=0x8000", nd[0, 0] == 0 and nz[0, 0] == 0x8000)
    c = codes.astype(np.float64) - 8
    hn = h.numpy()
    worst = 0.0
    for row in range(m):
        for block in range(2):
            if row == 0 and block == 0:
                continue
            sl = slice(block * 128, (block + 1) * 128)
            hc = hn[sl, sl]
            exact = (w[row, sl].astype(np.float64) @ hc @ c[row, sl]) / (c[row, sl] @ hc @ c[row, sl])
            fitted = float(H.f16_bits_to_f32(nd[row:row + 1, block])[0])
            worst = max(worst, abs(fitted - exact))
            zz = H.f16_bits_to_f32(nz[row:row + 1, block])[0]
            if np.float16(zz) != np.float16(H.f16_bits_to_f32(H.f32_to_f16_trunc_np(np.float32(-8) * np.float32(fitted)).reshape(1))[0]):
                check("refit z == f16(-8d)", False)
    check("refit: 5 nonzero halves match closed-form Hessian LS after F16 truncation", worst < 2e-4, f"max err {worst:.2e}")


def test_padding_and_score(threads):
    """DN: S tail ones, pad AFTER prescale, last group mixes real/zero, Hrot against an explicit dense R."""
    rng = np.random.default_rng(31)
    kl, kp, m, n = 1152, 1280, 6, 2000
    x = rng.standard_normal((n, kl)).astype(np.float32) * np.exp(rng.normal(0, 0.7, kl)).astype(np.float32)
    h = ((x.T.astype(np.float64) @ x) / n).astype(np.float32)
    w = (rng.standard_normal((m, kl)) * 0.03).astype(np.float32)
    res = H.solve_core([w], h, n, k_phys=kp, alphas=(0.0, 0.35), rounds=1, label="pad", keff="skip")
    mem = res["members"][0]
    tail = mem["s16_bits"][kl:]
    check("DN S tail is 128 exact F16 ones (0x3C00)", len(tail) == 128 and np.all(tail == 0x3C00))
    moments = (np.diag(h.astype(np.float64)) * n).astype(np.float32)
    live = H.f32_to_f16_trunc_np(H.awq_scales(moments, res["log"]["alpha"]))
    check("DN S live part comes from the 1152 live moments only", np.array_equal(mem["s16_bits"][:kl], live))
    bad = H.awq_scales(np.concatenate((moments, np.zeros(128, np.float32))), 0.35)[:kl]
    check("(negative control) including pad zeros would change the live scales",
          not np.array_equal(H.f32_to_f16_trunc_np(bad), H.f32_to_f16_trunc_np(H.awq_scales(moments, 0.35))))
    check("DN last group's pad-side half is a real nonzero quantized half (d>0 for every row)",
          bool(np.all(mem["d_bits"][:, 9] > 0)) and bool(np.all(mem["d_bits"][:, 8] > 0)))
    # explicit dense check of Hrot and E
    s16 = H.f16_bits_to_f32(mem["s16_bits"])
    rb = np.zeros((kp, kp))
    r = H.rotation().numpy()
    for b in range(kp // 256):
        rb[b * 256:(b + 1) * 256, b * 256:(b + 1) * 256] = r
    hpad = np.zeros((kp, kp))
    hpad[:kl, :kl] = h.astype(np.float64)
    dinv = 1.0 / s16.astype(np.float64)
    hrot_ref = rb @ (dinv[:, None] * hpad * dinv[None, :]) @ rb.T
    hrot = H.rotated_hessian(torch.from_numpy(hpad), s16).numpy()
    check("blockwise Hrot == dense blockdiag(R) D^-1 H_pad D^-1 blockdiag(R)^T (pad rows/cols zero before transform)",
          np.abs(hrot - hrot_ref).max() < 1e-9 * np.abs(hrot_ref).max(), f"max diff {np.abs(hrot - hrot_ref).max():.2e}")
    s32 = np.concatenate((H.awq_scales(moments, res["log"]["alpha"]), np.ones(128, np.float32)))
    wp = np.zeros((m, kp), np.float32)
    wp[:, :kl] = w * s32[None, :kl]                      # prescale, THEN pad
    wrot = H.fwht256(torch.from_numpy(wp)).double().numpy()
    q = H.dequant_f32(mem["d_bits"], mem["z_bits"], mem["codes"]).double().numpy()
    e_dense = float((((wrot - q) @ hrot_ref) * (wrot - q)).sum())
    check("logged E_final == dense independent undamped E", abs(e_dense - res["log"]["E_final"]) <= 1e-7 * e_dense,
          f"dense {e_dense:.8g} vs logged {res['log']['E_final']:.8g}")
    check("E_final <= E(GPTQ) <= E(RTN S=1) for this group (GPTQ improves)",
          res["log"]["E_final"] <= res["log"]["E_gptq"] <= res["log"]["E_rtn_s1"],
          f"{res['log']['E_rtn_s1']:.4g} -> {res['log']['E_gptq']:.4g} -> {res['log']['E_final']:.4g}")
    # stacked siblings == separate solves at the same alpha (rows are independent given H)
    w2 = (rng.standard_normal((5, kl)) * 0.03).astype(np.float32)
    both = H.solve_core([w, w2], h, n, k_phys=kp, alphas=(0.35,), rounds=0, label="stk", keff="skip")
    one = H.solve_core([w2], h, n, k_phys=kp, alphas=(0.35,), rounds=0, label="one", keff="skip")
    check("Q-style row stacking: sibling codes identical to a separate solve",
          np.array_equal(both["members"][1]["codes"], one["members"][0]["codes"])
          and np.array_equal(both["members"][1]["s16_bits"], one["members"][0]["s16_bits"]))
    # fail closed
    for label, kwargs in (("N==0", dict(n=0)), ("non-finite H", dict(h=np.full_like(h, np.nan)))):
        try:
            H.solve_core([w], kwargs.get("h", h), kwargs.get("n", n), k_phys=kp, alphas=(0.0,), label="fc", keff="skip")
            check(f"fail closed: {label}", False)
        except H.GroupError as exc:
            check(f"fail closed: {label}", "fc" in str(exc), str(exc)[:80])


def test_record_roundtrip(work):
    rng = np.random.default_rng(41)
    m, k = 4, 512
    w = (rng.standard_normal((m, k)) * 0.02).astype(np.float32)
    w[1, :256] = 0.0  # a whole zero 256-group -> two d==0 halves (rotation mixes the 128-halves)
    d, z, c = H.rtn_symmetric(H.fwht256(torch.from_numpy(w)).numpy())
    check("fixture contains a d==0/z=0x8000 half", (d == 0).any() and np.all(z[d == 0] == 0x8000))
    s = H.f32_to_f16_trunc_np(np.exp(rng.normal(0, 0.3, k)).astype(np.float32))
    mem = {"s16_bits": s, "d_bits": d, "z_bits": z, "codes": c}
    path = Path(work) / "rt" / "model.test.weight.safetensors"
    H.save_record(path, "model.test.weight", mem, "ab" * 32)
    meta, s2, d2, z2, c2 = H.load_record(path)
    check("record metadata name/M/K/qt/source_sha", meta == {"name": "model.test.weight", "M": "4", "K": "512",
                                                             "qt": "44", "source_sha": "ab" * 32}, str(meta))
    check("record bit-exact round trip (S, d, z incl. signed zero, codes)",
          np.array_equal(s, s2) and np.array_equal(d, d2) and np.array_equal(z, z2) and np.array_equal(c, c2))
    from safetensors import safe_open
    with safe_open(str(path), framework="np") as f:
        shp = {key: (f.get_slice(key).get_shape(), f.get_slice(key).get_dtype()) for key in f.keys()}
    check("record tensor shapes/dtypes follow the contract",
          shp == {"S_f16": ([512], "F16"), "d_z_f16": ([4, 2, 2, 2], "F16"), "codes_u8": ([4, 512], "U8")}, str(shp))
    groups = H.pack_groups(d, z, c)
    d3, z3, c3 = H.unpack_groups(groups, m, k)
    check("pack/unpack of the 136 B group layout", np.array_equal(d, d3) and np.array_equal(z, z3) and np.array_equal(c, c3))


def test_hfhs_reader(work):
    rng = np.random.default_rng(51)
    a = rng.standard_normal((6, 6)).astype(np.float32)
    a = a @ a.T
    b = rng.standard_normal((9, 9)).astype(np.float32)
    p = Path(work) / "rt" / "L00.hfhs"
    p.parent.mkdir(parents=True, exist_ok=True)
    H.write_hfhs(p, [("model.llm.layers.0.attn.wq_du", 0, a), ("model.llm.layers.0.mlp.experts.3.down_proj", 0, b)])
    r = H.Hfhs(p)
    ok = np.array_equal(r.get("model.llm.layers.0.attn.wq_du"), a) and \
        np.array_equal(r.get("model.llm.layers.0.mlp.experts.3.down_proj"), b) and r.get("nope") is None
    r.close()
    check("HFHS v1 reader (header/name/expert/K/dtype/payload)", ok)


def test_scoring_invariance():
    """E is invariant to the AWQ scale in original coordinates: RTN at S=1 scored via Hrot == direct
    ||(W-What) x||^2 on the Hessian's own samples."""
    rng = np.random.default_rng(61)
    k, m, n = 512, 12, 700
    x = rng.standard_normal((n, k)).astype(np.float32) * np.exp(rng.normal(0, .6, k)).astype(np.float32)
    h = (x.T.astype(np.float64) @ x) / n
    w = (rng.standard_normal((m, k)) * 0.03).astype(np.float32)
    res = H.solve_core([w], h.astype(np.float32), n, alphas=(0.0, 0.35), rounds=1, label="inv", keff="skip")
    mem = res["members"][0]
    what = H.deq_to_original(mem["s16_bits"], mem["d_bits"], mem["z_bits"], mem["codes"]).double().numpy()
    direct = float(np.sum(((x.astype(np.float64) @ (w - what).T) ** 2)) / n)
    check("E (rotated, S-scaled, Hrot) == original-coordinate ||(W-What)x||^2/N (S16 vs S32 differ ~1e-3)",
          abs(direct - res["log"]["E_final"]) <= 2e-2 * direct, f"direct {direct:.6g} vs E {res['log']['E_final']:.6g}")


def test_e2e(work, threads):
    """Synthetic layer through `solve` and `heldout` (I/O path check; not a quality claim)."""
    import shutil
    root = Path(work) / "e2e"
    shutil.rmtree(root, ignore_errors=True)
    (root / "hfhs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "check").mkdir()
    layer, e = 5, 7
    rng = np.random.default_rng(71)
    spec = {H.tname(layer, "wq_du"): (512, 2304), H.tname(layer, "wr_du"): (16, 2304),
            H.tname(layer, "gate_attn"): (24, 2304), H.tname(layer, "wk_dv"): (256, 2304),
            H.tname(layer, "wv_dv"): (256, 2304), H.tname(layer, "wo_ud"): (2304, 3072),
            H.ename(layer, e, "gate_up_proj"): (2304, 2304), H.ename(layer, e, "down_proj"): (2304, 1152)}
    n = 3000
    entries, meta, src_entries, off, chunks = [], {}, [], 0, []
    xs = {}
    for key, kdim in (("q", 2304), ("k", 2304), ("v", 2304), ("o", 3072), ("gu", 2304), ("dn", 1152)):
        cs = np.exp(rng.normal(0, .7, kdim)).astype(np.float32)
        xs[key] = (rng.standard_normal((n, kdim)).astype(np.float32) * cs, cs)

    def h_of(key):
        x = xs[key][0]
        return ((x.T.astype(np.float64) @ x) / n).astype(np.float32)

    site_of = {"wq_du": "q", "wr_du": "q", "gate_attn": "q", "wk_dv": "k", "wv_dv": "v", "wo_ud": "o",
               "gate_up_proj": "gu", "down_proj": "dn"}
    hc = {}
    for name, (m, k) in spec.items():
        slot = name.split(".")[-2]
        key = site_of[slot]
        stem = name[:-7]
        if key not in hc:
            hc[key] = h_of(key)
        entries.append((stem, 0, hc[key]))
        meta[stem] = {"K": k, "N": n, "N_by_visit": {"0": n}}
        w = (rng.standard_normal((m, k)) * 0.03).astype(np.float32)
        src_entries.append({"name": name, "offset_bytes": off, "shape": [m, k], "numel": m * k})
        chunks.append(w.tobytes())
        off += w.nbytes
    H.write_hfhs(root / "hfhs" / "L05.hfhs", entries)
    (root / "hfhs" / "L05.json").write_text(json.dumps({"layer": 5, "names": meta}))
    (root / "src" / "L05.f32").write_bytes(b"".join(chunks))
    (root / "src" / "L05.json").write_text(json.dumps({
        "schema": "hipfire.inkling2.srcdump.v1", "layer": 5, "file": "L05.f32", "formats": {"w13": "nvfp4", "w2": "nvfp4"},
        "tensors": src_entries}))
    # check rows (hipfire.inkling2.check_rows.v1: one index per layer, sites -> file/K/rows/segments)
    sites = {}
    for site, key, kdim, nrow in (("q", "q", 2304, 96), ("gate_up", "gu", 2304, 40), ("down", "dn", 1152, 40)):
        x = (rng.standard_normal((nrow, kdim)).astype(np.float32) * xs[key][1])
        x.tofile(root / "check" / f"L05.{site}.f32")
        sites[site] = {"file": f"L05.{site}.f32", "K": kdim, "rows": nrow,
                       "segments": [{"visit": 0, "pass": 0, "expert": None if site == "q" else e, "row0": 0, "rows": nrow}]}
    (root / "check" / "L05.index.json").write_text(json.dumps({
        "schema": "hipfire.inkling2.check_rows.v1", "layer": 5, "sites": sites}))
    # a fake HFQ is not needed: pass the shipped A as an index-only template with an explicit sha override
    cmd = [sys.executable, str(Path(H.__file__)), "solve", "--hfhs-dir", str(root / "hfhs"), "--src-dir",
           str(root / "src"), "--template", str(root / "fake.hfq"), "--layers", "5", "--experts", str(e),
           "--out", str(root / "out"), "--workers", "2", "--threads", str(max(1, threads // 2)), "--keff", "skip",
           "--alphas", "0,0.35", "--source-sha", "cd" * 32]
    (root / "fake.hfq").write_bytes(b"x")
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    recs = sorted((root / "out" / "records").glob("*.safetensors"))
    check("e2e solve wrote one record per tensor (6 dense + GU + DN = 8)", len(recs) == 8, f"{len(recs)} records, "
          f"{time.monotonic() - t0:.0f}s")
    meta0, s0, d0, z0, c0 = H.load_record(root / "out" / "records" / (H.tname(layer, "wq_du") + ".safetensors"))
    _, s1, *_ = H.load_record(root / "out" / "records" / (H.tname(layer, "gate_attn") + ".safetensors"))
    check("Q siblings (wq_du / gate_attn) have byte-identical sidecars", np.array_equal(s0, s1))
    _, sdn, *_, cdn = H.load_record(root / "out" / "records" / (H.ename(layer, e, "down_proj") + ".safetensors"))
    check("e2e DN record is physical K 1280 with ones tail", cdn.shape == (2304, 1280) and np.all(sdn[1152:] == 0x3C00))
    lines = [json.loads(l) for l in (root / "out" / "log" / "L05.jsonl").read_text().splitlines()]
    need = {"alpha", "E_by_alpha", "E_rtn_s1", "refit", "N", "damp", "wall_seconds", "members"}
    check("e2e log has the required fields", len(lines) == 6 and all(need <= set(l) for l in lines),
          f"{len(lines)} groups")
    # resume: second invocation does nothing
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    check("e2e solve is resumable (0 groups to do)", "0 groups to do" in r.stdout, r.stdout.strip().splitlines()[0])
    # heldout
    hcmd = [sys.executable, str(Path(H.__file__)), "heldout", "--out", str(root / "out"), "--check-dir",
            str(root / "check"), "--hfhs-dir", str(root / "hfhs"), "--src-dir", str(root / "src"),
            "--template", str(root / "fake.hfq"), "--crosscheck", "0", "--source-sha", "cd" * 32]
    subprocess.run(hcmd, check=True, capture_output=True)
    held = json.loads((root / "out" / "heldout.json").read_text())
    ag = held["aggregate"]
    check("e2e heldout.json has layer/site/format aggregates", "layer05/Q" in ag and "layer05/GU" in ag
          and "format/nvfp4" in ag and "all" in ag, str(sorted(k for k in ag if not k.startswith("_"))))
    imp = ag["all"]["improvement_weighted"]
    check("e2e held-out: C improves over B on synthetic data", imp is not None and imp > 0, f"{imp:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--golden-bin")
    ap.add_argument("--work", default=str(Path(__file__).resolve().parents[2] / "h2" / "selftest"))
    ap.add_argument("--template")
    ap.add_argument("--src-dir")
    ap.add_argument("--source-st", help="source model.safetensors (bf16 attention) for the template parity test")
    ap.add_argument("--e2e", action="store_true")
    args = ap.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    H.init_threads(args.threads)
    t0 = time.monotonic()
    test_rotation()
    test_awq_law()
    test_rtn_numpy_reference()
    test_f16_and_rtn_vs_rust(args.golden_bin, args.work)
    test_rtn_vs_template(args.template, args.src_dir, args.source_st)
    test_refit()
    test_gptq()
    test_padding_and_score(args.threads)
    test_scoring_invariance()
    test_record_roundtrip(args.work)
    test_hfhs_reader(args.work)
    if args.e2e:
        test_e2e(args.work, args.threads)
    npass = sum(1 for r in RESULTS if r[1])
    nskip = sum(1 for r in RESULTS if r[1] is None)
    print(f"\nselftest: {npass} passed, {nskip} skipped, 0 failed in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
