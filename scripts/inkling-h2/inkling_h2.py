#!/usr/bin/env python3
"""Inkling-2-Mini (arch 24) `mq4-xts` H2 solver: per-producer-group AWQ alpha search, symmetric
qt44 RTN grid, Hessian GPTQ on the frozen grid and up to three d/z refit rounds. CPU only.

Developer tool (not a serving path). Port of the Qwen H2 recipe
(qat/v3/hessopt/hessopt.py + qat/v3/mi300x2/mi300x_torch_gptq_mqn.py) to Inkling's per-expert
Hessians. Contract: inkling/xts/CONTRACT.md (HFHS inputs, record outputs); recipe: MQ4XTS-PLAN.md.

  solve    --hfhs-dir X/calib/RUN/hfhs --src-dir X/calib/src --template B.hfq --layers 5,6,20 --out X/h2/RUN
  heldout  --out X/h2/RUN --check-dir X/calib/RUN/check --hfhs-dir ... --src-dir ... --template B.hfq
  bench    synthetic one-GU + one-DN throughput (no data needed)

Recipe facts fixed here (all frozen by the plan):
  * FWHT-256 seeds 42/1042, R = diag(post) H256 diag(pre) / 16 applied per 256-block of a scaled row.
  * S32 = AWQ law on diag(H)*N moments (geo-mean normalised in log space, clamp [1e-2, 1e2], same as
    calibration.rs compute_awq_scales); S16 = hipfire TRUNCATING f32->f16 (float16.rs::f32_to_f16, the
    serializer awq_scales_to_f16_bytes uses it; rx_quant.f32_to_f16_trunc_np is the same function).
  * W' = W * S32 (f32), DN zero-padded 1152 -> 1280 AFTER the prescale (S tail = ones).
  * Hrot = R D^-1 H D^-1 R^T, D = diag(S16) (f64, full K x K; DN H zero-padded before the transform).
  * grid = symmetric four-candidate RTN (mq_symmetric.rs::quantize_symmetric_half, bit exact).
  * GPTQ: stable descending diag(Hrot) act-order, 128-column lazy blocks, lambda = mean diag(Hrot)
    (x10 once on Cholesky failure, as the reference), codes clamp 0..15, written in original order.
    Deviation from the reference (documented): a zero-scale half emits code 8 (Rust RTN / contract
    convention: code 8 reconstructs 0) instead of 0; the dequantised value is identical (0).
  * alpha = argmin of undamped E = tr((W'rot - Q) Hrot (W'rot - Q)^T) summed over group members;
    refit rounds d = (c^T H_half w)/(c^T H_half c), c = q-8, d>=0, F16 d/z with z=f16(-8d) (d==0 -> 0x8000),
    codes re-solved on the frozen grid with the cached factor; last strictly improving round is kept.
"""
import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

# numba/torch thread pools are sized explicitly in `init_threads`; keep MKL/OMP from oversubscribing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numba as nb  # noqa: E402
import torch  # noqa: E402
from numba import njit, prange  # noqa: E402


ALPHAS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55)
SEED_PRE, SEED_POST = 42, 1042
GROUP_BYTES = 136
DOWN_LIVE, DOWN_PHYS = 1152, 1280
REFIT_ROUNDS = 3
DEFAULT_FORMATS = {5: "nvfp4", 6: "nvfp4", 20: "mxfp8"}

DENSE_GROUPS = {
    "Q": ("wq_du", "wr_du", "gate_attn"),
    "K": ("wk_dv",),
    "V": ("wv_dv",),
    "O": ("wo_ud",),
}
# H is taken from the first member of the group (the Q siblings share their input rows).
SITE_ALIASES = {
    "q": "Q", "k": "K", "v": "V", "o": "O", "attnout": "O", "attn_out": "O", "wo": "O", "wo_ud": "O",
    "gu": "GU", "gateup": "GU", "gate_up": "GU", "expertgateup": "GU", "gate_up_proj": "GU", "expert_gate_up": "GU",
    "dn": "DN", "down": "DN", "expertdown": "DN", "down_proj": "DN", "expert_down": "DN",
    "wq_du": "Q", "wk_dv": "K", "wv_dv": "V", "wo_ud": "O",
}


def init_threads(threads):
    torch.set_num_threads(threads)
    nb.set_num_threads(threads)


# ----------------------------------------------------------------------------------------------
# F16 truncation (hipfire float16::f32_to_f16), signs, rotation
# ----------------------------------------------------------------------------------------------
def f32_to_f16_trunc_np(values):
    """hipfire float16::f32_to_f16 (truncating) on a float32 array -> uint16 bits (== rx_quant's)."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    sign = (bits >> 31) & 1
    exp = ((bits >> 23) & 0xFF).astype(np.int64)
    frac = bits & 0x7FFFFF
    out = np.zeros(bits.shape, dtype=np.uint32)
    new_exp = exp - 127 + 15
    normal = (exp != 0xFF) & (new_exp > 0) & (new_exp < 31)
    out[normal] = (sign[normal] << 15) | (new_exp[normal].astype(np.uint32) << 10) | (frac[normal] >> 13)
    over = (exp != 0xFF) & (new_exp >= 31)
    out[over] = (sign[over] << 15) | (0x1F << 10)
    sub = (exp != 0xFF) & (new_exp <= 0) & (new_exp >= -10)
    f = frac[sub] | 0x800000
    shift = (1 - new_exp[sub] + 13).astype(np.uint32)
    out[sub] = (sign[sub] << 15) | (f >> shift)
    tiny = (exp != 0xFF) & (new_exp < -10)
    out[tiny] = sign[tiny] << 15
    nan = exp == 0xFF
    out[nan] = (sign[nan] << 15) | (0x1F << 10) | np.where(frac[nan] == 0, 0, (frac[nan] >> 13) | 1)
    return out.astype(np.uint16)


def f16_bits_to_f32(bits):
    return np.ascontiguousarray(bits, dtype="<u2").view("<f2").astype(np.float32)


def gen_fwht_signs(seed, n=256):
    out = np.empty(n, dtype=np.float32)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if (state >> 16) & 1 else -1.0
    return out


_SIGNS = {}


def signs():
    if not _SIGNS:
        _SIGNS["pre"] = gen_fwht_signs(SEED_PRE)
        _SIGNS["post"] = gen_fwht_signs(SEED_POST)
    return _SIGNS["pre"], _SIGNS["post"]


_ROT = {}


def rotation():
    """R (256x256, f64): rotated block = R @ block. R = diag(post) H256 diag(pre) / 16 (orthogonal)."""
    if "r" not in _ROT:
        had = np.array([[1]], dtype=np.int8)
        for _ in range(8):
            had = np.block([[had, had], [had, -had]])
        pre, post = signs()
        r = (post[:, None].astype(np.float64) * had * pre[None, :].astype(np.float64)) / 16.0
        _ROT["r"] = torch.from_numpy(np.ascontiguousarray(r))
    return _ROT["r"]


def fwht256(x, inverse=False):
    """FWHT-256 over consecutive 256-groups of the last dim; float32 stage-by-stage exactly as
    quant_fwht.rs::cpu_fwht_256 (signs1, butterflies stride 1..128, scale*signs2). x: f32 [.., K]."""
    assert x.dtype == torch.float32 and x.shape[-1] % 256 == 0
    pre, post = signs()
    if inverse:
        pre, post = post, pre
    pre_t = torch.from_numpy(pre)
    post_t = torch.from_numpy((post * np.float32(0.0625)).astype(np.float32))
    shape = x.shape
    y = x.reshape(-1, 256) * pre_t
    h = 1
    while h < 256:
        y = y.reshape(-1, 256 // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (y.reshape(-1, 256) * post_t).reshape(shape).contiguous()


# ----------------------------------------------------------------------------------------------
# AWQ scale law (calibration.rs::compute_awq_scales)
# ----------------------------------------------------------------------------------------------
def awq_scales(in_sum2, alpha):
    """Port of compute_awq_scales: moments -> geo-mean-normalised v^(alpha/2), clamp [1e-2, 1e2], f32."""
    v = np.asarray(in_sum2, dtype=np.float32).astype(np.float64)
    v = np.where(np.isnan(v), 1e-12, np.minimum(np.maximum(v, 1e-12), 1e30))
    ls = (float(np.float32(alpha)) * 0.5) * np.log(v)
    mean = ls.sum() / len(ls)
    out = np.exp(ls - mean).astype(np.float32)
    return np.clip(out, np.float32(1e-2), np.float32(1e2)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# numba kernels: symmetric RTN (bit-exact port of quantize_symmetric_half) and GPTQ block
# ----------------------------------------------------------------------------------------------
_MULT_BITS = (0x3F800000, 0x3FA49249, 0x3FDB6DB7, 0x40000000)  # 1.0, Rust consts, 2.0
_MULTS = tuple(struct.unpack("<f", struct.pack("<I", b))[0] for b in _MULT_BITS)


@njit(cache=True, inline="always")
def _trunc_f16_bits(x, buf, bufi):
    buf[0] = x
    bits = np.int64(bufi[0])
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        f = 0
        if frac != 0:
            f = (frac >> 13) | 1
        return (sign << 15) | (0x1F << 10) | f
    new_exp = exp - 127 + 15
    if new_exp >= 31:
        return (sign << 15) | (0x1F << 10)
    if new_exp <= 0:
        if new_exp < -10:
            return sign << 15
        f = frac | 0x800000
        shift = 1 - new_exp + 13
        return (sign << 15) | (f >> shift)
    return (sign << 15) | (new_exp << 10) | (frac >> 13)


@njit(cache=True, inline="always")
def _f16_to_f32(b):
    s = (b >> 15) & 1
    e = (b >> 10) & 31
    f = b & 1023
    if e == 0:
        v = f * 5.960464477539063e-08  # 2^-24, exact in f64
    elif e == 31:
        if f == 0:
            v = np.inf
        else:
            v = np.nan
    else:
        v = (1.0 + f / 1024.0) * 2.0 ** (e - 15)
    if s == 1:
        v = -v
    return np.float32(v)


@njit(cache=True)
def _rtn_half(vals, codes_out, m0, m1, m2, m3):
    """quantize_symmetric_half(values[128], max_code=15) -> (scale_bits, zero_bits), fills codes_out."""
    buf = np.empty(1, dtype=np.float32)
    bufi = buf.view(np.uint32)
    f0 = np.float32(0.0)
    half = np.float32(0.5)
    amax = f0
    for i in range(128):
        a = np.abs(vals[i])
        if a > amax:
            amax = a
    if amax == f0:
        for i in range(128):
            codes_out[i] = 8
        return 0, 0x8000
    base = (amax / np.float32(7.5)) * half
    best_bits = 0
    best_zero = 0
    best_mse = np.inf
    for mi in range(4):
        if mi == 0:
            mult = m0
        elif mi == 1:
            mult = m1
        elif mi == 2:
            mult = m2
        else:
            mult = m3
        sbits = _trunc_f16_bits(base * mult, buf, bufi)
        scale = _f16_to_f32(sbits)
        if scale == f0:
            continue
        zbits = _trunc_f16_bits(np.float32(-8.0) * scale, buf, bufi)
        zero = _f16_to_f32(zbits)
        inv = np.float32(1.0) / scale
        mse = 0.0
        for i in range(128):
            c = np.floor((vals[i] - zero) * inv + half)
            if c < f0:
                c = f0
            if c > np.float32(15.0):
                c = np.float32(15.0)
            # f32 mul_add(c, scale, zero): operands are tiny-mantissa values, f64 evaluation is exact
            fm = np.float32(np.float64(c) * np.float64(scale) + np.float64(zero))
            err = vals[i] - fm
            mse += np.float64(err) * np.float64(err)
        if mse < best_mse:
            best_mse = mse
            best_bits = sbits
            best_zero = zbits
    if best_bits == 0:
        for i in range(128):
            codes_out[i] = 8
        return 0, 0x8000
    scale = _f16_to_f32(best_bits)
    zero = _f16_to_f32(best_zero)
    inv = np.float32(1.0) / scale
    for i in range(128):
        c = np.floor((vals[i] - zero) * inv + half)
        if c < f0:
            c = f0
        if c > np.float32(15.0):
            c = np.float32(15.0)
        codes_out[i] = np.uint8(c)
    return best_bits, best_zero


@njit(cache=True, parallel=True)
def _rtn_kernel(w, dbits, zbits, codes, m0, m1, m2, m3):
    rows, k = w.shape
    nh = k // 128
    for idx in prange(rows * nh):
        r = idx // nh
        h = idx % nh
        s, z = _rtn_half(w[r, h * 128:(h + 1) * 128], codes[r, h * 128:(h + 1) * 128], m0, m1, m2, m3)
        dbits[r, h] = s
        zbits[r, h] = z


def rtn_symmetric(wrot):
    """wrot: f32 [M,K] numpy (K % 256 == 0) -> (d_bits u16 [M,K/128], z_bits u16, codes u8 [M,K])."""
    wrot = np.ascontiguousarray(wrot, dtype=np.float32)
    rows, k = wrot.shape
    assert k % 256 == 0
    d = np.empty((rows, k // 128), dtype=np.uint16)
    z = np.empty((rows, k // 128), dtype=np.uint16)
    c = np.empty((rows, k), dtype=np.uint8)
    _rtn_kernel(wrot, d, z, c, *(np.float32(m) for m in _MULTS))
    return d, z, c


@njit(cache=True, parallel=True)
def _gptq_block(res, dh, zh, u, perm, start, end, errs, codes):
    """Sequential GPTQ inside one lazy block, parallel over (independent) rows.
    res: f64 [M,K] permuted residual (updated in cols start..end); dh/zh: f32 [M,K/128] original-half
    scale/zero; u: f64 [K,K] upper inverse-Cholesky; errs: f64 [M,end-start]; codes: u8 [M,K] orig order."""
    rows = res.shape[0]
    f0 = np.float32(0.0)
    for r in prange(rows):
        for j in range(start, end):
            pj = perm[j]
            hh = pj // 128
            scale = dh[r, hh]
            zero = zh[r, hh]
            val = np.float32(res[r, j])
            if scale == f0:
                q = np.float32(8.0)
            else:
                q = np.floor((val - zero) * (np.float32(1.0) / scale) + np.float32(0.5))
                if q < f0:
                    q = f0
                if q > np.float32(15.0):
                    q = np.float32(15.0)
            qv = np.float64(np.float32(np.float64(q) * np.float64(scale) + np.float64(zero)))
            ujj = u[j, j]
            err = (res[r, j] - qv) / ujj
            errs[r, j - start] = err
            codes[r, pj] = np.uint8(q)
            for t in range(j + 1, end):
                res[r, t] -= err * u[j, t]


# ----------------------------------------------------------------------------------------------
# Hessian transform, factor, GPTQ, score, refit
# ----------------------------------------------------------------------------------------------
def rotated_hessian(h64, s16):
    """Hrot = R D^-1 H D^-1 R^T blockwise (full K x K), D = diag(S16). h64: f64 [K,K] torch; s16 f32 np."""
    k = h64.shape[0]
    s = torch.from_numpy(np.ascontiguousarray(s16, dtype=np.float32)).double()
    h = (h64 / s[:, None]) / s[None, :]
    r = rotation()
    blocks = k // 256
    h = torch.matmul(h.reshape(k, blocks, 256), r.T).reshape(k, k)
    h = torch.matmul(r, h.reshape(blocks, 256, k)).reshape(k, k)
    return ((h + h.T) * 0.5).contiguous()


class FactorError(RuntimeError):
    pass


def factor_inverse_cholesky(hrot, damp_fraction=1.0, impl="fast"):
    """Act-order (stable descending diag) + damped upper Cholesky of H^-1, with the reference damping
    fallback (x10 on failure up to 10*mean diag). Returns (u f64 [K,K] torch, perm np, damp, cond_est).
    impl='ref' is the reference chain (cholesky, solve_triangular(I), L^-T L^-1, cholesky(Hinv).T);
    impl='fast' is the same matrix from one Cholesky of the flipped H and one triangular solve:
    H = V V^T with V upper = P L P (P = flip), so H^-1 = U^T U with U = V^-1 = P L^-1 P."""
    k = hrot.shape[0]
    diag = hrot.diagonal().cpu().numpy()
    if not np.all(np.isfinite(diag)):
        raise FactorError("non-finite Hrot diagonal")
    perm = np.argsort(-diag, kind="stable").copy()
    p = torch.as_tensor(perm)
    h = hrot.index_select(0, p).index_select(1, p).contiguous()
    mean = float(h.diagonal().mean().item())
    if not mean > 0.0:
        raise FactorError("mean diag(Hrot) <= 0 (no activation energy)")
    damp = mean * damp_fraction
    max_damp = mean * 10.0
    eye = torch.eye(k, dtype=torch.float64)
    while True:
        try:
            hd = h + damp * eye
            if impl == "ref":
                l = torch.linalg.cholesky(hd)
                li = torch.linalg.solve_triangular(l, eye, upper=False)
                hinv = li.T @ li
                u = torch.linalg.cholesky(hinv).T.contiguous()
                ldiag = l.diagonal()
            else:
                l = torch.linalg.cholesky(hd.flip(0).flip(1))
                li = torch.linalg.solve_triangular(l, eye, upper=False)
                u = li.flip(0).flip(1).contiguous()
                ldiag = l.diagonal()
            break
        except torch.linalg.LinAlgError:
            damp = max_damp * 1e-9 if damp == 0 else damp * 10
            if damp > max_damp:
                raise FactorError(f"Cholesky failed after damping fallback (final damp {damp:.6g})")
    cond = float((ldiag.max() / ldiag.min()).item() ** 2)
    return u, perm, damp, cond


def gptq(wrot, d_bits, z_bits, u, perm, block=128):
    """Frozen-grid GPTQ. wrot: f32 torch [M,K]; d_bits/z_bits: u16 np [M,K/128]; u/perm from the factor.
    Returns codes u8 np [M,K] in original column order."""
    m, k = wrot.shape
    p = torch.as_tensor(perm)
    res_t = wrot.double().index_select(1, p).contiguous()
    res = res_t.numpy()
    dh = f16_bits_to_f32(d_bits)
    zh = f16_bits_to_f32(z_bits)
    codes = np.empty((m, k), dtype=np.uint8)
    un = u.numpy()
    errs_t = torch.empty((m, block), dtype=torch.float64)
    errs = errs_t.numpy()
    for start in range(0, k, block):
        end = min(start + block, k)
        _gptq_block(res, dh, zh, un, perm, start, end, errs, codes)
        if end < k:
            res_t[:, end:].addmm_(errs_t[:, : end - start], u[start:end, end:], beta=1.0, alpha=-1.0)
    if not np.all(np.isfinite(res)):
        raise FactorError("non-finite residual after GPTQ")
    return codes


def dequant_f32(d_bits, z_bits, codes):
    """Rotated-domain reconstruction z + c*d (f32), torch [M,K]."""
    d = torch.from_numpy(f16_bits_to_f32(d_bits)).repeat_interleave(128, dim=1)
    z = torch.from_numpy(f16_bits_to_f32(z_bits)).repeat_interleave(128, dim=1)
    c = torch.from_numpy(codes.astype(np.float32))
    return torch.addcmul(z, c, d)


def row_errors(wrot, hrot, d_bits, z_bits, codes):
    """Undamped per-row E_r = (w-q) Hrot (w-q)^T (f64), numpy [M]."""
    diff = wrot.double() - dequant_f32(d_bits, z_bits, codes).double()
    return torch.sum((diff @ hrot) * diff, dim=1).numpy()


def half_refit(wrot, hrot, d_bits, z_bits, codes):
    """d = (c^T H_half w)/(c^T H_half c) per row and 128-half (c = q-8, d>=0, F16 d/z, signed zero for d==0).
    Returns (d_bits, z_bits) u16 [M,K/128]."""
    m, k = wrot.shape
    blocks = k // 128
    c = torch.from_numpy(codes.astype(np.float64)).reshape(m, blocks, 128) - 8.0
    w = wrot.double().reshape(m, blocks, 128)
    hb = torch.stack([hrot[j * 128:(j + 1) * 128, j * 128:(j + 1) * 128] for j in range(blocks)])
    hc = torch.einsum("bij,mbj->mbi", hb, c)
    num = (w * hc).sum(-1)
    den = (c * hc).sum(-1)
    old_d = f16_bits_to_f32(d_bits)
    values = torch.where(den > 0, num / den, torch.from_numpy(old_d.astype(np.float64))).float().numpy()
    values = np.maximum(values, 0.0)
    values[old_d == 0] = 0.0
    nd = f32_to_f16_trunc_np(values)
    d = f16_bits_to_f32(nd)
    nz = f32_to_f16_trunc_np(np.float32(-8.0) * d)
    nz[nd == 0] = np.uint16(0x8000)
    return nd, nz


# ----------------------------------------------------------------------------------------------
# Group solve (pure compute; I/O free)
# ----------------------------------------------------------------------------------------------
class GroupError(RuntimeError):
    pass


def pack_groups(d_bits, z_bits, codes):
    """-> u8 [M*K/256, 136]: [d0 z0 d1 z1 (LE u16)] + 128 B nibbles (lo = code[2i])."""
    m, k = codes.shape
    g = k // 256
    hdr = np.stack((d_bits.reshape(m, g, 2), z_bits.reshape(m, g, 2)), axis=-1)  # [m,g,half,(d,z)]
    hdr = np.ascontiguousarray(hdr.astype("<u2")).view(np.uint8).reshape(m * g, 8)
    cc = codes.reshape(m * g, 256)
    nib = (cc[:, 0::2] | (cc[:, 1::2] << 4)).astype(np.uint8)
    return np.concatenate((hdr, nib), axis=1)


def unpack_groups(groups, m, k):
    """Inverse of pack_groups -> (d_bits, z_bits, codes)."""
    g = k // 256
    groups = np.asarray(groups, dtype=np.uint8).reshape(m * g, GROUP_BYTES)
    hdr = np.ascontiguousarray(groups[:, :8]).view("<u2").reshape(m, g, 2, 2)
    nib = groups[:, 8:]
    codes = np.stack((nib & 15, nib >> 4), axis=-1).reshape(m, k)
    return hdr[..., 0].reshape(m, k // 128).copy(), hdr[..., 1].reshape(m, k // 128).copy(), codes.astype(np.uint8)


def solve_core(ws, h_live, n, *, k_phys=None, alphas=ALPHAS, rounds=REFIT_ROUNDS, label="group",
               member_names=None, keff="eig", factor_impl="fast", progress=None):
    """Solve one producer group.

    ws: list of f32 [M_i, K_live] weights sharing one input (one H, one alpha, one S).
    h_live: f32/f64 [K_live,K_live] H = G/N. n: sample count N. k_phys: K_live (default) or 1280 for DN.
    Returns dict(members=[{s16,d_bits,z_bits,codes,E}], log=...). Raises GroupError naming `label`."""
    t_start = time.monotonic()
    names = member_names or [f"{label}#{i}" for i in range(len(ws))]
    kl = h_live.shape[0]
    kp = k_phys or kl
    if h_live.shape != (kl, kl) or any(w.shape[1] != kl for w in ws):
        raise GroupError(f"{label}: shape mismatch H {h_live.shape} vs W {[w.shape for w in ws]}")
    if kp % 256 or kp < kl:
        raise GroupError(f"{label}: bad physical K {kp} for logical {kl}")
    if not n or n <= 0:
        raise GroupError(f"{label}: N == {n}")
    if not np.all(np.isfinite(h_live)):
        raise GroupError(f"{label}: non-finite Hessian")
    for w, nm in zip(ws, names):
        if not np.all(np.isfinite(w)):
            raise GroupError(f"{nm}: non-finite source weights")
    rows = [w.shape[0] for w in ws]
    bounds = np.cumsum([0] + rows)
    w_all = np.zeros((int(bounds[-1]), kp), dtype=np.float32)
    for i, w in enumerate(ws):
        w_all[bounds[i]:bounds[i + 1], :kl] = w
    h64 = torch.zeros((kp, kp), dtype=torch.float64)
    h64[:kl, :kl] = torch.from_numpy(np.ascontiguousarray(h_live, dtype=np.float32)).double()
    diag = np.diag(np.asarray(h_live, dtype=np.float64))
    if (diag < 0).any():
        raise GroupError(f"{label}: negative Hessian diagonal")
    moments = (diag * float(n)).astype(np.float32)  # == imatrix in_sum2 (sum of x^2); N is absorbed by geo-mean

    log = {"label": label, "N": int(n), "K_live": kl, "K_phys": kp, "M": [int(r) for r in rows],
           "E_by_alpha": {}, "E_rtn_s1": None}
    t_keff = 0.0
    if keff == "eig":
        t0 = time.monotonic()
        ev = torch.linalg.eigvalsh(torch.from_numpy(np.ascontiguousarray(h_live, dtype=np.float64)))
        evmax = float(ev[-1])
        log["K_eff"] = int((ev > 1e-6 * evmax).sum().item())
        pos = ev[ev > 0]
        log["cond_eig"] = float(evmax / float(pos[0])) if len(pos) else math.inf
        t_keff = time.monotonic() - t0
    log["keff_seconds"] = round(t_keff, 3)

    best = None
    e_rtn_s1 = None
    t_alpha = []
    for alpha in alphas:
        t0 = time.monotonic()
        s32_live = awq_scales(moments, alpha)
        s32 = np.concatenate((s32_live, np.ones(kp - kl, dtype=np.float32)))
        s16_bits = f32_to_f16_trunc_np(s32)
        s16 = f16_bits_to_f32(s16_bits)
        wrot = fwht256(torch.from_numpy(w_all * s32[None, :]))
        d_bits, z_bits, rtn_codes = rtn_symmetric(wrot.numpy())
        hrot = rotated_hessian(h64, s16)
        if alpha == 0.0:
            e_rtn_s1 = float(row_errors(wrot, hrot, d_bits, z_bits, rtn_codes).sum())
        try:
            u, perm, damp, cond = factor_inverse_cholesky(hrot, 1.0, factor_impl)
            codes = gptq(wrot, d_bits, z_bits, u, perm)
        except FactorError as exc:
            raise GroupError(f"{names[0]}: {exc}") from exc
        row_e = row_errors(wrot, hrot, d_bits, z_bits, codes)
        e_members = [float(row_e[bounds[i]:bounds[i + 1]].sum()) for i in range(len(ws))]
        e_total = sum(e_members)
        if not math.isfinite(e_total):
            raise GroupError(f"{names[0]}: non-finite E at alpha {alpha}")
        log["E_by_alpha"][f"{alpha:g}"] = e_total
        t_alpha.append(time.monotonic() - t0)
        if progress:
            progress(label, alpha, e_total, t_alpha[-1])
        if best is None or e_total < best["E"]:
            best = dict(alpha=alpha, E=e_total, e_members=e_members, s16_bits=s16_bits, wrot=wrot, hrot=hrot,
                        d_bits=d_bits, z_bits=z_bits, codes=codes, u=u, perm=perm, damp=damp, cond=cond)
    if e_rtn_s1 is None:  # alpha 0 not in the grid: score S=1 RTN separately
        s16 = np.ones(kp, dtype=np.float32)
        wrot1 = fwht256(torch.from_numpy(w_all))
        d1, z1, c1 = rtn_symmetric(wrot1.numpy())
        e_rtn_s1 = float(row_errors(wrot1, rotated_hessian(h64, s16), d1, z1, c1).sum())
    log["E_rtn_s1"] = e_rtn_s1
    log["alpha"] = best["alpha"]
    log["E_gptq"] = best["E"]
    log["damp"] = best["damp"]
    log["cond_chol_est"] = best["cond"]
    log["alpha_seconds"] = [round(t, 3) for t in t_alpha]

    # refit rounds on the cached factor / Hrot (frozen S, perm, u)
    d_bits, z_bits, codes = best["d_bits"], best["z_bits"], best["codes"]
    e_cur, e_members = best["E"], best["e_members"]
    log["refit"] = []
    t0 = time.monotonic()
    for rnd in range(1, rounds + 1):
        nd, nz = half_refit(best["wrot"], best["hrot"], d_bits, z_bits, codes)
        ncodes = gptq(best["wrot"], nd, nz, best["u"], best["perm"])
        row_e = row_errors(best["wrot"], best["hrot"], nd, nz, ncodes)
        em = [float(row_e[bounds[i]:bounds[i + 1]].sum()) for i in range(len(ws))]
        e_new = sum(em)
        ok = bool(e_new < e_cur)
        log["refit"].append({"round": rnd, "E_before": e_cur, "E_after": e_new, "accepted": ok})
        if not ok:
            break
        d_bits, z_bits, codes, e_cur, e_members = nd, nz, ncodes, e_new, em
    log["refit_seconds"] = round(time.monotonic() - t0, 3)
    log["E_final"] = e_cur
    log["rounds_accepted"] = sum(1 for r in log["refit"] if r["accepted"])
    log["wall_seconds"] = round(time.monotonic() - t_start, 3)

    members = []
    for i in range(len(ws)):
        sl = slice(bounds[i], bounds[i + 1])
        members.append({"name": names[i], "s16_bits": best["s16_bits"], "d_bits": d_bits[sl], "z_bits": z_bits[sl],
                        "codes": codes[sl], "E": e_members[i]})
    log["members"] = {m["name"]: m["E"] for m in members}
    return {"members": members, "log": log}


# ----------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------
def record_arrays(member, k):
    m = member["codes"].shape[0]
    g = k // 256
    dz = np.stack((member["d_bits"].reshape(m, g, 2), member["z_bits"].reshape(m, g, 2)), axis=-1)
    return {
        "S_f16": np.ascontiguousarray(member["s16_bits"].astype("<u2")).view("<f2"),
        "d_z_f16": np.ascontiguousarray(dz.astype("<u2")).view("<f2").reshape(m, g, 2, 2),
        "codes_u8": np.ascontiguousarray(member["codes"]),
    }


def save_record(path, name, member, source_sha):
    from safetensors.numpy import save_file
    k = member["codes"].shape[1]
    m = member["codes"].shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.part")
    save_file(record_arrays(member, k), str(tmp),
              metadata={"name": name, "M": str(m), "K": str(k), "qt": "44", "source_sha": source_sha})
    os.replace(tmp, path)


def load_record(path):
    """-> (meta, s_bits u16 [K], d_bits u16 [M,K/128], z_bits u16, codes u8 [M,K])."""
    from safetensors import safe_open
    with safe_open(str(path), framework="np") as f:
        meta = f.metadata()
        s = f.get_tensor("S_f16")
        dz = f.get_tensor("d_z_f16")
        codes = f.get_tensor("codes_u8")
    assert s.dtype == np.float16 and dz.dtype == np.float16 and codes.dtype == np.uint8
    m, g = dz.shape[0], dz.shape[1]
    dzb = np.ascontiguousarray(dz).view("<u2").reshape(m, g, 2, 2)
    return (meta, np.ascontiguousarray(s).view("<u2").copy(),
            dzb[..., 0].reshape(m, g * 2).copy(), dzb[..., 1].reshape(m, g * 2).copy(), codes)


# ----------------------------------------------------------------------------------------------
# Inputs: HFHS, source dump, HFQ template
# ----------------------------------------------------------------------------------------------
class Hfhs:
    """Pure-python HFHS v1 reader (hessian_io.rs): header 24 B (magic, version u32, count u64, reserved u64),
    then per tensor: name_len u32, name, expert_idx u32, K u32, dtype u32 (1=F32, 2=F64), K*K payload."""

    def __init__(self, path):
        self.path = str(path)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self.mm
        if len(mm) < 24 or mm[:4] != b"HFHS":
            raise ValueError(f"{path}: not an HFHS file")
        version, count, _ = struct.unpack_from("<IQQ", mm, 4)
        if version != 1:
            raise ValueError(f"{path}: HFHS version {version}")
        pos = 24
        self.index = {}
        for _ in range(count):
            (nl,) = struct.unpack_from("<I", mm, pos)
            pos += 4
            name = bytes(mm[pos:pos + nl]).decode()
            pos += nl
            expert, k, dtype = struct.unpack_from("<III", mm, pos)
            pos += 12
            if dtype not in (1, 2):
                raise ValueError(f"{path}: {name}: unknown HFHS dtype {dtype}")
            size = k * k * (4 if dtype == 1 else 8)
            if pos + size > len(mm):
                raise ValueError(f"{path}: {name}: truncated payload")
            key = (name, expert)
            if key in self.index:
                raise ValueError(f"{path}: duplicate HFHS entry {name}")
            self.index[key] = (k, dtype, pos)
            pos += size

    def names(self):
        return [n for (n, _) in self.index]

    def get(self, name, expert=0):
        """Full K x K Hessian as f32 (F64 payloads are cast) or None when absent."""
        item = self.index.get((name, expert))
        if item is None:
            return None
        k, dtype, pos = item
        arr = np.frombuffer(self.mm, dtype="<f4" if dtype == 1 else "<f8", count=k * k, offset=pos)
        return arr.reshape(k, k).astype(np.float32, copy=True)

    def close(self):
        self.mm.close()
        self.f.close()


def write_hfhs(path, entries):
    """Test helper: entries = [(name, expert_idx, f32 [K,K])]."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIQQ", b"HFHS", 1, len(entries), 0))
        for name, expert, h in entries:
            nb_ = name.encode()
            f.write(struct.pack("<I", len(nb_)) + nb_ + struct.pack("<III", expert, h.shape[0], 1))
            f.write(np.ascontiguousarray(h, dtype="<f4").tobytes())


def _dig(js, name):
    """Find per-tensor entry `name` (with or without .weight) in a layer JSON (flat or nested one level)."""
    alt = name[:-7] if name.endswith(".weight") else name + ".weight"
    for cand in (name, alt):
        if cand in js and isinstance(js[cand], dict):
            return js[cand]
    for key in ("tensors", "entries", "names", "hessians"):
        sub = js.get(key)
        if isinstance(sub, dict):
            for cand in (name, alt):
                if cand in sub and isinstance(sub[cand], dict):
                    return sub[cand]
        elif isinstance(sub, list):
            for item in sub:
                if isinstance(item, dict) and item.get("name") in (name, alt):
                    return item
    return None


class LayerHessians:
    def __init__(self, hfhs_dir, layer):
        self.layer = layer
        self.hfhs = Hfhs(Path(hfhs_dir) / f"L{layer:02d}.hfhs")
        jpath = Path(hfhs_dir) / f"L{layer:02d}.json"
        self.js = json.loads(jpath.read_text())

    def fetch(self, tensor_name):
        """(H f32 [K,K], N, K) for a tensor (name incl. .weight). Fail closed on missing/zero N."""
        stem = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
        h = self.hfhs.get(stem, 0)
        if h is None:
            raise GroupError(f"{tensor_name}: no Hessian in L{self.layer:02d}.hfhs")
        ent = _dig(self.js, stem)
        if ent is None or "N" not in ent:
            raise GroupError(f"{tensor_name}: no N in L{self.layer:02d}.json")
        n = int(ent["N"])
        if n <= 0:
            raise GroupError(f"{tensor_name}: N == {n}")
        if "K" in ent and int(ent["K"]) != h.shape[0]:
            raise GroupError(f"{tensor_name}: JSON K {ent['K']} != HFHS K {h.shape[0]}")
        return h, n, h.shape[0]

    def expert_ids(self):
        ids = set()
        for name in self.hfhs.names():
            mm = re.search(r"\.mlp\.experts\.(\d+)\.gate_up_proj$", name)
            if mm:
                ids.add(int(mm.group(1)))
        return sorted(ids)


class SrcDump:
    """X/calib/src/L{LL}.f32 + L{LL}.json (schema hipfire.inkling2.srcdump.v1: tensors[{name, offset_bytes,
    shape, numel}], formats{w13,w2})."""

    def __init__(self, src_dir, layer):
        base = Path(src_dir) / f"L{layer:02d}"
        self.jpath = base.with_suffix(".json")
        self.js = json.loads(self.jpath.read_text())
        if self.js.get("schema") != "hipfire.inkling2.srcdump.v1":
            raise ValueError(f"{self.jpath}: unexpected schema {self.js.get('schema')!r}")
        self.path = base.with_suffix(".f32")
        self.size = self.path.stat().st_size
        self.map = np.memmap(self.path, dtype="<f4", mode="r")
        self.entries = {t["name"]: t for t in self.js["tensors"]}
        self.format = (self.js.get("formats") or {}).get("w13")

    def _locate(self, name):
        if name in self.entries:
            return self.entries[name]
        raise GroupError(f"{name}: not in source dump index {self.jpath}")

    def get(self, name):
        ent = self._locate(name)
        shape = tuple(int(x) for x in ent["shape"])
        off = int(ent["offset_bytes"])
        n = int(np.prod(shape))
        if off % 4 or off + n * 4 > self.size:
            raise GroupError(f"{name}: source dump range [{off}, +{n * 4}) out of file {self.path}")
        return np.array(self.map[off // 4:off // 4 + n], dtype=np.float32).reshape(shape)


def sha256_file(path, block=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def hfq_index(path):
    tools = Path(__file__).resolve().parents[2] / "tools"
    for cand in (tools, Path("/home/kaden/qcal/release-0.5.0/inkling/xts/tools")):
        if (cand / "hfq_census.py").exists():
            sys.path.insert(0, str(cand))
            break
    import hfq_census
    arch, meta, ts = hfq_census.index(path)
    return arch, meta, {t[0]: t for t in ts}


def hfq_tensor_bytes(path, entry):
    _, _, _, _, off, nbytes = entry
    return np.fromfile(path, dtype=np.uint8, count=nbytes, offset=off)


# ----------------------------------------------------------------------------------------------
# Group enumeration and the worker
# ----------------------------------------------------------------------------------------------
def tname(layer, slot):
    return f"model.llm.layers.{layer}.attn.{slot}.weight"


def ename(layer, e, proj):
    return f"model.llm.layers.{layer}.mlp.experts.{e}.{proj}.weight"


def enumerate_groups(layer, experts):
    groups = []
    for kind, slots in DENSE_GROUPS.items():
        groups.append({"gid": f"L{layer:02d}.{kind}", "layer": layer, "kind": kind,
                       "members": [tname(layer, s) for s in slots], "h_name": tname(layer, slots[0]),
                       "k_phys": None})
    for e in experts:
        groups.append({"gid": f"L{layer:02d}.e{e:03d}.GU", "layer": layer, "kind": "GU", "expert": e,
                       "members": [ename(layer, e, "gate_up_proj")], "h_name": ename(layer, e, "gate_up_proj"),
                       "k_phys": None})
        groups.append({"gid": f"L{layer:02d}.e{e:03d}.DN", "layer": layer, "kind": "DN", "expert": e,
                       "members": [ename(layer, e, "down_proj")], "h_name": ename(layer, e, "down_proj"),
                       "k_phys": DOWN_PHYS})
    return groups


_WORKER = {}


def _worker_init(threads, ctx):
    init_threads(threads)
    _WORKER.update(ctx)
    _WORKER["cache"] = {}


def _layer_data(layer):
    c = _WORKER["cache"]
    if layer not in c:
        c.clear()
        c[layer] = (LayerHessians(_WORKER["hfhs_dir"], layer), SrcDump(_WORKER["src_dir"], layer))
    return c[layer]


def run_group(group):
    ctx = _WORKER
    t0 = time.monotonic()
    try:
        lh, src = _layer_data(group["layer"])
        h, n, k = lh.fetch(group["h_name"])
        if group["kind"] == "Q":  # wr_du / gate_attn consume the same rows: verify the HFHS agrees
            for sib in group["members"][1:]:
                stem = sib[:-7]
                other = lh.hfhs.get(stem, 0)
                if other is not None and not np.array_equal(other, h):
                    raise GroupError(f"{sib}: Hessian differs from producer {group['h_name']}")
        ws = [src.get(nm) for nm in group["members"]]
        kl = ws[0].shape[1]
        if kl != k:
            raise GroupError(f"{group['gid']}: Hessian K {k} != weight K {kl}")
        if group["kind"] == "DN" and kl != DOWN_LIVE:
            raise GroupError(f"{group['gid']}: down logical K {kl} != {DOWN_LIVE}")
        min_n = ctx.get("min_n", 0)
        if ctx.get("enforce_min_n") and n < min_n:
            raise GroupError(f"{group['h_name']}: N {n} < required {min_n}")
        res = solve_core(ws, h, n, k_phys=group["k_phys"], alphas=ctx["alphas"], rounds=ctx["rounds"],
                         label=group["gid"], member_names=group["members"], keff=ctx["keff"],
                         factor_impl=ctx["factor_impl"])
        for mem in res["members"]:
            save_record(Path(ctx["out"]) / "records" / (mem["name"] + ".safetensors"), mem["name"], mem,
                        ctx["source_sha"])
        log = res["log"]
        log["layer"] = group["layer"]
        log["kind"] = group["kind"]
        if "expert" in group:
            log["expert"] = group["expert"]
        log["coverage_below_min_n"] = bool(n < min_n)
        log["total_seconds"] = round(time.monotonic() - t0, 3)
        import resource
        log["worker_max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        return {"ok": True, "gid": group["gid"], "layer": group["layer"], "log": log}
    except Exception as exc:  # fail closed: surfaced to the parent, which aborts
        import traceback
        return {"ok": False, "gid": group["gid"], "layer": group["layer"], "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()}


def parse_ranges(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_solve(args):
    import multiprocessing as mp
    if args.workers * args.threads > 48:
        raise SystemExit(f"workers*threads = {args.workers * args.threads} > 48")
    out = Path(args.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "log").mkdir(parents=True, exist_ok=True)
    source_sha = args.source_sha or sha256_file(args.template)
    layers = parse_ranges(args.layers)
    alphas = tuple(float(a) for a in args.alphas.split(",")) if args.alphas else ALPHAS
    ctx = {"hfhs_dir": args.hfhs_dir, "src_dir": args.src_dir, "out": str(out), "alphas": alphas,
           "rounds": args.rounds, "keff": args.keff, "factor_impl": args.factor_impl, "min_n": args.min_n,
           "enforce_min_n": args.enforce_min_n, "source_sha": source_sha}
    recipe = {"alphas": list(alphas), "rounds": args.rounds, "damp": 1.0, "seeds": [SEED_PRE, SEED_POST],
              "factor_impl": args.factor_impl, "source_sha": source_sha, "script": "inkling_h2.py",
              "script_sha256": sha256_file(__file__)}
    (out / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n")

    todo = []
    done = 0
    for layer in layers:
        lh = LayerHessians(args.hfhs_dir, layer)
        experts = parse_ranges(args.experts) if args.experts else lh.expert_ids()
        if not experts:
            raise SystemExit(f"L{layer:02d}: no expert Hessians found in HFHS")
        logp = out / "log" / f"L{layer:02d}.jsonl"
        have = set()
        if logp.exists():
            for line in logp.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if all((out / "records" / (n + ".safetensors")).exists() for n in rec.get("members", {})):
                        have.add(rec["label"])
        for g in enumerate_groups(layer, experts):
            if g["gid"] in have:
                done += 1
            else:
                todo.append(g)
        lh.hfhs.close()
    print(f"[solve] {len(todo)} groups to do, {done} already done; workers={args.workers} threads={args.threads}",
          flush=True)
    t0 = time.monotonic()
    failures = []
    n_done = 0
    ctxm = mp.get_context("spawn")
    with ctxm.Pool(args.workers, initializer=_worker_init, initargs=(args.threads, ctx)) as pool:
        for res in pool.imap_unordered(run_group, todo):
            if not res["ok"]:
                failures.append(res)
                print(f"[solve] FAILED {res['gid']}: {res['error']}\n{res['trace']}", file=sys.stderr, flush=True)
                pool.terminate()
                break
            n_done += 1
            with open(out / "log" / f"L{res['layer']:02d}.jsonl", "a") as f:
                f.write(json.dumps(res["log"], sort_keys=True) + "\n")
            lg = res["log"]
            print(f"[solve] {n_done}/{len(todo)} {res['gid']} alpha={lg['alpha']:g} "
                  f"E_rtn={lg['E_rtn_s1']:.4g} E_gptq={lg['E_gptq']:.4g} E_final={lg['E_final']:.4g} "
                  f"{lg['wall_seconds']}s (elapsed {time.monotonic() - t0:.0f}s)", flush=True)
    if failures:
        raise SystemExit(f"solve aborted: {failures[0]['gid']}: {failures[0]['error']}")
    import resource
    print(f"[solve] done in {time.monotonic() - t0:.0f}s; parent max RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB (workers log their own)", flush=True)


# ----------------------------------------------------------------------------------------------
# Held-out metric
# ----------------------------------------------------------------------------------------------
def deq_to_original(s_bits, d_bits, z_bits, codes):
    """Ŵ in original coordinates (f32 torch [M,K_phys]) = R^T deq / S16 (inverse FWHT then column divide)."""
    q = dequant_f32(d_bits, z_bits, codes)
    back = fwht256(q, inverse=True)
    return back / torch.from_numpy(f16_bits_to_f32(s_bits))[None, :]


def b_variant_original(w_live, k_phys):
    """B = symmetric RTN with S=1 recomputed in python -> (Ŵ f32 torch, groups u8 for cross-checking)."""
    m, kl = w_live.shape
    w = np.zeros((m, k_phys), dtype=np.float32)
    w[:, :kl] = w_live
    wrot = fwht256(torch.from_numpy(w))
    d, z, c = rtn_symmetric(wrot.numpy())
    s1 = np.full(k_phys, 0x3C00, dtype=np.uint16)
    return deq_to_original(s1, d, z, c), pack_groups(d, z, c)


def _site_key(token):
    key = SITE_ALIASES.get(re.sub(r"[^a-z_0-9]", "", token.lower()))
    if key is None:
        key = SITE_ALIASES.get(token.lower())
    return key


def load_check(check_dir, layer):
    """Yield (site_kind, rows f32 [n,K], per-row expert ids np.int64 [n] (-1 for dense sites)) for every site
    of a layer from `L{LL}.index.json` (schema hipfire.inkling2.check_rows.v1: sites -> {file, K, rows,
    segments[{row0, rows, expert, ...}]}) and the site's F32 row file."""
    ipath = Path(check_dir) / f"L{layer:02d}.index.json"
    js = json.loads(ipath.read_text())
    if js.get("schema") != "hipfire.inkling2.check_rows.v1":
        raise ValueError(f"{ipath}: unexpected schema {js.get('schema')!r}")
    for site, ent in sorted(js["sites"].items()):
        kind = _site_key(site)
        if kind is None:
            raise ValueError(f"{ipath}: unknown check site '{site}'")
        k, rows = int(ent["K"]), int(ent["rows"])
        p = Path(check_dir) / ent["file"]
        data = np.fromfile(p, dtype="<f4")
        if data.size != rows * k:
            raise ValueError(f"{p}: size {data.size} != rows {rows} x K {k}")
        experts = np.full(rows, -1, dtype=np.int64)
        covered = 0
        for seg in ent["segments"]:
            r0, n = int(seg["row0"]), int(seg["rows"])
            if seg.get("expert") is not None:
                experts[r0:r0 + n] = int(seg["expert"])
            covered += n
        if covered != rows:
            raise ValueError(f"{ipath}: site {site}: segments cover {covered} of {rows} rows")
        yield kind, data.reshape(rows, k), experts


def cmd_heldout(args):
    out = Path(args.out)
    formats = {}
    if args.formats:
        formats = {int(a): b for a, b in (x.split(":") for x in args.formats.split(","))}
    layers = parse_ranges(args.layers) if args.layers else sorted(
        {int(p.name[1:3]) for p in (out / "log").glob("L*.jsonl")})
    source_sha = args.source_sha or (sha256_file(args.template) if args.template else None)
    tindex = hfq_index(args.template)[2] if (args.template and args.crosscheck > 0) else {}
    per = []  # per tensor rows
    crosscheck = {"checked": 0, "equal": 0, "mismatch": []}
    n_cc = args.crosscheck
    for layer in layers:
        lh = LayerHessians(args.hfhs_dir, layer)
        src = SrcDump(args.src_dir, layer)
        formats.setdefault(layer, src.format or DEFAULT_FORMATS.get(layer, "unknown"))
        for kind, x, experts in load_check(args.check_dir, layer):
            if kind in DENSE_GROUPS:
                targets = [(nm, None, x) for nm in (tname(layer, s) for s in DENSE_GROUPS[kind])]
            else:
                if (experts < 0).any():
                    raise ValueError(f"L{layer:02d} {kind}: check rows without an expert id")
                proj = "gate_up_proj" if kind == "GU" else "down_proj"
                targets = [(ename(layer, int(e), proj), int(e), x[experts == e]) for e in np.unique(experts)]
            for name, e, xr in targets:
                rec_path = out / "records" / (name + ".safetensors")
                if not rec_path.exists() or xr.shape[0] == 0:
                    per.append({"name": name, "layer": layer, "site": kind, "expert": e, "rows": int(xr.shape[0]),
                                "skipped": "no record" if not rec_path.exists() else "no check rows"})
                    continue
                meta, s_bits, d_bits, z_bits, codes = load_record(rec_path)
                if source_sha and meta.get("source_sha") != source_sha:
                    raise ValueError(f"{rec_path}: source_sha {meta.get('source_sha')} != template {source_sha}")
                w = src.get(name)
                kl = w.shape[1]
                kp = codes.shape[1]
                if xr.shape[1] != kl:
                    raise ValueError(f"{name}: check K {xr.shape[1]} != weight K {kl}")
                what_c = deq_to_original(s_bits, d_bits, z_bits, codes)[:, :kl].double()
                what_b, groups_b = b_variant_original(w, kp)
                what_b = what_b[:, :kl].double()
                if n_cc > 0 and name in tindex:
                    n_cc -= 1
                    raw = hfq_tensor_bytes(args.template, tindex[name])
                    ok = bool(np.array_equal(raw.reshape(-1, GROUP_BYTES), groups_b))
                    crosscheck["checked"] += 1
                    crosscheck["equal"] += int(ok)
                    if not ok:
                        crosscheck["mismatch"].append(name)
                wd = torch.from_numpy(w).double()
                xt = torch.from_numpy(np.ascontiguousarray(xr)).double()
                eb = float(((xt @ (wd - what_b).T) ** 2).sum())
                ec = float(((xt @ (wd - what_c).T) ** 2).sum())
                _, n_route, _ = lh.fetch(name)
                per.append({"name": name, "layer": layer, "site": kind, "expert": e, "rows": int(xr.shape[0]),
                            "N": n_route, "E_B": eb, "E_C": ec})
    agg = _aggregate(per, formats)
    stop = _stop_rule(agg)
    result = {"formats": {str(k): v for k, v in formats.items()}, "source_sha": source_sha,
              "crosscheck_B_vs_template": crosscheck, "pilot_stop_rule": stop, "aggregate": agg, "tensors": per}
    (out / "heldout.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"crosscheck_B_vs_template": crosscheck, "pilot_stop_rule": stop, "aggregate": agg}, indent=1))


def _stop_rule(agg, threshold=0.05):
    """Plan section 3: stop before the full solve if the route-count-weighted held-out reconstruction
    improvement is < 5% or either source format shows systematic worsening (weighted improvement < 0 or
    more than half of its tensors worse than B)."""
    overall = agg.get("all", {}).get("improvement_weighted")
    fmts = {}
    for key, val in agg.items():
        if key.startswith("format/"):
            fmt = key.split("/", 1)[1]
            w = agg["_worse_tensors_by_format"].get(fmt, {"tensors": 0, "worse": 0})
            frac = w["worse"] / w["tensors"] if w["tensors"] else None
            fmts[fmt] = {"improvement_weighted": val["improvement_weighted"], "worse_tensor_fraction": frac,
                         "systematic_worsening": bool(val["improvement_weighted"] is not None and (
                             val["improvement_weighted"] < 0 or (frac is not None and frac > 0.5)))}
    passed = overall is not None and overall >= threshold and not any(f["systematic_worsening"] for f in fmts.values())
    return {"overall_improvement_weighted": overall, "threshold": threshold, "formats": fmts,
            "verdict": "CONTINUE" if passed else "STOP"}


def _aggregate(per, formats):
    """Route-count-weighted (weight N per tensor on the per-row mean error) and plain sums per layer/site
    and per source format; improvement = 1 - C/B."""
    def new():
        return {"tensors": 0, "rows": 0, "sumE_B": 0.0, "sumE_C": 0.0, "wE_B": 0.0, "wE_C": 0.0, "wsum": 0.0}

    groups = {}

    def add(key, row):
        g = groups.setdefault(key, new())
        g["tensors"] += 1
        g["rows"] += row["rows"]
        g["sumE_B"] += row["E_B"]
        g["sumE_C"] += row["E_C"]
        w = row["N"]
        g["wE_B"] += w * row["E_B"] / row["rows"]
        g["wE_C"] += w * row["E_C"] / row["rows"]
        g["wsum"] += w

    for row in per:
        if "skipped" in row:
            continue
        add(f"layer{row['layer']:02d}/{row['site']}", row)
        add(f"layer{row['layer']:02d}/all", row)
        fmt = formats.get(row["layer"], "unknown")
        add(f"format/{fmt}", row)
        add("all", row)
    res = {}
    for k, g in sorted(groups.items()):
        res[k] = {"tensors": g["tensors"], "rows": g["rows"],
                  "improvement_weighted": (1 - g["wE_C"] / g["wE_B"]) if g["wE_B"] > 0 else None,
                  "improvement_sum": (1 - g["sumE_C"] / g["sumE_B"]) if g["sumE_B"] > 0 else None,
                  "E_B_weighted": g["wE_B"], "E_C_weighted": g["wE_C"]}
    # per-tensor worsening counts per format
    worse = {}
    for row in per:
        if "skipped" in row:
            continue
        fmt = formats.get(row["layer"], "unknown")
        w = worse.setdefault(fmt, {"tensors": 0, "worse": 0})
        w["tensors"] += 1
        w["worse"] += int(row["E_C"] > row["E_B"])
    res["_worse_tensors_by_format"] = worse
    res["_skipped_tensors"] = sum(1 for r in per if "skipped" in r)
    return res


# ----------------------------------------------------------------------------------------------
# Synthetic throughput benchmark
# ----------------------------------------------------------------------------------------------
def synth_group(m, kl, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    colscale = np.exp(rng.normal(0, 0.8, kl)).astype(np.float32)
    x = rng.standard_normal((n, kl), dtype=np.float32) * colscale
    h = (x.T.astype(np.float64) @ x.astype(np.float64)) / n
    w = (rng.standard_normal((m, kl)) * 0.02).astype(np.float32)
    return w, h.astype(np.float32), n


def cmd_bench(args):
    init_threads(args.threads)
    rows = []
    for kind, (m, kl, kp) in {"GU": (2304, 2304, 2304), "DN": (2304, 1152, 1280)}.items():
        w, h, n = synth_group(m, kl)
        alphas = tuple(float(a) for a in args.alphas.split(",")) if args.alphas else ALPHAS
        times = []

        def prog(label, alpha, e, t):
            times.append(t)

        t0 = time.monotonic()
        res = solve_core([w], h, n, k_phys=kp, alphas=alphas, label=f"bench.{kind}", keff=args.keff,
                         factor_impl=args.factor_impl, progress=prog)
        wall = time.monotonic() - t0
        lg = res["log"]
        rows.append({"kind": kind, "threads": args.threads, "wall_s": round(wall, 3),
                     "alpha_s_mean": round(float(np.mean(times)), 3), "alpha_s": [round(t, 3) for t in times],
                     "keff_s": lg["keff_seconds"], "refit_s": lg["refit_seconds"],
                     "rounds_accepted": lg["rounds_accepted"], "alpha": lg["alpha"],
                     "E_rtn_s1": lg["E_rtn_s1"], "E_gptq": lg["E_gptq"], "E_final": lg["E_final"]})
        print(json.dumps(rows[-1]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--hfhs-dir", required=True)
    s.add_argument("--src-dir", required=True)
    s.add_argument("--template", required=True, help="B.hfq; its sha256 becomes the record source_sha")
    s.add_argument("--source-sha", help="override sha256(template) (tests only)")
    s.add_argument("--layers", required=True, help="e.g. 5,6,20 or 0-23")
    s.add_argument("--out", required=True)
    s.add_argument("--experts", help="subset, e.g. 0-127 or 3,9 (default: all experts in the HFHS)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--rounds", type=int, default=REFIT_ROUNDS)
    s.add_argument("--alphas", help="override comma list (default frozen grid)")
    s.add_argument("--keff", choices=("eig", "skip"), default="eig")
    s.add_argument("--factor-impl", choices=("fast", "ref"), default="fast")
    s.add_argument("--min-n", type=int, default=256)
    s.add_argument("--enforce-min-n", action="store_true")
    s.set_defaults(fn=cmd_solve)
    h = sub.add_parser("heldout")
    h.add_argument("--out", required=True)
    h.add_argument("--check-dir", required=True)
    h.add_argument("--hfhs-dir", required=True)
    h.add_argument("--src-dir", required=True)
    h.add_argument("--template", required=True)
    h.add_argument("--layers")
    h.add_argument("--formats", help="layer:format list, default 5:nvfp4,6:nvfp4,20:mxfp8")
    h.add_argument("--crosscheck", type=int, default=4, help="B tensors to compare against template bytes")
    h.add_argument("--source-sha", help="override sha256(template) (tests only)")
    h.set_defaults(fn=cmd_heldout)
    b = sub.add_parser("bench")
    b.add_argument("--threads", type=int, default=12)
    b.add_argument("--alphas")
    b.add_argument("--keff", choices=("eig", "skip"), default="eig")
    b.add_argument("--factor-impl", choices=("fast", "ref"), default="fast")
    b.set_defaults(fn=cmd_bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
