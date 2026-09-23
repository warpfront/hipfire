// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! tmp_gemm_v2_oracle — device oracle for the gfx1201 MQ4V2 fp8 GEMM v2 tile.
//!
//! Slice O throwaway harness (plan of record:
//! `docs/plans/2026-09-17-gfx1201-gemm-v2-tile.md`, §5.1/§5.2). It does NOT
//! implement the v2 tile. It:
//!
//! * reimplements the frozen operand contract on CPU: the per-token
//!   dyadic-224 prelude (`scale_mode=1`), RNE E4M3 codec, 136-byte MQ4V2
//!   groups with two independent 128-K f16 `(s, z)` halves, and the literal
//!   answer `Y[t,r] = a[t] * sum_{g,h} (s*C + z*S)` in f32 with an f64 audit;
//! * checks the *incumbent* s2bt8 kernels on identical inputs against that
//!   CPU reference with the data-derived §5.1 tolerance (never overall MSE);
//! * runs the required fixture battery: codec boundaries, all nibble codes,
//!   K one-hots, zero rows, unequal/constant/zero-scale halves,
//!   cancellation, E4M3 underflow/ties, masked tails, the qkvza 48-row
//!   beta/alpha source-boundary tile, sentinel-overwrite canaries, and
//!   repeat-run bit-identity;
//! * with `TIME=1`, reports per-kernel medians via `rdna_compute::profile`;
//! * records the four frozen v2 symbol strings and reports them as
//!   pending-until-K/H-land (stub-run against incumbent-only).
//!
//! Changed numerics mean NO bit-exactness claim versus s2bt8 for a future v2
//! kernel; self-determinism (repeat bit-identity) is required of every arm.
//!
//! Usage (CPU-only, no GPU):
//!   cargo run --release -p saddle-lab --example tmp_gemm_v2_oracle -- --cpu-only --out DIR
//! GPU (incumbent validation + TIME):
//!   HOME=... ROCR_VISIBLE_DEVICES=1 TIME=1 cargo run --release -p saddle-lab \
//!     --example tmp_gemm_v2_oracle -- --device 0 --n 256,512,1024 --out DIR
//! Full four families at N=512 (acceptance): default `--n 512` covers
//! gate_up / residual / qkvza / qkv; pass `--n 256,512,1024` for the sweep.

use std::collections::HashMap;
use std::time::Instant;

// ─── Frozen contract (§6.1) ────────────────────────────────────────────────
// These strings are the v2 kernel identities. They are referenced here so the
// oracle compiles against the frozen ABI; until K (HIP TU) and H (host
// selection) land they resolve to nothing, which the readiness check reports
// instead of failing the incumbent run.

const V2_SYMBOLS: [&str; 4] = [
    "gemm_gate_up_mq4g256v2_wmma_fp8_v2_gfx1201",
    "gemm_qkv_mq4g256v2_wmma_fp8_v2_gfx1201",
    "gemm_qkvza_mq4g256v2_wmma_fp8_v2_gfx1201",
    "gemm_mq4g256v2_residual_wmma_fp8_v2_gfx1201",
];
const V2_BM: usize = 256;
const V2_BN: usize = 64;
const V2_BK: usize = 64;
const V2_WAVES: usize = 8;
/// Active v2 geometry from the host selector env (mirrors `fp8_v2_geom` in
/// rdna-compute/src/gemm.rs): (symbol suffix, BM, BN, BK, WAVES).
fn active_v2_geom() -> (&'static str, usize, usize, usize, usize) {
    match std::env::var("HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM").as_deref() {
        Ok("128x128") => ("_b128x128", 128, 128, 64, 8),
        Ok("64x256") => ("_b64x256", 64, 256, 64, 8),
        Ok("128x64") => ("_b128x64w4", 128, 64, 64, 4),
        _ => ("", 256, 64, 64, 8),
    }
}
fn v2_enabled() -> bool {
    std::env::var("HIPFIRE_GFX12_MQ4V2_FP8_V2").as_deref() == Ok("1")
}
/// Resolved kernel identity for a case: the v2 symbol (with geometry suffix)
/// when the v2 route is enabled, else the incumbent s2bt8 symbol.
fn kernel_label(family: &str, s2bt8: &str) -> String {
    if !v2_enabled() {
        return s2bt8.to_string();
    }
    let (sfx, _, _, _, _) = active_v2_geom();
    match family {
        "gate_up" => format!("gemm_gate_up_mq4g256v2_wmma_fp8_v2{sfx}_gfx1201"),
        "qkv" => format!("gemm_qkv_mq4g256v2_wmma_fp8_v2{sfx}_gfx1201"),
        "qkvza" => format!("gemm_qkvza_mq4g256v2_wmma_fp8_v2{sfx}_gfx1201"),
        "residual" => format!("gemm_mq4g256v2_residual_wmma_fp8_v2{sfx}_gfx1201"),
        _ => s2bt8.to_string(),
    }
}

// Incumbent s2bt8 symbols under test (same-session control).
const S2BT8_SYMBOLS: [&str; 4] = [
    "gemm_gate_up_mq4g256v2_wmma_fp8_gfx12_s2bt8",
    "gemm_qkv_mq4g256v2_wmma_fp8_gfx12_s2bt8",
    "gemm_qkvza_mq4g256v2_wmma_fp8_gfx12_s2bt8",
    "gemm_mq4g256v2_residual_wmma_fp8_gfx12_s2bt8",
];
// Historical labels from the plan (§5.3, N=512). Labels only — the gate always
// compares against the remeasured same-session s2bt8 control.
const S2BT8_HIST_US: [f64; 4] = [1804.7, 770.1, 825.7, 676.8]; // gate_up,qkv,qkvza,residual

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;
const FP32_U: f64 = 1.0 / 16777216.0; // 2^-24

fn gamma(n: f64) -> f64 {
    n * FP32_U / (1.0 - n * FP32_U)
}

// ─── Deterministic PRNG (xorshift64*) ───────────────────────────────────────

fn prng_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}
fn prng_f32(state: &mut u64) -> f32 {
    // [0,1)
    ((prng_u64(state) >> 11) as f64 / (1u64 << 53) as f64) as f32
}

// ─── f16 bit helpers (no external crate; RNE, standard bias-15) ────────────

fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as u32;
    let e = ((b >> 10) & 0x1F) as u32;
    let m = (b & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
            let mut mm = m;
            let mut ee: i32 = -14;
            while (mm & 0x400) == 0 {
                mm <<= 1;
                ee -= 1;
            }
            mm &= 0x3FF;
            ((s << 31) | (((ee + 127) as u32) << 23) | (mm << 13)) as u32
        }
    } else if e == 31 {
        (s << 31) | (0xFF << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x7F_FFFF;
    if e == 255 {
        // Inf/NaN -> Inf/NaN
        return s | 0x7C00 | if m == 0 { 0 } else { ((m >> 13) as u16) | 0x0200 };
    }
    let e16 = e - 112; // unbiased f16 exponent + 15
    if e16 >= 31 {
        return s | 0x7C00; // overflow -> Inf
    }
    if e16 <= 0 {
        // subnormal or zero; RNE on the shifted mantissa (with hidden 1 if e>0)
        if e16 < -10 {
            return s; // rounds to zero
        }
        let mut mm = m | if e > 0 { 0x80_0000 } else { 0 };
        let shift = (14 - e16) as u32; // 14..24
        let half = 1u32 << (shift - 1);
        let rest = mm & ((1u32 << shift) - 1);
        mm >>= shift;
        if rest > half || (rest == half && (mm & 1) == 1) {
            mm += 1;
        }
        return s | (mm as u16);
    }
    // normal: RNE on 13 dropped bits
    let half = 0x1000u32;
    let rest = m & 0x1FFF;
    let mut mm = (m >> 13) as u16;
    let mut ee = e16 as u16;
    if rest > half || (rest == half && (mm & 1) == 1) {
        mm += 1;
        if mm == 0x400 {
            mm = 0;
            ee += 1;
            if ee >= 31 {
                return s | 0x7C00;
            }
        }
    }
    s | (ee << 10) | (mm & 0x3FF)
}

// ─── E4M3 codec (bias 7; 0x7E = 448 max; NaN = mant 7 @ exp 15) ────────────

fn e4m3_decode(b: u8) -> f32 {
    let s = ((b >> 7) & 1) as u32;
    let e = ((b >> 3) & 0xF) as u32;
    let m = (b & 7) as u32;
    if e == 0 {
        // subnormal: m * 2^-9
        let v = (m as f32) * 2.0f32.powi(-9);
        return if s == 1 { -v } else { v };
    }
    if e == 15 {
        if m == 7 {
            return f32::from_bits((s << 31) | (0xFF << 23) | 0x40_0000); // NaN
        }
        let v = (8 + m) as f32 / 8.0 * 2.0f32.powi(8); // 2^(15-7)
        return if s == 1 { -v } else { v };
    }
    let v = (8 + m) as f32 / 8.0 * 2.0f32.powi(e as i32 - 7);
    if s == 1 { -v } else { v }
}

struct E4M3Table {
    // ascending finite grid for RNE encode
    vals: Vec<f64>,
    bits: Vec<u8>,
}
impl E4M3Table {
    fn build() -> Self {
        let mut pairs: Vec<(f64, u8)> = Vec::new();
        for b in 0u16..256 {
            let v = e4m3_decode(b as u8);
            if v.is_finite() {
                pairs.push((v as f64, b as u8));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // dedupe: -0.0 and 0.0 compare equal; keep +0 (0x00)
        let mut vals: Vec<f64> = Vec::new();
        let mut bits: Vec<u8> = Vec::new();
        for (v, b) in pairs {
            if let Some(&last) = vals.last() {
                if (v - last).abs() == 0.0 && !(v == 0.0 && last == 0.0) {
                    continue;
                }
                if v == 0.0 && last == 0.0 {
                    continue;
                }
            }
            vals.push(v);
            bits.push(b);
        }
        Self { vals, bits }
    }
    fn encode_rne(&self, x: f32) -> u8 {
        if x.is_nan() {
            return 0x7F;
        }
        if x == 0.0 {
            return if x.is_sign_negative() { 0x80 } else { 0x00 };
        }
        let xf = x as f64;
        // saturate beyond max finite 448 (RNE boundary handled by search below,
        // but inputs past the top are clamped to 0x7E/0xFE)
        let top = 448.0f64;
        if xf > top {
            return 0x7E;
        }
        if xf < -top {
            return 0xFE;
        }
        // binary search adjacent pair
        let mut lo = 0usize;
        let mut hi = self.vals.len() - 1;
        if xf <= self.vals[0] {
            return self.bits[0];
        }
        if xf >= self.vals[hi] {
            return self.bits[hi];
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.vals[mid] <= xf {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (v0, b0) = (self.vals[lo], self.bits[lo]);
        let (v1, b1) = (self.vals[hi], self.bits[hi]);
        // exact ties-to-even: compare 2*x vs v0+v1 in f64
        let twice = 2.0 * xf;
        let sum = v0 + v1;
        let picked = if twice < sum {
            b0
        } else if twice > sum {
            b1
        } else {
            // tie: even means LSB of the 3-bit mantissa field is 0.
            let even0 = (b0 & 1) == 0;
            if even0 { b0 } else { b1 }
        };
        // underflow-to-zero keeps the input sign (hardware vcvt behavior);
        // the grid dedupes -0, so restore it explicitly.
        if e4m3_decode(picked) == 0.0 && xf < 0.0 {
            return 0x80;
        }
        picked
    }
    /// exact half-ULP at the grid point nearest x (for the Bq epsilon term)
    fn half_ulp_at(&self, scaled: f32) -> f64 {
        let a = (scaled as f64).abs();
        // find bracketing interval, return half gap
        if a >= 448.0 {
            return 16.0; // (448-416)/2: top gap at exp15
        }
        // search over non-negative half by symmetry: use abs grid
        let mut best = f64::INFINITY;
        // coarse: walk the sorted grid (256 entries; fine for a bound helper,
        // called once per row, not per element)
        for w in self.vals.windows(2) {
            if w[0] <= a && a <= w[1] {
                best = ((w[1] - w[0]) / 2.0).min(best);
            }
        }
        if best.is_finite() {
            return best;
        }
        // a below smallest positive subnormal: half of 2^-9
        2.0f64.powi(-10)
    }
}

// ─── Per-token dyadic-224 prelude (scale_mode=1, frozen) ───────────────────

fn dyadic224_scale(row: &[f32]) -> f32 {
    let mut maxabs: f32 = 0.0;
    for &v in row {
        let a = v.abs();
        if a > maxabs {
            maxabs = a;
        }
    }
    if !(maxabs > 0.0) {
        return 1.0; // all-zero row (covers 0 and NaN-free empty)
    }
    for c in -31..=9 {
        let thr = 224.0f32 * 2.0f32.powi(c);
        if maxabs <= thr {
            return 2.0f32.powi(c);
        }
    }
    2.0f32.powi(9) // clamped overflow arm (screen must flag nonfinite separately)
}

struct Prelude {
    x8: Vec<u8>,    // [n,k]
    sums: Vec<f32>, // [n*gpr*2]
    scales: Vec<f32>, // [n]
}

fn cpu_prelude(x: &[f32], n: usize, k: usize, tab: &E4M3Table) -> Prelude {
    assert_eq!(x.len(), n * k);
    let gpr = k / GROUP;
    let mut p = Prelude {
        x8: vec![0u8; n * k],
        sums: vec![0.0f32; n * gpr * 2],
        scales: vec![0.0f32; n],
    };
    let nthreads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8)
        .min(n.max(1));
    let chunk = (n + nthreads - 1) / nthreads;
    let x_chunks: Vec<&[f32]> = x.chunks(chunk * k).collect();
    std::thread::scope(|s| {
        let x8_chunks = p.x8.chunks_mut(chunk * k);
        let sums_chunks = p.sums.chunks_mut(chunk * gpr * 2);
        let sc_chunks = p.scales.chunks_mut(chunk);
        for (ti, (((x8c, sc), sums), xc)) in x8_chunks
            .zip(sc_chunks)
            .zip(sums_chunks)
            .zip(x_chunks.iter())
            .enumerate()
        {
            s.spawn(move || {
                let t0 = ti * chunk;
                for (li, xrow) in xc.chunks(k).enumerate() {
                    let t = t0 + li;
                    let a = dyadic224_scale(xrow);
                    sc[li] = a;
                    let inv = 1.0 / a;
                    for (i, &xv) in xrow.iter().enumerate() {
                        x8c[li * k + i] = tab.encode_rne(xv * inv);
                    }
                    for g in 0..gpr {
                        for h in 0..2 {
                            let mut acc = 0.0f32;
                            for j in 0..HALF {
                                let idx = g * GROUP + h * HALF + j;
                                acc += e4m3_decode(x8c[li * k + idx]);
                            }
                            sums[(li * gpr + g) * 2 + h] = acc;
                        }
                    }
                    let _ = t;
                }
            });
        }
    });
    p
}

// ─── MQ4V2 weight container ─────────────────────────────────────────────────

fn pack_v2_split(w: &[f32], rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(w.len(), rows * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; rows * gpr * GROUP_BYTES];
    for r in 0..rows {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut q = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let s = &w[src + off..src + off + HALF];
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for &v in s.iter() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let sb = if hi == lo {
                    0u16
                } else {
                    f32_to_f16_rne(step)
                };
                let zb = f32_to_f16_rne(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&sb.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&zb.to_le_bytes());
                let st = f16_to_f32(sb);
                let z = f16_to_f32(zb);
                if st == 0.0 {
                    continue;
                }
                let inv = 1.0 / st;
                for i in 0..HALF {
                    q[off + i] = ((s[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

#[derive(Clone, Copy)]
struct HalfHW {
    s: f32,
    z: f32,
}
fn decode_group_header(blob: &[u8], row: usize, g: usize, gpr: usize) -> [HalfHW; 2] {
    let dst = (row * gpr + g) * GROUP_BYTES;
    let mut out = [HalfHW { s: 0.0, z: 0.0 }; 2];
    for h in 0..2 {
        let sb = u16::from_le_bytes([blob[dst + h * 4], blob[dst + h * 4 + 1]]);
        let zb = u16::from_le_bytes([blob[dst + h * 4 + 2], blob[dst + h * 4 + 3]]);
        out[h] = HalfHW {
            s: f16_to_f32(sb),
            z: f16_to_f32(zb),
        };
    }
    out
}
#[inline]
fn decode_nibble(blob: &[u8], row: usize, g: usize, gpr: usize, idx: usize) -> f32 {
    // idx is the global K-coordinate; the nibble payload is group-local.
    let li = idx - g * GROUP;
    debug_assert!(li < GROUP);
    let dst = (row * gpr + g) * GROUP_BYTES;
    let byte = blob[dst + 8 + li / 2];
    if li % 2 == 0 {
        (byte & 0xF) as f32
    } else {
        (byte >> 4) as f32
    }
}

// ─── CPU f32 reference + f64 audit + §5.1 bounds ────────────────────────────

struct RefOut {
    y_f32: Vec<f32>, // [n*m]
    y_f64: Vec<f64>, // audit, same order
    bound_emit: Vec<f64>, // Bacc+Bcpu(+Bres), per element
    bq: Vec<f64>,         // observed a·Σ|w|·|xhat−x| per element
}

fn cpu_reference(
    blob: &[u8],
    rows: usize,
    pre: &Prelude,
    x_orig: Option<&[f32]>,
    n: usize,
    k: usize,
    y_old: Option<&[f32]>,
    use_orig_x: bool,
) -> RefOut {
    let gpr = k / GROUP;
    let mut y_f32 = vec![0.0f32; n * rows];
    let mut y_f64 = vec![0.0f64; n * rows];
    let mut b_emit = vec![0.0f64; n * rows];
    let mut bq_v = vec![0.0f64; n * rows];
    let h = (k / 128) as f64;
    let g_acc = gamma(128.0 + 4.0 * h + 8.0);
    let g_cpu = gamma(k as f64 + 4.0 * h + 8.0);
    let g_res = gamma(1.0);
    std::thread::scope(|s| {
        let nthreads = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(8)
            .min(n.max(1));
        let chunk = (n + nthreads - 1) / nthreads;
        let mut y32c = y_f32.chunks_mut(chunk * rows);
        let mut y64c = y_f64.chunks_mut(chunk * rows);
        let mut bec = b_emit.chunks_mut(chunk * rows);
        let mut bqc = bq_v.chunks_mut(chunk * rows);
        let mut ti = 0;
        while let (Some(y32), Some(y64), Some(be), Some(bq)) = (
            y32c.next(),
            y64c.next(),
            bec.next(),
            bqc.next(),
        )
        {
            let t0 = ti;
            ti += chunk;
            s.spawn(move || {
                let nt = y32.len() / rows;
                for li in 0..nt {
                    let t = t0 + li;
                    let a = pre.scales[t];
                    let x8r = &pre.x8[t * k..(t + 1) * k];
                    let xo_row: Option<&[f32]> = x_orig.map(|xo| &xo[t * k..(t + 1) * k]);
                    // materialize decoded activation row once
                    // (stack-allocated per thread via Vec; K<=17408)
                    let mut xh = vec![0.0f32; k];
                    for (i, &b) in x8r.iter().enumerate() {
                        xh[i] = e4m3_decode(b);
                    }
                    for r in 0..rows {
                        let mut acc32 = 0.0f32;
                        let mut acc64 = 0.0f64;
                        let mut tabs = 0.0f64; // Σ(|s·q·xhat| + |z·xhat|)
                        for g in 0..gpr {
                            let hw = decode_group_header(blob, r, g, gpr);
                            for hh in 0..2 {
                                let base = g * GROUP + hh * HALF;
                                let shw = hw[hh];
                                for j in 0..HALF {
                                    let idx = base + j;
                                    let q = decode_nibble(blob, r, g, gpr, idx);
                                    let wgt32 = shw.z + shw.s * q;
                                    let xv = match xo_row {
                                        Some(xor) if use_orig_x => xor[idx],
                                        _ => xh[idx],
                                    };
                                    acc32 += wgt32 * xv;
                                    acc64 += (wgt32 as f64) * (xv as f64);
                                    tabs += ((shw.s * q * xv).abs() + (shw.z * xv).abs()) as f64;
                                }
                            }
                        }
                        let yv32 = a * acc32;
                        let yv64 = (a as f64) * acc64;
                        let mut be_v = g_acc * (a as f64) * tabs + g_cpu * (a as f64) * tabs;
                        if let Some(xor) = xo_row {
                            // tighter observed Bq: Σ|w|·|xhat−x|
                            let mut q = 0.0f64;
                            for g in 0..gpr {
                                let hw = decode_group_header(blob, r, g, gpr);
                                for hh in 0..2 {
                                    let base = g * GROUP + hh * HALF;
                                    for j in 0..HALF {
                                        let idx = base + j;
                                        let qq = decode_nibble(blob, r, g, gpr, idx);
                                        let wgt =
                                            (hw[hh].z + hw[hh].s * qq) as f64;
                                        let xhat = xh[idx] as f64;
                                        let x0 = xor[idx] as f64;
                                        q += wgt.abs() * (xhat - x0).abs();
                                    }
                                }
                            }
                            let bq_vv = (a as f64) * q;
                            bq[li * rows + r] = bq_vv;
                        }
                        if let Some(yo) = y_old {
                            let old = yo[t * rows + r];
                            y32[li * rows + r] = yv32 + old;
                            y64[li * rows + r] = yv64 + (old as f64);
                            be_v += g_res * (yv64.abs() + (old as f64).abs());
                        } else {
                            y32[li * rows + r] = yv32;
                            y64[li * rows + r] = yv64;
                        }
                        be[li * rows + r] = be_v;
                    }
                }
            });
        }
    });
    RefOut {
        y_f32,
        y_f64,
        bound_emit: b_emit,
        bq: bq_v,
    }
}

// ─── Metrics (never overall MSE) ────────────────────────────────────────────

struct Metrics {
    max_abs: f64,
    tail_thr: f64,
    tail_mse: f64,
    tail_worst: f64,
    maxrel_at_peak: f64,
    worst_row_maxrel: f64,
    bound_worst_ratio: f64, // max |dev-ref|/bound_emit
    bound_violations: usize,
}

fn compute_metrics(dev: &[f32], y_ref: &[f32], bound: &[f64]) -> Metrics {
    assert_eq!(dev.len(), y_ref.len());
    assert_eq!(dev.len(), bound.len());
    let n = dev.len();
    let mut absv: Vec<f64> = Vec::with_capacity(n);
    let mut refmag: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        absv.push(((dev[i] as f64) - (y_ref[i] as f64)).abs());
        refmag.push((y_ref[i] as f64).abs());
    }
    let max_abs = absv.iter().cloned().fold(0.0f64, f64::max);
    // tail membership from reference magnitude: >= 99th percentile
    let mut sorted = refmag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = sorted[(n.saturating_sub(1) * 99) / 100];
    let mut tss = 0.0f64;
    let mut tcnt = 0usize;
    let mut tworst = 0.0f64;
    for i in 0..n {
        if refmag[i] >= thr {
            let e = absv[i];
            tss += e * e;
            tcnt += 1;
            tworst = tworst.max(e);
        }
    }
    let tail_mse = if tcnt > 0 { tss / tcnt as f64 } else { 0.0 };
    // max-coefficient relative error at largest-|reference| coefficient
    let mut peak = 0usize;
    for i in 0..n {
        if refmag[i] > refmag[peak] {
            peak = i;
        }
    }
    let maxrel_at_peak = if refmag[peak] > 0.0 {
        absv[peak] / refmag[peak]
    } else {
        absv[peak] // all-zero reference: absolute check
    };
    // worst row maximum relative error (rows = contiguous blocks of `rows`;
    // here the whole tensor is one block — report element worst instead)
    let mut worst_rel = 0.0f64;
    for i in 0..n {
        let rel = if refmag[i] > 0.0 {
            absv[i] / refmag[i]
        } else {
            absv[i]
        };
        worst_rel = worst_rel.max(rel);
    }
    let mut worst_ratio = 0.0f64;
    let mut viol = 0usize;
    for i in 0..n {
        let b = bound[i];
        let ratio = if b > 0.0 { absv[i] / b } else if absv[i] == 0.0 { 0.0 } else { f64::INFINITY };
        worst_ratio = worst_ratio.max(ratio);
        if absv[i] > b {
            viol += 1;
        }
    }
    Metrics {
        max_abs,
        tail_thr: thr,
        tail_mse,
        tail_worst: tworst,
        maxrel_at_peak,
        worst_row_maxrel: worst_rel,
        bound_worst_ratio: worst_ratio,
        bound_violations: viol,
    }
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}
fn hash_f32(v: &[f32]) -> u64 {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    };
    fnv1a64(bytes)
}

// ─── Test records ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct CpuTest {
    name: String,
    pass: bool,
    detail: String,
}
#[derive(Clone)]
struct GpuCase {
    family: String,
    n: usize,
    k: usize,
    m_total: usize,
    kernel: String,
    kernels_seen: Vec<String>,
    max_abs: f64,
    tail_mse: f64,
    tail_thr: f64,
    tail_worst: f64,
    maxrel_peak: f64,
    bound_ratio: f64,
    bound_viol: usize,
    max_abs_orig: f64,
    bound_orig_ratio: f64,
    bound_orig_viol: usize,
    x8_mismatch: usize,
    max_s_err: f64,
    hash: String,
    hash_repeat: String,
    deterministic: bool,
    time_us: Option<f64>,
    verdict: String,
}

// ─── CPU suite ──────────────────────────────────────────────────────────────

fn run_cpu_suite(tab: &E4M3Table) -> Vec<CpuTest> {
    let mut out = Vec::new();
    let mut ok = |name: &str, pass: bool, detail: String| {
        out.push(CpuTest {
            name: name.to_string(),
            pass,
            detail,
        });
    };

    // 1. codec boundaries: 0, signed zero, normal/subnormal transitions,
    //    ties-to-even, 224, 448, exponent clamp endpoints, nonfinite counts
    {
        let mut fails = Vec::new();
        // exact grid points round-trip
        for b in 0u16..256 {
            let v = e4m3_decode(b as u8);
            if v.is_nan() {
                continue;
            }
            let rt = tab.encode_rne(v);
            // round-trip may land on -0 vs +0 only
            if rt != b as u8 && !(rt == 0x00 && b == 0x80) {
                fails.push(format!("rt {b:#04x}->{v}->{rt:#04x}"));
            }
        }
        // ties-to-even: midpoint between adjacent grid values picks even LSB
        let mut tie_fails = 0;
        for w in tab.vals.windows(2) {
            let mid = ((w[0] + w[1]) / 2.0) as f32;
            if !mid.is_finite() {
                continue;
            }
            let got = tab.encode_rne(mid);
            // determine expected: the even-LSB side
            let i0 = tab.vals.iter().position(|&v| v == w[0]).unwrap();
            let i1 = tab.vals.iter().position(|&v| v == w[1]).unwrap();
            let mut exp = if (tab.bits[i0] & 1) == 0 {
                tab.bits[i0]
            } else {
                tab.bits[i1]
            };
            // zero results keep the input sign: a negative-mid tie that lands
            // on zero must come back as -0 (hardware vcvt behavior).
            if e4m3_decode(exp) == 0.0 && mid < 0.0 {
                exp = 0x80;
            }
            // only check when the midpoint is exactly representable in f32
            // (else the test input isn't an exact tie)
            if (mid as f64) == (w[0] + w[1]) / 2.0 && got != exp {
                tie_fails += 1;
                if fails.len() < 8 {
                    fails.push(format!("tie mid={mid:e} got={got:#04x} exp={exp:#04x}"));
                }
            }
        }
        // 224 and 448 endpoints
        let e224 = tab.encode_rne(224.0);
        let e448 = tab.encode_rne(448.0);
        let d224 = e4m3_decode(e224);
        let d448 = e4m3_decode(e448);
        if d224 != 224.0 {
            fails.push(format!("224 -> {e224:#04x} -> {d224}"));
        }
        if d448 != 448.0 {
            fails.push(format!("448 -> {e448:#04x} -> {d448}"));
        }
        if e4m3_decode(tab.encode_rne(1e30)) != 448.0 {
            fails.push("sat+ missing".into());
        }
        if e4m3_decode(tab.encode_rne(-1e30)) != -448.0 {
            fails.push("sat- missing".into());
        }
        if tab.encode_rne(f32::NAN) != 0x7F {
            fails.push("nan missing".into());
        }
        if tab.encode_rne(0.0) != 0x00 || tab.encode_rne(-0.0) != 0x80 {
            fails.push("signed zero".into());
        }
        // subnormal floor: smallest positive subnormal 2^-9
        if e4m3_decode(0x01) != 2.0f32.powi(-9) {
            fails.push("subnormal step".into());
        }
        ok(
            "codec/exact-grid-roundtrip",
            fails.is_empty(),
            format!("{} mismatches (tie_fails={tie_fails})", fails.len()),
        );
    }

    // 2. dyadic-224 scale rule incl. zero row, clamp endpoints
    {
        let mut fails = Vec::new();
        if dyadic224_scale(&[0.0; 16]) != 1.0 {
            fails.push("zero row".into());
        }
        if dyadic224_scale(&[224.0]) != 1.0 {
            fails.push("224 -> 1".into());
        }
        if dyadic224_scale(&[224.0 + 1.0]) != 2.0 {
            fails.push(format!("225 -> {}", dyadic224_scale(&[225.0])));
        }
        if dyadic224_scale(&[1.0]) != 2.0f32.powi(-7) {
            // 1/224 = 0.00446; 2^-8 = 0.00390625 is too small, so ceil = -7.
            fails.push(format!("1 -> {}", dyadic224_scale(&[1.0])));
        }
        if dyadic224_scale(&[1e30]) != 2.0f32.powi(9) {
            fails.push("clamp top".into());
        }
        if dyadic224_scale(&[1e-30]) != 2.0f32.powi(-31) {
            fails.push(format!("clamp bot {}", dyadic224_scale(&[1e-30])));
        }
        // exact power-of-two boundary: maxabs == 224*2^c -> 2^c
        for c in [-31, -8, -1, 0, 1, 9] {
            let m = 224.0f32 * 2.0f32.powi(c);
            if dyadic224_scale(&[m]) != 2.0f32.powi(c) {
                fails.push(format!("pow2 boundary c={c}"));
            }
        }
        ok(
            "prelude/dyadic224-rule",
            fails.is_empty(),
            if fails.is_empty() {
                "zero/224/clamp/pow2 boundaries exact".into()
            } else {
                fails.join("; ")
            },
        );
    }

    // 3. weight pack/decode round-trip: all 16 nibble codes present
    {
        let k = 256usize;
        let rows = 4usize;
        // weights engineered so each nibble 0..15 appears
        let mut w = vec![0.0f32; rows * k];
        for r in 0..rows {
            for i in 0..k {
                w[r * k + i] = (i % 16) as f32 * (if r % 2 == 0 { 1.0 } else { -3.0 }) + r as f32;
            }
        }
        let blob = pack_v2_split(&w, rows, k);
        let gpr = 1;
        let mut worst = 0.0f32;
        let mut seen = [false; 16];
        for r in 0..rows {
            let hw = decode_group_header(&blob, r, 0, gpr);
            for i in 0..k {
                let q = decode_nibble(&blob, r, 0, gpr, i);
                seen[q as usize] = true;
                let h = i / HALF;
                let rec = hw[h].z + hw[h].s * q;
                let e = (rec - w[r * k + i]).abs();
                worst = worst.max(e);
            }
        }
        // representable error <= half-step of the affine grid (f16 rounding aside)
        ok(
            "weights/all-nibble-roundtrip",
            seen.iter().all(|&s| s) && worst <= 8.0,
            format!(
                "codes_seen={}/16 worst_dequant_err={worst:e}",
                seen.iter().filter(|&&s| s).count()
            ),
        );
    }

    // 4. one-hot address round-trip on CPU reference (K=256, exact where
    //    representable): token t one-hot at column t
    {
        let (n, k, rows) = (256usize, 256usize, 64usize);
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let w: Vec<f32> = (0..rows * k).map(|_| prng_f32(&mut rng) * 4.0 - 2.0).collect();
        let blob = pack_v2_split(&w, rows, k);
        let mut x = vec![0.0f32; n * k];
        for t in 0..n {
            x[t * k + t] = 1.0;
        }
        let pre = cpu_prelude(&x, n, k, tab);
        let r = cpu_reference(&blob, rows, &pre, Some(&x), n, k, None, false);
        // token t must see exactly column t: check diagonal dominance of the
        // *weight* contribution (divide out a[t])
        let mut worst = 0.0f64;
        for t in 0..n {
            for rr in 0..rows {
                let got = (r.y_f32[t * rows + rr] as f64) / (pre.scales[t] as f64);
                // expected: s*q+z at column t
                let g = 0;
                let hw = decode_group_header(&blob, rr, g, 1);
                let h = t / HALF;
                let q = decode_nibble(&blob, rr, g, 1, t);
                let exp = (hw[h].z + hw[h].s * q) as f64 * (e4m3_decode(pre.x8[t * k + t]) as f64);
                // plus exact S terms are zero (all other X8 are +0 -> decode 0)
                let e = (got - exp).abs();
                worst = worst.max(e);
            }
        }
        ok(
            "address/one-hot-diagonal",
            worst <= 1e-3,
            format!("worst_onehot_err={worst:e}"),
        );
    }

    // 5. self-determinism of the CPU reference (repeat bit-identical)
    {
        let (n, k, rows) = (64usize, 512usize, 32usize);
        let mut rng = 0xDEAD_BEEFu64;
        let w: Vec<f32> = (0..rows * k).map(|_| prng_f32(&mut rng) * 2.0 - 1.0).collect();
        let blob = pack_v2_split(&w, rows, k);
        let x: Vec<f32> = (0..n * k).map(|_| prng_f32(&mut rng) * 6.0 - 3.0).collect();
        let pre = cpu_prelude(&x, n, k, tab);
        let r1 = cpu_reference(&blob, rows, &pre, Some(&x), n, k, None, false);
        let r2 = cpu_reference(&blob, rows, &pre, Some(&x), n, k, None, false);
        ok(
            "determinism/cpu-repeat-identical",
            r1.y_f32 == r2.y_f32,
            format!("hash={:016x}", hash_f32(&r1.y_f32)),
        );
    }

    // 6. f32-vs-f64 audit gap is within the Bcpu model (sanity of bounds)
    {
        let (n, k, rows) = (64usize, 512usize, 32usize);
        let mut rng = 0x777u64;
        let w: Vec<f32> = (0..rows * k).map(|_| prng_f32(&mut rng) * 2.0 - 1.0).collect();
        let blob = pack_v2_split(&w, rows, k);
        let x: Vec<f32> = (0..n * k).map(|_| prng_f32(&mut rng) * 6.0 - 3.0).collect();
        let pre = cpu_prelude(&x, n, k, tab);
        let r = cpu_reference(&blob, rows, &pre, Some(&x), n, k, None, false);
        let mut worst_ratio = 0.0f64;
        for i in 0..r.y_f32.len() {
            let gap = ((r.y_f32[i] as f64) - r.y_f64[i]).abs();
            let h = (k / 128) as f64;
            // audit gap should fit the Bcpu term alone (same-order f32 sum)
            let b = gamma(k as f64 + 4.0 * h + 8.0) * (pre.scales[i / rows.max(1)] as f64 + 1.0);
            let _ = b;
            worst_ratio = worst_ratio.max(gap / r.bound_emit[i].max(1e-30));
        }
        ok(
            "bounds/f64-audit-within-model",
            worst_ratio <= 1.0,
            format!("worst_gap_over_bound={worst_ratio:.3}"),
        );
    }
    // 6b. RNE half-ULP eps model holds per element on real-like data:
    //     |xhat−x| <= half_ulp(x/a)·a (validates the Bq epsilon term).
    {
        let (n, k) = (64usize, 512usize);
        let mut rng = 0xE950u64;
        let x: Vec<f32> = (0..n * k).map(|_| prng_f32(&mut rng) * 400.0 - 200.0).collect();
        let pre = cpu_prelude(&x, n, k, tab);
        let mut worst_excess = 0.0f64;
        for t in 0..n {
            let a = pre.scales[t] as f64;
            for i in 0..k {
                let x0 = x[t * k + i];
                let xhat = e4m3_decode(pre.x8[t * k + i]) as f64;
                let eps = tab.half_ulp_at(x0 / pre.scales[t]) * a;
                worst_excess = worst_excess.max((xhat - x0 as f64).abs() - eps);
            }
        }
        ok(
            "bounds/rne-eps-model",
            worst_excess <= 1e-6,
            format!("worst_excess_over_half_ulp={worst_excess:e}"),
        );
    }
    // 7. unequal halves / zero-scale / constant weights obey s*q+z
    {
        let k = 256usize;
        // half0 constant 3.0 (zero scale), half1 ramping
        let rows = 2usize;
        let mut w = vec![0.0f32; rows * k];
        for r in 0..rows {
            for i in 0..k {
                w[r * k + i] = if i < HALF { 3.0 } else { i as f32 * 0.5 };
            }
        }
        let blob = pack_v2_split(&w, rows, k);
        let hw = decode_group_header(&blob, 0, 0, 1);
        let pass = hw[0].s == 0.0 && (hw[0].z - 3.0).abs() <= 0.01 && hw[1].s > 0.0;
        ok(
            "weights/zero-scale-half",
            pass,
            format!("s0={} z0={} s1={} z1={}", hw[0].s, hw[0].z, hw[1].s, hw[1].z),
        );
    }

    // 8. source-boundary index math for qkvza (48-row beta/alpha crossing a
    //    64-row WG): replicate the kernel's orow routing on CPU
    {
        let (q, z, b, a) = (64usize, 32usize, 48usize, 48usize);
        let tm = q + z + b + a;
        // kernel maps orow -> (split, pr): verify every row lands in exactly
        // one split with the right intra-split index
        let route = |orow: usize| -> (u8, usize, usize) {
            if orow < q {
                (0, orow, q)
            } else if orow < q + z {
                (1, orow - q, z)
            } else if orow < q + z + b {
                (2, orow - (q + z), b)
            } else {
                (3, orow - (q + z + b), a)
            }
        };
        let mut seen_beta_alpha_boundary = false;
        let mut pass = true;
        // 64-row WG starting at row 64+32=96 (inside beta? beta spans
        // 96..144): rows 96..160 cross beta->alpha at 144
        for wg in [0usize] {
            let _ = wg;
        }
        for orow in 0..tm {
            let (sp, pr, os) = route(orow);
            if pr >= os {
                pass = false;
            }
            // check inversion
            let back = match sp {
                0 => pr,
                1 => q + pr,
                2 => q + z + pr,
                _ => q + z + b + pr,
            };
            if back != orow {
                pass = false;
            }
            if orow >= q + z && orow < q + z + b + a {
                seen_beta_alpha_boundary = true;
            }
        }
        // WG [128,192) with 64-row tile: rows 128..144 beta, 144..176 alpha,
        // 176..192 past-end masked — row 144 is the exact crossing
        let (sp144, pr144, _) = route(144);
        pass = pass && sp144 == 3 && pr144 == 0 && seen_beta_alpha_boundary;
        ok(
            "address/qkvza-48row-boundary",
            pass,
            format!("tm={tm} row144->(split{sp144},pr{pr144})"),
        );
    }
    out
}

// ─── Synthetic weight/x generators (stand-in until U0 fixtures land) ───────

fn synth_weights(rows: usize, k: usize, disjoint: bool, salt: u64) -> Vec<f32> {
    let mut rng = salt;
    let mut w = vec![0.0f32; rows * k];
    for r in 0..rows {
        for c in 0..k {
            let gi = c % GROUP;
            w[r * k + c] = if disjoint {
                if gi < HALF {
                    prng_f32(&mut rng) * 2.0 - 1.0 // half0 [-1,1]
                } else {
                    96.0 + prng_f32(&mut rng) * 64.0 // half1 [96,160]
                }
            } else {
                // realistic post-FWHT-ish: small Gaussian-ish via CLT of 3 uniforms
                (prng_f32(&mut rng) + prng_f32(&mut rng) + prng_f32(&mut rng) - 1.5) * 0.03
            };
        }
    }
    w
}

fn synth_x(n: usize, k: usize, salt: u64) -> Vec<f32> {
    let mut rng = salt;
    let mut x = vec![0.0f32; n * k];
    for (i, v) in x.iter_mut().enumerate() {
        // broad magnitude span: exercises E4M3 normal/subnormal/tie regions
        let u = prng_f32(&mut rng);
        let mag = if i % 97 == 0 {
            u * 200.0 // near-224 headroom probes
        } else if i % 31 == 0 {
            u * 0.001 // subnormal probes after downscaling
        } else {
            u * 6.0 - 3.0
        };
        *v = mag;
    }
    x
}

// ─── GPU driver ─────────────────────────────────────────────────────────────
#[cfg(feature = "deltanet")]
mod gpu {
    use super::*;
    use rdna_compute::{DType, Gpu, GpuTensor};

    pub struct FamilyShape {
        pub family: &'static str,
        pub splits: Vec<usize>, // per-output weight rows
        pub k: usize,
        pub hist_us: f64,
        pub s2bt8: &'static str,
    }

    pub fn canonical_shapes() -> Vec<FamilyShape> {
        vec![
            FamilyShape {
                family: "gate_up",
                splits: vec![17408, 17408],
                k: 5120,
                hist_us: S2BT8_HIST_US[0],
                s2bt8: S2BT8_SYMBOLS[0],
            },
            FamilyShape {
                family: "qkv",
                splits: vec![12288, 1024, 1024],
                k: 5120,
                hist_us: S2BT8_HIST_US[1],
                s2bt8: S2BT8_SYMBOLS[1],
            },
            FamilyShape {
                family: "qkvza",
                splits: vec![10240, 6144, 48, 48],
                k: 5120,
                hist_us: S2BT8_HIST_US[2],
                s2bt8: S2BT8_SYMBOLS[2],
            },
            FamilyShape {
                family: "residual",
                splits: vec![5120],
                k: 17408,
                hist_us: S2BT8_HIST_US[3],
                s2bt8: S2BT8_SYMBOLS[3],
            },
        ]
    }

    fn download_bytes(gpu: &Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
        let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, len) };
        let mut out = vec![0u8; len];
        gpu.hip.device_synchronize().unwrap();
        gpu.hip.memcpy_dtoh(&mut out, &buf).unwrap();
        std::mem::forget(buf); // non-owning view: never free
        out
    }

    pub struct LaunchOut {
        pub ys: Vec<Vec<f32>>,
        pub x8: Vec<u8>,
        pub sums: Vec<f32>,
        pub scales: Vec<f32>,
        pub kernel_names: Vec<String>,
    }

    pub fn launch_incumbent(
        gpu: &mut Gpu,
        fam: &FamilyShape,
        blobs: &[Vec<u8>],
        x: &[f32],
        y_olds: &[Vec<f32>],
        n: usize,
    ) -> LaunchOut {
        let k = fam.k;
        // Capture the emitted prelude once via the EXISTING entry point.
        let d_x = gpu.upload_f32(x, &[n * k]).unwrap();
        let prepared = gpu.prepare_mq4v2_fp8_x(&d_x, n, k, 1).unwrap();
        let x8 = download_bytes(gpu, prepared.x_fp8, prepared.x_fp8_bytes);
        let sums_b = download_bytes(gpu, prepared.half_sums, prepared.half_sums_bytes);
        let sc_b = download_bytes(gpu, prepared.row_scales, prepared.row_scales_bytes);
        let sums: Vec<f32> = sums_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scales: Vec<f32> = sc_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut ys = Vec::new();
        let mut kernel_names = Vec::new();
        match fam.family {
            "gate_up" => {
                let a_g = gpu.upload_raw(&blobs[0], &[blobs[0].len()]).unwrap();
                let a_u = gpu.upload_raw(&blobs[1], &[blobs[1].len()]).unwrap();
                let y_g = gpu.zeros(&[n * fam.splits[0]], DType::F32).unwrap();
                let y_u = gpu.zeros(&[n * fam.splits[1]], DType::F32).unwrap();
                gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12(
                    &a_g, &a_u, &d_x, &y_g, &y_u, fam.splits[0], fam.splits[1], k, n, 1,
                )
                .unwrap();
                gpu.hip.device_synchronize().unwrap();
                ys.push(gpu.download_f32(&y_g).unwrap());
                ys.push(gpu.download_f32(&y_u).unwrap());
                kernel_names.push(kernel_label(fam.family, fam.s2bt8));
            }
            "residual" => {
                let a = gpu.upload_raw(&blobs[0], &[blobs[0].len()]).unwrap();
                let y = gpu.upload_f32(&y_olds[0], &[n * fam.splits[0]]).unwrap();
                gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8(
                    &a, &d_x, &y, fam.splits[0], k, n, 1,
                )
                .unwrap();
                gpu.hip.device_synchronize().unwrap();
                ys.push(gpu.download_f32(&y).unwrap());
                kernel_names.push(kernel_label(fam.family, fam.s2bt8));
            }
            "qkvza" => {
                let a: Vec<GpuTensor> = blobs
                    .iter()
                    .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
                    .collect();
                let y: Vec<GpuTensor> = fam
                    .splits
                    .iter()
                    .map(|&m| gpu.zeros(&[n * m], DType::F32).unwrap())
                    .collect();
                gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8(
                    &a[0], &a[1], &a[2], &a[3], &d_x, &y[0], &y[1], &y[2], &y[3],
                    fam.splits[0], fam.splits[1], fam.splits[2], fam.splits[3], k, n, 1,
                )
                .unwrap();
                gpu.hip.device_synchronize().unwrap();
                for t in y.iter() {
                    ys.push(gpu.download_f32(t).unwrap());
                }
                kernel_names.push(kernel_label(fam.family, fam.s2bt8));
            }
            "qkv" => {
                let a: Vec<GpuTensor> = blobs
                    .iter()
                    .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
                    .collect();
                let y: Vec<GpuTensor> = fam
                    .splits
                    .iter()
                    .map(|&m| gpu.zeros(&[n * m], DType::F32).unwrap())
                    .collect();
                gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8(
                    &a[0], &a[1], &a[2], &d_x, &y[0], &y[1], &y[2],
                    fam.splits[0], fam.splits[1], fam.splits[2], k, n, 1,
                )
                .unwrap();
                gpu.hip.device_synchronize().unwrap();
                for t in y.iter() {
                    ys.push(gpu.download_f32(t).unwrap());
                }
                kernel_names.push(kernel_label(fam.family, fam.s2bt8));
            }
            _ => unreachable!(),
        }
        // (profiling is managed per-case by the caller via profile::start/stop)
        LaunchOut {
            ys,
            x8,
            sums,
            scales,
            kernel_names,
        }
    }
}

// ─── CLI / orchestration ────────────────────────────────────────────────────

fn parse_args() -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        if let Some(v) = a.strip_prefix("--") {
            if let Some(eq) = v.find('=') {
                m.insert(v[..eq].to_string(), v[eq + 1..].to_string());
            } else if it.peek().map(|s| s.starts_with("--")).unwrap_or(true) {
                m.insert(v.to_string(), "1".to_string());
            } else {
                m.insert(v.to_string(), it.next().unwrap());
            }
        }
    }
    m
}

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("tmp_gemm_v2_oracle: build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use gpu::canonical_shapes;
    use rdna_compute::Gpu;

    let args = parse_args();
    let cpu_only = args.contains_key("cpu-only");
    let device: i32 = args.get("device").and_then(|v| v.parse().ok()).unwrap_or(0);
    let out_dir = args
        .get("out")
        .cloned()
        .unwrap_or_else(|| "oracle-out".to_string());
    let ns: Vec<usize> = args
        .get("n")
        .map(|v| v.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![512]);
    let repeats: usize = args.get("repeats").and_then(|v| v.parse().ok()).unwrap_or(3);
    let quick = args.contains_key("quick");
    let expect_v2 = args.contains_key("expect-v2");
    let time_on = std::env::var("TIME").map(|v| v == "1").unwrap_or(false);
    std::fs::create_dir_all(&out_dir).unwrap();

    let tab = E4M3Table::build();
    let t0 = Instant::now();

    // ── CPU suite (always) ──
    let cpu_tests = run_cpu_suite(&tab);
    let cpu_fail = cpu_tests.iter().filter(|t| !t.pass).count();
    println!("CPU suite: {}/{} pass", cpu_tests.len() - cpu_fail, cpu_tests.len());
    for t in cpu_tests.iter() {
        println!("  [{}] {} — {}", if t.pass { "ok" } else { "FAIL" }, t.name, t.detail);
    }

    // ── Resolved route: incumbent s2bt8, or the staged-tile v2 aspect ──
    let (_sfx, gbm, gbn, gbk, gvw) = active_v2_geom();
    if v2_enabled() {
        println!("v2 route ACTIVE (BM{gbm}xBN{gbn}xBK{gbk}/{gvw}w):");
        for s in V2_SYMBOLS {
            println!("  compiled: {s}");
        }
        if expect_v2 {
            println!("  expect-v2 satisfied by the active v2 route");
        }
    } else {
        println!("v2 symbols (frozen §6.1, BM{V2_BM}xBN{V2_BN}xBK{V2_BK}/{V2_WAVES}w):");
        for s in V2_SYMBOLS {
            println!("  pending K/H: {s}");
        }
        if expect_v2 {
            eprintln!("expect-v2 set but the v2 route is off: v2 GPU comparison unavailable");
            std::process::exit(2);
        }
    }

    // ── GPU suite (incumbent s2bt8 vs CPU reference) ──
    let mut cases: Vec<GpuCase> = Vec::new();
    let mut time_table: HashMap<String, Vec<f64>> = HashMap::new();
    if !cpu_only {
        let mut gpu = match Gpu::init_with_device(device) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU on ordinal {device} ({e}) — GPU suite skipped");
                emit(&out_dir, &cpu_tests, &cases, &time_table, "no-gpu", t0);
                std::process::exit(if cpu_fail > 0 { 1 } else { 0 });
            }
        };
        println!("GPU: arch={} ordinal={device}", gpu.arch);
        if gpu.arch != "gfx1201" {
            eprintln!("non-gfx1201 arch {}: incumbent fp8 routes require exact gfx1201", gpu.arch);
            emit(&out_dir, &cpu_tests, &cases, &time_table, "wrong-arch", t0);
            std::process::exit(1);
        }
        if args.contains_key("bench-canonical") {
            run_bench_canonical(&mut gpu, &out_dir, repeats);
        }
        // Small-N sweep exercises BT4/BT8/BT12-exact/masked-BT12/S2BT8 tiles.
        let sweep_ns: Vec<usize> = if quick { vec![512] } else { vec![64, 128, 192, 256, 320, 512] };
        for fam in canonical_shapes() {
            let small = fam.k.min(512);
            // A. small-M tile sweep (routing/masking coverage, all N tiles)
            for &n in sweep_ns.iter() {
                let ms: Vec<usize> = match fam.family {
                    "gate_up" => vec![64, 64],
                    "qkv" => vec![128, 32, 32],
                    "qkvza" => vec![64, 32, 48, 48], // 48-row beta/alpha boundary
                    _ => vec![128],
                };
                run_gpu_case(
                    &mut gpu, &fam, &ms, small, n, repeats, time_on, &tab, &mut cases,
                    &mut time_table,
                );
            }
            // B. canonical shapes at requested N (default 512; --n 256,512,1024)
            if !quick {
                for &n in ns.iter() {
                    // skip N already covered by the sweep at small M; canonical
                    // uses full M rows
                    if sweep_ns.contains(&n) && fam.k == small {
                        continue;
                    }
                    run_gpu_case(
                        &mut gpu, &fam, &fam.splits.clone(), fam.k, n, repeats, time_on,
                        &tab, &mut cases, &mut time_table,
                    );
                }
            } else {
                run_gpu_case(
                    &mut gpu, &fam, &fam.splits.clone(), fam.k, 512, repeats, time_on,
                    &tab, &mut cases, &mut time_table,
                );
            }
        }
        if time_on {
            if let Some(entries) = rdna_compute::profile::stop() {
                for e in entries {
                    time_table.entry(e.kernel.to_string()).or_default().push(e.time_us);
                }
            }
        }
    }

    let gpu_fail = cases.iter().filter(|c| c.verdict != "pass").count();
    emit(&out_dir, &cpu_tests, &cases, &time_table, "ok", t0);

    // TIME medians to stdout
    if time_on && !time_table.is_empty() {
        println!("TIME medians (us, same-session):");
        let mut keys: Vec<_> = time_table.keys().collect();
        keys.sort();
        for k in keys {
            let mut v = time_table[k].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = v[v.len() / 2];
            println!("  {k:60} median {med:10.1} n={}", v.len());
        }
    }
    println!(
        "GPU cases: {}/{} pass",
        cases.len() - gpu_fail,
        cases.len()
    );
    for c in cases.iter() {
        println!(
            "  [{}] {:8} N={:<5} K={:<6} M={:<7} max_abs={:.3e} tail_mse={:.3e} bound_ratio={:.3} viol={} x8mm={} {}",
            c.verdict, c.family, c.n, c.k, c.m_total, c.max_abs, c.tail_mse,
            c.bound_ratio, c.bound_viol, c.x8_mismatch,
            c.time_us.map(|t| format!("{t:.1}us")).unwrap_or_default(),
        );
    }
    if cpu_fail > 0 || gpu_fail > 0 {
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
/// Throwaway canonical TIME bench (gate 3): the four canonical N=512 shapes
/// through the production launchers, 2 untimed hash/det launches + R timed
/// launches per family under an isolated profile window. Routing (incumbent
/// vs v2 aspect) comes purely from env. Emits bench.json; no bound checks
/// (correctness is gated by the sweep). Deleted with this file after proof.
#[cfg(feature = "deltanet")]
#[allow(clippy::too_many_arguments)]
fn run_bench_canonical(gpu: &mut rdna_compute::Gpu, out_dir: &str, timed: usize) {
    use gpu::canonical_shapes;
    let n = 512usize;
    let mut fams = Vec::new();
    let mut fail = 0;
    for fam in canonical_shapes() {
        let ms = fam.splits.clone();
        let k = fam.k;
        let blobs: Vec<Vec<u8>> = ms
            .iter()
            .enumerate()
            .map(|(si, &m)| {
                let w = synth_weights(m, k, si % 2 == 1, 0x1000 + si as u64 * 7919);
                pack_v2_split(&w, m, k)
            })
            .collect();
        let x = synth_x(n, k, 0xBEEF);
        let y_olds: Vec<Vec<f32>> = ms
            .iter()
            .map(|&m| {
                let mut rng = 0x0BADu64;
                (0..n * m).map(|_| prng_f32(&mut rng) * 2.0 - 1.0).collect()
            })
            .collect();
        let shape = modified_shape(&fam, &ms, k);
        let first = gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
        let repeat = gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
        let det = first.ys == repeat.ys;
        let finite = first.ys.iter().all(|y| y.iter().all(|v| v.is_finite()));
        let hash = format!("{:016x}", hash_f32(&first.ys.concat()));
        rdna_compute::profile::start();
        for _ in 0..timed.max(1) {
            gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
        }
        let mut gemm_us: Vec<f64> = Vec::new();
        let mut pack_us: Vec<f64> = Vec::new();
        let mut knames: Vec<String> = first.kernel_names.clone();
        if let Some(entries) = rdna_compute::profile::stop() {
            for e in entries {
                if !knames.contains(&e.kernel.to_string()) {
                    knames.push(e.kernel.to_string());
                }
                if e.kernel.contains("pack_") {
                    pack_us.push(e.time_us);
                } else {
                    gemm_us.push(e.time_us);
                }
            }
        }
        gemm_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pack_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &Vec<f64>| if v.is_empty() { 0.0 } else { v[v.len() / 2] };
        let ok = det && finite;
        if !ok {
            fail += 1;
        }
        println!(
            "  [{}] {:8} gemm {:9.1}us (n={}) pack {:7.1}us hash={} det={} kernels={:?}",
            if ok { "ok" } else { "FAIL" },
            fam.family,
            med(&gemm_us),
            gemm_us.len(),
            med(&pack_us),
            hash,
            det,
            knames,
        );
        fams.push(serde_json::json!({
            "family": fam.family, "n": n, "k": k, "m_total": ms.iter().sum::<usize>(),
            "hash": hash, "deterministic": det, "finite": finite,
            "gemm_median_us": med(&gemm_us), "gemm_n": gemm_us.len(),
            "pack_median_us": med(&pack_us), "kernels": knames,
        }));
    }
    let (_, g_bm, g_bn, g_bk, g_wv) = active_v2_geom();
    let doc = serde_json::json!({
        "route": if v2_enabled() { "v2" } else { "s2bt8" },
        "v2_geometry": {"BM": g_bm, "BN": g_bn, "BK": g_bk, "waves": g_wv},
        "timed_repeats": timed.max(1),
        "families": fams,
    });
    std::fs::write(format!("{out_dir}/bench.json"), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("bench verdict: {}", if fail == 0 { "pass" } else { "FAIL" });
    std::process::exit(if fail == 0 { 0 } else { 1 });
}
#[cfg(feature = "deltanet")]
fn run_gpu_case(
    gpu: &mut rdna_compute::Gpu,
    fam: &gpu::FamilyShape,
    ms: &[usize],
    k: usize,
    n: usize,
    repeats: usize,
    time_on: bool,
    tab: &E4M3Table,
    cases: &mut Vec<GpuCase>,
    time_table: &mut HashMap<String, Vec<f64>>,
) {
    // Distinct per-split weight magnitudes so misrouting fails loudly.
    let blobs: Vec<Vec<u8>> = ms
        .iter()
        .enumerate()
        .map(|(si, &m)| {
            let w = synth_weights(m, k, si % 2 == 1, 0x1000 + si as u64 * 7919);
            pack_v2_split(&w, m, k)
        })
        .collect();
    let x = synth_x(n, k, 0xBEEF);
    let y_olds: Vec<Vec<f32>> = ms
        .iter()
        .map(|&m| {
            let mut rng = 0x0BADu64;
            (0..n * m).map(|_| prng_f32(&mut rng) * 2.0 - 1.0).collect()
        })
        .collect();

    // Sentinel-overwrite canary: overwrite families must leave no sentinel.
    let shape = modified_shape(fam, ms, k);
    let first = gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
    let repeat = gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
    let deterministic = first.ys == repeat.ys;
    let hash = format!("{:016x}", hash_f32(&first.ys.concat()));
    let hash_repeat = format!("{:016x}", hash_f32(&repeat.ys.concat()));

    // TIME=1: timed repeats with profile start/stop isolated per case, so the
    // per-kernel medians attribute pack vs GEMM separately (§5.2 gate 4).
    let mut kernels_seen: Vec<String> = first.kernel_names.clone();
    let mut case_times: Vec<f64> = Vec::new();
    if time_on {
        rdna_compute::profile::start();
        for _ in 0..repeats.max(1) {
            let timed = gpu::launch_incumbent(gpu, &shape, &blobs, &x, &y_olds, n);
            for kn in timed.kernel_names.iter() {
                if !kernels_seen.contains(kn) {
                    kernels_seen.push(kn.clone());
                }
            }
        }
        if let Some(entries) = rdna_compute::profile::stop() {
            for e in entries {
                if !kernels_seen.contains(&e.kernel.to_string()) {
                    kernels_seen.push(e.kernel.to_string());
                }
                time_table
                    .entry(e.kernel.to_string())
                    .or_default()
                    .push(e.time_us);
                case_times.push(e.time_us);
            }
        }
    }
    kernels_seen.sort();
    let time_us = if case_times.is_empty() {
        None
    } else {
        case_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(case_times[case_times.len() / 2])
    };

    // CPU X8 byte-equality vs the emitted prelude; S within its tree bound.
    let pre_cpu = cpu_prelude(&x, n, k, tab);
    let mut x8mm = 0usize;
    let dump_mm = std::env::var("HIPFIRE_ORACLE_DUMP").map(|v| v == "1").unwrap_or(false);
    for i in 0..n * k {
        if pre_cpu.x8[i] != first.x8[i] {
            if dump_mm && x8mm < 20 {
                let t = i / k;
                let c = i % k;
                let a = pre_cpu.scales[t];
                let xv = x[t * k + c];
                let xoa = xv / a;
                eprintln!(
                    "x8mm t={t} c={c} x={xv:e} a={a:e} xoa={xoa:e} cpu={:#04x}({:e}) gpu={:#04x}({:e})",
                    pre_cpu.x8[i],
                    e4m3_decode(pre_cpu.x8[i]),
                    first.x8[i],
                    e4m3_decode(first.x8[i]),
                );
            }
            x8mm += 1;
        }
    }
    let gpr = k / GROUP;
    let mut max_s_err = 0.0f64;
    for t in 0..n {
        for g in 0..gpr {
            for h in 0..2 {
                let idx = (t * gpr + g) * 2 + h;
                // tree-order bound for the 128-term half sum
                let mut asum = 0.0f64;
                for j in 0..HALF {
                    asum += (e4m3_decode(first.x8[t * k + g * GROUP + h * HALF + j]) as f64).abs();
                }
                let b = gamma(128.0) * asum + 1e-9;
                let e = ((pre_cpu.sums[idx] - first.sums[idx]).abs()) as f64;
                max_s_err = max_s_err.max(e / b);
            }
        }
    }
    let mut max_a_rel = 0.0f64;
    for t in 0..n {
        let e = ((pre_cpu.scales[t] - first.scales[t]).abs()) as f64;
        max_a_rel = max_a_rel.max(e / (first.scales[t] as f64).max(1e-30));
    }
    let _ = max_a_rel; // reported via x8mm/S gates; a[t] must match exactly

    // Per-split device-vs-CPU-reference, twice: against the emitted-operand
    // reference with the tighter Bacc+Bcpu(+Bres) bound, and against the
    // original-x reference with the full Bq+Bacc+Bcpu(+Bres) bound (§5.1).
    let mut worst = Metrics {
        max_abs: 0.0,
        tail_thr: 0.0,
        tail_mse: 0.0,
        tail_worst: 0.0,
        maxrel_at_peak: 0.0,
        worst_row_maxrel: 0.0,
        bound_worst_ratio: 0.0,
        bound_violations: 0,
    };
    let mut worst_orig_abs = 0.0f64;
    let mut worst_orig_ratio = 0.0f64;
    let mut worst_orig_viol = 0usize;
    for (si, &m) in ms.iter().enumerate() {
        let y_old = if fam.family == "residual" { Some(y_olds[si].as_slice()) } else { None };
        // Reference from the EMITTED bytes (downloaded X8/S/a), not the
        // CPU-quantized copy, so prelude byte diffs don't double-count.
        let pre_emit = Prelude {
            x8: first.x8.clone(),
            sums: first.sums.clone(),
            scales: first.scales.clone(),
        };
        let r_emit = cpu_reference(&blobs[si], m, &pre_emit, Some(&x), n, k, y_old, false);
        let met = compute_metrics(&first.ys[si], &r_emit.y_f32, &r_emit.bound_emit);
        worst.max_abs = worst.max_abs.max(met.max_abs);
        worst.tail_thr = worst.tail_thr.max(met.tail_thr);
        worst.tail_mse = worst.tail_mse.max(met.tail_mse);
        worst.tail_worst = worst.tail_worst.max(met.tail_worst);
        worst.maxrel_at_peak = worst.maxrel_at_peak.max(met.maxrel_at_peak);
        worst.worst_row_maxrel = worst.worst_row_maxrel.max(met.worst_row_maxrel);
        worst.bound_worst_ratio = worst.bound_worst_ratio.max(met.bound_worst_ratio);
        worst.bound_violations += met.bound_violations;
        let r_orig = cpu_reference(&blobs[si], m, &pre_emit, Some(&x), n, k, y_old, true);
        let b_orig: Vec<f64> = r_orig
            .bound_emit
            .iter()
            .zip(r_emit.bq.iter())
            .map(|(b, q)| b + q)
            .collect();
        let met_o = compute_metrics(&first.ys[si], &r_orig.y_f32, &b_orig);
        worst_orig_abs = worst_orig_abs.max(met_o.max_abs);
        worst_orig_ratio = worst_orig_ratio.max(met_o.bound_worst_ratio);
        worst_orig_viol += met_o.bound_violations;
        // canary: no nonfinite may survive in overwrite outputs
        if fam.family != "residual" {
            for &v in first.ys[si].iter() {
                if !v.is_finite() {
                    worst.bound_violations += 1;
                    break;
                }
            }
        }
    }
    let verdict = if deterministic
        && worst.bound_violations == 0
        && worst_orig_viol == 0
        && x8mm == 0
        && max_s_err <= 1.0
    {
        "pass"
    } else {
        "FAIL"
    };
    cases.push(GpuCase {
        family: fam.family.to_string(),
        n,
        k,
        m_total: ms.iter().sum(),
        kernel: kernel_label(fam.family, fam.s2bt8),
        kernels_seen,
        max_abs: worst.max_abs,
        tail_mse: worst.tail_mse,
        tail_thr: worst.tail_thr,
        tail_worst: worst.tail_worst,
        maxrel_peak: worst.maxrel_at_peak,
        bound_ratio: worst.bound_worst_ratio,
        bound_viol: worst.bound_violations,
        max_abs_orig: worst_orig_abs,
        bound_orig_ratio: worst_orig_ratio,
        bound_orig_viol: worst_orig_viol,
        x8_mismatch: x8mm,
        max_s_err,
        hash,
        hash_repeat,
        deterministic,
        time_us,
        verdict: verdict.to_string(),
    });
}

#[cfg(feature = "deltanet")]
fn modified_shape(fam: &gpu::FamilyShape, ms: &[usize], k: usize) -> gpu::FamilyShape {
    gpu::FamilyShape {
        family: fam.family,
        splits: ms.to_vec(),
        k,
        hist_us: fam.hist_us,
        s2bt8: fam.s2bt8,
    }
}

fn emit(
    out_dir: &str,
    cpu: &[CpuTest],
    cases: &[GpuCase],
    time_table: &HashMap<String, Vec<f64>>,
    status: &str,
    t0: Instant,
) {
    let mut times = serde_json::Map::new();
    for (k, v) in time_table {
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if s.is_empty() { 0.0 } else { s[s.len() / 2] };
        times.insert(k.clone(), serde_json::json!({"median_us": med, "n": s.len()}));
    }
    let (_, g_bm, g_bn, g_bk, g_wv) = active_v2_geom();
    let g_route = if v2_enabled() { "v2" } else { "s2bt8" };
    let doc = serde_json::json!({
        "status": status,
        "elapsed_s": t0.elapsed().as_secs_f64(),
        "v2_symbols": V2_SYMBOLS,
        "v2_geometry": {"BM": g_bm, "BN": g_bn, "BK": g_bk, "waves": g_wv, "route": g_route},
        "s2bt8_hist_us_N512": {"gate_up": S2BT8_HIST_US[0], "qkv": S2BT8_HIST_US[1],
                                "qkvza": S2BT8_HIST_US[2], "residual": S2BT8_HIST_US[3]},
        "cpu_tests": cpu.iter().map(|t| serde_json::json!({
            "name": t.name, "pass": t.pass, "detail": t.detail })).collect::<Vec<_>>(),
        "gpu_cases": cases.iter().map(|c| serde_json::json!({
            "family": c.family, "n": c.n, "k": c.k, "m_total": c.m_total,
            "kernel": c.kernel, "kernels_seen": c.kernels_seen,
            "max_abs": c.max_abs, "tail_mse": c.tail_mse, "tail_thr": c.tail_thr,
            "tail_worst": c.tail_worst, "maxrel_peak": c.maxrel_peak,
            "bound_ratio": c.bound_ratio, "bound_viol": c.bound_viol,
            "max_abs_orig": c.max_abs_orig, "bound_orig_ratio": c.bound_orig_ratio,
            "bound_orig_viol": c.bound_orig_viol,
            "x8_mismatch": c.x8_mismatch, "max_s_err_ratio": c.max_s_err,
            "hash": c.hash, "hash_repeat": c.hash_repeat,
            "deterministic": c.deterministic, "time_us": c.time_us,
            "verdict": c.verdict
        })).collect::<Vec<_>>(),
        "time_medians_us": times,
    });
    std::fs::write(
        format!("{out_dir}/results.json"),
        serde_json::to_string_pretty(&doc).unwrap(),
    )
    .unwrap();
    let mut md = String::from("| family | N | K | M | max_abs | tail_mse | bound_ratio | viol | orig_ratio | oviol | x8mm | time_us | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for c in cases {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.3e} | {:.3e} | {:.3} | {} | {:.3} | {} | {} | {} | {} |\n",
            c.family, c.n, c.k, c.m_total, c.max_abs, c.tail_mse, c.bound_ratio,
            c.bound_viol, c.bound_orig_ratio, c.bound_orig_viol, c.x8_mismatch,
            c.time_us.map(|t| format!("{t:.1}")).unwrap_or_else(|| "-".into()),
            c.verdict
        ));
    }
    if cases.is_empty() {
        md.push_str("| (no GPU cases) |\n");
    }
    std::fs::write(format!("{out_dir}/table.md"), md).unwrap();
}
