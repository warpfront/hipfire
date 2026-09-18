// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ4E8 header-constraint study encoders (slice A).
//!
//! Frozen study contract (`scratch-2026-09-17/Mq4e8Study/composer-contract.txt`
//! plus Main/planner freezes): emit STANDARD qt44 MQ4G256V2 136 B groups —
//! `[f16 s0][f16 z0][f16 s1][f16 z1][128 B paired u4 codes]` — where the scales
//! are constrained per `Mq4HeaderConstraint` and the zeros are refit with
//! fixed-scale Lloyd (scales stay FIXED during Lloyd; no codebook learning).
//! No loader/GEMV change: every emitted scale/zero is exactly f16
//! representable (asserted; failed encodes return Err and must never be
//! published). The default `quantize_mq4g256v2` path is untouched.

use hipfire_quantize::float16::{f16_to_f32, f32_to_f16};
use rayon::prelude::*;

use crate::quant_fwht::{cpu_fwht_256, MQ4V2_GROUP_BYTES};

/// Header constraint selector. CLI: `mq4e8-c0/c1/c2/c3/c4/c4z/c4s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mq4HeaderConstraint {
    /// C0: unconstrained minmax f16 steps + fixed-scale Lloyd (control that
    /// isolates Lloyd's effect from the header restriction).
    Unconstrained,
    /// C1: per-half power-of-two fraction of G=max(s0,s1), d in 0..4, nearest
    /// EXACTLY-f16 candidate (G always valid), tie larger scale.
    HalfPow2,
    /// C2: unsigned m/16 shape — B=f16_trunc(G/16) low-4-significand-bits
    /// clear (floor 0x0010 on positive underflow), m_h in 1..16, s_h=B*m_h.
    HalfInt16,
    /// C3: both half scales = G; zeros stay independently fitted per-128.
    GroupShared,
    /// C4: per-row master B=f16_trunc(rowmax/256) floored at bits 0x0001 on
    /// positive underflow; per-half e in 0..8 nearest, tie larger scale.
    RowPow2,
    /// C4z: C4 scales with B low-4 bits clear (floor 0x0010) + integer zero
    /// multiplier k in [-15,0], z=s*k, Lloyd-refit k.
    RowPow2IntZero,
    /// C4s: EXACT C4 scales/master, fixed k=-8 (z=-8*s), single assignment,
    /// no zero refit.
    RowPow2KMinus8,
}

/// CLI suffix for logs/metadata.
pub(crate) fn mq4e8_suffix(c: Mq4HeaderConstraint) -> &'static str {
    match c {
        Mq4HeaderConstraint::Unconstrained => "c0",
        Mq4HeaderConstraint::HalfPow2 => "c1",
        Mq4HeaderConstraint::HalfInt16 => "c2",
        Mq4HeaderConstraint::GroupShared => "c3",
        Mq4HeaderConstraint::RowPow2 => "c4",
        Mq4HeaderConstraint::RowPow2IntZero => "c4z",
        Mq4HeaderConstraint::RowPow2KMinus8 => "c4s",
    }
}

/// Per-encode diagnostics. Aggregated across groups/rows; the pipeline logs
/// one line per tensor (C2 clamp counts, C4 master floors, zero-master rows).
#[derive(Debug, Default)]
pub(crate) struct Mq4e8Stats {
    pub c2_clamp_lo: u64,
    pub c2_clamp_hi: u64,
    pub master_floor: u64,
    pub zero_master_rows: Vec<u32>,
    pub zero_scale_halves: u64,
}

/// Thread-local counters merged after the parallel pass.
#[derive(Debug, Default, Clone, Copy)]
struct LocalStats {
    c2_clamp_lo: u64,
    c2_clamp_hi: u64,
    master_floor: u64,
    zero_scale_halves: u64,
}

impl Mq4e8Stats {
    fn merge(&mut self, l: LocalStats) {
        self.c2_clamp_lo += l.c2_clamp_lo;
        self.c2_clamp_hi += l.c2_clamp_hi;
        self.master_floor += l.master_floor;
        self.zero_scale_halves += l.zero_scale_halves;
    }
}

const LLOYD_ITERS: usize = 8;
const NEG_POW2: [f32; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
const POS_POW2: [f32; 9] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
/// Full 2^-e table for e 0..8 (single-multiply exact ratio checks).
const NEG_POW2_8: [f32; 9] = [
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
];

/// Single shared assignment DAG (matches `quantize_mq4g256v2` bit-for-bit for
/// the same (v, z, s) triple): q=clamp(floor((v-z)*(1/s)+0.5),0,15), f32.
#[inline]
fn assign_q(v: f32, z: f32, inv: f32) -> u8 {
    ((v - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8
}

fn mean_f64(vals: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &v in vals {
        acc += v as f64;
    }
    acc / vals.len() as f64
}

/// mean(w - s*q) accumulated in f64, index order. The f64 product keeps the
/// residual exact-ish; the caller converts to f32 before f16 truncation.
fn mean_resid_f64(vals: &[f32], s: f32, q: &[u8]) -> f64 {
    let sf = s as f64;
    let mut acc = 0.0f64;
    for (i, &v) in vals.iter().enumerate() {
        acc += v as f64 - sf * q[i] as f64;
    }
    acc / vals.len() as f64
}

/// Unconstrained minmax f16-truncated half steps — IDENTICAL derivation to
/// `quantize_mq4g256v2` (same pinned fixture values): step=(hi-lo)/15,
/// bits=f32_to_f16(step), 0 when hi==lo. Returns (step_bits, step_f32, lo_f32).
fn base_half_steps(g: &[f32; 256]) -> ([u16; 2], [f32; 2], [f32; 2]) {
    let mut bits = [0u16; 2];
    let mut steps = [0.0f32; 2];
    let mut los = [0.0f32; 2];
    for h in 0..2 {
        let off = h * 128;
        let slice = &g[off..off + 128];
        let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        let mut sc_bits = f32_to_f16(step_f32);
        if hi == lo {
            sc_bits = 0;
        }
        bits[h] = sc_bits;
        steps[h] = f16_to_f32(sc_bits);
        los[h] = lo;
    }
    (bits, steps, los)
}

/// C1: per-half nearest EXACTLY-f16 candidate among G*2^-d (d 0..4), tie
/// larger scale. d=0 (G itself, an f16 value) is always exact so the set is
/// never empty. Also returns chosen d per half (signed ratio e=d0-d1 carried
/// by the caller for reporting only).
fn c1_scales(sh: [f32; 2], ctx: &str) -> Result<([u16; 2], [f32; 2], [u32; 2]), String> {
    let g = sh[0].max(sh[1]);
    let mut out_bits = [0u16; 2];
    let mut out_s = [0.0f32; 2];
    let mut out_d = [0u32; 2];
    for h in 0..2 {
        let mut best_d = 0u32;
        let mut best_s = g;
        let mut best_dist = f32::INFINITY;
        let mut found = false;
        for d in 0..=4u32 {
            let cand = g * NEG_POW2[d as usize];
            if f16_to_f32(f32_to_f16(cand)) != cand {
                continue;
            }
            if cand * POS_POW2[d as usize] != g {
                continue;
            }
            let dist = (cand - sh[h]).abs();
            if !found || dist < best_dist || (dist == best_dist && cand > best_s) {
                best_d = d;
                best_s = cand;
                best_dist = dist;
                found = true;
            }
        }
        if !found {
            return Err(format!("{ctx}: C1 half {h}: no exact candidate (G={g:e})"));
        }
        out_bits[h] = f32_to_f16(best_s);
        out_s[h] = best_s;
        out_d[h] = best_d;
    }
    Ok((out_bits, out_s, out_d))
}

/// C2: B=f16_trunc(G/16) low-4-pattern-bits clear (floor 0x0010 on positive
/// underflow); m_h=clamp(floor(sh/B+0.5),1,16); s_h=B*m_h asserted exact f16.
fn c2_scales(
    sh: [f32; 2],
    st: &mut LocalStats,
    ctx: &str,
) -> Result<([u16; 2], [f32; 2]), String> {
    let g = sh[0].max(sh[1]);
    if g == 0.0 {
        return Ok(([0, 0], [0.0, 0.0]));
    }
    let mut b = f32_to_f16(g / 16.0);
    b &= 0xFFF0;
    if b == 0 {
        b = 0x0010;
        st.master_floor += 1;
    }
    let bf = f16_to_f32(b);
    let mut out_bits = [0u16; 2];
    let mut out_s = [0.0f32; 2];
    for h in 0..2 {
        let t = (sh[h] / bf + 0.5).floor();
        if t < 1.0 {
            st.c2_clamp_lo += 1;
        } else if t > 16.0 {
            st.c2_clamp_hi += 1;
        }
        let m = (t as i32).clamp(1, 16);
        let s = bf * m as f32;
        let sb = f32_to_f16(s);
        if f16_to_f32(sb) != s || !s.is_finite() {
            return Err(format!(
                "{ctx}: C2 half {h}: s=B*m not exact f16 (B={bf:e} m={m} s={s:e})"
            ));
        }
        out_bits[h] = sb;
        out_s[h] = s;
    }
    Ok((out_bits, out_s))
}

/// Fixed-scale Lloyd with FREE per-128 zeros (C0..C4): init z=f16(lo); up to
/// 8 assign/refit rounds; stop when both zero bit patterns stabilize; always
/// finish with a final assignment. Zero-scale halves: q=0, z=f16(mean(w)).
fn lloyd_free_zeros(
    g: &[f32; 256],
    s: [f32; 2],
    z_bits: &mut [u16; 2],
    q: &mut [u8; 256],
    st: &mut LocalStats,
) {
    let mut active = [false; 2];
    for h in 0..2 {
        if s[h] == 0.0 {
            for i in 0..128 {
                q[h * 128 + i] = 0;
            }
            z_bits[h] = f32_to_f16(mean_f64(&g[h * 128..h * 128 + 128]) as f32);
            st.zero_scale_halves += 1;
        } else {
            active[h] = true;
        }
    }
    if !active[0] && !active[1] {
        return;
    }
    let inv = [
        if active[0] { 1.0 / s[0] } else { 0.0 },
        if active[1] { 1.0 / s[1] } else { 0.0 },
    ];
    let mut z = [f16_to_f32(z_bits[0]), f16_to_f32(z_bits[1])];
    for _ in 0..LLOYD_ITERS {
        for h in 0..2 {
            if !active[h] {
                continue;
            }
            for i in 0..128 {
                q[h * 128 + i] = assign_q(g[h * 128 + i], z[h], inv[h]);
            }
        }
        let mut nz = *z_bits;
        for h in 0..2 {
            if !active[h] {
                continue;
            }
            nz[h] = f32_to_f16(mean_resid_f64(&g[h * 128..h * 128 + 128], s[h], &q[h * 128..h * 128 + 128]) as f32);
        }
        if nz == *z_bits {
            break;
        }
        *z_bits = nz;
        z = [f16_to_f32(nz[0]), f16_to_f32(nz[1])];
    }
    for h in 0..2 {
        if !active[h] {
            continue;
        }
        for i in 0..128 {
            q[h * 128 + i] = assign_q(g[h * 128 + i], z[h], inv[h]);
        }
    }
}

/// C4z integer-zero Lloyd: k in [-15,0], z=s*k; init k from original minmax
/// zero; up to 8 assign/refit rounds on k; stop when both k stabilize; final
/// assignment. Emitted z bits asserted exactly f16.
fn lloyd_int_zero(
    g: &[f32; 256],
    s: [f32; 2],
    z0_init: [f32; 2],
    z_bits: &mut [u16; 2],
    q: &mut [u8; 256],
    st: &mut LocalStats,
    ctx: &str,
) -> Result<(), String> {
    let mut kk = [0i32; 2];
    let mut active = [false; 2];
    for h in 0..2 {
        if s[h] == 0.0 {
            for i in 0..128 {
                q[h * 128 + i] = 0;
            }
            z_bits[h] = f32_to_f16(0.0);
            st.zero_scale_halves += 1;
        } else {
            active[h] = true;
            kk[h] = ((z0_init[h] / s[h] + 0.5).floor() as i32).clamp(-15, 0);
        }
    }
    if !active[0] && !active[1] {
        return Ok(());
    }
    let inv = [
        if active[0] { 1.0 / s[0] } else { 0.0 },
        if active[1] { 1.0 / s[1] } else { 0.0 },
    ];
    for _ in 0..LLOYD_ITERS {
        let z = [s[0] * kk[0] as f32, s[1] * kk[1] as f32];
        for h in 0..2 {
            if !active[h] {
                continue;
            }
            for i in 0..128 {
                q[h * 128 + i] = assign_q(g[h * 128 + i], z[h], inv[h]);
            }
        }
        let mut nk = kk;
        for h in 0..2 {
            if !active[h] {
                continue;
            }
            let m = mean_resid_f64(&g[h * 128..h * 128 + 128], s[h], &q[h * 128..h * 128 + 128]);
            nk[h] = ((m / s[h] as f64 + 0.5).floor() as i32).clamp(-15, 0);
        }
        if nk == kk {
            break;
        }
        kk = nk;
    }
    for h in 0..2 {
        if !active[h] {
            continue;
        }
        let z = s[h] * kk[h] as f32;
        let zb = f32_to_f16(z);
        if f16_to_f32(zb) != z || !z.is_finite() {
            return Err(format!(
                "{ctx}: C4z half {h}: z=s*k not exact f16 (s={:e} k={} z={:e})",
                s[h], kk[h], z
            ));
        }
        z_bits[h] = zb;
        for i in 0..128 {
            q[h * 128 + i] = assign_q(g[h * 128 + i], z, inv[h]);
        }
    }
    Ok(())
}

/// C4s: fixed k=-8 (z=-8*s), single assignment, no refit. z asserted exact.
fn assign_fixed_kminus8(
    g: &[f32; 256],
    s: [f32; 2],
    z_bits: &mut [u16; 2],
    q: &mut [u8; 256],
    st: &mut LocalStats,
    ctx: &str,
) -> Result<(), String> {
    for h in 0..2 {
        if s[h] == 0.0 {
            for i in 0..128 {
                q[h * 128 + i] = 0;
            }
            z_bits[h] = f32_to_f16(0.0);
            st.zero_scale_halves += 1;
            continue;
        }
        let z = -8.0f32 * s[h];
        let zb = f32_to_f16(z);
        if f16_to_f32(zb) != z || !z.is_finite() {
            return Err(format!(
                "{ctx}: C4s half {h}: z=-8*s not exact f16 (s={:e} z={:e})",
                s[h], z
            ));
        }
        z_bits[h] = zb;
        let inv = 1.0 / s[h];
        for i in 0..128 {
            q[h * 128 + i] = assign_q(g[h * 128 + i], z, inv);
        }
    }
    Ok(())
}

fn emit_group(out: &mut [u8], s_bits: [u16; 2], z_bits: [u16; 2], q: &[u8; 256]) {
    debug_assert_eq!(out.len(), MQ4V2_GROUP_BYTES);
    out[0..2].copy_from_slice(&s_bits[0].to_le_bytes());
    out[2..4].copy_from_slice(&z_bits[0].to_le_bytes());
    out[4..6].copy_from_slice(&s_bits[1].to_le_bytes());
    out[6..8].copy_from_slice(&z_bits[1].to_le_bytes());
    for i in 0..128 {
        out[8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
    }
}

/// Group-local constraints C0..C3.
fn encode_groups(
    w: &[f32],
    n_blocks: usize,
    signs1: &[f32],
    signs2: &[f32],
    constraint: Mq4HeaderConstraint,
    output: &mut [u8],
) -> Result<Mq4e8Stats, String> {
    let suffix = mq4e8_suffix(constraint);
    let locals: Result<Vec<LocalStats>, String> = output
        .par_chunks_mut(MQ4V2_GROUP_BYTES)
        .enumerate()
        .map(|(b, chunk)| {
            let ctx = format!("mq4e8-{suffix} group {b}");
            let mut st = LocalStats::default();
            let mut group = [0.0f32; 256];
            group.copy_from_slice(&w[b * 256..(b + 1) * 256]);
            cpu_fwht_256(&mut group, signs1, signs2);
            let (base_bits, sh, los) = base_half_steps(&group);
            let (s_bits, s) = match constraint {
                Mq4HeaderConstraint::Unconstrained => (base_bits, sh),
                Mq4HeaderConstraint::HalfPow2 => {
                    let (sb, ss, _d) = c1_scales(sh, &ctx)?;
                    (sb, ss)
                }
                Mq4HeaderConstraint::HalfInt16 => c2_scales(sh, &mut st, &ctx)?,
                Mq4HeaderConstraint::GroupShared => {
                    let gb = if sh[1] > sh[0] { base_bits[1] } else { base_bits[0] };
                    let gs = f16_to_f32(gb);
                    ([gb, gb], [gs, gs])
                }
                _ => return Err(format!("{ctx}: row constraint in group path")),
            };
            let mut z_bits = [f32_to_f16(los[0]), f32_to_f16(los[1])];
            let mut q = [0u8; 256];
            lloyd_free_zeros(&group, s, &mut z_bits, &mut q, &mut st);
            emit_group(chunk, s_bits, z_bits, &q);
            Ok(st)
        })
        .collect();
    let mut stats = Mq4e8Stats::default();
    for l in locals? {
        stats.merge(l);
    }
    Ok(stats)
}

/// Per-half exponent choice for row-master constraints: e in 0..8 minimizing
/// |B*2^e - sh|, tie larger resulting scale (smaller e).
fn choose_row_exp(sh: f32, b: f32) -> u32 {
    let mut best_e = 0u32;
    let mut best_s = b;
    let mut best_dist = f32::INFINITY;
    let mut first = true;
    for e in 0..=8u32 {
        let cand = b * POS_POW2[e as usize];
        let dist = (cand - sh).abs();
        if first || dist < best_dist || (dist == best_dist && cand > best_s) {
            best_e = e;
            best_s = cand;
            best_dist = dist;
            first = false;
        }
    }
    best_e
}

/// Row-master B derivation. Returns (bits, value, floored, is_zero_row).
/// C4/C4s: floor positive-underflow at bits 0x0001. C4z: clear low 4 bits,
/// floor positive-underflow at bits 0x0010.
fn row_master_b(rowmax: f32, constraint: Mq4HeaderConstraint, st: &mut LocalStats) -> (u16, f32) {
    if rowmax == 0.0 {
        return (0, 0.0);
    }
    let mut b = f32_to_f16(rowmax / 256.0);
    if constraint == Mq4HeaderConstraint::RowPow2IntZero {
        b &= 0xFFF0;
        if b == 0 {
            b = 0x0010;
            st.master_floor += 1;
        }
    } else if b == 0 {
        b = 0x0001;
        st.master_floor += 1;
    }
    (b, f16_to_f32(b))
}

/// Row-global constraints C4/C4z/C4s. The ENTIRE row of K elements is rotated
/// before the master is derived; K is never split into master regions.
fn encode_rows(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    constraint: Mq4HeaderConstraint,
    output: &mut [u8],
) -> Result<Mq4e8Stats, String> {
    let suffix = mq4e8_suffix(constraint);
    let gpr = k / 256;
    struct RowOut {
        stats: LocalStats,
        zero_row: bool,
        err: Option<String>,
    }
    let rows: Vec<RowOut> = w
        .par_chunks(k)
        .zip(output.par_chunks_mut(gpr * MQ4V2_GROUP_BYTES))
        .enumerate()
        .map(|(r, (wrow, orow))| {
            let ctx = format!("mq4e8-{suffix} row {r}");
            let mut st = LocalStats::default();
            // Pass 1: rotate whole row, per-half steps, row max.
            let mut rot = vec![0.0f32; k];
            let mut sh = vec![[0.0f32; 2]; gpr];
            let mut los = vec![[0.0f32; 2]; gpr];
            let mut rowmax = 0.0f32;
            for c in 0..gpr {
                let mut group = [0.0f32; 256];
                group.copy_from_slice(&wrow[c * 256..(c + 1) * 256]);
                cpu_fwht_256(&mut group, signs1, signs2);
                rot[c * 256..(c + 1) * 256].copy_from_slice(&group);
                let (_bb, ss, ll) = base_half_steps(&group);
                sh[c] = ss;
                los[c] = ll;
                rowmax = rowmax.max(ss[0]).max(ss[1]);
            }
            let (_b_bits, b) = row_master_b(rowmax, constraint, &mut st);
            if b == 0.0 {
                // Zero row: q=0 everywhere; C4 keeps f16(mean) zeros, C4z/C4s
                // use z=0 (s=0 forces the zero-scale rule).
                for c in 0..gpr {
                    let mut group = [0.0f32; 256];
                    group.copy_from_slice(&rot[c * 256..(c + 1) * 256]);
                    let mut z_bits = [0u16; 2];
                    let mut q = [0u8; 256];
                    if constraint == Mq4HeaderConstraint::RowPow2 {
                        let s = [0.0f32; 2];
                        lloyd_free_zeros(&group, s, &mut z_bits, &mut q, &mut st);
                    } else {
                        for h in 0..2 {
                            for i in 0..128 {
                                q[h * 128 + i] = 0;
                            }
                            z_bits[h] = f32_to_f16(0.0);
                            st.zero_scale_halves += 1;
                        }
                    }
                    emit_group(&mut orow[c * MQ4V2_GROUP_BYTES..(c + 1) * MQ4V2_GROUP_BYTES], [0, 0], z_bits, &q);
                }
                return RowOut { stats: st, zero_row: true, err: None };
            }
            // Pass 2: per-half exponents, Lloyd/assign, emit.
            for c in 0..gpr {
                let gctx = format!("{ctx} group {c}");
                let mut group = [0.0f32; 256];
                group.copy_from_slice(&rot[c * 256..(c + 1) * 256]);
                let mut s_bits = [0u16; 2];
                let mut s = [0.0f32; 2];
                for h in 0..2 {
                    let e = choose_row_exp(sh[c][h], b);
                    let cand = b * POS_POW2[e as usize];
                    let sb = f32_to_f16(cand);
                    // Exactness: finite, f16 round-trip clean, and the integer
                    // ratio holds (s * 2^-e == B, single f32 multiply — exact
                    // because power-of-two scaling of a representable value
                    // cannot round when the result is representable).
                    if !cand.is_finite()
                        || f16_to_f32(sb) != cand
                        || cand * NEG_POW2_8[e as usize] != b
                    {
                        return RowOut {
                            stats: st,
                            zero_row: false,
                            err: Some(format!("{gctx}: row-master scale inexact (B={b:e} e={e} s={cand:e})")),
                        };
                    }
                    s_bits[h] = sb;
                    s[h] = cand;
                }
                let mut z_bits = [f32_to_f16(los[c][0]), f32_to_f16(los[c][1])];
                let mut q = [0u8; 256];
                match constraint {
                    Mq4HeaderConstraint::RowPow2 => {
                        lloyd_free_zeros(&group, s, &mut z_bits, &mut q, &mut st);
                    }
                    Mq4HeaderConstraint::RowPow2IntZero => {
                        let z0 = [f16_to_f32(z_bits[0]), f16_to_f32(z_bits[1])];
                        if let Err(e) = lloyd_int_zero(&group, s, z0, &mut z_bits, &mut q, &mut st, &gctx) {
                            return RowOut { stats: st, zero_row: false, err: Some(e) };
                        }
                    }
                    Mq4HeaderConstraint::RowPow2KMinus8 => {
                        if let Err(e) = assign_fixed_kminus8(&group, s, &mut z_bits, &mut q, &mut st, &gctx) {
                            return RowOut { stats: st, zero_row: false, err: Some(e) };
                        }
                    }
                    _ => {
                        return RowOut {
                            stats: st,
                            zero_row: false,
                            err: Some(format!("{gctx}: group constraint in row path")),
                        };
                    }
                }
                emit_group(&mut orow[c * MQ4V2_GROUP_BYTES..(c + 1) * MQ4V2_GROUP_BYTES], s_bits, z_bits, &q);
            }
            RowOut { stats: st, zero_row: false, err: None }
        })
        .collect();
    let mut stats = Mq4e8Stats::default();
    for (r, ro) in rows.iter().enumerate() {
        if let Some(e) = &ro.err {
            return Err(format!("row {r}: {e}"));
        }
        stats.merge(ro.stats);
        if ro.zero_row {
            stats.zero_master_rows.push(r as u32);
        }
    }
    Ok(stats)
}

/// Constrained MQ4G256V2 encoder: standard 136 B groups, constrained scales,
/// fixed-scale Lloyd zeros. `w` is row-major `[m, k]` (AWQ pre-scale, if any,
/// already applied); K % 256 required. Deterministic under rayon.
pub(crate) fn quantize_mq4g256v2_constrained(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    constraint: Mq4HeaderConstraint,
) -> Result<(Vec<u8>, Mq4e8Stats), String> {
    if k % 256 != 0 {
        return Err(format!("mq4e8-{}: K%256 != 0 (K={k})", mq4e8_suffix(constraint)));
    }
    if m == 0 || k == 0 {
        return Err(format!("mq4e8-{}: empty tensor (m={m} k={k})", mq4e8_suffix(constraint)));
    }
    if w.len() != m * k {
        return Err(format!(
            "mq4e8-{}: buffer size {} != m*k {}*{}",
            mq4e8_suffix(constraint),
            w.len(),
            m,
            k
        ));
    }
    if signs1.len() < 256 || signs2.len() < 256 {
        return Err(format!("mq4e8-{}: sign tables < 256", mq4e8_suffix(constraint)));
    }
    let n_blocks = m * k / 256;
    let mut output = vec![0u8; n_blocks * MQ4V2_GROUP_BYTES];
    let stats = match constraint {
        Mq4HeaderConstraint::Unconstrained
        | Mq4HeaderConstraint::HalfPow2
        | Mq4HeaderConstraint::HalfInt16
        | Mq4HeaderConstraint::GroupShared => {
            encode_groups(w, n_blocks, signs1, signs2, constraint, &mut output)
        }
        Mq4HeaderConstraint::RowPow2
        | Mq4HeaderConstraint::RowPow2IntZero
        | Mq4HeaderConstraint::RowPow2KMinus8 => {
            encode_rows(w, m, k, signs1, signs2, constraint, &mut output)
        }
    }?;
    Ok((output, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_fwht::{gen_fwht_signs, quantize_mq4g256v2};

    fn xorshift_weights(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(m * k);
        for _ in 0..m * k {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s as f64) / (u64::MAX as f64);
            let v = if u > 0.995 { (u - 0.5) * 40.0 } else { (u * 2.0 - 1.0) * 3.0 };
            out.push(v as f32);
        }
        out
    }

    fn decode_group(out: &[u8]) -> ([f32; 2], [f32; 2], [u8; 256]) {
        let s = [
            f16_to_f32(u16::from_le_bytes([out[0], out[1]])),
            f16_to_f32(u16::from_le_bytes([out[4], out[5]])),
        ];
        let z = [
            f16_to_f32(u16::from_le_bytes([out[2], out[3]])),
            f16_to_f32(u16::from_le_bytes([out[6], out[7]])),
        ];
        let mut q = [0u8; 256];
        for i in 0..128 {
            q[2 * i] = out[8 + i] & 0xF;
            q[2 * i + 1] = (out[8 + i] >> 4) & 0xF;
        }
        (s, z, q)
    }

    /// Gate 2 tripwire: the DEFAULT path bytes for a fixed synthetic input are
    /// pinned (fnv1a64 of the pre-change capture `quant-a/byteid_pre.bin`).
    /// Anyone touching `quantize_mq4g256v2` breaks this test. The full
    /// byte-identity proof (byteid_pre.bin vs byteid_post.bin md5) lives in
    /// the gate-2 evidence log; this test also refreshes byteid_post.bin.
    #[test]
    fn default_path_bytes_stable() {
        let m = 4usize;
        let k = 512usize;
        let w = xorshift_weights(m, k, 0x1234_5678_9abc_def0);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let bytes = quantize_mq4g256v2(&w, m, k, &s1, &s2);
        assert_eq!(bytes.len(), m * k / 256 * 136);
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in &bytes {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        assert_eq!(h, 0x3728_c9de_e30d_9aea, "default mq4v2 bytes changed!");
        // Refresh the post-change capture where the evidence dir exists;
        // failure is ignored (hash assert above is the real guard).
        let _ = std::fs::write(
            "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/byteid_post.bin",
            &bytes,
        );
    }

    /// Gate 1 parity oracle: planner's independent re-derivation published
    /// input (48x5120 f32, AWQ applied, pre-FWHT) + expected .qt44 bytes for
    /// every constraint. My encoder must match byte-for-byte. Skips (not
    /// fails) when the study evidence files are absent.
    #[test]
    fn parity_oracle_l0_in_proj_a() {
        let root = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study";
        let in_path = format!("{root}/oracle_L0_in_proj_a_awq_f32.bin");
        let Ok(raw) = std::fs::read(&in_path) else {
            eprintln!("parity_oracle: input missing, skipping");
            return;
        };
        assert_eq!(raw.len(), 48 * 5120 * 4, "oracle input size");
        let mut w = Vec::with_capacity(48 * 5120);
        for chunk in raw.chunks_exact(4) {
            w.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let cases = [
            (Mq4HeaderConstraint::Unconstrained, "C0"),
            (Mq4HeaderConstraint::HalfPow2, "C1"),
            (Mq4HeaderConstraint::HalfInt16, "C2"),
            (Mq4HeaderConstraint::GroupShared, "C3"),
            (Mq4HeaderConstraint::RowPow2, "C4"),
            (Mq4HeaderConstraint::RowPow2IntZero, "C4z"),
            (Mq4HeaderConstraint::RowPow2KMinus8, "C4s"),
        ];
        for (c, tag) in cases {
            let exp_path = format!("{root}/oracle_L0_in_proj_a_{tag}.qt44");
            let Ok(expected) = std::fs::read(&exp_path) else {
                eprintln!("parity_oracle: {tag} expected file missing, skipping");
                continue;
            };
            let (mine, st) = quantize_mq4g256v2_constrained(&w, 48, 5120, &s1, &s2, c)
                .unwrap_or_else(|e| panic!("{tag} encode failed: {e}"));
            if mine != expected {
                let ng = mine.len() / 136;
                let mut diff_groups = 0usize;
                let mut first = None;
                for g in 0..ng {
                    if mine[g * 136..(g + 1) * 136] != expected[g * 136..(g + 1) * 136] {
                        diff_groups += 1;
                        if first.is_none() {
                            first = Some(g);
                        }
                    }
                }
                panic!(
                    "{tag} mismatch: {diff_groups}/{ng} groups differ, first group {first:?}; \
                     stats={{clamp_lo={} clamp_hi={} floor={} zerorows={:?} zscale={}}}",
                    st.c2_clamp_lo, st.c2_clamp_hi, st.master_floor,
                    st.zero_master_rows, st.zero_scale_halves
                );
            }
            eprintln!(
                "parity_oracle {tag}: MATCH ({} groups, clamp {}/{} floor {} zerorows {} zscale {})",
                mine.len() / 136,
                st.c2_clamp_lo, st.c2_clamp_hi, st.master_floor,
                st.zero_master_rows.len(), st.zero_scale_halves
            );
        }
    }

    /// Scale-relation exactness on every group for all constraints (synthetic).
    #[test]
    fn constrained_scale_relations_hold() {
        let m = 8usize;
        let k = 1024usize;
        let w = xorshift_weights(m, k, 777);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let all = [
            Mq4HeaderConstraint::Unconstrained,
            Mq4HeaderConstraint::HalfPow2,
            Mq4HeaderConstraint::HalfInt16,
            Mq4HeaderConstraint::GroupShared,
            Mq4HeaderConstraint::RowPow2,
            Mq4HeaderConstraint::RowPow2IntZero,
            Mq4HeaderConstraint::RowPow2KMinus8,
        ];
        for c in all {
            let (out, _st) = quantize_mq4g256v2_constrained(&w, m, k, &s1, &s2, c)
                .unwrap_or_else(|e| panic!("{c:?} failed: {e}"));
            assert_eq!(out.len(), m * k / 256 * 136);
            for b in 0..m * k / 256 {
                let (s, z, q) = decode_group(&out[b * 136..(b + 1) * 136]);
                for h in 0..2 {
                    // every emitted scale/zero round-trips f16 exactly
                    let sb = f32_to_f16(s[h]);
                    assert_eq!(f16_to_f32(sb), s[h], "{c:?} group {b} half {h} scale");
                    let zb = f32_to_f16(z[h]);
                    assert_eq!(f16_to_f32(zb), z[h], "{c:?} group {b} half {h} zero");
                    assert!(s[h] >= 0.0);
                    for i in 0..128 {
                        assert!(q[h * 128 + i] < 16);
                    }
                }
                match c {
                    Mq4HeaderConstraint::GroupShared => {
                        assert_eq!(s[0], s[1], "C3 scales must match");
                    }
                    Mq4HeaderConstraint::RowPow2KMinus8 => {
                        if s[0] > 0.0 {
                            assert_eq!(z[0], -8.0 * s[0], "C4s z=-8s");
                        }
                        if s[1] > 0.0 {
                            assert_eq!(z[1], -8.0 * s[1], "C4s z=-8s");
                        }
                    }
                    Mq4HeaderConstraint::RowPow2IntZero => {
                        for h in 0..2 {
                            if s[h] > 0.0 {
                                // z = s*k for some integer k in [-15,0]
                                let kf = z[h] / s[h];
                                assert!(
                                    kf >= -15.0 && kf <= 0.0 && kf == kf.floor(),
                                    "C4z z/s must be integer in [-15,0], got {kf}"
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// C2 clamp counters fire on a spiky input; C1 keeps integer half-ratios.
    #[test]
    fn c1_ratio_and_c2_clamps() {
        let m = 2usize;
        let k = 256usize;
        let mut w = xorshift_weights(m, k, 31337);
        // spike one coefficient to force a wide half vs narrow half
        w[0] = 50.0;
        w[300] = -40.0;
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let (out1, _) =
            quantize_mq4g256v2_constrained(&w, m, k, &s1, &s2, Mq4HeaderConstraint::HalfPow2).unwrap();
        for b in 0..m * k / 256 {
            let (s, _, _) = decode_group(&out1[b * 136..(b + 1) * 136]);
            if s[0] > 0.0 && s[1] > 0.0 {
                let (big, small) = if s[0] >= s[1] { (s[0], s[1]) } else { (s[1], s[0]) };
                let mut ok = false;
                for d in 0..=4u32 {
                    if small * POS_POW2[d as usize] == big {
                        ok = true;
                        break;
                    }
                }
                assert!(ok, "C1 halves must be integer-ratio related: {s:?}");
            }
        }
        let (out2, st2) =
            quantize_mq4g256v2_constrained(&w, m, k, &s1, &s2, Mq4HeaderConstraint::HalfInt16).unwrap();
        for b in 0..m * k / 256 {
            let (s, _, _) = decode_group(&out2[b * 136..(b + 1) * 136]);
            for h in 0..2 {
                assert_eq!(f16_to_f32(f32_to_f16(s[h])), s[h]);
            }
        }
        eprintln!("c2 clamps lo={} hi={}", st2.c2_clamp_lo, st2.c2_clamp_hi);
    }

    /// Rejects non-study shapes instead of silently emitting.
    #[test]
    fn rejects_unaligned_k() {
        let w = xorshift_weights(2, 100, 5);
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        assert!(quantize_mq4g256v2_constrained(&w, 2, 100, &s1, &s2, Mq4HeaderConstraint::HalfPow2).is_err());
    }

    /// Study helper: dump the EXACT f32 AWQ scales the requant recipe applies
    /// per qt44 tensor (same condition as the dispatch: AWQ_ALPHA +
    /// imatrix_weights_for + awq_eligible), for exact gate-4 references.
    /// Input census comes from the fixture index (quant-a/qt44_names.json).
    /// Skips when study files are absent. Output: quant-a/awq_scales.bin
    /// {u32 n, per record: u16 namelen, name, u32 m, u32 k, u8 awq,
    /// [u32 K, K x f32 le] if awq}.
    #[test]
    fn dump_awq_reference_scales() {
        use crate::calibration::{
            awq_eligible, compute_awq_scales, imatrix_weights_for, load_imatrix,
            AWQ_ALPHA, IMATRIX,
        };
        let root = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a";
        let Ok(names_json) = std::fs::read_to_string(format!("{root}/qt44_names.json")) else {
            eprintln!("dump_awq: census missing, skipping");
            return;
        };
        let im_path = "/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf";
        if !std::path::Path::new(im_path).exists() {
            eprintln!("dump_awq: imatrix missing, skipping");
            return;
        }
        let names: Vec<(String, usize, usize)> =
            serde_json::from_str(&names_json).expect("census json");
        assert_eq!(names.len(), 497, "qt44 census size");
        let _ = IMATRIX.set(load_imatrix(std::path::Path::new(im_path)));
        let _ = AWQ_ALPHA.set(0.55f32);
        let out_path = format!("{root}/awq_scales.bin");
        let mut buf = Vec::new();
        buf.extend_from_slice(&(names.len() as u32).to_le_bytes());
        let mut n_awq = 0usize;
        for (name, m, k) in &names {
            let nb = name.as_bytes();
            buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            buf.extend_from_slice(nb);
            buf.extend_from_slice(&(*m as u32).to_le_bytes());
            buf.extend_from_slice(&(*k as u32).to_le_bytes());
            let scales = match (AWQ_ALPHA.get().copied(), imatrix_weights_for(name)) {
                (Some(a), Some(w)) if awq_eligible(name) => {
                    let s = compute_awq_scales(w, a);
                    assert_eq!(s.len(), *k, "imatrix len != K for {name}");
                    Some(s)
                }
                _ => None,
            };
            match scales {
                Some(s) => {
                    n_awq += 1;
                    buf.push(1u8);
                    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    for &v in &s {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                None => buf.push(0u8),
            }
        }
        std::fs::write(&out_path, &buf).expect("write scales dump");
        eprintln!("dump_awq: {} tensors, {} AWQ -> {out_path}", names.len(), n_awq);
    }
}
