// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ4G256V2L (qt=52): MQ4v2 wire layout + per-tensor 16-level Lloyd-Max codebook.
//!
//! Wire bytes are byte-identical in layout to MQ4G256V2 (qt=44): 136 B/group with
//! the same per-128 fp16 `(sc,zp)` affine header. Codes index a per-tensor
//! codebook `L[0..16)` in normalized `[0,15]` units so dequant is
//! `w = sc · L[q] + zp`. The codebook itself is NOT in the weight blob — the
//! pipeline emits it as an F32 sidecar `<stem>.lloyd_levels.weight` shape `[16]`.

use crate::lloyd_gpu::{
    gpu_lloyd_enabled, gpu_lloyd_free_fit, gpu_lloyd_polish_init, gpu_lloyd_polish_scores,
};
use crate::quant_fwht::{cpu_fwht_256, MQ4V2_GROUP_BYTES};
use hipfire_quantize::float16::{f16_to_f32, f32_to_f16};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Max Lloyd-Max iterations (probe used 25; early-stop at max|ΔL| < 1e-6).
pub(crate) const LLOYD_V2_MAX_ITERS: usize = 25;
/// Target pool size for the per-tensor fit (stride-subsample when larger).
pub(crate) const LLOYD_V2_POOL_TARGET: usize = 2_000_000;

type Mq4v2BlockHeader = ([u16; 2], [u16; 2], [f32; 2], [f32; 2], [bool; 2]);

/// Per-128 affine headers of a rotated 256-group, shared by the collect
/// passes so the degenerate rule cannot drift between them.
#[inline]
fn mq4v2_block_header(g: &[f32; 256]) -> Mq4v2BlockHeader {
    let mut scales = [0u16; 2];
    let mut zeros = [0u16; 2];
    let mut sts = [0.0f32; 2];
    let mut zs = [0.0f32; 2];
    let mut degenerate = [false; 2];
    for h in 0..2 {
        let off = h * 128;
        let slice = &g[off..off + 128];
        let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        let mut sc_bits = f32_to_f16(step_f32);
        let z_bits = f32_to_f16(lo);
        if hi == lo {
            sc_bits = 0;
        }
        scales[h] = sc_bits;
        zeros[h] = z_bits;
        let st = f16_to_f32(sc_bits);
        let z = f16_to_f32(z_bits);
        sts[h] = st;
        zs[h] = z;
        degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
    }
    (scales, zeros, sts, zs, degenerate)
}

/// Result of a per-tensor MQ4v2-Lloyd encode.
#[derive(Clone, Debug)]
pub(crate) struct Mq4v2LloydResult {
    pub data: Vec<u8>,
    /// E4M3-constrained Lloyd-Max levels in normalized [0,15] units (stored).
    pub levels: [f32; 16],
    /// Weighted MSE of the uniform grid `0..15` (weight = sc² per half).
    pub mse_uniform: f64,
    /// Weighted MSE of the unconstrained Lloyd codebook (report only).
    pub mse_lloyd_free: f64,
    /// Weighted MSE of the E4M3-constrained Lloyd codebook (stored/used).
    pub mse_lloyd: f64,
    /// Σ sc² over evaluated values (for family aggregation).
    pub weight_sum: f64,
}

/// Running per-family weighted-MSE accumulator for the end-of-run table.
#[derive(Default, Debug)]
struct FamilyAccum {
    n_tensors: usize,
    w_sum: f64,
    sse_unif: f64,
    sse_free: f64,
    sse_lloyd: f64,
}

static FAMILY_STATS: Mutex<Option<BTreeMap<&'static str, FamilyAccum>>> = Mutex::new(None);

/// Reset family stats (call once at start of a quantize run).
pub(crate) fn lloyd_v2_stats_reset() {
    let mut g = FAMILY_STATS.lock();
    *g = Some(BTreeMap::new());
}

/// Record one tensor's weighted MSE into the family table.
pub(crate) fn lloyd_v2_stats_record(name: &str, r: &Mq4v2LloydResult) {
    let fam = lloyd_family_of(name);
    let mut g = FAMILY_STATS.lock();
    let map = g.get_or_insert_with(BTreeMap::new);
    let e = map.entry(fam).or_default();
    e.n_tensors += 1;
    e.w_sum += r.weight_sum;
    e.sse_unif += r.mse_uniform * r.weight_sum;
    e.sse_free += r.mse_lloyd_free * r.weight_sum;
    e.sse_lloyd += r.mse_lloyd * r.weight_sum;
}

/// Print the per-family weighted-RMS table (uniform / free / constrained).
/// Gate uses constrained gain vs uniform (default floor: 15% MSE).
pub(crate) fn lloyd_v2_stats_print(min_gain_pct: f64) -> bool {
    let g = FAMILY_STATS.lock();
    let Some(map) = g.as_ref() else {
        eprintln!("(no MQ4V2-Lloyd family stats)");
        return false;
    };
    if map.is_empty() {
        eprintln!("(no MQ4V2-Lloyd tensors)");
        return false;
    }
    eprintln!(
        "=== MQ4V2-Lloyd weighted-RMS by family (weight-space, rotated; constrained=E4M3) ==="
    );
    eprintln!(
        "  {:<10} {:>7} {:>11} {:>11} {:>11} {:>8} {:>8} {:>9}",
        "family", "tensors", "rms_unif", "rms_free", "rms_e4m3", "free%", "e4m3%", "wsum"
    );
    let mut ok = true;
    let mut tot = FamilyAccum::default();
    for (fam, e) in map.iter() {
        if e.w_sum <= 0.0 || e.n_tensors == 0 {
            continue;
        }
        let mu = e.sse_unif / e.w_sum;
        let mf = e.sse_free / e.w_sum;
        let ml = e.sse_lloyd / e.w_sum;
        let gain_f = if mu > 0.0 {
            100.0 * (1.0 - mf / mu)
        } else {
            0.0
        };
        let gain_l = if mu > 0.0 {
            100.0 * (1.0 - ml / mu)
        } else {
            0.0
        };
        eprintln!(
            "  {fam:<10} {:>7} {:>11.4e} {:>11.4e} {:>11.4e} {:>8.2} {:>8.2} {:>9.3e}",
            e.n_tensors,
            mu.sqrt(),
            mf.sqrt(),
            ml.sqrt(),
            gain_f,
            gain_l,
            e.w_sum
        );
        if gain_l < min_gain_pct {
            ok = false;
        }
        tot.n_tensors += e.n_tensors;
        tot.w_sum += e.w_sum;
        tot.sse_unif += e.sse_unif;
        tot.sse_free += e.sse_free;
        tot.sse_lloyd += e.sse_lloyd;
    }
    if tot.w_sum > 0.0 {
        let mu = tot.sse_unif / tot.w_sum;
        let mf = tot.sse_free / tot.w_sum;
        let ml = tot.sse_lloyd / tot.w_sum;
        let gain_f = if mu > 0.0 {
            100.0 * (1.0 - mf / mu)
        } else {
            0.0
        };
        let gain_l = if mu > 0.0 {
            100.0 * (1.0 - ml / mu)
        } else {
            0.0
        };
        eprintln!(
            "  {:<10} {:>7} {:>11.4e} {:>11.4e} {:>11.4e} {:>8.2} {:>8.2} {:>9.3e}",
            "TOTAL",
            tot.n_tensors,
            mu.sqrt(),
            mf.sqrt(),
            ml.sqrt(),
            gain_f,
            gain_l,
            tot.w_sum
        );
        if gain_l < min_gain_pct {
            ok = false;
        }
    }
    ok
}

fn lloyd_family_of(name: &str) -> &'static str {
    let n = name;
    if n.contains("mlp.down_proj") || (n.contains("w2") && n.contains("down")) {
        "mlp_down"
    } else if n.contains("mlp.gate_proj") || n.contains("gate_proj") {
        "mlp_gate"
    } else if n.contains("mlp.up_proj") || n.contains("up_proj") {
        "mlp_up"
    } else if n.contains("o_proj") || n.contains("out_proj") {
        "attn_o"
    } else if n.contains("q_proj")
        || n.contains("k_proj")
        || n.contains("v_proj")
        || n.contains("qkv")
        || n.contains("in_proj")
    {
        "attn_qkv"
    } else if n.contains("linear_attn") || n.contains("ssm") || n.contains("mamba") {
        "ssm"
    } else {
        "other"
    }
}

/// Assign `v` to the nearest of 16 sorted levels via midpoints.
#[inline]
pub(crate) fn assign_level_midpoints(v: f64, levels: &[f64; 16]) -> usize {
    // searchsorted on mids = (L[i]+L[i+1])/2
    for i in 0..15 {
        let mid = 0.5 * (levels[i] + levels[i + 1]);
        if v < mid {
            return i;
        }
    }
    15
}

/// Weighted MSE of assigning `vals` to `levels` with weights `w`.
pub(crate) fn weighted_mse(vals: &[f64], w: &[f64], levels: &[f64; 16]) -> f64 {
    debug_assert_eq!(vals.len(), w.len());
    let mut sse = 0.0f64;
    let mut wsum = 0.0f64;
    for (&v, &wt) in vals.iter().zip(w.iter()) {
        let idx = assign_level_midpoints(v, levels);
        let e = v - levels[idx];
        sse += wt * e * e;
        wsum += wt;
    }
    if wsum > 0.0 {
        sse / wsum
    } else {
        0.0
    }
}

/// Score four candidate codebooks in one element pass. Each accumulator is
/// updated in the same element order as `weighted_mse`.
fn weighted_mse_4(vals: &[f64], w: &[f64], levels: &[[f64; 16]; 4]) -> [f64; 4] {
    debug_assert_eq!(vals.len(), w.len());
    let mut sse = [0.0f64; 4];
    let mut wsum = [0.0f64; 4];
    for (&v, &wt) in vals.iter().zip(w.iter()) {
        for c in 0..4 {
            let lvl = &levels[c];
            let idx = assign_level_midpoints(v, lvl);
            let e = v - lvl[idx];
            sse[c] += wt * e * e;
            wsum[c] += wt;
        }
    }
    let mut out = [0.0f64; 4];
    for c in 0..4 {
        out[c] = if wsum[c] > 0.0 { sse[c] / wsum[c] } else { 0.0 };
    }
    out
}

/// Finite E4M3 (bias=7) values with -8 ≤ e < 8, sorted unique.
/// Includes ±0 once. Domain S = { 7.5 + e }.
/// E4M3: normals sign·2^(exp−7)·(1+m/8) exp=1..14 m=0..7;
/// subnormals exp=0: sign·m·2^−9 (m=1..7). Skip exp=15 (NaN/Inf codes).
fn e4m3_finite_neg8_to_8() -> Vec<f64> {
    let mut out = Vec::with_capacity(160);
    out.push(0.0);
    // Subnormals: exp=0 → ± m · 2^{-9}, m=1..7
    for m in 1u32..=7 {
        let v = (m as f64) * 2f64.powi(-9);
        out.push(v);
        out.push(-v);
    }
    // Normals: exp=1..14, m=0..7; keep -8 ≤ v < 8
    for exp in 1i32..=14 {
        let scale = 2f64.powi(exp - 7);
        for m in 0u32..=7 {
            let v = scale * (1.0 + (m as f64) / 8.0);
            if v < 8.0 {
                out.push(v);
                if -v >= -8.0 {
                    out.push(-v);
                }
            }
            // include e = -8 exactly if representable
            if (v - 8.0).abs() < 1e-15 {
                // 8.0 itself is excluded (e < 8); -8.0 is included
            }
        }
    }
    // Force-include -8.0 if any E4M3 value equals 8.0 under abs (exp=10,m=0 → 8.0)
    // Contract: -8 <= e < 8, so e=-8 is in S.
    out.push(-8.0);
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    out
}

/// Sorted unique S = {7.5 + e : e ∈ E4M3, -8 ≤ e < 8}. Cached.
fn lloyd_e4m3_representable_r() -> &'static [f64] {
    use std::sync::LazyLock;
    static R: LazyLock<Vec<f64>> = LazyLock::new(|| {
        e4m3_finite_neg8_to_8()
            .into_iter()
            .map(|e| 7.5 + e)
            .collect()
    });
    &R
}

/// Snap `x` to nearest value in sorted `r`. Ties go UP (toward +∞).
#[inline]
fn snap_to_r(x: f64, r: &[f64]) -> f64 {
    if r.is_empty() {
        return x;
    }
    match r.binary_search_by(|p| p.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => r[i],
        Err(i) => {
            if i == 0 {
                r[0]
            } else if i >= r.len() {
                r[r.len() - 1]
            } else {
                let lo = r[i - 1];
                let hi = r[i];
                let dlo = (x - lo).abs();
                let dhi = (hi - x).abs();
                // ties up → prefer hi when equal distance
                if dlo < dhi {
                    lo
                } else {
                    hi
                }
            }
        }
    }
}

/// Index of `x` in sorted `r` (exact match within 1e-12), or None.
fn r_index_of(x: f64, r: &[f64]) -> Option<usize> {
    r.iter().position(|&rv| (rv - x).abs() < 1e-12)
}

/// Snap all 16 levels to R, re-sort, keep 16 distinct by walking right on R.
fn snap_levels_to_r(levels: &mut [f64; 16], r: &[f64]) {
    for l in levels.iter_mut() {
        *l = snap_to_r(*l, r);
    }
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for i in 1..16 {
        if levels[i] <= levels[i - 1] + 1e-15 {
            let prev = levels[i - 1];
            let mut nxt = None;
            for &cand in r {
                if cand > prev + 1e-15 {
                    nxt = Some(cand);
                    break;
                }
            }
            levels[i] = nxt.unwrap_or(prev);
        }
    }
}

/// Unconstrained Lloyd-Max fit (report / oracle). Init = uniform 0..15.
///
/// Returns `(levels, per-iter MSE)` where `mse_hist[0]` is the uniform-grid MSE
/// (iteration 0 assignment, before any centroid move).
pub(crate) fn lloyd_max_fit(vals: &[f64], w: &[f64], max_iters: usize) -> ([f64; 16], Vec<f64>) {
    debug_assert_eq!(vals.len(), w.len());
    let mut levels = [0.0f64; 16];
    for i in 0..16 {
        levels[i] = i as f64;
    }
    let mut mse_hist = Vec::with_capacity(max_iters.saturating_add(1));

    if vals.is_empty() {
        mse_hist.push(0.0);
        return (levels, mse_hist);
    }

    for _it in 0..max_iters {
        let mut sw = [0.0f64; 16];
        let mut sv = [0.0f64; 16];
        let mut sse = 0.0f64;
        let mut wsum = 0.0f64;
        for (&v, &wt) in vals.iter().zip(w.iter()) {
            let idx = assign_level_midpoints(v, &levels);
            sw[idx] += wt;
            sv[idx] += wt * v;
            let e = v - levels[idx];
            sse += wt * e * e;
            wsum += wt;
        }
        mse_hist.push(if wsum > 0.0 { sse / wsum } else { 0.0 });

        let mut new_levels = levels;
        for k in 0..16 {
            if sw[k] > 0.0 {
                new_levels[k] = sv[k] / sw[k];
            }
        }
        new_levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut moved = 0.0f64;
        for k in 0..16 {
            moved = moved.max((new_levels[k] - levels[k]).abs());
        }
        levels = new_levels;
        if moved < 1e-6 {
            break;
        }
    }
    (levels, mse_hist)
}

/// One assign→centroid→snap Lloyd step on the E4M3 grid. Returns moved amount.
fn constrained_lloyd_step(vals: &[f64], w: &[f64], levels: &mut [f64; 16], r: &[f64]) -> f64 {
    let mut sw = [0.0f64; 16];
    let mut sv = [0.0f64; 16];
    for (&v, &wt) in vals.iter().zip(w.iter()) {
        let idx = assign_level_midpoints(v, levels);
        sw[idx] += wt;
        sv[idx] += wt * v;
    }
    let mut new_levels = *levels;
    for k in 0..16 {
        if sw[k] > 0.0 {
            new_levels[k] = sv[k] / sw[k];
        }
    }
    snap_levels_to_r(&mut new_levels, r);
    let mut moved = 0.0f64;
    for k in 0..16 {
        moved = moved.max((new_levels[k] - levels[k]).abs());
    }
    *levels = new_levels;
    moved
}

/// Coordinate-descent polish: for each level try ±1..±2 grid neighbors,
/// accept any strict WMSE improvement, re-sort, keep 16 distinct. Repeat
/// until a full pass makes no change (capped; ≤256k polish samples).
fn coordinate_descent_polish(vals: &[f64], w: &[f64], levels: &mut [f64; 16], r: &[f64]) {
    if vals.is_empty() || r.len() < 16 {
        return;
    }
    // Cap polish cost: stride-subsample to ≤256k if the fit pool is huge.
    let target = 256_000usize;
    let stride = if vals.len() <= target {
        1
    } else {
        (vals.len() + target - 1) / target
    };
    let (pvals, pwts): (Vec<f64>, Vec<f64>) = if stride <= 1 {
        (vals.to_vec(), w.to_vec())
    } else {
        let mut pv = Vec::with_capacity((vals.len() + stride - 1) / stride);
        let mut pw = Vec::with_capacity(pv.capacity());
        let mut i = 0usize;
        while i < vals.len() {
            pv.push(vals[i]);
            pw.push(w[i]);
            i += stride;
        }
        (pv, pw)
    };

    if gpu_lloyd_enabled() {
        gpu_lloyd_polish_init(&pvals, &pwts).unwrap_or_else(|e| {
            eprintln!("error: HIPFIRE_QUANT_GPU=1 polish init failed: {e}");
            std::process::exit(2);
        });
    }
    let mut mse = weighted_mse(&pvals, &pwts, levels);
    for _pass in 0..16 {
        let mut improved = false;
        for i in 0..16 {
            let Some(idx0) = r_index_of(levels[i], r) else {
                continue;
            };
            // Build all four candidates first; this is pure level math and
            // cannot change which one wins.
            let mut cands = [None, None, None, None];
            for (ci, &d) in [-2i32, -1, 1, 2].iter().enumerate() {
                let j = idx0 as i32 + d;
                if j < 0 || j as usize >= r.len() {
                    continue;
                }
                let mut cand = *levels;
                cand[i] = r[j as usize];
                cand.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mut ok = true;
                for k in 1..16 {
                    if cand[k] <= cand[k - 1] + 1e-15 {
                        let prev = cand[k - 1];
                        let mut nxt = None;
                        for &c in r {
                            if c > prev + 1e-15 {
                                nxt = Some(c);
                                break;
                            }
                        }
                        match nxt {
                            Some(v) => cand[k] = v,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if !ok {
                    continue;
                }
                for k in 1..16 {
                    if cand[k] <= cand[k - 1] + 1e-15 {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cands[ci] = Some(cand);
                }
            }
            let mut set = [*levels, *levels, *levels, *levels];
            for (ci, c) in cands.iter().enumerate() {
                if let Some(cand) = c {
                    set[ci] = *cand;
                }
            }
            let scores = if gpu_lloyd_enabled() {
                // Four distinct candidates in one launch.
                gpu_lloyd_polish_scores(&set).unwrap_or_else(|e| {
                    eprintln!("error: HIPFIRE_QUANT_GPU=1 polish scoring failed: {e}");
                    std::process::exit(2);
                })
            } else {
                weighted_mse_4(&pvals, &pwts, &set)
            };
            for ci in 0..4 {
                if cands[ci].is_none() {
                    continue;
                }
                let m = scores[ci];
                if m + 1e-15 < mse {
                    *levels = set[ci];
                    mse = m;
                    improved = true;
                    break; // next level
                }
            }
        }
        if !improved {
            break;
        }
    }
}

/// E4M3-constrained Lloyd-Max fit (production codebook).
///
/// 1. Fit unconstrained Lloyd (uniform init).
/// 2. Snap every level onto S = {e+7.5 : E4M3, -8≤e<8}.
/// 3. Warm-start assign→centroid→snap iterations from the snapped exact
///    solution (cold start from uniform traps at integers).
/// 4. Coordinate-descent polish (±2 grid neighbors).
///
/// Returns `(snapped_levels, MSE history of warm-start phase)`.
/// E4M3-constrained Lloyd-Max fit, reusing an unconstrained fit the caller
/// already computed. Recomputing the free fit here was ~28% of total quantize
/// time (P0 profile); the production path already needs it for its MSE table.
pub(crate) fn lloyd_max_fit_e4m3_from_free(
    vals: &[f64],
    w: &[f64],
    free: [f64; 16],
    uniform_mse: f64,
    max_iters: usize,
) -> ([f64; 16], Vec<f64>) {
    debug_assert_eq!(vals.len(), w.len());
    let r = lloyd_e4m3_representable_r();
    let mut mse_hist = Vec::with_capacity(27);
    mse_hist.push(uniform_mse); // uniform reference (iteration 0 of the free fit)

    // (2) snap exact solution
    let mut levels = free;
    snap_levels_to_r(&mut levels, &r);

    if vals.is_empty() {
        return (levels, mse_hist);
    }

    let mut prev_mse = weighted_mse(vals, w, &levels);
    mse_hist.push(prev_mse);

    // (3) warm-start constrained Lloyd
    for _it in 0..max_iters {
        let before = levels;
        let moved = constrained_lloyd_step(vals, w, &mut levels, &r);
        let new_mse = weighted_mse(vals, w, &levels);
        if new_mse > prev_mse + 1e-15 {
            levels = before; // reject MSE-raising snap
            break;
        }
        mse_hist.push(new_mse);
        prev_mse = new_mse;
        if moved < 1e-15 {
            break;
        }
    }
    // (4) coordinate-descent polish
    coordinate_descent_polish(vals, w, &mut levels, &r);
    let final_mse = weighted_mse(vals, w, &levels);
    if mse_hist.last().copied().unwrap_or(f64::INFINITY) > final_mse + 1e-15 {
        mse_hist.push(final_mse);
    }

    (levels, mse_hist)
}

#[cfg(test)]
/// True iff every level equals some value in S (within 1e-12).
pub(crate) fn levels_in_e4m3_r(levels: &[f32; 16]) -> bool {
    let r = lloyd_e4m3_representable_r();
    levels.iter().all(|&l| {
        let x = l as f64;
        r.iter().any(|&rv| (rv - x).abs() < 1e-12)
    })
}

/// Stride-subsample indices so the pool is ~`target` elements (or all if smaller).
fn stride_subsample_len(n: usize, target: usize) -> usize {
    if n <= target {
        1
    } else {
        // ceil(n / target)
        (n + target - 1) / target
    }
}

/// Quantize F32 weights to MQ4G256V2L: same affine header as `quantize_mq4g256v2`,
/// codes by nearest Lloyd level, plus the fitted per-tensor codebook.
///
/// `w` is row-major `[m, k]` (K % 256 == 0 preferred; trailing partial groups
/// padded with zeros like v2). AWQ pre-scale, if any, must already be applied.
pub(crate) fn quantize_mq4g256v2_lloyd(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Mq4v2LloydResult {
    let _ = (m, k);
    let group_size = 256usize;
    let block_bytes = MQ4V2_GROUP_BYTES;
    let n = w.len();
    let n_blocks = (n + group_size - 1) / group_size;
    // ── Pass 1: FWHT + affine headers; collect only the ~2M fit pool ──
    // The report table scores the fit subsample, so the full pool is no
    // longer needed. Two phases: count non-degenerate halves per block,
    // then collect only stride-selected values using the exact global index
    // the old concatenated pool would have assigned them.
    let (fit_vals, fit_wts): (Vec<f64>, Vec<f64>) = {
        let counts: Vec<u8> = (0..n_blocks)
            .into_par_iter()
            .map(|b| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let mut group = [0.0f32; 256];
                group[..end - start].copy_from_slice(&w[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let (_, _, _, _, degenerate) = mq4v2_block_header(&group);
                degenerate.iter().filter(|&&d| !d).count() as u8
            })
            .collect();
        let mut starts = Vec::with_capacity(n_blocks.saturating_add(1));
        let mut halves = 0usize;
        starts.push(0usize);
        for &c in &counts {
            halves += c as usize;
            starts.push(halves * 128);
        }
        let pool_len = halves * 128;
        let stride = stride_subsample_len(pool_len, LLOYD_V2_POOL_TARGET);
        let block_pools: Vec<(Vec<f64>, Vec<f64>)> = (0..n_blocks)
            .into_par_iter()
            .map(|b| {
                let start = b * group_size;
                let end = (start + group_size).min(n);
                let mut group = [0.0f32; 256];
                group[..end - start].copy_from_slice(&w[start..end]);
                cpu_fwht_256(&mut group, signs1, signs2);
                let (_, _, sts, zs, degenerate) = mq4v2_block_header(&group);
                let mut vals = Vec::new();
                let mut wts = Vec::new();
                let mut seen = 0usize;
                for h in 0..2 {
                    if degenerate[h] {
                        continue;
                    }
                    let off = h * 128;
                    let inv = 1.0f32 / sts[h];
                    let wh = (sts[h] as f64) * (sts[h] as f64);
                    for i in 0..128 {
                        let gidx = starts[b] + seen * 128 + i;
                        if gidx % stride == 0 {
                            vals.push(((group[off + i] - zs[h]) * inv) as f64);
                            wts.push(wh);
                        }
                    }
                    seen += 1;
                }
                (vals, wts)
            })
            .collect();
        let mut fv = Vec::with_capacity(pool_len.div_ceil(stride) + 256);
        let mut fw = Vec::with_capacity(fv.capacity());
        for (v, wt) in block_pools {
            fv.extend(v);
            fw.extend(wt);
        }
        (fv, fw)
    };

    // Unconstrained fit (report) + E4M3-constrained fit (stored / encode).
    let (levels_free, free_hist) = if gpu_lloyd_enabled() {
        gpu_lloyd_free_fit(&fit_vals, &fit_wts).unwrap_or_else(|e| {
            eprintln!("error: HIPFIRE_QUANT_GPU=1 failed: {e}");
            std::process::exit(2);
        })
    } else {
        lloyd_max_fit(&fit_vals, &fit_wts, LLOYD_V2_MAX_ITERS)
    };
    let (levels_f64, _) = lloyd_max_fit_e4m3_from_free(
        &fit_vals,
        &fit_wts,
        levels_free,
        free_hist.first().copied().unwrap_or(0.0),
        LLOYD_V2_MAX_ITERS,
    );
    let mut levels_f32 = [0.0f32; 16];
    for i in 0..16 {
        levels_f32[i] = levels_f64[i] as f32;
    }

    // Full-pool MSE (uniform / free / constrained) for the family table.
    let uniform = {
        let mut u = [0.0f64; 16];
        for i in 0..16 {
            u[i] = i as f64;
        }
        u
    };
    // Report-only family table. Score the same ~2M subsample the codebook
    // was fit on: the full pool reaches 1.27B f64 pairs (20 GB) on 27B and
    // scanning it three times was 43% of total time (P0). Weight bytes are
    // unaffected; the family table moves <0.01pp and the 12% gate is intact.
    let mse_uniform = weighted_mse(&fit_vals, &fit_wts, &uniform);
    let mse_lloyd_free = weighted_mse(&fit_vals, &fit_wts, &levels_free);
    let mse_lloyd = weighted_mse(&fit_vals, &fit_wts, &levels_f64);
    let weight_sum: f64 = fit_wts.iter().sum();

    // ── Pass 2: encode with Lloyd codes, headers identical to v2 ──
    let mut output = vec![0u8; n_blocks * block_bytes];
    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let mut group = [0.0f32; 256];
            let actual_len = end - start;
            group[..actual_len].copy_from_slice(&w[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            let mut scales = [0u16; 2];
            let mut zeros = [0u16; 2];
            let mut sts = [0.0f32; 2];
            let mut zs = [0.0f32; 2];
            let mut degenerate = [false; 2];
            for h in 0..2 {
                let off = h * 128;
                let slice = &group[off..off + 128];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let mut sc_bits = f32_to_f16(step_f32);
                let z_bits = f32_to_f16(lo);
                if hi == lo {
                    sc_bits = 0;
                }
                scales[h] = sc_bits;
                zeros[h] = z_bits;
                let st = f16_to_f32(sc_bits);
                let z = f16_to_f32(z_bits);
                sts[h] = st;
                zs[h] = z;
                degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
            }
            out_chunk[0..2].copy_from_slice(&scales[0].to_le_bytes());
            out_chunk[2..4].copy_from_slice(&zeros[0].to_le_bytes());
            out_chunk[4..6].copy_from_slice(&scales[1].to_le_bytes());
            out_chunk[6..8].copy_from_slice(&zeros[1].to_le_bytes());

            let mut q = [0u8; 256];
            for h in 0..2 {
                let off = h * 128;
                if degenerate[h] {
                    for i in 0..128 {
                        q[off + i] = 0;
                    }
                } else {
                    let st = sts[h];
                    let z = zs[h];
                    let inv = 1.0 / st;
                    for i in 0..128 {
                        let v = ((group[off + i] - z) * inv) as f64;
                        q[off + i] = assign_level_midpoints(v, &levels_f64) as u8;
                    }
                }
            }
            for i in 0..128 {
                let lo_q = q[2 * i];
                let hi_q = q[2 * i + 1];
                out_chunk[8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4);
            }
        });

    Mq4v2LloydResult {
        data: output,
        levels: levels_f32,
        mse_uniform,
        mse_lloyd_free,
        mse_lloyd,
        weight_sum,
    }
}

/// Pack `levels` as little-endian f32[16] bytes for the HFQ sidecar.
pub(crate) fn lloyd_levels_to_f32_bytes(levels: &[f32; 16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    for &v in levels {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Sidecar tensor name for a weight tensor (mirrors AWQ strip rule).
pub(crate) fn lloyd_levels_sidecar_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.lloyd_levels.weight"),
        None => format!("{weight_name}.lloyd_levels.weight"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::{write_hfq, HfqTensor, QuantType};
    use crate::quant_fwht::{gen_fwht_signs, quantize_mq4g256v2};
    // (intentionally empty — tempfile via std::env::temp_dir)

    fn synthetic_weights(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(m * k);
        for _ in 0..m * k {
            // xorshift
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s as f64) / (u64::MAX as f64);
            // mixture: mostly N(0,1)-ish via Box-Muller-lite + occasional spikes
            let g = (u * 2.0 - 1.0) * 1.5;
            out.push(g as f32);
        }
        out
    }

    /// Local stream-(2) research only. Dumps a full-fit fixture:
    /// `u64 n`, `f64[16] expected levels`, `f64[n] vals`, `f64[n] wts`.
    #[test]
    #[ignore = "local GPU research: LLOYD_FIT_FIXTURE=<path> cargo test --release -p hipfire-quantize --bin hipfire-quantize -- --ignored lloyd_fit_fixture --nocapture"]
    fn lloyd_fit_fixture() {
        let Ok(path) = std::env::var("LLOYD_FIT_FIXTURE") else {
            eprintln!("set LLOYD_FIT_FIXTURE to write the fixture");
            return;
        };
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let w = synthetic_weights(64, 32768, 12345);
        let mut vals = Vec::new();
        let mut wts = Vec::new();
        for group in w.chunks_exact(256) {
            let mut g = [0.0f32; 256];
            g.copy_from_slice(group);
            cpu_fwht_256(&mut g, &s1, &s2);
            let (_, _, sts, zs, degenerate) = mq4v2_block_header(&g);
            for h in 0..2 {
                if degenerate[h] {
                    continue;
                }
                let inv = 1.0 / sts[h];
                let wh = (sts[h] as f64) * (sts[h] as f64);
                for i in 0..128 {
                    vals.push(((g[h * 128 + i] - zs[h]) * inv) as f64);
                    wts.push(wh);
                }
            }
        }
        let (levels, _) = lloyd_max_fit(&vals, &wts, LLOYD_V2_MAX_ITERS);
        let mut bytes = Vec::with_capacity(8 + 16 * 8 + vals.len() * 16);
        bytes.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in &levels {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &wts {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {path}: n={} levels={levels:?}", vals.len());
    }

    #[test]
    fn iteration_zero_matches_uniform_mse_exactly() {
        let mut vals = Vec::new();
        let mut wts = Vec::new();
        // Deterministic values in [0,15]
        for i in 0..10_000 {
            let v = (i as f64 * 0.00173) % 15.0;
            vals.push(v);
            wts.push(0.5 + (i % 7) as f64 * 0.1);
        }
        let (levels, hist) = lloyd_max_fit(&vals, &wts, 25);
        assert!(!hist.is_empty());
        let mut uniform = [0.0f64; 16];
        for i in 0..16 {
            uniform[i] = i as f64;
        }
        let mse_u = weighted_mse(&vals, &wts, &uniform);
        assert!(
            (hist[0] - mse_u).abs() < 1e-15,
            "iter0 mse {} != uniform {}",
            hist[0],
            mse_u
        );
        // levels should have moved off the grid for this non-uniform density
        let moved: f64 = levels
            .iter()
            .enumerate()
            .map(|(i, &l)| (l - i as f64).abs())
            .sum();
        assert!(moved > 1e-3, "levels did not move: {levels:?}");
    }

    #[test]
    fn mse_is_non_increasing_per_iteration() {
        let mut vals = Vec::new();
        let mut wts = Vec::new();
        for i in 0..50_000 {
            // heavy-tailed: more mass near 0 and 15
            let t = (i as f64) / 50_000.0;
            let v = if t < 0.5 {
                t * t * 15.0
            } else {
                15.0 - (1.0 - t) * (1.0 - t) * 15.0
            };
            vals.push(v);
            wts.push(1.0 + (i % 3) as f64);
        }
        let (_levels, hist) = lloyd_max_fit(&vals, &wts, 25);
        for w in hist.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "MSE increased: {} -> {} (hist={hist:?})",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn constrained_levels_are_in_e4m3_r() {
        let mut vals = Vec::new();
        let mut wts = Vec::new();
        for i in 0..20_000 {
            let v = (i as f64 * 0.000791) % 15.0;
            vals.push(v);
            wts.push(1.0 + (i % 5) as f64 * 0.2);
        }
        let (free, free_hist) = lloyd_max_fit(&vals, &wts, 25);
        let (levels, hist) = lloyd_max_fit_e4m3_from_free(
            &vals,
            &wts,
            free,
            free_hist.first().copied().unwrap_or(0.0),
            25,
        );
        let mut lf = [0.0f32; 16];
        for i in 0..16 {
            lf[i] = levels[i] as f32;
        }
        assert!(levels_in_e4m3_r(&lf), "levels not in R: {levels:?}");
        // MSE of successive snapped codebooks is non-increasing (we stop if it
        // would rise, so the recorded hist is monotone).
        for w in hist.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "constrained MSE rose: {} -> {} hist={hist:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn quantize_stores_e4m3_levels_and_beats_uniform() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let w = synthetic_weights(16, 512, 99);
        let r = quantize_mq4g256v2_lloyd(&w, 16, 512, &s1, &s2);
        assert!(
            levels_in_e4m3_r(&r.levels),
            "stored levels not in R: {:?}",
            r.levels
        );
        assert!(
            r.mse_lloyd <= r.mse_uniform + 1e-12,
            "constrained {} > unif {}",
            r.mse_lloyd,
            r.mse_uniform
        );
        assert!(
            r.mse_lloyd_free <= r.mse_uniform + 1e-12,
            "free {} > unif {}",
            r.mse_lloyd_free,
            r.mse_uniform
        );
    }

    #[test]
    fn metadata_round_trip_lloyd_levels_sidecar() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let w = synthetic_weights(4, 256, 42);
        let r = quantize_mq4g256v2_lloyd(&w, 4, 256, &s1, &s2);
        assert_eq!(r.data.len(), 4 * MQ4V2_GROUP_BYTES);

        let dir = std::env::temp_dir().join(format!("mq4v2l_meta_rt_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hfq");

        let levels_bytes = lloyd_levels_to_f32_bytes(&r.levels);
        let tensors = vec![
            HfqTensor {
                name: "layer.weight".into(),
                quant_type: QuantType::MQ4G256V2L,
                shape: vec![4, 256],
                group_size: 256,
                data: r.data.clone(),
                spilled_len: 0,
            },
            HfqTensor {
                name: lloyd_levels_sidecar_name("layer.weight"),
                quant_type: QuantType::F32,
                shape: vec![16],
                group_size: 0,
                data: levels_bytes.clone(),
                spilled_len: 0,
            },
        ];
        write_hfq(&path, 5, r#"{"test":true}"#, &tensors, None).unwrap();

        // Re-read: locate sidecar by scanning the index + data region.
        let file = std::fs::read(&path).unwrap();
        assert_eq!(&file[0..4], b"HFQM");
        let data_off = u64::from_le_bytes(file[24..32].try_into().unwrap()) as usize;
        // Brace-scan metadata
        let mut depth = 0i32;
        let mut json_end = 0usize;
        let mut in_str = false;
        let mut esc = false;
        for (i, &b) in file[32..data_off].iter().enumerate() {
            let c = b as char;
            if in_str {
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    in_str = false;
                }
            } else if c == '"' {
                in_str = true;
            } else if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
        let mut pos = 32 + json_end;
        let n_t = u32::from_le_bytes(file[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut off = data_off;
        let mut found = false;
        for _ in 0..n_t {
            let nlen = u16::from_le_bytes(file[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = std::str::from_utf8(&file[pos..pos + nlen]).unwrap();
            pos += nlen;
            let qt = file[pos];
            pos += 1;
            let nd = file[pos] as usize;
            pos += 1;
            let mut shape = Vec::new();
            for _ in 0..nd {
                shape.push(u32::from_le_bytes(file[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            let _gs = u32::from_le_bytes(file[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let dsz = u64::from_le_bytes(file[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if name == "layer.lloyd_levels.weight" {
                assert_eq!(qt, QuantType::F32 as u8);
                assert_eq!(shape, vec![16]);
                assert_eq!(dsz, 64);
                let blob = &file[off..off + dsz];
                assert_eq!(blob, levels_bytes.as_slice());
                let mut back = [0.0f32; 16];
                for i in 0..16 {
                    back[i] = f32::from_le_bytes(blob[i * 4..i * 4 + 4].try_into().unwrap());
                }
                assert_eq!(back, r.levels);
                found = true;
            }
            off += dsz;
        }
        assert!(found, "lloyd_levels sidecar missing from round-trip HFQ");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn v2_encoder_bytes_unchanged_by_lloyd_module() {
        // Guard: loading/compiling the Lloyd path must not alter quantize_mq4g256v2.
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let w = synthetic_weights(8, 512, 7);
        let a = quantize_mq4g256v2(&w, 8, 512, &s1, &s2);
        let b = quantize_mq4g256v2(&w, 8, 512, &s1, &s2);
        assert_eq!(a, b);
        // Lloyd output is same length as v2 (no prefix).
        let r = quantize_mq4g256v2_lloyd(&w, 8, 512, &s1, &s2);
        assert_eq!(r.data.len(), a.len());
        // Headers (first 8 bytes of each group) must match v2 exactly.
        let n_blocks = a.len() / MQ4V2_GROUP_BYTES;
        for g in 0..n_blocks {
            let o = g * MQ4V2_GROUP_BYTES;
            assert_eq!(
                &r.data[o..o + 8],
                &a[o..o + 8],
                "affine header mismatch at group {g}"
            );
        }
        // Lloyd MSE must beat uniform on this synthetic tensor.
        assert!(
            r.mse_lloyd <= r.mse_uniform + 1e-12,
            "lloyd {} > unif {}",
            r.mse_lloyd,
            r.mse_uniform
        );
    }

    #[test]
    fn qt52_identity() {
        assert_eq!(QuantType::MQ4G256V2L as u8, 52);
        assert_eq!(QuantType::from_u8(52), Some(QuantType::MQ4G256V2L));
    }
}
