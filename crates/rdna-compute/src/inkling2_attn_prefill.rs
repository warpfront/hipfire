// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire -- see LICENSE and NOTICE in the project root.

//! Batched multi-row prefill attention (F1 kernel #3, beta port): K-block-once
//! multi-query attention over the prompt block, F16 K/V only.
//!
//! Ported from the stack's `inkling2_attn_prefill` for the beta upstream PR:
//! the Q8_0 twin (brief F1Q) is omitted -- beta has no Q8 KV path (brief 15
//! not on beta), so the F16 twin is the only batched attention route here.
//! Beta-rel-bias rule: out-of-extent distances CLAMP to the last table
//! column, exactly like beta's `inkling2_rel_bias_gather_f32` (NOT the
//! stack's zero-term rule, which belongs to the separate rel-bias PR).
//!
//! * [`prefill_key_plan`] -- the absolute-position key schedule one workgroup
//!   streams (shared by the host twin and the HIP kernel below).
//! * [`qtile_attn_reference`] -- host twin of the kernel's exact op order
//!   (legacy tree-reduction order, phantom-sink init, rel-bias r-order and
//!   clamp past extent, per-query masking by skip). Bitwise-equal to the
//!   single-row seam by construction; the unit tests below pin that.
//! * [`INKLING2_ATTN_PREFILL_HIP_SRC`] -- the kernel source (grid
//!   `(nqtile, 8)`, block 128, QT=16 queries x 3 Q-heads per workgroup, one
//!   K/V row load per position step in LDS).

/// Queries per workgroup (one KV head's 3 Q-heads x QT queries share each
/// K/V row load).
pub const PREFILL_QT: usize = 16;

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::HipResult;

/// Absolute-position key schedule for one query tile.
///
/// Query `b` (absolute position `p = base_pos + b`, ring `rows`) attends the
/// `n_kv = min(p + 1, rows)` trailing positions `t in [lo_b, p]` with
/// `lo_b = p + 1 - n_kv`, slot `t % rows`, distance `p - t`. The workgroup
/// streams the union `t in [t_start, t_end)` once; query `b` skips `t`
/// outside its window with zero ops (bitwise-transparent masking: an
/// inactive step must not touch `(m, l, o)`).
///
/// Returns `(t_start, t_end, windows)` with `windows[b] = (lo_b, hi_b)`
/// (`hi_b = p + 1`, exclusive) for `b in 0..b_rows`.
pub fn prefill_key_plan(
    base_pos: usize,
    b_rows: usize,
    rows: usize,
) -> (usize, usize, Vec<(usize, usize)>) {
    assert!(b_rows > 0 && rows > 0, "inkling2 attn prefill: empty plan");
    let mut windows = Vec::with_capacity(b_rows);
    let mut t_start = usize::MAX;
    let mut t_end = 0usize;
    for b in 0..b_rows {
        let p = base_pos + b;
        let n_kv = (p + 1).min(rows);
        assert!(n_kv > 0, "inkling2 attn prefill: empty key window");
        let lo = p + 1 - n_kv;
        let hi = p + 1;
        t_start = t_start.min(lo);
        t_end = t_end.max(hi);
        windows.push((lo, hi));
    }
    (t_start, t_end, windows)
}

/// Legacy-order 128-wide dot product: per-lane multiply then the exact
/// pairwise tree of `inkling2_gf_attn_decode_f32` (`s = 64, 32, ..., 1`,
/// `sh[d] += sh[d + s]` for `d < s`). Syncs do not affect values, so the
/// host transcription drops them but keeps the add order.
fn tree_dot128(q: &[f32], k: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), 128);
    debug_assert_eq!(k.len(), 128);
    let mut sh = [0.0f32; 128];
    for d in 0..128 {
        sh[d] = q[d] * k[d];
    }
    let mut s = 64usize;
    while s >= 1 {
        for d in 0..s {
            sh[d] += sh[d + s];
        }
        s /= 2;
    }
    sh[0]
}

/// In-register relative bias for one `(query, head, distance)`, mirroring
/// beta's `inkling2_rel_bias_gather_f32` exactly: ascending-`r` accumulation,
/// out-of-extent distances clamp to the last table column, then subtract
/// `sink[h]`.
fn rel_bias_reg(rel_bh: &[f32], table: &[f32], sink_bh: f32, dist: usize, extent: usize) -> f32 {
    debug_assert_eq!(rel_bh.len(), 16);
    let d = if dist < extent { dist } else { extent - 1 };
    let mut v = 0.0f32;
    for r in 0..16 {
        v += table[r * extent + d] * rel_bh[r];
    }
    v - sink_bh
}

/// One query row of single-row seam attention (transcription of
/// `inkling2_gf_attn_decode_f32` op order): phantom-sink init `m = 0, l = 1`,
/// ascending-`t` online softmax, `out = o / l`.
///
/// `q_b`: `[24, 128]`, `rel_b`: `[24, 16]`, `sink_b`: `[24]`,
/// `k_cache`/`v_cache`: 8 heads x `[rows, 128]` host f32 rows.
#[allow(clippy::too_many_arguments)]
pub fn single_row_attn_reference(
    q_b: &[f32],
    rel_b: &[f32],
    sink_b: &[f32],
    table: &[f32],
    k_cache: &[&[f32]],
    v_cache: &[&[f32]],
    lo: usize,
    hi: usize,
    rows: usize,
    inv_sqrt_d: f32,
    extent: usize,
) -> Vec<f32> {
    assert_eq!(k_cache.len(), 8);
    assert_eq!(v_cache.len(), 8);
    let mut out = vec![0.0f32; 24 * 128];
    for h in 0..24 {
        let kh = h / 3;
        let qh = &q_b[h * 128..(h + 1) * 128];
        let mut m = 0.0f32;
        let mut l = 1.0f32;
        let mut o = [0.0f32; 128];
        for t in lo..hi {
            let kk = t % rows;
            let krow = &k_cache[kh][kk * 128..(kk + 1) * 128];
            let dot = tree_dot128(qh, krow);
            // Absolute position of query b: dist = p - t = (hi - 1) - t.
            let dist = (hi - 1) - t;
            let bias = rel_bias_reg(&rel_b[h * 16..(h + 1) * 16], table, sink_b[h], dist, extent);
            let s = dot * inv_sqrt_d + bias;
            let m_new = m.max(s);
            let a = (m - m_new).exp();
            let e = (s - m_new).exp();
            l = l * a + e;
            let vrow = &v_cache[kh][kk * 128..(kk + 1) * 128];
            for d in 0..128 {
                o[d] = o[d] * a + e * vrow[d];
            }
            m = m_new;
        }
        for d in 0..128 {
            out[h * 128 + d] = o[d] / l;
        }
    }
    out
}

/// Batched host twin: QT-tiled, position-streaming, per-query skip masking.
///
/// Loop nest mirrors the kernel: qtile outer, absolute position `t` stream,
/// queries of the tile inner (one shared K/V row per `t`), Q-heads
/// innermost. Each query's own op sequence is identical to
/// [`single_row_attn_reference`]; interleaving queries within a `t` step
/// changes no per-query value. Returns `[B, 24, 128]`.
///
/// Inputs are the blocked-visit blocks: `q`: `[B, 24, 128]` f32 (q32 stack
/// rows), `rel`: `[B, 24, 16]` (r32 rows), `sinks`: `[B, 24]`,
/// `table`: `[16, extent]`, `k_cache`/`v_cache`: 8 x `[rows, 128]` host f32.
#[allow(clippy::too_many_arguments)]
pub fn qtile_attn_reference(
    q: &[f32],
    rel: &[f32],
    sinks: &[f32],
    table: &[f32],
    k_cache: &[&[f32]],
    v_cache: &[&[f32]],
    b_rows: usize,
    base_pos: usize,
    rows: usize,
    inv_sqrt_d: f32,
    extent: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), b_rows * 24 * 128);
    assert_eq!(rel.len(), b_rows * 24 * 16);
    assert_eq!(sinks.len(), b_rows * 24);
    let (t_start, t_end, windows) = prefill_key_plan(base_pos, b_rows, rows);
    let mut out = vec![0.0f32; b_rows * 24 * 128];
    // Per-query running softmax state (global memory in the kernel).
    let mut ms = vec![0.0f32; b_rows * 24];
    let mut ls = vec![1.0f32; b_rows * 24];
    let mut os = vec![0.0f32; b_rows * 24 * 128];
    for t in t_start..t_end {
        let kk = t % rows;
        for b in 0..b_rows {
            let (lo, hi) = windows[b];
            if t < lo || t >= hi {
                continue; // masked: zero ops, state untouched.
            }
            let p = base_pos + b;
            let dist = p - t;
            for h in 0..24 {
                let kh = h / 3;
                let qh = &q[(b * 24 + h) * 128..(b * 24 + h + 1) * 128];
                let krow = &k_cache[kh][kk * 128..(kk + 1) * 128];
                let dot = tree_dot128(qh, krow);
                let bias = rel_bias_reg(
                    &rel[(b * 24 + h) * 16..(b * 24 + h + 1) * 16],
                    table,
                    sinks[b * 24 + h],
                    dist,
                    extent,
                );
                let s = dot * inv_sqrt_d + bias;
                let m = ms[b * 24 + h];
                let m_new = m.max(s);
                let a = (m - m_new).exp();
                let e = (s - m_new).exp();
                ls[b * 24 + h] = ls[b * 24 + h] * a + e;
                let vrow = &v_cache[kh][kk * 128..(kk + 1) * 128];
                for d in 0..128 {
                    let o = &mut os[(b * 24 + h) * 128 + d];
                    *o = *o * a + e * vrow[d];
                }
                ms[b * 24 + h] = m_new;
            }
        }
    }
    for b in 0..b_rows {
        for h in 0..24 {
            let l = ls[b * 24 + h];
            for d in 0..128 {
                out[(b * 24 + h) * 128 + d] = os[(b * 24 + h) * 128 + d] / l;
            }
        }
    }
    out
}
pub const INKLING2_ATTN_PREFILL_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define INKLING2_PREFILL_QT 16

// One workgroup = QT queries x one KV head (3 Q heads).
// Grid (nqtile, 8), block 128. Mirrors inkling2_gf_attn_decode_f32 op order
// per (query, head); queries interleave within a t step, state is per-query.
extern "C" __global__ void inkling2_attn_prefill_qtile_f32(
    const float* __restrict__ q,      // [B, 24, 128]
    const float* __restrict__ rel,    // [B, 24, 16]
    const float* __restrict__ table,  // [16, extent]
    const float* __restrict__ sinks,  // [B, 24]
    const unsigned short* __restrict__ k0, const unsigned short* __restrict__ k1,
    const unsigned short* __restrict__ k2, const unsigned short* __restrict__ k3,
    const unsigned short* __restrict__ k4, const unsigned short* __restrict__ k5,
    const unsigned short* __restrict__ k6, const unsigned short* __restrict__ k7,
    const unsigned short* __restrict__ v0, const unsigned short* __restrict__ v1,
    const unsigned short* __restrict__ v2, const unsigned short* __restrict__ v3,
    const unsigned short* __restrict__ v4, const unsigned short* __restrict__ v5,
    const unsigned short* __restrict__ v6, const unsigned short* __restrict__ v7,
    // Fix #1b: pre-overwrite snapshot (SWA ring-wrap). t in [ev_lo, ev_hi)
    // reads sk/sv row (t - ev_lo); all other t read the ring slot t % rows.
    // ev_hi <= ev_lo disables the redirect (snap pointers still passed,
    // never dereferenced). Branch is uniform per t step.
    const unsigned short* __restrict__ sk0, const unsigned short* __restrict__ sk1,
    const unsigned short* __restrict__ sk2, const unsigned short* __restrict__ sk3,
    const unsigned short* __restrict__ sk4, const unsigned short* __restrict__ sk5,
    const unsigned short* __restrict__ sk6, const unsigned short* __restrict__ sk7,
    const unsigned short* __restrict__ sv0, const unsigned short* __restrict__ sv1,
    const unsigned short* __restrict__ sv2, const unsigned short* __restrict__ sv3,
    const unsigned short* __restrict__ sv4, const unsigned short* __restrict__ sv5,
    const unsigned short* __restrict__ sv6, const unsigned short* __restrict__ sv7,
    int ev_lo, int ev_hi,
    float* __restrict__ out,          // [B, 24, 128]
    int b_rows, int base_pos, int rows, int extent, float inv_sqrt_d)
{
    int qt = (int)blockIdx.x;   // query tile
    int kh = (int)blockIdx.y;   // 0..8 KV head (Q heads kh*3..kh*3+2)
    int d = (int)threadIdx.x;   // 0..127
    const unsigned short* ks[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    const unsigned short* vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    const unsigned short* sks[8] = {sk0, sk1, sk2, sk3, sk4, sk5, sk6, sk7};
    const unsigned short* svs[8] = {sv0, sv1, sv2, sv3, sv4, sv5, sv6, sv7};
    int b0 = qt * INKLING2_PREFILL_QT;
    int nq = b_rows - b0;
    if (nq > INKLING2_PREFILL_QT) nq = INKLING2_PREFILL_QT;
    if (nq <= 0) return;
    // Per-lane state in registers: [QT][3].
    float qq[INKLING2_PREFILL_QT][3];
    float mm[INKLING2_PREFILL_QT][3];
    float ll[INKLING2_PREFILL_QT][3];
    float oo[INKLING2_PREFILL_QT][3];
    for (int qb = 0; qb < INKLING2_PREFILL_QT; ++qb) {
        for (int h = 0; h < 3; ++h) {
            mm[qb][h] = 0.0f;   // phantom sink, like the decode kernel
            ll[qb][h] = 1.0f;
            oo[qb][h] = 0.0f;
            float qv = 0.0f;
            if (qb < nq) {
                int b = b0 + qb;
                qv = q[((size_t)b * 24 + kh * 3 + h) * 128 + d];
            }
            qq[qb][h] = qv;
        }
    }
    __shared__ float krow[128];
    __shared__ float vrow[128];
    // Wave-partial exchange: 4 waves x 1 float, double-buffered by
    // (qb * 3 + h) & 1 so one barrier per (qb, h) suffices (fix #1a).
    __shared__ float wsum[2][4];
    // Union t range of the tile: first query's lo .. last query's hi.
    int p_first = base_pos + b0;
    int nkv_first = p_first + 1 < rows ? p_first + 1 : rows;
    int p_last = base_pos + b0 + nq - 1;
    int t_start = p_first + 1 - nkv_first;
    int t_end = p_last + 1;
    for (int t = t_start; t < t_end; ++t) {
        // Fix #1b: evicted keys come from the pre-overwrite snapshot.
        int in_ev = (t >= ev_lo && t < ev_hi);
        int kk = in_ev ? (t - ev_lo) : (t % rows);
        const unsigned short* kptr = in_ev ? sks[kh] : ks[kh];
        const unsigned short* vptr = in_ev ? svs[kh] : vs[kh];
        // One shared K/V row load per t step (K-block-once at row granularity).
        krow[d] = __half2float(__ushort_as_half(kptr[(size_t)kk * 128 + d]));
        vrow[d] = __half2float(__ushort_as_half(vptr[(size_t)kk * 128 + d]));
        __syncthreads();
        for (int qb = 0; qb < nq; ++qb) {
            int b = b0 + qb;
            int p = base_pos + b;
            int n_kv = p + 1 < rows ? p + 1 : rows;
            int lo = p + 1 - n_kv;
            if (t < lo || t > p) continue;   // masked: zero ops.
            int dist = p - t;
            // Beta clamp past rel_extent (NOT a zero-term): dist >= extent
            // reads the last table column, exactly like the single-row
            // `inkling2_rel_bias_gather_f32` kernel beta's eager path runs.
            int dd = dist < extent ? dist : extent - 1;
            for (int h = 0; h < 3; ++h) {
                int H = kh * 3 + h;
                // Fix #1a: wave-shuffle dot over 128 lanes (4 waves of 32).
                // Each wave reduces its own 32 products sync-free; lane 0 of
                // each wave posts the partial; one barrier exchanges wsum;
                // every lane sums the 4 partials (uniform s, no broadcast).
                float x = qq[qb][h] * krow[d];
                for (int off = 16; off > 0; off >>= 1)
                    x += __shfl_down(x, off, 32);
                int ws = (qb * 3 + h) & 1;
                if ((d & 31) == 0) wsum[ws][d >> 5] = x;
                __syncthreads();
                float s = (wsum[ws][0] + wsum[ws][1]) + (wsum[ws][2] + wsum[ws][3]);
                float bias = -sinks[(size_t)b * 24 + H];
                for (int r = 0; r < 16; ++r) {
                    bias += table[r * extent + dd] * rel[((size_t)b * 24 + H) * 16 + r];
                }
                s = s * inv_sqrt_d + bias;
                float m_new = fmaxf(mm[qb][h], s);
                float a = expf(mm[qb][h] - m_new);
                float e = expf(s - m_new);
                ll[qb][h] = ll[qb][h] * a + e;
                oo[qb][h] = oo[qb][h] * a + e * vrow[d];
                mm[qb][h] = m_new;
            }
        }
    }
    for (int qb = 0; qb < nq; ++qb) {
        int b = b0 + qb;
        for (int h = 0; h < 3; ++h) {
            int H = kh * 3 + h;
            out[((size_t)b * 24 + H) * 128 + d] = oo[qb][h] / ll[qb][h];
        }
    }
}
"#;

/// Fix #1b: SWA pre-overwrite snapshot. Copies K/V ring rows
/// `[ev_lo, ev_lo + ev_len)` (slot `t % rows`) to linear per-head side
/// buffers (row `t - ev_lo`), so a wrapping block's batched attention can
/// read evicted keys after the block's own writes. Grid `(ev_len, 8)`,
/// block 128; each thread copies one K and one V element.
pub const INKLING2_SWA_SNAPSHOT_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void inkling2_swa_snapshot_f16(
    const unsigned short* __restrict__ k0, const unsigned short* __restrict__ k1,
    const unsigned short* __restrict__ k2, const unsigned short* __restrict__ k3,
    const unsigned short* __restrict__ k4, const unsigned short* __restrict__ k5,
    const unsigned short* __restrict__ k6, const unsigned short* __restrict__ k7,
    const unsigned short* __restrict__ v0, const unsigned short* __restrict__ v1,
    const unsigned short* __restrict__ v2, const unsigned short* __restrict__ v3,
    const unsigned short* __restrict__ v4, const unsigned short* __restrict__ v5,
    const unsigned short* __restrict__ v6, const unsigned short* __restrict__ v7,
    unsigned short* __restrict__ sk0, unsigned short* __restrict__ sk1,
    unsigned short* __restrict__ sk2, unsigned short* __restrict__ sk3,
    unsigned short* __restrict__ sk4, unsigned short* __restrict__ sk5,
    unsigned short* __restrict__ sk6, unsigned short* __restrict__ sk7,
    unsigned short* __restrict__ sv0, unsigned short* __restrict__ sv1,
    unsigned short* __restrict__ sv2, unsigned short* __restrict__ sv3,
    unsigned short* __restrict__ sv4, unsigned short* __restrict__ sv5,
    unsigned short* __restrict__ sv6, unsigned short* __restrict__ sv7,
    int ev_lo, int ev_len, int rows)
{
    int x = (int)blockIdx.x;
    int kh = (int)blockIdx.y;   // 0..8 KV head
    int d = (int)threadIdx.x;   // 0..127
    if (x >= ev_len) return;
    int t = ev_lo + x;
    int kk = t % rows;
    const unsigned short* ks[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    const unsigned short* vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    unsigned short* sks[8] = {sk0, sk1, sk2, sk3, sk4, sk5, sk6, sk7};
    unsigned short* svs[8] = {sv0, sv1, sv2, sv3, sv4, sv5, sv6, sv7};
    sks[kh][(size_t)x * 128 + d] = ks[kh][(size_t)kk * 128 + d];
    svs[kh][(size_t)x * 128 + d] = vs[kh][(size_t)kk * 128 + d];
}
"#;

impl Gpu {
    /// Batched multi-query attention over the prompt block (brief F1,
    /// kernel #3). One launch replaces the B per-row legacy/blocked
    /// attention calls of a block: grid `(nqtile, 8)`, block 128. `q` is
    /// `[B, 24, 128]` F32, `rel` `[B, 24, 16]` F32, `table` `[16, extent]`
    /// F32, `sinks` `[B, 24]` F32, each `k`/`v` `[rows, 128]` F16 (one
    /// tensor per KV head, the same `gs.k_cache[visit]` / `gs.v_cache[visit]`
    /// layout the seam writes), `out` `[B, 24, 128]` F32. Fix #1b: `sk`/`sv`
    /// are per-head pre-overwrite snapshot buffers (`[snap_rows, 128]` F16,
    /// `snap_rows >= ev_hi - ev_lo`); `t in [ev_lo, ev_hi)` reads the
    /// snapshot row `t - ev_lo`, all other `t` read the ring slot `t % rows`.
    /// `ev_hi <= ev_lo` disables the redirect (empty SWA evict set). Fails
    /// closed on any shape/dtype mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_prefill_qtile(
        &mut self,
        q: &GpuTensor,
        rel: &GpuTensor,
        table: &GpuTensor,
        sinks: &GpuTensor,
        k: &[&GpuTensor; 8],
        v: &[&GpuTensor; 8],
        sk: &[&GpuTensor; 8],
        sv: &[&GpuTensor; 8],
        ev_lo: usize,
        ev_hi: usize,
        out: &GpuTensor,
        b_rows: usize,
        base_pos: usize,
        rows: usize,
        extent: usize,
        inv_sqrt_d: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if b_rows == 0 || rows == 0 || extent == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "inkling2 attn prefill: empty batch/rows/extent",
            ));
        }
        if !inv_sqrt_d.is_finite() || inv_sqrt_d <= 0.0 {
            return Err(hip_bridge::HipError::new(
                0,
                "inkling2 attn prefill: inv_sqrt_d must be finite positive",
            ));
        }
        if q.dtype != DType::F32 || q.shape != vec![b_rows, 24 * 128] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 attn prefill: q must be F32 [B,3072]=[{b_rows},3072], got {:?} {:?}",
                    q.dtype, q.shape
                ),
            ));
        }
        if rel.dtype != DType::F32 || rel.shape != vec![b_rows, 24 * 16] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 attn prefill: rel must be F32 [B,384]=[{b_rows},384], got {:?} {:?}",
                    rel.dtype, rel.shape
                ),
            ));
        }
        if table.dtype != DType::F32 || table.shape.iter().product::<usize>() != 16 * extent {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 attn prefill: table must be F32 [16,extent]=16x{extent} row-major ({} elements), got {:?} {:?}",
                    16 * extent,
                    table.dtype,
                    table.shape
                ),
            ));
        }
        if sinks.dtype != DType::F32 || sinks.shape != vec![b_rows, 24] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 attn prefill: sinks must be F32 [B,24]=[{b_rows},24], got {:?} {:?}",
                    sinks.dtype, sinks.shape
                ),
            ));
        }
        for (i, t) in k.iter().chain(v.iter()).enumerate() {
            if t.dtype != DType::F16 || t.shape != vec![rows, 128] {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 attn prefill: k/v[{i}] must be F16 [rows,128]=[{rows},128], got {:?} {:?}",
                        t.dtype, t.shape
                    ),
                ));
            }
        }
        // Fix #1b: snapshot buffers carry evicted rows [ev_lo, ev_hi) linearly.
        let ev_len = ev_hi.saturating_sub(ev_lo);
        for (i, t) in sk.iter().chain(sv.iter()).enumerate() {
            if t.dtype != DType::F16 || t.shape.len() != 2 || t.shape[1] != 128 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 attn prefill: sk/sv[{i}] must be F16 [S,128], got {:?} {:?}",
                        t.dtype, t.shape
                    ),
                ));
            }
            if t.shape[0] < ev_len {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 attn prefill: sk/sv[{i}] rows {} < evict len {ev_len} (ev [{ev_lo},{ev_hi}))",
                        t.shape[0]
                    ),
                ));
            }
        }
        if out.dtype != DType::F32 || out.shape != vec![b_rows, 24 * 128] {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "inkling2 attn prefill: out must be F32 [B,3072]=[{b_rows},3072], got {:?} {:?}",
                    out.dtype, out.shape
                ),
            ));
        }
        let func_name = "inkling2_attn_prefill_qtile_f32";
        self.ensure_kernel(
            "inkling2_attn_prefill",
            INKLING2_ATTN_PREFILL_HIP_SRC,
            func_name,
        )?;
        let mut qp = q.buf.as_ptr();
        let mut rp = rel.buf.as_ptr();
        let mut tp = table.buf.as_ptr();
        let mut sp = sinks.buf.as_ptr();
        // Device pointers, one stable local per kernel parameter: `params`
        // carries the ADDRESS of each argument, so the values must live in
        // named locals (pushing the device pointer itself makes the driver
        // read the argument from device memory -> access violation).
        let mut k0p = k[0].buf.as_ptr();
        let mut k1p = k[1].buf.as_ptr();
        let mut k2p = k[2].buf.as_ptr();
        let mut k3p = k[3].buf.as_ptr();
        let mut k4p = k[4].buf.as_ptr();
        let mut k5p = k[5].buf.as_ptr();
        let mut k6p = k[6].buf.as_ptr();
        let mut k7p = k[7].buf.as_ptr();
        let mut v0p = v[0].buf.as_ptr();
        let mut v1p = v[1].buf.as_ptr();
        let mut v2p = v[2].buf.as_ptr();
        let mut v3p = v[3].buf.as_ptr();
        let mut v4p = v[4].buf.as_ptr();
        let mut v5p = v[5].buf.as_ptr();
        let mut v6p = v[6].buf.as_ptr();
        let mut v7p = v[7].buf.as_ptr();
        let mut sk0p = sk[0].buf.as_ptr();
        let mut sk1p = sk[1].buf.as_ptr();
        let mut sk2p = sk[2].buf.as_ptr();
        let mut sk3p = sk[3].buf.as_ptr();
        let mut sk4p = sk[4].buf.as_ptr();
        let mut sk5p = sk[5].buf.as_ptr();
        let mut sk6p = sk[6].buf.as_ptr();
        let mut sk7p = sk[7].buf.as_ptr();
        let mut sv0p = sv[0].buf.as_ptr();
        let mut sv1p = sv[1].buf.as_ptr();
        let mut sv2p = sv[2].buf.as_ptr();
        let mut sv3p = sv[3].buf.as_ptr();
        let mut sv4p = sv[4].buf.as_ptr();
        let mut sv5p = sv[5].buf.as_ptr();
        let mut sv6p = sv[6].buf.as_ptr();
        let mut sv7p = sv[7].buf.as_ptr();
        let mut elov = ev_lo as i32;
        let mut evhi = ev_hi as i32;
        let mut op = out.buf.as_ptr();
        let mut bv = b_rows as i32;
        let mut bpv = base_pos as i32;
        let mut rv = rows as i32;
        let mut ev = extent as i32;
        let mut iv = inv_sqrt_d;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut tp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut k0p as *mut _ as *mut c_void,
            &mut k1p as *mut _ as *mut c_void,
            &mut k2p as *mut _ as *mut c_void,
            &mut k3p as *mut _ as *mut c_void,
            &mut k4p as *mut _ as *mut c_void,
            &mut k5p as *mut _ as *mut c_void,
            &mut k6p as *mut _ as *mut c_void,
            &mut k7p as *mut _ as *mut c_void,
            &mut v0p as *mut _ as *mut c_void,
            &mut v1p as *mut _ as *mut c_void,
            &mut v2p as *mut _ as *mut c_void,
            &mut v3p as *mut _ as *mut c_void,
            &mut v4p as *mut _ as *mut c_void,
            &mut v5p as *mut _ as *mut c_void,
            &mut v6p as *mut _ as *mut c_void,
            &mut v7p as *mut _ as *mut c_void,
            &mut sk0p as *mut _ as *mut c_void,
            &mut sk1p as *mut _ as *mut c_void,
            &mut sk2p as *mut _ as *mut c_void,
            &mut sk3p as *mut _ as *mut c_void,
            &mut sk4p as *mut _ as *mut c_void,
            &mut sk5p as *mut _ as *mut c_void,
            &mut sk6p as *mut _ as *mut c_void,
            &mut sk7p as *mut _ as *mut c_void,
            &mut sv0p as *mut _ as *mut c_void,
            &mut sv1p as *mut _ as *mut c_void,
            &mut sv2p as *mut _ as *mut c_void,
            &mut sv3p as *mut _ as *mut c_void,
            &mut sv4p as *mut _ as *mut c_void,
            &mut sv5p as *mut _ as *mut c_void,
            &mut sv6p as *mut _ as *mut c_void,
            &mut sv7p as *mut _ as *mut c_void,
            &mut elov as *mut _ as *mut c_void,
            &mut evhi as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut bv as *mut _ as *mut c_void,
            &mut bpv as *mut _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
            &mut ev as *mut _ as *mut c_void,
            &mut iv as *mut _ as *mut c_void,
        ];
        let nqtile = b_rows.div_ceil(PREFILL_QT) as u32;
        self.launch_maybe_blob(
            func_name,
            [nqtile, 8, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(qp);
                blob.push_ptr(rp);
                blob.push_ptr(tp);
                blob.push_ptr(sp);
                for t in k.iter().chain(v.iter()) {
                    blob.push_ptr(t.buf.as_ptr());
                }
                for t in sk.iter().chain(sv.iter()) {
                    blob.push_ptr(t.buf.as_ptr());
                }
                blob.push_i32(elov);
                blob.push_i32(evhi);
                blob.push_ptr(op);
                blob.push_i32(bv);
                blob.push_i32(bpv);
                blob.push_i32(rv);
                blob.push_i32(ev);
                blob.push_f32(iv);
                blob
            },
        )
    }

    /// Fix #1b: snapshot K/V ring rows `[ev_lo, ev_lo + ev_len)` (slot
    /// `t % rows`) into linear per-head F16 side buffers (row `t - ev_lo`).
    /// Must run BEFORE the block's KV writes (the values it preserves are
    /// exactly what those writes evict). Zero-length snapshots are a no-op.
    /// Grid `(ev_len, 8)`, block 128. Fail-closed on shape/dtype mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_swa_snapshot_f16(
        &mut self,
        k: &[&GpuTensor; 8],
        v: &[&GpuTensor; 8],
        sk: &[&GpuTensor; 8],
        sv: &[&GpuTensor; 8],
        ev_lo: usize,
        ev_len: usize,
        rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if ev_len == 0 {
            return Ok(());
        }
        if rows == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "inkling2 swa snapshot: empty ring rows",
            ));
        }
        for (i, t) in k.iter().chain(v.iter()).enumerate() {
            if t.dtype != DType::F16 || t.shape != vec![rows, 128] {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 swa snapshot: k/v[{i}] must be F16 [rows,128]=[{rows},128], got {:?} {:?}",
                        t.dtype, t.shape
                    ),
                ));
            }
        }
        for (i, t) in sk.iter().chain(sv.iter()).enumerate() {
            if t.dtype != DType::F16 || t.shape.len() != 2 || t.shape[1] != 128 {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 swa snapshot: sk/sv[{i}] must be F16 [S,128], got {:?} {:?}",
                        t.dtype, t.shape
                    ),
                ));
            }
            if t.shape[0] < ev_len {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "inkling2 swa snapshot: sk/sv[{i}] rows {} < evict len {ev_len}",
                        t.shape[0]
                    ),
                ));
            }
        }
        let func_name = "inkling2_swa_snapshot_f16";
        self.ensure_kernel(
            "inkling2_swa_snapshot",
            INKLING2_SWA_SNAPSHOT_HIP_SRC,
            func_name,
        )?;
        // Named-local backing for the legacy void** path (see qtile above).
        let mut k0p = k[0].buf.as_ptr();
        let mut k1p = k[1].buf.as_ptr();
        let mut k2p = k[2].buf.as_ptr();
        let mut k3p = k[3].buf.as_ptr();
        let mut k4p = k[4].buf.as_ptr();
        let mut k5p = k[5].buf.as_ptr();
        let mut k6p = k[6].buf.as_ptr();
        let mut k7p = k[7].buf.as_ptr();
        let mut v0p = v[0].buf.as_ptr();
        let mut v1p = v[1].buf.as_ptr();
        let mut v2p = v[2].buf.as_ptr();
        let mut v3p = v[3].buf.as_ptr();
        let mut v4p = v[4].buf.as_ptr();
        let mut v5p = v[5].buf.as_ptr();
        let mut v6p = v[6].buf.as_ptr();
        let mut v7p = v[7].buf.as_ptr();
        let mut sk0p = sk[0].buf.as_ptr();
        let mut sk1p = sk[1].buf.as_ptr();
        let mut sk2p = sk[2].buf.as_ptr();
        let mut sk3p = sk[3].buf.as_ptr();
        let mut sk4p = sk[4].buf.as_ptr();
        let mut sk5p = sk[5].buf.as_ptr();
        let mut sk6p = sk[6].buf.as_ptr();
        let mut sk7p = sk[7].buf.as_ptr();
        let mut sv0p = sv[0].buf.as_ptr();
        let mut sv1p = sv[1].buf.as_ptr();
        let mut sv2p = sv[2].buf.as_ptr();
        let mut sv3p = sv[3].buf.as_ptr();
        let mut sv4p = sv[4].buf.as_ptr();
        let mut sv5p = sv[5].buf.as_ptr();
        let mut sv6p = sv[6].buf.as_ptr();
        let mut sv7p = sv[7].buf.as_ptr();
        let mut elov = ev_lo as i32;
        let mut elenv = ev_len as i32;
        let mut rv = rows as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut k0p as *mut _ as *mut c_void,
            &mut k1p as *mut _ as *mut c_void,
            &mut k2p as *mut _ as *mut c_void,
            &mut k3p as *mut _ as *mut c_void,
            &mut k4p as *mut _ as *mut c_void,
            &mut k5p as *mut _ as *mut c_void,
            &mut k6p as *mut _ as *mut c_void,
            &mut k7p as *mut _ as *mut c_void,
            &mut v0p as *mut _ as *mut c_void,
            &mut v1p as *mut _ as *mut c_void,
            &mut v2p as *mut _ as *mut c_void,
            &mut v3p as *mut _ as *mut c_void,
            &mut v4p as *mut _ as *mut c_void,
            &mut v5p as *mut _ as *mut c_void,
            &mut v6p as *mut _ as *mut c_void,
            &mut v7p as *mut _ as *mut c_void,
            &mut sk0p as *mut _ as *mut c_void,
            &mut sk1p as *mut _ as *mut c_void,
            &mut sk2p as *mut _ as *mut c_void,
            &mut sk3p as *mut _ as *mut c_void,
            &mut sk4p as *mut _ as *mut c_void,
            &mut sk5p as *mut _ as *mut c_void,
            &mut sk6p as *mut _ as *mut c_void,
            &mut sk7p as *mut _ as *mut c_void,
            &mut sv0p as *mut _ as *mut c_void,
            &mut sv1p as *mut _ as *mut c_void,
            &mut sv2p as *mut _ as *mut c_void,
            &mut sv3p as *mut _ as *mut c_void,
            &mut sv4p as *mut _ as *mut c_void,
            &mut sv5p as *mut _ as *mut c_void,
            &mut sv6p as *mut _ as *mut c_void,
            &mut sv7p as *mut _ as *mut c_void,
            &mut elov as *mut _ as *mut c_void,
            &mut elenv as *mut _ as *mut c_void,
            &mut rv as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            func_name,
            [ev_len as u32, 8, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                for t in k.iter().chain(v.iter()) {
                    blob.push_ptr(t.buf.as_ptr());
                }
                for t in sk.iter().chain(sv.iter()) {
                    blob.push_ptr(t.buf.as_ptr());
                }
                blob.push_i32(elov);
                blob.push_i32(elenv);
                blob.push_i32(rv);
                blob
            },
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn rng_next(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn rng_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = (rng_next(state) >> 11) as f32 / (1u64 << 53) as f32;
        lo + u * (hi - lo)
    }

    struct Case {
        b: usize,
        base_pos: usize,
        rows: usize,
        extent: usize,
    }

    fn run_case(c: &Case) -> (Vec<f32>, Vec<f32>) {
        let mut s = 0x9e37_79b9_7f4a_7c15u64
            .wrapping_add(c.b as u64)
            .wrapping_add((c.base_pos as u64) << 20)
            .wrapping_add((c.rows as u64) << 40);
        let rnd = |s: &mut u64, n: usize, lo: f32, hi: f32| -> Vec<f32> {
            (0..n).map(|_| rng_f32(s, lo, hi)).collect()
        };
        let q = rnd(&mut s, c.b * 24 * 128, -2.0, 2.0);
        let rel = rnd(&mut s, c.b * 24 * 16, -1.0, 1.0);
        let table = rnd(&mut s, 16 * c.extent, -0.5, 0.5);
        let sinks = rnd(&mut s, c.b * 24, -1.0, 1.0);
        let kh_cache: Vec<Vec<f32>> = (0..8)
            .map(|_| rnd(&mut s, c.rows * 128, -2.0, 2.0))
            .collect();
        let vh_cache: Vec<Vec<f32>> = (0..8)
            .map(|_| rnd(&mut s, c.rows * 128, -2.0, 2.0))
            .collect();
        let k_refs: Vec<&[f32]> = kh_cache.iter().map(|v| v.as_slice()).collect();
        let v_refs: Vec<&[f32]> = vh_cache.iter().map(|v| v.as_slice()).collect();
        let inv = 1.0f32 / 128f32.sqrt();
        let batched = qtile_attn_reference(
            &q, &rel, &sinks, &table, &k_refs, &v_refs, c.b, c.base_pos, c.rows, inv, c.extent,
        );
        let (_, _, windows) = prefill_key_plan(c.base_pos, c.b, c.rows);
        let mut single = vec![0.0f32; c.b * 24 * 128];
        for b in 0..c.b {
            let (lo, hi) = windows[b];
            let row = single_row_attn_reference(
                &q[b * 24 * 128..(b + 1) * 24 * 128],
                &rel[b * 24 * 16..(b + 1) * 24 * 16],
                &sinks[b * 24..(b + 1) * 24],
                &table,
                &k_refs,
                &v_refs,
                lo,
                hi,
                c.rows,
                inv,
                c.extent,
            );
            single[b * 24 * 128..(b + 1) * 24 * 128].copy_from_slice(&row);
        }
        (batched, single)
    }

    fn assert_bitwise(case_name: &str, c: &Case) {
        let (batched, single) = run_case(c);
        assert_eq!(batched.len(), single.len());
        let mut ndiff = 0usize;
        let mut maxd = 0.0f32;
        for (i, (a, b)) in batched.iter().zip(single.iter()).enumerate() {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            if a.to_bits() != b.to_bits() {
                ndiff += 1;
                if ndiff <= 4 {
                    eprintln!("{case_name}: word {i}: batched {a:e} vs single {b:e}");
                }
            }
        }
        eprintln!(
            "{case_name}: B={} pos={} rows={} words={} max|d|={maxd:e} bitwise_diffs={ndiff}",
            c.b,
            c.base_pos,
            c.rows,
            batched.len()
        );
        assert_eq!(
            ndiff, 0,
            "{case_name}: {ndiff} bitwise diffs, max|d|={maxd:e}"
        );
    }

    #[test]
    fn qtile_full_tiles_no_wrap() {
        // B=33: two full QT=16 tiles + tail of 1; no ring wrap, no clamp.
        assert_bitwise(
            "full_tiles",
            &Case {
                b: 33,
                base_pos: 100,
                rows: 4096,
                extent: 1024,
            },
        );
    }

    #[test]
    fn qtile_swa_wrap() {
        // SWA ring rows=8, pos 20..36: every window capped at 8, slots wrap.
        assert_bitwise(
            "swa_wrap",
            &Case {
                b: 17,
                base_pos: 20,
                rows: 8,
                extent: 1024,
            },
        );
    }

    #[test]
    fn qtile_dist_clamp_past_extent() {
        // base_pos 2000 with extent 1024: dist exceeds extent, clamp path
        // (beta rule, matching the single-row gather kernel).
        assert_bitwise(
            "clamp_past_extent",
            &Case {
                b: 5,
                base_pos: 2000,
                rows: 4096,
                extent: 1024,
            },
        );
    }

    #[test]
    fn qtile_single_key() {
        // Minimal: B=1 at pos 0, single key, tail tile.
        assert_bitwise(
            "single_key",
            &Case {
                b: 1,
                base_pos: 0,
                rows: 4,
                extent: 1024,
            },
        );
    }

    #[test]
    fn qtile_blocked_boundary_span() {
        // n_kv 255..304 across the block: spans the 256 legacy/blocked
        // routing boundary (both routes share the legacy math, so the twin
        // must agree with the transcription at every width).
        assert_bitwise(
            "boundary",
            &Case {
                b: 50,
                base_pos: 254,
                rows: 4096,
                extent: 1024,
            },
        );
    }

    /// Gate: one batched qtile launch == per-row single-row reference on the
    /// same F16 K/V bytes, at 4 shapes (single-key, SWA wrap, full tiles +
    /// tail, dist clamp past extent). Bar: max|d| <= 1e-4.
    #[test]
    #[ignore = "requires gfx1100 and the HIP toolchain"]
    fn gpu_qtile_matches_single_row_f16() {
        let mut gpu = match crate::gpu_test::init() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no usable GPU ({e:?})");
                return;
            }
        };
        assert_eq!(gpu.arch, "gfx1100");
        let inv = 1.0f32 / 128f32.sqrt();
        for (b, base_pos, rows, extent) in [
            (1usize, 0usize, 8usize, 64usize),
            (17, 20, 8, 64),
            (33, 100, 512, 256),
            (5, 2000, 2048, 1024),
        ] {
            let mut s = 0xA77E_1cca_b10c5u64
                .wrapping_add(b as u64)
                .wrapping_add((base_pos as u64) << 20)
                .wrapping_add((rows as u64) << 40);
            let rnd = |s: &mut u64, n: usize, lo: f32, hi: f32| -> Vec<f32> {
                (0..n).map(|_| rng_f32(s, lo, hi)).collect()
            };
            let qh = rnd(&mut s, b * 24 * 128, -2.0, 2.0);
            let relh = rnd(&mut s, b * 24 * 16, -1.0, 1.0);
            let tableh = rnd(&mut s, 16 * extent, -0.5, 0.5);
            let sinksh = rnd(&mut s, b * 24, -1.0, 1.0);
            // K/V caches as F16 device bytes (the seam's stored format);
            // the host reference runs on the exact F16 -> F32 values.
            let mut kh_f16: Vec<Vec<u16>> = Vec::with_capacity(8);
            let mut vh_f16: Vec<Vec<u16>> = Vec::with_capacity(8);
            let mut kh_f32: Vec<Vec<f32>> = Vec::with_capacity(8);
            let mut vh_f32: Vec<Vec<f32>> = Vec::with_capacity(8);
            for _ in 0..8 {
                let k32 = rnd(&mut s, rows * 128, -2.0, 2.0);
                let v32 = rnd(&mut s, rows * 128, -2.0, 2.0);
                kh_f16.push(
                    k32.iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect(),
                );
                vh_f16.push(
                    v32.iter()
                        .map(|&x| half::f16::from_f32(x).to_bits())
                        .collect(),
                );
                kh_f32.push(
                    kh_f16
                        .last()
                        .unwrap()
                        .iter()
                        .map(|&w| half::f16::from_bits(w).to_f32())
                        .collect(),
                );
                vh_f32.push(
                    vh_f16
                        .last()
                        .unwrap()
                        .iter()
                        .map(|&w| half::f16::from_bits(w).to_f32())
                        .collect(),
                );
            }
            let q = gpu.upload_f32(&qh, &[b, 24 * 128]).expect("q upload");
            let rel = gpu.upload_f32(&relh, &[b, 24 * 16]).expect("rel upload");
            let table = gpu
                .upload_f32(&tableh, &[16 * extent])
                .expect("table upload");
            let sinks = gpu.upload_f32(&sinksh, &[b, 24]).expect("sinks upload");
            let ks: Vec<crate::dispatch::GpuTensor> = kh_f16
                .iter()
                .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("k upload"))
                .collect();
            let vs: Vec<crate::dispatch::GpuTensor> = vh_f16
                .iter()
                .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("v upload"))
                .collect();
            let ka: [&crate::dispatch::GpuTensor; 8] = [
                &ks[0], &ks[1], &ks[2], &ks[3], &ks[4], &ks[5], &ks[6], &ks[7],
            ];
            let va: [&crate::dispatch::GpuTensor; 8] = [
                &vs[0], &vs[1], &vs[2], &vs[3], &vs[4], &vs[5], &vs[6], &vs[7],
            ];
            let out = gpu
                .alloc_tensor(&[b, 24 * 128], DType::F32)
                .expect("out alloc");
            // No SWA wrap in these shapes: empty evict set (ev 0,0), dummy
            // single-row snapshot buffers (never dereferenced).
            let sk_dummy = gpu.alloc_tensor(&[1, 128], DType::F16).expect("sk dummy");
            let sv_dummy = gpu.alloc_tensor(&[1, 128], DType::F16).expect("sv dummy");
            let ska: [&crate::dispatch::GpuTensor; 8] = [&sk_dummy; 8];
            let sva: [&crate::dispatch::GpuTensor; 8] = [&sv_dummy; 8];
            gpu.inkling2_attn_prefill_qtile(
                &q, &rel, &table, &sinks, &ka, &va, &ska, &sva, 0, 0, &out, b, base_pos, rows,
                extent, inv,
            )
            .expect("qtile launch");
            let got = gpu.download_f32(&out).expect("out download");
            // Reference: per-row single-row op order on the same F16 bytes.
            let k_refs: Vec<&[f32]> = kh_f32.iter().map(|x| x.as_slice()).collect();
            let v_refs: Vec<&[f32]> = vh_f32.iter().map(|x| x.as_slice()).collect();
            let (_, _, windows) = prefill_key_plan(base_pos, b, rows);
            let mut want = vec![0.0f32; b * 24 * 128];
            for row in 0..b {
                let (lo, hi) = windows[row];
                let r = single_row_attn_reference(
                    &qh[row * 24 * 128..(row + 1) * 24 * 128],
                    &relh[row * 24 * 16..(row + 1) * 24 * 16],
                    &sinksh[row * 24..(row + 1) * 24],
                    &tableh,
                    &k_refs,
                    &v_refs,
                    lo,
                    hi,
                    rows,
                    inv,
                    extent,
                );
                want[row * 24 * 128..(row + 1) * 24 * 128].copy_from_slice(&r);
            }
            let mut maxd = 0.0f32;
            let mut nbit = 0usize;
            for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
                maxd = maxd.max((a - w).abs());
                if a.to_bits() != w.to_bits() {
                    nbit += 1;
                    if nbit <= 4 {
                        eprintln!("b={b} base={base_pos}: word {i}: got {a:e} want {w:e}");
                    }
                }
            }
            eprintln!(
                "b={b} base={base_pos} rows={rows} extent={extent}: words={} max|d|={maxd:e} bitwise_diffs={nbit}",
                got.len()
            );
            assert!(
                maxd <= 1e-4,
                "b={b} base={base_pos}: max|d|={maxd:e} exceeds 1e-4"
            );
        }
    }

    /// Gate: SWA snapshot + qtile evict redirect (fix #1b). Simulates one
    /// wrapping SWA block (b=5, base=10, rows=8: evict `[2,7)`) end to end:
    /// the snapshot kernel copies pre-write ring rows to linear side buffers
    /// (gated bitwise), then the qtile reads the post-write ring with the
    /// redirect and must reproduce true-key single-row attention (1e-4 bar).
    /// Vacuity guard: the post-write ring provably differs from the true
    /// keys on redirected rows, so a dead redirect cannot pass.
    #[test]
    #[ignore = "requires gfx1100 and the HIP toolchain"]
    fn gpu_qtile_swa_redirect() {
        let mut gpu = match crate::gpu_test::init() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skip: no usable GPU ({e:?})");
                return;
            }
        };
        assert_eq!(gpu.arch, "gfx1100");
        let inv = 1.0f32 / 128f32.sqrt();
        let (b, base_pos, rows, extent) = (5usize, 10usize, 8usize, 64usize);
        let (ev_lo, ev_hi) = (2usize, 7usize); // [base-rows, base+b-rows)
        let ev_len = ev_hi - ev_lo;
        let npos = base_pos + b; // positions [0, 15)
        let mut s = 0x5EED_1b5Au64
            .wrapping_add((b as u64) << 20)
            .wrapping_add((base_pos as u64) << 40);
        let rnd = |s: &mut u64, n: usize, lo: f32, hi: f32| -> Vec<f32> {
            (0..n).map(|_| rng_f32(s, lo, hi)).collect()
        };
        let qh = rnd(&mut s, b * 24 * 128, -2.0, 2.0);
        let relh = rnd(&mut s, b * 24 * 16, -1.0, 1.0);
        let tableh = rnd(&mut s, 16 * extent, -0.5, 0.5);
        let sinksh = rnd(&mut s, b * 24, -1.0, 1.0);
        // True keys per position (F16-quantized; both arms see identical bytes).
        let mut true_kbits: Vec<Vec<u16>> = Vec::with_capacity(8);
        let mut true_vbits: Vec<Vec<u16>> = Vec::with_capacity(8);
        let mut true_k32: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut true_v32: Vec<Vec<f32>> = Vec::with_capacity(8);
        for _ in 0..8 {
            let k32 = rnd(&mut s, npos * 128, -2.0, 2.0);
            let v32 = rnd(&mut s, npos * 128, -2.0, 2.0);
            true_kbits.push(
                k32.iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            );
            true_vbits.push(
                v32.iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            );
            true_k32.push(
                true_kbits
                    .last()
                    .unwrap()
                    .iter()
                    .map(|&w| half::f16::from_bits(w).to_f32())
                    .collect(),
            );
            true_v32.push(
                true_vbits
                    .last()
                    .unwrap()
                    .iter()
                    .map(|&w| half::f16::from_bits(w).to_f32())
                    .collect(),
            );
        }
        // Ring slot s holds the latest position = s (mod rows): pre-write
        // (positions < base) and post-write (positions < base + b).
        let slot_of = |pos: usize| pos % rows;
        let mut pre_kh: Vec<Vec<u16>> = Vec::with_capacity(8);
        let mut pre_vh: Vec<Vec<u16>> = Vec::with_capacity(8);
        let mut post_kh: Vec<Vec<u16>> = Vec::with_capacity(8);
        let mut post_vh: Vec<Vec<u16>> = Vec::with_capacity(8);
        for kh in 0..8 {
            let mut pre_k = vec![0u16; rows * 128];
            let mut pre_v = vec![0u16; rows * 128];
            let mut post_k = vec![0u16; rows * 128];
            let mut post_v = vec![0u16; rows * 128];
            // Latest position per residue strictly before `bound`.
            for slot in 0..rows {
                let mut p_pre = None;
                let mut p_post = None;
                for pos in 0..npos {
                    if slot_of(pos) == slot {
                        p_post = Some(pos);
                        if pos < base_pos {
                            p_pre = Some(pos);
                        }
                    }
                }
                let pp = p_pre.expect("every residue occurs before base");
                let pq = p_post.unwrap();
                pre_k[slot * 128..(slot + 1) * 128]
                    .copy_from_slice(&true_kbits[kh][pp * 128..(pp + 1) * 128]);
                pre_v[slot * 128..(slot + 1) * 128]
                    .copy_from_slice(&true_vbits[kh][pp * 128..(pp + 1) * 128]);
                post_k[slot * 128..(slot + 1) * 128]
                    .copy_from_slice(&true_kbits[kh][pq * 128..(pq + 1) * 128]);
                post_v[slot * 128..(slot + 1) * 128]
                    .copy_from_slice(&true_vbits[kh][pq * 128..(pq + 1) * 128]);
            }
            pre_kh.push(pre_k);
            pre_vh.push(pre_v);
            post_kh.push(post_k);
            post_vh.push(post_v);
        }
        // Vacuity guard: the redirect must matter — post-write slot of an
        // evicted t holds the block's overwrite, not the true key.
        assert_ne!(
            post_kh[0][3 * 128],
            true_kbits[0][3 * 128],
            "test vacuous: post-write ring already holds true key at t=3"
        );
        // Upload pre-write ring + snapshot side buffers; run the snapshot.
        let pre_k: Vec<crate::dispatch::GpuTensor> = pre_kh
            .iter()
            .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("pre k"))
            .collect();
        let pre_v: Vec<crate::dispatch::GpuTensor> = pre_vh
            .iter()
            .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("pre v"))
            .collect();
        let sk: Vec<crate::dispatch::GpuTensor> = (0..8)
            .map(|_| gpu.alloc_tensor(&[ev_len, 128], DType::F16).expect("sk"))
            .collect();
        let sv: Vec<crate::dispatch::GpuTensor> = (0..8)
            .map(|_| gpu.alloc_tensor(&[ev_len, 128], DType::F16).expect("sv"))
            .collect();
        let ka: [&crate::dispatch::GpuTensor; 8] = [
            &pre_k[0], &pre_k[1], &pre_k[2], &pre_k[3], &pre_k[4], &pre_k[5], &pre_k[6], &pre_k[7],
        ];
        let va: [&crate::dispatch::GpuTensor; 8] = [
            &pre_v[0], &pre_v[1], &pre_v[2], &pre_v[3], &pre_v[4], &pre_v[5], &pre_v[6], &pre_v[7],
        ];
        let ska: [&crate::dispatch::GpuTensor; 8] = [
            &sk[0], &sk[1], &sk[2], &sk[3], &sk[4], &sk[5], &sk[6], &sk[7],
        ];
        let sva: [&crate::dispatch::GpuTensor; 8] = [
            &sv[0], &sv[1], &sv[2], &sv[3], &sv[4], &sv[5], &sv[6], &sv[7],
        ];
        gpu.inkling2_swa_snapshot_f16(&ka, &va, &ska, &sva, ev_lo, ev_len, rows)
            .expect("snapshot launch");
        // Snapshot gate: bitwise vs the true evicted rows.
        for kh in 0..8 {
            let got_k = gpu.download_f16_bits(&sk[kh]).expect("dl sk");
            let got_v = gpu.download_f16_bits(&sv[kh]).expect("dl sv");
            let want_k = &true_kbits[kh][ev_lo * 128..ev_hi * 128];
            let want_v = &true_vbits[kh][ev_lo * 128..ev_hi * 128];
            assert_eq!(got_k, want_k, "snapshot k head {kh} mismatch");
            assert_eq!(got_v, want_v, "snapshot v head {kh} mismatch");
        }
        eprintln!("swa snapshot: 8+8 heads bitwise exact over ev [{ev_lo},{ev_hi})");
        // Upload post-write ring + q/rel/table/sinks; run qtile redirect.
        let post_k: Vec<crate::dispatch::GpuTensor> = post_kh
            .iter()
            .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("post k"))
            .collect();
        let post_v: Vec<crate::dispatch::GpuTensor> = post_vh
            .iter()
            .map(|bits| gpu.upload_f16_bits(bits, &[rows, 128]).expect("post v"))
            .collect();
        let q = gpu.upload_f32(&qh, &[b, 24 * 128]).expect("q upload");
        let rel = gpu.upload_f32(&relh, &[b, 24 * 16]).expect("rel upload");
        let table = gpu
            .upload_f32(&tableh, &[16 * extent])
            .expect("table upload");
        let sinks = gpu.upload_f32(&sinksh, &[b, 24]).expect("sinks upload");
        let out = gpu
            .alloc_tensor(&[b, 24 * 128], DType::F32)
            .expect("out alloc");
        let pka: [&crate::dispatch::GpuTensor; 8] = [
            &post_k[0], &post_k[1], &post_k[2], &post_k[3], &post_k[4], &post_k[5], &post_k[6],
            &post_k[7],
        ];
        let pva: [&crate::dispatch::GpuTensor; 8] = [
            &post_v[0], &post_v[1], &post_v[2], &post_v[3], &post_v[4], &post_v[5], &post_v[6],
            &post_v[7],
        ];
        gpu.inkling2_attn_prefill_qtile(
            &q, &rel, &table, &sinks, &pka, &pva, &ska, &sva, ev_lo, ev_hi, &out, b, base_pos,
            rows, extent, inv,
        )
        .expect("qtile redirect launch");
        let got = gpu.download_f32(&out).expect("out download");
        // Reference: true-key single-row attention. Position-indexed caches
        // with `rows = npos` make the reference's `kk = t % rows` read row t.
        let k_refs: Vec<&[f32]> = true_k32.iter().map(|x| x.as_slice()).collect();
        let v_refs: Vec<&[f32]> = true_v32.iter().map(|x| x.as_slice()).collect();
        let mut want = vec![0.0f32; b * 24 * 128];
        for row in 0..b {
            let p = base_pos + row;
            let n_kv = (p + 1).min(rows);
            let (lo, hi) = (p + 1 - n_kv, p + 1);
            let r = single_row_attn_reference(
                &qh[row * 24 * 128..(row + 1) * 24 * 128],
                &relh[row * 24 * 16..(row + 1) * 24 * 16],
                &sinksh[row * 24..(row + 1) * 24],
                &tableh,
                &k_refs,
                &v_refs,
                lo,
                hi,
                npos,
                inv,
                extent,
            );
            want[row * 24 * 128..(row + 1) * 24 * 128].copy_from_slice(&r);
        }
        let mut maxd = 0.0f32;
        for (a, w) in got.iter().zip(want.iter()) {
            maxd = maxd.max((a - w).abs());
        }
        eprintln!(
            "swa redirect: b={b} base={base_pos} rows={rows} ev=[{ev_lo},{ev_hi}) max|d|={maxd:e}"
        );
        assert!(maxd <= 1e-4, "swa redirect: max|d|={maxd:e} exceeds 1e-4");
    }
}
