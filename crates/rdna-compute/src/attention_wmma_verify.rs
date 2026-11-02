// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Ben1078
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1100 WMMA split-K speculative-verify attention over a plain Q8_0
//! KV cache (default on for exact gfx1100; `HIPFIRE_GFX1100_WMMA_VERIFY_ATTN=0`
//! restores the incumbent verify attention).
//!
//! An MTP K=3 verify attends 4 query rows (x 24 q heads) over the whole Q8
//! KV cache in every full-attention layer. Past the multi-row crossover
//! (`HIPFIRE_FA_PERTOKEN_MIN_CTX`, 4096 by default) the eager gfx1100 verify
//! takes the Q8 FA2 split-KV verifier (`kernel.gfx1100_fa2_split_verify`, a
//! fixed grid of 8 splits x 4 kv heads = 32 workgroups), or the VerifyAttn
//! rows twin / the R4/R8 tile. Those leave most of the 7900 XTX's 48 WGPs
//! idle while the KV stream is the whole cost, so verify attention grows
//! with the context. This route
//! (`kernels/src/attention_verify_wmma.gfx1100.hip`) runs one block per
//! (kv head, key span) on a fixed grid of 4 x [`wmma_verify_attn_spans`]
//! blocks: each block owns the interleaved 32-key tiles `b, b + nblk, ...`,
//! dequantizes them into f16 K / V^T LDS planes, and computes S = Q K^T and
//! O = P V with wave32 WMMA for all 6 q heads x `rows` queries of its kv
//! head (16-query tiles), with an FA2-order online softmax. A small reduce
//! (`attention_verify_wmma_reduce_gfx1100`) merges the spans.
//!
//! Selection ([`Gpu::wmma_verify_attn_active`]): the flag on (the
//! exact-gfx1100 default; the parent `kernel.verify_attn` /
//! `HIPFIRE_VERIFY_ATTN=0` opts out too), exact gfx1100, 1..=8 rows,
//! 24 q / 4 kv heads, head_dim 256, eager only (never under graph capture or
//! Redline recording, whose launch contracts keep the incumbent routes).
//! The dispatch layer calls it inside the admitted multi-row verify attend
//! step (`batch_chunk_fa_attend`, plain Q8 K and V, after the K/V write), so
//! tree, independent and short-context (<= the crossover) attention are
//! unchanged. Every module is declared in the gfx1100 kernel registry, so the
//! gfx1100 pack carries them precompiled.
//!
//! Numerics: Q, dequantized K/V and P are f16 WMMA operands with f32
//! accumulation (the FA2 class); the softmax uses exp2 with a lazy rescale
//! (the running max moves only when a tile exceeds it by 8 log2 units, exact
//! algebra, P <= 256). Not byte-identical to the incumbent routes,
//! KLD-gated. Deterministic: no atomics, fixed grid, fixed reduce order.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

/// Module and symbols (registered for gfx1100 in `kernel_registry.rs`).
pub const WMMA_VERIFY_ATTN_MODULE: &str = "attention_verify_wmma_gfx1100";
/// Main kernels by 16-query tile count (rows 1..=2, 3..=5, 6..=8).
pub const WMMA_VERIFY_ATTN_SYMBOLS: [&str; 3] = [
    "attention_verify_wmma_q8_qt1_gfx1100",
    "attention_verify_wmma_q8_qt2_gfx1100",
    "attention_verify_wmma_q8_qt3_gfx1100",
];
pub const WMMA_VERIFY_ATTN_REDUCE: &str = "attention_verify_wmma_reduce_gfx1100";

/// Largest verify block (rows) the route takes.
pub const WMMA_VERIFY_ATTN_MAX_ROWS: usize = 8;
/// f32 words per partial: m, l, 2 pad, 256 O values (`VW_PSTRIDE`).
pub const WMMA_VERIFY_ATTN_PSTRIDE: usize = 260;
/// Q heads per kv head the kernel is written for (24 / 4).
const GQA: usize = 6;

/// 16-query WMMA tiles for `rows` verify rows (6 q heads per kv head).
pub fn wmma_verify_attn_qt(rows: usize) -> usize {
    (GQA * rows).div_ceil(16).max(1)
}

/// Blocks (key spans) per kv head: one resident round on the 7900 XTX at
/// 3 blocks per WGP (<= 41.7 KB LDS) for 1..=5 rows, 2 per WGP (45.8 KB)
/// for 6..=8 rows, x 48 WGPs / 4 kv heads; below 24K context 24 per kv head
/// measured ~6% faster at 9K (fewer partials, longer spans).
pub fn wmma_verify_attn_spans(rows: usize, ctx_len: usize) -> usize {
    if wmma_verify_attn_qt(rows) >= 3 || ctx_len < 24 * 1024 {
        24
    } else {
        36
    }
}

/// f32 partials the route needs for `rows` rows at `ctx_len` (24 q heads).
pub fn wmma_verify_attn_partials_len(rows: usize, n_heads: usize, ctx_len: usize) -> usize {
    rows * n_heads * wmma_verify_attn_spans(rows, ctx_len) * WMMA_VERIFY_ATTN_PSTRIDE
}

/// One stderr line the first time each main kernel runs, so a harness log
/// proves the route was active.
static WMMA_VERIFY_ATTN_ENGAGED: AtomicU32 = AtomicU32::new(0);

impl Gpu {
    /// Whether a `rows`-row verify attention of this head shape takes the
    /// WMMA split-K route (see the module docs for the gates; the caller
    /// checks the KV codec).
    pub fn wmma_verify_attn_active(
        &self,
        rows: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> bool {
        self.flags.gfx1100_wmma_verify_attn
            && self.flags.verify_attn
            && self.arch == "gfx1100"
            && self.arch_caps.is_gfx1100()
            && (1..=WMMA_VERIFY_ATTN_MAX_ROWS).contains(&rows)
            && n_heads == GQA * n_kv_heads
            && n_kv_heads == 4
            && head_dim == 256
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
    }

    /// WMMA split-K verify attention: `out[r] = softmax(q[r] K^T / 16) V` for
    /// `rows` query rows at absolute positions `positions[0..rows]` (each
    /// row attends keys `0..=positions[r]`) over plain Q8_0 `k_cache` /
    /// `v_cache` (`[pos][4 kv heads][8 blocks of 34 B]`). `q` / `out` are f32
    /// `[rows][n_heads * 256]`. K/V must already hold every position up to
    /// the largest `positions[r]`; no key past it is read. `ctx_len` (the
    /// live causal extent) only picks the span count. `partials` is f32
    /// scratch of at least [`wmma_verify_attn_partials_len`]. `Ok(false)` =
    /// not admitted (gates, or partials too small); the caller falls back.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_verify_wmma_q8_gfx1100(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ctx_len: usize,
        rows: usize,
        partials: &GpuTensor,
    ) -> HipResult<bool> {
        if !self.wmma_verify_attn_active(rows, n_heads, n_kv_heads, head_dim) {
            return Ok(false);
        }
        let q_dim = n_heads * head_dim;
        let nblk = wmma_verify_attn_spans(rows, ctx_len);
        if partials.numel() < wmma_verify_attn_partials_len(rows, n_heads, ctx_len)
            || q.numel() < rows * q_dim
            || out.numel() < rows * q_dim
            || positions.numel() < rows
        {
            return Ok(false);
        }
        self.bind_thread()?;
        let qt = wmma_verify_attn_qt(rows);
        let symbol = WMMA_VERIFY_ATTN_SYMBOLS[qt - 1];
        self.ensure_kernel(
            WMMA_VERIFY_ATTN_MODULE,
            kernels::ATTENTION_VERIFY_WMMA_GFX1100_SRC,
            symbol,
        )?;
        self.ensure_kernel(
            WMMA_VERIFY_ATTN_MODULE,
            kernels::ATTENTION_VERIFY_WMMA_GFX1100_SRC,
            WMMA_VERIFY_ATTN_REDUCE,
        )?;
        let bit = 1u32 << (qt - 1);
        if WMMA_VERIFY_ATTN_ENGAGED.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            eprintln!(
                "[wmma-verify-attn] route active: {symbol} rows={rows} spans={nblk} ctx={ctx_len}"
            );
        }
        let q_ptr = q.buf.as_ptr();
        let k_ptr = k_cache.buf.as_ptr();
        let v_ptr = v_cache.buf.as_ptr();
        let p_ptr = partials.buf.as_ptr();
        let pos_ptr = positions.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let nh = n_heads as i32;
        // Scores in log2 units: one v_exp_f32 per score.
        let scale_log2 = std::f32::consts::LOG2_E / (head_dim as f32).sqrt();
        let n_blk = nblk as i32;
        let max_spans = nblk as i32;
        let batch_offset = 0i32;
        let rows_i = rows as i32;
        {
            let func = &self.functions[symbol];
            let mut params: Vec<*mut c_void> = vec![
                &q_ptr as *const _ as *mut c_void,
                &k_ptr as *const _ as *mut c_void,
                &v_ptr as *const _ as *mut c_void,
                &p_ptr as *const _ as *mut c_void,
                &pos_ptr as *const _ as *mut c_void,
                &nh as *const _ as *mut c_void,
                &scale_log2 as *const _ as *mut c_void,
                &n_blk as *const _ as *mut c_void,
                &max_spans as *const _ as *mut c_void,
                &batch_offset as *const _ as *mut c_void,
                &rows_i as *const _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [n_kv_heads as u32, nblk as u32, 1],
                    [(128 * qt) as u32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }
        let func = &self.functions[WMMA_VERIFY_ATTN_REDUCE];
        let mut params: Vec<*mut c_void> = vec![
            &p_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &n_blk as *const _ as *mut c_void,
            &max_spans as *const _ as *mut c_void,
            &batch_offset as *const _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, rows as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_tiles_cover_six_heads_per_row() {
        for rows in 1..=WMMA_VERIFY_ATTN_MAX_ROWS {
            let qt = wmma_verify_attn_qt(rows);
            assert!(
                qt * 16 >= GQA * rows && (qt - 1) * 16 < GQA * rows,
                "rows={rows}"
            );
            assert!((1..=WMMA_VERIFY_ATTN_SYMBOLS.len()).contains(&qt));
        }
        assert_eq!(wmma_verify_attn_qt(4), 2); // MTP K=3
    }

    #[test]
    fn span_policy_and_partials() {
        assert_eq!(wmma_verify_attn_spans(4, 9_000), 24);
        assert_eq!(wmma_verify_attn_spans(4, 40_000), 36);
        assert_eq!(wmma_verify_attn_spans(8, 100_000), 24);
        // 4 rows x 24 heads x 36 spans x 260 floats = 3.6 MB at long context.
        assert_eq!(wmma_verify_attn_partials_len(4, 24, 100_000), 898_560);
    }

    #[test]
    fn symbols_live_in_the_registered_gfx1100_module() {
        let entries = crate::kernel_registry::entries("gfx1100", "").unwrap();
        let entry = entries
            .iter()
            .find(|e| e.module == WMMA_VERIFY_ATTN_MODULE)
            .expect("module registered for gfx1100");
        assert_eq!(entry.source(), kernels::ATTENTION_VERIFY_WMMA_GFX1100_SRC);
        for s in WMMA_VERIFY_ATTN_SYMBOLS
            .iter()
            .chain([&WMMA_VERIFY_ATTN_REDUCE])
        {
            assert!(entry.symbols.contains(s), "{s}");
        }
        for arch in ["gfx1151", "gfx1201", "gfx906", "gfx942"] {
            let other = crate::kernel_registry::entries(arch, "").unwrap();
            assert!(
                other.iter().all(|e| e.module != WMMA_VERIFY_ATTN_MODULE),
                "{arch}"
            );
        }
    }
}
