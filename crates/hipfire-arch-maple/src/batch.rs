// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched-prefill scratch, chunk math, and the dense qt51 GEMM.

use rdna_compute::{DType, Gpu, GpuTensor};

/// Row-tile granularity of the grouped WMMA kernels.
pub const MOE_GROUPED_BLOCK_M: usize = 16;

/// Default prompt chunk. Larger B raises tokens-per-expert
/// (`B*k_top/n_exp`) and shrinks BLOCK_M padding waste, which is why this is
/// 256 rather than 64.
pub const MAPLE_PREFILL_CHUNK: usize = 256;

/// Hard scratch ceiling. `forward_batch` ERRORS above this rather than
/// silently splitting — splitting is the caller's job.
pub const MAPLE_PREFILL_MAX_B: usize = 512;

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Padded row count for a DENSE (single-expert) grouped GEMM over `b` rows.
pub fn dense_m_total(b: usize) -> usize {
    align_up(b, MOE_GROUPED_BLOCK_M)
}

/// Upper bound on the padded scattered-slot count. Every LIVE expert can waste
/// up to `BLOCK_M-1` pad slots; with fewer slots than experts, only
/// `total_slots` experts can be live.
pub fn moe_grouped_m_total_bound(total_slots: usize, n_exp: usize) -> usize {
    let live = total_slots.min(n_exp);
    align_up(
        total_slots + live * (MOE_GROUPED_BLOCK_M - 1),
        MOE_GROUPED_BLOCK_M,
    )
}

/// Split `n_tokens` into `(start, len)` chunks of at most `chunk`.
pub fn prefill_chunks(n_tokens: usize, chunk: usize) -> Vec<(usize, usize)> {
    assert!(chunk > 0, "chunk must be positive");
    let mut out = Vec::new();
    let mut start = 0;
    while start < n_tokens {
        let n = chunk.min(n_tokens - start);
        out.push((start, n));
        start += n;
    }
    out
}

/// Host-side `sorted_slot_index` for a dense (single-expert) GEMM over `b`
/// rows: identity for real rows, `-1` for the BLOCK_M padding tail so the
/// kernel skips it.
pub fn dense_slot_index_host(b: usize) -> Vec<i32> {
    let m_total = dense_m_total(b);
    let mut v = vec![-1i32; m_total];
    for (i, slot) in v.iter_mut().enumerate().take(b) {
        *slot = i as i32;
    }
    v
}

/// Host-side `expert_tile_ids`: every row tile uses expert 0.
pub fn dense_tile_ids_host(b: usize) -> Vec<i32> {
    vec![0i32; dense_m_total(b) / MOE_GROUPED_BLOCK_M]
}

/// Dense `Y[b × m] = X[b × k] @ W[m × k]^T` for an `MQ2G256LloydU` weight,
/// by driving the grouped MoE WMMA kernel as a SINGLE-EXPERT case.
///
/// `w_ptrs` is a 1-entry `[2] f32` table holding the weight's device pointer,
/// `tile_ids` is all-zero, `slot_index` is identity+(-1) padding, and
/// `x_row_div = 1` because for a dense call the slot index IS the row index
/// (the MoE case divides by `k_top`).
///
/// `x_f16_src` MUST be an **F16** tensor the caller converted itself, laid out
/// `[b × k]` row-major. The grouped entry hands an F16 `x_src` straight to the
/// kernel (whose `X_src` is `_Float16*`) and never touches the shared FP16
/// scratch.
///
/// Passing an F32 tensor here still "works" but routes through the kernel's
/// `ensure_fp16_x`, whose conversion is cached keyed on the SOURCE POINTER. A
/// caller that reuses one F32 scratch buffer across layers with new contents
/// would then get the FIRST layer's activations for every later layer —
/// silently, with no error. Maple converts explicitly for exactly that reason.
#[allow(clippy::too_many_arguments)]
pub fn dense_qt51_gemm(
    gpu: &mut Gpu,
    w_ptrs: &GpuTensor,
    tile_ids: &GpuTensor,
    slot_index: &GpuTensor,
    x_f16_src: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    b: usize,
) -> Result<(), String> {
    // Enforce the contract documented above. Passing F32 here does not fail —
    // it quietly routes through the kernel's pointer-keyed `ensure_fp16_x` and
    // serves the FIRST layer's activations for every later layer. Debug-only,
    // so release pays nothing.
    debug_assert_eq!(
        x_f16_src.dtype,
        DType::F16,
        "dense_qt51_gemm: x must be pre-converted F16"
    );
    gpu.gemm_mq2g256_lloyd_moe_grouped_wmma(
        w_ptrs,
        tile_ids,
        slot_index,
        x_f16_src,
        y,
        m,
        k,
        1, // x_row_div: slot index IS the row index for a dense call
        dense_m_total(b),
        b,
    )
    .map_err(|e| format!("maple: dense qt51 gemm (m={m} k={k} b={b}): {e:?}"))
}

/// Upload a 1-entry device-pointer table for one weight, in the `[2] f32`
/// (8-byte) layout the indexed kernels expect.
pub fn upload_single_expert_ptr_table(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
) -> Result<GpuTensor, String> {
    let bytes = (w.buf.buf.as_ptr() as u64).to_ne_bytes();
    let t = gpu
        .alloc_tensor(&[2], DType::F32)
        .map_err(|e| format!("maple: alloc ptr table: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&t.buf, &bytes)
        .map_err(|e| format!("maple: htod ptr table: {e:?}"))?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_m_total_rounds_up_to_block_m() {
        // The grouped kernel works in BLOCK_M=16 row tiles. B=17 must round to
        // 32, leaving 15 padding rows whose output must never be read back.
        assert_eq!(dense_m_total(1), 16);
        assert_eq!(dense_m_total(16), 16);
        assert_eq!(dense_m_total(17), 32);
        assert_eq!(dense_m_total(256), 256);
    }

    #[test]
    fn moe_m_total_bound_covers_worst_case_padding() {
        // Every LIVE expert can waste up to BLOCK_M-1 pad slots. With more
        // slots than experts, all n_exp are live.
        let slots = 256 * 8;
        let bound = moe_grouped_m_total_bound(slots, 256);
        assert!(bound >= slots, "bound must cover the real slots");
        assert!(bound >= slots + 256 * (MOE_GROUPED_BLOCK_M - 1) - 15);
        assert_eq!(bound % MOE_GROUPED_BLOCK_M, 0, "must be a whole tile count");
    }

    #[test]
    fn moe_m_total_bound_uses_live_experts_not_all_experts() {
        // 2 slots cannot light up 256 experts; the bound must not pad for 256.
        let bound = moe_grouped_m_total_bound(2, 256);
        assert!(bound < 256 * MOE_GROUPED_BLOCK_M, "over-padded: {bound}");
    }

    #[test]
    fn chunks_tile_the_prompt_exactly_and_in_order() {
        let c = prefill_chunks(600, 256);
        assert_eq!(c, vec![(0, 256), (256, 256), (512, 88)]);
        // Total covered == prompt length, no gaps, no overlap.
        assert_eq!(c.iter().map(|(_, n)| n).sum::<usize>(), 600);
        let mut next = 0;
        for (start, n) in c {
            assert_eq!(start, next);
            next += n;
        }
    }

    #[test]
    fn chunks_handle_exact_multiples_and_short_prompts() {
        assert_eq!(prefill_chunks(256, 256), vec![(0, 256)]);
        assert_eq!(prefill_chunks(1, 256), vec![(0, 1)]);
        assert_eq!(prefill_chunks(0, 256), vec![]);
    }

    #[test]
    fn dense_slot_index_is_identity_then_minus_one_padding() {
        // Real rows map to themselves; padding rows MUST be -1 so the kernel
        // skips them. A 0 there would silently recompute row 0 into the tail.
        let v = dense_slot_index_host(17);
        assert_eq!(v.len(), 32, "padded to BLOCK_M");
        for (i, s) in v.iter().enumerate().take(17) {
            assert_eq!(*s, i as i32);
        }
        assert!(v[17..].iter().all(|&s| s == -1), "padding must be -1");
    }

    #[test]
    fn dense_tile_ids_are_all_expert_zero() {
        // Single-expert specialization: every 16-row tile uses expert 0.
        let t = dense_tile_ids_host(256);
        assert_eq!(t.len(), 256 / 16);
        assert!(t.iter().all(|&e| e == 0));
    }
}
