// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE scatter, permute, combine, and unscatter dispatch methods.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

/// The shared expert's BF16 activation carried by a fused MoE launch: the
/// operands of [`Gpu::shared_expert_activation_bf16_f32`].
pub struct SharedExpertActivation<'a> {
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub out: &'a GpuTensor,
    pub selector: &'a GpuTensor,
    pub selectors: usize,
}

/// Physical-span guard shared by the launchers below and the IU4 launchers in
/// `gemm.rs`: `Err` unless `tensor`'s device buffer holds at least the product
/// of `dims` bytes (callers pass `[rows, width, ..., element_bytes]`).  It
/// reads the allocation, not the tensor's logical shape, so a short buffer is
/// refused before the launch rather than overrun by it.
pub(crate) fn require_span(tensor: &GpuTensor, dims: &[usize], what: &str) -> HipResult<()> {
    let need = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("{what}: byte span overflows")))?;
    let have = tensor.buf.size();
    if have < need {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("{what}: buffer holds {have} bytes, the launch needs {need}"),
        ));
    }
    Ok(())
}

/// Kernel behind the Qwen4 grouped QT53 down launcher
/// ([`Gpu::gemm_mq4g128v2_moe_grouped_top10`]); all three are bitwise equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoeDownKernel {
    /// Portable generic kernel, one output row per workgroup.
    Multirow,
    /// gfx11+ exact-shape O8×R16 (one wave owns a 16-slot tile).
    O8R16,
    /// gfx1151 X-in-LDS twin of [`Self::O8R16`] (four waves share each
    /// group's X).
    XsTwin,
}

impl Gpu {
    /// Combine pass for the atomic-free MoE down path. Sums K_TOP expert
    /// outputs per (token, m) weighted by topk_weights, accumulates into
    /// the residual stream. No cross-token contention — each token writes
    /// to its own M-column slice.
    pub fn moe_down_combine_k8_batched(
        &mut self,
        expert_outputs: &GpuTensor, // [batch_size × k_top × m] f32
        topk_weights: &GpuTensor,   // [batch_size × k_top] f32
        x_residual: &GpuTensor,     // [batch_size × m] f32 in-place +=
        m: usize,
        k_top: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let use_vec4 = self.arch_caps.is_rdna3_dgpu()
            && self.flags.moe_down_combine_vec4
            && k_top == 8
            && m.is_multiple_of(4);
        let func_name = if use_vec4 {
            self.ensure_kernel(
                "moe_down_combine_k8_batched_vec4",
                kernels::MOE_DOWN_COMBINE_K8_BATCHED_VEC4_SRC,
                "moe_down_combine_k8_batched_vec4",
            )?;
            "moe_down_combine_k8_batched_vec4"
        } else {
            self.ensure_kernel(
                "moe_down_combine_k8_batched",
                kernels::MOE_DOWN_COMBINE_K8_BATCHED_SRC,
                "moe_down_combine_k8_batched",
            )?;
            "moe_down_combine_k8_batched"
        };
        let eop = expert_outputs.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let m_val = m as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &eop as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        // BW: expert_outputs read N*K_TOP*M, topk_weights N*K_TOP, x_residual r+w 2*N*M.
        let bytes = (batch_size * k_top * m + batch_size * k_top + 2 * batch_size * m) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_down_combine_k8_batched",
            bytes,
        );
        let block_m: u32 = 256;
        let columns_per_thread = if use_vec4 { 4 } else { 1 };
        let grid_x = (m as u32).div_ceil(block_m * columns_per_thread);
        let result = self.launch_maybe_blob(
            func_name,
            [batch_size as u32, grid_x, 1],
            [block_m, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(eop);
                b.push_ptr(wp);
                b.push_ptr(xrp);
                b.push_i32(m_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// gfx1100 decode experiment: finish one atomic-free routed-MoE row and
    /// immediately produce the following layer's RMS-normalized MQ rotation.
    /// This replaces `moe_down_combine_k8_batched` plus
    /// `fused_rmsnorm_mq_rotate_vecsum` while retaining the exact arithmetic
    /// order of both kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100(
        &mut self,
        expert_outputs: &GpuTensor,
        topk_weights: &GpuTensor,
        x_residual: &GpuTensor,
        norm_weight: &GpuTensor,
        x_rot: &GpuTensor,
        m: usize,
        k_top: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        let (module, src, kernel) = if self.arch_caps.is_gfx1151() {
            (
                "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
                kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1151_SRC,
                "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
            )
        } else {
            (
                "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1100",
                kernels::MOE_DOWN_COMBINE_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC,
                "moe_down_combine_rmsnorm_mq_rotate_vecsum",
            )
        };
        self.ensure_kernel(module, src, kernel)?;

        let eop = expert_outputs.buf.as_ptr();
        let twp = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let nwp = norm_weight.buf.as_ptr();
        let s1p = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2p = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xop = x_rot.buf.as_ptr();
        let mv = m as i32;
        let ktv = k_top as i32;
        let epsv = eps;
        let mut params: Vec<*mut c_void> = vec![
            &eop as *const _ as *mut c_void,
            &twp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &nwp as *const _ as *mut c_void,
            &s1p as *const _ as *mut c_void,
            &s2p as *const _ as *mut c_void,
            &xop as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &ktv as *const _ as *mut c_void,
            &epsv as *const _ as *mut c_void,
        ];

        let bytes = (k_top * m + k_top + 2 * m + m + 512 + m) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "fused", kernel, bytes);
        let result = self.launch_maybe_blob(kernel, [1, 1, 1], [256, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(eop);
            b.push_ptr(twp);
            b.push_ptr(xrp);
            b.push_ptr(nwp);
            b.push_ptr(s1p);
            b.push_ptr(s2p);
            b.push_ptr(xop);
            b.push_i32(mv);
            b.push_i32(ktv);
            b.push_f32(epsv);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.scratch.invalidate_x_caches_for(xrp);
        self.scratch.invalidate_x_caches_for(xop);
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 1: per-expert histogram.
    /// Single-CTA LDS-atomic histogram of `topk_indices[total_slots]`.
    /// Output `expert_token_counts[num_experts]` holds RAW counts; Phase 2
    /// rewrites them in place as padded counts.
    pub fn moe_scatter_histogram_k8(
        &mut self,
        topk_indices: &GpuTensor,        // [total_slots] i32
        expert_token_counts: &GpuTensor, // [num_experts] i32, written
        total_slots: usize,
        num_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_histogram_k8",
            kernels::MOE_SCATTER_HISTOGRAM_K8_SRC,
            "moe_scatter_histogram_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let cp = expert_token_counts.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (total_slots + num_experts) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_scatter_histogram_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_scatter_histogram_k8",
            [1, 1, 1],
            [256, 1, 1],
            lds_bytes,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(cp);
                b.push_i32(ts_val);
                b.push_i32(ne_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 2: pad + exclusive scan.
    /// Rewrites `expert_token_counts` raw → padded (to a multiple of
    /// `block_m`) and writes `expert_offsets[num_experts + 1]` with the
    /// exclusive prefix sum. `expert_offsets[num_experts]` is M_total.
    pub fn moe_scatter_offsets_k8(
        &mut self,
        expert_token_counts: &GpuTensor, // [E] i32, in: raw, out: padded
        expert_offsets: &GpuTensor,      // [E+1] i32, written
        num_experts: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_offsets_k8",
            kernels::MOE_SCATTER_OFFSETS_K8_SRC,
            "moe_scatter_offsets_k8",
        )?;
        let cp = expert_token_counts.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let ne_val = num_experts as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (3 * num_experts + 1) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "moe_scatter_offsets_k8", bytes);
        let result = self.launch_maybe_blob(
            "moe_scatter_offsets_k8",
            [1, 1, 1],
            [256, 1, 1],
            lds_bytes,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_i32(ne_val);
                b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// SGLang-style MoE scatter pipeline — Phase 3: scatter + tile ids.
    /// Writes `sorted_slot_index[m_total]` with each flat slot index at
    /// its bucket position (padding stays at the -1 sentinel) and
    /// `expert_tile_ids[m_total / block_m]` for the grouped-GEMM loop.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_permute_k8(
        &mut self,
        topk_indices: &GpuTensor,      // [total_slots] i32
        expert_offsets: &GpuTensor,    // [E+1] i32, exclusive padded scan
        sorted_slot_index: &GpuTensor, // [m_total] i32, written
        expert_tile_ids: &GpuTensor,   // [m_total / block_m] i32, written
        inverse_perm: &GpuTensor,      // [total_slots] i32, written: flat → sorted_pos
        total_slots: usize,
        num_experts: usize,
        m_total: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_permute_k8",
            kernels::MOE_SCATTER_PERMUTE_K8_SRC,
            "moe_scatter_permute_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let invp = inverse_perm.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mt_val = m_total as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &invp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        // BW: topk_indices + offsets + sorted_slot_index (init + writes)
        //     + expert_tile_ids (writes).
        let bytes = (total_slots + num_experts + 2 * m_total + m_total / block_m.max(1)) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "moe_scatter_permute_k8", bytes);
        let result = self.launch_maybe_blob(
            "moe_scatter_permute_k8",
            [1, 1, 1],
            [256, 1, 1],
            lds_bytes,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_ptr(sp);
                b.push_ptr(tp);
                b.push_ptr(invp);
                b.push_i32(ts_val);
                b.push_i32(ne_val);
                b.push_i32(mt_val);
                b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fused single-CTA scatter pipeline. Replaces histogram + offsets +
    /// permute with one launch — saves ~2 launches × ~75µs per MoE layer
    /// (≈2-3ms across 40 A3B layers).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_fused_k8(
        &mut self,
        topk_indices: &GpuTensor,        // [total_slots] i32
        expert_token_counts: &GpuTensor, // [E] i32, out: padded
        expert_offsets: &GpuTensor,      // [E+1] i32, out: exclusive scan
        sorted_slot_index: &GpuTensor,   // [m_total_max] i32, out
        expert_tile_ids: &GpuTensor,     // [m_total / block_m] i32, out
        inverse_perm: &GpuTensor,        // [total_slots] i32, out
        total_slots: usize,
        num_experts: usize,
        m_total_max: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_scatter_fused_k8",
            kernels::moe_scatter_fused_k8_src(self.arch_caps.is_gfx1151()),
            "moe_scatter_fused_k8",
        )?;
        let ip = topk_indices.buf.as_ptr();
        let cp = expert_token_counts.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let invp = inverse_perm.buf.as_ptr();
        let ts_val = total_slots as i32;
        let ne_val = num_experts as i32;
        let mtm_val = m_total_max as i32;
        let bm_val = block_m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &invp as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
            &ne_val as *const _ as *mut c_void,
            &mtm_val as *const _ as *mut c_void,
            &bm_val as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (total_slots + 2 * num_experts + 2 * total_slots + num_experts) * 4;
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "moe_scatter_fused_k8", bytes);
        let result = self.launch_maybe_blob(
            "moe_scatter_fused_k8",
            [1, 1, 1],
            [256, 1, 1],
            lds_bytes,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_ptr(sp);
                b.push_ptr(tp);
                b.push_ptr(invp);
                b.push_i32(ts_val);
                b.push_i32(ne_val);
                b.push_i32(mtm_val);
                b.push_i32(bm_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Path 2 down combine. Per (token, m) iterates K_TOP slots via
    /// `inverse_perm[token*K_TOP + k]`, applies topk_weights, and += into
    /// `x_residual`. No atomic contention (each (token, m) is owned by
    /// one thread).
    pub fn moe_down_combine_grouped_k8(
        &mut self,
        y_down_grouped: &GpuTensor, // [m_total × dim] f32
        inverse_perm: &GpuTensor,   // [N*K_TOP] i32
        topk_weights: &GpuTensor,   // [N × K_TOP] f32
        x_residual: &GpuTensor,     // [N × dim] f32 in-place +=
        dim: usize,
        k_top: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_down_combine_grouped_k8",
            kernels::MOE_DOWN_COMBINE_GROUPED_K8_SRC,
            "moe_down_combine_grouped_k8",
        )?;
        let yp = y_down_grouped.buf.as_ptr();
        let ip = inverse_perm.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let xrp = x_residual.buf.as_ptr();
        let dim_val = dim as i32;
        let kt_val = k_top as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (dim as u32 + block - 1) / block;
        let bytes = (n * dim * 4 * 2 + n * k_top * 4 + n * k_top * 4) as usize;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_down_combine_grouped_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_down_combine_grouped_k8",
            [n as u32, grid_x, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(xrp);
                b.push_i32(dim_val);
                b.push_i32(kt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Materialize grouped path-2 down rows in canonical flat
    /// `(token, k_rank)` slot order. `inverse_perm[flat_slot]` points to the
    /// grouped row produced by the rank-local atomic bucket order. This pass
    /// only copies live slots; the root applies top-k weights and combines
    /// them after cross-rank gathering.
    pub fn moe_down_unscatter_k8(
        &mut self,
        y_down_grouped: &GpuTensor, // [m_total × dim] f32
        inverse_perm: &GpuTensor,   // [total_slots] i32
        down_expanded: &GpuTensor,  // [capacity × dim] f32, written
        dim: usize,
        total_slots: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_down_unscatter_k8",
            kernels::MOE_DOWN_UNSCATTER_K8_SRC,
            "moe_down_unscatter_k8",
        )?;
        let yp = y_down_grouped.buf.as_ptr();
        let ip = inverse_perm.buf.as_ptr();
        let ep = down_expanded.buf.as_ptr();
        let dim_val = dim as i32;
        let ts_val = total_slots as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &ep as *const _ as *mut c_void,
            &dim_val as *const _ as *mut c_void,
            &ts_val as *const _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_y = (dim as u32).div_ceil(block);
        // Y_down_grouped read + down_expanded write + inverse_perm read.
        let bytes = (total_slots * dim * 4 * 2 + total_slots * 4) as usize;
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", "moe_down_unscatter_k8", bytes);
        let result = self.launch_maybe_blob(
            "moe_down_unscatter_k8",
            // Keep total_slots in grid.x; grid.y is the column tile.
            [total_slots as u32, grid_y, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(ip);
                b.push_ptr(ep);
                b.push_i32(dim_val);
                b.push_i32(ts_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Path 2 unscatter combine for gate_up. Reads Y_grouped[m_total ×
    /// 2*mi] and writes the gate half (rows 0..mi) into `y_gate[token,
    /// k_rank, :]` and the up half (rows mi..2*mi) into `y_up[token,
    /// k_rank, :]`, where (token, k_rank) is recovered from
    /// `sorted_slot_index[slot]`. Padding slots are skipped.
    pub fn moe_gate_up_unscatter_k8(
        &mut self,
        y_grouped: &GpuTensor,         // [m_total × (2*mi)] f32
        sorted_slot_index: &GpuTensor, // [m_total] i32
        y_gate: &GpuTensor,            // [N × K_TOP × mi] f32, written
        y_up: &GpuTensor,              // [N × K_TOP × mi] f32, written
        mi: usize,
        k_top: usize,
        m_total: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_gate_up_unscatter_k8",
            kernels::MOE_GATE_UP_UNSCATTER_K8_SRC,
            "moe_gate_up_unscatter_k8",
        )?;
        let yp = y_grouped.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let gp = y_gate.buf.as_ptr();
        let up = y_up.buf.as_ptr();
        let mi_val = mi as i32;
        let kt_val = k_top as i32;
        let mt_val = m_total as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (mi as u32 + block - 1) / block;
        // BW: Y_grouped read (m_total*2*mi*4) + y_gate write (m_total*mi*4)
        //     + y_up write (m_total*mi*4) + sorted_slot_index (m_total*4).
        let bytes = (m_total * 2 * mi + m_total * 2 * mi + m_total) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_gate_up_unscatter_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_gate_up_unscatter_k8",
            // m_total in grid.x (limit 2^31), mi-tile in grid.y — m_total exceeds
            // the 65535 grid.y limit past ~8k prefill tokens (m_total = N*K_TOP).
            [m_total as u32, grid_x, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(gp);
                b.push_ptr(up);
                b.push_i32(mi_val);
                b.push_i32(kt_val);
                b.push_i32(mt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    pub fn moe_unscatter_silu_clamp_k8(
        &mut self,
        y_grouped: &GpuTensor,         // [m_total × (2*mi)] f32
        sorted_slot_index: &GpuTensor, // [m_total] i32
        moe_gate_batch: &GpuTensor,    // [N × K_TOP × mi] f32, written
        mi: usize,
        k_top: usize,
        m_total: usize,
        swiglu_limit: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "moe_unscatter_silu_clamp_k8",
            kernels::MOE_UNSCATTER_SILU_CLAMP_K8_SRC,
            "moe_unscatter_silu_clamp_k8",
        )?;
        let yp = y_grouped.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let gp = moe_gate_batch.buf.as_ptr();
        let mi_val = mi as i32;
        let kt_val = k_top as i32;
        let mt_val = m_total as i32;
        let mut swiglu_lim = swiglu_limit;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
            &mt_val as *const _ as *mut c_void,
            &mut swiglu_lim as *mut _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid_x = (mi as u32 + block - 1) / block;
        // BW: Y_grouped read (m_total*2*mi*4) + moe_gate_batch write
        // (m_total*mi*4) + sorted_slot_index (m_total*4).  Half the
        // write traffic vs the unfused path (no y_up output).
        let bytes = (m_total * 2 * mi + m_total * mi + m_total) * 4;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            "moe_unscatter_silu_clamp_k8",
            bytes,
        );
        let result = self.launch_maybe_blob(
            "moe_unscatter_silu_clamp_k8",
            // m_total on grid.x (limit 2^31); the mi tile stays on grid.y.
            // grid.y = m_total exceeds 65535 for top-8 chunks >= 8192 tokens.
            [m_total as u32, grid_x, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(gp);
                b.push_i32(mi_val);
                b.push_i32(kt_val);
                b.push_i32(mt_val);
                b.push_f32(swiglu_lim);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    pub fn hash_router_normalize_f32(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        token_id: i32,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32",
            kernels::HASH_ROUTER_NORMALIZE_SRC,
            "hash_router_normalize_f32",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut tid = token_id;
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(sp);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_i32(tid);
            b.push_i32(ne);
            b.push_i32(kv);
            b.push_f32(rs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32",
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn hash_router_normalize_f32_batched(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_ids: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_batched",
            kernels::HASH_ROUTER_NORMALIZE_BATCHED_SRC,
            "hash_router_normalize_f32_batched",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_ids.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tb as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(sp);
            b.push_ptr(tb);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_i32(ne);
            b.push_i32(kv);
            b.push_f32(rs);
            b.push_i32(bs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32_batched",
            [batch_size as u32, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn hash_router_normalize_f32_buf(
        &mut self,
        tid2eid: &GpuTensor,
        scores: &GpuTensor,
        token_id_buf: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        n_exp: i32,
        k: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "hash_router_normalize_f32_buf",
            kernels::HASH_ROUTER_NORMALIZE_BUF_SRC,
            "hash_router_normalize_f32_buf",
        )?;
        let tp = tid2eid.buf.as_ptr();
        let sp = scores.buf.as_ptr();
        let tb = token_id_buf.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let mut ne = n_exp;
        let mut kv = k;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tb as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(sp);
            b.push_ptr(tb);
            b.push_ptr(ip);
            b.push_ptr(wp);
            b.push_i32(ne);
            b.push_i32(kv);
            b.push_f32(rs);
            b
        };
        self.launch_maybe_blob(
            "hash_router_normalize_f32_buf",
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_moe_topk_bias_aware_batched_f32(
        &mut self,
        scores: &GpuTensor,  // [B, n_exp]
        bias: &GpuTensor,    // [n_exp]
        indices: &GpuTensor, // [B, k_top]
        weights: &GpuTensor, // [B, k_top]
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware_batched",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_BATCHED_SRC,
            "deepseek4_moe_topk_bias_aware_batched_f32",
        )?;
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "deepseek4_moe_topk_bias_aware_batched_f32",
            [batch_size as u32, 1, 1],
            [n_exp as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(bp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(ne);
                b.push_i32(kt);
                b.push_f32(rs);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn deepseek4_moe_topk_bias_aware_f32(
        &mut self,
        scores: &GpuTensor,  // [n_exp] fp32
        bias: &GpuTensor,    // [n_exp] fp32 (zero if hash-routed)
        indices: &GpuTensor, // [k_top] i32 (typed as F32; raw bytes)
        weights: &GpuTensor, // [k_top] fp32
        n_exp: i32,
        k_top: i32,
        route_scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_moe_topk_bias_aware",
            kernels::V4F_MOE_TOPK_BIAS_AWARE_SRC,
            "deepseek4_moe_topk_bias_aware_f32",
        )?;
        let sp = scores.buf.as_ptr();
        let bp = bias.buf.as_ptr();
        let ip = indices.buf.as_ptr();
        let wp = weights.buf.as_ptr();
        let mut ne = n_exp;
        let mut kt = k_top;
        let mut rs = route_scale;
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut kt as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "deepseek4_moe_topk_bias_aware_f32",
            [1, 1, 1],
            [n_exp as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(bp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(ne);
                b.push_i32(kt);
                b.push_f32(rs);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_batched_f32(
        &mut self,
        kv_cache: &GpuTensor, // [N_compressed, head_dim] shared
        topk_idx: &GpuTensor, // [B, K] i32
        out: &GpuTensor,      // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_batched",
            kernels::V4F_TOPK_KV_GATHER_BATCHED_SRC,
            "deepseek4_topk_kv_gather_batched_f32",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_batched_f32",
            [k_active as u32, batch_size as u32, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(nc);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_batched_tiled_gfx1201(
        &mut self,
        kv_cache: &GpuTensor, // [N_compressed, head_dim] shared
        topk_idx: &GpuTensor, // [B, K] i32
        out: &GpuTensor,      // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Portable LDS-transpose gather. Named for the arch that first shipped
        // it, but there is no gfx12 ISA dependency: 32x33 LDS tile + an index
        // cache, no WMMA, no gfx12 builtins. Verified to compile clean for
        // gfx1151 (VGPR 14, occupancy 16, LDS 4352, zero spills).
        debug_assert!(
            self.arch.eq_ignore_ascii_case("gfx1201") || self.arch.eq_ignore_ascii_case("gfx1151")
        );
        let kernel_name = "deepseek4_topk_kv_gather_batched_tiled_gfx1201";
        self.ensure_kernel(
            kernel_name,
            kernels::V4F_TOPK_KV_GATHER_BATCHED_TILED_GFX1201_SRC,
            kernel_name,
        )?;
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            kernel_name,
            [
                ((k_active + 31) / 32) as u32,
                ((head_dim + 31) / 32) as u32,
                batch_size as u32,
            ],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(nc);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201(
        &mut self,
        cache_ptrs: &[usize; 4],
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
        world: i32,
        block_rows: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.is_gfx1201());
        assert!(matches!(world, 3 | 4));
        let symbol = "deepseek4_topk_kv_gather_batched_tiled_sharded_gfx1201";
        self.ensure_kernel(
            symbol,
            kernels::V4F_TOPK_KV_GATHER_BATCHED_TILED_SHARDED_GFX1201_SRC,
            symbol,
        )?;
        let cp0 = cache_ptrs[0] as *mut c_void;
        let cp1 = cache_ptrs[1] as *mut c_void;
        let cp2 = cache_ptrs[2] as *mut c_void;
        let cp3 = cache_ptrs[3] as *mut c_void;
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut ranks = world;
        let mut block = block_rows;
        let mut params: Vec<*mut c_void> = vec![
            &cp0 as *const _ as *mut c_void,
            &cp1 as *const _ as *mut c_void,
            &cp2 as *const _ as *mut c_void,
            &cp3 as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut ranks as *mut _ as *mut c_void,
            &mut block as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [
                ((k_active + 31) / 32) as u32,
                ((head_dim + 31) / 32) as u32,
                batch_size as u32,
            ],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp0);
                b.push_ptr(cp1);
                b.push_ptr(cp2);
                b.push_ptr(cp3);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(nc);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b.push_i32(bs);
                b.push_i32(ranks);
                b.push_i32(block);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        max_k: i32, // upper bound on K — sets the captured grid size
        head_dim: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_BUF_SRC,
            "deepseek4_topk_kv_gather_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let ncp = n_compressed_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &ncp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp);
            b.push_ptr(ip);
            b.push_ptr(op);
            b.push_ptr(kbp);
            b.push_ptr(ncp);
            b.push_i32(hd);
            b.push_i32(os);
            b.push_i32(co);
            b.push_f32(sc);
            b
        };
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_f32_buf",
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_topk_kv_gather_f32_buf_sharded_gfx1201(
        &mut self,
        cache_ptrs: &[usize; 4],
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        world: i32,
        block_rows: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.is_gfx1201());
        assert!(matches!(world, 3 | 4));
        let symbol = "deepseek4_topk_kv_gather_f32_buf_sharded_gfx1201";
        self.ensure_kernel(
            symbol,
            kernels::V4F_TOPK_KV_GATHER_BUF_SHARDED_GFX1201_SRC,
            symbol,
        )?;
        let cp0 = cache_ptrs[0] as *mut c_void;
        let cp1 = cache_ptrs[1] as *mut c_void;
        let cp2 = cache_ptrs[2] as *mut c_void;
        let cp3 = cache_ptrs[3] as *mut c_void;
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let ncp = n_compressed_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut ranks = world;
        let mut block = block_rows;
        let mut params: Vec<*mut c_void> = vec![
            &cp0 as *const _ as *mut c_void,
            &cp1 as *const _ as *mut c_void,
            &cp2 as *const _ as *mut c_void,
            &cp3 as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &ncp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut ranks as *mut _ as *mut c_void,
            &mut block as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp0);
                b.push_ptr(cp1);
                b.push_ptr(cp2);
                b.push_ptr(cp3);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_ptr(kbp);
                b.push_ptr(ncp);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b.push_i32(ranks);
                b.push_i32(block);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_identity_batched_f32(
        &mut self,
        kv_cache: &GpuTensor, // [N_compressed, head_dim] shared
        out: &GpuTensor,      // [B, head_dim, out_stride]
        k_active: i32,
        head_dim: i32,
        out_stride: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_batched",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SRC,
            "deepseek4_topk_kv_gather_identity_batched_f32",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_identity_batched_f32",
            [k_active as u32, batch_size as u32, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(bs);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_identity_batched_sharded_gfx1201(
        &mut self,
        cache_ptrs: &[usize; 4],
        out: &GpuTensor,
        k_active: i32,
        head_dim: i32,
        out_stride: i32,
        batch_size: i32,
        world: i32,
        block_rows: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.is_gfx1201());
        assert!(matches!(world, 3 | 4));
        let symbol = "deepseek4_topk_kv_gather_identity_batched_sharded_gfx1201";
        self.ensure_kernel(
            symbol,
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BATCHED_SHARDED_GFX1201_SRC,
            symbol,
        )?;
        let cp0 = cache_ptrs[0] as *mut c_void;
        let cp1 = cache_ptrs[1] as *mut c_void;
        let cp2 = cache_ptrs[2] as *mut c_void;
        let cp3 = cache_ptrs[3] as *mut c_void;
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut bs = batch_size;
        let mut ranks = world;
        let mut block = block_rows;
        let mut params: Vec<*mut c_void> = vec![
            &cp0 as *const _ as *mut c_void,
            &cp1 as *const _ as *mut c_void,
            &cp2 as *const _ as *mut c_void,
            &cp3 as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut ranks as *mut _ as *mut c_void,
            &mut block as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [k_active as u32, batch_size as u32, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp0);
                b.push_ptr(cp1);
                b.push_ptr(cp2);
                b.push_ptr(cp3);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(bs);
                b.push_i32(ranks);
                b.push_i32(block);
                b
            },
        )
    }
    pub fn deepseek4_topk_kv_gather_identity_f32_buf(
        &mut self,
        kv_cache: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SRC,
            "deepseek4_topk_kv_gather_identity_f32_buf",
        )?;
        let cp = kv_cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp);
            b.push_ptr(op);
            b.push_ptr(kbp);
            b.push_i32(hd);
            b.push_i32(os);
            b
        };
        self.launch_maybe_blob(
            "deepseek4_topk_kv_gather_identity_f32_buf",
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }
    pub fn deepseek4_topk_kv_gather_identity_f32_buf_sharded_gfx1201(
        &mut self,
        cache_ptrs: &[usize; 4],
        out: &GpuTensor,
        k_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
        world: i32,
        block_rows: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.is_gfx1201());
        assert!(matches!(world, 3 | 4));
        let symbol = "deepseek4_topk_kv_gather_identity_f32_buf_sharded_gfx1201";
        self.ensure_kernel(
            symbol,
            kernels::V4F_TOPK_KV_GATHER_IDENTITY_BUF_SHARDED_GFX1201_SRC,
            symbol,
        )?;
        let cp0 = cache_ptrs[0] as *mut c_void;
        let cp1 = cache_ptrs[1] as *mut c_void;
        let cp2 = cache_ptrs[2] as *mut c_void;
        let cp3 = cache_ptrs[3] as *mut c_void;
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut ranks = world;
        let mut block = block_rows;
        let mut params: Vec<*mut c_void> = vec![
            &cp0 as *const _ as *mut c_void,
            &cp1 as *const _ as *mut c_void,
            &cp2 as *const _ as *mut c_void,
            &cp3 as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut ranks as *mut _ as *mut c_void,
            &mut block as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp0);
                b.push_ptr(cp1);
                b.push_ptr(cp2);
                b.push_ptr(cp3);
                b.push_ptr(op);
                b.push_ptr(kbp);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(ranks);
                b.push_i32(block);
                b
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_topk_kv_gather_f16_buf(
        &mut self,
        cache: &GpuTensor,
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        n_compressed_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.supports_ds4_f16_compressor_cache());
        assert_eq!(cache.dtype, DType::F16);
        let symbol = "deepseek4_topk_kv_gather_f16_buf";
        self.ensure_kernel(symbol, kernels::DEEPSEEK4_COMPRESSOR_CACHE_F16_SRC, symbol)?;
        let cp = cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let ncp = n_compressed_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &ncp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_ptr(kbp);
                b.push_ptr(ncp);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b
            },
        )
    }

    pub fn deepseek4_topk_kv_gather_identity_f16_buf(
        &mut self,
        cache: &GpuTensor,
        out: &GpuTensor,
        k_buf: &GpuTensor,
        max_k: i32,
        head_dim: i32,
        out_stride: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.supports_ds4_f16_compressor_cache());
        assert_eq!(cache.dtype, DType::F16);
        let symbol = "deepseek4_topk_kv_gather_identity_f16_buf";
        self.ensure_kernel(symbol, kernels::DEEPSEEK4_COMPRESSOR_CACHE_F16_SRC, symbol)?;
        let cp = cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let kbp = k_buf.buf.as_ptr();
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &kbp as *const _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [max_k as u32, 1, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_ptr(kbp);
                b.push_i32(hd);
                b.push_i32(os);
                b
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deepseek4_topk_kv_gather_batched_tiled_f16(
        &mut self,
        cache: &GpuTensor,
        topk_idx: &GpuTensor,
        out: &GpuTensor,
        k_active: i32,
        head_dim: i32,
        n_compressed: i32,
        out_stride: i32,
        col_offset: i32,
        scale: f32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.supports_ds4_f16_compressor_cache());
        assert_eq!(cache.dtype, DType::F16);
        let symbol = "deepseek4_topk_kv_gather_batched_tiled_f16";
        self.ensure_kernel(symbol, kernels::DEEPSEEK4_COMPRESSOR_CACHE_F16_SRC, symbol)?;
        let cp = cache.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut nc = n_compressed;
        let mut os = out_stride;
        let mut co = col_offset;
        let mut sc = scale;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [
                (k_active as u32).div_ceil(32),
                (head_dim as u32).div_ceil(32),
                batch_size as u32,
            ],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(nc);
                b.push_i32(os);
                b.push_i32(co);
                b.push_f32(sc);
                b.push_i32(bs);
                b
            },
        )
    }

    pub fn deepseek4_topk_kv_gather_identity_batched_f16(
        &mut self,
        cache: &GpuTensor,
        out: &GpuTensor,
        k_active: i32,
        head_dim: i32,
        out_stride: i32,
        batch_size: i32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(self.arch_caps.supports_ds4_f16_compressor_cache());
        assert_eq!(cache.dtype, DType::F16);
        let symbol = "deepseek4_topk_kv_gather_identity_batched_f16";
        self.ensure_kernel(symbol, kernels::DEEPSEEK4_COMPRESSOR_CACHE_F16_SRC, symbol)?;
        let cp = cache.buf.as_ptr();
        let op = out.buf.as_ptr();
        let mut k = k_active;
        let mut hd = head_dim;
        let mut os = out_stride;
        let mut bs = batch_size;
        let mut params: Vec<*mut c_void> = vec![
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut os as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            symbol,
            [k_active as u32, batch_size as u32, 1],
            [head_dim as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_i32(k);
                b.push_i32(hd);
                b.push_i32(os);
                b.push_i32(bs);
                b
            },
        )
    }
    /// Qwen4's fixed 512-way/top-10 GPU router.  The incumbent k=8 routers
    /// remain separate symbols and launchers.  `round_logits` (the BF16 recipe)
    /// reads the logits through [`Gpu::bf16_round_trip_f32`]'s rounding (the
    /// buffer is unchanged) and stores the weights through it too.
    ///
    /// Fail-closed: a row with any NaN or +/-Inf logit (after the optional
    /// rounding) gets expert id `-1` and weight `0` in all ten slots and is
    /// recorded as (`layer`, `row_base` + its row in this launch) in the
    /// device fault word ([`crate::router_fault`]); the host reports it at the
    /// next readback check ([`Gpu::take_device_wait_error`]). `layer` is the
    /// sealed call's layer index and `row_base` the global row of this launch's
    /// row 0 (the prefill chunk offset; 0 for decode). Finite rows are bitwise
    /// what they were before.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_router_softmax_top10_f32(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        tokens: usize,
        normalize_topk_prob: bool,
        round_logits: bool,
        layer: usize,
        row_base: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // `HIPFIRE_QWEN4_ROUTER_FAST` (exact gfx1151 / gfx1201): eager batched calls take
        // the wave-per-token kernel, bytewise the same outputs; decode (one
        // token), recording and graph capture keep the incumbent symbol.
        let fast = self.route_rows(tokens) > 1
            && self.flags.qwen4_router_fast_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode;
        let func: &str = if fast {
            "moe_router_softmax_top10_f32_fast"
        } else {
            "moe_router_softmax_top10_f32"
        };
        let src = if fast {
            kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_FAST_SRC
        } else {
            kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_SRC
        };
        self.ensure_kernel(func, src, func)?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let ne = 512i32;
        let kt = 10i32;
        let norm = i32::from(normalize_topk_prob);
        let round = i32::from(round_logits);
        let token_count = tokens as i32;
        // A non-finite row fails closed (ids -1, weights 0) and records
        // (`layer`, `row_base` + row) in the Gpu-lifetime fault word.
        let fault = self.router_fault_ptr();
        let layer_arg = layer as i32;
        let row_base_arg = row_base as i32;
        let mut params: Vec<*mut c_void> = vec![
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &kt as *const _ as *mut c_void,
            &norm as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
        ];
        if fast {
            params.push(&token_count as *const _ as *mut c_void);
        }
        params.push(&fault as *const _ as *mut c_void);
        params.push(&layer_arg as *const _ as *mut c_void);
        params.push(&row_base_arg as *const _ as *mut c_void);
        let bytes = (tokens * 512 + tokens * 10 * 2) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", func, bytes);
        let grid = if fast { tokens.div_ceil(8) } else { tokens } as u32;
        let result =
            self.launch_maybe_blob(func, [grid, 1, 1], [256, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(ne);
                b.push_i32(kt);
                b.push_i32(norm);
                b.push_i32(round);
                if fast {
                    b.push_i32(token_count);
                }
                b.push_ptr(fault as *const c_void);
                b.push_i32(layer_arg);
                b.push_i32(row_base_arg);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Fixed E512/top10 route with a load-owned 512-byte membership mask.
    /// `keep` must remain device-readable until this stream finishes.
    ///
    /// Same fail-closed fault ABI as [`Gpu::moe_router_softmax_top10_f32`]
    /// (`layer`, `row_base`, ids `-1` / weights `0`, the [`crate::router_fault`]
    /// word), except that excluded experts' logits are never read: they may be
    /// anything, including NaN/Inf, without faulting the row. A NaN or +/-Inf
    /// logit of a *kept* expert (after the optional rounding), or a failed
    /// softmax denominator, faults the row. Rows with finite kept logits are
    /// bitwise what they were before the fault ABI.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_router_softmax_top10_masked_f32(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        tokens: usize,
        normalize_topk_prob: bool,
        keep: usize,
        round_logits: bool,
        layer: usize,
        row_base: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // The kernel is fixed E512/top-10 and reads only the first 512 mask bytes.
        require_span(
            logits,
            &[tokens, 512, 4],
            "moe_router_softmax_top10_masked_f32 logits",
        )?;
        require_span(
            topk_idx,
            &[tokens, 10, 4],
            "moe_router_softmax_top10_masked_f32 ids",
        )?;
        require_span(
            topk_w,
            &[tokens, 10, 4],
            "moe_router_softmax_top10_masked_f32 weights",
        )?;
        if keep == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "moe_router_softmax_top10_masked_f32: null keep mask",
            ));
        }
        let func = "moe_router_softmax_top10_masked_f32";
        self.ensure_kernel(func, kernels::MOE_ROUTER_SOFTMAX_TOP10_MASKED_F32_SRC, func)?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let kp = keep as *mut c_void;
        let round = i32::from(round_logits);
        let ne = 512i32;
        let kt = 10i32;
        let norm = i32::from(normalize_topk_prob);
        let fault = self.router_fault_ptr();
        let layer_arg = layer as i32;
        let row_base_arg = row_base as i32;
        let mut params = [
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &kt as *const _ as *mut c_void,
            &norm as *const _ as *mut c_void,
            &round as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &fault as *const _ as *mut c_void,
            &layer_arg as *const _ as *mut c_void,
            &row_base_arg as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            func,
            [tokens as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_i32(ne);
                b.push_i32(kt);
                b.push_i32(norm);
                b.push_i32(round);
                b.push_ptr(kp);
                b.push_ptr(fault as *const c_void);
                b.push_i32(layer_arg);
                b.push_i32(row_base_arg);
                b
            },
        )
    }

    /// Qwen4 indexed combine.  `expert_outputs` stays unweighted until this
    /// source-ordered fixed-ten reduction.  The source implementation visits
    /// the selected experts in ascending expert-index order, while preserving
    /// each route slot's weight.
    pub fn moe_down_combine_top10_batched(
        &mut self,
        expert_outputs: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        hidden: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            expert_outputs,
            &[tokens, 10, hidden, 4],
            "moe_down_combine_top10_batched expert outputs",
        )?;
        const FUNC: &str = "moe_down_combine_top10_batched";
        self.ensure_kernel(FUNC, kernels::MOE_DOWN_COMBINE_TOP10_BATCHED_SRC, FUNC)?;
        let ep = expert_outputs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rp = residual.buf.as_ptr();
        let hv = hidden as i32;
        let tv = tokens as i32;
        let nx = n_experts as i32;
        let mut params = [
            &ep as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
        ];
        // 64 threads: a decode row spreads over 40 CUs instead of 10.
        let block = 64u32;
        let grid_x = (hidden as u32).div_ceil(block);
        let bytes = (tokens * 10 * hidden + 2 * tokens * 10 + 2 * tokens * hidden) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [grid_x, tokens as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(rp);
                b.push_i32(hv);
                b.push_i32(tv);
                b.push_i32(nx);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4 grouped combine.  `inverse_perm` is the sealed flat-to-grouped
    /// map and invalid rows are ignored by the bounds-safe kernel.  The
    /// top-k expert IDs are consumed to reproduce indexed BF16 expert order.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_combine_grouped_top10(
        &mut self,
        grouped_down: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        hidden: usize,
        grouped_rows: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        self.moe_down_combine_grouped_top10_impl(
            grouped_down,
            inverse_perm,
            topk_indices,
            topk_weights,
            residual,
            hidden,
            grouped_rows,
            tokens,
            n_experts,
        )
    }

    /// [`Gpu::moe_down_combine_grouped_top10`] reading the grouped rows as BF16 bits (written by a
    /// `*_bf16out` MoE GEMM).  The per-token rank order is resolved once into `order`
    /// (scratch of at least `tokens * 10 * 8` bytes) before the combine reads it.
    /// With `initial_zero` the combine starts from +0.0 instead of `residual`'s
    /// contents (bitwise the same as zero-filling `residual`'s `tokens * hidden`
    /// elements first).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_combine_grouped_top10_bf16in(
        &mut self,
        grouped_down: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        order: &GpuTensor,
        hidden: usize,
        grouped_rows: usize,
        tokens: usize,
        initial_zero: bool,
        n_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if hidden % 8 != 0 || order.buf.size() < tokens * 10 * 8 {
            return Err(hip_bridge::HipError::new(
                0,
                "moe_down_combine_grouped_top10_bf16in: hidden % 8 != 0 or order scratch too small",
            ));
        }
        require_span(
            grouped_down,
            &[grouped_rows, hidden, 2],
            "moe_down_combine_grouped_top10_bf16in grouped rows",
        )?;
        require_span(
            inverse_perm,
            &[tokens, 10, 4],
            "moe_down_combine_grouped_top10_bf16in inverse permutation",
        )?;
        require_span(
            residual,
            &[tokens, hidden, 4],
            "moe_down_combine_grouped_top10_bf16in target",
        )?;
        const MODULE: &str = "moe_down_combine_grouped_top10";
        // H8a (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`): `residual` is written, not
        // accumulated into; the caller then owes no preceding zero fill.
        let combine: &'static str = if initial_zero {
            "moe_down_combine_grouped_top10_bf16in_zinit"
        } else {
            "moe_down_combine_grouped_top10_bf16in"
        };
        for func in ["moe_combine_order_top10", combine] {
            self.ensure_kernel(MODULE, kernels::MOE_DOWN_COMBINE_GROUPED_TOP10_SRC, func)?;
        }
        let gp = grouped_down.buf.as_ptr();
        let ip = inverse_perm.buf.as_ptr();
        let tp = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rp = residual.buf.as_ptr();
        let op = order.buf.as_ptr();
        let hv = hidden as i32;
        let gr = grouped_rows as i32;
        let tv = tokens as i32;
        let nx = n_experts as i32;
        let mut params = [
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "moe_combine_order_top10",
            [(tokens as u32).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_ptr(op);
                b.push_i32(gr);
                b.push_i32(tv);
                b.push_i32(nx);
                b
            },
        )?;
        let mut params = [
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            combine,
            [(hidden as u32 / 8).div_ceil(64), tokens as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(op);
                b.push_ptr(rp);
                b.push_i32(hv);
                b.push_i32(tv);
                b
            },
        )
    }

    fn moe_down_combine_grouped_top10_impl(
        &mut self,
        grouped_down: &GpuTensor,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        residual: &GpuTensor,
        hidden: usize,
        grouped_rows: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            grouped_down,
            &[grouped_rows, hidden, 4],
            "moe_down_combine_grouped_top10 grouped rows",
        )?;
        require_span(
            inverse_perm,
            &[tokens, 10, 4],
            "moe_down_combine_grouped_top10 inverse permutation",
        )?;
        const FUNC: &str = "moe_down_combine_grouped_top10";
        self.ensure_kernel(FUNC, kernels::MOE_DOWN_COMBINE_GROUPED_TOP10_SRC, FUNC)?;
        let gp = grouped_down.buf.as_ptr();
        let ip = inverse_perm.buf.as_ptr();
        let tp = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rp = residual.buf.as_ptr();
        let hv = hidden as i32;
        let gr = grouped_rows as i32;
        let tv = tokens as i32;
        let nx = n_experts as i32;
        let mut params = [
            &gp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
        ];
        let (block, grid_x) = (256u32, (hidden as u32).div_ceil(256));
        let bytes = (grouped_rows * hidden + 3 * tokens * 10 + 2 * tokens * hidden) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [grid_x, tokens as u32, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(gp);
                b.push_ptr(ip);
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_ptr(rp);
                b.push_i32(hv);
                b.push_i32(gr);
                b.push_i32(tv);
                b.push_i32(nx);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4's independent top-10 scatter entry.  The implementation shares
    /// the proven single-CTA body through a new symbol; no k=8 launch is
    /// widened.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_fused_top10(
        &mut self,
        topk_indices: &GpuTensor,
        expert_token_counts: &GpuTensor,
        expert_offsets: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        inverse_perm: &GpuTensor,
        total_slots: usize,
        num_experts: usize,
        grouped_rows: usize,
        block_m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const FUNC: &str = "moe_scatter_fused_top10";
        self.ensure_kernel(FUNC, kernels::MOE_SCATTER_FUSED_TOP10_SRC, FUNC)?;
        let ip = topk_indices.buf.as_ptr();
        let cp = expert_token_counts.buf.as_ptr();
        let op = expert_offsets.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let invp = inverse_perm.buf.as_ptr();
        let ts = total_slots as i32;
        let ne = num_experts as i32;
        let gr = grouped_rows as i32;
        let bm = block_m as i32;
        let mut params = [
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &invp as *const _ as *mut c_void,
            &ts as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &bm as *const _ as *mut c_void,
        ];
        let lds_bytes = (num_experts * 4) as u32;
        let bytes = (total_slots + 2 * num_experts + 2 * total_slots + num_experts) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result =
            self.launch_maybe_blob(FUNC, [1, 1, 1], [256, 1, 1], lds_bytes, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(cp);
                b.push_ptr(op);
                b.push_ptr(sp);
                b.push_ptr(tp);
                b.push_ptr(invp);
                b.push_i32(ts);
                b.push_i32(ne);
                b.push_i32(gr);
                b.push_i32(bm);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4's independent top-10 gate/up unscatter entry.  The underlying
    /// permutation body has a dynamic K_TOP argument; this wrapper supplies
    /// the sealed value and uses a distinct symbol.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_unscatter_top10(
        &mut self,
        grouped_gate_up: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        gate: &GpuTensor,
        up: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            grouped_gate_up,
            &[grouped_rows, 2, mi, 4],
            "moe_gate_up_unscatter_top10 grouped rows",
        )?;
        require_span(
            gate,
            &[tokens, 10, mi, 4],
            "moe_gate_up_unscatter_top10 gate",
        )?;
        require_span(up, &[tokens, 10, mi, 4], "moe_gate_up_unscatter_top10 up")?;
        const FUNC: &str = "moe_gate_up_unscatter_top10";
        self.ensure_kernel(FUNC, kernels::MOE_GATE_UP_UNSCATTER_TOP10_SRC, FUNC)?;
        let yp = grouped_gate_up.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let up_ptr = up.buf.as_ptr();
        let mi_val = mi as i32;
        let kt_val = 10i32;
        let rows_val = grouped_rows as i32;
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &kt_val as *const _ as *mut c_void,
            &rows_val as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid_y = (mi as u32).div_ceil(block);
        let bytes = (grouped_rows * 2 * mi + tokens * 10 * 2 * mi + grouped_rows) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [grouped_rows as u32, grid_y, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(gp);
                b.push_ptr(up_ptr);
                b.push_i32(mi_val);
                b.push_i32(kt_val);
                b.push_i32(rows_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4 top-10 grouped gate/up unscatter fused with SwiGLU: writes the
    /// activation `[tokens × 10 × mi]` directly (BF16 round trips around the
    /// SwiGLU when `bf16_round_trip`), bitwise the unfused sequence of
    /// `moe_gate_up_unscatter_top10`, `bf16_round_trip_f32` on gate/up,
    /// `silu_mul_f32` and `bf16_round_trip_f32` on the activation.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_unscatter_silu_top10(
        &mut self,
        grouped_gate_up: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        bf16_round_trip: bool,
    ) -> HipResult<()> {
        self.moe_gate_up_unscatter_silu_top10_impl(
            grouped_gate_up,
            sorted_slot_index,
            activation,
            mi,
            grouped_rows,
            bf16_round_trip,
            false,
        )
    }

    /// Unscatter of the BF16 grouped SwiGLU activation (written by
    /// [`Gpu::gemm_mq4g256v2_moe_grouped_top10_silu_bf16out`]) fused with the
    /// 128-wide MQ4G128V2 rotation, written to the shared FP16 X scratch as
    /// the MoE down GEMM's input (`[grouped slots' flat rows x mi]`).  Bytes
    /// equal `moe_gate_up_unscatter_silu_top10_bf16in` -> `rotate_x_mq_128_v2_f16`.
    /// Valid until the next FP16 conversion.
    pub fn moe_unscatter_rotate128_f16(
        &mut self,
        grouped_act: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<GpuTensor> {
        if mi % 128 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "fused unscatter/rotate needs mi % 128 == 0",
            ));
        }
        self.bind_thread()?;
        require_span(
            grouped_act,
            &[grouped_rows, mi, 2],
            "moe_unscatter_rotate128_f16 grouped rows",
        )?;
        const FUNC: &str = "moe_unscatter_rotate128_f16";
        self.ensure_kernel(
            "moe_gate_up_unscatter_silu_top10",
            kernels::MOE_GATE_UP_UNSCATTER_SILU_TOP10_SRC,
            FUNC,
        )?;
        self.ensure_mq_signs_128()?;
        let out = self.qwen4_f16_x_scratch(total_slots * mi)?;
        let yp = grouped_act.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let op = out.buf.as_ptr();
        let mi_val = mi as i32;
        let rows_val = grouped_rows as i32;
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &rows_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [grouped_rows as u32, (mi / 128) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(mi_val);
                b.push_i32(rows_val);
                b
            },
        )?;
        Ok(out)
    }

    /// [`Gpu::moe_gate_up_unscatter_silu_top10`] followed by the in-place
    /// [`Gpu::rotate_x_mq_128_v2`] of its activation rows, one launch (bitwise
    /// both): `rotated` receives the MoE down's F32 input.  With `shared`, the
    /// same launch also runs [`Gpu::shared_expert_activation_bf16_f32`].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_unscatter_silu_rotate128_top10(
        &mut self,
        grouped_gate_up: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        rotated: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        bf16_round_trip: bool,
        shared: Option<&SharedExpertActivation<'_>>,
    ) -> HipResult<()> {
        if !mi.is_multiple_of(128) {
            return Err(hip_bridge::HipError::new(
                0,
                "fused unscatter/rotate needs mi % 128 == 0",
            ));
        }
        self.bind_thread()?;
        require_span(
            grouped_gate_up,
            &[grouped_rows, 2, mi, 4],
            "moe_gate_up_unscatter_silu_rotate128_top10 grouped rows",
        )?;
        const FUNC: &str = "moe_gate_up_unscatter_silu_rotate128_top10";
        self.ensure_kernel(
            "moe_gate_up_unscatter_silu_top10",
            kernels::MOE_GATE_UP_UNSCATTER_SILU_TOP10_SRC,
            FUNC,
        )?;
        self.ensure_mq_signs_128()?;
        let yp = grouped_gate_up.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let rp = rotated.buf.as_ptr();
        let mi_val = mi as i32;
        let rows_val = grouped_rows as i32;
        let rt_val = bf16_round_trip as i32;
        let null = std::ptr::null_mut::<c_void>();
        let (sgp, sup, sop, selp, shared_n, selectors) = match shared {
            Some(s) => (
                s.gate.buf.as_ptr(),
                s.up.buf.as_ptr(),
                s.out.buf.as_ptr(),
                s.selector.buf.as_ptr(),
                s.gate.numel(),
                s.selectors,
            ),
            None => (null, null, null, null, 0, 0),
        };
        let shared_blocks = shared_n.max(selectors).div_ceil(mi);
        let shared_n_val = shared_n as i32;
        let selectors_val = selectors as i32;
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &rows_val as *const _ as *mut c_void,
            &rt_val as *const _ as *mut c_void,
            &sgp as *const _ as *mut c_void,
            &sup as *const _ as *mut c_void,
            &sop as *const _ as *mut c_void,
            &selp as *const _ as *mut c_void,
            &shared_n_val as *const _ as *mut c_void,
            &selectors_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [(grouped_rows + shared_blocks) as u32, (mi / 128) as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(rp);
                b.push_i32(mi_val);
                b.push_i32(rows_val);
                b.push_i32(rt_val);
                b.push_ptr(sgp);
                b.push_ptr(sup);
                b.push_ptr(sop);
                b.push_ptr(selp);
                b.push_i32(shared_n_val);
                b.push_i32(selectors_val);
                b
            },
        )?;
        self.invalidate_x_caches_for(rp);
        if !sop.is_null() {
            self.invalidate_x_caches_for(sop);
        }
        Ok(())
    }

    /// [`Gpu::moe_gate_up_unscatter_silu_top10`] reading the grouped rows as BF16 bits (written by a
    /// `*_bf16out` MoE GEMM).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_unscatter_silu_top10_bf16in(
        &mut self,
        grouped_gate_up: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        bf16_round_trip: bool,
    ) -> HipResult<()> {
        self.moe_gate_up_unscatter_silu_top10_impl(
            grouped_gate_up,
            sorted_slot_index,
            activation,
            mi,
            grouped_rows,
            bf16_round_trip,
            true,
        )
    }

    fn moe_gate_up_unscatter_silu_top10_impl(
        &mut self,
        grouped_gate_up: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        activation: &GpuTensor,
        mi: usize,
        grouped_rows: usize,
        bf16_round_trip: bool,
        grouped_bf16: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            grouped_gate_up,
            &[grouped_rows, 2, mi, if grouped_bf16 { 2 } else { 4 }],
            "moe_gate_up_unscatter_silu_top10 grouped rows",
        )?;
        const FUNC: &str = "moe_gate_up_unscatter_silu_top10";
        let func = if grouped_bf16 {
            "moe_gate_up_unscatter_silu_top10_bf16in"
        } else {
            FUNC
        };
        self.ensure_kernel(FUNC, kernels::MOE_GATE_UP_UNSCATTER_SILU_TOP10_SRC, func)?;
        let yp = grouped_gate_up.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let ap = activation.buf.as_ptr();
        let mi_val = mi as i32;
        let rows_val = grouped_rows as i32;
        let rt_val = i32::from(bf16_round_trip);
        let mut params = [
            &yp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &mi_val as *const _ as *mut c_void,
            &rows_val as *const _ as *mut c_void,
            &rt_val as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid_y = (mi as u32).div_ceil(block);
        let bytes = (grouped_rows * 3 * mi + grouped_rows) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", func, bytes);
        let result = self.launch_maybe_blob(
            func,
            [grouped_rows as u32, grid_y, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(sp);
                b.push_ptr(ap);
                b.push_i32(mi_val);
                b.push_i32(rows_val);
                b.push_i32(rt_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4 qt3/Q8F16 indexed down.  The kernel writes unweighted expanded
    /// rows; `moe_down_combine_top10_batched` owns all route weighting.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_0_moe_down_top10_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        hidden_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const FUNC: &str = "gemv_q8_0_moe_down_top10_indexed_batched_expanded";
        self.ensure_kernel(
            FUNC,
            kernels::GEMV_Q8_0_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC,
            FUNC,
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = hidden_batch.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let mv = m as i32;
        let kv = k as i32;
        let nv = batch_size as i32;
        let nev = n_exp as i32;
        let mut params = [
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        let bytes = batch_size * 10 * (m * k / 32 * 34 + k * 4 + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [m as u32, 10, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b.push_i32(nev);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4 qt3/Q8F16 grouped down.  This correctness-first launcher keeps
    /// K=640 as-is rather than forcing it through a 256-wide MQ geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q8_0_moe_grouped_top10(
        &mut self,
        expert_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const FUNC: &str = "gemm_q8_0_moe_grouped_top10";
        self.ensure_kernel(FUNC, kernels::GEMM_Q8_0_MOE_GROUPED_TOP10_SRC, FUNC)?;
        let ep = expert_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_src.buf.as_ptr();
        let yp = y_grouped.buf.as_ptr();
        let mv = m as i32;
        let kv = k as i32;
        let rd = x_row_div as i32;
        let gr = grouped_rows as i32;
        let xr = x_src_rows as i32;
        let ne = n_exp as i32;
        let mut params = [
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rd as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &xr as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
        ];
        let bytes = grouped_rows * m * 4 + grouped_rows * k * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [m as u32, grouped_rows.div_ceil(16) as u32, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(rd);
                b.push_i32(gr);
                b.push_i32(xr);
                b.push_i32(ne);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Qwen4 qt53 indexed down. The kernel emits unweighted expanded rows;
    /// `moe_down_combine_top10_batched` owns route weighting exactly once.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &mut self,
        expert_ptrs: &GpuTensor,
        topk_indices: &GpuTensor,
        hidden_batch: &GpuTensor,
        expert_outputs: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            expert_outputs,
            &[batch_size, 10, m, 4],
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded outputs",
        )?;
        require_span(
            hidden_batch,
            &[batch_size, 10, k, 4],
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded activations",
        )?;
        // K = 640 on exact gfx1151, eager: four weight rows per wave reuse the
        // pick's x (bitwise the one-row kernel).
        let r4 = k == 640
            && self.flags.qwen4_moe_down_r4_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode;
        const MODULE: &str = "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded";
        let (func, rows_per_block) = if r4 {
            (
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4",
                4,
            )
        } else {
            (MODULE, 1)
        };
        self.ensure_kernel(
            MODULE,
            kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC,
            func,
        )?;
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let xp = hidden_batch.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let mv = m as i32;
        let kv = k as i32;
        let nv = batch_size as i32;
        let nev = n_exp as i32;
        let mut params = [
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        let bytes = batch_size.saturating_mul(10).saturating_mul(
            m.saturating_mul(k.div_ceil(128).saturating_mul(68))
                .saturating_add(k.saturating_mul(4))
                .saturating_add(m.saturating_mul(4)),
        );
        let timer = crate::profile::begin_timer(&self.hip, "gemv", func, bytes);
        let result = self.launch_maybe_blob(
            func,
            [m.div_ceil(rows_per_block) as u32, 10, batch_size as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b.push_i32(nev);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Qwen4 qt53 grouped prefill down. K is logical intermediate width and
    /// may be non-multiple-of-128; the F32 parity kernel uses the exact padded
    /// grouped capacity supplied by the sealed preflight.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g128v2_moe_grouped_top10(
        &mut self,
        expert_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        // gfx1151, >= 512 tokens (x_row_div == 1: one X row per top-10 slot):
        // F16 WMMA grouped down, 8.6 -> 5.4 ms at 1131 tokens.  Not
        // bit-exact (F16 dequant/inputs); gated by KLD against the BF16
        // source like the gate/up arm.  HIPFIRE_QWEN4_F16_WMMA=0 opts out.
        if x_row_div == 1 && self.qwen4_moe_down_wmma_applies(m, k, x_src_rows) {
            // Uncached: the activation buffer is reused with new contents per layer.
            let xp = self.convert_fp16_x_uncached(x_src, x_src_rows * k)?;
            return self.gemm_mq4g128v2_moe_grouped_wmma_gfx1151(
                expert_ptrs,
                expert_tile_ids,
                sorted_slot_index,
                xp,
                y_grouped,
                m,
                k,
                grouped_rows,
                false,
            );
        }
        let kernel = if crate::gemm::qwen4_moe_down_twin_applies(
            *crate::gemm::QWEN4_MOE_DOWN_TWIN,
            self.arch_caps.is_gfx1151(),
            m,
            k,
            x_row_div,
            x_src_rows,
        ) {
            MoeDownKernel::XsTwin
        } else if self.arch_caps.has_gfx11_plus_simt() && m == 2560 && k == 640 {
            MoeDownKernel::O8R16
        } else {
            MoeDownKernel::Multirow
        };
        self.gemm_mq4g128v2_moe_grouped_top10_with(
            expert_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_src,
            y_grouped,
            m,
            k,
            x_row_div,
            grouped_rows,
            x_src_rows,
            n_exp,
            kernel,
        )
    }

    /// Whether the F16 WMMA grouped QT53 down applies (gfx11 WMMA, one X row per
    /// top-10 slot, >= QWEN4_F16_WMMA_MIN_TOKENS tokens, not opted out).
    pub fn qwen4_moe_down_wmma_applies(&self, m: usize, k: usize, x_src_rows: usize) -> bool {
        self.arch_caps.has_wmma_w32()
            && m == 2560
            && k == 640
            && self.route_rows(x_src_rows) >= 10 * crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
            && *crate::gemm::QWEN4_F16_WMMA
    }

    /// Whether a `rows`-row grouped Flash-Next MoE prefill takes the F16 WMMA arm end to
    /// end with SwiGLU in the gate/up epilogue and BF16 grouped down rows (gfx11 BF16-out
    /// entries): the gate/up and down predicates together, `mi % 128 == 0`. The EP
    /// owner-routed prefill on gfx1100 runs exactly this route.
    pub fn qwen4_moe_f16_wmma_grouped_applies(
        &self,
        rows: usize,
        mi: usize,
        hidden: usize,
    ) -> bool {
        self.arch_caps.has_wmma_w32()
            && mi % 128 == 0
            && self.qwen4_moe_gateup_wmma_applies(2 * mi, hidden, rows)
            && self.qwen4_moe_down_wmma_applies(hidden, mi, rows.saturating_mul(10))
    }

    /// [`Gpu::gemm_mq4g128v2_moe_grouped_top10`] on the F16 WMMA route with X
    /// already rotated to F16 (`rotate_x_mq_128_v2_f16`): the route's bytes
    /// without its conversion pass; `bf16_out` stores Y as BF16 bits (RNE) for
    /// the BF16-input combine.  Callers check
    /// [`Gpu::qwen4_moe_down_wmma_applies`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g128v2_moe_grouped_top10_xf16(
        &mut self,
        expert_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_f16: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        grouped_rows: usize,
        bf16_out: bool,
    ) -> HipResult<()> {
        self.gemm_mq4g128v2_moe_grouped_wmma_gfx1151(
            expert_ptrs,
            expert_tile_ids,
            sorted_slot_index,
            x_f16.buf.as_ptr(),
            y_grouped,
            m,
            k,
            grouped_rows,
            bf16_out,
        )
    }

    /// F16 WMMA grouped QT53 down (gfx1151, K % 128 == 0) over F16 X
    /// (`x_f16`, one row per top-10 slot): one 16x16 output tile per wave.
    #[allow(clippy::too_many_arguments)]
    fn gemm_mq4g128v2_moe_grouped_wmma_gfx1151(
        &mut self,
        expert_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_f16: *mut c_void,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        grouped_rows: usize,
        bf16_out: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        require_span(
            y_grouped,
            &[grouped_rows, m, if bf16_out { 2 } else { 4 }],
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151 grouped rows",
        )?;
        // gfx1151 (`HIPFIRE_QWEN4_MOE_X_LDS`, default `auto`): eight row
        // tiles per workgroup share X through LDS (bitwise the per-wave
        // kernel); M % 128 == 0 on this route's shapes. This launcher is the
        // F16 WMMA MoE route's down GEMM: the Qwen4 prefill dispatch reaches it
        // only when the layer's symmetric IU4 arm is not selected.
        let x8 = bf16_out
            && crate::gemm::qwen4_moe_x_lds_armed(
                *crate::gemm::QWEN4_MOE_X_LDS,
                self.arch_caps.is_gfx1151(),
                false,
            )
            && m.is_multiple_of(128);
        let func = match (x8, bf16_out) {
            (true, _) => "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out",
            (false, true) => "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out",
            (false, false) => "gemm_mq4g128v2_moe_grouped_wmma_gfx1151",
        };
        const FUNC: &str = "gemm_mq4g128v2_moe_grouped_wmma_gfx1151";
        self.ensure_kernel(
            FUNC,
            kernels::GEMM_MQ4G128V2_MOE_GROUPED_WMMA_GFX1151_SRC,
            func,
        )?;
        let xp = x_f16;
        let ep = expert_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let yp = y_grouped.buf.as_ptr();
        let mv = m as i32;
        let kv = k as i32;
        let rd = 1i32;
        let gr = grouped_rows as i32;
        let ne = crate::gemm::moe_expert_table_len(expert_ptrs);
        let mut params = [
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rd as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
        ];
        let bytes =
            grouped_rows.saturating_mul(m.saturating_mul(4).saturating_add(k.saturating_mul(2)));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func, bytes);
        let result = self.launch_maybe_blob(
            func,
            if x8 {
                [(m / 128) as u32, grouped_rows.div_ceil(16) as u32, 1]
            } else {
                [m.div_ceil(16) as u32, grouped_rows.div_ceil(16) as u32, 1]
            },
            [if x8 { 256 } else { 32 }, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(rd);
                b.push_i32(gr);
                b.push_i32(ne);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_mq4g128v2_moe_grouped_top10_with(
        &mut self,
        expert_ptrs: &GpuTensor,
        expert_tile_ids: &GpuTensor,
        sorted_slot_index: &GpuTensor,
        x_src: &GpuTensor,
        y_grouped: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
        n_exp: usize,
        kernel: MoeDownKernel,
    ) -> HipResult<()> {
        require_span(
            y_grouped,
            &[grouped_rows, m, 4],
            "gemm_mq4g128v2_moe_grouped_top10 grouped rows",
        )?;
        let (func, module, source, grid_x, grid_z, block_x) = match kernel {
            MoeDownKernel::XsTwin => (
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs4_gfx1151",
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_XS_GFX1151_SRC,
                m.div_ceil(32),
                1u32,
                128u32,
            ),
            MoeDownKernel::O8R16 => (
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_GFX1151_SRC,
                m.div_ceil(8),
                1u32,
                32u32,
            ),
            MoeDownKernel::Multirow => (
                "gemm_mq4g128v2_moe_grouped_top10_multirow",
                "gemm_mq4g128v2_moe_grouped_top10_multirow",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_MULTIROW_SRC,
                m,
                1u32,
                32u32,
            ),
        };
        self.ensure_kernel(module, source, func)?;
        let ep = expert_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let xp = x_src.buf.as_ptr();
        let yp = y_grouped.buf.as_ptr();
        let mv = m as i32;
        let kv = k as i32;
        let rd = x_row_div as i32;
        let gr = grouped_rows as i32;
        let xr = x_src_rows as i32;
        let ne = n_exp as i32;
        let mut params = [
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rd as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &xr as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
        ];
        let bytes =
            grouped_rows.saturating_mul(m.saturating_mul(4).saturating_add(k.saturating_mul(4)));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func, bytes);
        let result = self.launch_maybe_blob(
            func,
            [grid_x as u32, grouped_rows.div_ceil(16) as u32, grid_z],
            [block_x, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(rd);
                b.push_i32(gr);
                b.push_i32(xr);
                b.push_i32(ne);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Flash-Next expert-parallel balanced ownership (one 512-thread workgroup;
    /// the certified builder kernel [`QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL`], whose
    /// output bytes are [`qwen4_ep_owner_map`]'s). gfx1201 only: EP runs on
    /// gfx1201, any other device fails closed (there is no hipcc fallback).
    ///
    /// Reads the call's `slots` routed ids from `topk_indices` (i32 bit
    /// patterns, slot order) and rewrites all [`QWEN4_EP_OWNER_EXPERTS`] entries
    /// of the rank's live gate/up and down pointer tables
    /// ([`Qwen4EpOwnerTables`]) plus `slots` entries of `slot_owner`.
    /// Stream-ordered on the active stream; the next launcher reading the live
    /// tables must follow it on the same stream.
    pub fn qwen4_ep_owner_assign(
        &mut self,
        topk_indices: &GpuTensor,
        slots: usize,
        ranks: usize,
        self_rank: usize,
        tables: &Qwen4EpOwnerTables,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let refuse = |what: &str| {
            Err(hip_bridge::HipError::new(
                0,
                &format!("qwen4_ep_owner_assign: {what}"),
            ))
        };
        if !qwen4_ep_owner_assign_pm_arch(&self.arch) {
            return refuse(&format!(
                "the balanced-ownership kernel is a gfx1201 builder module (device is {}); no hipcc fallback",
                self.arch
            ));
        }
        if !(2..=QWEN4_EP_OWNER_ASSIGN_MAX_RANKS).contains(&ranks) || self_rank >= ranks {
            return refuse(&format!(
                "ranks must be 2..={QWEN4_EP_OWNER_ASSIGN_MAX_RANKS} and self_rank < ranks"
            ));
        }
        if slots == 0 || slots > QWEN4_EP_OWNER_EXPERTS {
            return refuse("slots must be 1..=512");
        }
        if topk_indices.buf.size() < slots * std::mem::size_of::<i32>() {
            return refuse("topk_indices holds fewer than `slots` ids");
        }
        if !tables.is_complete() {
            return refuse("a table address is null");
        }
        self.ensure_embedded_kernel(
            QWEN4_EP_OWNER_ASSIGN_PM_MODULE,
            kernels::QWEN4_EP_OWNER_ASSIGN_PM_GFX1201,
            QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL,
        )?;
        let ids = topk_indices.buf.as_ptr();
        let full_gate_up = tables.full_gate_up as *mut c_void;
        let full_down = tables.full_down as *mut c_void;
        let live_gate_up = tables.live_gate_up as *mut c_void;
        let live_down = tables.live_down as *mut c_void;
        let slot_owner = tables.slot_owner as *mut c_void;
        let dummy_gate_up = tables.dummy_gate_up as u64;
        let dummy_down = tables.dummy_down as u64;
        let slots_i32 = slots as i32;
        let ranks_i32 = ranks as i32;
        let self_i32 = self_rank as i32;
        let mut params = [
            &ids as *const _ as *mut c_void,
            &full_gate_up as *const _ as *mut c_void,
            &full_down as *const _ as *mut c_void,
            &live_gate_up as *const _ as *mut c_void,
            &live_down as *const _ as *mut c_void,
            &slot_owner as *const _ as *mut c_void,
            &dummy_gate_up as *const _ as *mut c_void,
            &dummy_down as *const _ as *mut c_void,
            &slots_i32 as *const _ as *mut c_void,
            &ranks_i32 as *const _ as *mut c_void,
            &self_i32 as *const _ as *mut c_void,
        ];
        let bytes = QWEN4_EP_OWNER_EXPERTS * 8 * 4 + slots * 8;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "elementwise",
            QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL,
            [1, 1, 1],
            [QWEN4_EP_OWNER_EXPERTS as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ids);
                b.push_ptr(full_gate_up);
                b.push_ptr(full_down);
                b.push_ptr(live_gate_up);
                b.push_ptr(live_down);
                b.push_ptr(slot_owner);
                b.push_u64(dummy_gate_up);
                b.push_u64(dummy_down);
                b.push_i32(slots_i32);
                b.push_i32(ranks_i32);
                b.push_i32(self_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

/// Experts per layer `qwen4_ep_owner_assign` handles (Flash-Next's 512; one
/// thread each).
pub const QWEN4_EP_OWNER_EXPERTS: usize = 512;

/// Module key of the embedded certified builder (PeaceMaker) image of
/// [`Gpu::qwen4_ep_owner_assign`] (`peacemaker-isa emit --kernel
/// qwen4_ep_owner_assign_pm --arch gfx1201`).
pub const QWEN4_EP_OWNER_ASSIGN_PM_MODULE: &str = "qwen4_ep_owner_assign_pm_gfx1201";
/// Device symbol of the builder kernel.
pub const QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL: &str = "qwen4_ep_owner_assign_pm_gfx1201";

/// The balanced-ownership kernel ships as a gfx1201 builder module only
/// (expert parallelism is a gfx1201 feature); every other device refuses.
pub fn qwen4_ep_owner_assign_pm_arch(arch: &str) -> bool {
    arch == "gfx1201"
}

/// Largest rank count the certified balanced-ownership kernel covers (its
/// contract is `2..=4` ranks); larger EP degrees run the exact static split.
pub const QWEN4_EP_OWNER_ASSIGN_MAX_RANKS: usize = 4;

/// Module key (and device symbol: the builder names both alike) of the embedded certified
/// builder (PeaceMaker) image of [`Gpu::ep_tile_mask`] (`peacemaker-isa emit --kernel
/// ep_tile_mask_pm --arch gfx1201|gfx1100`).
pub const EP_TILE_MASK_PM_GFX1201_SYMBOL: &str = "ep_tile_mask_pm_gfx1201";
pub const EP_TILE_MASK_PM_GFX1100_SYMBOL: &str = "ep_tile_mask_pm_gfx1100";
/// Module key / device symbol of the two-rank specialization of
/// [`Gpu::ep_gather_grouped_rows_by_tile_owner`] (`peacemaker-isa emit --kernel
/// ep_gather_grouped_rows_by_tile_owner_pm --arch gfx1201|gfx1100`; one peer base, 48-byte
/// kernarg).
pub const EP_TILE_PULL_PM_GFX1201_SYMBOL: &str = "ep_gather_grouped_rows_by_tile_owner_pm_gfx1201";
pub const EP_TILE_PULL_PM_GFX1100_SYMBOL: &str = "ep_gather_grouped_rows_by_tile_owner_pm_gfx1100";
/// Module key / device symbol of the rank-generic specialization of
/// [`Gpu::ep_gather_grouped_rows_by_tile_owner`] (`peacemaker-isa emit --kernel
/// ep_gather_grouped_rows_by_tile_owner_n8_pm --arch gfx1201|gfx1100`; eight source bases,
/// 104-byte kernarg).
pub const EP_TILE_PULL_N8_PM_GFX1201_SYMBOL: &str =
    "ep_gather_grouped_rows_by_tile_owner_n8_pm_gfx1201";
pub const EP_TILE_PULL_N8_PM_GFX1100_SYMBOL: &str =
    "ep_gather_grouped_rows_by_tile_owner_n8_pm_gfx1100";
/// Largest rank count of the tile pull (the N8 kernarg's source slots).
pub const EP_TILE_PULL_MAX_RANKS: usize = 8;

/// The tile mask/pull builder kernels exist for exactly gfx1100 and gfx1201. The balanced
/// owner assignment stays gfx1201-only ([`qwen4_ep_owner_assign_pm_arch`]).
pub fn ep_tile_pm_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1201")
}

/// Embedded image and device symbol (also the module key) of the tile mask for `arch`.
fn ep_tile_mask_kernel(arch: &str) -> Option<(&'static [u8], &'static str)> {
    match arch {
        "gfx1201" => Some((
            kernels::EP_TILE_MASK_PM_GFX1201,
            EP_TILE_MASK_PM_GFX1201_SYMBOL,
        )),
        "gfx1100" => Some((
            kernels::EP_TILE_MASK_PM_GFX1100,
            EP_TILE_MASK_PM_GFX1100_SYMBOL,
        )),
        _ => None,
    }
}

/// Embedded image and device symbol (also the module key) of the tile pull for `arch`:
/// the two-rank kernel for `ranks == 2`, the eight-slot rank-generic one for `3..=8`.
fn ep_tile_pull_kernel(arch: &str, ranks: usize) -> Option<(&'static [u8], &'static str)> {
    match (arch, ranks == 2) {
        ("gfx1201", true) => Some((
            kernels::EP_TILE_PULL_PM_GFX1201,
            EP_TILE_PULL_PM_GFX1201_SYMBOL,
        )),
        ("gfx1100", true) => Some((
            kernels::EP_TILE_PULL_PM_GFX1100,
            EP_TILE_PULL_PM_GFX1100_SYMBOL,
        )),
        ("gfx1201", false) => Some((
            kernels::EP_TILE_PULL_N8_PM_GFX1201,
            EP_TILE_PULL_N8_PM_GFX1201_SYMBOL,
        )),
        ("gfx1100", false) => Some((
            kernels::EP_TILE_PULL_N8_PM_GFX1100,
            EP_TILE_PULL_N8_PM_GFX1100_SYMBOL,
        )),
        _ => None,
    }
}

/// Grouped rows of one tile (the symmetric IU4 GEMMs' `m` tile).
pub const EP_TILE_ROWS: usize = 16;
/// Threads (tiles) per workgroup of the mask kernel.
const EP_TILE_MASK_BLOCK: u32 = 256;
/// uint4 moved per workgroup trip of the pull kernel; a tile is a multiple of it.
const EP_TILE_PULL_TRIP_U4: usize = 1024;

impl Gpu {
    /// Owner-routed chunk MoE tile mask (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): write a COPY of
    /// the grouped-GEMM tile ids, `tile_ids_owned[t] = tile_ids[t]` when the tile's expert is
    /// owned by `rank` (`owner_table[tile_ids[t]] == rank`, an i32 per expert) and `-1`
    /// otherwise (the padding sentinel included). `tile_ids` is never written. One thread per
    /// tile over `tiles`, the certified builder kernel ([`EP_TILE_MASK_PM_GFX1201_SYMBOL`] /
    /// [`EP_TILE_MASK_PM_GFX1100_SYMBOL`]); gfx1100 and gfx1201 only
    /// (there is no hipcc fallback). Stream-ordered on the active stream.
    pub fn ep_tile_mask(
        &mut self,
        tile_ids: &GpuTensor,
        tile_ids_owned: &GpuTensor,
        owner_table: &GpuTensor,
        tiles: usize,
        n_experts: usize,
        rank: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let refuse = |what: &str| {
            Err(hip_bridge::HipError::new(
                0,
                &format!("ep_tile_mask: {what}"),
            ))
        };
        let Some((mask_image, mask_symbol)) =
            ep_tile_mask_kernel(&self.arch).filter(|_| ep_tile_pm_arch(&self.arch))
        else {
            return refuse(&format!(
                "the tile mask is a gfx1100/gfx1201 builder module (device is {}); no hipcc fallback",
                self.arch
            ));
        };
        let word = std::mem::size_of::<i32>();
        if tiles == 0
            || i32::try_from(tiles).is_err()
            || n_experts == 0
            || i32::try_from(n_experts).is_err()
        {
            return refuse("need 0 < tiles and 0 < n_experts, both within i32");
        }
        if rank > i32::MAX as usize {
            return refuse("rank exceeds i32");
        }
        if tile_ids.buf.size() < tiles * word || tile_ids_owned.buf.size() < tiles * word {
            return refuse("tile id buffers hold fewer than `tiles` words");
        }
        if owner_table.buf.size() < n_experts * word {
            return refuse("owner table holds fewer than `n_experts` words");
        }
        if tile_ids.buf.as_ptr() == tile_ids_owned.buf.as_ptr() {
            return refuse("the owned copy must not alias the source tile ids");
        }
        self.ensure_embedded_kernel(mask_symbol, mask_image, mask_symbol)?;
        let ids = tile_ids.buf.as_ptr();
        let owned = tile_ids_owned.buf.as_ptr();
        let table = owner_table.buf.as_ptr();
        let (tiles_i32, experts_i32, rank_i32) = (tiles as i32, n_experts as i32, rank as i32);
        let mut params = [
            &ids as *const _ as *mut c_void,
            &owned as *const _ as *mut c_void,
            &table as *const _ as *mut c_void,
            &tiles_i32 as *const _ as *mut c_void,
            &experts_i32 as *const _ as *mut c_void,
            &rank_i32 as *const _ as *mut c_void,
        ];
        let timer =
            crate::profile::begin_timer(&self.hip, "elementwise", mask_symbol, tiles * word * 2);
        let result = self.launch_maybe_blob(
            mask_symbol,
            [(tiles as u32).div_ceil(EP_TILE_MASK_BLOCK), 1, 1],
            [EP_TILE_MASK_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ids);
                b.push_ptr(owned);
                b.push_ptr(table);
                b.push_i32(tiles_i32);
                b.push_i32(experts_i32);
                b.push_i32(rank_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Owner-routed chunk MoE tile pull: for every live tile whose expert another rank owns
    /// (`0 <= tile_ids[t] < n_experts` and `owner_table[tile_ids[t]] != rank`), copy the
    /// tile's [`EP_TILE_ROWS`] grouped rows (`row_u4` uint4 each) of the OWNER's grouped-rows
    /// buffer, the virtual address `sources[owner_table[tile_ids[t]]]` (peer-mapped device
    /// memory), onto the same rows of `dst`. `ranks` (2..=8, host validation only, not a
    /// kernel argument) bounds the slots in use; `sources[rank]` may be `dst`'s own address
    /// (the local slot is never read), every other slot below `ranks` must be a mapped peer
    /// base; slots from `ranks` up are passed as zero. Two certified specializations behind
    /// this one API, selected by `ranks` inside the launcher: `ranks == 2` launches the
    /// two-rank kernel ([`EP_TILE_PULL_PM_GFX1201_SYMBOL`] / [`EP_TILE_PULL_PM_GFX1100_SYMBOL`])
    /// with the one peer base `sources[1 - rank]` (the only other valid owner; 48-byte kernarg,
    /// no owner range check); `ranks` in `3..=8` launches the rank-generic kernel
    /// ([`EP_TILE_PULL_N8_PM_GFX1201_SYMBOL`] / [`EP_TILE_PULL_N8_PM_GFX1100_SYMBOL`], 104-byte
    /// kernarg) with all eight slots. Rows of owned tiles, of the padding
    /// tail and of workgroups past `tiles` are never written. Bit-exact uint4 copy, gfx1100
    /// and gfx1201 only. Stream-ordered on the active stream; the caller orders it after every
    /// peer's producer event.
    #[allow(clippy::too_many_arguments)]
    pub fn ep_gather_grouped_rows_by_tile_owner(
        &mut self,
        tile_ids: &GpuTensor,
        owner_table: &GpuTensor,
        sources: [usize; EP_TILE_PULL_MAX_RANKS],
        dst: &GpuTensor,
        tiles: usize,
        n_experts: usize,
        ranks: usize,
        rank: usize,
        row_u4: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let refuse = |what: &str| {
            Err(hip_bridge::HipError::new(
                0,
                &format!("ep_gather_grouped_rows_by_tile_owner: {what}"),
            ))
        };
        if !ep_tile_pm_arch(&self.arch) {
            return refuse(&format!(
                "the tile pull is a gfx1100/gfx1201 builder module (device is {}); no hipcc fallback",
                self.arch
            ));
        }
        let word = std::mem::size_of::<i32>();
        if !(2..=EP_TILE_PULL_MAX_RANKS).contains(&ranks) || rank >= ranks {
            return refuse("ranks must be 2..=8 and rank below ranks");
        }
        if tiles == 0
            || n_experts == 0
            || sources
                .iter()
                .enumerate()
                .any(|(r, &base)| r < ranks && r != rank && base == 0)
        {
            return refuse("need tiles > 0, n_experts > 0 and a base for every peer rank");
        }
        let tile_u4 = EP_TILE_ROWS.checked_mul(row_u4).unwrap_or(0);
        if tile_u4 == 0 || tile_u4 % EP_TILE_PULL_TRIP_U4 != 0 {
            return refuse("a tile must be a non-zero multiple of 1024 uint4");
        }
        let tile_bytes = tile_u4 * 16;
        let span = tiles
            .checked_mul(tile_bytes)
            .filter(|&span| span <= u32::MAX as usize);
        let Some(span) = span else {
            return refuse("tiles * tile bytes must fit 32 bits");
        };
        if i32::try_from(tiles).is_err()
            || i32::try_from(n_experts).is_err()
            || i32::try_from(tile_u4).is_err()
        {
            return refuse("an extent exceeds i32");
        }
        if tile_ids.buf.size() < tiles * word {
            return refuse("tile_ids holds fewer than `tiles` words");
        }
        if owner_table.buf.size() < n_experts * word {
            return refuse("owner table holds fewer than `n_experts` words");
        }
        if dst.buf.size() < span {
            return refuse("the destination holds fewer than `tiles` tiles");
        }
        for (r, &base) in sources.iter().enumerate().take(ranks) {
            if r != rank && base == dst.buf.as_ptr() as usize {
                return refuse("a peer source is this rank's own buffer");
            }
        }
        let Some((image, symbol)) = ep_tile_pull_kernel(&self.arch, ranks) else {
            return refuse("no tile pull kernel for this device");
        };
        self.ensure_embedded_kernel(symbol, image, symbol)?;
        let ids = tile_ids.buf.as_ptr();
        let table = owner_table.buf.as_ptr();
        // N8: slots past `ranks` are never named by a valid owner table: pass them as null.
        // EP2: the only other valid owner is the peer, `sources[1 - rank]`.
        let srcs: [*mut c_void; EP_TILE_PULL_MAX_RANKS] = std::array::from_fn(|r| {
            if r < ranks {
                sources[r] as *mut c_void
            } else {
                std::ptr::null_mut()
            }
        });
        let peer = srcs[rank ^ 1];
        let out = dst.buf.as_ptr();
        let (experts_i32, rank_i32, tile_u4_i32, tiles_i32) =
            (n_experts as i32, rank as i32, tile_u4 as i32, tiles as i32);
        // EP2 kernarg (48 B): ids, table, peer, dst, n_experts, rank, tile_u4, tiles (params[..8]).
        // N8 kernarg (104 B): ids@0, table@8, src0..src7@16..72, dst@80, n_experts@88, rank@92,
        // tile_u4@96, tiles@100 (params[..15]).
        let mut params = [std::ptr::null_mut::<c_void>(); 15];
        let used = if ranks == 2 {
            params[..8].copy_from_slice(&[
                &ids as *const _ as *mut c_void,
                &table as *const _ as *mut c_void,
                &peer as *const _ as *mut c_void,
                &out as *const _ as *mut c_void,
                &experts_i32 as *const _ as *mut c_void,
                &rank_i32 as *const _ as *mut c_void,
                &tile_u4_i32 as *const _ as *mut c_void,
                &tiles_i32 as *const _ as *mut c_void,
            ]);
            8
        } else {
            params[0] = &ids as *const _ as *mut c_void;
            params[1] = &table as *const _ as *mut c_void;
            for (slot, src) in params[2..10].iter_mut().zip(srcs.iter()) {
                *slot = src as *const _ as *mut c_void;
            }
            params[10] = &out as *const _ as *mut c_void;
            params[11] = &experts_i32 as *const _ as *mut c_void;
            params[12] = &rank_i32 as *const _ as *mut c_void;
            params[13] = &tile_u4_i32 as *const _ as *mut c_void;
            params[14] = &tiles_i32 as *const _ as *mut c_void;
            15
        };
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", symbol, span);
        let result = self.launch_maybe_blob(
            symbol,
            [tiles as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params[..used],
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ids);
                b.push_ptr(table);
                if ranks == 2 {
                    b.push_ptr(peer);
                } else {
                    for src in srcs {
                        b.push_ptr(src);
                    }
                }
                b.push_ptr(out);
                b.push_i32(experts_i32);
                b.push_i32(rank_i32);
                b.push_i32(tile_u4_i32);
                b.push_i32(tiles_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

/// Device addresses [`Gpu::qwen4_ep_owner_assign`] reads and writes for one
/// layer of one rank. `full_*` are the layer's read-only full pointer tables
/// (512 `u64` each), `live_*` the rank's tables the routed producers read
/// (rewritten whole per call), `slot_owner` the i32 owner of each slot, and
/// `dummy_*` this rank's zero dummy expert (gate/up, down).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4EpOwnerTables {
    pub full_gate_up: usize,
    pub full_down: usize,
    pub live_gate_up: usize,
    pub live_down: usize,
    pub slot_owner: usize,
    pub dummy_gate_up: usize,
    pub dummy_down: usize,
}

impl Qwen4EpOwnerTables {
    /// Every address is set (a null one would be dereferenced on the device).
    pub fn is_complete(&self) -> bool {
        [
            self.full_gate_up,
            self.full_down,
            self.live_gate_up,
            self.live_down,
            self.slot_owner,
            self.dummy_gate_up,
            self.dummy_down,
        ]
        .iter()
        .all(|&address| address != 0)
    }
}

/// Host reference of `qwen4_ep_owner_assign`: deal the call's distinct routed
/// experts to `ranks` alternately in ascending expert-id order.
///
/// Returns `(owner per expert, owner per slot)`: `owners[e]` is
/// `Some(idx(e) % ranks)` for a routed expert (`idx` = number of routed
/// experts with a smaller id) and `None` for an expert no slot routes to;
/// `slot_owners[s]` is the owner of `ids[s]`, `-1` for a negative or
/// out-of-range id. Pure integer arithmetic: the same on every rank for the
/// same replicated ids, and per-rank distinct-expert counts differ by at most
/// one.
pub fn qwen4_ep_owner_map(
    ids: &[i32],
    n_exp: usize,
    ranks: usize,
) -> (Vec<Option<usize>>, Vec<i32>) {
    assert!(ranks >= 1, "qwen4_ep_owner_map: ranks must be at least 1");
    let in_range = |id: i32| usize::try_from(id).ok().filter(|&expert| expert < n_exp);
    let mut routed = vec![false; n_exp];
    for &id in ids {
        if let Some(expert) = in_range(id) {
            routed[expert] = true;
        }
    }
    let mut owners = vec![None; n_exp];
    let mut dealt = 0usize;
    for (expert, &is_routed) in routed.iter().enumerate() {
        if is_routed {
            owners[expert] = Some(dealt % ranks);
            dealt += 1;
        }
    }
    let slot_owners = ids
        .iter()
        .map(|&id| {
            in_range(id)
                .and_then(|expert| owners[expert])
                .map_or(-1, |owner| owner as i32)
        })
        .collect();
    (owners, slot_owners)
}

#[cfg(test)]
mod owner_map_tests {
    use super::*;

    const EXPERTS: usize = QWEN4_EP_OWNER_EXPERTS;

    fn distinct_per_rank(owners: &[Option<usize>], ranks: usize) -> Vec<usize> {
        let mut counts = vec![0usize; ranks];
        for owner in owners.iter().flatten() {
            counts[*owner] += 1;
        }
        counts
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    #[test]
    fn decode_ten_slots_alternate_in_ascending_id_order() {
        // Routed ids sorted: 3, 17, 40, 41, 100, 200, 201, 300, 449, 511.
        let ids = [200, 3, 511, 41, 17, 449, 300, 40, 201, 100];
        let (owners, slots) = qwen4_ep_owner_map(&ids, EXPERTS, 2);
        let sorted = [3, 17, 40, 41, 100, 200, 201, 300, 449, 511];
        for (position, expert) in sorted.iter().enumerate() {
            assert_eq!(owners[*expert], Some(position % 2), "expert {expert}");
        }
        assert_eq!(owners.iter().flatten().count(), 10);
        assert_eq!(distinct_per_rank(&owners, 2), vec![5, 5]);
        for (slot, id) in ids.iter().enumerate() {
            assert_eq!(slots[slot], owners[*id as usize].unwrap() as i32);
        }
        // Four ranks: 3, 3, 2, 2.
        let (owners4, _) = qwen4_ep_owner_map(&ids, EXPERTS, 4);
        assert_eq!(distinct_per_rank(&owners4, 4), vec![3, 3, 2, 2]);
        for (position, expert) in sorted.iter().enumerate() {
            assert_eq!(owners4[*expert], Some(position % 4), "expert {expert}");
        }
    }

    #[test]
    fn static_modulo_would_skew_where_the_map_balances() {
        // Ten even experts: `e % 2` puts all ten on rank 0.
        let ids: Vec<i32> = (0..10).map(|j| 2 * j * 7).collect();
        let (owners, _) = qwen4_ep_owner_map(&ids, EXPERTS, 2);
        assert_eq!(distinct_per_rank(&owners, 2), vec![5, 5]);
    }

    #[test]
    fn verify_forty_slots_with_repeats_share_one_owner_per_expert() {
        // Four rows of ten slots; experts repeat across rows.
        let mut ids = Vec::new();
        for row in 0..4i32 {
            for j in 0..10i32 {
                ids.push((j * 37 + row * 11) % 90);
            }
        }
        for ranks in [2usize, 4] {
            let (owners, slots) = qwen4_ep_owner_map(&ids, EXPERTS, ranks);
            assert_eq!(slots.len(), ids.len());
            for (slot, id) in ids.iter().enumerate() {
                assert_eq!(slots[slot], owners[*id as usize].unwrap() as i32);
            }
            let counts = distinct_per_rank(&owners, ranks);
            let (lo, hi) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
            assert!(hi - lo <= 1, "ranks {ranks}: {counts:?}");
            assert_eq!(
                counts.iter().sum::<usize>(),
                owners.iter().flatten().count()
            );
            // Repeats across rows share one owner: fewer distinct experts than slots.
            assert!(
                owners.iter().flatten().count() < ids.len(),
                "the fixture must repeat experts"
            );
        }
    }

    #[test]
    fn negative_and_out_of_range_ids_are_unowned() {
        let ids = [-1, 5, -7, 9, 512, 5, i32::MIN, i32::MAX, 9];
        let (owners, slots) = qwen4_ep_owner_map(&ids, EXPERTS, 2);
        assert_eq!(owners.iter().flatten().count(), 2);
        assert_eq!(owners[5], Some(0));
        assert_eq!(owners[9], Some(1));
        assert_eq!(slots, vec![-1, 0, -1, 1, -1, 0, -1, -1, 1]);
    }

    #[test]
    fn no_routed_expert_means_no_owner() {
        let (owners, slots) = qwen4_ep_owner_map(&[-1, -1], EXPERTS, 4);
        assert!(owners.iter().all(Option::is_none));
        assert_eq!(slots, vec![-1, -1]);
    }

    #[test]
    fn per_rank_counts_differ_by_at_most_one_and_are_deterministic() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for ranks in [2usize, 4] {
            for slots in [10usize, 20, 40, 80] {
                for _ in 0..200 {
                    let ids: Vec<i32> = (0..slots)
                        .map(|_| match lcg(&mut state) % 11 {
                            0 => -1,
                            _ => (lcg(&mut state) % EXPERTS as u64) as i32,
                        })
                        .collect();
                    let first = qwen4_ep_owner_map(&ids, EXPERTS, ranks);
                    assert_eq!(first, qwen4_ep_owner_map(&ids, EXPERTS, ranks));
                    let counts = distinct_per_rank(&first.0, ranks);
                    let (lo, hi) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
                    assert!(hi - lo <= 1, "ranks {ranks} slots {slots}: {counts:?}");
                    for (slot, id) in ids.iter().enumerate() {
                        let expected = usize::try_from(*id)
                            .ok()
                            .and_then(|expert| first.0[expert])
                            .map_or(-1, |owner| owner as i32);
                        assert_eq!(first.1[slot], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn table_addresses_must_all_be_set() {
        let all = Qwen4EpOwnerTables {
            full_gate_up: 0x1000,
            full_down: 0x2000,
            live_gate_up: 0x3000,
            live_down: 0x4000,
            slot_owner: 0x5000,
            dummy_gate_up: 0x6000,
            dummy_down: 0x7000,
        };
        assert!(all.is_complete());
        assert!(!Qwen4EpOwnerTables {
            dummy_down: 0,
            ..all
        }
        .is_complete());
        assert!(!Qwen4EpOwnerTables {
            full_gate_up: 0,
            ..all
        }
        .is_complete());
    }

    #[test]
    fn the_builder_kernel_is_gfx1201_only() {
        assert!(qwen4_ep_owner_assign_pm_arch("gfx1201"));
        for arch in [
            "gfx1200",
            "gfx1100",
            "gfx1101",
            "gfx1150",
            "gfx1151",
            "gfx90a",
            "gfx1201:xnack-",
            "",
        ] {
            assert!(!qwen4_ep_owner_assign_pm_arch(arch), "{arch:?}");
        }
    }

    #[test]
    fn the_embedded_image_exports_the_launcher_symbol() {
        let image = kernels::QWEN4_EP_OWNER_ASSIGN_PM_GFX1201;
        let symbol = QWEN4_EP_OWNER_ASSIGN_PM_SYMBOL.as_bytes();
        assert!(
            image.windows(4).any(|w| w == b"\x7fELF"),
            "no device ELF in the bundle"
        );
        assert!(
            image.windows(symbol.len()).any(|w| w == symbol),
            "symbol missing from the image"
        );
        assert_eq!(
            QWEN4_EP_OWNER_ASSIGN_PM_MODULE,
            "qwen4_ep_owner_assign_pm_gfx1201"
        );
    }

    /// Device check of the production launcher (the embedded certified builder
    /// kernel) against the host reference: ranks 2 and 4, decode-, verify- and
    /// few-row-sized id sets plus awkward slot counts, repeats, negative and
    /// out-of-range ids. Needs a gfx1201 GPU; never run by the CPU-only gate.
    #[test]
    #[ignore = "requires a gfx1201 GPU and working HIP toolchain"]
    fn kernel_matches_the_host_reference() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) if qwen4_ep_owner_assign_pm_arch(&gpu.arch) => gpu,
            _ => {
                eprintln!("skip: the balanced-ownership kernel is gfx1201 only");
                return;
            }
        };
        let to_bytes = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let full_gate_up: Vec<u64> = (0..EXPERTS as u64)
            .map(|e| 0x1_0000_0000 + e * 0x100)
            .collect();
        let full_down: Vec<u64> = (0..EXPERTS as u64)
            .map(|e| 0x2_0000_0000 + e * 0x80)
            .collect();
        let (dummy_gate_up, dummy_down) = (0xD000_0000u64, 0xE000_0000u64);
        let upload =
            |gpu: &mut Gpu, bytes: &[u8]| gpu.upload_raw(bytes, &[bytes.len()]).expect("upload");
        let full_gate_up_gpu = upload(&mut gpu, &to_bytes(&full_gate_up));
        let full_down_gpu = upload(&mut gpu, &to_bytes(&full_down));
        let live_gate_up_gpu = upload(&mut gpu, &vec![0xAAu8; EXPERTS * 8]);
        let live_down_gpu = upload(&mut gpu, &vec![0xAAu8; EXPERTS * 8]);
        let owners_gpu = upload(&mut gpu, &vec![0x55u8; EXPERTS * 4]);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for ranks in [2usize, 4] {
            for slots in [10usize, 40, 80, 1, 7, 33, 200, 512] {
                for _ in 0..16 {
                    let ids: Vec<i32> = (0..slots)
                        .map(|_| match lcg(&mut state) % 11 {
                            0 => -1,
                            1 => {
                                [512, 513, i32::MIN, i32::MAX, -513][(lcg(&mut state) % 5) as usize]
                            }
                            _ => (lcg(&mut state) % EXPERTS as u64) as i32,
                        })
                        .collect();
                    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
                    let ids_gpu = upload(&mut gpu, &id_bytes);
                    let (owners, slot_owners) = qwen4_ep_owner_map(&ids, EXPERTS, ranks);
                    for rank in 0..ranks {
                        let tables = Qwen4EpOwnerTables {
                            full_gate_up: full_gate_up_gpu.buf.as_ptr() as usize,
                            full_down: full_down_gpu.buf.as_ptr() as usize,
                            live_gate_up: live_gate_up_gpu.buf.as_ptr() as usize,
                            live_down: live_down_gpu.buf.as_ptr() as usize,
                            slot_owner: owners_gpu.buf.as_ptr() as usize,
                            dummy_gate_up: dummy_gate_up as usize,
                            dummy_down: dummy_down as usize,
                        };
                        // The kernel reads `full_*` through device pointers.
                        gpu.qwen4_ep_owner_assign(&ids_gpu, slots, ranks, rank, &tables)
                            .expect("assign launch");
                        gpu.hip.device_synchronize().expect("sync");
                        let mut raw = vec![0u8; EXPERTS * 8];
                        gpu.hip
                            .memcpy_dtoh(&mut raw, &live_gate_up_gpu.buf)
                            .expect("gate/up");
                        let live_gu: Vec<u64> = raw
                            .chunks_exact(8)
                            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        gpu.hip
                            .memcpy_dtoh(&mut raw, &live_down_gpu.buf)
                            .expect("down");
                        let live_dn: Vec<u64> = raw
                            .chunks_exact(8)
                            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        let mut owner_raw = vec![0u8; slots * 4];
                        gpu.hip
                            .memcpy_dtoh(&mut owner_raw, &owners_gpu.buf)
                            .expect("slot owners");
                        let device_slots: Vec<i32> = owner_raw
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        assert_eq!(device_slots, slot_owners, "ranks {ranks} rank {rank}");
                        for expert in 0..EXPERTS {
                            let mine = owners[expert] == Some(rank);
                            assert_eq!(
                                live_gu[expert],
                                if mine {
                                    full_gate_up[expert]
                                } else {
                                    dummy_gate_up
                                }
                            );
                            assert_eq!(
                                live_dn[expert],
                                if mine { full_down[expert] } else { dummy_down }
                            );
                        }
                    }
                    gpu.free_tensor(ids_gpu).expect("free ids");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gfx1151 O8×R16 grouped QT53 down kernel must reproduce the generic
    /// multirow kernel bit for bit: two experts, dead (-1) slots, an all-dead
    /// tile, a sentinel expert tile and a partial final tile.
    #[test]
    #[ignore = "requires a gfx11/gfx12 GPU and working HIP toolchain"]
    fn down_o8_r16_is_bit_identical_to_multirow() {
        const M: usize = 2560;
        const K: usize = 640;
        const GROUPED: usize = 330;
        const SLOTS: usize = 400;
        let mut gpu = match Gpu::init() {
            Ok(gpu) if gpu.arch_caps.has_gfx11_plus_simt() => gpu,
            _ => {
                eprintln!("skip: needs a gfx11/gfx12 GPU");
                return;
            }
        };
        let row_bytes = K / 128 * 68;
        let expert = |seed: u32| -> Vec<u8> {
            let mut bytes = vec![0u8; M * row_bytes];
            let mut state = seed;
            for chunk in bytes.chunks_mut(68) {
                for byte in chunk.iter_mut() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *byte = (state >> 24) as u8;
                }
                let jitter = (chunk[10] & 0x3f) as u16;
                chunk[0..2].copy_from_slice(&(0x2c00u16 + jitter).to_le_bytes());
                chunk[2..4].copy_from_slice(&(0xb000u16 + jitter).to_le_bytes());
            }
            bytes
        };
        let experts: Vec<GpuTensor> = [expert(3), expert(4)]
            .iter()
            .map(|bytes| {
                gpu.upload_raw(bytes, &[bytes.len()])
                    .expect("expert upload")
            })
            .collect();
        let ptrs: Vec<u8> = experts
            .iter()
            .flat_map(|tensor| (tensor.buf.as_ptr() as u64).to_le_bytes())
            .collect();
        let ptrs_gpu = gpu.upload_raw(&ptrs, &[ptrs.len()]).expect("ptr upload");
        let tiles: Vec<i32> = (0..GROUPED.div_ceil(16))
            .map(|t| if t % 5 == 2 { -1 } else { (t % 2) as i32 })
            .collect();
        let slots: Vec<i32> = (0..GROUPED)
            .map(|s| match s % 23 {
                3 | 17 => -1,
                _ if (48..64).contains(&s) => -1,
                _ => ((s * 37) % SLOTS) as i32,
            })
            .collect();
        let to_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let tiles_gpu = gpu
            .upload_raw(&to_bytes(&tiles), &[tiles.len() * 4])
            .expect("tiles");
        let slots_gpu = gpu
            .upload_raw(&to_bytes(&slots), &[slots.len() * 4])
            .expect("slots");
        let x: Vec<f32> = (0..SLOTS * K)
            .map(|i| ((i * 7919 % 4001) as f32 - 2000.0) / 1777.0)
            .collect();
        let x_gpu = gpu.upload_f32(&x, &[x.len()]).expect("x upload");
        let on_gfx1151 = gpu.arch_caps.is_gfx1151();
        let mut run = |kernel: MoeDownKernel| {
            let sentinel = vec![f32::from_bits(0x7fc0_1234); GROUPED * M];
            let y = gpu
                .upload_f32(&sentinel, &[sentinel.len()])
                .expect("y upload");
            gpu.gemm_mq4g128v2_moe_grouped_top10_with(
                &ptrs_gpu, &tiles_gpu, &slots_gpu, &x_gpu, &y, M, K, 1, GROUPED, SLOTS, 2, kernel,
            )
            .expect("down launch");
            let out = gpu.download_f32(&y).expect("y download");
            gpu.free_tensor(y).expect("free y");
            out
        };
        let reference = run(MoeDownKernel::Multirow);
        let candidate = run(MoeDownKernel::O8R16);
        assert!(reference.iter().any(|v| v.is_finite() && *v != 0.0));
        let differing = reference
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "O8xR16 down differs from multirow in {differing} cells"
        );
        if on_gfx1151 {
            let twin = run(MoeDownKernel::XsTwin);
            let twin_differing = candidate
                .iter()
                .zip(&twin)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                twin_differing, 0,
                "X-in-LDS twin down differs from O8xR16 in {twin_differing} cells"
            );
        }
    }

    /// The fused grouped gate/up unscatter + SwiGLU must equal the unfused
    /// unscatter -> BF16 round trip (gate, up) -> silu_mul -> BF16 round trip
    /// sequence bit for bit, with padding slots interleaved in the groups.
    #[test]
    fn unscatter_silu_matches_unfused_sequence() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        const MI: usize = 640;
        const TOKENS: usize = 13;
        let slots = TOKENS * 10;
        // Every flat slot appears once, spread over groups with -1 padding.
        let mut sorted = Vec::new();
        for flat in 0..slots {
            sorted.push(((flat * 37) % slots) as i32);
            if flat % 7 == 3 {
                sorted.push(-1);
            }
        }
        let grouped = sorted.len();
        let y: Vec<f32> = (0..grouped * 2 * MI)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 20011) as f32 - 10005.0) / 1777.0)
            .collect();
        let y_gpu = gpu.upload_f32(&y, &[y.len()]).expect("y");
        let bytes: Vec<u8> = sorted.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sorted_gpu = gpu.upload_raw(&bytes, &[bytes.len()]).expect("sorted");
        let n = slots * MI;
        let gate = gpu.zeros(&[n], DType::F32).expect("gate");
        let up = gpu.zeros(&[n], DType::F32).expect("up");
        let reference = gpu.zeros(&[n], DType::F32).expect("reference");
        gpu.moe_gate_up_unscatter_top10(&y_gpu, &sorted_gpu, &gate, &up, MI, grouped, TOKENS)
            .expect("unscatter");
        gpu.bf16_round_trip_f32(&gate).expect("rt gate");
        gpu.bf16_round_trip_f32(&up).expect("rt up");
        gpu.silu_mul_f32(&gate, &up, &reference).expect("silu");
        gpu.bf16_round_trip_f32(&reference).expect("rt act");
        let fused = gpu.zeros(&[n], DType::F32).expect("fused");
        gpu.moe_gate_up_unscatter_silu_top10(&y_gpu, &sorted_gpu, &fused, MI, grouped, true)
            .expect("fused");
        let a = gpu.download_f32(&reference).expect("download");
        let b = gpu.download_f32(&fused).expect("download");
        assert!(a.iter().any(|v| *v != 0.0));
        let differing = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "fused unscatter+SwiGLU differs in {differing} cells"
        );
    }

    /// The two-matrix K = 2560 Q8 row launch must write each matrix's rows
    /// bitwise as its own `gemv_q8_0_staged_rows` launch does.
    #[test]
    fn q8_k2560_staged_rows_x2_is_bitwise_two_launches() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        const K: usize = 2560;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut matrix = |m: usize| {
            let mut bytes = vec![0u8; m * K / 32 * 34];
            for block in bytes.chunks_exact_mut(34) {
                let scale = 0x1C00u16 + (next() % 0x0800) as u16;
                block[..2].copy_from_slice(&scale.to_le_bytes());
                for b in &mut block[2..] {
                    *b = next() as u8;
                }
            }
            gpu.upload_raw(&bytes, &[bytes.len()]).expect("q8 matrix")
        };
        let (m0, m1) = (640usize, 640usize);
        let (a0, a1) = (matrix(m0), matrix(m1));
        if !gpu.gemv_q8_0_staged_rows_supported(K, 3) {
            eprintln!("skip: staged rows unsupported on this arch");
            return;
        }
        for rows in 2..=8usize {
            let x: Vec<f32> = (0..rows * K)
                .map(|i| ((i * 2_654_435_761) % 8191) as f32 / 4096.0 - 1.0)
                .collect();
            let x = gpu.upload_f32(&x, &[rows * K]).expect("x");
            let outs: Vec<GpuTensor> = (0..4)
                .map(|i| {
                    gpu.zeros(&[rows * if i % 2 == 0 { m0 } else { m1 }], DType::F32)
                        .expect("y")
                })
                .collect();
            gpu.gemv_q8_0_staged_rows(&a0, &x, &outs[0], m0, K, rows)
                .expect("single 0");
            gpu.gemv_q8_0_staged_rows(&a1, &x, &outs[1], m1, K, rows)
                .expect("single 1");
            gpu.gemv_q8_0_k2560_staged_rows_x2(&a0, &outs[2], m0, &a1, &outs[3], m1, &x, rows)
                .expect("x2");
            for (single, paired) in [(&outs[0], &outs[2]), (&outs[1], &outs[3])] {
                let a = gpu.download_f32(single).expect("download");
                let b = gpu.download_f32(paired).expect("download");
                assert!(a.iter().any(|v| *v != 0.0));
                let differing = a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                assert_eq!(differing, 0, "x2 differs in {differing} cells, rows={rows}");
            }
        }
    }

    /// H8a: the BF16-row combine started from +0.0 (`initial_zero`) must leave
    /// a dirty target bytewise as the zero-filled target the unchanged symbol
    /// accumulates into, with dead (-1) and duplicate routes and ragged tokens.
    #[test]
    fn combine_bf16in_zero_init_is_bytewise_the_zero_filled_accumulate() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        const HIDDEN: usize = 2560;
        for tokens in [1usize, 37, 530] {
            let grouped_rows = tokens * 10 + 16;
            let wave = |seed: usize, n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                        (h as f32 - 4095.0) / 1365.0
                    })
                    .collect()
            };
            let grouped: Vec<u8> = wave(1, grouped_rows * HIDDEN)
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let inverse: Vec<i32> = (0..tokens * 10)
                .map(|i| {
                    if i % 13 == 5 {
                        -1
                    } else {
                        ((i * 7919) % grouped_rows) as i32
                    }
                })
                .collect();
            let experts: Vec<i32> = (0..tokens * 10).map(|i| ((i * 37) % 64) as i32).collect();
            let weights: Vec<f32> = wave(2, tokens * 10).iter().map(|v| v.abs() / 3.0).collect();
            let to_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let grouped = gpu.upload_raw(&grouped, &[grouped.len()]).expect("grouped");
            let inverse = gpu
                .upload_raw(&to_bytes(&inverse), &[tokens * 40])
                .expect("inverse");
            let experts = gpu
                .upload_raw(&to_bytes(&experts), &[tokens * 40])
                .expect("experts");
            let weights = gpu.upload_f32(&weights, &[tokens * 10]).expect("weights");
            let order = gpu
                .upload_raw(&vec![0u8; tokens * 10 * 8], &[tokens * 80])
                .expect("order");
            let zero_filled = gpu.zeros(&[tokens * HIDDEN], DType::F32).expect("zeros");
            let dirty = gpu
                .upload_f32(&wave(3, tokens * HIDDEN), &[tokens * HIDDEN])
                .expect("dirty");
            for (target, initial_zero) in [(&zero_filled, false), (&dirty, true)] {
                gpu.moe_down_combine_grouped_top10_bf16in(
                    &grouped,
                    &inverse,
                    &experts,
                    &weights,
                    target,
                    &order,
                    HIDDEN,
                    grouped_rows,
                    tokens,
                    initial_zero,
                    64,
                )
                .expect("combine");
            }
            let a = gpu.download_f32(&zero_filled).expect("download");
            let b = gpu.download_f32(&dirty).expect("download");
            assert!(a.iter().any(|v| *v != 0.0));
            let differing = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "zero-init combine differs in {differing} cells, tokens={tokens}"
            );
        }
    }

    /// Runs `moe_unscatter_silu_clamp_k8` over `tokens` top-8 tokens with
    /// interleaved -1 padding slots and checks (a) every output cell against
    /// an independent CPU SwiGLU-clamp evaluation and (b) byte identity with
    /// the same kernel run over three smaller slot chunks. Returns the
    /// grouped slot count (the launch's m_total).
    fn silu_clamp_k8_case(gpu: &mut Gpu, tokens: usize) -> usize {
        const MI: usize = 8;
        const K_TOP: usize = 8;
        const LIMIT: f32 = 7.0;
        let slots = tokens * K_TOP;
        let mut sorted = Vec::new();
        for i in 0..slots {
            // 37 is coprime with 8 * tokens for the sizes used: a permutation.
            sorted.push(((i * 37) % slots) as i32);
            if i % 97 == 96 {
                sorted.push(-1);
            }
        }
        let grouped = sorted.len();
        let y: Vec<f32> = (0..grouped * 2 * MI)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 20011) as f32 - 10005.0) / 1000.0)
            .collect();
        let to_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();

        let y_gpu = gpu.upload_f32(&y, &[y.len()]).expect("y");
        let sorted_bytes = to_bytes(&sorted);
        let sorted_gpu = gpu
            .upload_raw(&sorted_bytes, &[sorted_bytes.len()])
            .expect("sorted");
        let big = gpu.zeros(&[slots * MI], DType::F32).expect("big");
        gpu.moe_unscatter_silu_clamp_k8(&y_gpu, &sorted_gpu, &big, MI, K_TOP, grouped, LIMIT)
            .expect("launch");
        let big_host = gpu.download_f32(&big).expect("download big");

        let mut expected = vec![0.0f32; slots * MI];
        for (slot, &flat) in sorted.iter().enumerate() {
            if flat < 0 {
                continue;
            }
            for c in 0..MI {
                let g = y[slot * 2 * MI + c].min(LIMIT);
                let u = y[slot * 2 * MI + MI + c].clamp(-LIMIT, LIMIT);
                expected[flat as usize * MI + c] = g / (1.0 + (-g).exp()) * u;
            }
        }
        for (i, (got, want)) in big_host.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "cell {i} (tokens={tokens}): got {got}, want {want}"
            );
        }

        let chunked = gpu.zeros(&[slots * MI], DType::F32).expect("chunked");
        let step = grouped / 3 + 1;
        let mut start = 0usize;
        while start < grouped {
            let end = (start + step).min(grouped);
            let y_chunk = gpu
                .upload_f32(&y[start * 2 * MI..end * 2 * MI], &[(end - start) * 2 * MI])
                .expect("y chunk");
            let s_bytes = to_bytes(&sorted[start..end]);
            let s_chunk = gpu
                .upload_raw(&s_bytes, &[s_bytes.len()])
                .expect("sorted chunk");
            gpu.moe_unscatter_silu_clamp_k8(
                &y_chunk,
                &s_chunk,
                &chunked,
                MI,
                K_TOP,
                end - start,
                LIMIT,
            )
            .expect("chunk launch");
            start = end;
        }
        let chunk_host = gpu.download_f32(&chunked).expect("download chunked");
        let differing = big_host
            .iter()
            .zip(&chunk_host)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "tokens={tokens}: launch differs from chunked in {differing} cells"
        );
        grouped
    }

    /// Small slot count and an 8200-token top-8 chunk (65600 slots + padding
    /// > 65535, the old grid.y limit) both match the CPU reference and the
    /// chunked run.
    #[test]
    fn unscatter_silu_clamp_k8_small_and_past_65535_slots() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        assert!(silu_clamp_k8_case(&mut gpu, 100) < 65536);
        assert!(silu_clamp_k8_case(&mut gpu, 8200) > 65535);
    }

    /// Exactly representable inputs (small integers times {0.25..2}), so the
    /// GPU result must equal the sequential CPU fold bit for bit whether or
    /// not the compiler contracts multiply-add.
    fn combine_k8_case(gpu: &mut Gpu, n: usize, grouped: bool) {
        const M: usize = 8;
        const K: usize = 8;
        let slots = n * K;
        let weights: Vec<f32> = (0..slots)
            .map(|i| [0.5f32, 0.25, 1.0, 2.0][(i / K + i % K) % 4])
            .collect();
        let x0: Vec<f32> = (0..n * M).map(|i| ((i / M) % 13) as f32).collect();
        // Grouped: inverse_perm is a permutation (37 coprime with 8 * n here).
        let inv: Vec<i32> = (0..slots).map(|f| ((f * 37) % slots) as i32).collect();
        let src: Vec<f32> = if grouped {
            (0..slots * M)
                .map(|i| (((i / M) * 29 + (i % M) * 5) % 53) as f32 - 26.0)
                .collect()
        } else {
            (0..slots * M)
                .map(|i| ((i / M / K * 131 + (i / M % K) * 17 + (i % M) * 7) % 61) as f32 - 30.0)
                .collect()
        };
        let mut expected = x0.clone();
        for t in 0..n {
            for c in 0..M {
                let mut acc = 0.0f32;
                for k in 0..K {
                    let v = if grouped {
                        src[inv[t * K + k] as usize * M + c]
                    } else {
                        src[(t * K + k) * M + c]
                    };
                    acc += weights[t * K + k] * v;
                }
                expected[t * M + c] += acc;
            }
        }
        let src_gpu = gpu.upload_f32(&src, &[src.len()]).expect("src");
        let w_gpu = gpu.upload_f32(&weights, &[weights.len()]).expect("w");
        let x_gpu = gpu.upload_f32(&x0, &[x0.len()]).expect("x");
        if grouped {
            let inv_bytes: Vec<u8> = inv.iter().flat_map(|v| v.to_le_bytes()).collect();
            let inv_gpu = gpu.upload_raw(&inv_bytes, &[inv_bytes.len()]).expect("inv");
            gpu.moe_down_combine_grouped_k8(&src_gpu, &inv_gpu, &w_gpu, &x_gpu, M, K, n)
                .expect("grouped combine");
        } else {
            gpu.moe_down_combine_k8_batched(&src_gpu, &w_gpu, &x_gpu, M, K, n)
                .expect("batched combine");
        }
        let got = gpu.download_f32(&x_gpu).expect("download");
        let differing = got
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "n={n} grouped={grouped}: {differing} cells differ from CPU fold"
        );
    }

    /// Token axis on grid.x: 100 tokens (small) and 65600 tokens (> 65535,
    /// the old grid.y limit) for both k8 combine launchers.
    #[test]
    fn moe_down_combine_k8_small_and_past_65535_tokens() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) => gpu,
            Err(_) => {
                eprintln!("skip: no GPU");
                return;
            }
        };
        for grouped in [false, true] {
            combine_k8_case(&mut gpu, 100, grouped);
            combine_k8_case(&mut gpu, 65_600, grouped);
        }
    }

    /// `HIPFIRE_QWEN4_ROUTER_FAST`: the wave-per-token router must store the
    /// incumbent's expert ids and weights byte for byte on every arch that
    /// admits it, including tied logits and a partial final workgroup, and both
    /// must fail a token with -inf / +inf / NaN logits closed (ids -1, weights 0)
    /// and record its (layer, row) in the router fault word.
    #[test]
    #[ignore = "requires a gfx1151/gfx1201 GPU and working HIP toolchain"]
    fn router_fast_is_bit_identical_to_incumbent() {
        const TOKENS: usize = 37;
        let mut gpu = match Gpu::init() {
            Ok(gpu) if matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") => gpu,
            _ => {
                eprintln!("skip: needs a gfx1151/gfx1201 GPU");
                return;
            }
        };
        let mut logits: Vec<f32> = (0..TOKENS * 512)
            .map(|i| ((i * 7919 % 4093) as f32 - 2046.0) / 311.0)
            .collect();
        for t in 0..TOKENS {
            let row = &mut logits[t * 512..(t + 1) * 512];
            match t % 6 {
                // Ten-way and wider ties at the top and across the cut.
                1 => row.iter_mut().step_by(37).for_each(|v| *v = 9.5),
                2 => row[100..140].iter_mut().for_each(|v| *v = -7.25),
                3 => row
                    .iter_mut()
                    .step_by(5)
                    .for_each(|v| *v = f32::NEG_INFINITY),
                4 => row[17] = f32::INFINITY,
                5 => row[300] = f32::NAN,
                _ => {}
            }
        }
        let logits_gpu = gpu.upload_f32(&logits, &[logits.len()]).expect("logits");
        const LAYER: usize = 11;
        const ROW_BASE: usize = 100;
        let faulting = |t: usize| matches!(t % 6, 3 | 4 | 5);
        for normalize in [false, true] {
            for round in [false, true] {
                let mut run = |fast: bool| {
                    std::sync::Arc::make_mut(&mut gpu.flags).qwen4_router_fast = fast;
                    gpu.clear_router_fault();
                    let ids = gpu
                        .upload_f32(
                            &vec![f32::from_bits(0x7fc0_4321); TOKENS * 10],
                            &[TOKENS * 10],
                        )
                        .expect("ids");
                    let weights = gpu
                        .upload_f32(
                            &vec![f32::from_bits(0x7fc0_1234); TOKENS * 10],
                            &[TOKENS * 10],
                        )
                        .expect("weights");
                    gpu.moe_router_softmax_top10_f32(
                        &logits_gpu,
                        &ids,
                        &weights,
                        TOKENS,
                        normalize,
                        round,
                        LAYER,
                        ROW_BASE,
                    )
                    .expect("router launch");
                    gpu.hip.device_synchronize().expect("sync");
                    // The non-finite rows are recorded (layer, global row); the check
                    // reports one of them and clears the word.
                    let fault = gpu
                        .take_router_fault()
                        .expect("non-finite rows are recorded");
                    assert_eq!(fault.layer as usize, LAYER);
                    assert!(
                        faulting(fault.row as usize - ROW_BASE),
                        "row {} is finite",
                        fault.row
                    );
                    assert!(gpu.take_router_fault().is_none(), "a take clears the word");
                    (
                        gpu.download_raw_bytes(&ids).expect("ids"),
                        gpu.download_raw_bytes(&weights).expect("weights"),
                    )
                };
                let incumbent = run(false);
                let fast = run(true);
                assert_eq!(
                    fast.0, incumbent.0,
                    "expert ids differ (normalize={normalize}, round={round})"
                );
                assert_eq!(
                    fast.1, incumbent.1,
                    "weights differ (normalize={normalize}, round={round})"
                );
                // Non-finite rows fail closed whole (ids -1, weights +0); finite rows
                // hold ten valid ids and positive weights.
                for t in 0..TOKENS {
                    let ids = &incumbent.0[t * 40..(t + 1) * 40];
                    let weights = &incumbent.1[t * 40..(t + 1) * 40];
                    if faulting(t) {
                        assert!(ids.iter().all(|&b| b == 0xff), "row {t} ids are not -1");
                        assert!(weights.iter().all(|&b| b == 0), "row {t} weights are not 0");
                    } else {
                        for slot in ids.chunks_exact(4) {
                            let id = i32::from_ne_bytes([slot[0], slot[1], slot[2], slot[3]]);
                            assert!((0..512).contains(&id), "finite row {t} id {id}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod span_guard_tests {
    use super::*;

    fn tensor(bytes: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(0x1000 as *mut c_void, bytes) },
            shape: vec![bytes / 4],
            dtype: DType::F32,
        }
    }

    #[test]
    fn exact_span_passes_and_one_word_short_is_refused() {
        let (rows, width, bf16) = (89_600usize, 640usize, 2usize);
        let need = rows * width * bf16;
        require_span(&tensor(need), &[rows, width, bf16], "grouped").unwrap();
        let err = require_span(&tensor(need - 4), &[rows, width, bf16], "grouped").unwrap_err();
        let message = format!("{err:?}");
        assert!(
            message.contains("grouped") && message.contains(&need.to_string()),
            "{message}"
        );
    }

    #[test]
    fn overflowing_span_is_refused() {
        assert!(require_span(&tensor(64), &[usize::MAX, 2], "huge").is_err());
    }
}

#[cfg(test)]
mod ep_tile_tests {
    use super::*;

    #[test]
    fn the_embedded_images_export_their_launcher_symbols() {
        for (image, symbol) in [
            (
                kernels::EP_TILE_MASK_PM_GFX1201,
                EP_TILE_MASK_PM_GFX1201_SYMBOL,
            ),
            (
                kernels::EP_TILE_MASK_PM_GFX1100,
                EP_TILE_MASK_PM_GFX1100_SYMBOL,
            ),
            (
                kernels::EP_TILE_PULL_PM_GFX1201,
                EP_TILE_PULL_PM_GFX1201_SYMBOL,
            ),
            (
                kernels::EP_TILE_PULL_PM_GFX1100,
                EP_TILE_PULL_PM_GFX1100_SYMBOL,
            ),
            (
                kernels::EP_TILE_PULL_N8_PM_GFX1201,
                EP_TILE_PULL_N8_PM_GFX1201_SYMBOL,
            ),
            (
                kernels::EP_TILE_PULL_N8_PM_GFX1100,
                EP_TILE_PULL_N8_PM_GFX1100_SYMBOL,
            ),
        ] {
            assert!(
                image.windows(4).any(|w| w == b"\x7fELF"),
                "no device ELF in the {symbol} bundle"
            );
            assert!(
                image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                "{symbol} missing from its image"
            );
        }
    }

    #[test]
    fn arch_gate_and_kernel_selection() {
        assert!(ep_tile_pm_arch("gfx1100") && ep_tile_pm_arch("gfx1201"));
        for arch in ["gfx1200", "gfx1151", "gfx1101", "gfx942", "gfx11", ""] {
            assert!(!ep_tile_pm_arch(arch), "{arch:?}");
            assert!(
                ep_tile_mask_kernel(arch).is_none() && ep_tile_pull_kernel(arch, 2).is_none(),
                "{arch:?}"
            );
        }
        // The owner assignment stays gfx1201-only.
        assert!(
            qwen4_ep_owner_assign_pm_arch("gfx1201") && !qwen4_ep_owner_assign_pm_arch("gfx1100")
        );
        for (arch, mask, two, n8) in [
            (
                "gfx1201",
                EP_TILE_MASK_PM_GFX1201_SYMBOL,
                EP_TILE_PULL_PM_GFX1201_SYMBOL,
                EP_TILE_PULL_N8_PM_GFX1201_SYMBOL,
            ),
            (
                "gfx1100",
                EP_TILE_MASK_PM_GFX1100_SYMBOL,
                EP_TILE_PULL_PM_GFX1100_SYMBOL,
                EP_TILE_PULL_N8_PM_GFX1100_SYMBOL,
            ),
        ] {
            assert_eq!(ep_tile_mask_kernel(arch).unwrap().1, mask);
            assert_eq!(ep_tile_pull_kernel(arch, 2).unwrap().1, two);
            for ranks in 3..=EP_TILE_PULL_MAX_RANKS {
                assert_eq!(
                    ep_tile_pull_kernel(arch, ranks).unwrap().1,
                    n8,
                    "{arch} ranks {ranks}"
                );
            }
            assert!(two.ends_with(arch) && n8.ends_with(arch) && mask.ends_with(arch));
            assert_eq!(n8, two.replace("_pm_", "_n8_pm_"));
        }
    }

    /// Device check of the production launchers (the embedded certified builder kernels): the
    /// mask equals the host law, the source ids are untouched, and the pull copies exactly the
    /// non-owned live tiles of a second device buffer (standing in for the peer's) onto the
    /// destination. Needs a gfx1100 or gfx1201 GPU; never run by the CPU-only gate.
    #[test]
    #[ignore = "requires a gfx1100/gfx1201 GPU and working HIP toolchain"]
    fn mask_and_pull_match_the_host_reference() {
        let mut gpu = match Gpu::init() {
            Ok(gpu) if ep_tile_pm_arch(&gpu.arch) => gpu,
            _ => {
                eprintln!("skip: the tile kernels are gfx1100/gfx1201 only");
                return;
            }
        };
        const EXPERTS: usize = 512;
        const ROW_U4: usize = 320; // 2560 BF16
        let tile_bytes = EP_TILE_ROWS * ROW_U4 * 16;
        let upload =
            |gpu: &mut Gpu, bytes: &[u8]| gpu.upload_raw(bytes, &[bytes.len()]).expect("upload");
        let words = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        // 2 ranks launch the two-rank specialization, 3..=8 the rank-generic one; every non-local
        // slot is the one stand-in peer buffer.
        for ranks in [2usize, 3, 4, 5, 6, 8] {
            let owner: Vec<i32> = (0..EXPERTS as i32).map(|e| e % ranks as i32).collect();
            let owner_gpu = upload(&mut gpu, &words(&owner));
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let mut next = |bound: usize| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as usize) % bound
            };
            for tiles in [1usize, 3, 255, 256, 257, 700] {
                let live = tiles - next(tiles / 3 + 1);
                let mut ids: Vec<i32> = (0..live).map(|_| next(EXPERTS) as i32).collect();
                ids.resize(tiles, -1);
                let peer: Vec<u8> = (0..tiles * tile_bytes)
                    .map(|i| (next(251) as u8) ^ (i as u8))
                    .collect();
                let ids_gpu = upload(&mut gpu, &words(&ids));
                let peer_gpu = upload(&mut gpu, &peer);
                for rank in 0..ranks {
                    let owned_gpu = upload(&mut gpu, &vec![0x5Au8; tiles * 4]);
                    let dst_gpu = upload(&mut gpu, &vec![0xC3u8; tiles * tile_bytes]);
                    gpu.ep_tile_mask(&ids_gpu, &owned_gpu, &owner_gpu, tiles, EXPERTS, rank)
                        .expect("mask");
                    // The local slot is this rank's destination, every other used slot the stand-in peer.
                    let mut sources = [0usize; EP_TILE_PULL_MAX_RANKS];
                    for slot in sources.iter_mut().take(ranks) {
                        *slot = peer_gpu.buf.as_ptr() as usize;
                    }
                    sources[rank] = dst_gpu.buf.as_ptr() as usize;
                    gpu.ep_gather_grouped_rows_by_tile_owner(
                        &ids_gpu, &owner_gpu, sources, &dst_gpu, tiles, EXPERTS, ranks, rank,
                        ROW_U4,
                    )
                    .expect("pull");
                    gpu.hip.device_synchronize().expect("sync");
                    let mut raw = vec![0u8; tiles * 4];
                    gpu.hip
                        .memcpy_dtoh(&mut raw, &owned_gpu.buf)
                        .expect("owned");
                    let device_owned: Vec<i32> = raw
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let want_owned: Vec<i32> = ids
                        .iter()
                        .map(|&id| {
                            if id >= 0 && owner[id as usize] == rank as i32 {
                                id
                            } else {
                                -1
                            }
                        })
                        .collect();
                    assert_eq!(
                        device_owned, want_owned,
                        "tiles {tiles} rank {rank}/{ranks}: mask"
                    );
                    gpu.hip.memcpy_dtoh(&mut raw, &ids_gpu.buf).expect("ids");
                    assert_eq!(
                        raw,
                        words(&ids),
                        "tiles {tiles}: the source ids were modified"
                    );
                    let mut dst = vec![0u8; tiles * tile_bytes];
                    gpu.hip.memcpy_dtoh(&mut dst, &dst_gpu.buf).expect("dst");
                    let mut want = vec![0xC3u8; tiles * tile_bytes];
                    for (t, &id) in ids.iter().enumerate() {
                        if id >= 0 && owner[id as usize] != rank as i32 {
                            want[t * tile_bytes..(t + 1) * tile_bytes]
                                .copy_from_slice(&peer[t * tile_bytes..(t + 1) * tile_bytes]);
                        }
                    }
                    assert!(dst == want, "tiles {tiles} rank {rank}/{ranks}: pull");
                    gpu.free_tensor(owned_gpu).expect("free owned");
                    gpu.free_tensor(dst_gpu).expect("free dst");
                }
                gpu.free_tensor(ids_gpu).expect("free ids");
                gpu.free_tensor(peer_gpu).expect("free peer");
            }
            gpu.free_tensor(owner_gpu).expect("free owner");
        }
    }
}
