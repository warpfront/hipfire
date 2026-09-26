// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE scatter, permute, combine, and unscatter dispatch methods.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

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
            [grid_x, batch_size as u32, 1],
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
            kernels::MOE_SCATTER_FUSED_K8_SRC,
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
            [grid_x, n as u32, 1],
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
            [grid_x, m_total as u32, 1],
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
    /// remain separate symbols and launchers.
    pub fn moe_router_softmax_top10_f32(
        &mut self,
        logits: &GpuTensor,
        topk_idx: &GpuTensor,
        topk_w: &GpuTensor,
        tokens: usize,
        normalize_topk_prob: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const FUNC: &str = "moe_router_softmax_top10_f32";
        self.ensure_kernel(FUNC, kernels::MOE_ROUTER_SOFTMAX_TOP10_F32_SRC, FUNC)?;
        let lp = logits.buf.as_ptr();
        let ip = topk_idx.buf.as_ptr();
        let wp = topk_w.buf.as_ptr();
        let ne = 512i32;
        let kt = 10i32;
        let norm = i32::from(normalize_topk_prob);
        let mut params = [
            &lp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
            &kt as *const _ as *mut c_void,
            &norm as *const _ as *mut c_void,
        ];
        let bytes = (tokens * 512 + tokens * 10 * 2) * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
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
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
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
    ) -> HipResult<()> {
        self.bind_thread()?;
        const FUNC: &str = "moe_down_combine_top10_batched";
        self.ensure_kernel(FUNC, kernels::MOE_DOWN_COMBINE_TOP10_BATCHED_SRC, FUNC)?;
        let ep = expert_outputs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let rp = residual.buf.as_ptr();
        let hv = hidden as i32;
        let tv = tokens as i32;
        let mut params = [
            &ep as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
        ];
        let block = 256u32;
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
        )
    }

    /// [`Gpu::moe_down_combine_grouped_top10`] reading the grouped rows as BF16 bits (written by a
    /// `*_bf16out` MoE GEMM).  The per-token rank order is resolved once into `order`
    /// (scratch of at least `tokens * 10 * 8` bytes) before the combine reads it.
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
    ) -> HipResult<()> {
        self.bind_thread()?;
        if hidden % 8 != 0 || order.buf.size() < tokens * 10 * 8 {
            return Err(hip_bridge::HipError::new(
                0,
                "moe_down_combine_grouped_top10_bf16in: hidden % 8 != 0 or order scratch too small",
            ));
        }
        const MODULE: &str = "moe_down_combine_grouped_top10";
        for func in [
            "moe_combine_order_top10",
            "moe_down_combine_grouped_top10_bf16in",
        ] {
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
        let mut params = [
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
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
            "moe_down_combine_grouped_top10_bf16in",
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
    ) -> HipResult<()> {
        self.bind_thread()?;
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
        let mut params = [
            &gp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
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
        const FUNC: &str = "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded";
        self.ensure_kernel(
            FUNC,
            kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC,
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
        let bytes = batch_size.saturating_mul(10).saturating_mul(
            m.saturating_mul(k.div_ceil(128).saturating_mul(68))
                .saturating_add(k.saturating_mul(4))
                .saturating_add(m.saturating_mul(4)),
        );
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
        let o8_r16 = self.arch_caps.qwen4_tuned_routes() && m == 2560 && k == 640;
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
            o8_r16,
        )
    }

    /// Whether the F16 WMMA grouped QT53 down applies (Qwen4-tuned arch, one X row per
    /// top-10 slot, >= QWEN4_F16_WMMA_MIN_TOKENS tokens, not opted out).
    pub fn qwen4_moe_down_wmma_applies(&self, m: usize, k: usize, x_src_rows: usize) -> bool {
        self.arch_caps.qwen4_tuned_routes()
            && m == 2560
            && k == 640
            && x_src_rows >= 10 * crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
            && *crate::gemm::QWEN4_F16_WMMA
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
        let func = if bf16_out {
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out"
        } else {
            "gemm_mq4g128v2_moe_grouped_wmma_gfx1151"
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
        ];
        let bytes =
            grouped_rows.saturating_mul(m.saturating_mul(4).saturating_add(k.saturating_mul(2)));
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func, bytes);
        let result = self.launch_maybe_blob(
            func,
            [m.div_ceil(16) as u32, grouped_rows.div_ceil(16) as u32, 1],
            [32, 1, 1],
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
        o8_r16: bool,
    ) -> HipResult<()> {
        let (func, source, grid_x) = if o8_r16 {
            (
                "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_O8_R16_GFX1151_SRC,
                m.div_ceil(8),
            )
        } else {
            (
                "gemm_mq4g128v2_moe_grouped_top10_multirow",
                kernels::GEMM_MQ4G128V2_MOE_GROUPED_TOP10_MULTIROW_SRC,
                m,
            )
        };
        self.ensure_kernel(func, source, func)?;
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
            [grid_x as u32, grouped_rows.div_ceil(16) as u32, 1],
            [32, 1, 1],
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gfx1151 O8×R16 grouped QT53 down kernel must reproduce the generic
    /// multirow kernel bit for bit: two experts, dead (-1) slots, an all-dead
    /// tile, a sentinel expert tile and a partial final tile.
    #[test]
    #[ignore = "requires a Qwen4-tuned GPU and working HIP toolchain"]
    fn down_o8_r16_is_bit_identical_to_multirow() {
        const M: usize = 2560;
        const K: usize = 640;
        const GROUPED: usize = 330;
        const SLOTS: usize = 400;
        let mut gpu = match Gpu::init() {
            Ok(gpu) if gpu.arch_caps.qwen4_tuned_routes() => gpu,
            _ => {
                eprintln!("skip: Qwen4-tuned routes are off on this GPU");
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
        let mut run = |o8_r16: bool| {
            let sentinel = vec![f32::from_bits(0x7fc0_1234); GROUPED * M];
            let y = gpu
                .upload_f32(&sentinel, &[sentinel.len()])
                .expect("y upload");
            gpu.gemm_mq4g128v2_moe_grouped_top10_with(
                &ptrs_gpu, &tiles_gpu, &slots_gpu, &x_gpu, &y, M, K, 1, GROUPED, SLOTS, 2, o8_r16,
            )
            .expect("down launch");
            let out = gpu.download_f32(&y).expect("y download");
            gpu.free_tensor(y).expect("free y");
            out
        };
        let reference = run(false);
        let candidate = run(true);
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
}
