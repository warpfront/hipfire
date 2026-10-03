// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE row fold (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the ten-row combine, the
//! shared-down fold, the HC write and the next HC read's norm + gate
//! projection in one launch per token (`kernels/src/hc_row_fold.hip`).
//!
//! idea from Gufo upstream (gufo-org/gufo @1071b361, MIT): row-level ownership
//! of the MoE/shared fold, the HC update and the next normalization.
//!
//! Bytewise the unfused chain `moe_down_combine_grouped_top10_bf16in_zinit` ->
//! `gemm_wmma_lds_128_256_32_64_k64_hcsd` -> `hyper_norm_gate` (F16 read output); the
//! shared-down GEMM becomes a plain BF16-bits store
//! ([`Gpu::gemm_bf16_xf32_f16_wmma_qwen4_bf16st`]).

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult, KernargBlob};

const SRC: &str = include_str!("../../../kernels/src/hc_row_fold.hip");
const ENTRY: &str = "hc_row_fold_norm_gate";

/// Operands of [`Gpu::hc_row_fold_norm_gate`].
pub struct HcRowFold<'a> {
    /// Grouped expert-down rows, BF16 bits `[grouped_rows, hidden]`.
    pub grouped_down: &'a GpuTensor,
    /// `moe_combine_order_top10` output, `rows * 10` `(row, weight)` pairs.
    pub order: &'a GpuTensor,
    /// Shared-down output, BF16 bits `[rows, hidden]`.
    pub shared_bf16: &'a GpuTensor,
    /// Per-token shared selector `[rows]` (F32).
    pub selector: &'a GpuTensor,
    /// HC streams, BF16 bits `[rows, 4 * hidden]`, rewritten in place.
    pub streams: &'a GpuTensor,
    /// This block's HC write gate logits `[rows, 4]` (F32), from its paired read.
    pub gates: &'a GpuTensor,
    /// The next HC read's norm weight (BF16 `[4 * hidden]`).
    pub norm_weight: &'a GpuTensor,
    /// The next HC read's paired write's gate weight (BF16 `[4, 4 * hidden]`).
    pub gate_weight: &'a GpuTensor,
    /// Receives that next write's gate logits `[rows, 4]` (F32).
    pub next_gates: &'a GpuTensor,
    /// Receives the next read's F16 normalized row `[rows, ld16]`.
    pub normalized_f16: &'a GpuTensor,
    /// Row pitch of `normalized_f16` in elements (`Gpu::f16_row_pitch` of
    /// `4 * hidden`, what the HC read's F16 GEMMs read).
    pub ld16: usize,
    pub rows: usize,
    pub hidden: usize,
}

impl Gpu {
    /// `moe_combine_order_top10` alone: the per-token rank order (`order`, at
    /// least `tokens * 10 * 8` bytes) the BF16-row combine and the HC row fold
    /// both consume.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_order_top10(
        &mut self,
        inverse_perm: &GpuTensor,
        topk_indices: &GpuTensor,
        topk_weights: &GpuTensor,
        order: &GpuTensor,
        grouped_rows: usize,
        tokens: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if order.buf.size() < tokens * 10 * 8 {
            return Err(HipError::new(0, "moe_combine_order_top10: order scratch too small"));
        }
        const FUNC: &str = "moe_combine_order_top10";
        self.ensure_kernel("moe_down_combine_grouped_top10", kernels::MOE_DOWN_COMBINE_GROUPED_TOP10_SRC, FUNC)?;
        let ip = inverse_perm.buf.as_ptr();
        let tp = topk_indices.buf.as_ptr();
        let wp = topk_weights.buf.as_ptr();
        let op = order.buf.as_ptr();
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
        self.launch_maybe_blob(FUNC, [(tokens as u32).div_ceil(64), 1, 1], [64, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(ip);
            b.push_ptr(tp);
            b.push_ptr(wp);
            b.push_ptr(op);
            b.push_i32(gr);
            b.push_i32(tv);
            b
        })
    }

    /// Whether [`Gpu::hc_row_fold_norm_gate`] takes `hidden` columns: exact
    /// gfx1151, four branches of `hidden == 2560` (the width the byte-identity
    /// harness covers; the LDS partition is sized for it).
    pub fn hc_row_fold_applies(&self, hidden: usize) -> bool {
        self.arch_caps.is_gfx1151()
            && hidden == 2560
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
    }

    /// The combine + fold + HC write + next norm/gate row kernel.
    pub fn hc_row_fold_norm_gate(&mut self, p: &HcRowFold<'_>) -> HipResult<()> {
        let wide = 4 * p.hidden;
        if !self.hc_row_fold_applies(p.hidden)
            || p.rows == 0
            || p.norm_weight.dtype != DType::BF16
            || p.gate_weight.dtype != DType::BF16
            || p.norm_weight.numel() != wide
            || p.gate_weight.numel() < 4 * wide
            || p.order.buf.size() < p.rows * 10 * 8
            || p.shared_bf16.buf.size() < p.rows * p.hidden * 2
            || p.selector.numel() < p.rows
            || p.streams.buf.size() < p.rows * wide * 2
            || p.gates.numel() < p.rows * 4
            || p.next_gates.numel() < p.rows * 4
            || p.ld16 < wide
            || p.ld16 % 8 != 0
            || p.normalized_f16.buf.size() < p.rows * p.ld16 * 2
        {
            return Err(HipError::new(0, "hc_row_fold_norm_gate: route does not apply"));
        }
        self.bind_thread()?;
        self.ensure_kernel("hc_row_fold", SRC, ENTRY)?;
        let gp = p.grouped_down.buf.as_ptr();
        let op = p.order.buf.as_ptr();
        let sp = p.shared_bf16.buf.as_ptr();
        let selp = p.selector.buf.as_ptr();
        let stp = p.streams.buf.as_ptr();
        let gtp = p.gates.buf.as_ptr();
        let nwp = p.norm_weight.buf.as_ptr();
        let gwp = p.gate_weight.buf.as_ptr();
        let ngp = p.next_gates.buf.as_ptr();
        let nop = p.normalized_f16.buf.as_ptr();
        let hv = p.hidden as i32;
        let ldv = p.ld16 as i32;
        let lds = (p.hidden * 8 + 4 * 256 * 4 + p.hidden * 2 + 16) as u32;
        let mut params = [
            &gp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &selp as *const _ as *mut c_void,
            &stp as *const _ as *mut c_void,
            &gtp as *const _ as *mut c_void,
            &nwp as *const _ as *mut c_void,
            &gwp as *const _ as *mut c_void,
            &ngp as *const _ as *mut c_void,
            &nop as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &ldv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(ENTRY, [p.rows as u32, 1, 1], [256, 1, 1], lds, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(gp);
            b.push_ptr(op);
            b.push_ptr(sp);
            b.push_ptr(selp);
            b.push_ptr(stp);
            b.push_ptr(gtp);
            b.push_ptr(nwp);
            b.push_ptr(gwp);
            b.push_ptr(ngp);
            b.push_ptr(nop);
            b.push_i32(hv);
            b.push_i32(ldv);
            b
        })
    }
}
