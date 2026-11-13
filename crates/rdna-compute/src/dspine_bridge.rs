// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DSpine arch-24 code-bridge launchers, gfx1100-only.
//!
//! The kernels live in `kernels/src/dspine_bridge.gfx1100.hip` and are
//! self-contained here via `include_str!` (no shared-registry edits), the
//! same pattern as `dflash_draft_fusion`. Every launcher validates the
//! gfx1100 arch up front and routes through `launch_maybe_blob` +
//! `KernargBlob` so the fast path stays hipGraph-capturable even though the
//! v1 bridged trunk refuses capture (see `DraftLayerBridge`).
//!
//! Operation order mirrors `hipfire_runtime::dspine::cpu_oracle`: F32
//! accumulation everywhere, BF16 only at the codebook storage boundary. The
//! three bridge GEMMs (projection, packed, delta) ride the existing F32
//! draft GEMM dispatch — the bridge weights are lifted BF16→F32 at load.

use crate::{Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;

pub(crate) const DSPINE_SRC: &str = include_str!("../../../kernels/src/dspine_bridge.gfx1100.hip");

impl Gpu {
    /// Row RMS stats + F32-normalized rows (`dspine_prepare_read_gfx1100`).
    /// `x` is the trunk's f32 block rows [m, H]; outputs `rms` [m] f32 and
    /// `normed` [m, H] f32 (the projection GEMM's input).
    pub fn dspine_prepare_read(
        &mut self,
        x: &GpuTensor,
        rms: &GpuTensor,
        normed: &GpuTensor,
        m: usize,
        h: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_prepare_read_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut x_ptr = x.buf.as_ptr();
        let mut rms_ptr = rms.buf.as_ptr();
        let mut normed_ptr = normed.buf.as_ptr();
        let mut h_val = h as i32;
        let mut eps_val = eps;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut rms_ptr as *mut _ as *mut c_void,
            &mut normed_ptr as *mut _ as *mut c_void,
            &mut h_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [m as u32, 1, 1],
            [256, 1, 1],
            256 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(rms_ptr);
                b.push_ptr(normed_ptr);
                b.push_i32(h_val);
                b.push_f32(eps_val);
                b
            },
        )
    }

    /// Code normalization (`dspine_normalize_codes_gfx1100`). `projected`
    /// [m, C] f32 (GEMM output); `codebook` [V, C] bf16 raw storage;
    /// `anchor_ids` [m/ T] u32; `codes` [m, C] f32 (the packed GEMM's input).
    pub fn dspine_normalize_codes(
        &mut self,
        projected: &GpuTensor,
        codebook: &GpuTensor,
        anchor_ids: &GpuTensor,
        codes: &GpuTensor,
        m: usize,
        c: usize,
        t: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_normalize_codes_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut p_ptr = projected.buf.as_ptr();
        let mut cb_ptr = codebook.buf.as_ptr();
        let mut a_ptr = anchor_ids.buf.as_ptr();
        let mut out_ptr = codes.buf.as_ptr();
        let mut c_val = c as i32;
        let mut t_val = t as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut p_ptr as *mut _ as *mut c_void,
            &mut cb_ptr as *mut _ as *mut c_void,
            &mut a_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut c_val as *mut _ as *mut c_void,
            &mut t_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [m as u32, 1, 1],
            [256, 1, 1],
            256 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p_ptr);
                b.push_ptr(cb_ptr);
                b.push_ptr(a_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(c_val);
                b.push_i32(t_val);
                b
            },
        )
    }

    /// Gate + predecessor value product (`dspine_feature_message_gfx1100`).
    /// `packed` [m, 3R] f32 (GEMM output, segments query/sender/value);
    /// `gate_bias` [R] f32; `message` [m, R] f32. Block-first rows write 0.
    pub fn dspine_feature_message(
        &mut self,
        packed: &GpuTensor,
        gate_bias: &GpuTensor,
        message: &GpuTensor,
        m: usize,
        r: usize,
        t: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_feature_message_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut p_ptr = packed.buf.as_ptr();
        let mut b_ptr = gate_bias.buf.as_ptr();
        let mut out_ptr = message.buf.as_ptr();
        let mut r_val = r as i32;
        let mut t_val = t as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut p_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut r_val as *mut _ as *mut c_void,
            &mut t_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [m as u32, 1, 1],
            [256, 1, 1],
            256 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(p_ptr);
                b.push_ptr(b_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(r_val);
                b.push_i32(t_val);
                b
            },
        )
    }

    /// In-place delta add (`dspine_finish_gfx1100`). `x` [m, H] f32 trunk
    /// block rows updated with `rms[row] * delta[row]` on non-block-first
    /// rows; block-first rows pass through.
    pub fn dspine_finish(
        &mut self,
        x: &GpuTensor,
        rms: &GpuTensor,
        delta: &GpuTensor,
        m: usize,
        h: usize,
        t: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_finish_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut x_ptr = x.buf.as_ptr();
        let mut rms_ptr = rms.buf.as_ptr();
        let mut d_ptr = delta.buf.as_ptr();
        // long long on the HIP side; same 8-byte arg-slot layout as u64 and
        // H is non-negative.
        let mut h_val = h as u64;
        let mut t_val = t as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut rms_ptr as *mut _ as *mut c_void,
            &mut d_ptr as *mut _ as *mut c_void,
            &mut h_val as *mut _ as *mut c_void,
            &mut t_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [((m * h + 255) / 256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(rms_ptr);
                b.push_ptr(d_ptr);
                b.push_u64(h_val);
                b.push_i32(t_val);
                b
            },
        )
    }

    /// Fused serial-refinement chain (): one
    /// workgroup per T-row block; rows walked serially from ,
    /// each row scoring its K head-logit candidates against the sender
    /// tables and advancing the chain to the argmax. / [n, K]
    /// f32/u32;  [n, R] f32; the three tables [V, R] f32;
    ///  [n] u32 out.
    #[allow(clippy::too_many_arguments)]
    pub fn dspine_serial_chain(
        &mut self,
        base: &GpuTensor,
        cand: &GpuTensor,
        rms: &GpuTensor,
        query: &GpuTensor,
        tsv_gate: &GpuTensor,
        tsv_val: &GpuTensor,
        tg: &GpuTensor,
        gate_bias: &GpuTensor,
        anchor_sid: u32,
        draft_ids: &GpuTensor,
        n: usize,
        k: usize,
        r: usize,
        t: usize,
        packed_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_serial_chain_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut base_ptr = base.buf.as_ptr();
        let mut cand_ptr = cand.buf.as_ptr();
        let mut rms_ptr = rms.buf.as_ptr();
        let mut q_ptr = query.buf.as_ptr();
        let mut sgate_ptr = tsv_gate.buf.as_ptr();
        let mut sval_ptr = tsv_val.buf.as_ptr();
        let mut tg_ptr = tg.buf.as_ptr();
        let mut gb_ptr = gate_bias.buf.as_ptr();
        let mut anchor = anchor_sid;
        let mut out_ptr = draft_ids.buf.as_ptr();
        let mut k_val = k as i32;
        let mut r_val = r as i32;
        let mut t_val = t as i32;
        let mut ps_val = packed_stride as i32;
        let mut n_val = n as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut base_ptr as *mut _ as *mut c_void,
            &mut cand_ptr as *mut _ as *mut c_void,
            &mut rms_ptr as *mut _ as *mut c_void,
            &mut q_ptr as *mut _ as *mut c_void,
            &mut sgate_ptr as *mut _ as *mut c_void,
            &mut sval_ptr as *mut _ as *mut c_void,
            &mut tg_ptr as *mut _ as *mut c_void,
            &mut gb_ptr as *mut _ as *mut c_void,
            &mut anchor as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut r_val as *mut _ as *mut c_void,
            &mut t_val as *mut _ as *mut c_void,
            &mut ps_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [((n + t - 1) / t) as u32, 1, 1],
            [256, 1, 1],
            512 * 4 + 16 * 16 * 4 + 16 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(base_ptr);
                b.push_ptr(cand_ptr);
                b.push_ptr(rms_ptr);
                b.push_ptr(q_ptr);
                b.push_ptr(sgate_ptr);
                b.push_ptr(sval_ptr);
                b.push_ptr(tg_ptr);
                b.push_ptr(gb_ptr);
                b.push_u32(anchor);
                b.push_ptr(out_ptr);
                b.push_i32(k_val);
                b.push_i32(r_val);
                b.push_i32(t_val);
                b.push_i32(ps_val);
                b.push_i32(n_val);
                b
            },
        )
    }


    /// Per-row top-16 over [rows, V] with gathered base scores: the serial
    /// refinement's candidate shortlist (see `dspine_topk16_gfx1100`).
    pub fn dspine_topk16(
        &mut self,
        logits: &GpuTensor,
        cand: &GpuTensor,
        base: &GpuTensor,
        rows: usize,
        v: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_topk16_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut l_ptr = logits.buf.as_ptr();
        let mut cand_ptr = cand.buf.as_ptr();
        let mut base_ptr = base.buf.as_ptr();
        let mut v_val = v as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut l_ptr as *mut _ as *mut c_void,
            &mut cand_ptr as *mut _ as *mut c_void,
            &mut base_ptr as *mut _ as *mut c_void,
            &mut v_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [rows as u32, 1, 1],
            [256, 1, 1],
            256 * 16 * 4 * 2,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(l_ptr);
                b.push_ptr(cand_ptr);
                b.push_ptr(base_ptr);
                b.push_i32(v_val);
                b
            },
        )
    }


}
