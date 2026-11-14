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
        rms_scale: f32,
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
        let mut rs_val = rms_scale;
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
            &mut rs_val as *mut _ as *mut c_void,
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
                b.push_f32(rs_val);
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



    /// Pre-write hidden rows rescaled by the target final-norm weights
    /// (`out[r] = x[row_offset + r] * nw`): the upstream serial_table's
    /// `base` term input. See `dspine_scale_rows_gfx1100`.
    pub fn dspine_scale_rows(
        &mut self,
        x: &GpuTensor,
        nw: &GpuTensor,
        out: &GpuTensor,
        row_offset: usize,
        rows: usize,
        h: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_scale_rows_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut x_ptr = x.buf.as_ptr();
        let mut nw_ptr = nw.buf.as_ptr();
        let mut o_ptr = out.buf.as_ptr();
        let mut ro = row_offset as i32;
        let mut r_val = rows as i32;
        let mut h_val = h as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut nw_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut ro as *mut _ as *mut c_void,
            &mut r_val as *mut _ as *mut c_void,
            &mut h_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [((rows * h + 255) / 256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(nw_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(ro);
                b.push_i32(r_val);
                b.push_i32(h_val);
                b
            },
        )
    }


    /// Gather the pre-write scores at the post-write top-k candidate ids
    /// (see `dspine_gather16_gfx1100`).
    pub fn dspine_gather16(
        &mut self,
        src: &GpuTensor,
        cand: &GpuTensor,
        out: &GpuTensor,
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
        const SYM: &str = "dspine_gather16_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut s_ptr = src.buf.as_ptr();
        let mut c_ptr = cand.buf.as_ptr();
        let mut o_ptr = out.buf.as_ptr();
        let mut v_val = v as i32;
        let mut total = (rows * 16) as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut s_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut v_val as *mut _ as *mut c_void,
            &mut total as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [((rows * 16 + 255) / 256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(s_ptr);
                b.push_ptr(c_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(v_val);
                b.push_i32(total);
                b
            },
        )
    }


    /// Decode each draft row's 16 candidate mq4v2 head rows into the
    /// per-row slab the lattice score stage reads. Grid: [rows].
    #[allow(clippy::too_many_arguments)]
    pub fn dspine_lattice_head_rows(
        &mut self,
        cand: &GpuTensor,
        lm_head_raw: &GpuTensor,
        lm_rows: &GpuTensor,
        rows: usize,
        h: usize,
        r: usize,
        row_bytes: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_lattice_head_rows_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut cand_ptr = cand.buf.as_ptr();
        let mut lh_ptr = lm_head_raw.buf.as_ptr();
        let mut lr_ptr = lm_rows.buf.as_ptr();
        let mut h_val = h as i32;
        let mut r_val = r as i32;
        let mut rows_val = rows as i32;
        let mut rb_val = row_bytes as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut cand_ptr as *mut _ as *mut c_void,
            &mut lh_ptr as *mut _ as *mut c_void,
            &mut lr_ptr as *mut _ as *mut c_void,
            &mut h_val as *mut _ as *mut c_void,
            &mut r_val as *mut _ as *mut c_void,
            &mut rows_val as *mut _ as *mut c_void,
            &mut rb_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [rows as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(cand_ptr);
                b.push_ptr(lh_ptr);
                b.push_ptr(lr_ptr);
                b.push_i32(h_val);
                b.push_i32(r_val);
                b.push_i32(rows_val);
                b.push_i32(rb_val);
                b
            },
        )
    }

    /// `base_pre[row][c] = (raw * out_norm) . lm_head[cand[row][c]]`: the
    /// sender-free term of the serial_table score, against the decoded
    /// candidate head rows. Grid: [rows].
    #[allow(clippy::too_many_arguments)]
    pub fn dspine_lattice_base_pre(
        &mut self,
        raw: &GpuTensor,
        out_norm: &GpuTensor,
        lm_rows: &GpuTensor,
        mq_signs1: &GpuTensor,
        mq_signs2: &GpuTensor,
        base_pre: &GpuTensor,
        rows: usize,
        h: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        const SYM: &str = "dspine_lattice_base_pre_gfx1100";
        self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;

        let mut raw_ptr = raw.buf.as_ptr();
        let mut on_ptr = out_norm.buf.as_ptr();
        let mut lr_ptr = lm_rows.buf.as_ptr();
        let mut ms1_ptr = mq_signs1.buf.as_ptr();
        let mut ms2_ptr = mq_signs2.buf.as_ptr();
        let mut bp_ptr = base_pre.buf.as_ptr();
        let mut h_val = h as i32;
        let mut rows_val = rows as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut raw_ptr as *mut _ as *mut c_void,
            &mut on_ptr as *mut _ as *mut c_void,
            &mut lr_ptr as *mut _ as *mut c_void,
            &mut ms1_ptr as *mut _ as *mut c_void,
            &mut ms2_ptr as *mut _ as *mut c_void,
            &mut bp_ptr as *mut _ as *mut c_void,
            &mut h_val as *mut _ as *mut c_void,
            &mut rows_val as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SYM,
            [rows as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(raw_ptr);
                b.push_ptr(on_ptr);
                b.push_ptr(lr_ptr);
                b.push_ptr(ms1_ptr);
                b.push_ptr(ms2_ptr);
                b.push_ptr(bp_ptr);
                b.push_i32(h_val);
                b.push_i32(rows_val);
                b
            },
        )
    }

    /// DSpine refinement lattice (upstream build_dspine_lattice): exact
    /// per-(candidate, predecessor) bridge-write recomputation and head
    /// scoring, in three stages so every (row, predecessor) pair runs in its
    /// own block:
    ///
    ///   1. `dspine_lattice_msg_gfx1100` — predecessor codes + message
    ///      (`[rows*16, R]`),
    ///   2. `gemm_f32_batched` — `write_last [H,R] @ msg [rows*16,R]^T`, i.e.
    ///      the per-pair delta, read of the weight matrix once instead of
    ///      once per pair,
    ///   3. `dspine_lattice_score_gfx1100` — raw + rms*delta, rmsnorm, head
    ///      norm, FWHT, candidate scores.
    #[allow(clippy::too_many_arguments)]
    pub fn dspine_lattice(
        &mut self,
        raw: &GpuTensor,
        rms: &GpuTensor,
        query: &GpuTensor,
        codebook: &GpuTensor,
        pred_gate: &GpuTensor,
        prev_w: &GpuTensor,
        gate_bias: &GpuTensor,
        write_last: &GpuTensor,
        out_norm: &GpuTensor,
        mq_signs1: &GpuTensor,
        mq_signs2: &GpuTensor,
        prev_cand: &GpuTensor,
        anchor_code: &GpuTensor,
        lm_row_scratch: &GpuTensor,
        refined_out: &GpuTensor,
        scores: &GpuTensor,
        dbg_seg: &GpuTensor,
        msg: &GpuTensor,
        rows: usize,
        h: usize,
        r: usize,
        c_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1100() {
            return Err(hip_bridge::HipError::new(
                0,
                "dspine bridge kernels are gfx1100-only in v1",
            ));
        }
        let pairs = rows * 16;

        // 1. message per (row, predecessor).
        {
            const SYM: &str = "dspine_lattice_msg_gfx1100";
            self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;
            let mut q_ptr = query.buf.as_ptr();
            let mut cb_ptr = codebook.buf.as_ptr();
            let mut pg_ptr = pred_gate.buf.as_ptr();
            let mut pv_ptr = prev_w.buf.as_ptr();
            let mut gb_ptr = gate_bias.buf.as_ptr();
            let mut pc_ptr = prev_cand.buf.as_ptr();
            let mut ac_ptr = anchor_code.buf.as_ptr();
            let mut wl_ptr = write_last.buf.as_ptr();
            let mut mo_ptr = msg.buf.as_ptr();
            let mut dbg_ptr = dbg_seg.buf.as_ptr();
            let mut r_val = r as i32;
            let mut c_val = c_size as i32;
            let mut rows_val = rows as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut q_ptr as *mut _ as *mut c_void,
                &mut cb_ptr as *mut _ as *mut c_void,
                &mut pg_ptr as *mut _ as *mut c_void,
                &mut pv_ptr as *mut _ as *mut c_void,
                &mut gb_ptr as *mut _ as *mut c_void,
                &mut pc_ptr as *mut _ as *mut c_void,
                &mut ac_ptr as *mut _ as *mut c_void,
                &mut wl_ptr as *mut _ as *mut c_void,
                &mut mo_ptr as *mut _ as *mut c_void,
                &mut dbg_ptr as *mut _ as *mut c_void,
                &mut r_val as *mut _ as *mut c_void,
                &mut c_val as *mut _ as *mut c_void,
                &mut rows_val as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                SYM,
                [pairs as u32, 1, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(q_ptr);
                    b.push_ptr(cb_ptr);
                    b.push_ptr(pg_ptr);
                    b.push_ptr(pv_ptr);
                    b.push_ptr(gb_ptr);
                    b.push_ptr(pc_ptr);
                    b.push_ptr(ac_ptr);
                    b.push_ptr(wl_ptr);
                    b.push_ptr(mo_ptr);
                    b.push_ptr(dbg_ptr);
                    b.push_i32(r_val);
                    b.push_i32(c_val);
                    b.push_i32(rows_val);
                    b
                },
            )?;
        }

        // 2. delta for every pair in one GEMM: [H,R] x [pairs,R]^T -> [pairs,H].
        self.gemm_f32_batched(write_last, msg, refined_out, h, r, pairs)?;

        // 3. refine + score per (row, predecessor).
        {
            const SYM: &str = "dspine_lattice_score_gfx1100";
            self.ensure_kernel(SYM, DSPINE_SRC, SYM)?;
            let mut raw_ptr = raw.buf.as_ptr();
            let mut rms_ptr = rms.buf.as_ptr();
            let mut d_ptr = refined_out.buf.as_ptr();
            let mut on_ptr = out_norm.buf.as_ptr();
            let mut ms1_ptr = mq_signs1.buf.as_ptr();
            let mut ms2_ptr = mq_signs2.buf.as_ptr();
            let mut lr_ptr = lm_row_scratch.buf.as_ptr();
            let mut ro_ptr = refined_out.buf.as_ptr();
            let mut sc_ptr = scores.buf.as_ptr();
            let mut dbg_ptr = dbg_seg.buf.as_ptr();
            let mut h_val = h as i32;
            let mut rows_val = rows as i32;
            let mut dbg_off = (c_size + r) as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut raw_ptr as *mut _ as *mut c_void,
                &mut rms_ptr as *mut _ as *mut c_void,
                &mut d_ptr as *mut _ as *mut c_void,
                &mut on_ptr as *mut _ as *mut c_void,
                &mut ms1_ptr as *mut _ as *mut c_void,
                &mut ms2_ptr as *mut _ as *mut c_void,
                &mut lr_ptr as *mut _ as *mut c_void,
                &mut ro_ptr as *mut _ as *mut c_void,
                &mut sc_ptr as *mut _ as *mut c_void,
                &mut dbg_ptr as *mut _ as *mut c_void,
                &mut h_val as *mut _ as *mut c_void,
                &mut rows_val as *mut _ as *mut c_void,
                &mut dbg_off as *mut _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                SYM,
                [pairs as u32, 1, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(raw_ptr);
                    b.push_ptr(rms_ptr);
                    b.push_ptr(d_ptr);
                    b.push_ptr(on_ptr);
                    b.push_ptr(ms1_ptr);
                    b.push_ptr(ms2_ptr);
                    b.push_ptr(lr_ptr);
                    b.push_ptr(ro_ptr);
                    b.push_ptr(sc_ptr);
                    b.push_ptr(dbg_ptr);
                    b.push_i32(h_val);
                    b.push_i32(rows_val);
                    b.push_i32(dbg_off);
                    b
                },
            )?;
        }

        Ok(())
    }
}
