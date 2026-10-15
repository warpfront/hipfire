// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek4 mHC (manifold-constrained hyper-connection) decode ops.
//!
//! Each `execute_*` launches the same kernels, in the same order, with the
//! same arguments as the legacy arch path, so results stay bit-exact. Buffers
//! are allocated by the arch `ensure_*` functions, never here.

use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

/// mHC pre step for one layer: control projection, finalize, and input map.
pub struct HcPreOp<'a> {
    /// Residual streams `[hc_mult, hidden]` (read).
    pub streams: &'a GpuTensor,
    /// Control projection weights `hc_fn` (read).
    pub hc_fn: &'a GpuTensor,
    /// Control bias `hc_base` (read).
    pub hc_base: &'a GpuTensor,
    /// Control scale `hc_scale` (read).
    pub hc_scale: &'a GpuTensor,
    /// Control vector: pre `[0..4]`, post `[4..8]`, comb `[8..24]` (written).
    pub hc_c: &'a GpuTensor,
    /// Layer input `[hidden]` (written).
    pub hc_x_in: &'a GpuTensor,
    /// Layer index, used only in error labels.
    pub layer: usize,
    /// Hidden size.
    pub hidden: usize,
    /// Control projection width, `hidden * hc_mult`.
    pub x_dim: usize,
    /// HC epsilon.
    pub hc_eps: f32,
    /// Post-scale of the pre/post sigmoid.
    pub post_scale: f32,
    /// Sinkhorn iterations.
    pub sinkhorn_iters: i32,
    /// gfx1151 route of the plain control projection.
    pub gfx1151_route: bool,
    /// Control projection also finalizes `hc_c`; no separate finalize runs.
    pub fused_control_finalize: bool,
    /// `hc_finalize_control` replaces the alpha, sigmoid, and sinkhorn chain.
    pub fused_finalize: bool,
    /// `hc_finalize_input_map` also writes `hc_x_in`; no separate input map runs.
    pub fused_input_map: bool,
    /// Fused control uses the once-only rsqrt.
    pub control_rsqrt_once: bool,
}

impl HcPreOp<'_> {
    /// Checks geometry only; launches nothing.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.hidden == 0 || self.x_dim == 0 {
            return Err(DispatchError::Hip("hc pre has empty geometry".into()));
        }
        Ok(())
    }
}

/// Runs one mHC pre step: `hc_c` control and finalize, then the `hc_x_in` input map.
pub fn execute_hc_pre(gpu: &mut Gpu, op: &HcPreOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    const N_CTRL: usize = 24;
    let layer = op.layer;
    let n_ctrl = N_CTRL as i32;
    let x_dim = op.x_dim as i32;
    let c_view = op.hc_c.sub_offset(0, N_CTRL);
    let pre_view = op.hc_c.sub_offset(0, 4);
    let comb_view = op.hc_c.sub_offset(8, 16);
    if op.fused_control_finalize {
        gpu.hc_compute_control_vec4_finalize(
            op.streams,
            op.hc_fn,
            op.hc_base,
            op.hc_c,
            op.hc_scale,
            n_ctrl,
            x_dim,
            op.hc_eps,
            op.post_scale,
            op.sinkhorn_iters,
            op.control_rsqrt_once,
            false,
        )
        .map_err(|e| hip_err(format!("hc_compute_control_vec4_finalize layer {layer}"), e))?;
    } else {
        gpu.hc_compute_control(
            op.gfx1151_route,
            op.streams,
            op.hc_fn,
            op.hc_base,
            &c_view,
            n_ctrl,
            x_dim,
        )
        .map_err(|e| hip_err(format!("hc_compute_control layer {layer}"), e))?;
        if op.fused_input_map {
            gpu.hc_finalize_input_map(
                op.hc_c,
                op.hc_scale,
                op.hc_base,
                op.streams,
                op.hc_x_in,
                op.hidden as i32,
                op.hc_eps,
                op.post_scale,
                op.sinkhorn_iters,
            )
            .map_err(|e| hip_err(format!("hc_finalize_input_map layer {layer}"), e))?;
        } else if op.fused_finalize {
            gpu.hc_finalize_control(
                op.hc_c,
                op.hc_scale,
                op.hc_base,
                op.hc_eps,
                op.post_scale,
                op.sinkhorn_iters,
            )
            .map_err(|e| hip_err(format!("hc_finalize_control layer {layer}"), e))?;
        } else {
            gpu.hc_apply_alpha(&c_view, op.hc_scale, op.hc_base)
                .map_err(|e| hip_err(format!("hc_apply_alpha layer {layer}"), e))?;
            gpu.hc_pre_post_sigmoid_scale_f32(op.hc_c, op.hc_eps, op.post_scale)
                .map_err(|e| hip_err(format!("hc_pre_post_sigmoid_scale layer {layer}"), e))?;
            gpu.hc_sinkhorn_4x4(&comb_view, op.hc_eps, op.sinkhorn_iters)
                .map_err(|e| hip_err(format!("hc_sinkhorn_4x4 layer {layer}"), e))?;
        }
    }
    if !op.fused_input_map {
        gpu.hc_input_map_4stream(&pre_view, op.streams, op.hc_x_in, op.hidden as i32)
            .map_err(|e| hip_err(format!("hc_input_map layer {layer}"), e))?;
    }
    Ok(())
}

/// Transform source of one mix step.
pub enum HcMixBranch<'a> {
    /// Block output (`attn_out` or `ffn_out`) `[hidden]` (read).
    Transform(&'a GpuTensor),
    /// Three rank-local partials reduced inside the mix (gfx1201 TP3).
    Peer3([&'a GpuTensor; 3]),
    /// Four rank-local partials reduced inside the mix (gfx1201 TP4).
    Peer4([&'a GpuTensor; 4]),
}

/// Mix step after a sub-block: combines the block output into the residual streams.
pub struct HcPostOp<'a> {
    /// Transform source: one block output or peer partials.
    pub branch: HcMixBranch<'a>,
    /// Residual streams before the mix (read).
    pub streams_in: &'a GpuTensor,
    /// Mix output (written): `residual_streams_next` or the `q` scratch.
    pub streams_out: &'a GpuTensor,
    /// Control vector holding the post and comb slices (read).
    pub hc_c: &'a GpuTensor,
    /// Residual streams that receive a device copy of `streams_out` (written).
    pub writeback: Option<&'a GpuTensor>,
    /// Hidden size.
    pub hidden: usize,
    /// Bytes copied to `writeback`, `hc_mult * hidden * 4`.
    pub copy_bytes: usize,
    /// Kernel error label, without the `: {error}` suffix.
    pub kernel_what: &'static str,
    /// Copy error label, without the `: {error}` suffix.
    pub copy_what: &'static str,
}

impl HcPostOp<'_> {
    /// Checks geometry only; launches nothing.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.hidden == 0 {
            return Err(DispatchError::Hip("hc post has empty geometry".into()));
        }
        if self.writeback.is_some() && self.copy_bytes == 0 {
            return Err(DispatchError::Hip("hc post write-back has no bytes".into()));
        }
        Ok(())
    }
}

/// Runs one mix step: `streams_out = mix(streams_in, comb, post, branch)`, then the optional copy back.
pub fn execute_hc_post(gpu: &mut Gpu, op: &HcPostOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let post_view = op.hc_c.sub_offset(4, 4);
    let comb_view = op.hc_c.sub_offset(8, 16);
    let hidden = op.hidden as i32;
    match &op.branch {
        HcMixBranch::Transform(transform) => gpu
            .hc_mix_4stream(
                op.streams_in,
                &comb_view,
                &post_view,
                transform,
                op.streams_out,
                hidden,
            )
            .map_err(|e| hip_err(op.kernel_what, e))?,
        HcMixBranch::Peer3(partials) => gpu
            .hc_mix_4stream_peer3_gfx1201(
                op.streams_in,
                &comb_view,
                &post_view,
                *partials,
                op.streams_out,
                hidden,
            )
            .map_err(|e| hip_err(op.kernel_what, e))?,
        HcMixBranch::Peer4(partials) => gpu
            .hc_mix_4stream_peer4_gfx1201(
                op.streams_in,
                &comb_view,
                &post_view,
                *partials,
                op.streams_out,
                hidden,
            )
            .map_err(|e| hip_err(op.kernel_what, e))?,
    }
    if let Some(dst) = op.writeback {
        gpu.memcpy_dtod_auto(&dst.buf, &op.streams_out.buf, op.copy_bytes)
            .map_err(|e| hip_err(op.copy_what, e))?;
    }
    Ok(())
}

/// Head prologue: head mix, input map, optional MTP capture, RMSNorm, optional FWHT rotation.
pub struct HcHeadOp<'a> {
    /// Residual streams `[hc_mult, hidden]` (read).
    pub streams: &'a GpuTensor,
    /// Head mix weights `hc_head_fn` (read).
    pub hc_head_fn: &'a GpuTensor,
    /// Head mix bias `hc_head_base` (read).
    pub hc_head_base: &'a GpuTensor,
    /// Head pre mix `[hc_mult]` (written).
    pub head_hc_pre: &'a GpuTensor,
    /// Head input map `[hidden]` (written).
    pub head_hc_out: &'a GpuTensor,
    /// Output RMSNorm weight (read).
    pub output_norm: &'a GpuTensor,
    /// Normalized final hidden `[hidden]` (written).
    pub final_norm: &'a GpuTensor,
    /// Rotated final hidden for FWHT heads (written); `None` skips the rotation.
    pub final_norm_rot: Option<&'a GpuTensor>,
    /// Full stream capture `[hc_mult, hidden]` for MTP (written); `None` skips it.
    pub mtp_last_hidden: Option<&'a GpuTensor>,
    /// Head mix scale.
    pub head_scale: f32,
    /// Streams per token.
    pub hc_mult: usize,
    /// Hidden size.
    pub hidden: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// HC epsilon.
    pub hc_eps: f32,
}

impl HcHeadOp<'_> {
    /// Checks geometry only; launches nothing.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.hc_mult == 0 || self.hidden == 0 {
            return Err(DispatchError::Hip("hc head has empty geometry".into()));
        }
        Ok(())
    }
}

/// Runs the head prologue: head mix, input map, optional MTP capture, RMSNorm, optional rotation.
pub fn execute_hc_head(gpu: &mut Gpu, op: &HcHeadOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let x_dim = op.hidden * op.hc_mult;
    gpu.hc_head_compute_pre(
        op.streams,
        op.hc_head_fn,
        op.hc_head_base,
        op.head_hc_pre,
        op.hc_mult as i32,
        x_dim as i32,
        op.head_scale,
        op.rms_norm_eps,
        op.hc_eps,
    )
    .map_err(|e| hip_err("hc_head_compute_pre", e))?;
    gpu.hc_input_map_4stream(op.head_hc_pre, op.streams, op.head_hc_out, op.hidden as i32)
        .map_err(|e| hip_err("hc_input_map (head)", e))?;
    if let Some(dst) = op.mtp_last_hidden {
        gpu.memcpy_dtod_auto(&dst.buf, &op.streams.buf, op.hc_mult * op.hidden * 4)
            .map_err(|e| hip_err("capture full HC streams → mtp_last_hidden", e))?;
    }
    gpu.rmsnorm_f32(
        op.head_hc_out,
        op.output_norm,
        op.final_norm,
        op.rms_norm_eps,
    )
    .map_err(|e| hip_err("final rmsnorm_f32", e))?;
    if let Some(rot) = op.final_norm_rot {
        gpu.rotate_x_mq(op.final_norm, rot, op.hidden)
            .map_err(|e| hip_err("rotate_x_mq final_norm", e))?;
    }
    Ok(())
}
