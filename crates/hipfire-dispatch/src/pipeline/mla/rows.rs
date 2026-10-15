// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-batched (prefill / verify) MLA operations with no decode counterpart:
//! every operation runs `rows` consecutive positions
//! `start_pos..start_pos + rows` through the `*_batched` kernels in one launch
//! chain. Operations decode shares take a `Batched` route instead.
//!
//! `capture_safe` operations read every position-dependent count from device
//! arrays uploaded once per window, so a captured verify graph replays at any
//! depth; the direct route passes host-derived counts as kernel arguments.
//! Buffers are allocated by the architecture's ensure phase, never here.

use rdna_compute::{Gpu, GpuTensor};

use super::project::{project_prepared, project_prepared_rows};
use super::RowsProjection;
use crate::types::DispatchError;

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

pub(super) fn htod_i32(
    gpu: &mut Gpu,
    dst: &GpuTensor,
    values: &[i32],
    what: &str,
) -> Result<(), DispatchError> {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    gpu.memcpy_htod_auto(&dst.buf, bytes)
        .map_err(|e| hip_err(what, e))
}

/// Q8 embedding lookup of `tokens` and the HC stream broadcast of each row.
pub struct EmbedStreamsOp<'a> {
    pub token_embd: &'a GpuTensor,
    pub tokens: &'a GpuTensor,
    pub embed: &'a GpuTensor,
    pub streams: &'a GpuTensor,
    pub hidden: usize,
    pub hc_mult: usize,
    pub rows: usize,
}

pub fn execute_embed_streams(gpu: &mut Gpu, op: &EmbedStreamsOp<'_>) -> Result<(), DispatchError> {
    gpu.embedding_lookup_q8_batched(op.token_embd, op.embed, op.tokens, op.rows, op.hidden)
        .map_err(|e| hip_err("embedding_lookup_q8_batched", e))?;
    gpu.hc_streams_init_from_embed_batched(
        op.embed,
        op.streams,
        op.hidden as i32,
        op.hc_mult as i32,
        op.rows as i32,
    )
    .map_err(|e| hip_err("hc_streams_init_from_embed_batched", e))
}

/// The gfx1151 three-row pack: every E8 projection of the attention input
/// (`wq_a`, `wkv`, compressor and indexer projections) as one B3 grid.
pub struct AttentionInputPackB3<'a> {
    pub weights: [&'a GpuTensor; 7],
    pub outputs: [&'a GpuTensor; 7],
    pub rows: [usize; 7],
}

/// SWA visibility of `rows` rows: the pre-window ring plus the window's own
/// KV rows, staged `[rows, head_dim, win]` (K = V tied).
pub struct SwaStageOp<'a> {
    pub swa_k: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub staged: &'a GpuTensor,
    pub positions: &'a GpuTensor,
    pub start_pos: u32,
    pub win: usize,
    pub head_dim: usize,
    pub capture_safe: bool,
    pub rows: usize,
    pub layer: usize,
}

pub fn execute_swa_stage(gpu: &mut Gpu, op: &SwaStageOp<'_>) -> Result<(), DispatchError> {
    let staged = if op.capture_safe {
        gpu.swa_visibility_stage_batched_pos(
            op.swa_k,
            op.kv,
            op.staged,
            op.positions,
            op.win as i32,
            op.head_dim as i32,
            op.rows as i32,
        )
    } else {
        gpu.swa_visibility_stage_batched(
            op.swa_k,
            op.kv,
            op.staged,
            op.start_pos as i32,
            op.win as i32,
            op.head_dim as i32,
            op.rows as i32,
        )
    };
    staged.map_err(|e| {
        hip_err(
            format_args!("swa_visibility_stage_batched l{}", op.layer),
            e,
        )
    })
}

/// DSpark target-layer capture: the mean of the four HC streams of each row
/// (`hc_input_map` with 0.25 weights into `tmp`), scattered into slot `slot`
/// of the `[rows, n_slots, hidden]` capture.
pub struct StreamMeanCaptureOp<'a> {
    pub quarter: &'a GpuTensor,
    pub streams: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub caps: &'a GpuTensor,
    pub hidden: usize,
    pub n_slots: usize,
    pub slot: usize,
    /// One typed strided copy (capture / replay) instead of a D2D per row.
    pub typed_copy: bool,
    pub rows: usize,
    pub layer: usize,
}

pub fn execute_stream_mean_capture(
    gpu: &mut Gpu,
    op: &StreamMeanCaptureOp<'_>,
) -> Result<(), DispatchError> {
    let (l, rows, hidden) = (op.layer, op.rows, op.hidden);
    gpu.hc_input_map_4stream_batched(op.quarter, op.streams, op.tmp, hidden as i32, rows as i32)
        .map_err(|e| hip_err(format_args!("dspark_capture mean-pool l{l}"), e))?;
    if op.typed_copy {
        return gpu
            .copy_f32_strided_slot_buffer(
                op.caps,
                op.tmp,
                rows,
                hidden,
                op.n_slots * hidden,
                op.slot * hidden,
            )
            .map_err(|e| hip_err(format_args!("dspark_capture typed scatter l{l}"), e));
    }
    let row_bytes = hidden * std::mem::size_of::<f32>();
    for b in 0..rows {
        let dst = (b * op.n_slots + op.slot) * row_bytes;
        gpu.memcpy_dtod_at_auto(&op.caps.buf, dst, &op.tmp.buf, b * row_bytes, row_bytes)
            .map_err(|e| hip_err(format_args!("dspark_capture scatter l{l} b{b}"), e))?;
    }
    Ok(())
}

/// Latent KV of each context row of `x` (`[n, hidden]`) at its absolute
/// position, written into a sliding-window ring: `wkv` projection,
/// `kv_norm`, tail YaRN RoPE (KV heads only), ring write.
pub struct RingKvOp<'a> {
    pub wkv: &'a GpuTensor,
    pub kv_norm: &'a GpuTensor,
    pub x: &'a GpuTensor,
    /// `[hidden]` rotation scratch (rotated `wkv` only), `[kv_dim]` KV and
    /// one-slot position scratch.
    pub x_rot: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub position: &'a GpuTensor,
    pub ring: &'a GpuTensor,
    pub positions: Vec<u32>,
    pub hidden: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub win: usize,
    pub eps: f32,
    pub rope: super::compressor::CompressorRope,
    pub e8_soa: super::E8SoaKernel,
    pub layer: usize,
}

pub fn execute_ring_kv(gpu: &mut Gpu, op: &RingKvOp<'_>) -> Result<(), DispatchError> {
    let (l, hidden) = (op.layer, op.hidden);
    let kv_dim = op.n_kv * op.head_dim;
    let rotated = crate::types::dtype_needs_rotation(op.wkv.dtype);
    let r = &op.rope;
    for (j, &pos) in op.positions.iter().enumerate() {
        let x = op.x.sub_offset(j * hidden, hidden);
        if rotated {
            gpu.rotate_x_mq(&x, op.x_rot, hidden)
                .map_err(|e| hip_err(format_args!("dspark rotate main_x[{l}]"), e))?;
        }
        let x_rot = if rotated { op.x_rot } else { &x };
        project_prepared(gpu, op.wkv, x_rot, &x, op.kv, kv_dim, hidden, op.e8_soa)?;
        gpu.rmsnorm_f32(op.kv, op.kv_norm, op.kv, op.eps)
            .map_err(|e| hip_err(format_args!("dspark main_kv norm[{l}]"), e))?;
        htod_i32(gpu, op.position, &[pos as i32], "dspark htod pos1")?;
        gpu.rope_tail_yarn_interleaved_batched(
            op.kv,
            op.kv,
            op.position,
            0,
            op.n_kv as i32,
            op.head_dim as i32,
            r.rope_dim as i32,
            r.freq_base,
            r.freq_scale,
            r.ext_factor,
            r.attn_factor,
            r.corr_low,
            r.corr_high,
            0,
            1,
        )
        .map_err(|e| hip_err(format_args!("dspark main_kv rope[{l}]"), e))?;
        gpu.swa_ring_write_batched_f32(
            op.kv,
            op.ring,
            op.n_kv as i32,
            op.head_dim as i32,
            op.win as i32,
            pos as i32,
            1,
        )
        .map_err(|e| hip_err(format_args!("dspark ring write[{l}]"), e))?;
    }
    Ok(())
}

/// Bidirectional block staging: every block row sees the ring's committed
/// window (`n_committed` columns), then the block's own KV rows, then zeros
/// up to `stage_w`.
pub struct BlockStageOp<'a> {
    pub ring: &'a GpuTensor,
    pub block_kv: &'a GpuTensor,
    pub staged: &'a GpuTensor,
    pub win: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_committed: usize,
    pub block: usize,
    pub stage_w: usize,
    pub layer: usize,
}

pub fn execute_block_stage(gpu: &mut Gpu, op: &BlockStageOp<'_>) -> Result<(), DispatchError> {
    gpu.dspark_stage_kv(
        op.ring,
        op.block_kv,
        op.staged,
        op.win,
        op.kv_dim,
        op.head_dim,
        op.n_committed,
        op.block,
        op.stage_w,
    )
    .map_err(|e| hip_err(format_args!("dspark stage_kv[{}]", op.layer), e))
}
