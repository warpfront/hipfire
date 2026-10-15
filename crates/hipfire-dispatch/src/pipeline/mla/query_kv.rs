// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 single-row query/KV preparation: Q-LoRA prepare and project,
//! joint KV projection, and tail-only YaRN RoPE on q and kv.

use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;

use super::{project_prepared, E8SoaKernel};
use crate::types::DispatchError;

/// Tail-only YaRN RoPE on `q` and `kv` at the per-row positions in `pos`, or
/// (`inverse`) its inverse on the attention output (`kv` is `q`,
/// `kv_heads == 0`) so the rotation cancels across attention.
pub struct ApplyTailRopeOp<'a> {
    pub q: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub pos: &'a GpuTensor,
    pub gfx1151_route: bool,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub corr_low: f32,
    pub corr_high: f32,
    pub inverse: bool,
}

impl ApplyTailRopeOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let kv_heads_ok = self.kv_heads > 0 || self.inverse;
        if self.heads == 0 || !kv_heads_ok || self.head_dim == 0 || self.rope_dim == 0 {
            return Err(DispatchError::Hip(
                "apply tail rope has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_apply_tail_rope(
    gpu: &mut Gpu,
    op: &ApplyTailRopeOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    gpu.rope_tail_yarn_interleaved(
        op.gfx1151_route,
        op.q,
        op.kv,
        op.pos,
        op.heads as i32,
        op.kv_heads as i32,
        op.head_dim as i32,
        op.rope_dim as i32,
        op.freq_base,
        op.freq_scale,
        op.ext_factor,
        op.attn_factor,
        op.corr_low,
        op.corr_high,
        op.inverse as i32,
    )
    .map_err(|e| DispatchError::Hip(format!("rope_tail_yarn_interleaved: {e:?}")))
}

/// `kv = kv_norm(wkv @ tmp)`: projection (skipped when `preprojected`), then
/// in-place RMSNorm.
pub struct KvJointOp<'a> {
    pub wkv: &'a GpuTensor,
    pub kv_norm: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub e8_soa: E8SoaKernel,
    pub kv_dim: usize,
    pub hidden: usize,
    pub eps: f32,
    pub preprojected: bool,
    pub layer_idx: usize,
}

impl KvJointOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.kv_dim == 0 || self.hidden == 0 {
            return Err(DispatchError::Hip("kv joint has empty geometry".into()));
        }
        Ok(())
    }
}

pub fn execute_kv_joint(gpu: &mut Gpu, op: &KvJointOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    // wkv @ tmp → kv.  Dispatch on weight dtype (MQ4G256 / F32-from-F16).
    if !op.preprojected {
        project_prepared(
            gpu,
            op.wkv,
            op.tmp,
            op.tmp_plain,
            op.kv,
            op.kv_dim,
            op.hidden,
            op.e8_soa,
        )?;
    }
    // kv_norm RMSNorm in place (upstream DeepSeek V4: `kv = self.kv_norm(kv)`
    // after wkv, before apply_rotary_emb).
    gpu.rmsnorm_f32(op.kv, op.kv_norm, op.kv, op.eps)
        .map_err(|e| DispatchError::Hip(format!("kv_norm rmsnorm layer {}: {e:?}", op.layer_idx)))
}

/// Attention-side RMSNorm of `hc_x_in`: `tmp_plain` always, plus the FWHT-rotated
/// `tmp` when the wq_a route needs it (one fused launch).
pub struct QLoraPrepareOp<'a> {
    pub hc_x_in: &'a GpuTensor,
    pub attn_norm: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub hidden: usize,
    pub eps: f32,
    pub rotate: bool,
    pub gfx1151_route: bool,
    pub layer_idx: usize,
}

impl QLoraPrepareOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.hidden == 0 {
            return Err(DispatchError::Hip(
                "q lora prepare has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_q_lora_prepare(gpu: &mut Gpu, op: &QLoraPrepareOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    // 1. RMSNorm (+ optional FWHT) hc_x_in → tmp / tmp_plain. When both
    //    outputs are needed (the common DeepSeek V4 case), use the fused variant
    //    that writes both in one launch.
    if op.rotate {
        gpu.deepseek4_fused_rmsnorm_rotate_mq_plain(
            op.hc_x_in,
            op.attn_norm,
            op.tmp,
            op.tmp_plain,
            op.hidden,
            op.eps,
            op.gfx1151_route,
        )
        .map_err(|e| {
            DispatchError::Hip(format!(
                "fused_rmsnorm_rotate_mq_plain layer {}: {e:?}",
                op.layer_idx
            ))
        })
    } else {
        // Plain only: wq_a is Q8/F16/F32 → tmp not consumed downstream,
        // but compressor + indexer still read tmp_plain so it's required.
        gpu.rmsnorm_f32(op.hc_x_in, op.attn_norm, op.tmp_plain, op.eps)
            .map_err(|e| {
                DispatchError::Hip(format!(
                    "rmsnorm_f32 attn-side plain l{}: {e:?}",
                    op.layer_idx
                ))
            })
    }
}

/// Q-LoRA projection: `q_lat = wq_a @ tmp`, q_norm (fused with the FWHT for
/// wq_b when `fused_qnorm_rotate`), `q = wq_b @ q_lat_rot`, then per-head RMSNorm.
pub struct QLoraProjectOp<'a> {
    pub wq_a: &'a GpuTensor,
    pub wq_b: &'a GpuTensor,
    pub q_norm: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub q_lat: &'a GpuTensor,
    pub q_lat_rot: &'a GpuTensor,
    /// Shared q storage; the `[heads, head_dim]` view is taken in execute.
    pub q: &'a GpuTensor,
    pub q_head_ones: &'a GpuTensor,
    pub e8_soa: E8SoaKernel,
    pub hidden: usize,
    pub q_lora_rank: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub wq_a_preprojected: bool,
    pub wq_b_needs_fwht: bool,
    /// Implies `wq_b_needs_fwht`: q_norm and the FWHT run as one fused launch.
    pub fused_qnorm_rotate: bool,
    pub layer_idx: usize,
}

impl QLoraProjectOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.hidden == 0 || self.q_lora_rank == 0 || self.heads == 0 || self.head_dim == 0 {
            return Err(DispatchError::Hip(
                "q lora project has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_q_lora_project(gpu: &mut Gpu, op: &QLoraProjectOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    // 2. wq_a @ tmp → q_lat. M = q_lora_rank, K = hidden.
    if !op.wq_a_preprojected {
        project_prepared(
            gpu,
            op.wq_a,
            op.tmp,
            op.tmp_plain,
            op.q_lat,
            op.q_lora_rank,
            op.hidden,
            op.e8_soa,
        )?;
    }

    // 2.5-3. Apply q_norm to the Q-LoRA bottleneck, then rotate for the
    // second projection. On gfx1151's E8 route the normalized plain tensor is
    // otherwise dead, so fuse RMSNorm + FWHT into q_lat_rot.
    if op.fused_qnorm_rotate {
        gpu.fused_rmsnorm_rotate_mq(op.q_lat, op.q_norm, op.q_lat_rot, op.q_lora_rank, op.eps)
            .map_err(|e| {
                DispatchError::Hip(format!("fused q_norm+rotate layer {}: {e:?}", op.layer_idx))
            })?;
    } else {
        gpu.rmsnorm_f32(op.q_lat, op.q_norm, op.q_lat, op.eps)
            .map_err(|e| {
                DispatchError::Hip(format!("q_norm rmsnorm layer {}: {e:?}", op.layer_idx))
            })?;
        if op.wq_b_needs_fwht {
            gpu.rotate_x_mq(op.q_lat, op.q_lat_rot, op.q_lora_rank)
                .map_err(|e| {
                    DispatchError::Hip(format!("rotate_x_mq q_lat layer {}: {e:?}", op.layer_idx))
                })?;
        }
    }

    // 4. wq_b @ q_lat_rot → q. M = n_heads * head_dim, K = q_lora_rank.
    //    Use q_lat (un-rotated) for F16 path; q_lat_rot for MQ4 path.
    let q_total = op.heads * op.head_dim;
    let mut q_view = op.q.sub_offset(0, q_total);
    q_view.shape = vec![op.heads, op.head_dim];
    project_prepared(
        gpu,
        op.wq_b,
        op.q_lat_rot,
        op.q_lat,
        &q_view,
        q_total,
        op.q_lora_rank,
        op.e8_soa,
    )?;

    // 4.5. Per-head RMSNorm of Q (upstream DeepSeek V4:
    //     `q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`).
    gpu.rmsnorm_f32(&q_view, op.q_head_ones, &q_view, op.eps)
        .map_err(|e| {
            DispatchError::Hip(format!("q per-head rmsnorm layer {}: {e:?}", op.layer_idx))
        })
}

/// One job of a [`PackedProjectionsOp`]: `output[..rows] = weight · x`.
#[derive(Clone, Copy)]
pub struct PackedProjection<'a> {
    pub weight: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
}

/// Projections of one shared rotated activation `x` (`k` wide) into several
/// outputs in one mixed-row launch (exact gfx1201, E8 SoA weights). Each
/// output row keeps its single-projection arithmetic. Jobs run in slot order;
/// the first `None` ends the list.
pub struct PackedProjectionsOp<'a> {
    pub jobs: [Option<PackedProjection<'a>>; 5],
    pub x: &'a GpuTensor,
    pub k: usize,
    /// Error label, without the `: {error}` suffix.
    pub what: String,
}

impl PackedProjectionsOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.k == 0 || self.jobs[0].is_none() {
            return Err(DispatchError::Hip(format!(
                "{}: packed projections have no jobs",
                self.what
            )));
        }
        Ok(())
    }
}

pub fn execute_packed_projections(
    gpu: &mut Gpu,
    op: &PackedProjectionsOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let jobs = op.jobs.iter().map_while(|job| *job);
    let weights: SmallVec<[&GpuTensor; 5]> = jobs.clone().map(|job| job.weight).collect();
    let outputs: SmallVec<[&GpuTensor; 5]> = jobs.clone().map(|job| job.output).collect();
    let rows: SmallVec<[usize; 5]> = jobs.map(|job| job.rows).collect();
    gpu.gemv_mfp4g32_e8_soa_mixed_jobs_gfx1201(&weights, op.x, &outputs, &rows, op.k)
        .map_err(|e| DispatchError::Hip(format!("{}: {e:?}", op.what)))
}
