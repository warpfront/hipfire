// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 query/KV preparation: Q-LoRA prepare and project, joint KV
//! projection, tail-only YaRN RoPE on q and kv, and the packed attention-input
//! projections. Each op carries `rows`; its `route` selects the single-row
//! decode chain or the rows-window chain.

use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;

use super::compressor::CompressorRope;
use super::rows::AttentionInputPackB3;
use super::{project_prepared, project_prepared_rows, E8SoaKernel, RowsProjection};
use crate::types::DispatchError;

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

/// Tail-only YaRN RoPE on `q` and `kv` at the per-row positions in `pos`, or
/// (`inverse`) its inverse on the attention output (`kv` is `q`,
/// `kv_heads == 0`) so the rotation cancels across attention.
pub struct ApplyTailRopeOp<'a> {
    pub q: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub pos: &'a GpuTensor,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// YaRN parameters and rotated tail width.
    pub rope: CompressorRope,
    pub inverse: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: ApplyTailRopeRoute,
}

pub enum ApplyTailRopeRoute {
    /// Single row; `gfx1151_route` picks the gfx1151 kernel.
    Decode { gfx1151_route: bool },
    Batched,
}

impl ApplyTailRopeOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, ApplyTailRopeRoute::Decode { .. });
        let kv_heads_ok = self.kv_heads > 0 || self.inverse;
        if decode
            && (self.heads == 0 || !kv_heads_ok || self.head_dim == 0 || self.rope.rope_dim == 0)
        {
            return Err(DispatchError::Hip(
                "apply tail rope has empty geometry".into(),
            ));
        }
        super::attention::check_rows(self.rows, decode, || format!("apply tail rope l{}", self.layer))
    }
}

pub fn execute_apply_tail_rope(
    gpu: &mut Gpu,
    op: &ApplyTailRopeOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let rope = op.rope;
    match op.route {
        ApplyTailRopeRoute::Decode { gfx1151_route } => gpu
            .rope_tail_yarn_interleaved(
                gfx1151_route,
                op.q,
                op.kv,
                op.pos,
                op.heads as i32,
                op.kv_heads as i32,
                op.head_dim as i32,
                rope.rope_dim as i32,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.corr_low,
                rope.corr_high,
                op.inverse as i32,
            )
            .map_err(|e| hip_err("rope_tail_yarn_interleaved", e)),
        ApplyTailRopeRoute::Batched => {
            let dir = if op.inverse { " (inv)" } else { "" };
            gpu.rope_tail_yarn_interleaved_batched(
                op.q,
                op.kv,
                op.pos,
                op.heads as i32,
                op.kv_heads as i32,
                op.head_dim as i32,
                rope.rope_dim as i32,
                rope.freq_base,
                rope.freq_scale,
                rope.ext_factor,
                rope.attn_factor,
                rope.corr_low,
                rope.corr_high,
                op.inverse as i32,
                op.rows as i32,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("rope_tail_yarn_interleaved_batched{dir} l{}", op.layer),
                    e,
                )
            })
        }
    }
}

/// `kv = kv_norm(wkv @ tmp)`: projection (skipped when `preprojected`), then
/// in-place RMSNorm.
pub struct KvJointOp<'a> {
    pub wkv: &'a GpuTensor,
    pub kv_norm: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub tmp_plain: &'a GpuTensor,
    pub kv: &'a GpuTensor,
    pub kv_dim: usize,
    pub hidden: usize,
    pub eps: f32,
    pub preprojected: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: KvJointRoute<'a>,
}

pub enum KvJointRoute<'a> {
    /// Single row; `e8_soa` is the decode projection kernel.
    Decode { e8_soa: E8SoaKernel },
    /// Rows window: projection through the batched `x_f16` tile.
    Batched {
        x_f16: &'a GpuTensor,
        projection: RowsProjection,
    },
}

impl KvJointOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, KvJointRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || format!("KV joint l{}", self.layer))?;
        if matches!(self.route, KvJointRoute::Decode { .. })
            && (self.kv_dim == 0 || self.hidden == 0)
        {
            return Err(DispatchError::Hip("kv joint has empty geometry".into()));
        }
        Ok(())
    }
}

pub fn execute_kv_joint(gpu: &mut Gpu, op: &KvJointOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    match &op.route {
        KvJointRoute::Decode { e8_soa } => {
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
                    *e8_soa,
                )?;
            }
            // kv_norm RMSNorm in place (upstream DeepSeek V4: `kv = self.kv_norm(kv)`
            // after wkv, before apply_rotary_emb).
            gpu.rmsnorm_f32(op.kv, op.kv_norm, op.kv, op.eps)
                .map_err(|e| hip_err(format_args!("kv_norm rmsnorm layer {}", op.layer), e))
        }
        KvJointRoute::Batched { x_f16, projection } => {
            if !op.preprojected {
                project_prepared_rows(
                    gpu,
                    op.wkv,
                    op.tmp,
                    op.tmp_plain,
                    op.kv,
                    op.kv_dim,
                    op.hidden,
                    op.rows,
                    Some(*x_f16),
                    projection,
                )?;
            }
            gpu.rmsnorm_batched(op.kv, op.kv_norm, op.kv, op.rows, op.kv_dim, op.eps)
                .map_err(|e| hip_err(format_args!("kv_norm rmsnorm_batched l{}", op.layer), e))
        }
    }
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
    /// `dtype_needs_rotation(wq_a.dtype)`: also write the rotated `tmp`.
    pub rotate: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: QLoraPrepareRoute,
}

pub enum QLoraPrepareRoute {
    /// Single row; `gfx1151_route` picks the gfx1151 fused kernel.
    Decode { gfx1151_route: bool },
    /// Rows window.
    Batched,
}

impl QLoraPrepareOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, QLoraPrepareRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || format!("Q-LoRA prepare l{}", self.layer))?;
        if matches!(self.route, QLoraPrepareRoute::Decode { .. }) && self.hidden == 0 {
            return Err(DispatchError::Hip(
                "q lora prepare has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_q_lora_prepare(
    gpu: &mut Gpu,
    op: &QLoraPrepareOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    match op.route {
        QLoraPrepareRoute::Decode { gfx1151_route } => {
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
                    gfx1151_route,
                )
                .map_err(|e| {
                    hip_err(
                        format_args!("fused_rmsnorm_rotate_mq_plain layer {}", op.layer),
                        e,
                    )
                })
            } else {
                // Plain only: wq_a is Q8/F16/F32 → tmp not consumed downstream,
                // but compressor + indexer still read tmp_plain so it's required.
                gpu.rmsnorm_f32(op.hc_x_in, op.attn_norm, op.tmp_plain, op.eps)
                    .map_err(|e| {
                        hip_err(format_args!("rmsnorm_f32 attn-side plain l{}", op.layer), e)
                    })
            }
        }
        QLoraPrepareRoute::Batched => {
            if op.rotate {
                gpu.fused_rmsnorm_rotate_mq_plain_batched(
                    op.hc_x_in,
                    op.attn_norm,
                    op.tmp,
                    op.tmp_plain,
                    op.hidden,
                    op.eps,
                    op.rows,
                )
                .map_err(|e| {
                    hip_err(
                        format_args!("fused_rmsnorm_rotate_mq_plain_batched l{}", op.layer),
                        e,
                    )
                })
            } else {
                gpu.rmsnorm_batched(
                    op.hc_x_in,
                    op.attn_norm,
                    op.tmp_plain,
                    op.rows,
                    op.hidden,
                    op.eps,
                )
                .map_err(|e| hip_err(format_args!("rmsnorm_batched attn-side plain l{}", op.layer), e))
            }
        }
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
    pub hidden: usize,
    pub q_lora_rank: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub wq_a_preprojected: bool,
    /// `dtype_needs_rotation(wq_b.dtype)`: rotate `q_lat` before the wq_b projection.
    pub wq_b_needs_fwht: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: QLoraProjectRoute<'a>,
}

pub enum QLoraProjectRoute<'a> {
    /// Single row. `fused_qnorm_rotate` implies `wq_b_needs_fwht`: q_norm and
    /// the FWHT run as one fused launch.
    Decode {
        e8_soa: E8SoaKernel,
        fused_qnorm_rotate: bool,
    },
    /// Rows window.
    Batched {
        x_f16: &'a GpuTensor,
        projection: RowsProjection,
    },
}

impl QLoraProjectOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, QLoraProjectRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || format!("Q-LoRA project l{}", self.layer))?;
        if matches!(self.route, QLoraProjectRoute::Decode { .. })
            && (self.hidden == 0 || self.q_lora_rank == 0 || self.heads == 0 || self.head_dim == 0)
        {
            return Err(DispatchError::Hip(
                "q lora project has empty geometry".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_q_lora_project(
    gpu: &mut Gpu,
    op: &QLoraProjectOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    match &op.route {
        QLoraProjectRoute::Decode {
            e8_soa,
            fused_qnorm_rotate,
        } => {
            let e8_soa = *e8_soa;
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
                    e8_soa,
                )?;
            }

            // 2.5-3. Apply q_norm to the Q-LoRA bottleneck, then rotate for the
            // second projection. On gfx1151's E8 route the normalized plain tensor is
            // otherwise dead, so fuse RMSNorm + FWHT into q_lat_rot.
            if *fused_qnorm_rotate {
                gpu.fused_rmsnorm_rotate_mq(
                    op.q_lat,
                    op.q_norm,
                    op.q_lat_rot,
                    op.q_lora_rank,
                    op.eps,
                )
                .map_err(|e| hip_err(format_args!("fused q_norm+rotate layer {}", op.layer), e))?;
            } else {
                gpu.rmsnorm_f32(op.q_lat, op.q_norm, op.q_lat, op.eps)
                    .map_err(|e| hip_err(format_args!("q_norm rmsnorm layer {}", op.layer), e))?;
                if op.wq_b_needs_fwht {
                    gpu.rotate_x_mq(op.q_lat, op.q_lat_rot, op.q_lora_rank)
                        .map_err(|e| {
                            hip_err(format_args!("rotate_x_mq q_lat layer {}", op.layer), e)
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
                e8_soa,
            )?;

            // 4.5. Per-head RMSNorm of Q (upstream DeepSeek V4:
            //     `q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`).
            gpu.rmsnorm_f32(&q_view, op.q_head_ones, &q_view, op.eps)
                .map_err(|e| hip_err(format_args!("q per-head rmsnorm layer {}", op.layer), e))
        }
        QLoraProjectRoute::Batched { x_f16, projection } => {
            let rows = op.rows;
            if !op.wq_a_preprojected {
                project_prepared_rows(
                    gpu,
                    op.wq_a,
                    op.tmp,
                    op.tmp_plain,
                    op.q_lat,
                    op.q_lora_rank,
                    op.hidden,
                    rows,
                    Some(*x_f16),
                    projection,
                )?;
            }
            gpu.rmsnorm_batched(op.q_lat, op.q_norm, op.q_lat, rows, op.q_lora_rank, op.eps)
                .map_err(|e| hip_err(format_args!("q_norm rmsnorm_batched l{}", op.layer), e))?;
            if op.wq_b_needs_fwht {
                gpu.rotate_x_mq_batched(op.q_lat, op.q_lat_rot, op.q_lora_rank, rows)
                    .map_err(|e| {
                        hip_err(format_args!("rotate_x_mq_batched q_lat l{}", op.layer), e)
                    })?;
            }
            project_prepared_rows(
                gpu,
                op.wq_b,
                op.q_lat_rot,
                op.q_lat,
                op.q,
                op.heads * op.head_dim,
                op.q_lora_rank,
                rows,
                Some(*x_f16),
                projection,
            )?;
            gpu.rmsnorm_batched(
                op.q,
                op.q_head_ones,
                op.q,
                rows * op.heads,
                op.head_dim,
                op.eps,
            )
            .map_err(|e| hip_err(format_args!("q per-head rmsnorm_batched l{}", op.layer), e))
        }
    }
}

/// One job of a [`PackedProjectionsRoute::Decode`]: `output[..rows] = weight · x`.
#[derive(Clone, Copy)]
pub struct PackedProjection<'a> {
    pub weight: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
}

/// Projections of one shared rotated activation `x` (`k` wide) into several
/// outputs. Each output keeps its single-projection arithmetic.
pub struct PackedProjectionsOp<'a> {
    pub x: &'a GpuTensor,
    pub k: usize,
    pub rows: usize,
    pub layer: usize,
    pub route: PackedProjectionsRoute<'a>,
}

pub enum PackedProjectionsRoute<'a> {
    /// Mixed-row jobs in one launch (exact gfx1201, E8 SoA weights). Jobs run in
    /// slot order; the first `None` ends the list. `what` is the error label,
    /// without the `: {error}` suffix.
    Decode {
        jobs: [Option<PackedProjection<'a>>; 5],
        what: String,
    },
    /// Rows window: the gfx1151 B3 pack of the attention inputs (`wq_a` among them).
    Batched { pack: AttentionInputPackB3<'a> },
}

impl PackedProjectionsOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, PackedProjectionsRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || format!("packed projections l{}", self.layer))?;
        if let PackedProjectionsRoute::Decode { jobs, what } = &self.route {
            if self.k == 0 || jobs[0].is_none() {
                return Err(DispatchError::Hip(format!(
                    "{what}: packed projections have no jobs"
                )));
            }
        }
        Ok(())
    }
}

pub fn execute_packed_projections(
    gpu: &mut Gpu,
    op: &PackedProjectionsOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    match &op.route {
        PackedProjectionsRoute::Decode { jobs, what } => {
            let jobs = jobs.iter().map_while(|job| *job);
            let weights: SmallVec<[&GpuTensor; 5]> = jobs.clone().map(|job| job.weight).collect();
            let outputs: SmallVec<[&GpuTensor; 5]> = jobs.clone().map(|job| job.output).collect();
            let rows: SmallVec<[usize; 5]> = jobs.map(|job| job.rows).collect();
            gpu.gemv_mfp4g32_e8_soa_mixed_jobs_gfx1201(&weights, op.x, &outputs, &rows, op.k)
                .map_err(|e| hip_err(what, e))
        }
        PackedProjectionsRoute::Batched { pack } => {
            gpu.gemv_mfp4g32_e8_soa_batched_pack_b3_gfx1151(
                pack.weights,
                op.x,
                pack.outputs,
                pack.rows,
                op.k,
            )
            .map_err(|e| {
                hip_err(
                    format_args!("packed attention-input MFP4-E8 B3 l{}", op.layer),
                    e,
                )
            })
        }
    }
}
