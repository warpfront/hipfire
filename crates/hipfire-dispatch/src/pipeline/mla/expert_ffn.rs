// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared-plus-routed expert FFN of one DeepSeek-V4 row or rows window.
//!
//! `x = rmsnorm(x_in)` (plain and FWHT-rotated) → one shared SwiGLU expert with
//! a `swiglu_limit` clamp into `out` → a `sqrt(softplus)` router → `k` routed
//! MQ2-Lloyd experts selected either bias-aware (selection on `score + bias`,
//! weights on the unbiased score) or by a per-token hash table, accumulated
//! with `route_scale` into `out` or into a zeroed partial that is added to `out`
//! afterwards.
//!
//! Decode rows follow their `ExpertFfnSchedule`; batched rows always run serially.

use std::sync::LazyLock;

use rdna_compute::{Gpu, GpuTensor};

use super::project::project_pair_b3;
use super::{project_prepared, project_prepared_rows, E8SoaKernel, RowsProjection};
use crate::families::moe::{
    MoeBiasAwareMq2Backend, MoeBiasAwareParams, MoeBiasAwarePrefillParams, MoeFamily,
};
use crate::types::{dtype_needs_rotation, DispatchError};

static MOE: LazyLock<MoeFamily> = LazyLock::new(MoeFamily::new);

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

/// FFN input norm: `x_plain = rmsnorm(x_in)` and, when any consumer reads the
/// rotated activation, `x_rot = FWHT(x_plain)` in the same launch.
pub fn execute_ffn_norm(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let (layer, hidden) = (op.layer, op.hidden);
    match &op.route {
        ExpertFfnRoute::Decode { rotate_nox, .. } if op.rotate => gpu
            .deepseek4_fused_rmsnorm_rotate_mq_plain(
                op.x_in,
                op.norm,
                op.x_rot,
                op.x_plain,
                hidden,
                op.eps,
                *rotate_nox,
            )
            .map_err(|e| {
                hip_err(
                    format!("fused_rmsnorm_rotate_mq_plain ffn layer {layer}"),
                    e,
                )
            }),
        ExpertFfnRoute::Decode { .. } => gpu
            .rmsnorm_f32(op.x_in, op.norm, op.x_plain, op.eps)
            .map_err(|e| hip_err(format!("rmsnorm_f32 ffn-side plain l{layer}"), e)),
        ExpertFfnRoute::Batched { .. } if op.rotate => gpu
            .fused_rmsnorm_rotate_mq_plain_batched(
                op.x_in, op.norm, op.x_rot, op.x_plain, hidden, op.eps, op.rows,
            )
            .map_err(|e| {
                hip_err(
                    format!("fused_rmsnorm_rotate_mq_plain_batched ffn l{layer}"),
                    e,
                )
            }),
        ExpertFfnRoute::Batched { .. } => gpu
            .rmsnorm_batched(op.x_in, op.norm, op.x_plain, op.rows, hidden, op.eps)
            .map_err(|e| hip_err(format!("rmsnorm_batched ffn-side l{layer}"), e)),
    }
}

/// Shared SwiGLU expert: `out = w2 · clamp_silu_mul(w1 · x, w3 · x)` over the
/// first `shared_im` intermediate rows (dense TP shards the intermediate).
pub fn execute_shared_expert(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let (layer, hidden, im) = (op.layer, op.hidden, op.shared_im);
    match &op.route {
        ExpertFfnRoute::Decode { e8_soa, .. } => {
            let e8 = *e8_soa;
            let gate = op.shared_gate.sub_offset(0, im);
            let up = op.shared_up.sub_offset(0, im);
            let silu_rot = op.shared_rot.sub_offset(0, im);
            project_prepared(
                gpu,
                op.shared_w1,
                op.x_rot,
                op.x_plain,
                &gate,
                im,
                hidden,
                e8,
            )?;
            project_prepared(gpu, op.shared_w3, op.x_rot, op.x_plain, &up, im, hidden, e8)?;
            // SwiGLU with the swiglu_limit clamp, fused with the FWHT rotation when
            // the down weight is rotated (saves a launch and the gate round trip).
            if dtype_needs_rotation(op.shared_w2.dtype) {
                gpu.deepseek4_fused_silu_mul_clamp_mq_rotate(
                    &gate,
                    &up,
                    &silu_rot,
                    im,
                    op.swiglu_limit,
                )
                .map_err(|e| {
                    hip_err(
                        format!("deepseek4_fused_silu_mul_clamp_mq_rotate layer {layer}"),
                        e,
                    )
                })?;
            } else {
                gpu.deepseek4_silu_mul_clamp_f32(&gate, &up, &gate, op.swiglu_limit)
                    .map_err(|e| hip_err(format!("deepseek4_silu_mul_clamp layer {layer}"), e))?;
            }
            // Rotated down reads silu_rot, plain down reads the clamped gate.
            project_prepared(gpu, op.shared_w2, &silu_rot, &gate, op.out, hidden, im, e8)
        }
        ExpertFfnRoute::Batched {
            x_f16, projection, ..
        } => {
            let (rows, x_f16) = (op.rows, *x_f16);
            let paired = project_pair_b3(
                gpu,
                op.shared_w1,
                op.shared_w3,
                op.x_rot,
                op.shared_gate,
                op.shared_up,
                im,
                hidden,
                rows,
                projection,
            )?;
            if !paired {
                for (w, y) in [(op.shared_w1, op.shared_gate), (op.shared_w3, op.shared_up)] {
                    project_prepared_rows(
                        gpu,
                        w,
                        op.x_rot,
                        op.x_plain,
                        y,
                        im,
                        hidden,
                        rows,
                        Some(x_f16),
                        projection,
                    )?;
                }
            }
            gpu.deepseek4_silu_mul_clamp_f32_batched(
                op.shared_gate,
                op.shared_up,
                op.shared_gate,
                im,
                rows,
                op.swiglu_limit,
            )
            .map_err(|e| {
                hip_err(
                    format!("deepseek4_silu_mul_clamp_f32_batched shared l{layer}"),
                    e,
                )
            })?;
            if dtype_needs_rotation(op.shared_w2.dtype) {
                gpu.rotate_x_mq_batched(op.shared_gate, op.shared_rot, im, rows)
                    .map_err(|e| hip_err(format!("rotate_x_mq_batched shared silu l{layer}"), e))?;
            }
            project_prepared_rows(
                gpu,
                op.shared_w2,
                op.shared_rot,
                op.shared_gate,
                op.out,
                hidden,
                im,
                rows,
                Some(x_f16),
                projection,
            )
        }
    }
}

/// Router scores: `scores = sqrt(softplus(gate · x))` (unbiased).
pub fn execute_moe_route(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let Some(router) = &op.router else {
        return Ok(());
    };
    let (layer, hidden) = (op.layer, op.hidden);
    match &op.route {
        ExpertFfnRoute::Decode {
            e8_soa,
            fused_router,
            ..
        } => {
            if *fused_router {
                return gpu
                    .gemv_mfp4g32_e8_soa_u4_buffer_sqrt_softplus_gfx1151(
                        router.gate,
                        op.x_rot,
                        router.scores,
                        router.n_exp,
                        hidden,
                    )
                    .map_err(|e| {
                        hip_err(format!("fused gate gemv + sqrt_softplus layer {layer}"), e)
                    });
            }
            project_prepared(
                gpu,
                router.gate,
                op.x_rot,
                op.x_plain,
                router.scores,
                router.n_exp,
                hidden,
                *e8_soa,
            )?;
            gpu.sqrt_softplus_f32(router.scores)
                .map_err(|e| hip_err(format!("sqrt_softplus layer {layer}"), e))
        }
        ExpertFfnRoute::Batched {
            x_f16, projection, ..
        } => {
            project_prepared_rows(
                gpu,
                router.gate,
                op.x_rot,
                op.x_plain,
                router.scores,
                router.n_exp,
                hidden,
                op.rows,
                Some(*x_f16),
                projection,
            )?;
            gpu.sqrt_softplus_f32(router.scores)
                .map_err(|e| hip_err(format!("sqrt_softplus_f32 moe scores l{layer}"), e))
        }
    }
}

/// Where a hash-routed layer reads the current token id.
#[derive(Clone, Copy)]
pub enum HashToken<'a> {
    /// Device scalar, re-read by every graph replay.
    Device(&'a GpuTensor),
    /// Kernel argument.
    Host(u32),
}

/// Per-token hash selection: experts `tid2eid[token]`, weights the normalized
/// unbiased scores at those experts times `route_scale`.
#[derive(Clone, Copy)]
pub enum HashSelection<'a> {
    /// Device `tid2eid` table.
    Device {
        tid2eid: &'a GpuTensor,
        token: HashToken<'a>,
    },
    /// Host gather: download the scores, gather `row` (this token's `k` table
    /// entries), normalize and upload. A non-positive score sum skips the
    /// routed experts for this token.
    Host { row: &'a [u32] },
}

/// Hash-routed experts of one row (`k` MQ2-Lloyd experts, gate/up and down
/// through the indexed kernels or the model's native MQ2 backend).
pub struct HashRoutedExpertsOp<'a> {
    pub selection: HashSelection<'a>,
    pub scores: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    pub rot_batch: &'a GpuTensor,
    pub gate_up_ptrs: &'a GpuTensor,
    pub down_ptrs: &'a GpuTensor,
    pub gate_up_dummy: Option<&'a GpuTensor>,
    pub native: Option<&'a dyn MoeBiasAwareMq2Backend>,
    /// Accumulation target (`out += route_scale · Σ w_k · down_k`).
    pub out: &'a GpuTensor,
    pub hidden: usize,
    pub im: usize,
    pub k: usize,
    pub n_exp: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Down-projection kernel variant without the activation reload.
    pub down_nox: bool,
    pub layer: usize,
}

impl HashRoutedExpertsOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.k == 0 || self.n_exp == 0 || self.im == 0 || self.hidden == 0 {
            return Err(DispatchError::Hip(
                "hash routed experts have empty geometry".into(),
            ));
        }
        if let HashSelection::Host { row } = self.selection {
            if row.len() != self.k {
                return Err(DispatchError::Hip(format!(
                    "hash l{}: host tid2eid row has {} entries, want {}",
                    self.layer,
                    row.len(),
                    self.k
                )));
            }
        }
        Ok(())
    }
}

/// Host gather of the hash selection; `false` when the gathered score sum is
/// non-positive (the routed experts are then skipped for this token).
fn hash_select_host(
    gpu: &mut Gpu,
    op: &HashRoutedExpertsOp<'_>,
    row: &[u32],
) -> Result<bool, DispatchError> {
    let layer = op.layer;
    let scores = gpu
        .download_f32(op.scores)
        .map_err(|e| hip_err(format!("d2h scores hash l{layer}"), e))?;
    let ids: Vec<u32> = row.iter().map(|&i| i.min((op.n_exp - 1) as u32)).collect();
    let Some(weights) = gather_normalized_weights(&scores, &ids) else {
        return Ok(false);
    };
    let idx_bytes: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&op.topk_indices.buf, &idx_bytes)
        .map_err(|e| hip_err(format!("htod topk_indices hash l{layer}"), e))?;
    let w_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|&w| (w * op.route_scale).to_le_bytes())
        .collect();
    gpu.memcpy_htod_auto(&op.topk_weights.buf, &w_bytes)
        .map_err(|e| hip_err(format!("htod topk_weights hash l{layer}"), e))?;
    Ok(true)
}

/// The scores at `indices` (out of range reads 0), normalized to sum to 1;
/// `None` when the sum is non-positive.
fn gather_normalized_weights(scores: &[f32], indices: &[u32]) -> Option<Vec<f32>> {
    let mut wts: Vec<f32> = indices
        .iter()
        .map(|&i| *scores.get(i as usize).unwrap_or(&0.0))
        .collect();
    let s: f32 = wts.iter().sum();
    if s <= 0.0 {
        return None;
    }
    for w in wts.iter_mut() {
        *w /= s;
    }
    Some(wts)
}

pub fn execute_hash_routed_experts(
    gpu: &mut Gpu,
    op: &HashRoutedExpertsOp<'_>,
) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    let (layer, k, im, hidden) = (op.layer, op.k, op.im, op.hidden);
    match op.selection {
        HashSelection::Device {
            tid2eid,
            token: HashToken::Device(token),
        } => gpu
            .hash_router_normalize_f32_buf(
                tid2eid,
                op.scores,
                token,
                op.topk_indices,
                op.topk_weights,
                op.n_exp as i32,
                k as i32,
                op.route_scale,
            )
            .map_err(|e| hip_err(format!("hash_router_normalize_buf hash l{layer}"), e))?,
        HashSelection::Device {
            tid2eid,
            token: HashToken::Host(token),
        } => gpu
            .hash_router_normalize_f32(
                tid2eid,
                op.scores,
                op.topk_indices,
                op.topk_weights,
                token as i32,
                op.n_exp as i32,
                k as i32,
                op.route_scale,
            )
            .map_err(|e| hip_err(format!("hash_router_normalize hash l{layer}"), e))?,
        HashSelection::Host { row } => {
            if !hash_select_host(gpu, op, row)? {
                return Ok(());
            }
        }
    }

    if let Some(native) = op.native {
        native
            .gate_up(
                gpu,
                op.gate_up_ptrs,
                op.gate_up_dummy,
                op.topk_indices,
                op.x_rot,
                op.gate_batch,
                op.up_batch,
                2 * im,
                hidden,
                k,
            )
            .map_err(|e| DispatchError::Hip(format!("native gate_up hash l{layer}: {e}")))?;
    } else {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            op.gate_up_ptrs,
            op.topk_indices,
            op.x_rot,
            op.gate_batch,
            op.up_batch,
            2 * im,
            hidden,
            k,
        )
        .map_err(|e| hip_err(format!("fused gate_up hash l{layer}"), e))?;
    }
    gpu.deepseek4_silu_mul_clamp_f32_batched(
        op.gate_batch,
        op.up_batch,
        op.gate_batch,
        im,
        k,
        op.swiglu_limit,
    )
    .map_err(|e| hip_err(format!("deepseek4_silu_mul_clamp batched hash l{layer}"), e))?;
    if let Some(native) = op.native {
        native
            .rotate_x_batched(gpu, op.gate_batch, op.rot_batch, im, k)
            .map_err(|e| DispatchError::Hip(format!("native rotate hash l{layer}: {e}")))?;
    } else {
        gpu.rotate_x_mq_batched(op.gate_batch, op.rot_batch, im, k)
            .map_err(|e| hip_err(format!("rotate batched hash l{layer}"), e))?;
    }
    if let Some(native) = op.native {
        native
            .down_residual_scaled(
                gpu,
                op.down_ptrs,
                op.topk_indices,
                op.topk_weights,
                op.rot_batch,
                op.out,
                hidden,
                im,
                k,
            )
            .map_err(|e| DispatchError::Hip(format!("native down hash l{layer}: {e}")))
    } else {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
            op.down_ptrs,
            op.topk_indices,
            op.topk_weights,
            op.rot_batch,
            op.out,
            hidden,
            im,
            k,
            op.down_nox,
        )
        .map_err(|e| hip_err(format!("fused down hash l{layer}"), e))
    }
}

/// Routed experts of one row.
pub enum RoutedExpertsOp<'a> {
    /// Bias-aware top-k through the shared MoE family.
    BiasAware {
        params: MoeBiasAwareParams<'a>,
        layer: usize,
    },
    /// Per-token hash table.
    Hash(HashRoutedExpertsOp<'a>),
}

impl RoutedExpertsOp<'_> {
    pub fn validate_for_gpu(&self, gpu: &Gpu) -> Result<(), DispatchError> {
        match self {
            Self::BiasAware { params, .. } => {
                if params.k_top == 0 || params.n_exp == 0 || params.batch_size != 1 {
                    return Err(DispatchError::Hip(
                        "bias-aware routed experts need one row and k > 0".into(),
                    ));
                }
                Ok(())
            }
            Self::Hash(op) => op.validate_for_gpu(gpu),
        }
    }
}

pub fn execute_routed_experts(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    match &op.route {
        ExpertFfnRoute::Decode {
            routed: Some(RoutedExpertsOp::BiasAware { params, layer }),
            ..
        } => MOE
            .run_bias_aware(gpu, params)
            .map_err(|e| DispatchError::Hip(format!("ffn_routed l{layer} dispatch: {e}"))),
        ExpertFfnRoute::Decode {
            routed: Some(RoutedExpertsOp::Hash(hash)),
            ..
        } => execute_hash_routed_experts(gpu, hash),
        ExpertFfnRoute::Batched {
            routed: Some(params),
            ..
        } => {
            let layer = op.layer;
            MOE.run_bias_aware_prefill(gpu, params)
                .map_err(|e| DispatchError::Hip(format!("ffn_batched l{layer} dispatch: {e}")))
        }
        ExpertFfnRoute::Decode { routed: None, .. }
        | ExpertFfnRoute::Batched { routed: None, .. } => Ok(()),
    }
}

/// How the shared expert and the routed experts are ordered.
pub enum ExpertFfnSchedule<'a> {
    /// Shared into `out`, then routed accumulating into `out`.
    Serial,
    /// The shared expert runs on `side` while the routed experts accumulate
    /// into the zeroed `partial` on the active stream; the streams join and
    /// `out += partial` (the serial accumulation order at the HC boundary).
    Overlap {
        side: &'a hip_bridge::Stream,
        fork: &'a hip_bridge::Event,
        join: &'a hip_bridge::Event,
        partial: &'a GpuTensor,
    },
    /// The overlap topology issued serially on one stream (retained replay
    /// places the two branches on separate queues).
    Split { partial: &'a GpuTensor },
}

/// Router of one FFN: `scores = sqrt(softplus(gate · x))`, unbiased. Routed
/// experts read `scores`.
pub struct ExpertFfnRouter<'a> {
    pub gate: &'a GpuTensor,
    pub scores: &'a GpuTensor,
    pub n_exp: usize,
}

/// The FFN of one row or rows window: norm, shared expert, router and routed
/// experts. `route` holds only the knobs that differ between decode and batched.
pub struct ExpertFfnOp<'a> {
    pub x_in: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub x_plain: &'a GpuTensor,
    pub shared_w1: &'a GpuTensor,
    pub shared_w2: &'a GpuTensor,
    pub shared_w3: &'a GpuTensor,
    /// Gate scratch (≥ `shared_im`); also the plain down-projection input.
    pub shared_gate: &'a GpuTensor,
    pub shared_up: &'a GpuTensor,
    /// FWHT-rotated activation scratch (≥ `shared_im`) for an FWHT down weight.
    pub shared_rot: &'a GpuTensor,
    pub out: &'a GpuTensor,
    /// `Some` when the router runs; routed experts read its scores.
    pub router: Option<ExpertFfnRouter<'a>>,
    pub hidden: usize,
    pub shared_im: usize,
    pub swiglu_limit: f32,
    pub eps: f32,
    /// Also write `x_rot`: routed experts or FWHT shared gate/up read it.
    pub rotate: bool,
    pub rows: usize,
    pub layer: usize,
    pub route: ExpertFfnRoute<'a>,
}

/// The knobs that differ between decode and batched rows.
pub enum ExpertFfnRoute<'a> {
    Decode {
        /// Per-arch choice of the fused norm+rotate kernel variant.
        rotate_nox: bool,
        e8_soa: E8SoaKernel,
        /// Router GEMV with `sqrt(softplus)` folded into its store (gfx1151 MFP4 E8 only).
        fused_router: bool,
        /// `None` when the routed experts are off or absent on this layer.
        routed: Option<RoutedExpertsOp<'a>>,
        schedule: ExpertFfnSchedule<'a>,
    },
    Batched {
        x_f16: &'a GpuTensor,
        projection: RowsProjection,
        routed: Option<MoeBiasAwarePrefillParams<'a>>,
    },
}

impl ExpertFfnOp<'_> {
    pub fn validate_for_gpu(&self, gpu: &Gpu) -> Result<(), DispatchError> {
        let decode = matches!(self.route, ExpertFfnRoute::Decode { .. });
        super::attention::check_rows(self.rows, decode, || format!("expert FFN l{}", self.layer))?;
        if self.hidden == 0 || self.shared_im == 0 {
            return Err(DispatchError::Hip("expert ffn has empty geometry".into()));
        }
        if self.router.as_ref().is_some_and(|r| r.n_exp == 0) {
            return Err(DispatchError::Hip("moe route has empty geometry".into()));
        }
        if let ExpertFfnRoute::Decode {
            routed: Some(routed),
            ..
        } = &self.route
        {
            routed.validate_for_gpu(gpu)?;
        }
        Ok(())
    }
}

/// Runs `f` with `side` as the active stream, restoring the caller's stream
/// even when `f` fails.
fn on_stream<T>(
    gpu: &mut Gpu,
    side: &hip_bridge::Stream,
    f: impl FnOnce(&mut Gpu) -> Result<T, DispatchError>,
) -> Result<T, DispatchError> {
    // SAFETY: the alias never outlives this call and is never destroyed; the
    // owner keeps the stream alive across the whole decode step.
    let alias = unsafe { side.alias() };
    let primary = gpu.active_stream.replace(alias);
    let result = f(gpu);
    gpu.active_stream = primary;
    result
}

fn execute_routed_branch(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    execute_moe_route(gpu, op)?;
    execute_routed_experts(gpu, op)
}

fn execute_decode_schedule(
    gpu: &mut Gpu,
    op: &ExpertFfnOp<'_>,
    schedule: &ExpertFfnSchedule<'_>,
) -> Result<(), DispatchError> {
    let layer = op.layer;
    match *schedule {
        ExpertFfnSchedule::Serial => {
            execute_shared_expert(gpu, op)?;
            execute_routed_branch(gpu, op)
        }
        ExpertFfnSchedule::Split { partial } => {
            gpu.zero_f32(partial)
                .map_err(|e| hip_err(format!("zero Redline FFN routed partial l{layer}"), e))?;
            execute_shared_expert(gpu, op)?;
            execute_routed_branch(gpu, op)?;
            gpu.add_inplace_f32(op.out, partial)
                .map_err(|e| hip_err(format!("combine Redline FFN routed partial l{layer}"), e))
        }
        ExpertFfnSchedule::Overlap {
            side,
            fork,
            join,
            partial,
        } => {
            let primary = gpu.active_stream.as_ref().ok_or_else(|| {
                DispatchError::Hip(format!("FFN overlap l{layer}: no active stream"))
            })?;
            gpu.hip
                .memset_async(&partial.buf, 0, partial.byte_size(), primary)
                .map_err(|e| hip_err(format!("zero FFN overlap partial l{layer}"), e))?;
            gpu.hip
                .event_record(fork, Some(primary))
                .map_err(|e| hip_err(format!("record FFN overlap fork l{layer}"), e))?;
            gpu.hip
                .stream_wait_event(side, fork)
                .map_err(|e| hip_err(format!("wait FFN overlap fork l{layer}"), e))?;
            on_stream(gpu, side, |gpu| {
                execute_shared_expert(gpu, op)?;
                gpu.hip
                    .event_record(join, Some(side))
                    .map_err(|e| hip_err(format!("record FFN overlap join l{layer}"), e))
            })?;
            // The routed branch stays on the primary stream and accumulates only into the
            // zeroed partial, so it cannot race the shared expert's `out`.
            let routed = execute_routed_branch(gpu, op);
            let joined = match gpu.active_stream.as_ref() {
                Some(primary) => gpu
                    .hip
                    .stream_wait_event(primary, join)
                    .map_err(|e| hip_err(format!("wait FFN overlap join l{layer}"), e)),
                None => Err(DispatchError::Hip(format!(
                    "FFN overlap l{layer}: no active stream"
                ))),
            };
            routed?;
            joined?;
            gpu.add_inplace_f32(op.out, partial)
                .map_err(|e| hip_err(format!("combine FFN overlap partial l{layer}"), e))
        }
    }
}

/// Norm, shared expert, router, then routed experts, in each route's order.
pub fn execute_expert_ffn(gpu: &mut Gpu, op: &ExpertFfnOp<'_>) -> Result<(), DispatchError> {
    execute_ffn_norm(gpu, op)?;
    match &op.route {
        ExpertFfnRoute::Decode { schedule, .. } => execute_decode_schedule(gpu, op, schedule),
        ExpertFfnRoute::Batched { .. } => {
            execute_shared_expert(gpu, op)?;
            execute_routed_branch(gpu, op)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::gather_normalized_weights;

    #[test]
    fn gather_normalized_weights_basic() {
        let scores = vec![0.0, 2.0, 0.0, 1.0, 0.0];
        let idx = vec![1u32, 3];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // scores at idx = [2, 1] → normalized [2/3, 1/3]
        assert!((wts[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((wts[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn gather_normalized_weights_zero_sum_returns_none() {
        let scores = vec![0.0; 8];
        let idx = vec![0u32, 1, 2];
        assert!(gather_normalized_weights(&scores, &idx).is_none());
    }

    #[test]
    fn gather_normalized_weights_out_of_range_idx_is_zero() {
        // Hash table can in theory point past scores; we treat OOR as 0
        // (better than panicking — tid2eid is supposed to be in range).
        let scores = vec![1.0, 2.0, 3.0];
        let idx = vec![1u32, 999];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // sum = 2 + 0 = 2 → normalized [1.0, 0.0]
        assert!((wts[0] - 1.0).abs() < 1e-6);
        assert!((wts[1] - 0.0).abs() < 1e-6);
    }
}
