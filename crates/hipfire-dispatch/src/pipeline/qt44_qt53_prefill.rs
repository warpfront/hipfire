// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact QT44/QT53 top-10 grouped-prefill lowering.
//!
//! This module is deliberately narrow.  It is reached only after the sealed
//! MoE predicate has admitted the immutable grouped-kernel geometry and replicated route.
//! Generic MoE prefill continues to own the k=8 path; no QT53 format is
//! admitted by this module through a representative-dtype fallback.

use super::layer_ops::hip;
use crate::families::gemv::WeightRef;
use crate::families::moe::{MoePrefillParams, MoeRouteCapability};
use crate::types::DispatchError;
use rdna_compute::moe::SharedExpertActivation;
use rdna_compute::tensor_ops::{bf16_scaled_add_batched, Bf16ScaledAddBatched};
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn f32_view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

#[inline]
fn grouped_geometry(p: &MoePrefillParams<'_>) -> bool {
    let Some(policy) = p.route_policy else {
        return false;
    };
    p.batch_size > 0
        && crate::families::moe::grouped_route_geometry_supported(
            &policy,
            p.n_exp,
            p.k_top,
            p.gate_up_k,
            p.mi,
            p.gate_up_k,
            p.down_m,
            p.down_k,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
        )
}

fn require_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if grouped_geometry(p) {
        Ok(())
    } else {
        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "grouped-prefill-shape",
            arch: "",
            quant: "declared route",
        })
    }
}

/// Prepare the routed expert activation basis.  The route's router/shared
/// projections are BF16 and consume the natural input, while routed QT44
/// gate/up consumes the FWHT basis.
pub(crate) fn input_basis(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if gateup_rotates_f16(gpu, p, use_path2) {
        return Ok(());
    }
    hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))
}

/// Whether the path-2 F16 WMMA gate/up rotates the activation straight to
/// F16 itself (the bytes the F32 rotation + conversion produce): only when no
/// shared projection reads the F32 basis `x_rot_batch`.
fn gateup_rotates_f16(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2
        && gateup_bf16(gpu, p)
        && p.prelude.shared.as_ref().is_none_or(|shared| {
            [&shared.weights.selector, &shared.weights.gate]
                .iter()
                .all(|w| w.dtype != DType::MQ4G256V2)
        })
}

fn batch_projection(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> Result<(), DispatchError> {
    match weight.dtype {
        // BF16 rows read the natural activation; this route materializes the
        // packed basis once per layer, so no per-projection rotation is owed.
        DType::BF16 => super::project_weight(gpu, weight, x, y, batch_size, None),
        DType::F32 => hip(gpu.gemm_f32_batched(weight.buf, x, y, weight.m, weight.k, batch_size)),
        DType::MQ4G256V2 => {
            hip(gpu.gemm_mq4g256v2(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        DType::MQ4G128V2 => {
            hip(gpu.gemm_mq4g128v2_batched(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        // Qwen4's few-row decode copies: rows <= 8 run the decode GEMV per row.
        DType::Q8_0 => {
            hip(gpu.gemm_q8_0_batched_f32_chunked(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qt44-qt53-shared-projection-dtype",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// The grouped route's router is a row-batched projection even though the
/// generic MoE table intentionally has no BF16 router entry on RDNA3.
pub(crate) fn router_projection(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // Projected by the shared gate/up stage's launch instead.
    if router_with_shared(p) {
        return Ok(());
    }
    batch_projection(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
    )
}

/// Whether a few-row forward (<= 8 rows, where the grouped BF16 row kernel is
/// bitwise each projection's own) projects the router together with the
/// all-BF16 shared selector/gate/up in one launch (the shared gate/up stage,
/// which the route always runs right after the router stage).
fn router_with_shared(p: &MoePrefillParams<'_>) -> bool {
    p.batch_size <= 8
        && p.prelude.router.dtype == DType::BF16
        && p.prelude.shared.as_ref().is_some_and(|shared| {
            [
                &shared.weights.selector,
                &shared.weights.gate,
                &shared.weights.up,
            ]
            .iter()
            .all(|w| w.dtype == DType::BF16)
        })
}

/// Shared selector and gate/up are ordinary batched projections.  This is the
/// typed route's BF16 exception to the generic MoE prefill table: the common
/// family remains fail-closed for QT53 while this exact route uses the native
/// BF16 XF32 GEMM launcher.
pub(crate) fn shared_gate_up(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let weights = &shared.weights;
    if router_with_shared(p) {
        // All read the natural activation: one launch (one F16 conversion).
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&p.prelude.router, p.prelude.router_logits),
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else if [&weights.selector, &weights.gate, &weights.up]
        .iter()
        .all(|w| w.dtype == DType::BF16)
    {
        // All read the natural activation: one shared F16 conversion.
        super::layer_ops::project_weights(
            gpu,
            p.x_norm_batch,
            p.batch_size,
            None,
            &[
                (&weights.selector, shared.scalar),
                (&weights.gate, shared.gate_out),
                (&weights.up, shared.up_out),
            ],
        )?;
    } else {
        let x_selector = match shared.weights.selector.dtype {
            DType::BF16 | DType::F32 => p.x_norm_batch,
            DType::MQ4G256V2 => p.x_rot_batch,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "qt44-qt53-shared-selector-dtype",
                    arch: "",
                    quant: "unsupported",
                })
            }
        };
        batch_projection(
            gpu,
            &shared.weights.selector,
            x_selector,
            shared.scalar,
            p.batch_size,
        )?;
        let x_gate = match shared.weights.gate.dtype {
            DType::BF16 | DType::F32 | DType::Q8_0 => p.x_norm_batch,
            DType::MQ4G256V2 => p.x_rot_batch,
            _ => {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "qt44-qt53-shared-gate-dtype",
                    arch: "",
                    quant: "unsupported",
                })
            }
        };
        batch_projection(
            gpu,
            &shared.weights.gate,
            x_gate,
            shared.gate_out,
            p.batch_size,
        )?;
        batch_projection(gpu, &shared.weights.up, x_gate, shared.up_out, p.batch_size)?;
    }
    // BF16 recipe: every reader rounds what it reads (the top-10 router its
    // logits, the shared activation its selector, gate and up), so no
    // separate round trip is owed here.
    Ok(())
}

/// Apply the shared expert activation after the selector and gate/up projections.
pub(crate) fn shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // Path 2's fused unscatter launch runs it (see `shared_in_unscatter`).
    if use_path2 && shared_in_unscatter(gpu, p) {
        return Ok(());
    }
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    // The live rows only: the shared buffers are sized for the chunk cap.
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    if p.recipe.bf16_round_trip() {
        // The selector's round trip and sigmoid ride in the same launch; the
        // shared down's scaled add rounds the selector again on read.
        hip(gpu.shared_expert_activation_bf16_f32(
            &gate,
            &up,
            &rotated,
            shared.scalar,
            p.batch_size,
        ))?;
    } else {
        hip(gpu.silu_mul_f32(&gate, &up, &rotated))?;
    }
    if shared.weights.down.dtype == DType::MQ4G128V2 {
        hip(gpu.rotate_x_mq_128_v2(
            shared.rotated,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        ))?;
    }
    Ok(())
}

/// Fold the shared expert into the residual at the recipe's chosen placement,
/// applying the selector sigmoid exactly once.
pub(crate) fn shared_down(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if down.dtype == DType::BF16 && p.recipe.bf16_round_trip()
        && hip(gpu.qwen4_shared_down_bf16_epi(
            down.buf, shared.rotated, target, shared.scalar,
            down.m, down.k, p.batch_size,
        ))?
    {
        return Ok(());
    }
    let out = f32_view(p.down_expanded, 0, p.batch_size * p.down_m);
    match down.dtype {
        DType::BF16 | DType::F32 | DType::Q8_0 | DType::MQ4G256V2 | DType::MQ4G128V2 => {
            // The shared activation is already in the basis consumed by this
            // projection (G128 for QT53); the BF16 storage boundary, if any,
            // was applied by the recipe before this launch.
            batch_projection(gpu, down, shared.rotated, &out, p.batch_size)?;
        }
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "qt44-qt53-shared-down-dtype",
                arch: "",
                quant: "unsupported",
            })
        }
    }
    if p.recipe.bf16_round_trip() {
        // The scaled add rounds `out`, the scalar and the residual to BF16
        // itself and stores a BF16 value, so no round trip brackets it.
        bf16_scaled_add_batched(
            gpu,
            &Bf16ScaledAddBatched {
                residual: target,
                value: &out,
                scalar: shared.scalar,
                rows: p.batch_size,
                elements: p.down_m,
            },
        )
        .map_err(|error| DispatchError::Hip(error.to_string()))?;
    } else {
        hip(gpu.sigmoid_scaled_residual_add_batched_f32(
            target,
            &out,
            shared.scalar,
            p.batch_size,
            p.down_m,
        ))?;
    }
    Ok(())
}

/// The HC write that follows a sealed MoE call and may ride the shared down's
/// GEMM epilogue (`HIPFIRE_QWEN4_HC_FUSE` >= 3): its streams, the gates the
/// paired HC read produced, and the tensor it reads as `mixed` (which must be
/// the routed target).
pub(crate) struct HcSharedDown {
    pub streams: GpuTensor,
    pub gates: GpuTensor,
    pub mixed: *mut std::ffi::c_void,
    pub rows: usize,
    pub hidden: usize,
    pub state_bf16: bool,
    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the next HC read's operands, when the whole
    /// MoE tail runs as one row kernel (see [`row_fold_applies`]).
    pub row_fold: Option<HcRowFoldNext>,
}

/// The HC read that follows the HC write of a row-folded MoE call, and that
/// read's paired write: the row kernel computes the read's norm (its F16 row)
/// and that write's gate logits.
pub(crate) struct HcRowFoldNext {
    pub norm_weight: GpuTensor,
    pub gate_weight: GpuTensor,
    pub next_gates: GpuTensor,
}

impl HcRowFoldNext {
    /// What the row kernel needs of `read` and its paired `write`.
    pub(crate) fn offer(
        read: &super::layer_ops::HyperReadOp<'_>,
        write: &super::layer_ops::HyperWriteOp<'_>,
    ) -> Self {
        Self {
            norm_weight: f32_view(read.norm_weight, 0, read.norm_weight.numel()),
            gate_weight: f32_view(write.block_inject.buf, 0, write.block_inject.buf.numel()),
            next_gates: f32_view(write.gates, 0, read.rows * write.branches),
        }
    }
}

impl HcSharedDown {
    /// What the shared down needs of `write`, or `None` when its operands
    /// are not the plain four-branch F32-typed streams / gates it folds into.
    pub(crate) fn offer(write: &super::layer_ops::HyperWriteOp<'_>) -> Option<Self> {
        let wide = write.branches.checked_mul(write.hidden)?;
        let (streams_len, gates_len) = (write.rows.checked_mul(wide)?, write.rows.checked_mul(4)?);
        (write.branches == 4
            && write.input.buf.as_ptr() == write.output.buf.as_ptr()
            && write.input.dtype == DType::F32
            && write.input.numel() >= streams_len
            && write.gates.dtype == DType::F32
            && write.gates.numel() >= gates_len)
            .then(|| Self {
                streams: f32_view(write.input, 0, streams_len),
                gates: f32_view(write.gates, 0, gates_len),
                mixed: write.mixed.buf.as_ptr(),
                rows: write.rows,
                hidden: write.hidden,
                state_bf16: write.state_bf16,
                row_fold: None,
            })
    }
}

/// [`shared_down`] with the paired HC write folded into the GEMM epilogue
/// (`hc`).  Bitwise the shared down followed by `hyper_write`: the BF16 shared
/// down on the Qwen4 F16 WMMA route, the BF16 scaled add, then the write.  The
/// routed target is not rewritten (it is dead after the write).  Returns
/// `false`, having launched nothing, when `hc` does not fit this call; the
/// caller then runs [`shared_down`] and the write unfused.
pub(crate) fn shared_down_hc(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
) -> Result<bool, DispatchError> {
    require_geometry(p)?;
    let Some(shared) = p.prelude.shared.as_ref() else {
        return Ok(false);
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if !hc_shared_down_admits(gpu, p, hc) {
        if hc.row_fold.is_some() {
            return Err(DispatchError::Hip(
                "qt44 shared down: the HC row fold was planned but its route no longer applies".into(),
            ));
        }
        return Ok(false);
    }
    if let Some(next) = &hc.row_fold {
        shared_down_row_fold(gpu, p, hc, next)?;
        return Ok(true);
    }
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
        down.buf,
        shared.rotated,
        down.m,
        down.k,
        p.batch_size,
        target,
        shared.scalar,
        &hc.streams,
        &hc.gates,
        hc.state_bf16,
        false,
    ))?;
    Ok(true)
}

/// Whether the shared down of this call can carry `hc` (the conditions of
/// [`shared_down_hc`]).
fn hc_shared_down_admits(gpu: &Gpu, p: &MoePrefillParams<'_>, hc: &HcSharedDown) -> bool {
    let Some(shared) = p.prelude.shared.as_ref() else {
        return false;
    };
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_fuse_level() >= 3
        && down.dtype == DType::BF16
        && down.rotation.is_none()
        && down.awq_scale.is_none()
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && down.m == p.down_m
        && down.k == shared.intermediate
        && hc.rows == p.batch_size
        && hc.hidden == p.down_m
        && hc.mixed == target.buf.as_ptr()
        && target.numel() >= p.batch_size * p.down_m
        && shared.scalar.numel() >= p.batch_size
        && gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(down.buf, down.m, down.k, p.batch_size)
}

/// Whether this call's whole tail (combine, shared fold, `hc`'s HC write, the
/// next read's norm) can run as the HC row fold
/// (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the zero-initialized BF16-row combine of
/// path 2 into the routed target (which, nothing reading it, then holds the
/// BF16 shared-down rows), the HC-write shared down, BF16 streams.
pub(crate) fn row_fold_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    hc: &HcSharedDown,
) -> bool {
    let target = p.routed_out.unwrap_or(p.x_batch);
    gpu.flags.qwen4_hc_row_fold_enabled()
        && hc.state_bf16
        && combine_initial_zero_applies(gpu, p, use_path2)
        && hc_shared_down_admits(gpu, p, hc)
        && gpu.hc_row_fold_applies(hc.hidden)
        && p.k_top == 10
        && p.down_expanded.buf.size() >= p.batch_size * 10 * 8
        && target.buf.size() >= p.batch_size * p.down_m * 2
}

/// The row-folded combine stage: only the rank order (the row kernel reads the
/// grouped rows itself).
pub(crate) fn combine_order_only(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    hip(gpu.moe_combine_order_top10(
        p.inverse_perm,
        p.topk_indices,
        p.topk_weights,
        p.down_expanded,
        grouped_rows,
        p.batch_size,
    ))
}

/// The row-folded shared down: the shared-down GEMM stores its BF16 rows into
/// the (otherwise unwritten) routed target, then one row kernel per token
/// combines the ten expert rows, folds the shared row, writes the HC streams
/// and runs the next read's norm + gate projection.  Bitwise
/// [`combine`] (`initial_zero`) + [`shared_down_hc`] + the next read's
/// `hyper_norm_gate` (F16 read output at `f16_row_pitch`).
fn shared_down_row_fold(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    hc: &HcSharedDown,
    next: &HcRowFoldNext,
) -> Result<(), DispatchError> {
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    // The next read's F16 row lives in the shared F16 scratch, which the
    // shared-down GEMM's own F16 input also uses (it is dead by the time the
    // row kernel writes): size it first so nothing reallocates under the GEMM.
    let ld16 = gpu.f16_row_pitch(4 * hc.hidden);
    let x16 = hip(gpu.qwen4_f16_x_scratch(hc.rows * ld16))?;
    hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(
        down.buf,
        shared.rotated,
        down.m,
        down.k,
        p.batch_size,
        target,
    ))?;
    hip(gpu.hc_row_fold_norm_gate(&rdna_compute::hc_row_fold::HcRowFold {
        grouped_down: p.y_down_grouped,
        order: p.down_expanded,
        shared_bf16: target,
        selector: shared.scalar,
        streams: &hc.streams,
        gates: &hc.gates,
        norm_weight: &next.norm_weight,
        gate_weight: &next.gate_weight,
        next_gates: &next.next_gates,
        normalized_f16: &x16,
        ld16,
        rows: hc.rows,
        hidden: hc.hidden,
    }))
}

/// Whether path 2 takes the symmetric IU4 arm (fn-moe-sym): the layer's
/// experts were verified symmetric at load (the policy says so), the recipe
/// keeps the BF16 boundaries the kernels implement, the activation is F32,
/// and the device admits it (`HIPFIRE_QWEN4_MOE_SYM_IU4`: default on for
/// gfx1151, `=1` on gfx1201; >= 512 rows, C2 producers). It replaces scatter,
/// gate/up, unscatter/rotation and down; the combine reads its BF16 rows like
/// the F16 WMMA arm's.
fn sym_iu4(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2
        && p
            .route_policy
            .is_some_and(|policy| policy.capability == MoeRouteCapability::Qt44Qt53GroupedSymmetric)
        && p.recipe.bf16_round_trip()
        && p.x_norm_batch.dtype == DType::F32
        && p.down_k == p.mi
        && gpu.qwen4_moe_sym_iu4_applies(p.batch_size)
}

/// Path-2 grouping (the stage runs on path 2 only).
pub(crate) fn scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
    if sym_iu4(gpu, p, true) {
        return hip(gpu.moe_scatter_stable_top10(
            p.topk_indices,
            p.expert_token_counts,
            p.expert_offsets,
            p.sorted_slot_index,
            p.expert_tile_ids,
            p.inverse_perm,
            total_slots,
            p.n_exp,
            grouped_rows,
        ));
    }
    hip(gpu.moe_scatter_fused_top10(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        total_slots,
        p.n_exp,
        grouped_rows,
        crate::families::moe::MOE_GROUPED_BLOCK_M,
    ))
}

/// Whether the grouped gate/up takes the F16 WMMA arm and hands the unscatter
/// BF16 rows: the recipe rounds gate/up to BF16 before SiLU anyway.  The
/// BF16-output entries exist in the gfx11 source only.
fn gateup_bf16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    p.recipe.bf16_round_trip()
        && gpu.arch_caps.has_wmma_w32()
        && gpu.qwen4_moe_gateup_wmma_applies(2 * p.mi, p.gate_up_k, p.batch_size)
}

/// Whether the grouped down takes the F16 WMMA arm: rotated straight to F16 in
/// `down`, output as BF16 for the combine, which rounds every row to BF16.
fn down_wmma(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    p.down_k == p.mi && gpu.qwen4_moe_down_wmma_applies(p.down_m, p.down_k, p.batch_size * p.k_top)
}

/// Whether gate/up stores the SwiGLU activation (BF16) straight from its
/// epilogue and `down` unscatters it fused with the 128-wide rotation.
fn gateup_silu(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    gateup_bf16(gpu, p) && down_wmma(gpu, p) && p.mi % 128 == 0
}

/// Whether this layer's routed experts live in host-mapped memory (spilled
/// past the VRAM budget); the symmetric IU4 GEMMs then take a wider expert-run
/// tile. A layer's experts share one residency, so expert 0 decides.
fn experts_host_mapped(p: &MoePrefillParams<'_>) -> bool {
    p.expert_stage_ptrs.is_none() && p.routed_experts.host_mapped()
}

pub(crate) fn gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let gate_up_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_gate_up_ptrs, |stage| stage.gate_up);
    if sym_iu4(gpu, p, use_path2) {
        let xq = hip(gpu.qwen4_moe_rotate256_i4(p.x_norm_batch, p.gate_up_k, p.batch_size))?;
        return hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
            experts_host_mapped(p),
        ));
    }
    if use_path2 && gateup_bf16(gpu, p) {
        let x_f16 = if gateup_rotates_f16(gpu, p, use_path2) {
            // The grouped kernel reads packed rows (x_row * K).
            Some(hip(gpu.rotate_x_mq_batched_f16(
                p.x_norm_batch,
                p.gate_up_k,
                p.batch_size,
                p.gate_up_k,
            ))?)
        } else {
            None
        };
        let gemm = if gateup_silu(gpu, p) {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_silu_bf16out
        } else {
            Gpu::gemm_mq4g256v2_moe_grouped_top10_bf16out
        };
        hip(gemm(
            gpu,
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            x_f16.as_ref().unwrap_or(p.x_rot_batch),
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else if use_path2 {
        hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
            gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            p.k_top,
            grouped_rows,
            p.batch_size,
        ))
    } else {
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            gate_up_ptrs,
            p.topk_indices,
            p.x_rot_batch,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            p.gate_up_k,
            p.batch_size,
        ))?;
        let active = p
            .batch_size
            .checked_mul(p.k_top)
            .and_then(|slots| slots.checked_mul(p.mi))
            .ok_or_else(|| DispatchError::Hip("grouped gate/up extent overflows".into()))?;
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
            hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))?;
        }
        Ok(())
    }
}

/// Path-2 unscatter fused with the SwiGLU activation: the grouped gate/up
/// rows go straight to the rotation input (`rot_batch`) with the recipe's BF16
/// round trips, bitwise the unfused unscatter -> round trip -> silu_mul ->
/// round trip sequence, rotated in the same launch when
/// [`unscatter_rotates`] (else [`activation`] rotates).
pub(crate) fn unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    // Symmetric IU4: SiLU in the gate/up epilogue, rotation + A4 in `down`.
    if sym_iu4(gpu, p, true) {
        return Ok(());
    }
    // SiLU in the gate/up epilogue, unscatter + rotation in the down stage.
    if gateup_silu(gpu, p) {
        return Ok(());
    }
    if gateup_bf16(gpu, p) {
        return hip(gpu.moe_gate_up_unscatter_silu_top10_bf16in(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
        ));
    }
    if unscatter_rotates(gpu, p) {
        // The shared expert's activation rides in the same launch; its stage
        // skipped it (the shared down reads it after the combine).
        let shared = p
            .prelude
            .shared
            .as_ref()
            .filter(|_| shared_in_unscatter(gpu, p))
            .map(|shared| {
                let live = p.batch_size * shared.intermediate;
                (
                    f32_view(shared.gate_out, 0, live),
                    f32_view(shared.up_out, 0, live),
                    f32_view(shared.rotated, 0, live),
                    shared.scalar,
                )
            });
        let shared = shared
            .as_ref()
            .map(|(gate, up, out, selector)| SharedExpertActivation {
                gate,
                up,
                out,
                selector,
                selectors: p.batch_size,
            });
        return hip(gpu.moe_gate_up_unscatter_silu_rotate128_top10(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.rot_batch,
            p.mi,
            grouped_rows,
            p.recipe.bf16_round_trip(),
            shared.as_ref(),
        ));
    }
    hip(gpu.moe_gate_up_unscatter_silu_top10(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.rot_batch,
        p.mi,
        grouped_rows,
        p.recipe.bf16_round_trip(),
    ))
}

/// Whether path 2's F32 unscatter also applies the down's 128-wide rotation
/// (in the same launch), leaving [`activation`] nothing to do.
fn unscatter_rotates(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    !sym_iu4(gpu, p, true)
        && !gateup_bf16(gpu, p)
        && !down_wmma(gpu, p)
        && p.mi.is_multiple_of(128)
}

/// Whether path 2's fused unscatter launch also runs the BF16 shared expert
/// activation: its only reader, a natural-basis shared down, runs after the
/// combine (so after the unscatter).
fn shared_in_unscatter(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    unscatter_rotates(gpu, p)
        && p.recipe.bf16_round_trip()
        && p.recipe.shared_after_combine()
        && p.prelude
            .shared
            .as_ref()
            .is_some_and(|shared| shared.weights.down.dtype != DType::MQ4G128V2)
}

pub(crate) fn activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
    // Path 2's unscatter already produced the activation.
    if !use_path2 {
        hip(gpu.silu_mul_f32(p.gate_batch, p.up_batch, p.rot_batch))?;
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(p.rot_batch))?;
        }
    }
    // The F16 WMMA and symmetric IU4 downs rotate the grouped rows themselves
    // (see `down`); the F32 unscatter may already have rotated.
    if use_path2 && (sym_iu4(gpu, p, use_path2) || down_wmma(gpu, p) || unscatter_rotates(gpu, p)) {
        return Ok(());
    }
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))
}

/// A few-row forward (speculative verify) on grouped path 2 without the WMMA
/// down: the route slots are nearly all distinct experts, so the grouped down
/// tile (16 slot rows per expert) runs almost empty; the decode kernel's
/// per-slot down over the unscattered, rotated rows streams the experts
/// faster.  Its per-slot dot is the grouped kernel's (bitwise), combined in
/// slot order.
fn indexed_down(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool {
    use_path2 && p.batch_size <= 8 && !down_wmma(gpu, p)
}

/// Release the stage immediately after the routed down, before combine.
fn release_stage(gpu: &Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if let Some(stage) = p.expert_stage_ptrs {
        hip(gpu.hip.event_record(stage.free, gpu.active_stream.as_ref()))?;
    }
    Ok(())
}

pub(crate) fn down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let down_ptrs = p
        .expert_stage_ptrs
        .map_or(p.expert_down_ptrs, |stage| stage.down);
    let total_slots = p.batch_size * p.k_top;
    if sym_iu4(gpu, p, use_path2) {
        // Grouped BF16 SwiGLU rows -> FWHT128 -> compact flat-slot A4 ->
        // grouped IU4 down, BF16 per grouped row for the BF16-input combine.
        let xq = hip(gpu.qwen4_moe_rotate128_i4(
            p.y_gate_up_grouped,
            p.sorted_slot_index,
            p.down_k,
            grouped_rows,
            total_slots,
        ))?;
        hip(gpu.gemm_qwen4_moe_down_iu4_sym(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &xq,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            experts_host_mapped(p),
        ))?;
        return release_stage(gpu, p);
    }
    if indexed_down(gpu, p, use_path2) {
        return down(gpu, p, false, grouped_rows);
    }
    if use_path2 && down_wmma(gpu, p) {
        // Rotation and GEMM in one stage: the rotated F16 rows live in the
        // shared FP16 scratch, which another stage's GEMM would overwrite.
        let x_f16 = if gateup_silu(gpu, p) {
            hip(gpu.moe_unscatter_rotate128_f16(
                p.y_gate_up_grouped,
                p.sorted_slot_index,
                p.mi,
                grouped_rows,
                total_slots,
            ))?
        } else {
            hip(gpu.rotate_x_mq_128_v2_f16(p.rot_batch, p.mi, total_slots))?
        };
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            &x_f16,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            grouped_rows,
            true,
        ))?;
    } else if use_path2 {
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10(
            down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            p.n_exp,
        ))?;
        // A BF16-source recipe rounds each grouped expert output before
        // weighted combination, just as on the scalar route.  The grouped
        // combine kernel performs that same RNE rounding on every row it
        // reads, so a separate pass over all (padded) grouped rows would only
        // re-round values that are already BF16-exact.
    } else {
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.down_m,
            p.down_k,
            p.batch_size,
            p.n_exp,
        ))?;
        // The combine rounds every expert output it reads to BF16 itself.
    }
    release_stage(gpu, p)
}

/// Whether [`combine`] can start from +0.0 instead of the zero-filled target
/// (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`): the BF16-row combine of path 2, with
/// nothing writing the target between its zero fill and the combine (the
/// shared down, the only other writer, runs after it).
pub(crate) fn combine_initial_zero_applies(
    gpu: &Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
) -> bool {
    use_path2
        && !indexed_down(gpu, p, use_path2)
        && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2))
        && p.recipe.shared_after_combine()
}

/// `initial_zero`: the combine writes the target from +0.0 rather than
/// accumulating into zeros the caller filled (requires
/// [`combine_initial_zero_applies`]).
pub(crate) fn combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
    initial_zero: bool,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if initial_zero && !combine_initial_zero_applies(gpu, p, use_path2) {
        return Err(DispatchError::Hip(
            "qt44 combine: zero-initialized target requested off the BF16-row path".into(),
        ));
    }
    if indexed_down(gpu, p, use_path2) {
        return combine(gpu, p, false, grouped_rows, false);
    }
    if use_path2 && (down_wmma(gpu, p) || sym_iu4(gpu, p, use_path2)) {
        // Both downs store BF16 grouped rows. The rank order goes to
        // `down_expanded`: unused on this route until the shared down, which
        // runs after the combine.
        hip(gpu.moe_down_combine_grouped_top10_bf16in(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_expanded,
            p.down_m,
            grouped_rows,
            p.batch_size,
            initial_zero,
        ))?;
    } else if use_path2 {
        hip(gpu.moe_down_combine_grouped_top10(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            grouped_rows,
            p.batch_size,
        ))?;
    } else {
        let expanded = f32_view(p.down_expanded, 0, p.batch_size * p.k_top * p.down_m);
        hip(gpu.moe_down_combine_top10_batched(
            &expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            p.batch_size,
        ))?;
    }
    // A shared down placed after the combine rounds `target` to BF16 as its
    // first read, so the round trip is owed only when nothing follows.
    if p.recipe.bf16_round_trip() && !p.recipe.shared_after_combine() {
        hip(gpu.bf16_round_trip_f32(target))?;
    }
    Ok(())
}
