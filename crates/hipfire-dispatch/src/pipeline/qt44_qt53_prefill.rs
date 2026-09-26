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
use crate::families::moe::MoePrefillParams;
use crate::types::DispatchError;
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
    batch_projection(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
    )
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
    if [&weights.selector, &weights.gate, &weights.up]
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
            DType::BF16 | DType::F32 => p.x_norm_batch,
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
    if p.recipe.bf16_round_trip() {
        // gate/up are round-tripped where the activation reads them (their
        // only reader), in the same pass.
        hip(gpu.bf16_round_trip_f32(p.prelude.router_logits))?;
        hip(gpu.bf16_round_trip_f32(shared.scalar))?;
    }
    Ok(())
}

/// Apply the shared expert activation after the selector and gate/up projections.
pub(crate) fn shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("grouped prefill shared weights missing".into()))?;
    let scalar = f32_view(shared.scalar, 0, p.batch_size);
    if p.recipe.bf16_round_trip() {
        #[cfg(feature = "deltanet")]
        hip(gpu.sigmoid_f32(&scalar))?;
        #[cfg(not(feature = "deltanet"))]
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "grouped-shared-sigmoid-requires-deltanet",
            arch: "",
            quant: "",
        });
        hip(gpu.bf16_round_trip_f32(&scalar))?;
    }
    // The live rows only: the shared buffers are sized for the chunk cap.
    let live = p.batch_size * shared.intermediate;
    let (gate, up, rotated) = (
        f32_view(shared.gate_out, 0, live),
        f32_view(shared.up_out, 0, live),
        f32_view(shared.rotated, 0, live),
    );
    if p.recipe.bf16_round_trip() {
        hip(gpu.silu_mul_bf16_rt_f32(&gate, &up, &rotated))?;
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
    let out = f32_view(p.down_expanded, 0, p.batch_size * p.down_m);
    match down.dtype {
        DType::BF16 | DType::F32 | DType::MQ4G256V2 | DType::MQ4G128V2 => {
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

pub(crate) fn scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
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
/// BF16 rows: the recipe rounds gate/up to BF16 before SiLU anyway.
fn gateup_bf16(gpu: &Gpu, p: &MoePrefillParams<'_>) -> bool {
    p.recipe.bf16_round_trip()
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

pub(crate) fn gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if use_path2 && gateup_bf16(gpu, p) {
        let x_f16 = if gateup_rotates_f16(gpu, p, use_path2) {
            Some(hip(gpu.rotate_x_mq_batched_f16(
                p.x_norm_batch,
                p.gate_up_k,
                p.batch_size,
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
            p.expert_gate_up_ptrs,
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
            p.expert_gate_up_ptrs,
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
            p.expert_gate_up_ptrs,
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
/// round trip sequence.  [`activation`] then only rotates.
pub(crate) fn unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
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
    hip(gpu.moe_gate_up_unscatter_silu_top10(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.rot_batch,
        p.mi,
        grouped_rows,
        p.recipe.bf16_round_trip(),
    ))
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
    // The F16 WMMA down rotates straight to F16 itself (see `down`).
    if use_path2 && down_wmma(gpu, p) {
        return Ok(());
    }
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))
}

pub(crate) fn down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * p.k_top;
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
            p.expert_down_ptrs,
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
            p.expert_down_ptrs,
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
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.down_m,
            p.down_k,
            p.batch_size,
            p.n_exp,
        ))?;
        let expanded = f32_view(p.down_expanded, 0, total_slots * p.down_m);
        if p.recipe.bf16_round_trip() {
            hip(gpu.bf16_round_trip_f32(&expanded))?;
        }
    }
    Ok(())
}

pub(crate) fn combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if use_path2 && down_wmma(gpu, p) {
        // The rank order goes to `down_expanded`: unused on this route until
        // the shared down, which runs after the combine.
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
