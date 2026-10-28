// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Operation contracts for stateful layer execution.
//!
//! The contracts in this module describe tensor roles, extents, and ordering;
//! they do not describe a model family.  Architecture crates bind their
//! resident weights, state, and fixed-capacity scratch to these borrowed views.
//! [`crate::pipeline::steps::validate_steps`] preflights these composite
//! operations and sealed MoE calls before unrelated architecture effects.
//! Existing scalar `Step` variants retain their established launch-time
//! validation; this module does not broaden that legacy contract.
//! No operation owns a tensor or allocates execution-local storage.

use crate::context::DispatchCtx;
use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use crate::types::KernelKey;
use rdna_compute::tensor_ops::{
    argmax_f32, bf16_roundtrip_f32, gated_delta_chunk_route, gated_delta_conv_params,
    gated_delta_conv_params_batched, gated_delta_conv_params_qknorm_batched,
    gated_delta_conv_qknorm_route, gated_delta_gate_batched, gated_delta_gate_batched_rotate,
    gated_delta_step_batched, gated_delta_step_gate_wmma, gated_delta_step_gate_wmma_qknormed,
    gated_delta_step_gated, hc_activation_fused_f32, hc_state_bf16_add_f32, hc_state_bf16_to_f32,
    hyper_norm, hyper_norm_f16, hyper_norm_gate, hyper_norm_gate_outputs, hyper_read_projected,
    hyper_read_up_fused, hyper_read_up_wmma, hyper_write, hyper_write_norm,
    indexed_attention_attention, indexed_attention_attention_batch_with_arena,
    indexed_attention_cache_append_batch, indexed_attention_decode_prologue,
    indexed_attention_index_key_append_batch, indexed_attention_norm_rope_batch,
    indexed_attention_pool_rope_incremental, indexed_attention_reuse_selection,
    indexed_attention_select_batch_mirrored, scale_f32, ArgmaxF32, Bf16Roundtrip, GatedDeltaConv,
    GatedDeltaConvBatched, GatedDeltaGate, GatedDeltaGateBatched, GatedDeltaParams,
    GatedDeltaParamsBatched, GatedDeltaStep, GatedDeltaStepBatched, GdnStateFormat,
    HcActivationFused, HyperNextGates, HyperNorm, HyperNormGate, HyperNormGateOutputs,
    HyperReadProjected, HyperReadUpFused, HyperWrite, IndexedAttentionAttention,
    IndexedAttentionAttentionBatch, IndexedAttentionCacheAppendBatch,
    IndexedAttentionDecodePrologue, IndexedAttentionIndexKeyAppendBatch,
    IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope, IndexedAttentionReuseSelection,
    IndexedAttentionSelectBatch, QsaF16ArenaRef, QsaKvFormat, ScaleF32,
};
use rdna_compute::tensor_ops::{
    gated_delta_conv_params_batched_mixed, gated_delta_conv_params_qknorm_batched_mixed,
    gated_delta_core_lanes, gated_delta_gate_batched_arena, gated_delta_gate_batched_rotate_arena,
    gated_delta_step_gate_wmma_arena, gated_delta_step_gate_wmma_qknormed_arena,
    indexed_attention_cache_append_batch_arena, indexed_attention_index_key_append_batch_arena,
    indexed_attention_norm_rope_batch_arena, indexed_attention_rope_table, Bf16Arena, GdnCoreLane,
};
use rdna_compute::{
    DType, Gpu, GpuTensor, TrunkFamily, QSA_NORM_EPI_HEAD_DIM, QSA_NORM_EPI_ROTARY_DIM,
};
use smallvec::SmallVec;
use std::sync::LazyLock;

#[inline]
pub(super) fn hip<T>(result: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    result.map_err(|error| DispatchError::Hip(error.to_string()))
}

#[inline]
pub(super) fn checked_mul(a: usize, b: usize, label: &'static str) -> Result<usize, DispatchError> {
    a.checked_mul(b)
        .ok_or_else(|| DispatchError::Hip(format!("{label} extent overflows")))
}

pub(super) fn require_tensor(
    tensor: &GpuTensor,
    elements: usize,
    dtype: DType,
    label: &'static str,
) -> Result<(), DispatchError> {
    if tensor.dtype != dtype || tensor.numel() < elements {
        return Err(DispatchError::Hip(format!(
            "{label} requires {elements} elements of {dtype:?}, got {:?} with {} elements",
            tensor.dtype,
            tensor.numel()
        )));
    }
    Ok(())
}

/// Payload types the shared stateful-op projections can consume: source BF16,
/// or a Qwen4 matrix quantization whose FWHT basis the caller rotates into the
/// op's rotation scratch before the projection runs.
fn projection_weight_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::BF16
            | DType::Q8_0
            | DType::MQ4G256V2
            | DType::MQ4G128V2
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MFP4G32E8SOA
    )
}

pub(super) fn require_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    label: &'static str,
) -> Result<(), DispatchError> {
    if !projection_weight_dtype(weight.dtype) || weight.m != m || weight.k != k {
        return Err(DispatchError::Hip(format!(
            "{label} has incompatible shape or dtype"
        )));
    }
    // Packed formats are sized by their own group geometry, which the artifact
    // boundary validates at admission; only the native layout is an element
    // count that this layer can check.
    if weight.dtype == DType::BF16 {
        let elements = checked_mul(m, k, "weight elements")?;
        if weight.buf.numel() < elements {
            return Err(DispatchError::Hip(format!(
                "{label} has incompatible shape or dtype"
            )));
        }
    }
    Ok(())
}

#[inline]
pub(super) fn view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

/// Weight projection used by the stateful operations.  Multi-row calls use the
/// batched GEMM launcher and decode uses the GEMV launcher, for whichever of
/// the admissible payloads the tensor carries.
///
/// The FWHT matrix tiers (MQ4G256V2 / MQ4G128V2 / MQ6G256V2) speak a rotated
/// basis, not the natural input, so the activation is rotated into `rotation`
/// (`rows * weight.k` elements) first — the same basis the routed experts and the
/// MTP head consume.  BF16 and Q8F16 payloads read the input as it stands.
///
/// gfx942 BF16 weights (every row count, one row included) take the BF16 MFMA
/// GEMM (`KernelKey::GemmBf16Mfma`, `Gpu::gemm_bf16_mfma_gfx942`) on a BF16
/// copy of the activation instead of the GEMV / multirow GEMM arms; see
/// [`g942_trunk_mfma_route`]. `HIPFIRE_G942_TRUNK_GEMV=1` restores the one-row
/// GEMV arm (rows > 1 stay on MFMA).
pub fn project_weight(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    output: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    project_weights(gpu, input, rows, rotation, &[(weight, output)])
}

/// [`project_weight`] for several weights reading the same `input`.  On the
/// gfx1151 MQ6 BT8 and MQ4 prefill routes the input is rotated straight to F16
/// once per K and every MQ6 / MQ4 GEMM reads that copy (the per-weight path
/// rotates to F32 and converts to F16 again for each); the bytes each GEMM sees
/// are unchanged.
///
/// Few-row verify (2..=`Gpu::few_row_limit` route rows: 8, or the verify
/// scope's admitted exact cap): the MQ4G256V2 members of one rotated input run
/// through `Gpu::gemv_mq4g256v2_rows_x4`, four per launch (a lone member takes
/// it too, via `project_rotated`), each output bitwise the AR `gemv_mq4g256v2`;
/// the MQ6G256V2 members take `gemm_mq6g256v2_f32_rows_x4` the same way. One row
/// keeps the single-matrix GEMVs; more route rows than the limit keep the GEMMs.
pub fn project_weights(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
) -> Result<(), DispatchError> {
    project_weights_with_epilogues(gpu, input, rows, rotation, projections, &[])
}

/// [`project_weights`] whose MQ6 int8 GEMMs may run the fused QSA norm+RoPE
/// epilogue: `epilogues` is empty or parallel to `projections`, and a member
/// that took the fused entry sets its [`NormRopeEpi::fused`].
fn project_weights_with_epilogues(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    epilogues: &[Option<&NormRopeEpi<'_>>],
) -> Result<(), DispatchError> {
    // The chunked GDN in-projection group: BF16 qkv ahead of F32 a/b/z.
    let members = [false, true, true, true];
    let gdn_regions = projections.len() == 4 && projections[0].1.dtype == DType::BF16;
    project_weights_inner(
        gpu,
        input,
        rows,
        rotation,
        projections,
        gdn_regions.then_some(&members[..]),
        epilogues,
    )
}

/// A QSA q / k projection's fused RMSNorm + RoPE epilogue request
/// ([`Gpu::gemm_mq6g256v2_i8_prequant_norm_rope`]): the layer's `norm` weight,
/// the head pitch in output rows (`512` for the interleaved q-gate rows,
/// `256` for plain k rows) and the RoPE `table`. `fused` records whether the
/// projection ran fused.
struct NormRopeEpi<'a> {
    norm: &'a GpuTensor,
    head_stride_rows: usize,
    table: &'a GpuTensor,
    fused: std::cell::Cell<bool>,
}

/// Whether `weight`'s prefill GEMM of `rows` runs the gfx1201 native-FP8 MQ6
/// route ([`Gpu::gemm_mq6g256v2_fp8_applies`]) on an F32 activation: an MQ6G256V2
/// weight outside few-row and verify forwards (`route_rows <= few_row_limit`,
/// the row-exact wide verify included), which keep their decode-twin launches.
/// Producers of the projection's input ([`GatedDeltaNetOp`]'s gate output) read
/// it to keep the activation F32 rather than BF16, which the FP8 pack cannot read.
fn mq6_fp8_route(gpu: &Gpu, weight: &WeightRef<'_>, rows: usize) -> bool {
    weight.dtype == DType::MQ6G256V2
        && gpu.route_rows(rows) > gpu.few_row_limit()
        && gpu.gemm_mq6g256v2_fp8_applies(weight.m, weight.k, rows)
}

/// [`mq6_fp8_route`] for the exact-INT8 MQ6 route
/// ([`Gpu::gemm_mq6g256v2_i8_applies`], gfx1201 / gfx1151), checked before it:
/// the same few-row / verify exclusion and the same F32-activation requirement.
fn mq6_i8_route(gpu: &Gpu, weight: &WeightRef<'_>, rows: usize) -> bool {
    weight.dtype == DType::MQ6G256V2
        && gpu.route_rows(rows) > gpu.few_row_limit()
        && gpu.gemm_mq6g256v2_i8_applies(weight.m, weight.k, rows)
}

/// Only an admitted eager verify may extend the singleton routes past eight.
/// Keep this tied to the actual scoped limit, not the executor's storage cap.
pub(super) fn wide_exact_route(route: usize, limit: usize) -> bool {
    route > rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS && route <= limit
}

pub(super) fn wide_exact_rows(gpu: &Gpu, rows: usize) -> bool {
    wide_exact_route(gpu.route_rows(rows), gpu.few_row_limit())
}

fn wide_projection_uses_singletons(dtype: DType, route: usize, limit: usize) -> bool {
    wide_exact_route(route, limit)
        && matches!(
            dtype,
            DType::MQ2G256V2 | DType::MQ3G256V2 | DType::MQ5G256V2
        )
}

/// [`project_weights`] with the MQ6 a/b/z row-region fold (three F32 outputs
/// of shared-rotation MQ6 siblings, one launch) enabled for exactly the
/// projections `region_members` flags (parallel to `projections`) instead of
/// inferred from the group's shape: [`project_trunk`] flags the original GDN
/// a/b/z of a group whose siblings were taken by another route, so qkv can
/// never join the fold.
fn project_weights_inner(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    region_members: Option<&[bool]>,
    epilogues: &[Option<&NormRopeEpi<'_>>],
) -> Result<(), DispatchError> {
    debug_assert!(epilogues.is_empty() || epilogues.len() == projections.len());
    debug_assert!(region_members.is_none_or(|m| m.len() == projections.len()));
    let route = gpu.route_rows(rows);
    // A verify forward row-exact past the historic few-row limit keeps the
    // few-row decode-twin launches below; the shared-F16-rotation WMMA routes
    // are not row-exact, so they decline there (and nowhere else: outside a
    // verify forward `few_row_limit()` is the historic limit).
    let wide_exact = wide_exact_route(route, gpu.few_row_limit());
    let shared = |w: &WeightRef<'_>, gpu: &Gpu| {
        !wide_exact
            && match w.dtype {
                DType::MQ6G256V2 => gpu.gemm_mq6g256v2_xf16_applies(w.k, rows),
                // gfx1151, more than 8 rows (N >= 96): PM or exact bt4 on the shared F16 rotation.
                DType::MQ4G256V2 => gpu.gemm_mq4g256v2_xf16_applies(w.k, rows),
                _ => false,
            }
    };
    // Shared-rotation weights first, grouped by K: other projections reuse
    // the same F16 scratch and would overwrite the rotated copy.
    let mut done = vec![false; projections.len()];
    // gfx1201 qwen4 prefill: MQ6G256V2 members of one K share one FWHT + FP8
    // activation pack (`mq6_fp8_prepare_x`) and each runs the native-FP8 GEMM
    // (BF16 or F32 output, as the caller allocated it).  Before every other
    // route: the packed planes live in shared FP8 scratch, so all MQ6 members
    // launch before anything can repack it.  Verify and few-row forwards
    // (`route <= few_row_limit`, incl. the row-exact wide verify) never take
    // it; MQ4 members left in the group keep the shared F16 rotation below.
    if input.dtype == DType::F32 {
        // The exact-INT8 route (gfx1201 / gfx1151) goes first: one FWHT + A8
        // block quantization per K feeds every MQ6 member, output BF16 or F32.
        for i in 0..projections.len() {
            let (weight, _) = projections[i];
            if done[i] || !mq6_i8_route(gpu, weight, rows) {
                continue;
            }
            let xq = hip(gpu.mq6_i8_prepare_x(input, weight.k, rows))?;
            for j in i..projections.len() {
                let (w, out) = projections[j];
                if !done[j] && w.k == weight.k && mq6_i8_route(gpu, w, rows) {
                    let epilogue = epilogues.get(j).copied().flatten();
                    let fused = match epilogue {
                        Some(e) => hip(gpu.gemm_mq6g256v2_i8_prequant_norm_rope(
                            w.buf,
                            xq,
                            out,
                            w.m,
                            w.k,
                            rows,
                            e.head_stride_rows,
                            e.norm,
                            e.table,
                        ))?,
                        None => false,
                    };
                    if let Some(e) = epilogue {
                        e.fused.set(fused);
                    }
                    if !fused {
                        hip(gpu.gemm_mq6g256v2_i8_prequant(w.buf, xq, out, w.m, w.k, rows))?;
                    }
                    done[j] = true;
                }
            }
        }
        for i in 0..projections.len() {
            let (weight, _) = projections[i];
            if done[i] || !mq6_fp8_route(gpu, weight, rows) {
                continue;
            }
            let prepared = hip(gpu.mq6_fp8_prepare_x(input, weight.k, rows))?;
            for j in i..projections.len() {
                let (w, out) = projections[j];
                if !done[j] && w.k == weight.k && mq6_fp8_route(gpu, w, rows) {
                    hip(gpu.gemm_mq6g256v2_fp8_prepared(w.buf, &prepared, out, w.m, w.k, rows))?;
                    done[j] = true;
                }
            }
        }
    }
    for i in 0..projections.len() {
        let (weight, _) = projections[i];
        if done[i] || !shared(weight, gpu) {
            continue;
        }
        let x_f16 = hip(gpu.rotate_x_mq_batched_f16(input, weight.k, rows))?;
        // The chunked GDN qkv output is BF16; its three F32 a/b/z siblings
        // may share one row-region launch without sharing their accumulators.
        if let Some(members) = region_members {
            for fold in [DType::MQ6G256V2, DType::MQ4G256V2] {
                // The MQ6 fold has its own switch; the MQ4 one is the PM module's.
                if fold == DType::MQ6G256V2 && !gpu.qwen4_mq6_x4_regions() {
                    continue;
                }
                let mut indices = [0usize; 3];
                let mut count = 0;
                for j in i..projections.len() {
                    let (w, out) = projections[j];
                    if members[j]
                        && !done[j]
                        && w.dtype == fold
                        && w.k == weight.k
                        && shared(w, gpu)
                        && out.dtype == DType::F32
                    {
                        if count < 3 {
                            indices[count] = j;
                        }
                        count += 1;
                    }
                }
                if count == 3 {
                    let regions = indices.map(|j| {
                        let (w, out) = projections[j];
                        (w.buf, out, w.m)
                    });
                    let folded = if fold == DType::MQ4G256V2 {
                        gpu.gemm_mq4g256v2_xf16_regions(&regions, &x_f16, weight.k, rows)
                    } else {
                        gpu.gemm_mq6g256v2_xf16_regions(&regions, &x_f16, weight.k, rows)
                    };
                    if hip(folded)? {
                        for j in indices {
                            done[j] = true;
                        }
                    }
                }
            }
        }
        for j in i..projections.len() {
            let (w, out) = projections[j];
            if !done[j] && w.k == weight.k && shared(w, gpu) {
                hip(if w.dtype == DType::MQ4G256V2 {
                    gpu.gemm_mq4g256v2_xf16(w.buf, &x_f16, out, w.m, w.k, rows)
                } else {
                    gpu.gemm_mq6g256v2_xf16(w.buf, &x_f16, out, w.m, w.k, rows)
                })?;
                done[j] = true;
            }
        }
    }
    // gfx942 BF16 weights of one K: the input is staged to BF16 once (or used
    // as is when it already is BF16) and every member runs the MFMA GEMM, one
    // output row-tile = 32 batch rows with the kernel masking the tail, so
    // each output row is independent of `rows` (one row included). Ahead of
    // the BF16-activation refusal below: a BF16 input is a valid MFMA operand.
    let g942 = gpu.arch_caps.is_gfx942();
    if g942 {
        let gemv_opt_out = g942_trunk_gemv();
        let route = |j: usize| {
            g942_trunk_mfma_route(
                true,
                gemv_opt_out,
                rows,
                input.dtype,
                projections[j].0.dtype,
                projections[j].1.dtype,
            )
        };
        for i in 0..projections.len() {
            if done[i] || !route(i) {
                continue;
            }
            let k = projections[i].0.k;
            let group: SmallVec<[usize; 4]> = (i..projections.len())
                .filter(|&j| !done[j] && projections[j].0.k == k && route(j))
                .collect();
            project_g942_mfma(gpu, input, rows, rotation, projections, &group, k)?;
            for j in group {
                done[j] = true;
            }
        }
    }
    // Only the shared F16 rotation reads a BF16 activation.
    if input.dtype != DType::F32 && done.iter().any(|d| !d) {
        return Err(DispatchError::UnsupportedVariant {
            family: "layer-operations",
            variant: "non-f32-activation",
            arch: "",
            quant: "unsupported",
        });
    }
    // BF16 weights on the F16 WMMA route: one F16 conversion per K.
    for i in 0..projections.len() {
        let (weight, _) = projections[i];
        if wide_exact || done[i] || weight.dtype != DType::BF16 {
            continue;
        }
        let group: SmallVec<[(&GpuTensor, &GpuTensor, usize); 4]> = projections[i..]
            .iter()
            .filter(|(w, _)| w.dtype == DType::BF16 && w.k == weight.k)
            .map(|(w, out)| (w.buf, *out, w.m))
            .collect();
        if hip(gpu.gemm_bf16_xf32_f16_wmma_qwen4(&group, input, weight.k, rows))? {
            for j in i..projections.len() {
                let w = projections[j].0;
                if w.dtype == DType::BF16 && w.k == weight.k {
                    done[j] = true;
                }
            }
        }
    }
    // Quantized weights of one FWHT basis and K read the same rotated input:
    // rotate it once for the whole group instead of once per weight.
    // Few rows (speculative verify) on gfx11+: BF16 weights of one K share
    // multi-row launches, four matrices at a time, each weight read once.
    if (2..=gpu.few_row_limit()).contains(&route) && gpu.arch_caps.has_gfx11_plus_simt() {
        for i in 0..projections.len() {
            let (weight, _) = projections[i];
            if done[i] || weight.dtype != DType::BF16 {
                continue;
            }
            let group: SmallVec<[usize; 4]> = (i..projections.len())
                .filter(|&j| {
                    !done[j]
                        && projections[j].0.dtype == DType::BF16
                        && projections[j].0.k == weight.k
                })
                .collect();
            for chunk in group.chunks(4) {
                let part = |n: usize| {
                    let (w, y) = projections[chunk[n.min(chunk.len() - 1)]];
                    (w.buf, y, if n < chunk.len() { w.m } else { 0 })
                };
                hip(gpu.gemv_bf16_xf32_x4_rows(
                    [part(0), part(1), part(2), part(3)],
                    input,
                    weight.k,
                    rows,
                ))?;
                for &j in chunk {
                    done[j] = true;
                }
            }
        }
    }
    for i in 0..projections.len() {
        if done[i] {
            continue;
        }
        let (weight, _) = projections[i];
        let rotated = rotate_input(gpu, weight, input, rows, rotation)?;
        let x = rotated.as_ref().unwrap_or(input);
        let mut group: SmallVec<[usize; 4]> = SmallVec::new();
        for j in i..projections.len() {
            let (w, _) = projections[j];
            let same_input = j == i
                || (rotated.is_some()
                    && w.k == weight.k
                    && rotation_basis(w.dtype) == rotation_basis(weight.dtype));
            if !done[j] && same_input {
                group.push(j);
                done[j] = true;
            }
        }
        // Decode and few-row verify: two or more MQ6 GEMVs of the group share
        // one launch, four at a time; each row is the single-matrix kernel's.
        let mut mq6: SmallVec<[usize; 4]> = group
            .iter()
            .copied()
            .filter(|&j| route <= gpu.few_row_limit() && projections[j].0.dtype == DType::MQ6G256V2)
            .collect();
        if mq6.len() < 2 {
            mq6.clear();
        }
        for chunk in mq6.chunks(4) {
            let part = |n: usize| {
                let (w, y) = projections[chunk[n.min(chunk.len() - 1)]];
                (w.buf, y, if n < chunk.len() { w.m } else { 0 })
            };
            let parts = [part(0), part(1), part(2), part(3)];
            hip(if rows == 1 {
                gpu.gemv_mq6g256v2_x4(parts, x, weight.k)
            } else {
                gpu.gemm_mq6g256v2_f32_rows_x4(parts, x, weight.k, rows)
            })?;
        }
        // Few-row verify (2..=`few_row_limit` rows): the group's MQ4G256V2 members share
        // `gemv_mq4g256v2_rows_x4` launches, four at a time (a lone member
        // takes the same launch with empty slots); each output is bitwise the
        // AR singleton `gemv_mq4g256v2`. One row keeps that singleton.
        let mq4: SmallVec<[usize; 4]> = group
            .iter()
            .copied()
            .filter(|&j| {
                (2..=gpu.few_row_limit()).contains(&route)
                    && projections[j].0.dtype == DType::MQ4G256V2
            })
            .collect();
        for chunk in mq4.chunks(4) {
            let part = |n: usize| {
                let (w, y) = projections[chunk[n.min(chunk.len() - 1)]];
                (w.buf, y, if n < chunk.len() { w.m } else { 0 })
            };
            let parts = [part(0), part(1), part(2), part(3)];
            hip(gpu.gemv_mq4g256v2_rows_x4(parts, x, weight.k, rows))?;
        }
        for &j in group
            .iter()
            .filter(|j| !mq6.contains(j) && !mq4.contains(j))
        {
            let (w, out) = projections[j];
            project_rotated(gpu, w, x, out, rows)?;
        }
    }
    Ok(())
}

/// GDN Z|beta|alpha fold of [`project_trunk`]: the folded rows and the
/// indices of Z, beta (`in_proj_b`) and alpha (`in_proj_a`) in its
/// projection list.
struct ZbaFold<'a> {
    rows: &'a GpuTensor,
    z: usize,
    beta: usize,
    alpha: usize,
    padded_outputs: Option<[&'a GpuTensor; 3]>,
}

/// Qwen4 trunk projections (GDN in/out, QSA in/out) of one shared input.
///
/// `a4[i]` says the per-layer mask (`HIPFIRE_QWEN4_TRUNK_IU4`) selected
/// `projections[i]`. Each projection takes one of two routes, run as
/// groups that each finish before the next starts (the A4 sidecar, the
/// rotation scratch and the shared F16 X scratch are reused group to group):
///
/// 1. Symmetric MQ4G256V2 and masked, on a verified trunk
///    ([`Gpu::qwen4_trunk_iu4_applies`]): the dense IU4 route. One A4
///    producer per shared input (`rotate_x_mq_i4`, FWHT + `block_i4_128`, no
///    F32 store) feeds SET GEMMs (`gemm_mq4g256v2_mmq_set_prequant_iu4`: Halo
///    V2B / `pm_v2b` where eligible, X5 / symfold otherwise); a GDN `zba`
///    fold, when all of Z, beta and alpha are in this group, runs
///    Z|beta|alpha as one V2B SET with the split in its epilogue.
/// 2. Everything else: [`project_weights`]. That includes every other MQ4G256V2
///    projection of a prefill input (more than 8 rows, gfx1151): the shared F16
///    rotation per K with the exact PM / bt4 GEMMs (see `project_weights_inner`).
///    When siblings were split off a GDN in-projection
///    group (`gdn_in_proj`), the MQ6 / MQ4 row-region fold stays enabled for the
///    group's original a/b/z only (never qkv).
#[allow(clippy::too_many_arguments)]
fn project_trunk(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: &GpuTensor,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    a4: &[bool],
    zba: Option<ZbaFold<'_>>,
    gdn_in_proj: bool,
) -> Result<(), DispatchError> {
    project_trunk_masked(
        gpu,
        input,
        rows,
        rotation,
        projections,
        a4,
        zba,
        gdn_in_proj,
        &[],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn project_trunk_masked(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: &GpuTensor,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    a4: &[bool],
    zba: Option<ZbaFold<'_>>,
    gdn_in_proj: bool,
    covered: &[bool],
    epilogues: &[Option<&NormRopeEpi<'_>>],
) -> Result<(), DispatchError> {
    debug_assert!(epilogues.is_empty() || epilogues.len() == projections.len());
    debug_assert_eq!(a4.len(), projections.len());
    debug_assert!(covered.is_empty() || covered.len() == projections.len());
    // Decode's MQ6 qkv and MQ4 a/b/z already share one aligned-256 FWHT.
    // Keep qkv first; only replace the three independent MQ4 launches.
    // A stored A4 mask / ZBA fold may be present at decode, but neither
    // per-call IU4 route applies to this one-row F32 projection group.
    if gdn_in_proj
        && projections.len() == 4
        && covered.iter().all(|&done| !done)
        && rows == 1
        && gpu.route_rows(rows) == 1
        && gpu.arch_caps.is_gfx1151()
        && gpu.arch_caps.gemv_rows_default() == 2
        && gpu.flags.qwen4_decode_zba_x3_enabled()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && !gpu.qwen4_trunk_iu4_applies(rows)
        && gpu.qwen4_trunk_iu4_ragged(rows).is_none()
        && input.dtype == DType::F32
    {
        let (qkv, y_qkv) = projections[0];
        let (a, y_a) = projections[1];
        let (b, y_b) = projections[2];
        let (z, y_z) = projections[3];
        if qkv.dtype == DType::MQ6G256V2
            && y_qkv.dtype == DType::F32
            && [a, b, z].iter().all(|w| {
                w.dtype == DType::MQ4G256V2
                    && w.k == qkv.k
                    // This shape takes a specialized non-r2 lm-head route.
                    && !(w.m == 248_320 && w.k == 2_048)
            })
            && [y_a, y_b, y_z].iter().all(|y| y.dtype == DType::F32)
        {
            let elements = checked_mul(rows, qkv.k, "rotation scratch")?;
            let x = view(rotation, 0, elements);
            hip(gpu.rotate_x_mq(input, &x, qkv.k))?;
            project_rotated(gpu, qkv, &x, y_qkv, rows)?;
            return hip(gpu.gemv_mq4g256v2_multirow_r2_x3(
                a.buf, y_a, a.m, b.buf, y_b, b.m, z.buf, y_z, z.m, &x, qkv.k,
            ));
        }
    }
    let mut done: SmallVec<[bool; 4]> = if covered.is_empty() {
        SmallVec::from_elem(false, projections.len())
    } else {
        covered.iter().copied().collect()
    };
    let k = projections
        .iter()
        .find(|(w, _)| w.dtype == DType::MQ4G256V2)
        .map(|(w, _)| w.k);
    if let (Some(k), DType::F32) = (k, input.dtype) {
        if wide_exact_rows(gpu, rows) {
            // Wide-exact route keeps the singleton path; fall through to the
            // generic handling below (syntax-only nesting: edition 2021 has no
            // let-chains; semantics identical to `&& !wide_exact_rows`).
        } else {
            let mq4: SmallVec<[usize; 4]> = (0..projections.len())
                .filter(|&j| {
                    !done[j]
                        && projections[j].0.dtype == DType::MQ4G256V2
                        && projections[j].0.k == k
                })
                .collect();
            let dense: SmallVec<[usize; 4]> = if gpu.qwen4_trunk_iu4_applies(rows) {
                mq4.iter().copied().filter(|&j| a4[j]).collect()
            } else {
                SmallVec::new()
            };
            if !gpu.qwen4_trunk_iu4_applies(rows) {
                if let (Some(n_pad), Some(fold)) = (gpu.qwen4_trunk_iu4_ragged(rows), zba.as_ref())
                {
                    if let Some([y_z, y_beta, y_alpha]) = fold.padded_outputs {
                        let z = projections[fold.z].0;
                        if [fold.z, fold.beta, fold.alpha]
                            .iter()
                            .all(|&j| a4[j] && mq4.contains(&j))
                            && projections[fold.beta].0.m == 48
                            && projections[fold.alpha].0.m == 48
                            && hip(gpu.qwen4_trunk_zba_ragged(
                                input, rows, n_pad, k, fold.rows, z.m, y_z, y_beta, y_alpha,
                            ))?
                        {
                            done[fold.z] = true;
                            done[fold.beta] = true;
                            done[fold.alpha] = true;
                        }
                    }
                }
            }
            if !dense.is_empty() {
                let reservation = hip(gpu.reserve_int4_mmq(k, rows))?;
                let prepared =
                    hip(gpu.rotate_x_mq_i4_batched(input, None, None, reservation, k, rows))?;
                if let Some(fold) = zba {
                    let (z, y_z) = projections[fold.z];
                    if [fold.z, fold.beta, fold.alpha]
                        .iter()
                        .all(|j| dense.contains(j))
                        && hip(gpu.gemm_qwen4_trunk_zba_iu4(
                            fold.rows,
                            &prepared,
                            y_z,
                            projections[fold.beta].1,
                            projections[fold.alpha].1,
                            z.m,
                            k,
                            rows,
                        ))?
                    {
                        done[fold.z] = true;
                        done[fold.beta] = true;
                        done[fold.alpha] = true;
                    }
                }
                for &j in &dense {
                    if !done[j] {
                        let (w, out) = projections[j];
                        let xq = hip(gpu.int4_mmq_prepared_ptr(&prepared, k, rows))?;
                        hip(gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(w.buf, xq, out, w.m, k, rows))?;
                        done[j] = true;
                    }
                }
            }
        }
    }
    if done.iter().all(|d| !d) {
        return project_weights_with_epilogues(
            gpu,
            input,
            rows,
            Some(rotation),
            projections,
            epilogues,
        );
    }
    debug_assert!(
        covered.is_empty() || (0..projections.len()).all(|j| !covered[j] || done[j]),
        "covered family must never be computed by native projection dispatch"
    );
    let rest: SmallVec<[(&WeightRef<'_>, &GpuTensor); 4]> = (0..projections.len())
        .filter(|&j| !done[j])
        .map(|j| projections[j])
        .collect();
    if rest.is_empty() {
        return Ok(());
    }
    // Only the original GDN a/b/z (positions 1.. of the in-projection group)
    // may join the row-region fold; qkv (position 0) never does.
    let members: SmallVec<[bool; 4]> = (0..projections.len())
        .filter(|&j| !done[j])
        .map(|j| gdn_in_proj && j != 0)
        .collect();
    let epilogues_rest: SmallVec<[Option<&NormRopeEpi<'_>>; 4]> = if epilogues.is_empty() {
        SmallVec::new()
    } else {
        (0..projections.len())
            .filter(|&j| !done[j])
            .map(|j| epilogues[j])
            .collect()
    };
    project_weights_inner(
        gpu,
        input,
        rows,
        Some(rotation),
        &rest,
        gdn_in_proj.then_some(&members[..]),
        &epilogues_rest,
    )
}

/// FWHT basis a quantized projection payload reads: the aligned-K 256-wide
/// one (MQ4G256V2, MQ6G256V2, MFP4G32E8SOA) or MQ4G128V2's row-local 128-wide
/// one. `None` for payloads that read the natural activation.
#[derive(PartialEq)]
pub(super) enum RotationBasis {
    Aligned256,
    RowLocal128,
}

pub(super) fn rotation_basis(dtype: DType) -> Option<RotationBasis> {
    match dtype {
        DType::MQ4G256V2
        | DType::MQ6G256V2
        | DType::MQ5G256V2
        | DType::MQ3G256V2
        | DType::MQ2G256V2
        | DType::MFP4G32E8SOA => Some(RotationBasis::Aligned256),
        DType::MQ4G128V2 => Some(RotationBasis::RowLocal128),
        _ => None,
    }
}

/// Rotate `input` into `weight`'s FWHT basis in the rotation scratch, or
/// `None` when the payload reads the natural activation.
fn rotate_input(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
) -> Result<Option<GpuTensor>, DispatchError> {
    let basis = match weight.dtype {
        // BF16 and Q8F16 read the natural activation: neither carries an FWHT
        // basis, so no rotation is owed and the scratch stays untouched.
        DType::BF16 | DType::Q8_0 => return Ok(None),
        dtype => rotation_basis(dtype).ok_or(DispatchError::UnsupportedVariant {
            family: "layer-operations",
            variant: "unprojectable-payload",
            arch: "",
            quant: "unsupported",
        })?,
    };
    let rotation = rotation.ok_or(DispatchError::UnsupportedVariant {
        family: "layer-operations",
        variant: "rotation-scratch-absent",
        arch: "",
        quant: "quantized",
    })?;
    let elements = checked_mul(rows, weight.k, "rotation scratch")?;
    let scratch = view(rotation, 0, elements);
    match basis {
        RotationBasis::Aligned256 if gpu.route_rows(rows) > 1 => {
            hip(gpu.rotate_x_mq_batched(input, &scratch, weight.k, rows))?
        }
        RotationBasis::Aligned256 => hip(gpu.rotate_x_mq(input, &scratch, weight.k))?,
        RotationBasis::RowLocal128 => hip(gpu.rotate_x_mq_128_v2(input, &scratch, weight.k, rows))?,
    }
    Ok(Some(scratch))
}

/// Run one projection on an input already in the weight's basis.
///
/// gfx942 BF16 weights never reach this on the MFMA route
/// ([`project_weights_inner`] takes them first); the BF16 arms below are
/// the non-gfx942 arms and, on gfx942, the `HIPFIRE_G942_TRUNK_GEMV=1` one-row
/// GEMV.
fn project_rotated(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    output: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    let route = gpu.route_rows(rows);
    // These are post-load HIPFIRE_QWEN4_REQUANT experiments, not packed
    // Flash-Next source tiers. Their batched entries unconditionally use
    // F16 WMMA; retain each format's own singleton rather than silently
    // changing its arithmetic. This baseline grants no throughput qualification.
    if wide_projection_uses_singletons(weight.dtype, route, gpu.few_row_limit()) {
        for row in 0..rows {
            let input = view(x, row * weight.k, weight.k);
            let out = view(output, row * weight.m, weight.m);
            hip(match weight.dtype {
                DType::MQ2G256V2 => {
                    gpu.gemv_mq2g256v2(weight.buf, &input, &out, weight.m, weight.k)
                }
                DType::MQ3G256V2 => {
                    gpu.gemv_mq3g256v2(weight.buf, &input, &out, weight.m, weight.k)
                }
                DType::MQ5G256V2 => {
                    gpu.gemv_mq5g256v2(weight.buf, &input, &out, weight.m, weight.k)
                }
                _ => unreachable!(),
            })?;
        }
        return Ok(());
    }
    if weight.dtype == DType::MQ6G256V2 && (2..=gpu.few_row_limit()).contains(&route) {
        return hip(gpu.gemm_mq6g256v2_f32_rows(weight.buf, x, output, weight.m, weight.k, rows));
    }
    if weight.dtype == DType::BF16
        && (2..=gpu.few_row_limit()).contains(&route)
        && gpu.arch_caps.has_gfx11_plus_simt()
    {
        let part = (weight.buf, output, weight.m);
        let none = (weight.buf, output, 0);
        return hip(gpu.gemv_bf16_xf32_x4_rows([part, none, none, none], x, weight.k, rows));
    }
    // Few-row verify (2..=`few_row_limit` route rows): the exact MQ4 rows kernel, bitwise the AR
    // `gemv_mq4g256v2` per activation row. One row keeps that GEMV; more route
    // rows than the limit keep the WMMA `gemm_mq4g256v2`.
    if weight.dtype == DType::MQ4G256V2 && (2..=gpu.few_row_limit()).contains(&route) {
        let part = (weight.buf, output, weight.m);
        let none = (weight.buf, output, 0);
        return hip(gpu.gemv_mq4g256v2_rows_x4([part, none, none, none], x, weight.k, rows));
    }
    let result = match (weight.dtype, route > 1) {
        (DType::BF16, false) => gpu.gemv_bf16_xf32(weight.buf, x, output, weight.m, weight.k),
        (DType::BF16, true) => {
            // Long prefill on gfx11 or gfx1201: the KLD-gated F16 WMMA route,
            // else exact.
            match gpu.gemm_bf16_xf32_f16_wmma_qwen4(
                &[(weight.buf, output, weight.m)],
                x,
                weight.k,
                rows,
            ) {
                Ok(true) => Ok(()),
                Ok(false) if rows == 1 => {
                    gpu.gemv_bf16_xf32(weight.buf, x, output, weight.m, weight.k)
                }
                Ok(false) => {
                    gpu.gemm_bf16_xf32_multirow(weight.buf, x, output, weight.m, weight.k, rows)
                }
                Err(error) => Err(error),
            }
        }
        (DType::MQ4G256V2, false) => gpu.gemv_mq4g256v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ4G256V2, true) => {
            gpu.gemm_mq4g256v2(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MQ6G256V2, false) => gpu.gemv_mq6g256v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ6G256V2, true) => {
            gpu.gemm_mq6g256v2(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MQ5G256V2, false) => gpu.gemv_mq5g256v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ5G256V2, true) => {
            gpu.gemm_mq5g256v2(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MQ3G256V2, false) => gpu.gemv_mq3g256v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ3G256V2, true) => {
            gpu.gemm_mq3g256v2(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MQ2G256V2, false) => gpu.gemv_mq2g256v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ2G256V2, true) => {
            gpu.gemm_mq2g256v2(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::Q8_0, false) => gpu.gemv_q8_0(weight.buf, x, output, weight.m, weight.k),
        // The F32-preserving chunked entry runs the scalar `gemm_q8_0_batched`
        // kernel in 64-row launches, so the arithmetic the Q8 trunk already
        // had is unchanged.
        //
        // Deliberately not `gemm_q8_0_batched_chunked`: on gfx11/gfx1151 that
        // routes to the WMMA kernel, which rounds the activations and the
        // dequantised weights to F16. The trunk is Q8 precisely because these
        // projections write straight into the residual stream, and decode reads
        // them with the F32 `gemv_q8_0` — an F16-rounded prefill would also
        // disagree with decode.
        (DType::Q8_0, true) => {
            gpu.gemm_q8_0_batched_f32_chunked(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MFP4G32E8SOA, false) => {
            gpu.gemv_mfp4g32_e8_soa_prerotated(weight.buf, x, output, weight.m, weight.k)
        }
        (DType::MFP4G32E8SOA, true) => {
            gpu.gemm_mfp4g32_e8_soa_wmma(weight.buf, x, output, weight.m, weight.k, rows)
        }
        (DType::MQ4G128V2, false) => gpu.gemv_mq4g128v2(weight.buf, x, output, weight.m, weight.k),
        (DType::MQ4G128V2, true) => {
            gpu.gemm_mq4g128v2_batched(weight.buf, x, output, weight.m, weight.k, rows)
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "unsupported projection payload",
        )),
    };
    hip(result)
}

/// `HIPFIRE_G942_TRUNK_GEMV=1` (read once): on gfx942, one-row BF16 trunk
/// projections keep the legacy GEMV arm instead of the MFMA GEMM.
fn g942_trunk_gemv() -> bool {
    static OPT_OUT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_G942_TRUNK_GEMV", false)
    });
    *OPT_OUT
}

/// Whether a BF16-weight projection takes the gfx942 MFMA GEMM
/// (`KernelKey::GemmBf16Mfma`) at `rows` batch rows.
///
/// gfx942 only, at every row count including one: the kernel's 32-row batch
/// tile masks the tail, each output element is one fixed-order K walk of its
/// own activation row (128-wide K chunks ascending, MFMA 16-wide steps
/// ascending, no split-K, no atomics), so a row's bytes do not depend on
/// `rows`. The activation is F32 (staged to BF16) or already BF16; the output
/// is F32, or BF16 via an F32 landing in the op's rotation scratch. The one
/// exception is `gemv_opt_out` (`HIPFIRE_G942_TRUNK_GEMV=1`) at `rows == 1`,
/// which keeps the legacy GEMV arm.
fn g942_trunk_mfma_route(
    gfx942: bool,
    gemv_opt_out: bool,
    rows: usize,
    input: DType,
    weight: DType,
    output: DType,
) -> bool {
    gfx942
        && rows > 0
        && weight == DType::BF16
        && matches!(input, DType::F32 | DType::BF16)
        && matches!(output, DType::F32 | DType::BF16)
        && !(gemv_opt_out && rows == 1)
}

/// The gfx942 MFMA arm of [`project_weights_inner`] for the `group` (indices
/// into `projections`) of BF16 weights sharing `input` and `k`: one F32 -> BF16
/// staging of the activation in the shared 2-byte activation scratch (no
/// per-call allocation; none when `input` is BF16), then one
/// `gemm_bf16_mfma_gfx942` per member. The launch extent is the real `rows`;
/// the kernel pads its batch tile internally, so nothing is padded or copied
/// here. A BF16 output lands as F32 in `rotation` (`rows * m` elements; BF16
/// weights need no rotation, so the scratch is free) and is rounded RNE into
/// the output.
/// The calibration tap sees the original F32 `input`, never the staged copy.
fn project_g942_mfma(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    group: &[usize],
    k: usize,
) -> Result<(), DispatchError> {
    let n = checked_mul(rows, k, "gfx942 trunk activation")?;
    require_tensor(input, n, input.dtype, "gfx942 trunk activation")?;
    // Calibration tap: the true F32 activation, before the BF16 staging (the
    // `GemmBf16Mfma` family arm skips BF16 inputs, so it would never see it).
    if input.dtype == DType::F32 {
        for &j in group {
            gpu.maybe_capture_activation(projections[j].0.buf, input, rows, k);
        }
    }
    let staged = if input.dtype == DType::BF16 {
        view(input, 0, n)
    } else {
        let mut staged = hip(gpu.qwen4_f16_x_scratch(n))?;
        staged.dtype = DType::BF16;
        hip(gpu.convert_f32_to_bf16(input, &staged, n))?;
        staged
    };
    // The registered `KernelKey::GemmBf16Mfma` entry (arch-gated to gfx942 by
    // the registry; refuses a non-BF16 activation), not a direct launch.
    let ctx = DispatchCtx::new(gpu);
    for &j in group {
        let (w, out) = projections[j];
        let elements = checked_mul(rows, w.m, "gfx942 trunk output")?;
        require_tensor(out, elements, out.dtype, "gfx942 trunk output")?;
        if out.dtype == DType::BF16 {
            let landing = rotation
                .filter(|r| r.dtype == DType::F32 && r.numel() >= elements)
                .ok_or(DispatchError::UnsupportedVariant {
                    family: "layer-operations",
                    variant: "bf16-output-f32-landing-absent",
                    arch: "gfx942",
                    quant: "BF16",
                })?;
            let landing = view(landing, 0, elements);
            g942_gemm_bf16_mfma(gpu, &ctx, w, &staged, &landing, rows)?;
            hip(gpu.convert_f32_to_bf16(&landing, out, elements))?;
        } else {
            g942_gemm_bf16_mfma(gpu, &ctx, w, &staged, out, rows)?;
        }
    }
    Ok(())
}

/// One `KernelKey::GemmBf16Mfma` launch through the GEMM family (`x` BF16,
/// `y` F32).
fn g942_gemm_bf16_mfma(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
    GEMM.run_key(
        KernelKey::GemmBf16Mfma,
        ctx,
        gpu,
        &GemmParams {
            w,
            x,
            y,
            batch_size: rows,
        },
    )
}

/// Hyper-connection read: grouped RMSNorm, low-rank down/up projections,
/// source BF16 boundaries, sigmoid gating, and branch reduction.
pub struct HyperReadOp<'a> {
    /// The HC streams hold BF16 bits for this forward
    /// ([`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub input_mix_down: WeightRef<'a>,
    pub input_mix_up: WeightRef<'a>,
    pub normalized: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    pub low_rank: usize,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
}

impl HyperReadOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.branches == 0 || self.hidden == 0 || self.low_rank == 0 {
            return Err(DispatchError::Hip("hyper read has empty geometry".into()));
        }
        let wide = checked_mul(self.branches, self.hidden, "hyper read wide")?;
        require_tensor(
            self.input,
            checked_mul(self.rows, wide, "hyper read input")?,
            DType::F32,
            "hyper read input",
        )?;
        require_tensor(self.norm_weight, wide, DType::BF16, "hyper read norm")?;
        require_tensor(
            self.normalized,
            checked_mul(self.rows, wide, "hyper read normalized")?,
            DType::F32,
            "hyper read normalized",
        )?;
        require_tensor(
            self.low,
            checked_mul(self.rows, self.low_rank, "hyper read low")?,
            DType::F32,
            "hyper read low",
        )?;
        require_tensor(
            self.up,
            checked_mul(self.rows, wide, "hyper read up")?,
            DType::F32,
            "hyper read up",
        )?;
        require_tensor(
            self.mixed,
            checked_mul(self.rows, self.hidden, "hyper read mixed")?,
            DType::F32,
            "hyper read mixed",
        )?;
        require_weight(&self.input_mix_down, self.low_rank, wide, "hyper read down")?;
        require_weight(&self.input_mix_up, wide, self.low_rank, "hyper read up")?;
        Ok(())
    }
}

pub fn execute_hyper_read(gpu: &mut Gpu, op: &HyperReadOp<'_>) -> Result<(), DispatchError> {
    execute_hyper_read_inner(gpu, op, false, None, None, &mut false, false)
}

/// Whether the HC read `read` can hand its paired HC write `write` the gates
/// (`HIPFIRE_QWEN4_HC_FUSE` >= 1): the same stream tensor and norm weight, four
/// branches of `hidden % 256 == 0 <= 2560`, the same rows and storage, and a
/// plain BF16 gate projection.  The caller guarantees that only
/// stream-neutral mixer steps run between the two.
pub fn hyper_read_pairs_write(gpu: &Gpu, read: &HyperReadOp<'_>, write: &HyperWriteOp<'_>) -> bool {
    gpu.flags.qwen4_hc_fuse_level() >= 1
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && read.input.buf.as_ptr() == write.input.buf.as_ptr()
        && read.input.buf.as_ptr() == write.output.buf.as_ptr()
        && read.input.numel() == write.input.numel()
        && read.norm_weight.buf.as_ptr() == write.norm_weight.buf.as_ptr()
        && read.rows == write.rows
        && read.branches == 4
        && write.branches == 4
        && read.hidden == write.hidden
        && HyperNormGate::supports(read.branches, read.hidden)
        && read.state_bf16 == write.state_bf16
        && write.block_inject.dtype == DType::BF16
        && write.block_inject.m == 4
        && write.block_inject.k == 4 * write.hidden
        && write.block_inject.awq_scale.is_none()
        && write.gates.dtype == DType::F32
        && write.gates.numel() >= write.rows * write.branches
}

/// [`execute_hyper_read`] that also computes `write`'s gates when its F16
/// WMMA route admits it; returns whether it did (the write then runs
/// [`execute_hyper_write_pregated`]).  Requires [`hyper_read_pairs_write`].
pub fn execute_hyper_read_paired(
    gpu: &mut Gpu,
    read: &HyperReadOp<'_>,
    write: &HyperWriteOp<'_>,
) -> Result<bool, DispatchError> {
    let mut written = false;
    execute_hyper_read_inner(gpu, read, false, None, Some(write), &mut written, false)?;
    Ok(written)
}

/// Whether `read` + its paired `write` can take the HC row fold's output
/// instead of running `hyper_norm_gate_outputs` (`HIPFIRE_QWEN4_HC_ROW_FOLD`):
/// [`hyper_read_pairs_write`] on the F16 WMMA read route, where that launch is
/// the read's norm and the write's gates in one.
pub fn hyper_read_prenorm_applies(
    gpu: &Gpu,
    read: &HyperReadOp<'_>,
    write: &HyperWriteOp<'_>,
) -> bool {
    hyper_read_pairs_write(gpu, read, write) && hyper_read_wmma_route(gpu, read)
}

/// [`execute_hyper_read_paired`] for a read whose `hyper_norm_gate_outputs`
/// already ran (the HC row fold wrote its F16 normalized row and `write`'s
/// gates).  Requires [`hyper_read_prenorm_applies`].
pub fn execute_hyper_read_prenormed(
    gpu: &mut Gpu,
    read: &HyperReadOp<'_>,
    write: &HyperWriteOp<'_>,
) -> Result<(), DispatchError> {
    let mut written = false;
    execute_hyper_read_inner(gpu, read, false, None, Some(write), &mut written, true)?;
    if written {
        Ok(())
    } else {
        Err(DispatchError::Hip(
            "hyper read: prenormed route did not apply".into(),
        ))
    }
}

/// HC keeps singleton K/4 or K/8 owners even when token rows are tiled.
fn hc_decode_partitions(dtype: DType, k: usize, route: usize, limit: usize) -> Option<usize> {
    if route > limit {
        return None;
    }
    match dtype {
        DType::BF16 if k.is_multiple_of(32) => Some(4),
        DType::Q8_0 if k.is_multiple_of(256) => Some(8),
        _ => None,
    }
}

/// The F16 WMMA read route of [`execute_hyper_read_inner`]: the norm writes
/// the F16 down input directly and the BF16 WMMA up read reuses it.
fn hyper_read_wmma_route(gpu: &Gpu, op: &HyperReadOp<'_>) -> bool {
    let wide = op.branches * op.hidden;
    let up_fused = gpu.arch_caps.has_gfx11_plus_simt()
        && gpu.route_rows(op.rows) > 1
        && op.branches == 4
        && op.input_mix_up.dtype == DType::BF16
        && op.input_mix_up.m == wide
        && op.input_mix_up.k == op.low_rank
        && op.low_rank % 8 == 0
        && (257..=512).contains(&op.low_rank)
        && op.hidden % 8 == 0;
    let f16 = !wide_exact_rows(gpu, op.rows)
        && up_fused
        && op.input_mix_down.dtype == DType::BF16
        && gpu.qwen4_f16_wmma_applies(op.input_mix_down.buf, op.input_mix_down.k, op.rows);
    f16 && op.low_rank % 16 == 0 && op.low_rank <= 504 && op.hidden % 16 == 0
}

/// `normalized_ready`: a preceding fused launch already wrote this read's
/// single-row `normalized` (see [`execute_hyper_write_then_read`]).
fn execute_hyper_read_inner(
    gpu: &mut Gpu,
    op: &HyperReadOp<'_>,
    normalized_ready: bool,
    rotate_into: Option<&GpuTensor>,
    paired: Option<&HyperWriteOp<'_>>,
    gates_written: &mut bool,
    prenormed: bool,
) -> Result<(), DispatchError> {
    let wide = checked_mul(op.branches, op.hidden, "hyper read wide")?;
    let input = view(op.input, 0, op.rows * wide);
    let normalized = view(op.normalized, 0, op.rows * wide);
    let low = view(op.low, 0, op.rows * op.low_rank);
    let up = view(op.up, 0, op.rows * wide);
    let mixed = view(op.mixed, 0, op.rows * op.hidden);
    let norm = HyperNorm {
        input: &input,
        norm_weight: op.norm_weight,
        normalized: &normalized,
        branches: op.branches,
        hidden: op.hidden,
        state_bf16: op.state_bf16,
    };
    // Multi-row: the up projection feeds the branch mix directly; bitwise
    // identical to the GEMM + hyper_read_projected pair below.
    let up_fused = gpu.arch_caps.has_gfx11_plus_simt()
        && gpu.route_rows(op.rows) > 1
        && op.branches == 4
        && op.input_mix_up.dtype == DType::BF16
        && op.input_mix_up.m == wide
        && op.input_mix_up.k == op.low_rank
        && op.low_rank % 8 == 0
        && (257..=512).contains(&op.low_rank)
        && op.hidden % 8 == 0;
    // F16 WMMA down projection: the norm writes its F16 input directly.  The
    // BF16 WMMA up read reuses that F16 copy; the SIMT fused read instead
    // takes `normalized` as BF16 bits, which only it reads.
    let f16 = !wide_exact_rows(gpu, op.rows)
        && up_fused
        && op.input_mix_down.dtype == DType::BF16
        && gpu.qwen4_f16_wmma_applies(op.input_mix_down.buf, op.input_mix_down.k, op.rows);
    let wmma_read = f16 && op.low_rank % 16 == 0 && op.low_rank <= 504 && op.hidden % 16 == 0;
    let mut normalized_f16 = None;
    let mut activated = false;
    // The WMMA read takes `low` as packed BF16, staged in `up` (which that
    // route does not otherwise use).
    let low_bf16 = wmma_read.then(|| {
        let mut packed = view(op.up, 0, op.rows * op.low_rank);
        packed.dtype = DType::BF16;
        packed
    });
    if f16 {
        if prenormed && !(wmma_read && paired.is_some()) {
            return Err(DispatchError::Hip(
                "hyper read: prenormed input needs the paired F16 WMMA read route".into(),
            ));
        }
        let x16 = hip(gpu.qwen4_f16_x_scratch(op.rows * wide))?;
        match paired.filter(|_| wmma_read) {
            // H4: the norm also projects the paired write's gates from the
            // same streams; that write then skips its own norm + gate launch.
            Some(write) => {
                let gates = view(write.gates, 0, op.rows * write.branches);
                // `prenormed`: the HC row fold already wrote `x16` and `gates`.
                if !prenormed {
                    hip(hyper_norm_gate_outputs(
                        gpu,
                        &HyperNormGateOutputs {
                            input: &input,
                            norm_weight: op.norm_weight,
                            gate_weight: write.block_inject.buf,
                            gates: &gates,
                            normalized_f16: &x16,
                            rows: op.rows,
                            branches: op.branches,
                            hidden: op.hidden,
                            state_bf16: op.state_bf16,
                        },
                    ))?;
                }
                *gates_written = true;
            }
            None => hip(hyper_norm_f16(gpu, &norm, &x16, !wmma_read))?,
        }
        // gfx1201 / gfx1151 HC-down + activation as one launch (byte-identical to the
        // F16 WMMA GEMM + `hc_activation_fused_f32` chain on the same arch): the packed BF16
        // `low_bf16` is the only output the WMMA read consumes.  The NPU trunk pair route
        // (`HcFusionCursor` windows) reaches this same call through `execute_hyper_read_inner`.
        match low_bf16.as_ref().filter(|_| {
            wmma_read
                && op.low_rank == op.input_mix_down.m
                && gpu.qwen4_hc_down_fused_applies(
                    op.input_mix_down.m,
                    op.input_mix_down.k,
                    op.rows,
                )
        }) {
            Some(packed) => {
                hip(gpu.gemm_bf16_xf16_hc_down_fused(
                    op.input_mix_down.buf,
                    &x16,
                    packed,
                    op.input_mix_down.m,
                    op.input_mix_down.k,
                    op.rows,
                    1.0 / op.branches as f32,
                ))?;
                activated = true;
            }
            None => hip(gpu.gemm_bf16_xf16_f16_wmma(
                op.input_mix_down.buf,
                &x16,
                &low,
                op.input_mix_down.m,
                op.input_mix_down.k,
                op.rows,
            ))?,
        }
        normalized_f16 = Some(x16);
    } else if gpu.arch_caps.has_gfx11_plus_simt()
        && hc_decode_partitions(
            op.input_mix_down.dtype,
            op.input_mix_down.k,
            gpu.route_rows(op.rows),
            gpu.few_row_limit(),
        )
        .is_some()
    {
        // Decode and few-row verify: the long-K down GEMV splits each row
        // across four (BF16) or eight (Q8_0) waves and applies the activation
        // below in its epilogue; a verify row is bitwise its decode row.
        if !normalized_ready {
            hip(hyper_norm(gpu, &norm))?;
        }
        let down = &op.input_mix_down;
        let act = Some(1.0 / op.branches as f32);
        hip(if down.dtype == DType::Q8_0 {
            gpu.gemv_q8_0_k8_rows(down.buf, &normalized, &low, down.m, down.k, act, op.rows)
        } else {
            gpu.gemv_bf16_xf32_k4_rows(down.buf, &normalized, &low, down.m, down.k, act, op.rows)
        })?;
        activated = true;
    } else {
        if !normalized_ready {
            hip(hyper_norm(gpu, &norm))?;
        }
        project_weight(
            gpu,
            &op.input_mix_down,
            &normalized,
            &low,
            op.rows,
            Some(op.rotation),
        )?;
    }
    // `activated` implies the gfx11+ route: the down GEMV already applied it.
    if gpu.arch_caps.has_gfx11_plus_simt() {
        if !activated {
            hip(hc_activation_fused_f32(
                gpu,
                &HcActivationFused {
                    values: &low,
                    scale: 1.0 / op.branches as f32,
                    bf16_out: low_bf16.as_ref(),
                },
            ))?;
        }
    } else {
        // The three source BF16 boundaries, each one launch over all rows of
        // `low`: the round trip is elementwise, so this is bitwise the
        // per-row launches it replaces. Its BF16 copy is staged in `up`,
        // which the up projection below overwrites before anything reads it.
        let elements = op.rows * op.low_rank;
        let mut staging = view(op.up, 0, elements);
        staging.dtype = DType::BF16;
        let roundtrip = Bf16Roundtrip {
            input: &low,
            scratch: &staging,
            output: &low,
            elements,
        };
        hip(bf16_roundtrip_f32(gpu, &roundtrip))?;
        hip(scale_f32(
            gpu,
            &ScaleF32 {
                values: &low,
                scale: 1.0 / op.branches as f32,
            },
        ))?;
        hip(bf16_roundtrip_f32(gpu, &roundtrip))?;
        hip(gpu.silu_f32(&low, &low))?;
        hip(bf16_roundtrip_f32(gpu, &roundtrip))?;
    }
    if up_fused {
        // The F16 WMMA route's BF16 WMMA read (not bit-exact, see
        // hyper_read_up_wmma) takes the norm's F16 copy.
        let (normalized, normalized_bf16) = match normalized_f16.as_ref().filter(|_| wmma_read) {
            Some(x16) => (x16, false),
            None => (&normalized, f16),
        };
        let read = HyperReadUpFused {
            up_weight: op.input_mix_up.buf,
            low: low_bf16.as_ref().unwrap_or(&low),
            normalized,
            mixed: &mixed,
            rows: op.rows,
            hidden: op.hidden,
            low_rank: op.low_rank,
            normalized_bf16,
        };
        if wmma_read {
            return hip(hyper_read_up_wmma(gpu, &read));
        }
        return hip(hyper_read_up_fused(gpu, &read));
    }
    project_weight(gpu, &op.input_mix_up, &low, &up, op.rows, Some(op.rotation))?;
    if let Some(rotated) = rotate_into.filter(|r| {
        op.branches == 4 && op.hidden.is_multiple_of(256) && r.numel() >= op.rows * op.hidden
    }) {
        // The next step rotates `mixed` into `rotated` first: write that
        // rotation here too (the step's rotate then skips).
        return hip(gpu.hyper_read_projected_rotate(
            &normalized,
            &up,
            &mixed,
            &view(rotated, 0, op.rows * op.hidden),
            op.hidden,
            op.rows,
        ));
    }
    hip(hyper_read_projected(
        gpu,
        &HyperReadProjected {
            input: &input,
            norm_weight: op.norm_weight,
            up: &up,
            normalized: &normalized,
            mixed: &mixed,
            branches: op.branches,
            hidden: op.hidden,
        },
    ))
}
/// Hyper-connection write: grouped normalization, branch gate projection, and
/// in-place residual injection in the source-defined BF16 order.
pub struct HyperWriteOp<'a> {
    /// The HC streams hold BF16 bits for this forward
    /// ([`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub block_inject: WeightRef<'a>,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
}

impl HyperWriteOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.branches == 0 || self.hidden == 0 {
            return Err(DispatchError::Hip("hyper write has empty geometry".into()));
        }
        let wide = checked_mul(self.branches, self.hidden, "hyper write wide")?;
        require_tensor(
            self.input,
            self.rows * wide,
            DType::F32,
            "hyper write input",
        )?;
        require_tensor(self.norm_weight, wide, DType::BF16, "hyper write norm")?;
        require_tensor(
            self.normalized,
            self.rows * wide,
            DType::F32,
            "hyper write normalized",
        )?;
        require_tensor(
            self.mixed,
            self.rows * self.hidden,
            DType::F32,
            "hyper write mixed",
        )?;
        require_tensor(
            self.gates,
            self.rows * self.branches,
            DType::F32,
            "hyper write gates",
        )?;
        require_tensor(
            self.output,
            self.rows * wide,
            DType::F32,
            "hyper write output",
        )?;
        require_weight(
            &self.block_inject,
            self.branches,
            wide,
            "hyper write projection",
        )?;
        Ok(())
    }
}

pub fn execute_hyper_write(gpu: &mut Gpu, op: &HyperWriteOp<'_>) -> Result<(), DispatchError> {
    execute_hyper_write_inner(gpu, op, false)
}

/// [`execute_hyper_write`] whose gates the paired read already produced
/// ([`execute_hyper_read_paired`]) from the same, unchanged, streams.
pub fn execute_hyper_write_pregated(
    gpu: &mut Gpu,
    op: &HyperWriteOp<'_>,
) -> Result<(), DispatchError> {
    execute_hyper_write_inner(gpu, op, true)
}

/// Whether the pregated HC write `write` and the HC read `read` right after it (whose
/// own paired write is `next_write`) run as the gfx1201 mixer-tail row fold
/// (`HIPFIRE_QWEN4_HC_ROW_FOLD`): exactly the BF16-stream write `hyper_write` gives
/// `hyper_write_bf16x8_gfx1201` (see [`Gpu::hc_row_fold_gfx1201_applies`]) and a read
/// that takes [`execute_hyper_read_prenormed`] ([`hyper_read_prenorm_applies`]).
pub fn hyper_write_read_fold_applies(
    gpu: &Gpu,
    write: &HyperWriteOp<'_>,
    read: &HyperReadOp<'_>,
    next_write: &HyperWriteOp<'_>,
) -> bool {
    let Some(wide) = write.branches.checked_mul(write.hidden) else {
        return false;
    };
    let Some(streams_len) = write.rows.checked_mul(wide) else {
        return false;
    };
    gpu.flags.qwen4_hc_row_fold_mixer_gfx1201_enabled()
        && write.state_bf16
        && write.branches == 4
        && write.input.buf.as_ptr() == write.output.buf.as_ptr()
        && read.input.buf.as_ptr() == write.output.buf.as_ptr()
        && read.rows == write.rows
        && read.hidden == write.hidden
        && write.input.dtype == DType::F32
        && write.input.numel() >= streams_len
        && write.mixed.dtype == DType::F32
        && write.mixed.numel() >= write.rows * write.hidden
        && write.gates.dtype == DType::F32
        && write.gates.numel() >= write.rows * write.branches
        && gpu.hc_row_fold_gfx1201_applies(write.rows, write.hidden, write.input, write.mixed)
        && hyper_read_prenorm_applies(gpu, read, next_write)
        && hc_row_fold_next_gfx1201_ok(read, next_write)
}

/// The operand shapes the gfx1201 row folds need of the next HC read and its paired
/// write (the launch's own checks), so a mismatch plans no fold instead of failing the
/// launch.
pub fn hc_row_fold_next_gfx1201_ok(read: &HyperReadOp<'_>, next_write: &HyperWriteOp<'_>) -> bool {
    let Some(wide) = read.branches.checked_mul(read.hidden) else {
        return false;
    };
    read.norm_weight.dtype == DType::BF16
        && read.norm_weight.numel() == wide
        && next_write.block_inject.buf.dtype == DType::BF16
        && next_write.block_inject.buf.numel() >= 4 * wide
        && next_write.gates.dtype == DType::F32
        && next_write.gates.numel() >= read.rows * 4
}

/// [`execute_hyper_write_pregated`] of `write` followed by the norm + gate projection
/// of `read`'s [`execute_hyper_read_prenormed`] input, in one launch (the gfx1201
/// mixer-tail row fold; bytewise `hyper_write_bf16x8_gfx1201` then
/// `hyper_norm_gate_outputs`).  The read must then run
/// [`execute_hyper_read_prenormed`].  Requires [`hyper_write_read_fold_applies`].
pub fn execute_hyper_write_read_fold(
    gpu: &mut Gpu,
    write: &HyperWriteOp<'_>,
    read: &HyperReadOp<'_>,
    next_write: &HyperWriteOp<'_>,
) -> Result<(), DispatchError> {
    if !hyper_write_read_fold_applies(gpu, write, read, next_write) {
        return Err(DispatchError::Hip(
            "hyper write: the gfx1201 row fold was planned but its route no longer applies".into(),
        ));
    }
    let wide = checked_mul(write.branches, write.hidden, "hyper write wide")?;
    let streams = view(write.input, 0, write.rows * wide);
    let mixed = view(write.mixed, 0, write.rows * write.hidden);
    let gates = view(write.gates, 0, write.rows * write.branches);
    let next_gates = view(next_write.gates, 0, read.rows * next_write.branches);
    // The read's F16 row lives in the shared F16 scratch it takes it from.
    let x16 = hip(gpu.qwen4_f16_x_scratch(read.rows * wide))?;
    hip(
        gpu.hc_row_write_norm_gate_gfx1201(&rdna_compute::hc_row_fold::HcRowWriteNormGateGfx1201 {
            mixed: &mixed,
            streams: &streams,
            gates: &gates,
            norm_weight: read.norm_weight,
            gate_weight: next_write.block_inject.buf,
            next_gates: &next_gates,
            normalized_f16: &x16,
            rows: write.rows,
            hidden: write.hidden,
        }),
    )
}

/// H4 attention epilogue (`HIPFIRE_QWEN4_HC_FUSE` >= 2): an MQ6G256V2 (or, on the
/// gfx1151 PM module, MQ4G256V2) output
/// projection that applies the paired HC write `write` in its GEMM epilogue.
/// `write`'s gates must already be in `write.gates` (the paired read produced
/// them from the streams the write overwrites; see
/// [`execute_hyper_read_paired`]).  The rotated activation is the unfused
/// route's, the GEMM accumulates in its order and the HC streams receive the
/// `hyper_write` expression, so the result is bitwise the projection followed
/// by [`execute_hyper_write_pregated`].  The attention output is not
/// materialized.  Returns `false`, having launched nothing, when the shape or
/// route is not covered; the caller then projects and writes unfused.
fn project_output_into_hyper_write(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    a4: bool,
    write: &HyperWriteOp<'_>,
) -> Result<bool, DispatchError> {
    let Some(wide) = write.branches.checked_mul(write.hidden) else {
        return Ok(false);
    };
    // MQ4G256V2 (gfx1151 PM module only) is declined where the unfused route
    // would run the IU4 trunk projection (`a4`: its trunk bit) instead.
    let covered = match weight.dtype {
        DType::MQ6G256V2 => gpu.gemm_mq6g256v2_hcw_applies(weight.m, weight.k, rows),
        DType::MQ4G256V2 => {
            gpu.gemm_mq4g256v2_hcw_applies(weight.m, weight.k, rows)
                && !(a4 && gpu.qwen4_trunk_iu4_applies(rows))
        }
        _ => false,
    };
    if wide_exact_rows(gpu, rows)
        || gpu.flags.qwen4_hc_fuse_level() < 2
        || !covered
        || weight.m != write.hidden
        || write.branches != 4
        || write.rows != rows
        || write.input.dtype != DType::F32
        || write.input.buf.as_ptr() != write.output.buf.as_ptr()
        || write.input.numel() < rows * wide
        || write.gates.dtype != DType::F32
        || write.gates.numel() < rows * write.branches
    {
        return Ok(false);
    }
    if weight.dtype == DType::MQ4G256V2 {
        let x = hip(gpu.rotate_x_mq_batched_f16(input, weight.k, rows))?;
        hip(gpu.gemm_mq4g256v2_hcw(
            weight.buf,
            &x,
            None,
            weight.m,
            weight.k,
            rows,
            write.input,
            write.gates,
            write.state_bf16,
        ))?;
        return Ok(true);
    }
    let (x, x_is_f16) = if gpu.arch_caps.is_gfx1151() || input.dtype == DType::BF16 {
        (
            hip(gpu.rotate_x_mq_batched_f16(input, weight.k, rows))?,
            true,
        )
    } else {
        let rotated = rotate_input(gpu, weight, input, rows, rotation)?.ok_or(
            DispatchError::UnsupportedVariant {
                family: "layer-operations",
                variant: "rotation-scratch-absent",
                arch: "",
                quant: "quantized",
            },
        )?;
        (rotated, false)
    };
    hip(gpu.gemm_mq6g256v2_hcw(
        weight.buf,
        &x,
        x_is_f16,
        None,
        weight.m,
        weight.k,
        rows,
        write.input,
        write.gates,
        write.state_bf16,
    ))?;
    Ok(true)
}

fn execute_hyper_write_inner(
    gpu: &mut Gpu,
    op: &HyperWriteOp<'_>,
    gates_ready: bool,
) -> Result<(), DispatchError> {
    let wide = checked_mul(op.branches, op.hidden, "hyper write wide")?;
    let input = view(op.input, 0, op.rows * wide);
    let normalized = view(op.normalized, 0, op.rows * wide);
    let mixed = view(op.mixed, 0, op.rows * op.hidden);
    let gates = view(op.gates, 0, op.rows * op.branches);
    let output = view(op.output, 0, op.rows * wide);
    if !gates_ready {
        hyper_write_gates(gpu, op, &input, &normalized, &gates)?;
    }
    hip(hyper_write(
        gpu,
        &HyperWrite {
            input: &input,
            normalized: &normalized,
            mixed: &mixed,
            gates: &gates,
            output: &output,
            branches: op.branches,
            hidden: op.hidden,
            state_bf16: op.state_bf16,
        },
    ))
}

/// Gate quarters a fused hyper write hands to the next hyper write of the same
/// streams (see [`execute_hyper_write_then_read`]): 16 floats per row past
/// the live gates in `gates`' capacity, two alternating slots.
pub fn hyper_gate_quarters(op: &HyperWriteOp<'_>, slot: usize) -> Option<GpuTensor> {
    let len = op.rows.checked_mul(16)?;
    let start = op
        .rows
        .checked_mul(op.branches)?
        .checked_add(len.checked_mul(slot)?)?;
    (op.gates.dtype == DType::F32 && op.gates.numel() >= start + len)
        .then(|| view(op.gates, start, len))
}

/// A single-row F32 hyper write immediately followed by the hyper read of the
/// streams it writes: one `hyper_write_norm` launch writes the streams and the
/// read's norm (bitwise the separate launches). With `quarters_in` the write
/// takes its gates from quarters an earlier fused write produced (its norm
/// and gate GEMV already ran); with `next` it also produces the quarters for
/// `next`, the following hyper write of these streams, into `next`'s slot,
/// and zero-fills `clear`'s prefix (the caller skips that step). With
/// `rotate_into` the read also leaves `mq_rotate_x(mixed)` there for the step
/// that follows.
/// Returns `None`, having launched nothing, when the pair does not have that
/// shape, else whether `next`'s quarters were produced.
pub fn execute_hyper_write_then_read(
    gpu: &mut Gpu,
    write: &HyperWriteOp<'_>,
    read: &HyperReadOp<'_>,
    quarters_in: Option<&GpuTensor>,
    next: Option<(&HyperWriteOp<'_>, &GpuTensor)>,
    clear: Option<&ClearOp<'_>>,
    rotate_into: Option<&GpuTensor>,
) -> Result<Option<bool>, DispatchError> {
    let fusable = (1..=gpu.few_row_limit()).contains(&gpu.route_rows(write.rows))
        && read.rows == write.rows
        && !write.state_bf16
        && !read.state_bf16
        && write.branches == 4
        && write.hidden == 2560
        && read.hidden == write.hidden
        && read.branches == write.branches
        && read.input.buf.as_ptr() == write.output.buf.as_ptr();
    if !fusable {
        return Ok(None);
    }
    let wide = checked_mul(read.branches, read.hidden, "hyper read wide")?;
    let rows = write.rows;
    let next_gates = next
        .filter(|(op, _)| {
            op.block_inject.dtype == DType::BF16
                && op.block_inject.m == 4
                && op.block_inject.k == wide
                && op.block_inject.awq_scale.is_none()
        })
        .map(|(op, quarters)| HyperNextGates {
            norm_weight: op.norm_weight,
            inject: op.block_inject.buf,
            quarters,
        });
    let input = view(write.input, 0, rows * wide);
    let mixed = view(write.mixed, 0, rows * write.hidden);
    let gates = view(write.gates, 0, rows * write.branches);
    let output = view(write.output, 0, rows * wide);
    let write_normalized = view(write.normalized, 0, rows * wide);
    let normalized = view(read.normalized, 0, rows * wide);
    let clear = match clear {
        Some(op) => {
            op.validate_for_gpu(gpu)?;
            Some(view(op.tensor, 0, op.elements))
        }
        None => None,
    };
    if quarters_in.is_none() {
        hyper_write_gates(gpu, write, &input, &write_normalized, &gates)?;
    }
    hip(hyper_write_norm(
        gpu,
        &HyperWrite {
            input: &input,
            normalized: &write_normalized,
            mixed: &mixed,
            gates: &gates,
            output: &output,
            branches: write.branches,
            hidden: write.hidden,
            state_bf16: false,
        },
        read.norm_weight,
        &normalized,
        quarters_in,
        next_gates.as_ref(),
        clear.as_ref(),
    ))?;
    execute_hyper_read_inner(gpu, read, true, rotate_into, None, &mut false, false)?;
    Ok(Some(next_gates.is_some()))
}

/// The hyper write's gates: its norm and BF16 gate projection of `input`.
fn hyper_write_gates(
    gpu: &mut Gpu,
    op: &HyperWriteOp<'_>,
    input: &GpuTensor,
    normalized: &GpuTensor,
    gates: &GpuTensor,
) -> Result<(), DispatchError> {
    let wide = checked_mul(op.branches, op.hidden, "hyper write wide")?;
    // Decode and few-row verify (route rows within `few_row_limit`): the gate GEMV splits each row's
    // K across four waves, and a verify row is bitwise the decode row it
    // replaces (greedy MTP == AR depends on it).
    let k4 = gpu.arch_caps.has_gfx11_plus_simt()
        && hc_decode_partitions(
            op.block_inject.dtype,
            op.block_inject.k,
            gpu.route_rows(op.rows),
            gpu.few_row_limit(),
        ) == Some(4);
    // Multi-row: one launch normalizes each row into LDS and projects the
    // BF16 gate from there; `normalized` (read by nothing below) is not
    // written.  Bitwise identical to hyper_norm + project_weight.
    let fused = gpu.arch_caps.has_gfx11_plus_simt()
        && gpu.route_rows(op.rows) > 1
        && !k4
        && op.block_inject.dtype == DType::BF16
        && op.block_inject.m == op.branches
        && op.block_inject.k == wide
        && HyperNormGate::supports(op.branches, op.hidden);
    if fused {
        hip(hyper_norm_gate(
            gpu,
            &HyperNormGate {
                input,
                norm_weight: op.norm_weight,
                gate_weight: op.block_inject.buf,
                gates,
                rows: op.rows,
                branches: op.branches,
                hidden: op.hidden,
                state_bf16: op.state_bf16,
            },
        ))?;
    } else {
        hip(hyper_norm(
            gpu,
            &HyperNorm {
                input,
                norm_weight: op.norm_weight,
                normalized,
                branches: op.branches,
                hidden: op.hidden,
                state_bf16: op.state_bf16,
            },
        ))?;
        if k4 {
            // Four gate rows of K = branches * hidden; four waves per row
            // instead of one.
            hip(gpu.gemv_bf16_xf32_k4_rows(
                op.block_inject.buf,
                normalized,
                gates,
                op.block_inject.m,
                op.block_inject.k,
                None,
                op.rows,
            ))?;
        } else {
            project_weight(
                gpu,
                &op.block_inject,
                normalized,
                gates,
                op.rows,
                Some(op.rotation),
            )?;
        }
    }
    Ok(())
}

/// Which part of a mixer op runs: `Whole` is the op as it has always run;
/// a multi-lane forward splits it into the row-local input projections
/// (`InProj`, over every lane's rows), the per-lane state `Core`, and the
/// row-local output projection (`OutProj`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerPhase {
    Whole,
    InProj,
    Core,
    OutProj,
}

/// Read-only fused projection inputs, borrowed only for a Core dispatch.
#[derive(Clone, Copy)]
pub struct GdnTrunkArenas<'a> {
    pub qkv: Option<Bf16Arena<'a>>,
    pub z: Option<Bf16Arena<'a>>,
    pub a: Option<Bf16Arena<'a>>,
    pub b: Option<Bf16Arena<'a>>,
}
#[derive(Clone, Copy)]
pub struct QsaTrunkArenas<'a> {
    pub q: Option<Bf16Arena<'a>>,
    pub k: Option<Bf16Arena<'a>>,
    pub v: Option<Bf16Arena<'a>>,
    pub idx: Option<Bf16Arena<'a>>,
}

fn core_arena<T>(phase: MixerPhase, arenas: Option<T>) -> Option<T> {
    if phase == MixerPhase::Core {
        arenas
    } else {
        None
    }
}

/// Gated DeltaNet recurrence with explicit convolution and projection order.
pub struct GatedDeltaNetOp<'a> {
    pub qkv: WeightRef<'a>,
    pub conv: &'a GpuTensor,
    pub in_proj_a: WeightRef<'a>,
    pub in_proj_b: WeightRef<'a>,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub z: WeightRef<'a>,
    pub norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
    pub recurrent: &'a GpuTensor,
    /// Storage format of `recurrent` (and of a row capture's ring: the plain
    /// `format.storage()` slot). Declared by the owner of the state; never
    /// inferred from the tensor.
    pub format: GdnStateFormat,
    /// Live F16 residual of a `Q8Ef` state (`Some` exactly when
    /// `format.has_residual()`); the same owner as `recurrent`.
    pub residual: Option<&'a GpuTensor>,
    pub conv_state: &'a GpuTensor,
    pub projection: &'a GpuTensor,
    pub projection2: &'a GpuTensor,
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub recurrent_output: &'a GpuTensor,
    pub bf16_scratch: &'a GpuTensor,
    pub z_output: &'a GpuTensor,
    pub output_scratch: &'a GpuTensor,
    pub input: &'a GpuTensor,
    pub output_tensor: &'a GpuTensor,
    pub rows: usize,
    pub start_position: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub input_width: usize,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
    /// Few-row speculative verify: per-row rollback points (see
    /// [`GdnRowCapture`]); ignored by the one-row and chunked routes.
    pub row_capture: Option<GdnRowCapture<'a>>,
    /// Qwen4 IU4 trunk (`HIPFIRE_QWEN4_TRUNK_IU4`): Z, beta and alpha rows
    /// folded into one QT44 matrix (`z.m + 256` rows: Z, `in_proj_b`,
    /// `in_proj_a`, zero rows) for the V2B Z|beta|alpha SET; `None` runs the
    /// three projections separately.
    pub zba_fold: Option<&'a GpuTensor>,
    /// [`TrunkFamily`] bits of this layer's GDN projections that take the A4
    /// route when their weight is symmetric MQ4G256V2 and the route applies
    /// (`0` = none; tests).
    pub trunk_a4: u16,
    /// Which part of the mixer this call runs (see [`MixerPhase`]).
    pub phase: MixerPhase,
}

/// Where a few-row GDN forward leaves what a later rollback to any accepted
/// row prefix needs: the recurrent state after the last row (slot `rows - 1`
/// of `states`, a `[rows]` ring of states in `recurrent`'s format, written
/// instead of updating `recurrent`, which stays the pre-forward state), the
/// convolution input rows (`inputs`, `[rows, qkv]` F32; every reader of the
/// convolution history rounds it to BF16), and the recurrence inputs a
/// rollback re-runs the kept rows from (`recurrence`: the convolution output
/// `[rows, qkv]`, then gate and beta `[rows, value_heads]` each).
pub struct GdnRowCapture<'a> {
    pub states: GpuTensor,
    pub inputs: &'a GpuTensor,
    pub recurrence: &'a GpuTensor,
}

impl GatedDeltaNetOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    /// A row capture spans the stateful part of the op: the whole mixer, or
    /// its core when the input projections ran in an earlier call (the core
    /// then copies the qkv projection rows into the capture's `inputs`). The
    /// projection-only phases cannot honor one.
    fn check_phase(&self) -> Result<(), DispatchError> {
        if matches!(self.phase, MixerPhase::InProj | MixerPhase::OutProj)
            && self.row_capture.is_some()
        {
            return Err(DispatchError::Hip(
                "GDN row capture needs the whole mixer op or its core".into(),
            ));
        }
        Ok(())
    }

    fn validate_layout(&self) -> Result<(), DispatchError> {
        self.check_phase()?;
        // Only the tensors a phase touches are checked; the state (core) tensors
        // are never reached by the projection phases.
        let inproj = matches!(self.phase, MixerPhase::Whole | MixerPhase::InProj);
        let core = matches!(self.phase, MixerPhase::Whole | MixerPhase::Core);
        let outproj = matches!(self.phase, MixerPhase::Whole | MixerPhase::OutProj);
        if self.rows == 0
            || self.input_width == 0
            || self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || self.conv_kernel == 0
            || self.value_heads % self.key_heads != 0
        {
            return Err(DispatchError::Hip(
                "gated delta net has invalid geometry".into(),
            ));
        }
        let qk = checked_mul(self.key_heads, self.key_dim, "gated delta qk")?;
        let value = checked_mul(self.value_heads, self.value_dim, "gated delta value")?;
        let qkv_width = checked_mul(2, qk, "gated delta qkv")?
            .checked_add(value)
            .ok_or_else(|| DispatchError::Hip("gated delta qkv overflows".into()))?;
        let qkv = checked_mul(self.rows, qkv_width, "gated delta qkv rows")?;
        let rows_input = checked_mul(self.rows, self.input_width, "gated delta input")?;
        let rows_value = checked_mul(self.rows, value, "gated delta value rows")?;
        let rows_heads = checked_mul(self.rows, self.value_heads, "gated delta head rows")?;
        if inproj {
            require_tensor(self.input, rows_input, DType::F32, "gated delta input")?;
        }
        if inproj || core {
            require_tensor(self.projection, qkv, DType::F32, "gated delta projection")?;
        }
        if core {
            require_tensor(
                self.projection2,
                qkv,
                DType::F32,
                "gated delta projection scratch",
            )?;
            let state_format = self.format;
            if !state_format.supports(self.key_dim, self.value_dim) {
                return Err(DispatchError::Hip(format!(
                    "gated delta {} state needs 128x128 heads",
                    state_format.name()
                )));
            }
            require_tensor(
                self.recurrent,
                state_format.state_units(self.value_heads, self.key_dim, self.value_dim),
                state_format.dtype(),
                "gated delta state",
            )?;
            match (state_format.has_residual(), self.residual) {
                (true, Some(residual)) => require_tensor(
                    residual,
                    state_format.residual_units(self.value_heads, self.key_dim, self.value_dim),
                    DType::F16,
                    "gated delta state residual",
                )?,
                (true, None) => {
                    return Err(DispatchError::Hip(format!(
                        "gated delta {} state needs its live F16 residual",
                        state_format.name()
                    )));
                }
                (false, Some(_)) => {
                    return Err(DispatchError::Hip(format!(
                        "gated delta {} state takes no residual",
                        state_format.name()
                    )));
                }
                (false, None) => {}
            }
            require_tensor(
                self.conv_state,
                checked_mul(self.conv_kernel - 1, qkv_width, "gated delta conv state")?,
                DType::F32,
                "gated delta conv state",
            )?;
            require_tensor(
                self.conv,
                checked_mul(qkv_width, self.conv_kernel, "gated delta conv")?,
                DType::BF16,
                "gated delta conv",
            )?;
        }
        if inproj || core {
            require_tensor(self.a, rows_heads, DType::F32, "gated delta A")?;
            require_tensor(self.b, rows_heads, DType::F32, "gated delta B")?;
        }
        if core {
            require_tensor(self.gate, rows_heads, DType::F32, "gated delta gate")?;
            require_tensor(self.beta, rows_heads, DType::F32, "gated delta beta")?;
            require_tensor(
                self.recurrent_output,
                rows_value,
                DType::F32,
                "gated delta recurrent output",
            )?;
        }
        if inproj || core {
            require_tensor(self.z_output, rows_value, DType::F32, "gated delta Z")?;
        }
        if core || outproj {
            require_tensor(
                self.output_scratch,
                rows_value,
                DType::F32,
                "gated delta output scratch",
            )?;
        }
        require_tensor(
            self.bf16_scratch,
            value.max(self.input_width),
            DType::BF16,
            "gated delta BF16 scratch",
        )?;
        if core {
            require_tensor(
                self.a_log,
                self.value_heads,
                DType::BF16,
                "gated delta A-log",
            )?;
            require_tensor(
                self.dt_bias,
                self.value_heads,
                DType::BF16,
                "gated delta dt bias",
            )?;
            require_tensor(self.norm, self.value_dim, DType::BF16, "gated delta norm")?;
        }
        if outproj {
            require_tensor(
                self.output_tensor,
                rows_input,
                DType::F32,
                "gated delta output",
            )?;
        }
        if inproj {
            require_weight(&self.qkv, qkv_width, self.input_width, "gated delta qkv")?;
            require_weight(
                &self.in_proj_a,
                self.value_heads,
                self.input_width,
                "gated delta a",
            )?;
            require_weight(
                &self.in_proj_b,
                self.value_heads,
                self.input_width,
                "gated delta b",
            )?;
            require_weight(&self.z, value, self.input_width, "gated delta z")?;
        }
        if core || outproj {
            require_weight(
                &self.output,
                self.input_width,
                value,
                "gated delta output projection",
            )?;
        }
        Ok(())
    }
}

/// Whether the GDN mixer takes the batched persistent k4 route for `rows`.
/// gfx942 takes it at one row too (decode): its persistent recurrence is
/// pinned to the sequential kernel's arithmetic and the batched convolution,
/// gate parameters and gate compute each row as the per-row kernels do, so a
/// decode row is bitwise the per-row route's, and the step runs as one
/// launch over all value heads instead of the sequential kernel.
fn gdn_persistent_batch(gpu: &Gpu, op: &GatedDeltaNetOp<'_>, rows: usize) -> bool {
    gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
        && (gpu.route_rows(rows) > 1 || gpu.arch_caps.is_gfx942())
        && op.key_dim == 128
        && op.value_dim == 128
        && op.conv_kernel == 4
}

/// Whether the native GDN mixer stores `op`'s qkv projection as BF16 (the
/// chunked batched route with a BF16-storing MQ6/MQ4 GEMM), i.e. the dtype of
/// `op.projection` that [`execute_gated_delta_net`] writes and its convolution
/// reads. Exported so producers outside the dispatch (the NPU trunk twin) write
/// the same layout instead of re-deriving it.
pub fn gdn_qkv_projection_bf16(gpu: &Gpu, op: &GatedDeltaNetOp<'_>) -> bool {
    let persistent_batch = gdn_persistent_batch(gpu, op, op.rows);
    let dims = GatedDeltaStepBatched {
        projection: op.projection2,
        gate: op.gate,
        beta: op.beta,
        state: op.recurrent,
        format: op.format,
        residual: op.residual,
        output: op.recurrent_output,
        row_states: None,
        rows: op.rows,
        qkv_width: op.key_heads * op.key_dim * 2 + op.value_heads * op.value_dim,
        key_heads: op.key_heads,
        value_heads: op.value_heads,
        key_dim: op.key_dim,
        value_dim: op.value_dim,
        position: op.start_position,
    };
    (!wide_exact_rows(gpu, op.rows)
        && persistent_batch
        && gated_delta_chunk_route(gpu, &dims)
        && match op.qkv.dtype {
            DType::MQ6G256V2 => gpu.gemm_mq6g256v2_xf16_applies(op.qkv.k, op.rows),
            DType::MQ4G256V2 => {
                gpu.gemm_mq4g256v2_pm_applies(op.qkv.k, op.rows)
                    && !(op.trunk_a4 & TrunkFamily::GdnQkv.bit() != 0
                        && gpu.qwen4_trunk_iu4_applies(op.rows))
            }
            _ => false,
        })
        || gdn_dense_qkv_bf16(gpu, op, &dims)
}

/// gfx1201 FN dense-scan prefill (`HIPFIRE_QWEN4_GDN_QKV_BF16`, opt-in, needs
/// `HIPFIRE_QWEN4_GDN_CHAIN_FUSE`): the qkv projection is stored as BF16 bits by the exact-INT8 MQ6 GEMM's BF16
/// entry (RNE of the F32 accumulator). The dense producer's convolution reads each qkv value only
/// through that same BF16 rounding, so the chain's bytes are unchanged while the GEMM writes and
/// the producer reads half the bytes (20 KB instead of 40 KB per row). A pure function of the
/// whole bank (`op.rows`, never a row window), so the input-projection and core phases of a
/// split mixer agree on the stored dtype; `step` carries the bank's dims.
#[cfg(feature = "deltanet")]
fn gdn_dense_qkv_bf16(
    gpu: &Gpu,
    op: &GatedDeltaNetOp<'_>,
    step: &GatedDeltaStepBatched<'_>,
) -> bool {
    gpu.flags.qwen4_gdn_qkv_bf16_enabled()
        && op.row_capture.is_none()
        && step.rows == op.rows
        && !wide_exact_rows(gpu, op.rows)
        && gdn_persistent_batch(gpu, op, op.rows)
        && !gated_delta_chunk_route(gpu, step)
        && rdna_compute::fn_gdn_dense::applies(gpu, step)
        && mq6_i8_route(gpu, &op.qkv, op.rows)
}

#[cfg(not(feature = "deltanet"))]
fn gdn_dense_qkv_bf16(_: &Gpu, _: &GatedDeltaNetOp<'_>, _: &GatedDeltaStepBatched<'_>) -> bool {
    false
}

pub fn execute_gated_delta_net(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
) -> Result<(), DispatchError> {
    execute_gated_delta_net_hc(gpu, op, None, &mut false)
}

/// [`execute_gated_delta_net`] whose output projection may carry the paired HC
/// write `hc` (whose gates the paired read already produced).  `fused` reports
/// that it did: `hc`'s streams are then written and the write step is done.
pub fn execute_gated_delta_net_hc(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
    hc: Option<&HyperWriteOp<'_>>,
    fused: &mut bool,
) -> Result<(), DispatchError> {
    execute_gated_delta_net_hc_impl(gpu, op, hc, fused, None, None, None)
}

/// Core-only direct read; the ordinary/emulated operation has no arena args.
pub fn execute_gated_delta_net_arena(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
    arenas: GdnTrunkArenas<'_>,
) -> Result<(), DispatchError> {
    if op.phase != MixerPhase::Core {
        return Err(DispatchError::Hip(
            "GDN trunk arena requires Core phase".into(),
        ));
    }
    op.validate_for_gpu(gpu)?;
    execute_gated_delta_net_hc_impl(gpu, op, None, &mut false, Some(arenas), None, None)
}

/// Rows `[lo, lo + rows)` of a bank-wide mixer launch. The in-bank session
/// sub-launch runs a bank's Core (and its PLE depthwise step) as two windows
/// split at a snapshot row; each window ends with the ordinary chunk-end state
/// round trip (GDN Q8 frame `position + lo + rows - 1`, QSA `next_lengths`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreWindow {
    pub lo: usize,
    pub rows: usize,
}

impl CoreWindow {
    fn within(self, bank_rows: usize) -> Result<(), DispatchError> {
        if self.rows == 0
            || self
                .lo
                .checked_add(self.rows)
                .is_none_or(|end| end > bank_rows)
        {
            return Err(DispatchError::Hip(format!(
                "core window {}+{} is outside the bank's {bank_rows} rows",
                self.lo, self.rows
            )));
        }
        Ok(())
    }
}

/// Absolute position of a Core launch's first row (`lo` rows into the bank, or
/// the bank start for a whole launch): the GDN ring cursor and the Q8 frame
/// `position + rows - 1` of a window follow from it.
fn core_window_position(start_position: usize, lo: Option<usize>) -> usize {
    start_position + lo.unwrap_or(0)
}

/// A fused-arena view whose row 0 is row `lo` of the source arena (a
/// non-owning sub-view of the same device bytes).
struct ShiftedArena {
    tensor: GpuTensor,
    row_pitch_elems: usize,
    col0_elems: usize,
}

impl ShiftedArena {
    fn new(arena: Option<Bf16Arena<'_>>, lo: usize) -> Result<Option<Self>, DispatchError> {
        arena
            .map(|arena| {
                let skip = lo
                    .checked_mul(arena.row_pitch_elems)
                    .filter(|skip| *skip <= arena.tensor.numel())
                    .ok_or_else(|| DispatchError::Hip("arena row window overflows".into()))?;
                Ok(Self {
                    tensor: arena.tensor.sub_offset(skip, arena.tensor.numel() - skip),
                    row_pitch_elems: arena.row_pitch_elems,
                    col0_elems: arena.col0_elems,
                })
            })
            .transpose()
    }

    fn arena(&self) -> Bf16Arena<'_> {
        Bf16Arena {
            tensor: &self.tensor,
            row_pitch_elems: self.row_pitch_elems,
            col0_elems: self.col0_elems,
        }
    }
}

/// Run a partial-bank launch with the routing of a canonical whole-chunk
/// prefill (`Gpu::route_rows` constant): the window's rows are fewer than its
/// bank's, but the bank-wide input projections were stored, and the Core
/// kernels' chunk routes must be chosen, for a full chunk (banks are at least
/// `QWEN4_F16_WMMA_MIN_TOKENS` rows, so this is the routing the bank itself
/// gets). Restores the scope on every path.
fn with_bank_routing<T>(
    gpu: &mut Gpu,
    f: impl FnOnce(&mut Gpu) -> Result<T, DispatchError>,
) -> Result<T, DispatchError> {
    let prior = std::mem::replace(&mut gpu.prefill_invariant_scope, true);
    let result = f(gpu);
    gpu.prefill_invariant_scope = prior;
    result
}

/// Core of `window`'s rows only; the input projections ran bank-wide into the
/// op's scratch (`arenas` is `None` when the native route projected them).
pub fn execute_gated_delta_net_core_rows(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
    arenas: Option<GdnTrunkArenas<'_>>,
    window: CoreWindow,
) -> Result<(), DispatchError> {
    if op.phase != MixerPhase::Core {
        return Err(DispatchError::Hip(
            "GDN core window requires Core phase".into(),
        ));
    }
    window.within(op.rows)?;
    op.validate_for_gpu(gpu)?;
    let shifted = arenas
        .map(|a| {
            Ok::<_, DispatchError>([
                ShiftedArena::new(a.qkv, window.lo)?,
                ShiftedArena::new(a.z, window.lo)?,
                ShiftedArena::new(a.a, window.lo)?,
                ShiftedArena::new(a.b, window.lo)?,
            ])
        })
        .transpose()?;
    let view_arenas = shifted.as_ref().map(|s| GdnTrunkArenas {
        qkv: s[0].as_ref().map(ShiftedArena::arena),
        z: s[1].as_ref().map(ShiftedArena::arena),
        a: s[2].as_ref().map(ShiftedArena::arena),
        b: s[3].as_ref().map(ShiftedArena::arena),
    });
    let whole = window.lo == 0 && window.rows == op.rows;
    if whole {
        return execute_gated_delta_net_hc_impl(gpu, op, None, &mut false, view_arenas, None, None);
    }
    with_bank_routing(gpu, |gpu| {
        execute_gated_delta_net_hc_impl(gpu, op, None, &mut false, view_arenas, None, Some(window))
    })
}

/// QSA Core of `window`'s rows: the same op over a row sub-range whose cache
/// position and lengths are those after the rows before it
/// ([`IndexedAttentionOp::next_lengths_after`]).
pub fn execute_indexed_attention_core_rows(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    arenas: Option<QsaTrunkArenas<'_>>,
    window: CoreWindow,
) -> Result<(), DispatchError> {
    if op.phase != MixerPhase::Core {
        return Err(DispatchError::Hip(
            "QSA core window requires Core phase".into(),
        ));
    }
    window.within(op.rows)?;
    op.validate_for_gpu(gpu)?;
    let CoreWindow { lo, rows } = window;
    let index_width = (op.index_heads + op.index_kv_heads) * op.index_dim;
    let q_width = op.heads * op.head_dim;
    let kv_width = op.kv_heads * op.head_dim;
    let index_scratch = view(op.index_scratch, lo * index_width, rows * index_width);
    let qgate_scratch = view(op.qgate_scratch, lo * 2 * q_width, rows * 2 * q_width);
    let k_scratch = view(op.k_scratch, lo * kv_width, rows * kv_width);
    let v_scratch = view(op.v_scratch, lo * kv_width, rows * kv_width);
    let qsa_output = view(op.qsa_output, lo * q_width, rows * q_width);
    let (full_len, raw_len, pooled_len, selected_len, position) = if lo == 0 {
        (
            op.state.full_len,
            op.state.raw_len,
            op.state.pooled_len,
            op.state.selected_len,
            op.state.position,
        )
    } else {
        op.next_lengths_after(lo)?
    };
    let state = IndexedAttentionState {
        format: op.state.format,
        full_keys: op.state.full_keys,
        full_values: op.state.full_values,
        raw_index_keys: op.state.raw_index_keys,
        pooled_keys: op.state.pooled_keys,
        selected_indices: op.state.selected_indices,
        f16_arena: op.state.f16_arena,
        pooled_bf16: op.state.pooled_bf16,
        full_capacity: op.state.full_capacity,
        raw_capacity: op.state.raw_capacity,
        pooled_capacity: op.state.pooled_capacity,
        selected_capacity: op.state.selected_capacity,
        position_capacity: op.state.position_capacity,
        full_len,
        raw_len,
        pooled_len,
        selected_len,
        position,
    };
    let sub = IndexedAttentionOp {
        indexer_qk: op.indexer_qk,
        indexer_q_norm: op.indexer_q_norm,
        indexer_k_norm: op.indexer_k_norm,
        q: op.q,
        k: op.k,
        v: op.v,
        q_norm: op.q_norm,
        k_norm: op.k_norm,
        output: op.output,
        state,
        input: op.input,
        index_scratch: &index_scratch,
        qgate_scratch: &qgate_scratch,
        k_scratch: &k_scratch,
        v_scratch: &v_scratch,
        qsa_output: &qsa_output,
        // Per-row selections are consumed inside the launch.
        selected_scratch: op.selected_scratch,
        attention_output: op.attention_output,
        bf16_scratch: op.bf16_scratch,
        rows,
        index_heads: op.index_heads,
        index_kv_heads: op.index_kv_heads,
        index_dim: op.index_dim,
        budget: op.budget,
        compress: op.compress,
        heads: op.heads,
        kv_heads: op.kv_heads,
        head_dim: op.head_dim,
        rotary_dim: op.rotary_dim,
        rope: op.rope,
        mrope: op.mrope,
        input_width: op.input_width,
        rotation: op.rotation,
        mode: op.mode,
        trunk_a4: op.trunk_a4,
        phase: MixerPhase::Core,
    };
    sub.validate_for_gpu(gpu)?;
    let shifted = arenas
        .map(|a| {
            Ok::<_, DispatchError>([
                ShiftedArena::new(a.q, lo)?,
                ShiftedArena::new(a.k, lo)?,
                ShiftedArena::new(a.v, lo)?,
                ShiftedArena::new(a.idx, lo)?,
            ])
        })
        .transpose()?;
    let view_arenas = shifted.as_ref().map(|s| QsaTrunkArenas {
        q: s[0].as_ref().map(ShiftedArena::arena),
        k: s[1].as_ref().map(ShiftedArena::arena),
        v: s[2].as_ref().map(ShiftedArena::arena),
        idx: s[3].as_ref().map(ShiftedArena::arena),
    });
    if window.lo == 0 && window.rows == op.rows {
        return execute_indexed_attention_hc_impl(gpu, &sub, None, &mut false, view_arenas, None);
    }
    with_bank_routing(gpu, |gpu| {
        execute_indexed_attention_hc_impl(gpu, &sub, None, &mut false, view_arenas, None)
    })
}

/// The PLE grouped depthwise step over `window`'s rows (state carries across
/// windows exactly as across chunks).
pub fn execute_grouped_depthwise_rows(
    gpu: &mut Gpu,
    op: &GroupedDepthwiseOp<'_>,
    window: CoreWindow,
) -> Result<(), DispatchError> {
    window.within(op.rows)?;
    let CoreWindow { lo, rows } = window;
    let channels = op.branches * op.hidden;
    if channels % 2 != 0 || op.embed_dim % 2 != 0 {
        return Err(DispatchError::Hip(
            "grouped depthwise window needs even row widths".into(),
        ));
    }
    // HC streams hold BF16 bits (2 bytes) when `state_bf16`; the PLE rows are
    // F16 (2 bytes) when `rows_f16`, else F32. Both tensors are F32-labeled.
    let streams = view(
        op.streams,
        if op.state_bf16 {
            lo * channels / 2
        } else {
            lo * channels
        },
        rows * channels,
    );
    let ple_rows = view(
        op.rows_tensor,
        if op.rows_f16 {
            lo * op.embed_dim / 2
        } else {
            lo * op.embed_dim
        },
        rows * op.embed_dim,
    );
    let sub = GroupedDepthwiseOp {
        state_bf16: op.state_bf16,
        rows_f16: op.rows_f16,
        key: op.key,
        value: op.value,
        norm_key: op.norm_key,
        norm_query: op.norm_query,
        norm_conv: op.norm_conv,
        conv: op.conv,
        state: op.state,
        streams: &streams,
        rows_tensor: &ple_rows,
        query: op.query,
        key_scratch: op.key_scratch,
        value_scratch: op.value_scratch,
        gated: op.gated,
        normed: op.normed,
        output: op.output,
        rows,
        branches: op.branches,
        hidden: op.hidden,
        embed_dim: op.embed_dim,
        kernel_size: op.kernel_size,
        dilation: op.dilation,
        epsilon: op.epsilon,
        rotation: op.rotation,
    };
    if window.lo == 0 && window.rows == op.rows {
        return execute_grouped_depthwise(gpu, &sub);
    }
    with_bank_routing(gpu, |gpu| execute_grouped_depthwise(gpu, &sub))
}

/// Native input projections only, skipping qkv/a/b/z covered by the trunk.
pub fn execute_gated_delta_net_uncovered(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
    covered: [bool; 4],
) -> Result<(), DispatchError> {
    execute_gated_delta_net_hc_impl(gpu, op, None, &mut false, None, Some(covered), None)
}

/// The Core phase of several one-row AR lanes' GDN mixers (consecutive
/// `Step::GatedDeltaNet` Core steps of a lane step) through the multi-lane
/// kernels: each lane gets exactly the per-row branch's arguments for its row
/// (convolution + gate parameters, then the fused step + gated norm), minus
/// the decode rotation, which a Core phase's output projection never reads (the
/// interpreter drops the memo after the step). `Ok(false)`: the set does not
/// take the route, nothing launched and `before_launch` was not called; run
/// the ops one by one. `before_launch` runs exactly once, after every refusal
/// check and before the first launch; its error is returned as is and nothing
/// launches.
pub fn execute_gated_delta_net_core_lanes(
    gpu: &mut Gpu,
    ops: &[&GatedDeltaNetOp<'_>],
    before_launch: impl FnOnce(&mut Gpu) -> Result<(), DispatchError>,
) -> Result<bool, DispatchError> {
    let Some(first) = ops.first() else {
        return Ok(false);
    };
    let qkv = 2 * first.key_heads * first.key_dim + first.value_heads * first.value_dim;
    let value = first.value_heads * first.value_dim;
    let heads = first.value_heads;
    let mut views = Vec::with_capacity(ops.len());
    for op in ops {
        op.check_phase()?;
        if op.phase != MixerPhase::Core
            || op.rows != 1
            || op.row_capture.is_some()
            || gdn_persistent_batch(gpu, op, 1)
            || 2 * op.key_heads * op.key_dim + op.value_heads * op.value_dim != qkv
            || op.value_heads != heads
        {
            return Ok(false);
        }
        let projection2 = view(op.projection2, 0, qkv);
        views.push((
            view(op.projection, 0, qkv),
            view(&projection2, 0, op.key_heads * op.key_dim),
            view(
                &projection2,
                op.key_heads * op.key_dim,
                op.key_heads * op.key_dim,
            ),
            view(&projection2, 2 * op.key_heads * op.key_dim, value),
            projection2,
            [
                view(op.a, 0, heads),
                view(op.b, 0, heads),
                view(op.gate, 0, heads),
                view(op.beta, 0, heads),
            ],
            [
                view(op.recurrent_output, 0, value),
                view(op.z_output, 0, value),
                view(op.output_scratch, 0, value),
            ],
        ));
    }
    let lanes: Vec<GdnCoreLane<'_>> = ops
        .iter()
        .zip(&views)
        .map(
            |(
                op,
                (projection, q, k, v, projection2, [a, b, gate, beta], [recurrent, z, output]),
            )| {
                let history_rows = op.conv_kernel.saturating_sub(1);
                GdnCoreLane {
                    conv: GatedDeltaConv {
                        input: projection,
                        kernel: op.conv,
                        history: op.conv_state,
                        output: projection2,
                        next_history: op.conv_state,
                        channels: qkv,
                        history_rows,
                        kernel_size: op.conv_kernel,
                        cursor: if history_rows == 0 {
                            0
                        } else {
                            op.start_position % history_rows
                        },
                        row_index: 0,
                    },
                    params: GatedDeltaParams {
                        a,
                        b,
                        a_log: op.a_log,
                        dt_bias: op.dt_bias,
                        gate,
                        beta,
                    },
                    step: GatedDeltaStep {
                        q,
                        k,
                        v,
                        gate,
                        beta,
                        state: op.recurrent,
                        format: op.format,
                        residual: op.residual,
                        output: recurrent,
                        key_heads: op.key_heads,
                        value_heads: op.value_heads,
                        key_dim: op.key_dim,
                        value_dim: op.value_dim,
                        position: op.start_position,
                    },
                    gate: GatedDeltaGate {
                        recurrent_output: recurrent,
                        z,
                        norm: op.norm,
                        output,
                        value_heads: op.value_heads,
                        value_dim: op.value_dim,
                    },
                }
            },
        )
        .collect();
    let Some(plan) = hip(gated_delta_core_lanes(gpu, &lanes, heads))? else {
        return Ok(false);
    };
    before_launch(gpu)?;
    hip(plan.launch(gpu))?;
    gpu.scratch.prerotated = None;
    Ok(true)
}

fn execute_gated_delta_net_hc_impl(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
    hc: Option<&HyperWriteOp<'_>>,
    fused: &mut bool,
    arenas: Option<GdnTrunkArenas<'_>>,
    projection_only: Option<[bool; 4]>,
    window: Option<CoreWindow>,
) -> Result<(), DispatchError> {
    op.check_phase()?;
    let phase = if projection_only.is_some() {
        MixerPhase::InProj
    } else {
        op.phase
    };
    let arenas = core_arena(phase, arenas);
    // A row window is a Core launch over rows `[lo, lo + rows)` of a bank whose
    // input projections ran bank-wide: row-shaped inputs are offset by `lo`,
    // scratch only this launch touches starts at row 0, and the stored-layout
    // decisions (BF16 qkv / output) stay those of the whole bank.
    let (lo, rows) = window.map_or((0, op.rows), |w| (w.lo, w.rows));
    if let Some(w) = window {
        if phase != MixerPhase::Core
            || w.rows == 0
            || w.lo + w.rows > op.rows
            || op.row_capture.is_some()
        {
            return Err(DispatchError::Hip(
                "GDN core window needs a Core op without row capture inside its rows".into(),
            ));
        }
    }
    let qk = op.key_heads * op.key_dim;
    let value = op.value_heads * op.value_dim;
    let qkv = 2 * qk + value;
    let mut projection = view(op.projection, 0, op.rows * qkv);
    let projection2 = view(op.projection2, 0, rows * qkv);
    let a = view(op.a, lo * op.value_heads, rows * op.value_heads);
    let b = view(op.b, lo * op.value_heads, rows * op.value_heads);
    let mut gate = view(op.gate, 0, rows * op.value_heads);
    let mut beta = view(op.beta, 0, rows * op.value_heads);
    let z = view(op.z_output, lo * value, rows * value);
    let history_rows = op.conv_kernel.saturating_sub(1);
    let persistent_batch = gdn_persistent_batch(gpu, op, rows);
    if arenas.is_some() && (!persistent_batch || op.row_capture.is_some()) {
        return Err(DispatchError::Hip(
            "GDN trunk arena needs batched k4 without row capture".into(),
        ));
    }
    let recurrent_output = view(op.recurrent_output, 0, rows * value);
    if op.row_capture.is_some() && !persistent_batch {
        return Err(DispatchError::Hip(
            "GDN row capture needs the few-row persistent recurrence route".into(),
        ));
    }
    let capture = op.row_capture.as_ref();
    // A row capture keeps the qkv projection and the recurrence inputs:
    // write them straight into it. A split core receives the projection
    // the input phase left in `op.projection`, copied into the capture
    // unchanged (the convolution reads the same values either way).
    let mut conv_output = view(op.projection2, 0, rows * qkv);
    if let Some(capture) = capture {
        let inputs = view(capture.inputs, 0, op.rows * qkv);
        if phase == MixerPhase::Core {
            hip(gpu.copy_d2d(&projection, &inputs, inputs.byte_size()))?;
        }
        projection = inputs;
        let heads = op.rows * op.value_heads;
        conv_output = view(capture.recurrence, 0, op.rows * qkv);
        gate = view(capture.recurrence, op.rows * qkv, heads);
        beta = view(capture.recurrence, op.rows * qkv + heads, heads);
    }
    let dims = GatedDeltaStepBatched {
        projection: &projection2,
        gate: &gate,
        beta: &beta,
        state: op.recurrent,
        format: op.format,
        residual: op.residual,
        output: &recurrent_output,
        row_states: capture.map(|c| &c.states),
        rows,
        qkv_width: qkv,
        key_heads: op.key_heads,
        value_heads: op.value_heads,
        key_dim: op.key_dim,
        value_dim: op.value_dim,
        position: core_window_position(op.start_position, window.map(|w| w.lo)),
    };
    let chunked = persistent_batch && gated_delta_chunk_route(gpu, &dims);
    if chunked && capture.is_some() {
        return Err(DispatchError::Hip(
            "GDN row capture is a few-row verify contract, not the chunked prefill route".into(),
        ));
    }
    // On the chunked route the qkv projection is read (by the convolution)
    // only through its BF16 rounding, so the MQ6 / MQ4 GEMM stores it as BF16 bits
    // (RNE of the F32 projection, the same value the convolution would round).
    // MQ4 has BF16 output only in the PM module and never when the IU4 route
    // (`a4` = the projection's trunk bit; F32 output only) takes the projection.
    let bf16_store = |w: &WeightRef<'_>, a4: u16, gpu: &Gpu| {
        !wide_exact_rows(gpu, op.rows)
            && chunked
            && match w.dtype {
                DType::MQ6G256V2 => gpu.gemm_mq6g256v2_xf16_applies(w.k, op.rows),
                DType::MQ4G256V2 => {
                    gpu.gemm_mq4g256v2_pm_applies(w.k, op.rows)
                        && !(op.trunk_a4 & a4 != 0 && gpu.qwen4_trunk_iu4_applies(op.rows))
                }
                _ => false,
            }
    };
    // The output projection reads `gdn_output`: stored as BF16 only when the
    // projection rotates it straight to F16.  The native-FP8 MQ6 route packs an
    // F32 activation, so where it takes the projection the gate output stays F32
    // (the same BF16-rounded values, twice the bytes).
    let bf16_input_store = |gpu: &Gpu| {
        bf16_store(&op.output, TrunkFamily::GdnOut.bit(), gpu)
            && !mq6_i8_route(gpu, &op.output, op.rows)
            && !mq6_fp8_route(gpu, &op.output, op.rows)
    };
    if arenas.is_some_and(|a| a.qkv.is_some())
        || bf16_store(&op.qkv, TrunkFamily::GdnQkv.bit(), gpu)
        || gdn_dense_qkv_bf16(
            gpu,
            op,
            &GatedDeltaStepBatched {
                rows: op.rows,
                ..dims
            },
        )
    {
        projection.dtype = DType::BF16;
    }
    if window.is_some() {
        if !persistent_batch {
            return Err(DispatchError::Hip(
                "GDN core window needs the batched k4 route".into(),
            ));
        }
        // After the dtype decision: BF16 rows are 2 bytes wide.
        projection = projection.sub_offset(lo * qkv, rows * qkv);
    }
    if matches!(phase, MixerPhase::Whole | MixerPhase::InProj) {
        project_trunk_masked(
            gpu,
            op.input,
            op.rows,
            op.rotation,
            &[
                (&op.qkv, &projection),
                (&op.in_proj_a, &a),
                (&op.in_proj_b, &b),
                (&op.z, &z),
            ],
            &[
                op.trunk_a4 & TrunkFamily::GdnQkv.bit() != 0,
                op.trunk_a4 & TrunkFamily::GdnA.bit() != 0,
                op.trunk_a4 & TrunkFamily::GdnB.bit() != 0,
                op.trunk_a4 & TrunkFamily::GdnZ.bit() != 0,
            ],
            op.zba_fold.map(|rows| ZbaFold {
                rows,
                z: 3,
                beta: 2,
                alpha: 1,
                // The prefill scope excludes compact scratch. The ordinary
                // Qwen4 plan dedicates these full-capacity fields; unused token
                // rows are dead and the shared helper checks all byte extents.
                // Z/a/b stay F32 on both routes; qkv BF16-store and out HC
                // fusion gates are unchanged and never arm ragged IU4.
                padded_outputs: Some([op.z_output, op.b, op.a]),
            }),
            true,
            projection_only.as_ref().map_or(&[][..], |m| &m[..]),
            &[],
        )?;
    }
    if phase == MixerPhase::InProj {
        return Ok(());
    }
    let mut gdn_output = view(op.output_scratch, 0, op.rows * value);
    // The out projection's A8 blocks, when the dense gate produced them.
    #[cfg_attr(not(feature = "deltanet"), allow(unused_mut))]
    let mut fused_xq: Option<*mut std::ffi::c_void> = None;
    if phase == MixerPhase::OutProj {
        // The core ran in an earlier call; only the dtype decision it made for
        // its output is owed here.
        if persistent_batch && bf16_input_store(gpu) {
            gdn_output.dtype = DType::BF16;
        }
    } else if persistent_batch {
        let start_cursor =
            core_window_position(op.start_position, window.map(|w| w.lo)) % history_rows;
        // The F16 prefill route's chunked recurrence reads the convolution
        // output as packed BF16 (every value is BF16-rounded already).
        if chunked {
            conv_output.dtype = DType::BF16;
        }
        let conv = GatedDeltaConvBatched {
            input: &projection,
            kernel: op.conv,
            history: op.conv_state,
            output: &conv_output,
            next_history: op.conv_state,
            rows,
            channels: qkv,
            history_rows,
            kernel_size: op.conv_kernel,
            start_cursor,
        };
        let gate_params = GatedDeltaParamsBatched {
            a: &a,
            b: &b,
            a_log: op.a_log,
            dt_bias: op.dt_bias,
            gate: &gate,
            beta: &beta,
            rows,
            heads: op.value_heads,
        };
        let step = GatedDeltaStepBatched {
            projection: &conv_output,
            ..dims
        };
        // Opt-in `HIPFIRE_QWEN4_GDN_CONV_QKNORM`: on the exact gfx1151 chunked
        // route (never a row capture, recorder or graph capture, and never the
        // persistent gfx1201 route) the convolution also stores the normalized
        // Q/K the recurrence reads, so its Q/K columns of `conv_output` are not
        // written and the separate Q/K-norm launch is skipped.
        let fused_qk =
            chunked && capture.is_none() && gated_delta_conv_qknorm_route(gpu, &step, &conv);
        // gfx1201 FN dense route: its producer replaces the convolution, the
        // gate-parameter pass and the step (the gate below finishes it).
        #[cfg(feature = "deltanet")]
        let dense = arenas.is_none() && !chunked && rdna_compute::fn_gdn_dense::applies(gpu, &step);
        #[cfg(not(feature = "deltanet"))]
        let dense = false;
        // Convolution and gate parameters in one launch.
        if dense {
            #[cfg(feature = "deltanet")]
            hip(rdna_compute::fn_gdn_dense::run(
                gpu,
                &conv,
                &gate_params,
                &step,
            ))?;
        } else if let Some(src) = arenas {
            if fused_qk {
                hip(gated_delta_conv_params_qknorm_batched_mixed(
                    gpu,
                    &conv,
                    &gate_params,
                    &step,
                    false,
                    src.qkv,
                    src.a,
                    src.b,
                ))?;
            } else {
                hip(gated_delta_conv_params_batched_mixed(
                    gpu,
                    &conv,
                    &gate_params,
                    src.qkv,
                    src.a,
                    src.b,
                ))?;
            }
        } else if fused_qk {
            hip(gated_delta_conv_params_qknorm_batched(
                gpu,
                &conv,
                &gate_params,
                &step,
                false,
            ))?;
        } else {
            hip(gated_delta_conv_params_batched(gpu, &conv, &gate_params))?;
        }
        // The chunked route's output is BF16-rounded: stored as BF16 when the
        // output projection rotates it straight to F16 (half the bytes).
        if bf16_input_store(gpu) {
            gdn_output.dtype = DType::BF16;
        }
        if window.is_some() {
            // The core's output rows are read bank-wide by the output projection:
            // this window writes its own rows (BF16 rows are 2 bytes wide).
            gdn_output = gdn_output.sub_offset(lo * value, rows * value);
        }
        let gated = GatedDeltaGateBatched {
            recurrent_output: &recurrent_output,
            z: &z,
            norm: op.norm,
            output: &gdn_output,
            rows,
            value_heads: op.value_heads,
            value_dim: op.value_dim,
        };
        // The F16 prefill route fuses the gate into the chunked WMMA
        // recurrence (KLD-gated); otherwise the exact kernels run.
        if chunked {
            if let Some(src) = arenas.and_then(|a| a.z) {
                if fused_qk {
                    hip(gated_delta_step_gate_wmma_qknormed_arena(
                        gpu, &step, &gated, src,
                    ))?;
                } else {
                    hip(gated_delta_step_gate_wmma_arena(gpu, &step, &gated, src))?;
                }
            } else if fused_qk {
                hip(gated_delta_step_gate_wmma_qknormed(gpu, &step, &gated))?;
            } else {
                hip(gated_delta_step_gate_wmma(gpu, &step, &gated))?;
            }
        } else {
            if !dense {
                hip(gated_delta_step_batched(gpu, &step))?;
            }
            // An FWHT-basis output projection rotates the gate output first:
            // the gate writes that rotation too (its rotate then skips).
            let rotate = rotation_basis(op.output.dtype) == Some(RotationBasis::Aligned256)
                && op.output.k == value
                && op.value_heads.is_multiple_of(2);
            if dense {
                // `recurrent_output` is scratch with no reader after the gate.
                #[cfg(feature = "deltanet")]
                if rotate {
                    // `HIPFIRE_QWEN4_GDN_I8_FUSE`: the out projection's INT8 route
                    // (a lone MQ6 weight: its F32 activation is this gate output)
                    // reads the gate's own rotated A8 blocks, so neither the F32
                    // gate output nor the separate producer launch exists. Any
                    // other shape of the call (split core / out-projection phases,
                    // a row window, an HC-write epilogue that would read the F32
                    // output) keeps the F32 gate output and the unfused chain.
                    if phase == MixerPhase::Whole
                        && window.is_none()
                        && !(hc.is_some() && gpu.flags.qwen4_hc_fuse_level() >= 2)
                        && mq6_i8_route(gpu, &op.output, op.rows)
                        && rdna_compute::fn_gdn_dense::gate_i8_applies(gpu, &gated)
                    {
                        fused_xq = Some(hip(rdna_compute::fn_gdn_dense::gate_i8(gpu, &gated))?);
                    } else {
                        let rotated = view(op.rotation, lo * value, rows * value);
                        hip(rdna_compute::fn_gdn_dense::gate(
                            gpu,
                            &gated,
                            Some(&rotated),
                        ))?;
                    }
                } else {
                    hip(rdna_compute::fn_gdn_dense::gate(gpu, &gated, None))?;
                }
            } else if rotate {
                let rotated = view(op.rotation, lo * value, rows * value);
                if let Some(src) = arenas.and_then(|a| a.z) {
                    hip(gated_delta_gate_batched_rotate_arena(
                        gpu, &gated, &rotated, src,
                    ))?;
                } else {
                    hip(gated_delta_gate_batched_rotate(gpu, &gated, &rotated))?;
                }
            } else {
                if let Some(src) = arenas.and_then(|a| a.z) {
                    hip(gated_delta_gate_batched_arena(gpu, &gated, src))?;
                } else {
                    hip(gated_delta_gate_batched(gpu, &gated))?;
                }
            }
        }
    } else {
        for row in 0..op.rows {
            let position = op.start_position.saturating_add(row);
            let cursor = if history_rows == 0 {
                0
            } else {
                position % history_rows
            };
            let projection_row = view(&projection, row * qkv, qkv);
            let projection2_row = view(&projection2, row * qkv, qkv);
            let a_row = view(&a, row * op.value_heads, op.value_heads);
            let b_row = view(&b, row * op.value_heads, op.value_heads);
            let gate_row = view(op.gate, row * op.value_heads, op.value_heads);
            let beta_row = view(op.beta, row * op.value_heads, op.value_heads);
            // Convolution and gate parameters of the row in one launch.
            hip(gated_delta_conv_params(
                gpu,
                &GatedDeltaConv {
                    input: &projection_row,
                    kernel: op.conv,
                    history: op.conv_state,
                    output: &projection2_row,
                    next_history: op.conv_state,
                    channels: qkv,
                    history_rows,
                    kernel_size: op.conv_kernel,
                    cursor,
                    row_index: row,
                },
                &GatedDeltaParams {
                    a: &a_row,
                    b: &b_row,
                    a_log: op.a_log,
                    dt_bias: op.dt_bias,
                    gate: &gate_row,
                    beta: &beta_row,
                },
                op.value_heads,
            ))?;
            let q = view(&projection2_row, 0, qk);
            let k = view(&projection2_row, qk, qk);
            let v = view(&projection2_row, 2 * qk, value);
            let recurrent_output = view(op.recurrent_output, row * value, value);
            // `gated_delta_gate` reads the recurrent output through its BF16
            // boundary itself; no round-trip pass precedes it.
            let z_row = view(&z, row * value, value);
            let gdn_output = view(op.output_scratch, row * value, value);
            // Decode: the step also writes the output projection's
            // 256-wide rotation of this row (its rotate then skips).
            let rotate_into = (op.rows == 1
                && rotation_basis(op.output.dtype) == Some(RotationBasis::Aligned256)
                && op.output.k == value)
                .then(|| view(op.rotation, 0, value));
            hip(gated_delta_step_gated(
                gpu,
                &GatedDeltaStep {
                    q: &q,
                    k: &k,
                    v: &v,
                    gate: &gate_row,
                    beta: &beta_row,
                    state: op.recurrent,
                    format: op.format,
                    residual: op.residual,
                    output: &recurrent_output,
                    key_heads: op.key_heads,
                    value_heads: op.value_heads,
                    key_dim: op.key_dim,
                    value_dim: op.value_dim,
                    position,
                },
                &GatedDeltaGate {
                    recurrent_output: &recurrent_output,
                    z: &z_row,
                    norm: op.norm,
                    output: &gdn_output,
                    value_heads: op.value_heads,
                    value_dim: op.value_dim,
                },
                rotate_into.as_ref(),
            ))?;
        }
    }
    if phase == MixerPhase::Core {
        return Ok(());
    }
    if let Some(xq) = fused_xq {
        // The gate emitted the A8 blocks: the INT8 route's GEMM, minus its producer.
        let output_batch = view(op.output_tensor, 0, op.rows * op.output.m);
        return hip(gpu.gemm_mq6g256v2_i8_prequant(
            op.output.buf,
            xq,
            &output_batch,
            op.output.m,
            op.output.k,
            op.rows,
        ));
    }
    // H4 (`HIPFIRE_QWEN4_HC_FUSE` >= 2): the paired HC write rides the output
    // projection's epilogue; the projection then never lands in `output_tensor`.
    if let Some(write) = hc {
        if project_output_into_hyper_write(
            gpu,
            &op.output,
            &gdn_output,
            op.rows,
            Some(op.rotation),
            op.trunk_a4 & TrunkFamily::GdnOut.bit() != 0,
            write,
        )? {
            *fused = true;
            return Ok(());
        }
    }
    let output_batch = view(op.output_tensor, 0, op.rows * op.output.m);
    project_trunk(
        gpu,
        &gdn_output,
        op.rows,
        op.rotation,
        &[(&op.output, &output_batch)],
        &[op.trunk_a4 & TrunkFamily::GdnOut.bit() != 0],
        None,
        false,
    )?;
    // The output stays the F32 projection: its reader, the HC write, rounds
    // it to BF16 as it reads it, so no round-trip pass is owed here.
    Ok(())
}

/// Borrowed cache tensors plus scalar metadata for one operation.  The
/// architecture commits these scalar values to persistent state after the
/// shared step list succeeds. The K/V caches are `format`'s rows and the
/// raw and pooled index keys its `index_dtype`.
/// `f16_arena` is the layer's persistent F16 K/V copy of an F32 cache: the
/// gathered prefill attention converts only new or invalid rows into it
/// (`None` = per-launch conversion into the shared workspace).
pub struct IndexedAttentionState<'a> {
    pub format: QsaKvFormat,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub pooled_keys: &'a GpuTensor,
    pub selected_indices: &'a GpuTensor,
    pub f16_arena: Option<QsaF16ArenaRef<'a>>,
    pub pooled_bf16: Option<rdna_compute::qsa_pooled_mirror::QsaPooledBf16MirrorRef<'a>>,
    pub full_capacity: usize,
    pub raw_capacity: usize,
    pub pooled_capacity: usize,
    pub selected_capacity: usize,
    pub position_capacity: usize,
    pub full_len: usize,
    pub raw_len: usize,
    pub pooled_len: usize,
    pub selected_len: usize,
    pub position: usize,
}

/// Which part of the indexed-attention body one call runs.
#[derive(Clone, Copy)]
pub enum IndexedAttentionMode<'a> {
    /// Append the rows, select for every row, attend, project the output.
    Full,
    /// Append K/V and index keys only: no query projection, selection,
    /// attention or output projection. The persistent selection goes
    /// stale, so `selected_len` commits as 0.
    AppendOnly,
    /// One row: keep the persistent selection, append this position to it
    /// on device, attend over it, project the output.
    ReuseSelection { selected_len_out: &'a GpuTensor },
}

pub struct IndexedAttentionOp<'a> {
    pub indexer_qk: WeightRef<'a>,
    pub indexer_q_norm: &'a GpuTensor,
    pub indexer_k_norm: &'a GpuTensor,
    pub q: WeightRef<'a>,
    pub k: WeightRef<'a>,
    pub v: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
    pub state: IndexedAttentionState<'a>,
    pub input: &'a GpuTensor,
    pub index_scratch: &'a GpuTensor,
    pub qgate_scratch: &'a GpuTensor,
    pub k_scratch: &'a GpuTensor,
    pub v_scratch: &'a GpuTensor,
    pub qsa_output: &'a GpuTensor,
    pub selected_scratch: &'a GpuTensor,
    pub attention_output: &'a GpuTensor,
    pub bf16_scratch: &'a GpuTensor,
    pub rows: usize,
    pub index_heads: usize,
    pub index_kv_heads: usize,
    pub index_dim: usize,
    pub budget: usize,
    pub compress: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// RoPE width of the queries, keys and the indexer
    /// (`head_dim * partial_rotary_factor`; Flash-Next: 64).  Even, within
    /// `2..=min(head_dim, index_dim)`.  The one-launch decode prologue and
    /// the pooled-key RoPE kernel derive `min(dim, 64)` themselves: a width
    /// they would not derive runs the batched launches (queries/keys) or is
    /// refused (pooled keys).
    pub rotary_dim: usize,
    /// The model's rotary embedding. The QSA kernels hard-code the RoPE base;
    /// its scaling is lowered at [`Self::rotary_dim`] for every QSA rotation
    /// (queries, keys, indexer queries and pooled keys).
    pub rope: rdna_compute::rope::Rope,
    /// Image prompt coordinates; absent for every text-only route.
    pub mrope: Option<rdna_compute::qsa_mrope::QsaMrope<'a>>,
    pub input_width: usize,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
    pub mode: IndexedAttentionMode<'a>,
    /// Qwen4 IU4 trunk (`HIPFIRE_QWEN4_TRUNK_IU4`): [`TrunkFamily`] bits of
    /// this layer's QSA projections that take the A4 route when their weight
    /// is symmetric MQ4G256V2 and the route applies (`0` = none; MTP and
    /// tests).
    pub trunk_a4: u16,
    /// Which part of the mixer this call runs (see [`MixerPhase`]).
    pub phase: MixerPhase,
}

impl IndexedAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    /// Selection reuse and the K/V-only append are single-row, whole-op modes.
    fn check_phase(&self) -> Result<(), DispatchError> {
        if self.phase != MixerPhase::Whole && !matches!(self.mode, IndexedAttentionMode::Full) {
            return Err(DispatchError::Hip(
                "indexed attention selection reuse and K/V-only append need the whole mixer op"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The RoPE width ([`Self::rotary_dim`]) against the op's geometry and the
    /// fixed width the pooled-key RoPE kernel rotates.
    fn validate_rotary(&self) -> Result<(), DispatchError> {
        if let Some(mrope) = self.mrope {
            let end = self
                .state
                .position
                .checked_add(self.rows)
                .ok_or_else(|| DispatchError::Hip("image mRoPE position overflow".into()))?;
            mrope
                .validate(self.rotary_dim, end)
                .map_err(|error| DispatchError::Hip(error.to_string()))?;
        }
        if self.rotary_dim < 2
            || self.rotary_dim % 2 != 0
            || self.rotary_dim > self.head_dim
            || self.rotary_dim > self.index_dim
        {
            return Err(DispatchError::Hip(format!(
                "indexed attention rotary width {} must be even and within 2..=min(head_dim {}, index_dim {})",
                self.rotary_dim, self.head_dim, self.index_dim
            )));
        }
        // The pooled-key RoPE kernel rotates `min(index_kv_width, 64)` channels
        // and only runs with the core.
        let pooled = checked_mul(
            self.index_kv_heads,
            self.index_dim,
            "indexed attention index KV",
        )?
        .min(64);
        if matches!(self.phase, MixerPhase::Whole | MixerPhase::Core) && self.rotary_dim != pooled {
            return Err(DispatchError::Hip(format!(
                "indexed attention rotary width {} differs from the pooled-key RoPE kernel's {pooled}",
                self.rotary_dim
            )));
        }
        Ok(())
    }

    fn validate_layout(&self) -> Result<(), DispatchError> {
        self.check_phase()?;
        // Only the tensors a phase touches are checked; the state (core) checks
        // are never reached by the projection phases.
        let inproj = matches!(self.phase, MixerPhase::Whole | MixerPhase::InProj);
        let core = matches!(self.phase, MixerPhase::Whole | MixerPhase::Core);
        let outproj = matches!(self.phase, MixerPhase::Whole | MixerPhase::OutProj);
        if let IndexedAttentionMode::ReuseSelection { selected_len_out } = self.mode {
            if self.rows != 1 {
                return Err(DispatchError::Hip(
                    "indexed attention selection reuse needs one row".into(),
                ));
            }
            if self.state.selected_len == 0 {
                return Err(DispatchError::Hip(
                    "indexed attention selection reuse after a K/V-only append".into(),
                ));
            }
            if selected_len_out.dtype != DType::Raw || selected_len_out.numel() < 4 {
                return Err(DispatchError::Hip(
                    "indexed attention selection-length output must be a four-byte Raw scalar"
                        .into(),
                ));
            }
        }
        if self.rows == 0
            || self.input_width == 0
            || self.index_heads == 0
            || self.index_kv_heads == 0
            || self.index_dim == 0
            || self.index_dim > 256
            || self.index_dim % 2 != 0
            || self.compress == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.head_dim > 256
            || (self.head_dim < 64 && self.head_dim % 2 != 0)
            || self.kv_heads > self.heads
            || self.heads % self.kv_heads != 0
            || (core
                && (self.state.full_capacity == 0
                    || self.state.raw_capacity == 0
                    || self.state.pooled_capacity == 0
                    || self.state.selected_capacity == 0
                    || self.state.position_capacity == 0))
        {
            return Err(DispatchError::Hip(
                "indexed attention has invalid geometry".into(),
            ));
        }
        let index_width = checked_mul(
            self.index_heads
                .checked_add(self.index_kv_heads)
                .ok_or_else(|| {
                    DispatchError::Hip("indexed attention index width overflows".into())
                })?,
            self.index_dim,
            "indexed attention index width",
        )?;
        let index_kv_width = checked_mul(
            self.index_kv_heads,
            self.index_dim,
            "indexed attention index KV",
        )?;
        self.validate_rotary()?;
        let q_width = checked_mul(self.heads, self.head_dim, "indexed attention Q width")?;
        let qgate_width = checked_mul(2, q_width, "indexed attention Q/G width")?;
        let kv_width = checked_mul(self.kv_heads, self.head_dim, "indexed attention KV width")?;
        let rows_index = checked_mul(self.rows, index_width, "indexed attention index rows")?;
        let rows_q = checked_mul(self.rows, q_width, "indexed attention Q rows")?;
        let rows_kv = checked_mul(self.rows, kv_width, "indexed attention KV rows")?;
        let rows_input = checked_mul(self.rows, self.input_width, "indexed attention input")?;
        if core
            && (self.state.position > self.state.position_capacity
                || self.state.full_len > self.state.full_capacity
                || self.state.raw_len > self.state.raw_capacity
                || self.state.pooled_len > self.state.pooled_capacity
                || self.state.selected_len > self.state.selected_capacity
                || self.state.position != self.state.full_len
                || self.state.position != self.state.raw_len
                || self.state.pooled_len != self.state.position / self.compress
                || self.state.selected_len > self.state.position
                || self.rows > self.state.position_capacity - self.state.position)
        {
            return Err(DispatchError::Hip(
                "indexed attention state length/capacity mismatch".into(),
            ));
        }
        if inproj {
            require_tensor(
                self.input,
                rows_input,
                DType::F32,
                "indexed attention input",
            )?;
        }
        if inproj || core {
            require_tensor(
                self.index_scratch,
                rows_index,
                DType::F32,
                "indexed attention index scratch",
            )?;
            require_tensor(
                self.qgate_scratch,
                checked_mul(self.rows, qgate_width, "indexed attention q/g rows")?,
                DType::F32,
                "indexed attention q/g scratch",
            )?;
            require_tensor(
                self.k_scratch,
                rows_kv,
                DType::F32,
                "indexed attention k scratch",
            )?;
            require_tensor(
                self.v_scratch,
                rows_kv,
                DType::F32,
                "indexed attention v scratch",
            )?;
        }
        if core || outproj {
            require_tensor(
                self.qsa_output,
                rows_q,
                DType::F32,
                "indexed attention output scratch",
            )?;
        }
        if outproj {
            require_tensor(
                self.attention_output,
                checked_mul(
                    self.rows,
                    self.input_width,
                    "indexed attention projected rows",
                )?,
                DType::F32,
                "indexed attention projected output",
            )?;
        }
        if core {
            let kv_row_units = self.state.format.kv_row_units(self.kv_heads, self.head_dim);
            if !self.state.format.supports(self.kv_heads, self.head_dim) {
                return Err(DispatchError::Hip(format!(
                    "indexed attention {} K/V does not support {} KV heads x {}",
                    self.state.format.name(),
                    self.kv_heads,
                    self.head_dim
                )));
            }
            require_tensor(
                self.state.full_keys,
                checked_mul(
                    self.state.full_capacity,
                    kv_row_units,
                    "indexed attention full keys",
                )?,
                self.state.format.kv_dtype(),
                "indexed attention full keys",
            )?;
            require_tensor(
                self.state.full_values,
                checked_mul(
                    self.state.full_capacity,
                    kv_row_units,
                    "indexed attention full values",
                )?,
                self.state.format.kv_dtype(),
                "indexed attention full values",
            )?;
            require_tensor(
                self.state.raw_index_keys,
                checked_mul(
                    self.state.raw_capacity,
                    index_kv_width,
                    "indexed attention raw keys",
                )?,
                self.state.format.index_dtype(),
                "indexed attention raw keys",
            )?;
            require_tensor(
                self.state.pooled_keys,
                checked_mul(
                    self.state.pooled_capacity,
                    index_kv_width,
                    "indexed attention pooled keys",
                )?,
                self.state.format.index_dtype(),
                "indexed attention pooled keys",
            )?;
            if let Some(arena) = &self.state.f16_arena {
                let arena_elements = checked_mul(
                    rdna_compute::tensor_ops::qsa_f16_arena_rows(self.state.full_capacity),
                    kv_width,
                    "indexed attention F16 arena",
                )?;
                if self.state.format != QsaKvFormat::F32 || self.head_dim != 256 {
                    return Err(DispatchError::Hip(
                        "indexed attention F16 arena needs the F32 K/V state and head_dim 256"
                            .into(),
                    ));
                }
                require_tensor(
                    arena.keys,
                    arena_elements,
                    DType::F16,
                    "indexed attention F16 arena keys",
                )?;
                require_tensor(
                    arena.values,
                    arena_elements,
                    DType::F16,
                    "indexed attention F16 arena values",
                )?;
            }
            if let Some(mirror) = &self.state.pooled_bf16 {
                require_tensor(
                    mirror.mirror,
                    checked_mul(
                        self.state.pooled_capacity,
                        index_kv_width,
                        "indexed attention pooled BF16 mirror",
                    )?,
                    DType::BF16,
                    "indexed attention pooled BF16 mirror",
                )?;
            }
            require_tensor(
                self.state.selected_indices,
                checked_mul(
                    self.state.selected_capacity,
                    std::mem::size_of::<i32>(),
                    "indexed attention selected bytes",
                )?,
                DType::Raw,
                "indexed attention selected",
            )?;
            require_tensor(
                self.selected_scratch,
                checked_mul(
                    checked_mul(
                        self.rows,
                        self.state.selected_capacity,
                        "indexed attention selected rows",
                    )?,
                    std::mem::size_of::<i32>(),
                    "indexed attention selected scratch bytes",
                )?,
                DType::Raw,
                "indexed attention selected scratch",
            )?;
            require_tensor(
                self.indexer_q_norm,
                self.index_dim,
                DType::BF16,
                "indexed attention index Q norm",
            )?;
            require_tensor(
                self.indexer_k_norm,
                self.index_dim,
                DType::BF16,
                "indexed attention index K norm",
            )?;
            require_tensor(
                self.q_norm,
                self.head_dim,
                DType::BF16,
                "indexed attention Q norm",
            )?;
            require_tensor(
                self.k_norm,
                self.head_dim,
                DType::BF16,
                "indexed attention K norm",
            )?;
        }
        require_tensor(
            self.bf16_scratch,
            q_width.max(kv_width),
            DType::BF16,
            "indexed attention BF16 scratch",
        )?;
        if inproj {
            require_weight(
                &self.indexer_qk,
                index_width,
                self.input_width,
                "indexed attention index projection",
            )?;
            require_weight(
                &self.q,
                qgate_width,
                self.input_width,
                "indexed attention q projection",
            )?;
            require_weight(
                &self.k,
                kv_width,
                self.input_width,
                "indexed attention k projection",
            )?;
            require_weight(
                &self.v,
                kv_width,
                self.input_width,
                "indexed attention v projection",
            )?;
        }
        if outproj {
            require_weight(
                &self.output,
                self.input_width,
                q_width,
                "indexed attention output projection",
            )?;
        }
        Ok(())
    }

    /// Host bookkeeping for a successful QSA step, shared by HIP and retained
    /// replay. The architecture commits it only after the forward succeeds.
    pub fn next_lengths(&self) -> Result<(usize, usize, usize, usize, usize), DispatchError> {
        self.next_lengths_after(self.rows)
    }

    /// [`Self::next_lengths`] after only the first `rows` rows of this op: the
    /// state a launch over those rows commits (a session snapshot row inside a
    /// bank, or the start of the window after it).
    pub fn next_lengths_after(
        &self,
        rows: usize,
    ) -> Result<(usize, usize, usize, usize, usize), DispatchError> {
        let final_position = self
            .state
            .position
            .checked_add(rows)
            .ok_or_else(|| DispatchError::Hip("indexed attention position overflows".into()))?;
        if self.compress == 0 {
            return Err(DispatchError::Hip(
                "indexed attention compress is zero".into(),
            ));
        }
        let complete = final_position / self.compress;
        let selected_len = match self.mode {
            IndexedAttentionMode::Full => {
                let budget_blocks = self.budget / self.compress;
                (budget_blocks.min(complete) * self.compress + final_position
                    - complete * self.compress)
                    .min(self.state.selected_capacity)
            }
            IndexedAttentionMode::AppendOnly => 0,
            // The reuse kernel keeps the prior rows (all before this
            // position) and appends this one; a dropped row pads with -1,
            // which the attention skips exactly.
            IndexedAttentionMode::ReuseSelection { .. } => (self.state.selected_len + 1)
                .min(self.state.selected_capacity)
                .min(final_position),
        };
        Ok((
            final_position,
            final_position,
            complete,
            selected_len,
            final_position,
        ))
    }
}

/// Developer observer of the QSA reference harness: called with `(gpu, qsa
/// slot, op)` right after a QSA step's projections, so the op's index / query
/// + gate / K / V scratch holds the raw projected rows and the cache, pool
/// and selection are still the step's pre-prologue state.
pub type QsaProjectionHook =
    Box<dyn FnMut(&mut Gpu, usize, &IndexedAttentionOp<'_>) -> Result<(), String>>;

thread_local! {
    // Const-initialized `None`: no allocation, one thread-local read per QSA
    // step while no harness installed a hook.
    static QSA_PROJECTION_HOOK: std::cell::RefCell<Option<(usize, QsaProjectionHook)>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear) this thread's [`QsaProjectionHook`]; the slot counter
/// restarts at 0.
pub fn set_qsa_projection_hook(hook: Option<QsaProjectionHook>) {
    QSA_PROJECTION_HOOK.with(|cell| *cell.borrow_mut() = hook.map(|hook| (0, hook)));
}

/// Whether this thread has a [`QsaProjectionHook`] installed.
pub fn qsa_projection_hook_installed() -> bool {
    QSA_PROJECTION_HOOK.with(|cell| cell.borrow().is_some())
}

/// Restart the hook's QSA slot counter (a forward's first QSA step is slot 0).
pub fn reset_qsa_projection_slot() {
    QSA_PROJECTION_HOOK.with(|cell| {
        if let Some((slot, _)) = cell.borrow_mut().as_mut() {
            *slot = 0;
        }
    });
}

fn qsa_projection_hook_run(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
) -> Result<(), DispatchError> {
    QSA_PROJECTION_HOOK.with(|cell| match cell.borrow_mut().as_mut() {
        None => Ok(()),
        Some((slot, hook)) => {
            let index = *slot;
            *slot += 1;
            hook(gpu, index, op).map_err(DispatchError::Hip)
        }
    })
}

/// The output projection of the full-mode body: the paired HC write rides its
/// epilogue when it can (`fused`), else the projection lands in
/// `attention_output`.
fn qsa_project_output(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    qsa_output_batch: &GpuTensor,
    hc: Option<&HyperWriteOp<'_>>,
    fused: &mut bool,
) -> Result<(), DispatchError> {
    let fused_hc = match hc {
        Some(write) => project_output_into_hyper_write(
            gpu,
            &op.output,
            qsa_output_batch,
            op.rows,
            Some(op.rotation),
            op.trunk_a4 & TrunkFamily::QsaO.bit() != 0,
            write,
        )?,
        None => false,
    };
    if fused_hc {
        *fused = true;
    } else {
        project_trunk(
            gpu,
            qsa_output_batch,
            op.rows,
            op.rotation,
            &[(
                &op.output,
                &view(op.attention_output, 0, op.rows * op.output.m),
            )],
            &[op.trunk_a4 & TrunkFamily::QsaO.bit() != 0],
            None,
            false,
        )?;
    }
    Ok(())
}

pub fn execute_indexed_attention(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
) -> Result<(), DispatchError> {
    execute_indexed_attention_hc(gpu, op, None, &mut false)
}

/// [`execute_indexed_attention`] whose output projection may carry the paired
/// HC write `hc` (see [`execute_gated_delta_net_hc`]).
pub fn execute_indexed_attention_hc(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    hc: Option<&HyperWriteOp<'_>>,
    fused: &mut bool,
) -> Result<(), DispatchError> {
    execute_indexed_attention_hc_impl(gpu, op, hc, fused, None, None)
}

pub fn execute_indexed_attention_arena(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    arenas: QsaTrunkArenas<'_>,
) -> Result<(), DispatchError> {
    if op.phase != MixerPhase::Core {
        return Err(DispatchError::Hip(
            "QSA trunk arena requires Core phase".into(),
        ));
    }
    op.validate_for_gpu(gpu)?;
    execute_indexed_attention_hc_impl(gpu, op, None, &mut false, Some(arenas), None)
}

/// Native input projections only, skipping idx/q/k/v covered by the trunk.
pub fn execute_indexed_attention_uncovered(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    covered: [bool; 4],
) -> Result<(), DispatchError> {
    execute_indexed_attention_hc_impl(gpu, op, None, &mut false, None, Some(covered))
}

fn qsa_norm_source(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRopeBatch<'_>,
    src: Option<Bf16Arena<'_>>,
) -> Result<(), DispatchError> {
    hip(match src {
        Some(src) => indexed_attention_norm_rope_batch_arena(gpu, p, src),
        None => indexed_attention_norm_rope_batch(gpu, p),
    })
}

fn execute_indexed_attention_hc_impl(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
    hc: Option<&HyperWriteOp<'_>>,
    fused: &mut bool,
    arenas: Option<QsaTrunkArenas<'_>>,
    projection_only: Option<[bool; 4]>,
) -> Result<(), DispatchError> {
    op.check_phase()?;
    if op.mrope.is_some() && arenas.is_some() {
        return Err(DispatchError::Hip(
            "image mRoPE cannot consume NPU projection arenas".into(),
        ));
    }
    let norm_rope = |gpu: &mut Gpu, p: &IndexedAttentionNormRopeBatch<'_>| match op.mrope {
        Some(mrope) => rdna_compute::qsa_mrope::norm_mrope(gpu, p, mrope),
        None => indexed_attention_norm_rope_batch(gpu, p),
    };
    let pool_rope = |gpu: &mut Gpu, p: &IndexedAttentionPoolRope<'_>| match op.mrope {
        Some(mrope) => rdna_compute::qsa_mrope::pool_mrope(gpu, p, mrope),
        None => indexed_attention_pool_rope_incremental(gpu, p),
    };
    let phase = if projection_only.is_some() {
        MixerPhase::InProj
    } else {
        op.phase
    };
    let arenas = core_arena(phase, arenas);
    let index_width = (op.index_heads + op.index_kv_heads) * op.index_dim;
    let index_q_width = op.index_heads * op.index_dim;
    let index_kv_width = op.index_kv_heads * op.index_dim;
    let q_width = op.heads * op.head_dim;
    let kv_width = op.kv_heads * op.head_dim;
    let initial_position = op.state.position;
    let rows = op.rows;
    let (_, _, complete, selected_len, _) = op.next_lengths()?;
    let index_batch = view(op.index_scratch, 0, op.rows * index_width);
    let qgate_batch = view(op.qgate_scratch, 0, op.rows * 2 * q_width);
    let k_batch = view(op.k_scratch, 0, op.rows * kv_width);
    let v_batch = view(op.v_scratch, 0, op.rows * kv_width);
    let qsa_output_batch = view(op.qsa_output, 0, op.rows * q_width);
    let selected_batch = view(
        op.selected_scratch,
        0,
        op.rows * op.state.selected_capacity * std::mem::size_of::<i32>(),
    );
    let all = [
        (&op.indexer_qk, &index_batch),
        (&op.q, &qgate_batch),
        (&op.k, &k_batch),
        (&op.v, &v_batch),
    ];
    let without_q = [all[0], all[2], all[3]];
    let projections: &[_] = match op.mode {
        IndexedAttentionMode::AppendOnly => &without_q,
        _ => &all,
    };
    let a4_all = [
        op.trunk_a4 & TrunkFamily::QsaIdx.bit() != 0,
        op.trunk_a4 & TrunkFamily::QsaQ.bit() != 0,
        op.trunk_a4 & TrunkFamily::QsaK.bit() != 0,
        op.trunk_a4 & TrunkFamily::QsaV.bit() != 0,
    ];
    let a4_without_q = [a4_all[0], a4_all[2], a4_all[3]];
    let a4: &[bool] = match op.mode {
        IndexedAttentionMode::AppendOnly => &a4_without_q,
        _ => &a4_all,
    };
    // Fused q/k RMSNorm + RoPE in the int8 PM GEMM epilogue (QSA prefill's
    // batched-launch branch, byte-identical to its two `norm_rope` launches).
    // Only a whole-phase eager call with no NPU arena source and no
    // projection hook (which reads the raw projections) takes it; the
    // projection records which of q / k actually ran fused.
    let yarn = op.rope.yarn(op.rotary_dim);
    let table_floats = rows * QSA_NORM_EPI_ROTARY_DIM;
    let fuse_candidate = op.mrope.is_none()
        && phase == MixerPhase::Whole
        && arenas.is_none()
        && projection_only.is_none()
        && !qsa_projection_hook_installed()
        && gpu.route_rows(rows) > gpu.few_row_limit()
        && op.head_dim == QSA_NORM_EPI_HEAD_DIM
        && op.rotary_dim == QSA_NORM_EPI_ROTARY_DIM
        && op.input.dtype == DType::F32
        && op.qsa_output.dtype == DType::F32
        && op.qsa_output.numel() >= table_floats;
    let q_candidate = fuse_candidate
        && !matches!(op.mode, IndexedAttentionMode::AppendOnly)
        && op.q.m == 2 * q_width
        && mq6_i8_route(gpu, &op.q, rows)
        && gpu.gemm_mq6g256v2_i8_norm_rope_applies(op.q.m, op.q.k, rows, &qgate_batch, 512);
    let k_candidate = fuse_candidate
        && op.k.m == kv_width
        && mq6_i8_route(gpu, &op.k, rows)
        && gpu.gemm_mq6g256v2_i8_norm_rope_applies(op.k.m, op.k.k, rows, &k_batch, 256);
    // The rope table lives in `qsa_output`: it is first written by the
    // attention launch, after both fused GEMMs have consumed the table.
    let rope_table = view(op.qsa_output, 0, table_floats);
    if q_candidate || k_candidate {
        hip(indexed_attention_rope_table(
            gpu,
            &rope_table,
            rows,
            initial_position,
            op.rotary_dim,
            yarn,
        ))?;
    }
    let q_epi = NormRopeEpi {
        norm: op.q_norm,
        head_stride_rows: 512,
        table: &rope_table,
        fused: std::cell::Cell::new(false),
    };
    let k_epi = NormRopeEpi {
        norm: op.k_norm,
        head_stride_rows: 256,
        table: &rope_table,
        fused: std::cell::Cell::new(false),
    };
    let epi_all = [
        None,
        q_candidate.then_some(&q_epi),
        k_candidate.then_some(&k_epi),
        None,
    ];
    let epi_without_q = [epi_all[0], epi_all[2], epi_all[3]];
    let epi: &[Option<&NormRopeEpi<'_>>] = match op.mode {
        IndexedAttentionMode::AppendOnly => &epi_without_q,
        _ => &epi_all,
    };
    if matches!(phase, MixerPhase::Whole | MixerPhase::InProj) {
        project_trunk_masked(
            gpu,
            op.input,
            op.rows,
            op.rotation,
            projections,
            a4,
            None,
            false,
            projection_only.as_ref().map_or(&[][..], |m| &m[..]),
            epi,
        )?;
        if projection_only.is_none() {
            qsa_projection_hook_run(gpu, op)?;
        }
    }
    let (q_fused, k_fused) = (q_epi.fused.get(), k_epi.fused.get());
    if phase == MixerPhase::InProj {
        return Ok(());
    }
    if phase == MixerPhase::OutProj {
        return qsa_project_output(gpu, op, &qsa_output_batch, hc, fused);
    }
    // Admitted eager verify keeps full-width decode prologue/selection
    // arithmetic. Attention binds the actual singleton AR policy separately
    // for each row's absolute position, batching only adjacent equal policies.
    // The prologue's kernel rotates `min(dim, 64)` channels of the queries,
    // keys and indexer; any other width takes the batched launches below.
    let prologue_rotary =
        op.rotary_dim == op.head_dim.min(64) && op.rotary_dim == op.index_dim.min(64);
    // A rewound position means these F32 rows are about to be replaced.
    // Clamp before the append, even if a later launch fails.
    if let Some(arena) = &op.state.f16_arena {
        arena.valid_len.clamp(initial_position);
    }
    if let Some(mirror) = &op.state.pooled_bf16 {
        mirror.valid_blocks.clamp(initial_position / op.compress);
    }
    if let Some(src) = arenas {
        qsa_norm_source(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &index_batch,
                norm: op.indexer_q_norm,
                rows,
                row_stride: index_width,
                heads: op.index_heads,
                head_dim: op.index_dim,
                head_stride: op.index_dim,
                position_start: initial_position,
                rotary_dim: op.rotary_dim,
                yarn,
            },
            src.idx,
        )?;
        let append = IndexedAttentionIndexKeyAppendBatch {
            index_rows: &index_batch,
            raw_index_keys: op.state.raw_index_keys,
            rows,
            index_q_width,
            index_kv_width,
            position_start: initial_position,
        };
        if let Some(idx) = src.idx {
            hip(indexed_attention_index_key_append_batch_arena(
                gpu, &append, idx,
            ))?;
        } else if op.state.format.index_dtype() == DType::BF16 {
            hip(indexed_attention_index_key_append_batch(gpu, &append))?;
        } else {
            hip(gpu.bf16_round_trip_f32_strided(
                &index_batch,
                rows,
                index_q_width,
                index_width,
                index_kv_width,
            ))?;
            let keys = view(
                &index_batch,
                index_q_width,
                rows * index_width - index_q_width,
            );
            hip(gpu.copy_rows_strided_f32(
                &keys,
                op.state.raw_index_keys,
                rows,
                index_kv_width,
                index_width,
                index_kv_width,
                initial_position * index_kv_width,
                Some(index_kv_width),
            ))?;
        }
        qsa_norm_source(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &qgate_batch,
                norm: op.q_norm,
                rows,
                row_stride: 2 * q_width,
                heads: op.heads,
                head_dim: op.head_dim,
                head_stride: 2 * op.head_dim,
                position_start: initial_position,
                rotary_dim: op.rotary_dim,
                yarn,
            },
            src.q,
        )?;
        qsa_norm_source(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &k_batch,
                norm: op.k_norm,
                rows,
                row_stride: kv_width,
                heads: op.kv_heads,
                head_dim: op.head_dim,
                head_stride: op.head_dim,
                position_start: initial_position,
                rotary_dim: op.rotary_dim,
                yarn,
            },
            src.k,
        )?;
        let append = IndexedAttentionCacheAppendBatch {
            key: &k_batch,
            value: &v_batch,
            full_keys: op.state.full_keys,
            full_values: op.state.full_values,
            rows,
            kv_heads: op.kv_heads,
            head_dim: op.head_dim,
            position_start: initial_position,
            format: op.state.format,
        };
        if let Some(v) = src.v {
            hip(indexed_attention_cache_append_batch_arena(
                gpu, &append, None, v,
            ))?;
        } else {
            hip(indexed_attention_cache_append_batch(gpu, &append))?;
        }
    } else if op.mrope.is_none()
        && gpu.route_rows(rows) <= gpu.few_row_limit()
        && op.index_dim <= 256
        && op.head_dim <= 256
        && prologue_rotary
    {
        // Decode / few-row (and wide-verify, grid.y = rows) prologue: the norms,
        // RoPE, cache append and index-key round trip and copy below, in one launch.
        hip(indexed_attention_decode_prologue(
            gpu,
            &IndexedAttentionDecodePrologue {
                index_row: &index_batch,
                qgate: &qgate_batch,
                keys: &k_batch,
                values: &v_batch,
                full_keys: op.state.full_keys,
                full_values: op.state.full_values,
                raw_index_keys: op.state.raw_index_keys,
                index_q_norm: op.indexer_q_norm,
                q_norm: op.q_norm,
                k_norm: op.k_norm,
                index_heads: op.index_heads,
                index_dim: op.index_dim,
                index_kv_width,
                heads: op.heads,
                kv_heads: op.kv_heads,
                head_dim: op.head_dim,
                position: initial_position,
                rows,
                format: op.state.format,
                yarn,
            },
        ))?;
    } else {
        hip(norm_rope(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &index_batch,
                norm: op.indexer_q_norm,
                rows,
                row_stride: index_width,
                heads: op.index_heads,
                head_dim: op.index_dim,
                head_stride: op.index_dim,
                position_start: initial_position,
                rotary_dim: op.rotary_dim,
                yarn,
            },
        ))?;
        if op.state.format.index_dtype() == DType::BF16 {
            // Round in place and append into the BF16 arena, as the decode
            // prologue does.
            hip(indexed_attention_index_key_append_batch(
                gpu,
                &IndexedAttentionIndexKeyAppendBatch {
                    index_rows: &index_batch,
                    raw_index_keys: op.state.raw_index_keys,
                    rows,
                    index_q_width,
                    index_kv_width,
                    position_start: initial_position,
                },
            ))?;
        } else {
            hip(gpu.bf16_round_trip_f32_strided(
                &index_batch,
                rows,
                index_q_width,
                index_width,
                index_kv_width,
            ))?;
            let index_k_batch = view(
                &index_batch,
                index_q_width,
                rows * index_width - index_q_width,
            );
            // The destination row offset travels as a scalar (`= position *
            // index_kv_width`) against the base tensor, and the recorder declares it,
            // so the tape keeps a position-independent pointer and replay
            // re-derives the offset for its own position instead of replaying the
            // capture-position row.
            hip(gpu.copy_rows_strided_f32(
                &index_k_batch,
                op.state.raw_index_keys,
                rows,
                index_kv_width,
                index_width,
                index_kv_width,
                initial_position * index_kv_width,
                Some(index_kv_width),
            ))?;
        }
        if !q_fused {
            hip(norm_rope(
                gpu,
                &IndexedAttentionNormRopeBatch {
                    values: &qgate_batch,
                    norm: op.q_norm,
                    rows,
                    row_stride: 2 * q_width,
                    heads: op.heads,
                    head_dim: op.head_dim,
                    head_stride: 2 * op.head_dim,
                    position_start: initial_position,
                    rotary_dim: op.rotary_dim,
                    yarn,
                },
            ))?;
        }
        if !k_fused {
            hip(norm_rope(
                gpu,
                &IndexedAttentionNormRopeBatch {
                    values: &k_batch,
                    norm: op.k_norm,
                    rows,
                    row_stride: kv_width,
                    heads: op.kv_heads,
                    head_dim: op.head_dim,
                    head_stride: op.head_dim,
                    position_start: initial_position,
                    rotary_dim: op.rotary_dim,
                    yarn,
                },
            ))?;
        }
        hip(indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k_batch,
                value: &v_batch,
                full_keys: op.state.full_keys,
                full_values: op.state.full_values,
                rows,
                position_start: initial_position,
                kv_heads: op.kv_heads,
                head_dim: op.head_dim,
                format: op.state.format,
            },
        ))?;
    }

    // Every QSA launch declares a position-independent shape: the incremental
    // pool grid is `ceil(rows / compress)` (offset by the `first_block` kernarg)
    // and both dynamic-LDS reservations come from the declared capacities while
    // the active lengths stay scalars. Measured bit-identical to the position-derived
    // shapes with no throughput delta (docs/design/qwen4-program-retained-pm4.md).
    if complete > 0 {
        // Pool only the blocks this launch's rows complete: blocks below
        // `position / compress` hold the same kernel's output for raw keys no
        // later row rewrites (a rollback rewinds to a position, and the block
        // containing it is re-pooled). Re-pooling the whole prefix every
        // prefill chunk was quadratic in the context.
        hip(pool_rope(
            gpu,
            &IndexedAttentionPoolRope {
                raw_keys: op.state.raw_index_keys,
                pooled: op.state.pooled_keys,
                norm: Some(op.indexer_k_norm),
                block_count: complete,
                compress: op.compress,
                index_dim: index_kv_width,
                position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                    position_start: initial_position,
                    rows,
                }),
                grid_bound: op.state.pooled_capacity,
                yarn,
            },
        ))?;
    }
    if let Some(mirror) = &op.state.pooled_bf16 {
        hip(rdna_compute::qsa_pooled_mirror::qsa_pooled_bf16_sync(
            gpu,
            op.state.pooled_keys,
            mirror.mirror,
            mirror.valid_blocks,
            complete,
            index_kv_width,
        ))?;
    }
    let selected_len_out = match op.mode {
        IndexedAttentionMode::Full => None,
        IndexedAttentionMode::AppendOnly => return Ok(()),
        IndexedAttentionMode::ReuseSelection { selected_len_out } => Some(selected_len_out),
    };
    // Reuse keeps the persistent selection: no per-row selection or mirror.
    if let Some(selected_len_out) = selected_len_out {
        hip(indexed_attention_reuse_selection(
            gpu,
            &IndexedAttentionReuseSelection {
                selected: op.state.selected_indices,
                selected_len: op.state.selected_len,
                position: initial_position,
                capacity: op.state.selected_capacity,
                selected_len_out,
            },
        ))?;
        // Extent-check metadata only (never a kernel argument): the reuse
        // selection above keeps token ids `<= initial_position`, so the row's
        // visible end is `initial_position + 1`. Computed only under the flag.
        let end_position = if gpu.flags.qwen4_qsa_extent_check {
            initial_position + 1
        } else {
            0
        };
        hip(indexed_attention_attention(
            gpu,
            &IndexedAttentionAttention {
                q_with_gate: &qgate_batch,
                full_keys: op.state.full_keys,
                full_values: op.state.full_values,
                selected: op.state.selected_indices,
                output: &qsa_output_batch,
                n_heads: op.heads,
                n_kv_heads: op.kv_heads,
                head_dim: op.head_dim,
                selected_len,
                full_capacity: op.state.full_capacity,
                end_position,
                format: op.state.format,
            },
        ))?;
        return project_weight(
            gpu,
            &op.output,
            &qsa_output_batch,
            &view(op.attention_output, 0, op.output.m),
            1,
            Some(op.rotation),
        );
    }
    let budget_blocks = op.budget / op.compress;
    let select = IndexedAttentionSelectBatch {
        query: &index_batch,
        pooled: op.state.pooled_keys,
        pooled_bf16: op.state.pooled_bf16.as_ref().and_then(|mirror| {
            (mirror.valid_blocks.valid(complete) >= complete).then_some(mirror.mirror)
        }),
        // Extent-check metadata only; computed solely under the flag.
        pooled_bf16_valid_blocks: if gpu.flags.qwen4_qsa_extent_check {
            op.state
                .pooled_bf16
                .as_ref()
                .map_or(0, |mirror| mirror.valid_blocks.valid(complete))
        } else {
            0
        },
        selected: &selected_batch,
        rows,
        query_row_stride: index_width,
        block_count: complete,
        index_heads: op.index_heads,
        index_dim: op.index_dim,
        budget_blocks,
        compress: op.compress,
        position_start: initial_position,
        capacity: op.state.selected_capacity,
        shape_blocks: op.state.pooled_capacity,
    };
    // The final row's selection also lands in the persistent selected indices
    // (the copy at the end is then skipped).
    let selection_persisted = hip(indexed_attention_select_batch_mirrored(
        gpu,
        &select,
        op.state.selected_indices,
    ))?;
    hip(indexed_attention_attention_batch_with_arena(
        gpu,
        &IndexedAttentionAttentionBatch {
            q_with_gate: &qgate_batch,
            full_keys: op.state.full_keys,
            full_values: op.state.full_values,
            selected: &selected_batch,
            output: &qsa_output_batch,
            rows,
            position_start: initial_position,
            n_heads: op.heads,
            n_kv_heads: op.kv_heads,
            head_dim: op.head_dim,
            budget_blocks,
            compress: op.compress,
            capacity: op.state.selected_capacity,
            full_capacity: op.state.full_capacity,
            format: op.state.format,
            shape_selected: op.state.selected_capacity,
        },
        op.state.f16_arena.as_ref(),
    ))?;
    if phase != MixerPhase::Core {
        qsa_project_output(gpu, op, &qsa_output_batch, hc, fused)?;
    }
    let final_selected = view(
        &selected_batch,
        (op.rows - 1) * op.state.selected_capacity * std::mem::size_of::<i32>(),
        op.state.selected_capacity * std::mem::size_of::<i32>(),
    );
    // A recorded launch, not a `copy_d2d`: the retained tape replays dispatches,
    // so a device copy inside the body would be state the replay cannot
    // reproduce. `copy_f32_buffer` moves the same bytes with an explicit ABI.
    if !selection_persisted {
        hip(gpu.copy_f32_buffer(
            op.state.selected_indices,
            &final_selected,
            op.state.selected_capacity,
        ))?;
    }
    Ok(())
}

/// Grouped causal convolution contract used by PLE-like layers.  Geometry and
/// kernel/dilation are supplied by the architecture; the low-level wrapper
/// does not know a model's fixed layer index or hidden width.
pub struct GroupedDepthwiseOp<'a> {
    /// The HC streams hold BF16 bits for this forward
    /// ([`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
    /// `rows_tensor` already holds the PLE rows as F16 (first `rows * embed_dim`
    /// halves of its storage; `HIPFIRE_QWEN4_PLE_FUSE`'s F16 gather): the key
    /// and value projections read them directly and the fused tail must run.
    /// Only admitted when `embed_dim == hidden`.
    pub rows_f16: bool,
    pub key: WeightRef<'a>,
    pub value: WeightRef<'a>,
    pub norm_key: &'a GpuTensor,
    pub norm_query: &'a GpuTensor,
    pub norm_conv: &'a GpuTensor,
    pub conv: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub streams: &'a GpuTensor,
    pub rows_tensor: &'a GpuTensor,
    pub query: &'a GpuTensor,
    pub key_scratch: &'a GpuTensor,
    pub value_scratch: &'a GpuTensor,
    pub gated: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    /// Width of the staged PLE rows: the key/value projections' K
    /// (`ple_embed_dim`; Flash-Next: equal to `hidden`).
    pub embed_dim: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub epsilon: f32,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
}

impl GroupedDepthwiseOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    fn validate_layout(&self) -> Result<(), DispatchError> {
        if self.rows == 0
            || self.embed_dim == 0
            || self.branches == 0
            || self.hidden == 0
            || self.kernel_size == 0
            || self.dilation == 0
        {
            return Err(DispatchError::Hip(
                "grouped depthwise operation has invalid geometry".into(),
            ));
        }
        let channels = checked_mul(self.branches, self.hidden, "grouped channels")?;
        let history_rows =
            checked_mul(self.kernel_size - 1, self.dilation, "grouped history rows")?;
        let rows_channels = checked_mul(self.rows, channels, "grouped row channels")?;
        let rows_hidden = checked_mul(self.rows, self.hidden, "grouped row hidden")?;
        let rows_embed = checked_mul(self.rows, self.embed_dim, "grouped row embeddings")?;
        let conv_elements = checked_mul(channels, self.kernel_size, "grouped convolution")?;
        let state_elements = checked_mul(history_rows, channels, "grouped state")?;
        require_tensor(self.streams, rows_channels, DType::F32, "grouped streams")?;
        require_tensor(self.query, rows_channels, DType::F32, "grouped query")?;
        require_tensor(
            self.key_scratch,
            rows_channels,
            DType::F32,
            "grouped key scratch",
        )?;
        require_tensor(self.gated, rows_channels, DType::F32, "grouped gated")?;
        require_tensor(self.normed, rows_channels, DType::F32, "grouped normed")?;
        require_tensor(self.output, rows_channels, DType::F32, "grouped output")?;
        if self.rows_f16 && self.embed_dim != self.hidden {
            return Err(DispatchError::Hip(
                "grouped PLE F16 rows need the embedding width to equal hidden".into(),
            ));
        }
        require_tensor(self.rows_tensor, rows_embed, DType::F32, "grouped rows")?;
        require_tensor(
            self.value_scratch,
            rows_hidden,
            DType::F32,
            "grouped value scratch",
        )?;
        require_tensor(self.state, state_elements, DType::F32, "grouped state")?;
        require_tensor(self.norm_key, channels, DType::BF16, "grouped key norm")?;
        require_tensor(self.norm_query, channels, DType::BF16, "grouped query norm")?;
        require_tensor(
            self.norm_conv,
            channels,
            DType::BF16,
            "grouped convolution norm",
        )?;
        require_tensor(self.conv, conv_elements, DType::BF16, "grouped convolution")?;
        require_weight(&self.key, channels, self.embed_dim, "grouped key")?;
        require_weight(&self.value, self.hidden, self.embed_dim, "grouped value")?;
        Ok(())
    }
}

pub fn execute_grouped_depthwise(
    gpu: &mut Gpu,
    op: &GroupedDepthwiseOp<'_>,
) -> Result<(), DispatchError> {
    if op.rows_f16 && wide_exact_rows(gpu, op.rows) {
        return Err(DispatchError::Hip(
            "row-exact PLE requires the original F32 rows, not prefill F16 staging".into(),
        ));
    }
    let channels = op.branches * op.hidden;
    let streams = view(op.streams, 0, op.rows * channels);
    let query = view(op.query, 0, op.rows * channels);
    let key = view(op.key_scratch, 0, op.rows * channels);
    let value = view(op.value_scratch, 0, op.rows * op.hidden);
    let gated = view(op.gated, 0, op.rows * channels);
    let normed = view(op.normed, 0, op.rows * channels);
    let output = view(op.output, 0, op.rows * channels);
    // `HIPFIRE_QWEN4_PLE_FUSE`: on exact gfx1151 / gfx1201 with BF16 HC streams
    // (never a recorder, retained tape or graph capture) three launches replace
    // the widen / gate / norm / convolution / stream-add chain below, with the
    // same streams and convolution state byte for byte.
    let fused = op.state_bf16
        && gpu.flags.qwen4_ple_fuse_enabled()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && op.branches <= 4
        && op.hidden % 32 == 0
        && op.rows * op.branches * 2 <= op.normed.numel();
    if op.rows_f16 && !fused {
        return Err(DispatchError::Hip(
            "PLE rows were staged as F16 but the fused PLE tail is not admitted".into(),
        ));
    }
    if op.rows_f16 {
        // SAFETY: `rows_tensor` holds `rows * embed_dim` F32 elements
        // (`embed_dim == hidden`, validated), so its first
        // `rows * embed_dim * 2` bytes hold the F16 rows; the view is non-owning.
        let x_f16 = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    op.rows_tensor.buf.as_ptr(),
                    op.rows * op.embed_dim * 2,
                )
            },
            shape: vec![op.rows * op.embed_dim],
            dtype: DType::F16,
        };
        for (weight, out) in [(&op.key, &key), (&op.value, &value)] {
            hip(gpu.gemm_bf16_xf16_f16_wmma(weight.buf, &x_f16, out, weight.m, weight.k, op.rows))?;
        }
    } else {
        project_weight(
            gpu,
            &op.key,
            op.rows_tensor,
            &key,
            op.rows,
            Some(op.rotation),
        )?;
        project_weight(
            gpu,
            &op.value,
            op.rows_tensor,
            &value,
            op.rows,
            Some(op.rotation),
        )?;
    }
    if fused {
        let scalars = view(op.normed, 0, op.rows * op.branches * 2);
        return hip(rdna_compute::grouped_ops::grouped_ple_fused_bf16s(
            gpu,
            &rdna_compute::grouped_ops::GroupedPleFused {
                key: &key,
                value: &value,
                streams: &streams,
                scalars: &scalars,
                norm_key: op.norm_key,
                norm_query: op.norm_query,
                norm_conv: op.norm_conv,
                conv_weight: op.conv,
                state: op.state,
                tokens: op.rows,
                groups: op.branches,
                group_size: op.hidden,
                kernel_size: op.kernel_size,
                dilation: op.dilation,
                epsilon: op.epsilon,
            },
        ));
    }
    // A recorded launch, not a `copy_d2d`: a retained tape replays dispatches, so
    // a device copy inside the body would be state the replay cannot reproduce.
    if op.state_bf16 {
        hip(hc_state_bf16_to_f32(
            gpu,
            &streams,
            &query,
            op.rows * channels,
        ))?;
    } else {
        hip(gpu.copy_f32_buffer(&query, &streams, op.rows * channels))?;
    }
    hip(rdna_compute::grouped_ops::grouped_gate_bf16(
        gpu,
        &rdna_compute::grouped_ops::GroupedGate {
            key: &key,
            query: &query,
            value: &value,
            norm_key: op.norm_key,
            norm_query: op.norm_query,
            gated: &gated,
            rows: op.rows,
            groups: op.branches,
            group_size: op.hidden,
            epsilon: op.epsilon,
        },
    ))?;
    hip(rdna_compute::grouped_ops::grouped_norm_bf16(
        gpu,
        &rdna_compute::grouped_ops::GroupedNorm {
            input: &gated,
            weight: op.norm_conv,
            output: &normed,
            tokens: op.rows,
            groups: op.branches,
            group_size: op.hidden,
            epsilon: op.epsilon,
        },
    ))?;
    hip(
        rdna_compute::grouped_ops::grouped_depthwise_conv_silu_add_bf16(
            gpu,
            &rdna_compute::grouped_ops::GroupedDepthwiseConv {
                gated: &gated,
                normed: &normed,
                conv_weight: op.conv,
                state: op.state,
                output: &output,
                tokens: op.rows,
                channels,
                kernel_size: op.kernel_size,
                dilation: op.dilation,
            },
        ),
    )?;
    if op.state_bf16 {
        return hip(hc_state_bf16_add_f32(
            gpu,
            &streams,
            &output,
            op.rows * channels,
        ));
    }
    hip(gpu.add_f32(&streams, &output, &streams))
}

/// Clear an F32 scratch prefix before a sealed routed execution.
pub struct ClearOp<'a> {
    pub tensor: &'a GpuTensor,
    pub elements: usize,
}

impl ClearOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        require_tensor(self.tensor, self.elements, DType::F32, "clear target")
    }
}

pub fn execute_clear(gpu: &mut Gpu, op: &ClearOp<'_>) -> Result<(), DispatchError> {
    // A recorded launch, not a `hipMemset`: a retained tape replays dispatches,
    // so a memset inside the body would be state the replay cannot reproduce.
    // `zero_f32` writes exact +0.0f, so the result is byte-identical.
    if op.elements > op.tensor.numel() {
        return Err(DispatchError::Hip(format!(
            "clear op declares {} elements but the tensor holds {}",
            op.elements,
            op.tensor.numel()
        )));
    }
    let span = op.tensor.sub_offset(0, op.elements);
    hip(gpu.zero_f32(&span))
}

/// Embedding-table lookup for `rows` token ids (`token_ids`: `rows` i32 bytes).
/// MQ4 v2 tables stage rows in `rotated` (`rows * dim` F32) before FWHT decode.
pub struct EmbeddingOp<'a> {
    pub table: &'a GpuTensor,
    pub rotated: &'a GpuTensor,
    pub token_ids: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingPath {
    Mq4V2,
    Bf16,
    Q8,
    Hfq4G256,
    Hfq4G128,
}

fn embedding_path(dtype: DType) -> Option<EmbeddingPath> {
    match dtype {
        DType::MQ4G256V2 | DType::MQ4G128V2 => Some(EmbeddingPath::Mq4V2),
        DType::BF16 => Some(EmbeddingPath::Bf16),
        DType::Q8_0 => Some(EmbeddingPath::Q8),
        DType::HFQ4G256 => Some(EmbeddingPath::Hfq4G256),
        DType::HFQ4G128 => Some(EmbeddingPath::Hfq4G128),
        _ => None,
    }
}

fn unsupported_embedding(dtype: DType) -> DispatchError {
    DispatchError::Hip(format!(
        "embedding table has unsupported resident dtype {dtype:?}"
    ))
}

impl EmbeddingOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.dim == 0 {
            return Err(DispatchError::Hip("embedding has empty geometry".into()));
        }
        let path = embedding_path(self.table.dtype)
            .ok_or_else(|| unsupported_embedding(self.table.dtype))?;
        let elements = checked_mul(self.rows, self.dim, "embedding rows")?;
        require_tensor(self.output, elements, DType::F32, "embedding output")?;
        if path == EmbeddingPath::Mq4V2 {
            require_tensor(self.rotated, elements, DType::F32, "embedding rotation")?;
        }
        if self.token_ids.buf.size() < checked_mul(self.rows, 4, "embedding token ids")? {
            return Err(DispatchError::Hip(
                "embedding token ids hold fewer than rows i32 values".into(),
            ));
        }
        Ok(())
    }
}

/// BF16 rows are widened directly; packed MQv2 rows use the rotated staging
/// buffer and FWHT decode; Q8 rows are block-decoded in place.
pub fn execute_embedding(gpu: &mut Gpu, op: &EmbeddingOp<'_>) -> Result<(), DispatchError> {
    let path =
        embedding_path(op.table.dtype).ok_or_else(|| unsupported_embedding(op.table.dtype))?;
    let (table, output, ids) = (op.table, op.output, op.token_ids);
    hip(match path {
        EmbeddingPath::Mq4V2 => {
            gpu.embedding_lookup_mq4v2_batched(table, op.rotated, output, ids, op.rows, op.dim)
        }
        EmbeddingPath::Bf16 => {
            gpu.embedding_lookup_bf16_batched(table, output, ids, op.rows, op.dim)
        }
        EmbeddingPath::Q8 => gpu.embedding_lookup_q8_batched(table, output, ids, op.rows, op.dim),
        EmbeddingPath::Hfq4G256 => {
            gpu.embedding_lookup_hfq4g256_batched(table, output, ids, op.rows, op.dim)
        }
        EmbeddingPath::Hfq4G128 => {
            gpu.embedding_lookup_hfq4g128_batched(table, output, ids, op.rows, op.dim)
        }
    })
}

/// Grouped hyper-connection RMS norm: every row of `input` holds `branches`
/// streams of `hidden`, normalized against one `branches * hidden` weight.
pub struct HyperNormOp<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    pub state_bf16: bool,
}

impl HyperNormOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let wide = checked_mul(self.branches, self.hidden, "hyper norm width")?;
        let elements = self.input.numel();
        if wide == 0
            || self.input.dtype != DType::F32
            || self.normalized.dtype != DType::F32
            || elements == 0
            || !elements.is_multiple_of(wide)
            || self.normalized.numel() != elements
            || self.norm_weight.dtype != DType::BF16
            || self.norm_weight.numel() != wide
        {
            return Err(DispatchError::Hip(
                "hyper norm has incompatible shape or dtype".into(),
            ));
        }
        Ok(())
    }
}

pub fn execute_hyper_norm(gpu: &mut Gpu, op: &HyperNormOp<'_>) -> Result<(), DispatchError> {
    hip(hyper_norm(
        gpu,
        &HyperNorm {
            input: op.input,
            norm_weight: op.norm_weight,
            normalized: op.normalized,
            branches: op.branches,
            hidden: op.hidden,
            state_bf16: op.state_bf16,
        },
    ))
}

/// The kernel ladder a [`ProjectOp`] runs on. The ladders give different bits
/// for the same payload (Q8: scalar F32 vs WMMA), so the binding family names
/// the one it was validated on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectLadder {
    /// [`project_weights`]: shared F16 rotation, few-row verify GEMVs, gfx942
    /// BF16 MFMA. Qwen4.
    Trunk,
    /// The dispatcher-entry batched GEMMs, each keeping its own arch routing:
    /// `gemm_q8_0_batched_chunked`, `gemm_hfq4g256` (MQ4G256 after its batched
    /// FWHT), `gemm_f32_batched`, gfx942 BF16 MFMA. LFM2 decode lanes.
    Gemm,
    /// [`Self::Gemm`] for an LM head: the `*_batched_lmhead` GEMMs (HFQ4 /
    /// HFQ6 / HFQ3 payloads and their MQ rotations). Q8, F32 and BF16 keep
    /// the `Gemm` arms. No calibration tap, except the BF16 MFMA arm's.
    GemmLmHead,
}

/// One projection over `rows` row-major input rows on its [`ProjectLadder`],
/// optionally followed by `residual += output` (a separate add, not a fused
/// residual GEMM).
pub struct ProjectOp<'a> {
    pub weight: WeightRef<'a>,
    pub input: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    /// FWHT basis scratch (`rows * k`), required by the rotated payloads.
    pub rotation: Option<&'a GpuTensor>,
    pub ladder: ProjectLadder,
    /// F32 `[rows, m]`, distinct from `output`.
    pub residual: Option<&'a GpuTensor>,
}

impl ProjectOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let (m, k) = (self.weight.m, self.weight.k);
        if self.rows == 0 {
            return Err(DispatchError::Hip("projection has no rows".into()));
        }
        // The `Gemm` ladders admit their payloads in `gemm_ladder_rotation` below.
        if self.ladder == ProjectLadder::Trunk {
            require_weight(&self.weight, m, k, "projection")?;
        }
        let input = checked_mul(self.rows, k, "projection input")?;
        require_tensor(self.input, input, DType::F32, "projection input")?;
        let output = checked_mul(self.rows, m, "projection output")?;
        require_tensor(self.output, output, DType::F32, "projection output")?;
        if let Some(residual) = self.residual {
            require_tensor(residual, output, DType::F32, "projection residual")?;
        }
        let rotated = match self.ladder {
            ProjectLadder::Trunk => !matches!(self.weight.dtype, DType::BF16 | DType::Q8_0),
            ladder => gemm_ladder_rotation(self.weight.dtype, ladder == ProjectLadder::GemmLmHead)?,
        };
        if rotated {
            let rotation = self.rotation.ok_or_else(|| {
                DispatchError::Hip("rotated projection payload needs rotation scratch".into())
            })?;
            require_tensor(rotation, input, DType::F32, "projection rotation")?;
        }
        Ok(())
    }
}

pub fn execute_project(gpu: &mut Gpu, op: &ProjectOp<'_>) -> Result<(), DispatchError> {
    match op.ladder {
        ProjectLadder::Trunk => {
            project_weight(gpu, &op.weight, op.input, op.output, op.rows, op.rotation)?
        }
        ProjectLadder::Gemm | ProjectLadder::GemmLmHead => project_gemm(gpu, op)?,
    }
    match op.residual {
        Some(residual) => hip(gpu.add_inplace_f32(residual, op.output)),
        None => Ok(()),
    }
}

/// Whether the `Gemm` / `GemmLmHead` ladder admits `dtype` and reads it in an
/// FWHT basis.
fn gemm_ladder_rotation(dtype: DType, lm_head: bool) -> Result<bool, DispatchError> {
    match (dtype, lm_head) {
        (DType::F32 | DType::BF16 | DType::Q8_0 | DType::HFQ4G256, _) => Ok(false),
        (DType::HFQ6G256, true) => Ok(false),
        (DType::MQ4G256, _) | (DType::MQ6G256 | DType::MQ3G256, true) => Ok(true),
        _ => Err(DispatchError::UnsupportedVariant {
            family: "layer-operations",
            variant: "gemm-ladder-payload",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// The `Gemm` / `GemmLmHead` arms of [`execute_project`]. The calibration tap
/// sees the pre-rotation activation (the AWQ / Hessian basis).
fn project_gemm(gpu: &mut Gpu, op: &ProjectOp<'_>) -> Result<(), DispatchError> {
    let (w, rows, y) = (&op.weight, op.rows, op.output);
    let (m, k) = (w.m, w.k);
    let lm_head = op.ladder == ProjectLadder::GemmLmHead;
    let rotated = gemm_ladder_rotation(w.dtype, lm_head)?;
    if w.dtype == DType::BF16 {
        return project_g942_mfma(gpu, op.input, rows, op.rotation, &[(w, y)], &[0], k);
    }
    if !lm_head {
        gpu.maybe_capture_activation(w.buf, op.input, rows, k);
    }
    let scratch = if rotated {
        let rotation = op.rotation.ok_or(DispatchError::UnsupportedVariant {
            family: "layer-operations",
            variant: "rotation-scratch-absent",
            arch: "",
            quant: "quantized",
        })?;
        let scratch = view(rotation, 0, checked_mul(rows, k, "rotation scratch")?);
        hip(crate::pipeline::batched::rotate_x_mq_batched_for(
            gpu, w, op.input, &scratch, k, rows,
        ))?;
        Some(scratch)
    } else {
        None
    };
    let x = scratch.as_ref().unwrap_or(op.input);
    hip(match (w.dtype, lm_head) {
        (DType::F32, _) => gpu.gemm_f32_batched(w.buf, x, y, m, k, rows),
        (DType::Q8_0, _) => gpu.gemm_q8_0_batched_chunked(w.buf, x, y, m, k, rows),
        (DType::HFQ4G256 | DType::MQ4G256, false) => gpu.gemm_hfq4g256(w.buf, x, y, m, k, rows),
        (DType::HFQ4G256 | DType::MQ4G256, true) => {
            gpu.gemm_hfq4g256_batched_lmhead(w.buf, x, y, m, k, rows)
        }
        (DType::HFQ6G256 | DType::MQ6G256, true) => {
            gpu.gemm_hfq6g256_batched_lmhead(w.buf, x, y, m, k, rows)
        }
        (DType::MQ3G256, true) => gpu.gemm_hfq3g256_batched_lmhead(w.buf, x, y, m, k, rows),
        _ => unreachable!("gemm_ladder_rotation admits only the arms above"),
    })
}

/// Largest causal window the gated short-convolution kernel holds per thread.
pub const GATED_SHORT_CONV_MAX_KERNEL: usize = 8;

/// Double-gated causal short convolution over `rows` independent lanes (the
/// LFM2 LIV operator).  Row `r` of `bcx` holds `[b | c | x]`, each `channels`
/// wide; the op writes `c ⊙ conv(b ⊙ x)` into row `r` of `output` and advances
/// lane `r`'s causal history in `state` (`[rows, channels, kernel_size - 1]`,
/// oldest first).  `weight` is the depthwise filter `[channels, kernel_size]`,
/// last tap on the current input.
pub struct GatedShortConvOp<'a> {
    pub bcx: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub weight: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub channels: usize,
    pub kernel_size: usize,
}

impl GatedShortConvOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.channels == 0 {
            return Err(DispatchError::Hip(
                "gated short conv has empty geometry".into(),
            ));
        }
        if !(2..=GATED_SHORT_CONV_MAX_KERNEL).contains(&self.kernel_size) {
            return Err(DispatchError::Hip(format!(
                "gated short conv kernel size {} outside 2..={GATED_SHORT_CONV_MAX_KERNEL}",
                self.kernel_size
            )));
        }
        let lanes = checked_mul(self.rows, self.channels, "gated short conv lanes")?;
        require_tensor(
            self.bcx,
            checked_mul(lanes, 3, "gated short conv input")?,
            DType::F32,
            "gated short conv input",
        )?;
        require_tensor(
            self.state,
            checked_mul(lanes, self.kernel_size - 1, "gated short conv state")?,
            DType::F32,
            "gated short conv state",
        )?;
        require_tensor(
            self.weight,
            checked_mul(self.channels, self.kernel_size, "gated short conv weight")?,
            DType::F32,
            "gated short conv weight",
        )?;
        require_tensor(self.output, lanes, DType::F32, "gated short conv output")
    }
}

pub fn execute_gated_short_conv(
    gpu: &mut Gpu,
    op: &GatedShortConvOp<'_>,
) -> Result<(), DispatchError> {
    hip(gpu.conv1d_gated_decode_f32(
        op.bcx,
        op.state,
        op.weight,
        op.output,
        op.rows,
        op.channels,
        op.kernel_size,
    ))
}

/// Lane-isolated Q8 attention for `rows` independent decode lanes: row `r`
/// writes its K/V at `positions[r]` into lane `r`'s slice
/// (`[r * lane_capacity, (r + 1) * lane_capacity)` positions) of the Q8 caches,
/// then attends over that slice only.
pub struct LaneAttentionOp<'a> {
    /// F32 `[rows, n_heads * head_dim]`, RoPE'd.
    pub q: &'a GpuTensor,
    /// F32 `[rows, n_kv_heads * head_dim]` each, RoPE'd K.
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    /// `rows` I32 positions.
    pub positions: &'a GpuTensor,
    /// F32 `[rows, n_heads * head_dim]`.
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub lane_capacity: usize,
    pub max_ctx_len: usize,
}

impl LaneAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.lane_capacity == 0 || self.head_dim == 0 {
            return Err(DispatchError::Hip(
                "lane attention has empty geometry".into(),
            ));
        }
        let q = checked_mul(self.rows, self.n_heads * self.head_dim, "lane attention q")?;
        let kv = checked_mul(
            self.rows,
            self.n_kv_heads * self.head_dim,
            "lane attention kv",
        )?;
        require_tensor(self.q, q, DType::F32, "lane attention q")?;
        require_tensor(self.k, kv, DType::F32, "lane attention k")?;
        require_tensor(self.v, kv, DType::F32, "lane attention v")?;
        require_tensor(self.output, q, DType::F32, "lane attention output")?;
        if self.positions.numel() < self.rows {
            return Err(DispatchError::Hip(format!(
                "lane attention: {} positions for {} rows",
                self.positions.numel(),
                self.rows
            )));
        }
        Ok(())
    }
}

pub fn execute_lane_attention(
    gpu: &mut Gpu,
    op: &LaneAttentionOp<'_>,
) -> Result<(), DispatchError> {
    for (cache, src) in [(op.k_cache, op.k), (op.v_cache, op.v)] {
        hip(gpu.kv_cache_write_q8_0_independent(
            cache,
            src,
            op.positions,
            op.n_kv_heads,
            op.head_dim,
            op.rows,
            op.lane_capacity,
        ))?;
    }
    hip(gpu.attention_q8_0_kv_independent(
        op.q,
        op.k_cache,
        op.v_cache,
        op.output,
        op.positions,
        op.n_heads,
        op.n_kv_heads,
        op.head_dim,
        op.lane_capacity,
        op.max_ctx_len,
        op.rows,
    ))
}

/// `output[r] = rows_input[r] + row` for every one of `rows` rows of `width`.
pub struct BroadcastAddOp<'a> {
    pub rows_input: &'a GpuTensor,
    pub row: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub width: usize,
}

impl BroadcastAddOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0 || self.width == 0 {
            return Err(DispatchError::Hip(
                "broadcast add has empty geometry".into(),
            ));
        }
        let elements = checked_mul(self.rows, self.width, "broadcast add rows")?;
        require_tensor(self.rows_input, elements, DType::F32, "broadcast add input")?;
        require_tensor(self.output, elements, DType::F32, "broadcast add output")?;
        require_tensor(self.row, self.width, DType::F32, "broadcast add row")
    }
}

pub fn execute_broadcast_add(gpu: &mut Gpu, op: &BroadcastAddOp<'_>) -> Result<(), DispatchError> {
    let row = view(op.row, 0, op.width);
    for r in 0..op.rows {
        let offset = r * op.width;
        hip(gpu.add_f32(
            &view(op.rows_input, offset, op.width),
            &row,
            &view(op.output, offset, op.width),
        ))?;
    }
    Ok(())
}

/// Final hyper read uses the same operation contract as a regular read; this
/// helper only supplies the projection-free final output shape.
pub fn execute_final_hyper(gpu: &mut Gpu, op: &HyperReadOp<'_>) -> Result<(), DispatchError> {
    op.validate_for_gpu(gpu)?;
    execute_hyper_read(gpu, op)
}

pub fn validate_lm_head(
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
) -> Result<(), DispatchError> {
    if rows == 0 || requested_rows == 0 || requested_rows > rows || weight.m == 0 || weight.k == 0 {
        return Err(DispatchError::Hip("LM-head geometry is invalid".into()));
    }
    if !matches!(weight.dtype, DType::BF16 | DType::F32) {
        return Err(DispatchError::UnsupportedVariant {
            family: "lm-head",
            variant: "dtype",
            arch: "",
            quant: "unsupported",
        });
    }
    require_tensor(hidden_batch, rows * weight.k, DType::F32, "LM-head hidden")?;
    require_tensor(
        logits,
        requested_rows * weight.m,
        DType::F32,
        "LM-head output",
    )
}

pub fn execute_lm_head(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
) -> Result<(), DispatchError> {
    validate_lm_head(weight, hidden_batch, logits, rows, requested_rows)?;
    // Under the batch-invariant prefill profile a one-row head is the final-row
    // head's GEMV (never the routed GEMM), so a one-row forward and a longer
    // forward's final row produce the same bytes.
    if requested_rows == rows && (rows > 1 || !gpu.prefill_invariant_scope) {
        match weight.dtype {
            DType::BF16 => project_weight(gpu, weight, hidden_batch, logits, rows, None),
            DType::F32 => hip(gpu.gemm_f32_batched(
                weight.buf,
                hidden_batch,
                logits,
                weight.m,
                weight.k,
                rows,
            )),
            _ => unreachable!(),
        }
    } else if requested_rows == 1 {
        let input = view(hidden_batch, (rows - 1) * weight.k, weight.k);
        match weight.dtype {
            // gfx942: the one-row MFMA route (or the `HIPFIRE_G942_TRUNK_GEMV=1`
            // GEMV inside it); elsewhere the final-row GEMV stays.
            DType::BF16 if gpu.arch_caps.is_gfx942() => {
                project_weight(gpu, weight, &input, logits, 1, None)
            }
            DType::BF16 => hip(gpu.gemv_bf16_xf32(weight.buf, &input, logits, weight.m, weight.k)),
            DType::F32 => {
                hip(gpu.gemm_f32_batched(weight.buf, &input, logits, weight.m, weight.k, 1))
            }
            _ => unreachable!(),
        }
    } else {
        Err(DispatchError::Hip(
            "LM-head supports all rows or final row only".into(),
        ))
    }
}

pub fn execute_argmax(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    indices: &GpuTensor,
    rows: usize,
    vocab: usize,
) -> Result<(), DispatchError> {
    hip(argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits,
            indices,
            rows,
            vocab,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_route_uses_the_admitted_eager_limit_not_storage_capacity() {
        for limit in [8, 9, 16, 32, 63, 64] {
            for rows in 1..=65 {
                assert_eq!(wide_exact_route(rows, limit), rows > 8 && rows <= limit);
            }
        }
        // CPU/unknown-arch and recorder limits never inherit a storage cap64.
        for limit in [0, 1, 4, 8] {
            for rows in 1..=64 {
                assert!(!wide_exact_route(rows, limit));
            }
        }
        assert!(!wide_exact_route(usize::MAX, 64));
    }

    #[test]
    fn hc_partitions_follow_the_singleton_descriptor_not_token_count() {
        for rows in 1..=64 {
            for k in [32, 320, 2560, 10240] {
                assert_eq!(hc_decode_partitions(DType::BF16, k, rows, 64), Some(4));
            }
            for k in [256, 2560, 10240] {
                assert_eq!(hc_decode_partitions(DType::Q8_0, k, rows, 64), Some(8));
            }
            assert_eq!(
                hc_decode_partitions(DType::BF16, 10240, rows, 8),
                (rows <= 8).then_some(4),
            );
        }
        assert_eq!(hc_decode_partitions(DType::BF16, 33, 64, 64), None);
        assert_eq!(hc_decode_partitions(DType::Q8_0, 320, 64, 64), None);
        assert_eq!(hc_decode_partitions(DType::F32, 10240, 64, 64), None);
        assert_eq!(hc_decode_partitions(DType::BF16, 10240, 5, 4), None);
    }

    #[test]
    fn requant_experiments_do_not_leak_wmma_into_admitted_wide_verify() {
        for dtype in [DType::MQ2G256V2, DType::MQ3G256V2, DType::MQ5G256V2] {
            for rows in 1..=64 {
                assert_eq!(wide_projection_uses_singletons(dtype, rows, 64), rows > 8);
                assert!(!wide_projection_uses_singletons(dtype, rows, 8));
            }
        }
        for dtype in [DType::BF16, DType::Q8_0, DType::MQ4G256V2, DType::MQ6G256V2] {
            assert!(!wide_projection_uses_singletons(dtype, 64, 64));
        }
    }

    fn tensor(elements: usize, dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = vec![elements];
        tensor.dtype = dtype;
        tensor
    }

    struct GdnFixture {
        qkv: GpuTensor,
        conv: GpuTensor,
        in_proj_a: GpuTensor,
        in_proj_b: GpuTensor,
        a_log: GpuTensor,
        dt_bias: GpuTensor,
        z: GpuTensor,
        norm: GpuTensor,
        output: GpuTensor,
        recurrent: GpuTensor,
        conv_state: GpuTensor,
        projection: GpuTensor,
        projection2: GpuTensor,
        a: GpuTensor,
        b: GpuTensor,
        gate: GpuTensor,
        beta: GpuTensor,
        recurrent_output: GpuTensor,
        bf16_scratch: GpuTensor,
        rotation: GpuTensor,
        z_output: GpuTensor,
        output_scratch: GpuTensor,
        input: GpuTensor,
        output_tensor: GpuTensor,
    }

    impl GdnFixture {
        fn new() -> Self {
            Self {
                qkv: tensor(8, DType::BF16),
                conv: tensor(8, DType::BF16),
                in_proj_a: tensor(2, DType::BF16),
                in_proj_b: tensor(2, DType::BF16),
                a_log: tensor(1, DType::BF16),
                dt_bias: tensor(1, DType::BF16),
                z: tensor(4, DType::BF16),
                norm: tensor(2, DType::BF16),
                output: tensor(4, DType::BF16),
                recurrent: tensor(2, DType::F32),
                conv_state: tensor(4, DType::F32),
                projection: tensor(4, DType::F32),
                projection2: tensor(4, DType::F32),
                a: tensor(1, DType::F32),
                b: tensor(1, DType::F32),
                gate: tensor(1, DType::F32),
                beta: tensor(1, DType::F32),
                recurrent_output: tensor(2, DType::F32),
                bf16_scratch: tensor(2, DType::BF16),
                rotation: tensor(2, DType::F32),
                z_output: tensor(2, DType::F32),
                output_scratch: tensor(2, DType::F32),
                input: tensor(2, DType::F32),
                output_tensor: tensor(2, DType::F32),
            }
        }

        fn op(&self) -> GatedDeltaNetOp<'_> {
            fn weight(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
                WeightRef {
                    buf,
                    dtype: DType::BF16,
                    m,
                    k,
                    row_stride: k,
                    rotation: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                }
            }

            GatedDeltaNetOp {
                qkv: weight(&self.qkv, 4, 2),
                conv: &self.conv,
                in_proj_a: weight(&self.in_proj_a, 1, 2),
                in_proj_b: weight(&self.in_proj_b, 1, 2),
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                z: weight(&self.z, 2, 2),
                norm: &self.norm,
                output: weight(&self.output, 2, 2),
                recurrent: &self.recurrent,
                format: GdnStateFormat::F32,
                residual: None,
                conv_state: &self.conv_state,
                projection: &self.projection,
                projection2: &self.projection2,
                a: &self.a,
                b: &self.b,
                gate: &self.gate,
                beta: &self.beta,
                recurrent_output: &self.recurrent_output,
                bf16_scratch: &self.bf16_scratch,
                rotation: &self.rotation,
                z_output: &self.z_output,
                output_scratch: &self.output_scratch,
                input: &self.input,
                output_tensor: &self.output_tensor,
                rows: 1,
                start_position: 0,
                key_heads: 1,
                value_heads: 1,
                key_dim: 1,
                value_dim: 2,
                conv_kernel: 2,
                input_width: 2,
                row_capture: None,
                zba_fold: None,
                trunk_a4: 0,
                phase: MixerPhase::Whole,
            }
        }
    }

    #[test]
    fn embedding_dispatch_preserves_bf16_and_routed_mqv2_contracts() {
        assert_eq!(embedding_path(DType::BF16), Some(EmbeddingPath::Bf16));
        assert_eq!(embedding_path(DType::MQ4G256V2), Some(EmbeddingPath::Mq4V2));
        assert_eq!(embedding_path(DType::MQ4G128V2), Some(EmbeddingPath::Mq4V2));
        assert_eq!(embedding_path(DType::Q8_0), Some(EmbeddingPath::Q8));
    }

    #[test]
    fn embedding_dispatch_rejects_unadmitted_dtype() {
        assert!(embedding_path(DType::F32).is_none());
    }

    #[test]
    fn gdn_accepts_bf16_metadata_and_rejects_f32_metadata() {
        let mut fixture = GdnFixture::new();
        assert!(fixture.op().validate_layout().is_ok());

        fixture.a_log.dtype = DType::F32;
        let error = fixture
            .op()
            .validate_layout()
            .expect_err("F32 A-log must not pass the BF16 kernel contract");
        assert!(error.to_string().contains("gated delta A-log"));

        fixture.a_log.dtype = DType::BF16;
        fixture.dt_bias.dtype = DType::F32;
        let error = fixture
            .op()
            .validate_layout()
            .expect_err("F32 dt bias must not pass the BF16 kernel contract");
        assert!(error.to_string().contains("gated delta dt bias"));
    }

    /// Null-tensor QSA op over `compress` 4, budget 8, 16 selected slots.
    fn indexed_attention<'a>(
        t: &'a GpuTensor,
        rows: usize,
        position: usize,
        selected_len: usize,
        mode: IndexedAttentionMode<'a>,
    ) -> IndexedAttentionOp<'a> {
        let weight = WeightRef {
            buf: t,
            dtype: DType::BF16,
            m: 1,
            k: 1,
            row_stride: 1,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        IndexedAttentionOp {
            mrope: None,
            rotary_dim: 2,
            rope: rdna_compute::rope::Rope {
                theta: 10_000_000.0,
                scaling: rdna_compute::rope::RopeScaling::None,
            },
            indexer_qk: weight,
            indexer_q_norm: t,
            indexer_k_norm: t,
            q: weight,
            k: weight,
            v: weight,
            q_norm: t,
            k_norm: t,
            output: weight,
            state: IndexedAttentionState {
                format: QsaKvFormat::F32,
                full_keys: t,
                full_values: t,
                raw_index_keys: t,
                pooled_keys: t,
                selected_indices: t,
                f16_arena: None,
                pooled_bf16: None,
                full_capacity: 64,
                raw_capacity: 64,
                pooled_capacity: 16,
                selected_capacity: 16,
                position_capacity: 64,
                full_len: position,
                raw_len: position,
                pooled_len: position / 4,
                selected_len,
                position,
            },
            input: t,
            index_scratch: t,
            qgate_scratch: t,
            k_scratch: t,
            v_scratch: t,
            qsa_output: t,
            selected_scratch: t,
            attention_output: t,
            bf16_scratch: t,
            rows,
            index_heads: 1,
            index_kv_heads: 1,
            index_dim: 2,
            budget: 8,
            compress: 4,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            input_width: 1,
            rotation: t,
            mode,
            trunk_a4: 0,
            phase: MixerPhase::Whole,
        }
    }

    #[test]
    fn qsa_core_windows_compose_to_the_whole_launch() {
        let t = tensor(1, DType::F32);
        for position in [0usize, 20, 8191] {
            for rows in [4647usize, 4672, 8192] {
                let whole = indexed_attention(&t, rows, position, 3, IndexedAttentionMode::Full);
                for r in [1usize, 64, 4096, rows - 1] {
                    // The first window commits exactly what a chunk of r rows commits.
                    let first = whole.next_lengths_after(r).unwrap();
                    let chunk = indexed_attention(&t, r, position, 3, IndexedAttentionMode::Full);
                    assert_eq!(
                        first,
                        chunk.next_lengths().unwrap(),
                        "pos {position} rows {rows} r {r}"
                    );
                    assert_eq!(
                        (first.0, first.1, first.4),
                        (position + r, position + r, position + r)
                    );
                    assert_eq!(first.2, (position + r) / 4);
                    // The second starts from that commit and ends where the whole launch does.
                    let second = indexed_attention(
                        &t,
                        rows - r,
                        first.4,
                        first.3,
                        IndexedAttentionMode::Full,
                    );
                    assert_eq!(
                        second.next_lengths().unwrap(),
                        whole.next_lengths().unwrap()
                    );
                }
                assert_eq!(
                    whole.next_lengths_after(rows).unwrap(),
                    whole.next_lengths().unwrap()
                );
            }
        }
    }

    #[test]
    fn gdn_core_windows_continue_the_ring_cursor_and_requantize_at_chunk_ends() {
        // The Q8 frame of a launch is `position + rows - 1` (tensor_ops); a window
        // starts at `start + lo`, so the first window's frame is `start + r - 1` and
        // the second's the whole bank's, with the conv ring cursor carried over.
        let history_rows = 3;
        for start in [0usize, 8192, 49152, 57343] {
            for rows in [4647usize, 4672, 8192] {
                for r in [1usize, 64, 4096, rows - 1] {
                    let (first, second) = (
                        CoreWindow { lo: 0, rows: r },
                        CoreWindow {
                            lo: r,
                            rows: rows - r,
                        },
                    );
                    let position = |w: CoreWindow| core_window_position(start, Some(w.lo));
                    assert_eq!(position(first), start);
                    assert_eq!(position(first) + first.rows - 1, start + r - 1);
                    assert_eq!(position(second), start + r);
                    assert_eq!(position(second) + second.rows - 1, start + rows - 1);
                    assert_eq!(
                        position(second) % history_rows,
                        (position(first) % history_rows + r) % history_rows
                    );
                }
            }
        }
        assert_eq!(core_window_position(7, None), 7);
    }

    #[test]
    fn indexed_attention_next_lengths_follow_mode() {
        let t = tensor(1, DType::F32);
        let scalar = tensor(4, DType::Raw);
        let reuse = IndexedAttentionMode::ReuseSelection {
            selected_len_out: &scalar,
        };
        // Past the budget: 2 budget blocks (8) plus the 1 row of the open block.
        let full = indexed_attention(&t, 1, 20, 12, IndexedAttentionMode::Full);
        assert_eq!(full.next_lengths().unwrap(), (21, 21, 5, 9, 21));
        let append = indexed_attention(&t, 1, 20, 12, IndexedAttentionMode::AppendOnly);
        assert_eq!(append.next_lengths().unwrap().3, 0);
        assert_eq!(
            indexed_attention(&t, 1, 20, 12, reuse)
                .next_lengths()
                .unwrap()
                .3,
            13
        );
        assert_eq!(
            indexed_attention(&t, 1, 20, 16, reuse)
                .next_lengths()
                .unwrap()
                .3,
            16
        );
    }

    #[test]
    fn gdn_projection_phases_refuse_a_row_capture() {
        let fixture = GdnFixture::new();
        let scratch = tensor(64, DType::F32);
        for (phase, refused) in [
            (MixerPhase::InProj, true),
            (MixerPhase::OutProj, true),
            (MixerPhase::Core, false),
            (MixerPhase::Whole, false),
        ] {
            let mut op = fixture.op();
            op.phase = phase;
            op.row_capture = Some(GdnRowCapture {
                states: tensor(64, DType::F32),
                inputs: &scratch,
                recurrence: &scratch,
            });
            assert_eq!(op.check_phase().is_err(), refused, "{phase:?}");
        }
    }

    #[test]
    fn gdn_projection_phases_skip_the_core_metadata_checks() {
        let mut fixture = GdnFixture::new();
        fixture.a_log.dtype = DType::F32;
        let mut op = fixture.op();
        for (phase, ok) in [
            (MixerPhase::Whole, false),
            (MixerPhase::InProj, true),
            (MixerPhase::Core, false),
            (MixerPhase::OutProj, true),
        ] {
            op.phase = phase;
            assert_eq!(op.validate_layout().is_ok(), ok, "{phase:?}");
        }
    }

    #[test]
    fn indexed_attention_reuse_validation_refuses_stale_or_multirow_selection() {
        let t = tensor(1, DType::F32);
        let scalar = tensor(4, DType::Raw);
        let reuse = IndexedAttentionMode::ReuseSelection {
            selected_len_out: &scalar,
        };
        let error = indexed_attention(&t, 2, 20, 12, reuse)
            .validate_layout()
            .expect_err("selection reuse is one row");
        assert!(error.to_string().contains("needs one row"));
        let error = indexed_attention(&t, 1, 20, 0, reuse)
            .validate_layout()
            .expect_err("a K/V-only append leaves no selection to reuse");
        assert!(error.to_string().contains("after a K/V-only append"));
    }

    #[test]
    fn indexed_attention_split_phase_refuses_reuse_and_append_modes() {
        let t = tensor(1, DType::F32);
        let scalar = tensor(4, DType::Raw);
        let reuse = IndexedAttentionMode::ReuseSelection {
            selected_len_out: &scalar,
        };
        for phase in [MixerPhase::InProj, MixerPhase::Core, MixerPhase::OutProj] {
            for mode in [reuse, IndexedAttentionMode::AppendOnly] {
                let mut op = indexed_attention(&t, 1, 20, 12, mode);
                op.phase = phase;
                let error = op
                    .validate_layout()
                    .expect_err("split phases are full-mode only");
                assert!(
                    error.to_string().contains("need the whole mixer op"),
                    "{phase:?}"
                );
            }
        }
    }

    #[test]
    fn indexed_attention_projection_phases_skip_state_checks() {
        let t = tensor(1, DType::F32);
        let mut op = indexed_attention(&t, 1, 20, 12, IndexedAttentionMode::Full);
        op.state.full_len = 0;
        let error = op.validate_layout().expect_err("whole op checks the state");
        assert!(error.to_string().contains("state length/capacity mismatch"));
        for phase in [MixerPhase::InProj, MixerPhase::OutProj] {
            op.phase = phase;
            let error = op
                .validate_layout()
                .expect_err("null tensors are too small");
            assert!(
                !error.to_string().contains("state length/capacity mismatch"),
                "{phase:?}: {error}"
            );
        }
        op.phase = MixerPhase::Core;
        let error = op.validate_layout().expect_err("core checks the state");
        assert!(error.to_string().contains("state length/capacity mismatch"));
    }

    /// H4 attention epilogue: the MQ6G256V2 output projection that carries its
    /// paired HC write must leave the streams bytewise as the projection
    /// followed by the pregated write does, F32 and BF16-bit streams, F32 and
    /// BF16 activations, ragged row counts.
    #[test]
    fn output_projection_carrying_the_hyper_write_is_bytewise_the_unfused_pair() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        std::sync::Arc::make_mut(&mut gpu.flags).qwen4_hc_fuse = 2;
        if gpu.flags.qwen4_hc_fuse_level() < 2 {
            eprintln!("skip: needs gfx1151 or gfx1201");
            return;
        }
        let (hidden, k) = (2560usize, 1024usize);
        let wide = 4 * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bf16_bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let group = 200usize; // MQ6G256V2_GROUP_BYTES
        let mut weights = vec![0u8; hidden * k / 256 * group];
        let mut state = 11u32;
        for chunk in weights.chunks_mut(group) {
            for byte in chunk.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            for (offset, bits) in [(0, 0x2000u16), (2, 0xa800), (4, 0x2100), (6, 0xa900)] {
                chunk[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let weight_buf = gpu.upload_raw(&weights, &[weights.len()]).expect("weights");
        let weight = plain_weight(&weight_buf, DType::MQ6G256V2, hidden, k);
        let norm = gpu.zeros(&[wide], DType::BF16).expect("norm");
        let inject_buf = gpu.zeros(&[4 * wide], DType::BF16).expect("inject");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for rows in [131usize, 530] {
            let rotation = gpu.zeros(&[rows * k], DType::F32).expect("rotation");
            let gates = gpu
                .upload_f32(&wave(5, rows * 4, 1.5), &[rows * 4])
                .expect("gates");
            let mixed = gpu.zeros(&[rows * hidden], DType::F32).expect("mixed");
            let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
            // Only gfx1151's unfused route reads a BF16 activation.
            let activations: &[bool] = if gpu.arch_caps.is_gfx1151() {
                &[false, true]
            } else {
                &[false]
            };
            for &bf16_activation in activations {
                let values = wave(7, rows * k, 2.0);
                let input = if bf16_activation {
                    let mut t = gpu
                        .upload_raw(&bf16_bytes(&values), &[rows * k * 2])
                        .expect("input");
                    t.dtype = DType::BF16;
                    t.shape = vec![rows * k];
                    t
                } else {
                    gpu.upload_f32(&values, &[rows * k]).expect("input")
                };
                for state_bf16 in [false, true] {
                    let initial = wave(1, rows * wide, 3.0);
                    let raw = if state_bf16 {
                        let mut raw = bf16_bytes(&initial);
                        raw.resize(rows * wide * 4, 0);
                        raw
                    } else {
                        initial.iter().flat_map(|v| v.to_le_bytes()).collect()
                    };
                    let mut streams = || {
                        let mut t = gpu.upload_raw(&raw, &[rows * wide * 4]).expect("streams");
                        t.dtype = DType::F32;
                        t.shape = vec![rows * wide];
                        t
                    };
                    let (reference, fused) = (streams(), streams());
                    let before = bits(&gpu, &reference);
                    macro_rules! write_op {
                        ($streams:expr) => {
                            HyperWriteOp {
                                state_bf16,
                                input: $streams,
                                norm_weight: &norm,
                                block_inject: plain_weight(&inject_buf, DType::BF16, 4, wide),
                                normalized: &normalized,
                                mixed: &mixed,
                                gates: &gates,
                                output: $streams,
                                rows,
                                branches: 4,
                                hidden,
                                rotation: &rotation,
                            }
                        };
                    }
                    project_weight(&mut gpu, &weight, &input, &mixed, rows, Some(&rotation))
                        .expect("project");
                    execute_hyper_write_pregated(&mut gpu, &write_op!(&reference)).expect("write");
                    assert!(
                        project_output_into_hyper_write(
                            &mut gpu,
                            &weight,
                            &input,
                            rows,
                            Some(&rotation),
                            false,
                            &write_op!(&fused),
                        )
                        .expect("fused"),
                        "fused epilogue declined rows={rows}"
                    );
                    let (want, got) = (bits(&gpu, &reference), bits(&gpu, &fused));
                    assert_ne!(want, before, "the write changed nothing");
                    let differing = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing, 0,
                        "{differing} streams words differ: rows={rows} state_bf16={state_bf16} \
                         bf16_activation={bf16_activation}"
                    );
                }
            }
        }
    }

    fn plain_weight(buf: &GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
        WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: k,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }

    #[test]
    fn indexed_attention_rotary_validation() {
        let t = tensor(1, DType::F32);
        let base = || indexed_attention(&t, 1, 8, 0, IndexedAttentionMode::Full);
        assert!(base().validate_rotary().is_ok());
        for (rotary, head_dim, index_dim, why) in [
            (0usize, 8usize, 8usize, "zero"),
            (3, 8, 8, "odd"),
            (6, 4, 8, "above head_dim"),
            (4, 8, 2, "above index_dim"),
        ] {
            let mut op = base();
            (op.rotary_dim, op.head_dim, op.index_dim) = (rotary, head_dim, index_dim);
            let error = op.validate_rotary().expect_err(why);
            assert!(
                error.to_string().contains("must be even and within"),
                "{why}: {error}"
            );
        }
        // The pooled-key kernel's fixed width: refused whenever the core runs,
        // accepted for a projection-only phase.
        let mut op = base();
        (op.rotary_dim, op.head_dim, op.index_dim) = (2, 8, 8);
        let error = op.validate_rotary().expect_err("pooled-key width");
        assert!(error.to_string().contains("pooled-key"), "{error}");
        op.phase = MixerPhase::InProj;
        assert!(op.validate_rotary().is_ok());
        // Flash-Next geometry: 64 of 256 / 128.
        let mut op = base();
        (op.rotary_dim, op.head_dim, op.index_dim) = (64, 256, 128);
        assert!(op.validate_rotary().is_ok());
        // A partial-rotary tiny profile: 4 of 8 / 4.
        let mut op = base();
        (op.rotary_dim, op.head_dim, op.index_dim) = (4, 8, 4);
        assert!(op.validate_rotary().is_ok());
    }

    #[test]
    fn grouped_depthwise_validates_embedding_width() {
        // hidden 16, 2 branches (32 channels), PLE rows 8 wide, kernel 2.
        let (rows, hidden, branches, embed) = (3usize, 16usize, 2usize, 8usize);
        let channels = branches * hidden;
        let f32s = |n: usize| tensor(n, DType::F32);
        let bf16s = |n: usize| tensor(n, DType::BF16);
        let (key_w, value_w) = (bf16s(channels * embed), bf16s(hidden * embed));
        let (norm_key, norm_query, norm_conv) = (bf16s(channels), bf16s(channels), bf16s(channels));
        let conv = bf16s(channels * 2);
        let (state, streams) = (f32s(channels), f32s(rows * channels));
        let rows_tensor = f32s(rows * embed);
        let (query, key_scratch, value_scratch) = (
            f32s(rows * channels),
            f32s(rows * channels),
            f32s(rows * hidden),
        );
        let (gated, normed, output) = (
            f32s(rows * channels),
            f32s(rows * channels),
            f32s(rows * channels),
        );
        let rotation = f32s(1);
        let op = || GroupedDepthwiseOp {
            state_bf16: false,
            rows_f16: false,
            key: plain_weight(&key_w, DType::BF16, channels, embed),
            value: plain_weight(&value_w, DType::BF16, hidden, embed),
            norm_key: &norm_key,
            norm_query: &norm_query,
            norm_conv: &norm_conv,
            conv: &conv,
            state: &state,
            streams: &streams,
            rows_tensor: &rows_tensor,
            query: &query,
            key_scratch: &key_scratch,
            value_scratch: &value_scratch,
            gated: &gated,
            normed: &normed,
            output: &output,
            rows,
            branches,
            hidden,
            embed_dim: embed,
            kernel_size: 2,
            dilation: 1,
            epsilon: 1e-6,
            rotation: &rotation,
        };
        assert!(op().validate_layout().is_ok());

        let refusal = |op: &GroupedDepthwiseOp<'_>| {
            op.validate_layout()
                .expect_err("must be refused")
                .to_string()
        };
        let mut bad = op();
        bad.embed_dim = 0;
        assert!(refusal(&bad).contains("invalid geometry"));
        // Key/value K must be the embedding width, not hidden.
        let mut bad = op();
        bad.key = plain_weight(&key_w, DType::BF16, channels, hidden);
        assert!(refusal(&bad).contains("grouped key"));
        let mut bad = op();
        bad.value = plain_weight(&value_w, DType::BF16, hidden, hidden);
        assert!(refusal(&bad).contains("grouped value"));
        // The staged rows hold rows * embed_dim, not rows * hidden.
        let short = f32s(rows * embed - 1);
        let mut bad = op();
        bad.rows_tensor = &short;
        assert!(refusal(&bad).contains("grouped rows"));
        // F16 rows are only defined for embed_dim == hidden.
        let mut bad = op();
        bad.rows_f16 = true;
        assert!(refusal(&bad).contains("F16 rows"));
        // embed_dim == hidden (Flash-Next): hidden-wide rows and K, F16 admitted.
        let wide_rows = f32s(rows * hidden);
        let (wide_key, wide_value) = (bf16s(channels * hidden), bf16s(hidden * hidden));
        let mut fn_like = op();
        fn_like.embed_dim = hidden;
        fn_like.rows_tensor = &wide_rows;
        fn_like.key = plain_weight(&wide_key, DType::BF16, channels, hidden);
        fn_like.value = plain_weight(&wide_value, DType::BF16, hidden, hidden);
        assert!(fn_like.validate_layout().is_ok());
        fn_like.rows_f16 = true;
        assert!(fn_like.validate_layout().is_ok());
    }
}

#[cfg(test)]
mod trunk_arena_dispatch_tests {
    use super::*;

    #[test]
    fn trunk_projection_route_geometry_uses_live_rows() {
        for (rows, dense, ragged) in [
            (8192, true, None),
            (8151, false, Some(8192)),
            (512, true, None),
        ] {
            assert_eq!(rdna_compute::gemm::qwen4_trunk_iu4_dense_rows(rows), dense);
            assert_eq!(
                rdna_compute::gemm::qwen4_trunk_iu4_ragged_rows(rows),
                ragged
            );
        }
    }
    #[test]
    fn partial_views_preserve_native_sources_per_family() {
        let tensor = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::dangling_mut(), 256 * 10240 * 2)
            },
            shape: vec![256, 10240],
            dtype: DType::BF16,
        };
        let qkv = Bf16Arena {
            tensor: &tensor,
            row_pitch_elems: 10240,
            col0_elems: 0,
        };
        let gdn = core_arena(
            MixerPhase::Core,
            Some(GdnTrunkArenas {
                qkv: Some(qkv),
                z: None,
                a: None,
                b: None,
            }),
        )
        .unwrap();
        assert!(gdn.qkv.is_some());
        assert!(gdn.z.is_none() && gdn.a.is_none() && gdn.b.is_none());
        let qsa = core_arena(
            MixerPhase::Core,
            Some(QsaTrunkArenas {
                q: None,
                k: Some(qkv),
                v: None,
                idx: None,
            }),
        )
        .unwrap();
        assert!(qsa.k.is_some());
        assert!(qsa.q.is_none() && qsa.v.is_none() && qsa.idx.is_none());
    }
    #[test]
    fn only_explicit_core_views_select_arena_consumers() {
        for phase in [
            MixerPhase::Whole,
            MixerPhase::InProj,
            MixerPhase::Core,
            MixerPhase::OutProj,
        ] {
            assert_eq!(core_arena::<u8>(phase, None), None);
            assert_eq!(
                core_arena(phase, Some(1)),
                (phase == MixerPhase::Core).then_some(1)
            );
        }
    }
    #[test]
    fn g942_trunk_bf16_projection_route_is_every_row_mfma() {
        use DType::{BF16, F16, F32, MQ4G256V2};
        // Every row count, one row included, on gfx942.
        for rows in [1, 2, 7, 8, 9, 31, 32, 33, 8192] {
            assert!(
                g942_trunk_mfma_route(true, false, rows, F32, BF16, F32),
                "rows {rows}"
            );
            assert!(
                g942_trunk_mfma_route(true, false, rows, BF16, BF16, F32),
                "rows {rows}"
            );
            assert!(
                g942_trunk_mfma_route(true, false, rows, F32, BF16, BF16),
                "rows {rows}"
            );
        }
        // Never off gfx942, never without rows.
        assert!(!g942_trunk_mfma_route(false, false, 64, F32, BF16, F32));
        assert!(!g942_trunk_mfma_route(true, false, 0, F32, BF16, F32));
        // Only BF16 weights, F32/BF16 activations and outputs.
        assert!(!g942_trunk_mfma_route(true, false, 64, F32, MQ4G256V2, F32));
        assert!(!g942_trunk_mfma_route(true, false, 64, F16, BF16, F32));
        assert!(!g942_trunk_mfma_route(true, false, 64, F32, BF16, F16));
    }
    #[test]
    fn g942_trunk_gemv_opt_out_restores_only_the_one_row_arm() {
        use DType::{BF16, F32};
        assert!(!g942_trunk_mfma_route(true, true, 1, F32, BF16, F32));
        for rows in [2, 8, 9, 32, 33, 8192] {
            assert!(
                g942_trunk_mfma_route(true, true, rows, F32, BF16, F32),
                "rows {rows}"
            );
        }
        // The opt-out is a gfx942 lever: elsewhere the route is already off.
        assert!(!g942_trunk_mfma_route(false, true, 1, F32, BF16, F32));
    }
}
