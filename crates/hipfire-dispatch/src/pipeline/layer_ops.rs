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

use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use rdna_compute::tensor_ops::{
    argmax_f32, bf16_roundtrip_f32, gated_delta_chunk_route, gated_delta_conv_params,
    gated_delta_conv_params_batched, gated_delta_conv_params_qknorm_batched,
    gated_delta_conv_qknorm_route, gated_delta_gate_batched, gated_delta_gate_batched_rotate,
    gated_delta_step_batched, gated_delta_step_gate_wmma, gated_delta_step_gate_wmma_qknormed,
    gated_delta_step_gated,
    hc_activation_fused_f32, hc_state_bf16_add_f32, hc_state_bf16_to_f32, hyper_norm,
    hyper_norm_f16, hyper_norm_gate, hyper_norm_gate_outputs, hyper_read_projected,
    hyper_read_up_fused, hyper_read_up_wmma,
    hyper_write, hyper_write_norm, indexed_attention_attention, indexed_attention_attention_batch,
    indexed_attention_cache_append_batch, indexed_attention_decode_prologue,
    indexed_attention_index_key_append_batch, indexed_attention_norm_rope_batch,
    indexed_attention_pool_rope_incremental, indexed_attention_reuse_selection,
    indexed_attention_select_batch_mirrored, scale_f32, ArgmaxF32, Bf16Roundtrip, GatedDeltaConv,
    GatedDeltaConvBatched, GatedDeltaGate, GatedDeltaGateBatched, GatedDeltaParams,
    GatedDeltaParamsBatched, GatedDeltaStep, GatedDeltaStepBatched, GdnStateFormat, HcActivationFused,
    HyperNextGates, HyperNorm, HyperNormGate, HyperNormGateOutputs, HyperReadProjected,
    HyperReadUpFused, HyperWrite,
    IndexedAttentionAttention, IndexedAttentionAttentionBatch, IndexedAttentionCacheAppendBatch,
    IndexedAttentionDecodePrologue, IndexedAttentionIndexKeyAppendBatch,
    IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope, IndexedAttentionReuseSelection,
    IndexedAttentionSelectBatch, QsaKvFormat, ScaleF32,
};
use rdna_compute::{DType, Gpu, GpuTensor, TrunkFamily};
use smallvec::SmallVec;

#[inline]
pub(super) fn hip<T>(result: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    result.map_err(|error| DispatchError::Hip(error.to_string()))
}

#[inline]
fn checked_mul(a: usize, b: usize, label: &'static str) -> Result<usize, DispatchError> {
    a.checked_mul(b)
        .ok_or_else(|| DispatchError::Hip(format!("{label} extent overflows")))
}

fn require_tensor(
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

fn require_weight(
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
fn view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
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
/// gfx1151 MQ6 BT8 route the input is rotated straight to F16 once per K and
/// every MQ6 GEMM reads that copy (the per-weight path rotates to F32 and
/// converts to F16 again for each); the bytes each GEMM sees are unchanged.
pub fn project_weights(
    gpu: &mut Gpu,
    input: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
) -> Result<(), DispatchError> {
    // The chunked GDN in-projection group: BF16 qkv ahead of F32 a/b/z.
    let members = [false, true, true, true];
    let gdn_regions = projections.len() == 4 && projections[0].1.dtype == DType::BF16;
    project_weights_inner(gpu, input, rows, rotation, projections, gdn_regions.then_some(&members[..]))
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
) -> Result<(), DispatchError> {
    debug_assert!(region_members.is_none_or(|m| m.len() == projections.len()));
    let shared = |w: &WeightRef<'_>, gpu: &Gpu| {
        w.dtype == DType::MQ6G256V2 && gpu.gemm_mq6g256v2_xf16_applies(w.k, rows)
    };
    // Shared-rotation weights first, grouped by K: other projections reuse
    // the same F16 scratch and would overwrite the rotated copy.
    let mut done = vec![false; projections.len()];
    for i in 0..projections.len() {
        let (weight, _) = projections[i];
        if done[i] || !shared(weight, gpu) {
            continue;
        }
        let x_f16 = hip(gpu.rotate_x_mq_batched_f16(input, weight.k, rows))?;
        // The chunked GDN qkv output is BF16; its three F32 a/b/z siblings
        // may share one row-region launch without sharing their accumulators.
        if let (Some(members), true) = (region_members, gpu.qwen4_mq6_x4_regions()) {
            let mut indices = [0usize; 3];
            let mut count = 0;
            for j in i..projections.len() {
                let (w, out) = projections[j];
                if members[j]
                    && !done[j]
                    && w.k == weight.k
                    && shared(w, gpu)
                    && out.dtype == DType::F32
                {
                    if count < 3 { indices[count] = j; }
                    count += 1;
                }
            }
            if count == 3 {
                let regions = indices.map(|j| {
                    let (w, out) = projections[j];
                    (w.buf, out, w.m)
                });
                if hip(gpu.gemm_mq6g256v2_xf16_regions(&regions, &x_f16, weight.k, rows))? {
                    for j in indices { done[j] = true; }
                }
            }
        }
        for j in i..projections.len() {
            let (w, out) = projections[j];
            if !done[j] && w.k == weight.k && shared(w, gpu) {
                hip(gpu.gemm_mq6g256v2_xf16(w.buf, &x_f16, out, w.m, w.k, rows))?;
                done[j] = true;
            }
        }
    }
    // Only the shared MQ6 rotation reads a BF16 activation.
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
        if done[i] || weight.dtype != DType::BF16 {
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
    if (2..=8).contains(&rows) && gpu.arch_caps.has_gfx11_plus_simt() {
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
            .filter(|&j| rows <= 8 && projections[j].0.dtype == DType::MQ6G256V2)
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
        for &j in group.iter().filter(|j| !mq6.contains(j)) {
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
}

/// Qwen4 trunk projections (GDN in/out, QSA in/out) of one shared input.
///
/// `a4[i]` says the per-layer mask (`HIPFIRE_QWEN4_TRUNK_IU4`) selected
/// `projections[i]`. Each projection takes one of three routes, run as
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
/// 2. Any other MQ4G256V2 projection of an F32 prefill input (more than 8
///    rows, gfx1151): exact (F16) activations, one shared rotation.
/// 3. Everything else (MQ6 and the rest): [`project_weights`] exactly as if
///    no MQ4 sibling existed. When siblings were split off a GDN in-projection
///    group (`gdn_in_proj`), the MQ6 row-region fold stays enabled for the
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
    debug_assert_eq!(a4.len(), projections.len());
    let mut done: SmallVec<[bool; 4]> = SmallVec::from_elem(false, projections.len());
    let k = projections.iter().find(|(w, _)| w.dtype == DType::MQ4G256V2).map(|(w, _)| w.k);
    if let (Some(k), DType::F32) = (k, input.dtype) {
        let mq4: SmallVec<[usize; 4]> = (0..projections.len())
            .filter(|&j| projections[j].0.dtype == DType::MQ4G256V2 && projections[j].0.k == k)
            .collect();
        let dense: SmallVec<[usize; 4]> = if gpu.qwen4_trunk_iu4_applies(rows) {
            mq4.iter().copied().filter(|&j| a4[j]).collect()
        } else {
            SmallVec::new()
        };
        if !dense.is_empty() {
            let reservation = hip(gpu.reserve_int4_mmq(k, rows))?;
            let prepared =
                hip(gpu.rotate_x_mq_i4_batched(input, None, None, reservation, k, rows))?;
            if let Some(fold) = zba {
                let (z, y_z) = projections[fold.z];
                if [fold.z, fold.beta, fold.alpha].iter().all(|j| dense.contains(j))
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
        let exact: SmallVec<[usize; 4]> = mq4.iter().copied().filter(|&j| !done[j]).collect();
        if rows > 8 && !exact.is_empty() {
            let x = view(rotation, 0, rows * k);
            hip(gpu.rotate_x_mq_batched(input, &x, k, rows))?;
            for &j in &exact {
                let (w, out) = projections[j];
                if !hip(gpu.gemm_qwen4_trunk_mq4_xf16(w.buf, &x, out, w.m, k, rows))? {
                    break;
                }
                done[j] = true;
            }
        }
    }
    if done.iter().all(|d| !d) {
        return project_weights(gpu, input, rows, Some(rotation), projections);
    }
    let rest: SmallVec<[(&WeightRef<'_>, &GpuTensor); 4]> =
        (0..projections.len()).filter(|&j| !done[j]).map(|j| projections[j]).collect();
    if rest.is_empty() {
        return Ok(());
    }
    // Only the original GDN a/b/z (positions 1.. of the in-projection group)
    // may join the row-region fold; qkv (position 0) never does.
    let members: SmallVec<[bool; 4]> = (0..projections.len())
        .filter(|&j| !done[j])
        .map(|j| gdn_in_proj && j != 0)
        .collect();
    project_weights_inner(
        gpu,
        input,
        rows,
        Some(rotation),
        &rest,
        gdn_in_proj.then_some(&members[..]),
    )
}

/// FWHT basis a quantized projection payload reads: the aligned-K 256-wide
/// one (MQ4G256V2, MQ6G256V2, MFP4G32E8SOA) or MQ4G128V2's row-local 128-wide
/// one. `None` for payloads that read the natural activation.
#[derive(PartialEq)]
enum RotationBasis {
    Aligned256,
    RowLocal128,
}

fn rotation_basis(dtype: DType) -> Option<RotationBasis> {
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
        RotationBasis::Aligned256 if rows > 1 => {
            hip(gpu.rotate_x_mq_batched(input, &scratch, weight.k, rows))?
        }
        RotationBasis::Aligned256 => hip(gpu.rotate_x_mq(input, &scratch, weight.k))?,
        RotationBasis::RowLocal128 => hip(gpu.rotate_x_mq_128_v2(input, &scratch, weight.k, rows))?,
    }
    Ok(Some(scratch))
}

/// Run one projection on an input already in the weight's basis.
fn project_rotated(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    output: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    if weight.dtype == DType::MQ6G256V2 && (2..=8).contains(&rows) {
        return hip(gpu.gemm_mq6g256v2_f32_rows(weight.buf, x, output, weight.m, weight.k, rows));
    }
    if weight.dtype == DType::BF16 && (2..=8).contains(&rows) && gpu.arch_caps.has_gfx11_plus_simt()
    {
        let part = (weight.buf, output, weight.m);
        let none = (weight.buf, output, 0);
        return hip(gpu.gemv_bf16_xf32_x4_rows([part, none, none, none], x, weight.k, rows));
    }
    let result = match (weight.dtype, rows > 1) {
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
        // `gemm_q8_0_batched` is capped at MAX_BATCH=64 (it asserts), so a
        // prefill wider than that fails outright. The F32-preserving chunked
        // entry sub-batches at 64 and calls that SAME kernel per sub-batch, so
        // the arithmetic the Q8 trunk already had is unchanged.
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
    pub bf16_scratch: &'a GpuTensor,
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
        require_tensor(
            self.bf16_scratch,
            self.low_rank.max(self.hidden),
            DType::BF16,
            "hyper read BF16 scratch",
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
pub fn hyper_read_pairs_write(
    gpu: &Gpu,
    read: &HyperReadOp<'_>,
    write: &HyperWriteOp<'_>,
) -> bool {
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
        Err(DispatchError::Hip("hyper read: prenormed route did not apply".into()))
    }
}

/// The F16 WMMA read route of [`execute_hyper_read_inner`]: the norm writes
/// the F16 down input directly and the BF16 WMMA up read reuses it.
fn hyper_read_wmma_route(gpu: &Gpu, op: &HyperReadOp<'_>) -> bool {
    let wide = op.branches * op.hidden;
    let up_fused = gpu.arch_caps.has_gfx11_plus_simt()
        && op.rows > 1
        && op.branches == 4
        && op.input_mix_up.dtype == DType::BF16
        && op.input_mix_up.m == wide
        && op.input_mix_up.k == op.low_rank
        && op.low_rank % 8 == 0
        && (257..=512).contains(&op.low_rank)
        && op.hidden % 8 == 0;
    let f16 = up_fused
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
        && op.rows > 1
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
    let f16 = up_fused
        && op.input_mix_down.dtype == DType::BF16
        && gpu.qwen4_f16_wmma_applies(op.input_mix_down.buf, op.input_mix_down.k, op.rows);
    let wmma_read = f16 && op.low_rank % 16 == 0 && op.low_rank <= 504 && op.hidden % 16 == 0;
    let mut normalized_f16 = None;
    let mut activated = false;
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
        hip(gpu.gemm_bf16_xf16_f16_wmma(
            op.input_mix_down.buf,
            &x16,
            &low,
            op.input_mix_down.m,
            op.input_mix_down.k,
            op.rows,
        ))?;
        normalized_f16 = Some(x16);
    } else if gpu.arch_caps.has_gfx11_plus_simt()
        && op.rows <= 8
        && ((op.input_mix_down.dtype == DType::BF16 && op.input_mix_down.k.is_multiple_of(32))
            || (op.input_mix_down.dtype == DType::Q8_0
                && op.input_mix_down.k.is_multiple_of(256)))
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
    // The WMMA read takes `low` as packed BF16, staged in `up` (which that
    // route does not otherwise use).
    let low_bf16 = wmma_read.then(|| {
        let mut packed = view(op.up, 0, op.rows * op.low_rank);
        packed.dtype = DType::BF16;
        packed
    });
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
        for row in 0..op.rows {
            let low_row = view(&low, row * op.low_rank, op.low_rank);
            hip(bf16_roundtrip_f32(
                gpu,
                &Bf16Roundtrip {
                    input: &low_row,
                    scratch: op.bf16_scratch,
                    output: &low_row,
                    elements: op.low_rank,
                },
            ))?;
        }
        hip(scale_f32(
            gpu,
            &ScaleF32 {
                values: &low,
                scale: 1.0 / op.branches as f32,
            },
        ))?;
        for row in 0..op.rows {
            let low_row = view(&low, row * op.low_rank, op.low_rank);
            hip(bf16_roundtrip_f32(
                gpu,
                &Bf16Roundtrip {
                    input: &low_row,
                    scratch: op.bf16_scratch,
                    output: &low_row,
                    elements: op.low_rank,
                },
            ))?;
        }
        hip(gpu.silu_f32(&low, &low))?;
        for row in 0..op.rows {
            let low_row = view(&low, row * op.low_rank, op.low_rank);
            hip(bf16_roundtrip_f32(
                gpu,
                &Bf16Roundtrip {
                    input: &low_row,
                    scratch: op.bf16_scratch,
                    output: &low_row,
                    elements: op.low_rank,
                },
            ))?;
        }
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

/// H4 attention epilogue (`HIPFIRE_QWEN4_HC_FUSE` >= 2): an MQ6G256V2 output
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
    write: &HyperWriteOp<'_>,
) -> Result<bool, DispatchError> {
    let Some(wide) = write.branches.checked_mul(write.hidden) else {
        return Ok(false);
    };
    if gpu.flags.qwen4_hc_fuse_level() < 2
        || weight.dtype != DType::MQ6G256V2
        || weight.m != write.hidden
        || write.branches != 4
        || write.rows != rows
        || write.input.dtype != DType::F32
        || write.input.buf.as_ptr() != write.output.buf.as_ptr()
        || write.input.numel() < rows * wide
        || write.gates.dtype != DType::F32
        || write.gates.numel() < rows * write.branches
        || !gpu.gemm_mq6g256v2_hcw_applies(weight.m, weight.k, rows)
    {
        return Ok(false);
    }
    let (x, x_is_f16) = if gpu.arch_caps.is_gfx1151() || input.dtype == DType::BF16 {
        (hip(gpu.rotate_x_mq_batched_f16(input, weight.k, rows))?, true)
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
    let fusable = (1..=8).contains(&write.rows)
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
    // Decode and few-row verify (<= 8 rows): the gate GEMV splits each row's
    // K across four waves, and a verify row is bitwise the decode row it
    // replaces (greedy MTP == AR depends on it).
    let k4 = gpu.arch_caps.has_gfx11_plus_simt()
        && op.rows <= 8
        && op.block_inject.dtype == DType::BF16
        && op.block_inject.k.is_multiple_of(32);
    // Multi-row: one launch normalizes each row into LDS and projects the
    // BF16 gate from there; `normalized` (read by nothing below) is not
    // written.  Bitwise identical to hyper_norm + project_weight.
    let fused = gpu.arch_caps.has_gfx11_plus_simt()
        && op.rows > 1
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

    /// A row capture spans the whole op: a split phase cannot honor it.
    fn check_phase(&self) -> Result<(), DispatchError> {
        if self.phase != MixerPhase::Whole && self.row_capture.is_some() {
            return Err(DispatchError::Hip(
                "GDN row capture needs the whole mixer op".into(),
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
            let state_format = GdnStateFormat::of(self.recurrent).map_err(|error| {
                DispatchError::Hip(format!("gated delta state: {error}"))
            })?;
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
    op.check_phase()?;
    let phase = op.phase;
    let qk = op.key_heads * op.key_dim;
    let value = op.value_heads * op.value_dim;
    let qkv = 2 * qk + value;
    let mut projection = view(op.projection, 0, op.rows * qkv);
    let projection2 = view(op.projection2, 0, op.rows * qkv);
    let a = view(op.a, 0, op.rows * op.value_heads);
    let b = view(op.b, 0, op.rows * op.value_heads);
    let mut gate = view(op.gate, 0, op.rows * op.value_heads);
    let mut beta = view(op.beta, 0, op.rows * op.value_heads);
    let z = view(op.z_output, 0, op.rows * value);
    let history_rows = op.conv_kernel.saturating_sub(1);
    let persistent_batch = gpu.arch_caps.has_gfx11_plus_simt()
        && op.rows > 1
        && op.key_dim == 128
        && op.value_dim == 128
        && op.conv_kernel == 4;
    let recurrent_output = view(op.recurrent_output, 0, op.rows * value);
    if op.row_capture.is_some() && !persistent_batch {
        return Err(DispatchError::Hip(
            "GDN row capture needs the few-row persistent recurrence route".into(),
        ));
    }
    let capture = op.row_capture.as_ref();
    // A row capture keeps the qkv projection and the recurrence inputs:
    // write them straight into it.
    let mut conv_output = view(op.projection2, 0, op.rows * qkv);
    if let Some(capture) = capture {
        projection = view(capture.inputs, 0, op.rows * qkv);
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
        output: &recurrent_output,
        row_states: capture.map(|c| &c.states),
        rows: op.rows,
        qkv_width: qkv,
        key_heads: op.key_heads,
        value_heads: op.value_heads,
        key_dim: op.key_dim,
        value_dim: op.value_dim,
        position: op.start_position,
    };
    let chunked = persistent_batch && gated_delta_chunk_route(gpu, &dims);
    if chunked && capture.is_some() {
        return Err(DispatchError::Hip(
            "GDN row capture is a few-row verify contract, not the chunked prefill route".into(),
        ));
    }
    // On the chunked route the qkv projection is read (by the convolution)
    // only through its BF16 rounding, so the MQ6 GEMM stores it as BF16 bits.
    let bf16_store = |w: &WeightRef<'_>, gpu: &Gpu| {
        chunked && w.dtype == DType::MQ6G256V2 && gpu.gemm_mq6g256v2_xf16_applies(w.k, op.rows)
    };
    if bf16_store(&op.qkv, gpu) {
        projection.dtype = DType::BF16;
    }
    if matches!(phase, MixerPhase::Whole | MixerPhase::InProj) {
        project_trunk(
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
            op.zba_fold.map(|rows| ZbaFold { rows, z: 3, beta: 2, alpha: 1 }),
            true,
        )?;
    }
    if phase == MixerPhase::InProj {
        return Ok(());
    }
    let mut gdn_output = view(op.output_scratch, 0, op.rows * value);
    if phase == MixerPhase::OutProj {
        // The core ran in an earlier call; only the dtype decision it made for
        // its output is owed here.
        if persistent_batch && bf16_store(&op.output, gpu) {
            gdn_output.dtype = DType::BF16;
        }
    } else if persistent_batch {
        let start_cursor = op.start_position % history_rows;
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
            rows: op.rows,
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
            rows: op.rows,
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
        let fused_qk = chunked
            && capture.is_none()
            && gated_delta_conv_qknorm_route(gpu, &step, &conv);
        // gfx1201 FN dense route: its producer replaces the convolution, the
        // gate-parameter pass and the step (the gate below finishes it).
        #[cfg(feature = "deltanet")]
        let dense = !chunked && rdna_compute::fn_gdn_dense::applies(gpu, &step);
        #[cfg(not(feature = "deltanet"))]
        let dense = false;
        // Convolution and gate parameters in one launch.
        if dense {
            #[cfg(feature = "deltanet")]
            hip(rdna_compute::fn_gdn_dense::run(gpu, &conv, &gate_params, &step))?;
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
        if bf16_store(&op.output, gpu) {
            gdn_output.dtype = DType::BF16;
        }
        let gated = GatedDeltaGateBatched {
            recurrent_output: &recurrent_output,
            z: &z,
            norm: op.norm,
            output: &gdn_output,
            rows: op.rows,
            value_heads: op.value_heads,
            value_dim: op.value_dim,
        };
        // The F16 prefill route fuses the gate into the chunked WMMA
        // recurrence (KLD-gated); otherwise the exact kernels run.
        if chunked {
            if fused_qk {
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
                    let rotated = view(op.rotation, 0, op.rows * value);
                    hip(rdna_compute::fn_gdn_dense::gate(gpu, &gated, Some(&rotated)))?;
                } else {
                    hip(rdna_compute::fn_gdn_dense::gate(gpu, &gated, None))?;
                }
            } else if rotate {
                let rotated = view(op.rotation, 0, op.rows * value);
                hip(gated_delta_gate_batched_rotate(gpu, &gated, &rotated))?;
            } else {
                hip(gated_delta_gate_batched(gpu, &gated))?;
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
    // H4 (`HIPFIRE_QWEN4_HC_FUSE` >= 2): the paired HC write rides the output
    // projection's epilogue; the projection then never lands in `output_tensor`.
    if let Some(write) = hc {
        if project_output_into_hyper_write(
            gpu,
            &op.output,
            &gdn_output,
            op.rows,
            Some(op.rotation),
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
/// shared step list succeeds. The K/V caches are `format`'s rows and the raw
/// and pooled index keys its `index_dtype`.
pub struct IndexedAttentionState<'a> {
    pub format: QsaKvFormat,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub pooled_keys: &'a GpuTensor,
    pub selected_indices: &'a GpuTensor,
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
        let q_width = checked_mul(self.heads, self.head_dim, "indexed attention Q width")?;
        let qgate_width = checked_mul(2, q_width, "indexed attention Q/G width")?;
        let kv_width = checked_mul(self.kv_heads, self.head_dim, "indexed attention KV width")?;
        let rows_index = checked_mul(self.rows, index_width, "indexed attention index rows")?;
        let rows_q = checked_mul(self.rows, q_width, "indexed attention Q rows")?;
        let rows_kv = checked_mul(self.rows, kv_width, "indexed attention KV rows")?;
        let rows_input = checked_mul(self.rows, self.input_width, "indexed attention input")?;
        if core && (self.state.position > self.state.position_capacity
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
        let final_position = self
            .state
            .position
            .checked_add(self.rows)
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

fn qsa_projection_hook_run(gpu: &mut Gpu, op: &IndexedAttentionOp<'_>) -> Result<(), DispatchError> {
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
            &[(&op.output, &view(op.attention_output, 0, op.rows * op.output.m))],
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
    op.check_phase()?;
    let phase = op.phase;
    let index_width = (op.index_heads + op.index_kv_heads) * op.index_dim;
    let index_q_width = op.index_heads * op.index_dim;
    let index_kv_width = op.index_kv_heads * op.index_dim;
    let q_width = op.heads * op.head_dim;
    let kv_width = op.kv_heads * op.head_dim;
    let initial_position = op.state.position;
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
    if matches!(phase, MixerPhase::Whole | MixerPhase::InProj) {
        project_trunk(gpu, op.input, op.rows, op.rotation, projections, a4, None, false)?;
        qsa_projection_hook_run(gpu, op)?;
    }
    if phase == MixerPhase::InProj {
        return Ok(());
    }
    if phase == MixerPhase::OutProj {
        return qsa_project_output(gpu, op, &qsa_output_batch, hc, fused);
    }

    if op.rows <= 8 && op.index_dim <= 256 && op.head_dim <= 256 {
        // Decode / few-row verify: the norms, RoPE, cache append and
        // index-key round trip and copy below, in one launch.
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
                rows: op.rows,
                format: op.state.format,
            },
        ))?;
    } else {
        hip(indexed_attention_norm_rope_batch(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &index_batch,
                norm: op.indexer_q_norm,
                rows: op.rows,
                row_stride: index_width,
                heads: op.index_heads,
                head_dim: op.index_dim,
                head_stride: op.index_dim,
                position_start: initial_position,
                rotary_dim: op.index_dim.min(64),
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
                    rows: op.rows,
                    index_q_width,
                    index_kv_width,
                    position_start: initial_position,
                },
            ))?;
        } else {
            hip(gpu.bf16_round_trip_f32_strided(
                &index_batch,
                op.rows,
                index_q_width,
                index_width,
                index_kv_width,
            ))?;
            let index_k_batch = view(
                &index_batch,
                index_q_width,
                op.rows * index_width - index_q_width,
            );
            // The destination row offset travels as a scalar (`= position *
            // index_kv_width`) against the base tensor, and the recorder declares it,
            // so the tape keeps a position-independent pointer and replay
            // re-derives the offset for its own position instead of replaying the
            // capture-position row.
            hip(gpu.copy_rows_strided_f32(
                &index_k_batch,
                op.state.raw_index_keys,
                op.rows,
                index_kv_width,
                index_width,
                index_kv_width,
                initial_position * index_kv_width,
                Some(index_kv_width),
            ))?;
        }
        hip(indexed_attention_norm_rope_batch(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &qgate_batch,
                norm: op.q_norm,
                rows: op.rows,
                row_stride: 2 * q_width,
                heads: op.heads,
                head_dim: op.head_dim,
                head_stride: 2 * op.head_dim,
                position_start: initial_position,
                rotary_dim: op.head_dim.min(64),
            },
        ))?;
        hip(indexed_attention_norm_rope_batch(
            gpu,
            &IndexedAttentionNormRopeBatch {
                values: &k_batch,
                norm: op.k_norm,
                rows: op.rows,
                row_stride: kv_width,
                heads: op.kv_heads,
                head_dim: op.head_dim,
                head_stride: op.head_dim,
                position_start: initial_position,
                rotary_dim: op.head_dim.min(64),
            },
        ))?;
        hip(indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k_batch,
                value: &v_batch,
                full_keys: op.state.full_keys,
                full_values: op.state.full_values,
                rows: op.rows,
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
        hip(indexed_attention_pool_rope_incremental(
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
                    rows: op.rows,
                }),
                grid_bound: op.state.pooled_capacity,
            },
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
        selected: &selected_batch,
        rows: op.rows,
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
    hip(indexed_attention_attention_batch(
        gpu,
        &IndexedAttentionAttentionBatch {
            q_with_gate: &qgate_batch,
            full_keys: op.state.full_keys,
            full_values: op.state.full_values,
            selected: &selected_batch,
            output: &qsa_output_batch,
            rows: op.rows,
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
    /// `rows_tensor` already holds the PLE rows as F16 (first `rows * hidden`
    /// halves of its storage; `HIPFIRE_QWEN4_PLE_FUSE`'s F16 gather): the key
    /// and value projections read them directly and the fused tail must run.
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
    pub kernel_size: usize,
    pub dilation: usize,
    pub epsilon: f32,
    /// FWHT basis scratch for quantized payloads (`rows * k` elements); the
    /// BF16 path never reads it.
    pub rotation: &'a GpuTensor,
}

impl GroupedDepthwiseOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        if self.rows == 0
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
        require_tensor(self.rows_tensor, rows_hidden, DType::F32, "grouped rows")?;
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
        require_weight(&self.key, channels, self.hidden, "grouped key")?;
        require_weight(&self.value, self.hidden, self.hidden, "grouped value")?;
        Ok(())
    }
}

pub fn execute_grouped_depthwise(
    gpu: &mut Gpu,
    op: &GroupedDepthwiseOp<'_>,
) -> Result<(), DispatchError> {
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
        // SAFETY: `rows_tensor` holds `rows * hidden` F32 elements, so its first
        // `rows * hidden * 2` bytes hold the F16 rows; the view is non-owning.
        let x_f16 = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    op.rows_tensor.buf.as_ptr(),
                    op.rows * op.hidden * 2,
                )
            },
            shape: vec![op.rows * op.hidden],
            dtype: DType::F16,
        };
        for (weight, out) in [(&op.key, &key), (&op.value, &value)] {
            hip(gpu.gemm_bf16_xf16_f16_wmma(
                weight.buf, &x_f16, out, weight.m, weight.k, op.rows,
            ))?;
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

/// One [`project_weight`] over `rows` row-major input rows.
pub struct ProjectOp<'a> {
    pub weight: WeightRef<'a>,
    pub input: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    /// FWHT basis scratch (`rows * k`), required by the rotated payloads.
    pub rotation: Option<&'a GpuTensor>,
}

impl ProjectOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        let (m, k) = (self.weight.m, self.weight.k);
        if self.rows == 0 {
            return Err(DispatchError::Hip("projection has no rows".into()));
        }
        require_weight(&self.weight, m, k, "projection")?;
        let input = checked_mul(self.rows, k, "projection input")?;
        require_tensor(self.input, input, DType::F32, "projection input")?;
        let output = checked_mul(self.rows, m, "projection output")?;
        require_tensor(self.output, output, DType::F32, "projection output")?;
        if !matches!(self.weight.dtype, DType::BF16 | DType::Q8_0) {
            let rotation = self.rotation.ok_or_else(|| {
                DispatchError::Hip("rotated projection payload needs rotation scratch".into())
            })?;
            require_tensor(rotation, input, DType::F32, "projection rotation")?;
        }
        Ok(())
    }
}

pub fn execute_project(gpu: &mut Gpu, op: &ProjectOp<'_>) -> Result<(), DispatchError> {
    project_weight(gpu, &op.weight, op.input, op.output, op.rows, op.rotation)
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
    if requested_rows == rows {
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
    fn gdn_split_phase_refuses_a_row_capture() {
        let fixture = GdnFixture::new();
        let scratch = tensor(64, DType::F32);
        for phase in [MixerPhase::InProj, MixerPhase::Core, MixerPhase::OutProj] {
            let mut op = fixture.op();
            op.phase = phase;
            op.row_capture = Some(GdnRowCapture {
                states: tensor(64, DType::F32),
                inputs: &scratch,
                recurrence: &scratch,
            });
            let error = op
                .validate_layout()
                .expect_err("a row capture needs the whole mixer op");
            assert!(
                error.to_string().contains("GDN row capture needs the whole mixer op"),
                "{phase:?}"
            );
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
                assert!(error.to_string().contains("need the whole mixer op"), "{phase:?}");
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
            let error = op.validate_layout().expect_err("null tensors are too small");
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
}
