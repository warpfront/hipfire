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
    argmax_f32, bf16_roundtrip_f32, gated_delta_chunk_route, gated_delta_conv,
    gated_delta_conv_batched, gated_delta_gate, gated_delta_gate_batched, gated_delta_params,
    gated_delta_params_batched, gated_delta_step, gated_delta_step_batched,
    gated_delta_step_gate_wmma, hc_activation_fused_f32, hc_state_bf16_add_f32,
    hc_state_bf16_to_f32, hyper_norm, hyper_norm_f16, hyper_norm_gate, hyper_read_projected,
    hyper_read_up_fused, hyper_read_up_wmma, hyper_write, indexed_attention_attention_batch,
    indexed_attention_cache_append_batch, indexed_attention_norm_rope_batch,
    indexed_attention_pool_rope, indexed_attention_select_batch, scale_f32, ArgmaxF32,
    Bf16Roundtrip, GatedDeltaConv, GatedDeltaConvBatched, GatedDeltaGate, GatedDeltaGateBatched,
    GatedDeltaParams, GatedDeltaParamsBatched, GatedDeltaStep, GatedDeltaStepBatched,
    HcActivationFused, HyperNorm, HyperNormGate, HyperReadProjected, HyperReadUpFused, HyperWrite,
    IndexedAttentionAttentionBatch, IndexedAttentionCacheAppendBatch,
    IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope, IndexedAttentionSelectBatch, ScaleF32,
};
use rdna_compute::{DType, Gpu, GpuTensor};
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
    for (i, &(weight, output)) in projections.iter().enumerate() {
        if !done[i] {
            project_one(gpu, weight, input, output, rows, rotation)?;
        }
    }
    Ok(())
}

fn project_one(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    output: &GpuTensor,
    rows: usize,
    rotation: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    let rotated = match weight.dtype {
        // BF16 and Q8F16 read the natural activation: neither carries an FWHT
        // basis, so no rotation is owed and the scratch stays untouched.
        DType::BF16 | DType::Q8_0 => None,
        // MQ4G256V2 and MQ6G256V2 share the aligned-K 256-wide FWHT basis;
        // MQ4G128V2 carries the row-local 128-wide one.
        DType::MQ4G256V2 | DType::MQ4G128V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA => {
            let rotation = rotation.ok_or(DispatchError::UnsupportedVariant {
                family: "layer-operations",
                variant: "rotation-scratch-absent",
                arch: "",
                quant: "quantized",
            })?;
            let elements = checked_mul(rows, weight.k, "rotation scratch")?;
            let scratch = view(rotation, 0, elements);
            if matches!(
                weight.dtype,
                DType::MQ4G256V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA
            ) {
                if rows > 1 {
                    hip(gpu.rotate_x_mq_batched(input, &scratch, weight.k, rows))?;
                } else {
                    hip(gpu.rotate_x_mq(input, &scratch, weight.k))?;
                }
            } else {
                hip(gpu.rotate_x_mq_128_v2(input, &scratch, weight.k, rows))?;
            }
            Some(scratch)
        }
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "layer-operations",
                variant: "unprojectable-payload",
                arch: "",
                quant: "unsupported",
            })
        }
    };
    let x = rotated.as_ref().unwrap_or(input);
    if weight.dtype == DType::MQ6G256V2 && (2..=4).contains(&rows) {
        return hip(gpu.gemm_mq6g256v2_f32_rows(weight.buf, x, output, weight.m, weight.k, rows));
    }
    let result = match (weight.dtype, rows > 1) {
        (DType::BF16, false) => gpu.gemv_bf16_xf32(weight.buf, x, output, weight.m, weight.k),
        (DType::BF16, true) => {
            // gfx1151 long prefill: the KLD-gated F16 WMMA route, else exact.
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
        // disagree with decode. Same reasoning as gemma4's `lowered` path, which
        // maps Q8_0 to `GemmQ8_0BatchedF32Chunked` for the same reason.
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
    // Qwen4-tuned multi-row: the up projection feeds the branch mix directly;
    // bitwise identical to the GEMM + hyper_read_projected pair below.
    let up_fused = gpu.arch_caps.qwen4_tuned_routes()
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
    if f16 {
        let x16 = hip(gpu.qwen4_f16_x_scratch(op.rows * wide))?;
        hip(hyper_norm_f16(gpu, &norm, &x16, !wmma_read))?;
        hip(gpu.gemm_bf16_xf16_f16_wmma(
            op.input_mix_down.buf,
            &x16,
            &low,
            op.input_mix_down.m,
            op.input_mix_down.k,
            op.rows,
        ))?;
        normalized_f16 = Some(x16);
    } else {
        hip(hyper_norm(gpu, &norm))?;
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
    if gpu.arch_caps.qwen4_tuned_routes() {
        hip(hc_activation_fused_f32(
            gpu,
            &HcActivationFused {
                values: &low,
                scale: 1.0 / op.branches as f32,
                bf16_out: low_bf16.as_ref(),
            },
        ))?;
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
    let wide = checked_mul(op.branches, op.hidden, "hyper write wide")?;
    let input = view(op.input, 0, op.rows * wide);
    let normalized = view(op.normalized, 0, op.rows * wide);
    let mixed = view(op.mixed, 0, op.rows * op.hidden);
    let gates = view(op.gates, 0, op.rows * op.branches);
    let output = view(op.output, 0, op.rows * wide);
    // Qwen4-tuned multi-row: one launch normalizes each row into LDS and projects
    // the BF16 gate from there; `normalized` (read by nothing below) is not
    // written.  Bitwise identical to hyper_norm + project_weight.
    let fused = gpu.arch_caps.qwen4_tuned_routes()
        && op.rows > 1
        && op.block_inject.dtype == DType::BF16
        && op.block_inject.m == op.branches
        && op.block_inject.k == wide
        && HyperNormGate::supports(op.branches, op.hidden);
    if fused {
        hip(hyper_norm_gate(
            gpu,
            &HyperNormGate {
                input: &input,
                norm_weight: op.norm_weight,
                gate_weight: op.block_inject.buf,
                gates: &gates,
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
                input: &input,
                norm_weight: op.norm_weight,
                normalized: &normalized,
                branches: op.branches,
                hidden: op.hidden,
                state_bf16: op.state_bf16,
            },
        ))?;
        project_weight(
            gpu,
            &op.block_inject,
            &normalized,
            &gates,
            op.rows,
            Some(op.rotation),
        )?;
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
}

impl GatedDeltaNetOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        self.validate_layout()
    }

    fn validate_layout(&self) -> Result<(), DispatchError> {
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
        require_tensor(self.input, rows_input, DType::F32, "gated delta input")?;
        require_tensor(self.projection, qkv, DType::F32, "gated delta projection")?;
        require_tensor(
            self.projection2,
            qkv,
            DType::F32,
            "gated delta projection scratch",
        )?;
        require_tensor(
            self.recurrent,
            checked_mul(value, self.key_dim, "gated delta state")?,
            DType::F32,
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
        require_tensor(self.a, rows_heads, DType::F32, "gated delta A")?;
        require_tensor(self.b, rows_heads, DType::F32, "gated delta B")?;
        require_tensor(self.gate, rows_heads, DType::F32, "gated delta gate")?;
        require_tensor(self.beta, rows_heads, DType::F32, "gated delta beta")?;
        require_tensor(
            self.recurrent_output,
            rows_value,
            DType::F32,
            "gated delta recurrent output",
        )?;
        require_tensor(self.z_output, rows_value, DType::F32, "gated delta Z")?;
        require_tensor(
            self.output_scratch,
            rows_value,
            DType::F32,
            "gated delta output scratch",
        )?;
        require_tensor(
            self.bf16_scratch,
            value.max(self.input_width),
            DType::BF16,
            "gated delta BF16 scratch",
        )?;
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
        require_tensor(
            self.output_tensor,
            rows_input,
            DType::F32,
            "gated delta output",
        )?;
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
        require_weight(
            &self.output,
            self.input_width,
            value,
            "gated delta output projection",
        )?;
        Ok(())
    }
}

pub fn execute_gated_delta_net(
    gpu: &mut Gpu,
    op: &GatedDeltaNetOp<'_>,
) -> Result<(), DispatchError> {
    let qk = op.key_heads * op.key_dim;
    let value = op.value_heads * op.value_dim;
    let qkv = 2 * qk + value;
    let mut projection = view(op.projection, 0, op.rows * qkv);
    let projection2 = view(op.projection2, 0, op.rows * qkv);
    let a = view(op.a, 0, op.rows * op.value_heads);
    let b = view(op.b, 0, op.rows * op.value_heads);
    let gate = view(op.gate, 0, op.rows * op.value_heads);
    let beta = view(op.beta, 0, op.rows * op.value_heads);
    let z = view(op.z_output, 0, op.rows * value);
    let history_rows = op.conv_kernel.saturating_sub(1);
    let persistent_batch = gpu.arch_caps.qwen4_tuned_routes()
        && op.rows > 1
        && op.key_dim == 128
        && op.value_dim == 128
        && op.conv_kernel == 4;
    let recurrent_output = view(op.recurrent_output, 0, op.rows * value);
    let dims = GatedDeltaStepBatched {
        projection: &projection2,
        gate: &gate,
        beta: &beta,
        state: op.recurrent,
        output: &recurrent_output,
        rows: op.rows,
        qkv_width: qkv,
        key_heads: op.key_heads,
        value_heads: op.value_heads,
        key_dim: op.key_dim,
        value_dim: op.value_dim,
    };
    let chunked = persistent_batch && gated_delta_chunk_route(gpu, &dims);
    // On the chunked route the qkv projection is read (by the convolution)
    // only through its BF16 rounding, so the MQ6 GEMM stores it as BF16 bits.
    let bf16_store = |w: &WeightRef<'_>, gpu: &Gpu| {
        chunked && w.dtype == DType::MQ6G256V2 && gpu.gemm_mq6g256v2_xf16_applies(w.k, op.rows)
    };
    if bf16_store(&op.qkv, gpu) {
        projection.dtype = DType::BF16;
    }
    project_weights(
        gpu,
        op.input,
        op.rows,
        Some(op.rotation),
        &[
            (&op.qkv, &projection),
            (&op.in_proj_a, &a),
            (&op.in_proj_b, &b),
            (&op.z, &z),
        ],
    )?;
    let mut gdn_output = view(op.output_scratch, 0, op.rows * value);
    if persistent_batch {
        let start_cursor = op.start_position % history_rows;
        // The F16 prefill route's chunked recurrence reads the convolution
        // output as packed BF16 (every value is BF16-rounded already).
        let mut conv_output = view(op.projection2, 0, op.rows * qkv);
        if chunked {
            conv_output.dtype = DType::BF16;
        }
        hip(gated_delta_conv_batched(
            gpu,
            &GatedDeltaConvBatched {
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
            },
        ))?;
        hip(gated_delta_params_batched(
            gpu,
            &GatedDeltaParamsBatched {
                a: &a,
                b: &b,
                a_log: op.a_log,
                dt_bias: op.dt_bias,
                gate: &gate,
                beta: &beta,
                rows: op.rows,
                heads: op.value_heads,
            },
        ))?;
        // The chunked route's output is BF16-rounded: stored as BF16 when the
        // output projection rotates it straight to F16 (half the bytes).
        if bf16_store(&op.output, gpu) {
            gdn_output.dtype = DType::BF16;
        }
        let step = GatedDeltaStepBatched {
            projection: &conv_output,
            ..dims
        };
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
            hip(gated_delta_step_gate_wmma(gpu, &step, &gated))?;
        } else {
            hip(gated_delta_step_batched(gpu, &step))?;
            hip(gated_delta_gate_batched(gpu, &gated))?;
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
            hip(gated_delta_conv(
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
            ))?;
            let a_row = view(&a, row * op.value_heads, op.value_heads);
            let b_row = view(&b, row * op.value_heads, op.value_heads);
            let gate_row = view(op.gate, row * op.value_heads, op.value_heads);
            let beta_row = view(op.beta, row * op.value_heads, op.value_heads);
            hip(gated_delta_params(
                gpu,
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
            hip(gated_delta_step(
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
                },
            ))?;
            let bf16 = view(op.bf16_scratch, 0, value);
            hip(bf16_roundtrip_f32(
                gpu,
                &Bf16Roundtrip {
                    input: &recurrent_output,
                    scratch: &bf16,
                    output: &recurrent_output,
                    elements: value,
                },
            ))?;
            let z_row = view(&z, row * value, value);
            let gdn_output = view(op.output_scratch, row * value, value);
            hip(gated_delta_gate(
                gpu,
                &GatedDeltaGate {
                    recurrent_output: &recurrent_output,
                    z: &z_row,
                    norm: op.norm,
                    output: &gdn_output,
                    value_heads: op.value_heads,
                    value_dim: op.value_dim,
                },
            ))?;
        }
    }
    let output_batch = view(op.output_tensor, 0, op.rows * op.output.m);
    project_weight(
        gpu,
        &op.output,
        &gdn_output,
        &output_batch,
        op.rows,
        Some(op.rotation),
    )?;
    // One elementwise launch over all rows; identical to the per-row scratch
    // round trip for every non-NaN value.
    hip(gpu.bf16_round_trip_f32(&output_batch))?;
    Ok(())
}

/// Borrowed cache tensors plus scalar metadata for one operation.  The
/// architecture commits these scalar values to persistent state after the
/// shared step list succeeds.
pub struct IndexedAttentionState<'a> {
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
}

impl IndexedAttentionOp<'_> {
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
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
            || self.state.full_capacity == 0
            || self.state.raw_capacity == 0
            || self.state.pooled_capacity == 0
            || self.state.selected_capacity == 0
            || self.state.position_capacity == 0
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
        if self.state.position > self.state.position_capacity
            || self.state.full_len > self.state.full_capacity
            || self.state.raw_len > self.state.raw_capacity
            || self.state.pooled_len > self.state.pooled_capacity
            || self.state.selected_len > self.state.selected_capacity
            || self.state.position != self.state.full_len
            || self.state.position != self.state.raw_len
            || self.state.pooled_len != self.state.position / self.compress
            || self.state.selected_len > self.state.position
            || self.rows > self.state.position_capacity - self.state.position
        {
            return Err(DispatchError::Hip(
                "indexed attention state length/capacity mismatch".into(),
            ));
        }
        require_tensor(
            self.input,
            rows_input,
            DType::F32,
            "indexed attention input",
        )?;
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
        require_tensor(
            self.qsa_output,
            rows_q,
            DType::F32,
            "indexed attention output scratch",
        )?;
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
        require_tensor(
            self.state.full_keys,
            checked_mul(
                self.state.full_capacity,
                kv_width,
                "indexed attention full keys",
            )?,
            DType::F32,
            "indexed attention full keys",
        )?;
        require_tensor(
            self.state.full_values,
            checked_mul(
                self.state.full_capacity,
                kv_width,
                "indexed attention full values",
            )?,
            DType::F32,
            "indexed attention full values",
        )?;
        require_tensor(
            self.state.raw_index_keys,
            checked_mul(
                self.state.raw_capacity,
                index_kv_width,
                "indexed attention raw keys",
            )?,
            DType::F32,
            "indexed attention raw keys",
        )?;
        require_tensor(
            self.state.pooled_keys,
            checked_mul(
                self.state.pooled_capacity,
                index_kv_width,
                "indexed attention pooled keys",
            )?,
            DType::F32,
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
        require_tensor(
            self.bf16_scratch,
            q_width.max(kv_width),
            DType::BF16,
            "indexed attention BF16 scratch",
        )?;
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
        require_weight(
            &self.output,
            self.input_width,
            q_width,
            "indexed attention output projection",
        )?;
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
        let budget_blocks = self.budget / self.compress;
        let selected_len = (budget_blocks.min(complete) * self.compress + final_position
            - complete * self.compress)
            .min(self.state.selected_capacity);
        Ok((
            final_position,
            final_position,
            complete,
            selected_len,
            final_position,
        ))
    }
}

pub fn execute_indexed_attention(
    gpu: &mut Gpu,
    op: &IndexedAttentionOp<'_>,
) -> Result<(), DispatchError> {
    let index_width = (op.index_heads + op.index_kv_heads) * op.index_dim;
    let index_q_width = op.index_heads * op.index_dim;
    let index_kv_width = op.index_kv_heads * op.index_dim;
    let q_width = op.heads * op.head_dim;
    let kv_width = op.kv_heads * op.head_dim;
    let initial_position = op.state.position;
    let (_, _, complete, _, _) = op.next_lengths()?;
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
    project_weights(
        gpu,
        op.input,
        op.rows,
        Some(op.rotation),
        &[
            (&op.indexer_qk, &index_batch),
            (&op.q, &qgate_batch),
            (&op.k, &k_batch),
            (&op.v, &v_batch),
        ],
    )?;

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
    // index_kv_width`) against the base tensor, and the recorder declares it, so
    // the tape keeps a position-independent pointer and replay re-derives the
    // offset for its own position instead of replaying the capture-position row.
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
            kv_width,
        },
    ))?;

    // Every QSA launch declares a position-independent shape: the pool grid and
    // both dynamic-LDS reservations come from the declared capacities while the
    // active lengths stay scalars. Measured bit-identical to the position-derived
    // shapes with no throughput delta (docs/design/qwen4-program-retained-pm4.md).
    if complete > 0 {
        hip(indexed_attention_pool_rope(
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
    let budget_blocks = op.budget / op.compress;
    hip(indexed_attention_select_batch(
        gpu,
        &IndexedAttentionSelectBatch {
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
        },
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
            shape_selected: op.state.selected_capacity,
        },
    ))?;
    project_weight(
        gpu,
        &op.output,
        &qsa_output_batch,
        &view(op.attention_output, 0, op.rows * op.output.m),
        op.rows,
        Some(op.rotation),
    )?;
    let final_selected = view(
        &selected_batch,
        (op.rows - 1) * op.state.selected_capacity * std::mem::size_of::<i32>(),
        op.state.selected_capacity * std::mem::size_of::<i32>(),
    );
    // A recorded launch, not a `copy_d2d`: the retained tape replays dispatches,
    // so a device copy inside the body would be state the replay cannot
    // reproduce. `copy_f32_buffer` moves the same bytes with an explicit ABI.
    hip(gpu.copy_f32_buffer(
        op.state.selected_indices,
        &final_selected,
        op.state.selected_capacity,
    ))?;
    Ok(())
}

/// Grouped causal convolution contract used by PLE-like layers.  Geometry and
/// kernel/dilation are supplied by the architecture; the low-level wrapper
/// does not know a model's fixed layer index or hidden width.
pub struct GroupedDepthwiseOp<'a> {
    /// The HC streams hold BF16 bits for this forward
    /// ([`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
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
            }
        }
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
}
