// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// Qwen3.5 model-specific kernel extensions.
//
// These operations are unique to Qwen3.5's linear attention layers (DeltaNet):
// the gated linear recurrence with quantized/FP32 state, conv-state ring
// buffer management. They
// don't fit into the standard dispatch families because the state is
// model-owned and the recurrence is an inherently sequential kernel.

use rdna_compute::{Gpu, GpuTensor};

// ── State quantization ─────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

// ── Parameter structs ──────────────────────────────────

/// Parameters for a single-token DeltaNet state update.
///
/// The gated delta net recurrence:
///   S' = gate · S + beta · (k ⊗ v)
///   output = S · q
///
/// where S is the recurrent state (n_heads × head_dim × head_dim),
/// quantized per the `quant` field.
pub struct DeltaNetStepParams<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
}

/// Parameters for batched sequential DeltaNet updates (prefill path).
///
/// Q, K, V, gate, beta, and output are batched [n_tokens, n_heads, head_dim].
/// The state is updated in-place for all n_tokens.
pub struct DeltaNetBatchParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub s_scales: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub quant: StateQuant,
}

/// Parameters for DeltaNet conv-state ring-buffer management.
pub struct ConvStateParams<'a> {
    pub state: &'a GpuTensor,
    pub input: &'a GpuTensor,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub position: usize,
}

// ── Trait ──────────────────────────────────────────────

pub trait Qwen35ModelExt {
    /// Run a single-token DeltaNet state update.
    ///
    /// Dispatches to `gated_delta_net_f32`, `gated_delta_net_q8`,
    /// or `gated_delta_net_q4` based on `params.quant`.
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String>;

    /// Run batched sequential DeltaNet updates (prefill).
    ///
    /// Dispatches to `gated_delta_net_q8_batch_seq` for Q8 state,
    /// or loops the single-token kernel for FP32/Q4.
    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String>;

    /// Zero the conv-state ring buffer.
    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        conv_state_size: usize,
    ) -> Result<(), String>;
}

// ── Default implementations ────────────────────────────

impl Qwen35ModelExt for () {
    fn run_delta_net_step(&self, gpu: &mut Gpu, params: &DeltaNetStepParams) -> Result<(), String> {
        match params.quant {
            StateQuant::FP32 => gpu.gated_delta_net_f32(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::Q8 => gpu.gated_delta_net_q8(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.s_scales,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
            StateQuant::Q4 => gpu.gated_delta_net_q4(
                params.q,
                params.k,
                params.v,
                params.gate,
                params.beta,
                params.state,
                params.s_scales,
                params.output,
                1,
                params.n_heads,
                params.head_dim,
            ),
        }
        .map_err(|e| format!("delta_net_step: {e:?}"))
    }

    fn run_delta_net_batch(
        &self,
        gpu: &mut Gpu,
        params: &DeltaNetBatchParams,
    ) -> Result<(), String> {
        match params.quant {
            StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                params.q_batch,
                params.k_batch,
                params.v_batch,
                params.gate_batch,
                params.beta_batch,
                params.state,
                params.s_scales,
                params.output_batch,
                params.n_tokens,
                params.n_heads,
                params.head_dim,
            ),
            _ => {
                // FP32/Q4: loop single-token kernel.
                // Q4 batch variant doesn't exist in the kernel set yet.
                let stride = params.n_heads * params.head_dim;
                for i in 0..params.n_tokens {
                    let q = params.q_batch.sub_offset(i * stride, stride);
                    let k = params.k_batch.sub_offset(i * stride, stride);
                    let v = params.v_batch.sub_offset(i * stride, stride);
                    let g = params
                        .gate_batch
                        .sub_offset(i * params.n_heads, params.n_heads);
                    let b = params
                        .beta_batch
                        .sub_offset(i * params.n_heads, params.n_heads);
                    let o = params.output_batch.sub_offset(i * stride, stride);
                    match params.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &q,
                            &k,
                            &v,
                            &g,
                            &b,
                            params.state,
                            &o,
                            1,
                            params.n_heads,
                            params.head_dim,
                        ),
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &q,
                            &k,
                            &v,
                            &g,
                            &b,
                            params.state,
                            params.s_scales,
                            &o,
                            1,
                            params.n_heads,
                            params.head_dim,
                        ),
                        _ => unreachable!(),
                    }
                    .map_err(|e| format!("delta_net_batch token {i}: {e:?}"))?;
                }
                Ok(())
            }
        }
        .map_err(|e| format!("delta_net_batch: {e:?}"))
    }

    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        _conv_state_size: usize,
    ) -> Result<(), String> {
        gpu.hip
            .memset(&state.buf, 0, state.buf.size())
            .map_err(|e| format!("reset_conv_state: {e:?}"))
    }
}
