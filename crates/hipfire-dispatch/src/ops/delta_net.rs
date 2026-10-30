// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
// DeltaNet linear-attention kernel extensions.
//
// These operations are unique to DeltaNet's linear attention layers:
// the gated linear recurrence with quantized/FP32 state, conv-state ring
// buffer management, and tree-batched speculative-decode variants. They
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

/// Parameters for tree-batched DeltaNet (speculative-decode path).
///
/// Adds a tape buffer and parent-indices array so sibling tokens read
/// the correct parent's post-update state.
pub struct DeltaNetTreeParams<'a> {
    pub q_batch: &'a GpuTensor,
    pub k_batch: &'a GpuTensor,
    pub v_batch: &'a GpuTensor,
    pub gate_batch: &'a GpuTensor,
    pub beta_batch: &'a GpuTensor,
    pub s_q8_init: &'a GpuTensor,
    pub s_scales_init: &'a GpuTensor,
    pub s_tape_q8: &'a GpuTensor,
    pub s_tape_scales: &'a GpuTensor,
    pub parent_indices: &'a GpuTensor,
    pub output_batch: &'a GpuTensor,
    pub n_tokens: usize,
    pub n_heads: usize,
    pub head_dim: usize,
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

pub trait DeltaNetOps {
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

    /// Run tree-batched DeltaNet (speculative-decode path).
    ///
    /// Only supported with Q8 state (the tree tape mechanism is Q8-specific).
    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String>;

    /// Zero the conv-state ring buffer.
    fn reset_conv_state(
        &self,
        gpu: &mut Gpu,
        state: &GpuTensor,
        conv_state_size: usize,
    ) -> Result<(), String>;
}

// ── Default implementations ────────────────────────────

impl DeltaNetOps for () {
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
                None,
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
                None,
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

    fn run_delta_net_tree(&self, gpu: &mut Gpu, params: &DeltaNetTreeParams) -> Result<(), String> {
        gpu.gated_delta_net_q8_tree_batch_seq(
            params.q_batch,
            params.k_batch,
            params.v_batch,
            params.gate_batch,
            params.beta_batch,
            params.s_q8_init,
            params.s_scales_init,
            params.s_tape_q8,
            params.s_tape_scales,
            params.parent_indices,
            params.output_batch,
            params.n_tokens,
            params.n_heads,
            params.head_dim,
        )
        .map_err(|e| format!("delta_net_tree: {e:?}"))
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

#[cfg(test)]
mod tests {
    use super::DeltaNetOps;

    // Verify the trait is implemented for () without needing a real Gpu.
    fn _assert_impl() {
        fn needs_delta_net_ops<T: DeltaNetOps>() {}
        needs_delta_net_ops::<()>();
    }
}

/// Batched sigmoid(beta) / alpha-gate prep over `batch_size` rows
/// (`Gpu::fused_sigmoid_alpha_gate_f32_batched`), forwarded unchanged.
pub fn sigmoid_alpha_gate_batched(
    gpu: &mut Gpu,
    beta: &GpuTensor,
    alpha: &GpuTensor,
    dt_bias: &GpuTensor,
    a_log: &GpuTensor,
    n: usize,
    batch_size: usize,
) -> hip_bridge::HipResult<()> {
    gpu.fused_sigmoid_alpha_gate_f32_batched(beta, alpha, dt_bias, a_log, n, batch_size)
}

// ── Generic gated delta rule (per-head or per-key-channel decay) ─────────
//
// Independent of `DeltaNetStepParams`/`DeltaNetBatchParams`: those carry one
// scalar gate per (token, head) and FP32/Q8/Q4 state selection. This op carries
// explicit decay geometry, head mapping and state strides, and runs N>=1 tokens
// in ONE launch (state tile resident in registers across tokens).

/// Axis of the log-decay operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DecayAxis {
    /// `log_decay[t, h]`: one scalar per (token, head).
    PerHead,
    /// `log_decay[t, h, k]`: one value per key channel.
    PerKeyChannel,
}

/// Element strides (f32 elements) of every operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedDeltaStrides {
    pub q_row: usize,
    pub k_row: usize,
    pub v_row: usize,
    pub decay_row: usize,
    pub beta_row: usize,
    pub out_row: usize,
    pub state_slot: usize,
    pub state_head: usize,
    /// Elements between consecutive value rows of one head's `[V,K]` state.
    pub state_row: usize,
}

impl GatedDeltaStrides {
    /// Dense `[N,Hqk,K]` / `[N,H,V]` / `[N,H]` rows with a dense `[H,V,K]`
    /// state per slot.
    pub fn contiguous(
        n_heads: usize,
        qk_group: usize,
        key_dim: usize,
        value_dim: usize,
        decay_axis: DecayAxis,
    ) -> Self {
        let hq = n_heads / qk_group.max(1);
        Self {
            q_row: hq * key_dim,
            k_row: hq * key_dim,
            v_row: n_heads * value_dim,
            decay_row: match decay_axis {
                DecayAxis::PerHead => n_heads,
                DecayAxis::PerKeyChannel => n_heads * key_dim,
            },
            beta_row: n_heads,
            out_row: n_heads * value_dim,
            state_slot: n_heads * value_dim * key_dim,
            state_head: value_dim * key_dim,
            state_row: key_dim,
        }
    }
}

/// One gated-delta-rule layer step over `n_tokens` rows.
///
/// Per token: `S *= exp(log_decay)` (per key column), `pred = S·k`,
/// `S += beta * (v - pred) ⊗ k`, `y = S·q` (read after write). State is FP32
/// `[H,V,K]` (value-major, key contiguous), updated in place. Operands are FP32
/// planes; BF16 boundaries are applied by the producer, except the optional
/// `round_output_bf16` rounding of `y`.
pub struct GatedDeltaOp<'a> {
    pub query: &'a GpuTensor,
    pub key: &'a GpuTensor,
    /// Value operand (already write-ratio scaled by the caller when needed).
    pub value: &'a GpuTensor,
    pub log_decay: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    /// Optional device `i32` slot selector (graph-capture safe); overrides `slot`.
    pub slot_idx: Option<&'a GpuTensor>,
    /// Optional device `i32` token count (<= `n_tokens`).
    pub n_dev: Option<&'a GpuTensor>,
    pub n_tokens: usize,
    pub n_heads: usize,
    /// Value heads per q/k head (1 = no repetition).
    pub qk_group: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub decay_axis: DecayAxis,
    pub round_output_bf16: bool,
    pub slot: usize,
    pub strides: GatedDeltaStrides,
}

/// Execute `op` with the generic serial vector-decay kernel. Unsupported
/// geometry is a named refusal, never a per-token loop.
pub fn execute_gated_delta(
    gpu: &mut Gpu,
    op: &GatedDeltaOp<'_>,
) -> Result<(), crate::types::DispatchError> {
    use crate::types::DispatchError;
    if op.key_dim != rdna_compute::gated_delta_vector::GATED_DELTA_VECTOR_HEAD_DIM
        || op.value_dim != rdna_compute::gated_delta_vector::GATED_DELTA_VECTOR_HEAD_DIM
    {
        return Err(DispatchError::UnsupportedVariant {
            family: "gated_delta",
            variant: "vector_serial_head_dim",
            arch: "any",
            quant: "fp32",
        });
    }
    let s = &op.strides;
    let launch = rdna_compute::gated_delta_vector::GatedDeltaVectorLaunch {
        query: op.query,
        key: op.key,
        value: op.value,
        log_decay: op.log_decay,
        beta: op.beta,
        state: op.state,
        output: op.output,
        slot_idx: op.slot_idx,
        n_dev: op.n_dev,
        n_tokens: op.n_tokens,
        n_heads: op.n_heads,
        qk_group: op.qk_group,
        head_dim: op.key_dim,
        decay_per_key: op.decay_axis == DecayAxis::PerKeyChannel,
        round_out_bf16: op.round_output_bf16,
        slot: op.slot,
        q_row_stride: s.q_row,
        k_row_stride: s.k_row,
        v_row_stride: s.v_row,
        decay_row_stride: s.decay_row,
        beta_row_stride: s.beta_row,
        out_row_stride: s.out_row,
        state_head_stride: s.state_head,
        state_row_stride: s.state_row,
        state_slot_stride: s.state_slot,
    };
    gpu.gated_delta_net_vector_serial_f32(&launch)
        .map_err(|e| DispatchError::Hip(e.to_string()))
}
