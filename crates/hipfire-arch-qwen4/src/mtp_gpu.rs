// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 MTP GPU execution resources.
//!
//! This module is the only production MTP implementation for Qwen4.  It owns
//! one bound MTP MoE table, one reusable operator scratch set, and one bounded
//! device-side QSA state. The opt-in CPU/reference equations in
//! `reference_mtp` are parity fixtures, never called from this path.

use crate::config::Qwen4Config;
use crate::gpu_forward::{
    dense_ref, hyper_desc, hyper_read_desc, qsa_desc, seal_moe_decode, Qwen4GpuForwardError,
    Qwen4MoeLayerRuntime, Qwen4MoeScratch,
};
use crate::program::{Qwen4HyperReadWeights, Qwen4HyperWriteWeights};
use crate::weights::{Qwen4Weights, WeightError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, BroadcastAddOp, ClearOp, DraftHead, DraftHeadLayout,
    DraftHeadRequestState,
    DraftHeadPolicy, EmbeddingOp, HyperNormOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode,
    IndexedAttentionOp, IndexedAttentionState, ProjectOp, Step,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::spec::SpecGrammar;
use rdna_compute::tensor_ops::QsaKvFormat;
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;
use std::fmt;

const MTP_BRANCHES: usize = 4;
/// Prompt tokens per batched MTP K/V append ([`Qwen4MtpGpu::append_rows`]).
pub(crate) const MTP_APPEND_ROWS: usize = 256;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MTP_MODEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_mtp_model_id() -> u64 {
    NEXT_MTP_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

fn invalid(message: impl Into<String>) -> MtpGpuError {
    MtpGpuError::Invalid(message.into())
}

/// `rows`-row HC read of the MTP streams into `hc_mixed`.
fn hc_read<'a>(
    read: &Qwen4HyperReadWeights<'a>,
    scratch: &'a MtpGpuScratch,
    config: &Qwen4Config,
    rows: usize,
) -> HyperReadOp<'a> {
    HyperReadOp {
        state_bf16: false,
        input: &scratch.wide,
        norm_weight: read.norm,
        input_mix_down: read.input_mix_down,
        input_mix_up: read.input_mix_up,
        normalized: &scratch.hc_normalized,
        low: &scratch.hc_low,
        up: &scratch.hc_up,
        mixed: &scratch.hc_mixed,
        bf16_scratch: &scratch.hc_bf16,
        rows,
        branches: config.hc_count,
        hidden: config.hidden_size,
        low_rank: config.hc_lowrank,
        rotation: &scratch.rotation,
    }
}

/// One-row HC write of `mixed` into the MTP streams, in place.
fn hc_write<'a>(
    write: &Qwen4HyperWriteWeights<'a>,
    mixed: &'a GpuTensor,
    scratch: &'a MtpGpuScratch,
    config: &Qwen4Config,
) -> HyperWriteOp<'a> {
    HyperWriteOp {
        state_bf16: false,
        input: &scratch.wide,
        norm_weight: write.norm,
        block_inject: write.block_inject,
        normalized: &scratch.hc_normalized,
        mixed,
        gates: &scratch.hc_gates,
        output: &scratch.wide,
        rows: 1,
        branches: config.hc_count,
        hidden: config.hidden_size,
        rotation: &scratch.rotation,
    }
}

/// Errors from native GPU MTP execution and resource management.
#[derive(Debug)]
pub enum MtpGpuError {
    Hip(hip_bridge::HipError),
    Weights(WeightError),
    Forward(Qwen4GpuForwardError),
    Invalid(String),
    Grammar(String),
}

impl fmt::Display for MtpGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => write!(f, "Qwen4 MTP GPU HIP error: {error}"),
            Self::Weights(error) => write!(f, "Qwen4 MTP GPU weights error: {error}"),
            Self::Forward(error) => write!(f, "Qwen4 MTP GPU forward error: {error}"),
            Self::Invalid(error) => write!(f, "Qwen4 MTP GPU invalid input: {error}"),
            Self::Grammar(error) => write!(f, "Qwen4 MTP grammar error: {error}"),
        }
    }
}

impl std::error::Error for MtpGpuError {}

impl From<hip_bridge::HipError> for MtpGpuError {
    fn from(error: hip_bridge::HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<WeightError> for MtpGpuError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

impl From<Qwen4GpuForwardError> for MtpGpuError {
    fn from(error: Qwen4GpuForwardError) -> Self {
        Self::Forward(error)
    }
}

impl From<DispatchError> for MtpGpuError {
    fn from(error: DispatchError) -> Self {
        Self::Forward(Qwen4GpuForwardError::Dispatch(error.to_string()))
    }
}

/// Fixed operator buffers for `rows` MTP tokens (one for decode steps,
/// [`MTP_APPEND_ROWS`] for the batched prompt append).  Every field is
/// allocated once when the model's MTP capability is attached; calls only
/// create subviews.  The MoE buffers are single-token.
pub struct MtpGpuScratch {
    token_ids: GpuTensor,
    embedding_rot: GpuTensor,
    token_embedding: GpuTensor,
    embedding_norm: GpuTensor,
    hidden_norm: GpuTensor,
    projected_embedding: GpuTensor,
    projected_hidden: GpuTensor,
    wide: GpuTensor,
    backbone_hidden: GpuTensor,
    hc_normalized: GpuTensor,
    hc_low: GpuTensor,
    hc_up: GpuTensor,
    hc_mixed: GpuTensor,
    hc_gates: GpuTensor,
    /// BF16 scratch of the shared HC read and indexed attention.
    hc_bf16: GpuTensor,
    rotation: GpuTensor,
    index: GpuTensor,
    q_and_gate: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    qsa_output: GpuTensor,
    /// Per-row selection of a fresh QSA select (mirrored into the state).
    qsa_selected: GpuTensor,
    router_logits: GpuTensor,
    moe_x_rot: GpuTensor,
    moe_gate_up: GpuTensor,
    moe_gate: GpuTensor,
    moe_up: GpuTensor,
    moe_hidden: GpuTensor,
    moe_output: GpuTensor,
    moe_gate_batch: GpuTensor,
    moe_up_batch: GpuTensor,
    moe_rot_batch: GpuTensor,
    moe_topk_indices: GpuTensor,
    moe_topk_weights: GpuTensor,
    moe_down_expanded: GpuTensor,
    moe_scalar: GpuTensor,
    host_token_bytes: [u8; 4],
}

/// Allocations of one [`MtpGpuScratch`], in field order.
const MTP_SCRATCH_TENSORS: usize = 36;

impl MtpGpuScratch {
    /// Element count and dtype of each scratch tensor for `rows` token rows,
    /// in field order.
    fn shapes(
        config: &Qwen4Config,
        rows: usize,
    ) -> Result<[(usize, DType); MTP_SCRATCH_TENSORS], MtpGpuError> {
        let hidden = config.hidden_size;
        let wide = MTP_BRANCHES
            .checked_mul(hidden)
            .ok_or_else(|| invalid("MTP wide dimension overflow"))?;
        let q_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| invalid("MTP Q width overflow"))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| invalid("MTP KV width overflow"))?;
        let index_width = (config.indexer_n_heads + config.indexer_kv_heads)
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| invalid("MTP index width overflow"))?;
        let hc_up = wide
            .checked_mul(config.hc_lowrank)
            .ok_or_else(|| invalid("MTP HC up scratch overflow"))?;
        let max_rotation = wide.max(hidden).max(config.hc_lowrank).max(q_width);
        let routed = config.num_experts_per_tok * config.moe_intermediate_size;
        Ok([
            (rows * std::mem::size_of::<i32>(), DType::Raw),
            (rows * hidden, DType::F32),
            (rows * hidden, DType::F32),
            (rows * hidden, DType::F32),
            (rows * wide, DType::F32),
            (rows * hidden, DType::F32),
            (rows * wide, DType::F32),
            (rows * wide, DType::F32),
            (wide, DType::F32),
            (rows * wide, DType::F32),
            (rows * config.hc_lowrank, DType::F32),
            (hc_up.max(rows * wide), DType::F32),
            (rows * hidden, DType::F32),
            (config.hc_count, DType::F32),
            (
                hidden.max(config.hc_lowrank).max(q_width).max(kv_width),
                DType::BF16,
            ),
            (rows * max_rotation, DType::F32),
            (rows * index_width, DType::F32),
            (rows * 2 * q_width, DType::F32),
            (rows * kv_width, DType::F32),
            (rows * kv_width, DType::F32),
            (rows * q_width, DType::F32),
            (
                rows * config.qsa_selected_capacity() * std::mem::size_of::<i32>(),
                DType::Raw,
            ),
            (config.num_experts, DType::F32),
            (hidden, DType::F32),
            (2 * config.moe_intermediate_size, DType::F32),
            (config.moe_intermediate_size, DType::F32),
            (config.moe_intermediate_size, DType::F32),
            (config.moe_intermediate_size, DType::F32),
            (hidden, DType::F32),
            (routed, DType::F32),
            (routed, DType::F32),
            (routed, DType::F32),
            (config.num_experts_per_tok, DType::F32),
            (config.num_experts_per_tok, DType::F32),
            (
                config.num_experts_per_tok * hidden + hidden.div_ceil(4),
                DType::F32,
            ),
            (config.shared_expert_intermediate_size.max(1), DType::F32),
        ])
    }

    /// Device bytes [`Self::new`] allocates for `rows` token rows.
    fn device_bytes(config: &Qwen4Config, rows: usize) -> Result<usize, MtpGpuError> {
        Ok(Self::shapes(config, rows)?
            .iter()
            .map(|&(elements, dtype)| elements * dtype.size())
            .sum())
    }

    fn new(gpu: &mut Gpu, config: &Qwen4Config, rows: usize) -> Result<Self, MtpGpuError> {
        let shapes = Self::shapes(config, rows)?;
        let mut allocated = Vec::with_capacity(MTP_SCRATCH_TENSORS);
        let result = (|| {
            for (elements, dtype) in shapes {
                allocated.push(gpu.zeros(&[elements], dtype)?);
            }
            Ok::<(), MtpGpuError>(())
        })();
        if let Err(error) = result {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut next = || allocated.remove(0);
        Ok(Self {
            token_ids: next(),
            embedding_rot: next(),
            token_embedding: next(),
            embedding_norm: next(),
            hidden_norm: next(),
            projected_embedding: next(),
            projected_hidden: next(),
            wide: next(),
            backbone_hidden: next(),
            hc_normalized: next(),
            hc_low: next(),
            hc_up: next(),
            hc_mixed: next(),
            hc_gates: next(),
            hc_bf16: next(),
            rotation: next(),
            index: next(),
            q_and_gate: next(),
            qsa_k: next(),
            qsa_v: next(),
            qsa_output: next(),
            qsa_selected: next(),
            router_logits: next(),
            moe_x_rot: next(),
            moe_gate_up: next(),
            moe_gate: next(),
            moe_up: next(),
            moe_hidden: next(),
            moe_output: next(),
            moe_gate_batch: next(),
            moe_up_batch: next(),
            moe_rot_batch: next(),
            moe_topk_indices: next(),
            moe_topk_weights: next(),
            moe_down_expanded: next(),
            moe_scalar: next(),
            host_token_bytes: [0; 4],
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let tensors = [
            self.token_ids,
            self.embedding_rot,
            self.token_embedding,
            self.embedding_norm,
            self.hidden_norm,
            self.projected_embedding,
            self.projected_hidden,
            self.wide,
            self.backbone_hidden,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.hc_gates,
            self.hc_bf16,
            self.rotation,
            self.index,
            self.q_and_gate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
            self.qsa_selected,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.moe_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_scalar,
        ];
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MtpSnapshotDimensions {
    max_seq: usize,
    selected_capacity: usize,
    shape_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MtpStateMark {
    full_len: usize,
    raw_len: usize,
    pooled_len: usize,
    selected_len: usize,
    position: usize,
    step_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MtpGpuStateSnapshot {
    model_id: u64,
    request_epoch: u64,
    state_generation: u64,
    arena_generation: u64,
    dimensions: MtpSnapshotDimensions,
}

/// Reusable model-owned MTP rollback storage.  Full KV, raw index, and pooled
/// arenas are append-only; only selected indices and the wide hidden carry are
/// copied because those buffers are overwritten by each forward.
struct MtpGpuStateSnapshotArena {
    selected_indices: GpuTensor,
    selected_len_out: GpuTensor,
    wide_hidden: GpuTensor,
    mark: MtpStateMark,
    model_id: u64,
    active: bool,
    generation: u64,
}

impl MtpGpuStateSnapshotArena {
    fn new(
        gpu: &mut Gpu,
        selected_indices: &GpuTensor,
        selected_len_out: &GpuTensor,
        wide_hidden: &GpuTensor,
        model_id: u64,
        mark: MtpStateMark,
    ) -> Result<Self, MtpGpuError> {
        let selected_backup = gpu.zeros(&[selected_indices.numel()], DType::Raw)?;
        let selected_len_backup = match gpu.zeros(&[selected_len_out.numel()], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(selected_backup);
                return Err(error.into());
            }
        };
        let wide_backup = match gpu.zeros(&[wide_hidden.numel()], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(selected_backup);
                let _ = gpu.free_tensor(selected_len_backup);
                return Err(error.into());
            }
        };
        Ok(Self {
            selected_indices: selected_backup,
            selected_len_out: selected_len_backup,
            wide_hidden: wide_backup,
            mark,
            model_id,
            active: false,
            generation: 0,
        })
    }

    fn dimensions(state: &MtpGpuState) -> MtpSnapshotDimensions {
        let mut hash = 0xcbf29ce484222325u64;
        let mut mix = |value: usize| {
            hash ^= value as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        };
        mix(state.full_capacity);
        mix(state.raw_capacity);
        mix(state.pooled_capacity);
        mix(state.selected_capacity);
        mix(state.full_keys.numel());
        mix(state.full_values.numel());
        mix(state.raw_index_keys.numel());
        mix(state.pooled_keys.numel());
        mix(state.selected_indices.numel());
        mix(state.selected_indices.buf.size());
        mix(state.selected_len_out.numel());
        mix(state.selected_len_out.buf.size());
        mix(state.wide_hidden.numel());
        mix(state.wide_hidden.buf.size());
        MtpSnapshotDimensions {
            max_seq: state.full_capacity,
            selected_capacity: state.selected_capacity,
            shape_hash: hash,
        }
    }

    fn validate_layout(&self, state: &MtpGpuState) -> Result<(), MtpGpuError> {
        // Allocation capacities can differ for equal-shaped tensors.
        if self.selected_indices.numel() != state.selected_indices.numel()
            || self.selected_len_out.numel() != state.selected_len_out.numel()
            || self.wide_hidden.numel() != state.wide_hidden.numel()
        {
            return Err(invalid("MTP snapshot arena shape mismatch"));
        }
        Ok(())
    }

    fn validate_ticket(
        &self,
        state: &MtpGpuState,
        ticket: &MtpGpuStateSnapshot,
    ) -> Result<(), MtpGpuError> {
        if !self.active {
            return Err(invalid("MTP snapshot ticket is inactive or consumed"));
        }
        if ticket.model_id != self.model_id
            || ticket.model_id != state.model_id
            || ticket.request_epoch != state.request_epoch
            || ticket.state_generation != state.generation
            || ticket.arena_generation != self.generation
            || ticket.dimensions != Self::dimensions(state)
        {
            return Err(invalid("MTP snapshot ticket does not belong to this state"));
        }
        validate_mtp_mark(&self.mark, state)?;
        self.validate_layout(state)
    }

    fn invalidate(&mut self) {
        self.active = false;
        self.generation = self.generation.wrapping_add(1);
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [
            self.selected_indices,
            self.selected_len_out,
            self.wide_hidden,
        ] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}
fn validate_mtp_mark(mark: &MtpStateMark, state: &MtpGpuState) -> Result<(), MtpGpuError> {
    if mark.full_len > state.full_capacity
        || mark.raw_len > state.raw_capacity
        || mark.pooled_len > state.pooled_capacity
        || mark.selected_len > state.selected_capacity
        || mark.position > state.full_capacity
    {
        return Err(invalid("MTP snapshot mark exceeds state capacity"));
    }
    Ok(())
}

/// Device-resident bounded QSA state for the one MTP attention layer.
pub struct MtpGpuState {
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw_index_keys: GpuTensor,
    pooled_keys: GpuTensor,
    selected_indices: GpuTensor,
    selected_len_out: GpuTensor,
    wide_hidden: GpuTensor,
    full_capacity: usize,
    raw_capacity: usize,
    pooled_capacity: usize,
    selected_capacity: usize,
    position: usize,
    full_len: usize,
    raw_len: usize,
    pooled_len: usize,
    selected_len: usize,
    step_index: usize,
    request_epoch: u64,
    generation: u64,
    model_id: u64,
    snapshot_arena: MtpGpuStateSnapshotArena,
    /// Durable prefix-cache checkpoint (see `Qwen4State::capture_prefix`):
    /// separate from the speculative arena; `active` = valid.
    prefix_arena: Option<MtpGpuStateSnapshotArena>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MtpStateParityMetadata {
    pub(crate) full_len: usize,
    pub(crate) raw_len: usize,
    pub(crate) pooled_len: usize,
    pub(crate) selected_len: usize,
    pub(crate) position: usize,
    pub(crate) step_index: usize,
}

pub(crate) struct MtpStateParityBuffers<'a> {
    pub(crate) full_keys: &'a GpuTensor,
    pub(crate) full_values: &'a GpuTensor,
    pub(crate) raw_index_keys: &'a GpuTensor,
    pub(crate) pooled_keys: &'a GpuTensor,
    pub(crate) selected_indices: &'a GpuTensor,
    pub(crate) selected_len_out: &'a GpuTensor,
    pub(crate) wide_hidden: &'a GpuTensor,
}

impl MtpStateParityMetadata {
    fn from_state(state: &MtpGpuState) -> Self {
        Self {
            full_len: state.full_len,
            raw_len: state.raw_len,
            pooled_len: state.pooled_len,
            selected_len: state.selected_len,
            position: state.position,
            step_index: state.step_index,
        }
    }
}

impl MtpGpuState {
    fn mark(&self) -> MtpStateMark {
        MtpStateMark {
            full_len: self.full_len,
            raw_len: self.raw_len,
            pooled_len: self.pooled_len,
            selected_len: self.selected_len,
            position: self.position,
            step_index: self.step_index,
        }
    }
    /// Device bytes [`Self::new`] allocates: the context-sized QSA arenas,
    /// then the selection and wide-hidden carry, twice with their snapshot
    /// backups.
    fn device_bytes(config: &Qwen4Config, max_seq: usize) -> Option<usize> {
        let carry = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?
            .checked_add(std::mem::size_of::<i32>())?
            .checked_add(
                MTP_BRANCHES
                    .checked_mul(config.hidden_size)?
                    .checked_mul(std::mem::size_of::<f32>())?,
            )?;
        config
            .qsa_context_arena_bytes(max_seq, QsaKvFormat::F32)?
            .checked_add(carry.checked_mul(2)?)
    }
    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq: usize,
    ) -> Result<Self, MtpGpuError> {
        let full_width = config.num_key_value_heads * config.head_dim;
        let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
        let pooled_capacity = max_seq.div_ceil(config.indexer_compress_ratio);
        let selected_capacity = config.qsa_selected_capacity();
        let full_keys = gpu.zeros(&[max_seq * full_width], DType::F32)?;
        let full_values = match gpu.zeros(&[max_seq * full_width], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                return Err(error.into());
            }
        };
        let raw_index_keys = match gpu.zeros(&[max_seq * raw_width], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                return Err(error.into());
            }
        };
        let pooled_keys = match gpu.zeros(&[pooled_capacity * raw_width], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                let _ = gpu.free_tensor(raw_index_keys);
                return Err(error.into());
            }
        };
        let selected_bytes = selected_capacity * std::mem::size_of::<i32>();
        let selected_indices = match gpu.zeros(&[selected_bytes], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                let _ = gpu.free_tensor(raw_index_keys);
                let _ = gpu.free_tensor(pooled_keys);
                return Err(error.into());
            }
        };
        let selected_len_out = match gpu.zeros(&[std::mem::size_of::<i32>()], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                let _ = gpu.free_tensor(raw_index_keys);
                let _ = gpu.free_tensor(pooled_keys);
                let _ = gpu.free_tensor(selected_indices);
                return Err(error.into());
            }
        };
        let wide_hidden = match gpu.zeros(&[MTP_BRANCHES * config.hidden_size], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                let _ = gpu.free_tensor(raw_index_keys);
                let _ = gpu.free_tensor(pooled_keys);
                let _ = gpu.free_tensor(selected_indices);
                let _ = gpu.free_tensor(selected_len_out);
                return Err(error.into());
            }
        };
        let model_id = next_mtp_model_id();
        let snapshot_arena = match MtpGpuStateSnapshotArena::new(
            gpu,
            &selected_indices,
            &selected_len_out,
            &wide_hidden,
            model_id,
            MtpStateMark {
                full_len: 0,
                raw_len: 0,
                pooled_len: 0,
                selected_len: 0,
                position: 0,
                step_index: 0,
            },
        ) {
            Ok(arena) => arena,
            Err(error) => {
                let _ = gpu.free_tensor(full_keys);
                let _ = gpu.free_tensor(full_values);
                let _ = gpu.free_tensor(raw_index_keys);
                let _ = gpu.free_tensor(pooled_keys);
                let _ = gpu.free_tensor(selected_indices);
                let _ = gpu.free_tensor(selected_len_out);
                let _ = gpu.free_tensor(wide_hidden);
                return Err(error);
            }
        };
        Ok(Self {
            full_keys,
            full_values,
            raw_index_keys,
            pooled_keys,
            selected_indices,
            selected_len_out,
            wide_hidden,
            full_capacity: max_seq,
            raw_capacity: max_seq,
            pooled_capacity,
            selected_capacity,
            position: 0,
            full_len: 0,
            raw_len: 0,
            pooled_len: 0,
            selected_len: 0,
            step_index: 0,
            request_epoch: 0,
            generation: 0,
            model_id,
            snapshot_arena,
            prefix_arena: None,
        })
    }

    pub(crate) fn reset(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.snapshot_arena.invalidate();
        self.invalidate_prefix();
        for tensor in [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
            &self.selected_indices,
            &self.selected_len_out,
            &self.wide_hidden,
        ] {
            gpu.hip.memset(&tensor.buf, 0, tensor.buf.size())?;
        }
        self.position = 0;
        self.full_len = 0;
        self.raw_len = 0;
        self.pooled_len = 0;
        self.selected_len = 0;
        self.step_index = 0;
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }
    pub(crate) fn snapshot(&mut self, gpu: &mut Gpu) -> Result<MtpGpuStateSnapshot, MtpGpuError> {
        let dimensions = MtpGpuStateSnapshotArena::dimensions(self);
        let mark = self.mark();
        {
            let arena = &self.snapshot_arena;
            if arena.active {
                return Err(invalid("MTP snapshot arena is already active"));
            }
            if arena.model_id != self.model_id {
                return Err(invalid("MTP snapshot arena model mismatch"));
            }
            arena.validate_layout(self)?;
        }
        validate_mtp_mark(&mark, self)?;
        let arena_generation = self.snapshot_arena.generation;
        let result = (|| -> Result<(), MtpGpuError> {
            gpu.copy_d2d(
                &self.selected_indices,
                &self.snapshot_arena.selected_indices,
                self.selected_indices.byte_size(),
            )?;
            gpu.copy_d2d(
                &self.selected_len_out,
                &self.snapshot_arena.selected_len_out,
                self.selected_len_out.byte_size(),
            )?;
            gpu.copy_d2d(
                &self.wide_hidden,
                &self.snapshot_arena.wide_hidden,
                self.wide_hidden.byte_size(),
            )?;
            self.snapshot_arena.mark = mark;
            self.snapshot_arena.active = true;
            Ok(())
        })();
        if let Err(error) = result {
            self.snapshot_arena.invalidate();
            return Err(error);
        }
        Ok(MtpGpuStateSnapshot {
            model_id: self.model_id,
            request_epoch: self.request_epoch,
            state_generation: self.generation,
            arena_generation,
            dimensions,
        })
    }

    fn restore_impl(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
        consume: bool,
    ) -> Result<(), MtpGpuError> {
        {
            let arena = &self.snapshot_arena;
            arena.validate_ticket(self, &snapshot)?;
        }
        let mark = self.snapshot_arena.mark;
        let result = (|| -> Result<(), MtpGpuError> {
            gpu.copy_d2d(
                &self.snapshot_arena.selected_indices,
                &self.selected_indices,
                self.selected_indices.byte_size(),
            )?;
            gpu.copy_d2d(
                &self.snapshot_arena.selected_len_out,
                &self.selected_len_out,
                self.selected_len_out.byte_size(),
            )?;
            gpu.copy_d2d(
                &self.snapshot_arena.wide_hidden,
                &self.wide_hidden,
                self.wide_hidden.byte_size(),
            )?;
            self.full_len = mark.full_len;
            self.raw_len = mark.raw_len;
            self.pooled_len = mark.pooled_len;
            self.selected_len = mark.selected_len;
            self.position = mark.position;
            self.step_index = mark.step_index;
            if consume {
                self.snapshot_arena.invalidate();
            }
            Ok(())
        })();
        if result.is_err() && consume {
            self.snapshot_arena.invalidate();
        }
        self.retire_prefix_past_position();
        result
    }

    pub(crate) fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), MtpGpuError> {
        self.restore_impl(gpu, snapshot, true)
    }

    pub(crate) fn restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), MtpGpuError> {
        self.restore_impl(gpu, snapshot, false)
    }

    /// Keep the first `keep` tokens consumed since the ticket's mark,
    /// retaining the ticket. The K/V, raw and pooled key arenas are append
    /// storage (later rows are invisible until overwritten; a block is
    /// re-pooled when its last row is appended again), and the selection and
    /// own hidden are rebuilt by the next window's first step, so only the
    /// marks move. The selection length is capped at the kept position: a
    /// selection never covers more rows than are visible.
    pub(crate) fn truncate_retain(
        &mut self,
        snapshot: MtpGpuStateSnapshot,
        keep: usize,
        compress: usize,
    ) -> Result<(), MtpGpuError> {
        self.snapshot_arena.validate_ticket(self, &snapshot)?;
        let mark = self.snapshot_arena.mark;
        let position = mark.position + keep;
        if position > self.position {
            return Err(invalid(format!(
                "MTP truncate to {position} is past the consumed end {}",
                self.position
            )));
        }
        self.position = position;
        self.full_len = position;
        self.raw_len = position;
        self.pooled_len = position / compress;
        self.selected_len = self.selected_len.min(position);
        self.step_index = mark.step_index.wrapping_add(keep);
        self.retire_prefix_past_position();
        Ok(())
    }

    pub(crate) fn validate_commit(&self, snapshot: MtpGpuStateSnapshot) -> Result<(), MtpGpuError> {
        self.snapshot_arena.validate_ticket(self, &snapshot)
    }

    pub(crate) fn commit_validated(&mut self, _snapshot: MtpGpuStateSnapshot) {
        debug_assert!(
            self.snapshot_arena.active,
            "MTP commit_validated requires an active snapshot"
        );
        self.snapshot_arena.invalidate();
    }
    #[cfg(test)]
    fn commit(&mut self, snapshot: MtpGpuStateSnapshot) -> Result<(), MtpGpuError> {
        self.validate_commit(snapshot)?;
        self.commit_validated(snapshot);
        Ok(())
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn parity_metadata(&self) -> MtpStateParityMetadata {
        MtpStateParityMetadata::from_state(self)
    }

    pub(crate) fn parity_buffers(&self) -> MtpStateParityBuffers<'_> {
        MtpStateParityBuffers {
            full_keys: &self.full_keys,
            full_values: &self.full_values,
            raw_index_keys: &self.raw_index_keys,
            pooled_keys: &self.pooled_keys,
            selected_indices: &self.selected_indices,
            selected_len_out: &self.selected_len_out,
            wide_hidden: &self.wide_hidden,
        }
    }

    pub(crate) fn parity_set_metadata(&mut self, metadata: MtpStateParityMetadata) {
        self.full_len = metadata.full_len;
        self.raw_len = metadata.raw_len;
        self.pooled_len = metadata.pooled_len;
        self.selected_len = metadata.selected_len;
        self.position = metadata.position;
        self.step_index = metadata.step_index;
    }

    pub(crate) fn attach_prefix_arena(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        if self.prefix_arena.is_none() {
            self.prefix_arena = Some(MtpGpuStateSnapshotArena::new(
                gpu,
                &self.selected_indices,
                &self.selected_len_out,
                &self.wide_hidden,
                self.model_id,
                self.mark(),
            )?);
        }
        Ok(())
    }

    pub(crate) fn prefix_position(&self) -> Option<usize> {
        self.prefix_arena
            .as_ref()
            .filter(|arena| arena.active)
            .map(|arena| arena.mark.position)
    }

    pub(crate) fn invalidate_prefix(&mut self) {
        if let Some(arena) = self.prefix_arena.as_mut() {
            arena.invalidate();
        }
    }

    fn retire_prefix_past_position(&mut self) {
        if self.prefix_position().is_some_and(|saved| self.position < saved) {
            self.invalidate_prefix();
        }
    }

    /// Copy the in-place-overwritten MTP owners (selection, device selected
    /// length, own wide hidden) and the marks incl. `step_index`.
    pub(crate) fn capture_prefix(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.invalidate_prefix();
        let mark = self.mark();
        validate_mtp_mark(&mark, self)?;
        let arena = self
            .prefix_arena
            .as_ref()
            .ok_or_else(|| invalid("MTP prefix arena is not attached"))?;
        if arena.model_id != self.model_id {
            return Err(invalid("MTP prefix arena model mismatch"));
        }
        arena.validate_layout(self)?;
        for (live, saved) in [
            (&self.selected_indices, &arena.selected_indices),
            (&self.selected_len_out, &arena.selected_len_out),
            (&self.wide_hidden, &arena.wide_hidden),
        ] {
            gpu.copy_d2d(live, saved, live.byte_size())?;
        }
        let arena = self
            .prefix_arena
            .as_mut()
            .ok_or_else(|| invalid("MTP prefix arena is not attached"))?;
        arena.mark = mark;
        arena.active = true;
        Ok(())
    }

    /// Restore the durable checkpoint. Like a reset it starts a new request
    /// epoch and retires every speculative ticket; the caller resets on error.
    pub(crate) fn restore_prefix(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        let arena = self
            .prefix_arena
            .as_ref()
            .filter(|arena| arena.active)
            .ok_or_else(|| invalid("MTP prefix checkpoint is not valid"))?;
        if arena.model_id != self.model_id {
            return Err(invalid("MTP prefix arena model mismatch"));
        }
        validate_mtp_mark(&arena.mark, self)?;
        arena.validate_layout(self)?;
        let mark = arena.mark;
        self.snapshot_arena.invalidate();
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.generation = self.generation.wrapping_add(1);
        let arena = self
            .prefix_arena
            .as_ref()
            .ok_or_else(|| invalid("MTP prefix arena is not attached"))?;
        for (live, saved) in [
            (&self.selected_indices, &arena.selected_indices),
            (&self.selected_len_out, &arena.selected_len_out),
            (&self.wide_hidden, &arena.wide_hidden),
        ] {
            gpu.copy_d2d(saved, live, live.byte_size())?;
        }
        self.full_len = mark.full_len;
        self.raw_len = mark.raw_len;
        self.pooled_len = mark.pooled_len;
        self.selected_len = mark.selected_len;
        self.position = mark.position;
        self.step_index = mark.step_index;
        Ok(())
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let Self {
            full_keys,
            full_values,
            raw_index_keys,
            pooled_keys,
            selected_indices,
            selected_len_out,
            wide_hidden,
            snapshot_arena,
            prefix_arena,
            ..
        } = self;
        let mut first = snapshot_arena.free_gpu(gpu);
        if let Some(error) = prefix_arena.and_then(|arena| arena.free_gpu(gpu)) {
            first.get_or_insert(error);
        }
        for tensor in [
            full_keys,
            full_values,
            raw_index_keys,
            pooled_keys,
            selected_indices,
            selected_len_out,
            wide_hidden,
        ] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// Model-owned native MTP operator and state.
pub struct Qwen4MtpGpu {
    pub(crate) scratch: MtpGpuScratch,
    /// [`MTP_APPEND_ROWS`]-row buffers of the batched prompt append.
    append_scratch: MtpGpuScratch,
    pub(crate) state: MtpGpuState,
    pub(crate) moe: Qwen4MoeLayerRuntime,
    pub(crate) max_seq: usize,
    /// Draft ranking (`HIPFIRE_MTP_DRAFT_HEAD=mq2..mq6[r]`, default `mq2r`;
    /// anything else = the model's own head). On the shipped Q8_0 head an
    /// MQ2 copy with exact re-scoring of its top 8 drafts the Q8_0 head's own
    /// argmax at a quarter of its read; plain MQ3 loses acceptance.
    pub(crate) draft: DraftHead,
    /// Draft policy at the durable prefix checkpoint.
    prefix_policy: DraftHeadRequestState,
}

/// Draft-head front: tokens below this id, plus EOS and the control ids
/// after it, covered every non-control token of the committed English/code
/// prompt set except rare multilingual ones and cut the draft head read 60%.
const DRAFT_FRONT: usize = 100_000;
/// Draft steps that rank the whole vocabulary after an input token outside
/// the front (non-English text keeps the full head).
const DRAFT_FULL_HOLD: u32 = 64;

/// What one MTP head step must produce.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MtpStep {
    /// The full step and its argmax draft token.
    Predict,
    /// The full step: its head state feeds a following step without hidden.
    Advance,
    /// Only this position's K/V and index-key cache rows. For the last step
    /// of a chain: the next step brings its own backbone hidden and reselects
    /// (the head state and selection are left stale; a reuse is refused).
    Append,
}

/// Draft ranking policy (`HIPFIRE_MTP_DRAFT_HEAD`, default `mq2r`) and row
/// layout of the MTP draft head.
fn draft_head_config(config: &Qwen4Config) -> (DraftHeadPolicy, DraftHeadLayout) {
    let policy = DraftHeadPolicy::parse(
        &hipfire_config::developer_var("HIPFIRE_MTP_DRAFT_HEAD")
            .unwrap_or_else(|_| "mq2r".to_string()),
    );
    let layout = DraftHeadLayout {
        vocab: config.vocab_size,
        hidden: config.hidden_size,
        front: DRAFT_FRONT,
        special: config.eos_token_id as usize,
        full_hold: DRAFT_FULL_HOLD,
    };
    (policy, layout)
}

impl Qwen4MtpGpu {
    /// `(resident, load scratch)` device bytes [`Self::new`] takes at
    /// `max_seq` for a language head stored as `head_dtype`: what the
    /// attached head keeps, and the draft head's build scratch on top of it
    /// (`DraftHead::device_bytes`), released before `new` returns.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        max_seq: usize,
        head_dtype: DType,
    ) -> Option<(usize, usize)> {
        let (policy, layout) = draft_head_config(config);
        let (draft, scratch) = DraftHead::device_bytes(head_dtype, layout, policy)?;
        let resident = MtpGpuScratch::device_bytes(config, 1)
            .ok()?
            .checked_add(MtpGpuScratch::device_bytes(config, MTP_APPEND_ROWS).ok()?)?
            .checked_add(MtpGpuState::device_bytes(config, max_seq)?)?
            .checked_add(Qwen4MoeLayerRuntime::device_bytes(config)?)?
            .checked_add(draft)?;
        Some((resident, scratch))
    }

    pub(crate) fn new(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        max_seq: usize,
    ) -> Result<Self, MtpGpuError> {
        config.mtp.validate().map_err(MtpGpuError::Invalid)?;
        if max_seq == 0 || max_seq > config.max_position_embeddings {
            return Err(invalid("MTP max_seq is outside model capacity"));
        }
        let scratch = MtpGpuScratch::new(gpu, config, 1)?;
        let append_scratch = match MtpGpuScratch::new(gpu, config, MTP_APPEND_ROWS) {
            Ok(append_scratch) => append_scratch,
            Err(error) => {
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        let state = match MtpGpuState::new(gpu, config, max_seq) {
            Ok(state) => state,
            Err(error) => {
                let _ = append_scratch.free_gpu(gpu);
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        let moe = match Qwen4MoeLayerRuntime::from_moe(gpu, weights, &weights.mtp.moe, 0, config) {
            Ok(moe) => moe,
            Err(error) => {
                let _ = state.free_gpu(gpu);
                let _ = append_scratch.free_gpu(gpu);
                let _ = scratch.free_gpu(gpu);
                return Err(error.into());
            }
        };
        let (policy, layout) = draft_head_config(config);
        let draft = weights
            .resident(&weights.root.lm_head)
            .map_err(MtpGpuError::from)
            .and_then(|head| DraftHead::new(gpu, head, layout, policy).map_err(MtpGpuError::from));
        let draft = match draft {
            Ok(draft) => draft,
            Err(error) => {
                let _ = moe.free_gpu(gpu);
                let _ = state.free_gpu(gpu);
                let _ = append_scratch.free_gpu(gpu);
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        Ok(Self {
            scratch,
            append_scratch,
            state,
            moe,
            max_seq,
            draft,
            prefix_policy: DraftHeadRequestState::default(),
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        let Self {
            scratch,
            append_scratch,
            state,
            moe,
            draft,
            ..
        } = self;
        let draft_error = draft.free_gpu(gpu);
        let scratch_error = scratch.free_gpu(gpu);
        let append_scratch_error = append_scratch.free_gpu(gpu);
        let state_error = state.free_gpu(gpu);
        let moe_error = moe.free_gpu(gpu);
        draft_error
            .or(scratch_error)
            .or(append_scratch_error)
            .or(state_error)
            .or(moe_error)
            .map_or(Ok(()), |error| Err(MtpGpuError::Hip(error)))
    }
    /// Deliberately dirty the reusable MoE buffers for the bounded profile
    /// regression.  The production caller still owns initialization: this
    /// helper only makes stale scratch/output reads observable in that mode.
    pub(crate) fn dirty_moe_reuse(&self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        for tensor in [
            &self.scratch.router_logits,
            &self.scratch.moe_x_rot,
            &self.scratch.moe_gate_up,
            &self.scratch.moe_gate,
            &self.scratch.moe_up,
            &self.scratch.moe_hidden,
            &self.scratch.moe_output,
            &self.scratch.moe_gate_batch,
            &self.scratch.moe_up_batch,
            &self.scratch.moe_rot_batch,
            &self.scratch.moe_topk_indices,
            &self.scratch.moe_topk_weights,
            &self.scratch.moe_down_expanded,
            &self.scratch.moe_scalar,
        ] {
            gpu.hip.memset(&tensor.buf, 0x3f, tensor.buf.size())?;
        }
        Ok(())
    }

    /// A cold request also starts the draft policy fresh: the full-vocabulary
    /// hold and margin are request-local, never inherited from the previous
    /// request.
    pub(crate) fn reset(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.draft.reset_request_state();
        self.state.reset(gpu)
    }

    pub(crate) fn attach_prefix_arena(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.state.attach_prefix_arena(gpu)
    }

    pub(crate) fn prefix_position(&self) -> Option<usize> {
        self.state.prefix_position()
    }

    pub(crate) fn invalidate_prefix(&mut self) {
        self.state.invalidate_prefix();
    }

    /// Checkpoint the head state and the draft policy. Prompt appends only
    /// `observe` (a qualifying input sets the full hold); drafts are the only
    /// decrement, so at a prompt boundary this is the canonical prompt summary.
    pub(crate) fn capture_prefix(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.state.capture_prefix(gpu)?;
        self.prefix_policy = self.draft.request_state();
        Ok(())
    }

    pub(crate) fn restore_prefix(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.state.restore_prefix(gpu)?;
        self.draft.set_request_state(self.prefix_policy);
        Ok(())
    }
    pub(crate) fn snapshot(&mut self, gpu: &mut Gpu) -> Result<MtpGpuStateSnapshot, MtpGpuError> {
        self.state.snapshot(gpu)
    }

    pub(crate) fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), MtpGpuError> {
        self.state.restore(gpu, snapshot)
    }
    pub(crate) fn restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), MtpGpuError> {
        self.state.restore_retain(gpu, snapshot)
    }

    pub(crate) fn truncate_retain(
        &mut self,
        snapshot: MtpGpuStateSnapshot,
        keep: usize,
        compress: usize,
    ) -> Result<(), MtpGpuError> {
        self.state.truncate_retain(snapshot, keep, compress)
    }

    pub(crate) fn validate_commit(&self, snapshot: MtpGpuStateSnapshot) -> Result<(), MtpGpuError> {
        self.state.validate_commit(snapshot)
    }

    pub(crate) fn commit_validated(&mut self, snapshot: MtpGpuStateSnapshot) {
        self.state.commit_validated(snapshot);
    }

    pub(crate) fn position(&self) -> usize {
        self.state.position()
    }

    pub(crate) fn parity_state(&self) -> &MtpGpuState {
        &self.state
    }

    pub(crate) fn forward_token_with_logits(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        token: u32,
        position: usize,
        fresh_qsa_selection: bool,
        logits: &GpuTensor,
    ) -> Result<u32, MtpGpuError> {
        if logits.dtype != DType::F32 || logits.numel() != self.draft.logits().numel() {
            return Err(invalid("MTP logits destination shape mismatch"));
        }
        let next = self
            .forward_token(
                gpu,
                weights,
                config,
                token,
                None,
                position,
                fresh_qsa_selection,
                MtpStep::Predict,
            )?
            .ok_or_else(|| invalid("MTP prediction requested but no token produced"))?;
        gpu.copy_d2d(self.draft.logits(), logits, logits.byte_size())?;
        Ok(next)
    }

    pub(crate) fn forward_token(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
        step: MtpStep,
    ) -> Result<Option<u32>, MtpGpuError> {
        let wide = MTP_BRANCHES
            .checked_mul(config.hidden_size)
            .ok_or_else(|| invalid("MTP wide dimension overflow"))?;
        if let Some(hidden) = backbone_hidden {
            if hidden.dtype != DType::F32 || hidden.numel() != wide {
                return Err(invalid(format!(
                    "MTP backbone hidden must be F32 with {wide} elements"
                )));
            }
        } else {
            gpu.copy_d2d(
                &self.state.wide_hidden,
                &self.scratch.backbone_hidden,
                self.state.wide_hidden.byte_size(),
            )?;
        }
        if position != self.state.position {
            return Err(invalid(format!(
                "MTP position mismatch: expected {}, got {position}",
                self.state.position
            )));
        }
        if position >= self.max_seq {
            return Err(invalid("MTP position exceeds QSA capacity"));
        }
        self.draft.observe(token);
        self.scratch
            .host_token_bytes
            .copy_from_slice(&(token as i32).to_ne_bytes());
        gpu.memcpy_htod_auto(&self.scratch.token_ids.buf, &self.scratch.host_token_bytes)?;

        let (full_len, raw_len, pooled_len, selected_len, next_position) = {
            let scratch = &self.scratch;
            let state = &self.state;
            let backbone_hidden = backbone_hidden.unwrap_or(&scratch.backbone_hidden);
            let hidden = config.hidden_size;
            let ctx = DispatchCtx::new(gpu);
            let attn = hyper_desc(weights, &weights.mtp.attn_hyper)?;
            let mlp = hyper_desc(weights, &weights.mtp.mlp_hyper)?;
            let final_read = hyper_read_desc(weights, &weights.mtp.final_hyper)?;
            let qsa = qsa_desc(weights, &weights.mtp.attention)?;
            let mode = match (step, fresh_qsa_selection) {
                (MtpStep::Append, _) => IndexedAttentionMode::AppendOnly,
                (_, true) => IndexedAttentionMode::Full,
                (_, false) => IndexedAttentionMode::ReuseSelection {
                    selected_len_out: &state.selected_len_out,
                },
            };
            let attention = IndexedAttentionOp {
                indexer_qk: qsa.indexer_qk,
                indexer_q_norm: qsa.indexer_q_norm,
                indexer_k_norm: qsa.indexer_k_norm,
                q: qsa.q,
                k: qsa.k,
                v: qsa.v,
                q_norm: qsa.q_norm,
                k_norm: qsa.k_norm,
                output: qsa.output,
                state: IndexedAttentionState {
                    // The one MTP layer keeps the exact F32 QSA state.
                    format: QsaKvFormat::F32,
                    full_keys: &state.full_keys,
                    full_values: &state.full_values,
                    raw_index_keys: &state.raw_index_keys,
                    pooled_keys: &state.pooled_keys,
                    selected_indices: &state.selected_indices,
                    full_capacity: state.full_capacity,
                    raw_capacity: state.raw_capacity,
                    pooled_capacity: state.pooled_capacity,
                    selected_capacity: state.selected_capacity,
                    position_capacity: state.full_capacity,
                    full_len: state.full_len,
                    raw_len: state.raw_len,
                    pooled_len: state.pooled_len,
                    selected_len: state.selected_len,
                    position: state.position,
                },
                input: &scratch.hc_mixed,
                index_scratch: &scratch.index,
                qgate_scratch: &scratch.q_and_gate,
                k_scratch: &scratch.qsa_k,
                v_scratch: &scratch.qsa_v,
                qsa_output: &scratch.qsa_output,
                selected_scratch: &scratch.qsa_selected,
                attention_output: &scratch.projected_embedding,
                bf16_scratch: &scratch.hc_bf16,
                rows: 1,
                index_heads: config.indexer_n_heads,
                index_kv_heads: config.indexer_kv_heads,
                index_dim: config.indexer_head_dim,
                budget: config.indexer_budget,
                compress: config.indexer_compress_ratio,
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                input_width: hidden,
                rotation: &scratch.rotation,
                mode,
                trunk_a4: 0,
            };
            let lengths = attention.next_lengths()?;
            let moe = if step == MtpStep::Append {
                None
            } else {
                Some(seal_moe_decode(
                    &ctx,
                    config,
                    0,
                    &self.moe,
                    &scratch.hc_mixed,
                    &scratch.moe_output,
                    Qwen4MoeScratch {
                        router_logits: &scratch.router_logits,
                        scalar_buf: &scratch.moe_scalar,
                        x_rot_local: &scratch.moe_x_rot,
                        gate_up_buf: &scratch.moe_gate_up,
                        gate_buf: &scratch.moe_gate,
                        up_buf: &scratch.moe_up,
                        ffn_hidden: &scratch.moe_hidden,
                        // The shared down projection must not overwrite the
                        // routed accumulator.
                        ffn_out: &scratch.projected_embedding,
                        gate_batch: &scratch.moe_gate_batch,
                        up_batch: &scratch.moe_up_batch,
                        rot_batch: &scratch.moe_rot_batch,
                        topk_indices: &scratch.moe_topk_indices,
                        topk_weights: &scratch.moe_topk_weights,
                        down_expanded: &scratch.moe_down_expanded,
                    },
                )?)
            };
            let mut steps: SmallVec<[Step<'_>; 16]> = SmallVec::new();
            steps.push(Step::Embed(EmbeddingOp {
                table: weights.resident(&weights.root.embedding)?,
                rotated: &scratch.embedding_rot,
                token_ids: &scratch.token_ids,
                output: &scratch.token_embedding,
                rows: 1,
                dim: hidden,
            }));
            steps.push(Step::HyperNorm(HyperNormOp {
                input: &scratch.token_embedding,
                norm_weight: weights.resident(&weights.mtp.pre_fc_norm_embedding)?,
                normalized: &scratch.embedding_norm,
                branches: 1,
                hidden,
                state_bf16: false,
            }));
            steps.push(Step::Project(ProjectOp {
                weight: dense_ref(weights, &weights.mtp.fc_embedding)?,
                input: &scratch.embedding_norm,
                output: &scratch.projected_embedding,
                rows: 1,
                rotation: Some(&scratch.rotation),
            }));
            steps.push(Step::HyperNorm(HyperNormOp {
                input: backbone_hidden,
                norm_weight: weights.resident(&weights.mtp.pre_fc_norm_hidden)?,
                normalized: &scratch.hidden_norm,
                branches: MTP_BRANCHES,
                hidden,
                state_bf16: false,
            }));
            // Both branch rows in one projection (each bitwise its own).
            steps.push(Step::Project(ProjectOp {
                weight: dense_ref(weights, &weights.mtp.fc_hidden)?,
                input: &scratch.hidden_norm,
                output: &scratch.projected_hidden,
                rows: MTP_BRANCHES,
                rotation: Some(&scratch.rotation),
            }));
            steps.push(Step::BroadcastAdd(BroadcastAddOp {
                rows_input: &scratch.projected_hidden,
                row: &scratch.projected_embedding,
                output: &scratch.wide,
                rows: MTP_BRANCHES,
                width: hidden,
                group: MTP_BRANCHES,
            }));
            steps.push(Step::HyperRead(hc_read(&attn.read, scratch, config, 1)));
            steps.push(Step::IndexedAttention(attention));
            if let Some(moe) = moe {
                steps.push(Step::HyperWrite(hc_write(
                    &attn.write,
                    &scratch.projected_embedding,
                    scratch,
                    config,
                )));
                steps.push(Step::HyperRead(hc_read(&mlp.read, scratch, config, 1)));
                steps.push(Step::Clear(ClearOp {
                    tensor: &scratch.moe_output,
                    elements: hidden,
                }));
                steps.push(Step::Moe(moe));
                steps.push(Step::HyperWrite(hc_write(
                    &mlp.write,
                    &scratch.moe_output,
                    scratch,
                    config,
                )));
                if step == MtpStep::Predict {
                    steps.push(Step::HyperRead(hc_read(&final_read, scratch, config, 1)));
                }
            }
            validate_steps(gpu, &steps)?;
            execute_validated_steps(gpu, &ctx, &steps)?;
            lengths
        };

        let next_token = if step == MtpStep::Predict {
            let head = weights.resident(&weights.root.lm_head)?;
            Some(self.draft.draft(gpu, head, &self.scratch.hc_mixed)?)
        } else {
            None
        };
        if step != MtpStep::Append {
            gpu.copy_d2d(
                &self.scratch.wide,
                &self.state.wide_hidden,
                self.scratch.wide.byte_size(),
            )?;
        }
        let state = &mut self.state;
        state.position = next_position;
        state.full_len = full_len;
        state.raw_len = raw_len;
        state.pooled_len = pooled_len;
        state.selected_len = selected_len;
        state.step_index = state.step_index.wrapping_add(1);
        Ok(next_token)
    }

    /// [`MtpStep::Append`] for up to [`MTP_APPEND_ROWS`] consecutive prompt
    /// tokens in one step program: `backbone_hidden` holds their wide target
    /// hidden rows (`tokens.len() * hc_count * hidden` F32).  Appends the same
    /// K/V and index-key cache rows as one append per token, through the
    /// multi-row projection routes (draft-side numerics only: the target
    /// verifies every draft).
    pub(crate) fn append_rows(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        tokens: &[u32],
        backbone_hidden: &GpuTensor,
        position: usize,
    ) -> Result<(), MtpGpuError> {
        let rows = tokens.len();
        let hidden = config.hidden_size;
        let wide = MTP_BRANCHES * hidden;
        if rows == 0 || rows > MTP_APPEND_ROWS {
            return Err(invalid(format!(
                "MTP append of {rows} rows is outside 1..={MTP_APPEND_ROWS}"
            )));
        }
        if backbone_hidden.dtype != DType::F32 || backbone_hidden.numel() != rows * wide {
            return Err(invalid(format!(
                "MTP append hidden must be F32 with {} elements",
                rows * wide
            )));
        }
        if position != self.state.position {
            return Err(invalid(format!(
                "MTP position mismatch: expected {}, got {position}",
                self.state.position
            )));
        }
        if position + rows > self.max_seq {
            return Err(invalid("MTP append exceeds QSA capacity"));
        }
        let mut token_bytes = Vec::with_capacity(rows * 4);
        for &token in tokens {
            self.draft.observe(token);
            token_bytes.extend_from_slice(&(token as i32).to_ne_bytes());
        }
        let scratch = &self.append_scratch;
        gpu.memcpy_htod_auto(&scratch.token_ids.buf, &token_bytes)?;

        let (full_len, raw_len, pooled_len, selected_len, next_position) = {
            let state = &self.state;
            let ctx = DispatchCtx::new(gpu);
            let attn = hyper_desc(weights, &weights.mtp.attn_hyper)?;
            let qsa = qsa_desc(weights, &weights.mtp.attention)?;
            // HyperNorm takes its row count from the tensor length.
            let token_embedding = scratch.token_embedding.sub_offset(0, rows * hidden);
            let embedding_norm = scratch.embedding_norm.sub_offset(0, rows * hidden);
            let hidden_norm = scratch.hidden_norm.sub_offset(0, rows * wide);
            let attention = IndexedAttentionOp {
                indexer_qk: qsa.indexer_qk,
                indexer_q_norm: qsa.indexer_q_norm,
                indexer_k_norm: qsa.indexer_k_norm,
                q: qsa.q,
                k: qsa.k,
                v: qsa.v,
                q_norm: qsa.q_norm,
                k_norm: qsa.k_norm,
                output: qsa.output,
                state: IndexedAttentionState {
                    format: QsaKvFormat::F32,
                    full_keys: &state.full_keys,
                    full_values: &state.full_values,
                    raw_index_keys: &state.raw_index_keys,
                    pooled_keys: &state.pooled_keys,
                    selected_indices: &state.selected_indices,
                    full_capacity: state.full_capacity,
                    raw_capacity: state.raw_capacity,
                    pooled_capacity: state.pooled_capacity,
                    selected_capacity: state.selected_capacity,
                    position_capacity: state.full_capacity,
                    full_len: state.full_len,
                    raw_len: state.raw_len,
                    pooled_len: state.pooled_len,
                    selected_len: state.selected_len,
                    position: state.position,
                },
                input: &scratch.hc_mixed,
                index_scratch: &scratch.index,
                qgate_scratch: &scratch.q_and_gate,
                k_scratch: &scratch.qsa_k,
                v_scratch: &scratch.qsa_v,
                qsa_output: &scratch.qsa_output,
                selected_scratch: &scratch.qsa_selected,
                attention_output: &scratch.projected_embedding,
                bf16_scratch: &scratch.hc_bf16,
                rows,
                index_heads: config.indexer_n_heads,
                index_kv_heads: config.indexer_kv_heads,
                index_dim: config.indexer_head_dim,
                budget: config.indexer_budget,
                compress: config.indexer_compress_ratio,
                heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                input_width: hidden,
                rotation: &scratch.rotation,
                mode: IndexedAttentionMode::AppendOnly,
                trunk_a4: 0,
            };
            let lengths = attention.next_lengths()?;
            let steps = [
                Step::Embed(EmbeddingOp {
                    table: weights.resident(&weights.root.embedding)?,
                    rotated: &scratch.embedding_rot,
                    token_ids: &scratch.token_ids,
                    output: &token_embedding,
                    rows,
                    dim: hidden,
                }),
                Step::HyperNorm(HyperNormOp {
                    input: &token_embedding,
                    norm_weight: weights.resident(&weights.mtp.pre_fc_norm_embedding)?,
                    normalized: &embedding_norm,
                    branches: 1,
                    hidden,
                    state_bf16: false,
                }),
                Step::Project(ProjectOp {
                    weight: dense_ref(weights, &weights.mtp.fc_embedding)?,
                    input: &embedding_norm,
                    output: &scratch.projected_embedding,
                    rows,
                    rotation: Some(&scratch.rotation),
                }),
                Step::HyperNorm(HyperNormOp {
                    input: backbone_hidden,
                    norm_weight: weights.resident(&weights.mtp.pre_fc_norm_hidden)?,
                    normalized: &hidden_norm,
                    branches: MTP_BRANCHES,
                    hidden,
                    state_bf16: false,
                }),
                Step::Project(ProjectOp {
                    weight: dense_ref(weights, &weights.mtp.fc_hidden)?,
                    input: &hidden_norm,
                    output: &scratch.projected_hidden,
                    rows: rows * MTP_BRANCHES,
                    rotation: Some(&scratch.rotation),
                }),
                Step::BroadcastAdd(BroadcastAddOp {
                    rows_input: &scratch.projected_hidden,
                    row: &scratch.projected_embedding,
                    output: &scratch.wide,
                    rows: rows * MTP_BRANCHES,
                    width: hidden,
                    group: MTP_BRANCHES,
                }),
                Step::HyperRead(hc_read(&attn.read, scratch, config, rows)),
                Step::IndexedAttention(attention),
            ];
            validate_steps(gpu, &steps)?;
            execute_validated_steps(gpu, &ctx, &steps)?;
            lengths
        };
        let state = &mut self.state;
        state.position = next_position;
        state.full_len = full_len;
        state.raw_len = raw_len;
        state.pooled_len = pooled_len;
        state.selected_len = selected_len;
        state.step_index = state.step_index.wrapping_add(rows);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::state::{Qwen4State, StateError};

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok().or_else(|| {
            eprintln!("skip: Qwen4 MTP arena tests require a GPU");
            None
        })
    }

    fn new_compact_mtp_state(gpu: &mut Gpu) -> MtpGpuState {
        let config = compact_test_config();
        MtpGpuState::new(gpu, &config, 8).expect("compact Qwen4 MTP state")
    }

    fn invalid_contains<T>(result: Result<T, MtpGpuError>, needle: &str) -> bool {
        matches!(result, Err(MtpGpuError::Invalid(message)) if message.contains(needle))
    }

    #[test]
    fn mtp_snapshot_arena_enforces_ticket_lifecycle_before_copy() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let mut state = new_compact_mtp_state(&mut gpu);
        state.position = 2;

        let ticket = state.snapshot(&mut gpu).expect("first snapshot");
        assert!(invalid_contains(state.snapshot(&mut gpu), "already active"));

        let mut stale_model = ticket;
        stale_model.model_id = stale_model.model_id.wrapping_add(1);
        assert!(invalid_contains(
            state.restore(&mut gpu, stale_model),
            "does not belong"
        ));
        assert_eq!(state.position, 2);
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("request snapshot");
        state.request_epoch = state.request_epoch.wrapping_add(1);
        assert!(invalid_contains(
            state.restore(&mut gpu, ticket),
            "does not belong"
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("generation snapshot");
        state.generation = state.generation.wrapping_add(1);
        assert!(invalid_contains(
            state.restore(&mut gpu, ticket),
            "does not belong"
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("arena generation snapshot");
        state.snapshot_arena.generation = state.snapshot_arena.generation.wrapping_add(1);
        assert!(invalid_contains(
            state.restore(&mut gpu, ticket),
            "does not belong"
        ));
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("shape snapshot");
        let mut stale_shape = ticket;
        stale_shape.dimensions.shape_hash ^= 1;
        assert!(invalid_contains(
            state.restore(&mut gpu, stale_shape),
            "does not belong"
        ));
        assert_eq!(state.position, 2);
        state.snapshot_arena.invalidate();

        let ticket = state.snapshot(&mut gpu).expect("restore-consume snapshot");
        state.restore(&mut gpu, ticket).expect("restore");
        assert!(invalid_contains(
            state.restore(&mut gpu, ticket),
            "inactive or consumed"
        ));

        let ticket = state.snapshot(&mut gpu).expect("commit snapshot");
        state.commit(ticket).expect("commit");
        assert!(invalid_contains(
            state.commit(ticket),
            "inactive or consumed"
        ));

        let ticket = state.snapshot(&mut gpu).expect("reset snapshot");
        state.reset(&mut gpu).expect("reset");
        assert!(invalid_contains(
            state.restore(&mut gpu, ticket),
            "inactive or consumed"
        ));
        if let Some(error) = state.free_gpu(&mut gpu) {
            panic!("free MTP state: {error:?}");
        }
    }

    #[test]
    fn dual_owner_rollback_restores_after_injected_replay_error() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let config = compact_test_config();
        let mut target = Qwen4State::new(&mut gpu, &config, 8, crate::state::Qwen4StateFormat::F32)
            .expect("compact target state");
        let mut mtp = MtpGpuState::new(&mut gpu, &config, 8).expect("compact MTP state");
        let target_recurrent_size = target.gdn[0].recurrent.byte_size();
        let mut target_bytes = vec![0u8; target_recurrent_size];
        target_bytes[..4].copy_from_slice(&1.25f32.to_ne_bytes());
        gpu.hip
            .memcpy_htod(&target.gdn[0].recurrent.buf, &target_bytes)
            .expect("upload target state");
        let selected_size = mtp.selected_indices.byte_size();
        let mut selected_bytes = vec![0u8; selected_size];
        selected_bytes[..4].copy_from_slice(&3i32.to_ne_bytes());
        selected_bytes[4..8].copy_from_slice(&5i32.to_ne_bytes());
        gpu.hip
            .memcpy_htod(&mtp.selected_indices.buf, &selected_bytes)
            .expect("upload MTP selection");
        gpu.hip
            .memcpy_htod(&mtp.selected_len_out.buf, &2i32.to_ne_bytes())
            .expect("upload MTP selection length");
        target.position = 1;
        mtp.position = 1;
        mtp.selected_len = 2;
        mtp.step_index = 1;

        let target_ticket = target.snapshot(&mut gpu).expect("target snapshot");
        let mtp_ticket = mtp.snapshot(&mut gpu).expect("MTP snapshot");

        gpu.hip
            .memset(
                &target.gdn[0].recurrent.buf,
                0,
                target.gdn[0].recurrent.buf.size(),
            )
            .expect("mutate target state");
        gpu.hip
            .memset(
                &mtp.selected_indices.buf,
                0,
                mtp.selected_indices.buf.size(),
            )
            .expect("mutate MTP selection");
        gpu.hip
            .memset(
                &mtp.selected_len_out.buf,
                0,
                mtp.selected_len_out.buf.size(),
            )
            .expect("mutate MTP selection length");
        target.position = 6;
        mtp.position = 6;
        mtp.selected_len = 0;
        mtp.step_index = 0;

        let replay: Result<(), &str> = Err("injected replay error");
        if let Err(error) = replay {
            target.restore(&mut gpu, target_ticket).expect(error);
            mtp.restore(&mut gpu, mtp_ticket).expect(error);
        }
        assert!(replay.is_err());
        assert_eq!(target.position, 1);
        assert_eq!(mtp.position, 1);
        assert_eq!(mtp.selected_len, 2);
        assert_eq!(mtp.step_index, 1);

        let mut restored_target = [0u8; 4];
        gpu.hip
            .memcpy_dtoh(&mut restored_target, &target.gdn[0].recurrent.buf)
            .expect("download target state");
        assert_eq!(restored_target, 1.25f32.to_ne_bytes());
        let mut restored_selection = [0u8; 8];
        gpu.hip
            .memcpy_dtoh(&mut restored_selection, &mtp.selected_indices.buf)
            .expect("download MTP selection");
        assert_eq!(&restored_selection[..4], &3i32.to_ne_bytes());
        assert_eq!(&restored_selection[4..8], &5i32.to_ne_bytes());
        let mut restored_len = [0u8; 4];
        gpu.hip
            .memcpy_dtoh(&mut restored_len, &mtp.selected_len_out.buf)
            .expect("download MTP selection length");
        assert_eq!(restored_len, 2i32.to_ne_bytes());
        target.free_gpu(&mut gpu).expect("free target state");
        if let Some(error) = mtp.free_gpu(&mut gpu) {
            panic!("free MTP state: {error:?}");
        }
    }
}

// Keep the grammar import in this module's public dependency closure while the
// full acceptance adapter is assembled in `mtp_spec.rs`.
fn _grammar_marker(_: Option<&mut dyn SpecGrammar>) {}
