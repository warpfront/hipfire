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
    execute_moe, Qwen4GpuForwardError, Qwen4MoeLayerRuntime, Qwen4MoeScratch,
};
use crate::projection::{dispatch_embedding, dispatch_gemv};
use crate::weights::{HyperConnectionWeights, Qwen4Weights, TensorRef, WeightError};
use hipfire_runtime::spec::SpecGrammar;
use rdna_compute::tensor_ops::{
    argmax_f32, hc_activation_fused_f32, hyper_norm, hyper_read_projected, hyper_write,
    indexed_attention_attention, indexed_attention_cache_append, indexed_attention_norm_rope,
    indexed_attention_pool_rope, indexed_attention_reuse_selection, indexed_attention_select,
    ArgmaxF32, HcActivationFused, HyperNorm, HyperReadProjected, HyperWrite,
    IndexedAttentionAttention, IndexedAttentionCacheAppend, IndexedAttentionNormRope,
    IndexedAttentionPoolRope, IndexedAttentionReuseSelection, IndexedAttentionSelect,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt;

const MTP_BRANCHES: usize = 4;
const MTP_ROTARY_DIM: usize = 64;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MTP_MODEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_mtp_model_id() -> u64 {
    NEXT_MTP_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

fn invalid(message: impl Into<String>) -> MtpGpuError {
    MtpGpuError::Invalid(message.into())
}

fn view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    tensor.sub_offset(offset, len)
}

fn f32_view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert_eq!(tensor.dtype, DType::F32);
    view(tensor, offset, len)
}

fn hc_read(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    weights: &Qwen4Weights,
    norm_ref: &TensorRef,
    down_ref: &TensorRef,
    up_ref: &TensorRef,
    input: &GpuTensor,
    normalized: &GpuTensor,
    low: &GpuTensor,
    up: &GpuTensor,
    mixed: &GpuTensor,
    rotation: &GpuTensor,
) -> Result<(), MtpGpuError> {
    let norm = weights.resident(norm_ref)?;
    let down = weights.resident(down_ref)?;
    let up_weight = weights.resident(up_ref)?;
    hyper_norm(
        gpu,
        &HyperNorm {
            input,
            norm_weight: norm,
            normalized,
            branches: config.hc_count,
            hidden: config.hidden_size,
            state_bf16: false,
        },
    )?;
    dispatch_gemv(
        gpu,
        down,
        normalized,
        rotation,
        low,
        config.hc_lowrank,
        config.hc_count * config.hidden_size,
    )?;
    hc_activation_fused_f32(
        gpu,
        &HcActivationFused {
            values: low,
            scale: 1.0 / config.hc_count as f32,
            bf16_out: None,
        },
    )?;
    dispatch_gemv(
        gpu,
        up_weight,
        low,
        rotation,
        up,
        config.hc_count * config.hidden_size,
        config.hc_lowrank,
    )?;
    let projected_up = f32_view(up, 0, config.hc_count * config.hidden_size);
    hyper_read_projected(
        gpu,
        &HyperReadProjected {
            input,
            norm_weight: norm,
            up: &projected_up,
            normalized,
            mixed,
            branches: config.hc_count,
            hidden: config.hidden_size,
        },
    )?;
    Ok(())
}

fn hc_write(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    weights: &Qwen4Weights,
    hyper: &HyperConnectionWeights,
    input: &GpuTensor,
    normalized: &GpuTensor,
    mixed: &GpuTensor,
    gates: &GpuTensor,
    output: &GpuTensor,
    rotation: &GpuTensor,
) -> Result<(), MtpGpuError> {
    let norm = weights.resident(&hyper.hc_norm)?;
    let inject = weights.resident(&hyper.block_inject)?;
    hyper_norm(
        gpu,
        &HyperNorm {
            input,
            norm_weight: norm,
            normalized,
            branches: config.hc_count,
            hidden: config.hidden_size,
            state_bf16: false,
        },
    )?;
    dispatch_gemv(
        gpu,
        inject,
        normalized,
        rotation,
        gates,
        config.hc_count,
        config.hc_count * config.hidden_size,
    )?;
    hyper_write(
        gpu,
        &HyperWrite {
            input,
            normalized,
            mixed,
            gates,
            output,
            branches: config.hc_count,
            hidden: config.hidden_size,
            state_bf16: false,
        },
    )?;
    Ok(())
}

/// Embed the pending MTP token from the trunk-owned tied table.
///
/// The resident-dtype→lookup decision belongs to
/// [`crate::projection::dispatch_embedding`] alone.  This module used to carry
/// its own dtype table, which silently admitted fewer tiers than the trunk did;
/// sharing one dispatch is what keeps the draft head and the target from
/// disagreeing about the embedding tier an artifact may use.
fn embed_token(
    gpu: &mut Gpu,
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    scratch: &MtpGpuScratch,
) -> Result<(), MtpGpuError> {
    let embedding = weights.resident(&weights.root.embedding)?;
    dispatch_embedding(
        gpu,
        embedding,
        &scratch.embedding_rot,
        &scratch.token_embedding,
        &scratch.token_ids,
        1,
        config.hidden_size,
    )
    .map_err(MtpGpuError::Hip)
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

/// Fixed operator buffers for one MTP token.  Every field is allocated once
/// when the model's MTP capability is attached; calls only create subviews.
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
    rotation: GpuTensor,
    index: GpuTensor,
    index_query: GpuTensor,
    index_key: GpuTensor,
    q_and_gate: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    qsa_output: GpuTensor,
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
    logits: GpuTensor,
    top1: GpuTensor,
    host_token_bytes: [u8; 4],
}

impl MtpGpuScratch {
    fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Self, MtpGpuError> {
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
        let index_query_width = config
            .indexer_n_heads
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| invalid("MTP index-query width overflow"))?;
        let hc_up = wide
            .checked_mul(config.hc_lowrank)
            .ok_or_else(|| invalid("MTP HC up scratch overflow"))?;
        let max_rotation = wide.max(hidden).max(config.hc_lowrank).max(q_width);
        let mut allocated = Vec::new();
        let mut alloc = |shape: &[usize], dtype: DType| -> Result<(), MtpGpuError> {
            allocated.push(gpu.zeros(shape, dtype)?);
            Ok(())
        };
        let result = (|| {
            alloc(&[std::mem::size_of::<i32>()], DType::Raw)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[config.hc_lowrank], DType::F32)?;
            alloc(&[hc_up], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[config.hc_count], DType::F32)?;
            alloc(&[max_rotation], DType::F32)?;
            alloc(&[index_width], DType::F32)?;
            alloc(&[index_query_width], DType::F32)?;
            alloc(
                &[config.indexer_kv_heads * config.indexer_head_dim],
                DType::F32,
            )?;
            alloc(&[2 * q_width], DType::F32)?;
            alloc(&[kv_width], DType::F32)?;
            alloc(&[kv_width], DType::F32)?;
            alloc(&[q_width], DType::F32)?;
            alloc(&[config.num_experts], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[2 * config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.moe_intermediate_size], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(
                &[config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(&[config.num_experts_per_tok], DType::F32)?;
            alloc(&[config.num_experts_per_tok], DType::F32)?;
            alloc(
                &[config.num_experts_per_tok * hidden + hidden.div_ceil(4)],
                DType::F32,
            )?;
            alloc(&[config.shared_expert_intermediate_size.max(1)], DType::F32)?;
            alloc(&[config.vocab_size], DType::F32)?;
            alloc(&[std::mem::size_of::<i32>()], DType::Raw)?;
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
            rotation: next(),
            index: next(),
            index_query: next(),
            index_key: next(),
            q_and_gate: next(),
            qsa_k: next(),
            qsa_v: next(),
            qsa_output: next(),
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
            logits: next(),
            top1: next(),
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
            self.rotation,
            self.index,
            self.index_query,
            self.index_key,
            self.q_and_gate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
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
            self.logits,
            self.top1,
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
        })
    }

    pub(crate) fn reset(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.snapshot_arena.invalidate();
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
            ..
        } = self;
        let mut first = snapshot_arena.free_gpu(gpu);
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
    pub(crate) state: MtpGpuState,
    pub(crate) moe: Qwen4MoeLayerRuntime,
    pub(crate) max_seq: usize,
}

impl Qwen4MtpGpu {
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
        let scratch = MtpGpuScratch::new(gpu, config)?;
        let state = match MtpGpuState::new(gpu, config, max_seq) {
            Ok(state) => state,
            Err(error) => {
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        let moe = match Qwen4MoeLayerRuntime::from_moe(gpu, weights, &weights.mtp.moe, 0, config) {
            Ok(moe) => moe,
            Err(error) => {
                let _ = state.free_gpu(gpu);
                let _ = scratch.free_gpu(gpu);
                return Err(error.into());
            }
        };
        Ok(Self {
            scratch,
            state,
            moe,
            max_seq,
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        let Self {
            scratch,
            state,
            moe,
            ..
        } = self;
        let scratch_error = scratch.free_gpu(gpu);
        let state_error = state.free_gpu(gpu);
        let moe_error = moe.free_gpu(gpu);
        scratch_error
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

    pub(crate) fn reset(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.state.reset(gpu)
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
        if logits.dtype != DType::F32 || logits.numel() != self.scratch.logits.numel() {
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
                true,
            )?
            .ok_or_else(|| invalid("MTP prediction requested but no token produced"))?;
        gpu.copy_d2d(&self.scratch.logits, logits, logits.byte_size())?;
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
        predict: bool,
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
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| invalid("MTP position overflow"))?;
        let scratch = &mut self.scratch;
        let state = &mut self.state;
        let backbone_hidden = backbone_hidden.unwrap_or(&scratch.backbone_hidden);
        scratch
            .host_token_bytes
            .copy_from_slice(&(token as i32).to_ne_bytes());
        gpu.memcpy_htod_auto(&scratch.token_ids.buf, &scratch.host_token_bytes)?;
        embed_token(gpu, weights, config, scratch)?;

        let embedding_norm_weight = weights.resident(&weights.mtp.pre_fc_norm_embedding)?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &scratch.token_embedding,
                norm_weight: embedding_norm_weight,
                normalized: &scratch.embedding_norm,
                branches: 1,
                hidden: config.hidden_size,
                state_bf16: false,
            },
        )?;
        let fc_embedding = weights.resident(&weights.mtp.fc_embedding)?;
        dispatch_gemv(
            gpu,
            fc_embedding,
            &scratch.embedding_norm,
            &scratch.rotation,
            &scratch.projected_embedding,
            config.hidden_size,
            config.hidden_size,
        )?;

        let hidden_norm_weight = weights.resident(&weights.mtp.pre_fc_norm_hidden)?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: backbone_hidden,
                norm_weight: hidden_norm_weight,
                normalized: &scratch.hidden_norm,
                branches: MTP_BRANCHES,
                hidden: config.hidden_size,
                state_bf16: false,
            },
        )?;
        let fc_hidden = weights.resident(&weights.mtp.fc_hidden)?;
        for branch in 0..MTP_BRANCHES {
            let hidden_row = f32_view(
                &scratch.hidden_norm,
                branch * config.hidden_size,
                config.hidden_size,
            );
            let projected_row = f32_view(
                &scratch.projected_hidden,
                branch * config.hidden_size,
                config.hidden_size,
            );
            let wide_row = f32_view(
                &scratch.wide,
                branch * config.hidden_size,
                config.hidden_size,
            );
            dispatch_gemv(
                gpu,
                fc_hidden,
                &hidden_row,
                &scratch.rotation,
                &projected_row,
                config.hidden_size,
                config.hidden_size,
            )?;
            gpu.add_f32(&projected_row, &scratch.projected_embedding, &wide_row)?;
        }

        hc_read(
            gpu,
            config,
            weights,
            &weights.mtp.attn_hyper.hc_norm,
            &weights.mtp.attn_hyper.input_mix_down,
            &weights.mtp.attn_hyper.input_mix_up,
            &scratch.wide,
            &scratch.hc_normalized,
            &scratch.hc_low,
            &scratch.hc_up,
            &scratch.hc_mixed,
            &scratch.rotation,
        )?;
        let qsa = &weights.mtp.attention;
        let index_dim = config.indexer_head_dim;
        let index_query_width = config.indexer_n_heads * index_dim;
        let index_key_width = config.indexer_kv_heads * index_dim;
        let index_width = index_query_width + index_key_width;
        let index_weight = weights.resident(&qsa.indexer_qk)?;
        dispatch_gemv(
            gpu,
            index_weight,
            &scratch.hc_mixed,
            &scratch.rotation,
            &scratch.index,
            index_width,
            config.hidden_size,
        )?;
        let index_query = f32_view(&scratch.index, 0, index_query_width);
        let index_key = f32_view(&scratch.index, index_query_width, index_key_width);
        gpu.copy_d2d(
            &index_query,
            &scratch.index_query,
            scratch.index_query.byte_size(),
        )?;
        gpu.copy_d2d(
            &index_key,
            &scratch.index_key,
            scratch.index_key.byte_size(),
        )?;
        let index_query_norm = weights.resident(&qsa.indexer_q_norm)?;
        let index_key_norm = weights.resident(&qsa.indexer_k_norm)?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &scratch.index_query,
                norm: index_query_norm,
                heads: config.indexer_n_heads,
                head_dim: index_dim,
                head_stride: config.indexer_head_dim,
                position,
                rotary_dim: MTP_ROTARY_DIM.min(index_dim),
            },
        )?;
        // Keep raw BF16 index keys in the cache; source key RMSNorm and RoPE
        // happen after four-token block pooling.
        gpu.bf16_round_trip_f32(&scratch.index_key)?;
        let raw_offset = position
            .checked_mul(index_key_width)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| invalid("MTP index cache offset overflow"))?;
        gpu.memcpy_dtod_at_auto(
            &state.raw_index_keys.buf,
            raw_offset,
            &scratch.index_key.buf,
            0,
            scratch.index_key.byte_size(),
        )?;

        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_weight = weights.resident(&qsa.q)?;
        let k_weight = weights.resident(&qsa.k)?;
        let v_weight = weights.resident(&qsa.v)?;
        let q_and_gate = &scratch.q_and_gate;
        dispatch_gemv(
            gpu,
            q_weight,
            &scratch.hc_mixed,
            &scratch.rotation,
            q_and_gate,
            2 * q_width,
            config.hidden_size,
        )?;
        dispatch_gemv(
            gpu,
            k_weight,
            &scratch.hc_mixed,
            &scratch.rotation,
            &scratch.qsa_k,
            kv_width,
            config.hidden_size,
        )?;
        dispatch_gemv(
            gpu,
            v_weight,
            &scratch.hc_mixed,
            &scratch.rotation,
            &scratch.qsa_v,
            kv_width,
            config.hidden_size,
        )?;
        let q_norm = weights.resident(&qsa.q_norm)?;
        let k_norm = weights.resident(&qsa.k_norm)?;
        // q_proj already emits [Q, gate] for each head. Normalize and rotate
        // each Q half in place while preserving its adjacent gate half.
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: q_and_gate,
                norm: q_norm,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                head_stride: 2 * config.head_dim,
                position,
                rotary_dim: MTP_ROTARY_DIM.min(config.head_dim),
            },
        )?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &scratch.qsa_k,
                norm: k_norm,
                heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                head_stride: config.head_dim,
                position,
                rotary_dim: MTP_ROTARY_DIM.min(config.head_dim),
            },
        )?;
        indexed_attention_cache_append(
            gpu,
            &IndexedAttentionCacheAppend {
                key: &scratch.qsa_k,
                value: &scratch.qsa_v,
                full_keys: &state.full_keys,
                full_values: &state.full_values,
                position,
                kv_width,
            },
        )?;
        let visible = next_position;
        let complete = visible / config.indexer_compress_ratio;
        if complete > 0 {
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &state.raw_index_keys,
                    pooled: &state.pooled_keys,
                    norm: Some(index_key_norm),
                    block_count: complete,
                    compress: config.indexer_compress_ratio,
                    index_dim: index_key_width,
                    position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                        position_start: visible.saturating_sub(1),
                        rows: 1,
                    }),
                    grid_bound: complete,
                },
            )?;
        }
        let budget_blocks = config.indexer_budget / config.indexer_compress_ratio;
        // Reselect for the first step of each proposal (and every prefill token);
        // the request cursor still tracks state, not the selection's lifetime.
        if fresh_qsa_selection {
            indexed_attention_select(
                gpu,
                &IndexedAttentionSelect {
                    query: &scratch.index_query,
                    pooled: &state.pooled_keys,
                    selected: &state.selected_indices,
                    block_count: complete,
                    index_heads: config.indexer_n_heads,
                    index_dim,
                    budget_blocks,
                    compress: config.indexer_compress_ratio,
                    visible,
                    capacity: state.selected_capacity,
                },
            )?;
            let selected = budget_blocks.min(complete) * config.indexer_compress_ratio + visible
                - complete * config.indexer_compress_ratio;
            state.selected_len = selected.min(state.selected_capacity);
        } else {
            indexed_attention_reuse_selection(
                gpu,
                &IndexedAttentionReuseSelection {
                    selected: &state.selected_indices,
                    selected_len: state.selected_len,
                    position,
                    capacity: state.selected_capacity,
                    selected_len_out: &state.selected_len_out,
                },
            )?;
            let mut selected_len_bytes = [0u8; std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut selected_len_bytes, &state.selected_len_out.buf)?;
            let selected_len = i32::from_ne_bytes(selected_len_bytes);
            if selected_len < 0 {
                return Err(invalid(
                    "Qwen4 QSA reuse returned a negative selection length",
                ));
            }
            let selected_len = selected_len as usize;
            if selected_len == 0 || selected_len > state.selected_capacity || selected_len > visible
            {
                return Err(invalid(format!(
                    "Qwen4 QSA reuse returned invalid selection length {selected_len} \
                     (capacity {}, visible {visible})",
                    state.selected_capacity
                )));
            }
            state.selected_len = selected_len;
        }
        let selected = state.selected_len;
        indexed_attention_attention(
            gpu,
            &IndexedAttentionAttention {
                q_with_gate: q_and_gate,
                full_keys: &state.full_keys,
                full_values: &state.full_values,
                selected: &state.selected_indices,
                output: &scratch.qsa_output,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                selected_len: selected,
                full_capacity: state.full_capacity,
            },
        )?;
        let attention_output = weights.resident(&qsa.output)?;
        dispatch_gemv(
            gpu,
            attention_output,
            &scratch.qsa_output,
            &scratch.rotation,
            &scratch.projected_embedding,
            config.hidden_size,
            q_width,
        )?;
        hc_write(
            gpu,
            config,
            weights,
            &weights.mtp.attn_hyper,
            &scratch.wide,
            &scratch.hc_normalized,
            &scratch.projected_embedding,
            &scratch.hc_gates,
            &scratch.wide,
            &scratch.rotation,
        )?;

        hc_read(
            gpu,
            config,
            weights,
            &weights.mtp.mlp_hyper.hc_norm,
            &weights.mtp.mlp_hyper.input_mix_down,
            &weights.mtp.mlp_hyper.input_mix_up,
            &scratch.wide,
            &scratch.hc_normalized,
            &scratch.hc_low,
            &scratch.hc_up,
            &scratch.hc_mixed,
            &scratch.rotation,
        )?;
        execute_moe(
            gpu,
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
                // The shared down projection must not overwrite the routed accumulator.
                ffn_out: &scratch.projected_embedding,
                gate_batch: &scratch.moe_gate_batch,
                up_batch: &scratch.moe_up_batch,
                rot_batch: &scratch.moe_rot_batch,
                topk_indices: &scratch.moe_topk_indices,
                topk_weights: &scratch.moe_topk_weights,
                down_expanded: &scratch.moe_down_expanded,
            },
        )?;
        hc_write(
            gpu,
            config,
            weights,
            &weights.mtp.mlp_hyper,
            &scratch.wide,
            &scratch.hc_normalized,
            &scratch.moe_output,
            &scratch.hc_gates,
            &scratch.wide,
            &scratch.rotation,
        )?;
        let next_token = if predict {
            hc_read(
                gpu,
                config,
                weights,
                &weights.mtp.final_hyper.hc_norm,
                &weights.mtp.final_hyper.input_mix_down,
                &weights.mtp.final_hyper.input_mix_up,
                &scratch.wide,
                &scratch.hc_normalized,
                &scratch.hc_low,
                &scratch.hc_up,
                &scratch.hc_mixed,
                &scratch.rotation,
            )?;
            let lm_head = weights.resident(&weights.root.lm_head)?;
            dispatch_gemv(
                gpu,
                lm_head,
                &scratch.hc_mixed,
                &scratch.rotation,
                &scratch.logits,
                config.vocab_size,
                config.hidden_size,
            )?;
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: &scratch.logits,
                    indices: &scratch.top1,
                    rows: 1,
                    vocab: config.vocab_size,
                },
            )?;
            let mut token_bytes = [0u8; 4];
            gpu.hip.memcpy_dtoh(&mut token_bytes, &scratch.top1.buf)?;
            let next_token = u32::from_ne_bytes(token_bytes);
            if next_token as usize >= config.vocab_size {
                return Err(invalid(format!(
                    "MTP argmax token {next_token} is outside vocab {}",
                    config.vocab_size
                )));
            }
            Some(next_token)
        } else {
            None
        };
        gpu.copy_d2d(&scratch.wide, &state.wide_hidden, scratch.wide.byte_size())?;
        state.position = next_position;
        state.full_len = visible;
        state.raw_len = visible;
        state.pooled_len = complete;
        state.selected_len = selected;
        state.step_index = state.step_index.wrapping_add(1);
        Ok(next_token)
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
        let mut target = Qwen4State::new(&mut gpu, &config, 8).expect("compact target state");
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
