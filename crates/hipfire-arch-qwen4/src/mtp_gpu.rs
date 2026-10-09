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
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::program::{Qwen4HyperReadWeights, Qwen4HyperWriteWeights, Qwen4QsaWeights};
use crate::state::{meta_bytes, meta_words, whole_part};
use crate::weights::{Qwen4Weights, WeightError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, BroadcastAddOp, ClearOp, DraftHead, DraftHeadLayout,
    DraftHeadRequestState,
    DraftHeadPolicy, EmbeddingOp, HyperNormOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode,
    IndexedAttentionOp, IndexedAttentionState, ProjectOp, Step,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::kv_backend::{
    KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use hipfire_runtime::session_cache::{RowStream, StateLayout};
use hipfire_runtime::spec::SpecGrammar;
use rdna_compute::tensor_ops::{
    hyper_norm, hyper_read_projected, indexed_attention_append_prologue,
    indexed_attention_pool_rope_incremental, HyperNorm, HyperReadProjected,
    IndexedAttentionAppendPrologue, IndexedAttentionPoolRope, QsaKvFormat, QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;
use std::fmt;

const MTP_BRANCHES: usize = 4;
/// Rows of every MTP HC read and write: one token.
const MTP_HC_ROWS: usize = 1;
/// Metadata words of an MTP head session snapshot: a reserved zero word and
/// the state mark.
const MTP_SESSION_WORDS: usize = 7;
/// Prompt rows one batched Append pass (`Qwen4MtpGpu::append_rows`) runs per
/// launch sequence: the capacity of [`MtpAppendScratch`].
pub(crate) const MTP_FILL_ROWS: usize = 1024;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MTP_MODEL_ID: AtomicU64 = AtomicU64::new(1);

fn next_mtp_model_id() -> u64 {
    NEXT_MTP_MODEL_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

fn invalid(message: impl Into<String>) -> MtpGpuError {
    MtpGpuError::Invalid(message.into())
}

/// One-row HC read of the MTP streams into `hc_mixed`.
fn hc_read<'a>(
    read: &Qwen4HyperReadWeights<'a>,
    scratch: &'a MtpGpuScratch,
    config: &Qwen4Config,
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
        rows: MTP_HC_ROWS,
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
        rows: MTP_HC_ROWS,
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
    /// Element count and dtype of each scratch tensor, in field order.
    fn shapes(config: &Qwen4Config) -> Result<[(usize, DType); MTP_SCRATCH_TENSORS], MtpGpuError> {
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
        let hc_up = hc_up_elements(MTP_HC_ROWS, wide, config.hc_lowrank)
            .ok_or_else(|| invalid("MTP HC up scratch overflow"))?;
        let max_rotation = wide.max(hidden).max(config.hc_lowrank).max(q_width);
        let routed = config.num_experts_per_tok * config.moe_intermediate_size;
        Ok([
            (std::mem::size_of::<i32>(), DType::Raw),
            (hidden, DType::F32),
            (hidden, DType::F32),
            (hidden, DType::F32),
            (wide, DType::F32),
            (hidden, DType::F32),
            (wide, DType::F32),
            (wide, DType::F32),
            (wide, DType::F32),
            (wide, DType::F32),
            (config.hc_lowrank, DType::F32),
            (hc_up, DType::F32),
            (hidden, DType::F32),
            (config.hc_count, DType::F32),
            (
                hidden.max(config.hc_lowrank).max(q_width).max(kv_width),
                DType::BF16,
            ),
            (max_rotation, DType::F32),
            (index_width, DType::F32),
            (2 * q_width, DType::F32),
            (kv_width, DType::F32),
            (kv_width, DType::F32),
            (q_width, DType::F32),
            (
                config.qsa_selected_capacity() * std::mem::size_of::<i32>(),
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

    /// Device bytes [`Self::new`] allocates.
    fn device_bytes(config: &Qwen4Config) -> Result<usize, MtpGpuError> {
        Ok(Self::shapes(config)?
            .iter()
            .map(|&(elements, dtype)| elements * dtype.size())
            .sum())
    }

    fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Self, MtpGpuError> {
        let shapes = Self::shapes(config)?;
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

/// Row widths shared by [`MtpAppendScratch`] and `Qwen4MtpGpu::append_rows`.
#[derive(Clone, Copy)]
struct AppendWidths {
    hidden: usize,
    /// `MTP_BRANCHES * hidden`: one row of the HC streams.
    wide: usize,
    /// Index query heads then index key: `(indexer_n_heads + indexer_kv_heads) * indexer_head_dim`.
    index_width: usize,
    index_kv_width: usize,
    kv_width: usize,
}

impl AppendWidths {
    fn new(config: &Qwen4Config) -> Option<Self> {
        let index_kv_width = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)?;
        Some(Self {
            hidden: config.hidden_size,
            wide: MTP_BRANCHES.checked_mul(config.hidden_size)?,
            index_width: config
                .indexer_n_heads
                .checked_mul(config.indexer_head_dim)?
                .checked_add(index_kv_width)?,
            index_kv_width,
            kv_width: config
                .num_key_value_heads
                .checked_mul(config.head_dim)?,
        })
    }
}

/// Allocations of one [`MtpAppendScratch`], in field order.
const MTP_APPEND_TENSORS: usize = 14;

/// Row-batched operator buffers of the prompt-fill Append pass
/// (`Qwen4MtpGpu::append_rows`): the per-row [`MtpGpuScratch`] tensors that an
/// Append step touches, each with `rows` rows. Never aliases the per-row scratch.
pub(crate) struct MtpAppendScratch {
    /// Row capacity.
    rows: usize,
    /// `rows` i32 token ids, as raw bytes.
    token_ids: GpuTensor,
    token_embedding: GpuTensor,
    embedding_norm: GpuTensor,
    projected_embedding: GpuTensor,
    hidden_norm: GpuTensor,
    projected_hidden: GpuTensor,
    wide: GpuTensor,
    hc_normalized: GpuTensor,
    hc_low: GpuTensor,
    hc_up: GpuTensor,
    hc_mixed: GpuTensor,
    index: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    host_token_bytes: Vec<u8>,
}

impl MtpAppendScratch {
    /// Row capacity: the most rows one `append_rows` call takes.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Element count and dtype of each scratch tensor, in field order.
    fn shapes(config: &Qwen4Config, rows: usize) -> Option<[(usize, DType); MTP_APPEND_TENSORS]> {
        let w = AppendWidths::new(config)?;
        let per = |width: usize| rows.checked_mul(width);
        Some([
            (rows.checked_mul(std::mem::size_of::<i32>())?, DType::Raw),
            (per(w.hidden)?, DType::F32),
            (per(w.hidden)?, DType::F32),
            (per(w.hidden)?, DType::F32),
            (per(w.wide)?, DType::F32),
            (per(w.wide)?, DType::F32),
            (per(w.wide)?, DType::F32),
            (per(w.wide)?, DType::F32),
            (per(config.hc_lowrank)?, DType::F32),
            (per(w.wide)?, DType::F32),
            (per(w.hidden)?, DType::F32),
            (per(w.index_width)?, DType::F32),
            (per(w.kv_width)?, DType::F32),
            (per(w.kv_width)?, DType::F32),
        ])
    }

    /// Device bytes [`Self::new`] allocates for `rows` rows.
    pub(crate) fn device_bytes(config: &Qwen4Config, rows: usize) -> Option<usize> {
        Self::shapes(config, rows)?
            .iter()
            .try_fold(0usize, |total, &(elements, dtype)| {
                total.checked_add(elements.checked_mul(dtype.size())?)
            })
    }

    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        rows: usize,
    ) -> Result<Self, MtpGpuError> {
        if rows == 0 {
            return Err(invalid("MTP append scratch needs at least one row"));
        }
        let shapes = Self::shapes(config, rows)
            .ok_or_else(|| invalid("MTP append scratch extent overflow"))?;
        let mut allocated = Vec::with_capacity(MTP_APPEND_TENSORS);
        for (elements, dtype) in shapes {
            match gpu.zeros(&[elements], dtype) {
                Ok(tensor) => allocated.push(tensor),
                Err(error) => {
                    for tensor in allocated {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(error.into());
                }
            }
        }
        let mut tensors = allocated.into_iter();
        let mut next = || tensors.next().expect("append scratch tensor count");
        Ok(Self {
            rows,
            token_ids: next(),
            token_embedding: next(),
            embedding_norm: next(),
            projected_embedding: next(),
            hidden_norm: next(),
            projected_hidden: next(),
            wide: next(),
            hc_normalized: next(),
            hc_low: next(),
            hc_up: next(),
            hc_mixed: next(),
            index: next(),
            qsa_k: next(),
            qsa_v: next(),
            host_token_bytes: vec![0; rows * std::mem::size_of::<i32>()],
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        let tensors = [
            self.token_ids,
            self.token_embedding,
            self.embedding_norm,
            self.projected_embedding,
            self.hidden_norm,
            self.projected_hidden,
            self.wide,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.index,
            self.qsa_k,
            self.qsa_v,
        ];
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), |error| Err(error.into()))
    }
}

/// The MTP weights an Append step reads, resolved and shape-checked once for
/// the batched pass (the per-row path resolves the same handles through its
/// `Step` descriptors, `forward_token`).
struct AppendWeights<'a> {
    embedding: &'a GpuTensor,
    norm_embedding: &'a GpuTensor,
    norm_hidden: &'a GpuTensor,
    fc_embedding: WeightRef<'a>,
    fc_hidden: WeightRef<'a>,
    /// Attention-side HC read: its norm, down (`wide -> hc_lowrank`) and up
    /// (`hc_lowrank -> wide`) projections.
    hc: Qwen4HyperReadWeights<'a>,
    qsa: Qwen4QsaWeights<'a>,
}

impl<'a> AppendWeights<'a> {
    fn resolve(
        weights: &'a Qwen4Weights,
        config: &Qwen4Config,
        w: &AppendWidths,
    ) -> Result<Self, MtpGpuError> {
        if config.hc_count != MTP_BRANCHES {
            return Err(invalid("MTP batched append needs four HC branches"));
        }
        let check = |label: &str,
                     weight: &WeightRef<'_>,
                     dtype: DType,
                     m: usize,
                     k: usize|
         -> Result<(), MtpGpuError> {
            if weight.dtype != dtype || weight.m != m || weight.k != k {
                return Err(invalid(format!(
                    "MTP batched append {label} is {:?} [{}x{}], expected {dtype:?} [{m}x{k}]",
                    weight.dtype, weight.m, weight.k
                )));
            }
            Ok(())
        };
        let embedding = weights.resident(&weights.root.embedding)?;
        if embedding.dtype != DType::Q8_0 {
            return Err(invalid("MTP batched append needs a Q8_0 embedding table"));
        }
        let fc_embedding = dense_ref(weights, &weights.mtp.fc_embedding)?;
        check("fc_embedding", &fc_embedding, DType::BF16, w.hidden, w.hidden)?;
        let fc_hidden = dense_ref(weights, &weights.mtp.fc_hidden)?;
        check("fc_hidden", &fc_hidden, DType::BF16, w.hidden, w.hidden)?;
        let hc = hyper_desc(weights, &weights.mtp.attn_hyper)?.read;
        if !w.wide.is_multiple_of(32) {
            return Err(invalid("MTP batched append HC down needs K % 32 == 0"));
        }
        check("HC down", &hc.input_mix_down, DType::BF16, config.hc_lowrank, w.wide)?;
        check("HC up", &hc.input_mix_up, DType::BF16, w.wide, config.hc_lowrank)?;
        let qsa = qsa_desc(weights, &weights.mtp.attention)?;
        // The tiled Q8_0 projection is the K = 2560 staged kernel.
        if w.hidden != 2560 {
            return Err(invalid("MTP batched append Q8_0 projections need hidden 2560"));
        }
        check("indexer_qk", &qsa.indexer_qk, DType::Q8_0, w.index_width, w.hidden)?;
        check("k_proj", &qsa.k, DType::Q8_0, w.kv_width, w.hidden)?;
        check("v_proj", &qsa.v, DType::Q8_0, w.kv_width, w.hidden)?;
        Ok(Self {
            embedding,
            norm_embedding: weights.resident(&weights.mtp.pre_fc_norm_embedding)?,
            norm_hidden: weights.resident(&weights.mtp.pre_fc_norm_hidden)?,
            fc_embedding,
            fc_hidden,
            hc,
            qsa,
        })
    }
}

/// F32 elements of every subview the shared HC read takes from its `up`
/// buffer for `rows` rows: the F32 up projection (`rows × wide`), and on the
/// WMMA read route the packed BF16 `low`, staged as a `rows × rank` view of
/// the same F32 tensor before its dtype is relabelled (`layer_ops.rs`
/// `execute_hyper_read_inner`). The up weight itself is never staged here.
fn hc_up_elements(rows: usize, wide: usize, rank: usize) -> Option<usize> {
    Some(rows.checked_mul(wide)?.max(rows.checked_mul(rank)?))
}

/// Refuse an HC read whose `up` subviews exceed the allocated scratch.
fn require_hc_up(up: &GpuTensor, rows: usize, wide: usize, rank: usize) -> Result<(), MtpGpuError> {
    let needed =
        hc_up_elements(rows, wide, rank).ok_or_else(|| invalid("MTP HC up extent overflow"))?;
    let needed_bytes = needed
        .checked_mul(DType::F32.size())
        .ok_or_else(|| invalid("MTP HC up extent overflow"))?;
    if up.dtype != DType::F32 || up.numel() < needed || up.buf.size() < needed_bytes {
        return Err(invalid(format!(
            "MTP HC up scratch holds {} F32 elements; {rows} HC rows use {needed}",
            up.numel()
        )));
    }
    Ok(())
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

/// Map more of one VMM context arena so it covers `rows` rows of a
/// `capacity`-row tensor (byte stride = reserved size / capacity), by the
/// shared [`KvChunkPlan`] policy, and zero the newly mapped bytes.
fn grow_context_arena(
    gpu: &mut Gpu,
    tensor: &mut GpuTensor,
    capacity: usize,
    rows: usize,
) -> Result<(), MtpGpuError> {
    let mapped = gpu
        .vmm_mapped_bytes(tensor)
        .ok_or_else(|| invalid("MTP QSA arena is not a registered VMM owner"))?;
    let granularity = gpu
        .vmm_granularity(tensor)
        .ok_or_else(|| invalid("MTP QSA arena has no VMM granularity"))?;
    let plan = KvChunkPlan::new(
        tensor.byte_size() / capacity,
        capacity,
        DEFAULT_KV_CHUNK_TOKENS,
        granularity,
        DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
    )
    .map_err(|error| invalid(format!("MTP QSA chunk plan: {error}")))?;
    let Some(growth) = plan
        .growth(mapped, rows)
        .map_err(|error| invalid(format!("MTP QSA growth: {error}")))?
    else {
        return Ok(());
    };
    let device_id = gpu.device_id;
    gpu.grow_vmm_tensor(tensor, growth.size_bytes, &[device_id])?;
    // The accessible prefix stops at the logical size even when the last
    // page maps past it.
    let end = (growth.offset_bytes + growth.size_bytes).min(tensor.buf.size());
    if end > growth.offset_bytes {
        let element = tensor.dtype.size();
        let fresh = tensor.sub_offset(
            growth.offset_bytes / element,
            (end - growth.offset_bytes) / element,
        );
        gpu.hip.memset(&fresh.buf, 0, fresh.buf.size())?;
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
    /// Index compression ratio: one pooled row per `compress` tokens.
    compress: usize,
    backend: Qwen4KvBackend,
    /// Tokens every context arena's mapped prefix covers (the fast no-growth
    /// gate); legacy storage covers its full capacity.
    mapped_tokens: usize,
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
    /// Device bytes [`Self::new_with_backend`] commits: the context-sized
    /// QSA arenas as `context` commits them (legacy: all of them; VMM: the
    /// first chunk's mapped pages, the rest being virtual until forwards
    /// map it), then the selection and wide-hidden carry, twice with their
    /// snapshot backups.
    fn device_bytes(config: &Qwen4Config, context: &Qwen4ContextCommit) -> Option<usize> {
        let carry = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?
            .checked_add(std::mem::size_of::<i32>())?
            .checked_add(
                MTP_BRANCHES
                    .checked_mul(config.hidden_size)?
                    .checked_mul(std::mem::size_of::<f32>())?,
            )?;
        context
            .committed_layer_bytes(config, QsaKvFormat::F32)?
            .checked_add(carry.checked_mul(2)?)
    }
    /// Legacy full-capacity state; see [`Self::new_with_backend`].
    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq: usize,
    ) -> Result<Self, MtpGpuError> {
        Self::new_with_backend(gpu, config, max_seq, Qwen4KvBackend::Legacy)
    }

    /// State whose context arenas (full K/V, raw and pooled index keys) are
    /// `backend` storage of `max_seq` logical tokens: legacy arenas are full
    /// zeroed allocations; VMM arenas reserve that VA and map nothing until
    /// [`Self::ensure_mapped_capacity`]. The fixed carry is always allocated.
    pub(crate) fn new_with_backend(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq: usize,
        backend: Qwen4KvBackend,
    ) -> Result<Self, MtpGpuError> {
        let full_width = config.num_key_value_heads * config.head_dim;
        let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
        let compress = config.indexer_compress_ratio;
        let pooled_capacity = max_seq.div_ceil(compress);
        let selected_capacity = config.qsa_selected_capacity();
        let device_id = gpu.device_id;
        let shapes = [
            (max_seq * full_width, DType::F32, true),
            (max_seq * full_width, DType::F32, true),
            (max_seq * raw_width, DType::F32, true),
            (pooled_capacity * raw_width, DType::F32, true),
            (
                selected_capacity * std::mem::size_of::<i32>(),
                DType::Raw,
                false,
            ),
            (std::mem::size_of::<i32>(), DType::Raw, false),
            (MTP_BRANCHES * config.hidden_size, DType::F32, false),
        ];
        let mut allocated = Vec::with_capacity(shapes.len());
        for (elements, dtype, context) in shapes {
            let tensor = match (context, backend) {
                // SAFETY: every access to a VMM context arena is bounded by
                // `mapped_tokens`, which `ensure_mapped_capacity` raises only
                // after mapping and zeroing the covering pages.
                (true, Qwen4KvBackend::Vmm) => unsafe {
                    gpu.alloc_vmm_tensor(&[elements], dtype, 0, &[device_id])
                },
                _ => gpu.zeros(&[elements], dtype),
            };
            match tensor {
                Ok(tensor) => allocated.push(tensor),
                Err(error) => {
                    for tensor in allocated {
                        let _ = gpu.free_tensor(tensor);
                    }
                    return Err(error.into());
                }
            }
        }
        let mut tensors = allocated.into_iter();
        let mut next = || tensors.next().expect("one tensor per MTP state shape");
        let (full_keys, full_values, raw_index_keys, pooled_keys) =
            (next(), next(), next(), next());
        let (selected_indices, selected_len_out, wide_hidden) = (next(), next(), next());
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
                for tensor in [
                    full_keys,
                    full_values,
                    raw_index_keys,
                    pooled_keys,
                    selected_indices,
                    selected_len_out,
                    wide_hidden,
                ] {
                    let _ = gpu.free_tensor(tensor);
                }
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
            compress,
            backend,
            mapped_tokens: match backend {
                Qwen4KvBackend::Legacy => max_seq,
                Qwen4KvBackend::Vmm => 0,
            },
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

    /// Map (VMM) the context arenas to cover tokens `0..required_tokens`
    /// before any access to them: full K/V and raw index keys by token,
    /// pooled keys by `ceil(required_tokens / compress)` blocks, each with its
    /// own byte stride and the shared [`KvChunkPlan`] growth policy. Newly
    /// mapped pages are zeroed, as legacy arenas are. A request beyond the
    /// admitted capacity is refused before any mapping; legacy storage only
    /// checks that bound. Never maps during graph capture. After a failed
    /// map the completed arenas stay mapped and the coverage is unchanged.
    pub(crate) fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> Result<(), MtpGpuError> {
        if required_tokens > self.full_capacity {
            return Err(invalid(format!(
                "MTP QSA requires {required_tokens} tokens; admitted capacity is {}",
                self.full_capacity
            )));
        }
        if required_tokens <= self.mapped_tokens {
            return Ok(());
        }
        // Legacy coverage is the full capacity, so only VMM reaches here.
        if gpu.graphs.capture_mode {
            return Err(invalid("MTP QSA growth requested during graph capture"));
        }
        let pooled_required = required_tokens.div_ceil(self.compress);
        let (full, raw, pooled) = (self.full_capacity, self.raw_capacity, self.pooled_capacity);
        for (tensor, capacity, rows) in [
            (&mut self.full_keys, full, required_tokens),
            (&mut self.full_values, full, required_tokens),
            (&mut self.raw_index_keys, raw, required_tokens),
            (&mut self.pooled_keys, pooled, pooled_required),
        ] {
            grow_context_arena(gpu, tensor, capacity, rows)?;
        }
        self.mapped_tokens = self.mapped_token_coverage(gpu)?;
        debug_assert!(self.mapped_tokens >= required_tokens);
        Ok(())
    }

    /// Tokens every context arena's mapped prefix covers.
    fn mapped_token_coverage(&self, gpu: &Gpu) -> Result<usize, MtpGpuError> {
        let rows = |tensor: &GpuTensor, capacity: usize| -> Result<usize, MtpGpuError> {
            let mapped = gpu
                .vmm_mapped_bytes(tensor)
                .ok_or_else(|| invalid("MTP QSA arena is not a registered VMM owner"))?;
            Ok((mapped / (tensor.byte_size() / capacity)).min(capacity))
        };
        let full = rows(&self.full_keys, self.full_capacity)?
            .min(rows(&self.full_values, self.full_capacity)?)
            .min(rows(&self.raw_index_keys, self.raw_capacity)?);
        let pooled = rows(&self.pooled_keys, self.pooled_capacity)?
            .saturating_mul(self.compress)
            .min(self.full_capacity);
        Ok(full.min(pooled))
    }

    /// Physical bytes committed to the context arenas: the mapped prefixes
    /// (VMM), or the full allocations (legacy).
    pub(crate) fn mapped_context_bytes(&self, gpu: &Gpu) -> Result<usize, MtpGpuError> {
        let arenas = [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
        ];
        let mut bytes = 0usize;
        for tensor in arenas {
            let committed = match self.backend {
                Qwen4KvBackend::Legacy => tensor.byte_size(),
                Qwen4KvBackend::Vmm => gpu
                    .vmm_mapped_bytes(tensor)
                    .ok_or_else(|| invalid("MTP QSA arena is not a registered VMM owner"))?,
            };
            bytes = bytes
                .checked_add(committed)
                .ok_or_else(|| invalid("MTP QSA mapped bytes overflow"))?;
        }
        Ok(bytes)
    }

    /// Tokens the context arenas may currently be accessed for.
    pub(crate) fn mapped_tokens(&self) -> usize {
        self.mapped_tokens
    }

    pub(crate) fn backend(&self) -> Qwen4KvBackend {
        self.backend
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
            // An unmapped VMM arena has an empty accessible prefix.
            if tensor.buf.size() > 0 {
                gpu.hip.memset(&tensor.buf, 0, tensor.buf.size())?;
            }
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

    /// Layout of the state at `mark`: full K/V, raw and pooled rows below it,
    /// then selection, device selected length and wide hidden.
    fn session_layout(&self, mark: &MtpStateMark) -> StateLayout<'_> {
        StateLayout {
            fixed: vec![
                whole_part(&self.selected_indices),
                whole_part(&self.selected_len_out),
                whole_part(&self.wide_hidden),
            ],
            rows: [
                (&self.full_keys, self.full_capacity, mark.full_len),
                (&self.full_values, self.full_capacity, mark.full_len),
                (&self.raw_index_keys, self.raw_capacity, mark.raw_len),
                (&self.pooled_keys, self.pooled_capacity, mark.pooled_len),
            ]
            .into_iter()
            .map(|(tensor, capacity, rows)| RowStream {
                buf: &tensor.buf,
                row_bytes: tensor.byte_size() / capacity,
                rows,
            })
            .collect(),
        }
    }

    /// Session-cache capture: `MTP_SESSION_WORDS` metadata words and the
    /// device ranges of the live state.
    fn session_parts(&self) -> Result<(Vec<u64>, StateLayout<'_>), MtpGpuError> {
        let mark = self.mark();
        validate_mtp_mark(&mark, self)?;
        let words = [
            mark.full_len,
            mark.raw_len,
            mark.pooled_len,
            mark.selected_len,
            mark.position,
            mark.step_index,
        ]
        .map(|value| value as u64);
        Ok((
            [0].into_iter().chain(words).collect(),
            self.session_layout(&mark),
        ))
    }

    fn parse_session_mark(&self, words: &[u64]) -> Result<MtpStateMark, MtpGpuError> {
        if words.len() != MTP_SESSION_WORDS {
            return Err(invalid("MTP session snapshot shape mismatch"));
        }
        let mark = MtpStateMark {
            full_len: words[1] as usize,
            raw_len: words[2] as usize,
            pooled_len: words[3] as usize,
            selected_len: words[4] as usize,
            position: words[5] as usize,
            step_index: words[6] as usize,
        };
        validate_mtp_mark(&mark, self)?;
        Ok(mark)
    }

    /// Map the context arenas to cover every row the snapshot holds and, like
    /// a reset, start a new request epoch that retires every speculative
    /// ticket. Live buffers stay in place: the caller copies into the
    /// returned destination layout.
    fn prepare_session_restore(
        &mut self,
        gpu: &mut Gpu,
        words: &[u64],
    ) -> Result<StateLayout<'_>, MtpGpuError> {
        let mark = self.parse_session_mark(words)?;
        let required = mark
            .position
            .max(mark.full_len)
            .max(mark.raw_len)
            .max(
                mark.pooled_len
                    .saturating_mul(self.compress)
                    .min(self.full_capacity),
            );
        self.ensure_mapped_capacity(gpu, required)?;
        self.snapshot_arena.invalidate();
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.generation = self.generation.wrapping_add(1);
        Ok(self.session_layout(&mark))
    }

    fn finish_session_restore(&mut self, words: &[u64]) -> Result<(), MtpGpuError> {
        let mark = self.parse_session_mark(words)?;
        self.full_len = mark.full_len;
        self.raw_len = mark.raw_len;
        self.pooled_len = mark.pooled_len;
        self.selected_len = mark.selected_len;
        self.position = mark.position;
        self.step_index = mark.step_index;
        Ok(())
    }

    /// Context-arena bytes still unmapped below capacity (0 for legacy).
    fn context_growth_bytes(&self) -> u64 {
        if self.backend == Qwen4KvBackend::Legacy || self.full_capacity == 0 {
            return 0;
        }
        let capacity: u64 = [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
        ]
        .iter()
        .map(|tensor| tensor.byte_size() as u64)
        .sum();
        capacity * self.full_capacity.saturating_sub(self.mapped_tokens) as u64
            / self.full_capacity as u64
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
    /// Draft ranking (`HIPFIRE_MTP_DRAFT_HEAD=mq2..mq6[r]`, default `mq2r`;
    /// anything else = the model's own head). On the shipped Q8_0 head an
    /// MQ2 copy with exact re-scoring of its top 8 drafts the Q8_0 head's own
    /// argmax at a quarter of its read; plain MQ3 loses acceptance.
    pub(crate) draft: DraftHead,
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
    /// `(resident, load scratch)` device bytes [`Self::new_with_backend`]
    /// commits for QSA context storage `context` and a language head stored
    /// as `head_dtype`: what the attached head keeps, and the draft head's
    /// build scratch on top of it (`DraftHead::device_bytes`), released
    /// before construction returns.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        context: &Qwen4ContextCommit,
        head_dtype: DType,
    ) -> Option<(usize, usize)> {
        let (policy, layout) = draft_head_config(config);
        let (draft, scratch) = DraftHead::device_bytes(head_dtype, layout, policy)?;
        let resident = MtpGpuScratch::device_bytes(config)
            .ok()?
            .checked_add(MtpGpuState::device_bytes(config, context)?)?
            .checked_add(Qwen4MoeLayerRuntime::device_bytes(config)?)?
            .checked_add(draft)?;
        Some((resident, scratch))
    }

    /// MTP head whose QSA context arenas are `backend` storage (admission
    /// resolves it); see [`MtpGpuState::new_with_backend`].
    pub(crate) fn new_with_backend(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        max_seq: usize,
        backend: Qwen4KvBackend,
    ) -> Result<Self, MtpGpuError> {
        config.mtp.validate().map_err(MtpGpuError::Invalid)?;
        if max_seq == 0 || max_seq > config.max_position_embeddings {
            return Err(invalid("MTP max_seq is outside model capacity"));
        }
        let scratch = MtpGpuScratch::new(gpu, config)?;
        let state = match MtpGpuState::new_with_backend(gpu, config, max_seq, backend) {
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
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        Ok(Self {
            scratch,
            state,
            moe,
            max_seq,
            draft,
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        let Self {
            scratch,
            state,
            moe,
            draft,
            ..
        } = self;
        let draft_error = draft.free_gpu(gpu);
        let scratch_error = scratch.free_gpu(gpu);
        let state_error = state.free_gpu(gpu);
        let moe_error = moe.free_gpu(gpu);
        draft_error
            .or(scratch_error)
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

    /// See [`MtpGpuState::ensure_mapped_capacity`].
    pub(crate) fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> Result<(), MtpGpuError> {
        self.state.ensure_mapped_capacity(gpu, required_tokens)
    }

    /// See [`MtpGpuState::mapped_context_bytes`].
    pub(crate) fn mapped_context_bytes(&self, gpu: &Gpu) -> Result<usize, MtpGpuError> {
        self.state.mapped_context_bytes(gpu)
    }

    /// Session-cache capture of the head state plus the draft policy.
    /// Prompt appends only `observe` (a qualifying input sets the full hold);
    /// drafts are the only decrement, so at a prompt boundary the policy is
    /// the canonical prompt summary.
    pub(crate) fn session_parts(&self) -> Result<(Vec<u8>, StateLayout<'_>), MtpGpuError> {
        let (mut words, parts) = self.state.session_parts()?;
        let policy = self.draft.request_state();
        words.extend([
            u64::from(policy.full_steps),
            u64::from(policy.margin.to_bits()),
        ]);
        Ok((meta_bytes(&words), parts))
    }

    fn session_words(meta: &[u8]) -> Result<Vec<u64>, MtpGpuError> {
        meta_words(meta)
            .filter(|words| words.len() == MTP_SESSION_WORDS + 2)
            .ok_or_else(|| invalid("MTP session snapshot shape mismatch"))
    }

    pub(crate) fn prepare_session_restore(
        &mut self,
        gpu: &mut Gpu,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, MtpGpuError> {
        let words = Self::session_words(meta)?;
        self.state
            .prepare_session_restore(gpu, &words[..MTP_SESSION_WORDS])
    }

    pub(crate) fn finish_session_restore(&mut self, meta: &[u8]) -> Result<(), MtpGpuError> {
        let words = Self::session_words(meta)?;
        self.state
            .finish_session_restore(&words[..MTP_SESSION_WORDS])?;
        let policy = &words[MTP_SESSION_WORDS..];
        self.draft.set_request_state(DraftHeadRequestState {
            full_steps: u32::try_from(policy[0])
                .map_err(|_| invalid("MTP session draft policy"))?,
            margin: f32::from_bits(
                u32::try_from(policy[1]).map_err(|_| invalid("MTP session draft policy"))?,
            ),
        });
        Ok(())
    }

    /// See [`MtpGpuState::context_growth_bytes`].
    pub(crate) fn context_growth_bytes(&self) -> u64 {
        self.state.context_growth_bytes()
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
        // Mapping is the caller's (`ensure_mapped_capacity`), never here.
        if position >= self.state.mapped_tokens {
            return Err(invalid(format!(
                "MTP position {position} is past the mapped QSA coverage of {} tokens",
                self.state.mapped_tokens
            )));
        }
        require_hc_up(&self.scratch.hc_up, MTP_HC_ROWS, wide, config.hc_lowrank)?;
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
            }));
            steps.push(Step::HyperRead(hc_read(&attn.read, scratch, config)));
            steps.push(Step::IndexedAttention(attention));
            if let Some(moe) = moe {
                steps.push(Step::HyperWrite(hc_write(
                    &attn.write,
                    &scratch.projected_embedding,
                    scratch,
                    config,
                )));
                steps.push(Step::HyperRead(hc_read(&mlp.read, scratch, config)));
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
                    steps.push(Step::HyperRead(hc_read(&final_read, scratch, config)));
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

    /// Whether [`Self::append_rows`] can run for this model on this GPU:
    /// gfx1151 only (the tiled Q8_0 projection is the K = 2560 staged GEMV
    /// the per-row path takes there; other arches run `gemv_q8_0` and have no
    /// batched twin) and the shipped weight formats and shapes.
    pub(crate) fn append_rows_supported(
        gpu: &Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
    ) -> bool {
        gpu.arch_caps.is_gfx1151()
            && AppendWidths::new(config)
                .is_some_and(|w| AppendWeights::resolve(weights, config, &w).is_ok())
    }

    /// [`MtpStep::Append`] for `tokens.len()` consecutive prompt rows starting
    /// at `position`, row `i` paired with `hidden` row `i` (the per-row loop's
    /// (token p, hidden p) pairing). Runs the per-row `forward_token` Append
    /// op sequence once over all rows, calling the rdna-compute entry points
    /// directly (never the `Step` dispatcher or `project_weight`, whose
    /// multi-row routes change the arithmetic): each op is the single-row
    /// kernel with a rows grid, or a multi-row kernel whose per-row
    /// reduction order is the single-row kernel's.
    ///
    /// `hidden` holds at least `tokens.len()` rows of `MTP_BRANCHES * hidden`
    /// F32. Draft request state sees every token in order, as `forward_token`
    /// observes each; the QSA state commits `position + tokens.len()` rows.
    /// `wide_hidden`, the selection and its length are not touched (the
    /// per-row Append leaves them alone too).
    ///
    /// `device_ids`, when given, holds `tokens` as `tokens.len()` device i32
    /// ids (at least that many bytes) that the embedding gather reads in place
    /// of the scratch upload. The backing allocation must remain live and
    /// any later writes must be ordered after the enqueued embedding gather;
    /// returning from this call does not imply device completion.
    pub(crate) fn append_rows(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden: &GpuTensor,
        device_ids: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), MtpGpuError> {
        let rows = tokens.len();
        let w = AppendWidths::new(config).ok_or_else(|| invalid("MTP append width overflow"))?;
        if rows == 0 || rows > scratch.rows {
            return Err(invalid(format!(
                "MTP batched append of {rows} rows needs 1..={} (scratch capacity)",
                scratch.rows
            )));
        }
        let hidden_elements = rows
            .checked_mul(w.wide)
            .ok_or_else(|| invalid("MTP batched append hidden extent overflow"))?;
        if hidden.dtype != DType::F32 || hidden.numel() < hidden_elements {
            return Err(invalid(format!(
                "MTP batched append hidden must be F32 with at least {hidden_elements} elements"
            )));
        }
        if !gpu.arch_caps.is_gfx1151() {
            return Err(invalid("MTP batched append is gfx1151 only"));
        }
        if position != self.state.position {
            return Err(invalid(format!(
                "MTP position mismatch: expected {}, got {position}",
                self.state.position
            )));
        }
        let end = position
            .checked_add(rows)
            .ok_or_else(|| invalid("MTP batched append position overflow"))?;
        if end > self.max_seq {
            return Err(invalid("MTP position exceeds QSA capacity"));
        }
        // Mapping is the caller's (`ensure_mapped_capacity`), never here.
        if end > self.state.mapped_tokens {
            return Err(invalid(format!(
                "MTP rows through position {} are past the mapped QSA coverage of {} tokens",
                end - 1,
                self.state.mapped_tokens
            )));
        }
        let ids_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| invalid("MTP batched append token id extent overflow"))?;
        if let Some(ids) = device_ids {
            if ids.buf.size() < ids_bytes {
                return Err(invalid(format!(
                    "MTP batched append device token ids hold {} bytes, need {ids_bytes}",
                    ids.buf.size()
                )));
            }
        }
        let compress = config.indexer_compress_ratio;
        if compress == 0 {
            return Err(invalid("MTP indexer compress ratio is zero"));
        }
        let mw = AppendWeights::resolve(weights, config, &w)?;
        for &token in tokens {
            self.draft.observe(token);
        }
        // Reused device ids (the trunk's already uploaded rows, same bytes as
        // `tokens`) skip the host fill and copy; otherwise the scratch uploads.
        let token_ids = match device_ids {
            Some(ids) => ids,
            None => {
                for (bytes, &token) in scratch.host_token_bytes.chunks_exact_mut(4).zip(tokens) {
                    bytes.copy_from_slice(&(token as i32).to_ne_bytes());
                }
                gpu.memcpy_htod_auto(
                    &scratch.token_ids.buf,
                    &scratch.host_token_bytes[..rows * std::mem::size_of::<i32>()],
                )?;
                &scratch.token_ids
            }
        };

        // Exactly-`rows` views: HC norm and read derive their row count from
        // the tensor extent.
        let view = |tensor: &GpuTensor, width: usize| tensor.sub_offset(0, rows * width);
        let backbone = view(hidden, w.wide);
        let token_embedding = view(&scratch.token_embedding, w.hidden);
        let embedding_norm = view(&scratch.embedding_norm, w.hidden);
        let projected_embedding = view(&scratch.projected_embedding, w.hidden);
        let hidden_norm = view(&scratch.hidden_norm, w.wide);
        let projected_hidden = view(&scratch.projected_hidden, w.wide);
        let wide = view(&scratch.wide, w.wide);
        let hc_normalized = view(&scratch.hc_normalized, w.wide);
        let hc_low = view(&scratch.hc_low, config.hc_lowrank);
        let hc_up = view(&scratch.hc_up, w.wide);
        let hc_mixed = view(&scratch.hc_mixed, w.hidden);
        let index = view(&scratch.index, w.index_width);
        let qsa_k = view(&scratch.qsa_k, w.kv_width);
        let qsa_v = view(&scratch.qsa_v, w.kv_width);

        // Embed (`Step::Embed`): the Q8_0 table, one grid row per token.
        gpu.embedding_lookup_q8_batched(
            mw.embedding,
            &token_embedding,
            token_ids,
            rows,
            w.hidden,
        )?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &token_embedding,
                norm_weight: mw.norm_embedding,
                normalized: &embedding_norm,
                branches: 1,
                hidden: w.hidden,
                state_bf16: false,
            },
        )?;
        // G1 (byte identity vs the per-row `gemv_bf16_xf32`) governs every
        // `gemm_bf16_xf32_multirow` below. On gfx1151 `gemm.rs` selects by
        // (m, k, batch): (2560, 2560) is on the r16 allowlist for batch >= 64
        // (`r16w4t` / `r16w4` / `r16`, documented bitwise to the four-row
        // kernel) and the base four-row kernel below that; (10240, 320) is
        // not on the allowlist and K = 320 takes the `pto2` tile
        // (257..=512, LDS weights, also documented bitwise to the four-row
        // kernel). If G1 shows any difference for a pair, replace that call
        // with a rows-grid twin of the single-row kernel (`blockIdx.y`
        // token tile over `gemv_bf16_xf32_rows_body`) before tuning
        // anything. Never route these through `project_weight` /
        // `gemm_bf16_xf32_f16_wmma_qwen4` (F16 WMMA from 512 rows).
        gpu.gemm_bf16_xf32_multirow(
            mw.fc_embedding.buf,
            &embedding_norm,
            &projected_embedding,
            mw.fc_embedding.m,
            mw.fc_embedding.k,
            rows,
        )?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &backbone,
                norm_weight: mw.norm_hidden,
                normalized: &hidden_norm,
                branches: MTP_BRANCHES,
                hidden: w.hidden,
                state_bf16: false,
            },
        )?;
        // The four branch rows of a token are four projection rows
        // (per-row: `gemv_bf16_xf32_x4_rows` over rows = MTP_BRANCHES).
        gpu.gemm_bf16_xf32_multirow(
            mw.fc_hidden.buf,
            &hidden_norm,
            &projected_hidden,
            mw.fc_hidden.m,
            mw.fc_hidden.k,
            rows * MTP_BRANCHES,
        )?;
        gpu.broadcast_add_rows_f32(
            &projected_hidden,
            &projected_embedding,
            &wide,
            MTP_BRANCHES,
            w.hidden,
            rows,
        )?;

        // Attention-side HC read (`execute_hyper_read_inner`, rows = 1 route).
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &wide,
                norm_weight: mw.hc.norm,
                normalized: &hc_normalized,
                branches: config.hc_count,
                hidden: w.hidden,
                state_bf16: false,
            },
        )?;
        // The long-K down projection keeps the K4 quarter fold and applies the
        // 1/branches activation in its epilogue, as the per-row route does.
        gpu.gemv_bf16_xf32_k4_rows_tiled(
            mw.hc.input_mix_down.buf,
            &hc_normalized,
            &hc_low,
            mw.hc.input_mix_down.m,
            mw.hc.input_mix_down.k,
            Some(1.0 / config.hc_count as f32),
            rows,
        )?;
        gpu.gemm_bf16_xf32_multirow(
            mw.hc.input_mix_up.buf,
            &hc_low,
            &hc_up,
            mw.hc.input_mix_up.m,
            mw.hc.input_mix_up.k,
            rows,
        )?;
        hyper_read_projected(
            gpu,
            &HyperReadProjected {
                input: &wide,
                norm_weight: mw.hc.norm,
                up: &hc_up,
                normalized: &hc_normalized,
                mixed: &hc_mixed,
                branches: config.hc_count,
                hidden: w.hidden,
            },
        )?;

        // QSA append-only: index, K and V projections of the mixed row.
        gpu.gemv_q8_0_k2560_staged_rows_tiled(
            mw.qsa.indexer_qk.buf,
            &hc_mixed,
            &index,
            mw.qsa.indexer_qk.m,
            rows,
        )?;
        gpu.gemv_q8_0_k2560_staged_rows_tiled(mw.qsa.k.buf, &hc_mixed, &qsa_k, mw.qsa.k.m, rows)?;
        gpu.gemv_q8_0_k2560_staged_rows_tiled(mw.qsa.v.buf, &hc_mixed, &qsa_v, mw.qsa.v.m, rows)?;
        {
            let state = &self.state;
            indexed_attention_append_prologue(
                gpu,
                &IndexedAttentionAppendPrologue {
                    index_row: &index,
                    keys: &qsa_k,
                    values: &qsa_v,
                    full_keys: &state.full_keys,
                    full_values: &state.full_values,
                    raw_index_keys: &state.raw_index_keys,
                    index_q_norm: mw.qsa.indexer_q_norm,
                    k_norm: mw.qsa.k_norm,
                    index_heads: config.indexer_n_heads,
                    index_dim: config.indexer_head_dim,
                    index_kv_width: w.index_kv_width,
                    kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    position,
                    rows,
                    // The one MTP layer keeps the exact F32 QSA state.
                    format: QsaKvFormat::F32,
                },
            )?;
            // Pool exactly the blocks these rows complete, from raw keys the
            // prologue just wrote (a block straddling `end` waits for the next
            // call, whose first block is `position / compress`).
            let complete = end / compress;
            if complete > 0 {
                indexed_attention_pool_rope_incremental(
                    gpu,
                    &IndexedAttentionPoolRope {
                        raw_keys: &state.raw_index_keys,
                        pooled: &state.pooled_keys,
                        norm: Some(mw.qsa.indexer_k_norm),
                        block_count: complete,
                        compress,
                        index_dim: w.index_kv_width,
                        position: Some(QsaPositionBinding {
                            position_start: position,
                            rows,
                        }),
                        grid_bound: state.pooled_capacity,
                    },
                )?;
            }
        }

        let state = &mut self.state;
        state.position = end;
        state.full_len = end;
        state.raw_len = end;
        state.pooled_len = end / compress;
        state.selected_len = 0;
        state.step_index = state.step_index.wrapping_add(rows);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::state::Qwen4State;

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

    /// Non-owning F32/BF16 descriptor of `elements` for host-side bound
    /// checks; never dereferenced.
    fn descriptor(elements: usize, dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    std::ptr::null_mut::<std::ffi::c_void>(),
                    elements * dtype.size(),
                )
            },
            shape: vec![elements],
            dtype,
        }
    }

    /// One-token scratch census at canonical geometry (H 2560, W 10240,
    /// rank 320): `hc_up` is one W-wide F32 row, the 36-tensor total is
    /// 659,824 bytes, and the shrink saves (W×320 − W)×4 = 13,066,240 bytes
    /// (12.4609375 MiB) from the former 13,726,064-byte (13.0902 MiB) set.
    #[test]
    fn mtp_scratch_hc_up_is_one_row_census() {
        let config = compact_test_config();
        let shapes = MtpGpuScratch::shapes(&config).expect("canonical scratch shapes");
        let wide = MTP_BRANCHES * config.hidden_size;
        assert_eq!(shapes[11], (wide, DType::F32), "hc_up is rows x W F32");
        let bytes = MtpGpuScratch::device_bytes(&config).expect("scratch bytes");
        assert_eq!(bytes, 659_824);
        let former = 13_726_064usize;
        assert_eq!(former - bytes, 13_066_240);
        assert_eq!((former - bytes) as f64 / (1u64 << 20) as f64, 12.4609375);
    }

    /// Every subview the shared HC read takes from `up` must fit the
    /// scratch: a second row (either the F32 projection or the BF16 packing
    /// view), a rank wider than the row, or a non-F32 buffer is refused.
    #[test]
    fn hc_up_subview_larger_than_scratch_is_rejected() {
        let config = compact_test_config();
        let wide = MTP_BRANCHES * config.hidden_size;
        let rank = config.hc_lowrank;
        let up = descriptor(
            hc_up_elements(MTP_HC_ROWS, wide, rank).expect("one-row extent"),
            DType::F32,
        );
        require_hc_up(&up, MTP_HC_ROWS, wide, rank).expect("one-row read fits");
        assert!(invalid_contains(
            require_hc_up(&up, 2, wide, rank),
            "2 HC rows use"
        ));
        // BF16 packing view (`rows × rank` of the F32 tensor) wider than a row.
        assert!(invalid_contains(
            require_hc_up(&up, 1, wide, wide + 1),
            "1 HC rows use"
        ));
        let short = descriptor(wide - 1, DType::F32);
        assert!(invalid_contains(
            require_hc_up(&short, 1, wide, rank),
            "HC rows use"
        ));
        let bf16 = descriptor(2 * wide, DType::BF16);
        assert!(invalid_contains(
            require_hc_up(&bf16, 1, wide, rank),
            "HC rows use"
        ));
        // The prior W×rank allocation is far above every one-row subview.
        assert!(hc_up_elements(MTP_HC_ROWS, wide, rank).unwrap() < wide * rank);
    }

    /// Readback census of the allocated one-token scratch on the device.
    #[test]
    fn mtp_scratch_allocation_matches_census() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let config = compact_test_config();
        let scratch = MtpGpuScratch::new(&mut gpu, &config).expect("canonical MTP scratch");
        let wide = MTP_BRANCHES * config.hidden_size;
        assert_eq!(scratch.hc_up.numel(), wide);
        assert_eq!(scratch.hc_up.buf.size(), wide * 4);
        let shapes = MtpGpuScratch::shapes(&config).expect("shapes");
        let tensors = [
            &scratch.token_ids,
            &scratch.embedding_rot,
            &scratch.token_embedding,
            &scratch.embedding_norm,
            &scratch.hidden_norm,
            &scratch.projected_embedding,
            &scratch.projected_hidden,
            &scratch.wide,
            &scratch.backbone_hidden,
            &scratch.hc_normalized,
            &scratch.hc_low,
            &scratch.hc_up,
            &scratch.hc_mixed,
            &scratch.hc_gates,
            &scratch.hc_bf16,
            &scratch.rotation,
            &scratch.index,
            &scratch.q_and_gate,
            &scratch.qsa_k,
            &scratch.qsa_v,
            &scratch.qsa_output,
            &scratch.qsa_selected,
            &scratch.router_logits,
            &scratch.moe_x_rot,
            &scratch.moe_gate_up,
            &scratch.moe_gate,
            &scratch.moe_up,
            &scratch.moe_hidden,
            &scratch.moe_output,
            &scratch.moe_gate_batch,
            &scratch.moe_up_batch,
            &scratch.moe_rot_batch,
            &scratch.moe_topk_indices,
            &scratch.moe_topk_weights,
            &scratch.moe_down_expanded,
            &scratch.moe_scalar,
        ];
        let mut allocated = 0usize;
        for (tensor, (elements, dtype)) in tensors.iter().zip(shapes) {
            assert_eq!((tensor.numel(), tensor.dtype), (elements, dtype));
            assert!(tensor.buf.size() >= tensor.byte_size());
            allocated += tensor.byte_size();
        }
        assert_eq!(allocated, MtpGpuScratch::device_bytes(&config).unwrap());
        assert_eq!(allocated, 659_824);
        if let Some(error) = scratch.free_gpu(&mut gpu) {
            panic!("free MTP scratch: {error:?}");
        }
    }

    fn read_bytes(gpu: &mut Gpu, tensor: &GpuTensor, offset: usize, len: usize) -> Vec<u8> {
        let view = tensor.sub_offset(offset / tensor.dtype.size(), len / tensor.dtype.size());
        let mut bytes = vec![0u8; len];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &view.buf)
            .expect("download");
        bytes
    }

    /// Legacy: full allocation, ensure is only the admitted-capacity bound.
    #[test]
    fn legacy_mtp_state_ensure_is_a_bounds_check() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let config = compact_test_config();
        let max_seq = 4096;
        let mut state =
            MtpGpuState::new_with_backend(&mut gpu, &config, max_seq, Qwen4KvBackend::Legacy)
                .expect("legacy MTP state");
        let full = config
            .qsa_context_arena_bytes(max_seq, QsaKvFormat::F32)
            .unwrap();
        assert_eq!(state.mapped_tokens(), max_seq);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), full);
        state
            .ensure_mapped_capacity(&mut gpu, max_seq)
            .expect("within capacity");
        assert!(invalid_contains(
            state.ensure_mapped_capacity(&mut gpu, max_seq + 1),
            "admitted capacity"
        ));
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), full);
        if let Some(error) = state.free_gpu(&mut gpu) {
            panic!("free MTP state: {error:?}");
        }
    }

    /// VMM: nothing mapped at construction; each arena grows by its own
    /// stride (pooled by ceil(tokens / compress)) through the shared chunk
    /// plan, earlier rows survive growth, fresh pages read zero, and a
    /// request past the admitted capacity maps nothing.
    #[test]
    fn vmm_mtp_state_maps_on_demand() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        if !crate::kv_backend::qwen4_vmm_supported(&gpu) {
            eprintln!("skip: Qwen4 VMM unsupported on {}", gpu.arch);
            return;
        }
        let config = compact_test_config();
        let max_seq = 16384;
        let mut state =
            MtpGpuState::new_with_backend(&mut gpu, &config, max_seq, Qwen4KvBackend::Vmm)
                .expect("VMM MTP state");
        assert_eq!(state.backend(), Qwen4KvBackend::Vmm);
        assert_eq!(state.mapped_tokens(), 0);
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), 0);
        // Reset of an unmapped state touches no context page.
        state.reset(&mut gpu).expect("reset unmapped state");

        state
            .ensure_mapped_capacity(&mut gpu, 1)
            .expect("first token");
        let arenas = |state: &MtpGpuState| {
            [
                (&state.full_keys, state.full_capacity, 1usize),
                (&state.full_values, state.full_capacity, 1),
                (&state.raw_index_keys, state.raw_capacity, 1),
                (&state.pooled_keys, state.pooled_capacity, state.compress),
            ]
            .map(|(tensor, capacity, per_row)| {
                let plan = KvChunkPlan::new(
                    tensor.byte_size() / capacity,
                    capacity,
                    DEFAULT_KV_CHUNK_TOKENS,
                    gpu.vmm_granularity(tensor).expect("VMM granularity"),
                    DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
                )
                .expect("chunk plan");
                (plan, per_row, gpu.vmm_mapped_bytes(tensor).expect("mapped"))
            })
        };
        let first = arenas(&state);
        let mut sum = 0;
        let mut coverage = usize::MAX;
        for (plan, per_row, mapped) in first {
            let rows = 1usize.div_ceil(per_row);
            assert_eq!(mapped, plan.mapped_bytes_for_tokens(rows).unwrap());
            sum += mapped;
            coverage = coverage.min(plan.token_capacity(mapped) * per_row);
        }
        assert_eq!(state.mapped_tokens(), coverage.min(max_seq));
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), sum);
        let full = config
            .qsa_context_arena_bytes(max_seq, QsaKvFormat::F32)
            .unwrap();
        assert!(
            sum < full,
            "first map {sum} must stay below the full {full}"
        );

        // A growth boundary: pattern the full-key rows below it, grow past it.
        let boundary = state.mapped_tokens();
        assert!(boundary < max_seq);
        let row = state.full_keys.byte_size() / state.full_capacity;
        let pattern: Vec<u8> = (0..boundary * row).map(|i| (i % 251) as u8 + 1).collect();
        gpu.hip
            .memcpy_htod(&state.full_keys.buf, &pattern)
            .expect("pattern mapped rows");
        let unchanged = state.mapped_context_bytes(&gpu).unwrap();
        assert!(invalid_contains(
            state.ensure_mapped_capacity(&mut gpu, max_seq + 1),
            "admitted capacity"
        ));
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), unchanged);
        state
            .ensure_mapped_capacity(&mut gpu, boundary)
            .expect("no growth");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), unchanged);
        state
            .ensure_mapped_capacity(&mut gpu, boundary + 1)
            .expect("grow");
        assert!(state.mapped_tokens() > boundary);
        assert!(state.mapped_context_bytes(&gpu).unwrap() > unchanged);
        assert_eq!(
            read_bytes(&mut gpu, &state.full_keys, 0, pattern.len()),
            pattern
        );
        let fresh = read_bytes(&mut gpu, &state.full_keys, boundary * row, row);
        assert!(fresh.iter().all(|&byte| byte == 0), "fresh page is zeroed");

        // Reset zeroes the mapped prefix and keeps the mapping.
        let mapped = state.mapped_context_bytes(&gpu).unwrap();
        state.reset(&mut gpu).expect("reset mapped state");
        assert_eq!(state.mapped_context_bytes(&gpu).unwrap(), mapped);
        let cleared = read_bytes(&mut gpu, &state.full_keys, 0, pattern.len());
        assert!(cleared.iter().all(|&byte| byte == 0));

        state
            .ensure_mapped_capacity(&mut gpu, max_seq)
            .expect("full map");
        assert_eq!(state.mapped_tokens(), max_seq);
        let mapped_full = state.mapped_context_bytes(&gpu).unwrap();
        assert!(mapped_full >= full);
        if let Some(error) = state.free_gpu(&mut gpu) {
            panic!("free MTP state: {error:?}");
        }
        assert_eq!(gpu.vmm_allocation_count(), 0, "VMM owners released");
    }
}

// Keep the grammar import in this module's public dependency closure while the
// full acceptance adapter is assembled in `mtp_spec.rs`.
fn _grammar_marker(_: Option<&mut dyn SpecGrammar>) {}
