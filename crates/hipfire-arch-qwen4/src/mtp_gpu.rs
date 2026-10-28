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
    dense_ref, hyper_desc, hyper_read_desc, program_dims, qsa_desc, seal_moe_decode, MtpEpGather,
    Qwen4EpDummyExpert, Qwen4GpuForwardError, Qwen4MoeLayerRuntime, Qwen4MoeScratch,
};
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::program::{
    Qwen4FfnDims, Qwen4HyperReadWeights, Qwen4HyperWriteWeights, Qwen4QsaWeights,
    QWEN4_QSA_KERNEL_ROPE_THETA,
};
use crate::state::{meta_bytes, meta_words, whole_part};
use crate::weights::{Qwen4EpPlacement, Qwen4FfnWeights, Qwen4Weights, WeightError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::MoeRowGather;
use hipfire_dispatch::pipeline::gated_mlp::{
    Activation, GatedMlpOp, GatedMlpScratch, MlpLadder, MlpOutput, MlpStorage,
};
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, BroadcastAddOp, ClearOp, DraftHead, DraftHeadLayout,
    DraftHeadPolicy, DraftHeadRequestState, EmbeddingOp, HyperNormOp, HyperReadOp, HyperWriteOp,
    IndexedAttentionMode, IndexedAttentionOp, IndexedAttentionState, MixerPhase, ProjectLadder,
    ProjectOp, Step,
};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::kv_backend::{
    KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
};
use hipfire_runtime::session_cache::{RowStream, StateLayout};
use hipfire_runtime::spec::SpecGrammar;
use rdna_compute::tensor_ops::{
    hc_activation_fused_f32, hyper_norm, hyper_read_projected, indexed_attention_append_prologue,
    indexed_attention_pool_rope_incremental, HcActivationFused, HyperNorm, HyperReadProjected,
    IndexedAttentionAppendPrologue, IndexedAttentionPoolRope, QsaKvFormat, QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;
use std::fmt;

/// HC branches of the Flash-Next batched Append route
/// (`Qwen4MtpGpu::append_rows`); every other MTP path takes `config.hc_count`.
const MTP_APPEND_BRANCHES: usize = 4;
/// Rows of every MTP HC read and write: one token.
const MTP_HC_ROWS: usize = 1;
/// Metadata words of an MTP head session snapshot: the boundary-row word
/// (the boundary position plus one, or 0) and the state mark.
const MTP_SESSION_WORDS: usize = 7;
/// Prompt rows one batched Append pass (`Qwen4MtpGpu::append_rows`) runs per
/// launch sequence: the capacity of [`MtpAppendScratch`].
pub(crate) const MTP_FILL_ROWS: usize = 1024;

/// `HIPFIRE_QWEN4_MTP_APPEND_DET`: the DET route of the batched prompt-fill
/// Append ([`MtpFillContext::PromptFill`] only), on by default on exact
/// gfx1151; `0` is the kill switch.
pub(crate) const MTP_APPEND_DET_ENV: &str = "HIPFIRE_QWEN4_MTP_APPEND_DET";
/// Default of [`MTP_APPEND_DET_ENV`] when it is unset (or neither `0` nor
/// `1`): requested, so exact gfx1151 takes DET and every other arch keeps the
/// incumbent kernels ([`append_det_active`]).
const MTP_APPEND_DET_DEFAULT: bool = true;

/// Which caller a batched Append pass (`Qwen4MtpGpu::append_rows`) serves. It
/// is stated by every caller and never inferred from the hidden row, the
/// scratch or any option: only [`Self::PromptFill`] can take the DET route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MtpFillContext {
    /// `mtp_prefill` / lane prompt tiles: the head consumes prompt rows after
    /// the trunk's chunked forward.
    PromptFill,
    /// The decode takeover step's head fill of the accepted window rows. Always
    /// the incumbent kernels, whatever `HIPFIRE_QWEN4_MTP_APPEND_DET` says.
    DecodeFill,
}

/// `HIPFIRE_QWEN4_MTP_APPEND_DET` as requested: on unless `0`
/// ([`MTP_APPEND_DET_DEFAULT`]); read at every prompt fill. See
/// [`mtp_append_det_enabled`] for the effective gate.
pub(crate) fn mtp_append_det_requested() -> bool {
    hipfire_config::developer_bool(MTP_APPEND_DET_ENV, MTP_APPEND_DET_DEFAULT)
}

/// The DET gate from its two inputs: requested and exact gfx1151 (gfx1150,
/// gfx1152, gfx1201 and every other arch never take it).
fn append_det_active(requested: bool, exact_gfx1151: bool) -> bool {
    requested && exact_gfx1151
}

/// Whether DET (`HIPFIRE_QWEN4_MTP_APPEND_DET`, default on) is in force on this GPU: exact
/// gfx1151 only. This is the machine-level gate; whether one Append pass takes
/// the DET route also depends on its [`MtpFillContext`] ([`append_rows_det`]).
pub(crate) fn mtp_append_det_enabled(gpu: &Gpu) -> bool {
    append_det_active(mtp_append_det_requested(), gpu.arch_caps.is_gfx1151())
}

/// Whether an Append pass of `context` takes the DET route when the machine
/// gate `det_enabled` is on: prompt fill only.
fn append_rows_det(context: MtpFillContext, det_enabled: bool) -> bool {
    det_enabled && context == MtpFillContext::PromptFill
}

/// Fewest rows one `append_rows` pass takes: the incumbent multirow
/// projections need two rows or more, the DET kernels take any `n >= 1`.
fn append_rows_min(det: bool) -> usize {
    if det {
        1
    } else {
        2
    }
}

/// Whether a prompt-fill sub-chunk of `rows` rows runs through
/// `append_rows` (and so its scratch): always under DET, where a lone row
/// takes the same kernels as every other row; otherwise only from two rows,
/// a lone row staying the per-row `append_token`.
pub(crate) fn prompt_fill_uses_rows(det: bool, rows: usize) -> bool {
    rows >= append_rows_min(det)
}

/// The DET dense kernel's shape limits (`gemm_bf16_xf32_mtp_dense_det_pm`):
/// `M % 16 == 0`, `K % 64 == 0`, `N >= 1`.
fn det_dense_shape_ok(m: usize, k: usize, n: usize) -> bool {
    m != 0 && m % 16 == 0 && k != 0 && k % 64 == 0 && n >= 1
}
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
    /// BF16 scratch of the indexed attention.
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
    ffn_output: GpuTensor,
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
        let wide = config
            .hc_count
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
        // FFN extents come from the typed dims: a MoE profile's routed and
        // shared widths, or a dense profile's one intermediate width (MoE
        // extents are zero for dense, dense for MoE; every buffer is then
        // raised to one element below).
        let (num_experts, top_k, moe_intermediate, shared_intermediate, dense_intermediate) =
            match program_dims(config).ffn {
                Qwen4FfnDims::Moe {
                    num_experts,
                    experts_per_token,
                    intermediate,
                    shared_intermediate,
                } => (
                    num_experts,
                    experts_per_token,
                    intermediate,
                    shared_intermediate,
                    0,
                ),
                Qwen4FfnDims::Dense { intermediate } => (0, 0, 0, 0, intermediate),
            };
        let max_rotation = wide
            .max(hidden)
            .max(config.hc_lowrank)
            .max(q_width)
            .max(dense_intermediate);
        let routed = top_k * moe_intermediate;
        // One-token gate/up rows: routed slots * intermediate for MoE (the
        // per-token batch scratch), the dense intermediate otherwise.
        let ffn_batch = routed.max(dense_intermediate);
        let mut shapes = [
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
            (num_experts, DType::F32),
            (hidden, DType::F32),
            (2 * moe_intermediate, DType::F32),
            (moe_intermediate, DType::F32),
            (moe_intermediate, DType::F32),
            (moe_intermediate, DType::F32),
            (hidden, DType::F32),
            (ffn_batch, DType::F32),
            (ffn_batch, DType::F32),
            (routed, DType::F32),
            (top_k, DType::F32),
            (top_k, DType::F32),
            (top_k * hidden + hidden.div_ceil(4), DType::F32),
            (shared_intermediate, DType::F32),
        ];
        for shape in &mut shapes {
            // Every buffer holds at least one element.
            shape.0 = shape.0.max(1);
        }
        Ok(shapes)
    }

    /// Device bytes [`Self::new`] allocates.
    fn device_bytes(config: &Qwen4Config) -> Result<usize, MtpGpuError> {
        Ok(Self::shapes(config)?
            .iter()
            .map(|&(elements, dtype)| elements * dtype.size())
            .sum())
    }

    /// [`Self::device_bytes`] per allocation of [`Self::new`], in field
    /// order; `None` on overflow.
    pub(crate) fn allocation_sizes(config: &Qwen4Config) -> Option<Vec<u64>> {
        Self::shapes(config)
            .ok()?
            .iter()
            .map(|&(elements, dtype)| u64::try_from(elements.checked_mul(dtype.size())?).ok())
            .collect()
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
            ffn_output: next(),
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
            self.ffn_output,
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
    /// `hc_count * hidden`: one row of the HC streams.
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
            wide: config.hc_count.checked_mul(config.hidden_size)?,
            index_width: config
                .indexer_n_heads
                .checked_mul(config.indexer_head_dim)?
                .checked_add(index_kv_width)?,
            index_kv_width,
            kv_width: config.num_key_value_heads.checked_mul(config.head_dim)?,
        })
    }
}

/// Allocations of one [`MtpAppendScratch`], in field order.
const MTP_APPEND_TENSORS: usize = 15;

/// Row-batched operator buffers of the prompt-fill Append pass
/// (`Qwen4MtpGpu::append_rows`): the per-row [`MtpGpuScratch`] tensors that an
/// Append step touches, each with `rows` rows. Never aliases the per-row scratch.
///
/// Also owns the prompt chunk's token ids ([`Self::upload_prompt_ids`]): the
/// whole target chunk uploads once, and each sub-chunk's `append_rows` reads
/// its rows in place through [`Self::prompt_ids_view`].
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
    /// `id_rows` i32 token ids of one prompt chunk, as raw bytes.
    prompt_ids: GpuTensor,
    /// Token-id row capacity of `prompt_ids`.
    id_rows: usize,
    host_token_bytes: Vec<u8>,
    host_prompt_ids: Vec<u8>,
}

impl MtpAppendScratch {
    /// Row capacity: the most rows one `append_rows` call takes.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Token-id row capacity of [`Self::upload_prompt_ids`].
    pub(crate) fn prompt_id_rows(&self) -> usize {
        self.id_rows
    }

    /// Element count and dtype of each scratch tensor, in field order.
    fn shapes(
        config: &Qwen4Config,
        rows: usize,
        id_rows: usize,
    ) -> Option<[(usize, DType); MTP_APPEND_TENSORS]> {
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
            (id_rows.checked_mul(std::mem::size_of::<i32>())?, DType::Raw),
        ])
    }

    /// Device bytes [`Self::new`] allocates for `rows` rows and `id_rows`
    /// prompt-chunk token ids.
    pub(crate) fn device_bytes(config: &Qwen4Config, rows: usize, id_rows: usize) -> Option<usize> {
        Self::shapes(config, rows, id_rows)?
            .iter()
            .try_fold(0usize, |total, &(elements, dtype)| {
                total.checked_add(elements.checked_mul(dtype.size())?)
            })
    }

    /// [`Self::device_bytes`] per allocation of [`Self::new`], in field
    /// order; `None` on overflow.
    pub(crate) fn allocation_sizes(
        config: &Qwen4Config,
        rows: usize,
        id_rows: usize,
    ) -> Option<Vec<u64>> {
        Self::shapes(config, rows, id_rows)?
            .iter()
            .map(|&(elements, dtype)| u64::try_from(elements.checked_mul(dtype.size())?).ok())
            .collect()
    }

    pub(crate) fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        rows: usize,
        id_rows: usize,
    ) -> Result<Self, MtpGpuError> {
        if rows == 0 || id_rows == 0 {
            return Err(invalid("MTP append scratch needs at least one row"));
        }
        let shapes = Self::shapes(config, rows, id_rows)
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
            prompt_ids: next(),
            id_rows,
            host_token_bytes: vec![0; rows * std::mem::size_of::<i32>()],
            host_prompt_ids: vec![0; id_rows * std::mem::size_of::<i32>()],
        })
    }

    /// Upload `tokens` (at most the id-row capacity) as the prompt chunk's i32
    /// ids: the same bytes `append_rows` would upload per sub-chunk. The copy
    /// is a blocking `hipMemcpy`, so call it where the device is already idle
    /// (after the target forward's readback) and after every earlier
    /// `append_rows` that reads the previous chunk's ids has been enqueued.
    pub(crate) fn upload_prompt_ids(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<(), MtpGpuError> {
        if tokens.is_empty() || tokens.len() > self.id_rows {
            return Err(invalid(format!(
                "MTP prompt ids of {} rows need 1..={}",
                tokens.len(),
                self.id_rows
            )));
        }
        for (bytes, &token) in self.host_prompt_ids.chunks_exact_mut(4).zip(tokens) {
            bytes.copy_from_slice(&(token as i32).to_ne_bytes());
        }
        gpu.memcpy_htod_auto(
            &self.prompt_ids.buf,
            &self.host_prompt_ids[..tokens.len() * std::mem::size_of::<i32>()],
        )?;
        Ok(())
    }

    /// Non-owning view of `rows` prompt-chunk ids from row `first` of the
    /// last [`Self::upload_prompt_ids`] upload, for `append_rows`'s
    /// `device_ids`. The view must not be freed.
    pub(crate) fn prompt_ids_view(
        &self,
        first: usize,
        rows: usize,
    ) -> Result<GpuTensor, MtpGpuError> {
        match first.checked_add(rows) {
            Some(end) if end <= self.id_rows => Ok(self.prompt_ids.sub_offset(
                first * std::mem::size_of::<i32>(),
                rows * std::mem::size_of::<i32>(),
            )),
            _ => Err(invalid("MTP prompt id view is outside the id capacity")),
        }
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
            self.prompt_ids,
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

/// The bound native MTP head; a profile without one has no MTP executor.
fn bound_mtp(weights: &Qwen4Weights) -> Result<&crate::weights::Qwen4MtpWeights, MtpGpuError> {
    weights
        .mtp
        .as_ref()
        .ok_or_else(|| invalid("profile has no native MTP head"))
}

impl<'a> AppendWeights<'a> {
    fn resolve(
        weights: &'a Qwen4Weights,
        config: &Qwen4Config,
        w: &AppendWidths,
    ) -> Result<Self, MtpGpuError> {
        if !config.profile.is_flash_next() {
            return Err(invalid(format!(
                "MTP batched append is the Flash-Next route (K = 2560 staged Q8_0 \
                 projections, {MTP_APPEND_BRANCHES} HC branches); profile {} takes the per-row route",
                config.profile.id()
            )));
        }
        if config.hc_count != MTP_APPEND_BRANCHES {
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
        let mtp = bound_mtp(weights)?;
        let fc_embedding = dense_ref(weights, &mtp.fc_embedding)?;
        check(
            "fc_embedding",
            &fc_embedding,
            DType::BF16,
            w.hidden,
            w.hidden,
        )?;
        let fc_hidden = dense_ref(weights, &mtp.fc_hidden)?;
        check("fc_hidden", &fc_hidden, DType::BF16, w.hidden, w.hidden)?;
        let hc = hyper_desc(weights, &mtp.attn_hyper)?.read;
        if !w.wide.is_multiple_of(32) {
            return Err(invalid("MTP batched append HC down needs K % 32 == 0"));
        }
        check(
            "HC down",
            &hc.input_mix_down,
            DType::BF16,
            config.hc_lowrank,
            w.wide,
        )?;
        check(
            "HC up",
            &hc.input_mix_up,
            DType::BF16,
            w.wide,
            config.hc_lowrank,
        )?;
        let qsa = qsa_desc(weights, &mtp.attention)?;
        // The tiled Q8_0 projection is the K = 2560 staged kernel.
        if w.hidden != 2560 {
            return Err(invalid(
                "MTP batched append Q8_0 projections need hidden 2560",
            ));
        }
        check(
            "indexer_qk",
            &qsa.indexer_qk,
            DType::Q8_0,
            w.index_width,
            w.hidden,
        )?;
        check("k_proj", &qsa.k, DType::Q8_0, w.kv_width, w.hidden)?;
        check("v_proj", &qsa.v, DType::Q8_0, w.kv_width, w.hidden)?;
        Ok(Self {
            embedding,
            norm_embedding: weights.resident(&mtp.pre_fc_norm_embedding)?,
            norm_hidden: weights.resident(&mtp.pre_fc_norm_hidden)?,
            fc_embedding,
            fc_hidden,
            hc,
            qsa,
        })
    }
}

/// One BF16 dense projection of `append_rows` over `rows` rows (`weight` is
/// `[m, k]`, `x` F32 `[rows, k]`, `y` F32 `[rows, m]`): the incumbent
/// `gemm_bf16_xf32_multirow` (G1), or with `det` the fixed-row F16 WMMA
/// `gemm_bf16_xf32_mtp_dense_det_pm`.
fn append_dense_projection(
    gpu: &mut Gpu,
    det: bool,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
) -> Result<(), MtpGpuError> {
    if det {
        gpu.gemm_bf16_xf32_mtp_dense_det_pm(weight.buf, x, y, weight.m, weight.k, rows)?;
    } else {
        gpu.gemm_bf16_xf32_multirow(weight.buf, x, y, weight.m, weight.k, rows)?;
    }
    Ok(())
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
    /// The entry stays active after a committed window so the window can
    /// still be rolled back (`retain_validated`).
    retained: bool,
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
            retained: false,
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
        self.retained = false;
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

/// Word 0 of an MTP session capture at `position`: the position plus one when
/// the boundary row was saved for exactly that position, else 0 (the row is
/// stale and a restore refuses the snapshot).
fn boundary_word(saved: Option<usize>, position: usize) -> u64 {
    match saved {
        Some(saved) if saved == position => position as u64 + 1,
        _ => 0,
    }
}

/// Whether word 0 of a snapshot at `position` names a boundary row saved for
/// that position.
fn boundary_word_names(word: u64, position: usize) -> bool {
    word == position as u64 + 1
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
    /// The target's boundary hidden (row `position - 1`) kept for the
    /// session snapshot; no snapshot backup of its own.
    boundary_hidden: GpuTensor,
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
    /// Position `boundary_hidden` was saved for ([`Self::save_boundary_hidden`]),
    /// or restored with at a session restore.
    boundary_saved: Option<usize>,
    /// Set by a session restore whose snapshot held a boundary row for its
    /// position, until [`Self::restore_boundary_hidden`] takes it.
    boundary_restored: Option<usize>,
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
    /// snapshot backups, and the one target boundary-hidden row.
    pub(crate) fn device_bytes(
        config: &Qwen4Config,
        context: &Qwen4ContextCommit,
    ) -> Option<usize> {
        let row = config
            .hc_count
            .checked_mul(config.hidden_size)?
            .checked_mul(std::mem::size_of::<f32>())?;
        let carry = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?
            .checked_add(std::mem::size_of::<i32>())?
            .checked_add(row)?;
        context
            .committed_layer_bytes(config, QsaKvFormat::F32)?
            .checked_add(carry.checked_mul(2)?)?
            .checked_add(row)
    }

    /// [`Self::device_bytes`] per allocation of [`Self::new_with_backend`]
    /// and [`MtpGpuStateSnapshotArena::new`], in that order: the four context
    /// arenas as `context` commits them (legacy: whole; VMM: each arena's
    /// mapped pages, pooled keys by `ceil(tokens / compress)` rows), the
    /// selection, its length word and the wide hidden carry, the target
    /// boundary-hidden row, then the snapshot backups of the selection, the
    /// length word and the wide hidden. Zero-byte entries are omitted.
    pub(crate) fn allocation_sizes(
        config: &Qwen4Config,
        context: &Qwen4ContextCommit,
    ) -> Option<Vec<u64>> {
        let compress = config.indexer_compress_ratio;
        if compress == 0 {
            return None;
        }
        let f32_bytes = std::mem::size_of::<f32>();
        let row = config
            .hc_count
            .checked_mul(config.hidden_size)?
            .checked_mul(f32_bytes)?;
        let selected = config
            .qsa_selected_capacity()
            .checked_mul(std::mem::size_of::<i32>())?;
        let length = std::mem::size_of::<i32>();
        let kv_row = QsaKvFormat::F32.kv_row_bytes(config.num_key_value_heads, config.head_dim);
        let index_row = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)?
            .checked_mul(QsaKvFormat::F32.index_dtype().size())?;
        let pooled_rows = context.max_seq.div_ceil(compress);
        let pooled_tokens = context.tokens.div_ceil(compress);
        let arena = |row_bytes: usize, rows: usize, tokens: usize| -> Option<usize> {
            match context.backend {
                Qwen4KvBackend::Legacy => row_bytes.checked_mul(rows),
                Qwen4KvBackend::Vmm => crate::state::vmm_context_arena_mapped_bytes(
                    row_bytes,
                    rows,
                    tokens,
                    context.granularity,
                ),
            }
        };
        let sizes = [
            arena(kv_row, context.max_seq, context.tokens)?,
            arena(kv_row, context.max_seq, context.tokens)?,
            arena(index_row, context.max_seq, context.tokens)?,
            arena(index_row, pooled_rows, pooled_tokens)?,
            selected,
            length,
            row,
            row,
            selected,
            length,
            row,
        ];
        sizes
            .into_iter()
            .filter(|&bytes| bytes > 0)
            .map(|bytes| u64::try_from(bytes).ok())
            .collect()
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
            (config.hc_count * config.hidden_size, DType::F32, false),
            (config.hc_count * config.hidden_size, DType::F32, false),
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
        let (selected_indices, selected_len_out, wide_hidden, boundary_hidden) =
            (next(), next(), next(), next());
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
                    boundary_hidden,
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
            boundary_hidden,
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
            boundary_saved: None,
            boundary_restored: None,
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

    /// Admitted token capacity of the context arenas (the bound
    /// [`Self::ensure_mapped_capacity`] refuses above).
    pub(crate) fn context_capacity(&self) -> usize {
        self.full_capacity
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
        self.reset_inner(gpu, true)
    }

    /// Cold reset of a side request: as [`Self::reset`] except the
    /// append-only row streams (full K/V, raw index keys, pooled keys) keep
    /// their bytes.
    pub(crate) fn reset_for_side(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.reset_inner(gpu, false)
    }

    /// Per-stream row ends of a side request writing tokens `[0, through)`,
    /// in [`Self::session_layout`] stream order: full keys, full values, raw
    /// index keys, pooled keys.
    pub(crate) fn side_request_rows(&self, through: usize) -> Vec<usize> {
        vec![through, through, through, through.div_ceil(self.compress)]
    }

    fn reset_inner(&mut self, gpu: &mut Gpu, clear_rows: bool) -> Result<(), MtpGpuError> {
        self.snapshot_arena.invalidate();
        for (index, tensor) in [
            &self.full_keys,
            &self.full_values,
            &self.raw_index_keys,
            &self.pooled_keys,
            &self.selected_indices,
            &self.selected_len_out,
            &self.wide_hidden,
            &self.boundary_hidden,
        ]
        .into_iter()
        .enumerate()
        {
            // The first four are the append-only row streams.
            if !clear_rows && index < 4 {
                continue;
            }
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
        self.boundary_saved = None;
        self.boundary_restored = None;
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// A snapshot ticket of this state's arena is outstanding.
    pub(crate) fn snapshot_active(&self) -> bool {
        self.snapshot_arena.active && !self.snapshot_arena.retained
    }

    pub(crate) fn snapshot(&mut self, gpu: &mut Gpu) -> Result<MtpGpuStateSnapshot, MtpGpuError> {
        // A retained entry belongs to a window already committed: the next
        // snapshot supersedes it (its tickets stop validating).
        self.release_retained();
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

    /// Keep a validated ticket's arena entry active after a committed
    /// window so a terminal stop inside it can still restore the head. The
    /// next [`Self::snapshot`], any reset or a session restore drops it.
    pub(crate) fn retain_validated(&mut self, _snapshot: MtpGpuStateSnapshot) {
        debug_assert!(
            self.snapshot_arena.active,
            "MTP retain_validated requires an active snapshot"
        );
        self.snapshot_arena.retained = true;
    }

    /// Drop the retained arena entry, if any (a no-op otherwise).
    pub(crate) fn release_retained(&mut self) {
        if self.snapshot_arena.retained {
            self.snapshot_arena.invalidate();
        }
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
    /// then selection, device selected length, wide hidden and the target
    /// boundary hidden.
    fn session_layout(&self, mark: &MtpStateMark) -> StateLayout<'_> {
        StateLayout {
            fixed: vec![
                whole_part(&self.selected_indices),
                whole_part(&self.selected_len_out),
                whole_part(&self.wide_hidden),
                whole_part(&self.boundary_hidden),
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
    /// device ranges of the live state. Word 0 is the boundary row's
    /// position plus one when [`Self::save_boundary_hidden`] saved it for
    /// exactly this position, else 0 (the row is then stale). A capture
    /// without the row is allowed (digest contexts read it only), but such a
    /// snapshot is refused by every restore ([`Self::parse_session_mark`]).
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
        let boundary = boundary_word(self.boundary_saved, mark.position);
        Ok((
            [boundary].into_iter().chain(words).collect(),
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
        // Word 0 is the boundary row's position plus one: a snapshot whose
        // row was not saved for its own position cannot be restored (the
        // row would be stale), so the cache entry is refused.
        if !boundary_word_names(words[0], mark.position) {
            return Err(invalid(format!(
                "MTP session snapshot at {} holds no boundary hidden for its position",
                mark.position
            )));
        }
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
        let required = mark.position.max(mark.full_len).max(mark.raw_len).max(
            mark.pooled_len
                .saturating_mul(self.compress)
                .min(self.full_capacity),
        );
        self.ensure_mapped_capacity(gpu, required)?;
        self.snapshot_arena.invalidate();
        self.request_epoch = self.request_epoch.wrapping_add(1);
        self.generation = self.generation.wrapping_add(1);
        // The boundary row is overwritten by the copy this layout drives.
        self.boundary_saved = None;
        self.boundary_restored = None;
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
        // `parse_session_mark` proved word 0 names this position: the row the
        // restore just copied in is the boundary hidden for it.
        self.boundary_saved = Some(mark.position);
        self.boundary_restored = Some(mark.position);
        Ok(())
    }

    /// Keep the target's boundary hidden for the position the head is at: the
    /// target hidden of row `position - 1`, which the caller holds in
    /// `pending` right after a prompt chunk's append and before the session
    /// capture at this boundary. Written into the head's own row so the
    /// capture carries it; a live `pending` is only read, never changed.
    pub(crate) fn save_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), MtpGpuError> {
        self.boundary_row_matches(pending)?;
        self.boundary_saved = None;
        gpu.copy_d2d(
            pending,
            &self.boundary_hidden,
            self.boundary_hidden.byte_size(),
        )?;
        self.boundary_saved = Some(self.position);
        Ok(())
    }

    /// After an actual session restore (never a live continuation, which
    /// leaves `pending` as the live hidden), copy the restored boundary row
    /// into `pending` and report `true` (a restore refuses any snapshot
    /// without a boundary row for its position, so a restored head always
    /// holds one). `false`: no restore since, or the row was already taken;
    /// `pending` is then untouched.
    pub(crate) fn restore_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<bool, MtpGpuError> {
        if self.boundary_restored != Some(self.position) {
            return Ok(false);
        }
        self.boundary_row_matches(pending)?;
        gpu.copy_d2d(
            &self.boundary_hidden,
            pending,
            self.boundary_hidden.byte_size(),
        )?;
        self.boundary_restored = None;
        Ok(true)
    }

    fn boundary_row_matches(&self, pending: &GpuTensor) -> Result<(), MtpGpuError> {
        if pending.dtype != DType::F32 || pending.numel() != self.boundary_hidden.numel() {
            return Err(invalid(format!(
                "MTP boundary hidden needs one F32 row of {} elements",
                self.boundary_hidden.numel()
            )));
        }
        Ok(())
    }

    /// Context-arena bytes still unmapped below `through` tokens (at most
    /// capacity; 0 for legacy), row-exact per arena like the target's
    /// ([`crate::state::context_arena_bytes_through`]).
    fn context_growth_bytes(&self, through: usize) -> u64 {
        if self.backend == Qwen4KvBackend::Legacy || self.full_capacity == 0 {
            return 0;
        }
        let through = through.min(self.full_capacity);
        [
            (&self.full_keys, self.full_capacity, 1),
            (&self.full_values, self.full_capacity, 1),
            (&self.raw_index_keys, self.raw_capacity, 1),
            (&self.pooled_keys, self.pooled_capacity, self.compress),
        ]
        .into_iter()
        .map(|(tensor, capacity, tokens_per_row)| {
            let bytes = |tokens| {
                crate::state::context_arena_bytes_through(
                    tensor,
                    capacity,
                    tokens_per_row,
                    false,
                    tokens,
                )
            };
            bytes(through).saturating_sub(bytes(self.mapped_tokens))
        })
        .sum()
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
            boundary_hidden,
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
            boundary_hidden,
        ] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// Every byte family of a head state: full K/V and raw/pooled index rows
/// below their lengths, the selected indices up to `selected_len`, the
/// device selected length, the wide hidden, and the marks, position and step
/// index as little-endian u64s.
#[cfg(feature = "lab")]
pub fn lab_head_items(gpu: &Gpu, head: &MtpGpuState) -> Result<Vec<(String, Vec<u8>)>, String> {
    gpu.hip
        .device_synchronize()
        .map_err(|error| format!("Qwen4 MTP lab head sync: {error}"))?;
    let download = |tensor: &GpuTensor, bytes: usize| -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; bytes];
        if bytes > 0 {
            if bytes > tensor.buf.size() {
                return Err(format!(
                    "Qwen4 MTP lab head read of {bytes} bytes exceeds the {} mapped",
                    tensor.buf.size()
                ));
            }
            gpu.hip
                .memcpy_dtoh(&mut out, &tensor.buf)
                .map_err(|error| format!("Qwen4 MTP lab head download: {error}"))?;
        }
        Ok(out)
    };
    let rows = |tensor: &GpuTensor, capacity: usize, len: usize| -> Result<Vec<u8>, String> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let row_bytes = tensor
            .byte_size()
            .checked_div(capacity)
            .ok_or_else(|| "Qwen4 MTP lab head arena has no capacity".to_string())?;
        download(tensor, len * row_bytes)
    };
    let words = [
        head.full_len,
        head.raw_len,
        head.pooled_len,
        head.selected_len,
        head.position,
        head.step_index,
    ]
    .iter()
    .flat_map(|&word| (word as u64).to_le_bytes())
    .collect::<Vec<u8>>();
    Ok(vec![
        (
            "head.full_keys".to_string(),
            rows(&head.full_keys, head.full_capacity, head.full_len)?,
        ),
        (
            "head.full_values".to_string(),
            rows(&head.full_values, head.full_capacity, head.full_len)?,
        ),
        (
            "head.raw_index_keys".to_string(),
            rows(&head.raw_index_keys, head.raw_capacity, head.raw_len)?,
        ),
        (
            "head.pooled_keys".to_string(),
            rows(&head.pooled_keys, head.pooled_capacity, head.pooled_len)?,
        ),
        (
            "head.selected_indices".to_string(),
            download(
                &head.selected_indices,
                head.selected_len * std::mem::size_of::<i32>(),
            )?,
        ),
        (
            "head.selected_len_out".to_string(),
            download(&head.selected_len_out, head.selected_len_out.byte_size())?,
        ),
        (
            "head.wide_hidden".to_string(),
            download(&head.wide_hidden, head.wide_hidden.byte_size())?,
        ),
        ("head.marks".to_string(), words),
    ])
}

/// Model-owned native MTP operator and state.
pub struct Qwen4MtpGpu {
    pub(crate) scratch: MtpGpuScratch,
    pub(crate) state: MtpGpuState,
    /// The MTP layer's routed-expert runtime: `Some` exactly for a MoE FFN.
    pub(crate) moe: Option<Qwen4MoeLayerRuntime>,
    pub(crate) max_seq: usize,
    /// Draft ranking (`HIPFIRE_MTP_DRAFT_HEAD=mq2..mq6[r]`, default `mq2r`;
    /// anything else = the model's own head). On the shipped Q8_0 head an
    /// MQ2 copy with exact re-scoring of its top 8 drafts the Q8_0 head's own
    /// argmax at a quarter of its read; plain MQ3 loses acceptance.
    pub(crate) draft: DraftHead,
    /// The head's MoE runtime is an EP decode table (the rank-compact
    /// experts + the zero dummy expert): each MoE head step then exchanges
    /// its expanded rows through the forward's row gather.
    pub(crate) ep_bound: bool,
    /// MoE head steps sealed under EP so far: a monotonic invocation counter
    /// (never reset, never part of a state snapshot) whose parity picks the
    /// head's gather buffer `2 + j % 2`, so a probe-and-restore never reuses
    /// a buffer on two consecutive invocations.
    pub(crate) moe_steps: u64,
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
    /// Only this position's K/V and index-key cache rows. For the last step
    /// of a chain: the next step brings its own backbone hidden and reselects
    /// (the head state and selection are left stale; a reuse is refused).
    Append,
}

/// The FFN call of one non-Append MTP step, in the profile's FFN kind.
enum MtpFfnCall<'a> {
    Moe(hipfire_dispatch::pipeline::sealed_moe::SealedMoeCall<'a>),
    Dense(GatedMlpOp<'a>),
}

/// Draft ranking policy (`HIPFIRE_MTP_DRAFT_HEAD`, default `mq2r`) and row
/// layout of the MTP draft head. Only the Flash-Next head takes the
/// quantized ranking copy (its requant and re-score kernels are K = 2560 /
/// 256-group shaped); a developer profile ranks the model's own head.
fn draft_head_config(config: &Qwen4Config) -> (DraftHeadPolicy, DraftHeadLayout) {
    let policy = if config.profile.is_flash_next() {
        DraftHeadPolicy::parse(
            &hipfire_config::developer_var("HIPFIRE_MTP_DRAFT_HEAD")
                .unwrap_or_else(|_| "mq2r".to_string()),
        )
    } else {
        DraftHeadPolicy {
            copy: None,
            rescore: false,
        }
    };
    let layout = DraftHeadLayout {
        vocab: config.vocab_size,
        hidden: config.hidden_size,
        front: DRAFT_FRONT,
        special: config.eos_token_id as usize,
        full_hold: DRAFT_FULL_HOLD,
    };
    (policy, layout)
}

/// The config half of [`admit_mtp`]: the profile declares a head and its RoPE
/// base is the one the QSA kernels implement. Flash-Next keeps its pinned
/// `Qwen4MtpConfig::validate` contract unchanged.
fn admit_mtp_config(config: &Qwen4Config) -> Result<(), MtpGpuError> {
    let mtp = config
        .mtp
        .as_ref()
        .ok_or_else(|| invalid("profile has no native MTP head"))?;
    if config.profile.is_flash_next() {
        mtp.validate().map_err(MtpGpuError::Invalid)?;
    } else {
        mtp.validate_shape().map_err(MtpGpuError::Invalid)?;
        // The MTP QSA runs the trunk's RoPE kernels, which hard-code the base.
        if mtp.rope_theta != QWEN4_QSA_KERNEL_ROPE_THETA {
            return Err(invalid(format!(
                "profile {} MTP rope_theta {} differs from the {} base hard-coded in the QSA RoPE kernels",
                config.profile.id(),
                mtp.rope_theta,
                QWEN4_QSA_KERNEL_ROPE_THETA
            )));
        }
    }
    Ok(())
}

/// Admit the head before anything is allocated: the config contract of
/// [`admit_mtp_config`] and FFN weights of the profile's FFN kind and width.
fn admit_mtp(weights: &Qwen4Weights, config: &Qwen4Config) -> Result<(), MtpGpuError> {
    admit_mtp_config(config)?;
    let head = bound_mtp(weights)?;
    match (&config.ffn, &head.ffn) {
        (crate::config::Qwen4FfnConfig::Moe { .. }, Qwen4FfnWeights::Moe(_)) => Ok(()),
        (
            crate::config::Qwen4FfnConfig::Dense { intermediate_size },
            Qwen4FfnWeights::Dense(dense),
        ) => {
            let gate = dense_ref(weights, &dense.gate)?;
            let up = dense_ref(weights, &dense.up)?;
            let down = dense_ref(weights, &dense.down)?;
            if (gate.m, gate.k) != (*intermediate_size, config.hidden_size)
                || (up.m, up.k) != (*intermediate_size, config.hidden_size)
                || (down.m, down.k) != (config.hidden_size, *intermediate_size)
            {
                return Err(invalid(format!(
                    "MTP dense FFN is gate {}x{} / up {}x{} / down {}x{}, profile expects intermediate {intermediate_size} and hidden {}",
                    gate.m, gate.k, up.m, up.k, down.m, down.k, config.hidden_size
                )));
            }
            Ok(())
        }
        _ => Err(invalid(
            "MTP FFN weights are not of the profile's FFN kind (dense/MoE)",
        )),
    }
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

    /// [`Self::device_bytes`] per allocation: `(resident, load scratch)`, one
    /// entry per device allocation [`Self::new_with_backend`] makes and keeps
    /// (operator scratch, QSA state with its snapshot mirrors, the MoE
    /// pointer tables, the draft head), and the draft head's build scratch
    /// released before construction returns. Each side sums to the matching
    /// `device_bytes` figure; `None` on overflow.
    pub(crate) fn allocation_sizes(
        config: &Qwen4Config,
        context: &Qwen4ContextCommit,
        head_dtype: DType,
    ) -> Option<(Vec<u64>, Vec<u64>)> {
        let (policy, layout) = draft_head_config(config);
        let (draft, draft_scratch) = DraftHead::allocation_sizes(head_dtype, layout, policy)?;
        let widen = |sizes: Vec<usize>| -> Option<Vec<u64>> {
            sizes
                .into_iter()
                .map(|bytes| u64::try_from(bytes).ok())
                .collect()
        };
        let mut resident = MtpGpuScratch::allocation_sizes(config)?;
        resident.extend(MtpGpuState::allocation_sizes(config, context)?);
        resident.extend(Qwen4MoeLayerRuntime::allocation_sizes(config)?);
        resident.extend(widen(draft)?);
        Some((resident, widen(draft_scratch)?))
    }

    /// MTP head whose QSA context arenas are `backend` storage (admission
    /// resolves it); see [`MtpGpuState::new_with_backend`]. Under expert
    /// parallelism `ep` is the rank's binding from its attached forward (the
    /// placement the weights were loaded with and the rank's zero dummy
    /// expert): the head's MoE runtime is then the rank's EP decode table.
    pub(crate) fn new_with_backend(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        max_seq: usize,
        backend: Qwen4KvBackend,
        ep: Option<(Qwen4EpPlacement, &Qwen4EpDummyExpert)>,
    ) -> Result<Self, MtpGpuError> {
        admit_mtp(weights, config)?;
        if weights.ep_placement() != ep.map(|(placement, _)| placement) {
            return Err(invalid(
                "the MTP head's EP binding must be the forward's placement the weights were loaded with (EP weights are rank-compact)",
            ));
        }
        let ep_bound = ep.is_some();
        if max_seq == 0 || max_seq > config.context_limit() {
            return Err(invalid("MTP max_seq is outside model capacity"));
        }
        let head = bound_mtp(weights)?;
        let scratch = MtpGpuScratch::new(gpu, config)?;
        let state = match MtpGpuState::new_with_backend(gpu, config, max_seq, backend) {
            Ok(state) => state,
            Err(error) => {
                let _ = scratch.free_gpu(gpu);
                return Err(error);
            }
        };
        // The MTP head is never an expert-tier layer: its experts are the
        // separately loaded `mtp.` tensors (host-mapped by name), and its
        // runtime label 0 must never resolve trunk layer 0's tier view.
        // Tiers are single-device only, so an EP head cannot coexist with a
        // tier lease at all.
        if ep.is_some() && weights.expert_tiers().is_some() {
            let _ = state.free_gpu(gpu);
            let _ = scratch.free_gpu(gpu);
            return Err(invalid(
                "expert tiers are single-device: an EP MTP head cannot attach to tiered weights",
            ));
        }
        let moe = match head.ffn.moe() {
            Some(moe_weights) => {
                let built = match ep {
                    Some((placement, dummy)) => Qwen4MoeLayerRuntime::from_moe_ep(
                        gpu,
                        weights,
                        moe_weights,
                        0,
                        config,
                        placement,
                        dummy,
                    ),
                    None => Qwen4MoeLayerRuntime::from_moe_untiered(
                        gpu,
                        weights,
                        moe_weights,
                        0,
                        config,
                    ),
                };
                match built {
                    Ok(moe) if moe.tier_lease().is_some() => {
                        let _ = moe.free_gpu(gpu);
                        let _ = state.free_gpu(gpu);
                        let _ = scratch.free_gpu(gpu);
                        return Err(invalid(
                            "the MTP head's MoE runtime must be untiered (its experts are the `mtp.` tensors)",
                        ));
                    }
                    Ok(moe) => Some(moe),
                    Err(error) => {
                        let _ = state.free_gpu(gpu);
                        let _ = scratch.free_gpu(gpu);
                        return Err(error.into());
                    }
                }
            }
            None => None,
        };
        let (policy, layout) = draft_head_config(config);
        let draft = weights
            .resident(&weights.root.lm_head)
            .map_err(MtpGpuError::from)
            .and_then(|head| DraftHead::new(gpu, head, layout, policy).map_err(MtpGpuError::from));
        let draft = match draft {
            Ok(draft) => draft,
            Err(error) => {
                let _ = moe.map(|moe| moe.free_gpu(gpu));
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
            ep_bound,
            moe_steps: 0,
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
        let moe_error = moe.and_then(|moe| moe.free_gpu(gpu));
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
            &self.scratch.ffn_output,
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

    /// Side-request cold reset: fresh draft policy as [`Self::reset`], rows
    /// of the append-only streams untouched.
    pub(crate) fn reset_for_side(&mut self, gpu: &mut Gpu) -> Result<(), MtpGpuError> {
        self.draft.reset_request_state();
        self.state.reset_for_side(gpu)
    }

    /// See [`MtpGpuState::side_request_rows`].
    pub(crate) fn side_request_rows(&self, through: usize) -> Vec<usize> {
        self.state.side_request_rows(through)
    }

    /// See [`MtpGpuState::ensure_mapped_capacity`].
    pub(crate) fn ensure_mapped_capacity(
        &mut self,
        gpu: &mut Gpu,
        required_tokens: usize,
    ) -> Result<(), MtpGpuError> {
        self.state.ensure_mapped_capacity(gpu, required_tokens)
    }

    /// Admitted token capacity of the head's context arenas.
    pub(crate) fn context_capacity(&self) -> usize {
        self.state.context_capacity()
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

    /// Validate a snapshot's head metadata (shape, mark, boundary row)
    /// without touching any state: a restore is refused here before any owner
    /// mutates.
    pub(crate) fn check_session_restore(&self, meta: &[u8]) -> Result<(), MtpGpuError> {
        let words = Self::session_words(meta)?;
        self.state
            .parse_session_mark(&words[..MTP_SESSION_WORDS])
            .map(|_| ())
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
    pub(crate) fn context_growth_bytes(&self, through: usize) -> u64 {
        self.state.context_growth_bytes(through)
    }

    /// See [`MtpGpuState::save_boundary_hidden`].
    pub(crate) fn save_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<(), MtpGpuError> {
        self.state.save_boundary_hidden(gpu, pending)
    }

    /// See [`MtpGpuState::restore_boundary_hidden`].
    pub(crate) fn restore_boundary_hidden(
        &mut self,
        gpu: &mut Gpu,
        pending: &GpuTensor,
    ) -> Result<bool, MtpGpuError> {
        self.state.restore_boundary_hidden(gpu, pending)
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

    pub(crate) fn retain_validated(&mut self, snapshot: MtpGpuStateSnapshot) {
        self.state.retain_validated(snapshot);
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
        ep: Option<MtpEpGather<'_>>,
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
                ep,
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
        ep: Option<MtpEpGather<'_>>,
    ) -> Result<Option<u32>, MtpGpuError> {
        let lengths = self.head_block(
            gpu,
            weights,
            config,
            token,
            backbone_hidden,
            position,
            fresh_qsa_selection,
            step,
            ep,
        )?;
        let next_token = if step == MtpStep::Predict {
            let head = weights.resident(&weights.root.lm_head)?;
            Some(self.draft.draft(gpu, head, &self.scratch.hc_mixed)?)
        } else {
            None
        };
        self.finish_head_step(gpu, step, lengths)?;
        Ok(next_token)
    }

    /// [`Self::forward_token`] with [`MtpStep::Predict`] and the draft read
    /// deferred: the head block, the draft's device work into `top1_out`
    /// (Raw, at least `Gpu::TOPK8_OUT_BYTES`; parsed later by
    /// `self.draft.draft_finish`), the wide-hidden copy and the state
    /// bookkeeping, in `forward_token`'s order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_token_deferred(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
        top1_out: &GpuTensor,
        ep: Option<MtpEpGather<'_>>,
    ) -> Result<(), MtpGpuError> {
        if top1_out.dtype != DType::Raw || top1_out.byte_size() < Gpu::TOPK8_OUT_BYTES {
            return Err(invalid(format!(
                "MTP deferred draft output must be Raw with at least {} bytes",
                Gpu::TOPK8_OUT_BYTES
            )));
        }
        let lengths = self.head_block(
            gpu,
            weights,
            config,
            token,
            backbone_hidden,
            position,
            fresh_qsa_selection,
            MtpStep::Predict,
            ep,
        )?;
        let head = weights.resident(&weights.root.lm_head)?;
        self.draft
            .draft_enqueue(gpu, head, &self.scratch.hc_mixed, top1_out)?;
        self.finish_head_step(gpu, MtpStep::Predict, lengths)
    }

    /// The tail of a head step the draft does not touch: the wide-hidden
    /// copy (every step but an append) and the state bookkeeping.
    fn finish_head_step(
        &mut self,
        gpu: &mut Gpu,
        step: MtpStep,
        (full_len, raw_len, pooled_len, selected_len, next_position): (
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
    ) -> Result<(), MtpGpuError> {
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
        Ok(())
    }

    /// The head step's checks and device work up to (not including) the
    /// draft: returns the lengths the step publishes in
    /// [`Self::finish_head_step`].
    #[allow(clippy::too_many_arguments)]
    fn head_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
        step: MtpStep,
        ep: Option<MtpEpGather<'_>>,
    ) -> Result<(usize, usize, usize, usize, usize), MtpGpuError> {
        let wide = config
            .hc_count
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

        let mtp = bound_mtp(weights)?;
        let lengths = {
            let scratch = &self.scratch;
            let state = &self.state;
            let backbone_hidden = backbone_hidden.unwrap_or(&scratch.backbone_hidden);
            let hidden = config.hidden_size;
            let ctx = DispatchCtx::new(gpu);
            let attn = hyper_desc(weights, &mtp.attn_hyper)?;
            let mlp = hyper_desc(weights, &mtp.mlp_hyper)?;
            let final_read = hyper_read_desc(weights, &mtp.final_hyper)?;
            let qsa = qsa_desc(weights, &mtp.attention)?;
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
                    f16_arena: None,
                    pooled_bf16: None,
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
                rotary_dim: config.rotary_dim(),
                rope: config.rope(),
                mrope: None,
                input_width: hidden,
                rotation: &scratch.rotation,
                mode,
                trunk_a4: 0,
                phase: MixerPhase::Whole,
            };
            let lengths = attention.next_lengths()?;
            // Under EP a MoE head step exchanges its expanded rows through the
            // forward's row gather: gather buffer `2 + j % 2` and layer index
            // `trunk_layers + j % 2` for the head's monotonic MoE invocation
            // counter `j` (one increment per MoE head step, never rewound).
            let moe_ep: Option<(usize, &GpuTensor, &dyn MoeRowGather)> = if self.ep_bound
                && step != MtpStep::Append
                && matches!(&mtp.ffn, Qwen4FfnWeights::Moe(_))
            {
                let ep = ep
                    .ok_or_else(|| invalid("an EP MTP head step needs the forward's row gather"))?;
                let parity = (self.moe_steps % 2) as usize;
                self.moe_steps = self.moe_steps.wrapping_add(1);
                Some((
                    config.num_hidden_layers + parity,
                    ep.buffers[2 + parity],
                    ep.hook as &dyn MoeRowGather,
                ))
            } else {
                None
            };
            // The MTP layer's FFN is the trunk's kind (`attach` refuses a
            // head whose FFN weights disagree with the profile).
            let ffn = if step == MtpStep::Append {
                None
            } else {
                Some(match (&mtp.ffn, self.moe.as_ref()) {
                    (Qwen4FfnWeights::Moe(_), Some(runtime)) => {
                        MtpFfnCall::Moe(seal_moe_decode(
                            &ctx,
                            config,
                            moe_ep.map_or(0, |(layer, _, _)| layer),
                            runtime,
                            &scratch.hc_mixed,
                            &scratch.ffn_output,
                            Qwen4MoeScratch {
                                router_logits: &scratch.router_logits,
                                scalar_buf: &scratch.moe_scalar,
                                x_rot_local: &scratch.moe_x_rot,
                                gate_up_buf: &scratch.moe_gate_up,
                                gate_buf: &scratch.moe_gate,
                                up_buf: &scratch.moe_up,
                                ffn_hidden: &scratch.moe_hidden,
                                // The shared down projection must not
                                // overwrite the routed accumulator.
                                ffn_out: &scratch.projected_embedding,
                                gate_batch: &scratch.moe_gate_batch,
                                up_batch: &scratch.moe_up_batch,
                                rot_batch: &scratch.moe_rot_batch,
                                topk_indices: &scratch.moe_topk_indices,
                                topk_weights: &scratch.moe_topk_weights,
                                down_expanded: match moe_ep {
                                    Some((_, buffer, _)) => buffer,
                                    None => &scratch.moe_down_expanded,
                                },
                            },
                            moe_ep.map(|(_, _, hook)| hook),
                        )?)
                    }
                    (Qwen4FfnWeights::Dense(dense), None) => {
                        let gate = dense_ref(weights, &dense.gate)?;
                        MtpFfnCall::Dense(GatedMlpOp {
                            rows: 1,
                            hidden,
                            intermediate: gate.m,
                            x: &scratch.hc_mixed,
                            pre_norm: None,
                            gate,
                            up: dense_ref(weights, &dense.up)?,
                            down: dense_ref(weights, &dense.down)?,
                            activation: Activation::Silu,
                            storage: MlpStorage::Bf16Source,
                            output: MlpOutput::Write(&scratch.ffn_output),
                            ladder: MlpLadder::Trunk,
                            // The MoE batch scratch, sized to the dense
                            // intermediate for a dense profile.
                            scratch: GatedMlpScratch {
                                gate: &scratch.moe_gate_batch,
                                up: &scratch.moe_up_batch,
                                act: &scratch.moe_gate_batch,
                                x_rot: &scratch.rotation,
                            },
                        })
                    }
                    _ => return Err(invalid("MTP FFN weights and runtime disagree")),
                })
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
                norm_weight: weights.resident(&mtp.pre_fc_norm_embedding)?,
                normalized: &scratch.embedding_norm,
                branches: 1,
                hidden,
                state_bf16: false,
            }));
            steps.push(Step::Project(ProjectOp {
                weight: dense_ref(weights, &mtp.fc_embedding)?,
                input: &scratch.embedding_norm,
                output: &scratch.projected_embedding,
                rows: 1,
                rotation: Some(&scratch.rotation),
                ladder: ProjectLadder::Trunk,
                residual: None,
            }));
            steps.push(Step::HyperNorm(HyperNormOp {
                input: backbone_hidden,
                norm_weight: weights.resident(&mtp.pre_fc_norm_hidden)?,
                normalized: &scratch.hidden_norm,
                branches: config.hc_count,
                hidden,
                state_bf16: false,
            }));
            // Both branch rows in one projection (each bitwise its own).
            steps.push(Step::Project(ProjectOp {
                weight: dense_ref(weights, &mtp.fc_hidden)?,
                input: &scratch.hidden_norm,
                output: &scratch.projected_hidden,
                rows: config.hc_count,
                rotation: Some(&scratch.rotation),
                ladder: ProjectLadder::Trunk,
                residual: None,
            }));
            steps.push(Step::BroadcastAdd(BroadcastAddOp {
                rows_input: &scratch.projected_hidden,
                row: &scratch.projected_embedding,
                output: &scratch.wide,
                rows: config.hc_count,
                width: hidden,
            }));
            steps.push(Step::HyperRead(hc_read(&attn.read, scratch, config)));
            steps.push(Step::IndexedAttention(attention));
            if let Some(ffn) = ffn {
                steps.push(Step::HyperWrite(hc_write(
                    &attn.write,
                    &scratch.projected_embedding,
                    scratch,
                    config,
                )));
                steps.push(Step::HyperRead(hc_read(&mlp.read, scratch, config)));
                match ffn {
                    MtpFfnCall::Moe(moe) => {
                        steps.push(Step::Clear(ClearOp {
                            tensor: &scratch.ffn_output,
                            elements: hidden,
                        }));
                        steps.push(Step::Moe(moe));
                    }
                    MtpFfnCall::Dense(dense) => steps.push(Step::GatedMlp(dense)),
                }
                steps.push(Step::HyperWrite(hc_write(
                    &mlp.write,
                    &scratch.ffn_output,
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
        Ok(lengths)
    }

    /// Whether [`Self::append_rows`] can run for this model on this GPU:
    /// exact gfx1151 or gfx1201 (the tiled Q8_0 projection repeats the dot
    /// of the per-row Q8_0 GEMV there: the K = 2560 staged GEMV on gfx1151,
    /// `gemv_q8_0` on gfx1201; other arches have no proven batched twin)
    /// and the shipped weight formats and shapes.
    pub(crate) fn append_rows_supported(
        gpu: &Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
    ) -> bool {
        (gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201())
            && AppendWidths::new(config)
                .is_some_and(|w| AppendWeights::resolve(weights, config, &w).is_ok())
    }

    /// [`MtpStep::Append`] for `tokens.len()` consecutive prompt rows starting
    /// at `position`, row `i` paired with `hidden` row `i` (the caller supplies
    /// the pairing: the token at position `p` pairs with the target hidden of
    /// position `p - 1`, prompt position 0 with its own hidden, as the per-row
    /// loop does). Runs the per-row `forward_token` Append
    /// op sequence once over all rows, calling the rdna-compute entry points
    /// directly (never the `Step` dispatcher or `project_weight`, whose
    /// multi-row routes change the arithmetic): each op is the single-row
    /// kernel with a rows grid, or a multi-row kernel whose per-row
    /// reduction order is the single-row kernel's.
    ///
    /// `hidden` holds at least `tokens.len()` rows of `hc_count * hidden`
    /// F32. Draft request state sees every token in order, as `forward_token`
    /// observes each; the QSA state commits `position + tokens.len()` rows.
    /// `wide_hidden`, the selection and its length are not touched (the
    /// per-row Append leaves them alone too).
    ///
    /// `context` names the caller. [`MtpFillContext::PromptFill`] takes the
    /// DET route (all seven projections through the fixed-row native kernels)
    /// on exact gfx1151 (`HIPFIRE_QWEN4_MTP_APPEND_DET`, default on), which also takes
    /// a lone row; [`MtpFillContext::DecodeFill`] always runs the incumbent
    /// kernels (at least two rows).
    ///
    /// `device_ids`, when given, holds `tokens` as `tokens.len()` device i32
    /// ids (at least that many bytes) that the embedding gather reads in place
    /// of the scratch upload: the trunk forward's uploaded rows, or the
    /// caller's once-per-chunk `MtpAppendScratch::upload_prompt_ids` rows.
    /// The backing allocation must remain live and any later writes must be
    /// ordered after the enqueued embedding gather; returning from this call
    /// does not imply device completion. With `device_ids` this call never
    /// blocks the host; without it the scratch upload is a blocking
    /// `hipMemcpy` that waits for all previously queued device work.
    pub(crate) fn append_rows(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        context: MtpFillContext,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden: &GpuTensor,
        device_ids: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), MtpGpuError> {
        let rows = tokens.len();
        let w = AppendWidths::new(config).ok_or_else(|| invalid("MTP append width overflow"))?;
        let det = append_rows_det(context, mtp_append_det_enabled(gpu));
        let min_rows = append_rows_min(det);
        if rows < min_rows || rows > scratch.rows {
            return Err(invalid(format!(
                "MTP batched append of {rows} rows needs {min_rows}..={} (scratch capacity{})",
                scratch.rows,
                if det {
                    ""
                } else {
                    "; the incumbent multirow projections take two rows or more"
                }
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
        if !(gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1201()) {
            return Err(invalid("MTP batched append needs exact gfx1151 or gfx1201"));
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
        if det {
            // Refused before the draft state observes anything: the DET dense
            // kernel's M / K limits against the seven projection weights.
            for (label, weight) in [
                ("fc_embedding", &mw.fc_embedding),
                ("fc_hidden", &mw.fc_hidden),
                ("HC down", &mw.hc.input_mix_down),
                ("HC up", &mw.hc.input_mix_up),
                ("indexer_qk", &mw.qsa.indexer_qk),
                ("k_proj", &mw.qsa.k),
                ("v_proj", &mw.qsa.v),
            ] {
                if !det_dense_shape_ok(weight.m, weight.k, rows) {
                    return Err(invalid(format!(
                        "MTP DET append {label} [{}x{}] is outside the DET kernel's limits (M % 16 == 0, K % 64 == 0)",
                        weight.m, weight.k
                    )));
                }
            }
        }
        for &token in tokens {
            self.draft.observe(token);
        }
        // Given device ids (the trunk's already uploaded rows or the prompt
        // chunk's one upload, same bytes as `tokens`) skip the host fill and
        // copy; otherwise the scratch uploads.
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
        gpu.embedding_lookup_q8_batched(mw.embedding, &token_embedding, token_ids, rows, w.hidden)?;
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
        // Incumbent (the default, and every `DecodeFill`): G1 (byte identity vs
        // the per-row reference) governs every `gemm_bf16_xf32_multirow` and
        // tiled projection below; the reference
        // is the per-row kernel of each operator on the running arch:
        //   - fc_embedding and HC up: singleton `gemv_bf16_xf32` (gfx1201
        //     runs the shipped `gemv_bf16_xf32` kernel);
        //   - fc_hidden (four branch rows per token): on gfx1201
        //     `gemv_bf16_xf32_x4_rows_r4`, not four singleton GEMVs;
        //   - HC down: the K4 singleton (`gemv_bf16_xf32_k4`, scale 1/4);
        //   - QSA indexer_qk / K / V (Q8_0): on gfx1151 the K = 2560 staged
        //     `gemv_q8_0`, on gfx1201 the unstaged `gemv_q8_0` (the staged
        //     tiled kernel repeats the same dot arithmetic).
        // On gfx1151 `gemm.rs` selects by (m, k, batch): (2560, 2560) is on
        // the r16 allowlist for batch >= 64 (`r16w4t` / `r16w4` / `r16`,
        // documented bitwise to the four-row kernel) and the base four-row
        // kernel below that; (10240, 320) is not on the allowlist and
        // K = 320 takes the `pto2` tile (257..=512, LDS weights, also
        // documented bitwise to the four-row kernel). The same selector
        // runs on gfx1201 (the `r16*_gfx1151` names are not gfx1151-only).
        // If G1 shows any difference for a pair, replace that call
        // with a rows-grid twin of the single-row kernel (`blockIdx.y`
        // token tile over `gemv_bf16_xf32_rows_body`) before tuning
        // anything. Never route these through `project_weight` /
        // `gemm_bf16_xf32_f16_wmma_qwen4` (F16 WMMA from 512 rows).
        //
        // DET (`PromptFill` on exact gfx1151, `HIPFIRE_QWEN4_MTP_APPEND_DET`
        // on unless `0`, `det` below): all seven projections run the fixed-row
        // native kernels instead (F16 WMMA, F32 accumulate, one ascending-K
        // chain per output, no split-K, no epilogue). That arithmetic is
        // deterministic and row-invariant (a row's bytes do not depend on its
        // batch or sub-chunk) but it is NOT G1: the operands are F16-narrowed
        // (BF16 weights through a model-lifetime F16 shadow, X rounded to F16
        // per call; a narrower exponent range than BF16/F32) and the
        // gfx11 WMMA's internal dot order is its own, so the outputs are not
        // byte-equal to the incumbent kernels or to the per-row reference.
        append_dense_projection(
            gpu,
            det,
            &mw.fc_embedding,
            &embedding_norm,
            &projected_embedding,
            rows,
        )?;
        hyper_norm(
            gpu,
            &HyperNorm {
                input: &backbone,
                norm_weight: mw.norm_hidden,
                normalized: &hidden_norm,
                branches: config.hc_count,
                hidden: w.hidden,
                state_bf16: false,
            },
        )?;
        // The four branch rows of a token are four projection rows
        // (per-row: `gemv_bf16_xf32_x4_rows` over rows = hc_count).
        append_dense_projection(
            gpu,
            det,
            &mw.fc_hidden,
            &hidden_norm,
            &projected_hidden,
            rows * config.hc_count,
        )?;
        gpu.broadcast_add_rows_f32(
            &projected_hidden,
            &projected_embedding,
            &wide,
            config.hc_count,
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
        if det {
            // DET: the raw fixed-ascending-K F16 WMMA down projection, then the
            // HC read activation (`hc_activation_fused_f32`, scale 1/branches,
            // in place over the F32 `hc_low` rows) the incumbent fuses into the
            // K4 GEMV epilogue; the same SiLU expression, not a second one.
            gpu.gemm_bf16_xf32_mtp_dense_det_pm(
                mw.hc.input_mix_down.buf,
                &hc_normalized,
                &hc_low,
                mw.hc.input_mix_down.m,
                mw.hc.input_mix_down.k,
                rows,
            )?;
            hc_activation_fused_f32(
                gpu,
                &HcActivationFused {
                    values: &hc_low,
                    scale: 1.0 / config.hc_count as f32,
                    bf16_out: None,
                },
            )?;
        } else {
            // The long-K down projection keeps the K4 quarter fold and applies
            // the 1/branches activation in its epilogue, as the per-row route does.
            gpu.gemv_bf16_xf32_k4_rows_tiled(
                mw.hc.input_mix_down.buf,
                &hc_normalized,
                &hc_low,
                mw.hc.input_mix_down.m,
                mw.hc.input_mix_down.k,
                Some(1.0 / config.hc_count as f32),
                rows,
            )?;
        }
        append_dense_projection(gpu, det, &mw.hc.input_mix_up, &hc_low, &hc_up, rows)?;
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
        if det {
            // DET: the fixed-row Q8_0 projection (`mtp_q8_det_pm_launch`), not
            // the staged GEMV's arithmetic. The mixed row converts F32 -> F16
            // RNE once (the shared X scratch, same stream) for all three
            // projections; each launch reads it in place.
            let mixed16 = gpu.qwen4_convert_f16_x(&hc_mixed, rows * w.hidden)?;
            for (weight, output) in [
                (&mw.qsa.indexer_qk, &index),
                (&mw.qsa.k, &qsa_k),
                (&mw.qsa.v, &qsa_v),
            ] {
                gpu.mtp_q8_det_pm_launch(weight.buf, &mixed16, output, weight.m, weight.k, rows)?;
            }
        } else {
            gpu.gemv_q8_0_k2560_staged_rows_tiled(
                mw.qsa.indexer_qk.buf,
                &hc_mixed,
                &index,
                mw.qsa.indexer_qk.m,
                rows,
            )?;
            gpu.gemv_q8_0_k2560_staged_rows_tiled(
                mw.qsa.k.buf,
                &hc_mixed,
                &qsa_k,
                mw.qsa.k.m,
                rows,
            )?;
            gpu.gemv_q8_0_k2560_staged_rows_tiled(
                mw.qsa.v.buf,
                &hc_mixed,
                &qsa_v,
                mw.qsa.v.m,
                rows,
            )?;
        }
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
                    yarn: config.rope().yarn(config.head_dim.min(64)),
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
                        yarn: config.rope().yarn(w.index_kv_width.min(64)),
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

/// One developer-profile MTP head row read back to the host
/// ([`crate::bundle::Qwen4Bundle::mtp_dev_row`]).
#[cfg(feature = "qwen4-dev-profiles")]
pub struct MtpDevRow {
    /// The draft token the head ranked.
    pub draft_token: u32,
    /// Head logits over the vocabulary (the model's own LM head, token order).
    pub logits: Vec<f32>,
    /// The mixed `[hidden]` row the LM head reads (after the final HC read).
    pub sample_hidden: Vec<f32>,
    /// The `[hc_count * hidden]` streams after the layer's MLP HC write.
    pub multi_hidden: Vec<f32>,
}

#[cfg(feature = "qwen4-dev-profiles")]
impl crate::bundle::Qwen4Bundle {
    /// Developer-profile direct input of the native MTP head: one Predict
    /// step of `token` at `position` (the head's next position) conditioned
    /// on the caller's `[hc_count * hidden]` backbone streams `backbone`
    /// instead of the head's own previous hidden, with a fresh QSA
    /// selection. The step is the production `forward_token`; only the
    /// hidden source and the read-back are different. Refused for the
    /// pinned Flash-Next profile.
    pub fn mtp_dev_row(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone: &[f32],
        position: usize,
    ) -> Result<MtpDevRow, crate::bundle::BundleError> {
        use crate::bundle::BundleError;
        let forward = |message: String| BundleError::Forward(message);
        if self.config.profile.is_flash_next() {
            return Err(forward(
                "the direct MTP row feed is for developer profiles only".into(),
            ));
        }
        let wide = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| forward("MTP wide dimension overflow".into()))?;
        if backbone.len() != wide {
            return Err(forward(format!(
                "MTP backbone row has {} values, expected {wide}",
                backbone.len()
            )));
        }
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| forward("Qwen4 MTP resources are not attached".into()))?;
        let required = position
            .checked_add(1)
            .ok_or_else(|| forward("Qwen4 MTP position overflows".into()))?;
        mtp.ensure_mapped_capacity(gpu, required)
            .map_err(|error| forward(error.to_string()))?;
        let input = gpu
            .upload_f32(backbone, &[wide])
            .map_err(BundleError::Hip)?;
        let step = mtp.forward_token(
            gpu,
            &self.weights,
            &self.config,
            token,
            Some(&input),
            position,
            true,
            MtpStep::Predict,
            // Developer profiles never run under EP.
            None,
        );
        let freed = gpu.free_tensor(input);
        let draft_token = step
            .map_err(|error| forward(error.to_string()))?
            .ok_or_else(|| forward("MTP prediction requested but no token produced".into()))?;
        freed.map_err(BundleError::Hip)?;
        gpu.hip.device_synchronize().map_err(BundleError::Hip)?;
        let mut row = MtpDevRow {
            draft_token,
            logits: Vec::new(),
            sample_hidden: Vec::new(),
            multi_hidden: Vec::new(),
        };
        for (tensor, out) in [
            (mtp.draft.logits(), &mut row.logits),
            (&mtp.scratch.hc_mixed, &mut row.sample_hidden),
            (&mtp.state.wide_hidden, &mut row.multi_hidden),
        ] {
            gpu.download_f32_into(tensor, out)
                .map_err(BundleError::Hip)?;
        }
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::state::Qwen4State;

    fn sum(sizes: &[u64]) -> u64 {
        sizes.iter().sum()
    }

    /// Operator and fill scratch inventories are their owners' own shape
    /// tables: one entry per tensor, summing to the byte queries, with the
    /// append scratch refusing overflow like its byte query.
    #[test]
    fn scratch_allocation_sizes_follow_the_shape_tables() {
        let config = compact_test_config();
        let sizes = MtpGpuScratch::allocation_sizes(&config).unwrap();
        assert_eq!(sizes.len(), MTP_SCRATCH_TENSORS);
        assert_eq!(
            Some(sum(&sizes) as usize),
            MtpGpuScratch::device_bytes(&config).ok()
        );
        for (rows, id_rows) in [(1usize, 1usize), (1, 1536), (256, 4096), (1024, 1536)] {
            let sizes = MtpAppendScratch::allocation_sizes(&config, rows, id_rows).unwrap();
            assert_eq!(sizes.len(), MTP_APPEND_TENSORS);
            assert_eq!(
                Some(sum(&sizes) as usize),
                MtpAppendScratch::device_bytes(&config, rows, id_rows)
            );
        }
        assert_eq!(
            MtpAppendScratch::allocation_sizes(&config, usize::MAX, 1),
            None
        );
        assert_eq!(
            MtpAppendScratch::allocation_sizes(&config, 1, usize::MAX),
            None
        );
        assert_eq!(MtpAppendScratch::device_bytes(&config, usize::MAX, 1), None);
    }

    /// The head state's inventory: four context arenas (legacy whole, VMM
    /// mapped pages), the carries, and exactly one snapshot backup per
    /// in-place carry; sums to `MtpGpuState::device_bytes`.
    #[test]
    fn state_allocation_sizes_charge_arenas_carries_and_backups_once() {
        use hipfire_runtime::kv_backend::{
            KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        };
        let config = compact_test_config();
        let row = (config.hc_count * config.hidden_size * 4) as u64;
        let selected = (config.qsa_selected_capacity() * 4) as u64;
        let granularity = 2 << 20;
        for context in [
            Qwen4ContextCommit::legacy(1024),
            Qwen4ContextCommit::legacy(262_144),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 8193, 1025, granularity),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 1536, granularity),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 100, 8192, granularity),
        ] {
            let sizes = MtpGpuState::allocation_sizes(&config, &context).unwrap();
            assert_eq!(
                Some(sum(&sizes) as usize),
                MtpGpuState::device_bytes(&config, &context),
                "{context:?}"
            );
            // Carries and backups follow the four arenas.
            assert_eq!(
                sizes[4..],
                [selected, 4, row, row, selected, 4, row],
                "{context:?}"
            );
            // The arenas against the owner's plan per stride.
            let mapped = |row_bytes: usize, rows: usize, tokens: usize| match context.backend {
                Qwen4KvBackend::Legacy => (row_bytes * rows) as u64,
                Qwen4KvBackend::Vmm => KvChunkPlan::new(
                    row_bytes,
                    rows,
                    DEFAULT_KV_CHUNK_TOKENS,
                    granularity,
                    DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
                )
                .unwrap()
                .mapped_bytes_for_tokens(tokens)
                .unwrap() as u64,
            };
            let kv = config.num_key_value_heads * config.head_dim * 4;
            let index = config.indexer_kv_heads * config.indexer_head_dim * 4;
            let pooled = context.max_seq.div_ceil(config.indexer_compress_ratio);
            let pooled_tokens = context.tokens.div_ceil(config.indexer_compress_ratio);
            assert_eq!(
                sizes[..4],
                [
                    mapped(kv, context.max_seq, context.tokens),
                    mapped(kv, context.max_seq, context.tokens),
                    mapped(index, context.max_seq, context.tokens),
                    mapped(index, pooled, pooled_tokens),
                ],
                "{context:?}"
            );
        }
        assert_eq!(
            MtpGpuState::allocation_sizes(&config, &Qwen4ContextCommit::legacy(usize::MAX)),
            None
        );
    }

    /// Resident and build-scratch inventories of the whole head are the
    /// operator scratch, state, MoE tables and draft head, and sum to
    /// `Qwen4MtpGpu::device_bytes` for every head dtype and context commit.
    #[test]
    fn head_allocation_sizes_sum_to_the_head_device_bytes() {
        let config = compact_test_config();
        for context in [
            Qwen4ContextCommit::legacy(4096),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 262_144, 1536, 2 << 20),
        ] {
            for head in [
                DType::BF16,
                DType::Q8_0,
                DType::MQ6G256V2,
                DType::MQ5G256V2,
                DType::F32,
            ] {
                let (resident, scratch) =
                    Qwen4MtpGpu::allocation_sizes(&config, &context, head).unwrap();
                assert_eq!(
                    Some((sum(&resident) as usize, sum(&scratch) as usize)),
                    Qwen4MtpGpu::device_bytes(&config, &context, head),
                    "{head:?} {context:?}"
                );
                let mut parts = MtpGpuScratch::allocation_sizes(&config).unwrap();
                parts.extend(MtpGpuState::allocation_sizes(&config, &context).unwrap());
                parts.extend(Qwen4MoeLayerRuntime::allocation_sizes(&config).unwrap());
                assert_eq!(resident[..parts.len()], parts[..]);
                assert!(resident.iter().chain(&scratch).all(|&bytes| bytes > 0));
            }
        }
    }

    /// A capture carries a restorable boundary row only when the row was saved
    /// for exactly the captured position, at any position (a prefill chunk
    /// boundary such as 8192, or a live-position park such as 17051).
    #[test]
    fn boundary_word_is_restorable_only_for_the_position_it_was_saved_at() {
        for position in [1usize, 4096, 8192, 17051, 32767] {
            let word = boundary_word(Some(position), position);
            assert_eq!(word, position as u64 + 1);
            assert!(boundary_word_names(word, position));
            // Never saved: the capture is refused at restore.
            let unsaved = boundary_word(None, position);
            assert_eq!(unsaved, 0);
            assert!(!boundary_word_names(unsaved, position));
            // Saved for another position: stale, refused.
            for other in [0usize, position - 1, position + 1, 2 * position] {
                let stale = boundary_word(Some(other), position);
                assert_eq!(stale, 0, "saved {other}, captured {position}");
                assert!(!boundary_word_names(stale, position));
            }
            // A word carrying another position never names this one.
            assert!(!boundary_word_names(position as u64, position));
            assert!(!boundary_word_names(position as u64 + 2, position));
        }
    }

    #[test]
    fn append_det_is_exact_gfx1151_only_and_zero_kills_it() {
        // `0` (not requested): never on, on any arch.
        assert!(!append_det_active(false, true));
        assert!(!append_det_active(false, false));
        // Requested: only exact gfx1151 (`is_gfx1151` is `arch == "gfx1151"`,
        // so gfx1150 / gfx1152 / gfx1201 all arrive as `false`).
        assert!(append_det_active(true, true));
        assert!(!append_det_active(true, false));
        assert_eq!(MTP_APPEND_DET_ENV, "HIPFIRE_QWEN4_MTP_APPEND_DET");
    }

    #[test]
    fn append_det_default_resolves_on_for_gfx1151_only() {
        // Unset resolves to the default: on for exact gfx1151, off on every
        // other arch through the same `ArchCaps::is_gfx1151` predicate
        // `mtp_append_det_enabled` reads.
        use rdna_compute::arch_caps::ArchCaps;
        use rdna_compute::feature_flags::FeatureFlags;
        for arch in [
            "gfx1151", "gfx1150", "gfx1152", "gfx1100", "gfx1201", "gfx942",
        ] {
            let caps = ArchCaps::new(arch, std::sync::Arc::new(FeatureFlags::for_test(arch)));
            assert_eq!(
                append_det_active(MTP_APPEND_DET_DEFAULT, caps.is_gfx1151()),
                arch == "gfx1151",
                "{arch}",
            );
        }
    }

    #[test]
    fn only_prompt_fill_takes_the_det_route() {
        for enabled in [false, true] {
            // The decode takeover never takes DET, whatever the machine gate.
            assert!(!append_rows_det(MtpFillContext::DecodeFill, enabled));
        }
        assert!(append_rows_det(MtpFillContext::PromptFill, true));
        // Without the gate prompt fill keeps the incumbent route.
        assert!(!append_rows_det(MtpFillContext::PromptFill, false));
    }

    #[test]
    fn lone_prompt_row_takes_rows_only_under_det() {
        // Default: a lone row stays the per-row `append_token`.
        assert!(!prompt_fill_uses_rows(false, 1));
        assert!(!prompt_fill_uses_rows(false, 0));
        assert!(prompt_fill_uses_rows(false, 2));
        assert!(prompt_fill_uses_rows(false, 1024));
        // DET: one row is the same kernels; an empty pass is still no pass.
        assert!(prompt_fill_uses_rows(true, 1));
        assert!(prompt_fill_uses_rows(true, 2));
        assert!(!prompt_fill_uses_rows(true, 0));
        assert_eq!(append_rows_min(false), 2);
        assert_eq!(append_rows_min(true), 1);
    }

    #[test]
    fn det_dense_shape_limits_cover_the_append_projections() {
        // fc_embedding / fc_hidden, HC down (320 x 10240), HC up (10240 x 320)
        // and the Q8_0 projections' K = 2560, at one row and at a full pass.
        for (m, k) in [(2560, 2560), (320, 10240), (10240, 320)] {
            for n in [1, 2, 4096] {
                assert!(det_dense_shape_ok(m, k, n), "{m}x{k} n={n}");
            }
        }
        assert!(!det_dense_shape_ok(2560, 2560, 0));
        assert!(!det_dense_shape_ok(2568, 2560, 1));
        assert!(!det_dense_shape_ok(2560, 2592, 1));
        assert!(!det_dense_shape_ok(0, 2560, 1));
    }

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
        let wide = config.hc_count * config.hidden_size;
        assert_eq!(shapes[11], (wide, DType::F32), "hc_up is rows x W F32");
        let bytes = MtpGpuScratch::device_bytes(&config).expect("scratch bytes");
        assert_eq!(bytes, 659_824);
        let former = 13_726_064usize;
        assert_eq!(former - bytes, 13_066_240);
        assert_eq!((former - bytes) as f64 / (1u64 << 20) as f64, 12.4609375);
    }

    fn flash_next_mtp_plan_bytes(config: &Qwen4Config) -> usize {
        // The pre-profile-derived plan, written out from the MoE scalars: 36
        // tensors, F32 unless noted.
        let h = config.hidden_size;
        let w = config.hc_count * h;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let index = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let hc_up = hc_up_elements(1, w, config.hc_lowrank).unwrap();
        let rotation = w.max(h).max(config.hc_lowrank).max(q);
        let routed = config.num_experts_per_tok * config.moe_intermediate_size;
        let bf16 = h.max(config.hc_lowrank).max(q).max(kv);
        let raw = 4 + config.qsa_selected_capacity() * 4;
        let f32_elements = h
            + h
            + h
            + w
            + h
            + w
            + w
            + w
            + w
            + config.hc_lowrank
            + hc_up
            + h
            + config.hc_count
            + rotation
            + index
            + 2 * q
            + kv
            + kv
            + q
            + config.num_experts
            + h
            + 2 * config.moe_intermediate_size
            + 3 * config.moe_intermediate_size
            + h
            + 3 * routed
            + 2 * config.num_experts_per_tok
            + config.num_experts_per_tok * h
            + h.div_ceil(4)
            + config.shared_expert_intermediate_size;
        f32_elements * 4 + raw + bf16 * 2
    }

    #[test]
    fn flash_next_mtp_scratch_bytes_are_unchanged() {
        let config = compact_test_config();
        assert_eq!(
            MtpGpuScratch::device_bytes(&config).unwrap(),
            flash_next_mtp_plan_bytes(&config)
        );
    }

    fn tiny_dense() -> Qwen4Config {
        crate::config::synthetic_test_config(
            crate::config::Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        )
    }

    #[test]
    fn dense_mtp_scratch_is_profile_derived_and_never_empty() {
        let config = tiny_dense();
        let shapes = MtpGpuScratch::shapes(&config).unwrap();
        assert!(shapes.iter().all(|&(elements, _)| elements >= 1));
        // Two HC branches, not the Flash-Next four.
        let wide = config.hc_count * config.hidden_size;
        assert_eq!(wide, 32);
        assert_eq!(shapes[7], (wide, DType::F32), "wide streams");
        assert_eq!(shapes[8], (wide, DType::F32), "backbone hidden");
        // Dense gate/up rows reuse the batch scratch at the intermediate width.
        assert_eq!(shapes[29], (8, DType::F32));
        assert_eq!(shapes[30], (8, DType::F32));
        // MoE-only buffers are the one-element floor.
        for index in [22, 24, 25, 26, 27, 31, 32, 33, 35] {
            assert_eq!(shapes[index].0, 1, "tensor {index}");
        }
        // The FWHT basis scratch covers the dense intermediate width.
        assert!(shapes[15].0 >= 8);
        assert!(MtpGpuScratch::device_bytes(&config).is_ok());
    }

    #[test]
    fn mtp_refuses_a_rope_base_the_qsa_kernels_do_not_implement() {
        let mut config = tiny_dense();
        let message = admit_mtp_config(&config).unwrap_err().to_string();
        assert!(
            message.contains("10000 ")
                && message.contains(&QWEN4_QSA_KERNEL_ROPE_THETA.to_string()),
            "{message}"
        );
        config.mtp.as_mut().unwrap().rope_theta = QWEN4_QSA_KERNEL_ROPE_THETA;
        admit_mtp_config(&config).expect("kernel base admitted");
        config.mtp = None;
        assert!(admit_mtp_config(&config)
            .unwrap_err()
            .to_string()
            .contains("no native MTP head"));
        admit_mtp_config(&compact_test_config()).expect("Flash-Next head admitted");
    }

    #[test]
    fn developer_profiles_rank_their_own_head() {
        let (policy, _) = draft_head_config(&tiny_dense());
        assert_eq!((policy.copy, policy.rescore), (None, false));
    }

    /// Every subview the shared HC read takes from `up` must fit the
    /// scratch: a second row (either the F32 projection or the BF16 packing
    /// view), a rank wider than the row, or a non-F32 buffer is refused.
    #[test]
    fn hc_up_subview_larger_than_scratch_is_rejected() {
        let config = compact_test_config();
        let wide = config.hc_count * config.hidden_size;
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
        let wide = config.hc_count * config.hidden_size;
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
            &scratch.ffn_output,
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
