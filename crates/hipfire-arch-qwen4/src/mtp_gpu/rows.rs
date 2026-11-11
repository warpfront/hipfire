// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched head rounds (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`, default off):
//! one MTP head step for up to [`MTP_DRAFT_LANES_MAX`] drafting lanes at once.
//!
//! The head's row-local stages (embedding, both pre-FC norms and FC
//! projections, the HC reads and writes, the MoE with one EP row gather, the
//! final HC read, the draft head) run once over `n` rows; the stateful core
//! (the indexed attention over each lane's own head arena) runs once per lane
//! on that lane's row, and on exact gfx1201 the attention's Q8_0 input and
//! output projections run once over the `n` rows around those cores
//! ([`projection_batch`]). Each lane keeps its own [`MtpGpuState`] and draft
//! request state; row `i` of every row stage is the one-token step of lane
//! `i`, so the per-lane bytes equal `n` serial `forward_token_deferred`
//! calls (proved per round by the lab oracle).
//!
//! This module is a child of `mtp_gpu` and uses its private items directly.

use super::*;
use crate::gpu_forward::seal_moe_rows;
use crate::program::Qwen4LayerScratch;

/// Whether batched head rounds are on: `HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`
/// asks for them on any load, `=0` kills them, and unset they are on exactly
/// where the lane route exists: an expert-parallel Qwen4 load
/// (`topology.ep` >= 2, [`hipfire_config::topology::qwen4_ep`]).
pub(crate) fn draft_rounds_enabled() -> bool {
    hipfire_config::developer_bool(
        MTP_DRAFT_ROUNDS_ENV,
        hipfire_config::topology::qwen4_ep().is_ok_and(|ep| ep >= 2),
    )
}

/// `HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS_XBATCH` (default on; `0` ranks every row
/// with the one-row launches): whether a batched round's draft head shares
/// one x-batched GEMV over the ranking copy between its rows.
const MTP_DRAFT_ROUNDS_XBATCH_ENV: &str = "HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS_XBATCH";

/// `HIPFIRE_QWEN4_MTP_DRAFT_PROJ_BATCH` (default on; `0` restores one whole
/// indexed-attention op per lane): see [`projection_batch`].
const MTP_DRAFT_PROJ_BATCH_ENV: &str = "HIPFIRE_QWEN4_MTP_DRAFT_PROJ_BATCH";

/// Whether a batched round of `n` lanes splits the indexed attention into ONE
/// input projection over the `n` rows, each lane's core (its own positions,
/// K/V cache, pool and selection, one row) and ONE output projection over the
/// `n` rows, instead of one whole op per lane. Exact gfx1201 with every QSA
/// projection Q8_0 on the few-row route only: there the `n`-row launches are
/// `gemv_q8_0_rows_exact` (each row bitwise the rows == 1 `gemv_q8_0` the
/// whole op launches, weights read once per call); anything else keeps the
/// whole ops.
fn projection_batch(gpu: &Gpu, qsa: &Qwen4QsaWeights<'_>, n: usize) -> bool {
    use rdna_compute::gemv::{q8_0_rows_exact_route, Q8RowsExactRoute};
    n >= 2
        && gpu.arch_caps.is_gfx1201()
        && (2..=gpu.few_row_limit()).contains(&gpu.route_rows(n))
        && [&qsa.indexer_qk, &qsa.q, &qsa.k, &qsa.v, &qsa.output]
            .iter()
            .all(|w| {
                w.dtype == DType::Q8_0
                    && q8_0_rows_exact_route(&gpu.arch, w.k) != Q8RowsExactRoute::PerRowSingleton
            })
        && hipfire_config::developer_bool(MTP_DRAFT_PROJ_BATCH_ENV, true)
}

/// One drafting lane of a batched head round: the lane's own head arena and
/// draft request state plus this round's input.
pub(crate) struct DraftRow<'a> {
    pub(crate) state: &'a mut MtpGpuState,
    pub(crate) request: &'a mut DraftHeadRequestState,
    pub(crate) token: u32,
    /// The lane's backbone hidden (`wide` F32 elements) at depth 0; `None`
    /// takes the lane's own `wide_hidden`.
    pub(crate) backbone_hidden: Option<&'a GpuTensor>,
    pub(crate) position: usize,
    pub(crate) fresh_qsa_selection: bool,
}

/// Allocations of one [`MtpRowsScratch`], in field order.
const ROWS_SCRATCH_TENSORS: usize = 44;

/// Operator scratch of a batched head round: every one-row head buffer
/// ([`MtpGpuScratch`]) times [`MTP_DRAFT_LANES_MAX`] rows, plus the few-row
/// grouped MoE's grouping buffers and the indexed attention's projection rows
/// (index / q+gate / K / V / attention output, [`projection_batch`]). The
/// lanes' attention cores run one after another on their own rows of those and
/// share the one-row [`MtpGpuScratch`] selection scratch.
pub(crate) struct MtpRowsScratch {
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
    ffn_output: GpuTensor,
    router_logits: GpuTensor,
    moe_x_rot: GpuTensor,
    moe_gate_up: GpuTensor,
    moe_gate: GpuTensor,
    moe_up: GpuTensor,
    moe_hidden: GpuTensor,
    moe_scalar: GpuTensor,
    moe_shared_output: GpuTensor,
    moe_gate_batch: GpuTensor,
    moe_up_batch: GpuTensor,
    moe_rot_batch: GpuTensor,
    moe_topk_indices: GpuTensor,
    moe_topk_weights: GpuTensor,
    /// The shared-down output of an EP call (the singleton scratch's role of
    /// `moe_down_expanded`), and the expanded-row scratch without EP.
    moe_down_expanded: GpuTensor,
    moe_expert_token_counts: GpuTensor,
    moe_expert_offsets: GpuTensor,
    moe_sorted_slot_index: GpuTensor,
    moe_expert_tile_ids: GpuTensor,
    moe_inverse_perm: GpuTensor,
    moe_y_gate_up_grouped: GpuTensor,
    moe_y_down_grouped: GpuTensor,
    /// The batched draft ranking: `rows * vocab` F32 logits and `rows * hidden` F32
    /// rotated hidden rows.
    draft_logits: GpuTensor,
    draft_rotation: GpuTensor,
    /// The indexed attention's rows under [`projection_batch`]: `rows *`
    /// index width, `rows * 2 * q_width`, `rows * kv_width` (K and V) and
    /// `rows * q_width` F32 (the attention output the output projection reads).
    qsa_index: GpuTensor,
    qsa_qgate: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    qsa_output: GpuTensor,
    /// Pinned, event-fenced staging of the round's token upload
    /// ([`upload_token_ids`]); `None` when [`MTP_TOKEN_RING_ENV`] is `0`.
    token_ring: Option<LaneTokenRing>,
}

/// The exact-extent row views a `rows`-lane step needs (HyperNorm infers its
/// row count from the tensor extent) and the lane count they were made for.
struct RowsView {
    token_embedding: GpuTensor,
    embedding_norm: GpuTensor,
    hidden_norm: GpuTensor,
    backbone_hidden: GpuTensor,
    /// `[rows, router_columns]`: the grouped MoE seal checks this shape.
    router_logits: GpuTensor,
}

impl MtpRowsScratch {
    /// Element count and dtype of each tensor, in field order; `None` for a
    /// dense-FFN head (the batched rounds run the MoE head only).
    pub(super) fn shapes(
        config: &Qwen4Config,
    ) -> Result<Option<[(usize, DType); ROWS_SCRATCH_TENSORS]>, MtpGpuError> {
        let config = config.mtp_geometry();
        let rows = MTP_DRAFT_LANES_MAX;
        let hidden = config.hidden_size;
        let wide = config
            .hc_count
            .checked_mul(hidden)
            .ok_or_else(|| invalid("MTP wide dimension overflow"))?;
        let q_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| invalid("MTP Q width overflow"))?;
        let index_width = (config.indexer_n_heads + config.indexer_kv_heads)
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| invalid("MTP index width overflow"))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| invalid("MTP KV width overflow"))?;
        let Qwen4FfnDims::Moe {
            num_experts,
            experts_per_token,
            intermediate,
            shared_intermediate,
        } = program_dims(config).ffn
        else {
            return Ok(None);
        };
        let hc_up = hc_up_elements(rows, wide, config.hc_lowrank)
            .ok_or_else(|| invalid("MTP rows HC up scratch overflow"))?;
        let max_rotation = wide
            .max(hidden)
            .max(config.hc_lowrank)
            .max(q_width)
            .max(intermediate);
        let slots = rows * experts_per_token;
        let grouped = crate::program::grouped_m_total_bound(slots, num_experts)
            .map_err(|error| invalid(error.to_string()))?;
        let routed = slots * intermediate;
        let i32_bytes = std::mem::size_of::<i32>();
        let mut shapes = [
            (rows * i32_bytes, DType::Raw),
            (rows * hidden, DType::F32),
            (rows * hidden, DType::F32),
            (rows * hidden, DType::F32),
            (rows * wide, DType::F32),
            (rows * hidden, DType::F32),
            (rows * wide, DType::F32),
            (rows * wide, DType::F32),
            (rows * wide, DType::F32),
            (rows * wide, DType::F32),
            (rows * config.hc_lowrank, DType::F32),
            (hc_up, DType::F32),
            (rows * hidden, DType::F32),
            (rows * config.hc_count, DType::F32),
            (rows * max_rotation, DType::F32),
            (rows * hidden, DType::F32),
            (rows * num_experts, DType::F32),
            (rows * hidden, DType::F32),
            (rows * 2 * intermediate, DType::F32),
            (rows * shared_intermediate, DType::F32),
            (rows * shared_intermediate, DType::F32),
            (rows * shared_intermediate, DType::F32),
            (rows * shared_intermediate, DType::F32),
            (rows * hidden, DType::F32),
            (routed, DType::F32),
            (routed, DType::F32),
            (routed, DType::F32),
            (slots * i32_bytes, DType::Raw),
            (slots, DType::F32),
            (slots * hidden + hidden.div_ceil(4), DType::F32),
            (num_experts * i32_bytes, DType::Raw),
            ((num_experts + 1) * i32_bytes, DType::Raw),
            (grouped * i32_bytes, DType::Raw),
            ((grouped / 16) * i32_bytes, DType::Raw),
            (slots * i32_bytes, DType::Raw),
            (grouped * 2 * intermediate, DType::F32),
            (grouped * hidden, DType::F32),
            (rows * config.vocab_size, DType::F32),
            (rows * hidden, DType::F32),
            (rows * index_width, DType::F32),
            (rows * 2 * q_width, DType::F32),
            (rows * kv_width, DType::F32),
            (rows * kv_width, DType::F32),
            (rows * q_width, DType::F32),
        ];
        for shape in &mut shapes {
            shape.0 = shape.0.max(1);
        }
        Ok(Some(shapes))
    }

    /// Device bytes [`Self::new`] allocates: zero unless
    /// [`draft_rounds_enabled`] and the head is MoE.
    pub(crate) fn device_bytes(config: &Qwen4Config) -> Option<usize> {
        if !draft_rounds_enabled() {
            return Some(0);
        }
        Some(
            Self::shapes(config)
                .ok()?
                .map_or(0, |shapes| shapes.iter().map(|&(n, d)| n * d.size()).sum()),
        )
    }

    /// [`Self::device_bytes`] per allocation, in field order.
    pub(crate) fn allocation_sizes(config: &Qwen4Config) -> Option<Vec<u64>> {
        if !draft_rounds_enabled() {
            return Some(Vec::new());
        }
        Self::shapes(config)
            .ok()?
            .map_or(Some(Vec::new()), |shapes| {
                shapes
                    .iter()
                    .map(|&(n, d)| u64::try_from(n.checked_mul(d.size())?).ok())
                    .collect()
            })
    }

    /// The rows scratch when batched rounds are requested and the head is MoE.
    pub(crate) fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Option<Self>, MtpGpuError> {
        if !draft_rounds_enabled() {
            return Ok(None);
        }
        let Some(shapes) = Self::shapes(config)? else {
            return Ok(None);
        };
        let mut allocated = Vec::with_capacity(ROWS_SCRATCH_TENSORS);
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
        Ok(Some(Self {
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
            ffn_output: next(),
            router_logits: next(),
            moe_x_rot: next(),
            moe_gate_up: next(),
            moe_gate: next(),
            moe_up: next(),
            moe_hidden: next(),
            moe_scalar: next(),
            moe_shared_output: next(),
            moe_gate_batch: next(),
            moe_up_batch: next(),
            moe_rot_batch: next(),
            moe_topk_indices: next(),
            moe_topk_weights: next(),
            moe_down_expanded: next(),
            moe_expert_token_counts: next(),
            moe_expert_offsets: next(),
            moe_sorted_slot_index: next(),
            moe_expert_tile_ids: next(),
            moe_inverse_perm: next(),
            moe_y_gate_up_grouped: next(),
            moe_y_down_grouped: next(),
            draft_logits: next(),
            draft_rotation: next(),
            qsa_index: next(),
            qsa_qgate: next(),
            qsa_k: next(),
            qsa_v: next(),
            qsa_output: next(),
            token_ring: new_token_ring(MTP_DRAFT_LANES_MAX * std::mem::size_of::<i32>()),
        }))
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        // The ring drains its pending copies before `token_ids` is freed.
        let ring_error = self.token_ring.and_then(|ring| ring.free_gpu(gpu));
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
            self.ffn_output,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.moe_scalar,
            self.moe_shared_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.draft_logits,
            self.draft_rotation,
            self.qsa_index,
            self.qsa_qgate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
        ];
        let mut first = ring_error;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }

    fn view(&self, rows: usize, config: &Qwen4Config) -> Result<RowsView, MtpGpuError> {
        let hidden = config.hidden_size;
        let wide = config.hc_count * hidden;
        Ok(RowsView {
            router_logits: crate::gpu_forward::matrix_view(
                &self.router_logits,
                rows,
                crate::gpu_forward::router_columns(program_dims(config)),
            )?,
            token_embedding: self.token_embedding.sub_offset(0, rows * hidden),
            embedding_norm: self.embedding_norm.sub_offset(0, rows * hidden),
            hidden_norm: self.hidden_norm.sub_offset(0, rows * wide),
            backbone_hidden: self.backbone_hidden.sub_offset(0, rows * wide),
        })
    }

    fn tensors<'a>(&'a self, view: &'a RowsView) -> HeadRowTensors<'a> {
        HeadRowTensors {
            token_ids: &self.token_ids,
            embedding_rot: &self.embedding_rot,
            token_embedding: &view.token_embedding,
            embedding_norm: &view.embedding_norm,
            hidden_norm: &view.hidden_norm,
            projected_embedding: &self.projected_embedding,
            projected_hidden: &self.projected_hidden,
            wide: &self.wide,
            hc_normalized: &self.hc_normalized,
            hc_low: &self.hc_low,
            hc_up: &self.hc_up,
            hc_mixed: &self.hc_mixed,
            hc_gates: &self.hc_gates,
            rotation: &self.rotation,
            ffn_output: &self.ffn_output,
        }
    }

    /// The MoE seal's scratch description: the rows scratch with
    /// `down_expanded` as the expanded-row buffer (the EP gather buffer under
    /// EP). Fields the MoE never reads alias `wide`.
    fn layer_scratch<'a>(
        &'a self,
        down_expanded: &'a GpuTensor,
        router_logits: &'a GpuTensor,
    ) -> Qwen4LayerScratch<'a> {
        let unused = &self.wide;
        Qwen4LayerScratch {
            streams: unused,
            hc_normalized: &self.hc_normalized,
            hc_low: &self.hc_low,
            hc_up: &self.hc_up,
            hc_mixed: &self.hc_mixed,
            hc_gates: &self.hc_gates,
            rotation: &self.rotation,
            projection: unused,
            projection2: unused,
            gdn_a: unused,
            gdn_b: unused,
            gdn_gate: unused,
            gdn_beta: unused,
            gdn_recurrent_output: unused,
            gdn_z: unused,
            gdn_output: unused,
            qsa_index: unused,
            qsa_qgate: unused,
            qsa_k: unused,
            qsa_v: unused,
            qsa_output: unused,
            attention_output: unused,
            ffn_output: &self.ffn_output,
            moe_shared_output: &self.moe_shared_output,
            moe_router_logits: router_logits,
            moe_x_rot: &self.moe_x_rot,
            moe_gate_up: &self.moe_gate_up,
            moe_scalar: &self.moe_scalar,
            moe_gate: &self.moe_gate,
            moe_up: &self.moe_up,
            moe_hidden: &self.moe_hidden,
            moe_gate_batch: &self.moe_gate_batch,
            moe_up_batch: &self.moe_up_batch,
            moe_rot_batch: &self.moe_rot_batch,
            moe_topk_indices: &self.moe_topk_indices,
            moe_topk_weights: &self.moe_topk_weights,
            moe_down_expanded: down_expanded,
            moe_expert_token_counts: &self.moe_expert_token_counts,
            moe_expert_offsets: &self.moe_expert_offsets,
            moe_sorted_slot_index: &self.moe_sorted_slot_index,
            moe_expert_tile_ids: &self.moe_expert_tile_ids,
            moe_inverse_perm: &self.moe_inverse_perm,
            moe_y_gate_up_grouped: &self.moe_y_gate_up_grouped,
            moe_y_down_grouped: &self.moe_y_down_grouped,
        }
    }
}

impl MtpGpuState {
    /// Publish a finished head step's lengths (the tail of
    /// `finish_head_step`, minus the wide-hidden copy).
    pub(super) fn commit_step(
        &mut self,
        (full_len, raw_len, pooled_len, selected_len, next_position): (
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
    ) {
        self.position = next_position;
        self.full_len = full_len;
        self.raw_len = raw_len;
        self.pooled_len = pooled_len;
        self.selected_len = selected_len;
        self.step_index = self.step_index.wrapping_add(1);
    }
}

impl Qwen4MtpGpu {
    /// One Predict head step for `lanes.len()` (2..=[`MTP_DRAFT_LANES_MAX`])
    /// drafting lanes at once, the draft's device work deferred into
    /// `top1_out` (Raw, `n` slots of `Gpu::TOPK8_OUT_BYTES`: lane `i`'s
    /// result in slot `i`, parsed by `draft.draft_finish_request`).
    ///
    /// Equivalent, lane by lane, to `n` calls of `forward_token_deferred`
    /// with each lane's state and request swapped in: the row stages run once
    /// over the `n` rows, the indexed attention once per lane on its own
    /// row, and the lane's epilogue (row `i` of the output wide into the
    /// lane's `wide_hidden`, the state's lengths and step) follows the draft.
    /// Under EP the call is one MoE invocation: the head's gather parity
    /// advances once, on every rank, whatever `n` is.
    ///
    /// The caller maps each lane's QSA context through its position
    /// (`MtpGpuState::ensure_mapped_capacity`), as `mapped_mtp` does for the
    /// one-token step.
    pub(crate) fn forward_rows_deferred(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        lanes: &mut [DraftRow<'_>],
        top1_out: &GpuTensor,
        ep: Option<MtpEpGather<'_>>,
    ) -> Result<(), MtpGpuError> {
        let config = config.mtp_geometry();
        let n = lanes.len();
        if !draft_rounds_enabled() {
            return Err(invalid(format!(
                "batched MTP head rounds need {MTP_DRAFT_ROUNDS_ENV}=1"
            )));
        }
        let min_rows = if self.s4_round.is_some() { 1 } else { 2 };
        if !(min_rows..=MTP_DRAFT_LANES_MAX).contains(&n) {
            return Err(invalid(format!(
                "a batched MTP head round takes {min_rows}..={MTP_DRAFT_LANES_MAX} lanes, got {n}"
            )));
        }
        if top1_out.dtype != DType::Raw || top1_out.byte_size() < n * Gpu::TOPK8_OUT_BYTES {
            return Err(invalid(format!(
                "MTP batched draft output must be Raw with at least {} bytes",
                n * Gpu::TOPK8_OUT_BYTES
            )));
        }
        let Self {
            scratch,
            moe,
            moe_full,
            rows: rows_scratch,
            max_seq,
            draft,
            ep_bound,
            moe_steps,
            lane_local,
            s4_round,
            s4_owners,
            ..
        } = self;
        let s4 = *s4_round;
        if let Some(s4) = s4 {
            if s4.own_rows().len() != n {
                return Err(invalid(format!(
                    "sharded MoE pass: this rank owns {} of the round's rows, the call carries {n}",
                    s4.own_rows().len()
                )));
            }
        }
        let rs = rows_scratch.as_mut().ok_or_else(|| {
            invalid(format!(
                "MTP rows scratch is not allocated: set {MTP_DRAFT_ROUNDS_ENV}=1 (MoE head) before load"
            ))
        })?;
        let mtp = bound_mtp(weights)?;
        // A lane-local call (an owned draft lane) runs the full-table runtime.
        let local = *lane_local && *ep_bound;
        let runtime = match (
            &mtp.ffn,
            if local {
                moe_full.as_ref()
            } else {
                moe.as_ref()
            },
        ) {
            (Qwen4FfnWeights::Moe(_), Some(runtime)) => runtime,
            _ => return Err(invalid("batched MTP head rounds need the MoE head")),
        };
        let hidden = config.hidden_size;
        let wide = config
            .hc_count
            .checked_mul(hidden)
            .ok_or_else(|| invalid("MTP wide dimension overflow"))?;
        let wide_bytes = wide * std::mem::size_of::<f32>();
        // `HIPFIRE_QWEN4_DRAFT_GLUE_FOLD`: the round's launch glue (broadcast adds, staging
        // copies) runs as one launch each; the bytes are the per-lane launches'.
        let glue_fold = gpu.flags.qwen4_draft_glue_fold_enabled();

        // Stage 1 (per lane, host): the position guards and the backbone
        // hidden staging, exactly `head_prologue`'s; then the draft history
        // and ONE H2D of the `n` tokens.
        for (row, lane) in lanes.iter().enumerate() {
            if let Some(backbone) = lane.backbone_hidden {
                if backbone.dtype != DType::F32 || backbone.numel() != wide {
                    return Err(invalid(format!(
                        "MTP backbone hidden must be F32 with {wide} elements"
                    )));
                }
            }
            let state = &*lane.state;
            if lane.position != state.position {
                return Err(invalid(format!(
                    "MTP lane {row} position mismatch: expected {}, got {}",
                    state.position, lane.position
                )));
            }
            if lane.position >= *max_seq {
                return Err(invalid("MTP position exceeds QSA capacity"));
            }
            if lane.position >= state.mapped_tokens {
                return Err(invalid(format!(
                    "MTP lane {row} position {} is past the mapped QSA coverage of {} tokens",
                    lane.position, state.mapped_tokens
                )));
            }
        }
        require_hc_up(&rs.hc_up, n, wide, config.hc_lowrank)?;
        let view = rs.view(n, config)?;
        {
            let targets: SmallVec<[GpuTensor; MTP_DRAFT_LANES_MAX]> = (0..n)
                .map(|row| rs.backbone_hidden.sub_offset(row * wide, wide))
                .collect();
            let pairs: SmallVec<[(&GpuTensor, &GpuTensor, usize); MTP_DRAFT_LANES_MAX]> = lanes
                .iter()
                .zip(&targets)
                .map(|(lane, target)| {
                    (
                        lane.backbone_hidden.unwrap_or(&lane.state.wide_hidden),
                        target,
                        wide_bytes,
                    )
                })
                .collect();
            copy_tensors(gpu, &pairs)?;
        }
        let mut token_bytes = [0u8; MTP_DRAFT_LANES_MAX * std::mem::size_of::<i32>()];
        for (row, lane) in lanes.iter_mut().enumerate() {
            draft.observe_request(lane.request, lane.token);
            token_bytes[row * 4..row * 4 + 4].copy_from_slice(&(lane.token as i32).to_ne_bytes());
        }
        upload_token_ids(
            rs.token_ring.as_mut(),
            gpu,
            &rs.token_ids.buf,
            &token_bytes[..n * 4],
        )?;
        let rs: &MtpRowsScratch = rs;

        // Stages 2-4: the row stages once over all rows, the attention per
        // lane (under [`projection_batch`] only its core: the projections run
        // once over all rows). Everything the program borrows lives in this scope.
        let lengths = {
            let tensors = rs.tensors(&view);
            let one_launch = gpu.flags.qwen4_mtp_fc_one_launch_enabled()
                && weights.resident(&mtp.fc_hidden)?.dtype == DType::BF16
                && gpu.bf16_rows_exact_one_launch(n * config.hc_count);
            let views = HeadRowViews::new(&tensors, config, n, one_launch);
            let ctx = DispatchCtx::new(gpu);
            let attn = hyper_desc(weights, &mtp.attn_hyper)?;
            let mlp = hyper_desc(weights, &mtp.mlp_hyper)?;
            let final_read = hyper_read_desc(weights, &mtp.final_hyper)?;
            let attention_inputs: SmallVec<[GpuTensor; MTP_DRAFT_LANES_MAX]> = (0..n)
                .map(|row| rs.hc_mixed.sub_offset(row * hidden, hidden))
                .collect();
            let attention_outputs: SmallVec<[GpuTensor; MTP_DRAFT_LANES_MAX]> = (0..n)
                .map(|row| rs.projected_embedding.sub_offset(row * hidden, hidden))
                .collect();
            let batch = projection_batch(gpu, &qsa_desc(weights, &mtp.attention)?, n);
            // Each lane's rows of the batched projections' buffers (row-major
            // `[n, width]`): index, q+gate, K, V, attention output.
            let lane_rows: SmallVec<[[GpuTensor; 5]; MTP_DRAFT_LANES_MAX]> = if batch {
                let index_width =
                    (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
                let q_width = config.num_attention_heads * config.head_dim;
                let kv_width = config.num_key_value_heads * config.head_dim;
                (0..n)
                    .map(|row| {
                        [
                            rs.qsa_index.sub_offset(row * index_width, index_width),
                            rs.qsa_qgate.sub_offset(row * 2 * q_width, 2 * q_width),
                            rs.qsa_k.sub_offset(row * kv_width, kv_width),
                            rs.qsa_v.sub_offset(row * kv_width, kv_width),
                            rs.qsa_output.sub_offset(row * q_width, q_width),
                        ]
                    })
                    .collect()
            } else {
                SmallVec::new()
            };
            let batch_input = rs.hc_mixed.sub_offset(0, n * hidden);
            let batch_output = rs.projected_embedding.sub_offset(0, n * hidden);
            let mut attentions: SmallVec<[IndexedAttentionOp<'_>; MTP_DRAFT_LANES_MAX]> =
                SmallVec::new();
            let mut lengths: SmallVec<[(usize, usize, usize, usize, usize); MTP_DRAFT_LANES_MAX]> =
                SmallVec::new();
            for (row, lane) in lanes.iter().enumerate() {
                let state = &*lane.state;
                let qsa = qsa_desc(weights, &mtp.attention)?;
                let mode = if lane.fresh_qsa_selection {
                    IndexedAttentionMode::Full
                } else {
                    IndexedAttentionMode::ReuseSelection {
                        selected_len_out: &state.selected_len_out,
                    }
                };
                let mut attention = head_attention_op(
                    config,
                    scratch,
                    state,
                    qsa,
                    mode,
                    &attention_inputs[row],
                    &attention_outputs[row],
                );
                if let Some([index, qgate, k, v, output]) = lane_rows.get(row) {
                    attention.phase = MixerPhase::Core;
                    attention.index_scratch = index;
                    attention.qgate_scratch = qgate;
                    attention.k_scratch = k;
                    attention.v_scratch = v;
                    attention.qsa_output = output;
                }
                lengths.push(attention.next_lengths()?);
                attentions.push(attention);
            }
            // The batched input / output projections: lane 0's state rides
            // along only to fill the op (the projection phases never read it).
            let projection = |phase: MixerPhase| {
                let mut op = head_attention_op(
                    config,
                    scratch,
                    &*lanes[0].state,
                    qsa_desc(weights, &mtp.attention)?,
                    IndexedAttentionMode::Full,
                    &batch_input,
                    &batch_output,
                );
                op.rows = n;
                op.phase = phase;
                op.index_scratch = &rs.qsa_index;
                op.qgate_scratch = &rs.qsa_qgate;
                op.k_scratch = &rs.qsa_k;
                op.v_scratch = &rs.qsa_v;
                op.qsa_output = &rs.qsa_output;
                op.rotation = &rs.rotation;
                Ok::<_, MtpGpuError>(op)
            };
            let projections = if batch {
                Some((
                    projection(MixerPhase::InProj)?,
                    projection(MixerPhase::OutProj)?,
                ))
            } else {
                None
            };
            // Under EP the batched call is one MoE invocation: the gather
            // buffer `2 + j % 2` and layer index `trunk_layers + j % 2` for
            // the head's monotonic invocation counter `j`, advanced once. A
            // lane-local call keeps the layer index and buffer (every buffer
            // role of the EP call) with the no-op row gather and leaves `j`
            // alone, so replicated calls keep their parity in lockstep.
            let mut local_hook: Option<LaneLocalRowGather<'_>> = None;
            // A sharded-MoE pass (S4) runs its MoE as the replicated call over
            // all the round's rows after the pre-MoE steps (below): no MoE is
            // sealed or counted here.
            let moe_ep: Option<(usize, &GpuTensor, &dyn MoeRowGather)> = if *ep_bound
                && s4.is_none()
            {
                let ep = ep
                    .ok_or_else(|| invalid("an EP MTP head step needs the forward's row gather"))?;
                let parity = (*moe_steps % 2) as usize;
                let hook: &dyn MoeRowGather = if local {
                    local_hook.insert(LaneLocalRowGather { ep: Some(ep.hook) })
                } else {
                    *moe_steps = moe_steps.wrapping_add(1);
                    ep.hook as &dyn MoeRowGather
                };
                Some((
                    config.num_hidden_layers + parity,
                    ep.buffers[2 + parity],
                    hook,
                ))
            } else {
                None
            };
            let layer_scratch = rs.layer_scratch(
                moe_ep.map_or(&rs.moe_down_expanded, |(_, buffer, _)| buffer),
                &view.router_logits,
            );
            let sealed = if s4.is_none() {
                Some(seal_moe_rows(
                    &ctx,
                    config,
                    moe_ep.map_or(0, |(layer, _, _)| layer),
                    runtime,
                    n,
                    &layer_scratch,
                    moe_ep.map(|(_, _, hook)| (hook, &rs.moe_down_expanded)),
                )?)
            } else {
                None
            };
            let mut steps: SmallVec<[Step<'_>; 16]> = SmallVec::new();
            push_head_rows_front(
                &mut steps,
                weights,
                config,
                mtp,
                &tensors,
                &views,
                n,
                glue_fold,
                &view.backbone_hidden,
                &attn.read,
            )?;
            let (in_proj, out_proj) = projections.map_or((None, None), |(i, o)| (Some(i), Some(o)));
            steps.extend(in_proj.map(Step::IndexedAttention));
            for attention in attentions {
                steps.push(Step::IndexedAttention(attention));
            }
            steps.extend(out_proj.map(Step::IndexedAttention));
            if let Some(s4) = s4 {
                // Steps up to the MLP HC read, then the activation all-gather and
                // the global MoE pass, then the steps after the FFN.
                push_head_rows_pre_ffn(&mut steps, config, &tensors, n, &attn.write, &mlp.read);
                validate_steps(gpu, &steps)?;
                execute_validated_steps(gpu, &ctx, &steps)?;
                let ep =
                    ep.ok_or_else(|| invalid("a sharded MoE pass needs the forward's row gather"))?;
                let owners = s4_owners
                    .as_ref()
                    .ok_or_else(|| invalid("a sharded MoE pass needs the slot-owner table"))?;
                Self::s4_moe_pass(gpu, config, runtime, rs, owners, ep, moe_steps, s4, n)?;
                let mut post: SmallVec<[Step<'_>; 16]> = SmallVec::new();
                push_head_rows_post_ffn(
                    &mut post,
                    config,
                    &tensors,
                    n,
                    &mlp.write,
                    &final_read,
                    true,
                );
                validate_steps(gpu, &post)?;
                execute_validated_steps(gpu, &ctx, &post)?;
                #[cfg(feature = "lab")]
                lab_expert_trace::record(
                    gpu,
                    &rs.moe_topk_indices,
                    s4.g,
                    config.num_experts_per_tok,
                    weights.ep_placement(),
                    false,
                )?;
            } else {
                let sealed = sealed
                    .ok_or_else(|| invalid("a batched head round without a sealed MoE call"))?;
                push_head_rows_back(
                    &mut steps,
                    config,
                    &tensors,
                    n,
                    &attn.write,
                    &mlp.read,
                    &mlp.write,
                    &final_read,
                    Some(MtpFfnCall::Moe(sealed)),
                    true,
                );
                validate_steps(gpu, &steps)?;
                execute_validated_steps(gpu, &ctx, &steps)?;
                #[cfg(feature = "lab")]
                lab_expert_trace::record(
                    gpu,
                    &rs.moe_topk_indices,
                    n,
                    config.num_experts_per_tok,
                    weights.ep_placement(),
                    local,
                )?;
            }
            lengths
        };

        // Stage 5: the draft head over the `n` collapsed rows, then each
        // lane's epilogue, in `forward_token_deferred`'s order.
        let head = weights.resident(&weights.root.lm_head)?;
        {
            let mut requests: SmallVec<[&mut DraftHeadRequestState; MTP_DRAFT_LANES_MAX]> =
                lanes.iter_mut().map(|lane| &mut *lane.request).collect();
            // `HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS_XBATCH=0` ranks each row with the
            // one-row launches (bisection switch; the bytes are the same).
            let batch = hipfire_config::developer_bool(MTP_DRAFT_ROUNDS_XBATCH_ENV, true)
                .then_some((&rs.draft_logits, &rs.draft_rotation));
            draft.draft_rows_enqueue(gpu, head, &rs.hc_mixed, &mut requests, top1_out, batch)?;
        }
        {
            let sources: SmallVec<[GpuTensor; MTP_DRAFT_LANES_MAX]> = (0..n)
                .map(|row| rs.wide.sub_offset(row * wide, wide))
                .collect();
            let pairs: SmallVec<[(&GpuTensor, &GpuTensor, usize); MTP_DRAFT_LANES_MAX]> = lanes
                .iter()
                .zip(&sources)
                .map(|(lane, source)| (source, &lane.state.wide_hidden, wide_bytes))
                .collect();
            copy_tensors(gpu, &pairs)?;
        }
        for (row, lane) in lanes.iter_mut().enumerate() {
            lane.state.commit_step(lengths[row]);
        }
        Ok(())
    }
}

/// Copy rows `(src_row, dst_row)` between two row-major tensors of `row_bytes`
/// rows (distinct allocations): every maximal run of constant source and
/// destination strides is one pitched device copy.
fn copy_row_pairs(
    gpu: &Gpu,
    src: &GpuTensor,
    dst: &GpuTensor,
    pairs: &[(usize, usize)],
    row_bytes: usize,
) -> Result<(), MtpGpuError> {
    let mut at = 0;
    while at < pairs.len() {
        let (src0, dst0) = pairs[at];
        let (src_stride, dst_stride) = match pairs.get(at + 1) {
            Some(&(s, d)) if s > src0 && d > dst0 => (s - src0, d - dst0),
            _ => (1, 1),
        };
        let mut len = 1;
        while let Some(&(s, d)) = pairs.get(at + len) {
            let (prev_s, prev_d) = pairs[at + len - 1];
            if s.checked_sub(prev_s) == Some(src_stride)
                && d.checked_sub(prev_d) == Some(dst_stride)
            {
                len += 1;
            } else {
                break;
            }
        }
        gpu.memcpy_dtod_2d_at_ordered_async(
            &dst.buf,
            dst0 * row_bytes,
            dst_stride * row_bytes,
            &src.buf,
            src0 * row_bytes,
            src_stride * row_bytes,
            row_bytes,
            len,
        )?;
        at += len;
    }
    Ok(())
}

impl Qwen4MtpGpu {
    /// The MoE of one sharded draft pass (S4), the same call on every rank
    /// whatever its own row count: the rank's own rows (`own_rows` of them, rows
    /// `0..own_rows` of the rows scratch's MLP HC read output) are copied into
    /// an EP gather buffer at their global slots, all-gathered through the
    /// forward's row-gather hook (`gather_rows` over slot owners `ExpertModulo`
    /// on the round's owner table: the unchanged bit-exact owner-slot copy), and
    /// the replicated EP head MoE call (layer index and buffer `2 + parity`,
    /// `moe_steps` advanced once) runs over the `s4.g` global rows; this rank's
    /// rows of its output are copied to the rows scratch's FFN output.
    ///
    /// The staging buffer is the first of gather buffers 0 and 1 that is not the
    /// rank's previous invocation's, so the transport's no-same-buffer-twice
    /// hazard rule holds for the activation crossing and for the MoE crossing
    /// that follows it (buffer `2 + parity`).
    #[allow(clippy::too_many_arguments)]
    fn s4_moe_pass(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        runtime: &Qwen4MoeLayerRuntime,
        rs: &MtpRowsScratch,
        owners: &GpuTensor,
        ep: MtpEpGather<'_>,
        moe_steps: &mut u64,
        s4: S4Round,
        own_rows: usize,
    ) -> Result<(), MtpGpuError> {
        let hidden = config.hidden_size;
        let row_bytes = hidden * std::mem::size_of::<f32>();
        let g = s4.g;
        let own = s4.own_rows();
        if own.len() != own_rows || s4.rank >= 2 || !(2..=MTP_DRAFT_LANES_MAX).contains(&g) {
            return Err(invalid(format!(
                "sharded MoE pass of {g} rows: rank {} owns {} rows, the call holds {own_rows}",
                s4.rank,
                own.len()
            )));
        }
        let stage = ep.buffers[usize::from(ep.hook.last_gather_buffer() == Some(0))];
        let parity = (*moe_steps % 2) as usize;
        let layer = config.num_hidden_layers + parity;
        let pack: SmallVec<[(usize, usize); MTP_DRAFT_LANES_MAX]> =
            own.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        copy_row_pairs(gpu, &rs.hc_mixed, stage, &pack, row_bytes)?;
        let offset = usize::from(s4.owners) * 8 * std::mem::size_of::<i32>();
        let slot_owners = GpuTensor {
            buf: owners.buf.byte_view(offset, 8 * std::mem::size_of::<i32>()),
            shape: vec![8],
            dtype: DType::Raw,
        };
        ep.hook
            .gather_rows(gpu, layer as u16, g, stage, &slot_owners)
            .map_err(|error| invalid(format!("sharded MoE activation gather: {error}")))?;
        let x = stage.sub_offset(0, g * hidden);
        let view = rs.view(g, config)?;
        let mut layer_scratch = rs.layer_scratch(ep.buffers[2 + parity], &view.router_logits);
        layer_scratch.hc_mixed = &x;
        layer_scratch.ffn_output = &rs.projected_embedding;
        *moe_steps = moe_steps.wrapping_add(1);
        let ctx = DispatchCtx::new(gpu);
        let sealed = seal_moe_rows(
            &ctx,
            config,
            layer,
            runtime,
            g,
            &layer_scratch,
            Some((ep.hook as &dyn MoeRowGather, &rs.moe_down_expanded)),
        )?;
        let mut steps: SmallVec<[Step<'_>; 16]> = SmallVec::new();
        steps.push(Step::Clear(ClearOp {
            tensor: &rs.projected_embedding,
            elements: g * hidden,
        }));
        steps.push(Step::Moe(sealed));
        validate_steps(gpu, &steps)?;
        execute_validated_steps(gpu, &ctx, &steps)?;
        let back: SmallVec<[(usize, usize); MTP_DRAFT_LANES_MAX]> =
            own.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        copy_row_pairs(
            gpu,
            &rs.projected_embedding,
            &rs.ffn_output,
            &back,
            row_bytes,
        )
    }

    /// A sharded draft pass on a rank that owns none of its rows: the
    /// activation all-gather and the global MoE call with an empty own set
    /// (the round's `S4Round` must be set). Writes no head state.
    pub(crate) fn s4_participate(
        &mut self,
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        config: &Qwen4Config,
        ep: Option<MtpEpGather<'_>>,
    ) -> Result<(), MtpGpuError> {
        let config = config.mtp_geometry();
        let Some(s4) = self.s4_round else {
            return Err(invalid(
                "a sharded MoE participant call outside a sharded pass",
            ));
        };
        let mtp = bound_mtp(weights)?;
        let Self {
            moe,
            rows,
            moe_steps,
            s4_owners,
            ..
        } = self;
        let runtime = match (&mtp.ffn, moe.as_ref()) {
            (Qwen4FfnWeights::Moe(_), Some(runtime)) => runtime,
            _ => return Err(invalid("a sharded MoE pass needs the MoE head")),
        };
        let rs = rows
            .as_ref()
            .ok_or_else(|| invalid("a sharded MoE pass needs the rows scratch"))?;
        let owners = s4_owners
            .as_ref()
            .ok_or_else(|| invalid("a sharded MoE pass needs the slot-owner table"))?;
        let ep = ep.ok_or_else(|| invalid("a sharded MoE pass needs the forward's row gather"))?;
        Self::s4_moe_pass(gpu, config, runtime, rs, owners, ep, moe_steps, s4, 0)?;
        #[cfg(feature = "lab")]
        lab_expert_trace::record(
            gpu,
            &rs.moe_topk_indices,
            s4.g,
            config.num_experts_per_tok,
            weights.ep_placement(),
            false,
        )?;
        Ok(())
    }
}
