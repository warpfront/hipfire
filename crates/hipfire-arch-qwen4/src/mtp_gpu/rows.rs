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
//! on that lane's row. Each lane keeps its own [`MtpGpuState`] and draft
//! request state; row `i` of every row stage is the one-token step of lane
//! `i`, so the per-lane bytes equal `n` serial `forward_token_deferred`
//! calls (proved per round by the lab oracle).
//!
//! This module is a child of `mtp_gpu` and uses its private items directly.

use super::*;
use crate::gpu_forward::seal_moe_rows;
use crate::program::Qwen4LayerScratch;

/// Whether batched head rounds are requested (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`).
pub(crate) fn draft_rounds_enabled() -> bool {
    hipfire_config::developer_bool(MTP_DRAFT_ROUNDS_ENV, false)
}

/// `HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS_XBATCH` (default on; `0` ranks every row
/// with the one-row launches): whether a batched round's draft head shares
/// one x-batched GEMV over the ranking copy between its rows.
const MTP_DRAFT_ROUNDS_XBATCH_ENV: &str = "HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS_XBATCH";

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
const ROWS_SCRATCH_TENSORS: usize = 39;

/// Operator scratch of a batched head round: every one-row head buffer
/// ([`MtpGpuScratch`]) times [`MTP_DRAFT_LANES_MAX`] rows, plus the few-row
/// grouped MoE's grouping buffers. The indexed attention keeps using the
/// one-row [`MtpGpuScratch`] (lanes' attention runs one after another).
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
    fn shapes(
        config: &Qwen4Config,
    ) -> Result<Option<[(usize, DType); ROWS_SCRATCH_TENSORS]>, MtpGpuError> {
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
        }))
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
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
        ];
        let mut first = None;
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
        let n = lanes.len();
        if !draft_rounds_enabled() {
            return Err(invalid(format!(
                "batched MTP head rounds need {MTP_DRAFT_ROUNDS_ENV}=1"
            )));
        }
        if !(2..=MTP_DRAFT_LANES_MAX).contains(&n) {
            return Err(invalid(format!(
                "a batched MTP head round takes 2..={MTP_DRAFT_LANES_MAX} lanes, got {n}"
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
            rows: rows_scratch,
            max_seq,
            draft,
            ep_bound,
            moe_steps,
            ..
        } = self;
        let rs = rows_scratch.as_ref().ok_or_else(|| {
            invalid(format!(
                "MTP rows scratch is not allocated: set {MTP_DRAFT_ROUNDS_ENV}=1 (MoE head) before load"
            ))
        })?;
        let mtp = bound_mtp(weights)?;
        let runtime = match (&mtp.ffn, moe.as_ref()) {
            (Qwen4FfnWeights::Moe(_), Some(runtime)) => runtime,
            _ => return Err(invalid("batched MTP head rounds need the MoE head")),
        };
        let hidden = config.hidden_size;
        let wide = config
            .hc_count
            .checked_mul(hidden)
            .ok_or_else(|| invalid("MTP wide dimension overflow"))?;
        let wide_bytes = wide * std::mem::size_of::<f32>();

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
        for (row, lane) in lanes.iter().enumerate() {
            let source = lane.backbone_hidden.unwrap_or(&lane.state.wide_hidden);
            let target = rs.backbone_hidden.sub_offset(row * wide, wide);
            gpu.copy_d2d(source, &target, wide_bytes)?;
        }
        let mut token_bytes = [0u8; MTP_DRAFT_LANES_MAX * std::mem::size_of::<i32>()];
        for (row, lane) in lanes.iter_mut().enumerate() {
            draft.observe_request(lane.request, lane.token);
            token_bytes[row * 4..row * 4 + 4].copy_from_slice(&(lane.token as i32).to_ne_bytes());
        }
        gpu.memcpy_htod_auto(&rs.token_ids.buf, &token_bytes[..n * 4])?;

        // Stages 2-4: the row stages once over all rows, the attention per
        // lane. Everything the program borrows lives in this scope.
        let lengths = {
            let tensors = rs.tensors(&view);
            let views = HeadRowViews::new(&tensors, config, n);
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
                let attention = head_attention_op(
                    config,
                    scratch,
                    state,
                    qsa,
                    mode,
                    &attention_inputs[row],
                    &attention_outputs[row],
                );
                lengths.push(attention.next_lengths()?);
                attentions.push(attention);
            }
            // Under EP the batched call is one MoE invocation: the gather
            // buffer `2 + j % 2` and layer index `trunk_layers + j % 2` for
            // the head's monotonic invocation counter `j`, advanced once.
            let moe_ep: Option<(usize, &GpuTensor, &dyn MoeRowGather)> = if *ep_bound {
                let ep = ep
                    .ok_or_else(|| invalid("an EP MTP head step needs the forward's row gather"))?;
                let parity = (*moe_steps % 2) as usize;
                *moe_steps = moe_steps.wrapping_add(1);
                Some((
                    config.num_hidden_layers + parity,
                    ep.buffers[2 + parity],
                    ep.hook as &dyn MoeRowGather,
                ))
            } else {
                None
            };
            let layer_scratch = rs.layer_scratch(
                moe_ep.map_or(&rs.moe_down_expanded, |(_, buffer, _)| buffer),
                &view.router_logits,
            );
            let sealed = seal_moe_rows(
                &ctx,
                config,
                moe_ep.map_or(0, |(layer, _, _)| layer),
                runtime,
                n,
                &layer_scratch,
                moe_ep.map(|(_, _, hook)| (hook, &rs.moe_down_expanded)),
            )?;
            let mut steps: SmallVec<[Step<'_>; 16]> = SmallVec::new();
            push_head_rows_front(
                &mut steps,
                weights,
                config,
                mtp,
                &tensors,
                &views,
                n,
                &view.backbone_hidden,
                &attn.read,
            )?;
            for attention in attentions {
                steps.push(Step::IndexedAttention(attention));
            }
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
        for (row, lane) in lanes.iter_mut().enumerate() {
            let source = rs.wide.sub_offset(row * wide, wide);
            gpu.copy_d2d(&source, &lane.state.wide_hidden, wide_bytes)?;
            lane.state.commit_step(lengths[row]);
        }
        Ok(())
    }
}
