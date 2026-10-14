// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 typed program descriptors and shared MoE binding.
//!
//! The architecture crate binds resident weights, state tensors, and scratch
//! views into these descriptors.  The shared dispatch crate owns operation
//! lowering and execution; this module only supplies family-owned typed data
//! and the source-bound MoE sealer inputs.

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoePrefillParams, MoePrefillPrelude,
    MoeQ8RouterPolicy, MoeRecipe, MoeRoutePolicy, MoeSharedDecode, MoeSharedDtypes,
    MoeSharedPrefill, MoeSharedWeights, RoutedExpertWeights,
};
use hipfire_dispatch::pipeline::sealed_moe::PrefillRouteMode;
use hipfire_dispatch::pipeline::{
    execute_final_hyper as execute_shared_final_hyper, project_weight, seal_decode, seal_prefill,
    BoundMoeExperts, ExpertBindingCache, ExpertTable, HyperReadOp,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor};

#[inline]
fn hip<T>(result: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    result.map_err(|error| DispatchError::Hip(error.to_string()))
}

#[inline]
fn view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

#[inline]
fn checked_mul(a: usize, b: usize, label: &'static str) -> Result<usize, DispatchError> {
    a.checked_mul(b)
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {label} scratch overflows")))
}

/// FFN extents of a program: dense gated MLP or the shared-expert MoE.
#[derive(Clone, Copy, Debug)]
pub enum Qwen4FfnDims {
    Dense {
        intermediate: usize,
    },
    Moe {
        num_experts: usize,
        experts_per_token: usize,
        intermediate: usize,
        shared_intermediate: usize,
    },
}

/// RoPE base theta hard-coded by the QSA RoPE kernels (the `indexed_attention_*`
/// rotations in `kernels/src/tensor_ops.hip` and the NPU-trunk arena twin in
/// `kernels/src/npu_trunk_consumers.hip` use 10,000,000; YaRN scaling is a
/// separate kernel argument).  A profile whose `rope_theta` differs cannot run
/// full-attention layers on them.
pub const QWEN4_QSA_KERNEL_ROPE_THETA: f64 = 10_000_000.0;

/// Dimensions used by all Qwen4 layer-major operations.  The architecture
/// binds these from its validated config; no family-specific config type
/// crosses the dispatch boundary.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ProgramDims {
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub ffn: Qwen4FfnDims,
    /// PLE short-convolution taps; 0 for a profile without PLE (the PLE
    /// operation validates its own extent when it runs).
    pub ple_conv_kernel_dim: usize,
    pub norm_eps: f32,
}

impl Qwen4ProgramDims {
    #[inline]
    pub fn wide(self) -> usize {
        self.hc_count * self.hidden
    }
    #[inline]
    pub fn q_width(self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    #[inline]
    pub fn kv_width(self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    #[inline]
    pub fn gdn_qk(self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    #[inline]
    pub fn gdn_value(self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    #[inline]
    pub fn gdn_qkv(self) -> usize {
        2 * self.gdn_qk() + self.gdn_value()
    }
    #[inline]
    pub fn index_width(self) -> usize {
        (self.indexer_n_heads + self.indexer_kv_heads) * self.indexer_head_dim
    }
}
/// Central scratch geometry plan. Allocation remains family-owned, but every
/// extent used by the family binder comes from this checked plan.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ScratchLayout {
    pub rows: usize,
    pub hidden: usize,
    pub wide: usize,
    pub hc_low: usize,
    pub q_width: usize,
    pub kv_width: usize,
    pub index_width: usize,
    pub gdn_qkv: usize,
    pub gdn_value: usize,
    pub slots: usize,
    pub grouped_rows: usize,
    pub shared_rows: usize,
    pub routed_rows: usize,
    /// `rows * intermediate` of a dense FFN (zero for MoE).
    pub dense_rows: usize,
    pub expanded_rows: usize,
}

impl Qwen4ScratchLayout {
    pub fn for_rows(dims: Qwen4ProgramDims, rows: usize) -> Result<Self, DispatchError> {
        validate_dims(dims)?;
        if rows == 0 {
            return Err(DispatchError::Hip("Qwen4 scratch rows are zero".into()));
        }
        let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
        // MoE extents are zero for a dense FFN; `dense_rows` is zero for MoE.
        let (slots, grouped_rows, shared_rows, routed_rows, dense_rows) = match dims.ffn {
            Qwen4FfnDims::Moe {
                num_experts,
                experts_per_token,
                intermediate,
                shared_intermediate,
            } => {
                let slots = checked_mul(rows, experts_per_token, "MoE slots")?;
                (
                    slots,
                    grouped_m_total_bound(slots, num_experts)?,
                    checked_mul(rows, shared_intermediate, "shared rows")?,
                    checked_mul(slots, intermediate, "routed rows")?,
                    0,
                )
            }
            Qwen4FfnDims::Dense { intermediate } => {
                (0, 0, 0, 0, checked_mul(rows, intermediate, "dense rows")?)
            }
        };
        Ok(Self {
            rows,
            hidden: dims.hidden,
            wide,
            hc_low: checked_mul(rows, dims.hc_lowrank, "HC low")?,
            q_width: checked_mul(dims.num_attention_heads, dims.head_dim, "Q width")?,
            kv_width: checked_mul(dims.num_key_value_heads, dims.head_dim, "KV width")?,
            index_width: dims.index_width(),
            gdn_qkv: dims.gdn_qkv(),
            gdn_value: dims.gdn_value(),
            slots,
            grouped_rows,
            shared_rows,
            routed_rows,
            dense_rows,
            expanded_rows: checked_mul(slots, dims.hidden, "expanded route")?,
        })
    }
}

/// Read-only hyper-connection operands used by HC read/final collapse.
///
/// The final trunk mixer has exactly this shape; it intentionally has no
/// block-inject operand because final HC is a read, not a write.
pub struct Qwen4HyperReadWeights<'a> {
    pub norm: &'a GpuTensor,
    pub input_mix_down: WeightRef<'a>,
    pub input_mix_up: WeightRef<'a>,
}

/// Write-side hyper-connection operands used by layer HC writes.
pub struct Qwen4HyperWriteWeights<'a> {
    pub norm: &'a GpuTensor,
    pub block_inject: WeightRef<'a>,
}

/// Complete per-layer hyper-connection descriptor.
pub struct Qwen4HyperWeights<'a> {
    pub read: Qwen4HyperReadWeights<'a>,
    pub write: Qwen4HyperWriteWeights<'a>,
}

pub struct Qwen4GdnWeights<'a> {
    pub qkv: WeightRef<'a>,
    pub conv: &'a GpuTensor,
    pub in_proj_a: WeightRef<'a>,
    pub in_proj_b: WeightRef<'a>,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub z: WeightRef<'a>,
    pub norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
}

pub struct Qwen4QsaWeights<'a> {
    pub indexer_qk: WeightRef<'a>,
    pub indexer_q_norm: &'a GpuTensor,
    pub indexer_k_norm: &'a GpuTensor,
    pub q: WeightRef<'a>,
    pub k: WeightRef<'a>,
    pub v: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
}

pub enum Qwen4AttentionWeights<'a> {
    Linear(Qwen4GdnWeights<'a>),
    Full(Qwen4QsaWeights<'a>),
}

pub struct Qwen4LayerDescription<'a> {
    pub attn_hyper: Qwen4HyperWeights<'a>,
    pub mlp_hyper: Qwen4HyperWeights<'a>,
    pub attention: Qwen4AttentionWeights<'a>,
    pub ffn: Qwen4FfnBinding<'a>,
}

/// Dense FFN projections, source-bound once by the family and consumed by
/// `Step::DenseFfn`.
#[derive(Clone, Copy)]
pub struct Qwen4DenseBinding<'a> {
    pub gate: WeightRef<'a>,
    pub up: WeightRef<'a>,
    pub down: WeightRef<'a>,
    pub intermediate: usize,
}

/// The FFN binding of one layer.
#[derive(Clone, Copy)]
pub enum Qwen4FfnBinding<'a> {
    Dense(Qwen4DenseBinding<'a>),
    Moe(Qwen4MoeBinding<'a>),
}

/// MoE resources are source-bound once by the family and consumed by the
/// shared prefill sealer/lowering. The dispatch layer owns all route policy.
#[derive(Clone, Copy)]
pub struct Qwen4MoeBinding<'a> {
    pub table: &'a ExpertTable,
    pub cache: &'a ExpertBindingCache,
    pub routed_experts: &'a dyn RoutedExpertWeights,
    pub router: WeightRef<'a>,
    pub shared: MoeSharedWeights<'a>,
    pub route_policy: MoeRoutePolicy,
    pub intermediate: usize,
    pub experts_all_gate_up_mq4: bool,
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub expert_stage_ptrs: Option<hipfire_dispatch::families::moe::MoeStageTables<'a>>,
    /// Host entries the pointer tables above were uploaded from, so dispatch can
    /// prove the table contents name the live experts (a retained body requires it).
    pub expert_gate_up_entries: &'a [usize],
    pub expert_down_entries: &'a [usize],
    pub layer_idx: u16,
    pub norm_topk_prob: bool,
    /// Expert-parallel row exchange (`None` outside EP): installed into the
    /// one-row decode MoE and the few-row grouped prefill MoE so non-owned
    /// expanded rows are filled from their owners.
    pub row_gather: Option<&'a dyn hipfire_dispatch::families::moe::MoeRowGather>,
    /// Few-row EP prefill only: where the unfused shared down writes its F32
    /// rows (the layer's `moe_down_expanded` is then a gather buffer). `None`
    /// everywhere else.
    pub shared_down_out: Option<&'a GpuTensor>,
    /// Owner-routed EP chunk exchange (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`): installed
    /// into the chunk (prefill) MoE of a layer whose compact table runs the
    /// symmetric IU4 route on the owned tiles only. `None` everywhere else.
    pub tile_exchange: Option<hipfire_dispatch::families::moe::MoeTileExchangeParams<'a>>,
}

/// Family-owned reusable views into the central program's scratch layout.
/// Every field is a borrowed subview; this struct performs no allocation.
pub struct Qwen4LayerScratch<'a> {
    pub streams: &'a GpuTensor,
    pub hc_normalized: &'a GpuTensor,
    pub hc_low: &'a GpuTensor,
    pub hc_up: &'a GpuTensor,
    pub hc_mixed: &'a GpuTensor,
    pub hc_gates: &'a GpuTensor,
    pub rotation: &'a GpuTensor,
    pub projection: &'a GpuTensor,
    pub projection2: &'a GpuTensor,
    pub gdn_a: &'a GpuTensor,
    pub gdn_b: &'a GpuTensor,
    pub gdn_gate: &'a GpuTensor,
    pub gdn_beta: &'a GpuTensor,
    pub gdn_recurrent_output: &'a GpuTensor,
    pub gdn_bf16: &'a GpuTensor,
    pub gdn_z: &'a GpuTensor,
    pub gdn_output: &'a GpuTensor,
    pub qsa_index: &'a GpuTensor,
    pub qsa_qgate: &'a GpuTensor,
    pub qsa_k: &'a GpuTensor,
    pub qsa_v: &'a GpuTensor,
    pub qsa_output: &'a GpuTensor,
    pub attention_output: &'a GpuTensor,
    pub ffn_output: &'a GpuTensor,
    pub moe_shared_output: &'a GpuTensor,
    pub moe_router_logits: &'a GpuTensor,
    pub moe_x_rot: &'a GpuTensor,
    pub moe_gate_up: &'a GpuTensor,
    pub moe_scalar: &'a GpuTensor,
    pub moe_gate: &'a GpuTensor,
    pub moe_up: &'a GpuTensor,
    pub moe_hidden: &'a GpuTensor,
    pub moe_gate_batch: &'a GpuTensor,
    pub moe_up_batch: &'a GpuTensor,
    pub moe_rot_batch: &'a GpuTensor,
    pub moe_topk_indices: &'a GpuTensor,
    pub moe_topk_weights: &'a GpuTensor,
    pub moe_down_expanded: &'a GpuTensor,
    pub moe_expert_token_counts: &'a GpuTensor,
    pub moe_expert_offsets: &'a GpuTensor,
    pub moe_sorted_slot_index: &'a GpuTensor,
    pub moe_expert_tile_ids: &'a GpuTensor,
    pub moe_inverse_perm: &'a GpuTensor,
    pub moe_y_gate_up_grouped: &'a GpuTensor,
    pub moe_y_down_grouped: &'a GpuTensor,
}

pub struct Qwen4PleWeights<'a> {
    pub key: WeightRef<'a>,
    pub value: WeightRef<'a>,
    pub norm_key: &'a GpuTensor,
    pub norm_query: &'a GpuTensor,
    pub norm_conv: &'a GpuTensor,
    pub conv: &'a GpuTensor,
}

/// Return the tile-aligned grouped scratch bound for a bounded chunk.
pub fn grouped_m_total_bound(total_slots: usize, n_exp: usize) -> Result<usize, DispatchError> {
    let live = total_slots.min(n_exp);
    let padded = total_slots
        .checked_add(checked_mul(
            live,
            hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M - 1,
            "grouped MoE",
        )?)
        .ok_or_else(|| DispatchError::Hip("Qwen4 grouped MoE scratch overflows".into()))?;
    Ok(
        padded.div_ceil(hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M)
            * hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M,
    )
}

pub fn execute_final_hyper(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperReadWeights<'_>,
    streams: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
    state_bf16: bool,
) -> Result<(), DispatchError> {
    validate_final_hyper(dims, weights, streams, scratch, rows)?;
    let wide = checked_mul(rows, dims.wide(), "final hyper input")?;
    let low = checked_mul(rows, dims.hc_lowrank, "final hyper low")?;
    let hidden = checked_mul(rows, dims.hidden, "final hyper output")?;
    execute_shared_final_hyper(
        gpu,
        &HyperReadOp {
            state_bf16,
            rotation: scratch.rotation,
            input: &view(streams, 0, wide),
            norm_weight: weights.norm,
            input_mix_down: weights.input_mix_down,
            input_mix_up: weights.input_mix_up,
            normalized: &view(scratch.hc_normalized, 0, wide),
            low: &view(scratch.hc_low, 0, low),
            up: &view(scratch.hc_up, 0, wide),
            mixed: &view(scratch.hc_mixed, 0, hidden),
            bf16_scratch: scratch.gdn_bf16,
            rows,
            branches: dims.hc_count,
            hidden: dims.hidden,
            low_rank: dims.hc_lowrank,
        },
    )
}

/// Validate all final HC operands without launching a kernel.
///
/// Forward callers run this before constructing or executing mutable layer
/// steps so a missing final resource cannot surface after state mutation.
pub fn validate_final_hyper(
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperReadWeights<'_>,
    streams: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    validate_dims(dims)?;
    if rows == 0 {
        return Err(DispatchError::Hip("Qwen4 final hyper rows are zero".into()));
    }
    let wide = dims.wide();
    let low = dims.hc_lowrank;
    let wide_rows = checked_mul(rows, wide, "final hyper input")?;
    let hidden = checked_mul(rows, dims.hidden, "final hyper output")?;
    require_tensor(weights.norm, wide, DType::BF16, "final HC norm")?;
    require_projection_weight(
        &weights.input_mix_down,
        low,
        wide,
        "final HC input mix down",
    )?;
    require_dense_weight(
        &weights.input_mix_up,
        wide,
        low,
        DType::BF16,
        "final HC input mix up",
    )?;
    require_tensor(streams, wide_rows, DType::F32, "final hyper streams")?;
    require_tensor(scratch.hc_mixed, hidden, DType::F32, "final hyper mixed")
}

/// Validate the LM-head weight, hidden stream, and requested output before
/// any stateful layer program is launched.
pub fn validate_lm_head(
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
) -> Result<(), DispatchError> {
    if rows == 0 || requested_rows == 0 || requested_rows > rows {
        return Err(DispatchError::Hip(
            "Qwen4 LM-head row request is invalid".into(),
        ));
    }
    if weight.m == 0 || weight.k == 0 {
        return Err(DispatchError::Hip("Qwen4 LM-head geometry is empty".into()));
    }
    // The head follows the trunk: a packed payload consumes the FWHT basis and
    // is validated by its own group geometry at the artifact boundary, while
    // F32 keeps the widened legacy layout.
    match weight.dtype {
        DType::BF16
        | DType::Q8_0
        | DType::MQ4G256V2
        | DType::MQ4G128V2
        | DType::MQ6G256V2
        | DType::MQ5G256V2
        | DType::MQ3G256V2
        | DType::MQ2G256V2
        | DType::MFP4G32E8SOA => require_weight(weight, weight.m, weight.k, "LM-head weight")?,
        DType::F32 => {
            require_dense_weight(weight, weight.m, weight.k, DType::F32, "LM-head weight")?
        }
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "qwen4-program",
                variant: "lm-head",
                arch: "",
                quant: "unsupported",
            })
        }
    }
    let hidden_elements = checked_mul(rows, weight.k, "LM-head hidden")?;
    require_tensor(hidden_batch, hidden_elements, DType::F32, "LM-head hidden")?;
    let output_elements = checked_mul(requested_rows, weight.m, "LM-head output")?;
    if requested_rows == rows && rows > 1 && weight.dtype != DType::F32 {
        // Refuse a vocabulary no single logits row of which fits the
        // projection output descriptor before any stateful program runs.
        lm_head_tile_rows(weight.m)?;
    }
    require_tensor(logits, output_elements, DType::F32, "LM-head output")?;
    Ok(())
}

/// Largest F32 `Y` extent, in bytes, one packed projection may write.  The
/// projection kernels address `Y` through a raw-buffer descriptor whose
/// num-records field is `i32::MAX` (the PM raw-buffer descriptor limit): a
/// store past it is silently dropped and a 32-bit byte offset past 4 GiB
/// wraps.  A tile's logical output stays within this bound (below 2^31 bytes).
const LM_HEAD_OUTPUT_LIMIT_BYTES: usize = i32::MAX as usize;

/// Row-tile granularity kept once the bound leaves room for it, so every
/// full tile is a whole number of the projection kernels' 256-row tiles.
const LM_HEAD_TILE_ALIGN: usize = 256;

/// Rows one all-row LM-head projection may cover for a vocabulary of `m`
/// logits: `i32::MAX / (m * 4)` rounded down to whole 256-row tiles when that
/// is at least 256, the exact bound otherwise.  A vocabulary whose single row
/// already reaches the bound has no admissible tile and is refused.
fn lm_head_tile_rows(m: usize) -> Result<usize, DispatchError> {
    let row_bytes = m
        .checked_mul(DType::F32.size())
        .ok_or_else(|| DispatchError::Hip("Qwen4 LM-head logits row width overflows".into()))?;
    let safe = LM_HEAD_OUTPUT_LIMIT_BYTES.checked_div(row_bytes).unwrap_or(0);
    if safe == 0 {
        return Err(DispatchError::Hip(format!(
            "Qwen4 LM-head logits row of {m} F32 values exceeds the {LM_HEAD_OUTPUT_LIMIT_BYTES}-byte projection output limit"
        )));
    }
    Ok(if safe >= LM_HEAD_TILE_ALIGN {
        safe / LM_HEAD_TILE_ALIGN * LM_HEAD_TILE_ALIGN
    } else {
        safe
    })
}

/// Contiguous row tiles `(first_row, rows)` covering `0..rows` of an all-row
/// LM head, each below the projection output limit.  No allocation: the
/// plan is arithmetic over the tile size.
#[derive(Clone, Copy, Debug)]
struct LmHeadTiles {
    tile_rows: usize,
    rows: usize,
    next: usize,
}

impl LmHeadTiles {
    fn new(m: usize, rows: usize) -> Result<Self, DispatchError> {
        Ok(Self {
            tile_rows: lm_head_tile_rows(m)?,
            rows,
            next: 0,
        })
    }
}

impl Iterator for LmHeadTiles {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.rows {
            return None;
        }
        let row = self.next;
        let len = self.tile_rows.min(self.rows - row);
        self.next = row + len;
        Some((row, len))
    }
}

/// Element offsets of tile row `row` in the hidden batch (`row * k`) and the
/// logits (`row * m`), checked: the logits offset passes 2^32 elements at
/// realistic long contexts.
fn lm_head_tile_offsets(row: usize, k: usize, m: usize) -> Result<(usize, usize), DispatchError> {
    Ok((
        checked_mul(row, k, "LM-head tile hidden offset")?,
        checked_mul(row, m, "LM-head tile output offset")?,
    ))
}

/// Compute language logits for either every requested row or only the final
/// row of an already-batched hidden stream.  The ordinary public chunk path
/// passes `requested_rows == rows`, preserving its all-row capture contract;
/// a consumer that only needs the final AR row can pass `1` without a second
/// forward implementation.
pub fn execute_lm_head(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
    rotation: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    validate_lm_head(weight, hidden_batch, logits, rows, requested_rows)?;
    // Under the batch-invariant prefill profile a one-row head is the final-row
    // head (the GEMV, never the routed GEMM), so a one-row forward and a longer
    // forward's final row produce the same bytes.
    if requested_rows == rows && (rows > 1 || !gpu.prefill_invariant_scope) {
        if weight.dtype == DType::F32 {
            return hip(gpu.gemm_f32_batched(
                weight.buf,
                hidden_batch,
                logits,
                weight.m,
                weight.k,
                rows,
            ));
        }
        // One projection contract for the head and the trunk: packed payloads
        // rotate `rows * k` into the caller's basis scratch and decode through
        // the GEMV or batched GEMM launcher that matches the row count.
        //
        // The projection kernels address their F32 output through a raw-buffer
        // descriptor bounded by `i32::MAX` bytes (stores past it are dropped,
        // and byte offsets past 4 GiB wrap), so an all-row head whose logical
        // output reaches that extent projects whole row tiles below it.  Every
        // tile is the same projection over its own rows: same hidden rows,
        // same logits rows, same rotation scratch rows as the one-call form.
        let tiles = LmHeadTiles::new(weight.m, rows)?;
        if tiles.tile_rows >= rows {
            return project_weight(gpu, weight, hidden_batch, logits, rows, rotation);
        }
        if let Some(rotation) = rotation {
            let needed = checked_mul(rows, weight.k, "LM-head rotation")?;
            if rotation.numel() < needed {
                return Err(DispatchError::Hip(format!(
                    "Qwen4 LM-head rotation capacity too small: need {needed} elements, have {}",
                    rotation.numel()
                )));
            }
        }
        for (row, len) in tiles {
            let (input_offset, output_offset) = lm_head_tile_offsets(row, weight.k, weight.m)?;
            let input = view(
                hidden_batch,
                input_offset,
                checked_mul(len, weight.k, "LM-head tile hidden")?,
            );
            let output = view(
                logits,
                output_offset,
                checked_mul(len, weight.m, "LM-head tile output")?,
            );
            // The rotation scratch is read at the tile's own `row * k`, the
            // rows the one-call form would have rotated; bounded by
            // `rows * k`, validated above.
            let tile_rotation = rotation.map(|scratch| view(scratch, input_offset, len * weight.k));
            project_weight(gpu, weight, &input, &output, len, tile_rotation.as_ref())?;
        }
        return Ok(());
    } else if requested_rows == 1 {
        let input = view(
            hidden_batch,
            checked_mul(rows - 1, weight.k, "final LM-head row")?,
            weight.k,
        );
        if weight.dtype == DType::F32 {
            return hip(gpu.gemm_f32_batched(weight.buf, &input, logits, weight.m, weight.k, 1));
        }
        return project_weight(gpu, weight, &input, logits, 1, rotation);
    } else {
        Err(DispatchError::Hip(
            "Qwen4 LM-head supports all rows or final row only".into(),
        ))
    }
}

fn validate_dims(dims: Qwen4ProgramDims) -> Result<(), DispatchError> {
    let products = [
        (dims.hidden, dims.hc_count, "wide"),
        (dims.indexer_n_heads, dims.indexer_head_dim, "indexer Q"),
        (dims.indexer_kv_heads, dims.indexer_head_dim, "indexer K"),
        (dims.num_attention_heads, dims.head_dim, "Q width"),
        (dims.num_key_value_heads, dims.head_dim, "KV width"),
        (
            dims.linear_num_key_heads,
            dims.linear_key_head_dim,
            "GDN QK",
        ),
        (
            dims.linear_num_value_heads,
            dims.linear_value_head_dim,
            "GDN V",
        ),
    ];
    for (a, b, label) in products {
        checked_mul(a, b, label)?;
    }
    checked_mul(2, dims.gdn_qk(), "GDN QKV")?
        .checked_add(dims.gdn_value())
        .ok_or_else(|| DispatchError::Hip("Qwen4 GDN QKV overflows".into()))?;
    if dims.hc_count == 0
        || dims.hc_lowrank == 0
        || dims.indexer_compress_ratio == 0
        || dims.indexer_budget == 0
        || dims.norm_eps <= 0.0
    {
        return Err(DispatchError::Hip(
            "Qwen4 layer descriptor has invalid geometry".into(),
        ));
    }
    match dims.ffn {
        Qwen4FfnDims::Moe {
            num_experts,
            experts_per_token,
            intermediate,
            shared_intermediate,
        } => {
            if intermediate == 0
                || shared_intermediate == 0
                || num_experts == 0
                || experts_per_token == 0
            {
                return Err(DispatchError::Hip(
                    "Qwen4 layer descriptor has invalid geometry".into(),
                ));
            }
            if experts_per_token > num_experts {
                return Err(DispatchError::Hip(
                    "Qwen4 top-k exceeds expert capacity".into(),
                ));
            }
        }
        Qwen4FfnDims::Dense { intermediate } => {
            if intermediate == 0 {
                return Err(DispatchError::Hip(
                    "Qwen4 layer descriptor has invalid geometry".into(),
                ));
            }
        }
    }
    Ok(())
}

fn require_tensor(
    tensor: &GpuTensor,
    elements: usize,
    dtype: DType,
    name: &'static str,
) -> Result<(), DispatchError> {
    if tensor.dtype != dtype {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected {dtype:?}, got {:?}",
            tensor.dtype
        )));
    }
    let bytes = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {name} size overflows")))?;
    if tensor.numel() < elements || tensor.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} capacity too small: need {elements} elements, have {}",
            tensor.numel()
        )));
    }
    Ok(())
}

/// Projection payloads the shared lowering can consume: source BF16, or either
/// Qwen4 matrix quantization whose FWHT basis the caller rotates in.  Packed
/// payloads are sized by their own group geometry, which the artifact boundary
/// validates at admission, so only the native layout carries an extent check.
fn require_projection_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    name: &'static str,
) -> Result<(), DispatchError> {
    require_weight(weight, m, k, name)?;
    match weight.dtype {
        DType::BF16 => {
            let bytes = m
                .checked_mul(k)
                .and_then(|elements| elements.checked_mul(DType::BF16.size()))
                .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {name} size overflows")))?;
            if weight.buf.buf.size() < bytes {
                return Err(DispatchError::Hip(format!(
                    "Qwen4 {name} capacity too small: need {bytes} bytes, have {}",
                    weight.buf.buf.size()
                )));
            }
            Ok(())
        }
        DType::MQ4G256V2
        | DType::MQ4G128V2
        | DType::MQ6G256V2
        | DType::MQ5G256V2
        | DType::MQ3G256V2
        | DType::MQ2G256V2
        | DType::MFP4G32E8SOA
        | DType::Q8_0 => Ok(()),
        dtype => Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected BF16 or a Qwen4 matrix quantization, got {dtype:?}"
        ))),
    }
}

fn require_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    name: &'static str,
) -> Result<(), DispatchError> {
    if weight.m != m || weight.k != k {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} geometry mismatch: expected {m}x{k}, got {}x{}",
            weight.m, weight.k
        )));
    }
    Ok(())
}

fn require_dense_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    dtype: DType,
    name: &'static str,
) -> Result<(), DispatchError> {
    require_weight(weight, m, k, name)?;
    if weight.dtype != dtype {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected {dtype:?}, got {:?}",
            weight.dtype
        )));
    }
    let bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(dtype.size()))
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {name} size overflows")))?;
    if weight.buf.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} capacity too small: need {bytes} bytes, have {}",
            weight.buf.buf.size()
        )));
    }
    Ok(())
}

/// Seal the architecture-bound MoE call before any operation in the layer
/// step list can launch.  The returned call retains the shared sealed-MoE
/// ownership/lifecycle contract; the architecture owns the surrounding
/// operation order.  The decode-vs-prefill route is decided on `route_rows`
/// (`Gpu::route_rows(rows)`: the canonical chunk under the invariant prefill
/// profile, so a one-row prefill still seals the grouped prefill call);
/// `rows` sizes the call.
pub fn seal_moe_for_layer<'a>(
    ctx: &'a hipfire_dispatch::context::DispatchCtx,
    dims: Qwen4ProgramDims,
    moe: Qwen4MoeBinding<'a>,
    scratch: &'a Qwen4LayerScratch<'a>,
    rows: usize,
    route_rows: usize,
) -> Result<hipfire_dispatch::pipeline::sealed_moe::SealedMoeCall<'a>, DispatchError> {
    let bound = BoundMoeExperts::from_cache(moe.table, moe.cache)
        .map_err(|error| DispatchError::Hip(format!("bound Qwen4 experts: {error:?}")))?;
    if route_rows == 1 {
        let params = build_moe_decode(dims, moe, scratch.hc_mixed, scratch)?;
        seal_decode(bound, ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 decode MoE: {error:?}")))
    } else {
        let params = build_moe_prefill(dims, moe, scratch, scratch.moe_router_logits, rows)?;
        seal_prefill(bound, ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 prefill MoE: {error:?}")))
    }
}

/// MoE extents of a program whose FFN is MoE; a dense program has no MoE
/// route to seal or lower.
struct MoeDims {
    num_experts: usize,
    experts_per_token: usize,
    intermediate: usize,
    shared_intermediate: usize,
}

fn moe_dims(dims: Qwen4ProgramDims) -> Result<MoeDims, DispatchError> {
    match dims.ffn {
        Qwen4FfnDims::Moe {
            num_experts,
            experts_per_token,
            intermediate,
            shared_intermediate,
        } => Ok(MoeDims {
            num_experts,
            experts_per_token,
            intermediate,
            shared_intermediate,
        }),
        Qwen4FfnDims::Dense { .. } => Err(DispatchError::Hip(
            "Qwen4 dense FFN program has no MoE route to seal".into(),
        )),
    }
}

/// Build the indexed decode descriptor for a single-token layer invocation.
///
/// Rows-one execution deliberately keeps the old decode contract: one
/// normalized activation, GEMV-sized scratch, and `seal_decode`'s exact
/// indexed route.  Grouped prefill descriptors are built only for a real
/// multi-row tile.
fn build_moe_decode<'a>(
    dims: Qwen4ProgramDims,
    moe: Qwen4MoeBinding<'a>,
    input: &'a GpuTensor,
    scratch: &'a Qwen4LayerScratch<'a>,
) -> Result<MoeParams<'a>, DispatchError> {
    let moe_dims = moe_dims(dims)?;
    let (first_gate_up, first_down) = moe
        .routed_experts
        .get(0)
        .ok_or_else(|| DispatchError::Hip("Qwen4 routed expert table is empty".into()))?;
    let experts_all_gate_up_mq4 = moe.experts_all_gate_up_mq4;
    let dtypes = MoeDtypes {
        router: moe.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: moe.shared.selector.dtype,
            gate: moe.shared.gate.dtype,
            up: moe.shared.up.dtype,
            down: moe.shared.down.dtype,
        }),
        experts_all_gate_up_mq4,
        routed_gate_up: first_gate_up.dtype,
        routed_down: first_down.dtype,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    Ok(MoeParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
        route_policy: Some(moe.route_policy),
        normalization: MoeNormalization::Provided,
        batch_size: 1,
        hidden: dims.hidden,
        mi: moe_dims.intermediate,
        k: moe_dims.experts_per_token,
        n_exp: moe_dims.num_experts,
        norm_topk_prob: moe.norm_topk_prob,
        selection_bias: None,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        cpu_deferred_experts: false,
        ep_mode: MoeEpMode::None,
        layer_idx: moe.layer_idx,
        x_norm: input,
        x_residual: scratch.ffn_output,
        routed_out: None,
        skip_shared: false,
        router: moe.router,
        shared: Some(MoeSharedDecode {
            weights: MoeSharedWeights {
                selector: moe.shared.selector,
                gate: moe.shared.gate,
                up: moe.shared.up,
                down: moe.shared.down,
            },
            intermediate: moe_dims.shared_intermediate,
            scalar: scratch.moe_scalar,
            gate_out: scratch.moe_gate,
            up_out: scratch.moe_up,
        }),
        expert_gate_up_ptrs: moe.expert_gate_up_ptrs,
        expert_down_ptrs: moe.expert_down_ptrs,
        expert_ptrs_host: Some(hipfire_dispatch::families::moe::MoePointerEntries {
            gate_up: moe.expert_gate_up_entries,
            down: moe.expert_down_entries,
        }),
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: dims.hidden,
        routed_down_m: dims.hidden,
        routed_down_k: moe_dims.intermediate,
        routed_experts: moe.routed_experts,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.moe_router_logits,
        x_rot_local: scratch.moe_x_rot,
        gate_up_buf: scratch.moe_gate_up,
        ffn_hidden: scratch.moe_hidden,
        ffn_out: scratch.moe_shared_output,
        gate_batch: scratch.moe_gate_batch,
        up_batch: scratch.moe_up_batch,
        rot_batch: scratch.moe_rot_batch,
        topk_indices: scratch.moe_topk_indices,
        topk_weights: scratch.moe_topk_weights,
        down_expanded: scratch.moe_down_expanded,
        ep_row_gather: moe.row_gather,
    })
}

fn build_moe_prefill<'a>(
    dims: Qwen4ProgramDims,
    moe: Qwen4MoeBinding<'a>,
    scratch: &'a Qwen4LayerScratch<'a>,
    router_matrix: &'a GpuTensor,
    rows: usize,
) -> Result<MoePrefillParams<'a>, DispatchError> {
    let (first_gate_up, first_down) = moe
        .routed_experts
        .get(0)
        .ok_or_else(|| DispatchError::Hip("Qwen4 routed expert table is empty".into()))?;
    let moe_dims = moe_dims(dims)?;
    let slots = checked_mul(rows, moe_dims.experts_per_token, "MoE slots")?;
    let m_total_max = grouped_m_total_bound(slots, moe_dims.num_experts)?;
    let shared = MoeSharedPrefill {
        weights: MoeSharedWeights {
            selector: moe.shared.selector,
            gate: moe.shared.gate,
            up: moe.shared.up,
            down: moe.shared.down,
        },
        intermediate: moe.intermediate,
        scalar: scratch.moe_scalar,
        gate_out: scratch.moe_gate,
        up_out: scratch.moe_up,
        rotated: scratch.moe_hidden,
    };
    let dtypes = MoeDtypes {
        router: moe.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: moe.shared.selector.dtype,
            gate: moe.shared.gate.dtype,
            up: moe.shared.up.dtype,
            down: moe.shared.down.dtype,
        }),
        experts_all_gate_up_mq4: moe.experts_all_gate_up_mq4,
        routed_gate_up: first_gate_up.dtype,
        routed_down: first_down.dtype,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let prelude = MoePrefillPrelude {
        normalization: MoeNormalization::Provided,
        router: moe.router,
        router_logits: router_matrix,
        router_scores: router_matrix,
        norm_topk_prob: moe.norm_topk_prob,
        route: PrefillRouteMode::Replicated,
        shared: Some(shared),
        q8_router_policy: MoeQ8RouterPolicy::DispatcherEntry,
    };
    Ok(MoePrefillParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
        route_policy: Some(moe.route_policy),
        prelude,
        batch_size: rows,
        mi: moe_dims.intermediate,
        down_m: dims.hidden,
        down_k: moe_dims.intermediate,
        gate_up_k: dims.hidden,
        k_top: moe_dims.experts_per_token,
        n_exp: moe_dims.num_experts,
        m_total_max,
        force_mq4_grouped_fp16: false,
        cpu_deferred_experts: false,
        topk_indices: scratch.moe_topk_indices,
        topk_weights: scratch.moe_topk_weights,
        x_batch: scratch.ffn_output,
        x_norm_batch: scratch.hc_mixed,
        x_rot_batch: scratch.moe_x_rot,
        expert_gate_up_ptrs: moe.expert_gate_up_ptrs,
        expert_down_ptrs: moe.expert_down_ptrs,
        expert_stage_ptrs: moe.expert_stage_ptrs,
        routed_experts: moe.routed_experts,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        gate_batch: scratch.moe_gate_batch,
        up_batch: scratch.moe_up_batch,
        rot_batch: scratch.moe_rot_batch,
        down_expanded: scratch.moe_down_expanded,
        expert_token_counts: scratch.moe_expert_token_counts,
        expert_offsets: scratch.moe_expert_offsets,
        sorted_slot_index: scratch.moe_sorted_slot_index,
        expert_tile_ids: scratch.moe_expert_tile_ids,
        inverse_perm: scratch.moe_inverse_perm,
        y_gate_up_grouped: scratch.moe_y_gate_up_grouped,
        y_down_grouped: scratch.moe_y_down_grouped,
        paro_gate_up: None,
        paro_down: None,
        down_awq_scale: None,
        routed_out: None,
        ep_row_gather: moe.row_gather,
        shared_down_out: moe.shared_down_out,
        ep_layer_idx: moe.layer_idx,
        ep_tile_exchange: moe.tile_exchange,
    })
}

#[cfg(test)]
mod lm_head_tile_tests {
    use super::*;

    /// Qwen4 vocabulary the long-context KLD product head projects.
    const VOCAB: usize = 248_320;
    const LIMIT: u128 = i32::MAX as u128;

    fn plan(m: usize, rows: usize) -> Vec<(usize, usize)> {
        LmHeadTiles::new(m, rows).expect("plan").collect()
    }

    /// Contiguous, exact, in-order coverage of `0..rows` whose every tile's
    /// logical F32 output stays below the descriptor limit.
    fn assert_exact_safe(m: usize, rows: usize) {
        let tiles = plan(m, rows);
        let mut next = 0usize;
        for &(row, len) in &tiles {
            assert_eq!(row, next, "m={m} rows={rows}: tiles must be contiguous");
            assert!(len > 0, "m={m} rows={rows}: empty tile");
            assert!(
                len as u128 * m as u128 * 4 <= LIMIT,
                "m={m} rows={rows}: tile at row {row} of {len} rows writes {} bytes",
                len as u128 * m as u128 * 4
            );
            next = row + len;
        }
        assert_eq!(next, rows, "m={m} rows={rows}: tiles must cover every row");
    }

    #[test]
    fn long_context_chunks_are_covered_below_the_output_limit() {
        for rows in [2048usize, 4096, 4608, 6144, 8192] {
            assert_exact_safe(VOCAB, rows);
        }
    }

    #[test]
    fn ragged_row_counts_are_covered_exactly() {
        for rows in [1usize, 2, 255, 257, 2047, 2049, 4097, 4607, 6145, 8191, 8193, 10_000] {
            assert_exact_safe(VOCAB, rows);
        }
    }

    #[test]
    fn vocab_248320_projects_four_2048_row_tiles_at_8192() {
        assert_eq!(lm_head_tile_rows(VOCAB).unwrap(), 2048);
        assert_eq!(
            plan(VOCAB, 8192),
            vec![(0, 2048), (2048, 2048), (4096, 2048), (6144, 2048)]
        );
        // 2048 rows is exactly one tile: the small path keeps its one call.
        assert_eq!(plan(VOCAB, 2048), vec![(0, 2048)]);
        assert_eq!(
            plan(VOCAB, 4608),
            vec![(0, 2048), (2048, 2048), (4096, 512)]
        );
        assert_eq!(
            plan(VOCAB, 6144),
            vec![(0, 2048), (2048, 2048), (4096, 2048)]
        );
    }

    #[test]
    fn short_batches_keep_the_single_call() {
        for rows in [1usize, 8, 9, 96, 512, 2047, 2048] {
            let tiles = LmHeadTiles::new(VOCAB, rows).unwrap();
            assert!(tiles.tile_rows >= rows, "rows={rows} must not be sliced");
        }
    }

    #[test]
    fn tile_size_rounds_to_whole_256_row_tiles_when_it_can() {
        // 1_000_000 columns: 4 MB per row, 536 safe rows -> 512.
        assert_eq!(lm_head_tile_rows(1_000_000).unwrap(), 512);
        // Tiny vocabulary: far above one tile, still a multiple of 256 and safe.
        let tile = lm_head_tile_rows(1024).unwrap();
        assert_eq!(tile % LM_HEAD_TILE_ALIGN, 0);
        assert!(tile as u128 * 1024 * 4 <= LIMIT);
        assert!((tile + LM_HEAD_TILE_ALIGN) as u128 * 1024 * 4 > LIMIT);
        // Wide enough that fewer than 256 rows fit: the exact bound, positive.
        assert_eq!(lm_head_tile_rows(3_000_000).unwrap(), 178);
        assert_exact_safe(3_000_000, 1000);
        // The widest admissible row is a one-row tile.
        let widest = i32::MAX as usize / 4;
        assert_eq!(lm_head_tile_rows(widest).unwrap(), 1);
        assert_exact_safe(widest, 5);
    }

    #[test]
    fn impossibly_wide_rows_are_refused() {
        // One row alone is past the descriptor bound.
        assert!(lm_head_tile_rows(i32::MAX as usize / 4 + 1).is_err());
        assert!(LmHeadTiles::new(i32::MAX as usize, 8).is_err());
        // Width in bytes overflows usize.
        let error = lm_head_tile_rows(usize::MAX / 2).unwrap_err();
        assert!(format!("{error:?}").contains("overflows"), "{error:?}");
        assert!(lm_head_tile_rows(usize::MAX).is_err());
        assert!(lm_head_tile_rows(0).is_err());
    }

    #[test]
    fn offsets_past_four_gib_do_not_overflow() {
        let k = 2048usize;
        for rows in [4608usize, 8192, 16_384] {
            for (row, len) in plan(VOCAB, rows) {
                let (input, output) = lm_head_tile_offsets(row, k, VOCAB).unwrap();
                assert_eq!(input as u128, row as u128 * k as u128);
                assert_eq!(output as u128, row as u128 * VOCAB as u128);
                // The tile's end byte is exact in 64 bits too.
                let end = (output as u128 + len as u128 * VOCAB as u128) * 4;
                assert_eq!(end, (row + len) as u128 * VOCAB as u128 * 4);
            }
        }
        // Row 8192 of a 248320-wide head starts at 8.14 GB: past 4 GiB.
        let (_, output) = lm_head_tile_offsets(8192, k, VOCAB).unwrap();
        assert!(output as u128 * 4 > 1u128 << 32);
    }

    #[test]
    fn offset_arithmetic_overflow_is_an_error() {
        assert!(lm_head_tile_offsets(usize::MAX / 2, 4, 1).is_err());
        assert!(lm_head_tile_offsets(usize::MAX / 2, 1, 4).is_err());
        assert!(lm_head_tile_offsets(usize::MAX, 2, 1).is_err());
    }
}
