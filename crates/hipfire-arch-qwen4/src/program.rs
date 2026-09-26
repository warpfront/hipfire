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
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
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
    pub expanded_rows: usize,
}

impl Qwen4ScratchLayout {
    pub fn for_rows(dims: Qwen4ProgramDims, rows: usize) -> Result<Self, DispatchError> {
        validate_dims(dims)?;
        if rows == 0 {
            return Err(DispatchError::Hip("Qwen4 scratch rows are zero".into()));
        }
        let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
        let slots = checked_mul(rows, dims.experts_per_token, "MoE slots")?;
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
            grouped_rows: grouped_m_total_bound(slots, dims.num_experts)?,
            shared_rows: checked_mul(rows, dims.shared_intermediate, "shared rows")?,
            routed_rows: checked_mul(slots, dims.moe_intermediate, "routed rows")?,
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
    pub moe: Qwen4MoeBinding<'a>,
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
    /// Host entries the pointer tables above were uploaded from, so dispatch can
    /// prove the table contents name the live experts (a retained body requires it).
    pub expert_gate_up_entries: &'a [usize],
    pub expert_down_entries: &'a [usize],
    pub layer_idx: u16,
    pub norm_topk_prob: bool,
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
    pub moe_output: &'a GpuTensor,
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
    require_tensor(logits, output_elements, DType::F32, "LM-head output")?;
    Ok(())
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
    if requested_rows == rows {
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
        return project_weight(gpu, weight, hidden_batch, logits, rows, rotation);
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
        || dims.moe_intermediate == 0
        || dims.shared_intermediate == 0
        || dims.ple_conv_kernel_dim == 0
        || dims.num_experts == 0
        || dims.experts_per_token == 0
        || dims.norm_eps <= 0.0
    {
        return Err(DispatchError::Hip(
            "Qwen4 layer descriptor has invalid geometry".into(),
        ));
    }
    if dims.experts_per_token > dims.num_experts {
        return Err(DispatchError::Hip(
            "Qwen4 top-k exceeds expert capacity".into(),
        ));
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
/// operation order.
pub fn seal_moe_for_layer<'a>(
    ctx: &'a hipfire_dispatch::context::DispatchCtx,
    dims: Qwen4ProgramDims,
    layer: &Qwen4LayerDescription<'a>,
    scratch: &'a Qwen4LayerScratch<'a>,
    rows: usize,
) -> Result<hipfire_dispatch::pipeline::sealed_moe::SealedMoeCall<'a>, DispatchError> {
    let bound = BoundMoeExperts::from_cache(layer.moe.table, layer.moe.cache)
        .map_err(|error| DispatchError::Hip(format!("bound Qwen4 experts: {error:?}")))?;
    if rows == 1 {
        let params = build_moe_decode(dims, layer.moe, scratch.hc_mixed, scratch)?;
        seal_decode(bound, ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 decode MoE: {error:?}")))
    } else {
        let params = build_moe_prefill(dims, layer.moe, scratch, scratch.moe_router_logits, rows)?;
        seal_prefill(bound, ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 prefill MoE: {error:?}")))
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
        mi: dims.moe_intermediate,
        k: dims.experts_per_token,
        n_exp: dims.num_experts,
        norm_topk_prob: moe.norm_topk_prob,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        ep_mode: MoeEpMode::None,
        layer_idx: moe.layer_idx,
        x_norm: input,
        x_residual: scratch.moe_output,
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
            intermediate: dims.shared_intermediate,
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
        routed_down_k: dims.moe_intermediate,
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
    let slots = checked_mul(rows, dims.experts_per_token, "MoE slots")?;
    let m_total_max = grouped_m_total_bound(slots, dims.num_experts)?;
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
        mi: dims.moe_intermediate,
        down_m: dims.hidden,
        down_k: dims.moe_intermediate,
        gate_up_k: dims.hidden,
        k_top: dims.experts_per_token,
        n_exp: dims.num_experts,
        m_total_max,
        force_mq4_grouped_fp16: false,
        topk_indices: scratch.moe_topk_indices,
        topk_weights: scratch.moe_topk_weights,
        x_batch: scratch.moe_output,
        x_norm_batch: scratch.hc_mixed,
        x_rot_batch: scratch.moe_x_rot,
        expert_gate_up_ptrs: moe.expert_gate_up_ptrs,
        expert_down_ptrs: moe.expert_down_ptrs,
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
    })
}
