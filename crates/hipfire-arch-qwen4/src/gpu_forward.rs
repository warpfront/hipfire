// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP Qwen4 forward execution.
//!
//! This module is the production device path.  The CPU/reference equations in
//! reference_forward are intentionally separate and are never called
//! here.  All
//! learned projections consume the resident tensor supplied by the assembled
//! bundle; MQv2 projections are rotated and dispatched through their native
//! qt44/qt53 kernels, while recurrent/cache state remains resident on the GPU.

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config};
use crate::ple::{PLE_HEAD_COUNT, PLE_ROW_WIDTH};
use crate::program::{
    execute_final_hyper, execute_lm_head, validate_final_hyper, validate_lm_head,
    Qwen4AttentionWeights, Qwen4GdnWeights, Qwen4HyperReadWeights, Qwen4HyperWeights,
    Qwen4LayerDescription, Qwen4LayerScratch, Qwen4MoeBinding, Qwen4PleWeights, Qwen4ProgramDims,
    Qwen4QsaWeights, Qwen4ScratchLayout,
};
use crate::projection::{dispatch_embedding, row_stride, ProjectionView};
use crate::weights::{
    HyperConnectionReadWeights, HyperConnectionWeights, MoeWeights, Qwen4LayerWeights,
    Qwen4Weights, TensorRef, WeightError,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoeRecipe, MoeRouteCapability,
    MoeRouteFormats, MoeRouteGeometry, MoeRoutePolicy, MoeSharedDecode, MoeSharedDtypes,
    MoeSharedWeights, RoutedExpertWeights,
};
use hipfire_dispatch::pipeline::sealed_moe::{
    retained_body_action, specialized_sealed_moe_retained_admission, RetainedBodyAction,
};
use hipfire_dispatch::pipeline::{
    execute_steps, execute_validated_steps, seal_decode, validate_steps, BoundMoeExperts, ClearOp,
    ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources, ExpertTable,
    GatedDeltaNetOp, GroupedDepthwiseOp, HyperReadOp, HyperWriteOp, IndexedAttentionOp,
    IndexedAttentionState, Step,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::external_rows::RowFetch;
use hipfire_runtime::weight_manifest::ExpertSourceLayout;
use rdna_compute::replay::ShadowBodyRoute;
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::time::Instant;

const EPSILON: f32 = 1.0e-6;
/// Maximum number of rows resident in the reusable Qwen4 forward scratch.
///
/// Public serving calls may receive longer prompts; the forward owner tiles
/// those requests over this bounded capacity instead of allocating
/// prompt-sized grouped MoE buffers.  Every chunk re-streams all expert
/// weights, so the cap equals the Qwen4 contract's 2048-token `max_seq`:
/// a prompt is one chunk (1131 tokens: 3 chunks at 512 were 8% slower).
pub(crate) const QWEN4_PREFILL_CHUNK_CAP: usize = 2048;
const QWEN4_STEP_INLINE_CAPACITY: usize = 384;
const QWEN4_QSA_INLINE_CAPACITY: usize = 12;

/// Which rows of a batched Qwen4 prefill write language-model logits.
///
/// `All` is the public chunk/capture contract.  `Final` keeps the complete
/// trunk batch but emits only its final row for ordinary autoregressive AR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4OutputRows {
    All,
    Final,
}

impl Qwen4OutputRows {
    pub(crate) fn count(self, n: usize) -> usize {
        match self {
            Self::All => n,
            Self::Final => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4OutputPolicy {
    Rows(Qwen4OutputRows),
    None,
}

impl Qwen4OutputPolicy {
    fn requested_rows(self, n: usize) -> Option<usize> {
        match self {
            Self::Rows(rows) => Some(rows.count(n)),
            Self::None => None,
        }
    }
}

/// Host-side forward phases timed when profiling is enabled.
#[derive(Clone, Copy)]
pub(crate) enum Qwen4ProfilePhase {
    MoeSeal,
    PleWait,
    PleStage,
    PleUpload,
    PleApply,
}

impl Qwen4ProfilePhase {
    pub(crate) const ALL: [Self; 5] = [
        Self::MoeSeal,
        Self::PleWait,
        Self::PleStage,
        Self::PleUpload,
        Self::PleApply,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::MoeSeal => "moe_seal",
            Self::PleWait => "ple_wait",
            Self::PleStage => "ple_stage",
            Self::PleUpload => "ple_upload",
            Self::PleApply => "ple_apply",
        }
    }
}

/// Calls and host nanoseconds per [`Qwen4ProfilePhase`], indexed by phase.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Qwen4ProfileStats {
    pub(crate) calls: [u64; Qwen4ProfilePhase::ALL.len()],
    pub(crate) ns: [u64; Qwen4ProfilePhase::ALL.len()],
}

thread_local! {
    static QWEN4_PROFILE_ENABLED: Cell<bool> = const { Cell::new(false) };
    static QWEN4_PROFILE_STATS: RefCell<Qwen4ProfileStats> =
        RefCell::new(Qwen4ProfileStats::default());
}

#[inline(always)]
fn qwen4_profile_enabled() -> bool {
    QWEN4_PROFILE_ENABLED.with(|enabled| enabled.get())
}

pub(crate) fn qwen4_profile_enable(enabled: bool) {
    QWEN4_PROFILE_ENABLED.with(|current| current.set(enabled));
}

pub(crate) fn qwen4_profile_reset() {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow_mut() = Qwen4ProfileStats::default());
}

pub(crate) fn qwen4_profile_snapshot() -> Qwen4ProfileStats {
    QWEN4_PROFILE_STATS.with(|stats| *stats.borrow())
}

#[inline(always)]
fn qwen4_profile_start() -> Option<Instant> {
    if qwen4_profile_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline(always)]
fn qwen4_profile_record(phase: Qwen4ProfilePhase, started: Option<Instant>) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed().as_nanos() as u64;
    QWEN4_PROFILE_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        let slot = phase as usize;
        stats.calls[slot] = stats.calls[slot].saturating_add(1);
        stats.ns[slot] = stats.ns[slot].saturating_add(elapsed);
    });
}

/// Errors returned by the native ordinary-HIP path.
#[derive(Debug)]
pub enum Qwen4GpuForwardError {
    Hip(hip_bridge::HipError),
    Weights(WeightError),
    Invalid(String),
    Dispatch(String),
    Ple(String),
}

impl fmt::Display for Qwen4GpuForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => write!(f, "Qwen4 GPU forward HIP error: {error}"),
            Self::Weights(error) => write!(f, "Qwen4 GPU forward weights error: {error}"),
            Self::Invalid(error) => write!(f, "Qwen4 GPU forward invalid input: {error}"),
            Self::Dispatch(error) => write!(f, "Qwen4 GPU forward dispatch error: {error}"),
            Self::Ple(error) => write!(f, "Qwen4 GPU forward PLE error: {error}"),
        }
    }
}

impl std::error::Error for Qwen4GpuForwardError {}

impl From<hip_bridge::HipError> for Qwen4GpuForwardError {
    fn from(error: hip_bridge::HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<WeightError> for Qwen4GpuForwardError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

fn invalid(message: impl Into<String>) -> Qwen4GpuForwardError {
    Qwen4GpuForwardError::Invalid(message.into())
}

fn checked_bytes(rows: usize, stride: usize) -> Result<usize, Qwen4GpuForwardError> {
    crate::projection::checked_bytes(rows, stride)
        .ok_or_else(|| invalid("packed row byte size overflow"))
}

fn f32_view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert_eq!(tensor.dtype, DType::F32);
    tensor.sub_offset(offset, len)
}

fn view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    tensor.sub_offset(offset, len)
}
fn matrix_view(
    tensor: &GpuTensor,
    rows: usize,
    columns: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("Qwen4 matrix view shape overflows"))?;
    let mut view = tensor.sub_offset(0, elements);
    view.shape = vec![rows, columns];
    Ok(view)
}

fn weight_dims(reference: &TensorRef) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if reference.shape.len() != 2 {
        return Err(invalid(format!(
            "projection {} has logical shape {:?}, expected rank two",
            reference.name, reference.shape
        )));
    }
    Ok((reference.shape[0], reference.shape[1]))
}
fn dense_ref<'a>(
    weights: &'a Qwen4Weights,
    reference: &TensorRef,
) -> Result<WeightRef<'a>, Qwen4GpuForwardError> {
    let tensor = weights.resident(reference)?;
    let (m, k) = weight_dims(reference)?;
    Ok(WeightRef {
        buf: tensor,
        dtype: tensor.dtype,
        m,
        k,
        row_stride: row_stride(tensor.dtype, k),
        rotation: None,
        awq_scale: None,
    })
}

/// The admitted Qwen4 grouped route is declared by this architecture
/// constructor and consumed as data by shared sealing/lowering.
fn qwen4_route_policy(config: &Qwen4Config, shared_down: DType) -> MoeRoutePolicy {
    MoeRoutePolicy {
        capability: MoeRouteCapability::Qt44Qt53Grouped,
        geometry: MoeRouteGeometry {
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hidden: config.hidden_size,
            routed_intermediate: config.moe_intermediate_size,
            shared_intermediate: config.shared_expert_intermediate_size,
        },
        formats: MoeRouteFormats {
            router: DType::BF16,
            shared_selector: DType::BF16,
            shared_gate: DType::BF16,
            shared_up: DType::BF16,
            shared_down,
            routed_gate_up: DType::MQ4G256V2,
            routed_down: DType::MQ4G128V2,
        },
    }
}
fn program_dims(config: &Qwen4Config) -> Qwen4ProgramDims {
    Qwen4ProgramDims {
        hidden: config.hidden_size,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        indexer_n_heads: config.indexer_n_heads,
        indexer_kv_heads: config.indexer_kv_heads,
        indexer_head_dim: config.indexer_head_dim,
        indexer_budget: config.indexer_budget,
        indexer_compress_ratio: config.indexer_compress_ratio,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        linear_num_key_heads: config.linear_num_key_heads,
        linear_num_value_heads: config.linear_num_value_heads,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        moe_intermediate: config.moe_intermediate_size,
        shared_intermediate: config.shared_expert_intermediate_size,
        num_experts: config.num_experts,
        experts_per_token: config.num_experts_per_tok,
        ple_conv_kernel_dim: config.ple_conv_kernel_size,
        norm_eps: EPSILON,
    }
}

fn hyper_read_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionReadWeights,
) -> Result<Qwen4HyperReadWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4HyperReadWeights {
        norm: weights.resident(&hyper.hc_norm)?,
        input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
        input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
    })
}

fn hyper_desc<'a>(
    weights: &'a Qwen4Weights,
    hyper: &HyperConnectionWeights,
) -> Result<Qwen4HyperWeights<'a>, Qwen4GpuForwardError> {
    let norm = weights.resident(&hyper.hc_norm)?;
    Ok(Qwen4HyperWeights {
        read: Qwen4HyperReadWeights {
            norm,
            input_mix_down: dense_ref(weights, &hyper.input_mix_down)?,
            input_mix_up: dense_ref(weights, &hyper.input_mix_up)?,
        },
        write: crate::program::Qwen4HyperWriteWeights {
            norm,
            block_inject: dense_ref(weights, &hyper.block_inject)?,
        },
    })
}

fn gdn_desc<'a>(
    weights: &'a Qwen4Weights,
    gdn: &crate::weights::GdnWeights,
) -> Result<Qwen4GdnWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4GdnWeights {
        qkv: dense_ref(weights, &gdn.qkv)?,
        conv: weights.resident(&gdn.conv)?,
        in_proj_a: dense_ref(weights, &gdn.in_proj_a)?,
        in_proj_b: dense_ref(weights, &gdn.in_proj_b)?,
        a_log: weights.resident(&gdn.a_log)?,
        dt_bias: weights.resident(&gdn.dt_bias)?,
        z: dense_ref(weights, &gdn.z)?,
        norm: weights.resident(&gdn.norm)?,
        output: dense_ref(weights, &gdn.output)?,
    })
}

fn qsa_desc<'a>(
    weights: &'a Qwen4Weights,
    qsa: &crate::weights::QsaWeights,
) -> Result<Qwen4QsaWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4QsaWeights {
        indexer_qk: dense_ref(weights, &qsa.indexer_qk)?,
        indexer_q_norm: weights.resident(&qsa.indexer_q_norm)?,
        indexer_k_norm: weights.resident(&qsa.indexer_k_norm)?,
        q: dense_ref(weights, &qsa.q)?,
        k: dense_ref(weights, &qsa.k)?,
        v: dense_ref(weights, &qsa.v)?,
        q_norm: weights.resident(&qsa.q_norm)?,
        k_norm: weights.resident(&qsa.k_norm)?,
        output: dense_ref(weights, &qsa.output)?,
    })
}
fn ple_desc<'a>(
    weights: &'a Qwen4Weights,
    ple: &crate::weights::PleWeights,
) -> Result<Qwen4PleWeights<'a>, Qwen4GpuForwardError> {
    Ok(Qwen4PleWeights {
        key: dense_ref(weights, &ple.key)?,
        value: dense_ref(weights, &ple.value)?,
        norm_key: weights.resident(&ple.norm_key)?,
        norm_query: weights.resident(&ple.norm_query)?,
        norm_conv: weights.resident(&ple.norm_conv)?,
        conv: weights.resident(&ple.conv)?,
    })
}

fn layer_desc<'a>(
    weights: &'a Qwen4Weights,
    layer: &Qwen4LayerWeights,
    moe: &'a Qwen4MoeLayerRuntime,
    config: &Qwen4Config,
) -> Result<Qwen4LayerDescription<'a>, Qwen4GpuForwardError> {
    let attention = match layer.kind {
        LayerType::LinearAttention => Qwen4AttentionWeights::Linear(gdn_desc(
            weights,
            layer
                .gdn
                .as_ref()
                .ok_or_else(|| invalid("GDN weights missing"))?,
        )?),
        LayerType::FullAttention => Qwen4AttentionWeights::Full(qsa_desc(
            weights,
            layer
                .attention
                .as_ref()
                .ok_or_else(|| invalid("QSA weights missing"))?,
        )?),
    };
    Ok(Qwen4LayerDescription {
        attn_hyper: hyper_desc(weights, &layer.attn_hyper)?,
        mlp_hyper: hyper_desc(weights, &layer.mlp_hyper)?,
        attention,
        moe: Qwen4MoeBinding {
            route_policy: qwen4_route_policy(config, moe.shared_down.dtype),
            table: &moe.table,
            cache: &moe.cache,
            routed_experts: moe,
            router: moe.router.dispatch_ref(),
            shared: MoeSharedWeights {
                selector: moe.shared_scalar.dispatch_ref(),
                gate: moe.shared_gate.dispatch_ref(),
                up: moe.shared_up.dispatch_ref(),
                down: moe.shared_down.dispatch_ref(),
            },
            intermediate: moe
                .experts
                .first()
                .map(|expert| expert.gate_up.m)
                .unwrap_or_default()
                / 2,
            experts_all_gate_up_mq4: moe.experts_all_gate_up_mq4,
            expert_gate_up_ptrs: &moe.expert_gate_up_ptrs,
            expert_down_ptrs: &moe.expert_down_ptrs,
            expert_gate_up_entries: &moe.expert_gate_up_entries,
            expert_down_entries: &moe.expert_down_entries,
            layer_idx: layer.layer as u16,
            norm_topk_prob: config.norm_topk_prob,
        },
    })
}

pub(crate) struct Qwen4ExpertView {
    gate_up: ProjectionView,
    down: ProjectionView,
}

struct ExpertViewSet<'a> {
    experts: &'a [Qwen4ExpertView],
}

impl RoutedExpertWeights for ExpertViewSet<'_> {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

/// Architecture-owned sealed-MoE resources.  Pointer tables and expert views
/// are built once with the assembled resident weights and reused for every
/// token; no per-token resource or metadata allocation occurs.
pub(crate) struct Qwen4MoeLayerRuntime {
    table: ExpertTable,
    cache: ExpertBindingCache,
    router: ProjectionView,
    shared_scalar: ProjectionView,
    shared_gate: ProjectionView,
    shared_up: ProjectionView,
    shared_down: ProjectionView,
    experts: Vec<Qwen4ExpertView>,
    experts_all_gate_up_mq4: bool,
    expert_gate_up_ptrs: GpuTensor,
    expert_down_ptrs: GpuTensor,
    /// Host entries the two tables above were uploaded from. Kept so dispatch can
    /// prove the table contents name the live expert tensors before a retained
    /// body records them.
    expert_gate_up_entries: Vec<usize>,
    expert_down_entries: Vec<usize>,
}

impl RoutedExpertWeights for Qwen4MoeLayerRuntime {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

impl Qwen4MoeLayerRuntime {
    fn new(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        layer: &Qwen4LayerWeights,
        config: &Qwen4Config,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::from_moe(gpu, weights, &layer.moe, layer.layer, config)
    }

    pub(crate) fn from_moe(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let router_source = weights.resident(&moe.gate)?;
        let scalar_source = weights.resident(&moe.shared_gate_scalar)?;
        let shared_gate_source = weights.resident(&moe.shared_gate)?;
        let shared_up_source = weights.resident(&moe.shared_up)?;
        let shared_down_source = weights.resident(&moe.shared_down)?;
        let router =
            ProjectionView::from_source(router_source, config.num_experts, config.hidden_size);
        let shared_scalar = ProjectionView::from_source(scalar_source, 1, config.hidden_size);
        let shared_gate = ProjectionView::from_source(
            shared_gate_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_up = ProjectionView::from_source(
            shared_up_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_down = ProjectionView::from_source(
            shared_down_source,
            config.hidden_size,
            config.shared_expert_intermediate_size,
        );

        let gate_up_source = weights.resident(&moe.experts_gate_up)?;
        let down_source = weights.resident(&moe.experts_down)?;
        let gate_up_dtype = gate_up_source.dtype;
        let down_dtype = down_source.dtype;
        let gate_up_rows = 2 * config.moe_intermediate_size;
        let gate_up_row_stride = row_stride(gate_up_dtype, config.hidden_size);
        let down_row_stride = row_stride(down_dtype, config.moe_intermediate_size);
        let gate_up_expert_bytes = checked_bytes(gate_up_rows, gate_up_row_stride)?;
        let down_expert_bytes = checked_bytes(config.hidden_size, down_row_stride)?;
        let expected_gate_up_bytes = checked_bytes(config.num_experts, gate_up_expert_bytes)?;
        let expected_down_bytes = checked_bytes(config.num_experts, down_expert_bytes)?;
        if gate_up_source.buf.size() < expected_gate_up_bytes {
            return Err(invalid(format!(
                "layer {} gate/up packed bytes {} < expected {}",
                layer_label,
                gate_up_source.buf.size(),
                expected_gate_up_bytes
            )));
        }
        if down_source.buf.size() < expected_down_bytes {
            return Err(invalid(format!(
                "layer {} down packed bytes {} < expected {}",
                layer_label,
                down_source.buf.size(),
                expected_down_bytes
            )));
        }

        let (gate_up_source_name, down_source_name) = match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
                (gate_up.as_str(), down.as_str())
            }
            layout => {
                return Err(invalid(format!(
                    "layer {} uses unsupported expert source layout {layout:?}",
                    layer_label
                )));
            }
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        let mut metadata = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            let gate_up_offset = expert * gate_up_expert_bytes;
            let down_offset = expert * down_expert_bytes;
            let gate_up = ProjectionView::from_packed(
                gate_up_source,
                gate_up_offset,
                gate_up_expert_bytes,
                gate_up_dtype,
                gate_up_rows,
                config.hidden_size,
            );
            let down = ProjectionView::from_packed(
                down_source,
                down_offset,
                down_expert_bytes,
                down_dtype,
                config.hidden_size,
                config.moe_intermediate_size,
            );
            experts.push(Qwen4ExpertView { gate_up, down });

            let gate_up_resource = ExpertResource::new(
                gate_up_source_name,
                "qwen4-resident-mqv2-v1",
                vec![gate_up_rows, config.hidden_size],
                gate_up_dtype,
                gate_up_expert_bytes,
                gate_up_row_stride,
                16,
                dtype_rotation_plan(gate_up_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let down_resource = ExpertResource::new(
                down_source_name,
                "qwen4-resident-mqv2-v1",
                vec![config.hidden_size, config.moe_intermediate_size],
                down_dtype,
                down_expert_bytes,
                down_row_stride,
                16,
                dtype_rotation_plan(down_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let resources =
                ExpertResources::fused(gate_up_resource, down_resource).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert resources: {error:?}"))
                })?;
            metadata.push(
                ExpertMetadata::new(expert, 0, expert, resources).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert metadata: {error:?}"))
                })?,
            );
        }
        let table = ExpertTable::new(metadata)
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("expert table: {error:?}")))?;
        let mut cache = table
            .prepare_binding(0, 1, gpu.device_id)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert binding: {error:?}"))
            })?;

        let gate_entries = experts
            .iter()
            .map(|expert| expert.gate_up.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let down_entries = experts
            .iter()
            .map(|expert| expert.down.tensor.buf.as_ptr() as usize)
            .collect::<Vec<_>>();
        let gate_ptrs = gate_entries
            .iter()
            .flat_map(|entry| entry.to_ne_bytes())
            .collect::<Vec<_>>();
        let down_ptrs = down_entries
            .iter()
            .flat_map(|entry| entry.to_ne_bytes())
            .collect::<Vec<_>>();
        let expert_gate_up_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => return Err(error.into()),
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_gate_up_ptrs.buf, &gate_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            return Err(error.into());
        }
        let expert_down_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                return Err(error.into());
            }
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_down_ptrs.buf, &down_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(error.into());
        }
        let expert_views = ExpertViewSet { experts: &experts };
        if let Err(error) = cache.bind_live(
            &table,
            &expert_views,
            &expert_gate_up_ptrs,
            &expert_down_ptrs,
            None,
            None,
        ) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "expert live binding: {error:?}"
            )));
        }
        Ok(Self {
            table,
            cache,
            router,
            shared_scalar,
            shared_gate,
            shared_up,
            shared_down,
            experts_all_gate_up_mq4: gate_up_dtype == DType::MQ4G256V2,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
            expert_gate_up_entries: gate_entries,
            expert_down_entries: down_entries,
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.expert_gate_up_ptrs, self.expert_down_ptrs] {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

/// Scratch views required by one sealed Qwen4 MoE invocation.  The caller owns
/// these reusable buffers; the sealed expert table and live pointer resources
/// remain in [`Qwen4MoeLayerRuntime`].
pub(crate) struct Qwen4MoeScratch<'a> {
    pub(crate) router_logits: &'a GpuTensor,
    pub(crate) scalar_buf: &'a GpuTensor,
    pub(crate) x_rot_local: &'a GpuTensor,
    pub(crate) gate_up_buf: &'a GpuTensor,
    pub(crate) gate_buf: &'a GpuTensor,
    pub(crate) up_buf: &'a GpuTensor,
    pub(crate) ffn_hidden: &'a GpuTensor,
    pub(crate) ffn_out: &'a GpuTensor,
    pub(crate) gate_batch: &'a GpuTensor,
    pub(crate) up_batch: &'a GpuTensor,
    pub(crate) rot_batch: &'a GpuTensor,
    pub(crate) topk_indices: &'a GpuTensor,
    pub(crate) topk_weights: &'a GpuTensor,
    pub(crate) down_expanded: &'a GpuTensor,
}

pub(crate) fn execute_moe(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    layer_index: usize,
    runtime: &Qwen4MoeLayerRuntime,
    input: &GpuTensor,
    output: &GpuTensor,
    scratch: Qwen4MoeScratch<'_>,
) -> Result<(), Qwen4GpuForwardError> {
    let ctx = DispatchCtx::new(gpu);
    gpu.hip.memset(&output.buf, 0, output.buf.size())?;
    let dtypes = MoeDtypes {
        router: runtime.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: runtime.shared_scalar.dtype,
            gate: runtime.shared_gate.dtype,
            up: runtime.shared_up.dtype,
            down: runtime.shared_down.dtype,
        }),
        experts_all_gate_up_mq4: runtime
            .experts
            .iter()
            .all(|expert| expert.gate_up.dtype == DType::MQ4G256V2),
        routed_gate_up: runtime
            .experts
            .first()
            .map(|expert| expert.gate_up.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_down: runtime
            .experts
            .first()
            .map(|expert| expert.down.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let params = MoeParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        },
        route_policy: Some(qwen4_route_policy(config, runtime.shared_down.dtype)),
        normalization: MoeNormalization::Provided,
        batch_size: 1,
        hidden: config.hidden_size,
        mi: config.moe_intermediate_size,
        k: config.num_experts_per_tok,
        n_exp: config.num_experts,
        norm_topk_prob: config.norm_topk_prob,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        ep_mode: MoeEpMode::None,
        layer_idx: layer_index as u16,
        x_norm: input,
        x_residual: output,
        routed_out: None,
        skip_shared: false,
        router: runtime.router.dispatch_ref(),
        shared: Some(MoeSharedDecode {
            weights: MoeSharedWeights {
                selector: runtime.shared_scalar.dispatch_ref(),
                gate: runtime.shared_gate.dispatch_ref(),
                up: runtime.shared_up.dispatch_ref(),
                down: runtime.shared_down.dispatch_ref(),
            },
            intermediate: config.shared_expert_intermediate_size,
            scalar: scratch.scalar_buf,
            gate_out: scratch.gate_buf,
            up_out: scratch.up_buf,
        }),
        expert_gate_up_ptrs: &runtime.expert_gate_up_ptrs,
        expert_down_ptrs: &runtime.expert_down_ptrs,
        expert_ptrs_host: None,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: config.hidden_size,
        routed_down_m: config.hidden_size,
        routed_down_k: config.moe_intermediate_size,
        routed_experts: runtime,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.router_logits,
        x_rot_local: scratch.x_rot_local,
        gate_up_buf: scratch.gate_up_buf,
        ffn_hidden: scratch.ffn_hidden,
        ffn_out: scratch.ffn_out,
        gate_batch: scratch.gate_batch,
        up_batch: scratch.up_batch,
        rot_batch: scratch.rot_batch,
        topk_indices: scratch.topk_indices,
        topk_weights: scratch.topk_weights,
        down_expanded: scratch.down_expanded,
    };
    let bound = BoundMoeExperts::from_cache(&runtime.table, &runtime.cache)
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("bound MoE experts: {error:?}")))?;
    let seal_started = qwen4_profile_start();
    let sealed_result = seal_decode(bound, &ctx, params);
    qwen4_profile_record(Qwen4ProfilePhase::MoeSeal, seal_started);
    let sealed = sealed_result
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}")))?;
    execute_steps(gpu, &ctx, &[Step::Moe(sealed)])
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 MoE: {error:?}")))?;
    Ok(())
}

/// Device buffers shared by one forward object.  All allocations happen in
/// `new`; token/chunk execution only creates borrowed subviews.
pub struct Qwen4GpuForwardScratch {
    pub max_chunk: usize,
    pub token_ids: GpuTensor,
    pub embedding_rot: GpuTensor,
    pub embeddings: GpuTensor,
    pub streams: GpuTensor,
    pub hc_normalized: GpuTensor,
    pub hc_low: GpuTensor,
    pub hc_up: GpuTensor,
    pub hc_mixed: GpuTensor,
    pub hc_gates: GpuTensor,
    pub rotation: GpuTensor,
    pub projection: GpuTensor,
    pub projection2: GpuTensor,
    pub gdn_a: GpuTensor,
    pub gdn_b: GpuTensor,
    pub gdn_gate: GpuTensor,
    pub gdn_beta: GpuTensor,
    pub gdn_recurrent_output: GpuTensor,
    pub gdn_bf16: GpuTensor,
    pub gdn_z: GpuTensor,
    pub gdn_output: GpuTensor,
    pub qsa_index: GpuTensor,
    pub qsa_qgate: GpuTensor,
    pub qsa_k: GpuTensor,
    pub qsa_v: GpuTensor,
    pub qsa_output: GpuTensor,
    pub qsa_selected: GpuTensor,
    pub attention_output: GpuTensor,
    pub ple_staged: GpuTensor,
    pub ple_rows: GpuTensor,
    pub ple_key: GpuTensor,
    pub ple_value: GpuTensor,
    pub ple_query: GpuTensor,
    pub ple_gated: GpuTensor,
    pub ple_normed: GpuTensor,
    pub ple_output: GpuTensor,
    pub router_logits: GpuTensor,
    pub moe_x_rot: GpuTensor,
    pub moe_gate_up: GpuTensor,
    pub moe_gate: GpuTensor,
    pub moe_up: GpuTensor,
    pub moe_hidden: GpuTensor,
    /// Separate shared-expert down output.  The public MoE output is the
    /// routed accumulator; aliasing these buffers makes the shared GEMV
    /// overwrite the routed contribution before the source-ordered add.
    pub moe_shared_output: GpuTensor,
    pub moe_output: GpuTensor,
    pub moe_gate_batch: GpuTensor,
    pub moe_up_batch: GpuTensor,
    pub moe_rot_batch: GpuTensor,
    pub moe_topk_indices: GpuTensor,
    pub moe_topk_weights: GpuTensor,
    pub moe_down_expanded: GpuTensor,
    pub moe_scalar: GpuTensor,
    pub moe_expert_token_counts: GpuTensor,
    pub moe_expert_offsets: GpuTensor,
    pub moe_sorted_slot_index: GpuTensor,
    pub moe_expert_tile_ids: GpuTensor,
    pub moe_inverse_perm: GpuTensor,
    pub moe_y_gate_up_grouped: GpuTensor,
    pub moe_y_down_grouped: GpuTensor,
    pub logits: GpuTensor,
}

impl Qwen4GpuForwardScratch {
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        if max_chunk == 0 {
            return Err(invalid("max_chunk is zero"));
        }
        let max_chunk = max_chunk.min(QWEN4_PREFILL_CHUNK_CAP);
        let dims = program_dims(config);
        let layout = Qwen4ScratchLayout::for_rows(dims, max_chunk)
            .map_err(|error| invalid(error.to_string()))?;
        let hidden = layout.hidden;
        let wide = layout.wide;
        let q_width = layout.q_width;
        let qsa_qgate = 2 * q_width;
        let qsa_index = layout.index_width;
        let gdn_value = layout.gdn_value;
        let gdn_qkv = layout.gdn_qkv;
        let max_projection = wide.max(qsa_qgate).max(gdn_qkv).max(hidden);
        let max_rotation = wide.max(hidden).max(config.moe_intermediate_size);
        let ple_channels = config.ple_embed_dim * config.hc_count;
        let max_ple_bytes = max_chunk
            .checked_mul(PLE_HEAD_COUNT)
            .and_then(|bytes| bytes.checked_mul(PLE_ROW_WIDTH * 2))
            .ok_or_else(|| invalid("PLE staging size overflow"))?;
        let i32_bytes = std::mem::size_of::<i32>();
        let qsa_selected_bytes = max_chunk
            .checked_mul(config.qsa_selected_capacity())
            .and_then(|elements| elements.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("QSA selected scratch size overflow"))?;
        let route_i32_bytes = layout
            .slots
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE route index scratch overflow"))?;
        let expert_i32_bytes = config
            .num_experts
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE expert index scratch overflow"))?;
        let expert_offsets_bytes = config
            .num_experts
            .checked_add(1)
            .and_then(|count| count.checked_mul(i32_bytes))
            .ok_or_else(|| invalid("MoE expert offset scratch overflow"))?;
        let grouped_bound = crate::program::grouped_m_total_bound(layout.slots, config.num_experts)
            .map_err(|error| invalid(error.to_string()))?;
        let grouped_i32_bytes = grouped_bound
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped index scratch overflow"))?;
        let tile_i32_bytes = (grouped_bound / 16)
            .checked_mul(i32_bytes)
            .ok_or_else(|| invalid("MoE grouped tile scratch overflow"))?;
        let mut allocated = Vec::new();
        let mut alloc = |shape: &[usize], dtype: DType| -> Result<(), Qwen4GpuForwardError> {
            let tensor = gpu.zeros(shape, dtype)?;
            allocated.push(tensor);
            Ok(())
        };
        let result = (|| {
            alloc(&[max_chunk * std::mem::size_of::<i32>()], DType::Raw)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * wide], DType::F32)?;
            alloc(&[max_chunk * wide], DType::F32)?;
            alloc(&[max_chunk * config.hc_lowrank], DType::F32)?;
            alloc(&[max_chunk * wide], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * config.hc_count], DType::F32)?;
            alloc(&[max_chunk * max_rotation], DType::F32)?;
            alloc(&[max_chunk * max_projection], DType::F32)?;
            alloc(&[max_chunk * max_projection], DType::F32)?;
            alloc(&[max_chunk * config.linear_num_value_heads], DType::F32)?;
            alloc(&[max_chunk * config.linear_num_value_heads], DType::F32)?;
            alloc(&[max_chunk * config.linear_num_value_heads], DType::F32)?;
            alloc(&[max_chunk * config.linear_num_value_heads], DType::F32)?;
            alloc(&[max_chunk * gdn_value], DType::F32)?;
            alloc(&[gdn_value], DType::BF16)?;
            alloc(&[max_chunk * gdn_value], DType::F32)?;
            alloc(&[max_chunk * gdn_value], DType::F32)?;
            alloc(&[max_chunk * qsa_index], DType::F32)?;
            alloc(&[max_chunk * qsa_qgate], DType::F32)?;
            alloc(
                &[max_chunk * config.num_key_value_heads * config.head_dim],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.num_key_value_heads * config.head_dim],
                DType::F32,
            )?;
            alloc(&[max_chunk * q_width], DType::F32)?;
            alloc(&[qsa_selected_bytes], DType::Raw)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * PLE_HEAD_COUNT * PLE_ROW_WIDTH], DType::BF16)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * ple_channels], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * ple_channels], DType::F32)?;
            alloc(&[max_chunk * ple_channels], DType::F32)?;
            alloc(&[max_chunk * ple_channels], DType::F32)?;
            alloc(&[max_chunk * ple_channels], DType::F32)?;
            alloc(&[max_chunk * config.num_experts], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * 2 * config.moe_intermediate_size], DType::F32)?;
            alloc(
                &[max_chunk * config.shared_expert_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.shared_expert_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.shared_expert_intermediate_size],
                DType::F32,
            )?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(
                &[max_chunk * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.num_experts_per_tok * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(&[route_i32_bytes], DType::Raw)?;
            alloc(&[max_chunk * config.num_experts_per_tok], DType::F32)?;
            alloc(
                &[max_chunk * config.num_experts_per_tok * hidden],
                DType::F32,
            )?;
            alloc(
                &[max_chunk * config.shared_expert_intermediate_size],
                DType::F32,
            )?;
            alloc(&[expert_i32_bytes], DType::Raw)?;
            alloc(&[expert_offsets_bytes], DType::Raw)?;
            alloc(&[grouped_i32_bytes], DType::Raw)?;
            alloc(&[tile_i32_bytes], DType::Raw)?;
            alloc(&[route_i32_bytes], DType::Raw)?;
            alloc(
                &[grouped_bound * 2 * config.moe_intermediate_size],
                DType::F32,
            )?;
            alloc(&[grouped_bound * hidden], DType::F32)?;
            alloc(&[config.vocab_size], DType::F32)?;
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut next = || allocated.remove(0);
        Ok((
            Self {
                max_chunk,
                token_ids: next(),
                embedding_rot: next(),
                embeddings: next(),
                streams: next(),
                hc_normalized: next(),
                hc_low: next(),
                hc_up: next(),
                hc_mixed: next(),
                hc_gates: next(),
                rotation: next(),
                projection: next(),
                projection2: next(),
                gdn_a: next(),
                gdn_b: next(),
                gdn_gate: next(),
                gdn_beta: next(),
                gdn_recurrent_output: next(),
                gdn_bf16: next(),
                gdn_z: next(),
                gdn_output: next(),
                qsa_index: next(),
                qsa_qgate: next(),
                qsa_k: next(),
                qsa_v: next(),
                qsa_output: next(),
                qsa_selected: next(),
                attention_output: next(),
                ple_staged: next(),
                ple_rows: next(),
                ple_key: next(),
                ple_value: next(),
                ple_query: next(),
                ple_gated: next(),
                ple_normed: next(),
                ple_output: next(),
                router_logits: next(),
                moe_x_rot: next(),
                moe_gate_up: next(),
                moe_gate: next(),
                moe_up: next(),
                moe_hidden: next(),
                moe_output: next(),
                moe_shared_output: next(),
                moe_gate_batch: next(),
                moe_up_batch: next(),
                moe_rot_batch: next(),
                moe_topk_indices: next(),
                moe_topk_weights: next(),
                moe_down_expanded: next(),
                moe_scalar: next(),
                moe_expert_token_counts: next(),
                moe_expert_offsets: next(),
                moe_sorted_slot_index: next(),
                moe_expert_tile_ids: next(),
                moe_inverse_perm: next(),
                moe_y_gate_up_grouped: next(),
                moe_y_down_grouped: next(),
                logits: next(),
            },
            vec![0; max_chunk * std::mem::size_of::<i32>()],
            vec![0; max_ple_bytes],
        ))
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let tensors = [
            self.token_ids,
            self.embedding_rot,
            self.embeddings,
            self.streams,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.hc_gates,
            self.rotation,
            self.projection,
            self.projection2,
            self.gdn_a,
            self.gdn_b,
            self.gdn_gate,
            self.gdn_beta,
            self.gdn_recurrent_output,
            self.gdn_bf16,
            self.gdn_z,
            self.gdn_output,
            self.qsa_index,
            self.qsa_qgate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
            self.qsa_selected,
            self.attention_output,
            self.ple_staged,
            self.ple_rows,
            self.ple_key,
            self.ple_value,
            self.ple_query,
            self.ple_gated,
            self.ple_normed,
            self.ple_output,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.moe_output,
            self.moe_shared_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_scalar,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
            self.logits,
        ];
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

fn layer_scratch<'a>(
    scratch: &'a Qwen4GpuForwardScratch,
    router_logits: &'a GpuTensor,
) -> Qwen4LayerScratch<'a> {
    Qwen4LayerScratch {
        streams: &scratch.streams,
        rotation: &scratch.rotation,
        hc_normalized: &scratch.hc_normalized,
        hc_low: &scratch.hc_low,
        hc_up: &scratch.hc_up,
        hc_mixed: &scratch.hc_mixed,
        hc_gates: &scratch.hc_gates,
        projection: &scratch.projection,
        projection2: &scratch.projection2,
        gdn_a: &scratch.gdn_a,
        gdn_b: &scratch.gdn_b,
        gdn_gate: &scratch.gdn_gate,
        gdn_beta: &scratch.gdn_beta,
        gdn_recurrent_output: &scratch.gdn_recurrent_output,
        gdn_bf16: &scratch.gdn_bf16,
        gdn_z: &scratch.gdn_z,
        gdn_output: &scratch.gdn_output,
        qsa_index: &scratch.qsa_index,
        qsa_qgate: &scratch.qsa_qgate,
        qsa_k: &scratch.qsa_k,
        qsa_v: &scratch.qsa_v,
        qsa_output: &scratch.qsa_output,
        attention_output: &scratch.attention_output,
        moe_output: &scratch.moe_output,
        moe_shared_output: &scratch.moe_shared_output,
        moe_router_logits: router_logits,
        moe_x_rot: &scratch.moe_x_rot,
        moe_gate_up: &scratch.moe_gate_up,
        moe_scalar: &scratch.moe_scalar,
        moe_gate: &scratch.moe_gate,
        moe_up: &scratch.moe_up,
        moe_hidden: &scratch.moe_hidden,
        moe_gate_batch: &scratch.moe_gate_batch,
        moe_up_batch: &scratch.moe_up_batch,
        moe_rot_batch: &scratch.moe_rot_batch,
        moe_topk_indices: &scratch.moe_topk_indices,
        moe_topk_weights: &scratch.moe_topk_weights,
        moe_down_expanded: &scratch.moe_down_expanded,
        moe_expert_token_counts: &scratch.moe_expert_token_counts,
        moe_expert_offsets: &scratch.moe_expert_offsets,
        moe_sorted_slot_index: &scratch.moe_sorted_slot_index,
        moe_expert_tile_ids: &scratch.moe_expert_tile_ids,
        moe_inverse_perm: &scratch.moe_inverse_perm,
        moe_y_gate_up_grouped: &scratch.moe_y_gate_up_grouped,
        moe_y_down_grouped: &scratch.moe_y_down_grouped,
    }
}

/// Reusable production forward owner.  Construct it once per assembled bundle
/// and use `forward_token`/`forward_chunk` repeatedly; both methods share the
/// exact same chunk implementation.
pub struct Qwen4GpuForward {
    pub scratch: Qwen4GpuForwardScratch,
    host_token_bytes: Vec<u8>,
    host_ple_bytes: Vec<u8>,
    moe: Vec<Qwen4MoeLayerRuntime>,
}

impl Qwen4GpuForward {
    pub fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let (scratch, host_token_bytes, host_ple_bytes) =
            Qwen4GpuForwardScratch::new(gpu, &bundle.config, max_chunk)?;
        let mut moe = Vec::with_capacity(bundle.config.num_hidden_layers);
        let result = (|| {
            for layer in &bundle.weights.layer_refs {
                moe.push(Qwen4MoeLayerRuntime::new(
                    gpu,
                    &bundle.weights,
                    layer,
                    &bundle.config,
                )?);
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            for layer in moe {
                let _ = layer.free_gpu(gpu);
            }
            let _ = scratch.free_gpu(gpu);
            return Err(error);
        }
        Ok(Self {
            scratch,
            host_token_bytes,
            host_ple_bytes,
            moe,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let Qwen4GpuForward { scratch, moe, .. } = self;
        let mut first = scratch.free_gpu(gpu);
        for layer in moe {
            if let Some(error) = layer.free_gpu(gpu) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first.map_or(Ok(()), Err)
    }

    fn validate_request(
        &self,
        bundle: &Qwen4Bundle,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        if tokens.is_empty() {
            return Err(invalid("Qwen4 forward cannot process an empty token slice"));
        }
        let config = &bundle.config;
        let end_position = bundle
            .state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if tokens
            .iter()
            .copied()
            .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        let requested_rows = output_rows.count(tokens.len());
        let expected_logits = requested_rows
            .checked_mul(config.vocab_size)
            .ok_or_else(|| invalid("Qwen4 logits shape overflows"))?;
        if logits.dtype != DType::F32 || logits.numel() != expected_logits {
            return Err(invalid(format!(
                "logits must be F32 with {expected_logits} elements for {output_rows:?} output"
            )));
        }
        if let Some(top1) = top1 {
            let expected = requested_rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or_else(|| invalid("Qwen4 top1 shape overflows"))?;
            if top1.dtype != DType::Raw || top1.buf.size() < expected {
                return Err(invalid(
                    "top1 output must be Raw with one i32 per requested output row",
                ));
            }
        }
        if let Some(capture) = wide_hidden_capture {
            let expected = tokens
                .len()
                .checked_mul(program_dims(config).wide())
                .ok_or_else(|| invalid("Qwen4 wide hidden capture shape overflows"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        Ok(())
    }

    fn preflight_output_resources(
        &self,
        bundle: &Qwen4Bundle,
        rows: usize,
        logits: &GpuTensor,
        requested_rows: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let config = &bundle.config;
        let dims = program_dims(config);
        let final_hyper = hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?;
        let lm_head = dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?;
        let preflight_scratch = layer_scratch(&self.scratch, &self.scratch.router_logits);
        validate_final_hyper(
            dims,
            &final_hyper,
            &self.scratch.streams,
            &preflight_scratch,
            rows,
        )
        .map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
        })?;
        let hidden_elements = rows
            .checked_mul(config.hidden_size)
            .ok_or_else(|| invalid("Qwen4 final hidden shape overflows"))?;
        let final_hidden = view(&self.scratch.hc_mixed, 0, hidden_elements);
        validate_lm_head(&lm_head, &final_hidden, logits, rows, requested_rows).map_err(
            |error| Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}")),
        )?;
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&token),
            logits,
            top1,
            None,
            Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final),
        )
    }

    /// Run bounded forward tiles, writing all logits rows or only the last.
    /// Final-only intermediate tiles still commit model state but skip the head.
    pub fn forward_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_rows: Qwen4OutputRows,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.validate_request(
            bundle,
            tokens,
            logits,
            top1,
            wide_hidden_capture,
            output_rows,
        )?;
        let vocab = bundle.config.vocab_size;
        let max_chunk = self.scratch.max_chunk;
        let preflight_rows = if output_rows == Qwen4OutputRows::Final {
            (tokens.len() - 1) % max_chunk + 1
        } else {
            tokens.len().min(max_chunk)
        };
        let preflight_output_rows = output_rows.count(preflight_rows);
        let preflight_logits = logits.sub_offset(0, preflight_output_rows * vocab);
        self.preflight_output_resources(
            bundle,
            preflight_rows,
            &preflight_logits,
            preflight_output_rows,
        )?;
        let capture_width = wide_hidden_capture.map(|_| program_dims(&bundle.config).wide());
        let mut offset = 0usize;
        while offset < tokens.len() {
            let rows = (tokens.len() - offset).min(max_chunk);
            let final_chunk = offset + rows == tokens.len();
            let selected_rows = output_rows.count(rows);
            let logits_offset = if output_rows == Qwen4OutputRows::All {
                offset
                    .checked_mul(vocab)
                    .ok_or_else(|| invalid("Qwen4 chunk logits offset overflows"))?
            } else {
                0
            };
            let logits_len = selected_rows
                .checked_mul(vocab)
                .ok_or_else(|| invalid("Qwen4 chunk logits shape overflows"))?;
            let logits_chunk = logits.sub_offset(logits_offset, logits_len);
            let top1_chunk = top1
                .filter(|_| output_rows == Qwen4OutputRows::All || final_chunk)
                .map(|destination| {
                    destination.sub_offset(
                        if output_rows == Qwen4OutputRows::All {
                            offset * std::mem::size_of::<i32>()
                        } else {
                            0
                        },
                        selected_rows * std::mem::size_of::<i32>(),
                    )
                });
            let capture_chunk = wide_hidden_capture
                .zip(capture_width)
                .map(|(capture, wide)| {
                    let capture_offset = offset
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture offset overflows"))?;
                    let capture_len = rows
                        .checked_mul(wide)
                        .ok_or_else(|| invalid("Qwen4 wide capture shape overflows"))?;
                    Ok::<_, Qwen4GpuForwardError>(capture.sub_offset(capture_offset, capture_len))
                })
                .transpose()?;
            let output_policy = if output_rows == Qwen4OutputRows::All || final_chunk {
                Qwen4OutputPolicy::Rows(output_rows)
            } else {
                Qwen4OutputPolicy::None
            };
            self.forward_chunk_inner(
                bundle,
                gpu,
                &tokens[offset..offset + rows],
                &logits_chunk,
                top1_chunk.as_ref(),
                capture_chunk.as_ref(),
                output_policy,
            )?;
            offset = offset
                .checked_add(rows)
                .ok_or_else(|| invalid("Qwen4 chunk offset overflows"))?;
        }
        Ok(())
    }

    fn forward_chunk_inner(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_policy: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        let n = tokens.len();
        if n == 0 || n > self.scratch.max_chunk {
            return Err(invalid(format!(
                "chunk length {n} exceeds configured capacity {}",
                self.scratch.max_chunk
            )));
        }
        let config = bundle.config.clone();
        let end_position = bundle
            .state
            .position
            .checked_add(n)
            .ok_or_else(|| invalid("Qwen4 forward position overflows"))?;
        if end_position > bundle.state.max_seq_len {
            return Err(invalid(format!(
                "Qwen4 forward end position {end_position} exceeds context capacity {}",
                bundle.state.max_seq_len
            )));
        }
        if tokens
            .iter()
            .copied()
            .any(|token| (token as usize) >= config.vocab_size)
        {
            return Err(invalid(
                "Qwen4 token id is outside the embedding vocabulary",
            ));
        }
        let dims = program_dims(&config);
        let requested_rows = output_policy.requested_rows(n);
        if let Some(requested_rows) = requested_rows {
            let expected_logits = requested_rows
                .checked_mul(config.vocab_size)
                .ok_or_else(|| invalid("logit shape overflow"))?;
            if logits.dtype != DType::F32 || logits.numel() != expected_logits {
                return Err(invalid(format!(
                    "logits must be F32 with {expected_logits} elements for {output_policy:?} output"
                )));
            }
            if let Some(top1) = top1 {
                let expected = requested_rows
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| invalid("top1 shape overflow"))?;
                if top1.dtype != DType::Raw || top1.buf.size() < expected {
                    return Err(invalid(
                        "top1 output must be Raw with one i32 per requested output row",
                    ));
                }
            }
        } else if top1.is_some() {
            return Err(invalid("Qwen4 discarded output cannot receive top1 rows"));
        }
        if let Some(capture) = wide_hidden_capture.as_ref() {
            let expected = n
                .checked_mul(dims.wide())
                .ok_or_else(|| invalid("wide hidden capture shape overflow"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        let ple_layer_index = config
            .ple_layer_ids
            .first()
            .and_then(|id| id.checked_sub(1))
            .ok_or_else(|| invalid("PLE layer id missing"))?;
        if ple_layer_index >= config.num_hidden_layers {
            return Err(invalid("PLE layer id is outside the trunk"));
        }
        let final_hyper = if requested_rows.is_some() {
            Some(hyper_read_desc(
                &bundle.weights,
                &bundle.weights.root.final_hyper,
            )?)
        } else {
            None
        };
        let lm_head = if requested_rows.is_some() {
            Some(dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?)
        } else {
            None
        };
        if let Some(requested_rows) = requested_rows {
            let final_hyper = final_hyper
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 final hyper preflight disappeared"))?;
            let lm_head = lm_head
                .as_ref()
                .ok_or_else(|| invalid("Qwen4 LM head preflight disappeared"))?;
            let preflight_scratch = layer_scratch(&self.scratch, &self.scratch.router_logits);
            validate_final_hyper(
                dims,
                final_hyper,
                &self.scratch.streams,
                &preflight_scratch,
                n,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 final hyper: {error:?}"))
            })?;
            let hidden_elements = n
                .checked_mul(config.hidden_size)
                .ok_or_else(|| invalid("Qwen4 final hidden shape overflow"))?;
            let final_hidden = view(&self.scratch.hc_mixed, 0, hidden_elements);
            validate_lm_head(lm_head, &final_hidden, logits, n, requested_rows).map_err(
                |error| {
                    Qwen4GpuForwardError::Dispatch(format!("validate Qwen4 LM head: {error:?}"))
                },
            )?;
        }

        let router_logits = matrix_view(&self.scratch.router_logits, n, dims.num_experts)?;
        let scratch_desc = layer_scratch(&self.scratch, &router_logits);
        // Decided once: every op touching the HC streams in this forward agrees.
        let bf16_state = gpu.qwen4_bf16_streams(n);
        if config.num_hidden_layers.saturating_mul(7).saturating_add(1) > QWEN4_STEP_INLINE_CAPACITY
        {
            return Err(invalid(
                "Qwen4 generic layer program inline capacity exhausted",
            ));
        }
        let ctx = DispatchCtx::new(gpu);
        let mut steps: SmallVec<[Step<'_>; QWEN4_STEP_INLINE_CAPACITY]> = SmallVec::new();
        let mut gdn_slot = 0usize;
        let mut qsa_slot = 0usize;
        for layer_index in 0..config.num_hidden_layers {
            let layer = &bundle.weights.layer_refs[layer_index];
            if layer_index == ple_layer_index {
                let ple_weights = layer
                    .ple
                    .as_ref()
                    .ok_or_else(|| invalid("PLE weights missing"))?;
                if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                    return Err(invalid(
                        "Qwen4 generic layer program inline capacity exhausted",
                    ));
                }
                let ple = ple_desc(&bundle.weights, ple_weights)?;
                steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                    rotation: &self.scratch.rotation,
                    state_bf16: bf16_state,
                    key: ple.key,
                    value: ple.value,
                    norm_key: ple.norm_key,
                    norm_query: ple.norm_query,
                    norm_conv: ple.norm_conv,
                    conv: ple.conv,
                    state: &bundle.state.ple_conv,
                    streams: &self.scratch.streams,
                    rows_tensor: &self.scratch.ple_rows,
                    query: &self.scratch.ple_query,
                    key_scratch: &self.scratch.ple_key,
                    value_scratch: &self.scratch.ple_value,
                    gated: &self.scratch.ple_gated,
                    normed: &self.scratch.ple_normed,
                    output: &self.scratch.ple_output,
                    rows: n,
                    branches: dims.hc_count,
                    hidden: config.hidden_size,
                    kernel_size: dims.ple_conv_kernel_dim,
                    dilation: config.ple_conv_dilation(),
                    epsilon: EPSILON,
                }));
            }

            let description = layer_desc(&bundle.weights, layer, &self.moe[layer_index], &config)?;
            let attn_read = &description.attn_hyper.read;
            if steps.len() >= QWEN4_STEP_INLINE_CAPACITY {
                return Err(invalid(
                    "Qwen4 generic layer program inline capacity exhausted",
                ));
            }
            steps.push(Step::HyperRead(HyperReadOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: attn_read.norm,
                input_mix_down: attn_read.input_mix_down,
                input_mix_up: attn_read.input_mix_up,
                normalized: &self.scratch.hc_normalized,
                low: &self.scratch.hc_low,
                up: &self.scratch.hc_up,
                mixed: &self.scratch.hc_mixed,
                bf16_scratch: &self.scratch.gdn_bf16,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
                low_rank: dims.hc_lowrank,
            }));

            match layer.kind {
                LayerType::LinearAttention => {
                    let state = bundle
                        .state
                        .gdn
                        .get(gdn_slot)
                        .ok_or_else(|| invalid("GDN state slot missing"))?;
                    gdn_slot += 1;
                    let weights = match &description.attention {
                        Qwen4AttentionWeights::Linear(weights) => weights,
                        _ => return Err(invalid("GDN descriptor kind mismatch")),
                    };
                    steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                        rotation: &self.scratch.rotation,
                        qkv: weights.qkv,
                        conv: weights.conv,
                        in_proj_a: weights.in_proj_a,
                        in_proj_b: weights.in_proj_b,
                        a_log: weights.a_log,
                        dt_bias: weights.dt_bias,
                        z: weights.z,
                        norm: weights.norm,
                        output: weights.output,
                        recurrent: &state.recurrent,
                        conv_state: &state.conv,
                        projection: &self.scratch.projection,
                        projection2: &self.scratch.projection2,
                        a: &self.scratch.gdn_a,
                        b: &self.scratch.gdn_b,
                        gate: &self.scratch.gdn_gate,
                        beta: &self.scratch.gdn_beta,
                        recurrent_output: &self.scratch.gdn_recurrent_output,
                        bf16_scratch: &self.scratch.gdn_bf16,
                        z_output: &self.scratch.gdn_z,
                        output_scratch: &self.scratch.gdn_output,
                        input: &self.scratch.hc_mixed,
                        output_tensor: &self.scratch.attention_output,
                        rows: n,
                        start_position: bundle.state.position,
                        key_heads: dims.linear_num_key_heads,
                        value_heads: dims.linear_num_value_heads,
                        key_dim: dims.linear_key_head_dim,
                        value_dim: dims.linear_value_head_dim,
                        conv_kernel: dims.linear_conv_kernel_dim,
                        input_width: dims.hidden,
                    }));
                }
                LayerType::FullAttention => {
                    let state = bundle
                        .state
                        .qsa
                        .get(qsa_slot)
                        .ok_or_else(|| invalid("QSA state slot missing"))?;
                    qsa_slot += 1;
                    let weights = match &description.attention {
                        Qwen4AttentionWeights::Full(weights) => weights,
                        _ => return Err(invalid("QSA descriptor kind mismatch")),
                    };
                    steps.push(Step::IndexedAttention(IndexedAttentionOp {
                        rotation: &self.scratch.rotation,
                        indexer_qk: weights.indexer_qk,
                        indexer_q_norm: weights.indexer_q_norm,
                        indexer_k_norm: weights.indexer_k_norm,
                        q: weights.q,
                        k: weights.k,
                        v: weights.v,
                        q_norm: weights.q_norm,
                        k_norm: weights.k_norm,
                        output: weights.output,
                        state: IndexedAttentionState {
                            full_keys: &state.full_keys,
                            full_values: &state.full_values,
                            raw_index_keys: &state.raw_index_keys,
                            pooled_keys: &state.pooled_keys,
                            selected_indices: &state.selected_indices,
                            full_capacity: state.full_capacity,
                            raw_capacity: state.raw_capacity,
                            pooled_capacity: state.pooled_capacity,
                            selected_capacity: state.selected_capacity,
                            position_capacity: state.position_capacity,
                            full_len: state.full_len,
                            raw_len: state.raw_len,
                            pooled_len: state.pooled_len,
                            selected_len: state.selected_len,
                            position: state.position,
                        },
                        input: &self.scratch.hc_mixed,
                        index_scratch: &self.scratch.qsa_index,
                        qgate_scratch: &self.scratch.qsa_qgate,
                        k_scratch: &self.scratch.qsa_k,
                        v_scratch: &self.scratch.qsa_v,
                        qsa_output: &self.scratch.qsa_output,
                        selected_scratch: &self.scratch.qsa_selected,
                        attention_output: &self.scratch.attention_output,
                        bf16_scratch: &self.scratch.gdn_bf16,
                        rows: n,
                        index_heads: dims.indexer_n_heads,
                        index_kv_heads: dims.indexer_kv_heads,
                        index_dim: dims.indexer_head_dim,
                        budget: dims.indexer_budget,
                        compress: dims.indexer_compress_ratio,
                        heads: dims.num_attention_heads,
                        kv_heads: dims.num_key_value_heads,
                        head_dim: dims.head_dim,
                        input_width: dims.hidden,
                    }));
                }
            }

            let attn_write = &description.attn_hyper.write;
            steps.push(Step::HyperWrite(HyperWriteOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: attn_write.norm,
                block_inject: attn_write.block_inject,
                normalized: &self.scratch.hc_normalized,
                mixed: &self.scratch.attention_output,
                gates: &self.scratch.hc_gates,
                output: &self.scratch.streams,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
            }));

            let mlp_read = &description.mlp_hyper.read;
            steps.push(Step::HyperRead(HyperReadOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: mlp_read.norm,
                input_mix_down: mlp_read.input_mix_down,
                input_mix_up: mlp_read.input_mix_up,
                normalized: &self.scratch.hc_normalized,
                low: &self.scratch.hc_low,
                up: &self.scratch.hc_up,
                mixed: &self.scratch.hc_mixed,
                bf16_scratch: &self.scratch.gdn_bf16,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
                low_rank: dims.hc_lowrank,
            }));

            let sealed =
                crate::program::seal_moe_for_layer(&ctx, dims, &description, &scratch_desc, n)
                    .map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}"))
                    })?;
            steps.push(Step::Clear(ClearOp {
                tensor: &self.scratch.moe_output,
                elements: n * config.hidden_size,
            }));
            steps.push(Step::Moe(sealed));

            let mlp_write = &description.mlp_hyper.write;
            steps.push(Step::HyperWrite(HyperWriteOp {
                rotation: &self.scratch.rotation,
                state_bf16: bf16_state,
                input: &self.scratch.streams,
                norm_weight: mlp_write.norm,
                block_inject: mlp_write.block_inject,
                normalized: &self.scratch.hc_normalized,
                mixed: &self.scratch.moe_output,
                gates: &self.scratch.hc_gates,
                output: &self.scratch.streams,
                rows: n,
                branches: dims.hc_count,
                hidden: config.hidden_size,
            }));
        }
        validate_steps(gpu, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight Qwen4 typed program: {error:?}"))
        })?;
        // A retained replay does not execute host code inside its steps. Derive
        // the same QSA commit for either route before any forward-side effects.
        let mut qsa_commits: SmallVec<
            [(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY],
        > = SmallVec::new();
        for step in &steps {
            if let Step::IndexedAttention(op) = step {
                if qsa_commits.len() >= QWEN4_QSA_INLINE_CAPACITY {
                    return Err(invalid("Qwen4 QSA commit inline capacity exhausted"));
                }
                let (full_len, raw_len, pooled_len, selected_len, position) =
                    op.next_lengths().map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!("derive Qwen4 QSA state: {error:?}"))
                    })?;
                qsa_commits.push((
                    qsa_commits.len(),
                    full_len,
                    raw_len,
                    pooled_len,
                    selected_len,
                    position,
                ));
            }
        }
        for (index, token) in tokens.iter().copied().enumerate() {
            let bytes = &mut self.host_token_bytes[index * 4..index * 4 + 4];
            bytes.copy_from_slice(&(token as i32).to_ne_bytes());
        }
        gpu.memcpy_htod_auto(
            &self.scratch.token_ids.buf,
            &self.host_token_bytes[..n * std::mem::size_of::<i32>()],
        )?;
        let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
        let embedding_rot = view(&self.scratch.embedding_rot, 0, n * config.hidden_size);
        let embeddings = view(&self.scratch.embeddings, 0, n * config.hidden_size);
        let ids = view(&self.scratch.token_ids, 0, n * std::mem::size_of::<i32>());
        dispatch_embedding(
            gpu,
            embedding,
            &embedding_rot,
            &embeddings,
            &ids,
            n,
            config.hidden_size,
        )?;

        let mut ple = RowFetch::begin(&bundle.ple_rows)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let next_history = bundle.state.ple_history;
        let next_position = bundle.state.position;
        // Boundary samples, hoisted so the post-body finalize outside this closure
        // can report them: they are taken where the window opens, which is what
        // makes the census an exact bracket of one recorded body.
        let mut diagnostic_capture = false;
        let mut launched_before = 0u64;
        let mut effects_before = (0u64, 0u64, 0u64, 0u64);
        let attempt = (|| -> Result<
            SmallVec<[(usize, usize, usize, usize, usize, usize); QWEN4_QSA_INLINE_CAPACITY]>,
            Qwen4GpuForwardError,
        > {
            ple.prefetch(next_history.row_ids(&bundle.ple_metadata, tokens))
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
            let ple_rows = view(&self.scratch.ple_rows, 0, n * config.hidden_size);
            let staged = view(
                &self.scratch.ple_staged,
                0,
                n * PLE_HEAD_COUNT * PLE_ROW_WIDTH,
            );

            // Expand the embedding batch into the row-major stream basis with
            // one broadcast kernel.  The shared HC program consumes
            // [row, branch, hidden], so a launch-per-row/branch memcpy loop is
            // both unnecessary and visible in short AR prefill profiles.
            let wide = dims.wide();
            if bf16_state {
                gpu.hc_streams_init_from_embed_batched_bf16(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            } else {
                gpu.hc_streams_init_from_embed_batched(
                    &embeddings,
                    &self.scratch.streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    n as i32,
                )?;
            }

            // PLE rows: wait for the host fetch, stage, upload and widen them.
            // A prefill defers this until layer 0 is enqueued (the rows are first
            // read at the PLE layer), so the host fetch overlaps GPU work; a
            // single-token forward keeps it ahead of the retained-body boundary.
            let host_ple_bytes = &mut self.host_ple_bytes;
            let mut stage_ple = |gpu: &mut Gpu| -> Result<(), Qwen4GpuForwardError> {
                if ple.lease().is_some() {
                    return Ok(());
                }
                let wait_started = qwen4_profile_start();
                let lease_result = ple.wait();
                qwen4_profile_record(Qwen4ProfilePhase::PleWait, wait_started);
                let lease =
                    lease_result.map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                let upload_len = lease
                    .as_bytes()
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
                    .len();
                let stage_started = qwen4_profile_start();
                let stage_result = lease.stage_into(&mut host_ple_bytes[..upload_len]);
                qwen4_profile_record(Qwen4ProfilePhase::PleStage, stage_started);
                stage_result.map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                let upload_started = qwen4_profile_start();
                let upload_result = gpu.memcpy_htod_auto(&staged.buf, &host_ple_bytes[..upload_len]);
                qwen4_profile_record(Qwen4ProfilePhase::PleUpload, upload_started);
                upload_result?;
                lease
                    .validate_after_upload()
                    .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                let apply_started = qwen4_profile_start();
                gpu.grouped_gather_convert_bf16(
                    &staged,
                    &ple_rows,
                    n,
                    PLE_HEAD_COUNT,
                    PLE_ROW_WIDTH,
                )?;
                qwen4_profile_record(Qwen4ProfilePhase::PleApply, apply_started);
                Ok(())
            };
            let ple_split = steps
                .iter()
                .position(|step| matches!(step, Step::GroupedDepthwise(_)))
                .filter(|_| n > 1);
            if ple_split.is_none() {
                stage_ple(gpu)?;
            }

            // ── Retained-body boundary ────────────────────────────────────────
            // Everything above materialises this forward's inputs — token ids,
            // embeddings, HC stream init, and the PLE staging/upload/gather — and
            // stays outside a retained tape, so it runs on both the HIP path and a
            // replay. Everything below is the body the tape covers, and this
            // boundary owns the whole per-forward retained-body lifecycle (arm,
            // route, close, prepare) so any driver — the generation loop, a
            // benchmark carrier — leaves the controller in a defined state.
            //
            // Eligible only for a plain single-token continuation: the wide-hidden
            // capture is the speculative-verify shape and must never enter the tape.
            let eligible = n == 1 && wide_hidden_capture.is_none();
            gpu.replay.set_forward_eligible(eligible);
            // A manual shadow controller states the executor directly: one prepared
            // tape is compared across the exact-kernarg HIP oracle, the retained
            // transport, and ordinary HIP. The shadow arms never record, so they do
            // not run the production lifecycle below.
            let shadow_route = if eligible {
                gpu.replay.shadow_body_route()
            } else {
                None
            };
            if shadow_route.is_none() {
                match retained_body_action(
                    eligible,
                    gpu.replay.state(),
                    specialized_sealed_moe_retained_admission(),
                ) {
                    RetainedBodyAction::Ineligible => {}
                    RetainedBodyAction::Refuse { reason } => {
                        // The route cannot be retained: poison before the body so this
                        // and every later forward runs on HIP.
                        gpu.replay.poison(reason);
                        eprintln!("[redline] qwen4 retained body unavailable: {reason}");
                    }
                    RetainedBodyAction::Arm { diagnostic } => {
                        diagnostic_capture = diagnostic;
                        launched_before = hip_bridge::launch_counters::launch_kernel::count();
                        effects_before = (
                            hip_bridge::launch_counters::memcpy_htod::count(),
                            hip_bridge::launch_counters::memcpy_dtod::count(),
                            hip_bridge::launch_counters::memcpy_dtoh::count(),
                            hip_bridge::launch_counters::memset::count(),
                        );
                        gpu.replay
                            .begin_auto_capture_if_armed()
                            .map_err(|reason| invalid(reason))?;
                    }
                    RetainedBodyAction::Route => {}
                }
            }
            let pm4_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_pm4_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_pm4(),
            };
            let aql_route = match shadow_route {
                Some(ShadowBodyRoute::Plan) => gpu.replay.prepared_aql_plan_ready(),
                Some(_) => false,
                None => gpu.replay.should_route_aql(),
            };
            if ple_split.is_some() && (pm4_route || aql_route || shadow_route.is_some()) {
                return Err(invalid("Qwen4 retained body with deferred PLE staging"));
            }
            let routed = if pm4_route {
                // SAFETY: the boundary above staged every host input the tape's
                // recorded launches read, and every pointer in the tape is owned by
                // this bundle for the plan's lifetime.
                match unsafe { gpu.replay.replay_pm4(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 PM4 replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else if shadow_route == Some(ShadowBodyRoute::HipOracle) {
                // The exact-kernarg oracle: re-execute the recorded HIP launch
                // sequence. This is the arm that proves the tape names the dispatches
                // ordinary HIP issues, not merely that some retained transport runs.
                let launches = gpu.replay.recorded_launches().len();
                match gpu.replay_recorded_hip_prefix_at(launches, next_position) {
                    Ok(_) => true,
                    Err(reason) => {
                        return Err(invalid(format!(
                            "Qwen4 recorded-HIP oracle failed: {reason}"
                        )));
                    }
                }
            } else if aql_route {
                // SAFETY: same contract as the PM4 route above.
                match unsafe { gpu.replay.replay_linear_aql(next_position) } {
                    Ok(_) => true,
                    Err(reason) => {
                        gpu.replay.poison(format!("Qwen4 AQL replay failed: {reason}"));
                        return Err(invalid(format!("Qwen4 retained replay failed: {reason}")));
                    }
                }
            } else {
                false
            };

            if !routed {
                let execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
                    execute_validated_steps(gpu, &ctx, steps).map_err(|error| {
                        Qwen4GpuForwardError::Dispatch(format!(
                            "execute Qwen4 layer program: {error:?}"
                        ))
                    })
                };
                match ple_split {
                    Some(split) => {
                        execute(gpu, &steps[..split])?;
                        stage_ple(gpu)?;
                        execute(gpu, &steps[split..])?;
                    }
                    None => execute(gpu, &steps)?,
                }
            }

            if let Some(capture) = wide_hidden_capture.as_ref() {
                let wide_elements = n * wide;
                let source = view(&self.scratch.streams, 0, wide_elements);
                let destination = f32_view(capture, 0, wide_elements);
                if bf16_state {
                    rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                        gpu,
                        &source,
                        &destination,
                        wide_elements,
                    )?;
                } else {
                    gpu.copy_d2d(&source, &destination, source.byte_size())?;
                }
            }

            if let Some(requested_rows) = requested_rows {
                let final_hyper = final_hyper
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 final hyper disappeared before execution"))?;
                let lm_head = lm_head
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 LM head disappeared before execution"))?;
                execute_final_hyper(
                    gpu,
                    dims,
                    final_hyper,
                    &self.scratch.streams,
                    &scratch_desc,
                    n,
                    bf16_state,
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 final hyper: {error:?}"))
                })?;
                let final_hidden = view(&self.scratch.hc_mixed, 0, n * config.hidden_size);
                // The head shares the trunk's basis scratch: each projection
                // consumes its rotated rows before the next one writes.
                let head_rotation =
                    view(&self.scratch.rotation, 0, n * config.hidden_size);
                execute_lm_head(
                    gpu,
                    lm_head,
                    &final_hidden,
                    logits,
                    n,
                    requested_rows,
                    Some(&head_rotation),
                )
                .map_err(
                    |error| {
                        Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 LM head: {error:?}"))
                    },
                )?;

                if let Some(top1) = top1 {
                    argmax_f32(
                        gpu,
                        &ArgmaxF32 { logits,
                        indices: top1,
                        rows: requested_rows,
                        vocab: config.vocab_size, },
                    )?;
                }
            }

            drop(steps);
            Ok(qsa_commits)
        })();
        match attempt {
            Ok(qsa_commits) => {
                ple.complete();
                for (slot, full_len, raw_len, pooled_len, selected_len, position) in qsa_commits {
                    let state = bundle
                        .state
                        .qsa_mut(slot)
                        .ok_or_else(|| invalid("QSA state slot disappeared"))?;
                    state.full_len = full_len;
                    state.raw_len = raw_len;
                    state.pooled_len = pooled_len;
                    state.selected_len = selected_len;
                    state.position = position;
                }
                let mut next_history = next_history;
                for token in tokens.iter().copied() {
                    next_history.push(token);
                }
                bundle.state.ple_history = next_history;
                bundle.state.position = next_position
                    .checked_add(n)
                    .ok_or_else(|| invalid("Qwen4 forward position overflows at commit"))?;
                // Close the retained window this forward opened, report the census
                // and install (or refuse) the prepared plan. This is the only place
                // that knows both ends of one recorded body, so the window can never
                // outlive the forward that opened it.
                if gpu.replay.should_auto_finalize_capture() {
                    finish_qwen4_capture(gpu, diagnostic_capture, launched_before, effects_before);
                }
                Ok(())
            }
            Err(error) => {
                if gpu.replay.is_recording() {
                    // A capture window that failed inside the body must not keep
                    // failing every later forward.
                    gpu.replay
                        .poison("qwen4 capture failed; later forwards fall back to HIP");
                    eprintln!("[redline] qwen4 capture failed; later forwards fall back to HIP");
                }
                match ple.abort() {
                    Ok(_) => Err(error),
                    Err(cleanup) => Err(Qwen4GpuForwardError::Ple(format!(
                        "forward attempt failed: {error}; PLE cleanup failed: {cleanup}"
                    ))),
                }
            }
        }
    }
}

/// Close a recorded Qwen4 body: census, then install the prepared plan, or refuse
/// it for a measurement-only capture.
///
/// Prepare failure after a successful HIP warmup poisons the route (sticky
/// fallback) rather than failing the request that already produced output.
fn finish_qwen4_capture(
    gpu: &mut Gpu,
    diagnostic_capture: bool,
    launched_before: u64,
    effects_before: (u64, u64, u64, u64),
) {
    let launched = hip_bridge::launch_counters::launch_kernel::count();
    let effects = (
        hip_bridge::launch_counters::memcpy_htod::count().saturating_sub(effects_before.0),
        hip_bridge::launch_counters::memcpy_dtod::count().saturating_sub(effects_before.1),
        hip_bridge::launch_counters::memcpy_dtoh::count().saturating_sub(effects_before.2),
        hip_bridge::launch_counters::memset::count().saturating_sub(effects_before.3),
    );
    let summary = gpu.replay.capture_summary();
    let recorded = summary.launch_count;
    let computed = launched.saturating_sub(launched_before) as usize;
    eprintln!(
        "[redline] qwen4 capture census: computed={computed} recorded={recorded} \
         unique_kernels={} sequence_hash={:016x}",
        summary.unique_kernel_count, summary.sequence_hash
    );
    eprintln!(
        "[redline] qwen4 capture census: effects inside window — htod={} dtod={} dtoh={} memset={}",
        effects.0, effects.1, effects.2, effects.3
    );
    if computed != recorded {
        eprintln!(
            "[redline] qwen4 capture census: MISMATCH — {computed} launched, {recorded} recorded \
             (#397-style truncated tape)"
        );
    }
    if effects.1 > 0 || effects.2 > 0 || effects.3 > 0 {
        eprintln!(
            "[redline] qwen4 capture census: effect-incomplete — {} device copy(ies), {} readback(s) \
             and {} memset(s) inside the window are state the tape cannot replay",
            effects.1, effects.2, effects.3
        );
    }
    if diagnostic_capture {
        let reason =
            "diagnostic specialized-MoE capture only; no plan installed (the model runs on \
                      HIP)";
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4: {reason}");
        return;
    }
    if let Err(reason) = gpu.hip.device_synchronize() {
        let reason = format!("qwen4 capture sync failed: {reason:?}");
        gpu.replay.poison(reason.clone());
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    if let Err(reason) = gpu.replay.finish_capture() {
        gpu.replay.poison(reason);
        eprintln!("[redline] qwen4 capture could not be closed or prepared: {reason}");
        return;
    }
    let launches = gpu.replay.recorded_launches().len();
    let prepared = if gpu.replay.uses_pm4_transport() {
        gpu.replay
            .prepare_pm4_prefix(gpu.device_id as usize, launches)
            .map(|_| ())
    } else {
        gpu.replay
            .prepare_linear_aql(gpu.device_id as usize)
            .map(|_| ())
    };
    match prepared {
        Ok(()) => eprintln!(
            "[redline] qwen4 retained body prepared: transport={} dispatches={launches}",
            gpu.replay.transport_name()
        ),
        Err(reason) => {
            let reason = format!("Redline prepare after warmup failed: {reason}");
            gpu.replay.poison(reason.clone());
            eprintln!("[redline] falling back to HIP: {reason}");
        }
    }
}
