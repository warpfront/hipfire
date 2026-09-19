// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight structs (dense / MoE layers), EP shard provenance and seals,
//! `Qwen35Weights`, and the persistent DeltaNet state (`DeltaNetState`).

use super::config::LayerType;
use super::config::Qwen35Config;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::screen_weight_tensor;
use hipfire_runtime::tp_shard::ExpertAssign;
use hipfire_runtime::MmqScreenable;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

// ─── Weight structs ─────────────────────────────────────────────────────

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,   // input_layernorm [dim]
    pub wqkv: WeightTensor,     // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,       // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,  // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,   // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,       // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,     // dt_bias [n_heads]
    pub conv_weight: GpuTensor, // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor, // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,       // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,    // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,   // mlp.gate_proj
    pub w_up: WeightTensor,     // mlp.up_proj
    pub w_down: WeightTensor,   // mlp.down_proj
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,  // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,  // k_proj
    pub wv: WeightTensor,  // v_proj
    pub wo: WeightTensor,  // o_proj
    pub q_norm: GpuTensor, // q_norm [head_dim]
    pub k_norm: GpuTensor, // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,    // [hidden, moe_intermediate]
}

struct ResidentExpertWeights<'a>(&'a [ExpertWeights]);

impl hipfire_dispatch::families::moe::RoutedExpertWeights for ResidentExpertWeights<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(
        &self,
        expert_idx: usize,
    ) -> Option<(
        hipfire_dispatch::families::gemv::WeightRef<'_>,
        hipfire_dispatch::families::gemv::WeightRef<'_>,
    )> {
        self.0
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

pub(crate) fn bind_live_expert_cache(
    cache: &mut hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    table: &hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    experts: &[ExpertWeights],
    gate_up_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    down_awq_ptrs: Option<&GpuTensor>,
    dtype_tags: Option<&GpuTensor>,
) -> HipResult<()> {
    cache
        .bind_live(
            table,
            &ResidentExpertWeights(experts),
            gate_up_ptrs,
            down_ptrs,
            down_awq_ptrs,
            dtype_tags,
        )
        .map_err(HipError::from)
}

/// Owning storage for a layer's packed uniform-MQ4 routed experts.
///
/// `experts` still carries one [`WeightTensor`] view per routed expert so the
/// CPU fallback and every existing indexed dispatch keep their exact metadata
/// and pointer-table ABI. Those views are non-owning subranges of these two
/// buffers; only this owner pair may be returned to the GPU pool.
pub(crate) struct PackedExpertOwners {
    pub(crate) gate_up: GpuTensor,
    pub(crate) down: GpuTensor,
}

/// SP2: build the per-expert (gate_up, down) quant-tier tables that
/// [`hipfire_dispatch::families::moe::MoeDtypes`] uses to detect an
/// intra-layer mixed-tier layer.
///
/// A table is `Some(Box<[DType]>)` only when the layer genuinely spans >1
/// distinct tier; a uniform layer — or paged mode where `experts` is empty —
/// yields `None`, which `MoeResolution::resolve` collapses to the unchanged
/// uniform fast path. The boxed tables are built once while the owner loads;
/// dispatch calls only borrow them.
/// `MoeFfnWeights::per_expert_tier_tables` borrows these load-time arrays.
/// Uniform and empty layers return `None` for both projections, preserving
/// the old uniform path.
pub(crate) fn cached_expert_tier_tables(
    global: Option<&[(DType, DType)]>,
    experts: &[ExpertWeights],
) -> (Option<Box<[DType]>>, Option<Box<[DType]>>) {
    let (gate_up, down): (Vec<DType>, Vec<DType>) = if let Some(global) = global {
        (
            global.iter().map(|(gate_up, _)| *gate_up).collect(),
            global.iter().map(|(_, down)| *down).collect(),
        )
    } else {
        (
            experts.iter().map(|e| e.gate_up.gpu_dtype).collect(),
            experts.iter().map(|e| e.down.gpu_dtype).collect(),
        )
    };
    (
        cached_mixed_tier_table(gate_up),
        cached_mixed_tier_table(down),
    )
}

fn cached_mixed_tier_table(tiers: Vec<DType>) -> Option<Box<[DType]>> {
    match tiers.first() {
        None => None,
        Some(&first) if tiers.iter().all(|&dtype| dtype == first) => None,
        Some(_) => Some(tiers.into_boxed_slice()),
    }
}

/// Fallible per-expert tag mapping for the pinned graded MQ4R family.
///
/// Tags 0..6 retain exact V1 pair meaning. Tags 7..18 are the frozen MQV2
/// mixed-layout identities from the kernel contract — V1/V2 never collapse.
/// Every other ordered pair, every GL dtype in either position, and every
/// unknown dtype is `Err`. Single source of truth consumed by both
/// projections via one stored tag.
pub fn mixed_expert_tag(gate_dtype: DType, down_dtype: DType) -> HipResult<u8> {
    // GL in either position is always rejected – the tag-branched decoder has
    // no GL branch and would silently mis-decode as MQ4.
    if matches!(gate_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
        || matches!(down_dtype, DType::MQ2G256GL | DType::MQ3G256GL)
    {
        return Err(HipError::new(
            0,
            &format!("graded EP: GL dtype not supported (gate={gate_dtype:?} down={down_dtype:?})"),
        ));
    }
    match (gate_dtype, down_dtype) {
        // Tags 0..6 — V1 pair identities (unchanged).
        (DType::MQ4G256, DType::MQ6G256) => Ok(0),
        (DType::MQ4G256, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ4G256, DType::MQ4G256) => Ok(2),
        (DType::MQ4G256, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MQ4G256, DType::MFP4G32E8) => Ok(4),
        (DType::MQ4G256, DType::MFP3G32E8) => Ok(5),
        (DType::MQ4G256, DType::MFP2G32E8) => Ok(6),
        // Matching non-MQ4 V1 pairs reuse the same tag numbers.
        (DType::MQ6G256, DType::MQ6G256) => Ok(0),
        (DType::MQ2G256Lloyd, DType::MQ2G256Lloyd) => Ok(1),
        (DType::MQ3G256Lloyd, DType::MQ3G256Lloyd) => Ok(3),
        (DType::MFP4G32E8, DType::MFP4G32E8) => Ok(4),
        (DType::MFP3G32E8, DType::MFP3G32E8) => Ok(5),
        (DType::MFP2G32E8, DType::MFP2G32E8) => Ok(6),
        // Tags 7..18 — frozen MQV2 mixed identities (never collapse to 0..6).
        (DType::MQ4G256V2, DType::MQ4G256V2) => Ok(7),
        (DType::MQ6G256V2, DType::MQ6G256V2) => Ok(8),
        (DType::MQ4G256V2, DType::MQ6G256) => Ok(9),
        (DType::MQ4G256V2, DType::MQ2G256Lloyd) => Ok(10),
        (DType::MQ4G256V2, DType::MQ4G256) => Ok(11),
        (DType::MQ4G256, DType::MQ4G256V2) => Ok(12),
        (DType::MQ4G256V2, DType::MQ3G256Lloyd) => Ok(13),
        (DType::MQ4G256V2, DType::MFP4G32E8) => Ok(14),
        (DType::MQ4G256V2, DType::MFP3G32E8) => Ok(15),
        (DType::MQ4G256V2, DType::MFP2G32E8) => Ok(16),
        (DType::MQ4G256V2, DType::MQ6G256V2) => Ok(17),
        (DType::MQ4G256, DType::MQ6G256V2) => Ok(18),
        _ => Err(HipError::new(
            0,
            &format!("graded EP: unsupported dtype pair gate={gate_dtype:?} down={down_dtype:?}"),
        )),
    }
}

pub(crate) fn dtype_from_quant_type(qt: u8) -> HipResult<DType> {
    match qt {
        13 => Ok(DType::MQ4G256),
        15 => Ok(DType::MQ6G256),
        19 => Ok(DType::MQ2G256Lloyd),
        20 => Ok(DType::MQ3G256Lloyd),
        30 => Ok(DType::MQ4G256Lloyd),
        34 => Ok(DType::MFP4G32E8),
        36 => Ok(DType::MFP3G32E8),
        38 => Ok(DType::MQ2G256GL),
        39 => Ok(DType::MQ3G256GL),
        40 => Ok(DType::TQ2G128),
        41 => Ok(DType::BQ1G128),
        44 => Ok(DType::MQ4G256V2),
        52 => Ok(DType::MQ4G256V2Lloyd),
        45 => Ok(DType::MQ4CG256),
        // Neutral-size Magnum V2 family (qt47-50): preserve qtype distinction
        // through WeightTensor/GpuTensor; do not map to legacy MQ2/3/5/6.
        47 => Ok(DType::MQ6G256V2),
        48 => Ok(DType::MQ5G256V2),
        49 => Ok(DType::MQ3G256V2),
        50 => Ok(DType::MQ2G256V2),
        // qt=6 (HFQ4G256) and qt=37 (MFP2G32E8) are shipped formats and MUST stay
        // mapped here. Dropping an arm from this match is not a compile error — it
        // degrades to "graded EP: unsupported quant_type", so the loss stays
        // invisible until a model of that format fails to load.
        6 => Ok(DType::HFQ4G256),
        37 => Ok(DType::MFP2G32E8),
        3 => Ok(DType::Q8_0),
        1 => Ok(DType::F16),
        2 => Ok(DType::F32),
        other => Err(HipError::new(
            0,
            &format!("graded EP: unsupported quant_type {other}"),
        )),
    }
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,   // [shared_expert_intermediate, hidden]
    pub down: WeightTensor, // [hidden, shared_expert_intermediate]
}

/// Source representation captured at load time for one routed projection.
/// Names/fingerprints come from the opened source, while bytes/stride reflect
/// the encoded allocation that the owner actually published.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct MoeProjectionSource {
    pub(crate) name: String,
    pub(crate) fingerprint: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub(crate) encoded_bytes: usize,
    pub(crate) row_stride: usize,
    pub(crate) alignment: usize,
    pub(crate) quant_tag: String,
    pub(crate) basis: String,
    pub(crate) sidecars: Box<[String]>,
}

#[derive(Clone)]
pub(crate) struct MoeExpertSourceRecord {
    /// Fused HFQ gate||up source, when the carrier stores one projection.
    pub(crate) gate_up: Option<MoeProjectionSource>,
    /// Separate Paro gate/up source records, when the carrier stores them
    /// independently even though the runtime uploads a fused gate||up view.
    pub(crate) gate: Option<MoeProjectionSource>,
    pub(crate) up: Option<MoeProjectionSource>,
    pub(crate) down: MoeProjectionSource,
}

fn source_metadata(
    source: &MoeProjectionSource,
) -> hipfire_runtime::sealed_moe::ExpertSourceMetadata {
    let mut metadata = hipfire_runtime::sealed_moe::ExpertSourceMetadata::new(
        source.name.clone(),
        source.fingerprint.clone(),
        source.shape.clone(),
        source.dtype,
        source.encoded_bytes,
        source.row_stride,
        source.alignment,
        source.quant_tag.clone(),
        source.basis.clone(),
    );
    metadata.sidecar_source_names = source.sidecars.to_vec();
    metadata
}

fn source_bytes(source: &MoeProjectionSource, fused_half: bool) -> HipResult<usize> {
    if !fused_half {
        return Ok(source.encoded_bytes);
    }
    let rows = *source
        .shape
        .first()
        .ok_or_else(|| HipError::new(0, "fused gate/up source has no rows"))?;
    if rows == 0 || rows % 2 != 0 {
        return Err(HipError::new(
            0,
            &format!(
                "fused gate/up source '{}' has odd row count {rows}",
                source.name
            ),
        ));
    }
    let half = (rows / 2)
        .checked_mul(source.row_stride)
        .ok_or_else(|| HipError::new(0, "fused gate/up projection bytes overflow"))?;
    if half.checked_mul(2) != Some(source.encoded_bytes) {
        return Err(HipError::new(
            0,
            &format!(
                "fused gate/up source '{}' bytes {} do not equal two row-stride halves {}",
                source.name,
                source.encoded_bytes,
                half.saturating_mul(2)
            ),
        ));
    }
    Ok(half)
}

/// Planning target for [`build_expert_binding`]: replicated Single owners or
/// sealed expert-parallel owners on an explicit mesh. The EP variant carries
/// the mesh, the full physical device list, the loading rank, and the planner
/// assignment — ownership always comes back out of the sealed planner, never
/// from a hand-built expert map.
pub(crate) enum ExpertBindingTarget<'a> {
    Single {
        physical_device: i32,
    },
    ExpertParallel {
        mesh: &'a DeviceMesh,
        physical_devices: &'a [i32],
        local_rank: usize,
        assignment: ExpertAssign,
    },
}

/// Construct the runtime-owned sealed plan and dispatch metadata pair for a
/// loaded routed expert set. All source names, shapes, bytes, tags, bases,
/// and sidecars are captured before any expert buffer is uploaded. Single
/// keeps replicated policies and `indexed-single`; EP emits `ExpertSharded`
/// policies, `ExpertParallelism::ExpertParallel`, and the canonical
/// indexed-slot execution, then runs the ordinary planner plus the C1
/// rank-aware adapter.
pub(crate) fn build_expert_binding(
    records: Box<[MoeExpertSourceRecord]>,
    router: MoeProjectionSource,
    sidecars: Vec<MoeProjectionSource>,
    layer_idx: usize,
    n_layers: usize,
    target: ExpertBindingTarget<'_>,
) -> HipResult<(
    hipfire_runtime::sealed_moe::ExpertExecutionPlan,
    hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
)> {
    use hipfire_dispatch::pipeline::sealed_moe::ROOT_ROUTED_EP_EXECUTION;
    use hipfire_runtime::sealed_moe::{
        adapt_expert_execution_plan, plan_expert_execution, plan_single_expert_execution,
        ExpertSourceMetadata,
    };
    use hipfire_runtime::weight_manifest::{
        plan_manifest, ExpertGroupSpec, ExpertParallelism, ExpertProjectionResources,
        ExpertResourceRequirements, ExpertSourceLayout, ShardPolicy, WeightEntry,
    };
    if records.is_empty() {
        return Err(HipError::new(0, "cannot bind an empty expert source set"));
    }
    let n_experts = records.len();
    let mut source_by_name = std::collections::BTreeMap::<String, ExpertSourceMetadata>::new();
    let mut add_source = |source: &MoeProjectionSource| -> HipResult<()> {
        let metadata = source_metadata(source);
        if let Some(previous) = source_by_name.get(&metadata.name) {
            if previous != &metadata {
                return Err(HipError::new(
                    0,
                    &format!("expert source '{}' metadata is inconsistent", metadata.name),
                ));
            }
        } else {
            source_by_name.insert(metadata.name.clone(), metadata);
        }
        Ok(())
    };
    add_source(&router)?;
    for sidecar in &sidecars {
        add_source(sidecar)?;
    }
    for record in records.iter() {
        if let Some(gate_up) = &record.gate_up {
            add_source(gate_up)?;
        } else {
            add_source(record.gate.as_ref().ok_or_else(|| {
                HipError::new(0, "expert source set has no gate or fused gate/up source")
            })?)?;
            add_source(record.up.as_ref().ok_or_else(|| {
                HipError::new(0, "expert source set has no up or fused gate/up source")
            })?)?;
        }
        add_source(&record.down)?;
    }
    let first = records
        .first()
        .ok_or_else(|| HipError::new(0, "cannot bind an empty expert source set"))?;
    let first_gate = first
        .gate_up
        .as_ref()
        .or(first.gate.as_ref())
        .ok_or_else(|| HipError::new(0, "expert source set has no gate source"))?;
    let first_up = first
        .gate_up
        .as_ref()
        .or(first.up.as_ref())
        .ok_or_else(|| HipError::new(0, "expert source set has no up source"))?;
    let fused = first.gate_up.is_some();
    let mut gate_names = Vec::with_capacity(n_experts);
    let mut down_names = Vec::with_capacity(n_experts);
    let mut sidecar_names = std::collections::BTreeSet::new();
    let mut resources = Vec::with_capacity(n_experts);
    let mut up_names = Vec::with_capacity(n_experts);
    for record in records.iter() {
        let (gate, up) = match (&record.gate_up, &record.gate, &record.up) {
            (Some(gate_up), None, None) => (gate_up, gate_up),
            (None, Some(gate), Some(up)) => (gate, up),
            _ => {
                return Err(HipError::new(
                    0,
                    "expert source must be fused or separate gate/up",
                ))
            }
        };
        if gate.shape != up.shape {
            return Err(HipError::new(
                0,
                &format!("expert gate/up shapes differ for '{}'", gate.name),
            ));
        }
        let gate_bytes = source_bytes(gate, fused)?;
        let up_bytes = source_bytes(up, fused)?;
        let alignment = gate.alignment;
        if alignment == 0
            || !alignment.is_power_of_two()
            || up.alignment != alignment
            || record.down.alignment != alignment
        {
            return Err(HipError::new(
                0,
                "expert projection alignments are inconsistent",
            ));
        }
        let down_bytes = source_bytes(&record.down, false)?;
        if gate_bytes % alignment != 0 || up_bytes % alignment != 0 || down_bytes % alignment != 0 {
            return Err(HipError::new(
                0,
                "expert projection bytes violate alignment",
            ));
        }
        gate_names.push(gate.name.clone());
        up_names.push(up.name.clone());
        down_names.push(record.down.name.clone());
        for sidecar in gate
            .sidecars
            .iter()
            .chain(up.sidecars.iter())
            .chain(record.down.sidecars.iter())
        {
            sidecar_names.insert(sidecar.clone());
        }
        resources.push(ExpertProjectionResources {
            gate_bytes,
            up_bytes,
            down_bytes,
            alignment,
        });
    }
    let hidden = first_gate
        .shape
        .get(1)
        .copied()
        .ok_or_else(|| HipError::new(0, "expert gate source has no columns"))?;
    let intermediate = if fused {
        first_gate
            .shape
            .first()
            .copied()
            .ok_or_else(|| HipError::new(0, "expert gate source has no rows"))?
            / 2
    } else {
        first_gate
            .shape
            .first()
            .copied()
            .ok_or_else(|| HipError::new(0, "expert gate source has no rows"))?
    };
    let manifest_dtype = first_gate.dtype;
    // Expert projections replicate on Single and shard on EP; the router and
    // every sidecar replicate on both targets.
    let expert_policy = match &target {
        ExpertBindingTarget::Single { .. } => ShardPolicy::Replicate,
        ExpertBindingTarget::ExpertParallel { assignment, .. } => ShardPolicy::ExpertSharded {
            n_experts,
            assign: *assignment,
        },
    };
    let mut manifest = Vec::with_capacity(4 + source_by_name.len());
    manifest.push(WeightEntry::layer(
        router.name.clone(),
        layer_idx,
        router.shape.clone(),
        router.dtype,
        ShardPolicy::Replicate,
    ));
    for name in &gate_names {
        manifest.push(WeightEntry::layer(
            name.clone(),
            layer_idx,
            vec![n_experts, first_gate.shape[0], hidden],
            manifest_dtype,
            expert_policy.clone(),
        ));
    }
    if !fused {
        for name in &up_names {
            manifest.push(WeightEntry::layer(
                name.clone(),
                layer_idx,
                vec![n_experts, first_up.shape[0], hidden],
                manifest_dtype,
                expert_policy.clone(),
            ));
        }
    }
    let down_shape = records[0].down.shape.clone();
    manifest.extend(down_names.iter().map(|name| {
        WeightEntry::layer(
            name.clone(),
            layer_idx,
            vec![n_experts, down_shape[0], down_shape[1]],
            records[0].down.dtype,
            expert_policy.clone(),
        )
    }));
    for name in &sidecar_names {
        let metadata = source_by_name
            .get(name)
            .ok_or_else(|| HipError::new(0, "expert sidecar metadata disappeared"))?;
        manifest.push(WeightEntry::layer(
            name.clone(),
            layer_idx,
            metadata.logical_shape.clone(),
            metadata.dtype,
            ShardPolicy::Replicate,
        ));
    }
    let (parallelism, spec_assignment, execution) = match &target {
        ExpertBindingTarget::Single { .. } => (
            ExpertParallelism::Single,
            ExpertAssign::Stride,
            "indexed-single".to_string(),
        ),
        ExpertBindingTarget::ExpertParallel { assignment, .. } => (
            ExpertParallelism::ExpertParallel,
            *assignment,
            ROOT_ROUTED_EP_EXECUTION.to_string(),
        ),
    };
    let spec = ExpertGroupSpec {
        group: format!("qwen35.moe.layer.{layer_idx}"),
        layer: Some(layer_idx),
        n_experts,
        parallelism: parallelism,
        assignment: spec_assignment,
        source_layout: if fused {
            ExpertSourceLayout::PerExpertFused {
                gate_up: gate_names,
                down: down_names,
                sidecars: sidecar_names.iter().cloned().collect(),
            }
        } else {
            ExpertSourceLayout::PerExpertSeparate {
                gate: gate_names,
                up: up_names,
                down: down_names,
                sidecars: sidecar_names.iter().cloned().collect(),
            }
        },
        resources: ExpertResourceRequirements::new(resources),
        router: router.name,
        execution: execution,
    };
    let sources: Vec<ExpertSourceMetadata> = source_by_name.into_values().collect();
    let (plan, table, cache) = match &target {
        ExpertBindingTarget::Single { physical_device } => {
            plan_single_expert_execution(&manifest, &spec, &sources, n_layers, *physical_device)
                .map_err(|error| HipError::new(0, &error))?
        }
        ExpertBindingTarget::ExpertParallel {
            mesh,
            physical_devices,
            local_rank,
            ..
        } => {
            // The ordinary manifest + expert planners: ownership, local slots,
            // and the collective schedule come from the sealed plan, never
            // from a hand-built expert map.
            let manifest_plan = plan_manifest(&manifest, &[], mesh, n_layers)
                .map_err(|error| HipError::new(0, &error))?;
            let mut plans = plan_expert_execution(
                &manifest,
                &manifest_plan,
                std::slice::from_ref(&spec),
                &sources,
                mesh,
                physical_devices,
            )
            .map_err(|error| HipError::new(0, &error))?;
            if plans.len() != 1 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "qwen35: EP expert planner returned {} plans, expected exactly one",
                        plans.len()
                    ),
                ));
            }
            let plan = plans.pop().expect("exactly one EP expert plan");
            let (table, cache) = adapt_expert_execution_plan(&plan, *local_rank)
                .map_err(|error| HipError::new(0, &error))?;
            (plan, table, cache)
        }
    };
    Ok((plan, table, cache))
}

#[cfg(test)]
pub(crate) fn test_expert_binding() -> HipResult<(
    hipfire_runtime::sealed_moe::ExpertExecutionPlan,
    hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
)> {
    let source = |name: &str, shape: Vec<usize>, encoded_bytes: usize| MoeProjectionSource {
        name: name.to_string(),
        fingerprint: "test-owner".to_string(),
        shape,
        dtype: DType::F32,
        encoded_bytes,
        row_stride: 4,
        alignment: 1,
        quant_tag: "test-f32".to_string(),
        basis: "None".to_string(),
        sidecars: Box::new([]),
    };
    let router = source("test.router", vec![1, 1], 4);
    let gate_up = source("test.expert.gate_up", vec![2, 1], 8);
    let down = source("test.expert.down", vec![1, 1], 4);
    build_expert_binding(
        Box::new([MoeExpertSourceRecord {
            gate_up: Some(gate_up),
            gate: None,
            up: None,
            down,
        }]),
        router,
        Vec::new(),
        0,
        1,
        ExpertBindingTarget::Single { physical_device: 0 },
    )
}
#[cfg(test)]
mod sealed_ep_owner_tests {
    use super::*;
    use hipfire_runtime::device_mesh::DimKind;
    use hipfire_runtime::tp_shard::ShardConfig;

    fn test_source(name: &str, shape: Vec<usize>, encoded_bytes: usize) -> MoeProjectionSource {
        MoeProjectionSource {
            name: name.to_string(),
            fingerprint: "test-owner".to_string(),
            shape,
            dtype: DType::F32,
            encoded_bytes,
            row_stride: 4,
            alignment: 1,
            quant_tag: "test-f32".to_string(),
            basis: "None".to_string(),
            sidecars: Box::new([]),
        }
    }

    fn test_records(n_experts: usize) -> Box<[MoeExpertSourceRecord]> {
        (0..n_experts)
            .map(|expert| MoeExpertSourceRecord {
                gate_up: Some(test_source(
                    &format!("test.expert.{expert}.gate_up"),
                    vec![2, 1],
                    8,
                )),
                gate: None,
                up: None,
                down: test_source(&format!("test.expert.{expert}.down"), vec![1, 1], 4),
            })
            .collect()
    }
    fn ep_mesh(n_ranks: usize) -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Ep, n_ranks)]).expect("test EP mesh")
    }
    /// Build one layer's sealed EP binding for `local_rank` and assert the
    /// frozen ownership invariants: the rank's local expert sequence equals
    /// `rank_ownership.global_expert_ids`, every global entry matches
    /// `global_to_local`, and the cache names the sealed physical device.
    fn exercise_ep_binding(
        n_experts: usize,
        n_ranks: usize,
        assignment: ExpertAssign,
        expected_owned: &[Vec<usize>],
    ) {
        let mesh = ep_mesh(n_ranks);
        let physical_devices: Vec<i32> = (0..n_ranks as i32).collect();
        // The EP planner requires the router source to lead with n_experts
        // rows (production builds `[n_exp, dim]`); the Single unit path
        // tolerates the degenerate `[1, 1]` carrier.
        let router = test_source("test.router", vec![n_experts, 1], n_experts * 4);
        for (rank, expected) in expected_owned.iter().enumerate() {
            let (plan, table, cache) = build_expert_binding(
                test_records(n_experts),
                router.clone(),
                Vec::new(),
                0,
                1,
                ExpertBindingTarget::ExpertParallel {
                    mesh: &mesh,
                    physical_devices: &physical_devices,
                    local_rank: rank,
                    assignment,
                },
            )
            .unwrap_or_else(|e| panic!("EP binding rank {rank}: {e:?}"));
            assert_eq!(
                plan.rank_ownership()[rank].global_expert_ids,
                *expected,
                "rank {rank} local expert sequence"
            );
            assert_eq!(plan.rank_ownership()[rank].logical_rank, rank);
            assert_eq!(
                plan.rank_ownership()[rank].physical_device,
                physical_devices[rank]
            );
            for (slot, &global) in expected.iter().enumerate() {
                assert_eq!(
                    plan.global_to_local()[rank][global],
                    Some(slot),
                    "rank {rank} slot {slot} global {global}"
                );
            }
            // Every global has exactly one owner across all ranks.
            for global in 0..n_experts {
                let owners = (0..n_ranks)
                    .filter(|&r| plan.global_to_local()[r][global].is_some())
                    .collect::<Vec<_>>();
                assert_eq!(owners, vec![plan.owner_rank(global).expect("owner")]);
            }
            assert_eq!(cache.local_rank(), rank);
            assert_eq!(cache.physical_device(), physical_devices[rank]);
            assert_eq!(cache.local_expert_ids(), expected.as_slice());
            let contract = table
                .execution_contract()
                .expect("sealed EP table carries a contract");
            assert!(contract.is_root_routed_ep(), "root-routed EP execution");
            assert_eq!(
                plan_expert_to_rank(&plan),
                (0..n_experts)
                    .map(|global| plan.owner_rank(global).expect("owner") as u8)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn ep_stride_ownership_is_plan_derived() {
        exercise_ep_binding(4, 2, ExpertAssign::Stride, &[vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn ep_contiguous_ownership_is_plan_derived() {
        exercise_ep_binding(4, 2, ExpertAssign::Contiguous, &[vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn ep4_stride_local_slots_are_compact() {
        exercise_ep_binding(
            8,
            4,
            ExpertAssign::Stride,
            &[vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        );
    }

    #[test]
    fn ep_assignment_inference_rejects_arbitrary_maps() {
        let stride = ShardConfig::new(2, true, 4, ExpertAssign::Stride).expect("stride");
        assert_eq!(
            infer_ep_assignment(&stride, 4).expect("stride"),
            ExpertAssign::Stride
        );
        let contiguous =
            ShardConfig::new(2, true, 4, ExpertAssign::Contiguous).expect("contiguous");
        assert_eq!(
            infer_ep_assignment(&contiguous, 4).expect("contiguous"),
            ExpertAssign::Contiguous
        );
        let arbitrary = ShardConfig {
            tp_size: 2,
            tp_kv_replicate: true,
            expert_to_rank: vec![0, 0, 0, 1],
        };
        assert!(
            infer_ep_assignment(&arbitrary, 4).is_err(),
            "arbitrary expert maps are rejected"
        );
        let short = ShardConfig {
            tp_size: 2,
            tp_kv_replicate: true,
            expert_to_rank: vec![0, 1, 0],
        };
        assert!(infer_ep_assignment(&short, 4).is_err(), "short maps fail");
        // Balanced-contiguous over 2 experts on 4 ranks strands ranks 2-3.
        let stranded = ShardConfig {
            tp_size: 4,
            tp_kv_replicate: true,
            expert_to_rank: vec![0, 1],
        };
        assert!(
            infer_ep_assignment(&stranded, 2).is_err(),
            "zero-owner ranks are rejected"
        );
    }
}

// ─── Sealed EP owner helpers ────────────────────────────────────────────────
//
// Shared by the streaming EP loader (`load::load_weights_ep_rank`) and the
// two-phase post-load shard (`forward::shard_moe_experts`). Every helper
// derives ownership, geometry, and tags from a sealed
// [`hipfire_runtime::sealed_moe::ExpertExecutionPlan`]; nothing here
// reconstructs `e % N` or consults a caller map after the plan exists.

/// Infer the planner assignment a `ShardConfig` map encodes, rejecting
/// everything else before any GPU allocation. Only exact Stride
/// (`expert_to_rank[e] == e % tp`) and exact Contiguous (balanced ranges)
/// maps are admitted; arbitrary expert maps, short/long maps, out-of-range
/// ranks, and ranks left with zero owners all fail here.
pub(crate) fn infer_ep_assignment(
    shard: &hipfire_runtime::tp_shard::ShardConfig,
    n_experts: usize,
) -> HipResult<ExpertAssign> {
    use hipfire_runtime::tp_shard::ShardConfig;
    let tp = shard.tp_size;
    if tp == 0 {
        return Err(HipError::new(0, "qwen35: EP shard has no ranks"));
    }
    if n_experts == 0 {
        return Err(HipError::new(0, "qwen35: EP sharding needs routed experts"));
    }
    if shard.expert_to_rank.len() != n_experts {
        return Err(HipError::new(
            0,
            &format!(
                "qwen35: EP shard map covers {} experts, expected {n_experts}",
                shard.expert_to_rank.len()
            ),
        ));
    }
    let stride = (0..n_experts).all(|expert| shard.expert_to_rank[expert] as usize == expert % tp);
    let contiguous = (0..tp).all(|rank| {
        ShardConfig::balanced_range(rank, tp, n_experts)
            .all(|expert| shard.expert_to_rank[expert] as usize == rank)
    });
    let assignment = if stride {
        ExpertAssign::Stride
    } else if contiguous {
        ExpertAssign::Contiguous
    } else {
        return Err(HipError::new(
            0,
            "qwen35: EP shard map is neither Stride nor Contiguous; arbitrary expert maps are rejected",
        ));
    };
    for rank in 0..tp {
        if !shard
            .expert_to_rank
            .iter()
            .any(|&owner| owner as usize == rank)
        {
            return Err(HipError::new(
                0,
                &format!("qwen35: EP shard leaves rank {rank} with zero owners"),
            ));
        }
    }
    Ok(assignment)
}

/// Global expert → owning rank map derived from a sealed plan, in global-id
/// order. This is the ONLY sanctioned source of `Qwen35EpShardInfo` ownership.
pub(crate) fn plan_expert_to_rank(
    plan: &hipfire_runtime::sealed_moe::ExpertExecutionPlan,
) -> Vec<u8> {
    plan.experts()
        .iter()
        .map(|record| record.owner_rank as u8)
        .collect()
}

/// Global (gate_up, down) dtype pairs from a sealed plan, in global-id order.
pub(crate) fn global_dtype_pairs_from_plan(
    plan: &hipfire_runtime::sealed_moe::ExpertExecutionPlan,
) -> Vec<(DType, DType)> {
    plan.experts()
        .iter()
        .map(|record| (record.gate.dtype, record.down.dtype))
        .collect()
}

/// Global (gate_up, down) dtype pairs from loader source records, in
/// global-id order.
pub(crate) fn global_dtype_pairs_from_records(
    records: &[MoeExpertSourceRecord],
) -> Vec<(DType, DType)> {
    records
        .iter()
        .map(|record| {
            let gate_dtype = record
                .gate_up
                .as_ref()
                .or(record.gate.as_ref())
                .map(|source| source.dtype)
                .unwrap_or(DType::F32);
            (gate_dtype, record.down.dtype)
        })
        .collect()
}

/// Validate every global dtype pair against the frozen graded-tag registry
/// (rejecting GL and unknown pairs) and return the per-global tag bytes.
/// CPU-only: runs before any dummy, table, or expert allocation.
pub(crate) fn validate_ep_global_tags(pairs: &[(DType, DType)]) -> HipResult<Vec<u8>> {
    pairs
        .iter()
        .enumerate()
        .map(|(expert, &(gate_dtype, down_dtype))| {
            mixed_expert_tag(gate_dtype, down_dtype).map_err(|error| {
                HipError::new(
                    0,
                    &format!("qwen35: EP global expert {expert} {}", error.message),
                )
            })
        })
        .collect()
}

/// One distinct non-owned storage layout needing an owned zero dummy pair.
/// Geometry comes from sealed source metadata (never from a resident tensor
/// that a later failure might free out from under the bind).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct EpDummySpec {
    pub gate_dtype: DType,
    pub gate_m: usize,
    pub gate_k: usize,
    pub gate_bytes: usize,
    pub gate_stride: usize,
    pub down_dtype: DType,
    pub down_m: usize,
    pub down_k: usize,
    pub down_bytes: usize,
    pub down_stride: usize,
}

/// Allocate one owned zero buffer per dummy spec and build the borrowed
/// [`ExpertWeights`] views the compact bind matches non-owned globals
/// against. Returns `(owners, views)`; the views alias the owners via
/// `shallow_clone` and must be freed metadata-only. A mid-loop allocation
/// error frees every owner acquired so far.
pub(crate) fn alloc_ep_dummies(
    gpu: &mut Gpu,
    specs: &[EpDummySpec],
) -> HipResult<(Vec<GpuTensor>, Vec<ExpertWeights>)> {
    fn zero_buf(gpu: &Gpu, bytes: usize, role: &str) -> HipResult<GpuTensor> {
        if bytes == 0 {
            return Err(HipError::new(
                0,
                &format!("qwen35: EP dummy {role} has zero bytes"),
            ));
        }
        gpu.upload_raw(&vec![0u8; bytes], &[bytes])
    }
    let mut owners = Vec::with_capacity(specs.len() * 2);
    let mut views = Vec::with_capacity(specs.len());
    for spec in specs {
        let gate_owner = match zero_buf(gpu, spec.gate_bytes, "gate/up") {
            Ok(tensor) => tensor,
            Err(error) => {
                for owner in owners.drain(..) {
                    let _ = gpu.free_tensor(owner);
                }
                return Err(error);
            }
        };
        let down_owner = match zero_buf(gpu, spec.down_bytes, "down") {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(gate_owner);
                for owner in owners.drain(..) {
                    let _ = gpu.free_tensor(owner);
                }
                return Err(error);
            }
        };
        let gate_up = WeightTensor {
            buf: gate_owner.shallow_clone(),
            gpu_dtype: spec.gate_dtype,
            m: spec.gate_m,
            k: spec.gate_k,
            row_stride: spec.gate_stride,
            paro: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        let down = WeightTensor {
            buf: down_owner.shallow_clone(),
            gpu_dtype: spec.down_dtype,
            m: spec.down_m,
            k: spec.down_k,
            row_stride: spec.down_stride,
            paro: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        owners.push(gate_owner);
        owners.push(down_owner);
        views.push(ExpertWeights { gate_up, down });
    }
    Ok((owners, views))
}

/// Upload the global `[n_exp]` pointer tables (plus the optional global
/// dtype-tag table) for one sealed EP owner. A mid-sequence failure frees
/// every table acquired so far.
pub(crate) fn upload_ep_tables(
    gpu: &mut Gpu,
    gate_up_entries: &[u64],
    down_entries: &[u64],
    dtype_tags: Option<&[u8]>,
) -> HipResult<(GpuTensor, GpuTensor, Option<GpuTensor>)> {
    let table_len = gate_up_entries
        .len()
        .checked_mul(2)
        .ok_or_else(|| HipError::new(0, "qwen35: EP pointer table size overflows"))?;
    if down_entries.len() != gate_up_entries.len() {
        return Err(HipError::new(
            0,
            "qwen35: EP gate/up and down pointer entries disagree",
        ));
    }
    let gate_up_table = gpu.alloc_tensor(&[table_len], DType::F32)?;
    let gate_up_bytes: Vec<u8> = gate_up_entries
        .iter()
        .flat_map(|ptr| ptr.to_ne_bytes())
        .collect();
    if let Err(error) = gpu.hip.memcpy_htod(&gate_up_table.buf, &gate_up_bytes) {
        let _ = gpu.free_tensor(gate_up_table);
        return Err(error);
    }
    let down_table = match gpu.alloc_tensor(&[table_len], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => {
            let _ = gpu.free_tensor(gate_up_table);
            return Err(error);
        }
    };
    let down_bytes: Vec<u8> = down_entries
        .iter()
        .flat_map(|ptr| ptr.to_ne_bytes())
        .collect();
    if let Err(error) = gpu.hip.memcpy_htod(&down_table.buf, &down_bytes) {
        let _ = gpu.free_tensor(gate_up_table);
        let _ = gpu.free_tensor(down_table);
        return Err(error);
    }
    let tag_table = match dtype_tags {
        None => None,
        Some(tags) => {
            let tensor = match gpu.alloc_tensor(&[tags.len()], DType::Raw) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = gpu.free_tensor(gate_up_table);
                    let _ = gpu.free_tensor(down_table);
                    return Err(error);
                }
            };
            if let Err(error) = gpu.hip.memcpy_htod(&tensor.buf, tags) {
                let _ = gpu.free_tensor(gate_up_table);
                let _ = gpu.free_tensor(down_table);
                let _ = gpu.free_tensor(tensor);
                return Err(error);
            }
            Some(tensor)
        }
    };
    Ok((gate_up_table, down_table, tag_table))
}

/// Run the C1 compact live bind for one sealed EP owner: every owned global
/// entry must point at its plan-mapped local tensor, every non-owned entry
/// at an owned layout-compatible zero dummy, on the sealed physical device.
/// AWQ is never admitted on EP owners (`None`/`None`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_compact_ep_cache(
    cache: &mut hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    table: &hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    local_experts: &[hipfire_dispatch::pipeline::sealed_moe::CompactLiveWeight<'_>],
    gate_up_entries: &[usize],
    down_entries: &[usize],
    gate_up_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    dtype_tags: Option<&GpuTensor>,
    zero_dummies: &[hipfire_dispatch::pipeline::sealed_moe::CompactLiveWeight<'_>],
    device_id: i32,
) -> HipResult<()> {
    cache
        .bind_live_compact(
            table,
            local_experts,
            gate_up_entries,
            down_entries,
            None,
            gate_up_ptrs,
            down_ptrs,
            None,
            dtype_tags,
            zero_dummies,
            device_id,
        )
        .map_err(HipError::from)
}

pub struct MoeFfnWeights {
    pub router: WeightTensor, // [num_experts, hidden]
    /// Routed expert weights. Populated when this layer is fully resident
    /// (`paged_experts == false`); **empty `Vec`** when `paged_experts == true`
    /// (the [`hipfire_runtime::weight_pager::WeightPager`] owns the buffers, and the
    /// indexed kernels read pointers from `expert_*_ptrs` which the pager
    /// patches per-token via `patch_expert_ptr_table`).
    pub experts: Vec<ExpertWeights>, // num_experts (= 256 for A3B); empty in paged mode
    /// Two allocation owners for the uniform MQ4 packed path. `None` preserves
    /// the literal per-expert ownership used by mixed quant, Paro, paged, and
    /// EP-streaming routes.
    pub(crate) packed_expert_owners: Option<PackedExpertOwners>,
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Route A MoE-AWQ: per-expert down `awq_scale` pointer table
    /// (`[num_experts * 2]` f32 = num_experts × u64). `Some` only when the
    /// `.hfq` carries per-expert `down_proj.awq_scale` sidecars (all-or-none).
    /// Holds *non-owning* device pointers into each `experts[i].down.awq_scale`
    /// — freed as a buffer only; the scales are freed via
    /// `ExpertWeights::down.free_all`.
    pub expert_down_awq_ptrs: Option<GpuTensor>,

    /// Per-expert mixed-precision decode: `[num_experts]` u8 (DType::Raw,
    /// 1 B/expert) dtype-tag table. `Some` only when the layer's routed
    /// experts carry MIXED down dtypes (graded MQ6 hot / MQ2-Lloyd cold);
    /// the merged dtype-tag-branched down kernel reads `tags[expert_id]`
    /// per block (0=MQ6, 1=MQ2-Lloyd). `None` ⇒ uniform path, byte-identical.
    /// Owned device buffer (no aliasing) — freed as a buffer in free_moe_ffn.
    pub expert_dtype_tags: Option<GpuTensor>,

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,

    /// ParoQuant only: shared per-layer rotation sidecars for the routed
    /// experts. shisa-ai's PARO checkpoint quantizes all 256 experts with
    /// one rotation tuple per projection-group (gate||up vs down), so we
    /// upload the sidecars ONCE per layer and broadcast a non-owning
    /// `ParoRotation` (built via `DeviceBuffer::from_raw`) into every
    /// `ExpertWeights.gate_up.paro` / `ExpertWeights.down.paro`. The
    /// owning storage lives here so the aliases stay valid for the
    /// lifetime of the layer. `None` for HFQ MoE (per-tensor PARO sidecars
    /// or no PARO at all).
    pub paro_shared: Option<MoeParoSidecars>,

    /// EP global (gate_up_dtype, down_dtype) table — CPU-side immutable
    /// snapshot of the *full-model* expert dtypes (`len == num_experts`).
    /// `Some` only on the EP `load_weights_ep_rank` path; `None` preserves
    /// byte-identical single-GPU behavior. When present, every graded-mix
    /// decision (uniform/mixed flags, representative dtypes, tier tables,
    /// dummy layout sizes, device tag upload) is derived from this global
    /// table, never from the compact local `experts` slice.
    pub(crate) global_expert_dtypes: Option<Box<[(DType, DType)]>>,
    /// Cached mixed-only per-expert dtype tiers.  Uniform and empty layers
    /// keep these as `None`; the forward paths borrow the slices directly.
    pub(crate) mixed_expert_gate_up_tiers: Option<Box<[DType]>>,
    pub(crate) mixed_expert_down_tiers: Option<Box<[DType]>>,

    /// EP streaming dummies: one owned zero buffer per distinct
    /// non-owned storage layout. Non-owned global slots alias into the
    /// matching entry. Owned so `free_moe_ffn` can reclaim them. Each entry
    /// is one projection buffer (gate/up and down layouts are separate
    /// entries); the borrowed [`ExpertWeights`] views in
    /// `ep_dummy_experts` alias these buffers and must never be freed
    /// directly.
    pub(crate) ep_dummy_buffers: Vec<GpuTensor>,
    /// Borrowed [`ExpertWeights`] views into `ep_dummy_buffers`, one pair per
    /// distinct non-owned layout. Built once at commit so the compact bind
    /// and every later sealed call can borrow layout-compatible zero dummies
    /// without allocating. Freed metadata-only; the buffers die with
    /// `ep_dummy_buffers`.
    pub(crate) ep_dummy_experts: Vec<ExpertWeights>,
    /// Displaced experts retired by post-load [`super::forward::shard_moe_experts`]:
    /// formerly owned tensors that no longer belong to this rank. Retained
    /// (never read) until teardown so a committed two-phase shard cannot
    /// turn cleanup into a reported failure, and so no buffer is released
    /// while a pre-swap capture might still reference it. Empty on every
    /// load path.
    pub(crate) retired_expert_weights: Vec<ExpertWeights>,
    /// Immutable runtime execution plan from the source manifest.  The
    /// dispatch table/cache below are adapted exclusively from this plan.
    pub(crate) expert_execution_plan: hipfire_runtime::sealed_moe::ExpertExecutionPlan,
    /// Immutable CPU-only dispatch metadata and its load-time rank-local
    /// binding proof.  The borrowed call view is recreated from these fields
    /// without allocating or rebuilding expert maps.
    pub(crate) expert_table: hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
    pub(crate) expert_binding: hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
}

impl MoeFfnWeights {
    /// Borrow the cached mixed-only tier slices for one dispatch call.
    /// Uniform and empty layers return `None`; this accessor performs no
    /// allocation or per-expert traversal.
    pub(crate) fn per_expert_tier_tables(&self) -> (Option<&[DType]>, Option<&[DType]>) {
        (
            self.mixed_expert_gate_up_tiers.as_deref(),
            self.mixed_expert_down_tiers.as_deref(),
        )
    }

    /// Borrow the load-time validated table/cache pair for one dispatch call.
    /// `BoundMoeExperts::from_cache` performs only identity checks and does not
    /// allocate per token.
    pub(crate) fn bound_experts(
        &self,
    ) -> HipResult<hipfire_dispatch::pipeline::sealed_moe::BoundMoeExperts<'_>> {
        hipfire_dispatch::pipeline::sealed_moe::BoundMoeExperts::from_cache(
            &self.expert_table,
            &self.expert_binding,
        )
        .map_err(HipError::from)
    }
}

/// Owning storage for the per-layer shared ParoQuant rotation sidecars.
/// One tuple per projection-group:
///   - `gate_up_*`: applied to the post-RMSNorm hidden activation (K = hidden_dim).
///     Shared by all 256 experts' gate AND up projections, and by the fused
///     gate_up `WeightTensor`'s `paro` alias.
///   - `down_*`: applied to the post-SiLU intermediate activation (K = mi).
///     Shared by all 256 experts' down projection.
pub struct MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
    pub krot: u32,
    pub group_size: u32,
    /// Exact source names for the two projection-specific sidecar tuples.
    /// Gate/up and down are deliberately not collapsed: the loaded expert
    /// tensors carry distinct rotation identities and must bind to the
    /// matching tuple only.
    pub(crate) gate_up_sidecar_names: Box<[String]>,
    pub(crate) down_sidecar_names: Box<[String]>,
    pub(crate) source_fingerprint: String,
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}
impl LayerWeights {
    /// Return every GPU allocation owned by one layer to `gpu`.
    ///
    /// This is the single layer-level teardown used by both normal unload and
    /// whole-model load rollback. In particular, it preserves the packed,
    /// paged, EP, and Paro ownership branches in `free_moe_ffn`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            LayerWeights::DeltaNet(l) => {
                let _ = gpu.free_tensor(l.attn_norm);
                l.wqkv.free_all(gpu);
                l.wz.free_all(gpu);
                l.w_alpha.free_all(gpu);
                l.w_beta.free_all(gpu);
                let _ = gpu.free_tensor(l.a_log);
                let _ = gpu.free_tensor(l.dt_bias);
                let _ = gpu.free_tensor(l.conv_weight);
                let _ = gpu.free_tensor(l.norm_weight);
                l.wo.free_all(gpu);
                let _ = gpu.free_tensor(l.ffn_norm);
                l.w_gate.free_all(gpu);
                l.w_up.free_all(gpu);
                l.w_down.free_all(gpu);
            }
            LayerWeights::FullAttn(l) => {
                let _ = gpu.free_tensor(l.attn_norm);
                l.wq.free_all(gpu);
                l.wk.free_all(gpu);
                l.wv.free_all(gpu);
                l.wo.free_all(gpu);
                let _ = gpu.free_tensor(l.q_norm);
                let _ = gpu.free_tensor(l.k_norm);
                let _ = gpu.free_tensor(l.ffn_norm);
                l.w_gate.free_all(gpu);
                l.w_up.free_all(gpu);
                l.w_down.free_all(gpu);
            }
            LayerWeights::DeltaNetMoe(l) => {
                let _ = gpu.free_tensor(l.attn_norm);
                l.wqkv.free_all(gpu);
                l.wz.free_all(gpu);
                l.w_alpha.free_all(gpu);
                l.w_beta.free_all(gpu);
                let _ = gpu.free_tensor(l.a_log);
                let _ = gpu.free_tensor(l.dt_bias);
                let _ = gpu.free_tensor(l.conv_weight);
                let _ = gpu.free_tensor(l.norm_weight);
                l.wo.free_all(gpu);
                let _ = gpu.free_tensor(l.ffn_norm);
                free_moe_ffn(gpu, l.ffn);
            }
            LayerWeights::FullAttnMoe(l) => {
                let _ = gpu.free_tensor(l.attn_norm);
                l.wq.free_all(gpu);
                l.wk.free_all(gpu);
                l.wv.free_all(gpu);
                l.wo.free_all(gpu);
                let _ = gpu.free_tensor(l.q_norm);
                let _ = gpu.free_tensor(l.k_norm);
                let _ = gpu.free_tensor(l.ffn_norm);
                free_moe_ffn(gpu, l.ffn);
            }
        }
    }
}
/// Immutable source identity captured before any EP GPU allocation.
/// Exact equality over canonical path, platform file identity (dev, ino),
/// length, mtime, arch_id, exact metadata_json, ordered tensor manifest
/// (name, quant_type, shape, group_size, data_offset, data_size) with
/// absolute offsets (base offset included), and overlay status.
/// Not a hash – any reordering or header difference is inequality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35HfqSourceIdentity {
    pub canonical_path: std::path::PathBuf,
    pub dev: u64,
    pub ino: u64,
    pub file_len: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
    pub arch_id: u32,
    pub metadata_json: String,
    pub tensor_manifest: Vec<(String, u8, Vec<u32>, u32, usize, usize)>,
    pub has_overlay: bool,
}

impl Qwen35HfqSourceIdentity {
    pub fn capture(hfq: &HfqFile) -> Self {
        let path = hfq.path().to_path_buf();
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let (dev, ino, file_len, mtime_secs, mtime_nanos) = {
            match std::fs::metadata(&path) {
                Ok(md) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            md.dev(),
                            md.ino(),
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                    #[cfg(not(unix))]
                    {
                        let mtime = md
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                        (
                            0u64,
                            0u64,
                            md.len(),
                            mtime.map(|d| d.as_secs() as i64).unwrap_or(0),
                            mtime.map(|d| d.subsec_nanos()).unwrap_or(0),
                        )
                    }
                }
                Err(_) => (0, 0, 0, 0, 0),
            }
        };
        let tensors = hfq.tensors();
        let manifest = tensors
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    t.quant_type,
                    t.shape.clone(),
                    t.group_size,
                    t.data_offset,
                    t.data_size,
                )
            })
            .collect();
        Self {
            canonical_path: canonical,
            dev,
            ino,
            file_len,
            mtime_secs,
            mtime_nanos,
            arch_id: hfq.arch_id,
            metadata_json: hfq.metadata_json.clone(),
            tensor_manifest: manifest,
            has_overlay: hfq.has_overlay(),
        }
    }
}
/// Derive a stable source fingerprint from the complete HFQ identity.  This
/// intentionally hashes the opened artifact identity (path/file identity,
/// metadata, and every tensor's encoded layout), never model-layer or dtype
/// guesses.  The sealed dispatch stores the compact digest while the full
/// [`Qwen35HfqSourceIdentity`] remains available to the loader's provenance
/// path.
pub(crate) fn hfq_source_fingerprint(hfq: &HfqFile) -> String {
    use std::hash::{Hash, Hasher};
    let identity = Qwen35HfqSourceIdentity::capture(hfq);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    identity.canonical_path.hash(&mut hasher);
    identity.dev.hash(&mut hasher);
    identity.ino.hash(&mut hasher);
    identity.file_len.hash(&mut hasher);
    identity.mtime_secs.hash(&mut hasher);
    identity.mtime_nanos.hash(&mut hasher);
    identity.arch_id.hash(&mut hasher);
    identity.metadata_json.hash(&mut hasher);
    identity.has_overlay.hash(&mut hasher);
    for (name, quant_type, shape, group_size, data_offset, data_size) in &identity.tensor_manifest {
        name.hash(&mut hasher);
        quant_type.hash(&mut hasher);
        shape.hash(&mut hasher);
        group_size.hash(&mut hasher);
        data_offset.hash(&mut hasher);
        data_size.hash(&mut hasher);
    }
    format!("hfq:{:016x}", hasher.finish())
}

/// Derive the same kind of artifact-bound fingerprint for a generic
/// [`ModelSource`] (including the safetensors-backed Paro loader).  Tensor
/// names, shape, dtype, encoded byte ranges, and quantization metadata are
/// included so a projection cannot be rebound to a different source merely
/// because its logical dimensions happen to match.
pub(crate) fn model_source_fingerprint(
    source: &dyn hipfire_runtime::model_source::ModelSource,
) -> String {
    use std::hash::{Hash, Hasher};
    let mut names = source.tensor_names();
    names.sort_unstable();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.path().hash(&mut hasher);
    source.metadata_json().hash(&mut hasher);
    source.arch_id().hash(&mut hasher);
    if let Some(qc) = source.quant_config() {
        qc.method.hash(&mut hasher);
        qc.bits.hash(&mut hasher);
        qc.group_size.hash(&mut hasher);
        qc.krot.hash(&mut hasher);
        qc.dynamic_excludes.hash(&mut hasher);
    }
    for name in names {
        name.hash(&mut hasher);
        if let Some(info) = source.tensor_info(name) {
            info.name.hash(&mut hasher);
            info.dtype.hash(&mut hasher);
            info.shape.hash(&mut hasher);
            info.quant_type.hash(&mut hasher);
            info.data_offset.hash(&mut hasher);
            info.data_size.hash(&mut hasher);
        }
    }
    format!("source:{:016x}", hasher.finish())
}

/// Frozen config fingerprint for EP seal. Contains every Qwen35Config primitive.
/// Equality is exact (f32 via to_bits). EP admission still rejects paged/REAP but
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpConfigFingerprint {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps_bits: u32,
    pub eos_token: u32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta_bits: u32,
    pub partial_rotary_factor_bits: u32,
    pub is_vl_text: bool,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub hidden_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub has_shared_expert: bool,
    pub norm_topk_prob: bool,
    pub layer_types: Vec<LayerType>,
    pub paged_experts: bool,
    pub vram_budget_bytes: u64,
    pub has_reap_keep: bool,
}

impl Qwen35EpConfigFingerprint {
    pub fn capture(config: &Qwen35Config) -> Self {
        Self {
            dim: config.dim,
            n_layers: config.n_layers,
            vocab_size: config.vocab_size,
            norm_eps_bits: config.norm_eps.to_bits(),
            eos_token: config.eos_token,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            rope_theta_bits: config.rope_theta.to_bits(),
            partial_rotary_factor_bits: config.partial_rotary_factor.to_bits(),
            is_vl_text: config.is_vl_text,
            mrope_interleaved: config.mrope_interleaved,
            mrope_section: config.mrope_section,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            conv_kernel_dim: config.conv_kernel_dim,
            hidden_dim: config.hidden_dim,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            has_shared_expert: config.has_shared_expert,
            norm_topk_prob: config.norm_topk_prob,
            layer_types: config.layer_types.clone(),
            paged_experts: config.paged_experts,
            vram_budget_bytes: config.vram_budget_bytes,
            has_reap_keep: config.reap_keep.is_some(),
        }
    }
}

/// Device-pointer-free descriptor for a GpuTensor. Excludes DeviceBuffer pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_len: usize,
}

impl GpuTensorDescriptor {
    pub fn from_tensor(t: &GpuTensor) -> Self {
        Self {
            shape: t.shape.clone(),
            dtype: t.dtype,
            byte_len: t.buf.size(),
        }
    }
}

/// Paro sidecar descriptor excluding pointers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParoDescriptor {
    pub krot: u32,
    pub group_size: u32,
    pub is_alias: bool,
    pub pairs: GpuTensorDescriptor,
    pub theta: GpuTensorDescriptor,
    pub channel_scales: GpuTensorDescriptor,
}

/// Weight tensor descriptor excluding device pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTensorDescriptor {
    pub gpu_dtype: DType,
    pub m: usize,
    pub k: usize,
    pub row_stride: usize,
    pub buf: GpuTensorDescriptor,
    pub awq_scale: Option<GpuTensorDescriptor>,
    pub paro: Option<ParoDescriptor>,
}

impl WeightTensorDescriptor {
    pub fn from_weight(w: &WeightTensor) -> Self {
        Self {
            gpu_dtype: w.gpu_dtype,
            m: w.m,
            k: w.k,
            row_stride: w.row_stride,
            buf: GpuTensorDescriptor::from_tensor(&w.buf),
            awq_scale: w.awq_scale.as_ref().map(GpuTensorDescriptor::from_tensor),
            paro: w.paro.as_ref().map(|p| ParoDescriptor {
                krot: p.krot,
                group_size: p.group_size,
                is_alias: p.is_alias,
                pairs: GpuTensorDescriptor::from_tensor(&p.pairs),
                theta: GpuTensorDescriptor::from_tensor(&p.theta),
                channel_scales: GpuTensorDescriptor::from_tensor(&p.channel_scales),
            }),
        }
    }
}

/// Per-expert local descriptor: global id + gate_up/down descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LocalExpertDescriptor {
    pub global_expert_id: usize,
    pub gate_up: WeightTensorDescriptor,
    pub down: WeightTensorDescriptor,
}
/// Complete rank weight/layout seal. Immutable, allocation-free comparison via `matches_*`.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35RankSeal {
    pub token_embd: GpuTensorDescriptor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensorDescriptor,
    pub output: WeightTensorDescriptor,
    pub moe_has_mq6: bool,
    pub has_pager: bool,
    pub lm_head_aliases_embd: bool,
    pub layer_seals: Vec<Qwen35LayerSeal>,
    pub global_expert_dtypes: Vec<Vec<(DType, DType)>>,
    pub local_expert_descriptors: Vec<Vec<Qwen35LocalExpertDescriptor>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qwen35LayerSeal {
    DeltaNet {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    FullAttn {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        w_gate: WeightTensorDescriptor,
        w_up: WeightTensorDescriptor,
        w_down: WeightTensorDescriptor,
    },
    DeltaNetMoe {
        attn_norm: GpuTensorDescriptor,
        wqkv: WeightTensorDescriptor,
        wz: WeightTensorDescriptor,
        w_alpha: WeightTensorDescriptor,
        w_beta: WeightTensorDescriptor,
        a_log: GpuTensorDescriptor,
        dt_bias: GpuTensorDescriptor,
        conv_weight: GpuTensorDescriptor,
        norm_weight: GpuTensorDescriptor,
        wo: WeightTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
    FullAttnMoe {
        attn_norm: GpuTensorDescriptor,
        wq: WeightTensorDescriptor,
        wk: WeightTensorDescriptor,
        wv: WeightTensorDescriptor,
        wo: WeightTensorDescriptor,
        q_norm: GpuTensorDescriptor,
        k_norm: GpuTensorDescriptor,
        ffn_norm: GpuTensorDescriptor,
        moe: Qwen35MoeFfnSeal,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeFfnSeal {
    pub router: WeightTensorDescriptor,
    pub shared_gate: WeightTensorDescriptor,
    pub shared_up: WeightTensorDescriptor,
    pub shared_down: WeightTensorDescriptor,
    pub shared_expert_gate: WeightTensorDescriptor,
    pub expert_gate_up_ptrs: GpuTensorDescriptor,
    pub expert_down_ptrs: GpuTensorDescriptor,
    pub expert_down_awq_ptrs: Option<GpuTensorDescriptor>,
    pub expert_dtype_tags: Option<GpuTensorDescriptor>,
    pub layer_idx: u16,
    pub has_packed_owners: bool,
    pub global_expert_dtypes: Option<Vec<(DType, DType)>>,
    pub num_local_experts: usize,
}

impl Qwen35RankSeal {
    pub fn capture(weights: &Qwen35Weights, expert_to_rank: Option<&[u8]>, rank: usize) -> Self {
        let mut global_expert_dtypes = Vec::with_capacity(weights.layers.len());
        let mut local_expert_descriptors = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let ffn = match layer {
                LayerWeights::DeltaNetMoe(weights) => Some(&weights.ffn),
                LayerWeights::FullAttnMoe(weights) => Some(&weights.ffn),
                _ => None,
            };
            let Some(ffn) = ffn else {
                global_expert_dtypes.push(Vec::new());
                local_expert_descriptors.push(Vec::new());
                continue;
            };
            global_expert_dtypes.push(
                ffn.global_expert_dtypes
                    .as_ref()
                    .map(|dtypes| dtypes.to_vec())
                    .unwrap_or_default(),
            );
            let owned: Vec<usize> = match expert_to_rank {
                Some(map) => map
                    .iter()
                    .enumerate()
                    .filter_map(|(global_id, &owner)| (owner as usize == rank).then_some(global_id))
                    .collect(),
                None => (0..ffn.experts.len()).collect(),
            };
            let mut locals = Vec::with_capacity(ffn.experts.len());
            for (local_pos, expert) in ffn.experts.iter().enumerate() {
                locals.push(Qwen35LocalExpertDescriptor {
                    global_expert_id: owned.get(local_pos).copied().unwrap_or(local_pos),
                    gate_up: WeightTensorDescriptor::from_weight(&expert.gate_up),
                    down: WeightTensorDescriptor::from_weight(&expert.down),
                });
            }
            locals.sort_by_key(|descriptor| descriptor.global_expert_id);
            local_expert_descriptors.push(locals);
        }
        let layer_seals = weights
            .layers
            .iter()
            .map(|l| match l {
                LayerWeights::DeltaNet(w) => Qwen35LayerSeal::DeltaNet {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                    wz: WeightTensorDescriptor::from_weight(&w.wz),
                    w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                    w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                    a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                    dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                    conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                    norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::FullAttn(w) => Qwen35LayerSeal::FullAttn {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wq: WeightTensorDescriptor::from_weight(&w.wq),
                    wk: WeightTensorDescriptor::from_weight(&w.wk),
                    wv: WeightTensorDescriptor::from_weight(&w.wv),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                    k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    w_gate: WeightTensorDescriptor::from_weight(&w.w_gate),
                    w_up: WeightTensorDescriptor::from_weight(&w.w_up),
                    w_down: WeightTensorDescriptor::from_weight(&w.w_down),
                },
                LayerWeights::DeltaNetMoe(w) => Qwen35LayerSeal::DeltaNetMoe {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wqkv: WeightTensorDescriptor::from_weight(&w.wqkv),
                    wz: WeightTensorDescriptor::from_weight(&w.wz),
                    w_alpha: WeightTensorDescriptor::from_weight(&w.w_alpha),
                    w_beta: WeightTensorDescriptor::from_weight(&w.w_beta),
                    a_log: GpuTensorDescriptor::from_tensor(&w.a_log),
                    dt_bias: GpuTensorDescriptor::from_tensor(&w.dt_bias),
                    conv_weight: GpuTensorDescriptor::from_tensor(&w.conv_weight),
                    norm_weight: GpuTensorDescriptor::from_tensor(&w.norm_weight),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    moe: Qwen35MoeFfnSeal {
                        router: WeightTensorDescriptor::from_weight(&w.ffn.router),
                        shared_gate: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.gate),
                        shared_up: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.up),
                        shared_down: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.down),
                        shared_expert_gate: WeightTensorDescriptor::from_weight(
                            &w.ffn.shared_expert_gate,
                        ),
                        expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                            &w.ffn.expert_gate_up_ptrs,
                        ),
                        expert_down_ptrs: GpuTensorDescriptor::from_tensor(&w.ffn.expert_down_ptrs),
                        expert_down_awq_ptrs: w
                            .ffn
                            .expert_down_awq_ptrs
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        expert_dtype_tags: w
                            .ffn
                            .expert_dtype_tags
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        layer_idx: w.ffn.layer_idx,
                        has_packed_owners: w.ffn.packed_expert_owners.is_some(),
                        global_expert_dtypes: w
                            .ffn
                            .global_expert_dtypes
                            .as_ref()
                            .map(|b| b.to_vec()),
                        num_local_experts: w.ffn.experts.len(),
                    },
                },
                LayerWeights::FullAttnMoe(w) => Qwen35LayerSeal::FullAttnMoe {
                    attn_norm: GpuTensorDescriptor::from_tensor(&w.attn_norm),
                    wq: WeightTensorDescriptor::from_weight(&w.wq),
                    wk: WeightTensorDescriptor::from_weight(&w.wk),
                    wv: WeightTensorDescriptor::from_weight(&w.wv),
                    wo: WeightTensorDescriptor::from_weight(&w.wo),
                    q_norm: GpuTensorDescriptor::from_tensor(&w.q_norm),
                    k_norm: GpuTensorDescriptor::from_tensor(&w.k_norm),
                    ffn_norm: GpuTensorDescriptor::from_tensor(&w.ffn_norm),
                    moe: Qwen35MoeFfnSeal {
                        router: WeightTensorDescriptor::from_weight(&w.ffn.router),
                        shared_gate: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.gate),
                        shared_up: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.up),
                        shared_down: WeightTensorDescriptor::from_weight(&w.ffn.shared_expert.down),
                        shared_expert_gate: WeightTensorDescriptor::from_weight(
                            &w.ffn.shared_expert_gate,
                        ),
                        expert_gate_up_ptrs: GpuTensorDescriptor::from_tensor(
                            &w.ffn.expert_gate_up_ptrs,
                        ),
                        expert_down_ptrs: GpuTensorDescriptor::from_tensor(&w.ffn.expert_down_ptrs),
                        expert_down_awq_ptrs: w
                            .ffn
                            .expert_down_awq_ptrs
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        expert_dtype_tags: w
                            .ffn
                            .expert_dtype_tags
                            .as_ref()
                            .map(GpuTensorDescriptor::from_tensor),
                        layer_idx: w.ffn.layer_idx,
                        has_packed_owners: w.ffn.packed_expert_owners.is_some(),
                        global_expert_dtypes: w
                            .ffn
                            .global_expert_dtypes
                            .as_ref()
                            .map(|b| b.to_vec()),
                        num_local_experts: w.ffn.experts.len(),
                    },
                },
            })
            .collect();
        Self {
            token_embd: GpuTensorDescriptor::from_tensor(&weights.token_embd),
            embd_format: weights.embd_format,
            output_norm: GpuTensorDescriptor::from_tensor(&weights.output_norm),
            output: WeightTensorDescriptor::from_weight(&weights.output),
            moe_has_mq6: weights.moe_has_mq6,
            has_pager: weights.pager.is_some(),
            lm_head_aliases_embd: weights.lm_head_aliases_embd,
            layer_seals,
            global_expert_dtypes,
            local_expert_descriptors,
        }
    }
    pub fn matches_config(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35EpShardInfo {
    pub(crate) rank: u8,
    pub(crate) rank_count: u8,
    pub(crate) expert_to_rank: Box<[u8]>,
    pub device_id: i32,
    pub source_identity: std::sync::Arc<Qwen35HfqSourceIdentity>,
    pub config_fingerprint: Qwen35EpConfigFingerprint,
    pub rank_seal: Qwen35RankSeal,
}

impl Qwen35EpShardInfo {
    /// Owning rank for this shard (0 <= rank < rank_count).
    pub fn rank(&self) -> u8 {
        self.rank
    }
    /// Total number of ranks in the EP group (exactly 4 for the MQ4R route).
    pub fn rank_count(&self) -> u8 {
        self.rank_count
    }
    /// Global expert → owning rank map (`len == config.num_experts`, each entry < rank_count).
    pub fn expert_to_rank(&self) -> &[u8] {
        &self.expert_to_rank
    }
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    pub fn source_identity(&self) -> &Qwen35HfqSourceIdentity {
        &self.source_identity
    }
    pub fn config_fingerprint(&self) -> &Qwen35EpConfigFingerprint {
        &self.config_fingerprint
    }
    pub fn rank_seal(&self) -> &Qwen35RankSeal {
        &self.rank_seal
    }
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    /// True when any MoE FFN projection in the loaded model is MQ6. gfx1151's
    /// grouped-i8 MQ4 shortcut is model-level unsafe for these promoted A3B
    /// checkpoints, even in layers whose local routed experts remain MQ4.
    pub moe_has_mq6: bool,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. The forward path uses
    /// interior mutability (`borrow_mut`) at the MoE dispatch site to call
    /// `ensure_resident` / `patch_expert_ptr_table`. `None` means the model
    /// is fully resident — no behavior change vs main.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,

    /// True when the tied lm_head aliases the embedding table buffer
    /// (single-GPU path). When true, `output.buf` is a non-owning view of
    /// `token_embd.buf` and must NOT be freed in `free_gpu`.
    pub lm_head_aliases_embd: bool,

    /// Immutable EP shard provenance. `Some` only when loaded via
    /// `load_weights_ep_rank` on the exact 4×gfx1201 MQ4R route; all
    /// ordinary (single-GPU, TP, paged) loads leave `None`.
    pub(crate) ep_shard: Option<Qwen35EpShardInfo>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        if !self.lm_head_aliases_embd {
            self.output.free_all(gpu);
        }
        for layer in self.layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = self.pager {
            pager_cell.into_inner().free_all(gpu);
        }
    }

    /// Multi-GPU companion to `free_gpu`. Each layer freed on its
    /// band-owning device per `gpus.device_for_layer(i)`; `token_embd`
    /// freed on dev 0; `output_norm + output` on `gpus.output_device`.
    /// Mirror of `load_weights_multi` placement. The `pager` field is
    /// always `None` on the multi path (paged-experts is not wired into
    /// pp>1 yet); a non-None pager would need its own per-band drain
    /// strategy and is rejected at load.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        debug_assert!(
            self.pager.is_none(),
            "free_gpu_multi: pager must be None on pp>1 path"
        );
        let _ = gpus.devices[0].free_tensor(self.token_embd);
        let out_dev = gpus.output_device;
        let _ = gpus.devices[out_dev].free_tensor(self.output_norm);
        self.output.free_all(&mut gpus.devices[out_dev]);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    l.w_gate.free_all(gpu);
                    l.w_up.free_all(gpu);
                    l.w_down.free_all(gpu);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wqkv.free_all(gpu);
                    l.wz.free_all(gpu);
                    l.w_alpha.free_all(gpu);
                    l.w_beta.free_all(gpu);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    l.wq.free_all(gpu);
                    l.wk.free_all(gpu);
                    l.wv.free_all(gpu);
                    l.wo.free_all(gpu);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}
impl Qwen35Weights {
    /// Immutable EP shard provenance, if loaded via `load_weights_ep_rank`.
    /// `None` on every ordinary single-GPU/TP/paged load.
    pub fn ep_shard(&self) -> Option<&Qwen35EpShardInfo> {
        self.ep_shard.as_ref()
    }
}

impl MmqScreenable for Qwen35Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.output, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match layer {
                LayerWeights::DeltaNet(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttn(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.w_gate,
                        &weights.w_up,
                        &weights.w_down,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                // Routed and shared experts live outside ordinary WeightTensor
                // storage in paged/EP modes. Screen the resident attention and
                // dense router weights here; expert screening is separate work.
                LayerWeights::DeltaNetMoe(weights) => {
                    for weight in [
                        &weights.wqkv,
                        &weights.wz,
                        &weights.w_alpha,
                        &weights.w_beta,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
                LayerWeights::FullAttnMoe(weights) => {
                    for weight in [
                        &weights.wq,
                        &weights.wk,
                        &weights.wv,
                        &weights.wo,
                        &weights.ffn.router,
                    ] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                }
            }
        }
        (safe, unsafe_count)
    }
}

/// Free a [`WeightTensor`] through a caller-supplied GPU-tensor cleanup seam.
/// The callback receives every owned sidecar and the weight buffer exactly once.
pub(crate) fn free_weight_with<F>(weight: WeightTensor, free: &mut F)
where
    F: FnMut(GpuTensor),
{
    if let Some(paro) = weight.paro {
        if !paro.is_alias {
            free(paro.pairs);
            free(paro.theta);
            free(paro.channel_scales);
        }
    }
    if let Some(awq) = weight.awq_scale {
        free(awq);
    }
    free(weight.buf);
}

/// Free a [`WeightTensor`]'s owning sidecars without freeing its weight buffer.
/// Used for non-owning views into [`PackedExpertOwners`] and into the EP
/// dummy owner buffers (`ep_dummy_buffers`).
pub(crate) fn free_weight_metadata_with<F>(weight: WeightTensor, free: &mut F)
where
    F: FnMut(GpuTensor),
{
    if let Some(paro) = weight.paro {
        if !paro.is_alias {
            free(paro.pairs);
            free(paro.theta);
            free(paro.channel_scales);
        }
    }
    if let Some(awq) = weight.awq_scale {
        free(awq);
    }
}

/// Free a staged MoE owner through a caller-supplied GPU-tensor cleanup seam.
///
/// The ownership branches here are authoritative for all current routed-expert
/// layouts: ordinary per-expert weights, packed uniform-MQ4 owners, ParoQuant
/// shared sidecars, EP dummy buffers, and paged-mode's empty expert vector.
/// Each callback invocation consumes one actual owning buffer exactly once.
pub(crate) fn free_moe_ffn_with(ffn: MoeFfnWeights, free: &mut impl FnMut(GpuTensor)) {
    free_weight_with(ffn.router, free);
    free_weight_with(ffn.shared_expert_gate, free);
    free_weight_with(ffn.shared_expert.gate, free);
    free_weight_with(ffn.shared_expert.up, free);
    free_weight_with(ffn.shared_expert.down, free);
    free(ffn.expert_gate_up_ptrs);
    free(ffn.expert_down_ptrs);
    // Non-owning pointer table — free the buffer only; the per-expert scales it
    // points into are owned by `experts[i].down.awq_scale` and freed below via
    // `free_weight_with`.
    if let Some(t) = ffn.expert_down_awq_ptrs {
        free(t);
    }
    // Owned device buffer (built from per-expert gpu_dtype). Free it.
    if let Some(t) = ffn.expert_dtype_tags {
        free(t);
    }
    if let Some(owners) = ffn.packed_expert_owners {
        // Packed expert WeightTensors are non-owning views. Free only metadata
        // that remains individually owned, then return each layer blob once.
        // Shard-retired views alias the same blobs, so they are metadata-only
        // here as well.
        for e in ffn.experts.into_iter().chain(ffn.retired_expert_weights) {
            free_weight_metadata_with(e.gate_up, free);
            free_weight_metadata_with(e.down, free);
        }
        free(owners.gate_up);
        free(owners.down);
    } else {
        for e in ffn.experts {
            free_weight_with(e.gate_up, free);
            free_weight_with(e.down, free);
        }
        // Retired owners are only populated by post-load sharding of a
        // literal (non-packed) model, so each holds its own buffers.
        for e in ffn.retired_expert_weights {
            free_weight_with(e.gate_up, free);
            free_weight_with(e.down, free);
        }
    }
    // EP dummy views borrow `ep_dummy_buffers`; release their (empty)
    // metadata only, then return each owning zero buffer once.
    for e in ffn.ep_dummy_experts {
        free_weight_metadata_with(e.gate_up, free);
        free_weight_metadata_with(e.down, free);
    }
    // ParoQuant MoE: free the owning shared sidecars (per-expert `paro` fields
    // alias these and must NOT be freed separately — they're non-owning views).
    if let Some(s) = ffn.paro_shared {
        free(s.gate_up_pairs);
        free(s.gate_up_theta);
        free(s.gate_up_channel_scales);
        free(s.down_pairs);
        free(s.down_theta);
        free(s.down_channel_scales);
    }
    for d in ffn.ep_dummy_buffers {
        free(d);
    }
}

pub(crate) fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    let mut free = |tensor| {
        let _ = gpu.free_tensor(tensor);
    };
    free_moe_ffn_with(ffn, &mut free);
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    /// Non-owning single-lane view into state allocated by
    /// [`Self::new_batched_with_quant`]. Used only to seed prompts through the
    /// existing sequential prefill path. The returned view must not be freed.
    pub(crate) fn q8_lane_view(
        &self,
        config: &Qwen35Config,
        lane: usize,
        batch: usize,
    ) -> HipResult<Self> {
        if self.quant != StateQuant::Q8 || lane >= batch {
            return Err(HipError::new(
                0,
                "DeltaNet q8_lane_view requires Q8 state and a valid lane",
            ));
        }
        let n_heads = config.linear_num_value_heads;
        let hd = config.linear_value_head_dim;
        let s_elems = n_heads * hd * hd;
        let scale_elems = n_heads * hd;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_elems = conv_channels * (config.conv_kernel_dim - 1);

        let byte_view = |t: &GpuTensor, off: usize, bytes: usize, dtype: DType| {
            let ptr = unsafe { (t.buf.as_ptr() as *mut u8).add(off) as *mut std::ffi::c_void };
            GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) },
                shape: vec![bytes / dtype.size()],
                dtype,
            }
        };
        Ok(Self {
            s_matrices: self
                .s_matrices
                .iter()
                .map(|t| byte_view(t, lane * s_elems, s_elems, DType::Raw))
                .collect(),
            s_scales: self
                .s_scales
                .iter()
                .map(|t| byte_view(t, lane * scale_elems * 4, scale_elems * 4, DType::F32))
                .collect(),
            conv_states: self
                .conv_states
                .iter()
                .map(|t| byte_view(t, lane * conv_elems * 4, conv_elems * 4, DType::F32))
                .collect(),
            s_ef_residual: self
                .s_ef_residual
                .iter()
                .map(|t| byte_view(t, lane * s_elems * 2, s_elems * 2, DType::F16))
                .collect(),
            quant: StateQuant::Q8,
        })
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        Self::new_batched_with_quant(gpu, config, quant, 1)
    }

    /// Allocate lane-major recurrent state for independent-sequence decode.
    ///
    /// The ordinary state has an implicit batch of one.  This variant keeps
    /// the same per-layer vectors, but every tensor is laid out as
    /// `[batch, ...single-lane shape...]`.  It is intentionally consumed only
    /// by [`Qwen35DecodeBatchState`]: passing it to sequential prefill would
    /// advance lane 0 and leave the other lanes stale.
    pub fn new_batched_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
        batch: usize,
    ) -> HipResult<Self> {
        assert!(batch > 0, "DeltaNetState batch must be non-zero");
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size_per_lane = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144
        let s_size = batch * s_size_per_lane;

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = batch * conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work). The multi-GPU band split (`new_with_quant_multi` below) uses
        // the same rule and geometry. Residual is f16 per-element.
        let ef_enabled = quant == StateQuant::Q8
            && hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        // GpuTensor has no freeing Drop (free needs &mut Gpu). Mirror
        // alloc_k_v_vmm_filtered: on any mid-loop failure free every tensor
        // already pushed before propagating.
        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        let result = (|| -> HipResult<()> {
            for _ in 0..n_delta_layers {
                match quant {
                    StateQuant::FP32 => {
                        s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                        s_scales.push(gpu.zeros(&[batch * n_heads], DType::F32)?);
                    }
                    StateQuant::Q8 => {
                        // int8 state: s_size bytes (1 byte each), per-row scales
                        let buf = gpu.hip.malloc(s_size)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                    StateQuant::Q4 => {
                        // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                        let buf = gpu.hip.malloc(s_size / 2)?;
                        if let Err(e) = gpu.hip.memset(&buf, 0, s_size / 2) {
                            let _ = gpu.hip.free(buf);
                            return Err(e);
                        }
                        s_matrices.push(GpuTensor {
                            buf,
                            shape: vec![s_size / 2],
                            dtype: DType::F32,
                        });
                        s_scales.push(gpu.zeros(&[batch * n_heads * s_dim], DType::F32)?);
                    }
                }
                if ef_enabled {
                    s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
                }
                conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
            }
            Ok(())
        })();
        if let Err(err) = result {
            for tensor in s_matrices
                .drain(..)
                .chain(s_scales.drain(..))
                .chain(conv_states.drain(..))
                .chain(s_ef_residual.drain(..))
            {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(err);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    ///
    /// Returns `Err` on the first HIP memset/memset_async failure so production
    /// rollback can attest `rolled_back:false`.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        // Sticky-fault poison: a 700/719 here means the context is dead (the
        // gate saw the same 719 repeat across requests on memsets alone), so
        // latch process-wide and let the daemon fail fast instead of burning
        // doomed prefills. All other errors pass through unlatched.
        let result: HipResult<()> = (|| {
            match gpu.active_stream.as_ref() {
                Some(stream) => {
                    for s in &self.s_matrices {
                        gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                    }
                    for s in &self.s_scales {
                        gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                    }
                    for s in &self.conv_states {
                        gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                    }
                    for s in &self.s_ef_residual {
                        gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream)?;
                    }
                }
                None => {
                    for s in &self.s_matrices {
                        gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                    }
                    for s in &self.s_scales {
                        gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                    }
                    for s in &self.conv_states {
                        gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                    }
                    for s in &self.s_ef_residual {
                        gpu.hip.memset(&s.buf, 0, s.buf.size())?;
                    }
                }
            }
            Ok(())
        })();
        hipfire_runtime::reset_core::note_hip_result(result, "qwen35::DeltaNetState::reset")
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        // Same EF sigma-delta rule as the single path (`new_batched_with_quant`):
        // Q8 + `HIPFIRE_DN_STATE_EF` default-ON. Each residual is f16 per state
        // element, allocated on the LA-layer's owning device so `ef_residual()`
        // returns `Some` and the DeltaNet kernel takes the deterministic
        // sigma-delta branch instead of stochastic dither. No further plumbing
        // is needed: `forward_scratch_layers_multi` already runs each layer on
        // `gpus.devices[device_for_layer]` and passes
        // `ef_residual(delta_layer_idx)` alongside the same-device S/scales.
        let ef_enabled = quant == StateQuant::Q8
            && hipfire_config::developer_var("HIPFIRE_DN_STATE_EF")
                .map(|v| v != "0")
                .unwrap_or(true);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut s_ef_residual = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    let buf = g.hip.malloc(s_size)?;
                    if let Err(e) = g.hip.memset(&buf, 0, s_size) {
                        let _ = g.hip.free(buf);
                        return Err(e);
                    }
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    let buf = g.hip.malloc(s_size / 2)?;
                    if let Err(e) = g.hip.memset(&buf, 0, s_size / 2) {
                        let _ = g.hip.free(buf);
                        return Err(e);
                    }
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size / 2],
                        dtype: DType::F32,
                    });
                    s_scales.push(g.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            if ef_enabled {
                s_ef_residual.push(g.zeros(&[s_size], DType::F16)?);
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // Per-device EF residuals (Q8 + HIPFIRE_DN_STATE_EF default-ON);
                // empty only when EF is off (FP32/Q4/HIPFIRE_DN_STATE_EF=0) ⇒
                // ef_residual() returns None ⇒ kernel uses stochastic requant.
                s_ef_residual,
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        // Per-device EF residuals ride along when EF is on (Q8 default-ON).
        for (i, t) in self.s_ef_residual.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    #[test]
    fn cached_expert_tier_tables_preserve_mixed_order() {
        let uniform = [(DType::MQ4G256, DType::MQ4G256); 3];
        let (gate_up, down) = cached_expert_tier_tables(Some(&uniform), &[]);
        assert!(gate_up.is_none());
        assert!(down.is_none());

        let mixed = [
            (DType::MQ4G256, DType::MQ6G256),
            (DType::MQ6G256, DType::MQ4G256),
            (DType::MQ4G256, DType::MQ4G256),
        ];
        let (gate_up, down) = cached_expert_tier_tables(Some(&mixed), &[]);
        assert_eq!(
            gate_up.as_deref(),
            Some(&[DType::MQ4G256, DType::MQ6G256, DType::MQ4G256][..])
        );
        assert_eq!(
            down.as_deref(),
            Some(&[DType::MQ6G256, DType::MQ4G256, DType::MQ4G256][..])
        );
    }

    #[test]
    fn dtype_from_quant_type_neutral_v2_one_to_one() {
        // Each qt maps one-to-one to its V2 DType and exact block bytes;
        // V2 DTypes are distinct from legacy MQ2/3/5/6.
        assert_eq!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256V2);
        assert_eq!(dtype_from_quant_type(48).unwrap(), DType::MQ5G256V2);
        assert_eq!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256V2);
        assert_eq!(dtype_from_quant_type(50).unwrap(), DType::MQ2G256V2);
        // Legacy unchanged.
        assert_eq!(dtype_from_quant_type(15).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(47).unwrap(), DType::MQ6G256);
        assert_ne!(dtype_from_quant_type(49).unwrap(), DType::MQ3G256);
        // Bad qt fails closed.
        assert!(dtype_from_quant_type(99).is_err());
        // qt44/45 still map to their V2/C DTypes.
        assert_eq!(dtype_from_quant_type(44).unwrap(), DType::MQ4G256V2);
        assert_eq!(dtype_from_quant_type(45).unwrap(), DType::MQ4CG256);
    }

    // ── mixed_expert_tag — frozen V1 0..6 + MQV2 7..18 ────────────────────
    // Exact ordered-pair → tag map; V1/V2 never collapse; unknown pairs Err.

    #[test]
    fn mixed_expert_tag_v1_pairs_retain_tags_0_through_6() {
        use DType::*;
        let accepted: &[(DType, DType, u8)] = &[
            (MQ4G256, MQ6G256, 0),
            (MQ4G256, MQ2G256Lloyd, 1),
            (MQ4G256, MQ4G256, 2),
            (MQ4G256, MQ3G256Lloyd, 3),
            (MQ4G256, MFP4G32E8, 4),
            (MQ4G256, MFP3G32E8, 5),
            (MQ4G256, MFP2G32E8, 6),
            (MQ6G256, MQ6G256, 0),
            (MQ2G256Lloyd, MQ2G256Lloyd, 1),
            (MQ3G256Lloyd, MQ3G256Lloyd, 3),
            (MFP4G32E8, MFP4G32E8, 4),
            (MFP3G32E8, MFP3G32E8, 5),
            (MFP2G32E8, MFP2G32E8, 6),
        ];
        for &(g, d, tag) in accepted {
            assert_eq!(
                mixed_expert_tag(g, d).expect("V1 accepted pair"),
                tag,
                "V1 gate={g:?} down={d:?}"
            );
        }
    }

    #[test]
    fn mixed_expert_tag_mqv2_pairs_emit_exact_tags_7_through_18() {
        use DType::*;
        // Frozen contract: exact tags 7..18; no V2→V1 collapse.
        let accepted: &[(DType, DType, u8)] = &[
            (MQ4G256V2, MQ4G256V2, 7),
            (MQ6G256V2, MQ6G256V2, 8),
            (MQ4G256V2, MQ6G256, 9),
            (MQ4G256V2, MQ2G256Lloyd, 10),
            (MQ4G256V2, MQ4G256, 11),
            (MQ4G256, MQ4G256V2, 12),
            (MQ4G256V2, MQ3G256Lloyd, 13),
            (MQ4G256V2, MFP4G32E8, 14),
            (MQ4G256V2, MFP3G32E8, 15),
            (MQ4G256V2, MFP2G32E8, 16),
            (MQ4G256V2, MQ6G256V2, 17),
            (MQ4G256, MQ6G256V2, 18),
        ];
        for &(g, d, tag) in accepted {
            assert_eq!(
                mixed_expert_tag(g, d).expect("MQV2 accepted pair"),
                tag,
                "MQV2 gate={g:?} down={d:?}"
            );
        }
    }

    #[test]
    fn mixed_expert_tag_v2_never_collapses_to_v1_tags() {
        use DType::*;
        // Former collapses that incorrectly reused tags 0..6 must now be distinct.
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ6G256).unwrap(),
            mixed_expert_tag(MQ4G256, MQ6G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ4G256V2).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256V2, MQ4G256).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_ne!(
            mixed_expert_tag(MQ4G256, MQ4G256V2).unwrap(),
            mixed_expert_tag(MQ4G256, MQ4G256).unwrap()
        );
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ6G256).unwrap(), 9);
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ4G256V2).unwrap(), 7);
        assert_eq!(mixed_expert_tag(MQ4G256V2, MQ4G256).unwrap(), 11);
        assert_eq!(mixed_expert_tag(MQ4G256, MQ4G256V2).unwrap(), 12);
    }

    #[test]
    fn mixed_expert_tag_unknown_pairs_err() {
        use DType::*;
        // Cross-direction and unsupported V2 combos stay Err.
        let rejected: &[(DType, DType)] = &[
            (MQ6G256, MQ4G256),
            (MQ6G256V2, MQ4G256V2),
            (MQ6G256V2, MQ4G256),
            (MQ6G256, MQ6G256V2),
            (MQ6G256V2, MQ6G256),
            (MQ4G256V2, MQ5G256V2),
            (MQ2G256V2, MQ2G256V2),
            (MQ3G256V2, MQ3G256V2),
            (MQ5G256V2, MQ5G256V2),
            (MQ4G256Lloyd, MQ4G256Lloyd),
            (Q8_0, Q8_0),
            (F16, F16),
            (ParoQ4G128, ParoQ4G128),
            (MQ2G256Lloyd, MQ4G256),
            (MQ3G256Lloyd, MQ4G256V2),
            (MFP4G32E8, MQ4G256),
        ];
        for &(g, d) in rejected {
            let err = mixed_expert_tag(g, d).expect_err(&format!("gate={g:?} down={d:?}"));
            assert!(
                err.message.contains("unsupported dtype pair"),
                "gate={g:?} down={d:?}: {}",
                err.message
            );
        }
    }

    #[test]
    fn mixed_expert_tag_gl_either_side_err() {
        use DType::*;
        for gate in [MQ2G256GL, MQ3G256GL, MQ4G256, MQ4G256V2] {
            for down in [MQ2G256GL, MQ3G256GL, MQ4G256, MQ4G256V2] {
                let gl =
                    matches!(gate, MQ2G256GL | MQ3G256GL) || matches!(down, MQ2G256GL | MQ3G256GL);
                if !gl {
                    continue;
                }
                let err = mixed_expert_tag(gate, down).expect_err("GL must reject");
                assert!(
                    err.message.contains("GL dtype not supported"),
                    "gate={gate:?} down={down:?}: {}",
                    err.message
                );
            }
        }
    }
}
