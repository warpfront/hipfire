// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-owned, CPU-only sealing for routed expert execution.
//!
//! This module deliberately stops at immutable metadata.  It does not allocate
//! GPU storage, create communicators, or infer placement from a filename.  A
//! loader supplies source metadata, [`plan_expert_execution`] validates the
//! complete list, and dispatch borrows the resulting plan for each call.

use crate::device_mesh::{CollectiveHint, DeviceMesh, DimKind, MeshEpoch};
use crate::tp_shard::ExpertAssign;
use crate::weight_manifest::{
    ExpertGroupSpec, ExpertParallelism, ExpertResourceRequirements, ExpertSourceLayout,
    ManifestPlan, WeightEntry,
};
use rdna_compute::DType;
use std::collections::{BTreeSet, HashMap, HashSet};

/// CPU metadata for one manifest source.  The loader must populate this from
/// the actual encoded tensor, never from an artifact basename or model config.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertSourceMetadata {
    /// Exact manifest source identity (`WeightEntry::name`).
    pub name: String,
    /// Stable source/container fingerprint.  All sources in one sealed group
    /// must come from the same source generation.
    pub fingerprint: String,
    /// Logical tensor shape of this source as loaded by the carrier.  Packed
    /// layouts include the expert dimension; per-expert layouts do not.
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
    /// Encoded bytes for this source, including all experts for packed layouts.
    pub encoded_bytes: usize,
    /// Encoded bytes between adjacent logical rows.
    pub row_stride: usize,
    pub alignment: usize,
    /// Carrier/quantizer identity.  Equal dtypes with different tags are not
    /// interchangeable.
    pub quant_tag: String,
    /// Natural or pre-rotated activation basis, supplied by the source loader.
    pub basis: String,
    /// Sidecar source identities referenced by this source, if any.
    pub sidecar_source_names: Vec<String>,
    /// Read-only alias owner.  Writable allocations are always owned by the
    /// owner source; aliases are checked as bounded views below.
    pub alias_owner: Option<String>,
    pub alias_byte_offset: Option<usize>,
}

impl ExpertSourceMetadata {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        fingerprint: impl Into<String>,
        logical_shape: Vec<usize>,
        dtype: DType,
        encoded_bytes: usize,
        row_stride: usize,
        alignment: usize,
        quant_tag: impl Into<String>,
        basis: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            fingerprint: fingerprint.into(),
            logical_shape,
            dtype,
            encoded_bytes,
            row_stride,
            alignment,
            quant_tag: quant_tag.into(),
            basis: basis.into(),
            sidecar_source_names: Vec::new(),
            alias_owner: None,
            alias_byte_offset: None,
        }
    }

    pub fn alias(mut self, owner: impl Into<String>, byte_offset: usize) -> Self {
        self.alias_owner = Some(owner.into());
        self.alias_byte_offset = Some(byte_offset);
        self
    }
}

/// One projection's validated source representation for one global expert.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertProjectionBinding {
    pub source_name: String,
    pub fingerprint: String,
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
    /// Bytes assigned to this logical projection by the manifest resource
    /// record.  For a fused gate/up source, gate and up each have their own
    /// exact slice and their sum equals `encoded_bytes`.
    pub projection_bytes: usize,
    /// Encoded bytes in the source view for this expert.
    pub encoded_bytes: usize,
    pub row_stride: usize,
    pub alignment: usize,
    pub quant_tag: String,
    pub basis: String,
    /// Exact sidecar identities for this projection.  A fused carrier keeps
    /// gate/up sidecars on the one carrier; separate carriers retain their
    /// independent sidecar lists.
    pub sidecar_source_names: Vec<String>,
    pub alias_owner: Option<String>,
    pub alias_byte_offset: Option<usize>,
}

/// One globally ordered expert and its deterministic owner/local slot.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertRecord {
    pub global_expert_id: usize,
    /// For replicated Single/TP plans rank zero is the designated owner.  All
    /// rank-local tables still expose every replicated expert explicitly.
    pub owner_rank: usize,
    pub local_slot: usize,
    pub gate: ExpertProjectionBinding,
    pub up: ExpertProjectionBinding,
    pub down: ExpertProjectionBinding,
    pub sidecars: Vec<ExpertProjectionBinding>,
}

/// Dimensions of the full expert matrices and the actual local shard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertProjectionDimensions {
    pub hidden: usize,
    pub intermediate: usize,
    pub global_gate_rows: usize,
    pub global_up_rows: usize,
    pub global_down_cols: usize,
    pub local_gate_rows: usize,
    pub local_up_rows: usize,
    pub local_down_cols: usize,
    pub local_down_rows: usize,
}

/// Ordered rank-local ownership view.  `logical_rank` indexes the mesh; the
/// physical selector is intentionally kept as a separate identity.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertRankOwnership {
    pub logical_rank: usize,
    pub physical_device: i32,
    pub global_expert_ids: Vec<usize>,
}

/// Immutable model-level sealed expert plan.  All fields are private so a
/// caller cannot manufacture a plan or mutate ownership after validation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExpertExecutionPlan {
    mesh_epoch: MeshEpoch,
    physical_devices: Vec<i32>,
    rank_devices: Vec<usize>,
    rank_ownership: Vec<ExpertRankOwnership>,
    global_to_local: Vec<Vec<Option<usize>>>,
    experts: Vec<ExpertRecord>,
    dimensions: ExpertProjectionDimensions,
    resources: ExpertResourceRequirements,
    group: String,
    layer: Option<usize>,
    n_experts: usize,
    parallelism: ExpertParallelism,
    assignment: ExpertAssign,
    router: String,
    execution: String,
    /// Activation basis of the gate/up input.  Down is intentionally tracked
    /// independently because Qwen4 uses FWHT-G256 gate/up and FWHT-G128 down.
    activation_basis: String,
    down_activation_basis: String,
    source_fingerprint: String,
    collective_rows: Vec<crate::weight_manifest::CollectiveScheduleEntry>,
}

impl ExpertExecutionPlan {
    pub fn mesh_epoch(&self) -> MeshEpoch {
        self.mesh_epoch
    }

    pub fn physical_devices(&self) -> &[i32] {
        &self.physical_devices
    }

    /// Logical mesh devices participating in this group's collective, in the
    /// exact manifest order.
    pub fn rank_devices(&self) -> &[usize] {
        &self.rank_devices
    }

    pub fn rank_ownership(&self) -> &[ExpertRankOwnership] {
        &self.rank_ownership
    }

    pub fn global_to_local(&self) -> &[Vec<Option<usize>>] {
        &self.global_to_local
    }

    pub fn experts(&self) -> &[ExpertRecord] {
        &self.experts
    }

    pub fn dimensions(&self) -> ExpertProjectionDimensions {
        self.dimensions
    }

    pub fn resources(&self) -> &ExpertResourceRequirements {
        &self.resources
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn parallelism(&self) -> ExpertParallelism {
        self.parallelism
    }

    pub fn assignment(&self) -> ExpertAssign {
        self.assignment
    }

    pub fn router(&self) -> &str {
        &self.router
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    pub fn activation_basis(&self) -> &str {
        &self.activation_basis
    }
    pub fn down_activation_basis(&self) -> &str {
        &self.down_activation_basis
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    pub fn collective_rows(&self) -> &[crate::weight_manifest::CollectiveScheduleEntry] {
        &self.collective_rows
    }

    /// Deterministic execution fingerprint of the whole sealed plan. Same
    /// plan on any rank or process renders the same string; any change to
    /// ownership, devices, epoch, sources, execution, or schedule changes
    /// it. Ranks compare this (never pointers) to prove they decode the
    /// same sealed generation.
    pub fn execution_fingerprint(&self) -> String {
        execution_contract_for_plan(self)
            .map(|contract| contract.fingerprint())
            .unwrap_or_else(|_| "sealed-ep/invalid".to_string())
    }

    pub fn owner_rank(&self, global_expert_id: usize) -> Option<usize> {
        self.experts
            .get(global_expert_id)
            .map(|expert| expert.owner_rank)
    }

    pub fn local_slot(&self, logical_rank: usize, global_expert_id: usize) -> Option<usize> {
        self.global_to_local
            .get(logical_rank)
            .and_then(|row| row.get(global_expert_id))
            .copied()
            .flatten()
    }
}

fn context(spec: &ExpertGroupSpec) -> String {
    format!("expert group '{}' layer {:?}", spec.group, spec.layer)
}

fn manifest_entry<'a>(
    spec: &ExpertGroupSpec,
    manifest: &'a [WeightEntry],
    name: &str,
    label: &str,
) -> Result<&'a WeightEntry, String> {
    manifest
        .iter()
        .find(|entry| entry.name == name && entry.layer == spec.layer)
        .ok_or_else(|| format!("{}: {label} source '{name}' is absent", context(spec)))
}

fn layout_projection_names(
    layout: &ExpertSourceLayout,
    expert: usize,
) -> Result<(String, String, String, Vec<String>), String> {
    let result = match layout {
        ExpertSourceLayout::PackedFused {
            gate_up,
            down,
            sidecars,
        } => (
            gate_up.clone(),
            gate_up.clone(),
            down.clone(),
            sidecars.clone(),
        ),
        ExpertSourceLayout::PackedSeparate {
            gate,
            up,
            down,
            sidecars,
        } => (gate.clone(), up.clone(), down.clone(), sidecars.clone()),
        ExpertSourceLayout::PerExpertFused {
            gate_up,
            down,
            sidecars,
        } => (
            gate_up
                .get(expert)
                .ok_or_else(|| format!("gate_up source missing expert {expert}"))?
                .clone(),
            gate_up
                .get(expert)
                .ok_or_else(|| format!("gate_up source missing expert {expert}"))?
                .clone(),
            down.get(expert)
                .ok_or_else(|| format!("down source missing expert {expert}"))?
                .clone(),
            sidecars.clone(),
        ),
        ExpertSourceLayout::PerExpertSeparate {
            gate,
            up,
            down,
            sidecars,
        } => (
            gate.get(expert)
                .ok_or_else(|| format!("gate source missing expert {expert}"))?
                .clone(),
            up.get(expert)
                .ok_or_else(|| format!("up source missing expert {expert}"))?
                .clone(),
            down.get(expert)
                .ok_or_else(|| format!("down source missing expert {expert}"))?
                .clone(),
            sidecars.clone(),
        ),
    };
    Ok(result)
}

fn is_packed(layout: &ExpertSourceLayout) -> bool {
    matches!(
        layout,
        ExpertSourceLayout::PackedFused { .. } | ExpertSourceLayout::PackedSeparate { .. }
    )
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProjectionKind {
    Gate,
    Up,
    Down,
}

fn fused_gate_up(layout: &ExpertSourceLayout) -> bool {
    matches!(
        layout,
        ExpertSourceLayout::PackedFused { .. } | ExpertSourceLayout::PerExpertFused { .. }
    )
}

fn projection_shape(
    spec: &ExpertGroupSpec,
    source_name: &str,
    storage_shape: &[usize],
    projection: ProjectionKind,
    fused: bool,
) -> Result<Vec<usize>, String> {
    if fused && matches!(projection, ProjectionKind::Gate | ProjectionKind::Up) {
        let rows = storage_shape[0];
        if rows % 2 != 0 {
            return Err(format!(
                "{}: fused gate_up source '{}' has odd row count {}",
                context(spec),
                source_name,
                rows
            ));
        }
        return Ok(vec![rows / 2, storage_shape[1]]);
    }
    Ok(storage_shape.to_vec())
}

fn source_map<'a>(
    sources: &'a [ExpertSourceMetadata],
) -> Result<HashMap<&'a str, &'a ExpertSourceMetadata>, String> {
    let mut map = HashMap::with_capacity(sources.len());
    for source in sources {
        if source.name.is_empty() {
            return Err("expert source has an empty manifest name".to_string());
        }
        if map.insert(source.name.as_str(), source).is_some() {
            return Err(format!(
                "duplicate expert source metadata '{} '",
                source.name
            ));
        }
        if source.fingerprint.is_empty()
            || source.logical_shape.is_empty()
            || source.logical_shape.contains(&0)
            || source.encoded_bytes == 0
            || source.row_stride == 0
            || source.alignment == 0
            || !source.alignment.is_power_of_two()
            || source.quant_tag.is_empty()
            || source.basis.is_empty()
        {
            return Err(format!(
                "expert source '{}' has malformed shape, bytes, alignment, tag, or basis metadata",
                source.name
            ));
        }
        if source.encoded_bytes % source.row_stride != 0 {
            return Err(format!(
                "expert source '{}' encoded bytes {} are not a row-stride multiple {}",
                source.name, source.encoded_bytes, source.row_stride
            ));
        }
        if let Some(expected) = expected_quant_row_stride(
            source.dtype,
            source.logical_shape.last().copied().unwrap_or(0),
        ) {
            if source.row_stride != expected {
                return Err(format!(
                    "expert source '{}' row_stride {} does not match {:?} encoded geometry {}",
                    source.name, source.row_stride, source.dtype, expected
                ));
            }
        }
        match (&source.alias_owner, source.alias_byte_offset) {
            (None, None) => {}
            (Some(_), Some(_)) => {}
            _ => {
                return Err(format!(
                    "expert source '{}' has an incomplete alias descriptor",
                    source.name
                ));
            }
        }
        let mut sidecars = HashSet::new();
        for sidecar in &source.sidecar_source_names {
            if sidecar.is_empty() || !sidecars.insert(sidecar) {
                return Err(format!(
                    "expert source '{}' has duplicate/empty sidecar identity",
                    source.name
                ));
            }
        }
    }

    Ok(map)
}

fn expected_quant_row_stride(dtype: DType, columns: usize) -> Option<usize> {
    let (group_bytes, group_width) = match dtype {
        // qt44: two f16 headers plus 128 payload bytes per 256 weights.
        DType::MQ4G256V2 => (136usize, 256usize),
        // qt53: two f16 headers plus 64 payload bytes per 128 weights.
        DType::MQ4G128V2 => (68usize, 128usize),
        _ => return None,
    };
    columns
        .checked_add(group_width - 1)
        .and_then(|rounded| rounded.checked_div(group_width))
        .and_then(|groups| groups.checked_mul(group_bytes))
}
fn manifest_source_names(specs: &[ExpertGroupSpec]) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    for spec in specs {
        if spec.router.is_empty() {
            return Err(format!("{}: router source name is empty", context(spec)));
        }
        names.insert(spec.router.clone());
        for expert in 0..spec.n_experts {
            let (gate, up, down, sidecars) = layout_projection_names(&spec.source_layout, expert)?;
            for name in [gate, up, down] {
                if name.is_empty() {
                    return Err(format!("{}: expert source name is empty", context(spec)));
                }
                names.insert(name);
            }
            for sidecar in sidecars {
                if sidecar.is_empty() {
                    return Err(format!("{}: expert sidecar name is empty", context(spec)));
                }
                names.insert(sidecar);
            }
        }
    }
    Ok(names)
}

/// Include source-side sidecars and alias owners in the global source set.
/// Missing closure members are left in the set so the caller can report them
/// as globally missing metadata; semantic alias checks stay group-local.
fn source_reference_closure(
    roots: &BTreeSet<String>,
    source_by_name: &HashMap<&str, &ExpertSourceMetadata>,
) -> BTreeSet<String> {
    let mut referenced = roots.clone();
    let mut pending = roots.iter().cloned().collect::<Vec<_>>();
    while let Some(name) = pending.pop() {
        let Some(source) = source_by_name.get(name.as_str()) else {
            continue;
        };
        for sidecar in &source.sidecar_source_names {
            if referenced.insert(sidecar.clone()) {
                pending.push(sidecar.clone());
            }
        }
        if let Some(owner) = source.alias_owner.as_deref() {
            if referenced.insert(owner.to_owned()) {
                pending.push(owner.to_owned());
            }
        }
    }
    referenced
}

fn validate_group_aliases<'a>(
    group_context: &str,
    names: &HashSet<&'a str>,
    source_by_name: &HashMap<&'a str, &'a ExpertSourceMetadata>,
) -> Result<(), String> {
    for name in names {
        let source = source_by_name.get(name).ok_or_else(|| {
            format!("{group_context}: source '{name}' disappeared during sealing")
        })?;
        if source.alias_owner.is_none() {
            continue;
        }
        let mut current = *source;
        let mut seen = HashSet::new();
        let mut offset = 0usize;
        loop {
            if !seen.insert(current.name.as_str()) {
                return Err(format!(
                    "{group_context}: source alias cycle at '{}'",
                    source.name
                ));
            }
            let Some(owner_name) = current.alias_owner.as_deref() else {
                break;
            };
            let owner = source_by_name.get(owner_name).ok_or_else(|| {
                format!(
                    "{group_context}: source '{}' aliases missing owner '{}'",
                    source.name, owner_name
                )
            })?;
            let alias_offset = current.alias_byte_offset.ok_or_else(|| {
                format!(
                    "{group_context}: source '{}' has no alias byte offset",
                    current.name
                )
            })?;
            if current.row_stride != owner.row_stride || current.alignment != owner.alignment {
                return Err(format!(
                    "{group_context}: source '{}' alias row_stride/alignment ({}/{}) differ from owner '{}' ({}/{})",
                    current.name,
                    current.row_stride,
                    current.alignment,
                    owner.name,
                    owner.row_stride,
                    owner.alignment
                ));
            }
            if alias_offset % owner.alignment != 0 {
                return Err(format!(
                    "{group_context}: source '{}' alias byte offset {} is not aligned to owner '{}' alignment {}",
                    current.name, alias_offset, owner.name, owner.alignment
                ));
            }
            offset = offset.checked_add(alias_offset).ok_or_else(|| {
                format!(
                    "{group_context}: source '{}' alias offset overflows",
                    source.name
                )
            })?;
            current = *owner;
        }
        let end = offset.checked_add(source.encoded_bytes).ok_or_else(|| {
            format!(
                "{group_context}: source '{}' alias range overflows",
                source.name
            )
        })?;
        if end > current.encoded_bytes {
            return Err(format!(
                "{group_context}: source '{}' alias range [{offset}, {end}) exceeds owner '{}' bytes {}",
                source.name, current.name, current.encoded_bytes
            ));
        }
        if source.dtype != current.dtype
            || source.fingerprint != current.fingerprint
            || source.quant_tag != current.quant_tag
            || source.basis != current.basis
        {
            return Err(format!(
                "{group_context}: source '{}' alias metadata differs from owner '{}'",
                source.name, current.name
            ));
        }
    }
    Ok(())
}

fn group_source_closure<'a>(
    group_context: &str,
    roots: &HashSet<&'a str>,
    source_by_name: &HashMap<&'a str, &'a ExpertSourceMetadata>,
) -> Result<HashSet<&'a str>, String> {
    let mut referenced = roots.clone();
    let mut pending = roots.iter().copied().collect::<Vec<_>>();
    while let Some(name) = pending.pop() {
        let source = source_by_name.get(name).ok_or_else(|| {
            format!("{group_context}: source '{name}' disappeared during sealing")
        })?;
        for sidecar in &source.sidecar_source_names {
            if !source_by_name.contains_key(sidecar.as_str()) {
                return Err(format!(
                    "{group_context}: source '{name}' references missing sidecar '{}'",
                    sidecar
                ));
            }
            if referenced.insert(sidecar.as_str()) {
                pending.push(sidecar.as_str());
            }
        }
        if let Some(owner) = source.alias_owner.as_deref() {
            if !source_by_name.contains_key(owner) {
                return Err(format!(
                    "{group_context}: source '{name}' aliases missing owner '{owner}'"
                ));
            }
            if referenced.insert(owner) {
                pending.push(owner);
            }
        }
    }
    Ok(referenced)
}
fn source_shape(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    source: &ExpertSourceMetadata,
    packed: bool,
    projection: ProjectionKind,
    fused: bool,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    let storage_shape = if packed {
        if source.logical_shape.len() != entry.logical_shape.len()
            || source.logical_shape.first() != Some(&spec.n_experts)
        {
            return Err(format!(
                "{}: source '{}' packed shape {:?} does not match manifest {:?}",
                context(spec),
                source.name,
                source.logical_shape,
                entry.logical_shape
            ));
        }
        source.logical_shape[1..].to_vec()
    } else {
        if entry.logical_shape.len() < 2 || source.logical_shape != entry.logical_shape[1..] {
            return Err(format!(
                "{}: source '{}' per-expert shape {:?} does not match manifest {:?}",
                context(spec),
                source.name,
                source.logical_shape,
                entry.logical_shape
            ));
        }
        source.logical_shape.clone()
    };
    if storage_shape.len() != 2 || storage_shape.iter().any(|&dim| dim == 0) {
        return Err(format!(
            "{}: source '{}' projection shape {:?} is not a non-zero matrix",
            context(spec),
            source.name,
            storage_shape
        ));
    }
    let logical_shape = projection_shape(spec, &source.name, &storage_shape, projection, fused)?;
    if logical_shape.iter().any(|&dim| dim == 0) {
        return Err(format!(
            "{}: source '{}' fused projection shape {:?} is not a non-zero matrix",
            context(spec),
            source.name,
            logical_shape
        ));
    }
    Ok((storage_shape, logical_shape))
}

fn manifest_shape(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    projection: ProjectionKind,
    fused: bool,
) -> Result<Vec<usize>, String> {
    let shape = entry.logical_shape.get(1..).ok_or_else(|| {
        format!(
            "{}: {} manifest shape is missing expert dimensions",
            context(spec),
            entry.name
        )
    })?;
    if shape.len() != 2 || shape.iter().any(|&dim| dim == 0) {
        return Err(format!(
            "{}: expert manifest projection geometry must be non-zero matrices",
            context(spec)
        ));
    }
    projection_shape(spec, &entry.name, shape, projection, fused)
}

fn validate_source_capacity(
    spec: &ExpertGroupSpec,
    source: &ExpertSourceMetadata,
    storage_shape: &[usize],
    packed: bool,
) -> Result<(), String> {
    let row_count = if packed {
        spec.n_experts
            .checked_mul(storage_shape[0])
            .ok_or_else(|| {
                format!(
                    "{}: source '{}' logical row count overflows usize",
                    context(spec),
                    source.name
                )
            })?
    } else {
        storage_shape[0]
    };
    let required = row_count.checked_mul(source.row_stride).ok_or_else(|| {
        format!(
            "{}: source '{}' row capacity overflows usize",
            context(spec),
            source.name
        )
    })?;
    if source.encoded_bytes < required {
        return Err(format!(
            "{}: source '{}' encoded bytes {} cannot cover {} logical rows at row_stride {}; requires at least {} bytes",
            context(spec),
            source.name,
            source.encoded_bytes,
            row_count,
            source.row_stride,
            required
        ));
    }
    Ok(())
}

fn source_bytes_per_expert(
    source: &ExpertSourceMetadata,
    packed: bool,
    n_experts: usize,
) -> Result<usize, String> {
    if !packed {
        return Ok(source.encoded_bytes);
    }
    source
        .encoded_bytes
        .checked_div(n_experts)
        .filter(|_| source.encoded_bytes % n_experts == 0)
        .ok_or_else(|| format!("source '{}' bytes do not divide n_experts", source.name))
}

fn validate_group_devices(
    spec: &ExpertGroupSpec,
    manifest: &[WeightEntry],
    plan: &ManifestPlan,
    mesh: &DeviceMesh,
) -> Result<Vec<usize>, String> {
    let first_name = match &spec.source_layout {
        ExpertSourceLayout::PackedFused { gate_up, .. }
        | ExpertSourceLayout::PackedSeparate { gate: gate_up, .. } => gate_up.as_str(),
        ExpertSourceLayout::PerExpertFused { gate_up, .. } => gate_up
            .first()
            .ok_or_else(|| format!("{}: missing gate_up source", context(spec)))?
            .as_str(),
        ExpertSourceLayout::PerExpertSeparate { gate, .. } => gate
            .first()
            .ok_or_else(|| format!("{}: missing gate source", context(spec)))?
            .as_str(),
    };
    let entry = manifest_entry(spec, manifest, first_name, "gate")?;
    let placement = plan
        .weights
        .iter()
        .find(|placement| placement.name == entry.name && placement.layer == entry.layer)
        .ok_or_else(|| {
            format!(
                "{}: manifest plan has no placement for '{} '",
                context(spec),
                entry.name
            )
        })?;
    if placement.devices.is_empty() {
        return Err(format!("{}: source placement is empty", context(spec)));
    }
    if placement
        .devices
        .iter()
        .any(|&device| device >= mesh.n_devices())
        || placement
            .devices
            .windows(2)
            .any(|window| window[0] == window[1])
    {
        return Err(format!(
            "{}: source placement has invalid logical ranks",
            context(spec)
        ));
    }
    let (axis, expected_policy) = match spec.parallelism {
        ExpertParallelism::Single => {
            if mesh.n_devices() != 1 || placement.devices != [0] {
                return Err(format!(
                    "{}: Single expert group requires one logical mesh device",
                    context(spec)
                ));
            }
            return Ok(vec![0]);
        }
        ExpertParallelism::TensorParallel => (DimKind::Tp, "tensor"),
        ExpertParallelism::ExpertParallel => (DimKind::Ep, "expert"),
    };
    let orthogonal_axis = match axis {
        DimKind::Ep => DimKind::Tp,
        DimKind::Tp => DimKind::Ep,
        DimKind::Pp => unreachable!("expert plans cannot use the PP axis"),
    };
    if mesh.size_of(orthogonal_axis) > 1 {
        return Err(format!(
            "{}: {expected_policy}-parallel plan must model the complete EP×TP Cartesian mesh; orthogonal axis {:?} has size {}",
            context(spec),
            orthogonal_axis,
            mesh.size_of(orthogonal_axis)
        ));
    }

    if mesh.size_of(axis) != placement.devices.len() {
        return Err(format!(
            "{}: {expected_policy}-parallel placement {:?} does not cover axis {:?}",
            context(spec),
            placement.devices,
            axis
        ));
    }
    let mut is_axis_group = false;
    for logical in &placement.devices {
        let coord = mesh
            .coord_of(*logical)
            .map_err(|error| format!("{}: invalid mesh rank: {error}", context(spec)))?;
        let group = mesh
            .group_along(axis, &coord)
            .map_err(|error| format!("{}: invalid mesh group: {error}", context(spec)))?;
        if group == placement.devices {
            is_axis_group = true;
            break;
        }
    }
    if !is_axis_group {
        return Err(format!(
            "{}: placement {:?} is not an ordered {:?} mesh group",
            context(spec),
            placement.devices,
            axis
        ));
    }
    Ok(placement.devices.clone())
}

fn validate_manifest_row(
    spec: &ExpertGroupSpec,
    entry: &WeightEntry,
    plan: &ManifestPlan,
    expected_kind: Option<DimKind>,
) -> Result<Option<crate::weight_manifest::CollectiveScheduleEntry>, String> {
    let row = plan
        .collective_schedule
        .iter()
        .find(|row| row.name == entry.name && entry.layer == Some(row.layer))
        .cloned();
    match (expected_kind, row) {
        (None, Some(_)) => Err(format!(
            "{}: Single group source '{}' unexpectedly authorizes a collective",
            context(spec),
            entry.name
        )),
        (Some(kind), Some(row)) => {
            if !matches!(
                row.hint,
                CollectiveHint::AllReduce { kind: actual } if actual == kind
            ) {
                return Err(format!(
                    "{}: source '{}' has unauthorized collective {:?}",
                    context(spec),
                    entry.name,
                    row.hint
                ));
            }
            Ok(Some(row))
        }
        (Some(_), None) => Err(format!(
            "{}: source '{}' is missing its manifest collective row",
            context(spec),
            entry.name
        )),
        (None, None) => Ok(None),
    }
}

fn validate_and_build_group(
    manifest: &[WeightEntry],
    plan: &ManifestPlan,
    spec: &ExpertGroupSpec,
    source_by_name: &HashMap<&str, &ExpertSourceMetadata>,
    physical_devices: &[i32],
    mesh: &DeviceMesh,
) -> Result<ExpertExecutionPlan, String> {
    let group_context = context(spec);
    let packed = is_packed(&spec.source_layout);
    let fused = fused_gate_up(&spec.source_layout);
    let rank_devices = validate_group_devices(spec, manifest, plan, mesh)?;

    let mut collective_rows = Vec::new();
    let mut all_names = HashSet::new();
    let mut fingerprint: Option<String> = None;
    let mut basis: Option<String> = None;
    let mut down_basis: Option<String> = None;
    let mut dimensions: Option<ExpertProjectionDimensions> = None;
    let mut records = Vec::with_capacity(spec.n_experts);
    let mut source_entries = Vec::new();

    for expert in 0..spec.n_experts {
        let (gate_name, up_name, down_name, sidecar_names) =
            layout_projection_names(&spec.source_layout, expert)?;
        let gate_entry = manifest_entry(spec, manifest, &gate_name, "gate")?;
        let up_entry = manifest_entry(spec, manifest, &up_name, "up")?;
        let down_entry = manifest_entry(spec, manifest, &down_name, "down")?;
        source_entries.extend([gate_entry, up_entry, down_entry]);
        for (label, entry) in [("gate", gate_entry), ("up", up_entry), ("down", down_entry)] {
            // TP gate/up are column shards and intentionally have no
            // collective row; only the row-sharded down projection reduces.
            let source_kind = match (spec.parallelism, label) {
                (ExpertParallelism::Single, _) => None,
                (ExpertParallelism::TensorParallel, "down") => Some(DimKind::Tp),
                (ExpertParallelism::TensorParallel, _) => None,
                (ExpertParallelism::ExpertParallel, _) => Some(DimKind::Ep),
            };
            if let Some(row) = validate_manifest_row(spec, entry, plan, source_kind)? {
                if !collective_rows.iter().any(
                    |existing: &crate::weight_manifest::CollectiveScheduleEntry| {
                        existing.name == row.name && existing.layer == row.layer
                    },
                ) {
                    collective_rows.push(row);
                }
            }
        }
        let gate = source_by_name
            .get(gate_name.as_str())
            .ok_or_else(|| format!("{group_context}: missing metadata for source '{gate_name}'"))?;
        let up = source_by_name
            .get(up_name.as_str())
            .ok_or_else(|| format!("{group_context}: missing metadata for source '{up_name}'"))?;
        let down = source_by_name
            .get(down_name.as_str())
            .ok_or_else(|| format!("{group_context}: missing metadata for source '{down_name}'"))?;
        let (gate_storage_shape, gate_shape) =
            source_shape(spec, gate_entry, gate, packed, ProjectionKind::Gate, fused)?;
        let (up_storage_shape, up_shape) =
            source_shape(spec, up_entry, up, packed, ProjectionKind::Up, fused)?;
        let (down_storage_shape, down_shape) =
            source_shape(spec, down_entry, down, packed, ProjectionKind::Down, false)?;
        validate_source_capacity(spec, gate, &gate_storage_shape, packed)?;
        validate_source_capacity(spec, up, &up_storage_shape, packed)?;
        validate_source_capacity(spec, down, &down_storage_shape, packed)?;
        if gate_shape != up_shape {
            return Err(format!(
                "{group_context}: expert {expert} gate/up shapes differ: {:?} vs {:?}",
                gate_shape, up_shape
            ));
        }
        let resource =
            spec.resources.experts.get(expert).ok_or_else(|| {
                format!("{group_context}: resource record missing expert {expert}")
            })?;
        let gate_bytes = source_bytes_per_expert(gate, packed, spec.n_experts)?;
        let up_bytes = source_bytes_per_expert(up, packed, spec.n_experts)?;
        let down_bytes = source_bytes_per_expert(down, packed, spec.n_experts)?;
        let expected_gate_up = resource
            .gate_bytes
            .checked_add(resource.up_bytes)
            .ok_or_else(|| {
                format!("{group_context}: expert {expert} gate/up resource sum overflows")
            })?;
        if gate_name == up_name {
            if gate_bytes != expected_gate_up {
                return Err(format!(
                    "{group_context}: expert {expert} fused gate/up bytes {gate_bytes} != declared {expected_gate_up}"
                ));
            }
        } else if gate_bytes != resource.gate_bytes || up_bytes != resource.up_bytes {
            return Err(format!(
                "{group_context}: expert {expert} separate gate/up bytes do not match resources"
            ));
        }
        if down_bytes != resource.down_bytes {
            return Err(format!(
                "{group_context}: expert {expert} down bytes {down_bytes} != declared {}",
                resource.down_bytes
            ));
        }
        for (role, source, entry) in [
            ("gate", gate, gate_entry),
            ("up", up, up_entry),
            ("down", down, down_entry),
        ] {
            if let Some(previous) = &fingerprint {
                if previous != &source.fingerprint {
                    return Err(format!(
                        "{group_context}: source fingerprint mismatch at '{}'",
                        source.name
                    ));
                }
            } else {
                fingerprint = Some(source.fingerprint.clone());
            }
            let projection_basis = if role == "down" {
                &mut down_basis
            } else {
                &mut basis
            };
            if let Some(previous) = projection_basis {
                if previous != &source.basis {
                    return Err(format!(
                        "{group_context}: {role} activation basis mismatch at '{}'",
                        source.name
                    ));
                }
            } else {
                *projection_basis = Some(source.basis.clone());
            }
            all_names.insert(source.name.as_str());
            if !entry.dtype_constraint.accepts(source.dtype) {
                return Err(format!(
                    "{group_context}: source '{}' dtype {:?} violates manifest constraint",
                    source.name, source.dtype
                ));
            }
        }
        let manifest_gate_shape = manifest_shape(spec, gate_entry, ProjectionKind::Gate, fused)?;
        let manifest_up_shape = manifest_shape(spec, up_entry, ProjectionKind::Up, fused)?;
        let manifest_down_shape = manifest_shape(spec, down_entry, ProjectionKind::Down, false)?;
        let hidden = manifest_gate_shape[1];
        let global_gate_rows = manifest_gate_shape[0];
        let global_up_rows = manifest_up_shape[0];
        let global_down_cols = manifest_down_shape[1];
        if global_gate_rows != global_up_rows
            || manifest_down_shape[0] != hidden
            || gate_shape[1] != hidden
            || up_shape[1] != hidden
            || down_shape[0] != hidden
        {
            return Err(format!(
                "{group_context}: expert {expert} projection geometry is inconsistent"
            ));
        }
        let (local_gate_rows, local_up_rows, local_down_cols) = if spec.parallelism
            == ExpertParallelism::TensorParallel
        {
            let tp = rank_devices.len();
            if tp == 0
                || global_gate_rows % tp != 0
                || global_up_rows % tp != 0
                || global_down_cols % tp != 0
            {
                return Err(format!(
                    "{group_context}: TP projection dimensions are not divisible by {tp}"
                ));
            }
            let local_gate_rows = global_gate_rows / tp;
            let local_up_rows = global_up_rows / tp;
            let local_down_cols = global_down_cols / tp;
            if gate_shape[0] != local_gate_rows
                || up_shape[0] != local_up_rows
                || down_shape[1] != local_down_cols
            {
                return Err(format!(
                    "{group_context}: source '{}' presents full-width buffers where TP shards are required",
                    gate_name
                ));
            }
            (local_gate_rows, local_up_rows, local_down_cols)
        } else {
            if gate_shape[0] != global_gate_rows
                || up_shape[0] != global_up_rows
                || down_shape[1] != global_down_cols
            {
                return Err(format!(
                    "{group_context}: source '{}' shape differs from manifest geometry",
                    gate_name
                ));
            }
            (global_gate_rows, global_up_rows, global_down_cols)
        };
        let current_dimensions = ExpertProjectionDimensions {
            hidden,
            intermediate: global_gate_rows,
            global_gate_rows,
            global_up_rows,
            global_down_cols,
            local_gate_rows,
            local_up_rows,
            local_down_cols,
            local_down_rows: down_shape[0],
        };
        if let Some(previous) = dimensions {
            if previous != current_dimensions {
                return Err(format!(
                    "{group_context}: expert {expert} dimensions differ from earlier experts"
                ));
            }
        } else {
            dimensions = Some(current_dimensions);
        }

        let owner_rank = match spec.parallelism {
            ExpertParallelism::Single | ExpertParallelism::TensorParallel => 0,
            ExpertParallelism::ExpertParallel => match spec.assignment {
                ExpertAssign::Stride => expert % rank_devices.len(),
                ExpertAssign::Contiguous => {
                    if spec.n_experts % rank_devices.len() != 0 {
                        return Err(format!(
                            "{group_context}: contiguous EP experts {} do not divide rank count {}",
                            spec.n_experts,
                            rank_devices.len()
                        ));
                    }
                    expert / (spec.n_experts / rank_devices.len())
                }
            },
        };
        let local_slot = match spec.parallelism {
            ExpertParallelism::Single | ExpertParallelism::TensorParallel => expert,
            ExpertParallelism::ExpertParallel => records
                .iter()
                .filter(|record: &&ExpertRecord| record.owner_rank == owner_rank)
                .count(),
        };
        let projection = |source: &ExpertSourceMetadata,
                          projection_bytes: usize,
                          encoded_bytes: usize,
                          logical_shape: &[usize]|
         -> ExpertProjectionBinding {
            ExpertProjectionBinding {
                source_name: source.name.clone(),
                fingerprint: source.fingerprint.clone(),
                logical_shape: logical_shape.to_vec(),
                dtype: source.dtype,
                projection_bytes,
                encoded_bytes,
                row_stride: source.row_stride,
                alignment: source.alignment,
                quant_tag: source.quant_tag.clone(),
                basis: source.basis.clone(),
                sidecar_source_names: source.sidecar_source_names.clone(),
                alias_owner: source.alias_owner.clone(),
                alias_byte_offset: source.alias_byte_offset,
            }
        };
        let mut sidecars = Vec::new();
        for sidecar_name in sidecar_names {
            let sidecar = source_by_name.get(sidecar_name.as_str()).ok_or_else(|| {
                format!("{group_context}: missing metadata for sidecar '{sidecar_name}'")
            })?;
            all_names.insert(sidecar.name.as_str());
            if let Some(previous) = &fingerprint {
                if previous != &sidecar.fingerprint {
                    return Err(format!(
                        "{group_context}: sidecar fingerprint mismatch at '{}'",
                        sidecar.name
                    ));
                }
            }
            let sidecar_encoded_bytes = source_bytes_per_expert(sidecar, packed, spec.n_experts)?;
            sidecars.push(projection(
                sidecar,
                sidecar.encoded_bytes,
                sidecar_encoded_bytes,
                &sidecar.logical_shape,
            ));
        }
        records.push(ExpertRecord {
            global_expert_id: expert,
            owner_rank,
            local_slot,
            gate: projection(gate, resource.gate_bytes, gate_bytes, &gate_shape),
            up: projection(up, resource.up_bytes, up_bytes, &up_shape),
            down: projection(down, resource.down_bytes, down_bytes, &down_shape),
            sidecars,
        });
    }

    // Aliased owners and sidecar references are part of this group's source
    // closure.  Metadata belonging only to another group is not inspected.
    let referenced = group_source_closure(&group_context, &all_names, source_by_name)?;
    validate_group_aliases(&group_context, &referenced, source_by_name)?;

    let mut rank_ownership = Vec::with_capacity(rank_devices.len());
    let mut global_to_local = vec![vec![None; spec.n_experts]; rank_devices.len()];
    for (logical_rank, &device) in rank_devices.iter().enumerate() {
        let global_expert_ids: Vec<usize> = records
            .iter()
            .filter(|record| {
                spec.parallelism != ExpertParallelism::ExpertParallel
                    || record.owner_rank == logical_rank
            })
            .map(|record| record.global_expert_id)
            .collect();
        for (slot, &expert) in global_expert_ids.iter().enumerate() {
            global_to_local[logical_rank][expert] = Some(slot);
        }
        rank_ownership.push(ExpertRankOwnership {
            logical_rank,
            physical_device: physical_devices[device],
            global_expert_ids,
        });
    }

    Ok(ExpertExecutionPlan {
        mesh_epoch: mesh.epoch(),
        physical_devices: physical_devices.to_vec(),
        rank_devices,
        rank_ownership,
        global_to_local,
        experts: records,
        dimensions: dimensions.ok_or_else(|| format!("{group_context}: no experts were sealed"))?,
        resources: spec.resources.clone(),
        group: spec.group.clone(),
        layer: spec.layer,
        n_experts: spec.n_experts,
        parallelism: spec.parallelism,
        assignment: spec.assignment,
        router: spec.router.clone(),
        execution: spec.execution.clone(),
        activation_basis: basis.ok_or_else(|| format!("{group_context}: no activation basis"))?,
        down_activation_basis: down_basis
            .ok_or_else(|| format!("{group_context}: no down activation basis"))?,
        source_fingerprint: fingerprint
            .ok_or_else(|| format!("{group_context}: no source fingerprint"))?,
        collective_rows,
    })
}

fn validate_physical_device_ids(
    physical_devices: &[i32],
    emulation_enabled: bool,
) -> Result<(), String> {
    let mut physical_seen = HashSet::new();
    for (rank, &device) in physical_devices.iter().enumerate() {
        if device < 0 {
            return Err(format!(
                "physical device id {device} is invalid at rank {rank}"
            ));
        }
        if !emulation_enabled && !physical_seen.insert(device) {
            return Err(format!(
                "physical device id {device} is duplicate at rank {rank}"
            ));
        }
    }
    Ok(())
}

/// Validate and construct all sealed expert plans.  Every source, rank, and
/// expert is checked before the first plan is returned; the function performs
/// no persistent ownership mutation and is safe to retry after an error.
pub fn plan_expert_execution(
    manifest: &[WeightEntry],
    plan: &ManifestPlan,
    specs: &[ExpertGroupSpec],
    sources: &[ExpertSourceMetadata],
    mesh: &DeviceMesh,
    physical_devices: &[i32],
) -> Result<Vec<ExpertExecutionPlan>, String> {
    if plan.mesh_epoch != mesh.epoch() {
        return Err(format!(
            "manifest plan mesh epoch {} does not match mesh epoch {}",
            plan.mesh_epoch.as_u64(),
            mesh.epoch().as_u64()
        ));
    }
    if physical_devices.len() != mesh.n_devices() {
        return Err(format!(
            "physical device count {} != logical mesh count {}",
            physical_devices.len(),
            mesh.n_devices()
        ));
    }
    let emulation_enabled = crate::config::get().emulate_gpus.is_some();
    validate_physical_device_ids(physical_devices, emulation_enabled)?;
    crate::weight_manifest::validate_expert_group_specs(specs, manifest)?;
    let source_by_name = source_map(sources)?;
    let manifest_names = manifest_source_names(specs)?;
    let referenced_names = source_reference_closure(&manifest_names, &source_by_name);
    let missing: Vec<String> = referenced_names
        .iter()
        .filter(|name| !source_by_name.contains_key(name.as_str()))
        .cloned()
        .collect();
    let extra: Vec<String> = source_by_name
        .keys()
        .filter(|name| !referenced_names.contains(**name))
        .map(|name| (**name).to_owned())
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(format!(
            "expert source set mismatch: missing {:?}, extra {:?}",
            missing, extra
        ));
    }
    let mut built = Vec::with_capacity(specs.len());
    for spec in specs {
        built.push(validate_and_build_group(
            manifest,
            plan,
            spec,
            &source_by_name,
            physical_devices,
            mesh,
        )?);
    }
    Ok(built)
}

fn rotation_plan_from_basis(basis: &str) -> Result<hipfire_dispatch::types::RotationPlan, String> {
    use hipfire_dispatch::types::RotationPlan;
    match basis {
        "None" | "none" | "natural" => Ok(RotationPlan::None),
        "FwhtG256" | "fwht-g256" => Ok(RotationPlan::FwhtG256),
        "FwhtG128" | "fwht-g128" => Ok(RotationPlan::FwhtG128),
        "Mq8Internal" | "mq8-internal" => Ok(RotationPlan::Mq8Internal),
        "Givens" | "givens" => Ok(RotationPlan::Givens),
        other => Err(format!("unknown expert activation basis '{other}'")),
    }
}

fn adapt_projection(
    projection: &ExpertProjectionBinding,
    shape: Vec<usize>,
    encoded_bytes: usize,
    layout_sidecars: &[ExpertProjectionBinding],
) -> Result<hipfire_dispatch::pipeline::sealed_moe::ExpertResource, String> {
    use hipfire_dispatch::pipeline::sealed_moe::{ExpertResource, ResourceAlias};

    if projection.encoded_bytes != encoded_bytes {
        return Err(format!(
            "expert source '{}' encoded byte identity changed while adapting ({}/{} bytes)",
            projection.source_name, projection.encoded_bytes, encoded_bytes
        ));
    }
    let basis = rotation_plan_from_basis(&projection.basis)?;
    let resource = ExpertResource::new(
        projection.source_name.clone(),
        projection.fingerprint.clone(),
        shape,
        projection.dtype,
        encoded_bytes,
        projection.row_stride,
        projection.alignment,
        basis,
    )
    .map_err(|error| {
        format!(
            "expert source '{}' metadata: {error:?}",
            projection.source_name
        )
    })?;
    let sidecars = if projection.sidecar_source_names.is_empty() {
        layout_sidecars
            .iter()
            .map(|sidecar| sidecar.source_name.clone())
            .collect()
    } else {
        projection.sidecar_source_names.clone()
    };
    let resource = resource.with_sidecars(sidecars).map_err(|error| {
        format!(
            "expert source '{}' sidecars: {error:?}",
            projection.source_name
        )
    })?;
    if let Some(owner) = projection.alias_owner.as_ref() {
        let alias = ResourceAlias::new(
            owner.clone(),
            projection.alias_byte_offset.unwrap_or(0),
            encoded_bytes,
        )
        .map_err(|error| {
            format!(
                "expert source '{}' alias: {error:?}",
                projection.source_name
            )
        })?;
        return resource.with_alias(alias).map_err(|error| {
            format!(
                "expert source '{}' alias: {error:?}",
                projection.source_name
            )
        });
    }
    if projection.alias_byte_offset.is_some() {
        return Err(format!(
            "expert source '{}' has an alias offset without an owner",
            projection.source_name
        ));
    }
    Ok(resource)
}

/// Adapt one validated runtime plan into the dispatch-owned metadata pair for
/// one logical rank.
///
/// This is the only production bridge from runtime source/placement metadata
/// to dispatch tables.  It performs no inference: every source shape, byte
/// range, rotation basis, sidecar identity, alias range, owner rank, and local
/// slot comes from the already validated [`ExpertExecutionPlan`].  The caller
/// names the rank it is adapting for; the adapter selects
/// `plan.rank_ownership()[local_rank]`, verifies the logical rank, the
/// `global_to_local` map, the mesh device list, the physical device, and the
/// compact local slots, then prepares that rank's cache.  A tampered or
/// stale plan (renumbered ranks, remapped slots, swapped devices) fails here,
/// before any table is published.
pub fn adapt_expert_execution_plan(
    plan: &ExpertExecutionPlan,
    local_rank: usize,
) -> Result<
    (
        hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
        hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    ),
    String,
> {
    use hipfire_dispatch::pipeline::sealed_moe::{ExpertResources, ExpertTable};

    let rank_count = plan.rank_ownership.len();
    if rank_count == 0 || plan.rank_devices.is_empty() || plan.rank_devices.len() != rank_count {
        return Err("sealed expert plan has no rank ownership".to_string());
    }
    if local_rank >= rank_count {
        return Err(format!(
            "sealed expert local rank {local_rank} is outside rank count {rank_count}"
        ));
    }
    let ownership = &plan.rank_ownership[local_rank];
    if ownership.logical_rank != local_rank {
        return Err(format!(
            "sealed expert ownership entry {local_rank} names logical rank {}",
            ownership.logical_rank
        ));
    }
    let mesh_device = *plan
        .rank_devices
        .get(local_rank)
        .ok_or_else(|| format!("sealed expert plan has no mesh device for rank {local_rank}"))?;
    let physical_device = *plan
        .physical_devices
        .get(mesh_device)
        .ok_or_else(|| "sealed expert plan rank device is out of range".to_string())?;
    if ownership.physical_device != physical_device {
        return Err(format!(
            "sealed expert rank {local_rank} physical device {} disagrees with mesh device {physical_device}",
            ownership.physical_device
        ));
    }
    if plan.global_to_local.len() != rank_count {
        return Err(format!(
            "sealed expert plan maps {} ranks, expected {rank_count}",
            plan.global_to_local.len()
        ));
    }
    for (rank, row) in plan.global_to_local.iter().enumerate() {
        if row.len() != plan.n_experts {
            return Err(format!(
                "sealed expert rank {rank} maps {} experts, expected {}",
                row.len(),
                plan.n_experts
            ));
        }
    }
    // Every record's sealed owner/slot must agree with the rank map, on all
    // ranks, not just the adapted one: a stale global table is rejected even
    // when the local row happens to look intact.
    for record in &plan.experts {
        let mapped = plan
            .global_to_local
            .get(record.owner_rank)
            .and_then(|row| row.get(record.global_expert_id))
            .copied()
            .flatten();
        if mapped != Some(record.local_slot) {
            return Err(format!(
                "sealed expert {} owner/slot ({}/{}) disagrees with the rank map",
                record.global_expert_id, record.owner_rank, record.local_slot
            ));
        }
    }
    let local_row = &plan.global_to_local[local_rank];
    let owned_count = local_row.iter().flatten().count();
    if owned_count != ownership.global_expert_ids.len() {
        return Err(format!(
            "sealed expert rank {local_rank} maps {owned_count} slots but owns {} experts",
            ownership.global_expert_ids.len()
        ));
    }
    for (slot, &expert) in ownership.global_expert_ids.iter().enumerate() {
        if local_row.get(expert).copied().flatten() != Some(slot) {
            return Err(format!(
                "sealed expert rank {local_rank} slot {slot} disagrees on expert {expert}"
            ));
        }
    }
    let mut records = Vec::with_capacity(plan.experts.len());
    for record in &plan.experts {
        let gate = &record.gate;
        let up = &record.up;
        let down = adapt_projection(
            &record.down,
            down_shape(&record.down),
            record.down.encoded_bytes,
            &record.sidecars,
        )?;
        let resources = if gate.source_name == up.source_name {
            if gate.fingerprint != up.fingerprint
                || gate.dtype != up.dtype
                || gate.row_stride != up.row_stride
                || gate.alignment != up.alignment
                || gate.quant_tag != up.quant_tag
                || gate.basis != up.basis
                || gate.sidecar_source_names != up.sidecar_source_names
                || gate.alias_owner != up.alias_owner
                || gate.alias_byte_offset != up.alias_byte_offset
                || gate.encoded_bytes != up.encoded_bytes
            {
                return Err(format!(
                    "expert {} fused gate/up source '{}' metadata differs between logical halves",
                    record.global_expert_id, gate.source_name
                ));
            }
            let gate_rows = gate.logical_shape.first().copied().ok_or_else(|| {
                format!(
                    "expert {} fused gate source has no rows",
                    record.global_expert_id
                )
            })?;
            let up_rows = up.logical_shape.first().copied().ok_or_else(|| {
                format!(
                    "expert {} fused up source has no rows",
                    record.global_expert_id
                )
            })?;
            let gate_cols = gate.logical_shape.get(1).copied().ok_or_else(|| {
                format!(
                    "expert {} fused gate source has no columns",
                    record.global_expert_id
                )
            })?;
            let up_cols = up.logical_shape.get(1).copied().ok_or_else(|| {
                format!(
                    "expert {} fused up source has no columns",
                    record.global_expert_id
                )
            })?;
            if gate_cols != up_cols
                || gate.projection_bytes.checked_add(up.projection_bytes)
                    != Some(gate.encoded_bytes)
            {
                return Err(format!(
                    "expert {} fused gate/up byte or shape halves do not reconstruct the carrier",
                    record.global_expert_id
                ));
            }
            let gate_up_rows = gate_rows.checked_add(up_rows).ok_or_else(|| {
                format!(
                    "expert {} fused gate/up rows overflow",
                    record.global_expert_id
                )
            })?;
            let gate_up = adapt_projection(
                gate,
                vec![gate_up_rows, gate_cols],
                gate.encoded_bytes,
                &record.sidecars,
            )?;
            ExpertResources::fused(gate_up, down)
        } else {
            if gate.projection_bytes != gate.encoded_bytes
                || up.projection_bytes != up.encoded_bytes
            {
                return Err(format!(
                    "expert {} separate gate/up projection byte identity changed",
                    record.global_expert_id
                ));
            }
            let gate_shape = gate.logical_shape.clone();
            let up_shape = up.logical_shape.clone();
            let gate = adapt_projection(gate, gate_shape, gate.encoded_bytes, &record.sidecars)?;
            let up = adapt_projection(up, up_shape, up.encoded_bytes, &record.sidecars)?;
            ExpertResources::separate(gate, up, down)
        }
        .map_err(|error| {
            format!(
                "expert {} resource metadata: {error:?}",
                record.global_expert_id
            )
        })?;
        records.push(
            hipfire_dispatch::pipeline::sealed_moe::ExpertMetadata::new(
                record.global_expert_id,
                record.owner_rank,
                record.local_slot,
                resources,
            )
            .map_err(|error| format!("expert {} metadata: {error:?}", record.global_expert_id))?,
        );
    }
    let table = ExpertTable::new(records)
        .map_err(|error| format!("sealed expert table adaptation: {error:?}"))?;
    let contract = execution_contract_for_plan(plan)?;
    let table = table
        .with_execution_contract(contract)
        .map_err(|error| format!("sealed expert contract adaptation: {error:?}"))?;
    let cache = table
        .prepare_binding(local_rank, rank_count, physical_device)
        .map_err(|error| format!("sealed expert cache adaptation: {error:?}"))?;
    Ok((table, cache))
}

fn down_shape(projection: &ExpertProjectionBinding) -> Vec<usize> {
    projection.logical_shape.clone()
}

/// Build the deterministic dispatch execution contract from one sealed plan.
/// Every field (group/layer, source fingerprint, mesh epoch, physical rank
/// list, parallelism, assignment, owner/slot map, execution string, ordered
/// collective rows) is copied from the attested plan; nothing is inferred.
pub fn execution_contract_for_plan(
    plan: &ExpertExecutionPlan,
) -> Result<hipfire_dispatch::pipeline::sealed_moe::ExpertExecutionContract, String> {
    use hipfire_dispatch::pipeline::sealed_moe::{
        ContractAssignment, ContractAxis, ContractCollectiveHint, ContractCollectiveRow,
        ContractParallelism, ExpertExecutionContract,
    };
    let parallelism = match plan.parallelism {
        ExpertParallelism::Single => ContractParallelism::Single,
        ExpertParallelism::TensorParallel => ContractParallelism::TensorParallel,
        ExpertParallelism::ExpertParallel => ContractParallelism::ExpertParallel,
    };
    let assignment = match plan.assignment {
        ExpertAssign::Stride => ContractAssignment::Stride,
        ExpertAssign::Contiguous => ContractAssignment::Contiguous,
    };
    let map_axis = |kind: DimKind| match kind {
        DimKind::Pp => ContractAxis::Pp,
        DimKind::Tp => ContractAxis::Tp,
        DimKind::Ep => ContractAxis::Ep,
    };
    let collective_rows = plan
        .collective_rows
        .iter()
        .map(|row| ContractCollectiveRow {
            name: row.name.clone(),
            layer: row.layer,
            hint: match row.hint {
                CollectiveHint::AllReduce { kind } => ContractCollectiveHint::AllReduce {
                    kind: map_axis(kind),
                },
                CollectiveHint::BandXfer { src, dst } => {
                    ContractCollectiveHint::BandXfer { src, dst }
                }
            },
        })
        .collect::<Vec<_>>();
    ExpertExecutionContract::new(
        plan.group.clone(),
        plan.layer,
        plan.source_fingerprint.clone(),
        plan.mesh_epoch.as_u64(),
        plan.physical_devices.clone(),
        parallelism,
        assignment,
        plan.experts
            .iter()
            .map(|record| record.owner_rank)
            .collect(),
        plan.experts
            .iter()
            .map(|record| record.local_slot)
            .collect(),
        plan.execution.clone(),
        collective_rows,
    )
    .map_err(|error| format!("sealed expert contract: {error:?}"))
}

/// Build and adapt a Single-device expert plan from loader-attested metadata.
///
/// Keeping manifest planning here ensures Qwen and Cohere loaders cannot
/// independently reconstruct ownership or collective policy.
pub fn plan_single_expert_execution(
    manifest: &[WeightEntry],
    spec: &ExpertGroupSpec,
    sources: &[ExpertSourceMetadata],
    n_layers: usize,
    physical_device: i32,
) -> Result<
    (
        ExpertExecutionPlan,
        hipfire_dispatch::pipeline::sealed_moe::ExpertTable,
        hipfire_dispatch::pipeline::sealed_moe::ExpertBindingCache,
    ),
    String,
> {
    if spec.parallelism != ExpertParallelism::Single {
        return Err(format!(
            "Single expert adapter received {:?} parallelism",
            spec.parallelism
        ));
    }
    let mesh = DeviceMesh::single().map_err(|error| format!("single expert mesh: {error}"))?;
    let manifest_plan = crate::weight_manifest::plan_manifest(manifest, &[], &mesh, n_layers)
        .map_err(|error| format!("single expert manifest planning: {error}"))?;
    let mut plans = plan_expert_execution(
        manifest,
        &manifest_plan,
        std::slice::from_ref(spec),
        sources,
        &mesh,
        &[physical_device],
    )?;
    let execution = plans
        .pop()
        .ok_or_else(|| "single expert planner returned no execution plan".to_string())?;
    let (table, cache) = adapt_expert_execution_plan(&execution, 0)?;
    Ok((execution, table, cache))
}

/// Repartition an already sealed Single expert plan into an expert-parallel
/// plan on `mesh` with `assignment` and `execution`.
///
/// This is the only sanctioned Single -> EP transition, and it is used for
/// post-load sharding. It accepts nothing but a sealed Single plan, rebuilds
/// ordinary planner input (manifest entries, group spec, source metadata)
/// from that plan's attested source/resource metadata, and reruns the same
/// manifest planner plus [`plan_expert_execution`] every fresh EP plan goes
/// through. Ownership, local slots, and collective rows therefore come from
/// the planner, never from a hand-built `e % N` table, and the EP collective
/// schedule is never omitted.
///
/// Source reconstruction mirrors the sealed layout: a base whose experts
/// share projection names rebuilds packed sources (with the total encoded
/// bytes recovered as per-expert bytes times `n_experts`); a base with
/// per-expert names rebuilds per-expert sources. Fused gate/up carriers are
/// detected per expert and must be uniform. Sidecars and aliases are carried
/// with their attested identities; an alias whose owner has no attested
/// metadata of its own cannot be repartitioned and fails here.
pub fn repartition_expert_execution_plan(
    base: &ExpertExecutionPlan,
    mesh: &DeviceMesh,
    physical_devices: &[i32],
    assignment: ExpertAssign,
    execution: impl Into<String>,
) -> Result<ExpertExecutionPlan, String> {
    use crate::weight_manifest::{
        ExpertGroupSpec, ExpertParallelism, ExpertSourceLayout, ShardPolicy,
    };
    if base.parallelism != ExpertParallelism::Single {
        return Err(format!(
            "repartition requires a sealed Single expert plan, got {:?}",
            base.parallelism
        ));
    }
    let n_experts = base.n_experts;
    if n_experts == 0 || base.experts.len() != n_experts {
        return Err("repartition requires a sealed plan covering nonzero experts".to_string());
    }
    if mesh.size_of(DimKind::Ep) == 0 {
        return Err("repartition requires a mesh with an expert-parallel axis".to_string());
    }
    let fused = base.experts[0].gate.source_name == base.experts[0].up.source_name;
    for record in &base.experts {
        if (record.gate.source_name == record.up.source_name) != fused {
            return Err(format!(
                "repartition refuses expert {} with mixed fused/separate gate/up identity",
                record.global_expert_id
            ));
        }
    }
    let packed = ["gate", "up", "down"].iter().all(|role| {
        let first = projection_source_name(&base.experts[0], role);
        base.experts
            .iter()
            .all(|record| projection_source_name(record, role) == first)
    });
    if !packed {
        // Per-expert layouts need distinct source names per expert within
        // each projection role; anything else is not a planner-admissible
        // layout and fails here instead of deep inside validation.
        for role in ["gate", "up", "down"] {
            let mut seen = HashSet::new();
            for record in &base.experts {
                let name = match role {
                    "gate" => record.gate.source_name.as_str(),
                    "up" => record.up.source_name.as_str(),
                    _ => record.down.source_name.as_str(),
                };
                if !seen.insert(name) {
                    return Err(format!(
                        "repartition refuses non-packed expert {role} source '{name}' shared across experts"
                    ));
                }
            }
        }
    }
    let reference_sidecars: Vec<String> = base.experts[0]
        .sidecars
        .iter()
        .map(|sidecar| sidecar.source_name.clone())
        .collect();
    for record in &base.experts {
        let names = record
            .sidecars
            .iter()
            .map(|sidecar| sidecar.source_name.clone())
            .collect::<Vec<_>>();
        if names != reference_sidecars {
            return Err(format!(
                "repartition refuses expert {} with non-uniform sidecar set",
                record.global_expert_id
            ));
        }
    }
    // Rebuild one source per distinct attested name. Packed sources recover
    // their full extent (`[n_experts, ..storage]`, per-expert bytes times
    // `n_experts`); per-expert sources keep the attested per-expert extent.
    let mut sources = Vec::new();
    let mut seen_sources = HashSet::new();
    let mut push_source = |binding: &ExpertProjectionBinding,
                           storage_shape: Vec<usize>,
                           context: &str|
     -> Result<(), String> {
        if !seen_sources.insert(binding.source_name.clone()) {
            let previous = sources
                .iter()
                .find(|source: &&ExpertSourceMetadata| source.name == binding.source_name)
                .expect("repartition source bookkeeping lost a source");
            let rebuilt = rebuilt_source(binding, &storage_shape, n_experts, packed)?;
            if *previous != rebuilt {
                return Err(format!(
                    "repartition refuses attested source '{}' with divergent {context} metadata",
                    binding.source_name
                ));
            }
            return Ok(());
        }
        sources.push(rebuilt_source(binding, &storage_shape, n_experts, packed)?);
        Ok(())
    };
    for record in &base.experts {
        let gate_storage = fused_storage(&record.gate.logical_shape, fused);
        let up_storage = fused_storage(&record.up.logical_shape, fused);
        let down_storage = record.down.logical_shape.clone();
        push_source(&record.gate, gate_storage, "gate")?;
        if record.up.source_name != record.gate.source_name {
            push_source(&record.up, up_storage, "up")?;
        }
        push_source(&record.down, down_storage, "down")?;
        for sidecar in &record.sidecars {
            push_source(sidecar, sidecar.logical_shape.clone(), "sidecar")?;
        }
    }
    for source in &sources {
        if let Some(owner) = source.alias_owner.as_deref() {
            if !sources.iter().any(|candidate| candidate.name == owner) {
                return Err(format!(
                    "repartition refuses alias '{}' whose owner '{owner}' has no attested metadata",
                    source.name
                ));
            }
        }
    }
    // Synthesize the manifest the planner expects: expert projections are
    // expert-sharded with the requested assignment, sidecars and the router
    // replicate. Entry shapes lead with `n_experts`, which the expert-shard
    // policy requires; dtypes and layer match the attested plan.
    let mut manifest = Vec::new();
    let layer = base.layer;
    let mut push_entry = |name: &str, shape: Vec<usize>, dtype: DType, policy: ShardPolicy| {
        let entry = match layer {
            Some(layer) => WeightEntry::layer(name, layer, shape, dtype, policy),
            None => WeightEntry::model(name, shape, dtype, policy),
        };
        manifest.push(entry);
    };
    let expert_policy = ShardPolicy::ExpertSharded {
        n_experts,
        assign: assignment,
    };
    let mut seen_entries = HashSet::new();
    for record in &base.experts {
        let projections = [
            (
                &record.gate,
                fused_storage(&record.gate.logical_shape, fused),
            ),
            (&record.up, fused_storage(&record.up.logical_shape, fused)),
            (&record.down, record.down.logical_shape.clone()),
        ];
        for (binding, storage) in projections {
            if seen_entries.insert(binding.source_name.clone()) {
                let mut shape = vec![n_experts];
                shape.extend_from_slice(&storage);
                push_entry(
                    &binding.source_name,
                    shape,
                    binding.dtype,
                    expert_policy.clone(),
                );
            }
        }
        for sidecar in &record.sidecars {
            if seen_entries.insert(sidecar.source_name.clone()) {
                push_entry(
                    &sidecar.source_name,
                    sidecar.logical_shape.clone(),
                    sidecar.dtype,
                    ShardPolicy::Replicate,
                );
            }
        }
    }
    push_entry(
        &base.router,
        vec![n_experts],
        DType::F32,
        ShardPolicy::Replicate,
    );
    let source_layout = if packed {
        let first = &base.experts[0];
        if fused {
            ExpertSourceLayout::PackedFused {
                gate_up: first.gate.source_name.clone(),
                down: first.down.source_name.clone(),
                sidecars: reference_sidecars.clone(),
            }
        } else {
            ExpertSourceLayout::PackedSeparate {
                gate: first.gate.source_name.clone(),
                up: first.up.source_name.clone(),
                down: first.down.source_name.clone(),
                sidecars: reference_sidecars.clone(),
            }
        }
    } else if fused {
        ExpertSourceLayout::PerExpertFused {
            gate_up: base
                .experts
                .iter()
                .map(|record| record.gate.source_name.clone())
                .collect(),
            down: base
                .experts
                .iter()
                .map(|record| record.down.source_name.clone())
                .collect(),
            sidecars: reference_sidecars.clone(),
        }
    } else {
        ExpertSourceLayout::PerExpertSeparate {
            gate: base
                .experts
                .iter()
                .map(|record| record.gate.source_name.clone())
                .collect(),
            up: base
                .experts
                .iter()
                .map(|record| record.up.source_name.clone())
                .collect(),
            down: base
                .experts
                .iter()
                .map(|record| record.down.source_name.clone())
                .collect(),
            sidecars: reference_sidecars.clone(),
        }
    };
    let n_layers = layer.map(|layer| layer + 1).unwrap_or(1);
    let manifest_plan = crate::weight_manifest::plan_manifest(&manifest, &[], mesh, n_layers)
        .map_err(|error| format!("repartition manifest planning: {error}"))?;
    let spec = ExpertGroupSpec {
        group: base.group.clone(),
        layer,
        n_experts,
        parallelism: ExpertParallelism::ExpertParallel,
        assignment,
        source_layout,
        resources: base.resources.clone(),
        router: base.router.clone(),
        execution: execution.into(),
    };
    let router_metadata = ExpertSourceMetadata::new(
        base.router.clone(),
        base.source_fingerprint.clone(),
        vec![n_experts],
        DType::F32,
        4 * n_experts,
        4,
        4,
        "f32",
        "natural",
    );
    sources.push(router_metadata);
    let mut plans = plan_expert_execution(
        &manifest,
        &manifest_plan,
        std::slice::from_ref(&spec),
        &sources,
        mesh,
        physical_devices,
    )
    .map_err(|error| format!("repartition expert planning: {error}"))?;
    plans
        .pop()
        .ok_or_else(|| "repartition planner returned no execution plan".to_string())
}

/// Per-expert storage rows behind one attested logical shape. Fused gate/up
/// records store one logical half; the carrier holds both halves.
fn fused_storage(logical_shape: &[usize], fused: bool) -> Vec<usize> {
    if !fused || logical_shape.len() != 2 {
        return logical_shape.to_vec();
    }
    vec![logical_shape[0] * 2, logical_shape[1]]
}

/// Rebuild one planner source from its attested per-expert binding.
fn rebuilt_source(
    binding: &ExpertProjectionBinding,
    storage_shape: &[usize],
    n_experts: usize,
    packed: bool,
) -> Result<ExpertSourceMetadata, String> {
    let (logical_shape, encoded_bytes) = if packed {
        let mut shape = vec![n_experts];
        shape.extend_from_slice(storage_shape);
        let bytes = binding
            .encoded_bytes
            .checked_mul(n_experts)
            .ok_or_else(|| {
                format!(
                    "repartition source '{}' packed bytes overflow",
                    binding.source_name
                )
            })?;
        (shape, bytes)
    } else {
        (storage_shape.to_vec(), binding.encoded_bytes)
    };
    let mut source = ExpertSourceMetadata::new(
        binding.source_name.clone(),
        binding.fingerprint.clone(),
        logical_shape,
        binding.dtype,
        encoded_bytes,
        binding.row_stride,
        binding.alignment,
        binding.quant_tag.clone(),
        binding.basis.clone(),
    );
    source.sidecar_source_names = binding.sidecar_source_names.clone();
    source.alias_owner = binding.alias_owner.clone();
    source.alias_byte_offset = binding.alias_byte_offset;
    Ok(source)
}

/// Attested source name behind one projection role of one sealed record.
fn projection_source_name<'a>(record: &'a ExpertRecord, role: &str) -> &'a str {
    match role {
        "gate" => record.gate.source_name.as_str(),
        "up" => record.up.source_name.as_str(),
        _ => record.down.source_name.as_str(),
    }
}

/// Tiny arithmetic expert matrices used by the CPU mesh conformance driver.
/// This intentionally computes real gate/up/activation/down products; it is
/// not a mock collective or an input echo.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuExpertMatrices {
    pub gate: Vec<Vec<f32>>,
    pub up: Vec<Vec<f32>>,
    pub down: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuRankProgram {
    pub logical_rank: usize,
    pub physical_device: i32,
    /// `(global expert ID, matrices)` in compact local-slot order.
    pub experts: Vec<(usize, CpuExpertMatrices)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuLaunch {
    pub logical_rank: usize,
    pub physical_device: i32,
    pub global_expert_id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CpuMeshResult {
    pub output: Vec<f32>,
    pub collective_count: usize,
}

fn matrix_vector(matrix: &[Vec<f32>], input: &[f32]) -> Result<Vec<f32>, String> {
    if matrix.is_empty() || matrix.iter().any(|row| row.len() != input.len()) {
        return Err("CPU expert matrix has invalid input width".to_string());
    }
    Ok(matrix
        .iter()
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect())
}

fn validate_cpu_programs(
    plan: &ExpertExecutionPlan,
    ranks: &[CpuRankProgram],
    input: &[f32],
    residual: &[f32],
    routes: &[(usize, f32)],
) -> Result<(), String> {
    if plan.parallelism == ExpertParallelism::TensorParallel {
        return Err(
            "CPU mesh driver currently requires complete expert matrices (not TP shards)"
                .to_string(),
        );
    }
    if input.len() != plan.dimensions.hidden || residual.len() != plan.dimensions.hidden {
        return Err("CPU mesh input/residual width does not match sealed hidden width".to_string());
    }
    if ranks.len() != plan.rank_ownership.len() {
        return Err("CPU rank program count does not match sealed rank count".to_string());
    }
    for (rank, program) in ranks.iter().enumerate() {
        let ownership = &plan.rank_ownership[rank];
        if program.logical_rank != ownership.logical_rank
            || program.physical_device != ownership.physical_device
        {
            return Err(format!(
                "CPU rank {rank} has stale logical/physical identity"
            ));
        }
        if program.experts.len() != ownership.global_expert_ids.len()
            || program
                .experts
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>()
                != ownership.global_expert_ids
        {
            return Err(format!(
                "CPU rank {rank} expert slots do not match sealed ownership"
            ));
        }
        for (expert, matrices) in &program.experts {
            if *expert >= plan.n_experts {
                return Err(format!(
                    "CPU rank {rank} contains out-of-range expert {expert}"
                ));
            }
            let check = |matrix: &[Vec<f32>], rows: usize, cols: usize| {
                matrix.len() == rows && matrix.iter().all(|row| row.len() == cols)
            };
            if !check(
                &matrices.gate,
                plan.dimensions.local_gate_rows,
                plan.dimensions.hidden,
            ) || !check(
                &matrices.up,
                plan.dimensions.local_up_rows,
                plan.dimensions.hidden,
            ) || !check(
                &matrices.down,
                plan.dimensions.local_down_rows,
                plan.dimensions.local_down_cols,
            ) {
                return Err(format!(
                    "CPU expert {expert} matrix geometry mismatches sealed plan"
                ));
            }
        }
    }
    for &(expert, weight) in routes {
        if expert >= plan.n_experts || !weight.is_finite() {
            return Err(format!(
                "CPU route contains invalid expert {expert} or weight"
            ));
        }
        if plan.owner_rank(expert).is_none() {
            return Err(format!("CPU route expert {expert} has no sealed owner"));
        }
    }
    Ok(())
}

/// Evaluate one expert's raw (unweighted) down row for the CPU mesh driver.
fn cpu_expert_row(matrices: &CpuExpertMatrices, input: &[f32]) -> Result<Vec<f32>, String> {
    let gate = matrix_vector(&matrices.gate, input)?;
    let up = matrix_vector(&matrices.up, input)?;
    let activated: Vec<f32> = gate.iter().zip(up).map(|(gate, up)| gate * up).collect();
    matrix_vector(&matrices.down, &activated)
}

/// Look up the sealed owner and matrices behind one routed expert.
fn cpu_owner_matrices<'a>(
    plan: &ExpertExecutionPlan,
    ranks: &'a [CpuRankProgram],
    expert: usize,
) -> Result<(usize, &'a CpuExpertMatrices), String> {
    let owner = plan
        .owner_rank(expert)
        .ok_or_else(|| format!("route expert {expert} has no owner"))?;
    let matrices = ranks
        .get(owner)
        .and_then(|program| {
            program
                .experts
                .iter()
                .find(|(id, _)| *id == expert)
                .map(|(_, matrices)| matrices)
        })
        .ok_or_else(|| format!("owner rank {owner} lacks expert {expert}"))?;
    Ok((owner, matrices))
}

/// Execute a tiny CPU MoE through the same rank ownership and one-reduction
/// semantics as the sealed mesh route.  `launches` is appended only after the
/// complete preflight succeeds, so an invalid later rank proves zero launches.
///
/// Each owning rank materializes its raw selected down rows; the single
/// canonical fold then accumulates `weight[i] * row[i]` in flat slot order
/// and adds the total to the residual once. Rank count never enters the
/// arithmetic: sharded EP computes exactly the Single association. (A
/// per-rank partial-sum association is kept only as a tests-only legacy
/// diagnostic next to the adversarial test below.)
pub fn execute_cpu_mesh(
    plan: &ExpertExecutionPlan,
    ranks: &[CpuRankProgram],
    input: &[f32],
    residual: &[f32],
    routes: &[(usize, f32)],
    launches: &mut Vec<CpuLaunch>,
) -> Result<CpuMeshResult, String> {
    validate_cpu_programs(plan, ranks, input, residual, routes)?;
    let mut rows = Vec::with_capacity(routes.len());
    for &(expert, _) in routes {
        let (owner, matrices) = cpu_owner_matrices(plan, ranks, expert)?;
        rows.push(cpu_expert_row(matrices, input)?);
        launches.push(CpuLaunch {
            logical_rank: owner,
            physical_device: ranks[owner].physical_device,
            global_expert_id: expert,
        });
    }
    let mut folded = vec![0.0f32; plan.dimensions.hidden];
    for (row, &(_, weight)) in rows.iter().zip(routes.iter()) {
        for (slot, value) in folded.iter_mut().zip(row.iter()) {
            *slot += weight * value;
        }
    }
    let mut output = residual.to_vec();
    for (out, value) in output.iter_mut().zip(folded) {
        *out += value;
    }
    Ok(CpuMeshResult {
        output,
        collective_count: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_mesh::DimKind;
    use crate::weight_manifest::{ExpertProjectionResources, PlacementHint, ShardPolicy};

    fn ep_fixture() -> (
        Vec<WeightEntry>,
        ManifestPlan,
        Vec<ExpertGroupSpec>,
        Vec<ExpertSourceMetadata>,
        DeviceMesh,
    ) {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 4)]).unwrap();
        let expert_policy = ShardPolicy::ExpertSharded {
            n_experts: 4,
            assign: ExpertAssign::Stride,
        };
        let manifest = vec![
            WeightEntry::layer("router", 0, vec![4, 2], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer("gate", 0, vec![4, 2, 2], DType::F32, expert_policy.clone()),
            WeightEntry::layer("up", 0, vec![4, 2, 2], DType::F32, expert_policy.clone()),
            WeightEntry::layer("down", 0, vec![4, 2, 2], DType::F32, expert_policy),
        ];
        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let resources = ExpertResourceRequirements {
            experts: vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 4,
                };
                4
            ],
        };
        let spec = ExpertGroupSpec {
            group: "ffn".into(),
            layer: Some(0),
            n_experts: 4,
            parallelism: ExpertParallelism::ExpertParallel,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedSeparate {
                gate: "gate".into(),
                up: "up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources,
            router: "router".into(),
            execution: "cpu.test".into(),
        };
        let source = |name: &str| {
            ExpertSourceMetadata::new(
                name,
                "fixture-v1",
                vec![4, 2, 2],
                DType::F32,
                64,
                8,
                4,
                "f32",
                "natural",
            )
        };
        (
            manifest,
            plan,
            vec![spec],
            vec![
                source("gate"),
                source("up"),
                source("down"),
                ExpertSourceMetadata::new(
                    "router",
                    "fixture-v1",
                    vec![4, 2],
                    DType::F32,
                    32,
                    16,
                    4,
                    "f32",
                    "natural",
                ),
            ],
            mesh,
        )
    }

    #[test]
    fn asymmetric_four_rank_cpu_mesh_uses_physical_ids_and_one_reduction() {
        let (manifest, manifest_plan, specs, sources, mesh) = ep_fixture();
        let physical = [7, 2, 11, 5];
        let plans = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &physical,
        )
        .unwrap();
        let plan = &plans[0];
        assert_eq!(
            plan.rank_ownership()
                .iter()
                .map(|rank| rank.physical_device)
                .collect::<Vec<_>>(),
            physical
        );
        let ranks = (0..4)
            .map(|rank| {
                let expert = rank;
                let scale = expert as f32 + 1.0;
                let diagonal = vec![vec![scale, 0.0], vec![0.0, scale]];
                CpuRankProgram {
                    logical_rank: rank,
                    physical_device: physical[rank],
                    experts: vec![(
                        expert,
                        CpuExpertMatrices {
                            gate: diagonal.clone(),
                            up: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                            down: diagonal,
                        },
                    )],
                }
            })
            .collect::<Vec<_>>();
        let mut launches = Vec::new();
        let result = execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[0.5, -0.25],
            &[(0, 1.0), (1, 0.5), (2, 1.0), (3, 0.25)],
            &mut launches,
        )
        .unwrap();
        assert_eq!(result.collective_count, 1);
        assert_eq!(
            launches
                .iter()
                .map(|launch| launch.physical_device)
                .collect::<Vec<_>>(),
            physical
        );
        assert_eq!(result.output, vec![16.5, 63.75]);
    }

    #[test]
    fn combined_sources_plan_independent_groups_and_reject_unreferenced() {
        let (mut manifest, _, mut specs, mut sources, mesh) = ep_fixture();
        let physical = [7, 2, 11, 5];

        // Give the first group its own source names, then clone its
        // shape/policy for a second independent group.  Both groups are
        // planned from one combined source metadata list.
        manifest[1].name = "gate_a".into();
        manifest[2].name = "up_a".into();
        manifest[3].name = "down_a".into();
        let mut first = specs.pop().unwrap();
        first.group = "first".into();
        first.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "gate_a".into(),
            up: "up_a".into(),
            down: "down_a".into(),
            sidecars: Vec::new(),
        };
        sources[0].name = "gate_a".into();
        sources[1].name = "up_a".into();
        sources[2].name = "down_a".into();

        let mut second_manifest = manifest[1..].to_vec();
        second_manifest[0].name = "gate_b".into();
        second_manifest[1].name = "up_b".into();
        second_manifest[2].name = "down_b".into();
        manifest.extend(second_manifest);

        let mut second = first.clone();
        second.group = "second".into();
        second.source_layout = ExpertSourceLayout::PackedSeparate {
            gate: "gate_b".into(),
            up: "up_b".into(),
            down: "down_b".into(),
            sidecars: Vec::new(),
        };
        let mut second_sources = sources.clone();
        second_sources[0].name = "gate_b".into();
        second_sources[1].name = "up_b".into();
        second_sources[2].name = "down_b".into();
        second_sources.retain(|source| source.name != "router");
        sources.extend(second_sources);
        specs = vec![first, second];

        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let planned =
            plan_expert_execution(&manifest, &plan, &specs, &sources, &mesh, &physical).unwrap();
        assert_eq!(planned.len(), 2);
        assert_eq!(planned[0].group(), "first");
        assert_eq!(planned[1].group(), "second");

        let mut with_extra = sources.clone();
        let mut extra = with_extra[0].clone();
        extra.name = "truly_unreferenced".into();
        with_extra.push(extra);
        let error = plan_expert_execution(&manifest, &plan, &specs, &with_extra, &mesh, &physical)
            .unwrap_err();
        assert!(error.contains("truly_unreferenced"), "got: {error}");
    }

    #[test]
    fn invalid_later_rank_fails_preflight_without_launching_earlier_ranks() {
        let (manifest, manifest_plan, specs, sources, mesh) = ep_fixture();
        let physical = [7, 2, 11, 5];
        let plan = &plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &physical,
        )
        .unwrap()[0];
        let mut ranks = (0..4)
            .map(|rank| CpuRankProgram {
                logical_rank: rank,
                physical_device: physical[rank],
                experts: vec![(
                    rank,
                    CpuExpertMatrices {
                        gate: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        up: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        down: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    },
                )],
            })
            .collect::<Vec<_>>();
        ranks[3].physical_device = 99;
        let mut launches = Vec::new();
        assert!(execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[(0, 1.0)],
            &mut launches,
        )
        .is_err());
        assert!(launches.is_empty());
    }

    #[test]
    fn duplicate_physical_ids_and_alias_cycles_are_refused_without_state() {
        let (manifest, manifest_plan, specs, sources, mesh) = ep_fixture();
        assert!(plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &[7, 7, 11, 5]
        )
        .is_err());
        assert!(validate_physical_device_ids(&[7, 7, 11, 5], false).is_err());
        let mut cyclic = sources.clone();
        cyclic[0] = cyclic[0].clone().alias("up", 0);
        cyclic[1] = cyclic[1].clone().alias("gate", 0);
        assert!(plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &cyclic,
            &mesh,
            &[7, 2, 11, 5]
        )
        .is_err());
    }

    #[test]
    fn physical_device_aliases_require_explicit_emulation() {
        assert!(validate_physical_device_ids(&[7, 7, 11, 5], false).is_err());
        assert!(validate_physical_device_ids(&[7, 7, 11, 5], true).is_ok());
        assert!(validate_physical_device_ids(&[-1, -1], true).is_err());
    }
    fn per_expert_fixture() -> (
        Vec<WeightEntry>,
        ManifestPlan,
        Vec<ExpertGroupSpec>,
        Vec<ExpertSourceMetadata>,
        DeviceMesh,
    ) {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let policy = ShardPolicy::ExpertSharded {
            n_experts: 2,
            assign: ExpertAssign::Stride,
        };
        let mut manifest = vec![WeightEntry::layer(
            "router",
            0,
            vec![2, 2],
            DType::F32,
            ShardPolicy::Replicate,
        )];
        for name in ["gate0", "gate1", "up0", "up1", "down0", "down1"] {
            manifest.push(WeightEntry::layer(
                name,
                0,
                vec![2, 2, 2],
                DType::F32,
                policy.clone(),
            ));
        }
        let resources = ExpertResourceRequirements::new(vec![
            ExpertProjectionResources {
                gate_bytes: 16,
                up_bytes: 16,
                down_bytes: 16,
                alignment: 8,
            };
            2
        ]);
        let spec = ExpertGroupSpec {
            group: "per-expert".into(),
            layer: Some(0),
            n_experts: 2,
            parallelism: ExpertParallelism::ExpertParallel,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PerExpertSeparate {
                gate: vec!["gate0".into(), "gate1".into()],
                up: vec!["up0".into(), "up1".into()],
                down: vec!["down0".into(), "down1".into()],
                sidecars: Vec::new(),
            },
            resources,
            router: "router".into(),
            execution: "cpu.test".into(),
        };
        let source = |name: &str| {
            ExpertSourceMetadata::new(
                name,
                "fixture-v2",
                vec![2, 2],
                DType::F32,
                16,
                8,
                8,
                "f32",
                "natural",
            )
        };
        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        (
            manifest,
            plan,
            vec![spec],
            ["gate0", "gate1", "up0", "up1", "down0", "down1", "router"]
                .into_iter()
                .map(source)
                .collect(),
            mesh,
        )
    }

    #[test]
    fn mixed_later_expert_dtype_and_basis_sidecar_identity_are_checked() {
        let (manifest, manifest_plan, specs, mut sources, mesh) = per_expert_fixture();
        for name in ["gate1", "up1", "down1"] {
            let source = sources
                .iter_mut()
                .find(|source| source.name == name)
                .unwrap();
            source.dtype = DType::MQ2G256LloydU;
            source.quant_tag = "mq2-lloyd-unrotated".into();
        }
        let physical = [7, 2];
        let plans = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &physical,
        )
        .unwrap();
        assert_eq!(plans[0].experts()[0].gate.dtype, DType::F32);
        assert_eq!(plans[0].experts()[1].gate.dtype, DType::MQ2G256LloydU);
        assert_eq!(plans[0].experts()[1].down.dtype, DType::MQ2G256LloydU);

        let mut wrong_basis = sources.clone();
        wrong_basis[1].basis = "fwht-g256".into();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &wrong_basis,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("activation basis mismatch"));

        let mut missing_sidecar = sources;
        missing_sidecar[0].sidecar_source_names = vec!["missing-sidecar".into()];
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &missing_sidecar,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("missing") && error.contains("missing-sidecar"));
    }

    #[test]
    fn duplicate_missing_and_out_of_range_alias_records_are_rejected() {
        let (manifest, manifest_plan, specs, sources, mesh) = per_expert_fixture();
        let physical = [7, 2];

        let mut duplicate = sources.clone();
        duplicate.push(sources[0].clone());
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &duplicate,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("duplicate expert source metadata"));

        let mut missing = sources.clone();
        missing.retain(|source| source.name != "gate1");
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &missing,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("missing") && error.contains("gate1"));

        let mut out_of_range = sources.clone();
        out_of_range[0] = out_of_range[0].clone().alias("up0", usize::MAX);
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &out_of_range,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("alias"));
    }

    #[test]
    fn mesh_epoch_rank_order_and_owner_device_are_bound_before_launch() {
        let (manifest, manifest_plan, specs, sources, mesh) = ep_fixture();
        let physical = [7, 2, 11, 5];
        let other_mesh = DeviceMesh::rect(&[(DimKind::Ep, 4)]).unwrap();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &other_mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("mesh epoch"));

        let planned = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &physical,
        )
        .unwrap();
        let plan = &planned[0];
        assert_eq!(plan.mesh_epoch(), manifest_plan.mesh_epoch);
        assert_eq!(plan.physical_devices(), &physical);
        assert_eq!(plan.source_fingerprint(), "fixture-v1");
        let mut wrong_source = sources.clone();
        wrong_source[0].fingerprint = "fixture-other-generation".into();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &wrong_source,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(
            error.contains("source fingerprint mismatch"),
            "got: {error}"
        );
        let identity = |rank: usize| CpuRankProgram {
            logical_rank: rank,
            physical_device: physical[rank],
            experts: vec![(
                rank,
                CpuExpertMatrices {
                    gate: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    up: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    down: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                },
            )],
        };
        let mut ranks = (0..4).map(identity).collect::<Vec<_>>();
        ranks.swap(0, 1);
        let mut launches = Vec::new();
        assert!(execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[(0, 1.0)],
            &mut launches,
        )
        .is_err());
        assert!(launches.is_empty());

        let mut ranks = (0..4).map(identity).collect::<Vec<_>>();
        ranks[2].physical_device = 123;
        assert!(execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[(0, 1.0)],
            &mut launches,
        )
        .is_err());
        assert!(launches.is_empty());
    }

    #[test]
    fn cpu_mesh_rejects_out_of_range_owner_and_preserves_once_only_residual() {
        let (manifest, manifest_plan, specs, sources, mesh) = ep_fixture();
        let physical = [7, 2, 11, 5];
        let planned = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &physical,
        )
        .unwrap();
        let plan = &planned[0];
        let mut ranks = (0..4)
            .map(|rank| CpuRankProgram {
                logical_rank: rank,
                physical_device: physical[rank],
                experts: vec![(
                    rank,
                    CpuExpertMatrices {
                        gate: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        up: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        down: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    },
                )],
            })
            .collect::<Vec<_>>();
        ranks[0].experts[0].0 = plan.n_experts();
        let mut launches = Vec::new();
        assert!(execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[3.0, -2.0],
            &[(0, 1.0)],
            &mut launches,
        )
        .is_err());
        assert!(launches.is_empty());

        let mut ranks = (0..4)
            .map(|rank| CpuRankProgram {
                logical_rank: rank,
                physical_device: physical[rank],
                experts: vec![(
                    rank,
                    CpuExpertMatrices {
                        gate: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        up: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                        down: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    },
                )],
            })
            .collect::<Vec<_>>();
        let mut launches = Vec::new();
        let result = execute_cpu_mesh(
            plan,
            &ranks,
            &[1.0, 2.0],
            &[3.0, -2.0],
            &[(0, 1.0), (1, 0.5)],
            &mut launches,
        )
        .unwrap();
        assert_eq!(result.collective_count, 1);
        assert_eq!(launches.len(), 2);
        assert_eq!(result.output, vec![4.5, 4.0]);
    }

    fn tp_fixture() -> (
        Vec<WeightEntry>,
        ManifestPlan,
        Vec<ExpertGroupSpec>,
        Vec<ExpertSourceMetadata>,
        DeviceMesh,
    ) {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let policy = |inner| ShardPolicy::ExpertTensorSharded {
            n_experts: 2,
            inner: Box::new(inner),
        };
        let manifest = vec![
            WeightEntry::layer("router", 0, vec![2, 2], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "gate",
                0,
                vec![2, 4, 2],
                DType::F32,
                policy(ShardPolicy::ColumnShard { axis: 1 }),
            ),
            WeightEntry::layer(
                "up",
                0,
                vec![2, 4, 2],
                DType::F32,
                policy(ShardPolicy::ColumnShard { axis: 1 }),
            ),
            WeightEntry::layer(
                "down",
                0,
                vec![2, 2, 4],
                DType::F32,
                policy(ShardPolicy::RowShard { axis: 2 }),
            ),
        ];
        let spec = ExpertGroupSpec {
            group: "tp".into(),
            layer: Some(0),
            n_experts: 2,
            parallelism: ExpertParallelism::TensorParallel,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedSeparate {
                gate: "gate".into(),
                up: "up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources: ExpertResourceRequirements::new(vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 8,
                };
                2
            ]),
            router: "router".into(),
            execution: "cpu.tp".into(),
        };
        let source = |name: &str| {
            ExpertSourceMetadata::new(
                name,
                "tp-v1",
                vec![2, 2, 2],
                DType::F32,
                32,
                8,
                8,
                "f32",
                "natural",
            )
        };
        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        (
            manifest,
            plan,
            vec![spec],
            vec![
                source("gate"),
                source("up"),
                source("down"),
                ExpertSourceMetadata::new(
                    "router",
                    "tp-v1",
                    vec![2, 2],
                    DType::F32,
                    16,
                    8,
                    8,
                    "f32",
                    "natural",
                ),
            ],
            mesh,
        )
    }

    #[test]
    fn tensor_parallel_plan_exposes_local_shards_not_full_width_buffers() {
        let (manifest, manifest_plan, specs, sources, mesh) = tp_fixture();
        let plans =
            plan_expert_execution(&manifest, &manifest_plan, &specs, &sources, &mesh, &[7, 2])
                .unwrap();
        let dimensions = plans[0].dimensions();
        assert_eq!(dimensions.global_gate_rows, 4);
        assert_eq!(dimensions.global_down_cols, 4);
        assert_eq!(dimensions.local_gate_rows, 2);
        assert_eq!(dimensions.local_up_rows, 2);
        assert_eq!(dimensions.local_down_cols, 2);
        assert_eq!(dimensions.local_down_rows, 2);

        let mut full_width_sources = sources;
        for source in &mut full_width_sources {
            source.logical_shape = vec![2, 4, 2];
            source.encoded_bytes = 64;
        }
        assert!(plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &full_width_sources,
            &mesh,
            &[7, 2],
        )
        .is_err());
    }
    #[test]
    fn source_capacity_covers_packed_and_per_expert_rows() {
        let (manifest, manifest_plan, specs, sources, mesh) = per_expert_fixture();
        let physical = [7, 2];
        assert_eq!(
            plan_expert_execution(
                &manifest,
                &manifest_plan,
                &specs,
                &sources,
                &mesh,
                &physical
            )
            .unwrap()
            .len(),
            1
        );

        let mut short_specs = specs.clone();
        for resource in &mut short_specs[0].resources.experts {
            resource.gate_bytes = 8;
            resource.up_bytes = 8;
            resource.down_bytes = 8;
        }
        let mut short_sources = sources.clone();
        for source in &mut short_sources {
            if source.name != "router" {
                source.encoded_bytes = 8;
            }
        }
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &short_specs,
            &short_sources,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("cannot cover"), "got: {error}");

        let (packed_manifest, packed_plan, packed_specs, packed_sources, packed_mesh) =
            ep_fixture();
        let mut exact_specs = packed_specs.clone();
        for resource in &mut exact_specs[0].resources.experts {
            resource.gate_bytes = 8;
            resource.up_bytes = 8;
            resource.down_bytes = 8;
        }
        let mut exact_sources = packed_sources.clone();
        for source in &mut exact_sources {
            if source.name != "router" {
                source.encoded_bytes = 32;
                source.row_stride = 4;
                source.alignment = 4;
            }
        }
        assert!(plan_expert_execution(
            &packed_manifest,
            &packed_plan,
            &exact_specs,
            &exact_sources,
            &packed_mesh,
            &[7, 2, 11, 5],
        )
        .is_ok());

        let mut short_specs = exact_specs;
        for resource in &mut short_specs[0].resources.experts {
            resource.gate_bytes = 7;
            resource.up_bytes = 7;
            resource.down_bytes = 7;
            resource.alignment = 1;
        }
        let mut short_sources = exact_sources;
        for source in &mut short_sources {
            if source.name != "router" {
                source.encoded_bytes = 28;
                source.alignment = 1;
            }
        }
        let error = plan_expert_execution(
            &packed_manifest,
            &packed_plan,
            &short_specs,
            &short_sources,
            &packed_mesh,
            &[7, 2, 11, 5],
        )
        .unwrap_err();
        assert!(error.contains("cannot cover"), "got: {error}");
    }

    #[test]
    fn one_axis_moe_plan_rejects_non_degenerate_cross_axis_mesh() {
        let (manifest, _, specs, sources, _) = ep_fixture();
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2), (DimKind::Tp, 2)]).unwrap();
        let manifest_plan =
            crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &[7, 2, 11, 5],
        )
        .unwrap_err();
        assert!(error.contains("Cartesian"), "got: {error}");

        let (manifest, _, specs, sources, _) = tp_fixture();
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2), (DimKind::Tp, 2)]).unwrap();
        let manifest_plan =
            crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &[7, 2, 11, 5],
        )
        .unwrap_err();
        assert!(error.contains("Cartesian"), "got: {error}");
    }

    #[test]
    fn aliases_require_matching_stride_alignment_and_aligned_offsets() {
        let (manifest, manifest_plan, specs, sources, mesh) = per_expert_fixture();
        let physical = [7, 2];
        let mut valid = sources.clone();
        valid[0].sidecar_source_names = vec!["router".into()];
        valid[1].sidecar_source_names = vec!["router".into()];
        valid[1] = valid[1].clone().alias("gate0", 0);
        let valid_plan =
            plan_expert_execution(&manifest, &manifest_plan, &specs, &valid, &mesh, &physical)
                .unwrap();
        let (table, cache) = adapt_expert_execution_plan(&valid_plan[0], 0).unwrap();
        let gate = table.experts()[1].resources().gate().unwrap();
        assert_eq!(gate.sidecars(), &["router"]);
        assert_eq!(gate.alias().unwrap().owner(), "gate0");
        assert_eq!(cache.local_expert_ids(), &[0]);
        let mut subrange = sources.clone();
        let mut owner = subrange[0].clone();
        owner.name = "gate_owner".into();
        owner.logical_shape = vec![4, 2];
        owner.encoded_bytes = 32;
        subrange[0] = subrange[0].clone().alias("gate_owner", 0);
        subrange.push(owner);
        assert!(plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &subrange,
            &mesh,
            &physical,
        )
        .is_ok());

        let mut bad_stride = valid.clone();
        bad_stride[1].row_stride = 4;
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &bad_stride,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("row_stride"), "got: {error}");

        let mut bad_alignment = valid.clone();
        bad_alignment[1].alignment = 4;
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &bad_alignment,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("alignment"), "got: {error}");

        let mut offset_specs = specs.clone();
        offset_specs[0].resources.experts[0].gate_bytes = 32;
        let mut bad_offset = valid;
        bad_offset[0].encoded_bytes = 32;
        bad_offset[1] = bad_offset[1].clone().alias("gate0", 4);
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &offset_specs,
            &bad_offset,
            &mesh,
            &physical,
        )
        .unwrap_err();
        assert!(error.contains("aligned"), "got: {error}");
    }

    #[test]
    fn fused_gate_up_source_seals_two_logical_halves_with_shared_identity() {
        let mesh = DeviceMesh::single().unwrap();
        let manifest = vec![
            WeightEntry::layer("router", 0, vec![2, 1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "gate_up",
                0,
                vec![2, 4, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer("down", 0, vec![2, 2, 2], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "rotation",
                0,
                vec![2, 1],
                DType::F32,
                ShardPolicy::Replicate,
            ),
        ];
        let spec = ExpertGroupSpec {
            group: "fused".into(),
            layer: Some(0),
            n_experts: 2,
            parallelism: ExpertParallelism::Single,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "gate_up".into(),
                down: "down".into(),
                sidecars: vec!["rotation".into()],
            },
            resources: ExpertResourceRequirements::new(vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 8,
                };
                2
            ]),
            router: "router".into(),
            execution: "cpu.fused".into(),
        };
        let sources = vec![
            ExpertSourceMetadata::new(
                "router",
                "fused-v1",
                vec![2, 1],
                DType::F32,
                8,
                8,
                8,
                "f32",
                "natural",
            ),
            ExpertSourceMetadata::new(
                "gate_up",
                "fused-v1",
                vec![2, 4, 2],
                DType::F32,
                64,
                8,
                8,
                "f32",
                "natural",
            ),
            ExpertSourceMetadata::new(
                "down",
                "fused-v1",
                vec![2, 2, 2],
                DType::F32,
                32,
                8,
                8,
                "f32",
                "natural",
            ),
            ExpertSourceMetadata::new(
                "rotation",
                "fused-v1",
                vec![2, 1],
                DType::F32,
                8,
                4,
                4,
                "f32",
                "natural",
            ),
        ];
        let manifest_plan =
            crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let plans =
            plan_expert_execution(&manifest, &manifest_plan, &[spec], &sources, &mesh, &[9])
                .unwrap();
        let plan = &plans[0];
        let dimensions = plan.dimensions();
        assert_eq!(dimensions.global_gate_rows, 2);
        assert_eq!(dimensions.global_up_rows, 2);
        assert_eq!(dimensions.global_down_cols, 2);
        let record = &plan.experts()[0];
        assert_eq!(record.gate.source_name, "gate_up");
        assert_eq!(record.up.source_name, "gate_up");
        assert_eq!(record.gate.logical_shape, vec![2, 2]);
        assert_eq!(record.up.logical_shape, vec![2, 2]);
        assert_eq!(record.gate.projection_bytes, 16);
        assert_eq!(record.up.projection_bytes, 16);
        assert_eq!(record.gate.encoded_bytes, 32);
        assert_eq!(record.up.encoded_bytes, 32);
        let (table, cache) = adapt_expert_execution_plan(plan, 0).unwrap();
        let gate_up = table.experts()[0].resources().gate_up().unwrap();
        assert_eq!(gate_up.source_name(), "gate_up");
        assert_eq!(gate_up.shape(), &[4, 2]);
        assert_eq!(gate_up.encoded_bytes(), 32);
        assert_eq!(gate_up.sidecars(), &["rotation"]);
        assert_eq!(cache.physical_device(), 9);
    }

    fn ep_mesh_fixture(
        n_experts: usize,
        ranks: usize,
        assign: ExpertAssign,
        execution: &str,
    ) -> (
        Vec<WeightEntry>,
        ManifestPlan,
        Vec<ExpertGroupSpec>,
        Vec<ExpertSourceMetadata>,
        DeviceMesh,
    ) {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, ranks)]).unwrap();
        let expert_policy = ShardPolicy::ExpertSharded { n_experts, assign };
        let manifest = vec![
            WeightEntry::layer(
                "router",
                0,
                vec![n_experts, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer(
                "gate",
                0,
                vec![n_experts, 2, 2],
                DType::F32,
                expert_policy.clone(),
            ),
            WeightEntry::layer(
                "up",
                0,
                vec![n_experts, 2, 2],
                DType::F32,
                expert_policy.clone(),
            ),
            WeightEntry::layer("down", 0, vec![n_experts, 2, 2], DType::F32, expert_policy),
        ];
        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let resources = ExpertResourceRequirements {
            experts: vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 4,
                };
                n_experts
            ],
        };
        let spec = ExpertGroupSpec {
            group: "ffn".into(),
            layer: Some(0),
            n_experts,
            parallelism: ExpertParallelism::ExpertParallel,
            assignment: assign,
            source_layout: ExpertSourceLayout::PackedSeparate {
                gate: "gate".into(),
                up: "up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources,
            router: "router".into(),
            execution: execution.into(),
        };
        let source = |name: &str| {
            ExpertSourceMetadata::new(
                name,
                "fixture-v1",
                vec![n_experts, 2, 2],
                DType::F32,
                16 * n_experts,
                8,
                4,
                "f32",
                "natural",
            )
        };
        (
            manifest,
            plan,
            vec![spec],
            vec![
                source("gate"),
                source("up"),
                source("down"),
                ExpertSourceMetadata::new(
                    "router",
                    "fixture-v1",
                    vec![n_experts, 2],
                    DType::F32,
                    16 * n_experts,
                    16,
                    4,
                    "f32",
                    "natural",
                ),
            ],
            mesh,
        )
    }

    fn single_mesh_fixture(
        n_experts: usize,
    ) -> (
        Vec<WeightEntry>,
        ManifestPlan,
        Vec<ExpertGroupSpec>,
        Vec<ExpertSourceMetadata>,
        DeviceMesh,
    ) {
        let mesh = DeviceMesh::single().unwrap();
        let manifest = vec![
            WeightEntry::layer(
                "router",
                0,
                vec![n_experts, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer(
                "gate",
                0,
                vec![n_experts, 2, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer(
                "up",
                0,
                vec![n_experts, 2, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer(
                "down",
                0,
                vec![n_experts, 2, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
        ];
        let plan = crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1).unwrap();
        let resources = ExpertResourceRequirements {
            experts: vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 4,
                };
                n_experts
            ],
        };
        let spec = ExpertGroupSpec {
            group: "ffn".into(),
            layer: Some(0),
            n_experts,
            parallelism: ExpertParallelism::Single,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedSeparate {
                gate: "gate".into(),
                up: "up".into(),
                down: "down".into(),
                sidecars: Vec::new(),
            },
            resources,
            router: "router".into(),
            execution: "indexed-single".into(),
        };
        let source = |name: &str| {
            ExpertSourceMetadata::new(
                name,
                "fixture-v1",
                vec![n_experts, 2, 2],
                DType::F32,
                16 * n_experts,
                8,
                4,
                "f32",
                "natural",
            )
        };
        (
            manifest,
            plan,
            vec![spec],
            vec![
                source("gate"),
                source("up"),
                source("down"),
                ExpertSourceMetadata::new(
                    "router",
                    "fixture-v1",
                    vec![n_experts, 2],
                    DType::F32,
                    16 * n_experts,
                    16,
                    4,
                    "f32",
                    "natural",
                ),
            ],
            mesh,
        )
    }

    fn seal_ep(
        n_experts: usize,
        ranks: usize,
        assign: ExpertAssign,
        physical: &[i32],
    ) -> (ExpertExecutionPlan, DeviceMesh) {
        let (manifest, manifest_plan, specs, sources, mesh) =
            ep_mesh_fixture(n_experts, ranks, assign, "indexed-decode-routed-partial");
        let plan =
            plan_expert_execution(&manifest, &manifest_plan, &specs, &sources, &mesh, physical)
                .unwrap()
                .pop()
                .unwrap();
        (plan, mesh)
    }

    #[test]
    fn ep2_ep4_stride_and_contiguous_ownership_are_planner_derived() {
        for (ranks, assign) in [
            (2, ExpertAssign::Stride),
            (2, ExpertAssign::Contiguous),
            (4, ExpertAssign::Stride),
            (4, ExpertAssign::Contiguous),
        ] {
            let physical = vec![7, 2, 11, 5][..ranks].to_vec();
            let (plan, _) = seal_ep(8, ranks, assign, &physical);
            assert_eq!(plan.parallelism(), ExpertParallelism::ExpertParallel);
            assert_eq!(plan.assignment(), assign);
            let owner = |expert: usize| match assign {
                ExpertAssign::Stride => expert % ranks,
                ExpertAssign::Contiguous => expert / (8 / ranks),
            };
            let expected: Vec<Vec<usize>> = (0..ranks)
                .map(|rank| (0..8).filter(|&expert| owner(expert) == rank).collect())
                .collect();
            let actual: Vec<Vec<usize>> = plan
                .rank_ownership()
                .iter()
                .map(|ownership| ownership.global_expert_ids.clone())
                .collect();
            assert_eq!(actual, expected, "ranks={ranks} assign={assign:?}");
            for (rank, ownership) in plan.rank_ownership().iter().enumerate() {
                assert_eq!(ownership.logical_rank, rank);
                assert_eq!(ownership.physical_device, physical[rank]);
                for (slot, &expert) in ownership.global_expert_ids.iter().enumerate() {
                    assert_eq!(plan.owner_rank(expert), Some(rank));
                    assert_eq!(plan.local_slot(rank, expert), Some(slot));
                }
            }
            for rank in 0..ranks {
                let (table, cache) = adapt_expert_execution_plan(&plan, rank).unwrap();
                assert_eq!(cache.local_rank(), rank);
                assert_eq!(cache.rank_count(), ranks);
                assert_eq!(cache.physical_device(), physical[rank]);
                assert_eq!(cache.local_expert_ids(), &expected[rank]);
                let contract = table
                    .execution_contract()
                    .expect("adapted table carries a contract");
                assert_eq!(contract.owner_ranks(), &expected_owner_ranks(&plan));
                assert!(contract
                    .fingerprint()
                    .contains("indexed-decode-routed-partial"));
                assert!(
                    contract.is_root_routed_ep(),
                    "plan-bound EP contract must authorize RootRoutedPartial"
                );
            }
        }
    }

    fn expected_owner_ranks(plan: &ExpertExecutionPlan) -> Vec<usize> {
        (0..plan.n_experts())
            .map(|expert| plan.owner_rank(expert).unwrap())
            .collect()
    }

    #[test]
    fn adapt_for_rank_rejects_stale_identity_slots_and_devices() {
        let physical = [7, 2];
        let (plan, _) = seal_ep(8, 2, ExpertAssign::Stride, &physical);
        assert!(adapt_expert_execution_plan(&plan, 2).is_err());
        let mut bad = plan.clone();
        bad.rank_ownership[1].physical_device = 999;
        let error = adapt_expert_execution_plan(&bad, 1).unwrap_err();
        assert!(error.contains("physical device"), "got: {error}");
        let mut bad = plan.clone();
        bad.rank_ownership[1].logical_rank = 0;
        let error = adapt_expert_execution_plan(&bad, 1).unwrap_err();
        assert!(error.contains("logical rank"), "got: {error}");
        let mut bad = plan.clone();
        bad.global_to_local[1][1] = Some(7);
        let error = adapt_expert_execution_plan(&bad, 1).unwrap_err();
        assert!(error.contains("rank map"), "got: {error}");
        // A phantom slot for an expert owned elsewhere passes the record
        // cross-check but breaks the owned-count agreement.
        let mut bad = plan.clone();
        bad.global_to_local[1][0] = Some(0);
        let error = adapt_expert_execution_plan(&bad, 1).unwrap_err();
        assert!(error.contains("owns"), "got: {error}");
    }

    #[test]
    fn stale_mesh_epoch_and_physical_topology_fail_before_ownership() {
        let (manifest, manifest_plan, specs, sources, mesh) =
            ep_mesh_fixture(8, 2, ExpertAssign::Stride, "indexed-decode-routed-partial");
        let other_mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &other_mesh,
            &[7, 2],
        )
        .unwrap_err();
        assert!(error.contains("mesh epoch"), "got: {error}");
        let error = plan_expert_execution(
            &manifest,
            &manifest_plan,
            &specs,
            &sources,
            &mesh,
            &[7, 2, 11],
        )
        .unwrap_err();
        assert!(error.contains("physical device count"), "got: {error}");
        let error = validate_physical_device_ids(&[7, 7], false).unwrap_err();
        assert!(error.contains("duplicate"), "got: {error}");
    }

    #[test]
    fn collective_rows_cover_ep_schedules_but_not_single() {
        let (plan, _) = seal_ep(8, 2, ExpertAssign::Stride, &[7, 2]);
        let rows = plan.collective_rows();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert!(
                matches!(
                    row.hint,
                    crate::device_mesh::CollectiveHint::AllReduce { kind: DimKind::Ep }
                ),
                "unexpected collective row {row:?}"
            );
        }
        let names = rows.iter().map(|row| row.name.as_str()).collect::<Vec<_>>();
        assert_eq!(names, vec!["gate", "up", "down"]);
        let (s_manifest, s_plan, s_specs, s_sources, s_mesh) = single_mesh_fixture(8);
        let single =
            plan_expert_execution(&s_manifest, &s_plan, &s_specs, &s_sources, &s_mesh, &[7])
                .unwrap()
                .pop()
                .unwrap();
        assert!(single.collective_rows().is_empty());
    }

    #[test]
    fn repartition_single_to_ep_matches_a_fresh_ep_plan() {
        for (ranks, assign) in [
            (2, ExpertAssign::Stride),
            (2, ExpertAssign::Contiguous),
            (4, ExpertAssign::Stride),
            (4, ExpertAssign::Contiguous),
        ] {
            let (s_manifest, s_plan, s_specs, sources, s_mesh) = single_mesh_fixture(8);
            let single =
                plan_expert_execution(&s_manifest, &s_plan, &s_specs, &sources, &s_mesh, &[7])
                    .unwrap()
                    .pop()
                    .unwrap();
            assert_eq!(single.parallelism(), ExpertParallelism::Single);
            let (e_manifest, e_plan, e_specs, _, e_mesh) =
                ep_mesh_fixture(8, ranks, assign, "indexed-decode-routed-partial");
            let physical = vec![7, 2, 11, 5][..ranks].to_vec();
            let fresh =
                plan_expert_execution(&e_manifest, &e_plan, &e_specs, &sources, &e_mesh, &physical)
                    .unwrap()
                    .pop()
                    .unwrap();
            let repart = repartition_expert_execution_plan(
                &single,
                &e_mesh,
                &physical,
                assign,
                "indexed-decode-routed-partial",
            )
            .unwrap();
            assert_eq!(fresh, repart, "ranks={ranks} assign={assign:?}");
            assert_eq!(repart.execution(), "indexed-decode-routed-partial");
            for rank in 0..ranks {
                let (_, cache) = adapt_expert_execution_plan(&repart, rank).unwrap();
                assert_eq!(cache.physical_device(), physical[rank]);
            }
        }
        // Only sealed Single plans enter the transition.
        let (plan, mesh) = seal_ep(8, 2, ExpertAssign::Stride, &[7, 2]);
        let error =
            repartition_expert_execution_plan(&plan, &mesh, &[7, 2], ExpertAssign::Stride, "x")
                .unwrap_err();
        assert!(error.contains("Single"), "got: {error}");
    }

    #[test]
    fn execution_fingerprint_is_deterministic_and_topology_sensitive() {
        let (plan, _) = seal_ep(8, 2, ExpertAssign::Stride, &[7, 2]);
        assert_eq!(plan.execution_fingerprint(), plan.execution_fingerprint());
        let baseline = plan.execution_fingerprint();
        assert!(baseline.contains("sealed-ep/v1"));
        let mut bad = plan.clone();
        bad.physical_devices[0] = 999;
        assert_ne!(bad.execution_fingerprint(), baseline);
        let mut bad = plan.clone();
        bad.experts[0].owner_rank = 1;
        assert_ne!(bad.execution_fingerprint(), baseline);
        let mut bad = plan.clone();
        bad.execution = "indexed-decode-slot-order".into();
        assert_ne!(bad.execution_fingerprint(), baseline);
        let stale = execution_contract_for_plan(&bad).unwrap();
        assert!(
            !stale.is_root_routed_ep(),
            "cutover negative: old indexed-decode-slot-order must not authorize root-routed EP"
        );
        let mut bad = plan.clone();
        bad.execution = "other-execution".into();
        assert_ne!(bad.execution_fingerprint(), baseline);
        // A twin plan sealed on a fresh same-shape mesh carries a fresh
        // epoch, so its fingerprint differs even though ownership matches.
        let (twin, _) = seal_ep(8, 2, ExpertAssign::Stride, &[7, 2]);
        assert_ne!(twin.execution_fingerprint(), baseline);
        assert_eq!(
            twin.rank_ownership(),
            plan.rank_ownership(),
            "twin ownership must still match"
        );
    }

    /// Legacy per-parity partial-sum association. Diagnostic only: this is
    /// the association a sharded EP reduction used before the canonical
    /// slot-order fold. It must never return to production; it exists here
    /// solely to prove the adversarial test below is non-vacuous.
    fn legacy_even_odd_partial_output(
        plan: &ExpertExecutionPlan,
        ranks: &[CpuRankProgram],
        input: &[f32],
        residual: &[f32],
        routes: &[(usize, f32)],
    ) -> Result<Vec<f32>, String> {
        validate_cpu_programs(plan, ranks, input, residual, routes)?;
        let mut even = vec![0.0f32; plan.dimensions.hidden];
        let mut odd = vec![0.0f32; plan.dimensions.hidden];
        for &(expert, weight) in routes {
            let (owner, matrices) = cpu_owner_matrices(plan, ranks, expert)?;
            let row = cpu_expert_row(matrices, input)?;
            let partial = if owner % 2 == 0 { &mut even } else { &mut odd };
            for (slot, value) in partial.iter_mut().zip(row.iter()) {
                *slot += weight * value;
            }
        }
        let mut output = residual.to_vec();
        for (out, value) in output.iter_mut().zip(even.iter()) {
            *out += *value;
        }
        for (out, value) in output.iter_mut().zip(odd.iter()) {
            *out += *value;
        }
        Ok(output)
    }

    fn cancel_programs(plan: &ExpertExecutionPlan) -> Vec<CpuRankProgram> {
        const SCALES: [f32; 8] = [16777216.0, 1.0, -16777216.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        plan.rank_ownership()
            .iter()
            .map(|ownership| CpuRankProgram {
                logical_rank: ownership.logical_rank,
                physical_device: ownership.physical_device,
                experts: ownership
                    .global_expert_ids
                    .iter()
                    .map(|&expert| {
                        let scale = SCALES[expert];
                        let diagonal = vec![vec![scale, 0.0], vec![0.0, scale]];
                        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
                        (
                            expert,
                            CpuExpertMatrices {
                                gate: diagonal,
                                up: identity.clone(),
                                down: identity,
                            },
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn canonical_slot_fold_matches_single_for_ep2_and_ep4_while_legacy_differs() {
        // Slot terms: 2^24, 1, -(2^24), 1, 0, 0, 0, 0. The flat slot-order
        // fold keeps 1.0 (2^24 + 1 rounds back to 2^24 in f32); the
        // even/odd partial association folds 2.0 instead.
        let routes: Vec<(usize, f32)> = (0..8).map(|expert| (expert, 1.0)).collect();
        let (s_manifest, s_plan, s_specs, sources, s_mesh) = single_mesh_fixture(8);
        let single = plan_expert_execution(&s_manifest, &s_plan, &s_specs, &sources, &s_mesh, &[7])
            .unwrap()
            .pop()
            .unwrap();
        let single_ranks = cancel_programs(&single);
        let mut launches = Vec::new();
        let single_out = execute_cpu_mesh(
            &single,
            &single_ranks,
            &[1.0, 0.0],
            &[0.0, 0.0],
            &routes,
            &mut launches,
        )
        .unwrap();
        assert_eq!(single_out.output, vec![1.0, 0.0]);
        assert_eq!(single_out.collective_count, 1);
        for (ranks, assign) in [(2, ExpertAssign::Stride), (4, ExpertAssign::Stride)] {
            let physical = vec![7, 2, 11, 5][..ranks].to_vec();
            let (plan, _) = seal_ep(8, ranks, assign, &physical);
            let ep_ranks = cancel_programs(&plan);
            let mut launches = Vec::new();
            let ep_out = execute_cpu_mesh(
                &plan,
                &ep_ranks,
                &[1.0, 0.0],
                &[0.0, 0.0],
                &routes,
                &mut launches,
            )
            .unwrap();
            assert_eq!(launches.len(), 8);
            assert_eq!(
                ep_out.output, single_out.output,
                "canonical EP{ranks} must equal Single exactly"
            );
            let legacy =
                legacy_even_odd_partial_output(&plan, &ep_ranks, &[1.0, 0.0], &[0.0, 0.0], &routes)
                    .unwrap();
            assert_eq!(legacy, vec![2.0, 0.0]);
            let gap = (legacy[0] - single_out.output[0]).abs();
            assert!(
                gap > 0.0,
                "adversarial association must differ, got gap {gap}"
            );
        }
    }
}
