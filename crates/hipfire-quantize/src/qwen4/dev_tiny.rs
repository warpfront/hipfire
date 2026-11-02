// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer-only COMPLETE tiny Qwen4 producer (`--qwen4-dev-tiny`).
//!
//! Compiled only with the `qwen4-dev-profiles` feature.  It converts a local
//! safetensors directory whose `config.json` carries an explicit
//! `hipfire_profile` developer marker into one HFQM artifact, planning every
//! tensor, extent and quant tag from the typed S1 manifest
//! ([`Qwen4Manifest::build_with_recipe`]) — never from Flash-Next counts.
//! Two encodings:
//!
//! * `source-exact` ([`Qwen4TierRecipe::SyntheticSourceBf16`]): every tensor
//!   keeps its BF16 source bytes, the PLE tables included.
//! * `packed` ([`Qwen4TierRecipe::SyntheticPacked`]): each role takes the
//!   Flash-Next format its reduction width tiles (see
//!   [`synthetic_packed_decision`]) and stays BF16 otherwise.  A profile in
//!   which no role can pack is refused with the unsupported roles named; a
//!   source-exact artifact is never called packed.
//!
//! The transaction is the Flash-Next one: the full index is planned from
//! safetensors headers, a temporary file is written with bounded row chunks,
//! fsynced, re-checked against the source identity, reopened and compared with
//! the plan, admitted through the runtime's own `admit_hfqm_artifact` (which
//! must accept exactly the planned tags and extents), and only then published
//! without replacing an existing file.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use hipfire_arch_qwen4::artifact::admit_hfqm_artifact;
use hipfire_arch_qwen4::config::Qwen4Config;
use hipfire_arch_qwen4::weights::{
    synthetic_packed_decision, Qwen4Manifest, Qwen4TierRecipe, RECIPE_METADATA_KEY,
};
use hipfire_runtime::hfq::{HfqFile, HfqStreamEntry};
use hipfire_runtime::weight_manifest::ShardPolicy;
use rdna_compute::DType;
use serde_json::{json, Map, Value};

use super::{
    capacity_preflight, checked_bf16_bytes, checked_product, i64_role, is_vision_tensor,
    load_inventory, load_optional_config, load_tokenizer_metadata, matrix_quant_type,
    predicted_output_bytes, publish_qwen4, quantized_data_len_for_dtype, qwen4_quantized_dtype,
    read_i64_array, shape_u32, source_paths, stream_entry, sync_file, temporary_output_path,
    validate_row_chunk, EntryKind, I64Role, PlannedEntry, PleMetadata, Qwen4Error, Qwen4ReopenPlan,
    Qwen4Summary, ScratchTracker, SourceTensor, TokenizerMetadata, DEFAULT_ROW_CHUNK,
    MAX_CHUNK_BYTES, QWEN4_ARCH_ID, QWEN4_I64_QUANT_TYPE, QWEN4_PLE_VERSION,
};
use crate::quant_fwht::gen_fwht_signs;

/// Version of the developer `qwen4_recipe` metadata object.  Flash-Next writes
/// version 2 (no `id`); a developer artifact names its recipe by `id`.
const DEV_RECIPE_VERSION: u64 = 3;

/// How a developer tiny checkpoint is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4DevTinyEncoding {
    /// Every tensor keeps its BF16 source bytes.
    SourceExact,
    /// Each role is packed at the Flash-Next format its width tiles.
    Packed,
}

impl Qwen4DevTinyEncoding {
    fn recipe(self) -> Qwen4TierRecipe {
        match self {
            Self::SourceExact => Qwen4TierRecipe::SyntheticSourceBf16,
            Self::Packed => Qwen4TierRecipe::SyntheticPacked,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::SourceExact => "source-exact",
            Self::Packed => "packed",
        }
    }
}

/// One line of the role report: how many tensors of a role class were written
/// at a format, and why a BF16 role was not packed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DevTinyRoleLine {
    pub(crate) class: &'static str,
    pub(crate) format: String,
    pub(crate) count: usize,
    /// Empty when the role is packed.
    pub(crate) reason: &'static str,
}

/// What a completed developer transaction wrote.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct DevTinyReceipt {
    pub(crate) encoding: Qwen4DevTinyEncoding,
    pub(crate) recipe_id: &'static str,
    pub(crate) profile_id: String,
    pub(crate) ffn_kind: &'static str,
    pub(crate) summary: Qwen4Summary,
    /// Entries written in a packed (quantized) format.
    pub(crate) packed_entries: usize,
    /// Entries written as their BF16 source bytes.
    pub(crate) bf16_entries: usize,
    /// Length of the published file; always the predicted length.
    pub(crate) actual_output_bytes: u64,
    pub(crate) roles: Vec<DevTinyRoleLine>,
}

/// CLI entry: one developer tiny artifact; a role report goes to stderr.
pub(crate) fn run_dev_tiny_cli(
    input: &Path,
    output: &Path,
    encoding: Qwen4DevTinyEncoding,
) -> Result<Qwen4Summary, Qwen4Error> {
    let receipt = write_dev_tiny_artifact(input, output, encoding)?;
    eprintln!(
        "qwen4 dev-tiny: profile={} ffn={} recipe={} packed_entries={} bf16_entries={} \
         bytes={}",
        receipt.profile_id,
        receipt.ffn_kind,
        receipt.recipe_id,
        receipt.packed_entries,
        receipt.bf16_entries,
        receipt.actual_output_bytes
    );
    for line in &receipt.roles {
        if line.reason.is_empty() {
            eprintln!("  {} x{}: {}", line.class, line.count, line.format);
        } else {
            eprintln!(
                "  {} x{}: {} ({})",
                line.class, line.count, line.format, line.reason
            );
        }
    }
    Ok(receipt.summary)
}

/// The planned, not yet written, artifact.
struct DevPlan {
    entries: Vec<PlannedEntry>,
    ple_metadata: Option<PleMetadata>,
    padded_rows: u64,
    resident_entries: usize,
    expert_entries: usize,
    resident_bytes: u64,
    external_ple_bytes: u64,
    ple_shards: usize,
    roles: Vec<DevTinyRoleLine>,
}

struct PreparedDev {
    config: Qwen4Config,
    manifest: Qwen4Manifest,
    plan: DevPlan,
    stream_entries: Vec<HfqStreamEntry>,
    metadata_json: String,
    predicted: u64,
    recipe_id: &'static str,
}

fn validate_bf16_source(source: &SourceTensor, expected: &[u64]) -> Result<(), Qwen4Error> {
    if source.dtype != "BF16" {
        return Err(Qwen4Error::Invalid(format!(
            "{} has dtype {}, expected source BF16",
            source.name, source.dtype
        )));
    }
    if source.shape != expected {
        return Err(Qwen4Error::Invalid(format!(
            "{} has shape {:?}, expected manifest shape {:?}",
            source.name, source.shape, expected
        )));
    }
    let expected_bytes = checked_bf16_bytes(&source.shape, &source.name)?;
    if source.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            source.name,
            source.data_len(),
            expected_bytes
        )));
    }
    Ok(())
}

/// One manifest weight → its output entry, at the manifest's declared dtype.
/// The dtype is never re-derived here: a planner that second-guessed it would
/// write a geometry the admission would not read.
fn plan_weight(
    dtype: DType,
    source: SourceTensor,
    shape: &[u64],
) -> Result<PlannedEntry, Qwen4Error> {
    let name = source.name.clone();
    let wire_shape = shape_u32(shape, &name)?;
    if dtype == DType::BF16 {
        return Ok(PlannedEntry {
            name: name.clone(),
            quant_type: 16,
            shape: wire_shape,
            group_size: 0,
            data_len: checked_bf16_bytes(shape, &name)?,
            kind: EntryKind::Bf16,
            source,
        });
    }
    if !qwen4_quantized_dtype(dtype) {
        return Err(Qwen4Error::Invalid(format!(
            "manifest declares {name} as {dtype:?}, which no developer plan role can carry"
        )));
    }
    if shape.len() < 2 {
        return Err(Qwen4Error::Invalid(format!(
            "manifest declares {name} as {dtype:?} but its shape {shape:?} is not a matrix"
        )));
    }
    let rows = checked_product(&shape[..shape.len() - 1], &name)?;
    let k = shape[shape.len() - 1];
    let (quant_type, group_size) = matrix_quant_type(dtype)?;
    let data_len = quantized_data_len_for_dtype(dtype, rows, k)?;
    Ok(PlannedEntry {
        name,
        quant_type,
        shape: wire_shape,
        group_size,
        data_len,
        kind: EntryKind::Quant(dtype),
        source,
    })
}

/// Plan every output entry from the manifest and the source headers.  Names,
/// shapes, dtypes, extents and the presence of PLE/MTP all come from
/// `manifest`; the source must match it exactly (positively identified vision
/// tensors are the only records outside it, and are not written).
fn plan_dev_entries(
    tensors: Vec<SourceTensor>,
    manifest: &Qwen4Manifest,
    encoding: Qwen4DevTinyEncoding,
) -> Result<DevPlan, Qwen4Error> {
    let profile = manifest.identity.profile_id.as_str();
    let mut sources = BTreeMap::<String, SourceTensor>::new();
    for tensor in tensors {
        let name = tensor.name.clone();
        if sources.insert(name.clone(), tensor).is_some() {
            return Err(Qwen4Error::Invalid(format!(
                "duplicate source tensor {name} in the {profile} developer checkpoint"
            )));
        }
    }
    let mut declared = BTreeSet::<&str>::new();
    for entry in &manifest.weights {
        declared.insert(entry.name.as_str());
    }
    for record in &manifest.metadata {
        declared.insert(record.name.as_str());
    }
    for name in sources.keys() {
        if !declared.contains(name.as_str()) && !is_vision_tensor(name) {
            return Err(Qwen4Error::Invalid(format!(
                "source tensor {name} is not declared by the {profile} profile manifest"
            )));
        }
    }

    let mut resident = Vec::new();
    let mut ple_entries = Vec::new();
    let mut missing = Vec::<String>::new();
    let mut role_counts = BTreeMap::<(&'static str, String, &'static str), usize>::new();
    for entry in &manifest.weights {
        let tied = matches!(entry.policy, ShardPolicy::Tied { .. });
        let Some(source) = sources.remove(&entry.name) else {
            // A tied alias is a logical name, not a source record.
            if !tied {
                missing.push(entry.name.clone());
            }
            continue;
        };
        let shape = entry
            .logical_shape
            .iter()
            .map(|&dimension| {
                u64::try_from(dimension).map_err(|_| {
                    Qwen4Error::Invalid(format!(
                        "manifest shape of {} does not fit u64",
                        entry.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_bf16_source(&source, &shape)?;
        if tied {
            continue;
        }
        let decision = synthetic_packed_decision(&entry.name, &entry.logical_shape);
        let reason = if entry.dtype != DType::BF16 {
            ""
        } else if encoding == Qwen4DevTinyEncoding::SourceExact {
            "source-exact encoding"
        } else {
            decision.reason
        };
        *role_counts
            .entry((decision.class, format!("{:?}", entry.dtype), reason))
            .or_default() += 1;
        let planned = plan_weight(entry.dtype, source, &shape)?;
        if entry.residency.is_external() {
            ple_entries.push(planned);
        } else {
            resident.push(planned);
        }
    }

    let mut metadata_entries = Vec::new();
    let mut ple_arrays = BTreeMap::<I64Role, Vec<i64>>::new();
    for record in &manifest.metadata {
        let Some(source) = sources.remove(&record.name) else {
            missing.push(record.name.clone());
            continue;
        };
        let role = i64_role(&record.name).ok_or_else(|| {
            Qwen4Error::Invalid(format!(
                "manifest metadata record {} is not a PLE I64 record",
                record.name
            ))
        })?;
        if record.shape.len() != 1 {
            return Err(Qwen4Error::Invalid(format!(
                "manifest metadata record {} must be rank 1, got {:?}",
                record.name, record.shape
            )));
        }
        let len = record.shape[0];
        let values = read_i64_array(&source, len)?;
        ple_arrays.insert(role, values);
        let shape = vec![len as u64];
        metadata_entries.push(PlannedEntry {
            name: record.name.clone(),
            quant_type: QWEN4_I64_QUANT_TYPE,
            shape: shape_u32(&shape, &record.name)?,
            group_size: 0,
            data_len: (len as u64).checked_mul(8).ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} I64 length overflows", record.name))
            })?,
            kind: EntryKind::I64(role),
            source,
        });
    }
    if !missing.is_empty() {
        let shown: Vec<&String> = missing.iter().take(8).collect();
        return Err(Qwen4Error::Invalid(format!(
            "source is missing {} tensors declared by the {profile} profile manifest \
             (first {}: {shown:?})",
            missing.len(),
            shown.len()
        )));
    }

    let ple_metadata = if manifest.metadata.is_empty() {
        None
    } else {
        let mut take = |role: I64Role| {
            ple_arrays
                .remove(&role)
                .ok_or_else(|| Qwen4Error::Invalid(format!("PLE I64 record {role:?} is missing")))
        };
        Some(PleMetadata {
            multipliers: take(I64Role::Multipliers)?,
            vocab_sizes: take(I64Role::VocabSizes)?,
            prefix_offsets: take(I64Role::Offsets)?,
        })
    };
    let padded_rows = ple_entries.iter().try_fold(0u64, |sum, entry| {
        let rows =
            entry.source.shape.first().copied().ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} PLE shard has no rows", entry.name))
            })?;
        sum.checked_add(rows)
            .ok_or_else(|| Qwen4Error::Invalid("PLE physical row count overflows".to_string()))
    })?;
    if ple_metadata.is_some() != !ple_entries.is_empty() {
        return Err(Qwen4Error::Invalid(
            "the profile manifest declares PLE metadata and PLE shards inconsistently".to_string(),
        ));
    }

    let sum_bytes = |entries: &[PlannedEntry], what: &str| {
        entries.iter().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.data_len)
                .ok_or_else(|| Qwen4Error::Invalid(format!("{what} byte count overflows")))
        })
    };
    let resident_bytes = sum_bytes(&resident, "resident")?;
    let external_ple_bytes = sum_bytes(&ple_entries, "PLE")?;
    let expert_entries = resident
        .iter()
        .filter(|entry| {
            entry.name.ends_with(".experts.gate_up_proj")
                || entry.name.ends_with(".experts.down_proj")
        })
        .count();
    let resident_entries = resident.len();
    let ple_shards = ple_entries.len();
    let mut entries = ple_entries;
    entries.extend(resident);
    entries.extend(metadata_entries);
    let roles = role_counts
        .into_iter()
        .map(|((class, format, reason), count)| DevTinyRoleLine {
            class,
            format,
            count,
            reason,
        })
        .collect();
    Ok(DevPlan {
        entries,
        ple_metadata,
        padded_rows,
        resident_entries,
        expert_entries,
        resident_bytes,
        external_ple_bytes,
        ple_shards,
        roles,
    })
}

/// The refusal of a `packed` request for a profile in which no role can pack.
fn no_packable_role_error(profile: &str, roles: &[DevTinyRoleLine]) -> Qwen4Error {
    let mut unsupported = Vec::new();
    let mut kept = 0usize;
    for line in roles {
        if line.reason.starts_with("the Flash-Next recipe keeps") {
            kept += line.count;
        } else {
            unsupported.push(format!("{} x{}: {}", line.class, line.count, line.reason));
        }
    }
    Qwen4Error::Invalid(format!(
        "--qwen4-dev-tiny packed: profile {profile} has no role that can be packed; each \
         candidate role's reduction width does not tile the Flash-Next format it would take \
         and no packed artifact is written (unsupported roles: [{}]; {kept} further roles are \
         source-exact in every recipe). Use source-exact for this profile.",
        unsupported.join("; ")
    ))
}

fn build_dev_metadata(
    config_value: &Value,
    manifest: &Qwen4Manifest,
    plan: &DevPlan,
    encoding: Qwen4DevTinyEncoding,
    recipe_id: &'static str,
    tokenizer: &TokenizerMetadata,
    row_chunk: usize,
) -> Result<String, Qwen4Error> {
    let role_lines: Vec<Value> = plan
        .roles
        .iter()
        .map(|line| {
            let mut object = json!({
                "class": line.class,
                "format": line.format,
                "count": line.count,
            });
            if !line.reason.is_empty() {
                object["reason"] = json!(line.reason);
            }
            object
        })
        .collect();
    let mut root = Map::new();
    root.insert("format".to_string(), Value::String("hfqm".to_string()));
    root.insert("format_version".to_string(), json!(1));
    root.insert("arch_id".to_string(), json!(QWEN4_ARCH_ID));
    root.insert(
        "model_type".to_string(),
        Value::String("qwen4_exp".to_string()),
    );
    root.insert("config".to_string(), config_value.clone());
    root.insert(
        RECIPE_METADATA_KEY.to_string(),
        json!({
            "version": DEV_RECIPE_VERSION,
            "id": recipe_id,
            "developer_only": true,
            "encoding": encoding.label(),
            "profile_id": manifest.identity.profile_id,
            "equation_source": manifest.identity.equation_source,
            "ffn_kind": manifest.identity.ffn_kind,
            "roles": role_lines,
        }),
    );
    if let Some(ple) = &plan.ple_metadata {
        root.insert(
            "qwen4_ple".to_string(),
            json!({
                "version": QWEN4_PLE_VERSION,
                "multipliers": ple.multipliers.clone(),
                "head_vocab_sizes": ple.vocab_sizes.clone(),
                "head_offsets": ple.prefix_offsets.clone(),
                "padded_rows": plan.padded_rows,
            }),
        );
    }
    root.insert(
        "qwen4_streaming".to_string(),
        json!({
            "version": 1,
            "row_chunk": row_chunk,
            "resident_entries": plan.resident_entries,
            "expert_entries": plan.expert_entries,
            "external_ple_entries": plan.ple_shards,
        }),
    );
    root.insert(
        "tokenizer".to_string(),
        Value::String(tokenizer.tokenizer.clone()),
    );
    if let Some(config) = &tokenizer.tokenizer_config {
        root.insert("tokenizer_config".to_string(), config.clone());
    }
    if let Some(config) = &tokenizer.generation_config {
        root.insert("generation_config".to_string(), config.clone());
    }
    serde_json::to_string(&Value::Object(root))
        .map_err(|error| Qwen4Error::json("serialize developer Qwen4 metadata", error))
}

/// Parse and plan one developer artifact: typed profile → manifest → entries →
/// metadata → predicted length.  Reads nothing but the three small I64 PLE
/// records, so it is also the header-only dry plan.
fn prepare_dev_artifact(
    config_value: &Value,
    tokenizer: &TokenizerMetadata,
    tensors: Vec<SourceTensor>,
    encoding: Qwen4DevTinyEncoding,
    row_chunk: usize,
) -> Result<PreparedDev, Qwen4Error> {
    let config = Qwen4Config::from_developer_value(config_value).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 developer config is invalid: {error}"))
    })?;
    if !config.profile.is_synthetic() {
        return Err(Qwen4Error::Invalid(
            "--qwen4-dev-tiny needs a config.json carrying an explicit `hipfire_profile` \
             developer marker; the pinned Flash-Next profile is produced by --qwen4-flash-next"
                .to_string(),
        ));
    }
    let recipe = encoding.recipe();
    let recipe_id = recipe.synthetic_id().ok_or_else(|| {
        Qwen4Error::Invalid("a developer encoding must name a synthetic recipe".to_string())
    })?;
    let manifest = Qwen4Manifest::build_with_recipe(&config, recipe).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 manifest could not be built: {error}"))
    })?;
    let plan = plan_dev_entries(tensors, &manifest, encoding)?;
    if encoding == Qwen4DevTinyEncoding::Packed
        && !plan
            .entries
            .iter()
            .any(|entry| matches!(entry.kind, EntryKind::Quant(_)))
    {
        return Err(no_packable_role_error(
            manifest.identity.profile_id.as_str(),
            &plan.roles,
        ));
    }
    if plan.entries.len() > u32::MAX as usize {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 developer artifact has too many entries: {}",
            plan.entries.len()
        )));
    }
    let metadata_json = build_dev_metadata(
        config_value,
        &manifest,
        &plan,
        encoding,
        recipe_id,
        tokenizer,
        row_chunk,
    )?;
    let stream_entries: Vec<HfqStreamEntry> = plan
        .entries
        .iter()
        .map(|entry| HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();
    let predicted = predicted_output_bytes(&metadata_json, &stream_entries)?;
    Ok(PreparedDev {
        config,
        manifest,
        plan,
        stream_entries,
        metadata_json,
        predicted,
        recipe_id,
    })
}

/// The row chunk one entry streams with.  BF16 and I64 rows are copied raw and
/// the parent's raw-row path bounds `row_chunk x row_bytes`; a wide row (a
/// stacked routed-expert block flattens `[experts, rows, K]` into one row per
/// expert) therefore needs a smaller chunk than the default.  Quantized rows
/// bound themselves against the scratch ceiling.
fn entry_row_chunk(entry: &PlannedEntry, row_chunk: usize) -> Result<usize, Qwen4Error> {
    let element_bytes = match entry.kind {
        EntryKind::Bf16 => 2u64,
        EntryKind::I64(_) => 8u64,
        _ => return Ok(row_chunk),
    };
    let shape = &entry.source.shape;
    let width = if shape.len() <= 1 {
        checked_product(shape, &entry.name)?
    } else {
        checked_product(&shape[1..], &entry.name)?
    };
    let row_bytes = width
        .checked_mul(element_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row bytes overflow", entry.name)))?;
    let fit = (MAX_CHUNK_BYTES / row_bytes.max(1)).max(1);
    Ok(row_chunk.min(usize::try_from(fit).unwrap_or(usize::MAX)))
}

fn remove_on_error<T>(temporary: &Path, result: Result<T, Qwen4Error>) -> Result<T, Qwen4Error> {
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

/// Admit the written temporary file through the runtime's own admission and
/// require it to be exactly the planned artifact: same profile, same manifest
/// (recipe included), same tensor count, and per tensor the planned name, tag,
/// shape, group and byte extent.
fn admit_written(temporary: &Path, prepared: &PreparedDev) -> Result<(), Qwen4Error> {
    let hfq = HfqFile::open(temporary).map_err(|error| {
        Qwen4Error::io(
            format!("reopen {} for runtime admission", temporary.display()),
            error,
        )
    })?;
    let admitted = admit_hfqm_artifact(&hfq).map_err(|error| {
        Qwen4Error::Invalid(format!(
            "runtime admission refused the written developer artifact: {error}"
        ))
    })?;
    if admitted.config != prepared.config {
        return Err(Qwen4Error::Invalid(
            "admitted developer config differs from the planned config".to_string(),
        ));
    }
    if admitted.manifest != prepared.manifest {
        return Err(Qwen4Error::Invalid(
            "admitted developer manifest (recipe included) differs from the planned manifest"
                .to_string(),
        ));
    }
    let planned = &prepared.plan.entries;
    if admitted.source_tensor_count != planned.len() || hfq.tensors().len() != planned.len() {
        return Err(Qwen4Error::Invalid(format!(
            "admitted source inventory has {} tensors, the plan wrote {}",
            admitted.source_tensor_count,
            planned.len()
        )));
    }
    for (info, entry) in hfq.tensors().iter().zip(planned) {
        if info.name != entry.name
            || info.quant_type != entry.quant_type
            || info.shape != entry.shape
            || info.group_size != entry.group_size
            || info.data_size as u64 != entry.data_len
        {
            return Err(Qwen4Error::Invalid(format!(
                "admitted tensor {} (qt{} shape {:?} group {} extent {}) differs from the plan \
                 (qt{} shape {:?} group {} extent {})",
                info.name,
                info.quant_type,
                info.shape,
                info.group_size,
                info.data_size,
                entry.quant_type,
                entry.shape,
                entry.group_size,
                entry.data_len
            )));
        }
    }
    Ok(())
}

/// Build one developer tiny HFQM artifact transactionally.
///
/// `input` is a local safetensors directory (never `hf://`) whose
/// `config.json` carries the explicit `hipfire_profile` marker.  An existing
/// `output` is never replaced; any failure removes the temporary file.
pub(crate) fn write_dev_tiny_artifact(
    input: &Path,
    output: &Path,
    encoding: Qwen4DevTinyEncoding,
) -> Result<DevTinyReceipt, Qwen4Error> {
    let row_chunk = DEFAULT_ROW_CHUNK;
    validate_row_chunk(row_chunk)?;
    if input.to_str().is_some_and(|raw| raw.starts_with("hf://")) {
        return Err(Qwen4Error::Invalid(format!(
            "--qwen4-dev-tiny needs a local safetensors directory, not {}",
            input.display()
        )));
    }
    let input_metadata =
        fs::metadata(input).map_err(|error| Qwen4Error::io(input.display().to_string(), error))?;
    if !input_metadata.is_dir() {
        return Err(Qwen4Error::Invalid(format!(
            "--qwen4-dev-tiny needs a local safetensors directory, got {}",
            input.display()
        )));
    }
    if output.exists() {
        return Err(Qwen4Error::Invalid(format!(
            "refusing to replace existing Qwen4 artifact {}",
            output.display()
        )));
    }
    let source = source_paths(input)?;
    let tensors = load_inventory(&source)?;
    let config_value = load_optional_config(&source)?.ok_or_else(|| {
        Qwen4Error::Invalid("Qwen4 input is missing required config.json".to_string())
    })?;
    let tokenizer = load_tokenizer_metadata(&source)?;
    let prepared = prepare_dev_artifact(&config_value, &tokenizer, tensors, encoding, row_chunk)?;
    capacity_preflight(output, prepared.predicted)?;

    let signs1_256 = gen_fwht_signs(42, 256);
    let signs2_256 = gen_fwht_signs(1042, 256);
    let signs1_128 = gen_fwht_signs(43, 128);
    let signs2_128 = gen_fwht_signs(1043, 128);
    let temporary = temporary_output_path(output)?;
    let mut scratch = ScratchTracker::default();
    let write_result = hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &prepared.metadata_json,
        &prepared.stream_entries,
        |index, writer| {
            let entry = &prepared.plan.entries[index];
            let chunk = entry_row_chunk(entry, row_chunk).map_err(Qwen4Error::into_io)?;
            stream_entry(
                entry,
                chunk,
                &signs1_256,
                &signs2_256,
                &signs1_128,
                &signs2_128,
                &mut scratch,
                writer,
            )
            .map_err(Qwen4Error::into_io)
        },
    );
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(Qwen4Error::io(
            format!("write developer Qwen4 artifact {}", output.display()),
            error,
        ));
    }
    let scratch_high_water_bytes = scratch.high_water;
    remove_on_error(&temporary, sync_file(&temporary))?;
    remove_on_error(&temporary, source.verify_identity())?;
    let actual_output_bytes = remove_on_error(
        &temporary,
        fs::metadata(&temporary)
            .map(|metadata| metadata.len())
            .map_err(|error| Qwen4Error::io(format!("stat {}", temporary.display()), error)),
    )?;
    if actual_output_bytes != prepared.predicted {
        let _ = fs::remove_file(&temporary);
        return Err(Qwen4Error::Invalid(format!(
            "developer Qwen4 artifact is {actual_output_bytes} bytes, predicted {}",
            prepared.predicted
        )));
    }
    let reopened = remove_on_error(&temporary, Qwen4ReopenPlan::open_developer(&temporary))?;
    remove_on_error(
        &temporary,
        reopened.validate_against(&prepared.metadata_json, &prepared.stream_entries),
    )?;
    remove_on_error(&temporary, admit_written(&temporary, &prepared))?;
    remove_on_error(&temporary, publish_qwen4(&temporary, output))?;

    let plan = &prepared.plan;
    let packed_entries = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, EntryKind::Quant(_)))
        .count();
    let bf16_entries = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.kind, EntryKind::Bf16))
        .count();
    Ok(DevTinyReceipt {
        encoding,
        recipe_id: prepared.recipe_id,
        profile_id: prepared.manifest.identity.profile_id.clone(),
        ffn_kind: prepared.manifest.identity.ffn_kind,
        summary: Qwen4Summary {
            entries: plan.entries.len(),
            resident_entries: plan.resident_entries,
            expert_entries: plan.expert_entries,
            ple_shards: plan.ple_shards,
            resident_bytes: plan.resident_bytes,
            external_ple_bytes: plan.external_ple_bytes,
            predicted_output_bytes: prepared.predicted,
            scratch_high_water_bytes,
        },
        packed_entries,
        bf16_entries,
        actual_output_bytes,
        roles: plan.roles.clone(),
    })
}

#[cfg(test)]
mod tests;
