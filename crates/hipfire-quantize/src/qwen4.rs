// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Streaming producer for the native Qwen4/Qwen3.8-Flash-Next artifact.
//!
//! This path is deliberately separate from the legacy model recipes.  It reads
//! safetensors payloads with positional, bounded reads, quantizes complete
//! logical rows, and writes the HFQM index before any large payload is copied.
//! No source tensor (and in particular no stacked expert tensor or PLE shard) is
//! ever collected into one `Vec<u8>`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hipfire_arch_qwen4::config::Qwen4Config;
use hipfire_arch_qwen4::weights::{Qwen4Manifest, ROUTED_DOWN_DTYPE, ROUTED_GATE_UP_DTYPE};
use hipfire_runtime::weight_manifest::ShardPolicy;
use rdna_compute::DType;
use serde_json::{json, Map, Value};

use crate::quant_e8::quantize_mfp4g32_e8_soa_2d;
use crate::quant_fwht::{
    gen_fwht_signs, quantize_mq4g128v2, quantize_mq4g256v2, quantize_mq6g256v2,
};
use crate::quant_q4::quantize_q8f16;

/// SoA row geometry: `16 + ((n_blocks + 15) >> 4) << 4 + n_blocks * 16`, with
/// `n_blocks = k / 32`.  Returned as `(row_stride, extent)`; the padded scale
/// block is why this cannot go through the uniform `(group, bytes)` model.
///
/// The `K % 256` precondition is the encoder's own, not a cosmetic tightening:
/// rows are rotated in 256-wide FWHT segments and the decode GEMV/GEMM assert
/// the same alignment, so an E8 payload whose K only satisfied the 32-weight
/// block size would pass every byte count here and then panic at encode time.
fn mfp4e8soa_row_geometry(rows: u64, k: u64) -> Result<(u64, u64), Qwen4Error> {
    if k == 0 || k % MFP4G32E8SOA_K_ALIGNMENT != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "MFP4G32E8SOA requires a nonzero K multiple of {MFP4G32E8SOA_K_ALIGNMENT}, got K={k}"
        )));
    }
    let n_blocks = k / MFP4G32E8SOA_BLOCK_SIZE;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let row_stride = 16u64
        .checked_add(scale_padded)
        .and_then(|bytes| bytes.checked_add(n_blocks.checked_mul(16)?))
        .ok_or_else(|| Qwen4Error::Invalid("MFP4G32E8SOA row stride overflows".to_string()))?;
    let extent = rows
        .checked_mul(row_stride)
        .ok_or_else(|| Qwen4Error::Invalid("MFP4G32E8SOA extent overflows".to_string()))?;
    Ok((row_stride, extent))
}
use hipfire_quantize::float16::bf16_to_f32;

/// Native Qwen4 architecture ID reserved by the runtime registry.
pub(crate) const QWEN4_ARCH_ID: u32 = 16;

const QWEN4_PLE_VERSION: u32 = 1;
const PLE_SHARD_COUNT: usize = 128;
const PLE_ROWS_PER_SHARD: u64 = 2_500_012;
const PLE_ROW_WIDTH: u64 = 160;
/// Encoded bytes of one external PLE row.  PLE_ROW_WIDTH is a multiple of the
/// Q8F16 block, so five blocks cover a row exactly and a chunk-wide quantization
/// never straddles two rows.
const PLE_ENCODED_ROW_BYTES: u64 = (PLE_ROW_WIDTH / Q8F16_BLOCK_SIZE) * Q8F16_BLOCK_BYTES;
const PLE_HEAD_COUNT: usize = 16;
const PLE_NGRAM_SIZE: u32 = 3;
const ROUTED_EXPERTS: u64 = 512;
const ROUTER_TOP_K: u32 = 10;
const HIDDEN_WIDTH: u64 = 2_560;
const GATE_UP_INTERMEDIATE: u64 = 1_280;
const DOWN_INTERMEDIATE: u64 = 640;
const MQ4G256V2_GROUP_SIZE: u64 = 256;
const MQ4G256V2_GROUP_BYTES: u64 = 136;
const MQ4G128V2_GROUP_SIZE: u64 = 128;
const MQ4G128V2_GROUP_BYTES: u64 = 68;
const MQ4G256V2_QUANT_TYPE: u8 = 44;
const MQ4G128V2_QUANT_TYPE: u8 = 53;
/// MQ6G256V2 (qt=47) shares the aligned-K 256 group with MQ4G256V2 but carries
/// four more bits per weight (200 B/group instead of 136 B/group).
const MQ6G256V2_GROUP_SIZE: u64 = 256;
const MQ6G256V2_GROUP_BYTES: u64 = 200;
const MQ6G256V2_QUANT_TYPE: u8 = 47;
/// Q8F16 (qt=3): f16 scale + 32 int8 per block, 34 bytes per 32 weights.  Not
/// FWHT-rotated, and the tier the MoE fixed classes (embed, lm_head, router)
/// take instead of a four-bit matrix format.
const Q8F16_BLOCK_SIZE: u64 = 32;
const Q8F16_BLOCK_BYTES: u64 = 34;
const Q8F16_QUANT_TYPE: u8 = 3;
/// MFP4G32E8SOA (qt=35): E8 lattice VQ over 32-weight blocks, 4.25 bpw, FWHT-rotated
/// with the same 256-wide basis as the MQ4/MQ6 families.  SoA layout is the only
/// one with both a dense decode GEMV and a dense prefill GEMM on gfx1151.
const MFP4G32E8SOA_QUANT_TYPE: u8 = 35;
const MFP4G32E8SOA_BLOCK_SIZE: u64 = 32;
/// One FWHT-256 rotation segment: the unit the encoder rotates and the decode
/// kernels assert (`assert!(k % 256 == 0)`).
const MFP4G32E8SOA_K_ALIGNMENT: u64 = 256;
/// Raw signed-I64 records are a distinct HFQ type.  TidI32 is not a valid
/// representation for Qwen4's hash metadata.
const QWEN4_I64_QUANT_TYPE: u8 = 52;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REOPEN_REGION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
/// The pinned checkpoint contains 1,658 records: 1,655 BF16 (including the
/// positively excluded vision tower) and three exact I64 metadata arrays.
const PINNED_SOURCE_TENSOR_COUNT: usize = 1_658;
const PINNED_BF16_TENSOR_COUNT: usize = 1_655;
const PINNED_I64_TENSOR_COUNT: usize = 3;
const PINNED_VISION_TENSOR_COUNT: usize = 333;
const PINNED_OUTPUT_ENTRY_COUNT: usize = 1_325;
// Derived from the pinned config plus all 1,658 safetensors headers.  The
// payload is not read to compute this value; it is used as a regression anchor
// for the production preflight's dynamic prediction.
#[cfg(test)]
const PINNED_METADATA_BYTES: u64 = 4_000;
#[cfg(test)]
const PINNED_INDEX_BYTES: u64 = 116_418;
#[cfg(test)]
const PINNED_PAYLOAD_BYTES: u64 = 177_986_811_416;
/// Two tier changes move the pinned checkpoint's stored size away from its
/// source payload: the Q8 trunk tier (3,759,329,280 B below the previously
/// pinned matrices) and the PLE row tier (128 x 2,500,012 x (320 - 170) =
/// 48,000,230,400 B). Both terms are asserted in
/// `pinned_output_prediction_is_exact_before_payload_reads`, so a drift in
/// either tier trips the anchor.
#[cfg(test)]
const PINNED_PREDICTED_OUTPUT_BYTES: u64 = 126_227_374_616;
const PINNED_SHARD_COUNT: usize = 131;
/// Default bounded source row chunk. Quantized matrices derive a smaller
/// physical chunk when raw BF16, decoded F32, and encoded output together
/// approach the 96 MiB aggregate scratch ceiling.
pub(crate) const DEFAULT_ROW_CHUNK: usize = 4_096;
/// External PLE shards are quantized to Q8F16 rows on write; the chunk bounds
/// both the BF16 read and the f32 decode for one bounded pass. A 65,536-row
/// range is a ~20 MiB source request (plus decode and encode scratch) while
/// reducing HTTP request overhead for the 128 large PLE shards.
const EXTERNAL_PLE_ROW_CHUNK: usize = 65_536;
const MAX_CHUNK_BYTES: u64 = 96 * 1024 * 1024;
/// Do not let an accidental CLI value turn a row stream into a tensor buffer.
const MAX_ROW_CHUNK: usize = 4_096;
const CAPACITY_OVERRIDE_ENV: &str = "HIPFIRE_QWEN4_CAPACITY_BYTES";
/// Number of retries after the initial remote range attempt.  Retries are
/// deliberately bounded because every failed range is replayed from byte zero.
const REMOTE_RANGE_MAX_RETRIES: usize = 4;
const REMOTE_RANGE_RETRY_BACKOFF_MS: u64 = 250;
const REMOTE_RANGE_RETRY_BACKOFF_MAX_MS: u64 = 1_000;

fn validate_row_chunk(row_chunk: usize) -> Result<(), Qwen4Error> {
    if row_chunk == 0 || row_chunk > MAX_ROW_CHUNK {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 row chunk must be in 1..={MAX_ROW_CHUNK}, got {row_chunk}"
        )));
    }
    Ok(())
}

fn qwen4_quantized_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2 | DType::MQ4G128V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA | DType::Q8_0
    )
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Qwen4Mode {
    Production,
    CompactFixture,
}

#[derive(Debug)]
pub(crate) struct Qwen4Options<'a> {
    pub(crate) input: &'a Path,
    pub(crate) output: &'a Path,
    pub(crate) row_chunk: usize,
    pub(crate) mode: Qwen4Mode,
}

impl<'a> Qwen4Options<'a> {
    pub(crate) fn new(input: &'a Path, output: &'a Path) -> Self {
        Self {
            input,
            output,
            row_chunk: DEFAULT_ROW_CHUNK,
            mode: Qwen4Mode::Production,
        }
    }

    #[cfg(test)]
    fn compact_fixture(input: &'a Path, output: &'a Path) -> Self {
        Self {
            input,
            output,
            row_chunk: 2,
            mode: Qwen4Mode::CompactFixture,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Qwen4Summary {
    pub(crate) entries: usize,
    pub(crate) resident_entries: usize,
    pub(crate) expert_entries: usize,
    pub(crate) ple_shards: usize,
    pub(crate) resident_bytes: u64,
    pub(crate) external_ple_bytes: u64,
    pub(crate) predicted_output_bytes: u64,
    pub(crate) scratch_high_water_bytes: u64,
}

#[derive(Debug)]
pub(crate) enum Qwen4Error {
    Io {
        context: String,
        source: io::Error,
    },
    Json {
        context: String,
        source: serde_json::Error,
    },
    Invalid(String),
}

impl Qwen4Error {
    fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    fn into_io(self) -> io::Error {
        let message = self.to_string();
        match self {
            Self::Io { source, .. } => source,
            Self::Json { .. } | Self::Invalid(_) => {
                io::Error::new(io::ErrorKind::InvalidData, message)
            }
        }
    }
}

impl std::fmt::Display for Qwen4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Json { context, source } => write!(f, "{context}: {source}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Qwen4Error {}

/// CLI-facing producer entry point.  The existing quantizer routes do not call
/// this function unless `--qwen4-flash-next` is explicitly selected.
pub(crate) fn run_cli(input: &Path, output: &Path) -> Result<Qwen4Summary, Qwen4Error> {
    run_cli_with_mode(input, output, Qwen4Mode::Production)
}

pub(crate) fn run_cli_with_mode(
    input: &Path,
    output: &Path,
    mode: Qwen4Mode,
) -> Result<Qwen4Summary, Qwen4Error> {
    let mut options = Qwen4Options::new(input, output);
    options.mode = mode;
    write_qwen4_artifact(&options)
}

/// Build one transactional Qwen4 HFQM artifact.
///
/// The writer's index is planned entirely from safetensors headers first.  A
/// temporary file is written and reopened with the bounded plan reader before
/// the final rename, so a malformed source or short callback never publishes
/// a partially written candidate over an existing artifact.
pub(crate) fn write_qwen4_artifact(options: &Qwen4Options<'_>) -> Result<Qwen4Summary, Qwen4Error> {
    validate_row_chunk(options.row_chunk)?;
    if options.output.exists() {
        return Err(Qwen4Error::Invalid(format!(
            "refusing to replace existing Qwen4 artifact {}",
            options.output.display()
        )));
    }
    let source = source_paths(options.input)?;
    validate_source_shard_count(&source, options.mode)?;
    let tensors = load_inventory(&source)?;
    validate_source_inventory(&tensors, options.mode)?;
    let config_value = load_optional_config(&source)?.ok_or_else(|| {
        Qwen4Error::Invalid("Qwen4 input is missing required config.json".to_string())
    })?;
    if options.mode == Qwen4Mode::CompactFixture {
        return write_compact_artifact(options, &source, tensors, &config_value);
    }
    let tokenizer = load_tokenizer_metadata(&source)?;
    let config = Qwen4Config::from_value(&config_value)
        .map_err(|error| Qwen4Error::Invalid(format!("Qwen4 config is invalid: {error}")))?;
    let manifest = Qwen4Manifest::build(&config).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 manifest could not be built: {error}"))
    })?;

    let plan = plan_entries(tensors, options.row_chunk, &manifest)?;
    let ple_metadata = plan.ple_metadata.as_ref().ok_or_else(|| {
        Qwen4Error::Invalid("Qwen4 plan did not produce PLE metadata".to_string())
    })?;
    if plan.entries.len() != PINNED_OUTPUT_ENTRY_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 output manifest has {} entries, expected pinned {}",
            plan.entries.len(),
            PINNED_OUTPUT_ENTRY_COUNT
        )));
    }
    let metadata_json = build_metadata(Some(&config_value), &plan, ple_metadata, Some(&tokenizer))?;
    if plan.entries.len() > u32::MAX as usize {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 artifact has too many entries: {}",
            plan.entries.len()
        )));
    }
    let stream_entries: Vec<hipfire_runtime::hfq::HfqStreamEntry> = plan
        .entries
        .iter()
        .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();
    let predicted_output_bytes = predicted_output_bytes(&metadata_json, &stream_entries)?;
    capacity_preflight(options.output, predicted_output_bytes)?;

    let signs1_256 = gen_fwht_signs(42, 256);
    let signs2_256 = gen_fwht_signs(1042, 256);
    let signs1_128 = gen_fwht_signs(43, 128);
    let signs2_128 = gen_fwht_signs(1043, 128);
    let temporary = temporary_output_path(options.output)?;
    let mut scratch = ScratchTracker::default();
    let write_result = hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &metadata_json,
        &stream_entries,
        |index, writer| {
            stream_entry(
                &plan.entries[index],
                options.row_chunk,
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
            format!("write Qwen4 artifact {}", options.output.display()),
            error,
        ));
    }
    if let Err(error) = sync_file(&temporary) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = source.verify_identity() {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }

    // Reopen without mmap'ing the large payload.  The plan reader checks the
    // header/index extents, qtypes, shapes, and exact byte ranges against the
    // precomputed stream plan before retaining only index-sized allocations.
    let reopened = match Qwen4ReopenPlan::open(&temporary) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    if let Err(error) = reopened.validate_against(&metadata_json, &stream_entries) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = publish_qwen4(&temporary, options.output) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }

    Ok(Qwen4Summary {
        entries: plan.entries.len(),
        resident_entries: plan.resident_entries,
        expert_entries: plan.expert_entries,
        ple_shards: plan.ple_shards,
        resident_bytes: plan.resident_bytes,
        external_ple_bytes: plan.external_ple_bytes,
        predicted_output_bytes,
        scratch_high_water_bytes: scratch.high_water,
    })
}

fn write_compact_artifact(
    options: &Qwen4Options<'_>,
    source: &SourceSet,
    tensors: Vec<SourceTensor>,
    config: &Value,
) -> Result<Qwen4Summary, Qwen4Error> {
    let plan = plan_compact_entries(tensors, options.row_chunk)?;
    let ple = plan
        .ple_metadata
        .as_ref()
        .ok_or_else(|| Qwen4Error::Invalid("compact Qwen4 plan has no PLE metadata".to_string()))?;
    let metadata_json = build_metadata(Some(config), &plan, ple, None)?;
    let stream_entries: Vec<hipfire_runtime::hfq::HfqStreamEntry> = plan
        .entries
        .iter()
        .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();
    let predicted_output_bytes = predicted_output_bytes(&metadata_json, &stream_entries)?;
    capacity_preflight(options.output, predicted_output_bytes)?;
    let signs1_256 = gen_fwht_signs(42, 256);
    let signs2_256 = gen_fwht_signs(1042, 256);
    let signs1_128 = gen_fwht_signs(43, 128);
    let signs2_128 = gen_fwht_signs(1043, 128);
    let temporary = temporary_output_path(options.output)?;
    let mut scratch = ScratchTracker::default();
    let write_result = hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &metadata_json,
        &stream_entries,
        |index, writer| {
            stream_entry(
                &plan.entries[index],
                options.row_chunk,
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
            format!("write compact Qwen4 artifact {}", options.output.display()),
            error,
        ));
    }
    if let Err(error) = sync_file(&temporary) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = source.verify_identity() {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let reopened = match Qwen4ReopenPlan::open(&temporary) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    if let Err(error) = reopened.validate_against(&metadata_json, &stream_entries) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = publish_qwen4(&temporary, options.output) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(Qwen4Summary {
        entries: plan.entries.len(),
        resident_entries: plan.resident_entries,
        expert_entries: plan.expert_entries,
        ple_shards: plan.ple_shards,
        resident_bytes: plan.resident_bytes,
        external_ple_bytes: plan.external_ple_bytes,
        predicted_output_bytes,
        scratch_high_water_bytes: scratch.high_water,
    })
}

fn plan_compact_entries(
    tensors: Vec<SourceTensor>,
    row_chunk: usize,
) -> Result<EntryPlan, Qwen4Error> {
    let mut metadata_sources = BTreeMap::<I64Role, SourceTensor>::new();
    let mut ple_sources = BTreeMap::<usize, SourceTensor>::new();
    let mut resident = Vec::new();
    let mut metadata_entries = Vec::new();
    let mut expert_entries = 0usize;
    for tensor in tensors {
        if let Some(role) = i64_role(&tensor.name) {
            if tensor.dtype != "I64" {
                return Err(Qwen4Error::Invalid(format!(
                    "{} compact fixture metadata must be I64",
                    tensor.name
                )));
            }
            let expected = match role {
                I64Role::Multipliers => 3,
                I64Role::VocabSizes | I64Role::Offsets => PLE_HEAD_COUNT,
            };
            if tensor.shape != [expected as u64] || tensor.data_len() != (expected * 8) as u64 {
                return Err(Qwen4Error::Invalid(format!(
                    "{} compact fixture I64 shape/extent is invalid",
                    tensor.name
                )));
            }
            if metadata_sources.insert(role, tensor).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate compact fixture I64 role {role:?}"
                )));
            }
            continue;
        }
        if let Some(index) = ple_shard_index(&tensor.name) {
            if tensor.dtype != "BF16" || tensor.shape.len() != 2 || tensor.shape[1] != PLE_ROW_WIDTH
            {
                return Err(Qwen4Error::Invalid(format!(
                    "{} compact fixture PLE must be BF16 [rows, {PLE_ROW_WIDTH}]",
                    tensor.name
                )));
            }
            let expected = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
            if tensor.data_len() != expected {
                return Err(Qwen4Error::Invalid(format!(
                    "{} compact fixture PLE extent is {}, expected {expected}",
                    tensor.name,
                    tensor.data_len()
                )));
            }
            if ple_sources.insert(index, tensor).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate compact fixture PLE shard index {index}"
                )));
            }
            continue;
        }
        if tensor.dtype != "BF16" {
            return Err(Qwen4Error::Invalid(format!(
                "{} compact fixture source must be BF16 or I64",
                tensor.name
            )));
        }
        let source_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
        if tensor.data_len() != source_len {
            return Err(Qwen4Error::Invalid(format!(
                "{} compact fixture source extent is {}, expected {source_len}",
                tensor.name,
                tensor.data_len()
            )));
        }
        let shape = shape_u32(&tensor.shape, &tensor.name)?;
        let is_gate_up = tensor.name.ends_with(".experts.gate_up_proj") && tensor.shape.len() == 3;
        let is_down = tensor.name.ends_with(".experts.down_proj") && tensor.shape.len() == 3;
        let (quant_type, group_size, data_len, kind) = if is_gate_up || is_down {
            let rows = tensor.shape[0]
                .checked_mul(tensor.shape[1])
                .ok_or_else(|| Qwen4Error::Invalid(format!("{} rows overflow", tensor.name)))?;
            let k = tensor.shape[2];
            // A compact fixture is read back through the same manifest and the
            // same loader as a production artifact, so it must carry the tier
            // the family declares for that role rather than a tier of its own.
            let dtype = if is_gate_up {
                ROUTED_GATE_UP_DTYPE
            } else {
                ROUTED_DOWN_DTYPE
            };
            let source_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
            if tensor.data_len() != source_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{} compact fixture expert extent is {}, expected {source_len}",
                    tensor.name,
                    tensor.data_len()
                )));
            }
            let (quant_type, group_size) = matrix_quant_type(dtype)?;
            (
                quant_type,
                group_size,
                quantized_data_len_for_dtype(dtype, rows, k)?,
                EntryKind::Quant(dtype),
            )
        } else {
            (
                16,
                0,
                checked_bf16_bytes(&tensor.shape, &tensor.name)?,
                EntryKind::Bf16,
            )
        };
        if matches!(kind, EntryKind::Quant(_)) {
            expert_entries += 1;
        }
        resident.push(PlannedEntry {
            name: tensor.name.clone(),
            source: tensor,
            quant_type,
            shape,
            group_size,
            data_len,
            kind,
        });
    }
    let mut ple_metadata = PleMetadata {
        multipliers: vec![3, 5, 7],
        vocab_sizes: vec![127; PLE_HEAD_COUNT],
        prefix_offsets: (0..PLE_HEAD_COUNT)
            .map(|index| (index * 127) as i64)
            .collect(),
    };
    if let Some(source) = metadata_sources.get(&I64Role::Multipliers) {
        ple_metadata.multipliers = read_i64_array(source, 3)?;
    }
    if let Some(source) = metadata_sources.get(&I64Role::VocabSizes) {
        ple_metadata.vocab_sizes = read_i64_array(source, PLE_HEAD_COUNT)?;
    }
    if let Some(source) = metadata_sources.get(&I64Role::Offsets) {
        ple_metadata.prefix_offsets = read_i64_array(source, PLE_HEAD_COUNT)?;
    }
    validate_ple_metadata(
        &ple_metadata.multipliers,
        &ple_metadata.vocab_sizes,
        &ple_metadata.prefix_offsets,
    )?;
    for role in [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets] {
        if let Some(source) = metadata_sources.remove(&role) {
            metadata_entries.push(PlannedEntry {
                name: source.name.clone(),
                shape: shape_u32(&source.shape, &source.name)?,
                quant_type: QWEN4_I64_QUANT_TYPE,
                group_size: 0,
                data_len: source.data_len(),
                kind: EntryKind::I64(role),
                source,
            });
        }
    }
    resident.sort_by(|left, right| left.name.cmp(&right.name));
    if ple_sources.is_empty() {
        return Err(Qwen4Error::Invalid(
            "compact fixture has no PLE shards".to_string(),
        ));
    }
    for (expected, actual) in ple_sources.keys().enumerate() {
        if *actual != expected {
            return Err(Qwen4Error::Invalid(format!(
                "compact fixture PLE shard indexes must be contiguous from zero, found {actual}"
            )));
        }
    }
    let mut ple_entries = Vec::with_capacity(ple_sources.len());
    for (index, source) in ple_sources {
        let _ = index;
        let rows = source.shape.first().copied().ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} compact PLE shard has no rows", source.name))
        })?;
        ple_entries.push(PlannedEntry {
            name: source.name.clone(),
            shape: shape_u32(&source.shape, &source.name)?,
            quant_type: Q8F16_QUANT_TYPE,
            group_size: Q8F16_BLOCK_SIZE as u32,
            data_len: rows.checked_mul(PLE_ENCODED_ROW_BYTES).ok_or_else(|| {
                Qwen4Error::Invalid("compact PLE entry length overflows".to_string())
            })?,
            kind: EntryKind::Ple,
            source,
        });
    }
    let resident_bytes = resident.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.data_len)
            .ok_or_else(|| Qwen4Error::Invalid("compact resident byte count overflows".to_string()))
    })?;
    let external_ple_bytes = ple_entries.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.data_len)
            .ok_or_else(|| Qwen4Error::Invalid("compact PLE byte count overflows".to_string()))
    })?;
    let ple_shards = ple_entries.len();
    let resident_entries = resident.len();
    let mut entries = ple_entries;
    entries.extend(resident);
    entries.extend(metadata_entries);
    Ok(EntryPlan {
        entries,
        ple_metadata: Some(ple_metadata),
        row_chunk,
        resident_entries,
        expert_entries,
        resident_bytes,
        external_ple_bytes,
        ple_shards,
    })
}

fn temporary_output_path(output: &Path) -> Result<PathBuf, Qwen4Error> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("qwen4.hfq");
    for nonce in 0..32u32 {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = parent.join(format!(
            ".{file_name}.qwen4-tmp-{}-{timestamp:x}-{nonce}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => {
                fs::remove_file(&path).map_err(|error| {
                    Qwen4Error::io(
                        format!("reserve temporary Qwen4 artifact {}", path.display()),
                        error,
                    )
                })?;
                return Ok(path);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Qwen4Error::io(
                    format!("reserve temporary Qwen4 artifact {}", path.display()),
                    error,
                ));
            }
        }
    }
    Err(Qwen4Error::Invalid(format!(
        "could not allocate a unique temporary Qwen4 artifact beside {}",
        output.display()
    )))
}

fn sync_file(path: &Path) -> Result<(), Qwen4Error> {
    let file = File::open(path)
        .map_err(|error| Qwen4Error::io(format!("open {} for fsync", path.display()), error))?;
    file.sync_all()
        .map_err(|error| Qwen4Error::io(format!("fsync {}", path.display()), error))
}

fn publish_qwen4(temporary: &Path, output: &Path) -> Result<(), Qwen4Error> {
    if output.exists() {
        return Err(Qwen4Error::Invalid(format!(
            "refusing to replace existing Qwen4 artifact {}",
            output.display()
        )));
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let old = std::ffi::CString::new(temporary.as_os_str().as_bytes()).map_err(|_| {
            Qwen4Error::Invalid(format!(
                "temporary Qwen4 path contains NUL: {}",
                temporary.display()
            ))
        })?;
        let new = std::ffi::CString::new(output.as_os_str().as_bytes()).map_err(|_| {
            Qwen4Error::Invalid(format!(
                "output Qwen4 path contains NUL: {}",
                output.display()
            ))
        })?;
        // SAFETY: both C strings are NUL-terminated paths and AT_FDCWD
        // resolves them relative to the current working directory.
        let result = unsafe {
            libc::syscall(
                libc::SYS_renameat2,
                libc::AT_FDCWD,
                old.as_ptr(),
                libc::AT_FDCWD,
                new.as_ptr(),
                1u32, // RENAME_NOREPLACE
            )
        };
        if result != 0 {
            let error = io::Error::last_os_error();
            if renameat2_needs_hard_link_fallback(&error) {
                publish_qwen4_hard_link(temporary, output)?;
            } else {
                return Err(Qwen4Error::io(
                    format!("publish Qwen4 artifact {}", output.display()),
                    error,
                ));
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        fs::rename(temporary, output).map_err(|error| {
            Qwen4Error::io(
                format!("publish Qwen4 artifact {}", output.display()),
                error,
            )
        })?;
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    #[cfg(unix)]
    {
        let directory = File::open(parent).map_err(|error| {
            Qwen4Error::io(format!("open {} for fsync", parent.display()), error)
        })?;
        directory
            .sync_all()
            .map_err(|error| Qwen4Error::io(format!("fsync {}", parent.display()), error))?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn renameat2_needs_hard_link_fallback(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL) | Some(libc::ENOSYS) | Some(libc::EOPNOTSUPP)
    )
}

#[cfg(target_os = "linux")]
fn publish_qwen4_hard_link(temporary: &Path, output: &Path) -> Result<(), Qwen4Error> {
    // A hard link publishes the complete, fsynced temporary inode atomically
    // without replacing an existing destination.  If the process crashes
    // before unlinking the temporary name, both names remain for cleanup; the
    // final output is already a valid complete artifact.
    fs::hard_link(temporary, output).map_err(|error| {
        Qwen4Error::io(
            format!("publish Qwen4 artifact {} via hard link", output.display()),
            error,
        )
    })?;
    fs::remove_file(temporary).map_err(|error| {
        Qwen4Error::io(
            format!(
                "remove temporary Qwen4 artifact {} after publish",
                temporary.display()
            ),
            error,
        )
    })
}
#[derive(Debug, Default)]
struct ScratchTracker {
    high_water: u64,
}

impl ScratchTracker {
    fn observe(
        &mut self,
        raw_bytes: usize,
        value_bytes: usize,
        output_bytes: usize,
    ) -> Result<(), Qwen4Error> {
        let total = (raw_bytes as u64)
            .checked_add(value_bytes as u64)
            .and_then(|value| value.checked_add(output_bytes as u64))
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 scratch high-water overflows".to_string()))?;
        if total > MAX_CHUNK_BYTES {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 scratch high-water {total} exceeds bounded {MAX_CHUNK_BYTES} bytes"
            )));
        }
        self.high_water = self.high_water.max(total);
        Ok(())
    }
}
fn chunk_scratch_bytes(dtype: DType, rows: u64, k: u64) -> Result<u64, Qwen4Error> {
    let elements = rows
        .checked_mul(k)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 chunk element count overflows".to_string()))?;
    let raw_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 raw chunk bytes overflow".to_string()))?;
    let value_bytes = elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 decoded chunk bytes overflow".to_string()))?;
    let encoded_bytes = quantized_data_len_for_dtype(dtype, rows, k)?;
    raw_bytes
        .checked_add(value_bytes)
        .and_then(|bytes| bytes.checked_add(encoded_bytes))
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 chunk scratch bytes overflow".to_string()))
}

fn bounded_rows_for_scratch(dtype: DType, k: u64, max_rows: u64) -> Result<u64, Qwen4Error> {
    let mut low = 0;
    let mut high = max_rows;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if chunk_scratch_bytes(dtype, mid, k)? <= MAX_CHUNK_BYTES {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

fn predicted_output_bytes(
    metadata_json: &str,
    entries: &[hipfire_runtime::hfq::HfqStreamEntry],
) -> Result<u64, Qwen4Error> {
    let metadata_len = u64::try_from(metadata_json.len())
        .map_err(|_| Qwen4Error::Invalid("Qwen4 metadata length does not fit u64".to_string()))?;
    let mut index_len = 4u64;
    for entry in entries {
        let name_len = u64::try_from(entry.name.len())
            .map_err(|_| Qwen4Error::Invalid("Qwen4 entry name length overflows".to_string()))?;
        let dims = u64::try_from(entry.shape.len())
            .map_err(|_| Qwen4Error::Invalid("Qwen4 shape rank overflows".to_string()))?;
        index_len = index_len
            .checked_add(2)
            .and_then(|value| value.checked_add(name_len))
            .and_then(|value| value.checked_add(2))
            .and_then(|value| value.checked_add(dims.checked_mul(4)?))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 index length overflows".to_string()))?;
    }
    let data_start = 32u64
        .checked_add(metadata_len)
        .and_then(|value| value.checked_add(index_len))
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 output header length overflows".to_string()))?;
    let data_offset = data_start
        .checked_add(4095)
        .map(|value| value & !4095)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 output alignment overflows".to_string()))?;
    let payload = entries.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.data_len)
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 output payload length overflows".to_string()))
    })?;
    data_offset
        .checked_add(payload)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 output length overflows".to_string()))
}

fn capacity_preflight(path: &Path, predicted: u64) -> Result<(), Qwen4Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let available = match env::var(CAPACITY_OVERRIDE_ENV) {
        Ok(value) => Some(value.parse::<u64>().map_err(|_| {
            Qwen4Error::Invalid(format!(
                "{CAPACITY_OVERRIDE_ENV} must be an unsigned byte count, got {value:?}"
            ))
        })?),
        Err(_) => available_capacity(parent)?,
    };
    if let Some(available) = available {
        if available < predicted {
            return Err(Qwen4Error::Invalid(format!(
                "insufficient destination capacity for Qwen4 artifact: \
                 predicted {predicted} bytes, available {available} bytes"
            )));
        }
    }
    Ok(())
}

fn available_capacity(path: &Path) -> Result<Option<u64>, Qwen4Error> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bytes = path.as_os_str().as_bytes();
        let c_path = std::ffi::CString::new(bytes).map_err(|_| {
            Qwen4Error::Invalid(format!("destination path contains NUL: {}", path.display()))
        })?;
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: c_path is NUL-terminated and stats points to writable
        // statvfs storage owned by this function.
        let result = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
        if result != 0 {
            return Err(Qwen4Error::io(
                format!("stat free space for {}", path.display()),
                io::Error::last_os_error(),
            ));
        }
        // SAFETY: statvfs initialized stats on success.
        let stats = unsafe { stats.assume_init() };
        return stats
            .f_bavail
            .checked_mul(stats.f_frsize)
            .map(Some)
            .ok_or_else(|| Qwen4Error::Invalid("destination capacity overflows".to_string()));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RemoteSpec {
    owner: String,
    repo: String,
    revision: String,
}

/// Parse the only remote source syntax accepted by the Qwen4 producer.
///
/// A revision is deliberately required to be a full commit ID.  Resolving
/// `main`, a tag, or a short SHA would make an artifact depend on mutable
/// repository state and is therefore never allowed here.
fn parse_remote_spec(input: &str) -> Result<Option<RemoteSpec>, Qwen4Error> {
    let Some(rest) = input.strip_prefix("hf://") else {
        return Ok(None);
    };
    let (repository, revision) = rest.split_once('@').ok_or_else(|| {
        Qwen4Error::Invalid(
            "Qwen4 remote input must be hf://OWNER/REPO@40_HEX_REVISION".to_string(),
        )
    })?;
    if repository.is_empty() || revision.is_empty() || revision.contains('@') {
        return Err(Qwen4Error::Invalid(
            "Qwen4 remote input must be hf://OWNER/REPO@40_HEX_REVISION".to_string(),
        ));
    }
    let mut components = repository.split('/');
    let owner = components.next().unwrap_or_default();
    let repo = components.next().unwrap_or_default();
    if components.next().is_some() || !valid_hf_component(owner) || !valid_hf_component(repo) {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Qwen4 Hugging Face repository {repository:?}; expected OWNER/REPO"
        )));
    }
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 Hugging Face revision must be exactly 40 hexadecimal characters, got {revision:?}"
        )));
    }
    Ok(Some(RemoteSpec {
        owner: owner.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
    }))
}

fn valid_hf_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RemoteObjectIdentity {
    etag: String,
    length: u64,
}

#[derive(Debug, Default)]
struct RemoteIdentityState {
    revision: Option<String>,
    objects: HashMap<String, RemoteObjectIdentity>,
}

#[derive(Debug)]
enum RemoteAttemptError {
    Retryable(Qwen4Error),
    Fatal(Qwen4Error),
}

impl RemoteAttemptError {
    fn fatal(error: Qwen4Error) -> Self {
        Self::Fatal(error)
    }
}

fn retryable_remote_status(status: u16) -> bool {
    status == 429 || (500..=599).contains(&status)
}

fn retryable_transport_io(error: &io::Error) -> bool {
    if matches!(
        error.kind(),
        io::ErrorKind::UnexpectedEof
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::TimedOut
    ) {
        return true;
    }
    error
        .get_ref()
        .and_then(|source| source.downcast_ref::<ureq::Error>())
        .is_some_and(retryable_ureq_error)
}

fn retryable_ureq_error(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed => true,
        ureq::Error::Io(error) => retryable_transport_io(error),
        _ => false,
    }
}

fn remote_request_error(url: &str, error: ureq::Error) -> RemoteAttemptError {
    let retryable = retryable_ureq_error(&error);
    let error = Qwen4Error::Invalid(format!("remote GET {url} failed: {error}"));
    if retryable {
        RemoteAttemptError::Retryable(error)
    } else {
        RemoteAttemptError::Fatal(error)
    }
}

fn remote_body_error(path: &str, error: io::Error) -> RemoteAttemptError {
    let retryable = retryable_transport_io(&error);
    let error = Qwen4Error::io(format!("read remote {path} body"), error);
    if retryable {
        RemoteAttemptError::Retryable(error)
    } else {
        RemoteAttemptError::Fatal(error)
    }
}

fn sleep_before_remote_retry(retry_index: usize) {
    let multiplier = 1u64 << retry_index.min(2);
    let delay_ms = REMOTE_RANGE_RETRY_BACKOFF_MS
        .saturating_mul(multiplier)
        .min(REMOTE_RANGE_RETRY_BACKOFF_MAX_MS);
    std::thread::sleep(Duration::from_millis(delay_ms));
}

struct RemoteSource {
    spec: RemoteSpec,
    base_url: String,
    agent: ureq::Agent,
    authorization: Option<String>,
    identity: Mutex<RemoteIdentityState>,
}

impl std::fmt::Debug for RemoteSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteSource")
            .field("spec", &self.spec)
            .field("base_url", &self.base_url)
            .field("authenticated", &self.authorization.is_some())
            .finish()
    }
}

impl RemoteSource {
    fn new(spec: RemoteSpec) -> Result<Self, Qwen4Error> {
        let base_url = env::var("HIPFIRE_HF_BASE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("HF_ENDPOINT")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(|| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string();
        if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 Hugging Face endpoint must use http:// or https://, got {base_url:?}"
            )));
        }
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(24 * 60 * 60)))
            .timeout_recv_body(Some(Duration::from_secs(24 * 60 * 60)))
            .http_status_as_error(false)
            .build()
            .into();
        let authorization = env::var_os("HF_TOKEN")
            .and_then(|value| {
                let value = value.to_string_lossy().into_owned();
                (!value.is_empty()).then(|| format!("Bearer {value}"))
            })
            .or_else(|| {
                env::var_os("HUGGING_FACE_HUB_TOKEN").and_then(|value| {
                    let value = value.to_string_lossy().into_owned();
                    (!value.is_empty()).then(|| format!("Bearer {value}"))
                })
            });
        Ok(Self {
            spec,
            base_url,
            agent,
            authorization,
            identity: Mutex::new(RemoteIdentityState::default()),
        })
    }

    fn url_for(&self, path: &str) -> String {
        format!(
            "{}/{}/{}/resolve/{}/{}",
            self.base_url, self.spec.owner, self.spec.repo, self.spec.revision, path
        )
    }

    fn remember_identity(
        &self,
        path: &str,
        headers: &ureq::http::HeaderMap,
        length: u64,
    ) -> Result<(), Qwen4Error> {
        let etag = headers
            .get("etag")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Qwen4Error::Invalid(format!("remote {path} response is missing ETag")))?
            .to_string();
        // The immutable revision is part of the resolved URL.  HF's CDN
        // returns `x-repo-commit` for small JSON responses but commonly omits
        // it for shard ranges; when present, it must still agree with the
        // pinned URL revision.
        let response_revision = ["x-repo-commit", "x-revision", "x-repo-revision"]
            .iter()
            .find_map(|name| headers.get(*name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty());
        if let Some(response_revision) = response_revision {
            if response_revision != self.spec.revision {
                return Err(Qwen4Error::Invalid(format!(
                    "remote {path} response revision {response_revision:?} disagrees with pinned {}",
                    self.spec.revision
                )));
            }
        }
        let revision = self.spec.revision.clone();
        let mut identity = self.identity.lock().map_err(|_| {
            Qwen4Error::Invalid("remote source identity lock is poisoned".to_string())
        })?;
        if let Some(expected) = &identity.revision {
            if expected != &revision {
                return Err(Qwen4Error::Invalid(format!(
                    "remote source revision changed from {expected:?} to {revision:?}"
                )));
            }
        } else {
            identity.revision = Some(revision);
        }
        let object = RemoteObjectIdentity { etag, length };
        if let Some(expected) = identity.objects.get(path) {
            if expected != &object {
                return Err(Qwen4Error::Invalid(format!(
                    "remote source identity changed for {path}: \
                     expected ETag/length {:?}/{}, got {:?}/{}",
                    expected.etag, expected.length, object.etag, object.length
                )));
            }
        } else {
            identity.objects.insert(path.to_string(), object);
        }
        Ok(())
    }

    fn read_json(&self, path: &str, max_bytes: u64) -> Result<Value, Qwen4Error> {
        validate_remote_path(path, None)?;
        let bytes = self.read_bounded(path, max_bytes)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| Qwen4Error::json(format!("parse remote {path}"), error))
    }

    /// Read a bounded object with a Range request.  Even small JSON objects
    /// use 206 so a proxy cannot silently turn an unbounded download into a
    /// metadata request.
    fn read_bounded(&self, path: &str, max_bytes: u64) -> Result<Vec<u8>, Qwen4Error> {
        if max_bytes == 0 {
            return Err(Qwen4Error::Invalid(
                "Qwen4 bounded remote read must have non-zero capacity".to_string(),
            ));
        }
        let mut retry_index = 0;
        loop {
            match self.read_bounded_attempt(path, max_bytes) {
                Ok(bytes) => return Ok(bytes),
                Err(RemoteAttemptError::Fatal(error)) => return Err(error),
                Err(RemoteAttemptError::Retryable(_)) if retry_index < REMOTE_RANGE_MAX_RETRIES => {
                    sleep_before_remote_retry(retry_index);
                    retry_index += 1;
                }
                Err(RemoteAttemptError::Retryable(error)) => return Err(error),
            }
        }
    }

    fn read_bounded_attempt(
        &self,
        path: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>, RemoteAttemptError> {
        let mut response = self.range_request(path, 0, max_bytes)?;
        let content_range = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                    "remote {path} response is missing a valid Content-Range"
                )))
            })?;
        let (start, end, total) =
            parse_content_range(content_range).map_err(RemoteAttemptError::fatal)?;
        let body_len = end
            .checked_sub(start)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| {
                RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                    "remote {path} Content-Range length overflows"
                )))
            })?;
        if start != 0 || body_len > max_bytes || total != body_len {
            return Err(RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} bounded range {content_range:?} is not the complete object within {} bytes",
                max_bytes
            ))));
        }
        self.remember_identity(path, response.headers(), total)
            .map_err(RemoteAttemptError::fatal)?;
        let announced =
            content_length(response.headers(), path).map_err(RemoteAttemptError::fatal)?;
        if announced != body_len {
            return Err(RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} Content-Length {announced} disagrees with Content-Range length {body_len}"
            ))));
        }
        let length = usize::try_from(body_len).map_err(|_| {
            RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} response does not fit usize"
            )))
        })?;
        let mut bytes = vec![0u8; length];
        read_response_exact(&mut response, &mut bytes, path)
            .map_err(|error| remote_body_error(path, error))?;
        Ok(bytes)
    }

    /// Read exactly `dst.len()` bytes from one immutable remote shard range.
    /// `expected_total` is omitted only for the initial eight-byte read used
    /// to discover a safetensors shard's total length.
    ///
    /// Each attempt fills the caller's staging buffer but no stream writer
    /// sees it until this method returns successfully.  A failed body read is
    /// therefore safely replayed from byte zero with the same Range header.
    fn read_range(
        &self,
        path: &str,
        offset: u64,
        dst: &mut [u8],
        expected_total: Option<u64>,
    ) -> Result<u64, Qwen4Error> {
        if dst.is_empty() {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} requested an empty range"
            )));
        }
        validate_remote_path(path, Some("safetensors"))?;
        let length = u64::try_from(dst.len())
            .map_err(|_| Qwen4Error::Invalid(format!("remote {path} range is too large")))?;
        let mut retry_index = 0;
        loop {
            match self.read_range_attempt(path, offset, dst, length, expected_total) {
                Ok(total) => return Ok(total),
                Err(RemoteAttemptError::Fatal(error)) => return Err(error),
                Err(RemoteAttemptError::Retryable(_)) if retry_index < REMOTE_RANGE_MAX_RETRIES => {
                    sleep_before_remote_retry(retry_index);
                    retry_index += 1;
                }
                Err(RemoteAttemptError::Retryable(error)) => return Err(error),
            }
        }
    }

    fn read_range_attempt(
        &self,
        path: &str,
        offset: u64,
        dst: &mut [u8],
        length: u64,
        expected_total: Option<u64>,
    ) -> Result<u64, RemoteAttemptError> {
        let mut response = self.range_request(path, offset, length)?;
        let content_range = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                    "remote {path} response is missing a valid Content-Range"
                )))
            })?;
        let (start, end, total) =
            parse_content_range(content_range).map_err(RemoteAttemptError::fatal)?;
        let body_len = end
            .checked_sub(start)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| {
                RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                    "remote {path} Content-Range length overflows"
                )))
            })?;
        if start != offset
            || body_len != length
            || expected_total.is_some_and(|expected| total != expected)
        {
            let expected_total = expected_total
                .map(|value| format!(" / {value}"))
                .unwrap_or_default();
            return Err(RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} returned range {content_range:?}, expected bytes {offset}-{}{expected_total}",
                offset.saturating_add(length).saturating_sub(1)
            ))));
        }
        self.remember_identity(path, response.headers(), total)
            .map_err(RemoteAttemptError::fatal)?;
        let announced =
            content_length(response.headers(), path).map_err(RemoteAttemptError::fatal)?;
        if announced != length {
            return Err(RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} Content-Length {announced} disagrees with requested range length {length}"
            ))));
        }
        read_response_exact(&mut response, dst, path)
            .map_err(|error| remote_body_error(path, error))?;
        Ok(total)
    }

    fn range_request(
        &self,
        path: &str,
        offset: u64,
        length: u64,
    ) -> Result<ureq::http::Response<ureq::Body>, RemoteAttemptError> {
        if length == 0 {
            return Err(RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} requested an empty range"
            ))));
        }
        let end = offset.checked_add(length - 1).ok_or_else(|| {
            RemoteAttemptError::fatal(Qwen4Error::Invalid(format!(
                "remote {path} range overflows"
            )))
        })?;
        let url = self.url_for(path);
        let mut request = self
            .agent
            .get(&url)
            .header("Range", &format!("bytes={offset}-{end}"));
        if let Some(authorization) = &self.authorization {
            request = request.header("Authorization", authorization);
        }
        let response = request
            .call()
            .map_err(|error| remote_request_error(&url, error))?;
        let status = response.status().as_u16();
        if status != 206 {
            let error = Qwen4Error::Invalid(format!(
                "remote GET {url} returned HTTP {status}, expected 206 Partial Content"
            ));
            return Err(if retryable_remote_status(status) {
                RemoteAttemptError::Retryable(error)
            } else {
                RemoteAttemptError::Fatal(error)
            });
        }
        Ok(response)
    }

    fn enumerate_shards(&self) -> Result<Vec<String>, Qwen4Error> {
        let index = self.read_json("model.safetensors.index.json", MAX_CONFIG_BYTES)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                Qwen4Error::Invalid(
                    "remote model.safetensors.index.json is missing an object weight_map"
                        .to_string(),
                )
            })?;
        let mut paths = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                Qwen4Error::Invalid(
                    "remote safetensors index weight_map contains a non-string shard".to_string(),
                )
            })?;
            if !shard.ends_with(".safetensors") {
                continue;
            }
            validate_remote_path(shard, Some("safetensors"))?;
            paths.insert(shard.to_string());
        }
        if paths.is_empty() {
            return Err(Qwen4Error::Invalid(
                "remote safetensors index references no .safetensors shards".to_string(),
            ));
        }
        Ok(paths.into_iter().collect())
    }
}

fn validate_remote_path(path: &str, required_suffix: Option<&str>) -> Result<(), Qwen4Error> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        })
    {
        return Err(Qwen4Error::Invalid(format!(
            "remote Hugging Face path {path:?} is not a safe relative path"
        )));
    }
    if let Some(suffix) = required_suffix {
        if !path.ends_with(suffix) {
            return Err(Qwen4Error::Invalid(format!(
                "remote path {path:?} does not end with .{suffix}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalFileIdentity {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

fn local_file_identity(path: &Path) -> Result<LocalFileIdentity, Qwen4Error> {
    let metadata = fs::metadata(path)
        .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(LocalFileIdentity {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        Ok(LocalFileIdentity {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

fn verify_local_file_identity(path: &Path, expected: &LocalFileIdentity) -> Result<(), Qwen4Error> {
    let actual = local_file_identity(path)?;
    if &actual != expected {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source identity changed for {}",
            path.display()
        )));
    }
    Ok(())
}
fn parse_content_range(value: &str) -> Result<(u64, u64, u64), Qwen4Error> {
    let (unit, range) = value.split_once(' ').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if unit != "bytes" || range.contains(' ') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        )));
    }
    let (offsets, total) = range.split_once('/').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if total == "*" || total.is_empty() || total.contains('/') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; total length must be explicit"
        )));
    }
    let (start, end) = offsets.split_once('-').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if start.is_empty() || end.is_empty() || end.contains('-') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        )));
    }
    let start = start.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; start is not an integer"
        ))
    })?;
    let end = end.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; end is not an integer"
        ))
    })?;
    let total = total.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; total is not an integer"
        ))
    })?;
    if start > end || total <= end {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; require START <= END < TOTAL"
        )));
    }
    Ok((start, end, total))
}

fn content_length(headers: &ureq::http::HeaderMap, path: &str) -> Result<u64, Qwen4Error> {
    let value = headers
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            Qwen4Error::Invalid(format!("remote {path} response is missing Content-Length"))
        })?;
    value.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "remote {path} response has invalid Content-Length {value:?}"
        ))
    })
}

fn read_response_exact(
    response: &mut ureq::http::Response<ureq::Body>,
    dst: &mut [u8],
    path: &str,
) -> io::Result<()> {
    let mut reader = response.body_mut().as_reader();
    let mut filled = 0usize;
    while filled < dst.len() {
        let count = reader.read(&mut dst[filled..])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "remote {path} body ended after {filled} bytes, expected {}",
                    dst.len()
                ),
            ));
        }
        filled += count;
    }
    let mut extra = [0u8; 1];
    let count = reader.read(&mut extra)?;
    if count != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "remote {path} body exceeded the announced {} bytes",
                dst.len()
            ),
        ));
    }
    Ok(())
}

enum SourceSet {
    Local {
        paths: Vec<PathBuf>,
        identities: Vec<LocalFileIdentity>,
    },
    Remote {
        source: Arc<RemoteSource>,
        paths: Vec<String>,
    },
}

impl SourceSet {
    fn verify_identity(&self) -> Result<(), Qwen4Error> {
        match self {
            Self::Local { paths, identities } => {
                if paths.len() != identities.len() {
                    return Err(Qwen4Error::Invalid(
                        "Qwen4 local source identity seal is incomplete".to_string(),
                    ));
                }
                for (path, expected) in paths.iter().zip(identities) {
                    verify_local_file_identity(path, expected)?;
                }
            }
            Self::Remote { source, paths } => {
                let identity = source.identity.lock().map_err(|_| {
                    Qwen4Error::Invalid("remote source identity lock is poisoned".to_string())
                })?;
                for path in paths {
                    if !identity.objects.contains_key(path) {
                        return Err(Qwen4Error::Invalid(format!(
                            "remote source shard {path} was not identity-sealed"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_source_shard_count(source: &SourceSet, mode: Qwen4Mode) -> Result<(), Qwen4Error> {
    if mode == Qwen4Mode::Production {
        let count = match source {
            SourceSet::Local { paths, .. } => paths.len(),
            SourceSet::Remote { paths, .. } => paths.len(),
        };
        if count != PINNED_SHARD_COUNT {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 production admission requires exactly {PINNED_SHARD_COUNT} safetensors shards, found {count}"
            )));
        }
    }
    Ok(())
}

fn validate_source_inventory(tensors: &[SourceTensor], mode: Qwen4Mode) -> Result<(), Qwen4Error> {
    if mode != Qwen4Mode::Production {
        return Ok(());
    }
    if tensors.len() != PINNED_SOURCE_TENSOR_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 production admission requires exactly {PINNED_SOURCE_TENSOR_COUNT} source tensors, found {}",
            tensors.len()
        )));
    }
    let bf16 = tensors
        .iter()
        .filter(|tensor| tensor.dtype == "BF16")
        .count();
    let i64 = tensors
        .iter()
        .filter(|tensor| tensor.dtype == "I64")
        .count();
    let vision = tensors
        .iter()
        .filter(|tensor| is_vision_tensor(&tensor.name))
        .count();
    if bf16 != PINNED_BF16_TENSOR_COUNT || i64 != PINNED_I64_TENSOR_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 production inventory requires {PINNED_BF16_TENSOR_COUNT} BF16 and {PINNED_I64_TENSOR_COUNT} I64 tensors, found {bf16} BF16 and {i64} I64"
        )));
    }
    if vision != PINNED_VISION_TENSOR_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 production inventory requires {PINNED_VISION_TENSOR_COUNT} positively identified vision tensors, found {vision}"
        )));
    }
    if tensors
        .iter()
        .any(|tensor| tensor.dtype != "BF16" && tensor.dtype != "I64")
    {
        return Err(Qwen4Error::Invalid(
            "Qwen4 production inventory contains a source dtype other than BF16 or I64".to_string(),
        ));
    }
    Ok(())
}

enum SourceKind {
    Local {
        file: Arc<File>,
    },
    Remote {
        source: Arc<RemoteSource>,
        path: String,
    },
}

struct SourceShard {
    path: PathBuf,
    kind: SourceKind,
    file_len: u64,
    local_identity: Option<LocalFileIdentity>,
}

impl Clone for SourceShard {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            kind: match &self.kind {
                SourceKind::Local { file } => SourceKind::Local {
                    file: Arc::clone(file),
                },
                SourceKind::Remote { source, path } => SourceKind::Remote {
                    source: Arc::clone(source),
                    path: path.clone(),
                },
            },
            file_len: self.file_len,
            local_identity: self.local_identity.clone(),
        }
    }
}

impl std::fmt::Debug for SourceShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceShard")
            .field("path", &self.path)
            .field("file_len", &self.file_len)
            .finish()
    }
}

impl SourceShard {
    fn verify_identity(&self) -> Result<(), Qwen4Error> {
        if let Some(expected) = &self.local_identity {
            verify_local_file_identity(&self.path, expected)?;
        }
        Ok(())
    }

    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), Qwen4Error> {
        let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} source range overflows", self.path.display()))
        })?;
        if end > self.file_len {
            return Err(Qwen4Error::Invalid(format!(
                "{} source range [{offset}, {end}) exceeds {} bytes",
                self.path.display(),
                self.file_len
            )));
        }
        self.verify_identity()?;
        match &self.kind {
            SourceKind::Local { file } => read_exact_at(file, offset, dst)
                .map_err(|error| Qwen4Error::io(format!("read {}", self.path.display()), error)),
            SourceKind::Remote { source, path } => {
                source.read_range(path, offset, dst, Some(self.file_len))?;
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
struct SourceTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_start: u64,
    data_end: u64,
    shard: Arc<SourceShard>,
}

impl SourceTensor {
    fn data_len(&self) -> u64 {
        self.data_end - self.data_start
    }

    fn read_range(&self, relative: u64, dst: &mut [u8]) -> Result<(), Qwen4Error> {
        let len = dst.len() as u64;
        let end = relative
            .checked_add(len)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} range overflows", self.name)))?;
        if end > self.data_len() {
            return Err(Qwen4Error::Invalid(format!(
                "{} range [{relative}, {end}) exceeds tensor payload {}",
                self.name,
                self.data_len()
            )));
        }
        let offset = self
            .data_start
            .checked_add(relative)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} file offset overflows", self.name)))?;
        self.shard.read_exact_at(offset, dst)
    }
}

fn source_paths(input: &Path) -> Result<SourceSet, Qwen4Error> {
    if let Some(raw) = input.to_str() {
        if let Some(spec) = parse_remote_spec(raw)? {
            let source = Arc::new(RemoteSource::new(spec)?);
            let paths = source.enumerate_shards()?;
            return Ok(SourceSet::Remote { source, paths });
        }
    }
    let metadata =
        fs::metadata(input).map_err(|error| Qwen4Error::io(input.display().to_string(), error))?;
    if metadata.is_file() {
        if input.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 input file must have .safetensors extension: {}",
                input.display()
            )));
        }
        let identity = local_file_identity(input)?;
        return Ok(SourceSet::Local {
            paths: vec![input.to_path_buf()],
            identities: vec![identity],
        });
    }
    if !metadata.is_dir() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 input is neither a directory nor a safetensors file: {}",
            input.display()
        )));
    }
    let mut paths = Vec::new();
    let entries = fs::read_dir(input)
        .map_err(|error| Qwen4Error::io(format!("read Qwen4 input {}", input.display()), error))?;
    for entry in entries {
        let entry = entry.map_err(|error| Qwen4Error::io("read Qwen4 directory entry", error))?;
        let path = entry.path();
        if entry
            .file_type()
            .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
            .is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        {
            paths.push(path);
        }
    }
    paths.sort_by(|left, right| left.as_os_str().cmp(right.as_os_str()));
    if paths.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 input directory contains no .safetensors shards: {}",
            input.display()
        )));
    }
    let identities = paths
        .iter()
        .map(|path| local_file_identity(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SourceSet::Local { paths, identities })
}

fn load_inventory(source: &SourceSet) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let mut tensors = HashMap::<String, SourceTensor>::new();
    match source {
        SourceSet::Local { paths, .. } => {
            for path in paths {
                for tensor in parse_safetensors_header(path)? {
                    if tensors.insert(tensor.name.clone(), tensor).is_some() {
                        return Err(Qwen4Error::Invalid(format!(
                            "duplicate safetensors tensor name in Qwen4 input: {}",
                            path.display()
                        )));
                    }
                }
            }
        }
        SourceSet::Remote { source, paths } => {
            for path in paths {
                for tensor in parse_remote_safetensors_header(Arc::clone(source), path)? {
                    if tensors.insert(tensor.name.clone(), tensor).is_some() {
                        return Err(Qwen4Error::Invalid(format!(
                            "duplicate safetensors tensor name in remote shard {path}"
                        )));
                    }
                }
            }
        }
    }
    let mut tensors: Vec<_> = tensors.into_values().collect();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensors)
}

fn parse_safetensors_header(path: &Path) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let local_identity = local_file_identity(path)?;
    let file_len = local_identity.len;
    let mut length_bytes = [0u8; 8];
    read_exact_at(&file, 0, &mut length_bytes).map_err(|error| {
        Qwen4Error::io(
            format!("read safetensors header length {}", path.display()),
            error,
        )
    })?;
    let shard = Arc::new(SourceShard {
        path: path.to_path_buf(),
        kind: SourceKind::Local {
            file: Arc::new(file),
        },
        file_len,
        local_identity: Some(local_identity),
    });
    parse_safetensors_header_from_shard(shard, length_bytes)
}

fn parse_remote_safetensors_header(
    source: Arc<RemoteSource>,
    path: &str,
) -> Result<Vec<SourceTensor>, Qwen4Error> {
    validate_remote_path(path, Some("safetensors"))?;
    let mut length_bytes = [0u8; 8];
    let file_len = source.read_range(path, 0, &mut length_bytes, None)?;
    let shard = Arc::new(SourceShard {
        path: PathBuf::from(source.url_for(path)),
        kind: SourceKind::Remote {
            source: Arc::clone(&source),
            path: path.to_string(),
        },
        file_len,
        local_identity: None,
    });
    parse_safetensors_header_from_shard(shard, length_bytes)
}

fn parse_safetensors_header_from_shard(
    shard: Arc<SourceShard>,
    length_bytes: [u8; 8],
) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let path = shard.path.display().to_string();
    let file_len = shard.file_len;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{path} safetensors header length {header_len} is outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{path} header offset overflows")))?;
    if header_end > file_len {
        return Err(Qwen4Error::Invalid(format!(
            "{path} safetensors header is truncated: end {header_end}, file {file_len}"
        )));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| Qwen4Error::Invalid(format!("{path} header is too large")))?;
    let mut header_bytes = vec![0u8; header_len_usize];
    shard.read_exact_at(8, &mut header_bytes)?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| Qwen4Error::json(format!("parse safetensors header {path}"), error))?;
    let object = header.as_object().ok_or_else(|| {
        Qwen4Error::Invalid(format!("{path} safetensors header is not an object"))
    })?;
    let mut tensors = Vec::with_capacity(object.len().saturating_sub(1));
    for (name, descriptor) in object {
        if name == "__metadata__" {
            continue;
        }
        let descriptor = descriptor.as_object().ok_or_else(|| {
            Qwen4Error::Invalid(format!("{name} descriptor in {path} is not an object"))
        })?;
        let dtype = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing a dtype")))?
            .to_string();
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing a shape")))?
            .iter()
            .map(|dimension| {
                dimension.as_u64().ok_or_else(|| {
                    Qwen4Error::Invalid(format!("{name} has a non-integer shape dimension"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(Qwen4Error::Invalid(format!(
                "{name} data_offsets must have exactly two elements"
            )));
        }
        let relative_start = offsets[0]
            .as_u64()
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} has a non-integer start offset")))?;
        let relative_end = offsets[1]
            .as_u64()
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} has a non-integer end offset")))?;
        let relative_len = relative_end
            .checked_sub(relative_start)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} data_offsets are reversed")))?;
        let expected_len = match dtype.as_str() {
            "BF16" => checked_bf16_bytes(&shape, name)?,
            "I64" => checked_product(&shape, name)?
                .checked_mul(8)
                .ok_or_else(|| Qwen4Error::Invalid(format!("{name} I64 length overflows")))?,
            _ => relative_len,
        };
        if relative_len != expected_len {
            return Err(Qwen4Error::Invalid(format!(
                "{name} payload is {relative_len} bytes, expected {expected_len} for dtype {dtype}"
            )));
        }
        let data_start = header_end
            .checked_add(relative_start)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} data start overflows")))?;
        let data_end = header_end
            .checked_add(relative_end)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} data end overflows")))?;
        if data_end > file_len {
            return Err(Qwen4Error::Invalid(format!(
                "{name} payload [{data_start}, {data_end}) exceeds {file_len} bytes"
            )));
        }
        tensors.push(SourceTensor {
            name: name.clone(),
            dtype,
            shape,
            data_start,
            data_end,
            shard: Arc::clone(&shard),
        });
    }
    Ok(tensors)
}

fn load_optional_config(source: &SourceSet) -> Result<Option<Value>, Qwen4Error> {
    match source {
        SourceSet::Remote { source, .. } => {
            Ok(Some(source.read_json("config.json", MAX_CONFIG_BYTES)?))
        }
        SourceSet::Local { paths, .. } => {
            let first = paths.first().ok_or_else(|| {
                Qwen4Error::Invalid("Qwen4 local source has no safetensors shards".to_string())
            })?;
            let path = first
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("config.json");
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
                Err(error) => {
                    return Err(Qwen4Error::io(format!("stat {}", path.display()), error));
                }
            };
            if metadata.len() > MAX_CONFIG_BYTES {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 config {} is larger than {MAX_CONFIG_BYTES} bytes",
                    path.display()
                )));
            }
            let mut file = File::open(&path)
                .map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
            let mut bytes = Vec::with_capacity(metadata.len() as usize);
            file.read_to_end(&mut bytes)
                .map_err(|error| Qwen4Error::io(format!("read {}", path.display()), error))?;
            let value = serde_json::from_slice(&bytes)
                .map_err(|error| Qwen4Error::json(format!("parse {}", path.display()), error))?;
            Ok(Some(value))
        }
    }
}

/// Canonical HF tokenizer contract copied into every production Qwen4 HFQM
/// artifact. The tokenizer JSON is required and validated before any payload
/// quantization; sidecars remain optional but are embedded when present.
#[derive(Debug, Clone)]
struct TokenizerMetadata {
    tokenizer: String,
    tokenizer_config: Option<Value>,
    generation_config: Option<Value>,
}

fn load_tokenizer_metadata(source: &SourceSet) -> Result<TokenizerMetadata, Qwen4Error> {
    let (tokenizer_bytes, mut tokenizer_config, generation_config, chat_template) = match source {
        SourceSet::Local { paths, .. } => {
            let first = paths.first().ok_or_else(|| {
                Qwen4Error::Invalid("Qwen4 local source has no safetensors shards".to_string())
            })?;
            let parent = first.parent().unwrap_or_else(|| Path::new("."));
            (
                read_local_bytes(&parent.join("tokenizer.json"), MAX_TOKENIZER_BYTES, true)?
                    .expect("required local tokenizer"),
                read_local_json(&parent.join("tokenizer_config.json"), MAX_CONFIG_BYTES)?,
                read_local_json(&parent.join("generation_config.json"), MAX_CONFIG_BYTES)?,
                read_local_text(&parent.join("chat_template.jinja"), MAX_CONFIG_BYTES)?,
            )
        }
        SourceSet::Remote { source, .. } => (
            source.read_bounded("tokenizer.json", MAX_TOKENIZER_BYTES)?,
            read_remote_optional_json(source, "tokenizer_config.json", MAX_CONFIG_BYTES)?,
            read_remote_optional_json(source, "generation_config.json", MAX_CONFIG_BYTES)?,
            read_remote_optional_text(source, "chat_template.jinja", MAX_CONFIG_BYTES)?,
        ),
    };
    let tokenizer = String::from_utf8(tokenizer_bytes).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 tokenizer.json is not UTF-8: {error}"))
    })?;
    hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&tokenizer).map_err(|error| {
        Qwen4Error::Invalid(format!("Qwen4 tokenizer.json is invalid: {error}"))
    })?;

    if let Some(template) = chat_template {
        let config = tokenizer_config.get_or_insert_with(|| Value::Object(Map::new()));
        let object = config.as_object_mut().ok_or_else(|| {
            Qwen4Error::Invalid("Qwen4 tokenizer_config.json must be a JSON object".to_string())
        })?;
        let has_template = object
            .get("chat_template")
            .map(|value| !value.is_null())
            .unwrap_or(false);
        if !has_template {
            object.insert("chat_template".to_string(), Value::String(template));
        }
    }

    Ok(TokenizerMetadata {
        tokenizer,
        tokenizer_config,
        generation_config,
    })
}

fn read_local_bytes(
    path: &Path,
    max_bytes: u64,
    required: bool,
) -> Result<Option<Vec<u8>>, Qwen4Error> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound && !required => return Ok(None),
        Err(error) => return Err(Qwen4Error::io(format!("stat {}", path.display()), error)),
    };
    if metadata.len() > max_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 metadata file {} is larger than {max_bytes} bytes",
            path.display()
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        Qwen4Error::Invalid(format!(
            "Qwen4 metadata file {} is too large",
            path.display()
        ))
    })?;
    let mut file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|error| Qwen4Error::io(format!("read {}", path.display()), error))?;
    if bytes.len() as u64 > max_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 metadata file {} grew above {max_bytes} bytes while reading",
            path.display()
        )));
    }
    Ok(Some(bytes))
}

fn read_local_text(path: &Path, max_bytes: u64) -> Result<Option<String>, Qwen4Error> {
    let Some(bytes) = read_local_bytes(path, max_bytes, false)? else {
        return Ok(None);
    };
    String::from_utf8(bytes).map(Some).map_err(|error| {
        Qwen4Error::Invalid(format!(
            "Qwen4 metadata {} is not UTF-8: {error}",
            path.display()
        ))
    })
}

fn read_local_json(path: &Path, max_bytes: u64) -> Result<Option<Value>, Qwen4Error> {
    let Some(bytes) = read_local_bytes(path, max_bytes, false)? else {
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| Qwen4Error::json(format!("parse {}", path.display()), error))
}

fn read_remote_optional_json(
    source: &RemoteSource,
    path: &str,
    max_bytes: u64,
) -> Result<Option<Value>, Qwen4Error> {
    match source.read_json(path, max_bytes) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.to_string().contains("returned HTTP 404") => Ok(None),
        Err(error) => Err(error),
    }
}

fn read_remote_optional_text(
    source: &RemoteSource,
    path: &str,
    max_bytes: u64,
) -> Result<Option<String>, Qwen4Error> {
    match source.read_bounded(path, max_bytes) {
        Ok(bytes) => String::from_utf8(bytes)
            .map(Some)
            .map_err(|error| Qwen4Error::Invalid(format!("remote {path} is not UTF-8: {error}"))),
        Err(error) if error.to_string().contains("returned HTTP 404") => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum I64Role {
    Multipliers,
    VocabSizes,
    Offsets,
}

fn i64_role(name: &str) -> Option<I64Role> {
    if name.ends_with(".layer_multipliers") {
        Some(I64Role::Multipliers)
    } else if name.ends_with(".ngram_heads_vocab_sizes") {
        Some(I64Role::VocabSizes)
    } else if name.ends_with(".ngram_heads_offsets") {
        Some(I64Role::Offsets)
    } else {
        None
    }
}

fn ple_shard_index(name: &str) -> Option<usize> {
    let marker = ".ngram_embedding.shard_";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let digits_end = rest.find(".weight")?;
    if digits_end == 0 || &rest[digits_end..] != ".weight" {
        return None;
    }
    let digits = &rest[..digits_end];
    if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn is_vision_tensor(name: &str) -> bool {
    [
        "model.visual.",
        "model.vision_tower.",
        "model.vision_projection.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "visual.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManifestRole {
    Bf16,
    Matrix(DType),
    /// Rank-3 routed expert block (`[experts, rows, K]`).  The role carries the
    /// manifest's declared target so the plan packs exactly what the loader
    /// expects; a role that only knew "expert" would have to guess, and a guess
    /// that drifts from the manifest is a mis-decoded weight, not an error.
    GateUp(DType),
    Down(DType),
    Ple,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestExpectation {
    shape: Vec<u64>,
    role: ManifestRole,
    is_mtp: bool,
    ple_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetadataExpectation {
    role: I64Role,
    shape: Vec<u64>,
}

#[derive(Debug)]
struct ManifestIndex {
    required: BTreeMap<String, ManifestExpectation>,
    aliases: BTreeMap<String, ManifestExpectation>,
    metadata: BTreeMap<String, MetadataExpectation>,
    ple_names: BTreeMap<usize, String>,
    manifest_order: HashMap<String, usize>,
}
impl ManifestIndex {
    fn build(manifest: &Qwen4Manifest) -> Result<Self, Qwen4Error> {
        let mut required = BTreeMap::new();
        let mut aliases = BTreeMap::new();
        let mut ple_names = BTreeMap::new();
        let mut manifest_order = HashMap::new();
        for (order, entry) in manifest.weights.iter().enumerate() {
            manifest_order.insert(entry.name.clone(), order);
            let shape = entry
                .logical_shape
                .iter()
                .map(|&dimension| {
                    u64::try_from(dimension).map_err(|_| {
                        Qwen4Error::Invalid(format!(
                            "Qwen4 manifest shape for {} does not fit u64",
                            entry.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let role = if entry.residency.is_external() {
                ManifestRole::Ple
            } else if entry.name.ends_with(".experts.gate_up_proj") && shape.len() == 3 {
                ManifestRole::GateUp(entry.dtype)
            } else if entry.name.ends_with(".experts.down_proj") && shape.len() == 3 {
                ManifestRole::Down(entry.dtype)
            } else if qwen4_quantized_dtype(entry.dtype) {
                ManifestRole::Matrix(entry.dtype)
            } else if entry.dtype == DType::BF16 {
                ManifestRole::Bf16
            } else {
                // Silent demotion to BF16 here is exactly how a packed trunk
                // once shipped unpacked: the manifest asked for a matrix quant
                // the predicate did not recognise and the plan quietly carried
                // source bytes instead.  Fail instead of demoting.
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 manifest offers {} as {:?}, which no plan role can carry; \
                     a matrix target must be BF16, Q8_0 or MFP4G32E8SOA",
                    entry.name, entry.dtype
                )));
            };
            let ple_index = if role == ManifestRole::Ple {
                let index = ple_shard_index(&entry.name).ok_or_else(|| {
                    Qwen4Error::Invalid(format!(
                        "Qwen4 manifest external record is not a numeric PLE shard: {}",
                        entry.name
                    ))
                })?;
                if index >= PLE_SHARD_COUNT {
                    return Err(Qwen4Error::Invalid(format!(
                        "Qwen4 manifest PLE shard index {index} is outside 0..{}",
                        PLE_SHARD_COUNT - 1
                    )));
                }
                if ple_names.insert(index, entry.name.clone()).is_some() {
                    return Err(Qwen4Error::Invalid(format!(
                        "Qwen4 manifest has duplicate PLE shard index {index}"
                    )));
                }
                Some(index)
            } else {
                None
            };
            let expectation = ManifestExpectation {
                shape,
                role,
                is_mtp: entry.name.starts_with("mtp."),
                ple_index,
            };
            let is_alias = matches!(entry.policy, ShardPolicy::Tied { .. });
            if required.contains_key(&entry.name) || aliases.contains_key(&entry.name) {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 manifest has duplicate weight name {}",
                    entry.name
                )));
            }
            if is_alias {
                aliases.insert(entry.name.clone(), expectation);
            } else {
                required.insert(entry.name.clone(), expectation);
            }
        }
        if ple_names.len() != PLE_SHARD_COUNT {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 manifest requires {PLE_SHARD_COUNT} numeric PLE shards, found {}",
                ple_names.len()
            )));
        }

        let mut metadata = BTreeMap::new();
        for record in &manifest.metadata {
            let role = i64_role(&record.name).ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "Qwen4 manifest has unknown I64 metadata record {}",
                    record.name
                ))
            })?;
            let shape = record
                .shape
                .iter()
                .map(|&dimension| {
                    u64::try_from(dimension).map_err(|_| {
                        Qwen4Error::Invalid(format!(
                            "Qwen4 metadata shape for {} does not fit u64",
                            record.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if record.source_dtype != "I64" {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 metadata {} has source dtype {}, expected I64",
                    record.name, record.source_dtype
                )));
            }
            if required.contains_key(&record.name)
                || aliases.contains_key(&record.name)
                || metadata
                    .insert(record.name.clone(), MetadataExpectation { role, shape })
                    .is_some()
            {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 manifest has duplicate metadata name {}",
                    record.name
                )));
            }
        }
        let roles: BTreeSet<_> = metadata.values().map(|record| record.role).collect();
        if roles.len() != 3
            || !roles.contains(&I64Role::Multipliers)
            || !roles.contains(&I64Role::VocabSizes)
            || !roles.contains(&I64Role::Offsets)
        {
            return Err(Qwen4Error::Invalid(
                "Qwen4 manifest must declare exactly three distinct PLE I64 metadata roles"
                    .to_string(),
            ));
        }
        Ok(Self {
            required,
            aliases,
            metadata,
            ple_names,
            manifest_order,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpertKind {
    GateUp,
    Down,
}

fn checked_product(values: &[u64], what: &str) -> Result<u64, Qwen4Error> {
    values.iter().try_fold(1u64, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{what} dimensions overflow u64")))
    })
}

fn checked_bf16_bytes(shape: &[u64], what: &str) -> Result<u64, Qwen4Error> {
    checked_product(shape, what)?
        .checked_mul(2)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{what} BF16 byte length overflows u64")))
}

fn shape_u32(shape: &[u64], name: &str) -> Result<Vec<u32>, Qwen4Error> {
    if shape.len() > u8::MAX as usize {
        return Err(Qwen4Error::Invalid(format!(
            "{name} has {} dimensions, HFQM supports at most {}",
            shape.len(),
            u8::MAX
        )));
    }
    shape
        .iter()
        .map(|&dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4Error::Invalid(format!("{name} dimension {dimension} does not fit u32"))
            })
        })
        .collect()
}

fn validate_expert_shape(
    tensor: &SourceTensor,
    kind: ExpertKind,
    expected: &[u64],
) -> Result<(u64, u64), Qwen4Error> {
    if tensor.shape != expected {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {:?}, expected {:?}; Qwen4 never pads expert K",
            tensor.name, tensor.shape, expected
        )));
    }
    if tensor.shape.len() != 3 {
        return Err(Qwen4Error::Invalid(format!(
            "{} manifest expert role {kind:?} requires a rank-3 tensor",
            tensor.name
        )));
    }
    let rows = tensor.shape[0]
        .checked_mul(tensor.shape[1])
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row count overflows", tensor.name)))?;
    let k = tensor.shape[2];
    let expected_bytes = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    Ok((rows, k))
}

fn validate_matrix_shape(
    tensor: &SourceTensor,
    expected: &[u64],
) -> Result<(u64, u64), Qwen4Error> {
    if tensor.shape != expected {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {:?}, expected manifest matrix shape {:?}",
            tensor.name, tensor.shape, expected
        )));
    }
    if tensor.shape.len() != 2 {
        return Err(Qwen4Error::Invalid(format!(
            "{} manifest matrix role requires a rank-2 tensor",
            tensor.name
        )));
    }
    let rows = tensor.shape[0];
    let k = tensor.shape[1];
    let expected_bytes = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    Ok((rows, k))
}

/// Wire tag and group size for a packed matrix dtype.  A dtype with no tag is
/// an error rather than an `unreachable!`: the plan's role derivation and this
/// table drifting apart is how a quantized trunk once shipped unpacked, and a
/// panic is a poor way to learn that.
fn matrix_quant_type(dtype: DType) -> Result<(u8, u32), Qwen4Error> {
    match dtype {
        DType::MQ4G256V2 => Ok((MQ4G256V2_QUANT_TYPE, MQ4G256V2_GROUP_SIZE as u32)),
        DType::MQ4G128V2 => Ok((MQ4G128V2_QUANT_TYPE, MQ4G128V2_GROUP_SIZE as u32)),
        DType::MQ6G256V2 => Ok((MQ6G256V2_QUANT_TYPE, MQ6G256V2_GROUP_SIZE as u32)),
        DType::Q8_0 => Ok((Q8F16_QUANT_TYPE, Q8F16_BLOCK_SIZE as u32)),
        DType::MFP4G32E8SOA => Ok((MFP4G32E8SOA_QUANT_TYPE, MFP4G32E8SOA_BLOCK_SIZE as u32)),
        other => Err(Qwen4Error::Invalid(format!(
            "Qwen4 matrix quant dtype {other:?} has no wire tag; add it to matrix_quant_type"
        ))),
    }
}

fn quantized_data_len_for_dtype(dtype: DType, rows: u64, k: u64) -> Result<u64, Qwen4Error> {
    if dtype == DType::MFP4G32E8SOA {
        return mfp4e8soa_row_geometry(rows, k).map(|(_, extent)| extent);
    }
    let (group_size, group_bytes, require_aligned_k) = match dtype {
        DType::MQ4G256V2 => (MQ4G256V2_GROUP_SIZE, MQ4G256V2_GROUP_BYTES, true),
        DType::MQ4G128V2 => (MQ4G128V2_GROUP_SIZE, MQ4G128V2_GROUP_BYTES, false),
        DType::MQ6G256V2 => (MQ6G256V2_GROUP_SIZE, MQ6G256V2_GROUP_BYTES, true),
        DType::Q8_0 => (Q8F16_BLOCK_SIZE, Q8F16_BLOCK_BYTES, false),
        _ => {
            return Err(Qwen4Error::Invalid(format!(
                "unsupported Qwen4 matrix quant dtype {dtype:?}"
            )));
        }
    };
    if require_aligned_k && k % group_size != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "matrix K={k} is not aligned to quantizer group {group_size}"
        )));
    }
    let groups = if require_aligned_k {
        k / group_size
    } else {
        k.checked_add(group_size - 1)
            .and_then(|rounded| rounded.checked_div(group_size))
            .ok_or_else(|| {
                Qwen4Error::Invalid("quantized matrix group count overflows".to_string())
            })?
    };
    rows.checked_mul(groups)
        .and_then(|groups| groups.checked_mul(group_bytes))
        .ok_or_else(|| Qwen4Error::Invalid("quantized matrix byte length overflows".to_string()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    Bf16,
    /// A packed matrix: rank-2, or rank-3 for a routed expert block, whose
    /// leading dimensions flatten into the row count the format streams.  The
    /// dtype is the manifest's declaration, never re-derived here — a planner
    /// that second-guessed it would pack a geometry the loader cannot decode.
    Quant(DType),
    Ple,
    I64(I64Role),
}

#[derive(Clone, Debug)]
struct PlannedEntry {
    source: SourceTensor,
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_len: u64,
    kind: EntryKind,
}
#[derive(Clone, Debug)]
struct PleMetadata {
    multipliers: Vec<i64>,
    vocab_sizes: Vec<i64>,
    prefix_offsets: Vec<i64>,
}

#[derive(Debug)]
struct EntryPlan {
    entries: Vec<PlannedEntry>,
    ple_metadata: Option<PleMetadata>,
    row_chunk: usize,
    resident_entries: usize,
    expert_entries: usize,
    resident_bytes: u64,
    external_ple_bytes: u64,
    ple_shards: usize,
}

fn read_i64_array(tensor: &SourceTensor, expected_len: usize) -> Result<Vec<i64>, Qwen4Error> {
    if tensor.dtype != "I64" {
        return Err(Qwen4Error::Invalid(format!(
            "{} has dtype {}, expected I64 metadata",
            tensor.name, tensor.dtype
        )));
    }
    if tensor.shape != [expected_len as u64] {
        return Err(Qwen4Error::Invalid(format!(
            "{} has shape {:?}, expected [{expected_len}]",
            tensor.name, tensor.shape
        )));
    }
    let expected_bytes = (expected_len as u64)
        .checked_mul(8)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} I64 byte length overflows", tensor.name)))?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    let mut bytes = vec![0u8; expected_bytes as usize];
    tensor.read_range(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)")))
        .collect())
}

fn is_prime(value: i64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3i64;
    while divisor <= value / divisor {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn validate_ple_metadata(
    multipliers: &[i64],
    vocab_sizes: &[i64],
    prefix_offsets: &[i64],
) -> Result<u64, Qwen4Error> {
    if multipliers.len() != 3 {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE multipliers must have length 3, got {}",
            multipliers.len()
        )));
    }
    if vocab_sizes.len() != PLE_HEAD_COUNT || prefix_offsets.len() != PLE_HEAD_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE metadata requires {PLE_HEAD_COUNT} vocab sizes and offsets"
        )));
    }
    let mut valid_rows = 0u64;
    for (index, &vocab_size) in vocab_sizes.iter().enumerate() {
        if vocab_size <= 0 || !is_prime(vocab_size) {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 PLE vocabulary size at head {index} is not a positive prime: {vocab_size}"
            )));
        }
        valid_rows = valid_rows.checked_add(vocab_size as u64).ok_or_else(|| {
            Qwen4Error::Invalid("Qwen4 PLE valid row count overflows".to_string())
        })?;
        let expected_offset = if index == 0 {
            0
        } else {
            prefix_offsets[index - 1]
                .checked_add(vocab_sizes[index - 1])
                .ok_or_else(|| {
                    Qwen4Error::Invalid("Qwen4 PLE prefix offset overflows i64".to_string())
                })?
        };
        if prefix_offsets[index] != expected_offset {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 PLE offset {index} is {}, expected {expected_offset}",
                prefix_offsets[index]
            )));
        }
    }
    if prefix_offsets[0] != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE first prefix offset is {}, expected 0",
            prefix_offsets[0]
        )));
    }
    let final_end = prefix_offsets[PLE_HEAD_COUNT - 1]
        .checked_add(vocab_sizes[PLE_HEAD_COUNT - 1])
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE final offset overflows i64".to_string()))?;
    if final_end < 0 || final_end as u64 != valid_rows {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE final offset {final_end} does not equal valid rows {valid_rows}"
        )));
    }
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE physical row count overflows".to_string()))?;
    if valid_rows > physical_rows {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE valid rows {valid_rows} exceed physical rows {physical_rows}"
        )));
    }

    Ok(valid_rows)
}
fn validate_manifest_inventory_names(
    tensors: &[SourceTensor],
    inventory: &ManifestIndex,
) -> Result<(), Qwen4Error> {
    let mut seen_metadata = BTreeSet::new();
    let mut seen_required = BTreeSet::new();
    let mut seen_aliases = BTreeSet::new();
    for tensor in tensors {
        if inventory.metadata.contains_key(&tensor.name) {
            if !seen_metadata.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 I64 metadata record {}",
                    tensor.name
                )));
            }
        } else if inventory.required.contains_key(&tensor.name) {
            if !seen_required.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 manifest source record {}",
                    tensor.name
                )));
            }
        } else if inventory.aliases.contains_key(&tensor.name) {
            if !seen_aliases.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 manifest alias record {}",
                    tensor.name
                )));
            }
        } else if !is_vision_tensor(&tensor.name) {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 source tensor {} is not declared by the manifest",
                tensor.name
            )));
        }
    }
    let missing_metadata: Vec<_> = inventory
        .metadata
        .keys()
        .filter(|name| !seen_metadata.contains(*name))
        .map(String::as_str)
        .collect();
    if !missing_metadata.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source is missing manifest metadata records {missing_metadata:?}"
        )));
    }
    let missing_required: Vec<_> = inventory
        .required
        .keys()
        .filter(|name| !seen_required.contains(*name))
        .map(String::as_str)
        .collect();
    if !missing_required.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source is missing {} manifest records: {missing_required:?}",
            missing_required.len()
        )));
    }
    Ok(())
}

fn validate_manifest_source(
    tensor: &SourceTensor,
    expected: &ManifestExpectation,
) -> Result<(), Qwen4Error> {
    if tensor.dtype != "BF16" {
        return Err(Qwen4Error::Invalid(format!(
            "{} has dtype {}, expected source BF16",
            tensor.name, tensor.dtype
        )));
    }
    if tensor.shape != expected.shape {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {:?}, expected manifest shape {:?}",
            tensor.name, tensor.shape, expected.shape
        )));
    }
    let expected_bytes = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    Ok(())
}

fn plan_entries(
    tensors: Vec<SourceTensor>,
    row_chunk: usize,
    manifest: &Qwen4Manifest,
) -> Result<EntryPlan, Qwen4Error> {
    let inventory = ManifestIndex::build(manifest)?;
    validate_manifest_inventory_names(&tensors, &inventory)?;
    let mut metadata_sources: BTreeMap<I64Role, SourceTensor> = BTreeMap::new();
    let mut ple_sources: BTreeMap<usize, SourceTensor> = BTreeMap::new();
    let mut metadata_entries = Vec::new();
    let mut resident = Vec::new();
    let mut seen_metadata = BTreeSet::new();
    let mut seen_required = BTreeSet::new();
    let mut seen_aliases = BTreeSet::new();
    let mut expert_entries = 0usize;
    let mut gate_up_count = 0usize;
    let mut down_count = 0usize;
    let mut mtp_gate_up = false;
    let mut mtp_down = false;

    for tensor in tensors {
        if let Some(expected) = inventory.metadata.get(&tensor.name) {
            if !seen_metadata.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 I64 metadata record {}",
                    tensor.name
                )));
            }
            if tensor.dtype != "I64" {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has dtype {}, expected manifest I64 metadata",
                    tensor.name, tensor.dtype
                )));
            }
            if tensor.shape != expected.shape {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has {:?}, expected manifest metadata shape {:?}",
                    tensor.name, tensor.shape, expected.shape
                )));
            }
            let data_len = checked_product(&tensor.shape, &tensor.name)?
                .checked_mul(8)
                .ok_or_else(|| {
                    Qwen4Error::Invalid(format!("{} I64 payload length overflows", tensor.name))
                })?;
            if tensor.data_len() != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has {} payload bytes, expected {}",
                    tensor.name,
                    tensor.data_len(),
                    data_len
                )));
            }
            metadata_sources.insert(expected.role, tensor);
            continue;
        }

        if let Some(expected) = inventory.required.get(&tensor.name) {
            if !seen_required.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 manifest source record {}",
                    tensor.name
                )));
            }
            validate_manifest_source(&tensor, expected)?;
            match expected.role {
                ManifestRole::Ple => {
                    let index = expected
                        .ple_index
                        .expect("manifest PLE role has a numeric shard index");
                    if ple_sources.insert(index, tensor).is_some() {
                        return Err(Qwen4Error::Invalid(format!(
                            "duplicate Qwen4 PLE shard index {index}"
                        )));
                    }
                }
                ManifestRole::GateUp(dtype) | ManifestRole::Down(dtype) => {
                    let kind = match expected.role {
                        ManifestRole::GateUp(_) => ExpertKind::GateUp,
                        ManifestRole::Down(_) => ExpertKind::Down,
                        ManifestRole::Bf16 | ManifestRole::Matrix(_) | ManifestRole::Ple => {
                            unreachable!("matched expert manifest role")
                        }
                    };
                    let (rows, k) = validate_expert_shape(&tensor, kind, &expected.shape)?;
                    let (quant_type, group_size) = matrix_quant_type(dtype)?;
                    let data_len = quantized_data_len_for_dtype(dtype, rows, k)?;
                    match kind {
                        ExpertKind::GateUp => {
                            gate_up_count += 1;
                            mtp_gate_up |= expected.is_mtp;
                        }
                        ExpertKind::Down => {
                            down_count += 1;
                            mtp_down |= expected.is_mtp;
                        }
                    }
                    let shape = shape_u32(&expected.shape, &tensor.name)?;
                    expert_entries += 1;
                    resident.push(PlannedEntry {
                        source: tensor,
                        name: String::new(),
                        quant_type,
                        shape,
                        group_size,
                        data_len,
                        kind: EntryKind::Quant(dtype),
                    });
                }
                ManifestRole::Matrix(dtype) => {
                    let (rows, k) = validate_matrix_shape(&tensor, &expected.shape)?;
                    let expected_len = quantized_data_len_for_dtype(dtype, rows, k)?;
                    let shape = shape_u32(&expected.shape, &tensor.name)?;
                    resident.push(PlannedEntry {
                        source: tensor,
                        name: String::new(),
                        quant_type: matrix_quant_type(dtype)?.0,
                        shape,
                        group_size: matrix_quant_type(dtype)?.1,
                        data_len: expected_len,
                        kind: EntryKind::Quant(dtype),
                    });
                }
                ManifestRole::Bf16 => {
                    let expected_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
                    let shape = shape_u32(&expected.shape, &tensor.name)?;
                    resident.push(PlannedEntry {
                        source: tensor,
                        name: String::new(),
                        quant_type: 16,
                        shape,
                        group_size: 0,
                        data_len: expected_len,
                        kind: EntryKind::Bf16,
                    });
                }
            }
            continue;
        }

        if let Some(expected) = inventory.aliases.get(&tensor.name) {
            if !seen_aliases.insert(tensor.name.clone()) {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 manifest alias record {}",
                    tensor.name
                )));
            }
            validate_manifest_source(&tensor, expected)?;
            // Tied records are logical aliases, not duplicate source records.
            continue;
        }

        if is_vision_tensor(&tensor.name) {
            // The Qwen4 text artifact intentionally omits only positively
            // identified vision records.
            continue;
        }

        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source tensor {} is not declared by the manifest",
            tensor.name
        )));
    }

    let missing_metadata: Vec<_> = inventory
        .metadata
        .iter()
        .filter(|(name, _)| !seen_metadata.contains(*name))
        .map(|(name, _)| name.as_str())
        .collect();
    if !missing_metadata.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source is missing manifest metadata records {missing_metadata:?}"
        )));
    }
    let missing_required: Vec<_> = inventory
        .required
        .keys()
        .filter(|name| !seen_required.contains(*name))
        .map(String::as_str)
        .collect();
    if !missing_required.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 source is missing {} manifest records: {missing_required:?}",
            missing_required.len()
        )));
    }

    let multipliers = metadata_sources
        .get(&I64Role::Multipliers)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 layer_multipliers I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, 3))?;
    let vocab_sizes = metadata_sources
        .get(&I64Role::VocabSizes)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 ngram_heads_vocab_sizes I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, PLE_HEAD_COUNT))?;
    let prefix_offsets = metadata_sources
        .get(&I64Role::Offsets)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 ngram_heads_offsets I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, PLE_HEAD_COUNT))?;
    let _valid_rows = validate_ple_metadata(&multipliers, &vocab_sizes, &prefix_offsets)?;
    for role in [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets] {
        let source = metadata_sources
            .get(&role)
            .expect("manifest metadata role presence was validated")
            .clone();
        let shape = shape_u32(&source.shape, &source.name)?;
        let data_len = (checked_product(&source.shape, &source.name)?)
            .checked_mul(8)
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} I64 payload length overflows", source.name))
            })?;
        metadata_entries.push(PlannedEntry {
            name: source.name.clone(),
            source,
            quant_type: QWEN4_I64_QUANT_TYPE,
            shape,
            group_size: 0,
            data_len,
            kind: EntryKind::I64(role),
        });
    }

    if ple_sources.len() != PLE_SHARD_COUNT {
        let missing: Vec<usize> = (0..PLE_SHARD_COUNT)
            .filter(|index| !ple_sources.contains_key(index))
            .collect();
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 requires all {PLE_SHARD_COUNT} manifest PLE shards; found {}, missing {:?}",
            ple_sources.len(),
            missing
        )));
    }
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE physical row count overflows".to_string()))?;
    // The source shards are BF16 rows; the artifact stores Q8F16 rows. Both
    // byte counts cover the same physical table, so each is checked against its
    // own row width rather than against each other.
    let source_ple_bytes = physical_rows
        .checked_mul(PLE_ROW_WIDTH)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE source byte count overflows".to_string()))?;
    let external_ple_bytes = physical_rows
        .checked_mul(PLE_ENCODED_ROW_BYTES)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE payload byte count overflows".to_string()))?;
    let mapped_ple_bytes = ple_sources.values().try_fold(0u64, |sum, source| {
        sum.checked_add(source.data_len())
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE source byte sum overflows".to_string()))
    })?;
    if mapped_ple_bytes != source_ple_bytes {
        return Err(Qwen4Error::Invalid(
            "Qwen4 PLE shard payload lengths do not sum to the BF16 physical table".to_string(),
        ));
    }
    if gate_up_count == 0 || down_count == 0 {
        return Err(Qwen4Error::Invalid(
            "Qwen4 input is missing routed gate/up or down experts".to_string(),
        ));
    }
    if !mtp_gate_up || !mtp_down {
        return Err(Qwen4Error::Invalid(
            "Qwen4 input is missing native MTP gate/up and down experts".to_string(),
        ));
    }
    if row_chunk == 0 {
        return Err(Qwen4Error::Invalid(
            "Qwen4 row chunk cannot be zero".to_string(),
        ));
    }

    for entry in &mut resident {
        entry.name = entry.source.name.clone();
    }
    resident.sort_by_key(|entry| {
        inventory
            .manifest_order
            .get(&entry.name)
            .copied()
            .unwrap_or(usize::MAX)
    });

    let ple_metadata = PleMetadata {
        multipliers,
        vocab_sizes,
        prefix_offsets,
    };
    let mut ple_entries = Vec::with_capacity(PLE_SHARD_COUNT);
    for index in 0..PLE_SHARD_COUNT {
        let source = ple_sources.remove(&index).expect("validated PLE index set");
        let shape = shape_u32(&source.shape, &source.name)?;
        ple_entries.push(PlannedEntry {
            name: source.name.clone(),
            source,
            quant_type: Q8F16_QUANT_TYPE,
            shape,
            group_size: Q8F16_BLOCK_SIZE as u32,
            data_len: PLE_ROWS_PER_SHARD
                .checked_mul(PLE_ENCODED_ROW_BYTES)
                .ok_or_else(|| {
                    Qwen4Error::Invalid("Qwen4 PLE entry length overflows".to_string())
                })?,
            kind: EntryKind::Ple,
        });
    }

    let resident_bytes = resident.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.data_len)
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 resident byte count overflows".to_string()))
    })?;
    // A production artifact whose manifest declares no packed rank-2 matrix
    // would decode every trunk projection through the unpacked BF16 path.  That
    // is a recipe error rather than a variant, so it cannot be produced silently.
    if !resident
        .iter()
        .any(|entry| matches!(entry.kind, EntryKind::Quant(_)))
    {
        return Err(Qwen4Error::Invalid(
            "Qwen4 production plan packs no rank-2 matrix: the arch manifest declares \
             every trunk matrix BF16, which is the unpacked decode path"
                .to_string(),
        ));
    }
    let resident_entries = resident.len();
    let mut entries = ple_entries;
    entries.extend(resident);
    entries.extend(metadata_entries);
    Ok(EntryPlan {
        entries,
        ple_metadata: Some(ple_metadata),
        row_chunk,
        resident_entries,
        expert_entries,
        resident_bytes,
        external_ple_bytes,
        ple_shards: PLE_SHARD_COUNT,
    })
}

/// The trunk formats this plan actually wrote, read back from the plan rather
/// than restated: a hardcoded recipe can disagree with the artifact it
/// describes, and a redline preflight that reads it would then be blind.
fn plan_matrix_recipe(plan: &EntryPlan) -> Value {
    let mut formats = BTreeMap::<u8, (String, u64)>::new();
    for entry in &plan.entries {
        if !matches!(entry.kind, EntryKind::Quant(_)) {
            continue;
        }
        let format = match entry.quant_type {
            MQ4G256V2_QUANT_TYPE => "MQ4G256V2",
            MQ4G128V2_QUANT_TYPE => "MQ4G128V2",
            MQ6G256V2_QUANT_TYPE => "MQ6G256V2",
            MFP4G32E8SOA_QUANT_TYPE => "MFP4G32E8SOA",
            _ => "unknown",
        };
        formats.insert(
            entry.quant_type,
            (format.to_string(), u64::from(entry.group_size)),
        );
    }
    let listed: Vec<Value> = formats
        .iter()
        .map(|(quant_type, (format, group_size))| {
            json!({"quant_type": quant_type, "format": format, "group_size": group_size})
        })
        .collect();
    json!({"trunk": listed})
}

fn build_metadata(
    config: Option<&Value>,
    plan: &EntryPlan,
    ple: &PleMetadata,
    tokenizer: Option<&TokenizerMetadata>,
) -> Result<String, Qwen4Error> {
    let _valid_rows =
        validate_ple_metadata(&ple.multipliers, &ple.vocab_sizes, &ple.prefix_offsets)?;
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 physical PLE rows overflow".to_string()))?;

    let mut root = Map::new();
    root.insert("format".to_string(), Value::String("hfqm".to_string()));
    root.insert("format_version".to_string(), json!(1));
    root.insert("arch_id".to_string(), json!(QWEN4_ARCH_ID));
    root.insert(
        "model_type".to_string(),
        Value::String("qwen4_exp".to_string()),
    );
    root.insert(
        "config".to_string(),
        config
            .cloned()
            .unwrap_or_else(|| json!({"model_type": "qwen4_exp"})),
    );
    root.insert(
        "qwen4_recipe".to_string(),
        json!({
            "version": 2,
            "stacked_experts": true,
            "routing": {"num_experts": ROUTED_EXPERTS, "top_k": ROUTER_TOP_K},
            "matrix": plan_matrix_recipe(plan),
            "gate_up": {"quant_type": MQ4G256V2_QUANT_TYPE, "format": "MQ4G256V2", "group_size": MQ4G256V2_GROUP_SIZE, "k": HIDDEN_WIDTH},
            "down": {"quant_type": MQ4G128V2_QUANT_TYPE, "format": "MQ4G128V2", "group_size": MQ4G128V2_GROUP_SIZE, "k": DOWN_INTERMEDIATE},
            // The eight-bit classes: the token embedding and the language head,
            // which every MoE recipe in this tree holds at Q8F16 rather than at
            // a four-bit matrix tier.
            "eight_bit": ["embed_tokens.weight", "lm_head.weight"],
            // Parameters that keep source BF16 bytes; they are covered by
            // neither format above.
            "source_exact": [
                "hyper_connection",
                "mtp.fc_embedding.weight",
                "mtp.fc_hidden.weight"
            ],
            "mtp_experts": "same_as_trunk"
        }),
    );
    root.insert(
        "qwen4_ple".to_string(),
        json!({
            "version": QWEN4_PLE_VERSION,
            "multipliers": ple.multipliers.clone(),
            "head_vocab_sizes": ple.vocab_sizes.clone(),
            "head_offsets": ple.prefix_offsets.clone(),
            "padded_rows": physical_rows,
        }),
    );
    root.insert(
        "qwen4_streaming".to_string(),
        json!({
            "version": 1,
            "row_chunk": plan.row_chunk,
            "resident_entries": plan.resident_entries,
            "expert_entries": plan.expert_entries,
            "external_ple_entries": plan.ple_shards,
        }),
    );
    if let Some(tokenizer) = tokenizer {
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
    }
    serde_json::to_string(&Value::Object(root))
        .map_err(|error| Qwen4Error::json("serialize Qwen4 metadata", error))
}

fn decode_bf16(raw: &[u8], values: &mut Vec<f32>) -> Result<(), Qwen4Error> {
    if raw.len() % 2 != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "BF16 source chunk has odd byte length {}",
            raw.len()
        )));
    }
    values.clear();
    values.reserve(raw.len() / 2);
    for bytes in raw.chunks_exact(2) {
        values.push(bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    }
    Ok(())
}

fn stream_raw_rows(
    tensor: &SourceTensor,
    row_width: u64,
    element_bytes: u64,
    row_chunk: usize,
    scratch: &mut ScratchTracker,
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    let rows = if tensor.shape.len() <= 1 {
        1
    } else {
        tensor.shape[0]
    };
    let expected_elements = if tensor.shape.len() <= 1 {
        checked_product(&tensor.shape, &tensor.name)?
    } else {
        checked_product(&tensor.shape[1..], &tensor.name)?
    };
    if expected_elements != row_width {
        return Err(Qwen4Error::Invalid(format!(
            "{} row width {row_width} does not match shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    let row_bytes = row_width
        .checked_mul(element_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row bytes overflow", tensor.name)))?;
    let max_chunk_bytes = (row_chunk as u64)
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name)))?;
    if max_chunk_bytes > MAX_CHUNK_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{} logical-row chunk is {max_chunk_bytes} bytes, exceeds bounded {MAX_CHUNK_BYTES}-byte staging",
            tensor.name
        )));
    }
    let expected_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} payload length overflow", tensor.name)))?;
    if expected_len != tensor.data_len() {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_len
        )));
    }
    let mut raw = Vec::new();
    let mut row = 0u64;
    while row < rows {
        let count = (rows - row).min(row_chunk as u64);
        let bytes = count.checked_mul(row_bytes).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name))
        })?;
        let bytes_usize = usize::try_from(bytes).map_err(|_| {
            Qwen4Error::Invalid(format!("{} chunk does not fit usize", tensor.name))
        })?;
        raw.resize(bytes_usize, 0);
        let offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} row offset overflows", tensor.name)))?;
        tensor.read_range(offset, &mut raw)?;
        scratch.observe(raw.len(), 0, 0)?;
        writer
            .write_all(&raw)
            .map_err(|error| Qwen4Error::io(format!("write {}", tensor.name), error))?;
        row += count;
    }
    Ok(())
}

fn stream_quantized_rows(
    tensor: &SourceTensor,
    dtype: DType,
    rows: u64,
    k: u64,
    row_chunk: usize,
    expert_rows: Option<u64>,
    signs1_256: &[f32],
    signs2_256: &[f32],
    signs1_128: &[f32],
    signs2_128: &[f32],
    scratch: &mut ScratchTracker,
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    let row_bytes = k
        .checked_mul(2)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row bytes overflow", tensor.name)))?;
    let requested_rows = if let Some(expert_rows) = expert_rows {
        if expert_rows == 0 || rows % expert_rows != 0 {
            return Err(Qwen4Error::Invalid(format!(
                "{} expert rows {expert_rows} do not partition {rows} logical rows",
                tensor.name
            )));
        }
        let requested = row_chunk as u64;
        if requested < expert_rows {
            expert_rows
        } else {
            requested / expert_rows * expert_rows
        }
    } else {
        row_chunk as u64
    };
    let candidate_rows = requested_rows.min(rows);
    let bounded_rows = bounded_rows_for_scratch(dtype, k, candidate_rows)?;
    let rows_per_chunk = if let Some(expert_rows) = expert_rows {
        bounded_rows / expert_rows * expert_rows
    } else {
        bounded_rows
    };
    if rows_per_chunk == 0 {
        return Err(Qwen4Error::Invalid(format!(
            "{} cannot fit one logical row within bounded {MAX_CHUNK_BYTES}-byte scratch",
            tensor.name
        )));
    }
    let max_chunk_scratch = chunk_scratch_bytes(dtype, rows_per_chunk, k)?;
    if max_chunk_scratch > MAX_CHUNK_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{} scratch high-water {max_chunk_scratch} exceeds bounded {MAX_CHUNK_BYTES} bytes",
            tensor.name
        )));
    }
    let expected_source = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} source length overflow", tensor.name)))?;
    if expected_source != tensor.data_len() {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_source
        )));
    }
    let mut raw = Vec::new();
    let mut values = Vec::new();
    let mut row = 0u64;
    let k_usize = usize::try_from(k)
        .map_err(|_| Qwen4Error::Invalid(format!("{} K does not fit usize", tensor.name)))?;
    while row < rows {
        let count = (rows - row).min(rows_per_chunk);
        let bytes = count.checked_mul(row_bytes).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name))
        })?;
        let bytes_usize = usize::try_from(bytes).map_err(|_| {
            Qwen4Error::Invalid(format!("{} chunk does not fit usize", tensor.name))
        })?;
        raw.resize(bytes_usize, 0);
        let offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} row offset overflows", tensor.name)))?;
        tensor.read_range(offset, &mut raw)?;
        decode_bf16(&raw, &mut values)?;
        let count_usize = usize::try_from(count).map_err(|_| {
            Qwen4Error::Invalid(format!("{} row count does not fit usize", tensor.name))
        })?;
        let quantized = match dtype {
            DType::MQ4G256V2 => {
                quantize_mq4g256v2(&values, count_usize, k_usize, signs1_256, signs2_256)
            }
            DType::MQ4G128V2 => {
                quantize_mq4g128v2(&values, count_usize, k_usize, signs1_128, signs2_128)
                    .map_err(Qwen4Error::Invalid)?
            }
            DType::MQ6G256V2 => {
                quantize_mq6g256v2(&values, count_usize, k_usize, signs1_256, signs2_256)
            }
            DType::Q8_0 => quantize_q8f16(&values),
            DType::MFP4G32E8SOA => {
                quantize_mfp4g32_e8_soa_2d(&values, count_usize, k_usize, signs1_256, signs2_256)
            }
            _ => {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has unsupported Qwen4 quant dtype {dtype:?}",
                    tensor.name
                )));
            }
        };
        scratch.observe(
            raw.capacity(),
            values.capacity() * std::mem::size_of::<f32>(),
            quantized.len(),
        )?;
        writer
            .write_all(&quantized)
            .map_err(|error| Qwen4Error::io(format!("write {}", tensor.name), error))?;
        row += count;
    }
    Ok(())
}

fn stream_entry(
    entry: &PlannedEntry,
    row_chunk: usize,
    signs1_256: &[f32],
    signs2_256: &[f32],
    signs1_128: &[f32],
    signs2_128: &[f32],
    scratch: &mut ScratchTracker,
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    match entry.kind {
        EntryKind::Bf16 => {
            let row_width = if entry.source.shape.len() <= 1 {
                checked_product(&entry.source.shape, &entry.source.name)?
            } else {
                checked_product(&entry.source.shape[1..], &entry.source.name)?
            };
            stream_raw_rows(&entry.source, row_width, 2, row_chunk, scratch, writer)
        }
        EntryKind::Ple => {
            // External PLE shards are stored as Q8F16 rows: the source rows are
            // BF16, and `PLE_ROW_WIDTH` is a multiple of the block, so each row
            // quantizes independently inside a chunk-wide pass.
            let rows = entry.source.shape.first().copied().ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} PLE shard has no rows", entry.name))
            })?;
            stream_quantized_rows(
                &entry.source,
                DType::Q8_0,
                rows,
                PLE_ROW_WIDTH,
                EXTERNAL_PLE_ROW_CHUNK,
                None,
                signs1_256,
                signs2_256,
                signs1_128,
                signs2_128,
                scratch,
                writer,
            )
        }
        EntryKind::I64(_) => {
            let row_width = if entry.source.shape.len() <= 1 {
                checked_product(&entry.source.shape, &entry.source.name)?
            } else {
                checked_product(&entry.source.shape[1..], &entry.source.name)?
            };
            stream_raw_rows(&entry.source, row_width, 8, row_chunk, scratch, writer)
        }
        EntryKind::Quant(dtype) => {
            // Rank-2 is a plain matrix; rank-3 is a routed expert block that the
            // format flattens into rows-by-K, keeping each expert's rows
            // contiguous — exactly the slicing the loader performs when it hands
            // each expert a byte view of the stacked tensor.
            let rank = entry.source.shape.len();
            if rank < 2 {
                return Err(Qwen4Error::Invalid(format!(
                    "{} {dtype:?} stream requires a matrix source shape, got {:?}",
                    entry.name, entry.source.shape
                )));
            }
            let rows = checked_product(&entry.source.shape[..rank - 1], &entry.name)?;
            let k = entry.source.shape[rank - 1];
            let expert_rows = (rank == 3).then(|| entry.source.shape[1]);
            stream_quantized_rows(
                &entry.source,
                dtype,
                rows,
                k,
                row_chunk,
                expert_rows,
                signs1_256,
                signs2_256,
                signs1_128,
                signs2_128,
                scratch,
                writer,
            )
        }
    }
}

/// Bounded, mmap-free reopen plan used both by the producer and by fixture
/// tests.  It is intentionally a plan rather than a payload reader: loading
#[derive(Debug)]
pub(crate) struct Qwen4ReopenPlan {
    pub(crate) arch_id: u32,
    pub(crate) metadata_json: String,
    pub(crate) entries: Vec<Qwen4PlanEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen4PlanEntry {
    pub(crate) name: String,
    pub(crate) quant_type: u8,
    pub(crate) shape: Vec<u32>,
    pub(crate) group_size: u32,
    pub(crate) data_offset: u64,
    pub(crate) data_len: u64,
}

impl Qwen4ReopenPlan {
    pub(crate) fn open(path: &Path) -> Result<Self, Qwen4Error> {
        let file =
            File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
        let file_len = file
            .metadata()
            .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
            .len();
        if file_len < 32 {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} is shorter than its 32-byte header",
                path.display()
            )));
        }
        let mut header = [0u8; 32];
        read_exact_at(&file, 0, &mut header).map_err(|error| {
            Qwen4Error::io(
                format!("read Qwen4 artifact header {}", path.display()),
                error,
            )
        })?;
        if &header[..4] != b"HFQM" {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} has wrong HFQM magic",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("header slice"));
        if version != 1 {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} has unsupported version {version}",
                path.display()
            )));
        }
        let arch_id = u32::from_le_bytes(header[8..12].try_into().expect("header slice"));
        if arch_id != QWEN4_ARCH_ID {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact has arch id {arch_id}, expected {QWEN4_ARCH_ID}"
            )));
        }
        let tensor_count =
            u32::from_le_bytes(header[12..16].try_into().expect("header slice")) as usize;
        let metadata_offset = u64::from_le_bytes(header[16..24].try_into().expect("header slice"));
        let data_offset = u64::from_le_bytes(header[24..32].try_into().expect("header slice"));
        if metadata_offset < 32 || metadata_offset > data_offset || data_offset > file_len {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact has invalid metadata/data offsets {metadata_offset}/{data_offset} for file {file_len}"
            )));
        }
        let region_len = data_offset - metadata_offset;
        if region_len > MAX_REOPEN_REGION_BYTES {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 metadata/index region is {region_len} bytes, exceeds {MAX_REOPEN_REGION_BYTES}"
            )));
        }
        let region_len_usize = usize::try_from(region_len).map_err(|_| {
            Qwen4Error::Invalid("Qwen4 metadata/index region does not fit usize".to_string())
        })?;
        let mut region = vec![0u8; region_len_usize];
        read_exact_at(&file, metadata_offset, &mut region).map_err(|error| {
            Qwen4Error::io(
                format!("read Qwen4 metadata/index {}", path.display()),
                error,
            )
        })?;
        let json_end = json_blob_end(&region).ok_or_else(|| {
            Qwen4Error::Invalid("Qwen4 metadata JSON is not terminated".to_string())
        })?;
        let metadata_json = String::from_utf8(region[..json_end].to_vec()).map_err(|error| {
            Qwen4Error::Invalid(format!("Qwen4 metadata is not UTF-8: {error}"))
        })?;
        let metadata: Value = serde_json::from_str(&metadata_json)
            .map_err(|error| Qwen4Error::json("parse reopened Qwen4 metadata", error))?;
        validate_reopened_metadata(&metadata)?;

        let mut pos = json_end;
        let read_u8 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u8, Qwen4Error> {
            if *pos >= region.len() {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = region[*pos];
            *pos += 1;
            Ok(value)
        };
        let read_u16 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u16, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 2 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u16::from_le_bytes(region[*pos..*pos + 2].try_into().expect("index slice"));
            *pos += 2;
            Ok(value)
        };
        let read_u32 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u32, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 4 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u32::from_le_bytes(region[*pos..*pos + 4].try_into().expect("index slice"));
            *pos += 4;
            Ok(value)
        };
        let read_u64 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u64, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 8 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u64::from_le_bytes(region[*pos..*pos + 8].try_into().expect("index slice"));
            *pos += 8;
            Ok(value)
        };
        let index_count = read_u32(&region, &mut pos, "tensor count")? as usize;
        if index_count != tensor_count {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 header has {tensor_count} tensors but index has {index_count}"
            )));
        }
        let mut entries = Vec::with_capacity(index_count);
        let mut names = HashMap::with_capacity(index_count);
        let mut cumulative_offset = data_offset;
        for _ in 0..index_count {
            let name_len = read_u16(&region, &mut pos, "name length")? as usize;
            if region.len().saturating_sub(pos) < name_len {
                return Err(Qwen4Error::Invalid(
                    "Qwen4 index truncated while reading tensor name".to_string(),
                ));
            }
            let name =
                String::from_utf8(region[pos..pos + name_len].to_vec()).map_err(|error| {
                    Qwen4Error::Invalid(format!("Qwen4 tensor name is not UTF-8: {error}"))
                })?;
            pos += name_len;
            let quant_type = read_u8(&region, &mut pos, "quant type")?;
            if !matches!(
                quant_type,
                16 | MQ4G256V2_QUANT_TYPE
                    | MQ4G128V2_QUANT_TYPE
                    | MQ6G256V2_QUANT_TYPE
                    | Q8F16_QUANT_TYPE
                    | MFP4G32E8SOA_QUANT_TYPE
                    | QWEN4_I64_QUANT_TYPE
            ) {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 artifact tensor {name} uses unknown quant type {quant_type}"
                )));
            }
            let n_dims = read_u8(&region, &mut pos, "dimension count")? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u32(&region, &mut pos, "shape dimension")?);
            }
            let group_size = read_u32(&region, &mut pos, "group size")?;
            let data_len = read_u64(&region, &mut pos, "data length")?;
            let end = cumulative_offset.checked_add(data_len).ok_or_else(|| {
                Qwen4Error::Invalid(format!("Qwen4 tensor {name} payload overflows"))
            })?;
            if end > file_len {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 tensor {name} payload [{cumulative_offset},{end}) exceeds file {file_len}"
                )));
            }
            if names.insert(name.clone(), ()).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 output tensor {name}"
                )));
            }
            validate_output_entry_len(&name, quant_type, &shape, data_len)?;
            entries.push(Qwen4PlanEntry {
                name,
                quant_type,
                shape,
                group_size,
                data_offset: cumulative_offset,
                data_len,
            });
            cumulative_offset = end;
        }
        if cumulative_offset != file_len {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact payload ends at {cumulative_offset}, file ends at {file_len}"
            )));
        }
        if pos > region.len() {
            return Err(Qwen4Error::Invalid(
                "Qwen4 index exceeds metadata/data region".to_string(),
            ));
        }
        Ok(Self {
            arch_id,
            metadata_json,
            entries,
        })
    }
}
impl Qwen4ReopenPlan {
    fn validate_against(
        &self,
        metadata_json: &str,
        expected: &[hipfire_runtime::hfq::HfqStreamEntry],
    ) -> Result<(), Qwen4Error> {
        if self.arch_id != QWEN4_ARCH_ID {
            return Err(Qwen4Error::Invalid(format!(
                "reopened Qwen4 architecture id {} disagrees with planned {}",
                self.arch_id, QWEN4_ARCH_ID
            )));
        }
        if self.metadata_json != metadata_json {
            return Err(Qwen4Error::Invalid(
                "reopened Qwen4 metadata disagrees with planned config/recipe".to_string(),
            ));
        }
        if self.entries.len() != expected.len() {
            return Err(Qwen4Error::Invalid(format!(
                "reopened Qwen4 entry count {} disagrees with planned {}",
                self.entries.len(),
                expected.len()
            )));
        }
        let mut end = self
            .entries
            .first()
            .map(|entry| entry.data_offset)
            .unwrap_or(0);
        for (index, (actual, planned)) in self.entries.iter().zip(expected).enumerate() {
            if actual.name != planned.name
                || actual.quant_type != planned.quant_type
                || actual.shape != planned.shape
                || actual.group_size != planned.group_size
                || actual.data_len != planned.data_len
            {
                return Err(Qwen4Error::Invalid(format!(
                    "reopened Qwen4 entry {index} disagrees with planned source record {}",
                    planned.name
                )));
            }
            if actual.data_offset != end {
                return Err(Qwen4Error::Invalid(format!(
                    "reopened Qwen4 entry {} has non-contiguous payload offset {} (expected {end})",
                    actual.name, actual.data_offset
                )));
            }
            end = end.checked_add(actual.data_len).ok_or_else(|| {
                Qwen4Error::Invalid("reopened Qwen4 payload overflows".to_string())
            })?;
        }
        Ok(())
    }
}

fn validate_quantized_output_len(
    name: &str,
    shape: &[u32],
    data_len: u64,
    dtype: DType,
) -> Result<(), Qwen4Error> {
    if shape.len() < 2 || shape.iter().any(|&dimension| dimension == 0) {
        return Err(Qwen4Error::Invalid(format!(
            "{name} {dtype:?} output requires a nonzero rank-2-or-higher matrix shape, got {shape:?}"
        )));
    }
    let shape_u64 = shape
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    let rows = checked_product(&shape_u64[..shape_u64.len() - 1], name)?;
    let k = *shape_u64.last().expect("rank checked above");
    let expected = quantized_data_len_for_dtype(dtype, rows, k)?;
    if expected != data_len {
        return Err(Qwen4Error::Invalid(format!(
            "{name} {dtype:?} payload is {data_len} bytes, expected {expected} for rows={rows} K={k}"
        )));
    }
    Ok(())
}

fn validate_output_entry_len(
    name: &str,
    quant_type: u8,
    shape: &[u32],
    data_len: u64,
) -> Result<(), Qwen4Error> {
    match quant_type {
        QWEN4_I64_QUANT_TYPE => {
            if i64_role(name).is_none() {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 raw-I64 output {name} is not a recognized PLE metadata tensor"
                )));
            }
            let expected_shape: &[u32] = match i64_role(name).expect("role checked above") {
                I64Role::Multipliers => &[3],
                I64Role::VocabSizes | I64Role::Offsets => &[16],
            };
            if shape != expected_shape {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} raw-I64 shape {:?} is not {:?}",
                    shape, expected_shape
                )));
            }
            let expected = checked_product(
                &shape.iter().map(|&value| value as u64).collect::<Vec<_>>(),
                name,
            )?
            .checked_mul(8)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} I64 length overflows")))?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} I64 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        16 => {
            let expected = checked_product(
                &shape.iter().map(|&value| value as u64).collect::<Vec<_>>(),
                name,
            )?
            .checked_mul(2)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} BF16 length overflows")))?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} BF16 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        MQ4G256V2_QUANT_TYPE => {
            validate_quantized_output_len(name, shape, data_len, DType::MQ4G256V2)?;
        }
        MQ4G128V2_QUANT_TYPE => {
            validate_quantized_output_len(name, shape, data_len, DType::MQ4G128V2)?;
        }
        MQ6G256V2_QUANT_TYPE => {
            validate_quantized_output_len(name, shape, data_len, DType::MQ6G256V2)?;
        }
        Q8F16_QUANT_TYPE => {
            validate_quantized_output_len(name, shape, data_len, DType::Q8_0)?;
        }
        MFP4G32E8SOA_QUANT_TYPE => {
            validate_quantized_output_len(name, shape, data_len, DType::MFP4G32E8SOA)?;
        }
        other => {
            return Err(Qwen4Error::Invalid(format!(
                "{name} uses unsupported Qwen4 output quant_type {other}"
            )));
        }
    }
    Ok(())
}

fn validate_reopened_metadata(metadata: &Value) -> Result<(), Qwen4Error> {
    let object = metadata.as_object().ok_or_else(|| {
        Qwen4Error::Invalid("reopened Qwen4 metadata is not an object".to_string())
    })?;
    if object.get("arch_id").and_then(Value::as_u64) != Some(QWEN4_ARCH_ID as u64) {
        return Err(Qwen4Error::Invalid(
            "reopened Qwen4 metadata arch_id is not 16".to_string(),
        ));
    }
    let ple = object
        .get("qwen4_ple")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Qwen4Error::Invalid("reopened Qwen4 metadata is missing qwen4_ple".to_string())
        })?;
    const FIELDS: [&str; 5] = [
        "version",
        "multipliers",
        "head_vocab_sizes",
        "head_offsets",
        "padded_rows",
    ];
    for field in ple.keys() {
        if !FIELDS.contains(&field.as_str()) {
            return Err(Qwen4Error::Invalid(format!(
                "reopened qwen4_ple metadata has unknown field `{field}`"
            )));
        }
    }
    let version = ple.get("version").and_then(Value::as_u64).ok_or_else(|| {
        Qwen4Error::Invalid("reopened qwen4_ple version is not an integer".into())
    })?;
    if version != QWEN4_PLE_VERSION as u64 {
        return Err(Qwen4Error::Invalid(format!(
            "reopened qwen4_ple metadata has unsupported version {version}"
        )));
    }
    let read_i64_array = |field: &'static str, expected: usize| -> Result<Vec<i64>, Qwen4Error> {
        let value = ple.get(field).ok_or_else(|| {
            Qwen4Error::Invalid(format!("reopened qwen4_ple metadata is missing `{field}`"))
        })?;
        let values = value.as_array().ok_or_else(|| {
            Qwen4Error::Invalid(format!("reopened qwen4_ple `{field}` is not an array"))
        })?;
        if values.len() != expected {
            return Err(Qwen4Error::Invalid(format!(
                "reopened qwen4_ple `{field}` has {} values, expected {expected}",
                values.len()
            )));
        }
        values
            .iter()
            .map(|value| {
                value.as_i64().ok_or_else(|| {
                    Qwen4Error::Invalid(format!(
                        "reopened qwen4_ple `{field}` contains a non-i64 value"
                    ))
                })
            })
            .collect()
    };
    let multipliers = read_i64_array("multipliers", 3)?;
    let vocab_sizes = read_i64_array("head_vocab_sizes", PLE_HEAD_COUNT)?;
    let prefix_offsets = read_i64_array("head_offsets", PLE_HEAD_COUNT)?;
    let padded_rows = ple
        .get("padded_rows")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            Qwen4Error::Invalid("reopened qwen4_ple `padded_rows` is not an integer".into())
        })?;
    let valid_rows = validate_ple_metadata(&multipliers, &vocab_sizes, &prefix_offsets)?;
    let expected_padded_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE physical rows overflow".to_string()))?;
    if padded_rows != expected_padded_rows {
        return Err(Qwen4Error::Invalid(format!(
            "reopened qwen4_ple padded_rows is {padded_rows}, expected {expected_padded_rows}"
        )));
    }
    if padded_rows < valid_rows || padded_rows % 128 != 0 {
        return Err(Qwen4Error::Invalid(
            "reopened qwen4_ple padding is inconsistent with valid rows".to_string(),
        ));
    }
    Ok(())
}

fn json_blob_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string && byte == b'\\' {
            escape = true;
            continue;
        }
        if byte == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match byte {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < dst.len() {
        let read = file.seek_read(&mut dst[done..], offset + done as u64)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional read",
            ));
        }
        done += read;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_e8::dequant_mfp4g32_e8_soa;
    use crate::quant_fwht::cpu_fwht_256;
    use rdna_compute::Gpu;
    use std::env;
    use std::net::TcpListener;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    use std::thread::{self, JoinHandle};
    use tempfile::{tempdir, NamedTempFile};

    fn test_remote_source(base_url: &str) -> RemoteSource {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(5)))
            .http_status_as_error(false)
            .build()
            .into();
        RemoteSource {
            spec: RemoteSpec {
                owner: "owner".to_string(),
                repo: "repo".to_string(),
                revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            },
            base_url: base_url.trim_end_matches('/').to_string(),
            agent,
            authorization: None,
            identity: Mutex::new(RemoteIdentityState::default()),
        }
    }

    fn spawn_http_response(
        status: u16,
        content_range: Option<&str>,
        body: &[u8],
        announced_length: Option<usize>,
        assert_range: Option<&str>,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let address = listener.local_addr().expect("mock server address");
        let content_range = content_range.map(str::to_string);
        let body = body.to_vec();
        let announced = announced_length.unwrap_or(body.len());
        let assert_range = assert_range.map(str::to_string);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept mock request");
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).expect("read mock request");
            let request = String::from_utf8_lossy(&request[..count]);
            if let Some(expected) = assert_range {
                assert!(
                    request
                        .lines()
                        .any(|line| line.eq_ignore_ascii_case(&format!("Range: {expected}"))),
                    "request did not contain expected Range header: {request}"
                );
            }
            let reason = match status {
                206 => "Partial Content",
                200 => "OK",
                _ => "Response",
            };
            let mut response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Length: {announced}\r\n\
                     ETag: \"fixture-etag\"\r\n\
                     X-Repo-Commit: 0123456789abcdef0123456789abcdef01234567\r\n"
            );
            if let Some(content_range) = content_range {
                response.push_str(&format!("Content-Range: {content_range}\r\n"));
            }
            response.push_str("Connection: close\r\n\r\n");
            stream
                .write_all(response.as_bytes())
                .expect("write mock headers");
            stream.write_all(&body).expect("write mock body");
        });
        (format!("http://{address}"), handle)
    }

    #[derive(Clone, Copy)]
    struct MockHttpResponse {
        status: u16,
        content_range: Option<&'static str>,
        body: &'static [u8],
        announced_length: usize,
        etag: &'static str,
    }

    fn spawn_http_sequence(
        responses: Vec<MockHttpResponse>,
        assert_range: &str,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let address = listener.local_addr().expect("mock server address");
        let expected_range = assert_range.to_string();
        let handle = thread::spawn(move || {
            for response_spec in responses {
                let (mut stream, _) = listener.accept().expect("accept mock request");
                let mut request = [0u8; 4096];
                let count = stream.read(&mut request).expect("read mock request");
                let request = String::from_utf8_lossy(&request[..count]);
                assert!(
                    request
                        .lines()
                        .any(|line| line.eq_ignore_ascii_case(&format!("Range: {expected_range}"))),
                    "request did not contain expected Range header: {request}"
                );
                let reason = match response_spec.status {
                    206 => "Partial Content",
                    200 => "OK",
                    _ => "Response",
                };
                let mut response = format!(
                    "HTTP/1.1 {} {reason}\r\nContent-Length: {}\r\n\
                     ETag: \"{}\"\r\n\
                     X-Repo-Commit: 0123456789abcdef0123456789abcdef01234567\r\n",
                    response_spec.status, response_spec.announced_length, response_spec.etag
                );
                if let Some(content_range) = response_spec.content_range {
                    response.push_str(&format!("Content-Range: {content_range}\r\n"));
                }

                response.push_str("Connection: close\r\n\r\n");
                stream
                    .write_all(response.as_bytes())
                    .expect("write mock headers");
                stream
                    .write_all(response_spec.body)
                    .expect("write mock body");
            }
        });
        (format!("http://{address}"), handle)
    }
    fn spawn_http_range_counter(
        total: u64,
        expected_requests: usize,
    ) -> (String, JoinHandle<Vec<(u64, u64)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind range counter");
        let address = listener.local_addr().expect("range counter address");
        let handle = thread::spawn(move || {
            let mut ranges = Vec::with_capacity(expected_requests);
            for _ in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("accept range request");
                let mut request = Vec::new();
                loop {
                    let mut chunk = [0u8; 1024];
                    let count = stream.read(&mut chunk).expect("read range request");
                    assert!(count > 0, "range request ended before headers");
                    request.extend_from_slice(&chunk[..count]);
                    assert!(
                        request.len() <= 16 * 1024,
                        "range request headers exceeded test bound"
                    );
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8(request).expect("range request is UTF-8");
                let value = request
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("Range").then(|| value.trim())
                    })
                    .expect("range request header");
                let (start, end) = value
                    .strip_prefix("bytes=")
                    .expect("bytes range")
                    .split_once('-')
                    .expect("range bounds");
                let start = start.parse::<u64>().expect("range start");
                let end = end.parse::<u64>().expect("range end");
                assert!(start <= end && end < total, "range exceeds fixture");
                let length = end - start + 1;
                let response = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {length}\r\n\
                     Content-Range: bytes {start}-{end}/{total}\r\n\
                     ETag: \"fixture-etag\"\r\n\
                     X-Repo-Commit: 0123456789abcdef0123456789abcdef01234567\r\n\
                     Connection: close\r\n\r\n"
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write range headers");
                let mut body = [0u8; 16 * 1024];
                let mut written = 0u64;
                while written < length {
                    let count = (length - written).min(body.len() as u64) as usize;
                    for (index, byte) in body[..count].iter_mut().enumerate() {
                        *byte = ((start + written + index as u64) % 251) as u8;
                    }
                    stream.write_all(&body[..count]).expect("write range body");
                    written += count as u64;
                }
                ranges.push((start, end));
            }
            ranges
        });
        (format!("http://{address}"), handle)
    }

    #[test]
    fn remote_spec_requires_full_immutable_revision() {
        for input in [
            "hf://owner/repo",
            "hf://owner/repo@main",
            "hf://owner/repo@0123456789abcdef",
            "hf://owner/repo@0123456789abcdef0123456789abcdef0123456z",
            "hf://owner/repo@0123456789abcdef0123456789abcdef01234567@main",
        ] {
            assert!(
                parse_remote_spec(input).is_err(),
                "accepted invalid spec {input}"
            );
        }
        let spec = parse_remote_spec("hf://owner/repo@0123456789ABCDEF0123456789abcdef01234567")
            .expect("parse valid remote spec")
            .expect("remote spec");
        assert_eq!(spec.owner, "owner");
        assert_eq!(spec.repo, "repo");
        assert_eq!(spec.revision, "0123456789ABCDEF0123456789abcdef01234567");
        assert!(parse_remote_spec("weights/model.safetensors")
            .expect("local path is not remote")
            .is_none());
    }

    #[test]
    fn remote_url_targets_exact_repo_revision_and_object() {
        let source = test_remote_source("https://huggingface.co/");
        assert_eq!(
            source.url_for("weights/model.safetensors"),
            "https://huggingface.co/owner/repo/resolve/0123456789abcdef0123456789abcdef01234567/weights/model.safetensors"
        );
    }

    #[test]
    fn content_range_parser_requires_explicit_valid_bounds() {
        assert_eq!(parse_content_range("bytes 2-5/8").unwrap(), (2, 5, 8));
        for value in [
            "bytes 2-5/*",
            "bytes 5-2/8",
            "bytes 0-8/8",
            "bytes 0-3/8 extra",
            "items 0-3/8",
            "bytes 0-3",
            "bytes 0-a/8",
        ] {
            assert!(parse_content_range(value).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn remote_range_reader_accepts_exact_partial_response() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 2-5/8"), b"cdef", None, Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert_eq!(
            source
                .read_range("model.safetensors", 2, &mut bytes, Some(8))
                .unwrap(),
            8
        );
        assert_eq!(&bytes, b"cdef");
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_full_response() {
        let (base, handle) = spawn_http_response(200, None, b"cdef", None, None);
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_mismatched_content_range() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 0-3/8"), b"cdef", None, Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_short_body() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 2-5/8"), b"cde", Some(4), Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_retries_peer_disconnect_before_commit() {
        let (base, handle) = spawn_http_sequence(
            vec![
                MockHttpResponse {
                    status: 206,
                    content_range: Some("bytes 2-5/8"),
                    body: b"cd",
                    announced_length: 4,
                    etag: "fixture-etag",
                },
                MockHttpResponse {
                    status: 206,
                    content_range: Some("bytes 2-5/8"),
                    body: b"cdef",
                    announced_length: 4,
                    etag: "fixture-etag",
                },
            ],
            "bytes=2-5",
        );
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert_eq!(
            source
                .read_range("model.safetensors", 2, &mut bytes, Some(8))
                .unwrap(),
            8
        );
        assert_eq!(&bytes, b"cdef");
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_retries_retryable_statuses_before_success() {
        let (base, handle) = spawn_http_sequence(
            vec![
                MockHttpResponse {
                    status: 503,
                    content_range: None,
                    body: b"",
                    announced_length: 0,
                    etag: "fixture-etag",
                },
                MockHttpResponse {
                    status: 429,
                    content_range: None,
                    body: b"",
                    announced_length: 0,
                    etag: "fixture-etag",
                },
                MockHttpResponse {
                    status: 206,
                    content_range: Some("bytes 2-5/8"),
                    body: b"cdef",
                    announced_length: 4,
                    etag: "fixture-etag",
                },
            ],
            "bytes=2-5",
        );
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .unwrap();
        assert_eq!(&bytes, b"cdef");
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_stops_after_bounded_transient_failures() {
        let responses = (0..=REMOTE_RANGE_MAX_RETRIES)
            .map(|_| MockHttpResponse {
                status: 206,
                content_range: Some("bytes 2-5/8"),
                body: b"cd",
                announced_length: 4,
                etag: "fixture-etag",
            })
            .collect();
        let (base, handle) = spawn_http_sequence(responses, "bytes=2-5");
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        let error = source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .unwrap_err();
        assert!(
            error.to_string().contains("Peer disconnected"),
            "unexpected terminal error: {error}"
        );
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_identity_change_without_retry() {
        let (base, handle) = spawn_http_sequence(
            vec![
                MockHttpResponse {
                    status: 206,
                    content_range: Some("bytes 2-5/8"),
                    body: b"cdef",
                    announced_length: 4,
                    etag: "fixture-etag-a",
                },
                MockHttpResponse {
                    status: 206,
                    content_range: Some("bytes 2-5/8"),
                    body: b"cdef",
                    announced_length: 4,
                    etag: "fixture-etag-b",
                },
            ],
            "bytes=2-5",
        );
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .unwrap();
        let error = source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .unwrap_err();
        assert!(
            error.to_string().contains("identity changed"),
            "unexpected identity error: {error}"
        );
        handle.join().unwrap();
    }

    fn source_tensor(bytes: &[u8], shape: Vec<u64>, dtype: &str) -> (NamedTempFile, SourceTensor) {
        let mut file = NamedTempFile::new().expect("temp source");
        file.write_all(bytes).expect("source bytes");
        file.as_file().sync_all().expect("sync source");
        let path = file.path().to_path_buf();
        let source = Arc::new(SourceShard {
            path: path.clone(),
            kind: SourceKind::Local {
                file: Arc::new(file.reopen().expect("reopen source")),
            },
            file_len: bytes.len() as u64,
            local_identity: Some(local_file_identity(&path).expect("source identity")),
        });
        (
            file,
            SourceTensor {
                name: "fixture.tensor".to_string(),
                dtype: dtype.to_string(),
                shape,
                data_start: 0,
                data_end: bytes.len() as u64,
                shard: source,
            },
        )
    }

    fn named_source(
        name: &str,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: &str,
    ) -> (NamedTempFile, SourceTensor) {
        let (file, mut tensor) = source_tensor(bytes, shape, dtype);
        tensor.name = name.to_string();
        (file, tensor)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn qwen4_publication_fallback_errno_classification() {
        assert!(renameat2_needs_hard_link_fallback(
            &io::Error::from_raw_os_error(libc::EINVAL)
        ));
        assert!(renameat2_needs_hard_link_fallback(
            &io::Error::from_raw_os_error(libc::ENOSYS)
        ));
        assert!(renameat2_needs_hard_link_fallback(
            &io::Error::from_raw_os_error(libc::EOPNOTSUPP)
        ));
        assert!(!renameat2_needs_hard_link_fallback(
            &io::Error::from_raw_os_error(libc::ENOENT)
        ));
        assert!(!renameat2_needs_hard_link_fallback(&io::Error::new(
            io::ErrorKind::Other,
            "missing errno",
        )));
    }

    #[test]
    fn qwen4_publication_publishes_complete_temp_without_overwrite() {
        let destination = tempdir().expect("destination tempdir");
        let temporary = destination.path().join(".fixture.tmp");
        let output = destination.path().join("fixture.mq4r");
        fs::write(&temporary, b"complete artifact").expect("temporary artifact");

        publish_qwen4(&temporary, &output).expect("publish artifact");

        assert_eq!(
            fs::read(&output).expect("read published artifact"),
            b"complete artifact"
        );
        assert!(!temporary.exists(), "temporary artifact remained");
    }

    #[test]
    fn qwen4_publication_refuses_existing_destination() {
        let destination = tempdir().expect("destination tempdir");
        let temporary = destination.path().join(".fixture.tmp");
        let output = destination.path().join("fixture.mq4r");
        fs::write(&temporary, b"new artifact").expect("temporary artifact");
        fs::write(&output, b"existing artifact").expect("existing artifact");

        let error = publish_qwen4(&temporary, &output).expect_err("replace was allowed");

        assert!(
            error
                .to_string()
                .contains("refusing to replace existing Qwen4 artifact"),
            "unexpected publication error: {error}"
        );
        assert_eq!(
            fs::read(&temporary).expect("read temporary artifact"),
            b"new artifact"
        );
        assert_eq!(
            fs::read(&output).expect("read existing artifact"),
            b"existing artifact"
        );
    }

    #[test]
    fn compact_transaction_removes_temp_after_terminal_source_failure() {
        let destination = tempdir().expect("destination tempdir");
        let output = destination.path().join("fixture.hfq");
        let (mut ple_file, ple) = named_source(
            "model.ple.ngram_embedding.shard_0.weight",
            &vec![0u8; (PLE_ROW_WIDTH * 2) as usize],
            vec![1, PLE_ROW_WIDTH],
            "BF16",
        );
        let (_resident_file, resident) =
            named_source("fixture.tensor", &[0, 0, 0, 0], vec![2], "BF16");
        let source_path = ple_file.path().to_path_buf();
        let source_identity = local_file_identity(&source_path).expect("source identity");
        let source = SourceSet::Local {
            paths: vec![source_path],
            identities: vec![source_identity],
        };
        ple_file
            .as_file_mut()
            .set_len(0)
            .expect("truncate source after identity seal");
        let options = Qwen4Options::compact_fixture(Path::new("unused"), &output);
        let error = write_compact_artifact(&options, &source, vec![ple, resident], &json!({}))
            .expect_err("truncated source must fail");
        assert!(
            error.to_string().contains("source identity changed"),
            "unexpected terminal error: {error}"
        );
        assert!(!output.exists(), "failed transaction published output");
        let leftovers: Vec<_> = fs::read_dir(destination.path())
            .expect("read destination")
            .map(|entry| entry.expect("destination entry").path())
            .collect();
        assert!(
            leftovers.is_empty(),
            "failed transaction left temporary artifacts: {leftovers:?}"
        );
    }

    fn focused_manifest_index() -> ManifestIndex {
        let mut required = BTreeMap::new();
        required.insert(
            "required.weight".to_string(),
            ManifestExpectation {
                shape: vec![2],
                role: ManifestRole::Bf16,
                is_mtp: false,
                ple_index: None,
            },
        );
        let mut aliases = BTreeMap::new();
        aliases.insert(
            "mtp.embed_tokens.weight".to_string(),
            ManifestExpectation {
                shape: vec![2],
                role: ManifestRole::Bf16,
                is_mtp: true,
                ple_index: None,
            },
        );
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "metadata.i64".to_string(),
            MetadataExpectation {
                role: I64Role::Multipliers,
                shape: vec![1],
            },
        );
        ManifestIndex {
            required,
            aliases,
            metadata,
            ple_names: BTreeMap::new(),
            manifest_order: HashMap::new(),
        }
    }

    #[test]
    fn manifest_inventory_rejects_missing_source_records() {
        let inventory = focused_manifest_index();
        let (_metadata_file, metadata) = named_source("metadata.i64", &[0; 8], vec![1], "I64");
        let error = validate_manifest_inventory_names(&[metadata], &inventory).unwrap_err();
        assert!(error.to_string().contains("missing 1 manifest records"));
    }

    #[test]
    fn manifest_inventory_rejects_duplicate_records() {
        let inventory = focused_manifest_index();
        let (_metadata_file, metadata) = named_source("metadata.i64", &[0; 8], vec![1], "I64");
        let (_first_file, first) = named_source("required.weight", &[0; 4], vec![2], "BF16");
        let (_second_file, second) = named_source("required.weight", &[0; 4], vec![2], "BF16");
        let error =
            validate_manifest_inventory_names(&[metadata, first, second], &inventory).unwrap_err();
        assert!(error
            .to_string()
            .contains("duplicate Qwen4 manifest source record"));
    }

    #[test]
    fn manifest_inventory_excludes_only_positive_vision_records() {
        let inventory = focused_manifest_index();
        let (_metadata_file, metadata) = named_source("metadata.i64", &[0; 8], vec![1], "I64");
        let (_required_file, required) = named_source("required.weight", &[0; 4], vec![2], "BF16");
        let (_vision_file, vision) =
            named_source("model.visual.patch_embed.weight", &[0; 2], vec![1], "BF16");
        validate_manifest_inventory_names(&[metadata, required, vision], &inventory).unwrap();
    }
    #[test]
    fn pinned_inventory_uses_333_positive_vision_names_and_1325_outputs() {
        let positive: Vec<String> = (0..PINNED_VISION_TENSOR_COUNT)
            .map(|index| format!("model.visual.blocks.{index}.attn.weight"))
            .collect();
        assert_eq!(
            positive
                .iter()
                .filter(|name| is_vision_tensor(name))
                .count(),
            PINNED_VISION_TENSOR_COUNT
        );
        for name in [
            "model.language_model.visual_projection.weight",
            "model.visualization.weight",
            "model.language_model.layers.0.attn.weight",
        ] {
            assert!(
                !is_vision_tensor(name),
                "misclassified non-vision name {name}"
            );
        }
        assert_eq!(
            PINNED_SOURCE_TENSOR_COUNT - PINNED_VISION_TENSOR_COUNT,
            PINNED_OUTPUT_ENTRY_COUNT
        );
    }

    #[test]
    fn manifest_inventory_does_not_require_tied_alias_source() {
        let inventory = focused_manifest_index();
        let (_metadata_file, metadata) = named_source("metadata.i64", &[0; 8], vec![1], "I64");
        let (_required_file, required) = named_source("required.weight", &[0; 4], vec![2], "BF16");
        validate_manifest_inventory_names(&[metadata.clone(), required.clone()], &inventory)
            .unwrap();
        let (_alias_file, alias) =
            named_source("mtp.embed_tokens.weight", &[0; 4], vec![2], "BF16");
        validate_manifest_inventory_names(&[metadata, required, alias], &inventory).unwrap();
    }

    #[test]
    fn manifest_source_dtype_is_bf16_even_when_output_role_is_quantized() {
        let expected = ManifestExpectation {
            shape: vec![2, 2, 2],
            role: ManifestRole::GateUp(DType::MFP4G32E8SOA),
            is_mtp: false,
            ple_index: None,
        };
        let (_source_file, source) = named_source("gate_up", &[0; 16], vec![2, 2, 2], "BF16");
        validate_manifest_source(&source, &expected).unwrap();
        let (_output_file, output) = named_source("gate_up", &[0; 16], vec![2, 2, 2], "MQ4G256V2");
        let error = validate_manifest_source(&output, &expected).unwrap_err();
        assert!(error.to_string().contains("expected source BF16"));
    }

    #[test]
    fn qwen4_expert_and_trunk_geometry_follow_the_declared_tiers() {
        // Routed experts: rank-3 blocks flattened to rows-by-K at the group
        // geometry their declared tier uses.
        let gate_up_rows = ROUTED_EXPERTS * GATE_UP_INTERMEDIATE;
        assert_eq!(
            quantized_data_len_for_dtype(DType::MQ4G256V2, gate_up_rows, HIDDEN_WIDTH).unwrap(),
            gate_up_rows * (HIDDEN_WIDTH / 256) * MQ4G256V2_GROUP_BYTES
        );
        let down_rows = ROUTED_EXPERTS * HIDDEN_WIDTH;
        assert_eq!(
            quantized_data_len_for_dtype(DType::MQ4G128V2, down_rows, DOWN_INTERMEDIATE).unwrap(),
            down_rows * (DOWN_INTERMEDIATE / 128) * MQ4G128V2_GROUP_BYTES
        );

        // The E8 lattice format remains a producer capability, and its geometry
        // is `16 + pad16(K/32) + (K/32) * 16` bytes per row.  It is a capability
        // the routed down projection cannot use: the codec rotates and encodes
        // in 256-wide FWHT segments (`groups_per_row = K / 256` in the kernels),
        // and `moe_intermediate_size = 640` is not a multiple of 256.  That is a
        // refusal with a message, not a panic inside the encoder.
        assert_eq!(
            quantized_data_len_for_dtype(DType::MFP4G32E8SOA, 4, HIDDEN_WIDTH).unwrap(),
            4 * (16 + 80 + 80 * 16)
        );
        let refused = quantized_data_len_for_dtype(DType::MFP4G32E8SOA, 4, DOWN_INTERMEDIATE)
            .expect_err("K = 640 is not one FWHT-256 segment");
        assert!(
            refused.to_string().contains("multiple of 256"),
            "the refusal must name the alignment it needs: {refused}"
        );
        assert!(quantized_data_len_for_dtype(DType::MFP4G32E8SOA, 1, 128).is_err());

        // Rank-2 trunk matrices are eight-bit blocks and carry any K.
        assert_eq!(
            quantized_data_len_for_dtype(DType::Q8_0, 2, 129).unwrap(),
            2 * 5 * Q8F16_BLOCK_BYTES
        );
        assert!(validate_expert_shape_stub(&[512, 1280, 2559], ExpertKind::GateUp).is_err());
    }
    #[test]
    fn compact_plan_preserves_nonexpert_bf16_and_quantizes_only_rank3_experts() {
        let (_ple_file, ple) = named_source(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            &vec![0u8; PLE_ROW_WIDTH as usize * 2],
            vec![1, PLE_ROW_WIDTH],
            "BF16",
        );
        let (_q_file, q_proj) = named_source(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &vec![0u8; 2 * 256 * 2],
            vec![2, 256],
            "BF16",
        );
        let (_shared_file, shared_up) = named_source(
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            &vec![0u8; 2 * 256 * 2],
            vec![2, 256],
            "BF16",
        );
        let (_gate_file, gate_up) = named_source(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            &vec![0u8; 2 * 2 * 256 * 2],
            vec![2, 2, 256],
            "BF16",
        );
        let (_down_file, down) = named_source(
            "mtp.layers.0.mlp.experts.down_proj",
            &vec![0u8; 2 * 2 * 128 * 2],
            vec![2, 2, 128],
            "BF16",
        );

        let plan = plan_compact_entries(vec![ple, q_proj, shared_up, gate_up, down], 2)
            .expect("compact plan");
        let entry = |name: &str| {
            plan.entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("missing planned entry {name}"))
        };

        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        ] {
            let entry = entry(name);
            assert_eq!(entry.kind, EntryKind::Bf16, "{name} plan kind");
            assert_eq!(entry.quant_type, 16, "{name} quant type");
            assert_eq!(entry.group_size, 0, "{name} group size");
            assert_eq!(entry.data_len, 2 * 256 * 2, "{name} byte count");
        }

        // PLE shards are external rows and are quantized on write: one 160-value
        // row is five Q8F16 blocks, however many rows the fixture has.
        let ple =
            entry("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight");
        assert_eq!(ple.kind, EntryKind::Ple);
        assert_eq!(ple.quant_type, Q8F16_QUANT_TYPE);
        assert_eq!(ple.group_size, Q8F16_BLOCK_SIZE as u32);
        assert_eq!(ple.data_len, PLE_ENCODED_ROW_BYTES);
        assert_eq!(PLE_ENCODED_ROW_BYTES, 170);

        // Rank-3 experts flatten to four rows of K=256 and carry the tiers the
        // manifest declares for their roles: gate/up takes the aligned-K group
        // (136 B per 256 weights), and a K the aligned group cannot carry falls
        // to the row-local 68 B per 128.
        let gate_up = entry("model.language_model.layers.0.mlp.experts.gate_up_proj");
        assert_eq!(gate_up.kind, EntryKind::Quant(DType::MQ4G256V2));
        assert_eq!(gate_up.quant_type, MQ4G256V2_QUANT_TYPE);
        assert_eq!(gate_up.group_size, MQ4G256V2_GROUP_SIZE as u32);
        assert_eq!(gate_up.data_len, 4 * MQ4G256V2_GROUP_BYTES);

        let down = entry("mtp.layers.0.mlp.experts.down_proj");
        assert_eq!(down.kind, EntryKind::Quant(DType::MQ4G128V2));
        assert_eq!(down.quant_type, MQ4G128V2_QUANT_TYPE);
        assert_eq!(down.group_size, MQ4G128V2_GROUP_SIZE as u32);
        assert_eq!(down.data_len, 4 * MQ4G128V2_GROUP_BYTES);
        assert_eq!(plan.expert_entries, 2);
    }
    #[test]
    fn production_chunk_budget_covers_pinned_gate_up_and_rejects_oversized_values() {
        let rows = MAX_ROW_CHUNK as u64;
        let row_bytes = HIDDEN_WIDTH.checked_mul(2).unwrap();
        let raw_bytes = rows.checked_mul(row_bytes).unwrap();
        let value_bytes = rows
            .checked_mul(HIDDEN_WIDTH)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
            .unwrap();
        let encoded_bytes =
            quantized_data_len_for_dtype(DType::MFP4G32E8SOA, rows, HIDDEN_WIDTH).unwrap();
        let worst = raw_bytes
            .checked_add(value_bytes)
            .and_then(|bytes| bytes.checked_add(encoded_bytes))
            .unwrap();

        assert_eq!(MAX_CHUNK_BYTES, 96 * 1024 * 1024);
        assert_eq!(raw_bytes, 20_971_520);
        assert_eq!(value_bytes, 41_943_040);
        // E8 SoA at K = 2560: 16 B header + 80 scale bytes + 80 * 16 B codewords.
        assert_eq!(encoded_bytes, rows * (16 + 80 + 80 * 16));
        assert_eq!(encoded_bytes, 5_636_096);
        assert_eq!(worst, 68_550_656);
        assert!(worst < MAX_CHUNK_BYTES);

        let mut tracker = ScratchTracker::default();
        tracker
            .observe(
                usize::try_from(raw_bytes).unwrap(),
                usize::try_from(value_bytes).unwrap(),
                usize::try_from(encoded_bytes).unwrap(),
            )
            .unwrap();
        assert_eq!(tracker.high_water, worst);

        let mut over_cap = ScratchTracker::default();
        assert!(over_cap
            .observe(usize::try_from(MAX_CHUNK_BYTES).unwrap(), 1, 0)
            .is_err());
        assert!(validate_row_chunk(MAX_ROW_CHUNK).is_ok());
        assert!(validate_row_chunk(MAX_ROW_CHUNK + 1).is_err());
        assert!(validate_row_chunk(0).is_err());
    }
    #[test]
    fn production_chunk_budget_splits_pinned_qsa_o_projection_exactly() {
        // The pinned q_proj width is 24 heads * 256 head_dim = 6,144.
        // o_proj is [hidden=2,560, q_width=6,144], so the old single
        // chunk exceeded the aggregate raw+F32+encoded scratch bound.
        let rows = HIDDEN_WIDTH;
        let k = 24 * 256;
        let dtype = DType::Q8_0;
        let unsplit = chunk_scratch_bytes(dtype, rows, k).unwrap();
        assert_eq!(unsplit, 111_083_520);
        assert!(unsplit > MAX_CHUNK_BYTES);

        let bounded_rows = bounded_rows_for_scratch(dtype, k, rows).unwrap();
        assert_eq!(bounded_rows, 2_319);
        let bounded = chunk_scratch_bytes(dtype, bounded_rows, k).unwrap();
        assert_eq!(bounded, 100_626_048);
        assert!(bounded <= MAX_CHUNK_BYTES);
        assert!(chunk_scratch_bytes(dtype, bounded_rows + 1, k).unwrap() > MAX_CHUNK_BYTES);

        let raw_bytes = bounded_rows.checked_mul(k).unwrap().checked_mul(2).unwrap();
        let value_bytes = bounded_rows
            .checked_mul(k)
            .unwrap()
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .unwrap();
        let encoded_bytes = quantized_data_len_for_dtype(dtype, bounded_rows, k).unwrap();
        let mut tracker = ScratchTracker::default();
        tracker
            .observe(
                usize::try_from(raw_bytes).unwrap(),
                usize::try_from(value_bytes).unwrap(),
                usize::try_from(encoded_bytes).unwrap(),
            )
            .unwrap();
        assert_eq!(tracker.high_water, bounded);
    }

    #[test]
    fn qwen4_metadata_initializes_embedded_tokenizer() {
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<unk>": 0,
                    "<|startoftext|>": 1,
                    "<|endoftext|>": 2,
                    "h": 3,
                    "i": 4,
                    "\u{2581}": 5
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 1, "content": "<|startoftext|>", "special": true},
                {"id": 2, "content": "<|endoftext|>", "special": true}
            ]
        })
        .to_string();
        let ple = PleMetadata {
            multipliers: vec![3, 5, 7],
            vocab_sizes: vec![127; PLE_HEAD_COUNT],
            prefix_offsets: (0..PLE_HEAD_COUNT)
                .map(|index| (index * 127) as i64)
                .collect(),
        };
        let plan = EntryPlan {
            entries: Vec::new(),
            ple_metadata: None,
            row_chunk: 1,
            resident_entries: 0,
            expert_entries: 0,
            resident_bytes: 0,
            external_ple_bytes: 0,
            ple_shards: 0,
        };
        let tokenizer = TokenizerMetadata {
            tokenizer: tokenizer_json.clone(),
            tokenizer_config: Some(serde_json::json!({"add_bos_token": true})),
            generation_config: Some(serde_json::json!({
                "bos_token_id": 1,
                "eos_token_id": 2
            })),
        };
        let metadata_json =
            build_metadata(None, &plan, &ple, Some(&tokenizer)).expect("metadata JSON");
        let metadata: Value = serde_json::from_str(&metadata_json).expect("metadata object");
        assert_eq!(
            metadata.get("tokenizer").and_then(Value::as_str),
            Some(tokenizer_json.as_str())
        );
        let runtime_tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&metadata_json)
                .expect("embedded tokenizer initializes");
        assert_eq!(runtime_tokenizer.bos_id, 1);
        assert_eq!(runtime_tokenizer.eos_id, 2);
        assert_eq!(runtime_tokenizer.encode("hi").first().copied(), Some(1));
    }

    #[test]
    fn qwen4_source_metadata_loads_tokenizer_and_sidecars() {
        let dir = tempdir().expect("source directory");
        std::fs::write(dir.path().join("model.safetensors"), b"fixture").expect("source shard");
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "<unk>": 0,
                    "<|startoftext|>": 1,
                    "<|endoftext|>": 2,
                    "h": 3,
                    "i": 4,
                    "\u{2581}": 5
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 1, "content": "<|startoftext|>", "special": true},
                {"id": 2, "content": "<|endoftext|>", "special": true}
            ]
        })
        .to_string();
        std::fs::write(dir.path().join("tokenizer.json"), &tokenizer_json).expect("tokenizer");
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"add_bos_token":true}"#,
        )
        .expect("tokenizer config");
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"bos_token_id":1,"eos_token_id":2}"#,
        )
        .expect("generation config");
        std::fs::write(dir.path().join("chat_template.jinja"), "{{ bos_token }}")
            .expect("chat template");

        let source = source_paths(dir.path()).expect("source paths");
        let metadata = load_tokenizer_metadata(&source).expect("tokenizer metadata");
        assert_eq!(metadata.tokenizer, tokenizer_json);
        assert_eq!(
            metadata
                .tokenizer_config
                .as_ref()
                .and_then(|config| config.get("chat_template"))
                .and_then(Value::as_str),
            Some("{{ bos_token }}")
        );
        assert_eq!(
            metadata
                .generation_config
                .as_ref()
                .and_then(|config| config.get("bos_token_id"))
                .and_then(Value::as_u64),
            Some(1)
        );
        let envelope = serde_json::json!({
            "tokenizer": metadata.tokenizer,
            "tokenizer_config": metadata.tokenizer_config,
            "generation_config": metadata.generation_config,
        })
        .to_string();
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&envelope)
            .expect("runtime tokenizer");
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
    }

    #[test]
    fn pinned_output_prediction_is_exact_before_payload_reads() {
        assert_eq!(
            PINNED_SOURCE_TENSOR_COUNT - PINNED_VISION_TENSOR_COUNT,
            PINNED_OUTPUT_ENTRY_COUNT
        );
        let data_start = 32 + PINNED_METADATA_BYTES + PINNED_INDEX_BYTES;
        assert_eq!(data_start, 120_450);
        let data_offset = (data_start + 4_095) & !4_095;
        assert_eq!(data_offset, 122_880);
        // Two tier changes separate the pinned checkpoint's stored size from its
        // source payload: the Q8 trunk matrices (3,759,329,280 B below the
        // previously pinned tiers) and the PLE row tier, whose 128 shards of
        // 2,500,012 rows are written as Q8F16 rows (170 B) instead of BF16 rows
        // (320 B).
        let trunk_saved = 3_759_329_280u64;
        let ple_saved = (PLE_SHARD_COUNT as u64)
            .checked_mul(PLE_ROWS_PER_SHARD)
            .and_then(|rows| rows.checked_mul(PLE_ROW_WIDTH * 2 - PLE_ENCODED_ROW_BYTES))
            .expect("PLE saving");
        assert_eq!(ple_saved, 48_000_230_400);
        assert_eq!(
            data_offset + PINNED_PAYLOAD_BYTES - trunk_saved - ple_saved,
            PINNED_PREDICTED_OUTPUT_BYTES
        );
    }

    fn validate_expert_shape_stub(shape: &[u64], kind: ExpertKind) -> Result<(), Qwen4Error> {
        let expected = match kind {
            ExpertKind::GateUp => [ROUTED_EXPERTS, GATE_UP_INTERMEDIATE, HIDDEN_WIDTH],
            ExpertKind::Down => [ROUTED_EXPERTS, HIDDEN_WIDTH, DOWN_INTERMEDIATE],
        };
        if shape != expected {
            return Err(Qwen4Error::Invalid("shape mismatch".to_string()));
        }
        Ok(())
    }

    #[test]
    fn ple_suffix_sort_is_numeric_not_lexical() {
        let mut names = [
            "model.ple.ngram_embedding.shard_10.weight",
            "model.ple.ngram_embedding.shard_2.weight",
            "model.ple.ngram_embedding.shard_0.weight",
        ];
        names.sort_by_key(|name| ple_shard_index(name).unwrap());
        assert_eq!(names[0], "model.ple.ngram_embedding.shard_0.weight");
        assert_eq!(names[1], "model.ple.ngram_embedding.shard_2.weight");
        assert_eq!(names[2], "model.ple.ngram_embedding.shard_10.weight");
    }

    #[test]
    fn bf16_stream_preserves_source_bytes() {
        let bytes = [0x00, 0x3f, 0x80, 0xbf, 0x34, 0x12, 0xff, 0x7f];
        let (_file, tensor) = source_tensor(&bytes, vec![2, 2], "BF16");
        let mut output = Vec::new();
        let mut scratch = ScratchTracker::default();
        stream_raw_rows(&tensor, 2, 2, 1, &mut scratch, &mut output).unwrap();
        assert_eq!(output, bytes);
    }

    /// The production PLE rasterization for a fixture: every source BF16 row
    /// decoded to f32 and quantized with the production encoder, in row order.
    /// Tests compare against this so row order and chunk boundaries are what is
    /// under test, not the encoder internals.
    fn expected_ple_q8_bytes(bytes: &[u8], rows: u64) -> Vec<u8> {
        let row_bytes = PLE_ROW_WIDTH as usize * 2;
        let mut out = Vec::with_capacity(rows as usize * PLE_ENCODED_ROW_BYTES as usize);
        for row in 0..rows as usize {
            let mut values = vec![0f32; PLE_ROW_WIDTH as usize];
            decode_bf16(&bytes[row * row_bytes..(row + 1) * row_bytes], &mut values)
                .expect("decode fixture PLE row");
            out.extend_from_slice(&quantize_q8f16(&values));
        }
        out
    }

    #[test]
    fn external_ple_stream_uses_bounded_ranges_and_reduces_request_count() {
        let row_bytes = PLE_ROW_WIDTH * 2;
        let rows = EXTERNAL_PLE_ROW_CHUNK as u64 + 1;
        let total = rows.checked_mul(row_bytes).expect("fixture length");
        let max_range_bytes = (EXTERNAL_PLE_ROW_CHUNK as u64)
            .checked_mul(row_bytes)
            .expect("range length");
        let requests_per_shard = PLE_ROWS_PER_SHARD.div_ceil(EXTERNAL_PLE_ROW_CHUNK as u64);
        let default_requests_per_shard = PLE_ROWS_PER_SHARD.div_ceil(DEFAULT_ROW_CHUNK as u64);
        assert_eq!(max_range_bytes, 20 * 1024 * 1024);
        assert_eq!(requests_per_shard, 39);
        assert_eq!(default_requests_per_shard, 611);
        assert_eq!(
            PLE_SHARD_COUNT * requests_per_shard as usize,
            4_992,
            "all external PLE shards use the larger raw range"
        );
        assert_eq!(
            PLE_SHARD_COUNT * default_requests_per_shard as usize,
            78_208,
            "the old 4096-row request count remains the comparison baseline"
        );

        let (base, handle) = spawn_http_range_counter(total, 2);
        let remote = Arc::new(test_remote_source(&base));
        let path = "model.safetensors".to_string();
        let tensor = SourceTensor {
            name: "model.ple.ngram_embedding.shard_0.weight".to_string(),
            dtype: "BF16".to_string(),
            shape: vec![rows, PLE_ROW_WIDTH],
            data_start: 0,
            data_end: total,
            shard: Arc::new(SourceShard {
                path: PathBuf::from(&path),
                kind: SourceKind::Remote {
                    source: Arc::clone(&remote),
                    path,
                },
                file_len: total,
                local_identity: None,
            }),
        };
        let entry = PlannedEntry {
            source: tensor,
            name: "model.ple.ngram_embedding.shard_0.weight".to_string(),
            quant_type: Q8F16_QUANT_TYPE,
            shape: vec![rows as u32, PLE_ROW_WIDTH as u32],
            group_size: Q8F16_BLOCK_SIZE as u32,
            data_len: rows * PLE_ENCODED_ROW_BYTES,
            kind: EntryKind::Ple,
        };
        let mut scratch = ScratchTracker::default();
        let mut output = Vec::with_capacity(entry.data_len as usize);
        stream_entry(
            &entry,
            DEFAULT_ROW_CHUNK,
            &[],
            &[],
            &[],
            &[],
            &mut scratch,
            &mut output,
        )
        .expect("external PLE stream");
        let source: Vec<u8> = (0..total).map(|offset| (offset % 251) as u8).collect();
        let expected = expected_ple_q8_bytes(&source, rows);
        assert_eq!(
            output, expected,
            "quantized rows must be written in source row order"
        );
        assert_eq!(output.len() as u64, entry.data_len);
        // One bounded pass holds the BF16 read, its f32 decode, and the Q8F16
        // encode for every row of the chunk.
        let chunk_rows = rows.min(EXTERNAL_PLE_ROW_CHUNK as u64);
        assert_eq!(
            scratch.high_water,
            chunk_rows * (PLE_ROW_WIDTH * 2 + PLE_ROW_WIDTH * 4 + PLE_ENCODED_ROW_BYTES)
        );

        let ranges = handle.join().expect("range counter");
        assert_eq!(
            ranges,
            vec![(0, max_range_bytes - 1), (max_range_bytes, total - 1)]
        );
        assert_eq!(
            ranges.iter().map(|(start, end)| end - start + 1).max(),
            Some(max_range_bytes)
        );
    }

    #[test]
    fn external_ple_stream_preserves_artifact_bytes_and_plan() {
        let rows: u32 = 3;
        let row_bytes = PLE_ROW_WIDTH as usize * 2;
        let bytes: Vec<u8> = (0..(rows as usize) * row_bytes)
            .map(|offset| (offset % 251) as u8)
            .collect();
        let (_file, tensor) = source_tensor(&bytes, vec![rows as u64, PLE_ROW_WIDTH], "BF16");
        let expected = expected_ple_q8_bytes(&bytes, rows as u64);
        let entry = PlannedEntry {
            source: tensor,
            name: "model.ple.ngram_embedding.shard_0.weight".to_string(),
            quant_type: Q8F16_QUANT_TYPE,
            shape: vec![rows, PLE_ROW_WIDTH as u32],
            group_size: Q8F16_BLOCK_SIZE as u32,
            data_len: expected.len() as u64,
            kind: EntryKind::Ple,
        };

        let mut optimized = Vec::new();
        let mut optimized_scratch = ScratchTracker::default();
        stream_entry(
            &entry,
            DEFAULT_ROW_CHUNK,
            &[],
            &[],
            &[],
            &[],
            &mut optimized_scratch,
            &mut optimized,
        )
        .expect("optimized PLE stream");
        assert_eq!(
            optimized, expected,
            "PLE payload must be the quantized rows in source order"
        );

        let ple = PleMetadata {
            multipliers: vec![3, 5, 7],
            vocab_sizes: vec![127; PLE_HEAD_COUNT],
            prefix_offsets: (0..PLE_HEAD_COUNT)
                .map(|index| (index * 127) as i64)
                .collect(),
        };
        let plan = EntryPlan {
            entries: vec![entry.clone()],
            ple_metadata: Some(ple.clone()),
            row_chunk: DEFAULT_ROW_CHUNK,
            resident_entries: 0,
            expert_entries: 0,
            resident_bytes: 0,
            external_ple_bytes: expected.len() as u64,
            ple_shards: 1,
        };
        let metadata_json = build_metadata(None, &plan, plan.ple_metadata.as_ref().unwrap(), None)
            .expect("PLE metadata");
        let stream_entries = vec![hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        }];
        let artifact = NamedTempFile::new().expect("artifact");
        hipfire_runtime::hfq::write_hfqm_package_streaming(
            artifact.path(),
            QWEN4_ARCH_ID,
            &metadata_json,
            &stream_entries,
            |index, writer| {
                assert_eq!(index, 0);
                stream_entry(
                    &entry,
                    DEFAULT_ROW_CHUNK,
                    &[],
                    &[],
                    &[],
                    &[],
                    &mut ScratchTracker::default(),
                    writer,
                )
                .map_err(Qwen4Error::into_io)
            },
        )
        .expect("write artifact");
        let reopened = Qwen4ReopenPlan::open(artifact.path()).expect("reopen artifact");
        reopened
            .validate_against(&metadata_json, &stream_entries)
            .expect("artifact plan");
        let metadata: Value = serde_json::from_str(&metadata_json).expect("metadata JSON");
        assert_eq!(
            metadata["qwen4_streaming"]["row_chunk"].as_u64(),
            Some(DEFAULT_ROW_CHUNK as u64)
        );
        let file = std::fs::File::open(artifact.path()).expect("open artifact payload");
        let mut payload = vec![0u8; expected.len()];
        read_exact_at(&file, reopened.entries[0].data_offset, &mut payload)
            .expect("read artifact payload");
        assert_eq!(
            payload, expected,
            "artifact payload must carry the quantized PLE rows"
        );
    }

    #[test]
    fn i64_metadata_keeps_signed_values_exactly() {
        let values = [i64::MIN + 7, -19, i64::MAX - 11];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let (_file, tensor) = source_tensor(&bytes, vec![3], "I64");
        assert_eq!(read_i64_array(&tensor, 3).unwrap(), values);
        assert_eq!(i64_role("x.layer_multipliers"), Some(I64Role::Multipliers));
        assert_eq!(QWEN4_I64_QUANT_TYPE, 52);
        assert!(
            validate_output_entry_len("x.layer_multipliers", QWEN4_I64_QUANT_TYPE, &[3], 24,)
                .is_ok()
        );
        assert!(validate_output_entry_len("x.layer_multipliers", 22, &[3], 12).is_err());
    }

    #[test]
    fn reopen_plan_reads_small_fixture_without_mapping_payload() {
        let temp = NamedTempFile::new().unwrap();
        let mut offsets = Vec::new();
        for head in 0..PLE_HEAD_COUNT {
            offsets.push((head * 127) as i64);
        }
        let metadata = json!({
            "arch_id": 16,
            "qwen4_ple": {
                "version": 1,
                "multipliers": [3, 5, 7],
                "head_vocab_sizes": [127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127],
                "head_offsets": offsets,
                "padded_rows": PLE_SHARD_COUNT as u64 * PLE_ROWS_PER_SHARD
            }
        })
        .to_string();
        let payload = [0x00u8, 0x3f, 0x80, 0xbf];
        let entry = hipfire_runtime::hfq::HfqStreamEntry {
            name: "fixture.weight".to_string(),
            quant_type: 16,
            shape: vec![2],
            group_size: 0,
            data_len: payload.len() as u64,
        };
        hipfire_runtime::hfq::write_hfqm_package_streaming(
            temp.path(),
            16,
            &metadata,
            &[entry],
            |_, writer| writer.write_all(&payload),
        )
        .unwrap();
        let plan = Qwen4ReopenPlan::open(temp.path()).unwrap();
        assert_eq!(plan.arch_id, 16);
        assert_eq!(plan.entries.len(), 1);
        assert_eq!(plan.entries[0].data_len, payload.len() as u64);
    }
    /// Construct the production output plan from headers only.  Unlike
    /// `plan_entries`, this deliberately does not read the three I64 payloads;
    /// it is the bounded admission proof for a sparse full-checkpoint fixture.
    fn sparse_header_plan(
        tensors: &[SourceTensor],
        inventory: &ManifestIndex,
    ) -> Result<EntryPlan, Qwen4Error> {
        validate_manifest_inventory_names(tensors, inventory)?;
        let mut ple_entries = Vec::with_capacity(PLE_SHARD_COUNT);
        let mut resident = Vec::new();
        let mut metadata_entries = Vec::new();
        let mut expert_entries = 0usize;
        let mut gate_up_count = 0usize;
        let mut down_count = 0usize;
        let mut mtp_gate_up = false;
        let mut mtp_down = false;

        for tensor in tensors {
            if let Some(expected) = inventory.metadata.get(&tensor.name) {
                if tensor.dtype != "I64" || tensor.shape != expected.shape {
                    return Err(Qwen4Error::Invalid(format!(
                        "{} header metadata is {:?}/{:?}, expected I64/{:?}",
                        tensor.name, tensor.dtype, tensor.shape, expected.shape
                    )));
                }
                let data_len = checked_product(&tensor.shape, &tensor.name)?
                    .checked_mul(8)
                    .ok_or_else(|| {
                        Qwen4Error::Invalid(format!("{} I64 payload length overflows", tensor.name))
                    })?;
                if tensor.data_len() != data_len {
                    return Err(Qwen4Error::Invalid(format!(
                        "{} header payload is {}, expected {}",
                        tensor.name,
                        tensor.data_len(),
                        data_len
                    )));
                }
                metadata_entries.push(PlannedEntry {
                    source: tensor.clone(),
                    name: tensor.name.clone(),
                    quant_type: QWEN4_I64_QUANT_TYPE,
                    shape: shape_u32(&tensor.shape, &tensor.name)?,
                    group_size: 0,
                    data_len,
                    kind: EntryKind::I64(expected.role),
                });
                continue;
            }

            if let Some(expected) = inventory.required.get(&tensor.name) {
                validate_manifest_source(tensor, expected)?;
                match expected.role {
                    ManifestRole::Ple => {
                        let index = expected
                            .ple_index
                            .expect("manifest PLE role has a numeric shard index");
                        let data_len = (PLE_ROWS_PER_SHARD)
                            .checked_mul(PLE_ROW_WIDTH)
                            .and_then(|elements| elements.checked_mul(2))
                            .ok_or_else(|| {
                                Qwen4Error::Invalid(
                                    "sparse header PLE payload length overflows".to_string(),
                                )
                            })?;
                        if tensor.data_len() != data_len {
                            return Err(Qwen4Error::Invalid(format!(
                                "{} header PLE payload is {}, expected {}",
                                tensor.name,
                                tensor.data_len(),
                                data_len
                            )));
                        }
                        ple_entries.push(PlannedEntry {
                            source: tensor.clone(),
                            name: tensor.name.clone(),
                            quant_type: 16,
                            shape: shape_u32(&tensor.shape, &tensor.name)?,
                            group_size: 0,
                            data_len,
                            kind: EntryKind::Ple,
                        });
                    }
                    ManifestRole::GateUp(dtype) | ManifestRole::Down(dtype) => {
                        let kind = match expected.role {
                            ManifestRole::GateUp(_) => ExpertKind::GateUp,
                            ManifestRole::Down(_) => ExpertKind::Down,
                            _ => unreachable!("matched expert role"),
                        };
                        let (rows, k) = validate_expert_shape(tensor, kind, &expected.shape)?;
                        let (quant_type, group_size) = matrix_quant_type(dtype)?;
                        let data_len = quantized_data_len_for_dtype(dtype, rows, k)?;
                        resident.push(PlannedEntry {
                            source: tensor.clone(),
                            name: tensor.name.clone(),
                            quant_type,
                            shape: shape_u32(&tensor.shape, &tensor.name)?,
                            group_size,
                            data_len,
                            kind: EntryKind::Quant(dtype),
                        });
                        expert_entries += 1;
                        match kind {
                            ExpertKind::GateUp => {
                                gate_up_count += 1;
                                mtp_gate_up |= expected.is_mtp;
                            }
                            ExpertKind::Down => {
                                down_count += 1;
                                mtp_down |= expected.is_mtp;
                            }
                        }
                    }
                    ManifestRole::Matrix(dtype) => {
                        let (rows, k) = validate_matrix_shape(tensor, &expected.shape)?;
                        let data_len = quantized_data_len_for_dtype(dtype, rows, k)?;
                        resident.push(PlannedEntry {
                            source: tensor.clone(),
                            name: tensor.name.clone(),
                            quant_type: matrix_quant_type(dtype)?.0,
                            shape: shape_u32(&tensor.shape, &tensor.name)?,
                            group_size: matrix_quant_type(dtype)?.1,
                            data_len,
                            kind: EntryKind::Quant(dtype),
                        });
                    }
                    ManifestRole::Bf16 => {
                        let data_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
                        resident.push(PlannedEntry {
                            source: tensor.clone(),
                            name: tensor.name.clone(),
                            quant_type: 16,
                            shape: shape_u32(&tensor.shape, &tensor.name)?,
                            group_size: 0,
                            data_len,
                            kind: EntryKind::Bf16,
                        });
                    }
                }
                continue;
            }

            if inventory.aliases.contains_key(&tensor.name) || is_vision_tensor(&tensor.name) {
                continue;
            }
            return Err(Qwen4Error::Invalid(format!(
                "sparse header tensor {} is not declared by the manifest",
                tensor.name
            )));
        }

        if ple_entries.len() != PLE_SHARD_COUNT {
            return Err(Qwen4Error::Invalid(format!(
                "sparse header plan found {} PLE shards, expected {PLE_SHARD_COUNT}",
                ple_entries.len()
            )));
        }
        ple_entries.sort_by_key(|entry| ple_shard_index(&entry.name).expect("PLE entry name"));
        let external_ple_bytes = (PLE_SHARD_COUNT as u64)
            .checked_mul(PLE_ROWS_PER_SHARD)
            .and_then(|rows| rows.checked_mul(PLE_ROW_WIDTH as u64))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Qwen4Error::Invalid("sparse PLE bytes overflow".to_string()))?;
        let source_ple_bytes = ple_entries.iter().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.source.data_len())
                .ok_or_else(|| Qwen4Error::Invalid("sparse source PLE bytes overflow".to_string()))
        })?;
        if source_ple_bytes != external_ple_bytes {
            return Err(Qwen4Error::Invalid(format!(
                "sparse source PLE bytes {source_ple_bytes}, expected {external_ple_bytes}"
            )));
        }
        if gate_up_count == 0 || down_count == 0 || !mtp_gate_up || !mtp_down {
            return Err(Qwen4Error::Invalid(
                "sparse header plan is missing trunk or MTP gate/up/down experts".to_string(),
            ));
        }
        resident.sort_by_key(|entry| inventory.manifest_order.get(&entry.name).copied());
        metadata_entries.sort_by_key(|entry| inventory.manifest_order.get(&entry.name).copied());
        let resident_bytes = resident.iter().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.data_len)
                .ok_or_else(|| Qwen4Error::Invalid("sparse resident bytes overflow".to_string()))
        })?;
        let resident_entries = resident.len();
        let mut entries = ple_entries;
        entries.extend(resident);
        entries.extend(metadata_entries);
        Ok(EntryPlan {
            entries,
            ple_metadata: Some(PleMetadata {
                multipliers: hipfire_arch_qwen4::ple::PLE_MULTIPLIERS.to_vec(),
                vocab_sizes: hipfire_arch_qwen4::ple::PLE_HEAD_VOCAB_SIZES
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
                prefix_offsets: hipfire_arch_qwen4::ple::PLE_HEAD_OFFSETS
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            }),
            row_chunk: DEFAULT_ROW_CHUNK,
            resident_entries,
            expert_entries,
            resident_bytes,
            external_ple_bytes,
            ple_shards: PLE_SHARD_COUNT,
        })
    }

    #[test]
    #[ignore = "network-backed sparse pinned-checkpoint campaign; set QWEN4_SPARSE_INVENTORY_DIR"]
    fn production_pinned_sparse_headers_validate_without_payload_reads() {
        let root = env::var_os("QWEN4_SPARSE_INVENTORY_DIR")
            .map(std::path::PathBuf::from)
            .expect("QWEN4_SPARSE_INVENTORY_DIR must point to fetched sparse headers");
        let source = source_paths(&root).expect("sparse source paths");
        validate_source_shard_count(&source, Qwen4Mode::Production).expect("131 shard admission");
        let tensors = load_inventory(&source).expect("header inventory");
        assert_eq!(tensors.len(), PINNED_SOURCE_TENSOR_COUNT);
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.dtype == "BF16")
                .count(),
            PINNED_BF16_TENSOR_COUNT
        );
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.dtype == "I64")
                .count(),
            PINNED_I64_TENSOR_COUNT
        );
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| is_vision_tensor(&tensor.name))
                .count(),
            PINNED_VISION_TENSOR_COUNT
        );
        validate_source_inventory(&tensors, Qwen4Mode::Production).expect("pinned source census");

        #[cfg(unix)]
        let blocks_before: Vec<_> = match &source {
            SourceSet::Local { paths, .. } => paths
                .iter()
                .map(|path| std::fs::metadata(path).expect("sparse shard stat").blocks())
                .collect(),
            SourceSet::Remote { .. } => Vec::new(),
        };
        let config_value = load_optional_config(&source)
            .expect("sparse config")
            .expect("sparse config.json");
        let config = Qwen4Config::from_value(&config_value).expect("pinned Qwen4 config");
        let manifest = Qwen4Manifest::build(&config).expect("pinned Qwen4 manifest");
        let inventory = ManifestIndex::build(&manifest).expect("manifest index");
        let plan = sparse_header_plan(&tensors, &inventory).expect("header-only output plan");
        assert_eq!(plan.entries.len(), PINNED_OUTPUT_ENTRY_COUNT);
        assert_eq!(plan.ple_shards, PLE_SHARD_COUNT);
        assert_eq!(plan.external_ple_bytes, 102_400_491_520);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| matches!(entry.kind, EntryKind::I64(_)))
                .count(),
            PINNED_I64_TENSOR_COUNT
        );
        let ple = plan.ple_metadata.as_ref().expect("pinned PLE metadata");
        let tokenizer = load_tokenizer_metadata(&source).expect("pinned tokenizer metadata");
        let metadata_json = build_metadata(Some(&config_value), &plan, ple, Some(&tokenizer))
            .expect("metadata JSON");
        println!("sparse metadata bytes={}", metadata_json.len());
        let stream_entries: Vec<_> = plan
            .entries
            .iter()
            .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
                name: entry.name.clone(),
                quant_type: entry.quant_type,
                shape: entry.shape.clone(),
                group_size: entry.group_size,
                data_len: entry.data_len,
            })
            .collect();
        let predicted = predicted_output_bytes(&metadata_json, &stream_entries)
            .expect("header-only predicted output bytes");
        let data_start = 32 + metadata_json.len() as u64 + PINNED_INDEX_BYTES;
        let data_offset = (data_start + 4_095) & !4_095;
        assert_eq!(predicted, data_offset + PINNED_PAYLOAD_BYTES);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.kind == EntryKind::Ple)
                .count(),
            PLE_SHARD_COUNT
        );
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| matches!(entry.kind, EntryKind::I64(_)))
                .count(),
            PINNED_I64_TENSOR_COUNT
        );
        source.verify_identity().expect("sparse source identities");
        #[cfg(unix)]
        if let SourceSet::Local { paths, .. } = &source {
            let blocks_after: Vec<_> = paths
                .iter()
                .map(|path| std::fs::metadata(path).expect("sparse shard stat").blocks())
                .collect();
            assert_eq!(
                blocks_after, blocks_before,
                "header-only admission must not fault/write source payload blocks"
            );
        }
        println!(
            "PASS sparse production headers: shards=131 source_tensors={} excluded_vision={} output_entries={} ple_shards={} i64={} metadata_bytes={} predicted_output_bytes={} payload_reads=0",
            tensors.len(),
            PINNED_VISION_TENSOR_COUNT,
            plan.entries.len(),
            plan.ple_shards,
            PINNED_I64_TENSOR_COUNT,
            metadata_json.len(),
            predicted
        );
    }

    // ── E8-SoA numerics probe ──────────────────────────────────────────────
    //
    // The E8 tier is only trustworthy if the producer's encoder, the decode
    // path's rotation helper, and both kernel arms agree with a plain f32
    // reference.  The kernels are shared with other families; the wiring
    // (sign tables, rotation basis, SoA row stride, batch argument) is ours,
    // and every one of those has produced silent garbage in this tree before.

    fn e8_probe_bytes(values: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        }
    }

    fn e8_probe_bytes_mut(values: &mut [f32]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(values),
            )
        }
    }

    /// Deterministic small-magnitude weights: real weight rows, not lattice
    /// friendly constants.
    fn e8_probe_weights(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..m * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) & 0xffff) as f32 / 65536.0;
                (unit - 0.5) * 0.04
            })
            .collect()
    }

    /// Activations are multiples of 1/128, so every value is exact in f16: the
    /// batched WMMA arm stages its input through f16 internally and that
    /// rounding must not be part of the wiring error being measured.
    fn e8_probe_activations(rows: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..rows * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let value = ((state >> 40) % 257) as i32 - 128;
                value as f32 / 128.0
            })
            .collect()
    }

    fn e8_probe_fwht(values: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
        let mut rotated = values.to_vec();
        for segment in rotated.chunks_mut(256) {
            cpu_fwht_256(segment, signs1, signs2);
        }
        rotated
    }

    fn e8_probe_rel_l2(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len(), "probe length mismatch");
        let numerator: f32 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let denominator: f32 = expected.iter().map(|b| b * b).sum::<f32>().sqrt();
        numerator / denominator.max(f32::MIN_POSITIVE)
    }

    #[test]
    #[ignore = "device-backed: run `cargo test --release -p hipfire-quantize --bin hipfire-quantize -- --ignored e8_soa_projection --nocapture`"]
    fn e8_soa_projection_matches_f32_reference_on_device() {
        // 32 rows x 512 K keeps the two expert geometries honest (K % 256 == 0)
        // while the batch of 24 crosses the 16-token tile the batched arm
        // writes per block.
        const M: usize = 32;
        const K: usize = 512;
        const BATCH: usize = 24;
        const POISON: f32 = 12345.625;

        // The producer's own sign tables (see `Qwen4Plan`/`stream_entry`).
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);

        let weights = e8_probe_weights(M, K, 0x5eed_1234_abcd_0001);
        let packed = quantize_mfp4g32_e8_soa_2d(&weights, M, K, &signs1, &signs2);
        let (row_stride, extent) = mfp4e8soa_row_geometry(M as u64, K as u64).unwrap();
        assert_eq!(row_stride, 16 + 16 + 16 * 16);
        assert_eq!(packed.len() as u64, extent);

        let decoded = dequant_mfp4g32_e8_soa(&packed, M, K);
        let project = |rotated: &[f32]| -> Vec<f32> {
            (0..M)
                .map(|row| {
                    decoded[row * K..(row + 1) * K]
                        .iter()
                        .zip(rotated)
                        .map(|(weight, activation)| weight * activation)
                        .sum::<f32>()
                })
                .collect()
        };

        let x = e8_probe_activations(1, K, 0x0abc_def0_0000_0002);
        let x_rotated_cpu = e8_probe_fwht(&x, &signs1, &signs2);
        let gemv_reference = project(&x_rotated_cpu);
        // The FWHT is orthonormal, so `dot(rotated_weight, rotated_x)` is the
        // same scalar as the natural-basis dot product: this is the true output
        // the f32 weights would produce, and the gap to `gemv_reference` is the
        // tier's own contribution.
        let rotated_weights = e8_probe_fwht(&weights, &signs1, &signs2);
        let exact_reference: Vec<f32> = (0..M)
            .map(|row| {
                rotated_weights[row * K..(row + 1) * K]
                    .iter()
                    .zip(&x_rotated_cpu)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>()
            })
            .collect();

        // Encoder floor: element-wise, so no dot-product cancellation masks the
        // lattice granularity.  Reported, never asserted: the tier's quality is
        // measured on a real artifact, not on probe data.
        let element_error = e8_probe_rel_l2(&decoded, &rotated_weights);
        let norm_ratio = decoded
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            / rotated_weights
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
        println!(
            "E8-SoA probe encoder M={M} K={K}: element rel_l2={element_error:.4} \
             decoded/rotated norm ratio={norm_ratio:.4}"
        );

        let mut gpu = Gpu::init().expect("device init");
        let arch = gpu.arch.clone();

        let download = |gpu: &Gpu, tensor: &rdna_compute::GpuTensor, len: usize| -> Vec<f32> {
            let mut host = vec![0f32; len];
            gpu.hip
                .memcpy_dtoh(e8_probe_bytes_mut(&mut host), &tensor.buf)
                .expect("download");
            host
        };

        let weight_gpu = gpu
            .upload_raw(&packed, &[packed.len()])
            .expect("upload E8 SoA weights");
        let x_gpu = gpu.alloc_tensor(&[K], DType::F32).expect("alloc x");
        gpu.hip
            .memcpy_htod(&x_gpu.buf, e8_probe_bytes(&x))
            .expect("upload x");
        let rotated_gpu = gpu.alloc_tensor(&[K], DType::F32).expect("alloc x_rot");
        let y_gpu = gpu.alloc_tensor(&[M], DType::F32).expect("alloc y");
        gpu.rotate_x_mq(&x_gpu, &rotated_gpu, K)
            .expect("rotate_x_mq");
        gpu.gemv_mfp4g32_e8_soa_prerotated(&weight_gpu, &rotated_gpu, &y_gpu, M, K)
            .expect("gemv_mfp4g32_e8_soa_prerotated");
        gpu.hip.device_synchronize().expect("sync");

        let rotated_gpu_host = download(&gpu, &rotated_gpu, K);
        let gemv_host = download(&gpu, &y_gpu, M);
        let rotation_error = e8_probe_rel_l2(&rotated_gpu_host, &x_rotated_cpu);
        let gemv_wiring_error = e8_probe_rel_l2(&gemv_host, &gemv_reference);
        let format_error = e8_probe_rel_l2(&gemv_reference, &exact_reference);
        println!(
            "E8-SoA probe arch={arch} single row M={M} K={K}: rotation rel_l2={rotation_error:.3e} \
             gemv rel_l2={gemv_wiring_error:.3e} format rel_l2={format_error:.3e}"
        );
        assert!(
            rotation_error < 1e-5,
            "the decode rotation basis must equal the producer's FWHT-256 basis"
        );
        assert!(
            gemv_wiring_error < 5e-4,
            "single-row E8 GEMV disagrees with the encoded weight's own dequant"
        );
        assert!(
            element_error < 0.5,
            "the encoder must reproduce its own row to within the lattice's granularity"
        );
        assert!(
            (0.9..=1.1).contains(&norm_ratio),
            "the encoder must preserve row energy: a scale factor outside this band \
             means the QUANT_STEP was folded twice (or not at all)"
        );

        // Batched arm (the prefill shape): gfx1151 WMMA, per-row GEMV elsewhere.
        let batch_x = e8_probe_activations(BATCH, K, 0x0abc_def0_0000_0003);
        let mut batch_reference = Vec::with_capacity(BATCH * M);
        for row in 0..BATCH {
            let rotated = e8_probe_fwht(&batch_x[row * K..(row + 1) * K], &signs1, &signs2);
            batch_reference.extend(project(&rotated));
        }

        let batch_x_gpu = gpu
            .alloc_tensor(&[BATCH * K], DType::F32)
            .expect("alloc batch x");
        gpu.hip
            .memcpy_htod(&batch_x_gpu.buf, e8_probe_bytes(&batch_x))
            .expect("upload batch x");
        let batch_rot_gpu = gpu
            .alloc_tensor(&[BATCH * K], DType::F32)
            .expect("alloc batch x_rot");
        // Poison the destination: an unwritten output row keeps the sentinel and
        // is reported as such instead of passing as a plausible number.
        let mut poisoned = vec![POISON; BATCH * M];
        let batch_y_gpu = gpu
            .alloc_tensor(&[BATCH * M], DType::F32)
            .expect("alloc batch y");
        gpu.hip
            .memcpy_htod(&batch_y_gpu.buf, e8_probe_bytes(&poisoned))
            .expect("poison batch y");
        gpu.rotate_x_mq_batched(&batch_x_gpu, &batch_rot_gpu, K, BATCH)
            .expect("rotate_x_mq_batched");
        gpu.gemm_mfp4g32_e8_soa_wmma(&weight_gpu, &batch_rot_gpu, &batch_y_gpu, M, K, BATCH)
            .expect("gemm_mfp4g32_e8_soa_wmma");
        gpu.hip.device_synchronize().expect("sync");

        let batch_host = download(&gpu, &batch_y_gpu, BATCH * M);
        poisoned = batch_host.clone();
        let unwritten = poisoned.iter().filter(|value| **value == POISON).count();
        let batch_wiring_error = e8_probe_rel_l2(&batch_host, &batch_reference);
        let per_row_errors: Vec<f32> = (0..BATCH)
            .map(|row| {
                e8_probe_rel_l2(
                    &batch_host[row * M..(row + 1) * M],
                    &batch_reference[row * M..(row + 1) * M],
                )
            })
            .collect();
        println!(
            "E8-SoA probe batched M={M} K={K} B={BATCH}: rel_l2={batch_wiring_error:.3e} \
             unwritten_outputs={unwritten} worst_row rel_l2={:.3e}",
            per_row_errors.iter().cloned().fold(0f32, f32::max)
        );
        assert_eq!(
            unwritten, 0,
            "every batched output column must be written: the kernel covers 16 tokens per block"
        );
        assert!(
            batch_wiring_error < 5e-4,
            "batched E8 projection disagrees with the encoded weight's own dequant"
        );
    }
}
