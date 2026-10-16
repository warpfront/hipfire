// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact GPU state/rollback parity scenarios used by the Qwen4 parity CLI.
//!
//! Model logits are intentionally not produced here: the CLI's state mode
//! obtains those from the real loaded bundle/forward/spec path.  This module
//! only orchestrates compact state transitions and serializes device-backed
//! family digests.

use crate::config::compact_test_config;
use crate::gpu_forward::{
    qwen4_profile_enable, qwen4_profile_reset, qwen4_profile_snapshot, Qwen4GpuForwardScratch,
    Qwen4ProfilePhase, Qwen4ProfileStats,
};
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_gpu::{MtpGpuState, MtpStateParityMetadata};
use crate::mtp_spec::validate_native_mtp_prefill_request;
use crate::state::Qwen4State;
use hip_bridge::launch_counters;
use hipfire_runtime::external_rows::RowCacheStats;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use rdna_compute::tensor_ops::{
    indexed_attention_cache_append, indexed_attention_pool_rope, indexed_attention_reuse_selection,
    indexed_attention_select, IndexedAttentionCacheAppend, IndexedAttentionPoolRope,
    IndexedAttentionReuseSelection, IndexedAttentionSelect,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Kept CPU-only so every real-model state/profile loader uses a testable
// split-base attachment boundary before artifact admission.
fn attach_state_validation_rows(
    hfq: &mut hipfire_runtime::hfq::HfqFile,
    model_path: &Path,
    request: hipfire_config::memory::PleFormatRequest,
    arch: &str,
) -> Result<(), String> {
    hipfire_runtime::ple_sidecar::attach_validation_ple_rows(hfq, model_path, request, arch)?;
    Ok(())
}

#[test]
fn state_loader_split_base_refuses_missing_q4_without_derivation() {
    use hipfire_config::memory::PleFormatRequest;
    use hipfire_runtime::external_rows::RowEncoding;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
    use hipfire_runtime::ple_hot_pack::{
        encode_ple_table_header, PackKind, PackOrder, PleHotPackHeader, PleTableFields,
        TABLE_SEGMENT_BYTES,
    };
    use hipfire_runtime::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};
    use sha2::{Digest, Sha256};
    let dir = std::env::temp_dir().join(format!(
        "hipfire-state-split-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("state-split.hfqm");
    let metadata = json!({
        "qwen4_ple": {
            "version": 1, "multipliers": [-3, 5, 7],
            "head_vocab_sizes": vec![0; 16], "head_offsets": vec![0; 16], "padded_rows": 12
        },
        "hipfire_ple_rows": {
            "version": 1, "location": "sidecar", "shard_count": 3,
            "rows_per_shard": 4, "row_width": 64, "q8_rows_sha256": "ab".repeat(32)
        }
    });
    let tensor = HfqMemTensor {
        name: "model.language_model.layers.1.ple.ple_embedding.layer_multipliers".into(),
        quant_type: 54,
        shape: vec![3],
        group_size: 0,
        data: vec![0; 24],
    };
    write_hfqm_package_mem(&path, 16, &metadata.to_string(), &[tensor]).unwrap();
    let mut hfq = HfqFile::open(&path).unwrap();
    let error =
        attach_state_validation_rows(&mut hfq, &path, PleFormatRequest::Q4, "gfx1201").unwrap_err();
    assert!(error.contains(".ple-q4"), "{error}");
    assert!(hfq.ple_rows().is_none());
    assert!(!path.with_extension("hfqm.ple-q4").exists());
    // An existing, correctly bound Q4 table must be selected without deriving
    // or consulting process-global format/GPU settings.
    let sidecar = path.with_extension("hfqm.ple-q4");
    let data = vec![0u8; 12 * RowEncoding::Q4Lm.encoded_row_bytes(64)];
    let segment_digest = Sha256::digest(&data);
    let mut header = PleHotPackHeader {
        padded_rows: 12,
        rows_per_shard: 4,
        row_encoding: RowEncoding::Q4Lm,
        row_width: 64,
        encoded_row_bytes: RowEncoding::Q4Lm.encoded_row_bytes(64) as u64,
        ple_fingerprint: hipfire_runtime::ple_sidecar::ple_fingerprint_from_metadata(
            &metadata.to_string(),
        )
        .unwrap(),
        rank_payload_sha256: "0".repeat(64),
        tier_group: 256,
        kept_buckets: 256,
        tier_rows: vec![12],
        data_offset: 0,
        data_bytes: data.len() as u64,
        shard_offsets: Vec::new(),
        model_file_bytes: 0,
        version: 3,
        kind: PackKind::PleTable,
        order: PackOrder::Row,
        rank_offset: 0,
        rank_bytes: 0,
        table: Some(PleTableFields {
            codebook_f32_bits: Some(lm4_codebook_bits()),
            scale_grid_f32_bits: Some(lm4_scale_grid_bits()),
            source_encoding: RowEncoding::Q8F16,
            source_rows_sha256: "ab".repeat(32),
            segment_bytes: TABLE_SEGMENT_BYTES,
            segment_sha256: vec![format!("{segment_digest:x}")],
            data_sha256: format!("{:x}", Sha256::digest(segment_digest)),
            base_index_sha256: hfq
                .index_sha256()
                .unwrap()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
            base_file_bytes: hfq.file_len(),
        }),
        generator: "hipfire state parity CPU regression".to_string(),
    };
    let mut bytes = encode_ple_table_header(&mut header).unwrap();
    bytes.extend_from_slice(&data);
    std::fs::write(&sidecar, bytes).unwrap();
    let mut hfq = HfqFile::open(&path).unwrap();
    attach_state_validation_rows(&mut hfq, &path, PleFormatRequest::Q4, "gfx1201").unwrap();
    let rows = hfq.ple_rows().unwrap();
    assert_eq!(rows.path(), sidecar);
    assert_eq!(rows.encoding(), RowEncoding::Q4Lm);
    drop(hfq);
    std::fs::remove_dir_all(dir).unwrap();
}

const MAX_SEQ: usize = 8;
const PREFIX: usize = 4;
const DRAFTS: [u32; 2] = [1001, 1002];
const BONUS: u32 = 1003;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug)]
struct Family {
    hash: u64,
    numel: usize,
}

impl Family {
    fn new() -> Self {
        Self {
            hash: FNV_OFFSET,
            numel: 0,
        }
    }

    fn bytes(&mut self, bytes: &[u8], numel: usize) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(FNV_PRIME);
        }
        self.numel = self.numel.saturating_add(numel);
    }

    fn finish(self) -> Value {
        json!({"digest": format!("{:016x}", self.hash), "numel": self.numel})
    }
}

type Families = BTreeMap<String, Value>;

pub fn run_compact(gpu: &mut Gpu) -> Result<Value, String> {
    let config = compact_test_config();
    // The product's automatic QSA storage (VMM where supported), mapped for
    // the whole parity context: the oracle writes and reads whole arenas.
    let backend = Qwen4KvBackend::automatic(gpu);
    let format = crate::state::Qwen4StateFormat::F32;
    let mut ar = Qwen4State::new_with_backend(gpu, &config, MAX_SEQ, format, backend)
        .map_err(|error| format!("allocate compact AR state: {error}"))?;
    let mut native = match Qwen4State::new_with_backend(gpu, &config, MAX_SEQ, format, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            return Err(format!("allocate compact native target state: {error}"));
        }
    };
    let mut direct = match MtpGpuState::new_with_backend(gpu, &config, MAX_SEQ, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            return Err(format!("allocate compact direct MTP state: {error}"));
        }
    };
    let mut mtp = match MtpGpuState::new_with_backend(gpu, &config, MAX_SEQ, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            let _ = direct.free_gpu(gpu);
            return Err(format!("allocate compact native MTP state: {error}"));
        }
    };
    let mapped = (|| -> Result<(), String> {
        for state in [&mut ar, &mut native] {
            state
                .ensure_mapped_capacity(gpu, MAX_SEQ)
                .map_err(|error| error.to_string())?;
        }
        for state in [&mut direct, &mut mtp] {
            state
                .ensure_mapped_capacity(gpu, MAX_SEQ)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    })()
    .map_err(|error| format!("map compact {} QSA context: {error}", backend.name()));
    let result = mapped
        .and_then(|()| run_inner(gpu, &config, &mut ar, &mut native, &mut direct, &mut mtp))
        .map(|mut value| {
            value["qsa_backend"] = json!(backend.name());
            value
        });
    let cleanup = [
        ar.free_gpu(gpu).err().map(|error| error.to_string()),
        native.free_gpu(gpu).err().map(|error| error.to_string()),
        direct.free_gpu(gpu).map(|error| error.to_string()),
        mtp.free_gpu(gpu).map(|error| error.to_string()),
    ]
    .into_iter()
    .flatten()
    .next();
    result.and_then(|value| cleanup.map_or(Ok(value), Err))
}

fn run_inner(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    let boundary = prepare(gpu, config, ar, native, direct, mtp)?;
    let mut cases = Vec::new();
    for accepted in 0..=DRAFTS.len() {
        cases.push(acceptance_case(
            gpu, config, ar, native, direct, mtp, accepted,
        )?);
    }
    let scenarios = vec![
        boundary,
        stale_ticket(gpu, ar, direct)?,
        terminal_seed(gpu, config, ar, direct)?,
        rollback_failure(gpu, config, ar, direct, "cancellation", None)?,
        rollback_failure(
            gpu,
            config,
            ar,
            direct,
            "injected_replay_failure",
            Some("injected replay error"),
        )?,
        cache_suffix_refusal(),
    ];
    let pass = cases.iter().all(is_pass) && scenarios.iter().all(is_pass);
    Ok(json!({
        "schema": "hipfire.qwen4.state_parity.compact.v1",
        "fixture": "canonical_compact_state",
        "gpu_arch": gpu.arch,
        "acceptance_cases": cases,
        "scenarios": scenarios,
        "status": if pass {"pass"} else {"fail"},
    }))
}

fn is_pass(value: &Value) -> bool {
    value.get("status") == Some(&Value::String("pass".into()))
}

fn prepare(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    reset_target(gpu, ar)?;
    reset_target(gpu, native)?;
    reset_mtp(gpu, direct)?;
    reset_mtp(gpu, mtp)?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_pair(gpu, config, ar, native, position, token)?;
        append_mtp_pair(gpu, config, direct, mtp, position, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX)?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX)?;
    reuse(gpu, config, direct)?;
    reuse(gpu, config, mtp)?;
    let direct_buf = direct.parity_buffers();
    let mtp_buf = mtp.parity_buffers();
    let direct_len = download_i32(gpu, direct_buf.selected_len_out, 1)?[0];
    let mtp_len = download_i32(gpu, mtp_buf.selected_len_out, 1)?[0];
    let direct_selection = download_i32(gpu, direct_buf.selected_indices, PREFIX + 1)?;
    let mtp_selection = download_i32(gpu, mtp_buf.selected_indices, PREFIX + 1)?;
    let direct_meta = direct.parity_metadata();
    let mtp_meta = mtp.parity_metadata();
    let target_meta = ar.qsa.first().map(|qsa| {
        json!({
            "full_len": qsa.full_len,
            "raw_len": qsa.raw_len,
            "pooled_len": qsa.pooled_len,
            "selected_len": qsa.selected_len,
            "position": qsa.position,
        })
    });
    let pass = target_meta.is_some()
        && direct_meta == mtp_meta
        && direct_meta.full_len == PREFIX
        && direct_meta.raw_len == PREFIX
        && direct_meta.pooled_len == 1
        && direct_meta.selected_len == PREFIX
        && direct_len == (PREFIX + 1) as i32
        && direct_len == mtp_len
        && direct_selection == [0, 1, 2, 3, 4]
        && direct_selection == mtp_selection;
    Ok(json!({
        "case": "qsa_pooling_boundary",
        "status": if pass {"pass"} else {"fail"},
        "target_metadata": target_meta,
        "mtp_direct_metadata": metadata_json(direct_meta),
        "mtp_native_metadata": metadata_json(mtp_meta),
        "reuse_selected_len": {"direct": direct_len, "native": mtp_len, "expected": PREFIX + 1},
        "reuse_selected_indices": {"direct": direct_selection, "native": mtp_selection},
    }))
}

fn acceptance_case(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
    accepted: usize,
) -> Result<Value, String> {
    prepare(gpu, config, ar, native, direct, mtp)?;
    let target_ticket = native.snapshot(gpu).map_err(|error| error.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|error| error.to_string())?;
    for (index, &token) in DRAFTS.iter().enumerate() {
        append_target_single(gpu, config, native, PREFIX + index, token)?;
        append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
    }
    let committed = committed(accepted);
    if accepted < DRAFTS.len() {
        mutate_target(gpu, native)?;
        mutate_mtp(gpu, mtp)?;
        native
            .restore_retain(gpu, target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.restore_retain(gpu, mtp_ticket)
            .map_err(|error| error.to_string())?;
        for (index, &token) in committed.iter().enumerate() {
            append_target_single(gpu, config, native, PREFIX + index, token)?;
            append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
        }
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
    } else {
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
        append_target_single(gpu, config, native, PREFIX + DRAFTS.len(), BONUS)?;
        append_mtp_single(gpu, config, mtp, PREFIX + DRAFTS.len(), BONUS)?;
    }
    for (index, &token) in committed.iter().enumerate() {
        append_target_single(gpu, config, ar, PREFIX + index, token)?;
        append_mtp_single(gpu, config, direct, PREFIX + index, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX + committed.len())?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX + committed.len())?;
    let target = compare_family_maps(
        &target_families(gpu, config, ar)?,
        &target_families(gpu, config, native)?,
    );
    let mtp_families = compare_family_maps(
        &mtp_families(gpu, config, direct)?,
        &mtp_families(gpu, config, mtp)?,
    );
    let pass = target.0 && mtp_families.0;
    Ok(json!({
        "case": match accepted {0 => "zero", 1 => "one", _ => "all"},
        "accepted_drafts": accepted,
        "committed_tokens": committed,
        "status": if pass {"pass"} else {"fail"},
        "target_families": target.1,
        "mtp_families": mtp_families.1,
    }))
}

fn committed(accepted: usize) -> Vec<u32> {
    let mut result = DRAFTS[..accepted].to_vec();
    result.push(BONUS);
    result
}

fn reset_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn reset_mtp(gpu: &mut Gpu, state: &mut MtpGpuState) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn append_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_target_single(gpu, config, first, position, token)?;
    append_target_single(gpu, config, second, position, token)
}

fn append_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_mtp_single(gpu, config, first, position, token)?;
    append_mtp_single(gpu, config, second, position, token)
}

fn append_target_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    for (layer_index, qsa) in state.qsa.iter_mut().enumerate() {
        let key_values = row(full_width, layer_index, position, token, 0.01);
        let value_values = row(full_width, layer_index, position, token, -0.02);
        let raw_values = row(raw_width, layer_index, position, token, 0.03);
        let key = gpu
            .upload_f32(&key_values, &[full_width])
            .map_err(|e| e.to_string())?;
        let value = match gpu.upload_f32(&value_values, &[full_width]) {
            Ok(value) => value,
            Err(error) => {
                let _ = gpu.free_tensor(key);
                return Err(error.to_string());
            }
        };
        let result = (|| {
            qsa.invalidate_f16_arena();
            indexed_attention_cache_append(
                gpu,
                &IndexedAttentionCacheAppend {
                    key: &key,
                    value: &value,
                    full_keys: &qsa.full_keys,
                    full_values: &qsa.full_values,
                    position,
                    kv_width: full_width,
                },
            )
            .map_err(|e| e.to_string())?;
            write_raw(gpu, &qsa.raw_index_keys, position * raw_width, &raw_values)?;
            qsa.full_len = position + 1;
            qsa.raw_len = position + 1;
            qsa.position = position + 1;
            Ok::<(), String>(())
        })();
        let _ = gpu.free_tensor(key);
        let _ = gpu.free_tensor(value);
        result?;
    }
    state.position = position + 1;
    Ok(())
}

fn append_mtp_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let key_values = row(full_width, 0, position, token, 0.01);
    let value_values = row(full_width, 0, position, token, -0.02);
    let raw_values = row(raw_width, 0, position, token, 0.03);
    let key = gpu
        .upload_f32(&key_values, &[full_width])
        .map_err(|e| e.to_string())?;
    let value = match gpu.upload_f32(&value_values, &[full_width]) {
        Ok(value) => value,
        Err(error) => {
            let _ = gpu.free_tensor(key);
            return Err(error.to_string());
        }
    };
    let result = (|| {
        let metadata = state.parity_metadata();
        let buffers = state.parity_buffers();
        indexed_attention_cache_append(
            gpu,
            &IndexedAttentionCacheAppend {
                key: &key,
                value: &value,
                full_keys: buffers.full_keys,
                full_values: buffers.full_values,
                position,
                kv_width: full_width,
            },
        )
        .map_err(|e| e.to_string())?;
        write_raw(
            gpu,
            buffers.raw_index_keys,
            position * raw_width,
            &raw_values,
        )?;
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: position + 1,
            raw_len: position + 1,
            pooled_len: metadata.pooled_len,
            selected_len: metadata.selected_len,
            position: position + 1,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(key);
    let _ = gpu.free_tensor(value);
    result
}

fn row(width: usize, layer: usize, position: usize, token: u32, scale: f32) -> Vec<f32> {
    (0..width)
        .map(|index| {
            let phase = (layer * 17 + position * 31 + index * 7 + token as usize) as f32;
            (phase.sin() * 0.25 + 0.5) * scale
        })
        .collect()
}

fn select_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    select_target(gpu, config, first, visible)?;
    select_target(gpu, config, second, visible)
}

fn select_target(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        for qsa in &mut state.qsa {
            // Rewrites pooled keys outside the forward: drop the BF16 mirror.
            qsa.invalidate_f16_arena();
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &qsa.raw_index_keys,
                    pooled: &qsa.pooled_keys,
                    norm: None,
                    block_count: visible / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    index_dim: raw_width,
                    position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                        position_start: visible.saturating_sub(1),
                        rows: 1,
                    }),
                    grid_bound: visible / config.indexer_compress_ratio,
                    yarn: config.rope().yarn(raw_width.min(64)),
                },
            )
            .map_err(|e| e.to_string())?;
            indexed_attention_select(
                gpu,
                &IndexedAttentionSelect {
                    query: &query,
                    pooled: &qsa.pooled_keys,
                    selected: &qsa.selected_indices,
                    block_count: visible / config.indexer_compress_ratio,
                    index_heads: config.indexer_n_heads,
                    index_dim: config.indexer_head_dim,
                    budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    visible,
                    capacity: config.qsa_selected_capacity(),
                },
            )
            .map_err(|e| e.to_string())?;
            qsa.pooled_len = visible / config.indexer_compress_ratio;
            qsa.selected_len = visible;
        }
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn select_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    select_mtp(gpu, config, first, visible)?;
    select_mtp(gpu, config, second, visible)
}

fn select_mtp(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        let buffers = state.parity_buffers();
        indexed_attention_pool_rope(
            gpu,
            &IndexedAttentionPoolRope {
                raw_keys: buffers.raw_index_keys,
                pooled: buffers.pooled_keys,
                norm: None,
                block_count: visible / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                index_dim: raw_width,
                position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                    position_start: visible.saturating_sub(1),
                    rows: 1,
                }),
                grid_bound: visible / config.indexer_compress_ratio,
                yarn: config.rope().yarn(raw_width.min(64)),
            },
        )
        .map_err(|e| e.to_string())?;
        indexed_attention_select(
            gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: buffers.pooled_keys,
                selected: buffers.selected_indices,
                block_count: visible / config.indexer_compress_ratio,
                index_heads: config.indexer_n_heads,
                index_dim: config.indexer_head_dim,
                budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                visible,
                capacity: config.qsa_selected_capacity(),
            },
        )
        .map_err(|e| e.to_string())?;
        let metadata = state.parity_metadata();
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: metadata.full_len,
            raw_len: metadata.raw_len,
            pooled_len: visible / config.indexer_compress_ratio,
            selected_len: visible,
            position: metadata.position,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn reuse(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<(), String> {
    let buffers = state.parity_buffers();
    indexed_attention_reuse_selection(
        gpu,
        &IndexedAttentionReuseSelection {
            selected: buffers.selected_indices,
            selected_len: PREFIX,
            position: PREFIX,
            capacity: config.qsa_selected_capacity(),
            selected_len_out: buffers.selected_len_out,
        },
    )
    .map_err(|e| e.to_string())
}

fn query(config: &crate::config::Qwen4Config) -> Vec<f32> {
    (0..config.indexer_n_heads * config.indexer_head_dim)
        .map(|index| (index as f32 * 0.013).cos() * 0.125)
        .collect()
}

fn write_raw(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
    offset_elements: usize,
    values: &[f32],
) -> Result<(), String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    let view = tensor.sub_offset(offset_elements, values.len());
    gpu.hip
        .memcpy_htod(&view.buf, bytes)
        .map_err(|error| error.to_string())
}

fn write_scalar_f32(gpu: &mut Gpu, tensor: &GpuTensor, value: f32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 1);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn write_scalar_i32(gpu: &mut Gpu, tensor: &GpuTensor, value: i32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 4);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn mutate_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    for layer in &state.gdn {
        write_scalar_f32(gpu, &layer.recurrent, 91.0)?;
        write_scalar_f32(gpu, &layer.conv, 92.0)?;
    }
    for layer in &state.qsa {
        write_scalar_f32(gpu, &layer.partial_keys, 93.0)?;
        write_scalar_f32(gpu, &layer.partial_values, 94.0)?;
        write_scalar_i32(gpu, &layer.selected_indices, -93)?;
    }
    if let Some(ple_conv) = &state.ple_conv {
        write_scalar_f32(gpu, ple_conv, 95.0)?;
    }
    let hyper_feedback = state
        .hyper_feedback
        .as_ref()
        .expect("hyper_feedback is input-band state");
    write_scalar_f32(gpu, hyper_feedback, 96.0)
}

fn mutate_mtp(gpu: &mut Gpu, state: &MtpGpuState) -> Result<(), String> {
    let buffers = state.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, -97)?;
    write_scalar_i32(gpu, buffers.selected_len_out, 97)?;
    write_scalar_f32(gpu, buffers.wide_hidden, 98.0)
}

fn target_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &Qwen4State,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let mut families = BTreeMap::new();
    let mut recurrent = Family::new();
    let mut conv = Family::new();
    for layer in &state.gdn {
        append_f32(
            gpu,
            &mut recurrent,
            &layer.recurrent,
            layer.recurrent.numel(),
        )?;
        append_f32(gpu, &mut conv, &layer.conv, layer.conv.numel())?;
    }
    families.insert("gdn_recurrent".into(), recurrent.finish());
    families.insert("gdn_conv".into(), conv.finish());
    let mut qsa_families = [
        ("qsa_full_keys", Family::new()),
        ("qsa_full_values", Family::new()),
        ("qsa_raw_index_keys", Family::new()),
        ("qsa_pooled_keys", Family::new()),
        ("qsa_partial_keys", Family::new()),
        ("qsa_partial_values", Family::new()),
        ("qsa_selected_indices", Family::new()),
    ];
    let mut metadata = Family::new();
    for qsa in &state.qsa {
        append_f32(
            gpu,
            &mut qsa_families[0].1,
            &qsa.full_keys,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[1].1,
            &qsa.full_values,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[2].1,
            &qsa.raw_index_keys,
            qsa.raw_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[3].1,
            &qsa.pooled_keys,
            qsa.pooled_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[4].1,
            &qsa.partial_keys,
            qsa.partial_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[5].1,
            &qsa.partial_values,
            qsa.partial_len * full_width,
        )?;
        append_raw(
            gpu,
            &mut qsa_families[6].1,
            &qsa.selected_indices,
            qsa.selected_len * 4,
            qsa.selected_len,
        )?;
        metadata_usize(
            &mut metadata,
            [
                qsa.full_len,
                qsa.raw_len,
                qsa.pooled_len,
                qsa.partial_len,
                qsa.selected_len,
                qsa.position,
            ],
        );
    }
    metadata_usize(&mut metadata, [state.position, state.max_seq_len]);
    for (name, family) in qsa_families {
        families.insert(name.into(), family.finish());
    }
    if let Some(ple_conv) = &state.ple_conv {
        families.insert("ple_conv".into(), tensor(gpu, ple_conv)?);
    }
    let hyper_feedback = state
        .hyper_feedback
        .as_ref()
        .expect("hyper_feedback is input-band state");
    families.insert("hyper_feedback".into(), tensor(gpu, hyper_feedback)?);
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn mtp_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let metadata_values = state.parity_metadata();
    let buffers = state.parity_buffers();
    let mut families = BTreeMap::new();
    families.insert(
        "full_keys".into(),
        prefix(
            gpu,
            buffers.full_keys,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "full_values".into(),
        prefix(
            gpu,
            buffers.full_values,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "raw_index_keys".into(),
        prefix(
            gpu,
            buffers.raw_index_keys,
            metadata_values.raw_len * raw_width,
        )?,
    );
    families.insert(
        "pooled_keys".into(),
        prefix(
            gpu,
            buffers.pooled_keys,
            metadata_values.pooled_len * raw_width,
        )?,
    );
    families.insert(
        "selected_indices".into(),
        raw_prefix(
            gpu,
            buffers.selected_indices,
            metadata_values.selected_len,
            metadata_values.selected_len,
        )?,
    );
    families.insert(
        "selected_len_out".into(),
        raw_prefix(gpu, buffers.selected_len_out, 1, 1)?,
    );
    families.insert("wide_hidden".into(), tensor(gpu, buffers.wide_hidden)?);
    let mut metadata = Family::new();
    metadata_usize(
        &mut metadata,
        [
            metadata_values.full_len,
            metadata_values.raw_len,
            metadata_values.pooled_len,
            metadata_values.selected_len,
            metadata_values.position,
            metadata_values.step_index,
        ],
    );
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn tensor(gpu: &Gpu, tensor: &GpuTensor) -> Result<Value, String> {
    prefix(gpu, tensor, tensor.numel())
}

fn prefix(gpu: &Gpu, tensor: &GpuTensor, elements: usize) -> Result<Value, String> {
    let mut family = Family::new();
    append_f32(gpu, &mut family, tensor, elements)?;
    Ok(family.finish())
}

fn raw_prefix(
    gpu: &Gpu,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<Value, String> {
    let mut family = Family::new();
    append_raw(gpu, &mut family, tensor, elements, numel)?;
    Ok(family.finish())
}

fn append_f32(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let values = gpu.download_f32(&view).map_err(|error| error.to_string())?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    family.bytes(bytes, values.len());
    Ok(())
}

fn append_raw(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let mut bytes = vec![0u8; view.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &view.buf)
        .map_err(|error| error.to_string())?;
    family.bytes(&bytes, numel);
    Ok(())
}

fn metadata_usize<const N: usize>(family: &mut Family, values: [usize; N]) {
    let mut bytes = Vec::with_capacity(N * std::mem::size_of::<usize>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    family.bytes(&bytes, N);
}

fn compare_family_maps(left: &Families, right: &Families) -> (bool, Value) {
    let keys = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut result = Map::new();
    let mut pass = true;
    for key in keys {
        let left_value = left.get(&key);
        let right_value = right.get(&key);
        let left_digest = left_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let right_digest = right_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let matches = left_digest == right_digest
            && left_value.and_then(|value| value.get("numel"))
                == right_value.and_then(|value| value.get("numel"));
        pass &= matches;
        result.insert(
            key,
            json!({
                "ar_or_direct": left_value,
                "native": right_value,
                "match": matches,
            }),
        );
    }
    (pass, Value::Object(result))
}

fn metadata_json(metadata: MtpStateParityMetadata) -> Value {
    json!({
        "full_len": metadata.full_len,
        "raw_len": metadata.raw_len,
        "pooled_len": metadata.pooled_len,
        "selected_len": metadata.selected_len,
        "position": metadata.position,
        "step_index": metadata.step_index,
    })
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, count: usize) -> Result<Vec<i32>, String> {
    let mut values = vec![0i32; count];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, count * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .map_err(|e| e.to_string())?;
    Ok(values)
}

fn stale_ticket(
    gpu: &mut Gpu,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_refusal = target
        .restore(gpu, target_ticket)
        .err()
        .map(|e| e.to_string());
    let mtp_refusal = mtp.restore(gpu, mtp_ticket).err().map(|e| e.to_string());
    let unchanged = target.position == 0 && mtp.parity_metadata().position == 0;
    Ok(json!({
        "case": "stale_ticket",
        "status": if target_refusal.is_some() && mtp_refusal.is_some() && unchanged {"pass"} else {"fail"},
        "target_refusal": target_refusal,
        "mtp_refusal": mtp_refusal,
        "state_unchanged": unchanged,
    }))
}

fn terminal_seed(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    append_target_single(gpu, config, target, 0, config.eos_token_id)?;
    append_mtp_single(gpu, config, mtp, 0, config.eos_token_id)?;
    for qsa in &mut target.qsa {
        write_scalar_i32(gpu, &qsa.selected_indices, 0)?;
        qsa.selected_len = 1;
        qsa.position = 1;
    }
    target.position = 1;
    let metadata = mtp.parity_metadata();
    let buffers = mtp.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, 0)?;
    mtp.parity_set_metadata(MtpStateParityMetadata {
        full_len: 1,
        raw_len: 1,
        pooled_len: 0,
        selected_len: 1,
        position: 1,
        step_index: metadata.step_index,
    });
    let pass = target.position == 1 && mtp.parity_metadata().position == 1;
    Ok(json!({
        "case": "forced_terminal_seed",
        "status": if pass {"pass"} else {"fail"},
        "forced_token": config.eos_token_id,
        "terminal": true,
        "target_metadata": {"position": target.position, "selected_len": target.qsa.first().map(|qsa| qsa.selected_len)},
        "mtp_metadata": metadata_json(mtp.parity_metadata()),
    }))
}

fn rollback_failure(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
    case: &str,
    injected: Option<&str>,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_single(gpu, config, target, position, token)?;
        append_mtp_single(gpu, config, mtp, position, token)?;
    }
    select_target(gpu, config, target, PREFIX)?;
    select_mtp(gpu, config, mtp, PREFIX)?;
    let before_target = target_families(gpu, config, target)?;
    let before_mtp = mtp_families(gpu, config, mtp)?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    mutate_target(gpu, target)?;
    mutate_mtp(gpu, mtp)?;
    let failure = injected.unwrap_or("cancelled by caller").to_string();
    target
        .restore(gpu, target_ticket)
        .map_err(|e| e.to_string())?;
    mtp.restore(gpu, mtp_ticket).map_err(|e| e.to_string())?;
    let target_compare =
        compare_family_maps(&before_target, &target_families(gpu, config, target)?);
    let mtp_compare = compare_family_maps(&before_mtp, &mtp_families(gpu, config, mtp)?);
    let pass = target_compare.0 && mtp_compare.0;
    Ok(json!({
        "case": case,
        "status": if pass {"pass"} else {"fail"},
        "replay_error": failure,
        "target_rollback": target_compare.1,
        "mtp_rollback": mtp_compare.1,
    }))
}

/// Position-zero cache hit refusal. A valid cache hit resumes a session
/// snapshot at `0 < start_pos < prompt_len` with exactly the prompt suffix, so
/// `start_pos == 0` with `cache_hit == true` is an invalid zero-length cached
/// prefix (it is not an empty fill: the fill is the complete 3-token prompt).
fn cache_suffix_refusal() -> Value {
    // expect_err: a cache hit at position zero is invalid; valid suffix cache
    // hits (0 < start_pos < prompt_len) are supported and are not exercised here.
    let refusal = validate_native_mtp_prefill_request(&[1, 2, 3], &[1, 2, 3], 0, true)
        .expect_err("position-zero cache hit must be refused");
    json!({
        "case": "cache_suffix_refusal",
        "status": "pass",
        "refusal": refusal,
        "cache_hit": true,
        "prompt_len": 3,
        "fill_len": 3,
    })
}

/// JSON report returned by the state parity orchestration entrypoint.
pub struct StateParityReport(Value);

impl StateParityReport {
    pub fn into_json(self) -> Value {
        self.0
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&self.0).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

const PROFILE_CHECKPOINT_SCHEMA: &str = "hipfire.qwen4.profile_checkpoint.v1";
const PROFILE_CHECKPOINT_PREFIX: &str = "HIPFIRE_QWEN4_PROFILE_CHECKPOINT ";

const PROFILE_SOURCE_CALLBACK_SCHEMA: &str = "hipfire.qwen4.profile_source_callback.v1";
const PROFILE_SOURCE_CALLBACK_PREFIX: &str = "HIPFIRE_QWEN4_PROFILE_SOURCE_CALLBACK ";

fn profile_elapsed_between_ns(started: Instant, ended: Instant) -> u64 {
    ended
        .duration_since(started)
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn profile_source_residency(residency: WeightResidency) -> Value {
    match residency {
        WeightResidency::Resident => json!("resident"),
        WeightResidency::HostMapped => json!("host_mapped"),
        WeightResidency::ExternalRows {
            row_bytes,
            valid_rows,
        } => json!({
            "kind": "external_rows",
            "row_bytes": row_bytes,
            "valid_rows": valid_rows,
        }),
    }
}

fn emit_profile_source_callback(
    index: u64,
    entry: &WeightEntry,
    elapsed_ns: u64,
    cumulative_ns: u64,
) {
    let checkpoint = json!({
        "schema": PROFILE_SOURCE_CALLBACK_SCHEMA,
        "index": index,
        "name": &entry.name,
        "layer": entry.layer,
        "residency": profile_source_residency(entry.residency),
        "elapsed_ns": elapsed_ns,
        "cumulative_ns": cumulative_ns,
    });
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{PROFILE_SOURCE_CALLBACK_PREFIX}{checkpoint}");
    let _ = stderr.flush();
}

fn emit_profile_checkpoint(
    phase: &str,
    wall_ns: u64,
    hip: &Value,
    internal: Option<&Value>,
    ple: Option<&Value>,
) {
    let checkpoint = json!({
        "schema": PROFILE_CHECKPOINT_SCHEMA,
        "phase": phase,
        "wall_ns": wall_ns,
        "hip": hip,
        "internal": internal.cloned().unwrap_or(Value::Null),
        "ple": ple.cloned().unwrap_or(Value::Null),
    });
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{PROFILE_CHECKPOINT_PREFIX}{checkpoint}");
    let _ = stderr.flush();
}

fn emit_profile_load_checkpoint(phase: &str, wall_ns: u64) {
    let hip = hip_counter_snapshot();
    emit_profile_checkpoint(phase, wall_ns, &hip, None, None);
}

fn profile_duration_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn hip_counter_entry(calls: u64, ffi_ns: u64, bytes: u64) -> Value {
    json!({
        "calls": calls,
        "ffi_ns": ffi_ns,
        "bytes": bytes,
    })
}

fn hip_counter_snapshot() -> Value {
    json!({
        "ffi_total_calls": launch_counters::count(),
        "ffi_total_ns": launch_counters::time_ns(),
        "launch_kernel": hip_counter_entry(
            launch_counters::launch_kernel::count(),
            launch_counters::launch_kernel::time_ns(),
            launch_counters::launch_kernel::bytes(),
        ),
        "memcpy_dtod": hip_counter_entry(
            launch_counters::memcpy_dtod::count(),
            launch_counters::memcpy_dtod::time_ns(),
            launch_counters::memcpy_dtod::bytes(),
        ),
        "memcpy_htod": hip_counter_entry(
            launch_counters::memcpy_htod::count(),
            launch_counters::memcpy_htod::time_ns(),
            launch_counters::memcpy_htod::bytes(),
        ),
        "memcpy_dtoh": hip_counter_entry(
            launch_counters::memcpy_dtoh::count(),
            launch_counters::memcpy_dtoh::time_ns(),
            launch_counters::memcpy_dtoh::bytes(),
        ),
        "memset": hip_counter_entry(
            launch_counters::memset::count(),
            launch_counters::memset::time_ns(),
            launch_counters::memset::bytes(),
        ),
        "ensure_kernel_lookup": hip_counter_entry(
            launch_counters::ensure_kernel_lookup::count(),
            launch_counters::ensure_kernel_lookup::time_ns(),
            launch_counters::ensure_kernel_lookup::bytes(),
        ),
        "stream_sync": hip_counter_entry(
            launch_counters::stream_sync::count(),
            launch_counters::stream_sync::time_ns(),
            launch_counters::stream_sync::bytes(),
        ),
        "event_sync": hip_counter_entry(
            launch_counters::event_sync::count(),
            launch_counters::event_sync::time_ns(),
            launch_counters::event_sync::bytes(),
        ),
        "device_sync": hip_counter_entry(
            launch_counters::device_sync::count(),
            launch_counters::device_sync::time_ns(),
            launch_counters::device_sync::bytes(),
        ),
        "graph_launch": hip_counter_entry(
            launch_counters::graph_launch::count(),
            launch_counters::graph_launch::time_ns(),
            launch_counters::graph_launch::bytes(),
        ),
    })
}

fn qwen4_profile_stats_json(stats: Qwen4ProfileStats) -> Value {
    Value::Object(
        Qwen4ProfilePhase::ALL
            .into_iter()
            .map(|phase| {
                let slot = phase as usize;
                (
                    phase.name().to_string(),
                    json!({ "calls": stats.calls[slot], "host_ns": stats.ns[slot] }),
                )
            })
            .collect(),
    )
}
const PROFILE_DIRTY_BYTE: i32 = 0x3f;

fn dirty_qwen4_moe_scratch(gpu: &mut Gpu, scratch: &Qwen4GpuForwardScratch) -> Result<(), String> {
    for tensor in [
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
    ] {
        gpu.hip
            .memset(&tensor.buf, PROFILE_DIRTY_BYTE, tensor.buf.size())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn dirty_profile_target_moe_reuse(
    gpu: &mut Gpu,
    bundle: &crate::bundle::Qwen4Bundle,
) -> Result<(), String> {
    let forward = bundle
        .execution
        .as_ref()
        .ok_or_else(|| "qwen4 profile forward resources are not attached".to_string())?;
    dirty_qwen4_moe_scratch(gpu, &forward.scratch)
}

fn dirty_profile_mtp_moe_reuse(
    gpu: &mut Gpu,
    bundle: &crate::bundle::Qwen4Bundle,
) -> Result<(), String> {
    let mtp = bundle
        .mtp
        .as_ref()
        .ok_or_else(|| "qwen4 profile MTP resources are not attached".to_string())?;
    mtp.dirty_moe_reuse(gpu).map_err(|error| error.to_string())
}

fn sealed_moe_profile_evidence(stats: &Value, dirty_reuse: bool) -> Result<Value, String> {
    let seal = stats
        .get("moe_seal")
        .ok_or_else(|| "qwen4 profile has no MoE seal counters".to_string())?;
    let calls = seal
        .get("calls")
        .and_then(Value::as_u64)
        .ok_or_else(|| "qwen4 profile MoE seal counter is malformed".to_string())?;
    if calls == 0 {
        return Err("qwen4 profile executed no sealed MoE calls".to_string());
    }
    Ok(json!({
        "calls": calls,
        "host_ns": seal.get("host_ns").cloned().unwrap_or(Value::Null),
        "route": "BoundMoeExperts::from_cache -> seal_decode -> execute_steps(Step::Moe)",
        "sealed_route_executed": true,
        "dirty_reuse": dirty_reuse,
        "dirty_pattern_byte": PROFILE_DIRTY_BYTE,
    }))
}

fn ple_cache_stats_json(stats: RowCacheStats) -> Value {
    json!({
        "capacity_bytes": stats.capacity_bytes,
        "resident_bytes": stats.resident_bytes,
        "resident_pages": stats.resident_pages,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "reads": stats.reads,
        "coalesced_reads": stats.coalesced_reads,
        "read_bytes": stats.read_bytes,
        "evictions": stats.evictions,
        "queue_depth": stats.queue_depth,
        "outstanding_readers": stats.outstanding_readers,
        "outstanding_leases": stats.outstanding_leases,
        "staging_in_use": stats.staging_in_use,
        "staging_high_water": stats.staging_high_water,
    })
}

fn ple_cache_delta(before: RowCacheStats, after: RowCacheStats) -> Value {
    json!({
        "cache_hits": after.cache_hits.saturating_sub(before.cache_hits),
        "cache_misses": after.cache_misses.saturating_sub(before.cache_misses),
        "reads": after.reads.saturating_sub(before.reads),
        "coalesced_reads": after.coalesced_reads.saturating_sub(before.coalesced_reads),
        "read_bytes": after.read_bytes.saturating_sub(before.read_bytes),
        "evictions": after.evictions.saturating_sub(before.evictions),
    })
}

fn target_layer_families_json(config: &crate::config::Qwen4Config) -> Value {
    json!({
        "linear_attention_layers": config.n_linear_layers(),
        "full_attention_layers": config.n_full_layers(),
        "moe_layers": config.num_hidden_layers,
        "ple_layer_ids": config.ple_layer_ids,
        "attribution": "rocprofv3 kernel trace by symbol; HIP counters are phase totals",
    })
}

/// Map the bundle's target and MTP QSA context arenas for all `tokens` of
/// the (small) parity context: the probes read whole arenas, not only the
/// rows a forward has mapped.
fn map_bundle_context(
    gpu: &mut Gpu,
    bundle: &mut crate::bundle::Qwen4Bundle,
    tokens: usize,
) -> Result<(), String> {
    bundle
        .state
        .ensure_mapped_capacity(gpu, tokens)
        .map_err(|error| format!("map qwen4 target QSA context: {error}"))?;
    if let Some(mtp) = bundle.mtp.as_mut() {
        mtp.ensure_mapped_capacity(gpu, tokens)
            .map_err(|error| format!("map qwen4 MTP QSA context: {error}"))?;
    }
    Ok(())
}

fn cleanup_profile_resources(
    gpu: &mut Gpu,
    bundle: crate::bundle::Qwen4Bundle,
    pending: Option<GpuTensor>,
) -> Option<String> {
    let mut errors = Vec::new();
    if let Some(pending) = pending {
        if let Err(error) = gpu.free_tensor(pending) {
            errors.push(format!("free profile pending hidden: {error}"));
        }
    }
    if let Err(error) = bundle.free_gpu(gpu) {
        errors.push(format!("free profile bundle: {error}"));
    }
    if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    }
}

/// JSON report for one bounded production target token and one native MTP step.
pub struct ProfileReport(Value);

impl ProfileReport {
    pub fn into_json(self) -> Value {
        self.0
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&self.0).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

/// Same policy as the product loader (`hipfire-loader` carriers.rs):
/// UMA and external-residency entries stream bounded ranges; discrete
/// non-external entries (including `HostMapped`, which the store rejects as a
/// range) are borrowed from the resident source bytes.
fn qwen4_use_range_payload(is_uma: bool, residency: WeightResidency) -> bool {
    is_uma || residency.is_external()
}

fn qwen4_oracle_payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    recipe: &crate::weights::Qwen4TierRecipe,
    entry: &WeightEntry,
    use_ranges: bool,
) -> Result<hipfire_runtime::model_source::SourcePayload<'a>, String> {
    use hipfire_runtime::model_source::SourcePayload;
    // An expert-slab layer arena is one span of the container, not a tensor.
    if let Some(payload) = crate::expert_arena::recipe_arena_payload(source, recipe, entry) {
        return payload;
    }
    if qwen4_use_range_payload(use_ranges, entry.residency) {
        return source
            .tensor_range(&entry.name)
            .map_err(|error| error.to_string())?
            .map(SourcePayload::Range)
            .ok_or_else(|| {
                if entry.residency.is_external() {
                    format!("missing external tensor '{}'", entry.name)
                } else {
                    format!("missing resident tensor '{}'", entry.name)
                }
            });
    }
    let (info, bytes) = source
        .tensor_data(&entry.name)
        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
    Ok(SourcePayload::Borrowed { info, bytes })
}

/// Load the admitted production artifact, execute exactly one target token and
/// one native MTP token, and emit host/HIP/rocprof attribution context.
pub fn run_profile(model_path: &Path, corpus_path: &Path) -> Result<ProfileReport, String> {
    let profile_enabled = true;
    qwen4_profile_enable(false);
    let result = run_profile_inner(model_path, corpus_path, profile_enabled);
    qwen4_profile_enable(false);
    result
}

fn run_profile_inner(
    model_path: &Path,
    corpus_path: &Path,
    profile_enabled: bool,
) -> Result<ProfileReport, String> {
    let (tokens, corpus) = read_state_tokens(corpus_path)?;
    let input_token = *tokens
        .first()
        .ok_or_else(|| "profile corpus has no input token".to_string())?;

    launch_counters::reset();
    let load_started = Instant::now();
    let open_started = Instant::now();
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let open_ns = profile_duration_ns(open_started);
    emit_profile_load_checkpoint("open_hfq", open_ns);

    let gpu_init_started = Instant::now();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let gpu_init_ns = profile_duration_ns(gpu_init_started);
    emit_profile_load_checkpoint("gpu_init", gpu_init_ns);
    let admission_started = Instant::now();
    let receipt = {
        attach_state_validation_rows(
            &mut hfq,
            model_path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
        )?;
        crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
    }
    .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let admission_ns = profile_duration_ns(admission_started);
    emit_profile_load_checkpoint("admission", admission_ns);
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = crate::artifact::source_forward_scratch_mode(&gpu, &hfq, &config, &manifest)
        .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }

    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let use_ranges = gpu.is_uma();
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let manifest_started = Instant::now();
    let source_progress = RefCell::new((0_u64, None::<Instant>, None::<Instant>));
    let weights = oracle_expert_placement(&gpu, &manifest.weights)?;
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            let callback_started = Instant::now();
            let (index, elapsed_ns, cumulative_ns) = {
                let mut progress = source_progress.borrow_mut();
                let first_started = progress.1.as_ref().copied();
                let previous_started = progress.2.as_ref().copied();
                let index = progress.0;
                progress.0 = progress.0.saturating_add(1);
                progress.1.get_or_insert(callback_started);
                progress.2 = Some(callback_started);
                (
                    index,
                    previous_started
                        .map(|started| profile_elapsed_between_ns(started, callback_started))
                        .unwrap_or(0),
                    first_started
                        .map(|started| profile_elapsed_between_ns(started, callback_started))
                        .unwrap_or(0),
                )
            };
            emit_profile_source_callback(index, entry, elapsed_ns, cumulative_ns);

            qwen4_oracle_payload(&source, &manifest.identity.recipe, entry, use_ranges)
        },
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let manifest_ns = profile_duration_ns(manifest_started);
    emit_profile_load_checkpoint("manifest_fulfillment", manifest_ns);

    let assemble_started = Instant::now();
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_ple_rows(
        config.clone(),
        &manifest.identity.recipe,
        transaction,
        &placements,
        &mut gpu,
        1,
        metadata,
        receipt.ple_rows,
        crate::state::Qwen4StateFormat::F32,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let assemble_ns = profile_duration_ns(assemble_started);
    emit_profile_load_checkpoint("bundle_assembly", assemble_ns);

    let attach_forward_started = Instant::now();
    let setup_result = (|| -> Result<(), String> {
        bundle
            .attach_forward(&mut gpu, 1, scratch_mode)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        Ok(())
    })();
    let attach_forward_ns = profile_duration_ns(attach_forward_started);
    emit_profile_load_checkpoint("attach_forward", attach_forward_ns);
    if let Err(error) = setup_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }

    let attach_mtp_started = Instant::now();
    let attach_mtp_result = bundle
        .attach_mtp(&mut gpu, 1)
        .map_err(|error| format!("qwen4 MTP setup failed: {error}"));
    let attach_mtp_ns = profile_duration_ns(attach_mtp_started);
    emit_profile_load_checkpoint("attach_mtp", attach_mtp_ns);
    if let Err(error) = attach_mtp_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }

    let setup_started = Instant::now();
    let setup_result = map_bundle_context(&mut gpu, &mut bundle, 1).and_then(|()| {
        bundle
            .ensure_spec_hidden(&mut gpu, 1)
            .map_err(|error| format!("qwen4 profile hidden setup failed: {error}"))
    });
    if let Err(error) = setup_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }
    let hidden_width = match config.hc_count.checked_mul(config.hidden_size) {
        Some(width) => width,
        None => {
            let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
            let error = "qwen4 profile hidden width overflow".to_string();
            return Err(match cleanup {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    let pending = match gpu.zeros(&[hidden_width], DType::F32) {
        Ok(pending) => pending,
        Err(error) => {
            let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
            let error = format!("allocate qwen4 profile pending hidden: {error}");
            return Err(match cleanup {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    if let Err(error) = bundle
        .reset(&mut gpu)
        .map_err(|error| format!("reset qwen4 profile state: {error}"))
    {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, Some(pending));
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }
    let setup_ns = profile_duration_ns(setup_started);
    emit_profile_load_checkpoint("state_reset_and_scratch", setup_ns);
    let load_ns = profile_duration_ns(load_started);
    let load_hip = hip_counter_snapshot();
    emit_profile_checkpoint("load", load_ns, &load_hip, None, None);
    let layer_families = target_layer_families_json(&config);

    qwen4_profile_enable(profile_enabled);
    let execution_result = (|| -> Result<Value, String> {
        qwen4_profile_reset();
        launch_counters::reset();
        let ple_before = bundle
            .ple_rows()
            .ok_or("qwen4 profile: the bundle has no PLE rows")?
            .cache_stats();
        let target_started = Instant::now();
        dirty_profile_target_moe_reuse(&mut gpu, &bundle)?;
        let target_token = bundle
            .spec_capture_token(&mut gpu, input_token)
            .map_err(|error| format!("qwen4 profile target token: {error}"))?;
        let target_ns = profile_duration_ns(target_started);
        let target_hip = hip_counter_snapshot();
        let target_internal = qwen4_profile_stats_json(qwen4_profile_snapshot());
        let target_sealed = sealed_moe_profile_evidence(&target_internal, true)?;
        let target_ple_after = bundle
            .ple_rows()
            .ok_or("qwen4 profile: the bundle has no PLE rows")?
            .cache_stats();
        let target_d2h = target_hip
            .get("memcpy_dtoh")
            .cloned()
            .unwrap_or(Value::Null);
        let target_ple = json!({
            "before": ple_cache_stats_json(ple_before),
            "after": ple_cache_stats_json(target_ple_after),
            "delta": ple_cache_delta(ple_before, target_ple_after),
        });
        emit_profile_checkpoint(
            "target",
            target_ns,
            &target_hip,
            Some(&target_internal),
            Some(&target_ple),
        );

        qwen4_profile_reset();
        launch_counters::reset();
        let mtp_position = bundle
            .mtp_position()
            .map_err(|error| format!("read qwen4 profile MTP position: {error}"))?;
        let mtp_started = Instant::now();
        bundle
            .copy_spec_hidden_row_to(&mut gpu, 0, &pending)
            .map_err(|error| format!("qwen4 profile pending hidden copy: {error}"))?;
        dirty_profile_mtp_moe_reuse(&mut gpu, &bundle)?;
        let mtp_token = bundle
            .mtp_forward_token(&mut gpu, input_token, Some(&pending), mtp_position, true)
            .map_err(|error| format!("qwen4 profile native MTP token: {error}"))?;
        let mtp_ns = profile_duration_ns(mtp_started);
        let mtp_hip = hip_counter_snapshot();
        let mtp_internal = qwen4_profile_stats_json(qwen4_profile_snapshot());
        let mtp_sealed = sealed_moe_profile_evidence(&mtp_internal, true)?;
        let mtp_position_after = bundle
            .mtp_position()
            .map_err(|error| format!("read qwen4 profile MTP end position: {error}"))?;
        let mtp_d2h = mtp_hip.get("memcpy_dtoh").cloned().unwrap_or(Value::Null);
        emit_profile_checkpoint("mtp", mtp_ns, &mtp_hip, Some(&mtp_internal), None);

        Ok(json!({
            "target": {
                "input_token": input_token,
                "output_token": target_token,
                "state_position_after": bundle.state.position,
                "wall_ns": target_ns,
                "hip": target_hip,
                "internal": target_internal,
                "ple": target_ple,
            },
            "mtp": {
                "input_token": input_token,
                "output_token": mtp_token,
                "position": mtp_position,
                "position_after": mtp_position_after,
                "wall_ns": mtp_ns,
                "hip": mtp_hip,
                "internal": mtp_internal,
            },
            "d2h": {
                "target": target_d2h,
                "mtp": mtp_d2h,
                "attribution": "memcpy_dtoh counters are nested in the target/MTP phase that issued them",
            },
            "sealed_moe_validation": {
                "target": target_sealed,
                "mtp": mtp_sealed,
            },
        }))
    })();

    qwen4_profile_enable(false);
    let teardown_started = Instant::now();
    launch_counters::reset();
    let pending_cleanup = gpu
        .free_tensor(pending)
        .err()
        .map(|error| format!("free qwen4 profile pending hidden: {error}"));
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("free qwen4 profile bundle: {error}"));
    let teardown_ns = profile_duration_ns(teardown_started);
    let teardown_hip = hip_counter_snapshot();
    let teardown_error = pending_cleanup.or(bundle_cleanup);

    let execution = match execution_result {
        Ok(execution) => execution,
        Err(error) => {
            return Err(match teardown_error {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    if let Some(error) = teardown_error {
        return Err(error);
    }
    Ok(ProfileReport(json!({
        "schema": "hipfire.qwen4.profile.v1",
        "gpu_arch": gpu.arch,
        "model": model_path.display().to_string(),
        "corpus": corpus,
        "input_token": input_token,
        "hipfire_profile": {
            "enabled": profile_enabled,
            "environment": "profile mode enables seal/PLE counters locally",
            "scope": "Qwen4 seal/PLE host hooks; rocprofv3 is authoritative for kernel timing",
        },
        "load": {
            "wall_ns": load_ns,
            "hip": load_hip,
            "steps": {
                "open_hfq_ns": open_ns,
                "admission_ns": admission_ns,
                "gpu_init_ns": gpu_init_ns,
                "manifest_fulfillment_ns": manifest_ns,
                "bundle_assembly_ns": assemble_ns,
                "attach_forward_ns": attach_forward_ns,
                "attach_mtp_ns": attach_mtp_ns,
                "state_reset_and_scratch_ns": setup_ns,
            },
        },
        "bounded_work": {
            "target_tokens": 1,
            "native_mtp_steps": 1,
            "compact_state_parity": false,
            "quality_probe": false,
        },
        "target_layer_families": layer_families,
        "execution": execution,
        "teardown": {
            "wall_ns": teardown_ns,
            "hip": teardown_hip,
        },
        "status": "pass",
    })))
}

/// Load one real HFQ model, run the production AR/native-MTP path, and append
/// compact state-arena scenarios to the resulting JSON report.
pub fn run_state_parity(
    model_path: &Path,
    corpus_path: &Path,
) -> Result<StateParityReport, String> {
    let (tokens, corpus) = read_state_tokens(corpus_path)?;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let receipt = {
        attach_state_validation_rows(
            &mut hfq,
            model_path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
        )?;
        crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
    }
    .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = crate::artifact::source_forward_scratch_mode(&gpu, &hfq, &config, &manifest)
        .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let use_ranges = gpu.is_uma();
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let weights = oracle_expert_placement(&gpu, &manifest.weights)?;
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_oracle_payload(&source, &manifest.identity.recipe, entry, use_ranges),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_ple_rows(
        config.clone(),
        &manifest.identity.recipe,
        transaction,
        &placements,
        &mut gpu,
        2048,
        metadata,
        receipt.ple_rows,
        crate::state::Qwen4StateFormat::F32,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let real = (|| {
        bundle
            .attach_forward(&mut gpu, 2048, scratch_mode)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, 2048)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        map_bundle_context(&mut gpu, &mut bundle, 2048)?;
        real_model_probe(&mut gpu, &mut bundle, &tokens, &corpus)
    })();
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let real = real?;
    if let Some(error) = bundle_cleanup {
        return Err(error);
    }
    let compact = run_compact(&mut gpu)?;
    let status = if compact.get("status") == Some(&Value::String("pass".into()))
        && real.get("status") == Some(&Value::String("pass".into()))
    {
        "pass"
    } else {
        "fail"
    };
    Ok(StateParityReport(json!({
        "schema": "hipfire.qwen4.state_parity.v1",
        "gpu_arch": gpu.arch,
        "model": model_path,
        "corpus": corpus,
        "real_model": real,
        "compact_state": compact,
        "status": status,
    })))
}

fn real_model_probe(
    gpu: &mut Gpu,
    bundle: &mut crate::bundle::Qwen4Bundle,
    tokens: &[u32],
    corpus: &Value,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab_size;
    let ar_elements = tokens
        .len()
        .checked_mul(vocab)
        .ok_or_else(|| "real-model AR logits shape overflow".to_string())?;
    let ar_buffer = gpu
        .zeros(&[ar_elements], DType::F32)
        .map_err(|error| format!("allocate real-model AR logits: {error}"))?;
    let ar_result = (|| {
        bundle.reset(gpu).map_err(|error| error.to_string())?;
        bundle
            .forward_chunk(gpu, tokens, &ar_buffer, None)
            .map_err(|error| format!("real AR forward: {error}"))?;
        let last_row_start = tokens
            .len()
            .checked_sub(1)
            .and_then(|row| row.checked_mul(vocab))
            .ok_or_else(|| "real-model AR logits row is empty".to_string())?;
        download_host_logits(gpu, &ar_buffer.sub_offset(last_row_start, vocab))
    })();
    let ar_logits = match ar_result {
        Ok(logits) => logits,
        Err(error) => {
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    if let Err(error) = bundle.reset(gpu) {
        let _ = gpu.free_tensor(ar_buffer);
        return Err(error.to_string());
    }
    let mtp_buffer = match gpu.zeros(&[vocab], DType::F32) {
        Ok(buffer) => buffer,
        Err(error) => {
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("allocate real-model MTP logits: {error}"));
        }
    };
    let mut drafter = crate::mtp_spec::Qwen4MtpDrafter::new(DRAFTS.len(), 2048, None);
    use hipfire_runtime::spec::MtpDrafter;
    let seed = match drafter
        .mtp_prefill(gpu, bundle, tokens, tokens, 0, false, &|| false)
        .and_then(|outcome| outcome.into_first_token())
    {
        Ok(seed) => seed,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("real native MTP prefill: {error}"));
        }
    };
    let eos = bundle.config.eos_token_id;
    let window = match drafter.mtp_step(
        gpu,
        bundle,
        tokens.len(),
        seed,
        &[],
        DRAFTS.len(),
        eos,
        None,
    ) {
        Ok(window) => window,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("real native MTP step: {error}"));
        }
    };
    let probe_token = window.committed.last().copied().unwrap_or(seed);
    let mtp_position = match bundle.mtp_position() {
        Ok(position) => position,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("read native MTP position: {error}"));
        }
    };
    if let Err(error) =
        bundle.mtp_forward_token_logits(gpu, probe_token, mtp_position, true, &mtp_buffer)
    {
        MtpDrafter::mtp_free(Box::new(drafter), gpu);
        let _ = gpu.free_tensor(mtp_buffer);
        let _ = gpu.free_tensor(ar_buffer);
        return Err(format!("real native MTP logits: {error}"));
    }
    let mtp_logits = match download_host_logits(gpu, &mtp_buffer) {
        Ok(logits) => logits,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    MtpDrafter::mtp_free(Box::new(drafter), gpu);
    let families = match bundle_family_json(gpu, bundle) {
        Ok(families) => families,
        Err(error) => {
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    let max_abs = ar_logits
        .values
        .iter()
        .zip(&mtp_logits.values)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let finite = ar_logits.values.iter().all(|value| value.is_finite())
        && mtp_logits.values.iter().all(|value| value.is_finite());
    let result = json!({
        "status": if finite {"pass"} else {"fail"},
        "corpus": corpus,
        "ar_logit_row": tokens.len() - 1,
        "ar_token_count": tokens.len(),
        "accepted_drafts": window.accepted,
        "drafts_generated": window.drafts_generated,
        "committed": window.committed,
        "seed": seed,
        "probe_token": probe_token,
        "ar_logits": host_logits_json(&ar_logits),
        "native_mtp_logits": host_logits_json(&mtp_logits),
        "ar_vs_native_mtp": {
            "max_abs": max_abs,
            "same_within_1e-5": max_abs <= 1.0e-5,
            "ar_digest": ar_logits.digest,
            "native_mtp_digest": mtp_logits.digest,
        },
        "state_families": families,
    });
    gpu.free_tensor(mtp_buffer)
        .map_err(|error| error.to_string())?;
    gpu.free_tensor(ar_buffer)
        .map_err(|error| error.to_string())?;
    Ok(result)
}

#[derive(Clone, Debug)]
struct HostLogits {
    values: Vec<f32>,
    digest: String,
    top1: usize,
}

fn download_host_logits(gpu: &Gpu, tensor: &GpuTensor) -> Result<HostLogits, String> {
    let values = gpu
        .download_f32(tensor)
        .map_err(|error| error.to_string())?;
    let mut family = Family::new();
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    family.bytes(bytes, values.len());
    let digest = family
        .finish()
        .get("digest")
        .and_then(Value::as_str)
        .ok_or("host logits digest missing")?
        .to_string();
    let top1 = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Ok(HostLogits {
        values,
        digest,
        top1,
    })
}
fn host_logits_json(logits: &HostLogits) -> Value {
    json!({"digest": logits.digest, "numel": logits.values.len(), "top1": logits.top1})
}

fn bundle_family_json(gpu: &Gpu, bundle: &crate::bundle::Qwen4Bundle) -> Result<Value, String> {
    let target = target_families(gpu, &bundle.config, &bundle.state)?;
    let mtp = bundle
        .mtp
        .as_ref()
        .map(|mtp| mtp_families(gpu, &bundle.config, mtp.parity_state()))
        .transpose()?;
    // The draft head's request state (observed-token history the batched
    // and per-row fills both feed).
    let draft = bundle.mtp.as_ref().map(|mtp| {
        let state = mtp.draft.request_state();
        json!({"full_steps": state.full_steps, "margin_bits": state.margin.to_bits()})
    });
    Ok(json!({"target": target, "mtp": mtp, "draft_request_state": draft}))
}

/// The hardware oracles' routed-expert placement: every expert resident
/// unless `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` asks for layers `0..N` in VRAM
/// and the rest host-mapped (a discrete card that cannot hold them all, and
/// the only way to reach the long-prefill staging route). `auto` is refused:
/// an oracle's arms must share one fixed placement.
fn oracle_expert_placement(gpu: &Gpu, weights: &[WeightEntry]) -> Result<Vec<WeightEntry>, String> {
    let mut weights = weights.to_vec();
    match crate::expert_residency::expert_vram_layers_from_env()? {
        None => {}
        Some(crate::expert_residency::ExpertVramLayers::Layers(layers)) => {
            if gpu.is_uma() {
                return Err("host-mapped experts are only for discrete GPUs".to_string());
            }
            let moved = crate::expert_residency::place_routed_experts(&mut weights, layers);
            eprintln!("routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped");
        }
        Some(crate::expert_residency::ExpertVramLayers::Auto) => {
            return Err("hardware oracles take a fixed expert layer count, not auto".to_string());
        }
    }
    Ok(weights)
}

fn read_state_tokens(path: &Path) -> Result<(Vec<u32>, Value), String> {
    const COUNT: usize = 17;
    const SHA256: &str = "e53de8c7b501eaaea637648feb6f569dd17cd564c2f669b2924ccdf1b7e52e2f";
    let metadata_path = if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    };
    let metadata_text = std::fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: Value = serde_json::from_str(&metadata_text)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    if metadata.get("count").and_then(Value::as_u64) != Some(COUNT as u64) {
        return Err("state parity requires the canonical 17-token corpus".to_string());
    }
    let relative = metadata
        .get("path")
        .and_then(Value::as_str)
        .ok_or("state corpus metadata has no path")?;
    let payload = metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(PathBuf::from(relative));
    let bytes =
        std::fs::read(&payload).map_err(|error| format!("read {}: {error}", payload.display()))?;
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(&bytes);
    if bytes.len() != COUNT * 4 || format!("{:x}", digest.finalize()) != SHA256 {
        return Err(
            "state corpus byte count or SHA256 does not match canonical corpus".to_string(),
        );
    }
    let tokens = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok((
        tokens,
        json!({
            "metadata_path": metadata_path,
            "payload_path": payload,
            "count": COUNT,
            "sha256": SHA256
        }),
    ))
}

/// Native-MTP prompt fill digests for each prompt length: the MTP head and
/// target state families after `mtp_prefill`, the final prompt row's
/// logits, the pending hidden row, the seed, and one following draft
/// window. Run once per route setting (e.g.
/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL`) and compare the two outputs; routes are
/// read once per process.
///
/// `warm_chunks == 0` fills each length cold: the first `length` tokens of
/// `prompt_text` (tokenized without a template) when given, else a
/// deterministic spread over the vocabulary. Otherwise each length is the
/// suffix of a session-cache hit: a cold fill of a warm prompt of
/// `P = warm_chunks * chunk_rows` tokens (plus one row) publishes the
/// snapshot at `P`, then a prompt sharing those `P` tokens and followed by
/// `length` fresh ones is filled from the restored snapshot
/// (`start_pos = P`) and digested.
pub fn run_mtp_fill_digest(
    model_path: &Path,
    lengths: &[usize],
    max_seq: usize,
    warm_chunks: usize,
    prompt_text: Option<&str>,
) -> Result<Value, String> {
    use hipfire_runtime::arch_model::ArchModel;
    use hipfire_runtime::session_cache::{SessionCache, SessionRoute};
    use hipfire_runtime::spec::MtpDrafter;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let prompt_ids = prompt_text.map(|text| tokenizer.encode(text));
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let receipt = {
        attach_state_validation_rows(
            &mut hfq,
            model_path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
        )?;
        crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
    }
    .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let domain = hipfire_runtime::serve_contract::CacheDomain::for_model(
        &hfq,
        &tokenizer,
        None,
        "qwen4",
        gpu.device_id,
    );
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = crate::artifact::source_forward_scratch_mode(&gpu, &hfq, &config, &manifest)
        .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let use_ranges = gpu.is_uma();
    let weights = oracle_expert_placement(&gpu, &manifest.weights)?;
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_oracle_payload(&source, &manifest.identity.recipe, entry, use_ranges),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let warm = warm_chunks > 0;
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_ple_rows(
        config.clone(),
        &manifest.identity.recipe,
        transaction,
        &placements,
        &mut gpu,
        max_seq,
        metadata,
        receipt.ple_rows,
        crate::state::Qwen4StateFormat::F32,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let run = (|| -> Result<Value, String> {
        bundle
            .attach_forward(&mut gpu, max_seq, scratch_mode)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, max_seq)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        if warm {
            bundle.attach_session_cache(SessionCache::new(domain.clone(), u64::MAX >> 1));
        }
        map_bundle_context(&mut gpu, &mut bundle, max_seq)?;
        let vocab = config.vocab_size as u64;
        // A deterministic spread over the real vocabulary.
        let tokens = |salt: u64, count: usize| -> Vec<u32> {
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ salt;
            (0..count)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (10 + (state >> 33) % (vocab - 10)) as u32
                })
                .collect()
        };
        let prefix_len = if warm {
            let chunk = bundle
                .spec_chunk_rows()
                .ok_or("qwen4 forward resources are not attached")?;
            warm_chunks
                .checked_mul(chunk)
                .filter(|&len| len < max_seq)
                .ok_or("warm prefix does not fit MAX_SEQ")?
        } else {
            0
        };
        let mut drafter = crate::mtp_spec::Qwen4MtpDrafter::new(3, max_seq, None);
        let mut rows = Vec::new();
        for &length in lengths {
            let seed = if warm {
                // Publish the checkpoint at `prefix_len` (its prompt ends one
                // row past the boundary), then hit it with a fresh suffix.
                let shared = tokens(0x5eed ^ prefix_len as u64, prefix_len + 1);
                drafter
                    .mtp_prefill(&mut gpu, &mut bundle, &shared, &shared, 0, false, &|| false)?;
                bundle.session_commit();
                let mut prompt = shared[..prefix_len].to_vec();
                prompt.extend(tokens(0xa11 ^ length as u64, length));
                let start_pos = bundle.session_plan(&prompt, SessionRoute::Mtp);
                if start_pos != prefix_len {
                    return Err(format!(
                        "warm length {length}: session plan starts at {start_pos}, expected {prefix_len}"
                    ));
                }
                drafter
                    .mtp_prefill(
                        &mut gpu,
                        &mut bundle,
                        &prompt,
                        &prompt[prefix_len..],
                        prefix_len,
                        true,
                        &|| false,
                    )?
                    .into_first_token()?
            } else {
                let prompt = match prompt_ids.as_deref() {
                    Some(ids) => ids
                        .get(..length)
                        .ok_or_else(|| {
                            format!("prompt text has {} tokens, fewer than {length}", ids.len())
                        })?
                        .to_vec(),
                    None => tokens(length as u64, length),
                };
                drafter
                    .mtp_prefill(&mut gpu, &mut bundle, &prompt, &prompt, 0, false, &|| false)?
                    .into_first_token()?
            };
            let after_fill = bundle_family_json(&gpu, &bundle)?;
            // The prompt's last row: the final chunk's LM head writes it.
            let mut final_logits = Family::new();
            append_f32(
                &gpu,
                &mut final_logits,
                bundle
                    .spec_logits
                    .as_ref()
                    .ok_or("qwen4 spec logits are not attached")?,
                bundle.config.vocab_size,
            )?;
            let mut pending = Family::new();
            append_f32(
                &gpu,
                &mut pending,
                drafter.pending_hidden_for_parity()?,
                drafter.pending_hidden_for_parity()?.numel(),
            )?;
            let position = bundle.state.position;
            let eos = bundle.config.eos_token_id;
            let window =
                drafter.mtp_step(&mut gpu, &mut bundle, position, seed, &[seed], 3, eos, None)?;
            let after_window = bundle_family_json(&gpu, &bundle)?;
            rows.push(json!({
                "length": length,
                "seed": seed,
                "final_logits": final_logits.finish(),
                "pending_hidden": pending.finish(),
                "families_after_fill": after_fill,
                "window_committed": window.committed,
                "families_after_window": after_window,
            }));
            eprintln!("mtp-fill digest: length {length} start {prefix_len} seed {seed}");
        }
        Ok(json!({
            "batched_fill": hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_BATCHED_FILL", true),
            "max_seq": max_seq,
            "warm_prefix": prefix_len,
            "rows": rows,
        }))
    })();
    let cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let value = run?;
    if let Some(error) = cleanup {
        return Err(error);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_payload_policy_matches_product_loader() {
        assert!(!qwen4_use_range_payload(false, WeightResidency::Resident));
        assert!(!qwen4_use_range_payload(false, WeightResidency::HostMapped));
        assert!(qwen4_use_range_payload(true, WeightResidency::Resident));
        assert!(qwen4_use_range_payload(true, WeightResidency::HostMapped));
        assert!(qwen4_use_range_payload(
            false,
            WeightResidency::external_rows(128, 1)
        ));
    }

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok().or_else(|| {
            eprintln!("skip: Qwen4 compact state parity requires a GPU");
            None
        })
    }

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        value
            .get(name)
            .unwrap_or_else(|| panic!("report is missing field {name:?}: {value}"))
    }

    fn assert_pass(value: &Value) {
        assert_eq!(
            field(value, "status").as_str(),
            Some("pass"),
            "failed report case: {value}"
        );
    }

    fn assert_family_matches(value: &Value, name: &str) {
        let families = field(value, name)
            .as_object()
            .unwrap_or_else(|| panic!("{name} is not an object: {}", field(value, name)));
        assert!(!families.is_empty(), "{name} has no state families");
        for (family, comparison) in families {
            assert_eq!(
                field(comparison, "match").as_bool(),
                Some(true),
                "{name}.{family} did not match: {comparison}"
            );
        }
    }

    #[test]
    fn compact_runner_reports_device_backed_state_parity_scenarios() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let report = run_compact(&mut gpu).expect("compact state parity runner");

        assert_eq!(
            field(&report, "schema").as_str(),
            Some("hipfire.qwen4.state_parity.compact.v1")
        );
        assert_eq!(field(&report, "gpu_arch").as_str(), Some(gpu.arch.as_str()));
        assert_pass(&report);

        let cases = field(&report, "acceptance_cases")
            .as_array()
            .expect("acceptance_cases array");
        assert_eq!(cases.len(), 3);
        let expected_cases = [
            ("zero", 0usize, json!([1003u32])),
            ("one", 1usize, json!([1001u32, 1003u32])),
            ("all", 2usize, json!([1001u32, 1002u32, 1003u32])),
        ];
        let mut case_names = BTreeSet::new();
        for case in cases {
            assert_pass(case);
            let name = field(case, "case").as_str().expect("acceptance case name");
            let (_, accepted, committed_tokens) = expected_cases
                .iter()
                .find(|(expected, _, _)| *expected == name)
                .unwrap_or_else(|| panic!("unexpected acceptance case {name:?}"));
            assert_eq!(
                field(case, "accepted_drafts").as_u64(),
                Some(*accepted as u64)
            );
            assert_eq!(field(case, "committed_tokens"), committed_tokens);
            assert_family_matches(case, "target_families");
            assert_family_matches(case, "mtp_families");
            case_names.insert(name.to_string());
        }
        assert_eq!(
            case_names,
            expected_cases
                .iter()
                .map(|(name, _, _)| (*name).to_string())
                .collect()
        );

        let scenarios = field(&report, "scenarios")
            .as_array()
            .expect("scenarios array");
        assert_eq!(scenarios.len(), 6);
        let mut scenario_names = BTreeSet::new();
        for scenario in scenarios {
            assert_pass(scenario);
            scenario_names.insert(
                field(scenario, "case")
                    .as_str()
                    .expect("scenario name")
                    .to_string(),
            );
        }
        assert_eq!(
            scenario_names,
            [
                "qsa_pooling_boundary",
                "stale_ticket",
                "forced_terminal_seed",
                "cancellation",
                "injected_replay_failure",
                "cache_suffix_refusal",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        );

        let boundary = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("qsa_pooling_boundary"))
            .expect("QSA pooling boundary scenario");
        assert_eq!(
            field(field(boundary, "target_metadata"), "full_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "raw_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "pooled_len").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "selected_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "direct").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "native").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "expected").as_u64(),
            Some(5)
        );
        assert_eq!(
            field(boundary, "reuse_selected_indices"),
            &json!({
                "direct": [0, 1, 2, 3, 4],
                "native": [0, 1, 2, 3, 4],
            })
        );

        let stale = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("stale_ticket"))
            .expect("stale ticket scenario");
        assert!(field(stale, "target_refusal").as_str().is_some());
        assert!(field(stale, "mtp_refusal").as_str().is_some());
        assert_eq!(field(stale, "state_unchanged").as_bool(), Some(true));

        let terminal = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("forced_terminal_seed"))
            .expect("forced terminal seed scenario");
        assert_eq!(field(terminal, "terminal").as_bool(), Some(true));
        assert_eq!(
            field(terminal, "forced_token").as_u64(),
            Some(compact_test_config().eos_token_id as u64)
        );
        assert_eq!(
            field(field(terminal, "target_metadata"), "position").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(terminal, "mtp_metadata"), "position").as_u64(),
            Some(1)
        );

        for name in ["cancellation", "injected_replay_failure"] {
            let rollback = scenarios
                .iter()
                .find(|scenario| field(scenario, "case").as_str() == Some(name))
                .unwrap_or_else(|| panic!("{name} scenario"));
            assert_family_matches(rollback, "target_rollback");
            assert_family_matches(rollback, "mtp_rollback");
        }
        let injected = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("injected_replay_failure"))
            .expect("injected replay failure scenario");
        assert_eq!(
            field(injected, "replay_error").as_str(),
            Some("injected replay error")
        );

        let cache = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("cache_suffix_refusal"))
            .expect("cache suffix refusal scenario");
        assert_eq!(field(cache, "cache_hit").as_bool(), Some(true));
        assert_eq!(field(cache, "prompt_len").as_u64(), Some(3));
        assert_eq!(field(cache, "fill_len").as_u64(), Some(3));
        assert_eq!(
            field(cache, "refusal").as_str(),
            Some("Qwen4 native MTP cache hit at 0 needs a non-empty suffix of a 3-token prompt")
        );
    }

    #[test]
    fn cache_suffix_refusal_reports_position_zero_cache_hit_refusal() {
        let report = cache_suffix_refusal();
        assert_pass(&report);
        assert_eq!(
            field(&report, "case").as_str(),
            Some("cache_suffix_refusal")
        );
        assert_eq!(field(&report, "cache_hit").as_bool(), Some(true));
        assert_eq!(field(&report, "prompt_len").as_u64(), Some(3));
        assert_eq!(field(&report, "fill_len").as_u64(), Some(3));
        assert_eq!(
            field(&report, "refusal").as_str(),
            Some("Qwen4 native MTP cache hit at 0 needs a non-empty suffix of a 3-token prompt")
        );
    }
}

/// Hardware takeover oracle used by `mtp_takeover_fill_hw`. Route knobs must
/// be set before starting the process. Qwen4 pairs token p with hidden p,
/// including the seed row; an accepted EOS stays pending, not head-filled.
pub fn run_mtp_takeover_fill_oracle(model_path: &Path, max_seq: usize) -> Result<Value, String> {
    use crate::bundle::Qwen4Bundle;
    use crate::mtp_spec::Qwen4MtpDrafter;
    use hipfire_runtime::ngram_mod::NgramModConfig;
    use hipfire_runtime::prompt_frame::JinjaChatFrame;
    use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
    use hipfire_runtime::tokenizer::Tokenizer;

    // Preserve the existing family/metadata report, and additionally hash
    // storage bytes: dequantized equality alone is not byte identity.
    fn snapshot(
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        drafter: &Qwen4MtpDrafter,
    ) -> Result<Value, String> {
        let mut families = bundle_family_json(gpu, bundle)?;
        let mut target = Family::new();
        let mut head = Family::new();
        let append = |family: &mut Family, tensor: &GpuTensor, elements: usize| {
            append_raw(gpu, family, tensor, elements, elements)
        };
        for layer in &bundle.state.gdn {
            append(&mut target, &layer.recurrent, layer.recurrent.numel())?;
            append(&mut target, &layer.conv, layer.conv.numel())?;
        }
        let full = bundle.config.num_key_value_heads * bundle.config.head_dim;
        let raw = bundle.config.indexer_kv_heads * bundle.config.indexer_head_dim;
        for qsa in &bundle.state.qsa {
            for (tensor, count) in [
                (&qsa.full_keys, qsa.full_len * full),
                (&qsa.full_values, qsa.full_len * full),
                (&qsa.raw_index_keys, qsa.raw_len * raw),
                (&qsa.pooled_keys, qsa.pooled_len * raw),
                (&qsa.partial_keys, qsa.partial_len * raw),
                (&qsa.partial_values, qsa.partial_len * full),
                (&qsa.selected_indices, qsa.selected_len),
            ] {
                append(&mut target, tensor, count)?;
            }
        }
        if let Some(ple_conv) = &bundle.state.ple_conv {
            append(&mut target, ple_conv, ple_conv.numel())?;
        }
        let hyper_feedback = bundle
            .state
            .hyper_feedback
            .as_ref()
            .expect("hyper_feedback is input-band state");
        append(&mut target, hyper_feedback, hyper_feedback.numel())?;
        let state = bundle
            .mtp
            .as_ref()
            .ok_or("missing MTP head")?
            .parity_state();
        let meta = state.parity_metadata();
        let buffers = state.parity_buffers();
        for (tensor, count) in [
            (buffers.full_keys, meta.full_len * full),
            (buffers.full_values, meta.full_len * full),
            (buffers.raw_index_keys, meta.raw_len * raw),
            (buffers.pooled_keys, meta.pooled_len * raw),
            (buffers.selected_indices, meta.selected_len),
            (buffers.selected_len_out, 1),
            (buffers.wide_hidden, buffers.wide_hidden.numel()),
        ] {
            append(&mut head, tensor, count)?;
        }
        families["raw_storage"] = json!({"target": target.finish(), "mtp": head.finish()});
        let pending = drafter.pending_hidden_for_parity()?;
        Ok(json!({
            "families": families,
            "pending_hidden": raw_prefix(gpu, pending, pending.numel(), pending.numel())?,
            "target_position": bundle.state.position,
            "mtp_position": bundle.mtp_position().map_err(|error| error.to_string())?,
        }))
    }

    fn head_rows(
        gpu: &Gpu,
        bundle: &Qwen4Bundle,
        start: usize,
        rows: usize,
    ) -> Result<Value, String> {
        let state = bundle
            .mtp
            .as_ref()
            .ok_or("missing MTP head")?
            .parity_state();
        let buffers = state.parity_buffers();
        let full = bundle.config.num_key_value_heads * bundle.config.head_dim;
        let raw = bundle.config.indexer_kv_heads * bundle.config.indexer_head_dim;
        let mut family = Family::new();
        for (tensor, width) in [
            (buffers.full_keys, full),
            (buffers.full_values, full),
            (buffers.raw_index_keys, raw),
        ] {
            let view = tensor.sub_offset(start * width, rows * width);
            append_raw(gpu, &mut family, &view, rows * width, rows * width)?;
        }
        Ok(family.finish())
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len(), "hidden dimensions");
        let numerator: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        let denominator: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        (numerator / denominator.max(1e-30)).sqrt()
    }

    fn prefill(
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
    ) -> Result<u32, String> {
        // Arming again clears request-local history without reallocating the
        // pool. Cold prefill resets both device owners and pending hidden.
        drafter.configure_request(SpecRequestConfig {
            allow_ngram_modifier: true,
            ..SpecRequestConfig::default()
        });
        drafter
            .mtp_prefill(gpu, bundle, prompt, prompt, 0, false, &|| false)?
            .into_first_token()
    }

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let tokenizer =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|error| error.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    let rendered = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: None,
        user: "Write a Rust function fn parse_kv(line: &str) -> Option<(String, String)> that splits on the first '=', trims both sides, and rejects an empty key. Add three unit tests.",
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }.render()?;
    let prompt = tokenizer.encode(&rendered);
    const REF_STEPS: usize = 40;
    assert!(
        !prompt.is_empty() && prompt.len() + REF_STEPS + 8 < max_seq,
        "oracle context"
    );
    let mut gpu =
        Gpu::init().map_err(|error| format!("hardware oracle requires HIP GPU: {error}"))?;
    let receipt = {
        attach_state_validation_rows(
            &mut hfq,
            model_path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
        )?;
        crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
    }
    .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode = crate::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh =
        hipfire_runtime::device_mesh::DeviceMesh::single().map_err(|error| error.to_string())?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let use_ranges = gpu.is_uma();
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let weights = oracle_expert_placement(&gpu, &receipt.manifest.weights)?;
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            qwen4_oracle_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
                use_ranges,
            )
        },
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    // Match run_mtp_fill_digest: the reusable family walker uses logical
    // F32 row widths. This is a head-fill/rollback oracle, not a KV-format
    // admission test; retain the automatic VMM/legacy storage backend.
    let format = crate::state::Qwen4StateFormat::F32;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        format,
        backend,
        None,
    )
    .map_err(|error| error.to_string())?;
    // Longer than the entire oracle history: the armed pool is provably
    // empty/missing. Explicit takeovers supply the hit spans under test.
    let mut drafter = Qwen4MtpDrafter::new(3, max_seq, None).with_ngram(Some(NgramModConfig {
        capacity: 1024,
        n_match: prompt.len() + REF_STEPS + 8,
        n_min: 3,
        n_max: 3,
    }));
    let run = (|| -> Result<Value, String> {
        bundle
            .attach_forward(&mut gpu, max_seq, scratch_mode)
            .map_err(|error| error.to_string())?;
        bundle
            .attach_mtp(&mut gpu, max_seq)
            .map_err(|error| error.to_string())?;
        map_bundle_context(&mut gpu, &mut bundle, max_seq)?;
        let dim = bundle
            .config
            .hc_count
            .checked_mul(bundle.config.hidden_size)
            .ok_or("oracle hidden width overflow")?;
        let eos = bundle.config.eos_token_id;
        let interleaved = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL")
            .is_ok_and(|value| value == "1");

        // Fresh prefill, then target-only single-row greedy forwards. No
        // speculative head predictions enter the reference continuation.
        let seed0 = prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?;
        let mut reference = vec![seed0];
        let mut reference_hidden = Vec::with_capacity(REF_STEPS);
        for i in 0..REF_STEPS {
            assert_ne!(
                reference[i], eos,
                "reference stopped before oracle sequence"
            );
            let pick = bundle
                .spec_capture_token(&mut gpu, reference[i])
                .map_err(|error| error.to_string())?;
            let hidden = bundle
                .spec_hidden
                .as_ref()
                .ok_or("missing reference hidden")?;
            reference_hidden.push(
                gpu.download_f32(&hidden.sub_offset(0, dim))
                    .map_err(|error| error.to_string())?,
            );
            reference.push(pick);
        }
        let wrong = |token: u32| if token == 0 { 1 } else { token - 1 };
        assert_eq!(
            prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?,
            seed0
        );
        let mut index = 0usize;
        let mut windows = Vec::new();
        // The first EOS candidate is index 1, so two drafts are emitted but
        // only seed and candidate 0 are consumed.
        for (label, expected, stop) in [
            ("partial", 1, false),
            ("full", 3, false),
            ("zero", 0, false),
            ("accepted_eos_index1", 2, true),
            ("reject_index0", 0, false),
            ("reject_index1", 1, false),
            ("reject_index2", 2, false),
        ] {
            let position = prompt.len() + index;
            let mut candidates = reference[index + 1..index + 4].to_vec();
            if !stop && expected < 3 {
                candidates[expected] = wrong(candidates[expected]);
            }
            let window_eos = if stop { candidates[1] } else { eos };
            let consumed = expected + 1 - usize::from(stop);
            let untouched = head_rows(&gpu, &bundle, position + consumed, 4 - consumed)?;
            let window = drafter.mtp_takeover_step(
                &mut gpu,
                &mut bundle,
                position,
                reference[index],
                &reference[..=index],
                &candidates,
                window_eos,
            )?;
            let emitted = if stop { expected } else { expected + 1 };
            assert_eq!(window.accepted, expected, "{label}: accepted prefix");
            assert_eq!(window.drafts_generated, 3, "{label}: candidates offered");
            assert_eq!(
                window.committed,
                reference[index + 1..index + 1 + emitted],
                "{label}: greedy IDs"
            );
            assert_eq!(
                bundle.state.position,
                position + consumed,
                "{label}: target position"
            );
            assert_eq!(
                bundle.mtp_position().map_err(|error| error.to_string())?,
                position + consumed,
                "{label}: head position"
            );
            assert_eq!(
                head_rows(&gpu, &bundle, position + consumed, 4 - consumed)?,
                untouched,
                "{label}: unconsumed head rows must never be written",
            );
            // Interleaved capture overwrites row 0 each time; that physical
            // row is the logical verify row consumed-1.
            let capture_row = if interleaved { 0 } else { consumed - 1 };
            let captured = bundle
                .spec_hidden
                .as_ref()
                .ok_or("missing verify hidden")?
                .sub_offset(capture_row * dim, dim);
            let pending = drafter.pending_hidden_for_parity()?;
            assert_eq!(
                raw_prefix(&gpu, pending, dim, dim)?,
                raw_prefix(&gpu, &captured, dim, dim)?,
                "{label}: pending hidden must equal last consumed verify row",
            );
            let mut report = snapshot(&gpu, &bundle, &drafter)?;
            report["label"] = json!(label);
            report["accepted"] = json!(window.accepted);
            report["consumed"] = json!(consumed);
            report["committed"] = json!(window.committed);
            windows.push(report);
            index += consumed;
        }
        // No reset/demotion between the explicit hits and this real native
        // miss. The same request remains armed; history is too short to hit.
        assert!(
            drafter.request_stats().mtp_ngram,
            "modifier request must remain armed"
        );
        let before_stats = drafter.request_stats();
        let miss = drafter.mtp_step(
            &mut gpu,
            &mut bundle,
            prompt.len() + index,
            reference[index],
            &reference[..=index],
            3,
            eos,
            None,
        )?;
        assert_eq!(
            miss.committed.first(),
            Some(&reference[index + 1]),
            "native miss resumes at reference token"
        );
        if !interleaved {
            assert!(
                miss.drafts_generated >= 1,
                "native miss must draft, not retire to AR"
            );
        }
        let after_stats = drafter.request_stats();
        assert_eq!(
            after_stats.ngram_mod_windows, before_stats.ngram_mod_windows,
            "empty pool must miss"
        );
        assert!(
            after_stats.mtp_windows > before_stats.mtp_windows,
            "native window counter must advance"
        );

        // Two identical cold states, same first rejection, different rejected
        // tail. Compare live device families, raw storage and pending bytes.
        let mut starts = Vec::new();
        let mut tails = Vec::new();
        for variant in 0..2 {
            assert_eq!(
                prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?,
                seed0
            );
            starts.push(snapshot(&gpu, &bundle, &drafter)?);
            let untouched = head_rows(&gpu, &bundle, prompt.len() + 1, 3)?;
            let mut candidates = reference[1..4].to_vec();
            candidates[0] = wrong(candidates[0]);
            if variant == 1 {
                candidates[1] = wrong(candidates[1]);
                candidates[2] = wrong(candidates[2]);
            }
            let window = drafter.mtp_takeover_step(
                &mut gpu,
                &mut bundle,
                prompt.len(),
                seed0,
                &[seed0],
                &candidates,
                eos,
            )?;
            assert_eq!(window.accepted, 0, "tail fixture first rejection");
            assert_eq!(
                window.committed,
                reference[1..2],
                "tail fixture committed IDs"
            );
            assert_eq!(
                head_rows(&gpu, &bundle, prompt.len() + 1, 3)?,
                untouched,
                "tail fixture: rejected head rows must remain untouched"
            );
            tails.push(snapshot(&gpu, &bundle, &drafter)?);
        }
        assert_eq!(
            starts[0], starts[1],
            "tail fixture must start byte-identically"
        );
        assert_eq!(
            tails[0], tails[1],
            "rejected tail must not enter committed target/head/pending state"
        );

        // Compare token-p/hidden-p teacher forcing against the existing route.
        assert_eq!(
            prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?,
            seed0
        );
        let pairing_start = snapshot(&gpu, &bundle, &drafter)?;
        let full = drafter.mtp_takeover_step(
            &mut gpu,
            &mut bundle,
            prompt.len(),
            seed0,
            &[seed0],
            &reference[1..4],
            eos,
        )?;
        assert_eq!(full.accepted, 3, "pairing takeover must fully accept");
        let takeover = gpu
            .download_f32(drafter.pending_hidden_for_parity()?)
            .map_err(|error| error.to_string())?;
        let takeover_state = snapshot(&gpu, &bundle, &drafter)?;
        assert_eq!(
            prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?,
            seed0
        );
        assert_eq!(
            snapshot(&gpu, &bundle, &drafter)?,
            pairing_start,
            "pairing starts must match"
        );
        assert!(
            drafter.mtp_forced_advance(
                &mut gpu,
                &mut bundle,
                &reference[..4],
                prompt.len(),
                &|| false,
            )?,
            "teacher-forced route must handle all rows"
        );
        let forced_state = snapshot(&gpu, &bundle, &drafter)?;
        assert_eq!(
            takeover_state["target_position"],
            forced_state["target_position"]
        );
        assert_eq!(takeover_state["mtp_position"], forced_state["mtp_position"]);
        let forced = gpu
            .download_f32(drafter.pending_hidden_for_parity()?)
            .map_err(|error| error.to_string())?;
        let correct = rel_l2(&takeover, &forced);
        // h_(P+2), rather than the required h_(P+3), is the off-by-one
        // Qwen3.5-style pairing. Reference uses the same one-row forwards.
        let shifted = rel_l2(&takeover, &reference_hidden[2]);
        assert!(
            correct.is_finite() && shifted.is_finite() && correct < 0.1 * shifted,
            "teacher-forced token-p/hidden-p pairing: rel-L2={correct:e}, shifted={shifted:e}"
        );
        Ok(json!({
            "schema": "hipfire.qwen4.mtp_takeover_fill.v1",
            "gpu_arch": gpu.arch,
            "max_seq": max_seq,
            "prompt_tokens": prompt.len(),
            "state_format": "f32",
            "kv_backend": backend.name(),
            "reference": reference,
            "windows": windows,
            "rejected_tail": {"same_state": true, "before": starts[0], "after": tails[0]},
            "pairing": {
                "rel_l2": correct, "shifted_rel_l2": shifted,
                "shifted_row": -1,
                "pending_byte_identical": takeover_state["pending_hidden"] == forced_state["pending_hidden"],
                "families_byte_identical": takeover_state["families"] == forced_state["families"],
                "takeover": takeover_state, "forced": forced_state,
            },
            "native_miss": {"drafts_generated": miss.drafts_generated, "committed": miss.committed},
        }))
    })();
    Box::new(drafter).mtp_free(&mut gpu);
    let cleanup = bundle.free_gpu(&mut gpu).map_err(|error| error.to_string());
    let value = run?;
    cleanup?;
    Ok(value)
}
