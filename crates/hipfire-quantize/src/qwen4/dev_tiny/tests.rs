// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only controls for the developer tiny producer: complete source-exact and
//! packed artifacts of both profile families, exact predicted-versus-actual
//! extents, runtime admission, and transactional refusals.

use super::super::{SourceKind, SourceShard};
use super::*;
use hipfire_arch_qwen4::admit_developer_inventory_headers;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use hipfire_runtime::hfq::HfqTensorInfo;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::{tempdir, TempDir};

const TINY_MOE: &str = include_str!("../../../tests/fixtures/qwen4-dev/tiny-moe.config.json");
const TINY_DENSE: &str =
    include_str!("../../../tests/fixtures/qwen4-dev/tiny-dense-dev.config.json");
const TOKENIZER: &str = include_str!("../../../tests/fixtures/qwen4-dev/tokenizer.json");
const TOKENIZER_CONFIG: &str =
    include_str!("../../../tests/fixtures/qwen4-dev/tokenizer_config.json");

const VISION_NAME: &str = "model.visual.blocks.0.attn.qkv.weight";

struct Spec {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

type Tensors = Vec<(Spec, Vec<u8>)>;

/// The S0 fixture config with its hidden width (and optionally its FFN
/// width) replaced: the S0 width-5120 profiles differ from the tiny ones in
/// the hidden size only.
fn config_for(base: &str, hidden: usize, ffn_width: Option<usize>) -> Value {
    let mut config: Value = serde_json::from_str(base).expect("fixture config");
    config["text_config"]["hidden_size"] = json!(hidden);
    if let Some(width) = ffn_width {
        let is_moe = config["hipfire_profile"]["ffn"]["kind"].as_str() == Some("moe");
        config["hipfire_profile"]["ffn"]["intermediate_size"] = json!(width);
        if is_moe {
            config["text_config"]["moe_intermediate_size"] = json!(width);
            config["text_config"]["shared_expert_intermediate_size"] = json!(width);
            config["hipfire_profile"]["ffn"]["shared_intermediate_size"] = json!(width);
        }
    }
    config
}

/// Every source tensor of the profile: the source-exact manifest's weights and
/// PLE I64 records plus one vision tensor the producer must leave out.
fn specs(config: &Value) -> Vec<Spec> {
    let parsed = Qwen4Config::from_developer_value(config).expect("fixture developer config");
    let manifest = Qwen4Manifest::build_with_recipe(&parsed, Qwen4TierRecipe::SyntheticSourceBf16)
        .expect("source-exact manifest");
    let mut specs = Vec::new();
    for entry in &manifest.weights {
        if matches!(entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        specs.push(Spec {
            name: entry.name.clone(),
            dtype: "BF16",
            shape: entry.logical_shape.clone(),
        });
    }
    for record in &manifest.metadata {
        specs.push(Spec {
            name: record.name.clone(),
            dtype: "I64",
            shape: record.shape.clone(),
        });
    }
    specs.push(Spec {
        name: VISION_NAME.to_string(),
        dtype: "BF16",
        shape: vec![4, 4],
    });
    specs
}

fn bf16_bytes(name: &str, elements: usize) -> Vec<u8> {
    let mut state = name.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    });
    let mut out = Vec::with_capacity(elements * 2);
    for _ in 0..elements {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        out.extend_from_slice(&half::bf16::from_f32(unit).to_bits().to_le_bytes());
    }
    out
}

fn bytes_for(spec: &Spec, config: &Value) -> Vec<u8> {
    if spec.dtype == "I64" {
        let key = if spec.name.ends_with(".layer_multipliers") {
            "multipliers"
        } else if spec.name.ends_with(".ngram_heads_vocab_sizes") {
            "head_vocab_sizes"
        } else {
            "head_offsets"
        };
        let mut out = Vec::new();
        for value in config["qwen4_ple"][key].as_array().expect("PLE array") {
            out.extend_from_slice(&value.as_i64().expect("i64 PLE value").to_le_bytes());
        }
        return out;
    }
    bf16_bytes(&spec.name, spec.shape.iter().product())
}

fn write_safetensors(path: &Path, tensors: &Tensors) {
    let mut header = Map::new();
    let mut offset = 0usize;
    for (spec, bytes) in tensors {
        header.insert(
            spec.name.clone(),
            json!({
                "dtype": spec.dtype,
                "shape": spec.shape,
                "data_offsets": [offset, offset + bytes.len()],
            }),
        );
        offset += bytes.len();
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header");
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + header_bytes.len() + offset);
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    for (_, bytes) in tensors {
        file.extend_from_slice(bytes);
    }
    fs::write(path, file).expect("write safetensors fixture");
}

fn write_fixture(config: &Value, mutate: impl FnOnce(&mut Tensors)) -> TempDir {
    let dir = tempdir().expect("fixture dir");
    let mut tensors: Tensors = specs(config)
        .into_iter()
        .map(|spec| {
            let bytes = bytes_for(&spec, config);
            (spec, bytes)
        })
        .collect();
    mutate(&mut tensors);
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(config).expect("config"),
    )
    .expect("config.json");
    fs::write(dir.path().join("tokenizer.json"), TOKENIZER).expect("tokenizer.json");
    fs::write(dir.path().join("tokenizer_config.json"), TOKENIZER_CONFIG)
        .expect("tokenizer_config.json");
    write_safetensors(&dir.path().join("model.safetensors"), &tensors);
    dir
}

fn plain_fixture(config: &Value) -> TempDir {
    write_fixture(config, |_| {})
}

struct Produced {
    _input: TempDir,
    _out: TempDir,
    output: PathBuf,
    receipt: DevTinyReceipt,
}

fn produce(config: &Value, encoding: Qwen4DevTinyEncoding) -> Produced {
    let input = plain_fixture(config);
    let out = tempdir().expect("output dir");
    let output = out.path().join("model.hfq");
    let receipt = write_dev_tiny_artifact(input.path(), &output, encoding)
        .unwrap_or_else(|error| panic!("{encoding:?} artifact: {error}"));
    assert_eq!(
        receipt.actual_output_bytes, receipt.summary.predicted_output_bytes,
        "predicted and actual output extents"
    );
    assert_eq!(
        fs::metadata(&output).expect("published file").len(),
        receipt.actual_output_bytes
    );
    Produced {
        _input: input,
        _out: out,
        output,
        receipt,
    }
}

fn assert_nothing_published(out: &TempDir) {
    let leftovers: Vec<_> = fs::read_dir(out.path())
        .expect("read output dir")
        .map(|entry| entry.expect("entry").path())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a refused transaction left {leftovers:?}"
    );
}

fn layer(role: &str) -> String {
    format!("model.language_model.layers.0.{role}")
}

fn qt(hfq: &HfqFile, name: &str) -> (u8, u32) {
    let info = hfq
        .find_tensor_info(name)
        .unwrap_or_else(|| panic!("artifact carries {name}"));
    (info.quant_type, info.group_size)
}

#[test]
fn source_exact_tiny_profiles_are_complete_bit_exact_and_admitted() {
    for base in [TINY_MOE, TINY_DENSE] {
        let config = config_for(base, 16, None);
        let produced = produce(&config, Qwen4DevTinyEncoding::SourceExact);
        let receipt = &produced.receipt;
        assert_eq!(receipt.recipe_id, "synthetic-source-bf16");
        assert_eq!(receipt.packed_entries, 0);
        let hfq = HfqFile::open(&produced.output).expect("reopen");
        let text: Vec<Spec> = specs(&config)
            .into_iter()
            .filter(|spec| spec.name != VISION_NAME)
            .collect();
        assert_eq!(hfq.tensors().len(), text.len(), "complete text inventory");
        assert_eq!(receipt.summary.entries, text.len());
        for spec in &text {
            let (info, data) = hfq
                .tensor_data(&spec.name)
                .unwrap_or_else(|| panic!("artifact carries {}", spec.name));
            let expected_tag = if spec.dtype == "I64" { 54 } else { 16 };
            assert_eq!(info.quant_type, expected_tag, "{}", spec.name);
            assert_eq!(
                data,
                bytes_for(spec, &config).as_slice(),
                "{} payload",
                spec.name
            );
        }
        assert!(
            hfq.tensor_data(VISION_NAME).is_none(),
            "vision tensors are never written"
        );
        let metadata: Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
        assert_eq!(metadata[RECIPE_METADATA_KEY]["id"], "synthetic-source-bf16");
        assert_eq!(metadata["qwen4_ple"]["padded_rows"], 24);
    }
}

#[test]
fn packed_moe_uses_each_roles_flash_next_format_and_stays_admitted() {
    // hidden 256 tiles MQ4G256V2/Q8F16/MQ6G256V2; expert width 128 tiles
    // MQ4G128V2, so the routed pair packs together.
    let config = config_for(TINY_MOE, 256, Some(128));
    let produced = produce(&config, Qwen4DevTinyEncoding::Packed);
    assert_eq!(produced.receipt.recipe_id, "synthetic-packed");
    assert!(produced.receipt.packed_entries > 0);
    let hfq = HfqFile::open(&produced.output).expect("reopen");
    assert_eq!(qt(&hfq, "lm_head.weight"), (47, 256));
    assert_eq!(
        qt(&hfq, "model.language_model.embed_tokens.weight"),
        (3, 32)
    );
    assert_eq!(qt(&hfq, &layer("mlp.experts.gate_up_proj")), (44, 256));
    assert_eq!(qt(&hfq, &layer("mlp.experts.down_proj")), (53, 128));
    assert_eq!(qt(&hfq, "mtp.layers.0.mlp.experts.gate_up_proj"), (44, 256));
    assert_eq!(qt(&hfq, "mtp.layers.0.mlp.experts.down_proj"), (53, 128));
    assert_eq!(qt(&hfq, &layer("linear_attn.in_proj_qkv.weight")), (3, 32));
    // Router, shared expert, hyper-connection mixers, PLE projections and the
    // MTP connectors keep their source bytes, as in the Flash-Next recipe.
    for name in [
        layer("mlp.gate.weight"),
        layer("mlp.shared_expert.gate_proj.weight"),
        layer("attn_hyper_connection.hc_norm.weight"),
        "mtp.fc_embedding.weight".to_string(),
        layer("ple.key_proj.weight"),
    ] {
        assert_eq!(qt(&hfq, &name), (16, 0), "{name}");
    }
    let metadata: Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    assert_eq!(metadata[RECIPE_METADATA_KEY]["id"], "synthetic-packed");

    // Q8F16 is [f16 scale][32 x i8] per 32 weights: a decoded value is within
    // half a scale step of its BF16 source.
    let name = "model.language_model.embed_tokens.weight";
    let (_, packed) = hfq.tensor_data(name).expect("embedding");
    let spec = specs(&config)
        .into_iter()
        .find(|spec| spec.name == name)
        .expect("embedding spec");
    let source = bytes_for(&spec, &config);
    assert_eq!(packed.len(), (source.len() / 2 / 32) * 34);
    for (block, chunk) in packed.chunks_exact(34).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        for lane in 0..32 {
            let got = scale * f32::from(chunk[2 + lane] as i8);
            let index = block * 32 + lane;
            let want = bf16_to_f32(u16::from_le_bytes([
                source[2 * index],
                source[2 * index + 1],
            ]));
            assert!(
                (got - want).abs() <= scale.abs() * 0.6 + 1e-6,
                "block {block} lane {lane}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn packed_dense_packs_gate_up_and_keeps_a_narrow_down_bf16() {
    let config = config_for(TINY_DENSE, 256, None);
    let produced = produce(&config, Qwen4DevTinyEncoding::Packed);
    let hfq = HfqFile::open(&produced.output).expect("reopen");
    assert_eq!(qt(&hfq, &layer("mlp.gate_proj.weight")), (3, 32));
    assert_eq!(qt(&hfq, &layer("mlp.up_proj.weight")), (3, 32));
    // Down reduces over the width-8 intermediate: no Q8F16 block tiles it.
    assert_eq!(qt(&hfq, &layer("mlp.down_proj.weight")), (16, 0));
    assert_eq!(qt(&hfq, "lm_head.weight"), (47, 256));
    assert!(produced.receipt.packed_entries > 0);
    assert!(produced.receipt.bf16_entries > 0);
}

#[test]
fn packed_refuses_a_profile_where_no_role_tiles_a_format() {
    for base in [TINY_MOE, TINY_DENSE] {
        let config = config_for(base, 16, None);
        let input = plain_fixture(&config);
        let out = tempdir().expect("output dir");
        let output = out.path().join("model.hfq");
        let error = write_dev_tiny_artifact(input.path(), &output, Qwen4DevTinyEncoding::Packed)
            .expect_err("a packed artifact with zero packed roles must be refused")
            .to_string();
        assert!(error.contains("no role that can be packed"), "{error}");
        assert!(error.contains("token embedding"), "{error}");
        assert!(error.contains("language head"), "{error}");
        assert!(error.contains("source-exact for this profile"), "{error}");
        assert!(!output.exists());
        assert_nothing_published(&out);
        // The same profile still produces its source-exact artifact.
        let exact = tempdir().expect("output dir");
        write_dev_tiny_artifact(
            input.path(),
            &exact.path().join("model.hfq"),
            Qwen4DevTinyEncoding::SourceExact,
        )
        .expect("source-exact artifact of the same profile");
    }
}

fn tokenizer_metadata() -> TokenizerMetadata {
    TokenizerMetadata {
        tokenizer: TOKENIZER.to_string(),
        tokenizer_config: serde_json::from_str(TOKENIZER_CONFIG).ok(),
        generation_config: None,
    }
}

/// Source headers without payloads (a width-5120 checkpoint is ~110 MB): the
/// planner reads only the three small I64 PLE records, which live in a real
/// file; every other tensor is a header.
fn header_only_sources(config: &Value) -> (TempDir, Vec<SourceTensor>) {
    let dir = tempdir().expect("header dir");
    let path = dir.path().join("headers.bin");
    let specs = specs(config);
    let mut file_bytes = Vec::new();
    let mut i64_ranges = BTreeMap::<String, (u64, u64)>::new();
    for spec in specs.iter().filter(|spec| spec.dtype == "I64") {
        let bytes = bytes_for(spec, config);
        i64_ranges.insert(
            spec.name.clone(),
            (file_bytes.len() as u64, bytes.len() as u64),
        );
        file_bytes.extend_from_slice(&bytes);
    }
    fs::write(&path, &file_bytes).expect("header file");
    let shard = Arc::new(SourceShard {
        path: path.clone(),
        kind: SourceKind::Local {
            file: Arc::new(File::open(&path).expect("open header file")),
        },
        file_len: file_bytes.len() as u64,
        local_identity: None,
    });
    let tensors = specs
        .iter()
        .map(|spec| {
            let (start, len) = match i64_ranges.get(&spec.name) {
                Some(&range) => range,
                None => (0, spec.shape.iter().product::<usize>() as u64 * 2),
            };
            SourceTensor {
                name: spec.name.clone(),
                dtype: spec.dtype.to_string(),
                shape: spec
                    .shape
                    .iter()
                    .map(|&dimension| dimension as u64)
                    .collect(),
                data_start: start,
                data_end: start + len,
                shard: Arc::clone(&shard),
            }
        })
        .collect();
    (dir, tensors)
}

#[test]
fn width_5120_profiles_plan_exact_extents_and_admit_header_only() {
    for base in [TINY_MOE, TINY_DENSE] {
        let config = config_for(base, 5120, None);
        for encoding in [
            Qwen4DevTinyEncoding::SourceExact,
            Qwen4DevTinyEncoding::Packed,
        ] {
            let (_keep, tensors) = header_only_sources(&config);
            let prepared = prepare_dev_artifact(
                &config,
                &tokenizer_metadata(),
                tensors,
                encoding,
                DEFAULT_ROW_CHUNK,
            )
            .unwrap_or_else(|error| panic!("{encoding:?} 5120 plan: {error}"));
            let headers: Vec<HfqTensorInfo> = prepared
                .stream_entries
                .iter()
                .map(|entry| HfqTensorInfo {
                    name: entry.name.clone(),
                    quant_type: entry.quant_type,
                    shape: entry.shape.clone(),
                    group_size: entry.group_size,
                    data_offset: 0,
                    data_size: entry.data_len as usize,
                })
                .collect();
            let receipt =
                admit_developer_inventory_headers(QWEN4_ARCH_ID, &prepared.metadata_json, &headers)
                    .expect("the planned extents are the admitted extents");
            assert_eq!(receipt.manifest, prepared.manifest);
            assert_eq!(receipt.source_tensor_count, prepared.plan.entries.len());
            let packed = prepared
                .plan
                .entries
                .iter()
                .filter(|entry| matches!(entry.kind, EntryKind::Quant(_)))
                .count();
            let entry = |suffix: &str| {
                prepared
                    .plan
                    .entries
                    .iter()
                    .find(|entry| entry.name.ends_with(suffix))
                    .unwrap_or_else(|| panic!("plan carries *{suffix}"))
            };
            match encoding {
                Qwen4DevTinyEncoding::SourceExact => assert_eq!(packed, 0),
                Qwen4DevTinyEncoding::Packed => {
                    assert!(
                        packed > 0,
                        "{} packed nothing at width 5120",
                        prepared.manifest.identity.profile_id
                    );
                    assert!(matches!(
                        entry("lm_head.weight").kind,
                        EntryKind::Quant(DType::MQ6G256V2)
                    ));
                    if base == TINY_DENSE {
                        assert!(matches!(
                            entry(".mlp.gate_proj.weight").kind,
                            EntryKind::Quant(DType::Q8_0)
                        ));
                        assert!(matches!(
                            entry(".mlp.down_proj.weight").kind,
                            EntryKind::Bf16
                        ));
                    }
                    // The recipe id is the artifact's identity: the same packed
                    // inventory is refused under the source-exact id.
                    let mut metadata: Value =
                        serde_json::from_str(&prepared.metadata_json).expect("metadata");
                    metadata[RECIPE_METADATA_KEY]["id"] = json!("synthetic-source-bf16");
                    admit_developer_inventory_headers(
                        QWEN4_ARCH_ID,
                        &metadata.to_string(),
                        &headers,
                    )
                    .expect_err("packed tags under the source-exact recipe");
                }
            }
            if base == TINY_MOE {
                // Expert width 8 tiles no MQ4 group: gate/up and down both stay
                // BF16 (the routed pair packs together or not at all).
                for entry in prepared
                    .plan
                    .entries
                    .iter()
                    .filter(|entry| entry.name.contains(".experts."))
                {
                    assert!(matches!(entry.kind, EntryKind::Bf16), "{}", entry.name);
                    // A stacked expert block is a few very wide rows: the raw
                    // chunk must be bounded by bytes, not by the row default.
                    let chunk = entry_row_chunk(entry, DEFAULT_ROW_CHUNK).expect("chunk");
                    let width: u64 = entry.source.shape[1..].iter().product::<u64>() * 2;
                    assert!(chunk as u64 * width <= MAX_CHUNK_BYTES, "{}", entry.name);
                }
            }
        }
    }
}

fn refuse(
    config: &Value,
    encoding: Qwen4DevTinyEncoding,
    mutate: impl FnOnce(&mut Tensors),
    needle: &str,
) {
    let input = write_fixture(config, mutate);
    let out = tempdir().expect("output dir");
    let output = out.path().join("model.hfq");
    let error = write_dev_tiny_artifact(input.path(), &output, encoding)
        .expect_err("the source must be refused")
        .to_string();
    assert!(error.contains(needle), "expected {needle:?} in {error}");
    assert!(!output.exists());
    assert_nothing_published(&out);
}

#[test]
fn source_refusals_publish_nothing_and_leave_no_temporary_file() {
    let config = config_for(TINY_MOE, 16, None);
    let exact = Qwen4DevTinyEncoding::SourceExact;
    refuse(
        &config,
        exact,
        |tensors| tensors.retain(|(spec, _)| !spec.name.ends_with("layers.0.mlp.gate.weight")),
        "source is missing",
    );
    refuse(
        &config,
        exact,
        |tensors| {
            tensors.push((
                Spec {
                    name: "model.language_model.rogue.weight".to_string(),
                    dtype: "BF16",
                    shape: vec![2],
                },
                vec![0u8; 4],
            ));
        },
        "is not declared by the tiny-moe profile manifest",
    );
    refuse(
        &config,
        exact,
        |tensors| {
            for (spec, _) in tensors.iter_mut() {
                if spec.name == "lm_head.weight" {
                    spec.shape = vec![16, 32];
                }
            }
        },
        "expected manifest shape",
    );
    refuse(
        &config,
        exact,
        |tensors| {
            for (spec, bytes) in tensors.iter_mut() {
                if spec.name.ends_with("layers.0.mlp.gate.weight") {
                    spec.dtype = "F32";
                    *bytes = vec![0u8; spec.shape.iter().product::<usize>() * 4];
                }
            }
        },
        "expected source BF16",
    );
}

#[test]
fn identity_and_destination_refusals() {
    let config = config_for(TINY_MOE, 16, None);
    let input = plain_fixture(&config);
    let out = tempdir().expect("output dir");
    let exact = Qwen4DevTinyEncoding::SourceExact;

    // An existing output is never replaced, and survives untouched.
    let existing = out.path().join("existing.hfq");
    fs::write(&existing, b"keep me").expect("existing artifact");
    let error = write_dev_tiny_artifact(input.path(), &existing, exact)
        .expect_err("replacement refused")
        .to_string();
    assert!(
        error.contains("refusing to replace existing Qwen4 artifact"),
        "{error}"
    );
    assert_eq!(fs::read(&existing).expect("existing"), b"keep me");

    // Only a local safetensors directory is accepted.
    let remote = Path::new("hf://owner/repo@0123456789abcdef0123456789abcdef01234567");
    let error = write_dev_tiny_artifact(remote, &out.path().join("a.hfq"), exact)
        .expect_err("remote input refused")
        .to_string();
    assert!(error.contains("local safetensors directory"), "{error}");
    let error = write_dev_tiny_artifact(
        &input.path().join("model.safetensors"),
        &out.path().join("b.hfq"),
        exact,
    )
    .expect_err("a single file is not a directory")
    .to_string();
    assert!(error.contains("local safetensors directory"), "{error}");

    // A config without the explicit developer marker is not a developer profile.
    let mut stripped = config.clone();
    stripped
        .as_object_mut()
        .expect("config object")
        .remove("hipfire_profile");
    fs::write(
        input.path().join("config.json"),
        serde_json::to_vec(&stripped).expect("config"),
    )
    .expect("config.json");
    let error = write_dev_tiny_artifact(input.path(), &out.path().join("c.hfq"), exact)
        .expect_err("markerless config refused")
        .to_string();
    assert!(error.contains("developer"), "{error}");

    let leftovers: Vec<_> = fs::read_dir(out.path())
        .expect("read output dir")
        .map(|entry| entry.expect("entry").file_name())
        .collect();
    assert_eq!(
        leftovers.len(),
        1,
        "only the pre-existing file remains: {leftovers:?}"
    );
}
