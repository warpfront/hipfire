// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Tiny synthetic stock-FP8 checkpoint: two layers x two routed experts, two PLE
//! shards, one MTP layer (FP8 experts + BF16 scales), one vision tensor.  The
//! compact fixture mode is the only non-production mode; the pin tests drive
//! the production entry point against fixtures that cannot be the pinned
//! checkpoint.

use super::*;
use std::collections::BTreeMap;

const LAYERS: usize = 2;
const EXPERTS: usize = 2;
/// Gate rows (a whole 128-row scale block) and hidden K.
const RG: u64 = 128;
const K: u64 = 128;
const PLE_ROWS: u64 = 3;
const PLE_TABLE: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
/// round_up(256*128 + 2*ceil(256/128)*ceil(128/128), 256) = round_up(32_772, 256)
const GATE_UP_STRIDE: u64 = 33_024;
/// round_up(128*128 + 2*1*1, 256) = round_up(16_386, 256)
const DOWN_STRIDE: u64 = 16_640;

struct Tensor {
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

type Bag = BTreeMap<String, Tensor>;

fn codes(seed: usize, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let byte = ((i * 37 + seed * 11 + 3) % 256) as u8;
            if byte & 0x7f == 0x7f {
                0x80
            } else {
                byte
            }
        })
        .collect()
}

fn scale_bytes(seed: usize, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (0x3c00u16 + (seed * 5 + i) as u16).to_le_bytes())
        .collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Builder {
    bag: Bag,
    seed: usize,
}

impl Builder {
    fn f8(&mut self, name: &str, shape: &[u64]) {
        self.seed += 1;
        let n = shape.iter().product::<u64>() as usize;
        self.bag.insert(
            name.to_string(),
            Tensor {
                dtype: "F8_E4M3",
                shape: shape.to_vec(),
                data: codes(self.seed, n),
            },
        );
    }

    fn scale(&mut self, name: &str, shape: &[u64]) {
        self.seed += 1;
        let n = shape.iter().product::<u64>() as usize;
        self.bag.insert(
            name.to_string(),
            Tensor {
                dtype: "BF16",
                shape: shape.to_vec(),
                data: scale_bytes(self.seed, n),
            },
        );
    }

    fn bf16(&mut self, name: &str, shape: &[u64]) {
        self.seed += 1;
        let n = shape.iter().product::<u64>() as usize * 2;
        self.bag.insert(
            name.to_string(),
            Tensor {
                dtype: "BF16",
                shape: shape.to_vec(),
                data: (0..n).map(|i| (i * 13 + self.seed) as u8).collect(),
            },
        );
    }

    fn i64s(&mut self, name: &str, values: &[i64]) {
        self.bag.insert(
            name.to_string(),
            Tensor {
                dtype: "I64",
                shape: vec![values.len() as u64],
                data: i64_bytes(values),
            },
        );
    }

    fn experts(&mut self, prefix: &str) {
        for expert in 0..EXPERTS {
            for (proj, rows, cols) in [
                ("gate_proj", RG, K),
                ("up_proj", RG, K),
                ("down_proj", K, RG),
            ] {
                let base = format!("{prefix}.mlp.experts.{expert}.{proj}");
                self.f8(&format!("{base}.weight"), &[rows, cols]);
                self.scale(
                    &format!("{base}.weight_scale_inv"),
                    &[rows.div_ceil(128), cols.div_ceil(128)],
                );
            }
        }
    }
}

fn fixture_bag(ple_scale_shape: &[u64]) -> Bag {
    let mut b = Builder {
        bag: Bag::new(),
        seed: 0,
    };
    for layer in 0..LAYERS {
        b.experts(&format!("model.language_model.layers.{layer}"));
    }
    b.experts("mtp.layers.0");
    for shard in 0..2 {
        b.f8(
            &format!("{PLE_TABLE}.shard_{shard}.weight"),
            &[PLE_ROWS, 160],
        );
    }
    b.scale(&format!("{PLE_TABLE}.weight_scale"), ple_scale_shape);
    b.bf16("model.language_model.embed_tokens.weight", &[4, 8]);
    b.bf16("model.language_model.layers.0.input_layernorm.weight", &[8]);
    b.bf16("mtp.fc_hidden.weight", &[2, 2]);
    b.bf16("model.visual.patch_embed.weight", &[2, 2]);
    let root = "model.language_model.layers.1.ple.ple_embedding";
    b.i64s(&format!("{root}.layer_multipliers"), &[3, 5, 7]);
    b.i64s(&format!("{root}.ngram_heads_vocab_sizes"), &[127; 16]);
    let offsets: Vec<i64> = (0..16).map(|i| i * 127).collect();
    b.i64s(&format!("{root}.ngram_heads_offsets"), &offsets);
    b.bag
}

fn write_safetensors(path: &Path, items: &[(&String, &Tensor)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, tensor) in items {
        let start = data.len() as u64;
        data.extend_from_slice(&tensor.data);
        header.insert(
            (*name).clone(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, data.len() as u64],
            }),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(path, file).unwrap();
}

struct Fixture {
    src: tempfile::TempDir,
    out: tempfile::TempDir,
    bag: Bag,
}

impl Fixture {
    fn new() -> Self {
        Self::with(fixture_bag(&[1]))
    }

    fn with(bag: Bag) -> Self {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let names: Vec<&String> = bag.keys().collect();
        let half = names.len() / 2;
        for (index, chunk) in [&names[..half], &names[half..]].into_iter().enumerate() {
            let items: Vec<(&String, &Tensor)> =
                chunk.iter().map(|name| (*name, &bag[*name])).collect();
            write_safetensors(
                &src.path()
                    .join(format!("model-0000{}-of-00002.safetensors", index + 1)),
                &items,
            );
        }
        fs::write(
            src.path().join("config.json"),
            r#"{"model_type":"qwen4_exp","mtp_num_hidden_layers":1}"#,
        )
        .unwrap();
        Self { src, out, bag }
    }

    fn poisoned(name: &str, byte: u8) -> Self {
        let mut bag = fixture_bag(&[1]);
        bag.get_mut(name).unwrap().data[5] = byte;
        Self::with(bag)
    }

    fn output(&self) -> PathBuf {
        self.out.path().join("stock.hfq")
    }

    fn compact(&self, pin: Option<&str>) -> Result<StockFp8Summary, Qwen4Error> {
        write_stock_fp8_artifact(&StockFp8Options {
            input: self.src.path(),
            output: &self.output(),
            mode: Qwen4Mode::CompactFixture,
            pin,
        })
    }

    fn out_is_empty(&self) -> bool {
        fs::read_dir(self.out.path()).unwrap().count() == 0
    }

    /// Write a correct `fp8.sha256` for the fixture and return its SHA-256.
    fn write_manifest(&self, skip: Option<&str>) -> String {
        let mut text = String::new();
        for entry in fs::read_dir(self.src.path()).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if name == STOCK_FP8_MANIFEST_NAME || Some(name.as_str()) == skip {
                continue;
            }
            text.push_str(&format!(
                "{}  {name}\n",
                hex_lower(Sha256::digest(fs::read(&path).unwrap()).as_slice())
            ));
        }
        fs::write(self.src.path().join(STOCK_FP8_MANIFEST_NAME), &text).unwrap();
        hex_lower(Sha256::digest(text.as_bytes()).as_slice())
    }

    fn data(&self, name: &str) -> &[u8] {
        &self.bag[name].data
    }
}

fn sha_hex(bytes: &[u8]) -> String {
    hex_lower(Sha256::digest(bytes).as_slice())
}

fn expected_gate_up(fixture: &Fixture, prefix: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for e in 0..EXPERTS {
        let at = |proj: &str, suffix: &str| {
            fixture
                .data(&format!("{prefix}.mlp.experts.{e}.{proj}.{suffix}"))
                .to_vec()
        };
        out.extend(at("gate_proj", "weight"));
        out.extend(at("up_proj", "weight"));
        out.extend(at("gate_proj", "weight_scale_inv"));
        out.extend(at("up_proj", "weight_scale_inv"));
        let used = 256 * 128 + 4;
        out.extend(vec![0u8; GATE_UP_STRIDE as usize - used]);
    }
    out
}

fn expected_down(fixture: &Fixture, prefix: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for e in 0..EXPERTS {
        let base = format!("{prefix}.mlp.experts.{e}.down_proj");
        out.extend(fixture.data(&format!("{base}.weight")));
        out.extend(fixture.data(&format!("{base}.weight_scale_inv")));
        let used = 128 * 128 + 2;
        out.extend(vec![0u8; DOWN_STRIDE as usize - used]);
    }
    out
}

fn count80(bytes: &[u8]) -> u64 {
    bytes.iter().filter(|&&b| b == 0x80).count() as u64
}

fn receipt_of(fixture: &Fixture) -> Value {
    let path = receipt_path_for(&fixture.output());
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn stride_constants_follow_the_contract() {
    // S(R,K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256)
    assert_eq!(b128_block_stride(256, 128), Some(GATE_UP_STRIDE as usize));
    assert_eq!(b128_block_stride(128, 128), Some(DOWN_STRIDE as usize));
    // Production geometry: gate_up R=1280 K=2560, down R=2560 K=640.
    assert_eq!(b128_block_stride(1280, 2560), Some(3_277_312));
    assert_eq!(b128_block_stride(2560, 640), Some(1_638_656));
}

#[test]
fn fixture_payload_receipt_and_metadata_are_exact() {
    let fixture = Fixture::new();
    let summary = fixture.compact(None).unwrap();
    assert_eq!(summary.entries, 11);
    assert_eq!(summary.expert_entries, 4);
    assert_eq!(summary.ple_shards, 2);
    assert_eq!(summary.fp8_scanned, 20);
    assert_eq!(summary.mtp_omitted, 13);
    assert_eq!(summary.vision_omitted, 1);

    let output = fixture.output();
    let bytes = fs::read(&output).unwrap();
    assert_eq!(summary.artifact_sha256, sha_hex(&bytes));
    let plan = reopen_stock_artifact(&output).unwrap();
    assert_eq!(plan.entries.len(), 11);
    assert!(plan
        .entries
        .iter()
        .all(|e| !e.name.starts_with("mtp.") && !e.name.starts_with("model.visual.")));

    let by_name: BTreeMap<&str, &Qwen4PlanEntry> =
        plan.entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let payload = |name: &str| -> Vec<u8> {
        let entry = by_name[name];
        bytes[entry.data_offset as usize..(entry.data_offset + entry.data_len) as usize].to_vec()
    };
    for layer in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        let gate_up = by_name[format!("{prefix}.mlp.experts.gate_up_proj").as_str()];
        assert_eq!(gate_up.quant_type, 55);
        assert_eq!(gate_up.group_size, 128);
        assert_eq!(gate_up.shape, vec![EXPERTS as u32, 256, 128]);
        assert_eq!(gate_up.data_len, EXPERTS as u64 * GATE_UP_STRIDE);
        assert_eq!(
            payload(&format!("{prefix}.mlp.experts.gate_up_proj")),
            expected_gate_up(&fixture, &prefix)
        );
        let down = by_name[format!("{prefix}.mlp.experts.down_proj").as_str()];
        assert_eq!((down.quant_type, down.group_size), (55, 128));
        assert_eq!(down.shape, vec![EXPERTS as u32, 128, 128]);
        assert_eq!(down.data_len, EXPERTS as u64 * DOWN_STRIDE);
        assert_eq!(
            payload(&format!("{prefix}.mlp.experts.down_proj")),
            expected_down(&fixture, &prefix)
        );
    }
    for shard in 0..2 {
        let name = format!("{PLE_TABLE}.shard_{shard}.weight");
        let entry = by_name[name.as_str()];
        assert_eq!((entry.quant_type, entry.group_size), (56, 0));
        assert_eq!(entry.shape, vec![PLE_ROWS as u32, 160]);
        let body = payload(&name);
        assert_eq!(body.len(), 16 + PLE_ROWS as usize * 160);
        assert_eq!(
            &body[..2],
            fixture.data(&format!("{PLE_TABLE}.weight_scale"))
        );
        assert!(body[2..16].iter().all(|&b| b == 0));
        assert_eq!(&body[16..], fixture.data(&name));
    }
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
    ] {
        assert_eq!(by_name[name].quant_type, 16);
        assert_eq!(by_name[name].group_size, 0);
        assert_eq!(payload(name), fixture.data(name));
    }
    let root = "model.language_model.layers.1.ple.ple_embedding";
    for suffix in [
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
    ] {
        let name = format!("{root}.{suffix}");
        assert_eq!(by_name[name.as_str()].quant_type, QWEN4_I64_QUANT_TYPE);
        assert_eq!(payload(&name), fixture.data(&name));
    }

    // Metadata: recipe id, verbatim config, PLE marker.
    let metadata: Value = serde_json::from_str(&plan.metadata_json).unwrap();
    assert_eq!(metadata["qwen4_recipe"]["id"], STOCK_FP8_RECIPE_ID);
    assert_eq!(metadata["qwen4_recipe"]["mtp"], "omitted-ar-only");
    assert_eq!(metadata["qwen4_recipe"]["ple"]["quant_type"], 56);
    assert_eq!(metadata["qwen4_recipe"]["gate_up"]["quant_type"], 55);
    assert_eq!(
        metadata["config"],
        json!({"model_type": "qwen4_exp", "mtp_num_hidden_layers": 1})
    );
    assert_eq!(metadata["qwen4_ple"]["multipliers"], json!([3, 5, 7]));

    // Receipt: top-level artifact hash, census, 0x80 counts, source ranges.
    let receipt = receipt_of(&fixture);
    assert_eq!(receipt["artifact_sha256"], sha_hex(&bytes));
    assert_eq!(receipt["artifact"]["bytes"], bytes.len());
    assert_eq!(receipt["recipe"], STOCK_FP8_RECIPE_ID);
    assert_eq!(receipt["manifest_verified"], false);
    let fp8 = &receipt["census"]["fp8"];
    assert_eq!(fp8["total"], 20);
    assert_eq!(fp8["emitted"], 14);
    assert_eq!(fp8["omitted"], 6);
    assert_eq!(fp8["nan_codes"], 0);
    let all_f8: Vec<&String> = fixture
        .bag
        .iter()
        .filter(|(_, t)| t.dtype == "F8_E4M3")
        .map(|(name, _)| name)
        .collect();
    let total80: u64 = all_f8.iter().map(|n| count80(fixture.data(n))).sum();
    let omitted80: u64 = all_f8
        .iter()
        .filter(|n| n.starts_with("mtp."))
        .map(|n| count80(fixture.data(n)))
        .sum();
    assert!(
        total80 > 0 && omitted80 > 0,
        "fixture must contain 0x80 codes"
    );
    assert_eq!(fp8["count_0x80"]["total"], total80);
    assert_eq!(fp8["count_0x80"]["omitted"], omitted80);
    assert_eq!(fp8["count_0x80"]["emitted"], total80 - omitted80);
    assert_eq!(receipt["census"]["omitted"]["mtp"]["tensors"], 13);
    assert_eq!(receipt["census"]["omitted"]["vision"]["tensors"], 1);
    assert_eq!(receipt["omitted_fp8"].as_array().unwrap().len(), 6);
    for record in receipt["omitted_fp8"].as_array().unwrap() {
        let name = record["name"].as_str().unwrap();
        assert!(name.starts_with("mtp."));
        assert_eq!(record["sha256"], sha_hex(fixture.data(name)));
        assert_eq!(record["count_0x80"], count80(fixture.data(name)));
    }
    let tensors = receipt["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 11);
    for (record, placed) in tensors.iter().zip(&plan.entries) {
        assert_eq!(record["name"], placed.name);
        assert_eq!(record["artifact_offset"], placed.data_offset);
        assert_eq!(record["artifact_len"], placed.data_len);
        for source in record["sources"].as_array().unwrap() {
            let name = source["name"].as_str().unwrap();
            assert_eq!(source["sha256"], sha_hex(fixture.data(name)), "{name}");
            let shard =
                fs::read(fixture.src.path().join(source["file"].as_str().unwrap())).unwrap();
            let (start, end) = (
                source["start"].as_u64().unwrap() as usize,
                source["end"].as_u64().unwrap() as usize,
            );
            assert_eq!(&shard[start..end], fixture.data(name), "{name} range");
            if source["dtype"] == "F8_E4M3" {
                assert_eq!(source["count_0x80"], count80(fixture.data(name)));
            }
        }
    }
    assert!(receipt["elapsed_seconds"].as_f64().unwrap() >= 0.0);
}

#[test]
fn ple_scale_shapes_scalar_and_one_by_one_are_accepted() {
    for shape in [vec![], vec![1u64, 1]] {
        let fixture = Fixture::with(fixture_bag(&shape));
        fixture.compact(None).unwrap();
        let plan = reopen_stock_artifact(&fixture.output()).unwrap();
        let name = format!("{PLE_TABLE}.shard_1.weight");
        let entry = plan.entries.iter().find(|e| e.name == name).unwrap();
        let bytes = fs::read(fixture.output()).unwrap();
        assert_eq!(
            &bytes[entry.data_offset as usize..entry.data_offset as usize + 2],
            fixture.data(&format!("{PLE_TABLE}.weight_scale"))
        );
    }
}

#[test]
fn nan_codes_are_refused_everywhere_including_omitted_mtp() {
    let cases = [
        // Emitted routed expert, both NaN codes.
        (
            "model.language_model.layers.1.mlp.experts.1.down_proj.weight",
            0x7F,
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            0xFF,
        ),
        // Emitted PLE shard.
        (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
            0x7F,
        ),
        // Omitted MTP experts: scanned although never copied.
        ("mtp.layers.0.mlp.experts.1.up_proj.weight", 0x7F),
        ("mtp.layers.0.mlp.experts.0.down_proj.weight", 0xFF),
    ];
    for (name, byte) in cases {
        let fixture = Fixture::poisoned(name, byte);
        let error = fixture.compact(None).unwrap_err().to_string();
        assert!(error.contains(name), "{name}: {error}");
        assert!(error.contains(&format!("0x{byte:02X}")), "{name}: {error}");
        assert!(
            fixture.out_is_empty(),
            "{name}: no output or temp file may remain"
        );
    }
}

#[test]
fn unexpected_fp8_and_missing_expert_parts_are_refused() {
    let mut bag = fixture_bag(&[1]);
    bag.insert(
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
        Tensor {
            dtype: "F8_E4M3",
            shape: vec![2, 2],
            data: vec![1; 4],
        },
    );
    let fixture = Fixture::with(bag);
    let error = fixture.compact(None).unwrap_err().to_string();
    assert!(error.contains("shared_expert.up_proj.weight"), "{error}");
    assert!(fixture.out_is_empty());

    let mut bag = fixture_bag(&[1]);
    bag.remove("model.language_model.layers.1.mlp.experts.1.up_proj.weight_scale_inv");
    let fixture = Fixture::with(bag);
    let error = fixture.compact(None).unwrap_err().to_string();
    assert!(error.contains("missing"), "{error}");
    assert!(
        error.contains("layers.1.mlp.experts.1.up_proj.weight_scale_inv"),
        "{error}"
    );
    assert!(fixture.out_is_empty());
}

#[test]
fn existing_destinations_are_refused_untouched() {
    let fixture = Fixture::new();
    fs::write(fixture.output(), b"keep").unwrap();
    let error = fixture.compact(None).unwrap_err().to_string();
    assert!(error.contains("refusing"), "{error}");
    assert_eq!(fs::read(fixture.output()).unwrap(), b"keep");
    fs::remove_file(fixture.output()).unwrap();
    fs::write(receipt_path_for(&fixture.output()), b"keep").unwrap();
    let error = fixture.compact(None).unwrap_err().to_string();
    assert!(error.contains("refusing"), "{error}");
    assert!(!fixture.output().exists());
}

#[test]
fn production_refuses_a_manifest_that_is_not_the_pin_before_any_output() {
    // No manifest at all.
    let fixture = Fixture::new();
    let error = run_stock_fp8_cli(fixture.src.path(), &fixture.output(), Qwen4Mode::Production)
        .unwrap_err()
        .to_string();
    assert!(error.contains(STOCK_FP8_MANIFEST_NAME), "{error}");
    assert!(fixture.out_is_empty());

    // A well-formed manifest whose hash is not the pinned one.
    let own_pin = fixture.write_manifest(None);
    assert_ne!(own_pin, STOCK_FP8_PINNED_MANIFEST_SHA256);
    let error = run_stock_fp8_cli(fixture.src.path(), &fixture.output(), Qwen4Mode::Production)
        .unwrap_err()
        .to_string();
    assert!(error.contains("refused"), "{error}");
    assert!(error.contains(STOCK_FP8_PINNED_MANIFEST_SHA256), "{error}");
    assert!(fixture.out_is_empty());

    // Production never runs unpinned, whatever a caller passes.
    let error = write_stock_fp8_artifact(&StockFp8Options {
        input: fixture.src.path(),
        output: &fixture.output(),
        mode: Qwen4Mode::Production,
        pin: Some(&own_pin),
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("pinned"), "{error}");
    assert!(fixture.out_is_empty());
}

#[test]
fn a_pinned_manifest_must_agree_with_shards_and_sidecars() {
    // Agreement: converts, and the receipt records the verified manifest.
    let fixture = Fixture::new();
    let pin = fixture.write_manifest(None);
    fixture.compact(Some(&pin)).unwrap();
    let receipt = receipt_of(&fixture);
    assert_eq!(receipt["manifest_verified"], true);
    assert_eq!(receipt["manifest_sha256"], pin);
    assert_eq!(receipt["pinned_manifest_sha256"], pin);

    // A shard the manifest does not list.
    let fixture = Fixture::new();
    let pin = fixture.write_manifest(Some("model-00002-of-00002.safetensors"));
    let error = fixture.compact(Some(&pin)).unwrap_err().to_string();
    assert!(error.contains("disagree"), "{error}");
    assert!(fixture.out_is_empty());

    // config.json changed after the manifest was made.
    let fixture = Fixture::new();
    let pin = fixture.write_manifest(None);
    fs::write(
        fixture.src.path().join("config.json"),
        r#"{"model_type":"other"}"#,
    )
    .unwrap();
    let error = fixture.compact(Some(&pin)).unwrap_err().to_string();
    assert!(error.contains("does not match"), "{error}");
    assert!(fixture.out_is_empty());
}

// ------------------------------------------------- stock-fp8-mtp-sidecar (P1-P7)

use hipfire_arch_qwen4::stock_mtp_sidecar::{
    bind_stock_mtp_sidecar, qwen4_stock_mtp_sidecar_path, StockMtpSidecarEnvelope,
    STOCK_MTP_SIDECAR_KEY,
};
use hipfire_runtime::hfq::HfqFile;

const MTP_NAMES: [&str; 3] = [
    "mtp.fc_hidden.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
];

/// A fixture whose AR artifact (the sidecar's base) is already written, plus
/// the manifest pin it was converted under.
fn ar_fixture() -> (Fixture, String) {
    let fixture = Fixture::new();
    let pin = fixture.write_manifest(None);
    fixture.compact(Some(&pin)).unwrap();
    (fixture, pin)
}

fn sidecar_path(fixture: &Fixture) -> PathBuf {
    qwen4_stock_mtp_sidecar_path(&fixture.output())
}

fn mtp_run(
    input: &Fixture,
    base: &Path,
    output: &Path,
    pin: Option<&str>,
    base_sha256: Option<&str>,
) -> Result<mtp_sidecar::StockFp8MtpSummary, Qwen4Error> {
    mtp_sidecar::write_stock_fp8_mtp_sidecar(&mtp_sidecar::StockFp8MtpOptions {
        input: input.src.path(),
        base,
        output,
        mode: Qwen4Mode::CompactFixture,
        pin,
        base_sha256,
    })
}

fn mtp(fixture: &Fixture, pin: &str) -> Result<mtp_sidecar::StockFp8MtpSummary, Qwen4Error> {
    mtp_run(
        fixture,
        &fixture.output(),
        &sidecar_path(fixture),
        Some(pin),
        None,
    )
}

fn dir_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// A second source directory with `bag` as its shards and the first
/// fixture's `fp8.sha256` (the manifest check hashes the manifest, the shard
/// names and the config sidecar, never the shard bytes), so the same pin
/// binds both.
fn variant(of: &Fixture, bag: Bag) -> Fixture {
    let variant = Fixture::with(bag);
    fs::copy(
        of.src.path().join(STOCK_FP8_MANIFEST_NAME),
        variant.src.path().join(STOCK_FP8_MANIFEST_NAME),
    )
    .unwrap();
    variant
}

/// Copy of the fixture base with one same-length byte string replaced.
fn patched_base(fixture: &Fixture, from: &str, to: &str) -> PathBuf {
    assert_eq!(from.len(), to.len());
    let mut bytes = fs::read(fixture.output()).unwrap();
    let at = bytes
        .windows(from.len())
        .position(|window| window == from.as_bytes())
        .expect("pattern is in the base metadata");
    bytes[at..at + from.len()].copy_from_slice(to.as_bytes());
    let path = fixture.out.path().join("patched.hfq");
    fs::write(&path, &bytes).unwrap();
    path
}

fn sidecar_bytes_by_name(path: &Path) -> (Qwen4ReopenPlan, Vec<u8>) {
    (reopen_stock_artifact(path).unwrap(), fs::read(path).unwrap())
}

#[test]
fn p1_sidecar_holds_exactly_the_mtp_entries_in_order_with_stock_geometry() {
    let (fixture, pin) = ar_fixture();
    let summary = mtp(&fixture, &pin).unwrap();
    assert_eq!(summary.entries, 3);
    assert_eq!(summary.expert_entries, 2);
    assert_eq!(summary.mtp_fp8_scanned, 6);
    let sidecar = sidecar_path(&fixture);
    assert_eq!(summary.output, sidecar);
    assert!(sidecar.to_string_lossy().ends_with("stock.hfq.mtp"));
    let (plan, _) = sidecar_bytes_by_name(&sidecar);
    let names: Vec<&str> = plan.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, MTP_NAMES);
    let by_name: BTreeMap<&str, &Qwen4PlanEntry> =
        plan.entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let fc = by_name["mtp.fc_hidden.weight"];
    assert_eq!((fc.quant_type, fc.group_size), (16, 0));
    assert_eq!(fc.shape, vec![2, 2]);
    assert_eq!(fc.data_len, 8);
    let down = by_name["mtp.layers.0.mlp.experts.down_proj"];
    assert_eq!((down.quant_type, down.group_size), (55, 128));
    assert_eq!(down.shape, vec![EXPERTS as u32, 128, 128]);
    assert_eq!(down.data_len, EXPERTS as u64 * DOWN_STRIDE);
    let gate_up = by_name["mtp.layers.0.mlp.experts.gate_up_proj"];
    assert_eq!((gate_up.quant_type, gate_up.group_size), (55, 128));
    assert_eq!(gate_up.shape, vec![EXPERTS as u32, 256, 128]);
    assert_eq!(gate_up.data_len, EXPERTS as u64 * GATE_UP_STRIDE);
    assert_eq!(
        summary.payload_bytes,
        8 + down.data_len + gate_up.data_len
    );
    // No trunk, PLE or vision record rides along.
    assert!(plan.entries.iter().all(|e| e.name.starts_with("mtp.")));
}

#[test]
fn p2_expert_blocks_are_codes_then_scales_then_zero_pad() {
    let (fixture, pin) = ar_fixture();
    mtp(&fixture, &pin).unwrap();
    let (plan, bytes) = sidecar_bytes_by_name(&sidecar_path(&fixture));
    let payload = |name: &str| -> Vec<u8> {
        let entry = plan.entries.iter().find(|e| e.name == name).unwrap();
        bytes[entry.data_offset as usize..(entry.data_offset + entry.data_len) as usize].to_vec()
    };
    let gate_up = payload("mtp.layers.0.mlp.experts.gate_up_proj");
    assert_eq!(gate_up, expected_gate_up(&fixture, "mtp.layers.0"));
    let down = payload("mtp.layers.0.mlp.experts.down_proj");
    assert_eq!(down, expected_down(&fixture, "mtp.layers.0"));
    // The pad bytes are zero.
    for e in 0..EXPERTS {
        let block = &gate_up[e * GATE_UP_STRIDE as usize..(e + 1) * GATE_UP_STRIDE as usize];
        assert!(block[256 * 128 + 4..].iter().all(|&b| b == 0));
        let block = &down[e * DOWN_STRIDE as usize..(e + 1) * DOWN_STRIDE as usize];
        assert!(block[128 * 128 + 2..].iter().all(|&b| b == 0));
    }
    assert_eq!(
        payload("mtp.fc_hidden.weight"),
        fixture.data("mtp.fc_hidden.weight")
    );
}

#[test]
fn p3_metadata_is_one_envelope_bound_to_the_base_and_the_source() {
    let (fixture, pin) = ar_fixture();
    let summary = mtp(&fixture, &pin).unwrap();
    let (plan, bytes) = sidecar_bytes_by_name(&sidecar_path(&fixture));
    let value: Value = serde_json::from_str(&plan.metadata_json).unwrap();
    let object = value.as_object().unwrap();
    assert_eq!(object.len(), 1);
    assert!(object.contains_key(STOCK_MTP_SIDECAR_KEY));
    let envelope: StockMtpSidecarEnvelope = serde_json::from_value(value.clone()).unwrap();
    let manifest = envelope.hipfire_qwen4_mtp_sidecar;
    assert_eq!(manifest.version, 1);
    assert_eq!(manifest.format, "stock-fp8-mtp");

    let base = fs::read(fixture.output()).unwrap();
    let data_offset = u64::from_le_bytes(base[24..32].try_into().unwrap()) as usize;
    assert_eq!(manifest.base.recipe_id, STOCK_FP8_RECIPE_ID);
    assert_eq!(manifest.base.file_bytes, base.len() as u64);
    assert_eq!(manifest.base.index_sha256, sha_hex(&base[..data_offset]));
    assert_eq!(manifest.base.sha256, sha_hex(&base));
    assert_eq!(manifest.base.sha256, summary.base_sha256);
    assert_eq!(manifest.source.manifest_sha256, pin);
    assert_eq!(
        manifest.source.config_sha256,
        sha_hex(&fs::read(fixture.src.path().join("config.json")).unwrap())
    );
    assert_eq!(manifest.tensors.len(), plan.entries.len());
    for (digest, entry) in manifest.tensors.iter().zip(&plan.entries) {
        assert_eq!(digest.name, entry.name);
        let start = entry.data_offset as usize;
        assert_eq!(
            digest.sha256,
            sha_hex(&bytes[start..start + entry.data_len as usize])
        );
    }

    // Unknown fields are refused at both levels.
    let mut extra_top = value.clone();
    extra_top["extra"] = json!(1);
    assert!(serde_json::from_value::<StockMtpSidecarEnvelope>(extra_top).is_err());
    let mut extra_inner = value;
    extra_inner[STOCK_MTP_SIDECAR_KEY]["base"]["extra"] = json!(1);
    assert!(serde_json::from_value::<StockMtpSidecarEnvelope>(extra_inner).is_err());
}

#[test]
fn p4_ar_artifact_is_unchanged_by_the_sidecar_route() {
    let (fixture, pin) = ar_fixture();
    let before = fs::read(fixture.output()).unwrap();
    let before_sha = sha_hex(&before);

    // The AR producer is deterministic: an independent second run over the
    // same source is byte-identical (so one sha is pinned per fixture).
    let second = tempfile::tempdir().unwrap();
    let second_out = second.path().join("stock.hfq");
    write_stock_fp8_artifact(&StockFp8Options {
        input: fixture.src.path(),
        output: &second_out,
        mode: Qwen4Mode::CompactFixture,
        pin: Some(&pin),
    })
    .unwrap();
    assert_eq!(sha_hex(&fs::read(&second_out).unwrap()), before_sha);

    // The sidecar route reads the base and the source and never rewrites them.
    let summary = mtp(&fixture, &pin).unwrap();
    assert_eq!(sha_hex(&fs::read(fixture.output()).unwrap()), before_sha);
    assert_eq!(summary.base_sha256, before_sha);

    // Still the AR-only artifact: no mtp.* record, the omission is recorded.
    let plan = reopen_stock_artifact(&fixture.output()).unwrap();
    assert_eq!(plan.entries.len(), 11);
    assert!(plan.entries.iter().all(|e| !e.name.starts_with("mtp.")));
    let metadata: Value = serde_json::from_str(&plan.metadata_json).unwrap();
    assert_eq!(metadata["qwen4_recipe"]["mtp"], "omitted-ar-only");
    let receipt = receipt_of(&fixture);
    assert_eq!(receipt["artifact_sha256"], before_sha);
    assert_eq!(receipt["census"]["omitted"]["mtp"]["tensors"], 13);
    assert_eq!(receipt["census"]["fp8"]["omitted_mtp"], 6);
}

#[test]
fn p5_refusals_leave_no_output_and_name_the_cause() {
    let (fixture, pin) = ar_fixture();
    let sidecar = sidecar_path(&fixture);
    let receipt = receipt_path_for(&sidecar);
    let untouched = dir_names(fixture.out.path());
    let refused = |error: Qwen4Error, needle: &str| {
        let error = error.to_string();
        assert!(error.contains(needle), "{needle:?} not in {error}");
        assert!(!sidecar.exists() && !receipt.exists(), "{error}");
        assert_eq!(dir_names(fixture.out.path()), untouched, "{error}");
    };

    // The source pin is required.
    refused(
        mtp_run(&fixture, &fixture.output(), &sidecar, None, None).unwrap_err(),
        "pin",
    );
    // --base-sha256 mismatch.
    let wrong = "0".repeat(64);
    refused(
        mtp_run(&fixture, &fixture.output(), &sidecar, Some(&pin), Some(&wrong)).unwrap_err(),
        "--base-sha256",
    );
    // Base recipe is not stock-fp8.
    let base = patched_base(&fixture, "\"id\":\"stock-fp8\"", "\"id\":\"stock-fp9\"");
    let untouched_with_base = dir_names(fixture.out.path());
    let error = mtp_run(&fixture, &base, &sidecar, Some(&pin), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("base recipe is"), "{error}");
    assert!(!sidecar.exists() && !receipt.exists());
    assert_eq!(dir_names(fixture.out.path()), untouched_with_base);
    // Base source manifest differs from the input's pin.
    let mut other = pin.clone();
    other.replace_range(..1, if pin.starts_with('0') { "1" } else { "0" });
    let base = patched_base(&fixture, &pin, &other);
    let error = mtp_run(&fixture, &base, &sidecar, Some(&pin), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("source manifest"), "{error}");
    assert!(!sidecar.exists() && !receipt.exists());
    fs::remove_file(fixture.out.path().join("patched.hfq")).unwrap();

    // A NaN code in an mtp expert (both NaN codes), scanned before any write.
    for (name, byte) in [
        ("mtp.layers.0.mlp.experts.1.up_proj.weight", 0x7F),
        ("mtp.layers.0.mlp.experts.0.down_proj.weight", 0xFF),
    ] {
        let mut bag = fixture_bag(&[1]);
        bag.get_mut(name).unwrap().data[5] = byte;
        let poisoned = variant(&fixture, bag);
        let error = mtp_run(&poisoned, &fixture.output(), &sidecar, Some(&pin), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains(name), "{name}: {error}");
        assert!(error.contains(&format!("0x{byte:02X}")), "{name}: {error}");
        assert!(!sidecar.exists() && !receipt.exists(), "{error}");
        assert_eq!(dir_names(fixture.out.path()), untouched, "{error}");
    }
    // A missing expert part.
    let mut bag = fixture_bag(&[1]);
    bag.remove("mtp.layers.0.mlp.experts.1.up_proj.weight_scale_inv");
    let broken = variant(&fixture, bag);
    let error = mtp_run(&broken, &fixture.output(), &sidecar, Some(&pin), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing"), "{error}");
    assert!(error.contains("mtp.layers.0.mlp.experts.1.up_proj.weight_scale_inv"), "{error}");
    // An extra mtp expert layer.
    let mut bag = fixture_bag(&[1]);
    let mut builder = Builder { bag: Bag::new(), seed: 900 };
    builder.experts("mtp.layers.1");
    bag.extend(builder.bag);
    let extra = variant(&fixture, bag);
    let error = mtp_run(&extra, &fixture.output(), &sidecar, Some(&pin), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exactly one expert layer"), "{error}");
    // A wrong dtype on an mtp tensor.
    let mut bag = fixture_bag(&[1]);
    bag.insert(
        "mtp.fc_hidden.weight".to_string(),
        Tensor {
            dtype: "F8_E4M3",
            shape: vec![2, 2],
            data: vec![1; 4],
        },
    );
    let wrong_dtype = variant(&fixture, bag);
    let error = mtp_run(&wrong_dtype, &fixture.output(), &sidecar, Some(&pin), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("mtp.fc_hidden.weight"), "{error}");
    assert!(error.contains("expected BF16"), "{error}");
    assert!(!sidecar.exists() && !receipt.exists());
    assert_eq!(dir_names(fixture.out.path()), untouched);

    // An existing destination (sidecar, then receipt) is refused untouched.
    fs::write(&sidecar, b"keep").unwrap();
    let error = mtp(&fixture, &pin).unwrap_err().to_string();
    assert!(error.contains("refusing"), "{error}");
    assert_eq!(fs::read(&sidecar).unwrap(), b"keep");
    fs::remove_file(&sidecar).unwrap();
    fs::write(&receipt, b"keep").unwrap();
    let error = mtp(&fixture, &pin).unwrap_err().to_string();
    assert!(error.contains("refusing"), "{error}");
    assert!(!sidecar.exists());
    fs::remove_file(&receipt).unwrap();

    // The production entry point refuses anything but the pinned manifest and
    // a bad --base-sha256, before any output.
    let error = run_stock_fp8_mtp_sidecar_cli(
        fixture.src.path(),
        &fixture.output(),
        Some(&sidecar),
        Some("zz"),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--base-sha256"), "{error}");
    let error = run_stock_fp8_mtp_sidecar_cli(fixture.src.path(), &fixture.output(), Some(&sidecar), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("refused"), "{error}");
    assert!(error.contains(STOCK_FP8_PINNED_MANIFEST_SHA256), "{error}");
    let error = mtp_sidecar::write_stock_fp8_mtp_sidecar(&mtp_sidecar::StockFp8MtpOptions {
        input: fixture.src.path(),
        base: &fixture.output(),
        output: &sidecar,
        mode: Qwen4Mode::Production,
        pin: Some(&pin),
        base_sha256: None,
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("pinned"), "{error}");
    assert!(!sidecar.exists() && !receipt.exists());
    assert_eq!(dir_names(fixture.out.path()), untouched);
}

#[test]
fn p6_written_sidecar_rebinds_and_a_flipped_payload_byte_is_named() {
    let (fixture, pin) = ar_fixture();
    let wrong = sha_hex(b"not the base");
    assert!(mtp_run(&fixture, &fixture.output(), &sidecar_path(&fixture), Some(&pin), Some(&wrong)).is_err());
    let good = sha_hex(&fs::read(fixture.output()).unwrap());
    mtp_run(
        &fixture,
        &fixture.output(),
        &sidecar_path(&fixture),
        Some(&pin),
        Some(&good),
    )
    .unwrap();
    let sidecar = sidecar_path(&fixture);

    // The loader's bind passes every base/envelope check (reasons 1-10) and
    // stops only at the entry count: the 3-entry compact fixture is not the
    // 31-entry production head.
    let base = HfqFile::open_at_offset(&fixture.output(), 0).unwrap();
    let reason = match bind_stock_mtp_sidecar(&base, &sidecar) {
        Ok(_) => panic!("a 3-entry fixture cannot bind as the 31-entry head"),
        Err(reason) => reason,
    };
    assert!(
        reason.contains("holds 3 tensors") && reason.contains("declares 31"),
        "{reason}"
    );

    // The producer's own payload re-read passes, and names the tensor whose
    // byte was flipped (reason 16 wording).
    mtp_sidecar::verify_sidecar_payloads(&sidecar).unwrap();
    let (plan, mut bytes) = sidecar_bytes_by_name(&sidecar);
    let down = plan
        .entries
        .iter()
        .find(|e| e.name == "mtp.layers.0.mlp.experts.down_proj")
        .unwrap();
    bytes[down.data_offset as usize + 7] ^= 0x01;
    let tampered = fixture.out.path().join("tampered.mtp");
    fs::write(&tampered, &bytes).unwrap();
    let error = mtp_sidecar::verify_sidecar_payloads(&tampered)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("tensor mtp.layers.0.mlp.experts.down_proj: payload sha256"),
        "{error}"
    );
}

#[test]
fn p7_receipt_records_every_source_with_range_sha_and_count() {
    assert_eq!(STOCK_F8_MTP, 1_536);
    let (fixture, pin) = ar_fixture();
    let summary = mtp(&fixture, &pin).unwrap();
    let sidecar = sidecar_path(&fixture);
    let receipt_path = receipt_path_for(&sidecar);
    assert_eq!(summary.receipt_path, receipt_path);
    assert!(receipt_path.to_string_lossy().ends_with(".mtp.receipt.json"));
    let receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
    let bytes = fs::read(&sidecar).unwrap();
    assert_eq!(receipt["recipe"], "stock-fp8-mtp-sidecar");
    assert_eq!(receipt["artifact_sha256"], sha_hex(&bytes));
    assert_eq!(summary.sidecar_sha256, sha_hex(&bytes));
    assert_eq!(receipt["artifact"]["bytes"], bytes.len());
    assert_eq!(receipt["manifest_sha256"], pin);
    assert_eq!(receipt["manifest_verified"], true);
    assert_eq!(
        receipt["base"]["sha256"],
        sha_hex(&fs::read(fixture.output()).unwrap())
    );
    // Fixture MTP: 2 experts x {gate, up, down} F8 tensors; 13 mtp sources.
    let fp8 = &receipt["census"]["fp8"];
    assert_eq!(fp8["mtp"], 6);
    assert_eq!(fp8["total"], 6);
    assert_eq!(fp8["scanned"], 6);
    assert_eq!(fp8["nan_codes"], 0);
    assert_eq!(receipt["census"]["mtp"]["tensors"], 13);
    let total80: u64 = fixture
        .bag
        .iter()
        .filter(|(name, t)| name.starts_with("mtp.") && t.dtype == "F8_E4M3")
        .map(|(name, _)| count80(fixture.data(name)))
        .sum();
    assert!(total80 > 0, "fixture must contain 0x80 codes");
    assert_eq!(fp8["count_0x80"]["total"], total80);

    let plan = reopen_stock_artifact(&sidecar).unwrap();
    let tensors = receipt["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 3);
    let mut sources_seen = 0usize;
    for (record, placed) in tensors.iter().zip(&plan.entries) {
        assert_eq!(record["name"], placed.name);
        assert_eq!(record["artifact_offset"], placed.data_offset);
        assert_eq!(record["artifact_len"], placed.data_len);
        let start = placed.data_offset as usize;
        assert_eq!(
            record["payload_sha256"],
            sha_hex(&bytes[start..start + placed.data_len as usize])
        );
        for source in record["sources"].as_array().unwrap() {
            sources_seen += 1;
            let name = source["name"].as_str().unwrap();
            assert!(name.starts_with("mtp."), "{name}");
            assert_eq!(source["sha256"], sha_hex(fixture.data(name)), "{name}");
            let shard =
                fs::read(fixture.src.path().join(source["file"].as_str().unwrap())).unwrap();
            let (start, end) = (
                source["start"].as_u64().unwrap() as usize,
                source["end"].as_u64().unwrap() as usize,
            );
            assert_eq!(&shard[start..end], fixture.data(name), "{name} range");
            if source["dtype"] == "F8_E4M3" {
                assert_eq!(source["count_0x80"], count80(fixture.data(name)));
            }
        }
    }
    // 12 expert sources (codes + scales) + fc_hidden.
    assert_eq!(sources_seen, 13);
    assert!(receipt["elapsed_seconds"].as_f64().unwrap() >= 0.0);
}
