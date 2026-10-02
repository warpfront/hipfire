// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Black-box CLI contract of `hipfire-quantize` on tiny synthetic CPU
//! checkpoints: what gets written, and which invocations must fail.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use hipfire_runtime::hfq::HfqFile;
use safetensors::tensor::{Dtype, TensorView};

const MQ4G256: u8 = 13;
const MQ6G256: u8 = 15;
const Q8F16: u8 = 3;

fn bf16_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.01;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// A checkpoint dir with `config.json` and one BF16 shard of `[rows, 256]`
/// (or `[256]` for names containing `norm`) tensors.
fn checkpoint(dir: &Path, config: serde_json::Value, names: &[&str]) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let shape = if name.contains("norm") { vec![256] } else { vec![4, 256] };
            let n: usize = shape.iter().product();
            (name.to_string(), shape, bf16_bytes(n, i))
        })
        .collect();
    let views: HashMap<String, TensorView> = data
        .iter()
        .map(|(n, s, b)| (n.clone(), TensorView::new(Dtype::BF16, s.clone(), b).unwrap()))
        .collect();
    safetensors::serialize_to_file(&views, None, &dir.join("model.safetensors")).unwrap();
    dir.to_path_buf()
}

fn llama(dir: &Path) -> PathBuf {
    checkpoint(
        dir,
        serde_json::json!({"model_type": "llama", "num_hidden_layers": 1}),
        &["model.layers.0.self_attn.q_proj.weight", "model.norm.weight"],
    )
}

fn quantize(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_hipfire-quantize"));
    cmd.args(args).env("HIPFIRE_QUANT_THREADS", "2");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn qt(hfq: &Path, name: &str) -> u8 {
    HfqFile::open(hfq).unwrap().find_tensor_info(name).unwrap().quant_type
}

fn cohere(dir: &Path) -> PathBuf {
    checkpoint(
        dir,
        serde_json::json!({"model_type": "cohere2_moe", "num_hidden_layers": 1}),
        &[
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ],
    )
}

#[test]
fn cohere2_expert_tier_follows_format() {
    let tmp = tempfile::tempdir().unwrap();
    let input = cohere(&tmp.path().join("ck"));
    let expert = "model.layers.0.mlp.experts.0.gate_proj.weight";
    for (format, want) in [("mq4", MQ4G256), ("mq4v2", MQ4G256), ("mq4v1", MQ4G256), ("mq6", MQ6G256), ("q8", Q8F16)] {
        let out_path = tmp.path().join(format!("{format}.hfq"));
        let out = quantize(
            &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", format],
            &[],
        );
        assert!(out.status.success(), "{format}: {}", stderr(&out));
        assert_eq!(qt(&out_path, expert), want, "--format {format} expert dtype");
        assert_eq!(qt(&out_path, "model.layers.0.self_attn.q_proj.weight"), Q8F16);
    }
    // A format the cohere2 handler has no tier for is an error, not Q8.
    let out_path = tmp.path().join("hfq4.hfq");
    let out = quantize(
        &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", "hfq4"],
        &[],
    );
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(!out_path.exists());
}

#[test]
fn gemma4_unified_with_skipped_vision_does_not_claim_vision() {
    let tmp = tempfile::tempdir().unwrap();
    let input = checkpoint(
        &tmp.path().join("ck"),
        serde_json::json!({"model_type": "gemma4", "num_hidden_layers": 1}),
        &[
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight",
        ],
    );
    let out_path = tmp.path().join("g.hfq");
    let out = quantize(
        &[
            "--input", input.to_str().unwrap(),
            "--output", out_path.to_str().unwrap(),
            "--format", "q8",
            "--include-vision",
        ],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let hfq = HfqFile::open(&out_path).unwrap();
    assert!(hfq.find_tensor_info("model.vision_tower.encoder.layers.0.mlp.fc1.weight").is_none());
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).unwrap();
    assert_ne!(meta.get("has_vision"), Some(&serde_json::json!(true)));
}

#[test]
fn unusable_spill_dir_fails_without_output() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_path = tmp.path().join("o.hfq");
    let out = quantize(
        &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", "q8"],
        &[("HIPFIRE_SPILL_DIR", tmp.path().join("missing/dir").to_str().unwrap())],
    );
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("error: cannot create tensor spill file"));
    assert!(!out_path.exists());
}

#[cfg(unix)]
#[test]
fn failed_write_leaves_existing_output_untouched() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_dir = tmp.path().join("ro");
    std::fs::create_dir(&out_dir).unwrap();
    let out_path = out_dir.join("o.hfq");
    std::fs::write(&out_path, b"previous good artifact").unwrap();
    std::fs::set_permissions(&out_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    if std::fs::write(out_dir.join("probe"), b"").is_ok() {
        return; // running as root: permissions cannot inject the failure
    }
    let spill = tmp.path().join("spill");
    std::fs::create_dir(&spill).unwrap();
    let out = quantize(
        &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", "q8"],
        &[("HIPFIRE_SPILL_DIR", spill.to_str().unwrap())],
    );
    std::fs::set_permissions(&out_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert_eq!(std::fs::read(&out_path).unwrap(), b"previous good artifact");
    assert_eq!(std::fs::read_dir(&out_dir).unwrap().count(), 1, "temp file left behind");
}

#[test]
fn unknown_format_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    for format in ["mq4v3", "tq2"] {
        let out_path = tmp.path().join("o.hfq");
        let out = quantize(
            &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", format],
            &[],
        );
        assert_eq!(out.status.code(), Some(2), "{format}: {}", stderr(&out));
        assert!(!out_path.exists());
    }
}

#[test]
fn documented_formats_are_accepted() {
    // docs/QUANTIZE.md format table plus scripted spellings.
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    for format in [
        "mq4", "mq4v2", "mq4v1", "mq4g256", "magnum", "mq4c", "mq6", "mq6v2", "mq5v2", "mq3v2",
        "mq2v2", "hf4", "hfq4", "hfq4g256", "hf6", "hfq6", "hfq6g256", "q8", "q8f16", "q4f16",
        "mq3", "hfp4", "mfp4", "mfp4e8", "f16", "bf16", "oracle",
    ] {
        let out_path = tmp.path().join(format!("{format}.hfq"));
        let out = quantize(
            &["--input", input.to_str().unwrap(), "--output", out_path.to_str().unwrap(), "--format", format],
            &[],
        );
        assert!(out.status.success(), "--format {format}: {}", stderr(&out));
    }
}

#[test]
fn provenance_flags_are_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let input = llama(&tmp.path().join("ck"));
    let out_path = tmp.path().join("o.hfq");
    let out = quantize(
        &[
            "--input", input.to_str().unwrap(),
            "--output", out_path.to_str().unwrap(),
            "--format", "q8",
            "--source-url", "https://huggingface.co/org/model",
            "--license", "Apache-2.0",
        ],
        &[],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let meta: serde_json::Value =
        serde_json::from_str(&HfqFile::open(&out_path).unwrap().metadata_json).unwrap();
    let prov = &meta["hipfire_provenance"];
    assert_eq!(prov["source_url"], "https://huggingface.co/org/model");
    assert_eq!(prov["license"], "Apache-2.0");
}

#[test]
fn flags_a_route_never_reads_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let o = tmp.path().join("o.hfq");
    let o = o.to_str().unwrap();
    for argv in [
        &["--qwen4-flash-next", "--input", "x", "--output", o, "--tier", "xt"][..],
        &["--format", "maple", "--input", "x", "--output", o, "--mq4v2-symmetric"][..],
        &["--flux-pipe", "p", "--output", o, "--fixed-tier", "lm_head:q8"][..],
        &["--input", "x", "--output", o, "--head-only"][..],
    ] {
        let out = quantize(argv, &[]);
        assert_eq!(out.status.code(), Some(2), "{argv:?}: {}", stderr(&out));
        assert!(stderr(&out).contains("has no effect with"), "{argv:?}: {}", stderr(&out));
    }
    // Typos in --kmap-mode no longer fall back to `alternating`.
    let out = quantize(&["--input", "x", "--output", o, "--kmap-mode", "ful"], &[]);
    assert_eq!(out.status.code(), Some(2));
}
