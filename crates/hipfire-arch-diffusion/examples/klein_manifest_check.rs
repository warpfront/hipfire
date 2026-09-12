// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-checkpoint manifest check for a FLUX.2 (Klein) diffusers pipe
//! directory: opens `transformer/`, `text_encoder/`, and `vae/`, and checks
//! every canonical manifest key's source parts against the real safetensors
//! header — presence and element count, without decoding any tensor bytes.
//! CPU-only; no GPU. This is what produced the fixture key lists under
//! `tests/fixtures/klein/`.
//!
//! Build + run:
//! ```
//! cargo run --release --features lab --example klein_manifest_check \
//!   -p hipfire-arch-diffusion -- <pipe-dir> [--dump-dir DIR]
//! ```
//! `<pipe-dir>` is a diffusers pipe directory with `transformer/`,
//! `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/` subdirs (Klein 4B or
//! 9B). `--dump-dir DIR` writes sorted tensor-name lists to
//! `DIR/{transformer,text_encoder,vae}.txt`, one name per line — the source
//! for the committed fixture files.
//!
//! Exits 1 if any manifest key's source part is missing or has the wrong
//! element count, or if either VAE half fails to load.

use hipfire_arch_diffusion::config::FluxDiffusionConfig;
use hipfire_arch_diffusion::flux::FluxPlan;
use hipfire_arch_diffusion::manifest::expected_flux_keys;
use hipfire_arch_diffusion::qwen3::Qwen3Plan;
use hipfire_arch_diffusion::vae::{LatentNorm, VaeDecoderWeights, VaeEncoderWeights};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::PathBuf;

fn dump(dir: &Option<PathBuf>, file: &str, names: &[&str]) {
    let Some(dir) = dir else { return };
    let body = names.join("\n") + if names.is_empty() { "" } else { "\n" };
    std::fs::write(dir.join(file), body)
        .unwrap_or_else(|e| panic!("write {}/{file}: {e}", dir.display()));
}

fn main() {
    let mut pipe_dir: Option<PathBuf> = None;
    let mut dump_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--dump-dir" {
            dump_dir = Some(PathBuf::from(
                args.next().expect("--dump-dir requires a value"),
            ));
        } else if pipe_dir.is_none() {
            pipe_dir = Some(PathBuf::from(a));
        } else {
            panic!("unexpected argument: {a}");
        }
    }
    let pipe_dir = pipe_dir
        .unwrap_or_else(|| panic!("usage: klein_manifest_check <pipe_dir> [--dump-dir DIR]"));
    if let Some(d) = &dump_dir {
        std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("create dump dir: {e}"));
    }

    let mut ok = true;

    // ── transformer ──────────────────────────────────────────────────
    {
        let src = SafetensorsSource::open(&pipe_dir.join("transformer"))
            .unwrap_or_else(|e| panic!("open transformer: {e}"));
        let cfg_json: serde_json::Value = serde_json::from_str(src.metadata_json())
            .unwrap_or_else(|e| panic!("transformer config.json invalid: {e}"));
        let cfg_json = cfg_json.get("config").cloned().unwrap_or(cfg_json);
        let cfg = FluxDiffusionConfig::from_json(&cfg_json)
            .unwrap_or_else(|e| panic!("transformer config: {e}"));
        let plan = FluxPlan::flux2_diffusers(&cfg);

        let mut names: Vec<&str> = src.tensor_names();
        names.sort();
        dump(&dump_dir, "transformer.txt", &names);

        let keys = expected_flux_keys(&cfg);
        let mut n_parts = 0usize;
        let mut missing: Vec<String> = Vec::new();
        for key in &keys {
            let parts = plan
                .parts(&key.name)
                .unwrap_or_else(|e| panic!("plan: {e}"));
            for part in parts {
                n_parts += 1;
                match src.tensor_info(&part.name) {
                    Some(info) => {
                        let n: usize = info.shape.iter().product();
                        let want = part.rows * part.cols;
                        if n != want {
                            missing.push(format!(
                                "{} (elem count {n} != expected {want}, shape {:?})",
                                part.name, info.shape
                            ));
                        }
                    }
                    None => missing.push(part.name.clone()),
                }
            }
        }
        println!(
            "transformer: {family:?} hidden={hidden} layers={layers}/{single} heads={heads} \
             head_dim={hd} mlp={f} patch_in={patch_in} txt_hidden={txt} — {n_keys} keys, \
             {n_parts} parts, {n_missing} missing",
            family = cfg.family,
            hidden = cfg.hidden_size,
            layers = cfg.num_layers,
            single = cfg.num_single_layers,
            heads = cfg.num_attention_heads,
            hd = cfg.head_dim,
            f = cfg.mlp_width(),
            patch_in = cfg.patch_in(),
            txt = cfg.txt_hidden_dim,
            n_keys = keys.len(),
            n_missing = missing.len(),
        );
        for m in &missing {
            println!("  missing: {m}");
        }
        if !missing.is_empty() {
            ok = false;
        }
    }

    // ── text_encoder (Qwen3) ────────────────────────────────────────
    {
        let src = SafetensorsSource::open(&pipe_dir.join("text_encoder"))
            .unwrap_or_else(|e| panic!("open text_encoder: {e}"));
        let mut names: Vec<&str> = src.tensor_names();
        names.sort();
        dump(&dump_dir, "text_encoder.txt", &names);

        let plan = Qwen3Plan::detect(&src).unwrap_or_else(|e| panic!("qwen3 detect: {e}"));
        let layers = plan.config.layers;
        let mut missing: Vec<String> = Vec::new();
        let mut n_checked = 0usize;
        for i in [0usize, layers.saturating_sub(1)] {
            for (name, rows, cols) in plan.layer_keys(i) {
                n_checked += 1;
                match src.tensor_info(&name) {
                    Some(info) => {
                        let n: usize = info.shape.iter().product();
                        let want = rows * cols;
                        if n != want {
                            missing.push(format!("{name} (elem count {n} != expected {want})"));
                        }
                    }
                    None => missing.push(name),
                }
            }
            if layers <= 1 {
                break;
            }
        }
        n_checked += 1;
        if src.tensor_info("model.embed_tokens.weight").is_none() {
            missing.push("model.embed_tokens.weight".to_string());
        }
        println!(
            "text_encoder: qwen3 hidden={} layers={} heads={} kv_heads={} head_dim={} \
             intermediate={} vocab={} — {n_checked} keys checked, {n_missing} missing",
            plan.config.hidden,
            layers,
            plan.config.heads,
            plan.config.kv_heads,
            plan.config.head_dim,
            plan.config.intermediate,
            plan.config.vocab,
            n_missing = missing.len(),
        );
        for m in &missing {
            println!("  missing: {m}");
        }
        if !missing.is_empty() {
            ok = false;
        }
    }

    // ── vae ──────────────────────────────────────────────────────────
    {
        let src = SafetensorsSource::open(&pipe_dir.join("vae"))
            .unwrap_or_else(|e| panic!("open vae: {e}"));
        let mut names: Vec<&str> = src.tensor_names();
        names.sort();
        dump(&dump_dir, "vae.txt", &names);

        let enc_res = VaeEncoderWeights::load(&src);
        let dec_res = VaeDecoderWeights::load(&src);
        match (&enc_res, &dec_res) {
            (Ok(_enc), Ok(dec)) => {
                let norm_desc = match &dec.latent_norm {
                    LatentNorm::BatchNorm { mean, .. } => format!("BatchNorm({})", mean.len()),
                    LatentNorm::ScaleShift { scaling, shift } => {
                        format!("ScaleShift(scale={scaling}, shift={shift})")
                    }
                };
                println!("vae: encoder ok, decoder ok, latent_norm = {norm_desc}");
            }
            _ => {
                if let Err(e) = &enc_res {
                    println!("vae: encoder load FAILED: {e}");
                }
                if let Err(e) = &dec_res {
                    println!("vae: decoder load FAILED: {e}");
                }
                ok = false;
            }
        }
    }

    if !ok {
        eprintln!("klein_manifest_check: FAILED — see missing/failed entries above");
        std::process::exit(1);
    }
}
