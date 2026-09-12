// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixture gate: full CPU txt2img parity vs the diffusers golden.
//!
//! Reads the tiny diffusers pipe (`tiny-flux-pipe`) + the golden capture
//! (`tiny-pipeline-golden/golden.json`, `golden.png`), runs the Rust CPU
//! pipeline on the exact golden inputs, and compares every intermediate:
//!
//! - T5 encoder: per-layer + final hidden (txt)
//! - CLIP encoder: pooled (vec) + per-layer
//! - per-step denoise: latents_in, noise_pred, latents_out
//! - VAE decoder: per-block intermediates, decoded image tensor
//! - postprocess: PNG pixels byte-identical to `golden.png`
//!
//! Gate: ≥ 50 dB PSNR (equivalently ≤ ~3e-3 max_abs relative — same language
//! as the block gate) on `final` and `noise_pred` per step, with the
//! PNG pixel box requiring exact bytes.
//!
//! ```text
//! cargo run --release -p hipfire-arch-diffusion --features lab \
//!   --example flux_pipeline_parity -- \
//!   --pipe ~/.cache/trace-assets/tiny-flux-pipe \
//!   --golden crates/hipfire-arch-diffusion/tests/fixtures/tiny-pipeline
//! ```
//!
//! Dev loop: `--self` runs the pipeline on synthetic inputs (no weights, no
//! goldens) to check shape/finiteness only.

use std::path::PathBuf;

use hipfire_arch_diffusion::flux::MlpAct;
use hipfire_arch_diffusion::pipeline::{generate_txt2img, load_pipe, Txt2ImgInput};

const DB: f64 = 20.0; // PSNR → dB factor
const GATE_DB: f64 = 50.0;

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr length mismatch");
    let mut mse = 0.0f64;
    let mut max_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        max_abs = max_abs.max(d);
        mse += d * d;
    }
    mse /= a.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        DB * (1.0 / mse.sqrt()).log10()
    };
    println!("  max_abs={max_abs:.3e} mse={mse:.3e} psnr={psnr:.1} dB");
    psnr
}

fn parse_args() -> (PathBuf, PathBuf, bool, Option<String>) {
    let mut args = std::env::args().skip(1);
    let mut pipe = None;
    let mut golden = None;
    let mut self_only = false;
    let mut prompt = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--pipe" => pipe = args.next().map(PathBuf::from),
            "--golden" => golden = args.next().map(PathBuf::from),
            "--self" => self_only = true,
            "--prompt" => prompt = args.next(),
            other => eprintln!("ignoring unknown arg {other}"),
        }
    }
    (
        pipe.unwrap_or_default(),
        golden.unwrap_or_default(),
        self_only,
        prompt,
    )
}

fn main() {
    let (pipe_dir, golden_dir, self_only, prompt_override) = parse_args();
    let pipe_dir = if pipe_dir.as_os_str().is_empty() {
        PathBuf::from(std::env::var("PIPEFLUX_PIPE_DIR").unwrap_or_else(|_| {
            std::env::var("HOME").unwrap() + "/.cache/trace-assets/tiny-flux-pipe"
        }))
    } else {
        pipe_dir
    };
    let golden_dir = if golden_dir.as_os_str().is_empty() {
        PathBuf::from(std::env::var("PIPEFLUX_GOLDEN_DIR").unwrap_or_else(|_| {
            std::env::var("HOME").unwrap() + "/.cache/trace-assets/tiny-pipeline-golden"
        }))
    } else {
        golden_dir
    };

    let bundle = load_pipe(&pipe_dir).expect("pipe load failed");
    println!(
        "pipeline loaded: hidden={} blocks={}+{} txt_dim={} vae={:?}",
        bundle.transformer_cfg.hidden_size,
        bundle.transformer_cfg.num_layers,
        bundle.transformer_cfg.num_single_layers,
        bundle.transformer_cfg.txt_hidden_dim,
        bundle.vae.config.block_out_channels
    );

    if self_only {
        // Shape/finiteness smoke on the real loader: zero latents, 2 steps.
        let ids: Vec<u32> = (1..=32).collect();
        let mask = vec![1u8; 32];
        let out = generate_txt2img(
            &bundle,
            &Txt2ImgInput {
                txt_ids: &ids,
                txt_mask: &mask,
                // FLUX.1 txt2img: no reference-image tokens.
                references: &[],
                clip_ids: &ids,
                clip_mask: &mask,
                init_latents: None,
                height: 32,
                width: 32,
                steps: 1,
                mlp_act: MlpAct::GeluTanh,
                prompt_key: None,
            },
        )
        .expect("self run failed");
        let (w, h) = out.image_shape;
        assert_eq!(out.image.len(), 3 * w * h);
        assert!(out.image.iter().all(|v| v.is_finite()), "non-finite image");
        assert!(!out.png.is_empty(), "empty png");
        println!("self: image[{}x{}] png[{} B] finite ✓", w, h, out.png.len());
        return;
    }

    let golden_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir.join("golden.json")).expect("golden.json missing"),
    )
    .expect("golden.json invalid");
    if let Some(prompt) = prompt_override {
        // ── tokenizer gate: my tokenizers must reproduce the golden ids ──
        let cond = hipfire_arch_diffusion::pipeline::condition_prompt(&bundle, &prompt, 64)
            .unwrap_or_else(|e| panic!("condition_prompt: {e}"));
        let g = |k: &str| golden_json["inputs"][k].as_array().unwrap();
        let gold_t5: Vec<u32> = g("t5_ids")
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let gold_clip: Vec<u32> = g("clip_ids")
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            cond.txt_ids, gold_t5,
            "t5 tokenizer diverges from the golden capture for {prompt:?}"
        );
        assert_eq!(
            cond.clip_ids, gold_clip,
            "clip tokenizer diverges from the golden capture for {prompt:?}"
        );
        println!("tokenizers: byte-exact vs golden ids for {prompt:?} ✓");
        return;
    }
    let g = |k: &str| golden_json["inputs"][k].as_array().unwrap();
    let t5_ids: Vec<u32> = g("t5_ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let t5_mask: Vec<u8> = g("t5_mask")
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let clip_ids: Vec<u32> = g("clip_ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let clip_mask: Vec<u8> = g("clip_mask")
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let steps = golden_json["steps"].as_array().unwrap().len();
    let init_packed: Vec<f32> = golden_json["steps"][0]["latents_in"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let height = golden_json["height"].as_u64().unwrap() as usize;
    let width = golden_json["width"].as_u64().unwrap() as usize;

    // ── conditioning parity ────────────────────────────────────────────
    // The CPU reference encoder needs the FULL host tables; `load_pipe`
    // streams the linears to the GPU and keeps only the light set.
    let t5_host = bundle.t5_host().expect("materialise host T5 weights");
    let (t5_hidden, t5_layers) = hipfire_arch_diffusion::t5::encode(&t5_host, &t5_ids, &t5_mask);
    let golden_txt: Vec<f32> = g("txt")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    println!("  part   t5 final:");
    let db = psnr(&t5_hidden, &golden_txt);
    assert!(db >= GATE_DB, "t5 final below gate");
    for (i, layer) in t5_layers.iter().enumerate() {
        let gk = format!("t5_layer_{i}");
        let Some(gv) = golden_json["encoder_intermediates"]["t5"][&gk].as_array() else {
            println!("  part  t5 l{i}: (per-layer dump trimmed from fixture, skip)");
            continue;
        };
        let gv: Vec<f32> = gv.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        println!("  part  t5 l{i}:");
        assert!(psnr(layer, &gv) >= GATE_DB, "t5 layer {i} below gate");
    }

    // This gate compares against a T5/CLIP golden, so it is FLUX.1-only: a
    // FLUX.2 (Klein) pipe conditions on a Qwen3 text encoder and has no CLIP at all.
    let clip = match &bundle.cond {
        hipfire_arch_diffusion::pipeline::TextCond::T5Clip { clip, .. } => clip,
        hipfire_arch_diffusion::pipeline::TextCond::Qwen3 { .. } => {
            panic!("flux_pipeline_parity is a FLUX.1 gate; this pipe has no CLIP encoder")
        }
    };
    let (_, clip_pooled, clip_layers) =
        hipfire_arch_diffusion::clip::encode(clip, &clip_ids, &clip_mask);
    let golden_vec: Vec<f32> = g("vec")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    println!("  part  clip pooled:");
    assert!(
        psnr(&clip_pooled, &golden_vec) >= GATE_DB,
        "clip pooled below gate"
    );
    for (i, layer) in clip_layers.iter().enumerate() {
        let gk = format!("clip_layer_{i}");
        let Some(gv) = golden_json["encoder_intermediates"]["clip"][&gk].as_array() else {
            println!("  part clip l{i}: (per-layer dump trimmed from fixture, skip)");
            continue;
        };
        let gv: Vec<f32> = gv.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        println!("  part clip l{i}:");
        assert!(psnr(layer, &gv) >= GATE_DB, "clip layer {i} below gate");
    }

    // ── denoise loop ───────────────────────────────────────────────────
    let out = generate_txt2img(
        &bundle,
        &Txt2ImgInput {
            txt_ids: &t5_ids,
            txt_mask: &t5_mask,
            // FLUX.1 txt2img: no reference-image tokens.
            references: &[],
            clip_ids: &clip_ids,
            clip_mask: &clip_mask,
            init_latents: Some(&init_packed),
            height,
            width,
            steps,
            mlp_act: MlpAct::GeluTanh, // BFL/diffusers/ComfyUI agree: GELU-tanh
            prompt_key: None,
        },
    )
    .expect("pipeline run failed");

    let golden_steps = golden_json["steps"].as_array().unwrap();
    for (i, rec) in out.steps.iter().enumerate() {
        let gi = &golden_steps[i];
        let fmt = |k: &str| -> Vec<f32> {
            gi[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let db_in = psnr(&rec.latents_in, &fmt("latents_in"));
        let db_pred = psnr(&rec.noise_pred, &fmt("noise_pred"));
        let db_out = psnr(&rec.latents_out, &fmt("latents_out"));
        assert!(db_in >= GATE_DB, "step {i} latents_in");
        assert!(db_pred >= GATE_DB, "step {i} noise_pred");
        assert!(db_out >= GATE_DB, "step {i} latents_out");
        println!(
            "  step {i}: latents {db_in:.1} dB, noise_pred {db_pred:.1} dB, out {db_out:.1} dB"
        );
    }

    // ── VAE decode ─────────────────────────────────────────────────────
    let golden_img: Vec<f32> = golden_json["decode"]["image"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    println!("  part  vae image:");
    let db_img = psnr(&out.image, &golden_img);
    assert!(db_img >= GATE_DB, "vae decode below gate");

    // ── PNG bytes ──────────────────────────────────────────────────────
    let golden_png = std::fs::read(golden_dir.join("golden.png")).expect("golden.png missing");
    let g_png = image::load_from_memory(&golden_png)
        .expect("golden.png invalid")
        .to_rgb8();
    let (gw, gh) = g_png.dimensions();
    let my_png = image::load_from_memory(&out.png)
        .expect("our png invalid")
        .to_rgb8();
    assert_eq!((gw, gh), my_png.dimensions());
    let diff = g_png
        .as_raw()
        .iter()
        .zip(my_png.as_raw().iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(diff, 0, "PNG pixels differ in {diff} bytes vs golden.png");
    println!(
        "png: {gw}x{gh}, byte-identical pixels ✓ ({} bytes)",
        out.png.len()
    );

    println!("PIPELINE PARITY PASS — every part ≥ {GATE_DB} dB, PNG byte-identical");
}
