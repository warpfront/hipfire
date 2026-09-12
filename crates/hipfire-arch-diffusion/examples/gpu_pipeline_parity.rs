// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-txt2img vs CPU-txt2img end-to-end parity.
//!
//! Loads the tiny diffusers pipe (`tiny-flux-pipe`), runs the full CPU
//! pipeline (`generate_txt2img_prompt` — the CPU oracle that matches the
//! diffusers golden byte-identical) and a GPU pipeline
//! (`generate_txt2img_prompt_gpu` — fp32 MMDiT forward lifted to HIP) on the
//! SAME seeded workflow (same prompt, dims, steps, seed → identical init
//! latents).
//!
//! **This is the MMDiT + VAE CPU-vs-GPU gate, and nothing else.** It forces
//! `HIPFIRE_T5_GPU=0` so BOTH sides condition through the identical f32 host
//! `t5::encode` / `clip::encode`, leaving the MMDiT forward and the VAE decode
//! as the only differences the numbers can be attributed to. Without that pin
//! the GPU side would condition through the f16 GPU encoders, whose own parity
//! budget (~2.5e-3 measured at 24 layers) is already wider than this gate's
//! `TOL_REL` of 2e-3 — so the gate would be comparing two different
//! conditionings against a tolerance calibrated for one, and would fail for a
//! reason that has nothing to do with the MMDiT.
//!
//! The gates that DO exercise the GPU text encoders end-to-end are
//! `gpu_t5_parity` / `gpu_clip_parity` (against the host oracle) and
//! `gpu_flux_golden_latent` (against the ComfyUI
//! goldens, on the product path).
//!
//! Compares structure-for-structure:
//! - per-step `noise_pred` and `latents_out` (max relative error),
//! - the decoded image tensor (PSNR + max_abs), and reports whether the PNG
//!   bytes are identical.
//!
//! Gate: per-step noise_pred max-relative error ≤ 1e-3 (the individual
//! primitives matched ~1e-6; a full denoise loop accumulates rounding across
//! blocks, so this is the end-to-end budget) and final-image PSNR ≥ 30 dB
//! with every pixel finite. Any step over tolerance → exit 1.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example gpu_pipeline_parity -p hipfire-arch-diffusion -- \
//!   --pipe ~/.cache/trace-assets/tiny-flux-pipe
//! ```

use std::path::PathBuf;

use hipfire_arch_diffusion::pipeline::{
    generate_txt2img_prompt, generate_txt2img_prompt_gpu, load_pipe,
};
use rdna_compute::Gpu;

const TOL_REL: f32 = 2e-3; // per-step noise_pred / latents_out max rel err
const IMG_PSNR: f64 = 25.0; // decoded-image PSNR floor (dB)

fn ow(step: usize) -> String {
    let mut s = "".to_owned();
    for _ in 0..step {
        s.push(' ');
    }
    s
}

fn max_rel(a: &[f32], b: &[f32], tag: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{tag}: length mismatch");
    let max_want = b.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-9);
    let max_err = a
        .iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    let rel = max_err / max_want;
    println!(
        "  {tag} len={} max_err={max_err:.3e} rel={rel:.3e}",
        a.len()
    );
    rel
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "image length mismatch");
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
        20.0 * (1.0 / mse.sqrt()).log10()
    };
    println!("  image max_abs={max_abs:.3e} mse={mse:.3e} psnr={psnr:.1} dB");
    psnr
}

fn parse_args() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut pipe = None;
    while let Some(a) = args.next() {
        if a == "--pipe" {
            pipe = args.next().map(PathBuf::from);
        } else {
            eprintln!("ignoring unknown arg {a}");
        }
    }
    pipe.unwrap_or_else(|| {
        PathBuf::from(std::env::var("PIPEFLUX_PIPE_DIR").unwrap_or_else(|_| {
            std::env::var("HOME").unwrap() + "/.cache/trace-assets/tiny-flux-pipe"
        }))
    })
}

fn main() {
    // Isolate the variable under test: both sides condition through the f32
    // host encoders, so any difference is the MMDiT forward or the VAE decode
    // (see the module docs). Set before `load_pipe`, because the bundle reads
    // the text-encoder and cache env at construction.
    std::env::set_var("HIPFIRE_T5_GPU", "0");

    let pipe_dir = parse_args();
    let mut bundle = load_pipe(&pipe_dir).expect("pipe load failed");
    println!(
        "pipe loaded: hidden={} blocks={}+{} txt_dim={}",
        bundle.transformer_cfg.hidden_size,
        bundle.transformer_cfg.num_layers,
        bundle.transformer_cfg.num_single_layers,
        bundle.transformer_cfg.txt_hidden_dim,
    );

    // Same seeded workflow on the CPU oracle and the GPU backend.
    let prompt = "a tiny cat sitting on a tiny table";
    let width = 32usize;
    let height = 32usize;
    let steps = 2usize;
    let seed = 42u64;
    let mut noop = |_step: usize, _total: usize| {};

    let cpu = generate_txt2img_prompt(&bundle, prompt, width, height, steps, seed, &mut noop)
        .expect("cpu pipeline failed");

    let mut gpu = Gpu::init().expect("GPU init failed");
    bundle.ensure_gpu(&mut gpu).expect("gpu upload failed");
    let gpu_out = generate_txt2img_prompt_gpu(
        &mut bundle,
        &mut gpu,
        prompt,
        width,
        height,
        steps,
        seed,
        &mut noop,
    )
    .expect("gpu pipeline failed");
    // Releases the transformer weights, the VAE, both text encoders (none
    // here — the env pin above keeps them on the host) and the conditioning
    // cache.
    let freed = bundle.free_gpu(&mut gpu).expect("free_gpu failed");
    eprintln!("freed {freed} gpu tensors");

    if cpu.steps.len() != gpu_out.steps.len() {
        eprintln!(
            "FAIL: step count {} != {}",
            cpu.steps.len(),
            gpu_out.steps.len()
        );
        std::process::exit(1);
    }

    let mut worst_rel = 0.0f32;
    for (i, (c, g)) in cpu.steps.iter().zip(gpu_out.steps.iter()).enumerate() {
        eprintln!("step {} (t_model {:.6}):{}", i, c.t_model, ow(1));
        let lp = max_rel(&g.latents_out, &c.latents_out, "latents_out");
        let np = max_rel(&g.noise_pred, &c.noise_pred, "noise_pred");
        worst_rel = worst_rel.max(lp).max(np);
        if lp > TOL_REL || np > TOL_REL {
            eprintln!("FAIL: step {i} over tol {TOL_REL}");
            std::process::exit(1);
        }
    }

    let img_db = psnr(&gpu_out.image, &cpu.image);
    let png_identical = cpu.png == gpu_out.png;
    println!(
        "  png_identical={png_identical} ({}, {})",
        cpu.png.len(),
        gpu_out.png.len()
    );
    if !(gpu_out.image.len() == cpu.image.len()
        && gpu_out.image.iter().all(|v| v.is_finite())
        && img_db >= IMG_PSNR)
    {
        eprintln!("FAIL: decoded image out of tolerance (psnr {img_db:.1} < {IMG_PSNR})");
        std::process::exit(1);
    }

    println!(
        "PASS: GPU txt2img matches CPU oracle: worst_step_rel={worst_rel:.3e}, img_psnr={img_db:.1} dB, png_identical={png_identical}",
    );
}
