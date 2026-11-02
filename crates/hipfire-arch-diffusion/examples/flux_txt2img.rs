// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end FLUX.1-dev txt2img on the GPU: prompt → PNG.
//!
//! `gpu_flux_block_parity` loads block 0 and stops. This runs the whole
//! model: T5 + CLIP conditioning, the full 19 double + 38 single block denoise
//! loop for N steps, VAE decode, PNG.
//!
//! The VAE decode runs hipfire's own FLUX VAE decoder (both the LDM/taming and
//! diffusers weight namings; GPU path `vae_gpu`, parity vs ComfyUI rel_l2
//! 0.0031 on the golden latent), so this writes a real PNG end-to-end. Set
//! `HIPFIRE_VAE_CONFIG_ONLY=1` to stop at the latent instead and write
//! `<out>.latent` for an external decoder.
//!
//! It also writes the initial latent alongside the image, so the exact same
//! noise can be fed to another implementation for a like-for-like comparison.
//! The seed NUMBER is not portable across implementations — hipfire's
//! `scheduler::seeded_gaussian` and torch's `randn` produce different noise for
//! the same integer — so comparing by seed alone compares two different
//! problems. Comparing by latent is the honest form.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```
//! cargo run --release --features lab --example flux_txt2img \
//!   -p hipfire-arch-diffusion -- <pipe-dir> "<prompt>" [out.png]
//! ```
//! Env: WIDTH/HEIGHT (default 1024), STEPS (default 20), SEED (default 42),
//! DUMP_INIT (path to write the initial noise as a ComfyUI `.latent`).

use hipfire_arch_diffusion::pipeline::{generate_txt2img_prompt_gpu, load_pipe, postprocess_png};
use hipfire_arch_diffusion::vae::LatentNorm;
use rdna_compute::Gpu;

use std::path::PathBuf;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Peak host resident set size in MiB, from `/proc/self/status` `VmHWM`.
///
/// The number this harness exists to keep honest. `VmHWM` is a high-water
/// mark the kernel never lowers, so it reports the worst moment of the whole
/// run — including the weight upload — not the state at exit. On this iGPU
/// "VRAM" is system RAM, so a host table the run does not need is not merely
/// wasteful: it competes with the device allocations and pushes both into
/// zram swap. Returns `None` off Linux.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn report_peak_rss(tag: &str) {
    match peak_rss_mib() {
        Some(mib) => eprintln!("peak host RSS {tag}: {mib:.0} MiB (VmHWM)"),
        None => eprintln!("peak host RSS {tag}: unavailable (no /proc/self/status)"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pipe_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/home/user/flux-pipe".to_string()),
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "a photograph of a red apple on a wooden table".to_string());
    let out_path = args.next().unwrap_or_else(|| "flux_out.png".to_string());

    let width = env_usize("WIDTH", 1024);
    let height = env_usize("HEIGHT", 1024);
    let steps = env_usize("STEPS", 20);
    let seed = env_usize("SEED", 42) as u64;

    eprintln!("pipe   : {}", pipe_dir.display());
    eprintln!("prompt : {prompt}");
    eprintln!("size   : {width}x{height}  steps: {steps}  seed: {seed}");

    let t_load = Instant::now();
    let mut bundle = load_pipe(&pipe_dir).unwrap_or_else(|e| panic!("load_pipe: {e}"));
    eprintln!("loaded pipeline in {:.1} s", t_load.elapsed().as_secs_f64());
    report_peak_rss("after load_pipe");

    let mut gpu = Gpu::init().expect("GPU init failed");
    let t_up = Instant::now();
    bundle
        .ensure_gpu(&mut gpu)
        .unwrap_or_else(|e| panic!("ensure_gpu: {e}"));
    eprintln!("uploaded weights in {:.1} s", t_up.elapsed().as_secs_f64());
    report_peak_rss("after ensure_gpu");

    let t0 = Instant::now();
    let mut last = Instant::now();
    let mut step_times: Vec<f64> = Vec::new();
    let mut on_step = |i: usize, n: usize| {
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        step_times.push(dt);
        eprintln!("  step {}/{}  {:.3} s", i + 1, n, dt);
    };
    let out = generate_txt2img_prompt_gpu(
        &mut bundle,
        &mut gpu,
        &prompt,
        width,
        height,
        steps,
        seed,
        &mut on_step,
    )
    .unwrap_or_else(|e| panic!("generate: {e}"));
    let total = t0.elapsed().as_secs_f64();
    report_peak_rss("after generate");

    // With HIPFIRE_VAE_CONFIG_ONLY the run stops at the latent, and the image
    // is decoded elsewhere. Write the final latent in ComfyUI's `.latent`
    // format — a safetensors holding one `latent_tensor` of shape
    // [1, 16, h/8, w/8] — so ComfyUI's own VAE decodes it. Using the SAME
    // decoder on both sides is what makes the comparison attributable to the
    // transformer, which is the part this project optimized.
    if out.png.is_empty() {
        let up = hipfire_arch_diffusion::pipeline::vae_upscale(&bundle);
        let cfg = &bundle.transformer_cfg;
        let (lh, lw) = (height / up, width / up);
        let n_img = (lh / 2) * (lw / 2);
        let packed = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
        let final_latents = &out
            .steps
            .last()
            .expect("at least one denoise step")
            .latents_out;
        let unpacked = hipfire_arch_diffusion::scheduler::unpack_latents(
            final_latents,
            n_img,
            packed / 4,
            lh,
            lw,
        );
        // Derive the channel count from what `unpack_latents` actually
        // returned rather than from the config: `latent_channels` in a
        // diffusers transformer config is the PACKED count, so recomputing it
        // here would disagree with the tensor by the 2×2 patch factor.
        assert_eq!(
            unpacked.len() % (lh * lw),
            0,
            "unpacked latent not [ch,h,w]"
        );
        let ch = unpacked.len() / (lh * lw);
        // The denoise loop works in MODEL space; a ComfyUI LATENT is in VAE
        // space. ComfyUI applies `process_latent_out` (÷scale + shift) when a
        // sampler emits a latent, so a raw model-space tensor handed to
        // VAEDecode decodes ~2.77x too small — washed out and noise-shot, but
        // structurally right, which makes it look like a model bug. This is
        // the same transform hipfire's own decode path applies.
        let (sf, shf) = match &bundle.meta.latent_norm {
            LatentNorm::ScaleShift { scaling, shift } => (*scaling, *shift),
            LatentNorm::BatchNorm { .. } => panic!("this probe assumes FLUX.1 ScaleShift latents"),
        };
        let vae_space = hipfire_arch_diffusion::scheduler::scale_latents(&unpacked, sf, shf);
        let lat_path = format!("{out_path}.latent");
        std::fs::write(
            &lat_path,
            hipfire_arch_diffusion::pipeline::comfy_latent_bytes(&vae_space, ch, lh, lw),
        )
        .unwrap_or_else(|e| panic!("write {lat_path}: {e}"));
        println!("wrote {lat_path}  [1,{ch},{lh},{lw}]");
        println!("  decode it with ComfyUI (LoadLatent -> VAEDecode -> SaveImage)");
    } else {
        let png = postprocess_png(&out.image, width, height);
        std::fs::write(&out_path, &png).unwrap_or_else(|e| panic!("write {out_path}: {e}"));
        println!("wrote {out_path} ({} bytes)", png.len());
    }

    // Steady-state per-step cost: drop the first step, which carries the
    // per-shape JIT for every kernel the forward touches.
    let steady: Vec<f64> = step_times.iter().skip(1).copied().collect();
    let mean_step = if steady.is_empty() {
        total / steps as f64
    } else {
        steady.iter().sum::<f64>() / steady.len() as f64
    };
    println!("total {total:.2} s for {steps} steps");
    println!("steady-state {mean_step:.3} s/step (first step dropped: JIT)");

    // Reproduce the exact initial noise the run used, so another
    // implementation can be driven from the same latent instead of the same
    // seed integer. Same call, same seed, same length as the generator inside
    // `generate_txt2img_prompt_gpu`.
    if let Ok(p) = std::env::var("DUMP_INIT") {
        let cfg = &bundle.transformer_cfg;
        let up = hipfire_arch_diffusion::pipeline::vae_upscale(&bundle);
        let ch_packed = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
        let (lh, lw) = (height / up, width / up);
        let n_img = (lh / 2) * (lw / 2);
        let noise = hipfire_arch_diffusion::scheduler::seeded_gaussian(n_img * ch_packed, seed);
        let unpacked =
            hipfire_arch_diffusion::scheduler::unpack_latents(&noise, n_img, ch_packed / 4, lh, lw);
        // Written in VAE space, the container ComfyUI's `.latent` files use,
        // so `LoadLatent` reads it back as exactly this noise in model space.
        //
        // What this is FOR: inspecting or diffing hipfire's noise against
        // another implementation's. Comparing by latent is the only honest
        // form, because the same seed INTEGER gives different noise —
        // `seeded_gaussian` and torch's `randn` are different generators.
        //
        // What it is NOT for, and an earlier version of this comment claimed
        // it was: making ComfyUI SAMPLE from hipfire's noise. There is no
        // stock route for that. `KSamplerAdvanced(add_noise=disable)` looks
        // like one and is not — on a flow-matching model the sampler computes
        // `noise_scaling(σ₀=1, noise=0, latent) = 1*0 + (1-1)*latent = 0` and
        // starts from zeros, which denoises into a plausible-looking image
        // that shares nothing with our noise. The golden gate therefore runs
        // the comparison the other way round, taking ComfyUI's noise OUT; see
        // `gpu_flux_golden_latent` and its fixture's `note_noise`.
        let (sf, shf) = match &bundle.meta.latent_norm {
            LatentNorm::ScaleShift { scaling, shift } => (*scaling, *shift),
            LatentNorm::BatchNorm { .. } => panic!("this probe assumes FLUX.1 ScaleShift latents"),
        };
        let vae_space = hipfire_arch_diffusion::scheduler::scale_latents(&unpacked, sf, shf);
        let ch = unpacked.len() / (lh * lw);
        std::fs::write(
            &p,
            hipfire_arch_diffusion::pipeline::comfy_latent_bytes(&vae_space, ch, lh, lw),
        )
        .unwrap_or_else(|e| panic!("dump init latent {p}: {e}"));
        println!("wrote initial noise {p}  [1,{ch},{lh},{lw}]");
    }
}
