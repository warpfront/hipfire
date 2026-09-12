// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end FLUX.2 Klein image generation on the GPU: prompt → PNG.
//!
//! The Klein twin of `flux_txt2img`, and the first entry point that runs the
//! whole FLUX.2 stack on the device — the Qwen3 conditioning tower
//! (`qwen3_gpu::encode_taps`, three [`KLEIN_TAPS`] layers concatenated into a
//! device-resident `[len, 3*hidden]` txt stream), the shared-modulation
//! MMDiT forward with 4-axis id-table RoPE, the empirical-mu sigma schedule,
//! and the VAE decode. Only the PNG postprocess runs on the host.
//!
//! Reference images (`--image`, up to 4) take the edit path: each is decoded,
//! area-capped and snapped to a multiple of 16, VAE-encoded on the device,
//! packed and normalized into the same latent space the generated tokens live
//! in, and appended to the image stream at RoPE time id `10 * (i + 1)` where
//! they condition every step without ever being denoised. With a reference
//! and no `--size`, the output takes the first reference's size.
//!
//! Build + run (GPU required, gpu-lock it):
//! ```text
//! flock /tmp/hipfire-gpu.lock cargo run --release --features lab \
//!   -p hipfire-arch-diffusion --example klein_txt2img -- \
//!   <pipe-dir> --prompt "..." --out /tmp/klein.png [--steps 4] [--seed 7] \
//!   [--size 1024x1024] [--image ref.png]...
//! ```
//!
//! [`KLEIN_TAPS`]: hipfire_arch_diffusion::qwen3::KLEIN_TAPS

use hipfire_arch_diffusion::pipeline::{generate_img_prompt_gpu, load_pipe};
use hipfire_arch_diffusion::refimg::{load_reference, RefImage};
use rdna_compute::Gpu;

use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: klein_txt2img <pipe_dir> --prompt \"...\" --out /path.png \
                     [--steps 4] [--seed 7] [--size WxH] [--image path]...";

struct Args {
    pipe_dir: PathBuf,
    prompt: String,
    out: String,
    steps: usize,
    seed: u64,
    size: Option<(usize, usize)>,
    images: Vec<PathBuf>,
}

/// Parse the flag form. Fails closed on an unknown flag or a missing value:
/// a typo'd `--setps 4` silently generating 4 steps' worth of the DEFAULT
/// schedule is exactly the kind of thing that makes a bench number wrong
/// without ever looking wrong.
fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let pipe_dir = PathBuf::from(it.next().ok_or(USAGE)?);
    let mut a = Args {
        pipe_dir,
        prompt: String::new(),
        out: "klein_out.png".into(),
        steps: 4,
        seed: 7,
        size: None,
        images: Vec::new(),
    };
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--prompt" => a.prompt = val()?,
            "--out" => a.out = val()?,
            "--steps" => {
                a.steps = val()?
                    .parse()
                    .map_err(|e| format!("--steps: {e}\n{USAGE}"))?
            }
            "--seed" => {
                a.seed = val()?
                    .parse()
                    .map_err(|e| format!("--seed: {e}\n{USAGE}"))?
            }
            "--size" => {
                let s = val()?;
                let (w, h) = s
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("--size wants WxH, got {s:?}\n{USAGE}"))?;
                a.size = Some((
                    w.parse().map_err(|e| format!("--size width: {e}"))?,
                    h.parse().map_err(|e| format!("--size height: {e}"))?,
                ));
            }
            "--image" => a.images.push(PathBuf::from(val()?)),
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
    }
    if a.prompt.is_empty() {
        return Err(format!("--prompt is required\n{USAGE}"));
    }
    Ok(a)
}

/// Peak host resident set size in MiB, from `/proc/self/status` `VmHWM`.
/// A high-water mark the kernel never lowers, so it reports the worst moment
/// of the whole run — including the streamed weight upload — not the state at
/// exit. On a unified-memory box a host table the run does not need competes
/// with the device allocations rather than merely wasting space.
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
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    eprintln!("pipe   : {}", a.pipe_dir.display());
    eprintln!("prompt : {}", a.prompt);
    match a.size {
        Some((w, h)) => eprintln!("size   : {w}x{h}  steps: {}  seed: {}", a.steps, a.seed),
        None => eprintln!(
            "size   : (from reference, else 1024x1024)  steps: {}  seed: {}",
            a.steps, a.seed
        ),
    }

    let refs: Vec<RefImage> = a
        .images
        .iter()
        .map(|p| load_reference(p).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    for (i, r) in refs.iter().enumerate() {
        eprintln!(
            "ref {i}  : {} -> {}x{}",
            a.images[i].display(),
            r.width,
            r.height
        );
    }

    let t_load = Instant::now();
    let mut bundle = load_pipe(&a.pipe_dir).unwrap_or_else(|e| panic!("load_pipe: {e}"));
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
    let mut step_ms: Vec<f64> = Vec::new();
    let mut on_step = |i: usize, n: usize| {
        let ms = last.elapsed().as_secs_f64() * 1e3;
        last = Instant::now();
        step_ms.push(ms);
        eprintln!("  step {i}/{n}  {ms:.1} ms");
    };
    let (w, h) = match a.size {
        Some((w, h)) => (Some(w), Some(h)),
        None => (None, None),
    };
    let out = generate_img_prompt_gpu(
        &mut bundle,
        &mut gpu,
        &a.prompt,
        w,
        h,
        a.steps,
        a.seed,
        &refs,
        &mut on_step,
    )
    .unwrap_or_else(|e| panic!("generate: {e}"));
    let total = t0.elapsed().as_secs_f64();
    report_peak_rss("after generate");

    if out.png.is_empty() {
        // `HIPFIRE_VAE_CONFIG_ONLY=1` stops at the latent.
        println!(
            "no PNG (HIPFIRE_VAE_CONFIG_ONLY): {} steps kept",
            out.steps.len()
        );
    } else {
        std::fs::write(&a.out, &out.png).unwrap_or_else(|e| panic!("write {}: {e}", a.out));
        let (ow, oh) = out.image_shape;
        println!("wrote {} ({} bytes, {ow}x{oh})", a.out, out.png.len());
    }

    // Velocity statistics for the FIRST step. When an image comes back as
    // noise or a flat colour these are the two numbers that say which stage
    // to look at — a near-zero std means the transformer produced no signal,
    // a std far from ~1 means the latent scaling is off — and they cost
    // nothing to record while the run is still in hand.
    if let Some(first) = out.steps.first() {
        let v = &first.noise_pred;
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        println!(
            "step 1 velocity: mean {mean:+.5} std {:.5} over {} values (t={:.4})",
            var.sqrt(),
            v.len(),
            first.t_model
        );
    }

    // Steady state drops the first step, which carries the per-shape JIT for
    // every kernel the forward touches (`docs/methodology/perf-benchmarking.md`).
    let steady: Vec<f64> = step_ms.iter().skip(1).copied().collect();
    let mean_step = if steady.is_empty() {
        total * 1e3 / a.steps as f64
    } else {
        steady.iter().sum::<f64>() / steady.len() as f64
    };
    println!(
        "per-step ms: {}",
        step_ms
            .iter()
            .map(|m| format!("{m:.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("total {total:.2} s for {} steps", a.steps);
    println!("steady-state {mean_step:.0} ms/step (first step dropped: JIT)");
}
