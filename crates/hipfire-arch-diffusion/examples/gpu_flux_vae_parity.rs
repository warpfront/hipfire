// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! **GPU VAE decode parity**: decode a ComfyUI `.latent` (VAE space) with
//! hipfire's GPU FLUX VAE decoder (`vae_gpu`) and dump the resulting image
//! for a like-for-like diff against ComfyUI's `VAEDecode` of the same latent.
//!
//! Companion to `flux_vae_parity` (the CPU reference). Both decode the SAME
//! latent, so any difference against ComfyUI is decoder math, and the GPU
//! output can also be diffed against the CPU output to prove the GPU kernels
//! reproduce the reference. This path is the one the full `flux_txt2img`
//! pipeline will use, since the single-threaded CPU decode of a 1024x1024
//! image takes minutes.
//!
//! Loads only the VAE (not the transformer / T5 / CLIP). Requires a GPU
//! (gpu-lock it).
//!
//! Build + run:
//! ```
//! cargo run --release --features lab --example gpu_flux_vae_parity \
//!   -p hipfire-arch-diffusion -- <pipe-dir> <latent> <out-dir>
//! ```
//! Writes `<out-dir>/gpu_vae_out.f32` (raw `[3][h][w]` decoder output),
//! `<out-dir>/gpu_vae_out.png` (the `(x+1)/2`-mapped RGB), a shapes file, and
//! the named stage dumps (`gpu_vae_conv_in.f32`, ..._mid_r0, ..._mid_attn,
//! ..._mid_r1, ..._up) for bisection against the CPU stages.

use hipfire_arch_diffusion::pipeline::{postprocess_png, read_comfy_latent};
use hipfire_arch_diffusion::vae;
use hipfire_arch_diffusion::vae_gpu;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::PathBuf;

fn write_f32(path: &std::path::Path, data: &[f32]) {
    std::fs::write(path, unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    })
    .unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pipe_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/home/user/flux-pipe".to_string()),
    );
    let latent_path = PathBuf::from(args.next().unwrap_or_else(|| {
        "crates/hipfire-arch-diffusion/tests/fixtures/flux-golden/golden.latent".to_string()
    }));
    let out_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/home/user/vaedump".to_string()),
    );
    std::fs::create_dir_all(&out_dir).unwrap();

    let vae_src =
        SafetensorsSource::open(&pipe_dir.join("vae")).unwrap_or_else(|e| panic!("open vae: {e}"));
    let weights =
        vae::VaeDecoderWeights::load(&vae_src).unwrap_or_else(|e| panic!("vae load: {e}"));
    eprintln!(
        "vae loaded: {} up blocks, scaling {:?} shift {:?}",
        weights.up_blocks.len(),
        weights.config.scaling_factor,
        weights.config.shift_factor
    );

    let raw = std::fs::read(&latent_path).unwrap_or_else(|e| panic!("read latent: {e}"));
    let (data, shape) = read_comfy_latent(&raw).unwrap_or_else(|e| panic!("latent: {e}"));
    assert_eq!(shape[0], 1, "latent batch");
    let (ch, lh, lw) = (shape[1], shape[2], shape[3]);
    assert_eq!(data.len(), ch * lh * lw, "latent element count");
    eprintln!("latent: [1,{ch},{lh},{lw}]");

    let mut gpu = Gpu::init().expect("GPU init failed");
    let gw = vae_gpu::GpuVaeDecoderWeights::from_host(&mut gpu, &weights)
        .unwrap_or_else(|e| panic!("vae gpu upload: {e}"));

    let t0 = std::time::Instant::now();
    let stages = vae_gpu::gpu_decode_stages(&mut gpu, &gw, &data, lh, lw)
        .unwrap_or_else(|e| panic!("vae gpu decode: {e}"));
    let dt = t0.elapsed();

    let (oh, ow) = (stages.out_h, stages.out_w);
    assert_eq!(
        stages.out.len(),
        weights.config.out_channels * oh * ow,
        "decode output shape"
    );
    assert_eq!((oh, ow), (lh * 8, lw * 8), "3 up-samples => 8x");

    // Raw decoder output (channel-major [3][h][w]) + named stages for
    // bisection against the CPU reference and ComfyUI.
    write_f32(&out_dir.join("gpu_vae_out.f32"), &stages.out);
    write_f32(&out_dir.join("gpu_vae_conv_in.f32"), &stages.conv_in);
    write_f32(&out_dir.join("gpu_vae_mid_r0.f32"), &stages.mid_r0);
    write_f32(&out_dir.join("gpu_vae_mid_attn.f32"), &stages.mid_attn);
    write_f32(&out_dir.join("gpu_vae_mid_r1.f32"), &stages.mid_r1);
    write_f32(&out_dir.join("gpu_vae_up.f32"), &stages.up);
    std::fs::write(
        out_dir.join("gpu_vae_shapes.json"),
        format!(
            "{{\n  \"out\": [{}, {}, {}]\n}}\n",
            weights.config.out_channels, oh, ow
        ),
    )
    .unwrap();

    // PNG for eyeballing.
    let png = postprocess_png(&stages.out, ow, oh);
    std::fs::write(out_dir.join("gpu_vae_out.png"), &png).unwrap();

    let freed = gw.free_gpu(&mut gpu);

    let out = &stages.out;
    let rms =
        (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / out.len() as f64).sqrt();
    let (mn, mx) = out
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
            (a.min(*v), b.max(*v))
        });
    println!(
        "gpu decoded [1,3,{oh},{ow}] in {dt:.2?}: rms {rms:.4}  min {mn:.4}  max {mx:.4}  png {} bytes",
        png.len()
    );
    println!("freed {freed} gpu tensors");
    println!(
        "wrote {}/gpu_vae_out.f32 + gpu_vae_out.png + stages",
        out_dir.display()
    );
}
