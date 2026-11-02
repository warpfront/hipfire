// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! **VAE decode parity**: decode a ComfyUI `.latent` (VAE space) with hipfire's
//! own FLUX VAE decoder and dump the resulting image for a like-for-like diff
//! against ComfyUI's `VAEDecode` of the same latent.
//!
//! The golden latent already matches ComfyUI's to rel_l2 0.0935
//! (`gpu_flux_golden_latent`), so decoding the SAME latent on both sides
//! isolates the VAE decoder: any pixel difference is decoder math, not
//! denoise. This closes the last gap in the end-to-end pipeline — before it,
//! hipfire made latents and ComfyUI made pixels; now hipfire makes pixels.
//!
//! Loads only the VAE (not the transformer / T5 / CLIP), so it is fast.
//! CPU-only; no GPU.
//!
//! Build + run:
//! ```
//! cargo run --release --features lab --example flux_vae_parity \
//!   -p hipfire-arch-diffusion -- <pipe-dir> <latent> <out-dir>
//! ```
//! Writes `<out-dir>/hip_vae_out.f32` (raw `[3][h][w]` decoder output) and
//! `<out-dir>/hip_vae_out.png` (the `(x+1)/2`-mapped RGB), plus a shapes file.

use hipfire_arch_diffusion::pipeline::{postprocess_png, read_comfy_latent};
use hipfire_arch_diffusion::vae;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::PathBuf;

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

    let stages = vae::decode_stages(&weights, &data, lh, lw);
    let out = &stages.out;
    let (oh, ow) = (lh * 8, lw * 8); // 3 up-samples ⇒ 8×
    assert_eq!(
        out.len(),
        weights.config.out_channels * oh * ow,
        "decode output shape"
    );

    // Raw decoder output (channel-major [3][h][w]), for a numeric diff against
    // ComfyUI's decode of the same latent.
    std::fs::write(out_dir.join("hip_vae_out.f32"), unsafe {
        std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4)
    })
    .unwrap();
    std::fs::write(
        out_dir.join("hip_vae_shapes.json"),
        format!(
            "{{\n  \"out\": [{}, {}, {}]\n}}\n",
            weights.config.out_channels, oh, ow
        ),
    )
    .unwrap();

    // PNG for eyeballing.
    let png = postprocess_png(out, ow, oh);
    std::fs::write(out_dir.join("hip_vae_out.png"), &png).unwrap();

    let rms =
        (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / out.len() as f64).sqrt();
    let (mn, mx) = out
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
            (a.min(*v), b.max(*v))
        });
    println!(
        "decoded [1,3,{oh},{ow}]: rms {rms:.4}  min {mn:.4}  max {mx:.4}  png {} bytes",
        png.len()
    );
    println!(
        "wrote {}/hip_vae_out.f32 + hip_vae_out.png",
        out_dir.display()
    );
}
