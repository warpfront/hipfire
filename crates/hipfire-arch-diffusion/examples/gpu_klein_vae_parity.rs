// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! **FLUX.2 Klein VAE GPU parity**: the 32-channel decode (with
//! `post_quant_conv`) and the encoder, checked against the CPU reference on a
//! real Klein `vae/`.
//!
//! Three numbers, all `rel_l2 = ||gpu - cpu|| / ||cpu||`:
//!
//! | check | route | threshold |
//! |---|---|---|
//! | encode | `vae::encode` vs `vae_gpu::gpu_encode` | `1e-4` (both f32) |
//! | decode | `vae::decode` vs `vae_gpu::gpu_decode`, same CPU latent | `5e-3` (decoder GEMM is f16) |
//! | round trip | `decode(encode(x))` vs `x` | `0.15` (the VAE is lossy) |
//!
//! The round trip is the check that catches a *structural* mistake the two
//! parity numbers cannot: encode and decode can agree with their own CPU
//! references while the pair is wired together wrongly (a channel-order slip,
//! the logvar half read as the mean, a stride-2 tap off by one). A latent that
//! does not actually mean what the decoder expects reconstructs at
//! `rel_l2 > 0.5`, far outside the lossy-but-correct band.
//!
//! Exits 1 if any of the three misses, so it works as a gate.
//!
//! Build + run (GPU; take the gpu lock):
//! ```
//! cargo run --release -p hipfire-arch-diffusion --features lab \
//!   --example gpu_klein_vae_parity -- <pipe-dir> [size]
//! ```
//! `size` defaults to 256 and only exists to make iteration cheap: the CPU
//! reference is a single-threaded naive convolution, so a 256x256 pass is
//! minutes of CPU. The gate runs at the default.

use hipfire_arch_diffusion::vae;
use hipfire_arch_diffusion::vae_gpu;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::path::PathBuf;

/// `(rel_l2, max_abs_diff)` of `got` against reference `want`.
fn rel_l2(want: &[f32], got: &[f32]) -> (f64, f32) {
    assert_eq!(want.len(), got.len(), "length mismatch");
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max_abs = 0f32;
    for (w, g) in want.iter().zip(got) {
        let dv = (w - g) as f64;
        num += dv * dv;
        den += (*w as f64) * (*w as f64);
        max_abs = max_abs.max((w - g).abs());
    }
    (num.sqrt() / den.sqrt().max(1e-30), max_abs)
}

/// Synthetic RGB test image, channel-major `[3][h][w]` in `[-1, 1]`.
///
/// Each channel is a DIFFERENT function of position — R ramps horizontally,
/// G vertically, B on a diagonal with a checker on top — so a channel-order
/// mistake anywhere in the encode/decode pair cannot cancel out in the round
/// trip. The checker also gives the encoder some high-frequency content, so a
/// downsampler that taps the wrong pixels shows up instead of being smoothed
/// away by a pure gradient.
fn synthetic_image(h: usize, w: usize) -> Vec<f32> {
    let mut x = vec![0f32; 3 * h * w];
    for y in 0..h {
        for col in 0..w {
            let fy = y as f32 / (h - 1) as f32;
            let fx = col as f32 / (w - 1) as f32;
            let checker = if ((y / 8) + (col / 8)) % 2 == 0 {
                0.15
            } else {
                -0.15
            };
            x[y * w + col] = 2.0 * fx - 1.0;
            x[h * w + y * w + col] = 2.0 * fy - 1.0;
            x[2 * h * w + y * w + col] = (fx + fy - 1.0 + checker).clamp(-1.0, 1.0);
        }
    }
    x
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pipe_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/home/user/comfy-models/klein/FLUX.2-klein-4B".to_string()),
    );
    let size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    assert!(
        size % 16 == 0 && size >= 64,
        "size must be a multiple of 16 and >= 64"
    );

    let vae_dir = pipe_dir.join("vae");
    let src = SafetensorsSource::open(&vae_dir)
        .unwrap_or_else(|e| panic!("open {}: {e}", vae_dir.display()));
    let enc = vae::VaeEncoderWeights::load(&src).unwrap_or_else(|e| panic!("encoder load: {e}"));
    let dec = vae::VaeDecoderWeights::load(&src).unwrap_or_else(|e| panic!("decoder load: {e}"));
    let cfg = &dec.config;
    eprintln!(
        "vae: latent {} in/out {}/{} blocks {:?} down {} up {} quant_conv {} post_quant_conv {}",
        cfg.latent_channels,
        cfg.in_channels,
        cfg.out_channels,
        cfg.block_out_channels,
        enc.down_blocks.len(),
        dec.up_blocks.len(),
        enc.quant_conv.is_some(),
        dec.post_quant_conv.is_some(),
    );
    assert!(
        dec.post_quant_conv.is_some(),
        "this is the FLUX.2 parity example: the Klein decoder must carry post_quant_conv"
    );

    let (h, w) = (size, size);
    let (lh, lw) = (h / 8, w / 8);
    let x = synthetic_image(h, w);

    let mut gpu = Gpu::init().expect("GPU init failed");
    let genc = vae_gpu::GpuVaeEncoderWeights::from_host(&mut gpu, &enc)
        .unwrap_or_else(|e| panic!("encoder upload: {e}"));
    let gdec = vae_gpu::GpuVaeDecoderWeights::from_host(&mut gpu, &dec)
        .unwrap_or_else(|e| panic!("decoder upload: {e}"));

    // ── 1. encode ────────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let cpu_lat = vae::encode(&enc, &x, h, w);
    let cpu_enc_dt = t0.elapsed();
    assert_eq!(
        cpu_lat.len(),
        cfg.latent_channels * lh * lw,
        "cpu latent shape"
    );
    let t0 = std::time::Instant::now();
    let gpu_lat = vae_gpu::gpu_encode(&mut gpu, &genc, &x, h, w)
        .unwrap_or_else(|e| panic!("gpu encode: {e}"));
    let gpu_enc_dt = t0.elapsed();
    let (enc_rel, enc_max) = rel_l2(&cpu_lat, &gpu_lat);

    // ── 2. decode (both from the SAME CPU latent, so this isolates the
    //       decoder from any encode difference) ─────────────────────────────
    let t0 = std::time::Instant::now();
    let cpu_img = vae::decode(&dec, &cpu_lat, lh, lw);
    let cpu_dec_dt = t0.elapsed();
    let t0 = std::time::Instant::now();
    let (gpu_img, oh, ow) = vae_gpu::gpu_decode(&mut gpu, &gdec, &cpu_lat, lh, lw)
        .unwrap_or_else(|e| panic!("gpu decode: {e}"));
    let gpu_dec_dt = t0.elapsed();
    assert_eq!((oh, ow), (h, w), "decode output dims");
    assert_eq!(
        gpu_img.len(),
        cfg.out_channels * h * w,
        "decode output shape"
    );
    let (dec_rel, dec_max) = rel_l2(&cpu_img, &gpu_img);

    // ── 3. round trip ────────────────────────────────────────────────────
    // `cpu_lat` IS `encode(x)` and `gpu_img` IS `decode(cpu_lat)`, so this
    // reuses the passes above rather than running a third forward.
    let (rt_rel, rt_max) = rel_l2(&x, &gpu_img);

    println!(
        "image        {h}x{w}, latent [{}][{lh}][{lw}]",
        cfg.latent_channels
    );
    println!(
        "encode  rel_l2 {enc_rel:.3e}  max_abs {enc_max:.3e}  (threshold 1e-4)  cpu {cpu_enc_dt:?} gpu {gpu_enc_dt:?}"
    );
    println!(
        "decode  rel_l2 {dec_rel:.3e}  max_abs {dec_max:.3e}  (threshold 5e-3)  cpu {cpu_dec_dt:?} gpu {gpu_dec_dt:?}"
    );
    println!("rndtrip rel_l2 {rt_rel:.3e}  max_abs {rt_max:.3e}  (threshold 1.5e-1)");

    let freed = genc.free_gpu(&mut gpu) + gdec.free_gpu(&mut gpu);
    eprintln!("freed {freed} device tensors");

    let mut fail = false;
    for (what, got, limit) in [
        ("encode", enc_rel, 1e-4),
        ("decode", dec_rel, 5e-3),
        ("round trip", rt_rel, 0.15),
    ] {
        // NaN must fail, so test the negation of "passes" explicitly.
        if got.is_nan() || got >= limit {
            eprintln!("FAIL: {what} rel_l2 {got:.3e} >= {limit:.3e}");
            fail = true;
        }
    }
    if fail {
        std::process::exit(1);
    }
    println!("PASS: all three under threshold");
}
