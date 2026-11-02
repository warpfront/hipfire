// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU smoke for the EmbeddingGemma-2 vision leg: the whole tower forward
//! (patch embed → 2 synthetic layers → 3×3 pooler → embedder) on a hand-made
//! patch list and on a real 48×48 PNG, with synthetic weights.
//!
//! It asserts SHAPE and FINITENESS only — numerical parity against the
//! float32 transformers reference is the integration slice's job, not this
//! example's. Run with `cargo run -p hipfire-arch-embeddinggemma2
//! --features lab --example vision_smoke`.

use hipfire_arch_embeddinggemma2::image::{process_image, ProcessedImage};
use hipfire_arch_embeddinggemma2::{VisionConfig, VisionTower};
use rdna_compute::Gpu;

/// A deliberately small tower: the kernels are all shape-generic, and 2
/// layers × hidden 64 keeps the smoke to milliseconds.
fn toy_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 64,
        num_layers: 2,
        intermediate_size: 128,
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 64,
        rms_norm_eps: 1e-6,
        patch_size: 16,
        pooling_kernel_size: 3,
        position_embedding_size: 256,
        rope_theta: 100.0,
        max_soft_tokens: 70,
    }
}

fn tiny_png() -> Vec<u8> {
    // 48×48 RGB: the aspect-preserving resize upscales it (the patch budget
    // is what decides the size, not the source resolution).
    let mut img = image::RgbImage::new(48, 48);
    for y in 0..48u32 {
        for x in 0..48u32 {
            img.put_pixel(x, y, image::Rgb([(x * 5) as u8, (y * 5) as u8, 96]));
        }
    }
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encode the smoke PNG");
    out.into_inner()
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let cfg = toy_config();
    let text_hidden = 512;
    let tower = VisionTower::synthetic(&mut gpu, &cfg, text_hidden, 0x5EED)?;

    // (1) A hand-built patch list: a 6x6 grid (2x2 pooling bins → 4 tokens)
    // with a deterministic mid-grey patch pattern, no container decode.
    let (grid_h, grid_w) = (6usize, 6usize);
    let n_patches = grid_h * grid_w;
    let mut patches = vec![0.0f32; n_patches * cfg.patch_pixels()];
    for (i, v) in patches.iter_mut().enumerate() {
        *v = ((i % 251) as f32) / 251.0;
    }
    let mut positions = Vec::with_capacity(n_patches * 2);
    for py in 0..grid_h {
        for px in 0..grid_w {
            positions.push(px as i32);
            positions.push(py as i32);
        }
    }
    let hand = ProcessedImage {
        patches,
        positions,
        grid_h,
        grid_w,
        n_patches,
        n_tokens: n_patches / (cfg.pooling_kernel_size * cfg.pooling_kernel_size),
    };
    let t0 = std::time::Instant::now();
    let rows = tower.encode_image(&mut gpu, &hand)?;
    let finite = rows.iter().all(|v| v.is_finite());
    println!(
        "hand-built {}x{} patch grid -> {} rows x {text_hidden} in {:.1} ms (finite: {finite})",
        grid_h,
        grid_w,
        rows.len() / text_hidden,
        t0.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(rows.len(), hand.n_tokens * text_hidden);
    assert!(finite, "the tower produced a non-finite row");

    // (2) A real PNG through the processor and the tower.
    let processed = process_image(&tiny_png(), &cfg)?;
    let t1 = std::time::Instant::now();
    let rows = tower.encode_image(&mut gpu, &processed)?;
    let finite = rows.iter().all(|v| v.is_finite());
    println!(
        "48x48 PNG -> {}x{} patch grid, {} soft tokens, {} rows x {text_hidden} in {:.1} ms \
         (finite: {finite})",
        processed.grid_h,
        processed.grid_w,
        processed.n_tokens,
        rows.len() / text_hidden,
        t1.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(rows.len(), processed.n_tokens * text_hidden);
    assert!(finite, "the tower produced a non-finite row");
    assert!(
        rows.iter().any(|v| *v != 0.0),
        "the tower produced all zeros"
    );

    tower.free_gpu(&mut gpu);
    println!("vision smoke ok");
    Ok(())
}
