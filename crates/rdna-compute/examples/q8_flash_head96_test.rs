// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Regression test for HEAD96-1: the Q8 flash tile kernel assumed head_dim is a
// multiple of 128 (32 lanes × 4 values per "half"). The LFM2-style audio
// encoder uses head_dim=96, where the tail lanes (1) read the NEXT kv head's
// Q8 block in the Q·K product and (2) wrote partials past the row end — into
// the neighboring tile's/head's {tile_max, tile_sum, out} row. (2) raced with
// the owning blocks (block execution order is undefined), so every run
// produced slightly different attention output — the ASR path fed a different
// encoder result on each identical request (root cause of the nondeterministic
// voice dictation, 2026-11-02).
//
// Two properties, head_dim=96, audio-encoder shape (8Q/4KV heads, window 32):
//   (1) CORRECT:    output matches an exact CPU softmax(QKᵀ·scale)·V reference
//                   over the sliding window (the broken kernel's cross-block
//                   score contamination fails this).
//   (2) DETERMINED: five identical launches produce bit-identical output (the
//                   broken kernel's partials race fails this).
//
// Run: cargo run --release -p rdna-compute --example q8_flash_head96_test

use rdna_compute::q8_flash_tile_size;
use rdna_compute::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 4;
const HD: usize = 96; // audio-encoder head dim — NOT a multiple of 128
const S: usize = 384; // sequence length
const W: usize = 32; // the encoder's sliding window
const BLK: usize = 34; // Q8_0 block: fp16 scale + 32 i8 codes

fn fill_kv() -> Vec<u8> {
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;
    let mut kv = vec![0u8; S * bytes_per_pos];
    for pos in 0..S {
        for blk_i in 0..(NKV * blocks_per_head) {
            let off = pos * bytes_per_pos + blk_i * BLK;
            // Scale = exactly 1.0 (f16 0x3C00, LE) so dequant is the raw i8;
            // the per-position/per-block code variety carries the signal.
            kv[off] = 0x00;
            kv[off + 1] = 0x3C;
            for j in 0..32 {
                kv[off + 2 + j] = (((pos * 3 + blk_i * 7 + j) % 11) as i32 - 5) as i8 as u8;
            }
        }
    }
    kv
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;

    // Q: [NH × HD], varied pattern.
    let q_data: Vec<f32> = (0..NH * HD)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
        .collect();
    let q = gpu.upload_f32(&q_data, &[NH * HD]).expect("q");

    // Single query at the sequence tail (pos = S-1), like the last encoder token.
    let pos_data: Vec<i32> = vec![(S - 1) as i32];
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, 4) };
    let positions = gpu.upload_raw(pos_bytes, &[1]).expect("pos");

    let tile = q8_flash_tile_size(&gpu.arch, NH, NKV, HD, S);
    let max_tiles = S.div_ceil(tile);
    let partials = gpu
        .zeros(&[NH * max_tiles * (2 + HD)], DType::F32)
        .expect("partials");

    let kv = fill_kv();
    let mut run = |gpu: &mut Gpu| -> Vec<f32> {
        let k = gpu.upload_raw(&kv, &[kv.len()]).expect("k");
        let v = gpu.upload_raw(&kv, &[kv.len()]).expect("v");
        let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
        gpu.attention_flash_q8_0_windowed(
            &q, &k, &v, &out, &positions.buf, S, NH, NKV, HD, S, &partials, W as i32,
        )
        .expect("windowed attn launch");
        gpu.download_f32(&out).expect("download")
    };

    // ── CPU reference: exact windowed softmax(QKᵀ·scale)·V per head ──
    let scale = 1.0 / (HD as f32).sqrt();
    let dequant = |pos: usize, head: usize, d: usize| -> (f32, f32) {
        // (K value, V value) at (pos, kv_head=head/2 via kv_group, dim d).
        let kv_group = NH / NKV;
        let kv_h = head / kv_group;
        let blk = d / 32;
        let off = pos * bytes_per_pos + (kv_h * blocks_per_head + blk) * BLK;
        // Scale is exactly 1.0; V cache has the identical layout/content here.
        let code = kv[off + 2 + d % 32] as i8 as f32;
        (code, code)
    };
    let mut reference = vec![0.0f32; NH * HD];
    for h in 0..NH {
        let win_lo = S - W;
        let mut scores = vec![0.0f32; W];
        for (i, pos) in (win_lo..S).enumerate() {
            let mut dot = 0.0f32;
            for d in 0..HD {
                let (k, _) = dequant(pos, h, d);
                dot += q_data[h * HD + d] * k;
            }
            scores[i] = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let weights: Vec<f32> = scores
            .iter()
            .map(|&s| {
                let e = (s - max).exp();
                sum += e;
                e
            })
            .collect();
        for d in 0..HD {
            let mut acc = 0.0f32;
            for (i, pos) in (win_lo..S).enumerate() {
                let (_, v) = dequant(pos, h, d);
                acc += weights[i] * v;
            }
            reference[h * HD + d] = acc / sum;
        }
    }

    let out0 = run(&mut gpu);
    let diff = out0
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // ── Determinism: five identical launches must agree bit-for-bit ──
    let mut max_run_diff = 0.0f32;
    for _ in 0..4 {
        let o = run(&mut gpu);
        max_run_diff = max_run_diff
            .max(o.iter().zip(&out0).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max));
    }

    println!("head_dim={HD} (audio-encoder shape) seq={S} window={W} tile={tile}");
    println!("(1) |gpu - cpu reference| = {diff:.6}  (want < 0.01)");
    println!("(2) run-to-run max diff   = {max_run_diff:.3e}  (want 0 — bit-identical)");

    let mut ok = true;
    if !(diff < 0.01) {
        eprintln!("FAIL (1): GPU output diverges from the CPU reference (cross-block score contamination)");
        ok = false;
    }
    if max_run_diff != 0.0 {
        eprintln!("FAIL (2): identical launches produced different bytes (partials race)");
        ok = false;
    }
    if ok {
        println!("PASS");
        std::process::exit(0);
    }
    std::process::exit(1);
}
