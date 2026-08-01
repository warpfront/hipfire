// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Isolated DeepSeek V4 score-grid attention screen at the 2048/510 shape.
//! Run separate processes with `HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE=0/1`.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const N_HEADS: usize = 64;
const HEAD_DIM: usize = 512;
const SWA_WINDOW: usize = 128;
const TOPK_WINDOW: usize = 512;
const WARMUP: usize = 50;
const ITERS: usize = 1000;

fn lcg(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit - 0.5
        })
        .collect()
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[values.len() * 4], DType::Raw)
        .expect("allocate i32 tensor");
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload i32 tensor");
    tensor
}

fn output_hash(values: &[f32]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(
        gpu.arch, "gfx1151",
        "this screen is exact-architecture only"
    );

    let q = gpu
        .upload_f32(&lcg(0x1020_3040, N_HEADS * HEAD_DIM), &[N_HEADS, HEAD_DIM])
        .expect("upload q");
    let swa_k = gpu
        .upload_f32(
            &lcg(0x2030_4050, HEAD_DIM * SWA_WINDOW),
            &[HEAD_DIM, SWA_WINDOW],
        )
        .expect("upload swa k");
    let swa_v = gpu
        .upload_f32(
            &lcg(0x3040_5060, HEAD_DIM * SWA_WINDOW),
            &[HEAD_DIM, SWA_WINDOW],
        )
        .expect("upload swa v");
    let topk_k = gpu
        .upload_f32(
            &lcg(0x4050_6070, HEAD_DIM * TOPK_WINDOW),
            &[HEAD_DIM, TOPK_WINDOW],
        )
        .expect("upload top-k k");
    let topk_v = gpu
        .upload_f32(
            &lcg(0x5060_7080, HEAD_DIM * TOPK_WINDOW),
            &[HEAD_DIM, TOPK_WINDOW],
        )
        .expect("upload top-k v");
    let sink = gpu
        .upload_f32(&lcg(0x6070_8090, N_HEADS), &[N_HEADS])
        .expect("upload sink");
    let out = gpu
        .zeros(&[N_HEADS, HEAD_DIM], DType::F32)
        .expect("allocate output");
    let n_valid_swa = upload_i32(&mut gpu, &[SWA_WINDOW as i32]);
    let n_active_topk = upload_i32(&mut gpu, &[TOPK_WINDOW as i32]);

    for _ in 0..WARMUP {
        gpu.deepseek4_attn_swa_topk_f32_buf(
            &q,
            &swa_k,
            &swa_v,
            &topk_k,
            &topk_v,
            &sink,
            &out,
            &n_valid_swa,
            &n_active_topk,
            N_HEADS as i32,
            HEAD_DIM as i32,
            SWA_WINDOW as i32,
            TOPK_WINDOW as i32,
        )
        .expect("warmup launch");
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_attn_swa_topk_f32_buf(
            &q,
            &swa_k,
            &swa_v,
            &topk_k,
            &topk_v,
            &sink,
            &out,
            &n_valid_swa,
            &n_active_topk,
            N_HEADS as i32,
            HEAD_DIM as i32,
            SWA_WINDOW as i32,
            TOPK_WINDOW as i32,
        )
        .expect("timed launch");
    }
    gpu.hip.device_synchronize().expect("timed synchronize");
    let us_per_call = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let output = gpu.download_f32(&out).expect("download output");
    let mode =
        if std::env::var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_LARGE_SERIAL").as_deref() == Ok("1") {
            "large_serial"
        } else if std::env::var("HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID_XLANE").as_deref() == Ok("1") {
            "xlane"
        } else {
            "baseline"
        };
    println!(
        "mode={mode} us_per_call={us_per_call:.3} output_hash={:016x} finite={}",
        output_hash(&output),
        output.iter().all(|value| value.is_finite())
    );
}
