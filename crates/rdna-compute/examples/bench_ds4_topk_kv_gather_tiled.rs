// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cold-set DeepSeek V4 top-k gather screen at the 2048/510 route shape.
//! Run separate processes with `HIPFIRE_DS4_GATHER_TILED=0/1`.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const LAYERS: usize = 21;
const N_COMPRESSED: usize = 640;
const K: usize = 512;
const HEAD_DIM: usize = 512;
const OUT_STRIDE: usize = 512;
const WARMUP: usize = 20;
const ITERS: usize = 250;

const BASELINE_NAME: &str = "deepseek4_topk_kv_gather_f32_buf";
const BASELINE_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_buf.hip");
const TILED_NAME: &str = "deepseek4_topk_kv_gather_tiled_f32_buf";
const TILED_SRC: &str =
    include_str!("../../../kernels/src/deepseek4_topk_kv_gather_tiled.gfx1151.hip");

fn lcg(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(0x3f00_0000 | (state >> 9)) - 0.75
        })
        .collect()
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    gpu.upload_raw(bytes, &[bytes.len()])
        .expect("upload i32 tensor")
}

fn output_hash(values: &[f32]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn launch(
    gpu: &mut Gpu,
    symbol: &str,
    tiled: bool,
    kv_cache: &GpuTensor,
    topk_idx: &GpuTensor,
    out: &GpuTensor,
    k_buf: &GpuTensor,
    n_compressed_buf: &GpuTensor,
) {
    let mut args = KernargBlob::new();
    args.push_ptr(kv_cache.buf.as_ptr());
    args.push_ptr(topk_idx.buf.as_ptr());
    args.push_ptr(out.buf.as_ptr());
    args.push_ptr(k_buf.buf.as_ptr());
    args.push_ptr(n_compressed_buf.buf.as_ptr());
    args.push_i32(HEAD_DIM as i32);
    args.push_i32(OUT_STRIDE as i32);
    args.push_i32(0);
    args.push_f32(1.0);
    let grid = if tiled {
        [(K / 32) as u32, (HEAD_DIM / 32) as u32, 1]
    } else {
        [K as u32, 1, 1]
    };
    let block = if tiled { [256, 1, 1] } else { [512, 1, 1] };
    gpu.launch_kernel_blob(symbol, grid, block, 0, args.as_mut_slice())
        .expect("launch gather");
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1151", "this screen is exact-architecture only");
    let tiled = std::env::var("HIPFIRE_DS4_GATHER_TILED").as_deref() == Ok("1");
    let (logical, source, symbol) = if tiled {
        ("bench_ds4_gather_tiled", TILED_SRC, TILED_NAME)
    } else {
        ("bench_ds4_gather_baseline", BASELINE_SRC, BASELINE_NAME)
    };
    gpu.ensure_kernel_public(logical, source, symbol)
        .expect("compile gather");

    let topk: Vec<i32> = (0..K)
        .map(|k| ((k * 73 + 17) % N_COMPRESSED) as i32)
        .collect();
    let topk_idx = upload_i32(&mut gpu, &topk);
    let k_buf = upload_i32(&mut gpu, &[K as i32]);
    let n_compressed_buf = upload_i32(&mut gpu, &[N_COMPRESSED as i32]);
    let mut caches = Vec::with_capacity(LAYERS);
    let mut outputs = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        caches.push(
            gpu.upload_f32(
                &lcg(0x1234_5678 ^ layer as u32, N_COMPRESSED * HEAD_DIM),
                &[N_COMPRESSED, HEAD_DIM],
            )
            .expect("upload cache"),
        );
        outputs.push(
            gpu.zeros(&[HEAD_DIM, OUT_STRIDE], DType::F32)
                .expect("allocate output"),
        );
    }

    for _ in 0..WARMUP {
        for layer in 0..LAYERS {
            launch(
                &mut gpu,
                symbol,
                tiled,
                &caches[layer],
                &topk_idx,
                &outputs[layer],
                &k_buf,
                &n_compressed_buf,
            );
        }
    }
    gpu.hip.device_synchronize().expect("warmup synchronize");

    let started = Instant::now();
    for _ in 0..ITERS {
        for layer in 0..LAYERS {
            launch(
                &mut gpu,
                symbol,
                tiled,
                &caches[layer],
                &topk_idx,
                &outputs[layer],
                &k_buf,
                &n_compressed_buf,
            );
        }
    }
    gpu.hip.device_synchronize().expect("timed synchronize");
    let calls = (ITERS * LAYERS) as f64;
    let us_per_call = started.elapsed().as_secs_f64() * 1e6 / calls;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for output in &outputs {
        hash ^= output_hash(&gpu.download_f32(output).expect("download output"));
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    println!(
        "mode={} us_per_call={us_per_call:.3} output_hash={hash:016x}",
        if tiled { "tiled" } else { "baseline" }
    );
}
