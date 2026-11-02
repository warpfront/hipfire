// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipEvent microbench of the Qwen4 shared-expert INT8 route (`HIPFIRE_QWEN4_SHARED_INT8=1`,
//! `crates/rdna-compute/src/shared_i8.rs`) against the F16 WMMA route it replaces, at the
//! Flash-Next shape (hidden 2560, intermediate 640, BF16 weights), per layer-pass of `ROWS`
//! rows (default 16384, 4096).
//!
//! F16 chain (what the gfx1201 prefill runs for the shared expert, router/selector excluded
//! from both sides): gate/up GEMMs off the already-converted F16 X (the fused MoE front's
//! output: `gemm_bf16_xf32_f16_wmma_qwen4_groups_x16` with `x16`), `shared_expert_activation_
//! bf16_f32`, then the down GEMM off the F32 hidden (which converts it to F16 first).
//! INT8 chain: `qwen4_shared_i8_gate_up` (A8 producer of X + GEMM + SwiGLU epilogue),
//! `qwen4_shared_i8_hidden`, `qwen4_shared_i8_down`.
//!
//! Per row count: 9 batches of 20 back-to-back calls, one event pair per batch; per-call min
//! and median over the batches, for the whole chain and each stage. Analytic DRAM-ideal bytes
//! per pass are printed with the times. One fresh process per measurement.
//!
//! Run: HIPFIRE_QWEN4_SHARED_INT8=1 cargo run --release --features lab --example shared_i8_bench -p rdna-compute

use rdna_compute::shared_i8::SharedI8Weights;
use rdna_compute::{DType, Gpu, GpuTensor};

const I: usize = 640;
const H: usize = 2560;
const CALLS: usize = 20;
const BATCHES: usize = 9;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|h| h.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[v.len()]).expect("upload");
    t.dtype = dtype;
    t
}

/// (min, median) per-call microseconds over `BATCHES` batches of `CALLS` calls.
fn time_us(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> (f64, f64) {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut v = Vec::new();
    for _ in 0..BATCHES {
        let start = gpu.hip.event_create().unwrap();
        let stop = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&start, None).unwrap();
        for _ in 0..CALLS {
            f(gpu);
        }
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        v.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0 / CALLS as f64);
        gpu.hip.event_destroy(start).unwrap();
        gpu.hip.event_destroy(stop).unwrap();
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.to_string();
    let switch = std::env::var("HIPFIRE_QWEN4_SHARED_INT8").unwrap_or_default();
    println!(
        "arch {arch} HIPFIRE_QWEN4_SHARED_INT8={switch:?} calls/batch {CALLS} batches {BATCHES}"
    );
    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![16384, 4096]);
    let max_rows = *rows_list.iter().max().unwrap();

    let mut rng = Rng(0xbe9c_4455);
    let mut gen =
        |n: usize| -> Vec<u16> { (0..n).map(|_| bf16_bits(rng.gauss() * 0.03)).collect() };
    let gate = upload_u16(&mut gpu, &gen(I * H), DType::BF16);
    let up = upload_u16(&mut gpu, &gen(I * H), DType::BF16);
    let down = upload_u16(&mut gpu, &gen(H * I), DType::BF16);
    let w = SharedI8Weights {
        gate: &gate,
        up: &up,
        down: &down,
        intermediate: I,
        hidden: H,
        down_rows: H,
    };
    if !gpu.qwen4_shared_i8_applies(&w, 4096) {
        eprintln!(
            "the INT8 route does not apply (HIPFIRE_QWEN4_SHARED_INT8={switch:?}, arch {arch})"
        );
        std::process::exit(2);
    }
    let mut x_rng = Rng(0x77);
    let x_host: Vec<f32> = (0..max_rows * H).map(|_| x_rng.gauss()).collect();
    let x_all = gpu.upload_f32(&x_host, &[max_rows * H]).unwrap();
    let x16_host: Vec<u16> = x_host
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let x16_all = upload_u16(&mut gpu, &x16_host, DType::F16);
    drop(x_host);
    drop(x16_host);

    println!("units: us per layer-pass call (min / median)");
    for &n in &rows_list {
        let x = x_all.sub_offset(0, n * H);
        let x16 = x16_all.sub_offset(0, n * H);
        let g = gpu.zeros(&[n * I], DType::F32).unwrap();
        let u = gpu.zeros(&[n * I], DType::F32).unwrap();
        let hid_f32 = gpu.zeros(&[n * I], DType::F32).unwrap();
        let hid_bf = gpu.zeros(&[n * I], DType::F32).unwrap();
        let blocks = gpu.zeros(&[n * I], DType::F32).unwrap();
        let sel = gpu.zeros(&[n], DType::F32).unwrap();
        let out = gpu.zeros(&[n * H], DType::F32).unwrap();
        // The shadows are built on the first call.
        gpu.qwen4_shared_i8_gate_up(&w, &x, &hid_bf, n).unwrap();

        let f16_gu = time_us(&mut gpu, |gpu| {
            assert!(gpu
                .gemm_bf16_xf32_f16_wmma_qwen4_groups_x16(
                    &[&[(&gate, &g, I), (&up, &u, I)]],
                    &x,
                    Some(&x16),
                    H,
                    n
                )
                .unwrap());
        });
        let f16_act = time_us(&mut gpu, |gpu| {
            gpu.shared_expert_activation_bf16_f32(&g, &u, &hid_f32, &sel, n)
                .unwrap();
        });
        let f16_down = time_us(&mut gpu, |gpu| {
            assert!(gpu
                .gemm_bf16_xf32_f16_wmma_qwen4(&[(&down, &out, H)], &hid_f32, I, n)
                .unwrap());
        });
        let f16_chain = time_us(&mut gpu, |gpu| {
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_groups_x16(
                &[&[(&gate, &g, I), (&up, &u, I)]],
                &x,
                Some(&x16),
                H,
                n,
            )
            .unwrap();
            gpu.shared_expert_activation_bf16_f32(&g, &u, &hid_f32, &sel, n)
                .unwrap();
            gpu.gemm_bf16_xf32_f16_wmma_qwen4(&[(&down, &out, H)], &hid_f32, I, n)
                .unwrap();
        });

        let i8_prep = time_us(&mut gpu, |gpu| {
            gpu.mq6_i8_prepare_x(&x, H, n).unwrap();
        });
        let i8_gu = time_us(&mut gpu, |gpu| {
            gpu.qwen4_shared_i8_gate_up(&w, &x, &hid_bf, n).unwrap();
        });
        let i8_hidden = time_us(&mut gpu, |gpu| {
            gpu.qwen4_shared_i8_hidden(&w, &hid_bf, &blocks, &sel, n)
                .unwrap();
        });
        let i8_down = time_us(&mut gpu, |gpu| {
            gpu.qwen4_shared_i8_down(&w, &blocks, &out, n).unwrap();
        });
        let i8_chain = time_us(&mut gpu, |gpu| {
            gpu.qwen4_shared_i8_gate_up(&w, &x, &hid_bf, n).unwrap();
            gpu.qwen4_shared_i8_hidden(&w, &hid_bf, &blocks, &sel, n)
                .unwrap();
            gpu.qwen4_shared_i8_down(&w, &blocks, &out, n).unwrap();
        });

        // The fused MoE front: its F16-X + IU4 producer with and without the A8 third output.
        let front = if gpu.arch == "gfx1201" && gpu.qwen4_moe_front_fuse_applies() {
            let f0 = time_us(&mut gpu, |gpu| {
                gpu.qwen4_moe_front_producer(&x, H, n).unwrap();
            });
            let f1 = time_us(&mut gpu, |gpu| {
                gpu.qwen4_moe_front_producer_i8(&x, H, n).unwrap();
            });
            Some((f0, f1))
        } else {
            None
        };

        // Analytic DRAM-ideal bytes per pass (weights excluded: 3.3 + 1.6 MB int8 vs 9.8 MB
        // F16 shadows, identical order on both sides and L2-resident).
        let f16_bytes = n * H * 2 // x16 read (gate/up)
            + 2 * n * I * 4 // g, u written
            + 2 * n * I * 4 + n * I * 4 // activation reads g, u, writes hidden
            + n * I * 4 + n * I * 2 // down: hidden F32 read, F16 convert write
            + n * I * 2 // F16 hidden read by the GEMM
            + n * H * 4; // out
        let i8_bytes = n * H * 4 + n * (H / 128) * 136 // x read, A8 write
            + n * (H / 128) * 136 // A8 read by the GEMM
            + n * I * 2 // BF16 hidden write
            + n * I * 2 + n * (I / 128) * 136 // hidden read, blocks write
            + n * (I / 128) * 136 // blocks read by down
            + n * H * 4; // out
        let ops_f16 = 2.0 * n as f64 * (2.0 * I as f64 * H as f64 + H as f64 * I as f64);
        println!("rows {n}  FLOP/pass {:.1} G", ops_f16 / 1e9);
        println!(
            "  f16 chain {:8.1} / {:8.1}   (gate_up {:7.1}/{:7.1}  act {:6.1}/{:6.1}  down+cvt {:7.1}/{:7.1})",
            f16_chain.0, f16_chain.1, f16_gu.0, f16_gu.1, f16_act.0, f16_act.1, f16_down.0, f16_down.1
        );
        println!(
            "  int8 chain {:7.1} / {:8.1}   (gate_up {:7.1}/{:7.1} [of which A8 producer {:6.1}/{:6.1}]  hidden {:6.1}/{:6.1}  down {:7.1}/{:7.1})",
            i8_chain.0, i8_chain.1, i8_gu.0, i8_gu.1, i8_prep.0, i8_prep.1, i8_hidden.0, i8_hidden.1, i8_down.0, i8_down.1
        );
        if let Some((f0, f1)) = front {
            let fused_chain = i8_chain.0 - i8_prep.0 + (f1.0 - f0.0);
            println!(
                "  front producer f16out {:7.1}/{:7.1}  f16out+A8 {:7.1}/{:7.1}  (+{:.1} us); int8 chain with the A8 riding the front: {:.1} us (min), speedup {:.3}x, saved {:.1} us/layer = {:.1} ms/48 layers",
                f0.0, f0.1, f1.0, f1.1, f1.0 - f0.0, fused_chain,
                f16_chain.0 / fused_chain,
                f16_chain.0 - fused_chain,
                (f16_chain.0 - fused_chain) * 48.0 / 1000.0,
            );
        }
        println!(
            "  speedup (min) {:.3}x  saved {:.1} us/layer = {:.1} ms/48 layers   bytes f16 {:.1} MB int8 {:.1} MB (-{:.1} MB/layer)",
            f16_chain.0 / i8_chain.0,
            f16_chain.0 - i8_chain.0,
            (f16_chain.0 - i8_chain.0) * 48.0 / 1000.0,
            f16_bytes as f64 / 1e6,
            i8_bytes as f64 / 1e6,
            (f16_bytes - i8_bytes) as f64 / 1e6,
        );
    }
}
