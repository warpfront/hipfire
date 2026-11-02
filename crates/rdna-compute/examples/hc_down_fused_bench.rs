// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipEvent microbench: the fused gfx1201 / gfx1151 Qwen4 HC-down launch
//! (`Gpu::gemm_bf16_xf16_hc_down_fused`) against the launches' worth of work it replaces
//! (`gemm_bf16_xf16_f16_wmma` = gfx1201: split-K=4 tile + reducer; gfx1151: the 160 x 64 tile
//! from 2048 rows; then `hc_activation_fused_f32` with
//! `bf16_out`). Weight 320 x 10240 BF16 (F16 shadow built by the host), X F16 [rows, 10240].
//!
//! Per row count:
//! * chain / fused: 9 batches of 40 back-to-back calls, one event pair per batch -> per-call
//!   min and median over the batches (the chain call is the GEMM + activation pair).
//! * gemm / act / fused single: 40 individually event-bracketed calls, median per call (this
//!   includes the launch gap to the event, so it overstates the back-to-back cost).
//!
//! X is reused across calls (so X <= the MALL stays cache-hot for both routes alike; at 8192
//! rows X is 168 MB, DRAM-resident on both).
//!
//! Bytes per row (analytic, DRAM-ideal; each tensor counted once per touch, the 6.5 MB weight
//! shadow is excluded):
//!   gfx1201 before = X read 10240*2 + split partials write 4*320*4 + reducer partials read 4*320*4
//!            + Y write 320*4 + activation Y read 320*4 + BF16 write 320*2 = 33920 B
//!   gfx1151 before = X read TWICE (two 160-row M tiles) 2*10240*2 + Y write 320*4
//!            + activation Y read 320*4 + BF16 write 320*2 = 43200 B
//!   after  = X read 10240*2 + BF16 write 320*2                              = 21120 B
//!
//! `ROWS=a,b,..` overrides the row list. Run:
//! cargo run --release --features lab --example hc_down_fused_bench -p rdna-compute
//! (the repo rule is one fresh process per measurement; loop in the shell).

use rdna_compute::tensor_ops::{hc_activation_fused_f32, HcActivationFused};
use rdna_compute::{DType, Gpu, GpuTensor};

const M: usize = 320;
const K: usize = 10240;
const SCALE: f32 = 0.25;
const CALLS: usize = 40;
const BATCHES: usize = 9;

const BYTES_BEFORE_1201: usize = K * 2 + 4 * M * 4 + 4 * M * 4 + M * 4 + M * 4 + M * 2;
const BYTES_BEFORE_1151: usize = 2 * K * 2 + M * 4 + M * 4 + M * 2;
const BYTES_AFTER: usize = K * 2 + M * 2;

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * scale
        })
        .collect()
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

/// Per-call microseconds of `CALLS` back-to-back calls, one event pair around the batch.
fn batch_us(gpu: &mut Gpu, f: &mut impl FnMut(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..CALLS {
        f(gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    ms * 1000.0 / CALLS as f64
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// (min, median) per-call us over `BATCHES` batches, after a warm-up.
fn batched(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> (f64, f64) {
    for _ in 0..CALLS {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut v: Vec<f64> = (0..BATCHES).map(|_| batch_us(gpu, &mut f)).collect();
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    (min, median(&mut v))
}

/// Median us of `CALLS` individually event-bracketed calls.
fn single(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..4 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut v = Vec::with_capacity(CALLS);
    for _ in 0..CALLS {
        gpu.hip.event_record(&start, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        v.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0);
    }
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    median(&mut v)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let bytes_before = if gpu.arch == "gfx1151" {
        BYTES_BEFORE_1151
    } else {
        BYTES_BEFORE_1201
    };
    println!(
        "arch {} applies(320,10240,4096)={} HIPFIRE_QWEN4_HC_DOWN_FUSED={:?}",
        gpu.arch,
        gpu.qwen4_hc_down_fused_applies(M, K, 4096),
        std::env::var("HIPFIRE_QWEN4_HC_DOWN_FUSED").unwrap_or_default()
    );
    println!(
        "bytes/row accounting (weight shadow excluded, see the header): before = {bytes_before} B; \
         after = X 10240*2 + bf16 wr 320*2 = {BYTES_AFTER} B ({:.3}x fewer)",
        bytes_before as f64 / BYTES_AFTER as f64
    );
    println!("timing: {CALLS} calls/batch x {BATCHES} batches (min/med per call); gemm/act/fused1 = median of {CALLS} single event pairs");

    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![256, 512, 1024, 1536, 2048, 4096, 8192, 16384]);
    let max_rows = *rows_list.iter().max().unwrap();

    let wb: Vec<u16> = lcg(M * K, 11, 0.05).iter().map(|&v| bf16_bits(v)).collect();
    let weight = upload_u16(&mut gpu, &wb, DType::BF16);
    let xb: Vec<u16> = lcg(max_rows * K, 23, 2.0)
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();

    println!(
        "{:>6} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7} {:>9} {:>9}",
        "rows",
        "chn_min",
        "chn_med",
        "gemm",
        "act",
        "fus_min",
        "fus_med",
        "fus1",
        "applies",
        "speedup",
        "B/row_bef",
        "B/row_aft"
    );
    for &rows in &rows_list {
        let n = rows * M;
        let x = upload_u16(&mut gpu, &xb[..rows * K], DType::F16);
        let y = gpu.zeros(&[n], DType::F32).unwrap();
        let low = gpu.zeros(&[n], DType::BF16).unwrap();
        let fused_out = gpu.zeros(&[n], DType::BF16).unwrap();
        let applies = gpu.qwen4_hc_down_fused_applies(M, K, rows);

        let (cmin, cmed) = batched(&mut gpu, |g| {
            g.gemm_bf16_xf16_f16_wmma(&weight, &x, &y, M, K, rows)
                .unwrap();
            hc_activation_fused_f32(
                g,
                &HcActivationFused {
                    values: &y,
                    scale: SCALE,
                    bf16_out: Some(&low),
                },
            )
            .unwrap();
        });
        let gemm = single(&mut gpu, |g| {
            g.gemm_bf16_xf16_f16_wmma(&weight, &x, &y, M, K, rows)
                .unwrap();
        });
        let act = single(&mut gpu, |g| {
            hc_activation_fused_f32(
                g,
                &HcActivationFused {
                    values: &y,
                    scale: SCALE,
                    bf16_out: Some(&low),
                },
            )
            .unwrap();
        });
        {
            let name = if gpu.arch == "gfx1151" {
                "qwen4_hc_down_fused_gfx1151"
            } else {
                "qwen4_hc_down_fused_b64_k32"
            };
            let fused_ok = gpu
                .gemm_bf16_xf16_hc_down_fused(&weight, &x, &fused_out, M, K, rows, SCALE)
                .is_ok();
            let (fmin, fmed, f1) = if fused_ok {
                let (a, b) = batched(&mut gpu, |g| {
                    g.gemm_bf16_xf16_hc_down_fused(&weight, &x, &fused_out, M, K, rows, SCALE)
                        .unwrap();
                });
                let s = single(&mut gpu, |g| {
                    g.gemm_bf16_xf16_hc_down_fused(&weight, &x, &fused_out, M, K, rows, SCALE)
                        .unwrap();
                });
                (a, b, s)
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };
            println!(
                "{rows:>6} {cmin:>8.1} {cmed:>8.1} {gemm:>7.1} {act:>7.1} {fmin:>8.1} {fmed:>8.1} {f1:>8.1} {applies:>8} {:>6.3}x {bytes_before:>9} {BYTES_AFTER:>9}  {name}",
                cmin / fmin
            );
        }
        for t in [x, y, low, fused_out] {
            gpu.free_tensor(t).unwrap();
        }
    }
    println!("units: us per call; speedup = chn_min / fus_min; fus1/gemm/act include event-to-launch gaps");
}
