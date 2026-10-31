// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity of the re-grid raw top-K with its tie-row recompute
//! (`topk_values_tie_exact_f32`) against the shipping kernel
//! (`topk_values_batched_f32_shipping`): the `[rows, K]` indices and values must
//! be BYTE-identical for every input, tie-heavy rows included (BF16-valued
//! logits tie among the leading values constantly, which is what sends the
//! DFlash2 selector's rows through the tie recompute).
//!
//! Shapes: a 259,840-entry vocabulary (a Raon-sized head), an odd vocabulary
//! that defeats the vector loads, and a vocabulary shorter than the shipping
//! block; rows 1..=15; K in {1, 2, 6, 8, 16}. Row flavours: continuous random,
//! BF16-rounded random (ties), five levels (ties everywhere), constant, fewer
//! finite values than K (-inf padding), and NaN / +inf planted.
//!
//! Prints a timing table (median of launch batches) of the shipping kernel
//! against the re-grid + tie path at 15 rows. On an arch without the re-grid
//! route (`Gpu::select_regrid_enabled`) both calls are the shipping kernel and
//! the test says so.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly (through the GPU fabric):
//!
//!   cargo test -p rdna-compute --release --test topk_tie_exact_gpu -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

fn make_row(g: &mut Lcg, vocab: usize, flavour: usize) -> Vec<f32> {
    match flavour {
        0 => (0..vocab).map(|_| g.unit() * 8.0).collect(),
        1 => (0..vocab).map(|_| bf16(g.unit() * 14.0)).collect(),
        2 => (0..vocab).map(|_| (g.next() % 5) as f32).collect(),
        3 => vec![1.5; vocab],
        4 => {
            let mut r = vec![f32::NEG_INFINITY; vocab];
            for _ in 0..5 {
                let i = g.next() as usize % vocab;
                r[i] = bf16(g.unit());
            }
            r
        }
        _ => {
            let mut r: Vec<f32> = (0..vocab).map(|_| bf16(g.unit() * 3.0)).collect();
            r[vocab / 3] = f32::NAN;
            r[vocab / 2] = f32::INFINITY;
            r[vocab / 2 + 1] = f32::INFINITY;
            r
        }
    }
}

fn run(
    gpu: &mut Gpu,
    shipping: bool,
    logits: &GpuTensor,
    idx: &GpuTensor,
    val: &GpuTensor,
    vocab: usize,
    k: usize,
    rows: usize,
) {
    if shipping {
        gpu.topk_values_batched_f32_shipping(logits, idx, val, vocab, k, rows)
            .expect("shipping top-k");
    } else {
        gpu.topk_values_batched_f32(logits, idx, val, vocab, k, rows)
            .expect("regrid top-k");
    }
}

fn bits(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    gpu.download_f32(t).expect("download")[..n]
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn regrid_topk_with_tie_recompute_is_byte_identical_to_the_shipping_kernel() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.select_regrid_enabled() {
        eprintln!(
            "NOTE: re-grid route not enabled on this arch: both arms are the shipping kernel"
        );
    }
    let mut g = Lcg(0x5eed);
    let mut compared = 0usize;
    for vocab in [259_840usize, 100_003, 200] {
        for rows in [1usize, 2, 7, 15] {
            for flavour in 0..6 {
                let mut data = Vec::with_capacity(rows * vocab);
                for r in 0..rows {
                    // Mix flavours across the rows of one call.
                    data.extend(make_row(&mut g, vocab, (flavour + r) % 6));
                }
                let logits = gpu.upload_f32(&data, &[data.len()]).expect("upload");
                for k in [1usize, 2, 6, 8, 16] {
                    let mk = |gpu: &mut Gpu| {
                        let nan = vec![f32::from_bits(0x7fc0_dead); rows * k];
                        (
                            gpu.upload_f32(&nan, &[rows * k]).expect("idx"),
                            gpu.upload_f32(&nan, &[rows * k]).expect("val"),
                        )
                    };
                    let (si, sv) = mk(&mut gpu);
                    let (ri, rv) = mk(&mut gpu);
                    run(&mut gpu, true, &logits, &si, &sv, vocab, k, rows);
                    run(&mut gpu, false, &logits, &ri, &rv, vocab, k, rows);
                    gpu.hip.device_synchronize().expect("sync");
                    assert_eq!(
                        bits(&gpu, &si, rows * k),
                        bits(&gpu, &ri, rows * k),
                        "indices differ: vocab {vocab} rows {rows} flavour {flavour} K {k}"
                    );
                    assert_eq!(
                        bits(&gpu, &sv, rows * k),
                        bits(&gpu, &rv, rows * k),
                        "values differ: vocab {vocab} rows {rows} flavour {flavour} K {k}"
                    );
                    compared += 1;
                    for t in [si, sv, ri, rv] {
                        let _ = gpu.free_tensor(t);
                    }
                }
                let _ = gpu.free_tensor(logits);
            }
        }
    }
    eprintln!("topk tie-exact: {compared} (vocab, rows, flavour, K) cases byte-identical");

    // Timing at the DFlash2 selector's shape: 15 rows, vocab 259,840, K 16, a
    // tie-heavy row among them so the recompute runs.
    let (vocab, rows, k) = (259_840usize, 15usize, 16usize);
    let mut data = Vec::with_capacity(rows * vocab);
    for r in 0..rows {
        data.extend(make_row(&mut g, vocab, if r == 3 { 1 } else { 0 }));
    }
    let logits = gpu.upload_f32(&data, &[data.len()]).expect("upload");
    let idx = gpu
        .alloc_tensor(&[rows * k], DType::F32)
        .expect("alloc idx");
    let val = gpu
        .alloc_tensor(&[rows * k], DType::F32)
        .expect("alloc val");
    for (label, shipping) in [("shipping", true), ("regrid + tie", false)] {
        for _ in 0..3 {
            run(&mut gpu, shipping, &logits, &idx, &val, vocab, k, rows);
        }
        gpu.hip.device_synchronize().expect("sync");
        let mut ms = Vec::new();
        for _ in 0..9 {
            let t = Instant::now();
            for _ in 0..10 {
                run(&mut gpu, shipping, &logits, &idx, &val, vocab, k, rows);
            }
            gpu.hip.device_synchronize().expect("sync");
            ms.push(t.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "  {label:13} {rows} rows x vocab {vocab} K {k}: median {:.3} ms",
            ms[4]
        );
    }
}
