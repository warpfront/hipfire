// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE or LICENSE-APACHE in the project root.

//! T6 oracle: `gemm_f32_batched` at DFlash draft shapes.
//!
//! The documented contract is `Y[N,M] = B[N,K] @ A[M,K]^T` with row-major
//! `y[n*M + m]`. The draft dispatch (`hipfire-runtime/src/dflash.rs`, qt16 /
//! qt2 / F16-lift arms) routes F32 weights through this entry point, and
//! every consumer reads `y` as `[batch][out]`.
//!
//! This example checks two call conventions against a CPU reference over the
//! draft shape grid:
//!   - `weight_first`: `gemm(&w, &x, &y, M, K, N)` — must match the reference;
//!   - `swapped`     : `gemm(&x, &w, &y, N, K, M)` — the transpose of the
//!     reference under the same `[N,M]` read. The swapped form is numerically
//!     identical only when N == 1, so a batch-16 draft built on it emits
//!     garbage rows while batch-1 probes look clean.
//!
//! Run on gfx1100 (X99 GPU1):
//!   ROCR_VISIBLE_DEVICES=1 cargo run --release -p rdna-compute \
//!     --example gemm_f32_draft_shapes [-- device]
//!
//! Exit code 0 iff every weight_first shape matches within tolerance; the
//! swapped column is reported for diagnosis and never gates the exit.

use rdna_compute::Gpu;

const TOL: f64 = 2e-3;

fn fill_w(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| {
            let r = ((i as u64).wrapping_mul(2654435761).wrapping_add(97) >> 8) & 0xFFFF;
            (r as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn fill_x(n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| {
            let r = ((i as u64).wrapping_mul(40503).wrapping_add(12345) >> 7) & 0xFFFF;
            (r as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn dot(x: &[f32], w: &[f32], k: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..k {
        acc += x[i] as f64 * w[i] as f64;
    }
    acc
}

fn main() {
    let device = std::env::args()
        .nth(1)
        .map(|s| s.parse::<i32>().expect("device id"))
        .unwrap_or(0);
    let mut gpu = Gpu::init_with_device(device).expect("Gpu init");

    // Draft shape grid: batch (N), output rows (M), reduction (K).
    let ns = [16usize, 24];
    let ms = [5120usize, 17408, 25600];
    let ks = [1024usize, 4096, 5120];

    let mut failures = 0usize;
    let mut shapes = 0usize;
    println!(
        "{:>4} {:>6} {:>5} | {:>10} {:>10} | {:>10}",
        "N", "M", "K", "wf_maxrel", "sw_maxrel", "sw_layout"
    );
    for &n in &ns {
        for &m in &ms {
            for &k in &ks {
                shapes += 1;
                let w = fill_w(m, k);
                let x = fill_x(n, k);
                let refv = |row: usize, out: usize| dot(&x[row * k..(row + 1) * k], &w[out * k..(out + 1) * k], k);

                let wt = gpu.upload_f32(&w, &[m * k]).expect("w");
                let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
                let yt = gpu.upload_f32(&vec![0.0f32; n * m], &[n * m]).expect("y");
                gpu.gemm_f32_batched(&wt, &xt, &yt, m, k, n)
                    .expect("weight_first launch");
                let y = gpu.download_f32(&yt).expect("y d2h");

                let mut wf_maxrel = 0.0f64;
                for row in 0..n {
                    for out in 0..m.min(4) {
                        let r = refv(row, out);
                        let g = y[row * m + out] as f64;
                        let rel = (g - r).abs() / r.abs().max(1e-6);
                        wf_maxrel = wf_maxrel.max(rel);
                    }
                }

                // Swapped convention: x as A [N,K], w as B [M,K].
                let y2t = gpu.upload_f32(&vec![0.0f32; n * m], &[n * m]).expect("y2");
                gpu.gemm_f32_batched(&xt, &wt, &y2t, n, k, m)
                    .expect("swapped launch");
                let y2 = gpu.download_f32(&y2t).expect("y2 d2h");

                // Under a [N,M] read the swapped buffer holds the transpose.
                let mut sw_maxrel = 0.0f64;
                let mut sw_mismatch = 0usize;
                let mut sw_total = 0usize;
                for row in 0..n {
                    for out in 0..m.min(4) {
                        let r = refv(row, out);
                        let g = y2[row * m + out] as f64;
                        let rel = (g - r).abs() / r.abs().max(1e-6);
                        sw_maxrel = sw_maxrel.max(rel);
                        sw_total += 1;
                        if rel > TOL {
                            sw_mismatch += 1;
                        }
                    }
                }
                let sw_layout = if sw_total == 0 {
                    "-".to_string()
                } else {
                    format!("{}/{}", sw_mismatch, sw_total)
                };
                let ok = wf_maxrel <= TOL;
                if !ok {
                    failures += 1;
                }
                println!(
                    "{:>4} {:>6} {:>5} | {:>10.2e} {:>10.2e} | {:>10}",
                    n, m, k, wf_maxrel, sw_maxrel, sw_layout
                );
                let _ = gpu.free_tensor(wt);
                let _ = gpu.free_tensor(xt);
                let _ = gpu.free_tensor(yt);
                let _ = gpu.free_tensor(y2t);
            }
        }
    }
    println!(
        "RESULT shapes={} failures={} (weight_first tol {:.0e})",
        shapes, failures, TOL
    );
    if failures > 0 {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
