// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-reference parity test for the FLUX modulation kernels
//! (modulate_f32 + gated_add_f32). Verifies the GPU broadcast
//! vectors match the naive CPU formulas:
//!
//!   modulate_f32: out[r,i] = x[r,i]*(1 + scale[i]) + shift[i]
//!   gated_add_f32: acc[r,i] += gate[i]*x[r,i]
//!
//! Cases exercise row counts from 1 (the conditioning vector) up to the
//! image-token working set, and the real FLUX hidden width 3072.
//!
//! Correctness gate: max |kernel - cpu_ref| relative to output magnitude,
//! tolerated at 1e-6 (elementwise, should be near-bit-exact). Any subtest
//! over tolerance → exit 1.
//!
//! Build: `cargo run --release --example test_modulate_f32 -p rdna-compute`

use rdna_compute::{DType, Gpu};

const TOL: f32 = 1e-6;

fn run_modulate(gpu: &mut Gpu, n_rows: usize, d: usize, label: &str) -> usize {
    let x: Vec<f32> = (0..n_rows * d)
        .map(|i| (((i * 7919) % 509) as f32 - 254.0) * 0.01)
        .collect();
    let shift: Vec<f32> = (0..d)
        .map(|i| (((i * 2027) % 97) as f32 - 48.0) * 0.01)
        .collect();
    let scale: Vec<f32> = (0..d)
        .map(|i| (((i * 7013) % 103) as f32 - 51.0) * 0.02)
        .collect();
    let want: Vec<f32> = (0..n_rows * d)
        .map(|idx| x[idx] * (1.0 + scale[idx % d]) + shift[idx % d])
        .collect();

    let g_x = gpu.upload_f32(&x, &[n_rows, d]).unwrap();
    let g_sh = gpu.upload_f32(&shift, &[d]).unwrap();
    let g_sc = gpu.upload_f32(&scale, &[d]).unwrap();
    let g_out = gpu.zeros(&[n_rows, d], DType::F32).unwrap();
    gpu.modulate_f32(&g_x, &g_sh, &g_sc, &g_out, n_rows, d)
        .unwrap();
    let got = gpu.download_f32(&g_out).unwrap();

    let max_want = want.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
    let max_err = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let rel = max_err / max_want;
    if rel > TOL {
        eprintln!("{label} (modulate): FAIL rel={rel:.3e}");
        1
    } else {
        println!("{label} (modulate): ok {n_rows}x{d} rel={rel:.3e}");
        0
    }
}

fn run_gated_add(gpu: &mut Gpu, n_rows: usize, d: usize, label: &str) -> usize {
    let acc0: Vec<f32> = (0..n_rows * d)
        .map(|i| (((i * 1009) % 211) as f32 - 105.0) * 0.01)
        .collect();
    let gate: Vec<f32> = (0..d)
        .map(|i| (((i * 7013) % 103) as f32 - 51.0) * 0.02)
        .collect();
    let x: Vec<f32> = (0..n_rows * d)
        .map(|i| (((i * 4001) % 89) as f32 - 44.0) * 0.03)
        .collect();
    let want: Vec<f32> = (0..n_rows * d)
        .map(|idx| acc0[idx] + gate[idx % d] * x[idx])
        .collect();

    let g_acc = gpu.upload_f32(&acc0, &[n_rows, d]).unwrap();
    let g_gate = gpu.upload_f32(&gate, &[d]).unwrap();
    let g_x = gpu.upload_f32(&x, &[n_rows, d]).unwrap();
    gpu.gated_add_f32(&g_acc, &g_gate, &g_x, n_rows, d).unwrap();
    let got = gpu.download_f32(&g_acc).unwrap();

    let max_want = want.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
    let max_err = got
        .iter()
        .zip(want.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let rel = max_err / max_want;
    if rel > TOL {
        eprintln!("{label} (gated_add): FAIL rel={rel:.3e}");
        1
    } else {
        println!("{label} (gated_add): ok {n_rows}x{d} rel={rel:.3e}");
        0
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut fails = 0;

    // Conditioning vector modulation (n_rows = 1).
    fails += run_modulate(&mut gpu, 1, 3072, "cond-vec");
    // Real FLUX image-token working set.
    fails += run_modulate(&mut gpu, 1024, 3072, "img-tokens");
    // Small lab shape.
    fails += run_modulate(&mut gpu, 48, 384, "lab");
    // In-place (out aliases x) modulation soundness, small shape.
    {
        let d = 96usize;
        let n_rows = 16usize;
        let x: Vec<f32> = (0..n_rows * d)
            .map(|i| (((i * 1583) % 127) as f32 - 63.0) * 0.03)
            .collect();
        let shift: Vec<f32> = (0..d)
            .map(|i| (((i * 2027) % 97) as f32 - 48.0) * 0.02)
            .collect();
        let scale: Vec<f32> = (0..d)
            .map(|i| (((i * 7013) % 103) as f32 - 51.0) * 0.01)
            .collect();
        let want: Vec<f32> = (0..n_rows * d)
            .map(|idx| x[idx] * (1.0 + scale[idx % d]) + shift[idx % d])
            .collect();
        let g_x = gpu.upload_f32(&x, &[n_rows, d]).unwrap();
        let g_sh = gpu.upload_f32(&shift, &[d]).unwrap();
        let g_sc = gpu.upload_f32(&scale, &[d]).unwrap();
        gpu.modulate_f32(&g_x, &g_sh, &g_sc, &g_x, n_rows, d)
            .unwrap(); // in-place
        let got = gpu.download_f32(&g_x).unwrap();
        let max_want = want.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        let max_err = got
            .iter()
            .zip(want.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let rel = max_err / max_want;
        if rel > TOL {
            eprintln!("in-place (modulate): FAIL rel={rel:.3e}");
            fails += 1;
        } else {
            println!("in-place (modulate): ok {n_rows}x{d} rel={rel:.3e}");
        }
    }

    // Gated residual, double-block g1/g2 sizes.
    fails += run_gated_add(&mut gpu, 512, 3072, "gated-512");
    fails += run_gated_add(&mut gpu, 1, 3072, "gated-cond");
    fails += run_gated_add(&mut gpu, 48, 384, "gated-lab");

    if fails > 0 {
        eprintln!("FAIL: {fails} subtests failed");
        std::process::exit(1);
    }
    println!("PASS: modulate_f32 + gated_add_f32 parity vs CPU reference");
}
