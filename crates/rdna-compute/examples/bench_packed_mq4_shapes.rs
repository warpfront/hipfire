// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 HUSRCF
//! Synthetic actual-shape MQ4 FFN comparison, including activation packing.
//! Attribution only; production throughput comes from the daemon ABBA.
use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    // Optional numerical-only mode. Refuse existing directories and preserve
    // GPU-produced activation bytes for an independent CPU oracle.
    let dump = std::env::var_os("MQ4_ORACLE_DIR").map(std::path::PathBuf::from);
    if let Some(path) = &dump { std::fs::create_dir(path).unwrap(); }
    let mut gpu = Gpu::init_with_device(0).expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100");
    for (m, k, add) in [(17408usize, 5120usize, false), (5120, 17408, true)] {
        let n = 512;
        let mut w = vec![0u8; m * (k / 256) * 136];
        for (i, group) in w.chunks_exact_mut(136).enumerate() {
            let scale = (1 + i % 7) as f32 / 128.;
            group[..4].copy_from_slice(&scale.to_le_bytes());
            group[4..8].copy_from_slice(&(-7.5 * scale).to_le_bytes());
            for (j, q) in group[8..].iter_mut().enumerate() {
                *q = ((i * 13 + j * 7 + j / 3) & 255) as u8;
            }
        }
        let x: Vec<f32> = (0..n * k)
            .map(|i| (((i * 17 + i / k * 31) % 257) as f32 - 128.) / 127.)
            .collect();
        let dw = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let dx = gpu.upload_f32(&x, &[n, k]).unwrap();
        let dy = gpu.zeros(&[n, m], DType::F32).unwrap();
        let case_dir = dump.as_ref().map(|root| {
            let path = root.join(format!("m{m}-k{k}-n{n}"));
            std::fs::create_dir(&path).unwrap();
            std::fs::write(path.join("weights.bin"), &w).unwrap();
            write_f32(&path.join("input.bin"), &x);
            path
        });
        let mut reference = Vec::new();
        for packed in [false, true] {
            // Zero once for numerical comparison; timing uses the same ADD
            // epilogue repeatedly on both arms and excludes zeroing overhead.
            gpu.hip.memset(&dy.buf, 0, n * m * 4).unwrap();
            let mut launch = |gpu: &mut Gpu| {
                if packed {
                    gpu.gemm_mq4_packed(&dw, &dx, &dy, m, k, n, add)
                } else if add {
                    gpu.gemm_hfq4g256_residual(&dw, &dx, &dy, m, k, n)
                } else {
                    gpu.gemm_hfq4g256_mmq_set(&dw, &dx, &dy, m, k, n)
                }
            };
            launch(&mut gpu).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let result = gpu.download_f32(&dy).unwrap();
            assert!(result.iter().all(|v| v.is_finite()));
            if let Some(path) = &case_dir {
                let name = if packed { "packed" } else { "native" };
                write_f32(&path.join(format!("{name}-output.bin")), &result);
                let buf = gpu.scratch.q8_1_mmq_x_scratch.as_ref().unwrap();
                let mut bytes = vec![0u8; (k / 128) * n * 144];
                gpu.hip.memcpy_dtoh(&mut bytes, buf).unwrap();
                std::fs::write(path.join(format!("{name}-activation.bin")), bytes).unwrap();
            }
            if !packed {
                reference = result;
            } else {
                let mut sq = 0f64;
                let mut ref_sq = 0f64;
                let mut max_abs = 0f32;
                for (a, b) in result.iter().zip(&reference) {
                    let delta = a - b;
                    max_abs = max_abs.max(delta.abs());
                    sq += (delta as f64).powi(2);
                    ref_sq += (*b as f64).powi(2);
                }
                println!("NUMERIC m={m} k={k} n={n} add={add} max_abs={max_abs} relative_l2={}", (sq / ref_sq).sqrt());
            }
            if dump.is_some() { continue; }
            for _ in 0..10 { launch(&mut gpu).unwrap(); }
            gpu.hip.device_synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..50 { launch(&mut gpu).unwrap(); }
            gpu.hip.device_synchronize().unwrap();
            println!("TIMING m={m} k={k} n={n} add={add} packed={packed} us={:.3}", start.elapsed().as_secs_f64() * 1e6 / 50.);
        }
        gpu.free_tensor(dw).unwrap();
        gpu.free_tensor(dx).unwrap();
        gpu.free_tensor(dy).unwrap();
    }
}

fn write_f32(path: &std::path::Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}
