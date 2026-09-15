// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Throwaway differential for T0 gfx11 residual F16-LUT vs decode GEMV.
//! Not a committed test — gate-first residual experiment only.
//!
//! Usage (XTX / gfx1100):
//!   ROCR_VISIBLE_DEVICES=0 ./target/release/examples/tmp_lloyd_gfx11_diff \
//!     /path/to/model.hfq
//! Optional: N=511 or N=512 (default 512).

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::lloyd_lut::{
    apply_lloyd_centering, lloyd_levels_from_sidecar, lloyd_luts_from_levels, lloyd_sidecar_name,
};
use rdna_compute::Gpu;
use std::env;
use std::path::Path;

const WEIGHT: &str = "model.language_model.layers.10.mlp.down_proj.weight";

fn prng(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt)
        .wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    (x as f32) * (1.0 / 4294967296.0)
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        num.sqrt() / den.sqrt()
    }
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("usage: tmp_lloyd_gfx11_diff <model.hfq>  (env N= batch, default 512)");
            std::process::exit(2);
        });
    let n: usize = env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("no GPU: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "tmp_lloyd_gfx11_diff: arch={} has_wmma_w32={} N={n}",
        gpu.arch,
        gpu.arch_caps.has_wmma_w32()
    );

    let hfq = HfqFile::open(Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("open {path}: {e}");
        std::process::exit(1);
    });

    let (info, mut data) = hfq.tensor_data_vec(WEIGHT).unwrap_or_else(|| {
        eprintln!("missing tensor {WEIGHT}");
        std::process::exit(1);
    });
    if info.shape.len() != 2 {
        eprintln!("expected 2D weight, got shape {:?}", info.shape);
        std::process::exit(1);
    }
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    eprintln!(
        "weight {WEIGHT}: qt={} M={m} K={k} bytes={}",
        info.quant_type,
        data.len()
    );
    if k % 256 != 0 {
        eprintln!("K={k} not divisible by 256");
        std::process::exit(1);
    }

    let qt = info.quant_type;
    let lut_f16: Option<[u32; 8]> = if qt == 52 {
        let sc_name = lloyd_sidecar_name(WEIGHT);
        let (sc_info, sc_data) = hfq.tensor_data_vec(&sc_name).unwrap_or_else(|| {
            eprintln!("missing sidecar {sc_name}");
            std::process::exit(1);
        });
        let levels = lloyd_levels_from_sidecar(sc_info.quant_type, &sc_info.shape, &sc_data)
            .unwrap_or_else(|e| {
                eprintln!("sidecar parse: {e}");
                std::process::exit(1);
            });
        let (_e4, h) = lloyd_luts_from_levels(&levels);
        apply_lloyd_centering(&mut data, m, k).unwrap_or_else(|e| {
            eprintln!("centering: {e}");
            std::process::exit(1);
        });
        eprintln!("lloyd levels={levels:?}");
        Some(h)
    } else if qt == 44 {
        None
    } else {
        eprintln!("expected qt=52 (Lloyd) or qt=44 (uniform), got {qt}");
        std::process::exit(1);
    };

    // Deterministic F32 X [N x K] and NONZERO Y0 [N x M] (residual semantics).
    let x: Vec<f32> = (0..n * k)
        .map(|i| prng(i, 0xA11C_E001) * 2.0 - 1.0)
        .collect();
    let y0: Vec<f32> = (0..n * m)
        .map(|i| prng(i, 0xB0A7_C001) * 0.25 + 0.01)
        .collect();

    let d_a = gpu.upload_raw(&data, &[data.len()]).expect("upload A");
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let d_y = gpu.upload_f32(&y0, &[n, m]).expect("upload Y0");

    if let Some(lut) = lut_f16 {
        gpu.gemm_mq4g256v2_residual_wmma_gfx11_lloyd(&d_a, &d_x, &d_y, m, k, n, lut)
            .unwrap_or_else(|e| panic!("lloyd residual launch: {e}"));
    } else {
        gpu.gemm_mq4g256v2_residual_wmma(&d_a, &d_x, &d_y, m, k, n)
            .unwrap_or_else(|e| panic!("uniform residual launch: {e}"));
    }
    gpu.hip.device_synchronize().expect("sync gemm");
    let y_gemm = gpu.download_f32(&d_y).expect("download Y");

    let batches: Vec<usize> = {
        let mut v = vec![0usize, 1];
        if n >= 2 {
            v.push(n / 2);
            if n >= 3 {
                v.push(n.saturating_sub(2));
            }
            v.push(n - 1);
        }
        v.sort_unstable();
        v.dedup();
        v.into_iter().filter(|&b| b < n).collect()
    };

    eprintln!(
        "mode={} comparing GEMM residual delta vs single-row GEMV at batches {batches:?}",
        if qt == 52 { "lloyd-lut" } else { "uniform" }
    );

    for &b in &batches {
        let x_row = &x[b * k..(b + 1) * k];
        let y0_row = &y0[b * m..(b + 1) * m];
        let y_row = &y_gemm[b * m..(b + 1) * m];
        let mut delta: Vec<f32> = y_row
            .iter()
            .zip(y0_row.iter())
            .map(|(&y, &y0)| y - y0)
            .collect();

        let d_xr = gpu.upload_f32(x_row, &[k]).expect("x row");
        let d_yr = gpu.zeros(&[m], rdna_compute::DType::F32).expect("y row");
        if let Some(lut) = lut_f16 {
            gpu.gemv_mq4g256v2_lloyd(&d_a, &d_xr, &d_yr, m, k, lut)
                .unwrap_or_else(|e| panic!("gemv lloyd b={b}: {e}"));
        } else {
            gpu.gemv_mq4g256v2(&d_a, &d_xr, &d_yr, m, k)
                .unwrap_or_else(|e| panic!("gemv uniform b={b}: {e}"));
        }
        gpu.hip.device_synchronize().expect("sync gemv");
        let y_gemv = gpu.download_f32(&d_yr).expect("download gemv");
        let r = rel_l2(&delta, &y_gemv);
        // silence unused mut if compiler is picky about delta rebuild
        let _ = &mut delta;
        eprintln!(
            "  b={b}: rel-L2(Y[b]-Y0[b], GEMV) = {r:.6e}  \
             |delta|_2={:.6e} |gemv|_2={:.6e}",
            delta.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
            y_gemv.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
        );
        let _ = gpu.free_tensor(d_xr);
        let _ = gpu.free_tensor(d_yr);
    }

    eprintln!("tmp_lloyd_gfx11_diff: done (qt={qt})");
}
