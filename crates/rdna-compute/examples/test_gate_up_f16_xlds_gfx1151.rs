// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1151 XLDS routes of `gemm_gate_up_q8_0_wmma` (two
//! `gemm_q8_0_wmma_gfx1151_xlds` launches) and `gemm_mw16_residual_wmma_f16x`
//! (`gemm_f16_wmma_gfx1151_xlds`) must be bitwise the base kernels
//! (`HIPFIRE_GFX1151_XLDS=0`). Random weights, odd shapes (M not a multiple of
//! 64, gate_m != up_m, partial batch tiles) plus the DSpark drafter shapes,
//! N = 1, 5, 16. Panics on any differing f32 bit pattern (HIPF-111).
//!
//! Run on gfx1151: cargo run --release -p rdna-compute --features lab --example test_gate_up_f16_xlds_gfx1151

use rdna_compute::{DType, Gpu, GpuTensor};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn unit(s: &mut u64) -> f32 {
    (xorshift(s) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// Random Q8_0 rows: fp16 scale 0x2?xx (small positive), random int8 codes.
fn q8_weights(gpu: &mut Gpu, s: &mut u64, m: usize, k: usize) -> GpuTensor {
    let mut w = vec![0u8; m * (k / 32) * 34];
    for (i, b) in w.iter_mut().enumerate() {
        *b = if i % 34 == 1 {
            0x20 + (xorshift(s) % 8) as u8
        } else {
            xorshift(s) as u8
        };
    }
    gpu.upload_raw(&w, &[w.len()]).unwrap()
}

/// fp16 bits of `v` for |v| < 1 (normal range, mantissa truncated).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let sign = ((b >> 16) & 0x8000) as u16;
    if e <= 0 {
        return sign;
    }
    sign | ((e as u16) << 10) | ((b >> 13) & 0x3ff) as u16
}

fn f16_tensor(gpu: &mut Gpu, s: &mut u64, len: usize) -> GpuTensor {
    let bytes: Vec<u8> = (0..len)
        .flat_map(|_| f16_bits(unit(s)).to_le_bytes())
        .collect();
    let mut t = gpu.upload_raw(&bytes, &[len]).unwrap();
    t.dtype = DType::F16;
    t
}

fn set_xlds(gpu: &mut Gpu, on: bool) {
    std::sync::Arc::make_mut(&mut gpu.flags).gfx1151_xlds = on;
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1151", "XLDS routes are exact-gfx1151 only");
    let mut s = 0x2545F4914F6CDD1Du64;

    for &(gate_m, up_m, k) in &[
        (1000usize, 1100usize, 256usize),
        (77, 4100, 768),
        (12288, 12288, 4096),
    ] {
        let wg = q8_weights(&mut gpu, &mut s, gate_m, k);
        let wu = q8_weights(&mut gpu, &mut s, up_m, k);
        let x: Vec<f32> = (0..16 * k).map(|_| unit(&mut s)).collect();
        let xd = gpu.upload_f32(&x, &[16 * k]).unwrap();
        let yg = gpu.zeros(&[16 * gate_m], DType::F32).unwrap();
        let yu = gpu.zeros(&[16 * up_m], DType::F32).unwrap();
        for n in [1usize, 5, 16] {
            let mut out = Vec::new();
            for on in [false, true] {
                set_xlds(&mut gpu, on);
                gpu.gemm_gate_up_q8_0_wmma(
                    &wg,
                    &wu,
                    &xd.sub_offset(0, n * k),
                    &yg,
                    &yu,
                    gate_m,
                    up_m,
                    k,
                    n,
                )
                .unwrap();
                let mut v = gpu.download_f32(&yg.sub_offset(0, n * gate_m)).unwrap();
                v.extend(gpu.download_f32(&yu.sub_offset(0, n * up_m)).unwrap());
                out.push(v);
            }
            let same = same_bits(&out[0], &out[1]);
            println!(
                "q8 gate_up gate_m={gate_m} up_m={up_m} K={k} N={n}: {}",
                if same { "bitwise==base" } else { "DIFFERS" }
            );
            assert!(
                same,
                "gate_up XLDS differs at gate_m={gate_m} up_m={up_m} K={k} N={n}"
            );
        }
        for t in [wg, wu, xd, yg, yu] {
            gpu.free_tensor(t).unwrap();
        }
    }

    for &(m, k) in &[
        (1000usize, 256usize),
        (4100, 768),
        (77, 512),
        (151936, 4096),
    ] {
        let w = f16_tensor(&mut gpu, &mut s, m * k);
        let x = f16_tensor(&mut gpu, &mut s, 16 * k);
        let y = gpu.zeros(&[16 * m], DType::F32).unwrap();
        for n in [1usize, 5, 16] {
            let mut out = Vec::new();
            for on in [false, true] {
                set_xlds(&mut gpu, on);
                // Stale Y must not leak into the XLDS arm (it overwrites; base pre-zeroes).
                gpu.hip.memset(&y.buf, 0x7f, 16 * m * 4).unwrap();
                gpu.gemm_mw16_residual_wmma_f16x(&w, &x.sub_offset(0, n * k), &y, m, k, n)
                    .unwrap();
                out.push(gpu.download_f32(&y.sub_offset(0, n * m)).unwrap());
            }
            let same = same_bits(&out[0], &out[1]);
            println!(
                "f16 M={m} K={k} N={n}: {}",
                if same { "bitwise==base" } else { "DIFFERS" }
            );
            assert!(same, "F16 XLDS differs at M={m} K={k} N={n}");
        }
        for t in [w, x, y] {
            gpu.free_tensor(t).unwrap();
        }
    }
    println!("all XLDS routes bitwise == base");
}
