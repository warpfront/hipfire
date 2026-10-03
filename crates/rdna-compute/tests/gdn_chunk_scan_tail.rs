// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GDN chunk scan on a partial last chunk: every output row of a T-row
//! segment must match the same row of a longer run over the same prefix
//! (the scan is causal). Rows of a partial 16-row block exercise the
//! clamped-row loads and the `tok < rows` store mask.
//!
//! Provenance: the gfx1151 twin (`gdn_chunk_scan_gfx1151`) ran its final O
//! WMMAs under that mask (clang sank them into the branch), so valid rows of
//! a partial block came out ~5e3 off or NaN while full chunks stayed exact;
//! Qwen3.5 prompts over ~64 tokens then decoded garbage on gfx1151. Shipped
//! kernel and fixed twin: max diff 2.7e-6 (last row of a tail).
//!
//! `#[ignore]`d: needs a gfx1100/gfx1151/gfx1201 GPU with a working HIP
//! toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --features deltanet --test gdn_chunk_scan_tail -- --ignored
#![cfg(feature = "deltanet")]

use rdna_compute::norm::GdnScanOut;
use rdna_compute::{DType, Gpu};

/// ~40x the measured causal-rounding noise; the broken twin missed by 5e3.
const TOL: f32 = 1e-4;
const FULL: usize = 128;

fn run(gpu: &mut Gpu, t: usize, inputs: &Inputs) -> Vec<f32> {
    let q = gpu
        .upload_f16_bits(&inputs.q[..t * 16 * 128], &[t * 16 * 128])
        .unwrap();
    let k = gpu
        .upload_f16_bits(&inputs.k[..t * 16 * 128], &[t * 16 * 128])
        .unwrap();
    let v = gpu
        .upload_f16_bits(&inputs.v[..t * 48 * 128], &[t * 48 * 128])
        .unwrap();
    let g = gpu.upload_f32(&inputs.g[..t * 48], &[t * 48]).unwrap();
    let b = gpu.upload_f32(&inputs.b[..t * 48], &[t * 48]).unwrap();
    let sq = gpu.alloc_tensor(&[inputs.sq.len()], DType::Raw).unwrap();
    gpu.hip.memcpy_htod(&sq.buf, &inputs.sq).unwrap();
    let sc = gpu.upload_f32(&inputs.sc, &[inputs.sc.len()]).unwrap();
    let ef = gpu.upload_f16_bits(&inputs.ef, &[inputs.ef.len()]).unwrap();
    let a = gpu
        .zeros(&[t.div_ceil(64) * 64 * 48 * 64 / 2], DType::F32)
        .unwrap();
    let out = gpu.zeros(&[t * 48 * 128], DType::F32).unwrap();
    gpu.gdn_chunk_scan_segment(
        &q,
        &k,
        &v,
        &a,
        &g,
        &b,
        &sq,
        &sc,
        &ef,
        &out,
        GdnScanOut::F32,
        0,
        t,
        true,
    )
    .unwrap();
    let result = gpu.download_f32(&out).unwrap();
    for tensor in [q, k, v, g, b, sq, sc, ef, a, out] {
        gpu.free_tensor(tensor).unwrap();
    }
    result
}

struct Inputs {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    g: Vec<f32>,
    b: Vec<f32>,
    sq: Vec<u8>,
    sc: Vec<f32>,
    ef: Vec<u16>,
}

#[test]
#[ignore]
fn gdn_chunk_scan_partial_chunk_rows_match_full_chunk_rows() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201") {
        eprintln!(
            "SKIP — GDN chunk scan runs on gfx1100/gfx1151/gfx1201, not {}.",
            gpu.arch
        );
        return;
    }
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let h = rdna_compute::kv_slots::half_from_f32;
    let inputs = Inputs {
        q: (0..FULL * 16 * 128).map(|_| h(rnd() * 0.088)).collect(),
        k: (0..FULL * 16 * 128).map(|_| h(rnd() * 0.088)).collect(),
        v: (0..FULL * 48 * 128).map(|_| h(rnd())).collect(),
        g: (0..FULL * 48).map(|_| -0.05 - 0.05 * rnd().abs()).collect(),
        b: (0..FULL * 48).map(|_| 0.5 + 0.4 * rnd()).collect(),
        sq: (0..48 * 128 * 128)
            .map(|_| (rnd() * 100.0) as i8 as u8)
            .collect(),
        sc: (0..48 * 128).map(|_| 0.001 + 0.001 * rnd().abs()).collect(),
        ef: (0..48 * 128 * 128).map(|_| h(rnd() * 1e-5)).collect(),
    };
    let full = run(&mut gpu, FULL, &inputs);
    // Partial last blocks of 5, 4, 8 and 12 rows in chunk 1.
    for t in [69usize, 100, 104, 124] {
        let part = run(&mut gpu, t, &inputs);
        let worst = part
            .iter()
            .zip(&full)
            .map(|(a, b)| {
                if a.is_finite() {
                    (a - b).abs()
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0f32, f32::max);
        assert!(
            worst <= TOL,
            "{}: T={t} rows differ from the T={FULL} run by {worst}",
            gpu.arch
        );
    }
}
