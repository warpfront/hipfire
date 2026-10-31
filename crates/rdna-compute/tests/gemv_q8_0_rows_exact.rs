// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu::gemv_q8_0_rows_exact` K = 1..=16 oracle: every output row of a
//! `rows`-row call must be BIT-IDENTICAL (`f32::to_bits`) to the rows == 1
//! decode singleton `Gpu::gemv_q8_0` run alone on that `x` row. `gemv_q8_0` is
//! the launcher `weight_gemv` -> `GemvFamily::run_auto` -> `KernelKey::GemvQ8_0`
//! reaches for a Q8F16 weight, so a pass means a verify row equals the
//! autoregressive decode row bit for bit.
//!
//! Shapes (m, k): the decode projections of the Q8F16 families
//! (5120x128, 10240x512, 7680x576, 5120x2560, 2560x5120; K <= 1536 is the wide
//! singleton, K > 1536 the narrow one), a ragged m on each arm, and K with a
//! 1- and 3-block quad tail (160, 224; 576 has a 2-block tail) so the
//! unfused-tail chains are exercised. Rows 1..=16 on every shape, plus 17, 32
//! and 40 on a few (the launcher groups by 16). Weights, scales (random sign,
//! f16 exponents spread over 2^-6..2^2) and activations (magnitudes spread over
//! 2^-2..2^3) are deterministic (fixed seed); the output starts as a NaN
//! sentinel with a guard row past the end, so an unwritten, misplaced or
//! overrun row fails.
//!
//! Prints per-(shape, rows) PASS/FAIL with the mismatch count and max abs diff,
//! then a timing table (median of launch batches): rows_exact at rows = 1, 2, 4,
//! 8, 16 against `rows` singleton launches. Asserts every comparison is equal.
//!
//! `Q8_ROWS_EXACT_TWIN_INFO=1` (gfx1100 only) additionally prints, WITHOUT
//! asserting, how the lab `gemv_q8_0_*_rows_t8_pm_gfx1100` row-tile twin compares
//! with the singleton on the same data (their GPU byte gate is not recorded).
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test gemv_q8_0_rows_exact -- --ignored --nocapture

use rdna_compute::gemv::{q8_0_rows_exact_route, Q8RowsExactRoute};
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const SENTINEL: u32 = 0x7fc0_dead;

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

/// Raw Q8_0 weight, `m` rows of `k / 32` 34-byte blocks (f16 scale + 32 int8).
fn q8_weight(seed: &mut u64, m: usize, k: usize) -> Vec<u8> {
    let mut w = Vec::with_capacity(m * (k / 32) * 34);
    for _ in 0..m * (k / 32) {
        let sign = (next(seed) & 1) as u16;
        let exp = 9 + (next(seed) % 9) as u16; // 2^-6 ..= 2^2
        let mant = (next(seed) & 0x3ff) as u16;
        w.extend_from_slice(&((sign << 15) | (exp << 10) | mant).to_le_bytes());
        w.extend((0..32).map(|_| next(seed) as u8));
    }
    w
}

fn activations(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let u = (next(seed) as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            u * (1u32 << (next(seed) % 5)) as f32
        })
        .collect()
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(SENTINEL); n], &[n])
        .expect("upload sentinel")
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// `rows` launches of the rows == 1 decode singleton over row views.
fn singleton_rows(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) {
    for r in 0..rows {
        gpu.gemv_q8_0(w, &x.sub_offset(r * k, k), &y.sub_offset(r * m, m), m, k)
            .expect("singleton gemv_q8_0");
    }
}

/// (mismatching words, max abs diff over finite pairs) of two bit vectors.
fn compare(got: &[u32], want: &[u32]) -> (usize, f32) {
    let mut bad = 0;
    let mut max_abs = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        if g != w {
            bad += 1;
        }
        let d = (f32::from_bits(g) - f32::from_bits(w)).abs();
        if d.is_finite() && d > max_abs {
            max_abs = d;
        }
    }
    (bad, max_abs)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemv_q8_0_rows_exact_is_bit_identical_to_the_decode_singleton() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    let mut seed = 0x0ae5_0001u64;
    println!("arch {arch}");

    // (m, k, extra rows beyond 1..=16)
    let shapes: [(usize, usize, &[usize]); 11] = [
        (5120, 128, &[17, 32, 40]),
        (10240, 512, &[]),
        (7680, 576, &[17]),
        (5120, 2560, &[17, 32, 40]),
        (2560, 5120, &[]),
        (4999, 128, &[]),
        (1001, 224, &[]),
        (1000, 160, &[]),
        (1537, 2560, &[17]),
        (777, 5120, &[]),
        (3, 576, &[]),
    ];

    let mut failures: Vec<String> = Vec::new();
    let mut cases = 0usize;
    let mut weights: Vec<(usize, usize, GpuTensor)> = Vec::new();
    for (m, k, extra) in shapes {
        let w = gpu
            .upload_raw(&q8_weight(&mut seed, m, k), &[m * (k / 32) * 34])
            .expect("upload weight");
        let route = q8_0_rows_exact_route(&arch, k);
        println!(
            "== shape m={m} k={k}: singleton {} / rows_exact route {route:?}",
            if k <= 1536 { "gemv_q8_0_wide" } else { "gemv_q8_0" }
        );
        for rows in (1..=16usize).chain(extra.iter().copied()) {
            let x = gpu
                .upload_f32(&activations(&mut seed, rows * k), &[rows * k])
                .expect("upload x");
            let y = sentinel(&mut gpu, (rows + 1) * m);
            gpu.gemv_q8_0_rows_exact(&w, &x, &y, m, k, rows)
                .unwrap_or_else(|e| panic!("m={m} k={k} rows={rows}: {e:?}"));
            let yref = sentinel(&mut gpu, (rows + 1) * m);
            singleton_rows(&mut gpu, &w, &x, &yref, m, k, rows);
            let got = bits(&gpu, &y);
            let want = bits(&gpu, &yref);
            let guard_ok = got[rows * m..].iter().all(|&v| v == SENTINEL);
            let holes = want[..rows * m].iter().filter(|&&v| v == SENTINEL).count();
            let (bad, max_abs) = compare(&got[..rows * m], &want[..rows * m]);
            let pass = bad == 0 && guard_ok && holes == 0;
            cases += 1;
            println!(
                "{} m={m} k={k} rows={rows}: mismatching words {bad}/{}, max abs diff {max_abs:e}{}{}",
                if pass { "PASS" } else { "FAIL" },
                rows * m,
                if guard_ok { "" } else { ", WROTE PAST THE END" },
                if holes == 0 { "" } else { ", singleton left holes" },
            );
            if !pass {
                failures.push(format!("m={m} k={k} rows={rows} bad={bad} guard_ok={guard_ok}"));
            }
        }
        weights.push((m, k, w));
    }
    println!("{cases} comparisons, {} failed", failures.len());

    // Optional, informational: the lab gfx1100 PM row-tile twin vs the singleton.
    if arch == "gfx1100" && std::env::var("Q8_ROWS_EXACT_TWIN_INFO").as_deref() == Ok("1") {
        for (m, k, w) in &weights {
            for rows in [1usize, 2, 8, 9, 16] {
                let x = gpu
                    .upload_f32(&activations(&mut seed, rows * k), &[rows * k])
                    .expect("upload x");
                let y = sentinel(&mut gpu, rows * m);
                let yref = sentinel(&mut gpu, rows * m);
                singleton_rows(&mut gpu, w, &x, &yref, *m, *k, rows);
                match gpu.lab_gemv_q8_0_rows_t8(w, &x, &y, *m, *k, rows) {
                    Ok(()) => {
                        let (bad, max_abs) = compare(&bits(&gpu, &y), &bits(&gpu, &yref));
                        println!(
                            "INFO twin m={m} k={k} rows={rows}: {} (mismatching words {bad}, max abs diff {max_abs:e})",
                            if bad == 0 { "bit-identical" } else { "DIFFERS" }
                        );
                    }
                    Err(e) => println!("INFO twin m={m} k={k} rows={rows}: refused ({e:?})"),
                }
            }
        }
    }

    // Timing: rows_exact(rows) vs rows singleton launches, median of batches.
    println!("== timing (us per call, median of 9 batches x 50 calls; no-rows-sharing baseline = rows x singleton)");
    println!("   m     k  route   rows  rows_exact  rows x singleton  speedup");
    for &(m, k, ref w) in &weights[..5] {
        let route = q8_0_rows_exact_route(&arch, k);
        for rows in [1usize, 2, 4, 8, 16] {
            let x = gpu
                .upload_f32(&activations(&mut seed, rows * k), &[rows * k])
                .expect("upload x");
            let y = sentinel(&mut gpu, rows * m);
            let time = |gpu: &mut Gpu, exact: bool| -> f64 {
                let run = |gpu: &mut Gpu| {
                    if exact {
                        gpu.gemv_q8_0_rows_exact(w, &x, &y, m, k, rows).unwrap();
                    } else {
                        singleton_rows(gpu, w, &x, &y, m, k, rows);
                    }
                };
                for _ in 0..5 {
                    run(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                let samples = (0..9)
                    .map(|_| {
                        let t = Instant::now();
                        for _ in 0..50 {
                            run(gpu);
                        }
                        gpu.hip.device_synchronize().expect("sync");
                        t.elapsed().as_secs_f64() * 1e6 / 50.0
                    })
                    .collect();
                median(samples)
            };
            let exact = time(&mut gpu, true);
            let single = time(&mut gpu, false);
            let tag = match route {
                Q8RowsExactRoute::Wide => "wide",
                Q8RowsExactRoute::Narrow => "narrow",
                Q8RowsExactRoute::PerRowSingleton => "perrow",
            };
            println!(
                "{m:6} {k:5} {tag:6} {rows:5} {exact:10.1} {single:17.1} {:8.2}x",
                single / exact
            );
        }
    }

    assert!(
        failures.is_empty(),
        "gemv_q8_0_rows_exact differs from the rows == 1 singleton: {failures:?}"
    );
}
