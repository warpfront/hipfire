// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemv_q8_0_fast_r*` (the high-occupancy narrow-arm Q8_0 GEMV, K > 1536) must be
//! BIT-IDENTICAL (`f32::to_bits`) to the legacy one-wave-per-row kernel:
//!
//! 1. the decode launcher `Gpu::gemv_q8_0` (which runs the fast kernel where admitted)
//!    against `Gpu::gemv_q8_0_narrow_legacy` (the previous kernel, launched directly),
//!    on Raon-shaped weights: (m, k) = (2560, 5120), (5120, 2560), (10240, 2560),
//!    (7680, 2560), the 259,840-row lm_head (2560 in), ragged m, and K tails: 1568 / 2592
//!    / 2656 (odd block count: legacy fallback) and 2624 / 2688 / 3200 / 1600 / 10240
//!    (even block count with a ragged last chunk);
//! 2. `Gpu::gemv_q8_0_rows_exact` rows 1..=16 (and 17, 32: groups of 16) against the
//!    per-row new decode and per-row legacy decode, row by row.
//!
//! Deterministic data (fixed seed), NaN-sentinel output with a guard row past the end.
//! Prints per-case PASS/FAIL with mismatching words and max abs diff, then a timing
//! table (median of launch batches): legacy vs new decode at n = 1 per shape, and
//! `rows_exact` at rows 1, 2, 4, 8, 16 against rows x new decode and rows x legacy
//! decode. Asserts every comparison is equal.
//!
//! `Q8_FAST_SKIP_LMHEAD=1` skips the 706 MB lm_head shape.
//!
//! `#[ignore]`d: needs a HIP GPU (gfx1100 / gfx1201 take the fast kernel; elsewhere the
//! comparison is trivially legacy == legacy). Run explicitly, through the fabric lock:
//!
//!   cargo test -p rdna-compute --release --test gemv_q8_0_fast_parity -- --ignored --nocapture

use rdna_compute::gemv::q8_0_fast_admits;
use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const SENTINEL: u32 = 0x7fc0_dead;

fn next(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

/// Raw Q8_0 weight: `m` rows of `k / 32` 34-byte blocks (f16 scale + 32 int8), scale sign
/// random, exponent spread over 2^-6..2^2.
fn q8_weight(seed: &mut u64, m: usize, k: usize) -> Vec<u8> {
    let mut w = Vec::with_capacity(m * (k / 32) * 34);
    for _ in 0..m * (k / 32) {
        let sign = (next(seed) & 1) as u16;
        let exp = 9 + (next(seed) % 9) as u16;
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
    gpu.hip.device_synchronize().expect("sync");
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

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

/// `rows` launches of the decode GEMV (`legacy`: the previous narrow kernel) on row views.
#[allow(clippy::too_many_arguments)]
fn per_row(
    gpu: &mut Gpu,
    legacy: bool,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
) {
    for r in 0..rows {
        let (xr, yr) = (x.sub_offset(r * k, k), y.sub_offset(r * m, m));
        if legacy {
            gpu.gemv_q8_0_narrow_legacy(w, &xr, &yr, m, k)
        } else {
            gpu.gemv_q8_0(w, &xr, &yr, m, k)
        }
        .expect("decode gemv");
    }
}

/// Median microseconds per call of `run` (9 batches of `reps` calls).
fn time_us(gpu: &mut Gpu, reps: usize, mut run: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        run(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    median(
        (0..9)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..reps {
                    run(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                t.elapsed().as_secs_f64() * 1e6 / reps as f64
            })
            .collect(),
    )
}

#[test]
#[ignore = "needs a HIP GPU"]
fn gemv_q8_0_fast_is_bit_identical_to_the_legacy_decode_kernel() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    let mut seed = 0x0f45_0001u64;
    println!("arch {arch}");

    let skip_lm = std::env::var("Q8_FAST_SKIP_LMHEAD").as_deref() == Ok("1");
    // (m, k): Raon projections, lm_head, ragged m, K tails.
    let mut shapes: Vec<(usize, usize)> = vec![
        (2560, 5120),
        (5120, 2560),
        (10240, 2560),
        (7680, 2560),
        (4999, 2560),
        (1537, 2560),
        (777, 5120),
        (3, 2560),
        (1000, 1568),
        (1001, 2592),
        (999, 2656),
        (3001, 2624),
        (2001, 2688),
        (1000, 3200),
        (1000, 1600),
        (512, 10240),
    ];
    if !skip_lm {
        shapes.push((259_840, 2560));
    }

    let mut failures: Vec<String> = Vec::new();
    let mut cases = 0usize;
    let mut timed: Vec<(usize, usize, GpuTensor)> = Vec::new();
    for &(m, k) in &shapes {
        let big = m * k > 100_000_000;
        let w = gpu
            .upload_raw(&q8_weight(&mut seed, m, k), &[m * (k / 32) * 34])
            .expect("upload weight");
        let admitted = q8_0_fast_admits(&arch, k, w.buf.as_ptr() as usize, w.buf.as_ptr() as usize);
        println!(
            "== shape m={m} k={k}: fast kernel {}",
            if admitted {
                "admitted"
            } else {
                "not admitted (legacy)"
            }
        );

        // 1. decode launcher vs legacy kernel (a few activation rows).
        for trial in 0..if big { 1 } else { 3 } {
            let x = gpu
                .upload_f32(&activations(&mut seed, k), &[k])
                .expect("upload x");
            let y_new = sentinel(&mut gpu, m + 1);
            let y_old = sentinel(&mut gpu, m + 1);
            gpu.gemv_q8_0(&w, &x, &y_new, m, k).expect("decode (new)");
            gpu.gemv_q8_0_narrow_legacy(&w, &x, &y_old, m, k)
                .expect("decode (legacy)");
            let (got, want) = (bits(&gpu, &y_new), bits(&gpu, &y_old));
            let guard_ok = got[m] == SENTINEL;
            let holes = want[..m].iter().filter(|&&v| v == SENTINEL).count();
            let (bad, max_abs) = compare(&got[..m], &want[..m]);
            let pass = bad == 0 && guard_ok && holes == 0;
            cases += 1;
            println!(
                "{} decode new==legacy m={m} k={k} trial={trial}: mismatching words {bad}/{m}, max abs diff {max_abs:e}{}{}",
                if pass { "PASS" } else { "FAIL" },
                if guard_ok { "" } else { ", WROTE PAST THE END" },
                if holes == 0 { "" } else { ", legacy left holes" },
            );
            if !pass {
                failures.push(format!("decode m={m} k={k} trial={trial} bad={bad}"));
            }
            gpu.free_tensor(x).expect("free");
            gpu.free_tensor(y_new).expect("free");
            gpu.free_tensor(y_old).expect("free");
        }

        // 2. rows-exact r1..r16 (+17, 32) vs the per-row decode, new and legacy.
        let row_set: Vec<usize> = if big {
            vec![1, 2, 4, 8, 16]
        } else {
            (1..=16).chain([17, 32]).collect()
        };
        for rows in row_set {
            let x = gpu
                .upload_f32(&activations(&mut seed, rows * k), &[rows * k])
                .expect("upload x");
            let y = sentinel(&mut gpu, (rows + 1) * m);
            gpu.gemv_q8_0_rows_exact(&w, &x, &y, m, k, rows)
                .unwrap_or_else(|e| panic!("rows_exact m={m} k={k} rows={rows}: {e:?}"));
            let y_new = sentinel(&mut gpu, (rows + 1) * m);
            let y_old = sentinel(&mut gpu, (rows + 1) * m);
            per_row(&mut gpu, false, &w, &x, &y_new, m, k, rows);
            per_row(&mut gpu, true, &w, &x, &y_old, m, k, rows);
            let (got, new_ref, old_ref) = (bits(&gpu, &y), bits(&gpu, &y_new), bits(&gpu, &y_old));
            let n = rows * m;
            let guard_ok = got[n..].iter().all(|&v| v == SENTINEL);
            let (bad_new, max_new) = compare(&got[..n], &new_ref[..n]);
            let (bad_old, max_old) = compare(&got[..n], &old_ref[..n]);
            let pass = bad_new == 0 && bad_old == 0 && guard_ok;
            cases += 1;
            println!(
                "{} rows_exact m={m} k={k} rows={rows}: vs new decode {bad_new}/{n} (max {max_new:e}), vs legacy decode {bad_old}/{n} (max {max_old:e}){}",
                if pass { "PASS" } else { "FAIL" },
                if guard_ok { "" } else { ", WROTE PAST THE END" },
            );
            if !pass {
                failures.push(format!(
                    "rows_exact m={m} k={k} rows={rows} bad_new={bad_new} bad_old={bad_old}"
                ));
            }
            for t in [x, y, y_new, y_old] {
                gpu.free_tensor(t).expect("free");
            }
        }
        if !big && [(2560, 5120), (5120, 2560), (10240, 2560), (7680, 2560)].contains(&(m, k)) {
            timed.push((m, k, w));
        } else {
            gpu.free_tensor(w).expect("free");
        }
    }
    println!("{cases} comparisons, {} failed", failures.len());

    // Timing: legacy vs new decode at n = 1; rows_exact vs rows x decode.
    println!("== timing (us per call, median of 9 batches; K > 1536 shapes of the decode step)");
    println!("   m     k   legacy_decode  new_decode  speedup");
    for (m, k, w) in &timed {
        let (m, k) = (*m, *k);
        let x = gpu.upload_f32(&activations(&mut seed, k), &[k]).expect("x");
        let y = sentinel(&mut gpu, m);
        let t_old = time_us(&mut gpu, 50, |g| {
            g.gemv_q8_0_narrow_legacy(w, &x, &y, m, k).unwrap()
        });
        let t_new = time_us(&mut gpu, 50, |g| g.gemv_q8_0(w, &x, &y, m, k).unwrap());
        println!(
            "{m:6} {k:5} {t_old:14.2} {t_new:11.2} {:8.2}x",
            t_old / t_new
        );
    }
    println!("   m     k  rows  rows_exact  rows x new  rows x legacy  vs new  vs legacy");
    for (m, k, w) in &timed {
        let (m, k) = (*m, *k);
        for rows in [1usize, 2, 4, 8, 16] {
            let x = gpu
                .upload_f32(&activations(&mut seed, rows * k), &[rows * k])
                .expect("x");
            let y = sentinel(&mut gpu, rows * m);
            let t_rows = time_us(&mut gpu, 30, |g| {
                g.gemv_q8_0_rows_exact(w, &x, &y, m, k, rows).unwrap()
            });
            let t_new = time_us(&mut gpu, 30, |g| per_row(g, false, w, &x, &y, m, k, rows));
            let t_old = time_us(&mut gpu, 30, |g| per_row(g, true, w, &x, &y, m, k, rows));
            println!(
                "{m:6} {k:5} {rows:5} {t_rows:11.2} {t_new:11.2} {t_old:14.2} {:6.2}x {:9.2}x",
                t_new / t_rows,
                t_old / t_rows
            );
        }
    }

    assert!(
        failures.is_empty(),
        "fast narrow Q8_0 GEMV differs from the legacy decode kernel: {failures:?}"
    );
}
