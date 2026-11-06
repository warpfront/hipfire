// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node bench + accuracy for the gfx12 W8A8 Q8_0 GEMM (`gemm_q8_0_a8_batched`) and its FP8
//! twin (`gemm_q8_0_fp8_batched`: int8 weights as E4M3 x per-32 E4M3 activations)
//! against the F16-activation incumbent (`gemm_q8_0_batched_chunked`).
//!
//! Per `(M, K)` shape and row count `N`:
//!   * time: median over `ITERS` single calls (GPU events around the whole
//!     launcher call, so the incumbent includes its F32 -> F16 convert and the
//!     candidate its activation quantizer), plus the GEMM-only time of every
//!     tile variant and the quantizer alone;
//!   * accuracy on BF16-valued activations with realistic outliers (persistent
//!     massive channels and a few hot rows): new vs incumbent over the whole
//!     output (max / mean abs, relative = sum|d| / sum|ref|), and both vs an F64
//!     CPU reference on a random subset of outputs.
//!
//! Usage: bench_q8_0_a8_gfx12 [--shapes all|big] [--rows 512,4096] [--iters 21] [--out file.md]
//! Needs a gfx12 GPU (run through fabric-run.sh).

use hip_bridge::Event;
use rdna_compute::{DType, Gpu, Q8A8Tile};
use std::fmt::Write as _;

/// Raon-1-mini (M, K, label) projection sites.
const SHAPES: &[(usize, usize, &str)] = &[
    (5120, 2560, "RDA q/k/v/g, MLA g"),
    (2560, 5120, "o"),
    (576, 2560, "MLA kv_a"),
    (7680, 576, "MLA"),
    (10240, 512, "MLA q_b"),
];

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Approximately N(0, 1) (sum of 4 uniforms, variance 1/3 * 4 / 4 -> scaled).
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// Q8_0 weights `[m, k]`: bytes (34 B blocks) and the dequantized F64 matrix rows
/// needed by the reference (full matrix; m*k <= 5.3M doubles).
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f64>) {
    let nb = k / 32;
    let mut bytes = Vec::with_capacity(m * nb * 34);
    let mut deq = Vec::with_capacity(m * k);
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        let scf = sc.to_f32() as f64;
        for _ in 0..32 {
            let q = (rng.gauss() * 38.0).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
            deq.push(scf * q as f64);
        }
    }
    (bytes, deq)
}

/// BF16-valued activations: unit Gaussian bulk, ~0.8% persistent massive
/// channels (x25..x60) and ~1.5% hot rows (x8).
fn make_acts(n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let hot_ch: Vec<f32> = (0..k)
        .map(|_| {
            if rng.unit() < 0.008 {
                25.0 + 35.0 * rng.unit()
            } else {
                1.0
            }
        })
        .collect();
    let mut x = Vec::with_capacity(n * k);
    for _ in 0..n {
        let row = if rng.unit() < 0.015 { 8.0 } else { 1.0 };
        for c in 0..k {
            x.push(bf16_round(rng.gauss() * row * hot_ch[c]));
        }
    }
    x
}

struct Timer {
    e0: Event,
    e1: Event,
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(gpu: &mut Gpu, t: &Timer, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f32 {
    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters + 3 {
        gpu.hip.event_record(&t.e0, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&t.e1, None).unwrap();
        gpu.hip.event_synchronize(&t.e1).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&t.e0, &t.e1).unwrap();
        if i >= 3 {
            samples.push(ms);
        }
    }
    median(samples)
}

struct ErrStat {
    max_abs: f64,
    mean_abs: f64,
    rel: f64,
}

fn diff(a: &[f32], b: &[f32]) -> ErrStat {
    let (mut mx, mut s, mut sr) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x as f64 - y as f64).abs();
        mx = mx.max(d);
        s += d;
        sr += (y as f64).abs();
    }
    ErrStat {
        max_abs: mx,
        mean_abs: s / a.len() as f64,
        rel: s / sr.max(1e-30),
    }
}

/// Accuracy rows (one per route) against the F16 incumbent over the whole output and
/// against an F64 reference on a random subset of (token, row) outputs.
fn accuracy_rows(
    md: &mut String,
    (m, k, n): (usize, usize, usize),
    wdeq: &[f64],
    x_host: &[f32],
    seed: u64,
    inc: &[f32],
    routes: &[(&str, &[f32])],
) {
    let mut r = Rng(seed);
    let picks: Vec<(usize, usize)> = (0..384)
        .map(|_| (r.next_u32() as usize % n, r.next_u32() as usize % m))
        .collect();
    let ref_v: Vec<f32> = picks
        .iter()
        .map(|&(tok, row)| {
            let mut s = 0f64;
            for c in 0..k {
                s += wdeq[row * k + c] * x_host[tok * k + c] as f64;
            }
            s as f32
        })
        .collect();
    let sub = |y: &[f32]| -> Vec<f32> { picks.iter().map(|&(t, r)| y[t * m + r]).collect() };
    let mut emit = |name: &str, y: &[f32], d_inc: Option<ErrStat>| {
        let d_ref = diff(&sub(y), &ref_v);
        let (a, b, c) = d_inc.map_or(("-".into(), "-".into(), "-".into()), |d| {
            (
                format!("{:.4e}", d.max_abs),
                format!("{:.4e}", d.mean_abs),
                format!("{:.3e}", d.rel),
            )
        });
        let _ = writeln!(
            md,
            "| {m} | {k} | {n} | {name} | {a} | {b} | {c} | {:.4e} | {:.4e} | {:.3e} |",
            d_ref.max_abs, d_ref.mean_abs, d_ref.rel,
        );
    };
    emit("F16 incumbent", inc, None);
    for &(name, y) in routes {
        emit(name, y, Some(diff(y, inc)));
    }
}

const ACC_HEADER: &str = "| M | K | N | route | vs-incumbent max abs | mean abs | rel | vs-F64 max abs | mean abs | rel |\n|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let iters: usize = arg("--iters").map_or(21, |s| s.parse().unwrap());
    let rows: Vec<usize> = arg("--rows").map_or(vec![512, 4096], |s| {
        s.split(',').map(|v| v.parse().unwrap()).collect()
    });
    let big_only = arg("--shapes").as_deref() == Some("big");
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch {}", gpu.arch);
    assert!(
        gpu.q8_a8_supported(32) && gpu.q8_fp8_supported(32),
        "needs gfx12"
    );
    let timer = Timer {
        e0: gpu.hip.event_create().unwrap(),
        e1: gpu.hip.event_create().unwrap(),
    };
    let mut md = String::new();
    let _ = writeln!(
        md,
        "| M | K | N | F16 incumbent ms (TFLOP/s) | A8 ms (TFLOP/s) | FP8 ms (TFLOP/s) | A8 vs F16 | FP8 vs F16 | FP8 vs A8 | A8 quant ms | FP8 quant ms | A8 gemm 128x128 / 128x64 / 64x64 | FP8 gemm 128x128 / 128x64 / 64x64 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    );
    let mut acc_md = String::from(ACC_HEADER);
    let mut clean_md = String::from(ACC_HEADER);
    let mut rng = Rng(0x5eed_0a8a_0001);
    for &(m, k, label) in SHAPES {
        if big_only && m * k < 5120 * 2560 {
            continue;
        }
        let (wbytes, wdeq) = make_weights(m, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        for &n in &rows {
            println!("shape M={m} K={k} N={n} ({label})");
            let x_host = make_acts(n, k, &mut rng);
            let x = gpu.upload_f32(&x_host, &[n * k]).expect("x");
            let y_inc = gpu.alloc_tensor(&[n * m], DType::F32).expect("y");
            let y_a8 = gpu.alloc_tensor(&[n * m], DType::F32).expect("y2");
            let y_f8 = gpu.alloc_tensor(&[n * m], DType::F32).expect("y3");

            gpu.gemm_q8_0_batched_chunked(&w, &x, &y_inc, m, k, n)
                .unwrap();
            gpu.gemm_q8_0_a8_batched(&w, &x, &y_a8, m, k, n).unwrap();
            gpu.gemm_q8_0_fp8_batched(&w, &x, &y_f8, m, k, n).unwrap();
            gpu.hip.device_synchronize().unwrap();

            let t_inc = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_batched_chunked(&w, &x, &y_inc, m, k, n)
                    .unwrap()
            });
            let t_a8 = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_a8_batched(&w, &x, &y_a8, m, k, n).unwrap()
            });
            let t_f8 = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_fp8_batched(&w, &x, &y_f8, m, k, n).unwrap()
            });
            // Producer + per-tile GEMM-only times for both routes (they share one scratch,
            // so each route re-quantizes before its own tile loop).
            let t_q_a8 = time_ms(&mut gpu, &timer, iters, |g| {
                g.q8_a8_quantize_x(&x, n, k).unwrap();
            });
            let acts_a8 = gpu.q8_a8_quantize_x(&x, n, k).unwrap();
            // (the timing loop above leaves the sidecar quantized; re-quantize for the tile loop)
            let mut t_tiles_a8 = Vec::new();
            for tile in Q8A8Tile::ALL {
                t_tiles_a8.push(time_ms(&mut gpu, &timer, iters, |g| {
                    g.gemm_q8_0_a8_gemm_tile(&w, &acts_a8, &y_a8, m, tile)
                        .unwrap()
                }));
            }
            let t_q_f8 = time_ms(&mut gpu, &timer, iters, |g| {
                g.q8_fp8_quantize_x(&x, n, k).unwrap();
            });
            let acts_f8 = gpu.q8_fp8_quantize_x(&x, n, k).unwrap();
            let mut t_tiles_f8 = Vec::new();
            for tile in Q8A8Tile::ALL {
                t_tiles_f8.push(time_ms(&mut gpu, &timer, iters, |g| {
                    g.gemm_q8_0_fp8_gemm_tile(&w, &acts_f8, &y_f8, m, tile)
                        .unwrap()
                }));
            }
            // The tile loops left the last tile's output; recompute with the production
            // pick for the accuracy comparison.
            gpu.gemm_q8_0_a8_batched(&w, &x, &y_a8, m, k, n).unwrap();
            gpu.gemm_q8_0_fp8_batched(&w, &x, &y_f8, m, k, n).unwrap();
            gpu.hip.device_synchronize().unwrap();

            let tf = 2.0 * m as f64 * k as f64 * n as f64 / 1e12;
            let tiles = |t: &[f32]| format!("{:.4} / {:.4} / {:.4}", t[0], t[1], t[2]);
            let _ = writeln!(
                md,
                "| {m} | {k} | {n} | {t_inc:.4} ({:.1}) | {t_a8:.4} ({:.1}) | {t_f8:.4} ({:.1}) | {:.2}x | {:.2}x | {:.2}x | {t_q_a8:.4} | {t_q_f8:.4} | {} | {} |",
                tf / (t_inc as f64 / 1e3),
                tf / (t_a8 as f64 / 1e3),
                tf / (t_f8 as f64 / 1e3),
                t_inc / t_a8,
                t_inc / t_f8,
                t_a8 / t_f8,
                tiles(&t_tiles_a8),
                tiles(&t_tiles_f8),
            );

            let inc = gpu.download_f32(&y_inc).unwrap();
            let a8 = gpu.download_f32(&y_a8).unwrap();
            let f8 = gpu.download_f32(&y_f8).unwrap();
            accuracy_rows(
                &mut acc_md,
                (m, k, n),
                &wdeq,
                &x_host,
                0xf64 + (m * 31 + n) as u64,
                &inc,
                &[("A8", &a8), ("FP8", &f8)],
            );

            // Same comparison on plain BF16-rounded Gaussian activations (no outlier
            // channels / hot rows) for the smallest row count.
            if n == rows[0] {
                let xc: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
                let xct = gpu.upload_f32(&xc, &[n * k]).expect("xc");
                gpu.gemm_q8_0_batched_chunked(&w, &xct, &y_inc, m, k, n)
                    .unwrap();
                gpu.gemm_q8_0_a8_batched(&w, &xct, &y_a8, m, k, n).unwrap();
                gpu.gemm_q8_0_fp8_batched(&w, &xct, &y_f8, m, k, n).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let inc = gpu.download_f32(&y_inc).unwrap();
                let a8 = gpu.download_f32(&y_a8).unwrap();
                let f8 = gpu.download_f32(&y_f8).unwrap();
                accuracy_rows(
                    &mut clean_md,
                    (m, k, n),
                    &wdeq,
                    &xc,
                    0xc1ea + (m * 31 + n) as u64,
                    &inc,
                    &[("A8", &a8), ("FP8", &f8)],
                );
            }
        }
    }
    let out = format!(
        "## Node bench (median of {iters} calls, GPU events)\n\n{md}\n## Accuracy (BF16-valued X with massive channels + hot rows)\n\n{acc_md}\n## Accuracy (plain BF16-rounded Gaussian X)\n\n{clean_md}"
    );
    println!("\n{out}");
    if let Some(path) = arg("--out") {
        std::fs::write(&path, &out).expect("write --out");
    }
}
