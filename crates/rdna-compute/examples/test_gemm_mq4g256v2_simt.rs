// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-GPU numerical probe for `gemm_mq4g256v2_residual_simt` (gfx1010
//! FP32-intermediate batched residual MQ4G256V2 GEMM).
//!
//! Oracles (both must agree with the SIMT launcher within strict F32 tol):
//!   1. Existing per-row `gemv_hfq4g256_residual_mq4v2` on the same weights/X
//!   2. Independent host V2 dual-half dequant + F32 dot + residual
//!
//! Fixtures force dual-half headers to differ (disjoint half ranges) so a
//! wrong half-header select fails loudly. Also checks nonzero residual init,
//! ragged-N capacity tails untouched, finite outputs, and that public
//! plain / batched-lmhead entries zero Y before the residual kernel on
//! gfx1010 (stale Y must not pass).
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example test_gemm_mq4g256v2_simt \
//!     --features lab -- [M] [K] [N1 N2 ...]
//!
//! Defaults exercise N∈{1,3,7,8,9,16}, K∈{256,768,1024,2560}, M∈{16,33}
//! plus a representative larger M (CLI M or 512). Optional shape timing is
//! reported after warmup; no absolute speed gate.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

/// Strict F32 path: SIMT, GEMV, and CPU all use F32 dequant/dot.
const ABS_TOL: f32 = 5e-5;
const REL_TOL: f32 = 1e-5;

const DEFAULT_NS: &[usize] = &[1, 3, 7, 8, 9, 16];
const DEFAULT_KS: &[usize] = &[256, 768, 1024, 2560];
const SMALL_MS: &[usize] = &[16, 33];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cli_m: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
    let cli_k: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let cli_ns: Option<Vec<usize>> = if args.len() > 3 {
        let v: Vec<usize> = args[3..].iter().filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    } else {
        None
    };

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== gemm_mq4g256v2_residual_simt probe ===");
    eprintln!("arch={arch}");
    println!("RESULT_META arch={arch}");

    let large_m = cli_m.unwrap_or(512);
    let mut m_list: Vec<usize> = SMALL_MS.to_vec();
    if !m_list.contains(&large_m) {
        m_list.push(large_m);
    }

    let k_list: Vec<usize> = match cli_k {
        Some(k) => {
            assert!(k % GROUP == 0, "K must be multiple of 256, got {k}");
            vec![k]
        }
        None => DEFAULT_KS.to_vec(),
    };
    let n_list: Vec<usize> = cli_ns.unwrap_or_else(|| DEFAULT_NS.to_vec());
    for &n in &n_list {
        assert!(n > 0, "N must be > 0");
    }

    eprintln!("shapes: M={m_list:?} K={k_list:?} N={n_list:?}  tol abs={ABS_TOL} rel={REL_TOL}");

    let max_n = *n_list.iter().max().unwrap();
    let max_m = *m_list.iter().max().unwrap();
    let max_k = *k_list.iter().max().unwrap();

    // Shared host activations / residual init sized to the matrix envelope.
    let x_host = synth_x(max_n, max_k, 0xA11CE_u64);
    let y_init_host = synth_y_init(max_n, max_m, 0xBEEF_u64);
    // Guard sentinels beyond each active N: distinct nonzero pattern.
    let y_guard_host = synth_y_guard(max_n, max_m, 0xDEAD_u64);

    let mut failures = 0usize;
    let mut cases = 0usize;

    for &m in &m_list {
        for &k in &k_list {
            let groups_per_row = k / GROUP;
            let row_bytes = groups_per_row * GROUP_BYTES;
            let weight_bytes = synth_mq4g256v2_weights(m, k, 0xC0DE_FACEu64);
            // Sanity: dual-half headers must differ on at least one group.
            assert_dual_half_headers_differ(&weight_bytes, m, groups_per_row);

            let a_raw = gpu
                .upload_raw(&weight_bytes, &[m * row_bytes])
                .expect("upload weights");

            // Precompute CPU dequant rows once per (m,k).
            let w_dequant: Vec<Vec<f32>> = (0..m)
                .map(|row| dequant_row_v2(&weight_bytes, row, k))
                .collect();

            for &n in &n_list {
                cases += 1;
                let label = format!("M={m} K={k} N={n}");
                match run_shape(
                    &mut gpu,
                    &a_raw,
                    &w_dequant,
                    &weight_bytes,
                    &x_host,
                    &y_init_host,
                    &y_guard_host,
                    m,
                    k,
                    n,
                    max_n,
                    max_m,
                    max_k,
                ) {
                    Ok(rep) => {
                        let status = if rep.ok { "PASS" } else { "FAIL" };
                        eprintln!(
                            "  {label}: max_abs={:.6e} max_norm={:.6e} gemv_abs={:.6e} cpu_abs={:.6e} finite=1 residual_nz={} guard_ok={} [{status}]",
                            rep.max_abs_all,
                            rep.max_norm_all,
                            rep.max_abs_gemv,
                            rep.max_abs_cpu,
                            rep.residual_nonzero,
                            rep.guard_ok,
                        );
                        println!(
                            "RESULT case={label} arch={arch} max_abs={:.9e} max_norm={:.9e} gemv_abs={:.9e} cpu_abs={:.9e} residual_nz={} guard_ok={} status={}",
                            rep.max_abs_all,
                            rep.max_norm_all,
                            rep.max_abs_gemv,
                            rep.max_abs_cpu,
                            rep.residual_nonzero as u8,
                            rep.guard_ok as u8,
                            if rep.ok { "PASS" } else { "FAIL" }
                        );
                        if !rep.ok {
                            failures += 1;
                            if let Some(d) = &rep.detail {
                                eprintln!("    detail: {d}");
                            }
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        eprintln!("  {label}: ERROR {e}");
                        println!("RESULT case={label} arch={arch} status=ERROR err={e}");
                    }
                }
            }

            // Optional shape timing (no absolute gate) at max N for this M/K.
            if let Some(&n_time) = n_list.iter().filter(|&&n| n > 1).max() {
                if let Err(e) = time_shape(
                    &mut gpu,
                    &a_raw,
                    &x_host,
                    &y_init_host,
                    m,
                    k,
                    n_time,
                    max_n,
                    max_m,
                    max_k,
                    &arch,
                ) {
                    eprintln!("  timing M={m} K={k} N={n_time}: {e}");
                }
            }
        }
    }

    // Plain / batched-lmhead zeroing gate — only meaningful on gfx1010 where
    // the SIMT fallback is auto-selected for N>1.
    if arch.starts_with("gfx1010") {
        match run_plain_lmhead_zeroing(&mut gpu, &arch) {
            Ok(()) => {
                eprintln!("  plain/lmhead zeroing (gfx1010): PASS");
                println!("RESULT case=plain_lmhead_zeroing arch={arch} status=PASS");
            }
            Err(e) => {
                failures += 1;
                eprintln!("  plain/lmhead zeroing (gfx1010): FAIL {e}");
                println!("RESULT case=plain_lmhead_zeroing arch={arch} status=FAIL err={e}");
            }
        }
    } else {
        eprintln!("  plain/lmhead zeroing: SKIP (arch={arch}, not gfx1010)");
        println!("RESULT case=plain_lmhead_zeroing arch={arch} status=SKIP");
    }

    eprintln!("\n=== summary: {cases} shape cases, failures={failures} arch={arch} ===");
    println!(
        "RESULT_SUMMARY arch={arch} cases={cases} failures={failures} status={}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    if failures > 0 {
        std::process::exit(1);
    }
}

struct ShapeReport {
    max_abs_gemv: f32,
    max_abs_cpu: f32,
    max_abs_all: f32,
    max_norm_all: f32,
    residual_nonzero: bool,
    guard_ok: bool,
    ok: bool,
    detail: Option<String>,
}

fn run_shape(
    gpu: &mut Gpu,
    a_raw: &rdna_compute::GpuTensor,
    w_dequant: &[Vec<f32>],
    _weight_bytes: &[u8],
    x_host: &[f32],
    y_init_host: &[f32],
    y_guard_host: &[f32],
    m: usize,
    k: usize,
    n: usize,
    max_n: usize,
    max_m: usize,
    max_k: usize,
) -> Result<ShapeReport, String> {
    // Active views: x is [n,k] taken from envelope [max_n, max_k] with row stride max_k.
    // Build contiguous x_n / y_n for this shape.
    let mut x_n = vec![0.0f32; n * k];
    for b in 0..n {
        let src = &x_host[b * max_k..b * max_k + k];
        x_n[b * k..(b + 1) * k].copy_from_slice(src);
    }
    let mut y_n = vec![0.0f32; n * m];
    for b in 0..n {
        let src = &y_init_host[b * max_m..b * max_m + m];
        y_n[b * m..(b + 1) * m].copy_from_slice(src);
    }
    let residual_nonzero = y_n.iter().any(|v| *v != 0.0 && v.is_finite());
    if !residual_nonzero {
        return Err("y_init is all zero — residual probe needs nonzero init".into());
    }

    // Capacity buffer with guard region beyond n (when n < max_n).
    let y_cap = max_n * m;
    let mut y_full = vec![0.0f32; y_cap];
    y_full[..n * m].copy_from_slice(&y_n);
    if n < max_n {
        for b in n..max_n {
            let g = &y_guard_host[b * max_m..b * max_m + m];
            y_full[b * m..(b + 1) * m].copy_from_slice(g);
        }
    }
    let y_guard_snapshot = y_full[n * m..].to_vec();

    let x_gpu = gpu
        .upload_f32(&x_n, &[n * k])
        .map_err(|e| format!("upload x: {e}"))?;
    let y_gpu = gpu
        .upload_f32(&y_full, &[y_cap])
        .map_err(|e| format!("upload y: {e}"))?;
    // Active Y view is the full capacity buffer; kernel only writes first n rows.
    // Launcher takes y with numel covering active region; pass capacity tensor
    // and rely on kernel ragged-N predicates for the rest.

    gpu.gemm_mq4g256v2_residual_simt(a_raw, &x_gpu, &y_gpu, m, k, n)
        .map_err(|e| format!("simt launch: {e}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("sync: {e}"))?;

    let y_simt_full = gpu
        .download_f32(&y_gpu)
        .map_err(|e| format!("download y: {e}"))?;
    let y_simt = &y_simt_full[..n * m];
    let guard_ok = if n < max_n {
        y_simt_full[n * m..] == y_guard_snapshot[..]
    } else {
        true
    };

    // ── GEMV residual oracle (per batch row) ──
    let mut y_gemv = y_n.clone();
    let x_row = gpu
        .alloc_tensor(&[k], DType::F32)
        .map_err(|e| format!("alloc x_row: {e}"))?;
    let y_row = gpu
        .alloc_tensor(&[m], DType::F32)
        .map_err(|e| format!("alloc y_row: {e}"))?;
    for b in 0..n {
        gpu.hip
            .memcpy_htod(&x_row.buf, bytes_of(&x_n[b * k..(b + 1) * k]))
            .map_err(|e| format!("x_row upload: {e}"))?;
        gpu.hip
            .memcpy_htod(&y_row.buf, bytes_of(&y_gemv[b * m..(b + 1) * m]))
            .map_err(|e| format!("y_row upload: {e}"))?;
        gpu.gemv_hfq4g256_residual_mq4v2(a_raw, &x_row, &y_row, m, k)
            .map_err(|e| format!("gemv residual: {e}"))?;
        let got = gpu
            .download_f32(&y_row)
            .map_err(|e| format!("gemv download: {e}"))?;
        y_gemv[b * m..(b + 1) * m].copy_from_slice(&got);
    }

    // ── CPU V2 dequant + F32 dot + residual ──
    let mut y_cpu = y_n.clone();
    for b in 0..n {
        let x = &x_n[b * k..(b + 1) * k];
        for row in 0..m {
            let mut acc = 0.0f32;
            // Match GEMV 4-acc interleave + pairwise combine for tight F32 parity.
            let gpr = k / GROUP;
            let quads = gpr >> 2;
            let tail = gpr & 3;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            let w = &w_dequant[row];
            for q in 0..quads {
                let g = q << 2;
                acc0 = dog_group(acc0, w, x, g);
                acc1 = dog_group(acc1, w, x, g + 1);
                acc2 = dog_group(acc2, w, x, g + 2);
                acc3 = dog_group(acc3, w, x, g + 3);
            }
            if tail >= 1 {
                acc0 = dog_group(acc0, w, x, quads << 2);
            }
            if tail >= 2 {
                acc1 = dog_group(acc1, w, x, (quads << 2) + 1);
            }
            if tail >= 3 {
                acc2 = dog_group(acc2, w, x, (quads << 2) + 2);
            }
            acc = (acc0 + acc1) + (acc2 + acc3);
            y_cpu[b * m + row] += acc;
        }
    }

    // Finite check.
    for (i, v) in y_simt.iter().enumerate() {
        if !v.is_finite() {
            return Ok(ShapeReport {
                max_abs_gemv: f32::INFINITY,
                max_abs_cpu: f32::INFINITY,
                max_abs_all: f32::INFINITY,
                max_norm_all: f32::INFINITY,
                residual_nonzero,
                guard_ok,
                ok: false,
                detail: Some(format!("non-finite simt out at i={i} val={v}")),
            });
        }
    }

    let (max_abs_gemv, max_norm_gemv, gemv_fail) = compare_tol(y_simt, &y_gemv);
    let (max_abs_cpu, max_norm_cpu, cpu_fail) = compare_tol(y_simt, &y_cpu);
    let max_abs_all = max_abs_gemv.max(max_abs_cpu);
    let max_norm_all = max_norm_gemv.max(max_norm_cpu);

    let mut detail = None;
    let mut ok = gemv_fail.is_none() && cpu_fail.is_none() && guard_ok && residual_nonzero;
    if let Some((i, a, b)) = gemv_fail {
        ok = false;
        detail = Some(format!(
            "simt vs gemv first mismatch i={i} (b={} r={}) simt={a:.8e} gemv={b:.8e}",
            i / m,
            i % m
        ));
    } else if let Some((i, a, b)) = cpu_fail {
        ok = false;
        detail = Some(format!(
            "simt vs cpu first mismatch i={i} (b={} r={}) simt={a:.8e} cpu={b:.8e}",
            i / m,
            i % m
        ));
    } else if !guard_ok {
        ok = false;
        detail = Some("ragged-N guard region mutated".into());
    }

    Ok(ShapeReport {
        max_abs_gemv,
        max_abs_cpu,
        max_abs_all,
        max_norm_all,
        residual_nonzero,
        guard_ok,
        ok,
        detail,
    })
}

fn dog_group(mut acc: f32, w: &[f32], x: &[f32], g: usize) -> f32 {
    let base = g * GROUP;
    // Full 256-dot for the group (CPU reference; GPU splits across 32 lanes
    // then warp-reduces — mathematically same sum, F32 assoc may differ by
    // a few ulps which ABS/REL_TOL absorb).
    for i in 0..GROUP {
        acc += w[base + i] * x[base + i];
    }
    acc
}

fn compare_tol(got: &[f32], want: &[f32]) -> (f32, f32, Option<(usize, f32, f32)>) {
    assert_eq!(got.len(), want.len());
    let mut max_abs = 0.0f32;
    let mut max_norm = 0.0f32;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        let scale = w.abs().max(g.abs()).max(1.0);
        let nrm = d / scale;
        max_norm = max_norm.max(nrm);
        let tol = ABS_TOL + REL_TOL * scale;
        if d > tol && first.is_none() {
            first = Some((i, *g, *w));
        }
    }
    (max_abs, max_norm, first)
}

fn time_shape(
    gpu: &mut Gpu,
    a_raw: &rdna_compute::GpuTensor,
    x_host: &[f32],
    y_init_host: &[f32],
    m: usize,
    k: usize,
    n: usize,
    max_n: usize,
    max_m: usize,
    max_k: usize,
    arch: &str,
) -> Result<(), String> {
    let mut x_n = vec![0.0f32; n * k];
    for b in 0..n {
        x_n[b * k..(b + 1) * k].copy_from_slice(&x_host[b * max_k..b * max_k + k]);
    }
    let mut y_n = vec![0.0f32; n * m];
    for b in 0..n {
        y_n[b * m..(b + 1) * m].copy_from_slice(&y_init_host[b * max_m..b * max_m + m]);
    }
    let x_gpu = gpu.upload_f32(&x_n, &[n * k]).map_err(|e| format!("{e}"))?;
    let y_gpu = gpu.upload_f32(&y_n, &[n * m]).map_err(|e| format!("{e}"))?;

    // Warmup
    for _ in 0..3 {
        gpu.hip
            .memcpy_htod(&y_gpu.buf, bytes_of(&y_n))
            .map_err(|e| format!("{e}"))?;
        gpu.gemm_mq4g256v2_residual_simt(a_raw, &x_gpu, &y_gpu, m, k, n)
            .map_err(|e| format!("{e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("{e}"))?;
    }

    // Timed SIMT
    let iters = 5usize;
    let mut simt_us = 0.0f64;
    for _ in 0..iters {
        gpu.hip
            .memcpy_htod(&y_gpu.buf, bytes_of(&y_n))
            .map_err(|e| format!("{e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("{e}"))?;
        let t = Instant::now();
        gpu.gemm_mq4g256v2_residual_simt(a_raw, &x_gpu, &y_gpu, m, k, n)
            .map_err(|e| format!("{e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("{e}"))?;
        simt_us += t.elapsed().as_secs_f64() * 1e6;
    }
    simt_us /= iters as f64;

    // Timed GEMV × N
    let x_row = gpu
        .alloc_tensor(&[k], DType::F32)
        .map_err(|e| format!("{e}"))?;
    let y_row = gpu
        .alloc_tensor(&[m], DType::F32)
        .map_err(|e| format!("{e}"))?;
    let mut gemv_us = 0.0f64;
    for _ in 0..iters {
        gpu.hip.device_synchronize().map_err(|e| format!("{e}"))?;
        let t = Instant::now();
        for b in 0..n {
            gpu.hip
                .memcpy_htod(&x_row.buf, bytes_of(&x_n[b * k..(b + 1) * k]))
                .map_err(|e| format!("{e}"))?;
            gpu.hip
                .memcpy_htod(&y_row.buf, bytes_of(&y_n[b * m..(b + 1) * m]))
                .map_err(|e| format!("{e}"))?;
            gpu.gemv_hfq4g256_residual_mq4v2(a_raw, &x_row, &y_row, m, k)
                .map_err(|e| format!("{e}"))?;
        }
        gpu.hip.device_synchronize().map_err(|e| format!("{e}"))?;
        gemv_us += t.elapsed().as_secs_f64() * 1e6;
    }
    gemv_us /= iters as f64;

    eprintln!(
        "  timing M={m} K={k} N={n}: simt={simt_us:.1}µs gemv×{n}={gemv_us:.1}µs ratio={:.2}x (informational)",
        gemv_us / simt_us
    );
    println!(
        "RESULT_TIMING case=M={m}_K={k}_N={n} arch={arch} simt_us={simt_us:.3} gemv_us={gemv_us:.3} ratio={:.6}",
        gemv_us / simt_us
    );
    Ok(())
}

/// On gfx1010, plain + batched-lmhead must zero Y before residual += so stale
/// Y cannot pass. Uses a small fixed shape.
fn run_plain_lmhead_zeroing(gpu: &mut Gpu, _arch: &str) -> Result<(), String> {
    let m = 32usize;
    let k = 256usize;
    let n = 4usize;
    let weight_bytes = synth_mq4g256v2_weights(m, k, 0x51517u64);
    let row_bytes = (k / GROUP) * GROUP_BYTES;
    let a_raw = gpu
        .upload_raw(&weight_bytes, &[m * row_bytes])
        .map_err(|e| format!("upload: {e}"))?;
    let x_host = synth_x(n, k, 0x1111u64);
    let x_gpu = gpu
        .upload_f32(&x_host, &[n * k])
        .map_err(|e| format!("x: {e}"))?;

    // Stale Y: large nonzero constant.
    let stale: Vec<f32> = (0..n * m).map(|i| 1000.0 + (i as f32) * 0.125).collect();

    // Reference: residual SIMT from zero Y (= plain product).
    let y_zero = vec![0.0f32; n * m];
    let y_ref_gpu = gpu
        .upload_f32(&y_zero, &[n * m])
        .map_err(|e| format!("{e}"))?;
    gpu.gemm_mq4g256v2_residual_simt(&a_raw, &x_gpu, &y_ref_gpu, m, k, n)
        .map_err(|e| format!("ref simt: {e}"))?;
    let y_ref = gpu.download_f32(&y_ref_gpu).map_err(|e| format!("{e}"))?;

    // plain entry
    let y_plain_gpu = gpu
        .upload_f32(&stale, &[n * m])
        .map_err(|e| format!("{e}"))?;
    gpu.gemm_mq4g256v2(&a_raw, &x_gpu, &y_plain_gpu, m, k, n)
        .map_err(|e| format!("gemm_mq4g256v2: {e}"))?;
    let y_plain = gpu.download_f32(&y_plain_gpu).map_err(|e| format!("{e}"))?;
    let (abs_p, _, fail_p) = compare_tol(&y_plain, &y_ref);
    if let Some((i, a, b)) = fail_p {
        return Err(format!(
            "plain did not zero Y: i={i} got={a:.6e} want={b:.6e} (stale was {:.6e}) max_abs={abs_p:.6e}",
            stale[i]
        ));
    }

    // batched lmhead entry
    let y_lm_gpu = gpu
        .upload_f32(&stale, &[n * m])
        .map_err(|e| format!("{e}"))?;
    gpu.gemm_mq4g256v2_batched_lmhead(&a_raw, &x_gpu, &y_lm_gpu, m, k, n)
        .map_err(|e| format!("gemm_mq4g256v2_batched_lmhead: {e}"))?;
    let y_lm = gpu.download_f32(&y_lm_gpu).map_err(|e| format!("{e}"))?;
    let (abs_l, _, fail_l) = compare_tol(&y_lm, &y_ref);
    if let Some((i, a, b)) = fail_l {
        return Err(format!(
            "batched_lmhead did not zero Y: i={i} got={a:.6e} want={b:.6e} (stale was {:.6e}) max_abs={abs_l:.6e}",
            stale[i]
        ));
    }

    // Residual path must KEEP stale (not zero): control that residual ≠ plain.
    let y_res_gpu = gpu
        .upload_f32(&stale, &[n * m])
        .map_err(|e| format!("{e}"))?;
    gpu.gemm_mq4g256v2_residual_simt(&a_raw, &x_gpu, &y_res_gpu, m, k, n)
        .map_err(|e| format!("residual control: {e}"))?;
    let y_res = gpu.download_f32(&y_res_gpu).map_err(|e| format!("{e}"))?;
    let mut residual_differs = false;
    for i in 0..n * m {
        if (y_res[i] - y_ref[i]).abs() > ABS_TOL + REL_TOL * y_ref[i].abs().max(1.0) {
            residual_differs = true;
            // Expect ≈ stale + product
            let expect = stale[i] + y_ref[i];
            let d = (y_res[i] - expect).abs();
            let tol = ABS_TOL + REL_TOL * expect.abs().max(1.0);
            if d > tol {
                return Err(format!(
                    "residual control mismatch i={i} got={:.6e} want_stale_plus={expect:.6e}",
                    y_res[i]
                ));
            }
        }
    }
    if !residual_differs {
        return Err(
            "residual launcher appears to have cleared Y (plain-vs-residual control failed)".into(),
        );
    }

    Ok(())
}

// ── synthetic fixtures ────────────────────────────────────────────────────

fn synth_x(n: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    (0..n * k)
        .map(|_| {
            let u = (next() as f32) / (u32::MAX as f32);
            u * 2.0 - 1.0
        })
        .collect()
}

fn synth_y_init(n: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as u32
    };
    (0..n * m)
        .map(|_| {
            let u = (next() as f32) / (u32::MAX as f32);
            // Nonzero residual in (-0.5, 0.5), avoid exact zeros.
            let v = u - 0.5;
            if v.abs() < 1e-3 {
                0.25
            } else {
                v
            }
        })
        .collect()
}

fn synth_y_guard(n: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(999);
        (state >> 33) as u32
    };
    (0..n * m)
        .map(|i| {
            let u = (next() as f32) / (u32::MAX as f32);
            // Distinct sentinel magnitude so accidental writes stand out.
            42.0 + u + (i as f32) * 1e-4
        })
        .collect()
}

/// Deterministic MQ4G256V2 weights with deliberately different dual-half
/// headers: half0 ∈ [-0.5,0.5], half1 ∈ [1.5,2.5] before packing. Ranges
/// stay disjoint so a wrong half-header select fails loudly, but magnitudes
/// stay modest so F32 assoc between CPU sequential dots and GPU warp
/// reduction stays inside strict tol.

fn synth_mq4g256v2_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let mut w = vec![0.0f32; m * k];
    let mut state = seed;
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut next_f = || (next_u() as f32) / (u32::MAX as f32);
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            for i in 0..HALF {
                w[base + i] = next_f() - 0.5;
            }
            for i in HALF..GROUP {
                w[base + i] = 1.5 + next_f();
            }
        }
    }
    pack_mq4g256v2(&w, m, k)
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    assert_eq!(w.len(), m * k);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (codes[2 * i] & 0xF) | ((codes[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

fn dequant_row_v2(blob: &[u8], row: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut out = vec![0.0f32; k];
    for g in 0..gpr {
        let dst = (row * gpr + g) * GROUP_BYTES;
        let hdr = |h: usize| {
            let o = dst + h * 4;
            let s = u16::from_le_bytes([blob[o], blob[o + 1]]);
            let z = u16::from_le_bytes([blob[o + 2], blob[o + 3]]);
            (f16_to_f32(s), f16_to_f32(z))
        };
        for i in 0..GROUP {
            let h = i / HALF;
            let (sc, zp) = hdr(h);
            let byte = blob[dst + 8 + i / 2];
            let q = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            out[g * GROUP + i] = sc * q as f32 + zp;
        }
    }
    out
}

fn assert_dual_half_headers_differ(blob: &[u8], m: usize, gpr: usize) {
    let mut found = false;
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let h0 = &blob[dst..dst + 4];
            let h1 = &blob[dst + 4..dst + 8];
            if h0 != h1 {
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(
        found,
        "fixture bug: no dual-half header differs — half-select errors would be silent"
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut val = (b & 0x7fff_ffff) as i32;
    if val >= 0x4780_0000 {
        return sign | 0x7c00;
    }
    if val < 0x3880_0000 {
        let f = f32::from_bits(val as u32);
        let sub = (f * 2f32.powi(24)).round() as i32;
        return sign | (sub as u16 & 0x03ff);
    }
    val += 0x0000_1000;
    sign | (((val - 0x3800_0000) >> 13) as u16)
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
