// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real-GPU numerical probe for `gemm_mq4g256v2_residual_simt` and the four
//! temporary FP32 gfx1010 plain-set SIMT experiment variants
//! (`gemm_mq4g256v2_set_simt_variant`).
//!
//! Oracles (residual must agree with both within strict F32 tol):
//!   1. Existing per-row `gemv_hfq4g256_residual_mq4v2` on the same weights/X
//!   2. Independent host V2 dual-half dequant + F32 four-chain dot + residual
//!
//! Set arms additionally require raw bit-identical active output versus the
//! existing residual-SIMT launcher started from +0 Y (and GEMV-from-+0) on
//! finite synthetic fixtures. Stale Y must be overwritten; capacity tails
//! beyond active N stay untouched. Residual keeps nonzero Y+= semantics.
//!
//! Fixtures force dual-half headers to differ (disjoint half ranges) so a
//! wrong half-header select fails loudly. Also checks finite outputs and that
//! public plain / batched-lmhead entries overwrite stale Y on gfx1010 (stale Y
//! must not pass). The overwrite proof compares product equality — it does not
//! observe memset.
//!
//! Malformed-call gate (residual + every set arm; plain + batched_lmhead on
//! gfx1010): insufficient A/X/Y extents, wrong X/Y dtype, i32 M/K/N overflow,
//! and A/X/Y byte-count overflow. Every case must return `Err` **and** leave a
//! sentinel Y pattern byte-unchanged (catches mutation-before-validation).
//! Direct empty M/N no-ops must leave Y unchanged for residual and set arms.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example test_gemm_mq4g256v2_simt \
//!     --features lab -- [M] [K] [N1 N2 ...] [--timing|--timing-reverse]
//!
//! Defaults exercise N∈{1,3,7,8,9,16}, K∈{256,768,1024,2560}, M∈{16,33}
//! plus a representative larger M (CLI M or 512), then bounded non-cross-product
//! boundary shapes (R2 odd-M, tile edges, Spark K). Optional `--timing` /
//! `--timing-reverse` run the Spark projection census (N∈{64,46,8}) host-timed
//! launch+completion over residual-from-zero and each set arm (warmups 10,
//! measured ≥5, raw samples + median). Reverse flips the full five-entry order
//! for matched F/R/R/F fresh-process bias checks. Machine-readable `RESULT*` /
//! `RESULT_SET*` / `RESULT_MALFORMED*` / `RESULT_EMPTY*` / `RESULT_TIMING*` lines.

use rdna_compute::gemm::Mq4g256v2SimtSetVariant;
use rdna_compute::{DType, Gpu, GpuTensor};
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

const SET_VARIANTS: &[Mq4g256v2SimtSetVariant] = &[
    Mq4g256v2SimtSetVariant::SetR1T8Q4,
    Mq4g256v2SimtSetVariant::SetR1T8G1,
    Mq4g256v2SimtSetVariant::SetR1T16G1,
    Mq4g256v2SimtSetVariant::SetR2T8G1,
];

/// Non-cross-product boundary shapes from the validation recipe.
fn boundary_shapes() -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    // R2 clamp / store ownership on odd and tiny M.
    for &m in &[1usize, 2, 3] {
        for &n in &[1usize, 8, 9, 16] {
            out.push((m, 256, n));
        }
    }
    // Tile edges around T8/T16 with production-ish K.
    for &n in &[15usize, 16, 17] {
        out.push((33, 2560, n));
    }
    for &n in &[31usize, 32, 33] {
        out.push((33, 4096, n));
    }
    // Spark long-K loops + ragged production N tails.
    for &m in &[3usize, 33] {
        for &n in &[8usize, 46, 63, 64, 65] {
            out.push((m, 10240, n));
        }
    }
    out
}

fn set_variant_name(v: Mq4g256v2SimtSetVariant) -> &'static str {
    match v {
        Mq4g256v2SimtSetVariant::SetR1T8Q4 => "SetR1T8Q4",
        Mq4g256v2SimtSetVariant::SetR1T8G1 => "SetR1T8G1",
        Mq4g256v2SimtSetVariant::SetR1T16G1 => "SetR1T16G1",
        Mq4g256v2SimtSetVariant::SetR2T8G1 => "SetR2T8G1",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let timing_reverse = args.iter().any(|a| a == "--timing-reverse");
    let timing_mode = timing_reverse || args.iter().any(|a| a == "--timing");
    let timing_order = if timing_reverse { "reverse" } else { "forward" };
    let pos: Vec<String> = args
        .iter()
        .skip(1)
        .filter(|a| a.as_str() != "--timing" && a.as_str() != "--timing-reverse")
        .cloned()
        .collect();

    let cli_m: Option<usize> = pos.get(0).and_then(|s| s.parse().ok());
    let cli_k: Option<usize> = pos.get(1).and_then(|s| s.parse().ok());
    let cli_ns: Option<Vec<usize>> = if pos.len() > 2 {
        let v: Vec<usize> = pos[2..].iter().filter_map(|s| s.parse().ok()).collect();
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
    eprintln!("=== gemm_mq4g256v2 residual+set SIMT probe ===");
    eprintln!("arch={arch} timing_mode={timing_mode} order={timing_order}");
    println!(
        "RESULT_META arch={arch} timing_mode={} order={timing_order}",
        timing_mode as u8
    );

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

    // Envelope covers default grid + boundary shapes so guards stay valid.
    let bounds = boundary_shapes();
    let max_n = n_list
        .iter()
        .copied()
        .chain(bounds.iter().map(|s| s.2))
        .max()
        .unwrap_or(1);
    let max_m = m_list
        .iter()
        .copied()
        .chain(bounds.iter().map(|s| s.0))
        .max()
        .unwrap_or(1);
    let max_k = k_list
        .iter()
        .copied()
        .chain(bounds.iter().map(|s| s.1))
        .max()
        .unwrap_or(GROUP);

    // Shared host activations / residual init sized to the matrix envelope.
    let x_host = synth_x(max_n, max_k, 0xA11CE_u64);
    let y_init_host = synth_y_init(max_n, max_m, 0xBEEF_u64);
    // Guard sentinels beyond each active N: distinct nonzero pattern.
    let y_guard_host = synth_y_guard(max_n, max_m, 0xDEAD_u64);
    let y_stale_host = synth_y_stale(max_n, max_m, 0x51A1Eu64);

    let mut failures = 0usize;
    let mut cases = 0usize;

    // ── default residual + set cross-product ──
    for &m in &m_list {
        for &k in &k_list {
            let groups_per_row = k / GROUP;
            let row_bytes = groups_per_row * GROUP_BYTES;
            let weight_bytes = synth_mq4g256v2_weights(m, k, 0xC0DE_FACEu64);
            assert_dual_half_headers_differ(&weight_bytes, m, groups_per_row);

            let a_raw = gpu
                .upload_raw(&weight_bytes, &[m * row_bytes])
                .expect("upload weights");

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
                            "  residual {label}: max_abs={:.6e} max_norm={:.6e} gemv_abs={:.6e} cpu_abs={:.6e} finite=1 residual_nz={} guard_ok={} [{status}]",
                            rep.max_abs_all,
                            rep.max_norm_all,
                            rep.max_abs_gemv,
                            rep.max_abs_cpu,
                            rep.residual_nonzero,
                            rep.guard_ok,
                        );
                        println!(
                            "RESULT case={label} kind=residual arch={arch} max_abs={:.9e} max_norm={:.9e} gemv_abs={:.9e} cpu_abs={:.9e} residual_nz={} guard_ok={} status={}",
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
                        eprintln!("  residual {label}: ERROR {e}");
                        println!(
                            "RESULT case={label} kind=residual arch={arch} status=ERROR err={e}"
                        );
                    }
                }

                if arch.starts_with("gfx1010") {
                    match run_set_variants(
                        &mut gpu,
                        &a_raw,
                        &x_host,
                        &y_stale_host,
                        &y_guard_host,
                        m,
                        k,
                        n,
                        max_n,
                        max_m,
                        max_k,
                    ) {
                        Ok(reps) => {
                            for rep in reps {
                                cases += 1;
                                let status = if rep.ok { "PASS" } else { "FAIL" };
                                eprintln!(
                                    "  set {} {label}: bitdiff_res={} bitdiff_gemv={} guard_ok={} finite={} stale_overwritten={} [{status}]",
                                    rep.variant,
                                    rep.bitdiff_residual,
                                    rep.bitdiff_gemv,
                                    rep.guard_ok,
                                    rep.finite,
                                    rep.stale_overwritten,
                                );
                                println!(
                                    "RESULT_SET case={label} variant={} arch={arch} bitdiff_residual={} bitdiff_gemv={} guard_ok={} finite={} stale_overwritten={} status={}",
                                    rep.variant,
                                    rep.bitdiff_residual,
                                    rep.bitdiff_gemv,
                                    rep.guard_ok as u8,
                                    rep.finite as u8,
                                    rep.stale_overwritten as u8,
                                    if rep.ok { "PASS" } else { "FAIL" }
                                );
                                if !rep.ok {
                                    failures += 1;
                                    if let Some(d) = &rep.detail {
                                        eprintln!("    detail: {d}");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            failures += 1;
                            cases += SET_VARIANTS.len();
                            eprintln!("  set {label}: ERROR {e}");
                            println!("RESULT_SET case={label} arch={arch} status=ERROR err={e}");
                        }
                    }
                }
            }
        }
    }

    // ── bounded boundary coverage (non-cross-product) ──
    eprintln!("boundary shapes: {} cases", bounds.len());
    // Group by (m,k) to reuse uploads.
    let mut bound_groups: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for &(m, k, n) in &bounds {
        if let Some(g) = bound_groups
            .iter_mut()
            .find(|(mm, kk, _)| *mm == m && *kk == k)
        {
            if !g.2.contains(&n) {
                g.2.push(n);
            }
        } else {
            bound_groups.push((m, k, vec![n]));
        }
    }
    for (m, k, ns) in bound_groups {
        let groups_per_row = k / GROUP;
        let row_bytes = groups_per_row * GROUP_BYTES;
        let weight_bytes =
            synth_mq4g256v2_weights(m, k, 0xB0A7Du64.wrapping_add(m as u64 * 17 + k as u64));
        assert_dual_half_headers_differ(&weight_bytes, m, groups_per_row);
        let a_raw = gpu
            .upload_raw(&weight_bytes, &[m * row_bytes])
            .expect("upload boundary weights");
        let w_dequant: Vec<Vec<f32>> = (0..m)
            .map(|row| dequant_row_v2(&weight_bytes, row, k))
            .collect();
        for n in ns {
            cases += 1;
            let label = format!("bound_M={m} K={k} N={n}");
            match run_shape(
                &mut gpu,
                &a_raw,
                &w_dequant,
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
                    println!(
                        "RESULT case={label} kind=residual_bound arch={arch} max_abs={:.9e} residual_nz={} guard_ok={} status={}",
                        rep.max_abs_all,
                        rep.residual_nonzero as u8,
                        rep.guard_ok as u8,
                        if rep.ok { "PASS" } else { "FAIL" }
                    );
                    if !rep.ok {
                        failures += 1;
                        if let Some(d) = &rep.detail {
                            eprintln!("  residual {label}: FAIL {d}");
                        }
                    }
                }
                Err(e) => {
                    failures += 1;
                    eprintln!("  residual {label}: ERROR {e}");
                    println!(
                        "RESULT case={label} kind=residual_bound arch={arch} status=ERROR err={e}"
                    );
                }
            }
            if arch.starts_with("gfx1010") {
                match run_set_variants(
                    &mut gpu,
                    &a_raw,
                    &x_host,
                    &y_stale_host,
                    &y_guard_host,
                    m,
                    k,
                    n,
                    max_n,
                    max_m,
                    max_k,
                ) {
                    Ok(reps) => {
                        for rep in reps {
                            cases += 1;
                            println!(
                                "RESULT_SET case={label} variant={} arch={arch} bitdiff_residual={} bitdiff_gemv={} guard_ok={} finite={} stale_overwritten={} status={}",
                                rep.variant,
                                rep.bitdiff_residual,
                                rep.bitdiff_gemv,
                                rep.guard_ok as u8,
                                rep.finite as u8,
                                rep.stale_overwritten as u8,
                                if rep.ok { "PASS" } else { "FAIL" }
                            );
                            if !rep.ok {
                                failures += 1;
                                if let Some(d) = &rep.detail {
                                    eprintln!("  set {} {label}: FAIL {d}", rep.variant);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        cases += SET_VARIANTS.len();
                        eprintln!("  set {label}: ERROR {e}");
                        println!("RESULT_SET case={label} arch={arch} status=ERROR err={e}");
                    }
                }
            }
        }
    }

    // Plain / batched-lmhead overwrite gate — only meaningful on gfx1010 where
    // the SIMT fallback is auto-selected for N>1.
    if arch.starts_with("gfx1010") {
        match run_plain_lmhead_overwrite(&mut gpu) {
            Ok(()) => {
                eprintln!("  plain/lmhead overwrite (gfx1010): PASS");
                println!("RESULT case=plain_lmhead_overwrite arch={arch} status=PASS");
            }
            Err(e) => {
                failures += 1;
                eprintln!("  plain/lmhead overwrite (gfx1010): FAIL {e}");
                println!("RESULT case=plain_lmhead_overwrite arch={arch} status=FAIL err={e}");
            }
        }
    } else {
        eprintln!("  plain/lmhead overwrite: SKIP (arch={arch}, not gfx1010)");
        println!("RESULT case=plain_lmhead_overwrite arch={arch} status=SKIP");
    }

    // Direct empty M/N no-op: residual + every set arm leave Y unchanged.
    match run_empty_suite(&mut gpu, &arch) {
        Ok(n_ok) => {
            eprintln!("  empty suite: PASS ({n_ok} checks)");
            println!("RESULT case=empty_suite arch={arch} checks={n_ok} status=PASS");
        }
        Err(e) => {
            failures += 1;
            eprintln!("  empty suite: FAIL {e}");
            println!("RESULT case=empty_suite arch={arch} status=FAIL err={e}");
        }
    }

    // Malformed-call gate: Err + sentinel Y preserved.
    match run_malformed_suite(&mut gpu, &arch) {
        Ok(n_ok) => {
            eprintln!("  malformed suite: PASS ({n_ok} checks)");
            println!("RESULT case=malformed_suite arch={arch} checks={n_ok} status=PASS");
        }
        Err(e) => {
            failures += 1;
            eprintln!("  malformed suite: FAIL {e}");
            println!("RESULT case=malformed_suite arch={arch} status=FAIL err={e}");
        }
    }

    // Optional Spark-projection timing (does not affect numerical pass/fail).
    if timing_mode {
        if arch.starts_with("gfx1010") {
            if let Err(e) = run_spark_timing(&mut gpu, &arch, timing_reverse) {
                eprintln!("  spark timing: {e}");
                println!(
                    "RESULT_TIMING case=spark_timing arch={arch} order={timing_order} status=ERROR err={e}"
                );
            }
        } else {
            eprintln!("  spark timing: SKIP (arch={arch}, not gfx1010)");
            println!(
                "RESULT_TIMING case=spark_timing arch={arch} order={timing_order} status=SKIP"
            );
        }
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

struct SetReport {
    variant: &'static str,
    bitdiff_residual: usize,
    bitdiff_gemv: usize,
    guard_ok: bool,
    finite: bool,
    stale_overwritten: bool,
    ok: bool,
    detail: Option<String>,
}

fn run_shape(
    gpu: &mut Gpu,
    a_raw: &GpuTensor,
    w_dequant: &[Vec<f32>],
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
            y_cpu[b * m + row] += (acc0 + acc1) + (acc2 + acc3);
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

/// Zero-residual GPU oracles + each set arm from stale Y; require bitdiff=0.
fn run_set_variants(
    gpu: &mut Gpu,
    a_raw: &GpuTensor,
    x_host: &[f32],
    y_stale_host: &[f32],
    y_guard_host: &[f32],
    m: usize,
    k: usize,
    n: usize,
    max_n: usize,
    max_m: usize,
    max_k: usize,
) -> Result<Vec<SetReport>, String> {
    let mut x_n = vec![0.0f32; n * k];
    for b in 0..n {
        x_n[b * k..(b + 1) * k].copy_from_slice(&x_host[b * max_k..b * max_k + k]);
    }

    // Capacity Y with stale active region + guard beyond n.
    let y_cap = max_n * m;
    let mut y_stale_full = vec![0.0f32; y_cap];
    for b in 0..n {
        y_stale_full[b * m..(b + 1) * m].copy_from_slice(&y_stale_host[b * max_m..b * max_m + m]);
    }
    if n < max_n {
        for b in n..max_n {
            y_stale_full[b * m..(b + 1) * m]
                .copy_from_slice(&y_guard_host[b * max_m..b * max_m + m]);
        }
    }
    let y_guard_snapshot = y_stale_full[n * m..].to_vec();
    let y_stale_active = y_stale_full[..n * m].to_vec();

    let x_gpu = gpu
        .upload_f32(&x_n, &[n * k])
        .map_err(|e| format!("set upload x: {e}"))?;

    // Residual-from-+0 GPU oracle (existing residual symbol).
    let y_zero = vec![0.0f32; y_cap];
    let y_res_gpu = gpu
        .upload_f32(&y_zero, &[y_cap])
        .map_err(|e| format!("set residual oracle upload: {e}"))?;
    gpu.gemm_mq4g256v2_residual_simt(a_raw, &x_gpu, &y_res_gpu, m, k, n)
        .map_err(|e| format!("set residual oracle: {e}"))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("set residual sync: {e}"))?;
    let y_res_full = gpu
        .download_f32(&y_res_gpu)
        .map_err(|e| format!("set residual download: {e}"))?;
    let y_res = &y_res_full[..n * m];

    // GEMV-from-+0 oracle.
    let mut y_gemv = vec![0.0f32; n * m];
    let x_row = gpu
        .alloc_tensor(&[k], DType::F32)
        .map_err(|e| format!("set gemv x_row: {e}"))?;
    let y_row = gpu
        .alloc_tensor(&[m], DType::F32)
        .map_err(|e| format!("set gemv y_row: {e}"))?;
    for b in 0..n {
        gpu.hip
            .memcpy_htod(&x_row.buf, bytes_of(&x_n[b * k..(b + 1) * k]))
            .map_err(|e| format!("set gemv x: {e}"))?;
        let zero_row = vec![0.0f32; m];
        gpu.hip
            .memcpy_htod(&y_row.buf, bytes_of(&zero_row))
            .map_err(|e| format!("set gemv y: {e}"))?;
        gpu.gemv_hfq4g256_residual_mq4v2(a_raw, &x_row, &y_row, m, k)
            .map_err(|e| format!("set gemv: {e}"))?;
        let got = gpu
            .download_f32(&y_row)
            .map_err(|e| format!("set gemv dl: {e}"))?;
        y_gemv[b * m..(b + 1) * m].copy_from_slice(&got);
    }

    let mut reports = Vec::with_capacity(SET_VARIANTS.len());
    for &variant in SET_VARIANTS {
        let name = set_variant_name(variant);
        let y_gpu = gpu
            .upload_f32(&y_stale_full, &[y_cap])
            .map_err(|e| format!("{name} upload y: {e}"))?;
        gpu.gemm_mq4g256v2_set_simt_variant(a_raw, &x_gpu, &y_gpu, m, k, n, variant)
            .map_err(|e| format!("{name} launch: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("{name} sync: {e}"))?;
        let y_got_full = gpu
            .download_f32(&y_gpu)
            .map_err(|e| format!("{name} download: {e}"))?;
        let y_got = &y_got_full[..n * m];

        let guard_ok = if n < max_n {
            y_got_full[n * m..] == y_guard_snapshot[..]
        } else {
            true
        };
        let finite = y_got.iter().all(|v| v.is_finite());
        let bitdiff_residual = raw_bitdiff(y_got, y_res);
        let bitdiff_gemv = raw_bitdiff(y_got, &y_gemv);

        // Overwrite proof: active output matches residual-from-0 product, and
        // is not the stale pattern (unless product and stale coincide).
        let stale_overwritten = finite
            && bitdiff_residual == 0
            && (y_got != y_stale_active.as_slice() || y_res == y_stale_active.as_slice());

        let mut detail = None;
        let mut ok =
            finite && guard_ok && bitdiff_residual == 0 && bitdiff_gemv == 0 && stale_overwritten;
        if !finite {
            ok = false;
            detail = Some(format!("{name}: non-finite active output"));
        } else if bitdiff_residual != 0 {
            ok = false;
            let (i, a, b) = first_bit_mismatch(y_got, y_res).unwrap_or((0, 0.0, 0.0));
            detail = Some(format!(
                "{name} vs residual-from-0 bitdiff={bitdiff_residual} first i={i} (b={} r={}) got={a:.8e} want={b:.8e}",
                i / m,
                i % m
            ));
        } else if bitdiff_gemv != 0 {
            ok = false;
            let (i, a, b) = first_bit_mismatch(y_got, &y_gemv).unwrap_or((0, 0.0, 0.0));
            detail = Some(format!(
                "{name} vs gemv-from-0 bitdiff={bitdiff_gemv} first i={i} got={a:.8e} want={b:.8e}"
            ));
        } else if !guard_ok {
            ok = false;
            detail = Some(format!("{name}: ragged-N guard region mutated"));
        } else if !stale_overwritten {
            ok = false;
            detail = Some(format!("{name}: stale Y not overwritten"));
        }

        reports.push(SetReport {
            variant: name,
            bitdiff_residual,
            bitdiff_gemv,
            guard_ok,
            finite,
            stale_overwritten,
            ok,
            detail,
        });
    }
    Ok(reports)
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

fn raw_bitdiff(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn first_bit_mismatch(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    a.iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (x, y))| (i, *x, *y))
}

/// On gfx1010, plain + batched-lmhead must overwrite stale Y so residual
/// product (from +0) is observed. Does not assert memset; only product
/// equality. Uses a small fixed shape.
fn run_plain_lmhead_overwrite(gpu: &mut Gpu) -> Result<(), String> {
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
            "plain did not overwrite stale Y: i={i} got={a:.6e} want={b:.6e} (stale was {:.6e}) max_abs={abs_p:.6e}",
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
            "batched_lmhead did not overwrite stale Y: i={i} got={a:.6e} want={b:.6e} (stale was {:.6e}) max_abs={abs_l:.6e}",
            stale[i]
        ));
    }

    // Residual path must KEEP stale (not overwrite): control residual ≠ plain.
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
            "residual launcher appears to have overwritten Y (plain-vs-residual control failed)"
                .into(),
        );
    }

    Ok(())
}

// ── empty M/N direct no-op ────────────────────────────────────────────────

fn run_empty_suite(gpu: &mut Gpu, arch: &str) -> Result<usize, String> {
    let m0 = 8usize;
    let k0 = 256usize;
    let n0 = 4usize;
    let a_bytes = m0 * (k0 / GROUP) * GROUP_BYTES;
    let weight = synth_mq4g256v2_weights(m0, k0, 0xE0E0u64);
    let a_ok = gpu
        .upload_raw(&weight, &[a_bytes])
        .map_err(|e| format!("empty a: {e}"))?;
    let x_ok = gpu
        .upload_f32(&synth_x(n0.max(1), k0, 0x222u64), &[n0.max(1) * k0])
        .map_err(|e| format!("empty x: {e}"))?;
    const Y_LEN: usize = 32;
    let y = gpu
        .upload_f32(&vec![0.0f32; Y_LEN], &[Y_LEN])
        .map_err(|e| format!("empty y: {e}"))?;

    let mut checks = 0usize;
    let mut one = |entry: &str,
                   m: usize,
                   n: usize,
                   call: &mut dyn FnMut(&mut Gpu) -> Result<(), hip_bridge::HipError>|
     -> Result<(), String> {
        let nbytes = y.buf.size();
        let pattern = canary_bytes(b"empty", entry.as_bytes(), nbytes);
        gpu.hip
            .memcpy_htod(&y.buf, &pattern)
            .map_err(|e| format!("empty canary: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("empty sync pre: {e}"))?;
        call(gpu).map_err(|e| format!("{entry} empty M={m} N={n}: unexpected Err {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("empty sync post: {e}"))?;
        let mut got = vec![0u8; nbytes];
        gpu.hip
            .memcpy_dtoh(&mut got, &y.buf)
            .map_err(|e| format!("empty dl: {e}"))?;
        if got != pattern {
            return Err(format!(
                "{entry} empty M={m} N={n}: Y mutated (no-op must not touch Y)"
            ));
        }
        println!("RESULT_EMPTY entry={entry} case=M={m}_N={n} arch={arch} status=PASS");
        checks += 1;
        Ok(())
    };

    // Residual empty M / empty N.
    one("residual_simt", 0, n0, &mut |gpu| {
        gpu.gemm_mq4g256v2_residual_simt(&a_ok, &x_ok, &y, 0, k0, n0)
    })?;
    one("residual_simt", m0, 0, &mut |gpu| {
        gpu.gemm_mq4g256v2_residual_simt(&a_ok, &x_ok, &y, m0, k0, 0)
    })?;

    if arch.starts_with("gfx1010") {
        for &variant in SET_VARIANTS {
            let name = set_variant_name(variant);
            one(name, 0, n0, &mut |gpu| {
                gpu.gemm_mq4g256v2_set_simt_variant(&a_ok, &x_ok, &y, 0, k0, n0, variant)
            })?;
            one(name, m0, 0, &mut |gpu| {
                gpu.gemm_mq4g256v2_set_simt_variant(&a_ok, &x_ok, &y, m0, k0, 0, variant)
            })?;
        }
    }

    Ok(checks)
}

// ── malformed-call gate ───────────────────────────────────────────────────

/// Entry points exercised by the malformed suite.
#[derive(Clone, Copy)]
enum MalformedEntry {
    Residual,
    Plain,
    Lmhead,
    Set(Mq4g256v2SimtSetVariant),
}

impl MalformedEntry {
    fn name(self) -> &'static str {
        match self {
            Self::Residual => "residual_simt",
            Self::Plain => "plain",
            Self::Lmhead => "batched_lmhead",
            Self::Set(v) => set_variant_name(v),
        }
    }

    fn call(
        self,
        gpu: &mut Gpu,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), hip_bridge::HipError> {
        match self {
            Self::Residual => gpu.gemm_mq4g256v2_residual_simt(a, x, y, m, k, n),
            Self::Plain => gpu.gemm_mq4g256v2(a, x, y, m, k, n),
            Self::Lmhead => gpu.gemm_mq4g256v2_batched_lmhead(a, x, y, m, k, n),
            Self::Set(v) => gpu.gemm_mq4g256v2_set_simt_variant(a, x, y, m, k, n, v),
        }
    }
}

/// Malformed residual + set arms (+ gfx1010 plain/lmhead) checks.
///
/// Only honest small allocations + oversized scalar dims. No fabricated
/// buffer extents. Which guard rejects first is irrelevant; the observable
/// contract is `Err` and unchanged bytes in the **actual Y buffer passed**
/// to the call (catches mutation-before-validation on that tensor, including
/// wrong-dtype Y).
fn run_malformed_suite(gpu: &mut Gpu, arch: &str) -> Result<usize, String> {
    let m0 = 8usize;
    let k0 = 256usize;
    let n0 = 4usize; // >1 so plain/lmhead take the gfx1010 SIMT branch
    let a_bytes_ok = m0 * (k0 / GROUP) * GROUP_BYTES;
    let weight = synth_mq4g256v2_weights(m0, k0, 0xBAD_C0DEu64);
    let a_ok = gpu
        .upload_raw(&weight, &[a_bytes_ok])
        .map_err(|e| format!("malformed a_ok: {e}"))?;
    let x_ok = gpu
        .upload_f32(&synth_x(n0, k0, 0x111u64), &[n0 * k0])
        .map_err(|e| format!("malformed x_ok: {e}"))?;

    const Y_SENT_LEN: usize = 16;
    let y_f32 = gpu
        .upload_f32(&vec![0.0f32; Y_SENT_LEN], &[Y_SENT_LEN])
        .map_err(|e| format!("malformed y_f32: {e}"))?;

    let a_short = gpu
        .upload_raw(&[0u8; 64], &[64])
        .map_err(|e| format!("malformed a_short: {e}"))?;
    let x_short = gpu
        .upload_f32(&[1.0f32; 8], &[8])
        .map_err(|e| format!("malformed x_short: {e}"))?;
    let x_f16 = gpu
        .alloc_tensor(&[n0 * k0], DType::F16)
        .map_err(|e| format!("malformed x_f16: {e}"))?;
    let y_f16 = gpu
        .alloc_tensor(&[n0 * m0], DType::F16)
        .map_err(|e| format!("malformed y_f16: {e}"))?;

    let mut entries = vec![MalformedEntry::Residual];
    if arch.starts_with("gfx1010") {
        for &v in SET_VARIANTS {
            entries.push(MalformedEntry::Set(v));
        }
        entries.push(MalformedEntry::Plain);
        entries.push(MalformedEntry::Lmhead);
    }

    let mut checks = 0usize;

    let mut one = |entry: MalformedEntry,
                   tag: &str,
                   a: &GpuTensor,
                   x: &GpuTensor,
                   y: &GpuTensor,
                   m: usize,
                   k: usize,
                   n: usize|
     -> Result<(), String> {
        let nbytes = y.buf.size();
        if nbytes == 0 {
            return Err(format!("{}/{}: Y buffer reports size 0", entry.name(), tag));
        }
        let pattern = canary_bytes(tag.as_bytes(), entry.name().as_bytes(), nbytes);
        gpu.hip
            .memcpy_htod(&y.buf, &pattern)
            .map_err(|e| format!("{tag} y canary upload: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("{tag} sync pre: {e}"))?;

        let res = entry.call(gpu, a, x, y, m, k, n);
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("{tag} sync post: {e}"))?;

        match res {
            Ok(()) => {
                return Err(format!(
                    "{}/{}: expected Err, got Ok (would mask mutation-before-validation)",
                    entry.name(),
                    tag
                ));
            }
            Err(e) => {
                println!(
                    "RESULT_MALFORMED entry={} case={} arch={} status=ERR msg={}",
                    entry.name(),
                    tag,
                    arch,
                    e
                );
            }
        }

        let mut got = vec![0u8; nbytes];
        gpu.hip
            .memcpy_dtoh(&mut got, &y.buf)
            .map_err(|e| format!("{tag} y canary download: {e}"))?;
        if got != pattern {
            let first = got
                .iter()
                .zip(pattern.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(format!(
                "{}/{}: passed Y buffer mutated at byte {first} (len={nbytes})",
                entry.name(),
                tag
            ));
        }
        checks += 1;
        Ok(())
    };

    let m_i32 = (i32::MAX as usize).saturating_add(1);
    let k_i32 = ((i32::MAX as usize).saturating_add(256)) & !255;
    let n_i32 = (i32::MAX as usize).saturating_add(1);
    let m_a_ovf = usize::MAX / 8;
    let n_x_ovf = usize::MAX / 8;
    let m_y_ovf = 1024usize;
    let n_y_ovf = (usize::MAX / (m_y_ovf.saturating_mul(4).max(1))).saturating_add(1);

    for entry in entries {
        one(entry, "short_A", &a_short, &x_ok, &y_f32, m0, k0, n0)?;
        one(entry, "short_X", &a_ok, &x_short, &y_f32, m0, k0, n0)?;
        one(entry, "short_Y", &a_ok, &x_ok, &y_f32, m0, k0, n0)?;
        one(entry, "bad_x_dtype", &a_ok, &x_f16, &y_f32, m0, k0, n0)?;
        one(entry, "bad_y_dtype", &a_ok, &x_ok, &y_f16, m0, k0, n0)?;
        one(
            entry,
            "overflow_y_bytes",
            &a_ok,
            &x_ok,
            &y_f32,
            m_y_ovf,
            k0,
            n_y_ovf,
        )?;
        one(
            entry,
            "overflow_a_bytes",
            &a_ok,
            &x_ok,
            &y_f32,
            m_a_ovf,
            k0,
            n0,
        )?;
        one(
            entry,
            "overflow_x_bytes",
            &a_ok,
            &x_ok,
            &y_f32,
            m0,
            k0,
            n_x_ovf,
        )?;
        one(entry, "i32_overflow_M", &a_ok, &x_ok, &y_f32, m_i32, k0, n0)?;
        one(entry, "i32_overflow_K", &a_ok, &x_ok, &y_f32, m0, k_i32, n0)?;
        one(entry, "i32_overflow_N", &a_ok, &x_ok, &y_f32, m0, k0, n_i32)?;
    }

    Ok(checks)
}

/// Deterministic nonzero canary covering `nbytes` of a device Y buffer.
fn canary_bytes(tag: &[u8], entry: &[u8], nbytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; nbytes];
    let mut state = 0xC4A1_u64
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(nbytes as u64);
    for b in tag {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(*b as u64);
    }
    for b in entry {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(*b as u64);
    }
    for slot in &mut out {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *slot = ((state >> 33) as u8) | 1;
    }
    out
}

// ── optional Spark projection timing ──────────────────────────────────────

struct SparkShape {
    name: &'static str,
    m: usize,
    k: usize,
    multiplicity: usize,
}

const SPARK_SHAPES: &[SparkShape] = &[
    SparkShape {
        name: "qkv",
        m: 6144,
        k: 2560,
        multiplicity: 1,
    },
    SparkShape {
        name: "head_gate",
        m: 16,
        k: 2560,
        multiplicity: 1,
    },
    SparkShape {
        name: "attn_out",
        m: 2560,
        k: 4096,
        multiplicity: 1,
    },
    SparkShape {
        name: "ffn_gate_up",
        m: 10240,
        k: 2560,
        multiplicity: 2,
    },
    SparkShape {
        name: "ffn_down",
        m: 2560,
        k: 10240,
        multiplicity: 1,
    },
];

const SPARK_NS: &[usize] = &[64, 46, 8];
const TIMING_WARMUPS: usize = 10;
const TIMING_ITERS: usize = 5;

fn run_spark_timing(gpu: &mut Gpu, arch: &str, reverse: bool) -> Result<(), String> {
    let order = if reverse { "reverse" } else { "forward" };
    eprintln!(
        "=== spark projection timing (host-timed launch+completion; Y upload excluded; order={order}) ==="
    );

    // Five-entry schedule: residual baseline then four set arms. Reverse flips
    // the entire list (not just variants) for matched F/R bias checks.
    let mut entries: Vec<(&'static str, Option<Mq4g256v2SimtSetVariant>)> =
        Vec::with_capacity(1 + SET_VARIANTS.len());
    entries.push(("residual_from_zero", None));
    for &v in SET_VARIANTS {
        entries.push((set_variant_name(v), Some(v)));
    }
    if reverse {
        entries.reverse();
    }

    for shape in SPARK_SHAPES {
        let m = shape.m;
        let k = shape.k;
        let row_bytes = (k / GROUP) * GROUP_BYTES;
        let weight_bytes = synth_mq4g256v2_weights(m, k, 0x71AE_u64.wrapping_add(m as u64));
        let a_raw = gpu
            .upload_raw(&weight_bytes, &[m * row_bytes])
            .map_err(|e| format!("timing weights {}: {e}", shape.name))?;

        for &n in SPARK_NS {
            let x_host = synth_x(n, k, 0x71CE_u64);
            let x_gpu = gpu
                .upload_f32(&x_host, &[n * k])
                .map_err(|e| format!("timing x: {e}"))?;
            let y_elems = n * m;
            let y_zero = vec![0.0f32; y_elems];
            // Stale init so set path does real overwrite work.
            let y_stale: Vec<f32> = (0..y_elems)
                .map(|i| 1234.0 + (i as f32) * 0.001)
                .collect();
            let y_gpu = gpu
                .upload_f32(&y_zero, &[y_elems])
                .map_err(|e| format!("timing y: {e}"))?;

            for &(name, variant) in &entries {
                // Residual from +0; set arms from stale. Y htod outside timed window.
                let y_host: &[f32] = match variant {
                    None => &y_zero,
                    Some(_) => &y_stale,
                };
                let samples = time_launch_completion_us(gpu, &y_gpu, y_host, |gpu| match variant {
                    None => gpu.gemm_mq4g256v2_residual_simt(&a_raw, &x_gpu, &y_gpu, m, k, n),
                    Some(v) => {
                        gpu.gemm_mq4g256v2_set_simt_variant(&a_raw, &x_gpu, &y_gpu, m, k, n, v)
                    }
                })?;
                emit_timing(
                    arch,
                    shape.name,
                    name,
                    m,
                    k,
                    n,
                    shape.multiplicity,
                    order,
                    &samples,
                );
            }
        }
    }
    Ok(())
}

/// Host-timed launch+completion samples (µs). Wall clock covers Rust launch
/// call through `device_synchronize`; Y upload is outside the measured window.
/// Not a GPU-event kernel-only duration — rocprof owns true kernel times.
fn time_launch_completion_us<F>(
    gpu: &mut Gpu,
    y_gpu: &GpuTensor,
    y_host: &[f32],
    mut launch: F,
) -> Result<Vec<f64>, String>
where
    F: FnMut(&mut Gpu) -> Result<(), hip_bridge::HipError>,
{
    for _ in 0..TIMING_WARMUPS {
        gpu.hip
            .memcpy_htod(&y_gpu.buf, bytes_of(y_host))
            .map_err(|e| format!("warmup htod: {e}"))?;
        launch(gpu).map_err(|e| format!("warmup launch: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("warmup sync: {e}"))?;
    }
    let mut samples = Vec::with_capacity(TIMING_ITERS);
    for _ in 0..TIMING_ITERS {
        gpu.hip
            .memcpy_htod(&y_gpu.buf, bytes_of(y_host))
            .map_err(|e| format!("timed htod: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("pre sync: {e}"))?;
        let t = Instant::now();
        launch(gpu).map_err(|e| format!("timed launch: {e}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("post sync: {e}"))?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    Ok(samples)
}

fn emit_timing(
    arch: &str,
    shape: &str,
    entry: &str,
    m: usize,
    k: usize,
    n: usize,
    multiplicity: usize,
    order: &str,
    samples_us: &[f64],
) {
    let mut sorted = samples_us.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };
    let samples_str = samples_us
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "  timing {shape} entry={entry} order={order} M={m} K={k} N={n} mult={multiplicity} measurement=host_launch_completion: median={median:.1}µs samples=[{samples_str}]"
    );
    println!(
        "RESULT_TIMING shape={shape} entry={entry} M={m} K={k} N={n} multiplicity={multiplicity} arch={arch} order={order} measurement=host_launch_completion median_us={median:.3} samples_us={samples_str} warmups={TIMING_WARMUPS} iters={}",
        samples_us.len()
    );
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
            42.0 + u + (i as f32) * 1e-4
        })
        .collect()
}

/// Large nonzero stale pattern for set overwrite proofs.
fn synth_y_stale(n: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(0x51A1E);
        (state >> 33) as u32
    };
    (0..n * m)
        .map(|i| {
            let u = (next() as f32) / (u32::MAX as f32);
            1000.0 + 250.0 * u + (i as f32) * 0.125
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
