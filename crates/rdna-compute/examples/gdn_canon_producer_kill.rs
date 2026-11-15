// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S0 kill 1 (CHUNK-CANONICAL-DESIGN §6 S0 item 1, §7 G0/G1): the **producer
//! switch** localized at Ornith absolute row 19968 — the same raw activation
//! rows and the same weights projected by the gfx1201 prepared-IU4 QKVZA route
//! (the route a 537-row call takes) versus the short-call route
//! (`Gpu::gemm_qkvza_hfq4g256_mq4v2`, what a 25-row call takes: F16 WMMA
//! instead of IU4 below 64 rows).
//!
//! Canonical reference = prepared-IU4 route at 537 rows (`ensure_int4_mmq_x`
//! quantizer → four `gemm_mq4g256v2_mmq_set_prequant_iu4` SETs, i.e. the gfx1201
//! body of `gemm_qkvza_mq4g256v2_wmma_iu4_prepared`). For n in {1, 25, 63, 64,
//! 537, ...} the first n rows of every route are compared per field (QKV, Z,
//! alpha, beta) with the canonical rows: byte identity and max-abs. The
//! receipt dumps the selected route (kernel names after every launch), the
//! admission predicates that make the route row-count dependent, and the scan
//! output dtype the production reader predicate would select for that n
//! (BF16 only when the IU4 producer-quant fusion is admitted, else F32).
//!
//! **Fail-before is required**: the short-call route must differ from the
//! canonical rows for n < 64 (verdict `fail`, exit 1). Controls: the canonical
//! route repeated is byte-identical (determinism), and a sign flip of one input
//! row changes the compared rows (not blind). Weights are synthetic *symmetric*
//! MQ4G256V2 (zero = -7.5 * scale, representable in F16) so the IU4 and F16
//! routes read the same weight values; routes differ only by activation /
//! reduction arithmetic.
//!
//! Build: cargo build --release -p rdna-compute --features lab,deltanet --example gdn_canon_producer_kill
//! Run:   gdn_canon_producer_kill --json <receipt.json> [--ns 1,25,63,64,537] [--hv 48]
//! Exit: 0 route-independent | 1 producer switch detected (fail-before) | 2 controls
//! invalid | 3 environment.

#[path = "gdn_canon_common/mod.rs"]
mod canon;

use canon::util::*;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

const K: usize = 5120;
const HK: usize = 16;
const HD: usize = 128;
const GROUP_BYTES: usize = 136;
const N_REF: usize = 537;

/// Random valid *symmetric* MQ4G256V2 rows: `m` rows of `k / 256` groups of 136 B:
/// two 128-wide halves each `[f16 scale, f16 zero, 64 nibble bytes...]` packed as
/// 4 x f16 header + 128 nibble bytes. zero = -7.5 * scale exactly.
fn synth_mq4v2_sym(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            // scale = (64 + j) * 2^-14: 7 significant bits, so 7.5 * scale is exact in F16.
            let scale = (64 + rng.below(64)) as f32 / 16384.0;
            let zero = -7.5 * scale;
            out.extend_from_slice(&f16_bits(scale).to_le_bytes());
            out.extend_from_slice(&f16_bits(zero).to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&rng.next_u32().to_le_bytes());
        }
    }
    out
}

fn f16_bits(x: f32) -> u16 {
    // exact for the values used here; delegate to the reference codec (RNE)
    canon::q8ef::f32_to_f16_bits(x)
}

struct Fields {
    qkv: GpuTensor,
    z: GpuTensor,
    beta: GpuTensor,
    alpha: GpuTensor,
}

struct Weights {
    qkv: GpuTensor,
    z: GpuTensor,
    beta: GpuTensor,
    alpha: GpuTensor,
}

#[derive(Clone)]
struct Out {
    qkv: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn alloc_fields(gpu: &mut Gpu, rows: usize, m: [usize; 4]) -> R<Fields> {
    Ok(Fields {
        qkv: gpu.zeros(&[rows * m[0]], DType::F32)?,
        z: gpu.zeros(&[rows * m[1]], DType::F32)?,
        beta: gpu.zeros(&[rows * m[2]], DType::F32)?,
        alpha: gpu.zeros(&[rows * m[3]], DType::F32)?,
    })
}

fn read_fields(gpu: &Gpu, f: &Fields, n: usize, m: [usize; 4]) -> R<Out> {
    gpu.hip.device_synchronize()?;
    let take = |t: &GpuTensor, w: usize| -> R<Vec<f32>> {
        let all = as_f32(&gpu.download_raw_bytes(t)?);
        Ok(all[..n * w].to_vec())
    };
    Ok(Out {
        qkv: take(&f.qkv, m[0])?,
        z: take(&f.z, m[1])?,
        beta: take(&f.beta, m[2])?,
        alpha: take(&f.alpha, m[3])?,
    })
}

fn clear_fields(gpu: &Gpu, f: &Fields) -> R<()> {
    gpu.hip.device_synchronize()?;
    for t in [&f.qkv, &f.z, &f.beta, &f.alpha] {
        gpu.hip.memset(&t.buf, 0, t.byte_size())?;
    }
    Ok(())
}

/// Canonical route: standalone IU4 activation quantizer + four prepared SETs.
/// Returns the kernel name observed after each launch.
fn run_prepared_iu4(
    gpu: &mut Gpu,
    w: &Weights,
    x: &GpuTensor,
    f: &Fields,
    m: [usize; 4],
    n: usize,
) -> R<Vec<String>> {
    let mut seen = Vec::new();
    let mut note = |gpu: &Gpu| {
        seen.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    };
    let xq = gpu.ensure_int4_mmq_x(x, n, K)?;
    note(gpu);
    gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w.qkv, xq, &f.qkv, m[0], K, n)?;
    note(gpu);
    gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w.z, xq, &f.z, m[1], K, n)?;
    note(gpu);
    // gfx1201 prepared consumer: beta/alpha use the same prequant IU4 SET as QKV/Z.
    gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w.beta, xq, &f.beta, m[2], K, n)?;
    note(gpu);
    gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w.alpha, xq, &f.alpha, m[3], K, n)?;
    note(gpu);
    gpu.hip.device_synchronize()?;
    Ok(seen)
}

/// Production unprepared / short-call route (`KernelKey::FusedQkvzaMq4G256V2`,
/// batch Some(n) -> `gemm_qkvza_hfq4g256_mq4v2`).
fn run_short_entry(
    gpu: &mut Gpu,
    w: &Weights,
    x: &GpuTensor,
    f: &Fields,
    m: [usize; 4],
    n: usize,
) -> R<Vec<String>> {
    gpu.gemm_qkvza_hfq4g256_mq4v2(
        &w.qkv, &w.z, &w.beta, &w.alpha, x, &f.qkv, &f.z, &f.beta, &f.alpha, m[0], m[1], m[2],
        m[3], K, n,
    )?;
    gpu.hip.device_synchronize()?;
    Ok(vec![gpu.last_launched_kernel().unwrap_or("?").to_string()])
}

fn cmp_field(name: &str, got: &[f32], want: &[f32]) -> Value {
    let identical = bits_eq(got, want);
    let (maxabs, nonfinite) = maxabs_diff(got, want);
    let ref_peak = want.iter().fold(0.0f64, |a, v| a.max(v.abs() as f64));
    let diff_count = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    json!({
        "field": name,
        "byte_identical": identical,
        "maxabs": maxabs,
        "ref_peak_abs": ref_peak,
        "elements_differing": diff_count,
        "elements": want.len(),
        "nonfinite_mismatch": nonfinite,
    })
}

fn cmp_out(got: &Out, want: &Out, n: usize, m: [usize; 4]) -> (bool, Vec<Value>) {
    let slice = |v: &[f32], w: usize| -> Vec<f32> { v[..n * w].to_vec() };
    let fields = [
        ("QKV", &got.qkv, &want.qkv, m[0]),
        ("Z", &got.z, &want.z, m[1]),
        ("alpha", &got.alpha, &want.alpha, m[3]),
        ("beta", &got.beta, &want.beta, m[2]),
    ];
    let mut ok = true;
    let mut js = Vec::new();
    for (name, g, w, width) in fields {
        let j = cmp_field(name, &slice(g, width), &slice(w, width));
        ok &= j["byte_identical"].as_bool() == Some(true);
        js.push(j);
    }
    (ok, js)
}

fn run() -> R<i32> {
    let args = Args::from_env();
    let json_path = args.value("--json");
    let hv: usize = args
        .value("--hv")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48);
    let ns = args
        .csv_usize("--ns")?
        .unwrap_or_else(|| vec![1, 2, 16, 25, 32, 63, 64, 65, 128, 129, 537]);
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gdn_canon_producer_kill: GPU init failed: {e}");
            return Ok(EXIT_ENV);
        }
    };
    if gpu.arch != "gfx1201" {
        eprintln!("gdn_canon_producer_kill: needs gfx1201 (got {})", gpu.arch);
        return Ok(EXIT_ENV);
    }
    // The loader claims the symmetric contract after verifying headers; the synthetic
    // weights are symmetric by construction.
    gpu.mq4v2_symmetric = true;
    if !gpu.flags.iu4_prefill_enabled() {
        eprintln!(
            "gdn_canon_producer_kill: IU4 prefill disabled by flags (HIPFIRE_IU4_PREFILL=0?)"
        );
        return Ok(EXIT_ENV);
    }
    let m = [(2 * HK + hv) * HD, hv * HD, hv, hv]; // qkv, z, beta, alpha
    let k_wo = hv * HD;
    let up = |gpu: &mut Gpu, bytes: Vec<u8>| -> R<GpuTensor> {
        Ok(gpu.upload_raw(&bytes, &[bytes.len()])?)
    };
    let w = Weights {
        qkv: up(&mut gpu, synth_mq4v2_sym(m[0], K, SEED_BASE | 0x101))?,
        z: up(&mut gpu, synth_mq4v2_sym(m[1], K, SEED_BASE | 0x102))?,
        beta: up(&mut gpu, synth_mq4v2_sym(m[2], K, SEED_BASE | 0x103))?,
        alpha: up(&mut gpu, synth_mq4v2_sym(m[3], K, SEED_BASE | 0x104))?,
    };
    // Raw X (post-norm/rotation activations): varied per-row scale, zero mean.
    let mut xr = Rng::new(SEED_BASE | 0x105);
    let mut x_host = Vec::with_capacity(N_REF * K);
    for r in 0..N_REF {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..K {
            x_host.push(xr.range(-1.0, 1.0) * scale);
        }
    }
    let x = gpu.upload_f32(&x_host, &[N_REF * K])?;
    let f = alloc_fields(&mut gpu, N_REF, m)?;

    // ── canonical reference (537 rows) + determinism control ──────────────────
    clear_fields(&gpu, &f)?;
    let ref_kernels = run_prepared_iu4(&mut gpu, &w, &x, &f, m, N_REF)?;
    let reference = read_fields(&gpu, &f, N_REF, m)?;
    clear_fields(&gpu, &f)?;
    run_prepared_iu4(&mut gpu, &w, &x, &f, m, N_REF)?;
    let again = read_fields(&gpu, &f, N_REF, m)?;
    let (determinism_ok, _) = cmp_out(&again, &reference, N_REF, m);
    // Sensitivity control: flip a whole input row; compared rows must change.
    let mut x_flip = x_host.clone();
    for v in &mut x_flip[(N_REF - 1) * K..] {
        *v = -*v;
    }
    let x2 = gpu.upload_f32(&x_flip, &[N_REF * K])?;
    clear_fields(&gpu, &f)?;
    run_prepared_iu4(&mut gpu, &w, &x2, &f, m, N_REF)?;
    let flipped = read_fields(&gpu, &f, N_REF, m)?;
    let (flip_identical, _) = cmp_out(&flipped, &reference, N_REF, m);
    let sensitivity_ok = !flip_identical;
    gpu.free_tensor(x2)?;
    let controls_ok = determinism_ok && sensitivity_ok;

    // ── per-n cells ───────────────────────────────────────────────────────────
    let mut cells: Vec<Value> = Vec::new();
    let mut failing = 0usize;
    for &n in &ns {
        // admission / recipe predicates that make routing row-count dependent
        let iu4_flag = gpu.flags.iu4_prefill_enabled();
        let a8 = gpu.a8_prefill_active(n, K);
        let unprep_takes_iu4 = !a8 && iu4_flag && n >= 64;
        let producer_fused = gpu.iu4_producer_quant_fused_active(n, K);
        let reader_fused = gpu.iu4_producer_quant_fused_active(n, k_wo);
        let scan_out = if gpu.flags.a8_fused_prod && gpu.a8_prefill_active(n, k_wo) {
            "F32"
        } else if reader_fused {
            match gpu.gdn_scan_out(rdna_compute::norm::GdnScanOut::Bf16)? {
                rdna_compute::norm::GdnScanOut::Bf16 => "BF16",
                rdna_compute::norm::GdnScanOut::F32 => "F32",
            }
        } else {
            "F32"
        };

        clear_fields(&gpu, &f)?;
        let k_prep = run_prepared_iu4(&mut gpu, &w, &x, &f, m, n)?;
        let prep = read_fields(&gpu, &f, n, m)?;
        clear_fields(&gpu, &f)?;
        let k_short = run_short_entry(&mut gpu, &w, &x, &f, m, n)?;
        let short = read_fields(&gpu, &f, n, m)?;

        let (prep_ok, prep_fields) = cmp_out(&prep, &reference, n, m);
        let (short_ok, short_fields) = cmp_out(&short, &reference, n, m);
        if !short_ok {
            failing += 1;
        }
        // The prepared route itself must not depend on the outer extent either.
        if !prep_ok {
            failing += 1;
        }
        cells.push(json!({
            "n": n,
            "recipe": {
                "iu4_prefill_flag": iu4_flag,
                "a8_prefill_active": a8,
                "short_entry_takes_iu4": unprep_takes_iu4,
                "short_entry_arithmetic": if unprep_takes_iu4 { "IU4 W4A4 MMQ (beta/alpha via small-tail IU4 SET)" } else { "F16-activation WMMA (gfx12 bt tile by n)" },
                "iu4_producer_quant_fused_active_qkvza_k": producer_fused,
                "iu4_producer_quant_fused_active_wo_k": reader_fused,
                "predicted_scan_out_dtype": scan_out,
                "predicted_scan_out_assumes": "wo MQ4G256V2, Residual epilogue, head_dim 128, heads*128 == K_wo (gdn_out_reader)",
            },
            "kernels": {"canonical_prepared_iu4": k_prep, "short_entry": k_short},
            "canonical_n_vs_canonical_537": {"byte_identical": prep_ok, "fields": prep_fields},
            "short_entry_vs_canonical_537": {"byte_identical": short_ok, "fields": short_fields},
        }));
    }

    for t in [w.qkv, w.z, w.beta, w.alpha, x, f.qkv, f.z, f.beta, f.alpha] {
        gpu.free_tensor(t)?;
    }

    let any_short_fail = cells
        .iter()
        .any(|c| c["short_entry_vs_canonical_537"]["byte_identical"].as_bool() == Some(false));
    let verdict = if !controls_ok {
        "oracle_invalid"
    } else if failing > 0 {
        "fail"
    } else {
        "pass"
    };
    let receipt = envelope(
        "gdn_canon_producer_kill",
        &gpu.arch,
        verdict,
        json!({
            "kill": 1,
            "geometry": {"hk": HK, "hv": hv, "head_dim": HD, "k": K, "m": {"qkv": m[0], "z": m[1], "beta": m[2], "alpha": m[3]}},
            "reference": {"route": "prepared IU4: ensure_int4_mmq_x + 4x gemm_mq4g256v2_mmq_set_prequant_iu4", "rows": N_REF, "kernels": ref_kernels},
            "controls": {"determinism_identical": determinism_ok, "sign_flip_changes_output": sensitivity_ok},
            "fail_before_demonstrated": any_short_fail,
            "cells_failing": failing,
            "cells": cells,
            "weights": "synthetic symmetric MQ4G256V2 (zero = -7.5*scale); seeds SEED_BASE|0x101..0x105",
        }),
    );
    write_receipt(&json_path, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "oracle": "gdn_canon_producer_kill",
            "verdict": verdict,
            "fail_before_demonstrated": any_short_fail,
            "controls_ok": controls_ok,
        }))?
    );
    Ok(match verdict {
        "pass" => EXIT_PASS,
        "fail" => EXIT_KILL_FAILS,
        _ => EXIT_ORACLE_INVALID,
    })
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("gdn_canon_producer_kill: {e}");
            std::process::exit(EXIT_ENV);
        }
    }
}
