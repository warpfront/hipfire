// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the exact-gfx942 x-batched MQ2G256V2 GEMV of the MTP draft head's ranking
//! (`kernels/src/qwen4_g942_gemv_mq2g256v2_xbatch.gfx942.hip`, module
//! `qwen4_g942_gemv_mq2g256v2_xbatch`; `Gfx942Device::mq2g256v2_xbatch_head`).
//!
//! For every shape (K in {2048, 2560}, M covering every residue of the 32-row block and the
//! served 100276 / 248320 row counts) and every B = 1..=8, the whole `y` of ONE launch (0x7f
//! guard words on both sides must survive) must equal, byte for byte, the bytes of
//! `Gpu::gemv_mq2g256v2` launched once per column. The production wrapper
//! `Gpu::gemv_mq2g256v2_xbatch` (chunks of eight, B up to 16 here) is checked against the same
//! reference. The hipcc x-batched kernel of the other arches (`gemv_mq2g256v2_xbatch_with(..,
//! false)`) is run for information only. Random finite weights (f16 scale / zero-point headers,
//! random 2-bit payloads) and random activations; one case family draws the headers from a
//! wide dynamic range (finite, subnormal and large halves, no NaN / Inf) so the rounding of the
//! tail-group (unfused) and main-group (fused) chains is exercised.
//!
//! The kernel is loaded from the hipcc source, or, with
//! `HIPFIRE_QWEN4_G942_MQ2_XBATCH_IMAGE=/absolute/device.elf`, from that code object (the twin
//! proof: same module, same entry names, same kernarg ABI).
//!
//! `qwen4_g942_mq2_xbatch_oracle [--out FILE.json] [--bench]`; exit 0 only when every case
//! matches. `--bench` times the one-row kernel x B against the one-launch kernel at the served
//! shape (M = 248320, K = 2560).

use rdna_compute::{DType, Gpu};
use serde_json::json;
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// `m` rows of `k / 256` 72-byte groups: two (scale, zero-point) f16 pairs, then 64 packed
/// bytes. `wide`: header halves drawn over the whole finite range instead of near 1.
fn weights(rng: &mut Rng, m: usize, k: usize, wide: bool) -> Vec<u8> {
    let groups = m * (k / 256);
    let mut bytes = Vec::with_capacity(groups * 72);
    for _ in 0..groups {
        for _ in 0..2 {
            let (scale, zero) = if wide {
                let pick = |rng: &mut Rng| loop {
                    // Any finite half with a magnitude below 2^8: subnormals included.
                    let bits =
                        (rng.next() >> 40) as u16 & 0x5bff | ((rng.next() >> 50) as u16 & 0x8000);
                    let h = half::f16::from_bits(bits);
                    if h.is_finite() {
                        break h.to_f32();
                    }
                };
                (pick(rng), pick(rng))
            } else {
                let scale = 0.01 + 0.5 * rng.f32().abs();
                (scale, -0.5 * scale * 3.0 + 0.05 * rng.f32())
            };
            bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f16_bits(zero).to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    }
    bytes
}

const GUARD: usize = 64;

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

struct Case {
    m: usize,
    k: usize,
    wide: bool,
}

fn run_case(gpu: &mut Gpu, rng: &mut Rng, case: &Case) -> Result<Vec<serde_json::Value>> {
    let (m, k) = (case.m, case.k);
    let a_host = weights(rng, m, k, case.wide);
    let a = gpu.upload_raw(&a_host, &[a_host.len()]).map_err(err)?;
    let mut out = Vec::new();
    let max_b = 16;
    let x_host: Vec<f32> = (0..max_b * k)
        .map(|_| rng.f32() * if case.wide { 4.0 } else { 1.0 })
        .collect();
    let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
    // The one-row reference of every column.
    let y_ref = gpu.zeros(&[max_b * m], DType::F32).map_err(err)?;
    for b in 0..max_b {
        let xb = x.sub_offset(b * k, k);
        let yb = y_ref.sub_offset(b * m, m);
        gpu.gemv_mq2g256v2(&a, &xb, &yb, m, k).map_err(err)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let reference = gpu.download_raw_bytes(&y_ref).map_err(err)?;
    let y = gpu
        .zeros(&[max_b * m + 2 * GUARD], DType::F32)
        .map_err(err)?;
    let guarded = |gpu: &Gpu, y: &rdna_compute::GpuTensor| -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        gpu.download_raw_bytes(y).map_err(err)
    };
    for b in 1..=max_b {
        let written = b * m * 4;
        // One direct launch (B <= 8) into a guarded y.
        let mut kernel_bad = None;
        if b <= 8 {
            gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
            let body = y.sub_offset(GUARD, max_b * m);
            let xs = x.sub_offset(0, b * k);
            gpu.try_gfx942()
                .ok_or("not an exact gfx942 device")?
                .mq2g256v2_xbatch_head(&a, &xs, &body, m, k, b)
                .map_err(err)?;
            let bytes = guarded(gpu, &y)?;
            let got = &bytes[GUARD * 4..GUARD * 4 + written];
            let bad = diff(got, &reference[..written]);
            let guard_bad = bytes[..GUARD * 4].iter().any(|&v| v != 0x7f)
                || bytes[GUARD * 4 + written..].iter().any(|&v| v != 0x7f);
            kernel_bad = Some((bad, guard_bad));
        }
        // The production wrapper (chunks of eight).
        gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
        let body = y.sub_offset(GUARD, max_b * m);
        let xs = x.sub_offset(0, b * k);
        gpu.gemv_mq2g256v2_xbatch(&a, &xs, &body, m, k, b)
            .map_err(err)?;
        let bytes = guarded(gpu, &y)?;
        let got = &bytes[GUARD * 4..GUARD * 4 + written];
        let wrapper_bad = diff(got, &reference[..written]);
        let wrapper_guard_bad = bytes[..GUARD * 4].iter().any(|&v| v != 0x7f)
            || bytes[GUARD * 4 + written..].iter().any(|&v| v != 0x7f);
        // Information only: the hipcc x-batched kernel.
        gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
        gpu.gemv_mq2g256v2_xbatch_with(&a, &xs, &body, m, k, b, false)
            .map_err(err)?;
        let bytes = guarded(gpu, &y)?;
        let hipcc_bad = diff(
            &bytes[GUARD * 4..GUARD * 4 + written],
            &reference[..written],
        );
        out.push(json!({
            "m": m, "k": k, "b": b, "wide": case.wide,
            "kernel_bad_bytes": kernel_bad.map(|v| v.0),
            "kernel_guard_clobbered": kernel_bad.map(|v| v.1),
            "wrapper_bad_bytes": wrapper_bad,
            "wrapper_guard_clobbered": wrapper_guard_bad,
            "hipcc_xbatch_vs_singleton_bad_bytes_info": hipcc_bad,
        }));
    }
    gpu.free_tensor(a).map_err(err)?;
    gpu.free_tensor(x).map_err(err)?;
    gpu.free_tensor(y_ref).map_err(err)?;
    gpu.free_tensor(y).map_err(err)?;
    Ok(out)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bench(gpu: &mut Gpu, rng: &mut Rng) -> Result<Vec<serde_json::Value>> {
    let (m, k) = (248_320usize, 2560usize);
    // Three rotating weight copies (3 x 179 MB: past the MALL), as a served step sees them cold.
    let copies = 3;
    let a_host = weights(rng, m, k, false);
    let mut a = Vec::new();
    for _ in 0..copies {
        a.push(gpu.upload_raw(&a_host, &[a_host.len()]).map_err(err)?);
    }
    let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32()).collect();
    let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
    let y = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
    let mut rows = Vec::new();
    for b in [1usize, 2, 4, 5, 8] {
        let mut single = Vec::new();
        let mut batched = Vec::new();
        for rep in 0..21 {
            let w = &a[rep % copies];
            gpu.hip.device_synchronize().map_err(err)?;
            let t = Instant::now();
            for c in 0..b {
                gpu.gemv_mq2g256v2(w, &x.sub_offset(c * k, k), &y.sub_offset(c * m, m), m, k)
                    .map_err(err)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let singles_us = t.elapsed().as_secs_f64() * 1e6;
            let t = Instant::now();
            gpu.gemv_mq2g256v2_xbatch(w, &x.sub_offset(0, b * k), &y.sub_offset(0, b * m), m, k, b)
                .map_err(err)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let batch_us = t.elapsed().as_secs_f64() * 1e6;
            if rep >= 3 {
                single.push(singles_us);
                batched.push(batch_us);
            }
        }
        rows.push(json!({
            "b": b,
            "b_singleton_launches_us": median(single),
            "one_launch_us": median(batched),
        }));
    }
    for w in a {
        gpu.free_tensor(w).map_err(err)?;
    }
    gpu.free_tensor(x).map_err(err)?;
    gpu.free_tensor(y).map_err(err)?;
    Ok(rows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let do_bench = args.iter().any(|a| a == "--bench");
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.try_gfx942().is_none() {
        eprintln!("skip: needs an exact gfx942 device");
        std::process::exit(0);
    }
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut cases = Vec::new();
    let shapes = [
        1usize, 5, 7, 31, 32, 33, 63, 64, 65, 100, 257, 1025, 4099, 100_276, 248_320,
    ];
    for &k in &[2560usize, 2048] {
        for &m in &shapes {
            // The served row counts only at K = 2560 (the ranking copy of the head).
            if k == 2048 && m > 4099 {
                continue;
            }
            for wide in [false, true] {
                if wide && m > 4099 {
                    continue;
                }
                match run_case(&mut gpu, &mut rng, &Case { m, k, wide }) {
                    Ok(rows) => cases.extend(rows),
                    Err(e) => {
                        eprintln!("case M={m} K={k} wide={wide}: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
    let bad = cases
        .iter()
        .filter(|c| {
            c["kernel_bad_bytes"].as_u64().unwrap_or(0) != 0
                || c["kernel_guard_clobbered"].as_bool().unwrap_or(false)
                || c["wrapper_bad_bytes"].as_u64().unwrap_or(0) != 0
                || c["wrapper_guard_clobbered"].as_bool().unwrap_or(false)
        })
        .count();
    let hipcc_info: u64 = cases
        .iter()
        .map(|c| {
            c["hipcc_xbatch_vs_singleton_bad_bytes_info"]
                .as_u64()
                .unwrap_or(0)
        })
        .sum();
    let bench_rows = if do_bench {
        match bench(&mut gpu, &mut rng) {
            Ok(rows) => Some(rows),
            Err(e) => {
                eprintln!("bench: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let report = json!({
        "oracle": "qwen4_g942_mq2_xbatch",
        "image_hook": std::env::var("HIPFIRE_QWEN4_G942_MQ2_XBATCH_IMAGE").ok(),
        "cases": cases.len(),
        "failing_cases": bad,
        "hipcc_xbatch_vs_singleton_bad_bytes_info": hipcc_info,
        "bench": bench_rows,
        "detail": cases,
    });
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
            .expect("write report");
    }
    println!(
        "qwen4_g942_mq2_xbatch_oracle: {} cases, {} failing; hipcc xbatch vs singleton (info) {} bytes{}",
        cases.len(),
        bad,
        hipcc_info,
        bench_rows
            .as_ref()
            .map(|r| format!("\nbench {}", serde_json::to_string(r).unwrap()))
            .unwrap_or_default()
    );
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
