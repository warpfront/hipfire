// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of `Gpu::gemv_mq2g256v2_xbatch` (the MTP draft head's batched ranking
//! GEMV over the MQ2G256V2 copy): for every shape and B = 1..8, the whole `y` of the
//! x-batched kernel (launched as production does, in chunks of four, on the product
//! route: the PeaceMaker module on gfx1201 unless `HIPFIRE_QWEN4_MTP_DRAFT_HEAD_PM=0`)
//! must equal the bytes of `Gpu::gemv_mq2g256v2` launched once per column. Random
//! weights (finite f16 scale / zero-point headers, random packed 2-bit payloads) and
//! random activations; shapes cover K = 256 .. 4352 and every M residue mod 4.
//!
//! On a GPU the PeaceMaker module covers (gfx1201), every case also runs the module
//! and the hipcc kernel explicitly (`gemv_mq2g256v2_xbatch_with`) and compares their
//! whole `y` (written rows and the 0x7f guard past them). `--captures MANIFEST` adds
//! the WideNet F05 served captures (real ranking-copy weights and draft rows; each
//! case's PM and hipcc `y` against each other and against the captured `y`).
//!
//! `mq2_xbatch_oracle [--captures MANIFEST] --out FILE.json`; exit 0 only when every
//! case matches.

use rdna_compute::{DType, Gpu};
use serde_json::json;

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// xorshift64* stream.
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

/// `m` rows of `k / 256` 72-byte groups: two (scale, zero-point) f16 pairs, then 64
/// packed bytes.
fn weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = m * (k / 256);
    let mut bytes = Vec::with_capacity(groups * 72);
    for _ in 0..groups {
        for _ in 0..2 {
            let scale = 0.01 + 0.5 * (rng.f32().abs());
            let zero = -0.5 * scale * 3.0 + 0.05 * rng.f32();
            bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
            bytes.extend_from_slice(&f16_bits(zero).to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&rng.next().to_le_bytes());
        }
    }
    bytes
}

/// Whole-`y` bytes of the module (`pm`) or the hipcc kernel over `b` rows, the
/// buffer pre-filled with 0x7f.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &mut Gpu,
    a: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    y: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    b: usize,
    pm: bool,
) -> Result<Vec<u8>> {
    gpu.hip.memset(&y.buf, 0x7f, y.numel() * 4).map_err(err)?;
    gpu.gemv_mq2g256v2_xbatch_with(a, x, y, m, k, b, pm)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.download_raw_bytes(y).map_err(err)
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// One case per F05 capture of `manifest` (blobs under `<manifest dir>/blobs/`).
fn capture_cases(
    gpu: &mut Gpu,
    manifest: &str,
    cases: &mut Vec<serde_json::Value>,
) -> Result<usize> {
    let text = std::fs::read_to_string(manifest).map_err(err)?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(err)?;
    let dir = std::path::Path::new(manifest)
        .parent()
        .ok_or("manifest has no directory")?
        .join("blobs");
    let blob = |sha: &str| -> Result<Vec<u8>> {
        std::fs::read(dir.join(&sha[..2]).join(sha)).map_err(err)
    };
    let mut bad_total = 0;
    let mut seen = 0;
    for case in value["cases"].as_array().ok_or("no cases")? {
        if case["family"] != "F05" {
            continue;
        }
        seen += 1;
        let s = &case["scalars"];
        let (m, k, b) = (
            s["M"].as_u64().ok_or("M")? as usize,
            s["K"].as_u64().ok_or("K")? as usize,
            s["B"].as_u64().ok_or("B")? as usize,
        );
        let region = |id: &str| -> Result<&serde_json::Value> {
            case["regions"]
                .as_array()
                .and_then(|r| r.iter().find(|r| r["id"] == id))
                .ok_or_else(|| format!("region {id}"))
        };
        let a_host = blob(region("r0")?["pre_blob"].as_str().ok_or("r0")?)?;
        let x_host = blob(region("r1")?["pre_blob"].as_str().ok_or("r1")?)?;
        let y_captured = blob(region("r2")?["post_blob"].as_str().ok_or("r2")?)?;
        let a = gpu.upload_raw(&a_host, &[a_host.len()]).map_err(err)?;
        let x = gpu.upload_raw(&x_host, &[x_host.len()]).map_err(err)?;
        let y = gpu.zeros(&[(b + 1) * m], DType::F32).map_err(err)?;
        let hip = launch(gpu, &a, &x, &y, m, k, b, false)?;
        let pm = launch(gpu, &a, &x, &y, m, k, b, true)?;
        let written = b * m * 4;
        let pm_vs_hip = diff(&pm, &hip);
        let pm_vs_capture = diff(&pm[..written], &y_captured[..written.min(y_captured.len())]);
        bad_total += pm_vs_hip + pm_vs_capture;
        cases.push(json!({
            "capture": case["case_id"], "m": m, "k": k, "b": b,
            "pm_vs_hipcc_bytes": pm_vs_hip, "pm_vs_captured_y_bytes": pm_vs_capture,
        }));
        gpu.free_tensor(a).map_err(err)?;
        gpu.free_tensor(x).map_err(err)?;
        gpu.free_tensor(y).map_err(err)?;
    }
    if seen == 0 {
        return Err("no F05 captures in the manifest".into());
    }
    Ok(bad_total)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let out = flag("--out");
    let captures = flag("--captures");
    let run = || -> Result<(serde_json::Value, usize)> {
        let mut gpu = Gpu::init().map_err(err)?;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut cases = Vec::new();
        let mut pm_cases = Vec::new();
        let mut bad_total = 0usize;
        for (m, k) in [
            (1usize, 256usize),
            (2, 512),
            (3, 1024),
            (7, 256),
            (31, 768),
            (130, 2560),
            (1001, 2560),
            (4093, 2048),
            (33, 4352),
            (100003, 2560),
            (100276, 2560),
        ] {
            let a_bytes = weights(&mut rng, m, k);
            let a = gpu.upload_raw(&a_bytes, &[a_bytes.len()]).map_err(err)?;
            let x_host: Vec<f32> = (0..8 * k).map(|_| rng.f32() * 3.0).collect();
            let x = gpu.upload_f32(&x_host, &[x_host.len()]).map_err(err)?;
            let y_ref = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            let y = gpu.zeros(&[8 * m], DType::F32).map_err(err)?;
            for column in 0..8 {
                gpu.gemv_mq2g256v2(
                    &a,
                    &x.sub_offset(column * k, k),
                    &y_ref.sub_offset(column * m, m),
                    m,
                    k,
                )
                .map_err(err)?;
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let reference = gpu.download_raw_bytes(&y_ref).map_err(err)?;
            for b in 1..=8usize {
                gpu.hip.memset(&y.buf, 0x7f, 8 * m * 4).map_err(err)?;
                gpu.gemv_mq2g256v2_xbatch(&a, &x, &y, m, k, b)
                    .map_err(err)?;
                gpu.hip.device_synchronize().map_err(err)?;
                let got = gpu.download_raw_bytes(&y).map_err(err)?;
                let written = b * m * 4;
                let bad_written = got[..written]
                    .iter()
                    .zip(&reference[..written])
                    .filter(|(g, r)| g != r)
                    .count();
                let bad_guard = got[written..].iter().filter(|&&byte| byte != 0x7f).count();
                bad_total += bad_written + bad_guard;
                cases.push(json!({
                    "m": m, "k": k, "b": b,
                    "mismatch_bytes": bad_written, "overwritten_guard_bytes": bad_guard,
                }));
            }
            if gpu.mq2_xbatch_pm_covers(m, k) {
                for b in 1..=8usize {
                    let hip = launch(&mut gpu, &a, &x, &y, m, k, b, false)?;
                    let pm = launch(&mut gpu, &a, &x, &y, m, k, b, true)?;
                    let bad = diff(&pm, &hip);
                    bad_total += bad;
                    pm_cases.push(json!({"m": m, "k": k, "b": b, "pm_vs_hipcc_bytes": bad}));
                }
            }
            gpu.free_tensor(a).map_err(err)?;
            gpu.free_tensor(x).map_err(err)?;
            gpu.free_tensor(y_ref).map_err(err)?;
            gpu.free_tensor(y).map_err(err)?;
        }
        let pm_module = gpu.arch == "gfx1201";
        if pm_module && pm_cases.is_empty() {
            return Err("gfx1201: the PeaceMaker module covered no case".into());
        }
        if let Some(manifest) = captures.as_deref() {
            bad_total += capture_cases(&mut gpu, manifest, &mut pm_cases)?;
        }
        Ok((
            json!({
                "schema": "mq2-xbatch-oracle-v2", "arch": gpu.arch,
                "pm_image_sha256": rdna_compute::kernels::GEMV_MQ2_XBATCH_PM_GFX1201_SHA256,
                "cases": cases, "pm_cases": pm_cases, "mismatch_bytes_total": bad_total,
            }),
            bad_total,
        ))
    };
    match run() {
        Ok((value, bad)) => {
            if let Some(out) = out {
                std::fs::write(&out, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
                eprintln!("wrote {out}");
            }
            eprintln!("mq2 xbatch oracle: {bad} mismatching bytes");
            std::process::exit(i32::from(bad != 0));
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
