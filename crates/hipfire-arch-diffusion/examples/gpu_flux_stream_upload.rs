// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Device-side acceptance for the STREAMING weight upload (host
//! memory diet): prove `GpuFluxWeights::from_stream` puts the same bits on
//! the GPU as `GpuFluxWeights::from_host` did, and that it does so without
//! ever holding a model-sized host table.
//!
//! Why this is the gate that matters. The streaming path replaces
//! "decode BF16 → f32 host `Vec` → upload f32 → cast on the device" with
//! "convert BF16 → f16 on the host, one tensor at a time → upload f16".
//! The unit tests pin the host half — `f32_to_f16_rne` against a reference
//! round-to-nearest-even over every bf16 and every f16 bit pattern. The half
//! they CANNOT pin is the claim that the device's `(_Float16)` cast rounds
//! the same way. If it did not, every weight in the model would shift by up
//! to 1 ULP and the block-parity gates would move for a reason no diff
//! explains. So this harness runs both uploads over a synthetic checkpoint at
//! a VRAM-friendly lab geometry and asserts the downloaded halves are
//! BYTE-IDENTICAL, key by key.
//!
//! It also reports `VmHWM` around each upload, which is the other half of the
//! task: the eager path's peak includes the whole f32 table, the streaming
//! path's peak is one tensor.
//!
//! ```
//! source scripts/gpu-lock.sh && gpu_acquire "stream-upload"
//! cargo run --release -p hipfire-arch-diffusion --features lab \
//!   --example gpu_flux_stream_upload
//! gpu_release
//! ```

use hipfire_arch_diffusion::config::FluxDiffusionConfig;
use hipfire_arch_diffusion::flux::{load_weights, FluxLayout, FluxPlan};
use hipfire_arch_diffusion::flux_gpu::GpuFluxWeights;
use hipfire_arch_diffusion::manifest::expected_flux_keys;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::{DType, Gpu};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Same VRAM-friendly geometry the block gates use: every structural
/// feature of the real model (both block kinds, fused qkv/linear1, the norm
/// scales, the final head) at a size a laptop iGPU can hold twice over.
fn lab_cfg() -> FluxDiffusionConfig {
    let json = serde_json::json!({
        "in_channels": 64,
        "num_layers": 2,
        "num_single_layers": 2,
        "attention_head_dim": 32,
        "num_attention_heads": 4,
        "joint_attention_dim": 128,
        "pooled_projection_dim": 96,
        "guidance_embeds": true,
        "patch_size": 1,
        "axes_dims_rope": [8, 12, 12],
    });
    FluxDiffusionConfig::from_json(&json).expect("lab config")
}

/// Deterministic pseudo-values covering every corner the conversion has.
///
/// Real FLUX weights all sit in f16's normal range, so most of the spread is
/// there — that is where rounding is the whole story. But the host and the
/// device also have to agree on what happens OUTSIDE it, and disagreement
/// there is silent: a saturating weight would become `inf` on one path and
/// `65504` on the other, and nothing would report it. So roughly one word in
/// eight is pushed into the range corners:
///
/// - exponent ≥ +16 → beyond f16's largest normal, must give ±inf;
/// - exponent in [-24, -15] → f16 subnormal territory, where the rounding
///   shift is data-dependent;
/// - exponent ≤ -26 → below half the smallest subnormal, must flush to ±0.
fn synth_bits(i: u64) -> u16 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    // A bf16 word: sign + 8-bit exponent (bias 127) + 7-bit mantissa.
    let sign = ((x >> 15) & 1) as u16;
    let man = ((x >> 20) & 0x7F) as u16;
    let unbiased: i32 = match (x >> 40) % 8 {
        0 => 16 + (x % 40) as i32,                // overflow -> inf
        1 => -24 + (x % 10) as i32,               // f16 subnormals
        2 if x % 3 == 0 => -26 - (x % 60) as i32, // underflow -> zero
        _ => -8 + (x % 16) as i32,                // 2^-8 .. 2^7, the normal case
    };
    let exp = (unbiased + 127) as u16 & 0xFF;
    (sign << 15) | (exp << 7) | man
}

fn write_synthetic_checkpoint(dir: &Path, cfg: &FluxDiffusionConfig) {
    std::fs::create_dir_all(dir).expect("create checkpoint dir");
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({ "model_type": "flux" }).to_string(),
    )
    .expect("write config.json");

    let mut header = serde_json::Map::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut offset = 0usize;
    let mut idx = 0u64;
    for k in expected_flux_keys(cfg) {
        let n = k.rows * k.cols;
        let data: Vec<u8> = (0..n as u64)
            .flat_map(|i| synth_bits(idx + i).to_le_bytes())
            .collect();
        idx += n as u64;
        let shape = if k.cols == 1 {
            vec![k.rows]
        } else {
            vec![k.rows, k.cols]
        };
        let mut meta = serde_json::Map::new();
        meta.insert("dtype".into(), "BF16".into());
        meta.insert(
            "shape".into(),
            serde_json::Value::Array(shape.into_iter().map(|s| s.into()).collect()),
        );
        meta.insert(
            "data_offsets".into(),
            serde_json::json!([offset, offset + data.len()]),
        );
        offset += data.len();
        header.insert(k.name, meta.into());
        blobs.push(data);
    }
    let header_json = serde_json::Value::Object(header).to_string();
    let mut f = std::fs::File::create(dir.join("model.safetensors")).expect("create safetensors");
    f.write_all(&(header_json.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(header_json.as_bytes()).unwrap();
    for b in &blobs {
        f.write_all(b).unwrap();
    }
}

fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn main() {
    let cfg = lab_cfg();
    let dir: PathBuf =
        std::env::temp_dir().join(format!("hipfire-flux-stream-upload-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_synthetic_checkpoint(&dir, &cfg);
    let src = SafetensorsSource::open(&dir).expect("open synthetic checkpoint");
    let plan = FluxPlan::detect(&src, &cfg);
    assert_eq!(plan.layout, FluxLayout::Bfl, "synthetic checkpoint is BFL");
    plan.validate(&cfg).expect("plan covers the manifest");

    let keys = expected_flux_keys(&cfg);
    let params: usize = keys.iter().map(|k| k.rows * k.cols).sum();
    println!(
        "lab geometry: {} keys, {params} parameters ({:.1} MB as f32 on the host)",
        keys.len(),
        params as f64 * 4.0 / 1e6
    );

    let mut gpu = Gpu::init().expect("GPU init failed");

    // ── streaming upload ────────────────────────────────────────────────
    let rss_before_stream = peak_rss_mib();
    let streamed = GpuFluxWeights::from_stream(&mut gpu, &src, &plan, &cfg)
        .expect("GpuFluxWeights::from_stream");
    let rss_after_stream = peak_rss_mib();

    // ── the path it replaces: whole-model f32 host table, device cast ────
    let host = load_weights(&src, &cfg).expect("load_weights (eager f32)");
    let eager = GpuFluxWeights::from_host(&mut gpu, &host, &cfg).expect("from_host");
    let rss_after_eager = peak_rss_mib();

    // ── bit-for-bit comparison ─────────────────────────────────────────
    // Compare the UPLOADED key sets, not the manifest key list. With the
    // default `HIPFIRE_FLUX_F16_ACT`, the upload convention splits every
    // `single_blocks.N.linear2.weight` along K into `.linear2.w_attn.weight`
    // and `.linear2.w_mlp.weight` (see `upload_flux_key` / `stream_into` in
    // `flux_gpu.rs`), so the fused manifest name is a key of NEITHER map and
    // `get` on it would panic. Both paths must apply the same convention:
    // assert the key sets are equal first, then walk that set.
    let mut streamed_keys: Vec<&str> = streamed.tensors.keys().map(String::as_str).collect();
    let mut eager_keys: Vec<&str> = eager.tensors.keys().map(String::as_str).collect();
    streamed_keys.sort_unstable();
    eager_keys.sort_unstable();
    assert_eq!(
        streamed_keys, eager_keys,
        "streamed and eager uploads expose DIFFERENT key sets — the two paths \
         disagree about the linear2 split convention"
    );
    let uploaded: Vec<String> = streamed_keys.iter().map(|s| (*s).to_string()).collect();
    println!(
        "uploaded key sets agree: {} device tensors (manifest lists {} keys)",
        uploaded.len(),
        keys.len()
    );

    let mut f16_keys = 0usize;
    let mut f32_keys = 0usize;
    let mut words = 0usize;
    let (mut saturated, mut flushed, mut subnormal) = (0usize, 0usize, 0usize);
    for name in &uploaded {
        let a = streamed.get(name);
        let b = eager.get(name);
        assert_eq!(a.dtype, b.dtype, "dtype drift on `{name}`");
        assert_eq!(a.shape, b.shape, "shape drift on `{name}`");
        match a.dtype {
            DType::F16 => {
                let sa = gpu.download_f16_bits(a).expect("download streamed f16");
                let sb = gpu.download_f16_bits(b).expect("download eager f16");
                assert_eq!(
                    sa, sb,
                    "STREAMED WEIGHT DIFFERS FROM THE DEVICE CAST on `{name}` — the host \
                     f32_to_f16_rne and the device (_Float16) cast do not agree"
                );
                // Census of the range corners, so the fixture cannot quietly
                // stop exercising them (see `synth_bits`).
                for w in &sa {
                    match w & 0x7FFF {
                        0x7C00 => saturated += 1,
                        0x0000 => flushed += 1,
                        m if m < 0x0400 => subnormal += 1,
                        _ => {}
                    }
                }
                f16_keys += 1;
                words += sa.len();
            }
            _ => {
                let sa = gpu.download_f32(a).expect("download streamed f32");
                let sb = gpu.download_f32(b).expect("download eager f32");
                assert_eq!(
                    sa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    sb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "f32 key `{name}` differs"
                );
                f32_keys += 1;
                words += sa.len();
            }
        }
    }

    let freed = streamed.free_gpu(&mut gpu) + eager.free_gpu(&mut gpu);
    let _ = std::fs::remove_dir_all(&dir);

    println!("compared {f16_keys} f16 keys + {f32_keys} f32 keys ({words} elements)");
    println!("  range corners hit: {saturated} inf, {flushed} zero, {subnormal} subnormal");
    // A fixture that stopped producing out-of-range inputs would silently
    // narrow this gate to "rounding agrees", which is the easy half.
    assert!(
        saturated > 0 && flushed > 0 && subnormal > 0,
        "fixture no longer exercises the f16 range corners"
    );
    println!("freed {freed} device buffers");
    println!(
        "VmHWM  before stream: {rss_before_stream:.0} MiB  \
         after stream: {rss_after_stream:.0} MiB  \
         after eager f32 load: {rss_after_eager:.0} MiB"
    );
    println!("PASS: from_stream is bit-identical to from_host + device cast_f32_to_f16");
}
