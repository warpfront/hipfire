// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Diagnostic vision-tower-only harness for the Qwen3.5/3.8-VL sidecar
//! (`qwen3.8-27b-vision.hfq`; no text trunk is loaded).
//!
//! ```text
//! cargo run --release -p hipfire-arch-qwen35-vl --features lab --example vl_tower_dump -- \
//!   --model qwen3.8-27b-vision.hfq (--patches pixel_values.bin --grid-h H --grid-w W | --image IMG) \
//!   --out DIR [--reps R] [--stages 1] [--residency resident|host-mapped|streamed]
//! ```
//!
//! * `--patches` is raw little-endian F32 `[grid_h*grid_w, 1536]` in HF
//!   `pixel_values` order (written by `benchmarks/vision/dump_qwen35_tower_reference.py`).
//! * `--image` runs the product preprocessing and also writes `DIR/patches.bin`.
//! * One warm-up `vision_forward` (kernel JIT), then `--reps` (default 3) timed
//!   runs. `DIR/post_merger.bin` (F32 `[n/4, out_hidden]`) comes from the last
//!   run; `DIR/receipt.json` records sha256 of the model, grid, per-rep ms,
//!   median ms, arch, the tower residency and the `HIPFIRE_VIT_ATTN` /
//!   `HIPFIRE_VIT_GEMM` selectors.
//! * `--residency` places the tower's large matrices: `streamed` (default, the
//!   product's `vision.mode` auto/on placement: streamed per block from the
//!   sidecar through two VRAM staging slots) or `resident` (whole tower in
//!   VRAM). Both run the same kernels on the same weight bytes.
//! * `--stages 1` additionally produces the per-block stage dumps
//!   (`HIPFIRE_VL_DUMP_DIR`) in DIR. The dump directory is latched in the
//!   process config snapshot at first use, so enabling it for only one
//!   untimed run needs a fresh process: the timed process re-executes itself
//!   (internal `--dump-child 1`) with the env set, running a single untimed
//!   forward. Stage dumps force device syncs and never contribute to timings.

use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl::{
    load_vision_weights, vision_config_from_hfq, vision_forward, TowerResidency,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: vl_tower_dump --model VISION.hfq \
(--patches F32.bin --grid-h H --grid-w W | --image IMG) --out DIR [--reps R] [--stages 1] \
[--residency resident|host-mapped|streamed]";

fn digest(path: &Path) -> Res<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Selector env values as the receipt records them; unset means "auto".
fn selector(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v,
        _ => "auto".to_string(),
    }
}

fn main() -> Res<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if raw.is_empty() || raw.len() % 2 != 0 {
        return Err(USAGE.into());
    }
    let flags: BTreeMap<&str, &str> = raw
        .chunks_exact(2)
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    let get = |key: &str| -> Res<&str> {
        flags
            .get(key)
            .copied()
            .ok_or_else(|| format!("missing {key}; {USAGE}").into())
    };
    let model_path = Path::new(get("--model")?);
    let out = Path::new(get("--out")?);
    let reps: usize = match flags.get("--reps") {
        Some(v) => v.parse()?,
        None => 3,
    };
    if reps == 0 {
        return Err("--reps must be >= 1".into());
    }
    let stages = match flags.get("--stages") {
        Some(&"1") => true,
        Some(&"0") | None => false,
        Some(other) => return Err(format!("--stages takes 0 or 1, got {other}").into()),
    };
    let residency = match flags.get("--residency") {
        Some(&"resident") => TowerResidency::Resident,
        Some(&"host-mapped") => TowerResidency::HostMapped,
        Some(&"streamed") | None => TowerResidency::Streamed,
        Some(other) => {
            return Err(
                format!("--residency takes resident, host-mapped or streamed, got {other}").into(),
            )
        }
    };
    let dump_child = flags.get("--dump-child") == Some(&"1");
    fs::create_dir_all(out)?;
    if dump_child {
        // Latched by the process config snapshot on first read, i.e. before
        // HfqFile/Gpu initialisation below.
        std::env::set_var("HIPFIRE_VL_DUMP_DIR", out);
    }

    let hfq = HfqFile::open(model_path)?;
    let config = vision_config_from_hfq(&hfq).ok_or("no vision_config in sidecar metadata")?;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;

    let (patches, grid_h, grid_w, input_path, preprocessing) =
        if let Some(path) = flags.get("--image") {
            let (chw, height, width) = image::load_and_preprocess(
                Path::new(path),
                config.patch_size,
                config.spatial_merge_size,
            )?;
            let patches = image::extract_patches(
                &chw,
                3,
                height,
                width,
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size,
            );
            if !dump_child {
                fs::write(out.join("patches.bin"), f32_bytes(&patches))?;
            }
            (
                patches,
                height / config.patch_size,
                width / config.patch_size,
                Path::new(path),
                "hipfire_image",
            )
        } else {
            let path = Path::new(get("--patches")?);
            let bytes = fs::read(path)?;
            if bytes.len() % 4 != 0 {
                return Err("patch payload must contain whole F32 elements".into());
            }
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            (
                values,
                get("--grid-h")?.parse::<usize>()?,
                get("--grid-w")?.parse::<usize>()?,
                path,
                "external_patches",
            )
        };
    let n_patches = grid_h.checked_mul(grid_w).ok_or("patch grid overflow")?;
    let merge = config.spatial_merge_size;
    if n_patches == 0
        || grid_h % merge != 0
        || grid_w % merge != 0
        || patches.len() != n_patches * patch_dim
        || patches.iter().any(|v| !v.is_finite())
    {
        return Err(format!(
            "invalid patch grid or payload: grid {grid_h}x{grid_w}, {} floats, expected {}",
            patches.len(),
            n_patches * patch_dim
        )
        .into());
    }

    let mut gpu = Gpu::init()?;
    let arch = gpu.arch.clone();
    let weights = load_vision_weights(&hfq, &config, &mut gpu, residency)?;
    assert_eq!(weights.residency(), residency, "loader ignored placement");

    if dump_child {
        // Single untimed run; per-stage dumps land in `out` via HIPFIRE_VL_DUMP_DIR.
        let result = vision_forward(&mut gpu, &weights, &config, &patches, grid_h, grid_w);
        weights.free_gpu(&mut gpu);
        result?;
        return Ok(());
    }

    let mut last: Vec<f32> = Vec::new();
    let mut rep_ms: Vec<f64> = Vec::with_capacity(reps);
    let mut failure: Option<Box<dyn std::error::Error>> = None;
    // Warm-up (includes kernel JIT), then timed reps.
    for rep in 0..=reps {
        let started = Instant::now();
        match vision_forward(&mut gpu, &weights, &config, &patches, grid_h, grid_w) {
            Ok(output) => {
                gpu.hip.device_synchronize()?;
                let ms = started.elapsed().as_secs_f64() * 1e3;
                if rep == 0 {
                    eprintln!("[vl_tower_dump] warm-up {ms:.1} ms");
                } else {
                    eprintln!("[vl_tower_dump] rep {rep}/{reps} {ms:.1} ms");
                    rep_ms.push(ms);
                }
                if output.iter().any(|v| !v.is_finite()) || !output.iter().any(|v| *v != 0.0) {
                    failure = Some("non-finite or unwritten tower output".into());
                    break;
                }
                if !last.is_empty()
                    && !output
                        .iter()
                        .zip(&last)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    failure = Some("tower output changed across identical image requests".into());
                    break;
                }
                last = output;
            }
            Err(e) => {
                failure = Some(e.into());
                break;
            }
        }
    }
    weights.free_gpu(&mut gpu);
    if let Some(e) = failure {
        return Err(e);
    }
    let merged = n_patches / (merge * merge);
    if last.len() != merged * config.out_hidden_size || last.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "invalid tower output: {} floats, expected {}x{}",
            last.len(),
            merged,
            config.out_hidden_size
        )
        .into());
    }
    fs::write(out.join("post_merger.bin"), f32_bytes(&last))?;
    fs::write(
        out.join("post_merger.json"),
        json!({"shape": [merged, config.out_hidden_size], "dtype": "<f4"}).to_string(),
    )?;

    let mut sorted = rep_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    };

    if stages {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(&raw)
            .args(["--dump-child", "1"])
            .status()?;
        if !status.success() {
            return Err(format!("stage-dump child failed: {status}").into());
        }
    }

    let receipt = json!({
        "schema": "hipfire.qwen35vl.tower_dump.v1",
        "model_sha256": digest(model_path)?,
        "input_sha256": digest(input_path)?,
        "preprocessing": preprocessing,
        "patches_sha256": {
            "patches_bin": if flags.contains_key("--image") {
                Some(digest(&out.join("patches.bin"))?)
            } else {
                None
            },
        },
        "post_merger_sha256": digest(&out.join("post_merger.bin"))?,
        "arch": arch,
        "grid_thw": [[1, grid_h, grid_w]],
        "n_patches": n_patches,
        "post_merger_shape": [merged, config.out_hidden_size],
        "reps": reps,
        "rep_ms": rep_ms,
        "median_ms": median,
        "stages": stages,
        "residency": match residency {
            TowerResidency::Resident => "resident",
            TowerResidency::HostMapped => "host-mapped",
            TowerResidency::Streamed => "streamed",
        },
        "HIPFIRE_VIT_ATTN": selector("HIPFIRE_VIT_ATTN"),
        "HIPFIRE_VIT_GEMM": selector("HIPFIRE_VIT_GEMM"),
    });
    fs::write(
        out.join("receipt.json"),
        serde_json::to_string_pretty(&receipt)? + "\n",
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
