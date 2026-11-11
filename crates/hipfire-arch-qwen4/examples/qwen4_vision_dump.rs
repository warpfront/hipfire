// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Diagnostic-only vision tower, without loading the Flash Next text trunk.
//! cargo run --release -p hipfire-arch-qwen4 --features lab --example qwen4_vision_dump -- \
//!   --model vision.hfq --patches pixel_values.bin --grid-h 14 --grid-w 18 --out dump
//! `--image image.png` instead of `--patches/--grid-h/--grid-w` also exercises
//! the product image preprocessing. Stage dumps force synchronization; elapsed
//! times from this tool are not production performance evidence.

use hipfire_arch_qwen35_vl::qwen35_vl::vision_forward;
use hipfire_arch_qwen4::{image, vision};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

fn digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err("usage: --model VISION.hfq (--image IMAGE | --patches F32.bin --grid-h H --grid-w W) --out DIR".into());
    }
    let flags: BTreeMap<&str, &str> = args
        .chunks_exact(2)
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    let get = |key| {
        flags
            .get(key)
            .copied()
            .ok_or_else(|| format!("missing {key}"))
    };
    let model_path = Path::new(get("--model")?);
    let out = Path::new(get("--out")?);
    fs::create_dir_all(out)?;
    // Set before HfqFile/Gpu initialize the process developer-env snapshot.
    std::env::set_var("HIPFIRE_VL_DUMP_DIR", out);
    let hfq = HfqFile::open(model_path)?;
    let config = vision::config_from_hfq(&hfq)?;
    vision::validate_hfq(&hfq, &config)?;
    let (patches, grid_h, grid_w, input_path, preprocessing) =
        if let Some(path) = flags.get("--image") {
            let (chw, height, width) = image::load_and_preprocess(
                Path::new(path),
                config.patch_size,
                config.spatial_merge_size,
            )?;
            (
                image::extract_patches(
                    &chw,
                    3,
                    height,
                    width,
                    config.patch_size,
                    config.temporal_patch_size,
                    config.spatial_merge_size,
                ),
                height / config.patch_size,
                width / config.patch_size,
                Path::new(path),
                "hipfire",
            )
        } else {
            let path = Path::new(get("--patches")?);
            let bytes = fs::read(path)?;
            if !bytes.len().is_multiple_of(4) {
                return Err("patch payload must contain whole F32 elements".into());
            }
            let values = bytes
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
    let count = grid_h.checked_mul(grid_w).ok_or("patch grid overflow")?;
    if count == 0
        || count > 16128
        || !grid_h.is_multiple_of(2)
        || !grid_w.is_multiple_of(2)
        || patches.len() != count * 1536
        || patches.iter().any(|v| !v.is_finite())
    {
        return Err("invalid patch grid or payload".into());
    }
    let patch_bytes: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(out.join("pixel_values.bin"), &patch_bytes)?;
    fs::write(
        out.join("pixel_values.json"),
        json!({"shape": [count, 1536], "dtype": "<f4"}).to_string(),
    )?;
    let mut gpu = Gpu::init()?;
    let arch = gpu.arch.clone();
    let weights = vision::load_weights(&hfq, &config, &mut gpu)?;
    let started = std::time::Instant::now();
    let result = vision_forward(&mut gpu, &weights, &config, &patches, grid_h, grid_w);
    weights.free_gpu(&mut gpu);
    let output = result?;
    let seconds = started.elapsed().as_secs_f64();
    if output.len() != count / 4 * 2560 || output.iter().any(|v| !v.is_finite()) {
        return Err("invalid projected tower output".into());
    }
    let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(out.join("post_merger.bin"), bytes)?;
    fs::write(
        out.join("post_merger.json"),
        json!({"shape": [count / 4, 2560], "dtype": "<f4"}).to_string(),
    )?;
    let receipt = json!({
        "schema": "hipfire.qwen4.vision_dump.v1", "status": "diagnostic_complete",
        "model_sha256": digest(model_path)?, "input_sha256": digest(input_path)?,
        "binary_sha256": digest(&std::env::current_exe()?)?,
        "patches_sha256": digest(&out.join("pixel_values.bin"))?,
        "projected_sha256": digest(&out.join("post_merger.bin"))?,
        "arch": arch, "preprocessing": preprocessing, "grid_thw": [[1, grid_h, grid_w]],
        "projected_shape": [count / 4, 2560], "dump_forward_seconds": seconds,
        "quality_acceptance": false, "performance_evidence": false,
    });
    fs::write(
        out.join("receipt.json"),
        serde_json::to_string_pretty(&receipt)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
