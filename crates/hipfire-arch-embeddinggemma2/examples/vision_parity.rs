// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Differential instrument for the EmbeddingGemma-2 vision leg: dumps exactly
//! what hipfire feeds the tower and what the tower returns, so it can be
//! diffed against the float32 transformers reference *without* the trunk, the
//! daemon or the `.vl` quantization in the way.
//!
//! It loads the real vision tower from the model's `.vl` sidecar (the same
//! `load_vision_tower` the carrier calls), runs `encode_image` on one PNG at a
//! time and writes, per PNG, next to `<out-dir>`:
//!
//! - `<stem>.json` — geometry (`grid_h`, `grid_w`, `n_patches`, `n_tokens`,
//!   `rows`, `patch_pixels`), the same fields HF's processor reports.
//! - `<stem>.patches.f32` — `[n_patches, patch_pixels]` f32, the *processor*
//!   output (compare against HF `AutoProcessor`'s `pixel_values`).
//! - `<stem>.positions.i32` — `[n_patches, 2]` f32-order `(x, y)` (compare
//!   against `image_position_ids`).
//! - `<stem>.rows.f32` — `[n_tokens, text_hidden]` f32, the projected rows
//!   `encode_image` produced (compare against HF `vision_tower` + the
//!   `embed_vision` projection).
//!
//! `scripts/check_embeddinggemma2_vision_parity.py` drives it and prints the
//! per-element diffs; run it when a media leg drifts and the oracle's
//! end-to-end rel-L2 cannot say whether the processor or the tower is at
//! fault.
//!
//! ```
//! cargo run --release -p hipfire-arch-embeddinggemma2 --features lab \
//!   --example vision_parity -- <model-dir-with-vl-sidecar> <out-dir> <png>...
//! ```

use hipfire_arch_embeddinggemma2::config::{
    config_from_metadata, parse_config, parse_vision_config,
};
use hipfire_arch_embeddinggemma2::image::process_image;
use hipfire_arch_embeddinggemma2::vision::load_vision_tower;
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::Gpu;
use std::path::Path;

fn write_f32(path: &Path, v: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_i32(path: &Path, v: &[i32]) -> Result<(), String> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: vision_parity <model-dir> <out-dir> <png>...".into());
    }
    let (model_dir, out_dir) = (&args[0], &args[1]);
    let pngs = &args[2..];

    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{model_dir}/config.json"))
            .map_err(|e| format!("read {model_dir}/config.json: {e}"))?,
    )
    .map_err(|e| format!("parse {model_dir}/config.json: {e}"))?;
    let cfg = parse_config(config_from_metadata(&meta))?;
    let vcfg = parse_vision_config(config_from_metadata(&meta))?
        .ok_or("this checkpoint declares no vision_config")?;

    let sidecar = hipfire_runtime::sidecar::resolve_vl_sidecar(model_dir).ok_or_else(|| {
        format!("{model_dir} has no `.vl` sidecar (a sibling `<dir>.vl` or HIPFIRE_VL_FILE)")
    })?;
    eprintln!("vision_parity: tower from {}", sidecar.display());

    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let src = ModelSource::from_path(&sidecar.to_string_lossy())
        .map_err(|e| format!("open sidecar: {e}"))?;
    let tower = load_vision_tower(&mut gpu, &src, &vcfg, cfg.hidden_size)?;

    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {out_dir}: {e}"))?;
    for png in pngs {
        let bytes = std::fs::read(png).map_err(|e| format!("read {png}: {e}"))?;
        let img = process_image(&bytes, &vcfg)?;
        let t0 = std::time::Instant::now();
        let rows = tower.encode_image(&mut gpu, &img)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        let stem = Path::new(png)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("{png} has no file stem"))?;
        let rows_n = rows.len() / cfg.hidden_size;
        let json = serde_json::json!({
            "png": png,
            "grid_h": img.grid_h,
            "grid_w": img.grid_w,
            "n_patches": img.n_patches,
            "n_tokens": img.n_tokens,
            "rows": rows_n,
            "patch_pixels": vcfg.patch_pixels(),
            "text_hidden": cfg.hidden_size,
            "vision_hidden": vcfg.hidden_size,
            "pooling_kernel_size": vcfg.pooling_kernel_size,
        });
        let base = Path::new(out_dir).join(stem);
        std::fs::write(
            base.with_extension("json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .map_err(|e| format!("write json: {e}"))?;
        write_f32(&base.with_extension("patches.f32"), &img.patches)?;
        write_i32(&base.with_extension("positions.i32"), &img.positions)?;
        write_f32(&base.with_extension("rows.f32"), &rows)?;

        println!(
            "{stem}: {h}x{w} grid, {n_patches} patches, {n_tokens} soft -> {rows_n} rows x {th} \
             in {ms:.1} ms (patch mean {mean:.6})",
            h = img.grid_h,
            w = img.grid_w,
            n_patches = img.n_patches,
            n_tokens = img.n_tokens,
            th = cfg.hidden_size,
            mean = img.patches.iter().map(|v| *v as f64).sum::<f64>() / img.patches.len() as f64,
        );
    }
    tower.free_gpu(&mut gpu);
    Ok(())
}
