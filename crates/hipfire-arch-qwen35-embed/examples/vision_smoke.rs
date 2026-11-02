// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision-path smoke test for arch 17: load ONLY the SigLIP tower out of the
//! embedding checkpoint's directory and encode one image — no text model.
//!
//! Why isolated: on a desktop card the text model (~500 MB f16) plus the
//! tower (~240 MB) can fill VRAM, and the tower *forward* then OOMs. Running
//! the tower alone fits and is the direct test of the arch crate's own vision
//! code — the dir-based `visual.*` weight loader, `extract_patches` wiring,
//! and the `vision_forward` invocation. The tower math itself is the shared
//! `hipfire-arch-qwen35-vl` code, already validated against HF on gfx1100
//! (`benchmarks/vision/comparison-2026-05-23.md`).
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35-embed --example vision_smoke -- \
//!       <model_dir> <image.png>

use hipfire_arch_qwen35_embed::config::{config_from_metadata, parse_config};
use hipfire_arch_qwen35_embed::vision::{encode_image, load_vision_weights};
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .ok_or("usage: vision_smoke <model_dir> <image.png>")?;
    let image = args
        .get(2)
        .ok_or("usage: vision_smoke <model_dir> <image.png>")?;

    let src = ModelSource::from_path(dir)?;
    let st: &dyn hipfire_runtime::model_source::ModelSource = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    let meta: serde_json::Value =
        serde_json::from_str(st.metadata_json()).map_err(|e| e.to_string())?;
    let cfg = parse_config(config_from_metadata(&meta))?;
    let v = cfg
        .vision
        .as_ref()
        .ok_or("no vision_config in checkpoint")?;
    println!(
        "vision_config: hidden {} depth {} heads {} patch {} merge {} out {}",
        v.hidden_size,
        v.num_layers,
        v.num_heads,
        v.patch_size,
        v.spatial_merge_size,
        v.out_hidden_size,
    );

    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    println!("gpu: {}", gpu.arch);

    let weights = load_vision_weights(&mut gpu, st, &cfg)?;
    let (f, t) = gpu.hip.get_vram_info().unwrap_or((0, 0));
    println!(
        "tower weights loaded; VRAM {} MB free / {} MB",
        f / 1048576,
        t / 1048576
    );

    let bytes = std::fs::read(image).map_err(|e| format!("{image}: {e}"))?;
    let (merged, grid_h, grid_w) = encode_image(&mut gpu, &weights, &cfg, &bytes)?;
    let out_h = v.out_hidden_size;
    let n = merged.len() / out_h;
    let absmax = merged.iter().fold(0f32, |a, x| a.max(x.abs()));
    let bad = merged.iter().filter(|x| !x.is_finite()).count();
    // `encode_image` returns the PRE-merge patch grid; the merged token
    // count is `grid / spatial_merge_size` per side.
    let sms = v.spatial_merge_size.max(1);
    println!(
        "image -> {n} merged tokens ({}x{} patch grid, {}x{} merged), out_hidden {out_h}, absmax {absmax:.4}, nonfinite {bad}",
        grid_w, grid_h, grid_w / sms, grid_h / sms
    );
    weights.free_gpu(&mut gpu);
    if n == 0 || bad > 0 {
        return Err("vision smoke FAILED".into());
    }
    println!("vision smoke PASSED");
    Ok(())
}
