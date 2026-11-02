// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity instrument for the EmbeddingGemma-2 audio leg: the log-mel frontend
//! and the tower's `[M, text_hidden]` rows for one WAV, read straight from the
//! checkpoint (a safetensors dir, so no `.vl` pack, no quantizer, no daemon
//! and no text tower are involved).
//!
//! `--layers N` builds the tower with only its first `N` conformer layers —
//! the loader reads exactly `0..N` — which is what makes a layer-by-layer
//! bisect against HF's `Gemma4AudioModel` possible without adding a dump hook
//! to the shipping forward. Comparing `N` from 0 to `num_hidden_layers`
//! localises a divergence to the layer that first moves the rows, and
//! `--layers 0` compares the subsample stack plus `output_proj` + embedder
//! alone.
//!
//! Writes `<out>.mel.f32` (`[frames, mel_bins]`, row-major) and
//! `<out>.rows.f32` (`[M, text_hidden]`), both little-endian f32, and prints
//! the shapes. The companion check is a few lines of numpy:
//!
//! ```text
//! cargo run -p hipfire-arch-embeddinggemma2 --features lab --example audio_lab \
//!   -- --model <hf-snapshot-dir> --wav /tmp/eg2-0.5s.wav --out /tmp/eg2-a0 --layers 3
//! ```
//!
//! The WAV is the wire format itself (16 kHz mono, PCM16 or f32), so the
//! samples the daemon would decode are the samples this instrument decodes.

use hipfire_arch_embeddinggemma2::audio::process_audio;
use hipfire_arch_embeddinggemma2::{
    load_audio_tower, parse_audio_config, AudioConfig, AudioFrontendConfig, AudioTower,
};
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::Gpu;
use std::fs;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn write_f32(path: &str, data: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").ok_or("--model <safetensors dir | .vl pack> is required")?;
    let wav = arg(&args, "--wav").ok_or("--wav <path> is required")?;
    let out = arg(&args, "--out").unwrap_or_else(|| "/tmp/eg2-audio-lab".into());
    // The audio config comes from the checkpoint's `config.json`, which is
    // beside the tensors for a dir source but NOT for a `.vl` pack (that
    // carries weights only) — hence `--config`.
    let cfg_path = arg(&args, "--config").unwrap_or_else(|| format!("{model}/config.json"));
    let layers: Option<usize> = arg(&args, "--layers")
        .map(|v| v.parse::<usize>().map_err(|e| format!("--layers {v}: {e}")))
        .transpose()?;

    let cfg_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&cfg_path).map_err(|e| format!("{cfg_path}: {e}"))?,
    )
    .map_err(|e| format!("{cfg_path}: {e}"))?;
    let mut cfg: AudioConfig =
        parse_audio_config(&cfg_json)?.ok_or("config.json declares no audio tower")?;
    if let Some(n) = layers {
        if n > cfg.num_layers {
            return Err(format!(
                "--layers {n} exceeds the checkpoint's {}",
                cfg.num_layers
            ));
        }
        cfg.num_layers = n;
    }
    let fe = AudioFrontendConfig::default();
    fe.validate()?;

    let clip = process_audio(&fs::read(&wav).map_err(|e| format!("{wav}: {e}"))?, &fe)?;
    println!(
        "wav {wav}: frames {} valid {} M {} (mel_bins {})",
        clip.plan.frames, clip.plan.valid_frames, clip.plan.n_tokens, fe.mel_bins
    );

    let text_hidden = cfg_json
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;

    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let src = ModelSource::from_path(&model)?;
    let tower: AudioTower = load_audio_tower(&mut gpu, &src, &cfg, fe.mel_bins, text_hidden)?;
    let t0 = std::time::Instant::now();
    let rows = tower.encode_audio(&mut gpu, &clip)?;
    println!(
        "layers {} -> {} rows x {} in {:.1} ms",
        cfg.num_layers,
        clip.plan.n_tokens,
        text_hidden,
        t0.elapsed().as_secs_f64() * 1e3
    );
    tower.free_gpu(&mut gpu);
    drop(src);

    write_f32(&format!("{out}.mel.f32"), &clip.mel)?;
    write_f32(&format!("{out}.rows.f32"), &rows)?;
    println!(
        "wrote {out}.mel.f32 ({} = {}x{}) and {out}.rows.f32 ({} = {}x{})",
        clip.mel.len(),
        clip.plan.frames,
        fe.mel_bins,
        rows.len(),
        clip.plan.n_tokens,
        text_hidden
    );
    Ok(())
}
