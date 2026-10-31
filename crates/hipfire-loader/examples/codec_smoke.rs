// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 4b standalone validation: load the LFM3-Omni HFQ (depth decoder +
//! text backbone) AND the Liquid Codec (from `audio_tokenizer/`), MaskGIT-sample
//! several audio frames, decode them to a 24 kHz waveform, and write a WAV.
//!
//! Validates the codec decode path (FSQ + 8-layer LFM3 decoder + iSTFT
//! overlap-add) produces a finite, non-silent waveform from real codes.
//!
//! Usage: `codec_smoke [omni_hfq] [audio_tokenizer_dir] [prompt] [n_frames] [out.wav]`

use hipfire_arch_lfm3::forward::decode_step;
use hipfire_arch_lfm3_omni::{
    codec_decode, load_codec_weights, sample_audio_frame, write_wav, CodecConfig, DepthRng,
};
use hipfire_loader::{load_model, ModelState};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;

fn main() {
    let omni_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B.q8f16.hfq".to_string());
    let codec_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B/audio_tokenizer".to_string());
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "Hello, how are you today?".to_string());
    let n_frames: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let out_wav = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "/tmp/lfm3-omni-codec-smoke.wav".to_string());

    eprintln!("codec_smoke: initializing GPU...");
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("  GPU: {}", gpu.arch);

    // ── Load the omni model (text backbone + depth decoder) ──
    eprintln!("codec_smoke: loading omni {omni_path}...");
    let mut model = load_model(
        &omni_path,
        2048,
        None,
        None,
        None,
        None,
        &CaskConfig::default(),
        1,
        SpecLoadCfg::default(),
        &mut gpu,
    )
    .expect("load_model");
    let depth_cfg = model.lfm3_depth_config.as_ref().expect("depth config");
    let depth_weights = model.lfm3_depth_weights.as_ref().expect("depth weights");

    // ── Prefill the text backbone → last_hidden_state ──
    let tokens = model.tokenizer.as_ref().expect("tokenizer").encode(&prompt);
    eprintln!("  prompt {prompt:?} → {} tokens", tokens.len());
    let hidden: Vec<f32> = {
        let bundle = match model.state.as_mut() {
            Some(ModelState::Lfm3(b)) => b,
            _ => panic!("not lfm3"),
        };
        for (i, &t) in tokens.iter().enumerate() {
            decode_step(&bundle.config, &bundle.weights, &mut bundle.state, &mut gpu, t, i as u32)
                .unwrap_or_else(|e| panic!("decode_step @ {i}: {e}"));
        }
        gpu.download_f32(&bundle.state.final_norm_buf).expect("download hidden")
    };
    eprintln!("  hidden finite={}", hidden.iter().all(|v| v.is_finite()));

    // ── MaskGIT-sample n_frames audio frames ──
    eprintln!("codec_smoke: sampling {n_frames} audio frames...");
    let mut rng = DepthRng::new(42);
    let mut all_codes: Vec<u32> = Vec::with_capacity(n_frames * depth_cfg.num_codebooks);
    let mut history: Vec<Vec<u32>> = Vec::new();
    for f in 0..n_frames {
        let codes =
            sample_audio_frame(depth_cfg, depth_weights, &mut gpu, &hidden, &history, &mut rng)
                .unwrap_or_else(|e| panic!("sample_audio_frame @ {f}: {e}"));
        history.push(codes.clone());
        all_codes.extend_from_slice(&codes);
    }
    eprintln!(
        "  sampled {} frames × {} codebooks (first frame: {:?})",
        n_frames,
        depth_cfg.num_codebooks,
        &all_codes[..depth_cfg.num_codebooks]
    );

    // ── Load the codec + decode ──
    eprintln!("codec_smoke: loading codec from {codec_dir}...");
    let codec_source =
        SafetensorsSource::open(std::path::Path::new(&codec_dir)).expect("open codec dir");
    let codec_config_json = std::fs::read_to_string(format!("{codec_dir}/config.json"))
        .expect("read codec config.json");
    let cfg_val: serde_json::Value =
        serde_json::from_str(&codec_config_json).expect("parse codec config.json");
    let codec_cfg = CodecConfig::from_config_value(&cfg_val);
    eprintln!(
        "  codec: {} layers, codebook_size {}, fsq {:?}, frame_samples {}",
        codec_cfg.num_hidden_layers,
        codec_cfg.codebook_size(),
        codec_cfg.fsq_levels,
        codec_cfg.frame_samples()
    );
    let codec_weights =
        load_codec_weights(&codec_source, &codec_cfg, &mut gpu).expect("load_codec_weights");

    eprintln!("codec_smoke: decoding {n_frames} frames → waveform...");
    let (waveform, sr) =
        codec_decode(&codec_cfg, &codec_weights, &mut gpu, &all_codes).expect("codec_decode");
    codec_weights.free_gpu(&mut gpu);

    // ── Verify + write WAV ──
    let finite = waveform.iter().all(|v| v.is_finite());
    let max_amp = waveform.iter().cloned().fold(0.0f32, f32::max);
    let min_amp = waveform.iter().cloned().fold(0.0f32, f32::min);
    let rms = (waveform.iter().map(|v| v * v).sum::<f32>() / waveform.len().max(1) as f32).sqrt();
    let non_silent = rms > 1e-4;

    write_wav(std::path::Path::new(&out_wav), &waveform, sr).expect("write_wav");
    eprintln!(
        "  waveform: {} samples ({:.2}s @ {sr}Hz), finite={finite}, min={min_amp:.4}, max={max_amp:.4}, rms={rms:.4}, non_silent={non_silent}",
        waveform.len(),
        waveform.len() as f32 / sr as f32
    );
    println!("wav: {out_wav}  ({n_frames} frames, {} samples)", waveform.len());

    let pass = finite && non_silent && waveform.len() == n_frames * codec_cfg.frame_samples();
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
