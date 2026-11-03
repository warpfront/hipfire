// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 4c standalone validation: end-to-end omni TTS — a text prompt →
//! spoken audio output (WAV).
//!
//! Loads the LFM3-Omni HFQ (text backbone + depth decoder) + the Liquid Codec,
//! prefills a ChatML prompt ending in `<|audio_start|>`, runs the omni generate
//! loop in AUDIO mode (LM decode → depth decoder MaskGIT → codebook_embedding
//! feedback → next LM step), collects audio frames until end-of-audio, decodes
//! them to a 24 kHz waveform, and writes a WAV.
//!
//! This is the core of the omni interleaved generate state machine (the daemon
//! `generate_lfm3_omni` route reuses this loop). AUDIO mode only here; the full
//! TEXT/AUDIO/INTERLEAVED state machine lands with the daemon route.
//!
//! Usage: `omni_tts [omni_hfq] [audio_tokenizer_dir] [prompt] [max_frames] [out.wav]`

use hipfire_arch_lfm3_omni::{
    codec_decode, load_codec_weights, omni_generate_audio, write_wav, CodecConfig, DepthRng,
};
use hipfire_loader::load_model;
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;

/// `<|audio_start|>` (config audio_start_token_id) — switches generation to speech.
const AUDIO_START_TOKEN: u32 = 125_012;

fn main() {
    let omni_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B.q8f16.hfq".to_string());
    let codec_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Omni-2B/audio_tokenizer".to_string());
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "Say hello in a friendly voice.".to_string());
    let max_frames: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let out_wav = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "/tmp/lfm3-omni-tts.wav".to_string());

    eprintln!("omni_tts: initializing GPU...");
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("  GPU: {}", gpu.arch);

    eprintln!("omni_tts: loading omni {omni_path}...");
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
    let depth_cfg = model.lfm3_depth_config().expect("depth config").clone();

    // ── Build the prompt: ChatML user turn + assistant prefix + <|audio_start|> ──
    let tok = model.tokenizer.as_ref().expect("tokenizer");
    let chatml =
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>");
    let mut prompt_tokens = tok.encode(&chatml);
    // The chat template prepends bos_token (<|startoftext|> = 124894); the
    // tokenizer's encode doesn't add it, so prepend it manually.
    prompt_tokens.insert(0, 124_894);
    prompt_tokens.push(AUDIO_START_TOKEN);
    eprintln!(
        "  prompt: {chatml:?} + <|audio_start|> → {} tokens: {:?}…",
        prompt_tokens.len(),
        &prompt_tokens[..prompt_tokens.len().min(12)]
    );

    // ── Omni generate loop (AUDIO mode) ──
    eprintln!(
        "omni_tts: generating audio (max {max_frames} frames, eos_code={})...",
        depth_cfg.audio_eos_code
    );
    let (frames, n_emitted, hit_eos) = {
        let bundle = match model.state.as_mut() {
            Some(s) => (s.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<hipfire_loader::Lfm3Bundle>()
                .expect("not lfm3"),
            None => panic!("not lfm3"),
        };
        let depth_weights = bundle.depth_weights.as_ref().expect("depth weights");
        let mut rng = DepthRng::new(42);
        omni_generate_audio(
            &bundle.config,
            &bundle.weights,
            &mut bundle.state,
            &depth_cfg,
            depth_weights,
            &mut gpu,
            &prompt_tokens,
            max_frames,
            &mut rng,
        )
        .unwrap_or_else(|e| panic!("omni_generate_audio: {e}"))
    };
    eprintln!(
        "  generated {n_emitted} audio frames (hit_eos={hit_eos}, {} codebooks each)",
        depth_cfg.num_codebooks
    );

    if frames.is_empty() {
        eprintln!("omni_tts: no audio frames generated — nothing to decode.");
        std::process::exit(1);
    }

    // ── Load the codec + decode ──
    eprintln!("omni_tts: loading codec + decoding {n_emitted} frames...");
    let codec_source =
        SafetensorsSource::open(std::path::Path::new(&codec_dir)).expect("open codec dir");
    let cfg_val: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{codec_dir}/config.json")).expect("read codec config"),
    )
    .expect("parse codec config");
    let codec_cfg = CodecConfig::from_config_value(&cfg_val);
    let codec_weights =
        load_codec_weights(&codec_source, &codec_cfg, &mut gpu).expect("load_codec_weights");
    let (waveform, sr) =
        codec_decode(&codec_cfg, &codec_weights, &mut gpu, &frames).expect("codec_decode");
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
    println!(
        "wav: {out_wav}  ({n_emitted} frames, {} samples, {:.2}s, eos={hit_eos})",
        waveform.len(),
        waveform.len() as f32 / sr as f32
    );

    let pass = finite && non_silent && waveform.len() == n_emitted * codec_cfg.frame_samples();
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
