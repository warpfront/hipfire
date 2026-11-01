// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Phase 4d standalone validation: streaming ASR — process an audio input in
//! chunks via `audio_forward_streaming` (the encoder state carries across
//! chunks) and compare to the batch `audio_forward` on the same audio.
//!
//! The chunked mel boundary (center=False, per-chunk) + the downsampler's
//! 8-mel-frame grouping mean the streaming and batch token counts can differ
//! slightly; the validation checks the streaming produces a plausible token
//! count + finite embeddings of the right dim.
//!
//! Uses a synthetic sine wave (no test audio fixtures in this env).
//!
//! Usage: `asr_stream_smoke [audio_hfq] [chunk_samples]`

use hipfire_loader::{load_model, ModelState};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use rdna_compute::Gpu;

fn main() {
    let audio_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/liquid-ai/LFM3-Audio-2B.q8f16.hfq".to_string());
    let chunk_samples: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000); // 0.5s @ 16 kHz

    eprintln!("asr_stream_smoke: initializing GPU...");
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("  GPU: {}", gpu.arch);

    eprintln!("asr_stream_smoke: loading audio model {audio_path}...");
    let mut model = load_model(
        &audio_path,
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
    let (audio_cfg, audio_weights) = match (&model.lfm3_audio_config, &model.lfm3_audio_weights) {
        (Some(c), Some(w)) => (c.clone(), w),
        _ => panic!("loaded model has no LFM3-Audio encoder"),
    };
    let sr = audio_cfg.mel.sample_rate;
    eprintln!("  audio model loaded (sample_rate={sr})");

    // ── Synthetic audio: a 1.5 s 440 Hz sine @ 16 kHz ──
    let n_samples = sr * 3 / 2;
    let waveform: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
        .collect();
    eprintln!("  synthetic audio: {n_samples} samples ({:.2}s @ {sr}Hz)", n_samples as f32 / sr as f32);

    // ── Batch: mel (padded) + audio_forward over the full audio ──
    let mel = hipfire_arch_lfm3_audio::mel::log_mel_from_waveform(
        &waveform,
        &audio_cfg.mel,
    );
    let num_frames = mel.len() / audio_cfg.mel.n_mels;
    eprintln!("  batch: {num_frames} mel frames...");
    let batch_tokens = hipfire_arch_lfm3_audio::lfm3_audio::audio_forward(
        &mut gpu,
        &audio_weights,
        &audio_cfg,
        &mel,
        num_frames,
    )
    .expect("batch audio_forward");
    let batch_n = batch_tokens.len() / audio_cfg.proj.out_features;
    let batch_finite = batch_tokens.iter().all(|v| v.is_finite());

    // ── Streaming: chunks of `chunk_samples`, encoder state carries ──
    let mut cache = hipfire_arch_lfm3_audio::lfm3_audio::AudioEncoderStreamingCache::new(
        &mut gpu,
        &audio_cfg,
        batch_n + 16,
    )
    .expect("AudioEncoderStreamingCache::new");
    let mut stream_tokens: Vec<f32> = Vec::new();
    let mut n_chunks = 0usize;
    let mut lo = 0usize;
    while lo < waveform.len() {
        let hi = (lo + chunk_samples).min(waveform.len());
        let chunk = &waveform[lo..hi];
        let is_final = hi == waveform.len();
        let toks = hipfire_arch_lfm3_audio::lfm3_audio::audio_forward_streaming(
            &mut gpu,
            &audio_weights,
            &audio_cfg,
            &mut cache,
            chunk,
            is_final,
        )
        .expect("audio_forward_streaming");
        if !toks.is_empty() {
            stream_tokens.extend_from_slice(&toks);
        }
        n_chunks += 1;
        lo = hi;
    }
    // ASR-STR-1: the is_final flush on the last chunk above now emits the
    // partial trailing 8-frame group as one final token (HF
    // `(num_frames + 7) // 8`); before the fix that tail was dropped and a
    // clip shorter than one group yielded 0 tokens. Assert the flush actually
    // closed the stream.
    assert!(cache.finished, "is_final chunk must finalize the stream cache");
    cache.free_gpu(&mut gpu);
    let stream_n = stream_tokens.len() / audio_cfg.proj.out_features;
    let stream_finite = stream_tokens.iter().all(|v| v.is_finite());

    eprintln!(
        "  batch:    {batch_n} tokens x {} dims (finite={batch_finite})",
        audio_cfg.proj.out_features
    );
    eprintln!(
        "  streaming: {stream_n} tokens over {n_chunks} chunks x {} dims (finite={stream_finite})",
        audio_cfg.proj.out_features
    );

    // ── Verify ──
    // ASR-STR-1: streaming (with the is_final flush) now matches the batch
    // token count exactly — both go through the same full-stream mel +
    // downsampler; the batch path pads the waveform (pad_length) which can
    // add one trailing frame, so allow ±1 token but require >0 and finite.
    let pass = batch_n > 0 && stream_n > 0 && batch_finite && stream_finite
        && batch_tokens.len() == batch_n * audio_cfg.proj.out_features
        && stream_tokens.len() == stream_n * audio_cfg.proj.out_features
        && (batch_n as i64 - stream_n as i64).abs() <= 1;
    println!(
        "batch_tokens={batch_n} stream_tokens={stream_n} chunks={n_chunks} dim={}",
        audio_cfg.proj.out_features
    );
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
