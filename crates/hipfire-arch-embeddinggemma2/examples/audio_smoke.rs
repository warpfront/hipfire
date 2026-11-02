// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU smoke for the EmbeddingGemma-2 audio leg: the whole forward (mel
//! frontend → subsample conv stack → 2 synthetic conformer layers →
//! projectors) on a hand-made mel and on a real 0.5 s sine WAV, with
//! synthetic weights.
//!
//! It asserts SHAPE and FINITENESS only — numerical parity against the
//! float32 transformers reference is the integration slice's job, not this
//! example's. Run with `cargo run -p hipfire-arch-embeddinggemma2
//! --features lab --example audio_smoke`.

use hipfire_arch_embeddinggemma2::audio::{plan, process_audio, ProcessedAudio};
use hipfire_arch_embeddinggemma2::{AudioConfig, AudioFrontendConfig, AudioTower};
use rdna_compute::Gpu;

/// A deliberately small tower: the kernels are all shape-generic, and 2
/// layers × hidden 128 keeps the smoke to milliseconds. The geometry has to
/// stay self-consistent: 16 mel bins → two stride-2 convs → 4 frequencies,
/// and 4 × 32 channels = the 128-wide hidden.
fn toy_config() -> AudioConfig {
    AudioConfig {
        hidden_size: 128,
        num_layers: 2,
        n_heads: 2,
        head_dim: 64,
        rms_norm_eps: 1e-6,
        output_proj_dims: 256,
        conv_kernel_size: 5,
        attention_chunk_size: 12,
        attention_context_left: 13,
        attention_context_right: 0,
        attention_logit_cap: 50.0,
        attention_invalid_logits_value: -1e9,
        residual_weight: 0.5,
        subsampling_conv_channels: [16, 32],
        use_clipped_linears: true,
        gradient_clipping: 1e10,
    }
}

fn toy_frontend() -> AudioFrontendConfig {
    AudioFrontendConfig {
        mel_bins: 16,
        ..AudioFrontendConfig::default()
    }
}

/// A 16 kHz mono WAV around `pcm` (PCM16), the shape a caller must send.
fn wav16(pcm: &[f32]) -> Vec<u8> {
    let data_len = pcm.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for v in pcm {
        let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e:?}"))?;
    let cfg = toy_config();
    let fe = toy_frontend();
    let text_hidden = 512;
    let tower = AudioTower::synthetic(&mut gpu, &cfg, fe.mel_bins, text_hidden, 0xA0D10)?;

    // (1) A hand-built clip: 0.1 s plans to 10 mel frames, 9 real, M = 3.
    let hand_plan = plan(&fe, 16_000 / 10);
    assert_eq!(
        (hand_plan.frames, hand_plan.valid_frames, hand_plan.n_tokens),
        (10, 9, 3)
    );
    let mut mel = vec![0.0f32; hand_plan.frames * fe.mel_bins];
    for (i, v) in mel.iter_mut().enumerate() {
        // A deterministic, bounded pattern; the invalid frame stays zero.
        if i < hand_plan.valid_frames * fe.mel_bins {
            *v = ((i % 37) as f32) / 37.0 - 0.5;
        }
    }
    let hand = ProcessedAudio {
        mel,
        plan: hand_plan,
    };
    let t0 = std::time::Instant::now();
    let rows = tower.encode_audio(&mut gpu, &hand)?;
    let finite = rows.iter().all(|v| v.is_finite());
    println!(
        "  hand-built 0.1 s clip -> {} mel frames, M {} -> {} rows x {} in {:.1} ms \
         (finite: {finite})",
        hand.plan.frames,
        hand.plan.n_tokens,
        hand.plan.n_tokens,
        text_hidden,
        t0.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(rows.len(), hand.plan.n_tokens * text_hidden);
    assert!(finite, "the tower produced a non-finite row");
    assert!(
        rows.iter().any(|v| *v != 0.0),
        "the tower produced all zeros"
    );

    // (2) A real WAV through the frontend and the tower: 0.5 s → M = 13.
    let pcm: Vec<f32> = (0..8_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.8)
        .collect();
    let clip = process_audio(&wav16(&pcm), &fe)?;
    assert_eq!(clip.plan.frames, 50);
    assert_eq!(clip.plan.n_tokens, 13);
    let t1 = std::time::Instant::now();
    let rows = tower.encode_audio(&mut gpu, &clip)?;
    let finite = rows.iter().all(|v| v.is_finite());
    println!(
        "  0.5 s 440 Hz WAV -> {} mel frames, M {} -> {} rows x {} in {:.1} ms (finite: \
         {finite})",
        clip.plan.frames,
        clip.plan.n_tokens,
        clip.plan.n_tokens,
        text_hidden,
        t1.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(rows.len(), clip.plan.n_tokens * text_hidden);
    assert!(finite, "the tower produced a non-finite row");
    assert!(
        rows.iter().any(|v| *v != 0.0),
        "the tower produced all zeros"
    );

    tower.free_gpu(&mut gpu);
    println!("audio smoke ok");
    Ok(())
}
