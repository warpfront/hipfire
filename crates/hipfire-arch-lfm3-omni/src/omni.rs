// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni generate loop (Phase 4c): the interleaved audio/text state
//! machine. AUDIO mode here (TEXT / INTERLEAVED land with the daemon route).
//!
//! Ground truth: transformers `modeling_lfm3_omni.py`
//! `Lfm3OmniForConditionalGeneration._generate` (read 2026-10-31).
//!
//! AUDIO-mode loop:
//!   1. Prefill the prompt (ChatML + BOS + `<|audio_start|>`) via `decode_step`.
//!   2. `hidden = last_hidden_state` (`state.final_norm_buf`) at the
//!      `<|audio_start|>` position.
//!   3. For each frame: `sample_audio_frame(hidden)` → 12 codes →
//!      `get_audio_out_embeds(codes)` → `decode_step_with_embed` at the next
//!      position → new `hidden`. Collect frames until `audio_eos_code` (2016)
//!      in any codebook or `max_frames`.
//!
//! Returns the collected codes `[n_frames * num_codebooks]`, the frame count,
//! and whether end-of-audio was hit. The caller runs `codec_decode` on the
//! codes to produce the 24 kHz waveform.

use crate::depth::{get_audio_out_embeds, sample_audio_frame, DepthConfig, DepthRng, DepthWeights};
use hipfire_arch_lfm3::{decode_step, decode_step_with_embed, Lfm3Config, Lfm3State, Lfm3Weights};
use rdna_compute::Gpu;

/// Run the omni AUDIO-mode generate loop. `prompt_tokens` must end with the
/// `<|audio_start|>` token (125012). Returns `(codes, n_frames, hit_eos)`.
pub fn omni_generate_audio(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    depth_cfg: &DepthConfig,
    depth_weights: &DepthWeights,
    gpu: &mut Gpu,
    prompt_tokens: &[u32],
    max_frames: usize,
    rng: &mut DepthRng,
) -> Result<(Vec<u32>, usize, bool), String> {
    let eos_code = depth_cfg.audio_eos_code;

    // ── Prefill ──
    for (i, &t) in prompt_tokens.iter().enumerate() {
        decode_step(cfg, weights, state, gpu, t, i as u32)
            .map_err(|e| format!("omni: prefill decode_step @ {i}: {e}"))?;
    }
    let mut position = prompt_tokens.len() as u32 - 1;
    let mut hidden = gpu
        .download_f32(&state.final_norm_buf)
        .map_err(|e| format!("omni: download hidden: {e:?}"))?;

    // ── Audio loop ──
    let mut history: Vec<Vec<u32>> = Vec::new();
    let mut frames: Vec<u32> = Vec::new();
    let mut n_emitted = 0usize;
    let mut hit_eos = false;
    for step in 0..max_frames {
        let codes = sample_audio_frame(depth_cfg, depth_weights, gpu, &hidden, &history, rng)
            .map_err(|e| format!("omni: sample_audio_frame @ {step}: {e}"))?;
        let is_eos = codes.iter().any(|&c| c == eos_code);
        if is_eos {
            hit_eos = true;
            break;
        }
        history.push(codes.clone());
        frames.extend_from_slice(&codes);
        n_emitted += 1;

        // Feed the codes back to the LM via the shared codebook_embedding.
        let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &codes);
        position += 1;
        decode_step_with_embed(cfg, weights, state, gpu, &next_embed, position)
            .map_err(|e| format!("omni: decode_step_with_embed @ {step}: {e}"))?;
        hidden = gpu
            .download_f32(&state.final_norm_buf)
            .map_err(|e| format!("omni: download hidden: {e:?}"))?;
    }
    Ok((frames, n_emitted, hit_eos))
}
