// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni generate loop (Phase 4c): the interleaved audio/text state
//! machine. AUDIO + INTERLEAVED modes.
//!
//! Ground truth: transformers `modeling_lfm3_omni.py`
//! `Lfm3OmniForConditionalGeneration._generate` (read 2026-10-31).
//!
//! AUDIO mode (prompt ends with `<|audio_start|>`): MaskGIT-sample audio
//! frames, feeding each frame's `codebook_embedding` back to the LM, until
//! `audio_eos_code` or `max_frames`.
//!
//! INTERLEAVED mode (prompt ends with `<|mixed_start|>`): alternate text
//! blocks (`interleaved_n_text` = 6 tokens, greedy argmax) and audio blocks
//! (`interleaved_n_audio` = 12 frames, MaskGIT). `<|text_end|>` ends the text
//! and starts the audio tail (until eos / max). The block alternation uses
//! FIXED block lengths — it does NOT rely on the (currently non-firing) eos;
//! only the audio tail does, and that is bounded by `max_steps`.

use crate::depth::{get_audio_out_embeds, sample_audio_frame, DepthConfig, DepthRng, DepthWeights};
use hipfire_arch_lfm3::{decode_step, decode_step_with_embed, Lfm3Config, Lfm3State, Lfm3Weights};
use rdna_compute::Gpu;

/// Hidden-state checksum for the batch-vs-streaming divergence bisect
/// (HIPFIRE_OMNI_DEBUG=1): a cheap order-sensitive fold over the hidden.
fn hidden_cksum(hidden: &[f32]) -> f64 {
    let mut acc = 0x9E3779B97F4A7C15u64.wrapping_add(hidden.len() as u64);
    for (i, &v) in hidden.iter().enumerate() {
        acc = acc.wrapping_add((v.to_bits() as u64).wrapping_mul((i as u64) | 1));
    }
    (acc >> 11) as f64
}

fn omni_debug() -> bool {
    std::env::var("HIPFIRE_OMNI_DEBUG").is_ok()
}

// ─── Omni token ids (HF `Lfm3OmniConfig` defaults) ───────────────────────────
pub const AUDIO_START_TOKEN: u32 = 125_012;
pub const AUDIO_OUT_TOKEN: u32 = 125_032;
pub const TEXT_END_TOKEN: u32 = 125_014;
pub const MIXED_START_TOKEN: u32 = 125_015;
pub const IM_END_TOKEN: u32 = 124_900; // <|im_end|> (eos)
/// Interleaved block sizes (HF `interleaved_n_text` / `interleaved_n_audio`).
pub const INTERLEAVED_N_TEXT: usize = 6;
pub const INTERLEAVED_N_AUDIO: usize = 12;

/// Omni generation output.
pub struct OmniOutput {
    /// Generated text tokens (text slots + one `AUDIO_OUT_TOKEN` per audio frame).
    pub text_tokens: Vec<u32>,
    /// Audio codes `[n_audio_frames * num_codebooks]` (caller runs `codec_decode`).
    pub audio_codes: Vec<u32>,
    pub n_audio_frames: usize,
    pub hit_eos: bool,
    pub finished: bool,
}

/// Run the omni generate loop. The mode is inferred from the prompt's last
/// token: `<|audio_start|>` → AUDIO, `<|mixed_start|>` → INTERLEAVED.
pub fn omni_generate(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    depth_cfg: &DepthConfig,
    depth_weights: &DepthWeights,
    gpu: &mut Gpu,
    prompt_tokens: &[u32],
    max_steps: usize,
    rng: &mut DepthRng,
) -> Result<OmniOutput, String> {
    let last = *prompt_tokens.last().unwrap_or(&0);
    if last == AUDIO_START_TOKEN {
        let (codes, n, eos) = omni_generate_audio(
            cfg, weights, state, depth_cfg, depth_weights, gpu, prompt_tokens, max_steps, rng,
        )?;
        Ok(OmniOutput {
            text_tokens: vec![AUDIO_OUT_TOKEN; n],
            audio_codes: codes,
            n_audio_frames: n,
            hit_eos: eos,
            finished: eos,
        })
    } else if last == MIXED_START_TOKEN {
        omni_generate_interleaved(
            cfg, weights, state, depth_cfg, depth_weights, gpu, prompt_tokens, max_steps, rng,
        )
    } else {
        Err(format!(
            "omni: prompt must end with <|audio_start|> ({AUDIO_START_TOKEN}) or <|mixed_start|> ({MIXED_START_TOKEN}), got {last}"
        ))
    }
}

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

/// Run the omni INTERLEAVED-mode generate loop. `prompt_tokens` must end with
/// `<|mixed_start|>` (125015). Alternates text blocks (greedy argmax, 6 tokens)
/// and audio blocks (MaskGIT, 12 frames). `<|text_end|>` starts the audio tail.
fn omni_generate_interleaved(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    depth_cfg: &DepthConfig,
    depth_weights: &DepthWeights,
    gpu: &mut Gpu,
    prompt_tokens: &[u32],
    max_steps: usize,
    rng: &mut DepthRng,
) -> Result<OmniOutput, String> {
    let eos_code = depth_cfg.audio_eos_code;
    let n_text = INTERLEAVED_N_TEXT;
    let n_audio = INTERLEAVED_N_AUDIO;

    // ── Prefill ──
    let mut prev_logits: Option<Vec<f32>> = None;
    for (i, &t) in prompt_tokens.iter().enumerate() {
        prev_logits = Some(
            decode_step(cfg, weights, state, gpu, t, i as u32)
                .map_err(|e| format!("omni: prefill decode_step @ {i}: {e}"))?,
        );
    }
    let mut position = prompt_tokens.len() as u32 - 1;
    let mut hidden = gpu
        .download_f32(&state.final_norm_buf)
        .map_err(|e| format!("omni: download hidden: {e:?}"))?;

    // ── Interleaved state machine ──
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut audio_codes: Vec<u32> = Vec::new();
    let mut history: Vec<Vec<u32>> = Vec::new();
    let mut n_audio_frames = 0usize;
    let mut hit_eos = false;
    let mut finished = false;
    let mut slots_left = n_text;
    let mut speaking = false;
    let mut text_done = false;

    for step in 0..max_steps {
        if speaking {
            // Audio slot: MaskGIT-sample one frame.
            let codes =
                sample_audio_frame(depth_cfg, depth_weights, gpu, &hidden, &history, rng)
                    .map_err(|e| format!("omni: sample @ {step}: {e}"))?;
            let is_eos = codes.iter().any(|&c| c == eos_code);
            if is_eos {
                hit_eos = true;
                finished = true;
                break;
            }
            history.push(codes.clone());
            audio_codes.extend_from_slice(&codes);
            n_audio_frames += 1;
            text_tokens.push(AUDIO_OUT_TOKEN);

            let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &codes);
            position += 1;
            prev_logits = Some(
                decode_step_with_embed(cfg, weights, state, gpu, &next_embed, position)
                    .map_err(|e| format!("omni: audio step @ {step}: {e}"))?,
            );
            hidden = gpu
                .download_f32(&state.final_norm_buf)
                .map_err(|e| format!("omni: download hidden: {e:?}"))?;
            if omni_debug() {
                eprintln!(
                    "  [dbg-b] pos={position} AUDIO frame={n_audio_frames} cksum={:.0} logits_ck={:.0}",
                    hidden_cksum(&hidden),
                    hidden_cksum(prev_logits.as_ref().unwrap()),
                );
            }
            // In the audio tail (text_done), keep speaking until eos/max.
            // In an interleaved audio block, decrement + toggle to text at 0.
            if !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = false;
                    slots_left = n_text;
                }
            }
        } else {
            // Text slot: greedy argmax of the previous step's logits.
            let logits = prev_logits
                .as_ref()
                .ok_or_else(|| "omni: no prev logits for text slot".to_string())?;
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0 as u32;
            text_tokens.push(token);

            if token == TEXT_END_TOKEN {
                // End the text → start the audio tail (until eos/max).
                text_done = true;
                speaking = true;
                slots_left = n_audio;
                position += 1;
                prev_logits = Some(
                    decode_step(cfg, weights, state, gpu, token, position)
                        .map_err(|e| format!("omni: text_end step @ {step}: {e}"))?,
                );
                hidden = gpu
                    .download_f32(&state.final_norm_buf)
                    .map_err(|e| format!("omni: download hidden: {e:?}"))?;
                continue; // next step is audio (speaking=true)
            }
            if token == IM_END_TOKEN {
                finished = true;
                break;
            }

            position += 1;
            prev_logits = Some(
                decode_step(cfg, weights, state, gpu, token, position)
                    .map_err(|e| format!("omni: text step @ {step}: {e}"))?,
            );
            hidden = gpu
                .download_f32(&state.final_norm_buf)
                .map_err(|e| format!("omni: download hidden: {e:?}"))?;
            if omni_debug() {
                eprintln!(
                    "  [dbg-b] pos={position} TEXT token={token} cksum={:.0} logits_ck={:.0}",
                    hidden_cksum(&hidden),
                    hidden_cksum(prev_logits.as_ref().unwrap()),
                );
            }
            slots_left -= 1;
            if slots_left == 0 {
                speaking = true;
                slots_left = n_audio;
            }
        }
    }
    Ok(OmniOutput {
        text_tokens,
        audio_codes,
        n_audio_frames,
        hit_eos,
        finished,
    })
}

// ─── Streaming generation (Phase 4d) ─────────────────────────────────────────
// Prefill once, then generate in chunks (carrying the LM state + depth history
// + position + hidden across chunks). The daemon decodes each chunk via
// `codec_decode_streaming` + emits audio chunks for low-latency TTS.

/// Omni generation mode.
#[derive(Clone, Copy, PartialEq)]
pub enum OmniMode {
    Audio,
    Interleaved,
}

/// Persistent state across streaming `omni_stream_step` calls. The LFM3
/// `Lfm3State` (KV cache + conv state) lives in the caller's `LoadedModel` and
/// is passed to each step; this struct carries the omni-loop-level state
/// (position, hidden, depth history, INTERLEAVED block state).
pub struct OmniStreamState {
    pub mode: OmniMode,
    pub position: u32,
    pub hidden: Vec<f32>,
    pub history: Vec<Vec<u32>>,
    /// INTERLEAVED text-slot logits (from the previous step).
    pub prev_logits: Option<Vec<f32>>,
    pub slots_left: usize,
    pub speaking: bool,
    pub text_done: bool,
    pub finished: bool,
    pub hit_eos: bool,
}

/// One streaming step's output.
pub struct OmniStepOutput {
    pub audio_codes: Vec<u32>,
    pub n_audio_frames: usize,
    pub text_tokens: Vec<u32>,
    pub finished: bool,
}

/// Prefill the prompt + return the initial stream state. The mode is inferred
/// from the prompt's last token (`<|audio_start|>` → AUDIO, `<|mixed_start|>`
/// → INTERLEAVED). The caller's `Lfm3State` is advanced (KV cache + conv state
/// filled); `state.final_norm_buf` holds the hidden at the last prompt token.
pub fn omni_stream_prefill(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    gpu: &mut Gpu,
    prompt_tokens: &[u32],
) -> Result<OmniStreamState, String> {
    let last = *prompt_tokens.last().unwrap_or(&0);
    let mode = if last == MIXED_START_TOKEN {
        OmniMode::Interleaved
    } else {
        OmniMode::Audio
    };
    let mut prev_logits: Option<Vec<f32>> = None;
    for (i, &t) in prompt_tokens.iter().enumerate() {
        prev_logits = Some(
            decode_step(cfg, weights, state, gpu, t, i as u32)
                .map_err(|e| format!("omni stream: prefill @ {i}: {e}"))?,
        );
    }
    let hidden = gpu
        .download_f32(&state.final_norm_buf)
        .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
    Ok(OmniStreamState {
        mode,
        position: prompt_tokens.len() as u32 - 1,
        hidden,
        history: Vec::new(),
        prev_logits,
        slots_left: INTERLEAVED_N_TEXT,
        speaking: false,
        text_done: false,
        finished: false,
        hit_eos: false,
    })
}

/// Generate up to `max_steps` steps (audio frames in AUDIO mode; text/audio
/// steps in INTERLEAVED), carrying `stream`. Returns the audio codes + text
/// tokens generated in THIS step. The caller decodes the audio codes via
/// `codec_decode_streaming` + emits. `stream.finished` is set on eos/im_end.
pub fn omni_stream_step(
    cfg: &Lfm3Config,
    weights: &Lfm3Weights,
    state: &mut Lfm3State,
    depth_cfg: &DepthConfig,
    depth_weights: &DepthWeights,
    gpu: &mut Gpu,
    stream: &mut OmniStreamState,
    max_steps: usize,
    rng: &mut DepthRng,
) -> Result<OmniStepOutput, String> {
    let eos_code = depth_cfg.audio_eos_code;
    let mut audio_codes: Vec<u32> = Vec::new();
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut n_frames = 0usize;

    for _ in 0..max_steps {
        if stream.finished {
            break;
        }
        match stream.mode {
            OmniMode::Audio => {
                let codes = sample_audio_frame(depth_cfg, depth_weights, gpu, &stream.hidden, &stream.history, rng)
                    .map_err(|e| format!("omni stream: sample: {e}"))?;
                let is_eos = codes.iter().any(|&c| c == eos_code);
                if is_eos {
                    stream.hit_eos = true;
                    stream.finished = true;
                    break;
                }
                stream.history.push(codes.clone());
                audio_codes.extend_from_slice(&codes);
                n_frames += 1;
                text_tokens.push(AUDIO_OUT_TOKEN);
                let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &codes);
                stream.position += 1;
                stream.prev_logits = Some(decode_step_with_embed(cfg, weights, state, gpu, &next_embed, stream.position)
                    .map_err(|e| format!("omni stream: audio step: {e}"))?);
                stream.hidden = gpu.download_f32(&state.final_norm_buf)
                    .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
            }
            OmniMode::Interleaved => {
                if stream.speaking {
                    let codes = sample_audio_frame(depth_cfg, depth_weights, gpu, &stream.hidden, &stream.history, rng)
                        .map_err(|e| format!("omni stream: sample: {e}"))?;
                    let is_eos = codes.iter().any(|&c| c == eos_code);
                    if is_eos {
                        stream.hit_eos = true;
                        stream.finished = true;
                        break;
                    }
                    stream.history.push(codes.clone());
                    audio_codes.extend_from_slice(&codes);
                    n_frames += 1;
                    text_tokens.push(AUDIO_OUT_TOKEN);
                    let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &codes);
                    stream.position += 1;
                    stream.prev_logits = Some(decode_step_with_embed(cfg, weights, state, gpu, &next_embed, stream.position)
                        .map_err(|e| format!("omni stream: audio step: {e}"))?);
                    stream.hidden = gpu.download_f32(&state.final_norm_buf)
                        .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
                    if omni_debug() {
                        eprintln!(
                            "  [dbg-s] pos={} AUDIO frame={n_frames} cksum={:.0} logits_ck={:.0}",
                            stream.position,
                            hidden_cksum(&stream.hidden),
                            hidden_cksum(stream.prev_logits.as_ref().unwrap()),
                        );
                    }
                    if !stream.text_done {
                        stream.slots_left -= 1;
                        if stream.slots_left == 0 {
                            stream.speaking = false;
                            stream.slots_left = INTERLEAVED_N_TEXT;
                        }
                    }
                } else {
                    let logits = stream.prev_logits.as_ref()
                        .ok_or_else(|| "omni stream: no prev logits".to_string())?;
                    let token = logits.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap().0 as u32;
                    text_tokens.push(token);
                    if token == TEXT_END_TOKEN {
                        stream.text_done = true;
                        stream.speaking = true;
                        stream.slots_left = INTERLEAVED_N_AUDIO;
                        stream.position += 1;
                        stream.prev_logits = Some(decode_step(cfg, weights, state, gpu, token, stream.position)
                            .map_err(|e| format!("omni stream: text_end step: {e}"))?);
                        stream.hidden = gpu.download_f32(&state.final_norm_buf)
                            .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
                        if omni_debug() {
                            eprintln!(
                                "  [dbg-s] pos={} TEXT_END cksum={:.0}",
                                stream.position,
                                hidden_cksum(&stream.hidden)
                            );
                        }
                        continue;
                    }
                    if token == IM_END_TOKEN {
                        stream.finished = true;
                        break;
                    }
                    stream.position += 1;
                    stream.prev_logits = Some(decode_step(cfg, weights, state, gpu, token, stream.position)
                        .map_err(|e| format!("omni stream: text step: {e}"))?);
                    stream.hidden = gpu.download_f32(&state.final_norm_buf)
                        .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
                    if omni_debug() {
                        eprintln!(
                            "  [dbg-s] pos={} TEXT token={token} cksum={:.0} logits_ck={:.0}",
                            stream.position,
                            hidden_cksum(&stream.hidden),
                            hidden_cksum(stream.prev_logits.as_ref().unwrap()),
                        );
                    }
                    stream.slots_left -= 1;
                    if stream.slots_left == 0 {
                        stream.speaking = true;
                        stream.slots_left = INTERLEAVED_N_AUDIO;
                    }
                }
            }
        }
    }
    Ok(OmniStepOutput {
        audio_codes,
        n_audio_frames: n_frames,
        text_tokens,
        finished: stream.finished,
    })
}
