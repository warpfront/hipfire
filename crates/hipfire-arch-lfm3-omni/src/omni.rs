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
//!
//! Audio-eos contract (EOS-1): in INTERLEAVED the loop mirrors HF — the eos
//! frame is replaced with the full eos code, its embedding is fed back, and
//! the loop leaves interleaving into a plain-TEXT continuation that ends on
//! <|im_end|> / post-eos <|text_end|> / max_steps. AUDIO mode stops at
//! audio-eos instead (TTS contract; see `omni_generate_audio`).

use crate::depth::{get_audio_out_embeds, sample_audio_frame, DepthConfig, DepthRng, DepthWeights};
use hipfire_arch_lfm3::{decode_step, decode_step_with_embed, Lfm3Config, Lfm3State, Lfm3Weights};
use rdna_compute::Gpu;

// ─── Omni token ids (HF `Lfm3OmniConfig` defaults) ───────────────────────────
pub const AUDIO_START_TOKEN: u32 = 125_012;
pub const AUDIO_OUT_TOKEN: u32 = 125_032;
pub const TEXT_END_TOKEN: u32 = 125_014;
pub const MIXED_START_TOKEN: u32 = 125_015;
pub const IM_END_TOKEN: u32 = 124_900; // <|im_end|> (eos)
/// Input placeholders (HF `Lfm3OmniConfig`): `<|audio|>` / image / video.
pub const AUDIO_TOKEN: u32 = 125_031;
pub const IMAGE_TOKEN: u32 = 125_030;
pub const VIDEO_TOKEN: u32 = 125_029;
/// Interleaved block sizes (HF `interleaved_n_text` / `interleaved_n_audio`).
pub const INTERLEAVED_N_TEXT: usize = 6;
pub const INTERLEAVED_N_AUDIO: usize = 12;

/// HF text-slot token suppression (modeling_lfm3_omni.py:1255-1269 live path +
/// :1585-1593 `_generate` placeholder set). A text slot must never sample a
/// state-machine control token: the modality placeholders and the mode
/// markers have no text-slot handling in this loop, so sampling one pushes it
/// as literal text and feeds the LM a wrong next-step context.
///
/// `<|text_end|>` (125014) closes every text block and is deliberately NOT
/// suppressed (HF: "never suppressed even when it is listed as an end-of-turn
/// id"). `<|im_end|>` (124900) also stays samplable: HF's interleaved
/// `_generate` does not mask eos ids in text slots — it samples them and ends
/// the turn — and this loop's state machine handles it identically. (Only the
/// live-transcription path masks eos ids, because its slot semantics differ.)
pub fn text_argmax_suppressed(t: u32) -> bool {
    matches!(
        t,
        AUDIO_START_TOKEN
            | MIXED_START_TOKEN
            | AUDIO_OUT_TOKEN
            | AUDIO_TOKEN
            | IMAGE_TOKEN
            | VIDEO_TOKEN
    )
}

/// Greedy argmax over `logits` with [`text_argmax_suppressed`] applied —
/// suppressed ids are excluded the way HF excludes them (`-inf` before
/// `argmax`). Centralized so the interleaved-batch and streaming text
/// branches pick identically. A NaN logit can no longer win (the previous
/// unguarded `max_by` could latch onto one and freeze the slot).
pub fn argmax_text_token(logits: &[f32]) -> u32 {
    let mut best: Option<(u32, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        let t = i as u32;
        if text_argmax_suppressed(t) {
            continue;
        }
        match best {
            Some((_, bv)) if bv >= v => {}
            _ => best = Some((t, v)),
        }
    }
    best.map(|(t, _)| t).unwrap_or(0)
}

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
///
/// TTS-stop contract (EOS-1 deviation from HF, deliberate): the loop stops at
/// `audio_eos_code`. HF (modeling_lfm3_omni.py:1737-1749) instead replaces the
/// frame with the eos code, feeds its embedding, switches the row to TEXT and
/// keeps generating until <|im_end|>/<|text_end|>/max — behavior that serves
/// its open-ended chat surface. hipfire's AUDIO route is TTS-shaped (the
/// daemon decodes the codes to WAV and finishes), so audio-eos is the stop
/// signal here. The INTERLEAVED loop keeps the HF behavior.
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
    let mut position = prompt_tokens.len().saturating_sub(1) as u32;
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
    let mut position = prompt_tokens.len().saturating_sub(1) as u32;
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
    // EOS-1: HF never ends a turn on audio-eos — it replaces the sampled
    // frame with the full eos code, feeds that embedding back to the LM,
    // leaves interleaving, and continues in plain-TEXT mode
    // (modeling_lfm3_omni.py:1737-1749); the turn then ends on <|im_end|>,
    // a post-eos <|text_end|>, or max_steps. This loop mirrors that for
    // INTERLEAVED. AUDIO mode (`omni_generate_audio`) keeps the hipfire TTS
    // contract instead — audio-eos is the stop signal (documented deviation:
    // the TTS route's deliverable is the audio itself).
    let mut in_interleave = true;

    for step in 0..max_steps {
        if speaking {
            // Audio slot: MaskGIT-sample one frame.
            let codes =
                sample_audio_frame(depth_cfg, depth_weights, gpu, &hidden, &history, rng)
                    .map_err(|e| format!("omni: sample @ {step}: {e}"))?;
            let is_eos = codes.iter().any(|&c| c == eos_code);
            if is_eos {
                hit_eos = true;
                // HF: full-eos frame → its embedding → TEXT continuation.
                // The eos frame is NOT appended to history and NOT emitted.
                let eos_frame = vec![eos_code; codes.len()];
                let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &eos_frame);
                position += 1;
                prev_logits = Some(
                    decode_step_with_embed(cfg, weights, state, gpu, &next_embed, position)
                        .map_err(|e| format!("omni: audio eos step @ {step}: {e}"))?,
                );
                hidden = gpu
                    .download_f32(&state.final_norm_buf)
                    .map_err(|e| format!("omni: download hidden: {e:?}"))?;
                in_interleave = false;
                text_done = true;
                speaking = false;
                continue;
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
            // Text slot: greedy argmax of the previous step's logits, with the
            // HF control-token suppression (see `argmax_text_token`).
            let logits = prev_logits
                .as_ref()
                .ok_or_else(|| "omni: no prev logits for text slot".to_string())?;
            let token = argmax_text_token(logits);
            text_tokens.push(token);

            if token == TEXT_END_TOKEN {
                if in_interleave {
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
                // Left interleaving (post-eos TEXT continuation): with the row
                // no longer interleaving, HF ends the turn on text_end.
                finished = true;
                break;
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
            // Block alternation only while still interleaving and before the
            // audio tail — an HF `interleaving[row] and not text_done[row]`
            // guard the Rust loop was missing (it also gates the post-eos
            // plain-TEXT continuation, which must never flip back to audio).
            if in_interleave && !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = true;
                    slots_left = n_audio;
                }
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
    /// EOS-1: set when an audio-eos fired and the loop left interleaving for
    /// the HF plain-TEXT continuation (mode→TEXT in modeling_lfm3_omni.py).
    /// While set, block alternation is suspended and a sampled
    /// `<|text_end|>` finishes the turn instead of starting an audio tail.
    pub left_interleave: bool,
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
    // Mode inference matches omni_generate: unknown markers error instead of
    // silently degrading to AUDIO (the old default desynced prefill vs the
    // generate loop's contract for the same prompt shape).
    let mode = if last == AUDIO_START_TOKEN {
        OmniMode::Audio
    } else if last == MIXED_START_TOKEN {
        OmniMode::Interleaved
    } else {
        return Err(format!(
            "omni stream: prompt must end with <|audio_start|> ({AUDIO_START_TOKEN}) or <|mixed_start|> ({MIXED_START_TOKEN}), got {last}"
        ));
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
        position: prompt_tokens.len().saturating_sub(1) as u32,
        hidden,
        history: Vec::new(),
        prev_logits,
        slots_left: INTERLEAVED_N_TEXT,
        speaking: false,
        text_done: false,
        left_interleave: false,
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
                    // EOS-1 (AUDIO-mode deviation, deliberate): hipfire's TTS
                    // contract stops at audio-eos — the deliverable is the
                    // audio itself. HF (modeling_lfm3_omni.py:1737-1749) would
                    // switch the row to TEXT and keep generating; matching that
                    // here would leave a TTS stream with no terminal for the
                    // codec flush (see generate_lfm3_omni's streaming path).
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
                        // EOS-1 (INTERLEAVED): mirror HF — full-eos frame →
                        // embedding → plain-TEXT continuation, turn ends later
                        // on im_end / post-eos text_end / the caller's budget.
                        let eos_frame = vec![eos_code; codes.len()];
                        let next_embed = get_audio_out_embeds(depth_cfg, depth_weights, &eos_frame);
                        stream.position += 1;
                        stream.prev_logits = Some(decode_step_with_embed(cfg, weights, state, gpu, &next_embed, stream.position)
                            .map_err(|e| format!("omni stream: audio eos step: {e}"))?);
                        stream.hidden = gpu.download_f32(&state.final_norm_buf)
                            .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
                        stream.left_interleave = true;
                        stream.text_done = true;
                        stream.speaking = false;
                        continue;
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
                    // Same suppression as the batch loop (see `argmax_text_token`).
                    let token = argmax_text_token(logits);
                    text_tokens.push(token);
                    if token == TEXT_END_TOKEN {
                        if !stream.left_interleave {
                            // End the text → start the audio tail (until eos/max).
                            stream.text_done = true;
                            stream.speaking = true;
                            stream.slots_left = INTERLEAVED_N_AUDIO;
                            stream.position += 1;
                            stream.prev_logits = Some(decode_step(cfg, weights, state, gpu, token, stream.position)
                                .map_err(|e| format!("omni stream: text_end step: {e}"))?);
                            stream.hidden = gpu.download_f32(&state.final_norm_buf)
                                .map_err(|e| format!("omni stream: dl hidden: {e:?}"))?;
                            continue;
                        }
                        // Left interleaving (post-eos TEXT continuation): the
                        // turn ends here, matching HF's non-interleaving row.
                        stream.finished = true;
                        break;
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
                    // Block alternation only while still interleaving and
                    // before the audio tail (HF `interleaving and not
                    // text_done`); suspends it for the post-eos TEXT path.
                    if !stream.left_interleave && !stream.text_done {
                        stream.slots_left -= 1;
                        if stream.slots_left == 0 {
                            stream.speaking = true;
                            stream.slots_left = INTERLEAVED_N_AUDIO;
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// SUPPRESS-1: a suppressed control token must never win the text-slot
    /// argmax even when it carries the largest logit (HF masks it to -inf
    /// before argmax; the previous unguarded `max_by` could pick it and
    /// desync the state machine).
    #[test]
    fn text_argmax_skips_suppressed_control_tokens() {
        let mut logits = vec![-1.0f32; 130_000];
        let best = 12usize;
        logits[best] = 5.0;
        logits[AUDIO_START_TOKEN as usize] = 9.0;
        logits[MIXED_START_TOKEN as usize] = 8.0;
        logits[AUDIO_OUT_TOKEN as usize] = 7.0;
        logits[AUDIO_TOKEN as usize] = 6.0;
        logits[IMAGE_TOKEN as usize] = 6.5;
        logits[VIDEO_TOKEN as usize] = 5.5;
        assert_eq!(argmax_text_token(&logits), best as u32);
    }

    /// SUPPRESS-1: `<|text_end|>` (block closer) and `<|im_end|>` (turn end)
    /// stay samplable — HF masks neither in the interleaved text slot.
    #[test]
    fn text_argmax_allows_text_end_and_im_end() {
        let mut logits = vec![-1.0f32; 130_000];
        logits[TEXT_END_TOKEN as usize] = 5.0;
        assert_eq!(argmax_text_token(&logits), TEXT_END_TOKEN);

        let mut logits = vec![-1.0f32; 130_000];
        logits[IM_END_TOKEN as usize] = 5.0;
        assert_eq!(argmax_text_token(&logits), IM_END_TOKEN);
    }

    /// SUPPRESS-1: a NaN logit can no longer latch the argmax (the old
    /// `partial_cmp(..).unwrap_or(Equal)` chain could keep a NaN as max).
    #[test]
    fn text_argmax_ignores_nan() {
        let mut logits = vec![-1.0f32; 130_000];
        logits[10] = f32::NAN;
        logits[11] = 0.5;
        assert_eq!(argmax_text_token(&logits), 11);
    }
}
