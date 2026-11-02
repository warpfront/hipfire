// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni text-to-speech (`POST /tts`) — batch speech synthesis, plus the
//! shared TTS helpers the realtime voice session reuses.
//!
//! Ported from the pre-merge `hipfire-cli/src/main.rs` (`respond_tts` /
//! `tts_generate` / `chunk_text_for_tts` / `strip_realtime_tool_markers`) into
//! the modular `serve/` layout.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use base64::Engine as _;
use hyper::{header, Response};
use tokio_util::sync::CancellationToken;

use crate::serve::complete::next_attempt_id;
use crate::serve::http::{boxed_full, openai_error, request_id, BoxBody};
use crate::serve::{ModelOrigin, ServeShared};

/// Speech sample rate the omni codec decodes to (24 kHz).
pub(crate) const TTS_SAMPLE_RATE: u32 = 24_000;

/// Step budget for a `/tts` request, scaled by input length.
pub(crate) fn tts_max_steps_for_input(input: &str) -> u64 {
    const FLOOR: u64 = 256;
    const CEIL: u64 = 3072;
    let chars = input.chars().count() as u64;
    (chars + (chars / 4) + 256).clamp(FLOOR, CEIL)
}

/// Wrap raw mono PCM16 samples in a minimal 44-byte WAV header.
pub(crate) fn pcm16_wav_bytes(pcm: &[u8], rate: u32) -> Vec<u8> {
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + pcm.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

/// Strip LFM3 native tool-call markers from a voice reply so nothing is spoken.
pub(crate) fn strip_realtime_tool_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<tool_call>") {
        out.push_str(&remaining[..start]);
        let after = &remaining[start + "<tool_call>".len()..];
        remaining = match after.find("</tool_call>") {
            Some(end) => &after[end + "</tool_call>".len()..],
            None => "",
        };
    }
    out.push_str(remaining);
    out.trim().to_owned()
}

/// Split a reply into sentence/clause chunks for spoken delivery.
pub(crate) fn chunk_text_for_tts(text: &str) -> Vec<String> {
    const MIN_CHUNK: usize = 120;
    const MAX_CHUNK: usize = 300;
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut piece = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        piece.push(c);
        let sentence_end = matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '…' | ';' | '；');
        let is_ws_follow = chars.peek().is_some_and(|n| n.is_whitespace());
        if (sentence_end
            && is_ws_follow
            && current.chars().count() + piece.chars().count() >= MIN_CHUNK)
            || current.chars().count() + piece.chars().count() >= MAX_CHUNK
        {
            current.push_str(piece.trim_start_matches('\n'));
            current.push(' ');
            chunks.push(current.trim().to_owned());
            current = String::new();
            piece = String::new();
        }
    }
    current.push_str(&piece);
    let tail = current.trim();
    if !tail.is_empty() {
        chunks.push(tail.to_owned());
    }
    chunks
}

/// Drive one omni TTS generate on the daemon and collect the WAV bytes.
///
/// `cancel` (optional) aborts the attempt via `Engine::generate_cancellable`.
/// Blocking; call from `spawn_blocking`.
pub(crate) fn tts_generate(
    shared: &ServeShared,
    input: &str,
    voice: Option<&str>,
    max_steps: u64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<u8>> {
    let model = {
        let runtime = shared
            .runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        runtime.request_policy.operator_model.clone()
    };
    let model = model.ok_or_else(|| {
        anyhow!("tts requires a resident/default model (serve --model or serve.default_model)")
    })?;

    let mut runtime = shared
        .runtime
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    runtime.ensure_model(&model, &shared.meta, ModelOrigin::Request)?;

    let generate = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "tts": true,
        "mode": "audio",
        "voice": voice.unwrap_or("female"),
        "prompt": input,
        "max_tokens": max_steps,
        "max_steps": max_steps,
        "attempt_id": next_attempt_id(),
    });

    let mut wav: Option<Vec<u8>> = None;
    let mut streamed_pcm: Vec<u8> = Vec::new();
    let mut fold = |event: &serde_json::Value| -> hipfire_client::Result<()> {
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("audio") => {
                let b64 = event
                    .get("audio")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        hipfire_client::ClientError::Protocol(
                            "tts audio event missing audio field".into(),
                        )
                    })?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|error| {
                        hipfire_client::ClientError::Protocol(format!(
                            "tts audio base64 decode failed: {error}"
                        ))
                    })?;
                wav = Some(bytes);
            }
            Some("audio_chunk") => {
                if let Some(b64) = event.get("audio").and_then(serde_json::Value::as_str) {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                        if bytes.len() > 44 && bytes.starts_with(b"RIFF") {
                            streamed_pcm.extend_from_slice(&bytes[44..]);
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    };
    match cancel {
        Some(flag) => runtime.engine.generate_cancellable(&generate, flag, &mut fold)?,
        None => runtime.engine.generate(&generate, &mut fold)?,
    };

    match wav {
        Some(wav) if !wav.is_empty() => Ok(wav),
        _ if !streamed_pcm.is_empty() => Ok(pcm16_wav_bytes(&streamed_pcm, TTS_SAMPLE_RATE)),
        _ => bail!("tts generated no audio (the turn produced text only)"),
    }
}

/// Realtime-session TTS: acquire admission, then drive one TTS generate.
pub(crate) fn tts_generate_no_guard(
    shared: &Arc<ServeShared>,
    input: &str,
    voice: Option<&str>,
    max_steps: u64,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<u8>> {
    // NB: `cancel` here is the session's cooperative flag, not the admission
    // token; the realtime session holds its own turn-by-turn flow, so a short
    // synchronous admission acquire is sufficient.
    let _guard = shared
        .admission
        .acquire()
        .map_err(|error| anyhow!("admission: {error}"))?;
    let _ = Ordering::Relaxed;
    tts_generate(shared, input, voice, max_steps, cancel)
}

/// `POST /tts` — `{ "input": "...", "voice": "male"|"female" }` → `audio/wav`.
pub(crate) async fn handle_tts(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
) -> Response<BoxBody> {
    let input = match body
        .get("input")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        Some(text) => text.to_owned(),
        None => return openai_error("tts request requires a non-empty \"input\" string", 400),
    };
    let voice = body
        .get("voice")
        .and_then(serde_json::Value::as_str)
        .filter(|v| *v == "male" || *v == "female")
        .map(str::to_owned);
    let max_steps = tts_max_steps_for_input(&input);

    let cancel = CancellationToken::new();
    let _guard = match shared.admission.acquire_async(cancel).await {
        Ok(guard) => guard,
        Err(error) => return openai_error(&error.to_string(), 503),
    };

    let sh = Arc::clone(&shared);
    let result = tokio::task::spawn_blocking(move || {
        tts_generate(&sh, &input, voice.as_deref(), max_steps, None)
    })
    .await;
    match result {
        Ok(Ok(wav)) => Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "audio/wav")
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(boxed_full(wav))
            .unwrap(),
        Ok(Err(error)) => openai_error(&error.to_string(), 500),
        Err(_) => openai_error("tts worker panicked", 500),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_wav_header_is_well_formed() {
        let wav = pcm16_wav_bytes(&[0u8; 8], 24_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 8);
    }

    #[test]
    fn tts_max_steps_floor_and_ceiling() {
        assert_eq!(tts_max_steps_for_input(""), 256);
        assert_eq!(tts_max_steps_for_input(&"x".repeat(10_000)), 3072);
    }

    #[test]
    fn strip_tool_markers_removes_calls() {
        assert_eq!(
            strip_realtime_tool_markers("hello <tool_call>{\"a\":1}</tool_call> world"),
            "hello  world"
        );
    }
}
