// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni realtime voice sessions (`GET /v1/realtime`, WebSocket).
//!
//! Ported from the pre-merge `hipfire-cli/src/main.rs` (`run_realtime_session`
//! et al.). The transport is re-implemented on hyper's upgraded IO with minimal
//! RFC6455 framing (no extra dependency, no HTTP-server takeover). Protocol
//! surface (Liquid Agent `voice.js`): outbound `session.created`,
//! `response.output_text.delta`, `response.output_audio.delta`, `response.done`;
//! inbound `session.update`, `input_audio_buffer.{append,clear,commit}`,
//! `conversation.item.create`, `response.create`, `response.cancel`.

use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use base64::Engine as _;
use hyper::body::Incoming;
use hyper::rt::{Read as HyperRead, ReadBuf, Write as HyperWrite};
use hyper::upgrade::Upgraded;
use hyper::{header, Request, Response, StatusCode};
use tokio::sync::mpsc;

use crate::serve::audio::{
    chunk_text_for_tts, pcm16_wav_bytes, strip_realtime_tool_markers, tts_generate_no_guard,
    tts_max_steps_for_input,
};
use crate::serve::complete::next_attempt_id;
use crate::serve::http::{boxed_full, request_id, BoxBody};
use crate::serve::{ModelOrigin, ServeShared};

/// RFC6455 `Sec-WebSocket-Accept` for a client handshake key.
pub(crate) fn realtime_websocket_accept_key(client_key: &str) -> String {
    use sha1::{Digest, Sha1};
    const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut hasher = Sha1::new();
    hasher.update(client_key.as_bytes());
    hasher.update(GUID.as_bytes());
    let digest = hasher.finalize();
    base64::engine::general_purpose::STANDARD.encode(digest)
}

/// Per-connection session state, shared with the turn worker.
#[derive(Default)]
struct RealtimeSessionState {
    instructions: Option<String>,
    max_output_tokens: u64,
    output_audio: bool,
    input_pcm: Vec<u8>,
    pending_transcript: Option<String>,
    history: Vec<(String, String)>,
    queued_items: Vec<String>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

impl RealtimeSessionState {
    fn new() -> Self {
        Self {
            max_output_tokens: 400,
            output_audio: true,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            ..Default::default()
        }
    }
}

// ─── Minimal RFC6455 framing over hyper's rt IO ─────────────────────────────

async fn read_exact_rt<S: HyperRead + Unpin>(s: &mut S, out: &mut [u8]) -> std::io::Result<()> {
    let mut filled = 0usize;
    while filled < out.len() {
        let mut rb = ReadBuf::new(&mut out[filled..]);
        std::future::poll_fn(|cx| Pin::new(&mut *s).poll_read(cx, rb.unfilled())).await?;
        let n = rb.filled().len();
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "websocket closed",
            ));
        }
        filled += n;
    }
    Ok(())
}

async fn write_all_rt<S: HyperWrite + Unpin>(s: &mut S, mut buf: &[u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        let n = std::future::poll_fn(|cx| Pin::new(&mut *s).poll_write(cx, buf)).await?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "write zero",
            ));
        }
        buf = &buf[n..];
    }
    Ok(())
}

async fn flush_rt<S: HyperWrite + Unpin>(s: &mut S) -> std::io::Result<()> {
    std::future::poll_fn(|cx| Pin::new(&mut *s).poll_flush(cx)).await
}

/// Write one unmasked (server-side) frame.
async fn ws_write_frame<S: HyperWrite + Unpin>(
    s: &mut S,
    opcode: u8,
    payload: &[u8],
) -> Result<()> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x80 | opcode);
    if payload.len() < 126 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    write_all_rt(s, &frame).await?;
    flush_rt(s).await?;
    Ok(())
}

async fn ws_write_text<S: HyperWrite + Unpin>(s: &mut S, text: &str) -> Result<()> {
    ws_write_frame(s, 0x1, text.as_bytes()).await
}

/// Read one complete text message (reassembling fragments); `None` on close.
async fn ws_read_text<S: HyperRead + HyperWrite + Unpin>(s: &mut S) -> Result<Option<String>> {
    let mut message = Vec::new();
    loop {
        let mut head = [0u8; 2];
        if read_exact_rt(s, &mut head).await.is_err() {
            return Ok(None);
        }
        let opcode = head[0] & 0x0f;
        let masked = head[1] & 0x80 != 0;
        let len0 = (head[1] & 0x7f) as u64;
        let len = match len0 {
            126 => {
                let mut b = [0u8; 2];
                read_exact_rt(s, &mut b).await?;
                u16::from_be_bytes(b) as u64
            }
            127 => {
                let mut b = [0u8; 8];
                read_exact_rt(s, &mut b).await?;
                u64::from_be_bytes(b)
            }
            n => n,
        };
        let mut mask = [0u8; 4];
        if masked {
            read_exact_rt(s, &mut mask).await?;
        }
        let mut payload = vec![0u8; len as usize];
        read_exact_rt(s, &mut payload).await?;
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }
        match opcode {
            0x0 | 0x1 => {
                message.extend_from_slice(&payload);
                if head[0] & 0x80 != 0 {
                    return Ok(Some(String::from_utf8_lossy(&message).into_owned()));
                }
            }
            0x8 => {
                let _ = ws_write_frame(s, 0x8, &[]).await;
                return Ok(None);
            }
            0x9 => {
                let _ = ws_write_frame(s, 0xA, &payload).await;
            }
            _ => {}
        }
    }
}

// ─── Route ──────────────────────────────────────────────────────────────────

/// `GET /v1/realtime` — complete the WebSocket handshake and spawn the session.
pub(crate) fn handle_realtime(
    req: Request<Incoming>,
    shared: Arc<ServeShared>,
) -> Response<BoxBody> {
    let key = match req
        .headers()
        .get("sec-websocket-key")
        .and_then(|v| v.to_str().ok())
    {
        Some(k) => k.to_owned(),
        None => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(header::CONTENT_TYPE, "text/plain")
                .body(boxed_full(b"missing Sec-WebSocket-Key".to_vec()))
                .unwrap()
        }
    };
    let accept = realtime_websocket_accept_key(&key);
    tokio::spawn(async move {
        match hyper::upgrade::on(req).await {
            Ok(upgraded) => {
                if let Err(error) = run_session(upgraded, shared).await {
                    eprintln!("[realtime] session ended: {error:#}");
                }
            }
            Err(error) => eprintln!("[realtime] upgrade failed: {error}"),
        }
    });
    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(header::CONNECTION, "Upgrade")
        .header(header::UPGRADE, "websocket")
        .header("Sec-WebSocket-Accept", accept)
        .body(boxed_full(Vec::new()))
        .unwrap()
}

/// Blocking daemon call for the realtime text stage (thinking off).
fn realtime_text(
    shared: &Arc<ServeShared>,
    messages: &[serde_json::Value],
    max_tokens: usize,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<String> {
    let model = shared
        .runtime
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .request_policy
        .operator_model
        .clone()
        .ok_or_else(|| anyhow!("realtime requires a resident/default model"))?;
    let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
    runtime.ensure_model(&model, &shared.meta, ModelOrigin::Request)?;
    let generate = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "messages": messages,
        "max_tokens": max_tokens,
        "max_think_tokens": 1,
        "attempt_id": next_attempt_id(),
    });
    let mut reply = String::new();
    runtime.engine.generate(&generate, |event| {
        if cancel.load(Ordering::Relaxed) {
            return Err(hipfire_client::ClientError::Cancelled);
        }
        if event.get("type").and_then(serde_json::Value::as_str) == Some("token")
            && event.get("reasoning").and_then(serde_json::Value::as_bool) != Some(true)
        {
            if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                reply.push_str(text);
            }
        }
        Ok(())
    })?;
    Ok(reply)
}

/// Blocking ASR daemon call for the realtime commit (16 kHz mono PCM16).
fn realtime_asr(shared: &Arc<ServeShared>, pcm16_mono_16k: &[u8]) -> Result<String> {
    let wav = pcm16_wav_bytes(pcm16_mono_16k, 16_000);
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
    let model = shared
        .runtime
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .request_policy
        .operator_model
        .clone()
        .ok_or_else(|| anyhow!("realtime requires a resident/default model"))?;
    let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
    runtime.ensure_model(&model, &shared.meta, ModelOrigin::Request)?;
    let generate = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "prompt": "",
        "system": "Perform ASR in English.",
        "audio_base64": audio_b64,
        "max_tokens": 512,
        "temperature": 0.0,
        "attempt_id": next_attempt_id(),
    });
    let mut transcript = String::new();
    runtime.engine.generate(&generate, |event| {
        if event.get("type").and_then(serde_json::Value::as_str) == Some("token") {
            if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                transcript.push_str(text);
            }
        }
        Ok(())
    })?;
    Ok(transcript.trim().to_owned())
}

/// Drive one turn (text reply + spoken chunks), emitting events on `tx`.
fn run_turn(
    shared: &Arc<ServeShared>,
    state: &Arc<std::sync::Mutex<RealtimeSessionState>>,
    user_text: String,
    tx: &mpsc::UnboundedSender<String>,
) {
    let emit = |value: serde_json::Value| {
        let _ = tx.send(value.to_string());
    };
    let (instructions, max_tokens, want_audio, history) = {
        let st = state.lock().unwrap_or_else(|e| e.into_inner());
        (
            st.instructions.clone().unwrap_or_else(|| {
                "You are a helpful voice assistant. Keep replies short and conversational."
                    .to_owned()
            }),
            st.max_output_tokens as usize,
            st.output_audio,
            st.history.clone(),
        )
    };
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({ "role": "system", "content": instructions })];
    for (role, content) in &history {
        messages.push(serde_json::json!({ "role": role, "content": content }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": user_text }));

    let cancel_flag = Arc::clone(&state.lock().unwrap_or_else(|e| e.into_inner()).cancel);
    let reply = match realtime_text(shared, &messages, max_tokens, &cancel_flag) {
        Ok(reply) => reply,
        Err(error) => {
            if cancel_flag.load(Ordering::Relaxed) {
                emit(serde_json::json!({"type":"response.done","response":{"status":"cancelled"}}));
            } else {
                eprintln!("[realtime] text stage failed: {error:#}");
                emit(
                    serde_json::json!({"type":"error","error":{"message":format!("reply generation failed: {error:#}")}}),
                );
            }
            return;
        }
    };
    let reply_visible = strip_realtime_tool_markers(&reply);
    for chunk in chunk_text_for_tts(&reply_visible) {
        emit(serde_json::json!({"type":"response.output_text.delta","delta":chunk}));
    }
    {
        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
        st.history.push(("user".to_owned(), user_text));
        st.history
            .push(("assistant".to_owned(), reply_visible.clone()));
        let excess = st.history.len().saturating_sub(20);
        if excess > 0 {
            st.history.drain(..excess);
        }
    }
    if want_audio && !reply_visible.trim().is_empty() {
        for chunk in chunk_text_for_tts(&reply_visible) {
            let cancel_flag = Arc::clone(&state.lock().unwrap_or_else(|e| e.into_inner()).cancel);
            let max_steps = tts_max_steps_for_input(&chunk);
            match tts_generate_no_guard(shared, &chunk, None, max_steps, Some(&cancel_flag)) {
                Ok(pcm) => {
                    emit(serde_json::json!({
                        "type":"response.output_audio.delta",
                        "delta": base64::engine::general_purpose::STANDARD.encode(&pcm),
                    }));
                }
                Err(error) => {
                    if cancel_flag.load(Ordering::Relaxed) {
                        emit(
                            serde_json::json!({"type":"response.done","response":{"status":"cancelled"}}),
                        );
                    } else {
                        eprintln!("[realtime] tts stage failed: {error:#}");
                        emit(
                            serde_json::json!({"type":"response.done","response":{"status":"completed"}}),
                        );
                    }
                    return;
                }
            }
        }
    }
    let was_cancelled = state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .cancel
        .load(Ordering::Relaxed);
    emit(serde_json::json!({
        "type":"response.done",
        "response":{"status": if was_cancelled {"cancelled"} else {"completed"}},
    }));
}

async fn run_session(upgraded: Upgraded, shared: Arc<ServeShared>) -> Result<()> {
    let mut ws = upgraded;
    let state = Arc::new(std::sync::Mutex::new(RealtimeSessionState::new()));
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let session_created = serde_json::json!({
        "type": "session.created",
        "session": {
            "audio": {
                "input": { "format": { "rate": 16000 } },
                "output": { "format": { "rate": 24000 } },
            },
        },
    });
    ws_write_text(&mut ws, &session_created.to_string()).await?;

    loop {
        tokio::select! {
            inbound = ws_read_text(&mut ws) => {
                let text = match inbound? {
                    Some(t) => t,
                    None => break,
                };
                let event: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(error) => {
                        ws_write_text(&mut ws, &serde_json::json!({
                            "type": "error",
                            "error": { "message": format!("unparseable realtime event: {error}") },
                        }).to_string()).await?;
                        continue;
                    }
                };
                match event.get("type").and_then(serde_json::Value::as_str) {
                    Some("session.update") => {
                        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(i) = event.pointer("/session/instructions").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                            st.instructions = Some(i.to_owned());
                        }
                        if let Some(t) = event.pointer("/session/max_output_tokens").and_then(|v| v.as_u64()) {
                            st.max_output_tokens = t.clamp(64, 4096);
                        }
                        if let Some(m) = event.pointer("/session/output_modalities").and_then(|v| v.as_array()) {
                            st.output_audio = m.iter().filter_map(|v| v.as_str()).any(|m| m == "audio");
                        }
                    }
                    Some("input_audio_buffer.append") => {
                        if let Some(audio) = event.get("audio").and_then(|v| v.as_str()) {
                            match base64::engine::general_purpose::STANDARD.decode(audio) {
                                Ok(pcm) => state.lock().unwrap_or_else(|e| e.into_inner()).input_pcm.extend_from_slice(&pcm),
                                Err(error) => eprintln!("[realtime] input_audio_buffer.append decode failed: {error}"),
                            }
                        }
                    }
                    Some("input_audio_buffer.clear") => {
                        state.lock().unwrap_or_else(|e| e.into_inner()).input_pcm.clear();
                    }
                    Some("input_audio_buffer.commit") => {
                        let pcm = { std::mem::take(&mut state.lock().unwrap_or_else(|e| e.into_inner()).input_pcm) };
                        let min_samples = 16_000usize * 2 * 3 / 10;
                        if pcm.len() < min_samples {
                            continue;
                        }
                        let sh = Arc::clone(&shared);
                        let transcript = tokio::task::spawn_blocking(move || realtime_asr(&sh, &pcm)).await;
                        match transcript {
                            Ok(Ok(t)) => { state.lock().unwrap_or_else(|e| e.into_inner()).pending_transcript = Some(t); }
                            Ok(Err(error)) => {
                                ws_write_text(&mut ws, &serde_json::json!({
                                    "type":"error","error":{"message":format!("transcription failed: {error:#}")}
                                }).to_string()).await?;
                            }
                            Err(_) => {}
                        }
                    }
                    Some("conversation.item.create") => {
                        let item = event.get("item").cloned().unwrap_or(serde_json::Value::Null);
                        if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                            state.lock().unwrap_or_else(|e| e.into_inner()).queued_items.push(content.to_owned());
                        }
                    }
                    Some("response.create") => {
                        let turn = {
                            let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                            let mut parts: Vec<String> = Vec::new();
                            if let Some(t) = st.pending_transcript.take() {
                                if !t.trim().is_empty() { parts.push(t.trim().to_owned()); }
                            }
                            parts.extend(std::mem::take(&mut st.queued_items));
                            if parts.is_empty() { None } else {
                                st.cancel.store(false, Ordering::Relaxed);
                                Some(parts.join("\n"))
                            }
                        };
                        match turn {
                            None => {
                                ws_write_text(&mut ws, &serde_json::json!({
                                    "type":"response.done","response":{"status":"completed"}
                                }).to_string()).await?;
                            }
                            Some(user_text) => {
                                let sh = Arc::clone(&shared);
                                let st = Arc::clone(&state);
                                let tx2 = tx.clone();
                                tokio::task::spawn_blocking(move || run_turn(&sh, &st, user_text, &tx2));
                            }
                        }
                    }
                    Some("response.cancel") => {
                        state.lock().unwrap_or_else(|e| e.into_inner()).cancel.store(true, Ordering::Relaxed);
                    }
                    Some(other) => eprintln!("[realtime] ignoring event {other:?}"),
                    None => eprintln!("[realtime] ignoring event without type"),
                }
            }
            Some(out) = rx.recv() => {
                ws_write_text(&mut ws, &out).await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::realtime_websocket_accept_key;

    #[test]
    fn realtime_accept_key_matches_rfc6455_example() {
        // RFC6455 §1.3 worked example.
        assert_eq!(
            realtime_websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }
}
