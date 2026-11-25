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
    /// Conversation history as full OpenAI message values, not `(role, text)`
    /// pairs: a tool turn has to replay the assistant's `tool_calls` and the
    /// following `role: "tool"` results, which a pair cannot express.
    history: Vec<serde_json::Value>,
    queued_items: Vec<String>,
    /// `function_call_output` items received since the last `response.create`.
    /// Held out of `history` until the turn that consumes them runs, so a
    /// result arriving before its `response.create` is still delivered.
    pending_tool_results: Vec<serde_json::Value>,
    /// `tools` from `session.update`. `voice.js` declares them once per
    /// session; the text stage forwards them to the model per turn.
    tools: Vec<serde_json::Value>,
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

/// RFC6455 close codes the reader sends before giving up on a connection.
const WS_CLOSE_PROTOCOL_ERROR: u16 = 1002;
const WS_CLOSE_MESSAGE_TOO_BIG: u16 = 1009;

/// Send a close frame carrying `code`, then fail the read with `reason`.
async fn ws_fail<S: HyperWrite + Unpin>(s: &mut S, code: u16, reason: &str) -> anyhow::Error {
    let _ = ws_write_frame(s, 0x8, &code.to_be_bytes()).await;
    anyhow!("websocket closed with {code}: {reason}")
}

/// Read one complete text message (reassembling fragments); `None` on close.
///
/// The declared payload length is untrusted: a frame, or a fragmented message
/// as a whole, larger than `max_bytes` closes the connection with 1009 before
/// anything is allocated. Unmasked client frames and malformed control frames
/// close it with 1002 (RFC6455 §5.1, §5.5).
async fn ws_read_text<S: HyperRead + HyperWrite + Unpin>(
    s: &mut S,
    max_bytes: usize,
) -> Result<Option<String>> {
    let mut message = Vec::new();
    loop {
        let mut head = [0u8; 2];
        if read_exact_rt(s, &mut head).await.is_err() {
            return Ok(None);
        }
        let fin = head[0] & 0x80 != 0;
        let opcode = head[0] & 0x0f;
        let masked = head[1] & 0x80 != 0;
        let len0 = (head[1] & 0x7f) as u64;
        let is_control = opcode & 0x8 != 0;
        if !masked {
            return Err(ws_fail(s, WS_CLOSE_PROTOCOL_ERROR, "unmasked client frame").await);
        }
        if is_control && (!fin || len0 > 125) {
            return Err(ws_fail(s, WS_CLOSE_PROTOCOL_ERROR, "malformed control frame").await);
        }
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
        let budget = max_bytes.saturating_sub(if is_control { 0 } else { message.len() });
        let len = match usize::try_from(len) {
            Ok(len) if len <= budget => len,
            _ => {
                return Err(ws_fail(
                    s,
                    WS_CLOSE_MESSAGE_TOO_BIG,
                    &format!("frame of {len} bytes exceeds the {max_bytes}-byte limit"),
                )
                .await)
            }
        };
        let mut mask = [0u8; 4];
        read_exact_rt(s, &mut mask).await?;
        let mut payload = vec![0u8; len];
        read_exact_rt(s, &mut payload).await?;
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[i % 4];
        }
        match opcode {
            0x0 | 0x1 => {
                message.extend_from_slice(&payload);
                if fin {
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
/// One turn's text stage: visible prose plus any structured tool calls.
struct RealtimeTextOutcome {
    reply: String,
    /// Canonical calls (`{id?, name, arguments}`) from the daemon's
    /// `{"type":"tool_calls"}` event or the tool-safe `done`.
    calls: Vec<serde_json::Value>,
}

/// Blocking text generation for one realtime turn. `tools` is forwarded so the
/// model can answer with structured calls instead of prose; the daemon's LFM3
/// route renders the schemas through the vendor template and withholds the
/// `<tool_call>` span from the visible stream.
fn realtime_text(
    shared: &Arc<ServeShared>,
    messages: &[serde_json::Value],
    max_tokens: usize,
    tools: &[serde_json::Value],
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<RealtimeTextOutcome> {
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
    let mut generate = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "messages": messages,
        "max_tokens": max_tokens,
        "max_think_tokens": 1,
        "attempt_id": next_attempt_id(),
    });
    if !tools.is_empty() {
        generate["tools"] = serde_json::Value::Array(tools.to_vec());
    }
    let mut reply = String::new();
    let mut calls: Vec<serde_json::Value> = Vec::new();
    runtime.engine.generate(&generate, |event| {
        if cancel.load(Ordering::Relaxed) {
            return Err(hipfire_client::ClientError::Cancelled);
        }
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("token") => {
                if event.get("reasoning").and_then(serde_json::Value::as_bool) != Some(true) {
                    if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                        reply.push_str(text);
                    }
                }
            }
            // The LFM3 route emits calls as a dedicated event BEFORE the
            // commit handshake. Record them; the terminal `calls` array is the
            // fallback for a route that only stages them.
            Some("tool_calls") => {
                if let Some(arr) = event.get("calls").and_then(serde_json::Value::as_array) {
                    calls = arr.clone();
                }
            }
            Some("commit_ready") | Some("done") => {
                if event
                    .get("finish_reason")
                    .and_then(serde_json::Value::as_str)
                    == Some("tool_calls")
                {
                    if let Some(arr) = event.get("calls").and_then(serde_json::Value::as_array) {
                        if !arr.is_empty() {
                            calls = arr.clone();
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(RealtimeTextOutcome { reply, calls })
}

/// The history assistant message carrying one turn's calls, in the flat shape
/// [`hipfire_runtime::prompt_frame::Message`] deserializes (`{id?, name,
/// arguments}` with `arguments` as a JSON OBJECT — the LFM3 template refuses a
/// JSON-encoded string).
fn realtime_assistant_tool_message(calls: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": "",
        "tool_calls": calls,
    })
}

/// Project one client `conversation.item.create` value into an OpenAI
/// `role: "tool"` message, or `None` when it is not a `function_call_output`.
///
/// Accepts the OpenAI realtime shape (`{"type":"function_call_output",
/// "call_id","output"}`) and the nested-content variant voice.js may send
/// (`{"type":"function_call_output","call_id":"…","content":[{"type":"text",
/// "text":"…"}]}`), so the protocol tolerance lives in one tested place.
fn realtime_tool_result_message(item: &serde_json::Value) -> Option<serde_json::Value> {
    if item.get("type").and_then(serde_json::Value::as_str) != Some("function_call_output") {
        return None;
    }
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("tool_call_id"))
        .and_then(serde_json::Value::as_str)?;
    let output = item
        .get("output")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            // Nested content parts: join their text.
            let parts = item.get("content")?.as_array()?;
            let mut joined = String::new();
            for part in parts {
                if let Some(t) = part.get("text").and_then(serde_json::Value::as_str) {
                    joined.push_str(t);
                }
            }
            Some(joined)
        })
        .unwrap_or_default();
    Some(serde_json::json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": output,
    }))
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

/// Bound the replayed conversation. Tool turns add up to three messages rather
/// than two, so the cap counts messages and drops the oldest first.
fn trim_realtime_history(history: &mut Vec<serde_json::Value>) {
    const MAX_MESSAGES: usize = 40;
    let excess = history.len().saturating_sub(MAX_MESSAGES);
    if excess > 0 {
        history.drain(..excess);
    }
}

/// Drive one turn (text reply + spoken chunks), emitting events on `tx`.
fn run_turn(
    shared: &Arc<ServeShared>,
    state: &Arc<std::sync::Mutex<RealtimeSessionState>>,
    user_text: Option<String>,
    tx: &mpsc::UnboundedSender<String>,
) {
    let emit = |value: serde_json::Value| {
        let _ = tx.send(value.to_string());
    };
    let (instructions, max_tokens, want_audio, history, tools, tool_results) = {
        let st = state.lock().unwrap_or_else(|e| e.into_inner());
        (
            st.instructions.clone().unwrap_or_else(|| {
                "You are a helpful voice assistant. Keep replies short and conversational."
                    .to_owned()
            }),
            st.max_output_tokens as usize,
            st.output_audio,
            st.history.clone(),
            st.tools.clone(),
            st.pending_tool_results.clone(),
        )
    };
    // Order matters in a tool loop: the queued `tool` results must sit
    // immediately AFTER the assistant `tool_calls` message that produced them
    // (the end of `history`), and BEFORE any new user text.
    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({ "role": "system", "content": instructions })];
    messages.extend(history.iter().cloned());
    messages.extend(tool_results.iter().cloned());
    if let Some(text) = &user_text {
        messages.push(serde_json::json!({ "role": "user", "content": text }));
    }

    let cancel_flag = Arc::clone(&state.lock().unwrap_or_else(|e| e.into_inner()).cancel);
    let outcome = match realtime_text(shared, &messages, max_tokens, &tools, &cancel_flag) {
        Ok(outcome) => outcome,
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
    // Consume the queued results: they are now part of `history` below.
    state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pending_tool_results
        .clear();

    if !outcome.calls.is_empty() {
        // A tool turn is NOT spoken: the client executes the calls and answers
        // with `function_call_output`. voice.js owns approval + execution, the
        // server only orchestrates the protocol.
        let ids: Vec<String> = (0..outcome.calls.len())
            .map(|i| format!("call_{i}"))
            .collect();
        let ided: Vec<serde_json::Value> = outcome
            .calls
            .iter()
            .zip(&ids)
            .map(|(call, id)| {
                serde_json::json!({
                    "id": id,
                    "name": call.get("name").cloned().unwrap_or(serde_json::Value::Null),
                    "arguments": call.get("arguments").cloned()
                        .unwrap_or(serde_json::json!({})),
                })
            })
            .collect();
        // Type the arguments against the schemas the client declared, exactly
        // as the HTTP gateway does (`normalize_tool_calls`). The model's XML
        // carries every value as a string, so without this an `integer`
        // parameter reaches the client as `"3"`.
        let mut typed_calls: Vec<hipfire_runtime::prompt_frame::ToolCall> =
            serde_json::from_value(serde_json::Value::Array(ided.clone())).unwrap_or_default();
        crate::serve::complete::normalize_tool_calls(
            &mut typed_calls,
            &serde_json::json!({ "tools": tools }),
        );
        let ided: Vec<serde_json::Value> = typed_calls
            .iter()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                })
            })
            .collect();
        {
            let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
            for result in tool_results {
                st.history.push(result);
            }
            if let Some(text) = &user_text {
                st.history
                    .push(serde_json::json!({ "role": "user", "content": text }));
            }
            st.history.push(realtime_assistant_tool_message(&ided));
            trim_realtime_history(&mut st.history);
        }
        for (call, id) in ided.iter().zip(&ids) {
            let arguments = call
                .get("arguments")
                .map(|a| {
                    a.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| a.to_string())
                })
                .unwrap_or_else(|| "{}".to_owned());
            emit(serde_json::json!({
                "type": "response.function_call_arguments.done",
                "call_id": id,
                "name": call.get("name").cloned().unwrap_or(serde_json::Value::Null),
                "arguments": arguments,
            }));
        }
        emit(serde_json::json!({
            "type":"response.done",
            "response":{"status":"completed"},
        }));
        return;
    }

    let reply_visible = strip_realtime_tool_markers(&outcome.reply);
    for chunk in chunk_text_for_tts(&reply_visible) {
        emit(serde_json::json!({"type":"response.output_text.delta","delta":chunk}));
    }
    {
        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
        for result in tool_results {
            st.history.push(result);
        }
        if let Some(text) = &user_text {
            st.history
                .push(serde_json::json!({ "role": "user", "content": text }));
        }
        st.history.push(serde_json::json!({
            "role": "assistant",
            "content": reply_visible.clone(),
        }));
        trim_realtime_history(&mut st.history);
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

/// Most `conversation.item.create` entries (user items or tool results) a
/// session holds between two `response.create`s.
const MAX_QUEUED_ITEMS: usize = 64;

/// The `type: "error"` event the session sends for a refused client event.
fn realtime_error_event(message: &str) -> String {
    serde_json::json!({ "type": "error", "error": { "message": message } }).to_string()
}

async fn run_session(upgraded: Upgraded, shared: Arc<ServeShared>) -> Result<()> {
    let mut ws = upgraded;
    let state = Arc::new(std::sync::Mutex::new(RealtimeSessionState::new()));
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    // `serve.max_request_bytes` bounds one WebSocket message and the buffered
    // input audio, the same budget an HTTP request body gets.
    let max_bytes = usize::try_from(shared.max_request_bytes).unwrap_or(usize::MAX);

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
            inbound = ws_read_text(&mut ws, max_bytes) => {
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
                        // `tools` / `tool_choice` from the session. The server
                        // forwards the schemas on each turn's text stage; the
                        // model (LFM3-Omni) then answers with structured calls
                        // that `run_turn` releases as
                        // `response.function_call_arguments.done`. Liquid Agent
                        // keeps approvals and execution.
                        if let Some(tools) = event.pointer("/session/tools").and_then(|v| v.as_array()) {
                            st.tools = tools.clone();
                        }
                    }
                    Some("input_audio_buffer.append") => {
                        if let Some(audio) = event.get("audio").and_then(|v| v.as_str()) {
                            match base64::engine::general_purpose::STANDARD.decode(audio) {
                                Ok(pcm) => {
                                    let refused = {
                                        let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                                        let fits = st.input_pcm.len().saturating_add(pcm.len()) <= max_bytes;
                                        if fits {
                                            st.input_pcm.extend_from_slice(&pcm);
                                        }
                                        !fits
                                    };
                                    if refused {
                                        ws_write_text(&mut ws, &realtime_error_event(&format!(
                                            "input audio buffer is full ({max_bytes} bytes); commit or clear it"
                                        ))).await?;
                                    }
                                }
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
                        // A tool result goes into the conversation as a
                        // `role: "tool"` message, NOT into the user queue: the
                        // next `response.create` replays it in order so the
                        // model can answer on top of the call it made.
                        let refused = {
                            let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                            if let Some(result) = realtime_tool_result_message(&item) {
                                let full = st.pending_tool_results.len() >= MAX_QUEUED_ITEMS;
                                if !full {
                                    st.pending_tool_results.push(result);
                                }
                                full
                            } else if let Some(content) = item.get("content").and_then(|v| v.as_str()) {
                                let full = st.queued_items.len() >= MAX_QUEUED_ITEMS;
                                if !full {
                                    st.queued_items.push(content.to_owned());
                                }
                                full
                            } else {
                                false
                            }
                        };
                        if refused {
                            ws_write_text(&mut ws, &realtime_error_event(&format!(
                                "too many queued items ({MAX_QUEUED_ITEMS}); send response.create first"
                            ))).await?;
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
                            // A queued tool result also drives a turn: the
                            // model must get a chance to answer after the
                            // client executed the call. `run_turn` carries the
                            // result itself, so the user text is a neutral
                            // continuation prompt in that case.
                            let has_tool_result = !st.pending_tool_results.is_empty();
                            if parts.is_empty() && !has_tool_result {
                                None
                            } else {
                                st.cancel.store(false, Ordering::Relaxed);
                                // A turn driven only by tool results carries no
                                // new user message.
                                Some((!parts.is_empty()).then(|| parts.join("\n")))
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
    use super::{
        realtime_assistant_tool_message, realtime_tool_result_message,
        realtime_websocket_accept_key,
    };

    /// The OpenAI realtime `function_call_output` shape voice.js sends.
    #[test]
    fn function_call_output_becomes_a_tool_message() {
        let item = serde_json::json!({
            "type": "function_call_output",
            "call_id": "call_0",
            "output": "{\"ok\":true}",
        });
        let msg = realtime_tool_result_message(&item).expect("tool message");
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_0");
        assert_eq!(msg["content"], "{\"ok\":true}");
    }

    /// The nested-content variant, and the `tool_call_id` alias.
    #[test]
    fn function_call_output_accepts_nested_content_and_alias() {
        let item = serde_json::json!({
            "type": "function_call_output",
            "tool_call_id": "call_7",
            "content": [{"type": "text", "text": "12 degrees"}, {"type": "text", "text": " and raining"}],
        });
        let msg = realtime_tool_result_message(&item).expect("tool message");
        assert_eq!(msg["tool_call_id"], "call_7");
        assert_eq!(msg["content"], "12 degrees and raining");
    }

    /// Anything that is not a function-call output must fall through to the
    /// user-item path, not be misread as a tool result.
    #[test]
    fn ordinary_items_are_not_tool_results() {
        assert!(realtime_tool_result_message(&serde_json::json!({
            "type": "message", "role": "user", "content": "hello"
        }))
        .is_none());
        assert!(realtime_tool_result_message(&serde_json::json!({"content": "hi"})).is_none());
        // A function-call output with no call id cannot be correlated.
        assert!(realtime_tool_result_message(&serde_json::json!({
            "type": "function_call_output", "output": "x"
        }))
        .is_none());
    }

    /// The history assistant turn replays the calls in the flat shape
    /// `prompt_frame::Message` deserializes, with `arguments` as an OBJECT —
    /// LFM3's template raises on a JSON-encoded string.
    #[test]
    fn assistant_tool_message_keeps_arguments_an_object() {
        let calls = vec![serde_json::json!({
            "id": "call_0",
            "name": "book_room",
            "arguments": {"people": 3, "day": "Tuesday"},
        })];
        let msg = realtime_assistant_tool_message(&calls);
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "");
        assert!(msg["tool_calls"][0]["arguments"].is_object());
        // Round-trips through the daemon's message parser.
        let parsed: Vec<hipfire_runtime::prompt_frame::Message> =
            serde_json::from_value(serde_json::json!([msg.clone()])).expect("deserializes");
        assert_eq!(parsed[0].tool_calls[0].name, "book_room");
        assert_eq!(parsed[0].tool_calls[0].id.as_deref(), Some("call_0"));
    }

    #[test]
    fn realtime_accept_key_matches_rfc6455_example() {
        // RFC6455 §1.3 worked example.
        assert_eq!(
            realtime_websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    /// Feed `bytes` to `ws_read_text` with a `max_bytes` budget; return its
    /// result and the bytes the server wrote back.
    async fn read_with(bytes: &[u8], max_bytes: usize) -> (anyhow::Result<Option<String>>, Vec<u8>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (mut client, server) = tokio::io::duplex(1024);
        let mut server = hyper_util::rt::TokioIo::new(server);
        client.write_all(bytes).await.unwrap();
        client.shutdown().await.unwrap();
        let read = super::ws_read_text(&mut server, max_bytes).await;
        drop(server);
        let mut reply = Vec::new();
        client.read_to_end(&mut reply).await.unwrap();
        (read, reply)
    }

    /// A masked client frame carrying `payload` (zero mask key).
    fn masked(first: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 126);
        let mut f = vec![first, 0x80 | payload.len() as u8, 0, 0, 0, 0];
        f.extend_from_slice(payload);
        f
    }

    /// fivetide's #1066 review, blocking item 1: one 10-byte frame header
    /// declaring a 2^40-byte payload (`82 7f` + length) aborted the whole serve
    /// process (`memory allocation of 1099511627776 bytes failed`). Masked,
    /// the same header now closes with 1009 before anything is allocated.
    #[tokio::test]
    async fn declared_frame_length_is_bounded_before_allocation() {
        let mut frame = vec![0x82, 0xff];
        frame.extend_from_slice(&(1u64 << 40).to_be_bytes());
        let (read, reply) = read_with(&frame, 1 << 20).await;
        assert!(read.is_err());
        assert_eq!(reply, [0x88, 0x02, 0x03, 0xf1], "close 1009");
    }

    /// fivetide's exact bytes: the frame is unmasked, which a server must
    /// refuse (RFC6455 §5.1) — close 1002, no allocation.
    #[tokio::test]
    async fn unmasked_client_frame_is_refused() {
        let mut frame = vec![0x82, 0x7f];
        frame.extend_from_slice(&(1u64 << 40).to_be_bytes());
        let (read, reply) = read_with(&frame, 1 << 20).await;
        assert!(read.is_err());
        assert_eq!(reply, [0x88, 0x02, 0x03, 0xea], "close 1002");
    }

    /// Fragments are bounded as a whole message, not frame by frame.
    #[tokio::test]
    async fn fragmented_message_over_the_limit_is_refused() {
        let mut bytes = masked(0x01, b"0123456789");
        bytes.extend(masked(0x80, b"0123456789"));
        let (read, reply) = read_with(&bytes, 15).await;
        assert!(read.is_err());
        assert_eq!(reply, [0x88, 0x02, 0x03, 0xf1], "close 1009");
        let (read, _) = read_with(&bytes, 20).await;
        assert_eq!(read.unwrap().as_deref(), Some("01234567890123456789"));
    }
}
