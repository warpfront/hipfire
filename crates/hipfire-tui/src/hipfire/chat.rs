// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use hipfire_client::{stream_openai_chat, ClientError, OpenAiSseEvent, ServiceEndpoint};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug)]
pub enum ChatEvent {
    Reasoning(String),
    Content(String),
    /// Server-reported completion token count (final `usage` chunk, requested
    /// with `stream_options.include_usage`).
    Usage {
        completion_tokens: usize,
    },
    /// Prefill progress (`: prefill <done>/<total>` SSE comment): prompt
    /// tokens processed so far, before the first content/reasoning arrives.
    PrefillProgress {
        done: u64,
        total: u64,
    },
    Done,
    Error(String),
}

/// Request body for a streaming chat completion. Asks the server for the final
/// `usage` chunk so the TUI can show real token counts.
fn chat_request_body(
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f64>,
    top_p: Option<f64>,
) -> serde_json::Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream_options": { "include_usage": true },
    });
    // Per-session sampling overrides (set via /temp and /top_p).
    if let Some(t) = temperature {
        body["temperature"] = json!(t);
    }
    if let Some(p) = top_p {
        body["top_p"] = json!(p);
    }
    body
}

/// Map a typed SSE event to the TUI event it produces, if any.
fn chat_event_for(event: OpenAiSseEvent) -> Option<ChatEvent> {
    match event {
        OpenAiSseEvent::Reasoning { text } => Some(ChatEvent::Reasoning(text)),
        OpenAiSseEvent::Content { text } => Some(ChatEvent::Content(text)),
        OpenAiSseEvent::Usage { usage } => usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|n| ChatEvent::Usage {
                completion_tokens: n as usize,
            }),
        OpenAiSseEvent::PrefillProgress { done, total } => {
            Some(ChatEvent::PrefillProgress { done, total })
        }
        OpenAiSseEvent::Role { .. }
        | OpenAiSseEvent::ToolCall { .. }
        | OpenAiSseEvent::Finish { .. }
        | OpenAiSseEvent::Done => None,
    }
}

pub fn stream_chat(
    endpoint: &ServiceEndpoint,
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f64>,
    top_p: Option<f64>,
    tx: Sender<ChatEvent>,
    abort: Arc<AtomicBool>,
) -> Result<()> {
    let result = stream_chat_inner(endpoint, model, messages, temperature, top_p, &tx, &abort);
    if let Err(err) = result {
        let _ = tx.send(ChatEvent::Error(err.to_string()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_chat_inner(
    endpoint: &ServiceEndpoint,
    model: &str,
    messages: &[ChatMessage],
    temperature: Option<f64>,
    top_p: Option<f64>,
    tx: &Sender<ChatEvent>,
    abort: &Arc<AtomicBool>,
) -> Result<()> {
    let body = chat_request_body(model, messages, temperature, top_p);
    match stream_openai_chat(
        endpoint,
        body,
        Duration::from_secs(600),
        |event| {
            if let Some(ev) = chat_event_for(event) {
                let _ = tx.send(ev);
            }
            Ok(())
        },
        || abort.load(Ordering::Relaxed),
    ) {
        Ok(()) => {
            let _ = tx.send(ChatEvent::Done);
            Ok(())
        }
        // Explicit client cancel (Esc) is not an error and is never retried.
        Err(ClientError::Cancelled) => {
            let _ = tx.send(ChatEvent::Done);
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_reasoning_serialization() {
        // None is omitted; Some is emitted as reasoning_content.
        let msg_none = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: None,
        };
        let v = serde_json::to_value(&msg_none).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "answer");
        assert!(
            v.get("reasoning_content").is_none(),
            "None must not serialize"
        );

        let msg_some = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: Some("think".into()),
        };
        let v = serde_json::to_value(&msg_some).unwrap();
        assert_eq!(v["reasoning_content"], "think");
        assert_eq!(v["content"], "answer");

        // Deserializing legacy JSON without the field yields None.
        let legacy: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hello"
        }))
        .unwrap();
        assert!(legacy.reasoning_content.is_none());

        // Deserializing with reasoning_content populates it.
        let with_reasoning: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "plan"
        }))
        .unwrap();
        assert_eq!(with_reasoning.reasoning_content.as_deref(), Some("plan"));
    }

    #[test]
    fn chat_message_reasoning_empty_string_round_trips_but_omits_on_none() {
        // Some("") serializes as empty string (not omitted by is_none; caller should use None for empty).
        // The TUI fold normalizes empty to None before storing.
        let msg_empty = ChatMessage {
            role: "assistant".into(),
            content: "answer".into(),
            reasoning_content: Some(String::new()),
        };
        let v = serde_json::to_value(&msg_empty).unwrap();
        assert_eq!(v["reasoning_content"], "");
    }

    #[test]
    fn request_body_asks_for_stream_usage() {
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
        }];
        let body = chat_request_body("m", &msgs, Some(0.5), None);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["model"], "m");
        assert_eq!(body["temperature"], 0.5);
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn sse_usage_chunk_yields_usage_event() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut events = Vec::new();
        hipfire_client::read_openai_sse(
            std::io::Cursor::new(raw.as_bytes()),
            |ev| {
                if let Some(e) = chat_event_for(ev) {
                    events.push(e);
                }
                Ok(())
            },
            || false,
        )
        .expect("well-formed stream");
        assert!(matches!(&events[0], ChatEvent::Content(t) if t == "hi"));
        assert!(
            events.iter().any(|e| matches!(
                e,
                ChatEvent::Usage {
                    completion_tokens: 5
                }
            )),
            "usage chunk must produce a Usage event: {events:?}"
        );
    }

    #[test]
    fn sse_prefill_comment_yields_prefill_progress_event() {
        let raw = concat!(
            ": keepalive\n\n",
            ": prefill 512/2048\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut events = Vec::new();
        hipfire_client::read_openai_sse(
            std::io::Cursor::new(raw.as_bytes()),
            |ev| {
                if let Some(e) = chat_event_for(ev) {
                    events.push(e);
                }
                Ok(())
            },
            || false,
        )
        .expect("well-formed stream");
        assert!(matches!(
            &events[0],
            ChatEvent::PrefillProgress {
                done: 512,
                total: 2048
            }
        ));
        assert!(matches!(&events[1], ChatEvent::Content(t) if t == "hi"));
    }
}
