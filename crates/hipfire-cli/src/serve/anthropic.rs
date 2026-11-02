// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Anthropic Messages API adapter (`POST /v1/messages`).
//!
//! Concern: lowering a Messages request onto the chat-completions body the
//! serve pipeline already runs, and rendering that pipeline's [`Completion`]
//! and stream events back as an Anthropic `message` object or SSE events.
//! There is no second generation path: admission, batching eligibility,
//! session cache, the tool-call parser, think routing and stop handling are
//! the chat route's, so a Messages request and its lowered chat request are
//! the same request to everything below the HTTP layer.

use crate::serve::complete::{
    completion_hipfire, completion_timings, completion_usage, openai_finish_reason, Completion,
};
use crate::serve::http::{id_suffix, sse_event};
use hipfire_runtime::prompt_frame::ToolCall;
use serde_json::{json, Map, Value};

/// A Messages request lowered onto the chat-completions body.
#[derive(Debug)]
pub(crate) struct Lowered {
    pub(crate) chat: Value,
    pub(crate) stream: bool,
}

/// Lower a Messages request body. `Err` is a client error (HTTP 400) in
/// Anthropic's `invalid_request_error` shape.
pub(crate) fn lower_request(body: &Value) -> Result<Lowered, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_owned())?;
    let mut chat = Map::new();
    if let Some(model) = obj.get("model") {
        chat.insert("model".into(), model.clone());
    }
    let max_tokens = match present(obj, "max_tokens") {
        None => return Err("max_tokens: field required".into()),
        Some(value) => value
            .as_u64()
            .filter(|n| *n >= 1)
            .ok_or_else(|| "max_tokens must be a positive integer".to_owned())?,
    };
    chat.insert("max_tokens".into(), max_tokens.into());

    let mut messages = Vec::new();
    if let Some(system) = present(obj, "system") {
        let text = system_text(system)?;
        if !text.is_empty() {
            messages.push(json!({ "role": "system", "content": text }));
        }
    }
    let input = present(obj, "messages")
        .ok_or_else(|| "messages: field required".to_owned())?
        .as_array()
        .ok_or_else(|| "messages must be an array".to_owned())?;
    if input
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
    {
        return Err(
            "a final assistant message (response prefill) is not supported; \
             the last message must be a user turn"
                .into(),
        );
    }
    for (at, message) in input.iter().enumerate() {
        lower_message(message, at, &mut messages)?;
    }
    chat.insert("messages".into(), Value::Array(messages));

    if let Some(stops) = present(obj, "stop_sequences") {
        let list = stops
            .as_array()
            .filter(|list| list.iter().all(Value::is_string))
            .ok_or_else(|| "stop_sequences must be an array of strings".to_owned())?;
        // An empty list asks for nothing; forwarding `stop: []` would still
        // change batch eligibility on single-turn routes.
        if !list.is_empty() {
            chat.insert("stop".into(), stops.clone());
        }
    }
    for key in ["temperature", "top_p", "top_k"] {
        if let Some(value) = present(obj, key) {
            chat.insert(key.into(), value.clone());
        }
    }
    if let Some(tools) = present(obj, "tools") {
        let tools = lower_tools(tools)?;
        if !tools.is_empty() {
            chat.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(choice) = present(obj, "tool_choice") {
        chat.insert("tool_choice".into(), lower_tool_choice(choice)?);
    }
    lower_thinking(present(obj, "thinking"), &mut chat)?;
    if present(obj, "mcp_servers")
        .and_then(Value::as_array)
        .is_some_and(|servers| !servers.is_empty())
    {
        return Err("mcp_servers is not supported by hipfire serve".into());
    }
    let stream = match present(obj, "stream") {
        None => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => return Err("stream must be a boolean".into()),
    };
    Ok(Lowered {
        chat: Value::Object(chat),
        stream,
    })
}

/// A field that is present and not JSON `null`.
fn present<'a>(obj: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|value| !value.is_null())
}

/// `system`: a string or an array of `text` blocks, concatenated exactly
/// like chat-completions content parts (no separator).
fn system_text(system: &Value) -> Result<String, String> {
    match system {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for (at, block) in blocks.iter().enumerate() {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(format!("system.{at}: only text blocks are supported"));
                }
                text.push_str(block_text(block, &format!("system.{at}"))?);
            }
            Ok(text)
        }
        _ => Err("system must be a string or an array of text blocks".into()),
    }
}

fn block_text<'a>(block: &'a Value, path: &str) -> Result<&'a str, String> {
    block
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{path}.text: field required"))
}

fn block_type<'a>(block: &'a Value, path: &str) -> Result<&'a str, String> {
    block
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{path}.type: field required"))
}

fn lower_message(message: &Value, at: usize, out: &mut Vec<Value>) -> Result<(), String> {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("messages.{at}.role: field required"))?;
    let content = message
        .get("content")
        .filter(|content| !content.is_null())
        .ok_or_else(|| format!("messages.{at}.content: field required"))?;
    match role {
        "user" => lower_user(content, at, out),
        "assistant" => lower_assistant(content, at, out),
        other => Err(format!(
            "messages.{at}.role: unexpected role `{other}` (expected user or assistant)"
        )),
    }
}

/// A user turn: `tool_result` blocks become chat `tool` messages (in order,
/// ahead of the turn's own text); text concatenates like chat content
/// parts; one base64 image becomes an `image_url` part.
fn lower_user(content: &Value, at: usize, out: &mut Vec<Value>) -> Result<(), String> {
    let blocks = match content {
        Value::String(text) => {
            out.push(json!({ "role": "user", "content": text }));
            return Ok(());
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(format!(
                "messages.{at}.content must be a string or an array of content blocks"
            ))
        }
    };
    let mut text = String::new();
    let mut images = Vec::new();
    let mut tool_results = 0usize;
    for (j, block) in blocks.iter().enumerate() {
        let path = format!("messages.{at}.content.{j}");
        match block_type(block, &path)? {
            "text" => text.push_str(block_text(block, &path)?),
            "image" => images.push(image_part(block, &path)?),
            "tool_result" => {
                out.push(tool_result_message(block, &path)?);
                tool_results += 1;
            }
            other => {
                return Err(format!(
                    "{path}: content block type `{other}` is not supported in a user turn"
                ))
            }
        }
    }
    if images.is_empty() {
        if !text.is_empty() || tool_results == 0 {
            out.push(json!({ "role": "user", "content": text }));
        }
    } else {
        let mut parts = Vec::with_capacity(images.len() + 1);
        if !text.is_empty() {
            parts.push(json!({ "type": "text", "text": text }));
        }
        parts.extend(images);
        out.push(json!({ "role": "user", "content": parts }));
    }
    Ok(())
}

fn image_part(block: &Value, path: &str) -> Result<Value, String> {
    let source = block
        .get("source")
        .ok_or_else(|| format!("{path}.source: field required"))?;
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{path}.source.media_type: field required"))?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{path}.source.data: field required"))?;
            format!("data:{media_type};base64,{data}")
        }
        // The chat pipeline refuses remote images with its own message.
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{path}.source.url: field required"))?
            .to_owned(),
        _ => return Err(format!("{path}.source.type must be base64 or url")),
    };
    Ok(json!({ "type": "image_url", "image_url": { "url": url } }))
}

/// `tool_result` → chat `tool` message. `is_error: true` prefixes the
/// result with `Error: ` so the model sees the failure the client flagged.
fn tool_result_message(block: &Value, path: &str) -> Result<Value, String> {
    let id = block
        .get("tool_use_id")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{path}.tool_use_id: field required"))?;
    let mut text = String::new();
    match block.get("content") {
        None | Some(Value::Null) => {}
        Some(Value::String(content)) => text.push_str(content),
        Some(Value::Array(parts)) => {
            for (k, part) in parts.iter().enumerate() {
                let part_path = format!("{path}.content.{k}");
                match block_type(part, &part_path)? {
                    "text" => text.push_str(block_text(part, &part_path)?),
                    other => {
                        return Err(format!(
                            "{part_path}: `{other}` blocks inside tool_result are not supported"
                        ))
                    }
                }
            }
        }
        Some(_) => return Err(format!("{path}.content must be a string or an array")),
    }
    let content = if block.get("is_error").and_then(Value::as_bool) == Some(true) {
        format!("Error: {text}")
    } else {
        text
    };
    Ok(json!({ "role": "tool", "tool_call_id": id, "content": content }))
}

/// An assistant turn: text → `content`, `thinking` → `reasoning_content`
/// (replayed exactly like a chat client's), `tool_use` → `tool_calls` with
/// the input serialized as the OpenAI arguments string. `redacted_thinking`
/// is opaque and dropped.
fn lower_assistant(content: &Value, at: usize, out: &mut Vec<Value>) -> Result<(), String> {
    let blocks = match content {
        Value::String(text) => {
            out.push(json!({ "role": "assistant", "content": text }));
            return Ok(());
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(format!(
                "messages.{at}.content must be a string or an array of content blocks"
            ))
        }
    };
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for (j, block) in blocks.iter().enumerate() {
        let path = format!("messages.{at}.content.{j}");
        match block_type(block, &path)? {
            "text" => text.push_str(block_text(block, &path)?),
            "thinking" => reasoning.push_str(
                block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{path}.thinking: field required"))?,
            ),
            "redacted_thinking" => {}
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{path}.id: field required"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{path}.name: field required"))?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": input.to_string() },
                }));
            }
            other => {
                return Err(format!(
                    "{path}: content block type `{other}` is not supported in an assistant turn"
                ))
            }
        }
    }
    let mut entry = json!({ "role": "assistant", "content": text });
    if !reasoning.is_empty() {
        entry["reasoning_content"] = Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        entry["tool_calls"] = Value::Array(tool_calls);
    }
    out.push(entry);
    Ok(())
}

/// Client tools (`name`, `description`, `input_schema`) → chat function
/// tools. Anthropic server tools have no local executor and are refused.
fn lower_tools(tools: &Value) -> Result<Vec<Value>, String> {
    let tools = tools
        .as_array()
        .ok_or_else(|| "tools must be an array".to_owned())?;
    tools
        .iter()
        .enumerate()
        .map(|(k, tool)| {
            if let Some(kind) = tool.get("type").and_then(Value::as_str) {
                if kind != "custom" {
                    return Err(format!(
                        "tools.{k}: server tool type `{kind}` is not supported; \
                         only client tools (name, input_schema) are"
                    ));
                }
            }
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tools.{k}.name: field required"))?;
            let schema = tool
                .get("input_schema")
                .filter(|schema| schema.is_object())
                .ok_or_else(|| format!("tools.{k}.input_schema: field required (object)"))?;
            let mut function = json!({ "name": name, "parameters": schema });
            if let Some(description) = tool.get("description").and_then(Value::as_str) {
                function["description"] = Value::String(description.to_owned());
            }
            Ok(json!({ "type": "function", "function": function }))
        })
        .collect()
}

/// `auto` → `auto`, `any` → `required`, `none` → `none`, `tool` → the named
/// function. `disable_parallel_tool_use` is accepted and, like chat's
/// `parallel_tool_calls`, not enforced.
fn lower_tool_choice(choice: &Value) -> Result<Value, String> {
    match choice.get("type").and_then(Value::as_str) {
        Some("auto") => Ok(json!("auto")),
        Some("any") => Ok(json!("required")),
        Some("none") => Ok(json!("none")),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "tool_choice.name: field required".to_owned())?;
            Ok(json!({ "type": "function", "function": { "name": name } }))
        }
        _ => Err("tool_choice.type must be auto, any, tool or none".into()),
    }
}

/// `thinking` onto the reasoning contract: `enabled` (+ `budget_tokens` →
/// `max_think_tokens`) and `adaptive` turn thinking on, `disabled` off.
/// Absent follows the Messages API default, thinking off.
fn lower_thinking(thinking: Option<&Value>, chat: &mut Map<String, Value>) -> Result<(), String> {
    let mode = match thinking {
        None => "disabled",
        Some(thinking) => match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                if let Some(budget) = thinking.get("budget_tokens").filter(|v| !v.is_null()) {
                    let budget = budget.as_u64().ok_or_else(|| {
                        "thinking.budget_tokens must be a non-negative integer".to_owned()
                    })?;
                    chat.insert("max_think_tokens".into(), budget.into());
                }
                "enabled"
            }
            Some("adaptive") => "enabled",
            Some("disabled") => "disabled",
            _ => return Err("thinking.type must be enabled, adaptive or disabled".into()),
        },
    };
    chat.insert("thinking".into(), json!({ "type": mode }));
    Ok(())
}

/// `msg_…` id derived from the serve request id.
pub(crate) fn message_id(request_id: &str) -> String {
    format!("msg_{}", id_suffix(request_id))
}

/// The tool calls this completion releases: only on a tool-safe terminal,
/// exactly like the chat route's `tool_calls`.
fn released_calls(completion: &Completion) -> &[ToolCall] {
    if openai_finish_reason(&completion.done) == "tool_calls" {
        &completion.tool_calls
    } else {
        &[]
    }
}

fn tool_use_id(request_id: &str, index: usize, call: &ToolCall) -> String {
    call.id
        .clone()
        .unwrap_or_else(|| format!("toolu_{}_{index}", id_suffix(request_id)))
}

/// `stop` → `end_turn`, `length` → `max_tokens`, a released tool call →
/// `tool_use`. The daemon terminal does not say which stop sequence matched,
/// so a stop-sequence stop also reports `end_turn` (`stop_sequence: null`).
fn stop_reason(completion: &Completion) -> &'static str {
    match openai_finish_reason(&completion.done).as_str() {
        "tool_calls" if !completion.tool_calls.is_empty() => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

/// Anthropic usage: `input_tokens` excludes the prompt tokens served from
/// the session cache, which are reported as `cache_read_input_tokens`.
pub(crate) fn usage(completion: &Completion) -> Value {
    let usage = completion_usage(completion);
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    json!({
        "input_tokens": prompt.saturating_sub(cached),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": usage["completion_tokens"].as_u64().unwrap_or(0),
    })
}

/// Non-streaming `message` object.
pub(crate) fn message_json(completion: &Completion) -> Value {
    let mut content = Vec::new();
    if !completion.reasoning_content.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": completion.reasoning_content,
            "signature": "",
        }));
    }
    if !completion.content.is_empty() {
        content.push(json!({ "type": "text", "text": completion.content }));
    }
    for (index, call) in released_calls(completion).iter().enumerate() {
        content.push(json!({
            "type": "tool_use",
            "id": tool_use_id(&completion.id, index, call),
            "name": call.name,
            "input": call.arguments,
        }));
    }
    json!({
        "id": message_id(&completion.id),
        "type": "message",
        "role": "assistant",
        "model": completion.model,
        "content": content,
        "stop_reason": stop_reason(completion),
        "stop_sequence": Value::Null,
        "usage": usage(completion),
        "timings": completion_timings(completion),
        "hipfire": completion_hipfire(completion),
    })
}

/// Anthropic error body for an HTTP status.
pub(crate) fn error_body(message: &str, status: u16) -> Value {
    let kind = match status {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    };
    json!({ "type": "error", "error": { "type": kind, "message": message } })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
}

/// Streaming encoder: `message_start`, then one `content_block_start` /
/// `content_block_delta`… / `content_block_stop` run per thinking or text
/// span as the pipeline emits it, then the released `tool_use` blocks,
/// `message_delta` (stop reason and usage) and `message_stop`.
#[derive(Debug)]
pub(crate) struct MessageStream {
    request_id: String,
    model: String,
    open: Option<Block>,
    next_index: usize,
}

impl MessageStream {
    pub(crate) fn new(request_id: &str, model: &str) -> Self {
        Self {
            request_id: request_id.to_owned(),
            model: model.to_owned(),
            open: None,
            next_index: 0,
        }
    }

    /// `message_start`. Token counts are unknown until the terminal, so its
    /// usage is zero and `message_delta` carries the real counts.
    pub(crate) fn preamble(&mut self) -> Vec<u8> {
        sse_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": message_id(&self.request_id),
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                },
            }),
        )
    }

    /// One pipeline event (`token` / `reasoning`) as block events.
    pub(crate) fn event(&mut self, event: &Value) -> Option<Vec<u8>> {
        let block = match event.get("type").and_then(Value::as_str) {
            Some("token") => Block::Text,
            Some("reasoning") => Block::Thinking,
            _ => return None,
        };
        let text = event
            .get("text")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())?;
        let mut bytes = Vec::new();
        if self.open != Some(block) {
            self.close_block(&mut bytes);
            let content_block = match block {
                Block::Text => json!({ "type": "text", "text": "" }),
                Block::Thinking => json!({ "type": "thinking", "thinking": "", "signature": "" }),
            };
            self.start_block(content_block, &mut bytes);
            self.open = Some(block);
        }
        let delta = match block {
            Block::Text => json!({ "type": "text_delta", "text": text }),
            Block::Thinking => json!({ "type": "thinking_delta", "thinking": text }),
        };
        bytes.extend(sse_event(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": self.next_index - 1,
                "delta": delta,
            }),
        ));
        Some(bytes)
    }

    fn start_block(&mut self, content_block: Value, bytes: &mut Vec<u8>) {
        bytes.extend(sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": self.next_index,
                "content_block": content_block,
            }),
        ));
        self.next_index += 1;
    }

    fn close_block(&mut self, bytes: &mut Vec<u8>) {
        if self.open.take().is_some() {
            bytes.extend(sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": self.next_index - 1 }),
            ));
        }
    }

    /// Close the open block, release tool calls, then `message_delta` and
    /// `message_stop`.
    pub(crate) fn terminal(&mut self, completion: &Completion) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.close_block(&mut bytes);
        for (index, call) in released_calls(completion).iter().enumerate() {
            self.start_block(
                json!({
                    "type": "tool_use",
                    "id": tool_use_id(&completion.id, index, call),
                    "name": call.name,
                    "input": {},
                }),
                &mut bytes,
            );
            let block_index = self.next_index - 1;
            bytes.extend(sse_event(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": call.arguments.to_string(),
                    },
                }),
            ));
            bytes.extend(sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": block_index }),
            ));
        }
        bytes.extend(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason(completion),
                    "stop_sequence": Value::Null,
                },
                "usage": usage(completion),
                "timings": completion_timings(completion),
                "hipfire": completion_hipfire(completion),
            }),
        ));
        bytes.extend(sse_event(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));
        bytes
    }

    /// The `error` event ending a stream that fails after it started.
    pub(crate) fn failure(&mut self, message: &str, status: u16) -> Vec<u8> {
        sse_event("error", &error_body(message, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion(
        content: &str,
        reasoning: &str,
        calls: Vec<ToolCall>,
        finish: &str,
    ) -> Completion {
        Completion {
            id: "chatcmpl-7-3".into(),
            created: 1_700_000_000,
            model: "qwen3.5:4b".into(),
            content: content.into(),
            reasoning_content: reasoning.into(),
            preserve_thinking: false,
            tool_calls: calls,
            done: json!({
                "finish_reason": finish,
                "prompt_tokens": 11,
                "cached_tokens": 4,
                "tokens": 5,
            }),
            logprobs: None,
            reasoning: None,
        }
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            rendered_body: None,
        }
    }

    /// Parse an SSE byte stream into `(event, data)` pairs.
    fn events(bytes: &[u8]) -> Vec<(String, Value)> {
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .split("\n\n")
            .filter(|frame| !frame.is_empty())
            .map(|frame| {
                let mut lines = frame.lines();
                let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (event.to_owned(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[test]
    fn lowers_system_multi_turn_sampling_and_stops() {
        let lowered = lower_request(&json!({
            "model": "qwen3.5:4b",
            "max_tokens": 64,
            "system": [{ "type": "text", "text": "Be brief." }, { "type": "text", "text": " Always." }],
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [{ "type": "text", "text": "hello" }] },
                { "role": "user", "content": [{ "type": "text", "text": "again" }] },
            ],
            "stop_sequences": ["END"],
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "stream": true,
            "metadata": { "user_id": "u" },
        }))
        .unwrap();
        assert!(lowered.stream);
        assert_eq!(
            lowered.chat,
            json!({
                "model": "qwen3.5:4b",
                "max_tokens": 64,
                "messages": [
                    { "role": "system", "content": "Be brief. Always." },
                    { "role": "user", "content": "hi" },
                    { "role": "assistant", "content": "hello" },
                    { "role": "user", "content": "again" },
                ],
                "stop": ["END"],
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "thinking": { "type": "disabled" },
            })
        );
    }

    /// Text-only content blocks flatten to the plain-string chat shape, so a
    /// Messages request is batch-eligible exactly when its chat twin is.
    #[test]
    fn text_blocks_lower_to_the_batch_eligible_chat_shape() {
        let lowered = lower_request(&json!({
            "model": "qwen3.5:4b",
            "max_tokens": 8,
            "messages": [{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }],
            "stop_sequences": [],
        }))
        .unwrap();
        assert!(crate::serve::batch_messages_are_single_user(&lowered.chat));
        assert!(crate::serve::batch_messages_are_text_chat(&lowered.chat));
        assert!(
            lowered.chat.get("stop").is_none(),
            "empty stop list is not forwarded"
        );
        assert!(crate::serve::is_batch_eligible_request(
            &lowered.chat,
            None,
            Some("qwen3_5"),
            true,
            false
        ));
    }

    #[test]
    fn tool_use_round_trip_lowers_to_chat_tool_calls_and_tool_messages() {
        let lowered = lower_request(&json!({
            "model": "m",
            "max_tokens": 32,
            "tools": [{
                "name": "get_weather",
                "description": "Weather by city",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } },
            }],
            "tool_choice": { "type": "any" },
            "messages": [
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "need the tool", "signature": "" },
                    { "type": "text", "text": "Checking." },
                    { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": { "city": "Paris" } },
                ] },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": [{ "type": "text", "text": "18C" }] },
                    { "type": "text", "text": "thanks" },
                ] },
            ],
        }))
        .unwrap();
        let chat = &lowered.chat;
        assert_eq!(
            chat["tools"],
            json!([{ "type": "function", "function": {
                "name": "get_weather",
                "description": "Weather by city",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
            } }])
        );
        assert_eq!(chat["tool_choice"], json!("required"));
        assert_eq!(
            chat["messages"],
            json!([
                { "role": "user", "content": "weather in Paris?" },
                { "role": "assistant", "content": "Checking.", "reasoning_content": "need the tool",
                  "tool_calls": [{ "id": "toolu_1", "type": "function",
                                   "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } }] },
                { "role": "tool", "tool_call_id": "toolu_1", "content": "18C" },
                { "role": "user", "content": "thanks" },
            ])
        );
        // The pipeline's own normalizer sees the same call a chat client sends.
        let normalized =
            crate::serve::complete::normalize_openai_messages(chat.get("messages"), false);
        assert_eq!(
            normalized[1]["tool_calls"][0]["arguments"],
            json!({ "city": "Paris" })
        );
        assert_eq!(normalized[2]["tool_call_id"], json!("toolu_1"));
        // Tools keep the request off the batch route, as on chat.
        assert!(!crate::serve::is_batch_eligible_request(
            chat,
            None,
            Some("qwen3_5"),
            true,
            true
        ));
    }

    #[test]
    fn tool_choice_thinking_and_error_results_lower() {
        let base = json!({ "model": "m", "max_tokens": 8, "messages": [{ "role": "user", "content": "x" }] });
        let with = |key: &str, value: Value| {
            let mut body = base.clone();
            body[key] = value;
            lower_request(&body)
        };
        assert_eq!(
            with("tool_choice", json!({ "type": "auto" })).unwrap().chat["tool_choice"],
            json!("auto")
        );
        assert_eq!(
            with("tool_choice", json!({ "type": "none" })).unwrap().chat["tool_choice"],
            json!("none")
        );
        assert_eq!(
            with("tool_choice", json!({ "type": "tool", "name": "f" }))
                .unwrap()
                .chat["tool_choice"],
            json!({ "type": "function", "function": { "name": "f" } })
        );
        let enabled = with(
            "thinking",
            json!({ "type": "enabled", "budget_tokens": 2048 }),
        )
        .unwrap()
        .chat;
        assert_eq!(enabled["thinking"], json!({ "type": "enabled" }));
        assert_eq!(enabled["max_think_tokens"], json!(2048));
        let adaptive = with("thinking", json!({ "type": "adaptive" }))
            .unwrap()
            .chat;
        assert_eq!(adaptive["thinking"], json!({ "type": "enabled" }));
        assert!(adaptive.get("max_think_tokens").is_none());
        assert!(with("thinking", json!({ "type": "sometimes" })).is_err());

        let errored = lower_request(&json!({ "model": "m", "max_tokens": 8, "messages": [
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": true }] },
        ] }))
        .unwrap();
        assert_eq!(
            errored.chat["messages"],
            json!([{ "role": "tool", "tool_call_id": "t", "content": "Error: boom" }])
        );
    }

    #[test]
    fn lowering_refuses_what_it_cannot_honour() {
        let cases = [
            (json!({ "model": "m", "messages": [] }), "max_tokens"),
            (json!({ "model": "m", "max_tokens": 4 }), "messages"),
            (
                json!({ "model": "m", "max_tokens": 4, "messages": [
                    { "role": "user", "content": "x" }, { "role": "assistant", "content": "{" } ] }),
                "prefill",
            ),
            (
                json!({ "model": "m", "max_tokens": 4, "messages": [{ "role": "user", "content": "x" }],
                        "tools": [{ "type": "web_search_20250305", "name": "web_search" }] }),
                "server tool",
            ),
            (
                json!({ "model": "m", "max_tokens": 4, "messages": [{ "role": "user", "content": [
                    { "type": "document", "source": {} } ] }] }),
                "document",
            ),
            (
                json!({ "model": "m", "max_tokens": 4, "messages": [{ "role": "system", "content": "x" }] }),
                "role",
            ),
        ];
        for (body, needle) in cases {
            let error = lower_request(&body).expect_err(needle);
            assert!(error.contains(needle), "{needle}: {error}");
        }
    }

    #[test]
    fn base64_image_lowers_to_an_image_url_part() {
        let lowered = lower_request(&json!({ "model": "m", "max_tokens": 4, "messages": [{ "role": "user", "content": [
            { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } },
            { "type": "text", "text": "what is this" },
        ] }] }))
        .unwrap();
        assert_eq!(
            lowered.chat["messages"][0]["content"],
            json!([
                { "type": "text", "text": "what is this" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            ])
        );
        assert_eq!(
            crate::serve::complete::request_image_base64(lowered.chat.get("messages")).unwrap(),
            Some("AAAA".into())
        );
    }

    #[test]
    fn message_json_renders_thinking_text_tool_use_and_usage() {
        let done = completion(
            "Checking.",
            "need the tool",
            vec![call("get_weather", json!({ "city": "Paris" }))],
            "tool_calls",
        );
        let message = message_json(&done);
        assert_eq!(message["id"], json!("msg_7-3"));
        assert_eq!(message["type"], json!("message"));
        assert_eq!(message["stop_reason"], json!("tool_use"));
        assert_eq!(
            message["content"],
            json!([
                { "type": "thinking", "thinking": "need the tool", "signature": "" },
                { "type": "text", "text": "Checking." },
                { "type": "tool_use", "id": "toolu_7-3_0", "name": "get_weather", "input": { "city": "Paris" } },
            ])
        );
        assert_eq!(
            message["usage"],
            json!({ "input_tokens": 7, "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 4, "output_tokens": 5 })
        );

        // The rendered tool_use goes back in as the same chat call.
        let next = lower_request(&json!({ "model": "m", "max_tokens": 8, "messages": [
            { "role": "user", "content": "weather?" },
            { "role": "assistant", "content": message["content"] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_7-3_0", "content": "18C" }] },
        ] }))
        .unwrap();
        assert_eq!(
            next.chat["messages"][1]["tool_calls"][0],
            json!({ "id": "toolu_7-3_0", "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } })
        );
        assert_eq!(
            next.chat["messages"][1]["reasoning_content"],
            json!("need the tool")
        );
    }

    #[test]
    fn stop_reasons_map_and_unreleased_calls_stay_hidden() {
        assert_eq!(
            message_json(&completion("a", "", vec![], "stop"))["stop_reason"],
            json!("end_turn")
        );
        assert_eq!(
            message_json(&completion("a", "", vec![], "length"))["stop_reason"],
            json!("max_tokens")
        );
        let length_with_call =
            message_json(&completion("a", "", vec![call("f", json!({}))], "length"));
        assert_eq!(length_with_call["stop_reason"], json!("max_tokens"));
        assert_eq!(
            length_with_call["content"],
            json!([{ "type": "text", "text": "a" }])
        );
    }

    #[test]
    fn stream_event_order_is_message_start_blocks_delta_stop() {
        let mut stream = MessageStream::new("chatcmpl-7-3", "qwen3.5:4b");
        let mut bytes = stream.preamble();
        for event in [
            json!({ "type": "reasoning", "text": "need " }),
            json!({ "type": "reasoning", "text": "the tool" }),
            json!({ "type": "token", "text": "" }),
            json!({ "type": "token", "text": "Checking." }),
            json!({ "type": "tool_calls", "calls": [] }),
        ] {
            if let Some(chunk) = stream.event(&event) {
                bytes.extend(chunk);
            }
        }
        bytes.extend(stream.terminal(&completion(
            "Checking.",
            "need the tool",
            vec![call("get_weather", json!({ "city": "Paris" }))],
            "tool_calls",
        )));
        let events = events(&bytes);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        for (name, data) in &events {
            assert_eq!(
                data["type"],
                json!(name),
                "data.type matches the SSE event name"
            );
        }
        assert_eq!(events[0].1["message"]["id"], json!("msg_7-3"));
        assert_eq!(events[1].1["content_block"]["type"], json!("thinking"));
        assert_eq!(events[1].1["index"], json!(0));
        assert_eq!(
            events[3].1["delta"],
            json!({ "type": "thinking_delta", "thinking": "the tool" })
        );
        assert_eq!(events[5].1["content_block"]["type"], json!("text"));
        assert_eq!(
            events[6].1["delta"],
            json!({ "type": "text_delta", "text": "Checking." })
        );
        assert_eq!(events[6].1["index"], json!(1));
        assert_eq!(
            events[8].1["content_block"],
            json!({ "type": "tool_use", "id": "toolu_7-3_0", "name": "get_weather", "input": {} })
        );
        assert_eq!(
            events[9].1["delta"],
            json!({ "type": "input_json_delta", "partial_json": "{\"city\":\"Paris\"}" })
        );
        assert_eq!(events[10].1["index"], json!(2));
        assert_eq!(events[11].1["delta"]["stop_reason"], json!("tool_use"));
        assert_eq!(events[11].1["usage"]["output_tokens"], json!(5));
        assert_eq!(events[11].1["usage"]["input_tokens"], json!(7));
    }

    #[test]
    fn stream_without_content_still_closes_the_message() {
        let mut stream = MessageStream::new("chatcmpl-1-1", "m");
        let mut bytes = stream.preamble();
        bytes.extend(stream.terminal(&completion("", "", vec![], "length")));
        let names: Vec<String> = events(&bytes).into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["message_start", "message_delta", "message_stop"]);
        let failure = events(&stream.failure("boom", 500));
        assert_eq!(failure[0].0, "error");
        assert_eq!(
            failure[0].1,
            json!({ "type": "error", "error": { "type": "api_error", "message": "boom" } })
        );
    }

    #[test]
    fn error_types_follow_status() {
        for (status, kind) in [
            (400, "invalid_request_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (503, "overloaded_error"),
            (500, "api_error"),
        ] {
            assert_eq!(
                error_body("m", status)["error"]["type"],
                json!(kind),
                "{status}"
            );
        }
    }
}
