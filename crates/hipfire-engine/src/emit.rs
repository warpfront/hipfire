// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emit helpers — architecture-neutral JSONL wire protocol.
//!
//! Relocated verbatim from `crates/hipfire-daemon/src/main.rs` (wave 3).

use crate::terminal::active_attempt_id;

/// Whether the authoritative Jinja generation suffix opens a reasoning span.
/// This is deliberately tail-only: a literal `<think>` in user content must
/// not reclassify the assistant's visible answer as reasoning.
pub fn render_tail_opens_think(rendered: &str) -> bool {
    rendered.trim_end().ends_with("<think>")
}

/// Reduce the authoritative rendered-prompt state to the signal consumed by
/// speculative emitters. Jinja owns the generation suffix, so the request's
/// `assistant_prefix` is not authoritative once rendering succeeds.
pub fn spec_assistant_prefix(
    started_in_think: bool,
) -> hipfire_runtime::prompt_frame::AssistantPrefix {
    if started_in_think {
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    } else {
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain
    }
}

/// Emit `gen_start`. When `contract_version` is `Some(2)`, the CLI latches the
/// Increment-A semantic fold (visible-only tokens + buffered tool_calls).
/// Only routes that fully satisfy the producer contract may pass `Some(2)`.
/// Accepts any `Write` so deterministic tests can exercise the real writer
/// (including hostile request ids) without a live stdout handle.
pub fn emit_gen_start(
    stdout: &mut impl std::io::Write,
    id: &str,
    started_in_think: bool,
    contract_version: Option<u32>,
) {
    let envelope = hipfire_runtime::semantic::wire_gen_start(
        id,
        started_in_think,
        active_attempt_id(),
        contract_version,
    );
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// Qwen single-GPU AR and DFlash/spec share semantic contract v2 once the
/// router-backed producer path is active for the turn.
pub const QWEN_AR_SEMANTIC_CONTRACT_VERSION: u32 = 2;
pub const QWEN_DFLASH_SEMANTIC_CONTRACT_VERSION: u32 = QWEN_AR_SEMANTIC_CONTRACT_VERSION;

/// Borrowed per-token envelope shared by [`emit_visible_token`] and
/// [`emit_reasoning_token`].
///
/// Serializes field-for-field identically to the previous `serde_json::json!`
/// `Value`: `serde_json` builds with `preserve_order` workspace-wide (see
/// `crates/hipfire-runtime/Cargo.toml`), so the old `Value` kept `json!`
/// insertion order (`type`, `id`, `text`, `attempt_id`) — the same order as
/// the hand-rolled oracle in `hipfire-generate/src/dense.rs`. Streaming via
/// `to_writer` applies the same string escaping as `Value` display, with no
/// per-token `Value` allocation or text clone.
struct TokenEnvelope<'a> {
    kind: &'a str,
    id: &'a str,
    text: &'a str,
    attempt_id: u64,
}

impl serde::Serialize for TokenEnvelope<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut envelope = serializer.serialize_struct("TokenEnvelope", 4)?;
        envelope.serialize_field("type", self.kind)?;
        envelope.serialize_field("id", self.id)?;
        envelope.serialize_field("text", self.text)?;
        envelope.serialize_field("attempt_id", &self.attempt_id)?;
        envelope.end()
    }
}

/// Shared token writer: stream the envelope, then the JSONL newline, then
/// flush. Failure semantics mirror the previous `writeln!(stdout, "{envelope}")`
/// + `flush` (all errors ignored): `writeln!` skips its trailing newline when
/// the value write fails, so a failed serialize emits no extra newline, while
/// flush is always attempted.
fn write_token_envelope(stdout: &mut impl std::io::Write, kind: &str, id: &str, text: &str) {
    let envelope = TokenEnvelope {
        kind,
        id,
        text,
        attempt_id: active_attempt_id(),
    };
    if serde_json::to_writer(&mut *stdout, &envelope).is_ok() {
        let _ = stdout.write_all(b"\n");
    }
    let _ = stdout.flush();
}

/// Emit one classifier-authorized visible token event (no protocol markers).
pub fn emit_visible_token(stdout: &mut impl std::io::Write, id: &str, text: &str) {
    write_token_envelope(stdout, "token", id, text);
}

/// Emit one producer-classified reasoning fragment.
pub fn emit_reasoning_token(stdout: &mut impl std::io::Write, id: &str, text: &str) {
    write_token_envelope(stdout, "reasoning", id, text);
}

/// Canonical `{name, arguments}` array for staged terminal / tool_calls events.
pub fn tool_calls_canonical_json(
    calls: &[hipfire_runtime::prompt_frame::ToolCall],
) -> Vec<serde_json::Value> {
    calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "name": tc.name,
                "arguments": tc.arguments,
            })
        })
        .collect()
}

/// Embed canonical `calls` on a staged done payload when finish_reason is
/// tool_calls. Other finish reasons omit/empty calls so commit_ready and final
/// done stay payload-identical without a separate post-commit tool event.
pub fn stage_terminal_tool_calls(
    pending_done: &mut serde_json::Value,
    finish_reason: &str,
    calls: &[hipfire_runtime::prompt_frame::ToolCall],
) {
    if finish_reason == "tool_calls" {
        pending_done["calls"] = serde_json::Value::Array(tool_calls_canonical_json(calls));
    }
}

/// Emit structured `tool_calls` for a tool-safe terminal only.
pub fn emit_tool_calls_event(
    stdout: &mut impl std::io::Write,
    id: &str,
    calls: &[hipfire_runtime::prompt_frame::ToolCall],
) {
    if calls.is_empty() {
        return;
    }
    let calls_json = tool_calls_canonical_json(calls);
    let envelope = serde_json::json!({
        "type": "tool_calls",
        "id": id,
        "calls": calls_json,
        "attempt_id": active_attempt_id(),
    });
    let _ = writeln!(stdout, "{}", envelope);
}

// ── canonical JSON helpers (moved from daemon.rs for emit path) ──────────

pub fn write_canonical_json(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap()),
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, el) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical_json(el, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(*k).unwrap_or_else(|_| "\"\"".to_string()));
                out.push(':');
                write_canonical_json(&map[*k], out);
            }
            out.push('}');
        }
    }
}

pub fn canonical_json(v: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_json(v, &mut out);
    out
}

pub fn emit_error_with_id(
    stdout: &mut impl std::io::Write,
    id: &str,
    message: impl std::fmt::Display,
) {
    emit_active_attempt_error(
        stdout,
        Some(id),
        &message.to_string(),
        "internal",
        false,
        false,
    );
}

pub fn emit_error_no_id(stdout: &mut impl std::io::Write, message: impl std::fmt::Display) {
    emit_active_attempt_error(stdout, None, &message.to_string(), "internal", false, false);
}

pub fn emit_active_attempt_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    write_error_envelope(
        stdout,
        id,
        message,
        class,
        retryable,
        rolled_back,
        active_attempt_id(),
    );
}

pub fn emit_uncorrelated_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    write_error_envelope(stdout, id, message, class, retryable, rolled_back, 0);
}
fn write_error_envelope(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
    attempt_id: u64,
) {
    let mut envelope = serde_json::json!({
        "type": "error",
        "message": message,
        "class": class,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": attempt_id,
    });
    if let Some(id) = id {
        envelope["id"] = serde_json::Value::String(id.to_owned());
    }
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// Emit a single-line `{"type":"error","id":"...","message":"..."}` JSON
/// line on the IPC stream (active-attempt internal error, echoes TLS).
pub fn write_error(stdout: &mut impl std::io::Write, id: &str, message: &str) {
    emit_active_attempt_error(stdout, Some(id), message, "internal", false, false);
}

/// Typed active-attempt error writer used by generation failure paths.
pub fn write_typed_error(
    stdout: &mut impl std::io::Write,
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    emit_active_attempt_error(stdout, Some(id), message, class, retryable, rolled_back);
}

pub fn emit_qwen_ar_info(stdout: &mut impl std::io::Write, id: &str, message: &str) {
    let envelope = serde_json::json!({
        "type": "info",
        "id": id,
        "message": message,
        "attempt_id": active_attempt_id(),
    });
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

pub fn emit_qwen_ar_cancelled(
    stdout: &mut impl std::io::Write,
    id: &str,
    completion_tokens: usize,
) {
    let attempt_id = active_attempt_id();
    let aborted = hipfire_runtime::semantic::wire_aborted(id, "client_cancelled", attempt_id);
    let _ = writeln!(stdout, "{}", aborted);
    let done = hipfire_runtime::semantic::wire_aborted_done(id, completion_tokens, attempt_id);
    let _ = writeln!(stdout, "{}", done);
    let _ = stdout.flush();
}

#[cfg(test)]
mod token_emit_tests {
    use super::{emit_reasoning_token, emit_visible_token};
    use crate::terminal::set_active_attempt_id;

    #[test]
    fn visible_token_exact_bytes_with_escapes_and_unicode() {
        set_active_attempt_id(7);
        let mut sink = Vec::new();
        emit_visible_token(&mut sink, "r1", "a\"b\\c\ndé😀");
        assert_eq!(
            String::from_utf8(sink).unwrap(),
            "{\"type\":\"token\",\"id\":\"r1\",\"text\":\"a\\\"b\\\\c\\ndé😀\",\"attempt_id\":7}\n"
        );
    }

    #[test]
    fn reasoning_token_exact_bytes() {
        set_active_attempt_id(3);
        let mut sink = Vec::new();
        emit_reasoning_token(&mut sink, "req", "hello");
        assert_eq!(
            String::from_utf8(sink).unwrap(),
            "{\"type\":\"reasoning\",\"id\":\"req\",\"text\":\"hello\",\"attempt_id\":3}\n"
        );
    }

    #[test]
    fn token_bytes_match_legacy_value_oracle() {
        // Byte identity against the pre-change `json!` Value rendering
        // (insertion order under workspace `preserve_order`).
        let cases = [
            ("", ""),
            ("r1", "plain"),
            ("a\"b", "q\"\\"),
            ("uni", "é😀\t\r\n"),
            ("x", "\"\\/\u{8}\u{c}"),
        ];
        for (attempt, (id, text)) in cases.iter().copied().enumerate() {
            set_active_attempt_id(attempt as u64);
            let mut sink = Vec::new();
            emit_visible_token(&mut sink, id, text);
            let oracle = serde_json::json!({
                "type": "token",
                "id": id,
                "text": text,
                "attempt_id": attempt as u64,
            });
            assert_eq!(
                String::from_utf8(sink).unwrap(),
                format!("{oracle}\n"),
                "case {id:?}"
            );
        }
    }

    /// Writer that accepts `budget` bytes then fails every further write;
    /// records bytes and flush attempts.
    struct FailAfter {
        budget: usize,
        written: Vec<u8>,
        flush_calls: usize,
    }

    impl std::io::Write for FailAfter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.budget == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"));
            }
            let n = buf.len().min(self.budget);
            self.written.extend_from_slice(&buf[..n]);
            self.budget -= n;
            Ok(n)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flush_calls += 1;
            Ok(())
        }
    }

    #[test]
    fn failed_serialize_writes_no_newline_but_still_flushes() {
        set_active_attempt_id(1);
        let mut stdout = FailAfter {
            budget: 10,
            written: Vec::new(),
            flush_calls: 0,
        };
        // Must not panic; errors are ignored like the old `writeln!` path.
        emit_visible_token(&mut stdout, "req", "hello world, this is long");
        // Only the 10 accepted prefix bytes: no extra newline after failure.
        assert_eq!(stdout.written.len(), 10);
        assert!(!stdout.written.ends_with(b"\n"));
        assert_eq!(stdout.flush_calls, 1);
    }

    #[test]
    fn flush_failure_is_ignored() {
        struct FlushFails {
            written: Vec<u8>,
        }
        impl std::io::Write for FlushFails {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.written.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"))
            }
        }
        set_active_attempt_id(9);
        let mut stdout = FlushFails {
            written: Vec::new(),
        };
        emit_reasoning_token(&mut stdout, "req", "hi");
        assert_eq!(
            String::from_utf8(stdout.written).unwrap(),
            "{\"type\":\"reasoning\",\"id\":\"req\",\"text\":\"hi\",\"attempt_id\":9}\n"
        );
    }
}
