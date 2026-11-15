// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek Markup Language (DSML) — tool-calling format for V4 family.
//!
//! Spec source: `huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md`.
//!
//! DSML wraps tool calls in an XML-style block with special tokens
//! (`<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="…">`,
//! `<｜DSML｜parameter name="…" string="true|false">`, etc.). Parameter
//! value encoding depends on the `string="true|false"` attribute:
//!   - `string="true"`  → raw text payload (no JSON quoting).
//!   - `string="false"` → JSON-encoded value (numbers, booleans, arrays,
//!     objects, or quoted strings).
//!
//! This module provides:
//!   - [`tools_prompt_block`] — render the "## Tools" preamble that gets
//!     prepended to the system message when `tools` is non-empty.
//!   - [`render_assistant_tool_calls`] — serialise prior assistant
//!     tool_call messages back into DSML for multi-turn history.
//!   - [`StreamParser`] — incremental decoder for the streamed token
//!     output: emits [`StreamEvent::Token`] for plain content,
//!     [`StreamEvent::Reasoning`] for `<think>…</think>` content,
//!     [`StreamEvent::ToolCalls`] when a `<｜DSML｜tool_calls>` block
//!     closes. Markers split across token boundaries are buffered
//!     until they resolve.
//!
//! The parser is fail-closed on tool protocol: an unclosed
//! `<｜DSML｜tool_calls>` span at end-of-stream is reported as
//! [`StreamEvent::Malformed`] and must never be flushed as visible
//! [`StreamEvent::Token`] text. Callers terminalize with error XOR done
//! and suppress partial assistant-cache storage.

use serde_json::{json, Value};

// ── DSML constants — exact strings from the HF docs ─────────────────────

pub const TOOL_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
pub const TOOL_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
pub const INVOKE_OPEN_PREFIX: &str = "<｜DSML｜invoke name=\"";
pub const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
pub const PARAMETER_OPEN_PREFIX: &str = "<｜DSML｜parameter name=\"";
pub const PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

/// Variant inner tag observed in the wild: the V4F MQ2-Lloyd checkpoint
/// emits `<｜DSML｜tool name="X">…</｜DSML｜tool>` instead of the
/// canonical `<｜DSML｜invoke name="X">…</｜DSML｜invoke>` documented in
/// the HF encoder reference. Reproduced at both greedy (temp=0) and
/// sampled (temp=1.0, fixed seed) with the byte-identical HF system
/// prompt — the divergence is in the model weights, not our render.
/// We render `invoke` (matching HF) but parse both so the model's
/// emissions actually deserialise into structured tool calls.
const TOOL_OPEN_PREFIX_ALT: &str = "<｜DSML｜tool name=\"";
const TOOL_CLOSE_ALT: &str = "</｜DSML｜tool>";
pub const TOOL_RESULT_OPEN: &str = "<tool_result>";
pub const TOOL_RESULT_CLOSE: &str = "</tool_result>";

// ── Structured tool-call type ───────────────────────────────────────────

/// A single tool invocation produced by the model. `arguments` is the
/// reconstructed JSON object whose keys are parameter names and values
/// are decoded per the `string` attribute on each `<｜DSML｜parameter>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Turn-wide structured-call buffer shared by DS4 AR/EP absorb helpers,
/// [`crate::spec_emit::Deepseek4Emit`], and focused terminal tests.
///
/// Complete `ToolCalls` are retained here and never treated as live wire
/// events. [`Self::finalize`] is the only release/discard decision point and
/// requires the production terminal cause (`hit_length_cap`).
#[derive(Debug, Default, Clone)]
pub struct DsmlDeferredCalls {
    calls: Vec<ToolCall>,
    malformed: Option<String>,
}

/// Terminal outcome of [`DsmlDeferredCalls::finalize`].
#[derive(Debug, Clone, PartialEq)]
pub enum DsmlDeferredOutcome {
    /// Unclosed/malformed DSML — every buffered call discarded.
    Malformed { detail: String },
    /// Token cap — never tool-safe; no calls released.
    Length,
    /// Natural stop with no complete calls.
    Stop,
    /// Tool-safe terminal; release these calls exactly once.
    ToolCalls(Vec<ToolCall>),
}

impl DsmlDeferredCalls {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of complete calls buffered so far (may still be discarded).
    pub fn buffered_len(&self) -> usize {
        self.calls.len()
    }

    pub fn is_malformed(&self) -> bool {
        self.malformed.is_some()
    }

    pub fn malformed_detail(&self) -> Option<&str> {
        self.malformed.as_deref()
    }

    /// Absorb one parser event. Returns `Some` only for live visible channels
    /// (Token / Reasoning). ToolCalls are buffered; Malformed latches.
    pub fn absorb(&mut self, ev: StreamEvent) -> Option<StreamEvent> {
        match ev {
            StreamEvent::Token(t) => Some(StreamEvent::Token(t)),
            StreamEvent::Reasoning(t) => Some(StreamEvent::Reasoning(t)),
            StreamEvent::ToolCalls(calls) => {
                self.calls.extend(calls);
                None
            }
            StreamEvent::Malformed { detail } => {
                self.malformed = Some(detail);
                None
            }
        }
    }

    /// Absorb every event from an iterator (e.g. `parser.feed` / `finish`).
    /// Returns only the visible Token/Reasoning events, in order.
    pub fn absorb_all<I>(&mut self, events: I) -> Vec<StreamEvent>
    where
        I: IntoIterator<Item = StreamEvent>,
    {
        events
            .into_iter()
            .filter_map(|ev| self.absorb(ev))
            .collect()
    }

    /// Production terminal decision. Length always wins over buffered calls.
    /// Malformed discards every earlier complete call.
    pub fn finalize(self, hit_length_cap: bool) -> DsmlDeferredOutcome {
        if let Some(detail) = self.malformed {
            return DsmlDeferredOutcome::Malformed { detail };
        }
        if hit_length_cap {
            return DsmlDeferredOutcome::Length;
        }
        if self.calls.is_empty() {
            DsmlDeferredOutcome::Stop
        } else {
            DsmlDeferredOutcome::ToolCalls(self.calls)
        }
    }
}

impl ToolCall {
    /// Serialise this call back into DSML so it can be embedded into a
    /// prior-turn assistant message during multi-turn prompt rendering.
    pub fn to_dsml(&self) -> String {
        let mut out = String::new();
        out.push_str(INVOKE_OPEN_PREFIX);
        out.push_str(&xml_attr_escape(&self.name));
        out.push_str("\">\n");
        if let Value::Object(map) = &self.arguments {
            for (k, v) in map {
                // HF rule: string values are the raw text (no JSON
                // quoting); everything else is `json.dumps(v)` which
                // uses Python's default `", "` / `": "` separators —
                // match that via `py_compact_json` so the rendered
                // bytes are identical to the reference encoder.
                let (string_attr, payload) = match v {
                    Value::String(s) => ("true", s.clone()),
                    other => ("false", py_compact_json(other)),
                };
                out.push_str(PARAMETER_OPEN_PREFIX);
                out.push_str(&xml_attr_escape(k));
                out.push_str("\" string=\"");
                out.push_str(string_attr);
                out.push_str("\">");
                out.push_str(&payload);
                out.push_str(PARAMETER_CLOSE);
                out.push('\n');
            }
        }
        out.push_str(INVOKE_CLOSE);
        out
    }
}

/// Render a slice of tool calls into a `<｜DSML｜tool_calls>`-wrapped block
/// suitable for prepending to (or replacing the body of) a historical
/// assistant turn.
pub fn render_assistant_tool_calls(calls: &[ToolCall]) -> String {
    let mut out = String::new();
    out.push_str(TOOL_CALLS_OPEN);
    out.push('\n');
    for c in calls {
        out.push_str(&c.to_dsml());
        out.push('\n');
    }
    out.push_str(TOOL_CALLS_CLOSE);
    out
}

// ── Prompt-side: render the tools preamble ──────────────────────────────

/// The "## Tools" preamble. Byte-for-byte equivalent to `TOOLS_TEMPLATE`
/// in the upstream HF reference encoder
/// (`huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py`),
/// because the model's tool-emission behaviour is conditioned on this
/// exact string pattern — abbreviating or paraphrasing the template
/// causes the model to emit BPE-fragmented near-misses of the DSML
/// markers at greedy temperatures (observed 2026-05-23 with a shorter
/// preamble producing `<｜｜tool_coll｜>` / `<｜｜tool｜name=...>` garbage
/// instead of well-formed `<｜DSML｜tool_calls>` blocks).
///
/// `tools` is the OpenAI-format tools array — each entry shaped like
/// `{ "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }`.
/// We unwrap each entry's `function` field and render it as compact
/// (single-line) JSON, joined with `\n` — matching the HF encoder's
/// `tools_from_openai_format` + `"\n".join(tools_json)` pipeline.
pub fn tools_prompt_block(tools: &[Value]) -> String {
    let tool_schemas: String = tools
        .iter()
        .map(|t| {
            // Equivalent of HF's tools_from_openai_format: pull out the
            // inner "function" dict. Tolerate non-OpenAI entries by
            // falling back to the raw value. Rendered with Python-style
            // compact JSON (", " and ": " separators, insertion-order
            // keys) for byte parity with the HF encoder.
            let inner = t.get("function").unwrap_or(t);
            py_compact_json(inner)
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"{open}\" block like the following:

{open}
{invoke_open}$TOOL_NAME\">
{param_open}$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE{param_close}
...
{invoke_close}
{invoke_open}$TOOL_NAME2\">
...
{invoke_close}
{close}

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by {think_open}), you MUST output your complete reasoning inside {think_open}...{think_close} BEFORE any tool calls or final response.

Otherwise, output directly after {think_close} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
",
        open = TOOL_CALLS_OPEN,
        close = TOOL_CALLS_CLOSE,
        invoke_open = INVOKE_OPEN_PREFIX,
        invoke_close = INVOKE_CLOSE,
        param_open = PARAMETER_OPEN_PREFIX,
        param_close = PARAMETER_CLOSE,
        think_open = THINK_OPEN,
        think_close = THINK_CLOSE,
        tool_schemas = tool_schemas,
    )
}

/// Render a tool-result payload to embed inside the user-turn of a
/// follow-up message. The reference imatrix dataset renderer
/// (`gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py:101-108`)
/// XML-escapes `&`, `<`, `>` in the tool-result body so embedded
/// markup can't terminate the `</tool_result>` sentinel early or
/// confuse the model into reading an HTML-looking substring as a real
/// tag. V4F was trained on the escaped form, so we match it bytewise.
pub fn render_tool_result(result_text: &str) -> String {
    let escaped = escape_tool_result_body(result_text);
    format!("{TOOL_RESULT_OPEN}{escaped}{TOOL_RESULT_CLOSE}")
}

/// Match `build_ds4_imatrix_dataset.py::escape_tool_result` — escape
/// `&`, `<`, `>` so the tool-result body can't introduce stray markup
/// that the model reads as structural tokens. Order matters: `&` must
/// be escaped FIRST so the replacement strings (`&amp;`, `&lt;`,
/// `&gt;`) aren't double-escaped.
fn escape_tool_result_body(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Output-side: streaming parser ───────────────────────────────────────

/// Event emitted by [`StreamParser::feed`] / [`StreamParser::finish`].
#[derive(Debug, PartialEq)]
pub enum StreamEvent {
    /// Plain content the agent should surface to the user.
    Token(String),
    /// Content from inside a `<think>…</think>` block. Caller decides
    /// whether to surface it (e.g. mapped to OpenAI `reasoning_content`)
    /// or drop it.
    Reasoning(String),
    /// A completed `<｜DSML｜tool_calls>` block parsed into structured
    /// invocations.
    ToolCalls(Vec<ToolCall>),
    /// Fail-closed: tool protocol opened but never closed (or otherwise
    /// truncated) at end-of-stream. Callers must not surface the raw
    /// marker/body as [`Token`], must not expose executable tool calls,
    /// and should terminalize with a non-retryable error (not `done`).
    Malformed { detail: String },
}

/// Parser state at any moment.
#[derive(Debug)]
enum State {
    /// Plain content. Watching for `<think>` or `<｜DSML｜tool_calls>`.
    Normal,
    /// Inside a `<think>…</think>` block. Watching for `</think>`.
    InThink,
    /// Inside a `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block. Watching
    /// for `</｜DSML｜tool_calls>`.
    InToolCalls,
}

/// Incremental parser for the model's streamed token output. Feed each
/// token's decoded text via [`feed`](Self::feed); flush trailing buffered
/// content via [`finish`](Self::finish) at end-of-generation.
///
/// The parser handles markers that arrive split across token boundaries
/// (e.g. one token ends with `<th` and the next starts with `ink>`). It
/// holds back a small lookahead buffer (the length of the longest marker
/// prefix it might still be matching) and emits it as soon as the marker
/// is disambiguated.
pub struct StreamParser {
    state: State,
    /// Bytes seen but not yet emitted in the current state. In Normal,
    /// holds the unsettled tail (potential marker prefix). In InThink
    /// and InToolCalls, accumulates the block content.
    buf: String,
    /// Cached longest marker length (in bytes) we might still be inside,
    /// used as the "hold-back" window in Normal state. Computed once.
    normal_holdback: usize,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamParser {
    pub fn new() -> Self {
        let holdback = TOOL_CALLS_OPEN.len().max(THINK_OPEN.len());
        Self {
            state: State::Normal,
            buf: String::new(),
            normal_holdback: holdback,
        }
    }

    /// Construct a parser that starts already inside a `<think>…</think>`
    /// block. The first emitted token is treated as `Reasoning` content
    /// until the parser observes the closing `</think>`. Used by the
    /// V4F daemon's think-mode path where the prompt ends with the
    /// opening `<think>` tag (the model's first generated token is the
    /// start of the reasoning body, never the `<think>` itself), so a
    /// plain `new()` would mis-classify the entire reasoning stream as
    /// regular `Token` content.
    pub fn new_in_think() -> Self {
        let holdback = TOOL_CALLS_OPEN.len().max(THINK_OPEN.len());
        Self {
            state: State::InThink,
            buf: String::new(),
            normal_holdback: holdback,
        }
    }

    /// Feed a chunk of decoded text. May emit zero, one, or many events.
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buf.push_str(chunk);
        let mut events = Vec::new();
        loop {
            let progressed = self.step(&mut events);
            if !progressed {
                break;
            }
        }
        events
    }

    /// End-of-stream: flush whatever's buffered.
    ///
    /// - Unclosed `<think>` is still flushed as best-effort reasoning
    ///   (think is not executable protocol).
    /// - Unclosed `<｜DSML｜tool_calls>` **fails closed**: emits
    ///   [`StreamEvent::Malformed`] and never reconstructs the open
    ///   marker + body as visible [`StreamEvent::Token`] text.
    pub fn finish(mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match self.state {
            State::Normal => {
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Token(std::mem::take(&mut self.buf)));
                }
            }
            State::InThink => {
                // Unclosed think block — flush as reasoning. Defensive: if a
                // `</think>` is sitting unconsumed in the buffer at EOS (e.g. it
                // arrived in the final feed and the holdback never got a chance
                // to re-scan), split on it so the marker never leaks into
                // reasoning_content and the tail is surfaced as content.
                let buf = std::mem::take(&mut self.buf);
                match buf.find(THINK_CLOSE) {
                    Some(idx) => {
                        if idx > 0 {
                            events.push(StreamEvent::Reasoning(buf[..idx].to_string()));
                        }
                        let tail = &buf[idx + THINK_CLOSE.len()..];
                        if !tail.is_empty() {
                            events.push(StreamEvent::Token(tail.to_string()));
                        }
                    }
                    None if !buf.is_empty() => events.push(StreamEvent::Reasoning(buf)),
                    None => {}
                }
            }
            State::InToolCalls => {
                // Fail-closed: never saw the close. Do not flush raw DSML.
                events.push(StreamEvent::Malformed {
                    detail: "unclosed DSML tool_calls block at end of output".to_string(),
                });
            }
        }
        events
    }

    /// One inner step. Returns true if state advanced (more work might
    /// be possible), false if blocked waiting for more input.
    fn step(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        match self.state {
            State::Normal => self.step_normal(events),
            State::InThink => self.step_in_think(events),
            State::InToolCalls => self.step_in_tool_calls(events),
        }
    }

    fn step_normal(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        // Look for the FIRST occurrence of either opener.
        let first_think = self.buf.find(THINK_OPEN);
        let first_tools = self.buf.find(TOOL_CALLS_OPEN);
        let (cut, marker_len, new_state) = match (first_think, first_tools) {
            (None, None) => {
                // Emit everything except a potential marker prefix at
                // the tail. Hold back up to `normal_holdback` bytes.
                if self.buf.len() > self.normal_holdback {
                    let emit_len = self.buf.len() - self.normal_holdback;
                    // Don't split a multi-byte UTF-8 char.
                    let emit_len = utf8_safe_split(&self.buf, emit_len);
                    if emit_len > 0 {
                        let emitted: String = self.buf.drain(..emit_len).collect();
                        events.push(StreamEvent::Token(emitted));
                        return true;
                    }
                }
                return false;
            }
            (Some(t), None) => (t, THINK_OPEN.len(), State::InThink),
            (None, Some(t)) => (t, TOOL_CALLS_OPEN.len(), State::InToolCalls),
            (Some(a), Some(b)) => {
                if a <= b {
                    (a, THINK_OPEN.len(), State::InThink)
                } else {
                    (b, TOOL_CALLS_OPEN.len(), State::InToolCalls)
                }
            }
        };
        // Emit pre-marker bytes as Token.
        if cut > 0 {
            let head: String = self.buf.drain(..cut).collect();
            events.push(StreamEvent::Token(head));
        }
        // Drop the marker itself.
        let _: String = self.buf.drain(..marker_len).collect();
        self.state = new_state;
        true
    }

    fn step_in_think(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        // Watch for the think close AND — defensively — the tool-call opener.
        // A thinking model is instructed to emit `</think>` before any tool
        // calls, but it sometimes emits a complete `<｜DSML｜tool_calls>` block
        // WITHOUT closing think first (the reactive grammar permits it). If we
        // only matched `</think>`, that whole block would stream out as
        // reasoning_content and never parse — the agent dead-stops with no
        // tool calls (observed on a real Pi session). Treat a tool-call opener
        // seen inside think as an implicit think-close and hand off to the
        // InToolCalls parser. Whichever marker appears FIRST wins.
        let first_close = self.buf.find(THINK_CLOSE);
        let first_tools = self.buf.find(TOOL_CALLS_OPEN);
        let hit = match (first_close, first_tools) {
            (None, None) => None,
            (Some(c), None) => Some((c, THINK_CLOSE.len(), State::Normal)),
            (None, Some(t)) => Some((t, TOOL_CALLS_OPEN.len(), State::InToolCalls)),
            (Some(c), Some(t)) if c <= t => Some((c, THINK_CLOSE.len(), State::Normal)),
            (Some(_), Some(t)) => Some((t, TOOL_CALLS_OPEN.len(), State::InToolCalls)),
        };
        if let Some((idx, marker_len, new_state)) = hit {
            let content: String = self.buf.drain(..idx).collect();
            if !content.is_empty() {
                events.push(StreamEvent::Reasoning(content));
            }
            let _: String = self.buf.drain(..marker_len).collect();
            self.state = new_state;
            return true;
        }
        // No marker yet — emit reasoning up to a holdback large enough to hold
        // a partial `</think>` OR `<｜DSML｜tool_calls>` straddling the tail, so
        // the agent sees thinking progress without leaking a split marker.
        let holdback = THINK_CLOSE.len().max(TOOL_CALLS_OPEN.len());
        if self.buf.len() > holdback {
            let emit_len = utf8_safe_split(&self.buf, self.buf.len() - holdback);
            if emit_len > 0 {
                let emitted: String = self.buf.drain(..emit_len).collect();
                events.push(StreamEvent::Reasoning(emitted));
                return true;
            }
        }
        false
    }

    fn step_in_tool_calls(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        if let Some(idx) = self.buf.find(TOOL_CALLS_CLOSE) {
            let body: String = self.buf.drain(..idx).collect();
            let _: String = self.buf.drain(..TOOL_CALLS_CLOSE.len()).collect();
            self.state = State::Normal;
            // Any invoke or parameter that does not parse cleanly (misspelled
            // close, missing quote, empty block) fails the whole call: a partial
            // parse would merge, drop or empty calls (HIPF-262, HIPF-320).
            match parse_tool_calls_body(&body) {
                Ok(calls) => events.push(StreamEvent::ToolCalls(calls)),
                Err(detail) => events.push(StreamEvent::Malformed { detail }),
            }
            return true;
        }
        // No close yet: don't emit anything (tool calls are atomic; we
        // surface them only when the block is complete).
        false
    }
}

// ── Tool-call body parsing ──────────────────────────────────────────────

/// Parse the body of a `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` block
/// (without the wrapper tags themselves) into a vector of [`ToolCall`].
///
/// Strict: every invoke must carry a clean `name="…">` attr and its own
/// close before the next invoke open; every parameter must close before
/// the next parameter open. Any violation, or a block with no invoke,
/// returns `Err` so the caller fails the turn instead of releasing merged,
/// dropped or empty calls. Two observed V4F MQ2-Lloyd variants stay
/// accepted: the bare `</｜DSML｜>` invoke close and a parameter without
/// the `string="…"` attr.
pub fn parse_tool_calls_body(body: &str) -> Result<Vec<ToolCall>, String> {
    /// Generic short close — observed in V4F MQ2-Lloyd output when the
    /// model truncates the canonical close. The matched close wins when
    /// it comes first.
    const GENERIC_CLOSE: &str = "</｜DSML｜>";
    let next_open = |s: &str| -> Option<(usize, usize, &'static str)> {
        let invoke_hit = s
            .find(INVOKE_OPEN_PREFIX)
            .map(|i| (i, INVOKE_OPEN_PREFIX.len(), INVOKE_CLOSE));
        let tool_hit = s
            .find(TOOL_OPEN_PREFIX_ALT)
            .map(|i| (i, TOOL_OPEN_PREFIX_ALT.len(), TOOL_CLOSE_ALT));
        match (invoke_hit, tool_hit) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        }
    };
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some((open_rel, open_len, close_marker)) = next_open(&body[cursor..]) {
        // Only whitespace may sit between invokes: anything else is a
        // misspelled open whose call would be dropped silently.
        stray_text(&body[cursor..cursor + open_rel])?;
        let abs = cursor + open_rel + open_len;
        let close_attr = body[abs..]
            .find("\">")
            .map(|i| abs + i)
            .ok_or("DSML invoke without a closed name attr")?;
        let name = &body[abs..close_attr];
        if name.is_empty() || name.contains(['"', '<', '>', '\n']) {
            return Err(format!("DSML invoke with a malformed name attr: {name:?}"));
        }
        let body_start = close_attr + 2;
        let rest = &body[body_start..];
        let close = [
            rest.find(close_marker).map(|i| (i, close_marker.len())),
            rest.find(GENERIC_CLOSE).map(|i| (i, GENERIC_CLOSE.len())),
        ]
        .into_iter()
        .flatten()
        .min_by_key(|c| c.0);
        let (rel_off, used_close_len) = match close {
            Some(c) if next_open(rest).is_none_or(|o| c.0 < o.0) => c,
            _ => return Err(format!("DSML invoke {name:?} without its own close")),
        };
        let invoke_close = body_start + rel_off;
        out.push(ToolCall {
            name: name.to_string(),
            arguments: parse_parameters(&body[body_start..invoke_close])?,
        });
        cursor = invoke_close + used_close_len;
    }
    stray_text(&body[cursor..])?;
    if out.is_empty() {
        return Err("DSML tool_calls block without a parseable invoke".to_string());
    }
    Ok(out)
}

/// Parse the parameters inside a single invoke block into a JSON object
/// keyed by parameter name. A parameter without the `string="…"` attr is
/// read as a plain string; any other malformed parameter is an `Err`.
fn parse_parameters(body: &str) -> Result<Value, String> {
    let mut map = serde_json::Map::new();
    let mut cursor = 0;
    while let Some(param_start) = body[cursor..].find(PARAMETER_OPEN_PREFIX) {
        stray_text(&body[cursor..cursor + param_start])?;
        let abs = cursor + param_start + PARAMETER_OPEN_PREFIX.len();
        let bad = || format!("malformed DSML parameter at {:?}", &body[abs..]);
        // name="..." [string="true|false"]>
        let tag_end = abs + body[abs..].find('>').ok_or_else(bad)?;
        let tag = &body[abs..tag_end];
        let name_end = tag.find('"').ok_or_else(bad)?;
        let name = &tag[..name_end];
        let attrs = &tag[name_end + 1..];
        if name.is_empty() || name.contains(['<', '\n']) {
            return Err(bad());
        }
        let is_string = match attrs.trim() {
            "" => true,
            "string=\"true\"" => true,
            "string=\"false\"" => false,
            _ => return Err(bad()),
        };
        let content_start = tag_end + 1;
        let content_end = content_start
            + body[content_start..]
                .find(PARAMETER_CLOSE)
                .ok_or_else(bad)?;
        let raw_value = &body[content_start..content_end];
        if raw_value.contains(PARAMETER_OPEN_PREFIX) {
            return Err(bad());
        }
        let value: Value = if is_string {
            Value::String(raw_value.to_string())
        } else {
            serde_json::from_str(raw_value.trim())
                .unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        map.insert(name.to_string(), value);
        cursor = content_end + PARAMETER_CLOSE.len();
    }
    stray_text(&body[cursor..])?;
    Ok(Value::Object(map))
}

/// Text between DSML tags must be whitespace (the grammar emits only `\n`);
/// anything else is a misspelled tag whose call or parameter would be lost.
fn stray_text(s: &str) -> Result<(), String> {
    if s.trim().is_empty() {
        Ok(())
    } else {
        Err(format!("stray text between DSML tags: {s:?}"))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Render a `serde_json::Value` using Python's `json.dumps(...,
/// ensure_ascii=False)` default formatting: single-line, `", "` between
/// items, `": "` between key/value. Equivalent to Python's default
/// `separators=(", ", ": ")`. With the crate-level `preserve_order`
/// feature on `serde_json`, object key order survives from input to
/// output — matching the HF encoder's `tools_from_openai_format` →
/// `json.dumps` pipeline byte-for-byte.
fn py_compact_json(v: &Value) -> String {
    let mut out = String::new();
    write_py_compact(v, &mut out);
    out
}

fn write_py_compact(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            // Use serde_json's string escaping — Python's default
            // `ensure_ascii=False` keeps non-ASCII raw, which matches
            // serde_json's default for string values.
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string()));
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_compact(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".to_string()));
                out.push_str(": ");
                write_py_compact(val, out);
            }
            out.push('}');
        }
    }
}

/// XML-attribute escape: replace the four characters that would break
/// attribute parsing. The DSML format is forgiving (no quoting required
/// for parameter values) but tool/parameter names need to be valid.
fn xml_attr_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("&quot;"),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Find the largest split point ≤ `n` that doesn't cut through a
/// multi-byte UTF-8 character.
fn utf8_safe_split(s: &str, n: usize) -> usize {
    let bytes = s.as_bytes();
    let mut k = n.min(bytes.len());
    while k > 0 && (bytes[k] & 0b1100_0000) == 0b1000_0000 {
        k -= 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(p: &mut StreamParser, s: &str) -> Vec<StreamEvent> {
        p.feed(s)
    }

    #[test]
    fn plain_text_pass_through() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "hello world");
        events.extend(p.finish());
        let joined: String = events
            .iter()
            .map(|e| match e {
                StreamEvent::Token(t) => t.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(joined, "hello world");
    }

    #[test]
    fn think_block_isolated() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "before<think>reasoning</think>after");
        events.extend(p.finish());
        let mut tokens = String::new();
        let mut reasoning = String::new();
        for e in &events {
            match e {
                StreamEvent::Token(t) => tokens.push_str(t),
                StreamEvent::Reasoning(t) => reasoning.push_str(t),
                _ => {}
            }
        }
        assert_eq!(tokens, "beforeafter");
        assert_eq!(reasoning, "reasoning");
    }

    #[test]
    fn think_split_across_chunks() {
        let mut p = StreamParser::new();
        let mut events = drain(&mut p, "x<th");
        events.extend(drain(&mut p, "ink>r1"));
        events.extend(drain(&mut p, "r2</th"));
        events.extend(drain(&mut p, "ink>y"));
        events.extend(p.finish());
        let mut reasoning = String::new();
        let mut tokens = String::new();
        for e in &events {
            match e {
                StreamEvent::Token(t) => tokens.push_str(t),
                StreamEvent::Reasoning(t) => reasoning.push_str(t),
                _ => {}
            }
        }
        assert_eq!(reasoning, "r1r2");
        assert_eq!(tokens, "xy");
    }

    #[test]
    fn tool_call_round_trip() {
        let dsml = format!(
            "{open}\n{io}fn1\">\n{po}arg1\" string=\"true\">value one{pc}\n{po}arg2\" string=\"false\">42{pc}\n{ic}\n{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            io = INVOKE_OPEN_PREFIX,
            ic = INVOKE_CLOSE,
            po = PARAMETER_OPEN_PREFIX,
            pc = PARAMETER_CLOSE,
        );
        let mut p = StreamParser::new();
        let mut events = p.feed(&dsml);
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].name, "fn1");
        assert_eq!(calls[0][0].arguments["arg1"], json!("value one"));
        assert_eq!(calls[0][0].arguments["arg2"], json!(42));
    }

    #[test]
    fn tool_calls_inside_unclosed_think_are_parsed() {
        // Regression (real Pi session): a thinking model emitted a complete
        // `<｜DSML｜tool_calls>` block WITHOUT first closing `<think>`. The
        // parser must treat the tool-call opener as an implicit think-close
        // and still surface the calls — otherwise the whole block is swallowed
        // into reasoning_content and the agent dead-stops (stopReason=stop,
        // zero tool calls).
        let dsml = format!(
            "{think}let me look around{open}\n{io}bash\">\n{po}command\" string=\"true\">ls -la{pc}\n{ic}\n{close}",
            think = THINK_OPEN,
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            io = INVOKE_OPEN_PREFIX,
            ic = INVOKE_CLOSE,
            po = PARAMETER_OPEN_PREFIX,
            pc = PARAMETER_CLOSE,
        );
        let mut p = StreamParser::new();
        let mut events = p.feed(&dsml);
        events.extend(p.finish());
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Reasoning(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "let me look around");
        assert_eq!(calls.len(), 1, "tool calls must be surfaced, not swallowed");
        assert_eq!(calls[0][0].name, "bash");
        assert_eq!(calls[0][0].arguments["command"], json!("ls -la"));
    }

    #[test]
    fn malformed_unclosed_tool_calls_fail_closed_at_finish() {
        // Opens but never closes — finish() must report Malformed and
        // must not surface the open marker / body as Token text.
        let mut p = StreamParser::new();
        let _ = p.feed(TOOL_CALLS_OPEN);
        let _ = p.feed("\n<｜DSML｜invoke name=\"f\">");
        let events = p.finish();
        let tokens: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tokens.is_empty(),
            "fail-closed must not flush unclosed DSML as Token, got {tokens:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolCalls(_)))
                == false,
            "unclosed protocol must not yield ToolCalls"
        );
        let detail = events.iter().find_map(|e| match e {
            StreamEvent::Malformed { detail } => Some(detail.as_str()),
            _ => None,
        });
        assert!(
            detail.is_some_and(|d| d.contains("unclosed") && d.contains("tool_calls")),
            "expected Malformed detail, events={events:?}"
        );
    }

    #[test]
    fn closed_tool_calls_without_parseable_invoke_is_malformed() {
        // HIPF-262: DS4 DSpark emitted this exact block; it parsed to zero
        // calls and the turn ended as a plain stop.
        let mut p = StreamParser::new();
        let mut events = p.feed(
            "Sure.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
</｜DSML｜invinoke>\n</｜DSML｜tool_calls>",
        );
        events.extend(p.finish());
        assert!(!events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCalls(_))));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Malformed { .. })),
            "events={events:?}"
        );
    }

    #[test]
    fn malformed_unclosed_after_prose_keeps_prior_tokens_only() {
        let mut p = StreamParser::new();
        let mut events = p.feed("hello ");
        events.extend(p.feed(TOOL_CALLS_OPEN));
        events.extend(p.feed("partial"));
        events.extend(p.finish());
        let tokens: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(tokens, "hello ");
        assert!(!tokens.contains(TOOL_CALLS_OPEN));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::Malformed { .. })));
    }

    #[test]
    fn parses_tool_variant_inner_tag() {
        // The V4F MQ2-Lloyd checkpoint emits `<｜DSML｜tool name="X">`
        // instead of `<｜DSML｜invoke name="X">` (observed
        // deterministically at temp=0 AND temp=1.0/seed=42). Reproduces
        // the exact token stream the model committed during 2026-05-23
        // diagnostic — the parser must accept both.
        let body = "\n\n<｜DSML｜tool_calls>\n\
<｜DSML｜tool name=\"read\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/test.txt</｜DSML｜parameter>\n\
</｜DSML｜tool>\n\
</｜DSML｜tool_calls>";
        let mut p = StreamParser::new();
        let mut events = p.feed(body);
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "expected one ToolCalls event");
        assert_eq!(calls[0].len(), 1, "expected one tool call");
        assert_eq!(calls[0][0].name, "read");
        assert_eq!(calls[0][0].arguments["path"], json!("/tmp/test.txt"));
    }

    #[test]
    fn new_in_think_treats_stream_as_reasoning_until_close() {
        // Simulates the V4F think-mode path: prompt ends with `<think>`,
        // the parser starts inside the block, every emitted token is
        // reasoning until `</think>` lands, then content resumes.
        let mut p = StreamParser::new_in_think();
        let mut events = p.feed("step one ");
        events.extend(p.feed("step two</think>"));
        events.extend(p.feed("final answer"));
        events.extend(p.finish());
        let mut reasoning = String::new();
        let mut tokens = String::new();
        for e in &events {
            match e {
                StreamEvent::Reasoning(t) => reasoning.push_str(t),
                StreamEvent::Token(t) => tokens.push_str(t),
                _ => {}
            }
        }
        assert_eq!(reasoning, "step one step two");
        assert_eq!(tokens, "final answer");
    }

    #[test]
    fn recovers_from_bare_generic_close() {
        // V4F MQ2-Lloyd has been observed truncating the close to the
        // bare `</｜DSML｜>` form. The parser must still recover the
        // invoke's name + parameters.
        let body = "<｜DSML｜tool_calls>\n\
<｜DSML｜tool name=\"edit\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/x</｜DSML｜parameter>\n\
</｜DSML｜>\n\
</｜DSML｜tool_calls>";
        let calls = parse_tool_calls_body(
            &body
                .strip_prefix("<｜DSML｜tool_calls>\n")
                .unwrap()
                .strip_suffix("</｜DSML｜tool_calls>")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "edit");
        assert_eq!(calls[0].arguments["path"], json!("/tmp/x"));
    }

    #[test]
    fn continues_past_param_without_string_attr() {
        // The model occasionally emits `<｜DSML｜parameter name="X">val</｜DSML｜parameter>`
        // with no `string="..."` attr. Treat it as a string param and
        // keep going so later well-formed params are still collected.
        let body = "\
<｜DSML｜invoke name=\"f\">\n\
<｜DSML｜parameter name=\"a\">raw</｜DSML｜parameter>\n\
<｜DSML｜parameter name=\"b\" string=\"false\">42</｜DSML｜parameter>\n\
</｜DSML｜invoke>";
        let calls = parse_tool_calls_body(body).unwrap();
        assert_eq!(calls.len(), 1);
        // First param recovered as a plain string.
        assert_eq!(calls[0].arguments["a"], json!("raw"));
        // Second param parses as JSON int — proving the loop didn't bail.
        assert_eq!(calls[0].arguments["b"], json!(42));
    }

    fn feed_block(inner: &str) -> Vec<StreamEvent> {
        let mut p = StreamParser::new();
        let mut events = p.feed(&format!("{TOOL_CALLS_OPEN}\n{inner}{TOOL_CALLS_CLOSE}"));
        events.extend(p.finish());
        events
    }

    fn assert_malformed(inner: &str) {
        let events = feed_block(inner);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Malformed { .. }))
                && !events
                    .iter()
                    .any(|e| matches!(e, StreamEvent::ToolCalls(_))),
            "expected Malformed only, events={events:?}"
        );
    }

    const WEATHER: &str =
        "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n";
    const TIME: &str = "<｜DSML｜parameter name=\"tz\" string=\"true\">CET</｜DSML｜parameter>\n";

    #[test]
    fn first_invoke_close_misspelled_is_malformed_not_merged() {
        // HIPF-320 case B: was one get_weather with merged {city, tz}.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n{WEATHER}</｜DSML｜invinoke>\n\
             <｜DSML｜invoke name=\"get_time\">\n{TIME}</｜DSML｜invoke>\n"
        ));
    }

    #[test]
    fn second_invoke_close_misspelled_is_malformed_not_dropped() {
        // HIPF-320 case C: was get_weather only, get_time silently dropped.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n{WEATHER}</｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"get_time\">\n{TIME}</｜DSML｜invinoke>\n"
        ));
    }

    #[test]
    fn parameter_close_misspelled_is_malformed_not_empty_args() {
        // HIPF-320 case D: was get_weather with {} args.
        assert_malformed(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜paramter>\n\
             </｜DSML｜invoke>\n",
        );
        // A misspelled close followed by a good param must not merge values.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜paramter>\n\
             {TIME}</｜DSML｜invoke>\n"
        ));
    }

    #[test]
    fn invoke_name_attr_missing_quote_is_malformed() {
        // HIPF-320 case F: name swallowed the next invoke open.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather>\n{WEATHER}</｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"get_time\">\n{TIME}</｜DSML｜invoke>\n"
        ));
    }

    #[test]
    fn well_formed_two_invokes_still_parse() {
        let events = feed_block(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n{WEATHER}</｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"get_time\">\n{TIME}</｜DSML｜invoke>\n"
        ));
        let calls = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .expect("tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
        assert_eq!(calls[1].arguments, json!({"tz": "CET"}));
    }

    #[test]
    fn misspelled_open_or_attr_is_malformed_not_dropped() {
        // Misspelled invoke open, after and before a good invoke: the
        // misspelled call was dropped silently.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n{WEATHER}</｜DSML｜invoke>\n\
             <｜DSML｜invinoke name=\"get_time\">\n{TIME}</｜DSML｜invoke>\n"
        ));
        assert_malformed(&format!(
            "<｜DSML｜invinoke name=\"get_time\">\n{TIME}</｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"get_weather\">\n{WEATHER}</｜DSML｜invoke>\n"
        ));
        // Misspelled parameter open before a good one.
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜paramter name=\"tz\" string=\"true\">CET</｜DSML｜parameter>\n\
             {WEATHER}</｜DSML｜invoke>\n"
        ));
        // Misspelled parameter open, alone ({} args) and after a good one.
        assert_malformed(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜paramter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n",
        );
        assert_malformed(&format!(
            "<｜DSML｜invoke name=\"get_weather\">\n{WEATHER}\
             <｜DSML｜paramter name=\"tz\" string=\"true\">CET</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n"
        ));
        // Misspelled string attr; empty parameter name; empty block.
        assert_malformed(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜parameter name=\"city\" strng=\"true\">Paris</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n",
        );
        assert_malformed(
            "<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜parameter name=\"\" string=\"true\">Paris</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n",
        );
        assert_malformed("");
    }

    #[test]
    fn parses_mixed_invoke_and_tool_in_one_block() {
        // Defensive: if the model ever mixes the two tags in one block
        // (e.g. some training drift), both should still be parsed.
        let body = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"first\">\n\
<｜DSML｜parameter name=\"a\" string=\"false\">1</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
<｜DSML｜tool name=\"second\">\n\
<｜DSML｜parameter name=\"b\" string=\"true\">x</｜DSML｜parameter>\n\
</｜DSML｜tool>\n\
</｜DSML｜tool_calls>";
        let mut p = StreamParser::new();
        let mut events = p.feed(body);
        events.extend(p.finish());
        let calls: Vec<&Vec<ToolCall>> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCalls(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[0][0].name, "first");
        assert_eq!(calls[0][0].arguments["a"], json!(1));
        assert_eq!(calls[0][1].name, "second");
        assert_eq!(calls[0][1].arguments["b"], json!("x"));
    }
}
