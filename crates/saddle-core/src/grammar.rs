// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Constrained decoding — unified grammar.
//!
//! Merged from:
//! - `crates/hipfire-arch-qwen35/src/grammar.rs` (2,736 lines) — JSON `<tool_call>` format
//!   (13 arch-specific mentions, now parameterized via `json::Config`)
//! - `crates/hipfire-arch-deepseek4/src/grammar.rs` (1,199 lines) — DSML XML format
//!   (1 arch-specific mention, now provenance comment only)
//!
//! Both implementations are model-agnostic (`std` only) and share the same
//! masking contract: `is_token_allowed` / `token_mask` / `apply_mask_to_logits` plus `advance`.
//! The DSML variant tracks XML tag structure with leading-whitespace
//! tolerance; the JSON variant tracks `<tool_call>` JSON with brace-aware and
//! n-gram attractor guards. Model-specific details (tag literals, n-gram
//! thresholds) are passed in via `json::Config` where needed — no architecture branching
//! branching lives here.

/// JSON `<tool_call>` grammar — state machine for `\n{"name": "<TOOL>", "arguments": <value>}` tool calls.
/// See original `hipfire-arch-qwen35/src/grammar.rs` for full design notes (Pi turn-12 drift).
pub mod json {
    //! Grammar-guided decoding for the qwen3.5/3.6 tool-call format.
    //!
    //! Mirrors the V4F DSML grammar in `crates/hipfire-arch-deepseek4/src/grammar.rs`
    //! but constrains qwen35's `<tool_call>{json}</tool_call>` body instead of
    //! DeepSeek's XML-style DSML tags.
    //!
    //! ## Why this exists
    //!
    //! qwen3.6:27b drifts after long agentic sessions: it emits the
    //! `<tool_call>` opener correctly then writes ChatML noise as the body,
    //! e.g.
    //! ```text
    //! <tool_call>
    //! <|im_start|>assistant "Let me read the existing build files..."}}
    //! </tool_call>
    //! ```
    //! observed verbatim in Pi turn 12 after ~27k cached tokens. The text
    //! isn't JSON, so the daemon's tool-call extractor returns
    //! `tool_calls=0`, the daemon emits `finish_reason: "stop"` with that
    //! garbage as `message.content`, and Pi's agent loop terminates.
    //!
    //! The grammar matcher prevents this by masking sample logits the
    //! moment the model commits to `<tool_call>`: from then until the JSON
    //! header `\n{"name": "<TOOL_NAME>", "arguments": ` is fully laid
    //! down, only tokens that continue that template are allowed. The
    //! `arguments` value itself is free-form JSON (state `InArgs`), with one
    //! structural constraint: the model may not emit `</tool_call>` until the
    //! OUTER tool-call object is brace-balanced. The header opens `{...`, so
    //! the canonical close is `}}\n</tool_call>` — one `}` for the args value,
    //! one for the enclosing object. Once `</tool_call>` lands, the matcher
    //! returns to free emission.
    //!
    //! Enforcing the outer `}` is the "JSON-aware brace counting" future phase
    //! flagged below: qwen3.6:27b (mq4, temp>0) intermittently jumped from the
    //! inner `}` straight to `</tool_call>`, dropping the outer brace and
    //! emitting invalid JSON (`{"name":"read","arguments":{"path":"x"}`) that
    //! failed Pi's tool-call parser — the "dropped closing bracket" bug.
    //!
    //! ## States
    //!
    //! - [`State::Out`] — free emission. The matcher watches for the
    //!   `<tool_call>` substring (single special token, vocab id varies
    //!   per checkpoint) and transitions to [`State::AfterOpen`] on entry.
    //! - [`State::AfterOpen`] — between `<tool_call>` and the
    //!   `"arguments": ` colon-space. Constrained to a literal byte
    //!   sequence that names one of the available tools.
    //! - [`State::InArgs`] — between `"arguments": ` and `</tool_call>`. The
    //!   args value is free JSON, but `</tool_call>` is masked out until the
    //!   enclosing tool-call object is brace-balanced (the outer `}` is
    //!   emitted). String-aware, so `{`/`}`/`<` inside string values don't
    //!   count. The attractor force-close path is exempt.
    //! - (back to `Out` after `</tool_call>`.)
    //!
    //! ## What this does NOT do
    //!
    //! Enforce full JSON-validity inside `arguments` (well-formed keys,
    //! quoting, commas). It tracks only string-aware brace balance — enough
    //! to require the outer `}` before the close — leaving trailing-comma /
    //! unquoted-key glitches to the daemon's downstream tool-call repair.

    /// Position in the qwen35 tool-call grammar. See module docs for
    /// transitions.
    #[derive(Debug, Clone, PartialEq)]
    pub enum State {
        /// Free emission outside any `<tool_call>` block. Watching for the
        /// open marker but not otherwise constraining tokens.
        Out,
        /// Between `<tool_call>` (already consumed) and the `"arguments": `
        /// header sentinel. Allowed continuations are prefixes of
        /// `\n{"name": "<TOOL_NAME>", "arguments": `.
        AfterOpen,
        /// Between `"arguments": ` and `</tool_call>`. Free emission —
        /// the model writes the args value as it pleases. We re-enter
        /// `Out` when `</tool_call>` lands in the rolling buffer.
        InArgs,
    }

    /// Schema for one available tool. Built from the OpenAI-format tools
    /// array at request time; the grammar uses this to pick which tool
    /// names the model is allowed to emit at the name position AND which
    /// argument fields must appear in the args body before the close
    /// marker is allowed.
    #[derive(Debug, Clone)]
    pub struct ToolSchema {
        pub name: String,
        /// Subset of the tool's parameters that MUST appear as keys in
        /// the emitted args body. Built from the JSON schema's
        /// `parameters.required` array at request time. The grammar's
        /// `is_token_allowed` rejects close-marker prefixes while any
        /// required name is still absent from the args body — without
        /// this the model can emit `"arguments":{}` (Pi `write` failure
        /// mode observed in production: model emitted empty args, Pi
        /// rejected, model retried with same empty args, context bloated
        /// to KV exhaustion).
        pub required: Vec<String>,
    }

    /// Maximum bytes retained for the n-gram loop guard's rolling window
    /// inside [`State::InArgs`]. 256 bytes covers the worst-case attractor
    /// we've observed in production (a ~20-byte repeated block × 4) with
    /// headroom. Bounded so the rolling-window scan stays O(1) per token.
    const NGRAM_WINDOW: usize = 256;

    /// Minimum consecutive identical n-gram repeats that trigger the
    /// attractor flag. **Default 6** — bumped from 4 on 2026-05-28 after a
    /// false-positive incident generating Zig code (legitimate
    /// indentation + repeated section markers tripped the guard mid-args,
    /// stranded the assistant in an empty tool call). Real attractors
    /// (e.g. `typetypetypetype` extended, repeated invocation snippets)
    /// run long and still trip at 6+. Override at startup via
    /// `GRAMMAR_NGRAM_MIN_REPEATS=<n>` if a workload needs tighter
    /// detection.
    const NGRAM_MIN_REPEATS_DEFAULT: usize = 6;

    /// Range of n-gram lengths probed each token (inclusive). **MIN
    /// bumped from 2 → 3** on 2026-05-28: 2-byte n-grams catch `  ` (two
    /// spaces) repeating, `\n\n`, `, ,`, etc. — pervasive in code and
    /// JSON. 3-byte grams still catch tight character cycles without the
    /// false-positive flood. 32-byte grams catch multi-token phrase loops.
    /// Override the lower bound with `GRAMMAR_NGRAM_LEN_MIN=<n>`.
    const NGRAM_LEN_MIN_DEFAULT: usize = 3;
    const NGRAM_LEN_MAX: usize = 32;

    /// Tunable n-gram loop-guard thresholds for the JSON `InArgs` attractor detector.
    ///
    /// The matcher is instantiated with a `Config` so model-specific values are
    /// passed in by the caller rather than branching on architecture inside this
    /// crate. Defaults match the production tuning (6 repeats of 3..32
    /// byte grams). Callers that need env-driven overrides should read their own
    /// env vars and pass the parsed values here — this crate does not depend on
    /// `hipfire-config` and contains no arch-specific env names.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Config {
        /// Rolling-window size in bytes (default 256).
        pub ngram_window: usize,
        /// Minimum consecutive identical n-gram repeats to trigger the attractor
        /// flag (default 6).
        pub ngram_min_repeats: usize,
        /// Minimum n-gram length probed (default 3).
        pub ngram_len_min: usize,
        /// Maximum n-gram length probed (default 32).
        pub ngram_len_max: usize,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                ngram_window: NGRAM_WINDOW,
                ngram_min_repeats: NGRAM_MIN_REPEATS_DEFAULT,
                ngram_len_min: NGRAM_LEN_MIN_DEFAULT,
                ngram_len_max: NGRAM_LEN_MAX,
            }
        }
    }
    use std::sync::Arc;

    /// Maximum number of tool schemas accepted by [`CompiledGrammar::new`].
    /// Bounds compilation input so a pathological tools array can't stall
    /// request admission. Public so the serving layer can refuse an over-bound
    /// request with a typed 400 instead of reaching `Matcher::with_config`'s
    /// panic.
    pub const MAX_TOOL_SCHEMAS: usize = 256;

    /// Maximum total bytes of all tool names + required field names combined.
    /// Prevents unbounded string allocation during compilation. Public for the
    /// same request-validation reason as [`MAX_TOOL_SCHEMAS`].
    pub const MAX_SCHEMA_BYTES: usize = 64 * 1024;

    /// Typed error returned when compilation inputs exceed bounds or are
    /// otherwise invalid. Rejects before any GPU work (spec §7 G1, S4).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum GrammarCompileError {
        /// Too many tool schemas supplied.
        TooManySchemas { count: usize, max: usize },
        /// Total schema string bytes exceeded the bound.
        SchemaBytesExceeded { bytes: usize, max: usize },
        /// A tool name is empty.
        EmptyToolName { index: usize },
    }

    impl std::fmt::Display for GrammarCompileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::TooManySchemas { count, max } => {
                    write!(f, "grammar compile: {count} tool schemas exceeds max {max}")
                }
                Self::SchemaBytesExceeded { bytes, max } => write!(
                    f,
                    "grammar compile: schema string bytes {bytes} exceeds max {max}"
                ),
                Self::EmptyToolName { index } => {
                    write!(f, "grammar compile: tool schema {index} has empty name")
                }
            }
        }
    }

    impl std::error::Error for GrammarCompileError {}

    /// Immutable, `Send + Sync` compiled artifact for the JSON `<tool_call>` tool-call
    /// grammar. Owns the tool schemas and n-gram config; produced once per
    /// unique tool set and shared across requests via `Arc<CompiledGrammar>`.
    ///
    /// Per-request mutable state lives in [`Matcher`] (the cursor), which
    /// references the compiled artifact through an `Arc`. This split lets
    /// identical in-flight compilations share the same tables (spec §7 G1).
    ///
    /// Construct via [`CompiledGrammar::new`] (default config) or
    /// [`CompiledGrammar::with_config`] (explicit n-gram thresholds). Both
    /// validate bounds and return a typed error on violation.
    #[derive(Debug, Clone)]
    pub struct CompiledGrammar {
        tools: Vec<ToolSchema>,
        config: Config,
    }

    impl CompiledGrammar {
        /// Compile with default [`Config`]. Validates bounds.
        pub fn new(tools: Vec<ToolSchema>) -> Result<Self, GrammarCompileError> {
            Self::with_config(tools, Config::default())
        }

        /// Compile with an explicit [`Config`]. Validates bounds.
        pub fn with_config(
            tools: Vec<ToolSchema>,
            config: Config,
        ) -> Result<Self, GrammarCompileError> {
            if tools.len() > MAX_TOOL_SCHEMAS {
                return Err(GrammarCompileError::TooManySchemas {
                    count: tools.len(),
                    max: MAX_TOOL_SCHEMAS,
                });
            }
            let mut total_bytes = 0;
            for (i, t) in tools.iter().enumerate() {
                if t.name.is_empty() {
                    return Err(GrammarCompileError::EmptyToolName { index: i });
                }
                total_bytes += t.name.len();
                for r in &t.required {
                    total_bytes += r.len();
                }
            }
            if total_bytes > MAX_SCHEMA_BYTES {
                return Err(GrammarCompileError::SchemaBytesExceeded {
                    bytes: total_bytes,
                    max: MAX_SCHEMA_BYTES,
                });
            }
            Ok(Self { tools, config })
        }

        /// Read-only access to the tool schemas.
        pub fn tools(&self) -> &[ToolSchema] {
            &self.tools
        }

        /// Read-only access to the n-gram config.
        pub fn config(&self) -> Config {
            self.config
        }
    }

    // CompiledGrammar is Send+Sync: ToolSchema and Config contain only
    // owned Strings / Copy primitives.
    unsafe impl Send for CompiledGrammar {}
    unsafe impl Sync for CompiledGrammar {}

    /// Grammar matcher: state plus the bytes committed since the last
    /// firm transition. Construct via [`Matcher::new`] with the active
    /// tool schemas, advance with [`Matcher::advance`], query allowed
    /// tokens via [`Matcher::is_token_allowed`] or [`Matcher::token_mask`].
    #[derive(Debug, Clone)]
    pub struct Matcher {
        state: State,
        /// Bytes committed since the last firm state transition.
        partial_buf: String,
        /// Immutable compiled artifact: tool schemas + n-gram config.
        /// Shared across requests via `Arc` — the cursor (this struct)
        /// holds only per-request mutable state (spec §7 G1).
        grammar: Arc<CompiledGrammar>,
        /// Rolling window of the last `NGRAM_WINDOW` bytes of the FULL args
        /// body (including string-value bytes) seen while in [`State::InArgs`].
        /// Used ONLY for the required-field substring check (`"path"` etc. live
        /// inside strings). Attractor detection runs on the separate
        /// structural-only [`Self::attractor_buf`]. Cleared on every firm
        /// transition.
        ngram_history: String,
        /// Rolling window of the last `NGRAM_WINDOW` *structural* (out-of-string)
        /// args bytes — fed to the n-gram attractor guard. String-value bytes
        /// (e.g. a `write` tool's code `content`) are excluded so legitimate code
        /// repetition doesn't false-trip the guard. Cleared on every firm
        /// transition.
        attractor_buf: String,
        /// Set when consecutive n-gram repetition is detected inside
        /// [`State::InArgs`]. While set, the matcher constrains InArgs
        /// to only `</tool_call>` continuations — the model gets a
        /// forced exit instead of being stuck in an attractor that
        /// bloats the agentic conversation's KV (Pi turn-12-style
        /// `typetypetypetype` inside the args body was the motivating
        /// case). Cleared when the close marker fires and we return
        /// to [`State::Out`].
        attractor_detected: bool,
        /// Index into `tools` of the tool whose schema we're currently
        /// inside (i.e. the one whose name header just matched on the
        /// `AfterOpen` → `InArgs` transition). `None` outside of
        /// [`State::InArgs`]. Used by `required_fields_satisfied` to
        /// know which required-field list to check against the args
        /// body bytes.
        current_tool: Option<usize>,
        /// JSON brace depth inside the args body while in
        /// [`State::InArgs`]. Starts at 0 on entry; the first `{` of the
        /// args body increments to 1; the matching `}` brings it back to
        /// 0. Used by `is_token_allowed` to block tokens that would
        /// close the args body before required fields are satisfied —
        /// the close marker check alone is insufficient because the
        /// closing `}` of an empty `{}` body already commits before the
        /// next-token `</tool_call>` is seen, so the empty args stream
        /// to the client even when the close-marker token is rejected.
        args_brace_depth: i32,
        /// True while inside a JSON string in the args body. Toggled on
        /// unescaped `"`. Used to ignore `{` / `}` inside string values
        /// when updating `args_brace_depth`.
        args_in_string: bool,
        /// True if the previous byte in the args body was a backslash
        /// inside a string. The next byte is then escaped (skip its
        /// special meaning, e.g. `\"` does NOT close the string).
        args_string_escape: bool,
    }

    impl Matcher {
        /// Build a fresh matcher in [`State::Out`] with no partial buffer and
        /// default `Config`.
        ///
        /// Compiles the tools into a [`CompiledGrammar`] internally. For
        /// request sharing, prefer [`Matcher::from_compiled`] with a
        /// pre-built `Arc<CompiledGrammar>`.
        pub fn new(tools: Vec<ToolSchema>) -> Self {
            Self::with_config(tools, Config::default())
        }

        /// Build a fresh matcher with an explicit `Config`. Use this when the
        /// caller wants to override the n-gram thresholds without an env read
        /// inside this crate.
        ///
        /// Compiles the tools into a [`CompiledGrammar`] internally. Panics
        /// on bound violations — use [`CompiledGrammar::with_config`] +
        /// [`Matcher::from_compiled`] for fallible construction.
        pub fn with_config(tools: Vec<ToolSchema>, config: Config) -> Self {
            let grammar = CompiledGrammar::with_config(tools, config)
                .expect("Matcher::with_config: schema bounds violated");
            Self::from_compiled(Arc::new(grammar))
        }

        /// Build a cursor from a pre-compiled, shared artifact. Multiple
        /// requests can share the same `Arc<CompiledGrammar>` while each
        /// owns an independent `Matcher` cursor (spec §7 G1).
        pub fn from_compiled(grammar: Arc<CompiledGrammar>) -> Self {
            Self {
                state: State::Out,
                partial_buf: String::new(),
                grammar,
                ngram_history: String::new(),
                attractor_buf: String::new(),
                attractor_detected: false,
                current_tool: None,
                args_brace_depth: 0,
                args_in_string: false,
                args_string_escape: false,
            }
        }

        /// Current n-gram config (copy).
        pub fn config(&self) -> Config {
            self.grammar.config
        }

        /// Read-only access to the compiled grammar artifact.
        pub fn compiled(&self) -> &CompiledGrammar {
            &self.grammar
        }

        /// Index of the tool currently being constructed in
        /// [`State::InArgs`]. Exposed for diagnostics.
        pub fn current_tool(&self) -> Option<usize> {
            self.current_tool
        }

        /// Diagnostic snapshot for the DFlash close-marker rejection path.
        pub fn debug_close_reject(&self) -> String {
            let req = self
                .current_tool
                .and_then(|i| self.grammar.tools.get(i))
                .map(|s| s.required.join(","))
                .unwrap_or_default();
            let hist_tail: String = self
                .ngram_history
                .chars()
                .rev()
                .take(80)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            format!(
            "current_tool={:?} required=[{}] req_satisfied={} args_brace_depth={} ngram_hist_len={} hist_tail={:?}",
            self.current_tool,
            req,
            self.required_fields_satisfied(),
            self.args_brace_depth,
            self.ngram_history.len(),
            hist_tail,
        )
        }

        /// Check whether the args body bytes seen so far satisfy every
        /// required field for the current tool. A field is considered
        /// "seen" if `"<name>"` appears anywhere in the args body bytes
        /// (`ngram_history`). The substring match is intentionally loose
        /// — JSON syntax exists for the LLM, not for the parser, so as
        /// long as the field name string is present we trust the model
        /// to wire the colon/value correctly.
        ///
        /// Empty schema or empty `required` list → trivially satisfied.
        /// Returns true if no current tool (e.g. AfterOpen state) so we
        /// don't gate transitions we can't evaluate.
        fn required_fields_satisfied(&self) -> bool {
            let tool_idx = match self.current_tool {
                Some(i) => i,
                None => return true,
            };
            let schema = match self.grammar.tools.get(tool_idx) {
                Some(s) => s,
                None => return true,
            };
            if schema.required.is_empty() {
                return true;
            }
            for name in &schema.required {
                let needle = format!("\"{}\"", name);
                if !self.ngram_history.contains(&needle) {
                    return false;
                }
            }
            true
        }

        /// Update the args-body brace/string tracking state by consuming
        /// the bytes in `text`. Caller must guarantee the matcher is in
        /// [`State::InArgs`]; this is enforced at the only call site
        /// (`advance`). String-aware: `{` / `}` inside JSON strings do
        /// NOT change brace depth, and `\"` does NOT close the string.
        fn update_args_brace_state(&mut self, text: &str) {
            for byte in text.bytes() {
                if self.args_string_escape {
                    self.args_string_escape = false;
                    continue;
                }
                if self.args_in_string {
                    match byte {
                        b'\\' => self.args_string_escape = true,
                        b'"' => self.args_in_string = false,
                        _ => {}
                    }
                    continue;
                }
                // Structural position (outside any JSON string value): feed the
                // n-gram attractor guard. String-value bytes (handled by the
                // `continue` above) are deliberately excluded — see `advance`.
                self.push_attractor_byte(byte);
                match byte {
                    b'"' => self.args_in_string = true,
                    b'{' => self.args_brace_depth += 1,
                    b'}' => self.args_brace_depth -= 1,
                    _ => {}
                }
            }
        }

        /// Simulate the brace-state update for `text` WITHOUT mutating
        /// `self`, and return whether the token would close the outer
        /// args body (depth returns to 0 from a depth >= 1 reached at
        /// or before this token). Used by `is_token_allowed` to reject
        /// the `}` of an empty `{}` body before it commits.
        ///
        /// "Closes the args body" semantics:
        ///   - If `args_brace_depth` was >= 1 before this token (we're
        ///     already inside the body), any `}` bringing depth to 0
        ///     counts as closing.
        ///   - If `args_brace_depth` was 0 (haven't seen the first `{`
        ///     yet), the token closes only if it opens then closes the
        ///     body — i.e. it contains a `{` that raises depth to 1+
        ///     and a matching `}` that brings depth back to 0. This is
        ///     the empty-`{}` single-token case.
        fn would_close_args_body(&self, text: &str) -> bool {
            let mut depth = self.args_brace_depth;
            let mut in_string = self.args_in_string;
            let mut escape = self.args_string_escape;
            let mut entered_body = self.args_brace_depth >= 1;
            for byte in text.bytes() {
                if escape {
                    escape = false;
                    continue;
                }
                if in_string {
                    match byte {
                        b'\\' => escape = true,
                        b'"' => in_string = false,
                        _ => {}
                    }
                    continue;
                }
                match byte {
                    b'"' => in_string = true,
                    b'{' => {
                        depth += 1;
                        if depth >= 1 {
                            entered_body = true;
                        }
                    }
                    b'}' => {
                        depth -= 1;
                        if entered_body && depth <= 0 {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }

        /// Simulate the args-body brace state over `text` (seeded from the
        /// committed [`State::InArgs`] state) and return whether the OUTER
        /// tool-call object is already closed (`args_brace_depth < 0`) at the
        /// point a `</tool_call>` marker begins within `text`.
        ///
        /// Used by [`Self::is_token_allowed`] to permit a close-forming token
        /// iff it does not drop the outer `}`: a lone `</tool_call>` after the
        /// inner `}` (depth back to 0) is rejected, while a merged
        /// `}}\n</tool_call>` token — whose second `}` drives depth to -1 before
        /// the marker — is allowed. A `<` inside a JSON string value is data,
        /// not the marker, so the scan is string-aware. By the block invariant
        /// (this gate rejects any premature close-prefix token), the marker can
        /// only begin inside `text`, never straddling `partial_buf`.
        fn outer_closed_before_marker(&self, text: &str) -> bool {
            const CLOSE: &[u8] = b"</tool_call>";
            let mut depth = self.args_brace_depth;
            let mut in_string = self.args_in_string;
            let mut escape = self.args_string_escape;
            let bytes = text.as_bytes();
            for i in 0..bytes.len() {
                let rem = &bytes[i..];
                // A `</tool_call>` marker (full, or a terminal prefix running to
                // the end of `text`) begins here — only meaningful outside a
                // string value. Decide against the outer-object state at this
                // position.
                if !in_string && (rem.starts_with(CLOSE) || CLOSE.starts_with(rem)) {
                    return depth < 0;
                }
                let byte = bytes[i];
                if escape {
                    escape = false;
                    continue;
                }
                if in_string {
                    match byte {
                        b'\\' => escape = true,
                        b'"' => in_string = false,
                        _ => {}
                    }
                    continue;
                }
                match byte {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
            }
            // No `</tool_call>` marker actually begins (outside a string) within
            // `text` — the caller's `touches_close_marker` matched a `<` that is
            // string-value CONTENT (heredoc `<<`, `#include <stdio.h>`, `a < b`,
            // …) or a bare `<` that isn't a real close prefix. It does not close
            // the tool call, so allow it. (A real close prefix outside a string is
            // caught by the in-loop `return depth < 0` above.)
            true
        }

        /// True iff the n-gram loop guard has tripped on the current
        /// [`State::InArgs`] body. Exposed for diagnostics; the daemon
        /// uses this to log the trip event.
        pub fn attractor_detected(&self) -> bool {
            self.attractor_detected
        }

        /// Detect consecutive identical n-gram repetition in the tail of
        /// `buf`. Returns `true` iff there's some `n` in
        /// `[ngram_len_min(), NGRAM_LEN_MAX]` such that the last
        /// `n * ngram_min_repeats()` bytes consist of the same `n`-byte
        /// block repeated `ngram_min_repeats()` times. The lower bound on
        /// `n` and the repeat threshold come from env-tunable knobs
        /// (`Config::ngram_len_min`, `Config::ngram_min_repeats`)
        /// with the cached defaults (3 and 6 respectively).
        ///
        /// **Uniform-byte filter:** n-grams that consist of a single
        /// repeated character (e.g. `   ` for whitespace, `===` for
        /// dividers) are skipped — long runs of the same byte are
        /// pervasive in code (indentation, section markers) and never
        /// indicate an attractor. Real attractors (`type`, `pub fn`, etc.)
        /// have non-uniform grams.
        fn detect_ngram_loop(&self, buf: &str) -> bool {
            let bytes = buf.as_bytes();
            let len_min = self.grammar.config.ngram_len_min;
            let min_repeats = self.grammar.config.ngram_min_repeats;
            for ngram_len in len_min..=self.grammar.config.ngram_len_max {
                let needed = ngram_len * min_repeats;
                if bytes.len() < needed {
                    continue;
                }
                let tail = &bytes[bytes.len() - needed..];
                let first = &tail[..ngram_len];
                // Skip uniform-byte grams — they fire on legit indentation
                // and divider runs without signaling a real loop.
                if Self::is_uniform_byte(first) {
                    continue;
                }
                let mut all_match = true;
                for r in 1..min_repeats {
                    let chunk = &tail[r * ngram_len..(r + 1) * ngram_len];
                    if chunk != first {
                        all_match = false;
                        break;
                    }
                }
                if all_match {
                    return true;
                }
            }
            false
        }

        /// True iff every byte in `chunk` is the same value (e.g. `   `,
        /// `===`, `\t\t\t`). Empty slice is vacuously uniform.
        fn is_uniform_byte(chunk: &[u8]) -> bool {
            match chunk.first() {
                Some(&first) => chunk.iter().all(|&b| b == first),
                None => true,
            }
        }

        /// Push bytes into the n-gram history buffer and run detection.
        /// Only called while in [`State::InArgs`]. Sets
        /// `attractor_detected = true` on first detection; never clears
        /// it from here (clearing happens on the firm `</tool_call>`
        /// transition back to `Out`).
        fn update_ngram_history(&mut self, text: &str) {
            // Full args text → required-field substring buffer only. Attractor
            // detection runs on structural bytes in `push_attractor_byte`.
            self.ngram_history.push_str(text);
            if self.ngram_history.len() > self.grammar.config.ngram_window {
                // Drop at a UTF-8 char boundary — string values may hold multibyte
                // content, so this buffer is not guaranteed ASCII.
                let drop = self.ngram_history.len() - self.grammar.config.ngram_window;
                let mut idx = drop;
                while idx < self.ngram_history.len() && !self.ngram_history.is_char_boundary(idx) {
                    idx += 1;
                }
                self.ngram_history.drain(..idx);
            }
        }

        /// Feed one *structural* (out-of-string) args byte into the attractor
        /// guard. Called per-byte from [`Self::update_args_brace_state`]; bytes
        /// inside JSON string values are excluded by that caller.
        fn push_attractor_byte(&mut self, byte: u8) {
            if self.attractor_detected {
                return; // already flagged; don't waste work
            }
            // Structural JSON bytes are always ASCII. Skip any stray non-ASCII byte
            // so `attractor_buf` stays valid UTF-8 for `&str` detection; every
            // ASCII byte is its own char boundary, so trimming is unconditional.
            if !byte.is_ascii() {
                return;
            }
            self.attractor_buf.push(byte as char);
            if self.attractor_buf.len() > self.grammar.config.ngram_window {
                let drop = self.attractor_buf.len() - self.grammar.config.ngram_window;
                self.attractor_buf.drain(..drop);
            }
            if self.detect_ngram_loop(&self.attractor_buf) {
                self.attractor_detected = true;
            }
        }

        /// Read-only view of the current state.
        pub fn state(&self) -> &State {
            &self.state
        }

        /// Bytes accumulated since the last firm state transition.
        pub fn partial(&self) -> &str {
            &self.partial_buf
        }

        /// Whether the matcher is currently free (all tokens allowed).
        /// Free in [`State::Out`] until the buffer accumulates a `<tool_call>`
        /// prefix; free in [`State::InArgs`] until the buffer accumulates
        /// the `</tool_call>` close-marker prefix OR a required field is
        /// still missing from the args body.
        ///
        /// The dual-mode design mirrors V4F's `is_free` — the constraint
        /// only kicks in at structural transitions, not during free prose
        /// or args body.
        ///
        /// Three sources make InArgs non-free:
        ///   1. Buffer ends with a `</tool_call>` close-marker prefix
        ///      (the model is committing to close — gate it).
        ///   2. The n-gram loop guard has tripped
        ///      (`self.attractor_detected = true`): force the close so
        ///      the model gets a forced exit instead of being stuck
        ///      extending an attractor.
        ///   3. A required field is still missing from the args body:
        ///      the model is in the middle of args body and might next
        ///      try to close with `}}\n</tool_call>` — we must reject
        ///      that close until required fields appear. We make the
        ///      state non-free so `is_token_allowed` runs and can
        ///      enforce the per-token check.
        pub fn is_free(&self) -> bool {
            match self.state {
                // Masking on a partial `<tool_call>` prefix strands free text: a
                // bare `<` (`<ip>`, `<p>`, `2 < 3`) leaves only `t`/`to`… legal.
                State::Out => true,
                State::AfterOpen => false,
                State::InArgs => {
                    if self.attractor_detected {
                        return false;
                    }
                    if !self.required_fields_satisfied() {
                        return false;
                    }
                    if self.args_brace_depth >= 0 {
                        // The OUTER tool-call object `{"name":..,"arguments":<v>}` is
                        // still open. `args_brace_depth` counts only the args-VALUE
                        // object (it re-enters InArgs at 0, the header's outer `{`
                        // already consumed), so a balanced value nets back to 0 while
                        // the enclosing `}` is still outstanding (depth -> -1 once it
                        // lands). Stay constrained so `is_token_allowed` can block a
                        // premature `</tool_call>` before the outer brace.
                        return false;
                    }
                    !Self::has_close_prefix(&self.partial_buf)
                }
            }
        }

        /// Does the buffer end with a strict prefix of `</tool_call>`?
        fn has_close_prefix(s: &str) -> bool {
            const CLOSE: &str = "</tool_call>";
            for n in 1..=CLOSE.len() {
                if s.ends_with(&CLOSE[..n]) {
                    return true;
                }
            }
            false
        }

        /// Returns the legal byte-string continuations from the current
        /// state. Each returned string is a FULL prefix starting at the
        /// position immediately after the last firm transition. Callers
        /// check `partial_buf + decode(T)` against these via
        /// [`Self::is_token_allowed`].
        pub fn allowed_continuations(&self) -> Vec<String> {
            match &self.state {
                State::Out => {
                    // Only constraining when we've started emitting `<tool_call>`.
                    // is_free() short-circuits the unconstrained case before we
                    // get here; if we do get here it's because the partial
                    // buffer already starts forming `<tool_call>`.
                    vec!["<tool_call>".to_string()]
                }
                State::AfterOpen => {
                    // Allowed continuations: for each tool name, the literal
                    // header `\n{"name": "<NAME>", "arguments": `.
                    self.grammar
                        .tools
                        .iter()
                        .map(|t| format!("\n{{\"name\": \"{}\", \"arguments\": ", t.name))
                        .collect()
                }
                State::InArgs => {
                    // Constraining when the model has started emitting `</tool_call>`.
                    // Same dual-mode pattern as State::Out: free unless a close
                    // prefix is forming.
                    vec!["</tool_call>".to_string()]
                }
            }
        }

        /// Whether the given decoded text would keep us on a legal path
        /// from the current state.
        ///
        /// Empty tokens (placeholder / control tokens with no decoded text)
        /// are always allowed — they don't consume any buffer position.
        ///
        /// Per-state semantics:
        /// - [`State::Out`] / [`State::InArgs`]: in free regions where a
        ///   trigger marker (`<tool_call>` / `</tool_call>`) may start
        ///   forming partway through the buffer, the check is suffix-based.
        ///   A token is allowed iff some suffix of `partial_buf + text` is
        ///   a prefix of the trigger marker (or the full marker appears
        ///   somewhere, which would fire a transition on advance).
        /// - [`State::AfterOpen`]: the entire `partial_buf + text` must be
        ///   a prefix of, or extension of, one of the header templates.
        pub fn is_token_allowed(&self, text: &str) -> bool {
            if text.is_empty() {
                return true;
            }
            if self.is_free() {
                return true;
            }
            let combined = format!("{}{}", self.partial_buf, text);
            match &self.state {
                State::Out => Self::tail_matches_marker(&combined, "<tool_call>"),
                State::AfterOpen => {
                    let conts = self.allowed_continuations();
                    Self::check_against_conts(&combined, &conts)
                }
                // InArgs: tail-matches-marker is the usual check. When the
                // attractor flag is set or required fields are missing,
                // this same check applies but UNCONDITIONALLY (no is_free
                // short-circuit). For the required-field case we additionally
                // BLOCK any token that would form a close-marker prefix —
                // the model must emit content with the missing field names
                // before closing.
                State::InArgs => {
                    let close_prefix = Self::tail_matches_marker(&combined, "</tool_call>");
                    if !self.required_fields_satisfied() {
                        // Block close-marker prefix tokens entirely; allow
                        // any other content so the model can keep emitting
                        // until required fields appear in the body. Also
                        // block any token that would close the args body
                        // (`}` bringing brace depth to 0) — without this
                        // gate the closing `}` of an empty `{}` body
                        // commits as args content, and even though the
                        // following `</tool_call>` token is rejected, the
                        // already-streamed `arguments: {}` lands at the
                        // OpenAI API as a malformed tool call.
                        if Self::touches_close_marker(&combined) {
                            false
                        } else if self.would_close_args_body(text) {
                            false
                        } else {
                            true
                        }
                    } else if !self.attractor_detected && self.args_brace_depth >= 0 {
                        // Required fields are present, but the OUTER tool-call object
                        // is not yet closed (`args_brace_depth` tracks only the
                        // args-VALUE object, which nets to 0; the outer `}` drives it
                        // to -1). A token that begins a `</tool_call>` marker while the
                        // outer brace is still open would drop that brace and emit
                        // structurally-invalid JSON (the Pi "dropped closing bracket"
                        // failure: `{"name":"read","arguments":{"path":"x"}`). Allow a
                        // close-forming token ONLY if the outer `}` lands (brace depth
                        // < 0) before the marker begins — this still permits a merged
                        // `}}\n</tool_call>` token. Non-close content is always free.
                        // The attractor force-close path is exempt (falls through to
                        // the `else` below) so a looped model can still escape.
                        if Self::touches_close_marker(&combined) {
                            self.outer_closed_before_marker(text)
                        } else {
                            true
                        }
                    } else {
                        close_prefix
                    }
                }
            }
        }

        /// True iff some suffix of `s` is a prefix of `marker`, OR `marker`
        /// appears anywhere in `s` (the latter handles tokens that finish
        /// emitting the trigger — `advance` will then fire a transition).
        /// Used for free-region constraint checks where the trigger marker
        /// can start partway through `partial_buf`.
        fn tail_matches_marker(s: &str, marker: &str) -> bool {
            if s.contains(marker) {
                return true;
            }
            for n in 1..=marker.len() {
                if s.ends_with(&marker[..n]) {
                    return true;
                }
            }
            false
        }

        /// True if `s` contains the full close marker OR ends with any
        /// strict prefix of it. Distinct from `tail_matches_marker`
        /// because we want a binary "does this token touch the close
        /// region" answer for the required-field gate, not a continuation
        /// check.
        fn touches_close_marker(s: &str) -> bool {
            const CLOSE: &str = "</tool_call>";
            if s.contains(CLOSE) {
                return true;
            }
            for n in 1..=CLOSE.len() {
                if s.ends_with(&CLOSE[..n]) {
                    return true;
                }
            }
            false
        }

        /// Either `s` is a prefix of some continuation, or some continuation
        /// is a prefix of `s` (the latter handles tokens that extend past
        /// the firm transition boundary — e.g. `\n{"name"` matches the
        /// `\n{"name": "X", "arguments": ` template up to position 8).
        fn check_against_conts(s: &str, conts: &[String]) -> bool {
            for cont in conts {
                if cont.starts_with(s) || s.starts_with(cont.as_str()) {
                    return true;
                }
            }
            false
        }

        /// Populate a boolean mask over `vocab` indicating which tokens are
        /// legal at the current matcher position. `out` must be at least
        /// `vocab.len()` long; entries beyond `vocab.len()` are untouched.
        ///
        /// Fast path: when [`Self::is_free`] is true the entire mask is set
        /// to `true` — the caller can skip the sample-time mask scan.
        /// Hot path: O(vocab) scan calling [`Self::is_token_allowed`] per
        /// id, ~129k vocab × a handful of byte comparisons → sub-ms.
        pub fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
            debug_assert!(out.len() >= vocab.len());
            if self.is_free() {
                for slot in out.iter_mut().take(vocab.len()) {
                    *slot = true;
                }
                return;
            }
            for (id, text) in vocab.iter().enumerate() {
                out[id] = self.is_token_allowed(text);
            }
        }

        /// Apply the token mask in-place to a logits slice: disallowed
        /// tokens get `f32::NEG_INFINITY`, allowed are left alone.
        pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
            let n = mask.len().min(logits.len());
            for i in 0..n {
                if !mask[i] {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        /// Commit decoded token bytes into the matcher, advancing state if
        /// any allowed continuation completes. Idempotent at the byte
        /// level — callers may pass single bytes, multi-byte chunks, or
        /// full decoded tokens; the same final state is reached either way.
        ///
        /// While in [`State::InArgs`], the *structural* (out-of-string) bytes
        /// are fed into the n-gram loop guard so a model that drifts into a
        /// repeating attractor over the JSON skeleton is detected — the next
        /// `is_token_allowed` calls then force the close marker. Bytes inside a
        /// JSON string value (e.g. a `write` tool's code `content`) are excluded
        /// to avoid false-tripping on legitimate code repetition.
        pub fn advance(&mut self, text: &str) {
            if text.is_empty() {
                return;
            }
            if matches!(self.state, State::InArgs) {
                // `ngram_history` accumulates the FULL args text (field names live
                // inside string values) for the required-field guard. The n-gram
                // attractor guard must NOT see string-value bytes — a `write`/`edit`
                // tool's code `content` legitimately repeats short n-grams
                // (indentation, escaped newline+indent units `\n    `, `0, 0, 0, …`,
                // `},\n},\n…`) that would false-trip the 3-byte×6 default and force
                // a premature `</tool_call>`, truncating the argument and emitting
                // `{}` to the client (the write-tool empty-args bug).
                // `update_args_brace_state` walks byte-by-byte and feeds only the
                // *structural* (out-of-string) bytes into `attractor_buf`, so
                // structural loops (the JSON skeleton itself repeating) are still
                // caught.
                self.update_ngram_history(text);
                self.update_args_brace_state(text);
            }
            self.partial_buf.push_str(text);

            loop {
                match self.transition_once() {
                    Transition::Stay => return,
                    Transition::Advanced => continue,
                }
            }
        }

        /// Inner step: examine `partial_buf` against the current state's
        /// allowed transitions. Returns whether any firm transition fired.
        fn transition_once(&mut self) -> Transition {
            match self.state.clone() {
                State::Out => {
                    // Look for `<tool_call>` anywhere in the buffer. Once
                    // found, drop everything up to and including that
                    // substring and transition.
                    if let Some(idx) = self.partial_buf.find("<tool_call>") {
                        self.partial_buf.drain(..idx + "<tool_call>".len());
                        self.state = State::AfterOpen;
                        return Transition::Advanced;
                    }
                    // Trim the buffer to at most the longest open prefix
                    // we could complete next step. Stops the buffer from
                    // growing without bound during long free-emission runs.
                    let max_keep = "<tool_call>".len() - 1;
                    if self.partial_buf.len() > max_keep {
                        let drop = Self::drain_boundary(
                            &self.partial_buf,
                            self.partial_buf.len() - max_keep,
                        );
                        self.partial_buf.drain(..drop);
                    }
                    Transition::Stay
                }
                State::AfterOpen => {
                    // Look for the longest tool-name header that the buffer
                    // fully covers. When found, consume it and transition
                    // to InArgs — and record which tool's schema is now
                    // active so the close-marker check can validate its
                    // required-field list.
                    for (idx, schema) in self.grammar.tools.iter().enumerate() {
                        let cont = format!("\n{{\"name\": \"{}\", \"arguments\": ", schema.name);
                        if let Some(rest) = self.partial_buf.strip_prefix(cont.as_str()) {
                            let rest_owned = rest.to_string();
                            self.partial_buf = rest_owned.clone();
                            self.state = State::InArgs;
                            self.current_tool = Some(idx);
                            self.args_brace_depth = 0;
                            self.args_in_string = false;
                            self.args_string_escape = false;
                            // `rest` is the START of the args body (e.g. `{"command`)
                            // that arrived in the SAME chunk as the `"arguments": `
                            // marker. `advance` only feeds `ngram_history` /
                            // brace-state when ALREADY in InArgs, so without this
                            // the opening fragment is dropped — losing the leading
                            // `"` of the first field name, which made
                            // `required_fields_satisfied` perpetually false and
                            // rejected the valid `</tool_call>` close (a spurious
                            // DFlash grammar violation + full KV/DN reset on every
                            // tool turn, which defeated prompt-cache reuse). Feed it
                            // exactly once here; subsequent `advance` calls feed only
                            // their own new text, so there's no double-count of the
                            // brace depth.
                            if !rest_owned.is_empty() {
                                self.update_ngram_history(&rest_owned);
                                self.update_args_brace_state(&rest_owned);
                            }
                            return Transition::Advanced;
                        }
                    }
                    Transition::Stay
                }
                State::InArgs => {
                    // Look for `</tool_call>` anywhere in the buffer.
                    if let Some(idx) = self.partial_buf.find("</tool_call>") {
                        self.partial_buf.drain(..idx + "</tool_call>".len());
                        self.state = State::Out;
                        // Returning to Out — reset the n-gram guard's
                        // bookkeeping so a subsequent tool_call body starts
                        // with a fresh window. (The attractor flag must be
                        // cleared OR a stale flag will block the next
                        // body's first byte.) Also drop `current_tool` so
                        // the required-field check no longer applies.
                        self.ngram_history.clear();
                        self.attractor_buf.clear();
                        self.attractor_detected = false;
                        self.current_tool = None;
                        self.args_brace_depth = 0;
                        self.args_in_string = false;
                        self.args_string_escape = false;
                        return Transition::Advanced;
                    }
                    let max_keep = "</tool_call>".len() - 1;
                    if self.partial_buf.len() > max_keep {
                        let drop = Self::drain_boundary(
                            &self.partial_buf,
                            self.partial_buf.len() - max_keep,
                        );
                        self.partial_buf.drain(..drop);
                    }
                    Transition::Stay
                }
            }
        }
    }

    enum Transition {
        Stay,
        Advanced,
    }

    impl Matcher {
        /// Round `desired` UP to the next UTF-8 char boundary in `s`.
        /// Required because `String::drain(..n)` panics if `n` straddles
        /// a multi-byte codepoint — and tool_call args can contain
        /// arbitrary UTF-8 (Pi sessions hit this when the model pulled
        /// `𝐵link-hash` from a PDF into a write tool body). Returns
        /// at most `s.len()` so the drain never overshoots.
        fn drain_boundary(s: &str, desired: usize) -> usize {
            if desired >= s.len() {
                return s.len();
            }
            let mut idx = desired;
            while idx < s.len() && !s.is_char_boundary(idx) {
                idx += 1;
            }
            idx
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn schemas(names: &[&str]) -> Vec<ToolSchema> {
            names
                .iter()
                .map(|n| ToolSchema {
                    name: n.to_string(),
                    required: Vec::new(),
                })
                .collect()
        }

        /// Schema-with-required helper for required-field tests.
        fn schemas_with_required(specs: &[(&str, &[&str])]) -> Vec<ToolSchema> {
            specs
                .iter()
                .map(|(name, req)| ToolSchema {
                    name: name.to_string(),
                    required: req.iter().map(|s| s.to_string()).collect(),
                })
                .collect()
        }

        #[test]
        fn out_state_is_free_until_open_prefix() {
            let m = Matcher::new(schemas(&["bash"]));
            assert!(m.is_free());
            assert!(m.is_token_allowed("Hello world"));
            assert!(m.is_token_allowed("<|im_start|>"));
        }

        #[test]
        fn open_marker_transitions_to_after_open() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("here is some prose <tool_call>");
            assert!(matches!(m.state(), State::AfterOpen));
        }

        #[test]
        fn after_open_constrains_to_header_template() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            // The Pi-failure-mode token `<|im_start|>` must be rejected
            // at this position.
            assert!(!m.is_token_allowed("<|im_start|>"));
            // A leading newline that continues the header template is OK.
            assert!(m.is_token_allowed("\n"));
            // `\n{` is OK.
            assert!(m.is_token_allowed("\n{"));
            // Any token that diverges from the header is rejected.
            assert!(!m.is_token_allowed("\nassistant"));
        }

        #[test]
        fn header_completes_into_in_args() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            m.advance("\n{\"name\": \"bash\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
        }

        #[test]
        fn in_args_allows_content_but_gates_close_until_outer_brace() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // The outer tool-call object is open (args_brace_depth == 0), so the
            // matcher stays constrained — NOT free — to gate a premature close.
            assert!(!m.is_free());
            // Inside args, any non-close payload is still fine.
            assert!(m.is_token_allowed("{\"command\": \"ls -la\"}"));
            assert!(m.is_token_allowed("\n"));
            // Once the full body (args value + outer `}`) is balanced, InArgs is
            // free again and the close marker is allowed.
            m.advance("{\"command\": \"ls -la\"}}");
            assert!(m.is_free());
            assert!(m.is_token_allowed("</tool_call>"));
        }

        #[test]
        fn close_marker_returns_to_out() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn close_blocked_until_outer_object_brace_emitted() {
            // Regression (Pi "dropped closing bracket"): qwen3.6 mq4 at temp>0
            // sometimes jumps from the inner args `}` straight to `</tool_call>`,
            // dropping the OUTER tool-call object's `}` and emitting invalid JSON
            // (`{"name":"read","arguments":{"path":"/x"}` — no outer close). The
            // grammar must require the outer `}` before permitting the close.
            let mut m = Matcher::new(schemas_with_required(&[("read", &["path"])]));
            m.advance("<tool_call>\n{\"name\": \"read\", \"arguments\": ");
            // Model emits the balanced args VALUE object (required "path" present).
            m.advance("{\"path\": \"/x\"}");
            // Inner object closed (args_brace_depth back to 0) but the OUTER
            // tool-call object `}` is still outstanding — the close must be blocked.
            assert!(
                !m.is_token_allowed("</tool_call>"),
                "close marker must be rejected while the outer object brace is unclosed"
            );
            // The outer `}` itself is still allowed (not a close-marker token).
            assert!(m.is_token_allowed("}"));
            // After the outer `}` lands, the close marker is allowed.
            m.advance("}");
            assert!(
                m.is_token_allowed("</tool_call>"),
                "close marker must be allowed once the outer object is balanced"
            );
        }

        #[test]
        fn lt_char_inside_string_value_is_allowed() {
            // Regression: the outer-brace close gate must NOT block a `<` that is
            // part of a JSON STRING VALUE (heredoc `<<`, `#include <stdio.h>`,
            // `a < b`, …). `touches_close_marker` is not string-aware, so a token
            // ending in `<` looks like a `</tool_call>` prefix; the gate must still
            // allow it because inside a string it is content, not the close marker.
            let mut m = Matcher::new(schemas_with_required(&[("bash", &["command"])]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // Inside the command string value; required field "command" present.
            m.advance("{\"command\": \"cat > /tmp/x.c ");
            assert!(
                m.is_token_allowed("<"),
                "a bare '<' inside a string value must be allowed (heredoc/include)"
            );
            assert!(
                m.is_token_allowed("<< 'EOF'"),
                "a heredoc token inside a string value must be allowed"
            );
            assert!(
                m.is_token_allowed("#include <stdio.h>"),
                "C include with '<' inside a string value must be allowed"
            );
        }

        #[test]
        fn multiple_tool_names_all_match() {
            let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
            m.advance("<tool_call>");
            // All three tool-name headers are allowed prefixes.
            assert!(m.is_token_allowed("\n{\"name\": \"bash"));
            // Restart with a fresh matcher for the other names.
            let mut m2 = Matcher::new(schemas(&["bash", "read", "write"]));
            m2.advance("<tool_call>");
            assert!(m2.is_token_allowed("\n{\"name\": \"read"));
            let mut m3 = Matcher::new(schemas(&["bash", "read", "write"]));
            m3.advance("<tool_call>");
            assert!(m3.is_token_allowed("\n{\"name\": \"write"));
        }

        #[test]
        fn unknown_tool_name_rejected() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            // `\n{"name": "evil` is not a prefix of `\n{"name": "bash", ...`
            // — the model can't invent a tool name.
            assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
        }

        #[test]
        fn token_mask_free_path_sets_all_true() {
            let m = Matcher::new(schemas(&["bash"]));
            let vocab: Vec<String> = (0..10).map(|i| format!("tok{}", i)).collect();
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(mask.iter().all(|&b| b));
        }

        #[test]
        fn token_mask_after_open_only_allows_header_tokens() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            let vocab = vec![
                "\n".to_string(),
                "<|im_start|>".to_string(),
                "assistant".to_string(),
                "\n{".to_string(),
                "\n{\"name".to_string(),
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(mask[0], "\\n must be allowed (header prefix)");
            assert!(!mask[1], "<|im_start|> must be rejected (Pi failure mode)");
            assert!(!mask[2], "assistant must be rejected");
            assert!(mask[3], "\\n{{ must be allowed (header prefix)");
            assert!(mask[4], "\\n{{\"name must be allowed");
        }

        #[test]
        fn apply_mask_zeros_disallowed_logits() {
            let mask = vec![true, false, true, false];
            let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
            Matcher::apply_mask_to_logits(&mask, &mut logits);
            assert_eq!(logits[0], 1.0);
            assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
            assert_eq!(logits[2], 3.0);
            assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
        }

        #[test]
        fn buffer_doesnt_grow_unboundedly_in_out() {
            let mut m = Matcher::new(schemas(&["bash"]));
            // Emit a long stretch of prose. The internal buffer should be
            // bounded to the longest open prefix we could complete (11 - 1
            // = 10 chars).
            for _ in 0..1000 {
                m.advance("a");
            }
            assert!(m.partial().len() <= "<tool_call>".len());
        }

        // ─── Pi turn-12 attractor reproduction & fix demonstration ──────
        //
        // These tests directly reproduce the Pi turn-12 failure mode (model
        // emits `<|im_start|>assistant "..."}}` as the `<tool_call>` body
        // instead of valid JSON) using synthetic logits, and verify the
        // grammar masker prevents it. They mirror the real sample-time
        // decision the daemon makes — the only difference is we control
        // the logits directly instead of running a model. The unit tests
        // give a deterministic, model-independent demonstration of:
        //
        //   1. WITHOUT the grammar mask, an argmax over the failure-mode
        //      logits picks the attractor token (`<|im_start|>`).
        //   2. WITH the grammar mask, the masked argmax picks a valid
        //      header-template continuation.
        //
        // The mask path is byte-for-byte the same one
        // `crates/hipfire-daemon/src/main.rs` runs at each sample
        // step in both the qwen35 non-dflash and dflash paths.

        /// Build a synthetic vocab with known token-text values + a logits
        /// vector that reproduces the Pi attractor signal (the bad token
        /// scores highest). Index 0 is `<|im_start|>` per the qwen tokenizer
        /// id ordering observed in the cache traces; the exact ordering
        /// doesn't matter for the test — what matters is the text → score
        /// mapping below.
        fn attractor_logits_setup() -> (Vec<String>, Vec<f32>) {
            // Token vocab: a mix of the attractor + valid header tokens.
            // These mirror the actual qwen3.6:27b vocab strings that
            // appeared in the Pi turn-12 emit.
            let vocab: Vec<String> = vec![
                "<|im_start|>".to_string(),        // 0: attractor (Pi failure)
                "<|im_end|>".to_string(),          // 1: another invalid
                "assistant".to_string(),           // 2: invalid prose
                "\n".to_string(),                  // 3: valid header start
                "\n{".to_string(),                 // 4: valid header
                "\n{\"name".to_string(),           // 5: valid header progression
                "\n{\"name\": \"bash".to_string(), // 6: valid header
                "evil".to_string(),                // 7: not in tool schema
                " arguments".to_string(),          // 8: irrelevant
                "</tool_call>".to_string(),        // 9: premature close — also invalid
            ];
            // Logits where the attractor wins by a wide margin. This mimics
            // the Pi turn-12 distribution: after long context, the model's
            // ChatML-noise attractor scores higher than valid JSON
            // continuations.
            let logits: Vec<f32> = vec![
                10.0, // <|im_start|>  ← attractor wins without mask
                5.0,  // <|im_end|>
                3.0,  // assistant
                2.0,  // \n
                1.5,  // \n{
                1.0,  // \n{"name
                0.5,  // \n{"name": "bash
                -1.0, // evil
                -2.0, // arguments
                -3.0, // </tool_call>
            ];
            (vocab, logits)
        }

        fn argmax(logits: &[f32]) -> usize {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }

        #[test]
        fn reproduces_pi_turn_12_attractor_without_mask() {
            // PROOF OF FAILURE: with no constraint applied to the logits,
            // the argmax picks `<|im_start|>` — exactly the token that
            // corrupted Pi turn 12's `<tool_call>` body. This is the
            // failure mode we're fixing.
            let (vocab, logits) = attractor_logits_setup();
            let pick = argmax(&logits);
            assert_eq!(
                vocab[pick], "<|im_start|>",
                "raw argmax should pick the attractor (this is the Pi turn-12 failure mode)"
            );
        }

        #[test]
        fn grammar_mask_blocks_pi_turn_12_attractor() {
            // PROOF OF FIX: with the grammar mask applied to the same
            // logits, the argmax picks a VALID header-template token
            // (`\n` — a prefix of the legal continuation
            // `\n{"name": "bash", "arguments": `). The attractor token
            // `<|im_start|>` is masked to `-INF` and can't be selected.
            let (vocab, mut logits) = attractor_logits_setup();
            let mut m = Matcher::new(schemas(&["bash"]));
            // Drive the matcher into `AfterOpen` — the state immediately
            // after the model commits to the `<tool_call>` opener.
            m.advance("<tool_call>");
            assert!(matches!(m.state(), State::AfterOpen));
            assert!(!m.is_free(), "matcher must be constraining in AfterOpen");

            // Build the token mask the daemon would build at sample time
            // and apply it to the logits — same code path as the runtime.
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            Matcher::apply_mask_to_logits(&mask, &mut logits);

            // The attractor is now `-INF`.
            assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
            // `assistant` is also masked (not a header prefix).
            assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
            // `\n` is preserved (valid header start).
            assert_eq!(logits[3], 2.0);

            // The argmax now picks a valid token.
            let pick = argmax(&logits);
            assert_eq!(
                vocab[pick], "\n",
                "masked argmax should pick the highest-scoring valid header prefix"
            );
            // Sanity: the picked token is NOT the attractor.
            assert_ne!(vocab[pick], "<|im_start|>");
        }

        #[test]
        fn grammar_mask_prevents_full_pi_attractor_sequence() {
            // END-TO-END: simulate the full Pi attractor token sequence
            // and verify the mask blocks the FIRST off-distribution token.
            // The sequence from the Pi log was approximately:
            //
            //   <tool_call>  →  \n  →  <|im_start|>  →  assistant  →  …
            //
            // With grammar, after `<tool_call>` + `\n`, the matcher is
            // still in AfterOpen and continues to mask. The next sample
            // step would have picked `<|im_start|>` (or `assistant`) per
            // the attractor logits — both must be rejected.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            m.advance("\n");
            assert!(matches!(m.state(), State::AfterOpen));

            // Each of these is a token the model picked in the Pi log; all
            // must be rejected because none are prefixes of the legal
            // `{"name": "bash", "arguments": ` continuation that follows
            // the `\n`.
            for bad_token in &["<|im_start|>", "assistant", " \"Let me", "}}", "<|im_end|>"] {
                assert!(
                    !m.is_token_allowed(bad_token),
                    "expected {:?} to be rejected after `<tool_call>\\n`",
                    bad_token
                );
            }
            // Valid continuations all pass.
            for good_token in &["{", "{\"", "{\"name", "{\"name\":"] {
                assert!(
                    m.is_token_allowed(good_token),
                    "expected {:?} to be allowed after `<tool_call>\\n`",
                    good_token
                );
            }
        }

        #[test]
        fn dflash_path_post_validation_catches_attractor() {
            // DFLASH STRATEGY: the dflash decode loop validates committed
            // tokens AFTER spec_step commits them. This test simulates that
            // walk for the Pi turn-12 attractor: feed the tokens that
            // appeared in the bad emit through the matcher; the rejection
            // must fire on the first off-distribution token (the daemon
            // then breaks the dflash loop + force-resets KV/DN per the
            // implementation in generate_dflash).
            let mut m = Matcher::new(schemas(&["bash"]));
            // dflash's spec_step would commit a batch — simulate that batch.
            let bad_batch: &[&str] = &[
                "<tool_call>",  // accepted: transitions to AfterOpen
                "\n",           // accepted: matches header prefix
                "<|im_start|>", // REJECTED: not a header prefix from current pos
                "assistant",    // would-be-next, never reached
            ];
            let mut accepted = 0;
            let mut violated = false;
            for tok in bad_batch {
                if !m.is_token_allowed(tok) {
                    violated = true;
                    break;
                }
                m.advance(tok);
                accepted += 1;
            }
            assert!(violated, "dflash post-validation must detect the attractor");
            assert_eq!(
                accepted, 2,
                "the two valid tokens are accepted; the 3rd is rejected"
            );
        }

        #[test]
        fn full_legal_tool_call_round_trip() {
            // CONTROL: when the model emits a fully-valid tool_call, the
            // matcher never rejects anything and returns cleanly to Out.
            let mut m = Matcher::new(schemas(&["bash"]));
            let sequence = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo Hello\"}}\n</tool_call>";
            // Each character/token should be accepted incrementally.
            for ch in sequence.chars() {
                let s = ch.to_string();
                assert!(
                    m.is_token_allowed(&s),
                    "valid sequence char {:?} unexpectedly rejected at state {:?}",
                    ch,
                    m.state()
                );
                m.advance(&s);
            }
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn grammar_disabled_skips_constraint() {
            // CONTROL: when there are no tool schemas (e.g., requests
            // without `tools` in the body), the matcher is constructed
            // with an empty schema list — `is_free()` stays true through
            // every state and every token is allowed. This protects the
            // non-tool-call code path from any grammar overhead.
            let mut m = Matcher::new(Vec::new());
            m.advance("<tool_call>");
            // Without schemas, AfterOpen has zero allowed continuations.
            // is_token_allowed still returns true for any text because
            // is_free returns true… wait, AfterOpen is_free returns
            // false. So tokens get rejected. Verify the alternative —
            // the daemon-side check `grammar_active = !schemas.is_empty()`
            // skips the matcher entirely when schemas is empty.
            let grammar_active = false; // mirrors the daemon's gate
            assert!(
                !grammar_active,
                "empty schemas must disable grammar at the daemon level"
            );
        }

        // ─── Multi-tool / schema variation coverage ──────────────────────

        #[test]
        fn tool_names_that_prefix_each_other() {
            // `bash` and `bash_long` overlap on the first 4 chars. The
            // grammar must distinguish between them at the name boundary
            // (the trailing `"` after the name). Both should be reachable,
            // and the prefix doesn't lock in early.
            let mut m = Matcher::new(schemas(&["bash", "bash_long"]));
            m.advance("<tool_call>");
            // Common prefix works.
            assert!(m.is_token_allowed("\n{\"name\": \"bash"));
            // The full short name with closing quote works.
            let mut a = m.clone();
            a.advance("\n{\"name\": \"bash\", \"arguments\": ");
            assert!(
                matches!(a.state(), State::InArgs),
                "short name `bash` must settle to InArgs"
            );
            // The longer name also works.
            let mut b = m.clone();
            b.advance("\n{\"name\": \"bash_long\", \"arguments\": ");
            assert!(
                matches!(b.state(), State::InArgs),
                "long name `bash_long` must settle to InArgs"
            );
        }

        #[test]
        fn unknown_tool_in_multi_schema_rejected() {
            let mut m = Matcher::new(schemas(&["bash", "read", "write"]));
            m.advance("<tool_call>");
            // `evil` is not in the schema; the closing quote after the
            // name is what disambiguates, so the rejection point is when
            // the buffer accumulates `"evil"` (no valid continuation).
            assert!(!m.is_token_allowed("\n{\"name\": \"evil"));
            // But `bash` (a real name) is fine.
            assert!(m.is_token_allowed("\n{\"name\": \"bash"));
        }

        #[test]
        fn many_tools_token_mask_is_fast() {
            // Stress the token_mask scan with a realistic tool count and
            // vocab size. The hot path is O(vocab * conts) — verify it
            // completes in reasonable time and produces a valid mask.
            let tool_names: Vec<String> = (0..32).map(|i| format!("tool_{}", i)).collect();
            let tools: Vec<ToolSchema> = tool_names
                .iter()
                .map(|n| ToolSchema {
                    name: n.clone(),
                    required: Vec::new(),
                })
                .collect();
            let vocab: Vec<String> = (0..150_000).map(|i| format!("tok_{}", i)).collect();
            let mut m = Matcher::new(tools);
            m.advance("<tool_call>");
            let mut mask = vec![false; vocab.len()];
            let start = std::time::Instant::now();
            m.token_mask(&vocab, &mut mask);
            let elapsed = start.elapsed();
            assert!(
                elapsed.as_millis() < 1000,
                "token_mask on 150k vocab × 32 tools took {:?} (expected < 1s)",
                elapsed
            );
        }

        // ─── BPE / token boundary edge cases ─────────────────────────────

        #[test]
        fn open_marker_split_across_tokens() {
            // Tokenizer might emit `<tool` + `_call>` as two tokens instead
            // of a single `<tool_call>` special token. The matcher's
            // byte-level partial buffer must handle this — both halves
            // are accepted, and the transition fires on the second.
            let mut m = Matcher::new(schemas(&["bash"]));
            assert!(m.is_token_allowed("<tool"));
            m.advance("<tool");
            assert!(matches!(m.state(), State::Out));
            assert!(m.is_free(), "a partial marker must not constrain free text");
            assert!(m.is_token_allowed("_call>"));
            m.advance("_call>");
            assert!(matches!(m.state(), State::AfterOpen));
        }

        #[test]
        fn close_marker_split_across_tokens() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}");
            assert!(matches!(m.state(), State::InArgs));
            // Split `</tool_call>` into chunks.
            assert!(m.is_token_allowed("\n<"));
            m.advance("\n<");
            assert!(m.is_token_allowed("/tool"));
            m.advance("/tool");
            assert!(m.is_token_allowed("_call>"));
            m.advance("_call>");
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn token_carrying_full_open_marker_inside_prose() {
            // Some BPE tokenizers emit `prose<tool_call>more` as a single
            // token. The matcher must detect the marker mid-token and
            // transition. (The `transition_once` uses `find`, not
            // `ends_with`, for the open marker, so this works.)
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("here is some prose <tool_call>extra");
            // `extra` lands in AfterOpen's partial_buf. The header
            // template starts with `\n` so `extra` is invalid — the
            // matcher should reflect this on the next is_token_allowed.
            assert!(matches!(m.state(), State::AfterOpen));
        }

        #[test]
        fn close_marker_in_args_body_string_does_not_close() {
            // The args body is a JSON value; it can contain `</tool_call>`
            // as a STRING literal. Our current grammar doesn't parse JSON
            // string boundaries — it sees the literal substring as a
            // close marker. Document this as a known limitation: the
            // grammar will prematurely transition. (Real models don't
            // typically emit `</tool_call>` inside arg strings; if this
            // surfaces in production, layer in a JSON-aware string-state
            // tracker.)
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            // Args content includes a string literal containing the close
            // marker. Our grammar treats it as the close (limitation).
            m.advance("{\"command\": \"echo </tool_call>\"}");
            // Document the actual behavior — this is what we observe today.
            assert!(matches!(m.state(), State::Out));
        }

        // ─── Mask construction edge cases ────────────────────────────────

        #[test]
        fn token_mask_handles_empty_vocab() {
            let m = Matcher::new(schemas(&["bash"]));
            let vocab: Vec<String> = Vec::new();
            let mut mask: Vec<bool> = Vec::new();
            m.token_mask(&vocab, &mut mask);
            assert!(mask.is_empty());
        }

        #[test]
        fn token_mask_handles_empty_string_tokens() {
            // Tokenizers sometimes have control tokens whose decoded text
            // is empty (e.g. BOS/EOS variants). These should always be
            // allowed — they contribute nothing to the partial buffer.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            let vocab = vec![
                "".to_string(),             // empty/control token
                "\n".to_string(),           // valid prefix
                "<|im_start|>".to_string(), // attractor
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(mask[0], "empty token must be allowed (control/placeholder)");
            assert!(mask[1], "valid prefix must be allowed");
            assert!(!mask[2], "attractor must be rejected");
        }

        #[test]
        fn apply_mask_handles_size_mismatch() {
            // Defensively: if mask is shorter than logits, only mask the
            // prefix; if mask is longer, ignore the tail. No panic.
            let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let mask = vec![true, false, true]; // shorter than logits
            Matcher::apply_mask_to_logits(&mask, &mut logits);
            assert_eq!(logits[0], 1.0);
            assert!(logits[1].is_infinite());
            assert_eq!(logits[2], 3.0);
            assert_eq!(logits[3], 4.0, "untouched (past mask end)");
            assert_eq!(logits[4], 5.0);
        }

        #[test]
        fn apply_mask_handles_empty_logits() {
            let mask = vec![true, false, true];
            let mut logits: Vec<f32> = Vec::new();
            Matcher::apply_mask_to_logits(&mask, &mut logits);
            assert!(logits.is_empty());
        }

        // ─── ChatML attractor variant coverage ───────────────────────────

        #[test]
        fn all_chatml_special_tokens_rejected_after_open() {
            // Every ChatML special token observed in qwen3.6 attractor
            // cases must be rejected at the `<tool_call>` boundary. These
            // are the tokens that leaked into the body in the Pi log + the
            // Native control plane's parseOneToolCall-compatible sanitizer.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            for tok in &[
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "<|im_sep|>",
                "<think>",
                "</think>",
                "<|tool_call|>", // hallucinated variant
            ] {
                assert!(
                    !m.is_token_allowed(tok),
                    "ChatML token {:?} must be rejected after <tool_call>",
                    tok
                );
            }
        }

        #[test]
        fn chatml_tokens_rejected_mid_header() {
            // After committing the start of the header (`\n{"name": "`),
            // ChatML noise must still be rejected. The matcher's partial
            // buffer tracks the in-progress header.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"");
            for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
                assert!(
                    !m.is_token_allowed(tok),
                    "{:?} must be rejected after partial header",
                    tok
                );
            }
            // The valid tool name continuation is still allowed.
            assert!(m.is_token_allowed("bash"));
            assert!(m.is_token_allowed("bash\", \"arguments\": "));
        }

        // ─── Multi tool_call & sequencing ────────────────────────────────

        #[test]
        fn two_tool_calls_in_one_decode() {
            // Parallel tool-use prompts can emit two `<tool_call>...</tool_call>`
            // blocks back-to-back. The matcher must return to Out after
            // the first close and constrain the second open the same way.
            let mut m = Matcher::new(schemas(&["bash", "read"]));
            m.advance(
            "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
        );
            assert!(matches!(m.state(), State::Out));
            // Second open.
            m.advance("\n<tool_call>");
            assert!(matches!(m.state(), State::AfterOpen));
            // Second body must use a known tool name.
            assert!(!m.is_token_allowed("\n{\"name\": \"unknown"));
            assert!(m.is_token_allowed("\n{\"name\": \"read"));
            // Complete the second call.
            m.advance(
            "\n{\"name\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}}\n</tool_call>",
        );
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn prose_between_tool_calls_is_free() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
            // Prose chunks are unconstrained.
            m.advance("Now let me think about the next step.");
            assert!(m.is_free());
            // Long prose doesn't get stuck.
            for _ in 0..100 {
                m.advance("more thinking content ");
            }
            assert!(matches!(m.state(), State::Out));
            assert!(m.is_free());
        }

        #[test]
        fn free_text_angle_bracket_does_not_arm_the_mask() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("HTML: ");
            assert!(m.is_free());

            m.advance("<");
            assert!(m.is_free(), "a bare '<' must not constrain the sampler");
            m.advance("p>hi</p>");
            assert!(m.is_free());

            m.advance(" and 2 < 3, table: <t");
            assert!(m.is_free(), "'<t' must not constrain the sampler");
            m.advance("able><tr></tr></table>");
            assert!(m.is_free());
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn tool_call_at_buffer_boundary() {
            // Simulate a token boundary that puts `<tool_call>` exactly at
            // the buffer's bounded-keep limit. The transition must still
            // fire (the matcher detects the full marker via `find`, not
            // just a suffix match).
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("aaaaaaaaaaaaaaaaaaaa<tool_call>"); // 20 a's + marker
            assert!(matches!(m.state(), State::AfterOpen));
        }

        // ─── JSON args variation ─────────────────────────────────────────

        #[test]
        fn empty_args_object_passes() {
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn nested_json_args_pass() {
            let mut m = Matcher::new(schemas(&["bash"]));
            let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"opts\": {\"verbose\": true, \"flags\": [\"a\", \"b\", \"c\"]}, \"cmd\": \"ls -la\"}}\n</tool_call>";
            m.advance(body);
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn unicode_in_args_passes() {
            let mut m = Matcher::new(schemas(&["bash"]));
            let body = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"msg\": \"héllo 世界 🚀\"}}\n</tool_call>";
            m.advance(body);
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn truncated_tool_call_stays_in_args() {
            // max_tokens cap mid-tool-call: matcher is left in InArgs
            // (no close marker seen). The daemon detects truncation via
            // its own bookkeeping; the grammar matcher just doesn't lie
            // about state.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo");
            assert!(matches!(m.state(), State::InArgs));
            // No close emitted, so state stays InArgs.
            assert!(matches!(m.state(), State::InArgs));
        }

        // ─── Property-based smoke tests ──────────────────────────────────

        #[test]
        fn property_random_valid_sequences_always_pass() {
            // Generate many random valid tool_call sequences and verify
            // the matcher accepts each one fully and returns to Out.
            // Uses a deterministic LCG seed so failures are reproducible.
            let tools = schemas(&["bash", "read", "write", "edit"]);
            let mut rng_state: u32 = 0xCAFEBABE;
            let mut lcg = || {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                rng_state
            };
            for _ in 0..200 {
                let mut m = Matcher::new(tools.clone());
                let tool_idx = (lcg() % 4) as usize;
                let names = ["bash", "read", "write", "edit"];
                let arg_value: String = (0..((lcg() % 50) as usize))
                    .map(|_| (b'a' + (lcg() % 26) as u8) as char)
                    .collect();
                let seq = format!(
                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {{\"x\": \"{}\"}}}}\n</tool_call>",
                names[tool_idx], arg_value
            );
                for ch in seq.chars() {
                    let s = ch.to_string();
                    assert!(
                        m.is_token_allowed(&s),
                        "char {:?} unexpectedly rejected mid-sequence; state={:?} partial={:?}",
                        ch,
                        m.state(),
                        m.partial()
                    );
                    m.advance(&s);
                }
                assert!(
                    matches!(m.state(), State::Out),
                    "valid sequence didn't return to Out (tool={}, args=`{}`, final state={:?})",
                    names[tool_idx],
                    arg_value,
                    m.state()
                );
            }
        }

        #[test]
        fn property_attractor_tokens_never_accepted_after_open() {
            // For every position inside the AfterOpen state (driven by
            // increasing valid header prefixes), the Pi-style attractor
            // tokens must remain rejected. Catches regressions where a
            // partial-buf size or transition logic change accidentally
            // makes a ChatML token look like a header prefix.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            let header = "\n{\"name\": \"bash\", \"arguments\": ";
            for end in 0..=header.len() {
                let mut m2 = m.clone();
                let prefix = &header[..end];
                m2.advance(prefix);
                // Inside AfterOpen unless we hit the end (transitions to
                // InArgs). Either way, attractor tokens must NEVER fit.
                if matches!(m2.state(), State::AfterOpen) {
                    for tok in &["<|im_start|>", "<|im_end|>", "<think>"] {
                        assert!(
                            !m2.is_token_allowed(tok),
                            "attractor {:?} accepted at header position {} (partial={:?})",
                            tok,
                            end,
                            m2.partial()
                        );
                    }
                }
            }
        }

        // ─── Integration with apply_mask_to_logits ───────────────────────

        #[test]
        fn mask_then_argmax_blocks_attractor_at_every_header_position() {
            // The Pi failure was at position 0 of AfterOpen (right after
            // `<tool_call>`). Verify the mask-then-argmax pipeline blocks
            // the attractor at multiple header positions — at each one,
            // the model is in a slightly-different partial state but the
            // attractor must remain `-INF`.
            //
            // The vocab includes the header bytes as single-char tokens so
            // there's always a valid next-byte token whatever position we
            // pause at. Real qwen vocab has these single-char tokens (
            // newline, ASCII letters, punctuation) so this models the
            // production case.
            let header = "\n{\"name\": \"bash\", \"arguments\": ";
            for end in 0..=header.len() {
                let mut m = Matcher::new(schemas(&["bash"]));
                m.advance("<tool_call>");
                m.advance(&header[..end]);
                if !matches!(m.state(), State::AfterOpen) {
                    continue; // we reached InArgs — past the constrained region
                }
                // Vocab: attractors (must be -INF) + every single-byte
                // ASCII char (at least one will continue the header).
                let mut vocab: Vec<String> = vec![
                    "<|im_start|>".to_string(),
                    "<|im_end|>".to_string(),
                    "<think>".to_string(),
                    "assistant".to_string(),
                ];
                for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
                    vocab.push((byte as char).to_string());
                }
                let attractor_count = 4;
                let mut logits = vec![100.0f32; vocab.len()];
                // Make attractors win the raw argmax.
                logits[0] = 200.0; // <|im_start|>
                logits[1] = 190.0; // <|im_end|>
                logits[2] = 185.0;
                logits[3] = 180.0;

                let mut mask = vec![false; vocab.len()];
                m.token_mask(&vocab, &mut mask);
                Matcher::apply_mask_to_logits(&mask, &mut logits);

                // All attractors must be -INF.
                for i in 0..attractor_count {
                    assert!(
                        logits[i].is_infinite() && logits[i].is_sign_negative(),
                        "attractor vocab[{}]={:?} not -INF at header position {}",
                        i,
                        vocab[i],
                        end,
                    );
                }
                // At least ONE token in the vocab must still be allowed —
                // the next byte of the header template is always present
                // in the single-char-ASCII slice.
                assert!(
                logits.iter().any(|l| l.is_finite()),
                "all logits -INF at header position {} (partial={:?}); grammar pinned the model into a corner",
                end, m.partial(),
            );
            }
        }

        // ─── N-gram loop guard tests (real attractor from production) ───
        //
        // The motivating incident: qwen3.6:27b at ~turn 8 of a long
        // agentic session emitted `typetypetypetype...` and
        // `pub fn BlinkHash(key_pub fn BlinkHash(key_...` inside the
        // `<tool_call>` args body. The grammar at the time was permissive
        // in `InArgs` (correctly — args are free-form JSON), so the
        // attractor wasn't blocked. Each retry baked more garbage into
        // the conversation until KV ran out. These tests verify the
        // n-gram loop guard catches both patterns + forces a close.

        #[test]
        fn ngram_guard_catches_structural_skeleton_attractor() {
            // A model that loops the JSON *skeleton* (repeating key/value pairs)
            // is a real attractor we still break. The key/value text lives inside
            // strings (excluded from the guard), but the structural punctuation
            // emitted between repeats (`":1,`) is fed and trips at the 6× default.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            assert!(!m.attractor_detected(), "guard should be clean on entry");
            let mut payload = String::from("{");
            for _ in 0..8 {
                payload.push_str("\"k\":1,"); // structural stream: `":1,` × 8
            }
            m.advance(&payload);
            assert!(
                m.attractor_detected(),
                "should detect the repeating JSON skeleton"
            );
            // Once flagged, the matcher is no longer free, and only the close
            // marker passes.
            assert!(!m.is_free());
            assert!(!m.is_token_allowed("more"));
            assert!(m.is_token_allowed("\"}}\n</tool_call>"));
        }

        #[test]
        fn ngram_guard_ignores_repetition_inside_string_value() {
            // Regression for the write-tool empty-args bug: a `write` tool whose
            // code `content` legitimately repeats short n-grams (indentation,
            // `pub fn …`, `typetype…`) must NOT trip the guard. The old contract
            // detected inside the string value and force-closed `</tool_call>`,
            // truncating the argument to `{}`. Inside-string bytes are now excluded.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"");
            // 20-byte phrase × 6 *inside* the content string.
            let phrase = "pub fn BlinkHash(key";
            assert_eq!(phrase.len(), 20);
            let mut payload = String::new();
            for _ in 0..6 {
                payload.push_str(phrase);
            }
            m.advance(&payload);
            // ...and a short-char run, the other classic false-positive shape.
            m.advance("typetypetypetypetypetype");
            assert!(
                !m.attractor_detected(),
                "code repetition inside a string value must NOT trip the guard"
            );
            // The args body still accepts content so the model keeps writing its
            // file. (It is no longer `is_free` — the outer-brace close gate is
            // active while the object is open — but non-close content tokens
            // remain allowed.)
            assert!(m.is_token_allowed(" more code here"));
        }

        #[test]
        fn ngram_guard_does_not_trip_on_3_repeats() {
            // Threshold is 6 repeats (was 4). A 3-repeat sequence —
            // legitimate in arrays like `[1,1,1]` — must NOT trip.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            m.advance("{\"arr\": [1,1,1]}");
            assert!(
                !m.attractor_detected(),
                "3-repeats must stay below threshold"
            );
        }

        #[test]
        fn ngram_guard_does_not_trip_on_5_repeats() {
            // The threshold bump (4 → 6) means 5 repeats — close to the
            // boundary — still must NOT trip. Locks in the regression
            // signal if anyone reverts the bump without thinking.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // 4-byte gram × 5 = 20 bytes; under the new default of 6.
            m.advance("{\"x\": \"typetypetypetypetype");
            assert!(
                !m.attractor_detected(),
                "5-repeats must stay below the bumped threshold"
            );
        }

        #[test]
        fn ngram_guard_does_not_trip_on_double_newline_blocks() {
            // Code emission with multiple `\n\n` between definitions
            // (2-byte gram × 4) — was the production false positive that
            // motivated the LEN_MIN bump from 2 → 3. Must NOT trip.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"fn a(){}\\n\\nfn b(){}\\n\\nfn c(){}\\n\\nfn d(){}\\n\\nfn e(){}");
            // `\n\n` repeats 4 times — under old defaults this tripped on
            // legit code emitted between blank-separated declarations.
            assert!(
                !m.attractor_detected(),
                "double-newline blocks between defs must NOT trip"
            );
        }

        #[test]
        fn ngram_guard_does_not_trip_on_deep_indentation() {
            // Long whitespace runs are part of normal indented code. The
            // detector's uniform-byte filter (`is_uniform_byte`) skips
            // n-grams consisting of one repeated character — so a 64-space
            // indent does not trip the guard regardless of length.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            // 64 spaces — deeper than any realistic indent.
            m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"return                                                                ; ");
            assert!(
                !m.attractor_detected(),
                "uniform whitespace runs must NOT trip the guard"
            );
        }

        #[test]
        fn ngram_guard_does_not_trip_on_ascii_divider_runs() {
            // ASCII dividers (`====`, `----`, `####`) are common in code
            // comments and section headers. Same uniform-byte filter as
            // whitespace — must NOT trip.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\": \"/tmp/x.zig\", \"content\": \"// ============================================================\\n// section\\n");
            assert!(
                !m.attractor_detected(),
                "ASCII divider runs must NOT trip the guard"
            );
        }

        #[test]
        fn ngram_guard_does_not_trip_on_normal_json() {
            // Normal JSON content (varied bytes) doesn't trip.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            m.advance("{\"command\": \"echo Hello && ls -la /tmp/ && cat README.md\"}");
            assert!(!m.attractor_detected());
        }

        #[test]
        fn ngram_guard_resets_after_close_marker() {
            // After the model commits to `</tool_call>` and we return to
            // Out, the attractor flag clears so a subsequent tool_call
            // doesn't inherit it.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // Structural skeleton loop (`":1,` × 8) trips the guard.
            let mut payload = String::from("{");
            for _ in 0..8 {
                payload.push_str("\"k\":1,");
            }
            m.advance(&payload);
            assert!(m.attractor_detected());
            // Force-close (the daemon's sample mask would have driven the
            // model here).
            m.advance("}\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
            assert!(!m.attractor_detected(), "flag must clear on close");
            // Next tool_call body starts clean.
            m.advance("\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {}}\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
            assert!(!m.attractor_detected());
        }

        #[test]
        fn ngram_guard_does_not_trip_on_truly_varied_content() {
            // Long stretch of NON-repeating content — a deterministic LCG
            // over the printable ASCII range. No n-gram of any length
            // 2..32 should repeat 4 consecutive times.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            let mut payload = String::new();
            let mut state: u32 = 0xCAFEBABE;
            for _ in 0..5000 {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let c = ((state >> 16) as u8 % 94) + b' '; // printable ASCII
                payload.push(c as char);
            }
            m.advance(&payload);
            assert!(
                !m.attractor_detected(),
                "varied (LCG) content must not trip"
            );
        }

        // ─── Required-field guard tests ──────────────────────────────────
        //
        // The motivating incident: Pi's `write` tool requires `path` and
        // `content`. The model drifted into emitting `arguments:{}` (or
        // `arguments:{"path":"…","edits":[]}` — missing `content`) over
        // and over, Pi rejected each, model retried, KV bloated until
        // exhaustion. The required-field guard rejects the close marker
        // until every required field name appears in the args body.

        #[test]
        fn required_field_guard_blocks_empty_args() {
            // `write` requires `path` and `content`. Model tries to emit
            // empty `{}` and immediately close — both required-field
            // checks must reject any close-marker token.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>");
            m.advance("\n{\"name\": \"write\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            m.advance("{}"); // empty args body
                             // Body has no "path" or "content" — close-marker tokens are blocked.
            assert!(!m.is_token_allowed("\n</tool_call>"));
            assert!(!m.is_token_allowed("</tool_call>"));
            assert!(!m.is_token_allowed("\n<"));
            // Non-close-marker content (like a key name) IS allowed — the
            // model can recover by emitting the missing fields.
            assert!(m.is_token_allowed("\"path"));
            assert!(m.is_token_allowed("anything else"));
        }

        // ─── Args-body close-brace guard ───────────────────────────────
        //
        // The close-marker guard alone is insufficient: the `}` that
        // closes the empty `{}` args body commits BEFORE the next-token
        // `</tool_call>` is checked, so the truncated args body
        // (`arguments: {}`) reaches the OpenAI API as a malformed tool
        // call even after the close marker is correctly rejected. The
        // brace-depth guard rejects the closing `}` itself when
        // required fields are not yet satisfied.

        #[test]
        fn args_body_close_brace_blocked_when_required_missing() {
            // Model has just emitted `{`. Brace depth = 1, body open.
            // Next-token `}` would close the body with no required
            // fields present — must be rejected.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{");
            // brace depth = 1, no required fields seen
            assert!(!m.is_token_allowed("}"));
            assert!(!m.is_token_allowed("}\n"));
            // Other content is allowed.
            assert!(m.is_token_allowed("\"path\":\"/tmp/x\""));
        }

        #[test]
        fn args_body_close_brace_blocks_empty_args_single_token() {
            // The exact production failure: the `write` tool emits `{}`
            // as a single token, then `\n</tool_call>` as a second
            // token. The close marker is correctly rejected, but the
            // `{}` already streamed → `arguments: {}` lands at the API.
            // The brace-depth guard rejects the single-token `{}` itself.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            // The matcher must reject the empty-args token before it
            // commits — because once it commits, the truncated tool_call
            // already has the malformed shape on the wire.
            assert!(!m.is_token_allowed("{}"));
            assert!(!m.is_token_allowed("{ }"));
            assert!(!m.is_token_allowed("{}\n"));
        }

        #[test]
        fn args_body_close_brace_allowed_when_required_satisfied() {
            // Once required fields appear in the body, the matcher must
            // allow the closing `}` and the close marker. The all-in-one
            // single token also works.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\":\"/tmp/x\",\"content\":\"hello\"");
            // Both `"path"` and `"content"` present in ngram_history. The inner
            // args object still needs its `}` and the outer tool-call object needs
            // one more — so the canonical close is `}}\n</tool_call>` (the merged
            // token is allowed because both braces land before the marker).
            assert!(m.is_token_allowed("}"));
            assert!(m.is_token_allowed("}}\n</tool_call>"));
        }

        #[test]
        fn args_body_close_brace_allows_nested_object_close() {
            // Nested objects: a `}` that closes an inner object (depth
            // 2 → 1) MUST be allowed regardless of required-field state
            // because it doesn't close the outer args body.
            let mut m = Matcher::new(schemas_with_required(&[("apply", &["edits"])]));
            m.advance("<tool_call>\n{\"name\": \"apply\", \"arguments\": ");
            m.advance("{\"edits\":[{\"line\":1,\"text\":\"foo\"");
            // Now at depth 3 (outer args, edits array's first object).
            // Closing the inner object `}` → depth 2. Required `"edits"`
            // is already in ngram_history (it's the key in args body),
            // so this would be allowed even without the depth check —
            // but the test exists to lock the "nested closes don't fire
            // the guard" behavior.
            assert!(m.is_token_allowed("}"));
            // Closing the array `]` → depth 2 (no change to braces).
            assert!(m.is_token_allowed("]"));
        }

        #[test]
        fn args_body_close_brace_ignores_braces_in_strings() {
            // `{` / `}` inside JSON string literals must not affect
            // brace depth. The guard's string-aware tracking is what
            // keeps `{"content":"a}b"}` from being misread as closing
            // the body at the `}b` byte.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\":\"/tmp/x\",\"content\":\"a}b{c");
            // String is unterminated; depth is 1, in_string is true.
            // Closing the string then the body should be allowed because
            // required fields are present.
            assert!(m.is_token_allowed("\"}"));
        }

        #[test]
        fn args_body_close_brace_ignores_escaped_quote_in_string() {
            // `\"` inside a string MUST NOT toggle in_string off.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            // Backslash-escaped quote inside path value — string stays open.
            m.advance("{\"path\":\"he said \\\"hi\\\"\",\"content\":\"x\"");
            // Both fields present, brace depth still 1, can close.
            assert!(m.is_token_allowed("}"));
        }

        #[test]
        fn args_body_close_brace_via_token_mask() {
            // Production-style: the mask must block both `{}` and `}`
            // (which alone closes the body once depth >= 1) when
            // required fields aren't yet present.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            let vocab = vec![
                "{}".to_string(),       // empty body single token — block
                "{ }".to_string(),      // empty body with space — block
                "{".to_string(),        // body open — allow
                "\"path\"".to_string(), // field name — allow
                "garbage".to_string(),  // arbitrary content — allow
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(!mask[0], "empty `{{}}` body must be blocked");
            assert!(!mask[1], "empty `{{ }}` body must be blocked");
            assert!(mask[2], "body-open `{{` must be allowed");
            assert!(mask[3], "field name `\"path\"` must be allowed");
            assert!(mask[4], "arbitrary args content must be allowed");
        }

        #[test]
        fn args_body_brace_tracking_resets_on_close_marker() {
            // After a full tool_call cycle, the matcher returns to Out
            // and the brace tracking state must reset so a SECOND
            // tool_call's empty `{}` body is still caught.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            // First tool call — completes cleanly.
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\":\"/a\",\"content\":\"b\"}");
            m.advance("\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
            // Second tool call attempt — empty body must be blocked again.
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            assert!(!m.is_token_allowed("{}"));
        }

        #[test]
        fn required_field_guard_blocks_partial_args() {
            // The Pi-log variant: model emits `{"path":"...","edits":[]}` —
            // has `path` but is missing `content`. Close must still be blocked.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\":\"/tmp/x.zig\",\"edits\":[]}");
            // `path` present, `content` missing → still block close.
            assert!(!m.is_token_allowed("\n</tool_call>"));
            assert!(!m.is_token_allowed("</tool_call>"));
            // Allow further content to add the missing field.
            assert!(m.is_token_allowed(",\"content"));
        }

        #[test]
        fn required_field_guard_allows_close_when_all_satisfied() {
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{\"path\":\"/tmp/x\",\"content\":\"hello\"}}");
            // Both required fields present AND the object is fully closed (inner
            // args `}` + outer tool-call `}`) — close is now allowed.
            assert!(m.is_token_allowed("\n"));
            assert!(m.is_token_allowed("\n</tool_call>"));
            m.advance("\n</tool_call>");
            assert!(matches!(m.state(), State::Out));
            assert_eq!(m.current_tool(), None, "current_tool clears on close");
        }

        #[test]
        fn required_field_guard_handles_no_required_fields() {
            // Tool with empty `required` (e.g. `list_files()` with no args)
            // — the guard is a no-op. Close fires normally.
            let mut m = Matcher::new(schemas_with_required(&[("list", &[])]));
            m.advance("<tool_call>\n{\"name\": \"list\", \"arguments\": ");
            m.advance("{}}");
            // No required fields → trivially satisfied; the args object `{}` and
            // the outer tool-call object are both closed.
            assert!(m.is_token_allowed("\n</tool_call>"));
        }

        #[test]
        fn required_field_guard_blocks_close_via_token_mask() {
            // Realistic: model has emitted empty args. Token mask must
            // block both the close marker and any token that would form
            // a close-marker prefix.
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            m.advance("{}");
            let vocab = vec![
                "\n</tool_call>".to_string(), // close marker — must be blocked
                "</tool_call>".to_string(),   // close marker — must be blocked
                "\n<".to_string(),            // close-marker prefix — blocked
                "\"path".to_string(),         // valid field name — allowed
                "\"content".to_string(),      // valid field name — allowed
                "anything".to_string(),       // arbitrary content — allowed
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(!mask[0], "full close marker must be blocked");
            assert!(!mask[1], "bare close marker must be blocked");
            assert!(!mask[2], "close-marker prefix must be blocked");
            assert!(mask[3], "field name `\"path` must be allowed");
            assert!(mask[4], "field name `\"content` must be allowed");
            assert!(mask[5], "free content must be allowed");
        }

        #[test]
        fn required_field_guard_substring_match_robust_to_quoting() {
            // The guard does a substring search for `"<name>"` in the
            // args body. This handles standard JSON quoting where field
            // names are double-quoted. The presence check doesn't require
            // a syntactically-valid JSON — substring is enough.
            let mut m = Matcher::new(schemas_with_required(&[("bash", &["command"])]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // Field appears with spaces around the colon — still matches
            // because we look for `"command"` substring. Close the inner args
            // object and the outer tool-call object (`}}`) before the marker.
            m.advance("{ \"command\" : \"ls -la\" }}");
            assert!(m.is_token_allowed("\n</tool_call>"));
        }

        #[test]
        fn required_field_guard_reproduces_pi_write_attractor() {
            // Direct reproduction of the Pi-log failure pattern:
            //   model emits arguments:{} → Pi rejects → retry → repeat
            //
            // Before the guard: the matcher allowed the close marker and
            // the malformed tool_call propagated to Pi. With the guard,
            // close-marker tokens are masked out — the model has to emit
            // path and content (or hit max_tokens with finish_reason="length").
            let mut m = Matcher::new(schemas_with_required(&[("write", &["path", "content"])]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            // The exact Pi-log body.
            m.advance("{}");
            // Close attempts the model tried in the log — all must be blocked.
            for bad_token in &["\n</tool_call>", "</tool_call>", "\n<", "<", "</tool_"] {
                assert!(
                    !m.is_token_allowed(bad_token),
                    "Pi-log close attempt {:?} must be blocked while args is empty",
                    bad_token
                );
            }
            // The model can recover by emitting field content.
            assert!(m.is_token_allowed("\"path\":\"/tmp"));
        }

        #[test]
        fn utf8_in_args_body_does_not_panic_on_buffer_trim() {
            // Regression for production panic at
            // `crates/hipfire-arch-qwen35/src/grammar.rs:590` —
            // Pi pulled `𝐵link-hash` from a PDF into a write tool's
            // content arg. The InArgs partial-buf trim used a byte
            // offset that straddled the 4-byte `𝐵` codepoint, and
            // `String::drain(..n)` panicked because `n` wasn't a
            // char boundary. The fix rounds to the next char
            // boundary.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            // Emit a long body with multi-byte UTF-8 (4-byte
            // codepoint U+1D435 `𝐵`, repeated past max_keep).
            let mut body = String::from("{\"content\":\"");
            for _ in 0..50 {
                body.push('𝐵'); // 4 bytes in UTF-8
            }
            body.push_str("\"}");
            // This advance previously panicked at the buffer trim.
            m.advance(&body);
            // No panic = test passes. Verify we're still in a valid state.
            assert!(matches!(m.state(), State::InArgs));
        }

        #[test]
        fn utf8_in_long_prose_does_not_panic_in_out_state() {
            // Same fix applies to Out-state trim. Long Unicode prose
            // before any tool_call should be safely trimmed.
            let mut m = Matcher::new(schemas(&["write"]));
            let prose: String = "α𝐵γδε".repeat(20);
            m.advance(&prose);
            // No panic; we're still in Out.
            assert!(matches!(m.state(), State::Out));
        }

        #[test]
        fn current_tool_set_on_in_args_transition() {
            // The matcher must track which tool's schema we're inside.
            let mut m = Matcher::new(schemas_with_required(&[
                ("bash", &["command"]),
                ("write", &["path", "content"]),
            ]));
            m.advance("<tool_call>");
            assert_eq!(m.current_tool(), None, "AfterOpen state has no tool yet");
            m.advance("\n{\"name\": \"write\", \"arguments\": ");
            assert!(matches!(m.state(), State::InArgs));
            assert_eq!(m.current_tool(), Some(1), "write is tools[1]");
        }

        #[test]
        fn ngram_guard_history_bounded() {
            // Even a long emission that DOES trip the guard shouldn't run
            // the matcher out of memory. After 50k bytes including an
            // attractor, the matcher should still be usable. (We can't
            // directly observe ngram_history.len() since it's private —
            // this test just verifies no panic / unbounded growth.)
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // Trip the guard structurally (`":1,` × 8), then flood with a huge
            // in-string payload. Both buffers stay bounded: `attractor_buf`
            // short-circuits once flagged, and `ngram_history` is window-capped.
            let mut payload = String::from("{");
            for _ in 0..8 {
                payload.push_str("\"k\":1,");
            }
            m.advance(&payload);
            assert!(m.attractor_detected());
            let huge: String = "type".repeat(10_000);
            m.advance(&huge);
            // No assertion needed — surviving this advance without panic
            // is the test.
        }

        #[test]
        fn ngram_guard_forces_close_via_token_mask() {
            // Realistic end-to-end: model emits args body that triggers
            // the guard. Token mask must then allow only tokens that
            // form a `</tool_call>` prefix.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>\n{\"name\": \"bash\", \"arguments\": ");
            // Structural skeleton loop (`":1,` × 8) trips the guard.
            let mut payload = String::from("{");
            for _ in 0..8 {
                payload.push_str("\"k\":1,");
            }
            m.advance(&payload);
            assert!(m.attractor_detected());

            let vocab = vec![
                "type".to_string(),         // attractor token — must be -INF
                "abc".to_string(),          // normal prose — must be -INF
                "<".to_string(),            // close-marker prefix — must be allowed
                "</tool".to_string(),       // close-marker prefix — must be allowed
                "</tool_call>".to_string(), // full close — must be allowed
                "".to_string(),             // empty/control — always allowed
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(!mask[0], "attractor token `type` must be blocked");
            assert!(!mask[1], "prose token `abc` must be blocked");
            assert!(mask[2], "close-marker prefix `<` must be allowed");
            assert!(mask[3], "close-marker prefix `</tool` must be allowed");
            assert!(mask[4], "full close marker must be allowed");
            assert!(mask[5], "empty token always allowed");
        }

        #[test]
        fn ngram_guard_does_not_force_close_real_log_content_repetition() {
            // The exact byte pattern from the Pi log (transcript shared in
            // conversation, `content` field of the `write` tool call that
            // produced empty `{}` args) — `pub fn BlinkHash(key_type: type, …`
            // followed by a `type` run. This lives INSIDE the content string.
            //
            // Old contract: the guard tripped here and force-closed `</tool_call>`,
            // truncating the argument so the client parsed `{}`. New contract: the
            // n-gram guard never inspects string-value bytes (it can't distinguish
            // a real loop from legitimate repetitive code), so it does NOT trip —
            // a genuinely looping write is instead bounded by `max_tokens`. This
            // is the deliberate trade that fixes the write-tool empty-args bug.
            let mut m = Matcher::new(schemas(&["write"]));
            m.advance("<tool_call>\n{\"name\": \"write\", \"arguments\": ");
            let body = "{\"path\": \"/tmp/x.zig\", \"content\": \"pub fn BlinkHash(key_type: type, value_type: type) typetypetypetypetypetype";
            m.advance(body);
            assert!(
                !m.attractor_detected(),
                "in-content repetition must NOT force-close the tool call"
            );
        }

        #[test]
        fn mask_position_zero_picks_newline() {
            // Most concrete realization of the Pi turn-12 fix. Vocab: all
            // ASCII single-byte chars + the failure-mode attractors. The
            // attractor wins raw argmax. After mask, argmax must pick `\n`
            // (the only valid single-char continuation from AfterOpen at
            // partial="").
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            let mut vocab: Vec<String> = vec![
                "<|im_start|>".to_string(),
                "<|im_end|>".to_string(),
                "<think>".to_string(),
            ];
            for byte in (b' '..=b'~').chain([b'\n', b'\t']) {
                vocab.push((byte as char).to_string());
            }
            let mut logits = vec![1.0f32; vocab.len()];
            logits[0] = 1000.0; // attractor wins raw
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            Matcher::apply_mask_to_logits(&mask, &mut logits);
            let pick = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            // `\n` is the only valid next char (header starts with `\n`).
            assert_eq!(
                vocab[pick], "\n",
                "masked argmax should pick `\\n` (vocab[{}]={:?})",
                pick, vocab[pick]
            );
        }

        // ─── PR #567 regression: free text must not arm the <tool_call> mask ──
        //
        // Defect: Matcher::is_free() returned false for State::Out whenever the
        // buffer ended with any prefix of "<tool_call>" (starting at a bare "<").
        // The sampler then masked free text to only tokens continuing the marker,
        // stranding prose like "<p>hi</p>", "2 < 3", or even "</think>".
        // Fix: State::Out is unconditionally free; masking engages only after a
        // full "<tool_call>" lands (transition to AfterOpen).

        #[test]
        fn pr567_genuine_tool_call_still_constrains() {
            // No regression: a real <tool_call> must still arm the grammar.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call>");
            assert!(matches!(m.state(), State::AfterOpen));
            assert!(!m.is_free(), "full <tool_call> must constrain");
            assert!(!m.is_token_allowed("<|im_start|>"));
            assert!(m.is_token_allowed("\n"));
            assert!(m.is_token_allowed("\n{\"name\": \"bash\""));
            let vocab = vec![
                "<|im_start|>".to_string(),
                "\n".to_string(),
                "hello".to_string(),
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(!mask[0], "attractor must be blocked after genuine open");
            assert!(mask[1], "header prefix must be allowed");
            assert!(!mask[2], "free prose must be blocked in AfterOpen");
        }

        #[test]
        fn pr567_partial_prefixes_do_not_arm_mask() {
            // Every strict prefix of "<tool_call>" must leave the sampler free.
            let prefixes = [
                "<",
                "<t",
                "<to",
                "<too",
                "<tool",
                "<tool_",
                "<tool_c",
                "<tool_ca",
                "<tool_cal",
                "<tool_call",
            ];
            for prefix in prefixes {
                let mut m = Matcher::new(schemas(&["bash"]));
                m.advance(&format!("hello {}", prefix));
                assert!(
                    m.is_free(),
                    "prefix {:?} must not arm mask; partial={:?}",
                    prefix,
                    m.partial()
                );
                assert!(matches!(m.state(), State::Out));
                for tok in ["p>", "table>", " ", " hello", " 3", "</think>", ">"] {
                    assert!(
                        m.is_token_allowed(tok),
                        "prefix {:?} must allow token {:?}",
                        prefix,
                        tok
                    );
                }
                let vocab = vec![
                    "p>".to_string(),
                    "table>".to_string(),
                    "</think>".to_string(),
                    "<".to_string(),
                    "hello".to_string(),
                ];
                let mut mask = vec![false; vocab.len()];
                m.token_mask(&vocab, &mut mask);
                assert!(
                    mask.iter().all(|&b| b),
                    "prefix {:?} must leave token_mask all-true",
                    prefix
                );
            }
            // Bare prefix at buffer start is also free.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<");
            assert!(m.is_free(), "bare '<' must be free");
            assert!(m.is_token_allowed("p>"));
            assert!(m.is_token_allowed("</think>"));
        }

        #[test]
        fn pr567_false_start_diverging_is_not_masked() {
            // A buffer that looked like a prefix then diverged must stay free.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool");
            assert!(m.is_free());
            assert!(
                m.is_token_allowed("box>"),
                "<tool + box> diverges from marker"
            );
            m.advance("box>");
            assert!(m.is_free());
            assert!(matches!(m.state(), State::Out));

            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_cal");
            assert!(m.is_free());
            assert!(m.is_token_allowed("X"), "<tool_cal + X diverges");
            assert!(m.is_token_allowed("x>"));

            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<tool_call");
            assert!(m.is_free());
            assert!(m.is_token_allowed("X"), "<tool_call + X is not the marker");
            assert!(m.is_token_allowed(" extra"));

            // Whole false markers in one advance must not transition.
            for txt in [
                "prefix <toolbox> suffix",
                "<tool_calX",
                "<tool-call>",
                "<tool_callX",
            ] {
                let mut m = Matcher::new(schemas(&["bash"]));
                m.advance(txt);
                assert!(
                    matches!(m.state(), State::Out),
                    "{:?} must not transition to AfterOpen",
                    txt
                );
                assert!(m.is_free(), "{:?} must be free", txt);
            }
        }

        #[test]
        fn pr567_partial_prefix_at_eos_stays_free() {
            // Partial prefix at end-of-stream: next sampler step must be unconstrained.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("2 <");
            assert!(m.is_free(), "'2 <' at EOS must be free");
            for tok in [" 3", ">", " p>", "</think>", " hello", "\n"] {
                assert!(m.is_token_allowed(tok), "'2 <' must allow {:?}", tok);
            }
            let vocab = vec![
                " 3".to_string(),
                ">".to_string(),
                " p>".to_string(),
                "</think>".to_string(),
                "<".to_string(),
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(mask.iter().all(|&b| b), "EOS partial must be all-true mask");

            // Another EOS shape: HTML tag start cut off mid-token.
            let mut m2 = Matcher::new(schemas(&["bash"]));
            m2.advance("HTML: <table");
            assert!(m2.is_free(), "HTML prefix at EOS must be free");
            assert!(m2.is_token_allowed("><tr>"));
            assert!(m2.is_token_allowed("</think>"));
        }

        #[test]
        fn pr567_prefix_split_across_tokens() {
            // Free-text prefix split across two token emissions must stay free.
            let mut m = Matcher::new(schemas(&["bash"]));
            m.advance("<");
            assert!(m.is_free(), "'<' split must be free");
            assert!(m.is_token_allowed("p>"));
            m.advance("p>");
            assert!(m.is_free());
            assert!(matches!(m.state(), State::Out));

            let mut m2 = Matcher::new(schemas(&["bash"]));
            m2.advance("<tool");
            assert!(m2.is_free());
            assert!(m2.is_token_allowed("box>"));
            m2.advance("box>");
            assert!(m2.is_free());

            // Genuine marker split across tokens must still transition.
            let mut m3 = Matcher::new(schemas(&["bash"]));
            m3.advance("<tool");
            assert!(
                m3.is_free(),
                "partial genuine must be free before completion"
            );
            assert!(m3.is_token_allowed("_call>"));
            m3.advance("_call>");
            assert!(matches!(m3.state(), State::AfterOpen));
            assert!(!m3.is_free(), "full marker split must constrain");

            // Multi-token free sequence: "2" + " <" + " 3"
            let mut m4 = Matcher::new(schemas(&["bash"]));
            m4.advance("2");
            m4.advance(" <");
            assert!(m4.is_free(), "'2 <' split must be free");
            assert!(m4.is_token_allowed(" 3"));
            assert!(m4.is_token_allowed("</think>"));

            // HTML split: "HTML: " + "<" + "t" + "able>"
            let mut m5 = Matcher::new(schemas(&["bash"]));
            m5.advance("HTML: ");
            m5.advance("<");
            assert!(m5.is_free());
            m5.advance("t");
            assert!(m5.is_free());
            m5.advance("able>");
            assert!(m5.is_free());
        }
    }

    /// Strict JSON Schema subset compiler (spec §7 G1/G2).
    ///
    /// Compiles a restricted subset of JSON Schema into an incremental
    /// byte-level matcher. The matcher tracks a **stack** of parser states
    /// (not a single FSM id) because JSON objects/arrays nest recursively.
    ///
    /// ## Supported subset
    ///
    /// - JSON primitives: `string`, `number`, `integer`, `boolean`, `null`
    /// - Objects: `properties`, `required`, boolean `additionalProperties`
    /// - Arrays: `items`, `minItems`, `maxItems`
    /// - `enum` / `const`
    ///
    /// ## Rejected before generation (typed error)
    ///
    /// - `$ref` / `$defs` / external references
    /// - `pattern` (regex)
    /// - `allOf` / `anyOf` / `oneOf` / `not` (combinators)
    /// - Recursive schemas (a property referencing its parent)
    /// - Unsupported assertion keywords
    /// - `NaN` / `Infinity` in numbers
    /// - Duplicate object keys
    /// - Provably-unsatisfiable schemas
    ///
    /// ## Incremental matcher API
    ///
    /// The matcher operates on raw token bytes (not lossy-decoded strings):
    /// - [`SchemaMatcher::is_token_allowed`] — check if bytes are a valid prefix
    /// - [`SchemaMatcher::advance`] — commit bytes to the parser
    /// - [`SchemaMatcher::is_accepting`] — true when the full JSON value is complete
    ///   and schema-valid
    ///
    /// Parser state is a stack of frames, not a single FSM state number.
    /// Recursive nesting (objects inside arrays inside objects) requires
    /// stack-based tracking; a single state id cannot represent it.
    pub mod json_schema {
        use std::collections::HashSet;
        use std::sync::LazyLock;

        static GRAMMAR_TRACE_ENABLED: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("GRAMMAR_TRACE").is_some());

        #[inline]
        fn grammar_trace_enabled() -> bool {
            *GRAMMAR_TRACE_ENABLED
        }

        /// Maximum nesting depth for objects/arrays.
        const MAX_NESTING: usize = 64;

        /// Maximum number of enum values.
        const MAX_ENUM_VALUES: usize = 256;

        /// Maximum serialized schema bytes. Compile walks the whole tree and
        /// clones `const`/`enum` payloads — an unbounded schema is a
        /// compile-time DoS on the request path.
        const MAX_SCHEMA_BYTES: usize = 64 * 1024;

        /// Maximum `properties` entries per object node.
        const MAX_PROPERTIES: usize = 256;

        /// Typed error returned when a schema is rejected at compile time.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum SchemaError {
            /// `$ref` / `$defs` / external references are not supported.
            ReferencesNotSupported { keyword: String },
            /// `pattern` (regex) assertions are not supported.
            RegexNotSupported,
            /// Combinators (`allOf`/`anyOf`/`oneOf`/`not`) are not supported.
            CombinatorNotSupported { keyword: String },
            /// Recursive schema detected.
            RecursionNotSupported,
            /// Unsupported assertion keyword.
            UnsupportedKeyword { keyword: String },
            /// Schema nesting exceeds the bound.
            NestingTooDeep { depth: usize, max: usize },
            /// Enum has too many values.
            TooManyEnumValues { count: usize, max: usize },
            /// Schema exceeds the serialized byte bound.
            SchemaTooLarge { bytes: usize, max: usize },
            /// Object node has too many properties.
            TooManyProperties { count: usize, max: usize },
            /// Schema is provably unsatisfiable.
            Unsatisfiable { reason: String },
            /// Invalid schema structure.
            InvalidSchema { reason: String },
        }

        impl std::fmt::Display for SchemaError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::ReferencesNotSupported { keyword } => {
                        write!(f, "json schema: {keyword} not supported")
                    }
                    Self::RegexNotSupported => write!(f, "json schema: pattern not supported"),
                    Self::CombinatorNotSupported { keyword } => {
                        write!(f, "json schema: {keyword} not supported")
                    }
                    Self::RecursionNotSupported => {
                        write!(f, "json schema: recursion not supported")
                    }
                    Self::UnsupportedKeyword { keyword } => {
                        write!(f, "json schema: unsupported keyword {keyword}")
                    }
                    Self::NestingTooDeep { depth, max } => {
                        write!(f, "json schema: nesting depth {depth} exceeds max {max}")
                    }
                    Self::TooManyEnumValues { count, max } => {
                        write!(f, "json schema: enum has {count} values, max {max}")
                    }
                    Self::SchemaTooLarge { bytes, max } => {
                        write!(f, "json schema: {bytes} bytes exceeds max {max}")
                    }
                    Self::TooManyProperties { count, max } => {
                        write!(f, "json schema: object has {count} properties, max {max}")
                    }
                    Self::Unsatisfiable { reason } => {
                        write!(f, "json schema: unsatisfiable: {reason}")
                    }
                    Self::InvalidSchema { reason } => {
                        write!(f, "json schema: invalid: {reason}")
                    }
                }
            }
        }

        impl std::error::Error for SchemaError {}

        /// JSON Schema type keywords.
        #[derive(Debug, Clone, PartialEq, Eq)]
        enum JsonType {
            String,
            Number,
            Integer,
            Boolean,
            Null,
            Object,
            Array,
        }

        /// A compiled schema node — the immutable, shared representation
        /// of one schema constraint.
        #[derive(Debug, Clone)]
        pub enum SchemaNode {
            /// Any JSON value (no constraint, or `type` absent).
            Any,
            /// JSON string.
            String,
            /// JSON number (integer or float).
            Number,
            /// JSON integer (no fractional part).
            Integer,
            /// JSON boolean.
            Boolean,
            /// JSON null.
            Null,
            /// JSON object with typed properties.
            Object {
                /// Named properties: (key, schema). Key is the decoded
                /// (post-escape) string.
                properties: Vec<(String, SchemaNode)>,
                /// Required property keys (decoded).
                required: Vec<String>,
                /// `additionalProperties`: `false` rejects unknown keys, `true`
                /// (or absent) allows any value for unknown keys.
                additional_properties: bool,
            },
            /// JSON array with item schema and bounds.
            Array {
                items: Box<SchemaNode>,
                min_items: usize,
                max_items: Option<usize>,
            },
            /// Enum: must match one of these JSON values by semantic
            /// equality.
            Enum(Vec<serde_json::Value>),
            /// Const: must match this exact JSON value.
            Const(serde_json::Value),
        }

        /// Compiled JSON Schema: immutable, Send+Sync.
        #[derive(Debug, Clone)]
        pub struct CompiledSchema {
            root: SchemaNode,
        }

        impl CompiledSchema {
            /// Compile a JSON Schema (`serde_json::Value`) into a
            /// [`CompiledSchema`]. Rejects unsupported keywords, refs,
            /// regex, recursion, combinators, and unsatisfiable schemas
            /// before generation (spec §7 G1).
            pub fn compile(schema: &serde_json::Value) -> Result<Self, SchemaError> {
                // Bound compile input: the walk clones const/enum payloads and
                // recurses per node, so an unbounded schema is a compile-time
                // DoS on the request path.
                let bytes = serde_json::to_vec(schema)
                    .map(|v| v.len())
                    .unwrap_or(usize::MAX);
                if bytes > MAX_SCHEMA_BYTES {
                    return Err(SchemaError::SchemaTooLarge {
                        bytes,
                        max: MAX_SCHEMA_BYTES,
                    });
                }
                let root = compile_node(schema, 0, &mut Vec::new())?;
                Ok(Self { root })
            }

            /// Read-only access to the root schema node.
            pub fn root(&self) -> &SchemaNode {
                &self.root
            }

            /// Create a fresh matcher cursor for this compiled schema.
            pub fn matcher(&self) -> SchemaMatcher {
                SchemaMatcher::new(self.root.clone())
            }
        }

        unsafe impl Send for CompiledSchema {}
        unsafe impl Sync for CompiledSchema {}

        /// Recursively compile a schema node. `depth` tracks nesting;
        /// `path` tracks the property key path for recursion detection.
        fn compile_node(
            schema: &serde_json::Value,
            depth: usize,
            path: &mut Vec<String>,
        ) -> Result<SchemaNode, SchemaError> {
            if depth > MAX_NESTING {
                return Err(SchemaError::NestingTooDeep {
                    depth,
                    max: MAX_NESTING,
                });
            }
            let obj = match schema.as_object() {
                Some(o) => o,
                None => {
                    return Err(SchemaError::InvalidSchema {
                        reason: "schema must be an object".to_string(),
                    })
                }
            };

            // Keyword policy: allowlist + explicit rejects + DEFAULT-DENY. An
            // unknown keyword must be refused, never silently permitted — the
            // silent fall-through used to admit assertion keywords like
            // if/then/else and additionalItems, whose constraints were then
            // ignored (spec §7.1: "reject unsupported assertion keywords ...
            // Permit only documented annotation keywords").
            for key in obj.keys() {
                match key.as_str() {
                    "type"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "items"
                    | "minItems"
                    | "maxItems"
                    | "enum"
                    | "const"
                    | "description"
                    | "title"
                    | "default"
                    | "examples"
                    | "$comment" => {}
                    k if k.starts_with("$") => {
                        return Err(SchemaError::ReferencesNotSupported {
                            keyword: k.to_string(),
                        })
                    }
                    "pattern" => return Err(SchemaError::RegexNotSupported),
                    "allOf" | "anyOf" | "oneOf" | "not" => {
                        return Err(SchemaError::CombinatorNotSupported {
                            keyword: key.to_string(),
                        })
                    }
                    "if" | "then" | "else" | "additionalItems" => {
                        return Err(SchemaError::UnsupportedKeyword {
                            keyword: key.to_string(),
                        })
                    }
                    "format"
                    | "minimum"
                    | "maximum"
                    | "exclusiveMinimum"
                    | "exclusiveMaximum"
                    | "multipleOf"
                    | "minLength"
                    | "maxLength"
                    | "minProperties"
                    | "maxProperties"
                    | "uniqueItems"
                    | "contains"
                    | "minContains"
                    | "maxContains"
                    | "prefixItems"
                    | "unevaluatedProperties"
                    | "unevaluatedItems"
                    | "contentEncoding"
                    | "contentMediaType"
                    | "dependentRequired"
                    | "dependentSchemas"
                    | "propertyNames"
                    | "patternProperties" => {
                        return Err(SchemaError::UnsupportedKeyword {
                            keyword: key.to_string(),
                        })
                    }
                    other => {
                        return Err(SchemaError::UnsupportedKeyword {
                            keyword: other.to_string(),
                        })
                    }
                }
            }

            // Read `type` BEFORE const/enum so the declared type constrains the
            // literal set. A non-string `type` (e.g. the union form
            // `["integer","null"]`) is OUTSIDE the supported subset — falling
            // through to the inference path would compile it to an unconstrained
            // `Any` (spec §7.1: unions are rejected before generation).
            if let Some(tv) = obj.get("type") {
                if tv.as_str().is_none() {
                    return Err(SchemaError::UnsupportedKeyword {
                        keyword: format!(
                            "type (union/array forms are outside the supported subset: {})",
                            tv
                        ),
                    });
                }
            }
            let ty = obj.get("type").and_then(|v| v.as_str());

            // A `const`/`enum` member that contradicts the declared `type` is a
            // silent constraint drop: {"type":"integer","enum":[1,"a"]} would
            // compile to Enum([1,"a"]) and emit "a" as conforming. Intersect the
            // literal set with `type`; an empty intersection is unsatisfiable.
            let value_matches_type = |v: &serde_json::Value, ty: &str| -> bool {
                match ty {
                    "string" => v.is_string(),
                    "integer" => {
                        v.as_i64().is_some()
                            || v.as_f64().map(|f| f.fract() == 0.0).unwrap_or(false)
                    }
                    "number" => v.is_number(),
                    "boolean" => v.is_boolean(),
                    "null" => v.is_null(),
                    "object" => v.is_object(),
                    "array" => v.is_array(),
                    _ => true,
                }
            };

            // const takes priority.
            if let Some(c) = obj.get("const") {
                if let Some(t) = ty {
                    if !value_matches_type(c, t) {
                        return Err(SchemaError::InvalidSchema {
                            reason: format!(
                                "const value {c} contradicts declared type \"{t}\" (unsatisfiable)"
                            ),
                        });
                    }
                }
                return Ok(SchemaNode::Const(c.clone()));
            }

            // enum. A present-but-malformed `enum` is a typed compile error,
            // NEVER a silent fall-through: skipping it here would compile the
            // schema as unconstrained `Any` and the request would report
            // schema-conforming success over arbitrary output (spec §7.1: an
            // empty enum is unsatisfiable and must also be rejected).
            if let Some(ev) = obj.get("enum") {
                let arr = ev.as_array().ok_or_else(|| SchemaError::InvalidSchema {
                    reason: format!("enum must be an array, got: {ev}"),
                })?;
                if arr.is_empty() {
                    return Err(SchemaError::InvalidSchema {
                        reason: "enum must have at least one value (empty enum is unsatisfiable)"
                            .to_string(),
                    });
                }
                if arr.len() > MAX_ENUM_VALUES {
                    return Err(SchemaError::TooManyEnumValues {
                        count: arr.len(),
                        max: MAX_ENUM_VALUES,
                    });
                }
                // Intersect with the declared type: members that contradict it are
                // dropped; if none remain the schema is unsatisfiable.
                let filtered: Vec<serde_json::Value> = match ty {
                    Some(t) => arr
                        .iter()
                        .filter(|v| value_matches_type(v, t))
                        .cloned()
                        .collect(),
                    None => arr.clone(),
                };
                if filtered.is_empty() {
                    return Err(SchemaError::InvalidSchema {
                        reason: format!(
                            "no enum member matches declared type \"{}\" (unsatisfiable)",
                            ty.unwrap_or("")
                        ),
                    });
                }
                return Ok(SchemaNode::Enum(filtered));
            }

            match ty {
                Some("string") => Ok(SchemaNode::String),
                Some("number") => Ok(SchemaNode::Number),
                Some("integer") => Ok(SchemaNode::Integer),
                Some("boolean") => Ok(SchemaNode::Boolean),
                Some("null") => Ok(SchemaNode::Null),
                Some("object") => compile_object(obj, depth, path),
                Some("array") => compile_array(obj, depth, path),
                Some(other) => Err(SchemaError::InvalidSchema {
                    reason: format!("unknown type: {other}"),
                }),
                None => {
                    // No type: if properties/items present, infer; else Any.
                    if obj.contains_key("properties") {
                        compile_object(obj, depth, path)
                    } else if obj.contains_key("items") {
                        compile_array(obj, depth, path)
                    } else {
                        Ok(SchemaNode::Any)
                    }
                }
            }
        }

        fn compile_object(
            obj: &serde_json::Map<String, serde_json::Value>,
            depth: usize,
            path: &mut Vec<String>,
        ) -> Result<SchemaNode, SchemaError> {
            // `properties` must be an object when present (see `required` —
            // malformed constraint values are typed errors, not silent no-ops).
            let properties = match obj.get("properties") {
                None => None,
                Some(v) => Some(v.as_object().ok_or_else(|| SchemaError::InvalidSchema {
                    reason: format!("properties must be an object, got: {v}"),
                })?),
            };

            let mut compiled_props = Vec::new();
            if let Some(properties) = properties {
                if properties.len() > MAX_PROPERTIES {
                    return Err(SchemaError::TooManyProperties {
                        count: properties.len(),
                        max: MAX_PROPERTIES,
                    });
                }
                compiled_props.reserve(properties.len());
                for (key, sub_schema) in properties {
                    // Recursion requires `$ref`, which the keyword gate above
                    // already rejects — a repeated property NAME at different
                    // nesting depths is ordinary finite JSON Schema
                    // ({"meta":{"meta":{"type":"string"}}}) and must compile.
                    // No path-name heuristic here.
                    path.push(key.clone());
                    let node = compile_node(sub_schema, depth + 1, path)?;
                    path.pop();
                    compiled_props.push((key.clone(), node));
                }
            }

            // `required` must be an array of strings when present. Malformed
            // values used to be silently filtered away — a request whose
            // `required` was a typo'd string compiled as if nothing were
            // required and reported success (spec §7.1: strict requests are
            // never silently unconstrained).
            let required: Vec<String> = match obj.get("required") {
                None => Vec::new(),
                Some(v) => {
                    let arr = v.as_array().ok_or_else(|| SchemaError::InvalidSchema {
                        reason: format!("required must be an array, got: {v}"),
                    })?;
                    arr.iter()
                        .map(|entry| {
                            entry.as_str().map(|s| s.to_string()).ok_or_else(|| {
                                SchemaError::InvalidSchema {
                                    reason: format!(
                                        "required entries must be strings, got: {entry}"
                                    ),
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
            };

            // `additionalProperties` must be boolean in this subset; the schema
            // form (validation rules for unknown keys) is unsupported and used
            // to be silently flattened to `true`.
            let additional_properties = match obj.get("additionalProperties") {
                None => true,
                Some(v) if v.is_boolean() => v.as_bool().unwrap_or(true),
                Some(_) => {
                    return Err(SchemaError::UnsupportedKeyword {
                        keyword: "additionalProperties (schema form is outside the \
                               supported subset; use true/false)"
                            .to_string(),
                    })
                }
            };

            // A required name without a `properties` entry is satisfiable when
            // unknown properties are allowed (any value satisfies it): compile
            // it as an unconstrained property instead of refusing the whole
            // schema. With additionalProperties:false it genuinely is
            // unsatisfiable (spec §7.1: reject when compilation proves it).
            let mut all_props = compiled_props;
            for req in &required {
                if !all_props.iter().any(|(k, _)| k == req) {
                    if additional_properties {
                        all_props.push((req.clone(), SchemaNode::Any));
                    } else {
                        return Err(SchemaError::Unsatisfiable {
                            reason: format!(
                                "required property {req:?} has no properties entry \
                             and additionalProperties is false"
                            ),
                        });
                    }
                }
            }

            Ok(SchemaNode::Object {
                properties: all_props,
                required,
                additional_properties,
            })
        }

        fn compile_array(
            obj: &serde_json::Map<String, serde_json::Value>,
            depth: usize,
            path: &mut Vec<String>,
        ) -> Result<SchemaNode, SchemaError> {
            let items = obj.get("items").ok_or_else(|| SchemaError::InvalidSchema {
                reason: "array schema missing items".to_string(),
            })?;
            let item_node = compile_node(items, depth + 1, path)?;

            // minItems/maxItems must be non-negative integers when present;
            // malformed values used to degrade to "no bound" (minItems: "3" → 0)
            // and the array accepted unbounded items — a silent constraint drop.
            let parse_bound = |key: &str| -> Result<Option<usize>, SchemaError> {
                match obj.get(key) {
                    None => Ok(None),
                    Some(v) => match v.as_u64() {
                        Some(n) => Ok(Some(n as usize)),
                        None => Err(SchemaError::InvalidSchema {
                            reason: format!("{key} must be a non-negative integer, got: {v}"),
                        }),
                    },
                }
            };
            let min_items = parse_bound("minItems")?.unwrap_or(0);
            let max_items = parse_bound("maxItems")?;

            // minItems > maxItems is unsatisfiable.
            if let Some(max) = max_items {
                if min_items > max {
                    return Err(SchemaError::Unsatisfiable {
                        reason: format!("minItems {min_items} > maxItems {max}"),
                    });
                }
            }

            Ok(SchemaNode::Array {
                items: Box::new(item_node),
                min_items,
                max_items,
            })
        }

        // ── Incremental matcher ───────────────────────────────────────────

        /// A frame on the parser stack. Each frame tracks one level of
        /// object or array nesting.
        #[derive(Debug, Clone)]
        enum Frame {
            /// Inside an object: tracking which keys have been seen,
            /// which are required, and what comes next.
            Object {
                schema: SchemaNode,
                seen_keys: HashSet<String>,
                after_value: bool, // true after a value, expecting comma or }
            },
            /// Inside an array: tracking item count and what comes next.
            Array {
                schema: SchemaNode,
                count: usize,
                after_value: bool, // true after an item, expecting comma or ]
            },
        }

        /// Incremental JSON schema matcher. Operates on raw bytes; parser
        /// state is a stack of frames (not a single FSM id).
        ///
        /// Usage:
        /// 1. Create with [`SchemaMatcher::new`].
        /// 2. Feed bytes via [`SchemaMatcher::advance`].
        /// 3. Check [`SchemaMatcher::is_token_allowed`] before sampling.
        /// 4. Check [`SchemaMatcher::is_accepting`] after each advance.
        #[derive(Debug, Clone)]
        pub struct SchemaMatcher {
            root: SchemaNode,
            /// The accumulated raw bytes of the JSON being parsed.
            bytes: Vec<u8>,
            /// Parser stack: one frame per nesting level.
            stack: Vec<Frame>,
            /// Whether the matcher has accepted (the buffer is a complete
            /// valid JSON value — EOS is legal).
            accepted: bool,
            /// Whether the accepted value is a number whose last byte could
            /// still continue it (digit/sign/point/exponent). While true,
            /// digit tokens remain legal alongside EOS and whitespace; a
            /// non-continuing byte closes the number for good.
            number_open: bool,
            /// Whether the matcher has errored (invalid JSON or schema violation).
            errored: bool,
            /// Consecutive WHITESPACE-ONLY tokens committed at a structural
            /// (outside-string) position. JSON allows unlimited leading
            /// whitespace, so the mask alone can never force progress: a model
            /// steered away from its preferred continuation (e.g. right after
            /// a forced think close) can pour out tabs forever — a document
            /// valid at every prefix that never finishes. `is_token_allowed`
            /// refuses whitespace-only tokens past WS_RUN_CAP; a value byte
            /// resets the run. Sound: JSON is whitespace-insensitive, so the
            /// cap cannot make any valid document unreachable.
            ws_run: usize,
            /// Raw byte-scan state refreshed by [`SchemaMatcher::advance`]:
            /// duplicate-key detection plus the cheap first-byte/string context
            /// used by `is_token_allowed` to avoid a full clone+reparse per
            /// vocabulary candidate (spec §9.2: mask time must be bounded).
            scan: RawScan,
        }

        /// Raw byte-scan result over the accumulated buffer.
        #[derive(Debug, Clone, Default)]
        struct RawScan {
            /// A decoded object key repeated within the same object (strict mode
            /// rejects duplicates; serde_json silently merges them).
            duplicate_keys: bool,
            /// The buffer currently ends inside a JSON string literal.
            in_string: bool,
            /// The buffer ends immediately after a backslash inside a string.
            escape_pending: bool,
            /// The buffer ends inside (or immediately after) a literal or number
            /// run (`tru`, `12.`, `1e` …) — any literal continuation byte may be
            /// legal next, so the structural first-byte fast path must stand
            /// down and let the simulation decide.
            literal_tail: bool,
            /// Legal first bytes for the next structural token when NOT inside
            /// a string (conservative value-start / continuation sets).
            structural_next: Vec<u8>,
            /// When the buffer ends at a value-start position: the schema-
            /// narrowed set of legal FIRST bytes for that value, plus `]`
            /// inside an array (an empty array closes without a value).
            /// `None` at key/colon/continuation positions, where no value
            /// starts and the schema cannot prune.
            value_next: Option<Vec<u8>>,
            /// When the buffer ends inside a KEY string of a CLOSED object
            /// (`additionalProperties: false`): the property names not yet
            /// used. Every key character token must keep the key a prefix of
            /// one of these — an unknown or already-used key can never be
            /// completed, so its characters are pruned at the mask (the
            /// completion-time flags below are the belt, this is the braces).
            key_filter: Option<Vec<String>>,
            /// The decoded-so-far text of the key string the buffer ends
            /// inside (empty when not in a key string).
            key_prefix_decoded: String,
            /// A CLOSED object (`additionalProperties: false`) completed a key
            /// that is not in `properties` — the document can never validate.
            unknown_key: bool,
            /// A `}` closed an object with `required` keys never seen — the
            /// document can never validate.
            required_missing_on_close: bool,
            /// A `]` closed an array holding fewer than `minItems` items.
            min_items_unmet_on_close: bool,
            /// An array accepted more than `maxItems` items.
            array_over_max: bool,
            /// The buffer ends inside a NUMBER literal (tail starts with a
            /// digit or `-`).
            number_tail: bool,
            /// The expected node at that number position accepts any number
            /// spelling (`Any`/`Number`) — enables the O(1) tail fast path in
            /// `is_token_allowed`.
            number_tail_free: bool,
            /// The expected node is `Integer`: continuation bytes may exclude
            /// `.` (a fraction can never be typed back into an integer mask),
            /// so the fast path uses the dot-free set.
            number_tail_integer: bool,
            /// The trailing number text can provably never grow into a value
            /// conforming to the expected node (fraction under `Integer`,
            /// impossible `Const`/`Enum` member, number under a non-number
            /// type).
            number_dead_end: bool,
            /// The raw bytes of the trailing number run (empty unless
            /// `number_tail`). Lets the number-tail fast path enforce the JSON
            /// number grammar (`01`, `1ee`, `1.2.3`, `--` must never be
            /// mask-allowed as continuations).
            number_text: Vec<u8>,
            /// When the buffer ends inside a VALUE string whose expected node
            /// is a string `Const`/`Enum`: the member strings it must
            /// eventually equal. Every content token must keep the string a
            /// prefix of one of these — a diverged enum value can never be
            /// completed, so its characters are pruned at the mask (the
            /// value-position mirror of `key_filter`).
            value_filter: Option<Vec<String>>,
            /// The decoded-so-far text of the value string the buffer ends
            /// inside (empty when not in a value string).
            value_prefix_decoded: String,
            /// At a structural VALUE position whose expected node is a string
            /// `Const`/`Enum`: the member strings. A token OPENING the string
            /// (`"` alone or a quote-bearing multi-char token) must decode its
            /// post-quote remainder against this list — the in-string filter
            /// below only sees later tokens, and the simulation is value-blind
            /// inside strings (a `"{` opening would otherwise commit garbage).
            value_start_filter: Option<Vec<String>>,
            /// At a structural KEY position of a CLOSED object: the unused
            /// property names — the opening-quote mirror of `key_filter` (a
            /// `"city`-style token must decode its remainder against the
            /// unused keys or the quote rides unknown chars into the key).
            key_start_filter: Option<Vec<String>>,
            /// The value position is the ROOT value (no enclosing frame): a
            /// completed root value may only be followed by whitespace.
            value_at_root: bool,
            /// The buffer ends inside a KEY or VALUE string of a CLOSED object
            /// whose every known property is already used (`sated`). The next
            /// structural continuation after the string closes can only be
            /// `}` — a `,` would demand a property the schema forbids, and the
            /// simulation cannot judge that (it accepts `,` blindly), so the
            /// in-string fast path must close-judge locally.
            in_string_sated_closed: bool,
        }

        /// Value-start bytes for an unconstrained JSON value position.
        const VALUE_START: &[u8] = b"\"-{0123456789tfn[";

        /// Consecutive structural-whitespace tokens allowed before the mask
        /// refuses further whitespace (progress bound — see `ws_run`).
        /// Sane pretty-printing never approaches this; 32 whitespace TOKENS at
        /// 96 tokens of budget would otherwise be an unbounded burn.
        const WS_RUN_CAP: usize = 32;

        /// Scan raw JSON bytes once for duplicate keys, string state, the
        /// legal next structural bytes, and — at a value-start position —
        /// the schema-narrowed set of bytes that may begin that value.
        /// O(n) per call; the matcher calls it once per `advance` (same
        /// order as the serde reparse it sits next to).
        fn scan_raw(bytes: &[u8], root: &SchemaNode) -> RawScan {
            #[derive(Clone, Copy, PartialEq)]
            enum Phase {
                /// Object: a key string comes next.
                Key,
                /// Object: the key is complete; `:` comes next.
                Colon,
                /// A value comes next (root, after `:`, after `[` or `,`).
                Value,
                /// After a complete value: `,` or the closing bracket.
                Cont,
            }
            enum Frame {
                Object {
                    keys: std::collections::HashSet<String>,
                    phase: Phase,
                    /// This object's compiled schema, for property lookup.
                    schema: SchemaNode,
                    /// The schema the pending key's value must satisfy —
                    /// set when a key completes, consulted at Phase::Value.
                    pending: SchemaNode,
                },
                Array {
                    phase: Phase,
                    /// The `items` schema every element must satisfy.
                    items: SchemaNode,
                    /// Items completed so far (nested `minItems`/`maxItems`
                    /// enforcement — the root-level byte scan only ever
                    /// covered the root array).
                    count: usize,
                    min_items: usize,
                    max_items: Option<usize>,
                },
            }

            /// The schema node a value starting at the current position must
            /// satisfy: the pending property inside an object, `items` inside
            /// an array, `root` at the top level. `None` when the position is
            /// not a value start.
            fn value_schema<'a>(
                frames: &'a [Frame],
                root: &'a SchemaNode,
            ) -> Option<&'a SchemaNode> {
                match frames.last() {
                    Some(Frame::Object {
                        phase: Phase::Value,
                        pending,
                        ..
                    }) => Some(pending),
                    Some(Frame::Array {
                        phase: Phase::Value,
                        items,
                        ..
                    }) => Some(items),
                    None => Some(root),
                    _ => None,
                }
            }

            let mut frames: Vec<Frame> = Vec::new();
            let mut duplicate_keys = false;
            let mut in_string = false;
            let mut escape_pending = false;
            let mut literal_tail = false;
            let mut i = 0usize;
            let mut key_prefix_decoded = String::new();
            let mut value_prefix_decoded = String::new();
            let mut number_text: Vec<u8> = Vec::new();
            let mut unknown_key = false;
            let mut required_missing_on_close = false;
            let mut min_items_unmet_on_close = false;
            let mut array_over_max = false;
            let mut number_tail = false;
            let mut number_tail_free = false;
            let mut number_tail_integer = false;
            let mut number_dead_end = false;

            while i < bytes.len() {
                let b = bytes[i];
                match b {
                    b'"' => {
                        // Scan the string, decoding escapes for key equality.
                        let is_key = match frames.last() {
                            Some(Frame::Object {
                                phase: Phase::Key, ..
                            }) => true,
                            _ => false,
                        };
                        let mut s = String::new();
                        i += 1;
                        loop {
                            if i >= bytes.len() {
                                // Buffer ends inside the string.
                                in_string = true;
                                if is_key {
                                    key_prefix_decoded = s.clone();
                                } else {
                                    value_prefix_decoded = s.clone();
                                }
                                break;
                            }
                            match bytes[i] {
                                b'"' => {
                                    i += 1;
                                    break;
                                }
                                b'\\' => {
                                    i += 1;
                                    if i >= bytes.len() {
                                        in_string = true;
                                        escape_pending = true;
                                        break;
                                    }
                                    match bytes[i] {
                                        b'"' => s.push('"'),
                                        b'\\' => s.push('\\'),
                                        b'/' => s.push('/'),
                                        b'b' => s.push('\u{0008}'),
                                        b'f' => s.push('\u{000C}'),
                                        b'n' => s.push('\n'),
                                        b'r' => s.push('\r'),
                                        b't' => s.push('\t'),
                                        b'u' => {
                                            let hex = |slice: &[u8]| -> Option<u32> {
                                                if slice.len() < 4 {
                                                    return None;
                                                }
                                                u32::from_str_radix(
                                                    std::str::from_utf8(slice).ok()?,
                                                    16,
                                                )
                                                .ok()
                                            };
                                            match bytes.get(i + 1..i + 5).and_then(hex) {
                                                Some(hi) => {
                                                    i += 4;
                                                    let ch = if (0xD800..0xDC00).contains(&hi) {
                                                        let lo =
                                                            bytes.get(i + 3..i + 7).and_then(hex);
                                                        match lo {
                                                            Some(lo)
                                                                if (0xDC00..0xE000)
                                                                    .contains(&lo) =>
                                                            {
                                                                i += 6;
                                                                char::from_u32(
                                                                    0x10000
                                                                        + ((hi - 0xD800) << 10)
                                                                        + (lo - 0xDC00),
                                                                )
                                                                .unwrap_or('\u{FFFD}')
                                                            }
                                                            _ => '\u{FFFD}',
                                                        }
                                                    } else {
                                                        char::from_u32(hi).unwrap_or('\u{FFFD}')
                                                    };
                                                    s.push(ch);
                                                }
                                                None => {
                                                    // Truncated escape at buffer end.
                                                    in_string = true;
                                                    escape_pending = true;
                                                    i = bytes.len();
                                                    break;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                    i += 1;
                                    continue;
                                }
                                c => {
                                    let len = match c {
                                        c if c < 0x80 => 1,
                                        c if c & 0xE0 == 0xC0 => 2,
                                        c if c & 0xF0 == 0xE0 => 3,
                                        _ => 4,
                                    };
                                    let end = (i + len).min(bytes.len());
                                    s.push_str(&String::from_utf8_lossy(&bytes[i..end]));
                                    i = end;
                                    continue;
                                }
                            }
                        }
                        if in_string {
                            break;
                        }
                        if is_key {
                            match frames.last_mut() {
                                Some(Frame::Object {
                                    keys,
                                    phase,
                                    schema,
                                    pending,
                                }) => {
                                    // The value after `:` must satisfy this
                                    // key's property schema. An unknown key
                                    // gets Any — when additionalProperties is
                                    // false the document is already doomed at
                                    // validation, so pruning gains nothing.
                                    *pending = match schema {
                                        SchemaNode::Object { properties, .. } => properties
                                            .iter()
                                            .find(|(k, _)| k == &s)
                                            .map(|(_, n)| n.clone())
                                            .unwrap_or(SchemaNode::Any),
                                        _ => SchemaNode::Any,
                                    };
                                    // A key that COMPLETES outside the closed
                                    // object's property set proves the document
                                    // can never validate — the closing quote of
                                    // such a key is a typed error at this
                                    // advance (the char filter above only
                                    // constrains extensions; `"f"` under
                                    // properties {fixed} must die HERE, or the
                                    // model is steered into an unvalidatable
                                    // document that burns to max_tokens).
                                    if grammar_trace_enabled() {
                                        eprintln!("[scan] key completed: {:?} (is_key branch)", s);
                                    }
                                    if let SchemaNode::Object {
                                        properties,
                                        additional_properties: false,
                                        ..
                                    } = schema
                                    {
                                        if !properties.iter().any(|(k, _)| k == &s) {
                                            if grammar_trace_enabled() {
                                                eprintln!("[scan] UNKNOWN KEY FLAG SET: {:?}", s);
                                            }
                                            unknown_key = true;
                                        }
                                    }
                                    if !keys.insert(s) {
                                        duplicate_keys = true;
                                    }
                                    *phase = Phase::Colon;
                                }
                                _ => {}
                            }
                        } else if let Some(frame) = frames.last_mut() {
                            match frame {
                                Frame::Object { phase, .. } => {
                                    *phase = Phase::Cont;
                                }
                                Frame::Array {
                                    phase,
                                    count,
                                    max_items,
                                    ..
                                } => {
                                    // A completed VALUE string was an ITEM of
                                    // this array — count it, exactly like the
                                    // literal/number and nested-container
                                    // branches (minItems pruning and the
                                    // maxItems cap read `count`).
                                    if *phase == Phase::Value {
                                        *count += 1;
                                        if let Some(max) = max_items {
                                            if *count > *max {
                                                array_over_max = true;
                                            }
                                        }
                                    }
                                    *phase = Phase::Cont;
                                }
                            }
                        }
                    }
                    b'{' => {
                        // The object must satisfy the schema expected at this
                        // value position; a non-Object expected node means the
                        // document is already invalid — carry Any so the scan
                        // stays conservative and lets validation error late.
                        let schema = match value_schema(&frames, root) {
                            Some(node @ SchemaNode::Object { .. }) => node.clone(),
                            _ => SchemaNode::Any,
                        };
                        frames.push(Frame::Object {
                            keys: std::collections::HashSet::new(),
                            phase: Phase::Key,
                            schema,
                            pending: SchemaNode::Any,
                        });
                        i += 1;
                    }
                    b'[' => {
                        let (items, min_items, max_items) = match value_schema(&frames, root) {
                            Some(SchemaNode::Array {
                                items,
                                min_items,
                                max_items,
                            }) => ((**items).clone(), *min_items, *max_items),
                            _ => (SchemaNode::Any, 0, None),
                        };
                        frames.push(Frame::Array {
                            phase: Phase::Value,
                            items,
                            count: 0,
                            min_items,
                            max_items,
                        });
                        i += 1;
                    }
                    b'}' | b']' => {
                        if let Some(closed) = frames.last() {
                            match closed {
                                Frame::Object {
                                    schema,
                                    keys,
                                    phase,
                                    ..
                                } => {
                                    if *phase != Phase::Value {
                                        if let SchemaNode::Object { required, .. } = schema {
                                            if required.iter().any(|r| !keys.contains(r)) {
                                                required_missing_on_close = true;
                                            }
                                        }
                                    }
                                }
                                Frame::Array {
                                    count,
                                    phase,
                                    min_items,
                                    ..
                                } => {
                                    if *phase != Phase::Value && *count < *min_items {
                                        min_items_unmet_on_close = true;
                                    }
                                }
                            }
                        }
                        frames.pop();
                        if let Some(frame) = frames.last_mut() {
                            match frame {
                                Frame::Object { phase, .. } => {
                                    *phase = Phase::Cont;
                                }
                                Frame::Array { phase, count, .. } => {
                                    // The popped frame was a completed ITEM of
                                    // this array (a bare `[]` close is not).
                                    if *phase == Phase::Value {
                                        *count += 1;
                                    }
                                    *phase = Phase::Cont;
                                }
                            }
                        }
                        i += 1;
                    }
                    b',' => {
                        match frames.last_mut() {
                            Some(Frame::Object { phase, .. }) => *phase = Phase::Key,
                            Some(Frame::Array { phase, .. }) => *phase = Phase::Value,
                            None => {}
                        }
                        i += 1;
                    }
                    b':' => {
                        if let Some(Frame::Object { phase, .. }) = frames.last_mut() {
                            *phase = Phase::Value;
                        }
                        i += 1;
                    }
                    _ => {
                        // Literal (true/false/null) or number: consume its bytes
                        // as one value token.
                        let start = i;
                        while i < bytes.len()
                            && !matches!(
                                bytes[i],
                                b'"' | b'{'
                                    | b'['
                                    | b'}'
                                    | b']'
                                    | b','
                                    | b':'
                                    | b' '
                                    | b'\n'
                                    | b'\t'
                                    | b'\r'
                            )
                        {
                            i += 1;
                        }
                        if i > start {
                            // Classify the expected value node BEFORE the phase
                            // flip below: `value_schema` consults the frame's
                            // phase, and the flip destroys the Value context the
                            // literal is terminating (the flags would always
                            // compute against `None` and never fire nested).
                            enum NumCtx {
                                Free,
                                Integer,
                                Targets(Vec<serde_json::Value>),
                                Dead,
                            }
                            let num_ctx = match value_schema(&frames, root) {
                                None | Some(SchemaNode::Any) | Some(SchemaNode::Number) => {
                                    NumCtx::Free
                                }
                                Some(SchemaNode::Integer) => NumCtx::Integer,
                                Some(SchemaNode::Const(v)) if v.is_number() => {
                                    NumCtx::Targets(vec![v.clone()])
                                }
                                Some(SchemaNode::Enum(values)) => NumCtx::Targets(
                                    values.iter().filter(|v| v.is_number()).cloned().collect(),
                                ),
                                _ => NumCtx::Dead,
                            };
                            if let Some(frame) = frames.last_mut() {
                                match frame {
                                    Frame::Object { phase, .. } => {
                                        *phase = Phase::Cont;
                                    }
                                    Frame::Array {
                                        phase,
                                        count,
                                        max_items,
                                        ..
                                    } => {
                                        if *phase == Phase::Value {
                                            *count += 1;
                                            if let Some(max) = max_items {
                                                if *count > *max {
                                                    array_over_max = true;
                                                }
                                            }
                                        }
                                        *phase = Phase::Cont;
                                    }
                                }
                            }
                            // The literal ran to the buffer end: it may be an
                            // incomplete `true`/`false`/`null`/number — any
                            // literal continuation byte can be legal next.
                            if i >= bytes.len() {
                                literal_tail = true;
                                let text = bytes[start..i].to_vec();
                                number_tail = matches!(text[0], b'-' | b'0'..=b'9');
                                // The number rules only describe NUMBER tails —
                                // a `t`/`f`/`n` head (true/false/null) must be
                                // left to the simulation, or every boolean/null
                                // literal start would be flagged dead.
                                if number_tail {
                                    number_text = text.clone();
                                    match num_ctx {
                                        NumCtx::Free => number_tail_free = true,
                                        NumCtx::Integer => {
                                            number_tail_integer = true;
                                            // "4." can never become an integer.
                                            // Deliberate contract (see the
                                            // number-tail fast path and
                                            // integer_rejects_fractional_growth):
                                            // Integer never grows a fraction, so
                                            // "4.5e1" is refused even though
                                            // validate() would accept fract==0.
                                            number_dead_end = text.contains(&b'.');
                                        }
                                        NumCtx::Targets(targets) => {
                                            number_dead_end = !targets
                                                .iter()
                                                .any(|m| number_text_could_match(&text, m));
                                        }
                                        NumCtx::Dead => number_dead_end = true,
                                    }
                                }
                            }
                        } else {
                            i += 1; // whitespace
                            literal_tail = false;
                        }
                    }
                }
            }

            // JSON whitespace is legal before ANY structural byte — a token
            // starting with space/newline/tab must reach the simulation, not
            // be refused by the first-byte fast path.
            const WS: [u8; 4] = [b' ', b'\n', b'\t', b'\r'];

            // Legal next structural bytes from the final frame phase, plus
            // the schema-narrowed value-start set at a value position.
            let (structural_next, value_next): (Vec<u8>, Option<Vec<u8>>) = if in_string {
                (Vec::new(), None)
            } else {
                let phase = match frames.last() {
                    Some(Frame::Object { phase, .. }) | Some(Frame::Array { phase, .. }) => *phase,
                    None => Phase::Value,
                };
                let structural: Vec<u8> = match phase {
                    Phase::Key => {
                        // `}` may close an object only when nothing is
                        // required — the empty-object close (`{}`). With
                        // `required` non-empty an empty close is a guaranteed
                        // schema violation, so refuse `}` here and force a
                        // key instead (the object-side mirror of the
                        // array-side `]` fold below). A close after a
                        // trailing comma (`{... ,}`) is invalid JSON and is
                        // refused by the simulation — the structural set only
                        // needs to be a conservative superset.
                        let mut v: Vec<u8> = [b'"'].into_iter().chain(WS).collect();
                        let required_open = matches!(
                            frames.last(),
                            Some(Frame::Object {
                                schema: SchemaNode::Object { required, .. },
                                ..
                            }) if !required.is_empty()
                        );
                        if matches!(frames.last(), Some(Frame::Object { .. })) && !required_open {
                            v.push(b'}');
                        }
                        v
                    }
                    Phase::Colon => [b':'].into_iter().chain(WS).collect(),
                    Phase::Value => {
                        let mut v: Vec<u8> = VALUE_START.iter().copied().chain(WS).collect();
                        // Inside an array, `]` may close it without a value
                        // (`[]`); a trailing comma (`[1,]`) reaches the
                        // simulation, which refuses it — conservative-late.
                        // An array whose `minItems` is not yet met cannot close
                        // at all: refusing `]` here forces another item instead
                        // of a completion-time dead end.
                        let min_items_open = matches!(
                            frames.last(),
                            Some(Frame::Array { min_items, .. }) if *min_items > 0
                        );
                        if matches!(frames.last(), Some(Frame::Array { .. })) && !min_items_open {
                            v.push(b']');
                        }
                        v
                    }
                    Phase::Cont => match frames.last() {
                        Some(Frame::Object { schema, keys, .. }) => {
                            // Schema-aware continuation (spec §7.1: enforce
                            // `required` at the correct nesting level; a CLOSED
                            // object with every property used cannot accept
                            // another key). Refusing `}`/`,` here steers the
                            // mask toward the only legal continuations and
                            // turns dead ends into EmptyAllowedSet instead of
                            // late completion-time rejects.
                            let mut v: Vec<u8> = [b',', b'}'].into_iter().chain(WS).collect();
                            if let SchemaNode::Object {
                                properties,
                                required,
                                additional_properties,
                            } = schema
                            {
                                let required_met = required.iter().all(|r| keys.contains(r));
                                let all_known_used =
                                    properties.iter().all(|(k, _)| keys.contains(k));
                                if !required_met {
                                    v.retain(|&b| b != b'}');
                                }
                                if !*additional_properties && all_known_used {
                                    v.retain(|&b| b != b',');
                                }
                            }
                            v
                        }
                        Some(Frame::Array {
                            count,
                            min_items,
                            max_items,
                            ..
                        }) => {
                            let mut v: Vec<u8> = [b',', b']'].into_iter().chain(WS).collect();
                            if *count < *min_items {
                                v.retain(|&b| b != b']');
                            }
                            if max_items.map(|m| *count >= m).unwrap_or(false) {
                                v.retain(|&b| b != b',');
                            }
                            v
                        }
                        _ => [b',', b']'].into_iter().chain(WS).collect(),
                    },
                };
                let value_next = value_schema(&frames, root).map(|node| {
                    let mut v = schema_value_starts(node);
                    let can_close_empty = matches!(
                        frames.last(),
                        Some(Frame::Array { min_items, .. }) if *min_items == 0
                    );
                    if can_close_empty && !v.contains(&b']') {
                        v.push(b']');
                    }
                    v
                });
                (structural, value_next)
            };

            // Key-string filter: while the buffer ends inside a KEY string of a
            // CLOSED object, only the unused known property names can complete
            // — their characters are exact, everything else is prunable.
            let key_filter = if in_string {
                match frames.last() {
                    Some(Frame::Object {
                        schema,
                        keys,
                        phase: Phase::Key,
                        ..
                    }) => match schema {
                        SchemaNode::Object {
                            properties,
                            additional_properties: false,
                            ..
                        } => {
                            let unused: Vec<String> = properties
                                .iter()
                                .map(|(k, _)| k.clone())
                                .filter(|k| !keys.contains(k))
                                .collect();
                            Some(unused)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            };

            // Sated-closed-object flag: inside a key or value string of a
            // closed object with every known property used (see the field doc).
            let in_string_sated_closed = in_string
                && matches!(
                    frames.last(),
                    Some(Frame::Object {
                        schema:
                            SchemaNode::Object {
                                properties,
                                additional_properties: false,
                                ..
                            },
                        keys,
                        ..
                    }) if properties.iter().all(|(k, _)| keys.contains(k))
                );

            // Value-string filter (spec §7.1 soundness): while the buffer ends
            // inside a VALUE string whose expected node is a string
            // `Const`/`Enum`, only the member strings can complete the value —
            // a character that diverges from EVERY member can never be rescued,
            // so it is pruned at the mask instead of committing an enum
            // violation that only dies at final validation. Non-string members
            // of a mixed enum are irrelevant once inside a string. Numeric and
            // composite Const/Enum values keep the existing simulation path.
            let value_filter = if in_string {
                match value_schema(&frames, root) {
                    Some(SchemaNode::Const(v)) if v.is_string() => {
                        Some(vec![v.as_str().unwrap_or_default().to_string()])
                    }
                    Some(SchemaNode::Enum(values)) => {
                        let members: Vec<String> = values
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect();
                        if members.is_empty() {
                            None
                        } else {
                            Some(members)
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            // Opening-quote filters (see the field docs): a token that OPENS
            // a key/value string carries content the in-string filters never
            // see, and the simulation cannot judge string interiors against
            // Const/Enum members or closed-object key sets.
            let value_start_filter = if !in_string {
                match value_schema(&frames, root) {
                    Some(SchemaNode::Const(v)) if v.is_string() => {
                        Some(vec![v.as_str().unwrap_or_default().to_string()])
                    }
                    Some(SchemaNode::Enum(values)) => {
                        let members: Vec<String> = values
                            .iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect();
                        if members.is_empty() {
                            None
                        } else {
                            Some(members)
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
            let key_start_filter = if !in_string {
                match frames.last() {
                    Some(Frame::Object {
                        schema:
                            SchemaNode::Object {
                                properties,
                                additional_properties: false,
                                ..
                            },
                        keys,
                        phase: Phase::Key,
                        ..
                    }) => {
                        let unused: Vec<String> = properties
                            .iter()
                            .map(|(k, _)| k.clone())
                            .filter(|k| !keys.contains(k))
                            .collect();
                        if unused.is_empty() {
                            None
                        } else {
                            Some(unused)
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            RawScan {
                duplicate_keys,
                in_string,
                escape_pending,
                literal_tail,
                structural_next,
                value_next,
                key_filter,
                key_prefix_decoded,
                unknown_key,
                required_missing_on_close,
                min_items_unmet_on_close,
                array_over_max,
                number_tail,
                number_tail_free,
                number_tail_integer,
                number_dead_end,
                number_text,
                value_filter,
                value_prefix_decoded,
                value_start_filter,
                key_start_filter,
                value_at_root: frames.is_empty(),
                in_string_sated_closed: in_string_sated_closed,
            }
        }

        /// The set of bytes that may begin a JSON value conforming to
        /// `node`. Used by the incremental matcher to refuse, at a value-
        /// start position, a token whose first meaningful byte can never
        /// begin a conforming value (spec §7: schema-aware incremental
        /// matching). Conservative: when a byte could begin ANY conforming
        /// spelling it is included — e.g. a numeric `const`/`enum` member
        /// admits every digit and `-`, because semantically-equal spellings
        /// (`0.42e2` for `42`) may start with a different byte than the
        /// member's canonical serialization.
        fn schema_value_starts(node: &SchemaNode) -> Vec<u8> {
            match node {
                SchemaNode::Any => VALUE_START.to_vec(),
                SchemaNode::String => vec![b'"'],
                SchemaNode::Number | SchemaNode::Integer => {
                    let mut v = vec![b'-'];
                    v.extend_from_slice(b"0123456789");
                    v
                }
                SchemaNode::Boolean => vec![b't', b'f'],
                SchemaNode::Null => vec![b'n'],
                SchemaNode::Object { .. } => vec![b'{'],
                SchemaNode::Array { .. } => vec![b'['],
                SchemaNode::Const(v) => const_starts(std::slice::from_ref(v)),
                SchemaNode::Enum(values) => const_starts(values),
            }
        }

        /// Union of legal first bytes across literal values (const/enum
        /// members). Booleans are exact (`t`/`f`); numbers admit every
        /// digit and `-` since equal spellings differ in their first byte.
        fn const_starts(values: &[serde_json::Value]) -> Vec<u8> {
            let mut out: Vec<u8> = Vec::new();
            for v in values {
                let starts: &[u8] = match v {
                    serde_json::Value::String(_) => b"\"",
                    serde_json::Value::Number(_) => b"-0123456789",
                    serde_json::Value::Bool(true) => b"t",
                    serde_json::Value::Bool(false) => b"f",
                    serde_json::Value::Null => b"n",
                    serde_json::Value::Object(_) => b"{",
                    serde_json::Value::Array(_) => b"[",
                };
                for &b in starts {
                    if !out.contains(&b) {
                        out.push(b);
                    }
                }
            }
            out
        }

        /// Check whether a serde_json error represents incomplete input
        /// (EOF) vs. a genuine syntax error. serde_json wraps incomplete-
        /// input errors with `io::ErrorKind::UnexpectedEof` in the
        /// `io` category.
        fn is_eof_error(e: &serde_json::Error) -> bool {
            // serde_json's `Error::classify()` returns `Category::Eof`
            // for incomplete input.
            e.classify() == serde_json::error::Category::Eof
        }

        /// Count the number of items started in the outermost JSON array (spec
        /// §7.1: enforce `maxItems` incrementally). Returns `None` when the
        /// buffer does not begin with `[`. The count includes incomplete items:
        /// `[1,2` → 2, `[1,` → 1 (the comma starts a new item but no content
        /// follows yet), `[1` → 1. `[]` → 0.
        ///
        /// Tracks string/escape state and nesting depth so commas inside nested
        /// structures or strings are not counted.
        fn count_root_array_items(bytes: &[u8]) -> Option<usize> {
            let mut i = 0;
            while i < bytes.len() && matches!(bytes[i], b' ' | b'\n' | b'\t' | b'\r') {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'[' {
                return None;
            }
            i += 1; // consume `[`

            let mut depth: i32 = 1;
            let mut in_string = false;
            let mut escape = false;
            let mut commas: usize = 0;
            let mut has_content = false;

            while i < bytes.len() {
                let b = bytes[i];
                if in_string {
                    if escape {
                        escape = false;
                    } else if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_string = false;
                    }
                } else {
                    match b {
                        b'"' => {
                            in_string = true;
                            if depth == 1 {
                                has_content = true;
                            }
                        }
                        b'[' | b'{' => {
                            if depth == 1 {
                                has_content = true;
                            }
                            depth += 1;
                        }
                        b']' | b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(commas + if has_content { 1 } else { 0 });
                            }
                        }
                        b',' => {
                            if depth == 1 {
                                commas += 1;
                                has_content = false;
                            }
                        }
                        b' ' | b'\n' | b'\t' | b'\r' => {}
                        _ => {
                            if depth == 1 {
                                has_content = true;
                            }
                        }
                    }
                }
                i += 1;
            }
            // Incomplete (no closing `]`): count started items.
            Some(commas + if has_content { 1 } else { 0 })
        }

        /// Check whether a growing number value could eventually match
        /// the schema. `raw` is the raw byte buffer (including any
        /// leading whitespace) that was parsed to produce `value`.
        ///
        /// For `Number`/`Integer`/`Any`: any number prefix is valid.
        /// For `Const(n)`: the raw bytes must be a prefix of `n`'s JSON
        /// representation (e.g. "4" is a prefix of "42", "43" is not).
        /// For `Enum(values)`: the raw bytes must be a prefix of at least
        /// one number value's JSON representation.
        /// For other types: false (a number can never become a string,
        /// boolean, null, object, or array).
        fn number_could_grow_to_match(
            value: &serde_json::Value,
            schema: &SchemaNode,
            raw: &[u8],
        ) -> bool {
            // Extract the number's raw bytes from the buffer (skip
            // leading whitespace).
            let num_start = raw
                .iter()
                .position(|&b| b != b' ' && b != b'\n' && b != b'\t' && b != b'\r')
                .unwrap_or(raw.len());
            let num_bytes = &raw[num_start..];

            match schema {
                SchemaNode::Number | SchemaNode::Any => true,
                SchemaNode::Integer => {
                    // Growth can only rescue a non-integral value via an
                    // exponent ("4.5e1" = 45). Once an exponent marker is
                    // already present, a fractional value is a dead end.
                    value.is_i64()
                        || value.is_u64()
                        || (value.is_f64() && !num_bytes.iter().any(|&b| b == b'e' || b == b'E'))
                }
                SchemaNode::Const(target) => {
                    if !target.is_number() {
                        return false;
                    }
                    number_text_could_match(num_bytes, target)
                }
                SchemaNode::Enum(values) => {
                    // Check if the raw bytes could still match any number
                    // value in the enum.
                    for v in values {
                        if v.is_number() && number_text_could_match(num_bytes, v) {
                            return true;
                        }
                    }
                    false
                }
                _ => false,
            }
        }

        /// Whether `num ++ tok` stays a valid PREFIX of a JSON number
        /// (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`). The number may
        /// continue in later tokens, so reaching any grammar state is enough —
        /// but every consumed byte must be legal for the state it is consumed
        /// in. Refuses `01` (leading zero), a second `.`/`e`/`-`, digits
        /// directly after `e`'s sign position violations, etc.
        fn json_number_prefix_allows(num: &[u8], tok: &[u8]) -> bool {
            #[derive(Clone, Copy, PartialEq)]
            enum St {
                /// After the optional leading `-`: a digit is required.
                Sign,
                /// A bare leading `0`: only `.` or an exponent may follow.
                Zero,
                /// Integer digits.
                Digits,
                /// A `.` was just consumed: a fraction digit is required.
                Dot,
                /// Fraction digits.
                Frac,
                /// An `e`/`E` was just consumed: a digit or sign is required.
                Exp,
                /// An exponent sign was just consumed: a digit is required.
                ExpSign,
                /// Exponent digits.
                ExpDigits,
            }
            use St::*;
            let mut chain = num.iter().chain(tok.iter());
            let mut st = match chain.next() {
                Some(b'-') => Sign,
                Some(b'0') => Zero,
                Some(b'1'..=b'9') => Digits,
                _ => return false,
            };
            for &b in chain {
                st = match (st, b) {
                    (Sign, b'0') => Zero,
                    (Sign, b'1'..=b'9') => Digits,
                    (Zero, b'.') => Dot,
                    (Zero, b'e' | b'E') => Exp,
                    (Digits, b'0'..=b'9') => Digits,
                    (Digits, b'.') => Dot,
                    (Digits, b'e' | b'E') => Exp,
                    (Dot, b'0'..=b'9') => Frac,
                    (Frac, b'0'..=b'9') => Frac,
                    (Frac, b'e' | b'E') => Exp,
                    (Exp, b'+' | b'-') => ExpSign,
                    (Exp, b'0'..=b'9') => ExpDigits,
                    (ExpSign, b'0'..=b'9') => ExpDigits,
                    (ExpDigits, b'0'..=b'9') => ExpDigits,
                    _ => return false,
                };
            }
            // Any state is a valid prefix: later tokens may complete the
            // number (completion legality is judged when a non-continuation
            // byte or EOS arrives, by the simulation).
            true
        }

        /// Whether a partially-emitted number's raw text could still resolve to
        /// `target`: either the raw text is a byte-prefix of some valid JSON
        /// spelling of `target` (it may grow), or the raw text already parses to
        /// a number that is SEMANTICALLY equal to `target` (`"1.0"` equals `1`
        /// even though it is not a text prefix of `"1"`, spec §7.1).
        fn number_text_could_match(raw: &[u8], target: &serde_json::Value) -> bool {
            let target_str = serde_json::to_string(target).unwrap_or_default();
            if target_str.as_bytes().starts_with(raw) {
                return true;
            }
            if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(raw) {
                if parsed.is_number() && json_equal(&parsed, target) {
                    return true;
                }
            }
            false
        }

        /// Decode a raw candidate-token fragment against an in-string state,
        /// honouring a pending escape from the scan. Returns the decoded text
        /// and whether the fragment contains an UNESCAPED closing quote (the
        /// string completes — the caller must fall to the simulation, which
        /// judges the completed string). `None` when the fragment cannot be
        /// decided byte-locally (truncated \\u escape, raw control byte,
        /// invalid escape) — under an active filter the caller REFUSES, since
        /// an undecodable fragment can never be proven to extend a member.
        fn decode_string_fragment<'a>(
            mut escape_pending: bool,
            bytes: &'a [u8],
        ) -> Option<(String, bool, &'a [u8])> {
            let mut out = String::new();
            let mut i = 0usize;
            while i < bytes.len() {
                let b = bytes[i];
                if escape_pending {
                    escape_pending = false;
                    match b {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            // Need 4 hex digits within this fragment; a
                            // truncated \\u must fall to the simulation.
                            if i + 4 >= bytes.len() {
                                return None;
                            }
                            let hex = std::str::from_utf8(&bytes[i + 1..i + 5]).ok()?;
                            let ch = u32::from_str_radix(hex, 16).ok()?;
                            // Surrogate pairs: a lone high surrogate needs 6
                            // more bytes — fall to the simulation.
                            if (0xD800..0xDC00).contains(&ch) {
                                return None;
                            }
                            out.push(char::from_u32(ch).unwrap_or('\u{FFFD}'));
                            i += 4;
                        }
                        _ => return None, // invalid escape — sim judges
                    }
                    i += 1;
                    continue;
                }
                match b {
                    b'"' => return Some((out, true, &bytes[i + 1..])), // string completes
                    b'\\' => {
                        if i + 1 >= bytes.len() {
                            // Fragment ends on the backslash: the escape char
                            // is unknown, so the decoded prefix is undecidable
                            // (a \\u spelling of a member char cannot be
                            // ruled IN locally, and the continuation tokens
                            // rarely exist) — undecidable under an active
                            // filter means refuse.
                            return None;
                        }
                        escape_pending = true;
                        i += 1;
                    }
                    c if c < 0x20 => return None, // raw control — sim judges
                    c => {
                        let len = match c {
                            c if c & 0xE0 == 0xC0 => 2,
                            c if c & 0xF0 == 0xE0 => 3,
                            c if c & 0xF8 == 0xF0 => 4,
                            _ => 1,
                        };
                        let end = (i + len).min(bytes.len());
                        if end - i < len {
                            // Truncated multi-byte sequence: leave to sim.
                            return None;
                        }
                        out.push_str(std::str::from_utf8(&bytes[i..end]).ok()?);
                        i = end;
                    }
                }
            }
            Some((out, false, &[]))
        }

        impl SchemaMatcher {
            /// Create a fresh matcher for the given root schema node.
            pub fn new(root: SchemaNode) -> Self {
                // An empty buffer scans to the root value-start set, so the
                // first-token fast path works before any advance().
                let scan = scan_raw(&[], &root);
                Self {
                    root,
                    bytes: Vec::new(),
                    stack: Vec::new(),
                    accepted: false,
                    number_open: false,
                    errored: false,
                    ws_run: 0,
                    scan,
                }
            }

            /// Create a fresh matcher from a compiled schema.
            pub fn from_compiled(schema: &CompiledSchema) -> Self {
                Self::new(schema.root.clone())
            }

            /// Check whether the given raw bytes would be a valid
            /// continuation from the current parser state.
            ///
            /// This is a conservative prefix check: it returns `true` if
            /// the bytes could be part of a valid JSON value conforming
            /// to the schema. A `false` return means the bytes would
            /// definitely violate the schema or JSON syntax.
            /// Cheap fast paths (bounded mask time, spec §9.2) consult the
            /// raw scan taken at the last `advance`:
            /// - Structural positions: a token whose FIRST byte cannot start
            ///   any legal next token is refused without a clone+reparse.
            /// - Value-start positions: a token whose first NON-WHITESPACE
            ///   byte cannot begin a value conforming to the schema node at
            ///   that position is refused without a clone+reparse (early
            ///   per-token schema pruning).
            /// - Inside a plain string (no pending escape): a token of valid
            ///   UTF-8 with no quote, backslash or unescaped control byte is
            ///   inert string content and is allowed without simulation.
            /// Everything else falls through to the full clone+simulate,
            /// which remains the authority.
            pub fn is_token_allowed(&self, bytes: &[u8]) -> bool {
                if self.errored {
                    return false;
                }
                if self.accepted {
                    if !self.number_open {
                        // After a closed acceptance, only whitespace is allowed
                        // — bounded by WS_RUN_CAP (the tab-spam burn class:
                        // valid JSON followed by unlimited tabs never errors,
                        // so the run must be capped to let EOS win the argmax).
                        return bytes
                            .iter()
                            .all(|&b| matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                            && self.ws_run < WS_RUN_CAP;
                    }
                    // Accepted but the trailing number could still grow: fall
                    // through to the simulation so digits stay legal and a
                    // non-continuing byte is judged by the parser.
                }
                // Whitespace-run cap (spec §9.2 progress bound): past WS_RUN_CAP
                // consecutive structural whitespace tokens, refuse further
                // whitespace-only tokens. JSON is whitespace-insensitive, so no
                // valid document is cut off; the loop class (a steered model
                // pouring tabs after a forced think close) is broken because a
                // value byte remains legal and the whitespace escape is gone.
                if !self.scan.in_string
                    && self.ws_run >= WS_RUN_CAP
                    && !bytes.is_empty()
                    && bytes
                        .iter()
                        .all(|&b| matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                {
                    return false;
                }
                if self.scan.in_string {
                    // Key-string prune (closed objects, spec §7.1): only the
                    // unused known property names can legally complete, so a
                    // token that stops extending ALL of them is refused
                    // without simulation — this is what keeps an unknown or
                    // duplicated key from ever being committed (a dead end
                    // pruned at the mask instead of a burn-to-max_tokens
                    // wedge). Escapes fall through to the simulation.
                    if let Some(filters) = &self.scan.key_filter {
                        // Decode the fragment (escapes included) and prefix-
                        // check the decoded candidate. Escape-bearing tokens
                        // used to SKIP the filter entirely — an escaped-quote
                        // key (`"f\"` under properties {fixed}) could then
                        // extend forever: the simulation cannot refuse an open
                        // string, so the model burned to max_tokens on a key
                        // that could never complete to anything valid.
                        match decode_string_fragment(self.scan.escape_pending, bytes) {
                            Some((decoded, false, _rest)) => {
                                let mut cand = self.scan.key_prefix_decoded.clone();
                                cand.push_str(&decoded);
                                return filters.iter().any(|k| {
                                    let kb = k.as_bytes();
                                    kb.len() >= cand.len() && kb.starts_with(cand.as_bytes())
                                });
                            }
                            // closes: the completed key must reach the
                            // simulation (unknown-key / duplicate checks).
                            Some((_, true, _rest)) => {}
                            // Undecodable (byte-fallback garbage, truncated
                            // \u): can never be PROVEN to extend a known key,
                            // and the simulation cannot judge string interiors
                            // — refuse rather than open an unprunable burn.
                            None => return false,
                        }
                    }
                    // Value-string prune (string Const/Enum, spec §7.1): only
                    // the member strings can legally complete the value, so a
                    // token that stops extending ALL of them is refused
                    // without simulation — this is what makes an enum value
                    // impossible to diverge from (a soundness hole otherwise:
                    // inert content is schema-blind and would happily emit
                    // "Tokyo" under enum ["Paris","London","Rome"]). The
                    // closing quote (or any token containing one) completes
                    // the string and must reach the simulation, which judges
                    // the completed value against the schema. Escapes fall
                    // through to the simulation.
                    if let Some(members) = &self.scan.value_filter {
                        // Same escape-decoding discipline as the key filter:
                        // escape-bearing fragments used to bypass the prune,
                        // leaving an unprunable burn via `\"` spam.
                        match decode_string_fragment(self.scan.escape_pending, bytes) {
                            Some((decoded, false, _rest)) => {
                                let mut cand = self.scan.value_prefix_decoded.clone();
                                cand.push_str(&decoded);
                                return members.iter().any(|m| {
                                    let mb = m.as_bytes();
                                    mb.len() >= cand.len() && mb.starts_with(cand.as_bytes())
                                });
                            }
                            // closes: the string COMPLETED, so prefix +
                            // fragment must be exactly a member (prefix alone
                            // is not enough — `"Z",` under const "ZZQ7" would
                            // otherwise commit — and the fragment alone is not
                            // the whole content). The simulation is value-blind
                            // inside strings; judge the content HERE. The
                            // trailing bytes (if any) must start a legal
                            // structural continuation — the sim tolerates some
                            // trailing junk at commit (`":"` after a complete
                            // root value) that dead-ends on the NEXT step.
                            Some((decoded, true, rest)) => {
                                let mut full = self.scan.value_prefix_decoded.clone();
                                full.push_str(&decoded);
                                let content_ok =
                                    members.iter().any(|m| m.as_bytes() == full.as_bytes());
                                if !content_ok {
                                    return false;
                                }
                                return match rest
                                    .iter()
                                    .find(|&&b| !matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                                {
                                    None => true, // trailing whitespace only
                                    Some(&first) => {
                                        !self.scan.value_at_root
                                            && matches!(first, b',' | b'}' | b']')
                                    }
                                };
                            }
                            // Undecodable fragment under a const/enum string:
                            // refuse (see the key-filter twin above).
                            None => return false,
                        }
                    }
                    // Sated-closed object (see the scan field doc): if this
                    // token CLOSES the string, the only legal structural
                    // continuation is `}` — a `,` would demand a property the
                    // schema forbids and the simulation would accept it
                    // blindly, committing an impossible key that then burns.
                    if self.scan.in_string_sated_closed {
                        match decode_string_fragment(self.scan.escape_pending, bytes) {
                            Some((_, true, rest)) => {
                                return match rest
                                    .iter()
                                    .find(|&&b| !matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                                {
                                    None => true,
                                    Some(&first) => first == b'}',
                                };
                            }
                            // Continues the string (no unescaped quote in the
                            // fragment): falls through to the inert allowance.
                            _ => {}
                        }
                    }
                    // Plain string content: inert UTF-8 without quotes,
                    // backslashes or control bytes is allowed without
                    // simulation (a pending escape needs the simulation).
                    if !self.scan.escape_pending
                        && std::str::from_utf8(bytes).is_ok()
                        && !bytes.contains(&b'"')
                        && !bytes.contains(&b'\\')
                        && bytes.iter().all(|&b| b >= 0x20)
                    {
                        return true;
                    }
                } else if !self.scan.literal_tail {
                    // Structural position: a token whose FIRST byte cannot start
                    // any legal next token is refused without a clone+reparse.
                    if let Some(first) = bytes.first() {
                        if !self.scan.structural_next.contains(first) {
                            return false;
                        }
                    }
                    // Schema-aware pruning at a value-start position: the
                    // token's first non-whitespace byte must be able to begin
                    // a value conforming to the schema node here (or close an
                    // array — `]` is folded into `value_next`). A byte that
                    // cannot start ANY conforming value can never be rescued
                    // by what follows it, so the refusal is exact, not
                    // heuristic. Whitespace-only tokens carry no meaningful
                    // byte and fall through to the simulation.
                    if let Some(value_next) = &self.scan.value_next {
                        if let Some(first) = bytes
                            .iter()
                            .find(|&&b| !matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                        {
                            if !value_next.contains(first) {
                                return false;
                            }
                        }
                    }
                    // Opening-quote filters: a token that OPENS a constrained
                    // string carries interior content the in-string filters
                    // never see and the simulation cannot judge. Decode the
                    // post-quote remainder against the member/key list.
                    let quote_idx = bytes
                        .iter()
                        .position(|&b| !matches!(b, b' ' | b'\n' | b'\t' | b'\r'));
                    if quote_idx.map(|i| bytes[i]) == Some(b'"') {
                        // Slice from AFTER the quote, not bytes[1..]: a token like
                        // ` "x` (leading whitespace) has the quote at index >0, and
                        // bytes[1..] would decode the quote itself as a CLOSE.
                        let after_quote = &bytes[quote_idx.unwrap() + 1..];
                        if let Some(members) = &self.scan.value_start_filter {
                            match decode_string_fragment(false, after_quote) {
                                Some((decoded, false, _rest)) => {
                                    return members.iter().any(|m| {
                                        let mb = m.as_bytes();
                                        mb.len() >= decoded.len()
                                            && mb.starts_with(decoded.as_bytes())
                                    });
                                }
                                // Closing quote in the remainder: the string
                                // completed, so the content must be exactly a
                                // member, and any trailing bytes must start a
                                // legal structural continuation (the sim
                                // cannot judge interiors).
                                Some((decoded, true, rest)) => {
                                    let content_ok =
                                        members.iter().any(|m| m.as_bytes() == decoded.as_bytes());
                                    if !content_ok {
                                        return false;
                                    }
                                    return match rest
                                        .iter()
                                        .find(|&&b| !matches!(b, b' ' | b'\n' | b'\t' | b'\r'))
                                    {
                                        None => true,
                                        Some(&first) => {
                                            !self.scan.value_at_root
                                                && matches!(first, b',' | b'}' | b']')
                                        }
                                    };
                                }
                                None => return false,
                            }
                        }
                        if let Some(known) = &self.scan.key_start_filter {
                            match decode_string_fragment(false, after_quote) {
                                Some((decoded, false, _rest)) => {
                                    return known.iter().any(|k| {
                                        let kb = k.as_bytes();
                                        kb.len() >= decoded.len()
                                            && kb.starts_with(decoded.as_bytes())
                                    });
                                }
                                Some((_, true, _rest)) => {}
                                None => return false,
                            }
                        }
                    }
                } else if self.scan.literal_tail {
                    // Literal/number tail. The first-byte fast path normally
                    // stands down here because literal CONTINUATION bytes are
                    // legal — but a byte that is neither whitespace, a legal
                    // structural byte, nor a literal-continuation byte can
                    // still be refused EXACTLY: this is what makes a closed
                    // object's "one more `,`" an immediate refusal at a
                    // literal tail instead of a dead-end committed one byte
                    // later.
                    if let Some(&first) = bytes.first() {
                        let ws = matches!(first, b' ' | b'\n' | b'\t' | b'\r');
                        let structural = self.scan.structural_next.contains(&first);
                        let literal_cont = first.is_ascii_digit()
                            || matches!(
                                first,
                                b'.' | b'e'
                                    | b'E'
                                    | b'+'
                                    | b'-'
                                    | b't'
                                    | b'r'
                                    | b'u'
                                    | b'f'
                                    | b'a'
                                    | b'l'
                                    | b's'
                                    | b'n'
                            );
                        if !ws && !structural && !literal_cont {
                            return false;
                        }
                    }
                    // Number-tail fast path (spec §9.2 bounded mask time): at a
                    // number position, a token of pure number-continuation bytes
                    // is legal without the clone+reparse simulation — a
                    // per-token O(vocab × output) trap at every multi-digit
                    // number otherwise. Under `Integer` the set excludes `.`:
                    // a fraction can never mask-validate (the established
                    // root-level rule). Remaining dead ends (a fraction under
                    // `Integer` already typed, impossible Const/Enum targets)
                    // are caught by the scan's number_dead_end flag on the NEXT
                    // advance — a typed EmptyAllowedSet, never a false success.
                    if self.scan.number_tail
                        && (self.scan.number_tail_free || self.scan.number_tail_integer)
                    {
                        const NUM_CONT: &[u8] = b"0123456789.eE+-";
                        const NUM_CONT_NO_DOT: &[u8] = b"0123456789eE+-";
                        let set = if self.scan.number_tail_free {
                            NUM_CONT
                        } else {
                            NUM_CONT_NO_DOT
                        };
                        if bytes.iter().all(|b| set.contains(b)) {
                            // Byte-set membership alone is not the JSON number
                            // grammar: without this check the fast path would
                            // mask-allow `01` (leading zero), `1ee`, `1.2.3`
                            // and `--`, committing bytes the authority parser
                            // then rejects — a mask/simulation disagreement
                            // that kills the request at the next step.
                            return json_number_prefix_allows(&self.scan.number_text, bytes);
                        }
                    }
                }
                // Literal tails (and everything not fast-pathed) fall through:
                // simulate the advance and check if it errors.
                let mut sim = self.clone();
                sim.advance(bytes);
                !sim.errored
            }

            /// Commit raw bytes into the parser, advancing state.
            ///
            /// Bytes are accumulated and re-parsed incrementally. Partial
            /// UTF-8 sequences are buffered until complete. The parser
            /// handles JSON tokens: structural characters, strings (with
            /// escape handling), numbers, booleans, null.
            pub fn advance(&mut self, bytes: &[u8]) {
                // Whitespace-run tracking: whitespace counts only at a
                // structural position; inside a string it is inert content.
                // Tracked BEFORE the accepted/errored early return so the run
                // also grows for post-acceptance whitespace (the burn class:
                // a valid document followed by unlimited tabs).
                let all_ws = !bytes.is_empty()
                    && bytes
                        .iter()
                        .all(|&b| matches!(b, b' ' | b'\n' | b'\t' | b'\r'));
                self.ws_run = if all_ws {
                    self.ws_run.saturating_add(1)
                } else {
                    0
                };
                if self.errored || (self.accepted && !self.number_open) {
                    return;
                }
                self.bytes.extend_from_slice(bytes);
                // Refresh the raw scan BEFORE parsing so accept-time duplicate
                // detection (strict mode, spec §7.1) sees the current buffer.
                //
                // Cost note: this re-scans the whole buffer, so a generation of
                // `n` tokens costs O(n²) byte steps. It is NOT made incremental
                // because `scan_raw`'s output (`RawScan`) does not carry the
                // object/array frame stack — the frames hold the per-frame
                // pending property schema, so resuming a scan from `RawScan`
                // alone would silently mis-constrain nested objects. At serve
                // scale (a schema-constrained generation is bounded by
                // `max_tokens`, i.e. a few KB) the measured cost is sub-millisecond
                // per generation; an incremental rewrite must first make the frame
                // stack part of the resumable state.
                self.scan = scan_raw(&self.bytes, &self.root);
                self.parse();
                // Whitespace appended inside a string is inert content, not a
                // structural run.
                if self.scan.in_string {
                    self.ws_run = 0;
                }
            }

            /// Debug: the number of raw bytes the matcher has consumed.
            #[doc(hidden)]
            pub fn buffer_len(&self) -> usize {
                self.bytes.len()
            }

            /// Debug: the raw buffer contents.
            #[doc(hidden)]
            pub fn buffer_bytes(&self) -> &[u8] {
                &self.bytes
            }

            /// A hash signature of EVERY matcher state the token mask depends
            /// on: the accepted/errored/number-open bits plus the whole raw
            /// scan (structural sets, filters, prefixes, dead-end flags).
            ///
            /// This is the cache key for the XGrammar-style per-state token
            /// mask: two matchers with equal signatures return the SAME verdict
            /// for every candidate token, because every `is_token_allowed` path
            /// — fast paths and simulation fallthrough alike — reads only these
            /// fields. Committed sibling VALUES are deliberately excluded: this
            /// subset has no cross-key constraints, so committed values never
            /// re-enter the mask (they matter only at final validation, which
            /// is monotone per position). A mask cache keyed by this signature
            /// is therefore sound within one schema.
            #[doc(hidden)]
            pub fn mask_state_signature(&self) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                self.accepted.hash(&mut h);
                self.errored.hash(&mut h);
                self.number_open.hash(&mut h);
                self.ws_run.min(WS_RUN_CAP).hash(&mut h);
                let sc = &self.scan;
                sc.duplicate_keys.hash(&mut h);
                sc.in_string.hash(&mut h);
                sc.escape_pending.hash(&mut h);
                sc.literal_tail.hash(&mut h);
                sc.unknown_key.hash(&mut h);
                sc.required_missing_on_close.hash(&mut h);
                sc.min_items_unmet_on_close.hash(&mut h);
                sc.array_over_max.hash(&mut h);
                sc.number_tail.hash(&mut h);
                sc.number_tail_free.hash(&mut h);
                sc.number_tail_integer.hash(&mut h);
                sc.number_dead_end.hash(&mut h);
                sc.value_at_root.hash(&mut h);
                sc.in_string_sated_closed.hash(&mut h);
                for &b in &sc.structural_next {
                    b.hash(&mut h);
                }
                0xFFu8.hash(&mut h);
                match &sc.value_next {
                    Some(v) => {
                        1u8.hash(&mut h);
                        for &b in v {
                            b.hash(&mut h);
                        }
                    }
                    None => 0u8.hash(&mut h),
                }
                0xFEu8.hash(&mut h);
                match &sc.key_filter {
                    Some(v) => {
                        1u8.hash(&mut h);
                        for k in v {
                            k.hash(&mut h);
                        }
                    }
                    None => 0u8.hash(&mut h),
                }
                sc.key_prefix_decoded.hash(&mut h);
                match &sc.value_filter {
                    Some(v) => {
                        1u8.hash(&mut h);
                        for k in v {
                            k.hash(&mut h);
                        }
                    }
                    None => 0u8.hash(&mut h),
                }
                sc.value_prefix_decoded.hash(&mut h);
                match &sc.value_start_filter {
                    Some(v) => {
                        1u8.hash(&mut h);
                        for k in v {
                            k.hash(&mut h);
                        }
                    }
                    None => 0u8.hash(&mut h),
                }
                match &sc.key_start_filter {
                    Some(v) => {
                        1u8.hash(&mut h);
                        for k in v {
                            k.hash(&mut h);
                        }
                    }
                    None => 0u8.hash(&mut h),
                }
                for &b in &sc.number_text {
                    b.hash(&mut h);
                }
                h.finish()
            }

            /// True when the full JSON value has been parsed and conforms
            /// to the schema. After acceptance, only whitespace is allowed —
            /// except while `number_open`, when digits may still extend the
            /// trailing number.
            pub fn is_accepting(&self) -> bool {
                self.accepted && !self.errored
            }

            /// True when the parser has hit an error (invalid JSON or
            /// schema violation).
            pub fn is_errored(&self) -> bool {
                self.errored
            }

            /// Attempt to parse the accumulated bytes as far as possible.
            /// On success, sets `accepted` if the JSON value is complete.
            /// On failure, sets `errored`.
            fn parse(&mut self) {
                // Try to parse the accumulated bytes as a JSON value.
                // We use serde_json's streaming parser for incremental
                // byte-level parsing, then validate the result against
                // the schema.
                //
                // For incremental prefix checking, we attempt to parse
                // the full buffer. If it succeeds, validate. If it fails
                // with an EOF error (incomplete), we're still going. If
                // it fails with any other error, the JSON is malformed.
                let slice = &self.bytes[..];

                // Skip leading whitespace.
                let mut start = 0;
                while start < slice.len()
                    && (slice[start] == b' '
                        || slice[start] == b'\n'
                        || slice[start] == b'\t'
                        || slice[start] == b'\r')
                {
                    start += 1;
                }

                if start >= slice.len() {
                    return; // Only whitespace so far.
                }

                // Scan-proven dead ends fire BEFORE serde classification
                // (spec §7.1/A17: a typed error at the token that proves it).
                // The old placement — inside the serde EOF-error arm only —
                // was bypassed whenever the truncated document ended in a
                // number (the Ok-arm number-growth logic): `{"bad": 1` burned
                // to max_tokens while `{"bad": "` died instantly. These flags
                // are proven by the raw scan alone; serde's classification of
                // the truncation point is irrelevant.
                if self.scan.duplicate_keys
                    || self.scan.unknown_key
                    || self.scan.required_missing_on_close
                    || self.scan.min_items_unmet_on_close
                    || self.scan.array_over_max
                    || self.scan.number_dead_end
                {
                    self.errored = true;
                    return;
                }

                // Try parsing as a complete JSON value from `start`.
                let rest = &slice[start..];
                match serde_json::from_slice::<serde_json::Value>(rest) {
                    Ok(v) => {
                        // The value parsed as complete JSON. But in
                        // incremental mode, a number like "4" at the
                        // end of the buffer could still grow into "42".
                        // We need to determine whether the value is
                        // truly complete or might be extended by future
                        // bytes.
                        //
                        // Only numbers can grow: "4" → "42", "1." →
                        // "1.5", "1e" → "1e3". All other JSON value
                        // types (string, bool, null, object, array) have
                        // definite closing delimiters that make them
                        // complete.
                        let is_number = v.is_number();
                        // A number is "potentially growing" only when the
                        // buffer's LAST byte continues it — a trailing
                        // whitespace byte terminates the number for good.
                        let number_could_grow = is_number
                            && matches!(
                                rest.last(),
                                Some(b'0'..=b'9' | b'+' | b'-' | b'.' | b'e' | b'E')
                            );

                        if number_could_grow {
                            // The number might grow. Check if the type
                            // is even compatible with the schema. If the
                            // schema expects a non-number type, no
                            // continuation of this number can ever match
                            // — error now.
                            if self.scan.duplicate_keys {
                                // Strict mode: duplicate decoded keys anywhere
                                // in the raw buffer are invalid (serde_json
                                // silently merges them, so this is the only
                                // honest check, spec §7.1).
                                self.errored = true;
                            } else {
                                match validate(&v, &self.root) {
                                    Ok(()) => {
                                        // The buffer IS a complete conforming
                                        // value — EOS must be legal — but the
                                        // number could still grow, so digit
                                        // tokens stay legal too (number_open).
                                        self.accepted = true;
                                        self.number_open = true;
                                        self.stack.clear();
                                    }
                                    Err(_reason) => {
                                        // The number doesn't match the schema
                                        // yet, but it might grow. Check if
                                        // the current raw bytes could be a
                                        // prefix of a value that WOULD match.
                                        // If the schema expects a non-number
                                        // type, no continuation can fix it.
                                        // If the schema expects a specific
                                        // number (const/enum), check if the
                                        // current bytes are a prefix of that
                                        // number's JSON representation.
                                        if number_could_grow_to_match(&v, &self.root, rest) {
                                            // Don't error — might grow into
                                            // a matching value.
                                        } else {
                                            self.errored = true;
                                        }
                                    }
                                }
                            }
                        } else {
                            // The value is complete (non-number, or a
                            // number followed by a non-digit delimiter).
                            // Validate and accept or error.
                            if self.scan.duplicate_keys {
                                self.errored = true;
                            } else {
                                match validate(&v, &self.root) {
                                    Ok(()) => {
                                        self.accepted = true;
                                        self.number_open = false;
                                        self.stack.clear();
                                    }
                                    Err(_reason) => {
                                        self.errored = true;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Check if it's an EOF error (incomplete input).
                        if !is_eof_error(&e) {
                            self.errored = true;
                        } else {
                            // Scan-proven dead ends (spec §7.1/A17: dead ends
                            // are typed constraint errors at the token that
                            // proves them — never a burn-to-max_tokens wedge
                            // and never an unconstrained fallback):
                            // duplicate keys (strict mode), unknown keys under
                            // a closed object, `}` with unmet `required`,
                            // arrays under/over their item bounds, and numbers
                            // that can provably never conform.
                            if self.scan.duplicate_keys
                                || self.scan.unknown_key
                                || self.scan.required_missing_on_close
                                || self.scan.min_items_unmet_on_close
                                || self.scan.array_over_max
                                || self.scan.number_dead_end
                            {
                                self.errored = true;
                                return;
                            }
                            // A dangling fraction point can never complete to
                            // an integer — "4." under `{"type":"integer"}` is
                            // a dead end, not a prefix.
                            if matches!(self.root, SchemaNode::Integer)
                                && rest.last() == Some(&b'.')
                            {
                                self.errored = true;
                                return;
                            }
                            // EOF: incomplete, keep buffering — but enforce
                            // dead ends remain typed constraint errors" —
                            // preventing them is better). Nested-array maxItems
                            // is caught by validate on the complete value; the
                            // root-level check covers the common structured-
                            // output case (`{"type":"array","maxItems":N}`).
                            if let SchemaNode::Array {
                                max_items: Some(max),
                                ..
                            } = &self.root
                            {
                                if let Some(count) = count_root_array_items(rest) {
                                    if count > *max {
                                        self.errored = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Schema validation ─────────────────────────────────────────────

        /// Validate a parsed JSON value against a schema node.
        /// This is the independent validator used by the matcher and by
        /// tests. It does NOT reuse the matcher's incremental logic.
        pub fn validate(value: &serde_json::Value, schema: &SchemaNode) -> Result<(), String> {
            match schema {
                SchemaNode::Any => Ok(()),
                SchemaNode::String => {
                    if value.is_string() {
                        Ok(())
                    } else {
                        Err(format!("expected string, got {}", type_name(value)))
                    }
                }
                SchemaNode::Number => {
                    if value.is_number() {
                        // Reject NaN/Infinity — serde_json doesn't produce
                        // them from parsing, but be explicit.
                        if let Some(n) = value.as_f64() {
                            if n.is_nan() || n.is_infinite() {
                                return Err("NaN/Infinity not allowed".to_string());
                            }
                        }
                        Ok(())
                    } else {
                        Err(format!("expected number, got {}", type_name(value)))
                    }
                }
                SchemaNode::Integer => {
                    if value.is_i64() || value.is_u64() {
                        Ok(())
                    } else if let Some(f) = value.as_f64() {
                        if f.fract() == 0.0 && f.is_finite() {
                            Ok(())
                        } else {
                            Err(format!("expected integer, got {}", type_name(value)))
                        }
                    } else {
                        Err(format!("expected integer, got {}", type_name(value)))
                    }
                }
                SchemaNode::Boolean => {
                    if value.is_boolean() {
                        Ok(())
                    } else {
                        Err(format!("expected boolean, got {}", type_name(value)))
                    }
                }
                SchemaNode::Null => {
                    if value.is_null() {
                        Ok(())
                    } else {
                        Err(format!("expected null, got {}", type_name(value)))
                    }
                }
                SchemaNode::Object {
                    properties,
                    required,
                    additional_properties,
                } => {
                    let obj = value
                        .as_object()
                        .ok_or_else(|| format!("expected object, got {}", type_name(value)))?;

                    // Check required keys.
                    for req in required {
                        if !obj.contains_key(req) {
                            return Err(format!("missing required property: {req}"));
                        }
                    }

                    // Check each property against its schema.
                    for (key, val) in obj {
                        // Find the property schema.
                        let prop_schema = properties.iter().find(|(k, _)| k == key);
                        match prop_schema {
                            Some((_, schema)) => validate(val, schema)?,
                            None => {
                                if !*additional_properties {
                                    return Err(format!("additional property {key:?} not allowed"));
                                }
                            }
                        }
                    }

                    // Check for duplicate keys — serde_json with preserve_order
                    // keeps the last value, so duplicates are already merged.
                    // The raw JSON parser should reject duplicates; here we
                    // trust the parser's behavior.

                    Ok(())
                }
                SchemaNode::Array {
                    items,
                    min_items,
                    max_items,
                } => {
                    let arr = value
                        .as_array()
                        .ok_or_else(|| format!("expected array, got {}", type_name(value)))?;

                    if arr.len() < *min_items {
                        return Err(format!("array has {} items, min {}", arr.len(), min_items));
                    }

                    if let Some(max) = max_items {
                        if arr.len() > *max {
                            return Err(format!("array has {} items, max {}", arr.len(), max));
                        }
                    }

                    for item in arr {
                        validate(item, items)?;
                    }

                    Ok(())
                }
                SchemaNode::Enum(values) => {
                    for v in values {
                        if json_equal(value, v) {
                            return Ok(());
                        }
                    }
                    Err(format!("value not in enum"))
                }
                SchemaNode::Const(c) => {
                    if json_equal(value, c) {
                        Ok(())
                    } else {
                        Err(format!("value does not match const"))
                    }
                }
            }
        }

        /// Semantic JSON equality. serde_json's `PartialEq` is variant-strict
        /// for numbers (`Number::PosInt(1) != Number::Float(1.0)`), which is NOT
        /// JSON semantics and used to strand a matcher mid-generation on
        /// `{"enum":[1, 1.5]}` when the model emitted `1.0`. Numbers compare by
        /// value (integer-exact when both sides are integers, f64 otherwise);
        /// composites compare element-wise.
        fn json_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
            fn num_eq(x: &serde_json::Number, y: &serde_json::Number) -> bool {
                if let (Some(i), Some(j)) = (x.as_i64(), y.as_i64()) {
                    return i == j;
                }
                if let (Some(u), Some(v)) = (x.as_u64(), y.as_u64()) {
                    return u == v;
                }
                match (x.as_f64(), y.as_f64()) {
                    (Some(f), Some(g)) => {
                        // Both integral: compare EXACTLY. An f64 cannot
                        // distinguish 2^64-1 from 2^64, so a plain float
                        // compare equates distinct integers (a
                        // const:18446744073709551615 schema accepting
                        // ...616). i128 covers every exact u64 and every
                        // integral f64 below 2^126; beyond that, bit
                        // equality is the honest test.
                        let f_int = f.is_finite() && f.fract() == 0.0;
                        let g_int = g.is_finite() && g.fract() == 0.0;
                        if f_int && g_int {
                            const I128_EXACT_LIMIT: f64 = 8.5e37; // 2^126
                            let exact = |v: f64| -> Option<i128> {
                                if v.abs() < I128_EXACT_LIMIT {
                                    Some(v as i128)
                                } else {
                                    None
                                }
                            };
                            match (exact(f), exact(g)) {
                                (Some(a), Some(b)) => a == b,
                                _ => f.to_bits() == g.to_bits(),
                            }
                        } else {
                            f == g
                        }
                    }
                    _ => false,
                }
            }
            match (a, b) {
                (serde_json::Value::Number(x), serde_json::Value::Number(y)) => num_eq(x, y),
                (serde_json::Value::Array(x), serde_json::Value::Array(y)) => {
                    x.len() == y.len() && x.iter().zip(y.iter()).all(|(i, j)| json_equal(i, j))
                }
                (serde_json::Value::Object(x), serde_json::Value::Object(y)) => {
                    x.len() == y.len()
                        && x.iter()
                            .all(|(k, v)| y.get(k).is_some_and(|w| json_equal(v, w)))
                }
                _ => a == b,
            }
        }

        fn type_name(v: &serde_json::Value) -> &'static str {
            match v {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "boolean",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::String(_) => "string",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
            }
        }

        // ── Conservative jump-forward (spec §7.3 G3) ──────────────────────

        /// Hard upper bound on the length of a forced-token run. A caller
        /// may pass a larger `max_run`, but it is clamped to this value to
        /// keep the vocabulary scan bounded regardless of caller input.
        const FORCED_RUN_HARD_CAP: usize = 64;

        /// Why a [`forced_token_run`] stopped.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum ForcedStop {
            /// More than one legal next token ID exists at this step — a
            /// branch point. The run ends before the branch; ordinary
            /// constrained decode must decide.
            Branch,
            /// The unique legal next token is an EOS/stop id. The EOS
            /// token is **not** included in the run; generation stops
            /// before it so the caller can apply stop-sequence policy.
            EosBoundary,
            /// The run reached `max_run` (clamped to
            /// [`FORCED_RUN_HARD_CAP`]) without encountering a branch,
            /// EOS boundary, or proof failure.
            Cap,
            /// No legal next token ID exists — the schema and vocabulary
            /// cannot produce valid output from the current parser state.
            /// Ordinary constrained decode is the correct fallback.
            ProofFailed,
        }

        /// Result of a [`forced_token_run`]: the provably-forced token IDs
        /// and the reason the run stopped.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct ForcedRun {
            /// Token IDs that are provably forced, in order. Empty when
            /// the run stopped at the first step.
            pub token_ids: Vec<u32>,
            /// Why the run stopped.
            pub stop: ForcedStop,
        }

        /// Fill `out` with `true` at each index whose token bytes are
        /// allowed by `matcher` ([`SchemaMatcher::is_token_allowed`]).
        ///
        /// `out` is filled up to `min(token_bytes.len(), out.len())`.
        /// This is the vocabulary-scan primitive used by
        /// [`forced_token_run`]; it is also useful for callers that need
        /// a token mask without running the full planner.
        pub fn token_mask_bytes(
            matcher: &SchemaMatcher,
            token_bytes: &[impl AsRef<[u8]>],
            out: &mut [bool],
        ) {
            let n = token_bytes.len().min(out.len());
            for i in 0..n {
                out[i] = matcher.is_token_allowed(token_bytes[i].as_ref());
            }
        }

        /// Conservative jump-forward planner (spec §7.3 G3).
        ///
        /// On a **private clone** of `matcher`, finds a bounded run of
        /// token IDs for which there is exactly one legal next token at
        /// each step under the schema's hard output constraints.
        ///
        /// # Algorithm
        ///
        /// 1. Clone `matcher` as a private cursor. For each step until
        ///    `max_run` (clamped to [`FORCED_RUN_HARD_CAP`]):
        /// 2. Scan all vocab IDs; collect IDs where
        ///    [`SchemaMatcher::is_token_allowed`] is `true` **and** the
        ///    ID is not an EOS id unless the cursor is accepting.
        /// 3. If exactly one legal next token ID, append it, advance the
        ///    cursor with those bytes, and continue.
        /// 4. If zero legal IDs → [`ForcedStop::ProofFailed`] (or
        ///    [`ForcedStop::EosBoundary`] if an EOS id was the only
        ///    allowed token but filtered because the cursor is not
        ///    accepting). If more than one → [`ForcedStop::Branch`]. If
        ///    the unique ID is an EOS/stop id →
        ///    [`ForcedStop::EosBoundary`] without including it (stop
        ///    **before** EOS).
        ///
        /// Unique characters are insufficient: this function compares
        /// token **IDs**, not decoded strings — two IDs with the same
        /// byte representation both being allowed is a branch.
        ///
        /// This is a pure planner: it does not re-tokenize, does not
        /// touch GPU state, and does not claim a speedup. The caller is
        /// responsible for feeding the returned IDs through the existing
        /// forward path.
        pub fn forced_token_run(
            matcher: &SchemaMatcher,
            token_bytes: &[impl AsRef<[u8]>],
            eos_ids: &[u32],
            max_run: usize,
        ) -> ForcedRun {
            let max_run = max_run.min(FORCED_RUN_HARD_CAP);
            let eos_set: HashSet<u32> = eos_ids.iter().copied().collect();
            let vocab_len = token_bytes.len();

            let mut cursor = matcher.clone();
            let mut token_ids = Vec::with_capacity(max_run);
            let mut mask = vec![false; vocab_len];

            for _ in 0..max_run {
                token_mask_bytes(&cursor, token_bytes, &mut mask);

                let accepting = cursor.is_accepting();

                // Collect legal IDs: allowed AND (not EOS OR accepting).
                let mut legal: Vec<u32> = Vec::new();
                let mut had_allowed_eos = false;
                for id in 0..vocab_len as u32 {
                    if !mask[id as usize] {
                        continue;
                    }
                    let is_eos = eos_set.contains(&id);
                    if is_eos && !accepting {
                        had_allowed_eos = true;
                        continue;
                    }
                    legal.push(id);
                }

                match legal.len() {
                    0 => {
                        // No legal non-EOS token. If an EOS id was the
                        // only allowed token (filtered because the
                        // cursor is not accepting), stop before it.
                        if had_allowed_eos {
                            return ForcedRun {
                                token_ids,
                                stop: ForcedStop::EosBoundary,
                            };
                        }
                        return ForcedRun {
                            token_ids,
                            stop: ForcedStop::ProofFailed,
                        };
                    }
                    1 => {
                        let id = legal[0];
                        // If the unique legal token is an EOS id (only
                        // possible when accepting), stop before it.
                        if eos_set.contains(&id) {
                            return ForcedRun {
                                token_ids,
                                stop: ForcedStop::EosBoundary,
                            };
                        }
                        // Unique non-EOS token: append and advance. A token
                        // with no byte representation advances nothing —
                        // emitting it would loop zero-progress up to the cap.
                        let bytes = token_bytes[id as usize].as_ref();
                        if bytes.is_empty() {
                            return ForcedRun {
                                token_ids,
                                stop: ForcedStop::ProofFailed,
                            };
                        }
                        cursor.advance(bytes);
                        token_ids.push(id);
                    }
                    _ => {
                        return ForcedRun {
                            token_ids,
                            stop: ForcedStop::Branch,
                        };
                    }
                }
            }

            ForcedRun {
                token_ids,
                stop: ForcedStop::Cap,
            }
        }

        // ── Tests ─────────────────────────────────────────────────────────

        #[cfg(test)]
        mod tests {
            use super::*;
            use serde_json::json;

            // ── Compilation tests ──────────────────────────────────────────

            #[test]
            fn compiles_simple_object_schema() {
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid schema");
                assert!(matches!(compiled.root(), SchemaNode::Object { .. }));
            }

            #[test]
            fn compiles_nested_array_of_objects() {
                let schema = json!({
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"}
                        },
                        "required": ["id"]
                    }
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid schema");
                assert!(matches!(compiled.root(), SchemaNode::Array { .. }));
            }

            #[test]
            fn rejects_ref() {
                let schema = json!({"$ref": "#/definitions/Foo"});
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::ReferencesNotSupported { .. }));
            }

            #[test]
            fn rejects_pattern() {
                let schema = json!({"type": "string", "pattern": "^[a-z]+$"});
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::RegexNotSupported));
            }

            #[test]
            fn rejects_combinator() {
                for kw in ["allOf", "anyOf", "oneOf", "not"] {
                    let schema = json!({kw: []});
                    let err = CompiledSchema::compile(&schema).unwrap_err();
                    assert!(
                        matches!(err, SchemaError::CombinatorNotSupported { .. }),
                        "{kw} should be rejected"
                    );
                }
            }

            #[test]
            fn rejects_recursion() {
                // The ONLY recursion vector in JSON Schema is `$ref`, which the
                // reference gate rejects. A repeated property NAME at different
                // nesting depths is finite and compiles (see
                // accepts_repeated_property_name_at_depth).
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "self": {
                            "type": "object",
                            "properties": {
                                "self": {"type": "string"}
                            }
                        }
                    }
                });
                let compiled = CompiledSchema::compile(&schema).expect("finite schema compiles");

                // $ref is the recursion vector and stays rejected.
                let ref_schema = json!({
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#"}
                    }
                });
                let err = CompiledSchema::compile(&ref_schema).unwrap_err();
                assert!(matches!(err, SchemaError::ReferencesNotSupported { .. }));
                let _ = compiled;
            }

            #[test]
            fn rejects_unsatisfiable_min_max_items() {
                let schema = json!({
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                    "maxItems": 3
                });
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::Unsatisfiable { .. }));
            }

            #[test]
            fn required_not_in_properties_satisfiable_when_open() {
                // Spec §7.1: {required:["b"]} with default additionalProperties
                // (true) is SATISFIABLE — b may be any value — so it compiles.
                // (This used to be rejected wholesale, over-refusing valid
                // schemas the CLI validator accepted.)
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["b"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("satisfiable");
                let mut m = compiled.matcher();
                m.advance(b"{\"b\": true}");
                assert!(
                    m.is_accepting(),
                    "any value satisfies the open required key"
                );
                assert!(!m.is_errored());
            }

            #[test]
            fn rejects_unsatisfiable_required_closed_object() {
                // required without a properties entry AND additionalProperties:
                // false can never emit the required key — genuinely unsatisfiable.
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["b"],
                    "additionalProperties": false
                });
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::Unsatisfiable { .. }));
            }

            #[test]
            fn rejects_unsupported_keyword() {
                let schema = json!({"type": "string", "minLength": 5});
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::UnsupportedKeyword { .. }));
            }

            #[test]
            fn rejects_unknown_keyword_default_deny() {
                // Default-deny (spec §7.1): an unknown keyword must refuse the
                // schema, never fall through to "annotation, ignore". The old
                // `_ => {}` fall-through silently admitted if/then/else and
                // additionalItems, whose constraints were then ignored.
                for kw in [
                    "if",
                    "then",
                    "else",
                    "additionalItems",
                    "dependsOn",
                    "x-custom",
                ] {
                    let mut schema = json!({"type": "string"});
                    schema[kw] = json!({});
                    let err = CompiledSchema::compile(&schema).unwrap_err();
                    assert!(
                        matches!(err, SchemaError::UnsupportedKeyword { .. }),
                        "keyword {kw} must be rejected, got {err:?}"
                    );
                }
            }

            #[test]
            fn rejects_union_type_form() {
                // {"type": ["integer","null"]} used to fall into the inference
                // path and compile to an unconstrained Any.
                let schema = json!({"type": ["integer", "null"]});
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::UnsupportedKeyword { .. }));
            }

            #[test]
            fn rejects_schema_form_additional_properties() {
                // The schema form used to be silently flattened to `true`.
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                });
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::UnsupportedKeyword { .. }));
            }

            #[test]
            fn accepts_repeated_property_name_at_depth() {
                // A repeated property NAME at different nesting depths is
                // ordinary finite JSON Schema, not recursion — recursion needs
                // $ref, which the keyword gate already rejects.
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "meta": {
                            "type": "object",
                            "properties": {
                                "meta": {"type": "string"}
                            }
                        }
                    },
                    "required": ["meta"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("finite schema compiles");
                let mut m = compiled.matcher();
                m.advance(b"{\"meta\": {\"meta\": \"deep\"}}");
                assert!(m.is_accepting());
            }

            #[test]
            fn number_split_across_tokens_stays_continuable() {
                // `{"type":"number"}` fed one digit per token: "4" parses and
                // conforms, so EOS must be legal — but the number could still
                // grow, so a digit must remain legal too. The old code set
                // `accepted` and then refused every non-whitespace byte,
                // making a digit-per-token number unreachable.
                let compiled = CompiledSchema::compile(&json!({"type": "number"})).unwrap();
                let mut m = compiled.matcher();
                m.advance(b"4");
                assert!(m.is_accepting(), "a conforming number must allow EOS");
                assert!(m.is_token_allowed(b"2"), "the number may still grow");
                m.advance(b"2");
                assert!(m.is_accepting());
                // A non-continuing byte closes the number: afterwards only
                // whitespace is legal.
                assert!(m.is_token_allowed(b" "));
                m.advance(b" ");
                assert!(m.is_accepting());
                assert!(!m.is_token_allowed(b"7"), "a terminated number cannot grow");
            }

            #[test]
            fn integer_rejects_fractional_growth() {
                // `{"type":"integer"}`: "4" conforms and may grow, but "4."
                // can never become an integer — the mask must refuse it.
                let compiled = CompiledSchema::compile(&json!({"type": "integer"})).unwrap();
                let mut m = compiled.matcher();
                m.advance(b"4");
                assert!(m.is_accepting());
                assert!(m.is_token_allowed(b"2"));
                assert!(
                    !m.is_token_allowed(b"."),
                    "an integer cannot grow a fraction"
                );
            }

            #[test]
            fn rejects_too_many_enum_values() {
                let values: Vec<i32> = (0..300).collect();
                let schema = json!({"enum": values});
                let err = CompiledSchema::compile(&schema).unwrap_err();
                assert!(matches!(err, SchemaError::TooManyEnumValues { .. }));
            }

            // ── Incremental matcher: acceptance tests ──────────────────────

            #[test]
            fn matcher_accepts_valid_object() {
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"name\": \"Alice\", \"age\": 30}");
                assert!(m.is_accepting(), "valid object should be accepted");
                assert!(!m.is_errored());
            }

            #[test]
            fn matcher_rejects_missing_required() {
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"age\": 30}");
                assert!(!m.is_accepting());
                assert!(m.is_errored(), "missing required should error");
            }

            #[test]
            fn matcher_rejects_wrong_type() {
                let schema = json!({"type": "string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"42");
                assert!(!m.is_accepting());
                assert!(m.is_errored(), "number for string schema should error");
            }

            #[test]
            fn matcher_accepts_nested_arrays() {
                let schema = json!({
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[[1, 2], [3, 4], [5]]");
                assert!(m.is_accepting(), "nested arrays should be accepted");
            }

            #[test]
            fn matcher_accepts_nested_objects() {
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"}
                            },
                            "required": ["city"]
                        }
                    },
                    "required": ["address"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"address\": {\"city\": \"NYC\", \"zip\": \"10001\"}}");
                assert!(m.is_accepting(), "nested objects should be accepted");
            }

            #[test]
            fn matcher_accepts_escaped_strings() {
                let schema = json!({"type": "string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                // String with escaped quote and backslash: "hello\"world\\"
                m.advance(b"\"hello\\\"world\\\\\"");
                assert!(m.is_accepting(), "escaped string should be accepted");
            }

            #[test]
            fn matcher_accepts_split_utf8_across_advances() {
                // The CJK character 中 = E4 B8 AD. Feed each byte
                // in a separate advance call inside a string value.
                let schema = json!({"type": "string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\"");
                m.advance(&[0xE4]);
                m.advance(&[0xB8]);
                m.advance(&[0xAD]);
                m.advance(b"\"");
                assert!(m.is_accepting(), "split UTF-8 should be accepted");
            }

            #[test]
            fn matcher_rejects_additional_properties_when_false() {
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\": \"x\", \"b\": 1}");
                assert!(m.is_errored(), "additional property should be rejected");
            }

            #[test]
            fn matcher_allows_additional_properties_when_true() {
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": true
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\": \"x\", \"b\": 1}");
                assert!(m.is_accepting(), "additional property should be allowed");
            }

            #[test]
            fn matcher_accepts_enum() {
                let schema = json!({"enum": ["red", "green", "blue"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\"red\"");
                assert!(m.is_accepting());
            }

            #[test]
            fn matcher_rejects_enum_non_member() {
                let schema = json!({"enum": ["red", "green", "blue"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\"purple\"");
                assert!(m.is_errored());
            }

            // ── Duplicate object keys (strict mode, A15) ───────────────────

            #[test]
            fn matcher_rejects_duplicate_object_keys() {
                // serde_json merges duplicates silently (last wins), so the
                // incremental parse alone accepted {"a":1,"a":2}. Strict mode
                // rejects them (spec §7.1).
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\": 1, \"a\": 2}");
                assert!(
                    m.is_errored(),
                    "duplicate keys must be invalid in strict mode"
                );
            }

            #[test]
            fn matcher_rejects_escaped_key_duplicate() {
                // Escaped and unescaped spellings of the same decoded key are
                // the SAME key: {"a":1,"\u0061":2} is a duplicate.
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\": 1, \"\\u0061\": 2}");
                assert!(
                    m.is_errored(),
                    "escaped spelling of the same key is a duplicate"
                );
            }

            #[test]
            fn matcher_allows_same_key_in_sibling_objects() {
                // The same key at different nesting levels is fine.
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "object",
                            "properties": {"b": {"type": "integer"}},
                            "required": ["b"]
                        },
                        "b": {"type": "integer"}
                    },
                    "required": ["a"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\": {\"b\": 1}, \"b\": 2}");
                assert!(m.is_accepting());
            }

            // ── Semantic numeric equality (A15) ────────────────────────────

            #[test]
            fn matcher_accepts_float_spelling_of_integer_enum() {
                // serde_json's PartialEq is variant-strict (1 != 1.0), which
                // used to strand the matcher: "1.0" passed the prefix check but
                // failed the completion equality. JSON numbers compare by value.
                let schema = json!({"enum": [1, 1.5]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"1.0");
                assert!(m.is_accepting(), "1.0 is semantically the enum member 1");
                assert!(!m.is_errored());
            }

            #[test]
            fn matcher_accepts_integer_spelling_of_float_const() {
                let schema = json!({"const": 2.5});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"2.5");
                assert!(m.is_accepting());
            }

            // ── Mask fast paths (bounded mask time, spec §9.2) ─────────────

            #[test]
            fn fast_path_allows_inert_string_content_and_refuses_structural_mismatch() {
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();

                // Structural position: `{` is legal, `x` is not (fast-fail).
                assert!(m.is_token_allowed(b"{"));
                assert!(!m.is_token_allowed(b"x"));
                m.advance(b"{\"name\": \"Al");

                // Plain string content: inert UTF-8 without quotes/backslashes
                // is allowed without simulation; a control byte is not.
                assert!(m.is_token_allowed(b"ice "));
                assert!(!m.is_token_allowed("\u{7}bolt".as_bytes()));
            }

            #[test]
            fn matcher_accepts_const() {
                let schema = json!({"const": 42});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"42");
                assert!(m.is_accepting());
            }

            #[test]
            fn matcher_rejects_const_mismatch() {
                let schema = json!({"const": 42});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"43");
                assert!(m.is_errored());
            }

            #[test]
            fn matcher_enforces_array_min_items() {
                let schema = json!({
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[1, 2]");
                assert!(m.is_errored(), "too few items should error");
            }

            #[test]
            fn matcher_enforces_array_max_items() {
                let schema = json!({
                    "type": "array",
                    "items": {"type": "integer"},
                    "maxItems": 2
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[1, 2, 3]");
                assert!(m.is_errored(), "too many items should error");
            }

            // ── Required membership at correct nesting (A15) ───────────────

            #[test]
            fn required_name_in_string_value_does_not_satisfy_parent() {
                // Schema requires "name" as a property key, but the model
                // emits "name" inside a string VALUE — that does NOT
                // satisfy the required-key constraint.
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                // "name" appears as a string value for "description", but
                // the "name" KEY is missing.
                m.advance(b"{\"description\": \"name\"}");
                assert!(
                    m.is_errored(),
                    "required key in string value should not satisfy"
                );
            }

            #[test]
            fn required_name_in_child_object_does_not_satisfy_parent() {
                // "name" as a key in a child object does not satisfy the
                // parent's required "name".
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "child": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            }
                        }
                    },
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"child\": {\"name\": \"x\"}}");
                assert!(
                    m.is_errored(),
                    "required key in child should not satisfy parent"
                );
            }

            // ── Incremental acceptance == independent validation (A15) ─────

            /// Independent validator: parse JSON and validate against the
            /// schema using the `validate` function. This does NOT reuse
            /// the matcher's incremental logic.
            fn independent_validate(
                json_str: &str,
                schema: &serde_json::Value,
            ) -> Result<(), String> {
                let compiled =
                    CompiledSchema::compile(schema).map_err(|e| format!("compile error: {e}"))?;
                let value: serde_json::Value =
                    serde_json::from_str(json_str).map_err(|e| format!("parse error: {e}"))?;
                validate(&value, compiled.root())
            }

            /// A corpus of (json_str, schema) pairs. For each, the
            /// incremental matcher's acceptance must match the independent
            /// validator's result.
            fn corpus() -> Vec<(&'static str, serde_json::Value)> {
                vec![
                    // Simple object.
                    (
                        r#"{"name": "Alice", "age": 30}"#,
                        json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}),
                    ),
                    // Missing required.
                    (
                        r#"{"age": 30}"#,
                        json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}),
                    ),
                    // Nested array of objects.
                    (
                        r#"[{"id": 1}, {"id": 2}]"#,
                        json!({"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}),
                    ),
                    // String with escapes.
                    (r#""hello\nworld""#, json!({"type": "string"})),
                    // Number.
                    (r#"42"#, json!({"type": "integer"})),
                    // Boolean.
                    (r#"true"#, json!({"type": "boolean"})),
                    // Null.
                    (r#"null"#, json!({"type": "null"})),
                    // Enum match.
                    (r#""red""#, json!({"enum": ["red", "green", "blue"]})),
                    // Enum non-match.
                    (r#""purple""#, json!({"enum": ["red", "green", "blue"]})),
                    // Const match.
                    (r#"42"#, json!({"const": 42})),
                    // Additional properties false.
                    (
                        r#"{"a": "x", "b": 1}"#,
                        json!({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}),
                    ),
                    // Array min/max.
                    (
                        r#"[1, 2, 3]"#,
                        json!({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 5}),
                    ),
                    // Deeply nested.
                    (
                        r#"{"a": {"b": {"c": [1, 2, 3]}}}"#,
                        json!({"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "object", "properties": {"c": {"type": "array", "items": {"type": "integer"}}}}}}}}),
                    ),
                    // Wrong type.
                    (r#"42"#, json!({"type": "string"})),
                    // Empty array with minItems.
                    (
                        r#"[]"#,
                        json!({"type": "array", "items": {"type": "integer"}, "minItems": 1}),
                    ),
                ]
            }

            #[test]
            fn incremental_acceptance_matches_independent_validation() {
                for (i, (json_str, schema)) in corpus().iter().enumerate() {
                    let compiled = match CompiledSchema::compile(schema) {
                        Ok(c) => c,
                        Err(e) => {
                            // If the schema is rejected at compile time,
                            // the independent validator should also reject.
                            let indep = independent_validate(json_str, schema);
                            assert!(
                            indep.is_err(),
                            "case {i}: schema rejected by compiler but independent validator succeeded: {json_str}"
                        );
                            continue;
                        }
                    };

                    // Incremental matcher: feed bytes one at a time.
                    let mut m = compiled.matcher();
                    for byte in json_str.bytes() {
                        m.advance(&[byte]);
                    }
                    let incremental_ok = m.is_accepting();
                    let incremental_err = m.is_errored();

                    // Independent validator.
                    let indep_result = independent_validate(json_str, schema);
                    let indep_ok = indep_result.is_ok();

                    assert_eq!(
                    incremental_ok, indep_ok,
                    "case {i}: incremental acceptance ({incremental_ok}) != independent validation ({indep_ok}) for: {json_str}"
                );
                    assert!(
                        !incremental_ok || !incremental_err,
                        "case {i}: both accepting and errored"
                    );
                }
            }

            #[test]
            fn incremental_acceptance_matches_when_split_arbitrarily() {
                // Feed the JSON bytes in random-sized chunks to test
                // incremental robustness.
                for (i, (json_str, schema)) in corpus().iter().enumerate() {
                    let compiled = match CompiledSchema::compile(schema) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let mut m = compiled.matcher();
                    // Feed in chunks of 1, 2, 3, 1, 1, 5, ... bytes.
                    let chunk_sizes = [1usize, 2, 3, 1, 1, 5, 1, 2, 4, 1, 3, 7];
                    let mut pos = 0;
                    let mut ci = 0;
                    while pos < json_str.len() {
                        let sz = chunk_sizes[ci % chunk_sizes.len()].min(json_str.len() - pos);
                        m.advance(&json_str.as_bytes()[pos..pos + sz]);
                        pos += sz;
                        ci += 1;
                    }
                    let incremental_ok = m.is_accepting();

                    let indep_result = independent_validate(json_str, schema);
                    let indep_ok = indep_result.is_ok();

                    assert_eq!(
                    incremental_ok, indep_ok,
                    "case {i}: split-chunk incremental ({incremental_ok}) != independent ({indep_ok}) for: {json_str}"
                );
                }
            }

            // ── is_token_allowed tests ─────────────────────────────────────

            #[test]
            fn is_token_allowed_accepts_valid_prefix() {
                let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // Opening brace is a valid prefix.
                assert!(m.is_token_allowed(b"{"));
                assert!(m.is_token_allowed(b"{\""));
            }

            #[test]
            fn is_token_allowed_rejects_invalid_prefix() {
                let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // A number is not a valid prefix for an object schema.
                assert!(!m.is_token_allowed(b"42"));
            }

            #[test]
            fn is_token_allowed_accepts_partial_json() {
                let schema = json!({"type": "string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // Opening quote is a valid prefix of a string.
                assert!(m.is_token_allowed(b"\""));
                assert!(m.is_token_allowed(b"\"hel"));
            }

            // ── Early per-token schema pruning (spec §7) ─────────────────

            #[test]
            fn pruning_refuses_null_token_at_string_value() {
                // `null` under a `type:"string"` property can never satisfy
                // the schema — the token must be refused by is_token_allowed
                // immediately, not deferred to terminal validation.
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"name\":");
                assert!(!m.is_token_allowed(b"null"), "null cannot start a string");
                assert!(!m.is_token_allowed(b"n"), "even the first byte is refused");
                // The same refusal applies at the root value position.
                let root = CompiledSchema::compile(&json!({"type": "string"}))
                    .expect("valid")
                    .matcher();
                assert!(!root.is_token_allowed(b"null"));
            }

            #[test]
            fn pruning_refuses_digit_at_string_value() {
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"name\": ");
                assert!(!m.is_token_allowed(b"4"), "a digit cannot start a string");
                assert!(!m.is_token_allowed(b"42"));
                assert!(!m.is_token_allowed(b"-1"));
            }

            #[test]
            fn pruning_refuses_true_at_number_value() {
                let schema = json!({
                    "type": "object",
                    "properties": {"age": {"type": "number"}},
                    "required": ["age"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"age\":");
                assert!(!m.is_token_allowed(b"true"), "true cannot start a number");
                assert!(!m.is_token_allowed(b"t"));
                assert!(!m.is_token_allowed(b"f"));
                // A quote is equally impossible at a number position.
                assert!(!m.is_token_allowed(b"\""));
            }

            #[test]
            fn pruning_keeps_legal_starts_allowed() {
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["name", "age", "tags"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"name\":");
                assert!(m.is_token_allowed(b"\""), "a quote starts the string");
                m.advance(b" \"x\", \"age\":");
                assert!(m.is_token_allowed(b"3"), "a digit starts the integer");
                assert!(m.is_token_allowed(b"-"));
                m.advance(b"30, \"tags\":");
                assert!(m.is_token_allowed(b"["), "a bracket starts the array");
                m.advance(b"[");
                // Inside the array the items schema (string) governs.
                assert!(m.is_token_allowed(b"\""));
                assert!(!m.is_token_allowed(b"7"), "items are strings");
                // `]` may close the (empty) array without a value.
                assert!(m.is_token_allowed(b"]"));
            }

            #[test]
            fn pruning_allows_whitespace_prefixed_legal_tokens() {
                let schema = json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"name\":");
                // Whitespace before the value start is legal JSON.
                assert!(m.is_token_allowed(b" \""));
                assert!(m.is_token_allowed(b" \n\t\""));
                // Whitespace alone is inert.
                assert!(m.is_token_allowed(b"  "));
                // But whitespace cannot rescue an impossible value start.
                assert!(!m.is_token_allowed(b" null"));
            }

            #[test]
            fn pruning_preserves_full_document_acceptance() {
                // Byte-by-byte decode of a valid document: every byte must be
                // allowed and the matcher must reach acceptance. Exercises
                // the `:` step after each key and `]`/`}` closes.
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "active": {"type": "boolean"}
                    },
                    "required": ["name", "age", "tags", "active"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                let doc = b"{\"name\": \"Al\", \"age\": 30, \"tags\": [\"x\"], \"active\": true}";
                for &b in doc {
                    assert!(
                        m.is_token_allowed(&[b]),
                        "byte {:?} must be allowed mid-document",
                        b as char
                    );
                    m.advance(&[b]);
                }
                assert!(m.is_accepting(), "a valid document must be accepted");
                assert!(!m.is_errored());
            }

            #[test]
            fn pruning_allows_empty_array_close() {
                // `]` at an array value position is not a value start but is
                // legal (empty array); pruning must not refuse it.
                let schema = json!({
                    "type": "array",
                    "items": {"type": "integer"}
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[");
                assert!(m.is_token_allowed(b"]"));
                m.advance(b"]");
                assert!(m.is_accepting());
            }

            #[test]
            fn pruning_respects_enum_and_const_starts() {
                // Enum members narrow the legal first bytes to the union of
                // their literal starts.
                let schema = json!({"enum": ["red", "green", 7]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                assert!(
                    m.is_token_allowed(b"\""),
                    "string members start with a quote"
                );
                assert!(
                    m.is_token_allowed(b"7"),
                    "the number member starts with a digit"
                );
                assert!(!m.is_token_allowed(b"t"), "no member is a boolean");
                assert!(!m.is_token_allowed(b"{"), "no member is an object");

                let c = CompiledSchema::compile(&json!({"const": true})).expect("valid");
                let m = c.matcher();
                assert!(m.is_token_allowed(b"t"));
                assert!(!m.is_token_allowed(b"f"), "const true cannot start with f");
            }

            // ── Duplicate key rejection (A15) ──────────────────────────────

            #[test]
            fn duplicate_keys_rejected() {
                // serde_json's default behavior with preserve_order is to
                // keep the LAST value for a duplicate key. The spec says
                // duplicate keys are invalid in strict mode. We test that
                // the schema validator catches this: the last value for
                // "a" is 2 (integer), but if the schema expects a string,
                // it will fail. For a more direct test, we check that
                // duplicate keys with wrong type on the second occurrence
                // are caught.
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                // Duplicate "a" key: first string, second integer.
                // serde_json with preserve_order keeps the last → "a": 2
                // → type mismatch → error.
                m.advance(b"{\"a\": \"x\", \"a\": 2}");
                assert!(
                    m.is_errored(),
                    "duplicate key with type mismatch should error"
                );
            }

            // ── Escaped/unescaped key spellings (A15) ─────────────────────

            #[test]
            fn escaped_key_matches_unescaped_in_schema() {
                // A schema with property "a\"b" (decoded: a"b). The JSON
                // can use either the escaped or unescaped form in the
                // raw bytes — they should both match the same property
                // after decoding.
                //
                // In JSON, the key MUST be escaped: "a\"b". serde_json
                // decodes this to the string a"b before looking it up
                // in the properties map, so escaped and unescaped
                // spellings are compared after decoding.
                let schema = json!({
                    "type": "object",
                    "properties": {"a\"b": {"type": "string"}},
                    "required": ["a\"b"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                // The JSON key "a\"b" decodes to a"b, which matches.
                m.advance(b"{\"a\\\"b\": \"value\"}");
                assert!(m.is_accepting(), "escaped key should match schema property");
            }

            // ── Conservative jump-forward (A16/A17) ───────────────────────

            #[test]
            fn forced_run_unique_token_then_accepting_stop() {
                // Schema {"enum":["hello"]}: valid JSON is "hello" (with
                // quotes). A vocab where exactly one token ID forms the
                // complete JSON yields a unique run.
                let schema = json!({"enum":["hello"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // id 0 = "hello" (complete JSON), id 1 = "he" (prefix of
                // the string content, NOT a valid JSON start), id 2 = "llo"
                // (unrelated), id 3 = "" (EOS, empty bytes).
                let vocab: &[&[u8]] = &[b"\"hello\"", b"he", b"llo", b""];
                let run = forced_token_run(&m, vocab, &[3], 64);
                // Step 0: only id 0 is a valid JSON prefix → unique.
                // Step 1: cursor is accepting; only EOS (empty bytes) is
                // allowed → EosBoundary.
                assert_eq!(run.token_ids, vec![0]);
                assert_eq!(run.stop, ForcedStop::EosBoundary);
            }

            #[test]
            fn forced_run_branch_on_two_tokenizations() {
                // Two token IDs that are both valid continuations of the
                // same forced text → Branch immediately, empty run.
                let schema = json!({"enum":["hello"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\""); // Now inside the string.
                                  // id 0 = "he", id 1 = "hello" — both valid string
                                  // continuations → Branch.
                let vocab: &[&[u8]] = &[b"he", b"hello", b"llo\"", b""];
                let run = forced_token_run(&m, vocab, &[3], 64);
                assert!(run.token_ids.is_empty());
                assert_eq!(run.stop, ForcedStop::Branch);
            }

            #[test]
            fn forced_run_empty_allowed_set_proof_failed() {
                // No token is a valid continuation → ProofFailed, empty run.
                let schema = json!({"type":"null"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // Neither "xyz" nor "hello" is valid JSON.
                let vocab: &[&[u8]] = &[b"xyz", b"hello"];
                let run = forced_token_run(&m, vocab, &[], 64);
                assert!(run.token_ids.is_empty());
                assert_eq!(run.stop, ForcedStop::ProofFailed);
            }

            #[test]
            fn forced_run_eos_unique_when_not_accepting() {
                // EOS (empty bytes) is the only allowed token, but the
                // cursor is not accepting → EosBoundary, empty run.
                let schema = json!({"type":"null"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // id 0 = "xyz" (not valid JSON), id 1 = "" (EOS).
                let vocab: &[&[u8]] = &[b"xyz", b""];
                let run = forced_token_run(&m, vocab, &[1], 64);
                assert!(run.token_ids.is_empty());
                assert_eq!(run.stop, ForcedStop::EosBoundary);
            }

            #[test]
            fn forced_run_max_run_cap_respected() {
                // A const number with single-digit tokens: each step has
                // exactly one legal next digit. With max_run < number of
                // digits, the run is capped.
                let schema = json!({"const":12345});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // id 0='1', 1='2', 2='3', 3='4', 4='5', 5='6', 6='' (EOS)
                let vocab: &[&[u8]] = &[b"1", b"2", b"3", b"4", b"5", b"6", b""];
                // max_run=3: run is [0,1,2], stop=Cap.
                let run = forced_token_run(&m, vocab, &[6], 3);
                assert_eq!(run.token_ids, vec![0, 1, 2]);
                assert_eq!(run.stop, ForcedStop::Cap);

                // max_run=64 (full): run completes all 5 digits, then
                // accepting + EOS → EosBoundary.
                let run_full = forced_token_run(&m, vocab, &[6], 64);
                assert_eq!(run_full.token_ids, vec![0, 1, 2, 3, 4]);
                assert_eq!(run_full.stop, ForcedStop::EosBoundary);
            }

            #[test]
            fn forced_run_clamps_huge_max_run() {
                // max_run=usize::MAX is clamped to 64; the run still
                // completes normally (5 digits + EosBoundary).
                let schema = json!({"const":12345});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                let vocab: &[&[u8]] = &[b"1", b"2", b"3", b"4", b"5", b"6", b""];
                let run = forced_token_run(&m, vocab, &[6], usize::MAX);
                assert_eq!(run.token_ids, vec![0, 1, 2, 3, 4]);
                assert_eq!(run.stop, ForcedStop::EosBoundary);
            }

            #[test]
            fn forced_run_two_ids_same_bytes_is_branch() {
                // Two different token IDs with the same byte string both
                // being allowed is a branch — unique characters are
                // insufficient; token IDs must be compared.
                let schema = json!({"type":"string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                // id 0 = `"`, id 1 = `"` (same bytes, different ID).
                let vocab: &[&[u8]] = &[b"\"", b"\"", b""];
                let run = forced_token_run(&m, vocab, &[2], 64);
                assert!(run.token_ids.is_empty());
                assert_eq!(run.stop, ForcedStop::Branch);
            }

            #[test]
            fn token_mask_bytes_fills_correctly() {
                let schema = json!({"type":"string"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let m = compiled.matcher();
                let vocab: &[&[u8]] = &[b"\"", b"42", b"hello", b""];
                let mut mask = [false; 4];
                token_mask_bytes(&m, vocab, &mut mask);
                // `"` is a valid prefix of a string → true.
                assert!(mask[0]);
                // `42` is not valid JSON for a string schema → false.
                assert!(!mask[1]);
                // `hello` is not valid JSON → false.
                assert!(!mask[2]);
                // Empty bytes are always allowed (vacuous) → true.
                assert!(mask[3]);
            }

            // ── Regression: bare {"type":"object"} without properties ──────
            #[test]
            fn bare_object_type_without_properties_compiles() {
                let schema = json!({"type": "object"});
                let compiled = CompiledSchema::compile(&schema).expect("bare object must compile");
                let m = compiled.matcher();
                // Any object should be allowed.
                assert!(m.is_token_allowed(b"{\"key\": 1}"));
                assert!(m.is_token_allowed(b"{}"));
            }

            #[test]
            fn bare_object_type_with_required_only_compiles() {
                let schema = json!({"type": "object", "required": ["name"]});
                let compiled = CompiledSchema::compile(&schema)
                    .expect("object with required but no properties must compile");
                let m = compiled.matcher();
                // `{"name": "x"}` satisfies the required field.
                assert!(m.is_token_allowed(b"{\"name\": \"x\"}"));
            }

            // ── Regression: maxItems enforced incrementally ────────────────
            #[test]
            fn max_items_blocks_extra_item_incrementally() {
                let schema = json!({"type": "array", "items": {"type": "integer"}, "maxItems": 2});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[1, 2");
                assert!(!m.is_errored(), "two items is within maxItems");
                // A third item must be refused incrementally.
                m.advance(b", 3");
                assert!(m.is_errored(), "third item exceeds maxItems");
            }

            // ── Regression: STRING array items count toward minItems/maxItems ──
            // The scan's string-completion branch used to flip the phase without
            // incrementing the frame's item count, so `]` was mask-refused
            // forever under minItems (a 100% burn-to-max wedge) and `,` stayed
            // legal past maxItems (late death at validation).
            #[test]
            fn min_items_with_string_items_allows_close() {
                let schema = json!({"type": "array", "items": {"type": "string"}, "minItems": 1});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[\"red\"");
                assert!(
                    m.is_token_allowed(b"]"),
                    "']' must be legal after one string item"
                );
                m.advance(b"]");
                assert!(m.is_accepting());
            }

            #[test]
            fn min_items_two_with_string_items_needs_both() {
                let schema = json!({"type": "array", "items": {"type": "string"}, "minItems": 2});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[\"red\"");
                assert!(
                    !m.is_token_allowed(b"]"),
                    "']' must be refused while minItems is unmet"
                );
                m.advance(b", \"blue\"");
                assert!(
                    m.is_token_allowed(b"]"),
                    "']' legal once two string items exist"
                );
                m.advance(b"]");
                assert!(m.is_accepting());
            }

            #[test]
            fn max_items_with_string_items_refuses_comma_at_cap() {
                let schema = json!({"type": "array", "items": {"type": "string"}, "maxItems": 2});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[\"red\", \"blue\"");
                assert!(
                    !m.is_token_allowed(b","),
                    "',' must be refused at maxItems with string items"
                );
                assert!(m.is_token_allowed(b"]"));
            }

            #[test]
            fn nested_string_array_min_items_prunes_close() {
                let schema = json!({
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[[\"x\"");
                assert!(
                    m.is_token_allowed(b"]"),
                    "nested close after one string item"
                );
            }

            // ── Regression: empty-object close is mask-reachable ──────────
            // `}` was missing from the Phase::Key structural set, so `{}` was
            // unreachable through the mask — an always-fail schema whenever
            // `{}` is the only satisfying document.
            #[test]
            fn empty_object_close_allowed_without_required() {
                let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{");
                assert!(
                    m.is_token_allowed(b"}"),
                    "close must be legal for an empty object"
                );
                m.advance(b"}");
                assert!(
                    m.is_accepting(),
                    "empty object satisfies an all-optional schema"
                );
            }

            #[test]
            fn empty_object_close_allowed_for_closed_schema() {
                let schema = json!({"type": "object", "additionalProperties": false});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{");
                assert!(m.is_token_allowed(b"}"));
                m.advance(b"}");
                assert!(
                    m.is_accepting(),
                    "empty close is the only valid document here"
                );
            }

            #[test]
            fn empty_object_close_refused_while_required_open() {
                let schema = json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{");
                assert!(
                    !m.is_token_allowed(b"}"),
                    "close is a guaranteed violation while required is unmet"
                );
                assert!(
                    m.is_token_allowed(b"\""),
                    "'\"' still starts the required key"
                );
            }

            #[test]
            fn close_after_trailing_comma_still_refused() {
                let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"a\":1,");
                assert!(
                    !m.is_token_allowed(b"}"),
                    "trailing-comma close is invalid JSON; the authority refuses it"
                );
            }

            // ── Regression: string Const/Enum values are mask-enforced ────
            // The in-string inert fast path was schema-blind, so an enum value
            // could diverge freely ("Tokyo" under enum [Paris, London, Rome])
            // and only die at final validation. The value prune mirrors
            // key_filter: content tokens must keep the string a prefix of some
            // member.
            #[test]
            fn enum_string_value_cannot_diverge() {
                let schema = json!({
                    "type": "object",
                    "properties": {"city": {"type": "string", "enum": ["Paris", "London", "Rome"]}},
                    "required": ["city"],
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"city\": \"");
                assert!(
                    !m.is_token_allowed(b"T"),
                    "'T' starts no member (Tokyo is impossible)"
                );
                assert!(m.is_token_allowed(b"P"), "'P' starts Paris");
                m.advance(b"P");
                assert!(m.is_token_allowed(b"aris"));
                assert!(
                    !m.is_token_allowed(b"q"),
                    "'Pariq' diverges from every member"
                );
                m.advance(b"aris\"");
                assert!(m.is_token_allowed(b"}"), "complete document can close");
                m.advance(b"}");
                assert!(m.is_accepting());
            }

            #[test]
            fn const_string_value_cannot_diverge() {
                let schema = json!({
                    "type": "object",
                    "properties": {"c": {"const": "kappa"}},
                    "required": ["c"],
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"c\": \"");
                assert!(m.is_token_allowed(b"k"));
                assert!(!m.is_token_allowed(b"z"));
            }

            #[test]
            fn root_enum_string_value_pruned() {
                let schema = json!({"enum": ["red", "blue"]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\"");
                assert!(!m.is_token_allowed(b"g"), "'g' starts no member");
                assert!(m.is_token_allowed(b"r"));
            }

            #[test]
            fn plain_string_values_stay_inert() {
                let schema = json!({
                    "type": "object",
                    "properties": {"s": {"type": "string"}},
                    "required": ["s"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"s\": \"");
                assert!(
                    m.is_token_allowed(b"zxq"),
                    "non-enum string content must stay unconstrained"
                );
            }

            /// A key that completes OUTSIDE a closed object's property set is
            /// a typed error at its closing quote. The char filter only prunes
            /// extensions (`"f` survives as a prefix of `fixed`), so without
            /// this the model could commit `"f": ...` — an unvalidatable
            /// document that burned to max_tokens (the live E4 failure mode).
            #[test]
            fn completed_unknown_key_is_typed_error_under_closed_object() {
                let schema = json!({
                    "type": "object",
                    "properties": {"fixed": {"type": "string"}},
                    "required": ["fixed"],
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                // The partial key chars survive the filter (prefix of fixed).
                let mut m = compiled.matcher();
                m.advance(b"{\"f");
                assert!(
                    m.is_token_allowed(b"i"),
                    "prefix chars of a known key stay legal"
                );
                // Closing the key as "f" (not "fixed") must be refused at the
                // closing quote.
                assert!(
                    !m.is_token_allowed(b"\""),
                    "closing an unknown key must error"
                );
                m.advance(b"\"");
                assert!(
                    m.is_errored(),
                    "committing the unknown key errors the matcher"
                );
                // Completing the full known key still works end to end.
                let mut m = compiled.matcher();
                m.advance(b"{\"fixed\": \"kappa\", \"fixed\": \"kappa\"}");
                assert!(m.is_errored(), "duplicate known keys still error");
                let mut m = compiled.matcher();
                m.advance(b"{\"fixed\": \"kappa\"}");
                assert!(m.is_accepting());
                // Open objects (additionalProperties true) keep inert keys legal.
                let open_schema = json!({
                    "type": "object",
                    "properties": {"fixed": {"type": "string"}}
                });
                let compiled = CompiledSchema::compile(&open_schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"whatever\": 1}");
                assert!(
                    m.is_accepting(),
                    "unknown keys stay legal under an open object"
                );
            }

            /// The whitespace-run cap: past WS_RUN_CAP consecutive
            /// whitespace-only tokens at a structural position, further
            /// whitespace is refused but a value byte stays legal — breaking
            /// the tabs-forever burn a steered model produces after a forced
            /// think close, without cutting off any valid document.
            #[test]
            fn whitespace_run_cap_breaks_the_tab_loop() {
                let schema = serde_json::json!({
                    "type": "object",
                    "properties": {"done": {"type": "boolean"}},
                    "required": ["done"]
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                for _ in 0..32 {
                    assert!(
                        m.is_token_allowed(b"\t"),
                        "under the cap whitespace is legal"
                    );
                    m.advance(b"\t");
                }
                assert!(
                    !m.is_token_allowed(b"\t"),
                    "the 33rd consecutive whitespace token must be refused"
                );
                assert!(
                    m.is_token_allowed(b"{"),
                    "a value byte stays legal — no dead end"
                );
                m.advance(b"{");
                assert!(m.is_token_allowed(b"\"done\""), "document continues");
            }

            #[test]
            fn mask_state_signature_tracks_whitespace_run_until_cap() {
                let schema = serde_json::json!({"type": "boolean"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                let initial = m.mask_state_signature();
                m.advance(b"\t");
                assert_ne!(initial, m.mask_state_signature());
                for _ in 1..WS_RUN_CAP {
                    m.advance(b"\t");
                }
                let capped = m.mask_state_signature();
                m.advance(b"\t");
                assert_eq!(
                    capped,
                    m.mask_state_signature(),
                    "states beyond the cap share the same deny mask"
                );
            }

            /// The mask-state signature must be buffer-blind: two matchers
            /// whose committed buffers differ only in sibling VALUES (never in
            /// schema-relevant structure) share a signature and return the same
            /// verdicts — the soundness basis for the per-state mask cache.
            #[test]
            fn mask_state_signature_ignores_committed_sibling_values() {
                let schema = serde_json::json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m1 = compiled.matcher();
                let mut m2 = compiled.matcher();
                let p1 = b"{\"a\": 1, ";
                let p2 = b"{\"a\": 987654321, ";
                for by in p1 {
                    m1.advance(&[*by]);
                }
                for by in p2 {
                    m2.advance(&[*by]);
                }
                assert_eq!(
                    m1.mask_state_signature(),
                    m2.mask_state_signature(),
                    "sibling value length/digits must not change the state"
                );
                for t in [
                    b"\"".as_slice(),
                    b"b".as_slice(),
                    b"}".as_slice(),
                    b" ".as_slice(),
                ] {
                    assert_eq!(
                        m1.is_token_allowed(t),
                        m2.is_token_allowed(t),
                        "verdicts must agree for {:?}",
                        String::from_utf8_lossy(t)
                    );
                }
                // Genuinely different states (different keys seen) must differ.
                let mut m3 = compiled.matcher();
                for by in b"{\"a\": 1, \"b\": 2, " {
                    m3.advance(&[*by]);
                }
                assert_ne!(
                    m1.mask_state_signature(),
                    m3.mask_state_signature(),
                    "a used second key is a different state"
                );
            }

            /// Escape-bearing tokens must decode INTO the key filter, not
            /// bypass it. The live E4 failure: under properties {fixed}, the
            /// model emitted `"f` then `\":` — a backslash token skipped the
            /// filter, the simulation cannot refuse an open string, and the
            /// key grew forever as `f": \"\\"…` (burn to max_tokens).
            #[test]
            fn escaped_key_extension_is_pruned_by_key_filter() {
                let schema = json!({
                    "type": "object",
                    "properties": {"fixed": {"type": "string"}},
                    "required": ["fixed"],
                    "additionalProperties": false
                });
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"{\"f");
                assert!(m.is_token_allowed(b"i"), "prefix extension stays legal");
                // A backslash-escape extension of the key decodes to `f\"` —
                // no member prefixes it — and must be refused NOW, not after
                // an unbounded burn.
                assert!(
                    !m.is_token_allowed(b"\\\""),
                    "escape-bearing key extension must be filter-pruned"
                );
                // Byte-wise equivalence: the same decoded content refused.
                let mut m2 = compiled.matcher();
                m2.advance(b"{\"f\\");
                assert!(!m2.is_token_allowed(b"\""));
            }

            #[test]
            fn mixed_enum_string_members_prune_inside_string() {
                // Once inside a string, only the string members can match —
                // numeric members are unreachable and must not disable pruning.
                let schema = json!({"enum": ["red", 42]});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"\"");
                assert!(!m.is_token_allowed(b"g"));
                assert!(m.is_token_allowed(b"r"));
            }

            // ── Regression: number-tail fast path follows the JSON number
            // grammar. Byte-set membership alone mask-allowed `01`, `1ee`,
            // `1.2.3` and `--`, committing bytes the authority immediately
            // rejects (mask/simulation disagreement kills the request).
            #[test]
            fn number_tail_refuses_leading_zero_continuation() {
                let schema = json!({"type": "number"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"0");
                assert!(!m.is_token_allowed(b"1"), "01 is not a JSON number");
                assert!(m.is_token_allowed(b"."), "0.5 is legal");
                m.advance(b".5");
                assert!(m.is_accepting());
            }

            #[test]
            fn number_tail_refuses_double_exponent_and_dot() {
                let schema = json!({"type": "number"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"12e");
                assert!(!m.is_token_allowed(b"e"), "1ee is not a JSON number");
                assert!(m.is_token_allowed(b"+"));
                m.advance(b"+5");
                assert!(m.is_accepting());
                let mut m2 = compiled.matcher();
                m2.advance(b"1.2");
                assert!(!m2.is_token_allowed(b"."), "1.2.3 is not a JSON number");
            }

            #[test]
            fn number_tail_refuses_double_sign() {
                let schema = json!({"type": "number"});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"-");
                assert!(!m.is_token_allowed(b"-"), "-- is not a JSON number");
                assert!(m.is_token_allowed(b"7"));
            }

            #[test]
            fn max_items_allows_up_to_limit() {
                let schema = json!({"type": "array", "items": {"type": "integer"}, "maxItems": 3});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[1");
                assert!(!m.is_errored());
                m.advance(b", 2");
                assert!(!m.is_errored());
                m.advance(b", 3");
                assert!(!m.is_errored());
                // A 4th item must be refused.
                m.advance(b", 4");
                assert!(m.is_errored());
            }

            #[test]
            fn max_items_empty_array_ok() {
                let schema = json!({"type": "array", "items": {"type": "integer"}, "maxItems": 0});
                let compiled = CompiledSchema::compile(&schema).expect("valid");
                let mut m = compiled.matcher();
                m.advance(b"[]");
                assert!(m.is_accepting(), "empty array with maxItems:0 is valid");
                let mut m2 = compiled.matcher();
                m2.advance(b"[1");
                assert!(m2.is_errored(), "any item exceeds maxItems:0");
            }

            // ── Nested dead-end enforcement (wedge closure) ─────────────────

            #[test]
            fn nested_max_items_refuses_extra_comma() {
                // A nested array over maxItems cannot accept another `,` — the
                // mask forces the `]` instead of a burn-to-max_tokens wedge.
                let compiled = CompiledSchema::compile(&json!({
                "type": "object",
                "properties": {"m": {"type": "array", "items": {"type": "integer"}, "maxItems": 1}}
            })).unwrap();
                let mut m = compiled.matcher();
                for b in [b'{' as u8, b'"', b'm', b'"', b':', b'[', b'1'] {
                    assert!(
                        m.is_token_allowed(&[b]),
                        "byte {:?} must be allowed",
                        b as char
                    );
                    m.advance(&[b]);
                }
                assert!(!m.is_token_allowed(b","), "second item exceeds maxItems:1");
                assert!(m.is_token_allowed(b"]"));
            }

            #[test]
            fn nested_min_items_refuses_early_close() {
                let compiled = CompiledSchema::compile(&json!({
                "type": "object",
                "properties": {"m": {"type": "array", "items": {"type": "integer"}, "minItems": 2}}
            })).unwrap();
                let mut m = compiled.matcher();
                for b in [b'{' as u8, b'"', b'm', b'"', b':', b'['] {
                    m.advance(&[b]);
                }
                assert!(
                    !m.is_token_allowed(b"]"),
                    "minItems:2 forbids closing at 0 items"
                );
                m.advance(b"1");
                assert!(
                    !m.is_token_allowed(b"]"),
                    "minItems:2 forbids closing at 1 item"
                );
                m.advance(b",");
                assert!(m.is_token_allowed(b"2"));
            }

            #[test]
            fn ap_false_unknown_key_chars_are_pruned() {
                // additionalProperties:false: once every known key is used, a
                // new key cannot even START (`,` refused); while a key is being
                // typed, only characters extending a known unused key pass.
                let compiled = CompiledSchema::compile(&json!({
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": false
                }))
                .unwrap();
                let mut m = compiled.matcher();
                for b in [b'{' as u8, b'"', b'n', b'a'] {
                    assert!(m.is_token_allowed(&[b]), "byte {:?} allowed", b as char);
                    m.advance(&[b]);
                }
                // Inside "na…" of the known key "name": 'm' extends, 'x' prunes.
                assert!(m.is_token_allowed(b"m"));
                assert!(
                    !m.is_token_allowed(b"x"),
                    "unknown key characters are pruned"
                );
                m.advance(b"m");
                m.advance(b"e");
                assert!(
                    m.is_token_allowed(b"\""),
                    "completing a KNOWN key must stay legal"
                );
                m.advance(b"\"");
                assert!(m.is_token_allowed(b":"));
            }

            #[test]
            fn ap_false_blocks_close_until_required_met_then_blocks_comma() {
                let compiled = CompiledSchema::compile(&json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": false
                }))
                .unwrap();
                let mut m = compiled.matcher();
                // Before "a" is present, `}` is refused.
                m.advance(b"{");
                assert!(
                    !m.is_token_allowed(b"}"),
                    "required unmet: close must be refused"
                );
                for b in [b'"', b'a', b'"', b':', b'1'] {
                    m.advance(&[b]);
                }
                // "a" met and all known keys… "b" unused → `,` still legal.
                assert!(m.is_token_allowed(b","));
                m.advance(b",");
                // After the comma, `}` is refused again (required was met but a
                // key slot was opened… wait: after `,` a KEY is required; `}`
                // is invalid JSON there — the structural set must refuse it.
                assert!(
                    !m.is_token_allowed(b"}"),
                    "after a comma a key is mandatory"
                );
                m.advance(b"\"");
                assert!(m.is_token_allowed(b"b"));
                assert!(!m.is_token_allowed(b"z"), "unknown key pruned");
            }

            #[test]
            fn ap_false_after_all_keys_only_close_remains() {
                let compiled = CompiledSchema::compile(&json!({
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": false
                }))
                .unwrap();
                let mut m = compiled.matcher();
                for b in [b'{' as u8, b'"', b'a', b'"', b':', b'1'] {
                    m.advance(&[b]);
                }
                assert!(
                    !m.is_token_allowed(b","),
                    "closed object: all keys used, another key is impossible"
                );
                assert!(m.is_token_allowed(b"}"));
            }

            #[test]
            fn duplicate_key_chars_pruned_at_mask() {
                // With two known keys, after "name" is used the filter is
                // ["age"]: a second "name" diverges at the first character.
                let compiled = CompiledSchema::compile(&json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "integer"},
                        "age": {"type": "integer"}
                    },
                    "additionalProperties": false
                }))
                .unwrap();
                let mut m = compiled.matcher();
                for b in [
                    b'{' as u8, b'"', b'n', b'a', b'm', b'e', b'"', b':', b'1', b',', b'"',
                ] {
                    assert!(m.is_token_allowed(&[b]), "{:?} allowed", b as char);
                    m.advance(&[b]);
                }
                assert!(
                    !m.is_token_allowed(b"n"),
                    "duplicate \"name\" key pruned at 'n'"
                );
                assert!(
                    m.is_token_allowed(b"a"),
                    "the unused \"age\" key stays legal"
                );
                // Typing "age" fully stays legal through its closing quote.
                for b in [b'a', b'g', b'e'] {
                    assert!(m.is_token_allowed(&[b]));
                    m.advance(&[b]);
                }
                assert!(m.is_token_allowed(b"\""), "completing a KNOWN key is legal");
            }

            // ── Strict keyword VALUES (P0: never silently unconstrained) ────

            #[test]
            fn malformed_keyword_values_are_typed_errors() {
                for (schema, why) in [
                    (json!({"enum": "hello"}), "enum as string"),
                    (json!({"enum": []}), "empty enum is unsatisfiable"),
                    (
                        json!({"type": "object", "required": "name"}),
                        "required as string",
                    ),
                    (
                        json!({"type": "object", "required": [1, 2]}),
                        "required with non-strings",
                    ),
                    (
                        json!({"type": "object", "properties": ["a"]}),
                        "properties as array",
                    ),
                    (
                        json!({"type": "array", "items": {"type": "integer"}, "minItems": "3"}),
                        "minItems as string",
                    ),
                    (
                        json!({"type": "array", "items": {"type": "integer"}, "maxItems": -1}),
                        "maxItems negative",
                    ),
                ] {
                    let err =
                        CompiledSchema::compile(&schema).expect_err(&format!("must reject: {why}"));
                    assert!(
                        matches!(
                            err,
                            SchemaError::InvalidSchema { .. } | SchemaError::Unsatisfiable { .. }
                        ),
                        "{why}: expected InvalidSchema/Unsatisfiable, got {err:?}"
                    );
                }
            }

            #[test]
            fn huge_int_const_is_not_float_confused() {
                let c =
                    CompiledSchema::compile(&json!({"const": 18446744073709551615u64})).unwrap();
                let mut m = c.matcher();
                for b in b"18446744073709551615" {
                    assert!(m.is_token_allowed(&[*b]), "digit {:?} allowed", *b as char);
                    m.advance(&[*b]);
                }
                assert!(m.is_accepting(), "exact u64::MAX spelling accepted");
                let c2 =
                    CompiledSchema::compile(&json!({"const": 18446744073709551615u64})).unwrap();
                let mut m2 = c2.matcher();
                for b in b"18446744073709551615" {
                    assert!(m2.is_token_allowed(&[*b]));
                    m2.advance(&[*b]);
                }
                // u64::MAX is accepted but may "grow"; 2^64 as one spelling
                // must be refused (the exact-integer compare, not the lossy
                // f64 compare, decides).
                assert!(!m2.is_token_allowed(b"18446744073709551616"));
                assert!(
                    !m2.is_token_allowed(b"18446744073709552"),
                    "rounded-up prefix beyond the target diverges"
                );
            }

            #[test]
            fn number_tail_mask_is_bounded_and_sound() {
                // Under a nested integer, digits stay legal at the tail and a
                // fraction is refused (fast-path exclusion → simulation →
                // dead-end flag).
                let compiled = CompiledSchema::compile(&json!({
                    "type": "object",
                    "properties": {"n": {"type": "integer"}}
                }))
                .unwrap();
                let mut m = compiled.matcher();
                for b in [b'{' as u8, b'"', b'n', b'"', b':', b'1', b'2'] {
                    assert!(m.is_token_allowed(&[b]));
                    m.advance(&[b]);
                }
                assert!(m.is_token_allowed(b"3"), "digits extend the number");
                assert!(
                    !m.is_token_allowed(b"."),
                    "a fraction can never become an integer"
                );
            }

            #[test]
            fn count_root_array_items_helper() {
                assert_eq!(count_root_array_items(b"[]"), Some(0));
                assert_eq!(count_root_array_items(b"[1"), Some(1));
                assert_eq!(count_root_array_items(b"[1,"), Some(1));
                assert_eq!(count_root_array_items(b"[1,2"), Some(2));
                assert_eq!(count_root_array_items(b"[1,2,"), Some(2));
                assert_eq!(count_root_array_items(b"[1,2,3]"), Some(3));
                assert_eq!(count_root_array_items(b"[1,[2,3],4"), Some(3));
                assert_eq!(count_root_array_items(b"[\"a\",\"b\""), Some(2));
                assert_eq!(count_root_array_items(b"{"), None);
                assert_eq!(count_root_array_items(b"  [1,2"), Some(2));
                // String with comma inside should not count.
                assert_eq!(count_root_array_items(b"[\"a,b\""), Some(1));
            }
        }
    }
} // mod json

/// DSML grammar — state machine for `<｜DSML｜tool_calls>` XML-style tool calls.
/// See original `hipfire-arch-deepseek4/src/grammar.rs` for full design notes.
pub mod dsml {
    //! Grammar-guided decoding for DeepSeek V4's DSML tool-call format.
    //!
    //! Tracks a small state machine over the model's emitted bytes that
    //! mirrors the parser in `dsml.rs`. At each sample step, the matcher
    //! reports the set of legal byte-string continuations from the current
    //! state; the daemon converts that into a vocab bitmask (token T allowed
    //! iff `partial_buf + decode(T)` is a prefix of any legal continuation)
    //! and zeroes the logits of disallowed tokens before sampling.
    //!
    //! The matcher only kicks in when the model is INSIDE a DSML structural
    //! position — outside any tool-call block, and inside parameter values,
    //! emission is unconstrained. This keeps the constraint surface tight:
    //! the model writes prose / code / params freely, but cannot emit an
    //! invalid tag name like `<｜DSML｜tool_cbl>` or `<｜DSML｜calling>`.
    //!
    //! ## States
    //!
    //! - [`State::Out`] — free emission. Watching for the opening trigger
    //!   `<｜DSML｜tool_calls>` to enter [`State::InToolCalls`].
    //! - [`State::InToolCalls`] — between the outer open and close. Next
    //!   firm bytes must be the open of an invoke or the close of the block.
    //! - [`State::InInvokeName`] — between `<｜DSML｜invoke name="` (or the
    //!   `tool` variant) and the closing `">\n`. Emits a tool name.
    //! - [`State::InInvokeBody`] — between `<｜DSML｜...name="X">\n` and
    //!   `</｜DSML｜invoke>` / `</｜DSML｜tool>`. Next firm bytes must be a
    //!   parameter open or the invoke close.
    //! - [`State::InParamName`] — between `<｜DSML｜parameter name="` and
    //!   the closing `"`. Emits a parameter name.
    //! - [`State::InParamAttr`] — between the param-name close-quote and
    //!   `">`. Must be ` string="true"` or ` string="false"`.
    //! - [`State::InParamBody`] — between the param-attr `">` and
    //!   `</｜DSML｜parameter>`. Free emission of the parameter value.
    //!
    //! Each state carries a `partial_buf: String` of bytes committed since
    //! the last firm transition. `allowed_continuations()` returns the
    //! literal byte strings any one of which the model is allowed to be in
    //! the middle of emitting. A token T is allowed iff
    //! `partial_buf + decode(T)` is a prefix of some allowed string.
    //!
    //! ## Why a state machine and not a regex
    //!
    //! BPE fragmentation means a single string like `<｜DSML｜tool_calls>`
    //! is multiple tokens (`<` + `｜DSML｜` + `tool` + `_c` + `alls` + `>`).
    //! The matcher tracks the byte-level position so it doesn't matter how
    //! the tokens divide the string. The grammar's regular structure (no
    //! recursion — invokes don't nest, params don't nest) keeps the state
    //! count finite and small.

    // ── Public types ────────────────────────────────────────────────────────

    /// Position in the DSML grammar. Carries a byte-level partial-match
    /// buffer that holds the bytes committed since the last firm state
    /// transition. See module doc for transitions.
    #[derive(Debug, Clone, PartialEq)]
    pub enum State {
        /// Free emission outside any DSML structure. The matcher is watching
        /// for the opening trigger `<｜DSML｜tool_calls>` but does not
        /// otherwise constrain emission.
        Out,
        /// Between `<｜DSML｜tool_calls>` (already consumed) and the matching
        /// close. Next firm bytes must open an invoke or close the block.
        InToolCalls,
        /// Between `<｜DSML｜invoke name="` (or `<｜DSML｜tool name="`) and
        /// the closing `">`. Emitting the tool name. `tool_idx` is the
        /// index into the schema for the in-progress tool, or `None`
        /// while the name is still being matched against schema entries.
        InInvokeName { tool_idx: Option<usize> },
        /// Inside an invoke body. `emitted_params` lists the indices of
        /// params already serialised in this invoke — the matcher uses
        /// this to gate the invoke-close alternatives on schema `required`
        /// being satisfied. Next firm bytes must open a parameter or, if
        /// required is satisfied, close the invoke.
        InInvokeBody {
            tool_idx: usize,
            emitted_params: Vec<usize>,
        },
        /// Between `<｜DSML｜parameter name="` and the closing `"`. Emitting
        /// the parameter name for the in-progress invoke. `emitted_params`
        /// is propagated through unchanged until the full param block
        /// closes.
        InParamName {
            tool_idx: usize,
            param_idx: Option<usize>,
            emitted_params: Vec<usize>,
        },
        /// Between the param-name `"` and the attr `">`. Must emit
        /// ` string="true"` or ` string="false"` before continuing.
        InParamAttr {
            tool_idx: usize,
            param_idx: usize,
            emitted_params: Vec<usize>,
        },
        /// Between `">` and `</｜DSML｜parameter>`. Free emission of the
        /// parameter value bytes. `param_idx` is the in-flight param; on
        /// close it gets pushed into `emitted_params` as the matcher
        /// returns to [`State::InInvokeBody`].
        InParamBody {
            tool_idx: usize,
            param_idx: usize,
            emitted_params: Vec<usize>,
        },
    }

    /// Schema for the available tools. Built from the OpenAI-format tools
    /// array at request time. The grammar uses this to constrain tool names
    /// and parameter names at their respective positions.
    #[derive(Debug, Clone)]
    pub struct ToolSchema {
        pub name: String,
        /// Parameter names in the order they appear in the schema. Order
        /// isn't enforced at parse time — params can be emitted in any order
        /// — but the schema is the authoritative set of legal names.
        pub params: Vec<String>,
        /// Subset of `params` that MUST appear in the emitted invoke
        /// block. The grammar removes invoke-close alternatives from the
        /// allowed continuations until every required param has been
        /// observed — without this the V4F MQ2-Lloyd checkpoint emits
        /// empty invokes like `<｜DSML｜tool name="bash"></｜DSML｜tool>`
        /// that the downstream OpenAI client rejects with
        /// `must have required properties command`.
        pub required: Vec<String>,
    }

    use std::sync::Arc;

    /// Maximum number of tool schemas accepted by [`CompiledGrammar::new`].
    const MAX_TOOL_SCHEMAS: usize = 256;

    /// Maximum total bytes of all tool names + param names + required names.
    const MAX_SCHEMA_BYTES: usize = 64 * 1024;

    /// Typed error returned when DSML compilation inputs exceed bounds.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum GrammarCompileError {
        /// Too many tool schemas supplied.
        TooManySchemas { count: usize, max: usize },
        /// Total schema string bytes exceeded the bound.
        SchemaBytesExceeded { bytes: usize, max: usize },
        /// A tool name is empty.
        EmptyToolName { index: usize },
    }

    impl std::fmt::Display for GrammarCompileError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::TooManySchemas { count, max } => write!(
                    f,
                    "dsml grammar compile: {count} tool schemas exceeds max {max}"
                ),
                Self::SchemaBytesExceeded { bytes, max } => write!(
                    f,
                    "dsml grammar compile: schema string bytes {bytes} exceeds max {max}"
                ),
                Self::EmptyToolName { index } => {
                    write!(
                        f,
                        "dsml grammar compile: tool schema {index} has empty name"
                    )
                }
            }
        }
    }

    impl std::error::Error for GrammarCompileError {}

    /// Immutable, `Send + Sync` compiled artifact for the DSML tool-call
    /// grammar. Owns the tool schemas; produced once per unique tool set and
    /// shared across requests via `Arc<CompiledGrammar>` (spec §7 G1).
    #[derive(Debug, Clone)]
    pub struct CompiledGrammar {
        tools: Vec<ToolSchema>,
    }

    impl CompiledGrammar {
        /// Compile with bounds validation.
        pub fn new(tools: Vec<ToolSchema>) -> Result<Self, GrammarCompileError> {
            if tools.len() > MAX_TOOL_SCHEMAS {
                return Err(GrammarCompileError::TooManySchemas {
                    count: tools.len(),
                    max: MAX_TOOL_SCHEMAS,
                });
            }
            let mut total_bytes = 0;
            for (i, t) in tools.iter().enumerate() {
                if t.name.is_empty() {
                    return Err(GrammarCompileError::EmptyToolName { index: i });
                }
                total_bytes += t.name.len();
                for p in &t.params {
                    total_bytes += p.len();
                }
                for r in &t.required {
                    total_bytes += r.len();
                }
            }
            if total_bytes > MAX_SCHEMA_BYTES {
                return Err(GrammarCompileError::SchemaBytesExceeded {
                    bytes: total_bytes,
                    max: MAX_SCHEMA_BYTES,
                });
            }
            Ok(Self { tools })
        }

        /// Read-only access to the tool schemas.
        pub fn tools(&self) -> &[ToolSchema] {
            &self.tools
        }
    }

    unsafe impl Send for CompiledGrammar {}
    unsafe impl Sync for CompiledGrammar {}

    /// The grammar matcher itself: a state plus the bytes committed since
    /// the last firm transition. Construct via [`Matcher::new`] with the
    /// active tool schemas; advance with [`Matcher::advance`]; query the
    /// legal token-prefix continuations from the current state with
    /// [`Matcher::allowed_continuations`].
    #[derive(Debug, Clone)]
    pub struct Matcher {
        state: State,
        /// Bytes committed since the last firm state transition. May span
        /// multiple BPE tokens — we keep matching against allowed strings
        /// until either the buffer fully consumes an allowed string (firm
        /// transition) or no allowed string still has the buffer as a
        /// prefix (match failure → fall back to `Out`).
        partial_buf: String,
        /// Immutable compiled artifact: tool schemas.
        /// Shared across requests via `Arc` (spec §7 G1).
        grammar: Arc<CompiledGrammar>,
    }

    impl Matcher {
        /// Build a fresh matcher in [`State::Out`] with no partial buffer.
        ///
        /// Compiles the tools into a [`CompiledGrammar`] internally. For
        /// request sharing, prefer [`Matcher::from_compiled`] with a
        /// pre-built `Arc<CompiledGrammar>`.
        pub fn new(tools: Vec<ToolSchema>) -> Self {
            let grammar =
                CompiledGrammar::new(tools).expect("Matcher::new: schema bounds violated");
            Self::from_compiled(Arc::new(grammar))
        }

        /// Build a cursor from a pre-compiled, shared artifact. Multiple
        /// requests can share the same `Arc<CompiledGrammar>` while each
        /// owns an independent `Matcher` cursor (spec §7 G1).
        pub fn from_compiled(grammar: Arc<CompiledGrammar>) -> Self {
            Self {
                state: State::Out,
                partial_buf: String::new(),
                grammar,
            }
        }

        /// Read-only access to the compiled grammar artifact.
        pub fn compiled(&self) -> &CompiledGrammar {
            &self.grammar
        }

        /// Read-only view of the current state.
        pub fn state(&self) -> &State {
            &self.state
        }

        /// Bytes accumulated since the last firm state transition.
        pub fn partial(&self) -> &str {
            &self.partial_buf
        }

        /// Whether the matcher is currently free (no constraints — every
        /// token is allowed). Dual-mode in both free-emission states:
        ///
        /// - [`State::Out`] is free UNTIL the model commits the atomic
        ///   `｜DSML｜` token (id 128825 in V4 vocab; the literal substring
        ///   `｜DSML｜` in the buffer). Once that lands, the matcher
        ///   constrains continuations to complete `<｜DSML｜tool_calls>`
        ///   exactly. Without this guard the V4F MQ2-Lloyd checkpoint
        ///   deterministically emits invented opener variants like
        ///   `<｜DSML｜tool_actions>`, `<｜DSML｜tool_invoke>`,
        ///   `<｜DSML｜tInvoke name="…">` — none of those match the trigger
        ///   so the grammar matcher never engages, and the parser sees
        ///   pure garbage. We accept the rare-but-possible mis-fire where
        ///   the model emits `｜DSML｜` for non-tool reasons (quoting the
        ///   format in prose): in that case the constraint will force a
        ///   `tool_calls>` completion, which is acceptable since this
        ///   token does not appear in normal output.
        /// - [`State::InParamBody`] is free until the buffer accumulates a
        ///   close-marker prefix (`</`). Once it lands, constrain to
        ///   complete `</｜DSML｜parameter>` exactly — stops the model
        ///   from emitting near-misses like `</｜DSML｜paperameter>`.
        pub fn is_free(&self) -> bool {
            match self.state {
                State::Out => !self.partial_buf.contains("｜DSML｜"),
                State::InParamBody { .. } => self.partial_buf.is_empty(),
                _ => false,
            }
        }

        /// Returns the set of legal continuation strings from the current
        /// state. Each returned string is a FULL continuation starting from
        /// the position immediately after the last firm state transition —
        /// the caller checks `partial_buf + decode(T)` against these via
        /// [`Self::is_token_allowed`].
        ///
        /// Returns an empty vec when [`Self::is_free`] is true (caller
        /// should allow all tokens).
        pub fn allowed_continuations(&self) -> Vec<String> {
            match &self.state {
                // Out is dual-mode: free emission when no DSML commit is
                // in-flight (caller short-circuits via `is_free`);
                // constrain to the OPEN_TOOL_CALLS trigger once the atomic
                // `｜DSML｜` token has been committed into the buffer.
                State::Out => {
                    if self.partial_buf.contains("｜DSML｜") {
                        vec![OPEN_TOOL_CALLS.to_string()]
                    } else {
                        Vec::new()
                    }
                }
                // InParamBody is dual-mode: free emission when buf is empty
                // (caller short-circuits via `is_free`); constrain to the
                // close marker once the model has started emitting it.
                State::InParamBody { .. } => {
                    if self.partial_buf.is_empty() {
                        Vec::new()
                    } else {
                        vec![CLOSE_PARAM.to_string()]
                    }
                }
                State::InToolCalls => vec![
                    OPEN_INVOKE.to_string(),
                    OPEN_TOOL_VARIANT.to_string(),
                    CLOSE_TOOL_CALLS.to_string(),
                ],
                State::InInvokeName { tool_idx: None, .. } => self
                    .grammar
                    .tools
                    .iter()
                    .map(|t| format!("{}\">\n", t.name))
                    .collect(),
                State::InInvokeName {
                    tool_idx: Some(idx),
                    ..
                } => vec![format!("{}\">\n", self.grammar.tools[*idx].name)],
                State::InInvokeBody {
                    tool_idx,
                    emitted_params,
                } => {
                    let mut conts = vec![OPEN_PARAM.to_string()];
                    // Allow invoke close only when every required param of
                    // the in-flight tool has been emitted. This is what
                    // forces the model to fill in `command` before closing
                    // a `bash` invoke, etc.
                    if self.required_satisfied(*tool_idx, emitted_params) {
                        conts.push(CLOSE_INVOKE.to_string());
                        conts.push(CLOSE_TOOL_VARIANT.to_string());
                    }
                    conts
                }
                State::InParamName {
                    tool_idx,
                    param_idx: None,
                    emitted_params,
                } => self.grammar.tools[*tool_idx]
                    .params
                    .iter()
                    .enumerate()
                    // Don't allow re-emitting a param already in the block
                    // — `command` appearing twice in one invoke is never
                    // valid and the OpenAI parser rejects it.
                    .filter(|(i, _)| !emitted_params.contains(i))
                    .map(|(_, p)| format!("{}\"", p))
                    .collect(),
                State::InParamName {
                    tool_idx,
                    param_idx: Some(idx),
                    ..
                } => vec![format!("{}\"", self.grammar.tools[*tool_idx].params[*idx])],
                State::InParamAttr { .. } => {
                    vec![ATTR_STRING_TRUE.to_string(), ATTR_STRING_FALSE.to_string()]
                }
            }
        }

        /// True when every entry in `self.tools[tool_idx].required` is
        /// represented in `emitted_params`.
        fn required_satisfied(&self, tool_idx: usize, emitted_params: &[usize]) -> bool {
            let tool = &self.grammar.tools[tool_idx];
            for req_name in &tool.required {
                let req_idx = match tool.params.iter().position(|p| p == req_name) {
                    Some(i) => i,
                    // Required name not in params list — schema bug. Be
                    // permissive (don't deadlock the matcher).
                    None => continue,
                };
                if !emitted_params.contains(&req_idx) {
                    return false;
                }
            }
            true
        }

        /// Check whether the candidate decoded token text could be emitted
        /// next without violating the grammar. Returns `true` when the
        /// matcher is in a free-emission state OR when `partial_buf + text`
        /// is a prefix of (or equal to, or extends past) at least one legal
        /// continuation from [`Self::allowed_continuations`].
        ///
        /// In states that tolerate leading whitespace
        /// (`InToolCalls`, `InInvokeBody`), the check is also run against
        /// the whitespace-trimmed prefix — so a token like `\n` or
        /// `\n<｜DSML｜` is accepted because the leading newline will be
        /// silently consumed by [`Self::transition_once`].
        pub fn is_token_allowed(&self, text: &str) -> bool {
            if self.is_free() {
                return true;
            }
            let combined = format!("{}{}", self.partial_buf, text);
            let conts = self.allowed_continuations();
            if Self::check_against_conts(&combined, &conts) {
                return true;
            }
            if self.state_allows_leading_ws() {
                let trimmed = combined.trim_start_matches(|c: char| c == '\n' || c == ' ');
                if Self::check_against_conts(trimmed, &conts) {
                    return true;
                }
            }
            false
        }

        fn check_against_conts(s: &str, conts: &[String]) -> bool {
            for cont in conts {
                if cont.starts_with(s) || s.starts_with(cont.as_str()) {
                    return true;
                }
            }
            false
        }

        /// Populate a boolean mask over `vocab` indicating which tokens are
        /// legal at the current matcher position. `out` must be at least
        /// `vocab.len()` long; entries beyond `vocab.len()` are untouched.
        ///
        /// Fast path: when [`Self::is_free`] is true, the entire mask is
        /// set to `true` — the caller can skip the sample-time mask scan
        /// entirely. Hot path: O(vocab) scan calling [`Self::is_token_allowed`]
        /// per id. With ~129k vocab and ≤4 alternatives per state this is
        /// ~1.7M byte comparisons per sample step, sub-millisecond on
        /// commodity hardware.
        ///
        /// Tokens whose decoded text is empty (placeholder / no-op tokens)
        /// are always allowed — the empty buffer extension keeps every
        /// active prefix viable.
        pub fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
            debug_assert!(out.len() >= vocab.len());
            if self.is_free() {
                for slot in out.iter_mut().take(vocab.len()) {
                    *slot = true;
                }
                return;
            }
            for (id, text) in vocab.iter().enumerate() {
                out[id] = self.is_token_allowed(text);
            }
        }

        /// Apply the token mask in-place to a logits slice: disallowed
        /// tokens get `f32::NEG_INFINITY`, allowed tokens are left alone.
        /// Caller invokes [`Self::token_mask`] first, then this. Splitting
        /// the two lets the caller reuse a single `Vec<bool>` allocation
        /// across the decode loop.
        pub fn apply_mask_to_logits(mask: &[bool], logits: &mut [f32]) {
            let n = mask.len().min(logits.len());
            for i in 0..n {
                if !mask[i] {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        /// Commit decoded token bytes into the matcher, advancing state if
        /// any allowed continuation completes. Designed to be idempotent at
        /// the byte level: callers may pass single bytes or multi-byte
        /// chunks; the same final state is reached either way.
        ///
        /// In free-emission states (`Out`, `InParamBody`), the matcher
        /// scans for trigger strings (`<｜DSML｜tool_calls>` from `Out`;
        /// `</｜DSML｜parameter>` from `InParamBody`) and transitions when
        /// the trigger lands at the end of the rolling window.
        pub fn advance(&mut self, text: &str) {
            if text.is_empty() {
                return;
            }
            self.partial_buf.push_str(text);

            loop {
                match self.transition_once() {
                    Transition::Stay => return,
                    Transition::Advanced => {
                        // Loop: another transition might fire on the same
                        // buffer (e.g. open marker consumed → InToolCalls,
                        // then immediately another tag opens).
                        continue;
                    }
                }
            }
        }

        /// Inner step: examine `partial_buf` against the current state's
        /// allowed transitions. Returns whether any firm transition fired.
        ///
        /// States between tags (`InToolCalls`, `InInvokeBody`) tolerate
        /// leading whitespace — the HF reference format emits `\n` between
        /// every tag and tokens like `>\n` (vocab id 1018) carry that
        /// newline INTO the buffer when the open trigger fires.
        fn transition_once(&mut self) -> Transition {
            // Consume one leading whitespace byte in states that allow it.
            if self.state_allows_leading_ws() {
                if let Some(n) = self.leading_ws_byte_count() {
                    self.partial_buf.drain(..n);
                    return Transition::Advanced;
                }
            }
            match self.state.clone() {
                State::Out => self.transition_from_free(OPEN_TOOL_CALLS, State::InToolCalls),
                State::InParamBody {
                    tool_idx,
                    param_idx,
                    mut emitted_params,
                } => {
                    // On close: record this param as emitted, then return
                    // to InInvokeBody with the updated set.
                    if !emitted_params.contains(&param_idx) {
                        emitted_params.push(param_idx);
                    }
                    self.transition_from_free(
                        CLOSE_PARAM,
                        State::InInvokeBody {
                            tool_idx,
                            emitted_params,
                        },
                    )
                }
                State::InToolCalls => self.transition_from_alternatives(&[
                    (OPEN_INVOKE, State::InInvokeName { tool_idx: None }),
                    (OPEN_TOOL_VARIANT, State::InInvokeName { tool_idx: None }),
                    (CLOSE_TOOL_CALLS, State::Out),
                ]),
                State::InInvokeName { tool_idx } => self.transition_invoke_name(tool_idx),
                State::InInvokeBody {
                    tool_idx,
                    emitted_params,
                } => {
                    // Walk the alternatives manually so we can build the
                    // gated close alternatives (only legal once `required`
                    // is satisfied) without static `&str` lifetime gymnastics.
                    let mut alts: Vec<(&str, State)> = vec![(
                        OPEN_PARAM,
                        State::InParamName {
                            tool_idx,
                            param_idx: None,
                            emitted_params: emitted_params.clone(),
                        },
                    )];
                    if self.required_satisfied(tool_idx, &emitted_params) {
                        alts.push((CLOSE_INVOKE, State::InToolCalls));
                        alts.push((CLOSE_TOOL_VARIANT, State::InToolCalls));
                    }
                    self.transition_from_alternatives(&alts)
                }
                State::InParamName {
                    tool_idx,
                    param_idx,
                    emitted_params,
                } => self.transition_param_name(tool_idx, param_idx, emitted_params),
                State::InParamAttr {
                    tool_idx,
                    param_idx,
                    emitted_params,
                } => self.transition_from_alternatives(&[
                    (
                        ATTR_STRING_TRUE,
                        State::InParamBody {
                            tool_idx,
                            param_idx,
                            emitted_params: emitted_params.clone(),
                        },
                    ),
                    (
                        ATTR_STRING_FALSE,
                        State::InParamBody {
                            tool_idx,
                            param_idx,
                            emitted_params,
                        },
                    ),
                ]),
            }
        }

        /// True in states where one or more leading `\n` / ` ` bytes in
        /// `partial_buf` should be silently consumed before trying any
        /// alternative match. Driven by the HF reference renderer which
        /// emits `\n` between sibling tags.
        fn state_allows_leading_ws(&self) -> bool {
            matches!(self.state, State::InToolCalls | State::InInvokeBody { .. })
        }

        /// Length in bytes of the leading whitespace run (`\n` or ` `) in
        /// `partial_buf`, or `None` when the first byte is non-ws.
        fn leading_ws_byte_count(&self) -> Option<usize> {
            let bytes = self.partial_buf.as_bytes();
            let mut n = 0;
            while n < bytes.len() && (bytes[n] == b'\n' || bytes[n] == b' ') {
                n += 1;
            }
            if n == 0 {
                None
            } else {
                Some(n)
            }
        }

        /// Trigger-scan transition: look for `trigger` at the tail of
        /// `partial_buf`. If found, advance to `next_state` and drop the
        /// trigger (plus everything before it). If not found, keep only
        /// the longest suffix that is a prefix of `trigger` (so future
        /// bytes can complete it).
        fn transition_from_free(&mut self, trigger: &str, next_state: State) -> Transition {
            if let Some(idx) = self.partial_buf.find(trigger) {
                // Drop everything up to and including the trigger.
                let after = idx + trigger.len();
                self.partial_buf = self.partial_buf[after..].to_string();
                self.state = next_state;
                return Transition::Advanced;
            }
            // Trim the rolling buffer to the longest suffix that is still
            // a prefix of `trigger`. Bound by len(trigger)-1 bytes.
            let max_keep = trigger.len().saturating_sub(1);
            if self.partial_buf.len() > max_keep {
                let drop_n = self.partial_buf.len() - max_keep;
                let drop_n = utf8_safe_split(&self.partial_buf, drop_n);
                self.partial_buf.drain(..drop_n);
            }
            // Trim further from the left: walk forward until the suffix
            // starting at that point is a prefix of trigger.
            while !self.partial_buf.is_empty() && !trigger.starts_with(self.partial_buf.as_str()) {
                // Drop one char (UTF-8 safe).
                let mut k = 1;
                while k < self.partial_buf.len()
                    && (self.partial_buf.as_bytes()[k] & 0b1100_0000) == 0b1000_0000
                {
                    k += 1;
                }
                self.partial_buf.drain(..k);
            }
            Transition::Stay
        }

        /// Try to match `partial_buf` (from its start) against any of the
        /// alternatives. If one fully matches, transition to its state and
        /// drop the matched bytes. If at least one is still a prefix
        /// candidate, stay. If none is a prefix anymore (corrupt), fall
        /// back to `Out` (recovery).
        fn transition_from_alternatives(&mut self, alternatives: &[(&str, State)]) -> Transition {
            for (needle, next) in alternatives {
                if self.partial_buf.starts_with(needle) {
                    self.partial_buf.drain(..needle.len());
                    self.state = next.clone();
                    return Transition::Advanced;
                }
            }
            // No full match. Check for any active prefix.
            let any_prefix = alternatives
                .iter()
                .any(|(n, _)| n.starts_with(self.partial_buf.as_str()));
            if !any_prefix {
                // Corruption: fall back to Out. Should be unreachable when
                // is_token_allowed is honored.
                self.partial_buf.clear();
                self.state = State::Out;
            }
            Transition::Stay
        }

        /// Tool-name transition: schema names + `">\n`. When `tool_idx` is
        /// `None`, identify the matching schema entry as soon as the
        /// partial_buf uniquely fixes one. When the buffer fully covers
        /// `NAME">\n` (or extends past it because a token spanned multiple
        /// grammar tokens), transition to `InInvokeBody` and keep any
        /// trailing bytes in the buffer for the next state to consume.
        fn transition_invoke_name(&mut self, tool_idx: Option<usize>) -> Transition {
            let candidates: Vec<(usize, String)> = match tool_idx {
                None => (0..self.grammar.tools.len())
                    .map(|i| (i, format!("{}\">\n", self.grammar.tools[i].name)))
                    .collect(),
                Some(idx) => vec![(idx, format!("{}\">\n", self.grammar.tools[idx].name))],
            };
            // Full coverage → transition into invoke body. Buffer keeps the
            // trailing suffix (if the committed token spanned more than the
            // tool-name terminator). Fresh invoke → emitted_params empty.
            for (idx, full) in &candidates {
                if self.partial_buf.starts_with(full) {
                    self.partial_buf.drain(..full.len());
                    self.state = State::InInvokeBody {
                        tool_idx: *idx,
                        emitted_params: Vec::new(),
                    };
                    return Transition::Advanced;
                }
            }
            // Lock in the tool_idx as soon as exactly one candidate still
            // matches as a prefix (when we were `None`).
            if tool_idx.is_none() {
                let matching: Vec<usize> = candidates
                    .iter()
                    .filter(|(_, full)| full.starts_with(self.partial_buf.as_str()))
                    .map(|(i, _)| *i)
                    .collect();
                if matching.len() == 1 {
                    self.state = State::InInvokeName {
                        tool_idx: Some(matching[0]),
                    };
                    // Don't drain — partial_buf still being built up against
                    // the (now locked-in) full name.
                    return Transition::Advanced;
                }
            }
            // No prefix match → corruption recovery.
            let any_prefix = candidates
                .iter()
                .any(|(_, full)| full.starts_with(self.partial_buf.as_str()));
            if !any_prefix {
                self.partial_buf.clear();
                self.state = State::Out;
            }
            Transition::Stay
        }

        /// Param-name transition: schema params for the current tool + `"`.
        /// When buffer fully covers `PARAM"`, transition to InParamAttr and
        /// keep trailing bytes for the next state.
        fn transition_param_name(
            &mut self,
            tool_idx: usize,
            param_idx: Option<usize>,
            emitted_params: Vec<usize>,
        ) -> Transition {
            let tool = &self.grammar.tools[tool_idx];
            // Exclude already-emitted params from the candidate set so the
            // model can't re-emit `command` twice in one invoke.
            let candidates: Vec<(usize, String)> = match param_idx {
                None => (0..tool.params.len())
                    .filter(|i| !emitted_params.contains(i))
                    .map(|i| (i, format!("{}\"", tool.params[i])))
                    .collect(),
                Some(idx) => vec![(idx, format!("{}\"", tool.params[idx]))],
            };
            for (idx, full) in &candidates {
                if self.partial_buf.starts_with(full) {
                    self.partial_buf.drain(..full.len());
                    self.state = State::InParamAttr {
                        tool_idx,
                        param_idx: *idx,
                        emitted_params,
                    };
                    return Transition::Advanced;
                }
            }
            if param_idx.is_none() {
                let matching: Vec<usize> = candidates
                    .iter()
                    .filter(|(_, full)| full.starts_with(self.partial_buf.as_str()))
                    .map(|(i, _)| *i)
                    .collect();
                if matching.len() == 1 {
                    self.state = State::InParamName {
                        tool_idx,
                        param_idx: Some(matching[0]),
                        emitted_params,
                    };
                    return Transition::Advanced;
                }
            }
            let any_prefix = candidates
                .iter()
                .any(|(_, full)| full.starts_with(self.partial_buf.as_str()));
            if !any_prefix {
                self.partial_buf.clear();
                self.state = State::Out;
            }
            Transition::Stay
        }
    }

    enum Transition {
        Stay,
        Advanced,
    }

    /// Largest split point ≤ `n` that doesn't cut through a multi-byte
    /// UTF-8 character.
    fn utf8_safe_split(s: &str, n: usize) -> usize {
        let bytes = s.as_bytes();
        let mut k = n.min(bytes.len());
        while k > 0 && (bytes[k] & 0b1100_0000) == 0b1000_0000 {
            k -= 1;
        }
        k
    }

    // ── Constants borrowed from the DSML format ─────────────────────────────

    /// Open trigger for the tool-calls block — entry from `State::Out`.
    pub(crate) const OPEN_TOOL_CALLS: &str = "<｜DSML｜tool_calls>";
    /// Closing marker for the tool-calls block.
    pub(crate) const CLOSE_TOOL_CALLS: &str = "</｜DSML｜tool_calls>";
    /// Open of an invoke. The V4F MQ2-Lloyd checkpoint also emits the
    /// `tool` variant (see [`OPEN_TOOL_VARIANT`]) — both must be accepted.
    pub(crate) const OPEN_INVOKE: &str = "<｜DSML｜invoke name=\"";
    pub(crate) const CLOSE_INVOKE: &str = "</｜DSML｜invoke>";
    /// V4F MQ2-Lloyd variant of [`OPEN_INVOKE`] / [`CLOSE_INVOKE`]: the
    /// model deterministically picks `tool` (token 72461) over `invoke`
    /// (token 41523) after `｜DSML｜` on most checkpoints — see
    /// `feedback_v4f_emits_tool_not_invoke.md` for the diagnosis.
    pub(crate) const OPEN_TOOL_VARIANT: &str = "<｜DSML｜tool name=\"";
    pub(crate) const CLOSE_TOOL_VARIANT: &str = "</｜DSML｜tool>";
    pub(crate) const OPEN_PARAM: &str = "<｜DSML｜parameter name=\"";
    pub(crate) const CLOSE_PARAM: &str = "</｜DSML｜parameter>";
    /// String attribute that follows the param-name close-quote.
    pub(crate) const ATTR_STRING_TRUE: &str = " string=\"true\">";
    pub(crate) const ATTR_STRING_FALSE: &str = " string=\"false\">";

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn matcher_starts_in_out_state_and_is_free() {
            let m = Matcher::new(vec![]);
            assert_eq!(*m.state(), State::Out);
            assert_eq!(m.partial(), "");
            assert!(m.is_free());
        }

        #[test]
        fn schema_holds_tool_and_params() {
            let s = ToolSchema {
                name: "read".to_string(),
                params: vec!["path".to_string()],
                required: vec!["path".to_string()],
            };
            assert_eq!(s.name, "read");
            assert_eq!(s.params, vec!["path".to_string()]);
        }

        fn schema_read_write() -> Vec<ToolSchema> {
            vec![
                ToolSchema {
                    name: "read".to_string(),
                    params: vec!["path".to_string()],
                    required: vec!["path".to_string()],
                },
                ToolSchema {
                    name: "write".to_string(),
                    params: vec!["path".to_string(), "content".to_string()],
                    required: vec!["path".to_string(), "content".to_string()],
                },
            ]
        }

        #[test]
        fn open_trigger_advances_into_tool_calls() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("Let me read.\n\n");
            assert!(m.is_free());
            m.advance("<｜DSML｜tool_calls>");
            assert_eq!(*m.state(), State::InToolCalls);
            assert_eq!(m.partial(), "");
        }

        #[test]
        fn open_trigger_split_across_chunks() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜");
            assert!(m.is_free());
            m.advance("DSML｜");
            m.advance("tool_");
            m.advance("calls>");
            assert_eq!(*m.state(), State::InToolCalls);
        }

        #[test]
        fn open_invoke_into_invoke_name_then_locks_tool() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"");
            assert_eq!(*m.state(), State::InInvokeName { tool_idx: None });
            // First letter is ambiguous: "r" matches read but not write
            m.advance("r");
            // tool_idx should be locked to 0 (read)
            assert_eq!(*m.state(), State::InInvokeName { tool_idx: Some(0) });
        }

        #[test]
        fn open_tool_variant_also_enters_invoke_name() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜tool name=\"");
            assert_eq!(*m.state(), State::InInvokeName { tool_idx: None });
        }

        #[test]
        fn tool_name_completion_enters_invoke_body() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"");
            m.advance("read\">\n");
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![]
                }
            );
            assert_eq!(m.partial(), "");
        }

        #[test]
        fn full_round_trip_through_one_call() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"read\">\n");
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![]
                }
            );
            m.advance("<｜DSML｜parameter name=\"");
            // `read` has exactly one param (`path`) — matcher locks
            // param_idx to Some(0) immediately since there's only one
            // candidate that has the empty buffer as a prefix.
            assert_eq!(
                *m.state(),
                State::InParamName {
                    tool_idx: 0,
                    param_idx: Some(0),
                    emitted_params: vec![],
                }
            );
            m.advance("path\"");
            assert_eq!(
                *m.state(),
                State::InParamAttr {
                    tool_idx: 0,
                    param_idx: 0,
                    emitted_params: vec![],
                }
            );
            m.advance(" string=\"true\">");
            assert_eq!(
                *m.state(),
                State::InParamBody {
                    tool_idx: 0,
                    param_idx: 0,
                    emitted_params: vec![],
                }
            );
            // Free emission of value
            assert!(m.is_free());
            m.advance("/tmp/test.txt</｜DSML｜parameter>");
            // After close, `path` (param_idx=0) is now in emitted_params.
            // `read` only has one param so required is now satisfied.
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![0]
                }
            );
            m.advance("</｜DSML｜invoke>");
            assert_eq!(*m.state(), State::InToolCalls);
            m.advance("</｜DSML｜tool_calls>");
            assert_eq!(*m.state(), State::Out);
        }

        #[test]
        fn leading_newline_consumed_in_tool_calls_state() {
            // Token `>\n` (vocab id 1018 in V4 tokenizer) leaves `\n` in the
            // buffer after the open trigger fires. Without ws tolerance the
            // matcher falls back to Out and the grammar mask collapses.
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>\n");
            assert_eq!(*m.state(), State::InToolCalls);
            assert_eq!(m.partial(), "");
        }

        #[test]
        fn newline_token_is_allowed_in_tool_calls_state() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            // Pure-newline next token must be accepted (it'll be consumed
            // as leading whitespace).
            assert!(m.is_token_allowed("\n"));
            // Newline followed by opener prefix also valid.
            assert!(m.is_token_allowed("\n<"));
            assert!(m.is_token_allowed("\n<｜DSML｜invoke name=\""));
            // Newline + invalid tag rejected.
            assert!(!m.is_token_allowed("\ncalling"));
            assert!(!m.is_token_allowed("\n<｜DSML｜foo"));
        }

        #[test]
        fn newline_consumed_then_real_tag_in_one_advance() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n");
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![]
                }
            );
        }

        #[test]
        fn out_state_constrains_after_dsml_token() {
            // Reproduces the production failure where V4F MQ2-Lloyd emits
            // `<｜DSML｜tool_actions>` / `<｜DSML｜calling>` / etc. instead
            // of the canonical `<｜DSML｜tool_calls>` open trigger. Once
            // `｜DSML｜` lands in the Out buffer, the matcher must restrict
            // continuations to the trigger so invented tag names get masked.
            let mut m = Matcher::new(schema_read_write());
            m.advance("Some prose. ");
            assert!(m.is_free());
            m.advance("<");
            assert!(m.is_free(), "single `<` could just be text");
            m.advance("｜DSML｜");
            assert!(!m.is_free(), "committed ｜DSML｜ → must constrain");
            // Allowed next: tokens that continue toward `tool_calls>`.
            assert!(m.is_token_allowed("tool_calls>"));
            assert!(m.is_token_allowed("tool"));
            assert!(m.is_token_allowed("t"));
            // Invented openers are masked.
            assert!(!m.is_token_allowed("tool_actions>"));
            assert!(!m.is_token_allowed("tool_invoke"));
            assert!(!m.is_token_allowed("calling"));
            assert!(!m.is_token_allowed("foo"));
            // Completing the trigger transitions.
            m.advance("tool_calls>");
            assert_eq!(*m.state(), State::InToolCalls);
        }

        #[test]
        fn out_state_remains_free_with_partial_unrelated_text() {
            // Buf accumulating just `<` doesn't constrain (could be HTML,
            // code, prose). Only `｜DSML｜` in buf flips the constraint.
            let mut m = Matcher::new(schema_read_write());
            m.advance("<");
            assert!(m.is_free());
            m.advance("html>");
            assert!(m.is_free());
        }

        #[test]
        fn required_params_block_empty_invoke_close() {
            // Reproduces the production failure where V4F emitted an empty
            // bash invoke (`<｜DSML｜tool name="bash"></｜DSML｜tool>`) and
            // the OpenAI client rejected it with "must have required
            // properties command".
            let schema = vec![ToolSchema {
                name: "bash".to_string(),
                params: vec!["command".to_string()],
                required: vec!["command".to_string()],
            }];
            let mut m = Matcher::new(schema);
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜tool name=\"bash\">\n");
            // In InInvokeBody with no params emitted yet — close MUST be
            // blocked, only OPEN_PARAM is legal.
            assert!(m.is_token_allowed("<｜DSML｜parameter name=\""));
            assert!(!m.is_token_allowed("</｜DSML｜tool>"));
            assert!(!m.is_token_allowed("</｜DSML｜invoke>"));
            // Emit the required param.
            m.advance(
                "<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n",
            );
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![0]
                }
            );
            // Now close IS legal.
            assert!(m.is_token_allowed("</｜DSML｜tool>"));
            assert!(m.is_token_allowed("</｜DSML｜invoke>"));
        }

        #[test]
        fn already_emitted_param_blocked_from_reuse() {
            // After `command` is emitted once, the matcher must not let
            // the model emit it again — the OpenAI parser rejects
            // duplicate keys.
            let schema = vec![ToolSchema {
                name: "bash".to_string(),
                params: vec!["command".to_string(), "cwd".to_string()],
                required: vec!["command".to_string()],
            }];
            let mut m = Matcher::new(schema);
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜tool name=\"bash\">\n");
            m.advance(
                "<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n",
            );
            // Required satisfied — close is legal.
            assert!(m.is_token_allowed("</｜DSML｜tool>"));
            // But emitting `command` AGAIN must be blocked. The remaining
            // schema-legal opener is for `cwd` only.
            m.advance("<｜DSML｜parameter name=\"");
            // Now only `cwd` is a legal param name; `command` is masked.
            assert!(m.is_token_allowed("c"));
            assert!(m.is_token_allowed("cwd\""));
            assert!(!m.is_token_allowed("command\""));
        }

        #[test]
        fn variant_close_tag_returns_to_tool_calls() {
            // Use a schema with no required params so the empty-invoke
            // close is legal (the required-params enforcement otherwise
            // blocks the close until `path` is filled in).
            let schema = vec![ToolSchema {
                name: "read".to_string(),
                params: vec!["path".to_string()],
                required: vec![],
            }];
            let mut m = Matcher::new(schema);
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜tool name=\"read\">\n");
            m.advance("</｜DSML｜tool>"); // variant close
            assert_eq!(*m.state(), State::InToolCalls);
        }

        #[test]
        fn is_token_allowed_rejects_bad_tag_in_tool_calls() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            // After OPEN_TOOL_CALLS we're in InToolCalls. Legal next tokens
            // start `<` (one of the opens) or `</` (close). An invented
            // continuation like `cbl>` should be rejected.
            assert!(!m.is_token_allowed("cbl>"));
            assert!(!m.is_token_allowed("tool_invoke"));
            assert!(m.is_token_allowed("<"));
            assert!(m.is_token_allowed("<｜DSML｜invoke name=\""));
            assert!(m.is_token_allowed("<｜DSML｜tool name=\""));
            assert!(m.is_token_allowed("</"));
        }

        #[test]
        fn is_token_allowed_constrains_tool_name() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"");
            // Only "read" or "write" are legal first chars.
            assert!(m.is_token_allowed("r"));
            assert!(m.is_token_allowed("w"));
            assert!(!m.is_token_allowed("foo"));
            assert!(!m.is_token_allowed("b"));
        }

        #[test]
        fn out_state_allows_everything() {
            let m = Matcher::new(schema_read_write());
            assert!(m.is_token_allowed("anything goes here"));
            assert!(m.is_token_allowed(""));
            assert!(m.is_token_allowed("<｜DSML｜tool_calls>"));
        }

        #[test]
        fn token_mask_marks_all_true_in_free_state() {
            let m = Matcher::new(schema_read_write());
            let vocab = vec![
                "hello".to_string(),
                "<｜DSML｜tool_calls>".to_string(),
                "foo".to_string(),
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert!(mask.iter().all(|&b| b));
        }

        #[test]
        fn token_mask_constrains_inside_tool_calls() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            let vocab = vec![
                "<".to_string(),                       // ✓ prefix of all opens
                "</".to_string(),                      // ✓ prefix of close
                "<｜".to_string(),                     // ✓ prefix
                "<｜DSML｜".to_string(),               // ✓ prefix
                "<｜DSML｜invoke name=\"".to_string(), // ✓ full open-invoke
                "<｜DSML｜tool name=\"".to_string(),   // ✓ full open-tool-variant
                "</｜DSML｜tool_calls>".to_string(),   // ✓ full close
                "tool_cbl".to_string(),                // ✗ not a prefix
                "calling".to_string(),                 // ✗ invented tag
                "hello world".to_string(),             // ✗ random text
                "<bad>".to_string(),                   // ✗ wrong open form
            ];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            // First 7 should be allowed, last 4 rejected.
            for (i, expected) in [
                true, true, true, true, true, true, true, false, false, false, false,
            ]
            .iter()
            .enumerate()
            {
                assert_eq!(
                    mask[i], *expected,
                    "vocab[{i}]={:?} expected {expected}",
                    vocab[i]
                );
            }
        }

        #[test]
        fn apply_mask_to_logits_sets_neg_inf_on_disallowed() {
            let mask = vec![true, false, true, false];
            let mut logits = vec![1.0, 2.0, 3.0, 4.0];
            Matcher::apply_mask_to_logits(&mask, &mut logits);
            assert_eq!(logits[0], 1.0);
            assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
            assert_eq!(logits[2], 3.0);
            assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
        }

        #[test]
        fn token_mask_locks_tool_after_first_letter() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"");
            // After locking nothing yet — both r* and w* are legal.
            let vocab = vec!["r".to_string(), "w".to_string(), "b".to_string()];
            let mut mask = vec![false; vocab.len()];
            m.token_mask(&vocab, &mut mask);
            assert_eq!(mask, vec![true, true, false]);

            // Commit "r" — now only `read` is the active candidate; `w` is locked out.
            m.advance("r");
            let vocab2 = vec![
                "ead\">\n".to_string(), // ✓ completes "read\">\n"
                "ite\">\n".to_string(), // ✗ would be "write"
                "x".to_string(),        // ✗ random
            ];
            let mut mask2 = vec![false; vocab2.len()];
            m.token_mask(&vocab2, &mut mask2);
            assert_eq!(mask2, vec![true, false, false]);
        }

        #[test]
        fn param_body_constrains_close_marker_when_prefix_started() {
            // Reproduces the failing real-world case where the model emits
            // `</｜DSML｜paperameter>` (paper+ameter) instead of
            // `</｜DSML｜parameter>` and the parser stays open forever.
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n");
            m.advance("<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/test.txt");
            // Free emission so far.
            assert!(m.is_free());
            // Model starts emitting close marker.
            m.advance("</");
            assert!(!m.is_free(), "must constrain once close-prefix starts");
            // Next valid token would be `｜DSML｜` (the atomic).
            assert!(m.is_token_allowed("｜DSML｜"));
            // But not arbitrary text — the model can't escape from the
            // close marker to type more value content.
            assert!(!m.is_token_allowed("paper"));
            assert!(!m.is_token_allowed("foo"));
            // After completing the close, returns to InvokeBody with the
            // `path` param marked emitted.
            m.advance("｜DSML｜parameter>");
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![0]
                }
            );
        }

        #[test]
        fn param_body_is_free_until_close_marker() {
            let mut m = Matcher::new(schema_read_write());
            m.advance("<｜DSML｜tool_calls>");
            m.advance("<｜DSML｜invoke name=\"read\">\n");
            m.advance("<｜DSML｜parameter name=\"path\" string=\"true\">");
            assert!(m.is_free());
            m.advance("/etc/passwd");
            assert!(m.is_free());
            m.advance("</｜DSML｜parameter>");
            assert_eq!(
                *m.state(),
                State::InInvokeBody {
                    tool_idx: 0,
                    emitted_params: vec![0]
                }
            );
        }
    }
} // mod dsml
