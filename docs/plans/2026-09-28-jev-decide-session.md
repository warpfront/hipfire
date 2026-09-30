# Decide Session Mode (v1.1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `POST /v1/systemone` take the agent's chat `messages` (plus
optional `tools`) in place of `state`. The decide then answers its questions
at the end of the conversation the daemon has already cached, and leaves
that cache in place so the next chat turn reuses it. No model reset.

**Architecture:**

- `hipfire-engine::decide` (pure) gains the session pieces:
  - request-mode selection and `messages` parsing, normalised like
    `generate`;
  - the question-only text;
  - the conversation-end rule;
  - the start planner (extend / resume / cold);
  - the session token limits.
- `hipfire-generate::decide::session` renders every question with the chat
  path's cached Jinja render, then runs the lifecycle over the existing
  `Carrier` decide hooks:
  1. bring the model to the conversation end;
  2. snapshot;
  3. per question: restore, prefill the suffix, read the labels;
  4. restore;
  5. commit the conversation.
- `handle_decide_message` dispatches on the mode, so the daemon's `decide`
  arm is untouched.
- Serve projects `messages`, `tools` and `tool_choice` through the chat
  request projection and forwards them.

**Tech Stack:** Rust (the crates above), serde_json, hyper (serve),
Python 3 stdlib (fake daemon, gates).

**Spec:** `docs/specs/2026-09-28-jev-decide-design.md` §12. Read it first;
§12.2 lists the chat-cache facts this plan relies on, with `file:line`
references.

## Global Constraints

- Every v1 constraint in `docs/plans/2026-09-28-jev-decide.md` ("Global
  Constraints") still holds:
  - Jev wire format;
  - limits and confidence formulas;
  - probabilities never rounded;
  - status codes;
  - layering;
  - branch `feat/jev-decide` in `~/repos/hipfire-jev`;
  - never push to `fork`;
  - do not touch `~/repos/hipfire`.

  Plain (`state`) requests keep v1 behaviour exactly, including the reset.
- **Session mode never resets the model** except in the three cases spec
  §12.4 names: a cold start when the cached conversation doesn't match, a
  loaded speculator with a non-extend start, and fail-closed after an
  error. It never calls `production_fail_closed_rollback` on the success
  path of an `extend` or `resume`. A cold start keeps `asst_turn_cache`.
- **Every `DecideSnapshot` is freed on every path**, success and error.
  `DeltaNetSnapshot` has no `Drop`; `Snaps::free` is the only release
  path. A checkpoint borrowed for a resume goes back into
  `m.prefill_checkpoints`.
- **The daemon only dispatches.** Make no edits to
  `crates/hipfire-daemon/src/main.rs`. The `daemon_lines` ratchet ceiling is
  4882 and the file is at 4882. Session dispatch lives in
  `hipfire_generate::decide::handle_decide_message`.
- **CI ratchets pass:**
  - `bash scripts/leanup-ratchets.sh`;
  - `python3 scripts/check-crate-maps.py --check`, after regenerating the
    maps of the crates you touch;
  - `./scripts/no-gpu-ci.sh`.

  Add no `arch_key() == "…"` string dispatch: `arch_key_dispatch` is
  capped at 14. Add no new env vars: `check-env-docs.py`.
- **Commits** end with the trailer
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`,
  passed as a second `-m`.
- **GPU etiquette on this workstation (starling):**
  - one daemon at a time. Before any daemon or gate run, `pgrep -a -x
    daemon` must print nothing;
  - never kill a daemon you did not start. If one is running, stop and ask
    the user;
  - check `grep MemAvailable /proc/meminfo` before loading: Qwen3.5-4B
    needs ≥ 8 GiB free, 27B ≥ 30 GiB;
  - put hipcc on PATH: `export PATH=/opt/rocm-7.2.2/bin:$PATH`;
  - Bash calls are capped at 10 minutes. Run anything longer (gates,
    release builds from cold) in the background with `nohup … &` and poll
    the log with short commands. Never block in a foreground sleep.
- Run `rustfmt --edition 2021` only on the files a task touches. Never run
  `cargo fmt` workspace-wide, and never rustfmt
  `crates/hipfire-cli/src/main.rs`: it carries an unrelated format backlog.
  Write the tests added there already formatted.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `crates/hipfire-engine/src/decide.rs` | modify | `DecideMode` / `request_mode`, `parse_questions` (split out of `parse_request`), `SessionRequest` / `parse_session_request`, `build_question_text`, `conversation_end`, `SessionStart` / `plan_session_start`, `check_session_tokens`; tests. |
| `crates/hipfire-generate/src/decide.rs` | modify | `mod session;`, shared helpers `question_labels`, `gather_labels`, `decided_reply`; mode dispatch in `handle_decide_message`. |
| `crates/hipfire-generate/src/decide/session.rs` | create | `render_session_prompt`, `plan_session`, `execute_session`, `handle_session`; GPU-free render tests. |
| `crates/hipfire-cli/src/serve/decide.rs` | modify | Project and forward `messages` / `tools` / `tool_choice`. |
| `crates/hipfire-cli/src/serve/fake_daemon.py` | modify | `decide` branch accepts `messages`, 422 on both/neither, echoes the session shape in `timing`. |
| `crates/hipfire-cli/src/main.rs` | modify | 3 route tests next to the existing `systemone_*` tests. |
| `scripts/jev_eval/daemon_client.py` | modify | `chat`, `session_decide`. |
| `scripts/jev_eval/gates.py` | modify | Gates S1–S5 and a shared `measure_leak`. |
| `scripts/jev_eval/README.md` | modify | Session gates note. |
| `crates/{hipfire-engine,hipfire-generate,hipfire-loader,hipfire-cli}/map.md` | regenerate | `scripts/check-crate-maps.py`. |
| `crates/hipfire-daemon/src/main.rs` | **unchanged** | The `decide` arm already calls `handle_decide_message`. |

---

### Task 1: Session request parsing and pure planning (`hipfire-engine::decide`)

**Files:**
- Modify: `crates/hipfire-engine/src/decide.rs`

**Interfaces:**
- Produces:
  - `pub enum DecideMode { Plain, Session }` and
    `pub fn request_mode(v: &Value) -> Result<DecideMode, String>`;
  - `pub fn parse_questions(v: &Value) -> Result<Vec<Question>, String>`.
    `parse_request` now calls it, and its behaviour is unchanged;
  - `pub struct SessionRequest { pub messages: Vec<Message>, pub tools: Option<Vec<Value>>, pub questions: Vec<Question> }`
    and `pub fn parse_session_request(v: &Value) -> Result<SessionRequest, String>`;
  - `pub fn build_question_text(q: &Question, labels: &[String]) -> String`.
    `build_user_text` becomes `"State:\n{state}\n\n" + build_question_text`,
    byte-identical to today;
  - `pub fn conversation_end(seqs: &[Vec<u32>], probe: &[u32], turn_open: Option<u32>) -> usize`;
  - `pub enum SessionStart { Extend { from: usize }, Resume { ckpt_idx: usize, pos: usize }, Cold }`
    and
    `pub fn plan_session_start(prior: &[u32], conv: &[u32], seq_pos: usize, reuse_eligible: bool, ckpt_positions: &[usize], resume_enabled: bool) -> SessionStart`;
  - `pub fn check_session_tokens(questions: &[Question], conv_len: usize, seq_lens: &[usize]) -> Result<(), String>`.
- Every `Err(String)` above is a 422 message.

- [ ] **Step 1: Write the failing tests.** Append inside the existing
  `#[cfg(test)] mod tests` of `crates/hipfire-engine/src/decide.rs`,
  before its closing `}`.

```rust
    #[test]
    fn request_mode_requires_exactly_one_of_state_or_messages() {
        assert_eq!(request_mode(&json!({"state": "s"})), Ok(DecideMode::Plain));
        assert_eq!(request_mode(&json!({"messages": []})), Ok(DecideMode::Session));
        assert!(request_mode(&json!({"state": "s", "messages": []}))
            .unwrap_err()
            .contains("both"));
        assert!(request_mode(&json!({"questions": {}}))
            .unwrap_err()
            .contains("state"));
    }

    fn session_err(v: Value) -> String {
        match parse_session_request(&v) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_session_request_and_normalises_like_generate() {
        let raw = "a\n\n\n\nb";
        let r = parse_session_request(&json!({
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": raw},
                         {"role": "assistant", "content": "ok"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        }))
        .unwrap();
        assert_eq!(r.messages.len(), 3);
        // Same normalisation the daemon's generate arm applies to messages.
        assert_eq!(
            r.messages[1].content,
            hipfire_runtime::tokenizer::maybe_normalize_prompt(raw)
        );
        assert_eq!(r.tools.as_ref().map(Vec::len), Some(1));
        assert_eq!(r.questions.len(), 1);
        let no_tools = parse_session_request(&json!({
            "messages": [{"role": "user", "content": "u"}], "tools": [],
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        }))
        .unwrap();
        assert!(no_tools.tools.is_none(), "empty tools == no tools, as generate");
    }

    #[test]
    fn rejects_bad_session_shapes() {
        let q = json!({"q": {"type": "noul", "instructions": "i"}});
        assert!(session_err(json!({"messages": [], "questions": q})).contains("messages"));
        assert!(session_err(json!({"messages": "hi", "questions": q})).contains("messages"));
        assert!(session_err(json!({
            "messages": [{"role": "robot", "content": "x"}], "questions": q
        }))
        .contains("messages"));
        assert!(session_err(json!({
            "messages": [{"role": "user", "content": "x"}], "tools": {}, "questions": q
        }))
        .contains("tools"));
        assert!(session_err(json!({
            "messages": [{"role": "user", "content": "x"}],
            "questions": {"q": {"type": "noul"}}
        }))
        .contains("q.instructions"));
    }

    #[test]
    fn user_text_is_state_block_plus_question_text() {
        let q = choice_q(&["a", "b"]);
        let labels = vec!["A".to_string(), "B".to_string()];
        let qt = build_question_text(&q, &labels);
        assert_eq!(build_user_text("S", &q, &labels), format!("State:\nS\n\n{qt}"));
        assert!(qt.starts_with("Question: i\nOptions:\nA. a\nB. b\n"), "{qt}");
        assert!(qt.ends_with("\nAnswer with the option code only."), "{qt}");
    }

    #[test]
    fn conversation_end_is_the_question_turn_opener() {
        const OPEN: u32 = 0;
        // [conversation: 1 2][question turn: OPEN 5 7 …]
        let seqs = vec![vec![1, 2, OPEN, 5, 7, 9, 9], vec![1, 2, OPEN, 5, 7, 8, 8]];
        let probe = vec![1, 2, OPEN, 5, 7, 3];
        assert_eq!(conversation_end(&seqs, &probe, Some(OPEN)), 2);
        // A single question: the probe alone bounds the shared prefix.
        assert_eq!(conversation_end(&seqs[..1], &probe, Some(OPEN)), 2);
        // Openers inside the conversation are not the question turn's.
        let s2 = vec![vec![OPEN, 4, 1, OPEN, 5, 9]];
        assert_eq!(conversation_end(&s2, &[OPEN, 4, 1, OPEN, 5, 3], Some(OPEN)), 3);
        // No ChatML opener in the tokenizer: the shared prefix itself.
        assert_eq!(conversation_end(&seqs, &probe, None), 5);
        // Capped so every question keeps a non-empty suffix.
        assert_eq!(conversation_end(&[vec![1, 2, 3]], &[1, 2, 3, 4], None), 2);
    }

    #[test]
    fn session_start_extends_a_cached_prefix() {
        use SessionStart::*;
        let conv = [1, 2, 3, 4];
        assert_eq!(
            plan_session_start(&[1, 2, 3], &conv, 3, true, &[], true),
            Extend { from: 3 }
        );
        // Exact match (the usual agent case): nothing to prefill.
        assert_eq!(
            plan_session_start(&conv, &conv, 4, true, &[], true),
            Extend { from: 4 }
        );
    }

    #[test]
    fn session_start_resumes_or_goes_cold() {
        use SessionStart::*;
        let conv = [1, 2, 9, 9];
        // Divergence at 2: the latest checkpoint at or before the LCP.
        assert_eq!(
            plan_session_start(&[1, 2, 3, 4], &conv, 4, true, &[1, 2, 3], true),
            Resume { ckpt_idx: 1, pos: 2 }
        );
        assert_eq!(plan_session_start(&[1, 2, 3, 4], &conv, 4, true, &[3], true), Cold);
        assert_eq!(plan_session_start(&[1, 2, 3, 4], &conv, 4, true, &[1], false), Cold);
        // Cached conversation runs past the request's: rewind by checkpoint.
        assert_eq!(
            plan_session_start(&[1, 2, 9, 9, 5], &conv, 5, true, &[2, 4], true),
            Resume { ckpt_idx: 1, pos: 4 }
        );
        // Not reusable: ineligible, empty, or cursor out of step with tokens.
        assert_eq!(plan_session_start(&[1, 2], &conv, 2, false, &[], true), Cold);
        assert_eq!(plan_session_start(&[], &conv, 0, true, &[], true), Cold);
        assert_eq!(plan_session_start(&[1, 2], &conv, 7, true, &[], true), Cold);
    }

    #[test]
    fn session_limits_count_tokens_beyond_the_conversation() {
        let qs = vec![choice_q(&["a", "b"]), choice_q(&["c", "d"])];
        // A long conversation is bounded by max_seq, not Jev's limits.
        assert!(check_session_tokens(&qs, 100_000, &[100_010, 100_020]).is_ok());
        let e = check_session_tokens(&qs, 10, &[10 + MAX_QUESTION_TOKENS + 1, 20]).unwrap_err();
        assert!(e.contains("q:") && e.contains("per-question limit of 32000"), "{e}");
        let qs3 = vec![choice_q(&["a", "b"]); 3];
        let e = check_session_tokens(&qs3, 5, &[5 + 30_000, 5 + 30_000, 5 + 4_001]).unwrap_err();
        assert!(e.contains("request limit of 64000"), "{e}");
        assert!(check_session_tokens(&qs3, 5, &[5 + 30_000, 5 + 30_000, 5 + 4_000]).is_ok());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: compile errors, starting with `cannot find function request_mode`.

- [ ] **Step 3: Implement.** Five edits to
  `crates/hipfire-engine/src/decide.rs`.

(a) Imports. Directly below `use serde_json::{json, Value};`, add:

```rust
use hipfire_runtime::prompt_frame::Message;
```

(b) Split `parse_questions` out of `parse_request`. Cut the code from the
line `let qs = v` through the closing `}` of the `for (name, q) in qs {`
loop, and move it unchanged into a new function placed directly above
`parse_request`:

```rust
/// Validate the `questions` map (shared by plain and session requests).
/// Every `Err` is a 422 message naming the question and field.
pub fn parse_questions(v: &Value) -> Result<Vec<Question>, String> {
    // ← the moved `let qs = v … ;` through the end of the `for` loop
    Ok(questions)
}
```

`parse_request` then reads, in full:

```rust
/// Validate a Jev request. Every `Err` is a 422 message naming the field.
pub fn parse_request(v: &Value) -> Result<DecideRequest, String> {
    let state = v.get("state").ok_or("state is required")?;
    if !(state.is_string() || state.is_object() || state.is_array()) {
        return Err("state must be a string, JSON object or JSON array".into());
    }
    let questions = parse_questions(v)?;
    Ok(DecideRequest {
        state_text: render_state(state),
        questions,
    })
}
```

(c) Directly below `parse_request`, add:

```rust
/// Which decide mode a request selects (spec §12.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecideMode {
    /// Jev wire: `state` + `questions`; the model is reset around the decide.
    Plain,
    /// hipfire extension: `messages` (+ `tools`) + `questions`; the
    /// conversation is the state and the chat prompt cache is kept.
    Session,
}

/// Exactly one of `state` / `messages`. `Err` is a 422 message.
pub fn request_mode(v: &Value) -> Result<DecideMode, String> {
    match (v.get("state").is_some(), v.get("messages").is_some()) {
        (true, false) => Ok(DecideMode::Plain),
        (false, true) => Ok(DecideMode::Session),
        (true, true) => Err("exactly one of state or messages is allowed, got both".into()),
        (false, false) => Err("state is required (or messages, for session mode)".into()),
    }
}

/// A validated session-mode request (spec §12.1).
#[derive(Debug, Clone)]
pub struct SessionRequest {
    /// Chat messages, content normalised exactly as the daemon's `generate`
    /// arm normalises `messages`, so the render matches what chat cached.
    pub messages: Vec<Message>,
    /// OpenAI tool definitions; `None` when absent or empty (as `generate`).
    pub tools: Option<Vec<Value>>,
    pub questions: Vec<Question>,
}

/// Validate a session request. Every `Err` is a 422 message.
pub fn parse_session_request(v: &Value) -> Result<SessionRequest, String> {
    let raw = v
        .get("messages")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or("messages must be a non-empty array of chat messages")?;
    let mut messages: Vec<Message> = serde_json::from_value(Value::Array(raw.clone()))
        .map_err(|e| format!("messages: {e}"))?;
    for m in &mut messages {
        let normalized = hipfire_runtime::tokenizer::maybe_normalize_prompt(&m.content);
        if matches!(normalized, std::borrow::Cow::Owned(_)) {
            m.content = normalized.into_owned();
        }
    }
    let tools = match v.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) if a.is_empty() => None,
        Some(Value::Array(a)) => Some(a.clone()),
        Some(_) => return Err("tools must be an array of tool definitions".into()),
    };
    Ok(SessionRequest {
        messages,
        tools,
        questions: parse_questions(v)?,
    })
}
```

(d) Replace the whole `build_user_text` function with:

```rust
/// The question block (instructions, labelled options, answer line). Plain
/// mode prefixes the state block; session mode sends it as its own turn.
pub fn build_question_text(q: &Question, labels: &[String]) -> String {
    let descs: Vec<String> = match &q.kind {
        QuestionKind::Choice { descriptions, .. } => descriptions.clone(),
        QuestionKind::Score { levels } => levels.clone(),
        QuestionKind::Noul {
            true_desc,
            false_desc,
        } => vec![
            true_desc.clone().unwrap_or_else(|| "yes".into()),
            false_desc.clone().unwrap_or_else(|| "no".into()),
        ],
    };
    let mut s = format!("Question: {}\nOptions:\n", q.instructions);
    for (label, desc) in labels.iter().zip(descs.iter()) {
        s.push_str(&format!("{label}. {desc}\n"));
    }
    s.push_str("\nAnswer with the option code only.");
    s
}

/// The user-turn text for one plain-mode question. The state block comes
/// first so every question's rendered prompt shares the longest prefix.
pub fn build_user_text(state_text: &str, q: &Question, labels: &[String]) -> String {
    format!("State:\n{state_text}\n\n{}", build_question_text(q, labels))
}
```

(e) Directly below `check_total_tokens`, add:

```rust
/// End of the conversation inside the session renders (spec §12.3): the
/// position where the appended question turn starts. `seqs` are the
/// question renders, `probe` the same conversation with a one-character
/// user turn, `turn_open` the tokenizer's `<|im_start|>` id if it has one.
/// The shared prefix of all of them runs through the question turn's header;
/// the last opener inside it is that turn's. Without an opener token, the
/// shared prefix itself. Capped so every question suffix is non-empty.
pub fn conversation_end(seqs: &[Vec<u32>], probe: &[u32], turn_open: Option<u32>) -> usize {
    let Some(first) = seqs.first() else { return 0 };
    let min_q = seqs.iter().map(Vec::len).min().unwrap_or(0);
    let min_all = min_q.min(probe.len());
    let mut n = 0;
    while n < min_all && seqs.iter().all(|s| s[n] == first[n]) && probe[n] == first[n] {
        n += 1;
    }
    let n = n.min(min_q.saturating_sub(1));
    turn_open
        .and_then(|id| first[..n].iter().rposition(|&t| t == id))
        .unwrap_or(n)
}

/// Where a session decide starts (spec §12.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStart {
    /// The cached conversation is a prefix of the request's: prefill only
    /// `conv[from..]` (nothing on an exact match).
    Extend { from: usize },
    /// Restore prefill checkpoint `ckpt_idx` (at `pos` <= LCP) and prefill
    /// `conv[pos..]`.
    Resume { ckpt_idx: usize, pos: usize },
    /// Reset (keeping the assistant-turn cache) and prefill `conv` whole.
    Cold,
}

/// Pick the start from the cached `prior` tokens and the request's
/// conversation `conv`. `ckpt_positions` is ascending (as
/// `take_dn_checkpoint` appends them).
pub fn plan_session_start(
    prior: &[u32],
    conv: &[u32],
    seq_pos: usize,
    reuse_eligible: bool,
    ckpt_positions: &[usize],
    resume_enabled: bool,
) -> SessionStart {
    if !reuse_eligible || prior.is_empty() || seq_pos != prior.len() {
        return SessionStart::Cold;
    }
    let lcp = prior.iter().zip(conv).take_while(|(a, b)| a == b).count();
    if lcp == prior.len() {
        return SessionStart::Extend { from: lcp };
    }
    if resume_enabled {
        if let Some(idx) = ckpt_positions.iter().rposition(|&p| p > 0 && p <= lcp) {
            return SessionStart::Resume {
                ckpt_idx: idx,
                pos: ckpt_positions[idx],
            };
        }
    }
    SessionStart::Cold
}

/// Jev's limits applied to the question part of a session request: each
/// question's tokens past the conversation <= `MAX_QUESTION_TOKENS`, all of
/// them together <= `MAX_TOTAL_TOKENS`. The conversation itself is bounded
/// by `max_seq` (checked by the runner).
pub fn check_session_tokens(
    questions: &[Question],
    conv_len: usize,
    seq_lens: &[usize],
) -> Result<(), String> {
    let mut total = 0usize;
    for (q, &len) in questions.iter().zip(seq_lens) {
        let suffix = len.saturating_sub(conv_len);
        if suffix > MAX_QUESTION_TOKENS {
            return Err(format!(
                "{}: question is {suffix} tokens beyond the conversation, over the \
                 per-question limit of {MAX_QUESTION_TOKENS} tokens",
                q.name
            ));
        }
        total += suffix;
    }
    if total > MAX_TOTAL_TOKENS {
        return Err(format!(
            "questions are {total} tokens beyond the conversation, over the request \
             limit of {MAX_TOTAL_TOKENS} tokens"
        ));
    }
    Ok(())
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: all decide tests pass. That covers the new 8 and every v1 test
(`user_text_layout` proves `build_user_text` is byte-identical).

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2021 crates/hipfire-engine/src/decide.rs
cargo test -p hipfire-engine --lib decide
git add crates/hipfire-engine/src/decide.rs
git commit -m "feat(decide): session request parsing and pure planning" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Session runner (`hipfire-generate::decide::session`) and dispatch

**Files:**
- Modify: `crates/hipfire-generate/src/decide.rs`
- Create: `crates/hipfire-generate/src/decide/session.rs`
- Regenerate: `crates/hipfire-engine/map.md`,
  `crates/hipfire-generate/map.md`, `crates/hipfire-loader/map.md`. The
  loader and generate maps are already stale on this branch.

**Interfaces:**
- Consumes (Task 1): `d::{request_mode, DecideMode, parse_session_request, build_question_text, conversation_end, SessionStart, plan_session_start, check_session_tokens, usage_json, assemble_answer, DecideRequest, Question}`.
- Consumes (existing):
  - the `Carrier` hooks `decide_prefill_logits`, `decide_save`,
    `decide_restore`, `decide_kv_compact_offset` (`hipfire-loader`
    `lib.rs:189-231`, implemented for Qwen35 and Llama at
    `carriers.rs:749-833` and `1145-1197`);
  - `hipfire_loader::decide::DecideSnapshot { seq_pos, recurrent }`;
  - `hipfire_loader::AsstTurnCache`;
  - `crate::qwen::qwen_jinja_lookup_turn`;
  - `crate::ar::{ckpt_resume_enabled, truncate_checkpoints}`;
  - `hipfire_runtime::prompt_frame::{JinjaChatFrame, build_cached_history_jinja, template_emits_history_primer, Message}`;
  - `hipfire_engine::emit::render_tail_opens_think`.
- Produces:
  - `pub(crate) fn render_session_prompt(tok: &Tokenizer, template: &str, asst_turn_cache: &mut AsstTurnCache, messages: &[Message], tools: Option<&[Value]>, user_text: &str) -> Result<Vec<u32>, DecideError>`;
  - `pub(crate) fn handle_session(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, msg: &Value, id: &str) -> (Value, bool)`,
    where `bool` = the model was rolled back;
  - `handle_decide_message` dispatches `messages` requests to
    `handle_session`, and answers both/neither with 422. Its signature is
    unchanged.

- [ ] **Step 1: Write the failing tests.**

In `crates/hipfire-generate/src/decide.rs`, append inside the existing
`mod tests` before its closing `}`:

```rust
    #[test]
    fn gather_labels_picks_ids_and_rejects_out_of_range() {
        assert_eq!(
            gather_labels(&[2, 0], &[0.5, 1.5, 2.5]).unwrap(),
            vec![2.5, 0.5]
        );
        let e = gather_labels(&[3], &[0.0; 3]).unwrap_err();
        assert_eq!(e.status, 500);
        assert!(e.message.contains("label token id 3"), "{}", e.message);
    }
```

Create `crates/hipfire-generate/src/decide/session.rs` containing only the
test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_loader::AsstTurnCache;
    use hipfire_runtime::prompt_frame::{CachedAssistantBody, CachedAssistantTurn};

    /// Qwen3.5-style: history assistant turns render bare; thinking off
    /// primes `<think>\n\n</think>\n\n` (as prompt_frame.rs's history-primer
    /// test template).
    const BARE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}";
    /// Opens a think block whatever `enable_thinking` says.
    const ALWAYS_THINK: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}";
    const CONV: &str = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n";
    const TAIL: &str =
        "<|im_start|>user\nQuestion: Q?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    /// Byte-level ChatML tokenizer: every byte is one token (100 + byte), no
    /// merges; ChatML/think markers and five reserved splice sentinels are
    /// atomic specials. Same shape as prompt_frame.rs's `make_tokenizer`.
    fn tok() -> Tokenizer {
        // GPT-2 bytes_to_unicode.
        let mut bs: Vec<u32> = (u32::from(b'!')..=u32::from(b'~'))
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0u32..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut vocab = serde_json::Map::new();
        for (b, c) in bs.iter().zip(&cs) {
            let ch = char::from_u32(*c).expect("gpt2 byte char");
            vocab.insert(ch.to_string(), json!(100 + b));
        }
        let specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<|reserved_0|>",
            "<|reserved_1|>",
            "<|reserved_2|>",
            "<|reserved_3|>",
            "<|reserved_4|>",
        ];
        let mut added = Vec::new();
        for (id, s) in specials.iter().enumerate() {
            vocab.insert((*s).to_string(), json!(id));
            added.push(json!({"id": id, "content": s, "special": true}));
        }
        let hf = json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": added
        });
        Tokenizer::from_hf_json(&hf.to_string()).expect("test tokenizer")
    }

    fn msg(role: &str, content: &str) -> Message {
        serde_json::from_value(json!({"role": role, "content": content})).unwrap()
    }

    fn history() -> Vec<Message> {
        vec![msg("user", "hi"), msg("assistant", "yo")]
    }

    #[test]
    fn renders_conversation_plus_question_turn_with_closed_think() {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let r = render_session_prompt(&t, BARE, &mut cache, &history(), None, "Question: Q?")
            .unwrap();
        assert_eq!(r, t.encode(&format!("{CONV}{TAIL}")));
    }

    #[test]
    fn splices_cached_assistant_turn_like_the_chat_path() {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let fp = crate::common::asst_turn_fingerprint(
            &crate::common::normalize_asst_turn_for_fingerprint("yo"),
            &[],
        );
        let body = vec![200, 201, 202];
        cache.insert(
            fp,
            CachedAssistantTurn {
                reasoning: None,
                tools: Vec::new(),
                content: Some(CachedAssistantBody {
                    token_ids: body.clone(),
                    text: String::new(),
                }),
            },
        );
        let r = render_session_prompt(&t, BARE, &mut cache, &history(), None, "Question: Q?")
            .unwrap();
        let mut want = t.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
        // Bare history turns: the generation primer is re-supplied before
        // the verbatim body, exactly as the chat path splices it.
        want.extend(t.encode("<think>\n\n</think>\n\n"));
        want.extend(&body);
        want.extend(t.encode(&format!("<|im_end|>\n{TAIL}")));
        assert_eq!(r, want);
    }

    #[test]
    fn conversation_end_is_a_prefix_of_the_next_chat_turn() {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let msgs = history();
        let q = render_session_prompt(&t, BARE, &mut cache, &msgs, None, "Question: Q?").unwrap();
        let probe =
            render_session_prompt(&t, BARE, &mut cache, &msgs, None, PROBE_USER_TEXT).unwrap();
        let e = d::conversation_end(&[q.clone()], &probe, t.special_token_id(TURN_OPEN));
        assert_eq!(&q[..e], t.encode(CONV).as_slice());
        // The next chat turn (thinking on, as an agent may run it) renders
        // the same conversation first: committing q[..e] is a forward
        // extension for the chat prompt cache.
        let chat = JinjaChatFrame {
            tokenizer: &t,
            template: BARE,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        };
        let mut next = msgs.clone();
        next.push(msg("user", "next"));
        let chat_tokens = t.encode(&chat.render_messages(&next, None, None).unwrap());
        assert!(chat_tokens.starts_with(&q[..e]));
        assert!(chat_tokens.len() > e);
    }

    #[test]
    fn open_think_template_is_400_and_render_error_is_422() {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let e = render_session_prompt(&t, ALWAYS_THINK, &mut cache, &history(), None, "Q")
            .unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("thinking"), "{}", e.message);
        let e = render_session_prompt(
            &t,
            "{{ raise_exception('bad order') }}",
            &mut cache,
            &history(),
            None,
            "Q",
        )
        .unwrap_err();
        assert_eq!(e.status, 422);
        assert!(e.message.contains("messages"), "{}", e.message);
    }
}
```

In `crates/hipfire-generate/src/decide.rs`, directly below the `use`
lines, add:

```rust
mod session;
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-generate --lib decide`
Expected: compile errors, among them `cannot find function gather_labels`
and `cannot find function render_session_prompt`.

- [ ] **Step 3: Refactor `decide.rs` into shared helpers and dispatch.**
  Five edits to `crates/hipfire-generate/src/decide.rs`.

(a) Module doc. Replace the `//!` block (lines 4-7) with:

```rust
//! Decide runner. Plain mode (`state`): render every question's full prompt,
//! prefill the longest common token prefix once, snapshot, then per question
//! restore → prefill suffix → gather label logits; leaves the model reset
//! (spec §4.1). Session mode (`messages`, [`session`]): the cached chat
//! conversation is the state and is kept (spec §12).
```

(b) Add these helpers directly above `pub struct RenderedQuestions`:

```rust
/// Option labels for one question and their single-token ids (leading BOS
/// stripped), shared by plain and session mode.
fn question_labels(
    tok: &hipfire_runtime::tokenizer::Tokenizer,
    q: &d::Question,
) -> Result<(Vec<String>, Vec<u32>), DecideError> {
    let bos = tok.add_bos.then_some(tok.bos_id);
    let encode = |s: &str| tok.encode(s);
    let single = |s: &str| single_token_id(encode, bos, s).is_some();
    let labels = d::labels_for(q, single).map_err(|e| DecideError::new(422, e))?;
    let ids = labels
        .iter()
        .map(|l| {
            single_token_id(encode, bos, l).ok_or_else(|| {
                DecideError::new(500, format!("label {l:?} lost its single-token id"))
            })
        })
        .collect::<Result<Vec<u32>, _>>()?;
    Ok((labels, ids))
}

/// The logits of `ids`, in order.
fn gather_labels(ids: &[u32], logits: &[f32]) -> Result<Vec<f32>, DecideError> {
    ids.iter()
        .map(|&id| {
            logits.get(id as usize).copied().ok_or_else(|| {
                DecideError::new(
                    500,
                    format!(
                        "decide: label token id {id} outside logits (len {})",
                        logits.len()
                    ),
                )
            })
        })
        .collect()
}

/// The success reply for either mode.
fn decided_reply(id: &str, out: DecideOutcome) -> Value {
    json!({
        "type": "decided",
        "id": id,
        "answers": out.answers,
        "usage": out.usage,
        "timing": out.timing,
    })
}
```

(c) In `render_question_prompts`, delete the three lines `let bos = …`,
`let encode = …` and `let single = …`. Then replace the loop's first two
statements (`let labels = d::labels_for(…)?;` and the `let ids = labels…?;`
block) with:

```rust
        let (labels, ids) = question_labels(tok, q)?;
```

(d) In `run_questions`, replace the `let label_logits = label_ids[i] …
.collect::<Result<Vec<f32>, DecideError>>()?;` block and the
`per_q_logits.push(label_logits);` line after it with:

```rust
        per_q_logits.push(gather_labels(&label_ids[i], &logits)?);
```

(e) Update `handle_decide_message`. First, replace everything in its body
from `let Some(m) = model else {` to the function's closing `}` with:

```rust
    let Some(m) = model else {
        unreachable!("precheck refuses when no model is present");
    };
    match d::request_mode(msg) {
        Err(e) => return (DecideError::new(422, e).to_reply(id), false),
        Ok(d::DecideMode::Session) => return session::handle_session(m, gpu, msg, id),
        Ok(d::DecideMode::Plain) => {}
    }
    let plan = match plan_decide(m, msg) {
        Ok(p) => p,
        Err(e) => return (e.to_reply(id), false),
    };
    let reply = match execute_decide(m, gpu, plan) {
        Ok(out) => decided_reply(id, out),
        Err(e) => e.to_reply(id),
    };
    (reply, true)
}
```

Then replace its whole `///` doc comment with:

```rust
/// Full dispatch for a `decide` JSONL message: [`precheck`] refusals, the
/// mode check (both or neither of `state` / `messages`: 422), then plain
/// mode ([`plan_decide`] + [`execute_decide`]) or session mode
/// ([`session::handle_session`]). Returns `(reply, reset)`: `reset` is true
/// exactly when the model was rolled back -- every plain decide that ran, and
/// a session decide's cold start, speculator-mode end or fail-closed error --
/// so the caller advances `state_epoch` by the rule the daemon applies to
/// `reset`. Refusals and validation errors (422 etc.) return `false`.
```

- [ ] **Step 4: Implement `session.rs`.** Put this above the test module
  in `crates/hipfire-generate/src/decide/session.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Session-mode decide: the chat conversation is the state. Every question
//! is rendered as `messages` + one user turn through the chat path's cached
//! Jinja render. The conversation cache is reused (extend / checkpoint
//! resume / cold), the model is snapshotted at the conversation end, and each
//! question is answered past it. The conversation cache is then left in place
//! for the next chat turn. Spec: `docs/specs/2026-09-28-jev-decide-design.md` §12.

use super::{
    check_eviction_limit, decided_reply, execute_decide, gather_labels, hook, ms,
    question_labels, rollback, DecideError, DecideOutcome, DecidePlan, RenderedQuestions,
};
use hipfire_engine::decide as d;
use hipfire_loader::decide::DecideSnapshot;
use hipfire_loader::LoadedModel;
use hipfire_runtime::prompt_frame::{
    build_cached_history_jinja, template_emits_history_primer, JinjaChatFrame, Message,
};
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::time::Instant;

/// User content of the probe render that bounds the conversation (§12.3).
const PROBE_USER_TEXT: &str = ".";
/// ChatML turn opener: the conversation ends where the question turn opens.
const TURN_OPEN: &str = "<|im_start|>";

fn dev_flag_off(name: &str) -> bool {
    hipfire_config::developer_var(name).ok().as_deref() == Some("0")
}

/// Tokens of `messages` + one user turn `user_text` + the closed-think
/// assistant opener. The render matches the chat path's Jinja prompt-cache
/// render (ar.rs:3055-3135): the model's template, each cached assistant turn
/// spliced verbatim from `asst_turn_cache`, and the generation primer
/// re-supplied when the template renders history turns bare.
pub(crate) fn render_session_prompt(
    tok: &Tokenizer,
    template: &str,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    messages: &[Message],
    tools: Option<&[Value]>,
    user_text: &str,
) -> Result<Vec<u32>, DecideError> {
    let turn: Message = serde_json::from_value(json!({"role": "user", "content": user_text}))
        .map_err(|e| DecideError::new(500, format!("question turn: {e}")))?;
    let mut msgs = messages.to_vec();
    msgs.push(turn);
    let frame = JinjaChatFrame {
        tokenizer: tok,
        template,
        system: None,
        user: user_text,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let render_err =
        |e: String| DecideError::new(422, format!("messages: chat template render failed: {e}"));
    let cold_text = frame.render_messages(&msgs, tools, None).map_err(render_err)?;
    if hipfire_engine::emit::render_tail_opens_think(&cold_text) {
        return Err(DecideError::new(
            400,
            "model chat template cannot disable thinking; decide needs a closed-think assistant turn",
        ));
    }
    let cold = tok.encode(&cold_text);
    // The generation primer: what the cold render leaves after the final
    // assistant opener (the chat path's derivation, ar.rs:3062-3071).
    let opener_len = tok.encode("<|im_start|>assistant\n").len();
    let primer = match tok
        .special_token_id(TURN_OPEN)
        .and_then(|id| cold.iter().rposition(|&t| t == id))
    {
        Some(q) if q + opener_len <= cold.len() => cold[q + opener_len..].to_vec(),
        _ => Vec::new(),
    };
    let primer = if template_emits_history_primer(&frame, &primer) {
        Vec::new()
    } else {
        primer
    };
    build_cached_history_jinja(&frame, &msgs, tools, |msg| {
        crate::qwen::qwen_jinja_lookup_turn(&mut *asst_turn_cache, msg, &primer)
    })
    .map_err(render_err)
}

/// Everything [`execute_session`] needs. Computing it leaves KV, recurrent
/// state, `seq_pos` and `conversation_tokens` untouched; the render may
/// refresh `asst_turn_cache`'s LRU order, as a chat render does.
pub(crate) struct SessionPlan {
    questions: Vec<d::Question>,
    carrier: &'static dyn hipfire_loader::Carrier,
    label_ids: Vec<Vec<u32>>,
    seqs: Vec<Vec<u32>>,
    /// End of the rendered conversation (start of the question turn), E.
    conv_end: usize,
    start: d::SessionStart,
    no_snapshot: bool,
}

pub(crate) fn plan_session(m: &mut LoadedModel, req: &Value) -> Result<SessionPlan, DecideError> {
    let parsed = d::parse_session_request(req).map_err(|e| DecideError::new(422, e))?;
    let no_snapshot = req.get("_debug_no_snapshot").and_then(Value::as_bool) == Some(true);
    if m.pp > 1 || m.ep.is_some() {
        return Err(DecideError::new(
            409,
            "decide requires a single-GPU model (pp=1, no EP)",
        ));
    }
    if m.kv_adaptive.is_some() {
        return Err(DecideError::new(409, "decide requires kv_adaptive off"));
    }
    let carrier = hipfire_loader::carrier_for(m.arch_id)
        .filter(|c| c.decide_supported())
        .ok_or_else(|| {
            DecideError::new(
                400,
                format!("decide not supported for arch_id {}", m.arch_id),
            )
        })?;
    let template = m
        .chat_template
        .as_deref()
        .filter(|_| !dev_flag_off("HIPFIRE_JINJA_CHAT"))
        .ok_or_else(|| {
            DecideError::new(
                400,
                "decide session mode needs the model's Jinja chat template \
                 (none loaded, or HIPFIRE_JINJA_CHAT=0)",
            )
        })?;
    let tok = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| DecideError::new(400, "model has no tokenizer"))?;
    let cache = &mut m.asst_turn_cache;
    let tools = parsed.tools.as_deref();
    let mut label_ids = Vec::with_capacity(parsed.questions.len());
    let mut seqs = Vec::with_capacity(parsed.questions.len());
    for q in &parsed.questions {
        let (labels, ids) = question_labels(tok, q)?;
        let text = d::build_question_text(q, &labels);
        seqs.push(render_session_prompt(
            tok,
            template,
            cache,
            &parsed.messages,
            tools,
            &text,
        )?);
        label_ids.push(ids);
    }
    let probe = render_session_prompt(
        tok,
        template,
        cache,
        &parsed.messages,
        tools,
        PROBE_USER_TEXT,
    )?;
    let conv_end = d::conversation_end(&seqs, &probe, tok.special_token_id(TURN_OPEN));

    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    d::check_session_tokens(&parsed.questions, conv_end, &lens)
        .map_err(|e| DecideError::new(422, e))?;
    let longest = lens.iter().copied().max().unwrap_or(0);
    check_eviction_limit(
        longest,
        m.eviction.as_ref().map(|e| e.budget()),
        m.physical_cap,
    )?;
    if longest + 1 > m.max_seq {
        return Err(DecideError {
            status: 422,
            message: format!(
                "prompt needs {} tokens, model max_seq is {}",
                longest + 1,
                m.max_seq
            ),
            required_max_seq: Some(longest + 1),
        });
    }
    // The chat path's prompt-cache eligibility (ar.rs:3025-3029): never under
    // eviction or with the cache kill switch. A positional rewind also needs
    // an uncompacted KV cache.
    let reuse = m.eviction.is_none()
        && !dev_flag_off("HIPFIRE_QWEN_PROMPT_CACHE")
        && carrier.decide_kv_compact_offset(m) == Some(0);
    // With a speculator loaded the AR checkpoint ring is not the live one.
    let resume = crate::ar::ckpt_resume_enabled() && m.speculator.is_none();
    let ckpts: Vec<usize> = m.prefill_checkpoints.iter().map(|(p, _)| *p).collect();
    let start = if no_snapshot {
        d::SessionStart::Cold
    } else {
        d::plan_session_start(
            &m.conversation_tokens,
            &seqs[0][..conv_end],
            m.seq_pos,
            reuse,
            &ckpts,
            resume,
        )
    };
    Ok(SessionPlan {
        questions: parsed.questions,
        carrier,
        label_ids,
        seqs,
        conv_end,
        start,
        no_snapshot,
    })
}

/// The snapshots a session decide may hold. Both are freed on every path.
#[derive(Default)]
struct Snaps {
    /// At the conversation end: restored before each later question and at the end.
    conv: Option<DecideSnapshot>,
    /// At the cached conversation's end; speculator (read-only) extend only.
    entry: Option<DecideSnapshot>,
}

impl Snaps {
    fn free(&mut self, gpu: &mut rdna_compute::Gpu) {
        if let Some(s) = self.conv.take() {
            s.free(gpu);
        }
        if let Some(s) = self.entry.take() {
            s.free(gpu);
        }
    }
}

fn start_name(s: d::SessionStart) -> &'static str {
    match s {
        d::SessionStart::Extend { .. } => "extend",
        d::SessionStart::Resume { .. } => "resume",
        d::SessionStart::Cold => "cold",
    }
}

/// Restore prefill checkpoint `idx` through the carrier's decide hook, then
/// drop the checkpoints and conversation tokens past it, as the chat path's
/// resume does (ar.rs:3346-3390). The checkpoint goes back into the ring.
fn resume_from_checkpoint(
    carrier: &dyn hipfire_loader::Carrier,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    idx: usize,
    pos: usize,
) -> Result<(), DecideError> {
    let (p, snap) = m.prefill_checkpoints.remove(idx);
    debug_assert_eq!(p, pos, "planned resume position");
    let wrapped = DecideSnapshot {
        seq_pos: p,
        recurrent: Some(snap),
    };
    let restored = hook(carrier.decide_restore(m, gpu, &wrapped));
    let snap = wrapped.recurrent.expect("wrapped with Some");
    m.prefill_checkpoints.insert(idx, (p, snap));
    restored?;
    crate::ar::truncate_checkpoints(&mut m.prefill_checkpoints, idx + 1, gpu);
    m.conversation_tokens.truncate(pos);
    Ok(())
}

/// Cold start that keeps the assistant-turn cache: the attested rollback
/// (which clears it) with the cache moved out and back. The chat path's own
/// cold start keeps it too (ar.rs:3391-3434); entries are keyed by turn
/// content, not position, so they stay valid.
fn cold_reset_keep_turn_cache(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), DecideError> {
    let turns = std::mem::replace(
        &mut m.asst_turn_cache,
        hipfire_loader::AsstTurnCache::new_from_env(),
    );
    let r = rollback(m, gpu);
    m.asst_turn_cache = turns;
    r
}

/// Bring the model to the conversation end E, snapshot there, answer every
/// question past it, and restore to E. Returns where the conversation
/// prefill started (`from`).
#[allow(clippy::too_many_arguments)]
fn run_session(
    plan: &SessionPlan,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    read_only: bool,
    snaps: &mut Snaps,
    timing: &mut Value,
    per_q: &mut Vec<Vec<f32>>,
    reset: &mut bool,
) -> Result<usize, DecideError> {
    let carrier = plan.carrier;
    let e = plan.conv_end;
    let conv = &plan.seqs[0][..e];
    let t = Instant::now();
    let from = match plan.start {
        d::SessionStart::Extend { from } => from,
        d::SessionStart::Resume { ckpt_idx, pos } => {
            resume_from_checkpoint(carrier, m, gpu, ckpt_idx, pos)?;
            pos
        }
        d::SessionStart::Cold => {
            *reset = true;
            cold_reset_keep_turn_cache(m, gpu)?;
            0
        }
    };
    if read_only && matches!(plan.start, d::SessionStart::Extend { .. }) {
        snaps.entry = Some(hook(carrier.decide_save(m, gpu))?);
    }
    if from < e {
        // The conversation's own logits are never read.
        hook(carrier.decide_prefill_logits(m, gpu, &conv[from..], from, false))?;
    }
    snaps.conv = Some(hook(carrier.decide_save(m, gpu))?);
    timing["cached_tokens"] = json!(from);
    timing["conversation_prefill_tokens"] = json!(e - from);
    timing["state_prefill_ms"] = json!(ms(t));
    let conv_snap = snaps.conv.as_ref().expect("saved above");
    for (i, seq) in plan.seqs.iter().enumerate() {
        let t = Instant::now();
        if i > 0 {
            hook(carrier.decide_restore(m, gpu, conv_snap))?;
        }
        let logits = hook(carrier.decide_prefill_logits(m, gpu, &seq[e..], e, true))?;
        per_q.push(gather_labels(&plan.label_ids[i], &logits)?);
        timing["question_ms"]
            .as_array_mut()
            .expect("question_ms is an array")
            .push(json!(ms(t)));
    }
    // Back to the conversation end: seq_pos = E, recurrent state at E.
    hook(carrier.decide_restore(m, gpu, conv_snap))?;
    Ok(from)
}

/// Leave the conversation cache. Without a speculator: commit
/// `conversation_tokens = conv[..E]` (seq_pos is already E). With one
/// (read-only): an extend restores the cached end exactly; any other start
/// ends with the attested rollback.
fn finish_session(
    plan: &SessionPlan,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    read_only: bool,
    from: usize,
    snaps: &Snaps,
    reset: &mut bool,
) -> Result<(), DecideError> {
    let e = plan.conv_end;
    if !read_only {
        m.conversation_tokens.truncate(from);
        m.conversation_tokens
            .extend_from_slice(&plan.seqs[0][from..e]);
        debug_assert_eq!(m.seq_pos, m.conversation_tokens.len());
        return Ok(());
    }
    match (plan.start, snaps.entry.as_ref()) {
        (d::SessionStart::Extend { .. }, Some(entry)) => {
            hook(plan.carrier.decide_restore(m, gpu, entry))
        }
        _ => {
            *reset = true;
            rollback(m, gpu)
        }
    }
}

/// Run a planned session decide. Returns the outcome and whether the model
/// was rolled back (cold start, speculator-mode end, or fail-closed error).
pub(crate) fn execute_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    plan: SessionPlan,
) -> (Result<DecideOutcome, DecideError>, bool) {
    if plan.no_snapshot {
        // Gate S2's reference: v1's `_debug_no_snapshot` over the session
        // tokens, i.e. each question one full prefill from a reset model.
        let SessionPlan {
            questions,
            carrier,
            label_ids,
            seqs,
            ..
        } = plan;
        let v1 = DecidePlan {
            parsed: d::DecideRequest {
                state_text: String::new(),
                questions,
            },
            carrier,
            rendered: RenderedQuestions { label_ids, seqs },
            prefix_len: 0,
        };
        return (execute_decide(m, gpu, v1), true);
    }
    let t_total = Instant::now();
    let read_only = m.speculator.is_some();
    let e = plan.conv_end;
    let mut reset = false;
    let mut snaps = Snaps::default();
    let mut per_q: Vec<Vec<f32>> = Vec::with_capacity(plan.seqs.len());
    let mut timing = json!({
        "mode": "session",
        "start": start_name(plan.start),
        "prefix_tokens": e,
        "state_prefill_ms": 0.0,
        "question_ms": [],
        "committed": !read_only,
    });
    let run = run_session(
        &plan,
        m,
        gpu,
        read_only,
        &mut snaps,
        &mut timing,
        &mut per_q,
        &mut reset,
    );
    let fin = run.and_then(|from| finish_session(&plan, m, gpu, read_only, from, &snaps, &mut reset));
    snaps.free(gpu);
    if let Err(err) = fin {
        // Fail closed like every chat abort path (ar.rs:3317-3324): a
        // retained cache must never carry uncommitted state.
        let _ = rollback(m, gpu);
        return (Err(err), true);
    }
    let lens: Vec<usize> = plan.seqs.iter().map(Vec::len).collect();
    let mut answers = serde_json::Map::new();
    for (q, ll) in plan.questions.iter().zip(&per_q) {
        answers.insert(q.name.clone(), d::assemble_answer(q, ll));
    }
    timing["total_ms"] = json!(ms(t_total));
    (
        Ok(DecideOutcome {
            answers: Value::Object(answers),
            usage: d::usage_json(e, &lens),
            timing,
        }),
        reset,
    )
}

/// `decide` with `messages`: plan (validation, rendering, admission;
/// leaves model state untouched on `Err`), then execute.
pub(crate) fn handle_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    msg: &Value,
    id: &str,
) -> (Value, bool) {
    let plan = match plan_session(m, msg) {
        Ok(p) => p,
        Err(e) => return (e.to_reply(id), false),
    };
    let (res, reset) = execute_session(m, gpu, plan);
    let reply = match res {
        Ok(out) => decided_reply(id, out),
        Err(e) => e.to_reply(id),
    };
    (reply, reset)
}
```

- [ ] **Step 5: Run to verify it passes**

```bash
cargo test -p hipfire-generate --lib decide
cargo check --workspace --examples
```

Expected:

- all decide tests pass: the 5 new ones (1 in `decide.rs`, 4 in
  `session.rs`) and every v1 test;
- the workspace checks, including the `split_prefill_probe` example, which
  still calls `render_question_prompts` with an unchanged signature.

If `Tokenizer::from_hf_json` rejects the fixture, build the tokenizer
exactly as `make_tokenizer` in
`crates/hipfire-runtime/src/prompt_frame.rs:1939-1994` does (same
`added_tokens` list, string-built JSON). Don't loosen the assertions.

- [ ] **Step 6: Build the daemon and smoke the dispatch** (no model is
  loaded, so no GPU work runs)

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
cargo build --release -p hipfire-daemon
git diff --exit-code HEAD -- crates/hipfire-daemon/src/main.rs && wc -l < crates/hipfire-daemon/src/main.rs
pgrep -a -x daemon || echo "no daemon running"
```

The `git diff` must be empty and the line count must be 4882: this task
leaves the daemon untouched. Run the smoke below only if `pgrep` printed
`no daemon running`. If a daemon is running, skip the smoke, note it, and
never kill that daemon.

```bash
printf '%s\n' '{"type":"decide","id":"s1","messages":[{"role":"user","content":"x"}],"questions":{"q":{"type":"noul","instructions":"y?"}}}' \
  | timeout 60 target/release/daemon 2>/dev/null | grep '"decided"'
```

Expected output:
`{"error":{"message":"no model loaded","status":503},"id":"s1","type":"decided"}`.
Key order may differ. The precheck answers before the mode dispatch, as in
v1.

- [ ] **Step 7: Maps, ratchets, format, commit**

```bash
rustfmt --edition 2021 crates/hipfire-generate/src/decide.rs
python3 scripts/check-crate-maps.py hipfire-engine hipfire-generate hipfire-loader
python3 scripts/check-crate-maps.py --check
bash scripts/leanup-ratchets.sh
cargo test -p hipfire-generate --lib decide
git add crates/hipfire-generate/src/decide.rs crates/hipfire-generate/src/decide/session.rs \
  crates/hipfire-engine/map.md crates/hipfire-generate/map.md crates/hipfire-loader/map.md
git commit -m "feat(decide): session-mode runner over the chat prompt cache" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`rustfmt` on `decide.rs` also formats its child module `decide/session.rs`.
Expected: `--check` exits 0, and the ratchets report no violation, with
`daemon_lines` unchanged at ≤ 4882. If `--check` still names another
crate, it was stale before this task; regenerate only the crates this plan
touches and mention the rest in the hand-back.

---

### Task 3: Serve forwards session requests

**Files:**
- Modify: `crates/hipfire-cli/src/serve/decide.rs`
- Modify: `crates/hipfire-cli/src/serve/fake_daemon.py`
- Modify: `crates/hipfire-cli/src/main.rs` (tests, beside
  `systemone_daemon_validation_error_maps_to_422`)
- Regenerate: `crates/hipfire-cli/map.md`

**Interfaces:**
- Consumes:
  - `super::complete::{project_request_contract, include_reasoning_content}`
    (`complete.rs:1558`, `1585`). The contract's `messages` and
    `forwarded_tools` are the exact projection the chat path forwards;
  - `ServeRuntime::ensure_model` → `Result<hipfire_config::ResolvedConfig>`.
- Produces: the daemon request gains `messages` and `tools` (projected)
  when the body carries `messages`. Everything else (`state`, `questions`,
  retry, keep-warm, response) is unchanged.

- [ ] **Step 1: Update the fake daemon.** In
  `crates/hipfire-cli/src/serve/fake_daemon.py`, replace the whole
  `elif ty == "decide":` branch, from that line through its final `out(…)`
  call, with:

```python
    elif ty == "decide":
        qs = req.get("questions") or {}
        state = req.get("state")
        messages = req.get("messages")
        # Spec §12.1: exactly one of state / messages.
        if (state is None) == (messages is None):
            out({"type": "decided", "id": req.get("id"),
                 "error": {"status": 422,
                           "message": "exactly one of state or messages is required"}})
            continue
        if messages is not None:
            valid = isinstance(messages, list) and bool(messages) and bool(qs)
        else:
            valid = isinstance(state, (str, dict, list)) and bool(qs)
        if not valid:
            out({"type": "decided", "id": req.get("id"),
                 "error": {"status": 422, "message": "state/messages/questions invalid"}})
            continue
        # Task 8 review round 1: exercise the serve-side required_max_seq
        # reload-and-retry. Refuses until a `load` has raised this session's
        # max_seq to (or past) the threshold, which the default config never
        # reaches on its own (memory.max_seq defaults to 32768).
        if state == "t-needs-ctx" and LOADED_MAX_SEQ < 40000:
            out({"type": "decided", "id": req.get("id"),
                 "error": {"status": 422, "message": "prompt needs 40000 tokens",
                           "required_max_seq": 40000}})
            continue
        # A required_max_seq past serve's reload ceiling (MAX_SEQ_CEILING):
        # serve must return this 422 as-is without reloading.
        if state == "t-needs-huge-ctx":
            out({"type": "decided", "id": req.get("id"),
                 "error": {"status": 422, "message": "prompt needs 10000000 tokens",
                           "required_max_seq": 10000000}})
            continue
        # Task 8 review round 1: delays the reply so a test can prove the
        # HTTP-side admission guard is held for the full daemon round trip,
        # not released the instant the client disconnects.
        if state == "t-slow":
            time.sleep(1)
        answers = {}
        for name, q in qs.items():
            t = q.get("type")
            if t == "choice":
                keys = list((q.get("criteria") or {}).keys())
                p = 1.0 / len(keys)
                answers[name] = {"type": "choice", "choice": keys[0],
                                 "probabilities": {k: p for k in keys}, "confidence": 0.0}
            elif t == "score":
                n = len(q.get("criteria") or [])
                answers[name] = {"type": "score", "score": (n - 1) / 2,
                                 "probabilities": {str(i): 1.0 / n for i in range(n)},
                                 "confidence": 1.0 / n}
            else:
                answers[name] = {"type": "noul", "noul": 0.5}
        timing = {"prefix_tokens": 8, "total_ms": 1.0}
        if messages is not None:
            # Echo what serve forwarded so route tests can see the projection.
            timing.update({"mode": "session",
                           "session_roles": [m.get("role") for m in messages],
                           "session_tools": len(req.get("tools") or [])})
        out({"type": "decided", "id": req.get("id"), "answers": answers,
             "usage": {"input_tokens": 10 * len(qs), "output_tokens": 0},
             "timing": timing})
```

- [ ] **Step 2: Write the failing route tests.** In
  `crates/hipfire-cli/src/main.rs`, directly after the test
  `systemone_daemon_validation_error_maps_to_422`, add:

```rust
    #[cfg(unix)]
    fn timing_header(headers: &str) -> serde_json::Value {
        let raw = headers
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.trim()
                    .eq_ignore_ascii_case("x-hipfire-timing")
                    .then(|| v.trim().to_string())
            })
            .expect("x-hipfire-timing header");
        serde_json::from_str(&raw).unwrap()
    }

    #[cfg(unix)]
    fn session_body(model: &str, tool_choice: Option<&str>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "yo"}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "description": "d",
                "parameters": {"type": "object", "properties": {}}}}],
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        });
        if let Some(tc) = tool_choice {
            body["tool_choice"] = serde_json::json!(tc);
        }
        body
    }

    /// Session mode (spec §12.6): serve forwards the chat projection of
    /// `messages` and `tools`, and the Jev body shape is unchanged.
    #[cfg(unix)]
    #[test]
    fn systemone_session_forwards_projected_messages_and_tools() {
        let harness = Task11HttpHarness::spawn("systemone-session");
        let (status, headers, body) =
            post_systemone(harness.port(), &session_body(harness.model(), None));
        assert_eq!(status, 200, "{body}");
        let t = timing_header(&headers);
        assert_eq!(t["mode"], "session", "{t}");
        let roles: Vec<&str> = t["session_roles"]
            .as_array()
            .expect("session_roles")
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        assert!(roles.ends_with(&["user", "assistant"]), "{roles:?}");
        assert_eq!(t["session_tools"], 1, "{t}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["answers"]["q"]["type"], "noul");
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["answers", "model", "usage"], "{body}");
    }

    /// `tool_choice: "none"` drops tools exactly as the chat projection
    /// does, proving serve projects rather than passes `tools` through.
    #[cfg(unix)]
    #[test]
    fn systemone_session_applies_tool_choice_projection() {
        let harness = Task11HttpHarness::spawn("systemone-session-tc");
        let (status, headers, body) =
            post_systemone(harness.port(), &session_body(harness.model(), Some("none")));
        assert_eq!(status, 200, "{body}");
        assert_eq!(timing_header(&headers)["session_tools"], 0);
    }

    #[cfg(unix)]
    #[test]
    fn systemone_state_and_messages_together_is_422() {
        let harness = Task11HttpHarness::spawn("systemone-both");
        let mut body = session_body(harness.model(), None);
        body["state"] = serde_json::json!("s");
        let (status, _, body) = post_systemone(harness.port(), &body);
        assert_eq!(status, 422, "{body}");
    }
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test -p hipfire-cli systemone_s`. The filter matches exactly
the 3 new tests.
Expected: 3 failures, because serve does not forward `messages` yet.

- The two `systemone_session_*` tests get 422 instead of 200: the fake
  daemon sees neither `state` nor `messages`.
- `systemone_state_and_messages_together_is_422` gets 200 instead of 422:
  only `state` arrives.

- [ ] **Step 4: Implement** in `crates/hipfire-cli/src/serve/decide.rs`,
  in `run_decide`.

(a) Replace

```rust
            if let Err(e) = runtime.ensure_model(&target, &shared.meta, None) {
                return DecideOutcome::Err {
                    status: 500,
                    message: format!("{e:#}"),
                    required_max_seq: None,
                };
            }
```

with

```rust
            let resolved = match runtime.ensure_model(&target, &shared.meta, None) {
                Ok(r) => r,
                Err(e) => {
                    return DecideOutcome::Err {
                        status: 500,
                        message: format!("{e:#}"),
                        required_max_seq: None,
                    }
                }
            };
            // Session mode (spec §12.6): project `messages` / `tools` /
            // `tool_choice` exactly as a chat request is projected, so the
            // daemon renders the conversation the chat path cached.
            let session = match body.get("messages") {
                None => None,
                Some(_) => match super::complete::project_request_contract(
                    &body,
                    &resolved,
                    super::complete::include_reasoning_content(runtime.current_arch.as_deref()),
                ) {
                    Ok(c) => Some((c.messages, c.forwarded_tools)),
                    Err(e) => {
                        return DecideOutcome::Err {
                            status: 422,
                            message: format!("{e:#}"),
                            required_max_seq: None,
                        }
                    }
                },
            };
```

(b) The block's final expression `(runtime.engine.clone(), echo)` becomes
`(runtime.engine.clone(), echo, session)`, and its binding
`let (engine, model_echo) = {` becomes
`let (engine, model_echo, session) = {`.

(c) Directly after the `for key in ["state", "questions"] { … }` loop, add:

```rust
        if let Some((messages, tools)) = &session {
            msg["messages"] = messages.clone();
            if let Some(tools) = tools {
                msg["tools"] = tools.clone();
            }
        }
```

Leave the existing `_debug_no_snapshot` comment above that loop as it is.
Add one line to it:
`// messages / tools are forwarded projected (below), never raw.`

- [ ] **Step 5: Run the tests**

```bash
cargo test -p hipfire-cli systemone
cargo test -p hipfire-cli
```

Expected:

- `systemone`: every test passes, the 3 new ones plus the v1 ones;
- full suite: all pass, and the count grows by 3.

- [ ] **Step 6: Maps, format, commit**

```bash
rustfmt --edition 2021 crates/hipfire-cli/src/serve/decide.rs
python3 scripts/check-crate-maps.py hipfire-cli
python3 scripts/check-crate-maps.py --check
git add crates/hipfire-cli/src/serve/decide.rs crates/hipfire-cli/src/serve/fake_daemon.py \
  crates/hipfire-cli/src/main.rs crates/hipfire-cli/map.md
git commit -m "feat(serve): forward session-mode decide with the chat projection" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: GPU gates S1–S5

**Files:**
- Modify: `scripts/jev_eval/daemon_client.py`
- Modify: `scripts/jev_eval/gates.py`
- Modify: `scripts/jev_eval/README.md`

**Interfaces:**
- Consumes the daemon wire: `generate` with `messages`; its `done` event
  carries `cached_tokens` and `finish_reason` (`ar.rs:380-404`). Also
  `decide` with `messages` (Task 2), and `timing.{start, prefix_tokens,
  cached_tokens, conversation_prefill_tokens}`.
- Produces: `gates.py` also runs S1–S5 (spec §12.8). The exit code rule is
  unchanged: 0 iff no gate FAILs.

- [ ] **Step 1: Client helpers.** In `scripts/jev_eval/daemon_client.py`,
  add to `class Daemon` after `generate_greedy`:

```python
    def chat(self, messages, max_tokens=64):
        """One greedy chat turn over OpenAI-shaped `messages`, the way serve
        forwards /v1/chat/completions (messages + last user text as prompt).
        Returns (visible text, done event)."""
        self.attempt += 1
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        self.send({"type": "generate", "id": f"chat{self.attempt}", "attempt_id": self.attempt,
                   "prompt": prompt, "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens, "thinking_enabled": False})
        text = []
        while True:
            v = self.recv_until({"token", "done", "error"})
            if v["type"] == "token":
                text.append(v.get("text", ""))
            elif v["type"] == "done":
                return "".join(text), v
            else:
                raise RuntimeError(f"chat failed: {v}")

    def session_decide(self, messages, questions, **extra):
        """Session-mode decide (spec §12): the conversation is the state."""
        self.send({"type": "decide", "id": "s", "messages": messages,
                   "questions": questions, **extra})
        return self.recv_until({"decided", "error"})
```

- [ ] **Step 2: The gates.** In `scripts/jev_eval/gates.py`:

(a) In the module docstring, after the `0   (--cask only, per daemon) …`
entry, add:

```
  S1  session cache reuse: a session decide between two chat turns leaves
      the chat prompt cache usable. (a) no delta, default env: turn 2 text
      bit-identical to a run without the decide, decide start=extend with 0
      conversation prefill, turn 2 cached_tokens == decide prefix_tokens.
      (b) delta (decide includes the next user turn), exact env: same text,
      conversation prefill > 0, turn 2 cached_tokens == E. INCONCLUSIVE when
      the no-decide baseline gets no cache hit (no chat cache: Llama
      carrier, --cask).
  S2  session exactness (exact env): per question Δ == 0 vs a one-call full
      prefill of the same session tokens (`_debug_no_snapshot`), warm
      (extend) and with another conversation cached.
  S3  session no leak: N session decides alternating two conversations in
      pairs (extend and cold both exercised); fdinfo growth < 64 MiB.
  S4  session stale cache (exact env): with an unrelated conversation
      cached (cold) and with one sharing a long prefix cached (resume), the
      answers equal the warm answers (Δ == 0) and the next chat turn reuses
      the decide's conversation. INCONCLUSIVE if resume was not exercised.
  S5  session error replies: both / neither / empty / non-list messages ->
      422, then a session decide still extends with zero delta.
```

(b) Replace the whole `gate_leak` function with:

```python
def measure_leak(d, n, step, what):
    """Run step(i) n times. Growth of the daemon's own amdgpu GTT + VRAM
    (fdinfo) must stay < 64 MiB; the device-wide sysfs sum is printed
    alongside and is the fallback when fdinfo is unreadable."""
    g0, v0 = device_mem()
    p0 = process_mem(d.p.pid)
    for i in range(n):
        step(i)
    g1, v1 = device_mem()
    p1 = process_mem(d.p.pid)
    mib = lambda x: f"{x / 2**20:.1f}"  # noqa: E731
    if g0 is None and v0 is None:
        return False, "no mem_info_gtt_used / mem_info_vram_used sysfs node readable — cannot measure"
    dg = (g1 - g0) if g0 is not None and g1 is not None else 0
    dv = (v1 - v0) if v0 is not None and v1 is not None else 0
    sys_s = (f"system-wide (sysfs, all cards) {mib(dg + dv)} MiB "
             f"[gtt {mib(dg) if g0 is not None else 'n/a'} + vram {mib(dv) if v0 is not None else 'n/a'}]")
    # The sysfs counters are device-wide: on a shared workstation other
    # processes move them by tens of MiB over a 1000-decide run (observed
    # -101.6 and +64.3 MiB while the daemon's own fdinfo total moved 2 MiB).
    # When the daemon's own DRM fdinfo is readable it is the criterion; the
    # device-wide sum is printed alongside and is the fallback.
    if p0 is not None and p1 is not None:
        pg, pv = p1[0] - p0[0], p1[1] - p0[1]
        grew = pg + pv
        basis = f"daemon process (fdinfo) {mib(grew)} MiB [gtt {mib(pg)} + vram {mib(pv)}]"
    else:
        grew = dg + dv
        basis = "daemon fdinfo unreadable, using system-wide"
    ok = grew < 64 * 1024 * 1024
    return ok, f"device memory growth over {n} {what}: {basis} (<64); {sys_s}"


def gate_leak(d, n):
    need_answers(d.decide(STATE, QUESTIONS), "warmup")
    return measure_leak(
        d, n, lambda i: need_answers(d.decide(STATE, QUESTIONS), f"leak iter {i}"), "decides")
```

(c) Directly after the line `JEV_REQUEST_LIMIT = 64000`, add the session
gates:

```python
# ---- session mode (spec §12.8) ---------------------------------------------
SESSION_SYS = "You are a customer support agent for an online shop."
CONV_C = [{"role": "system", "content": SESSION_SYS},
          {"role": "user", "content": "Order #3527 was placed 21 days ago and tracking has not "
                                      "updated. Reply in one short sentence."}]
CONV_U = [{"role": "system", "content": "You are a poet."},
          {"role": "user", "content": "Write one short line about rain."}]
# A 2-3k-token system turn shared by CONV_L and CONV_X: the chat prefill of
# one leaves a DeltaNet checkpoint inside the prefix the other shares (the
# first checkpoint lands at the first prefill chunk, ar.rs:3880).
LONG_SYS = SESSION_SYS + " Policy notes: " + LOREM * 400
CONV_L = [{"role": "system", "content": LONG_SYS}, CONV_C[1]]
CONV_X = [{"role": "system", "content": LONG_SYS},
          {"role": "user", "content": "Name three primary colours in one short sentence."}]
FOLLOW = {"role": "user", "content": "Thanks. What should I do next? Reply in one short sentence."}
SESSION_Q = {
    "escalate": {"type": "noul",
                 "instructions": "Should this conversation be escalated to a human agent?"},
    "topic": {"type": "choice", "instructions": "What is the conversation about?",
              "criteria": {"shipping": "Delivery, tracking, lost parcels",
                           "billing": "Payments, refunds", "other": "Anything else"}},
    "mood": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Furious"]},
}


def chat_turn(d, messages, max_tokens=256):
    """One greedy chat turn; returns the conversation extended by the reply.
    The chat path does not store a length-capped turn in its prompt cache,
    which would make the reuse gates vacuous, so that fails the setup."""
    text, done = d.chat(messages, max_tokens=max_tokens)
    if done.get("finish_reason") != "stop":
        raise AssertionError(f"setup: chat turn finish_reason={done.get('finish_reason')!r} "
                             f"(need 'stop')")
    return messages + [{"role": "assistant", "content": text}]


def session(d, messages, what, **extra):
    r = d.session_decide(messages, SESSION_Q, **extra)
    return need_answers(r, what), r.get("timing", {})


def max_dlogp(a, b):
    return max(dlogp(a[n], b[n]) for n in a)


def gate_session_reuse(d, with_delta, reuse=True):
    d.reset()
    conv = chat_turn(d, CONV_C)
    nxt = conv + [FOLLOW]
    base_text, base_done = d.chat(nxt)
    base_cached = base_done.get("cached_tokens", 0)
    if not reuse or base_cached == 0:
        return INCONCLUSIVE, (f"no-decide baseline cached_tokens={base_cached}: no chat prompt "
                              f"cache to preserve on this arch/config")
    d.reset()
    if chat_turn(d, CONV_C) != conv:
        return False, "setup: greedy turn 1 is not reproducible after reset"
    _, t = session(d, nxt if with_delta else conv, "session decide")
    text, done = d.chat(nxt)
    cached = done.get("cached_tokens", 0)
    cpt = t.get("conversation_prefill_tokens")
    delta_ok = (cpt or 0) > 0 if with_delta else cpt == 0
    ok = (text == base_text and len(text) > 0 and t.get("start") == "extend" and delta_ok
          and cached == t.get("prefix_tokens") and cached >= base_cached)
    return ok, (f"{'delta' if with_delta else 'no delta'}: next-turn text identical="
                f"{text == base_text}; decide start={t.get('start')} (extend), "
                f"conversation_prefill_tokens={cpt} ({'>0' if with_delta else '==0'}); "
                f"next turn cached_tokens={cached} (== decide prefix_tokens "
                f"{t.get('prefix_tokens')}, >= no-decide {base_cached}); "
                f"texts {base_text[:40]!r} / {text[:40]!r}")


def gate_session_exact(d, reuse=True):
    d.reset()
    conv = chat_turn(d, CONV_C)
    warm, tw = session(d, conv, "warm session")
    ref, _ = session(d, conv, "full-prefill reference", _debug_no_snapshot=True)
    # The reference ends reset (v1 semantics), clearing the assistant-turn
    # cache: rebuild it, then cache a different conversation.
    d.reset()
    if chat_turn(d, CONV_C) != conv:
        return False, "setup: greedy turn 1 is not reproducible after reset"
    chat_turn(d, CONV_U)
    other, to = session(d, conv, "session with another conversation cached")
    dw, do = max_dlogp(warm, ref), max_dlogp(other, ref)
    starts_ok = (tw.get("start") == "extend" and to.get("start") != "extend") if reuse else True
    ok = dw == 0.0 and do == 0.0 and starts_ok
    return ok, (f"exact env: max |Δlogp| vs one-call full prefill of the same tokens: warm "
                f"(start={tw.get('start')}) = {dw:.3g}, another conversation cached "
                f"(start={to.get('start')}) = {do:.3g} (both ==0)")


def gate_session_stale(d, reuse=True):
    d.reset()
    conv = chat_turn(d, CONV_L)
    ref, _ = session(d, conv, "own conversation cached")
    chat_turn(d, CONV_U)
    cold, tc = session(d, conv, "unrelated conversation cached")
    chat_turn(d, CONV_X)
    res, tr = session(d, conv, "shared-prefix conversation cached")
    _, done = d.chat(conv + [FOLLOW], max_tokens=16)
    dc, dr = max_dlogp(cold, ref), max_dlogp(res, ref)
    ok = dc == 0.0 and dr == 0.0
    if reuse:
        ok = ok and tc.get("start") == "cold" and done.get("cached_tokens") == tr.get("prefix_tokens")
    detail = (f"exact env: unrelated cache start={tc.get('start')} Δ={dc:.3g}; shared-prefix cache "
              f"start={tr.get('start')} (cached_tokens={tr.get('cached_tokens')}) Δ={dr:.3g} (==0); "
              f"next chat cached_tokens={done.get('cached_tokens')} (== {tr.get('prefix_tokens')})")
    if ok and reuse and tr.get("start") != "resume":
        return INCONCLUSIVE, detail + " — no checkpoint inside the shared prefix: resume not exercised"
    return ok, detail


def gate_session_leak(d, n, reuse=True):
    d.reset()
    conv_a = chat_turn(d, CONV_C)
    conv_b = chat_turn(d, CONV_U)
    session(d, conv_a, "warmup")
    starts = {}

    def step(i):
        _, t = session(d, conv_a if (i // 2) % 2 == 0 else conv_b, f"leak iter {i}")
        starts[t.get("start")] = starts.get(t.get("start"), 0) + 1

    ok, detail = measure_leak(d, n, step, "session decides")
    want = {"extend", "cold"} if reuse else {"cold"}
    covered = want <= set(starts)
    return ok and covered, f"{detail}; starts={starts} (need {sorted(want)})"


def gate_session_errors(d, reuse=True):
    d.reset()
    conv = chat_turn(d, CONV_C)
    session(d, conv, "commit")
    notes, ok = [], True
    for label, fields in [("state and messages", {"state": "s", "messages": conv}),
                          ("neither", {}),
                          ("empty messages", {"messages": []}),
                          ("messages not a list", {"messages": "hi"})]:
        d.send({"type": "decide", "id": "e", "questions": SESSION_Q, **fields})
        e = d.recv_until({"decided", "error"}).get("error") or {}
        ok_i = e.get("status") == 422
        notes.append(f"{label}: status={e.get('status')} -> {ok_i}")
        ok &= ok_i
    _, t = session(d, conv, "after refusals")
    want = "extend" if reuse else "cold"
    after_ok = t.get("start") == want and (not reuse or t.get("conversation_prefill_tokens") == 0)
    notes.append(f"session decide after: start={t.get('start')} (=={want}), "
                 f"conversation_prefill_tokens={t.get('conversation_prefill_tokens')}")
    return ok and after_ok, "; ".join(notes)
```

(d) In `main()`, replace the exact-daemon call and the `default_body` list
with:

```python
        reuse = cask is None
        results += with_daemon(a.model, a.max_seq, log("exact"), EXACT_ENV, lambda d: [
            run("1a exact-mode bit-exactness", gate_exact_mode, d),
            run("S1b session reuse with delta (exact)", gate_session_reuse, d, True, reuse),
            run("S2 session exactness (exact)", gate_session_exact, d, reuse),
            run("S4 session stale cache (exact)", gate_session_stale, d, reuse)], cask)

        def default_body(d):
            r = []
            if floor is not None:
                r.append(run("1b default-mode noise floor", gate_noise_floor, d, floor))
            r += [run("1c restore order-invariance", gate_restore_order, d),
                  run("2 question isolation", gate_isolation, d),
                  run("3 model left clean", gate_clean, d),
                  run("4 no leak", gate_leak, d, a.leak_n),
                  run("5 error replies", gate_errors, d, a.max_seq, cask),
                  run("S1a session reuse, no delta", gate_session_reuse, d, False, reuse),
                  run("S3 session no leak", gate_session_leak, d, a.leak_n, reuse),
                  run("S5 session error replies", gate_session_errors, d, reuse)]
            return r
```

- [ ] **Step 3: Syntax check** (no GPU)

```bash
python3 -m py_compile scripts/jev_eval/gates.py scripts/jev_eval/daemon_client.py
python3 -c "import sys; sys.path.insert(0,'scripts/jev_eval'); import gates; print(len(gates.LONG_SYS))"
```

Expected: no error, and a length > 10000 characters.

- [ ] **Step 4: Run the gates on Qwen3.5-4B.** This takes longer than 10
  minutes, so run it in the background and poll.

```bash
export PATH=/opt/rocm-7.2.2/bin:$PATH
pgrep -a -x daemon || echo "no daemon running"
grep MemAvailable /proc/meminfo
```

Continue only if `pgrep` printed `no daemon running` and MemAvailable is
≥ 8 GiB. If another daemon is running, it belongs to someone else, for
example a long evaluation from another worktree. Wait, or ask the user;
never kill it.

```bash
cargo build --release -p hipfire-daemon -p hipfire-cli
cargo build --release -p hipfire-generate --features lab --example split_prefill_probe
mkdir -p target/jev-gates
nohup python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4 \
  --daemon-log target/jev-gates/q35 > target/jev-gates/q35.out 2>&1 &
echo $! > target/jev-gates/q35.pid
```

Poll with short commands until the `summary:` line appears:
`tail -n 3 target/jev-gates/q35.out`. Stop only the PID in
`target/jev-gates/q35.pid` if you must abort.

Expected:

- every v1 gate line is unchanged;
- `S1a`, `S1b`, `S2`, `S3` and `S5` report PASS;
- `S4` reports PASS, or INCONCLUSIVE with the "resume not exercised" note;
- the summary shows 0 FAIL.

If a gate FAILs:

- **S1a/S1b text differs:**
  1. Compare the decide's `prefix_tokens` with the baseline's
     `cached_tokens`.
  2. Rerun with `HIPFIRE_QWEN_CACHE_TRACE=1` in the gate's daemon env, and
     read the `[qwen-cache lcp]` / `ids` lines in
     `target/jev-gates/q35.default`.
  3. Report what you find before changing anything.
- **S2 Δ ≠ 0:** apply v1's procedure. Rerun with a single-question
  `SESSION_Q`, and separate restore residue from split numerics.

Don't loosen a threshold.

- [ ] **Step 5: CASK and the Llama carrier** (each in the background as
  above, sequentially, one daemon at a time)

```bash
nohup python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4 --cask \
  --daemon-log target/jev-gates/q35cask > target/jev-gates/q35cask.out 2>&1 &
```

Expected: every session start reports `cold`, S1a and S1b are
INCONCLUSIVE (no chat cache under eviction), and the rest PASS.

Then:

```bash
nohup python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3-0.6b.hf4 \
  --daemon-log target/jev-gates/llama > target/jev-gates/llama.out 2>&1 &
```

Expected on the Llama carrier:

- S1 is PASS or INCONCLUSIVE: arch 0/1 is not `cache_capable` in serve,
  but the daemon's LCP block may still hit;
- S4 is INCONCLUSIVE: there are no DeltaNet checkpoints, so no resume;
- everything else PASSes.

Confirm the printed `arch=` belongs to `LlamaCarrier`, as in v1 Task 7.

- [ ] **Step 6: README, final checks, commit**

In `scripts/jev_eval/README.md`, add under item 1:

```markdown
   The same runs include the session-mode gates S1–S5 (spec §12.8). They
   need no extra flags; under `--cask` every session start is `cold` and S1
   is INCONCLUSIVE by design (the chat prompt cache is off under eviction).
```

```bash
./scripts/no-gpu-ci.sh
bash scripts/leanup-ratchets.sh
python3 scripts/check-crate-maps.py --check
git add scripts/jev_eval/daemon_client.py scripts/jev_eval/gates.py scripts/jev_eval/README.md
git commit -m "test(decide): session-mode GPU gates S1-S5" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Record the S1–S5 lines from all three runs for the PR description.

- [ ] **Step 7: Hand back.** Don't push or open a PR without the user's
  go-ahead, and never push to `fork`.
