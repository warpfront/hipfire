# Jev-compatible Decide Endpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jev-wire-compatible `POST /v1/systemone` endpoint that answers
typed questions (choice / score / noul) about a state by reading next-token
logits over single-token option codes. No decoding.

**Architecture:**

- `hipfire-engine::decide` (pure):
  - parses and validates the request;
  - assigns tokenizer-checked codes;
  - builds prompt text;
  - turns label logits into Jev answers.
- `hipfire-loader` `Carrier` hooks (Qwen3.5 + Llama): prefill → logits,
  snapshot save and restore.
- `hipfire-generate::decide::run_decide` orchestrates:
  - render every question's full prompt;
  - prefill the longest common token prefix once and snapshot it;
  - per question, restore → prefill suffix → gather label logits;
  - roll back.
- The daemon exposes this as a `decide` JSONL command that replies with one
  `decided` message. Serve adds the HTTP route.

**Tech Stack:** Rust (workspace crates above), serde_json, hyper (serve),
Python 3 stdlib (fake daemon, eval harness).

**Spec:** `docs/specs/2026-09-28-jev-decide-design.md`. Read it first.

## Global Constraints

- Wire format is Jev's exactly:
  - request `{model, state, questions}`;
  - response `{model, answers, usage}`;
  - answers carry `type`;
  - no extra body fields. Timings go only in the `x-hipfire-timing` header.
- Limits:
  - `choice`: 2–255 criteria keys;
  - `score`: 2–10 levels;
  - `noul`: optional `criteria {true,false}`;
  - `instructions`: a non-empty string;
  - `questions`: non-empty.
- Confidence:
  - `choice` = (p_max − 1/K)/(1 − 1/K);
  - `score` = p_max;
  - `noul` has no confidence field.
- Probabilities are never rounded.
- `usage.input_tokens` = prefix_len + Σ(len_i − prefix_len); `output_tokens`
  = 0.
- Status codes: 422 validation, 400 unsupported arch, 409 unsupported
  runtime state, 503 no model or admission saturated, 500 daemon/GPU failure.
- Layering (`docs/ARCHITECTURE.md`):
  - `hipfire-engine` gets no arch-type logic;
  - the daemon only dispatches;
  - architecture-specific code goes behind `Carrier` hooks.
- `DeltaNetSnapshot` has no `Drop`. Every snapshot must reach `free_gpu`.
- After any decide the model is left reset: `production_fail_closed_rollback`,
  then `state_epoch += 1`.
- Environment:
  - branch `feat/jev-decide` in `~/repos/hipfire-jev`;
  - never push to the `fork` remote;
  - do not touch `~/repos/hipfire`, which is another checkout on a
    different branch.
- GPU runs on this workstation:
  - one daemon at a time;
  - check `MemAvailable` before loading 27B;
  - serve must use the repo build via
    `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon`, not the stale
    `~/.hipfire/bin/daemon`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `crates/hipfire-engine/src/decide.rs` | create | Request types, validation, codes, prompt text, LCP, answer assembly. Pure; unit-tested. |
| `crates/hipfire-engine/src/lib.rs` | modify | `pub mod decide;` |
| `crates/hipfire-loader/src/decide.rs` | create | `DecideSnapshot` (seq_pos + optional `DeltaNetSnapshot`) with `free`. |
| `crates/hipfire-loader/src/lib.rs` | modify | `pub mod decide;` plus 3 default `Carrier` hooks. |
| `crates/hipfire-loader/src/carriers.rs` | modify | Hook impls for `Qwen35Carrier` and `LlamaCarrier`. |
| `crates/hipfire-generate/src/decide.rs` | create | `run_decide` orchestration and `DecideError`. |
| `crates/hipfire-generate/src/lib.rs` | modify | `pub mod decide;` |
| `crates/hipfire-daemon/src/main.rs` | modify | `"decide"` dispatch arm. |
| `crates/hipfire-cli/src/serve/decide.rs` | create | `handle_decide`: model selection, admission, daemon call, retry-on-`required_max_seq`, response shaping. |
| `crates/hipfire-cli/src/serve/mod.rs` | modify | `pub(crate) mod decide;` |
| `crates/hipfire-cli/src/serve/http.rs` | modify | Route arm `POST /v1/systemone`. |
| `crates/hipfire-cli/src/serve/fake_daemon.py` | modify | `decide` branch. |
| `crates/hipfire-cli/src/main.rs` | modify | Task 11 harness tests for the route. |
| `scripts/jev_eval/daemon_client.py` | create | Minimal JSONL daemon driver used by gates. |
| `scripts/jev_eval/gates.py` | create | GPU correctness gates 1–4. |
| `scripts/jev_eval/run_jevbench.py` | create | jev-bench against local serve, output redirected. |
| `scripts/jev_eval/run_calibration.py` | create | jev-ood-calibration adapter. |
| `scripts/jev_eval/report.py` | create | Accuracy/ECE vs Jev's committed answers; latency. |
| `scripts/jev_eval/README.md` | create | How to run all of the above. |
| `scripts/no-gpu-ci.sh` | modify | Add `cargo test -p hipfire-engine --lib decide`. |

---

### Task 1: Request parsing and validation (`hipfire-engine::decide`)

**Files:**
- Create: `crates/hipfire-engine/src/decide.rs`
- Modify: `crates/hipfire-engine/src/lib.rs` (add `pub mod decide;` after
  `pub mod emit;`)

**Interfaces:**
- Produces:
  - `pub enum QuestionKind { Choice { keys: Vec<String>, descriptions: Vec<String> }, Score { levels: Vec<String> }, Noul { true_desc: Option<String>, false_desc: Option<String> } }`
  - `pub struct Question { pub name: String, pub instructions: String, pub kind: QuestionKind }`
  - `pub struct DecideRequest { pub state_text: String, pub questions: Vec<Question> }`
  - `pub fn parse_request(v: &serde_json::Value) -> Result<DecideRequest, String>`.
    `Err` is always a 422 message naming the question and field.
  - `pub fn render_state(v: &serde_json::Value) -> String`

- [ ] **Step 1: Write the failing tests** (bottom of the new
  `crates/hipfire-engine/src/decide.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_all_three_types() {
        let v = json!({
            "model": "jev-latest",
            "state": {"body": "hi"},
            "questions": {
                "q_choice": {"type": "choice", "instructions": "Which?",
                             "criteria": {"billing": "money", "shipping": "parcels"}},
                "q_score": {"type": "score", "instructions": "How bad?",
                            "criteria": ["low", "mid", "high"]},
                "q_noul": {"type": "noul", "instructions": "Angry?"}
            }
        });
        let r = parse_request(&v).unwrap();
        assert_eq!(r.questions.len(), 3);
        assert!(r.state_text.contains("\"body\": \"hi\""));
        let c = r.questions.iter().find(|q| q.name == "q_choice").unwrap();
        match &c.kind {
            QuestionKind::Choice { keys, descriptions } => {
                assert_eq!(keys, &vec!["billing".to_string(), "shipping".to_string()]);
                assert_eq!(descriptions, &vec!["money".to_string(), "parcels".to_string()]);
            }
            _ => panic!("wrong kind"),
        }
        let s = r.questions.iter().find(|q| q.name == "q_score").unwrap();
        assert!(matches!(&s.kind, QuestionKind::Score { levels } if levels.len() == 3));
        let n = r.questions.iter().find(|q| q.name == "q_noul").unwrap();
        assert!(matches!(&n.kind, QuestionKind::Noul { true_desc: None, false_desc: None }));
    }

    #[test]
    fn string_state_is_used_verbatim() {
        assert_eq!(render_state(&json!("plain text")), "plain text");
    }

    fn err_of(v: serde_json::Value) -> String {
        match parse_request(&v) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let q = |q: serde_json::Value| json!({"state": "s", "questions": {"q": q}});
        assert!(err_of(json!({"questions": {}})).contains("state"));
        assert!(err_of(json!({"state": 5, "questions": {"q": {}}})).contains("state"));
        assert!(err_of(json!({"state": "s", "questions": {}})).contains("questions"));
        assert!(err_of(q(json!({"type": "choice", "criteria": {"a": "x", "b": "y"}})))
            .contains("q.instructions"));
        assert!(err_of(q(json!({"type": "choice", "instructions": "i", "criteria": {"a": "x"}})))
            .contains("2..=255"));
        let many: serde_json::Map<String, serde_json::Value> =
            (0..256).map(|i| (format!("k{i}"), json!("d"))).collect();
        assert!(err_of(q(json!({"type": "choice", "instructions": "i", "criteria": many})))
            .contains("2..=255"));
        assert!(err_of(q(json!({"type": "score", "instructions": "i", "criteria": ["a"]})))
            .contains("2..=10"));
        assert!(err_of(q(json!({"type": "score", "instructions": "i",
                                "criteria": ["a","b","c","d","e","f","g","h","i","j","k"]})))
            .contains("2..=10"));
        assert!(err_of(q(json!({"type": "boolean", "instructions": "i"}))).contains("q.type"));
        assert!(err_of(q(json!({"type": "noul", "instructions": "i", "criteria": {"true": 1}})))
            .contains("q.criteria"));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: compile error, `cannot find function parse_request`.

- [ ] **Step 3: Implement** (top of `crates/hipfire-engine/src/decide.rs`,
  above the tests)

```rust
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Jev-compatible "decide": typed questions answered by next-token logit
//! readout over single-token option codes. Pure — no GPU, no arch types.
//! Spec: `docs/specs/2026-09-28-jev-decide-design.md`.

use serde_json::{json, Value};

pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_SCORE_LEVELS: usize = 10;

#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    Choice { keys: Vec<String>, descriptions: Vec<String> },
    Score { levels: Vec<String> },
    Noul { true_desc: Option<String>, false_desc: Option<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub name: String,
    pub instructions: String,
    pub kind: QuestionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecideRequest {
    pub state_text: String,
    pub questions: Vec<Question>,
}

/// String state verbatim; object/array pretty-printed so backtick `field`
/// references in instructions resolve against visible keys.
pub fn render_state(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

fn desc_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Validate a Jev request. Every `Err` is a 422 message naming the field.
pub fn parse_request(v: &Value) -> Result<DecideRequest, String> {
    let state = v.get("state").ok_or("state is required")?;
    if !(state.is_string() || state.is_object() || state.is_array()) {
        return Err("state must be a string, JSON object or JSON array".into());
    }
    let qs = v
        .get("questions")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .ok_or("questions must be a non-empty object of name -> question")?;
    let mut questions = Vec::with_capacity(qs.len());
    for (name, q) in qs {
        let instructions = q
            .get("instructions")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| format!("{name}.instructions must be a non-empty string"))?
            .to_string();
        let ty = q.get("type").and_then(Value::as_str).unwrap_or("");
        let criteria = q.get("criteria");
        let kind = match ty {
            "choice" => {
                let obj = criteria
                    .and_then(Value::as_object)
                    .ok_or_else(|| format!("{name}.criteria must be an object of option -> description"))?;
                if !(2..=MAX_CHOICE_OPTIONS).contains(&obj.len()) {
                    return Err(format!(
                        "{name}.criteria must have 2..={MAX_CHOICE_OPTIONS} options, got {}",
                        obj.len()
                    ));
                }
                QuestionKind::Choice {
                    keys: obj.keys().cloned().collect(),
                    descriptions: obj.values().map(desc_string).collect(),
                }
            }
            "score" => {
                let arr = criteria
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("{name}.criteria must be an array of level descriptions"))?;
                if !(2..=MAX_SCORE_LEVELS).contains(&arr.len()) {
                    return Err(format!(
                        "{name}.criteria must have 2..={MAX_SCORE_LEVELS} levels, got {}",
                        arr.len()
                    ));
                }
                QuestionKind::Score { levels: arr.iter().map(desc_string).collect() }
            }
            "noul" => {
                let (mut t, mut f) = (None, None);
                if let Some(c) = criteria {
                    let obj = c
                        .as_object()
                        .ok_or_else(|| format!("{name}.criteria must be an object {{true, false}}"))?;
                    for (k, val) in obj {
                        let s = val
                            .as_str()
                            .ok_or_else(|| format!("{name}.criteria.{k} must be a string"))?
                            .to_string();
                        match k.as_str() {
                            "true" => t = Some(s),
                            "false" => f = Some(s),
                            _ => return Err(format!("{name}.criteria keys must be true/false")),
                        }
                    }
                }
                QuestionKind::Noul { true_desc: t, false_desc: f }
            }
            other => {
                return Err(format!(
                    "{name}.type must be choice, score or noul, got {other:?}"
                ))
            }
        };
        questions.push(Question { name: name.clone(), instructions, kind });
    }
    Ok(DecideRequest { state_text: render_state(state), questions })
}
```

`json!` is used by later tasks in this file. If clippy flags it as unused at
this point, keep the import anyway; Task 3 uses it.

Note: `serde_json::Map` iteration order is alphabetical unless the
`preserve_order` feature is enabled. Question order therefore depends on that
feature. Answers are keyed by name, so order never affects output, and the
tests above look questions up by name.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-engine/src/decide.rs crates/hipfire-engine/src/lib.rs
git commit -m "feat(decide): Jev request parsing and validation"
```

---

### Task 2: Option codes, prompt text, shared prefix

**Files:**
- Modify: `crates/hipfire-engine/src/decide.rs`

**Interfaces:**
- Consumes: `Question`, `QuestionKind` (Task 1).
- Produces:
  - `pub const SYSTEM_PROMPT: &str`
  - `pub fn assign_codes(k: usize, is_single_token: impl Fn(&str) -> bool) -> Result<Vec<String>, String>`
  - `pub fn labels_for(q: &Question, is_single_token: impl Fn(&str) -> bool) -> Result<Vec<String>, String>`
    returns the readout label strings, in the order `assemble_answer`
    (Task 3) expects:
    - choice: codes in key order;
    - score: `"0".."k-1"`;
    - noul: `["Yes", "No"]`.
  - `pub fn build_user_text(state_text: &str, q: &Question, labels: &[String]) -> String`
  - `pub fn shared_prefix_len(seqs: &[Vec<u32>]) -> usize`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests`)

```rust
    #[test]
    fn codes_single_letters_first_then_two_letter() {
        let all = |_: &str| true;
        let c = assign_codes(28, all).unwrap();
        assert_eq!(&c[..3], &["A", "B", "C"]);
        assert_eq!(c[25], "Z");
        assert_eq!(&c[26..], &["AA", "AB"]);
    }

    #[test]
    fn codes_skip_multi_token_candidates() {
        // Pretend "B" and "AA" tokenize to more than one token.
        let c = assign_codes(27, |s: &str| s != "B" && s != "AA").unwrap();
        assert!(!c.contains(&"B".to_string()));
        assert!(!c.contains(&"AA".to_string()));
        assert_eq!(c.len(), 27);
        let uniq: std::collections::HashSet<_> = c.iter().collect();
        assert_eq!(uniq.len(), 27);
    }

    #[test]
    fn codes_error_when_tokenizer_cannot_supply_k() {
        let e = assign_codes(5, |s: &str| s.len() == 1 && s < "C").unwrap_err();
        assert!(e.contains("single-token"));
    }

    #[test]
    fn labels_per_type() {
        let all = |_: &str| true;
        let score = Question { name: "s".into(), instructions: "i".into(),
            kind: QuestionKind::Score { levels: vec!["a".into(), "b".into(), "c".into()] } };
        assert_eq!(labels_for(&score, all).unwrap(), vec!["0", "1", "2"]);
        let noul = Question { name: "n".into(), instructions: "i".into(),
            kind: QuestionKind::Noul { true_desc: None, false_desc: None } };
        assert_eq!(labels_for(&noul, all).unwrap(), vec!["Yes", "No"]);
    }

    #[test]
    fn labels_error_when_digit_or_yes_no_is_not_single_token() {
        let score = Question { name: "s".into(), instructions: "i".into(),
            kind: QuestionKind::Score { levels: vec!["a".into(), "b".into()] } };
        assert!(labels_for(&score, |s: &str| s != "1").is_err());
        let noul = Question { name: "n".into(), instructions: "i".into(),
            kind: QuestionKind::Noul { true_desc: None, false_desc: None } };
        assert!(labels_for(&noul, |s: &str| s != "Yes").is_err());
    }

    #[test]
    fn user_text_layout() {
        let q = Question { name: "q".into(), instructions: "Which queue?".into(),
            kind: QuestionKind::Choice { keys: vec!["billing".into(), "shipping".into()],
                descriptions: vec!["money".into(), "parcels".into()] } };
        let t = build_user_text("the state", &q, &["A".into(), "B".into()]);
        assert_eq!(t, "State:\nthe state\n\nQuestion: Which queue?\nOptions:\nA. money\nB. parcels\n\nAnswer with the option code only.");
        let n = Question { name: "n".into(), instructions: "Angry?".into(),
            kind: QuestionKind::Noul { true_desc: Some("shouting".into()), false_desc: None } };
        let t = build_user_text("s", &n, &["Yes".into(), "No".into()]);
        assert!(t.ends_with("Options:\nYes. shouting\nNo. no\n\nAnswer with the option code only."));
    }

    #[test]
    fn shared_prefix_is_lcp_capped_below_shortest() {
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3, 9], vec![1, 2, 3, 8, 7]]), 3);
        // Identical sequences: cap at len - 1 so each suffix is non-empty.
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3]]), 2);
        assert_eq!(shared_prefix_len(&[vec![5, 6, 7]]), 2);
        assert_eq!(shared_prefix_len(&[vec![1], vec![2]]), 0);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: compile errors for `assign_codes`, `labels_for`,
`build_user_text` and `shared_prefix_len`.

- [ ] **Step 3: Implement** (append above `#[cfg(test)]`)

```rust
pub const SYSTEM_PROMPT: &str =
    "You answer classification questions about the state. Reply with one option code.";

/// `A`..`Z`, then `AA`..`ZZ`, keeping only candidates the model's tokenizer
/// encodes as exactly one token. Deterministic order.
pub fn assign_codes(k: usize, is_single_token: impl Fn(&str) -> bool) -> Result<Vec<String>, String> {
    let letters: Vec<char> = ('A'..='Z').collect();
    let singles = letters.iter().map(|c| c.to_string());
    let doubles = letters
        .iter()
        .flat_map(|a| letters.iter().map(move |b| format!("{a}{b}")));
    let codes: Vec<String> = singles.chain(doubles).filter(|c| is_single_token(c)).take(k).collect();
    if codes.len() < k {
        return Err(format!(
            "model tokenizer supplies only {} single-token option codes, need {k}",
            codes.len()
        ));
    }
    Ok(codes)
}

pub fn labels_for(q: &Question, is_single_token: impl Fn(&str) -> bool) -> Result<Vec<String>, String> {
    let labels: Vec<String> = match &q.kind {
        QuestionKind::Choice { keys, .. } => return assign_codes(keys.len(), is_single_token),
        QuestionKind::Score { levels } => (0..levels.len()).map(|i| i.to_string()).collect(),
        QuestionKind::Noul { .. } => vec!["Yes".into(), "No".into()],
    };
    if let Some(bad) = labels.iter().find(|l| !is_single_token(l)) {
        return Err(format!("{}: label {bad:?} is not a single token for this model", q.name));
    }
    Ok(labels)
}

/// The user-turn text for one question. The state block comes first so every
/// question's rendered prompt shares the longest possible token prefix.
pub fn build_user_text(state_text: &str, q: &Question, labels: &[String]) -> String {
    let descs: Vec<String> = match &q.kind {
        QuestionKind::Choice { descriptions, .. } => descriptions.clone(),
        QuestionKind::Score { levels } => levels.clone(),
        QuestionKind::Noul { true_desc, false_desc } => vec![
            true_desc.clone().unwrap_or_else(|| "yes".into()),
            false_desc.clone().unwrap_or_else(|| "no".into()),
        ],
    };
    let mut s = format!("State:\n{state_text}\n\nQuestion: {}\nOptions:\n", q.instructions);
    for (label, desc) in labels.iter().zip(descs.iter()) {
        s.push_str(&format!("{label}. {desc}\n"));
    }
    s.push_str("\nAnswer with the option code only.");
    s
}

/// Longest common prefix of all sequences, capped at `min_len - 1` so each
/// question keeps a non-empty suffix whose last token yields the readout.
pub fn shared_prefix_len(seqs: &[Vec<u32>]) -> usize {
    let Some(first) = seqs.first() else { return 0 };
    let min_len = seqs.iter().map(Vec::len).min().unwrap_or(0);
    let mut n = 0;
    while n < min_len && seqs.iter().all(|s| s[n] == first[n]) {
        n += 1;
    }
    n.min(min_len.saturating_sub(1))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-engine/src/decide.rs
git commit -m "feat(decide): option codes, prompt text, shared-prefix split"
```

---

### Task 3: Answer assembly

**Files:**
- Modify: `crates/hipfire-engine/src/decide.rs`

**Interfaces:**
- Consumes: `Question`, `QuestionKind` (Task 1); label order from
  `labels_for` (Task 2).
- Produces:
  - `pub fn softmax(logits: &[f32]) -> Vec<f64>`
  - `pub fn assemble_answer(q: &Question, label_logits: &[f32]) -> serde_json::Value`.
    `label_logits[i]` is the logit of `labels_for(q)[i]`.
  - `pub fn usage_json(prefix_len: usize, seq_lens: &[usize]) -> serde_json::Value`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests`)

```rust
    fn choice_q(keys: &[&str]) -> Question {
        Question { name: "q".into(), instructions: "i".into(),
            kind: QuestionKind::Choice { keys: keys.iter().map(|s| s.to_string()).collect(),
                descriptions: keys.iter().map(|s| s.to_string()).collect() } }
    }

    #[test]
    fn softmax_is_normalised_and_stable() {
        let p = softmax(&[1000.0, 1000.0]);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
        let p = softmax(&[0.0, (3.0f32).ln()]);
        assert!((p[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn choice_confidence_matches_jev_formula_on_committed_rows() {
        // Probabilities + confidences copied from Jev's committed answers:
        // jev-ood-calibration results/jev_openbookqa.jsonl row 0 (K=4) and
        // jev-bench predictions/banking77.jsonl row 1 (K=77, top only).
        let q = choice_q(&["A", "B", "C", "D"]);
        let probs = [1e-9_f32, 0.69, 0.31, 1e-9];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let a = assemble_answer(&q, &logits);
        assert_eq!(a["type"], "choice");
        assert_eq!(a["choice"], "B");
        let conf = a["confidence"].as_f64().unwrap();
        assert!((conf - 0.58).abs() <= 0.02, "conf {conf}");
        // K=77 with p_top=0.44 → Jev reported 0.42.
        let k = 77;
        let pm = 0.44_f64;
        assert!(((pm - 1.0 / k as f64) / (1.0 - 1.0 / k as f64) - 0.42).abs() <= 0.02);
    }

    #[test]
    fn choice_uniform_has_zero_confidence_and_probs_keyed_by_user_keys() {
        let q = choice_q(&["billing", "shipping", "general"]);
        let a = assemble_answer(&q, &[0.0, 0.0, 0.0]);
        assert!(a["confidence"].as_f64().unwrap().abs() < 1e-12);
        let probs = a["probabilities"].as_object().unwrap();
        assert_eq!(probs.len(), 3);
        assert!((probs["shipping"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn score_mean_and_confidence() {
        let q = Question { name: "s".into(), instructions: "i".into(),
            kind: QuestionKind::Score { levels: vec!["a".into(), "b".into(), "c".into(), "d".into()] } };
        // Jev synth score row: probs [0, .05, .89, .06], confidence .89.
        let probs = [1e-9_f32, 0.05, 0.89, 0.06];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let a = assemble_answer(&q, &logits);
        assert_eq!(a["type"], "score");
        let score = a["score"].as_f64().unwrap();
        assert!((score - (0.05 + 2.0 * 0.89 + 3.0 * 0.06)).abs() < 1e-4, "score {score}");
        assert!((a["confidence"].as_f64().unwrap() - 0.89).abs() < 1e-4);
        assert!(a["probabilities"].get("2").is_some());
    }

    #[test]
    fn noul_is_p_yes_without_confidence() {
        let q = Question { name: "n".into(), instructions: "i".into(),
            kind: QuestionKind::Noul { true_desc: None, false_desc: None } };
        let a = assemble_answer(&q, &[(3.0f32).ln(), 0.0]);
        assert_eq!(a["type"], "noul");
        assert!((a["noul"].as_f64().unwrap() - 0.75).abs() < 1e-6);
        assert!(a.get("confidence").is_none());
    }

    #[test]
    fn usage_is_additive() {
        let u = usage_json(100, &[110, 125]);
        assert_eq!(u["input_tokens"], 135);
        assert_eq!(u["output_tokens"], 0);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: compile errors for `softmax`, `assemble_answer` and `usage_json`.

- [ ] **Step 3: Implement** (append above `#[cfg(test)]`)

```rust
/// Numerically stable softmax in f64.
pub fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&x| (x as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..p.len() {
        if p[i] > p[best] {
            best = i;
        }
    }
    best
}

/// Jev answer JSON for one question. `label_logits` follows `labels_for`
/// order. Probabilities are not rounded.
pub fn assemble_answer(q: &Question, label_logits: &[f32]) -> Value {
    let p = softmax(label_logits);
    let k = p.len() as f64;
    let top = argmax(&p);
    match &q.kind {
        QuestionKind::Choice { keys, .. } => {
            let probs: serde_json::Map<String, Value> =
                keys.iter().cloned().zip(p.iter().map(|&x| json!(x))).collect();
            let confidence = (p[top] - 1.0 / k) / (1.0 - 1.0 / k);
            json!({"type": "choice", "choice": keys[top], "probabilities": probs,
                   "confidence": confidence})
        }
        QuestionKind::Score { .. } => {
            let probs: serde_json::Map<String, Value> =
                p.iter().enumerate().map(|(i, &x)| (i.to_string(), json!(x))).collect();
            let score: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
            json!({"type": "score", "score": score, "probabilities": probs,
                   "confidence": p[top]})
        }
        QuestionKind::Noul { .. } => json!({"type": "noul", "noul": p[0]}),
    }
}

/// Additive accounting observed for Jev: shared prefix once, each suffix once.
pub fn usage_json(prefix_len: usize, seq_lens: &[usize]) -> Value {
    let input: usize = prefix_len + seq_lens.iter().map(|l| l - prefix_len).sum::<usize>();
    json!({"input_tokens": input, "output_tokens": 0})
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p hipfire-engine --lib decide`
Expected: `16 passed`.

- [ ] **Step 5: Add the tests to the no-GPU CI script and commit**

In `scripts/no-gpu-ci.sh`, under `== Rust no-GPU unit tests ==`, add after
`cargo test -p rdna-compute --lib`:

```bash
cargo test -p hipfire-engine --lib decide
```

```bash
git add crates/hipfire-engine/src/decide.rs scripts/no-gpu-ci.sh
git commit -m "feat(decide): answer assembly with Jev confidence formulas"
```

---

### Task 4: Carrier hooks and `DecideSnapshot`

**Files:**
- Create: `crates/hipfire-loader/src/decide.rs`
- Modify: `crates/hipfire-loader/src/lib.rs`:
  - add `pub mod decide;` near the other `pub mod` lines;
  - add three default methods at the end of `pub trait Carrier`.
- Modify: `crates/hipfire-loader/src/carriers.rs`: implement the hooks in
  `impl Carrier for Qwen35Carrier` (≈line 359) and
  `impl Carrier for LlamaCarrier` (≈line 753).

**Interfaces:**
- Produces:
  - `pub struct DecideSnapshot { pub seq_pos: usize, pub recurrent: Option<hipfire_arch_qwen35::speculative::DeltaNetSnapshot> }`
  - `impl DecideSnapshot { pub fn free(self, gpu: &mut rdna_compute::Gpu) }`
  - `Carrier::decide_prefill_logits(&self, m: &mut LoadedModel, gpu: &mut Gpu, tokens: &[u32], start_pos: usize) -> Option<Result<Vec<f32>, String>>`.
    Prefills `tokens` at `start_pos`, sets `m.seq_pos = start_pos + tokens.len()`
    and returns the last position's full-vocab logits. `None` means the
    architecture is unsupported.
  - `Carrier::decide_save(&self, m: &mut LoadedModel, gpu: &mut Gpu) -> Option<Result<DecideSnapshot, String>>`
  - `Carrier::decide_restore(&self, m: &mut LoadedModel, gpu: &mut Gpu, snap: &DecideSnapshot) -> Option<Result<(), String>>`

- [ ] **Step 1: Write the failing test** (in the new
  `crates/hipfire-loader/src/decide.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn only_qwen35_and_llama_carriers_claim_decide() {
        // Default hooks return None without touching the model, so we can
        // probe support through a trait-level flag instead of a GPU.
        for c in crate::registry_carriers() {
            let expected = matches!(c.name(), "qwen35" | "llama");
            assert_eq!(c.decide_supported(), expected, "carrier {}", c.name());
        }
    }
}
```

Before writing this, check the carrier names:
`grep -n 'fn name' crates/hipfire-loader/src/carriers.rs`. If the Qwen3.5
and Llama carriers report different strings, use those exact strings in the
`matches!`. Also check whether the registry array is public:
`grep -n 'static REGISTRY' crates/hipfire-loader/src/lib.rs`. Add this
accessor next to `carrier_for` in `lib.rs`:

```rust
/// All registered carriers (read-only view for capability tests).
pub fn registry_carriers() -> impl Iterator<Item = &'static dyn Carrier> {
    REGISTRY.iter().copied()
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-loader --lib decide`
Expected: compile errors, `decide_supported` / `registry_carriers` not found.

- [ ] **Step 3: Implement `DecideSnapshot`** (top of
  `crates/hipfire-loader/src/decide.rs`)

```rust
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-state snapshot for the decide runner: the KV cursor plus, for
//! hybrid Qwen3.5/3.6, the GatedDeltaNet recurrent state. KV rewind is
//! positional (valid only without eviction/compaction — the runner checks).

use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;

pub struct DecideSnapshot {
    pub seq_pos: usize,
    pub recurrent: Option<DeltaNetSnapshot>,
}

impl DecideSnapshot {
    /// `DeltaNetSnapshot` has no `Drop`; this is the only release path.
    pub fn free(self, gpu: &mut rdna_compute::Gpu) {
        if let Some(r) = self.recurrent {
            r.free_gpu(gpu);
        }
    }
}
```

- [ ] **Step 4: Add the trait defaults** (append inside `pub trait Carrier`
  in `lib.rs`, after the last existing method)

```rust
    // ── decide (Jev-compatible logit readout) ──────────────────────────────
    /// Whether this arch implements the three decide hooks below.
    fn decide_supported(&self) -> bool {
        false
    }
    /// Prefill `tokens` at `start_pos`, set `m.seq_pos = start_pos +
    /// tokens.len()`, return the last position's full-vocab logits.
    fn decide_prefill_logits(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _tokens: &[u32],
        _start_pos: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        None
    }
    /// Snapshot `seq_pos` + recurrent state.
    fn decide_save(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        None
    }
    /// Restore a snapshot taken by `decide_save` on the same model.
    fn decide_restore(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut Gpu,
        _snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        None
    }
```

- [ ] **Step 5: Implement the Qwen3.5 hooks** (inside
  `impl Carrier for Qwen35Carrier` in `carriers.rs`)

```rust
    fn decide_supported(&self) -> bool {
        true
    }
    fn decide_prefill_logits(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        start_pos: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let r = hipfire_arch_qwen35::qwen35::forward_prefill_batch(
            gpu, &b.weights, &b.config, tokens, start_pos, &mut b.kv_cache, &mut b.dn_state,
            &b.scratch, None, None, None, None,
        )
        .map_err(|e| format!("decide prefill: {e:?}"))
        .and_then(|()| gpu.download_f32(&b.scratch.logits).map_err(|e| format!("decide logits: {e:?}")));
        if r.is_ok() {
            m.seq_pos = start_pos + tokens.len();
        }
        Some(r)
    }
    fn decide_save(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        let seq_pos = m.seq_pos;
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let mut snap = match DeltaNetSnapshot::new_for(gpu, &b.dn_state) {
            Ok(s) => s,
            Err(e) => return Some(Err(format!("decide snapshot alloc: {e:?}"))),
        };
        if let Err(e) = snap.save_from(&b.dn_state, gpu) {
            snap.free_gpu(gpu);
            return Some(Err(format!("decide snapshot save: {e:?}")));
        }
        Some(Ok(crate::decide::DecideSnapshot { seq_pos, recurrent: Some(snap) }))
    }
    fn decide_restore(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        let Some(b) = m.qwen35_mut() else {
            return Some(Err("decide: qwen35 bundle missing".into()));
        };
        let Some(rec) = snap.recurrent.as_ref() else {
            return Some(Err("decide: qwen35 snapshot has no recurrent state".into()));
        };
        if let Err(e) = rec.restore_to(&mut b.dn_state, gpu) {
            return Some(Err(format!("decide restore: {e:?}")));
        }
        m.seq_pos = snap.seq_pos;
        Some(Ok(()))
    }
```

Check that `DeltaNetSnapshot` is in scope in `carriers.rs`
(`grep -n DeltaNetSnapshot crates/hipfire-loader/src/carriers.rs`). If it
isn't, add `use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;` at the
top.

- [ ] **Step 6: Implement the Llama hooks** (inside
  `impl Carrier for LlamaCarrier`)

```rust
    fn decide_supported(&self) -> bool {
        true
    }
    fn decide_prefill_logits(
        &self,
        m: &mut crate::LoadedModel,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        start_pos: usize,
    ) -> Option<Result<Vec<f32>, String>> {
        let Some(b) = m.llama_mut() else {
            return Some(Err("decide: llama bundle missing".into()));
        };
        let r = hipfire_runtime::llama::forward_prefill_batch(
            gpu, &b.weights, &b.config, tokens, start_pos, &mut b.kv, &b.scratch, None,
        )
        .map_err(|e| format!("decide prefill: {e:?}"))
        .and_then(|()| gpu.download_f32(&b.scratch.logits).map_err(|e| format!("decide logits: {e:?}")));
        if r.is_ok() {
            m.seq_pos = start_pos + tokens.len();
        }
        Some(r)
    }
    fn decide_save(
        &self,
        m: &mut crate::LoadedModel,
        _gpu: &mut rdna_compute::Gpu,
    ) -> Option<Result<crate::decide::DecideSnapshot, String>> {
        // Pure-attention: KV is positional, the cursor is the whole state.
        Some(Ok(crate::decide::DecideSnapshot { seq_pos: m.seq_pos, recurrent: None }))
    }
    fn decide_restore(
        &self,
        m: &mut crate::LoadedModel,
        _gpu: &mut rdna_compute::Gpu,
        snap: &crate::decide::DecideSnapshot,
    ) -> Option<Result<(), String>> {
        m.seq_pos = snap.seq_pos;
        Some(Ok(()))
    }
```

If a field name differs (`b.kv` vs `b.kv_cache`, `b.scratch`), use the
field from `pub struct LlamaBundle` in
`crates/hipfire-arch-llama/src/carrier.rs:27`. Its fields were `config`,
`weights`, `scratch`, `kv` at plan time.

- [ ] **Step 7: Run the tests and a workspace check**

Run: `cargo test -p hipfire-loader --lib decide && cargo check --workspace --examples`
Expected: `1 passed`; the check finishes with no errors.

- [ ] **Step 8: Commit**

```bash
git add crates/hipfire-loader/src/decide.rs crates/hipfire-loader/src/lib.rs crates/hipfire-loader/src/carriers.rs
git commit -m "feat(decide): Carrier prefill/snapshot hooks for Qwen3.5 and Llama"
```

---

### Task 5: `run_decide` orchestration (`hipfire-generate`)

**Files:**
- Create: `crates/hipfire-generate/src/decide.rs`
- Modify: `crates/hipfire-generate/src/lib.rs` (add `pub mod decide;` next
  to `pub mod common;`)

**Interfaces:**
- Consumes:
  - `hipfire_engine::decide::{parse_request, labels_for, build_user_text, shared_prefix_len, assemble_answer, usage_json, SYSTEM_PROMPT}` (Tasks 1–3);
  - `hipfire_loader::{carrier_for, decide::DecideSnapshot}` and the Carrier
    hooks (Task 4);
  - `crate::common::production_fail_closed_rollback`.
- Produces:
  - `pub struct DecideError { pub status: u16, pub message: String, pub required_max_seq: Option<usize> }`
  - `pub struct DecideOutcome { pub answers: serde_json::Value, pub usage: serde_json::Value, pub timing: serde_json::Value }`
  - `pub fn run_decide(m: &mut hipfire_loader::LoadedModel, gpu: &mut rdna_compute::Gpu, req: &serde_json::Value) -> Result<DecideOutcome, DecideError>`
  - `impl DecideError { pub fn to_reply(&self, id: &str) -> serde_json::Value }`
    builds the `decided` error reply.

The GPU path is verified by Task 7's gates. This task's unit test covers the
error-reply shape and the precondition checks that don't need a GPU.

- [ ] **Step 1: Write the failing test** (bottom of the new file)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_reply_shape() {
        let e = DecideError { status: 422, message: "too long".into(), required_max_seq: Some(9000) };
        let r = e.to_reply("d1");
        assert_eq!(r["type"], "decided");
        assert_eq!(r["id"], "d1");
        assert_eq!(r["error"]["status"], 422);
        assert_eq!(r["error"]["message"], "too long");
        assert_eq!(r["error"]["required_max_seq"], 9000);
        let e = DecideError { status: 409, message: "x".into(), required_max_seq: None };
        assert!(e.to_reply("d2")["error"].get("required_max_seq").is_none());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p hipfire-generate --lib decide`
Expected: compile error, `DecideError` not found.

- [ ] **Step 3: Implement** (top of `crates/hipfire-generate/src/decide.rs`)

```rust
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Decide runner: render every question's full prompt, prefill the longest
//! common token prefix once, snapshot, then per question restore → prefill
//! suffix → gather label logits. Leaves the model reset.
//! Spec: `docs/specs/2026-09-28-jev-decide-design.md` §4.1.

use hipfire_engine::decide as d;
use hipfire_loader::LoadedModel;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct DecideError {
    pub status: u16,
    pub message: String,
    pub required_max_seq: Option<usize>,
}

impl DecideError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self { status, message: message.into(), required_max_seq: None }
    }
    pub fn to_reply(&self, id: &str) -> Value {
        let mut err = json!({"status": self.status, "message": self.message});
        if let Some(n) = self.required_max_seq {
            err["required_max_seq"] = json!(n);
        }
        json!({"type": "decided", "id": id, "error": err})
    }
}

pub struct DecideOutcome {
    pub answers: Value,
    pub usage: Value,
    pub timing: Value,
}

fn rollback(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), DecideError> {
    let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
    if ep.rolled_back {
        Ok(())
    } else {
        Err(DecideError::new(
            500,
            format!("decide rollback failed: {}", ep.context.unwrap_or_default()),
        ))
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

pub fn run_decide(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    req: &Value,
) -> Result<DecideOutcome, DecideError> {
    let t_total = Instant::now();
    let parsed = d::parse_request(req).map_err(|e| DecideError::new(422, e))?;
    let no_snapshot = req.get("_debug_no_snapshot").and_then(Value::as_bool) == Some(true);

    if m.pp > 1 || m.ep.is_some() {
        return Err(DecideError::new(409, "decide requires a single-GPU model (pp=1, no EP)"));
    }
    if m.eviction.is_some() || m.kv_adaptive.is_some() {
        return Err(DecideError::new(409, "decide requires KV eviction (CASK) and kv_adaptive off"));
    }
    let carrier = hipfire_loader::carrier_for(m.arch_id)
        .filter(|c| c.decide_supported())
        .ok_or_else(|| DecideError::new(400, format!("decide not supported for arch_id {}", m.arch_id)))?;
    let tok = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| DecideError::new(400, "model has no tokenizer"))?;
    let single = |s: &str| tok.encode(s).len() == 1;

    // Labels, their token ids, and each question's full prompt tokens.
    let mut label_ids: Vec<Vec<u32>> = Vec::new();
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    for q in &parsed.questions {
        let labels = d::labels_for(q, single).map_err(|e| DecideError::new(422, e))?;
        label_ids.push(labels.iter().map(|l| tok.encode(l)[0]).collect());
        let user = d::build_user_text(&parsed.state_text, q, &labels);
        let (tokens, _) = hipfire_engine::prompt::batch_render_prompt_tokens(
            &user,
            Some(d::SYSTEM_PROMPT),
            hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            tok,
            m.chat_template.as_ref(),
            0,
            None,
            false,
            None,
        )
        .map_err(|e| DecideError::new(500, format!("prompt render: {e}")))?;
        seqs.push(tokens);
    }
    let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
    if longest + 1 > m.max_seq {
        return Err(DecideError {
            status: 422,
            message: format!("prompt needs {} tokens, model max_seq is {}", longest + 1, m.max_seq),
            required_max_seq: Some(longest + 1),
        });
    }
    let prefix_len = if no_snapshot { 0 } else { d::shared_prefix_len(&seqs) };

    rollback(m, gpu)?;
    let mut timing = json!({"prefix_tokens": prefix_len, "state_prefill_ms": 0.0, "question_ms": []});
    let mut per_q_logits: Vec<Vec<f32>> = Vec::with_capacity(seqs.len());

    let result: Result<(), DecideError> = (|| {
        let hook_err = |r: Option<Result<(), String>>| -> Result<(), DecideError> {
            r.ok_or_else(|| DecideError::new(400, "decide hook unsupported"))?
                .map_err(|e| DecideError::new(500, e))
        };
        let mut snapshot: Option<hipfire_loader::decide::DecideSnapshot> = None;
        if prefix_len > 0 {
            let t = Instant::now();
            carrier
                .decide_prefill_logits(m, gpu, &seqs[0][..prefix_len], 0)
                .ok_or_else(|| DecideError::new(400, "decide hook unsupported"))?
                .map_err(|e| DecideError::new(500, e))?;
            snapshot = Some(
                carrier
                    .decide_save(m, gpu)
                    .ok_or_else(|| DecideError::new(400, "decide hook unsupported"))?
                    .map_err(|e| DecideError::new(500, e))?,
            );
            timing["state_prefill_ms"] = json!(ms(t));
        }
        let loop_result: Result<(), DecideError> = (|| {
            for (i, seq) in seqs.iter().enumerate() {
                let t = Instant::now();
                match &snapshot {
                    Some(s) if i > 0 => hook_err(carrier.decide_restore(m, gpu, s))?,
                    Some(_) => {} // first question continues straight from the snapshot point
                    None => {
                        if i > 0 {
                            rollback(m, gpu)?;
                        }
                    }
                }
                let logits = carrier
                    .decide_prefill_logits(m, gpu, &seq[prefix_len..], prefix_len)
                    .ok_or_else(|| DecideError::new(400, "decide hook unsupported"))?
                    .map_err(|e| DecideError::new(500, e))?;
                per_q_logits.push(label_ids[i].iter().map(|&id| logits[id as usize]).collect());
                timing["question_ms"].as_array_mut().unwrap().push(json!(ms(t)));
            }
            Ok(())
        })();
        if let Some(s) = snapshot.take() {
            s.free(gpu);
        }
        loop_result
    })();
    // Always leave the model reset, even on failure.
    let cleanup = rollback(m, gpu);
    result?;
    cleanup?;

    let mut answers = serde_json::Map::new();
    for (q, ll) in parsed.questions.iter().zip(per_q_logits.iter()) {
        answers.insert(q.name.clone(), d::assemble_answer(q, ll));
    }
    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    timing["total_ms"] = json!(ms(t_total));
    Ok(DecideOutcome {
        answers: Value::Object(answers),
        usage: d::usage_json(prefix_len, &lens),
        timing,
    })
}
```

Notes for the implementer:

- `AssistantPrefix::Plain` is only the fallback when Jinja rendering is
  unavailable. With a chat template present and `enable_thinking = false`,
  the Qwen template itself emits the closed `<think>\n\n</think>\n\n`
  block. Gate 1 in Task 7 checks the rendered tail.
- The first question needs no restore: the model is already at the snapshot
  point.
- If `run_decide` fails to borrow-check because `tok` (borrowed from `m`)
  is alive across the `&mut m` hook calls, the fix is to finish every use of
  `tok` in the label/prompt loop, then drop it. The code above already does
  all tokenizer work before the first `&mut m` call. If the compiler still
  complains, clone the needed data out and scope `tok` in a block that ends
  before `rollback(m, gpu)?`.

- [ ] **Step 4: Run the test and a workspace check**

Run: `cargo test -p hipfire-generate --lib decide && cargo check --workspace --examples`
Expected: `1 passed`; the check finishes with no errors.

- [ ] **Step 5: Commit**

```bash
git add crates/hipfire-generate/src/decide.rs crates/hipfire-generate/src/lib.rs
git commit -m "feat(decide): run_decide orchestration with snapshot reuse and rollback"
```

---

### Task 6: Daemon `decide` command

**Files:**
- Modify: `crates/hipfire-daemon/src/main.rs`. Add a `"decide" => { … }`
  arm to the `match msg_type` dispatch, placed directly before the
  `"reset" => {` arm (≈line 3610).

**Interfaces:**
- Consumes: `hipfire_generate::decide::{run_decide, DecideError}` (Task 5).
- Produces the wire protocol:
  - request: `{"type":"decide","id":"<id>","state":…,"questions":{…},"_debug_no_snapshot"?:bool}`;
  - reply (always exactly one line):
    `{"type":"decided","id","answers","usage","timing"}` or
    `{"type":"decided","id","error":{"status","message","required_max_seq"?}}`.

- [ ] **Step 1: Implement the arm**

```rust
            "decide" => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let refuse = |status: u16, message: &str| {
                    hipfire_generate::decide::DecideError {
                        status,
                        message: message.to_string(),
                        required_max_seq: None,
                    }
                    .to_reply(&id)
                };
                let reply = if slot_backend.as_ref().is_some_and(|s| s.active_count() > 0)
                    || batch_scheduler.as_ref().is_some_and(|s| s.active_count() > 0)
                {
                    refuse(409, "decide refused: slot or continuous-batch requests active")
                } else if let Some(m) = &mut model {
                    let r = hipfire_generate::decide::run_decide(m, &mut gpu, &msg);
                    // run_decide always resets the model; advance the epoch like `reset`.
                    state_epoch = state_epoch.saturating_add(1);
                    match r {
                        Ok(out) => serde_json::json!({
                            "type": "decided",
                            "id": id,
                            "answers": out.answers,
                            "usage": out.usage,
                            "timing": out.timing,
                        }),
                        Err(e) => e.to_reply(&id),
                    }
                } else {
                    refuse(503, "no model loaded")
                };
                let _ = writeln!(stdout, "{reply}");
                let _ = stdout.flush();
            }
```

Check the surrounding names in the `"reset"` arm before pasting: `model`,
`gpu`, `slot_backend`, `batch_scheduler`, `state_epoch`, `stdout`. They were
correct at plan time (`main.rs:3610-3760`). If slot mode keeps its model
outside `model` (i.e. `model` is `None` whenever `slot_backend` is `Some`),
the 409 branch above answers first, which is the intended behaviour.

- [ ] **Step 2: Build**

Run: `cargo build --release -p hipfire-daemon`
Expected: builds `target/release/daemon` with no errors.

- [ ] **Step 3: Smoke test with no model loaded** (no GPU work happens:
  the daemon replies 503 before touching a model)

```bash
printf '%s\n' '{"type":"decide","id":"s1","state":"x","questions":{"q":{"type":"noul","instructions":"y?"}}}' \
  | timeout 60 target/release/daemon 2>/dev/null | grep '"decided"'
```

Expected output:
`{"error":{"message":"no model loaded","status":503},"id":"s1","type":"decided"}`.
Key order may differ.

If the daemon exits with
`FATAL: hipfire daemon already running (PID N)`, another daemon holds
`~/.hipfire/daemon.pid`. Stop it (`pkill -x daemon`) only after confirming
with the user that it's not their serve.

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-daemon/src/main.rs
git commit -m "feat(decide): daemon decide command"
```

---

### Task 7: GPU correctness gates

**Files:**
- Create: `scripts/jev_eval/daemon_client.py`
- Create: `scripts/jev_eval/gates.py`

**Interfaces:**
- Consumes: the daemon wire protocol (Task 6), plus `load` / `reset` /
  `generate` as the daemon already handles them.
- Produces: `python3 scripts/jev_eval/gates.py --model <path>` exits 0 only
  if gates 1–4 all pass, printing one `PASS`/`FAIL` line per gate with its
  numbers.

- [ ] **Step 1: Write the daemon driver**
  (`scripts/jev_eval/daemon_client.py`)

```python
"""Minimal JSONL driver for target/release/daemon (decide gates + evals)."""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Daemon:
    def __init__(self, binary=None):
        self.p = subprocess.Popen(
            [str(binary or REPO / "target/release/daemon")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.attempt = 0

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv_until(self, types):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("daemon exited")
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("type") in types:
                return v

    def load(self, model, max_seq=8192):
        self.send({"type": "load", "model": str(model), "params": {"max_seq": max_seq}})
        v = self.recv_until({"loaded", "error"})
        if v["type"] != "loaded":
            raise RuntimeError(f"load failed: {v}")
        return v

    def reset(self):
        self.attempt += 1
        self.send({"type": "reset", "attempt_id": self.attempt})
        v = self.recv_until({"reset", "error"})
        if v.get("type") != "reset" or not v.get("rolled_back"):
            raise RuntimeError(f"reset failed: {v}")

    def decide(self, state, questions, **extra):
        self.send({"type": "decide", "id": "g", "state": state, "questions": questions, **extra})
        return self.recv_until({"decided"})

    def generate_greedy(self, prompt, max_tokens=32):
        self.attempt += 1
        self.send({"type": "generate", "id": f"gen{self.attempt}", "attempt_id": self.attempt,
                   "prompt": prompt, "temperature": 0.0, "max_tokens": max_tokens,
                   "thinking_enabled": False})
        text = []
        while True:
            v = self.recv_until({"token", "done", "error"})
            if v["type"] == "token":
                text.append(v.get("text", ""))
            elif v["type"] == "done":
                return "".join(text)
            else:
                raise RuntimeError(f"generate failed: {v}")

    def close(self):
        try:
            self.send({"type": "unload"})
            self.recv_until({"unloaded"})
        finally:
            self.p.terminate()
```

Before relying on `generate_greedy`, check the generate field names the
daemon accepts on this branch. Temperature 0 should mean greedy; see
`crates/hipfire-daemon/src/main.rs` near the `"generate" =>` arm and the
serve request builder at `crates/hipfire-cli/src/serve/complete.rs:1693-1793`.
Adjust the keys if they differ, and keep `attempt_id` since the daemon
requires it.

- [ ] **Step 2: Write the gates** (`scripts/jev_eval/gates.py`)

```python
"""Decide GPU correctness gates (spec §10). Exit 0 iff all pass."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daemon_client import Daemon  # noqa: E402

STATE = {"channel": "email", "subject": "Where is my order?",
         "body": "Order #3527 was placed 21 days ago and tracking has not updated."}
QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which support queue should handle this ticket?",
              "criteria": {"billing": "Payments, refunds", "shipping": "Delivery status, lost parcels",
                           "technical": "App bugs, login", "general": "Anything else"}},
    "priority": {"type": "score", "instructions": "How urgent is this ticket?",
                 "criteria": ["Low", "Normal", "High", "Critical"]},
    "angry": {"type": "noul", "instructions": "The customer sounds angry."},
}


def dist(a):
    if a["type"] == "noul":
        return [a["noul"], 1 - a["noul"]]
    return list(a["probabilities"].values())


def gate_exact(d):
    snap = d.decide(STATE, QUESTIONS)
    full = d.decide(STATE, QUESTIONS, _debug_no_snapshot=True)
    assert "answers" in snap and "answers" in full, (snap, full)
    worst = 0.0
    for name in QUESTIONS:
        for p, q in zip(dist(snap["answers"][name]), dist(full["answers"][name])):
            worst = max(worst, abs(math.log(max(p, 1e-12)) - math.log(max(q, 1e-12))))
    ok = worst < 0.05 and snap["timing"]["prefix_tokens"] > 0
    return ok, f"max |Δlogp| snapshot vs full-prefill = {worst:.4f}, prefix_tokens={snap['timing']['prefix_tokens']}"


def gate_isolation(d):
    q = {"type": "noul", "instructions": "Does the text contain the secret code word PELICAN?"}
    base = d.decide("A short note about the weather.", {
        "probe": q,
        "leak": {"type": "noul", "instructions": "The secret code word is PELICAN. Is it sunny?"},
    })["answers"]["probe"]["noul"]
    in_state = d.decide("A short note about the weather. The secret code word is PELICAN.",
                        {"probe": q})["answers"]["probe"]["noul"]
    ok = base < 0.2 and in_state > 0.8
    return ok, f"P(code) with code in sibling question = {base:.3f} (<0.2), in state = {in_state:.3f} (>0.8)"


def gate_clean(d):
    prompt = "Where is my order? Order #3527 was placed 21 days ago. Reply in one sentence."
    d.reset()
    after_reset = d.generate_greedy(prompt)
    d.reset()
    d.decide(STATE, QUESTIONS)
    after_decide = d.generate_greedy(prompt)
    d.reset()
    unrelated = "Name three primary colours."
    base2 = d.generate_greedy(unrelated)
    d.reset()
    d.decide(STATE, QUESTIONS)
    after2 = d.generate_greedy(unrelated)
    ok = after_reset == after_decide and base2 == after2
    return ok, f"shared-prefix prompt identical={after_reset == after_decide}, unrelated identical={base2 == after2}"


def gtt_used():
    for p in Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"):
        return int(p.read_text())
    for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"):
        return int(p.read_text())
    return None


def gate_leak(d, n):
    d.decide(STATE, QUESTIONS)
    before = gtt_used()
    for _ in range(n):
        d.decide(STATE, QUESTIONS)
    after = gtt_used()
    if before is None:
        return False, "no mem_info_gtt_used / mem_info_vram_used sysfs node — cannot measure"
    grew = after - before
    ok = grew < 64 * 1024 * 1024
    return ok, f"device memory growth over {n} decides = {grew / 2**20:.1f} MiB (<64)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--leak-n", type=int, default=1000)
    a = ap.parse_args()
    d = Daemon()
    try:
        info = d.load(a.model)
        print(f"loaded {a.model}: arch={info.get('arch')}")
        results = [("1 snapshot exactness", gate_exact(d)),
                   ("2 question isolation", gate_isolation(d)),
                   ("3 model left clean", gate_clean(d)),
                   ("4 no leak", gate_leak(d, a.leak_n))]
    finally:
        d.close()
    failed = 0
    for name, (ok, detail) in results:
        print(f"{'PASS' if ok else 'FAIL'}  gate {name}: {detail}")
        failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the gates on Qwen3.5-4B** (hybrid, DeltaNet restore path)

```bash
grep MemAvailable /proc/meminfo
cargo build --release -p hipfire-daemon
python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4
```

Expected: four `PASS` lines, exit 0. Record the numbers for the PR.

If gate 1 fails with a small but real Δ, don't loosen the threshold to make
it pass. The expected causes are:

- **Restore incomplete:** `restore_to` missed conv state or the EF residual;
- **Chunk-boundary numerics:** prefilling prefix+suffix in two calls vs one
  changes chunk boundaries.

To separate them, rerun with a 1-question request. With one question there's
no restore, only the split, so a Δ there is purely chunk-boundary numerics.
Report the finding before changing anything.

- [ ] **Step 4: Run the gates on a Llama-carrier model**

```bash
python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3-0.6b.hf4
```

Confirm the printed `arch=` is served by `LlamaCarrier`. `carrier_for` maps
arch ids; check with
`grep -n claims_arch_id -A6 crates/hipfire-loader/src/carriers.rs`. If
`qwen3-0.6b` isn't a Llama-carrier model, pick a local model that is and
note which in the PR. Expected: four `PASS` lines.

- [ ] **Step 5: Commit**

```bash
git add scripts/jev_eval/daemon_client.py scripts/jev_eval/gates.py
git commit -m "test(decide): GPU correctness gates (exactness, isolation, clean state, leak)"
```

---

### Task 8: Serve route `POST /v1/systemone`

**Files:**
- Create: `crates/hipfire-cli/src/serve/decide.rs`
- Modify: `crates/hipfire-cli/src/serve/mod.rs` (add
  `pub(crate) mod decide;` beside `pub mod http;`)
- Modify: `crates/hipfire-cli/src/serve/http.rs` (route arm before
  `_ => openai_error("not found", 404)`)
- Modify: `crates/hipfire-cli/src/serve/fake_daemon.py` (`decide` branch)
- Modify: `crates/hipfire-cli/src/main.rs` (tests in the Task 11 harness
  module)

**Interfaces:**
- Consumes:
  - `ServeShared { runtime: Mutex<ServeRuntime>, meta, admission, max_request_bytes }`;
  - `ServeRuntime { engine, paths, registry, current_path, .. }` and
    `ensure_model`;
  - `crate::{find_model_path, registry_entry_for_path}`;
  - `hipfire_client::Engine::request`;
  - http.rs helpers `read_json_body`, `openai_error`, `json_response`,
    `request_id`, `BoxBody`.
- Produces:
  - `pub(crate) async fn handle_decide(shared: Arc<ServeShared>, body: serde_json::Value) -> Response<BoxBody>`.

- [ ] **Step 1: Add the fake-daemon branch** (`fake_daemon.py`, a new
  `elif` before `elif ty == "unload":`)

```python
    elif ty == "decide":
        qs = req.get("questions") or {}
        if not isinstance(req.get("state"), (str, dict, list)) or not qs:
            out({"type": "decided", "id": req.get("id"),
                 "error": {"status": 422, "message": "state/questions invalid"}})
            continue
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
        out({"type": "decided", "id": req.get("id"), "answers": answers,
             "usage": {"input_tokens": 10 * len(qs), "output_tokens": 0},
             "timing": {"prefix_tokens": 8, "total_ms": 1.0}})
```

- [ ] **Step 2: Write the failing serve tests** (in `main.rs`, next to
  `oversized_declared_body_is_rejected_before_body_read`)

```rust
    #[cfg(unix)]
    fn post_systemone(port: u16, body: &serde_json::Value) -> (u16, String, String) {
        let payload = serde_json::to_vec(body).unwrap();
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let head = format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(&payload).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let status: u16 = response[9..12].parse().unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").unwrap();
        (status, headers.to_string(), body.to_string())
    }

    #[cfg(unix)]
    #[test]
    fn systemone_named_model_returns_jev_shape() {
        let harness = Task11HttpHarness::spawn("systemone-named");
        let (status, headers, body) = post_systemone(harness.port(), &serde_json::json!({
            "model": harness.model(),
            "state": "hello",
            "questions": {"q": {"type": "choice", "instructions": "i",
                                "criteria": {"a": "x", "b": "y"}}}
        }));
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["answers"]["q"]["type"], "choice");
        assert_eq!(v["usage"]["output_tokens"], 0);
        assert!(v.get("timing").is_none(), "timing must not leak into the Jev body");
        assert!(headers.to_ascii_lowercase().contains("x-hipfire-timing:"), "{headers}");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_jev_latest_without_loaded_model_is_503() {
        let harness = Task11HttpHarness::spawn("systemone-unloaded");
        let (status, _, body) = post_systemone(harness.port(), &serde_json::json!({
            "model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        }));
        assert_eq!(status, 503, "{body}");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_jev_latest_uses_loaded_model() {
        let harness = Task11HttpHarness::spawn("systemone-loaded");
        let named = serde_json::json!({"model": harness.model(), "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        assert_eq!(post_systemone(harness.port(), &named).0, 200);
        let (status, _, body) = post_systemone(harness.port(), &serde_json::json!({
            "model": "jev-latest", "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i"}}
        }));
        assert_eq!(status, 200, "{body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_ne!(v["model"], "jev-latest");
    }

    #[cfg(unix)]
    #[test]
    fn systemone_daemon_validation_error_maps_to_422() {
        let harness = Task11HttpHarness::spawn("systemone-422");
        let (status, _, body) = post_systemone(harness.port(), &serde_json::json!({
            "model": harness.model(), "state": 5, "questions": {}
        }));
        assert_eq!(status, 422, "{body}");
    }
```

Before relying on `systemone_jev_latest_without_loaded_model_is_503`, check
that `Task11HttpHarness::spawn` does not pre-warm a model. Read `spawn_inner`
in `main.rs` and look for a startup load, e.g. a `HIPFIRE_MODEL` or
pre-warm path. If it does pre-warm, that test must instead unload first. It
can send the harness's existing unload path if there is one. Otherwise,
assert the pre-warmed model is echoed and drop the 503 case into a
`handle_decide` unit test that builds a `ServeRuntime` with
`current_path: None`.

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test -p hipfire-cli systemone`
Expected: 4 failures with status 404 (no route yet).

- [ ] **Step 4: Implement the handler** (`serve/decide.rs`)

```rust
//! `POST /v1/systemone` — Jev-compatible decide. The daemon is the single
//! validation authority; this layer selects the model, serialises against
//! chat traffic, and shapes Jev's response. Spec §6.

use super::http::{json_response, openai_error, request_id, BoxBody};
use super::ServeShared;
use hyper::{header, Response};
use std::sync::Arc;

fn is_local_model(runtime: &super::ServeRuntime, model: &str) -> bool {
    crate::registry_entry_for_path(&runtime.paths, &runtime.registry, model).is_some()
        || crate::find_model_path(&runtime.paths, &runtime.registry, model).is_some()
}

pub(crate) async fn handle_decide(
    shared: Arc<ServeShared>,
    mut body: serde_json::Value,
) -> Response<BoxBody> {
    let guard = match shared.admission.acquire() {
        Ok(g) => g,
        Err(e) => {
            let mut r = openai_error(&e.to_string(), 503);
            r.headers_mut().insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
            return r;
        }
    };
    let _guard = guard;
    let requested = body.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

    for attempt in 0..2 {
        // Resolve the model under the runtime lock, then release it.
        let (engine, model_echo) = {
            let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let target = if !requested.is_empty() && is_local_model(&runtime, &requested) {
                requested.clone()
            } else {
                match runtime.current_path.as_ref() {
                    Some(p) => p.display().to_string(),
                    None => {
                        return openai_error(
                            "no model loaded: name a local hipfire model in `model`, or load one first",
                            503,
                        )
                    }
                }
            };
            if let Err(e) = runtime.ensure_model(&target, &shared.meta, None) {
                return openai_error(&format!("{e:#}"), 500);
            }
            let echo = runtime
                .current_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or(target);
            (runtime.engine.clone(), echo)
        };

        let id = request_id();
        let mut msg = serde_json::json!({"type": "decide", "id": id});
        for key in ["state", "questions", "_debug_no_snapshot"] {
            if let Some(v) = body.get(key) {
                msg[key] = v.clone();
            }
        }
        let reply = match tokio::task::spawn_blocking(move || engine.request(&msg)).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return openai_error(&format!("daemon: {e}"), 500),
            Err(e) => return openai_error(&format!("decide worker failed: {e}"), 500),
        };
        if reply.get("type").and_then(|v| v.as_str()) != Some("decided") {
            return openai_error(&format!("unexpected daemon reply: {reply}"), 500);
        }
        if let Some(err) = reply.get("error") {
            let status = err.get("status").and_then(|v| v.as_u64()).unwrap_or(500) as u16;
            let message = err.get("message").and_then(|v| v.as_str()).unwrap_or("decide failed");
            if let (0, Some(n)) = (attempt, err.get("required_max_seq").and_then(|v| v.as_u64())) {
                let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let target = runtime.current_path.as_ref().map(|p| p.display().to_string());
                if let Some(target) = target {
                    if runtime.ensure_model(&target, &shared.meta, Some(n)).is_ok() {
                        drop(runtime);
                        body["model"] = serde_json::json!(target);
                        continue;
                    }
                }
            }
            return openai_error(message, status);
        }
        {
            let mut meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            meta.requests_served += 1;
        }
        let out = serde_json::json!({
            "model": model_echo,
            "answers": reply["answers"],
            "usage": reply["usage"],
        });
        let mut resp = json_response(out, 200);
        if let Ok(v) = header::HeaderValue::from_str(&reply["timing"].to_string()) {
            resp.headers_mut().insert("x-hipfire-timing", v);
        }
        return resp;
    }
    openai_error("decide: context growth retry exhausted", 500)
}
```

Make the http.rs helpers visible to the sibling module by changing their
visibility to `pub(crate)` where needed. `json_response`, `openai_error` and
`request_id` are already `pub(crate)`; `BoxBody` is `pub(crate) type`.
`ServeRuntime` and its fields are `pub(crate)` in `mod.rs`.

- [ ] **Step 5: Add the route arm** (`http.rs`, before
  `_ => openai_error("not found", 404),`)

```rust
        (Method::POST, "/v1/systemone") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };
            crate::serve::decide::handle_decide(shared, body_val).await
        }
```

- [ ] **Step 6: Run the tests**

Run: `cargo test -p hipfire-cli systemone`
Expected: `4 passed`.

Then run the full serve suite to catch regressions:
`cargo test -p hipfire-cli`
Expected: all pass. The count grows by 4 over `origin/master`.

- [ ] **Step 7: Commit**

```bash
git add crates/hipfire-cli/src/serve/decide.rs crates/hipfire-cli/src/serve/mod.rs \
        crates/hipfire-cli/src/serve/http.rs crates/hipfire-cli/src/serve/fake_daemon.py \
        crates/hipfire-cli/src/main.rs
git commit -m "feat(decide): serve POST /v1/systemone (Jev wire format)"
```

---

### Task 9: Evaluation harness and report

**Files:**
- Create: `scripts/jev_eval/run_jevbench.py`
- Create: `scripts/jev_eval/run_calibration.py`
- Create: `scripts/jev_eval/report.py`
- Create: `scripts/jev_eval/README.md`

**Interfaces:**
- Consumes:
  - a running `hipfire serve` with the repo daemon;
  - the external clones `~/repos/jev-evals/jev-bench` and
    `~/repos/jev-evals/jev-ood-calibration`, overridable via
    `JEVBENCH_DIR` / `JEVCAL_DIR`.
- Produces, under `--out` (default `bench/jev/<model>/`):
  - `jevbench/predictions/*.jsonl`
  - `jevbench/results/*.json`
  - `calibration/hipfire_<set>.jsonl`
  - `report.md`

- [ ] **Step 1: jev-bench wrapper** (`run_jevbench.py`). It redirects all
  writes away from the committed Jev files.

```python
"""Run Running-Dolphins/jev-bench against local hipfire serve.

jevbench.py writes predictions/results under its module ROOT, which would
overwrite the committed Jev baselines — so ROOT is redirected to --out and the
dataset cache is symlinked from the clone.
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True, help="hipfire model tag, echoed in results")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500, help="must be 500 to align with Jev's committed rows")
ap.add_argument("--workers", type=int, default=1, help="serve serialises decides; >1 only queues")
ap.add_argument("what", nargs="+", help="task names, 'all', or x-<experiment>")
a = ap.parse_args()

bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
if not (out / "data").exists():
    (out / "data").symlink_to(bench / "data")
os.environ.setdefault("TYPESAFE_API_KEY", "local-hipfire")
os.environ["JEV_WORKERS"] = str(a.workers)
sys.path.insert(0, str(bench))
import jevbench as jb  # noqa: E402

jb.ROOT = out
jb.API_URL = f"http://127.0.0.1:{a.port}/v1/systemone"
jb.MODEL = a.model
jb.WORKERS = a.workers

for w in a.what:
    if w.startswith("x-"):
        name = w[2:]
        res, rows = jb.EXPERIMENTS[name][0](a.n, False)
        jb.save(f"x-{name}", rows, res, False)
    else:
        for name in (list(jb.TASKS) if w == "all" else [w]):
            jb.run_task(name, a.n, False)
print(f"done → {out}")
```

Before first use, confirm `jb.save`'s signature matches the call in
`jevbench.py`'s `main()` (`save(f"x-{nm}", rows, out, False)`). Also check
that the port matches the port `hipfire serve` binds (`hipfire serve --help`).

- [ ] **Step 2: Calibration adapter** (`run_calibration.py`)

```python
"""Run the scienthoon/jev-ood-calibration sets against local hipfire serve,
writing rows in that repo's result schema (type, option_keys, probs, target,
pred, gold, confidence, source, usage)."""
import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=11435)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("sets", nargs="+", choices=["synth", "openbookqa", "commonsense_qa", "hellaswag"])
a = ap.parse_args()
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
URL = f"http://127.0.0.1:{a.port}/v1/systemone"
NOUL_TRUE, NOUL_FALSE = "Yes, the statement is true.", "No, the statement is false."


def hf_rows(ds, cfg, n):
    base = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}"
            f"&config={cfg}&split=validation")
    rows = []
    for off in range(0, n, 100):
        with urllib.request.urlopen(base + f"&offset={off}&length={min(100, n - off)}", timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    return rows


def public_records(name):
    # Mirrors convert.py: first ≤2000 validation rows, filter, Random(1).shuffle
    # (verified row-for-row against Jev's committed results on 2026-09-28).
    if name == "openbookqa":
        src = hf_rows("allenai/openbookqa", "main", 500)
        recs = [{"state": s["question_stem"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": s["answerKey"]}
                for s in src]
    elif name == "commonsense_qa":
        src = hf_rows("tau/commonsense_qa", "default", 1221)
        recs = [{"state": s["question"], "type": "choice", "question": "Which option is the correct answer?",
                 "options": dict(zip(s["choices"]["label"], s["choices"]["text"])), "label": s["answerKey"]}
                for s in src]
    else:
        src = hf_rows("Rowan/hellaswag", "default", 2000)
        recs = [{"state": s["ctx"], "type": "choice", "question": "Which ending most plausibly continues the text?",
                 "options": {str(i): e for i, e in enumerate(s["endings"])}, "label": str(s["label"]).strip()}
                for s in src]
    recs = [r for r in recs if r["label"] in r["options"]]
    random.Random(1).shuffle(recs)
    for r in recs:
        r["source"] = name
    return recs


def to_question(r):
    if r["type"] == "choice":
        keys = list(r["options"])
        return ({"type": "choice", "instructions": r["question"], "criteria": r["options"]},
                keys, [1.0 if k == r["label"] else 0.0 for k in keys])
    if r["type"] == "score":
        keys = [str(i) for i in range(len(r["levels"]))]
        return ({"type": "score", "instructions": r["question"], "criteria": [str(x) for x in r["levels"]]},
                keys, [1.0 if i == int(r["label"]) else 0.0 for i in range(len(keys))])
    return ({"type": "noul", "instructions": r["question"], "criteria": {"true": NOUL_TRUE, "false": NOUL_FALSE}},
            ["yes", "no"], [1.0, 0.0] if r["label"] is True else [0.0, 1.0])


def ask(r):
    q, keys, target = to_question(r)
    body = json.dumps({"model": a.model, "state": r["state"], "questions": {"q": q}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        v = json.load(resp)
    ans = v["answers"]["q"]
    probs = [ans["noul"], 1 - ans["noul"]] if r["type"] == "noul" else [ans["probabilities"][k] for k in keys]
    pred = keys[max(range(len(probs)), key=probs.__getitem__)]
    gold = keys[max(range(len(target)), key=target.__getitem__)]
    return {"type": r["type"], "option_keys": keys, "probs": probs, "target": target, "pred": pred,
            "gold": gold, "confidence": ans.get("confidence"), "source": r.get("source"),
            "usage": {"inputTokens": v["usage"]["input_tokens"], "outputTokens": 0}}


for s in a.sets:
    recs = ([json.loads(l) for l in open(cal / "data/val.jsonl")] if s == "synth" else public_records(s))
    path = out / f"hipfire_{s}.jsonl"
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps(ask(r)) + "\n")
            if i % 100 == 0:
                print(f"{s}: {i}/{len(recs)}", file=sys.stderr)
    print(f"wrote {path}")
```

The synthetic `score` labels are ints, and `noul` labels are Python bools.
`to_question` compares `r["label"] is True` precisely because `0 == False`
in Python. Don't replace it with a dict lookup keyed by bool.

- [ ] **Step 3: Report** (`report.py`). It uses the same metric for Jev and
  hipfire.

```python
"""Accuracy + top-probability ECE (10 bins) for hipfire vs Jev's committed answers."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="same --out used by run_jevbench/run_calibration")
a = ap.parse_args()
out = Path(a.out)
bench = Path(os.environ.get("JEVBENCH_DIR", Path.home() / "repos/jev-evals/jev-bench"))
cal = Path(os.environ.get("JEVCAL_DIR", Path.home() / "repos/jev-evals/jev-ood-calibration"))


def ece(pairs, bins=10):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        ins = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if ins:
            e += len(ins) / len(pairs) * abs(sum(p for p, _ in ins) / len(ins) - sum(c for _, c in ins) / len(ins))
    return e


def bench_stats(path):
    rows = [json.loads(l) for l in open(path)]
    return len(rows), sum(r["correct"] for r in rows) / len(rows), ece([(r["p_top"], r["correct"]) for r in rows])


def cal_stats(path):
    g = defaultdict(list)
    for r in map(json.loads, open(path)):
        if "probs" in r:
            g[r["type"]].append((max(r["probs"]), int(r["pred"] == r["gold"])))
    return {t: (len(v), sum(c for _, c in v) / len(v), ece(v)) for t, v in g.items()}


L = ["# hipfire decide vs Jev", "", "## jev-bench (500 fixed-seed rows per task)", "",
     "| task | n | Jev acc | hipfire acc | Jev ECE | hipfire ECE |", "|---|---:|---:|---:|---:|---:|"]
for p in sorted(glob.glob(str(out / "jevbench/predictions/*.jsonl"))):
    name = Path(p).stem
    if name.startswith("x-"):
        continue
    n, acc, e = bench_stats(p)
    jn, jacc, je = bench_stats(bench / "predictions" / f"{name}.jsonl")
    L.append(f"| {name} | {n} | {jacc:.3f} | {acc:.3f} | {je:.3f} | {e:.3f} |")
L += ["", "## jev-ood-calibration", "", "| set | type | n | Jev acc | hipfire acc | Jev ECE | hipfire ECE |",
      "|---|---|---:|---:|---:|---:|---:|"]
for p in sorted(glob.glob(str(out / "calibration/hipfire_*.jsonl"))):
    s = Path(p).stem.removeprefix("hipfire_")
    mine, jev = cal_stats(p), cal_stats(cal / "results" / f"jev_{s}.jsonl")
    for t, (n, acc, e) in mine.items():
        jn, jacc, je = jev.get(t, (0, float("nan"), float("nan")))
        L.append(f"| {s} | {t} | {n} | {jacc:.3f} | {acc:.3f} | {je:.3f} | {e:.3f} |")
(out / "report.md").write_text("\n".join(L) + "\n")
print((out / "report.md").read_text())
```

`jevbench.run_task` writes `p_top` and `correct` per prediction row (the
same schema as the committed files), so `bench_stats` reads both sides
identically.

- [ ] **Step 4: README** (`scripts/jev_eval/README.md`)

````markdown
# Decide (Jev-compatible) evaluation

External benchmarks are cloned outside the repo:

```bash
git clone https://github.com/Running-Dolphins/jev-bench ~/repos/jev-evals/jev-bench
git clone https://github.com/scienthoon/jev-ood-calibration ~/repos/jev-evals/jev-ood-calibration
```

1. GPU gates (must all PASS before merge):
   `python3 scripts/jev_eval/gates.py --model ~/.hipfire/models/qwen3.5-4b.mq4`
2. Start serve on the repo daemon (one daemon per machine):
   `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon target/release/hipfire serve`
3. jev-bench, all 12 tasks and 6 experiments, at n=500 (aligned with Jev's committed rows):
   `python3 scripts/jev_eval/run_jevbench.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/jevbench all x-oos x-language x-options x-order x-repeat x-descriptions`
4. Calibration sets:
   `python3 scripts/jev_eval/run_calibration.py --model qwen3.5:4b --out bench/jev/qwen3.5-4b/calibration synth openbookqa commonsense_qa hellaswag`
5. Report: `python3 scripts/jev_eval/report.py --out bench/jev/qwen3.5-4b`

Jev's own latencies include the network and are not comparable. Compare
decide latency against the same model generating the answer via
`/v1/chat/completions` (see "Latency" in the PR description).
````

- [ ] **Step 5: Dry-run the scripts' imports**

Run:
`python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" scripts/jev_eval/*.py`
Expected: no output (all files parse).

- [ ] **Step 6: Commit**

```bash
git add scripts/jev_eval/
git commit -m "feat(decide): jev-bench / calibration eval harness and report"
```

---

### Task 10: Full evaluation run, latency comparison, docs

**Files:**
- Modify: `docs/specs/2026-09-28-jev-decide-design.md` (status line →
  `IMPLEMENTED — v1`)
- Run: `python3 scripts/check-crate-maps.py` (regenerates `map.md` for
  `hipfire-engine`, `hipfire-loader`, `hipfire-generate`, `hipfire-cli`)
- Create: `bench/jev/<model>/report.md` for Qwen3.5-4B and Qwen3.6-27B

- [ ] **Step 1: Regenerate the crate maps and run the no-GPU CI**

```bash
python3 scripts/check-crate-maps.py
./scripts/no-gpu-ci.sh
```

Expected: the maps are updated and the CI script exits 0.

- [ ] **Step 2: Evaluate Qwen3.5-4B** (README steps 2–5)

Expected: `bench/jev/qwen3.5-4b/report.md` with 12 jev-bench rows and 6
calibration rows. Every row shows a hipfire number; none is `nan`.

- [ ] **Step 3: Evaluate Qwen3.6-27B**

Check first: `grep MemAvailable /proc/meminfo`. You need ≥ 30 GiB available.
Restart serve and repeat steps 3–5 of the README with
`--model qwen3.6:27b --out bench/jev/qwen3.6-27b`.

- [ ] **Step 4: Latency comparison.** Measure on 200 AG News rows, the same
  examples through both paths.

```bash
python3 - <<'EOF'
import json, statistics, sys, time, urllib.request
sys.path.insert(0, "/home/nick/repos/jev-evals/jev-bench")
import jevbench as jb
ex, q = jb.TASKS["ag-news"][0](500)
URL = "http://127.0.0.1:11435"
def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter(); urllib.request.urlopen(req, timeout=300).read(); return (time.perf_counter() - t) * 1e3
dec, gen = [], []
for e in ex[:200]:
    dec.append(post("/v1/systemone", {"model": "qwen3.5:4b", "state": e["state"], "questions": {"q": q}}))
    opts = "\n".join(f"{k}: {v}" for k, v in q["criteria"].items())
    gen.append(post("/v1/chat/completions", {"model": "qwen3.5:4b", "max_tokens": 8, "temperature": 0,
        "messages": [{"role": "user", "content": f"{e['state']}\n\n{q['instructions']}\n{opts}\nAnswer with the option name only."}]}))
for name, xs in (("decide", dec), ("generate", gen)):
    xs.sort(); print(f"{name}: p50={xs[len(xs)//2]:.1f}ms p95={xs[int(len(xs)*.95)]:.1f}ms")
EOF
```

Record both lines in the PR description. There is no pass/fail threshold;
this is the measurement the spec asks for. If `generate` is faster, report
it as found: that would mean the snapshot overhead dominates at this state
size.

- [ ] **Step 5: Update the spec status and commit**

```bash
sed -i 's/Status: \*\*DRAFT — awaiting review\*\*/Status: **IMPLEMENTED — v1**/' docs/specs/2026-09-28-jev-decide-design.md
git add docs/specs/2026-09-28-jev-decide-design.md crates/*/map.md bench/jev/
git commit -m "docs(decide): v1 evaluation reports and crate maps"
```

- [ ] **Step 6: Hand back**

Don't push or open a PR without the user's go-ahead. When approved, push to
`origin` (warpfront/hipfire), never `fork`.
