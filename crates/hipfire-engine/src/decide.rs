// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Jev-compatible "decide": typed questions answered by next-token logit
//! readout over single-token option codes. Pure — no GPU, no arch types.
//! Spec: `docs/specs/2026-09-28-jev-decide-design.md`.

use serde_json::{json, Value};

use hipfire_runtime::prompt_frame::Message;

pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_SCORE_LEVELS: usize = 10;
/// Upper bound on questions per request (serve-side DoS bound; each question
/// costs one suffix prefill).
pub const MAX_QUESTIONS: usize = 128;
/// hipfire bounds chosen to match Jev's published limits (spec §3/§7): the
/// shared state plus every question suffix, counted additively as in `usage_json`,
/// must be at most this. These bounds are counted in the loaded model's tokens
/// including chat-template overhead.
pub const MAX_TOTAL_TOKENS: usize = 64_000;
/// hipfire per-question bound matching Jev's limit: state + the longest question
/// (that question's full rendered prompt) must be at most this. Counted in the
/// loaded model's tokens including chat-template overhead, so a state Jev accepts
/// near the limit may be refused here.
pub const MAX_QUESTION_TOKENS: usize = 32_000;

#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    Choice {
        keys: Vec<String>,
        descriptions: Vec<String>,
    },
    Score {
        levels: Vec<String>,
    },
    Noul {
        true_desc: Option<String>,
        false_desc: Option<String>,
    },
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

/// Validate the `questions` map (shared by plain and session requests).
/// Every `Err` is a 422 message naming the question and field.
pub fn parse_questions(v: &Value) -> Result<Vec<Question>, String> {
    let qs = v
        .get("questions")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .ok_or("questions must be a non-empty object of name -> question")?;
    if qs.len() > MAX_QUESTIONS {
        return Err(format!(
            "questions must have at most {MAX_QUESTIONS} entries, got {}",
            qs.len()
        ));
    }
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
                let obj = criteria.and_then(Value::as_object).ok_or_else(|| {
                    format!("{name}.criteria must be an object of option -> description")
                })?;
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
                let arr = criteria.and_then(Value::as_array).ok_or_else(|| {
                    format!("{name}.criteria must be an array of level descriptions")
                })?;
                if !(2..=MAX_SCORE_LEVELS).contains(&arr.len()) {
                    return Err(format!(
                        "{name}.criteria must have 2..={MAX_SCORE_LEVELS} levels, got {}",
                        arr.len()
                    ));
                }
                QuestionKind::Score {
                    levels: arr.iter().map(desc_string).collect(),
                }
            }
            "noul" => {
                let (mut t, mut f) = (None, None);
                if let Some(c) = criteria {
                    let obj = c.as_object().ok_or_else(|| {
                        format!("{name}.criteria must be an object {{true, false}}")
                    })?;
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
                QuestionKind::Noul {
                    true_desc: t,
                    false_desc: f,
                }
            }
            other => {
                return Err(format!(
                    "{name}.type must be choice, score or noul, got {other:?}"
                ))
            }
        };
        questions.push(Question {
            name: name.clone(),
            instructions,
            kind,
        });
    }
    Ok(questions)
}

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
    let mut messages: Vec<Message> =
        serde_json::from_value(Value::Array(raw.clone())).map_err(|e| format!("messages: {e}"))?;
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

pub const SYSTEM_PROMPT: &str =
    "You answer classification questions about the state. Reply with one option code.";

/// `A`..`Z`, then `AA`..`ZZ`, keeping only candidates the model's tokenizer
/// encodes as exactly one token. Deterministic order.
pub fn assign_codes(
    k: usize,
    is_single_token: impl Fn(&str) -> bool,
) -> Result<Vec<String>, String> {
    let letters: Vec<char> = ('A'..='Z').collect();
    let singles = letters.iter().map(|c| c.to_string());
    let doubles = letters
        .iter()
        .flat_map(|a| letters.iter().map(move |b| format!("{a}{b}")));
    let codes: Vec<String> = singles
        .chain(doubles)
        .filter(|c| is_single_token(c))
        .take(k)
        .collect();
    if codes.len() < k {
        return Err(format!(
            "model tokenizer supplies only {} single-token option codes, need {k}",
            codes.len()
        ));
    }
    Ok(codes)
}

pub fn labels_for(
    q: &Question,
    is_single_token: impl Fn(&str) -> bool,
) -> Result<Vec<String>, String> {
    let labels: Vec<String> = match &q.kind {
        QuestionKind::Choice { keys, .. } => return assign_codes(keys.len(), is_single_token),
        QuestionKind::Score { levels } => (0..levels.len()).map(|i| i.to_string()).collect(),
        QuestionKind::Noul { .. } => vec!["Yes".into(), "No".into()],
    };
    if let Some(bad) = labels.iter().find(|l| !is_single_token(l)) {
        return Err(format!(
            "{}: label {bad:?} is not a single token for this model",
            q.name
        ));
    }
    Ok(labels)
}

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

/// Jev's per-question token limit on one question's full rendered prompt
/// (state + that question). Checked per question as it is rendered, so an
/// over-long state fails on the first question without rendering the rest.
pub fn check_question_tokens(name: &str, len: usize) -> Result<(), String> {
    if len > MAX_QUESTION_TOKENS {
        return Err(format!(
            "{name}: state + question is {len} tokens, over the per-question limit of \
             {MAX_QUESTION_TOKENS} tokens"
        ));
    }
    Ok(())
}

/// Jev's request token limit: shared prefix once plus every suffix (the
/// `usage_json` accounting at `prefix_len`) must be at most
/// `MAX_TOTAL_TOKENS`. Precondition: every `seq_lens[i] >= prefix_len`.
pub fn check_total_tokens(prefix_len: usize, seq_lens: &[usize]) -> Result<(), String> {
    let total = prefix_len
        + seq_lens
            .iter()
            .map(|l| l.saturating_sub(prefix_len))
            .sum::<usize>();
    if total > MAX_TOTAL_TOKENS {
        return Err(format!(
            "state + all questions is {total} tokens, over the request limit of \
             {MAX_TOTAL_TOKENS} tokens"
        ));
    }
    Ok(())
}

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
    debug_assert!(
        label_logits.len() >= 2,
        "assemble_answer: need at least 2 labels"
    );
    let p = softmax(label_logits);
    let k = p.len() as f64;
    let top = argmax(&p);
    match &q.kind {
        QuestionKind::Choice { keys, .. } => {
            let probs: serde_json::Map<String, Value> = keys
                .iter()
                .cloned()
                .zip(p.iter().map(|&x| json!(x)))
                .collect();
            let confidence = (p[top] - 1.0 / k) / (1.0 - 1.0 / k);
            json!({"type": "choice", "choice": keys[top], "probabilities": probs,
                   "confidence": confidence})
        }
        QuestionKind::Score { .. } => {
            let probs: serde_json::Map<String, Value> = p
                .iter()
                .enumerate()
                .map(|(i, &x)| (i.to_string(), json!(x)))
                .collect();
            let score: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
            json!({"type": "score", "score": score, "probabilities": probs,
                   "confidence": p[top]})
        }
        QuestionKind::Noul { .. } => json!({"type": "noul", "noul": p[0]}),
    }
}

/// Additive accounting observed for Jev: shared prefix once, each suffix once.
/// Precondition: every `seq_lens[i] >= prefix_len`.
pub fn usage_json(prefix_len: usize, seq_lens: &[usize]) -> Value {
    debug_assert!(
        seq_lens.iter().all(|&l| l >= prefix_len),
        "usage_json: sequence shorter than prefix_len"
    );
    let input: usize = prefix_len
        + seq_lens
            .iter()
            .map(|l| l.saturating_sub(prefix_len))
            .sum::<usize>();
    json!({"input_tokens": input, "output_tokens": 0})
}

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
                assert_eq!(
                    descriptions,
                    &vec!["money".to_string(), "parcels".to_string()]
                );
            }
            _ => panic!("wrong kind"),
        }
        let s = r.questions.iter().find(|q| q.name == "q_score").unwrap();
        assert!(matches!(&s.kind, QuestionKind::Score { levels } if levels.len() == 3));
        let n = r.questions.iter().find(|q| q.name == "q_noul").unwrap();
        assert!(matches!(
            &n.kind,
            QuestionKind::Noul {
                true_desc: None,
                false_desc: None
            }
        ));
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
        assert!(err_of(q(
            json!({"type": "choice", "criteria": {"a": "x", "b": "y"}})
        ))
        .contains("q.instructions"));
        assert!(err_of(q(
            json!({"type": "choice", "instructions": "i", "criteria": {"a": "x"}})
        ))
        .contains("2..=255"));
        let many: serde_json::Map<String, serde_json::Value> =
            (0..256).map(|i| (format!("k{i}"), json!("d"))).collect();
        assert!(err_of(q(
            json!({"type": "choice", "instructions": "i", "criteria": many})
        ))
        .contains("2..=255"));
        assert!(err_of(q(
            json!({"type": "score", "instructions": "i", "criteria": ["a"]})
        ))
        .contains("2..=10"));
        assert!(err_of(q(json!({"type": "score", "instructions": "i",
                                "criteria": ["a","b","c","d","e","f","g","h","i","j","k"]})))
        .contains("2..=10"));
        assert!(err_of(q(json!({"type": "boolean", "instructions": "i"}))).contains("q.type"));
        assert!(err_of(q(
            json!({"type": "noul", "instructions": "i", "criteria": {"true": 1}})
        ))
        .contains("q.criteria"));
    }

    #[test]
    fn rejects_more_than_max_questions() {
        let qs = |n: usize| -> serde_json::Map<String, serde_json::Value> {
            (0..n)
                .map(|i| {
                    (
                        format!("q{i}"),
                        json!({"type": "noul", "instructions": "i"}),
                    )
                })
                .collect()
        };
        assert_eq!(
            parse_request(&json!({"state": "s", "questions": qs(MAX_QUESTIONS)}))
                .unwrap()
                .questions
                .len(),
            MAX_QUESTIONS
        );
        let e = err_of(json!({"state": "s", "questions": qs(MAX_QUESTIONS + 1)}));
        assert!(e.contains("at most 128"), "{e}");
    }

    #[test]
    fn choice_keys_and_questions_keep_request_order() {
        // serde_json `preserve_order` (enabled in this crate's manifest):
        // criteria order = request order, independent of feature unification,
        // so option codes A, B, C map to keys in the order the caller wrote.
        let v: serde_json::Value = serde_json::from_str(
            r#"{"state": "s", "questions": {
                "zeta": {"type": "noul", "instructions": "i"},
                "alpha": {"type": "choice", "instructions": "i",
                          "criteria": {"shipping": "x", "billing": "y", "general": "z"}}}}"#,
        )
        .unwrap();
        let r = parse_request(&v).unwrap();
        let names: Vec<&str> = r.questions.iter().map(|q| q.name.as_str()).collect();
        assert_eq!(names, vec!["zeta", "alpha"]);
        match &r.questions[1].kind {
            QuestionKind::Choice { keys, descriptions } => {
                assert_eq!(keys, &vec!["shipping", "billing", "general"]);
                assert_eq!(descriptions, &vec!["x", "y", "z"]);
            }
            other => panic!("wrong kind {other:?}"),
        }
        let a = assemble_answer(&r.questions[1], &[0.0, 1.0, 0.0]);
        let keys: Vec<&String> = a["probabilities"].as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["shipping", "billing", "general"]);
        assert_eq!(a["choice"], "billing");
    }

    #[test]
    fn question_token_limit() {
        assert!(check_question_tokens("q", MAX_QUESTION_TOKENS).is_ok());
        let e = check_question_tokens("q", MAX_QUESTION_TOKENS + 1).unwrap_err();
        assert!(e.starts_with("q: "), "{e}");
        assert!(e.contains("per-question limit of 32000"), "{e}");
    }

    #[test]
    fn total_token_limit_counts_prefix_once() {
        // 3 x 30000-token prompts sharing a 25000-token prefix:
        // 25000 + 3 x 5000 = 40000 <= 64000.
        assert!(check_total_tokens(25_000, &[30_000; 3]).is_ok());
        // Exactly at the limit.
        assert!(check_total_tokens(4_000, &[34_000, 34_000]).is_ok());
        let e = check_total_tokens(4_000, &[34_000, 34_001]).unwrap_err();
        assert!(e.contains("64001 tokens"), "{e}");
        assert!(e.contains("request limit of 64000"), "{e}");
        // No shared prefix: every prompt counts in full.
        assert!(check_total_tokens(0, &[32_000, 32_001]).is_err());
    }

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
        let score = Question {
            name: "s".into(),
            instructions: "i".into(),
            kind: QuestionKind::Score {
                levels: vec!["a".into(), "b".into(), "c".into()],
            },
        };
        assert_eq!(labels_for(&score, all).unwrap(), vec!["0", "1", "2"]);
        let noul = Question {
            name: "n".into(),
            instructions: "i".into(),
            kind: QuestionKind::Noul {
                true_desc: None,
                false_desc: None,
            },
        };
        assert_eq!(labels_for(&noul, all).unwrap(), vec!["Yes", "No"]);
    }

    #[test]
    fn labels_error_when_digit_or_yes_no_is_not_single_token() {
        let score = Question {
            name: "s".into(),
            instructions: "i".into(),
            kind: QuestionKind::Score {
                levels: vec!["a".into(), "b".into()],
            },
        };
        assert!(labels_for(&score, |s: &str| s != "1").is_err());
        let noul = Question {
            name: "n".into(),
            instructions: "i".into(),
            kind: QuestionKind::Noul {
                true_desc: None,
                false_desc: None,
            },
        };
        assert!(labels_for(&noul, |s: &str| s != "Yes").is_err());
    }

    #[test]
    fn user_text_layout() {
        let q = Question {
            name: "q".into(),
            instructions: "Which queue?".into(),
            kind: QuestionKind::Choice {
                keys: vec!["billing".into(), "shipping".into()],
                descriptions: vec!["money".into(), "parcels".into()],
            },
        };
        let t = build_user_text("the state", &q, &["A".into(), "B".into()]);
        assert_eq!(t, "State:\nthe state\n\nQuestion: Which queue?\nOptions:\nA. money\nB. parcels\n\nAnswer with the option code only.");
        let n = Question {
            name: "n".into(),
            instructions: "Angry?".into(),
            kind: QuestionKind::Noul {
                true_desc: Some("shouting".into()),
                false_desc: None,
            },
        };
        let t = build_user_text("s", &n, &["Yes".into(), "No".into()]);
        assert!(t.ends_with("Options:\nYes. shouting\nNo. no\n\nAnswer with the option code only."));
    }

    #[test]
    fn shared_prefix_is_lcp_capped_below_shortest() {
        assert_eq!(
            shared_prefix_len(&[vec![1, 2, 3, 9], vec![1, 2, 3, 8, 7]]),
            3
        );
        // Identical sequences: cap at len - 1 so each suffix is non-empty.
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3], vec![1, 2, 3]]), 2);
        assert_eq!(shared_prefix_len(&[vec![5, 6, 7]]), 2);
        assert_eq!(shared_prefix_len(&[vec![1], vec![2]]), 0);
    }

    fn choice_q(keys: &[&str]) -> Question {
        Question {
            name: "q".into(),
            instructions: "i".into(),
            kind: QuestionKind::Choice {
                keys: keys.iter().map(|s| s.to_string()).collect(),
                descriptions: keys.iter().map(|s| s.to_string()).collect(),
            },
        }
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
        let q = Question {
            name: "s".into(),
            instructions: "i".into(),
            kind: QuestionKind::Score {
                levels: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            },
        };
        // Jev synth score row: probs [0, .05, .89, .06], confidence .89.
        let probs = [1e-9_f32, 0.05, 0.89, 0.06];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let a = assemble_answer(&q, &logits);
        assert_eq!(a["type"], "score");
        let score = a["score"].as_f64().unwrap();
        assert!(
            (score - (0.05 + 2.0 * 0.89 + 3.0 * 0.06)).abs() < 1e-4,
            "score {score}"
        );
        assert!((a["confidence"].as_f64().unwrap() - 0.89).abs() < 1e-4);
        assert!(a["probabilities"].get("2").is_some());
    }

    #[test]
    fn noul_is_p_yes_without_confidence() {
        let q = Question {
            name: "n".into(),
            instructions: "i".into(),
            kind: QuestionKind::Noul {
                true_desc: None,
                false_desc: None,
            },
        };
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

    #[test]
    #[should_panic(expected = "sequence shorter than prefix_len")]
    fn usage_rejects_seq_shorter_than_prefix() {
        usage_json(10, &[5]);
    }

    #[test]
    #[should_panic(expected = "need at least 2 labels")]
    fn assemble_rejects_single_label() {
        let q = Question {
            name: "q".into(),
            instructions: "i".into(),
            kind: QuestionKind::Choice {
                keys: vec!["A".into()],
                descriptions: vec!["only".into()],
            },
        };
        assemble_answer(&q, &[0.0]);
    }

    #[test]
    fn request_mode_requires_exactly_one_of_state_or_messages() {
        assert_eq!(request_mode(&json!({"state": "s"})), Ok(DecideMode::Plain));
        assert_eq!(
            request_mode(&json!({"messages": []})),
            Ok(DecideMode::Session)
        );
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
        assert!(
            no_tools.tools.is_none(),
            "empty tools == no tools, as generate"
        );
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
        assert_eq!(
            build_user_text("S", &q, &labels),
            format!("State:\nS\n\n{qt}")
        );
        assert!(
            qt.starts_with("Question: i\nOptions:\nA. a\nB. b\n"),
            "{qt}"
        );
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
        assert_eq!(
            conversation_end(&s2, &[OPEN, 4, 1, OPEN, 5, 3], Some(OPEN)),
            3
        );
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
            Resume {
                ckpt_idx: 1,
                pos: 2
            }
        );
        assert_eq!(
            plan_session_start(&[1, 2, 3, 4], &conv, 4, true, &[3], true),
            Cold
        );
        assert_eq!(
            plan_session_start(&[1, 2, 3, 4], &conv, 4, true, &[1], false),
            Cold
        );
        // Cached conversation runs past the request's: rewind by checkpoint.
        assert_eq!(
            plan_session_start(&[1, 2, 9, 9, 5], &conv, 5, true, &[2, 4], true),
            Resume {
                ckpt_idx: 1,
                pos: 4
            }
        );
        // Not reusable: ineligible, empty, or cursor out of step with tokens.
        assert_eq!(
            plan_session_start(&[1, 2], &conv, 2, false, &[], true),
            Cold
        );
        assert_eq!(plan_session_start(&[], &conv, 0, true, &[], true), Cold);
        assert_eq!(plan_session_start(&[1, 2], &conv, 7, true, &[], true), Cold);
    }

    #[test]
    fn session_limits_count_tokens_beyond_the_conversation() {
        let qs = vec![choice_q(&["a", "b"]), choice_q(&["c", "d"])];
        // A long conversation is bounded by max_seq, not Jev's limits.
        assert!(check_session_tokens(&qs, 100_000, &[100_010, 100_020]).is_ok());
        let e = check_session_tokens(&qs, 10, &[10 + MAX_QUESTION_TOKENS + 1, 20]).unwrap_err();
        assert!(
            e.contains("q:") && e.contains("per-question limit of 32000"),
            "{e}"
        );
        let qs3 = vec![choice_q(&["a", "b"]); 3];
        let e = check_session_tokens(&qs3, 5, &[5 + 30_000, 5 + 30_000, 5 + 4_001]).unwrap_err();
        assert!(e.contains("request limit of 64000"), "{e}");
        assert!(check_session_tokens(&qs3, 5, &[5 + 30_000, 5 + 30_000, 5 + 4_000]).is_ok());
    }
}
