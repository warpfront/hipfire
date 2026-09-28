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
    debug_assert!(label_logits.len() >= 2, "assemble_answer: need at least 2 labels");
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
/// Precondition: every `seq_lens[i] >= prefix_len`.
pub fn usage_json(prefix_len: usize, seq_lens: &[usize]) -> Value {
    debug_assert!(seq_lens.iter().all(|&l| l >= prefix_len), "usage_json: sequence shorter than prefix_len");
    let input: usize = prefix_len + seq_lens.iter().map(|l| l.saturating_sub(prefix_len)).sum::<usize>();
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

    #[test]
    #[should_panic(expected = "sequence shorter than prefix_len")]
    fn usage_rejects_seq_shorter_than_prefix() {
        usage_json(10, &[5]);
    }

    #[test]
    #[should_panic(expected = "need at least 2 labels")]
    fn assemble_rejects_single_label() {
        let q = Question { name: "q".into(), instructions: "i".into(),
            kind: QuestionKind::Choice { keys: vec!["A".into()],
                descriptions: vec!["only".into()] } };
        assemble_answer(&q, &[0.0]);
    }
}
