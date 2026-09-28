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
