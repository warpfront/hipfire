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
        Self {
            status,
            message: message.into(),
            required_max_seq: None,
        }
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

/// Unwraps a Carrier decide hook result. `None` means the carrier claimed
/// `decide_supported()` but didn't actually provide this hook — an internal
/// carrier contract violation, hence 500 (not a client-facing 4xx).
fn hook<T>(r: Option<Result<T, String>>) -> Result<T, DecideError> {
    r.ok_or_else(|| DecideError::new(500, "decide hook returned None despite decide_supported()"))?
        .map_err(|e| DecideError::new(500, e))
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
        return Err(DecideError::new(
            409,
            "decide requires a single-GPU model (pp=1, no EP)",
        ));
    }
    if m.eviction.is_some() || m.kv_adaptive.is_some() {
        return Err(DecideError::new(
            409,
            "decide requires KV eviction (CASK) and kv_adaptive off",
        ));
    }
    let carrier = hipfire_loader::carrier_for(m.arch_id)
        .filter(|c| c.decide_supported())
        .ok_or_else(|| {
            DecideError::new(
                400,
                format!("decide not supported for arch_id {}", m.arch_id),
            )
        })?;

    // All tokenizer work happens here, inside this block, before the first
    // `&mut m` hook call below — `tok` borrows `m.tokenizer` immutably and
    // must not still be alive when `rollback(m, gpu)` takes `&mut m`.
    let (label_ids, seqs): (Vec<Vec<u32>>, Vec<Vec<u32>>) = {
        let tok = m
            .tokenizer
            .as_ref()
            .ok_or_else(|| DecideError::new(400, "model has no tokenizer"))?;
        let single = |s: &str| tok.encode(s).len() == 1;

        let mut label_ids: Vec<Vec<u32>> = Vec::new();
        let mut seqs: Vec<Vec<u32>> = Vec::new();
        let mut any_started_in_think = false;
        for q in &parsed.questions {
            let labels = d::labels_for(q, single).map_err(|e| DecideError::new(422, e))?;
            label_ids.push(labels.iter().map(|l| tok.encode(l)[0]).collect());
            let user = d::build_user_text(&parsed.state_text, q, &labels);
            let (tokens, started_in_think) = hipfire_engine::prompt::batch_render_prompt_tokens(
                &user,
                Some(d::SYSTEM_PROMPT),
                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
                tok,
                m.chat_template.as_ref(),
                0,
                None,
                false,
                None,
            )
            .map_err(|e| DecideError::new(500, format!("prompt render: {e}")))?;
            any_started_in_think |= started_in_think;
            seqs.push(tokens);
        }
        if any_started_in_think {
            return Err(DecideError::new(
                400,
                "model chat template cannot disable thinking; decide needs a closed-think assistant turn",
            ));
        }
        (label_ids, seqs)
    };

    let longest = seqs.iter().map(Vec::len).max().unwrap_or(0);
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
    let prefix_len = if no_snapshot {
        0
    } else {
        d::shared_prefix_len(&seqs)
    };

    rollback(m, gpu)?;
    let mut timing =
        json!({"prefix_tokens": prefix_len, "state_prefill_ms": 0.0, "question_ms": []});
    let mut per_q_logits: Vec<Vec<f32>> = Vec::with_capacity(seqs.len());

    let result: Result<(), DecideError> = (|| {
        let mut snapshot: Option<hipfire_loader::decide::DecideSnapshot> = None;
        if prefix_len > 0 {
            let t = Instant::now();
            hook(carrier.decide_prefill_logits(m, gpu, &seqs[0][..prefix_len], 0))?;
            snapshot = Some(hook(carrier.decide_save(m, gpu))?);
            timing["state_prefill_ms"] = json!(ms(t));
        }
        let loop_result: Result<(), DecideError> = (|| {
            for (i, seq) in seqs.iter().enumerate() {
                let t = Instant::now();
                match &snapshot {
                    Some(s) if i > 0 => hook(carrier.decide_restore(m, gpu, s))?,
                    Some(_) => {} // first question continues straight from the snapshot point
                    None => {
                        if i > 0 {
                            rollback(m, gpu)?;
                        }
                    }
                }
                let logits =
                    hook(carrier.decide_prefill_logits(m, gpu, &seq[prefix_len..], prefix_len))?;
                let label_logits: Result<Vec<f32>, DecideError> = label_ids[i]
                    .iter()
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
                    .collect();
                per_q_logits.push(label_logits?);
                timing["question_ms"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!(ms(t)));
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

/// Full dispatch for a `decide` JSONL message: refuses with 409 while
/// `lanes_active` (a slot or continuous-batch request is in flight), refuses
/// with 503 when no model is loaded, otherwise runs [`run_decide`] and builds
/// the `decided` reply. Returns `(reply, ran_decide)` so the caller can
/// advance `state_epoch` only when `run_decide` actually ran -- the same rule
/// the daemon applies to `reset`.
pub fn handle_decide_message(
    model: Option<&mut LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    msg: &Value,
    lanes_active: bool,
) -> (Value, bool) {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let refuse = |status: u16, message: &str| {
        DecideError {
            status,
            message: message.to_string(),
            required_max_seq: None,
        }
        .to_reply(&id)
    };
    if lanes_active {
        (
            refuse(
                409,
                "decide refused: slot or continuous-batch requests active",
            ),
            false,
        )
    } else if let Some(m) = model {
        let reply = match run_decide(m, gpu, msg) {
            Ok(out) => json!({
                "type": "decided",
                "id": id,
                "answers": out.answers,
                "usage": out.usage,
                "timing": out.timing,
            }),
            Err(e) => e.to_reply(&id),
        };
        (reply, true)
    } else {
        (refuse(503, "no model loaded"), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_reply_shape() {
        let e = DecideError {
            status: 422,
            message: "too long".into(),
            required_max_seq: Some(9000),
        };
        let r = e.to_reply("d1");
        assert_eq!(r["type"], "decided");
        assert_eq!(r["id"], "d1");
        assert_eq!(r["error"]["status"], 422);
        assert_eq!(r["error"]["message"], "too long");
        assert_eq!(r["error"]["required_max_seq"], 9000);
        let e = DecideError {
            status: 409,
            message: "x".into(),
            required_max_seq: None,
        };
        assert!(e.to_reply("d2")["error"].get("required_max_seq").is_none());
    }

    #[test]
    fn hook_unwraps_carrier_results() {
        let e = hook::<u32>(None).unwrap_err();
        assert_eq!(e.status, 500);

        let e = hook::<u32>(Some(Err("x".to_string()))).unwrap_err();
        assert_eq!(e.status, 500);
        assert_eq!(e.message, "x");

        assert_eq!(hook(Some(Ok(7))).unwrap(), 7);
    }

    // Non-GPU branches of `handle_decide_message`: both short-circuit before
    // touching the model or GPU, so a real `Gpu` handle is only needed to
    // satisfy the signature. Skip gracefully where no GPU is available,
    // matching the `let Ok(mut gpu) = rdna_compute::Gpu::init() else { return; }`
    // convention used elsewhere in the tree.
    #[test]
    fn handle_decide_message_lanes_active_refuses_without_running() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let msg = json!({"type": "decide", "id": "lanes-1", "state": "x", "questions": {}});
        let (reply, ran) = handle_decide_message(None, &mut gpu, &msg, true);
        assert!(!ran);
        assert_eq!(reply["type"], "decided");
        assert_eq!(reply["id"], "lanes-1");
        assert_eq!(reply["error"]["status"], 409);
    }

    #[test]
    fn handle_decide_message_no_model_refuses_without_running() {
        let Ok(mut gpu) = rdna_compute::Gpu::init() else {
            return;
        };
        let msg = json!({"type": "decide", "id": "nomodel-1", "state": "x", "questions": {}});
        let (reply, ran) = handle_decide_message(None, &mut gpu, &msg, false);
        assert!(!ran);
        assert_eq!(reply["type"], "decided");
        assert_eq!(reply["id"], "nomodel-1");
        assert_eq!(reply["error"]["status"], 503);
        assert_eq!(reply["error"]["message"], "no model loaded");
    }
}
