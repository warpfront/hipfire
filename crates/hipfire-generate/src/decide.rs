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

/// Admission rule for decide when KV eviction (CASK / TriAttention) is
/// configured. Decide prefills the whole prompt through the carrier's
/// `forward_prefill_batch` and never calls `Eviction::maybe_evict` (that runs
/// only in the generate loops), so eviction cannot fire inside decide. What
/// bounds it is where the prefill writes: KV slot == position (uncompacted),
/// and the KV buffers hold only `LoadedModel::physical_cap` slots
/// (`CaskConfig::physical_cap`: by default budget + beta + 256, clamped to
/// max_seq) with no bounds check in the prefill. The limit is the smaller of the eviction
/// budget (`Eviction::budget()`, i.e. `cask_budget`) and `physical_cap`, so a
/// decide never holds more context than generate would retain after eviction.
/// Growing max_seq does not help here, hence no `required_max_seq`.
fn check_eviction_limit(
    longest: usize,
    eviction_budget: Option<usize>,
    physical_cap: usize,
) -> Result<(), DecideError> {
    let Some(budget) = eviction_budget else {
        return Ok(());
    };
    let limit = budget.min(physical_cap);
    if longest + 1 > limit {
        return Err(DecideError::new(
            422,
            format!(
                "prompt needs {} tokens, KV eviction (CASK) is configured and decide is limited to \
                 {limit} tokens (min of eviction budget {budget} and physical KV capacity \
                 {physical_cap})",
                longest + 1
            ),
        ));
    }
    Ok(())
}

/// Positional KV rewind requires `compact_offset == 0`. `None` means the
/// carrier claimed `decide_supported()` without this hook: contract violation.
fn check_uncompacted(compact_offset: Option<usize>) -> Result<(), DecideError> {
    match compact_offset {
        Some(0) => Ok(()),
        Some(n) => Err(DecideError::new(
            409,
            format!("decide requires an uncompacted KV cache (compact_offset={n} after reset)"),
        )),
        None => Err(DecideError::new(
            500,
            "decide_kv_compact_offset hook missing despite decide_supported()",
        )),
    }
}

/// The token id `s` encodes to when it is exactly one token, ignoring a
/// leading BOS that tokenizers with HF `add_bos_token` (Llama / Mistral
/// style) prepend in `Tokenizer::encode`. `bos` is that id when the tokenizer
/// adds one, else `None`. Used both for the single-token predicate and the
/// label-id lookup, so the two can never disagree.
fn single_token_id(encode: impl Fn(&str) -> Vec<u32>, bos: Option<u32>, s: &str) -> Option<u32> {
    let ids = encode(s);
    let ids = match (bos, ids.split_first()) {
        (Some(b), Some((&first, rest))) if first == b => rest,
        _ => &ids[..],
    };
    match ids {
        [id] => Some(*id),
        _ => None,
    }
}

/// Every question's rendered prompt: option labels' token ids and the full
/// token sequence, in request order.
pub struct RenderedQuestions {
    pub label_ids: Vec<Vec<u32>>,
    pub seqs: Vec<Vec<u32>>,
}

/// Render and tokenize each question's full decide prompt (labels, label
/// token ids, token sequence) exactly as the daemon's decide does. Each
/// sequence is checked against Jev's per-question token limit
/// (`d::check_question_tokens`) and the closed-think template check as soon
/// as it is rendered, so an over-long state or an unusable template fails on
/// the first question without rendering the rest. Shared by [`run_decide`] and the
/// `split_prefill_probe` example.
pub fn render_question_prompts(
    tok: &hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<&String>,
    parsed: &d::DecideRequest,
) -> Result<RenderedQuestions, DecideError> {
    let bos = tok.add_bos.then_some(tok.bos_id);
    let encode = |s: &str| tok.encode(s);
    let single = |s: &str| single_token_id(encode, bos, s).is_some();
    let mut out = RenderedQuestions {
        label_ids: Vec::with_capacity(parsed.questions.len()),
        seqs: Vec::with_capacity(parsed.questions.len()),
    };
    for q in &parsed.questions {
        let labels = d::labels_for(q, single).map_err(|e| DecideError::new(422, e))?;
        let ids = labels
            .iter()
            .map(|l| {
                single_token_id(encode, bos, l).ok_or_else(|| {
                    DecideError::new(500, format!("label {l:?} lost its single-token id"))
                })
            })
            .collect::<Result<Vec<u32>, _>>()?;
        let user = d::build_user_text(&parsed.state_text, q, &labels);
        let (tokens, started_in_think) = hipfire_engine::prompt::batch_render_prompt_tokens(
            &user,
            Some(d::SYSTEM_PROMPT),
            hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
            tok,
            chat_template,
            0,
            None,
            false,
            None,
        )
        .map_err(|e| DecideError::new(500, format!("prompt render: {e}")))?;
        if started_in_think {
            return Err(DecideError::new(
                400,
                "model chat template cannot disable thinking; decide needs a closed-think assistant turn",
            ));
        }
        d::check_question_tokens(&q.name, tokens.len()).map_err(|e| DecideError::new(422, e))?;
        out.label_ids.push(ids);
        out.seqs.push(tokens);
    }
    Ok(out)
}

/// Everything [`execute_decide`] needs, computed without touching model or
/// GPU state.
pub struct DecidePlan {
    parsed: d::DecideRequest,
    carrier: &'static dyn hipfire_loader::Carrier,
    rendered: RenderedQuestions,
    prefix_len: usize,
}

/// Validate, render and admit a decide request. Read-only on the model: any
/// `Err` here leaves model state untouched (no reset has happened).
pub fn plan_decide(m: &LoadedModel, req: &Value) -> Result<DecidePlan, DecideError> {
    let parsed = d::parse_request(req).map_err(|e| DecideError::new(422, e))?;
    let no_snapshot = req.get("_debug_no_snapshot").and_then(Value::as_bool) == Some(true);

    if m.pp > 1 || m.ep.is_some() {
        return Err(DecideError::new(
            409,
            "decide requires a single-GPU model (pp=1, no EP)",
        ));
    }
    // CASK/TriAttention eviction may be configured (serve loads with it on):
    // decide never calls `maybe_evict`, so it is admitted when its prompts fit
    // the eviction-safe limit (checked below, after tokenizing) and the cache
    // is uncompacted after the pre-decide rollback. kv_adaptive stays refused.
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
    let tok = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| DecideError::new(400, "model has no tokenizer"))?;
    let rendered = render_question_prompts(tok, m.chat_template.as_ref(), &parsed)?;
    let seqs = &rendered.seqs;
    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    // Jev's request limit is judged on the real shared prefix, independent
    // of the execution mode chosen below.
    d::check_total_tokens(d::shared_prefix_len(seqs), &lens)
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
    // A single question is one plain full prefill: no shared prefix, no
    // snapshot allocation, no restore.
    let prefix_len = if no_snapshot || seqs.len() == 1 {
        0
    } else {
        d::shared_prefix_len(seqs)
    };
    Ok(DecidePlan {
        parsed,
        carrier,
        rendered,
        prefix_len,
    })
}

/// Run a planned decide. Starts with a rollback (reset) of the model and
/// always ends with one, even on failure, so once this is called the model
/// state has been reset.
pub fn execute_decide(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    plan: DecidePlan,
) -> Result<DecideOutcome, DecideError> {
    let t_total = Instant::now();
    let DecidePlan {
        parsed,
        carrier,
        rendered: RenderedQuestions { label_ids, seqs },
        prefix_len,
    } = plan;

    rollback(m, gpu)?;
    // Positional KV rewind is only valid on an uncompacted cache; the rollback
    // above resets `compact_offset`, this attests it for the carrier's KV.
    check_uncompacted(carrier.decide_kv_compact_offset(m))?;
    let mut timing =
        json!({"prefix_tokens": prefix_len, "state_prefill_ms": 0.0, "question_ms": []});
    let mut per_q_logits: Vec<Vec<f32>> = Vec::with_capacity(seqs.len());

    let mut snapshot: Option<hipfire_loader::decide::DecideSnapshot> = None;
    let result = run_questions(
        carrier,
        m,
        gpu,
        &label_ids,
        &seqs,
        prefix_len,
        &mut snapshot,
        &mut timing,
        &mut per_q_logits,
    );
    if let Some(s) = snapshot.take() {
        s.free(gpu);
    }
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

/// Shared-prefix prefill + snapshot, then per question restore → suffix
/// prefill → label-logit gather. The snapshot is handed back through
/// `snapshot` so the caller frees it on every path.
#[allow(clippy::too_many_arguments)]
fn run_questions(
    carrier: &dyn hipfire_loader::Carrier,
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    label_ids: &[Vec<u32>],
    seqs: &[Vec<u32>],
    prefix_len: usize,
    snapshot: &mut Option<hipfire_loader::decide::DecideSnapshot>,
    timing: &mut Value,
    per_q_logits: &mut Vec<Vec<f32>>,
) -> Result<(), DecideError> {
    if prefix_len > 0 {
        let t = Instant::now();
        // The prefix's own logits are never read: skip the vocab download.
        hook(carrier.decide_prefill_logits(m, gpu, &seqs[0][..prefix_len], 0, false))?;
        *snapshot = Some(hook(carrier.decide_save(m, gpu))?);
        timing["state_prefill_ms"] = json!(ms(t));
    }
    for (i, seq) in seqs.iter().enumerate() {
        let t = Instant::now();
        match snapshot.as_ref() {
            Some(s) if i > 0 => hook(carrier.decide_restore(m, gpu, s))?,
            Some(_) => {} // first question continues straight from the snapshot point
            None if i > 0 => rollback(m, gpu)?,
            None => {} // first question: the pre-decide rollback already reset
        }
        let logits =
            hook(carrier.decide_prefill_logits(m, gpu, &seq[prefix_len..], prefix_len, true))?;
        let label_logits = label_ids[i]
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
            .collect::<Result<Vec<f32>, DecideError>>()?;
        per_q_logits.push(label_logits);
        timing["question_ms"]
            .as_array_mut()
            .expect("question_ms is an array")
            .push(json!(ms(t)));
    }
    Ok(())
}

pub fn run_decide(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    req: &Value,
) -> Result<DecideOutcome, DecideError> {
    let plan = plan_decide(m, req)?;
    execute_decide(m, gpu, plan)
}

/// The non-GPU part of [`handle_decide_message`]: the refusal reply, if any,
/// before a decide may run. Multi-slot mode first (the slot engine owns the
/// weights and there is no ordinary model to decide on), then in-flight
/// continuous-batch lanes, then no model loaded.
pub fn precheck(
    model_present: bool,
    lanes_active: bool,
    slot_mode: bool,
    id: &str,
) -> Option<Value> {
    let (status, message) = if slot_mode {
        (409, "decide unsupported in multi-slot mode")
    } else if lanes_active {
        (
            409,
            "decide refused: continuous-batch requests active",
        )
    } else if !model_present {
        (503, "no model loaded")
    } else {
        return None;
    };
    Some(DecideError::new(status, message).to_reply(id))
}

/// Full dispatch for a `decide` JSONL message: [`precheck`] refusals, then
/// [`plan_decide`] (validation, rendering, admission; read-only), then
/// [`execute_decide`], building the `decided` reply. Returns `(reply,
/// reset)`: `reset` is true only when `execute_decide` ran, i.e. the model
/// was actually rolled back, so the caller advances `state_epoch` exactly
/// when model state was reset -- the rule the daemon applies to `reset`.
/// Refusals and validation errors (422 etc.) return `false`.
pub fn handle_decide_message(
    model: Option<&mut LoadedModel>,
    gpu: &mut rdna_compute::Gpu,
    msg: &Value,
    lanes_active: bool,
    slot_mode: bool,
) -> (Value, bool) {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(reply) = precheck(model.is_some(), lanes_active, slot_mode, id) {
        return (reply, false);
    }
    let Some(m) = model else {
        unreachable!("precheck refuses when no model is present");
    };
    let plan = match plan_decide(m, msg) {
        Ok(p) => p,
        Err(e) => return (e.to_reply(id), false),
    };
    let reply = match execute_decide(m, gpu, plan) {
        Ok(out) => json!({
            "type": "decided",
            "id": id,
            "answers": out.answers,
            "usage": out.usage,
            "timing": out.timing,
        }),
        Err(e) => e.to_reply(id),
    };
    (reply, true)
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
    fn eviction_limit_admits_without_eviction_and_within_limit() {
        // No eviction: never limits here (max_seq is checked separately).
        assert!(check_eviction_limit(1_000_000, None, 16).is_ok());
        // budget < physical_cap: budget binds; longest + 1 <= limit admits.
        assert!(check_eviction_limit(16383, Some(16384), 16768).is_ok());
        let e = check_eviction_limit(16384, Some(16384), 16768).unwrap_err();
        assert_eq!(e.status, 422);
        assert_eq!(e.required_max_seq, None);
        assert!(
            e.message.contains("limited to 16384 tokens"),
            "{}",
            e.message
        );
        assert!(e.message.contains("16768"), "{}", e.message);
        // physical_cap < budget (clamped): physical_cap binds.
        assert!(check_eviction_limit(99, Some(512), 100).is_ok());
        let e = check_eviction_limit(100, Some(512), 100).unwrap_err();
        assert_eq!(e.status, 422);
        assert!(e.message.contains("limited to 100 tokens"), "{}", e.message);
        assert!(e.to_reply("x")["error"].get("required_max_seq").is_none());
    }

    #[test]
    fn uncompacted_check() {
        assert!(check_uncompacted(Some(0)).is_ok());
        let e = check_uncompacted(Some(7)).unwrap_err();
        assert_eq!(e.status, 409);
        assert!(e.message.contains("compact_offset=7"));
        assert_eq!(check_uncompacted(None).unwrap_err().status, 500);
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

    #[test]
    fn precheck_refusals_in_priority_order() {
        let status = |r: Option<Value>| r.map(|v| v["error"]["status"].as_u64().unwrap());
        // Multi-slot mode wins over everything, even with a "model" present.
        let r = precheck(false, false, true, "s1").unwrap();
        assert_eq!(r["type"], "decided");
        assert_eq!(r["id"], "s1");
        assert_eq!(r["error"]["status"], 409);
        assert_eq!(
            r["error"]["message"],
            "decide unsupported in multi-slot mode"
        );
        assert_eq!(status(precheck(true, true, true, "x")), Some(409));
        // Active continuous-batch lanes.
        let r = precheck(true, true, false, "l1").unwrap();
        assert_eq!(r["error"]["status"], 409);
        assert!(r["error"]["message"].as_str().unwrap().contains("active"));
        assert_eq!(status(precheck(false, true, false, "x")), Some(409));
        // No model loaded.
        let r = precheck(false, false, false, "n1").unwrap();
        assert_eq!(r["id"], "n1");
        assert_eq!(r["error"]["status"], 503);
        assert_eq!(r["error"]["message"], "no model loaded");
        assert!(r["error"].get("required_max_seq").is_none());
        // Admitted.
        assert!(precheck(true, false, false, "ok").is_none());
    }

    #[test]
    fn single_token_id_strips_leading_bos() {
        // Llama/Mistral-style: encode prepends BOS (id 1).
        let with_bos = |s: &str| -> Vec<u32> {
            match s {
                "A" => vec![1, 65],
                "AB" => vec![1, 65, 66],
                _ => vec![1],
            }
        };
        assert_eq!(single_token_id(with_bos, Some(1), "A"), Some(65));
        assert_eq!(single_token_id(with_bos, Some(1), "AB"), None);
        // Only BOS left (empty text): zero tokens, not one.
        assert_eq!(single_token_id(with_bos, Some(1), ""), None);
        // Without knowing the BOS, the prepended id makes "A" two tokens.
        assert_eq!(single_token_id(with_bos, None, "A"), None);
        // No-BOS tokenizer (Qwen): plain length check, and a first id that
        // merely equals some unrelated bos_id is not stripped when bos=None.
        let plain = |s: &str| -> Vec<u32> {
            match s {
                "A" => vec![1],
                _ => vec![7, 8],
            }
        };
        assert_eq!(single_token_id(plain, None, "A"), Some(1));
        assert_eq!(single_token_id(plain, None, "AA"), None);
        // Only a *leading* BOS is stripped.
        let trailing = |_: &str| -> Vec<u32> { vec![65, 1] };
        assert_eq!(single_token_id(trailing, Some(1), "A"), None);
    }
}
