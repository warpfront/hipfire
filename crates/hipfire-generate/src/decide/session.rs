// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Session-mode decide: the chat conversation is the state. Every question
//! is rendered as `messages` + one user turn through the chat path's cached
//! Jinja render. The conversation cache is reused (extend / checkpoint
//! resume / cold), the model is snapshotted at the conversation end, and each
//! question is answered past it. The conversation cache is then left in place
//! for the next chat turn. Spec: `docs/specs/2026-09-28-jev-decide-design.md` §12.

use super::{
    check_eviction_limit, check_uncompacted, decided_reply, execute_decide, gather_labels, hook,
    ms, note_calibration, question_labels, rollback, DecideError, DecideOutcome, DecidePlan,
    RenderedQuestions,
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
    let cold_text = frame
        .render_messages(&msgs, tools, None)
        .map_err(render_err)?;
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

/// What bounds a session decide's longest render (see [`render_and_admit`]).
struct SessionLimits {
    max_seq: usize,
    eviction_budget: Option<usize>,
    physical_cap: usize,
}

/// The admitted renders: per-question label ids and tokens, and E.
struct SessionRenders {
    label_ids: Vec<Vec<u32>>,
    seqs: Vec<Vec<u32>>,
    conv_end: usize,
}

/// `required_max_seq` when the conversation alone does not fit `max_seq`,
/// estimated without rendering every question (each render re-renders and
/// tokenizes the whole conversation). R_0 is rendered; every other R_i is
/// R_0 with question 0's turn text swapped for question i's, so it is
/// estimated as `r0_len - text_lens[0] + text_lens[i]` (each text tokenized
/// on its own). A template or tokenizer merge at the turn boundary can move
/// that by a token or two, hence the slack: the value only sizes serve's
/// reload, and the retry re-checks exactly.
fn estimate_required_max_seq(probe_len: usize, r0_len: usize, text_lens: &[usize]) -> usize {
    const SLACK: usize = 16;
    let longest_text = text_lens.iter().copied().max().unwrap_or(0);
    let text0 = text_lens.first().copied().unwrap_or(0);
    let estimate = (r0_len + longest_text).saturating_sub(text0) + SLACK;
    estimate.max(probe_len) + 1
}

/// Render every question and admit the request, failing as early as the
/// checks allow (spec §12.5). A hostile oversized conversation must not
/// cost up to [`d::MAX_QUESTIONS`] whole-conversation renders under
/// exclusive admission before its 422, so the order is:
///
/// 1. the probe (conversation + a `.` user turn): over the eviction limit is
///    a 422 before any question renders. A question turn is never shorter
///    than the probe's `.` turn, so every R_i would fail too.
/// 2. R_0. If the probe alone is over `max_seq`: 422 with an estimated
///    `required_max_seq` ([`estimate_required_max_seq`]); no further render.
/// 3. each R_i as it is rendered: the eviction limit and Jev's per-question
///    / request limits on the suffixes past a provisional E (the end over
///    the probe and R_0 only). More renders can only shorten the common
///    prefix, so the final E is <= the provisional one and every suffix only
///    grows: an early refusal is one the final check would make too.
/// 4. after all: E over every render, the exact limits, and `max_seq`
///    against the longest R_i (the conversation fits, so every render was
///    bounded by `max_seq` + one question).
fn render_and_admit(
    tok: &Tokenizer,
    questions: &[d::Question],
    limits: &SessionLimits,
    mut render: impl FnMut(&str) -> Result<Vec<u32>, DecideError>,
) -> Result<SessionRenders, DecideError> {
    let turn_open = tok.special_token_id(TURN_OPEN);
    let probe = render(PROBE_USER_TEXT)?;
    check_eviction_limit(probe.len(), limits.eviction_budget, limits.physical_cap)?;
    let mut label_ids = Vec::with_capacity(questions.len());
    let mut texts = Vec::with_capacity(questions.len());
    for q in questions {
        let (labels, ids) = question_labels(tok, q)?;
        texts.push(d::build_question_text(q, &labels));
        label_ids.push(ids);
    }
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(questions.len());
    let mut provisional_end = 0;
    for (i, text) in texts.iter().enumerate() {
        let r = render(text)?;
        check_eviction_limit(r.len(), limits.eviction_budget, limits.physical_cap)?;
        if i == 0 {
            if probe.len() + 1 > limits.max_seq {
                let text_lens: Vec<usize> = texts.iter().map(|t| tok.encode(t).len()).collect();
                let required = estimate_required_max_seq(probe.len(), r.len(), &text_lens);
                return Err(DecideError {
                    status: 422,
                    message: format!(
                        "conversation needs {} tokens before any question (about {required} \
                         with the longest question), model max_seq is {}",
                        probe.len() + 1,
                        limits.max_seq
                    ),
                    required_max_seq: Some(required),
                });
            }
            provisional_end = d::conversation_end(std::slice::from_ref(&r), &probe, turn_open);
        }
        seqs.push(r);
        let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
        d::check_session_tokens(&questions[..=i], provisional_end, &lens)
            .map_err(|e| DecideError::new(422, e))?;
    }
    let conv_end = d::conversation_end(&seqs, &probe, turn_open);
    let lens: Vec<usize> = seqs.iter().map(Vec::len).collect();
    d::check_session_tokens(questions, conv_end, &lens).map_err(|e| DecideError::new(422, e))?;
    let longest = lens.iter().copied().max().unwrap_or(0);
    check_eviction_limit(longest, limits.eviction_budget, limits.physical_cap)?;
    if longest + 1 > limits.max_seq {
        return Err(DecideError {
            status: 422,
            message: format!(
                "prompt needs {} tokens, model max_seq is {}",
                longest + 1,
                limits.max_seq
            ),
            required_max_seq: Some(longest + 1),
        });
    }
    Ok(SessionRenders {
        label_ids,
        seqs,
        conv_end,
    })
}

/// Everything [`execute_session`] needs. Computing it leaves KV, recurrent
/// state, `seq_pos` and `conversation_tokens` untouched; the render may
/// refresh `asst_turn_cache`'s LRU order, as a chat render does.
pub(crate) struct SessionPlan {
    questions: Vec<d::Question>,
    calibration: d::Calibration,
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
    let limits = SessionLimits {
        max_seq: m.max_seq,
        eviction_budget: m.eviction.as_ref().map(|e| e.budget()),
        physical_cap: m.physical_cap,
    };
    let SessionRenders {
        label_ids,
        seqs,
        conv_end,
    } = render_and_admit(tok, &parsed.questions, &limits, |text| {
        render_session_prompt(tok, template, cache, &parsed.messages, tools, text)
    })?;
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
        calibration: parsed.calibration,
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
            // As v1 after its rollback: attest the positional-rewind
            // precondition (the rollback zeroes compact_offset, incl. CASK).
            check_uncompacted(carrier.decide_kv_compact_offset(m))?;
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
/// ends with the attested rollback, keeping `asst_turn_cache` (global
/// constraint: a cold start keeps it).
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
            // Global constraint: a cold start keeps `asst_turn_cache`. This
            // arm is reached after a cold start too (speculator loaded, then
            // a non-Extend end), so use the cache-preserving reset here as
            // well, not the plain attested rollback (which clears it).
            cold_reset_keep_turn_cache(m, gpu)
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
            calibration,
            carrier,
            label_ids,
            seqs,
            ..
        } = plan;
        let v1 = DecidePlan {
            parsed: d::DecideRequest {
                state_text: String::new(),
                questions,
                calibration,
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
    let fin =
        run.and_then(|from| finish_session(&plan, m, gpu, read_only, from, &snaps, &mut reset));
    snaps.free(gpu);
    if let Err(err) = fin {
        // Fail closed like every chat abort path (ar.rs:3317-3324): a
        // retained cache must never carry uncommitted state.
        let _ = rollback(m, gpu);
        return (Err(err), true);
    }
    let lens: Vec<usize> = plan.seqs.iter().map(Vec::len).collect();
    let answers = d::assemble_answers(&plan.questions, &per_q, &plan.calibration);
    note_calibration(&mut timing, &plan.calibration);
    timing["total_ms"] = json!(ms(t_total));
    (
        Ok(DecideOutcome {
            answers,
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
        let r =
            render_session_prompt(&t, BARE, &mut cache, &history(), None, "Question: Q?").unwrap();
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
        let r =
            render_session_prompt(&t, BARE, &mut cache, &history(), None, "Question: Q?").unwrap();
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

    /// `n` choice questions of varying length over `history()` with a
    /// `conv_bytes`-byte user turn prepended.
    fn session_req(n: usize, conv_bytes: usize, long_q: Option<usize>) -> d::SessionRequest {
        let mut qs = serde_json::Map::new();
        for i in 0..n {
            let pad = if long_q == Some(i) {
                "x".repeat(33_000)
            } else {
                "y".repeat(i % 7)
            };
            qs.insert(
                format!("q{i:03}"),
                json!({"type": "choice", "instructions": format!("Q{i}{pad}?"),
                       "criteria": {"a": "first", "b": "second"}}),
            );
        }
        d::parse_session_request(&json!({
            "messages": [{"role": "user", "content": "z".repeat(conv_bytes)},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "yo"}],
            "questions": qs,
        }))
        .unwrap()
    }

    /// [`render_and_admit`] over the BARE template, counting renders.
    fn admit(
        req: &d::SessionRequest,
        limits: &SessionLimits,
    ) -> (Result<SessionRenders, DecideError>, usize) {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let mut renders = 0;
        let r = render_and_admit(&t, &req.questions, limits, |text| {
            renders += 1;
            render_session_prompt(&t, BARE, &mut cache, &req.messages, None, text)
        });
        (r, renders)
    }

    fn no_eviction(max_seq: usize) -> SessionLimits {
        SessionLimits {
            max_seq,
            eviction_budget: None,
            physical_cap: max_seq,
        }
    }

    #[test]
    fn admits_and_matches_the_full_render() {
        let req = session_req(5, 10, None);
        let (r, renders) = admit(&req, &no_eviction(1 << 20));
        let r = r.unwrap();
        assert_eq!(renders, 6, "probe + one render per question");
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let probe = render_session_prompt(&t, BARE, &mut cache, &req.messages, None, ".").unwrap();
        assert_eq!(
            r.conv_end,
            d::conversation_end(&r.seqs, &probe, t.special_token_id(TURN_OPEN))
        );
        assert_eq!(r.seqs.len(), 5);
        assert_eq!(r.label_ids.len(), 5);
    }

    /// A conversation over max_seq is refused after the probe and R_0 only,
    /// however many questions there are (review: no 128-question render
    /// stall under exclusive admission), and the estimated
    /// required_max_seq admits the request on the retry.
    #[test]
    fn oversized_conversation_is_refused_before_rendering_the_questions() {
        let req = session_req(d::MAX_QUESTIONS, 3000, None);
        let (r, renders) = admit(&req, &no_eviction(512));
        let e = r.err().expect("over max_seq");
        assert_eq!(renders, 2, "probe and R_0 only");
        assert_eq!(e.status, 422);
        let required = e.required_max_seq.expect("required_max_seq");
        let (retry, _) = admit(&req, &no_eviction(required));
        let longest = retry
            .expect("the estimate admits the retry")
            .seqs
            .iter()
            .map(Vec::len)
            .max();
        assert!(
            required <= longest.unwrap() + 1 + 17,
            "estimate {required} vs {longest:?}"
        );
    }

    #[test]
    fn probe_over_the_eviction_limit_renders_no_question() {
        let req = session_req(d::MAX_QUESTIONS, 3000, None);
        let limits = SessionLimits {
            max_seq: 1 << 20,
            eviction_budget: Some(1000),
            physical_cap: 1 << 20,
        };
        let (r, renders) = admit(&req, &limits);
        let e = r.err().expect("over the eviction limit");
        assert_eq!(renders, 1, "probe only");
        assert_eq!(e.status, 422);
        assert_eq!(e.required_max_seq, None);
        assert!(e.message.contains("KV eviction"), "{}", e.message);
    }

    /// Jev's per-question limit fails on the offending question without
    /// rendering the rest, as v1 does.
    #[test]
    fn over_long_question_fails_where_it_is_rendered() {
        let req = session_req(d::MAX_QUESTIONS, 10, Some(3));
        let (r, renders) = admit(&req, &no_eviction(1 << 20));
        let e = r.err().expect("over the per-question limit");
        assert_eq!(renders, 5, "probe + q0..=q3");
        assert_eq!(e.status, 422);
        assert!(e.message.contains("q003"), "{}", e.message);
        assert!(e.message.contains("per-question limit"), "{}", e.message);
    }

    #[test]
    fn required_max_seq_estimate_swaps_question_zero_for_the_longest() {
        assert_eq!(
            estimate_required_max_seq(90, 100, &[10, 30, 20]),
            100 + 20 + 16 + 1
        );
        // Never below the probe.
        assert_eq!(estimate_required_max_seq(500, 100, &[10]), 501);
    }

    #[test]
    fn open_think_template_is_400_and_render_error_is_422() {
        let t = tok();
        let mut cache = AsstTurnCache::new_from_env();
        let e =
            render_session_prompt(&t, ALWAYS_THINK, &mut cache, &history(), None, "Q").unwrap_err();
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
