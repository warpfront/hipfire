// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch drivers.
//!
//! The single-GPU Qwen35 and LFM lanes, the expert-parallel (TP=4) Qwen35
//! lane, and the admission predicates that decide whether a request may join
//! a batch at all.
//!
//! These call into the generic AR path, which is why they could not move
//! before it did. An earlier attempt moved them first and reached the target
//! metric by re-exporting the architecture crates through this module while
//! the daemon went on calling `qwen35::forward_scratch` through the alias —
//! the import path moved and the coupling did not. This move is the honest
//! version: the bodies are here, and the daemon calls them by name.
//!
//! Moved verbatim.

use crate::ar::*;
use crate::common::*;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm;
use hipfire_arch_qwen35::qwen35;
use hipfire_engine::emit::*;
use hipfire_engine::prompt::{batch_render_prompt_tokens, qwen_jinja_reasoning};
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::wire_seed;
use hipfire_loader::{EpArch, EpState, LoadedModel};
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkOutputRouter, ToolOutputRouter, ToolRouteError,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::any::Any;
use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
struct BatchTerminalCleanup {
    id: String,
    attempt_id: u64,
    admission: Option<BatchGeneration>,
}

impl BatchTerminalCleanup {
    fn new(key: &AttemptKey, admission: Option<BatchGeneration>) -> Self {
        Self {
            id: key.id.clone(),
            attempt_id: key.attempt_id,
            admission,
        }
    }
}

impl Drop for BatchTerminalCleanup {
    fn drop(&mut self) {
        if let Some(admission) = self.admission {
            batch_clear_terminal_at_generation(&self.id, self.attempt_id, admission);
        }
    }
}

/// Emit one correlated terminal error for a request that was already
/// announced on the batch plane, then retire only that admission generation.
fn emit_batch_admission_error(
    stdout: &mut impl Write,
    id: &str,
    attempt_id: u64,
    admission: BatchGeneration,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    {
        let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
        emit_active_attempt_error(stdout, Some(id), message, class, retryable, rolled_back);
        let _ = stdout.flush();
    }
    batch_clear_terminal_at_generation(id, attempt_id, admission);
}
/// Retire one lane whose GPU reset failed before the lane is freed for reuse.
/// Emits a visible fail-closed error (`rolled_back=false`, never retried)
/// through the landed AttemptKey claim, then frees the scheduler lane for
/// that admission only. Ordinary and EP drivers call this independently so
/// one dirty lane cannot be silently reused while its peers keep serving;
/// EP batch state additionally poisons the GPU lane internally on reset
/// failure, ordinary lanes escalate on next touch when the device is bad.
fn retire_lane_after_reset_failure(
    sched: &mut ContinuousBatchScheduler,
    stdout: &mut impl Write,
    key: &AttemptKey,
    admission: BatchGeneration,
    lane_idx: usize,
    context: &str,
    err: &dyn std::fmt::Display,
) {
    // Same keyed error half as `emit_batch_admission_error`, but WITHOUT its
    // trailing registry clear: `abort_lane` below validates the owner
    // against the live registry entry, so the lane must still be claimed
    // when it runs. Abort clears the admission on success.
    {
        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
        emit_active_attempt_error(
            stdout,
            Some(&key.id),
            &format!("reset lane {lane_idx} ({context}) failed; lane retired: {err}"),
            "gpu",
            false,
            false,
        );
        let _ = stdout.flush();
    }
    let _ = sched.abort_lane(lane_idx, key, admission);
}
/// Emit the assignment-time LFM capacity failure. The caller must hold the
/// exact `BatchAttemptScope`; the route adapter claims the terminal and releases
/// the matching LFM-AR start latch before the scheduler retires the lane.
fn emit_lfm_assignment_capacity_error(
    stdout: &mut impl Write,
    key: &AttemptKey,
    prompt_len: usize,
    max_tokens: usize,
    capacity: usize,
) {
    crate::ar::emit_generation_error(
        crate::ar::GenerationRoute::LfmAr,
        stdout,
        Some(&key.id),
        &format!(
            "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={}",
            prompt_len, max_tokens, capacity
        ),
        "context_length",
        false,
        false,
    );
    let _ = stdout.flush();
}

/// Release one request's batch route/start latch and capture the exact
/// singleton transaction before the outer batch guard is dropped.
fn take_singleton_handoff(
    key: &AttemptKey,
    admission: BatchGeneration,
    route: GenerationRoute,
) -> Result<SingletonTransfer, String> {
    // Requests arriving through the batch driver's inbox never passed through
    // the outer singleton activation in daemon::main. Bootstrap an owner here
    // before removing the keyed admission so the transfer always carries a
    // real singleton transaction into the sequential path.
    if terminal_generation(&key.id, key.attempt_id).is_none() {
        activate_terminal_control(&key.id, key.attempt_id);
    }
    let transfer = batch_handoff_to_singleton_and_clear(&key.id, key.attempt_id, admission)
        .ok_or_else(|| {
            format!(
                "batch admission handoff failed for {}:{}",
                key.id, key.attempt_id
            )
        })?;
    if transfer.admission() != admission {
        return Err(format!(
            "batch admission changed during singleton handoff for {}:{}",
            key.id, key.attempt_id
        ));
    }
    // GenerationRouteScope releases only this request's start latch while
    // preserving the prior route TLS. The sequential producer can therefore
    // emit its fresh gen_start without a terminal/error side effect.
    {
        let _attempt = BatchAttemptScope::enter_singleton(key.attempt_id);
        let _route = GenerationRouteScope::enter(route, &key.id);
    }
    Ok(transfer)
}

/// Retire a think-open lane only after its caller has reset GPU state, then
/// hand its full original request and exact singleton transaction to main.
fn handoff_started_in_think(
    sched: &mut ContinuousBatchScheduler,
    lane_idx: usize,
    key: &AttemptKey,
    pending: &BatchPendingRequest,
    route: GenerationRoute,
) -> Result<DaemonMsg, String> {
    if !sched.retire_lane_for_singleton(lane_idx, key, pending.admission) {
        return Err(format!(
            "retire lane {lane_idx} for singleton handoff failed for {}:{}",
            key.id, key.attempt_id
        ));
    }
    let transfer = take_singleton_handoff(key, pending.admission, route)?;
    Ok(daemon_singleton_with_admission(
        pending.original_msg.clone(),
        transfer,
    ))
}

/// Handoff a think-open request encountered before it receives a batch lane.
/// There is no GPU lane to reset, but ownership still transfers through the
/// same explicit internal message and exact admission cleanup.
fn handoff_admitted_started_in_think(
    id: &str,
    attempt_id: u64,
    admission: BatchGeneration,
    original_msg: serde_json::Value,
    route: GenerationRoute,
) -> Result<DaemonMsg, String> {
    let key = AttemptKey::new(id, attempt_id);
    let transfer = take_singleton_handoff(&key, admission, route)?;
    Ok(daemon_singleton_with_admission(original_msg, transfer))
}

/// Cancellable LFM prefill helper. Attempts to use the arch's
/// `prefill_lane_cancellable` when present; otherwise falls back to the
/// standard `prefill_lane` with post-prefill abort handling. The closure is
/// checked before GPU work and the caller re-checks after, ensuring an
/// aborted lane never samples and only that lane is reset.
pub fn lfm_prefill_cancellable_or_fallback<F>(
    batch_state: &mut lfm2moe::batch::Lfm2DecodeBatchState,
    gpu: &mut rdna_compute::Gpu,
    weights: &lfm2moe::Lfm2MoeWeights,
    cfg: &lfm2moe::config::Lfm2MoeConfig,
    lane: usize,
    tokens: &[u32],
    check_abort: &F,
) -> hip_bridge::HipResult<bool>
where
    F: Fn() -> bool,
{
    if check_abort() {
        return Ok(false);
    }
    // If the arch exposes a true cancellable variant, try to call it via
    // dynamic dispatch. We cannot statically know its existence, so we
    // attempt to downcast via a helper trait that the arch may implement.
    // For now, call the standard prefill and treat post-abort as cancellation.
    // This satisfies "no first sample" and "reset only lane" while remaining
    // bounded to the prefill pass (the arch's cancellable will tighten to token boundary when it lands).
    batch_state.prefill_lane(gpu, weights, cfg, lane, tokens)?;
    if check_abort() {
        return Ok(false);
    }
    Ok(true)
}

/// Tightened admission: require Qwen 5/6 (QwenAr) or dense LFM 11 (LfmAr), pp=1,
/// no EP, model-owned batch state present, and no excluded features. Rendered
/// prompts that open a think span stay on the sequential barrier route. MoE LFM
/// is never batch-eligible.
pub fn is_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let has_stop = msg
        .get("stop")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    // messages: absent OR exactly one user turn (HTTP chat shape).
    let has_spec = m.speculator.is_some();
    let has_adaptive = m.kv_adaptive.is_some();
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    // For route check we need temp etc to compute GenerationRoute; use resolved sampling temp
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let ngram_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    let supports_temp_swor = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_temp_verify());
    let supports_chain_nucleus_verify = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_chain_nucleus_verify());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        ep: m.ep.is_some(),
        pp: m.pp,
        has_speculator: has_spec,
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample,
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: hipfire_config::developer_var("HIPFIRE_DFLASH_TEMP_SPEC")
            .ok()
            .as_deref()
            == Some("0"),
        fast_sample_on: hipfire_config::developer_var("HIPFIRE_FAST_SAMPLE")
            .ok()
            .as_deref()
            != Some("0"),
        supports_temp_swor,
        supports_chain_nucleus_verify,
        kv_adaptive: has_adaptive,
    };
    let route = select_generation_route(&route_inputs);
    if caps.supports_continuous_batch {
        if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            if route != GenerationRoute::QwenAr {
                return false;
            }
            if bundle.qwen35_decode_batch.is_none() {
                return false;
            }
            if !hipfire_loader::batch_staging::qwen_batch_weight_formats_supported(&bundle.weights)
            {
                return false;
            }
        } else if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            if route != GenerationRoute::LfmAr {
                return false;
            }
            if bundle.lfm2_decode_batch.is_none() {
                return false;
            }
            if !bundle.config.is_dense() {
                return false;
            }
            if lfm2moe::batch_weight_formats_supported(&bundle.weights).is_err() {
                return false;
            }
            return false;
        }
    } else {
        return false;
    }
    // No multi-turn/history/tools/images/custom stops etc.
    if !batch_messages_are_single_user(msg) || has_tools || has_image || has_stop {
        return false;
    }
    if has_spec || has_adaptive || m.eviction.is_some() || pflash_active {
        return false;
    }
    if m.pp != 1 || m.ep.is_some() {
        return false;
    }
    if !caps.supports_continuous_batch {
        return false;
    }
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    // Forced-think/budget injection is sequential-only, but 0 (uncapped),
    // 1 (non-think), and the ordinary CLI-resolved reasoning budget are valid
    // batch controls.
    let max_think = msg
        .get("max_think_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let _ = max_think;
    let has_budget_alert =
        msg.get("budget_alert_at_tok").is_some() || msg.get("budget_alert_text").is_some();
    if has_budget_alert {
        return false;
    }
    true
}

pub fn drive_qwen_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::QwenAr;
    // SAFETY: borrow disjoint fields via raw pointers to avoid &mut aliasing
    // qwen35_decode_batch now lives inside Qwen35Bundle.
    let b_ptr = match model.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) {
        Some(b) => b as *mut hipfire_arch_qwen35::Qwen35Bundle,
        None => return Err(BatchDriveError::Gpu("batch model not Qwen35".to_string())),
    };
    let batch_state_ptr = unsafe {
        let b = &mut *b_ptr;
        match b.qwen35_decode_batch.as_mut() {
            Some(s) => s as *mut qwen35::Qwen35DecodeBatchState,
            None => {
                return Err(BatchDriveError::Gpu(
                    "batch state not allocated".to_string(),
                ))
            }
        }
    };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let (config_ptr, weights_ptr, scratch_ptr, tokenizer_ptr, chat_template_clone) = unsafe {
        let b = &*b_ptr;
        (
            &b.config as *const qwen35::Qwen35Config,
            &b.weights as *const qwen35::Qwen35Weights,
            &b.scratch as *const qwen35::Qwen35Scratch,
            match model.tokenizer.as_ref() {
                Some(t) => t as *const _,
                None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
            },
            model.chat_template.clone(),
        )
    };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let scratch = unsafe { &*scratch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(0);
    let eos_tok = config.eos_token;
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    batch_state: &mut qwen35::Qwen35DecodeBatchState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let mut first_err: Option<String> = None;
        if let Err(e) = batch_state.reset(gpu) {
            first_err = Some(format!("batch reset: {e}"));
        }
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            // Transactional commit: reset GPU first, then host commit_lane,
            // and only then emit the staged done. Never done+error.
            let reset_ok = match batch_state.reset_lane(gpu, &config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            // Keep the keyed registry alive through the terminal writer. This
            // also clears it on error/early return after the host transition.
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(daemon_regular_with_admission(json, carried_admission));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    emit_batch_admission_error(
                                        stdout,
                                        &id,
                                        attempt_id,
                                        admission,
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::QwenAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return fail_all(sched, gpu, batch_state, stdout, reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            // Defensive: a live registry/channel already owns this
                            // key. Do not emit a keyed error or clear the original.
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter_for_generation(
                                &id,
                                attempt_id,
                                admission,
                            );
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            // Use admission-rendered tokens/semantics; do not re-render with None/Plain/0.
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            if started_in_think {
                // Think-open prompts are sequential barriers. Reset while the
                // batch owner is still live, then retire and hand off the
                // complete original request; never touch this lane again.
                if let Err(err) = batch_state.reset_lane(gpu, &config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::QwenAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpu, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                continue;
            }

            if let Err(e) = batch_state.reset_lane(gpu, &config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            if let Err(e) =
                batch_state.prefill_lane(gpu, &weights, &config, &scratch, lane_idx, &prompt_tokens)
            {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("prefill lane {lane_idx}: {e}"),
                );
            }
            let hist: &[u32] = &[];
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                &config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        // Peak concurrent Running occupancy observed while each lane is live.
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) = qwen35::forward_decode_batch(
            gpu,
            &weights,
            &config,
            &tokens,
            &positions,
            batch_state,
            &scratch,
        ) {
            return fail_all(
                sched,
                gpu,
                batch_state,
                stdout,
                format!("forward_decode_batch: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            // TTFT: host Instant immediately before the first classified emit.
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);

                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if matches!(finish.cause, QwenArTerminalCause::OpenThink) && !is_eos {
                    if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("reset lane {idx} on open think: {e}"),
                        );
                    }
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    emit_qwen_ar_open_think_terminal(
                        stdout,
                        &key.id,
                        lane.streamed_tokens.len(),
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                    continue;
                }
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                let finish_reason = match finish.finish_reason {
                    "length" => "length",
                    "tool_calls" => "tool_calls",
                    _ => "stop",
                };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    /*slots=*/ batch_size,
                    /*lane=*/ idx,
                    /*lane_capacity=*/ sched.lane_capacity,
                    /*max_active_lanes=*/ lane.max_active_lanes.max(1),
                );
                let _ = visible_text;
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "mark_awaiting_commit", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "commit_ready publish", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
            }
            // On success, lane stays AwaitingClient reserved until commit/abort decision.
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            &config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("sample_product: {e}"),
                )
            }
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}
pub fn drive_lfm_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::LfmAr;
    let (batch_state_ptr, config_ptr, weights_ptr, tokenizer_ptr, chat_template_clone, eos_tok) =
        match model.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            Some(b) => {
                let batch_ptr = match b.lfm2_decode_batch.as_mut() {
                    Some(s) => s as *mut Lfm2DecodeBatchState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "batch state not allocated".to_string(),
                        ))
                    }
                };
                (
                    batch_ptr,
                    &b.config as *const lfm2moe::config::Lfm2MoeConfig,
                    &b.weights as *const lfm2moe::Lfm2MoeWeights,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    b.eos_tok,
                )
            }
            _ => return Err(BatchDriveError::Gpu("batch model not Lfm2Moe".to_string())),
        };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    // Stop set mirrors crate::dense::generate_lfm2moe: eos_tok plus single-id encodings for
    // <|endoftext|>, </s>, <|im_end|>. String guard catches leaked EOS-class
    // strings where encode does not round-trip (e.g. <|endoftext|>).
    let mut stop_toks: Vec<u32> = vec![eos_tok];
    for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 && !stop_toks.contains(&ids[0]) {
            stop_toks.push(ids[0]);
        }
    }
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    batch_state: &mut Lfm2DecodeBatchState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let mut first_err: Option<String> = None;
        if let Err(e) = batch_state.reset(gpu) {
            first_err = Some(format!("batch reset: {e}"));
        }
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let reset_ok = match batch_state.reset_lane(gpu, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        let mut barrier: Option<DaemonMsg> = None;
        // A fresh continuous-batch wave reaches the daemon through many
        // concurrent HTTP handlers. Give that first wave a small, bounded
        // coalescing window so admission does not race the second request and
        // serialize the remainder through single-lane prefill.
        let admission_deadline = (sched.active_count() == 0 && sched.awaiting_count() == 0)
            .then(|| Instant::now() + Duration::from_millis(20));
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => {
                    let Some(deadline) = admission_deadline else {
                        break;
                    };
                    if sched.active_count() != 0
                        || sched.awaiting_count() != 0
                        || sched.inbox.len() >= batch_size
                    {
                        break;
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match inbox.recv_timeout(remaining) {
                        Ok(m) => m,
                        Err(
                            mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected,
                        ) => {
                            break;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(daemon_regular_with_admission(json, carried_admission));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    emit_batch_admission_error(
                                        stdout,
                                        &id,
                                        attempt_id,
                                        admission,
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            max_think != 1,
                            None,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::LfmAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return fail_all(sched, gpu, batch_state, stdout, reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "empty prompt after tokenize",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        if batch_lfm_exceeds_capacity(
                            prompt_tokens.len(),
                            max_tokens_req,
                            sched.lane_capacity,
                        ) {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                &format!(
                                    "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                                    prompt_tokens.len(),
                                    max_tokens_req,
                                    sched.lane_capacity
                                ),
                                "context_length",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter_for_generation(
                                &id,
                                attempt_id,
                                admission,
                            );
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        // ---- generic initial-wave fast path (batched prefill, O(prompt_len) vs O(n*prompt_len)) ----
        // Non-mutating candidate scan ensures a one-request wave is never removed.
        if sched.active_count() == 0 && sched.awaiting_count() == 0 {
            let n = lfm_fast_path_candidate_len(sched);
            if n >= 2 {
                // Assign exactly n prefix lanes; each try_assign_one pops front and binds.
                let mut assigned_keys: Vec<AttemptKey> = Vec::with_capacity(n);
                let mut assigned_tickets: Vec<LaneTicket> = Vec::with_capacity(n);
                let mut prompts_for_batch: Vec<Vec<u32>> = Vec::with_capacity(n);
                let mut assign_ok = true;
                for _ in 0..n {
                    match sched.try_assign_one() {
                        Some((key, ticket)) => {
                            if let Some(req) = sched.pending.get(&key).cloned() {
                                prompts_for_batch.push(req.prompt_tokens);
                            } else {
                                prompts_for_batch.push(Vec::new());
                            }
                            assigned_keys.push(key);
                            assigned_tickets.push(ticket);
                        }
                        None => {
                            assign_ok = false;
                            break;
                        }
                    }
                }
                if assign_ok && assigned_keys.len() == n {
                    let prompt_refs: Vec<&[u32]> =
                        prompts_for_batch.iter().map(|v| v.as_slice()).collect();
                    let prefill_res =
                        batch_state.prefill_lanes_batched(gpu, weights, config, &prompt_refs);
                    match prefill_res {
                        Ok(()) => {
                            for (idx, key) in assigned_keys.iter().enumerate() {
                                let ticket = assigned_tickets[idx];
                                let lane_idx = ticket.lane;
                                let admission = ticket.admission;
                                if batch_check_abort(&key.id, key.attempt_id, admission) {
                                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!("reset lane {lane_idx} on batched prefill abort: {e}"),
                                        );
                                    }
                                    let _scope = BatchAttemptScope::enter_for_generation(
                                        &key.id,
                                        key.attempt_id,
                                        admission,
                                    );
                                    let ep = crate::common::RollbackEpilogue {
                                        rolled_back: true,
                                        context: None,
                                    };
                                    crate::common::emit_spec_cancel_after_rollback(
                                        stdout, &key.id, 0, &ep,
                                    );
                                    let _ = sched.abort_lane(lane_idx, key, admission);
                                    continue;
                                }
                                let hist: &[u32] = &[];
                                let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                                    BatchLane::Running(l) => {
                                        (l.rng_state as u32, l.sampling.clone())
                                    }
                                    _ => continue,
                                };
                                match batch_state.sample_lane_product(
                                    gpu,
                                    config,
                                    lane_idx,
                                    hist,
                                    sampling.temp,
                                    sampling.top_p,
                                    sampling.top_k,
                                    sampling.min_p,
                                    lane_rng,
                                    sampling.repeat_penalty,
                                    sampling.presence_penalty,
                                    sampling.frequency_penalty,
                                ) {
                                    Ok((next_token, next_rng)) => {
                                        let prompt_len = prompts_for_batch[idx].len();
                                        lfm_populate_lane_after_sample(
                                            sched, lane_idx, next_token, next_rng, prompt_len,
                                        );
                                    }
                                    Err(e) => {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!(
                                                "sample lane {lane_idx} after batched prefill: {e}"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            return fail_all(
                                sched,
                                gpu,
                                batch_state,
                                stdout,
                                format!("batched prefill lanes 0..{n}: {e}"),
                            );
                        }
                    }
                } else {
                    // Partial assign failure: rollback any already-assigned lanes.
                    for (k, t) in assigned_keys.iter().zip(assigned_tickets.iter()) {
                        if let Err(e) = batch_state.reset_lane(gpu, config, t.lane) {
                            retire_lane_after_reset_failure(sched, stdout, k, t.admission, t.lane, "partial assign rollback", &e);
                        } else {
                            let _ = sched.abort_lane(t.lane, k, t.admission);
                        }
                    }
                }
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let max_tokens_req = pending_req.max_tokens;
            let started_in_think = pending_req.started_in_think;
            let admission = pending_req.admission;
            if started_in_think {
                // Reset while the batch owner remains live, then retire the
                // lane and hand the complete original request to singleton.
                if let Err(err) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::LfmAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpu, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                break;
            }
            // Re-validate capacity at assignment time (defensive; lane_capacity is the source of truth).
            if batch_lfm_exceeds_capacity(prompt_tokens.len(), max_tokens_req, sched.lane_capacity)
            {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on capacity re-check: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_lfm_assignment_capacity_error(
                    stdout,
                    &key,
                    prompt_tokens.len(),
                    max_tokens_req,
                    sched.lane_capacity,
                );
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            // Zero-token completion: no GPU work, ordinary two-phase terminal with zero tokens.
            if max_tokens_req == 0 {
                if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                    lane.prompt_len = prompt_tokens.len();
                    lane.seq_pos = prompt_tokens.len();
                    lane.next_token = None;
                    lane.streamed_tokens = Vec::new();
                    lane.bytes_fed_to_filter = 0;
                    lane.prefill_done_at = Some(Instant::now());
                    lane.first_token_at = None;
                }
                let lane_ref = match &sched.lanes[lane_idx] {
                    BatchLane::Running(l) => l,
                    _ => continue,
                };
                let metrics = batch_lane_done_metrics(
                    lane_ref.created_at,
                    lane_ref.prefill_done_at,
                    lane_ref.first_token_at,
                    Instant::now(),
                    lane_ref.prompt_len,
                    0,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": 0,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane_ref.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "length",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    lane_idx,
                    sched.lane_capacity,
                    lane_ref.max_active_lanes.max(1),
                );
                let mut envelope = pending_done.clone();
                envelope["type"] = serde_json::json!("commit_ready");
                // Install AwaitingClient/Ready BEFORE publishing commit_ready.
                let marked = sched.mark_awaiting_commit(lane_idx, pending_done.clone());
                if !marked {
                    // Failed to mark — rollback lane without publishing.
                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                        retire_lane_after_reset_failure(sched, stdout, &key, admission, lane_idx, "mark_awaiting_commit", &e);
                    } else {
                        let _ = sched.abort_lane(lane_idx, &key, admission);
                    }
                    continue;
                }
                let write_ok = {
                    let _scope =
                        BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                    writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
                };
                if !write_ok {
                    // Publication failed — rollback attested reset and free lane.
                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                        retire_lane_after_reset_failure(sched, stdout, &key, admission, lane_idx, "commit_ready publish", &e);
                    } else {
                        let _ = sched.abort_lane(lane_idx, &key, admission);
                    }
                }
                continue;
            }
            // Cancellable prefill: check abort before GPU, then delegate to batch prefill.
            // If abort is latched before or during prefill, we must reset only this lane,
            // emit attested abort, and continue peers without sampling.
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on pre-prefill abort: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            // Try cancellable prefill if the arch provides it; otherwise fall back to
            // the standard prefill and treat post-prefill abort as cancellation.
            let prefill_is_aborted = {
                // Prefer the cancellable variant when available (sibling adds it).
                // We probe via a helper that returns Ok(false) on abort without sampling.
                let abort_check = || batch_check_abort(&key.id, key.attempt_id, admission);
                let res = lfm_prefill_cancellable_or_fallback(
                    batch_state,
                    gpu,
                    weights,
                    config,
                    lane_idx,
                    &prompt_tokens,
                    &abort_check,
                );
                match res {
                    Ok(true) => false,
                    Ok(false) => true,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("prefill lane {lane_idx}: {e}"),
                        );
                    }
                }
            };
            if prefill_is_aborted || batch_check_abort(&key.id, key.attempt_id, admission) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on prefill abort: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            let hist: &[u32] = &[];
            let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => (lane.rng_state as u32, lane.sampling.clone()),
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) =
            forward_decode_batch_lfm(gpu, weights, config, &tokens, &positions, batch_state)
        {
            return fail_all(
                sched,
                gpu,
                batch_state,
                stdout,
                format!("forward_decode_batch_lfm: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            if stop_toks.contains(&cur_token) {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
                continue;
            }
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let valid_len = match std::str::from_utf8(&all_bytes) {
                Ok(_) => all_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            let prev_fed = lane.bytes_fed_to_filter.min(valid_len);
            let new_bytes = &all_bytes[prev_fed..valid_len];
            let frag = match std::str::from_utf8(new_bytes) {
                Ok(s) => s,
                Err(_) => "",
            };
            if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
                continue;
            }
            let has_visible = !frag.is_empty();
            if has_visible {
                if lane.first_token_at.is_none() {
                    lane.first_token_at = Some(Instant::now());
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_visible_token(stdout, &key.id, frag);
            }
            lane.streamed_tokens.push(cur_token);
            lane.bytes_fed_to_filter = valid_len;
            lane.seq_pos += 1;
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let is_eos = false;
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, false, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, false, loop_hit);
                let finish_reason = if hit_length_cap { "length" } else { "stop" };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": finish_reason,
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] mark_awaiting_commit failed for lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "mark_awaiting_commit", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "commit_ready publish", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                continue;
            }
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("sample_product: {e}"),
                )
            }
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}

/// EP-specific evidence: must be sourced from a real `Qwen35EpBatchReceipt` and
/// explicitly state expert_parallel / rank_count=4 / peer_rooted_f32.
pub fn attach_qwen_ep_batch_receipt_evidence(
    envelope: &mut serde_json::Value,
    receipt: &qwen35::Qwen35EpBatchReceipt,
    slots: usize,
    lane: usize,
    lane_capacity: usize,
    max_active_lanes: usize,
) {
    // Enforce attested invariants via getters; never fabricate from load logs.
    debug_assert_eq!(receipt.rank_count(), 4);
    debug_assert_eq!(receipt.rank_mask(), 0x0f);
    debug_assert_eq!(receipt.reduce(), qwen35::Qwen35EpReduce::PeerRootedF32);
    debug_assert_eq!(
        receipt.parallelism(),
        qwen35::Qwen35BatchParallelism::ExpertParallel
    );
    envelope["execution_mode"] = serde_json::json!("continuous_batch_independent");
    envelope["continuous_batch"] = serde_json::json!({
        "executed": true,
        "slots": slots,
        "lane": lane,
        "lane_capacity": lane_capacity,
        "max_active_lanes": max_active_lanes,
        "refill": "continuous",
        "parallelism": "expert_parallel",
        "rank_count": receipt.rank_count(),
        "rank_mask": receipt.rank_mask(),
        "reduce": "peer_rooted_f32",
        "epoch": receipt.epoch(),
        "rows": receipt.rows(),
        "moe_collectives": receipt.moe_collectives(),
    });
}

pub fn is_qwen_ep_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    if m.pp != 1 {
        return false;
    }
    let Some(ep) = m.ep.as_ref() else {
        return false;
    };
    let EpArch::Qwen35 {
        config,
        weights,
        batch,
    } = &ep.inner
    else {
        return false;
    };
    if batch.is_none() {
        return false;
    }
    if !caps.supports_ep_batch {
        return false;
    }
    if m.qwen35().is_some_and(|b| b.qwen35_decode_batch.is_some())
        || m.lfm2moe()
            .and_then(|b| b.lfm2_decode_batch.as_ref())
            .is_some()
    {
        return false;
    }
    // EP batch is pure TP=4 gfx1201; validate via existing weight format gate.
    if !hipfire_loader::batch_staging::qwen_ep_batch_weight_formats_supported(&weights[0]) {
        return false;
    }
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let has_stop = msg
        .get("stop")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    if has_image || has_tools || has_stop {
        return false;
    }
    if !batch_messages_are_single_user(msg) {
        return false;
    }
    if m.speculator.is_some() || m.kv_adaptive.is_some() || m.eviction.is_some() || pflash_active {
        return false;
    }
    if batch.is_none() {
        return false;
    }
    // Ensure sampling controls are resolvable (mirrors sequential ladder)
    let _ = resolve_batch_sampling(msg, m);
    // Must be QwenAr route (non-spec)
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        // Topology already proven/staged above; ep:true would hit the global EP
        // short-circuit to Unknown for Qwen and make this QwenAr gate unreachable.
        ep: false,
        pp: m.pp,
        has_speculator: m.speculator.is_some(),
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample: m
            .speculator
            .as_ref()
            .map(|s| !s.requires_greedy())
            .unwrap_or(false),
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: hipfire_config::developer_var("HIPFIRE_DFLASH_TEMP_SPEC")
            .ok()
            .as_deref()
            == Some("0"),
        fast_sample_on: hipfire_config::developer_var("HIPFIRE_FAST_SAMPLE")
            .ok()
            .as_deref()
            != Some("0"),
        supports_temp_swor: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_temp_verify()),
        supports_chain_nucleus_verify: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_chain_nucleus_verify()),
        kv_adaptive: m.kv_adaptive.is_some(),
    };
    let route = select_generation_route(&route_inputs);
    if route != GenerationRoute::QwenAr {
        return false;
    }
    // Batch size coherence
    if continuous_batch_size != batch.as_ref().map(|b| b.max_batch()).unwrap_or(0) {
        return false;
    }
    true
}

pub fn drive_qwen35_ep_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::QwenAr;
    // Borrow EP batch state, config, weights via raw pointers to avoid aliasing.
    let ep_ptr = match model.ep.as_mut() {
        Some(ep) => ep as *mut EpState,
        None => return Err(BatchDriveError::Gpu("EP batch: no EP state".to_string())),
    };
    let (gpus_ptr, config_ptr, weights_ptr, batch_ptr, tokenizer_ptr, chat_template_clone, arch_id) = unsafe {
        let ep = &mut *ep_ptr;
        match &mut ep.inner {
            EpArch::Qwen35 {
                config,
                weights,
                batch,
            } => {
                let b = match batch.as_mut() {
                    Some(b) => b as *mut qwen35::Qwen35DecodeBatchEpState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "EP batch: batch not staged".to_string(),
                        ))
                    }
                };
                (
                    &mut ep.gpus as *mut hipfire_runtime::multi_gpu::Gpus,
                    config as *const qwen35::Qwen35Config,
                    weights as *const Vec<qwen35::Qwen35Weights>,
                    b,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    model.arch_id,
                )
            }
            _ => return Err(BatchDriveError::Gpu("EP batch: not Qwen35 EP".to_string())),
        }
    };
    let gpus: &mut hipfire_runtime::multi_gpu::Gpus = unsafe { &mut *gpus_ptr };
    let config: &qwen35::Qwen35Config = unsafe { &*config_ptr };
    let weights: &Vec<qwen35::Qwen35Weights> = unsafe { &*weights_ptr };
    let batch_state: &mut qwen35::Qwen35DecodeBatchEpState = unsafe { &mut *batch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let eos_tok = config.eos_token;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(eos_tok);
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    // Track last attested receipt for evidence; must be from runtime, never load logs.
    let mut last_receipt: Option<qwen35::Qwen35EpBatchReceipt> = None;
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
                    batch_state: &mut qwen35::Qwen35DecodeBatchEpState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let reset_res = batch_state.reset_all(gpus);
        let first_err = reset_res.err().map(|e| format!("EP batch reset_all: {e}"));
        let reason2 = if let Some(e) = first_err {
            format!("{reason}; {e}")
        } else {
            reason.clone()
        };
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason2}"),
                "gpu",
                false,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        Err(BatchDriveError::Poisoned(reason2))
    };
    loop {
        // handle awaiting commit/abort
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let reset_ok = match batch_state.reset_lane(gpus, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {idx} on commit: {e}"),
                    )
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!(),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        // EP batch-only admission; non-eligible becomes a
                        // barrier while preserving the reader-owned token.
                        if !is_qwen_ep_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier =
                                Some(daemon_regular_with_admission(json, Some(admission)));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    emit_batch_admission_error(
                                        stdout,
                                        &id,
                                        attempt_id,
                                        admission,
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::QwenAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return fail_all(sched, gpus, batch_state, stdout, reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!(
                                "[batch][EP] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope = BatchAttemptScope::enter_for_generation(
                                &id,
                                attempt_id,
                                admission,
                            );
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            let admission = pending_req.admission;
            if started_in_think {
                // Reset while the batch owner remains live, then retire the
                // lane and hand the complete original request to singleton.
                if let Err(err) = batch_state.reset_lane(gpus, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::QwenAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpus, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                break;
            }
            if let Err(e) = batch_state.reset_lane(gpus, config, lane_idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {lane_idx}: {e}"),
                );
            }
            let receipt =
                match batch_state.prefill_lane(gpus, weights, config, lane_idx, &prompt_tokens) {
                    Ok(r) => r,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP prefill lane {lane_idx}: {e}"),
                        )
                    }
                };
            last_receipt = Some(receipt);
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            // Use per-lane sampling that respects readiness; repeat penalties folded via retry window with product if needed.
            // For EP we call sample_lane (full product requires contiguous Ready lanes); per-lane keeps sparsity.
            let (next_token, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                lane_idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                lane_rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        // Build active mask and dense token/position vectors for EP forward_tick.
        let mut active_mask: u64 = 0;
        for &idx in &running {
            active_mask |= 1u64 << idx;
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        let receipt =
            match batch_state.forward_tick(gpus, weights, config, active_mask, &tokens, &positions)
            {
                Ok(r) => r,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP forward_tick: {e}"),
                    )
                }
            };
        last_receipt = Some(receipt);
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);
                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if matches!(finish.cause, QwenArTerminalCause::OpenThink) && !is_eos {
                    if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP reset lane {idx} on open think: {e}"),
                        );
                    }
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    let _scope =
                        BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                    emit_qwen_ar_open_think_terminal(
                        stdout,
                        &key.id,
                        lane.streamed_tokens.len(),
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                    continue;
                }
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                let finish_reason = match finish.finish_reason {
                    "length" => "length",
                    "tool_calls" => "tool_calls",
                    _ => "stop",
                };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                if let Some(receipt) = last_receipt.as_ref() {
                    attach_qwen_ep_batch_receipt_evidence(
                        &mut pending_done,
                        receipt,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                } else {
                    // Never fabricate: if no receipt yet, attach generic but still mark expert_parallel via default (should not happen on finishing lane after forward).
                    attach_continuous_batch_route_evidence(
                        &mut pending_done,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                    pending_done["continuous_batch"]["parallelism"] =
                        serde_json::json!("expert_parallel");
                    pending_done["continuous_batch"]["rank_count"] = serde_json::json!(4);
                    pending_done["continuous_batch"]["reduce"] =
                        serde_json::json!("peer_rooted_f32");
                }
                let _ = visible_text;
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch][EP] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "mark_awaiting_commit", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                    retire_lane_after_reset_failure(sched, stdout, &key, admission, idx, "commit_ready publish", &e);
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
            }
        }
        if survivors.is_empty() {
            continue;
        }
        // Per-lane sampling for survivors (sparse-aware). Use sample_lane to avoid contiguous prefix requirement.
        for idx in survivors.iter().cloned() {
            let sampling = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            };
            let rng = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.rng_state as u32,
                _ => continue,
            };
            let (tok, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample_lane survivor {idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                lane.next_token = Some(tok);
                lane.rng_state = next_rng as u64;
            }
        }
        // Also exercise sample_product when survivors form a contiguous full prefix (API coverage; sparse batches use sample_lane above).
        if survivors.len() == batch_size && survivors.iter().enumerate().all(|(i, &v)| i == v) {
            // Use the same sampling as first survivor for product validation; ignore error for non-product-capable batch shapes.
            if let Some(first) = survivors.first().and_then(|&idx| match &sched.lanes[idx] {
                BatchLane::Running(l) => Some(l.sampling.clone()),
                _ => None,
            }) {
                let dummy_repeat = vec![0u32; batch_size * 128];
                let dummy_lengths = vec![0u32; batch_size];
                let dummy_rng = vec![0u32; batch_size];
                let _ = batch_state.sample_product(
                    gpus,
                    config,
                    batch_size,
                    &dummy_repeat,
                    &dummy_lengths,
                    &dummy_rng,
                    first.temp,
                    first.top_p,
                    first.top_k,
                    first.min_p,
                    first.repeat_penalty,
                    first.presence_penalty,
                    first.frequency_penalty,
                );
            }
        }
    }
    Ok(())
}

/// Pre-activation protocol reject (missing/malformed attempt_id, or commands
/// with no active generate attempt). Always emits `attempt_id: 0`.
///
/// Do not use after `set_active_attempt_id` for a generate request.
pub fn emit_uncorrelated_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    crate::dense::write_error_envelope(stdout, id, message, class, retryable, rolled_back, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::ar::generation_test_lock()
    }

    #[derive(Clone, Copy)]
    enum MultiLaneTerminal {
        Done,
        Cancel,
        Error,
    }

    fn emit_multi_lane_terminal(
        route: GenerationRoute,
        terminal: MultiLaneTerminal,
        output: &mut Vec<u8>,
        id: &str,
        attempt_id: u64,
    ) {
        match terminal {
            MultiLaneTerminal::Done => {
                let pending = serde_json::json!({
                    "type": "done",
                    "id": id,
                    "attempt_id": attempt_id,
                    "finish_reason": "stop",
                });
                crate::ar::emit_generation_done_value(route, output, &pending);
            }
            MultiLaneTerminal::Cancel => {
                crate::ar::emit_generation_cancel(route, output, id, 0);
            }
            MultiLaneTerminal::Error => {
                crate::ar::emit_generation_error(
                    route,
                    output,
                    Some(id),
                    "multi-lane representative error",
                    "gpu",
                    false,
                    false,
                );
            }
        }
    }

    fn assert_multi_lane_route_latches_release(
        route: GenerationRoute,
        terminal: MultiLaneTerminal,
    ) {
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        assert_eq!(crate::ar::active_generation_route(), None);

        let admissions = [
            (
                "multi-lane-a",
                601_u64,
                batch_announce_terminal("multi-lane-a", 601).expect("lane A admission"),
            ),
            (
                "multi-lane-b",
                602_u64,
                batch_announce_terminal("multi-lane-b", 602).expect("lane B admission"),
            ),
        ];
        let mut output = Vec::new();

        for &(id, attempt_id, admission) in &admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, false);
        }
        for &(id, attempt_id, admission) in &admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_multi_lane_terminal(route, terminal, &mut output, id, attempt_id);
            assert_eq!(
                crate::ar::active_generation_route(),
                None,
                "terminal route must clear after {id}"
            );
        }
        for &(id, attempt_id, admission) in &admissions {
            assert!(batch_clear_terminal_at_generation(
                id, attempt_id, admission
            ));
        }

        // Re-announcing the exact wire keys must claim fresh route starts for
        // both lanes. A stale per-key route latch would suppress one of these.
        let fresh_admissions = [
            (
                "multi-lane-a",
                601_u64,
                batch_announce_terminal("multi-lane-a", 601).expect("lane A re-admission"),
            ),
            (
                "multi-lane-b",
                602_u64,
                batch_announce_terminal("multi-lane-b", 602).expect("lane B re-admission"),
            ),
        ];
        for &(id, attempt_id, admission) in &fresh_admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, false);
        }
        for &(id, attempt_id, admission) in &fresh_admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_multi_lane_terminal(route, terminal, &mut output, id, attempt_id);
            assert_eq!(
                crate::ar::active_generation_route(),
                None,
                "fresh terminal route must clear after {id}"
            );
            assert!(batch_clear_terminal_at_generation(
                id, attempt_id, admission
            ));
        }

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        for id in ["multi-lane-a", "multi-lane-b"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                    .count(),
                2,
                "both generations must start for {id}"
            );
        }
        let terminal_type = match terminal {
            MultiLaneTerminal::Done => "done",
            MultiLaneTerminal::Cancel => "aborted",
            MultiLaneTerminal::Error => "error",
        };
        let terminal_events = events
            .iter()
            .filter(|event| event["type"] == terminal_type)
            .count();
        assert_eq!(
            terminal_events, 4,
            "both lanes must emit both representative terminals"
        );
        for id in ["multi-lane-a", "multi-lane-b"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == terminal_type && event["id"] == id)
                    .count(),
                2,
                "both generations must terminate for {id}"
            );
        }
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        assert_eq!(crate::ar::active_generation_route(), None);
    }

    #[test]
    fn qwen_multi_lane_done_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::QwenAr, MultiLaneTerminal::Done);
    }

    #[test]
    fn lfm_multi_lane_cancel_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::LfmAr, MultiLaneTerminal::Cancel);
    }

    #[test]
    fn qwen_ep_multi_lane_error_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::QwenAr, MultiLaneTerminal::Error);
    }
    struct FlushGateWriter {
        pending: Vec<u8>,
        visible: Vec<u8>,
    }

    impl FlushGateWriter {
        fn events(&self) -> Vec<serde_json::Value> {
            std::str::from_utf8(&self.visible)
                .expect("visible terminal bytes are UTF-8")
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("visible terminal event is JSON"))
                .collect()
        }
    }

    impl Write for FlushGateWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.pending.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.visible.append(&mut self.pending);
            Ok(())
        }
    }

    #[derive(Default)]
    struct FailingWriter {
        bytes: Vec<u8>,
        fail_write: bool,
        fail_flush: bool,
    }

    impl FailingWriter {
        fn events(&self) -> Vec<serde_json::Value> {
            std::str::from_utf8(&self.bytes)
                .expect("writer bytes are UTF-8")
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("writer event is JSON"))
                .collect()
        }
    }

    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.fail_write {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected write failure",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected flush failure",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn route_terminals_flush_visible_done_error_and_cancel() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let done_lanes = [
            (
                "flush-done-a",
                701_u64,
                batch_announce_terminal("flush-done-a", 701).expect("done A admission"),
            ),
            (
                "flush-done-b",
                702_u64,
                batch_announce_terminal("flush-done-b", 702).expect("done B admission"),
            ),
        ];
        let mut output = FlushGateWriter {
            pending: Vec::new(),
            visible: Vec::new(),
        };
        for &(id, attempt_id, admission) in &done_lanes {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(
                GenerationRoute::QwenAr,
                &mut output,
                id,
                false,
            );
            output.flush().expect("start flush");
        }
        for (done_index, &(id, attempt_id, admission)) in done_lanes.iter().enumerate() {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            let pending = serde_json::json!({
                "type": "done",
                "id": id,
                "attempt_id": attempt_id,
                "finish_reason": "stop",
            });
            assert!(crate::ar::emit_generation_done_value(
                GenerationRoute::QwenAr,
                &mut output,
                &pending,
            ));
            assert_eq!(
                output
                    .events()
                    .iter()
                    .filter(|event| event["type"] == "done" && event["id"] == id)
                    .count(),
                1,
                "route done {done_index} must be visible after route emission"
            );
            assert!(batch_clear_terminal_at_generation(id, attempt_id, admission));
        }

        let error_admission =
            batch_announce_terminal("flush-error", 703).expect("error admission");
        {
            let _scope =
                BatchAttemptScope::enter_for_generation("flush-error", 703, error_admission);
            crate::ar::emit_generation_start(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-error",
                false,
            );
            output.flush().expect("error start flush");
            assert!(crate::ar::emit_generation_error(
                GenerationRoute::QwenAr,
                &mut output,
                Some("flush-error"),
                "representative error",
                "internal",
                false,
                true,
            ));
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "error" && event["id"] == "flush-error")
                .count(),
            1,
            "route error must be visible after route emission"
        );
        assert!(batch_clear_terminal_at_generation(
            "flush-error",
            703,
            error_admission
        ));

        let cancel_admission =
            batch_announce_terminal("flush-cancel", 704).expect("cancel admission");
        {
            let _scope =
                BatchAttemptScope::enter_for_generation("flush-cancel", 704, cancel_admission);
            crate::ar::emit_generation_start(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-cancel",
                false,
            );
            output.flush().expect("cancel start flush");
            assert!(crate::ar::emit_generation_cancel(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-cancel",
                1,
            ));
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "aborted" && event["id"] == "flush-cancel")
                .count(),
            1,
            "route cancel must be visible after route emission"
        );
        assert!(batch_clear_terminal_at_generation(
            "flush-cancel",
            704,
            cancel_admission
        ));

        let events = output.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "done"
                    && (event["id"] == "flush-done-a" || event["id"] == "flush-done-b"))
                .count(),
            2,
            "both committed done envelopes must be visible after route emission"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "error" && event["id"] == "flush-error")
                .count(),
            1,
            "route errors must be visible after route emission"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "aborted" && event["id"] == "flush-cancel")
                .count(),
            1,
            "route cancels must be visible after route emission"
        );

        // A stale duplicate cannot write a second terminal, even though the
        // route writer still performs its successful flush.
        for &(id, attempt_id, _) in &done_lanes {
            let pending = serde_json::json!({
                "type": "done",
                "id": id,
                "attempt_id": attempt_id,
                "finish_reason": "stop",
            });
            let before = output.visible.len();
            let _scope = BatchAttemptScope::enter_for(id, attempt_id);
            crate::ar::emit_generation_done_value(
                GenerationRoute::QwenAr,
                &mut output,
                &pending,
            );
            assert_eq!(output.visible.len(), before);
        }

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    #[derive(Clone, Copy)]
    enum WriterFailure {
        Write,
        Flush,
    }

    fn assert_route_terminal_failure_releases_latch(failure: WriterFailure) {
        let (id, attempt_id) = match failure {
            WriterFailure::Write => ("route-write-failure", 705_u64),
            WriterFailure::Flush => ("route-flush-failure", 706_u64),
        };
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let admission = batch_announce_terminal(id, attempt_id).expect("failure admission");
        let mut output = FailingWriter::default();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(GenerationRoute::QwenAr, &mut output, id, false);
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                .count(),
            1,
            "{id} initial route start",
        );

        match failure {
            WriterFailure::Write => output.fail_write = true,
            WriterFailure::Flush => output.fail_flush = true,
        }
        let pending = serde_json::json!({
            "type": "done",
            "id": id,
            "attempt_id": attempt_id,
            "finish_reason": "stop",
        });
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            assert!(
                !crate::ar::emit_generation_done_value(
                    GenerationRoute::QwenAr,
                    &mut output,
                    &pending,
                ),
                "{id} injected terminal failure must report undelivered",
            );
        }
        assert_eq!(
            crate::ar::active_generation_route(),
            None,
            "{id} failed terminal must clear the active route",
        );
        let after_failure = output.bytes.len();

        // Once the exact claim is consumed, a duplicate remains suppressed
        // even after the writer recovers.
        output.fail_write = false;
        output.fail_flush = false;
        {
            let _scope = BatchAttemptScope::enter_for(id, attempt_id);
            assert!(
                !crate::ar::emit_generation_done_value(
                    GenerationRoute::QwenAr,
                    &mut output,
                    &pending,
                ),
                "{id} duplicate terminal must stay suppressed",
            );
        }
        assert_eq!(
            output.bytes.len(),
            after_failure,
            "{id} duplicate terminal must not write",
        );
        assert!(batch_clear_terminal_at_generation(
            id, attempt_id, admission
        ));

        // Reusing the exact wire key must claim a fresh start after the
        // failed terminal consumed the previous lifecycle claim.
        let fresh_admission =
            batch_announce_terminal(id, attempt_id).expect("fresh failure admission");
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, fresh_admission);
            crate::ar::emit_generation_start(GenerationRoute::QwenAr, &mut output, id, false);
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                .count(),
            2,
            "{id} same-key reuse must emit a fresh route start",
        );
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, fresh_admission);
            assert!(crate::ar::emit_generation_done_value(
                GenerationRoute::QwenAr,
                &mut output,
                &pending,
            ));
        }
        assert!(batch_clear_terminal_at_generation(
            id,
            attempt_id,
            fresh_admission,
        ));
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    #[test]
    fn route_terminal_write_failure_consumes_claim_and_releases_latch() {
        let _guard = lock();
        assert_route_terminal_failure_releases_latch(WriterFailure::Write);
    }

    #[test]
    fn route_terminal_flush_failure_consumes_claim_and_releases_latch() {
        let _guard = lock();
        assert_route_terminal_failure_releases_latch(WriterFailure::Flush);
    }

    #[test]
    fn direct_driver_admission_errors_are_correlated_once_and_cleared() {
        let _guard = lock();
        for (driver, id, attempt_id, message, class) in [
            (
                "qwen",
                "qwen-direct",
                101_u64,
                "invalid messages field",
                "validation",
            ),
            (
                "lfm",
                "lfm-direct",
                202_u64,
                "prompt exceeds context capacity",
                "context_length",
            ),
            (
                "qwen35-ep",
                "qwen35-ep-direct",
                303_u64,
                "seed must fit in a u32",
                "validation",
            ),
        ] {
            let admission =
                batch_announce_terminal(id, attempt_id).expect("{driver} announce");

            let mut output = Vec::new();
            emit_batch_admission_error(
                &mut output,
                id,
                attempt_id,
                admission,
                message,
                class,
                false,
                false,
            );

            let lines: Vec<&str> = std::str::from_utf8(&output)
                .expect("UTF-8 error envelope")
                .lines()
                .filter(|line| !line.is_empty())
                .collect();
            assert_eq!(lines.len(), 1, "{driver} terminal count");
            let event: serde_json::Value =
                serde_json::from_str(lines[0]).expect("JSON error envelope");
            assert_eq!(event["type"], "error", "{driver} event type");
            assert_eq!(
                event["attempt_id"].as_u64(),
                Some(attempt_id),
                "{driver} attempt id"
            );
            assert_ne!(
                event["attempt_id"].as_u64(),
                Some(0),
                "{driver} attempt zero"
            );
            assert_eq!(event["id"].as_str(), Some(id), "{driver} request id");
            assert_eq!(event["class"].as_str(), Some(class), "{driver} error class");
            assert_eq!(
                batch_terminal_generation(id, attempt_id),
                None,
                "{driver} admission cleanup"
            );
        }
    }

    fn assert_started_in_think_handoff(
        route: GenerationRoute,
        id: &str,
        attempt_id: u64,
        abort_latched: bool,
    ) {
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "full prompt",
            "system": "full system",
            "messages": [{"role": "user", "content": "full prompt"}],
            "tools": [{"type": "function", "function": {"name": "keep"}}],
            "stop": ["<done>"],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 7,
            "seed": 9,
            "reasoning_effort": "low",
            "assistant_prefix": "open_think",
        });
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        activate_terminal_control(id, attempt_id);
        set_active_attempt_id(attempt_id);
        if abort_latched {
            apply_terminal_control("abort", id, attempt_id);
            batch_apply_terminal_control("abort", id, attempt_id);
        }
        let singleton_generation =
            terminal_generation(id, attempt_id).expect("singleton transaction");
        assert!(batch_transition_to_queued(id, attempt_id, admission));
        let key = AttemptKey::new(id, attempt_id);
        let sampling = BatchSampling {
            temp: 0.3,
            top_p: 0.8,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_window: 128,
        };
        let mut sched = ContinuousBatchScheduler::new(1, 64);
        assert!(sched.enqueue(BatchPendingRequest {
            key: key.clone(),
            admission,
            original_msg: original.clone(),
            prompt: "full prompt".to_string(),
            prompt_tokens: vec![1, 2, 3],
            started_in_think: true,
            system: Some("full system".to_string()),
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
            max_think_tokens: 16,
            max_tokens: 7,
            client_seed: Some(9),
            sampling,
        }));
        let (assigned_key, ticket) = sched.try_assign_one().expect("assigned think lane");
        assert_eq!(assigned_key, key);

        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, true);
        }
        let pending = sched.pending.get(&key).cloned().expect("pending request");
        let handoff = handoff_started_in_think(&mut sched, ticket.lane, &key, &pending, route)
            .expect("singleton handoff");
        assert_eq!(sched.active_count(), 0, "barrier lane retired");
        assert!(
            sched.pending.is_empty(),
            "barrier request removed from batch"
        );
        assert!(
            sched.try_assign_one().is_none(),
            "barrier request cannot continue in GPU lane"
        );
        assert_eq!(batch_terminal_generation(id, attempt_id), None);

        let (handoff_msg, transfer) = match handoff {
            DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
            _ => panic!("think barrier did not produce singleton ownership"),
        };
        assert_eq!(handoff_msg, original, "full request payload was preserved");
        assert_eq!(transfer.admission(), admission);

        // The outer main transaction has ended; adoption restores the exact
        // lifecycle generation and any pre-latched abort without reactivation.
        clear_terminal_control();
        assert!(adopt_singleton_transfer(id, attempt_id, transfer));
        assert_eq!(
            terminal_generation(id, attempt_id),
            Some(singleton_generation)
        );
        assert_eq!(check_abort(id), abort_latched);

        {
            let _scope = BatchAttemptScope::enter(attempt_id);
            crate::ar::emit_generation_start(route, &mut output, id, true);
            if abort_latched {
                crate::ar::emit_active_route_cancel(&mut output, id, 0);
            } else {
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "think barrier normal terminal",
                    "validation",
                    false,
                    false,
                );
            }
        }
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "gen_start", "fresh sequential start");
        if abort_latched {
            assert_eq!(events.len(), 4);
            assert_eq!(events[2]["type"], "aborted");
            assert_eq!(events[3]["type"], "done");
        } else {
            assert_eq!(events.len(), 3);
            assert_eq!(events[2]["type"], "error");
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    fn assert_admitted_started_in_think_handoff(
        route: GenerationRoute,
        id: &str,
        attempt_id: u64,
        abort_latched: bool,
    ) {
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "full prompt",
            "messages": [{"role": "user", "content": "full prompt"}],
            "tools": [{"type": "function", "function": {"name": "keep"}}],
            "stop": ["<done>"],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 7,
            "seed": 9,
            "reasoning_effort": "low",
            "assistant_prefix": "open_think",
        });
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        activate_terminal_control(id, attempt_id);
        set_active_attempt_id(attempt_id);
        if abort_latched {
            apply_terminal_control("abort", id, attempt_id);
            batch_apply_terminal_control("abort", id, attempt_id);
        }
        let singleton_generation =
            terminal_generation(id, attempt_id).expect("singleton transaction");
        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, true);
        }
        let handoff =
            handoff_admitted_started_in_think(id, attempt_id, admission, original.clone(), route)
                .expect("admitted singleton handoff");
        assert_eq!(batch_terminal_generation(id, attempt_id), None);
        let (handoff_msg, transfer) = match handoff {
            DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
            _ => panic!("admitted think barrier did not produce singleton ownership"),
        };
        assert_eq!(handoff_msg, original, "full admitted payload was preserved");
        assert_eq!(transfer.admission(), admission);

        clear_terminal_control();
        assert!(adopt_singleton_transfer(id, attempt_id, transfer));
        assert_eq!(
            terminal_generation(id, attempt_id),
            Some(singleton_generation)
        );
        assert_eq!(check_abort(id), abort_latched);
        {
            let _scope = BatchAttemptScope::enter(attempt_id);
            crate::ar::emit_generation_start(route, &mut output, id, true);
            if abort_latched {
                crate::ar::emit_active_route_cancel(&mut output, id, 0);
            } else {
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "admitted think barrier normal terminal",
                    "validation",
                    false,
                    false,
                );
            }
        }
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "gen_start", "fresh sequential start");
        if abort_latched {
            assert_eq!(events.len(), 4);
            assert_eq!(events[2]["type"], "aborted");
            assert_eq!(events[3]["type"], "done");
        } else {
            assert_eq!(events.len(), 3);
            assert_eq!(events[2]["type"], "error");
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn admitted_think_batch_driver_bootstraps_singleton_and_reuses_key() {
        let _guard = lock();
        let route = GenerationRoute::QwenAr;
        let id = "qwen-later-think";
        let attempt_id = 507_u64;
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "later queued prompt",
            "messages": [{"role": "user", "content": "later queued prompt"}],
            "max_tokens": 7,
            "reasoning_effort": "low",
        });
        let mut output = Vec::new();
        let mut admissions = Vec::new();
        let mut singleton_generations = Vec::new();

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        for _ in 0..2 {
            // This is the batch-driver admission path: the request is queued
            // before the think-open barrier, without singleton activation.
            let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
            assert!(batch_transition_to_queued(id, attempt_id, admission));
            assert_eq!(
                terminal_generation(id, attempt_id),
                None,
                "batch-driver request has no manually activated singleton"
            );

            let handoff = handoff_admitted_started_in_think(
                id,
                attempt_id,
                admission,
                original.clone(),
                route,
            )
            .expect("admitted singleton handoff");
            let (handoff_msg, transfer) = match handoff {
                DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
                _ => panic!("admitted think barrier did not produce singleton ownership"),
            };
            assert_eq!(handoff_msg, original, "full queued payload was preserved");
            assert_eq!(transfer.admission(), admission);
            assert_eq!(batch_terminal_generation(id, attempt_id), None);

            // Handoff bootstraps the singleton inside the tombstone; main
            // adopts that exact owner instead of rediscovering it.
            assert_eq!(terminal_generation(id, attempt_id), None);
            clear_terminal_control();
            assert!(adopt_singleton_transfer(id, attempt_id, transfer));
            let singleton_generation =
                terminal_generation(id, attempt_id).expect("adopted singleton transaction");
            assert!(singleton_generation > 0);
            admissions.push(admission);
            singleton_generations.push(singleton_generation);

            {
                let _attempt = BatchAttemptScope::enter_singleton(attempt_id);
                let _route = GenerationRouteScope::enter(route, id);
                crate::ar::emit_generation_start(route, &mut output, id, true);
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "admitted think barrier terminal",
                    "validation",
                    false,
                    false,
                );
            }
            assert_eq!(crate::ar::active_generation_route(), None);
            clear_terminal_control();
        }

        assert_ne!(admissions[0], admissions[1], "same key received fresh admissions");
        assert_ne!(
            singleton_generations[0], singleton_generations[1],
            "same key received fresh singleton lifecycles"
        );

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 4, "one start and one terminal per reuse");
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().expect("event type"))
                .collect::<Vec<_>>(),
            vec!["gen_start", "error", "gen_start", "error"]
        );

        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn admitted_started_in_think_barrier_preserves_all_batch_routes() {
        let _guard = lock();
        for (route, id, attempt_id, abort_latched) in [
            (GenerationRoute::QwenAr, "qwen-admitted-think", 504, true),
            (GenerationRoute::LfmAr, "lfm-admitted-think", 505, false),
            (GenerationRoute::QwenAr, "ep-admitted-think", 506, false),
        ] {
            assert_admitted_started_in_think_handoff(route, id, attempt_id, abort_latched);
        }
    }

    #[test]
    fn qwen_started_in_think_barrier_preserves_singleton_owner() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::QwenAr, "qwen-think", 501, true);
    }

    #[test]
    fn lfm_started_in_think_barrier_preserves_full_request() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::LfmAr, "lfm-think", 502, false);
    }

    #[test]
    fn ep_started_in_think_barrier_preserves_full_request() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::QwenAr, "ep-think", 503, false);
    }

    #[test]
    fn lfm_assignment_capacity_error_releases_route_latch_for_reuse() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let id = "lfm-assignment";
        let attempt_id = 404_u64;
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        let key = AttemptKey::new(id, attempt_id);
        let sampling = BatchSampling {
            temp: 0.3,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_window: 128,
        };
        let mut sched = ContinuousBatchScheduler::new(1, 8);
        assert!(sched.enqueue(BatchPendingRequest {
            key: key.clone(),
            admission,
            original_msg: serde_json::json!({
                "type": "generate",
                "id": id,
                "attempt_id": attempt_id,
                "prompt": "oversized",
                "max_tokens": 4,
            }),
            prompt: "oversized".to_string(),
            prompt_tokens: vec![1; 7],
            started_in_think: false,
            system: None,
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            max_think_tokens: 0,
            max_tokens: 4,
            client_seed: None,
            sampling,
        }));
        let (assigned_key, ticket) = sched.try_assign_one().expect("assigned lane");
        assert_eq!(assigned_key, key);

        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(
                crate::ar::GenerationRoute::LfmAr,
                &mut output,
                id,
                false,
            );
        }
        assert_eq!(
            crate::ar::active_generation_route(),
            Some(crate::ar::GenerationRoute::LfmAr)
        );

        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_lfm_assignment_capacity_error(&mut output, &key, 7, 4, 8);
        }
        assert_eq!(crate::ar::active_generation_route(), None);

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 error envelope")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 2, "assignment emits one start and one error");
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["attempt_id"].as_u64(), Some(attempt_id));
        assert_eq!(events[1]["type"], "error");
        assert_eq!(events[1]["id"].as_str(), Some(id));
        assert_eq!(events[1]["attempt_id"].as_u64(), Some(attempt_id));
        assert_eq!(events[1]["class"].as_str(), Some("context_length"));
        assert!(sched.abort_lane(ticket.lane, &key, admission));
        assert_eq!(batch_terminal_generation(id, attempt_id), None);

        // Reusing the exact wire key must claim a fresh route start rather
        let reuse_admission =
            batch_announce_terminal(id, attempt_id).expect("reused batch admission");
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, reuse_admission);
            crate::ar::emit_generation_start(
                crate::ar::GenerationRoute::LfmAr,
                &mut output,
                id,
                false,
            );
            assert_eq!(
                crate::ar::active_generation_route(),
                Some(crate::ar::GenerationRoute::LfmAr)
            );
        }
        let reused_events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(reused_events.len(), 3);
        assert_eq!(reused_events[2]["type"], "gen_start");
        assert_eq!(reused_events[2]["attempt_id"].as_u64(), Some(attempt_id));

        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, reuse_admission);
            crate::ar::emit_active_route_cancel(&mut output, id, 0);
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        assert!(batch_clear_terminal_at_generation(
            id,
            attempt_id,
            reuse_admission
        ));
        set_active_attempt_id(0);
        clear_terminal_control();
    }
    /// A lane whose GPU reset fails is retired with a visible
    /// `rolled_back=false` error keyed by its own AttemptKey claim; the
    /// peer lane keeps serving untouched. Ordinary and EP drivers call
    /// `retire_lane_after_reset_failure` independently per failed lane,
    /// so both flavors are exercised here through their own admissions.
    #[test]
    fn lane_reset_failure_retires_with_visible_unattested_error() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let mut sched = ContinuousBatchScheduler::new(2, 8);
        let mut lanes = Vec::new();
        for (id, attempt_id) in [("retire-ordinary", 701_u64), ("retire-ep", 702_u64)] {
            let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
            let key = AttemptKey::new(id, attempt_id);
            assert!(sched.enqueue(BatchPendingRequest {
                key: key.clone(),
                admission,
                original_msg: serde_json::json!({
                    "type": "generate",
                    "id": id,
                    "attempt_id": attempt_id,
                }),
                prompt: "hello".to_string(),
                prompt_tokens: vec![1, 2, 3],
                started_in_think: false,
                system: None,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                max_think_tokens: 0,
                max_tokens: 4,
                client_seed: None,
                sampling: BatchSampling {
                    temp: 0.0,
                    top_p: 1.0,
                    top_k: None,
                    min_p: None,
                    repeat_penalty: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repeat_window: 128,
                },
            }));
            let (assigned_key, ticket) = sched.try_assign_one().expect("assigned lane");
            assert_eq!(assigned_key, key);
            lanes.push((key, admission, ticket.lane));
        }
        let mut output = Vec::new();
        let (first, second) = (&lanes[0], &lanes[1]);
        retire_lane_after_reset_failure(
            &mut sched,
            &mut output,
            &first.0,
            first.1,
            first.2,
            "commit_ready publish",
            &"injected reset failure",
        );
        // The peer lane keeps serving untouched: it still holds its key and
        // its admission is still live after the first lane retired.
        assert!(
            sched.lanes[second.2].key().is_some_and(|k| k == &second.0),
            "peer lane keeps its key"
        );
        assert!(
            batch_terminal_generation(&second.0.id, second.0.attempt_id).is_some(),
            "peer admission stays live"
        );
        retire_lane_after_reset_failure(
            &mut sched,
            &mut output,
            &second.0,
            second.1,
            second.2,
            "mark_awaiting_commit",
            &"injected reset failure",
        );
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 error envelopes")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 2, "one visible error per retired lane");
        for (event, (key, _, _)) in events.iter().zip(lanes.iter()) {
            assert_eq!(event["type"], "error");
            assert_eq!(event["id"].as_str(), Some(key.id.as_str()));
            assert_eq!(event["attempt_id"].as_u64(), Some(key.attempt_id));
            assert_eq!(event["class"].as_str(), Some("gpu"));
            assert_eq!(event["retryable"], serde_json::json!(false));
            assert_eq!(event["rolled_back"], serde_json::json!(false));
            assert!(
                event["message"].as_str().is_some_and(|m| m.contains("lane retired")),
                "reset failure stays visible: {}",
                event["message"]
            );
        }
        for (key, admission, lane_idx) in &lanes {
            assert!(
                sched.lanes[*lane_idx].key().is_none(),
                "failed lane is retired, not reused"
            );
            assert_eq!(
                batch_terminal_generation(&key.id, key.attempt_id),
                None,
                "retired admission cleared independently"
            );
        }
        set_active_attempt_id(0);
        clear_terminal_control();
    }
}
