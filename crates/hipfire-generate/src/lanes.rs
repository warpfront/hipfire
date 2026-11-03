// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The shared fixed-lane continuous-batch driver (#666 G12).
//!
//! One driver owns admission, seeding, the decode step, terminals, cancel
//! and retire for every fixed-lane family; a [`LaneFamily`] adapter supplies
//! only the model work (prefill a lane, decode the running lanes, sample)
//! and the family's token-to-wire routing. Glimmer (arch 14) and LFM2
//! (arch 11) are the adapters; the LFM2 route stays refused at
//! [`hipfire_loader::continuous_batch_route`].

use crate::ar::*;
use crate::batch::{
    emit_batch_admission_error, emit_uncorrelated_error, handoff_admitted_started_in_think,
    handoff_started_in_think, is_batch_request_eligible, retire_lane_after_reset_failure,
    BatchTerminalCleanup,
};
use hipfire_arch_deepseek4::sampling::{sample_token, Xorshift};
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::Lfm2MoeBundle;
use hipfire_engine::emit::*;
use hipfire_engine::prompt::{batch_render_prompt_tokens, qwen_jinja_reasoning};
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::wire_seed;
use hipfire_loader::{LoadedModel, MuseGlimmerBundle};
use hipfire_runtime::prompt_frame::ToolCall;
use hipfire_runtime::tokenizer::{TokenTextStream, Tokenizer};
use std::any::Any;
use std::io::Write;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// One running lane's next draw, after its committed token was routed.
pub(crate) struct LaneDraw {
    pub lane: usize,
    /// The last `min(repeat_window, repeat_capacity)` streamed tokens.
    pub history: Vec<u32>,
    pub sampling: BatchSampling,
    pub rng: u32,
}

/// What routing one committed token did to its lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneEmit {
    /// The lane ends here; the token is not counted.
    Stop,
    /// Text (answer or reasoning) went out; stamps the lane's first token.
    Visible,
    /// Counted, but nothing went out (a partial character, held-back spans).
    Hidden,
}

/// The model work of one fixed-lane family. Errors fail the whole batch.
pub(crate) trait LaneFamily {
    fn route(&self) -> GenerationRoute;
    /// Token fed to rows with no running lane.
    fn idle_token(&self) -> u32;
    /// Most history tokens [`LaneFamily::sample`] reads per lane.
    fn repeat_capacity(&self) -> usize;
    fn reset(&mut self, gpu: &mut rdna_compute::Gpu) -> Result<(), String>;
    fn reset_lane(&mut self, gpu: &mut rdna_compute::Gpu, lane: usize) -> Result<(), String>;
    /// Fresh per-lane routing state for `req`, before its prefill.
    fn open_lane(&mut self, lane: usize, req: &BatchPendingRequest);
    /// `Ok(false)`: `abort` fired before or after the GPU work.
    fn prefill_lane(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        lane: usize,
        prompt: &[u32],
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String>;
    /// Whether [`LaneFamily::prefill_lanes`] exists.
    fn batched_prefill(&self) -> bool;
    /// Whether [`LaneFamily::sample`] honours each draw's own config, so
    /// lanes of different sampling configs may share a batch.
    fn per_lane_sampling(&self) -> bool;
    /// Equal-length prompts into lanes `0..prompts.len()` in one pass.
    fn prefill_lanes(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        prompts: &[&[u32]],
    ) -> Result<(), String>;
    /// First token of `lane` from its prefill.
    fn sample_first(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        lane: usize,
        sampling: &BatchSampling,
        rng: u32,
    ) -> Result<(u32, u32), String>;
    /// Row `r` takes `tokens[r]` at `positions[r]`; only `running` rows are live.
    fn decode(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        positions: &[usize],
        running: &[usize],
    ) -> Result<(), String>;
    /// Next `(token, rng)` per draw, from the last [`LaneFamily::decode`].
    fn sample(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        draws: &[LaneDraw],
    ) -> Result<Vec<(u32, u32)>, String>;
    /// Route committed `token` of `lane` to the wire.
    fn emit(
        &mut self,
        lane: usize,
        state: &mut QwenBatchLane,
        token: u32,
        stdout: &mut std::io::Stdout,
        id: &str,
    ) -> LaneEmit;
    /// Flush `lane` at its terminal; returns its parsed tool calls.
    fn finish(&mut self, lane: usize, stdout: &mut std::io::Stdout, id: &str) -> Vec<ToolCall>;
}

/// Prompt tokens and started-in-think of a lane-family request. The daemon
/// renders the first request of a wave and the driver the rest, both here.
pub fn render_lane_prompt(
    m: &LoadedModel,
    msg: &serde_json::Value,
) -> Result<(Vec<u32>, bool), String> {
    let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer missing")?;
    let prompt = batch_single_user_content(msg).unwrap_or_else(|| {
        msg.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello")
            .to_string()
    });
    let system = msg.get("system").and_then(|v| v.as_str());
    let max_think = msg
        .get("max_think_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let messages = match msg.get("messages") {
        Some(v) => Some(
            serde_json::from_value::<Vec<hipfire_runtime::prompt_frame::Message>>(v.clone())
                .map_err(|e| format!("invalid messages field: {e}"))?,
        ),
        None => None,
    };
    if let Some(b) = m.muse_glimmer() {
        // The singleton dispatch normalizes `prompt` and every message
        // (`maybe_normalize_prompt`, `parse_generate_messages`); render the
        // same tokens here.
        let glimmer_prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(
            msg.get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("Hello"),
        );
        let glimmer_messages = hipfire_engine::prompt::parse_generate_messages(msg)
            .map_err(|e| format!("invalid messages field: {e}"))?;
        // Same effort derivation the daemon hands the singleton route.
        let raw_effort = msg
            .get("reasoning_effort")
            .or_else(|| msg.get("thinking_mode"))
            .and_then(|v| v.as_str());
        let thinking = msg.get("thinking_enabled").and_then(|v| v.as_bool());
        let (_, effort) = qwen_jinja_reasoning(thinking, raw_effort, max_think);
        let unused = std::cell::Cell::new(0);
        let ids = crate::dense::glimmer_prompt_ids(
            tokenizer,
            m.chat_template.as_ref(),
            // Batch requests are a single user turn: no assistant turn to replay.
            &mut hipfire_loader::AsstTurnCache::new_from_env(),
            b.config.bos_token,
            &glimmer_prompt,
            system,
            effort.as_deref(),
            max_think,
            None,
            glimmer_messages.as_deref(),
            &unused,
            &unused,
        )?;
        return Ok((ids, false));
    }
    let assistant_prefix = match msg
        .get("assistant_prefix")
        .and_then(|v| v.as_str())
        .unwrap_or("plain")
    {
        "open_think" => hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
        "closed_think" => hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
        _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
    };
    batch_render_prompt_tokens(
        &prompt,
        system,
        assistant_prefix,
        tokenizer,
        m.chat_template.as_ref(),
        max_think,
        messages.as_deref(),
        max_think != 1,
        None,
    )
    .map_err(|e| e.to_string())
}

/// Emit the assignment-time capacity failure. The caller must hold the exact
/// `BatchAttemptScope`; the route adapter claims the terminal and releases
/// the matching start latch before the scheduler retires the lane.
pub(crate) fn emit_lane_assignment_capacity_error(
    route: GenerationRoute,
    stdout: &mut impl Write,
    key: &AttemptKey,
    prompt_len: usize,
    max_tokens: usize,
    capacity: usize,
) {
    crate::ar::emit_generation_error(
        route,
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

/// The generation route of a model served by this driver; `None` for every
/// other family.
pub fn lane_family_route(m: &LoadedModel) -> Option<GenerationRoute> {
    if m.muse_glimmer().is_some() {
        Some(GenerationRoute::GlimmerAr)
    } else if m.lfm2moe().is_some() {
        Some(GenerationRoute::LfmAr)
    } else {
        None
    }
}

/// Daemon: drive the staged fixed-lane batch of a lane family (Glimmer,
/// LFM2) until it drains or a barrier message needs the sequential route.
pub fn drive_lane_continuous_batch(
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
    // Admission reads the model while the family holds its bundle: the same
    // aliasing the per-family drivers before this one did.
    let model_ptr: *const LoadedModel = model;
    let mut family = lane_family(model).map_err(BatchDriveError::Gpu)?;
    let family: &mut dyn LaneFamily = &mut *family;
    let route = family.route();
    let idle = family.idle_token();
    sched.per_lane_sampling = family.per_lane_sampling();
    let loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    family: &mut dyn LaneFamily,
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
        if let Err(e) = family.reset(gpu) {
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
            if let Err(e) = family.reset_lane(gpu, idx) {
                return fail_all(
                    sched,
                    gpu,
                    family,
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
            let reset_ok = match family.reset_lane(gpu, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        family,
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
            if let Err(e) = family.reset_lane(gpu, idx) {
                return fail_all(
                    sched,
                    gpu,
                    family,
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
                            crate::ar::emit_generation_start(route, stdout, &id, false);
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            unsafe { &*model_ptr },
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
                        let (prompt_tokens, started_in_think) =
                            match render_lane_prompt(unsafe { &*model_ptr }, &json) {
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
                                &id, attempt_id, admission, json, route,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => return fail_all(sched, gpu, family, stdout, reason),
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
                        // A request a lane cannot hold is the singleton's
                        // (it serves to its cache cap): hand it over, never refuse.
                        if batch_lfm_exceeds_capacity(
                            prompt_tokens.len(),
                            max_tokens_req,
                            sched.lane_capacity,
                        ) {
                            let handoff = match handoff_admitted_started_in_think(
                                &id, attempt_id, admission, json, route,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => return fail_all(sched, gpu, family, stdout, reason),
                            };
                            barrier = Some(handoff);
                            break;
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
                        let sampling = resolve_batch_sampling(&json, unsafe { &*model_ptr });
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
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(route, stdout, &id, started_in_think);
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
            if n >= 2 && family.batched_prefill() {
                // Assign exactly n prefix lanes; each try_assign_one pops front and binds.
                let mut assigned_keys: Vec<AttemptKey> = Vec::with_capacity(n);
                let mut assigned_tickets: Vec<LaneTicket> = Vec::with_capacity(n);
                let mut prompts_for_batch: Vec<Vec<u32>> = Vec::with_capacity(n);
                let mut reqs_for_batch: Vec<BatchPendingRequest> = Vec::with_capacity(n);
                let mut assign_ok = true;
                for _ in 0..n {
                    match sched.try_assign_one() {
                        Some((key, ticket)) => {
                            let req = sched.pending.get(&key).cloned();
                            prompts_for_batch
                                .push(req.as_ref().map_or(Vec::new(), |r| r.prompt_tokens.clone()));
                            reqs_for_batch.extend(req);
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
                    for (t, req) in assigned_tickets.iter().zip(&reqs_for_batch) {
                        family.open_lane(t.lane, req);
                    }
                    let prompt_refs: Vec<&[u32]> =
                        prompts_for_batch.iter().map(|v| v.as_slice()).collect();
                    let prefill_res = family.prefill_lanes(gpu, &prompt_refs);
                    match prefill_res {
                        Ok(()) => {
                            for (idx, key) in assigned_keys.iter().enumerate() {
                                let ticket = assigned_tickets[idx];
                                let lane_idx = ticket.lane;
                                let admission = ticket.admission;
                                if batch_check_abort(&key.id, key.attempt_id, admission) {
                                    if let Err(e) = family.reset_lane(gpu, lane_idx) {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            family,
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
                                let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                                    BatchLane::Running(l) => {
                                        (l.rng_state as u32, l.sampling.clone())
                                    }
                                    _ => continue,
                                };
                                match family.sample_first(gpu, lane_idx, &sampling, lane_rng) {
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
                                            family,
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
                                family,
                                stdout,
                                format!("batched prefill lanes 0..{n}: {e}"),
                            );
                        }
                    }
                } else {
                    // Partial assign failure: rollback any already-assigned lanes.
                    for (k, t) in assigned_keys.iter().zip(assigned_tickets.iter()) {
                        if let Err(e) = family.reset_lane(gpu, t.lane) {
                            retire_lane_after_reset_failure(
                                sched,
                                stdout,
                                k,
                                t.admission,
                                t.lane,
                                "partial assign rollback",
                                &e,
                            );
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
                if let Err(err) = family.reset_lane(gpu, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        family,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff =
                    match handoff_started_in_think(sched, lane_idx, &key, &pending_req, route) {
                        Ok(msg) => msg,
                        Err(reason) => {
                            return fail_all(sched, gpu, family, stdout, reason);
                        }
                    };
                inbox.push_front(handoff);
                break;
            }
            // Re-validate capacity at assignment time (defensive; lane_capacity is the source of truth).
            if batch_lfm_exceeds_capacity(prompt_tokens.len(), max_tokens_req, sched.lane_capacity)
            {
                if let Err(e) = family.reset_lane(gpu, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        family,
                        stdout,
                        format!("reset lane {lane_idx} on capacity re-check: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_lane_assignment_capacity_error(
                    route,
                    stdout,
                    &key,
                    prompt_tokens.len(),
                    max_tokens_req,
                    sched.lane_capacity,
                );
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            if let Err(e) = family.reset_lane(gpu, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    family,
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
                    if let Err(e) = family.reset_lane(gpu, lane_idx) {
                        retire_lane_after_reset_failure(
                            sched,
                            stdout,
                            &key,
                            admission,
                            lane_idx,
                            "mark_awaiting_commit",
                            &e,
                        );
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
                    if let Err(e) = family.reset_lane(gpu, lane_idx) {
                        retire_lane_after_reset_failure(
                            sched,
                            stdout,
                            &key,
                            admission,
                            lane_idx,
                            "commit_ready publish",
                            &e,
                        );
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
                if let Err(e) = family.reset_lane(gpu, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        family,
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
            // Prefill checks abort before and after its GPU work; an aborted
            // lane never samples and only that lane is reset.
            let prefill_is_aborted = {
                let abort_check = || batch_check_abort(&key.id, key.attempt_id, admission);
                family.open_lane(lane_idx, &pending_req);
                let res = family.prefill_lane(gpu, lane_idx, &prompt_tokens, &abort_check);
                match res {
                    Ok(true) => false,
                    Ok(false) => true,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            family,
                            stdout,
                            format!("prefill lane {lane_idx}: {e}"),
                        );
                    }
                }
            };
            if prefill_is_aborted || batch_check_abort(&key.id, key.attempt_id, admission) {
                if let Err(e) = family.reset_lane(gpu, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        family,
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
            let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => (lane.rng_state as u32, lane.sampling.clone()),
                _ => continue,
            };
            let (next_token, next_rng) =
                match family.sample_first(gpu, lane_idx, &sampling, lane_rng) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            family,
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
                    tokens[i] = lane.next_token.unwrap_or(idle);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = idle;
                    positions[i] = 0;
                }
            }
        }
        // Rows past the last running lane are dead weight; idle rows below it
        // are masked by the family.
        let rows = running.last().map_or(0, |&i| i + 1);
        if let Err(e) = family.decode(gpu, &tokens[..rows], &positions[..rows], &running) {
            return fail_all(sched, gpu, family, stdout, format!("lane decode: {e}"));
        }
        let repeat_cap = family.repeat_capacity();
        let mut draws: Vec<LaneDraw> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> = Vec::new();
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
            let max_tokens = lane_max_tokens(&key, sched);
            let lane_capacity = sched.lane_capacity;
            let BatchLane::Running(lane) = &mut sched.lanes[idx] else {
                continue;
            };
            let cur_token = lane.next_token.unwrap_or(idle);
            let step = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                family.emit(idx, lane, cur_token, stdout, &key.id)
            };
            let hit_length_cap = match step {
                LaneEmit::Stop => false,
                LaneEmit::Visible | LaneEmit::Hidden => {
                    if step == LaneEmit::Visible && lane.first_token_at.is_none() {
                        lane.first_token_at = Some(Instant::now());
                    }
                    lane.streamed_tokens.push(cur_token);
                    lane.seq_pos += 1;
                    let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
                    let hit_max = lane.streamed_tokens.len() >= max_tokens;
                    let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, lane_capacity);
                    if !batch_should_finish_decode(false, hit_max, hit_lane_cap, false, loop_hit) {
                        let window = lane.sampling.repeat_window.min(repeat_cap);
                        let start = lane.streamed_tokens.len().saturating_sub(window);
                        draws.push(LaneDraw {
                            lane: idx,
                            history: lane.streamed_tokens[start..].to_vec(),
                            sampling: lane.sampling.clone(),
                            rng: lane.rng_state as u32,
                        });
                        continue;
                    }
                    batch_hit_length_cap(hit_max, hit_lane_cap, false, false, loop_hit)
                }
            };
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let tool_calls = family.finish(idx, stdout, &key.id);
            let finish_reason = if hit_length_cap {
                "length"
            } else if !tool_calls.is_empty() {
                "tool_calls"
            } else {
                "stop"
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
                lane_capacity,
                lane.max_active_lanes.max(1),
            );
            hipfire_engine::emit::stage_terminal_tool_calls(
                &mut pending_done,
                finish_reason,
                &tool_calls,
            );
            hipfire_engine::emit::emit_tool_calls_event(stdout, &key.id, &tool_calls);
            to_await.push((idx, key.clone(), admission, pending_done));
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = family.reset_lane(gpu, idx) {
                return fail_all(
                    sched,
                    gpu,
                    family,
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
                if let Err(e) = family.reset_lane(gpu, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "mark_awaiting_commit",
                        &e,
                    );
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
                if let Err(e) = family.reset_lane(gpu, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "commit_ready publish",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                continue;
            }
        }
        if draws.is_empty() {
            continue;
        }
        let picks = match family.sample(gpu, &draws) {
            Ok(v) => v,
            Err(e) => return fail_all(sched, gpu, family, stdout, format!("lane sample: {e}")),
        };
        for (draw, (tok, rng)) in draws.iter().zip(picks) {
            if let BatchLane::Running(lane) = &mut sched.lanes[draw.lane] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}

/// The lane family of a staged model. Glimmer's lanes are seeded through
/// the singleton state, so its sequential session mirror is dropped here.
fn lane_family(m: &mut LoadedModel) -> Result<Box<dyn LaneFamily + '_>, String> {
    let LoadedModel {
        tokenizer,
        state,
        conversation_tokens,
        seq_pos,
        ..
    } = m;
    let tokenizer = tokenizer.as_ref().ok_or("tokenizer missing")?;
    let state = state.as_mut().ok_or("model state missing")?;
    let any = state.as_mut() as &mut dyn Any;
    if any.is::<MuseGlimmerBundle>() {
        let b = any.downcast_mut::<MuseGlimmerBundle>().unwrap();
        let lanes = b.lanes.as_ref().ok_or("glimmer lanes not staged")?;
        let n = lanes.max_batch;
        b.reset_session_state();
        conversation_tokens.clear();
        *seq_pos = 0;
        let stop_set = crate::dense::glimmer_stop_set(tokenizer, b.config.eos_token, b.eos_tok);
        return Ok(Box::new(GlimmerLaneFamily {
            b,
            tokenizer,
            stop_set,
            first_logits: Vec::new(),
            logits: Vec::new(),
            picks: Vec::new(),
            lanes: (0..n).map(|_| GlimmerLane::new(0)).collect(),
        }));
    }
    let b = any
        .downcast_mut::<Lfm2MoeBundle>()
        .ok_or("batch model is not a lane family")?;
    let Lfm2MoeBundle {
        config,
        weights,
        eos_tok,
        lfm2_decode_batch,
        ..
    } = b;
    let state = lfm2_decode_batch
        .as_mut()
        .ok_or("batch state not allocated")?;
    let n = state.max_batch;
    Ok(Box::new(LfmLanes {
        state,
        config,
        weights,
        tokenizer,
        eos: *eos_tok,
        // Stop set shared with crate::dense::generate_lfm2moe.
        stop_toks: hipfire_arch_lfm2moe::carrier::stop_tokens(tokenizer, *eos_tok),
        tool_open: tokenizer.special_token_id("\x3C\x7Ctool_call_start\x7C\x3E"),
        tool_close: tokenizer.special_token_id("\x3C\x7Ctool_call_end\x7C\x3E"),
        spans: (0..n).map(|_| LfmToolSpan::default()).collect(),
    }))
}

/// LFM tool-call span suppression (mirrors dense::generate_lfm2moe): the
/// span — markers + body — is held back from the stream and parsed into
/// `calls` at the lane's finish.
#[derive(Default)]
struct LfmToolSpan {
    open: bool,
    span: Vec<u32>,
    body: Vec<u32>,
    calls: Vec<ToolCall>,
}

struct LfmLanes<'a> {
    state: &'a mut Lfm2DecodeBatchState,
    config: &'a hipfire_arch_lfm2moe::config::Lfm2MoeConfig,
    weights: &'a hipfire_arch_lfm2moe::Lfm2MoeWeights,
    tokenizer: &'a Tokenizer,
    eos: u32,
    stop_toks: Vec<u32>,
    tool_open: Option<u32>,
    tool_close: Option<u32>,
    spans: Vec<LfmToolSpan>,
}

impl LaneFamily for LfmLanes<'_> {
    fn route(&self) -> GenerationRoute {
        GenerationRoute::LfmAr
    }
    fn idle_token(&self) -> u32 {
        self.eos
    }
    fn repeat_capacity(&self) -> usize {
        self.state.sample_repeat_capacity
    }
    fn reset(&mut self, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        self.state.reset(gpu).map_err(|e| e.to_string())
    }
    fn reset_lane(&mut self, gpu: &mut rdna_compute::Gpu, lane: usize) -> Result<(), String> {
        self.state
            .reset_lane(gpu, self.config, lane)
            .map_err(|e| e.to_string())
    }
    fn open_lane(&mut self, lane: usize, _req: &BatchPendingRequest) {
        self.spans[lane] = LfmToolSpan::default();
    }
    fn prefill_lane(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        lane: usize,
        prompt: &[u32],
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if abort() {
            return Ok(false);
        }
        self.state
            .prefill_lane(gpu, self.weights, self.config, lane, prompt)
            .map_err(|e| e.to_string())?;
        Ok(!abort())
    }
    fn batched_prefill(&self) -> bool {
        true
    }
    fn per_lane_sampling(&self) -> bool {
        // `sample_product` takes one config per batch: lanes share a cohort.
        false
    }
    fn prefill_lanes(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        prompts: &[&[u32]],
    ) -> Result<(), String> {
        self.state
            .prefill_lanes_batched(gpu, self.weights, self.config, prompts)
            .map_err(|e| e.to_string())
    }
    fn sample_first(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        lane: usize,
        s: &BatchSampling,
        rng: u32,
    ) -> Result<(u32, u32), String> {
        self.state
            .sample_lane_product(
                gpu,
                self.config,
                lane,
                &[],
                s.temp,
                s.top_p,
                s.top_k,
                s.min_p,
                rng,
                s.repeat_penalty,
                s.presence_penalty,
                s.frequency_penalty,
            )
            .map_err(|e| e.to_string())
    }
    fn decode(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        positions: &[usize],
        _running: &[usize],
    ) -> Result<(), String> {
        hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm(
            gpu,
            self.weights,
            self.config,
            tokens,
            positions,
            self.state,
        )
        .map_err(|e| e.to_string())
    }
    fn sample(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        draws: &[LaneDraw],
    ) -> Result<Vec<(u32, u32)>, String> {
        let (n, cap) = (self.state.max_batch, self.state.sample_repeat_capacity);
        let mut repeat_tokens = vec![0u32; n * cap];
        let mut repeat_lengths = vec![0u32; n];
        let mut rng_states = vec![0u32; n];
        for d in draws {
            repeat_tokens[d.lane * cap..][..d.history.len()].copy_from_slice(&d.history);
            repeat_lengths[d.lane] = d.history.len() as u32;
            rng_states[d.lane] = d.rng;
        }
        // ponytail: one sampling config for the whole batch (the first
        // draw's), as before the shared driver; per-lane configs need a
        // per-lane sampler.
        let s = &draws[0].sampling;
        let sampled = self
            .state
            .sample_product(
                gpu,
                self.config,
                n,
                &repeat_tokens,
                &repeat_lengths,
                &rng_states,
                s.temp,
                s.top_p,
                s.top_k,
                s.min_p,
                s.repeat_penalty,
                s.presence_penalty,
                s.frequency_penalty,
            )
            .map_err(|e| e.to_string())?;
        Ok(draws.iter().map(|d| sampled[d.lane]).collect())
    }
    fn emit(
        &mut self,
        lane: usize,
        state: &mut QwenBatchLane,
        token: u32,
        stdout: &mut std::io::Stdout,
        id: &str,
    ) -> LaneEmit {
        if self.stop_toks.contains(&token) {
            return LaneEmit::Stop;
        }
        let span = &mut self.spans[lane];
        let mut span_diverted = false;
        if self.tool_open == Some(token) && !span.open {
            span.open = true;
            span.span.clear();
            span.span.push(token);
            span.body.clear();
            span_diverted = true;
        } else if span.open {
            span.span.push(token);
            if self.tool_close == Some(token) {
                span.open = false;
                let body_text = self.tokenizer.decode(&span.body);
                span.calls
                    .extend(hipfire_runtime::emit_text::parse_lfm_tool_call_body(
                        &body_text,
                    ));
            } else {
                span.body.push(token);
            }
            span_diverted = true;
        }
        let mut future_streamed = state.streamed_tokens.clone();
        future_streamed.push(token);
        let all_bytes = self.tokenizer.decode_bytes(&future_streamed);
        let valid_len = match std::str::from_utf8(&all_bytes) {
            Ok(_) => all_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        let prev_fed = state.bytes_fed_to_filter.min(valid_len);
        let frag = std::str::from_utf8(&all_bytes[prev_fed..valid_len]).unwrap_or("");
        // String guard: leaked EOS-class strings not registered as specials.
        if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
            return LaneEmit::Stop;
        }
        state.bytes_fed_to_filter = valid_len;
        if !frag.is_empty() && !span_diverted {
            emit_visible_token(stdout, id, frag);
            LaneEmit::Visible
        } else {
            LaneEmit::Hidden
        }
    }
    fn finish(&mut self, lane: usize, stdout: &mut std::io::Stdout, id: &str) -> Vec<ToolCall> {
        let span = std::mem::take(&mut self.spans[lane]);
        // Unclosed span at the terminal: re-emit the buffered raw text.
        if span.open {
            emit_visible_token(stdout, id, &self.tokenizer.decode(&span.span));
        }
        span.calls
    }
}

/// Per-lane Harmony routing, as the singleton `generate_muse_glimmer` AR loop.
struct GlimmerLane {
    text: TokenTextStream,
    router: crate::dense::GlimmerHarmonyRouter,
    tool_acc: String,
    rng: Xorshift,
    sampled: bool,
}

impl GlimmerLane {
    fn new(max_think_tokens: usize) -> Self {
        Self {
            text: TokenTextStream::new(),
            router: crate::dense::GlimmerHarmonyRouter::new(max_think_tokens),
            tool_acc: String::new(),
            rng: Xorshift::new(0),
            sampled: false,
        }
    }

    fn route(
        &mut self,
        events: Vec<crate::dense::GlimmerEmit>,
        stdout: &mut std::io::Stdout,
        id: &str,
    ) -> bool {
        let mut streamed = false;
        for ev in events {
            match ev {
                crate::dense::GlimmerEmit::Reasoning(t) => {
                    streamed = true;
                    emit_reasoning_token(stdout, id, &t)
                }
                crate::dense::GlimmerEmit::Token(t) => {
                    streamed = true;
                    emit_visible_token(stdout, id, &t)
                }
                crate::dense::GlimmerEmit::Tool(t) => self.tool_acc.push_str(&t),
            }
        }
        streamed
    }
}

/// Glimmer lanes: seeded by the singleton prefill plus a KV import, then
/// decoded together on [`hipfire_arch_muse_glimmer::lanes::GlimmerLanes`].
struct GlimmerLaneFamily<'a> {
    b: &'a mut MuseGlimmerBundle,
    tokenizer: &'a Tokenizer,
    stop_set: Vec<u32>,
    /// Last-row logits of the latest lane prefill.
    first_logits: Vec<f32>,
    /// `rows * vocab` logits of the last decode, fetched only for sampled lanes.
    logits: Vec<f32>,
    picks: Vec<u32>,
    lanes: Vec<GlimmerLane>,
}

impl GlimmerLaneFamily<'_> {
    fn draw(lane: &mut GlimmerLane, logits: &[f32], s: &BatchSampling) -> u32 {
        let top_k = s.top_k.map_or(0, |k| k as usize);
        sample_token(logits, s.temp, top_k, s.top_p, &mut lane.rng)
    }
}

impl LaneFamily for GlimmerLaneFamily<'_> {
    fn route(&self) -> GenerationRoute {
        GenerationRoute::GlimmerAr
    }
    fn idle_token(&self) -> u32 {
        self.b.eos_tok
    }
    fn repeat_capacity(&self) -> usize {
        // The singleton Glimmer route applies no repeat penalties either.
        0
    }
    fn reset(&mut self, _gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
        self.b.reset_session_state();
        Ok(())
    }
    fn reset_lane(&mut self, _gpu: &mut rdna_compute::Gpu, _lane: usize) -> Result<(), String> {
        // A lane's KV is overwritten by its next import and read only below
        // its position, so there is nothing to clear.
        Ok(())
    }
    fn open_lane(&mut self, lane: usize, req: &BatchPendingRequest) {
        self.lanes[lane] = GlimmerLane::new(req.max_think_tokens);
    }
    fn prefill_lane(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        lane: usize,
        prompt: &[u32],
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if abort() {
            return Ok(false);
        }
        let b = &mut *self.b;
        b.reset_session_state();
        self.first_logits = hipfire_arch_muse_glimmer::forward::prefill_with_capture(
            &b.config,
            &b.weights,
            &mut b.state,
            gpu,
            prompt,
            0,
            &[],
            &mut b.target_hidden_host,
        )?;
        let lanes = b.lanes.as_ref().ok_or("glimmer lanes not staged")?;
        lanes.import(gpu, &b.config, &b.state, lane, prompt.len())?;
        Ok(!abort())
    }
    fn batched_prefill(&self) -> bool {
        false
    }
    fn per_lane_sampling(&self) -> bool {
        true
    }
    fn prefill_lanes(
        &mut self,
        _gpu: &mut rdna_compute::Gpu,
        _prompts: &[&[u32]],
    ) -> Result<(), String> {
        Err("glimmer lanes have no batched prefill".into())
    }
    fn sample_first(
        &mut self,
        _gpu: &mut rdna_compute::Gpu,
        lane: usize,
        s: &BatchSampling,
        rng: u32,
    ) -> Result<(u32, u32), String> {
        let l = &mut self.lanes[lane];
        l.sampled = s.temp > 0.0;
        l.rng = Xorshift::new(rng as u64);
        Ok((Self::draw(l, &self.first_logits, s), rng))
    }
    fn decode(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        tokens: &[u32],
        positions: &[usize],
        running: &[usize],
    ) -> Result<(), String> {
        let mask = running.iter().fold(0u64, |m, &r| m | 1 << r);
        let want_logits = running.iter().any(|&r| self.lanes[r].sampled);
        let b = &*self.b;
        let lanes = b.lanes.as_ref().ok_or("glimmer lanes not staged")?;
        self.picks = lanes.decode(
            gpu,
            &b.config,
            &b.weights,
            &b.state,
            tokens,
            positions,
            mask,
            want_logits.then_some(&mut self.logits),
        )?;
        Ok(())
    }
    fn sample(
        &mut self,
        _gpu: &mut rdna_compute::Gpu,
        draws: &[LaneDraw],
    ) -> Result<Vec<(u32, u32)>, String> {
        let vocab = self.b.config.vocab_size;
        Ok(draws
            .iter()
            .map(|d| {
                let l = &mut self.lanes[d.lane];
                let tok = if l.sampled {
                    Self::draw(l, &self.logits[d.lane * vocab..][..vocab], &d.sampling)
                } else {
                    self.picks[d.lane]
                };
                (tok, d.rng)
            })
            .collect())
    }
    fn emit(
        &mut self,
        lane: usize,
        _state: &mut QwenBatchLane,
        token: u32,
        stdout: &mut std::io::Stdout,
        id: &str,
    ) -> LaneEmit {
        let l = &mut self.lanes[lane];
        let frag = l.text.push(self.tokenizer, token);
        let (events, should_stop) = l.router.push(&frag);
        let visible = l.route(events, stdout, id);
        // eos/eot always stop; `<|eom|>` stops through the router.
        if should_stop || self.stop_set.contains(&token) {
            LaneEmit::Stop
        } else if visible {
            LaneEmit::Visible
        } else {
            LaneEmit::Hidden
        }
    }
    fn finish(&mut self, lane: usize, stdout: &mut std::io::Stdout, id: &str) -> Vec<ToolCall> {
        let l = &mut self.lanes[lane];
        let tail = l.text.flush();
        let mut events = l.router.push(&tail).0;
        events.extend(l.router.flush());
        l.route(events, stdout, id);
        crate::dense::parse_glimmer_atem(&l.tool_acc).unwrap_or_default()
    }
}
