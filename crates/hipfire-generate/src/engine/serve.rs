// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The daemon side of the engine's N-request scheduler: the staged
//! continuous-batch admissions (`ContinuousBatchScheduler`) feed
//! [`Scheduler`], whose ticks run on the load's one-owner mesh through its
//! family ([`MeshFamily`]) and the family's lanes ([`MeshLanes`]). Every
//! request keeps the singleton engine route's contract: its own RNG, the
//! batch of one's prompt cadence and session restore on the resident state,
//! `loop: engine` on its terminal. Between ticks the driver settles client
//! terminals, aborts queued requests, drains the inbox and admits arrivals.
//!
//! A request that finished on the resident state keeps it until the client
//! commits its terminal: the commit then keeps its conversation live in the
//! session cache, as the singleton does, and moves it into a lane record. A
//! request that finished in a lane keeps that lane as its conversation's
//! record once committed ([`LaneRecord`]); a cancel frees it.

use super::chatml::{ChatMl, ChatMlStream, Host, WireProtocol};
use super::gemma::{Gemma4Stream, Gemma4Wire};
use super::sched::{Batch, Ended, Place, Scheduler};
use super::spec::{DraftForwards, Plan, SpecDone, SpecRequest};
use super::{admit_budget, DoneMetrics, Exit, Finish, Next, RequestRun, Verdict};
use crate::ar::*;
use crate::batch::{drain_qwen_batch_inbox, is_vmm_batch_request_eligible};
use crate::common::{
    emit_fail_closed_error_for_route, fit_max_tokens_if, RollbackEpilogue,
    UNATTESTED_CANCEL_MESSAGE,
};
use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_loader::{AsstTurnCache, LoadedModel};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::mesh_family::{
    DecodeOut, LaneRecord, MeshFamily, MeshLanes, MeshRequest, MeshRunner,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::Speculator;
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::{Stdout, Write};
use std::ops::Range;
use std::time::{Duration, Instant};

/// Serve the staged continuous batch of `route` (Qwen4 AR or Gemma 4 AR) on
/// the engine loop until no request is queued or live. `Err`: a
/// driver-level failure; every request it held got its terminal.
pub fn drive(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    m: &mut LoadedModel,
    params: VmmBatchParams,
    route: GenerationRoute,
    stdout: &mut Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let _serving = m.ple_serving_guard();
    // Every RequestRun samples with its own config and RNG: no cohort gate.
    sched.per_lane_sampling = true;
    let n_layers = m
        .state
        .as_deref_mut()
        .and_then(|state| state.mesh_family())
        .map(|family| family.n_layers())
        .ok_or("the model declares no engine family")
        .map_err(|e| BatchDriveError::Gpu(e.into()))?;
    let mut driver = Driver {
        lives: (0..sched.max_batch).map(|_| None).collect(),
        core: Scheduler::default(),
        max_batch_tokens: params.max_batch_tokens,
        route,
        gpu_arch: gpu.arch.clone(),
        poisoned: false,
    };
    let result = super::run::on_one_owner(m, gpu, n_layers, |m, runner, gpus| {
        let result = driver.serve(sched, m, runner, gpus, stdout, inbox);
        if let Err(error) = &result {
            driver.fail_all(sched, m, runner, gpus, stdout, error);
        }
        result
    })
    .and_then(|result| result);
    // An unattested rollback poisons the batch: the daemon drops the scheduler.
    result.map_err(|error| match driver.poisoned {
        true => BatchDriveError::Poisoned(error),
        false => BatchDriveError::Gpu(error),
    })
}

/// One admitted request's host side.
struct Live {
    key: AttemptKey,
    admission: BatchGeneration,
    prompt: Vec<u32>,
    budget: usize,
    sampler: SamplerConfig,
    seed: u32,
    wire: Wire,
    /// Lane identity (unique per request).
    tag: u64,
    t0: Instant,
    reused: usize,
    prompt_done: Option<Instant>,
    /// Its outputs while it is on the resident state.
    request: Option<MeshRequest>,
    /// Finished, its state still where it ran (`Resident` or `Lane`) until
    /// the terminal settles.
    held: Option<Place>,
    /// The committed conversation and stream kept past `commit_ready`.
    history: Vec<u32>,
    streamed: Vec<u32>,
    /// A speculative request: its drafter's configuration and `done` fields.
    spec: Option<SpecRequest>,
    /// The session route its drafter resumes on.
    spec_route: Option<SessionRoute>,
    /// An unrepaired terminal window reset its live state.
    invalidated: bool,
    /// A speculative request in a window lane: its drafter's `done` fields
    /// at the park, its lane windows counted on.
    lane_spec: Option<SpecDone>,
}

/// A request's wire protocol and its stream state, as its route declares it.
enum Wire {
    /// Qwen4 AR.
    ChatMl {
        started_in_think: bool,
        stop: Vec<String>,
        min_tokens: usize,
        max_think_tokens: usize,
        eos: u32,
        stream: Box<ChatMlStream>,
    },
    /// Gemma 4 AR.
    Gemma {
        enable_thinking: bool,
        max_think_tokens: usize,
        stop: Vec<u32>,
        stream: Gemma4Stream,
    },
}

impl Wire {
    fn emit(
        &mut self,
        host: &mut Host<'_>,
        streamed: &mut Vec<u32>,
        token: u32,
    ) -> Result<(Verdict, Vec<u32>), String> {
        match self {
            Wire::ChatMl {
                started_in_think,
                stop,
                min_tokens,
                max_think_tokens,
                eos,
                stream,
            } => {
                let spec = ChatMl {
                    started_in_think: *started_in_think,
                    tools: false,
                    stop,
                    min_tokens: *min_tokens,
                    max_think_tokens: *max_think_tokens,
                    eos: std::slice::from_ref(eos),
                };
                stream.emit(&spec, host, streamed, token)
            }
            Wire::Gemma {
                enable_thinking,
                max_think_tokens,
                stop,
                stream,
            } => {
                let spec = Gemma4Wire {
                    enable_thinking: *enable_thinking,
                    max_think_tokens: *max_think_tokens,
                    tools: false,
                    stop,
                };
                let out = stream.emit(&spec, host, token)?;
                if out.0 != Verdict::Eos {
                    streamed.push(token);
                }
                Ok(out)
            }
        }
    }

    /// [`super::Emit::window_cap`] of the request's stream.
    fn window_cap(&self) -> Option<usize> {
        match self {
            Wire::ChatMl {
                started_in_think,
                stream,
                ..
            } => stream.window_cap(*started_in_think),
            Wire::Gemma { .. } => None,
        }
    }

    fn finish(
        &mut self,
        out: &mut Stdout,
        id: &str,
        length: bool,
    ) -> Result<(&'static str, Vec<hipfire_runtime::prompt_frame::ToolCall>), String> {
        match self {
            Wire::ChatMl { stream, .. } => stream.finish(out, length),
            Wire::Gemma { stream, .. } => stream.finish(out, id, length),
        }
    }

    /// After the client committed: store the turn for reuse (ChatML only).
    fn commit_turn(&self, tokenizer: &Tokenizer, streamed: &[u32], cache: &mut AsstTurnCache) {
        if let Wire::ChatMl {
            started_in_think,
            stop,
            min_tokens,
            max_think_tokens,
            eos,
            stream,
        } = self
        {
            let spec = ChatMl {
                started_in_think: *started_in_think,
                tools: false,
                stop,
                min_tokens: *min_tokens,
                max_think_tokens: *max_think_tokens,
                eos: std::slice::from_ref(eos),
            };
            stream.commit_turn(&spec, tokenizer, streamed, cache);
        }
    }

    fn stamp_done(&self, done: &mut serde_json::Value, metrics: &DoneMetrics) {
        if let Wire::Gemma {
            enable_thinking,
            max_think_tokens,
            stop,
            ..
        } = self
        {
            Gemma4Wire {
                enable_thinking: *enable_thinking,
                max_think_tokens: *max_think_tokens,
                tools: false,
                stop,
            }
            .stamp_done(done, metrics);
        }
    }
}

struct Driver {
    lives: Vec<Option<Live>>,
    core: Scheduler,
    max_batch_tokens: usize,
    route: GenerationRoute,
    /// The device's arch (a drafter's n-gram takeover is per GPU).
    gpu_arch: String,
    /// A rollback was not attested: the resident state must never serve again.
    poisoned: bool,
}

/// One tick's view of the model, the devices and the requests.
struct Ctx<'a> {
    sched: &'a mut ContinuousBatchScheduler,
    lives: &'a mut [Option<Live>],
    runner: &'a mut MeshRunner,
    state: &'a mut dyn ArchModel,
    gpus: &'a mut Gpus,
    tokenizer: &'a Tokenizer,
    stdout: &'a mut Stdout,
    conversation: &'a mut Vec<u32>,
    seq_pos: &'a mut usize,
    asst_turn_cache: &'a mut AsstTurnCache,
    spec_deferred_seed: &'a mut Option<hipfire_loader::DeferredSpecSeed>,
    /// The load's drafter, while no speculative request owns it.
    speculator: &'a mut Option<Box<dyn Speculator>>,
    max_batch_tokens: usize,
    route: GenerationRoute,
}

static NEXT_TAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Driver {
    fn serve(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        runner: &mut MeshRunner,
        gpus: &mut Gpus,
        stdout: &mut Stdout,
        inbox: &mut DaemonInbox,
    ) -> Result<(), String> {
        loop {
            if let Some(reason) = runner.quarantined() {
                return Err(format!(
                    "engine mesh quarantined; process restart required: {reason}"
                ));
            }
            self.settle_terminals(sched, m, runner, gpus, stdout)?;
            self.arrivals(sched, m, stdout, inbox)?;
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
            if !self.core.busy() {
                // Only terminals awaiting their client are left.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let mut ended = Vec::new();
            if let Err(error) = self.tick(sched, m, runner, gpus, stdout, inbox, &mut ended) {
                // Requests that ended before the error and were not settled
                // hand their drafter back here: `fail_all` only reaches live
                // runs.
                for drafting in ended.into_iter().filter_map(|ended| ended.drafting) {
                    m.speculator.get_or_insert(drafting.drafter);
                }
                return Err(error);
            }
        }
    }

    /// One tick: the prompt epochs, with requests that arrive while the
    /// resident state is free joining its burst, then the lane epoch; every
    /// request that ended is settled (and leaves `ended`).
    #[allow(clippy::too_many_arguments)]
    fn tick(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        runner: &mut MeshRunner,
        gpus: &mut Gpus,
        stdout: &mut Stdout,
        inbox: &mut DaemonInbox,
        ended: &mut Vec<Ended>,
    ) -> Result<(), String> {
        loop {
            let mut ctx = Ctx::of(
                &mut self.lives,
                self.max_batch_tokens,
                self.route,
                sched,
                m,
                runner,
                gpus,
                stdout,
            )?;
            ended.extend(self.core.prompts(&mut ctx));
            if !self.core.takes_arrivals() || !self.arrivals(sched, m, stdout, inbox)? {
                break;
            }
        }
        let mut ctx = Ctx::of(
            &mut self.lives,
            self.max_batch_tokens,
            self.route,
            sched,
            m,
            runner,
            gpus,
            stdout,
        )?;
        ended.extend(self.core.lanes_step(&mut ctx));
        let mut attested = Ok(());
        for ended in ended.drain(..) {
            attested = attested.and(ctx.settle(&mut self.core, ended));
        }
        attested.inspect_err(|_| self.poisoned = true)
    }

    /// Cancel the queued requests the client aborted, drain the inbox and
    /// admit what the batch scheduler assigns. Whether it admitted any.
    fn arrivals(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        stdout: &mut Stdout,
        inbox: &mut DaemonInbox,
    ) -> Result<bool, String> {
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            let Some(admission) = sched.pending.get(&key).map(|req| req.admission) else {
                continue;
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_generation_start(self.route, stdout, &key.id, false);
                // Never admitted: a pre-admission cancel is legacy, as at the inbox drain.
                emit_generation_cancel(self.route, stdout, &key.id, 0);
                sched.abort_queued(&key, admission);
            }
        }
        let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer missing")?;
        let batch_size = sched.max_batch;
        let barrier = drain_qwen_batch_inbox(
            sched,
            m,
            batch_size,
            is_vmm_batch_request_eligible,
            tokenizer,
            m.chat_template.as_ref(),
            self.route,
            stdout,
            inbox,
            true,
        )?;
        if let Some(barrier) = barrier {
            inbox.push_front(barrier);
        }
        let mut admitted = false;
        while let Some((key, ticket)) = sched.try_assign_one() {
            self.admit(sched, m, stdout, key, ticket)?;
            admitted = true;
        }
        Ok(admitted)
    }

    /// Render and admit `key` (its scheduler lane `ticket.lane` is Running);
    /// a refusal is its terminal.
    fn admit(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        stdout: &mut Stdout,
        key: AttemptKey,
        ticket: LaneTicket,
    ) -> Result<(), String> {
        let idx = ticket.lane;
        let req = sched
            .pending
            .get(&key)
            .cloned()
            .ok_or("assigned request missing")?;
        let _scope =
            BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, ticket.admission);
        let route = self.route;
        let refuse = |sched: &mut ContinuousBatchScheduler,
                      stdout: &mut Stdout,
                      message: &str,
                      class: &str| {
            emit_generation_error(route, stdout, Some(&key.id), message, class, false, true);
            let _ = sched.abort_lane(idx, &key, ticket.admission);
        };
        let rendered = match route {
            GenerationRoute::Gemma4Ar => {
                super::gemma::render_request(m, &req).map(|ids| (ids, false))
            }
            _ => crate::fn_batch::render_request(m, &req),
        };
        let (prompt, started_in_think) = match rendered {
            Ok(rendered) => rendered,
            Err((message, class)) => {
                refuse(sched, stdout, &message, class);
                return Ok(());
            }
        };
        if prompt.is_empty() {
            let message = format!("{} prompt rendered to zero tokens", route.name());
            refuse(sched, stdout, &message, "validation");
            return Ok(());
        }
        let fit = req
            .original_msg
            .get("max_tokens_fit")
            .and_then(|v| v.as_bool())
            == Some(true);
        let budget = match admit_budget(prompt.len(), req.max_tokens, m.max_seq, |n, p, s| {
            // Gemma 4 clamps to the context left (intended diff: 2 tokens short of legacy at the cap).
            if route == GenerationRoute::Gemma4Ar {
                n.min(s.saturating_sub(p))
            } else {
                fit_max_tokens_if(fit, n, p, s)
            }
        }) {
            Ok(budget) => budget,
            Err(message) => {
                let class = if req.max_tokens == 0 {
                    "validation"
                } else {
                    "context_length"
                };
                refuse(sched, stdout, &message, class);
                return Ok(());
            }
        };
        let wire = if route == GenerationRoute::Gemma4Ar {
            let spec = Gemma4Wire {
                enable_thinking: super::gemma::enable_thinking(&req),
                max_think_tokens: req.max_think_tokens,
                tools: false,
                stop: &[],
            };
            Wire::Gemma {
                stream: Gemma4Stream::new(&spec),
                enable_thinking: spec.enable_thinking,
                max_think_tokens: spec.max_think_tokens,
                stop: crate::dense::gemma4_stop_set(m),
            }
        } else {
            let min_tokens = hipfire_runtime::stop_sequence::parse_min_tokens_field(
                req.original_msg.get("min_tokens"),
                req.max_tokens,
            );
            let stop =
                hipfire_runtime::stop_sequence::parse_stop_field(req.original_msg.get("stop"))
                    .map_err(|e| e.to_string());
            let (min_tokens, stop) = match (min_tokens, stop) {
                (Ok(min_tokens), Ok(stop)) => (min_tokens, stop),
                (Err(message), _) | (_, Err(message)) => {
                    refuse(sched, stdout, &message, "validation");
                    return Ok(());
                }
            };
            let eos = m
                .qwen4()
                .map(|bundle| bundle.config.eos_token_id)
                .ok_or("engine lanes need a Qwen4 bundle")?;
            let stream = ChatMlStream::new(
                &ChatMl {
                    started_in_think,
                    tools: false,
                    stop: &stop,
                    min_tokens,
                    max_think_tokens: req.max_think_tokens,
                    eos: &[eos],
                },
                &key.id,
            );
            Wire::ChatMl {
                started_in_think,
                stop,
                min_tokens,
                max_think_tokens: req.max_think_tokens,
                eos,
                stream: Box::new(stream),
            }
        };
        let s = &req.sampling;
        let sampler = SamplerConfig {
            temperature: s.temp,
            top_p: s.top_p,
            repeat_penalty: s.repeat_penalty,
            repeat_window: s.repeat_window,
            presence_penalty: s.presence_penalty,
            frequency_penalty: s.frequency_penalty,
            blocked_tokens: Vec::new(),
            top_k: s.top_k,
            min_p: s.min_p,
        };
        let seed = request_seed_for(&key, req.client_seed);
        let stops = matches!(&wire, Wire::ChatMl { stop, .. } if !stop.is_empty());
        let spec = speculative(
            m,
            route,
            &req,
            &sampler,
            seed,
            prompt.len(),
            budget,
            stops,
            &self.gpu_arch,
            // A running speculative request holds the load's drafter.
            m.speculator.as_deref().or(self.core.drafter()),
        );
        let tag = NEXT_TAG
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| "engine lane tags are exhausted")?;
        emit_generation_start(route, stdout, &key.id, started_in_think);
        if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
            lane.prompt_len = prompt.len();
        }
        let live = Live {
            seed,
            key,
            admission: ticket.admission,
            prompt,
            budget,
            sampler,
            wire,
            tag,
            t0: Instant::now(),
            reused: 0,
            prompt_done: None,
            request: None,
            held: None,
            history: Vec::new(),
            streamed: Vec::new(),
            spec,
            spec_route: None,
            invalidated: false,
            lane_spec: None,
        };
        self.lives[idx] = Some(live);
        self.core.admit(idx);
        Ok(())
    }

    /// Settle every terminal awaiting its client: a commit emits `done` (and
    /// keeps a resident request's conversation live), an abort or a deadline
    /// cancels it.
    fn settle_terminals(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        runner: &mut MeshRunner,
        gpus: &mut Gpus,
        stdout: &mut Stdout,
    ) -> Result<(), String> {
        for idx in 0..sched.max_batch {
            let BatchLane::AwaitingClient(t) = &sched.lanes[idx] else {
                continue;
            };
            let (key, admission, done) =
                (t.key.clone(), t.ticket.admission, t.pending_done.clone());
            let commit = if batch_check_abort(&key.id, key.attempt_id, admission)
                || Instant::now() >= t.deadline
            {
                false
            } else if batch_poll_decision(&key.id, key.attempt_id, admission)
                == Some(ClientTerminalDecision::Commit)
            {
                true
            } else {
                continue;
            };
            let live = self.lives[idx]
                .take()
                .ok_or("awaiting terminal has no request")?;
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            if !commit {
                match live.held {
                    Some(Place::Resident) => {
                        let LoadedModel {
                            state,
                            asst_turn_cache,
                            spec_deferred_seed,
                            speculator,
                            ..
                        } = &mut *m;
                        let state = state.as_deref_mut().ok_or("the model has no state")?;
                        // A finished speculative turn the client left: a
                        // target that keeps sessions keeps it live, as the
                        // legacy cancel does; else target and drafter reset.
                        let fed = if live.invalidated {
                            0
                        } else {
                            live.history.len()
                        };
                        let kept = live.spec.is_some()
                            && super::spec::keep_fed(
                                state,
                                gpus,
                                &live.history,
                                fed,
                                live.prompt.len(),
                            );
                        // A finished AR turn the client left rewinds in a lane.
                        let parked = if live.spec.is_none() {
                            state
                                .mesh_family()
                                .and_then(|family| family.mesh_lanes())
                                .and_then(|lanes| {
                                    park_cancelled(
                                        lanes,
                                        runner,
                                        gpus,
                                        live.tag,
                                        &live.prompt,
                                        &live.history,
                                        &mut None,
                                    )
                                })
                        } else {
                            None
                        };
                        let released = if kept {
                            Ok(())
                        } else if let Some(parked) = parked {
                            parked
                        } else {
                            match (live.spec.is_some(), speculator.as_deref_mut()) {
                                (true, Some(drafter)) => reset_drafting(state, gpus, drafter, true),
                                (true, None) => Err("the drafter did not come back".into()),
                                (false, _) => Ok(()),
                            }
                            .and_then(|()| {
                                rollback(
                                    state,
                                    runner,
                                    gpus,
                                    asst_turn_cache,
                                    spec_deferred_seed,
                                    None,
                                )
                            })
                        };
                        let generated =
                            done.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let cancelled = cancel(self.route, stdout, &key.id, generated, released);
                        let _ = sched.abort_lane(idx, &key, admission);
                        // An unattested state is never handed to the next request.
                        cancelled.inspect_err(|_| self.poisoned = true)?;
                        self.core.release_resident();
                        continue;
                    }
                    _ => {
                        if let Some(lanes) = lanes_of(m) {
                            cancel_lane(lanes, &mut runner.records, live.tag, &live.prompt);
                        }
                    }
                }
                let generated = done.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                emit_engine_cancel(self.route, stdout, &key.id, generated);
                let _ = sched.abort_lane(idx, &key, admission);
                continue;
            }
            if !sched.commit_lane_retain_terminal(idx, &key, admission) {
                emit_generation_error(
                    self.route,
                    stdout,
                    Some(&key.id),
                    "engine commit_lane failed",
                    "internal",
                    false,
                    false,
                );
                let _ = sched.abort_lane(idx, &key, admission);
                continue;
            }
            let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer disappeared")?;
            live.wire
                .commit_turn(tokenizer, &live.streamed, &mut m.asst_turn_cache);
            let keeps_records = lanes_of(m).is_some_and(|lanes| lanes.keeps_records());
            match live.held {
                Some(Place::Resident) => {
                    // The singleton's commit, then the conversation moves into
                    // a lane record so another conversation's turn on the
                    // resident state does not drop it. An unrepaired terminal
                    // window left no live state: its snapshots only.
                    let kept: &[u32] = if live.invalidated { &[] } else { &live.history };
                    let state = m.state.as_deref_mut().ok_or("the model has no state")?;
                    state.session_commit_live_mesh(gpus, kept);
                    // A resident speculative turn keeps no lane record: a
                    // record parks through the AR `park`, which does not move
                    // the drafter's state (only `park_window` does).
                    if keeps_records && !live.invalidated && live.spec.is_none() {
                        if let Some(lanes) = lanes_of(m) {
                            keep_record(lanes, runner, gpus, live.tag, &live.history);
                        }
                    }
                    self.core.release_resident();
                }
                _ if keeps_records => runner.records.push(LaneRecord {
                    tag: live.tag,
                    tokens: live.history,
                }),
                _ => {
                    if let Some(lanes) = lanes_of(m) {
                        lanes.free_lane(live.tag);
                    }
                }
            }
            emit_generation_done_value(self.route, stdout, &done);
            batch_clear_terminal_at_generation(&key.id, key.attempt_id, admission);
        }
        Ok(())
    }

    /// A driver-level failure: every request it holds gets its terminal.
    fn fail_all(
        &mut self,
        sched: &mut ContinuousBatchScheduler,
        m: &mut LoadedModel,
        runner: &mut MeshRunner,
        gpus: &mut Gpus,
        stdout: &mut Stdout,
        error: &str,
    ) {
        // A speculative request's drafter goes back to the load.
        for drafting in self.core.take_drafting() {
            m.speculator.get_or_insert(drafting.drafter);
        }
        for idx in 0..self.lives.len() {
            let Some(live) = self.lives[idx].take() else {
                continue;
            };
            if let Some(request) = live.request {
                let _ = runner.end(request, gpus);
            }
            if let Some(lanes) = m
                .state
                .as_deref_mut()
                .and_then(|state| state.mesh_family())
                .and_then(|family| family.mesh_lanes())
            {
                lanes.free_lane(live.tag);
            }
            let _scope = BatchAttemptScope::enter_for_generation(
                &live.key.id,
                live.key.attempt_id,
                live.admission,
            );
            emit_generation_error(
                self.route,
                stdout,
                Some(&live.key.id),
                error,
                "gpu",
                false,
                false,
            );
        }
        for (key, req) in &sched.pending {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, req.admission);
            emit_generation_error(
                self.route,
                stdout,
                Some(&key.id),
                error,
                "gpu",
                false,
                false,
            );
        }
        sched.fail_all_active();
    }
}

/// Free every conversation the engine keeps in a lane (a full reset).
pub fn forget_records(m: &mut LoadedModel) {
    let Some(records) = m
        .engine
        .as_mut()
        .map(|runner| std::mem::take(&mut runner.records))
    else {
        return;
    };
    if let Some(lanes) = lanes_of(m) {
        for record in records {
            lanes.free_lane(record.tag);
        }
    }
}

/// The load's engine lanes, when its family staged them.
fn lanes_of(m: &mut LoadedModel) -> Option<&mut dyn MeshLanes> {
    m.state
        .as_deref_mut()
        .and_then(|state| state.mesh_family())
        .and_then(|family| family.mesh_lanes())
}

/// The speculative route a batched request takes on this load, as its
/// singleton would: native Qwen4 MTP or DFlash2 (the route selector picks
/// `Qwen4Spec`), or Gemma 4 EAGLE (the opt-in, one window past the budget
/// fits). `None`: it decodes AR.
#[allow(clippy::too_many_arguments)]
fn speculative(
    m: &LoadedModel,
    route: GenerationRoute,
    req: &BatchPendingRequest,
    sampler: &SamplerConfig,
    seed: u32,
    prompt_len: usize,
    budget: usize,
    stops: bool,
    gpu_arch: &str,
    drafter: Option<&dyn Speculator>,
) -> Option<SpecRequest> {
    let drafter = drafter?;
    if route == GenerationRoute::Gemma4Ar {
        let cache_cap = m.gemma4()?.state.max_seq;
        return crate::dense::gemma4_eagle_admits(
            Some(drafter),
            false,
            sampler.temperature,
            prompt_len,
            budget,
            cache_cap,
        )
        .then(|| SpecRequest {
            config: crate::dense::gemma4_spec_config(sampler, seed, false),
            stamp: crate::dense::gemma4_eagle_engine_done,
        });
    }
    (crate::batch::batch_generation_route(&req.original_msg, m, Some(drafter))
        == GenerationRoute::Qwen4Spec)
        .then(|| SpecRequest {
            config: crate::qwen::qwen_spec_config(
                m,
                drafter,
                gpu_arch,
                sampler,
                u64::from(seed),
                stops,
                0.0,
            ),
            stamp: crate::qwen::qwen_spec_engine_done,
        })
}

/// The record whose conversation `prompt` continues (the longest): its
/// tokens are a proper prefix of `prompt`, the live-continuation rule.
fn continued_record(records: &[LaneRecord], prompt: &[u32]) -> Option<usize> {
    records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.tokens.len() < prompt.len() && prompt.starts_with(&r.tokens))
        .max_by_key(|(_, r)| r.tokens.len())
        .map(|(at, _)| at)
}

/// A cancelled lane request: its lane rewinds to its turn start and stays its
/// conversation's record, as a resident cancel rolls back to the prior turn;
/// a lane that cannot rewind is freed.
fn cancel_lane(lanes: &mut dyn MeshLanes, records: &mut Vec<LaneRecord>, tag: u64, prompt: &[u32]) {
    match lanes.rewind_point(tag) {
        Some(at) if lanes.keeps_records() && at < prompt.len() => records.push(LaneRecord {
            tag,
            tokens: prompt[..at].to_vec(),
        }),
        _ => lanes.free_lane(tag),
    }
}

/// Free a lane for a park when none is free: the oldest record gives its
/// lane up. `false`: none is free and no record holds one.
fn make_room(lanes: &mut dyn MeshLanes, records: &mut Vec<LaneRecord>) -> bool {
    if lanes.free_lanes() > 0 {
        return true;
    }
    if records.is_empty() {
        return false;
    }
    lanes.free_lane(records.remove(0).tag);
    true
}

/// Keep the resident conversation that just committed (`tokens`, live) as a
/// lane record; without room it stays the resident live state alone.
fn keep_record(
    lanes: &mut dyn MeshLanes,
    runner: &mut MeshRunner,
    gpus: &mut Gpus,
    tag: u64,
    tokens: &[u32],
) {
    if !make_room(lanes, &mut runner.records) {
        return;
    }
    // A record never decodes: its seed is never run.
    match lanes.park(gpus, tag, tokens, &[], 0, 0) {
        Ok(()) => runner.records.push(LaneRecord {
            tag,
            tokens: tokens.to_vec(),
        }),
        Err(error) => eprintln!("[engine] conversation not kept in a lane: {error}"),
    }
}

/// A resident AR request the client cancelled after its prompt ran, every
/// token of `history` run: its state moves into a lane, which rewinds to its
/// turn start as a cancelled lane request's does ([`cancel_lane`]), so another
/// conversation's turn on the resident state does not drop it. `None`: the
/// state did not move, and the caller rolls it back.
fn park_cancelled(
    lanes: &mut dyn MeshLanes,
    runner: &mut MeshRunner,
    gpus: &mut Gpus,
    tag: u64,
    prompt: &[u32],
    history: &[u32],
    request: &mut Option<MeshRequest>,
) -> Option<Result<(), String>> {
    // A window store parks AR requests never; a prompt cut short has no history.
    if !lanes.keeps_records()
        || lanes.windows()
        || history.len() < prompt.len()
        || !make_room(lanes, &mut runner.records)
    {
        return None;
    }
    // A record never decodes: its seed is never run. The park refuses a state
    // that does not sit after `history`, before it moves anything.
    if let Err(error) = lanes.park(gpus, tag, history, &[], 0, 0) {
        eprintln!("[engine] cancelled conversation not kept in a lane: {error}");
        return None;
    }
    if let Some(request) = request.take() {
        if let Err(error) = runner.end(request, gpus) {
            lanes.free_lane(tag);
            return Some(Err(error));
        }
    }
    cancel_lane(lanes, &mut runner.records, tag, prompt);
    Some(Ok(()))
}

/// A cancel's terminal after its release: the engine `aborted` done when the
/// release was attested, else the singleton's fail-closed error
/// (`rolled_back: false`, no `done`) and `Err` so the driver stops serving
/// the unattested state.
fn cancel(
    route: GenerationRoute,
    stdout: &mut impl Write,
    id: &str,
    generated: usize,
    released: Result<(), String>,
) -> Result<(), String> {
    match released {
        Ok(()) => {
            emit_engine_cancel(route, stdout, id, generated);
            Ok(())
        }
        Err(error) => {
            let epilogue = RollbackEpilogue {
                rolled_back: false,
                context: Some(error.clone()),
            };
            emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(id),
                UNATTESTED_CANCEL_MESSAGE,
                "validation",
                false,
                &epilogue,
            );
            Err(unattested(error))
        }
    }
}

fn unattested(error: String) -> String {
    format!("engine rollback not attested; process restart required: {error}")
}

/// The speculative half of a resident rollback, ahead of [`rollback`]: a
/// client `cancel` keeps the snapshots its prompt captured; the target's
/// session state and the drafter reset.
fn reset_drafting(
    state: &mut dyn ArchModel,
    gpus: &mut Gpus,
    drafter: &mut dyn Speculator,
    cancel: bool,
) -> Result<(), String> {
    if cancel {
        state.session_commit_live_mesh(gpus, &[]);
    }
    let gpu = &mut gpus.devices[0];
    state.reset_session_state(gpu)?;
    drafter.reset(gpu)
}

/// Test-only (`serve-fault-inject`): the next [`rollback`] fails before any
/// owner reset. Process-global and one-shot: the daemon's stdin reader arms
/// it from an `abort` carrying `test_fault_engine_reset: true`.
#[cfg(feature = "serve-fault-inject")]
static ENGINE_RESET_FAULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Arm the one-shot engine reset fault (see [`ENGINE_RESET_FAULT`]).
#[cfg(feature = "serve-fault-inject")]
#[doc(hidden)]
pub fn arm_engine_reset_fault() {
    ENGINE_RESET_FAULT.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The attested rollback of the resident state (the singleton's): the
/// request's outputs drained and freed, the turn cache and deferred seed
/// dropped, every owner reset.
fn rollback(
    state: &mut dyn ArchModel,
    runner: &mut MeshRunner,
    gpus: &mut Gpus,
    asst_turn_cache: &mut AsstTurnCache,
    spec_deferred_seed: &mut Option<hipfire_loader::DeferredSpecSeed>,
    request: Option<MeshRequest>,
) -> Result<(), String> {
    if let Some(request) = request {
        let _ = runner.end(request, gpus);
    }
    *spec_deferred_seed = None;
    asst_turn_cache.clear();
    let family = state
        .mesh_family()
        .ok_or("the model declares no engine family")?;
    #[cfg(feature = "serve-fault-inject")]
    if ENGINE_RESET_FAULT.swap(false, std::sync::atomic::Ordering::SeqCst) {
        return Err("injected engine reset fault".into());
    }
    runner.reset(family, gpus)
}

impl<'a> Ctx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn of(
        lives: &'a mut [Option<Live>],
        max_batch_tokens: usize,
        route: GenerationRoute,
        sched: &'a mut ContinuousBatchScheduler,
        m: &'a mut LoadedModel,
        runner: &'a mut MeshRunner,
        gpus: &'a mut Gpus,
        stdout: &'a mut Stdout,
    ) -> Result<Self, String> {
        let LoadedModel {
            state,
            tokenizer,
            seq_pos,
            conversation_tokens,
            asst_turn_cache,
            spec_deferred_seed,
            speculator,
            ..
        } = m;
        Ok(Self {
            sched,
            lives,
            runner,
            state: state.as_deref_mut().ok_or("the model has no state")?,
            gpus,
            tokenizer: tokenizer.as_ref().ok_or("tokenizer missing")?,
            stdout,
            conversation: conversation_tokens,
            seq_pos,
            asst_turn_cache,
            spec_deferred_seed,
            speculator,
            max_batch_tokens,
            route,
        })
    }

    fn family(&mut self) -> Result<&mut dyn MeshFamily, String> {
        self.state
            .mesh_family()
            .ok_or_else(|| "the model declares no engine family".to_string())
    }

    fn lanes(&mut self) -> Result<&mut dyn MeshLanes, String> {
        self.family()?
            .mesh_lanes()
            .ok_or_else(|| "the load staged no engine lanes".to_string())
    }

    fn live(&mut self, idx: usize) -> &mut Live {
        self.lives[idx]
            .as_mut()
            .expect("the scheduler names admitted requests")
    }

    /// [`rollback`] on this tick's borrows.
    fn rollback(&mut self, request: Option<MeshRequest>) -> Result<(), String> {
        rollback(
            &mut *self.state,
            self.runner,
            self.gpus,
            self.asst_turn_cache,
            self.spec_deferred_seed,
            request,
        )
    }

    /// Give an ended request its terminal: release what its state held, then
    /// `commit_ready` (finished) or the cancel / error.
    /// `Err`: the resident rollback was not attested; the driver fails closed.
    fn settle(&mut self, core: &mut Scheduler, ended: Ended) -> Result<(), String> {
        let Ended {
            idx,
            place,
            exit,
            generated,
            drafting,
            fed,
        } = ended;
        let (key, admission) = {
            let live = self.live(idx);
            (live.key.clone(), live.admission)
        };
        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
        // A speculative request's drafter goes back to the load, its windows
        // stamped on a finished request's `done`.
        let (done_spec, spec_reset) = match drafting {
            Some(drafting) => {
                let done_spec = (SpecDone::of(&drafting), drafting.invalidated);
                *self.speculator = Some(drafting.drafter);
                (Some(done_spec), true)
            }
            // A window lane's drafter went back at its park.
            None => (self.live(idx).lane_spec.take().map(|s| (s, false)), false),
        };
        let exit = match exit {
            Exit::Finished(finish) => match self.finish(idx, place, finish, done_spec) {
                Ok(()) => return Ok(()),
                Err(error) => Exit::Failed(error),
            },
            exit => exit,
        };
        // An aborted resident request may leave epochs in flight; a failed
        // drain of them is a device failure, not a client cancel.
        let exit = match exit {
            Exit::Aborted if place == Place::Resident => match self.runner.settle(self.gpus) {
                Ok(()) => Exit::Aborted,
                Err(error) => Exit::Failed(format!("drain failed: {error}")),
            },
            exit => exit,
        };
        let mut live = self.lives[idx]
            .take()
            .expect("an ended request is admitted");
        let released = match place {
            Place::Resident if spec_reset => {
                let cancel = matches!(exit, Exit::Aborted);
                let prompt_len = live.prompt.len();
                let history = match &self.sched.lanes[idx] {
                    BatchLane::Running(lane) => lane.conversation_tokens.as_slice(),
                    _ => &[],
                };
                // A client abort past the prompt keeps what the target ran
                // live where it keeps sessions.
                if cancel
                    && super::spec::keep_fed(&mut *self.state, self.gpus, history, fed, prompt_len)
                {
                    Ok(())
                } else {
                    match self.speculator.as_deref_mut() {
                        Some(drafter) => {
                            reset_drafting(&mut *self.state, self.gpus, drafter, cancel)
                        }
                        None => Err("the drafter did not come back".into()),
                    }
                    .and_then(|()| self.rollback(live.request))
                }
            }
            Place::Resident if matches!(exit, Exit::Aborted) => {
                let history = match &self.sched.lanes[idx] {
                    BatchLane::Running(lane) => lane.conversation_tokens.as_slice(),
                    _ => &[],
                };
                self.state
                    .mesh_family()
                    .and_then(|family| family.mesh_lanes())
                    .and_then(|lanes| {
                        park_cancelled(
                            lanes,
                            self.runner,
                            self.gpus,
                            live.tag,
                            &live.prompt,
                            history,
                            &mut live.request,
                        )
                    })
                    .unwrap_or_else(|| self.rollback(live.request.take()))
            }
            Place::Resident => self.rollback(live.request),
            Place::Lane if matches!(exit, Exit::Aborted) => {
                match self.state.mesh_family().and_then(|f| f.mesh_lanes()) {
                    Some(lanes) => {
                        cancel_lane(lanes, &mut self.runner.records, live.tag, &live.prompt);
                        Ok(())
                    }
                    None => Err("the load staged no engine lanes".to_string()),
                }
            }
            Place::Lane => self.lanes().map(|lanes| lanes.free_lane(live.tag)),
            Place::Queued => Ok(()),
        };
        // Rolled back (or kept): the next request may take the resident state.
        if place == Place::Resident {
            core.release_resident();
        }
        let settled = match exit {
            Exit::Aborted => cancel(self.route, self.stdout, &key.id, generated, released),
            Exit::Failed(error) => {
                emit_generation_error(
                    self.route,
                    self.stdout,
                    Some(&key.id),
                    &format!("{} engine: {error}", self.route.name()),
                    "gpu",
                    false,
                    released.is_ok(),
                );
                released.map_err(unattested)
            }
            Exit::Finished(_) => unreachable!("handled above"),
        };
        let _ = self.sched.abort_lane(idx, &key, admission);
        settled
    }

    /// A finished request: its stream flushed, its state kept where it ran
    /// until the terminal settles (a lane then becomes its conversation's
    /// record), its `done` staged and `commit_ready` written.
    fn finish(
        &mut self,
        idx: usize,
        place: Place,
        finish: Finish,
        spec: Option<(SpecDone, bool)>,
    ) -> Result<(), String> {
        if place == Place::Resident {
            // Greedy decode leaves its last epochs in flight.
            if let Some(request) = self.live(idx).request.take() {
                self.runner
                    .end(request, self.gpus)
                    .map_err(|error| format!("drain failed: {error}"))?;
            }
        }
        let (slots, lane_capacity) = (self.sched.lanes.len(), self.sched.lane_capacity);
        let stdout = &mut *self.stdout;
        let live = self.lives[idx].as_mut().ok_or("finished request missing")?;
        let (finish_reason, tool_calls) =
            live.wire
                .finish(stdout, &live.key.id, finish == Finish::Length)?;
        let BatchLane::Running(lane) = &mut self.sched.lanes[idx] else {
            return Err("a finished request's lane is not running".into());
        };
        let end = Instant::now();
        let prompt_done = live.prompt_done.unwrap_or(end);
        let millis = |d: Duration| d.as_secs_f64() * 1000.0;
        let metrics = DoneMetrics {
            prompt_tokens: live.prompt.len(),
            cached_tokens: live.reused,
            generated: lane.streamed_tokens.len(),
            prefill_ms: millis(prompt_done - live.t0),
            decode_ms: millis(end - prompt_done),
            total_ms: millis(end - live.t0),
        };
        let mut done = metrics.done_value(&live.key.id, finish_reason);
        live.wire.stamp_done(&mut done, &metrics);
        if let (Some((spec_done, invalidated)), Some(spec)) = (spec, live.spec.as_ref()) {
            (spec.stamp)(&mut done, &spec_done);
            live.invalidated = invalidated;
        }
        stage_terminal_tool_calls(&mut done, finish_reason, &tool_calls);
        attach_continuous_batch_route_evidence(
            &mut done,
            slots,
            idx,
            lane_capacity,
            lane.max_active_lanes.max(1),
        );
        live.history = std::mem::take(&mut lane.conversation_tokens);
        live.streamed = std::mem::take(&mut lane.streamed_tokens);
        live.held = Some(place);
        let ready = commit_ready_envelope(&done).ok_or("staged done is not an object")?;
        if !self.sched.mark_awaiting_commit(idx, done) {
            return Err("mark_awaiting_commit failed".into());
        }
        writeln!(self.stdout, "{ready}")
            .and_then(|()| self.stdout.flush())
            .map_err(|e| e.to_string())
    }
}

impl Batch for Ctx<'_> {
    /// The lanes minus those finished requests hold until their terminal
    /// settles (records give their lane up on demand).
    fn lane_capacity(&mut self) -> usize {
        let held = self
            .lives
            .iter()
            .flatten()
            .filter(|live| live.held == Some(Place::Lane))
            .count();
        self.lanes()
            .map_or(0, |lanes| lanes.lane_capacity().saturating_sub(held))
    }

    fn lane_rows(&mut self) -> usize {
        let cap = self.max_batch_tokens;
        self.lanes().map_or(0, |lanes| lanes.lane_rows().min(cap))
    }

    fn row_budget(&mut self) -> usize {
        self.max_batch_tokens
    }

    /// The singleton's request start on the resident state: the session plan
    /// picks the reused prefix (restore, live continuation or reset).
    fn begin(&mut self, idx: usize) -> Result<RequestRun, String> {
        self.conversation.clear();
        *self.seq_pos = 0;
        let live = self.lives[idx]
            .as_mut()
            .ok_or("begin of a missing request")?;
        let (prompt, budget) = (&live.prompt, live.budget);
        // A lane record this prompt continues becomes the live state, as the
        // conversation's own committed turn would have left it.
        if let Some(at) = continued_record(&self.runner.records, prompt) {
            let record = self.runner.records.remove(at);
            if let Some(lanes) = self
                .state
                .mesh_family()
                .and_then(|family| family.mesh_lanes())
            {
                // A speculative turn continues a window lane with its drafter.
                let drafter: Option<&mut dyn Speculator> =
                    match (&live.spec, self.speculator.as_mut()) {
                        (Some(_), Some(drafter)) => Some(&mut **drafter),
                        _ => None,
                    };
                if let Err(error) = lanes.unpark(self.gpus, record.tag, &record.tokens, drafter) {
                    eprintln!("[engine] lane record {} not continued: {error}", record.tag);
                    lanes.free_lane(record.tag);
                }
            }
        }
        if let Some(spec) = live.spec.as_ref() {
            // Speculative: the drafter's session plan; the request owns the
            // load's drafter until it ends ([`Ended::drafting`]).
            if let Some(reason) = self.runner.quarantined() {
                return Err(format!(
                    "engine mesh quarantined; process restart required: {reason}"
                ));
            }
            let drafter = self
                .speculator
                .as_deref_mut()
                .ok_or("the load's drafter is held by another request")?;
            let plan = Plan::new(
                &*self.state,
                drafter,
                prompt,
                &spec.config,
                live.sampler.temperature,
            )?;
            let drafter = self.speculator.take().expect("checked above");
            live.reused = plan.reused;
            live.spec_route = plan.route;
            return Ok(RequestRun::new(
                prompt.len(),
                plan.reused,
                budget,
                live.sampler.clone(),
                live.seed,
            )
            .with_drafting(plan.drafting(drafter)));
        }
        self.state
            .session_request_write_limit(prompt.len().saturating_add(budget).saturating_add(1));
        let reused = self.state.session_plan(prompt, SessionRoute::Ar);
        let family = self
            .state
            .mesh_family()
            .ok_or("the model declares no engine family")?;
        // Kept from here on: a failed begin rolls back through it.
        live.request = Some(self.runner.begin(&*family, self.gpus)?);
        family.begin_prompt(self.gpus, &live.prompt, reused)?;
        live.reused = reused;
        Ok(RequestRun::new(
            live.prompt.len(),
            reused,
            budget,
            live.sampler.clone(),
            live.seed,
        ))
    }

    fn prefill(
        &mut self,
        idx: usize,
        rows: Range<usize>,
    ) -> Result<(usize, Option<Vec<f32>>), String> {
        let live = self.lives[idx]
            .as_mut()
            .ok_or("prefill of a missing request")?;
        let _progress =
            hipfire_runtime::prefill_progress::scope(&live.key.id, Some(live.key.attempt_id));
        let request = live
            .request
            .as_ref()
            .ok_or("prefill off the resident state")?;
        let family = self
            .state
            .mesh_family()
            .ok_or("the model declares no engine family")?;
        let done = {
            let mut forwards = super::mesh::MeshForwards {
                runner: &mut *self.runner,
                request,
                family,
                gpus: &mut *self.gpus,
                prompt: &live.prompt,
                budget: live.budget,
                prompt_done: None,
            };
            let out = super::Forwards::prefill(&mut forwards, rows);
            (out, forwards.prompt_done)
        };
        if done.1.is_some() {
            live.prompt_done = done.1;
        }
        done.0
    }

    fn prefill_rerun(&mut self, idx: usize) -> Result<Option<usize>, String> {
        let live = self.lives[idx]
            .as_mut()
            .ok_or("rerun of a missing request")?;
        let request = live
            .request
            .as_ref()
            .ok_or("rerun off the resident state")?;
        let family = self
            .state
            .mesh_family()
            .ok_or("the model declares no engine family")?;
        super::Forwards::prefill_rerun(&mut super::mesh::MeshForwards {
            runner: &mut *self.runner,
            request,
            family,
            gpus: &mut *self.gpus,
            prompt: &live.prompt,
            budget: live.budget,
            prompt_done: None,
        })
    }

    fn decode(
        &mut self,
        idx: usize,
        token: Option<u32>,
        top1: bool,
        keep: bool,
    ) -> Result<(u32, Next), String> {
        let live = self.lives[idx]
            .as_mut()
            .ok_or("decode of a missing request")?;
        let request = live
            .request
            .as_ref()
            .ok_or("decode off the resident state")?;
        let family = self
            .state
            .mesh_family()
            .ok_or("the model declares no engine family")?;
        let (ran, out) = self
            .runner
            .decode_epoch(request, family, self.gpus, token, top1, keep)?;
        Ok((
            ran,
            match out {
                DecodeOut::Top1(id) => Next::Top1(id),
                DecodeOut::Logits(row) => Next::Logits(row),
                DecodeOut::Device => Next::Device,
            },
        ))
    }

    /// The resident epochs drain, the state moves into the request's lane,
    /// and its resident outputs are freed.
    fn park(&mut self, idx: usize, seed: u32, remaining: usize) -> Result<(), String> {
        self.runner.settle(self.gpus)?;
        let BatchLane::Running(lane) = &self.sched.lanes[idx] else {
            return Err("a parked request's lane is not running".into());
        };
        let live = self.lives[idx]
            .as_mut()
            .ok_or("park of a missing request")?;
        let lanes = self
            .state
            .mesh_family()
            .and_then(|family| family.mesh_lanes())
            .ok_or("the load staged no engine lanes")?;
        make_room(lanes, &mut self.runner.records);
        lanes.park(
            self.gpus,
            live.tag,
            &live.prompt,
            &lane.streamed_tokens,
            seed,
            remaining,
        )?;
        if let Some(request) = live.request.take() {
            if let Err(error) = self.runner.end(request, self.gpus) {
                // The request is in its lane now: settle rolls back the
                // resident state only, so the lane is freed here.
                if let Some(lanes) = self
                    .state
                    .mesh_family()
                    .and_then(|family| family.mesh_lanes())
                {
                    lanes.free_lane(live.tag);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Speculative requests on lanes that run windows (their windows are the
    /// singleton's greedy or naive sampled accept; sampled only on lanes that
    /// continue the seeded stream), AR ones on AR lanes. A minimum token count
    /// or a think budget keeps a speculative request resident: the lane's
    /// windows stop at EOS and are never forced.
    fn lane_takes(&mut self, idx: usize) -> bool {
        let (windows, sampled) = self.lanes().map_or((false, false), |lanes| {
            (lanes.windows(), lanes.sampled_windows())
        });
        let live = self.live(idx);
        match &live.spec {
            None => !windows,
            Some(_) => {
                windows
                    && (live.sampler.temperature <= 0.0 || sampled)
                    && hipfire_runtime::sampler::PenaltyTable::flags_for(&live.sampler) == 0
                    && matches!(
                        live.wire,
                        Wire::ChatMl {
                            min_tokens: 0,
                            max_think_tokens: 0,
                            ..
                        }
                    )
            }
        }
    }

    /// The drafter's state moves into the lane with the resident state; the
    /// drafter, reset, goes back to the load for the next request (the
    /// legacy promotion's release: the resident graphs are re-captured).
    fn park_window(&mut self, idx: usize, run: &mut RequestRun) -> Result<(), String> {
        // Checked before the swap: a refused boundary leaves `run` unchanged.
        if !matches!(run.next_work(), super::Work::Verify { ar: false, .. }) {
            return Err("a parked speculative request was not at a window boundary".into());
        }
        let remaining = run.remaining();
        let live = self.lives[idx]
            .as_mut()
            .ok_or("park of a missing request")?;
        let d = run.draft.as_mut().ok_or("a window lane needs a drafter")?;
        let lanes = self
            .state
            .mesh_family()
            .and_then(|family| family.mesh_lanes())
            .ok_or("the load staged no engine lanes")?;
        make_room(lanes, &mut self.runner.records);
        lanes.park_window(
            self.gpus,
            live.tag,
            &live.prompt,
            &d.emitted,
            &mut *d.drafter,
            remaining,
        )?;
        let mut d = run
            .take_window_drafting()
            .ok_or("a parked speculative request was not at a window boundary")?;
        live.lane_spec = Some(SpecDone::of(&d));
        let gpu = &mut self.gpus.devices[0];
        let reset = d.drafter.reset(gpu);
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        gpu.replay.rearm_after_layout_growth();
        *self.speculator = Some(d.drafter);
        if let Err(error) = reset {
            // The request fails from the resident state, which rolls back:
            // its lane is freed here.
            if let Some(lanes) = self
                .state
                .mesh_family()
                .and_then(|family| family.mesh_lanes())
            {
                lanes.free_lane(live.tag);
            }
            return Err(format!("drafter reset after its park: {error}"));
        }
        Ok(())
    }

    fn step_lanes(&mut self, rows: &[(usize, u32, bool)]) -> Result<Vec<Next>, String> {
        let tagged: Vec<(u64, u32, bool)> = rows
            .iter()
            .map(|&(idx, token, logits)| (self.live(idx).tag, token, logits))
            .collect();
        let gpus = &mut *self.gpus;
        let outs = self
            .state
            .mesh_family()
            .and_then(|family| family.mesh_lanes())
            .ok_or("the load staged no engine lanes")?
            .step_lanes(gpus, &tagged)?;
        Ok(outs
            .into_iter()
            .map(|out| match out {
                DecodeOut::Top1(id) => Next::Top1(id),
                DecodeOut::Logits(row) => Next::Logits(row),
                DecodeOut::Device => unreachable!("lane rows come back to the host"),
            })
            .collect())
    }

    fn step_windows(&mut self, rows: &[(usize, usize)]) -> Result<Vec<Vec<u32>>, String> {
        let tagged: Vec<(u64, usize)> = rows
            .iter()
            .map(|&(idx, drafts)| (self.live(idx).tag, drafts))
            .collect();
        let windows = self
            .state
            .mesh_family()
            .and_then(|family| family.mesh_lanes())
            .ok_or("the load staged no engine lanes")?
            .step_windows(self.gpus, &tagged)?;
        Ok(rows
            .iter()
            .zip(windows)
            .map(|(&(idx, _), window)| {
                if let Some(spec) = self.live(idx).lane_spec.as_mut() {
                    if window.drafted > 0 {
                        spec.tally.windows += 1;
                        spec.tally.accepted += window.accepted;
                    } else {
                        spec.tally.ar_steps += 1;
                    }
                }
                window.committed
            })
            .collect())
    }

    fn commit_prompt(&mut self, idx: usize) {
        let prompt_len = self.live(idx).prompt.len();
        let BatchLane::Running(lane) = &mut self.sched.lanes[idx] else {
            return;
        };
        let prompt = &self.lives[idx].as_ref().expect("admitted").prompt;
        lane.seq_pos = prompt_len;
        lane.conversation_tokens.extend_from_slice(prompt);
    }

    fn emit(&mut self, idx: usize, token: u32) -> Result<(Verdict, Vec<u32>), String> {
        let BatchLane::Running(lane) = &mut self.sched.lanes[idx] else {
            return Err("an emitting request's lane is not running".into());
        };
        let live = self.lives[idx]
            .as_mut()
            .ok_or("emit of a missing request")?;
        let _scope = BatchAttemptScope::enter_for_generation(
            &live.key.id,
            live.key.attempt_id,
            live.admission,
        );
        lane.first_token_at.get_or_insert_with(Instant::now);
        let QwenBatchLane {
            key,
            conversation_tokens,
            streamed_tokens,
            seq_pos,
            ..
        } = lane;
        let mut host = Host {
            id: &key.id,
            out: self.stdout,
            conversation: conversation_tokens,
            seq_pos,
            tokenizer: self.tokenizer,
            prompt: &live.prompt,
            t0: live.t0,
        };
        live.wire.emit(&mut host, streamed_tokens, token)
    }

    fn history(&self, idx: usize) -> &[u32] {
        match &self.sched.lanes[idx] {
            BatchLane::Running(lane) => &lane.conversation_tokens,
            _ => &[],
        }
    }

    fn aborted(&self, idx: usize) -> bool {
        self.lives[idx].as_ref().is_some_and(|live| {
            batch_check_abort(&live.key.id, live.key.attempt_id, live.admission)
        })
    }

    fn window_cap(&self, idx: usize) -> Option<usize> {
        self.lives[idx]
            .as_ref()
            .and_then(|live| live.wire.window_cap())
    }

    type Draft<'b>
        = DraftForwards<'b>
    where
        Self: 'b;

    /// The resident speculative request's forwards on the one device.
    fn draft(&mut self, idx: usize) -> Result<DraftForwards<'_>, String> {
        let live = self.lives[idx]
            .as_mut()
            .ok_or("draft of a missing request")?;
        let target = self
            .state
            .spec_target()
            .ok_or("the model is no speculative target")?;
        Ok(DraftForwards {
            gpu: &mut self.gpus.devices[0],
            target,
            id: &live.key.id,
            prompt: &live.prompt,
            route: live.spec_route,
            budget: live.budget,
            temperature: live.sampler.temperature,
            prompt_done: &mut live.prompt_done,
            attempt: Some(live.key.attempt_id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(out: &[u8]) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(out)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// Emit `id`'s start then its cancel after `released`, as the driver
    /// does inside an admitted attempt's scope; the wire after the start.
    fn cancelled(
        id: &str,
        attempt: u64,
        released: Result<(), String>,
    ) -> (Result<(), String>, Vec<serde_json::Value>) {
        let admission = batch_announce_terminal(id, attempt).expect("batch admission");
        let _scope = BatchAttemptScope::enter_for_generation(id, attempt, admission);
        let mut out = Vec::new();
        emit_generation_start(GenerationRoute::Qwen4Ar, &mut out, id, false);
        let started = out.len();
        let result = cancel(GenerationRoute::Qwen4Ar, &mut out, id, 3, released);
        (result, lines(&out[started..]))
    }

    #[test]
    fn a_cancel_whose_rollback_fails_fails_closed_without_done() {
        let _guard = crate::ar::generation_test_lock();
        let (result, wire) = cancelled("hipf533-unattested", 5331, Err("reset failed".into()));
        let error = result.unwrap_err();
        assert!(error.contains("process restart required"), "{error}");
        assert_eq!(wire.len(), 1, "one terminal, no done: {wire:?}");
        assert_eq!(wire[0]["type"], "error", "{wire:?}");
        assert_eq!(wire[0]["rolled_back"], false, "{wire:?}");
        assert!(wire[0]["message"].as_str().is_some_and(|m| m
            .starts_with(UNATTESTED_CANCEL_MESSAGE)
            && m.contains("reset failed")));

        let (result, wire) = cancelled("hipf533-attested", 5332, Ok(()));
        result.unwrap();
        assert!(
            wire.iter().any(|line| line["finish_reason"] == "aborted"),
            "{wire:?}"
        );
    }
}
