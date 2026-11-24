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

use super::chatml::{ChatMl, ChatMlStream, Host};
use super::sched::{Batch, Ended, Place, Scheduler};
use super::{admit_budget, DoneMetrics, Exit, Finish, Next, RequestRun, Verdict};
use crate::ar::*;
use crate::batch::{drain_qwen_batch_inbox, is_vmm_batch_request_eligible};
use crate::common::fit_max_tokens_if;
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
use hipfire_runtime::tokenizer::Tokenizer;
use std::io::{Stdout, Write};
use std::ops::Range;
use std::time::{Duration, Instant};

const ROUTE: GenerationRoute = GenerationRoute::Qwen4Ar;

/// Serve the staged continuous batch on the engine loop until no request is
/// queued or live. `Err`: a driver-level failure; every request it held got
/// its terminal.
pub fn drive(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    m: &mut LoadedModel,
    params: VmmBatchParams,
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
    };
    let result = super::run::on_one_owner(m, gpu, n_layers, |m, runner, gpus| {
        let result = driver.serve(sched, m, runner, gpus, stdout, inbox);
        if let Err(error) = &result {
            driver.fail_all(sched, m, runner, gpus, stdout, error);
        }
        result
    })
    .and_then(|result| result);
    result.map_err(BatchDriveError::Gpu)
}

/// One admitted request's host side.
struct Live {
    key: AttemptKey,
    admission: BatchGeneration,
    prompt: Vec<u32>,
    budget: usize,
    sampler: SamplerConfig,
    seed: u32,
    started_in_think: bool,
    stop: Vec<String>,
    min_tokens: usize,
    max_think_tokens: usize,
    eos: u32,
    stream: ChatMlStream,
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
}

impl Live {
    fn protocol(&self) -> ChatMl<'_> {
        ChatMl {
            started_in_think: self.started_in_think,
            tools: false,
            stop: &self.stop,
            min_tokens: self.min_tokens,
            max_think_tokens: self.max_think_tokens,
            eos: std::slice::from_ref(&self.eos),
        }
    }
}

struct Driver {
    lives: Vec<Option<Live>>,
    core: Scheduler,
    max_batch_tokens: usize,
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
    max_batch_tokens: usize,
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
                return Err(format!("engine mesh quarantined until restart: {reason}"));
            }
            self.settle_terminals(sched, m, runner, gpus, stdout)?;
            for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
                let Some(admission) = sched.pending.get(&key).map(|req| req.admission) else {
                    continue;
                };
                if batch_check_abort(&key.id, key.attempt_id, admission) {
                    let _scope =
                        BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                    emit_generation_start(ROUTE, stdout, &key.id, false);
                    emit_generation_cancel(ROUTE, stdout, &key.id, 0);
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
                ROUTE,
                stdout,
                inbox,
                true,
            )?;
            if let Some(barrier) = barrier {
                inbox.push_front(barrier);
            }
            while let Some((key, ticket)) = sched.try_assign_one() {
                self.admit(sched, m, stdout, key, ticket)?;
            }
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
            if !self.core.busy() {
                // Only terminals awaiting their client are left.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let mut ctx = Ctx::of(
                &mut self.lives,
                self.max_batch_tokens,
                sched,
                m,
                runner,
                gpus,
                stdout,
            )?;
            let ended = self.core.tick(&mut ctx);
            for ended in ended {
                ctx.settle(&mut self.core, ended);
            }
        }
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
        let refuse = |sched: &mut ContinuousBatchScheduler,
                      stdout: &mut Stdout,
                      message: &str,
                      class: &str| {
            emit_generation_error(ROUTE, stdout, Some(&key.id), message, class, false, true);
            let _ = sched.abort_lane(idx, &key, ticket.admission);
        };
        let (prompt, started_in_think) = match crate::fn_batch::render_request(m, &req) {
            Ok(rendered) => rendered,
            Err((message, class)) => {
                refuse(sched, stdout, &message, class);
                return Ok(());
            }
        };
        if prompt.is_empty() {
            refuse(
                sched,
                stdout,
                "qwen4 AR prompt rendered to zero tokens",
                "validation",
            );
            return Ok(());
        }
        let fit = req
            .original_msg
            .get("max_tokens_fit")
            .and_then(|v| v.as_bool())
            == Some(true);
        let budget = match admit_budget(prompt.len(), req.max_tokens, m.max_seq, |n, p, s| {
            fit_max_tokens_if(fit, n, p, s)
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
        let min_tokens = hipfire_runtime::stop_sequence::parse_min_tokens_field(
            req.original_msg.get("min_tokens"),
            req.max_tokens,
        );
        let stop = hipfire_runtime::stop_sequence::parse_stop_field(req.original_msg.get("stop"))
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
        let tag = NEXT_TAG
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| "engine lane tags are exhausted")?;
        emit_generation_start(ROUTE, stdout, &key.id, started_in_think);
        if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
            lane.prompt_len = prompt.len();
        }
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
        let live = Live {
            seed: request_seed_for(&key, req.client_seed),
            key,
            admission: ticket.admission,
            prompt,
            budget,
            sampler,
            started_in_think,
            stop,
            min_tokens,
            max_think_tokens: req.max_think_tokens,
            eos,
            stream,
            tag,
            t0: Instant::now(),
            reused: 0,
            prompt_done: None,
            request: None,
            held: None,
            history: Vec::new(),
            streamed: Vec::new(),
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
                            ..
                        } = &mut *m;
                        let state = state.as_deref_mut().ok_or("the model has no state")?;
                        let rolled_back = rollback(
                            state,
                            runner,
                            gpus,
                            asst_turn_cache,
                            spec_deferred_seed,
                            None,
                        );
                        if let Err(error) = rolled_back {
                            eprintln!("[engine] rollback of {}: {error}", key.id);
                        }
                        self.core.release_resident();
                    }
                    _ => {
                        if let Some(lanes) = lanes_of(m) {
                            lanes.free_lane(live.tag);
                        }
                    }
                }
                let generated = done.get("tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                emit_generation_cancel(ROUTE, stdout, &key.id, generated);
                let _ = sched.abort_lane(idx, &key, admission);
                continue;
            }
            if !sched.commit_lane_retain_terminal(idx, &key, admission) {
                emit_generation_error(
                    ROUTE,
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
            live.stream.commit_turn(
                &live.protocol(),
                tokenizer,
                &live.streamed,
                &mut m.asst_turn_cache,
            );
            match live.held {
                Some(Place::Resident) => {
                    // The singleton's commit, then the conversation moves into
                    // a lane record so another conversation's turn on the
                    // resident state does not drop it.
                    let state = m.state.as_deref_mut().ok_or("the model has no state")?;
                    if state.spec_session_reuse() {
                        state.session_commit_live_mesh(gpus, &live.history);
                        if let Some(lanes) = lanes_of(m) {
                            keep_record(lanes, runner, gpus, live.tag, &live.history);
                        }
                    }
                    self.core.release_resident();
                }
                _ => runner.records.push(LaneRecord {
                    tag: live.tag,
                    tokens: live.history,
                }),
            }
            emit_generation_done_value(ROUTE, stdout, &done);
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
                ROUTE,
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
            emit_generation_error(ROUTE, stdout, Some(&key.id), error, "gpu", false, false);
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
    runner.reset(family, gpus)
}

impl<'a> Ctx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn of(
        lives: &'a mut [Option<Live>],
        max_batch_tokens: usize,
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
            max_batch_tokens,
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
    fn settle(&mut self, core: &mut Scheduler, ended: Ended) {
        let Ended {
            idx,
            place,
            exit,
            generated,
        } = ended;
        let (key, admission) = {
            let live = self.live(idx);
            (live.key.clone(), live.admission)
        };
        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
        let exit = match exit {
            Exit::Finished(finish) => match self.finish(idx, place, finish) {
                Ok(()) => return,
                Err(error) => {
                    if place == Place::Resident {
                        core.release_resident();
                    }
                    Exit::Failed(error)
                }
            },
            exit => exit,
        };
        let live = self.lives[idx]
            .take()
            .expect("an ended request is admitted");
        let released = match place {
            Place::Resident => self.rollback(live.request),
            Place::Lane => self.lanes().map(|lanes| lanes.free_lane(live.tag)),
            Place::Queued => Ok(()),
        };
        match exit {
            Exit::Aborted => {
                emit_generation_cancel(ROUTE, self.stdout, &key.id, generated);
            }
            Exit::Failed(error) => {
                let rolled_back = released.is_ok();
                emit_generation_error(
                    ROUTE,
                    self.stdout,
                    Some(&key.id),
                    &format!("{} engine: {error}", ROUTE.name()),
                    "gpu",
                    false,
                    rolled_back,
                );
            }
            Exit::Finished(_) => unreachable!("handled above"),
        }
        let _ = self.sched.abort_lane(idx, &key, admission);
    }

    /// A finished request: its stream flushed, its state kept where it ran
    /// until the terminal settles (a lane then becomes its conversation's
    /// record), its `done` staged and `commit_ready` written.
    fn finish(&mut self, idx: usize, place: Place, finish: Finish) -> Result<(), String> {
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
        let (finish_reason, tool_calls) = live.stream.finish(stdout, finish == Finish::Length)?;
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
                if let Err(error) = lanes.unpark(self.gpus, record.tag, &record.tokens) {
                    eprintln!("[engine] lane record {} not continued: {error}", record.tag);
                    lanes.free_lane(record.tag);
                }
            }
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
        let spec = ChatMl {
            started_in_think: live.started_in_think,
            tools: false,
            stop: &live.stop,
            min_tokens: live.min_tokens,
            max_think_tokens: live.max_think_tokens,
            eos: std::slice::from_ref(&live.eos),
        };
        live.stream.emit(&spec, &mut host, streamed_tokens, token)
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
}
