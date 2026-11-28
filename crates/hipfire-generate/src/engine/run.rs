// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One engine-loop request, for any model that declares a `MeshFamily` and
//! any device set: the load's mesh (`LoadedModel::pp_gpus`) or the daemon's
//! device as a one-owner mesh. Admission, budget, `gen_start`, the batch-of-one
//! [`drive`], metrics, the terminal and the one attested rollback live here.

use super::chatml::{Host, ProtocolEmit, WireProtocol};
use super::mesh::MeshForwards;
use super::spec::{DraftForwards, Plan, SpecDone, SpecRequest};
use super::{admit_budget, drive, DoneMetrics, Exit, Finish, RequestRun};
use crate::ar::{
    emit_active_route_done_value, emit_engine_cancel, emit_generation_start, GenerationRoute,
};
use crate::common::{emit_fail_closed_error, emit_spec_cancel_after_rollback, RollbackEpilogue};
use hipfire_engine::emit::{emit_active_attempt_error, stage_terminal_tool_calls};
use hipfire_engine::terminal::{await_client_terminal_commit, ClientTerminalDecision};
use hipfire_loader::{AsstTurnCache, LoadedModel};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::mesh_family::MeshRunner;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::session_cache::SessionRoute;
use std::io::Stdout;
use std::time::Instant;

/// What [`run`] did with a request.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It emitted the request's terminal (done, error or cancel).
    Served,
    /// Nothing was emitted: the model or its devices cannot run this request
    /// on the engine loop (the reason); the caller keeps its own loop.
    Declined(String),
}

/// The request's sampling and budget.
pub struct Request {
    pub sampler: SamplerConfig,
    pub max_tokens: usize,
    /// Seeds this request's sampler RNG.
    pub seed: u32,
    /// The route already emitted `gen_start` (before its prompt render).
    pub opened: bool,
    /// Clamp `max_tokens` to the
    /// context left (prompt + budget + 1 trailer row <= max_seq; legacy Gemma 4
    /// emits 2 more tokens at the cap) instead of refusing it.
    pub context_fit: bool,
    /// A speculative route: the request decodes in windows of the load's
    /// drafter, which it owns while it runs.
    pub spec: Option<SpecRequest>,
}

/// Serve `prompt` on the engine loop.
#[allow(clippy::too_many_arguments)]
pub fn run(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut Stdout,
    id: &str,
    route: GenerationRoute,
    prompt: &[u32],
    protocol: &dyn WireProtocol,
    request: Request,
) -> Outcome {
    let Some(n_layers) = m
        .state
        .as_deref_mut()
        .and_then(|state| state.mesh_family())
        .map(|family| family.n_layers())
    else {
        return Outcome::Declined("the model declares no engine family".into());
    };
    let job = Job {
        stdout,
        id,
        route,
        prompt,
        protocol,
        request,
    };
    if let Some(mut gpus) = m.pp_gpus.take() {
        let outcome = with_runner(m, &gpus.mesh.clone(), |m, runner| {
            job.serve(m, runner, &mut gpus)
        });
        m.pp_gpus = Some(gpus);
        return outcome.unwrap_or_else(Outcome::Declined);
    }
    on_one_owner(m, gpu, n_layers, |m, runner, gpus| {
        job.serve(m, runner, gpus)
    })
    .unwrap_or_else(Outcome::Declined)
}

/// Whether this load serves an AR (`spec` false) or speculative request on
/// the engine before the request's own constraints, on the devices [`run`]
/// would use; `Err` is the decline [`run`] would give. For the load ack.
pub(super) fn load_admits(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    spec: bool,
) -> Result<(), String> {
    let n_layers = m
        .state
        .as_deref_mut()
        .and_then(|state| state.mesh_family())
        .map(|family| family.n_layers())
        .ok_or("the model declares no engine family")?;
    if let Some(gpus) = m.pp_gpus.take() {
        let admits = load_declines(m, &gpus, spec);
        m.pp_gpus = Some(gpus);
        return admits;
    }
    let mesh = match m.engine.as_ref().map(MeshRunner::mesh) {
        Some(mesh) if mesh.n_devices() == 1 => mesh.clone(),
        _ => DeviceMesh::single().map_err(|error| format!("one-owner mesh: {error}"))?,
    };
    Gpus::lend_single(gpu, &mesh, n_layers, |gpus| load_declines(m, gpus, spec))
}

/// The load's part of engine admission, before the request's own
/// constraints: its drafter, model, family and devices. `Err` is the
/// decline reason.
fn load_declines(m: &mut LoadedModel, gpus: &Gpus, spec: bool) -> Result<(), String> {
    if spec {
        if gpus.devices.len() != 1 {
            return Err("a pipeline mesh keeps the legacy speculative loop".into());
        }
        if m.speculator.is_none() {
            return Err("no speculator is attached".into());
        }
    } else if m.speculator.is_some() {
        // The engine's rollback attests the target only; a load with a
        // drafter keeps its legacy loop, whose rollback resets both.
        return Err("a speculator is attached".into());
    }
    let (Some(state), Some(_)) = (m.state.as_deref_mut(), m.tokenizer.as_ref()) else {
        return Err("the model has no state or tokenizer".into());
    };
    if spec && state.spec_target().is_none() {
        return Err("the model is no speculative target".into());
    }
    let family = state
        .mesh_family()
        .ok_or("the model declares no engine family")?;
    family.admit(gpus)
}

/// Run `body` on the daemon's device lent as the load's one-owner mesh, with
/// the load's runner on it. `Err`: no mesh or runner (nothing ran).
pub(super) fn on_one_owner<R>(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    n_layers: usize,
    body: impl FnOnce(&mut LoadedModel, &mut MeshRunner, &mut Gpus) -> R,
) -> Result<R, String> {
    let mesh = match m.engine.as_ref().map(MeshRunner::mesh) {
        Some(mesh) if mesh.n_devices() == 1 => mesh.clone(),
        _ => DeviceMesh::single().map_err(|error| format!("one-owner mesh: {error}"))?,
    };
    Gpus::lend_single(gpu, &mesh, n_layers, |gpus| {
        with_runner(m, &mesh, |m, runner| body(m, runner, gpus))
    })
}

/// The load's runner on `mesh` (a new one when the mesh changed), lent out of
/// the model for `body`.
fn with_runner<R>(
    m: &mut LoadedModel,
    mesh: &DeviceMesh,
    body: impl FnOnce(&mut LoadedModel, &mut MeshRunner) -> R,
) -> Result<R, String> {
    let mut runner = match m.engine.take() {
        Some(runner) if runner.mesh().epoch() == mesh.epoch() => runner,
        _ => MeshRunner::new(mesh.clone())?,
    };
    let outcome = body(m, &mut runner);
    m.engine = Some(runner);
    Ok(outcome)
}

struct Job<'a> {
    stdout: &'a mut Stdout,
    id: &'a str,
    route: GenerationRoute,
    prompt: &'a [u32],
    protocol: &'a dyn WireProtocol,
    request: Request,
}

impl Job<'_> {
    fn serve(self, m: &mut LoadedModel, runner: &mut MeshRunner, gpus: &mut Gpus) -> Outcome {
        if self.request.spec.is_some() {
            return self.serve_spec(m, runner, gpus);
        }
        if let Err(reason) = load_declines(m, gpus, false) {
            return Outcome::Declined(reason);
        }
        let Job {
            stdout,
            id,
            route,
            prompt,
            protocol,
            request,
        } = self;
        let LoadedModel {
            state,
            tokenizer,
            seq_pos,
            max_seq,
            conversation_tokens,
            asst_turn_cache,
            spec_deferred_seed,
            ..
        } = m;
        if let Some(reason) = runner.quarantined() {
            let message = format!("engine mesh quarantined; process restart required: {reason}");
            emit_active_attempt_error(stdout, Some(id), &message, "gpu", false, false);
            return Outcome::Served;
        }
        let (Some(state), Some(tokenizer)) = (state.as_deref_mut(), tokenizer.as_ref()) else {
            unreachable!("load_declines checked state and tokenizer");
        };
        let budget = match admit_budget(
            prompt.len(),
            request.max_tokens,
            *max_seq,
            |n, used, cap| {
                if request.context_fit {
                    n.min(cap.saturating_sub(used))
                } else {
                    crate::common::fit_max_tokens(n, used, cap)
                }
            },
        ) {
            Ok(budget) => budget,
            Err(message) => {
                let class = if request.max_tokens == 0 {
                    "validation"
                } else {
                    "context_length"
                };
                emit_active_attempt_error(stdout, Some(id), &message, class, false, false);
                return Outcome::Served;
            }
        };
        state.session_request_write_limit(prompt.len().saturating_add(budget).saturating_add(1));
        let reused = state.session_plan(prompt, SessionRoute::Ar);
        let t0 = Instant::now();
        let mut emitter = protocol.open(Host {
            id,
            out: stdout,
            conversation: conversation_tokens,
            seq_pos,
            tokenizer,
            prompt,
            t0,
        });
        emitter.clear_history();
        if !request.opened {
            emit_generation_start(route, emitter.out(), id, protocol.started_in_think());
        }
        let family = state
            .mesh_family()
            .expect("the family admitted this request above");
        let mut rollback = |emitter: &mut dyn ProtocolEmit,
                            state: &mut dyn hipfire_runtime::arch_model::ArchModel,
                            runner: &mut MeshRunner,
                            gpus: &mut Gpus,
                            error: Option<String>,
                            generated: usize| {
            *spec_deferred_seed = None;
            emitter.clear_history();
            asst_turn_cache.clear();
            // A client cancel keeps the snapshots the prompt already captured
            // (a failure drops them with the state), as the speculative one.
            if error.is_none() {
                state.session_commit_live_mesh(gpus, &[]);
            }
            let reset = match state.mesh_family() {
                Some(family) => runner.reset(family, gpus),
                None => Err("the model declares no engine family".into()),
            };
            let epilogue = match reset {
                Ok(()) => RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                },
                Err(error) => RollbackEpilogue {
                    rolled_back: false,
                    context: Some(error),
                },
            };
            match error {
                None if epilogue.rolled_back => {
                    emit_engine_cancel(route, emitter.out(), id, generated);
                }
                None => emit_spec_cancel_after_rollback(emitter.out(), id, generated, &epilogue),
                Some(message) => emit_fail_closed_error(
                    emitter.out(),
                    Some(id),
                    &format!("{} engine: {message}", route.name()),
                    "gpu",
                    false,
                    &epilogue,
                ),
            }
        };
        let mesh_request = match runner.begin(&*family, gpus) {
            Ok(mesh_request) => mesh_request,
            Err(error) => {
                rollback(&mut *emitter, &mut *state, runner, gpus, Some(error), 0);
                return Outcome::Served;
            }
        };
        let mut run = RequestRun::new(prompt.len(), reused, budget, request.sampler, request.seed);
        let (exit, prompt_done) = match family.begin_prompt(gpus, prompt, reused) {
            Err(error) => (Exit::Failed(error), None),
            Ok(()) => {
                let mut forwards = MeshForwards {
                    runner,
                    request: &mesh_request,
                    family: &mut *family,
                    gpus: &mut *gpus,
                    prompt,
                    budget,
                    prompt_done: None,
                };
                let exit = drive(&mut run, &mut forwards, &mut *emitter);
                (exit, forwards.prompt_done)
            }
        };
        // A finished or aborted request is acknowledged only once its last
        // epochs drained (greedy decode and the prompt leave them in flight);
        // a failed drain is a device failure, not a client cancel.
        let exit = match (exit, runner.end(mesh_request, gpus)) {
            (Exit::Finished(_) | Exit::Aborted, Err(error)) => {
                Exit::Failed(format!("drain failed: {error}"))
            }
            (exit, _) => exit,
        };
        let finish = match exit {
            Exit::Finished(finish) => finish,
            Exit::Aborted => {
                rollback(
                    &mut *emitter,
                    &mut *state,
                    runner,
                    gpus,
                    None,
                    run.generated(),
                );
                return Outcome::Served;
            }
            Exit::Failed(error) => {
                rollback(
                    &mut *emitter,
                    &mut *state,
                    runner,
                    gpus,
                    Some(error),
                    run.generated(),
                );
                return Outcome::Served;
            }
        };
        let (finish_reason, tool_calls) = match emitter.finish(finish == Finish::Length) {
            Ok(terminal) => terminal,
            Err(error) => {
                rollback(
                    &mut *emitter,
                    &mut *state,
                    runner,
                    gpus,
                    Some(error),
                    run.generated(),
                );
                return Outcome::Served;
            }
        };
        let end = Instant::now();
        let prompt_done = prompt_done.unwrap_or(end);
        let millis = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let metrics = DoneMetrics {
            prompt_tokens: prompt.len(),
            cached_tokens: reused,
            generated: run.generated(),
            prefill_ms: millis(prompt_done - t0),
            decode_ms: millis(end - prompt_done),
            total_ms: millis(end - t0),
        };
        let mut done = metrics.done_value(id, finish_reason);
        protocol.stamp_done(&mut done, &metrics);
        stage_terminal_tool_calls(&mut done, finish_reason, &tool_calls);
        if await_client_terminal_commit(emitter.out(), id, &done) != ClientTerminalDecision::Commit
        {
            rollback(
                &mut *emitter,
                &mut *state,
                runner,
                gpus,
                None,
                run.generated(),
            );
            return Outcome::Served;
        }
        state.session_commit_live_mesh(gpus, emitter.history());
        emitter.commit_turn(asst_turn_cache);
        emit_active_route_done_value(emitter.out(), &done);
        Outcome::Served
    }

    /// [`Self::serve`] of a speculative route: the load's drafter moves into
    /// the request's [`RequestRun`] and drives the target through the shared
    /// `Speculator` / `SpecTarget` seam on the one device; the session plan
    /// is the speculative one; a rollback also resets the drafter.
    fn serve_spec(self, m: &mut LoadedModel, runner: &mut MeshRunner, gpus: &mut Gpus) -> Outcome {
        if let Err(reason) = load_declines(m, gpus, true) {
            return Outcome::Declined(reason);
        }
        let Job {
            stdout,
            id,
            route,
            prompt,
            protocol,
            request,
        } = self;
        let spec = request.spec.expect("a speculative route");
        let LoadedModel {
            state,
            tokenizer,
            seq_pos,
            max_seq,
            physical_cap,
            conversation_tokens,
            asst_turn_cache,
            spec_deferred_seed,
            speculator,
            ..
        } = m;
        let drafter = speculator
            .as_mut()
            .expect("load_declines checked the drafter");
        if let Some(reason) = runner.quarantined() {
            let message = format!("engine mesh quarantined; process restart required: {reason}");
            emit_active_attempt_error(stdout, Some(id), &message, "gpu", false, false);
            return Outcome::Served;
        }
        let (Some(model), Some(tokenizer)) = (state.as_deref_mut(), tokenizer.as_ref()) else {
            unreachable!("load_declines checked state and tokenizer");
        };
        let budget = match admit_budget(
            prompt.len(),
            request.max_tokens,
            *max_seq,
            crate::common::fit_max_tokens,
        ) {
            Ok(budget) => budget,
            Err(message) => {
                let class = if request.max_tokens == 0 {
                    "validation"
                } else {
                    "context_length"
                };
                emit_active_attempt_error(stdout, Some(id), &message, class, false, false);
                return Outcome::Served;
            }
        };
        // The drafter's context bounds the request too (one window past the
        // budget); a request it cannot hold keeps the legacy AR fallback.
        let ctx_capacity = drafter.ctx_capacity().min(*physical_cap);
        if !crate::common::spec_ctx_request_fits(
            prompt.len(),
            budget,
            drafter.block_size(),
            ctx_capacity,
        ) {
            return Outcome::Declined(format!(
                "prompt {} + budget {budget} + one window exceed the drafter's context {ctx_capacity}",
                prompt.len()
            ));
        }
        let temperature = request.sampler.temperature;
        let plan = match Plan::new(&*model, &mut **drafter, prompt, &spec.config, temperature) {
            Ok(plan) => plan,
            Err(reason) => return Outcome::Declined(reason),
        };
        let t0 = Instant::now();
        let mut emitter = protocol.open(Host {
            id,
            out: stdout,
            conversation: conversation_tokens,
            seq_pos,
            tokenizer,
            prompt,
            t0,
        });
        emitter.clear_history();
        if !request.opened {
            emit_generation_start(route, emitter.out(), id, protocol.started_in_think());
        }
        let drafter = speculator.take().expect("checked above");
        let reused = plan.reused;
        let mut run = RequestRun::new(prompt.len(), reused, budget, request.sampler, request.seed)
            .with_drafting(plan.drafting(drafter));
        let mut prompt_done = None;
        let target = model.spec_target().expect("admitted above");
        let exit = drive(
            &mut run,
            &mut DraftForwards {
                gpu: &mut gpus.devices[0],
                target,
                id,
                prompt,
                route: plan.route,
                budget,
                temperature,
                prompt_done: &mut prompt_done,
                attempt: None,
            },
            &mut *emitter,
        );
        let outcome = SpecEnd {
            route,
            id,
            prompt,
            reused,
            t0,
            prompt_done,
            stamp: spec.stamp,
            protocol,
        }
        .settle(
            exit,
            &mut run,
            &mut *emitter,
            model,
            runner,
            gpus,
            spec_deferred_seed,
            asst_turn_cache,
        );
        *speculator = run.into_drafting().map(|d| d.drafter);
        outcome
    }
}

/// How a speculative request ends: its terminal, the session commit or the
/// rollback that resets target and drafter.
struct SpecEnd<'a> {
    route: GenerationRoute,
    id: &'a str,
    prompt: &'a [u32],
    reused: usize,
    t0: Instant,
    prompt_done: Option<Instant>,
    stamp: fn(&mut serde_json::Value, &SpecDone),
    protocol: &'a dyn WireProtocol,
}

impl SpecEnd<'_> {
    #[allow(clippy::too_many_arguments)]
    fn settle(
        self,
        exit: Exit,
        run: &mut RequestRun,
        emitter: &mut dyn ProtocolEmit,
        state: &mut dyn hipfire_runtime::arch_model::ArchModel,
        runner: &mut MeshRunner,
        gpus: &mut Gpus,
        spec_deferred_seed: &mut Option<hipfire_loader::DeferredSpecSeed>,
        asst_turn_cache: &mut AsstTurnCache,
    ) -> Outcome {
        let SpecEnd {
            route,
            id,
            prompt,
            reused,
            t0,
            prompt_done,
            stamp,
            protocol,
        } = self;
        let generated = run.generated();
        let mut rollback = |emitter: &mut dyn ProtocolEmit,
                            run: &mut RequestRun,
                            error: Option<String>| {
            // A client cancel past the prompt keeps what the target ran live
            // where it keeps sessions, as the legacy cancel does.
            if error.is_none()
                && super::spec::keep_fed(state, gpus, emitter.history(), run.position, prompt.len())
            {
                emit_engine_cancel(route, emitter.out(), id, generated);
                return;
            }
            *spec_deferred_seed = None;
            emitter.clear_history();
            asst_turn_cache.clear();
            // Else a client cancel keeps the snapshots the prompt already
            // captured (a failure drops them with the state).
            if error.is_none() {
                state.session_commit_live_mesh(gpus, &[]);
            }
            let gpu = &mut gpus.devices[0];
            let mut reset = state.reset_session_state(gpu);
            if let Some(d) = run.draft.as_mut() {
                reset = reset.and(d.drafter.reset(gpu));
            }
            // The runner's reset drains and attests the device.
            let reset = match (reset, state.mesh_family()) {
                (Ok(()), Some(family)) => runner.reset(family, gpus),
                (Ok(()), None) => Err("the model declares no engine family".into()),
                (Err(error), _) => Err(error),
            };
            let epilogue = match reset {
                Ok(()) => RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                },
                Err(error) => RollbackEpilogue {
                    rolled_back: false,
                    context: Some(error),
                },
            };
            match error {
                None if epilogue.rolled_back => {
                    emit_engine_cancel(route, emitter.out(), id, generated);
                }
                None => emit_spec_cancel_after_rollback(emitter.out(), id, generated, &epilogue),
                Some(message) => emit_fail_closed_error(
                    emitter.out(),
                    Some(id),
                    &format!("{} engine: {message}", route.name()),
                    "gpu",
                    false,
                    &epilogue,
                ),
            }
        };
        let finish = match exit {
            Exit::Finished(finish) => finish,
            Exit::Aborted => {
                rollback(emitter, run, None);
                return Outcome::Served;
            }
            Exit::Failed(error) => {
                rollback(emitter, run, Some(error));
                return Outcome::Served;
            }
        };
        let (finish_reason, tool_calls) = match emitter.finish(finish == Finish::Length) {
            Ok(terminal) => terminal,
            Err(error) => {
                rollback(emitter, run, Some(error));
                return Outcome::Served;
            }
        };
        let end = Instant::now();
        let prompt_done = prompt_done.unwrap_or(end);
        let millis = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let metrics = DoneMetrics {
            prompt_tokens: prompt.len(),
            cached_tokens: reused,
            generated,
            prefill_ms: millis(prompt_done - t0),
            decode_ms: millis(end - prompt_done),
            total_ms: millis(end - t0),
        };
        let mut done = metrics.done_value(id, finish_reason);
        protocol.stamp_done(&mut done, &metrics);
        let d = run.drafting().expect("a speculative request");
        stamp(&mut done, &SpecDone::of(d));
        let invalidated = d.invalidated;
        stage_terminal_tool_calls(&mut done, finish_reason, &tool_calls);
        if await_client_terminal_commit(emitter.out(), id, &done) != ClientTerminalDecision::Commit
        {
            rollback(emitter, run, None);
            return Outcome::Served;
        }
        // An unrepaired terminal window reset the live state: publish the
        // turn's snapshots only.
        let consumed: &[u32] = if invalidated { &[] } else { emitter.history() };
        state.session_commit_live_mesh(gpus, consumed);
        emitter.commit_turn(asst_turn_cache);
        emit_active_route_done_value(emitter.out(), &done);
        Outcome::Served
    }
}
