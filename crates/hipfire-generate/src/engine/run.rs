// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One engine-loop request, for any model that declares a `MeshFamily` and
//! any device set: the load's mesh (`LoadedModel::pp_gpus`) or the daemon's
//! device as a one-owner mesh. Admission, budget, `gen_start`, the batch-of-one
//! [`drive`], metrics, the terminal and the one attested rollback live here.

use super::chatml::{Host, ProtocolEmit, WireProtocol};
use super::mesh::MeshForwards;
use super::{admit_budget, drive, DoneMetrics, Exit, Finish, RequestRun};
use crate::ar::{
    emit_active_route_done_value, emit_engine_cancel, emit_generation_start, GenerationRoute,
};
use crate::common::{emit_fail_closed_error, emit_spec_cancel_after_rollback, RollbackEpilogue};
use hipfire_engine::emit::{emit_active_attempt_error, stage_terminal_tool_calls};
use hipfire_engine::terminal::{await_client_terminal_commit, ClientTerminalDecision};
use hipfire_loader::LoadedModel;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::mesh_family::{MeshFamily, MeshRunner};
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
            speculator,
            ..
        } = m;
        // The engine's rollback attests the target only; a load with a drafter
        // keeps its legacy loop, whose rollback resets both.
        if speculator.is_some() {
            return Outcome::Declined("a speculator is attached".into());
        }
        if let Some(reason) = runner.quarantined() {
            let message = format!("engine mesh quarantined; process restart required: {reason}");
            emit_active_attempt_error(stdout, Some(id), &message, "gpu", false, false);
            return Outcome::Served;
        }
        let (Some(state), Some(tokenizer)) = (state.as_deref_mut(), tokenizer.as_ref()) else {
            return Outcome::Declined("the model has no state or tokenizer".into());
        };
        let Some(family) = state.mesh_family() else {
            return Outcome::Declined("the model declares no engine family".into());
        };
        if let Err(reason) = family.admit(gpus) {
            return Outcome::Declined(reason);
        }
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
                            family: &mut dyn MeshFamily,
                            runner: &mut MeshRunner,
                            gpus: &mut Gpus,
                            error: Option<String>,
                            generated: usize| {
            *spec_deferred_seed = None;
            emitter.clear_history();
            asst_turn_cache.clear();
            let epilogue = match runner.reset(family, gpus) {
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
                rollback(&mut *emitter, family, runner, gpus, Some(error), 0);
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
        // A finished request is acknowledged only once its last decode
        // epochs drained (greedy decode leaves them in flight).
        let exit = match (exit, runner.end(mesh_request, gpus)) {
            (Exit::Finished(_), Err(error)) => Exit::Failed(format!("drain failed: {error}")),
            (exit, _) => exit,
        };
        let finish = match exit {
            Exit::Finished(finish) => finish,
            Exit::Aborted => {
                rollback(&mut *emitter, family, runner, gpus, None, run.generated());
                return Outcome::Served;
            }
            Exit::Failed(error) => {
                rollback(
                    &mut *emitter,
                    family,
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
                    family,
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
            rollback(&mut *emitter, family, runner, gpus, None, run.generated());
            return Outcome::Served;
        }
        state.session_commit_live_mesh(gpus, emitter.history());
        emitter.commit_turn(asst_turn_cache);
        emit_active_route_done_value(emitter.out(), &done);
        Outcome::Served
    }
}
