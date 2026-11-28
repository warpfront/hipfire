// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving entry of the one-row plain-AR decode of Inkling-2-Mini (arch 24).
//!
//! [`forward_plain_ar_gpu`] is the ONE function the generation loop and the
//! daemon's `bench_decode` call for a decode step. It selects between the
//! backends of the load ([`DecodeBackend`]):
//!
//! ```text
//! Pm4     the retained PM4 tape over the frozen C1 body
//!         ([`crate::decode_replay::DecodePm4`]): warm / record / replay. Taken
//!         when the load's replay controller is the PM4 route (the daemon's
//!         model default for the admitted row, [`PM4_DEFAULT_ON`], or an
//!         explicit HIPFIRE_REPLAY_BACKEND=redline)
//! Graph   the HIP-graph executor over the same body
//!         ([`crate::decode_replay::DecodeGraph`]): bucket arm / record /
//!         replay. The route under HIPFIRE_REPLAY_BACKEND=hip, and the
//!         fail-closed fallback of PM4 (a refused record / prepare leaves the
//!         step's token committed, then the load stays on the graph)
//! Eager   the incumbent blocked one-row decode ([`crate::forward_decode_gpu`])
//! ```
//!
//! Admission (§4 of the redline plan): gfx1201, the Mini fixed 48-visit
//! schedule over 24 physical layers, the typed-XTS certificate (24/24
//! symmetric layers, symmetric unembed, F16 router, AWQ sidecars, padded
//! down), the A8 expert tier with the fused producer recipe, F16 KV, the FA2
//! decode attention, one device, and no foreign capture / retained controller.
//! Anything else runs the incumbent eager decode with a typed reason
//! ([`EagerReason`]); nothing is truncated or approximated.
//!
//! Kill switch: the existing graph switches. `HIPFIRE_GRAPH=0` or
//! `HIPFIRE_AR_GRAPH=0` (the process-config spelling `experimental.graph.*`)
//! make every step eager (PM4 included). `HIPFIRE_REPLAY_BACKEND=hip` keeps the
//! graph. There is no new flag; the HIP graph is the default for the admitted row.
//!
//! Failure policy (inherited from the executors): a step refused before
//! submission returns its error with the state untouched; a PM4 record /
//! prepare refusal after the eager warm token returns that token's logits once
//! and the load moves to the graph (its tape released, the controller
//! poisoned); a graph capture failure likewise leaves the load sticky-eager; a
//! failure after execution began returns the error (the caller owns rollback:
//! `dense_fail_closed_error`), with no same-token retry and no host tag /
//! `seq_pos` advance. A PM4 failure whose retirement could not be proven
//! quarantines the Gpu's retained owners (nothing is freed or reused, the
//! bundle free path leaks its owners) and every later step errors. Only a caller that never calls this
//! function inside a prompt prefill may use it: the prefill one-row tail goes
//! through `forward_prefill_gpu` and never selects a graph.
//!
//! The per-load state ([`DecodeServe`]) lives on the GPU state of the bundle
//! (`Inkling2GpuState::decode_serve`), so a new load starts a fresh executor
//! and [`DecodeServe::release`] quiesces and destroys the active tape / graph
//! before the weights, KV and arena they name are freed.

use crate::arch_model::Inkling2Bundle;
use crate::decode_replay::{
    pm4_dev_opt_in, DecodeGraph, DecodePm4, GraphStats, Pm4Route, Pm4Stats, StepRoute,
};
use crate::decode_step::ensure_decode_arena;
use crate::gpu::{DOWN_K_PADDED, INKLING2_KV_DTYPE};
use crate::gpu_prefill::{forward_decode_gpu, prefill_supported};
use hipfire_dispatch::pipeline::PrefillActTier;
use rdna_compute::replay::ReplayState;
use rdna_compute::{DType, Gpu};

/// The single switch of the product default: `true` makes the retained PM4 tape
/// the default decode route of the admitted row (the daemon enables the load's
/// replay controller for it, `inkling2_retained_default`); `false` keeps the HIP
/// graph the default and leaves PM4 to an explicit `HIPFIRE_REPLAY_BACKEND=redline`.
pub const PM4_DEFAULT_ON: bool = false;

/// The decode backends of a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeBackend {
    Eager,
    Graph,
    Pm4,
}

/// Why a step ran the incumbent eager decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EagerReason {
    /// `HIPFIRE_GRAPH=0` or `HIPFIRE_AR_GRAPH=0`.
    Killed,
    /// The load / device / context is outside the admitted row.
    Ineligible(String),
    /// The load's M=1 decode arena could not be created.
    ArenaRefused(String),
    /// An earlier capture or graph launch of this load failed.
    Sticky(String),
}

impl std::fmt::Display for EagerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Killed => write!(f, "killed (HIPFIRE_GRAPH=0 / HIPFIRE_AR_GRAPH=0)"),
            Self::Ineligible(r) => write!(f, "not admitted: {r}"),
            Self::ArenaRefused(r) => write!(f, "decode arena refused: {r}"),
            Self::Sticky(r) => write!(f, "sticky eager: {r}"),
        }
    }
}

/// What one served step executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServeRoute {
    /// The incumbent blocked one-row decode ([`forward_decode_gpu`]).
    Eager(EagerReason),
    /// The graph executor's own report (replay, eager warm token + capture, or
    /// its in-executor eager step after a failed capture).
    Graph(StepRoute),
    /// The PM4 executor's own report (warm, record, replay, or its in-executor
    /// eager step after a refused record / prepare).
    Pm4(Pm4Route),
}

impl ServeRoute {
    /// The backend whose launches the step actually used.
    pub fn backend(&self) -> DecodeBackend {
        match self {
            Self::Graph(StepRoute::Replay) => DecodeBackend::Graph,
            Self::Pm4(Pm4Route::Replay) => DecodeBackend::Pm4,
            _ => DecodeBackend::Eager,
        }
    }
}

/// Cumulative per-load counters of the served steps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServeCounts {
    /// Steps that ran the incumbent eager decode (`forward_decode_gpu`).
    pub eager: u64,
    /// Steps that were one graph launch.
    pub graph_replays: u64,
    /// Steps that ran the eager warm token of a new key, then recorded it.
    pub graph_arm_steps: u64,
    /// Steps the executor ran eagerly after its capture failed.
    pub graph_capture_failed_steps: u64,
    /// Steps that were one retained PM4 submission.
    pub pm4_replays: u64,
    /// The load's first PM4 step (an eager warm token, nothing recorded).
    pub pm4_warm_steps: u64,
    /// Steps that were the recording token of a PM4 tape.
    pub pm4_record_steps: u64,
    /// Steps the PM4 executor ran eagerly after a refused record / prepare.
    pub pm4_failed_steps: u64,
}

/// A snapshot of the per-load serving state for logs, the bench ledger and the
/// tests.
#[derive(Clone, Debug)]
pub struct ServeStats {
    pub counts: ServeCounts,
    pub graph: GraphStats,
    pub pm4: Pm4Stats,
    pub last_route: Option<ServeRoute>,
    /// The graph executor's sticky reason.
    pub sticky: Option<String>,
    /// The PM4 executor's sticky reason (the load fell back to the graph).
    pub pm4_sticky: Option<String>,
    pub pm4_quarantined: bool,
    pub arena_refusal: Option<String>,
}

/// The per-load serving state (one graph executor and one PM4 executor per load).
pub struct DecodeServe {
    graph: DecodeGraph,
    pm4: DecodePm4,
    /// The PM4 route was left (sticky): its tape is released and the replay
    /// controller poisoned so the graph may run.
    pm4_left: bool,
    counts: ServeCounts,
    last: Option<ServeRoute>,
    arena_refusal: Option<String>,
    /// The first eager reason / graph arm was reported on stderr.
    announced: bool,
}

impl Default for DecodeServe {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeServe {
    pub fn new() -> Self {
        Self {
            graph: DecodeGraph::new(true),
            pm4: DecodePm4::new(true, false),
            pm4_left: false,
            counts: ServeCounts::default(),
            last: None,
            arena_refusal: None,
            announced: false,
        }
    }

    pub fn stats(&self) -> ServeStats {
        ServeStats {
            counts: self.counts.clone(),
            graph: self.graph.stats().clone(),
            pm4: self.pm4.stats().clone(),
            last_route: self.last.clone(),
            sticky: self.graph.sticky_reason().map(str::to_owned),
            pm4_sticky: self.pm4.sticky_reason().map(str::to_owned),
            pm4_quarantined: self.pm4.quarantined(),
            arena_refusal: self.arena_refusal.clone(),
        }
    }

    /// The active graph's key, if one is held.
    pub fn active_key(&self) -> Option<&crate::decode_step::DecodeStepKey> {
        self.graph.active_key()
    }

    /// The active PM4 tape's key, if one is held.
    pub fn pm4_active_key(&self) -> Option<&crate::decode_step::DecodeStepKey> {
        self.pm4.active_key()
    }

    /// Whether retirement of a PM4 submission could not be proven: the load's
    /// owners must not be freed or reused.
    pub fn quarantined(&self) -> bool {
        self.pm4.quarantined()
    }

    /// Quiesce, then destroy the active PM4 tape and graph. Called before
    /// anything they name (weights, KV, conv, arena, scratch) is freed. Returns
    /// `false` when retirement could not be proven: the Gpu's retained owners
    /// are quarantined and the caller must NOT free what the tape names. A graph
    /// whose quiesce fails has its handles invalidated (the memory is about to
    /// be released regardless).
    #[must_use]
    pub fn release(&mut self, gpu: &mut Gpu) -> bool {
        let mut safe = true;
        if let Err(e) = self.pm4.release(gpu) {
            eprintln!("[inkling2] decode pm4 release: {e}");
            if self.pm4.quarantined() {
                gpu.quarantine_retained_replay_owners(e);
                safe = false;
            }
        }
        if let Err(e) = self.graph.release(gpu) {
            eprintln!("[inkling2] decode graph release: {e}");
            gpu.invalidate_graph_state();
        }
        safe
    }

    /// Once per load: say which backend the first step took.
    fn announce(&mut self, what: &str) {
        if !self.announced {
            self.announced = true;
            eprintln!("[inkling2] decode backend: {what}");
        }
    }

    /// The static eligibility of this step, before the backend choice.
    fn eligibility(&self, bundle: &Inkling2Bundle, gpu: &Gpu) -> Result<(), EagerReason> {
        if !(gpu.flags.graph_ar && gpu.flags.graph_forward != Some(false)) {
            return Err(EagerReason::Killed);
        }
        if let Some(r) = &self.arena_refusal {
            return Err(EagerReason::ArenaRefused(r.clone()));
        }
        admit_row(bundle, gpu).map_err(EagerReason::Ineligible)?;
        // Foreign capture: never share the stream with it.
        if gpu.graphs.capture_mode {
            return Err(EagerReason::Ineligible(
                "inside a foreign graph capture".into(),
            ));
        }
        if let Some(s) = gpu.active_stream.as_ref() {
            if gpu.hip.stream_is_capturing(s).unwrap_or(true) {
                return Err(EagerReason::Ineligible(
                    "the owned stream is capturing".into(),
                ));
            }
        }
        Ok(())
    }

    /// Leave the PM4 route for good (sticky, not quarantined): release its tape
    /// (queue proven idle) and poison the replay controller, so the graph may
    /// run without the "retained controller active" refusal.
    fn leave_pm4(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.pm4_left {
            return Ok(());
        }
        let why = self
            .pm4
            .sticky_reason()
            .unwrap_or("pm4 route left")
            .to_owned();
        if let Err(e) = self.pm4.release(gpu) {
            if self.pm4.quarantined() {
                gpu.quarantine_retained_replay_owners(e.clone());
            }
            return Err(e);
        }
        if gpu.replay.state() != ReplayState::Fallback {
            gpu.replay.poison(format!("inkling2 pm4 route left: {why}"));
        }
        self.pm4_left = true;
        eprintln!(
            "[inkling2] decode backend: PM4 route left ({why}); falling back to the HIP graph"
        );
        Ok(())
    }

    /// The backend of this step.
    fn pick(
        &mut self,
        bundle: &Inkling2Bundle,
        gpu: &mut Gpu,
    ) -> Result<DecodeBackend, EagerReason> {
        self.eligibility(bundle, gpu)?;
        let pm4_wanted = pm4_dev_opt_in(gpu) && !self.pm4_left;
        if pm4_wanted && self.pm4.sticky_reason().is_none() {
            return Ok(DecodeBackend::Pm4);
        }
        if pm4_wanted {
            // Sticky PM4 (a refused record / prepare, a proven-quiescent failure).
            if let Err(e) = self.leave_pm4(gpu) {
                return Err(EagerReason::Ineligible(format!(
                    "pm4 release before the graph fallback failed: {e}"
                )));
            }
        }
        if gpu.replay.is_recording() || gpu.replay.is_enabled() {
            return Err(EagerReason::Ineligible(
                "the retained replay controller is active (not the PM4 route)".into(),
            ));
        }
        if let Some(r) = self.graph.sticky_reason() {
            return Err(EagerReason::Sticky(r.to_owned()));
        }
        Ok(DecodeBackend::Graph)
    }

    fn eager_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
        why: EagerReason,
    ) -> Result<(), String> {
        self.announce(&format!("eager ({why})"));
        *logits = forward_decode_gpu(bundle, gpu, token_id, pos)?;
        self.counts.eager += 1;
        self.last = Some(ServeRoute::Eager(why));
        Ok(())
    }

    fn step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), String> {
        if self.pm4.quarantined() {
            return Err(
                "inkling2 decode: the retained PM4 route lost retirement proof; this load is quarantined and must be reloaded"
                    .into(),
            );
        }
        let backend = match self.pick(bundle, gpu) {
            Ok(b) => b,
            Err(why) => return self.eager_step(bundle, gpu, token_id, pos, logits, why),
        };
        // Something else (a fail-closed rollback, the daemon) destroyed the
        // Gpu-level graph under the executor: drop its claim before it can
        // select a replay of a graph that no longer exists.
        if self.graph.active_key().is_some() && gpu.graphs.graph_exec.is_none() {
            self.graph.release(gpu)?;
        }
        if let Err(e) = ensure_decode_arena(gpu, bundle) {
            self.arena_refusal = Some(e.clone());
            return self.eager_step(
                bundle,
                gpu,
                token_id,
                pos,
                logits,
                EagerReason::ArenaRefused(e),
            );
        }
        match backend {
            DecodeBackend::Pm4 => self.pm4_step(bundle, gpu, token_id, pos, logits),
            _ => self.graph_step(bundle, gpu, token_id, pos, logits),
        }
    }

    fn pm4_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), String> {
        let report = match self.pm4.forward(bundle, gpu, token_id, pos, logits) {
            Ok(r) => r,
            Err(e) => {
                if self.pm4.quarantined() {
                    gpu.quarantine_retained_replay_owners(e.clone());
                }
                return Err(e);
            }
        };
        match &report.route {
            Pm4Route::Replay => self.counts.pm4_replays += 1,
            Pm4Route::Warm => {
                self.counts.pm4_warm_steps += 1;
                self.announce("retained PM4 (eager warm token)");
            }
            Pm4Route::Recorded => {
                self.counts.pm4_record_steps += 1;
                self.announce("retained PM4 (tape recorded)");
            }
            Pm4Route::Eager(why) => {
                self.counts.pm4_failed_steps += 1;
                eprintln!("[inkling2] decode backend: PM4 record refused ({why:?}); this load moves to the HIP graph");
            }
        }
        self.last = Some(ServeRoute::Pm4(report.route));
        Ok(())
    }

    fn graph_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), String> {
        let report = self.graph.forward(bundle, gpu, token_id, pos, logits)?;
        match &report.route {
            StepRoute::Replay => self.counts.graph_replays += 1,
            StepRoute::EagerCaptured => {
                self.counts.graph_arm_steps += 1;
                self.announce("graph (HIP graph, bucket armed)");
            }
            StepRoute::Eager(why) => {
                self.counts.graph_capture_failed_steps += 1;
                self.announce(&format!("eager (graph capture refused: {why:?})"));
            }
        }
        self.last = Some(ServeRoute::Graph(report.route));
        Ok(())
    }
}

/// The static admitted row: everything but the per-step flags and the
/// foreign-capture state.
pub fn admit_row(bundle: &Inkling2Bundle, gpu: &Gpu) -> Result<(), String> {
    let cfg = &bundle.config;
    if !gpu.arch_caps.is_gfx1201() {
        return Err(format!("device {} is not gfx1201", gpu.arch));
    }
    if cfg.loop_cfg.n_slot() != 48 || cfg.n_phys != 24 {
        return Err(format!(
            "not the Mini fixed schedule ({} visits over {} layers)",
            cfg.loop_cfg.n_slot(),
            cfg.n_phys
        ));
    }
    if !prefill_supported(bundle, gpu) {
        return Err("the blocked one-row body is unsupported on this load".into());
    }
    let (Some(gw), Some(gs)) = (bundle.gpu_weights.as_ref(), bundle.gpu_state.as_ref()) else {
        return Err("no gpu weights / state".into());
    };
    let Some(cap) = gw.xts_routes.as_ref() else {
        return Err("not a typed-XTS load".into());
    };
    if cap.n_layers != 24
        || cap.symmetric_layers != 24
        || !cap.unembed_symmetric
        || cap.awq_sidecars == 0
        || cap.down_k_physical != DOWN_K_PADDED
    {
        return Err(format!(
            "typed-XTS certificate not met (layers {}, symmetric {}, unembed {}, awq {}, down_k {})",
            cap.n_layers,
            cap.symmetric_layers,
            cap.unembed_symmetric,
            cap.awq_sidecars,
            cap.down_k_physical
        ));
    }
    if gs.prefill_routes.experts != PrefillActTier::A8 {
        return Err(format!(
            "expert tier {:?} is not A8",
            gs.prefill_routes.experts
        ));
    }
    if !gpu.flags.gfx12_producer_quant_fused_enabled() {
        return Err("the fused producer recipe is off".into());
    }
    if INKLING2_KV_DTYPE != DType::F16 {
        return Err("KV is not F16".into());
    }
    if !gs.attn_fa2 {
        return Err("the FA2 decode attention is off".into());
    }
    Ok(())
}

/// One plain sequential AR decode step: `token_id` at `pos` (the session
/// cursor), the unpadded-vocab logits in `logits`. Graph when admitted, the
/// incumbent eager decode otherwise (see the module docs for the policy).
pub fn forward_plain_ar_gpu(
    bundle: &mut Inkling2Bundle,
    gpu: &mut Gpu,
    token_id: u32,
    pos: usize,
    logits: &mut Vec<f32>,
) -> Result<(), String> {
    let mut serve = match bundle.gpu_state.as_mut() {
        Some(gs) => gs.decode_serve.take().unwrap_or_default(),
        None => {
            *logits = forward_decode_gpu(bundle, gpu, token_id, pos)?;
            return Ok(());
        }
    };
    let r = serve.step(bundle, gpu, token_id, pos, logits);
    if let Some(gs) = bundle.gpu_state.as_mut() {
        gs.decode_serve = Some(serve);
    }
    r
}

/// The serving counters of a load (`None` before its first served step).
pub fn serve_stats(bundle: &Inkling2Bundle) -> Option<ServeStats> {
    bundle
        .gpu_state
        .as_ref()
        .and_then(|gs| gs.decode_serve.as_ref())
        .map(DecodeServe::stats)
}

/// Quiesce and destroy the load's active PM4 tape and graph (idempotent).
/// Callers: the bundle's free path, and any caller about to change what a tape
/// names. `false` = retirement could not be proven and the Gpu's retained
/// owners are quarantined: the caller must not free what the tape names.
#[must_use]
pub fn release_decode_graph(bundle: &mut Inkling2Bundle, gpu: &mut Gpu) -> bool {
    if let Some(gs) = bundle.gpu_state.as_mut() {
        if let Some(mut serve) = gs.decode_serve.take() {
            return serve.release(gpu);
        }
    }
    true
}

/// Whether the daemon should enable the load's retained replay controller (PM4)
/// by default: the admitted row, on one device, with no drafter, and
/// [`PM4_DEFAULT_ON`]. A fact of the load's typed certificate and recipe
/// ([`admit_row`]), never of the model name, file name or weight-format label.
pub fn retained_default_qualified(bundle: &Inkling2Bundle, gpu: &Gpu) -> bool {
    PM4_DEFAULT_ON && admit_row(bundle, gpu).is_ok()
}
