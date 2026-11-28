// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HIP-graph ([`DecodeGraph`]) and retained-PM4 ([`DecodePm4`]) executors over
//! the frozen one-row decode body (dev ruler / oracle comparators; no product
//! wiring).
//!
//! [`DecodePm4`] drives the primary `gpu.replay` manually: a real eager warm
//! token, then one recording token (ordinary HIP while the recorder is open,
//! the whole 1,024-launch body in one tape, head and softcap included), seal,
//! `prepare_pm4_prefix` with the record count equal to the body count, and
//! every later matching step is stage (ordered, blocking), one
//! `replay_pm4_routed_checked` (waited to retirement), finish (D2H + commit).
//! No patching; default wait/acquire policy.
//!
//! The graph records [`crate::decode_step::enqueue_decode_body`] only, the
//! launch DAG between the one H2D staging copy and the one D2H logits read.
//! Staging, host validation and the commit (`finish_decode_step`) stay outside
//! it, so a replayed token is: plan, stage, `hipGraphLaunch`, finish.
//!
//! ```text
//! key == active key           plan, stage, graph launch, finish            (Replay)
//! otherwise                   destroy the old graph (quiesced first), run the
//!                             step eagerly (stage, body, finish: a real
//!                             token), then capture the body for this key
//!                             (capture executes nothing)                   (EagerCaptured)
//! capture refused / census    the step's logits and commit stand; sticky
//! mismatch                    eager fallback with a reason                 (Eager(CaptureFailed))
//! ```
//!
//! One active graph per executor (= per load): no per-bucket cache. The key is
//! [`DecodeStepKey`] plus a fingerprint of every address a kernarg blob bakes
//! (arena, FA2 partials, conv, K/V planes, the shared scratch the body
//! touches), so a moved pointee can never be replayed even when the epoch and
//! generation did not change. The kernarg blobs are graph-owned
//! (`GraphState::ar_forward_blobs`, moved out of the shared capture scratch at
//! `end_graph_capture`). Replay runs only on the owned stream and is refused
//! inside a foreign capture or an active retained recorder.
//!
//! Failure policy: a refusal before submission (validation, staging) leaves
//! everything untouched and keeps the graph. A capture that fails after the
//! eager warm token returns that token once and goes sticky-eager (not cleared
//! by a bucket change or a reset; only a new executor, i.e. a new load, clears
//! it). A failure after the graph launch began poisons the executor (sticky)
//! and is returned to the caller, which owns rollback; the same token is never
//! retried eagerly against partially changed state.

use crate::arch_model::Inkling2Bundle;
use crate::config::Inkling2Config;
use crate::decode_step::{
    enqueue_decode_body, ensure_decode_arena, finish_decode_step, forward_decode_frozen,
    plan_decode_step, stage_decode_input, DecodeArena, DecodeStepKey, DecodeStepPlan,
};
use crate::gpu::{Inkling2GpuState, Inkling2GpuWeights};
use hip_bridge::memory_effects;
use rdna_compute::replay::{ReplayBackendRequest, ReplayQuiescence, ReplayState};
use rdna_compute::Gpu;
use std::collections::BTreeSet;
use std::time::Instant;

/// Why a step ran eagerly instead of through the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EagerWhy {
    /// The executor was built disabled (the dev opt-in was not given).
    Disabled,
    /// An earlier capture/launch failed; the reason is the first failure.
    Sticky(String),
    /// This step's own capture failed or its census disagreed with the eager
    /// step's (the step itself is a valid eager token).
    CaptureFailed(String),
}

/// What one [`DecodeGraph::forward`] call executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepRoute {
    /// Frozen eager step, no graph involved.
    Eager(EagerWhy),
    /// Eager step (the real token of a new key), then the body was captured for
    /// the key; the next matching step replays.
    EagerCaptured,
    /// Stage, one graph launch, finish.
    Replay,
}

#[derive(Clone, Debug)]
pub struct StepReport {
    pub route: StepRoute,
    /// Wall time of the capture + instantiate, inside this call, after the
    /// eager token was committed (0 unless `route == EagerCaptured`).
    pub capture_ns: u64,
}

/// Cumulative executor counters (a new executor starts at zero).
#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub eager_steps: u64,
    pub replays: u64,
    pub captures: u64,
    pub capture_failures: u64,
    /// Graphs destroyed on a key change / release.
    pub destroys: u64,
    pub total_capture_ns: u64,
    pub last_capture_ns: u64,
    /// Kernel nodes of the active (or last captured) graph = graph-owned
    /// kernarg blobs = launches observed while recording.
    pub nodes: usize,
    /// Body launches of the eager step that preceded the last capture.
    pub eager_launches: u64,
    /// `hipGraphLaunch` calls issued by this executor.
    pub graph_launches: u64,
}

struct Active {
    key: DecodeStepKey,
    layout: u64,
}

/// The single-active-graph executor of one load.
pub struct DecodeGraph {
    enabled: bool,
    active: Option<Active>,
    sticky: Option<String>,
    stats: GraphStats,
}

/// The dev opt-in of the ruler: an explicit `HIPFIRE_GRAPH=1` with the AR graph
/// not switched off (`HIPFIRE_AR_GRAPH`, default on). The executor itself never
/// reads the environment; the ruler passes this to [`DecodeGraph::new`].
pub fn graph_dev_opt_in(gpu: &Gpu) -> bool {
    gpu.flags.graph_forward == Some(true) && gpu.flags.graph_ar
}

impl DecodeGraph {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            active: None,
            sticky: None,
            stats: GraphStats::default(),
        }
    }

    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }

    /// The first failure that made this executor eager for the rest of the load.
    pub fn sticky_reason(&self) -> Option<&str> {
        self.sticky.as_deref()
    }

    /// Key of the active graph, if one is held.
    pub fn active_key(&self) -> Option<&DecodeStepKey> {
        self.active.as_ref().map(|a| &a.key)
    }

    /// Quiesce the owned stream, then destroy the active graph and its blobs.
    /// If quiescence cannot be proven the graph is kept (never freed under
    /// possibly running work) and the executor goes sticky.
    pub fn release(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.active.is_none() {
            return Ok(());
        }
        if let Err(e) = quiesce(gpu) {
            self.go_sticky(format!("quiesce before destroy failed: {e}"));
            return Err(format!(
                "inkling2 graph: quiesce before destroy failed: {e}"
            ));
        }
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        self.active = None;
        self.stats.destroys += 1;
        Ok(())
    }

    fn go_sticky(&mut self, reason: String) {
        if self.sticky.is_none() {
            self.sticky = Some(reason);
        }
    }

    /// One decode step at `pos` for `token_id`; the logits land in `logits` and
    /// the host tags/cursors commit exactly as the frozen eager step does.
    pub fn forward(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<StepReport, String> {
        refuse_foreign(gpu)?;
        if let Some(why) = self.eager_reason() {
            forward_decode_frozen(bundle, gpu, token_id, pos, logits)?;
            self.stats.eager_steps += 1;
            return Ok(StepReport {
                route: StepRoute::Eager(why),
                capture_ns: 0,
            });
        }
        ensure_decode_arena(gpu, bundle)?;
        gpu.ensure_capture_stream()
            .map_err(|e| format!("inkling2 graph: owned stream: {e:?}"))?;
        let plan = plan_decode_step(bundle, gpu, pos)?;
        let layout = layout_fingerprint(bundle, gpu)?;
        // The Gpu-level graph can be dropped under the executor (fail-closed
        // rollback / daemon invalidation): a missing exec is a miss, never a
        // launch of a graph that no longer exists.
        let hit = self
            .active
            .as_ref()
            .map(|a| a.key == plan.key && a.layout == layout && gpu.graphs.graph_exec.is_some());
        let retained = match hit {
            Some(true) => true,
            Some(false) => {
                self.release(gpu)?;
                false
            }
            None => false,
        };
        if retained {
            return self.replay_step(bundle, gpu, token_id, pos, logits);
        }
        // New key: the real token runs eagerly (it also resolves every lazily
        // built module and scratch), then the body is captured for the key.
        let fx0 = memory_effects::snapshot();
        forward_decode_frozen(bundle, gpu, token_id, pos, logits)?;
        let eager_launches = memory_effects::snapshot().since(fx0).launches;
        self.stats.eager_steps += 1;
        let t0 = Instant::now();
        match self.capture(bundle, gpu, &plan.key, eager_launches) {
            Ok(()) => {
                let ns = t0.elapsed().as_nanos() as u64;
                self.stats.captures += 1;
                self.stats.last_capture_ns = ns;
                self.stats.total_capture_ns += ns;
                self.stats.eager_launches = eager_launches;
                self.active = Some(Active {
                    key: plan.key,
                    layout,
                });
                Ok(StepReport {
                    route: StepRoute::EagerCaptured,
                    capture_ns: ns,
                })
            }
            Err(reason) => {
                self.stats.capture_failures += 1;
                self.go_sticky(reason.clone());
                Ok(StepReport {
                    route: StepRoute::Eager(EagerWhy::CaptureFailed(reason)),
                    capture_ns: 0,
                })
            }
        }
    }

    fn eager_reason(&self) -> Option<EagerWhy> {
        if !self.enabled {
            Some(EagerWhy::Disabled)
        } else {
            self.sticky.clone().map(EagerWhy::Sticky)
        }
    }

    /// Stage (outside the graph), launch the retained graph, finish.
    fn replay_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<StepReport, String> {
        let pos32 =
            u32::try_from(pos).map_err(|_| "inkling2 graph: pos exceeds u32".to_string())?;
        let gw = bundle
            .gpu_weights
            .as_ref()
            .ok_or("inkling2 graph: no gpu weights")?;
        let gs = bundle
            .gpu_state
            .as_mut()
            .ok_or("inkling2 graph: no gpu state")?;
        let mut arena = gs
            .decode_arena
            .take()
            .ok_or("inkling2 graph: no decode arena")?;
        // Staging is a refusal point: nothing has been submitted if it fails.
        if let Err(e) = stage_decode_input(gpu, gw, &mut arena, token_id, pos32) {
            gs.decode_arena = Some(arena);
            return Err(e);
        }
        let launched = (|| {
            let stream = gpu
                .active_stream
                .as_ref()
                .ok_or("inkling2 graph: owned stream vanished")?;
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, stream)
                .map_err(|e| format!("inkling2 graph: launch: {e:?}"))?;
            Ok::<(), String>(())
        })();
        let r = match launched {
            Ok(()) => {
                self.stats.graph_launches += 1;
                finish_decode_step(gpu, gs, &arena, pos, logits)
            }
            Err(e) => Err(e),
        };
        gs.decode_arena = Some(arena);
        match r {
            Ok(()) => {
                self.stats.replays += 1;
                Ok(StepReport {
                    route: StepRoute::Replay,
                    capture_ns: 0,
                })
            }
            Err(e) => {
                // Execution began: the turn is invalid, no eager retry. Drain
                // the stream (best effort), drop the graph, stay eager.
                self.go_sticky(format!("graph replay failed: {e}"));
                let _ = self.release(gpu);
                Err(e)
            }
        }
    }

    /// Record the body for `key`. Capture executes nothing, so the arena,
    /// state and host cursors are exactly as the eager token left them.
    fn capture(
        &mut self,
        bundle: &Inkling2Bundle,
        gpu: &mut Gpu,
        key: &DecodeStepKey,
        eager_launches: u64,
    ) -> Result<(), String> {
        let gw = bundle.gpu_weights.as_ref().ok_or("no gpu weights")?;
        let gs = bundle.gpu_state.as_ref().ok_or("no gpu state")?;
        let arena = gs.decode_arena.as_ref().ok_or("no decode arena")?;
        let plan = crate::decode_step::DecodeStepPlan { key: key.clone() };
        quiesce(gpu).map_err(|e| format!("quiesce before capture: {e}"))?;
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        {
            let stream = gpu.active_stream.as_ref().ok_or("no owned stream")?;
            gpu.graphs
                .begin_graph_capture(&gpu.hip, gpu.device_id, stream)
                .map_err(|e| format!("begin capture: {e:?}"))?;
        }
        let fx0 = memory_effects::snapshot();
        let body = enqueue_decode_body(gpu, &bundle.config, gw, gs, arena, &plan);
        let fx = memory_effects::snapshot().since(fx0);
        if let Err(e) = body {
            self.abort_capture(gpu);
            return Err(format!("recording the body: {e}"));
        }
        {
            let stream = gpu.active_stream.as_ref().ok_or("no owned stream")?;
            if let Err(e) = gpu
                .graphs
                .end_graph_capture(&gpu.hip, gpu.device_id, stream)
            {
                self.abort_capture(gpu);
                return Err(format!("end capture / instantiate: {e:?}"));
            }
        }
        let nodes = gpu.graphs.ar_forward_blobs.len();
        // Every node a kernel launch carrying graph-owned bytes; no copy, memset
        // or readback inside; the count equals the eager step's launches.
        let ok = fx.launches as usize == nodes
            && fx.launches == eager_launches
            && (fx.htod, fx.dtod, fx.dtoh, fx.memset) == (0, 0, 0, 0);
        self.stats.nodes = nodes;
        if !ok {
            gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
            return Err(format!(
                "capture census: recorded launches {} / graph-owned blobs {nodes} / eager launches {eager_launches} / effects {fx:?}",
                fx.launches
            ));
        }
        Ok(())
    }

    fn abort_capture(&mut self, gpu: &mut Gpu) {
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.graphs
                .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
        }
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    }
}

/// Drain the owned stream (the quiesce a graph needs before it is destroyed or
/// re-recorded).
fn quiesce(gpu: &Gpu) -> Result<(), String> {
    match gpu.active_stream.as_ref() {
        Some(s) => gpu.hip.stream_synchronize(s).map_err(|e| format!("{e:?}")),
        None => gpu.hip.device_synchronize().map_err(|e| format!("{e:?}")),
    }
}

/// Replay runs only on the owned stream and never inside somebody else's
/// capture or an active retained recorder/replay.
fn refuse_foreign(gpu: &Gpu) -> Result<(), String> {
    if gpu.graphs.capture_mode {
        return Err("inkling2 graph: refused inside a foreign graph capture".into());
    }
    if gpu.replay.is_recording() || gpu.replay.is_enabled() {
        return Err(
            "inkling2 graph: refused while the retained replay controller is active".into(),
        );
    }
    if let Some(s) = gpu.active_stream.as_ref() {
        if gpu
            .hip
            .stream_is_capturing(s)
            .map_err(|e| format!("inkling2 graph: capture status: {e:?}"))?
        {
            return Err("inkling2 graph: refused on a stream that is capturing".into());
        }
    }
    Ok(())
}

/// Fold of every address a recorded kernarg blob can name: the arena slab, the
/// FA2 partition scratch, every conv history and K/V plane, and the shared
/// scratch the body touches. A pointee that moved changes it.
fn layout_fingerprint(bundle: &Inkling2Bundle, gpu: &Gpu) -> Result<u64, String> {
    let gs: &Inkling2GpuState = bundle
        .gpu_state
        .as_ref()
        .ok_or("inkling2 graph: no gpu state")?;
    let arena = gs
        .decode_arena
        .as_ref()
        .ok_or("inkling2 graph: no decode arena")?;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut fold = |p: usize| {
        h = (h.rotate_left(5) ^ p as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    };
    fold(arena.base_ptr());
    fold(gs.fa2_lse.buf.as_ptr() as usize);
    fold(gs.fa2_oacc.buf.as_ptr() as usize);
    for c in &gs.conv {
        fold(c.buf.as_ptr() as usize);
    }
    for (k, v) in gs.k_cache.iter().zip(&gs.v_cache) {
        for t in k.iter().chain(v) {
            fold(t.buf.as_ptr() as usize);
        }
    }
    let sc = &gpu.scratch;
    let buf = |o: &Option<hip_bridge::DeviceBuffer>| o.as_ref().map_or(0, |b| b.as_ptr() as usize);
    let ten =
        |o: &Option<rdna_compute::GpuTensor>| o.as_ref().map_or(0, |t| t.buf.as_ptr() as usize);
    fold(buf(&sc.int8_mmq_x_scratch));
    fold(buf(&sc.int8_mmq_down_x_scratch));
    fold(ten(&sc.mq_signs1));
    fold(ten(&sc.mq_signs2));
    fold(ten(&sc.mq_x_rot));
    Ok(h)
}

// ---------------------------------------------------------------------------
// Retained PM4 executor (S4 kill measurement; dev ruler / oracle, no product
// wiring)
// ---------------------------------------------------------------------------

/// Route-identity key the PM4 tape latches (`ReplayController::note_route_identity`):
/// the identity string folds the decode key, the layout fingerprint and the
/// arch, so a tape recorded for another owner/bucket/layout can never be
/// selected by an equal-looking key.
pub const PM4_ROUTE_KEY: u64 = 0x494e_4b32_5043_3031;

/// The dev opt-in of the ruler: `HIPFIRE_REPLAY_BACKEND=redline` with the PM4
/// transport on a gfx1201 card. The executor never reads the environment; the
/// ruler passes this to [`DecodePm4::new`].
pub fn pm4_dev_opt_in(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_gfx1201()
        && gpu.replay.request() == ReplayBackendRequest::Auto
        && gpu.replay.uses_pm4_transport()
}

/// What one [`DecodePm4::forward`] call executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pm4Route {
    /// Frozen eager step, nothing recorded.
    Eager(EagerWhy),
    /// The load's first step: a real eager token that resolves every lazily
    /// built module. Nothing is recorded.
    Warm,
    /// A real token run through ordinary HIP while the primary recorder was
    /// open, then the tape was sealed and prepared (next matching step replays).
    Recorded,
    /// Stage, one retained PM4 submission waited to retirement, finish.
    Replay,
}

#[derive(Clone, Debug)]
pub struct Pm4StepReport {
    pub route: Pm4Route,
    /// begin_capture .. finish_capture + HIP quiesce (0 unless `Recorded`).
    pub record_ns: u64,
    /// `prepare_pm4_prefix` (0 unless `Recorded`).
    pub prepare_ns: u64,
}

/// Cumulative executor counters (a new executor starts at zero).
#[derive(Clone, Debug, Default)]
pub struct Pm4Stats {
    pub eager_steps: u64,
    pub warm_steps: u64,
    pub recordings: u64,
    pub record_failures: u64,
    /// Tapes released on a key change / release.
    pub releases: u64,
    pub replays: u64,
    /// Exact recorded-HIP-blob oracle steps (proof only).
    pub blob_replays: u64,
    /// Recorded launches of the active (or last) tape == dispatches prepared.
    pub nodes: usize,
    pub unique_symbols: usize,
    /// Body launches of the warm eager step.
    pub eager_launches: u64,
    pub last_record_ns: u64,
    pub last_prepare_ns: u64,
    pub total_record_ns: u64,
    pub total_prepare_ns: u64,
    pub command_dwords: u32,
    pub queue_id: u64,
}

struct ActivePm4 {
    key: DecodeStepKey,
    layout: u64,
    identity: String,
    count: usize,
}

/// The single-active-tape PM4 executor of one load: the primary
/// `gpu.replay`, driven manually (`begin_capture` / `finish_capture` /
/// `prepare_pm4_prefix`), one complete body per tape (record count == body
/// count), default wait/acquire, no patching.
pub struct DecodePm4 {
    enabled: bool,
    /// Order staging/readback on the Gpu's owned stream (HIP side) instead of
    /// the blocking null-stream copies.
    stream_ordered: bool,
    active: Option<ActivePm4>,
    warmed: bool,
    sticky: Option<String>,
    quarantined: bool,
    stats: Pm4Stats,
    symbols: Vec<String>,
}

impl DecodePm4 {
    pub fn new(enabled: bool, stream_ordered: bool) -> Self {
        Self {
            enabled,
            stream_ordered,
            active: None,
            warmed: false,
            sticky: None,
            quarantined: false,
            stats: Pm4Stats::default(),
            symbols: Vec::new(),
        }
    }

    pub fn stats(&self) -> &Pm4Stats {
        &self.stats
    }

    pub fn sticky_reason(&self) -> Option<&str> {
        self.sticky.as_deref()
    }

    /// Retirement of a submission could not be proven: every owner the tape
    /// names must be treated as possibly in use; nothing is freed or reused.
    pub fn quarantined(&self) -> bool {
        self.quarantined
    }

    pub fn active_key(&self) -> Option<&DecodeStepKey> {
        self.active.as_ref().map(|a| &a.key)
    }

    /// Unique recorded kernel symbols of the active (or last) tape, sorted.
    pub fn recorded_symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Prove the queue idle, then release the prepared route and re-arm the
    /// controller (same backend/transport). Quiescence that cannot be proven
    /// keeps the tape, quarantines the executor and returns `Err`.
    pub fn release(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.active.is_none() {
            return Ok(());
        }
        if let Err(e) = quiesce(gpu) {
            self.go_sticky(format!("quiesce before release failed: {e}"));
            self.quarantined = true;
            return Err(format!("inkling2 pm4: quiesce before release failed: {e}"));
        }
        if let Err(f) = gpu.replay.shutdown() {
            self.go_sticky(format!("shutdown failed: {}", f.error));
            if f.quiescence == ReplayQuiescence::Unknown {
                self.quarantined = true;
            }
            return Err(format!("inkling2 pm4: shutdown failed: {}", f.error));
        }
        gpu.replay.rearm_after_layout_growth();
        self.active = None;
        self.stats.releases += 1;
        Ok(())
    }

    fn go_sticky(&mut self, reason: String) {
        if self.sticky.is_none() {
            self.sticky = Some(reason);
        }
    }

    fn eager_reason(&self) -> Option<EagerWhy> {
        if !self.enabled {
            Some(EagerWhy::Disabled)
        } else {
            self.sticky.clone().map(EagerWhy::Sticky)
        }
    }

    fn hit(&self, gpu: &Gpu, key: &DecodeStepKey, layout: u64, identity: &str) -> Option<bool> {
        self.active.as_ref().map(|a| {
            a.key == *key
                && a.layout == layout
                && a.identity == identity
                && gpu.replay.state() == ReplayState::Ready
                && gpu.replay.prepared_pm4_plan_ready()
                && gpu.replay.route_identity(PM4_ROUTE_KEY) == Some(identity)
        })
    }

    fn preflight(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.quarantined {
            return Err("inkling2 pm4: executor quarantined (retirement unproven)".into());
        }
        if gpu.graphs.capture_mode {
            return Err("inkling2 pm4: refused inside a foreign graph capture".into());
        }
        if gpu.replay.is_recording() {
            return Err("inkling2 pm4: refused while a foreign recorder window is open".into());
        }
        if let Some(s) = gpu.active_stream.as_ref() {
            if gpu
                .hip
                .stream_is_capturing(s)
                .map_err(|e| format!("inkling2 pm4: capture status: {e:?}"))?
            {
                return Err("inkling2 pm4: refused on a stream that is capturing".into());
            }
        }
        if self.enabled && !pm4_dev_opt_in(gpu) {
            return Err(
                "inkling2 pm4: needs HIPFIRE_REPLAY_BACKEND=redline HIPFIRE_REPLAY_TRANSPORT=pm4 on gfx1201"
                    .into(),
            );
        }
        if self.enabled && self.sticky.is_none() && gpu.replay.state() == ReplayState::Fallback {
            let why = gpu
                .replay
                .fallback_reason()
                .unwrap_or("unknown")
                .to_string();
            self.go_sticky(format!("replay controller poisoned: {why}"));
        }
        Ok(())
    }

    /// One decode step at `pos` for `token_id`; the logits land in `logits` and
    /// the host tags/cursors commit exactly as the frozen eager step does.
    pub fn forward(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<Pm4StepReport, String> {
        self.preflight(gpu)?;
        if let Some(why) = self.eager_reason() {
            forward_decode_frozen(bundle, gpu, token_id, pos, logits)?;
            self.stats.eager_steps += 1;
            return Ok(Pm4StepReport {
                route: Pm4Route::Eager(why),
                record_ns: 0,
                prepare_ns: 0,
            });
        }
        ensure_decode_arena(gpu, bundle)?;
        if self.stream_ordered {
            gpu.ensure_capture_stream()
                .map_err(|e| format!("inkling2 pm4: owned stream: {e:?}"))?;
        }
        let plan = plan_decode_step(bundle, gpu, pos)?;
        let layout = layout_fingerprint(bundle, gpu)?;
        let identity = route_identity(&plan.key, layout);
        match self.hit(gpu, &plan.key, layout, &identity) {
            Some(true) => return self.replay_step(bundle, gpu, token_id, pos, logits, false),
            Some(false) => self.release(gpu)?,
            None => {}
        }
        if !self.warmed {
            // The real token of a fresh load runs eagerly (it resolves every
            // lazily built module and scratch) and is not recorded.
            let fx0 = memory_effects::snapshot();
            forward_decode_frozen(bundle, gpu, token_id, pos, logits)?;
            self.stats.eager_launches = memory_effects::snapshot().since(fx0).launches;
            self.stats.eager_steps += 1;
            self.stats.warm_steps += 1;
            self.warmed = true;
            return Ok(Pm4StepReport {
                route: Pm4Route::Warm,
                record_ns: 0,
                prepare_ns: 0,
            });
        }
        self.record_step(bundle, gpu, token_id, pos, logits, plan, layout, identity)
    }

    /// Proof arm: replay the exact recorded HIP launch blobs of the active
    /// tape (`replay_recorded_hip_prefix_at`), through the same stage and
    /// finish. Refuses unless a tape for this very key is held.
    pub fn forward_blob_oracle(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<Pm4StepReport, String> {
        self.preflight(gpu)?;
        let plan = plan_decode_step(bundle, gpu, pos)?;
        let layout = layout_fingerprint(bundle, gpu)?;
        let identity = route_identity(&plan.key, layout);
        if self.hit(gpu, &plan.key, layout, &identity) != Some(true) {
            return Err("inkling2 pm4: the blob oracle needs a prepared tape for this key".into());
        }
        self.replay_step(bundle, gpu, token_id, pos, logits, true)
    }

    /// Stage, submit (PM4 queue waited to retirement, or the blob oracle on
    /// HIP), finish.
    fn replay_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
        blob: bool,
    ) -> Result<Pm4StepReport, String> {
        let count = self.active.as_ref().ok_or("inkling2 pm4: no tape")?.count;
        let pos32 = u32::try_from(pos).map_err(|_| "inkling2 pm4: pos exceeds u32".to_string())?;
        let gw = bundle
            .gpu_weights
            .as_ref()
            .ok_or("inkling2 pm4: no gpu weights")?;
        let gs = bundle
            .gpu_state
            .as_mut()
            .ok_or("inkling2 pm4: no gpu state")?;
        let mut arena = gs
            .decode_arena
            .take()
            .ok_or("inkling2 pm4: no decode arena")?;
        // Staging is a refusal point: nothing has been submitted if it fails.
        if let Err(e) = stage_decode_input(gpu, gw, &mut arena, token_id, pos32) {
            gs.decode_arena = Some(arena);
            return Err(e);
        }
        let submitted: Result<(), (String, ReplayQuiescence)> = if blob {
            gpu.replay_recorded_hip_prefix_at(count, pos).map_err(|e| {
                (
                    format!("recorded-blob replay: {e:?}"),
                    ReplayQuiescence::Unknown,
                )
            })
        } else {
            // SAFETY: the tape's allocations are owned by this load and the key,
            // layout fingerprint and route identity were just re-validated.
            unsafe { gpu.replay_pm4_routed_checked(pos) }.map_err(|f| (f.error, f.quiescence))
        };
        let r = match submitted {
            Ok(()) => finish_decode_step(gpu, gs, &arena, pos, logits),
            Err((e, q)) => {
                // Execution may have begun: no commit, no retry. A blob failure
                // is only recoverable if the device drains.
                let q = if blob && quiesce(gpu).is_ok() {
                    ReplayQuiescence::Proven
                } else {
                    q
                };
                gpu.replay
                    .poison(format!("inkling2 pm4 replay failed: {e}"));
                self.go_sticky(format!("pm4 replay failed: {e}"));
                if q == ReplayQuiescence::Unknown {
                    self.quarantined = true;
                }
                Err(format!(
                    "inkling2 pm4: replay at pos {pos} failed ({q:?}): {e}"
                ))
            }
        };
        gs.decode_arena = Some(arena);
        match r {
            Ok(()) => {
                if blob {
                    self.stats.blob_replays += 1;
                } else {
                    self.stats.replays += 1;
                }
                Ok(Pm4StepReport {
                    route: Pm4Route::Replay,
                    record_ns: 0,
                    prepare_ns: 0,
                })
            }
            Err(e) => {
                self.go_sticky(format!("replay step failed: {e}"));
                Err(e)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        bundle: &mut Inkling2Bundle,
        gpu: &mut Gpu,
        token_id: u32,
        pos: usize,
        logits: &mut Vec<f32>,
        plan: DecodeStepPlan,
        layout: u64,
        identity: String,
    ) -> Result<Pm4StepReport, String> {
        let pos32 = u32::try_from(pos).map_err(|_| "inkling2 pm4: pos exceeds u32".to_string())?;
        let cfg = &bundle.config;
        let gw = bundle
            .gpu_weights
            .as_ref()
            .ok_or("inkling2 pm4: no gpu weights")?;
        let gs = bundle
            .gpu_state
            .as_mut()
            .ok_or("inkling2 pm4: no gpu state")?;
        let mut arena = gs
            .decode_arena
            .take()
            .ok_or("inkling2 pm4: no decode arena")?;
        let r = self.record_inner(
            gpu, cfg, gw, gs, &mut arena, token_id, pos, pos32, &plan, layout, identity, logits,
        );
        gs.decode_arena = Some(arena);
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn record_inner(
        &mut self,
        gpu: &mut Gpu,
        cfg: &Inkling2Config,
        gw: &Inkling2GpuWeights,
        gs: &mut Inkling2GpuState,
        arena: &mut DecodeArena,
        token_id: u32,
        pos: usize,
        pos32: u32,
        plan: &DecodeStepPlan,
        layout: u64,
        identity: String,
        logits: &mut Vec<f32>,
    ) -> Result<Pm4StepReport, String> {
        // Staging is outside the window (and a refusal point).
        stage_decode_input(gpu, gw, arena, token_id, pos32)?;
        let t0 = Instant::now();
        // Recorder refusals before the body ran: the step is still a valid
        // eager token; the route goes sticky eager.
        let open = gpu
            .replay
            .begin_capture()
            .map_err(|e| format!("begin_capture refused: {e}"))
            .and_then(|()| {
                gpu.replay.set_forward_eligible(true);
                gpu.replay.note_route_identity(PM4_ROUTE_KEY, &identity)
            });
        if let Err(reason) = open {
            gpu.replay.rearm_after_layout_growth();
            enqueue_decode_body(gpu, cfg, gw, gs, arena, plan)?;
            finish_decode_step(gpu, gs, arena, pos, logits)?;
            return Ok(self.record_failed(reason));
        }
        // The recording token: ordinary HIP executes the body while the
        // launches are recorded.
        let fx0 = memory_effects::snapshot();
        let body = enqueue_decode_body(gpu, cfg, gw, gs, arena, plan);
        let fx = memory_effects::snapshot().since(fx0);
        if let Err(e) = body {
            gpu.replay.rearm_after_layout_growth();
            self.stats.record_failures += 1;
            self.go_sticky(format!("recording the body: {e}"));
            return Err(format!("inkling2 pm4: recording the body: {e}"));
        }
        let summary = gpu.replay.finish_capture();
        if let Err(e) = quiesce(gpu) {
            gpu.replay.rearm_after_layout_growth();
            self.stats.record_failures += 1;
            self.go_sticky(format!("quiesce after the recording token: {e}"));
            return Err(format!(
                "inkling2 pm4: quiesce after the recording token: {e}"
            ));
        }
        let record_ns = t0.elapsed().as_nanos() as u64;
        // The recording token is a real token: commit it whatever happens to
        // the tape.
        if let Err(e) = finish_decode_step(gpu, gs, arena, pos, logits) {
            gpu.replay.rearm_after_layout_growth();
            self.stats.record_failures += 1;
            self.go_sticky(format!("finish of the recording token: {e}"));
            return Err(e);
        }
        // Census, then prepare (both untimed in steady state).
        let t1 = Instant::now();
        let prepared = (|| -> Result<(usize, u32, u64), String> {
            let summary = summary.map_err(|e| format!("finish_capture: {e}"))?;
            let count = summary.launch_count;
            if count == 0
                || fx.launches as usize != count
                || gpu.replay.retained_body_launches() != count
                || count as u64 != self.stats.eager_launches
                || (fx.htod, fx.dtod, fx.dtoh, fx.memset) != (0, 0, 0, 0)
                || !gpu.replay.capture_window_effects().replayable()
            {
                return Err(format!(
                    "capture census: recorded {count} / launches in window {} / body {} / warm eager {} / effects {fx:?}",
                    fx.launches,
                    gpu.replay.retained_body_launches(),
                    self.stats.eager_launches
                ));
            }
            let (n, dwords, queue) = gpu
                .replay
                .prepare_pm4_prefix(gpu.device_id as usize, count)?;
            if n != count
                || gpu.replay.state() != ReplayState::Ready
                || gpu.replay.route_identity(PM4_ROUTE_KEY) != Some(identity.as_str())
            {
                return Err(format!(
                    "prepare: {n} dispatches of {count}, state {:?}",
                    gpu.replay.state()
                ));
            }
            Ok((count, dwords, queue))
        })();
        let prepare_ns = t1.elapsed().as_nanos() as u64;
        match prepared {
            Ok((count, dwords, queue)) => {
                let syms: BTreeSet<String> = gpu
                    .replay
                    .recorded_launches()
                    .iter()
                    .map(|l| l.kernel.clone())
                    .collect();
                self.symbols = syms.into_iter().collect();
                self.stats.recordings += 1;
                self.stats.nodes = count;
                self.stats.unique_symbols = self.symbols.len();
                self.stats.command_dwords = dwords;
                self.stats.queue_id = queue;
                self.stats.last_record_ns = record_ns;
                self.stats.last_prepare_ns = prepare_ns;
                self.stats.total_record_ns += record_ns;
                self.stats.total_prepare_ns += prepare_ns;
                self.active = Some(ActivePm4 {
                    key: plan.key.clone(),
                    layout,
                    identity,
                    count,
                });
                Ok(Pm4StepReport {
                    route: Pm4Route::Recorded,
                    record_ns,
                    prepare_ns,
                })
            }
            Err(reason) => {
                gpu.replay.rearm_after_layout_growth();
                Ok(self.record_failed(reason))
            }
        }
    }

    fn record_failed(&mut self, reason: String) -> Pm4StepReport {
        self.stats.record_failures += 1;
        self.stats.eager_steps += 1;
        self.go_sticky(reason.clone());
        Pm4StepReport {
            route: Pm4Route::Eager(EagerWhy::CaptureFailed(reason)),
            record_ns: 0,
            prepare_ns: 0,
        }
    }
}

/// Route identity string latched on the tape: the decode key, the layout
/// fingerprint and the arch.
fn route_identity(key: &DecodeStepKey, layout: u64) -> String {
    format!(
        "ink2-c1-decode;gen={};epoch={};arch={}#{};bucket={}-{};sched={:016x};layout={layout:016x}",
        key.load_generation,
        key.layout_epoch,
        key.device.arch,
        key.device.device_id,
        key.bucket.lo,
        key.bucket.hi,
        key.schedule.digest(),
    )
}
