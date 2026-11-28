// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HIP-graph executor over the frozen one-row decode body (dev ruler / oracle
//! comparator; no product wiring).
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
use crate::decode_step::{
    enqueue_decode_body, ensure_decode_arena, finish_decode_step, forward_decode_frozen,
    plan_decode_step, stage_decode_input, DecodeStepKey,
};
use crate::gpu::Inkling2GpuState;
use hip_bridge::memory_effects;
use rdna_compute::Gpu;
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
        let hit = self
            .active
            .as_ref()
            .map(|a| a.key == plan.key && a.layout == layout);
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
