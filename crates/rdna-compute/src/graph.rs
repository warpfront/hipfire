// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Graph-capture lifecycle for AR forward, DFlash verify, and DeltaNet replay.

use hip_bridge::{Graph, GraphExec, HipError, HipResult, HipRuntime, Stream};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

/// Set once any hipGraph is instantiated in this process.
///
/// A captured graph's nodes embed the device pointers that were live at
/// capture time. Releasing one of those buffers later makes every subsequent
/// replay read freed memory, which surfaces as
/// `HipError(700): illegal memory access` at the next synchronisation point —
/// typically somewhere unrelated, because the fault is reported late.
///
/// `scratch::grow_scratch_buffer` consults this before releasing a replaced
/// scratch buffer: architectures that never capture (Muse Glimmer has no
/// hipGraph decode path by design) free immediately, while architectures
/// that do capture (qwen35's verify / replay / AR-forward graphs) invalidate
/// first. Freeing under a live graph was measured to break qwen35 outright
/// — every turn of a 4-turn DFlash session returned empty with
/// `spec_step: HipError(700) ... reset_recurrent` — so every `Gpu` scratch
/// caller checks `scratch::scratch_will_grow` before delegating and drops
/// all captured execution state via `Gpu::invalidate_for_scratch_growth`
/// (the model-swap teardown `invalidate_for_layout_growth`: AR graph,
/// verify / replay graphs, retained Redline route). Graphs re-capture
/// lazily on the next replay; each growth event costs at most one
/// re-capture. An earlier revision retained the old buffers with
/// `std::mem::forget` instead — that leak is fixed; no path retains.
static ANY_GRAPH_CAPTURED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// True once any hipGraph has been instantiated in this process.
pub(crate) fn any_graph_captured() -> bool {
    ANY_GRAPH_CAPTURED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Latch the graph-captured flag. Called from every graph instantiation site.
pub(crate) fn mark_graph_captured() {
    ANY_GRAPH_CAPTURED.store(true, std::sync::atomic::Ordering::Relaxed);
}

// Thread-local cache of the last device id bound on this thread.
// Shared with `Gpu::bind_thread` / `Gpu::bind_thread_or_warn` in dispatch.rs.
thread_local! {
    pub(crate) static LAST_BOUND_DEVICE: Cell<i32> = const { Cell::new(-1) };
}

/// Bind the given device on the calling thread. Cached via thread_local
/// — only issues `hipSetDevice` when the cached id changes.
#[inline]
pub(crate) fn bind_thread(hip: &HipRuntime, device_id: i32) -> HipResult<()> {
    LAST_BOUND_DEVICE.with(|c| {
        if c.get() != device_id {
            hip.set_device(device_id)?;
            c.set(device_id);
        }
        Ok(())
    })?;
    debug_assert_eq!(
        hip.current_device()?,
        device_id,
        "bind_thread invariant: current device must match device_id",
    );
    Ok(())
}

/// `bind_thread` for infallible / `Drop` contexts. Logs to stderr on
/// `hipSetDevice` failure instead of swallowing it silently.
#[inline]
pub(crate) fn bind_thread_or_warn(hip: &HipRuntime, device_id: i32) {
    LAST_BOUND_DEVICE.with(|c| {
        if c.get() != device_id {
            match hip.set_device(device_id) {
                Ok(()) => c.set(device_id),
                Err(e) => eprintln!(
                    "WARN: bind_thread_or_warn(dev {}) failed: {} — \
                     subsequent ops run on the currently-bound device",
                    device_id, e,
                ),
            }
        }
    });
}

/// Per-B graph cache: verify (DFlash) and replay (DeltaNet tape) share this pattern.
/// Does not implement Clone because `Graph` / `GraphExec` are not Clone.
pub struct PerBGraphCache {
    pub cache: HashMap<usize, (Graph, GraphExec, Vec<Vec<u8>>)>,
    pub warmed_up: HashSet<usize>,
    /// Size being captured right now (between begin_* and end_*). None outside
    /// that window.
    pub capturing: Option<usize>,
    /// Subset of cache entries whose captured region also includes the
    /// DFlash verify lm_head + argmax tail. Callers check this before
    /// deciding whether to enqueue lm_head outside the graph.
    pub lmhead_argmax: HashSet<usize>,
}

/// Graph-capture state split across AR forward, DFlash verify, and DeltaNet replay.
pub struct GraphState {
    // AR forward (single-slot)
    pub capture_mode: bool,
    pub capture_blobs: Vec<Vec<u8>>,
    pub graph_exec: Option<GraphExec>,
    pub captured_graph: Option<Graph>,
    /// Kernarg blobs OWNED by the captured AR graph. Drained out of the shared
    /// `capture_blobs` at `end_graph_capture` (mirrors the verify cache's
    /// per-entry blob ownership at line ~235) so an interleaved verify capture —
    /// which CLEARS `capture_blobs` in `begin_verify_graph_capture` — cannot
    /// dangle the AR graph's kernarg pointers. Kept alive as long as
    /// `graph_exec`; freed in `drop_captured_graph` / `graph_destroy`.
    pub ar_forward_blobs: Vec<Vec<u8>>,
    pub ar_forward_kernel_dirty: bool,
    pub ar_forward_replay_enabled: bool,
    /// One-shot AR-graph eligibility, CONSUMED (reset to **true**) on read in
    /// `forward_scratch`. Plain sequential single-token decode is eligible by
    /// default; the spec-decode / MTP / verify callers set it FALSE right before
    /// their `forward_scratch` call so the plain-AR graph can't capture/replay
    /// in their non-sequential context.
    pub ar_graph_eligible: bool,

    /// Per-segment AR graphs for dense TP decode. Each entry owns its kernarg
    /// blobs (drained from shared `capture_blobs` at `end_graph_capture_segment`)
    /// so segment lifetime is independent of later captures that clear the
    /// shared blob vec. Destroyed in `drop_graph_segments` / `drop_captured_graph`
    /// / `graph_destroy`.
    pub ar_segments: Vec<(Graph, GraphExec, Vec<Vec<u8>>)>,

    // Verify (DFlash, per-B)
    pub verify: PerBGraphCache,

    // Replay (DeltaNet tape, per-n_steps)
    pub replay: PerBGraphCache,
}

impl GraphState {
    // ── hipGraph capture/replay (AR forward) ──────────────────────────────

    /// Begin capturing all kernel launches on the active stream into a graph.
    /// While capturing, dispatch methods that support it will use the blob
    /// launch path so that kernarg pointers survive until graph replay.
    pub fn begin_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        self.capture_blobs.clear();
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0) // 0 = hipStreamCaptureModeGlobal
    }

    /// Begin a graph capture that may contain system-scope coordination with
    /// independently captured peer-device streams. ROCm rejects that shape in
    /// Global mode; Relaxed mode keeps each rank graph independent while the
    /// captured signal kernels provide the explicit ordering contract.
    pub fn begin_graph_capture_relaxed(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        self.capture_blobs.clear();
        self.capture_mode = true;
        if let Err(e) = hip.stream_begin_capture(stream, 2) {
            // 2 = hipStreamCaptureModeRelaxed
            self.capture_mode = false;
            return Err(e);
        }
        Ok(())
    }

    /// End capture, instantiate the graph for replay.
    pub fn end_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        mark_graph_captured();
        self.captured_graph = Some(graph);
        self.graph_exec = Some(exec);
        // Take OWNERSHIP of this graph's kernarg blobs out of the shared
        // `capture_blobs` (mirrors `end_verify_graph_capture`). The heap
        // allocations move with the Vec, so the graph nodes' kernarg pointers
        // stay valid; and a later `begin_verify_graph_capture` clearing
        // `capture_blobs` can no longer dangle them.
        self.ar_forward_blobs = std::mem::take(&mut self.capture_blobs);
        Ok(())
    }

    /// End a per-segment capture (typically after `begin_graph_capture_relaxed`),
    /// instantiate, and push ownership of the graph + kernarg blobs.
    pub fn end_graph_capture_segment(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        // Always exit capture mode — even when bind/end/instantiate fails —
        // so a partial segment attempt cannot leave dispatch in blob mode.
        self.capture_mode = false;
        if let Err(e) = bind_thread(hip, device_id) {
            self.capture_blobs.clear();
            return Err(e);
        }
        let graph = match hip.stream_end_capture(stream) {
            Ok(g) => g,
            Err(e) => {
                self.capture_blobs.clear();
                return Err(e);
            }
        };
        let exec = match hip.graph_instantiate(&graph) {
            Ok(exec) => exec,
            Err(e) => {
                let _ = hip.graph_destroy(graph);
                self.capture_blobs.clear();
                return Err(e);
            }
        };
        mark_graph_captured();
        // Only move blobs into the segment after instantiate succeeds.
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.ar_segments.push((graph, exec, blobs));
        Ok(())
    }

    /// How many captured AR segment graphs are retained.
    pub fn graph_segment_count(&self) -> usize {
        self.ar_segments.len()
    }

    /// Abort an in-flight capture without instantiating or retaining a graph.
    /// Used when the capture body fails: leaves capture_mode off, drops partial
    /// blobs, ends the stream capture, and destroys the returned Graph.
    pub fn abort_graph_capture(&mut self, hip: &HipRuntime, device_id: i32, stream: &Stream) {
        self.capture_mode = false;
        self.capture_blobs.clear();
        bind_thread_or_warn(hip, device_id);
        if let Ok(graph) = hip.stream_end_capture(stream) {
            let _ = hip.graph_destroy(graph);
        }
    }

    /// Replay a previously captured AR segment by index.
    pub fn graph_segment_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        segment: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let entry = self
            .ar_segments
            .get(segment)
            .ok_or_else(|| HipError::new(0, &format!("no captured AR graph segment {segment}")))?;
        hip.graph_launch(&entry.1, stream)
    }

    /// Destroy every per-segment AR graph and clear shared capture blobs.
    pub fn drop_graph_segments(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        for (graph, exec, _blobs) in self.ar_segments.drain(..) {
            let _ = hip.graph_exec_destroy(exec);
            let _ = hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
    }

    /// Replay the captured graph.
    pub fn graph_launch(&self, hip: &HipRuntime, device_id: i32, stream: &Stream) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let exec = self
            .graph_exec
            .as_ref()
            .expect("no captured graph to replay");
        hip.graph_launch(exec, stream)
    }

    /// Caller signals end of a decode turn (EOS or max_tokens reached). If a
    /// captured graph exists and kernels are clean, replay is enabled for the
    /// next decode turn. Per the AR-forward hipGraph policy: "at least one
    /// captured full turn must run before replay can be enabled."
    /// No-op if no capture exists (e.g., turn ran fully direct because kernels
    /// were dirty or graph was disabled by the caller).
    pub fn end_decode_turn(&mut self) {
        if !self.ar_forward_kernel_dirty && self.graph_exec.is_some() {
            self.ar_forward_replay_enabled = true;
        }
    }

    /// Drop the currently captured graph (if any) without touching kernel /
    /// replay state. Used by the capture+launch hot-path to free the previous
    /// per-call capture before recording a fresh one — bare `graph_destroy()`
    /// would also mark kernels dirty + disable replay, which is wrong here.
    pub fn drop_captured_graph(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        if let Some(exec) = self.graph_exec.take() {
            let _ = hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
        self.ar_forward_blobs.clear();
        self.drop_graph_segments(hip, device_id);
    }

    /// Caller signals a kernel-module change (model load, dtype switch, etc).
    /// Forces the next AR forward call to dispatch direct (no capture) so any
    /// inline JIT / lazy hipMalloc happens outside a captured region. Replay
    /// stays disabled until a fresh full turn completes via `end_decode_turn`.
    pub fn mark_kernels_dirty(&mut self) {
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
    }

    /// Destroy the captured graph and free all retained kernarg blobs.
    pub fn graph_destroy(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        if let Some(exec) = self.graph_exec.take() {
            let _ = hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.captured_graph.take() {
            let _ = hip.graph_destroy(graph);
        }
        self.capture_blobs.clear();
        self.ar_forward_blobs.clear();
        self.ar_forward_kernel_dirty = true;
        self.ar_forward_replay_enabled = false;
        self.drop_graph_segments(hip, device_id);
    }

    // ── Per-B verify-forward graph cache ─────────────────────────────────

    /// Does a captured verify graph exist for batch size `b`?
    pub fn verify_has_graph(&self, b: usize) -> bool {
        self.verify.cache.contains_key(&b)
    }

    /// Does `b` need a warmup pass before capture can begin?
    pub fn verify_needs_warmup(&self, b: usize) -> bool {
        !self.verify.warmed_up.contains(&b)
    }

    /// Mark `b` as having completed its warmup.
    pub fn verify_mark_warmup_done(&mut self, b: usize) {
        self.verify.warmed_up.insert(b);
    }

    /// Begin capturing a verify-forward graph for batch size `b`. Subsequent
    /// `launch_maybe_blob` calls will push their kernargs into `capture_blobs`,
    /// which is drained into the per-B cache entry on `end_verify_graph_capture`.
    pub fn begin_verify_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        b: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        debug_assert!(
            self.verify.capturing.is_none(),
            "begin_verify_graph_capture: already capturing for b={:?}",
            self.verify.capturing
        );
        debug_assert!(
            !self.capture_mode,
            "begin_verify_graph_capture: capture_mode already set"
        );
        self.capture_blobs.clear();
        // A verify forward is about to run on the shared buffers — invalidate the
        // plain-AR graph so it can't replay across this spec excursion (the next
        // plain-AR forward will re-capture). Defense-in-depth alongside the
        // caller-side `ar_graph_eligible=false` + position-continuity gate.
        self.ar_forward_replay_enabled = false;
        self.ar_forward_kernel_dirty = true;
        self.verify.capturing = Some(b);
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0) // hipStreamCaptureModeGlobal
    }

    /// End capture, instantiate, stash into the per-B cache (taking ownership
    /// of the current `capture_blobs`).
    pub fn end_verify_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let b = self
            .verify
            .capturing
            .take()
            .expect("end_verify_graph_capture without matching begin");
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        mark_graph_captured();
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.verify.cache.insert(b, (graph, exec, blobs));
        Ok(())
    }

    /// Replay the cached verify graph for batch size `b`.
    pub fn verify_graph_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        b: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let entry = self
            .verify
            .cache
            .get(&b)
            .unwrap_or_else(|| panic!("no captured verify graph for b={}", b));
        hip.graph_launch(&entry.1, stream)
    }

    /// How many captured verify graphs are in the cache (for debug logs).
    pub fn verify_graph_count(&self) -> usize {
        self.verify.cache.len()
    }

    /// Does the captured verify graph for `b` include the lm_head + argmax tail?
    pub fn verify_graph_has_lmhead_argmax(&self, b: usize) -> bool {
        self.verify.lmhead_argmax.contains(&b)
    }

    /// Mark the captured verify graph for `b` as including lm_head + argmax.
    pub fn verify_mark_graph_lmhead_argmax(&mut self, b: usize) {
        self.verify.lmhead_argmax.insert(b);
    }

    /// Destroy all cached verify graphs and their blobs.
    pub fn verify_graph_destroy_all(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        for (_, (graph, exec, _blobs)) in self.verify.cache.drain() {
            let _ = hip.graph_exec_destroy(exec);
            let _ = hip.graph_destroy(graph);
        }
        self.verify.warmed_up.clear();
        self.verify.lmhead_argmax.clear();
        self.verify.capturing = None;
    }

    // ── Replay-graph cache (tape replay after verify) ────────────────────

    /// Does a captured replay graph exist for `n_steps`?
    pub fn replay_has_graph(&self, n_steps: usize) -> bool {
        self.replay.cache.contains_key(&n_steps)
    }

    /// Does `n_steps` need a warmup pass before capture can begin?
    pub fn replay_needs_warmup(&self, n_steps: usize) -> bool {
        !self.replay.warmed_up.contains(&n_steps)
    }

    /// Mark `n_steps` as having completed its warmup.
    pub fn replay_mark_warmup_done(&mut self, n_steps: usize) {
        self.replay.warmed_up.insert(n_steps);
    }

    /// Begin capturing a replay graph for `n_steps`.
    pub fn begin_replay_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        n_steps: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        debug_assert!(
            self.replay.capturing.is_none(),
            "begin_replay_graph_capture: already capturing for n_steps={:?}",
            self.replay.capturing
        );
        debug_assert!(
            !self.capture_mode,
            "begin_replay_graph_capture: capture_mode already set"
        );
        self.capture_blobs.clear();
        self.replay.capturing = Some(n_steps);
        self.capture_mode = true;
        hip.stream_begin_capture(stream, 0)
    }

    /// End capture, instantiate, stash into the per-n_steps cache.
    pub fn end_replay_graph_capture(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let n_steps = self
            .replay
            .capturing
            .take()
            .expect("end_replay_graph_capture without matching begin");
        self.capture_mode = false;
        let graph = hip.stream_end_capture(stream)?;
        let exec = hip.graph_instantiate(&graph)?;
        mark_graph_captured();
        let blobs = std::mem::take(&mut self.capture_blobs);
        self.replay.cache.insert(n_steps, (graph, exec, blobs));
        Ok(())
    }

    /// Replay the cached replay graph for `n_steps`.
    pub fn replay_graph_launch(
        &self,
        hip: &HipRuntime,
        device_id: i32,
        stream: &Stream,
        n_steps: usize,
    ) -> HipResult<()> {
        bind_thread(hip, device_id)?;
        let entry = self
            .replay
            .cache
            .get(&n_steps)
            .unwrap_or_else(|| panic!("no captured replay graph for n_steps={}", n_steps));
        hip.graph_launch(&entry.1, stream)
    }

    /// How many captured replay graphs are in the cache (for debug logs).
    pub fn replay_graph_count(&self) -> usize {
        self.replay.cache.len()
    }

    /// Destroy all cached replay graphs and their blobs.
    pub fn replay_graph_destroy_all(&mut self, hip: &HipRuntime, device_id: i32) {
        bind_thread_or_warn(hip, device_id);
        for (_, (graph, exec, _blobs)) in self.replay.cache.drain() {
            let _ = hip.graph_exec_destroy(exec);
            let _ = hip.graph_destroy(graph);
        }
        self.replay.warmed_up.clear();
        self.replay.capturing = None;
    }
}
