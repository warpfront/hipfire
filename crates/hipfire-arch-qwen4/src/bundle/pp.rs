// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 pipeline-parallel (`topology.pp`) ownership and the bundle entry
//! points that route through it.
//!
//! The published bundle is the output band (the root). It owns the peer bands
//! `0..n-1`; the one [`Gpus`] that owns every logical device of the mesh stays
//! with the caller and is passed to every entry point. The root's own
//! `execution` is its forward; each peer's `execution` is its band's forward.
//! There is no family topology, thread pool or generation loop here: a call
//! splits the request into canonical tiles, submits them in order through the
//! generic executor, then drains the whole request once.
//!
//! * Every GPU model operation runs on `gpus.devices` (logical device `i` is
//!   band `i`). The daemon's `Gpu` is ignored for PP forwards.
//! * Generation logits/top-1 tensors are allocated by the daemon on the
//!   output band's physical device and are written directly by the output
//!   band. All tiles are drained (one all-device synchronization) before the
//!   call returns, so the daemon observes the rows complete.
//! * Tiles are the common canonical prefill chunk, in order, never padded;
//!   only the last tile computes the head for `Final` rows. Routing rows are
//!   the actual rows; the explicit [`MeshRoute`] is preserved.
//! * A submit or drain failure quarantines the mesh: it refuses all further
//!   work. A failed drain keeps every resource (including the root's) alive
//!   until process teardown.

use super::*;
use crate::config::Qwen4Config;
use crate::gpu_forward::{
    qwen4_prefill_segment_rows, Qwen4MeshHead, Qwen4MeshLayout, SessionBoundaries,
};
use crate::mesh_bindings::{
    session_tickets, Qwen4MeshBindings, Qwen4RankBand, WindowCtx, WindowFinalize, WindowRestore,
};
use crate::program::qwen4_step_graph_with_output;
use crate::weights::Qwen4BandScope;
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{split_program, SplitPlan};
use hipfire_runtime::multi_gpu::executor::{
    ForwardEpoch, MeshCheckpoint, MeshCheckpointKey, MeshLifecycleError,
};
use hipfire_runtime::multi_gpu::transport::EdgeId;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::session_cache::{MeshSessionCache, MeshSessionRecord};
use hipfire_runtime::weight_manifest::{plan_manifest, StateEntry, WeightEntry};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "pp_lanes.rs"]
mod lanes;
pub use lanes::Qwen4PpLaneExecutor;

/// Band identity of one PP bundle, stamped on every band (root included) at
/// attach. It carries what a session scope and boundary schedule need while the
/// band's forward is taken or the mesh holder is detached: band index/count,
/// the band's layer range, the COMPLETE placement, the physical device of every
/// band, the mesh and load epoch, and the canonical prefill chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4PpBandMeta {
    index: usize,
    layers: Range<usize>,
    placement: Vec<Range<usize>>,
    devices: Vec<String>,
    mesh_epoch: u64,
    load_id: u64,
    chunk: usize,
}

impl Qwen4PpBandMeta {
    /// `devices[i]` names band `i`'s physical device (`{device_id}@{arch}`).
    pub(crate) fn new(
        index: usize,
        placement: &BandPlacement,
        devices: Vec<String>,
        mesh_epoch: u64,
        load_id: u64,
        chunk: usize,
    ) -> Self {
        let bands = (0..placement.stages())
            .map(|stage| placement.band(stage))
            .collect();
        Self::from_bands(index, bands, devices, mesh_epoch, load_id, chunk)
    }

    fn from_bands(
        index: usize,
        placement: Vec<Range<usize>>,
        devices: Vec<String>,
        mesh_epoch: u64,
        load_id: u64,
        chunk: usize,
    ) -> Self {
        Self {
            index,
            layers: placement[index].clone(),
            placement,
            devices,
            mesh_epoch,
            load_id,
            chunk,
        }
    }

    /// The canonical prefill chunk every band of the mesh shares.
    pub(crate) fn chunk(&self) -> usize {
        self.chunk
    }

    /// The scope suffix that keeps this band's pieces from aliasing any other
    /// band, placement, device set, mesh generation or load.
    pub(crate) fn scope_tag(&self) -> String {
        let join = |parts: Vec<String>| parts.join(",");
        let placement = join(
            self.placement
                .iter()
                .map(|band| format!("{}-{}", band.start, band.end))
                .collect(),
        );
        format!(
            "/pp{}of{}/layers{}-{}/place{}/dev{}/mesh{}/load{}",
            self.index,
            self.placement.len(),
            self.layers.start,
            self.layers.end,
            placement,
            join(self.devices.clone()),
            self.mesh_epoch,
            self.load_id,
        )
    }
}

/// The canonical session schedule of one prefill, computed once by the root and
/// consumed identically by every band.
struct SessionRun<'r> {
    prompt: &'r [u32],
    route: SessionRoute,
    /// First prompt position this prefill computes (the restored prefix).
    start: usize,
    tiles: Vec<Tile>,
    /// Ascending snapshot boundaries; a tile ends on every one inside the
    /// prefill, and every band captures there.
    boundaries: Vec<usize>,
    /// The tail-split row among `boundaries`: captured as a band-local tail,
    /// never an aggregate record (the tail is not a pool piece).
    lineage: Option<usize>,
}

/// The cold schedule of a prefill over `start..end`, all tiles up front. Same
/// scheduler as Single's `prefill_schedule_rows` (`qwen4_prefill_segment_rows`,
/// no NPU pairs) driven by a virtual cursor over the full boundary list, since
/// the driver's own cursor only advances as boundaries are captured: tiles are
/// the global chunks cut at every boundary and the lineage split, never
/// re-chunked relative to the restored prefix.
fn prefill_schedule(
    start: usize,
    end: usize,
    chunk: usize,
    boundaries: &[usize],
    lineage: Option<usize>,
) -> Vec<Tile> {
    let chunk = chunk.max(1);
    let mut tiles = Vec::new();
    let mut position = start;
    while position < end {
        let stop = lineage
            .filter(|&split| split > position)
            .map_or(end, |split| split.min(end));
        let mut upcoming = boundaries.iter().copied().filter(|&at| at > position);
        let next = upcoming.next();
        let after = upcoming.next();
        let rows = qwen4_prefill_segment_rows(
            position,
            stop,
            chunk,
            SessionBoundaries { next, after },
            false,
        );
        tiles.push(Tile {
            offset: position - start,
            rows,
        });
        position += rows;
    }
    tiles
}

/// The pieces of an aggregate candidate captured at `end`: every band named a
/// cold pool piece and `end` is not the lineage (tail) split. `None` means no
/// record can ever cover this boundary, so none is staged.
fn aggregate_candidate<P>(
    lineage: Option<usize>,
    end: usize,
    pieces: Vec<Option<P>>,
) -> Option<Vec<P>> {
    if lineage == Some(end) {
        return None;
    }
    pieces.into_iter().collect()
}

/// The refusal of message-end snapshots under PP, the same sentence admission
/// gives (`qwen4_pp_unsupported`): attach and the setter use it verbatim.
pub(super) fn turn_snapshots_refusal(stages: usize) -> String {
    format!(
        "qwen4: pipeline parallelism (topology.pp = {stages}) does not support session snapshots (HIPFIRE_QWEN4_TURN_SNAPSHOTS); drop the option or set topology.pp = 1"
    )
}

/// Attach-time session verdict from per-band facts (index order): whether each
/// band holds a session cache and whether each has message-end snapshots on.
/// Message-end snapshots refuse; all bands cached = sessions on, none cached =
/// session-free; a mix refuses (a valid cache is never silently left unused).
fn attach_session_policy(
    stages: usize,
    present: &[bool],
    turn_snapshots: &[bool],
) -> Result<bool, String> {
    if turn_snapshots.iter().any(|&on| on) {
        return Err(turn_snapshots_refusal(stages));
    }
    let cached = present.iter().filter(|&&has| has).count();
    match cached {
        0 => Ok(false),
        n if n == present.len() => Ok(true),
        _ => Err(
            "the bands disagree on session-cache presence: every band attaches its own cache or none does"
                .to_string(),
        ),
    }
}

/// Band `index` of the mesh: the peers `0..n-1`, then the root.
fn band_mut<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
    index: usize,
) -> &'a mut Qwen4Bundle {
    if index < bands.len() {
        &mut bands[index]
    } else {
        root
    }
}

/// Every band's session driver in owner order; `None` unless all have one.
fn session_drivers<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
) -> Option<Vec<&'a mut SessionDriver>> {
    let mut drivers = Vec::with_capacity(bands.len() + 1);
    for bundle in bands.iter_mut().chain(std::iter::once(root)) {
        drivers.push(bundle.session.as_mut()?);
    }
    Some(drivers)
}

/// Borrowed rank bands over the taken forwards (peers then root).
fn mesh_ranks<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
    forwards: &'a mut [Qwen4GpuForward],
) -> Vec<Qwen4RankBand<'a>> {
    let (peer_forwards, root_forward) = forwards.split_at_mut(bands.len());
    let mut ranks = Vec::with_capacity(bands.len() + 1);
    for (bundle, forward) in bands.iter_mut().zip(peer_forwards.iter_mut()) {
        ranks.push(Qwen4RankBand { bundle, forward });
    }
    ranks.push(Qwen4RankBand {
        bundle: root,
        forward: &mut root_forward[0],
    });
    ranks
}

/// The epoch a lifecycle call (reset, session restore) runs under: current mesh,
/// load and request. Its op id is not checked by those calls.
fn lifecycle_epoch(mesh: &Qwen4PpMesh, gpus: &Gpus, start_position: usize) -> ForwardEpoch {
    ForwardEpoch {
        mesh_epoch: gpus.mesh.epoch(),
        load_id: mesh.load_id,
        request_id: mesh.request_id,
        op_id: mesh.next_op,
        route: MeshRoute::Prefill,
        start_position,
        actual_rows: 0,
        route_rows: 0,
        policy_id: PP_POLICY_ID,
    }
}

/// Before any band maps context through `end`: while a band's growth does not
/// fit even after its cache released every unpinned snapshot, drop the oldest
/// aggregate record (releasing all its pins) so the band's own
/// `release_session_for_growth` can free those pieces. Once per horizon, never
/// clearing records that are not in the way: the cache stays opportunistic and
/// never holds context hostage.
fn reclaim_records_for_growth(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    end: usize,
) {
    if mesh.sessions.is_empty() {
        return;
    }
    let Qwen4PpMesh {
        bands, sessions, ..
    } = mesh;
    let stages = bands.len() + 1;
    loop {
        let mut short = false;
        for index in 0..stages {
            let bundle = band_mut(bands, root, index);
            let gpu = &mut gpus.devices[index];
            if gpu.bind_thread().is_err() || end <= bundle.state.mapped_context_tokens() {
                continue;
            }
            let (context, gather) = bundle.context_growth_bytes(gpu, end);
            let need = context.saturating_add(gather);
            let Some(session) = bundle.session.as_mut() else {
                continue;
            };
            if need == 0 || !session.needs_release(gpu, need) {
                continue;
            }
            session.release_for(gpu, need);
            short |= session.needs_release(gpu, need);
        }
        if !short {
            break;
        }
        let Some(mut drivers) = session_drivers(bands, root) else {
            break;
        };
        if !sessions.evict_oldest(&mut drivers) {
            break;
        }
    }
}

/// Numerical-policy identity of every PP epoch of this build.
const PP_POLICY_ID: u64 = 1;
const PP_MAX_BANDS: usize = 8;

/// Split-plan slots, one per output mode of a tile.
const MODE_TRUNK: usize = 0;
const MODE_LOGITS: usize = 1;
const MODE_LOGITS_TOP1: usize = 2;
const MODE_COUNT: usize = 3;

static NEXT_LOAD_ID: AtomicU64 = AtomicU64::new(1);

/// Next process-unique load id; fails instead of wrapping.
fn next_load_id() -> Result<u64, String> {
    NEXT_LOAD_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "load ids are exhausted".to_string())
}

/// Why unload must leak every resource instead of freeing: a failed PLE
/// quiesce (host readers may still run) or a failed device drain (in-flight
/// work may still reference them). `None` = safe to free.
fn teardown_leak_reason(
    quiesce: &Option<BundleError>,
    drained: &Result<(), String>,
) -> Option<String> {
    let mut reasons = Vec::new();
    if let Some(error) = quiesce {
        reasons.push(format!("PLE quiesce failed: {error}"));
    }
    if let Err(error) = drained {
        reasons.push(format!("device drain failed: {error}"));
    }
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

fn pp_error(message: impl AsRef<str>) -> BundleError {
    BundleError::Forward(format!("qwen4 topology.pp: {}", message.as_ref()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Health {
    Ready,
    /// Refuses all work; the reason is the first failure.
    Quarantined(String),
}

/// Attached PP bands, owned by the output-band root (`Qwen4Bundle::pp`). It
/// holds no hardware: every operation takes the caller-owned [`Gpus`]
/// explicitly. Nothing here frees a band's resources by dropping (unload is
/// explicit, see [`Qwen4Bundle::pp_free_gpu`]).
pub(crate) struct Qwen4PpMesh {
    /// Peer bands `0..n-1`; band `i` lives on `gpus.devices[i]`. Each has
    /// `execution: Some` between calls and `pp: None`.
    bands: Vec<Qwen4Bundle>,
    placement: BandPlacement,
    /// Split plans by output mode (`MODE_*`), built once at attach.
    plans: Vec<SplitPlan>,
    /// The common canonical prefill chunk of every band.
    chunk: usize,
    load_id: u64,
    request_id: u64,
    next_op: u64,
    /// Mesh identity of the `Gpus` this holder was validated against.
    mesh_epoch: Option<MeshEpoch>,
    health: Health,
    /// Aggregate session records over the per-band caches: owner-order piece
    /// keys and pins only, no device buffers. Cleared before any band's cache
    /// is cleared or freed.
    sessions: MeshSessionCache,
    /// Every band (root included) holds its own session cache, stamped with
    /// this mesh's load epoch at attach. Without it the mesh runs session-free.
    sessions_enabled: bool,
    /// The request id whose prefill completed with armed sessions and has not
    /// been committed or aborted: the only request whose staged candidates
    /// may publish.
    committable: Option<u64>,
    /// The one live aggregate MTP window (per-band local tickets and the head's).
    windows: WindowStore<Qwen4StateSnapshot, MtpGpuStateSnapshot>,
    /// Portable pinned decode edge staging, allocated on device zero: one
    /// `max_lanes`-row f32 region (the lane store's `rows_cap` without spec).
    /// Edge copies through it may be enqueued asynchronously; the host
    /// touches it only after the sender's [`Self::lane_events`] fence has
    /// completed, and a region is reused only after its consumer's copy has
    /// finished.
    lane_bounce: Option<LaneBounce>,
    /// Persistent decode-overlap fences, `[sender band][microbatch]`: each
    /// non-output band owns `min(max_lanes, 2 * stages)` events created on
    /// its own device (the output band sends nothing and has no row).
    /// Staged with the bounce, rolled back with it, and destroyed on their
    /// owning devices in [`Qwen4Bundle::pp_free_gpu`] after the edge drain.
    lane_events: Vec<Vec<hip_bridge::Event>>,
}

struct LaneBounce(hip_bridge::PinnedHostBuffer);

// SAFETY: portable pinned storage has no thread affinity. Access is exclusive
// through &mut mesh. Producer events gate other devices' reads; stream order
// protects a band's upload from its later download to the same region.
// Successful calls finish all copies. Failed calls drain all bands and
// quarantine the mesh, retaining storage if a drain fails; teardown leaks it
// rather than freeing memory that may still be in use.
unsafe impl Send for LaneBounce {}

impl LaneBounce {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_bytes_mut()
    }
}

/// What [`destroy_lane_events`] could not release.
struct LaneEventTeardown {
    /// First device bind failure; that device's events were leaked.
    bind: Option<String>,
    /// First `hipEventDestroy` failure on a bound device.
    destroy: Option<String>,
}

/// Destroy every sender band's events on its owning device. A device that
/// cannot be bound leaks its events (an event must not be destroyed under
/// another device's context); the remaining devices are still released.
fn destroy_lane_events(gpus: &mut Gpus, events: Vec<Vec<hip_bridge::Event>>) -> LaneEventTeardown {
    let mut teardown = LaneEventTeardown {
        bind: None,
        destroy: None,
    };
    for (index, row) in events.into_iter().enumerate() {
        let gpu = &gpus.devices[index];
        if let Err(error) = gpu.bind_thread() {
            teardown.bind.get_or_insert(error.to_string());
            std::mem::forget(row);
            continue;
        }
        for event in row {
            if let Err(error) = gpu.hip.event_destroy(event) {
                teardown.destroy.get_or_insert(error.to_string());
            }
        }
    }
    teardown
}

impl Qwen4PpMesh {
    /// Healthy, and `gpus` is the mesh this holder was validated against.
    fn require_ready(&self, gpus: &Gpus) -> Result<(), BundleError> {
        if let Health::Quarantined(reason) = &self.health {
            return Err(pp_error(format!("mesh is quarantined: {reason}")));
        }
        if !self.identity_matches(gpus) {
            return Err(pp_error("the supplied devices are not this mesh's"));
        }
        Ok(())
    }

    /// `gpus` is the device set recorded at attach: same mesh epoch and
    /// device count. Health is not consulted (a failed-attach holder must
    /// still be releasable).
    fn identity_matches(&self, gpus: &Gpus) -> bool {
        self.mesh_epoch == Some(gpus.mesh.epoch()) && gpus.devices.len() == self.stages()
    }

    fn quarantine(&mut self, reason: String) {
        if self.health == Health::Ready {
            self.health = Health::Quarantined(reason);
        }
    }

    /// Whether every band has a stamped session cache.
    pub(super) fn sessions_enabled(&self) -> bool {
        self.sessions_enabled
    }

    pub(super) fn stages(&self) -> usize {
        self.bands.len() + 1
    }
}

/// One tile of a request: `rows` tokens from `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    offset: usize,
    rows: usize,
}

/// Canonical in-order tiling: full `chunk` tiles, then the remainder. Never
/// padded or rounded.
fn plan_tiles(len: usize, chunk: usize) -> Vec<Tile> {
    let mut tiles = Vec::with_capacity(len.div_ceil(chunk.max(1)));
    let mut offset = 0;
    while offset < len {
        let rows = (len - offset).min(chunk);
        tiles.push(Tile { offset, rows });
        offset += rows;
    }
    tiles
}

/// Output mode of a tile: `All` rows every tile with a head, `Final` only the
/// last; no logits tensor means no head at all.
fn tile_mode(has_logits: bool, has_top1: bool, rows: Qwen4OutputRows, last_tile: bool) -> usize {
    if !has_logits || (rows == Qwen4OutputRows::Final && !last_tile) {
        MODE_TRUNK
    } else if has_top1 {
        MODE_LOGITS_TOP1
    } else {
        MODE_LOGITS
    }
}

/// Unload ordering: quiesce PLE first, then drain every device. Returns the
/// quiesce error (if any) and the drain verdict. Nothing is freed here.
fn quiesce_then_drain(
    quiesce: impl FnOnce() -> Result<(), BundleError>,
    drain: impl FnOnce() -> Result<(), String>,
) -> (Option<BundleError>, Result<(), String>) {
    let quiesced = quiesce().err();
    (quiesced, drain())
}

fn combined_manifest(
    root: &Qwen4Bundle,
    bands: &[Qwen4Bundle],
) -> (Vec<WeightEntry>, Vec<StateEntry>) {
    let mut weights = Vec::new();
    let mut state = Vec::new();
    for band in bands.iter().chain(std::iter::once(root)) {
        let manifest = band.manifest();
        weights.extend(manifest.weights.iter().cloned());
        state.extend(manifest.state.iter().cloned());
    }
    (weights, state)
}

/// One plan per output mode over the real manifest declarations combined from
/// the rank bands.
fn build_plans(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    state: &[StateEntry],
    mesh: &DeviceMesh,
    placement: &BandPlacement,
) -> Result<Vec<SplitPlan>, String> {
    let manifest = plan_manifest(weights, state, mesh, config.num_hidden_layers, placement)?;
    let mut plans = Vec::with_capacity(MODE_COUNT);
    for (head, top1) in [(false, false), (true, false), (true, true)] {
        let graph = qwen4_step_graph_with_output(config, MeshRoute::Prefill, head, top1)?;
        plans.push(split_program(&graph, &manifest, mesh).map_err(|error| error.to_string())?);
    }
    Ok(plans)
}

/// Move every band's forward out of its bundle (peers in band order, root
/// last). On any absence the forwards already taken are put back.
fn take_forwards(
    bands: &mut [Qwen4Bundle],
    root: &mut Qwen4Bundle,
) -> Result<Vec<Qwen4GpuForward>, BundleError> {
    let peers = bands.len();
    let mut forwards = Vec::with_capacity(peers + 1);
    for index in 0..=peers {
        let bundle = if index < peers {
            &mut bands[index]
        } else {
            &mut *root
        };
        match bundle.execution.take() {
            Some(forward) => forwards.push(forward),
            None => {
                restore_forwards(bands, root, forwards);
                return Err(pp_error("a band's forward resources are not attached"));
            }
        }
    }
    Ok(forwards)
}

/// Inverse of [`take_forwards`]; accepts a contiguous prefix.
fn restore_forwards(
    bands: &mut [Qwen4Bundle],
    root: &mut Qwen4Bundle,
    forwards: Vec<Qwen4GpuForward>,
) {
    let mut forwards = forwards.into_iter();
    for band in bands.iter_mut() {
        if let Some(forward) = forwards.next() {
            band.execution = Some(forward);
        }
    }
    if let Some(forward) = forwards.next() {
        root.execution = Some(forward);
    }
}

/// A failure while tiles were (or may have been) in flight.
struct TileError {
    message: String,
    /// Quarantine the mesh: work was submitted or a drain failed.
    fatal: bool,
}

impl TileError {
    fn refused(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
        }
    }
    fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
        }
    }
}

/// Aggregate ticket of one native-MTP window over EVERY band (frozen §3.3): one
/// handle for the per-band target tickets and the output head's, which live in
/// the mesh holder ([`WindowStore`]). It is a different type from the local
/// `Qwen4StateSnapshot` / `MtpGpuStateSnapshot`, so no local `validate_ticket`
/// can be handed one, and a handle whose window was finalized, restored,
/// released, superseded or reset stops validating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4PpWindow {
    key: MeshCheckpointKey,
    start: usize,
    rows: usize,
    bands: usize,
    with_head: bool,
    armed: bool,
}

impl Qwen4PpWindow {
    /// A ticket with a dummy key for CPU tests of ticket typing (it names no
    /// live window of any mesh).
    #[cfg(test)]
    pub(crate) fn for_test(start: usize, rows: usize, with_head: bool, armed: bool) -> Self {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).expect("a two-band mesh");
        Self {
            key: MeshCheckpointKey {
                mesh_epoch: mesh.epoch(),
                load_gen: 0,
                reset_gen: 0,
                request_gen: 0,
                window_gen: 0,
            },
            start,
            rows,
            bands: 2,
            with_head,
            armed,
        }
    }

    /// The committed position the window started at.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Rows the window's row capture was armed for.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Whether the window also holds the output MTP head's ticket.
    pub fn has_head(&self) -> bool {
        self.with_head
    }

    /// Whether every band opened armed with a ring covering `rows()`: the
    /// ring-aware repair is available (otherwise restore + replay).
    pub fn row_repair_ready(&self) -> bool {
        self.armed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowPhase {
    /// Snapshotted; verify/repair/restore/finalize/release apply.
    Open,
    /// Committed with every local ticket kept active: only a retained restore
    /// or a release apply.
    Retained,
}

struct WindowRecord<L, H> {
    ticket: Qwen4PpWindow,
    checkpoint: MeshCheckpoint,
    target: Vec<L>,
    head: Option<H>,
    phase: WindowPhase,
}

/// The one live aggregate window of a mesh and the generation counter that
/// makes every handle unique. Generic over the local ticket types so the
/// state machine is CPU-testable with fakes.
struct WindowStore<L, H> {
    next: u64,
    live: Option<WindowRecord<L, H>>,
}

impl<L: Copy, H: Copy> WindowStore<L, H> {
    fn new() -> Self {
        Self {
            next: 1,
            live: None,
        }
    }

    /// Reserve the generation of a new window. Refused while another window is
    /// open (the local arenas allow one active snapshot); a retained window is
    /// superseded: its handle stops validating, exactly as the next local
    /// snapshot releases the retained arena entries.
    fn begin(&mut self) -> Result<u64, String> {
        if self
            .live
            .as_ref()
            .is_some_and(|record| record.phase == WindowPhase::Open)
        {
            return Err("another window is still open".to_string());
        }
        let generation = self.next;
        self.next = generation
            .checked_add(1)
            .ok_or_else(|| "window generations are exhausted".to_string())?;
        self.live = None;
        Ok(generation)
    }

    fn install(
        &mut self,
        ticket: Qwen4PpWindow,
        checkpoint: MeshCheckpoint,
        target: Vec<L>,
        head: Option<H>,
    ) -> Result<(), String> {
        if self.live.is_some() {
            return Err("a window is already live".to_string());
        }
        checkpoint
            .check_shape(&ticket.key)
            .map_err(|error| error.to_string())?;
        if target.len() != ticket.bands
            || checkpoint.per_owner.len() != ticket.bands
            || head.is_some() != ticket.with_head
            || checkpoint.head.is_some() != ticket.with_head
        {
            return Err("window tickets do not match its bands and head".to_string());
        }
        self.live = Some(WindowRecord {
            ticket,
            checkpoint,
            target,
            head,
            phase: WindowPhase::Open,
        });
        Ok(())
    }

    /// The live record of exactly `ticket`, in one of `phases`.
    fn require(
        &self,
        ticket: &Qwen4PpWindow,
        phases: &[WindowPhase],
    ) -> Result<&WindowRecord<L, H>, String> {
        match &self.live {
            Some(record) if record.ticket == *ticket && phases.contains(&record.phase) => {
                Ok(record)
            }
            Some(record) if record.ticket == *ticket => Err(format!(
                "the window is {:?}; this transition does not apply",
                record.phase
            )),
            _ => Err(
                "window ticket is stale: finalized, restored, released, superseded or reset"
                    .to_string(),
            ),
        }
    }

    /// An open window was committed with its tickets kept.
    fn retain(&mut self, ticket: &Qwen4PpWindow) -> Result<(), String> {
        self.require(ticket, &[WindowPhase::Open])?;
        if let Some(record) = self.live.as_mut() {
            record.phase = WindowPhase::Retained;
        }
        Ok(())
    }

    /// Remove and return `ticket`'s record (`None` when it is not live).
    fn take(&mut self, ticket: &Qwen4PpWindow) -> Option<WindowRecord<L, H>> {
        if self
            .live
            .as_ref()
            .is_some_and(|record| record.ticket == *ticket)
        {
            self.live.take()
        } else {
            None
        }
    }

    fn remove(&mut self, ticket: &Qwen4PpWindow) -> bool {
        self.take(ticket).is_some()
    }

    fn clear(&mut self) {
        self.live = None;
    }

    /// Refuse real row-capture ring growth while a window is live (open or
    /// retained): growing reallocates the rings its tickets point into. `rings`
    /// are every band's allocated ring rows; an ensure of `rows` that every
    /// band already covers (or a block of fewer than two rows, which captures
    /// nothing) is a no-op and passes.
    fn check_row_capture_resize(
        &self,
        rings: impl IntoIterator<Item = usize>,
        rows: usize,
    ) -> Result<(), String> {
        if self.live.is_some() && rows >= 2 && rings.into_iter().any(|ring| ring < rows) {
            return Err("row capture cannot be resized while a window is live".to_string());
        }
        Ok(())
    }
}

/// Whether a window of `rows` rows arms the GDN row capture: a block of at
/// least two rows that every band's ring covers (Single's rule per bundle,
/// agreed across bands so every band can repair or none must).
fn arm_decision(rings: impl IntoIterator<Item = usize>, rows: usize) -> bool {
    rows >= 2 && rings.into_iter().all(|ring| ring >= rows)
}

/// Pure admission of a mesh epoch against the bands' `(armed, ring rows)`:
/// every band agrees, and only a verify of at least two rows the rings cover
/// may run armed (an armed snapshot skips the GDN copies, so a one-row armed
/// forward cannot be undone).
fn check_epoch_arming(bands: &[(bool, usize)], verify: bool, rows: usize) -> Result<(), String> {
    let armed = bands.first().is_some_and(|band| band.0);
    if bands.iter().any(|band| band.0 != armed)
        || (armed && (!verify || rows < 2 || bands.iter().any(|band| band.1 < rows)))
    {
        return Err(
            "the bands disagree on, or cannot honor, the armed row capture of this forward"
                .to_string(),
        );
    }
    Ok(())
}

/// Pure admission of a ring-aware repair against `window` and the bands'
/// `(ring rows, position)`.
fn check_repair_request(
    window: &Qwen4PpWindow,
    keep: usize,
    rows: usize,
    bands: &[(usize, usize)],
) -> Result<(), String> {
    let end = window
        .start
        .checked_add(rows)
        .ok_or_else(|| "repair position overflows".to_string())?;
    if !window.armed {
        return Err("the window was not opened with its row capture armed".to_string());
    }
    if rows < 2 || rows > window.rows || keep == 0 || keep >= rows {
        return Err(format!(
            "cannot keep {keep} of {rows} verified rows of a {}-row window",
            window.rows
        ));
    }
    if bands.len() != window.bands
        || bands
            .iter()
            .any(|&(ring, position)| ring < rows || position != end)
    {
        return Err(
            "a band's ring does not cover the verified rows or it is not at the verify's end"
                .to_string(),
        );
    }
    Ok(())
}

/// Read and clear every device's router fault word and bounded-wait timeout
/// record (`Gpus::take_device_wait_errors`). Call only after the mesh's work
/// completed: a fault on ANY band invalidates the whole pipelined request, so
/// the failure is for the caller to treat as fatal (quarantine).
fn take_mesh_device_faults(gpus: &mut Gpus) -> Result<(), String> {
    gpus.take_device_wait_errors()
        .map_err(|error| format!("mesh device fault: {error}"))
}

/// Synchronize every device of the mesh (each on its own bound thread), then
/// fail if any device recorded a router fault or wait timeout.
fn sync_all_devices(gpus: &mut Gpus) -> Result<(), String> {
    for (index, gpu) in gpus.devices.iter_mut().enumerate() {
        gpu.bind_thread()
            .map_err(|error| format!("device {index} bind before window drain: {error}"))?;
        gpu.hip
            .device_synchronize()
            .map_err(|error| format!("device {index} window drain failed: {error}"))?;
    }
    take_mesh_device_faults(gpus)
}

/// Run `body` over an adapter of every band (forwards taken, always put back).
fn with_bindings<R>(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    body: impl FnOnce(&mut Qwen4MeshBindings<'_>) -> R,
) -> Result<R, BundleError> {
    let mut forwards = take_forwards(&mut mesh.bands, root)?;
    let outcome = Qwen4MeshBindings::new(mesh_ranks(&mut mesh.bands, root, &mut forwards))
        .map(|mut bindings| body(&mut bindings));
    restore_forwards(&mut mesh.bands, root, forwards);
    outcome.map_err(pp_error)
}

/// Turn an aggregate lifecycle failure into a bundle error, quarantining the
/// holder when the owners could not be reconciled (the runtime has already
/// poisoned its own state).
fn lifecycle_failure(
    mesh: &mut Qwen4PpMesh,
    what: &str,
    error: &MeshLifecycleError,
    quarantine: bool,
) -> BundleError {
    let message = format!("window {what} failed: {error}");
    if quarantine {
        mesh.quarantine(message.clone());
    }
    pp_error(message)
}

/// Per-call work shared by every tile.
struct Request<'r> {
    tokens: &'r [u32],
    logits: Option<&'r GpuTensor>,
    top1: Option<&'r GpuTensor>,
    rows: Qwen4OutputRows,
    route: MeshRoute,
    /// The canonical session schedule: boundaries to capture at and the tiles
    /// cut at them. `None` = no capture; tiles are chunk-relative.
    session: Option<&'r SessionRun<'r>>,
    /// Final pre-collapse wide hidden rows of the whole request (F32, `hc *
    /// hidden` per row, on the output device), written by the output band.
    wide: Option<&'r GpuTensor>,
}

/// Compute every rank's layout for every tile and prepare all capacity before
/// any submission. Layouts are tile-major (`tile * stages + rank`).
fn prepare_tiles(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    forwards: &mut [Qwen4GpuForward],
    route: MeshRoute,
    tiles: &[Tile],
    start: usize,
) -> Result<Vec<Qwen4MeshLayout>, TileError> {
    let stages = mesh.stages();
    if let Some(last) = tiles.last() {
        reclaim_records_for_growth(root, mesh, gpus, start + last.offset + last.rows);
    }
    let mut layouts = Vec::with_capacity(tiles.len() * stages);
    for tile in tiles {
        let end = start + tile.offset + tile.rows;
        let mut agreed: Option<usize> = None;
        for rank in 0..stages {
            let bundle = if rank + 1 < stages {
                &mut mesh.bands[rank]
            } else {
                &mut *root
            };
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread()
                .map_err(|error| TileError::refused(error.to_string()))?;
            let forward = &mut forwards[rank];
            if forward.scratch.max_chunk != mesh.chunk {
                return Err(TileError::refused(
                    "a band's prefill chunk differs from the mesh's common chunk",
                ));
            }
            let layout = forward
                .mesh_layout(bundle, gpu, route, tile.rows)
                .map_err(|error| TileError::refused(error.to_string()))?;
            if layout.route_rows < tile.rows || agreed.is_some_and(|rows| rows != layout.route_rows)
            {
                return Err(TileError::refused("bands disagree on the routing rows"));
            }
            agreed = Some(layout.route_rows);
            forward
                .prepare_mesh_capacity(bundle, gpu, end, layout.route_rows)
                .map_err(|error| TileError::refused(error.to_string()))?;
            layouts.push(layout);
        }
    }
    Ok(layouts)
}

/// Submit every canonical tile in order, then acknowledge the request with one
/// all-device drain. Band positions/QSA marks are enqueue frontiers until that
/// drain succeeds; an error quarantines them, never acknowledges a partial
/// request or permits another request to consume the partial state.
/// Failed enqueue marks are not rolled back independently: the quarantined
/// mesh cannot reuse/publish them, and S4 recovery remains unload/reload.
///
/// Backpressure stays on the devices: `EdgeLease`'s two ready/done banks order
/// boundary copies, and each band's stream orders receive -> compute -> send
/// -> next receive, protecting its reusable device scratch.
///
/// Host inputs have separate, local lifetime fences, not mesh barriers:
/// * `forward_chunk_scoped` uploads token ids through `mesh_tokens.stage_buffer`
///   and `upload_async`; `PleHostStage::stage_buffer` drains the previous upload
///   fence before any overwrite (including a shorter tail).
/// * Its PLE upload copies the `RowFetch::wait` lease into `host_ple` through
///   the same fenced staging API before releasing the lease. The next tile may
///   reuse that host allocation only after its copy event completes.
/// * `ple_ahead` is a row-store ticket, not an alias of either upload buffer:
///   `RowFetch::begin_or_adopt` validates its epoch/ids, and the forward releases
///   its lease only after staging/upload. Lookahead never overwrites in-flight
///   host staging bytes. These fences must remain even without per-tile drains.
fn run_tiles(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    forwards: &mut [Qwen4GpuForward],
    request: &Request<'_>,
    tiles: &[Tile],
    layouts: &[Qwen4MeshLayout],
) -> Result<(), TileError> {
    let stages = mesh.stages();
    let vocab = root.config.vocab_size;
    let count = tiles.len();
    let mut tickets = Vec::with_capacity(count);
    let mut submitted = false;
    let submitted_all = (|| {
        for (index, tile) in tiles.iter().enumerate() {
            let last = index + 1 == count;
            let mode = tile_mode(
                request.logits.is_some(),
                request.top1.is_some(),
                request.rows,
                last,
            );
            // Output views; they outlive the adapter that borrows them.
            let (logits_view, top1_view) = match (mode, request.logits) {
                (MODE_TRUNK, _) | (_, None) => (None, None),
                (_, Some(logits)) => {
                    let (logits_offset, top1_offset, rows) = match request.rows {
                        Qwen4OutputRows::All => (tile.offset * vocab, tile.offset * 4, tile.rows),
                        Qwen4OutputRows::Final => (0, 0, 1),
                    };
                    (
                        Some(logits.sub_offset(logits_offset, rows * vocab)),
                        request.top1.map(|top1| {
                            top1.sub_offset(top1_offset, rows * std::mem::size_of::<i32>())
                        }),
                    )
                }
            };
            // This tile's rows of the caller's wide hidden capture.
            let wide_view = request.wide.map(|wide| {
                let width = root.config.hc_count * root.config.hidden_size;
                wide.sub_offset(tile.offset * width, tile.rows * width)
            });
            let tile_layouts = &layouts[index * stages..(index + 1) * stages];
            let op_id = mesh.next_op;
            mesh.next_op = op_id
                .checked_add(1)
                .ok_or_else(|| TileError::refused("mesh operation ids are exhausted"))?;
            let epoch = ForwardEpoch {
                mesh_epoch: gpus.mesh.epoch(),
                load_id: mesh.load_id,
                request_id: mesh.request_id,
                op_id,
                route: request.route,
                start_position: root.state.position,
                actual_rows: tile.rows,
                route_rows: tile_layouts[0].route_rows,
                policy_id: PP_POLICY_ID,
            };
            let mut edge_epochs = Vec::with_capacity(stages - 1);
            for src in 0..stages - 1 {
                let id = EdgeId {
                    src_device: src,
                    dst_device: src + 1,
                };
                match gpus.mesh_edge_epoch(id) {
                    Some(epoch) if epoch != 0 => edge_epochs.push(epoch),
                    _ => return Err(TileError::refused("a band boundary has no transport lease")),
                }
            }
            let (peer_forwards, root_forward) = forwards.split_at_mut(stages - 1);
            let mut ranks = Vec::with_capacity(stages);
            for (bundle, forward) in mesh.bands.iter_mut().zip(peer_forwards.iter_mut()) {
                ranks.push(Qwen4RankBand { bundle, forward });
            }
            ranks.push(Qwen4RankBand {
                bundle: &mut *root,
                forward: &mut root_forward[0],
            });
            let mut bindings = Qwen4MeshBindings::new(ranks).map_err(TileError::refused)?;
            bindings.set_tokens(&request.tokens[tile.offset..tile.offset + tile.rows]);
            bindings.set_edge_epochs(edge_epochs);
            if let Some(run) = request.session {
                bindings.set_session(run.prompt, run.route);
            }
            for (owner, layout) in tile_layouts.iter().enumerate() {
                bindings
                    .set_rank_layout(owner, *layout)
                    .map_err(TileError::refused)?;
            }
            bindings.set_head(logits_view.as_ref().map(|logits| Qwen4MeshHead {
                logits,
                top1: top1_view.as_ref(),
                rows: request.rows,
            }));
            bindings.set_wide_capture(wide_view.as_ref());
            // Treat even a refused submit conservatively once it may have entered
            // a callback: a later tile's failure also invalidates all earlier work.
            submitted = true;
            tickets.push(
                gpus.submit_mesh_epoch(&mesh.plans[mode], &epoch, &mut bindings)
                    .map_err(|error| TileError::fatal(format!("submission failed: {error}")))?,
            );
            if let Some(run) = request.session {
                let end = run.start + tile.offset + tile.rows;
                if run.boundaries.binary_search(&end).is_ok() {
                    // Enqueued on every band's own stream behind its work for
                    // this tile; nothing drains. Pieces stay pending until the
                    // generation layer's commit publishes the aggregate record.
                    gpus.capture_mesh_session(&mut bindings, &epoch)
                        .map_err(|error| {
                            TileError::fatal(format!("session capture failed: {error}"))
                        })?;
                    let pieces = bindings.take_session_captures();
                    if let Some(pieces) = aggregate_candidate(run.lineage, end, pieces) {
                        mesh.sessions.stage(MeshSessionRecord {
                            route: run.route,
                            pieces,
                        });
                    }
                }
            }
        }
        Ok::<(), TileError>(())
    })();
    if let Err(error) = submitted_all {
        if !submitted {
            return Err(error);
        }
        // Also covers local validation/id exhaustion after earlier tiles were
        // enqueued. Stop all edges/devices before releasing borrowed resources;
        // the caller quarantines the request even when this drain succeeds. Faults
        // the stopped work recorded are folded into the message, not discarded.
        let mut message = error.message;
        if let Err(drain) = gpus.drain_mesh_edges() {
            message = format!("{message}; {drain}");
        }
        if let Err(fault) = take_mesh_device_faults(gpus) {
            message = format!("{message}; {fault}");
        }
        return Err(TileError::fatal(message));
    }
    // The request is acknowledged only when the drain succeeded AND no device
    // recorded a router fault or a bounded-wait timeout while it ran: a faulted
    // row's logits/top-1 are fail-closed output of an invalid request, so the
    // caller must quarantine the mesh instead of reading them back. The fault
    // words are read even when the drain failed so the report is not left stale.
    let drained = gpus
        .drain_mesh_epochs(tickets)
        .map_err(|error| format!("drain failed: {error}"));
    let faults = take_mesh_device_faults(gpus);
    match (drained, faults) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(message), Ok(())) | (Ok(()), Err(message)) => Err(TileError::fatal(message)),
        (Err(drain), Err(fault)) => Err(TileError::fatal(format!("{drain}; {fault}"))),
    }
}

fn refuse_lane_spec(spec: Option<crate::lane::LaneSpecConfig>) -> Result<(), BundleError> {
    if spec.is_some() {
        return Err(BundleError::Forward(
            "qwen4 topology.pp: native-MTP lanes are not supported yet; stage AR lanes (serve.batch_spec = false)"
                .to_string(),
        ));
    }
    Ok(())
}

fn lane_edge_bytes(rows: usize, hc: usize, hidden: usize, bf16: bool) -> Option<usize> {
    rows.checked_mul(hc)?
        .checked_mul(hidden)?
        .checked_mul(if bf16 { 2 } else { 4 })
}

impl Qwen4Bundle {
    /// Stage exact AR stores on every band's device without changing the
    /// resident singleton state, sessions or admission plan.
    pub fn pp_stage_lanes(
        &mut self,
        gpus: &mut Gpus,
        max_lanes: usize,
        row_budget: usize,
        spec: Option<crate::lane::LaneSpecConfig>,
    ) -> Result<(), BundleError> {
        refuse_lane_spec(spec)?;
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = (|| {
            mesh.require_ready(gpus)?;
            if mesh.lane_bounce.is_some()
                || mesh
                    .bands
                    .iter()
                    .chain(std::iter::once(&*self))
                    .any(|band| band.lanes.is_some())
            {
                return Err(pp_error("lanes are already staged on a pipeline band"));
            }
            // Match stage_qwen4_lanes: need <= current free, with no added
            // reserve from another architecture's KV policy.
            for (index, band) in mesh.bands.iter().chain(std::iter::once(&*self)).enumerate() {
                let gpu = &mut gpus.devices[index];
                gpu.bind_thread().map_err(|e| pp_error(e.to_string()))?;
                let context = crate::Qwen4ContextCommit::new(
                    band.state.qsa_backend(),
                    band.state.max_seq_len,
                    mesh.chunk,
                    gpu.vmm_recommended_granularity().unwrap_or(1),
                );
                let need = crate::lane::Qwen4LaneStore::device_bytes_scoped(
                    &band.config,
                    band.state_format,
                    &context,
                    max_lanes,
                    row_budget,
                    None,
                    band.weights.band_scope(),
                )
                .ok_or_else(|| pp_error("lane store device bytes overflow"))?;
                let (free, _) = gpu.device_mem_info().map_err(|e| pp_error(e.to_string()))?;
                if need > free as u64 {
                    return Err(pp_error(format!(
                        "band {index} lane store needs {} MiB, {} MiB free",
                        need >> 20,
                        free >> 20,
                    )));
                }
                if band.execution.is_none() {
                    return Err(pp_error("a band's forward resources are not attached"));
                }
            }
            let bytes = lane_edge_bytes(
                max_lanes,
                self.config.hc_count,
                self.config.hidden_size,
                false,
            )
            .ok_or_else(|| pp_error("lane edge byte count overflows"))?;
            gpus.devices[0]
                .bind_thread()
                .map_err(|e| pp_error(e.to_string()))?;
            let bounce = LaneBounce(
                gpus.devices[0]
                    .hip
                    .host_buffer(bytes)
                    .map_err(|e| pp_error(e.to_string()))?,
            );
            let stages = mesh.stages();
            let mut staged = 0;
            let mut events: Vec<Vec<hip_bridge::Event>> = Vec::new();
            let outcome: Result<(), BundleError> = (|| {
                // One fence row per sender band (the output band sends
                // nothing), created on that band's device.
                let per_band = max_lanes.min(2 * stages);
                for index in 0..stages - 1 {
                    let gpu = &gpus.devices[index];
                    gpu.bind_thread().map_err(|e| pp_error(e.to_string()))?;
                    events.push(Vec::with_capacity(per_band));
                    for _ in 0..per_band {
                        let event = gpu
                            .hip
                            .event_create_with_flags(
                                hip_bridge::HIP_EVENT_DISABLE_TIMING
                                    | hip_bridge::HIP_EVENT_RELEASE_TO_SYSTEM,
                            )
                            .map_err(|e| pp_error(e.to_string()))?;
                        events[index].push(event);
                    }
                }
                for index in 0..stages {
                    let band = band_mut(&mut mesh.bands, self, index);
                    let gpu = &mut gpus.devices[index];
                    gpu.bind_thread().map_err(|e| pp_error(e.to_string()))?;
                    if gpu.route_kernels_finished() {
                        gpu.abort_route_kernel_load()
                            .map_err(|e| pp_error(e.to_string()))?;
                    }
                    band.stage_lanes(gpu, max_lanes, row_budget, None)?;
                    staged += 1;
                }
                Ok(())
            })();
            if let Err(error) = outcome {
                let mut cleanup = None;
                for index in 0..staged {
                    let band = band_mut(&mut mesh.bands, self, index);
                    let gpu = &mut gpus.devices[index];
                    if let Err(e) = gpu.bind_thread() {
                        cleanup.get_or_insert(e.to_string());
                        continue;
                    }
                    if let Some(store) = band.lanes.take() {
                        if let Err(e) = store.free_gpu(gpu) {
                            cleanup.get_or_insert(e);
                        }
                    }
                }
                let teardown = destroy_lane_events(gpus, events);
                cleanup = cleanup.or(teardown.bind).or(teardown.destroy);
                match gpus.devices[0].bind_thread() {
                    Ok(()) => drop(bounce),
                    Err(e) => {
                        cleanup.get_or_insert(e.to_string());
                        std::mem::forget(bounce);
                    }
                }
                if let Some(cleanup) = cleanup {
                    let reason = format!("lane staging failed: {error}; cleanup failed: {cleanup}");
                    mesh.quarantine(reason.clone());
                    return Err(pp_error(reason));
                }
                return Err(error);
            }
            mesh.lane_bounce = Some(bounce);
            mesh.lane_events = events;
            Ok(())
        })();
        self.pp = Some(mesh);
        result
    }

    /// The cold PP singleton's next tile, including its tail-lineage split,
    /// without arming or publishing any session.
    fn pp_lane_tile_len(&self, mesh: &Qwen4PpMesh, prompt: &[u32], prefilled: usize) -> usize {
        if prefilled >= prompt.len() {
            return 0;
        }
        if !mesh.sessions_enabled {
            return (prompt.len() - prefilled).min(mesh.chunk);
        }
        let mut boundaries = SessionState::snapshot_boundaries(self, prompt, 0, prompt.len());
        let lineage = SessionState::lineage_boundary(self, prompt, 0, prompt.len())
            .filter(|&at| at > 0 && at < prompt.len());
        if let Some(at) = lineage {
            boundaries.push(at);
            boundaries.sort_unstable();
            boundaries.dedup();
        }
        let stop = lineage.filter(|&at| at > prefilled).unwrap_or(prompt.len());
        let mut upcoming = boundaries.into_iter().filter(|&at| at > prefilled);
        qwen4_prefill_segment_rows(
            prefilled,
            stop,
            mesh.chunk,
            SessionBoundaries {
                next: upcoming.next(),
                after: upcoming.next(),
            },
            false,
        )
    }

    /// Run one PP singleton prefill tile with every band's lane state swapped
    /// in. Mesh forwards retain ordinary admission (lane_scope stays false);
    /// their chunk path does not replay retained bodies. Sessions are removed
    /// for the call, so no singleton request bookkeeping is published.
    fn pp_lane_prefill_tile_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
        tokens: &[u32],
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        for band in mesh.bands.iter().chain(std::iter::once(&*self)) {
            let lane = band
                .lanes
                .as_ref()
                .and_then(|store| store.lane(epoch))
                .ok_or_else(|| pp_error(format!("prefill: unknown or stale epoch {epoch:?}")))?;
            if band.execution.is_none() || lane.spec.is_some() {
                return Err(pp_error(
                    "prefill needs an attached forward and an AR lane on every band",
                ));
            }
        }
        let mut sessions = Vec::with_capacity(mesh.stages());
        for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *self)) {
            sessions.push(band.session.take());
            let lane = band.lanes.as_mut().unwrap().lane_mut(epoch).unwrap();
            band.state.clamp_f16_valid();
            lane.state.clamp_f16_valid();
            std::mem::swap(&mut band.state, &mut lane.state);
            band.execution.as_mut().unwrap().set_ple_lookahead(&[]);
        }
        // Advances only next_op (transport identity), not singleton request
        // identity, windows, committable state or session records.
        let result = self.pp_forward_detached(
            mesh,
            gpus,
            &Request {
                tokens,
                logits,
                top1,
                rows: Qwen4OutputRows::Final,
                route: MeshRoute::Prefill,
                session: None,
                wide: None,
            },
        );
        for (band, session) in mesh
            .bands
            .iter_mut()
            .chain(std::iter::once(&mut *self))
            .zip(sessions)
        {
            let lane = band.lanes.as_mut().unwrap().lane_mut(epoch).unwrap();
            band.state.clamp_f16_valid();
            lane.state.clamp_f16_valid();
            std::mem::swap(&mut band.state, &mut lane.state);
            band.session = session;
        }
        result
    }

    /// Whether this bundle is the output-band root of an attached PP mesh
    /// (including one whose attach failed and is awaiting `free_gpu`).
    pub fn pp_mesh_active(&self) -> bool {
        self.pp.is_some()
    }

    /// Lane topology of this bundle: `"pp"` (band-scoped stores), `"ep"`
    /// (replicated rank stores) or `"single"`.
    pub fn lane_topology(&self) -> &'static str {
        if self.pp.is_some() {
            "pp"
        } else if self.ep.is_some() {
            "ep"
        } else {
            "single"
        }
    }

    /// Every band's lane-store receipt in pipeline order (input band first,
    /// the output-band root LAST). A band without a store is an error.
    pub fn pp_lane_receipts(
        &self,
        gpus: &Gpus,
    ) -> Result<Vec<crate::lane::LaneStoreReceipt>, BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.bands
            .iter()
            .chain(std::iter::once(self))
            .enumerate()
            .map(|(index, band)| {
                let gpu = gpus
                    .devices
                    .get(index)
                    .ok_or_else(|| pp_error(format!("band {index} has no device")))?;
                band.lanes
                    .as_ref()
                    .map(|store| store.receipt(gpu))
                    .ok_or_else(|| pp_error(format!("band {index} lanes are not staged")))
            })
            .collect()
    }

    /// Read-only local states in pipeline band order, including the output band.
    pub fn pp_band_states(&self) -> Option<impl Iterator<Item = &Qwen4State>> {
        self.pp.as_ref().map(|mesh| {
            mesh.bands
                .iter()
                .map(|band| &band.state)
                .chain(std::iter::once(&self.state))
        })
    }

    /// Adopt the mesh's hardware and peer bands. `self` is the output band.
    ///
    /// Ownership is installed before any validation: on `Err` the holder
    /// stays attached (quarantined), so the caller releases everything with
    /// [`Self::free_gpu`], which takes the PP teardown path.
    pub fn attach_pp_mesh(
        &mut self,
        gpus: &mut Gpus,
        bands: Vec<Qwen4Bundle>,
        placement: BandPlacement,
    ) -> Result<(), BundleError> {
        if self.pp.is_some() {
            // Ownership cannot be adopted twice; releasing either set would
            // free resources with the wrong devices.
            std::mem::forget(bands);
            return Err(pp_error("a pipeline mesh is already attached"));
        }
        let mut holder = Box::new(Qwen4PpMesh {
            bands,
            placement,
            plans: Vec::new(),
            chunk: 0,
            // Assigned during validation so id exhaustion fails closed.
            load_id: 0,
            request_id: 1,
            next_op: 1,
            // Recorded before validation so a failed attach is still
            // releasable against the device set it was given.
            mesh_epoch: Some(gpus.mesh.epoch()),
            health: Health::Quarantined("attach did not complete".to_string()),
            sessions: MeshSessionCache::new(),
            sessions_enabled: false,
            committable: None,
            windows: WindowStore::new(),
            lane_bounce: None,
            lane_events: Vec::new(),
        });
        let verdict = self.validate_pp_attach(&mut holder, gpus);
        let result = match verdict {
            Ok(()) => {
                holder.health = Health::Ready;
                Ok(())
            }
            Err(message) => {
                holder.health = Health::Quarantined(message.clone());
                Err(pp_error(message))
            }
        };
        self.pp = Some(holder);
        result
    }

    fn validate_pp_attach(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
    ) -> Result<(), String> {
        let stages = mesh.placement.stages();
        let config = self.config.clone();
        if !(2..=PP_MAX_BANDS).contains(&stages)
            || mesh.bands.len() + 1 != stages
            || gpus.devices.len() != stages
            || gpus.mesh.n_devices() != stages
            || gpus.mesh.size_of(DimKind::Pp) != stages
            || gpus.mesh.size_of(DimKind::Tp) != 1
            || gpus.mesh.size_of(DimKind::Ep) != 1
        {
            return Err("the band count, device count and mesh shape disagree".into());
        }
        mesh.placement
            .validate_for(&gpus.mesh, config.num_hidden_layers)?;
        // Tiling and endpoint pins: every band owns exactly its placed band.
        let position = self.state.position;
        let max_seq = self.state.max_seq_len;
        let mut chunk = None;
        for (index, band) in mesh.bands.iter().chain(std::iter::once(&*self)).enumerate() {
            let scope = Qwen4BandScope::from_placement(&config, &mesh.placement, index)?;
            if band.weights.band_scope() != &scope {
                return Err("a band does not own the layers placed on it".into());
            }
            if band.config != config {
                return Err("the bands have different model configurations".into());
            }
            if band.ep.is_some()
                || band.lanes.is_some()
                || (band.mtp.is_some() && index + 1 < stages)
            {
                return Err(
                    "expert parallelism, lanes and an MTP head off the output band are not admitted"
                        .into(),
                );
            }
            if index + 1 < stages && band.pp.is_some() {
                return Err("a peer band carries its own pipeline mesh".into());
            }
            if band.state.position != position || band.state.max_seq_len != max_seq {
                return Err("the bands disagree on position or context capacity".into());
            }
            let forward = band
                .execution
                .as_ref()
                .ok_or("a band's forward resources are not attached")?;
            let band_chunk = forward.scratch.max_chunk;
            if band_chunk == 0 || chunk.is_some_and(|common| common != band_chunk) {
                return Err("the bands do not share one canonical prefill chunk".into());
            }
            chunk = Some(band_chunk);
        }
        mesh.load_id = next_load_id()?;
        mesh.mesh_epoch = Some(gpus.mesh.epoch());
        mesh.chunk = chunk.ok_or("no bands")?;
        // Band identity and session stamp: every band (root included) learns its
        // placement, devices and epochs now, so scopes and boundary chunks never
        // depend on a taken forward or the detached holder.
        let load_id = mesh.load_id;
        let mesh_epoch = gpus.mesh.epoch().as_u64();
        let devices: Vec<String> = gpus
            .devices
            .iter()
            .map(|gpu| format!("{}@{}", gpu.device_id, gpu.arch))
            .collect();
        let chunk = mesh.chunk;
        let present: Vec<bool> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| band.session.is_some())
            .collect();
        let turn_snapshots: Vec<bool> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| band.turn_snapshots)
            .collect();
        let sessions = attach_session_policy(stages, &present, &turn_snapshots)?;
        for (index, band) in mesh
            .bands
            .iter_mut()
            .chain(std::iter::once(&mut *self))
            .enumerate()
        {
            band.pp_band = Some(Qwen4PpBandMeta::new(
                index,
                &mesh.placement,
                devices.clone(),
                mesh_epoch,
                load_id,
                chunk,
            ));
            if let Some(session) = band.session.as_mut().filter(|_| sessions) {
                session
                    .bind_load_epoch(load_id)
                    .map_err(|error| format!("band {index} session cache: {error}"))?;
            }
        }
        mesh.sessions_enabled = sessions;
        // Prepared transport: the output owner and every boundary lease.
        if gpus.mesh_output_owner() != Some(stages - 1) {
            return Err("the mesh output owner is not the last band".into());
        }
        for src in 0..stages - 1 {
            let id = EdgeId {
                src_device: src,
                dst_device: src + 1,
            };
            if !matches!(gpus.mesh_edge_epoch(id), Some(epoch) if epoch != 0) {
                return Err("a band boundary has no transport lease".into());
            }
        }
        let (weights, state) = combined_manifest(self, &mesh.bands);
        mesh.plans = build_plans(&config, &weights, &state, &gpus.mesh, &mesh.placement)?;
        // Dry-run the adapter's ownership validation over the live bands.
        let forwards = take_forwards(&mut mesh.bands, self).map_err(|error| error.to_string())?;
        let mut forwards = forwards;
        let verdict = (|| {
            let (peer_forwards, root_forward) = forwards.split_at_mut(stages - 1);
            let mut ranks = Vec::with_capacity(stages);
            for (bundle, forward) in mesh.bands.iter_mut().zip(peer_forwards.iter_mut()) {
                ranks.push(Qwen4RankBand { bundle, forward });
            }
            ranks.push(Qwen4RankBand {
                bundle: &mut *self,
                forward: &mut root_forward[0],
            });
            Qwen4MeshBindings::new(ranks).map(|_| ())
        })();
        restore_forwards(&mut mesh.bands, self, forwards);
        verdict
    }

    /// Run `tokens` through the pipeline: tiles of the common canonical chunk
    /// in order, the head only where `rows`/`logits` ask for it. `logits` and
    /// `top1` are the daemon's tensors on the output band's physical device;
    /// they are complete when this returns. `wide_capture` (output device, F32,
    /// `hc * hidden` per row of `tokens`) receives the final pre-collapse wide
    /// hidden rows, independent of the head (a silent prompt-fill chunk passes
    /// `logits: None`); the caller keeps it alive until the call returns and
    /// consumes it before the next forward overwrites it.
    pub fn pp_forward(
        &mut self,
        gpus: &mut Gpus,
        tokens: &[u32],
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        rows: Qwen4OutputRows,
        route: MeshRoute,
        wide_capture: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_forward_detached(
            &mut mesh,
            gpus,
            &Request {
                tokens,
                logits,
                top1,
                rows,
                route,
                session: None,
                wide: wide_capture,
            },
        );
        self.pp = Some(mesh);
        result
    }

    /// Output-band device of the mesh, thread bound (the head restore runs on it;
    /// callers of the driver get the same device from `gpus.mesh_output_owner()`).
    fn pp_output_gpu<'g>(&self, gpus: &'g mut Gpus) -> Result<&'g mut Gpu, BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.require_ready(gpus)?;
        let owner = gpus
            .mesh_output_owner()
            .ok_or_else(|| pp_error("the mesh has no output owner"))?;
        let gpu = gpus
            .devices
            .get_mut(owner)
            .ok_or_else(|| pp_error("the output device is unavailable"))?;
        gpu.bind_thread().map_err(BundleError::Hip)?;
        Ok(gpu)
    }

    /// The canonical chunk every band shares (the most rows one mesh epoch
    /// takes), `None` outside PP.
    pub(crate) fn pp_chunk_rows(&self) -> Option<usize> {
        self.pp.as_ref().map(|mesh| mesh.chunk)
    }

    /// Pure preflight: a mesh is attached and healthy, and `gpus` is the device
    /// set it was attached with (same mesh epoch and device count). Touches no
    /// state; every `pp_*` entry point applies the same check itself.
    pub(crate) fn pp_require_ready(&self, gpus: &Gpus) -> Result<(), BundleError> {
        self.pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?
            .require_ready(gpus)
    }

    /// Set (or clear) the armed GDN row capture of EVERY band at once, host
    /// only. Windows arm it through [`Self::pp_window_open`]; the caller clears
    /// it after the verify exactly as Single does.
    pub(crate) fn pp_window_arm(&mut self, armed: bool) {
        self.state.row_capture_armed = armed;
        if let Some(mesh) = self.pp.as_mut() {
            for band in &mut mesh.bands {
                band.state.row_capture_armed = armed;
            }
        }
    }

    /// Allocate every band's few-row verify rollback ring for blocks of up to
    /// `rows` rows on its own device. `Ok(false)`: this GPU's GDN route captures
    /// no rows (on every band; a mix refuses). Refused only for real ring growth
    /// while a window is live; an ensure every band's ring already covers is a
    /// no-op and passes.
    pub(crate) fn pp_ensure_row_capture(
        &mut self,
        gpus: &mut Gpus,
        rows: usize,
    ) -> Result<bool, BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_ensure_row_capture_detached(&mut mesh, gpus, rows);
        self.pp = Some(mesh);
        result
    }

    fn pp_ensure_row_capture_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        rows: usize,
    ) -> Result<bool, BundleError> {
        mesh.require_ready(gpus)?;
        if rows < 2 {
            return Ok(false);
        }
        let capable: Vec<bool> = gpus
            .devices
            .iter()
            .map(|gpu| crate::mtp_spec::native_mtp_row_capture(gpu, &self.config))
            .collect();
        if capable.iter().all(|&capable| !capable) {
            return Ok(false);
        }
        if capable.iter().any(|&capable| !capable) {
            return Err(pp_error("the bands disagree on GDN row-capture support"));
        }
        // Growth reallocates the rings a live (open or retained) window's
        // tickets point into; refuse it before any bind or allocation. Rings
        // that already cover `rows` make every per-band ensure below a no-op.
        mesh.windows
            .check_row_capture_resize(
                mesh.bands
                    .iter()
                    .chain(std::iter::once(&*self))
                    .map(|band| band.state.row_capture_rows()),
                rows,
            )
            .map_err(pp_error)?;
        let history = self.config.linear_conv_kernel_dim - 1;
        for index in 0..mesh.stages() {
            let gpu = &mut gpus.devices[index];
            gpu.bind_thread().map_err(BundleError::Hip)?;
            band_mut(&mut mesh.bands, self, index)
                .state
                .ensure_row_capture(gpu, rows, history)
                .map_err(BundleError::State)?;
        }
        Ok(true)
    }

    /// The Single speculative verify (`spec_forward_rows`) over `block` on every
    /// band: one `MeshRoute::Verify` epoch of exactly `block.len()` rows (routing
    /// its actual rows, never rounded), at most one canonical chunk. The output
    /// band writes `logits` (all rows, `block.len() * vocab` F32), optional
    /// `top1` (Raw, 4 bytes per row) and the final pre-collapse wide hidden rows
    /// into `wide` (F32, at least `block.len() * hc * hidden`), all on the output
    /// device. Every band's armed row capture (see [`Self::pp_window_open`]) is
    /// written. The mesh is drained before return. The caller owns clearing the
    /// armed flag ([`Self::pp_window_arm`]).
    ///
    /// Refused BEFORE any mutation: an unhealthy mesh, an armed capture with a
    /// one-row block (an armed snapshot skips the GDN copies, so a one-row
    /// verify could be neither ring-repaired nor restored), or bands that
    /// disagree on arming. A failure after submission quarantines the mesh.
    pub(crate) fn pp_forward_verify(
        &mut self,
        gpus: &mut Gpus,
        block: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide: &GpuTensor,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_forward_detached(
            &mut mesh,
            gpus,
            &Request {
                tokens: block,
                logits: Some(logits),
                top1,
                rows: Qwen4OutputRows::All,
                route: MeshRoute::Verify,
                session: None,
                wide: Some(wide),
            },
        );
        self.pp = Some(mesh);
        result
    }

    /// Open the aggregate window of the next `rows`-row verify at the current
    /// position (frozen §3.3 step 1): EVERY band arms its row capture (iff
    /// `rows >= 2` and every band's ring holds `rows`) BEFORE any band
    /// snapshots, then `Gpus::checkpoint_mesh` snapshots every band's target
    /// state and, with `with_head`, the output MTP head, each on its own
    /// device. All-or-nothing: a failure restores the bands already captured.
    /// The returned ticket is the aggregate handle; the local tickets stay in
    /// the mesh holder. Refused while another window is open; a retained
    /// window is superseded (its handle stops validating).
    pub(crate) fn pp_window_open(
        &mut self,
        gpus: &mut Gpus,
        rows: usize,
        with_head: bool,
    ) -> Result<Qwen4PpWindow, BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_window_open_detached(&mut mesh, gpus, rows, with_head, true);
        self.pp = Some(mesh);
        result
    }

    /// [`Self::pp_window_open`] that never arms the row capture: `rows` is the
    /// most rows the window may span. The interleaved one-row windows kept for
    /// stop sequences advance the target by up to `rows` unarmed one-row verifies
    /// before the window is finalized (retained) or restored; no transition
    /// requires the position to be `start + rows`, only that every band and the
    /// head agree. Such a window has no ring-aware repair (`row_repair_ready()`
    /// is false).
    pub(crate) fn pp_window_open_unarmed(
        &mut self,
        gpus: &mut Gpus,
        rows: usize,
        with_head: bool,
    ) -> Result<Qwen4PpWindow, BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_window_open_detached(&mut mesh, gpus, rows, with_head, false);
        self.pp = Some(mesh);
        result
    }

    fn pp_window_open_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        rows: usize,
        with_head: bool,
        allow_arm: bool,
    ) -> Result<Qwen4PpWindow, BundleError> {
        mesh.require_ready(gpus)?;
        if rows == 0 || rows > mesh.chunk {
            return Err(pp_error(
                "a window covers between one row and one canonical chunk",
            ));
        }
        if with_head && self.mtp.is_none() {
            return Err(pp_error("the output band has no MTP head to snapshot"));
        }
        let stages = mesh.stages();
        let start = self.state.position;
        if mesh.bands.iter().any(|band| band.state.position != start) {
            return Err(pp_error("the bands disagree on position"));
        }
        let armed = allow_arm
            && arm_decision(
                mesh.bands
                    .iter()
                    .chain(std::iter::once(&*self))
                    .map(|band| band.state.row_capture_rows()),
                rows,
            );
        let generation = mesh.windows.begin().map_err(pp_error)?;
        let key = MeshCheckpointKey {
            mesh_epoch: gpus.mesh.epoch(),
            load_gen: mesh.load_id,
            reset_gen: 0,
            request_gen: mesh.request_id,
            window_gen: generation,
        };
        // Armed on every band before the first snapshot.
        self.pp_arm_bands(mesh, armed);
        let outcome = with_bindings(self, mesh, |bindings| {
            bindings.set_window(WindowCtx::checkpoint(stages));
            let checkpoint = gpus.checkpoint_mesh(&mut *bindings, &key, with_head);
            (checkpoint, bindings.take_window())
        });
        let (checkpoint, context) = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                self.pp_arm_bands(mesh, false);
                return Err(error);
            }
        };
        let checkpoint = match checkpoint {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                self.pp_arm_bands(mesh, false);
                let rolled_back_cleanly = error.restore_failures.is_empty();
                return Err(lifecycle_failure(
                    mesh,
                    "open",
                    &error,
                    !rolled_back_cleanly,
                ));
            }
        };
        let ticket = Qwen4PpWindow {
            key,
            start,
            rows,
            bands: stages,
            with_head,
            armed,
        };
        let installed = context
            .and_then(WindowCtx::into_target)
            .ok_or_else(|| "a band captured no window ticket".to_string())
            .and_then(|(target, head)| mesh.windows.install(ticket, checkpoint, target, head));
        match installed {
            Ok(()) => Ok(ticket),
            Err(message) => {
                // Tickets exist that no handle names: the state cannot be
                // reconciled.
                self.pp_arm_bands(mesh, false);
                mesh.quarantine(message.clone());
                Err(pp_error(message))
            }
        }
    }

    fn pp_arm_bands(&mut self, mesh: &mut Qwen4PpMesh, armed: bool) {
        self.state.row_capture_armed = armed;
        for band in &mut mesh.bands {
            band.state.row_capture_armed = armed;
        }
    }

    /// Roll EVERY band (and the output head, when the window has one) back to
    /// the window's start through `Gpus::restore_mesh`. `retain == false`
    /// consumes the window (the error path; the handle is dead afterwards);
    /// `retain == true` keeps every local ticket active for the retained
    /// terminal/stop-prefix restore + replay (the driver then replays through
    /// [`Self::pp_forward_verify`] and calls [`Self::pp_window_release`]).
    /// A failed rollback poisons and quarantines the mesh.
    pub(crate) fn pp_window_restore(
        &mut self,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        retain: bool,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_window_restore_detached(&mut mesh, gpus, window, retain);
        self.pp = Some(mesh);
        result
    }

    fn pp_window_restore_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        retain: bool,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let (checkpoint, target, head) = {
            let record = mesh
                .windows
                .require(window, &[WindowPhase::Open, WindowPhase::Retained])
                .map_err(pp_error)?;
            (
                record.checkpoint.clone(),
                record.target.clone(),
                record.head,
            )
        };
        let context = WindowCtx::live(
            &target,
            head,
            if retain {
                WindowRestore::Retain
            } else {
                WindowRestore::Consume
            },
            WindowFinalize::Commit,
        );
        let outcome = with_bindings(self, mesh, |bindings| {
            bindings.set_window(context);
            gpus.restore_mesh(&mut *bindings, &window.key, checkpoint)
        })?;
        match outcome {
            Ok(()) => {
                if !retain {
                    mesh.windows.remove(window);
                }
                Ok(())
            }
            Err(error) => {
                mesh.windows.remove(window);
                Err(lifecycle_failure(mesh, "restore", &error, error.poisoned()))
            }
        }
    }

    /// Keep the first `keep` rows (`1 + target_commit_accept_len`) of the
    /// verified `block`: every band runs its own row-aware repair
    /// (`rollback_verify_rows_retain_with`: its GDN recurrence prefix from its
    /// own ticket and captured rows, conv history, QSA marks and, on the input
    /// band, the PLE convolution rows its own forward normalized) through
    /// `Gpus::repair_mesh_prefix`. No target replay. Refused BEFORE any
    /// mutation unless the window was opened armed, `1 <= keep < block.len() <=
    /// window.rows()`, every band's ring covers the block and every band sits
    /// at the verify's end; after that a failing band poisons and quarantines
    /// the mesh. Every local ticket stays active for the caller's finalize or
    /// restore. The output head is not touched.
    pub(crate) fn pp_window_repair_keep(
        &mut self,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        keep: usize,
        block: &[u32],
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_window_repair_detached(&mut mesh, gpus, window, keep, block);
        self.pp = Some(mesh);
        result
    }

    fn pp_window_repair_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        keep: usize,
        block: &[u32],
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let (target, head) = {
            let record = mesh
                .windows
                .require(window, &[WindowPhase::Open])
                .map_err(pp_error)?;
            (record.target.clone(), record.head)
        };
        let bands: Vec<(usize, usize)> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| (band.state.row_capture_rows(), band.state.position))
            .collect();
        check_repair_request(window, keep, block.len(), &bands).map_err(pp_error)?;
        let epoch = ForwardEpoch {
            mesh_epoch: gpus.mesh.epoch(),
            load_id: mesh.load_id,
            request_id: mesh.request_id,
            op_id: mesh.next_op,
            route: MeshRoute::Verify,
            start_position: window.start,
            actual_rows: block.len(),
            route_rows: block.len(),
            policy_id: PP_POLICY_ID,
        };
        let context = WindowCtx::live(
            &target,
            head,
            WindowRestore::Consume,
            WindowFinalize::Commit,
        );
        let outcome = with_bindings(self, mesh, |bindings| {
            bindings.set_window(context);
            gpus.repair_mesh_prefix(&mut *bindings, &epoch, keep, block)
        })?;
        outcome.map_err(|message| {
            // The bands hold mixed prefixes and no ticket reconciles them.
            mesh.quarantine(message.clone());
            pp_error(message)
        })
    }

    fn pp_window_head_ticket(
        &self,
        window: &Qwen4PpWindow,
    ) -> Result<MtpGpuStateSnapshot, BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.windows
            .require(window, &[WindowPhase::Open])
            .map_err(pp_error)?
            .head
            .ok_or_else(|| pp_error("the window has no head ticket"))
    }

    /// [`Self::mtp_truncate_retain`] with the window's head ticket: keep the
    /// head's first `keep` appended rows. Host-only.
    pub(crate) fn pp_window_head_truncate_retain(
        &mut self,
        window: &Qwen4PpWindow,
        keep: usize,
    ) -> Result<(), BundleError> {
        let head = self.pp_window_head_ticket(window)?;
        self.mtp_truncate_retain(head, keep)
    }

    /// [`Self::mtp_restore_retain`] with the window's head ticket, on the
    /// output device (the head half of a restore-and-replay finish).
    pub(crate) fn pp_window_head_restore_retain(
        &mut self,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
    ) -> Result<(), BundleError> {
        let head = self.pp_window_head_ticket(window)?;
        let gpu = self.pp_output_gpu(gpus)?;
        self.mtp_restore_retain(gpu, head)
    }

    /// Whether `window` is still the live window: not finalized, restored
    /// (consuming), released, superseded or reset.
    pub(crate) fn pp_window_live(&self, window: &Qwen4PpWindow) -> bool {
        self.pp.as_ref().is_some_and(|mesh| {
            mesh.windows
                .require(window, &[WindowPhase::Open, WindowPhase::Retained])
                .is_ok()
        })
    }

    /// Finish the window (frozen §3.3 step 7). Call only after EVERY fallible
    /// head and pending-hidden GPU operation succeeded. Every band's and the
    /// head's ticket is validated by `Gpus::finalize_mesh` for every owner
    /// before any owner finalizes; it then commits (`retain == false`, tickets
    /// consumed) or retains (`retain == true`: every local ticket stays
    /// rollback-able for [`Self::pp_window_restore`]`(.., true)` until the next
    /// open, a reset or [`Self::pp_window_release`]). A failed validation
    /// rolls EVERY owner back through `restore_mesh` semantics (healthy owners
    /// with still-valid tickets are restored, otherwise the mesh is poisoned
    /// and quarantined) and kills the window; the caller does not restore.
    pub(crate) fn pp_window_finalize(
        &mut self,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        retain: bool,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_window_finalize_detached(&mut mesh, gpus, window, retain);
        self.pp = Some(mesh);
        result
    }

    fn pp_window_finalize_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        window: &Qwen4PpWindow,
        retain: bool,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let (checkpoint, target, head) = {
            let record = mesh
                .windows
                .require(window, &[WindowPhase::Open])
                .map_err(pp_error)?;
            (
                record.checkpoint.clone(),
                record.target.clone(),
                record.head,
            )
        };
        // The head finish, the ring repair and the pending-hidden copy enqueue
        // device work after the verify epoch drained: complete it all before any
        // ticket is validated or any owner's host-only transition runs. A device
        // that cannot synchronize leaves the window's state unknown.
        if let Err(message) = sync_all_devices(gpus) {
            mesh.windows.remove(window);
            mesh.quarantine(message.clone());
            return Err(pp_error(message));
        }
        let context = WindowCtx::live(
            &target,
            head,
            WindowRestore::Consume,
            if retain {
                WindowFinalize::Retain
            } else {
                WindowFinalize::Commit
            },
        );
        let outcome = with_bindings(self, mesh, |bindings| {
            bindings.set_window(context);
            gpus.finalize_mesh(&mut *bindings, &window.key, checkpoint)
        })?;
        match outcome {
            Ok(()) => {
                if retain {
                    mesh.windows.retain(window).map_err(pp_error)?;
                } else {
                    mesh.windows.remove(window);
                }
                Ok(())
            }
            Err(error) => {
                mesh.windows.remove(window);
                Err(lifecycle_failure(
                    mesh,
                    "finalize",
                    &error,
                    error.poisoned(),
                ))
            }
        }
    }

    /// Host-only release of a window (the aggregate `release_window`): every
    /// local ticket that still validates is committed, one that no longer does
    /// (a snapshot, reset or restore superseded it) is left alone. A handle that
    /// is no longer live releases nothing.
    pub(crate) fn pp_window_release(&mut self, window: &Qwen4PpWindow) {
        let Some(mut mesh) = self.pp.take() else {
            return;
        };
        if let Some(record) = mesh.windows.take(window) {
            for (index, local) in record.target.iter().enumerate() {
                let band = band_mut(&mut mesh.bands, self, index);
                if band.validate_commit(*local).is_ok() {
                    band.commit_validated(*local);
                }
            }
            if let Some(head) = record.head {
                if self.mtp_validate_commit(head).is_ok() {
                    self.mtp_commit_validated(head);
                }
            }
        }
        self.pp = Some(mesh);
    }

    fn pp_forward_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        request: &Request<'_>,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let tokens = request.tokens;
        let vocab = self.config.vocab_size;
        if tokens.is_empty() {
            return Err(pp_error("cannot process an empty token slice"));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| pp_error("wide hidden width overflows"))?;
        match request.route {
            MeshRoute::Verify => {
                if tokens.len() > mesh.chunk {
                    return Err(pp_error("a verify block exceeds the canonical chunk"));
                }
                if request.rows != Qwen4OutputRows::All
                    || request.logits.is_none()
                    || request.wide.is_none()
                {
                    return Err(pp_error(
                        "a verify needs all-row logits and a wide hidden capture",
                    ));
                }
            }
            MeshRoute::Decode if tokens.len() != 1 => {
                return Err(pp_error("a decode step is exactly one token"));
            }
            _ => {}
        }
        if let Some(wide) = request.wide {
            let elements = tokens
                .len()
                .checked_mul(width)
                .ok_or_else(|| pp_error("wide hidden capture shape overflows"))?;
            if wide.dtype != rdna_compute::DType::F32 || wide.numel() < elements {
                return Err(pp_error(
                    "wide hidden capture must be F32 with room for every row",
                ));
            }
        }
        let arming: Vec<(bool, usize)> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| (band.state.row_capture_armed, band.state.row_capture_rows()))
            .collect();
        check_epoch_arming(&arming, request.route == MeshRoute::Verify, tokens.len())
            .map_err(pp_error)?;
        if tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(pp_error("token id is outside the embedding vocabulary"));
        }
        let start = self.state.position;
        let end = start
            .checked_add(tokens.len())
            .filter(|&end| end <= self.state.max_seq_len)
            .ok_or_else(|| pp_error("the request exceeds the context capacity"))?;
        if mesh
            .bands
            .iter()
            .any(|band| band.state.position != start || band.state.max_seq_len < end)
        {
            return Err(pp_error(
                "the bands disagree on position or context capacity",
            ));
        }
        let out_rows = request.rows.count(tokens.len());
        match (request.logits, request.top1) {
            (None, Some(_)) => return Err(pp_error("top-1 rows require logits")),
            (Some(logits), top1) => {
                if logits.dtype != rdna_compute::DType::F32 || logits.numel() != out_rows * vocab {
                    return Err(pp_error(
                        "logits must be F32 with one vocabulary row per output row",
                    ));
                }
                if top1.is_some_and(|top1| {
                    top1.dtype != rdna_compute::DType::Raw
                        || top1.buf.size() < out_rows * std::mem::size_of::<i32>()
                }) {
                    return Err(pp_error("top-1 output must hold one i32 per output row"));
                }
            }
            (None, None) => {}
        }
        let tiles = match request.session {
            Some(run) => {
                if run.start != start
                    || run.tiles.iter().map(|tile| tile.rows).sum::<usize>() != tokens.len()
                    || run
                        .tiles
                        .iter()
                        .any(|tile| tile.rows == 0 || tile.rows > mesh.chunk)
                {
                    return Err(pp_error("the session schedule does not cover the request"));
                }
                run.tiles.clone()
            }
            None => plan_tiles(tokens.len(), mesh.chunk),
        };
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let outcome = prepare_tiles(
            self,
            mesh,
            gpus,
            &mut forwards,
            request.route,
            &tiles,
            start,
        )
        .and_then(|layouts| run_tiles(self, mesh, gpus, &mut forwards, request, &tiles, &layouts));
        restore_forwards(&mut mesh.bands, self, forwards);
        outcome.map_err(|error| {
            if error.fatal {
                mesh.quarantine(error.message.clone());
            }
            pp_error(error.message)
        })
    }

    /// One token (`Some`) or the device argmax of `logits` as the previous
    /// forward left them (`None`); returns the token. Mirrors
    /// `Qwen4GpuForward::forward_token_or_argmax` on the output device.
    pub fn pp_forward_token_or_argmax(
        &mut self,
        gpus: &mut Gpus,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_token_or_argmax_detached(&mut mesh, gpus, token, logits);
        self.pp = Some(mesh);
        result
    }

    fn pp_token_or_argmax_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        mesh.require_ready(gpus)?;
        let token = match token {
            Some(token) => token,
            None => {
                let vocab = self.config.vocab_size;
                if logits.dtype != rdna_compute::DType::F32 || logits.numel() != vocab {
                    return Err(pp_error("logits must be one F32 vocabulary row"));
                }
                let owner = gpus
                    .mesh_output_owner()
                    .ok_or_else(|| pp_error("the mesh has no output owner"))?;
                let gpu = gpus
                    .devices
                    .get_mut(owner)
                    .ok_or_else(|| pp_error("the output device is unavailable"))?;
                gpu.bind_thread().map_err(BundleError::Hip)?;
                rdna_compute::tensor_ops::argmax_f32_host(gpu, logits).map_err(BundleError::Hip)?
            }
        };
        self.pp_forward_detached(
            mesh,
            gpus,
            &Request {
                tokens: std::slice::from_ref(&token),
                logits: Some(logits),
                top1: None,
                rows: Qwen4OutputRows::Final,
                route: MeshRoute::Decode,
                session: None,
                wide: None,
            },
        )?;
        Ok(token)
    }

    /// Reset every band through the generic coordinator: drain every device,
    /// run each band's reset callback (bundle reset, uncommitted session
    /// captures dropped, PLE lookahead released), synchronize every device. The
    /// transport leases are untouched: the next epoch's ids stay monotonic. Any
    /// failure quarantines the mesh. Published aggregate records and their pins
    /// survive. Needs no daemon `Gpu`; errors when no mesh is attached.
    pub fn pp_reset(&mut self, gpus: &mut Gpus) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_reset_detached(&mut mesh, gpus);
        self.pp = Some(mesh);
        result
    }

    /// Prompt prefill of the full canonical `prompt`, keeping only the final
    /// logits row, on the explicit Prefill route. `reused == 0` resets the mesh
    /// (cold); `reused > 0` restores the aggregate record `pp_session_plan`
    /// chose through the mesh's session restore (every band probed before any
    /// is restored) and runs the rest. A missing or evicted piece starts cold;
    /// a failure once a restore began quarantines the mesh. Prompt, `reused`
    /// and `logits` are validated before any side effect.
    pub fn pp_prefill_final(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_prefill_final_detached(&mut mesh, gpus, prompt, reused, logits);
        self.pp = Some(mesh);
        result
    }

    fn pp_prefill_final_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let vocab = self.config.vocab_size;
        if prompt.is_empty() || reused >= prompt.len() {
            return Err(pp_error(
                "prefill requires a non-empty prompt and a reused prefix shorter than it",
            ));
        }
        if prompt.iter().any(|&token| token as usize >= vocab) {
            return Err(pp_error("token id is outside the embedding vocabulary"));
        }
        if prompt.len() > self.state.max_seq_len {
            return Err(pp_error("the prompt exceeds the context capacity"));
        }
        if logits.dtype != rdna_compute::DType::F32 || logits.numel() != vocab {
            return Err(pp_error("logits must be one F32 vocabulary row"));
        }
        // Message-end snapshots stay refused; they never narrow sessions silently.
        if mesh.sessions_enabled && self.turn_snapshots_active() {
            return Err(BundleError::Forward(turn_snapshots_refusal(mesh.stages())));
        }
        let sessions = mesh.sessions_enabled;
        if reused != 0 && !sessions {
            return Err(pp_error(
                "session reuse needs a session cache on every band",
            ));
        }
        // Nothing of an earlier, uncommitted request may publish.
        mesh.sessions.abort();
        mesh.committable = None;
        // A new prefill supersedes any open or retained MTP window.
        mesh.windows.clear();
        let route = SessionRoute::Ar;
        if sessions {
            self.pp_sync_session_policy(mesh);
        }
        let mut start = 0;
        if reused != 0 {
            start = self.pp_restore_planned(mesh, gpus, prompt, reused, route)?;
        }
        if start == 0 {
            self.pp_reset_detached(mesh, gpus)?;
            if sessions {
                self.pp_arm_cold(mesh, gpus, prompt, route)?;
            }
        }
        let run = if sessions {
            Some(self.pp_session_run(mesh, prompt, route, start)?)
        } else {
            None
        };
        let result = self.pp_forward_detached(
            mesh,
            gpus,
            &Request {
                tokens: &prompt[start..],
                logits: Some(logits),
                top1: None,
                rows: Qwen4OutputRows::Final,
                route: MeshRoute::Prefill,
                session: run.as_ref(),
                wide: None,
            },
        );
        match &result {
            Ok(()) if sessions => mesh.committable = Some(mesh.request_id),
            Ok(()) => {}
            Err(_) => mesh.sessions.abort(),
        }
        result
    }

    /// The root's tail/message-end settings are authoritative: every band
    /// adopts them before its scope or boundaries are consulted.
    fn pp_sync_session_policy(&self, mesh: &mut Qwen4PpMesh) {
        let policy = self.pp_session_policy();
        for band in &mut mesh.bands {
            band.pp_adopt_session_policy(policy);
        }
    }

    /// Restore the planned aggregate record for `reused` tokens, or decide to
    /// start cold (`Ok(0)`) when it is gone, stale or any band's piece is no
    /// longer resident (every owner is probed before the first restore; nothing
    /// is touched then). `Err` only after a restore began: the mesh is poisoned
    /// and quarantined.
    fn pp_restore_planned(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<usize, BundleError> {
        let stages = mesh.stages();
        // Other records' pins must not make the restore's context growth fail.
        reclaim_records_for_growth(self, mesh, gpus, prompt.len());
        let prefix = &prompt[..reused];
        let planned = match mesh.sessions.planned() {
            Some((len, record))
                if len == reused
                    && record.route == route
                    && record.pieces.len() == stages
                    && record.pieces.iter().enumerate().all(|(index, piece)| {
                        piece.matches_prefix(prefix)
                            && (if index + 1 < stages {
                                &mesh.bands[index]
                            } else {
                                &*self
                            })
                            .pp_session_piece_key(prefix, route)
                            .as_ref()
                                == Some(piece)
                    }) =>
            {
                Some(record.pieces.clone())
            }
            _ => None,
        };
        let Some(pieces) = planned else {
            eprintln!(
                "  qwen4 topology.pp: no valid session record for {reused} tokens; cold start"
            );
            self.pp_release_plan(mesh);
            return Ok(0);
        };
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let epoch = lifecycle_epoch(mesh, gpus, reused);
        let (outcome, started) = {
            let ranks = mesh_ranks(&mut mesh.bands, self, &mut forwards);
            match Qwen4MeshBindings::new(ranks) {
                Err(error) => (Err(error), false),
                Ok(mut bindings) => {
                    bindings.set_session(prompt, route);
                    let tickets = session_tickets(&epoch, &pieces);
                    let outcome = bindings
                        .set_session_restore(pieces)
                        .and_then(|()| gpus.restore_mesh_session(&mut bindings, &epoch, &tickets));
                    let started = bindings.session_restore_started();
                    (outcome, started)
                }
            }
        };
        restore_forwards(&mut mesh.bands, self, forwards);
        match outcome {
            Ok(()) => Ok(reused),
            Err(error) if started => {
                mesh.quarantine(error.clone());
                Err(pp_error(error))
            }
            Err(error) => {
                eprintln!("  qwen4 topology.pp: session restore refused ({error}); cold start");
                self.pp_release_plan(mesh);
                Ok(0)
            }
        }
    }

    fn pp_release_plan(&mut self, mesh: &mut Qwen4PpMesh) {
        let Qwen4PpMesh {
            bands, sessions, ..
        } = mesh;
        if let Some(mut drivers) = session_drivers(bands, self) {
            sessions.release_plan(&mut drivers);
        }
    }

    /// Cold start of every band's session turn at 0, after the mesh reset.
    fn pp_arm_cold(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> Result<(), BundleError> {
        let stages = mesh.stages();
        for index in 0..stages {
            let gpu = &mut gpus.devices[index];
            let armed = gpu
                .bind_thread()
                .map_err(|error| error.to_string())
                .and_then(|()| {
                    band_mut(&mut mesh.bands, self, index).pp_session_begin(gpu, prompt, route, 0)
                });
            if let Err(error) = armed {
                let message = format!("band {index} session start failed: {error}");
                mesh.quarantine(message.clone());
                return Err(pp_error(message));
            }
        }
        Ok(())
    }

    /// The canonical schedule: the root's armed boundaries (computed once) are
    /// installed on every band, and the tiles are cut at them.
    fn pp_session_run<'p>(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        prompt: &'p [u32],
        route: SessionRoute,
        start: usize,
    ) -> Result<SessionRun<'p>, BundleError> {
        let (boundaries, lineage) = self
            .pp_session_schedule()
            .ok_or_else(|| pp_error("the output band has no armed session schedule"))?;
        for band in &mut mesh.bands {
            band.pp_session_set_schedule(boundaries.clone(), lineage)
                .map_err(pp_error)?;
        }
        let tiles = prefill_schedule(start, prompt.len(), mesh.chunk, &boundaries, lineage);
        Ok(SessionRun {
            prompt,
            route,
            start,
            tiles,
            boundaries,
            lineage,
        })
    }

    /// The longest published prefix of `prompt` every band holds, strictly
    /// shorter than the prompt (0 on a miss, a refused option or an unhealthy
    /// mesh). Only the aggregate record plans: never the root's own cache or
    /// live state. The chosen record keeps its pins until the prefill restores
    /// it or the plan is abandoned.
    pub fn pp_session_plan(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> usize {
        let Some(mut mesh) = self.pp.take() else {
            return 0;
        };
        let planned = self.pp_session_plan_detached(&mut mesh, gpus, prompt, route);
        self.pp = Some(mesh);
        planned
    }

    fn pp_session_plan_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> usize {
        if route != SessionRoute::Ar {
            eprintln!(
                "  qwen4 topology.pp: MTP-route session caches are not supported; cold start"
            );
            return 0;
        }
        if !mesh.sessions_enabled
            || self.turn_snapshots_active()
            || mesh.require_ready(gpus).is_err()
        {
            return 0;
        }
        self.pp_sync_session_policy(mesh);
        let stages = mesh.stages();
        let planned = {
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => sessions.plan(&mut drivers, prompt, route),
                None => 0,
            }
        };
        if planned == 0 {
            return 0;
        }
        // A record published under other schedule settings names keys this
        // scope would not find: it is not a hit.
        let prefix = &prompt[..planned];
        let current = mesh.sessions.planned().is_some_and(|(len, record)| {
            len == planned
                && record.pieces.len() == stages
                && record.pieces.iter().enumerate().all(|(index, piece)| {
                    (if index + 1 < stages {
                        &mesh.bands[index]
                    } else {
                        &*self
                    })
                    .pp_session_piece_key(prefix, route)
                    .as_ref()
                        == Some(piece)
                })
        });
        if !current {
            self.pp_release_plan(mesh);
            return 0;
        }
        planned
    }

    /// Publish the request's aggregate record(s) after the generation layer's
    /// commit point, keeping each band's live state when `consumed` extends the
    /// turn. Host-only. Publishes only a completed prefill of the current
    /// request on a healthy mesh; otherwise the staged candidates are dropped
    /// and nothing publishes on any band.
    pub fn pp_session_commit_live(&mut self, gpus: &mut Gpus, consumed: &[u32]) {
        self.pp_session_commit_with(gpus, Some(consumed));
    }

    /// [`Self::pp_session_commit_live`] without keeping any live state.
    pub fn pp_session_commit(&mut self, gpus: &mut Gpus) {
        self.pp_session_commit_with(gpus, None);
    }

    fn pp_session_commit_with(&mut self, gpus: &Gpus, consumed: Option<&[u32]>) {
        let Some(mut mesh) = self.pp.take() else {
            return;
        };
        let valid = mesh.sessions_enabled
            && mesh.require_ready(gpus).is_ok()
            && mesh.committable == Some(mesh.request_id);
        mesh.committable = None;
        if valid {
            for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *self)) {
                band.pp_session_commit_local(consumed);
            }
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => {
                    sessions.publish(&mut drivers);
                }
                None => sessions.abort(),
            }
        } else {
            mesh.sessions.abort();
        }
        self.pp = Some(mesh);
    }

    /// Drop every aggregate record (releasing all pins) and then every band's
    /// session snapshots; the pins go first so no band's clear can leave a
    /// stale pin. Bands are cleared only on a healthy mesh (a quarantined one
    /// may still have work in flight).
    pub fn pp_session_clear(&mut self, gpus: &mut Gpus) {
        let Some(mut mesh) = self.pp.take() else {
            return;
        };
        mesh.committable = None;
        {
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => sessions.clear(&mut drivers),
                None => sessions.abort(),
            }
        }
        if mesh.sessions_enabled && mesh.require_ready(gpus).is_ok() {
            for index in 0..mesh.stages() {
                let gpu = &mut gpus.devices[index];
                if gpu.bind_thread().is_ok() {
                    if let Some(session) = band_mut(&mut mesh.bands, self, index).session.as_mut() {
                        session.clear(gpu);
                    }
                }
            }
        }
        self.pp = Some(mesh);
    }

    fn pp_reset_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let epoch = lifecycle_epoch(mesh, gpus, self.state.position);
        let outcome = {
            match Qwen4MeshBindings::new(mesh_ranks(&mut mesh.bands, self, &mut forwards)) {
                Ok(mut bindings) => Ok(gpus.reset_mesh(&mut bindings, &epoch)),
                Err(error) => Err(error),
            }
        };
        restore_forwards(&mut mesh.bands, self, forwards);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => return Err(pp_error(error)),
        };
        let next_request = mesh.request_id.checked_add(1);
        let mut failures: Vec<String> = Vec::new();
        if let Err(error) = outcome {
            failures.push(error);
        }
        if next_request.is_none() {
            failures.push("request ids are exhausted".to_string());
        }
        if !failures.is_empty() {
            // Captured graphs and the replay proof window describe state that
            // no longer matches the bands.
            for gpu in &mut gpus.devices {
                gpu.invalidate_graph_state();
                gpu.replay.invalidate_replay_observation_window();
            }
            let message = failures.join("; ");
            mesh.quarantine(message.clone());
            return Err(pp_error(message));
        }
        // A new request starts with clear router fault words, on every device and
        // only here: `reset_mesh` just synchronized them all, and a clear anywhere
        // mid-request would drop an earlier row's pending report (the single-GPU
        // `Qwen4Bundle::reset` convention).
        for gpu in &gpus.devices {
            gpu.clear_router_fault();
        }
        mesh.request_id = next_request.unwrap_or(mesh.request_id);
        // Uncommitted candidates belong to the request that just ended.
        mesh.sessions.abort();
        mesh.committable = None;
        // Every outstanding window's tickets died with the bands' state.
        mesh.windows.clear();
        self.state.row_capture_armed = false;
        for band in &mut mesh.bands {
            band.state.row_capture_armed = false;
        }
        Ok(())
    }

    /// `apply_penalty_table` on the output device with the root's staging.
    pub fn pp_apply_penalty_table(
        &mut self,
        gpus: &mut Gpus,
        logits: &GpuTensor,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        if table.is_noop() {
            return Ok(());
        }
        let vocab = self.config.vocab_size;
        let mesh = self
            .pp
            .as_mut()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.require_ready(gpus)?;
        let owner = gpus
            .mesh_output_owner()
            .ok_or_else(|| pp_error("the mesh has no output owner"))?;
        let gpu = gpus
            .devices
            .get_mut(owner)
            .ok_or_else(|| pp_error("the output device is unavailable"))?;
        let applied = gpu.bind_thread().map_err(BundleError::Hip).and_then(|()| {
            gpu.apply_penalty_table(
                &mut self.penalty_stage,
                logits,
                vocab,
                table.flags(),
                table.row_ends(),
                table.entries(),
            )
            .map_err(BundleError::Hip)
        });
        applied?;
        // The daemon reads on its own stream of this physical device. A failed
        // synchronization leaves the staged penalty work in an unknown state.
        if let Err(error) = gpu.hip.device_synchronize() {
            let message = format!("penalty synchronization failed: {error}");
            mesh.quarantine(message.clone());
            return Err(pp_error(message));
        }
        Ok(())
    }

    /// Mapped QSA context bytes of every band on its own device, MTP = 0.
    pub fn pp_qsa_context_committed_bytes(
        &self,
        gpus: &Gpus,
    ) -> Result<(usize, usize), BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        if !mesh.identity_matches(gpus) {
            return Err(pp_error("the supplied devices are not this mesh's"));
        }
        let mut total = 0usize;
        for (index, band) in mesh.bands.iter().chain(std::iter::once(self)).enumerate() {
            let bytes = band
                .state
                .mapped_context_bytes(&gpus.devices[index])
                .map_err(BundleError::State)?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| pp_error("context byte count overflows"))?;
        }
        Ok((total, 0))
    }

    /// Unload: quiesce PLE, drain every device, then free the peers and the
    /// root on their own devices of the caller's `gpus` (which the caller
    /// still owns and drops last). If the quiesce or the drain fails nothing
    /// is freed: the root and the holder are leaked with `mem::forget`
    /// (in-flight work or host readers may still reference them) and the
    /// caller must forget its `Gpus` too.
    pub fn pp_free_gpu(mut self, gpus: &mut Gpus) -> Result<(), BundleError> {
        let Some(mesh) = self.pp.take() else {
            std::mem::forget(self);
            return Err(pp_error("no pipeline mesh is attached"));
        };
        // The caller's `Gpus` must be the one this holder was attached with;
        // nothing is drained or freed on the wrong hardware.
        if !mesh.identity_matches(gpus) {
            std::mem::forget(self);
            std::mem::forget(mesh);
            return Err(pp_error(
                "unload was given a different device set; the mesh resources stay allocated until process exit",
            ));
        }
        let (quiesce_error, drained) = {
            let bands = &mesh.bands;
            let root = &self;
            let devices = &mut *gpus;
            quiesce_then_drain(
                || {
                    let mut first = None;
                    for band in bands.iter().chain(std::iter::once(root)) {
                        if let Err(error) = band.quiesce_ple() {
                            first.get_or_insert(error);
                        }
                    }
                    first.map_or(Ok(()), Err)
                },
                || devices.drain_mesh_edges(),
            )
        };
        let stages = mesh.stages();
        // A PLE quiesce failure may leave host readers or enqueuers alive, so
        // it forces the same leak as a failed drain, even when the GPU drain
        // succeeded. The caller then forgets `gpus` too.
        if let Some(reason) = teardown_leak_reason(&quiesce_error, &drained) {
            std::mem::forget(self);
            std::mem::forget(mesh);
            return Err(pp_error(format!(
                "unload could not release the mesh; its resources stay allocated until process exit: {reason}"
            )));
        }
        let Qwen4PpMesh {
            mut bands,
            mut sessions,
            lane_bounce,
            lane_events,
            ..
        } = *mesh;
        // Edges are drained, so no transfer or recorded fence is pending. Each
        // event dies on the device that created it. A bind failure leaves
        // the owning context unknown: that event leaks and so does the whole
        // mesh (below); a destroy failure on a bound device does not stop
        // the remaining release and is reported last.
        let events = destroy_lane_events(gpus, lane_events);
        if let Some(error) = events.bind {
            std::mem::forget(self);
            std::mem::forget(bands);
            std::mem::forget(sessions);
            std::mem::forget(lane_bounce);
            return Err(pp_error(format!(
                "unload could not bind a lane event owner: {error}"
            )));
        }
        // The host allocation belongs to device zero. Edges are already
        // drained, so no transfer can retain a reference to this storage.
        if let Err(error) = gpus.devices[0].bind_thread() {
            std::mem::forget(self);
            std::mem::forget(bands);
            std::mem::forget(sessions);
            std::mem::forget(lane_bounce);
            return Err(pp_error(format!(
                "unload could not bind the lane bounce owner: {error}"
            )));
        }
        drop(lane_bounce);
        // Aggregate pins go before any band's cache is freed.
        if let Some(mut drivers) = session_drivers(&mut bands, &mut self) {
            sessions.clear(&mut drivers);
        }
        let mut first = events.destroy.map(pp_error);
        for (index, band) in bands.into_iter().enumerate() {
            if let Err(error) = Qwen4Bundle::free_gpu(band, &mut gpus.devices[index]) {
                first.get_or_insert(error);
            }
        }
        if let Err(error) = Qwen4Bundle::free_gpu(self, &mut gpus.devices[stages - 1]) {
            first.get_or_insert(error);
        }
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn pp_lane_edge_bytes_preserve_bf16_and_f32_streams() {
        assert_eq!(lane_edge_bytes(8, 4, 64, true), Some(4096));
        assert_eq!(lane_edge_bytes(8, 4, 64, false), Some(8192));
        assert_eq!(lane_edge_bytes(usize::MAX, 4, 64, false), None);
    }

    #[test]
    fn pp_lane_spec_refusal_is_explicit_and_ar_is_admitted() {
        assert!(refuse_lane_spec(None).is_ok());
        let error = refuse_lane_spec(Some(crate::lane::LaneSpecConfig {
            k: 3,
            end_of_turn: None,
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains(
            "qwen4 topology.pp: native-MTP lanes are not supported yet; stage AR lanes (serve.batch_spec = false)"
        ));
    }

    #[test]
    fn tiles_are_canonical_in_order_and_unpadded() {
        assert_eq!(plan_tiles(1, 4096), [Tile { offset: 0, rows: 1 }]);
        assert_eq!(
            plan_tiles(9, 4),
            [
                Tile { offset: 0, rows: 4 },
                Tile { offset: 4, rows: 4 },
                Tile { offset: 8, rows: 1 },
            ]
        );
        assert_eq!(plan_tiles(8, 4).len(), 2);
        assert!(plan_tiles(0, 4).is_empty());
        let total: usize = plan_tiles(10_001, 1536).iter().map(|tile| tile.rows).sum();
        assert_eq!(total, 10_001);
    }

    #[test]
    fn head_is_computed_only_where_requested() {
        use Qwen4OutputRows::{All, Final};
        // Final rows: only the last tile carries the head.
        assert_eq!(tile_mode(true, false, Final, false), MODE_TRUNK);
        assert_eq!(tile_mode(true, false, Final, true), MODE_LOGITS);
        assert_eq!(tile_mode(true, true, Final, true), MODE_LOGITS_TOP1);
        // All rows: every tile.
        assert_eq!(tile_mode(true, false, All, false), MODE_LOGITS);
        assert_eq!(tile_mode(true, true, All, false), MODE_LOGITS_TOP1);
        // No logits tensor: trunk only, even on the last tile.
        assert_eq!(tile_mode(false, false, Final, true), MODE_TRUNK);
        assert_eq!(tile_mode(false, false, All, true), MODE_TRUNK);
    }

    #[test]
    fn quarantine_is_sticky_and_keeps_the_first_reason() {
        let mut health = Health::Ready;
        assert_eq!(health, Health::Ready);
        let reason = "drain failed".to_string();
        if health == Health::Ready {
            health = Health::Quarantined(reason.clone());
        }
        if health == Health::Ready {
            health = Health::Quarantined("later".to_string());
        }
        assert_eq!(health, Health::Quarantined(reason));
    }

    #[test]
    fn unload_quiesces_before_draining_and_reports_both() {
        let order = RefCell::new(Vec::new());
        let (quiesce, drained) = quiesce_then_drain(
            || {
                order.borrow_mut().push("quiesce");
                Ok(())
            },
            || {
                order.borrow_mut().push("drain");
                Ok(())
            },
        );
        assert!(quiesce.is_none() && drained.is_ok());
        assert_eq!(*order.borrow(), ["quiesce", "drain"]);

        // A failed drain is surfaced (the caller then leaks every resource);
        // a failed quiesce does not skip the drain.
        let order = RefCell::new(Vec::new());
        let (quiesce, drained) = quiesce_then_drain(
            || {
                order.borrow_mut().push("quiesce");
                Err(pp_error("quiesce"))
            },
            || {
                order.borrow_mut().push("drain");
                Err("device lost".to_string())
            },
        );
        assert!(quiesce.is_some());
        assert_eq!(drained, Err("device lost".to_string()));
        assert_eq!(*order.borrow(), ["quiesce", "drain"]);
    }
}

#[cfg(test)]
mod teardown_tests {
    use super::*;

    #[test]
    fn quiesce_failure_retains_resources_even_when_the_drain_succeeds() {
        let (quiesce, drained) = quiesce_then_drain(|| Err(pp_error("timed out")), || Ok(()));
        assert!(drained.is_ok());
        let reason = teardown_leak_reason(&quiesce, &drained).expect("must leak");
        assert!(reason.contains("PLE quiesce failed"));
        assert!(!reason.contains("drain failed"));
    }

    #[test]
    fn drain_failure_retains_resources_and_reports_both_causes() {
        let (quiesce, drained) = quiesce_then_drain(
            || Err(pp_error("timed out")),
            || Err("device lost".to_string()),
        );
        let reason = teardown_leak_reason(&quiesce, &drained).expect("must leak");
        assert!(reason.contains("PLE quiesce failed") && reason.contains("device lost"));
        let (quiesce, drained) = quiesce_then_drain(|| Ok(()), || Err("device lost".to_string()));
        assert!(teardown_leak_reason(&quiesce, &drained).is_some());
    }

    #[test]
    fn clean_quiesce_and_drain_frees() {
        let (quiesce, drained) = quiesce_then_drain(|| Ok(()), || Ok(()));
        assert!(teardown_leak_reason(&quiesce, &drained).is_none());
    }

    #[test]
    fn load_ids_never_wrap() {
        assert!(next_load_id().unwrap() >= 1);
        assert_eq!(u64::MAX.checked_add(1), None);
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashSet;

    const FORMAT: Qwen4StateFormat = Qwen4StateFormat::F32;

    fn meta(
        index: usize,
        bands: &[Range<usize>],
        devices: &[&str],
        mesh_epoch: u64,
        load_id: u64,
    ) -> Qwen4PpBandMeta {
        Qwen4PpBandMeta::from_bands(
            index,
            bands.to_vec(),
            devices.iter().map(|device| device.to_string()).collect(),
            mesh_epoch,
            load_id,
            8192,
        )
    }

    fn policy() -> PpSessionPolicy {
        PpSessionPolicy {
            turn_snapshots: false,
            tail_snapshot: true,
            tail_snapshot_min_gap: 1024,
            tail_snapshot_min_prompt: 16384,
            turn_end_token: Some(9),
        }
    }

    fn scope(band: &Qwen4PpBandMeta, policy: PpSessionPolicy, chunk: usize) -> String {
        session_scope(SessionRoute::Ar, chunk, FORMAT, false, Some((band, policy)))
    }

    #[test]
    fn band_scopes_differ_by_band_placement_device_epoch_load_and_policy() {
        let even = [0..24, 24..48];
        let skew = [0..20, 20..48];
        let devices = ["0@gfx1100", "1@gfx1100"];
        let base = meta(0, &even, &devices, 5, 1);
        let tail_off = PpSessionPolicy {
            tail_snapshot: false,
            ..policy()
        };
        let other_gap = PpSessionPolicy {
            tail_snapshot_min_gap: 1,
            ..policy()
        };
        let no_token = PpSessionPolicy {
            turn_end_token: None,
            ..policy()
        };
        let scopes = [
            scope(&base, policy(), 8192),
            // another band of the same mesh
            scope(&meta(1, &even, &devices, 5, 1), policy(), 8192),
            // same band index, another placement (even the same layer count of
            // band 0 differs in the complete placement)
            scope(&meta(0, &skew, &devices, 5, 1), policy(), 8192),
            scope(
                &meta(0, &[0..24, 24..47, 47..48], &devices, 5, 1),
                policy(),
                8192,
            ),
            // another physical device set / order / arch
            scope(
                &meta(0, &even, &["1@gfx1100", "0@gfx1100"], 5, 1),
                policy(),
                8192,
            ),
            scope(
                &meta(0, &even, &["0@gfx1100", "1@gfx1201"], 5, 1),
                policy(),
                8192,
            ),
            // another mesh generation and another load
            scope(&meta(0, &even, &devices, 6, 1), policy(), 8192),
            scope(&meta(0, &even, &devices, 5, 2), policy(), 8192),
            // another canonical chunk
            scope(&base, policy(), 4096),
            // the shared schedule settings
            scope(&base, tail_off, 8192),
            scope(&base, other_gap, 8192),
            scope(&base, no_token, 8192),
        ];
        let distinct: HashSet<&String> = scopes.iter().collect();
        assert_eq!(distinct.len(), scopes.len(), "{scopes:#?}");
        let digests: HashSet<_> = scopes
            .iter()
            .map(|scope| hipfire_runtime::serve_contract::sha256_len_prefixed(&[scope.as_bytes()]))
            .collect();
        assert_eq!(digests.len(), scopes.len(), "PP scope digests must differ");
    }

    #[test]
    fn band_scope_is_deterministic_and_carries_the_band_identity() {
        let even = [0..24, 24..48];
        let devices = ["0@gfx1100", "1@gfx1100"];
        let band = meta(1, &even, &devices, 5, 1);
        let first = scope(&band, policy(), 8192);
        assert_eq!(first, scope(&band.clone(), policy(), 8192));
        assert!(first.contains("/pp1of2/layers24-48/place0-24,24-48/"));
        assert!(first.contains("/dev0@gfx1100,1@gfx1100/mesh5/load1"));
        assert_eq!(band.chunk(), 8192);
    }

    #[test]
    fn single_session_scope_is_unchanged() {
        let historical = |turns: &str| {
            format!(
                "qwen4/{:?}/chunk{}/{:?}{turns}",
                SessionRoute::Ar,
                8192,
                FORMAT
            )
        };
        assert_eq!(
            session_scope(SessionRoute::Ar, 8192, FORMAT, false, None),
            historical("")
        );
        assert_eq!(
            session_scope(SessionRoute::Ar, 8192, FORMAT, true, None),
            historical("/turns")
        );
        assert_ne!(
            session_scope(SessionRoute::Mtp, 8192, FORMAT, false, None),
            historical("")
        );
        // A PP scope extends, never rewrites, the Single one.
        let band = meta(0, &[0..24, 24..48], &["0@a", "1@a"], 1, 1);
        assert!(scope(&band, policy(), 8192).starts_with(&historical("")));
    }

    fn ends(tiles: &[Tile], start: usize) -> Vec<usize> {
        tiles
            .iter()
            .map(|tile| start + tile.offset + tile.rows)
            .collect()
    }

    #[test]
    fn schedule_cuts_at_every_boundary_and_the_offgrid_lineage_then_realigns() {
        // Regular boundaries, an off-grid lineage split (20), then the next
        // global chunk boundary (24, 32): no capture skipped, no tile
        // re-chunked relative to the split.
        let boundaries = [8, 16, 20, 24, 32];
        let tiles = prefill_schedule(0, 30, 8, &boundaries, Some(20));
        assert_eq!(ends(&tiles, 0), [8, 16, 20, 24, 30]);
        assert_eq!(
            tiles.iter().map(|tile| tile.offset).collect::<Vec<_>>(),
            [0, 8, 16, 20, 24]
        );
        // A warm start continues the same global schedule.
        let warm = prefill_schedule(16, 30, 8, &boundaries, Some(20));
        assert_eq!(ends(&warm, 16), [20, 24, 30]);
    }

    #[test]
    fn schedule_without_boundaries_is_chunk_relative_and_never_padded() {
        assert_eq!(ends(&prefill_schedule(0, 20, 8, &[], None), 0), [8, 16, 20]);
        assert!(prefill_schedule(5, 5, 8, &[], None).is_empty());
        // An off-grid start still lands on the next global chunk boundary.
        let tiles = prefill_schedule(5, 20, 8, &[8, 16], None);
        assert_eq!(ends(&tiles, 5), [8, 16, 20]);
        assert_eq!(tiles[0], Tile { offset: 0, rows: 3 });
        // A lineage past the prompt end changes nothing.
        assert_eq!(
            prefill_schedule(0, 30, 8, &[8, 16, 24], Some(40)),
            prefill_schedule(0, 30, 8, &[8, 16, 24], None)
        );
    }

    #[test]
    fn every_boundary_inside_the_prefill_is_a_tile_end_exactly_once() {
        let chunk = 8;
        for start in 0..12usize {
            for end in start + 1..41 {
                for lineage in std::iter::once(None).chain((start + 1..end).map(Some)) {
                    let mut boundaries: Vec<usize> = (start / chunk + 1..=end / chunk)
                        .map(|k| k * chunk)
                        .collect();
                    boundaries.extend(lineage);
                    boundaries.sort_unstable();
                    boundaries.dedup();
                    let tiles = prefill_schedule(start, end, chunk, &boundaries, lineage);
                    let ends = ends(&tiles, start);
                    assert_eq!(ends.last(), Some(&end));
                    assert!(ends.windows(2).all(|pair| pair[0] < pair[1]));
                    assert!(tiles.iter().all(|tile| (1..=chunk).contains(&tile.rows)));
                    for boundary in boundaries.iter().filter(|&&at| at <= end) {
                        assert_eq!(
                            ends.iter().filter(|&&at| at == *boundary).count(),
                            1,
                            "start {start} end {end} lineage {lineage:?} boundary {boundary}"
                        );
                    }
                    // Offsets are contiguous from the first computed row.
                    let mut next = 0;
                    for tile in &tiles {
                        assert_eq!(tile.offset, next);
                        next += tile.rows;
                    }
                    assert_eq!(next, end - start);
                }
            }
        }
    }

    #[test]
    fn aggregate_candidates_need_a_cold_piece_on_every_band() {
        assert_eq!(
            aggregate_candidate(None, 8, vec![Some(1), Some(2)]),
            Some(vec![1, 2])
        );
        // One band named no cold piece: no record can cover this boundary.
        assert_eq!(aggregate_candidate(None, 8, vec![Some(1), None]), None);
        assert_eq!(aggregate_candidate::<u8>(None, 8, vec![None, None]), None);
        // The lineage (tail) split is band-local, never an aggregate record.
        assert_eq!(
            aggregate_candidate(Some(8), 8, vec![Some(1), Some(2)]),
            None
        );
        assert_eq!(
            aggregate_candidate(Some(8), 16, vec![Some(1), Some(2)]),
            Some(vec![1, 2])
        );
    }
}

#[cfg(test)]
mod attach_policy_tests {
    use super::*;

    #[test]
    fn all_or_none_session_caches_are_admitted() {
        assert_eq!(
            attach_session_policy(2, &[true, true], &[false, false]),
            Ok(true)
        );
        assert_eq!(
            attach_session_policy(3, &[true, true, true], &[false; 3]),
            Ok(true)
        );
        assert_eq!(
            attach_session_policy(2, &[false, false], &[false, false]),
            Ok(false)
        );
    }

    #[test]
    fn mixed_cache_presence_fails_closed_for_any_band_order() {
        for present in [[true, false], [false, true]] {
            let error = attach_session_policy(2, &present, &[false, false]).unwrap_err();
            assert!(
                error.contains("disagree on session-cache presence"),
                "{error}"
            );
        }
        assert!(attach_session_policy(3, &[true, false, true], &[false; 3]).is_err());
    }

    #[test]
    fn message_end_snapshots_refuse_with_the_admission_sentence() {
        // On any band, with or without caches.
        for turns in [[true, false], [false, true], [true, true]] {
            for present in [[true, true], [false, false], [true, false]] {
                let error = attach_session_policy(2, &present, &turns).unwrap_err();
                assert_eq!(error, turn_snapshots_refusal(2));
            }
        }
        assert_eq!(
            turn_snapshots_refusal(2),
            "qwen4: pipeline parallelism (topology.pp = 2) does not support session snapshots (HIPFIRE_QWEN4_TURN_SNAPSHOTS); drop the option or set topology.pp = 1"
        );
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::mesh_bindings::window_ticket;
    use hipfire_runtime::multi_gpu::executor::{
        checkpoint_all, finalize_all, restore_all, MeshCheckpointCallbacks, MeshCheckpointId,
    };

    type Store = WindowStore<u32, u8>;

    fn mesh() -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Pp, 3)]).unwrap()
    }

    fn key(mesh: &DeviceMesh, window: u64) -> MeshCheckpointKey {
        MeshCheckpointKey {
            mesh_epoch: mesh.epoch(),
            load_gen: 7,
            reset_gen: 0,
            request_gen: 1,
            window_gen: window,
        }
    }

    fn ticket(key: MeshCheckpointKey, with_head: bool, armed: bool) -> Qwen4PpWindow {
        Qwen4PpWindow {
            key,
            start: 100,
            rows: 5,
            bands: 3,
            with_head,
            armed,
        }
    }

    fn checkpoint(key: &MeshCheckpointKey, with_head: bool) -> MeshCheckpoint {
        MeshCheckpoint {
            per_owner: (0..3).map(|owner| window_ticket(key, owner)).collect(),
            head: with_head.then(|| window_ticket(key, 3)),
        }
    }

    fn open(store: &mut Store, mesh: &DeviceMesh, with_head: bool) -> Qwen4PpWindow {
        let generation = store.begin().unwrap();
        let key = key(mesh, generation);
        let window = ticket(key, with_head, true);
        store
            .install(
                window,
                checkpoint(&key, with_head),
                vec![1, 2, 3],
                with_head.then_some(9),
            )
            .unwrap();
        window
    }

    #[test]
    fn a_window_is_live_until_finalized_and_never_twice() {
        let mesh = mesh();
        let mut store = Store::new();
        let window = open(&mut store, &mesh, true);
        assert!(store.require(&window, &[WindowPhase::Open]).is_ok());
        assert!(store.require(&window, &[WindowPhase::Retained]).is_err());
        assert!(store.remove(&window));
        // A repeated finalize, restore or release finds nothing.
        assert!(store.require(&window, &[WindowPhase::Open]).is_err());
        assert!(!store.remove(&window));
    }

    #[test]
    fn a_second_window_cannot_open_over_an_open_one() {
        let mesh = mesh();
        let mut store = Store::new();
        let _first = open(&mut store, &mesh, false);
        assert!(store.begin().is_err());
    }

    #[test]
    fn superseding_a_retained_window_invalidates_its_handle() {
        let mesh = mesh();
        let mut store = Store::new();
        let first = open(&mut store, &mesh, true);
        store.retain(&first).unwrap();
        assert!(store.require(&first, &[WindowPhase::Open]).is_err());
        assert!(store
            .require(&first, &[WindowPhase::Open, WindowPhase::Retained])
            .is_ok());
        // Retaining twice is not a transition.
        assert!(store.retain(&first).is_err());
        let second = open(&mut store, &mesh, true);
        assert_ne!(first, second);
        assert!(store
            .require(&first, &[WindowPhase::Open, WindowPhase::Retained])
            .is_err());
        assert!(store.take(&first).is_none());
        assert!(store.require(&second, &[WindowPhase::Open]).is_ok());
    }

    #[test]
    fn handles_are_unique_per_window_and_reset_clears_them() {
        let mesh = mesh();
        let mut store = Store::new();
        let first = open(&mut store, &mesh, true);
        store.clear();
        assert!(store.require(&first, &[WindowPhase::Open]).is_err());
        let second = open(&mut store, &mesh, true);
        assert_ne!(first.key.window_gen, second.key.window_gen);
        assert!(store.require(&first, &[WindowPhase::Open]).is_err());
    }

    const GROWTH_REFUSAL: &str = "row capture cannot be resized while a window is live";

    /// The live window and its phase survive a guard call untouched.
    fn assert_live(store: &Store, window: &Qwen4PpWindow, phase: WindowPhase) {
        assert!(store.require(window, &[phase]).is_ok());
        assert_eq!(store.live.as_ref().map(|record| record.phase), Some(phase));
    }

    #[test]
    fn a_live_window_allows_row_capture_ensures_its_rings_already_cover() {
        let mesh = mesh();
        let mut store = Store::new();
        let window = open(&mut store, &mesh, true);
        for phase in [WindowPhase::Open, WindowPhase::Retained] {
            if phase == WindowPhase::Retained {
                store.retain(&window).unwrap();
            }
            // K7 (8 ring rows): an equal and every shallower request is a no-op.
            for rows in [8, 7, 5, 2] {
                assert_eq!(store.check_row_capture_resize([8, 8, 8], rows), Ok(()));
                assert_eq!(store.check_row_capture_resize([9, 8, 12], rows), Ok(()));
                assert_live(&store, &window, phase);
            }
        }
    }

    #[test]
    fn a_live_window_refuses_real_row_capture_growth_in_both_phases() {
        let mesh = mesh();
        let mut store = Store::new();
        let window = open(&mut store, &mesh, false);
        for phase in [WindowPhase::Open, WindowPhase::Retained] {
            if phase == WindowPhase::Retained {
                store.retain(&window).unwrap();
            }
            let refused = Err(GROWTH_REFUSAL.to_string());
            // Every band short, then one undersized band among enough ones
            // (at either end), then a zero-GDN band whose capacity metadata is 0.
            assert_eq!(store.check_row_capture_resize([3, 3, 3], 8), refused);
            assert_eq!(store.check_row_capture_resize([8, 8, 7], 8), refused);
            assert_eq!(store.check_row_capture_resize([7, 8, 8], 8), refused);
            assert_eq!(store.check_row_capture_resize([8, 8, 0], 8), refused);
            assert_eq!(store.check_row_capture_resize([0, 0, 0], 2), refused);
            assert_live(&store, &window, phase);
        }
    }

    #[test]
    fn a_block_of_fewer_than_two_rows_never_counts_as_growth() {
        let mesh = mesh();
        let mut store = Store::new();
        let window = open(&mut store, &mesh, true);
        for phase in [WindowPhase::Open, WindowPhase::Retained] {
            if phase == WindowPhase::Retained {
                store.retain(&window).unwrap();
            }
            for rows in [0, 1] {
                assert_eq!(store.check_row_capture_resize([0, 0, 0], rows), Ok(()));
                assert_live(&store, &window, phase);
            }
        }
    }

    #[test]
    fn row_capture_growth_is_allowed_once_the_window_is_gone() {
        let mesh = mesh();
        let mut store = Store::new();
        // No window ever opened: growth from empty rings is free.
        assert_eq!(store.check_row_capture_resize([0, 0, 0], 8), Ok(()));
        // A shallower K3 ring (4 rows) first, then K7 (8 rows) once released.
        let window = open(&mut store, &mesh, true);
        store.retain(&window).unwrap();
        assert!(store.check_row_capture_resize([4, 4, 4], 8).is_err());
        assert!(store.remove(&window));
        assert_eq!(store.check_row_capture_resize([4, 4, 4], 8), Ok(()));
        assert_eq!(store.check_row_capture_resize([8, 8, 0], 8), Ok(()));
        // The same after an open window is released rather than retained.
        let open_window = open(&mut store, &mesh, false);
        assert!(store.check_row_capture_resize([4, 4, 4], 8).is_err());
        assert!(store.remove(&open_window));
        assert_eq!(store.check_row_capture_resize([4, 4, 4], 8), Ok(()));
    }

    #[test]
    fn install_refuses_mismatched_tickets() {
        let mesh = mesh();
        let mut store = Store::new();
        let generation = store.begin().unwrap();
        let key = key(&mesh, generation);
        // Head promised, none captured.
        assert!(store
            .install(
                ticket(key, true, false),
                checkpoint(&key, true),
                vec![1, 2, 3],
                None
            )
            .is_err());
        // Wrong band count.
        assert!(store
            .install(
                ticket(key, false, false),
                checkpoint(&key, false),
                vec![1, 2],
                None
            )
            .is_err());
        // Tickets issued for another window of the mesh.
        let other = self::key(&mesh, generation + 1);
        assert!(store
            .install(
                ticket(key, false, false),
                checkpoint(&other, false),
                vec![1, 2, 3],
                None
            )
            .is_err());
        assert!(store.live.is_none());
    }

    #[test]
    fn arming_needs_two_rows_and_every_ring() {
        assert!(arm_decision([5, 5, 5], 5));
        assert!(arm_decision([8, 5, 9], 5));
        assert!(!arm_decision([8, 4, 9], 5));
        assert!(!arm_decision([8, 8], 1));
        assert!(!arm_decision([0, 0], 2));
    }

    #[test]
    fn only_a_covered_multi_row_verify_may_run_armed() {
        let ok = [(true, 5), (true, 6), (true, 5)];
        assert!(check_epoch_arming(&ok, true, 5).is_ok());
        assert!(check_epoch_arming(&ok, true, 6).is_err());
        assert!(check_epoch_arming(&ok, true, 1).is_err());
        assert!(check_epoch_arming(&ok, false, 5).is_err());
        // Bands disagreeing on arming.
        assert!(check_epoch_arming(&[(true, 5), (false, 5)], true, 5).is_err());
        // Unarmed forwards of any route and size.
        let off = [(false, 0), (false, 0)];
        assert!(check_epoch_arming(&off, false, 1536).is_ok());
        assert!(check_epoch_arming(&off, true, 1).is_ok());
    }

    #[test]
    fn repair_needs_an_armed_window_ring_cover_and_every_band_at_the_end() {
        let mesh = mesh();
        let armed = ticket(key(&mesh, 1), true, true);
        let at_end = [(5, 105), (5, 105), (6, 105)];
        assert!(check_repair_request(&armed, 2, 5, &at_end).is_ok());
        // A shorter (pruned) block of an armed window.
        let short = [(5, 103), (5, 103), (5, 103)];
        assert!(check_repair_request(&armed, 1, 3, &short).is_ok());
        assert!(check_repair_request(&armed, 0, 5, &at_end).is_err());
        assert!(check_repair_request(&armed, 5, 5, &at_end).is_err());
        assert!(check_repair_request(&armed, 2, 6, &at_end).is_err());
        assert!(check_repair_request(&armed, 2, 5, &[(5, 105), (4, 105), (5, 105)]).is_err());
        assert!(check_repair_request(&armed, 2, 5, &[(5, 105), (5, 104), (5, 105)]).is_err());
        assert!(check_repair_request(&armed, 2, 5, &at_end[..2]).is_err());
        let unarmed = ticket(key(&mesh, 1), true, false);
        assert!(check_repair_request(&unarmed, 2, 5, &at_end).is_err());
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Checkpoint(usize),
        Restore(usize),
        Validate(usize),
        Finalize(usize),
    }

    /// A fake owner set minting the adapter's own tickets and logging order.
    struct Fake {
        log: Vec<Event>,
        fail_validate: Option<usize>,
    }

    impl MeshCheckpointCallbacks for Fake {
        fn checkpoint_rank(
            &mut self,
            owner: usize,
            key: &MeshCheckpointKey,
        ) -> Result<MeshCheckpointId, String> {
            self.log.push(Event::Checkpoint(owner));
            Ok(window_ticket(key, owner))
        }
        fn restore_rank(
            &mut self,
            owner: usize,
            _: &MeshCheckpointKey,
            _: MeshCheckpointId,
        ) -> Result<(), String> {
            self.log.push(Event::Restore(owner));
            Ok(())
        }
        fn validate_finalize_rank(
            &mut self,
            owner: usize,
            _: &MeshCheckpointKey,
            _: MeshCheckpointId,
        ) -> Result<(), String> {
            self.log.push(Event::Validate(owner));
            if self.fail_validate == Some(owner) {
                return Err("superseded".to_string());
            }
            Ok(())
        }
        fn finalize_rank(&mut self, owner: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {
            self.log.push(Event::Finalize(owner));
        }
    }

    #[test]
    fn prepare_snapshots_bands_then_head_and_finalize_validates_all_first() {
        let mesh = mesh();
        let key = key(&mesh, 1);
        let mut fake = Fake {
            log: Vec::new(),
            fail_validate: None,
        };
        let aggregate = checkpoint_all(&mut fake, &key, 3, true).unwrap();
        assert_eq!(aggregate, checkpoint(&key, true));
        finalize_all(&mut fake, &key, aggregate).unwrap();
        use Event::*;
        assert_eq!(
            fake.log,
            [
                Checkpoint(0),
                Checkpoint(1),
                Checkpoint(2),
                Checkpoint(3),
                Validate(0),
                Validate(1),
                Validate(2),
                Validate(3),
                Finalize(0),
                Finalize(1),
                Finalize(2),
                Finalize(3),
            ]
        );
    }

    #[test]
    fn a_rejected_finalize_finalizes_nothing_and_restores_every_owner() {
        let mesh = mesh();
        let key = key(&mesh, 1);
        let mut fake = Fake {
            log: Vec::new(),
            fail_validate: Some(1),
        };
        let aggregate = checkpoint(&key, true);
        assert!(finalize_all(&mut fake, &key, aggregate).is_err());
        use Event::*;
        assert_eq!(
            fake.log,
            [
                Validate(0),
                Validate(1),
                Validate(2),
                Validate(3),
                Restore(3),
                Restore(2),
                Restore(1),
                Restore(0),
            ]
        );
    }

    #[test]
    fn a_ticket_of_another_window_is_refused_before_any_callback() {
        let mesh = mesh();
        let stale = checkpoint(&key(&mesh, 1), true);
        let live = key(&mesh, 2);
        let mut fake = Fake {
            log: Vec::new(),
            fail_validate: None,
        };
        assert!(finalize_all(&mut fake, &live, stale.clone()).is_err());
        assert!(restore_all(&mut fake, &live, stale).is_err());
        assert!(fake.log.is_empty());
    }
}
