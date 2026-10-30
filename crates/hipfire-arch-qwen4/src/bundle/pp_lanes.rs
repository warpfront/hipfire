// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The exact AR lane executor over a pipeline-parallel mesh.
//!
//! Every band (peers and the output-band root) holds its own
//! [`Qwen4LaneStore`] scoped to its layers; admission, release, poison, forced
//! seeds and step bookkeeping are replicated on every band, so the bands'
//! lane tables, hosts and phases never diverge. The plan validated at
//! provision, the picks committed and the phase machine are those of
//! [`crate::lane_exec::Qwen4LaneExecutor`], whose host helpers are reused.
//!
//! Decode lanes traverse a micro-batched wavefront across the bands. Each
//! lane keeps its exact forward program; only independent lanes' inter-device
//! ordering changes. Pinned, disjoint first-row bounce slices carry the HC
//! streams. Sender completion events gate receiver uploads, and stream order
//! protects scratch reuse. The output band drains once before picks are read.
//! `HIPFIRE_PP_LANE_OVERLAP=0` selects the sequential decode walk instead.
//! The micro-batch count `G` per step comes from the output-band root's
//! [`crate::lane::groups`] planner: maximal overlap `min(lanes, 2 * stages)`
//! under a per-lane stage policy, measured per-`G` complete-step walls under a
//! row-sharing policy, or the clamped `HIPFIRE_PP_LANE_GROUPS` override.
//! The prompt-completing and ordinary prefill tiles then run as singleton PP
//! tiles through [`Qwen4Bundle::pp_lane_prefill_tile_detached`]. On any
//! failure every plan participant is poisoned on every band and the phase
//! returns to idle; a device or transport failure quarantines the mesh.
//!
//! Sampled lanes ([`crate::lane::LaneSampling`], set on every band's host copy
//! by [`Qwen4Bundle::pp_set_lane_sampling`]) draw on the output band only,
//! where the logits are: after the decode wavefront's final sync (both the
//! overlapped and the sequential walk) each sampled AR lane downloads its
//! logits row (one D2H) and its pick becomes one draw of the AR singleton's
//! sampler against the lane's own RNG; a sampled lane's prompt-completing
//! tile likewise draws from its final row. Greedy lanes and greedy steps are
//! unchanged (no extra D2H).

use super::*;
use crate::bundle::Qwen4Bundle;
use crate::gpu_forward_lanes::{
    forward_lanes, forward_lanes_edge, lane_stream_bf16, LaneOutputs, LaneRowKind, LaneRows,
};
use crate::lane::{download_lane_rows, lane_sample_step, LaneStatus, Qwen4LaneStore};
use crate::lane_exec::{
    commit_host, pick_at, validate_plan, view_of, wants_pick, LaneDraft, LanePhase, LaneRoute,
    PlanLimits, StepKey,
};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, StepOutput,
};
use std::time::Instant;

const PROMOTION_REFUSAL: &str = "qwen4 topology.pp: promotion into lanes is not supported yet";

fn no_mesh() -> String {
    "qwen4 topology.pp: no pipeline mesh is attached".to_string()
}

fn not_staged() -> String {
    "Qwen4 lanes are not staged".to_string()
}

/// Healthy mesh over the `gpus` it was attached with.
fn ready(mesh: &Qwen4PpMesh, gpus: &Gpus) -> Result<(), String> {
    mesh.require_ready(gpus).map_err(|e| e.to_string())
}

/// Every band in order: peers, then the output-band root.
fn bands_ref<'b>(
    root: &'b Qwen4Bundle,
    mesh: &'b Qwen4PpMesh,
) -> impl Iterator<Item = &'b Qwen4Bundle> {
    mesh.bands.iter().chain(std::iter::once(root))
}

/// Lanes staged (AR) on every band, none under expert parallelism.
fn stage_checks(root: &Qwen4Bundle, mesh: &Qwen4PpMesh) -> Result<(), String> {
    for band in bands_ref(root, mesh) {
        if band.weights.ep_placement().is_some() {
            return Err(
                "qwen4 topology.pp lanes: a band carries an expert-parallel placement".to_string(),
            );
        }
        let store = band.lanes.as_ref().ok_or_else(not_staged)?;
        if store.spec().is_some() {
            return Err(
                "qwen4 topology.pp: native-MTP lanes are not supported on pipeline bands"
                    .to_string(),
            );
        }
    }
    Ok(())
}

/// Drop one band's step: participants poisoned, window gone, phase idle.
fn abort_band(band: &mut Qwen4Bundle, plan: &BatchStepPlan) {
    if let Some(store) = band.lanes.as_mut() {
        for r in &plan.requests {
            store.poison(&r.epoch);
        }
        store.step_started = None;
        store.phase = LanePhase::Idle;
    }
}

/// A failure inside a forward: `fatal` once device work may have been
/// submitted or a copy failed (the mesh is then quarantined).
struct StepFailure {
    message: String,
    fatal: bool,
}

impl StepFailure {
    fn new(message: impl ToString, fatal: bool) -> Self {
        Self {
            message: message.to_string(),
            fatal,
        }
    }
}

/// The exact AR lane executor of a PP mesh: borrows the output-band root
/// bundle (which owns the mesh) and the caller's devices.
pub struct Qwen4PpLaneExecutor<'a> {
    pub(super) bundle: &'a mut Qwen4Bundle,
    pub(super) gpus: &'a mut Gpus,
}

impl Qwen4PpLaneExecutor<'_> {
    /// Run `f` with the mesh detached from the root (always put back).
    fn with_mesh<R>(
        &mut self,
        f: impl FnOnce(&mut Qwen4Bundle, &mut Qwen4PpMesh, &mut Gpus) -> Result<R, String>,
    ) -> Result<R, String> {
        let mut mesh = self.bundle.pp.take().ok_or_else(no_mesh)?;
        let result = f(&mut *self.bundle, &mut mesh, &mut *self.gpus);
        self.bundle.pp = Some(mesh);
        result
    }

    /// The singleton PP tile length the lane's next prefill step must use.
    pub fn exact_prefill_chunk_len(&self, epoch: &RequestEpoch) -> Result<usize, String> {
        let mesh = self.bundle.pp.as_ref().ok_or_else(no_mesh)?;
        ready(mesh, self.gpus)?;
        let lane = self
            .bundle
            .lanes
            .as_ref()
            .ok_or_else(not_staged)?
            .lane(epoch)
            .ok_or_else(|| format!("exact_prefill_chunk_len: unknown or stale epoch {epoch:?}"))?;
        if lane.host.status != LaneStatus::Prefilling {
            return Err(format!(
                "exact_prefill_chunk_len: lane {epoch:?} is {:?}, not Prefilling",
                lane.host.status
            ));
        }
        Ok(self
            .bundle
            .pp_lane_tile_len(mesh, &lane.host.prompt, lane.host.prefilled))
    }

    /// AR lanes only: every listed lane must be a Decoding lane with a
    /// pending seed (on the root's table); each answers `Ar`. A failed
    /// validation poisons every listed lane on every band.
    pub fn draft_step(
        &mut self,
        requests: &[(RequestEpoch, usize)],
    ) -> Result<Vec<LaneDraft>, String> {
        self.with_mesh(|root, mesh, gpus| {
            ready(mesh, gpus)?;
            stage_checks(root, mesh)?;
            for band in bands_ref(root, mesh) {
                let store = band.lanes.as_ref().ok_or_else(not_staged)?;
                store.phase.check_draft()?;
            }
            let drafted = (|| {
                let store = root.lanes.as_ref().ok_or_else(not_staged)?;
                requests
                    .iter()
                    .map(|(epoch, _)| {
                        let lane = store.lane(epoch).ok_or_else(|| {
                            format!("draft_step: unknown or stale epoch {epoch:?}")
                        })?;
                        if lane.host.status != LaneStatus::Decoding {
                            return Err(format!(
                                "draft_step: lane {epoch:?} is {:?}, not Decoding",
                                lane.host.status
                            ));
                        }
                        if lane.host.pending_seed.is_none() {
                            return Err(format!("draft_step: lane {epoch:?} has no pending seed"));
                        }
                        Ok(LaneDraft {
                            epoch: *epoch,
                            route: LaneRoute::Ar,
                            drafts: Vec::new(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })();
            match drafted {
                Ok(drafts) => {
                    for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                        if let Some(store) = band.lanes.as_mut() {
                            store.step_started = Some(Instant::now());
                        }
                    }
                    Ok(drafts)
                }
                Err(error) => {
                    for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                        if let Some(store) = band.lanes.as_mut() {
                            for (epoch, _) in requests {
                                store.poison(epoch);
                            }
                            store.step_started = None;
                        }
                    }
                    Err(error)
                }
            }
        })
    }

    /// Validate the plan against every band's lane table and map every
    /// band's QSA arenas through the step's end position. A mapping failure
    /// poisons every participant on every band and returns the step to idle.
    pub fn provision_step(&mut self, plan: &BatchStepPlan) -> Result<(), String> {
        self.with_mesh(|root, mesh, gpus| {
            ready(mesh, gpus)?;
            stage_checks(root, mesh)?;
            // The plan is validated once, on the output-band root; every
            // other band must then hold the same lanes in the same state.
            let ends = {
                let root_ref: &Qwen4Bundle = &*root;
                let mesh_ref: &Qwen4PpMesh = &*mesh;
                let store = root_ref.lanes.as_ref().ok_or_else(not_staged)?;
                store.phase.check_idle()?;
                let ends = validate_plan(
                    plan,
                    PlanLimits {
                        max_lanes: store.max_lanes(),
                        row_budget: store.row_budget(),
                    },
                    |epoch| store.lane(epoch).map(view_of),
                    |host| root_ref.pp_lane_tile_len(mesh_ref, &host.prompt, host.prefilled),
                )?;
                for band in &mesh_ref.bands {
                    let peer = band.lanes.as_ref().ok_or_else(not_staged)?;
                    peer.phase.check_idle()?;
                    for (r, &end) in plan.requests.iter().zip(&ends) {
                        let lead = store.lane(&r.epoch).ok_or_else(|| {
                            format!("provision_step: epoch {:?} vanished", r.epoch)
                        })?;
                        let lane = peer.lane(&r.epoch).ok_or_else(|| {
                            format!("provision_step: a band lacks epoch {:?}", r.epoch)
                        })?;
                        if lane.host.slot != lead.host.slot
                            || lane.host.status != lead.host.status
                            || lane.host.prefilled != lead.host.prefilled
                            || lane.host.pending_seed != lead.host.pending_seed
                            || lane.host.prompt.len() != lead.host.prompt.len()
                            || lane.state.position != lead.state.position
                            || lane.state.max_seq_len < end
                        {
                            return Err(format!(
                                "provision_step: a band's lane {:?} disagrees with the output band",
                                r.epoch
                            ));
                        }
                    }
                }
                ends
            };
            let stages = mesh.stages();
            let mapped = (|| -> Result<(), String> {
                for index in 0..stages {
                    let band = band_mut(&mut mesh.bands, &mut *root, index);
                    let gpu = &mut gpus.devices[index];
                    gpu.bind_thread()
                        .map_err(|e| format!("provision_step: bind device {index}: {e}"))?;
                    let store = band.lanes.as_mut().ok_or_else(not_staged)?;
                    for (r, &end) in plan.requests.iter().zip(&ends) {
                        let lane = store.lane_mut(&r.epoch).ok_or_else(|| {
                            format!("provision_step: epoch {:?} vanished", r.epoch)
                        })?;
                        lane.state.ensure_mapped_capacity(gpu, end).map_err(|e| {
                            format!(
                                "provision_step: band {index} maps {end} positions for {:?}: {e}",
                                r.epoch
                            )
                        })?;
                    }
                }
                Ok(())
            })();
            if let Err(error) = mapped {
                for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                    abort_band(band, plan);
                }
                return Err(error);
            }
            let key = StepKey::of(plan);
            for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                if let Some(store) = band.lanes.as_mut() {
                    store.step_started.get_or_insert_with(Instant::now);
                    store.phase = LanePhase::Provisioned(key.clone());
                }
            }
            Ok(())
        })
    }

    /// Per-request decode map-ahead on every band (best effort, as the
    /// singleton route: any refusal is skipped, `provision_step` grows on
    /// demand).
    pub fn map_ahead_decode(&mut self, epoch: &RequestEpoch, remaining_new_tokens: usize) {
        let _ = self.with_mesh(|root, mesh, gpus| {
            ready(mesh, gpus)?;
            stage_checks(root, mesh)?;
            let stages = mesh.stages();
            let mut mapped_to = 0usize;
            for index in 0..stages {
                let band = band_mut(&mut mesh.bands, &mut *root, index);
                let gpu = &mut gpus.devices[index];
                if gpu.bind_thread().is_err() {
                    return Ok(());
                }
                let Some(lane) = band
                    .lanes
                    .as_mut()
                    .and_then(|store| store.lane_mut(epoch))
                    .filter(|lane| lane.host.status == LaneStatus::Decoding)
                else {
                    return Ok(());
                };
                let Some(target) = crate::state::decode_map_ahead_tokens(
                    lane.state.position,
                    Some(remaining_new_tokens),
                    0,
                    lane.state.max_seq_len,
                ) else {
                    return Ok(());
                };
                match lane.state.ensure_mapped_capacity(gpu, target) {
                    Ok(()) => mapped_to = target,
                    Err(error) => {
                        eprintln!(
                            "  qwen4 pp band {index} QSA map-ahead through {target} tokens failed ({error}); growing on demand"
                        );
                        return Ok(());
                    }
                }
            }
            eprintln!(
                "  qwen4 pp QSA context mapped ahead for {mapped_to} tokens on {stages} bands (decode budget {remaining_new_tokens})"
            );
            Ok(())
        });
    }

    /// The step's decode rows band after band (one exact `forward_lanes` per
    /// band, edges through the pinned bounce), then each prefill tile as a
    /// singleton PP tile, and one pick per decode / prompt-completing row.
    /// On error every participant is poisoned on every band, the phase
    /// returns to idle, and a device/transport failure quarantines the mesh.
    pub fn forward_step(&mut self, plan: &BatchStepPlan) -> Result<StepOutput, String> {
        let key = StepKey::of(plan);
        self.with_mesh(|root, mesh, gpus| {
            ready(mesh, gpus)?;
            stage_checks(root, mesh)?;
            let step_id = root.lanes.as_ref().ok_or_else(not_staged)?.next_step_id;
            for band in bands_ref(root, mesh) {
                let store = band.lanes.as_ref().ok_or_else(not_staged)?;
                store.phase.check_forward(&key)?;
                if store.next_step_id != step_id {
                    return Err("forward_step: the bands disagree on the step id".to_string());
                }
            }
            for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                if let Some(store) = band.lanes.as_mut() {
                    store.phase = LanePhase::Forwarded(key.clone(), step_id);
                }
            }
            match forward_inner(root, mesh, gpus, plan) {
                Ok(target_picks) => {
                    for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                        if let Some(store) = band.lanes.as_mut() {
                            store.next_step_id += 1;
                            store.step_started = None;
                        }
                    }
                    Ok(StepOutput {
                        step_id,
                        target_picks,
                    })
                }
                Err(failure) => {
                    for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *root)) {
                        abort_band(band, plan);
                    }
                    if failure.fatal {
                        mesh.quarantine(failure.message.clone());
                    }
                    Err(format!("forward_step: {}", failure.message))
                }
            }
        })
    }

    /// Apply the forwarded step's lane bookkeeping on every band. The plan
    /// identity, step id, pick count and every participant of every band are
    /// validated before anything mutates.
    pub fn commit_step(
        &mut self,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        self.with_mesh(|root, mesh, gpus| {
            ready(mesh, gpus)?;
            stage_checks(root, mesh)?;
            let key = StepKey::of(plan);
            if output.target_picks.len() != plan.total_rows() {
                return Err("commit_step: pick count != planned rows".to_string());
            }
            for band in bands_ref(root, mesh) {
                let store = band.lanes.as_ref().ok_or_else(not_staged)?;
                store.phase.check_commit(&key, output.step_id)?;
                for r in &plan.requests {
                    let lane = match store.lane(&r.epoch) {
                        Some(lane) if lane.host.status != LaneStatus::Poisoned => lane,
                        _ => {
                            return Err(format!(
                                "commit_step: epoch {:?} no longer owns its lane",
                                r.epoch
                            ));
                        }
                    };
                    if r.rows.len == 0 || r.rows.end() > output.target_picks.len() {
                        return Err(format!("commit_step: bad row range {:?}", r.rows));
                    }
                    if matches!(
                        r.kind,
                        RequestStepKind::Verify { .. } | RequestStepKind::Forced
                    ) {
                        return Err(format!(
                            "commit_step: {:?} rows are not admitted on AR lanes",
                            r.kind
                        ));
                    }
                    if wants_pick(&lane.host, r)
                        && output.target_picks[r.rows.end() - 1] == u32::MAX
                    {
                        return Err("commit_step: head row without a target pick".to_string());
                    }
                }
            }
            let mut advances = Vec::with_capacity(plan.requests.len());
            let stages = mesh.stages();
            for index in 0..stages {
                let last = index + 1 == stages;
                let band = band_mut(&mut mesh.bands, &mut *root, index);
                let store = band.lanes.as_mut().ok_or_else(not_staged)?;
                for r in &plan.requests {
                    let pick = output.target_picks[r.rows.end() - 1];
                    let lane = store.lane_mut(&r.epoch).expect("validated above");
                    let committed = commit_host(&mut lane.host, r, pick);
                    if last {
                        let position = lane.state.position;
                        advances.push(RequestAdvance {
                            epoch: r.epoch,
                            committed_ids: committed.into_iter().collect(),
                            committed_position: position,
                            accepted_drafts: 0,
                            verified_rows: 0,
                            finish: (position >= lane.state.max_seq_len)
                                .then(|| "length".to_string()),
                        });
                    }
                }
                store.step_started = None;
                store.phase = LanePhase::Idle;
            }
            Ok(advances)
        })
    }

    /// Drop the step on every band: participants poisoned, windows dropped,
    /// phase idle. Host-only; a quarantined mesh, wrong-owner devices or a
    /// missing mesh refuse like every other lane call (no-op, no mutation).
    pub fn abort_step(&mut self, plan: &BatchStepPlan) {
        let Some(mut mesh) = self.bundle.pp.take() else {
            return;
        };
        if mesh.require_ready(self.gpus).is_ok() {
            for band in mesh.bands.iter_mut() {
                abort_band(band, plan);
            }
            abort_band(&mut *self.bundle, plan);
        }
        self.bundle.pp = Some(mesh);
    }
}

/// The forward of a provisioned plan. Decode rows first (stores detached
/// from the bands only for that part), then the prefill tiles with every
/// store attached.
fn forward_inner(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    plan: &BatchStepPlan,
) -> Result<Vec<u32>, StepFailure> {
    let refused = |m: &str| StepFailure::new(m, false);
    let vocab = root.lane_vocab();
    let stages = mesh.stages();
    let mut picks = vec![u32::MAX; plan.total_rows()];
    let mut decode: Vec<&RequestRows> = Vec::new();
    let mut n_prefill = 0usize;
    for r in &plan.requests {
        match r.kind {
            RequestStepKind::Ar => decode.push(r),
            RequestStepKind::Prefill => n_prefill += 1,
            other => {
                return Err(refused(&format!(
                    "{other:?} rows are not admitted on AR lanes"
                )));
            }
        }
    }
    let n_dec: usize = decode.iter().map(|r| r.rows.len).sum();
    let rows_cap = root
        .lanes
        .as_ref()
        .ok_or_else(|| refused("Qwen4 lanes are not staged"))?
        .prefill_logits_row();
    if n_dec > rows_cap {
        return Err(refused(&format!(
            "{n_dec} decode rows exceed the step buffers ({rows_cap})"
        )));
    }

    if n_dec > 0 {
        let hc = root.config.hc_count;
        let hidden = root.config.hidden_size;
        let epochs: Vec<RequestEpoch> = decode.iter().map(|r| r.epoch).collect();
        let Qwen4PpMesh {
            bands,
            lane_bounce,
            lane_events,
            ..
        } = &mut *mesh;
        let bounce = lane_bounce
            .as_mut()
            .ok_or_else(|| refused("the lane edge staging is not allocated"))?
            .as_bytes_mut();
        let mut refs: Vec<&mut Qwen4Bundle> = bands
            .iter_mut()
            .chain(std::iter::once(&mut *root))
            .collect();
        // One edge format for the step: both devices of every edge must
        // agree on the HC storage of an AR lane step.
        let bf16 = lane_stream_bf16(&gpus.devices[0]);
        if gpus.devices.iter().any(|gpu| lane_stream_bf16(gpu) != bf16) {
            return Err(refused("the bands disagree on the decode stream dtype"));
        }
        let bytes = super::lane_edge_bytes(n_dec, hc, hidden, bf16)
            .ok_or_else(|| refused("the lane edge byte count overflows"))?;
        if bytes > bounce.len() {
            return Err(refused(&format!(
                "a decode edge of {bytes} bytes exceeds the staged bounce ({})",
                bounce.len()
            )));
        }
        for (index, band) in refs.iter().enumerate() {
            let streams = band.lane_edge_streams().map_err(|e| refused(&e))?;
            if bytes > streams.buf.size() {
                return Err(refused(&format!(
                    "band {index} streams scratch cannot carry a {bytes}-byte decode edge"
                )));
            }
        }
        // Detach every band's store for the decode; restored on every path.
        let mut stores: Vec<Qwen4LaneStore> = Vec::with_capacity(stages);
        for band in refs.iter_mut() {
            match band.lanes.take() {
                Some(store) => stores.push(store),
                None => {
                    for (band, store) in refs.iter_mut().zip(stores) {
                        band.lanes = Some(store);
                    }
                    return Err(refused("Qwen4 lanes are not staged"));
                }
            }
        }
        let call = DecodeCall {
            plan,
            decode: &decode,
            epochs: &epochs,
            n_dec,
            vocab,
            bytes,
        };
        let result = decode_run(
            &mut refs,
            &mut stores,
            gpus,
            bounce,
            lane_events,
            &call,
            &mut picks,
        );
        for (band, store) in refs.iter_mut().zip(stores) {
            band.lanes = Some(store);
        }
        result?;
    }

    // Prefill tiles on the singleton PP route, every store attached.
    let mut final_tiles: Vec<(usize, usize, RequestEpoch)> = Vec::new();
    for (ordinal, r) in plan
        .requests
        .iter()
        .filter(|r| r.kind == RequestStepKind::Prefill)
        .enumerate()
    {
        let tokens = &plan.batch.tokens[r.rows.begin..r.rows.end()];
        let completes = {
            let lane = root
                .lanes
                .as_ref()
                .and_then(|store| store.lane(&r.epoch))
                .ok_or_else(|| refused("prefill lane vanished"))?;
            wants_pick(&lane.host, r)
        };
        let outputs = if completes {
            let store = root
                .lanes
                .as_mut()
                .ok_or_else(|| refused("Qwen4 lanes are not staged"))?;
            let parts = store.step_parts(&[]).map_err(|e| refused(&e))?;
            let used = rows_cap + ordinal;
            Some((
                parts.logits.sub_offset(used * vocab, vocab),
                parts.top1.sub_offset(
                    used * std::mem::size_of::<i32>(),
                    std::mem::size_of::<i32>(),
                ),
            ))
        } else {
            None
        };
        let result = root.pp_lane_prefill_tile_detached(
            mesh,
            gpus,
            &r.epoch,
            tokens,
            outputs.as_ref().map(|(logits, _)| logits),
            outputs.as_ref().map(|(_, top1)| top1),
        );
        // A fatal tile failure already quarantined the mesh itself.
        result.map_err(|e| StepFailure::new(e, false))?;
        if completes {
            final_tiles.push((ordinal, r.rows.end() - 1, r.epoch));
        }
    }
    if !final_tiles.is_empty() {
        let output = stages - 1;
        let gpu = &mut gpus.devices[output];
        let fetched = (|| -> Result<(), String> {
            gpu.bind_thread().map_err(|e| e.to_string())?;
            let store = root.lanes.as_mut().ok_or_else(not_staged)?;
            {
                let parts = store.step_parts(&[])?;
                let used = (rows_cap + n_prefill) * std::mem::size_of::<i32>();
                if used > parts.top1_host.len() {
                    return Err("the prompt-completing rows exceed the host mirror".to_string());
                }
                gpu.hip
                    .memcpy_dtoh(&mut parts.top1_host[..used], &parts.top1.buf)
                    .map_err(|e| e.to_string())?;
                gpu.take_device_wait_error().map_err(|e| e.to_string())?;
                for &(ordinal, row, _) in &final_tiles {
                    picks[row] = pick_at(parts.top1_host, rows_cap + ordinal);
                }
            }
            // A sampled lane's prompt-completing pick is its first draw, from
            // the tile's final logits row on the output band (one D2H per
            // sampled lane); greedy lanes keep the argmax.
            for &(ordinal, row, epoch) in &final_tiles {
                if !store
                    .lane(&epoch)
                    .is_some_and(|lane| lane.host.sampling.is_some())
                {
                    continue;
                }
                let crate::lane::StepParts {
                    lanes,
                    logits,
                    sample_host,
                    ..
                } = store.step_parts(&[epoch])?;
                let lane = lanes
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("sampled lane {epoch:?} vanished"))?;
                let mut one = [picks[row]];
                let gpu_ref: &Gpu = gpu;
                lane_sample_step(
                    &mut lane.host,
                    None,
                    &mut one,
                    0,
                    vocab,
                    sample_host,
                    |scratch| {
                        download_lane_rows(gpu_ref, logits, rows_cap + ordinal, 1, vocab, scratch)
                    },
                )
                .map_err(|e| format!("lane {epoch:?}: {e}"))?;
                picks[row] = one[0];
            }
            Ok(())
        })();
        fetched.map_err(|e| StepFailure::new(e, true))?;
    }
    Ok(picks)
}

/// Shared inputs of the decode walk.
struct DecodeCall<'p> {
    plan: &'p BatchStepPlan,
    decode: &'p [&'p RequestRows],
    epochs: &'p [RequestEpoch],
    n_dec: usize,
    vocab: usize,
    /// Live edge payload bytes (`n_dec` rows of HC streams).
    bytes: usize,
}

/// The sampled lanes of a PP decode step, on the output band's store: each
/// sampled AR lane's pick becomes one draw of its sampler from its logits
/// row (one D2H per sampled lane, after the step's own sync), in place of the
/// device argmax. A step with no sampled lane returns before touching
/// anything, so a greedy step issues exactly the work it did before.
fn sample_decode_lanes(
    gpu: &Gpu,
    store: &mut Qwen4LaneStore,
    call: &DecodeCall<'_>,
    picks: &mut [u32],
) -> Result<(), StepFailure> {
    if !call.epochs.iter().any(|epoch| {
        store
            .lane(epoch)
            .is_some_and(|lane| lane.host.sampling.is_some())
    }) {
        return Ok(());
    }
    let crate::lane::StepParts {
        lanes,
        logits,
        sample_host,
        ..
    } = store
        .step_parts(call.epochs)
        .map_err(|e| StepFailure::new(e, true))?;
    let mut first = 0usize;
    for (lane, r) in lanes.into_iter().zip(call.decode) {
        if lane.host.sampling.is_some() {
            let mut one = [picks[r.rows.begin]];
            lane_sample_step(
                &mut lane.host,
                None,
                &mut one,
                0,
                call.vocab,
                sample_host,
                |scratch| download_lane_rows(gpu, logits, first, 1, call.vocab, scratch),
            )
            .map_err(|e| StepFailure::new(format!("lane {:?}: {e}", r.epoch), true))?;
            picks[r.rows.begin] = one[0];
        }
        first += r.rows.len;
    }
    Ok(())
}

/// The sequential kill-switch route; errors after the first launch are fatal.
fn decode_run_sequential(
    refs: &mut [&mut Qwen4Bundle],
    stores: &mut [Qwen4LaneStore],
    gpus: &mut Gpus,
    bounce: &mut [u8],
    call: &DecodeCall<'_>,
    picks: &mut [u32],
) -> Result<(), StepFailure> {
    let DecodeCall {
        plan,
        decode,
        epochs,
        n_dec,
        vocab,
        bytes,
    } = *call;
    let stages = refs.len();
    let int = std::mem::size_of::<i32>();
    let mut launched = false;
    for index in 0..stages {
        let last = index + 1 == stages;
        let gpu = &mut gpus.devices[index];
        gpu.bind_thread()
            .map_err(|e| StepFailure::new(format!("bind device {index}: {e}"), launched))?;
        let band: &mut Qwen4Bundle = &mut *refs[index];
        if index > 0 {
            // Receiver: host -> device, then drain before the launch.
            launched = true;
            let streams = band
                .lane_edge_streams()
                .map_err(|e| StepFailure::new(e, true))?;
            gpu.hip
                .memcpy_htod(&streams.buf, &bounce[..bytes])
                .map_err(|e| StepFailure::new(format!("band {index} edge upload: {e}"), true))?;
            gpu.hip
                .device_synchronize()
                .map_err(|e| StepFailure::new(format!("band {index} edge drain: {e}"), true))?;
        }
        let policy = stores[index].stage_policy();
        if last {
            let parts = stores[index]
                .step_parts(epochs)
                .map_err(|e| StepFailure::new(e, launched))?;
            let crate::lane::StepParts {
                lanes,
                logits,
                top1,
                top1_host,
                ..
            } = parts;
            let mut rows: Vec<LaneRows<'_>> = lanes
                .into_iter()
                .zip(decode.iter())
                .map(|(lane, r)| LaneRows {
                    state: &mut lane.state,
                    tokens: &plan.batch.tokens[r.rows.begin..r.rows.end()],
                    kind: LaneRowKind::Ar,
                })
                .collect();
            let out = LaneOutputs {
                logits: &logits.sub_offset(0, n_dec * vocab),
                top1: &top1.sub_offset(0, n_dec * int),
                wide_hidden: None,
            };
            launched = true;
            let forwarded = forward_lanes(gpu, band, &mut rows, &out, &policy);
            drop(rows);
            forwarded.map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
            if top1_host.len() < n_dec * int {
                return Err(StepFailure::new("the host top-1 mirror is too small", true));
            }
            gpu.hip
                .device_synchronize()
                .map_err(|e| StepFailure::new(format!("band {index} drain: {e}"), true))?;
            gpu.hip
                .memcpy_dtoh(&mut top1_host[..n_dec * int], &top1.buf)
                .map_err(|e| StepFailure::new(format!("top-1 download: {e}"), true))?;
            gpu.take_device_wait_error()
                .map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
            let mut first = 0usize;
            for r in decode {
                for j in 0..r.rows.len {
                    picks[r.rows.begin + j] = pick_at(top1_host, first + j);
                }
                first += r.rows.len;
            }
            sample_decode_lanes(&*gpu, &mut stores[index], call, picks)?;
        } else {
            let lanes = stores[index]
                .band_lanes_mut(epochs)
                .map_err(|e| StepFailure::new(e, launched))?;
            let mut rows: Vec<LaneRows<'_>> = lanes
                .into_iter()
                .zip(decode.iter())
                .map(|(lane, r)| LaneRows {
                    state: &mut lane.state,
                    tokens: &plan.batch.tokens[r.rows.begin..r.rows.end()],
                    kind: LaneRowKind::Ar,
                })
                .collect();
            launched = true;
            let forwarded = forward_lanes_edge(gpu, band, &mut rows, &policy);
            drop(rows);
            forwarded.map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
            // Sender: drain, then device -> host.
            gpu.hip
                .device_synchronize()
                .map_err(|e| StepFailure::new(format!("band {index} drain: {e}"), true))?;
            gpu.take_device_wait_error()
                .map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
            let streams = band
                .lane_edge_streams()
                .map_err(|e| StepFailure::new(e, true))?;
            gpu.hip
                .memcpy_dtoh(&mut bounce[..bytes], &streams.buf)
                .map_err(|e| StepFailure::new(format!("band {index} edge download: {e}"), true))?;
        }
    }
    Ok(())
}

/// A contiguous set of whole lanes and its packed decode-row range.
#[derive(Debug, PartialEq, Eq)]
struct DecodeMicroBatch {
    lane_begin: usize,
    lane_end: usize,
    first_row: usize,
    rows: usize,
}

impl DecodeMicroBatch {
    fn bounce_range(&self, row_bytes: usize) -> std::ops::Range<usize> {
        self.first_row * row_bytes..(self.first_row + self.rows) * row_bytes
    }
}

/// Greedily approach the remaining average into `groups` contiguous groups,
/// reserving one whole lane for each later group. Equal-sized lanes differ by
/// at most one lane per group. `groups` is clamped to `1..=lanes` (the
/// callers pass [`crate::lane::groups`] selections, already within
/// `min(lanes, 2 * stages)`); no lanes give no groups.
fn decode_micro_batches(
    rows: impl ExactSizeIterator<Item = usize> + Clone,
    groups: usize,
) -> Vec<DecodeMicroBatch> {
    let n = rows.len();
    let groups = if n == 0 { 0 } else { groups.clamp(1, n) };
    let mut remaining: usize = rows.clone().sum();
    let mut rows = rows.peekable();
    let mut batches = Vec::with_capacity(groups);
    let mut lane = 0;
    let mut first_row = 0;
    for group in 0..groups {
        let groups_left = groups - group;
        let target = remaining.div_ceil(groups_left);
        let lane_begin = lane;
        let mut count = rows.next().expect("one lane reserved per group");
        lane += 1;
        while lane < n - (groups_left - 1) {
            let next = *rows.peek().expect("unassigned lane");
            if groups_left > 1 && (count + next).abs_diff(target) > count.abs_diff(target) {
                break;
            }
            count += rows.next().expect("peeked lane");
            lane += 1;
        }
        batches.push(DecodeMicroBatch {
            lane_begin,
            lane_end: lane,
            first_row,
            rows: count,
        });
        first_row += count;
        remaining -= count;
    }
    batches
}

/// Diagonal order: predecessor band first, increasing groups on every band.
fn decode_wavefront(stages: usize, groups: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..groups.saturating_add(stages).saturating_sub(1)).flat_map(move |tick| {
        (0..stages).filter_map(move |band| {
            tick.checked_sub(band)
                .filter(|&group| group < groups)
                .map(|group| (band, group))
        })
    })
}

/// `HIPFIRE_PP_LANE_EDGE_HOST_WAIT=1` restores the host-side
/// `hipEventSynchronize` on every sender edge event (the receiver upload is
/// then enqueued only after the sender's group completes). Any other value or
/// unset: the receiver's stream waits on the sender event on the device.
fn edge_host_wait_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

fn decode_overlap_enabled(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Exact decode wavefront, default on; `HIPFIRE_PP_LANE_OVERLAP=0` restores
/// the sequential walk at this single dispatch site. No host bookkeeping
/// observes another band's partial progress: agreement is checked only after
/// all micro-batches finish. Fatal failures drain every band before returning
/// to the caller, which poisons participants and quarantines the mesh.
#[allow(clippy::too_many_arguments)]
fn decode_run(
    refs: &mut [&mut Qwen4Bundle],
    stores: &mut [Qwen4LaneStore],
    gpus: &mut Gpus,
    bounce: &mut [u8],
    events: &[Vec<hip_bridge::Event>],
    call: &DecodeCall<'_>,
    picks: &mut [u32],
) -> Result<(), StepFailure> {
    let overlap = hipfire_config::developer_var("HIPFIRE_PP_LANE_OVERLAP").ok();
    let result = if decode_overlap_enabled(overlap.as_deref()) {
        decode_run_overlap(refs, stores, gpus, bounce, events, call, picks)
    } else {
        decode_run_sequential(refs, stores, gpus, bounce, call, picks)
    };
    if let Err(mut error) = result {
        if error.fatal {
            // Keep trying every band even if one bind or drain fails. The
            // quarantined mesh retains its pinned storage until safe teardown.
            for (index, gpu) in gpus.devices.iter_mut().take(refs.len()).enumerate() {
                if let Err(e) = gpu.bind_thread().and_then(|_| gpu.hip.device_synchronize()) {
                    error
                        .message
                        .push_str(&format!("; band {index} failure drain: {e}"));
                }
                if let Err(e) = gpu.take_device_wait_error() {
                    error.message.push_str(&format!("; band {index}: {e}"));
                }
            }
        }
        return Err(error);
    }
    // All forwards publish only their passed lanes' host state. No later
    // band's state is read until the complete walk has succeeded.
    check_decode_positions(stores, call.epochs)
}

fn check_decode_positions(
    stores: &[Qwen4LaneStore],
    epochs: &[RequestEpoch],
) -> Result<(), StepFailure> {
    // Every band must have advanced every lane to the same position.
    for epoch in epochs {
        let mut at: Option<usize> = None;
        for store in stores {
            let position = store
                .lane(epoch)
                .map(|lane| lane.state.position)
                .ok_or_else(|| StepFailure::new("a decode lane vanished from a band", true))?;
            match at {
                Some(first) if first != position => {
                    return Err(StepFailure::new(
                        format!("the bands disagree on {epoch:?}'s position after the step"),
                        true,
                    ));
                }
                _ => at = Some(position),
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_run_overlap(
    refs: &mut [&mut Qwen4Bundle],
    stores: &mut [Qwen4LaneStore],
    gpus: &mut Gpus,
    bounce: &mut [u8],
    events: &[Vec<hip_bridge::Event>],
    call: &DecodeCall<'_>,
    picks: &mut [u32],
) -> Result<(), StepFailure> {
    let stages = refs.len();
    let started = Instant::now();
    let lane_rows = || call.decode.iter().map(|r| r.rows.len);
    let groups = stores[stages - 1].select_decode_groups(stages, lane_rows());
    let batches = decode_micro_batches(lane_rows(), groups);
    if events.len() < stages - 1
        || events
            .iter()
            .take(stages - 1)
            .any(|band| band.len() < batches.len())
    {
        return Err(StepFailure::new(
            "the decode completion events are not staged",
            false,
        ));
    }
    // Install the stream before any work can be submitted. Both forwards
    // and their boundary copies then use the same per-device stream.
    for gpu in gpus.devices.iter_mut().take(stages) {
        gpu.ensure_capture_stream()
            .map_err(|e| StepFailure::new(e.to_string(), false))?;
    }
    let int = std::mem::size_of::<i32>();
    let row_bytes = call.bytes / call.n_dec;
    let mut launched = false;
    let host_edge_wait = edge_host_wait_enabled(
        hipfire_config::developer_var("HIPFIRE_PP_LANE_EDGE_HOST_WAIT")
            .ok()
            .as_deref(),
    );
    for (index, group) in decode_wavefront(stages, batches.len()) {
        let batch = &batches[group];
        let edge = batch.bounce_range(row_bytes);
        if index > 0 && host_edge_wait {
            let sender = &mut gpus.devices[index - 1];
            sender
                .bind_thread()
                .map_err(|e| StepFailure::new(e.to_string(), launched))?;
            sender
                .hip
                .event_synchronize(&events[index - 1][group])
                .map_err(|e| {
                    StepFailure::new(format!("band {} edge wait: {e}", index - 1), true)
                })?;
        }
        let gpu = &mut gpus.devices[index];
        gpu.bind_thread()
            .map_err(|e| StepFailure::new(e.to_string(), launched))?;
        let band = &mut *refs[index];
        if index > 0 {
            let streams = band
                .lane_edge_streams()
                .map_err(|e| StepFailure::new(e, launched))?;
            launched = true;
            if !host_edge_wait {
                // Device-side edge fence: this band's stream holds the upload
                // (and every later launch) until the sender's group event
                // fires, so the single host thread never blocks on a sender
                // and can enqueue the next group's launches immediately.
                gpu.hip
                    .stream_wait_event_on(gpu.active_stream.as_ref(), &events[index - 1][group])
                    .map_err(|e| {
                        StepFailure::new(format!("band {index} edge fence wait: {e}"), true)
                    })?;
            }
            gpu.hip
                .memcpy_htod_async(
                    &streams.buf,
                    &bounce[edge.clone()],
                    gpu.active_stream.as_ref().expect("installed stream"),
                )
                .map_err(|e| StepFailure::new(format!("band {index} edge upload: {e}"), true))?;
        }
        let epochs = &call.epochs[batch.lane_begin..batch.lane_end];
        let decode = &call.decode[batch.lane_begin..batch.lane_end];
        let policy = stores[index].stage_policy();
        if index + 1 == stages {
            let parts = stores[index]
                .step_parts(epochs)
                .map_err(|e| StepFailure::new(e, launched))?;
            let mut rows: Vec<LaneRows<'_>> = parts
                .lanes
                .into_iter()
                .zip(decode.iter())
                .map(|(lane, r)| LaneRows {
                    state: &mut lane.state,
                    tokens: &call.plan.batch.tokens[r.rows.begin..r.rows.end()],
                    kind: LaneRowKind::Ar,
                })
                .collect();
            let out = LaneOutputs {
                logits: &parts
                    .logits
                    .sub_offset(batch.first_row * call.vocab, batch.rows * call.vocab),
                top1: &parts
                    .top1
                    .sub_offset(batch.first_row * int, batch.rows * int),
                wide_hidden: None,
            };
            launched = true;
            forward_lanes(gpu, band, &mut rows, &out, &policy)
                .map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
        } else {
            let lanes = stores[index]
                .band_lanes_mut(epochs)
                .map_err(|e| StepFailure::new(e, launched))?;
            let mut rows: Vec<LaneRows<'_>> = lanes
                .into_iter()
                .zip(decode.iter())
                .map(|(lane, r)| LaneRows {
                    state: &mut lane.state,
                    tokens: &call.plan.batch.tokens[r.rows.begin..r.rows.end()],
                    kind: LaneRowKind::Ar,
                })
                .collect();
            launched = true;
            forward_lanes_edge(gpu, band, &mut rows, &policy)
                .map_err(|e| StepFailure::new(format!("band {index}: {e}"), true))?;
            let streams = band
                .lane_edge_streams()
                .map_err(|e| StepFailure::new(e, true))?;
            // The next forward on this band follows this download; the next
            // band waits only for this group's event, never for the device.
            gpu.hip
                .memcpy_dtoh_async(&mut bounce[edge], &streams.buf, gpu.active_stream.as_ref())
                .map_err(|e| StepFailure::new(format!("band {index} edge download: {e}"), true))?;
            gpu.hip
                .event_record(&events[index][group], gpu.active_stream.as_ref())
                .map_err(|e| {
                    StepFailure::new(format!("band {index} edge completion: {e}"), true)
                })?;
        }
    }
    let output = stages - 1;
    let gpu = &mut gpus.devices[output];
    gpu.bind_thread()
        .map_err(|e| StepFailure::new(e.to_string(), true))?;
    gpu.hip
        .device_synchronize()
        .map_err(|e| StepFailure::new(format!("output band drain: {e}"), true))?;
    let parts = stores[output]
        .step_parts(&[])
        .map_err(|e| StepFailure::new(e, true))?;
    let used = call.n_dec * int;
    if parts.top1_host.len() < used {
        return Err(StepFailure::new("the host top-1 mirror is too small", true));
    }
    gpu.hip
        .memcpy_dtoh(&mut parts.top1_host[..used], &parts.top1.buf)
        .map_err(|e| StepFailure::new(format!("top-1 download: {e}"), true))?;
    let mut wait_error = None;
    for (index, gpu) in gpus.devices.iter().take(stages).enumerate() {
        if let Err(e) = gpu.take_device_wait_error() {
            wait_error.get_or_insert_with(|| format!("band {index}: {e}"));
        }
    }
    if let Some(error) = wait_error {
        return Err(StepFailure::new(error, true));
    }
    // The complete step: every band's work drained by the output-band sync
    // above, the top-1 copied back, no device error.
    let wall_us = started.elapsed().as_secs_f64() * 1e6;
    let mut first = 0;
    for r in call.decode {
        for j in 0..r.rows.len {
            picks[r.rows.begin + j] = pick_at(parts.top1_host, first + j);
        }
        first += r.rows.len;
    }
    sample_decode_lanes(&gpus.devices[output], &mut stores[output], call, picks)?;
    stores[output].observe_decode_groups(stages, lane_rows(), batches.len(), wall_us);
    Ok(())
}

impl Qwen4Bundle {
    /// The exact AR lane executor over this PP mesh's staged lanes (see
    /// [`Self::pp_stage_lanes`]); refused unless a healthy mesh with lanes on
    /// every band is attached.
    pub fn pp_lane_executor<'a>(
        &'a mut self,
        gpus: &'a mut Gpus,
    ) -> Result<Qwen4PpLaneExecutor<'a>, BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.require_ready(gpus)?;
        if bands_ref(&*self, mesh).any(|band| band.lanes.is_none()) {
            return Err(pp_error("lanes are not staged on every pipeline band"));
        }
        Ok(Qwen4PpLaneExecutor { bundle: self, gpus })
    }

    /// Run `f` over every band with its device bound, in band order.
    fn pp_each_band(
        &mut self,
        gpus: &mut Gpus,
        mut f: impl FnMut(&mut Gpu, &mut Qwen4Bundle) -> Result<(), BundleError>,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = (|| -> Result<(), BundleError> {
            mesh.require_ready(gpus)?;
            let stages = mesh.stages();
            for index in 0..stages {
                let gpu = &mut gpus.devices[index];
                gpu.bind_thread().map_err(|e| pp_error(e.to_string()))?;
                f(gpu, band_mut(&mut mesh.bands, &mut *self, index))?;
            }
            Ok(())
        })();
        self.pp = Some(mesh);
        result
    }

    /// Admit `epoch` with its canonical `prompt` into a free lane on every
    /// band; returns the lane slot (identical on every band, else the
    /// admission is rolled back and refused).
    pub fn pp_admit_lane(
        &mut self,
        gpus: &mut Gpus,
        epoch: hipfire_runtime::slot_batch::RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, BundleError> {
        let mut admitted = 0usize;
        let mut prompt = Some(prompt);
        let mut agreed_slot = None;
        let result = self.pp_each_band(gpus, |gpu, band| {
            let owned_prompt = if band.weights.band_scope().output {
                prompt
                    .take()
                    .expect("the output band is visited once, last")
            } else {
                prompt
                    .as_ref()
                    .expect("peer bands precede the output")
                    .clone()
            };
            let slot = band.admit_lane(gpu, epoch, owned_prompt)?;
            admitted += 1;
            match agreed_slot {
                Some(first) if first != slot => {
                    return Err(pp_error("the bands admitted the lane into different slots"));
                }
                None => agreed_slot = Some(slot),
                _ => {}
            }
            Ok(())
        });
        match result {
            Ok(()) => Ok(agreed_slot.expect("a PP mesh has at least two bands")),
            Err(error) => {
                // Roll back the bands that admitted before the failure.
                let mut mesh = self.pp.take();
                if let Some(mesh) = mesh.as_mut() {
                    for band in mesh.bands.iter_mut().take(admitted) {
                        let _ = band.release_lane(&epoch);
                    }
                }
                if admitted > mesh.as_ref().map_or(0, |mesh| mesh.bands.len()) {
                    let _ = self.release_lane(&epoch);
                }
                self.pp = mesh;
                Err(error)
            }
        }
    }

    /// Free `epoch`'s lane on every band.
    pub fn pp_release_lane(
        &mut self,
        gpus: &mut Gpus,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        let mesh = self.pp.as_ref().expect("checked by pp_require_ready");
        let mut slot: Option<usize> = None;
        for band in bands_ref(&*self, mesh) {
            let lane = band
                .lanes
                .as_ref()
                .and_then(|store| store.lane(epoch))
                .ok_or_else(|| pp_error(format!("release: unknown or stale epoch {epoch:?}")))?;
            match slot {
                Some(first) if first != lane.host.slot => {
                    return Err(pp_error(
                        "release: the bands hold the lane in different slots",
                    ));
                }
                _ => slot = Some(lane.host.slot),
            }
        }
        self.pp_host_each_band(|band| band.release_lane(epoch))
    }

    /// Run a host-only lane operation on every band in order (no device
    /// binding, nothing allocated). The mesh is detached for the loop and
    /// restored on every path; callers validate everything beforehand.
    fn pp_host_each_band(
        &mut self,
        mut f: impl FnMut(&mut Qwen4Bundle) -> Result<(), BundleError>,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let mut result = Ok(());
        for band in mesh.bands.iter_mut() {
            result = f(band);
            if result.is_err() {
                break;
            }
        }
        if result.is_ok() {
            result = f(&mut *self);
        }
        self.pp = Some(mesh);
        result
    }

    /// [`Self::set_lane_stage_policy`] on every band's store (root included).
    /// Host only. Every band is checked first (staged, idle) so a refusal
    /// changes nothing.
    pub(crate) fn pp_set_lane_stage_policy(
        &mut self,
        gpus: &mut Gpus,
        policy: crate::gpu_forward_lanes::LaneStagePolicy,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        let mesh = self.pp.as_ref().expect("checked by pp_require_ready");
        for (index, band) in bands_ref(&*self, mesh).enumerate() {
            band.lane_stage_policy_ready()
                .map_err(|e| pp_error(format!("band {index}: {e}")))?;
        }
        self.pp_host_each_band(|band| {
            band.set_lane_stage_policy_local(policy)
                .map_err(BundleError::Forward)
        })
    }

    /// Mark `epoch`'s lane untrusted on every band. Host-only and infallible;
    /// a quarantined mesh, wrong-owner `gpus` or missing mesh refuses like
    /// every other lane call (no-op, no mutation).
    pub fn pp_poison_lane(
        &mut self,
        gpus: &mut Gpus,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) {
        if self.pp_require_ready(gpus).is_err() {
            return;
        }
        if let Some(mesh) = self.pp.as_mut() {
            for band in mesh.bands.iter_mut() {
                band.poison_lane(epoch);
            }
        }
        self.poison_lane(epoch);
    }

    /// Feed `token` as `epoch`'s next AR row on every band.
    pub fn pp_force_lane_seed(
        &mut self,
        gpus: &mut Gpus,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
        token: u32,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        let vocab = self.config.vocab_size;
        if token as usize >= vocab {
            return Err(pp_error("forced seed is outside the embedding vocabulary"));
        }
        let mesh = self.pp.as_ref().expect("checked by pp_require_ready");
        for band in bands_ref(&*self, mesh) {
            let lane = band
                .lanes
                .as_ref()
                .and_then(|store| store.lane(epoch))
                .ok_or_else(|| pp_error(format!("force_seed: unknown or stale epoch {epoch:?}")))?;
            if lane.host.status != LaneStatus::Decoding {
                return Err(pp_error(format!(
                    "force_seed: lane {epoch:?} is {:?}, not Decoding",
                    lane.host.status
                )));
            }
        }
        if let Err(error) = self.pp_host_each_band(|band| band.force_lane_seed(epoch, token)) {
            // A partial application cannot be trusted: poison the lane everywhere.
            self.pp_poison_lane(gpus, epoch);
            return Err(error);
        }
        Ok(())
    }

    /// [`Self::set_lanes_sampled`] on every band's store (root included).
    /// Host only; an unattached or unstaged band is skipped by the store
    /// call, a missing mesh is logged.
    pub(crate) fn pp_set_lanes_sampled(&mut self, on: bool) {
        let set = self.pp_host_each_band(|band| {
            band.set_lanes_sampled_local(on);
            Ok(())
        });
        if let Err(error) = set {
            eprintln!("  qwen4 topology.pp: set_lanes_sampled: {error}");
        }
    }

    /// Make `epoch`'s lane a sampled lane on every band (host only). Every
    /// band is checked first (staged sampled, idle, lane Prefilling with
    /// nothing prefilled), so a refusal changes nothing. Only the output band
    /// draws (its lane's RNG advances; the picks are produced where the
    /// logits are), but every band's host copy carries the sampler, so the
    /// bands' lane hosts never diverge.
    pub fn pp_set_lane_sampling(
        &mut self,
        gpus: &mut Gpus,
        epoch: &hipfire_runtime::slot_batch::RequestEpoch,
        sampling: crate::lane::LaneSampling,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        let mesh = self.pp.as_ref().expect("checked by pp_require_ready");
        for (index, band) in bands_ref(&*self, mesh).enumerate() {
            band.lanes
                .as_ref()
                .ok_or_else(|| pp_error(format!("band {index}: lanes are not staged")))?
                .check_lane_sampling(epoch, &sampling)
                .map_err(|e| pp_error(format!("band {index}: {e}")))?;
        }
        if let Err(error) = self.pp_host_each_band(|band| band.set_lane_sampling(epoch, sampling)) {
            // A partial application cannot be trusted: poison the lane everywhere.
            self.pp_poison_lane(gpus, epoch);
            return Err(error);
        }
        Ok(())
    }

    /// Promotion of a resident singleton into a PP lane is refused: the
    /// resident state is the output band's alone and cannot become per-band
    /// lane states.
    #[allow(clippy::too_many_arguments)]
    pub fn pp_prepare_promotion(
        &mut self,
        gpus: &mut Gpus,
        _epoch: hipfire_runtime::slot_batch::RequestEpoch,
        _prompt: Vec<u32>,
        _generated: Vec<u32>,
        _pending_seed: u32,
        _position: usize,
        _parts: Option<crate::mtp_spec::Qwen4PromotionParts<'_>>,
    ) -> Result<usize, BundleError> {
        self.pp_require_ready(gpus)?;
        Err(BundleError::Forward(PROMOTION_REFUSAL.to_string()))
    }

    /// See [`Self::pp_prepare_promotion`]: refused.
    pub fn pp_abort_promotion(
        &mut self,
        gpus: &mut Gpus,
        _epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        Err(BundleError::Forward(PROMOTION_REFUSAL.to_string()))
    }

    /// See [`Self::pp_prepare_promotion`]: refused.
    pub fn pp_commit_promotion(
        &mut self,
        gpus: &mut Gpus,
        _epoch: &hipfire_runtime::slot_batch::RequestEpoch,
    ) -> Result<(), BundleError> {
        self.pp_require_ready(gpus)?;
        Err(BundleError::Forward(PROMOTION_REFUSAL.to_string()))
    }
}

#[cfg(test)]
mod pp_lane_tests {
    use super::{
        decode_micro_batches, decode_overlap_enabled, decode_wavefront, edge_host_wait_enabled,
    };
    use crate::lane::groups::max_groups;

    #[test]
    fn pp_lane_partition_balance_contiguity_and_no_split() {
        for stages in 1..=5 {
            for lanes in 1..=32 {
                let rows = vec![1; lanes];
                let groups = max_groups(lanes, stages);
                let batches = decode_micro_batches(rows.iter().copied(), groups);
                assert_eq!(batches.len(), lanes.min(2 * stages));
                let min = batches.iter().map(|b| b.rows).min().unwrap();
                let max = batches.iter().map(|b| b.rows).max().unwrap();
                assert!(max - min <= 1);
                let mut lane = 0;
                let mut first = 0;
                for batch in batches {
                    assert_eq!(batch.lane_begin, lane);
                    assert!(batch.lane_end > lane);
                    assert_eq!(batch.first_row, first);
                    assert_eq!(
                        batch.rows,
                        rows[batch.lane_begin..batch.lane_end].iter().sum::<usize>()
                    );
                    lane = batch.lane_end;
                    first += batch.rows;
                }
                assert_eq!(lane, lanes);
                assert_eq!(first, lanes);
            }
        }
        assert!(decode_micro_batches([].into_iter(), 4).is_empty());
        // Balance by row count, not just lane count; a heavy lane stays whole.
        let rows = [3, 1, 1, 1, 2, 2, 1, 1];
        let batches = decode_micro_batches(rows.into_iter(), 4);
        assert_eq!(
            batches.iter().map(|b| b.rows).collect::<Vec<_>>(),
            [3, 3, 4, 2]
        );
        assert_eq!(
            batches.iter().map(|b| b.lane_end).collect::<Vec<_>>(),
            [1, 4, 6, 8]
        );
        let heavy = decode_micro_batches([1, 100, 1].into_iter(), 2);
        assert_eq!(heavy.iter().map(|b| b.lane_end).collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn pp_lane_partition_is_balanced_whole_lane_for_every_valid_group_count() {
        let shapes: [&[usize]; 2] = [&[2, 1, 4, 1, 1], &[3, 1, 1, 1, 2, 2, 1, 1, 5, 1, 1, 2]];
        for stages in 1..=5 {
            for lanes in 1..=32usize {
                let unit = vec![1; lanes];
                for groups in 1..=max_groups(lanes, stages) {
                    let batches = decode_micro_batches(unit.iter().copied(), groups);
                    assert_eq!(batches.len(), groups);
                    let sizes: Vec<usize> = batches.iter().map(|b| b.rows).collect();
                    assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
                    assert!(sizes.windows(2).all(|w| w[0] >= w[1]));
                    assert_eq!(sizes.iter().sum::<usize>(), lanes);
                }
            }
            for shape in shapes {
                for groups in 1..=max_groups(shape.len(), stages) {
                    let batches = decode_micro_batches(shape.iter().copied(), groups);
                    assert_eq!(batches.len(), groups);
                    let (mut lane, mut first) = (0, 0);
                    for batch in &batches {
                        assert_eq!(batch.lane_begin, lane);
                        assert!(batch.lane_end > batch.lane_begin);
                        assert_eq!(batch.first_row, first);
                        assert_eq!(
                            batch.rows,
                            shape[batch.lane_begin..batch.lane_end]
                                .iter()
                                .sum::<usize>()
                        );
                        lane = batch.lane_end;
                        first += batch.rows;
                    }
                    assert_eq!(lane, shape.len());
                    assert_eq!(first, shape.iter().sum::<usize>());
                }
            }
        }
        // One group holds every lane; one lane per group at the maximum.
        let one = decode_micro_batches([2, 1, 3].into_iter(), 1);
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].lane_begin, one[0].lane_end, one[0].rows), (0, 3, 6));
        let each = decode_micro_batches([2, 1, 3].into_iter(), 3);
        assert_eq!(each.iter().map(|b| b.rows).collect::<Vec<_>>(), [2, 1, 3]);
        // An out-of-range request clamps to the lanes (and to one group).
        assert_eq!(decode_micro_batches([1, 1].into_iter(), 9).len(), 2);
        assert_eq!(decode_micro_batches([1, 1].into_iter(), 0).len(), 1);
    }

    #[test]
    fn pp_lane_wavefront_orders_every_pair_once() {
        for stages in 1..=8 {
            for groups in 1..=16 {
                let schedule: Vec<_> = decode_wavefront(stages, groups).collect();
                assert_eq!(schedule.len(), stages * groups);
                let mut seen = vec![false; stages * groups];
                for (band, group) in schedule {
                    assert!(!seen[band * groups + group]);
                    if band > 0 {
                        assert!(seen[(band - 1) * groups + group]);
                    }
                    if group > 0 {
                        assert!(seen[band * groups + group - 1]);
                    }
                    seen[band * groups + group] = true;
                }
                assert!(seen.into_iter().all(|value| value));
            }
        }
        assert_eq!(
            decode_wavefront(2, 4).collect::<Vec<_>>(),
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 1),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
        assert!(decode_wavefront(2, 0).next().is_none());
        assert!(decode_wavefront(0, 2).next().is_none());
    }

    #[test]
    fn pp_lane_bounce_slices_are_disjoint_and_fit_row_capacity() {
        let rows = [1, 3, 2, 1, 5, 1, 2, 1];
        let rows_cap: usize = rows.iter().sum();
        for stages in 1..=8 {
            for element_bytes in [2, 4] {
                let row_bytes = 4 * 256 * element_bytes;
                let mut end = 0;
                let groups = max_groups(rows.len(), stages);
                for batch in decode_micro_batches(rows.into_iter(), groups) {
                    let range = batch.bounce_range(row_bytes);
                    assert_eq!(range.start, end);
                    assert_eq!(range.len(), batch.rows * row_bytes);
                    assert!(range.end <= rows_cap * 4 * 256 * 4);
                    end = range.end;
                }
                assert_eq!(end, rows_cap * row_bytes);
            }
        }
    }

    #[test]
    fn pp_lane_overlap_kill_switch_parse() {
        assert!(!decode_overlap_enabled(Some("0")));
        for value in [
            None,
            Some("1"),
            Some(""),
            Some("false"),
            Some(" 0"),
            Some("0 "),
        ] {
            assert!(decode_overlap_enabled(value));
        }
    }

    #[test]
    fn pp_lane_edge_host_wait_switch_parse() {
        assert!(edge_host_wait_enabled(Some("1")));
        for value in [
            None,
            Some("0"),
            Some(""),
            Some("true"),
            Some(" 1"),
            Some("1 "),
        ] {
            assert!(!edge_host_wait_enabled(value));
        }
    }
}
