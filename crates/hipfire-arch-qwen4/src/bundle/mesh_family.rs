// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4's engine-loop family contract ([`MeshFamily`]): what the engine
//! needs to run a Single load on a one-owner mesh. Declarations and adapters
//! only; the runner (`hipfire_runtime::mesh_family::MeshRunner`) plans,
//! submits, drains, recovers and resets.

use super::*;
use crate::gpu_forward::Qwen4MeshHead;
use crate::mesh_bindings::{Qwen4MeshBindings, Qwen4RankBand};
use crate::program::qwen4_step_graph_with_output;
use hipfire_dispatch::pipeline::step_graph::StepGraph;
use hipfire_runtime::mesh_family::{DecodeOut, EpochInput, LaneWindow, MeshFamily, MeshLanes};
use hipfire_runtime::multi_gpu::executor::{MeshStepBindings, Result};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange,
};
use hipfire_runtime::weight_manifest::{StateEntry, WeightEntry};

fn text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// The one band of a one-owner mesh: device 0.
fn device(gpus: &mut Gpus) -> Result<&mut Gpu> {
    match gpus.devices.as_mut_slice() {
        [gpu] => Ok(gpu),
        _ => Err("a Qwen4 Single load runs on a one-owner mesh".into()),
    }
}

impl Qwen4Bundle {
    /// The Single load's forward, taken out for a call that also borrows the
    /// bundle, and put back.
    fn with_forward<R>(
        &mut self,
        body: impl FnOnce(&mut Self, &mut Qwen4GpuForward) -> Result<R>,
    ) -> Result<R> {
        let mut forward = self
            .execution
            .take()
            .ok_or("Qwen4 forward resources are not attached")?;
        let result = body(self, &mut forward);
        self.execution = Some(forward);
        result
    }

    /// Move the resident request into lane `tag` ([`Self::prepare_promotion`]
    /// with `generated` ending in its pending seed, then the swap); its
    /// context is mapped for `remaining` decode rows. A `sampling` lane
    /// draws from that sampler's stream.
    fn promote(
        &mut self,
        gpu: &mut Gpu,
        tag: u64,
        prompt: &[u32],
        generated: Vec<u32>,
        remaining: usize,
        parts: Option<crate::mtp_spec::Qwen4PromotionParts<'_>>,
        sampling: Option<crate::lane::LaneSampling>,
    ) -> Result<()> {
        let epoch = lane_epoch(tag);
        let seed = *generated.last().ok_or("park: no pending seed")?;
        let position = prompt.len() + generated.len() - 1;
        if self.state.position != position {
            return Err(format!(
                "park: the resident state sits at {}, not after its {position} tokens",
                self.state.position
            ));
        }
        self.prepare_promotion(
            gpu,
            epoch,
            prompt.to_vec(),
            generated,
            seed,
            position,
            parts,
        )
        .map_err(text)?;
        if let Some(sampling) = sampling {
            if let Err(error) =
                self.with_lanes(|store, _| store.set_promoted_sampling(&epoch, sampling))
            {
                let _ = self.abort_promotion(&epoch);
                return Err(text(error));
            }
        }
        // The prompt's boundary snapshots stay published: the reset that
        // follows the swap drops the resident turn.
        self.session_publish_captures();
        // A lane takes its turn's resident snapshot along (an MTP lane's
        // holds its head too): its rows move into the lane, so a cancel can
        // rewind the lane to it.
        let mark = self.session.as_mut().and_then(|s| s.take_mark(prompt));
        if let Err(error) = self.commit_promotion(gpu, &epoch) {
            let _ = self.release_lane(&epoch);
            if let Some(mark) = mark {
                self.discard_lane_mark(mark);
            }
            return Err(format!("lane promotion: {error}"));
        }
        if let Some(mark) = mark {
            // Marks of lanes released elsewhere (poisoned steps) go first.
            let stale: Vec<u64> = self
                .lane_marks
                .iter()
                .map(|&(t, _)| t)
                .filter(|&t| self.lanes().and_then(|s| s.lane(&lane_epoch(t))).is_none())
                .collect();
            for t in stale {
                if let Some(old) = self.take_lane_mark(t) {
                    self.discard_lane_mark(old);
                }
            }
            self.lane_marks.push((tag, mark));
        }
        if remaining > 0 {
            self.lane_executor()
                .map_err(text)?
                .map_ahead_decode(gpu, &epoch, remaining);
        }
        Ok(())
    }

    fn take_lane_mark(&mut self, tag: u64) -> Option<LiveMark> {
        let at = self.lane_marks.iter().position(|(t, _)| *t == tag)?;
        Some(self.lane_marks.swap_remove(at).1)
    }

    fn discard_lane_mark(&mut self, mark: LiveMark) {
        if let Some(session) = self.session.as_mut() {
            session.discard_mark(mark);
        }
    }

    /// [`MeshLanes::step_windows`] with emitter-forced advances
    /// ([`crate::lane_exec::Qwen4LaneExecutor::draft_step_forced`]).
    fn windows(
        &mut self,
        gpu: &mut Gpu,
        rows: &[(u64, usize)],
        forced: &[(RequestEpoch, Vec<u32>)],
    ) -> Result<Vec<LaneWindow>> {
        let requests: Vec<(RequestEpoch, usize)> = rows
            .iter()
            .map(|&(tag, drafts)| (lane_epoch(tag), drafts))
            .collect();
        let drafted = self
            .lane_executor()
            .map_err(text)?
            .draft_step_forced(gpu, &requests, forced)?;
        let store = self.lanes().ok_or("Qwen4 lanes are not staged")?;
        let mut slotted = Vec::with_capacity(rows.len());
        for (epoch, _) in &requests {
            let slot = store
                .slot_of(epoch)
                .ok_or_else(|| format!("engine lane {} is not admitted", epoch.request_tag))?;
            let window = drafted
                .iter()
                .find(|window| window.epoch == *epoch)
                .ok_or("the draft step omitted a lane")?;
            if matches!(window.route, crate::lane_exec::LaneRoute::Batched { .. })
                && window.drafts.is_empty()
            {
                return Err("a batched lane window drafted no tokens".into());
            }
            slotted.push((slot, *epoch, window.drafts.as_slice()));
        }
        slotted.sort_unstable_by_key(|&(slot, ..)| slot);
        let mut plan = BatchStepPlan::default();
        plan.batch.m_per_slot = vec![0; store.max_lanes()];
        for &(slot, epoch, drafts) in &slotted {
            let lane = store
                .lane(&epoch)
                .ok_or_else(|| format!("engine lane {} disappeared", epoch.request_tag))?;
            let seed = store
                .root_seed(&epoch)
                .or(lane.host.pending_seed)
                .ok_or("a decoding lane has no pending seed")?;
            let begin = plan.batch.tokens.len();
            let len = 1 + drafts.len();
            plan.batch.tokens.push(seed);
            plan.batch.tokens.extend_from_slice(drafts);
            for j in 0..len {
                plan.batch
                    .positions
                    .push(i32::try_from(lane.state.position + j).map_err(text)?);
                plan.batch.row_slot.push(i32::try_from(slot).map_err(text)?);
            }
            plan.batch.m_per_slot[slot] = len;
            plan.requests.push(RequestRows {
                epoch,
                rows: RowRange { begin, len },
                kind: if drafts.is_empty() {
                    RequestStepKind::Ar
                } else {
                    RequestStepKind::Verify {
                        draft_len: drafts.len(),
                    }
                },
            });
            plan.decode_rows += len;
        }
        let advances = {
            let mut executor = self.lane_executor().map_err(text)?;
            let out = executor
                .provision_step(gpu, &plan)
                .and_then(|()| executor.forward_step(gpu, &plan))
                .and_then(|stepped| executor.commit_step(gpu, &plan, stepped));
            if out.is_err() {
                executor.abort_step(&plan);
            }
            out?
        };
        requests
            .iter()
            .map(|(epoch, _)| {
                advances
                    .iter()
                    .find(|advance| advance.epoch == *epoch && !advance.committed_ids.is_empty())
                    .map(|advance| LaneWindow {
                        committed: advance.committed_ids.clone(),
                        drafted: advance.verified_rows.saturating_sub(1),
                        accepted: advance.accepted_drafts,
                    })
                    .ok_or_else(|| format!("lane {} committed nothing", epoch.request_tag))
            })
            .collect()
    }
}

impl MeshFamily for Qwen4Bundle {
    fn step_graph(&self, head: bool, top1: bool) -> Result<StepGraph> {
        qwen4_step_graph_with_output(&self.config, MeshRoute::Prefill, head, top1)
    }

    fn manifest(&self) -> (&[WeightEntry], &[StateEntry]) {
        let manifest = Qwen4Bundle::manifest(self);
        (&manifest.weights, &manifest.state)
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn admit(&self, gpus: &Gpus) -> Result<()> {
        let [gpu] = gpus.devices.as_slice() else {
            return Err("a Qwen4 Single load runs on a one-owner mesh".into());
        };
        if self.slowlink_profile().is_some() {
            return Err("the S1 slow-link prompt route is not a mesh program".into());
        }
        let forward = self
            .execution
            .as_ref()
            .ok_or("Qwen4 forward resources are not attached")?;
        forward.mesh_admission(self, gpu).map_err(text)
    }

    fn position(&self) -> usize {
        self.state.position
    }

    fn begin_prompt(&mut self, gpus: &mut Gpus, prompt: &[u32], reused: usize) -> Result<()> {
        self.session_begin(device(gpus)?, prompt, reused, SessionRoute::Ar)
            .map_err(text)
    }

    /// The Single prompt schedule: with the NPU trunk planned, its pair
    /// groups and session cuts (`prefill_schedule_rows`); otherwise the next
    /// session boundary, one forward chunk at most.
    fn next_prompt_rows(&mut self, gpus: &mut Gpus, prompt: &[u32]) -> Result<usize> {
        let gpu = device(gpus)?;
        self.poll_npu_trunk(gpu).map_err(text)?;
        let position = self.state.position;
        let chunk = self
            .spec_chunk_rows()
            .ok_or("Qwen4 forward resources are not attached")?;
        let end = if self.npu_trunk_planned() {
            position + self.prefill_schedule_rows(position, prompt.len(), chunk)
        } else {
            self.session_next_boundary()
                .unwrap_or(prompt.len())
                .min(position + chunk)
        };
        self.set_ple_lookahead(&prompt[end.min(prompt.len())..]);
        Ok(end.saturating_sub(position))
    }

    fn prompt_rows_done(&mut self, gpus: &mut Gpus, prompt: &[u32], end: usize) -> Result<()> {
        let gpu = device(gpus)?;
        if self.session_next_boundary() == Some(end) {
            self.session_at_boundary(gpu, &prompt[..end])
                .map_err(text)?;
        }
        if end == prompt.len() {
            // The terminal fence of the ordinary prompt phase, as `prefill_final`.
            self.seal_gdn_streams(gpu).map_err(text)?;
        }
        Ok(())
    }

    /// An NPU fault latched the route off: the prompt reruns GPU-only from
    /// the re-planned session start (`npu_trunk::containment`).
    fn take_prompt_rerun(&mut self, prompt: &[u32]) -> Option<usize> {
        self.take_npu_trunk_fault()
            .then(|| self.rank_session_plan(prompt, SessionRoute::Ar, prompt.len()))
    }

    fn reserve_decode(&mut self, gpus: &mut Gpus, new_tokens: usize) {
        if let Ok(gpu) = device(gpus) {
            let position = self.state.position;
            Qwen4Bundle::map_ahead_for_decode(self, gpu, position, new_tokens, 0);
        }
    }

    /// The Single decode path's device argmax (`forward_token_or_argmax`).
    fn device_token_decode(&self) -> bool {
        true
    }

    /// The token the forward committed as the newest PLE history slot.
    fn decoded_token(&self) -> Result<u32> {
        Ok(self.state.ple_history.previous()[1])
    }

    fn prepare_epoch(&mut self, gpus: &mut Gpus, route: MeshRoute, rows: usize) -> Result<usize> {
        let gpu = device(gpus)?;
        let layout = self.with_forward(|bundle, forward| {
            let layout = forward
                .mesh_layout(bundle, gpu, route, rows)
                .map_err(text)?;
            let end = bundle.state.position + rows;
            forward
                .prepare_mesh_capacity(bundle, gpu, end, layout.route_rows)
                .map_err(text)?;
            Ok(layout)
        })?;
        self.mesh_epoch_layout = Some(layout);
        Ok(layout.route_rows)
    }

    fn with_bindings(
        &mut self,
        input: EpochInput<'_>,
        run: &mut dyn FnMut(&mut dyn MeshStepBindings) -> Result<()>,
    ) -> Result<()> {
        let layout = self.mesh_epoch_layout.take();
        self.with_forward(|bundle, forward| {
            let mut bindings = Qwen4MeshBindings::new(vec![Qwen4RankBand { bundle, forward }])?;
            bindings.set_tokens(input.tokens);
            bindings.set_edge_epochs(input.edge_epochs);
            if let Some(layout) = layout {
                bindings.set_rank_layout(0, layout)?;
            }
            bindings.set_device_token(input.device_token);
            bindings.set_head(input.head.map(|head| Qwen4MeshHead {
                logits: head.logits,
                top1: head.top1,
                rows: crate::gpu_forward::Qwen4OutputRows::Final,
            }));
            run(&mut bindings)
        })
    }

    /// The staged lanes of a Single load: AR lanes, or MTP lanes that run
    /// speculative windows ([`MeshLanes::step_windows`]).
    fn mesh_lanes(&mut self) -> Option<&mut dyn MeshLanes> {
        self.lanes()?;
        (self.lane_topology() == "single").then_some(self as &mut dyn MeshLanes)
    }
}

/// The lane of an engine request: its tag, admitted once.
fn lane_epoch(tag: u64) -> RequestEpoch {
    RequestEpoch {
        request_tag: tag,
        owner_generation: 1,
    }
}

/// Qwen4's multi-request decode epoch: the fn-lanes executor's mixed-row step,
/// AR rows on an AR store, native-MTP windows on an MTP store. A request
/// enters through promotion (the resident state, and on an MTP store the
/// drafter's state, swaps into the lane), the legacy promotion path.
impl MeshLanes for Qwen4Bundle {
    fn lane_capacity(&self) -> usize {
        self.lanes().map_or(0, |store| store.max_lanes())
    }

    fn lane_rows(&self) -> usize {
        self.lanes().map_or(0, |store| store.row_budget())
    }

    fn free_lanes(&self) -> usize {
        self.lanes()
            .map_or(0, |store| store.max_lanes() - store.epochs().len())
    }

    fn park(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        ran: &[u32],
        seed: u32,
        remaining: usize,
    ) -> Result<()> {
        // The lane's committed picks end with its pending seed.
        let mut generated = ran.to_vec();
        generated.push(seed);
        self.promote(device(gpus)?, tag, prompt, generated, remaining, None, None)
    }

    fn windows(&self) -> bool {
        self.lanes().is_some_and(|store| store.spec().is_some())
    }

    fn park_window(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        emitted: &[u32],
        drafter: &mut dyn hipfire_runtime::spec::Speculator,
        remaining: usize,
    ) -> Result<()> {
        let drafter = drafter
            .drafter_any_mut()
            .and_then(|any| any.downcast_mut::<crate::mtp_spec::Qwen4MtpDrafter>())
            .ok_or("an MTP lane takes a native Qwen4 MTP drafter")?;
        // A lane that owes head rows starts with its head at the target.
        if self.lanes().is_some_and(|store| store.head_lag_rows() > 0) {
            drafter.catch_up_head_for_promotion(device(gpus)?, self)?;
        }
        let sampling = drafter.naive_lane_sampling()?;
        let parts = drafter.promotion_parts()?;
        self.promote(
            device(gpus)?,
            tag,
            prompt,
            emitted.to_vec(),
            remaining,
            Some(parts),
            sampling,
        )
    }

    fn sampled_windows(&self) -> bool {
        self.windows() && self.lanes().is_some_and(|store| store.promotes_sampled())
    }

    /// The legacy fn-lanes window ([`crate::lane_exec::Qwen4LaneExecutor`]):
    /// the root routes and drafts every lane, then one verify step over all
    /// of them commits each lane's greedy accept.
    fn step_windows(&mut self, gpus: &mut Gpus, rows: &[(u64, usize)]) -> Result<Vec<LaneWindow>> {
        self.windows(device(gpus)?, rows, &[])
    }

    /// The reverse of the promotion's swap: the lane's request states (on an
    /// MTP lane also its head, draft policy and pending hidden, into the
    /// drafter) move into the resident slot, and the session keeps them as
    /// its live state.
    fn unpark(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        tokens: &[u32],
        drafter: Option<&mut dyn hipfire_runtime::spec::Speculator>,
    ) -> Result<()> {
        let gpu = device(gpus)?;
        let epoch = lane_epoch(tag);
        if self.session.is_none() {
            return Err("no session cache: a lane record cannot continue live".into());
        }
        let marked = self
            .lane_marks
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, mark)| mark.position());
        // A window lane stops with its last pick unrun, which the singleton's
        // terminal flush runs: a forced one-row window runs it here, so the
        // record continues at the same row a resident turn would.
        let (head, owed, at, seed) = {
            let store = self.lanes().ok_or("Qwen4 lanes are not staged")?;
            let lane = store
                .lane(&epoch)
                .ok_or_else(|| format!("lane record {tag} disappeared"))?;
            (
                lane.spec.as_ref().map(|spec| spec.head.position()),
                lane.spec
                    .as_ref()
                    .and_then(|spec| spec.lag.as_ref())
                    .map_or(0, crate::mesh_rank_copy::HeadLagStore::owed),
                lane.state.position,
                store.root_seed(&epoch).or(lane.host.pending_seed),
            )
        };
        // Every row a lane ran is a committed pick: a position inside
        // `tokens` holds exactly that prefix; past it, a window ran beyond
        // its stop, or a cancelled lane rewinds to its mark of `tokens` (the
        // mark restores the head too: nothing of the lane's head is kept).
        let rewind = at > tokens.len() && marked == Some(tokens.len());
        if at > tokens.len() && !rewind {
            return Err(format!(
                "lane record {tag} sits at {at}, past its {} tokens",
                tokens.len()
            ));
        }
        if !rewind {
            // A retired lane's head lags by exactly its owed rows; a lane
            // without owed-row storage cannot catch up and is refused before
            // the forced row, which it could not use.
            if head.is_some_and(|head| head + owed != at) {
                return Err(format!(
                    "lane record {tag}: the MTP head is at {head:?} with {owed} owed rows, the trunk at {at}"
                ));
            }
            // The forced row appends the owed rows first.
            if let Some(seed) = unrun_seed(head.is_some(), at, seed, tokens) {
                self.windows(gpu, &[(tag, 0)], &[(epoch, vec![seed])])?;
            }
            // A record that ran no forced row catches its head up here.
            self.with_lanes(|store, bundle| {
                let lane = store
                    .lane_mut(&epoch)
                    .ok_or_else(|| format!("lane record {tag} disappeared"))?;
                if lane.spec.is_some() {
                    crate::mesh_rank_copy::flush_owed_head_rows(bundle, gpu, lane)?;
                }
                Ok::<(), String>(())
            })
            .map_err(text)?;
        }
        let lane = self
            .lanes()
            .and_then(|store| store.lane(&epoch))
            .ok_or_else(|| format!("lane record {tag} disappeared"))?;
        let position = lane.state.position;
        let mtp = match &lane.spec {
            None => false,
            // The mark restores the head with the trunk.
            Some(_) if rewind => true,
            Some(spec) => {
                if spec.head.position() != position {
                    return Err(format!(
                        "lane record {tag}: the MTP head is at {}, the trunk at {position}",
                        spec.head.position()
                    ));
                }
                let drafter = drafter
                    .and_then(|d| d.drafter_any_mut())
                    .and_then(|any| any.downcast_mut::<crate::mtp_spec::Qwen4MtpDrafter>())
                    .ok_or("an MTP lane record continues with a native Qwen4 MTP drafter")?;
                drafter.adopt_pending_hidden(gpu, &spec.pending_hidden)?;
                true
            }
        };
        // As the promotion: the resident request moves sealed.
        self.seal_gdn_streams(gpu).map_err(text)?;
        self.with_lanes(|store, bundle| {
            let lane = store
                .lane_mut(&epoch)
                .ok_or_else(|| format!("lane record {tag} disappeared"))?;
            bundle.state.clamp_f16_valid();
            lane.state.clamp_f16_valid();
            bundle.state.swap_requests(&mut lane.state);
            if let (Some(spec), Some(mtp)) = (lane.spec.as_mut(), bundle.mtp.as_mut()) {
                std::mem::swap(&mut mtp.state, &mut spec.head);
                mtp.draft.set_request_state(spec.draft);
            }
            store.release(&epoch)
        })
        .map_err(text)?;
        // The kept state continues: request-local PLE work of an earlier
        // request must not publish into it (`session_begin`'s live rule).
        self.invalidate_ple_epoch().map_err(text)?;
        let route = if mtp {
            SessionRoute::Mtp
        } else {
            SessionRoute::Ar
        };
        match self.take_lane_mark(tag) {
            Some(mark) if rewind => {
                let mut session = self.session.take().ok_or("a lane mark without a session")?;
                let restored = session.restore_mark(gpu, self, mark, tokens);
                self.session = Some(session);
                restored?;
            }
            mark => {
                if let Some(mark) = mark {
                    self.discard_lane_mark(mark);
                }
                if let Some(session) = self.session.as_mut() {
                    session.adopt_live(&tokens[..position], route);
                }
            }
        }
        Ok(())
    }

    fn step_lanes(&mut self, gpus: &mut Gpus, rows: &[(u64, u32, bool)]) -> Result<Vec<DecodeOut>> {
        let gpu = device(gpus)?;
        let store = self.lanes().ok_or("Qwen4 lanes are not staged")?;
        if store.spec().is_some() {
            return Err("MTP lanes step in windows".into());
        }
        // The executor takes a plan's requests in stable slot order.
        let mut slotted = Vec::with_capacity(rows.len());
        for &(tag, token, _) in rows {
            let epoch = lane_epoch(tag);
            let slot = store
                .slot_of(&epoch)
                .ok_or_else(|| format!("engine lane {tag} is not admitted"))?;
            slotted.push((slot, epoch, token));
        }
        slotted.sort_unstable_by_key(|&(slot, ..)| slot);
        let mut plan = BatchStepPlan::default();
        plan.batch.m_per_slot = vec![0; store.max_lanes()];
        let mut forced = Vec::new();
        for (begin, &(slot, epoch, token)) in slotted.iter().enumerate() {
            let lane = store
                .lane(&epoch)
                .ok_or_else(|| format!("engine lane {} disappeared", epoch.request_tag))?;
            // A token other than the lane's last pick (a forced template
            // token, or the request's own sampled draw) becomes its pending
            // seed first.
            if store.root_seed(&epoch).or(lane.host.pending_seed) != Some(token) {
                forced.push((epoch, token));
            }
            let position = i32::try_from(lane.state.position).map_err(text)?;
            plan.batch.tokens.push(token);
            plan.batch.positions.push(position);
            plan.batch.row_slot.push(i32::try_from(slot).map_err(text)?);
            plan.batch.m_per_slot[slot] = 1;
            plan.requests.push(RequestRows {
                epoch,
                rows: RowRange { begin, len: 1 },
                kind: RequestStepKind::Ar,
            });
        }
        plan.decode_rows = rows.len();
        for (epoch, token) in forced {
            self.force_lane_seed(&epoch, token).map_err(text)?;
        }
        let stepped = {
            let mut executor = self.lane_executor().map_err(text)?;
            let out = executor
                .provision_step(gpu, &plan)
                .and_then(|()| executor.forward_step(gpu, &plan));
            if out.is_err() {
                executor.abort_step(&plan);
            }
            out?
        };
        // A sampling request's row is read before the commit (the step's
        // logits are valid until the next forward): decode row `r` of the
        // plan is logits row `r`.
        let vocab = self.config.vocab_size;
        let mut rows_out = Vec::with_capacity(rows.len());
        for &(tag, _, logits) in rows {
            if !logits {
                rows_out.push(None);
                continue;
            }
            let row = plan
                .requests
                .iter()
                .position(|request| request.epoch == lane_epoch(tag))
                .ok_or_else(|| format!("lane {tag} is not in its step"))?;
            let read = self
                .lanes()
                .and_then(|store| store.logits())
                .ok_or_else(|| "the lane store has no step logits".to_string())
                .and_then(|all| {
                    let view = all.sub_offset(row * vocab, vocab);
                    let mut bytes = vec![0u8; vocab * 4];
                    gpu.dtoh_stream_ordered(&mut bytes, &view.buf)
                        .map_err(text)?;
                    gpu.take_device_wait_error().map_err(text)?;
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect::<Vec<f32>>())
                });
            match read {
                Ok(row) => rows_out.push(Some(row)),
                Err(error) => {
                    if let Ok(mut executor) = self.lane_executor() {
                        executor.abort_step(&plan);
                    }
                    return Err(format!("lane {tag} logits: {error}"));
                }
            }
        }
        let advances = {
            let mut executor = self.lane_executor().map_err(text)?;
            let advances = executor.commit_step(gpu, &plan, stepped);
            if advances.is_err() {
                executor.abort_step(&plan);
            }
            advances?
        };
        rows.iter()
            .zip(rows_out)
            .map(|(&(tag, _, _), row)| match row {
                Some(row) => Ok(DecodeOut::Logits(row)),
                None => advances
                    .iter()
                    .find(|advance| advance.epoch == lane_epoch(tag))
                    .and_then(|advance| advance.committed_ids.last().copied())
                    .map(DecodeOut::Top1)
                    .ok_or_else(|| format!("lane {tag} committed no pick")),
            })
            .collect()
    }

    fn free_lane(&mut self, tag: u64) {
        if let Some(mark) = self.take_lane_mark(tag) {
            self.discard_lane_mark(mark);
        }
        if let Err(error) = self.release_lane(&lane_epoch(tag)) {
            eprintln!("[engine] lane {tag} release: {error}");
        }
    }

    fn rewind_point(&self, tag: u64) -> Option<usize> {
        self.lane_marks
            .iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, mark)| mark.position())
    }
}

/// The unrun last pick of a window lane record at `at` (its pending `seed`)
/// that holds `tokens`: `Some` only when exactly that one pick is left. An
/// AR lane, or a lane at or past its tokens, has none.
fn unrun_seed(window: bool, at: usize, seed: Option<u32>, tokens: &[u32]) -> Option<u32> {
    let last = *tokens.get(at)?;
    (window && at + 1 == tokens.len() && seed == Some(last)).then_some(last)
}

#[cfg(test)]
mod tests {
    use super::unrun_seed;

    #[test]
    fn only_a_window_lane_one_pick_short_runs_its_seed() {
        let tokens = [1, 2, 3];
        assert_eq!(unrun_seed(true, 2, Some(3), &tokens), Some(3));
        // At or past its tokens (HIPF-591 B1: indexed out of bounds).
        assert_eq!(unrun_seed(true, 3, Some(3), &tokens), None);
        assert_eq!(unrun_seed(true, 4, Some(3), &tokens), None);
        assert_eq!(unrun_seed(false, 3, Some(3), &tokens), None);
        // AR lane, another seed, or more than one pick short.
        assert_eq!(unrun_seed(false, 2, Some(3), &tokens), None);
        assert_eq!(unrun_seed(true, 2, Some(9), &tokens), None);
        assert_eq!(unrun_seed(true, 1, Some(2), &tokens), None);
    }
}
