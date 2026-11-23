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
use hipfire_runtime::mesh_family::{DecodeOut, EpochInput, MeshFamily, MeshLanes};
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

    /// The staged AR lanes of a Single load (MTP lanes verify drafts: not
    /// this epoch).
    fn mesh_lanes(&mut self) -> Option<&mut dyn MeshLanes> {
        let ar = self.lanes()?.spec().is_none();
        (ar && self.lane_topology() == "single").then_some(self as &mut dyn MeshLanes)
    }
}

/// The lane of an engine request: its tag, admitted once.
fn lane_epoch(tag: u64) -> RequestEpoch {
    RequestEpoch {
        request_tag: tag,
        owner_generation: 1,
    }
}

/// Qwen4's multi-request decode epoch: the fn-lanes executor's mixed-row step
/// with AR rows only. A request enters through promotion (the resident state
/// swaps into the lane), the path byte-identical to the singleton's decode.
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
        let gpu = device(gpus)?;
        let epoch = lane_epoch(tag);
        // The lane's committed picks end with its pending seed.
        let mut generated = ran.to_vec();
        generated.push(seed);
        let position = prompt.len() + ran.len();
        if self.state.position != position {
            return Err(format!(
                "park: the resident state sits at {}, not after its {position} tokens",
                self.state.position
            ));
        }
        self.prepare_promotion(gpu, epoch, prompt.to_vec(), generated, seed, position, None)
            .map_err(text)?;
        // The prompt's boundary snapshots stay published: the reset that
        // follows the swap drops the resident turn.
        self.session_publish_captures();
        if let Err(error) = self.commit_promotion(gpu, &epoch) {
            let _ = self.release_lane(&epoch);
            return Err(format!("lane promotion: {error}"));
        }
        if remaining > 0 {
            self.lane_executor()
                .map_err(text)?
                .map_ahead_decode(gpu, &epoch, remaining);
        }
        Ok(())
    }

    /// The reverse of the promotion's swap: the lane's request states move
    /// into the resident slot, and the session keeps them as its live state.
    fn unpark(&mut self, gpus: &mut Gpus, tag: u64, tokens: &[u32]) -> Result<()> {
        let gpu = device(gpus)?;
        let epoch = lane_epoch(tag);
        if self.session.is_none() {
            return Err("no session cache: a lane record cannot continue live".into());
        }
        let position = self
            .lanes()
            .and_then(|store| store.lane(&epoch))
            .map(|lane| lane.state.position);
        if position != Some(tokens.len()) {
            return Err(format!(
                "lane record {tag} sits at {position:?}, not at its {} tokens",
                tokens.len()
            ));
        }
        // As the promotion: the resident request moves sealed.
        self.seal_gdn_streams(gpu).map_err(text)?;
        self.with_lanes(|store, bundle| {
            let lane = store
                .lane_mut(&epoch)
                .ok_or_else(|| format!("lane record {tag} disappeared"))?;
            bundle.state.clamp_f16_valid();
            lane.state.clamp_f16_valid();
            bundle.state.swap_requests(&mut lane.state);
            store.release(&epoch)
        })
        .map_err(text)?;
        // The kept state continues: request-local PLE work of an earlier
        // request must not publish into it (`session_begin`'s live rule).
        self.invalidate_ple_epoch().map_err(text)?;
        if let Some(session) = self.session.as_mut() {
            session.adopt_live(tokens, SessionRoute::Ar);
        }
        Ok(())
    }

    fn step_lanes(&mut self, gpus: &mut Gpus, rows: &[(u64, u32, bool)]) -> Result<Vec<DecodeOut>> {
        let gpu = device(gpus)?;
        let store = self.lanes().ok_or("Qwen4 lanes are not staged")?;
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
        if let Err(error) = self.release_lane(&lane_epoch(tag)) {
            eprintln!("[engine] lane {tag} release: {error}");
        }
    }
}
