// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single-card lanes as one rank of the mesh lane protocol
//! ([`crate::mesh_lane_protocol`], `release-0.4.3/mgmtp/DESIGN.md` §1, §3.2).
//!
//! One rank that is the root, the target owner, the head owner and the
//! sampler runs the SAME reducer the multicard topologies run:
//!
//! ```text
//! draft_step      plan_window -> begin_window -> draft rounds -> seal_verify
//! provision_step  (maps the lanes; checks the plan against the sealed rows)
//! forward_step    run_verify -> decide_commit -> prepare_commit (+ prefill tiles, price)
//! commit_step     publish_commit (root + rank) -> settle the step's price
//! ```
//!
//! The root ([`LaneProto::root`]) owns what the singleton lane route kept
//! per lane and per store: the window router (floor, agreement, lab route
//! cycle), the pooled [`crate::mtp_lanes::LaneCostModel`], the per-request
//! RNG ([`hipfire_runtime::spec_sampling::SpecRng`]; the greedy and AR lanes
//! never draw), the generated IDs and the pending seed. The lane table's
//! `LaneHost::{generated, pending_seed}` are the mirror of the root's
//! published state (the table, the driver and the PP executor read them);
//! they are written only from a published step output.
//!
//! The route is byte-identical to the one it replaced:
//!
//! - the root runs with [`MeshRoot::set_driver_terminates`] (the driver owns
//!   request termination; drafting runs through an EOS draft and a lane
//!   whose pending seed is EOS is still stepped once, as before);
//! - a window's price is the wall clock from the step's draft start to the
//!   synchronized end of its forward (the controller interval), merged with
//!   the rank's counts ([`merge_evidence`]: one rank, the merge is the
//!   identity on the wall) and settled eagerly over every live lane
//!   ([`MeshRoot::settle_evidence`]) at the end of the step, where the old
//!   forward observed it;
//! - on any error the participants are poisoned and forgotten and the root
//!   and rank return to Idle ([`MeshRoot::discard_step`]): on one card only
//!   the participants' device state is uncertain, so the mesh quarantine
//!   does not apply and every other lane keeps its state.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward_lanes::{
    forward_lanes, row_group_count, LaneOutputs, LaneRowKind, LaneRows,
};
use crate::lane::{
    download_lane_rows, lane_sample_step, LaneSampleOutcome, LaneSpec, LaneSpecConfig, LaneStatus,
    Qwen4LaneState, Qwen4LaneStore,
};
use crate::lane_exec::{pick_at, LaneDraft, LaneRoute};
use crate::mesh_lane_protocol::{
    CommitDecision, ControllerInterval, DraftProposal, DraftRound, LaneAdmission, LaneKind,
    LaneMeshExecutor, LaneRequest, MergedCostEvidence, MeshRoot, MeshStepKey, OpCounts,
    PreparedReceipt, ProtocolError, RankCostEvidence, RankDraftResult, RankError, RankOwnership,
    RankReady, RankSession, RankVerifyResult, RootConfig, SamplerOutput, SamplingMode, StagePolicy,
    StepPhase, VerifyPlan, WindowBudgets, WindowStart,
};
use crate::mesh_rank_copy::{
    arm_lane_verify, begin_lane_window, catch_up_lane, head_ticket_point, is_probe_op,
    native_keep_cap, prepare_rank_lane, probe_outcome, publish_rank_lane, save_probe_state,
    settle_probe, take_head_ticket, HeadTicketPoint, RankPrepare,
};
use crate::mtp_lanes::{lane_call, PendingWindow};
use crate::mtp_spec::{draft_accept_estimate, DRAFT_ACCEPT_MAX, MTP_ROW_WORTH};
use hipfire_runtime::ngram_mod::NgramModConfig;
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::GreedyAccept;
use rdna_compute::{Gpu, GpuTensor};

/// What the protocol holds of a store between the driver's entries: the
/// root (policy, RNG, IDs, price), the rank session (phase/ticket reducer)
/// and the in-flight step.
pub(crate) struct LaneProto {
    pub(crate) root: MeshRoot,
    pub(crate) rank: RankSession,
    pub(crate) flight: Option<LaneFlight>,
    /// The store stages the opt-in copy lane (every spec admission arms it).
    copy_armed: bool,
    /// The merged price of a published step that held a copy window: it
    /// settles through the next `plan_window` ([`MeshRoot::plan_window`]'s
    /// prior evidence), never eagerly.
    pub(crate) prior: Option<MergedCostEvidence>,
}

/// One step between `draft_step` and `commit_step`.
pub(crate) struct LaneFlight {
    pub(crate) start: WindowStart,
    /// Per lane (plan order): the probability the draft prefix is accepted.
    prefix: Vec<f32>,
    /// Head Draft invocations executed.
    pub(crate) head_calls: usize,
    /// The sealed verify (set by `draft_step`).
    pub(crate) verify: Option<VerifyPlan>,
    /// Per lane (plan order): the target's pick at every verify row (the
    /// lane's own draws on a sampled lane).
    picks: Vec<Vec<u32>>,
    /// Per lane: the naive window a sampled Verify lane drew.
    naive: Vec<Option<GreedyAccept>>,
    /// Row groups the forward executed.
    row_groups: usize,
    /// Owed plus retained head rows the rank appended (the session's
    /// accounting of the decision).
    head_append_rows: usize,
    /// Owed head rows the catch-ups of round 0 appended.
    catch_up_rows: usize,
    pub(crate) decision: Option<CommitDecision>,
    /// The rank's receipt and the controller interval of the step.
    pub(crate) evidence: Option<(RankCostEvidence, ControllerInterval)>,
    /// An infallible-by-contract publish that failed (surfaced by commit).
    publish_error: Option<String>,
}

impl LaneFlight {
    pub(crate) fn lane_index(&self, epoch: &RequestEpoch) -> Option<usize> {
        self.start
            .lanes
            .iter()
            .position(|l| &l.key.request_epoch == epoch)
    }

    /// The target picks of lane `i` (plan order).
    pub(crate) fn picks(&self, i: usize) -> &[u32] {
        &self.picks[i]
    }
}

/// A lane entering Decoding at the root.
pub(crate) struct DecodeAdmission<'a> {
    pub epoch: RequestEpoch,
    pub position: usize,
    pub seed: u32,
    /// The accept rule's EOS; `None` for a `min_tokens` lane
    /// ([`crate::lane::LaneHost::ignore_eos`]).
    pub eos: Option<u32>,
    pub rng_seed: u64,
    pub prompt: &'a [u32],
    pub spec_k: Option<usize>,
    /// A promoted singleton's drafter: per-depth agreement and whether its
    /// floor retired it.
    pub promoted: Option<(&'a [(f32, f32)], bool)>,
}

impl LaneProto {
    /// The protocol of a store. `copy`: the n-gram config of the opt-in copy
    /// lane ([`MeshRoot::with_copy_lane`]); `None` leaves copy off on every
    /// lane (the default store).
    pub(crate) fn new(copy: Option<NgramModConfig>) -> Result<Self, String> {
        let own = RankOwnership::single();
        let mut root = MeshRoot::new(
            own,
            &[own],
            RootConfig {
                topology_generation: 1,
                stage: StagePolicy {
                    moe_layers: 0,
                    moe_once: false,
                    stream_bf16: false,
                },
                floor_enabled: crate::mtp_lanes::floor_env_enabled(),
            },
        )
        .map_err(|e| format!("lane protocol root: {e}"))?;
        root.set_driver_terminates(true);
        if let Some(ngram) = copy {
            root = root
                .with_copy_lane(ngram)
                .map_err(|e| format!("lane protocol copy lane: {e}"))?;
        }
        Ok(Self {
            root,
            rank: RankSession::new(own),
            flight: None,
            copy_armed: copy.is_some(),
            prior: None,
        })
    }

    /// A lane's first decode step: the root learns the request (the seed is
    /// the prompt-completing pick, or the promoted singleton's pending seed).
    pub(crate) fn admit_decoding(&mut self, a: DecodeAdmission<'_>) -> Result<(), String> {
        // The singleton lane route reads the floor env at every admission.
        self.root
            .set_floor_enabled(crate::mtp_lanes::floor_env_enabled());
        let retired = a.promoted.is_some_and(|(_, retired)| retired);
        let admission = LaneAdmission {
            epoch: a.epoch,
            position: a.position,
            seed: a.seed,
            // Drafting runs through an EOS draft (`set_driver_terminates`);
            // the accept rule's EOS is the plan's.
            eos: a.eos,
            rng_seed: a.rng_seed,
            prompt: a.prompt.to_vec(),
            spec_k: a.spec_k,
            head: if retired {
                crate::mesh_lane_protocol::HeadState::Free
            } else {
                crate::mesh_lane_protocol::HeadState::Aligned
            },
            // The copy lane is opt-in per store (lab/oracle): a spec lane of
            // a copy-armed store arms it, except a promoted request the
            // floor already retired (it has no live head to confirm with).
            copy: self.copy_armed && a.spec_k.is_some() && !retired,
            head_lag: Vec::new(),
        };
        let admit = match a.promoted {
            Some((agreement, retired)) => self.root.admit_promoted(admission, agreement, retired),
            None => self.root.admit_lane(admission),
        }
        .map_err(|e| format!("lane protocol admit: {e}"))?;
        self.rank.admit(&admit).map_err(|e| e.reason)
    }

    /// Forget a released lane (a lane that never reached Decoding is not at
    /// the root).
    pub(crate) fn release(&mut self, epoch: &RequestEpoch) {
        let _ = self.root.release_lane(*epoch);
        let _ = self.rank.release(epoch);
    }

    /// Replace a Decoding lane's pending seed at the root (a lane that is not
    /// at the root yet has no pending seed to replace).
    pub(crate) fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        self.root
            .force_seed(*epoch, token)
            .map_err(|e| format!("force_seed: {e}"))
    }

    /// Drop the in-flight step and forget the poisoned participants.
    pub(crate) fn discard(&mut self, participants: &[RequestEpoch]) {
        self.flight = None;
        self.prior = None;
        let _ = self.root.discard_step(participants);
        let _ = self.rank.discard_step();
        for epoch in participants {
            self.rank.forget_lane(epoch);
        }
    }
}

fn rank_error(reason: impl Into<String>) -> RankError {
    RankError {
        rank: 0,
        reason: reason.into(),
    }
}

/// The store's one rank, with the device handles its phases run on.
pub(crate) struct LocalLaneRank<'a> {
    pub(crate) bundle: &'a mut Qwen4Bundle,
    pub(crate) gpu: &'a mut Gpu,
    pub(crate) store: &'a mut Qwen4LaneStore,
    pub(crate) session: &'a mut RankSession,
    pub(crate) flight: &'a mut Option<LaneFlight>,
}

impl LocalLaneRank<'_> {
    /// Mapping, head tickets and the pending window of every lane, before
    /// any draft (the capacity half of `begin_window`). A failure rejects the
    /// whole step. The device work of one lane is
    /// [`crate::mesh_rank_copy::begin_lane_window`], shared with the EP/PP
    /// ranks.
    fn preflight(&mut self, start: &WindowStart) -> Result<(), String> {
        let Self {
            bundle, gpu, store, ..
        } = self;
        for l in &start.lanes {
            let epoch = l.key.request_epoch;
            let lane = store
                .lane_mut(&epoch)
                .ok_or_else(|| format!("draft_step: unknown or stale epoch {epoch:?}"))?;
            if lane.host.status != LaneStatus::Decoding {
                return Err(format!(
                    "draft_step: lane {epoch:?} is {:?}, not Decoding",
                    lane.host.status
                ));
            }
            if lane.host.pending_seed != Some(l.seed) || lane.state.position != l.position {
                return Err(format!(
                    "draft_step: lane {epoch:?} (seed {:?}, position {}) disagrees with the root's plan (seed {}, position {})",
                    lane.host.pending_seed, lane.state.position, l.seed, l.position
                ));
            }
            begin_lane_window(bundle, gpu, lane, l)?;
        }
        Ok(())
    }

    fn flight_mut(&mut self) -> Result<&mut LaneFlight, RankError> {
        self.flight
            .as_mut()
            .ok_or_else(|| rank_error("no lane step in flight"))
    }

    /// Download the argmax picks and run the sampled lanes' draws of a
    /// sealed verify; the step's decode forward.
    fn verify_device(
        &mut self,
        plan: &VerifyPlan,
        forced: &[bool],
    ) -> Result<(Vec<Vec<u32>>, Vec<Option<GreedyAccept>>, usize), String> {
        let Self {
            bundle, gpu, store, ..
        } = self;
        let vocab = bundle.lane_vocab();
        let epochs: Vec<RequestEpoch> = plan.lanes.iter().map(|l| l.key.request_epoch).collect();
        let spec_config: Option<LaneSpecConfig> = store.spec();
        let lane_eos = spec_config
            .and_then(|config| config.end_of_turn)
            .unwrap_or(bundle.config.eos_token_id);
        let parts = store.step_parts(&epochs)?;
        let crate::lane::StepParts {
            mut lanes,
            logits,
            top1,
            policy,
            top1_host,
            rows_cap,
            sample_host,
            spec,
        } = parts;
        let hidden: Option<&GpuTensor> = spec.as_ref().map(|s| &s.hidden);
        let n_dec: usize = plan.lanes.iter().map(|l| l.rows.len()).sum();
        if n_dec > rows_cap {
            return Err(format!(
                "{n_dec} decode rows exceed the step buffers ({rows_cap})"
            ));
        }
        let dec_rows: Vec<usize> = plan.lanes.iter().map(|l| l.rows.len()).collect();

        // Arm the ring, then take the target ticket, of every Verify lane
        // (native, copy and forced windows alike).
        for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
            if vl.rows.len() == 1 {
                continue;
            }
            arm_lane_verify(bundle, gpu, &mut **lane, vl)?;
        }

        // One exact trunk step over every decode row, in plan order.
        {
            let mut rows: Vec<LaneRows<'_>> = Vec::with_capacity(plan.lanes.len());
            for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
                rows.push(LaneRows {
                    state: &mut lane.state,
                    tokens: &vl.rows,
                    kind: if vl.rows.len() == 1 {
                        LaneRowKind::Ar
                    } else {
                        LaneRowKind::Verify {
                            draft_len: vl.rows.len() - 1,
                        }
                    },
                });
            }
            let out = LaneOutputs {
                logits: &logits.sub_offset(0, n_dec * vocab),
                top1: &top1.sub_offset(0, n_dec * std::mem::size_of::<i32>()),
                wide_hidden: hidden,
            };
            let forwarded =
                forward_lanes(gpu, bundle, &mut rows, &out, &policy).map_err(|e| e.to_string());
            drop(rows);
            // I3: disarmed on every path out of the forward.
            for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
                if vl.rows.len() > 1 {
                    lane.state.row_capture_armed = false;
                }
            }
            forwarded?;
        }
        gpu.hip
            .memcpy_dtoh(
                &mut top1_host[..n_dec * std::mem::size_of::<i32>()],
                &top1.buf,
            )
            .map_err(|e| e.to_string())?;
        gpu.take_device_wait_error().map_err(|e| e.to_string())?;

        // A sampled lane's argmax picks are replaced by its sampler's draws
        // (`lane_sample_step`: one D2H of the request's logits rows); a
        // greedy lane takes the unchanged argmax path.
        let mut picks: Vec<Vec<u32>> = Vec::with_capacity(plan.lanes.len());
        let mut naive: Vec<Option<GreedyAccept>> = Vec::with_capacity(plan.lanes.len());
        let mut first = 0usize;
        for (i, (vl, lane)) in plan.lanes.iter().zip(lanes.iter_mut()).enumerate() {
            let rows = vl.rows.len();
            let mut lane_picks: Vec<u32> =
                (0..rows).map(|j| pick_at(top1_host, first + j)).collect();
            let mut window = None;
            // A forced advance draws nothing: no sampling, no RNG.
            if lane.host.sampling.is_some() && !forced[i] {
                let drafts = (rows > 1).then(|| &vl.rows[1..]);
                let gpu_ref: &Gpu = gpu;
                let logits_ref: &GpuTensor = logits;
                let sampled = lane_sample_step(
                    &mut lane.host,
                    drafts,
                    &mut lane_picks,
                    lane_eos,
                    vocab,
                    sample_host,
                    |scratch| download_lane_rows(gpu_ref, logits_ref, first, rows, vocab, scratch),
                )
                .map_err(|error| format!("lane {:?}: {error}", vl.key.request_epoch))?;
                if let LaneSampleOutcome::Naive(w) = sampled {
                    window = Some(w);
                }
            }
            picks.push(lane_picks);
            naive.push(window);
            first += rows;
        }
        Ok((picks, naive, row_group_count(&dec_rows)))
    }
}

impl LaneMeshExecutor for LocalLaneRank<'_> {
    fn ownership(&self) -> RankOwnership {
        self.session.ownership()
    }

    fn admit_lane(
        &mut self,
        admit: &crate::mesh_lane_protocol::LaneAdmit,
    ) -> Result<(), RankError> {
        self.session.admit(admit)
    }

    fn release_lane(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        self.session.release(epoch)
    }

    fn begin_window(&mut self, start: &WindowStart) -> RankReady {
        let capacity = self.preflight(start);
        let ready = self.session.begin(start, capacity);
        if matches!(
            ready.status,
            crate::mesh_lane_protocol::ReadyStatus::Ready { .. }
        ) {
            *self.flight = Some(LaneFlight {
                start: start.clone(),
                prefix: vec![1.0; start.lanes.len()],
                head_calls: 0,
                verify: None,
                picks: Vec::new(),
                naive: Vec::new(),
                row_groups: 0,
                head_append_rows: 0,
                catch_up_rows: 0,
                decision: None,
                evidence: None,
                publish_error: None,
            });
        }
        ready
    }

    fn reject_window(&mut self, key: &MeshStepKey) {
        self.session.reject(key);
        *self.flight = None;
    }

    fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
        let mut result = self.session.on_draft_round(round)?;
        let Self {
            bundle,
            gpu,
            store,
            flight,
            ..
        } = self;
        let flight = flight
            .as_mut()
            .ok_or_else(|| rank_error("no lane step in flight"))?;
        // Round 0: the plan's owed-row catch-ups run before any Draft op, and
        // the head tickets that follow a catch-up are taken right after it
        // (the ticket is the head as the catch-up left it).
        let mut catch_rows = 0usize;
        if !round.catch_up.is_empty() {
            let epochs: Vec<RequestEpoch> = round
                .catch_up
                .iter()
                .map(|(key, _)| key.request_epoch)
                .collect();
            let (mut lanes, _) = store.draft_parts(&epochs).map_err(rank_error)?;
            for ((key, catch_up), lane) in round.catch_up.iter().zip(lanes.iter_mut()) {
                catch_rows +=
                    catch_up_lane(bundle, gpu, &mut **lane, catch_up).map_err(rank_error)?;
                let plan = flight
                    .start
                    .lanes
                    .iter()
                    .find(|l| l.key == *key)
                    .ok_or_else(|| rank_error("catch-up for a lane outside the plan"))?;
                if head_ticket_point(plan) == HeadTicketPoint::AfterCatchUp {
                    take_head_ticket(bundle, gpu, &mut **lane).map_err(rank_error)?;
                }
            }
        }
        if catch_rows != result.catch_up_rows {
            return Err(rank_error(format!(
                "the rank appended {catch_rows} owed head rows, the round scheduled {}",
                result.catch_up_rows
            )));
        }
        flight.catch_up_rows += catch_rows;
        if round.ops.is_empty() {
            result.proposals = Some(Vec::new());
            return Ok(result);
        }
        let plan_of: Vec<usize> = round
            .ops
            .iter()
            .map(|op| {
                flight
                    .start
                    .lanes
                    .iter()
                    .position(|l| l.key == op.key)
                    .ok_or_else(|| rank_error("draft op for a lane outside the plan"))
            })
            .collect::<Result<_, _>>()?;
        let epochs: Vec<RequestEpoch> = round.ops.iter().map(|o| o.key.request_epoch).collect();
        let (mut lanes, spec_store) = store.draft_parts(&epochs).map_err(rank_error)?;
        let spec_store = spec_store
            .ok_or_else(|| rank_error("draft_step: batched window in a store without MTP lanes"))?;
        let slot_bytes = Gpu::TOPK8_OUT_BYTES;
        #[cfg(feature = "lab")]
        crate::lab_draft_rounds::fill_stale(gpu, &lanes).map_err(rank_error)?;
        // Lab oracle of the batched head round: serial vs batched on scratch
        // twins of this round's lanes, before the round's own work.
        #[cfg(feature = "lab")]
        if crate::lab_draft_rounds::probe_enabled() && round.ops.len() >= 2 {
            let inputs: Vec<(u32, usize)> =
                round.ops.iter().map(|op| (op.input, op.position)).collect();
            crate::lab_draft_rounds::probe_round(bundle, gpu, &mut lanes, &inputs, 0, round.round)
                .map_err(rank_error)?;
        }
        // One batched head call for the round (`HIPFIRE_QWEN4_MTP_DRAFT_ROUNDS=1`),
        // else every still-drafting lane enqueues its own draft step, in plan order.
        let batched = crate::mtp_lanes::draft_round_batchable(bundle, round.ops.len(), false);
        if batched {
            let mut inputs = Vec::with_capacity(round.ops.len());
            for (slot, (op, lane)) in round.ops.iter().zip(lanes.iter_mut()).enumerate() {
                if is_probe_op(&flight.start.lanes[plan_of[slot]], op.position) {
                    save_probe_state(&mut **lane);
                }
                inputs.push((op.input, op.position));
            }
            crate::mtp_lanes::enqueue_batched_draft(
                bundle,
                gpu,
                &mut lanes,
                &inputs,
                &spec_store.draft_top1,
            )
            .map_err(|e| rank_error(format!("draft_step: {e}")))?;
        }
        for (slot, (op, lane)) in round
            .ops
            .iter()
            .zip(lanes.iter_mut())
            .enumerate()
            .filter(|_| !batched)
        {
            // The confirmation probe's Draft is abandoned on a hit: save the
            // head's request state it is about to move.
            if is_probe_op(&flight.start.lanes[plan_of[slot]], op.position) {
                save_probe_state(&mut **lane);
            }
            let Qwen4LaneState { state, spec, .. } = &mut **lane;
            let spec = spec
                .as_mut()
                .ok_or_else(|| rank_error("draft_step: drafting lane lost its MTP state"))?;
            let LaneSpec {
                head,
                draft,
                pending_hidden,
                window,
                ..
            } = spec;
            let window_position = window
                .as_ref()
                .ok_or_else(|| rank_error("draft_step: window vanished"))?
                .position;
            let j = op.position - window_position;
            let top1 = spec_store
                .draft_top1
                .sub_offset(slot * slot_bytes, slot_bytes);
            let hidden = (j == 0).then_some(&*pending_hidden);
            let (input, position) = (op.input, op.position);
            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                b.mtp_forward_token_deferred(gpu, input, hidden, position, j == 0, &top1)
            }))
            .map_err(|e| rank_error(format!("draft_step: draft step {j}: {e}")))?;
        }
        // One D2H for every active lane.
        let bytes = round.ops.len() * slot_bytes;
        gpu.hip
            .memcpy_dtoh(
                &mut spec_store.draft_top1_host[..bytes],
                &spec_store.draft_top1.buf,
            )
            .map_err(|e| rank_error(format!("draft_step: draft download: {e}")))?;
        gpu.take_device_wait_error()
            .map_err(|e| rank_error(format!("draft_step: draft download: {e}")))?;
        let mut proposals = Vec::with_capacity(round.ops.len());
        for (slot, (op, lane)) in round.ops.iter().zip(lanes.iter_mut()).enumerate() {
            let top = &spec_store.draft_top1_host[slot * slot_bytes..(slot + 1) * slot_bytes];
            let i = plan_of[slot];
            let plan = &flight.start.lanes[i];
            let k_max = native_keep_cap(plan);
            let (token, margin) = if batched {
                crate::mtp_lanes::finish_batched_draft(bundle, &mut **lane, top).map_err(|e| {
                    rank_error(format!("draft_step: draft finish {}: {e}", round.round))
                })?
            } else {
                let Qwen4LaneState { state, spec, .. } = &mut **lane;
                let spec = spec
                    .as_mut()
                    .ok_or_else(|| rank_error("draft_step: drafting lane lost its MTP state"))?;
                let LaneSpec { head, draft, .. } = spec;
                lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                    let token = b.mtp_draft_finish(top)?;
                    Ok::<_, crate::bundle::BundleError>((token, b.mtp_draft_margin()))
                }))
                .map_err(|e| rank_error(format!("draft_step: draft finish {}: {e}", round.round)))?
            };
            // The probe: the head's argmax at the seed against the first
            // candidate (the root's own rule). A hit, or a miss with no
            // native route, abandons the probe's head step.
            if is_probe_op(plan, op.position) {
                let outcome = probe_outcome(&plan.copy_drafts, token, plan.native_depth);
                settle_probe(bundle, gpu, &mut **lane, outcome).map_err(rank_error)?;
                if outcome.restores_head() {
                    proposals.push(DraftProposal {
                        key: op.key,
                        token,
                        keep_going: false,
                        drop_last: false,
                        read: None,
                    });
                    continue;
                }
            }
            let window = lane
                .spec
                .as_mut()
                .and_then(|s| s.window.as_mut())
                .ok_or_else(|| rank_error("draft_step: window vanished"))?;
            let j = op.position - window.position;
            // The singleton's early stop, statement for statement.
            window.steps += 1;
            flight.prefix[i] *= draft_accept_estimate(margin);
            if j > 0 && flight.prefix[i] < MTP_ROW_WORTH {
                proposals.push(DraftProposal {
                    key: op.key,
                    token,
                    keep_going: false,
                    drop_last: true,
                    read: None,
                });
                continue;
            }
            let done = flight.prefix[i] * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH || j + 1 >= k_max;
            proposals.push(DraftProposal {
                key: op.key,
                token,
                keep_going: !done,
                drop_last: false,
                read: None,
            });
        }
        flight.head_calls += round.ops.len();
        result.proposals = Some(proposals);
        Ok(result)
    }

    fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
        let mut result = self.session.on_verify(plan)?;
        let forced: Vec<bool> = self
            .flight_mut()?
            .start
            .lanes
            .iter()
            .map(|l| matches!(l.kind, LaneKind::Forced(_)))
            .collect();
        let (picks, naive, row_groups) = self.verify_device(plan, &forced).map_err(rank_error)?;
        if row_groups != plan.ledger.row_groups {
            return Err(rank_error(format!(
                "the forward ran {row_groups} row groups, the sealed plan {}",
                plan.ledger.row_groups
            )));
        }
        result.sampler = Some(SamplerOutput {
            picks: plan
                .lanes
                .iter()
                .zip(&picks)
                .map(|(l, p)| (l.key, p.clone()))
                .collect(),
            rows: None,
        });
        let flight = self.flight_mut()?;
        flight.picks = picks;
        flight.naive = naive;
        flight.row_groups = row_groups;
        Ok(result)
    }

    fn prepare_commit(&mut self, decision: &CommitDecision) -> Result<PreparedReceipt, RankError> {
        let receipt = self.session.on_prepare(decision)?;
        let Self {
            bundle,
            gpu,
            store,
            flight,
            ..
        } = self;
        let flight = flight
            .as_mut()
            .ok_or_else(|| rank_error("no lane step in flight"))?;
        let verify = flight
            .verify
            .clone()
            .ok_or_else(|| rank_error("prepare_commit: the verify was never sealed"))?;
        let epochs: Vec<RequestEpoch> = verify.lanes.iter().map(|l| l.key.request_epoch).collect();
        let spec_config = store.spec();
        let skip_rollback = store.skip_rollback();
        let crate::lane::StepParts {
            mut lanes, spec, ..
        } = store.step_parts(&epochs).map_err(rank_error)?;
        let hidden: Option<&GpuTensor> = spec.as_ref().map(|s| &s.hidden);
        let mut first = 0usize;
        // Head rows the rank appended: round 0's catch-ups plus every lane's
        // prepare schedule; the session's receipt counts the decision's.
        let mut head_rows = flight.catch_up_rows;
        for (i, (d, lane)) in decision.lanes.iter().zip(lanes.iter_mut()).enumerate() {
            let vl = &verify.lanes[i];
            let rows = vl.rows.len();
            let epoch = vl.key.request_epoch;
            let lane_picks = &flight.picks[i];
            let plan = &flight.start.lanes[i];
            if lane.spec.is_some() {
                let config = spec_config
                    .ok_or_else(|| rank_error("MTP lane in a store without MTP lanes"))?;
                let hidden =
                    hidden.ok_or_else(|| rank_error("MTP lane without a hidden capture"))?;
                let prepared = prepare_rank_lane(
                    bundle,
                    gpu,
                    &mut *lane,
                    RankPrepare {
                        plan,
                        decision: d,
                        verify: vl,
                        config,
                        hidden,
                        row_begin: first,
                        picks: lane_picks,
                        skip_rollback,
                    },
                )
                .map_err(rank_error)?;
                head_rows += prepared.head_rows;
                if let Some(naive) = &flight.naive[i] {
                    if d.emitted != naive.committed || d.accepted != naive.accepted {
                        return Err(rank_error(format!(
                            "sampled Verify lane {epoch:?}: the root decided {:?} (accepted {}) \
                             but the naive window is {:?} (accepted {})",
                            d.emitted, d.accepted, naive.committed, naive.accepted
                        )));
                    }
                }
                if let Some(w) = lane.spec.as_mut().and_then(|s| s.window.as_mut()) {
                    w.prepared = Some(prepared.lane);
                }
            }
            if rows == 1 && !d.forced && d.emitted != lane_picks[..1] {
                return Err(rank_error(format!(
                    "AR lane {epoch:?}: the root decided {:?}, the target picked {:?}",
                    d.emitted, lane_picks
                )));
            }
            first += rows;
        }
        if head_rows != receipt.head_append_rows {
            return Err(rank_error(format!(
                "the rank appended {head_rows} head rows, the decision's schedule is {}",
                receipt.head_append_rows
            )));
        }
        flight.decision = Some(decision.clone());
        flight.head_append_rows = receipt.head_append_rows;
        Ok(receipt)
    }

    fn publish_commit(&mut self, decision: &CommitDecision) {
        let Self {
            bundle,
            gpu,
            store,
            session,
            flight,
        } = self;
        if let Some(flight) = flight.as_mut() {
            for d in &decision.lanes {
                let epoch = d.key.request_epoch;
                let Some(lane) = store.lane_mut(&epoch) else {
                    continue;
                };
                let Some(prepared) = lane
                    .spec
                    .as_mut()
                    .and_then(|s| s.window.as_mut())
                    .and_then(|w| w.prepared.take())
                else {
                    continue;
                };
                if let Err(error) = publish_rank_lane(bundle, gpu, lane, prepared) {
                    flight.publish_error.get_or_insert(error);
                }
            }
        }
        session.on_publish(decision);
    }

    fn abort_step(&mut self, _key: &MeshStepKey, _phase: StepPhase, _reason: &str) {
        let _ = self.session.discard_step();
        *self.flight = None;
    }

    fn evidence(&mut self, key: &MeshStepKey) -> Option<RankCostEvidence> {
        self.flight
            .as_ref()
            .filter(|f| &f.start.key == key)
            .and_then(|f| f.evidence.as_ref().map(|(e, _)| e.clone()))
    }
}

impl LaneFlight {
    /// The rank's counts of a prepared step, from what it executed and the
    /// root's decision.
    pub(crate) fn counts(&self, decision: &CommitDecision) -> OpCounts {
        let head_append_rows = self.head_append_rows;
        OpCounts {
            target_exchanges: 0,
            gather_bytes: 0,
            head_draft_calls: self.head_calls,
            head_append_rows,
            row_groups: self.row_groups,
            edge_count: 0,
            edge_bytes: 0,
            drafts_generated: decision.lanes.iter().map(|d| d.source.offered).sum(),
            accepted: decision.lanes.iter().map(|d| d.source.accepted).sum(),
            consumed_rows: decision.lanes.iter().map(|d| d.consumed_rows).sum(),
        }
    }

    /// The rank receipt of a prepared step: counts only (the controller
    /// interval is the price; a single rank has no phase timings worth
    /// merging).
    pub(crate) fn receipt(&self, decision: &CommitDecision) -> RankCostEvidence {
        RankCostEvidence {
            key: self.start.key.clone(),
            rank: 0,
            completed: StepPhase::Prepared,
            phases: Vec::new(),
            nodes: Vec::new(),
            counts: self.counts(decision),
            unpriced: false,
        }
    }

    /// A failed infallible-by-contract publish, surfaced by the commit.
    pub(crate) fn take_publish_error(&mut self) -> Option<String> {
        self.publish_error.take()
    }
}

impl LaneProto {
    /// Whether a drafted step is waiting for its forward.
    pub(crate) fn has_flight(&self) -> bool {
        self.flight.is_some()
    }
}
/// Route, begin and draft the step of `requests` (each lane's most drafts
/// this window, 0 = AR) through the protocol's `plan_window → begin_window →
/// draft rounds → seal_verify`, in slot order (the order the plan's rows
/// keep). Returns each requested lane's routed window, in request order.
/// The sealed step stays in the store's flight for `provision_step` and
/// `forward_step`; on `Err` the caller poisons the lanes and discards the
/// step.
pub(crate) fn plan_and_draft(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    requests: &[(RequestEpoch, usize)],
    forced: &[(RequestEpoch, Vec<u32>)],
) -> Result<Vec<LaneDraft>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let mut proto = store.take_proto()?;
    let result = plan_and_draft_with(bundle, gpu, store, &mut proto, requests, forced);
    store.put_proto(proto);
    result
}

fn plan_and_draft_with(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    proto: &mut LaneProto,
    requests: &[(RequestEpoch, usize)],
    forced: &[(RequestEpoch, Vec<u32>)],
) -> Result<Vec<LaneDraft>, String> {
    let mut order: Vec<(usize, RequestEpoch, usize, SamplingMode)> =
        Vec::with_capacity(requests.len());
    for (epoch, budget) in requests {
        let lane = store
            .lane(epoch)
            .ok_or_else(|| format!("draft_step: unknown or stale epoch {epoch:?}"))?;
        if lane.host.status != LaneStatus::Decoding {
            return Err(format!(
                "draft_step: lane {epoch:?} is {:?}, not Decoding",
                lane.host.status
            ));
        }
        if lane.host.pending_seed.is_none() {
            return Err(format!("draft_step: lane {epoch:?} has no pending seed"));
        }
        let sampling = if lane.host.sampling.is_some() {
            SamplingMode::NaiveSampled
        } else {
            SamplingMode::Greedy
        };
        order.push((lane.host.slot, *epoch, *budget, sampling));
    }
    // The plan's decode rows keep slot order: so does the sealed verify.
    order.sort_by_key(|&(slot, ..)| slot);
    // `budget` is the driver's draft allowance (tokens left and context
    // already folded in): `budget + 1` tokens and positions remain.
    let lane_requests: Vec<LaneRequest> = order
        .iter()
        .map(|&(_, epoch, budget, sampling)| {
            let forced = forced
                .iter()
                .find(|(e, _)| *e == epoch)
                .map(|(_, tokens)| tokens.clone());
            // A forced advance verifies its own `n` rows (`n` tokens and
            // positions remain at least).
            let left = forced.as_ref().map_or(0, Vec::len).max(budget + 1);
            LaneRequest {
                epoch,
                remaining: left,
                context_left: left,
                forced,
                sampling,
            }
        })
        .collect();
    let copy = store.copy();
    let verify_rows = store.verify_rows_per_lane();
    let budgets = WindowBudgets {
        max_lanes: store.max_lanes(),
        verify_rows_per_lane: verify_rows,
        max_rows: store.max_lanes() * verify_rows,
        copy_take_cap: copy.map_or(0, |c| c.take_cap),
        lab: store.routes(),
        pp_microbatch_lanes: None,
        head_lag_rows: copy.map_or(0, |c| c.config.head_lag_rows),
    };
    let LaneProto {
        root,
        rank,
        flight,
        prior,
        ..
    } = &mut *proto;
    // The price of a published copy window settles here (never eagerly).
    let prior_evidence = prior.take();
    let start = root
        .plan_window(prior_evidence.as_ref(), &lane_requests, &budgets)
        .map_err(|e| format!("draft_step: {e}"))?;
    let mut exec = LocalLaneRank {
        bundle,
        gpu,
        store,
        session: rank,
        flight,
    };
    let ready = exec.begin_window(&start);
    match root.collect_ready(&[ready]) {
        Ok(()) => {}
        Err(ProtocolError::Preflight(rejected)) => {
            exec.reject_window(&start.key);
            let reason = rejected
                .into_iter()
                .map(|(_, reason)| reason)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(reason);
        }
        Err(e) => return Err(format!("draft_step: {e}")),
    }
    loop {
        let round = match root.next_draft_round() {
            Ok(Some(round)) => round,
            Ok(None) => break,
            Err(e) => return Err(format!("draft_step: {e}")),
        };
        let result = exec.run_draft_round(&round).map_err(|e| e.reason)?;
        root.collect_draft(&round, &[result])
            .map_err(|e| format!("draft_step: {e}"))?;
    }
    let sealed = root.seal_verify().map_err(|e| format!("draft_step: {e}"))?;
    let mut out = Vec::with_capacity(requests.len());
    for (epoch, _) in requests {
        let i = start
            .lanes
            .iter()
            .position(|l| &l.key.request_epoch == epoch)
            .ok_or_else(|| format!("draft_step: {epoch:?} is not in the plan"))?;
        let route = start.lanes[i].route;
        // The sealed rows after the seed are the window's proposals whatever
        // their source: native drafts, copy candidates, forced tokens.
        let drafts = sealed.lanes[i].rows[1..].to_vec();
        if !drafts.is_empty() {
            let window = exec
                .store
                .lane_mut(epoch)
                .and_then(|lane| lane.spec.as_mut())
                .and_then(|spec| spec.window.as_mut())
                .ok_or_else(|| "draft_step: window vanished".to_string())?;
            window.drafts = drafts.clone();
        }
        out.push(LaneDraft {
            epoch: *epoch,
            route,
            drafts,
        });
    }
    if let Some(f) = exec.flight.as_mut() {
        f.verify = Some(sealed);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! The single rank over the scripted executor: the SAME reducer the
    //! multicard topologies run must reproduce, window for window, what the
    //! singleton lane route (per-lane policy + pooled cost model observed at
    //! the end of each forward) decided.
    use super::*;
    use crate::gpu_forward_lanes::{row_group_count, LANE_MAX_ROWS};
    use crate::lane::LabRoutes;
    use crate::mesh_lane_protocol::scripted::{FailAt, Script, ScriptedExecutor, WindowScript};
    use crate::mesh_lane_protocol::{
        run_mesh_step, GreedyRule, HeadState, MergedCostEvidence, ProposalSource, StepOutput,
    };
    use crate::mtp_lanes::{LaneCostModel, LaneSpecPolicy};
    use hipfire_runtime::spec::{accept_greedy_prefix, MtpWindow};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    const K: usize = 7;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn config(floor: bool) -> RootConfig {
        RootConfig {
            topology_generation: 1,
            stage: StagePolicy {
                moe_layers: 0,
                moe_once: false,
                stream_bf16: false,
            },
            floor_enabled: floor,
        }
    }

    fn budgets(lab: LabRoutes) -> WindowBudgets {
        WindowBudgets {
            max_lanes: 8,
            verify_rows_per_lane: LANE_MAX_ROWS,
            max_rows: 8 * LANE_MAX_ROWS,
            copy_take_cap: 0,
            lab,
            pp_microbatch_lanes: None,
            head_lag_rows: 0,
        }
    }

    /// The single-card protocol: a root and the scripted rank, as the store
    /// builds them (driver-owned termination).
    struct Single {
        root: MeshRoot,
        rank: ScriptedExecutor,
        script: Rc<RefCell<Script>>,
    }

    impl Single {
        fn new(floor: bool) -> Self {
            let own = RankOwnership::single();
            let mut root = MeshRoot::new(own, &[own], config(floor)).unwrap();
            root.set_driver_terminates(true);
            let script = Rc::new(RefCell::new(Script::default()));
            Self {
                root,
                rank: ScriptedExecutor::new(own, script.clone()),
                script,
            }
        }

        fn admit(&mut self, tag: u64, eos: Option<u32>, spec_k: Option<usize>) {
            let admit = self
                .root
                .admit_lane(LaneAdmission {
                    epoch: epoch(tag),
                    position: 10,
                    seed: 7,
                    eos,
                    rng_seed: 0,
                    prompt: vec![1, 2, 3],
                    spec_k,
                    head: HeadState::Aligned,
                    copy: false,
                    head_lag: Vec::new(),
                })
                .unwrap();
            self.rank.admit_lane(&admit).unwrap();
        }

        fn request(tag: u64, budget: usize) -> LaneRequest {
            LaneRequest {
                epoch: epoch(tag),
                remaining: budget + 1,
                context_left: budget + 1,
                forced: None,
                sampling: SamplingMode::Greedy,
            }
        }

        /// One whole step; returns the plan, the output and the evidence
        /// merged with `interval`, settled eagerly like the store does.
        fn step(
            &mut self,
            requests: &[LaneRequest],
            lab: LabRoutes,
            interval: ControllerInterval,
        ) -> (WindowStart, StepOutput, MergedCostEvidence) {
            let b = budgets(lab);
            let start = self.root.plan_window(None, requests, &b).unwrap();
            let ready = self.rank.begin_window(&start);
            self.root.collect_ready(&[ready]).unwrap();
            while let Some(round) = self.root.next_draft_round().unwrap() {
                let r = self.rank.run_draft_round(&round).unwrap();
                self.root.collect_draft(&round, &[r]).unwrap();
            }
            let plan = self.root.seal_verify().unwrap();
            let v = self.rank.run_verify(&plan).unwrap();
            self.root.collect_verify(&[v]).unwrap();
            let decision = self.root.decide_commit(&mut GreedyRule).unwrap();
            let receipt = self.rank.prepare_commit(&decision).unwrap();
            self.root.collect_prepared(&[receipt]).unwrap();
            let out = self.root.publish_commit().unwrap();
            self.rank.publish_commit(&decision);
            let ev = self.rank.evidence(&out.key).unwrap();
            let merged = self.root.merge_evidence(&[ev], interval).unwrap();
            self.root.settle_evidence(Some(&merged)).unwrap();
            (start, out, merged)
        }
    }

    fn wall(us: f64) -> ControllerInterval {
        ControllerInterval {
            wall_us: us,
            prefill_ran: false,
        }
    }

    /// The old route, statement for statement (`lane_exec::forward_inner`'s
    /// pricing block, `draft_lanes`' router call).
    struct OldRoute {
        cost: LaneCostModel,
        policies: HashMap<u64, LaneSpecPolicy>,
        lab: LabRoutes,
    }

    struct OldWindow {
        route: LaneRoute,
        drafts: Vec<u32>,
        window: MtpWindow,
    }

    impl OldRoute {
        fn new(lab: LabRoutes) -> Self {
            Self {
                cost: LaneCostModel::new(),
                policies: HashMap::new(),
                lab,
            }
        }

        fn step(
            &mut self,
            lanes: &[(u64, usize)],
            script: &Script,
            seq: &HashMap<u64, u64>,
            prefill_ran: bool,
            wall_us: f32,
        ) -> Vec<OldWindow> {
            let mut windows: Vec<(u64, usize, OldWindow)> = Vec::new();
            for &(tag, budget) in lanes {
                let policy = self
                    .policies
                    .entry(tag)
                    .or_insert_with(|| LaneSpecPolicy::with_floor(K, true));
                let route = policy.route(budget, self.lab);
                let s = &script.windows[&(tag, seq[&tag])];
                let drafts: Vec<u32> = match route {
                    LaneRoute::Batched { k_max } => s.drafts[..k_max].to_vec(),
                    _ => Vec::new(),
                };
                let rows = 1 + drafts.len();
                let window = if matches!(route, LaneRoute::Batched { .. }) {
                    let acc = accept_greedy_prefix(&drafts, &s.picks[..rows], None);
                    MtpWindow {
                        committed: acc.committed,
                        accepted: acc.accepted,
                        drafts_generated: drafts.len(),
                    }
                } else {
                    MtpWindow {
                        committed: vec![s.picks[0]],
                        accepted: 0,
                        drafts_generated: 0,
                    }
                };
                windows.push((
                    tag,
                    budget.min(policy.max_k()),
                    OldWindow {
                        route,
                        drafts,
                        window,
                    },
                ));
            }
            let dec_rows: Vec<usize> = windows.iter().map(|w| 1 + w.2.drafts.len()).collect();
            let verify_windows = windows.iter().filter(|w| !w.2.drafts.is_empty()).count();
            let verified: usize = windows.iter().map(|w| w.2.drafts.len()).sum();
            let decode_lanes = windows.len();
            let spill = row_group_count(&dec_rows).saturating_sub(1);
            let spill_per_window = spill as f64 / verify_windows.max(1) as f64;
            if !prefill_ran {
                self.cost.observe(decode_lanes, spill, verified, wall_us);
            }
            for (tag, budget, w) in &windows {
                let policy = self.policies.get_mut(tag).unwrap();
                match w.route {
                    LaneRoute::Batched { k_max } => {
                        let units = self.cost.window_units(
                            decode_lanes,
                            w.window.drafts_generated,
                            spill_per_window,
                        );
                        policy.observe_batched(&w.window, units, k_max, *budget);
                    }
                    LaneRoute::ArSynced => policy.observe_ar_synced(&w.window, *budget),
                    LaneRoute::ArRetired | LaneRoute::Ar => {}
                }
            }
            windows.into_iter().map(|(_, _, w)| w).collect()
        }
    }

    fn script_window(rng: &mut Xs, tag: u64, seq: u64, accept_pct: u64, script: &mut Script) {
        let drafts: Vec<u32> = (0..K as u32).map(|j| 100 + j + tag as u32 * 10).collect();
        // Accept a prefix: each draft survives with `accept_pct` percent.
        let mut a = 0;
        while a < K && rng.below(100) < accept_pct {
            a += 1;
        }
        let picks: Vec<u32> = (0..=K)
            .map(|j| if j < a { drafts[j] } else { 5000 + j as u32 })
            .collect();
        script.windows.insert(
            (tag, seq),
            WindowScript {
                drafts,
                picks,
                keep_going: Vec::new(),
                drop_last: Vec::new(),
            },
        );
    }

    /// Plan equality with the old route: the per-window route and drafts, the
    /// committed IDs, the pooled cost fit and every lane's retirement are the
    /// old route's, over lanes that skip steps, steps that ran a prefill tile
    /// and the three route modes.
    #[test]
    fn single_rank_reducer_reproduces_the_old_lane_route() {
        let mut retired_seen = false;
        let mut batched_seen = false;
        let mut ar_seen = false;
        for (lab, seed) in [
            (LabRoutes::Floor, 0x9E37_79B9_7F4A_7C15u64),
            (LabRoutes::Floor, 0xDEAD_BEEF_CAFE_F00D),
            (LabRoutes::Cycle, 0x1234_5678_9ABC_DEF1),
            (LabRoutes::Batched, 0x0F0F_0F0F_1111_2222),
        ] {
            let mut rng = Xs(seed);
            let mut single = Single::new(true);
            let mut old = OldRoute::new(lab);
            let lanes: u64 = 1 + rng.below(8);
            // Per-lane acceptance: from near-zero (retires under the floor)
            // to near-perfect.
            let accept: Vec<u64> = (0..lanes)
                .map(|_| [5, 20, 55, 85, 97][rng.below(5) as usize])
                .collect();
            for tag in 1..=lanes {
                single.admit(tag, None, Some(K));
            }
            let mut seq: HashMap<u64, u64> = (1..=lanes).map(|t| (t, 0)).collect();
            for step in 0..260 {
                // A lane sits out about one step in five.
                let mut chosen: Vec<(u64, usize)> = Vec::new();
                for tag in 1..=lanes {
                    if lanes == 1 || rng.below(5) != 0 {
                        chosen.push((tag, [0, 1, 2, 3, 5, 7, 7][rng.below(7) as usize]));
                    }
                }
                if chosen.is_empty() {
                    continue;
                }
                {
                    let mut script = single.script.borrow_mut();
                    for &(tag, _) in &chosen {
                        script_window(
                            &mut rng,
                            tag,
                            seq[&tag],
                            accept[tag as usize - 1],
                            &mut script,
                        );
                    }
                }
                let prefill_ran = rng.below(7) == 0;
                let wall_us = 9_000.0 + rng.below(90_000) as f64;
                let interval = ControllerInterval {
                    wall_us: if prefill_ran { f64::NAN } else { wall_us },
                    prefill_ran,
                };
                let requests: Vec<LaneRequest> =
                    chosen.iter().map(|&(t, b)| Single::request(t, b)).collect();
                let (start, out, merged) = single.step(&requests, lab, interval);
                assert_eq!(merged.priced, !prefill_ran, "step {step}");
                assert!(merged.units_valid, "step {step}");
                let want = old.step(
                    &chosen,
                    &single.script.borrow(),
                    &seq,
                    prefill_ran,
                    wall_us as f32,
                );
                for (i, ((l, o), w)) in start.lanes.iter().zip(&out.lanes).zip(&want).enumerate() {
                    assert_eq!(l.route, w.route, "{lab:?} step {step} lane {i} route");
                    let drafts = single.script.borrow().windows
                        [&(l.key.request_epoch.request_tag, l.key.window_seq)]
                        .drafts
                        .clone();
                    assert_eq!(
                        o.source.offered,
                        w.drafts.len(),
                        "step {step} lane {i} depth"
                    );
                    assert_eq!(o.emitted, w.window.committed, "step {step} lane {i} ids");
                    assert_eq!(o.source.accepted, w.window.accepted, "step {step} lane {i}");
                    if let LaneRoute::Batched { k_max } = l.route {
                        assert_eq!(w.drafts, drafts[..k_max], "step {step} lane {i} drafts");
                        batched_seen = true;
                        assert_eq!(l.source, ProposalSource::Native);
                    } else {
                        ar_seen = true;
                        assert_eq!(l.source, ProposalSource::None);
                    }
                    if l.route == LaneRoute::ArRetired {
                        retired_seen = true;
                        assert_eq!(
                            l.head,
                            HeadState::Free,
                            "a retired window never touches the head"
                        );
                    }
                }
                for &(tag, _) in &chosen {
                    *seq.get_mut(&tag).unwrap() += 1;
                }
                let (f, a, r, g) = single.root.cost_fit();
                let (of, oa, or, og) = old.cost.fit();
                assert_eq!(
                    (f, a, r, g),
                    (of, oa, or, og),
                    "{lab:?} step {step} cost fit"
                );
                for tag in 1..=lanes {
                    assert_eq!(
                        single.root.lane_retired(epoch(tag)),
                        Some(old.policies.get(&tag).is_some_and(LaneSpecPolicy::retired)),
                        "{lab:?} step {step} lane {tag} retirement"
                    );
                }
            }
        }
        assert!(batched_seen && ar_seen && retired_seen);
    }

    /// The singleton lane route drafts through an EOS draft and keeps a lane
    /// whose pending seed is EOS stepping (the driver ends it); the protocol's
    /// default stops drafting at EOS and closes the lane.
    #[test]
    fn driver_owned_termination_drafts_through_eos_and_keeps_the_lane() {
        for driver in [true, false] {
            let mut single = Single::new(false);
            single.root.set_driver_terminates(driver);
            single.admit(1, Some(77), Some(3));
            single.script.borrow_mut().windows.insert(
                (1, 0),
                WindowScript {
                    drafts: vec![5, 77, 9],
                    picks: vec![5, 77, 9, 1],
                    keep_going: Vec::new(),
                    drop_last: Vec::new(),
                },
            );
            let (start, out, _) =
                single.step(&[Single::request(1, 3)], LabRoutes::Batched, wall(30_000.0));
            assert_eq!(start.lanes[0].route, LaneRoute::Batched { k_max: 3 });
            let l = &out.lanes[0];
            // Accepted EOS draft: committed [5, 77], the bonus is not emitted.
            assert_eq!(l.emitted, vec![5, 77]);
            assert_eq!(l.consumed_rows, 2);
            if driver {
                assert_eq!(l.source.offered, 3, "the chain ran through the EOS draft");
                // The EOS pending seed is still stepped once, as before.
                single.script.borrow_mut().windows.insert(
                    (1, 1),
                    WindowScript {
                        drafts: vec![1, 2, 3],
                        picks: vec![4, 4, 4, 4],
                        keep_going: Vec::new(),
                        drop_last: Vec::new(),
                    },
                );
                let (_, again, _) =
                    single.step(&[Single::request(1, 3)], LabRoutes::Batched, wall(30_000.0));
                assert_eq!(again.lanes[0].emitted, vec![4]);
            } else {
                assert_eq!(l.source.offered, 2, "the default stops drafting at EOS");
                assert!(l.terminal);
                assert!(single
                    .root
                    .plan_window(None, &[Single::request(1, 3)], &budgets(LabRoutes::Batched))
                    .is_err());
            }
        }
    }

    /// A lane admitted with no EOS rule (`min_tokens`: the root passes `None`)
    /// runs its window through the EOS draft: every draft is accepted and the
    /// bonus is emitted, where the same window under the EOS rule truncates at
    /// the EOS. The driver, which owns termination, decides what an EOS means.
    #[test]
    fn a_lane_admitted_without_an_eos_rule_runs_its_window_through_an_eos() {
        let window = || WindowScript {
            drafts: vec![5, 77, 9],
            picks: vec![5, 77, 9, 1],
            keep_going: Vec::new(),
            drop_last: Vec::new(),
        };
        for (eos, emitted) in [(Some(77), vec![5, 77]), (None, vec![5, 77, 9, 1])] {
            let mut single = Single::new(false);
            single.root.set_driver_terminates(true);
            single.admit(1, eos, Some(3));
            single.script.borrow_mut().windows.insert((1, 0), window());
            let (_, out, _) =
                single.step(&[Single::request(1, 3)], LabRoutes::Batched, wall(30_000.0));
            let l = &out.lanes[0];
            assert_eq!(l.emitted, emitted, "eos rule {eos:?}");
            assert_eq!(l.source.offered, 3, "the chain runs through the EOS draft");
        }
    }

    /// A prefill tile or an unusable wall keeps the wall out of the fit while
    /// the floor still prices at the model's units; an incomplete rank or a
    /// warm bind spends the probe unpriced.
    #[test]
    fn units_stay_valid_when_only_the_wall_is_unusable() {
        let mut single = Single::new(true);
        single.admit(1, None, Some(K));
        for (case, interval, priced, units_valid) in [
            (
                "prefill",
                ControllerInterval {
                    wall_us: f64::NAN,
                    prefill_ran: true,
                },
                false,
                true,
            ),
            ("wall", wall(f64::NAN), false, true),
            ("ok", wall(40_000.0), true, true),
        ] {
            // The lane's window sequence is the number of windows it published.
            let win = single.script.borrow().windows.len() as u64;
            single.script.borrow_mut().windows.insert(
                (1, win),
                WindowScript {
                    drafts: vec![1, 2, 3],
                    picks: vec![1, 2, 3, 4],
                    keep_going: Vec::new(),
                    drop_last: Vec::new(),
                },
            );
            let (_, _, merged) =
                single.step(&[Single::request(1, 3)], LabRoutes::Batched, interval);
            assert_eq!(
                (merged.priced, merged.units_valid),
                (priced, units_valid),
                "{case}"
            );
        }
        // An incomplete rank is unusable for the floor too.
        single.script.borrow_mut().windows.insert(
            (1, 3),
            WindowScript {
                drafts: vec![1, 2, 3],
                picks: vec![1, 2, 3, 4],
                keep_going: Vec::new(),
                drop_last: Vec::new(),
            },
        );
        single.rank.evidence_unpriced = true;
        let (_, _, merged) = single.step(&[Single::request(1, 3)], LabRoutes::Batched, wall(1.0e4));
        assert!(!merged.priced && !merged.units_valid);
    }

    /// A failed step on one card poisons only its participants: the root
    /// and rank return to Idle with every other lane's state intact.
    #[test]
    fn discard_step_recovers_the_single_rank_without_quarantine() {
        let mut single = Single::new(true);
        single.admit(1, None, Some(K));
        single.admit(2, None, Some(K));
        for tag in [1, 2] {
            single.script.borrow_mut().windows.insert(
                (tag, 0),
                WindowScript {
                    drafts: vec![1, 2, 3],
                    picks: vec![1, 2, 3, 4],
                    keep_going: Vec::new(),
                    drop_last: Vec::new(),
                },
            );
        }
        single.rank.fail_at = Some(FailAt::Verify);
        let own = RankOwnership::single();
        let err = run_mesh_step(
            &mut single.root,
            &mut [&mut single.rank],
            None,
            &[Single::request(1, 3)],
            &budgets(LabRoutes::Batched),
            &mut GreedyRule,
        );
        assert!(err.is_err());
        assert!(single.root.quarantined().is_some());
        single.root.discard_step(&[epoch(1)]).unwrap();
        single.rank.session.discard_step().unwrap();
        single.rank.session.forget_lane(&epoch(1));
        single.rank.fail_at = None;
        assert_eq!(own, single.root.ownership());
        assert!(single.root.quarantined().is_none());
        assert!(!single.root.has_lane(epoch(1)));
        assert_eq!(single.root.lane_position(epoch(2)), Some((10, 7)));
        let (_, out, _) = single.step(&[Single::request(2, 3)], LabRoutes::Batched, wall(3.0e4));
        assert_eq!(out.lanes[0].emitted, vec![1, 2, 3, 4]);
        // A mesh refuses the single-rank recovery.
        let ep = [RankOwnership::ep(0), RankOwnership::ep(1)];
        let mut mesh = MeshRoot::new(ep[0], &ep, config(true)).unwrap();
        assert!(mesh.discard_step(&[]).is_err());
        let mut session = RankSession::new(ep[1]);
        assert!(session.discard_step().is_err());
    }

    /// A forced close token replaces the pending seed only: no position
    /// move, no generated ID, no RNG draw.
    #[test]
    fn force_seed_replaces_the_seed_only() {
        let mut single = Single::new(false);
        single.admit(1, None, None);
        let rng = single.root.rng_probe(epoch(1));
        single.root.force_seed(epoch(1), 4242).unwrap();
        assert_eq!(single.root.lane_position(epoch(1)), Some((10, 4242)));
        assert_eq!(single.root.generated(epoch(1)), Some(&[][..]));
        assert_eq!(single.root.rng_probe(epoch(1)), rng);
        single.script.borrow_mut().windows.insert(
            (1, 0),
            WindowScript {
                drafts: Vec::new(),
                picks: vec![9],
                keep_going: Vec::new(),
                drop_last: Vec::new(),
            },
        );
        let (start, out, _) = single.step(&[Single::request(1, 0)], LabRoutes::Floor, wall(1.0e4));
        assert_eq!(start.lanes[0].seed, 4242);
        assert_eq!(start.lanes[0].route, LaneRoute::Ar);
        assert_eq!(out.lanes[0].emitted, vec![9]);
        assert_eq!(single.root.lane_position(epoch(1)), Some((11, 9)));
        assert!(single.root.force_seed(epoch(2), 1).is_err());
    }

    /// A promotion keeps the singleton's retirement for good (the lane is
    /// head-free) and starts every other lane's floor fresh.
    #[test]
    fn promoted_retirement_stays_and_the_head_is_free() {
        let mut single = Single::new(true);
        let own_admit = single
            .root
            .admit_promoted(
                LaneAdmission {
                    epoch: epoch(1),
                    position: 40,
                    seed: 3,
                    eos: None,
                    rng_seed: 0,
                    prompt: vec![1, 2],
                    spec_k: Some(K),
                    head: HeadState::Free,
                    copy: false,
                    head_lag: Vec::new(),
                },
                &[(0.5, 4.0); K],
                true,
            )
            .unwrap();
        single.rank.admit_lane(&own_admit).unwrap();
        assert_eq!(single.root.lane_retired(epoch(1)), Some(true));
        single.script.borrow_mut().windows.insert(
            (1, 0),
            WindowScript {
                drafts: Vec::new(),
                picks: vec![8],
                keep_going: Vec::new(),
                drop_last: Vec::new(),
            },
        );
        let (start, _, _) = single.step(&[Single::request(1, 7)], LabRoutes::Floor, wall(2.0e4));
        assert_eq!(start.lanes[0].route, LaneRoute::ArRetired);
        assert_eq!(start.lanes[0].head, HeadState::Free);
    }

    #[test]
    fn single_world_validation() {
        let one = RankOwnership::single();
        assert!(MeshRoot::new(one, &[one], config(true)).is_ok());
        assert!(MeshRoot::new(one, &[one, one], config(true)).is_err());
        assert!(one.root && one.sampler && one.head && one.target);
    }
}
