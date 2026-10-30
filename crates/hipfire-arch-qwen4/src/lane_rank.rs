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
    forward_lanes, row_group_count, LaneOutputs, LaneRowKind, LaneRows, LANE_MAX_ROWS,
};
use crate::lane::{
    download_lane_rows, lane_sample_step, LaneSampleOutcome, LaneSpec, LaneSpecConfig, LaneStatus,
    Qwen4LaneState, Qwen4LaneStore,
};
use crate::lane_exec::{pick_at, LaneDraft, LaneRoute};
use crate::mesh_lane_protocol::{
    CommitDecision, ControllerInterval, DraftProposal, DraftRound, LaneAdmission, LaneMeshExecutor,
    LaneRequest, MeshRoot, MeshStepKey, OpCounts, PreparedReceipt, ProtocolError, RankCostEvidence,
    RankDraftResult, RankError, RankOwnership, RankReady, RankSession, RankVerifyResult,
    RootConfig, SamplerOutput, SamplingMode, StagePolicy, StepPhase, VerifyPlan, WindowBudgets,
    WindowStart,
};
use crate::mtp_lanes::{
    ar_synced_head, lane_call, prepare_lane_window, publish_lane_window, PendingWindow,
};
use crate::mtp_spec::{draft_accept_estimate, DRAFT_ACCEPT_MAX, MTP_ROW_WORTH};
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::MtpWindow;
use rdna_compute::{Gpu, GpuTensor};

/// What the protocol holds of a store between the driver's entries: the
/// root (policy, RNG, IDs, price), the rank session (phase/ticket reducer)
/// and the in-flight step.
pub(crate) struct LaneProto {
    pub(crate) root: MeshRoot,
    pub(crate) rank: RankSession,
    pub(crate) flight: Option<LaneFlight>,
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
    naive: Vec<Option<MtpWindow>>,
    /// Row groups the forward executed.
    row_groups: usize,
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
    pub eos: u32,
    pub rng_seed: u64,
    pub prompt: &'a [u32],
    pub spec_k: Option<usize>,
    /// A promoted singleton's drafter: per-depth agreement and whether its
    /// floor retired it.
    pub promoted: Option<(&'a [(f32, f32)], bool)>,
}

impl LaneProto {
    pub(crate) fn new() -> Result<Self, String> {
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
        Ok(Self {
            root,
            rank: RankSession::new(own),
            flight: None,
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
            eos: Some(a.eos),
            rng_seed: a.rng_seed,
            prompt: a.prompt.to_vec(),
            spec_k: a.spec_k,
            head: if retired {
                crate::mesh_lane_protocol::HeadState::Free
            } else {
                crate::mesh_lane_protocol::HeadState::Aligned
            },
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
    /// Mapping, head snapshots and the pending window of every lane, before
    /// any draft (the capacity half of `begin_window`). A failure rejects the
    /// whole step.
    fn preflight(&mut self, start: &WindowStart) -> Result<(), String> {
        let Self {
            bundle, gpu, store, ..
        } = self;
        for l in &start.lanes {
            let epoch = l.key.request_epoch;
            let lane = store
                .lane_mut(&epoch)
                .ok_or_else(|| format!("draft_step: unknown or stale epoch {epoch:?}"))?;
            let Qwen4LaneState { host, state, spec } = lane;
            if host.status != LaneStatus::Decoding {
                return Err(format!(
                    "draft_step: lane {epoch:?} is {:?}, not Decoding",
                    host.status
                ));
            }
            if host.pending_seed != Some(l.seed) || state.position != l.position {
                return Err(format!(
                    "draft_step: lane {epoch:?} (seed {:?}, position {}) disagrees with the root's plan (seed {}, position {})",
                    host.pending_seed, state.position, l.seed, l.position
                ));
            }
            let Some(spec) = spec.as_mut() else {
                continue;
            };
            if spec.window.is_some() {
                return Err(format!(
                    "draft_step: lane {epoch:?} has an unconsumed window (commit or abort the step first)"
                ));
            }
            let position = state.position;
            spec.window = Some(PendingWindow {
                route: l.route,
                position,
                seed: l.seed,
                drafts: Vec::new(),
                steps: 0,
                budget: l.budget,
                head_ticket: None,
                target_ticket: None,
                prepared: None,
            });
            let LaneRoute::Batched { k_max } = l.route else {
                continue;
            };
            // I1: the head sits at the lane's frontier.
            if state.position != spec.head.position() {
                return Err(format!(
                    "draft_step: lane {epoch:?} I1: trunk position {} != head position {}",
                    state.position,
                    spec.head.position()
                ));
            }
            // The head's last draft step and its full-accept append reach
            // `position + k_max`; the verify's rows end at the same bound.
            let end = position
                .checked_add(k_max + 1)
                .ok_or_else(|| "draft_step: position overflow".to_string())?;
            spec.head
                .ensure_mapped_capacity(gpu, end)
                .map_err(|e| format!("draft_step: map head arenas of {epoch:?}: {e}"))?;
            state
                .ensure_mapped_capacity(gpu, end)
                .map_err(|e| format!("draft_step: map trunk arenas of {epoch:?}: {e}"))?;
            let ticket = lane_call(bundle.with_lane(
                state,
                Some((&mut spec.head, &mut spec.draft)),
                |b| b.mtp_snapshot(gpu),
            ))
            .map_err(|e| format!("draft_step: head snapshot of {epoch:?}: {e}"))?;
            if let Some(window) = spec.window.as_mut() {
                window.head_ticket = Some(crate::mtp_spec::HeadTicket::Local(ticket));
            }
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
    fn verify_device(&mut self, plan: &VerifyPlan) -> Result<(Vec<Vec<u32>>, Vec<Option<MtpWindow>>, usize), String> {
        let Self {
            bundle,
            gpu,
            store,
            ..
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

        // Arm the ring, then take the target ticket, of every Verify lane.
        for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
            if vl.rows.len() == 1 {
                continue;
            }
            let epoch = vl.key.request_epoch;
            let rows = vl.rows.len();
            let Qwen4LaneState { state, spec, .. } = &mut **lane;
            let window = spec
                .as_mut()
                .and_then(|spec| spec.window.as_mut())
                .ok_or_else(|| format!("Verify lane {epoch:?} has no window"))?;
            if !matches!(window.route, LaneRoute::Batched { .. })
                || window.seed != vl.rows[0]
                || window.drafts != vl.rows[1..]
                || window.position != state.position
            {
                return Err(format!("Verify lane {epoch:?} disagrees with its window"));
            }
            if window.target_ticket.is_some() {
                return Err(format!(
                    "Verify lane {epoch:?} already holds a target ticket"
                ));
            }
            if state.row_capture_rows() < rows {
                return Err(format!(
                    "Verify lane {epoch:?}: the ring holds {} rows, the verify needs {rows}",
                    state.row_capture_rows()
                ));
            }
            // Armed before the snapshot: the GDN states then need no copy.
            state.row_capture_armed = true;
            let ticket = lane_call(bundle.with_lane(state, None, |b| b.snapshot(gpu)));
            match ticket {
                Ok(ticket) => {
                    window.target_ticket = Some(crate::mtp_spec::TargetTicket::Local(ticket))
                }
                Err(error) => {
                    state.row_capture_armed = false;
                    return Err(format!("target snapshot of {epoch:?}: {error}"));
                }
            }
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
        let mut naive: Vec<Option<MtpWindow>> = Vec::with_capacity(plan.lanes.len());
        let mut first = 0usize;
        for (vl, lane) in plan.lanes.iter().zip(lanes.iter_mut()) {
            let rows = vl.rows.len();
            let mut lane_picks: Vec<u32> = (0..rows).map(|j| pick_at(top1_host, first + j)).collect();
            let mut window = None;
            if lane.host.sampling.is_some() {
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

    fn admit_lane(&mut self, admit: &crate::mesh_lane_protocol::LaneAdmit) -> Result<(), RankError> {
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
        let epochs: Vec<RequestEpoch> = round.ops.iter().map(|o| o.key.request_epoch).collect();
        let (mut lanes, spec_store) = store.draft_parts(&epochs).map_err(rank_error)?;
        let spec_store = spec_store
            .ok_or_else(|| rank_error("draft_step: batched window in a store without MTP lanes"))?;
        let slot_bytes = Gpu::TOPK8_OUT_BYTES;
        // Every still-drafting lane enqueues its draft step, in plan order.
        for (slot, (op, lane)) in round.ops.iter().zip(lanes.iter_mut()).enumerate() {
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
            let i = flight
                .start
                .lanes
                .iter()
                .position(|l| l.key == op.key)
                .ok_or_else(|| rank_error("draft op for a lane outside the plan"))?;
            let k_max = flight.start.lanes[i].max_depth;
            let Qwen4LaneState { state, spec, .. } = &mut **lane;
            let spec = spec
                .as_mut()
                .ok_or_else(|| rank_error("draft_step: drafting lane lost its MTP state"))?;
            let LaneSpec {
                head,
                draft,
                window,
                ..
            } = spec;
            let (token, margin) = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                let token = b.mtp_draft_finish(top)?;
                Ok::<_, crate::bundle::BundleError>((token, b.mtp_draft_margin()))
            }))
            .map_err(|e| rank_error(format!("draft_step: draft finish {}: {e}", round.round)))?;
            let window = window
                .as_mut()
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
                });
                continue;
            }
            let done = flight.prefix[i] * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH || j + 1 >= k_max;
            proposals.push(DraftProposal {
                key: op.key,
                token,
                keep_going: !done,
                drop_last: false,
            });
        }
        flight.head_calls += round.ops.len();
        result.proposals = Some(proposals);
        Ok(result)
    }

    fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
        let mut result = self.session.on_verify(plan)?;
        let (picks, naive, row_groups) = self.verify_device(plan).map_err(rank_error)?;
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
        for (i, (d, lane)) in decision.lanes.iter().zip(lanes.iter_mut()).enumerate() {
            let vl = &verify.lanes[i];
            let rows = vl.rows.len();
            let epoch = vl.key.request_epoch;
            let lane_picks = &flight.picks[i];
            let route = flight.start.lanes[i].route;
            if lane.spec.is_some() {
                if rows > 1 {
                    let config = spec_config
                        .ok_or_else(|| rank_error("Verify row in a store without MTP lanes"))?;
                    let hidden = hidden
                        .ok_or_else(|| rank_error("Verify row without a hidden capture"))?;
                    let prepared = prepare_lane_window(
                        bundle,
                        gpu,
                        &mut *lane,
                        config,
                        hidden,
                        first,
                        lane_picks,
                        skip_rollback,
                    )
                    .map_err(rank_error)?;
                    let window = &prepared.result;
                    if window.committed != d.emitted || window.accepted != d.accepted {
                        return Err(rank_error(format!(
                            "Verify lane {epoch:?}: the window finished {:?} (accepted {}) but the root decided {:?} (accepted {})",
                            window.committed, window.accepted, d.emitted, d.accepted
                        )));
                    }
                    if let Some(naive) = &flight.naive[i] {
                        if window.committed != naive.committed || window.accepted != naive.accepted
                        {
                            return Err(rank_error(format!(
                                "sampled Verify lane {epoch:?}: the finish committed {:?} (accepted {}) \
                                 but the naive window is {:?} (accepted {})",
                                window.committed, window.accepted, naive.committed, naive.accepted
                            )));
                        }
                    }
                    if let Some(w) = lane.spec.as_mut().and_then(|s| s.window.as_mut()) {
                        w.prepared = Some(prepared);
                    }
                } else if route == LaneRoute::ArSynced {
                    let hidden = hidden
                        .ok_or_else(|| rank_error("ArSynced row without a hidden capture"))?;
                    ar_synced_head(bundle, gpu, &mut *lane, hidden, first).map_err(rank_error)?;
                }
            }
            if rows == 1 && d.emitted != lane_picks[..1] {
                return Err(rank_error(format!(
                    "AR lane {epoch:?}: the root decided {:?}, the target picked {:?}",
                    d.emitted, lane_picks
                )));
            }
            first += rows;
        }
        flight.decision = Some(decision.clone());
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
                if let Err(error) = publish_lane_window(bundle, gpu, lane, prepared) {
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
        let head_append_rows = decision
            .lanes
            .iter()
            .zip(&self.start.lanes)
            .filter(|(_, l)| l.head != crate::mesh_lane_protocol::HeadState::Free)
            .map(|(d, _)| d.consumed_rows)
            .sum();
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
) -> Result<Vec<LaneDraft>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let mut proto = store.take_proto()?;
    let result = plan_and_draft_with(bundle, gpu, store, &mut proto, requests);
    store.put_proto(proto);
    result
}

fn plan_and_draft_with(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    proto: &mut LaneProto,
    requests: &[(RequestEpoch, usize)],
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
        .map(|&(_, epoch, budget, sampling)| LaneRequest {
            epoch,
            remaining: budget + 1,
            context_left: budget + 1,
            copy_candidates: Vec::new(),
            forced: None,
            sampling,
        })
        .collect();
    let budgets = WindowBudgets {
        max_lanes: store.max_lanes(),
        verify_rows_per_lane: LANE_MAX_ROWS,
        max_rows: store.max_lanes() * LANE_MAX_ROWS,
        copy_take_cap: 0,
        lab: store.routes(),
        pp_microbatch_lanes: None,
    };
    let LaneProto { root, rank, flight } = &mut *proto;
    let start = root
        .plan_window(None, &lane_requests, &budgets)
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
    let sealed = root
        .seal_verify()
        .map_err(|e| format!("draft_step: {e}"))?;
    let mut out = Vec::with_capacity(requests.len());
    for (epoch, _) in requests {
        let i = start
            .lanes
            .iter()
            .position(|l| &l.key.request_epoch == epoch)
            .ok_or_else(|| format!("draft_step: {epoch:?} is not in the plan"))?;
        let route = start.lanes[i].route;
        let drafts = match route {
            LaneRoute::Batched { .. } => sealed.lanes[i].rows[1..].to_vec(),
            _ => Vec::new(),
        };
        if matches!(route, LaneRoute::Batched { .. }) {
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

