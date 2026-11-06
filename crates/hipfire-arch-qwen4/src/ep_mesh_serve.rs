// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Serve-daemon surface of the EP native-MTP lanes ([`crate::ep_mesh_lanes`]).
//!
//! The serve driver (`hipfire-generate::fn_batch`) stages the EP lane store
//! with native-MTP lanes (`stage_lanes_with_copy`), arms this adapter once
//! ([`Qwen4Bundle::ep_mesh_lanes_arm`]) and then runs **decode** through
//! [`Qwen4Bundle::ep_mesh_decode_step`] — one root-planned window per
//! scheduler tick for every chosen decoding lane, replayed concurrently on
//! every rank — while **prefill** still runs through the ordinary lane
//! executor (a Prefill-only plan is identical on every rank).
//!
//! The adapter owns what the oracle's driver owns: the [`EpMeshLanes`] root,
//! the lazy mesh admission of a lane at its first decode step (the prompt-
//! completing pick is the seed), mesh release, the forced-seed mirror and the
//! rank-0 lane statistics the done event reports. Nothing here chooses a
//! route: the root plans on merged rank evidence (DESIGN §3).
//!
//! Sampled lanes (`serve.batch_sampled`, neutral penalties) run on the mesh
//! too: the lane's [`LaneSampling`] becomes the root's per-request sampler
//! ([`EpMeshLanes::admit_lane_sampled`]) and every window is accepted by the
//! Leviathan rule ([`EpMeshLanes::step_sampled`]): rank 0 reads the head's
//! candidates and its target logits rows, the root alone draws.

use crate::bundle::Qwen4Bundle;
use crate::ep_mesh_lanes::{EpMeshLanes, MeshLaneAdmission, MeshStepError};
use crate::lane::{LaneSampling, LaneSpecStats, LaneStatus};
use crate::mesh_lane_protocol::{ConfirmOutcome, LaneRequest, ProposalSource, SamplingMode};
use hipfire_runtime::slot_batch::{RequestAdvance, RequestEpoch};
use hipfire_runtime::spec::SpecRequestConfig;
use rdna_compute::Gpu;
use std::time::{Duration, Instant};

/// The root-side sampler configuration of a sampled serve lane: the lane's
/// AR sampler (temperature, top-p, top-k, min-p) with neutral penalties, the
/// only kind a batch-sampled lane carries.
/// Whether a lane may verify in the wide pass: only an UNSEEDED SAMPLED lane. A greedy
/// lane (no sampling) and a sampled lane whose request carried an explicit seed keep
/// the row-exact groups, so they reproduce across batch compositions
/// (`HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY` still gates the wide pass itself).
pub(crate) fn lane_wide_eligible(sampling: Option<&LaneSampling>) -> bool {
    sampling.is_some_and(|s| !s.seeded)
}

fn sampler_config(sampling: &LaneSampling) -> SpecRequestConfig {
    SpecRequestConfig {
        temp: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p.unwrap_or(0.0),
        rng_seed: u64::from(sampling.seed),
        ..SpecRequestConfig::default()
    }
}

/// The longest host gap between two mesh decode steps that still counts as
/// part of the step's price (emission, scheduling and admission work of one
/// tick, not an idle loop).
pub const MESH_PRICE_GAP: Duration = Duration::from_millis(100);

/// One decoding lane of a mesh step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDecodeRequest {
    pub epoch: RequestEpoch,
    /// Tokens the request may still emit, the pending seed included (`>= 1`;
    /// `1` while emitter-forced tokens are queued: an AR window). The root
    /// alone derives the window's depth from it: native drafts up to the
    /// store's `k` and copy candidates up to the copy take cap, both below
    /// `tokens_left` and the lane's context.
    pub tokens_left: usize,
}

/// One lane's published window, ready for the driver's emission loop.
#[derive(Clone, Debug)]
pub struct MeshDecoded {
    /// The window's seed (the lane's pending seed before the step): the
    /// first forwarded token.
    pub seed: u32,
    pub advance: RequestAdvance,
}

/// What a mesh step reports besides the advances (receipts).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshStepFacts {
    /// Lanes in the step.
    pub lanes: usize,
    /// Windows by final source: native drafts, copy candidates, AR.
    pub native_windows: usize,
    pub copy_windows: usize,
    pub ar_windows: usize,
    /// Drafts offered / accepted over every lane.
    pub offered: usize,
    pub accepted: usize,
    /// Tokens emitted over every lane.
    pub emitted: usize,
    /// The step's `lane_switch` decision: native windows (`true`), or every
    /// lane on the one-token `ArSynced` AR window (`false`; the lane counters
    /// book those as draft-less native windows, the facts as `ar_windows`).
    pub native_mode: bool,
}

impl Qwen4Bundle {
    /// Arm the mesh lane transaction owner for the staged EP spec store, on rank 0's `gpu`
    /// (the all-rows MoE rule the root seals its plans with is read from it).
    /// Idempotent refusal: an already armed bundle errors. The serve driver
    /// owns request termination (EOS/stop/length), so the root never closes
    /// a lane and drafts through an EOS draft, as the single-card route.
    pub fn ep_mesh_lanes_arm(
        &mut self,
        gpu: &Gpu,
        topology_generation: u64,
    ) -> Result<(), String> {
        if self.mesh_lanes.is_some() {
            return Err("EP mesh lanes are already armed".to_string());
        }
        if self.ep.is_none() {
            return Err("EP mesh lanes need an attached EP mesh".to_string());
        }
        let store = self
            .lanes
            .as_ref()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        if store.spec().is_none() {
            return Err("EP mesh lanes need native-MTP lanes".to_string());
        }
        let mut mesh = EpMeshLanes::new(
            gpu,
            self,
            topology_generation,
            crate::mtp_lanes::ep_mesh_lane_floor_enabled(),
        )?;
        mesh.set_driver_terminates(true);
        self.mesh_lanes = Some(Box::new(mesh));
        Ok(())
    }

    /// Whether decode runs through the mesh lane transaction.
    pub fn ep_mesh_lanes_armed(&self) -> bool {
        self.mesh_lanes.is_some()
    }

    /// Whether the mesh has been quarantined by an aborted step (reload to
    /// recover).
    pub fn ep_mesh_lanes_quarantined(&self) -> Option<String> {
        self.mesh_lanes
            .as_ref()
            .and_then(|mesh| mesh.root().quarantined().map(ToString::to_string))
    }

    /// A prefill step ran since the previous mesh step: its interval is not a
    /// decode price.
    pub(crate) fn ep_mesh_note_prefill(&mut self) {
        if let Some(mesh) = self.mesh_lanes.as_mut() {
            mesh.note_prefill();
        }
    }

    /// Mirror a forced seed on the root (the ranks' tables already took it).
    pub(crate) fn ep_mesh_force_seed(
        &mut self,
        epoch: &RequestEpoch,
        token: u32,
    ) -> Result<(), String> {
        match self.mesh_lanes.as_mut() {
            Some(mesh) if mesh.has_lane(*epoch) => mesh.force_seed(*epoch, token),
            _ => Ok(()),
        }
    }

    /// Release `epoch` on the root and every rank's session (host only). A
    /// lane the mesh never admitted is a no-op.
    pub(crate) fn ep_mesh_release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        let Some(mut mesh) = self.mesh_lanes.take() else {
            return Ok(());
        };
        let result = if mesh.has_lane(*epoch) {
            mesh.release_lane_host(self, *epoch)
        } else {
            Ok(())
        };
        self.mesh_lanes = Some(mesh);
        result
    }

    /// The lane's counters as the done event reports them: the rank-0 store's
    /// window statistics with the mesh root's retirement.
    pub fn lane_spec_stats_for(&self, epoch: &RequestEpoch) -> Option<LaneSpecStats> {
        let store = self.lanes.as_ref()?;
        match self.mesh_lanes.as_ref() {
            Some(mesh) => {
                let lane = store.lane(epoch)?;
                let mut stats = lane.spec.as_ref()?.stats;
                stats.retired = mesh.root().lane_retired(*epoch).unwrap_or(stats.retired);
                Some(stats)
            }
            None => store.lane_spec_stats(epoch),
        }
    }

    /// One decode step of `requests` on the EP mesh: the root plans every
    /// lane's window (route, depth, source), the ranks replay it
    /// concurrently, the root decides, every rank repairs and publishes. A
    /// lane enters the mesh at its first step (its prompt is done and its
    /// pending seed is the prompt-completing pick). Returns one decoded
    /// window per request, in request order.
    ///
    /// On `Err` no output was published. A preflight refusal leaves the mesh
    /// usable; an abort quarantines it (every later step errors with the same
    /// record).
    pub fn ep_mesh_decode_step(
        &mut self,
        gpu: &mut Gpu,
        requests: &[MeshDecodeRequest],
    ) -> Result<(Vec<MeshDecoded>, MeshStepFacts), String> {
        let mut mesh = self
            .mesh_lanes
            .take()
            .ok_or_else(|| "EP mesh lanes are not armed".to_string())?;
        // Step-level concurrency switch with hysteresis: native windows while
        // the step carries at most `max_active` lanes (the measured win is at
        // low concurrency), AR windows from `max_active + 2` lanes up; in
        // between the previous decision stands. An AR window is a one-token
        // window the root plans as `ArSynced` (the head stays aligned), so a
        // lane crossing back loses no state.
        self.mesh_native_on = crate::lane_switch::native_mode(
            requests.len(),
            self.mesh_native_on,
            crate::lane_switch::max_active(),
        );
        let switched: Vec<MeshDecodeRequest>;
        let requests = if self.mesh_native_on {
            requests
        } else {
            switched = requests
                .iter()
                .map(|r| MeshDecodeRequest {
                    epoch: r.epoch,
                    tokens_left: 1,
                })
                .collect();
            &switched[..]
        };
        // The previous step's price is the wall from its start to this
        // step's plan. A serve loop that idled between the two (no decoding
        // lane, a long admission) would train the cost model on that stall:
        // such an interval is flagged like a prefill interval (unpriced).
        if self
            .mesh_last_end
            .is_some_and(|end| end.elapsed() > MESH_PRICE_GAP)
        {
            mesh.note_prefill();
        }
        let native_mode = self.mesh_native_on;
        let result =
            self.mesh_decode_step_with(&mut mesh, gpu, requests)
                .map(|(decoded, facts)| {
                    (
                        decoded,
                        MeshStepFacts {
                            native_mode,
                            ..facts
                        },
                    )
                });
        self.mesh_last_end = Some(Instant::now());
        self.mesh_lanes = Some(mesh);
        if result.is_err() {
            for request in requests {
                self.poison_lane(&request.epoch);
            }
        }
        result
    }

    fn mesh_decode_step_with(
        &mut self,
        mesh: &mut EpMeshLanes,
        gpu: &mut Gpu,
        requests: &[MeshDecodeRequest],
    ) -> Result<(Vec<MeshDecoded>, MeshStepFacts), String> {
        if requests.is_empty() {
            return Ok((Vec::new(), MeshStepFacts::default()));
        }
        let (max_seq, routes, eos) = {
            let store = self
                .lanes
                .as_ref()
                .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
            let spec = store
                .spec()
                .ok_or_else(|| "EP mesh lanes need native-MTP lanes".to_string())?;
            (
                self.state.max_seq_len,
                store.routes(),
                spec.end_of_turn.unwrap_or(self.config.eos_token_id),
            )
        };
        let spec_k = self
            .lanes
            .as_ref()
            .and_then(|store| store.spec())
            .map(|spec| spec.k)
            .ok_or_else(|| "EP mesh lanes need native-MTP lanes".to_string())?;
        // Validate every lane and admit the ones the mesh has not seen.
        let mut seeds: Vec<u32> = Vec::with_capacity(requests.len());
        let mut lane_requests: Vec<LaneRequest> = Vec::with_capacity(requests.len());
        let mut retired_before: Vec<bool> = Vec::with_capacity(requests.len());
        // `HIPFIRE_QWEN4_LANE_DEVICE_SAMPLE`: a sampled lane's verify rows are drawn on
        // the sampler owner's device (one lane-wide Leviathan launch against the drafts'
        // q, continuing the lane's xorshift32 stream from its prompt-completing draw);
        // `=0` reads the rows to the root's host sampler.
        let device_sample = gpu.flags.qwen4_lane_device_sample_enabled()
            && hipfire_runtime::lane_draw::lane_draws_supported(gpu);
        for request in requests {
            let (admission, sampling) = {
                let store = self
                    .lanes
                    .as_ref()
                    .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
                let lane = store.lane(&request.epoch).ok_or_else(|| {
                    format!("mesh step: unknown or stale epoch {:?}", request.epoch)
                })?;
                if lane.host.status != LaneStatus::Decoding {
                    return Err(format!(
                        "mesh step: lane {:?} is {:?}, not Decoding",
                        request.epoch, lane.host.status
                    ));
                }
                let sampling = lane.host.sampling;
                let seed = lane.host.pending_seed.ok_or_else(|| {
                    format!("mesh step: lane {:?} has no pending seed", request.epoch)
                })?;
                let admission = (!mesh.has_lane(request.epoch)).then(|| MeshLaneAdmission {
                    epoch: request.epoch,
                    position: lane.state.position,
                    seed,
                    // A `min_tokens` lane's windows run through an EOS.
                    eos: (!lane.host.ignore_eos).then_some(eos),
                    // The device draws continue the lane's AR stream where the prompt-
                    // completing draw left it (a restart at the request seed would reuse
                    // that draw's uniform for the first decode row).
                    rng_seed: sampling.map_or(request.epoch.request_tag, |s| {
                        if device_sample {
                            u64::from(lane.host.rng)
                        } else {
                            u64::from(s.seed)
                        }
                    }),
                    prompt: lane.host.prompt.clone(),
                    // An AR-only lane of a mixed store joins the root as an AR lane.
                    spec_k: lane.spec.is_some().then_some(spec_k),
                    copy_capable: store.lane_copy_capable(&request.epoch),
                });
                (admission, sampling)
            };
            if let Some(admission) = admission {
                match &sampling {
                    Some(sampling) => {
                        mesh.admit_lane_sampled(self, gpu, admission, &sampler_config(sampling))?
                    }
                    None => mesh.admit_lane(self, gpu, admission)?,
                }
            }
            let (_, seed) = mesh
                .root()
                .lane_position(request.epoch)
                .ok_or_else(|| format!("mesh step: the root lost lane {:?}", request.epoch))?;
            let position = self
                .lanes
                .as_ref()
                .and_then(|store| store.lane(&request.epoch))
                .map(|lane| lane.state.position)
                .ok_or_else(|| format!("mesh step: lane {:?} vanished", request.epoch))?;
            // The root bounds the window's drafts by the tokens the request
            // may still emit and by the positions the context has left.
            let context_left = max_seq.saturating_sub(position);
            if context_left == 0 {
                return Err(format!(
                    "mesh step: lane {:?} has no context left",
                    request.epoch
                ));
            }
            seeds.push(seed);
            retired_before.push(mesh.root().lane_retired(request.epoch).unwrap_or(false));
            lane_requests.push(LaneRequest {
                // Unseeded sampled lanes may verify in the wide pass.
                wide_verify: lane_wide_eligible(sampling.as_ref()),
                epoch: request.epoch,
                remaining: request.tokens_left.max(1),
                context_left,
                forced: None,
                sampling: if sampling.is_some() {
                    SamplingMode::Leviathan
                } else {
                    SamplingMode::Greedy
                },
            });
        }
        let budgets = mesh.window_budgets(self, routes)?;
        let report = mesh
            .step_sampled(self, gpu, &lane_requests, &budgets)
            .map_err(|error| match error {
                MeshStepError::Aborted(record) => {
                    format!(
                        "EP mesh lane step aborted (mesh quarantined, reload to recover): {record}"
                    )
                }
                other => format!("EP mesh lane step: {other}"),
            })?;
        // Publish: the rank-0 store mirrors the root (ranks already did at
        // `publish_commit`); statistics follow `commit_with_proto`.
        let mut facts = MeshStepFacts {
            lanes: requests.len(),
            ..MeshStepFacts::default()
        };
        let mut decoded = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            let out = report
                .output
                .lanes
                .iter()
                .find(|lane| lane.key.request_epoch == request.epoch)
                .ok_or_else(|| format!("mesh step: {:?} has no published window", request.epoch))?;
            let retired_now = mesh.root().lane_retired(request.epoch).unwrap_or(false);
            let lane = self
                .lanes
                .as_mut()
                .and_then(|store| store.lane_mut(&request.epoch))
                .ok_or_else(|| format!("mesh step: lane {:?} vanished", request.epoch))?;
            if let Some(spec) = lane.spec.as_mut() {
                let stats = &mut spec.stats;
                match out.source.source {
                    _ if out.forced => stats.forced_windows += 1,
                    ProposalSource::Copy => {
                        stats.copy_windows += 1;
                        stats.copy_drafts += out.source.offered as u64;
                        stats.copy_accepted += out.source.accepted as u64;
                    }
                    ProposalSource::Native => {
                        stats.windows += 1;
                        stats.drafts += out.source.offered as u64;
                        stats.accepted += out.source.accepted as u64;
                        stats.record_depths(out.source.offered, out.source.accepted);
                    }
                    ProposalSource::None if retired_before[index] => stats.ar_windows += 1,
                    ProposalSource::None => stats.windows += 1,
                }
                match out.source.confirm {
                    ConfirmOutcome::Confirmed => stats.copy_confirmed += 1,
                    ConfirmOutcome::Missed => stats.copy_missed += 1,
                    ConfirmOutcome::NotAsked => {}
                }
                stats.retired = retired_now;
            }
            match out.source.source {
                ProposalSource::Native => facts.native_windows += 1,
                ProposalSource::Copy => facts.copy_windows += 1,
                ProposalSource::None => facts.ar_windows += 1,
            }
            facts.offered += out.source.offered;
            facts.accepted += out.source.accepted;
            facts.emitted += out.emitted.len();
            let position = lane.state.position;
            decoded.push(MeshDecoded {
                seed: seeds[index],
                advance: RequestAdvance {
                    epoch: request.epoch,
                    committed_ids: out.emitted.clone(),
                    committed_position: position,
                    accepted_drafts: out.source.accepted,
                    verified_rows: out.source.offered + 1,
                    finish: (position >= lane.state.max_seq_len).then(|| "length".to_string()),
                },
            });
        }
        Ok((decoded, facts))
    }
}

#[cfg(test)]
mod wide_route_tests {
    use super::*;

    fn sampling(seeded: bool) -> LaneSampling {
        LaneSampling {
            temperature: 0.8,
            top_p: 0.95,
            top_k: None,
            min_p: None,
            seed: 12345,
            seeded,
        }
    }

    /// Greedy and seeded sampled requests land on the row-exact route; only an
    /// unseeded sampled request is wide-eligible.
    #[test]
    fn seeded_and_greedy_lanes_are_row_exact_unseeded_sampled_is_wide() {
        assert!(!lane_wide_eligible(None));
        assert!(!lane_wide_eligible(Some(&sampling(true))));
        assert!(lane_wide_eligible(Some(&sampling(false))));
    }
}
