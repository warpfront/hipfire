// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The family contract of the engine loop (G12) and the one runner that
//! executes it on any [`DeviceMesh`].
//!
//! A family declares, it does not orchestrate: its typed step program
//! ([`MeshFamily::step_graph`] over its manifest), its per-epoch
//! [`MeshStepBindings`], its prompt schedule (session boundaries, coprocessor
//! pair groups) and its session state adapter. [`MeshRunner`] owns everything
//! else: the split plans, epoch identity, provisioning of a one-owner mesh,
//! output buffers on the mesh's output owner, submit/drain, fault recovery and
//! the attested reset. Nothing here names a family or a GPU architecture.

use crate::device_mesh::DeviceMesh;
use crate::mesh_placement::{AdmittedMesh, AdmittedRank, BandPlacement};
use crate::mesh_split::{split_program, SplitPlan};
use crate::multi_gpu::executor::{ForwardEpoch, MeshRoute, MeshStepBindings, MeshTicket, Result};
use crate::multi_gpu::transport::EdgeId;
use crate::multi_gpu::Gpus;
use crate::weight_manifest::{plan_manifest, StateEntry, WeightEntry};
use hipfire_dispatch::pipeline::step_graph::StepGraph;
use rdna_compute::{DType, GpuTensor};
use std::sync::atomic::{AtomicU64, Ordering};

/// The output rows one epoch's head writes on the mesh's output owner: the
/// final row's logits, and optionally its device argmax.
#[derive(Clone, Copy)]
pub struct EpochHead<'a> {
    pub logits: &'a GpuTensor,
    pub top1: Option<&'a GpuTensor>,
}

/// One epoch a family binds: rows at the family's position on `route`.
pub struct EpochInput<'a> {
    pub tokens: &'a [u32],
    pub route: MeshRoute,
    pub route_rows: usize,
    /// `None` commits state only.
    pub head: Option<EpochHead<'a>>,
    /// `Some(row)`: the one Decode token is the device argmax of `row` as
    /// the previous epoch left it (`tokens` holds a placeholder).
    pub device_token: Option<&'a GpuTensor>,
    /// Transport epochs of the band boundaries, in logical band order.
    pub edge_epochs: Vec<u64>,
}

/// What a model family supplies to the engine loop.
pub trait MeshFamily {
    /// Typed step program of one output mode: trunk only, with the LM head,
    /// or with the head and its device argmax.
    fn step_graph(&self, head: bool, top1: bool) -> Result<StepGraph>;
    /// The manifest the program's resources are placed from.
    fn manifest(&self) -> (&[WeightEntry], &[StateEntry]);
    fn n_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// Numerical-policy identity of this family's epochs.
    fn policy_id(&self) -> u64 {
        1
    }
    /// Whether the family can run its epochs on `gpus` right now; an `Err`
    /// names why (the caller keeps its other loop).
    fn admit(&self, gpus: &Gpus) -> Result<()>;
    /// Committed rows of the live state.
    fn position(&self) -> usize;

    /// Begin `prompt` with `reused` rows restored (session snapshot or live
    /// continuation) or, at 0, a reset state.
    fn begin_prompt(&mut self, gpus: &mut Gpus, prompt: &[u32], reused: usize) -> Result<()>;
    /// Rows of the next prompt epoch from [`Self::position`]: the family's
    /// schedule unit (session boundary, forward chunk, coprocessor pair group).
    fn next_prompt_rows(&mut self, gpus: &mut Gpus, prompt: &[u32]) -> Result<usize>;
    /// The prompt epoch ending at `end` drained: capture a session boundary
    /// there, and at the prompt end seal the prompt state.
    fn prompt_rows_done(&mut self, gpus: &mut Gpus, prompt: &[u32], end: usize) -> Result<()>;
    /// After a failed prompt epoch: `Some(start)` when the family contained
    /// the fault (it latched the faulting route off) and the prompt reruns
    /// from `start`, `None` when the failure stands.
    fn take_prompt_rerun(&mut self, _prompt: &[u32]) -> Option<usize> {
        None
    }
    /// Best effort: map state ahead for `new_tokens` decode rows.
    fn reserve_decode(&mut self, _gpus: &mut Gpus, _new_tokens: usize) {}

    /// Whether a decode epoch can take its token from the device argmax of
    /// the previous row ([`EpochInput::device_token`]), so greedy decode
    /// never waits for the host between tokens.
    fn device_token_decode(&self) -> bool {
        false
    }
    /// The token the last submitted device-token decode epoch ran, known on
    /// the host once the epoch was submitted.
    fn decoded_token(&self) -> Result<u32> {
        Err("this family has no device-token decode".into())
    }

    /// Prepare capacity for an epoch of `rows` on `route` at the family's
    /// position; returns its routing rows.
    fn prepare_epoch(&mut self, gpus: &mut Gpus, route: MeshRoute, rows: usize) -> Result<usize>;
    /// Run `run` with this family's bindings of `input`.
    fn with_bindings(
        &mut self,
        input: EpochInput<'_>,
        run: &mut dyn FnMut(&mut dyn MeshStepBindings) -> Result<()>,
    ) -> Result<()>;

    /// The family's multi-request decode epoch, when this load staged one:
    /// decoding requests leave the resident state for lanes of their own, so
    /// the resident state takes the next prompt while they decode.
    fn mesh_lanes(&mut self) -> Option<&mut dyn MeshLanes> {
        None
    }
}

/// One decode epoch over N requests, each in its own state slot (a lane) at
/// its own position. Prompts never run here: they run on the resident state
/// at the singleton's prompt cadence (hybrid state is chunking-sensitive), and
/// a request moves into a lane once it decodes ([`Self::park`]). A lane row
/// is byte-identical to the singleton's decode row of the same state, so a
/// finished request's lane is also its conversation's live state
/// ([`LaneRecord`], [`Self::unpark`]).
pub trait MeshLanes {
    /// Requests the lanes hold at once.
    fn lane_capacity(&self) -> usize;
    /// Lanes no request or record holds.
    fn free_lanes(&self) -> usize;
    /// Lanes one [`Self::step_lanes`] runs at most.
    fn lane_rows(&self) -> usize;
    /// Move the resident request into a free lane as `tag`: it ran `prompt`
    /// and then `ran` (its decoded tokens), its next row runs `seed`, and up
    /// to `remaining` decode rows follow. The resident state is reset. A
    /// finished conversation parks with `remaining` 0: its seed never runs.
    fn park(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        ran: &[u32],
        seed: u32,
        remaining: usize,
    ) -> Result<()>;
    /// Move `tag`'s lane, holding exactly `tokens` of a committed
    /// conversation, back into the resident state as its live state (the
    /// session adopts it); the lane is freed. The resident state must hold
    /// no request.
    fn unpark(&mut self, gpus: &mut Gpus, tag: u64, tokens: &[u32]) -> Result<()>;
    /// One epoch: every `(tag, token, logits)` lane runs `token`; returns, in
    /// order, each lane's argmax of that row ([`DecodeOut::Top1`]) or, with
    /// `logits`, the row itself ([`DecodeOut::Logits`], for the request's
    /// own sampler). On `Err` the step's lanes are poisoned (free them).
    fn step_lanes(&mut self, gpus: &mut Gpus, rows: &[(u64, u32, bool)]) -> Result<Vec<DecodeOut>>;
    /// Free `tag`'s lane (host only; the next park reuses its state).
    fn free_lane(&mut self, tag: u64);
}

/// A committed conversation whose state stays in lane `tag`, holding exactly
/// `tokens`: its next turn continues it live ([`MeshLanes::unpark`]).
pub struct LaneRecord {
    pub tag: u64,
    pub tokens: Vec<u32>,
}

/// What a decode epoch leaves for choosing the next token.
pub enum DecodeOut {
    /// Device argmax of the row.
    Top1(u32),
    /// The row, for a host sampler.
    Logits(Vec<f32>),
    /// The row stays on the device: the next decode epoch takes its argmax.
    Device,
}

static NEXT_LOAD_ID: AtomicU64 = AtomicU64::new(1);

const MODE_TRUNK: usize = 0;
const MODE_LOGITS: usize = 1;
const MODE_TOP1: usize = 2;

/// Engine-side execution state of one loaded model on one mesh: split plans
/// and epoch identity. Host only; lives as long as the load.
pub struct MeshRunner {
    mesh: DeviceMesh,
    plans: Option<[SplitPlan; 3]>,
    load_id: u64,
    request_id: u64,
    next_op: u64,
    /// Submitted, not yet drained epochs of the request in flight, oldest first.
    outstanding: Vec<MeshTicket>,
    /// An epoch submission failed after it may have enqueued work that no
    /// ticket covers: nothing counts as drained until a successful reset.
    unsettled: bool,
    /// A failed device synchronization quarantined the mesh. Persistent for
    /// this runner (the load); the device itself carries the quarantine too,
    /// so every route, load and unload on it refuses until restart.
    quarantined: Option<String>,
    /// Committed conversations kept in the family's lanes, oldest first.
    pub records: Vec<LaneRecord>,
}

/// One request's output buffers on the mesh's output owner.
pub struct MeshRequest {
    owner: usize,
    logits: GpuTensor,
    top1: GpuTensor,
}

impl MeshRunner {
    /// A runner over `mesh` (the topology every request's `Gpus` carries).
    pub fn new(mesh: DeviceMesh) -> Result<Self> {
        let load_id = NEXT_LOAD_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "engine load ids are exhausted")?;
        Ok(Self {
            mesh,
            plans: None,
            load_id,
            request_id: 0,
            next_op: 1,
            outstanding: Vec::new(),
            unsettled: false,
            quarantined: None,
            records: Vec::new(),
        })
    }

    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Why this load's mesh refuses all work (a failed synchronization).
    pub fn quarantined(&self) -> Option<&str> {
        self.quarantined.as_deref()
    }

    /// Record a quarantine `gpus` reports, so it outlives this request's lend.
    fn note_quarantine(&mut self, gpus: &Gpus, error: &str) {
        if gpus.mesh_quarantined() && self.quarantined.is_none() {
            self.quarantined = Some(error.to_string());
        }
    }

    /// Start a request on `gpus`: provision a one-owner mesh (a multi-owner
    /// mesh arrives provisioned by its loader), plan the family's program on
    /// first use, allocate the outputs on the output owner.
    pub fn begin(&mut self, family: &dyn MeshFamily, gpus: &mut Gpus) -> Result<MeshRequest> {
        if gpus.mesh.epoch() != self.mesh.epoch() {
            return Err("the request's devices carry a different mesh than the runner".into());
        }
        if let Some(reason) = &self.quarantined {
            return Err(format!("engine mesh quarantined until restart: {reason}"));
        }
        family.admit(gpus)?;
        if gpus.mesh_output_owner().is_none() {
            provision_one_owner(gpus, family.n_layers())?;
        }
        if self.plans.is_none() {
            let placement = BandPlacement::uniform(&self.mesh, family.n_layers())?;
            let (weights, state) = family.manifest();
            let planned = plan_manifest(weights, state, &self.mesh, family.n_layers(), &placement)?;
            let plan = |head, top1| -> Result<SplitPlan> {
                split_program(&family.step_graph(head, top1)?, &planned, &self.mesh)
                    .map_err(|e| e.to_string())
            };
            self.plans = Some([plan(false, false)?, plan(true, false)?, plan(true, true)?]);
        }
        self.request_id = self
            .request_id
            .checked_add(1)
            .ok_or("engine request ids are exhausted")?;
        let owner = gpus
            .mesh_output_owner()
            .ok_or("the mesh has no output owner")?;
        let gpu = &mut gpus.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        let logits = gpu
            .zeros(&[family.vocab_size()], DType::F32)
            .map_err(|e| e.to_string())?;
        let top1 = match gpu.zeros(&[std::mem::size_of::<u32>()], DType::Raw) {
            Ok(top1) => top1,
            Err(error) => {
                let _ = gpu.free_tensor(logits);
                return Err(error.to_string());
            }
        };
        Ok(MeshRequest {
            owner,
            logits,
            top1,
        })
    }

    /// Release a request's outputs once every epoch that writes them drained;
    /// returns the drain verdict. A failed drain leaks them: work in flight
    /// may still write them.
    pub fn end(&mut self, request: MeshRequest, gpus: &mut Gpus) -> Result<()> {
        // ponytail: the two output tensors leak per failed request; reclaim
        // them after the rollback's attested reset if failures become common.
        if let Err(error) = self.settle(gpus) {
            std::mem::forget(request);
            return Err(error);
        }
        let Some(gpu) = gpus.devices.get_mut(request.owner) else {
            std::mem::forget(request);
            return Err("the mesh's devices are gone".into());
        };
        let _ = gpu.free_tensor(request.logits);
        let _ = gpu.free_tensor(request.top1);
        Ok(())
    }

    /// Drain every outstanding epoch with one synchronization of each device.
    /// An empty ticket list is no proof of an idle device after a failed
    /// submission (`unsettled`): that refuses until the reset drains it.
    pub fn settle(&mut self, gpus: &mut Gpus) -> Result<()> {
        if self.unsettled {
            return Err("an epoch failed mid-submission; the device is not drained".into());
        }
        // A quarantined device was never proven idle: the outputs stay leaked.
        if let Some(reason) = &self.quarantined {
            return Err(format!("engine mesh quarantined until restart: {reason}"));
        }
        if self.outstanding.is_empty() {
            return Ok(());
        }
        let result = gpus.drain_mesh_epochs(std::mem::take(&mut self.outstanding));
        if let Err(error) = &result {
            self.note_quarantine(gpus, error);
        }
        result
    }

    /// Run the next prompt epoch; returns its end and, at the prompt end,
    /// the final row's logits.
    pub fn prompt_epoch(
        &mut self,
        request: &MeshRequest,
        family: &mut dyn MeshFamily,
        gpus: &mut Gpus,
        prompt: &[u32],
    ) -> Result<(usize, Option<Vec<f32>>)> {
        let start = family.position();
        let rows = family.next_prompt_rows(gpus, prompt)?;
        let end = start
            .checked_add(rows)
            .filter(|&end| rows > 0 && end <= prompt.len())
            .ok_or_else(|| format!("prompt epoch of {rows} rows at {start} leaves the prompt"))?;
        let last = end == prompt.len();
        let head = last.then_some(EpochHead {
            logits: &request.logits,
            top1: None,
        });
        self.epoch(
            family,
            gpus,
            &prompt[start..end],
            MeshRoute::Prefill,
            head,
            None,
        )?;
        self.settle(gpus)?;
        family.prompt_rows_done(gpus, prompt, end)?;
        let logits = if last {
            Some(self.read_logits(request, gpus)?)
        } else {
            None
        };
        Ok((end, logits))
    }

    /// One decode epoch of `token`, or with `None` of the device argmax of
    /// the previous row (a device-token family only); returns the token it
    /// ran. With `top1` and `keep` on a device-token family nothing drains:
    /// the next epoch reads this row's argmax on the device
    /// ([`DecodeOut::Device`]). Otherwise the epoch drains and the row's
    /// argmax (`top1`) or the row comes back.
    pub fn decode_epoch(
        &mut self,
        request: &MeshRequest,
        family: &mut dyn MeshFamily,
        gpus: &mut Gpus,
        token: Option<u32>,
        top1: bool,
        keep: bool,
    ) -> Result<(u32, DecodeOut)> {
        let device = token.is_none().then_some(&request.logits);
        if device.is_some() && !family.device_token_decode() {
            return Err("a host-token decode epoch needs its token".into());
        }
        let placeholder = [token.unwrap_or(0)];
        let keep = top1 && keep && family.device_token_decode();
        let head = EpochHead {
            logits: &request.logits,
            top1: (top1 && !keep).then_some(&request.top1),
        };
        self.epoch(
            family,
            gpus,
            &placeholder,
            MeshRoute::Decode,
            Some(head),
            device,
        )?;
        let ran = match token {
            Some(token) => token,
            None => family.decoded_token()?,
        };
        if keep {
            return Ok((ran, DecodeOut::Device));
        }
        self.settle(gpus)?;
        if !top1 {
            return self
                .read_logits(request, gpus)
                .map(|row| (ran, DecodeOut::Logits(row)));
        }
        let bytes = gpus.devices[request.owner]
            .download_raw_bytes(&request.top1)
            .map_err(|e| e.to_string())?;
        let id = bytes
            .get(..4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or("top-1 readback is short")?;
        Ok((ran, DecodeOut::Top1(id)))
    }

    /// Bring a poisoned mesh back after a contained fault: drain every device
    /// and, for a one-owner mesh, provision it again.
    pub fn recover(&mut self, family: &dyn MeshFamily, gpus: &mut Gpus) -> Result<()> {
        self.outstanding.clear();
        if let Err(error) = gpus.drain_mesh_edges() {
            self.note_quarantine(gpus, &error);
            return Err(error);
        }
        self.unsettled = false;
        if gpus.devices.len() != 1 {
            return Err("a multi-owner mesh is recovered by its loader".into());
        }
        provision_one_owner(gpus, family.n_layers())
    }

    /// The attested rollback of a request: drain every device, reset every
    /// owner's state through the family's bindings, synchronize. A poisoned
    /// one-owner mesh is recovered first.
    pub fn reset(&mut self, family: &mut dyn MeshFamily, gpus: &mut Gpus) -> Result<()> {
        if !gpus.mesh_healthy() {
            self.recover(&*family, gpus)?;
        }
        // The reset drains every device and retires every outstanding epoch.
        self.outstanding.clear();
        let epoch = self.lifecycle_epoch(&*family, gpus);
        let input = EpochInput {
            tokens: &[],
            route: MeshRoute::Prefill,
            route_rows: 0,
            head: None,
            device_token: None,
            edge_epochs: edge_epochs(gpus)?,
        };
        let result = family.with_bindings(input, &mut |bindings| gpus.reset_mesh(bindings, &epoch));
        match &result {
            Ok(()) => self.unsettled = false,
            Err(error) => self.note_quarantine(gpus, error),
        }
        for gpu in gpus.devices.iter_mut() {
            gpu.invalidate_graph_state();
            gpu.replay.invalidate_replay_observation_window();
        }
        result
    }

    fn lifecycle_epoch(&self, family: &dyn MeshFamily, gpus: &Gpus) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: gpus.mesh.epoch(),
            load_id: self.load_id,
            request_id: self.request_id,
            op_id: self.next_op,
            route: MeshRoute::Prefill,
            start_position: family.position(),
            actual_rows: 0,
            route_rows: 0,
            policy_id: family.policy_id(),
        }
    }

    /// Submit one epoch; it stays outstanding until [`Self::settle`].
    fn epoch(
        &mut self,
        family: &mut dyn MeshFamily,
        gpus: &mut Gpus,
        tokens: &[u32],
        route: MeshRoute,
        head: Option<EpochHead<'_>>,
        device_token: Option<&GpuTensor>,
    ) -> Result<()> {
        let route_rows = family.prepare_epoch(gpus, route, tokens.len())?;
        let mode = match head {
            None => MODE_TRUNK,
            Some(EpochHead { top1: None, .. }) => MODE_LOGITS,
            Some(_) => MODE_TOP1,
        };
        let op_id = self.next_op;
        self.next_op = op_id
            .checked_add(1)
            .ok_or("mesh operation ids are exhausted")?;
        let epoch = ForwardEpoch {
            mesh_epoch: gpus.mesh.epoch(),
            load_id: self.load_id,
            request_id: self.request_id,
            op_id,
            route,
            start_position: family.position(),
            actual_rows: tokens.len(),
            route_rows,
            policy_id: family.policy_id(),
        };
        let plan = &self.plans.as_ref().ok_or("the request did not begin")?[mode];
        let input = EpochInput {
            tokens,
            route,
            route_rows,
            head,
            device_token,
            edge_epochs: edge_epochs(gpus)?,
        };
        // A submission may enqueue work and then fail before its ticket exists.
        self.unsettled = true;
        let outstanding = &mut self.outstanding;
        family.with_bindings(input, &mut |bindings| {
            outstanding.push(gpus.submit_mesh_epoch(plan, &epoch, bindings)?);
            Ok(())
        })?;
        self.unsettled = false;
        Ok(())
    }

    fn read_logits(&mut self, request: &MeshRequest, gpus: &mut Gpus) -> Result<Vec<f32>> {
        self.settle(gpus)?;
        gpus.devices[request.owner]
            .download_f32(&request.logits)
            .map_err(|e| e.to_string())
    }
}

/// Transport epochs of `gpus`' band boundaries in logical band order.
fn edge_epochs(gpus: &Gpus) -> Result<Vec<u64>> {
    (0..gpus.devices.len().saturating_sub(1))
        .map(|src| {
            gpus.mesh_edge_epoch(EdgeId {
                src_device: src,
                dst_device: src + 1,
            })
            .ok_or_else(|| format!("band boundary {src}->{} has no transport lease", src + 1))
        })
        .collect()
}

/// Provision a one-device `gpus` as its own one-owner mesh: no edges, the
/// device's launch stream installed (mesh work never runs on the null stream).
fn provision_one_owner(gpus: &mut Gpus, n_layers: usize) -> Result<()> {
    if gpus.devices.len() != 1 || gpus.mesh.n_devices() != 1 {
        return Err("only a one-device mesh is provisioned by the engine".into());
    }
    let gpu = &mut gpus.devices[0];
    gpu.ensure_capture_stream().map_err(|e| e.to_string())?;
    let admitted = AdmittedMesh {
        mesh: gpus.mesh.clone(),
        ranks: vec![AdmittedRank {
            logical: 0,
            device_ordinal: gpu.device_id,
            pci_bus_id: None,
            arch: gpu.arch.clone(),
        }],
        edges: Vec::new(),
        family: "engine".into(),
        numerical_policy: "one-owner".into(),
    };
    let placement = BandPlacement::uniform(&gpus.mesh, n_layers)?;
    // No edge or TP group exists; the capacity only has to be valid.
    gpus.provision_mesh_edges(&admitted, &placement, 16)
}
