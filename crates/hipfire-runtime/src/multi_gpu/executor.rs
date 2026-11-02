// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq

//! Default-off, single-threaded execution of an admitted split schedule.
//!
//! Preparation is CPU-only and validates the entire epoch before any submission.
//! Submission never synchronizes HIP, except the host-staged TP transport
//! (`TpTransport::Host`). Local marks are enqueued frontiers, not a
//! committed model position: only a successful all-device drain acknowledges an
//! epoch. A failed drain permanently quarantines the mesh's leases.

use super::tp_exchange::{tp_exchange_format, TpExchange16};
use super::transport::{self, EdgeDescriptor, EdgeId, EdgeLease, EdgeSpec};
use super::Gpus;
use crate::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use crate::mesh_placement::{AdmittedMesh, BandPlacement, EdgeTransport};
use crate::mesh_split::{CollectiveKind, PlannedCollective, RankItem, SplitPlan};
use hip_bridge::DeviceBuffer;
use hipfire_dispatch::pipeline::step_graph::OpId;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

pub type Result<T, E = String> = std::result::Result<T, E>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshRoute {
    Decode,
    Prefill,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardEpoch {
    pub mesh_epoch: MeshEpoch,
    pub load_id: u64,
    pub request_id: u64,
    pub op_id: u64,
    pub route: MeshRoute,
    pub start_position: usize,
    pub actual_rows: usize,
    pub route_rows: usize,
    pub policy_id: u64,
}

/// Borrowed local activation, never an alias on another rank. `descriptor.epoch`
/// is the provisioned transport epoch: `Gpus::mesh_edge_epoch` for a
/// `BandXfer`, `Gpus::mesh_tp_epoch` for `AllReduce`/`AllReduceAdd`.
///
/// `view` is the collective's operand on `owner`. `into` is the owner's
/// accumulation target (the `into` resource) of an `AllReduceAdd` and `None`
/// for every other collective. TP collectives fold F32 rows only. After an
/// `AllReduceAdd` the operand's contents are unspecified (they depend on the
/// transport); the splitter refuses any later read of it.
///
/// `partial_slot` is `Some(0 | 1)` only when the family alternates two
/// distinct partial buffers on consecutive TP collectives of an epoch (the
/// hand dense-TP route's mixer and FFN partials): the TP2 peer reduction then
/// takes the alternating protocol, which is the one the whole-token graph can
/// capture. Every rank of a group names the same slot. `None` is the ordinary
/// protocol (a partial buffer the next collective reuses) and the only value
/// for non-TP collectives.
pub struct BoundaryIo<'a> {
    pub view: &'a GpuTensor,
    pub into: Option<&'a GpuTensor>,
    pub descriptor: EdgeDescriptor,
    pub partial_slot: Option<usize>,
}

/// Family adapters own state, not GPUs, streams, topology or token decisions.
/// All device callbacks must enqueue on the owner's active stream without host
/// waits. `prepare_epoch` only validates already-prepared capacity and shapes.
pub trait MeshStepBindings {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()>;
    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()>;
    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>>;
    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()>;
    /// Host-only capability: whether this family implements
    /// [`Self::repair_prefix_rank`]. `Gpus::repair_mesh_prefix` refuses a
    /// family without it before the first callback, so the refusal leaves the
    /// mesh healthy instead of poisoning it.
    fn admits_prefix_repair(&self) -> bool {
        true
    }
    fn repair_prefix_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()>;
    fn capture_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId>;
    /// Host-only, pure session-piece probe: confirm this owner could restore
    /// `piece` (still resident, reset/window generation current). Must not
    /// touch a `Gpu`, enqueue work, pin or mutate any state:
    /// `Gpus::restore_mesh_session` probes EVERY owner before the first
    /// `restore_session_rank`, so a failed probe on any owner leaves every
    /// owner untouched.
    fn probe_session_rank(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()>;
    fn restore_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        checkpoint: MeshCheckpointId,
    ) -> Result<()>;
    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, epoch: &ForwardEpoch) -> Result<()>;
    /// Capture this owner's local state on its borrowed `Gpu`. Must return a
    /// ticket minted with `MeshCheckpointId::issue(*key, owner, ..)`.
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId>;
    /// Roll this owner back to `ticket` on its borrowed `Gpu`.
    fn restore_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only fallible pre-commit check; must not mutate state or touch a `Gpu`.
    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only commit/retire of this owner's ticket. Infallible: no GPU
    /// submission, no copies, no fallible allocation.
    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId);
}

/// Generation key shared by every ticket of one checkpoint window. A ticket is
/// only valid for the exact `(mesh, load, reset, request, window)` it was
/// issued under; any bump invalidates every outstanding ticket.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshCheckpointKey {
    pub mesh_epoch: MeshEpoch,
    pub load_gen: u64,
    pub reset_gen: u64,
    pub request_gen: u64,
    pub window_gen: u64,
}

/// Opaque bounded (`Copy`, fixed-size) ticket for one owner's local checkpoint.
///
/// The generic runtime never interprets `local`; families mint it with
/// [`MeshCheckpointId::issue`] and read it back with [`MeshCheckpointId::local`].
/// The owner index is part of the identity so a ticket issued by one owner is
/// never accepted for another (no rank-zero ticket broadcast to all stages).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshCheckpointId {
    key: MeshCheckpointKey,
    owner: usize,
    local: u64,
}

impl MeshCheckpointId {
    /// Mint a ticket for `owner` under `key`. `local` is family-opaque.
    pub fn issue(key: MeshCheckpointKey, owner: usize, local: u64) -> Self {
        Self { key, owner, local }
    }

    pub fn key(&self) -> MeshCheckpointKey {
        self.key
    }

    pub fn owner(&self) -> usize {
        self.owner
    }

    /// Family-local opaque value; meaningful only to the issuing owner.
    pub fn local(&self) -> u64 {
        self.local
    }
}

/// Aggregate record: one ticket per trunk owner (index == owner) plus the
/// optional output-head ticket, whose owner index is `per_owner.len()`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MeshCheckpoint {
    pub per_owner: Vec<MeshCheckpointId>,
    pub head: Option<MeshCheckpointId>,
}

impl MeshCheckpoint {
    /// Callback owner index used for the output head.
    pub fn head_owner(&self) -> usize {
        self.per_owner.len()
    }

    /// Check every ticket's generation key and owner index against `key`.
    /// Pure; runs before any callback in `restore_all`/`finalize_all`.
    pub fn check_shape(&self, key: &MeshCheckpointKey) -> Result<(), MeshLifecycleError> {
        if self.per_owner.is_empty() {
            return Err(MeshLifecycleError::shape(
                0,
                "checkpoint has no trunk owners",
            ));
        }
        let head_owner = self.head_owner();
        let slots = self
            .per_owner
            .iter()
            .enumerate()
            .chain(self.head.iter().map(|id| (head_owner, id)));
        for (owner, id) in slots {
            if id.owner != owner {
                return Err(MeshLifecycleError::shape(
                    owner,
                    "ticket belongs to a different owner",
                ));
            }
            if id.key != *key {
                return Err(MeshLifecycleError::shape(
                    owner,
                    "ticket generation does not match the live window (stale or foreign)",
                ));
            }
        }
        Ok(())
    }
}

/// Which lifecycle step produced a [`MeshLifecycleError`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshLifecyclePhase {
    /// A ticket set was malformed/stale, or a `Gpus` lifecycle call was refused
    /// by mesh admission (stale, poisoned, quarantined, or mismatched identity);
    /// no callback ran and no rollback was possible.
    Shape,
    Checkpoint,
    /// `validate_finalize_rank` failed; no `finalize_rank` ran.
    Validate,
    Restore,
}

/// Aggregate lifecycle failure.
///
/// `owner`/`cause` are the first failure that aborted the operation.
/// `restore_failures` lists every owner whose rollback `restore_rank` also
/// failed (empty when rollback was clean). When `poisoned()` is true the
/// request/model state is not known-good and the caller must poison it.
#[derive(Clone, Debug)]
pub struct MeshLifecycleError {
    pub phase: MeshLifecyclePhase,
    pub owner: usize,
    pub cause: String,
    pub restore_failures: Vec<(usize, String)>,
}

impl MeshLifecycleError {
    fn shape(owner: usize, msg: &str) -> Self {
        Self {
            phase: MeshLifecyclePhase::Shape,
            owner,
            cause: msg.to_string(),
            restore_failures: Vec::new(),
        }
    }

    /// True when rollback could not make every owner consistent again: some
    /// restore failed, or the tickets were unusable so none was attempted.
    pub fn poisoned(&self) -> bool {
        self.phase == MeshLifecyclePhase::Shape || !self.restore_failures.is_empty()
    }
}

impl fmt::Display for MeshLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mesh checkpoint {:?} failed at owner {}: {}",
            self.phase, self.owner, self.cause
        )?;
        if !self.restore_failures.is_empty() {
            write!(f, "; rollback failed on owner(s)")?;
            for (owner, err) in &self.restore_failures {
                write!(f, " {owner} ({err})")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for MeshLifecycleError {}

/// Pure per-owner lifecycle callbacks sequenced by [`checkpoint_all`],
/// [`restore_all`] and [`finalize_all`]. Object-safe and CPU-testable: it holds
/// no `Gpu` and is deliberately separate from [`MeshStepBindings`], whose
/// device-touching methods receive the owner's borrowed `&mut Gpu`. A callback
/// provider resolves logical owner indices to its locally borrowed resources;
/// the head's logical owner is distinct even when it shares the output GPU.
///
/// This trait is the coordinators' CPU-testable seam. The only production
/// implementer is the private `BoundOwners` adapter used by `Gpus`
/// (`checkpoint_mesh`/`restore_mesh`/`finalize_mesh`): it maps trunk owner `i`
/// to `devices[i]` and the head owner to `mesh_output_owner()`, binds the
/// device thread before each GPU-touching call, and delegates to
/// [`MeshStepBindings`]. Families never call the coordinators themselves.
///
/// `owner` is `0..trunk_owners` for trunk stages and `trunk_owners` for the
/// output head. `key` is the live window's generation key.
pub trait MeshCheckpointCallbacks {
    /// Capture this owner's local state. Must return a ticket minted with
    /// `MeshCheckpointId::issue(*key, owner, ..)`. On `Err` the callee leaves
    /// its own state unmodified/rolled back; the coordinator will not restore it.
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId>;

    /// Roll this owner back to `ticket`. Called for every owner on any failure,
    /// in reverse owner order, even if earlier restores failed.
    fn restore_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;

    /// Fallible pre-commit check (ticket still valid, captures drained, ...).
    /// Runs for every owner before any `finalize_rank`. Must not mutate state.
    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;

    /// Host-only commit/retire of this owner's ticket. Infallible: no GPU
    /// submission, no copies, no fallible allocation.
    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId);
}

/// Roll back `tickets` (slot index == owner; `head` at `tickets.len()`) in
/// reverse order. Attempts every slot; returns every failure.
fn restore_slots<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    tickets: &[MeshCheckpointId],
    head: Option<MeshCheckpointId>,
) -> Vec<(usize, String)> {
    let mut failures = Vec::new();
    if let Some(id) = head {
        if let Err(e) = cb.restore_rank(tickets.len(), key, id) {
            failures.push((tickets.len(), e));
        }
    }
    for (owner, id) in tickets.iter().enumerate().rev() {
        if let Err(e) = cb.restore_rank(owner, key, *id) {
            failures.push((owner, e));
        }
    }
    failures
}

/// Capture every trunk owner (ascending) and, when `with_head`, the output
/// head as owner `trunk_owners`. If any capture fails (or returns a ticket not
/// issued for that owner/key), every already-captured owner is restored in
/// reverse order and the original failure is returned.
pub fn checkpoint_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    trunk_owners: usize,
    with_head: bool,
) -> Result<MeshCheckpoint, MeshLifecycleError> {
    if trunk_owners == 0 {
        return Err(MeshLifecycleError::shape(
            0,
            "checkpoint needs at least one trunk owner",
        ));
    }
    let capture = |cb: &mut C, owner: usize| -> Result<MeshCheckpointId> {
        let id = cb.checkpoint_rank(owner, key)?;
        if id.owner != owner || id.key != *key {
            return Err(
                "checkpoint_rank returned a ticket not issued for this owner/window".to_string(),
            );
        }
        Ok(id)
    };

    let mut per_owner = Vec::with_capacity(trunk_owners);
    for owner in 0..trunk_owners {
        match capture(cb, owner) {
            Ok(id) => per_owner.push(id),
            Err(cause) => {
                let restore_failures = restore_slots(cb, key, &per_owner, None);
                return Err(MeshLifecycleError {
                    phase: MeshLifecyclePhase::Checkpoint,
                    owner,
                    cause,
                    restore_failures,
                });
            }
        }
    }
    let head = if with_head {
        match capture(cb, trunk_owners) {
            Ok(id) => Some(id),
            Err(cause) => {
                let restore_failures = restore_slots(cb, key, &per_owner, None);
                return Err(MeshLifecycleError {
                    phase: MeshLifecyclePhase::Checkpoint,
                    owner: trunk_owners,
                    cause,
                    restore_failures,
                });
            }
        }
    } else {
        None
    };
    Ok(MeshCheckpoint { per_owner, head })
}

/// Restore EVERY owner (head first, then trunk descending), continuing past
/// individual failures. Tickets are validated for shape before any callback.
/// The first failing owner is `owner`/`cause`; all failures are in
/// `restore_failures`.
pub fn restore_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    checkpoint.check_shape(key)?;
    let failures = restore_slots(cb, key, &checkpoint.per_owner, checkpoint.head);
    if failures.is_empty() {
        return Ok(());
    }
    let (owner, cause) = failures[0].clone();
    // `restore_failures` carries every failure including the primary one.
    Err(MeshLifecycleError {
        phase: MeshLifecyclePhase::Restore,
        owner,
        cause,
        restore_failures: failures,
    })
}

/// Commit the checkpoint. Phase 1 validates EVERY owner (trunk ascending, then
/// head), continuing past failures and reporting the first; phase 2 runs the infallible
/// `finalize_rank` on every owner in the same order. If validation fails, no
/// finalize runs and EVERY owner is restored (reverse order, continuing past
/// restore errors); the validation failure is returned with any rollback
/// failures attached. The success path performs no allocation.
pub fn finalize_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    checkpoint.check_shape(key)?;
    let head_owner = checkpoint.head_owner();

    // Validate EVERY owner (trunk ascending, head last) even after a failure;
    // remember the first failure. No finalize runs unless all validate.
    let mut failure: Option<(usize, String)> = None;
    for (owner, id) in checkpoint.per_owner.iter().enumerate() {
        if let Err(e) = cb.validate_finalize_rank(owner, key, *id) {
            failure.get_or_insert((owner, e));
        }
    }
    if let Some(id) = checkpoint.head {
        if let Err(e) = cb.validate_finalize_rank(head_owner, key, id) {
            failure.get_or_insert((head_owner, e));
        }
    }
    if let Some((owner, cause)) = failure {
        let restore_failures = restore_slots(cb, key, &checkpoint.per_owner, checkpoint.head);
        return Err(MeshLifecycleError {
            phase: MeshLifecyclePhase::Validate,
            owner,
            cause,
            restore_failures,
        });
    }

    for (owner, id) in checkpoint.per_owner.iter().enumerate() {
        cb.finalize_rank(owner, key, *id);
    }
    if let Some(id) = checkpoint.head {
        cb.finalize_rank(head_owner, key, id);
    }
    Ok(())
}

/// Acknowledgement capability. Not cloneable and bound to one `Gpus` instance.
#[derive(Debug)]
pub struct MeshTicket {
    issuer: u64,
    epoch: ForwardEpoch,
}

struct OwnedEdge {
    lease: EdgeLease,
    max_bytes: usize,
}

/// How a pure-TP group moves partials. Decided once at provisioning; there is
/// no per-call route choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpTransport {
    /// Every rank pair is byte-qualified peer, or one device under emulation.
    /// Each rank pulls every partial with the kernel copy `qualify_peer`
    /// proved (never `hipMemcpyPeer`, which writes nothing into allocations
    /// made after peer enable) and folds `p0 + p1 + …` locally.
    Peer,
    /// Some pair is not peer-qualified: the hand dense-TP route's host-staged
    /// rank-order reduction (`Gpus::all_reduce_sum_f32_host`). It synchronizes
    /// the rank streams, the one exception to "submission never waits".
    Host,
}

/// The provisioned pure-TP group: every rank, in rank order.
#[derive(Clone, Copy, Debug)]
struct TpGroup {
    transport: TpTransport,
    epoch: u64,
    max_bytes: usize,
}

/// One rank's `TpTransport::Peer` buffers, allocated at provisioning.
struct TpScratch {
    /// `ranks` F32 slots of `max_bytes`: slot `q` holds rank `q`'s partial,
    /// slot 0 then accumulates the rank-order sum.
    stage: GpuTensor,
    /// The 16-byte zero owner table of `transport::copy_bytes`.
    owners: GpuTensor,
}

#[derive(Default)]
pub(super) struct MeshExecutionState {
    edges: HashMap<EdgeId, OwnedEdge>,
    mesh_epoch: Option<MeshEpoch>,
    output_owner: Option<usize>,
    /// `Some` exactly for a provisioned pure-TP mesh.
    tp: Option<TpGroup>,
    /// Index == rank; empty unless `tp` is `TpTransport::Peer`.
    tp_scratch: Vec<TpScratch>,
    issuer: u64,
    last_op: Option<u64>,
    scope: Option<(u64, u64)>,
    /// Successfully issued, not yet drained epochs, oldest first. They all
    /// belong to one request and one mesh epoch with strictly increasing op
    /// ids (`admit`). A failed issue never appears here: it poisons the state.
    outstanding: Vec<ForwardEpoch>,
    poisoned: bool,
    quarantined: bool,
}

impl MeshExecutionState {
    fn poison(&mut self) {
        self.poisoned = true;
        for edge in self.edges.values() {
            edge.lease.poison();
        }
    }

    /// Pure admission check; mutates nothing. A new epoch may join undrained
    /// earlier epochs only for the same request (mesh epoch, load and numerical
    /// policy are pinned by `mesh_epoch`/`scope`) with a strictly larger op id.
    fn admit(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.mesh_epoch != Some(epoch.mesh_epoch) || self.poisoned || self.quarantined {
            return Err("mesh epoch is stale, busy, poisoned, or unprovisioned".into());
        }
        if self
            .outstanding
            .last()
            .is_some_and(|last| last.request_id != epoch.request_id)
        {
            return Err("mesh epoch is busy with another request's undrained epochs".into());
        }
        if self.last_op.is_some_and(|last| epoch.op_id <= last)
            || self
                .scope
                .is_some_and(|scope| scope != (epoch.load_id, epoch.policy_id))
        {
            return Err(
                "mesh operation id is not monotonic or load/numerical policy changed".into(),
            );
        }
        Ok(())
    }

    /// Side-effect-free ticket validation. The tickets must be exactly the
    /// outstanding epochs, oldest first, issued by this state: a device drain
    /// retires every outstanding epoch, so a subset, reordering, stale or
    /// foreign ticket is refused before any device is touched.
    fn validate_tickets(&self, tickets: &[MeshTicket]) -> Result<()> {
        if self.quarantined
            || tickets.is_empty()
            || tickets.len() != self.outstanding.len()
            || tickets
                .iter()
                .zip(&self.outstanding)
                .any(|(ticket, epoch)| ticket.issuer != self.issuer || ticket.epoch != *epoch)
        {
            return Err(
                "mesh tickets do not name exactly the outstanding healthy transport epochs in order"
                    .into(),
            );
        }
        Ok(())
    }

    /// Mesh health/identity admission shared by every lifecycle call; pure.
    /// Requires the exact provisioned mesh epoch, a healthy (not poisoned, not
    /// quarantined) mesh, and a valid owner/device mapping for `owners`
    /// trunk owners (the output owner, which hosts the head, must name a
    /// device). Unlike [`Self::admit`] this never looks at op ids: lifecycle
    /// calls legitimately follow an already executed epoch.
    fn admit_lifecycle_mesh(&self, mesh_epoch: MeshEpoch, owners: usize) -> Result<()> {
        if self.mesh_epoch != Some(mesh_epoch) || self.poisoned || self.quarantined {
            return Err(
                "mesh epoch is stale, poisoned, quarantined, or unprovisioned; no lifecycle \
                 callback was run"
                    .into(),
            );
        }
        match self.output_owner {
            Some(output) if owners != 0 && output < owners => Ok(()),
            _ => Err("mesh owner/device mapping is invalid; no lifecycle callback was run".into()),
        }
    }

    /// Load/request identities that are known to the executor: the pinned load
    /// (set by the first submission) and, while epochs are outstanding, the one
    /// request they belong to. Unknown identities are not invented.
    fn check_lifecycle_identity(&self, load_id: u64, request_id: u64) -> Result<()> {
        if self.scope.is_some_and(|(load, _)| load != load_id) {
            return Err("lifecycle call names a different model load than the mesh".into());
        }
        if self
            .outstanding
            .last()
            .is_some_and(|last| last.request_id != request_id)
        {
            return Err(
                "lifecycle call names a different request than the undrained epochs".into(),
            );
        }
        Ok(())
    }

    /// Admission for lifecycle calls that carry a [`ForwardEpoch`]: mesh
    /// health/identity, pinned load and numerical policy, outstanding request.
    fn admit_lifecycle_epoch(&self, epoch: &ForwardEpoch, owners: usize) -> Result<()> {
        self.admit_lifecycle_mesh(epoch.mesh_epoch, owners)?;
        if self
            .scope
            .is_some_and(|scope| scope != (epoch.load_id, epoch.policy_id))
        {
            return Err("lifecycle epoch changes the pinned load/numerical policy".into());
        }
        self.check_lifecycle_identity(epoch.load_id, epoch.request_id)
    }

    /// Admission for checkpoint-window lifecycle calls that carry a
    /// [`MeshCheckpointKey`] (`load_gen`/`request_gen` are the load and request
    /// ids of the epochs the window runs under).
    fn admit_lifecycle_key(&self, key: &MeshCheckpointKey, owners: usize) -> Result<()> {
        self.admit_lifecycle_mesh(key.mesh_epoch, owners)?;
        self.check_lifecycle_identity(key.load_gen, key.request_gen)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundaryMeta {
    descriptor: EdgeDescriptor,
    address: usize,
    extent: usize,
    capacity: usize,
    transport_epoch: u64,
    /// `AllReduceAdd` accumulation target `(address, extent)`; `None` otherwise.
    into: Option<(usize, usize)>,
    /// The operand's alternating partial slot ([`BoundaryIo::partial_slot`]).
    partial_slot: Option<usize>,
}

/// Small internal seam: CPU tests exercise the very same prepare/issue/drain
/// algorithms as HIP, without constructing fake `Gpu` values.
trait DrainBackend {
    fn rank_count(&self) -> usize;
    fn synchronize(&mut self, owner: usize) -> Result<()>;
    fn poison(&mut self);
    fn quarantine(&mut self);
}

trait Backend: DrainBackend {
    fn ready(&self, owner: usize) -> bool;
    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryMeta>;
    fn run_ops(&mut self, owner: usize, epoch: &ForwardEpoch, ops: &[OpId]) -> Result<()>;
    fn transfer(
        &mut self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
        meta: BoundaryMeta,
        publish: bool,
    ) -> Result<()>;
    /// Enqueue one `AllReduce`/`AllReduceAdd` on every rank of its group;
    /// `metas` are its prepared boundaries in group order.
    fn reduce(
        &mut self,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
        metas: &[BoundaryMeta],
    ) -> Result<()>;
    fn marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()>;
    /// Host-side validation hook run before any enqueue of `epoch`.
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()>;
}

#[derive(Debug)]
enum Action {
    Ops {
        owner: usize,
        start: usize,
        end: usize,
    },
    Transfer {
        owner: usize,
        collective: usize,
        meta: BoundaryMeta,
        publish: bool,
    },
    /// One group reduction; boundaries are `Prepared::metas[start..start + group.len()]`.
    Reduce {
        collective: usize,
        start: usize,
    },
    Marks(usize),
}

#[derive(Debug)]
struct Prepared {
    actions: Vec<Action>,
    ops: Vec<OpId>,
    issued: Vec<u64>,
    /// Every collective's validated boundaries, one per participant in group order.
    metas: Vec<BoundaryMeta>,
}

/// How the walk schedules a collective.
#[derive(Clone, Copy)]
enum Route {
    /// Point-to-point PP edge `(src, dst)`.
    Band(usize, usize),
    /// Reduction over every rank of a pure-TP mesh.
    Group,
}

fn endpoints(collective: &PlannedCollective, ranks: usize) -> Result<(usize, usize)> {
    let CollectiveKind::BandXfer { src, dst } = collective.kind else {
        return Err("collective is not a BandXfer".into());
    };
    if src >= ranks || dst >= ranks || src == dst || collective.group != [src, dst] {
        return Err("BandXfer must name exactly its two logical PP owners".into());
    }
    Ok((src, dst))
}

fn route(collective: &PlannedCollective, ranks: usize) -> Result<Route> {
    if let CollectiveKind::BandXfer { .. } = collective.kind {
        let (src, dst) = endpoints(collective, ranks)?;
        return Ok(Route::Band(src, dst));
    }
    // Pure TP: the group is every rank, ascending, and the sum is taken in that
    // order (`x + (p0 + p1 + …)`, the hand dense-TP routes' numeric contract).
    if ranks < 2 || !collective.group.iter().copied().eq(0..ranks) {
        return Err("TP collective group must be every mesh rank in ascending rank order".into());
    }
    Ok(Route::Group)
}

/// Group reductions fold F32 rows, and every operand and accumulation target
/// must be distinct memory: under emulation all ranks share one device, so an
/// aliased binding would fold a rank into itself.
fn validate_group(metas: &[BoundaryMeta], add: bool) -> Result<()> {
    let bytes = metas[0].descriptor.bytes;
    if metas[0].descriptor.dtype_bytes != 4 {
        return Err("TP collectives reduce F32 activations only".into());
    }
    // The alternating protocol has two slots, and every rank must be in the
    // same one: the TP2 reduction reads both ranks' partials under one slot.
    let slot = metas[0].partial_slot;
    if slot.is_some_and(|slot| slot > 1) || metas.iter().any(|meta| meta.partial_slot != slot) {
        return Err("TP collective participants must name the same partial slot, 0 or 1".into());
    }
    let mut spans = Vec::with_capacity(metas.len() * 2);
    for meta in metas {
        spans.push((meta.address, bytes));
        match meta.into {
            Some((address, extent))
                if add && address != 0 && address % 16 == 0 && extent >= bytes =>
            {
                spans.push((address, bytes));
            }
            None if !add => {}
            _ => {
                return Err(
                    "AllReduceAdd needs one accumulation target per rank holding the \
                     operand bytes; AllReduce takes none"
                        .into(),
                )
            }
        }
    }
    spans.sort_unstable();
    if spans
        .windows(2)
        .any(|pair| pair[0].0 + pair[0].1 > pair[1].0)
    {
        return Err("TP collective operands and accumulation targets overlap".into());
    }
    Ok(())
}

fn validate_boundary(meta: BoundaryMeta, epoch: &ForwardEpoch) -> Result<()> {
    let d = meta.descriptor;
    if d.epoch != meta.transport_epoch || d.op != epoch.op_id || d.rows != epoch.actual_rows {
        return Err("boundary descriptor epoch/op/rows mismatch".into());
    }
    if !matches!(d.dtype_bytes, 2 | 4)
        || d.bytes == 0
        || d.bytes > meta.capacity
        || d.bytes > meta.extent
        || d.bytes % d.rows != 0
        || (d.bytes / d.rows) % 16 != 0
        || meta.address == 0
        || meta.address % 16 != 0
        || d.bytes / 16 > i32::MAX as usize
    {
        return Err(
            "boundary descriptor exceeds lease or has invalid live extent/alignment".into(),
        );
    }
    Ok(())
}

fn prepare(plan: &SplitPlan, epoch: &ForwardEpoch, backend: &impl Backend) -> Result<Prepared> {
    if plan.mesh_epoch != epoch.mesh_epoch {
        return Err("split plan belongs to a stale mesh epoch".into());
    }
    if epoch.actual_rows == 0
        || epoch.route_rows < epoch.actual_rows
        || epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .is_none()
    {
        return Err("invalid forward epoch row geometry".into());
    }
    if plan.ranks.is_empty() || plan.ranks.len() != backend.rank_count() {
        return Err("split plan rank count does not match devices".into());
    }
    let ranks = plan.ranks.len();
    let mut seen_ops = HashSet::new();
    let mut routes = Vec::with_capacity(plan.collectives.len());
    let mut starts = Vec::with_capacity(plan.collectives.len());
    let mut metas = Vec::new();
    for (index, c) in plan.collectives.iter().enumerate() {
        if c.id.0 as usize != index || c.operand.0 as usize >= plan.resources.len() {
            return Err("invalid collective id or resource".into());
        }
        let route = route(c, ranks)?;
        let start = metas.len();
        for &owner in &c.group {
            let meta = backend.boundary(owner, c, epoch)?;
            validate_boundary(meta, epoch)?;
            metas.push(meta);
        }
        let own = &metas[start..];
        if own.iter().any(|meta| meta.descriptor != own[0].descriptor) {
            return Err("collective participants' boundary descriptors differ".into());
        }
        match route {
            Route::Band(..) if own.iter().any(|meta| meta.into.is_some()) => {
                return Err("BandXfer boundary names an accumulation target".into());
            }
            Route::Band(..) => {}
            Route::Group => {
                validate_group(own, matches!(c.kind, CollectiveKind::AllReduceAdd { .. }))?
            }
        }
        routes.push(route);
        starts.push(start);
    }
    let mut occurrences = vec![0usize; metas.len()];
    let mut item_count = 0;
    for (owner, rank) in plan.ranks.iter().enumerate() {
        if rank.device != owner || !backend.ready(owner) {
            return Err("rank program device mismatch or missing non-null launch stream".into());
        }
        for item in &rank.items {
            match *item {
                RankItem::Op(id) => {
                    if !seen_ops.insert((owner, id)) {
                        return Err("duplicate rank op in split plan".into());
                    }
                }
                RankItem::Collective(id) => {
                    let index = id.0 as usize;
                    let c = plan
                        .collectives
                        .get(index)
                        .ok_or("invalid rank collective id")?;
                    let side = c
                        .group
                        .iter()
                        .position(|&device| device == owner)
                        .ok_or("collective placed on a nonparticipant rank")?;
                    occurrences[starts[index] + side] += 1;
                }
            }
        }
        item_count += rank.items.len();
    }
    if occurrences.iter().any(|&count| count != 1) {
        return Err("collective must appear exactly once on each participant".into());
    }
    // The fused reduction replaces the Accumulate; running both adds twice.
    for c in &plan.collectives {
        if let CollectiveKind::AllReduceAdd { replaces, .. } = c.kind {
            if c.group
                .iter()
                .any(|&owner| seen_ops.contains(&(owner, replaces)))
            {
                return Err("AllReduceAdd's replaced Accumulate is also scheduled as an op".into());
            }
        }
    }
    // Build the topological walk before submission. One action per rank per pass
    // allows chunk overlap while receiving each bank before it can be reused.
    let mut prepared = Prepared {
        actions: Vec::with_capacity(item_count + ranks),
        ops: Vec::with_capacity(seen_ops.len()),
        issued: Vec::new(),
        metas,
    };
    let mut cursors = vec![0; ranks];
    let mut published = vec![false; plan.collectives.len()];
    let mut received = vec![false; plan.collectives.len()];
    let mut busy_edges = HashSet::new();
    let mut remaining = item_count;
    while remaining != 0 {
        let before = remaining;
        for (owner, rank) in plan.ranks.iter().enumerate() {
            if cursors[owner] == rank.items.len() {
                continue;
            }
            match rank.items[cursors[owner]] {
                RankItem::Op(_) => {
                    let start = prepared.ops.len();
                    while let Some(RankItem::Op(id)) = rank.items.get(cursors[owner]) {
                        prepared.ops.push(*id);
                        cursors[owner] += 1;
                        remaining -= 1;
                    }
                    prepared.actions.push(Action::Ops {
                        owner,
                        start,
                        end: prepared.ops.len(),
                    });
                }
                RankItem::Collective(id) => {
                    let index = id.0 as usize;
                    match routes[index] {
                        Route::Band(src, dst) => {
                            let edge = (src, dst);
                            let publish = owner == src;
                            if publish {
                                if busy_edges.contains(&edge) {
                                    continue;
                                }
                                published[index] = true;
                                busy_edges.insert(edge);
                            } else {
                                if !published[index] {
                                    continue;
                                }
                                received[index] = true;
                                busy_edges.remove(&edge);
                            }
                            prepared.actions.push(Action::Transfer {
                                owner,
                                collective: index,
                                meta: prepared.metas[starts[index] + usize::from(!publish)],
                                publish,
                            });
                            cursors[owner] += 1;
                            remaining -= 1;
                        }
                        Route::Group => {
                            // Enqueue once every rank has reached it, so every
                            // rank's producers precede the reduction.
                            let group = &plan.collectives[index].group;
                            if group.iter().any(|&device| {
                                plan.ranks[device].items.get(cursors[device])
                                    != Some(&RankItem::Collective(id))
                            }) {
                                continue;
                            }
                            prepared.actions.push(Action::Reduce {
                                collective: index,
                                start: starts[index],
                            });
                            for &device in group {
                                cursors[device] += 1;
                                remaining -= 1;
                                if device != owner
                                    && cursors[device] == plan.ranks[device].items.len()
                                {
                                    prepared.actions.push(Action::Marks(device));
                                }
                            }
                        }
                    }
                }
            }
            if cursors[owner] == rank.items.len() {
                prepared.actions.push(Action::Marks(owner));
            }
        }
        if before == remaining {
            return Err("cyclic or inconsistent collective schedule".into());
        }
    }
    // Empty owners still participate in acknowledgements and marks.
    for (owner, rank) in plan.ranks.iter().enumerate() {
        if rank.items.is_empty() {
            prepared.actions.push(Action::Marks(owner));
        }
    }
    debug_assert!(routes
        .iter()
        .zip(&received)
        .all(|(route, received)| matches!(route, Route::Group) || *received));
    prepared
        .issued
        .resize(prepared.actions.len().div_ceil(64), 0);
    Ok(prepared)
}

fn issue(
    prepared: &mut Prepared,
    plan: &SplitPlan,
    epoch: &ForwardEpoch,
    backend: &mut impl Backend,
) -> Result<()> {
    for (index, action) in prepared.actions.iter().enumerate() {
        let mask = 1u64 << (index % 64);
        if prepared.issued[index / 64] & mask != 0 {
            backend.poison();
            return Err("mesh epoch item was already issued".into());
        }
        // Mark before invoking user code: failure cannot retry a partial enqueue.
        prepared.issued[index / 64] |= mask;
        let result = match *action {
            Action::Ops { owner, start, end } => {
                backend.run_ops(owner, epoch, &prepared.ops[start..end])
            }
            Action::Transfer {
                owner,
                collective,
                meta,
                publish,
            } => backend.transfer(owner, &plan.collectives[collective], epoch, meta, publish),
            Action::Reduce { collective, start } => {
                let c = &plan.collectives[collective];
                backend.reduce(c, epoch, &prepared.metas[start..start + c.group.len()])
            }
            Action::Marks(owner) => backend.marks(owner, epoch),
        };
        if let Err(error) = result {
            backend.poison();
            return Err(error);
        }
    }
    Ok(())
}

fn drain(backend: &mut impl DrainBackend) -> Result<()> {
    let mut failure = None;
    for owner in 0..backend.rank_count() {
        if let Err(error) = backend.synchronize(owner) {
            failure.get_or_insert(error);
        }
    }
    if let Some(error) = failure {
        backend.poison();
        backend.quarantine();
        return Err(format!("mesh drain failed; resources quarantined: {error}"));
    }
    Ok(())
}

/// Access to the executor state machine from a backend; HIP and the CPU toy
/// tests drive the very same `submit`/`drain_tickets` through it.
trait HasState {
    fn state(&mut self) -> &mut MeshExecutionState;
}

/// Admit, prepare, enqueue. Validation (admission, `prepare_epoch`, `prepare`)
/// happens before any device enqueue and refuses without poisoning: outstanding
/// earlier epochs stay drainable. Only an enqueue failure (`issue`) poisons, and
/// the failed epoch is never recorded outstanding, so no ticket exists for it.
fn submit<B: Backend + HasState>(
    plan: &SplitPlan,
    epoch: &ForwardEpoch,
    backend: &mut B,
) -> Result<MeshTicket> {
    backend.state().admit(epoch)?;
    backend.prepare_epoch(epoch)?;
    let mut prepared = prepare(plan, epoch, &*backend)?;
    let state = backend.state();
    state.last_op = Some(epoch.op_id);
    state.scope = Some((epoch.load_id, epoch.policy_id));
    issue(&mut prepared, plan, epoch, backend)?;
    let state = backend.state();
    state.outstanding.push(*epoch);
    Ok(MeshTicket {
        issuer: state.issuer,
        epoch: *epoch,
    })
}

/// Validate every ticket before touching a device, synchronize every device
/// once, then retire all outstanding epochs. Failed sync poisons and
/// quarantines (S3 `drain`); a poisoned mesh retires its epochs but reports an
/// error for all of them.
fn drain_tickets<B: DrainBackend + HasState>(
    tickets: &[MeshTicket],
    backend: &mut B,
) -> Result<()> {
    backend.state().validate_tickets(tickets)?;
    drain(backend)?;
    let state = backend.state();
    state.outstanding.clear();
    if state.poisoned {
        return Err("mesh submission failed before drain".into());
    }
    Ok(())
}

/// The production backend. It borrows the whole `Gpus` because TP reductions
/// run on its rank-order reduction primitives and their scratch.
struct HipBackend<'a> {
    gpus: &'a mut Gpus,
    bindings: &'a mut dyn MeshStepBindings,
}

/// Refuse a binding tensor that is not plain device memory of `gpu`.
fn require_local(gpu: &Gpu, tensor: &GpuTensor) -> Result<()> {
    let attributes = gpu
        .hip
        .pointer_get_attributes(&tensor.buf)
        .map_err(|e| e.to_string())?;
    if attributes.device != gpu.device_id
        || hip_bridge::MemoryType::from_raw(attributes.mem_type)
            != Some(hip_bridge::MemoryType::Device)
        || attributes.is_managed != 0
    {
        return Err("boundary activation is not local device memory of its owner".into());
    }
    Ok(())
}

impl DrainBackend for HipBackend<'_> {
    fn rank_count(&self) -> usize {
        self.gpus.devices.len()
    }
    fn synchronize(&mut self, owner: usize) -> Result<()> {
        let gpu = &mut self.gpus.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())
    }
    fn poison(&mut self) {
        self.gpus.mesh_execution.poison();
    }
    fn quarantine(&mut self) {
        self.gpus.mesh_execution.quarantined = true;
        // Reached only after a failed device synchronization (see `drain` and
        // `reset_all`): like `HipDrain`, keep device ownership until teardown
        // because in-flight work may still reference registered VMM/host state.
        std::mem::forget(std::mem::take(&mut self.gpus.devices));
    }
}

/// The identity of a boundary binding, from its transport state and the
/// binding's buffers; no device calls. `require_local` validation is the
/// preparation pass's job (`HipBackend::boundary`), once per boundary.
fn boundary_meta(gpus: &Gpus, c: &PlannedCollective, io: &BoundaryIo<'_>) -> Result<BoundaryMeta> {
    let state = &gpus.mesh_execution;
    let (capacity, transport_epoch) = if let CollectiveKind::BandXfer { .. } = c.kind {
        let (src, dst) = endpoints(c, gpus.devices.len())?;
        let edge = state
            .edges
            .get(&EdgeId {
                src_device: src,
                dst_device: dst,
            })
            .ok_or("BandXfer has no provisioned lease")?;
        (edge.max_bytes, edge.lease.epoch())
    } else {
        let tp = state
            .tp
            .ok_or("TP collective on a mesh without a provisioned TP group")?;
        (tp.max_bytes, tp.epoch)
    };
    Ok(BoundaryMeta {
        descriptor: io.descriptor,
        address: io.view.buf.as_ptr() as usize,
        extent: io.view.buf.size(),
        capacity,
        transport_epoch,
        into: io
            .into
            .map(|into| (into.buf.as_ptr() as usize, into.buf.size())),
        partial_slot: io.partial_slot,
    })
}

impl Backend for HipBackend<'_> {
    fn ready(&self, owner: usize) -> bool {
        self.gpus.devices[owner].active_stream.is_some()
    }
    fn boundary(
        &self,
        owner: usize,
        c: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryMeta> {
        let io = self.bindings.boundary(owner, c, epoch)?;
        let gpu = &self.gpus.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        require_local(gpu, io.view)?;
        if let Some(into) = io.into {
            require_local(gpu, into)?;
        }
        boundary_meta(self.gpus, c, &io)
    }
    fn run_ops(&mut self, owner: usize, epoch: &ForwardEpoch, ops: &[OpId]) -> Result<()> {
        let gpu = &mut self.gpus.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        self.bindings.run_ops(owner, gpu, epoch, ops)
    }
    fn transfer(
        &mut self,
        owner: usize,
        c: &PlannedCollective,
        epoch: &ForwardEpoch,
        meta: BoundaryMeta,
        publish: bool,
    ) -> Result<()> {
        let io = self.bindings.boundary(owner, c, epoch)?;
        if boundary_meta(self.gpus, c, &io)? != meta {
            return Err("prepared boundary binding changed during submission".into());
        }
        let (src, dst) = endpoints(c, self.gpus.devices.len())?;
        let edge = &self.gpus.mesh_execution.edges[&EdgeId {
            src_device: src,
            dst_device: dst,
        }];
        let gpu = &mut self.gpus.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        if publish {
            edge.lease.publish(gpu, io.view, io.descriptor)
        } else {
            edge.lease.receive(gpu, io.view, io.descriptor)
        }
    }
    fn reduce(
        &mut self,
        c: &PlannedCollective,
        epoch: &ForwardEpoch,
        metas: &[BoundaryMeta],
    ) -> Result<()> {
        let tp = self
            .gpus
            .mesh_execution
            .tp
            .ok_or("TP collective on a mesh without a provisioned TP group")?;
        let count = metas[0].descriptor.bytes / 4;
        let mut partials = Vec::with_capacity(c.group.len());
        let mut targets = Vec::with_capacity(c.group.len());
        for (&owner, meta) in c.group.iter().zip(metas) {
            // Memory attributes were validated when the boundary was prepared;
            // here only the binding's identity is re-checked.
            let io = self.bindings.boundary(owner, c, epoch)?;
            if boundary_meta(self.gpus, c, &io)? != *meta {
                return Err("prepared boundary binding changed during submission".into());
            }
            partials.push(&io.view.buf);
            targets.extend(io.into.map(|into| &into.buf));
        }
        let gpus = &mut *self.gpus;
        match tp.transport {
            TpTransport::Peer => peer_reduce(
                gpus,
                tp.max_bytes,
                &partials,
                &targets,
                metas[0].descriptor.rows,
                count,
                metas[0].partial_slot,
            ),
            TpTransport::Host => host_reduce(gpus, &partials, &targets, count),
        }
    }
    fn marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.bindings.apply_rank_marks(owner, epoch)
    }
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.bindings.prepare_epoch(epoch)
    }
}

fn hip_error(error: hip_bridge::HipError) -> String {
    error.to_string()
}

fn f32_view(buf: &DeviceBuffer, count: usize) -> GpuTensor {
    GpuTensor {
        // SAFETY: a borrowed, validated binding buffer; the view never
        // outlives this enqueue.
        buf: unsafe { buf.alias() },
        shape: vec![count],
        dtype: DType::F32,
    }
}

fn rank_stream(gpu: &Gpu) -> Result<&hip_bridge::Stream> {
    gpu.active_stream
        .as_ref()
        .ok_or_else(|| "TP rank has no launch stream".to_string())
}

/// `TpTransport::Peer`. Every rank pulls every partial into its own slots
/// with the kernel copy `qualify_peer` proved, then folds `p0 + p1 + …` in
/// rank order, so every rank computes the same bytes. `AllReduceAdd` adds the
/// sum to the rank's target (`x + (p0 + p1 + …)`); `AllReduce` (`targets`
/// empty) overwrites the rank's partial with it. Producer and read events order
/// the cross-rank reads on the device streams; nothing waits on the host.
/// `partial_slot` is the group's [`BoundaryIo::partial_slot`], used by the TP2
/// direct-peer route only.
fn peer_reduce(
    gpus: &mut Gpus,
    max_bytes: usize,
    partials: &[&DeviceBuffer],
    targets: &[&DeviceBuffer],
    rows: usize,
    count: usize,
    partial_slot: Option<usize>,
) -> Result<()> {
    // TP2 `AllReduceAdd` across two devices: the hand route's direct-peer add
    // (`Gpus::all_reduce_sum_f32_peer_add_routed`), one launch per rank
    // computing `x + (p0 + p1)` (same fp32 order as the copy-fold below). A
    // `partial_slot` takes its alternating protocol, the one the whole-token
    // graph captures. Same-device (emulated) ranks keep the copy-fold.
    if gpus.devices.len() == 2
        && targets.len() == 2
        && gpus.devices[0].device_id != gpus.devices[1].device_id
    {
        return gpus
            .all_reduce_sum_f32_peer_add_routed(partials, targets, rows, count, partial_slot)
            .map_err(hip_error);
    }
    // TP3/TP4 decode inside a whole-token graph capture: one bf16 peer-push
    // kernel per rank replaces the pack, copies, fold and events below, whose
    // cross-device events cannot be captured. Same rounded, rank-ordered sum.
    if gpus.devices.len() >= 3
        && targets.len() == gpus.devices.len()
        && gpus.devices.iter().all(|gpu| gpu.graphs.capture_mode)
        && gpus.tp_exchange16_executor(rows, count) == Some(TpExchange16::Bf16)
        && gpus.tp_peer_pushn_active()
        && gpus.tp_peer_push_serves(count)
    {
        return gpus
            .tp_peer_push_reduce_captured(partials, targets, count)
            .map_err(hip_error);
    }
    // `HIPFIRE_TP_EXCHANGE` bf16|f16 (or `auto` per rows/TP): the ranks exchange 16-bit partials
    // (`AllReduceAdd` only; the no-target `AllReduce` keeps f32). The same
    // rounded, rank-ordered sum the hand route's `peer_add_routed` adds.
    let fmt = if targets.len() == gpus.devices.len() {
        gpus.tp_exchange16_executor(rows, count)
    } else {
        None
    };
    let Gpus {
        devices,
        rank_barrier_events: produced,
        rank_done_events: read,
        mesh_execution,
        ..
    } = gpus;
    let scratch = &mesh_execution.tp_scratch;
    let n = devices.len();
    if scratch.len() != n || produced.len() != n || read.len() != n {
        return Err("TP peer scratch or events are not provisioned".into());
    }
    let bytes = count * 4;
    let bytes16 = count * 2;
    let rooted = n > 2 && bytes > ROOTED_REDUCE_MIN_BYTES;
    // 1. Each rank's partial is complete at this point of its stream. A
    // 16-bit exchange first packs it into the rank's own slot `r` of its
    // stage, which is what the other ranks pull (the rooted fold reads the
    // root's own f32 partial directly, so the root does not pack).
    for (r, (gpu, event)) in devices.iter_mut().zip(produced.iter()).enumerate() {
        gpu.bind_thread().map_err(hip_error)?;
        if let Some(fmt) = fmt.filter(|_| !(rooted && r == 0)) {
            let own = scratch[r].stage.buf.as_ptr() as usize + r * max_bytes;
            gpu.tp_pack16(
                fmt,
                partials[r].as_ptr() as *const std::ffi::c_void,
                own as *mut std::ffi::c_void,
                count,
            )
            .map_err(hip_error)?;
        }
        gpu.hip
            .event_record(event, Some(rank_stream(gpu)?))
            .map_err(hip_error)?;
    }
    if rooted {
        return peer_reduce_rooted(
            devices, scratch, produced, read, max_bytes, partials, targets, count, fmt,
        );
    }
    // 2. After every producer: pull partial q into slot q, fold into slot 0.
    // The 16-bit fold sums every rank's packed slot straight into the target.
    for (r, (gpu, rank)) in devices.iter_mut().zip(scratch).enumerate() {
        gpu.bind_thread().map_err(hip_error)?;
        for (_, event) in produced.iter().enumerate().filter(|&(q, _)| q != r) {
            gpu.hip
                .stream_wait_event(rank_stream(gpu)?, event)
                .map_err(hip_error)?;
        }
        if let Some(fmt) = fmt {
            let base = rank.stage.buf.as_ptr() as usize;
            let mut sources = [std::ptr::null::<std::ffi::c_void>(); 4];
            for (q, source) in sources.iter_mut().take(n).enumerate() {
                let slot = base + q * max_bytes;
                if q != r {
                    let packed = scratch[q].stage.buf.as_ptr() as usize + q * max_bytes;
                    transport::copy_bytes(gpu, &rank.owners, packed, slot, bytes16)?;
                }
                *source = slot as *const std::ffi::c_void;
            }
            gpu.tp_fold16(
                fmt,
                std::ptr::null_mut(),
                targets[r].as_ptr(),
                std::ptr::null(),
                &sources[..n],
                count,
            )
            .map_err(hip_error)?;
        } else {
            let sum = rank.stage.sub_offset(0, count);
            for (q, partial) in partials.iter().enumerate() {
                let slot = rank.stage.buf.as_ptr() as usize + q * max_bytes;
                transport::copy_bytes(gpu, &rank.owners, partial.as_ptr() as usize, slot, bytes)?;
                if q > 0 {
                    let addend = rank.stage.sub_offset(q * (max_bytes / 4), count);
                    gpu.add_inplace_f32(&sum, &addend).map_err(hip_error)?;
                }
            }
        }
        gpu.hip
            .event_record(&read[r], Some(rank_stream(gpu)?))
            .map_err(hip_error)?;
    }
    // 3. Once no rank still reads this rank's partial, write the result.
    for (r, (gpu, rank)) in devices.iter_mut().zip(scratch).enumerate() {
        gpu.bind_thread().map_err(hip_error)?;
        for (_, event) in read.iter().enumerate().filter(|&(q, _)| q != r) {
            gpu.hip
                .stream_wait_event(rank_stream(gpu)?, event)
                .map_err(hip_error)?;
        }
        if fmt.is_some() {
            // The fold above already added the sum to the target.
            continue;
        }
        let sum = rank.stage.sub_offset(0, count);
        match targets.get(r) {
            Some(target) => {
                let x = f32_view(target, count);
                gpu.add_f32(&x, &sum, &x).map_err(hip_error)?;
            }
            None => transport::copy_bytes(
                gpu,
                &rank.owners,
                sum.buf.as_ptr() as usize,
                partials[r].as_ptr() as usize,
                bytes,
            )?,
        }
    }
    Ok(())
}

/// Smallest payload that takes the rooted peer reduction for three or more
/// ranks. The all-pull path moves `N(N-1)` copies of the payload but keeps
/// every rank symmetric with no second dependency hop, which wins for
/// latency-bound decode rows (one row of the 27B is 20 KB). Prefill payloads
/// (rows x dim x 4 bytes, up to ~40 MB) are bandwidth-bound, where the rooted
/// path's `2(N-1)` copies win. 256 KiB sits far above any decode row and far
/// below the smallest prefill chunk that matters.
const ROOTED_REDUCE_MIN_BYTES: usize = 256 * 1024;

/// Rooted `TpTransport::Peer` reduction (`N > 2`, large payloads). Rank 0
/// pulls partials `1..N` into its slots `1..N`, folds
/// `((p0 + p1) + p2) + …` into slot 0 (`p0` is read in place by the first
/// add, which is the same fp32 sum the all-pull path's copy-then-add forms),
/// and every other rank pulls that finished slot 0 into its own slot 0, then
/// applies it exactly as the all-pull path does. Every rank therefore ends
/// with the same bytes as the all-pull fold.
///
/// Events (the producers' `produced`, plus `read`): `read[0]` is recorded on
/// rank 0 after its last partial read and the fold, so it doubles as "sum
/// ready" for the broadcast pulls and as "rank 0 no longer reads partial `r`"
/// for rank `r`'s next write of its partial. `read[r]` (`r > 0`) is recorded on
/// rank `r` right after it pulled rank 0's slot 0; rank 0 waits those before
/// returning, so the next collective cannot overwrite slot 0 under a reader.
/// Nothing waits on the host and nothing is allocated.
///
/// With a 16-bit exchange (`fmt`) the non-root ranks have already packed their
/// partial into their own stage slot `r`; rank 0 pulls those `2 * count`-byte
/// copies into its slots `1..N` and folds `r(p0) + r(p1) + …` (`p0` rounded in
/// register) into the f32 slot 0. The broadcast of the f32 sum is unchanged.
#[allow(clippy::too_many_arguments)]
fn peer_reduce_rooted(
    devices: &mut [Gpu],
    scratch: &[TpScratch],
    produced: &[hip_bridge::Event],
    read: &[hip_bridge::Event],
    max_bytes: usize,
    partials: &[&DeviceBuffer],
    targets: &[&DeviceBuffer],
    count: usize,
    fmt: Option<TpExchange16>,
) -> Result<()> {
    let bytes = count * 4;
    // 2. Rank 0, after every producer: pull p1.., fold into slot 0.
    {
        let (gpu, rank) = (&mut devices[0], &scratch[0]);
        gpu.bind_thread().map_err(hip_error)?;
        for event in &produced[1..] {
            gpu.hip
                .stream_wait_event(rank_stream(gpu)?, event)
                .map_err(hip_error)?;
        }
        if let Some(fmt) = fmt {
            let base = rank.stage.buf.as_ptr() as usize;
            let mut sources = [std::ptr::null::<std::ffi::c_void>(); 4];
            for q in 1..partials.len() {
                let slot = base + q * max_bytes;
                let packed = scratch[q].stage.buf.as_ptr() as usize + q * max_bytes;
                transport::copy_bytes(gpu, &rank.owners, packed, slot, count * 2)?;
                sources[q - 1] = slot as *const std::ffi::c_void;
            }
            gpu.tp_fold16(
                fmt,
                base as *mut std::ffi::c_void,
                std::ptr::null_mut(),
                partials[0].as_ptr() as *const std::ffi::c_void,
                &sources[..partials.len() - 1],
                count,
            )
            .map_err(hip_error)?;
        } else {
            for (q, partial) in partials.iter().enumerate().skip(1) {
                let slot = rank.stage.buf.as_ptr() as usize + q * max_bytes;
                transport::copy_bytes(gpu, &rank.owners, partial.as_ptr() as usize, slot, bytes)?;
            }
            let sum = rank.stage.sub_offset(0, count);
            let first = rank.stage.sub_offset(max_bytes / 4, count);
            gpu.add_f32(&f32_view(partials[0], count), &first, &sum)
                .map_err(hip_error)?;
            for q in 2..partials.len() {
                let addend = rank.stage.sub_offset(q * (max_bytes / 4), count);
                gpu.add_inplace_f32(&sum, &addend).map_err(hip_error)?;
            }
        }
        gpu.hip
            .event_record(&read[0], Some(rank_stream(gpu)?))
            .map_err(hip_error)?;
    }
    // 3. Every other rank pulls the finished sum into its own slot 0 and
    // applies it; rank 0 applies its own.
    let root_sum = scratch[0].stage.buf.as_ptr() as usize;
    for (r, (gpu, rank)) in devices.iter_mut().zip(scratch).enumerate() {
        gpu.bind_thread().map_err(hip_error)?;
        let sum = rank.stage.sub_offset(0, count);
        if r > 0 {
            gpu.hip
                .stream_wait_event(rank_stream(gpu)?, &read[0])
                .map_err(hip_error)?;
            let slot = sum.buf.as_ptr() as usize;
            transport::copy_bytes(gpu, &rank.owners, root_sum, slot, bytes)?;
            gpu.hip
                .event_record(&read[r], Some(rank_stream(gpu)?))
                .map_err(hip_error)?;
        }
        match targets.get(r) {
            Some(target) => {
                let x = f32_view(target, count);
                gpu.add_f32(&x, &sum, &x).map_err(hip_error)?;
            }
            None => transport::copy_bytes(
                gpu,
                &rank.owners,
                sum.buf.as_ptr() as usize,
                partials[r].as_ptr() as usize,
                bytes,
            )?,
        }
    }
    // 4. Rank 0's slot 0 is free once every other rank has pulled it.
    let gpu = &mut devices[0];
    gpu.bind_thread().map_err(hip_error)?;
    for event in &read[1..] {
        gpu.hip
            .stream_wait_event(rank_stream(gpu)?, event)
            .map_err(hip_error)?;
    }
    Ok(())
}

/// `TpTransport::Host`: exactly the hand dense-TP route without peer access
/// (`qwen35` `dense_tp_allreduce_add`): host-staged rank-order sum into every
/// partial, then `x + sum` on each rank for `AllReduceAdd`.
fn host_reduce(
    gpus: &mut Gpus,
    partials: &[&DeviceBuffer],
    targets: &[&DeviceBuffer],
    count: usize,
) -> Result<()> {
    gpus.all_reduce_sum_f32_host(partials, count)
        .map_err(hip_error)?;
    for ((gpu, partial), target) in gpus.devices.iter_mut().zip(partials).zip(targets) {
        let (sum, x) = (f32_view(partial, count), f32_view(target, count));
        gpu.bind_thread().map_err(hip_error)?;
        gpu.add_f32(&x, &sum, &x).map_err(hip_error)?;
    }
    Ok(())
}

impl HasState for HipBackend<'_> {
    fn state(&mut self) -> &mut MeshExecutionState {
        &mut self.gpus.mesh_execution
    }
}

// ---------------------------------------------------------------------------
// Aggregate lifecycle (S6-W1): `Gpus` owns the owner sequencing. Families only
// implement the per-owner `MeshStepBindings` callbacks.
// ---------------------------------------------------------------------------

/// Per-owner lifecycle operations behind the aggregate coordinators. HIP
/// (`HipBackend`) and the CPU toy tests implement this one seam, and every
/// sequencing, admission and fail-closed rule lives in the generic helpers
/// below, so the tests execute the production algorithm.
///
/// `device` is the index into the admitted device list. For trunk owner `i` it
/// is `i`; for the head owner (`rank_count()`) it is the output owner. The
/// helpers call [`OwnerOps::bind`] before every GPU-touching method.
trait OwnerOps: DrainBackend + HasState {
    /// Bind the calling thread to `device`'s context.
    fn bind(&mut self, device: usize) -> Result<()>;
    fn checkpoint(
        &mut self,
        owner: usize,
        device: usize,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId>;
    fn restore(
        &mut self,
        owner: usize,
        device: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only; touches no `Gpu`.
    fn validate_finalize(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only and infallible; touches no `Gpu`.
    fn finalize(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId);
    /// Host-only; touches no `Gpu`.
    fn admits_prefix_repair(&self) -> bool;
    /// Trunk owner `owner` runs on device `owner`.
    fn repair_prefix(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()>;
    fn capture_session(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<MeshCheckpointId>;
    /// Host-only and pure; touches no `Gpu`.
    fn probe_session(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()>;
    fn restore_session(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()>;
    fn reset(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()>;
}

/// The only production [`MeshCheckpointCallbacks`] implementer. Trunk owner `i`
/// maps to `devices[i]`; the head owner (`devices.len()`) maps to
/// `mesh_output_owner()`. Every GPU-touching callback binds the owner's device
/// thread first; a bind or mapping failure is that owner's callback failure.
struct BoundOwners<'a, B: OwnerOps> {
    backend: &'a mut B,
}

impl<B: OwnerOps> BoundOwners<'_, B> {
    fn device(&mut self, owner: usize) -> Result<usize> {
        let trunk = self.backend.rank_count();
        if owner < trunk {
            return Ok(owner);
        }
        if owner == trunk {
            return match self.backend.state().output_owner {
                Some(output) if output < trunk => Ok(output),
                _ => Err("head owner has no valid mesh output owner device".into()),
            };
        }
        Err(format!("owner {owner} is outside the mesh owners"))
    }

    /// Resolve `owner` to its device and bind the thread to it.
    fn bound(&mut self, owner: usize) -> Result<usize> {
        let device = self.device(owner)?;
        self.backend.bind(device)?;
        Ok(device)
    }
}

impl<B: OwnerOps> MeshCheckpointCallbacks for BoundOwners<'_, B> {
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        let device = self.bound(owner)?;
        self.backend.checkpoint(owner, device, key)
    }

    fn restore_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        let device = self.bound(owner)?;
        self.backend.restore(owner, device, key, ticket)
    }

    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        // The fallible bind/mapping step of the commit lives here, in the
        // validation phase that `finalize_all` runs for every owner before any
        // finalize.
        self.bound(owner)?;
        self.backend.validate_finalize(owner, key, ticket)
    }

    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId) {
        // Finalization is host-only (no `Gpu` argument, no submission) and
        // infallible by contract, so it neither binds nor can fail: binding is
        // fallible and was already exercised for this owner by
        // `validate_finalize_rank`. Ignoring a bind error or panicking here
        // would break that contract.
        self.backend.finalize(owner, key, ticket);
    }
}

/// Fail closed: poison the mesh (and its leases) and quarantine it. Does not
/// forget devices; that is reserved for a failed device synchronization
/// (`DrainBackend::quarantine`).
fn fail_closed(backend: &mut impl OwnerOps) {
    backend.poison();
    backend.state().quarantined = true;
}

/// Refusal before any callback. Reported as `Shape`: no rollback was possible
/// and the tickets cannot be used on this mesh.
fn refused(cause: &str) -> MeshLifecycleError {
    MeshLifecycleError::shape(0, cause)
}

/// An aggregate rollback that could not make every owner consistent again
/// (a restore failed after callbacks ran) leaves state unknown: fail closed.
/// Clean rollbacks and pre-callback refusals do not.
fn close_if_unrecovered<T>(backend: &mut impl OwnerOps, result: &Result<T, MeshLifecycleError>) {
    if let Err(error) = result {
        if error.phase != MeshLifecyclePhase::Shape && !error.restore_failures.is_empty() {
            fail_closed(backend);
        }
    }
}

fn check_checkpoint_fits(
    checkpoint: &MeshCheckpoint,
    owners: usize,
) -> Result<(), MeshLifecycleError> {
    if checkpoint.per_owner.len() != owners {
        return Err(MeshLifecycleError::shape(
            checkpoint.per_owner.len().min(owners),
            "checkpoint trunk owner count does not match the mesh devices",
        ));
    }
    Ok(())
}

fn checkpoint_window<B: OwnerOps>(
    backend: &mut B,
    key: &MeshCheckpointKey,
    with_head: bool,
) -> Result<MeshCheckpoint, MeshLifecycleError> {
    let owners = backend.rank_count();
    backend
        .state()
        .admit_lifecycle_key(key, owners)
        .map_err(|e| refused(&e))?;
    let result = checkpoint_all(
        &mut BoundOwners {
            backend: &mut *backend,
        },
        key,
        owners,
        with_head,
    );
    close_if_unrecovered(backend, &result);
    result
}

fn restore_window<B: OwnerOps>(
    backend: &mut B,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    let owners = backend.rank_count();
    backend
        .state()
        .admit_lifecycle_key(key, owners)
        .map_err(|e| refused(&e))?;
    checkpoint.check_shape(key)?;
    check_checkpoint_fits(&checkpoint, owners)?;
    let result = restore_all(
        &mut BoundOwners {
            backend: &mut *backend,
        },
        key,
        checkpoint,
    );
    close_if_unrecovered(backend, &result);
    result
}

fn finalize_window<B: OwnerOps>(
    backend: &mut B,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    let owners = backend.rank_count();
    backend
        .state()
        .admit_lifecycle_key(key, owners)
        .map_err(|e| refused(&e))?;
    checkpoint.check_shape(key)?;
    check_checkpoint_fits(&checkpoint, owners)?;
    let result = finalize_all(
        &mut BoundOwners {
            backend: &mut *backend,
        },
        key,
        checkpoint,
    );
    close_if_unrecovered(backend, &result);
    result
}

/// Trunk owners ascending, stop at the first failure, then fail closed: the
/// owners hold mixed prefixes and no ticket can reconcile them.
fn repair_prefix_all<B: OwnerOps>(
    backend: &mut B,
    epoch: &ForwardEpoch,
    keep: usize,
    tokens: &[u32],
) -> Result<()> {
    let owners = backend.rank_count();
    backend.state().admit_lifecycle_epoch(epoch, owners)?;
    if !backend.admits_prefix_repair() {
        return Err("mesh prefix repair is not admitted by this family; nothing ran".into());
    }
    for owner in 0..owners {
        let result = backend
            .bind(owner)
            .and_then(|()| backend.repair_prefix(owner, epoch, keep, tokens));
        if let Err(error) = result {
            fail_closed(backend);
            return Err(format!(
                "mesh prefix repair failed on owner {owner}; mesh poisoned and quarantined: {error}"
            ));
        }
    }
    Ok(())
}

/// A session ticket must correspond to the epoch's mesh, load and request.
fn session_key_matches(key: &MeshCheckpointKey, epoch: &ForwardEpoch) -> bool {
    key.mesh_epoch == epoch.mesh_epoch
        && key.load_gen == epoch.load_id
        && key.request_gen == epoch.request_id
}

/// Every owner ascending, stopping at the first failure (reported with its
/// owner; the family releases already captured pieces through its own state).
/// Each returned piece must carry the capturing owner's index and a key that
/// corresponds to `epoch` and to every earlier piece.
fn capture_session_all<B: OwnerOps>(
    backend: &mut B,
    epoch: &ForwardEpoch,
) -> Result<Vec<MeshCheckpointId>> {
    let owners = backend.rank_count();
    backend.state().admit_lifecycle_epoch(epoch, owners)?;
    let mut pieces: Vec<MeshCheckpointId> = Vec::with_capacity(owners);
    for owner in 0..owners {
        let piece = backend
            .bind(owner)
            .and_then(|()| backend.capture_session(owner, epoch))
            .map_err(|e| format!("capture_session_rank failed on owner {owner}: {e}"))?;
        if piece.owner() != owner {
            return Err(format!(
                "capture_session_rank on owner {owner} returned a piece issued for owner {}",
                piece.owner()
            ));
        }
        if !session_key_matches(&piece.key(), epoch)
            || pieces
                .first()
                .is_some_and(|first| first.key() != piece.key())
        {
            return Err(format!(
                "capture_session_rank on owner {owner} returned a piece whose key does not \
                 match the epoch or the other owners' pieces"
            ));
        }
        pieces.push(piece);
    }
    Ok(pieces)
}

fn check_session_pieces(
    pieces: &[MeshCheckpointId],
    epoch: &ForwardEpoch,
    owners: usize,
) -> Result<()> {
    if pieces.len() != owners {
        return Err(format!(
            "session has {} pieces for {owners} mesh owners",
            pieces.len()
        ));
    }
    let key = pieces[0].key();
    if !session_key_matches(&key, epoch) {
        return Err("session pieces do not correspond to the epoch's mesh/load/request".into());
    }
    for (owner, piece) in pieces.iter().enumerate() {
        if piece.owner() != owner {
            return Err(format!(
                "session piece {owner} was issued for owner {}",
                piece.owner()
            ));
        }
        if piece.key() != key {
            return Err(format!(
                "session piece {owner} belongs to a different checkpoint window than piece 0"
            ));
        }
    }
    Ok(())
}

/// Shape check, then a host-only probe of EVERY owner (continuing past
/// failures, first failure remembered), and only if all succeed the restores in
/// ascending order. A probe failure or refusal restores nothing and leaves the
/// mesh healthy. Once a `restore_session_rank` has started, any failure leaves
/// owners with mixed state, so it fails closed.
fn restore_session_all<B: OwnerOps>(
    backend: &mut B,
    epoch: &ForwardEpoch,
    pieces: &[MeshCheckpointId],
) -> Result<()> {
    let owners = backend.rank_count();
    backend.state().admit_lifecycle_epoch(epoch, owners)?;
    check_session_pieces(pieces, epoch, owners)?;

    let mut probe_failure: Option<(usize, String)> = None;
    for (owner, piece) in pieces.iter().enumerate() {
        let result = backend
            .bind(owner)
            .and_then(|()| backend.probe_session(owner, epoch, *piece));
        if let Err(error) = result {
            probe_failure.get_or_insert((owner, error));
        }
    }
    if let Some((owner, error)) = probe_failure {
        return Err(format!(
            "session probe failed on owner {owner}; no owner was restored: {error}"
        ));
    }

    let mut attempted = false;
    for (owner, piece) in pieces.iter().enumerate() {
        let result = backend.bind(owner).and_then(|()| {
            attempted = true;
            backend.restore_session(owner, epoch, *piece)
        });
        if let Err(error) = result {
            if !attempted {
                return Err(format!(
                    "session restore could not start on owner {owner}; no owner was restored: \
                     {error}"
                ));
            }
            fail_closed(backend);
            return Err(format!(
                "session restore failed on owner {owner}; mesh poisoned and quarantined: {error}"
            ));
        }
    }
    Ok(())
}

/// Drain first (a failed drain poisons/quarantines like any failed drain and
/// no `reset_rank` runs), then `reset_rank` on EVERY owner regardless of
/// failures, then synchronize EVERY device regardless of reset failures. Any
/// failure fails closed and names every failing owner. Transport op ids are
/// deliberately untouched, so they stay monotonic across resets.
///
/// The pre-reset drain runs even with no outstanding epoch: family-enqueued
/// work (checkpoint copies, session captures) is not tracked as outstanding.
fn reset_all<B: OwnerOps>(backend: &mut B, epoch: &ForwardEpoch) -> Result<()> {
    let owners = backend.rank_count();
    backend.state().admit_lifecycle_epoch(epoch, owners)?;
    drain(backend).map_err(|e| format!("mesh reset aborted before any reset_rank: {e}"))?;
    backend.state().outstanding.clear();

    let mut failures: Vec<String> = Vec::new();
    for owner in 0..owners {
        let result = backend
            .bind(owner)
            .and_then(|()| backend.reset(owner, epoch));
        if let Err(error) = result {
            failures.push(format!("owner {owner} reset_rank: {error}"));
        }
    }
    let mut sync_failed = false;
    for owner in 0..owners {
        if let Err(error) = backend.synchronize(owner) {
            sync_failed = true;
            failures.push(format!("owner {owner} post-reset synchronize: {error}"));
        }
    }
    if failures.is_empty() {
        return Ok(());
    }
    if sync_failed {
        // A device that cannot synchronize may still run work against its
        // state: quarantine exactly like a failed drain.
        backend.poison();
        backend.quarantine();
    }
    fail_closed(backend);
    Err(format!(
        "mesh reset failed; mesh poisoned and quarantined: {}",
        failures.join("; ")
    ))
}

impl OwnerOps for HipBackend<'_> {
    fn bind(&mut self, device: usize) -> Result<()> {
        self.gpus.devices[device]
            .bind_thread()
            .map_err(|e| e.to_string())
    }
    fn checkpoint(
        &mut self,
        owner: usize,
        device: usize,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        self.bindings
            .checkpoint_rank(owner, &mut self.gpus.devices[device], key)
    }
    fn restore(
        &mut self,
        owner: usize,
        device: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        self.bindings
            .restore_rank(owner, &mut self.gpus.devices[device], key, ticket)
    }
    fn validate_finalize(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()> {
        self.bindings.validate_finalize_rank(owner, key, ticket)
    }
    fn finalize(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId) {
        self.bindings.finalize_rank(owner, key, ticket);
    }
    fn admits_prefix_repair(&self) -> bool {
        self.bindings.admits_prefix_repair()
    }
    fn repair_prefix(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()> {
        self.bindings
            .repair_prefix_rank(owner, &mut self.gpus.devices[owner], epoch, keep, tokens)
    }
    fn capture_session(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<MeshCheckpointId> {
        self.bindings
            .capture_session_rank(owner, &mut self.gpus.devices[owner], epoch)
    }
    fn probe_session(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()> {
        self.bindings.probe_session_rank(owner, epoch, piece)
    }
    fn restore_session(
        &mut self,
        owner: usize,
        epoch: &ForwardEpoch,
        piece: MeshCheckpointId,
    ) -> Result<()> {
        self.bindings
            .restore_session_rank(owner, &mut self.gpus.devices[owner], epoch, piece)
    }
    fn reset(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.bindings
            .reset_rank(owner, &mut self.gpus.devices[owner], epoch)
    }
}

/// Drain backend over the admitted device list.
///
/// A failed drain quarantines the whole device `Vec`: `quarantine` leaks it with
/// `std::mem::forget` because `Gpu::drop` frees registered VMM/host-mapped state
/// that in-flight work may still reference. `Gpus` is therefore unusable
/// afterwards (its `devices` is left empty) until process teardown.
struct HipDrain<'a> {
    devices: &'a mut Vec<Gpu>,
    state: &'a mut MeshExecutionState,
}

impl DrainBackend for HipDrain<'_> {
    fn rank_count(&self) -> usize {
        self.devices.len()
    }
    fn synchronize(&mut self, owner: usize) -> Result<()> {
        let gpu = &mut self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        // Device-wide synchronization includes family copy/PLE side streams.
        gpu.hip.device_synchronize().map_err(|e| e.to_string())
    }
    fn poison(&mut self) {
        self.state.poison();
    }
    fn quarantine(&mut self) {
        self.state.quarantined = true;
        // Gpu::drop frees registered VMM/host-mapped state; a failed drain keeps
        // device ownership until process teardown.
        std::mem::forget(std::mem::take(self.devices));
    }
}

impl HasState for HipDrain<'_> {
    fn state(&mut self) -> &mut MeshExecutionState {
        self.state
    }
}

fn pair(devices: &mut [Gpu], src: usize, dst: usize) -> (&mut Gpu, &mut Gpu) {
    if src < dst {
        let (a, b) = devices.split_at_mut(dst);
        (&mut a[src], &mut b[0])
    } else {
        let (a, b) = devices.split_at_mut(src);
        (&mut b[0], &mut a[dst])
    }
}

static NEXT_ISSUER: AtomicU64 = AtomicU64::new(1);

/// The executor runs pure PP (TP = EP = 1) or pure TP (PP = EP = 1) meshes;
/// composed TP×PP and any EP axis are refused before anything is touched.
fn admit_mesh_shape(mesh: &DeviceMesh) -> Result<()> {
    let (tp, pp) = (mesh.size_of(DimKind::Tp), mesh.size_of(DimKind::Pp));
    if mesh.size_of(DimKind::Ep) != 1 || (tp != 1 && pp != 1) {
        return Err(
            "mesh executor admits pure PP or pure TP meshes only (TP×PP and EP are refused)".into(),
        );
    }
    Ok(())
}

impl Gpus {
    /// Provision transactionally from admitted logical-to-physical identities:
    /// PP edge leases, or the pure-TP reduction group.
    /// Existing legacy layout/output accessors are deliberately untouched.
    pub fn provision_mesh_edges(
        &mut self,
        admitted: &AdmittedMesh,
        placement: &BandPlacement,
        max_bytes: usize,
    ) -> Result<()> {
        placement.validate_for(&admitted.mesh, placement.n_trunk_layers)?;
        if self.mesh_execution.mesh_epoch.is_some() || self.mesh_execution.quarantined {
            return Err("mesh edges already provisioned or quarantined".into());
        }
        admit_mesh_shape(&admitted.mesh)?;
        if admitted.mesh.n_devices() != self.devices.len()
            || admitted.ranks.len() != self.devices.len()
            || max_bytes == 0
            || max_bytes % 16 != 0
            || max_bytes / 16 > i32::MAX as usize
        {
            return Err("invalid admitted rank count or edge capacity".into());
        }
        for (owner, rank) in admitted.ranks.iter().enumerate() {
            let gpu = &self.devices[owner];
            if rank.logical != owner || rank.device_ordinal != gpu.device_id {
                return Err(format!(
                    "admitted rank {owner} (logical {}, device {}) is not devices[{owner}] (device {})",
                    rank.logical, rank.device_ordinal, gpu.device_id
                ));
            }
            if rank.arch != gpu.arch {
                return Err(format!(
                    "admitted rank {owner} arch {} does not match device arch {}",
                    rank.arch, gpu.arch
                ));
            }
            if gpu.active_stream.is_none() {
                return Err(format!(
                    "device of rank {owner} has no launch stream (mesh work never runs on the null stream)"
                ));
            }
            if let Some(expected_pci) = &rank.pci_bus_id {
                gpu.bind_thread().map_err(|e| e.to_string())?;
                if gpu
                    .hip
                    .device_pci_bus_id(gpu.device_id)
                    .map_err(|e| e.to_string())?
                    != *expected_pci
                {
                    return Err("admitted PCI identity does not match physical device".into());
                }
            }
        }
        let expected: HashSet<_> = placement
            .boundaries()
            .map(|(_, src, dst)| (src, dst))
            .collect();
        let supplied: HashSet<_> = admitted.edges.iter().map(|e| (e.src, e.dst)).collect();
        if expected != supplied || supplied.len() != admitted.edges.len() {
            return Err("admitted edge decisions do not match PP boundaries".into());
        }
        let issuer = NEXT_ISSUER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "mesh ticket identity exhausted")?;
        let (tp, tp_scratch) = if admitted.mesh.size_of(DimKind::Tp) > 1 {
            let (group, scratch) = self.provision_tp_group(max_bytes)?;
            (Some(group), scratch)
        } else {
            (None, Vec::new())
        };
        let mut edges = HashMap::with_capacity(admitted.edges.len());
        let result = (|| {
            for decision in &admitted.edges {
                let (src, dst) = pair(&mut self.devices, decision.src, decision.dst);
                if decision.transport == EdgeTransport::Peer && !transport::qualify_peer(src, dst)?
                {
                    return Err("admitted Peer edge failed byte qualification; no fallback".into());
                }
                let id = EdgeId {
                    src_device: decision.src,
                    dst_device: decision.dst,
                };
                let lease = transport::provision(
                    src,
                    dst,
                    EdgeSpec {
                        edge: id,
                        transport: decision.transport,
                        max_bytes,
                    },
                )?;
                edges.insert(id, OwnedEdge { lease, max_bytes });
            }
            Ok(())
        })();
        if let Err(error) = result {
            // A failed provision may already have quarantined its private lease.
            // Retain all successful leases until the explicit all-device drain.
            self.mesh_execution.edges = edges;
            self.mesh_execution.tp_scratch = tp_scratch;
            self.mesh_execution.mesh_epoch = Some(admitted.mesh.epoch());
            self.mesh_execution.poison();
            return match self.drain_mesh_edges() {
                Ok(()) => Err(error),
                Err(drain) => Err(format!("{error}; {drain}")),
            };
        }
        self.mesh_execution = MeshExecutionState {
            edges,
            mesh_epoch: Some(admitted.mesh.epoch()),
            output_owner: Some(placement.stages() - 1),
            tp,
            tp_scratch,
            issuer,
            ..Default::default()
        };
        Ok(())
    }

    /// Decide and warm the pure-TP reduction before any submission. Peer when
    /// every rank pair is one device under `HIPFIRE_EMULATE_GPUS` or passes
    /// `qualify_peer` (the kernel copy every peer read then uses); otherwise
    /// the hand route's host-staged reduction. Scratch, events, host staging
    /// and kernels exist before the first epoch, so submission allocates none.
    fn provision_tp_group(&mut self, max_bytes: usize) -> Result<(TpGroup, Vec<TpScratch>)> {
        let n = self.devices.len();
        let emulated = crate::config::get().emulate_gpus.is_some();
        let mut chosen = TpTransport::Peer;
        'pairs: for a in 0..n {
            for b in a + 1..n {
                let (x, y) = pair(&mut self.devices, a, b);
                let readable = if x.device_id == y.device_id {
                    if !emulated {
                        return Err(format!(
                            "TP ranks {a} and {b} share device {} outside HIPFIRE_EMULATE_GPUS",
                            x.device_id
                        ));
                    }
                    true
                } else {
                    transport::qualify_peer(x, y)?
                };
                if !readable {
                    chosen = TpTransport::Host;
                    break 'pairs;
                }
            }
        }
        let epoch = transport::next_epoch()?;
        let mut scratch = Vec::new();
        match chosen {
            TpTransport::Peer => {
                self.ensure_rank_barrier_events().map_err(hip_error)?;
                self.ensure_rank_done_events().map_err(hip_error)?;
                if let Err(error) = alloc_tp_scratch(&mut self.devices, max_bytes, &mut scratch) {
                    return match free_tp_scratch(&mut self.devices, scratch) {
                        Ok(()) => Err(error),
                        Err(free) => Err(format!("{error}; TP scratch leaked: {free}")),
                    };
                }
            }
            TpTransport::Host => self.host_ar_tmp = vec![vec![0.0; max_bytes / 4]; n],
        }
        let group = TpGroup {
            transport: chosen,
            epoch,
            max_bytes,
        };
        Ok((group, scratch))
    }

    /// Logical output owner whose admitted physical identity is retained in the
    /// device vector. This is separate from the legacy Qwen35 output accessor.
    pub fn mesh_output_owner(&self) -> Option<usize> {
        self.mesh_execution.output_owner
    }

    pub fn mesh_edge_epoch(&self, edge: EdgeId) -> Option<u64> {
        self.mesh_execution
            .edges
            .get(&edge)
            .map(|e| e.lease.epoch())
    }

    /// Transport epoch of the provisioned pure-TP group: the `descriptor.epoch`
    /// of every `AllReduce`/`AllReduceAdd` boundary. `None` for a PP mesh or a
    /// quarantined one.
    pub fn mesh_tp_epoch(&self) -> Option<u64> {
        let state = &self.mesh_execution;
        state.tp.filter(|_| !state.quarantined).map(|tp| tp.epoch)
    }

    /// How the provisioned pure-TP group moves partials: `"peer"` (every
    /// rank pair byte-qualified, or one device under emulation) or `"host"`
    /// (the host-staged rank-order reduction). `None` for a PP mesh or a
    /// quarantined one.
    pub fn mesh_tp_transport(&self) -> Option<&'static str> {
        let state = &self.mesh_execution;
        state
            .tp
            .filter(|_| !state.quarantined)
            .map(|tp| match tp.transport {
                TpTransport::Peer => "peer",
                TpTransport::Host => "host",
            })
    }

    /// The admission [`Self::submit_mesh_epoch`] applies first, without
    /// submitting: refuses a stale, poisoned, quarantined or unprovisioned mesh,
    /// another request's undrained epochs, a non-increasing op id and a changed
    /// load/numerical policy. Mutates nothing. For a family that replays a
    /// captured whole-token program instead of submitting the epoch.
    pub fn admit_mesh_epoch(&self, epoch: &ForwardEpoch) -> Result<()> {
        self.mesh_execution.admit(epoch)
    }

    /// Submit one epoch. Epochs of the same request may be outstanding at once
    /// (strictly increasing op ids); every ticket must be drained together by
    /// `drain_mesh_epochs`. A submission failure after any device enqueue
    /// poisons the mesh: all earlier outstanding tickets then drain to an error.
    pub fn submit_mesh_epoch(
        &mut self,
        plan: &SplitPlan,
        epoch: &ForwardEpoch,
        bindings: &mut dyn MeshStepBindings,
    ) -> Result<MeshTicket> {
        submit(
            plan,
            epoch,
            &mut HipBackend {
                gpus: self,
                bindings,
            },
        )
    }

    /// Single-ticket drain: valid only while that ticket names the sole
    /// outstanding epoch.
    pub fn drain_mesh_epoch(&mut self, ticket: MeshTicket) -> Result<()> {
        drain_tickets(
            &[ticket],
            &mut HipDrain {
                devices: &mut self.devices,
                state: &mut self.mesh_execution,
            },
        )
    }

    /// Drain every outstanding epoch with one synchronization of each device.
    /// `tickets` must be exactly the outstanding tickets, oldest first; any
    /// stale, foreign, missing or out-of-order ticket is refused before a
    /// device is touched and leaves the state unchanged. A failed device sync
    /// quarantines exactly as `drain_mesh_edges` does.
    pub fn drain_mesh_epochs(&mut self, tickets: Vec<MeshTicket>) -> Result<()> {
        drain_tickets(
            &tickets,
            &mut HipDrain {
                devices: &mut self.devices,
                state: &mut self.mesh_execution,
            },
        )
    }

    /// Capture every trunk owner (ascending) and, with `with_head`, the head
    /// on `mesh_output_owner()` (callback owner `devices.len()`), each on its
    /// own bound device thread. `key.mesh_epoch` must be the provisioned mesh
    /// epoch of a healthy mesh (not poisoned, not quarantined); the key's
    /// load/request must match the pinned load and the undrained epochs'
    /// request when those are known. Otherwise nothing is called and a
    /// `Shape` error is returned. Rollback follows [`checkpoint_all`]; if any
    /// owner's rollback also fails the mesh is poisoned and quarantined.
    pub fn checkpoint_mesh(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        key: &MeshCheckpointKey,
        with_head: bool,
    ) -> Result<MeshCheckpoint, MeshLifecycleError> {
        checkpoint_window(&mut self.lifecycle_backend(bindings), key, with_head)
    }

    /// Roll every owner back (head first, then trunk descending, continuing
    /// past failures: [`restore_all`]). The same mesh health/identity
    /// admission as [`Self::checkpoint_mesh`], plus the checkpoint's key and
    /// shape (one ticket per device, owner indices exact) are validated before
    /// any callback. Any failed owner restore poisons and quarantines the mesh.
    pub fn restore_mesh(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        key: &MeshCheckpointKey,
        checkpoint: MeshCheckpoint,
    ) -> Result<(), MeshLifecycleError> {
        restore_window(&mut self.lifecycle_backend(bindings), key, checkpoint)
    }

    /// Commit the checkpoint ([`finalize_all`]): every owner validates before
    /// any owner finalizes; a validation failure restores every owner. Same
    /// admission and shape checks as [`Self::restore_mesh`]; a rollback that
    /// itself fails poisons and quarantines the mesh, a clean one does not.
    /// The infallible host-only `finalize_rank` runs without binding; the
    /// fallible binding happened in the validation phase.
    pub fn finalize_mesh(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        key: &MeshCheckpointKey,
        checkpoint: MeshCheckpoint,
    ) -> Result<(), MeshLifecycleError> {
        finalize_window(&mut self.lifecycle_backend(bindings), key, checkpoint)
    }

    /// Accepted-prefix repair on every trunk owner, ascending, stopping at the
    /// first error. Refuses a stale/poisoned/quarantined mesh or an epoch whose
    /// load/policy/request disagrees with the mesh before any callback (its
    /// op id is NOT checked: the epoch has usually already executed). A
    /// failing owner poisons and quarantines the mesh and is named in the error.
    pub fn repair_mesh_prefix(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()> {
        repair_prefix_all(&mut self.lifecycle_backend(bindings), epoch, keep, tokens)
    }

    /// Capture one session piece per trunk owner, ascending, stopping at the
    /// first failure (named by owner; the family releases already captured
    /// pieces through its own state). Piece `i` carries owner `i`, and all
    /// pieces share one key whose mesh/load/request correspond to `epoch`.
    /// Same admission as [`Self::repair_mesh_prefix`].
    pub fn capture_mesh_session(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        epoch: &ForwardEpoch,
    ) -> Result<Vec<MeshCheckpointId>> {
        capture_session_all(&mut self.lifecycle_backend(bindings), epoch)
    }

    /// Restore a session. `pieces` must be exactly one per device with exact
    /// owner indices and one shared key corresponding to `epoch`; the host-only
    /// `probe_session_rank` then runs on EVERY owner (even after a failed
    /// probe) before any `restore_session_rank`. A refusal or failed probe
    /// restores nothing and leaves the mesh healthy; a failure once restores
    /// have started poisons and quarantines it.
    pub fn restore_mesh_session(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        epoch: &ForwardEpoch,
        pieces: &[MeshCheckpointId],
    ) -> Result<()> {
        restore_session_all(&mut self.lifecycle_backend(bindings), epoch, pieces)
    }

    /// Reset request state on every owner. First drains every device (a failed
    /// drain quarantines as in `drain_mesh_epochs` and NO `reset_rank` runs)
    /// and retires the outstanding epochs; then runs `reset_rank` on every
    /// owner regardless of failures; then synchronizes every device regardless
    /// of reset failures. Any failure poisons and quarantines the mesh and
    /// names the failing owners. The transport op-id frontier is untouched, so
    /// op ids stay monotonic across resets. Same admission as
    /// [`Self::repair_mesh_prefix`].
    pub fn reset_mesh(
        &mut self,
        bindings: &mut dyn MeshStepBindings,
        epoch: &ForwardEpoch,
    ) -> Result<()> {
        reset_all(&mut self.lifecycle_backend(bindings), epoch)
    }

    fn lifecycle_backend<'a>(
        &'a mut self,
        bindings: &'a mut dyn MeshStepBindings,
    ) -> HipBackend<'a> {
        HipBackend {
            gpus: self,
            bindings,
        }
    }

    /// Stop and drain every device before releasing any bank/event. A failed
    /// drain never frees any mesh lease, even if other devices are healthy.
    pub fn drain_mesh_edges(&mut self) -> Result<()> {
        if self.mesh_execution.quarantined {
            return Err("mesh leases quarantined until process teardown".into());
        }
        self.mesh_execution.poison();
        drain(&mut HipDrain {
            devices: &mut self.devices,
            state: &mut self.mesh_execution,
        })?;
        for (id, edge) in &mut self.mesh_execution.edges {
            let (src, dst) = pair(&mut self.devices, id.src_device, id.dst_device);
            if let Err(error) = edge.lease.drain(src, dst) {
                self.mesh_execution.quarantined = true;
                return Err(error);
            }
        }
        let scratch = std::mem::take(&mut self.mesh_execution.tp_scratch);
        if let Err(error) = free_tp_scratch(&mut self.devices, scratch) {
            self.mesh_execution.quarantined = true;
            return Err(format!("TP scratch release failed: {error}"));
        }
        self.mesh_execution = MeshExecutionState::default();
        Ok(())
    }
}

/// One `TpScratch` per rank, pushed as allocated so a failure can free the
/// prefix. Warms every kernel the peer reduction launches.
fn alloc_tp_scratch(
    devices: &mut [Gpu],
    max_bytes: usize,
    scratch: &mut Vec<TpScratch>,
) -> Result<()> {
    let ranks = devices.len();
    let bytes = ranks
        .checked_mul(max_bytes)
        .ok_or("TP scratch size overflow")?;
    for gpu in devices.iter_mut() {
        gpu.bind_thread().map_err(hip_error)?;
        let stage = gpu.hip.malloc(bytes).map_err(hip_error)?;
        let owners = match gpu.upload_raw(&[0; 16], &[16]) {
            Ok(owners) => owners,
            Err(error) => {
                let _ = gpu.hip.free(stage);
                return Err(hip_error(error));
            }
        };
        scratch.push(TpScratch {
            stage: GpuTensor {
                buf: stage,
                shape: vec![bytes / 4],
                dtype: DType::F32,
            },
            owners,
        });
        let rank = scratch.last().expect("pushed above");
        let base = rank.stage.buf.as_ptr() as usize;
        let (a, b) = (rank.stage.sub_offset(0, 4), rank.stage.sub_offset(4, 4));
        transport::copy_bytes(gpu, &rank.owners, base, base + 16, 16)?;
        gpu.add_inplace_f32(&a, &b).map_err(hip_error)?;
        gpu.add_f32(&a, &b, &a).map_err(hip_error)?;
        if ranks == 2 {
            // The TP2 direct-peer add (`peer_reduce`) launches this kernel.
            gpu.add_peer_residual_f32(&a, &b, b.buf.as_ptr() as *const std::ffi::c_void, 4)
                .map_err(hip_error)?;
        }
        if let Some(fmt) = tp_exchange_format() {
            // The 16-bit exchange packs and folds with these kernels.
            gpu.ensure_tp_exchange16_kernels(fmt).map_err(hip_error)?;
        }
        gpu.hip.device_synchronize().map_err(hip_error)?;
    }
    Ok(())
}

/// Synchronize each owning device, then free its `TpScratch`.
fn free_tp_scratch(devices: &mut [Gpu], scratch: Vec<TpScratch>) -> Result<()> {
    for (gpu, rank) in devices.iter_mut().zip(scratch) {
        gpu.bind_thread().map_err(hip_error)?;
        gpu.hip.device_synchronize().map_err(hip_error)?;
        gpu.hip.free(rank.stage.buf).map_err(hip_error)?;
        gpu.hip.free(rank.owners.buf).map_err(hip_error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_mesh::DeviceMesh;
    use crate::mesh_split::{CollectiveId, RankProgram, ResourceShard, ScheduleRef, ShardSpec};
    use hipfire_dispatch::pipeline::step_graph::{
        OpDecl, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StepGraph,
    };

    #[derive(Debug, PartialEq, Eq)]
    enum Call {
        Ops(usize, Vec<OpId>),
        Publish(u32),
        Receive(u32),
        Reduce(u32),
        Marks(usize),
        Sync(usize),
    }

    struct Toy {
        ranks: usize,
        calls: Vec<Call>,
        missing: bool,
        oversize: bool,
        wrong_descriptor: bool,
        no_stream: bool,
        fail_ops: Option<usize>,
        fail_drain: Option<usize>,
        fail_prepare: bool,
        prepared: Vec<u64>,
        poisoned: bool,
        quarantined: bool,
        /// A TP group is provisioned (group collectives have boundaries).
        tp: bool,
        /// Every rank binds the same group memory (an aliased binding).
        alias: bool,
        /// Per-rank `BoundaryIo::partial_slot` of every group collective
        /// (empty: `None` on every rank).
        partial_slots: Vec<Option<usize>>,
        /// The real executor state machine, driven by the production
        /// `submit`/`drain_tickets` through `HasState`.
        state: MeshExecutionState,
    }

    impl Toy {
        fn new(ranks: usize) -> Self {
            Self {
                ranks,
                calls: Vec::new(),
                missing: false,
                oversize: false,
                wrong_descriptor: false,
                no_stream: false,
                fail_ops: None,
                fail_drain: None,
                fail_prepare: false,
                prepared: Vec::new(),
                poisoned: false,
                quarantined: false,
                tp: false,
                alias: false,
                partial_slots: Vec::new(),
                state: MeshExecutionState::default(),
            }
        }

        /// Provisioned for `epoch`'s mesh with a distinct ticket issuer.
        fn provisioned(ranks: usize, epoch: &ForwardEpoch, issuer: u64) -> Self {
            let mut toy = Self::new(ranks);
            toy.state.mesh_epoch = Some(epoch.mesh_epoch);
            toy.state.issuer = issuer;
            toy
        }

        fn syncs(&self) -> Vec<usize> {
            self.calls
                .iter()
                .filter_map(|call| match call {
                    Call::Sync(owner) => Some(*owner),
                    _ => None,
                })
                .collect()
        }
    }

    impl DrainBackend for Toy {
        fn rank_count(&self) -> usize {
            self.ranks
        }
        fn synchronize(&mut self, owner: usize) -> Result<()> {
            self.calls.push(Call::Sync(owner));
            if self.fail_drain == Some(owner) {
                Err("injected drain failure".into())
            } else {
                Ok(())
            }
        }
        fn poison(&mut self) {
            self.poisoned = true;
            self.state.poison();
        }
        fn quarantine(&mut self) {
            self.quarantined = true;
            self.state.quarantined = true;
        }
    }

    impl Backend for Toy {
        fn ready(&self, _: usize) -> bool {
            !self.no_stream
        }
        fn boundary(
            &self,
            owner: usize,
            c: &PlannedCollective,
            epoch: &ForwardEpoch,
        ) -> Result<BoundaryMeta> {
            if self.missing {
                return Err("no lease".into());
            }
            let band = matches!(c.kind, CollectiveKind::BandXfer { .. });
            if !band && !self.tp {
                return Err("TP collective on a mesh without a provisioned TP group".into());
            }
            // Group collectives get distinct per-rank memory unless `alias`.
            let slot = if band || self.alias { 0 } else { owner };
            Ok(BoundaryMeta {
                descriptor: EdgeDescriptor {
                    epoch: 17,
                    op: epoch.op_id + u64::from(self.wrong_descriptor),
                    rows: epoch.actual_rows,
                    dtype_bytes: 4,
                    bytes: if self.oversize { 256 } else { 64 },
                },
                address: 0x1000 * (slot + 1),
                extent: 512,
                capacity: 64,
                transport_epoch: 17,
                into: matches!(c.kind, CollectiveKind::AllReduceAdd { .. })
                    .then_some((0x10_0000 + 0x1000 * slot, 512)),
                partial_slot: self.partial_slots.get(owner).copied().flatten(),
            })
        }
        fn run_ops(&mut self, owner: usize, _: &ForwardEpoch, ops: &[OpId]) -> Result<()> {
            self.calls.push(Call::Ops(owner, ops.to_vec()));
            if self.fail_ops == Some(owner) {
                Err("injected stage failure".into())
            } else {
                Ok(())
            }
        }
        fn transfer(
            &mut self,
            _: usize,
            c: &PlannedCollective,
            _: &ForwardEpoch,
            _: BoundaryMeta,
            publish: bool,
        ) -> Result<()> {
            self.calls.push(if publish {
                Call::Publish(c.id.0)
            } else {
                Call::Receive(c.id.0)
            });
            Ok(())
        }
        fn reduce(
            &mut self,
            c: &PlannedCollective,
            _: &ForwardEpoch,
            metas: &[BoundaryMeta],
        ) -> Result<()> {
            assert_eq!(metas.len(), c.group.len());
            self.calls.push(Call::Reduce(c.id.0));
            Ok(())
        }
        fn marks(&mut self, owner: usize, _: &ForwardEpoch) -> Result<()> {
            self.calls.push(Call::Marks(owner));
            Ok(())
        }
        fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
            self.prepared.push(epoch.op_id);
            if self.fail_prepare {
                Err("injected prepare failure".into())
            } else {
                Ok(())
            }
        }
    }

    impl HasState for Toy {
        fn state(&mut self) -> &mut MeshExecutionState {
            &mut self.state
        }
    }

    fn epoch(mesh: &DeviceMesh) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh.epoch(),
            load_id: 1,
            request_id: 2,
            op_id: 3,
            route: MeshRoute::Prefill,
            start_position: 0,
            actual_rows: 1,
            route_rows: 1,
            policy_id: 4,
        }
    }

    // A non-model toy family lowers two contiguous fake typed ops per band and
    // chunk into a StepGraph and its rank schedule. No family runtime is used.
    fn toy_plan(bands: usize, chunks: usize) -> (StepGraph, SplitPlan, ForwardEpoch) {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, bands)]).unwrap();
        let mut graph = StepGraph::new();
        let resource = graph
            .add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![16],
                alias: None,
            })
            .unwrap();
        let mut plan = SplitPlan {
            mesh_epoch: mesh.epoch(),
            ranks: (0..bands)
                .map(|device| RankProgram {
                    device,
                    items: Vec::new(),
                })
                .collect(),
            collectives: Vec::new(),
            resources: vec![ResourceShard {
                entry: ShardSpec::Whole,
                exit: ShardSpec::Whole,
            }],
        };
        for _chunk in 0..chunks {
            for owner in 0..bands {
                let mut producer = OpId(0);
                for _ in 0..2 {
                    producer = graph
                        .add_op(OpDecl {
                            kind: OpKind::Scale,
                            layer: Some(owner),
                            rule: SplitRule::RowComplete,
                            reads: vec![resource],
                            writes: vec![resource],
                        })
                        .unwrap();
                    plan.ranks[owner].items.push(RankItem::Op(producer));
                }
                if owner + 1 < bands {
                    let id = CollectiveId(plan.collectives.len() as u32);
                    plan.collectives.push(PlannedCollective {
                        id,
                        kind: CollectiveKind::BandXfer {
                            src: owner,
                            dst: owner + 1,
                        },
                        operand: resource,
                        producer,
                        group: vec![owner, owner + 1],
                        fulfils: ScheduleRef::Band(owner),
                    });
                    plan.ranks[owner].items.push(RankItem::Collective(id));
                    plan.ranks[owner + 1].items.push(RankItem::Collective(id));
                }
            }
        }
        (graph, plan, epoch(&mesh))
    }

    fn prove_graph(bands: usize) {
        let chunks = 4;
        let (graph, plan, epoch) = toy_plan(bands, chunks);
        let mut toy = Toy::new(bands);
        let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
        assert!(toy.calls.is_empty());
        issue(&mut prepared, &plan, &epoch, &mut toy).unwrap();
        let mut positions = HashMap::new();
        for (position, call) in toy.calls.iter().enumerate() {
            if let Call::Ops(owner, ops) = call {
                assert_eq!(ops.len(), 2, "contiguous op fusion preserved");
                for op in ops {
                    assert!(positions.insert(*op, position).is_none());
                    assert_eq!(graph.ops()[op.0 as usize].layer, Some(*owner));
                }
            }
        }
        assert_eq!(positions.len(), graph.ops().len());
        for c in &plan.collectives {
            let published: Vec<_> = toy
                .calls
                .iter()
                .enumerate()
                .filter(|(_, call)| **call == Call::Publish(c.id.0))
                .map(|(i, _)| i)
                .collect();
            let received: Vec<_> = toy
                .calls
                .iter()
                .enumerate()
                .filter(|(_, call)| **call == Call::Receive(c.id.0))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(published.len(), 1);
            assert_eq!(received.len(), 1);
            assert!(positions[&c.producer] < published[0]);
            assert!(published[0] < received[0]);
            let CollectiveKind::BandXfer { dst, .. } = c.kind else {
                unreachable!()
            };
            let rank = &plan.ranks[dst].items;
            let index = rank
                .iter()
                .position(|item| *item == RankItem::Collective(c.id))
                .unwrap();
            let RankItem::Op(consumer) = rank[index + 1] else {
                panic!("missing consumer")
            };
            assert!(received[0] < positions[&consumer]);
        }
        // Each stage/chunk's ops precede the next stage of that same chunk.
        for chunk in 0..chunks {
            for stage in 0..bands - 1 {
                assert!(
                    positions[&OpId(((chunk * bands + stage) * 2) as u32)]
                        < positions[&OpId(((chunk * bands + stage + 1) * 2) as u32)]
                );
            }
        }
        // Enqueueing is not globally serialized chunk-by-chunk: the first
        // stage can start chunk 1 before the next stage starts chunk 0.
        assert!(positions[&OpId((bands * 2) as u32)] < positions[&OpId(2)]);
        assert!(toy.calls.iter().all(|call| !matches!(call, Call::Sync(_))));
        assert!(issue(&mut prepared, &plan, &epoch, &mut toy).is_err());
        assert!(toy.poisoned);
    }

    #[test]
    fn two_band_toy_exactly_once_and_topological() {
        prove_graph(2);
    }

    #[test]
    fn six_band_toy_exactly_once_and_topological() {
        prove_graph(6);
    }

    #[test]
    fn real_splitter_non_family_graph_executes() {
        for bands in [2, 6] {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, bands)]).unwrap();
            let placement = BandPlacement::uniform(&mesh, bands).unwrap();
            let manifest =
                crate::weight_manifest::plan_manifest(&[], &[], &mesh, bands, &placement).unwrap();
            let mut graph = StepGraph::new();
            let x = graph
                .add_resource(ResourceDecl {
                    class: ResourceClass::Activation,
                    shape: vec![16],
                    alias: None,
                })
                .unwrap();
            for layer in 0..bands {
                graph
                    .add_op(OpDecl {
                        kind: OpKind::Scale,
                        layer: Some(layer),
                        rule: SplitRule::RowComplete,
                        reads: vec![x],
                        writes: vec![x],
                    })
                    .unwrap();
            }
            let plan = crate::mesh_split::split_program(&graph, &manifest, &mesh).unwrap();
            let mut toy = Toy::new(bands);
            let epoch = epoch(&mesh);
            let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
            issue(&mut prepared, &plan, &epoch, &mut toy).unwrap();
            assert_eq!(
                toy.calls
                    .iter()
                    .filter(|call| matches!(call, Call::Ops(..)))
                    .count(),
                bands
            );
            assert_eq!(
                toy.calls
                    .iter()
                    .filter(|call| matches!(call, Call::Publish(_)))
                    .count(),
                bands - 1
            );
        }
    }

    #[test]
    fn stale_epoch_refused_before_calls() {
        let (_, plan, mut epoch) = toy_plan(2, 1);
        epoch.mesh_epoch = DeviceMesh::single().unwrap().epoch();
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn wrong_device_refused_before_calls() {
        let (_, mut plan, epoch) = toy_plan(2, 1);
        plan.ranks[1].device = 0;
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn allreduce_on_a_pp_mesh_refused_before_calls() {
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllReduceAdd {
                into: ResourceId(0),
                replaces: OpId(0),
            },
        ] {
            let (_, mut plan, epoch) = toy_plan(2, 1);
            plan.collectives[0].kind = kind;
            let toy = Toy::new(2);
            assert!(prepare(&plan, &epoch, &toy)
                .unwrap_err()
                .contains("TP group"));
            assert!(toy.calls.is_empty());
        }
    }

    /// F32 elements per row of every TP fixture activation.
    const TP_H: usize = 1024;
    const TP_K: usize = 64;

    /// A synthetic TP step graph split by the real splitter: `a = col(x)`
    /// (Split), `p = row(a)` folded into `x` (`AllReduceAdd`), `q = row(a)`
    /// read whole by `n` (`AllReduce`). `ops` = `[col, row_p, x += p, row_q, n]`.
    struct TpFixture {
        plan: SplitPlan,
        p: ResourceId,
        q: ResourceId,
        ops: [OpId; 5],
    }

    fn tp_fixture(mesh: &DeviceMesh) -> TpFixture {
        use crate::weight_manifest::{plan_manifest, ShardPolicy, WeightEntry};
        let mut graph = StepGraph::new();
        let mut entries = Vec::new();
        let act = |graph: &mut StepGraph, len: usize| {
            graph
                .add_resource(ResourceDecl {
                    class: ResourceClass::Activation,
                    shape: vec![len],
                    alias: None,
                })
                .unwrap()
        };
        let mut weight = |graph: &mut StepGraph, name: &str, shape: Vec<usize>, policy| {
            entries.push(WeightEntry::layer(
                name,
                0,
                shape.clone(),
                DType::F16,
                policy,
            ));
            graph
                .add_resource(ResourceDecl {
                    class: ResourceClass::Weight {
                        name: name.into(),
                        layer: Some(0),
                        group: 1,
                    },
                    shape,
                    alias: None,
                })
                .unwrap()
        };
        let col = weight(
            &mut graph,
            "col",
            vec![TP_K, TP_H],
            ShardPolicy::ColumnShard { axis: 0 },
        );
        let row_p = weight(
            &mut graph,
            "row_p",
            vec![TP_H, TP_K],
            ShardPolicy::RowShard { axis: 1 },
        );
        let row_q = weight(
            &mut graph,
            "row_q",
            vec![TP_H, TP_K],
            ShardPolicy::RowShard { axis: 1 },
        );
        let [x, a, p, q, n] = [TP_H, TP_K, TP_H, TP_H, TP_H].map(|len| act(&mut graph, len));
        let mut op = |kind, rule, reads: &[ResourceId], writes: &[ResourceId]| {
            graph
                .add_op(OpDecl {
                    kind,
                    layer: Some(0),
                    rule,
                    reads: reads.to_vec(),
                    writes: writes.to_vec(),
                })
                .unwrap()
        };
        let project = |weight, input, output| SplitRule::Project {
            weight,
            input,
            output,
        };
        let ops = [
            op(OpKind::Project, project(col, x, a), &[col, x], &[a]),
            op(OpKind::Project, project(row_p, a, p), &[row_p, a], &[p]),
            op(
                OpKind::ResidualAdd,
                SplitRule::Accumulate { dst: x, src: p },
                &[x, p],
                &[x],
            ),
            op(OpKind::Project, project(row_q, a, q), &[row_q, a], &[q]),
            op(OpKind::RmsnormAutomatic, SplitRule::RowComplete, &[q], &[n]),
        ];
        let placement = BandPlacement::uniform(mesh, 1).unwrap();
        let manifest = plan_manifest(&entries, &[], mesh, 1, &placement).unwrap();
        let plan = crate::mesh_split::split_program(&graph, &manifest, mesh).unwrap();
        assert_eq!(
            plan.collectives
                .iter()
                .map(|c| (c.kind, c.operand))
                .collect::<Vec<_>>(),
            [
                (
                    CollectiveKind::AllReduceAdd {
                        into: x,
                        replaces: ops[2]
                    },
                    p
                ),
                (CollectiveKind::AllReduce, q),
            ]
        );
        TpFixture { plan, p, q, ops }
    }

    fn tp_toy(ranks: usize) -> (TpFixture, ForwardEpoch, Toy) {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, ranks)]).unwrap();
        let fixture = tp_fixture(&mesh);
        let epoch = epoch(&mesh);
        let mut toy = Toy::provisioned(ranks, &epoch, 9);
        toy.tp = true;
        (fixture, epoch, toy)
    }

    #[test]
    fn tp_reductions_issue_once_after_every_rank_producer() {
        for ranks in [2, 4] {
            let (f, epoch, mut toy) = tp_toy(ranks);
            let mut prepared = prepare(&f.plan, &epoch, &toy).unwrap();
            issue(&mut prepared, &f.plan, &epoch, &mut toy).unwrap();
            let ran = |owner: usize, op: OpId| {
                toy.calls.iter().position(
                    |call| matches!(call, Call::Ops(o, ops) if *o == owner && ops.contains(&op)),
                )
            };
            for (id, producer, consumer) in [(0, f.ops[1], f.ops[3]), (1, f.ops[3], f.ops[4])] {
                let reduces: Vec<_> = toy
                    .calls
                    .iter()
                    .enumerate()
                    .filter(|(_, call)| **call == Call::Reduce(id))
                    .map(|(at, _)| at)
                    .collect();
                assert_eq!(reduces.len(), 1, "TP{ranks} collective {id}");
                for owner in 0..ranks {
                    assert!(ran(owner, producer).unwrap() < reduces[0]);
                    assert!(reduces[0] < ran(owner, consumer).unwrap());
                }
            }
            assert!((0..ranks).all(|owner| ran(owner, f.ops[2]).is_none()));
            let marks = toy
                .calls
                .iter()
                .filter(|call| matches!(call, Call::Marks(_)))
                .count();
            assert_eq!(marks, ranks);
        }
    }

    #[test]
    fn malformed_tp_collectives_refused_without_side_effects() {
        let (f, epoch, mut toy) = tp_toy(4);
        let mut bad = Vec::new();
        for group in [vec![1, 0, 2, 3], vec![0, 1, 2], vec![0, 0, 1, 2, 3]] {
            let mut plan = f.plan.clone();
            plan.collectives[0].group = group;
            bad.push(plan);
        }
        // One rank skips the reduction.
        let mut plan = f.plan.clone();
        plan.ranks[3]
            .items
            .retain(|item| *item != RankItem::Collective(CollectiveId(1)));
        bad.push(plan);
        // The replaced Accumulate also runs as an op.
        let mut plan = f.plan.clone();
        plan.ranks[2].items.insert(0, RankItem::Op(f.ops[2]));
        bad.push(plan);
        for plan in &bad {
            assert!(submit(plan, &epoch, &mut toy).is_err());
        }
        // An aliased binding (every rank names the same memory).
        toy.alias = true;
        assert!(submit(&f.plan, &epoch, &mut toy)
            .unwrap_err()
            .contains("overlap"));
        toy.alias = false;
        assert!(toy.calls.is_empty() && !toy.poisoned);
        assert!(toy.state.outstanding.is_empty() && toy.state.last_op.is_none());
        // The same epoch still submits once the plan is well formed.
        submit(&f.plan, &epoch, &mut toy).unwrap();
    }

    #[test]
    fn tp_partial_slots_must_agree_and_stay_within_the_alternating_pair() {
        let (f, epoch, mut toy) = tp_toy(2);
        for slots in [
            vec![Some(0), None],
            vec![Some(0), Some(1)],
            vec![Some(2), Some(2)],
        ] {
            toy.partial_slots = slots;
            assert!(submit(&f.plan, &epoch, &mut toy)
                .unwrap_err()
                .contains("partial slot"));
        }
        assert!(toy.calls.is_empty() && !toy.poisoned);
        for slots in [vec![], vec![Some(0), Some(0)], vec![Some(1), Some(1)]] {
            toy.partial_slots = slots;
            let ticket = submit(&f.plan, &epoch, &mut toy).unwrap();
            drain_tickets(&[ticket], &mut toy).unwrap();
            toy.state.last_op = None;
        }
    }

    #[test]
    fn tp_drain_failure_quarantines_and_refuses_later_epochs() {
        let (f, mut epoch, mut toy) = tp_toy(2);
        let ticket = submit(&f.plan, &epoch, &mut toy).unwrap();
        toy.fail_drain = Some(1);
        assert!(drain_tickets(&[ticket], &mut toy)
            .unwrap_err()
            .contains("quarantined"));
        assert!(toy.poisoned && toy.quarantined);
        assert_eq!(toy.syncs(), vec![0, 1]);
        let calls = toy.calls.len();
        epoch.op_id += 1;
        assert!(submit(&f.plan, &epoch, &mut toy).is_err());
        assert_eq!(toy.calls.len(), calls);
    }

    #[test]
    fn mesh_shape_admits_pure_pp_or_pure_tp_only() {
        let rect = |axes: &[(DimKind, usize)]| DeviceMesh::rect(axes).unwrap();
        for admitted in [
            DeviceMesh::single().unwrap(),
            rect(&[(DimKind::Pp, 4)]),
            rect(&[(DimKind::Tp, 4)]),
        ] {
            assert!(admit_mesh_shape(&admitted).is_ok());
        }
        for refused in [
            rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]),
            rect(&[(DimKind::Tp, 2), (DimKind::Ep, 2)]),
            rect(&[(DimKind::Pp, 2), (DimKind::Ep, 2)]),
            rect(&[(DimKind::Ep, 2)]),
        ] {
            assert!(admit_mesh_shape(&refused).is_err());
        }
    }

    /// One rank's buffers in the GPU TP oracle.
    struct OracleRank {
        x: GpuTensor,
        p: GpuTensor,
        q: GpuTensor,
        n: GpuTensor,
        p_src: GpuTensor,
        q_src: GpuTensor,
    }

    /// Bindings of the GPU oracle. The row projections stand in for `W_r · a_r`
    /// with per-rank, per-epoch partials; `n` copies the reduced `q`.
    struct OracleBindings {
        ranks: Vec<OracleRank>,
        ops: [OpId; 5],
        p: ResourceId,
        q: ResourceId,
        tp_epoch: u64,
        bytes: usize,
    }

    impl MeshStepBindings for OracleBindings {
        fn prepare_epoch(&mut self, _: &ForwardEpoch) -> Result<()> {
            Ok(())
        }
        fn run_ops(
            &mut self,
            owner: usize,
            gpu: &mut Gpu,
            epoch: &ForwardEpoch,
            ops: &[OpId],
        ) -> Result<()> {
            let rank = &self.ranks[owner];
            let stream = gpu.active_stream.as_ref().ok_or("no launch stream")?;
            // This epoch's slice of the per-epoch partials.
            let slice = (epoch.op_id as usize - 1) * self.bytes;
            for &op in ops {
                let (dst, src, offset) = if op == self.ops[1] {
                    (&rank.p, &rank.p_src, slice)
                } else if op == self.ops[3] {
                    (&rank.q, &rank.q_src, slice)
                } else if op == self.ops[4] {
                    (&rank.n, &rank.q, 0)
                } else {
                    continue;
                };
                gpu.hip
                    .memcpy_dtod_async_at(&dst.buf, 0, &src.buf, offset, self.bytes, stream)
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        }
        fn boundary(
            &self,
            owner: usize,
            c: &PlannedCollective,
            epoch: &ForwardEpoch,
        ) -> Result<BoundaryIo<'_>> {
            let rank = &self.ranks[owner];
            let view = match c.operand {
                operand if operand == self.p => &rank.p,
                operand if operand == self.q => &rank.q,
                _ => return Err("unknown collective operand".into()),
            };
            Ok(BoundaryIo {
                view,
                into: matches!(c.kind, CollectiveKind::AllReduceAdd { .. }).then_some(&rank.x),
                descriptor: EdgeDescriptor {
                    epoch: self.tp_epoch,
                    op: epoch.op_id,
                    rows: epoch.actual_rows,
                    dtype_bytes: 4,
                    bytes: self.bytes,
                },
                partial_slot: None,
            })
        }
        fn apply_rank_marks(&mut self, _: usize, _: &ForwardEpoch) -> Result<()> {
            Ok(())
        }
        fn repair_prefix_rank(
            &mut self,
            _: usize,
            _: &mut Gpu,
            _: &ForwardEpoch,
            _: usize,
            _: &[u32],
        ) -> Result<()> {
            Err("unused".into())
        }
        fn capture_session_rank(
            &mut self,
            _: usize,
            _: &mut Gpu,
            _: &ForwardEpoch,
        ) -> Result<MeshCheckpointId> {
            Err("unused".into())
        }
        fn probe_session_rank(
            &mut self,
            _: usize,
            _: &ForwardEpoch,
            _: MeshCheckpointId,
        ) -> Result<()> {
            Err("unused".into())
        }
        fn restore_session_rank(
            &mut self,
            _: usize,
            _: &mut Gpu,
            _: &ForwardEpoch,
            _: MeshCheckpointId,
        ) -> Result<()> {
            Err("unused".into())
        }
        fn reset_rank(&mut self, _: usize, _: &mut Gpu, _: &ForwardEpoch) -> Result<()> {
            Err("unused".into())
        }
        fn checkpoint_rank(
            &mut self,
            _: usize,
            _: &mut Gpu,
            _: &MeshCheckpointKey,
        ) -> Result<MeshCheckpointId> {
            Err("unused".into())
        }
        fn restore_rank(
            &mut self,
            _: usize,
            _: &mut Gpu,
            _: &MeshCheckpointKey,
            _: MeshCheckpointId,
        ) -> Result<()> {
            Err("unused".into())
        }
        fn validate_finalize_rank(
            &mut self,
            _: usize,
            _: &MeshCheckpointKey,
            _: MeshCheckpointId,
        ) -> Result<()> {
            Err("unused".into())
        }
        fn finalize_rank(&mut self, _: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {}
    }

    /// `x + (p0 + p1 + …)` and `q0 + q1 + …` through the executor must equal
    /// the single-device rank-order sum byte for byte on every rank, over three
    /// undrained epochs with distinct partials (a stale read fails), for the
    /// provisioned transport and for the forced host-staged one. Every oracle
    /// buffer is allocated after provisioning enabled peer access, which is
    /// where `hipMemcpyPeer` writes nothing. Rank count: `HIPFIRE_EMULATE_GPUS`,
    /// else every visible device (the physical runbook arm).
    #[test]
    #[ignore = "GPU: run with HIPFIRE_EMULATE_GPUS=2 and =4, or on N physical devices"]
    fn tp_collectives_match_single_device_rank_order_sum() {
        tp_collective_oracle(3);
    }

    /// The same oracle over a payload above `ROOTED_REDUCE_MIN_BYTES` (80 rows
    /// x 1024 x 4 = 320 KiB), which takes the rooted peer reduction for TP3+.
    #[test]
    #[ignore = "GPU: run with HIPFIRE_EMULATE_GPUS=4, or on N>2 physical devices"]
    fn tp_collectives_rooted_large_payload_match_rank_order_sum() {
        assert!(80 * TP_H * 4 > ROOTED_REDUCE_MIN_BYTES);
        tp_collective_oracle(80);
    }

    fn tp_collective_oracle(rows: usize) {
        use crate::mesh_placement::AdmittedRank;
        const EPOCHS: usize = 3;
        let ranks = crate::config::get().emulate_gpus.unwrap_or_else(|| {
            hip_bridge::HipRuntime::load()
                .and_then(|hip| hip.device_count())
                .unwrap() as usize
        });
        assert!(ranks >= 2, "needs HIPFIRE_EMULATE_GPUS>=2 or two devices");
        let count = rows * TP_H;
        let bytes = count * 4;
        let mut gpus = Gpus::init_tp(ranks, 1).unwrap();
        for gpu in &mut gpus.devices {
            gpu.ensure_capture_stream().unwrap();
        }
        let mesh = gpus.mesh.clone();
        let f = tp_fixture(&mesh);
        let admitted = |mesh: &DeviceMesh, gpus: &Gpus| AdmittedMesh {
            mesh: mesh.clone(),
            ranks: gpus
                .devices
                .iter()
                .enumerate()
                .map(|(logical, gpu)| AdmittedRank {
                    logical,
                    device_ordinal: gpu.device_id,
                    pci_bus_id: None,
                    arch: gpu.arch.clone(),
                })
                .collect(),
            edges: Vec::new(),
            family: "tp-oracle".into(),
            numerical_policy: "f32".into(),
        };
        // A TP×EP mesh is refused and leaves nothing provisioned.
        let composed = DeviceMesh::rect(&[(DimKind::Tp, ranks), (DimKind::Ep, 2)]).unwrap();
        let refused = gpus
            .provision_mesh_edges(
                &admitted(&composed, &gpus),
                &BandPlacement::uniform(&composed, 1).unwrap(),
                bytes,
            )
            .unwrap_err();
        assert!(refused.contains("EP are refused"), "{refused}");
        assert_eq!(
            (gpus.mesh_tp_epoch(), gpus.mesh_output_owner()),
            (None, None)
        );

        // Values over 16 binades, so an f32 sum depends on its order.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut draw = |len: usize| {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let unit = (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5;
                    unit * f32::powi(2.0, ((state >> 20) % 17) as i32 - 8)
                })
                .collect::<Vec<f32>>()
        };
        let x0 = draw(count);
        // Rank r's partials for every epoch, epoch-major.
        let ps: Vec<_> = (0..ranks).map(|_| draw(EPOCHS * count)).collect();
        let qs: Vec<_> = (0..ranks).map(|_| draw(EPOCHS * count)).collect();
        let at = |parts: &[Vec<f32>], e: usize, i: usize| -> Vec<f32> {
            parts.iter().map(|part| part[e * count + i]).collect()
        };
        let fold = |terms: &[f32]| terms[1..].iter().fold(terms[0], |sum, t| sum + t);
        let mut x = x0.clone();
        // Controls: per-rank accumulation `(x + p0) + p1 + …` and, for three
        // or more ranks, the reversed rank order.
        let mut sequential = x0.clone();
        let mut reversed = x0.clone();
        for e in 0..EPOCHS {
            for i in 0..count {
                let terms = at(&ps, e, i);
                x[i] += fold(&terms);
                sequential[i] = terms.iter().fold(sequential[i], |acc, t| acc + t);
                let back: Vec<f32> = terms.iter().rev().copied().collect();
                reversed[i] += fold(&back);
            }
        }
        let n: Vec<f32> = (0..count).map(|i| fold(&at(&qs, EPOCHS - 1, i))).collect();
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_ne!(
            bits(&x),
            bits(&sequential),
            "control: summation order must matter"
        );
        if ranks >= 3 {
            assert_ne!(bits(&x), bits(&reversed), "control: rank order must matter");
        }
        let digest = |values: &[f32]| {
            values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
        };

        for (arm, force_host) in [("provisioned", false), ("host", true)] {
            gpus.provision_mesh_edges(
                &admitted(&mesh, &gpus),
                &BandPlacement::uniform(&mesh, 1).unwrap(),
                bytes,
            )
            .unwrap();
            let tp = gpus.mesh_execution.tp.as_mut().unwrap();
            if force_host {
                tp.transport = TpTransport::Host;
            }
            let transport = tp.transport;
            let mut bindings = OracleBindings {
                ranks: Vec::new(),
                ops: f.ops,
                p: f.p,
                q: f.q,
                tp_epoch: gpus.mesh_tp_epoch().unwrap(),
                bytes,
            };
            for (gpu, (p, q)) in gpus.devices.iter_mut().zip(ps.iter().zip(&qs)) {
                gpu.bind_thread().unwrap();
                let mut upload = |data: &[f32]| gpu.upload_f32(data, &[data.len()]).unwrap();
                bindings.ranks.push(OracleRank {
                    x: upload(&x0),
                    p: upload(&vec![f32::NAN; count]),
                    q: upload(&vec![f32::NAN; count]),
                    n: upload(&vec![f32::NAN; count]),
                    p_src: upload(p),
                    q_src: upload(q),
                });
                gpu.hip.device_synchronize().unwrap();
            }
            let mut tickets = Vec::new();
            for op_id in 1..=EPOCHS as u64 {
                let epoch = ForwardEpoch {
                    mesh_epoch: mesh.epoch(),
                    load_id: 1,
                    request_id: 1,
                    op_id,
                    route: MeshRoute::Prefill,
                    start_position: (op_id as usize - 1) * rows,
                    actual_rows: rows,
                    route_rows: rows,
                    policy_id: 1,
                };
                tickets.push(
                    gpus.submit_mesh_epoch(&f.plan, &epoch, &mut bindings)
                        .unwrap(),
                );
            }
            gpus.drain_mesh_epochs(tickets).unwrap();
            for (rank, (gpu, buffers)) in gpus.devices.iter().zip(&bindings.ranks).enumerate() {
                gpu.bind_thread().unwrap();
                assert_eq!(
                    bits(&gpu.download_f32(&buffers.x).unwrap()),
                    bits(&x),
                    "TP{ranks} {transport:?} rank {rank} x"
                );
                assert_eq!(
                    bits(&gpu.download_f32(&buffers.n).unwrap()),
                    bits(&n),
                    "TP{ranks} {transport:?} rank {rank} n"
                );
            }
            eprintln!(
                "TP{ranks} oracle ({arm} transport {transport:?}): {count} f32 x {EPOCHS} epochs \
                 byte-identical on every rank; x fnv64 {:016x}, n fnv64 {:016x}, controls: \
                 sequential {:016x}, reversed {:016x}",
                digest(&x),
                digest(&n),
                digest(&sequential),
                digest(&reversed)
            );
            gpus.drain_mesh_edges().unwrap();
        }
    }

    #[test]
    fn missing_lease_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::new(2);
        toy.missing = true;
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn oversize_descriptor_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::new(2);
        toy.oversize = true;
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn wrong_descriptor_and_null_stream_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        for no_stream in [false, true] {
            let mut toy = Toy::new(2);
            toy.no_stream = no_stream;
            toy.wrong_descriptor = !no_stream;
            assert!(prepare(&plan, &epoch, &toy).is_err());
            assert!(toy.calls.is_empty());
        }
    }

    #[test]
    fn stage_error_poisons_and_stops_without_publication() {
        let (_, plan, epoch) = toy_plan(6, 2);
        let mut toy = Toy::new(6);
        toy.fail_ops = Some(0);
        let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
        assert!(issue(&mut prepared, &plan, &epoch, &mut toy).is_err());
        assert!(toy.poisoned);
        assert_eq!(toy.calls.len(), 1);
        assert!(matches!(toy.calls[0], Call::Ops(0, _)));
    }

    #[test]
    fn drain_failure_quarantines_after_attempting_every_owner() {
        let mut toy = Toy::new(6);
        toy.fail_drain = Some(1);
        assert!(drain(&mut toy).is_err());
        assert!(toy.poisoned && toy.quarantined);
        assert_eq!(toy.calls, (0..6).map(Call::Sync).collect::<Vec<_>>());
    }

    #[test]
    fn successful_drain_acknowledges_every_owner() {
        let mut toy = Toy::new(2);
        drain(&mut toy).unwrap();
        assert_eq!(toy.calls, vec![Call::Sync(0), Call::Sync(1)]);
        assert!(!toy.poisoned && !toy.quarantined);
    }

    #[test]
    fn malformed_duplicate_and_cycle_refused_before_calls() {
        let (_, mut plan, epoch) = toy_plan(2, 1);
        plan.ranks[0].items.push(RankItem::Op(OpId(0)));
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        let (_, mut plan, epoch) = toy_plan(2, 2);
        // Destination waits for chunk 1 before chunk 0, source cannot reuse
        // the ordered edge until chunk 0 has been received.
        let first = plan.ranks[1]
            .items
            .iter()
            .position(|x| *x == RankItem::Collective(CollectiveId(0)))
            .unwrap();
        let second = plan.ranks[1]
            .items
            .iter()
            .position(|x| *x == RankItem::Collective(CollectiveId(1)))
            .unwrap();
        plan.ranks[1].items.swap(first, second);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    fn at(epoch: &ForwardEpoch, op_id: u64) -> ForwardEpoch {
        ForwardEpoch { op_id, ..*epoch }
    }

    fn submit_n(
        toy: &mut Toy,
        plan: &SplitPlan,
        epoch: &ForwardEpoch,
        first_op: u64,
        n: u64,
    ) -> Vec<MeshTicket> {
        (first_op..first_op + n)
            .map(|op| submit(plan, &at(epoch, op), toy).unwrap())
            .collect()
    }

    #[test]
    fn n_outstanding_epochs_drain_with_one_sync_per_device() {
        let (_, plan, epoch) = toy_plan(3, 1);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 4);
        assert_eq!(toy.prepared, vec![10, 11, 12, 13]);
        assert_eq!(toy.state.outstanding.len(), 4);
        assert!(toy.syncs().is_empty(), "submission never waits on devices");
        let publishes = toy
            .calls
            .iter()
            .filter(|call| matches!(call, Call::Publish(_)))
            .count();
        assert_eq!(publishes, 4 * 2);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1, 2], "every device synchronized once");
        assert!(toy.state.outstanding.is_empty());
        assert!(!toy.poisoned && !toy.quarantined && !toy.state.poisoned);
        // The healthy mesh admits the next epoch (and even a new request) after
        // the drain, with the op id still strictly increasing.
        let next = ForwardEpoch {
            request_id: 99,
            ..at(&epoch, 14)
        };
        submit(&plan, &next, &mut toy).unwrap();
        assert!(submit(&plan, &at(&epoch, 14), &mut toy).is_err());
    }

    #[test]
    fn single_ticket_drain_keeps_its_meaning() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 3, 1);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1]);
        assert!(toy.state.outstanding.is_empty());
        // A second drain of the same (now stale) ticket is refused and does not
        // synchronize again.
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1]);
        // One ticket cannot drain two outstanding epochs.
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let mut tickets = submit_n(&mut toy, &plan, &epoch, 3, 2);
        let newest = tickets.pop().unwrap();
        assert!(drain_tickets(&[newest], &mut toy).is_err());
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert!(toy.syncs().is_empty());
        assert_eq!(toy.state.outstanding.len(), 2);
    }

    #[test]
    fn admission_requires_same_request_and_increasing_ops() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        submit_n(&mut toy, &plan, &epoch, 10, 2);
        let prepared = toy.prepared.len();
        let calls = toy.calls.len();
        let other_request = ForwardEpoch {
            request_id: epoch.request_id + 1,
            ..at(&epoch, 12)
        };
        let other_load = ForwardEpoch {
            load_id: epoch.load_id + 1,
            ..at(&epoch, 12)
        };
        let other_policy = ForwardEpoch {
            policy_id: epoch.policy_id + 1,
            ..at(&epoch, 12)
        };
        let stale_mesh = ForwardEpoch {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            ..at(&epoch, 12)
        };
        for refused in [
            at(&epoch, 10),
            at(&epoch, 9),
            at(&epoch, 11),
            other_request,
            other_load,
            other_policy,
            stale_mesh,
        ] {
            assert!(submit(&plan, &refused, &mut toy).is_err());
        }
        // Admission refusals have no side effects at all: no bindings hook, no
        // enqueue, no poisoning, outstanding epochs untouched.
        assert_eq!(toy.prepared.len(), prepared);
        assert_eq!(toy.calls.len(), calls);
        assert!(!toy.poisoned && !toy.state.poisoned);
        assert_eq!(toy.state.outstanding.len(), 2);
        submit(&plan, &at(&epoch, 12), &mut toy).unwrap();
        assert_eq!(toy.state.outstanding.len(), 3);
    }

    #[test]
    fn validation_refusal_keeps_outstanding_epochs_drainable() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 2);
        let calls = toy.calls.len();
        toy.fail_prepare = true;
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        toy.fail_prepare = false;
        toy.oversize = true;
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        toy.oversize = false;
        assert_eq!(toy.calls.len(), calls, "validation never enqueues");
        assert!(!toy.poisoned && !toy.state.poisoned);
        assert_eq!(toy.state.last_op, Some(11), "op id burned only by enqueue");
        drain_tickets(&tickets, &mut toy).unwrap();
        assert!(!toy.poisoned && !toy.quarantined);
    }

    #[test]
    fn out_of_order_foreign_and_stale_tickets_refused_before_any_sync() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let mut tickets = submit_n(&mut toy, &plan, &epoch, 10, 3);
        let mut foreign = Toy::provisioned(2, &epoch, 2);
        let foreign_tickets = submit_n(&mut foreign, &plan, &epoch, 10, 3);

        assert!(drain_tickets(&[], &mut toy).is_err());
        // Missing newest, missing oldest, or not all of them.
        assert!(drain_tickets(&tickets[..2], &mut toy).is_err());
        assert!(drain_tickets(&tickets[1..], &mut toy).is_err());
        // Out of order.
        tickets.swap(0, 2);
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        tickets.swap(0, 2);
        // Foreign issuer, whole set and mixed.
        assert!(drain_tickets(&foreign_tickets, &mut toy).is_err());
        let mixed = vec![
            MeshTicket {
                issuer: 1,
                epoch: tickets[0].epoch,
            },
            MeshTicket {
                issuer: 2,
                epoch: tickets[1].epoch,
            },
            MeshTicket {
                issuer: 1,
                epoch: tickets[2].epoch,
            },
        ];
        assert!(drain_tickets(&mixed, &mut toy).is_err());
        // Right issuer, epoch that was never outstanding.
        let forged: Vec<_> = tickets
            .iter()
            .map(|t| MeshTicket {
                issuer: t.issuer,
                epoch: ForwardEpoch {
                    request_id: t.epoch.request_id + 1,
                    ..t.epoch
                },
            })
            .collect();
        assert!(drain_tickets(&forged, &mut toy).is_err());

        // Every refusal left the state untouched: no sync, no poison, no
        // quarantine, all three epochs still outstanding and drainable.
        assert!(toy.syncs().is_empty());
        assert!(!toy.poisoned && !toy.quarantined && !toy.state.poisoned);
        assert_eq!(toy.state.outstanding.len(), 3);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1]);
        // Now stale.
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1]);
        assert!(foreign.syncs().is_empty());
    }

    #[test]
    fn mid_sequence_submission_failure_poisons_every_outstanding_ticket() {
        let (_, plan, epoch) = toy_plan(3, 2);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 2);
        toy.fail_ops = Some(0);
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        assert!(toy.poisoned && toy.state.poisoned);
        assert!(!toy.quarantined);
        // The failed epoch holds no ticket; the earlier ones stay outstanding.
        assert_eq!(toy.state.outstanding.len(), 2);
        assert_eq!(toy.state.last_op, Some(12), "failed op id is burned");
        // Poisoned mesh admits nothing more, even once the cause is gone.
        toy.fail_ops = None;
        assert!(submit(&plan, &at(&epoch, 13), &mut toy).is_err());
        // Drain still quiesces every device once and retires the tickets, but
        // none of them reports success.
        let error = drain_tickets(&tickets, &mut toy).unwrap_err();
        assert!(error.contains("failed before drain"), "{error}");
        assert_eq!(toy.syncs(), vec![0, 1, 2]);
        assert!(toy.state.outstanding.is_empty());
        assert!(toy.state.poisoned && !toy.quarantined);
        assert!(submit(&plan, &at(&epoch, 14), &mut toy).is_err());
    }

    #[test]
    fn drain_failure_with_outstanding_epochs_quarantines_like_s3() {
        let (_, plan, epoch) = toy_plan(3, 1);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 3);
        toy.fail_drain = Some(1);
        let error = drain_tickets(&tickets, &mut toy).unwrap_err();
        assert!(error.contains("quarantined"), "{error}");
        assert!(toy.poisoned && toy.quarantined);
        assert!(toy.state.poisoned && toy.state.quarantined);
        assert_eq!(toy.syncs(), vec![0, 1, 2], "every owner attempted once");
        // Quarantine is terminal: nothing is admitted, drained or re-synced.
        assert!(submit(&plan, &at(&epoch, 13), &mut toy).is_err());
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1, 2]);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::device_mesh::DeviceMesh;
    use std::collections::HashSet;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Ev {
        Checkpoint(usize),
        Restore(usize),
        Validate(usize),
        Finalize(usize),
    }

    #[derive(Default)]
    struct Fake {
        log: Vec<Ev>,
        fail_checkpoint: HashSet<usize>,
        fail_validate: HashSet<usize>,
        fail_restore: HashSet<usize>,
        /// Owner whose checkpoint_rank returns a ticket minted for owner+100.
        wrong_owner_ticket: Option<usize>,
    }

    impl MeshCheckpointCallbacks for Fake {
        fn checkpoint_rank(
            &mut self,
            owner: usize,
            key: &MeshCheckpointKey,
        ) -> Result<MeshCheckpointId> {
            self.log.push(Ev::Checkpoint(owner));
            if self.fail_checkpoint.contains(&owner) {
                return Err("fake checkpoint failure".to_string());
            }
            let minted = if self.wrong_owner_ticket == Some(owner) {
                owner + 100
            } else {
                owner
            };
            Ok(MeshCheckpointId::issue(*key, minted, 7000 + owner as u64))
        }

        fn restore_rank(
            &mut self,
            owner: usize,
            key: &MeshCheckpointKey,
            ticket: MeshCheckpointId,
        ) -> Result<()> {
            assert_eq!(ticket.owner(), owner);
            assert_eq!(ticket.local(), 7000 + owner as u64);
            assert_eq!(ticket.key(), *key);
            self.log.push(Ev::Restore(owner));
            if self.fail_restore.contains(&owner) {
                return Err("fake restore failure".to_string());
            }
            Ok(())
        }

        fn validate_finalize_rank(
            &mut self,
            owner: usize,
            _key: &MeshCheckpointKey,
            _ticket: MeshCheckpointId,
        ) -> Result<()> {
            self.log.push(Ev::Validate(owner));
            if self.fail_validate.contains(&owner) {
                return Err("fake validate failure".to_string());
            }
            Ok(())
        }

        fn finalize_rank(
            &mut self,
            owner: usize,
            _key: &MeshCheckpointKey,
            _ticket: MeshCheckpointId,
        ) {
            self.log.push(Ev::Finalize(owner));
        }
    }

    fn key() -> MeshCheckpointKey {
        MeshCheckpointKey {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            load_gen: 1,
            reset_gen: 2,
            request_gen: 3,
            window_gen: 4,
        }
    }

    fn set(owners: &[usize]) -> HashSet<usize> {
        owners.iter().copied().collect()
    }

    #[test]
    fn checkpoint_all_captures_trunk_then_head_with_keyed_tickets() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Checkpoint(3)
            ]
        );
        assert_eq!(cp.per_owner.len(), 3);
        assert_eq!(cp.head_owner(), 3);
        for (owner, id) in cp.per_owner.iter().enumerate() {
            assert_eq!((id.owner(), id.key()), (owner, k));
        }
        let head = cp.head.unwrap();
        assert_eq!((head.owner(), head.key()), (3, k));
        assert!(cp.check_shape(&k).is_ok());
    }

    #[test]
    fn checkpoint_all_without_head_has_no_head_callback() {
        let mut cb = Fake::default();
        let cp = checkpoint_all(&mut cb, &key(), 2, false).unwrap();
        assert!(cp.head.is_none());
        assert_eq!(cb.log, vec![Ev::Checkpoint(0), Ev::Checkpoint(1)]);
    }

    #[test]
    fn checkpoint_failure_restores_captured_owners_in_reverse() {
        let mut cb = Fake {
            fail_checkpoint: set(&[2]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 4, true).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Checkpoint);
        assert_eq!(err.owner, 2);
        assert!(!err.poisoned());
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn head_checkpoint_failure_restores_every_trunk_owner() {
        let mut cb = Fake {
            fail_checkpoint: set(&[3]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, true).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 3));
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Checkpoint(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn checkpoint_failure_keeps_restoring_past_restore_errors_and_poisons() {
        let mut cb = Fake {
            fail_checkpoint: set(&[3]),
            fail_restore: set(&[1]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, true).unwrap_err();
        assert!(err.poisoned());
        assert_eq!(err.restore_failures.len(), 1);
        assert_eq!(err.restore_failures[0].0, 1);
        // owner 0 was still restored after owner 1's restore failed
        assert_eq!(
            &cb.log[4..],
            &[Ev::Restore(2), Ev::Restore(1), Ev::Restore(0)]
        );
    }

    #[test]
    fn checkpoint_rejects_ticket_minted_for_another_owner() {
        let mut cb = Fake {
            wrong_owner_ticket: Some(1),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, false).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 1));
        assert_eq!(
            cb.log,
            vec![Ev::Checkpoint(0), Ev::Checkpoint(1), Ev::Restore(0)]
        );
    }

    #[test]
    fn checkpoint_zero_owners_is_refused_without_callbacks() {
        let mut cb = Fake::default();
        let err = checkpoint_all(&mut cb, &key(), 0, true).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(cb.log.is_empty());
    }

    #[test]
    fn finalize_validates_every_owner_before_any_finalize() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        finalize_all(&mut cb, &k, cp).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Finalize(0),
                Ev::Finalize(1),
                Ev::Finalize(2),
                Ev::Finalize(3),
            ]
        );
    }

    #[test]
    fn finalize_without_head_skips_head_owner() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, false).unwrap();
        cb.log.clear();
        finalize_all(&mut cb, &k, cp).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Finalize(0),
                Ev::Finalize(1)
            ]
        );
    }

    #[test]
    fn validate_failure_blocks_all_finalize_and_restores_every_owner_with_head() {
        let mut cb = Fake {
            fail_validate: set(&[1]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 1));
        assert!(!err.poisoned());
        assert!(!cb.log.iter().any(|e| matches!(e, Ev::Finalize(_))));
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn head_validate_failure_restores_head_and_all_trunk_owners_without_finalize() {
        let mut cb = Fake {
            fail_validate: set(&[3]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 3));
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn validate_failure_restores_everyone_even_when_some_restores_fail() {
        let mut cb = Fake {
            fail_validate: set(&[0]),
            fail_restore: set(&[3, 1]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert!(err.poisoned());
        let failed: Vec<usize> = err.restore_failures.iter().map(|(o, _)| *o).collect();
        assert_eq!(failed, vec![3, 1]);
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn restore_all_attempts_every_owner_and_reports_all_failures() {
        let mut cb = Fake {
            fail_restore: set(&[2, 0]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = restore_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Restore, 2));
        assert!(err.poisoned());
        let failed: Vec<usize> = err.restore_failures.iter().map(|(o, _)| *o).collect();
        assert_eq!(failed, vec![2, 0]);
        assert_eq!(
            cb.log,
            vec![
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0)
            ]
        );
    }

    #[test]
    fn restore_all_clean_path_succeeds() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, true).unwrap();
        cb.log.clear();
        restore_all(&mut cb, &k, cp).unwrap();
        assert_eq!(cb.log, vec![Ev::Restore(2), Ev::Restore(1), Ev::Restore(0)]);
    }

    #[test]
    fn stale_window_generation_is_refused_before_any_callback() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, true).unwrap();
        cb.log.clear();
        let newer = MeshCheckpointKey {
            window_gen: k.window_gen + 1,
            ..k
        };
        let err = finalize_all(&mut cb, &newer, cp.clone()).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(err.poisoned());
        let err = restore_all(&mut cb, &newer, cp).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(cb.log.is_empty());
    }

    #[test]
    fn foreign_mesh_epoch_and_reset_generation_are_distinct_keys() {
        let k = key();
        let other_mesh = MeshCheckpointKey {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            ..k
        };
        let other_reset = MeshCheckpointKey {
            reset_gen: k.reset_gen + 1,
            ..k
        };
        assert_ne!(k, other_mesh);
        assert_ne!(k, other_reset);
        let id = MeshCheckpointId::issue(k, 0, 1);
        let cp = MeshCheckpoint {
            per_owner: vec![id],
            head: None,
        };
        assert!(cp.check_shape(&other_mesh).is_err());
        assert!(cp.check_shape(&other_reset).is_err());
        assert!(cp.check_shape(&k).is_ok());
    }

    #[test]
    fn ticket_for_wrong_owner_slot_is_rejected_by_shape_check() {
        let k = key();
        let cp = MeshCheckpoint {
            per_owner: vec![
                MeshCheckpointId::issue(k, 0, 1),
                MeshCheckpointId::issue(k, 0, 2),
            ],
            head: Some(MeshCheckpointId::issue(k, 1, 3)),
        };
        let err = cp.check_shape(&k).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Shape, 1));
        // head ticket must carry owner index == trunk owner count
        let cp = MeshCheckpoint {
            per_owner: vec![MeshCheckpointId::issue(k, 0, 1)],
            head: Some(MeshCheckpointId::issue(k, 0, 2)),
        };
        assert_eq!(cp.check_shape(&k).unwrap_err().owner, 1);
    }

    #[test]
    fn coordinator_works_through_dyn_callbacks() {
        let mut cb = Fake::default();
        let k = key();
        let dyn_cb: &mut dyn MeshCheckpointCallbacks = &mut cb;
        let cp = checkpoint_all(dyn_cb, &k, 2, false).unwrap();
        finalize_all(dyn_cb, &k, cp).unwrap();
        assert_eq!(cb.log.len(), 6);
    }

    #[test]
    fn early_validate_failure_still_validates_later_healthy_owners() {
        let mut cb = Fake {
            fail_validate: set(&[0]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 0));
        for owner in 1..=3 {
            assert!(
                cb.log.contains(&Ev::Validate(owner)),
                "owner {owner} was not validated after an early failure"
            );
        }
        assert!(!cb.log.iter().any(|e| matches!(e, Ev::Finalize(_))));
    }

    #[test]
    fn first_of_several_validate_failures_is_reported() {
        let mut cb = Fake {
            fail_validate: set(&[1, 2]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, false).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!(err.owner, 1);
        assert!(cb.log.contains(&Ev::Validate(2)));
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::device_mesh::DeviceMesh;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Op {
        Bind,
        Checkpoint,
        Restore,
        Validate,
        Repair,
        Capture,
        Probe,
        RestoreSession,
        Reset,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Ev {
        Bind(usize),
        /// (owner, device)
        Checkpoint(usize, usize),
        /// (owner, device)
        Restore(usize, usize),
        Validate(usize),
        Finalize(usize),
        Repair(usize),
        Capture(usize),
        Probe(usize),
        RestoreSession(usize),
        Reset(usize),
        Sync(usize),
    }

    /// CPU backend driven by the production `*_window`/`*_all` helpers through
    /// `OwnerOps`/`DrainBackend`/`HasState`, exactly like `HipBackend`.
    struct Toy {
        ranks: usize,
        log: Vec<Ev>,
        fail: HashSet<(Op, usize)>,
        /// Fail the n-th (0-based, global) `synchronize` call.
        fail_sync_call: Option<usize>,
        sync_calls: usize,
        capture_wrong_owner: Option<usize>,
        capture_wrong_request: Option<usize>,
        repair_args: Vec<(usize, usize, Vec<u32>)>,
        /// The family does not implement prefix repair.
        no_repair: bool,
        poisoned: bool,
        quarantined: bool,
        state: MeshExecutionState,
    }

    fn mesh_epoch() -> MeshEpoch {
        static EPOCH: std::sync::LazyLock<MeshEpoch> =
            std::sync::LazyLock::new(|| DeviceMesh::rect(&[(DimKind::Pp, 3)]).unwrap().epoch());
        *EPOCH
    }

    fn stale_epoch() -> MeshEpoch {
        DeviceMesh::single().unwrap().epoch()
    }

    fn epoch() -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh_epoch(),
            load_id: 1,
            request_id: 2,
            op_id: 3,
            route: MeshRoute::Decode,
            start_position: 0,
            actual_rows: 1,
            route_rows: 1,
            policy_id: 4,
        }
    }

    fn key() -> MeshCheckpointKey {
        session_key(&epoch())
    }

    fn session_key(epoch: &ForwardEpoch) -> MeshCheckpointKey {
        MeshCheckpointKey {
            mesh_epoch: epoch.mesh_epoch,
            load_gen: epoch.load_id,
            reset_gen: 5,
            request_gen: epoch.request_id,
            window_gen: 6,
        }
    }

    impl Toy {
        fn new(ranks: usize) -> Self {
            let mut state = MeshExecutionState::default();
            state.mesh_epoch = Some(mesh_epoch());
            state.output_owner = Some(ranks - 1);
            state.issuer = 1;
            Self {
                ranks,
                log: Vec::new(),
                fail: HashSet::new(),
                fail_sync_call: None,
                sync_calls: 0,
                capture_wrong_owner: None,
                capture_wrong_request: None,
                repair_args: Vec::new(),
                no_repair: false,
                poisoned: false,
                quarantined: false,
                state,
            }
        }

        fn failing(mut self, op: Op, owner: usize) -> Self {
            self.fail.insert((op, owner));
            self
        }

        fn hit(&self, op: Op, owner: usize) -> Result<()> {
            if self.fail.contains(&(op, owner)) {
                Err(format!("injected {op:?} failure on {owner}"))
            } else {
                Ok(())
            }
        }

        fn count(&self, pred: impl Fn(&Ev) -> bool) -> usize {
            self.log.iter().filter(|ev| pred(ev)).count()
        }

        fn fail_closed(&self) -> bool {
            self.state.poisoned && self.state.quarantined
        }

        fn healthy(&self) -> bool {
            !self.state.poisoned && !self.state.quarantined
        }
    }

    impl DrainBackend for Toy {
        fn rank_count(&self) -> usize {
            self.ranks
        }
        fn synchronize(&mut self, owner: usize) -> Result<()> {
            self.log.push(Ev::Sync(owner));
            let call = self.sync_calls;
            self.sync_calls += 1;
            if self.fail_sync_call == Some(call) {
                Err("injected sync failure".into())
            } else {
                Ok(())
            }
        }
        fn poison(&mut self) {
            self.poisoned = true;
            self.state.poison();
        }
        fn quarantine(&mut self) {
            self.quarantined = true;
            self.state.quarantined = true;
        }
    }

    impl HasState for Toy {
        fn state(&mut self) -> &mut MeshExecutionState {
            &mut self.state
        }
    }

    impl OwnerOps for Toy {
        fn bind(&mut self, device: usize) -> Result<()> {
            self.log.push(Ev::Bind(device));
            self.hit(Op::Bind, device)
        }
        fn checkpoint(
            &mut self,
            owner: usize,
            device: usize,
            key: &MeshCheckpointKey,
        ) -> Result<MeshCheckpointId> {
            self.log.push(Ev::Checkpoint(owner, device));
            self.hit(Op::Checkpoint, owner)?;
            Ok(MeshCheckpointId::issue(*key, owner, 700 + owner as u64))
        }
        fn restore(
            &mut self,
            owner: usize,
            device: usize,
            key: &MeshCheckpointKey,
            ticket: MeshCheckpointId,
        ) -> Result<()> {
            assert_eq!((ticket.owner(), ticket.key()), (owner, *key));
            self.log.push(Ev::Restore(owner, device));
            self.hit(Op::Restore, owner)
        }
        fn validate_finalize(
            &mut self,
            owner: usize,
            _: &MeshCheckpointKey,
            _: MeshCheckpointId,
        ) -> Result<()> {
            self.log.push(Ev::Validate(owner));
            self.hit(Op::Validate, owner)
        }
        fn finalize(&mut self, owner: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {
            self.log.push(Ev::Finalize(owner));
        }
        fn admits_prefix_repair(&self) -> bool {
            !self.no_repair
        }
        fn repair_prefix(
            &mut self,
            owner: usize,
            _: &ForwardEpoch,
            keep: usize,
            tokens: &[u32],
        ) -> Result<()> {
            self.log.push(Ev::Repair(owner));
            self.repair_args.push((owner, keep, tokens.to_vec()));
            self.hit(Op::Repair, owner)
        }
        fn capture_session(
            &mut self,
            owner: usize,
            epoch: &ForwardEpoch,
        ) -> Result<MeshCheckpointId> {
            self.log.push(Ev::Capture(owner));
            self.hit(Op::Capture, owner)?;
            let minted = if self.capture_wrong_owner == Some(owner) {
                owner + 1
            } else {
                owner
            };
            let mut key = session_key(epoch);
            if self.capture_wrong_request == Some(owner) {
                key.request_gen += 1;
            }
            Ok(MeshCheckpointId::issue(key, minted, 900 + owner as u64))
        }
        fn probe_session(
            &mut self,
            owner: usize,
            _: &ForwardEpoch,
            _: MeshCheckpointId,
        ) -> Result<()> {
            self.log.push(Ev::Probe(owner));
            self.hit(Op::Probe, owner)
        }
        fn restore_session(
            &mut self,
            owner: usize,
            _: &ForwardEpoch,
            _: MeshCheckpointId,
        ) -> Result<()> {
            self.log.push(Ev::RestoreSession(owner));
            self.hit(Op::RestoreSession, owner)
        }
        fn reset(&mut self, owner: usize, _: &ForwardEpoch) -> Result<()> {
            self.log.push(Ev::Reset(owner));
            self.hit(Op::Reset, owner)
        }
    }

    /// Every GPU-touching callback is immediately preceded by a bind of the
    /// device it runs on.
    fn assert_bound_before_gpu_calls(log: &[Ev]) {
        for (i, ev) in log.iter().enumerate() {
            let device = match *ev {
                Ev::Checkpoint(_, device) | Ev::Restore(_, device) => device,
                Ev::Repair(owner)
                | Ev::Capture(owner)
                | Ev::Probe(owner)
                | Ev::RestoreSession(owner)
                | Ev::Reset(owner) => owner,
                _ => continue,
            };
            assert!(
                i > 0 && log[i - 1] == Ev::Bind(device),
                "{ev:?} not bound: {log:?}"
            );
        }
    }

    fn healthy_artifacts() -> (MeshCheckpoint, Vec<MeshCheckpointId>) {
        let mut toy = Toy::new(3);
        let cp = checkpoint_window(&mut toy, &key(), true).unwrap();
        let pieces = capture_session_all(&mut toy, &epoch()).unwrap();
        (cp, pieces)
    }

    #[test]
    fn owners_map_to_devices_and_head_to_output_owner() {
        let mut toy = Toy::new(3);
        toy.state.output_owner = Some(1);
        let k = key();
        let cp = checkpoint_window(&mut toy, &k, true).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Checkpoint(0, 0),
                Ev::Bind(1),
                Ev::Checkpoint(1, 1),
                Ev::Bind(2),
                Ev::Checkpoint(2, 2),
                Ev::Bind(1),
                Ev::Checkpoint(3, 1),
            ]
        );
        assert_eq!(cp.head.unwrap().owner(), 3);

        toy.log.clear();
        restore_window(&mut toy, &k, cp.clone()).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(1),
                Ev::Restore(3, 1),
                Ev::Bind(2),
                Ev::Restore(2, 2),
                Ev::Bind(1),
                Ev::Restore(1, 1),
                Ev::Bind(0),
                Ev::Restore(0, 0),
            ]
        );
        assert!(toy.healthy());

        // Validation binds each owner (head on the output device); the
        // infallible host-only finalize never binds.
        toy.log.clear();
        finalize_window(&mut toy, &k, cp).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Validate(0),
                Ev::Bind(1),
                Ev::Validate(1),
                Ev::Bind(2),
                Ev::Validate(2),
                Ev::Bind(1),
                Ev::Validate(3),
                Ev::Finalize(0),
                Ev::Finalize(1),
                Ev::Finalize(2),
                Ev::Finalize(3),
            ]
        );
    }

    #[test]
    fn headless_checkpoint_has_no_head_callbacks() {
        let mut toy = Toy::new(2);
        let cp = checkpoint_window(&mut toy, &key(), false).unwrap();
        assert!(cp.head.is_none());
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Checkpoint(0, 0),
                Ev::Bind(1),
                Ev::Checkpoint(1, 1)
            ]
        );
    }

    #[test]
    fn invalid_output_owner_mapping_is_refused_before_callbacks() {
        for output in [None, Some(3), Some(9)] {
            let mut toy = Toy::new(3);
            toy.state.output_owner = output;
            assert!(checkpoint_window(&mut toy, &key(), true).is_err());
            assert!(reset_all(&mut toy, &epoch()).is_err());
            assert!(toy.log.is_empty());
        }
    }

    #[test]
    fn stale_poisoned_quarantined_and_foreign_meshes_refused_before_any_callback() {
        let (cp, pieces) = healthy_artifacts();
        let mutations: [(&str, fn(&mut MeshExecutionState)); 6] = [
            ("quarantined", |s| s.quarantined = true),
            ("poisoned", |s| s.poisoned = true),
            ("unprovisioned", |s| s.mesh_epoch = None),
            ("stale mesh", |s| s.mesh_epoch = Some(stale_epoch())),
            ("other load", |s| s.scope = Some((99, 4))),
            ("other request outstanding", |s| {
                s.outstanding.push(ForwardEpoch {
                    request_id: 77,
                    ..epoch()
                })
            }),
        ];
        for (name, mutate) in &mutations {
            let mut toy = Toy::new(3);
            mutate(&mut toy.state);
            let e = epoch();
            let k = key();
            let err = checkpoint_window(&mut toy, &k, true).unwrap_err();
            assert_eq!(err.phase, MeshLifecyclePhase::Shape, "{name}");
            assert!(restore_window(&mut toy, &k, cp.clone()).is_err(), "{name}");
            assert!(finalize_window(&mut toy, &k, cp.clone()).is_err(), "{name}");
            assert!(
                repair_prefix_all(&mut toy, &e, 1, &[1, 2]).is_err(),
                "{name}"
            );
            assert!(capture_session_all(&mut toy, &e).is_err(), "{name}");
            assert!(
                restore_session_all(&mut toy, &e, &pieces).is_err(),
                "{name}"
            );
            assert!(reset_all(&mut toy, &e).is_err(), "{name}");
            assert!(toy.log.is_empty(), "{name}: callbacks ran: {:?}", toy.log);
        }
        // Foreign checkpoint key / epoch identities.
        let mut toy = Toy::new(3);
        let foreign_mesh = MeshCheckpointKey {
            mesh_epoch: stale_epoch(),
            ..key()
        };
        assert!(checkpoint_window(&mut toy, &foreign_mesh, true).is_err());
        let foreign_epoch = ForwardEpoch {
            mesh_epoch: stale_epoch(),
            ..epoch()
        };
        assert!(repair_prefix_all(&mut toy, &foreign_epoch, 0, &[]).is_err());
        assert!(reset_all(&mut toy, &foreign_epoch).is_err());
        // Policy change is refused for epoch calls once the mesh is pinned.
        toy.state.scope = Some((1, 99));
        assert!(repair_prefix_all(&mut toy, &epoch(), 0, &[]).is_err());
        assert!(toy.log.is_empty());
    }

    #[test]
    fn lifecycle_does_not_enforce_submit_op_ids_on_executed_epochs() {
        let mut toy = Toy::new(3);
        toy.state.last_op = Some(50);
        toy.state.scope = Some((1, 4));
        toy.state.outstanding.push(epoch());
        let cp = checkpoint_window(&mut toy, &key(), false).unwrap();
        restore_window(&mut toy, &key(), cp).unwrap();
        repair_prefix_all(&mut toy, &epoch(), 1, &[7]).unwrap();
        assert!(toy.healthy());
        assert_eq!(toy.state.last_op, Some(50));
    }

    #[test]
    fn restore_and_finalize_refuse_wrong_shape_or_key_before_callbacks() {
        let mut other = Toy::new(2);
        let two_owner = checkpoint_window(&mut other, &key(), true).unwrap();
        let (three_owner, _) = healthy_artifacts();
        let newer = MeshCheckpointKey {
            window_gen: 99,
            ..key()
        };
        let mut toy = Toy::new(3);
        // Two tickets on a three-device mesh.
        assert!(restore_window(&mut toy, &key(), two_owner.clone()).is_err());
        assert!(finalize_window(&mut toy, &key(), two_owner).is_err());
        // Stale window generation.
        assert!(restore_window(&mut toy, &newer, three_owner.clone()).is_err());
        assert!(finalize_window(&mut toy, &newer, three_owner).is_err());
        assert!(toy.log.is_empty());
        assert!(
            toy.healthy(),
            "pre-callback refusal does not poison the mesh"
        );
    }

    #[test]
    fn checkpoint_failures_quarantine_only_when_rollback_fails() {
        // Clean rollback: recoverable, mesh stays healthy.
        let mut toy = Toy::new(3).failing(Op::Checkpoint, 2);
        let err = checkpoint_window(&mut toy, &key(), true).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 2));
        assert!(toy.healthy());
        assert_bound_before_gpu_calls(&toy.log);
        assert_eq!(toy.count(|e| matches!(e, Ev::Restore(..))), 2);

        // Rollback failure: unknown state, fail closed.
        let mut toy = Toy::new(3)
            .failing(Op::Checkpoint, 2)
            .failing(Op::Restore, 1);
        let err = checkpoint_window(&mut toy, &key(), true).unwrap_err();
        assert!(err.poisoned());
        assert!(toy.fail_closed());
        assert_eq!(toy.count(|e| matches!(e, Ev::Restore(..))), 2);

        // A bind failure is the owner's callback failure: the owner's capture
        // never runs, the earlier owner is rolled back.
        let mut toy = Toy::new(3).failing(Op::Bind, 1);
        let err = checkpoint_window(&mut toy, &key(), false).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 1));
        assert!(!toy.log.contains(&Ev::Checkpoint(1, 1)));
        assert!(toy.log.contains(&Ev::Restore(0, 0)));
        assert!(toy.healthy());
    }

    #[test]
    fn finalize_and_restore_failures_quarantine_per_rollback_result() {
        let (cp, _) = healthy_artifacts();
        // Validation failure, clean rollback: nothing finalized, mesh healthy.
        let mut toy = Toy::new(3).failing(Op::Validate, 1);
        let err = finalize_window(&mut toy, &key(), cp.clone()).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 1));
        assert!(!err.poisoned());
        assert_eq!(toy.count(|e| matches!(e, Ev::Finalize(_))), 0);
        assert_eq!(toy.count(|e| matches!(e, Ev::Restore(..))), 4);
        assert!(toy.healthy());

        // Validation bind failure on a device whose restore bind fails too.
        let mut toy = Toy::new(3).failing(Op::Bind, 1);
        let err = finalize_window(&mut toy, &key(), cp.clone()).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 1));
        assert_eq!(toy.count(|e| matches!(e, Ev::Finalize(_))), 0);
        assert!(toy.fail_closed());

        // Restore failures always fail closed.
        let mut toy = Toy::new(3).failing(Op::Restore, 0);
        let err = restore_window(&mut toy, &key(), cp.clone()).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Restore, 0));
        assert!(toy.fail_closed());
        assert_eq!(toy.count(|e| matches!(e, Ev::Restore(..))), 4);
        assert_bound_before_gpu_calls(&toy.log);
    }

    #[test]
    fn session_restore_probes_every_owner_before_the_first_restore() {
        let (_, pieces) = healthy_artifacts();
        let mut toy = Toy::new(3);
        restore_session_all(&mut toy, &epoch(), &pieces).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Probe(0),
                Ev::Bind(1),
                Ev::Probe(1),
                Ev::Bind(2),
                Ev::Probe(2),
                Ev::Bind(0),
                Ev::RestoreSession(0),
                Ev::Bind(1),
                Ev::RestoreSession(1),
                Ev::Bind(2),
                Ev::RestoreSession(2),
            ]
        );
        assert!(toy.healthy());
    }

    #[test]
    fn failed_probe_restores_nothing_and_still_probes_every_owner() {
        let (_, pieces) = healthy_artifacts();
        for failing in [vec![0], vec![2], vec![0, 1]] {
            let mut toy = Toy::new(3);
            for owner in &failing {
                toy.fail.insert((Op::Probe, *owner));
            }
            let error = restore_session_all(&mut toy, &epoch(), &pieces).unwrap_err();
            assert!(
                error.contains(&format!("owner {}", failing[0])),
                "first failure is named: {error}"
            );
            assert_eq!(toy.count(|e| matches!(e, Ev::Probe(_))), 3, "{failing:?}");
            assert_eq!(toy.count(|e| matches!(e, Ev::RestoreSession(_))), 0);
            assert!(toy.healthy(), "nothing mutated, mesh stays usable");
        }
        // A bind failure while probing is a probe failure for that owner only.
        let mut toy = Toy::new(3).failing(Op::Bind, 1);
        assert!(restore_session_all(&mut toy, &epoch(), &pieces).is_err());
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Probe(0),
                Ev::Bind(1),
                Ev::Bind(2),
                Ev::Probe(2)
            ]
        );
        assert!(toy.healthy());
    }

    #[test]
    fn session_restore_failure_after_start_quarantines_and_stops() {
        let (_, pieces) = healthy_artifacts();
        let mut toy = Toy::new(3).failing(Op::RestoreSession, 1);
        let error = restore_session_all(&mut toy, &epoch(), &pieces).unwrap_err();
        assert!(error.contains("owner 1"), "{error}");
        assert_eq!(
            toy.log[6..],
            [
                Ev::Bind(0),
                Ev::RestoreSession(0),
                Ev::Bind(1),
                Ev::RestoreSession(1)
            ]
        );
        assert!(toy.fail_closed());
    }

    #[test]
    fn session_shape_key_and_owner_mismatches_refused_before_callbacks() {
        let (_, pieces) = healthy_artifacts();
        let e = epoch();
        let k = session_key(&e);
        let other_window = MeshCheckpointKey {
            window_gen: k.window_gen + 1,
            ..k
        };
        let foreign_load = MeshCheckpointKey { load_gen: 99, ..k };
        let foreign_request = MeshCheckpointKey {
            request_gen: 99,
            ..k
        };
        let foreign_mesh = MeshCheckpointKey {
            mesh_epoch: stale_epoch(),
            ..k
        };
        let all_with = |key: MeshCheckpointKey| -> Vec<MeshCheckpointId> {
            (0..3).map(|o| MeshCheckpointId::issue(key, o, 1)).collect()
        };
        let cases: Vec<(&str, Vec<MeshCheckpointId>)> = vec![
            ("empty", vec![]),
            ("too few", pieces[..2].to_vec()),
            (
                "too many",
                [pieces.clone(), vec![MeshCheckpointId::issue(k, 3, 1)]].concat(),
            ),
            ("swapped", vec![pieces[1], pieces[0], pieces[2]]),
            ("duplicated owner", vec![pieces[0], pieces[0], pieces[2]]),
            (
                "one key differs",
                vec![
                    pieces[0],
                    MeshCheckpointId::issue(other_window, 1, 1),
                    pieces[2],
                ],
            ),
            ("foreign load", all_with(foreign_load)),
            ("foreign request", all_with(foreign_request)),
            ("foreign mesh", all_with(foreign_mesh)),
        ];
        for (name, bad) in cases {
            let mut toy = Toy::new(3);
            assert!(restore_session_all(&mut toy, &e, &bad).is_err(), "{name}");
            assert!(toy.log.is_empty(), "{name}: {:?}", toy.log);
            assert!(toy.healthy(), "{name}");
        }
    }

    #[test]
    fn capture_session_is_ascending_validated_and_reports_named_failures() {
        let mut toy = Toy::new(3);
        let pieces = capture_session_all(&mut toy, &epoch()).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Capture(0),
                Ev::Bind(1),
                Ev::Capture(1),
                Ev::Bind(2),
                Ev::Capture(2)
            ]
        );
        for (owner, piece) in pieces.iter().enumerate() {
            assert_eq!((piece.owner(), piece.key()), (owner, key()));
        }

        // Family failure: reported, later owners untouched, mesh not poisoned.
        let mut toy = Toy::new(3).failing(Op::Capture, 1);
        let error = capture_session_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("owner 1"), "{error}");
        assert_eq!(toy.count(|e| matches!(e, Ev::Capture(_))), 2);
        assert!(toy.healthy());

        // Wrong ticket owner and wrong request key are named failures too.
        let mut toy = Toy::new(3);
        toy.capture_wrong_owner = Some(2);
        let error = capture_session_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("owner 2"), "{error}");
        let mut toy = Toy::new(3);
        toy.capture_wrong_request = Some(1);
        let error = capture_session_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("owner 1"), "{error}");
        assert_bound_before_gpu_calls(&toy.log);
    }

    #[test]
    fn repair_is_ascending_and_first_failure_quarantines() {
        let mut toy = Toy::new(3);
        repair_prefix_all(&mut toy, &epoch(), 5, &[1, 2, 3]).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Bind(0),
                Ev::Repair(0),
                Ev::Bind(1),
                Ev::Repair(1),
                Ev::Bind(2),
                Ev::Repair(2)
            ]
        );
        assert_eq!(toy.repair_args[2], (2, 5, vec![1, 2, 3]));
        assert!(toy.healthy());

        let mut toy = Toy::new(3).failing(Op::Repair, 1);
        let error = repair_prefix_all(&mut toy, &epoch(), 5, &[1]).unwrap_err();
        assert!(error.contains("owner 1"), "{error}");
        assert_eq!(
            toy.count(|e| matches!(e, Ev::Repair(_))),
            2,
            "stop on first"
        );
        assert!(toy.fail_closed());
        // Quarantine is terminal for every later lifecycle call.
        let calls = toy.log.len();
        assert!(repair_prefix_all(&mut toy, &epoch(), 5, &[1]).is_err());
        assert!(reset_all(&mut toy, &epoch()).is_err());
        assert_eq!(toy.log.len(), calls);

        // A bind failure is the owner's failure as well.
        let mut toy = Toy::new(3).failing(Op::Bind, 0);
        assert!(repair_prefix_all(&mut toy, &epoch(), 5, &[1]).is_err());
        assert_eq!(toy.log, vec![Ev::Bind(0)]);
        assert!(toy.fail_closed());

        // A family without prefix repair is refused before any callback and
        // the mesh stays healthy.
        let mut toy = Toy::new(3);
        toy.no_repair = true;
        let error = repair_prefix_all(&mut toy, &epoch(), 5, &[1]).unwrap_err();
        assert!(error.contains("not admitted"), "{error}");
        assert!(toy.log.is_empty());
        assert!(toy.healthy());
    }

    #[test]
    fn reset_drains_first_then_resets_every_owner_then_synchronizes_every_device() {
        let mut toy = Toy::new(3);
        toy.state.last_op = Some(9);
        toy.state.scope = Some((1, 4));
        toy.state.outstanding.push(epoch());
        reset_all(&mut toy, &epoch()).unwrap();
        assert_eq!(
            toy.log,
            vec![
                Ev::Sync(0),
                Ev::Sync(1),
                Ev::Sync(2),
                Ev::Bind(0),
                Ev::Reset(0),
                Ev::Bind(1),
                Ev::Reset(1),
                Ev::Bind(2),
                Ev::Reset(2),
                Ev::Sync(0),
                Ev::Sync(1),
                Ev::Sync(2),
            ]
        );
        assert!(toy.state.outstanding.is_empty());
        assert!(toy.healthy());
        assert_eq!(
            toy.state.last_op,
            Some(9),
            "op ids stay monotonic across resets"
        );
        assert_eq!(toy.state.scope, Some((1, 4)));
    }

    #[test]
    fn failed_predrain_never_resets_and_quarantines() {
        let mut toy = Toy::new(3);
        toy.fail_sync_call = Some(1);
        toy.state.outstanding.push(epoch());
        let error = reset_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("before any reset_rank"), "{error}");
        assert_eq!(toy.log, vec![Ev::Sync(0), Ev::Sync(1), Ev::Sync(2)]);
        assert!(toy.fail_closed());
        assert!(
            toy.quarantined,
            "failed drain quarantines like drain_mesh_epochs"
        );
        assert_eq!(toy.state.outstanding.len(), 1, "outstanding not retired");
    }

    #[test]
    fn reset_failures_do_not_stop_other_owners_or_the_final_sync() {
        let mut toy = Toy::new(3).failing(Op::Reset, 1).failing(Op::Bind, 2);
        let error = reset_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("owner 1 reset_rank"), "{error}");
        assert!(error.contains("owner 2 reset_rank"), "{error}");
        assert_eq!(toy.count(|e| matches!(e, Ev::Reset(_))), 2);
        assert!(toy.log.contains(&Ev::Reset(0)));
        assert!(toy.log.contains(&Ev::Reset(1)));
        assert_eq!(
            toy.log[toy.log.len() - 3..],
            [Ev::Sync(0), Ev::Sync(1), Ev::Sync(2)]
        );
        assert_eq!(toy.count(|e| matches!(e, Ev::Sync(_))), 6);
        assert!(toy.fail_closed());
        assert!(
            !toy.quarantined,
            "reset failure alone does not forget devices"
        );
        assert_bound_before_gpu_calls(&toy.log);
    }

    #[test]
    fn post_reset_sync_failure_quarantines_like_a_failed_drain() {
        let mut toy = Toy::new(3);
        toy.fail_sync_call = Some(4);
        let error = reset_all(&mut toy, &epoch()).unwrap_err();
        assert!(error.contains("owner 1 post-reset synchronize"), "{error}");
        assert_eq!(toy.count(|e| matches!(e, Ev::Reset(_))), 3);
        assert_eq!(toy.count(|e| matches!(e, Ev::Sync(_))), 6);
        assert!(toy.fail_closed() && toy.quarantined);
    }
}
