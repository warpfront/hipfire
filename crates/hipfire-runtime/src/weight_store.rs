// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Transactional fulfillment for the pure weight manifest.
//!
//! [`crate::weight_manifest::plan_manifest`] owns the CPU-only "where". This
//! module owns the narrow "how" pilot for a plain LLaMA Single target: a
//! source callback supplies already-resolved bytes and dtype, the store uploads
//! them, and the first failure explicitly rolls back every resident buffer.
//!
//! The store is not a model owner. It has no `Drop` implementation and never
//! frees GPU buffers implicitly. A carrier moves a committed transaction into
//! its existing `ArchModel` owner; that owner consumes the architecture-private
//! attached owner during the existing teardown path.
//! `WeightStoreAssembly::take` transfers a resident handle to the owner that is
//! assembling typed weights, and therefore removes the cell from the store's
//! cleanup set.
use crate::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use crate::hfq_parallel::{
    read_ranges_into_host, upload_ranges_to_device, DeviceRangeCopy, DirectReadStats, DirectSource,
    HostRangeCopy, HFQ_READER_LANES,
};
use crate::mesh_placement::BandPlacement;
use crate::model_source::{SourceFormat, SourcePayload, SourceRangeDescriptor};
use crate::offload::HOST_MAPPED_PAD_BYTES;
use crate::tp_shard::ExpertAssign;
use crate::weight_manifest::{placement_devices, ShardPolicy, WeightEntry, WeightResidency};
use rdna_compute::byte_arena::{ModelWeightArenaPlan, NamedByteArenaSpan};
use rdna_compute::{DType, Gpu, GpuTensor};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

thread_local! {
    static RESIDENT_ALLOCATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static RESIDENT_RELEASES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFTER_UPLOAD: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static ARENA_ROOT_ALLOCATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static ARENA_ROOT_RELEASES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ARENA_ROOT_QUARANTINES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Test-only allocation accounting and deterministic post-upload fault seam.
///
/// The production loader calls the same release path regardless of whether
/// this seam is armed. Callers should use [`reset`] before a scenario and
/// [`clear_faults`] after it so a failed test cannot poison a later one.
#[doc(hidden)]
pub mod test_support {
    use super::{
        ARENA_ROOT_ALLOCATIONS, ARENA_ROOT_QUARANTINES, ARENA_ROOT_RELEASES, FAIL_AFTER_UPLOAD,
        RESIDENT_ALLOCATIONS, RESIDENT_RELEASES,
    };

    pub fn reset() {
        RESIDENT_ALLOCATIONS.with(|count| count.set(0));
        RESIDENT_RELEASES.with(|count| count.set(0));
        ARENA_ROOT_ALLOCATIONS.with(|count| count.set(0));
        ARENA_ROOT_RELEASES.with(|count| count.set(0));
        ARENA_ROOT_QUARANTINES.with(|count| count.set(0));
        clear_faults();
    }

    pub fn arm_fail_after_upload(upload_number: usize) {
        assert!(upload_number > 0, "upload fault threshold must be non-zero");
        FAIL_AFTER_UPLOAD.with(|fault| fault.set(Some(upload_number)));
    }

    pub fn clear_faults() {
        FAIL_AFTER_UPLOAD.with(|fault| fault.set(None));
    }

    pub fn resident_allocations() -> usize {
        RESIDENT_ALLOCATIONS.with(std::cell::Cell::get)
    }

    pub fn resident_releases() -> usize {
        RESIDENT_RELEASES.with(std::cell::Cell::get)
    }

    /// Exact weight-arena roots allocated since [`reset`].
    pub fn arena_root_allocations() -> usize {
        ARENA_ROOT_ALLOCATIONS.with(std::cell::Cell::get)
    }

    /// Weight-arena roots released through the owner release path since [`reset`].
    pub fn arena_root_releases() -> usize {
        ARENA_ROOT_RELEASES.with(std::cell::Cell::get)
    }

    /// Weight-arena roots leaked (never released) because the device could
    /// not be quiesced since [`reset`].
    pub fn arena_root_quarantines() -> usize {
        ARENA_ROOT_QUARANTINES.with(std::cell::Cell::get)
    }

    pub(super) fn record_resident_upload() -> bool {
        let allocation = RESIDENT_ALLOCATIONS.with(|count| {
            let next = count.get() + 1;
            count.set(next);
            next
        });
        FAIL_AFTER_UPLOAD.with(|fault| {
            let should_fail = fault
                .get()
                .is_some_and(|upload_number| allocation >= upload_number);
            if should_fail {
                fault.set(None);
            }
            should_fail
        })
    }
}

/// Stable logical placement identity. Layer is part of the key because a
/// per-layer name such as `wq` appears once for every decoder block.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WeightPlacementKey {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

impl WeightPlacementKey {
    pub fn new(name: impl Into<String>, layer: Option<usize>, device: usize) -> Self {
        Self {
            name: name.into(),
            layer,
            device,
        }
    }
}

/// The immutable projection applied to one logical source before upload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightProjectionKind {
    Static,
    ColumnShard,
    RowShard,
    FusedQkv,
    HeadSharded,
    VocabShard,
    ExpertCompact,
    ExpertTensor,
}

/// Value-owned placement metadata. It contains no GPU or source-file
/// representation and remains stable after a handle is taken from the store.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WeightProjection {
    pub kind: WeightProjectionKind,
    pub axis: Option<usize>,
    pub rank: usize,
    pub world_size: usize,
    pub logical_shape: Vec<usize>,
    pub dtype: DType,
}

fn projection_for(
    entry: &WeightEntry,
    rank: usize,
    world_size: usize,
    dtype: DType,
) -> WeightProjection {
    let (kind, axis) = match &entry.policy {
        ShardPolicy::ColumnShard { axis } => (WeightProjectionKind::ColumnShard, Some(*axis)),
        ShardPolicy::RowShard { axis } => (WeightProjectionKind::RowShard, Some(*axis)),
        ShardPolicy::FusedQkv { .. } => (WeightProjectionKind::FusedQkv, None),
        ShardPolicy::HeadSharded { .. } => (WeightProjectionKind::HeadSharded, None),
        ShardPolicy::VocabShard { axis } => (WeightProjectionKind::VocabShard, Some(*axis)),
        ShardPolicy::ExpertSharded { .. } => (WeightProjectionKind::ExpertCompact, None),
        ShardPolicy::ExpertTensorSharded { .. } => (WeightProjectionKind::ExpertTensor, None),
        ShardPolicy::Replicate | ShardPolicy::Pin(_) | ShardPolicy::Tied { .. } => {
            (WeightProjectionKind::Static, None)
        }
    };
    WeightProjection {
        kind,
        axis,
        rank,
        world_size,
        logical_shape: entry.logical_shape.clone(),
        dtype,
    }
}

/// A resident GPU tensor or a symbolic alias to another logical source.
///
/// Aliases own no buffer. Resident buffers have no implicit destructor; the
/// current model owner explicitly consumes them through its teardown method.
pub enum WeightHandle {
    Resident(GpuTensor),
    Alias(String),
}

/// Identity captured at the start of a load. It is deliberately immutable and
/// contains only mesh generation, logical rank, and physical device identity.
/// No policy or source representation is smuggled into the origin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightOrigin {
    mesh_epoch: MeshEpoch,
    logical_rank: usize,
    physical_device: i32,
}

impl WeightOrigin {
    pub fn from_parts(mesh_epoch: MeshEpoch, logical_rank: usize, physical_device: i32) -> Self {
        Self {
            mesh_epoch,
            logical_rank,
            physical_device,
        }
    }

    pub fn for_single(mesh: &DeviceMesh, gpu: &Gpu) -> Self {
        Self::from_parts(mesh.epoch(), 0, gpu.device_id)
    }

    pub fn mesh_epoch(self) -> MeshEpoch {
        self.mesh_epoch
    }

    pub fn logical_rank(self) -> usize {
        self.logical_rank
    }

    pub fn physical_device(self) -> i32 {
        self.physical_device
    }
}

/// Errors that are detected before a store is allowed to release a resident
/// buffer. Origin mismatch always returns the store to the caller unchanged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WeightStoreError {
    OriginMismatch {
        expected: WeightOrigin,
        actual: WeightOrigin,
    },
    UnboundOrigin,
    DuplicatePlacement(WeightPlacementKey),
    MissingPlacement(WeightPlacementKey),
    InvalidTarget(String),
}

impl std::fmt::Display for WeightStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OriginMismatch { expected, actual } => write!(
                f,
                "weight store origin mismatch: expected {:?}, got {:?}",
                expected, actual
            ),
            Self::UnboundOrigin => write!(f, "weight store has no target origin"),
            Self::DuplicatePlacement(key) => write!(
                f,
                "duplicate weight placement {}[layer {:?}] on device {}",
                key.name, key.layer, key.device
            ),
            Self::MissingPlacement(key) => write!(
                f,
                "missing weight placement {}[layer {:?}] on device {}",
                key.name, key.layer, key.device
            ),
            Self::InvalidTarget(message) => write!(f, "invalid weight store target: {message}"),
        }
    }
}

impl std::error::Error for WeightStoreError {}

/// Error identifying the first failed manifest cell. The store has already
/// been rolled back before this value is returned by [`fulfill_manifest`].
#[derive(Debug)]
pub struct FulfillError {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
    pub reason: String,
}

impl std::fmt::Display for FulfillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fulfill_manifest: {}[layer {:?}] on device {}: {}",
            self.name, self.layer, self.device, self.reason
        )
    }
}

impl std::error::Error for FulfillError {}

/// Real exact-allocated roots of a bounded weight arena. They are separate
/// owners, not placements: a placement is a borrowed view into one of these
/// roots and never owns bytes. The roots stay live through assembly
/// (`take`/guard/`finalize`) and are released exactly once, after every view
/// and after the Gpu's pointer-keyed shadow bindings, by
/// [`WeightStore::release_with`].
struct WeightArenaRoots {
    plan: Arc<ModelWeightArenaPlan>,
    /// Plan root order; each is an owning exact allocation.
    roots: Vec<GpuTensor>,
}

impl WeightArenaRoots {
    fn span(&self, name: &str) -> Option<&NamedByteArenaSpan> {
        self.plan.weights.member(name)
    }

    /// Borrowed view of one planned member, with the manifest's logical
    /// shape and the payload's source dtype.
    fn view(
        &self,
        span: &NamedByteArenaSpan,
        shape: &[usize],
        dtype: DType,
    ) -> Result<GpuTensor, String> {
        let root = self.roots.get(span.root).ok_or_else(|| {
            format!(
                "arena member '{}' names root {} of {} live roots",
                span.name,
                span.root,
                self.roots.len()
            )
        })?;
        span.offset
            .checked_add(span.bytes)
            .filter(|&end| end <= root.buf.size())
            .ok_or_else(|| {
                format!(
                    "arena member '{}' range {}+{} exceeds root {} capacity {}",
                    span.name,
                    span.offset,
                    span.bytes,
                    span.root,
                    root.buf.size()
                )
            })?;
        Ok(GpuTensor {
            buf: root.buf.byte_view(span.offset, span.bytes),
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Whether `tensor` is a borrowed view that lies wholly inside one root.
    fn contains_view(&self, tensor: &GpuTensor) -> bool {
        if !tensor.buf.is_borrowed() {
            return false;
        }
        let ptr = tensor.buf.as_ptr() as usize;
        self.roots.iter().any(|root| {
            let base = root.buf.as_ptr() as usize;
            ptr.checked_sub(base)
                .and_then(|offset| offset.checked_add(tensor.buf.size()))
                .is_some_and(|end| end <= root.buf.size())
        })
    }
}

/// The device operations the store's release path needs. [`GpuRelease`] is
/// the only production implementation; CPU tests substitute a recording fake
/// so the ordering contract (shadow bindings, views, quiesce, roots) is
/// checkable without a GPU.
trait WeightRelease {
    /// Return an individually allocated resident to its owner.
    fn free_resident(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()>;
    /// Clear every pointer-keyed weight cache and arena shadow binding. Runs
    /// before any arena root can be released or its address reused.
    fn clear_arena_caches(&mut self);
    /// Drain the device so no queued kernel can still read a root.
    fn quiesce(&mut self) -> hip_bridge::HipResult<()>;
    /// Release one real arena root.
    fn release_root(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()>;
}

struct GpuRelease<'g>(&'g mut Gpu);

impl WeightRelease for GpuRelease<'_> {
    fn free_resident(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
        // Pool return, not a raw HIP free: residents were uploaded through
        // the pool (`upload_pooled_bytes`), and the steady-state teardown
        // (`WeightTensor::free_all` -> `Gpu::free_tensor`) releases to the
        // same pool. A raw `hip.free` here would orphan pool bookkeeping.
        self.0.free_tensor(tensor)
    }

    fn clear_arena_caches(&mut self) {
        self.0.invalidate_weight_caches();
    }

    fn quiesce(&mut self) -> hip_bridge::HipResult<()> {
        self.0.bind_thread()?;
        self.0.hip.device_synchronize()
    }

    fn release_root(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
        // Exact hipMalloc owner: freed immediately, never parked in the pool
        // (a parked root-sized buffer would be handed to the next load as an
        // oversized candidate).
        self.0.release_tensor_immediate(tensor)
    }
}

/// Load-side placement container. It records one immutable projection per
/// `(name, layer, device)` and captures the target origin once. The container
/// itself has no consuming teardown API; lifecycle transitions are represented
/// by [`WeightLoadTransaction`] and the architecture-private attached owner.
///
/// Two records survive after typed assembly takes every handle for
/// publication: the allocation `journal` (insertion order, so rollback frees
/// in exact reverse allocation order instead of `HashMap` iteration order)
/// and the projection/alias census. Assembly takes a handle out of
/// `placements` but never removes its projection or alias edge, so the
/// published transaction still carries the complete validated provenance
/// (every fulfilled identity, its immutable projection, and every tied alias
/// source) beneath the crate-private attached owner. This census owns
/// nothing: resident allocations belong to the typed architecture weights,
/// and the census is dropped with the store without freeing.
#[derive(Default)]
pub struct WeightStore {
    placements: HashMap<WeightPlacementKey, WeightHandle>,
    projections: HashMap<WeightPlacementKey, WeightProjection>,
    /// External row descriptors are census entries, not handles. They are
    /// intentionally absent from `placements` and the allocation journal.
    external_rows: HashMap<WeightPlacementKey, SourceRangeDescriptor>,
    /// Every inserted identity in allocation order. Entries taken by assembly
    /// stay journaled; rollback skips keys that are no longer resident, so a
    /// key is freed at most once.
    journal: Vec<WeightPlacementKey>,
    /// Tied-source edges (`lm_head -> token_embd`) for local aliases and
    /// separately materialized endpoints. Retained after assembly takes handles.
    aliases: HashMap<WeightPlacementKey, String>,
    origin: Option<WeightOrigin>,
    /// Logical (unpadded) payload bytes per resident placement, for the
    /// developer weight digest. A pooled device buffer may be larger than
    /// the payload it carries.
    logical_bytes: HashMap<WeightPlacementKey, usize>,
    /// A host-mapped owner allocated one entry ahead while the previous
    /// entry's direct read was in flight. It is the newest allocation, so
    /// [`WeightStore::release_unchecked`] frees it before any journaled
    /// resident.
    staged_host: Option<StagedHostAlloc>,
    /// Set once per fulfillment when the developer digest is enabled; with
    /// it clear, `logical_bytes` is never touched (zero cost off).
    capture_logical: bool,
    /// Real exact-allocated weight arena roots; `None` outside an S1 arena
    /// fulfillment. Deliberately not part of `placements`/`journal`: roots
    /// outlive every taken view and are released exactly once, last.
    arena: Option<WeightArenaRoots>,
}

/// See [`WeightStore::staged_host`]. The host address is stored as an integer
/// so the census stays free of raw pointers.
struct StagedHostAlloc {
    index: usize,
    len: usize,
    tensor: GpuTensor,
    host: usize,
}

/// The only owner that may roll back resident allocations before publication.
///
/// A transaction owns the store until the architecture carrier consumes it
/// into its crate-private attached owner. It deliberately has no implicit
/// `Drop` cleanup because the GPU is not available to a destructor.
pub struct WeightLoadTransaction {
    store: Option<WeightStore>,
}

impl std::fmt::Debug for WeightLoadTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightLoadTransaction")
            .field("origin", &self.origin())
            .field("len", &self.len())
            .finish()
    }
}

impl WeightLoadTransaction {
    pub fn new(store: WeightStore) -> Self {
        Self { store: Some(store) }
    }

    pub fn origin(&self) -> Option<WeightOrigin> {
        self.store.as_ref().and_then(WeightStore::origin)
    }

    pub fn len(&self) -> usize {
        self.store.as_ref().map_or(0, WeightStore::len)
    }

    /// `true` only when the store owns nothing: no placement handles AND no
    /// live arena root (a fully assembled arena store has empty placements
    /// but still owns its roots).
    pub fn is_empty(&self) -> bool {
        self.store.as_ref().is_none_or(WeightStore::is_empty)
    }

    pub fn contains(&self, name: &str, layer: Option<usize>, device: usize) -> bool {
        self.store
            .as_ref()
            .is_some_and(|store| store.contains(name, layer, device))
    }

    pub fn get(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&WeightHandle> {
        self.store
            .as_ref()
            .and_then(|store| store.get(name, layer, device))
    }
    /// Return the sealed source range recorded for an external-row entry.
    /// External rows deliberately have no `WeightHandle`.
    pub fn external_descriptor(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&SourceRangeDescriptor> {
        self.store
            .as_ref()
            .and_then(|store| store.external_descriptor(name, layer, device))
    }

    pub fn external_rows_len(&self) -> usize {
        self.store
            .as_ref()
            .map_or(0, WeightStore::external_rows_len)
    }
    pub fn inventory_len(&self) -> usize {
        self.store.as_ref().map_or(0, WeightStore::inventory_len)
    }

    /// The weight/shadow arena plan this load was fulfilled from, if any.
    pub fn weight_arena_plan(&self) -> Option<&Arc<ModelWeightArenaPlan>> {
        self.store.as_ref().and_then(WeightStore::weight_arena_plan)
    }

    /// ACTUAL live capacities (`buf.size`) of the exact-allocated weight
    /// roots, in plan root order; empty without live roots.
    pub fn weight_arena_root_capacities(&self) -> Vec<usize> {
        self.store
            .as_ref()
            .map_or_else(Vec::new, WeightStore::arena_root_capacities)
    }

    /// Planned root requests, in plan root order (what was asked of the
    /// exact allocator); empty without an arena.
    pub fn weight_arena_root_requests(&self) -> Vec<usize> {
        self.store
            .as_ref()
            .map_or_else(Vec::new, WeightStore::arena_root_requests)
    }

    pub fn projection(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&WeightProjection> {
        self.store
            .as_ref()
            .and_then(|store| store.projection(name, layer, device))
    }

    pub fn devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        self.store
            .as_ref()
            .map_or_else(Vec::new, |store| store.devices_for(name, layer))
    }

    /// Compare the unpublished transaction's captured target with an admitted
    /// owner identity. This read-only check is used before the carrier wraps
    /// the transaction in its private attached owner.
    pub fn validate_origin_value(&self, expected: WeightOrigin) -> Result<(), WeightStoreError> {
        self.store
            .as_ref()
            .map_or(Err(WeightStoreError::UnboundOrigin), |store| {
                store.validate_origin_value(expected)
            })
    }

    /// Tied-source identity for an alias or a separately materialized endpoint.
    /// Survives assembly like the projection census.
    pub fn alias_source(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&str> {
        self.store
            .as_ref()
            .and_then(|store| store.alias_source(name, layer, device))
    }

    /// Start typed assembly while this load is still unpublished.
    pub fn begin_assembly(&mut self) -> WeightStoreAssembly<'_> {
        self.store
            .as_mut()
            .expect("weight load transaction was already consumed")
            .begin_assembly()
    }

    /// Consume this transaction and release every resident handle it owns.
    /// This is intentionally the only rollback operation exposed by the
    /// lifecycle API. Successful frees are reflected in the resident-release
    /// accounting; any failed pool return is returned to the caller.
    pub fn rollback(mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.rollback_via(&mut GpuRelease(gpu))
    }

    fn rollback_via(&mut self, ops: &mut dyn WeightRelease) -> hip_bridge::HipResult<()> {
        match self.store.take() {
            Some(store) => store.release_with(ops),
            None => Ok(()),
        }
    }
}

impl WeightStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_origin(origin: WeightOrigin) -> Self {
        Self {
            placements: HashMap::new(),
            projections: HashMap::new(),
            external_rows: HashMap::new(),
            journal: Vec::new(),
            aliases: HashMap::new(),
            arena: None,
            origin: Some(origin),
            logical_bytes: HashMap::new(),
            staged_host: None,
            capture_logical: false,
        }
    }

    pub fn origin(&self) -> Option<WeightOrigin> {
        self.origin
    }

    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// `true` only when no placement handle AND no live arena root remains.
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty() && !self.has_live_weight_roots()
    }

    /// Whether real arena roots are live (they outlive taken views).
    pub fn has_live_weight_roots(&self) -> bool {
        self.arena
            .as_ref()
            .is_some_and(|arena| !arena.roots.is_empty())
    }

    pub fn weight_arena_plan(&self) -> Option<&Arc<ModelWeightArenaPlan>> {
        self.arena.as_ref().map(|arena| &arena.plan)
    }

    /// ACTUAL live capacity (`buf.size`) of every exact weight root, in plan
    /// root order; empty when no arena roots are live.
    pub fn arena_root_capacities(&self) -> Vec<usize> {
        self.arena.as_ref().map_or_else(Vec::new, |arena| {
            arena.roots.iter().map(|root| root.buf.size()).collect()
        })
    }

    /// Planned root requests of the arena plan, in plan root order.
    pub fn arena_root_requests(&self) -> Vec<usize> {
        self.arena
            .as_ref()
            .map_or_else(Vec::new, |arena| arena.plan.weights.roots.clone())
    }

    pub fn contains(&self, name: &str, layer: Option<usize>, device: usize) -> bool {
        self.placements
            .contains_key(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn get(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&WeightHandle> {
        self.placements
            .get(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn projection(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&WeightProjection> {
        self.projections
            .get(&WeightPlacementKey::new(name, layer, device))
    }
    /// Return the sealed source range recorded for an external-row entry.
    pub fn external_descriptor(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&SourceRangeDescriptor> {
        self.external_rows
            .get(&WeightPlacementKey::new(name, layer, device))
    }

    pub fn external_rows_len(&self) -> usize {
        self.external_rows.len()
    }

    /// External rows have no placement handles; callers that need to inspect
    /// their source identities use [`Self::external_descriptor`].
    pub fn external_devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        let mut devices: Vec<_> = self
            .external_rows
            .keys()
            .filter(|key| key.name == name && key.layer == layer)
            .map(|key| key.device)
            .collect();
        devices.sort_unstable();
        devices
    }

    pub fn devices_for(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        let mut devices: Vec<_> = self
            .placements
            .keys()
            .filter(|key| key.name == name && key.layer == layer)
            .map(|key| key.device)
            .collect();
        devices.sort_unstable();
        devices
    }

    /// Retained provenance census size: every fulfilled identity's projection,
    /// including external descriptors and cells whose handles assembly
    /// already took. Owns nothing.
    pub fn inventory_len(&self) -> usize {
        self.projections.len()
    }

    /// Tied-source identity for either kind of handle. Retained after assembly.
    pub fn alias_source(&self, name: &str, layer: Option<usize>, device: usize) -> Option<&str> {
        self.aliases
            .get(&WeightPlacementKey::new(name, layer, device))
            .map(String::as_str)
    }

    fn insert(
        &mut self,
        key: WeightPlacementKey,
        handle: WeightHandle,
        projection: WeightProjection,
    ) -> Result<(), WeightStoreError> {
        if self.placements.contains_key(&key) || self.external_rows.contains_key(&key) {
            return Err(WeightStoreError::DuplicatePlacement(key));
        }
        if let WeightHandle::Alias(source) = &handle {
            self.aliases.insert(key.clone(), source.clone());
        }
        self.journal.push(key.clone());
        self.placements.insert(key.clone(), handle);
        self.projections.insert(key, projection);
        Ok(())
    }

    fn note_logical_bytes(&mut self, key: &WeightPlacementKey, bytes: usize) {
        if self.capture_logical {
            self.logical_bytes.insert(key.clone(), bytes);
        }
    }
    /// Record an external range in the census without creating a placement
    /// handle or journal entry.
    fn record_external(
        &mut self,
        key: WeightPlacementKey,
        descriptor: SourceRangeDescriptor,
        projection: WeightProjection,
    ) -> Result<(), WeightStoreError> {
        if self.placements.contains_key(&key)
            || self.external_rows.contains_key(&key)
            || self.projections.contains_key(&key)
        {
            return Err(WeightStoreError::DuplicatePlacement(key));
        }
        self.external_rows.insert(key.clone(), descriptor);
        self.projections.insert(key, projection);
        Ok(())
    }

    /// Stage a symbolic alias without GPU work. Used for tied declarations and
    /// CPU ownership tests; aliases never participate in release.
    pub fn stage_alias(
        &mut self,
        name: impl Into<String>,
        layer: Option<usize>,
        device: usize,
        source: impl Into<String>,
        projection: WeightProjection,
    ) -> Result<(), WeightStoreError> {
        self.insert(
            WeightPlacementKey::new(name, layer, device),
            WeightHandle::Alias(source.into()),
            projection,
        )
    }

    /// Move a handle out of the store. This is private to the assembly
    /// capability so arbitrary store holders cannot independently tear down a
    /// resident allocation. The projection and alias edge stay behind as
    /// retained provenance; only the handle leaves.
    fn take(&mut self, name: &str, layer: Option<usize>, device: usize) -> Option<WeightHandle> {
        let key = WeightPlacementKey::new(name, layer, device);
        self.placements.remove(&key)
    }

    fn take_with_projection(
        &mut self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<(WeightHandle, WeightProjection)> {
        let key = WeightPlacementKey::new(name, layer, device);
        let handle = self.placements.remove(&key)?;
        let projection = self.projections.get(&key)?.clone();
        Some((handle, projection))
    }

    fn begin_assembly(&mut self) -> WeightStoreAssembly<'_> {
        WeightStoreAssembly {
            store: self,
            taken: Vec::new(),
            committed: false,
        }
    }

    /// Compare a store's captured origin with an already-resolved target
    /// identity. This read-only seam cannot release or extract any handle.
    pub fn validate_origin_value(&self, expected: WeightOrigin) -> Result<(), WeightStoreError> {
        let actual = self.origin.ok_or(WeightStoreError::UnboundOrigin)?;
        if actual != expected {
            return Err(WeightStoreError::OriginMismatch { expected, actual });
        }
        Ok(())
    }

    /// Verify that this store is still being handled by the same mesh/device
    /// target. No GPU calls occur on mismatch.
    pub fn validate_origin(&self, mesh: &DeviceMesh, gpu: &Gpu) -> Result<(), WeightStoreError> {
        self.validate_origin_value(WeightOrigin::for_single(mesh, gpu))
    }

    /// Explicit rollback for a failed transaction. It consumes the partial
    /// store and frees every resident buffer on the single owning GPU.
    fn rollback(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.release_unchecked(gpu)
    }

    /// Free every resident buffer in exact reverse allocation order. The
    /// allocation journal records insertion order, so unlike `HashMap`
    /// iteration the teardown sequence is deterministic. Keys already taken
    /// by assembly (or restored then re-taken) resolve to no placement and
    /// are skipped, so each resident is freed at most once; aliases own no
    /// buffer and are removed without GPU work.
    fn release_unchecked(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.release_with(&mut GpuRelease(gpu))
    }

    /// Release order (arena mode adds steps 1, 3 and 4):
    /// 1. clear the Gpu's pointer-keyed weight caches and arena shadow
    ///    bindings, so no address can be served stale once a root is gone;
    /// 2. free the newest staged host owner, then individually owned residents
    ///    in exact reverse allocation order; borrowed views inside a live root
    ///    own nothing and are skipped (any other borrowed handle still reaches
    ///    the owner's refusal);
    /// 3. quiesce the device;
    /// 4. release every real root exactly once, last. If the device cannot be
    ///    quiesced the roots are quarantined (leaked, never released while a
    ///    kernel may still read them) and the error is reported.
    fn release_with(mut self, ops: &mut dyn WeightRelease) -> hip_bridge::HipResult<()> {
        let mut first_error: Option<hip_bridge::HipError> = None;
        let arena = self.arena.take();
        if arena.is_some() {
            ops.clear_arena_caches();
        }
        if let Some(staged) = self.staged_host.take() {
            if let Err(error) = ops.free_resident(staged.tensor) {
                first_error.get_or_insert(error);
            }
        }
        for key in self.journal.drain(..).rev() {
            let handle = match self.placements.remove(&key) {
                Some(handle) => handle,
                None => continue,
            };
            if let WeightHandle::Resident(tensor) = handle {
                if arena
                    .as_ref()
                    .is_some_and(|arena| arena.contains_view(&tensor))
                {
                    continue;
                }
                match ops.free_resident(tensor) {
                    Ok(()) => {
                        RESIDENT_RELEASES.with(|count| count.set(count.get() + 1));
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        if let Some(arena) = arena {
            let roots = arena.roots;
            if !roots.is_empty() {
                match ops.quiesce() {
                    Ok(()) => {
                        for root in roots.into_iter().rev() {
                            match ops.release_root(root) {
                                Ok(()) => {
                                    ARENA_ROOT_RELEASES.with(|count| count.set(count.get() + 1));
                                }
                                Err(error) => {
                                    if first_error.is_none() {
                                        first_error = Some(error);
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => {
                        ARENA_ROOT_QUARANTINES.with(|count| count.set(count.get() + roots.len()));
                        if first_error.is_none() {
                            first_error = Some(hip_bridge::HipError::new(
                                error.code,
                                &format!(
                                    "{error}; {} weight arena roots quarantined (leaked, not released)",
                                    roots.len()
                                ),
                            ));
                        }
                        // `DeviceBuffer` has no destructor: dropping the
                        // handles leaks the allocations on purpose.
                        drop(roots);
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// One resident/alias handle temporarily moved during typed assembly.
pub struct TakenWeight {
    pub key: WeightPlacementKey,
    pub handle: WeightHandle,
    pub projection: WeightProjection,
}

/// Rollback-owning assembly transaction. Dropping it restores every taken cell
/// to the parent store; it never frees a GPU buffer implicitly.
pub struct WeightStoreAssembly<'a> {
    store: &'a mut WeightStore,
    taken: Vec<TakenWeight>,
    committed: bool,
}

impl<'a> WeightStoreAssembly<'a> {
    pub fn take(&mut self, name: &str, layer: Option<usize>, device: usize) -> Option<usize> {
        let key = WeightPlacementKey::new(name, layer, device);
        let (handle, projection) = self.store.take_with_projection(name, layer, device)?;
        let slot = self.taken.len();
        self.taken.push(TakenWeight {
            key,
            handle,
            projection,
        });
        Some(slot)
    }

    pub fn commit(self) -> WeightStoreAssemblyGuard<'a> {
        WeightStoreAssemblyGuard { inner: self }
    }
}

impl Drop for WeightStoreAssembly<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for taken in self.taken.drain(..) {
            let _ = self.store.insert(taken.key, taken.handle, taken.projection);
        }
    }
}

/// Guard retained while the typed architecture object is being built. If it
/// is dropped before `finalize`, all handles return to the parent store.
pub struct WeightStoreAssemblyGuard<'a> {
    inner: WeightStoreAssembly<'a>,
}

impl WeightStoreAssemblyGuard<'_> {
    pub fn get(&self, slot: usize) -> Option<&WeightHandle> {
        self.inner.taken.get(slot).map(|taken| &taken.handle)
    }

    pub fn projection(&self, slot: usize) -> Option<&WeightProjection> {
        self.inner.taken.get(slot).map(|taken| &taken.projection)
    }

    /// Transfer the taken handles to the existing ArchModel-owned typed
    /// weights. This is the sole operation that removes them from rollback
    /// ownership.
    pub fn finalize(mut self) -> Vec<TakenWeight> {
        self.inner.committed = true;
        std::mem::take(&mut self.inner.taken)
    }
}

fn target_error(mesh: &DeviceMesh) -> Option<FulfillError> {
    (mesh.n_devices() != 1).then(|| FulfillError {
        name: "<mesh>".to_string(),
        layer: None,
        device: 0,
        reason: format!(
            "Single fulfillment requires one logical device, got {}",
            mesh.n_devices()
        ),
    })
}
/// Upload raw weight bytes through the GPU buffer pool and retag the tensor
/// with its logical shape.
///
/// This is the pool-backed twin of `Gpu::upload_raw`: the buffer comes from
/// `Gpu::alloc_tensor` (exact byte size, `DType::Raw`) instead of a raw
/// `hip.malloc`, so teardown (`Gpu::free_tensor` → pool return) hands the
/// buffer back to the same pool the next fulfillment allocates from. A raw
/// upload paired with a pooled free retains one model's worth of VRAM in the
/// pool's free-lists per load/unload cycle without ever reusing it — driver
/// free VRAM declines every cycle while the pool counters look flat.
pub fn upload_pooled_bytes(
    gpu: &mut Gpu,
    bytes: &[u8],
    logical_shape: &[usize],
) -> hip_bridge::HipResult<GpuTensor> {
    let mut tensor = gpu.alloc_tensor(&[bytes.len()], DType::Raw)?;
    if let Err(error) = gpu.hip.memcpy_htod(&tensor.buf, bytes) {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    tensor.shape = logical_shape.to_vec();
    Ok(tensor)
}

/// Upload one resident-class payload according to its declared residency:
/// device-pooled for [`WeightResidency::Resident`], pinned device-mapped host
/// RAM for [`WeightResidency::HostMapped`].
fn upload_resident_bytes(
    gpu: &mut Gpu,
    residency: WeightResidency,
    bytes: &[u8],
    logical_shape: &[usize],
) -> hip_bridge::HipResult<GpuTensor> {
    match residency {
        WeightResidency::HostMapped => gpu.upload_raw_host_mapped(bytes, logical_shape),
        WeightResidency::Resident | WeightResidency::ExternalRows { .. } => {
            let started = std::time::Instant::now();
            let uploaded = upload_pooled_bytes(gpu, bytes, logical_shape);
            rdna_compute::load_phases::add(
                "weights.device_upload",
                rdna_compute::load_phases::ns_since(started),
                bytes.len() as u64,
            );
            uploaded
        }
    }
}
/// Name suffix of a layer's expert slab arena entry
/// (`<prefix>.layers.{L}.mlp.experts.slab_arena`): raw bytes holding every
/// expert of the layer back to back.
pub const SLAB_ARENA_SUFFIX: &str = ".mlp.experts.slab_arena";

/// Zeroed bytes every slab arena allocation carries past its last entry. The
/// MQ3-I4 grouped kernels load a dword scale pair per two-epoch trip, so the
/// last row of the last expert's down tensor is read up to 2 bytes past the
/// scale plane; a device allocation is exact-fit, so the pad is explicit. A
/// host-mapped arena already carries `Gpu::upload_raw_host_mapped`'s 1 MiB
/// zeroed tail.
pub const SLAB_ARENA_TAIL_PAD_BYTES: usize = 4096;

/// Tail pad a resident range upload of `entry` allocates and zeroes.
fn range_tail_pad(entry: &WeightEntry) -> usize {
    if entry.dtype == DType::Raw && entry.name.ends_with(SLAB_ARENA_SUFFIX) {
        SLAB_ARENA_TAIL_PAD_BYTES
    } else {
        0
    }
}

/// Maximum host staging allocation for a range upload. The final device tensor
/// is allocated once; only this bounded staging buffer is proportional to the
/// source range.
pub const RANGE_UPLOAD_CHUNK_BYTES: usize = 8 * 1024 * 1024;
fn next_range_chunk(offset: usize, total_len: usize, alignment: usize) -> usize {
    debug_assert!(alignment > 0);
    debug_assert!(offset < total_len);
    let max_chunk = (RANGE_UPLOAD_CHUNK_BYTES / alignment).max(1) * alignment;
    (total_len - offset).min(max_chunk)
}

pub(crate) fn source_dtype(dtype: &str) -> Result<DType, String> {
    let normalized = dtype.trim().to_ascii_uppercase();
    let parsed = match normalized.as_str() {
        "F32" | "FLOAT32" => DType::F32,
        "F16" | "FLOAT16" | "FP16" => DType::F16,
        "BF16" | "BFLOAT16" => DType::BF16,
        "Q4K" => DType::Q4K,
        "Q6K" => DType::Q6K,
        "Q8_0" | "Q8-0" => DType::Q8_0,
        "Q4F16G64" => DType::Q4F16G64,
        "Q4F16G32" => DType::Q4F16G32,
        "Q8HFQ" => DType::Q8HFQ,
        "HFQ4G256" => DType::HFQ4G256,
        "HFQ4G128" => DType::HFQ4G128,
        "HFQ3G256" => DType::HFQ3G256,
        "HFQ3G128" => DType::HFQ3G128,
        "HFQ2G256" => DType::HFQ2G256,
        "HFQ2G128" => DType::HFQ2G128,
        "TQ2G128" => DType::TQ2G128,
        "BQ1G128" => DType::BQ1G128,
        "HFQ6G256" => DType::HFQ6G256,
        "MQ4G256" => DType::MQ4G256,
        "MQ4G256V2" => DType::MQ4G256V2,
        "MQ4G128V2" => DType::MQ4G128V2,
        "MQ4CG256" => DType::MQ4CG256,
        "MQ6G256V2" => DType::MQ6G256V2,
        "MQ5G256V2" => DType::MQ5G256V2,
        "MQ3G256V2" => DType::MQ3G256V2,
        "MQ2G256V2" => DType::MQ2G256V2,
        "MQ4G128" => DType::MQ4G128,
        "MQ8G256" => DType::MQ8G256,
        "MQ6G256" => DType::MQ6G256,
        "MQ5G256" => DType::MQ5G256,
        "MQ3G256" => DType::MQ3G256,
        "MQ2G256" => DType::MQ2G256,
        "MQ2G256LLOYD" => DType::MQ2G256Lloyd,
        "MQ2G256LLOYDU" => DType::MQ2G256LloydU,
        "MQ3G256LLOYD" => DType::MQ3G256Lloyd,
        "MQ4G256LLOYD" => DType::MQ4G256Lloyd,
        "MQ2G256GL" => DType::MQ2G256GL,
        "MQ3G256GL" => DType::MQ3G256GL,
        "HFP4G32" => DType::HFP4G32,
        "MFP4G32" => DType::MFP4G32,
        "MFP4G32LLOYD" => DType::MFP4G32Lloyd,
        "MFP4G32P" => DType::MFP4G32P,
        "MFP4G32E8" => DType::MFP4G32E8,
        "MFP4G32E8SOA" => DType::MFP4G32E8SOA,
        "MFP4G32E8G128" => DType::MFP4G32E8G128,
        "MFP3G32E8" => DType::MFP3G32E8,
        "MFP2G32E8" => DType::MFP2G32E8,
        "PAROQ4G128" => DType::ParoQ4G128,
        "FP8E4M3B128" => DType::FP8E4M3B128,
        "FP8E4M3PT" => DType::FP8E4M3PT,
        "RAW" => DType::Raw,
        _ => {
            return Err(format!("unsupported source dtype '{dtype}'"));
        }
    };
    Ok(parsed)
}

/// Bytes one row of `row_elements` occupies when stored as `dtype`.
///
/// A row-addressed table may admit more than one tier — a Qwen4 PLE shard is
/// BF16 (320-byte rows) or Q8F16 (170-byte rows) — so the *declared* dtype picks
/// the stride and the manifest's `row_bytes` is only the default for dtypes
/// whose row layout is not derived here.
/// Bytes of one external (row-addressed) row, or `None` for a dtype that is
/// not a row tier.  A partial Q8_0 block is refused rather than padded: the
/// encoder only ever writes whole blocks.
pub fn external_row_stride(dtype: DType, row_elements: usize) -> Option<usize> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => dtype.row_bytes(row_elements),
        DType::Q8_0 if row_elements % 32 == 0 => dtype.row_bytes(row_elements),
        // qt=56: one code byte per value; the shard's 16-byte scale header is
        // not per row ([`external_range_bytes`]).
        DType::FP8E4M3PT => dtype.row_bytes(row_elements),
        _ => None,
    }
}

/// Exact byte extent of one external row range of `physical_rows` rows of
/// `row_bytes`: the rows plus, for a qt=56 shard, its per-tensor header.
pub fn external_range_bytes(dtype: DType, physical_rows: usize, row_bytes: usize) -> Option<usize> {
    let rows = physical_rows.checked_mul(row_bytes)?;
    match dtype {
        DType::FP8E4M3PT => rows.checked_add(rdna_compute::fp8_e4m3::PT_HEADER_BYTES),
        _ => Some(rows),
    }
}

fn quant_block_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Q4K => 144,
        DType::HFQ4G256 | DType::MQ4G256 | DType::MQ4G256V2 => 136,
        DType::MQ4G128V2 => 68,
        DType::MQ4CG256 => 136,
        DType::HFQ6G256 | DType::MQ6G256 | DType::MQ6G256V2 => 200,
        DType::Q8_0 => 34,
        DType::Q4F16G64 => 36,
        DType::Q4F16G32 => 20,
        DType::HFQ4G128 | DType::MQ4G128 => 72,
        DType::HFQ3G256 | DType::MQ3G256 | DType::MQ3G256V2 => 104,
        DType::HFQ3G128 => 56,
        DType::HFQ2G256
        | DType::MQ2G256
        | DType::MQ2G256V2
        | DType::MQ2G256Lloyd
        | DType::MQ2G256LloydU
        | DType::MQ2G256GL => 72,
        DType::MQ5G256 | DType::MQ5G256V2 => 168,
        DType::MQ8G256 => 258,
        DType::MQ3G256Lloyd | DType::MQ3G256GL => 112,
        DType::MQ4G256Lloyd => 160,
        DType::HFP4G32
        | DType::MFP4G32
        | DType::MFP4G32Lloyd
        | DType::MFP4G32P
        | DType::MFP4G32E8
        | DType::MFP4G32E8SOA
        | DType::MFP4G32E8G128 => 16,
        DType::MFP3G32E8 => 13,
        DType::MFP2G32E8 => 9,
        DType::HFQ2G128 => 40,
        DType::TQ2G128 => 34,
        DType::BQ1G128 => 18,
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::ParoQ4G128 => 72,
        DType::ParoQ4G64 => 40,
        DType::Raw => 1,
        // Keep this arm in sync if a new storage dtype is added. `size()` is
        // a conservative byte alignment for formats without a block codec.
        other => other.size(),
    }
}

fn expected_float_bytes(shape: &[usize], dtype: DType) -> Option<usize> {
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return None;
    }
    shape
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .and_then(|elements| elements.checked_mul(dtype.size()))
}

/// Return the exact payload extent for formats whose packed bytes are
/// row-shaped rather than `bytes-per-element`.  `shape` may include stacked
/// expert dimensions; every dimension before K contributes to the row count.
/// In particular, MQ4G128V2 is `rows * ceil(K/128) * 68`, not a flat
/// `ceil(rows*K/128)` calculation.
fn expected_payload_bytes(shape: &[usize], dtype: DType) -> Result<Option<usize>, String> {
    if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(expected_float_bytes(shape, dtype));
    }
    let k_alignment = match dtype {
        DType::MQ4G256V2 => 256usize,
        DType::MQ4G128V2 => 1,
        // `[rows, K]` matrices only: the MQ4 FWHT rotation spans whole
        // 256-element groups of a row. Higher ranks are packed flat, e.g. the
        // GDN conv1d `[C, 1, 4]` (K = 4) in A3B mq4r files.
        DType::MQ4G256 if shape.len() == 2 => 256,
        // A `[rows, K]` HFQ4G128 matrix pads each row to whole groups. Files
        // quantized before `b4846285e` packed groups across rows when
        // `K % 128 != 0`; per-row kernels read past such a buffer. A conv1d
        // `[C, 1, W]` is packed flat.
        DType::HFQ4G128 if shape.len() == 2 => 1,
        // Floor-stride codecs: the kernels read `K / group` groups per row and
        // support no ragged tail, so a `[rows, K]` matrix needs K aligned.
        // Older quantizers packed ragged-K matrices flat across rows. A conv1d
        // `[C, 1, W]` is packed flat.
        DType::HFQ6G256 | DType::HFQ4G256 if shape.len() == 2 => 256,
        DType::HFQ2G128 | DType::HFQ3G128 if shape.len() == 2 => 128,
        DType::Q4F16G64 if shape.len() == 2 => 64,
        // qt=55 is expert-blocked, not row-strided: `[E, R, K]` (or `[R, K]`)
        // is E blocks of `b128_block_stride(R, K)` bytes.
        DType::FP8E4M3B128 => {
            let (experts, rows, k) = match *shape {
                [experts, rows, k] => (experts, rows, k),
                [rows, k] => (1, rows, k),
                _ => {
                    return Err(format!(
                        "{dtype:?} payload shape {shape:?} must be [experts, rows, K] or [rows, K]"
                    ))
                }
            };
            return rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, k)
                .filter(|_| experts != 0 && rows != 0 && k != 0)
                .map(Some)
                .ok_or_else(|| format!("{dtype:?} payload shape {shape:?} has no exact extent"));
        }
        _ => return Ok(None),
    };
    if shape.len() < 2 {
        return Err(format!(
            "{dtype:?} payload shape {:?} must include row and K dimensions",
            shape
        ));
    }
    let k = *shape.last().expect("shape length checked");
    if k == 0 {
        return Err(format!("{dtype:?} payload K dimension cannot be zero"));
    }
    if k % k_alignment != 0 {
        return Err(format!(
            "{dtype:?} payload requires K%{k_alignment}==0, got K={k}"
        ));
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| format!("{dtype:?} payload row count overflows usize"))?;
    let row_bytes = match dtype {
        // `DType::row_bytes` has no HFQ4G128 arm; ParoQ4G128 is its layout.
        DType::HFQ4G128 => DType::ParoQ4G128.row_bytes(k),
        DType::HFQ2G128 | DType::HFQ3G128 | DType::Q4F16G64 => {
            (k / k_alignment).checked_mul(quant_block_bytes(dtype))
        }
        _ => dtype.row_bytes(k),
    };
    row_bytes
        .and_then(|row| rows.checked_mul(row))
        .ok_or_else(|| format!("{dtype:?} payload byte length overflows usize"))
        .map(Some)
}

pub(crate) fn validate_payload_extent(
    shape: &[usize],
    dtype: DType,
    actual: usize,
) -> Result<(), String> {
    if let Some(expected) = expected_payload_bytes(shape, dtype)? {
        if actual != expected {
            return Err(format!(
                "source payload has {actual} bytes, expected {expected} for {dtype:?} {:?}",
                shape
            ));
        }
    }
    Ok(())
}

/// Checked byte length of a range payload: non-empty and a whole number of
/// `dtype` quant blocks.
fn range_payload_len(descriptor: &SourceRangeDescriptor, dtype: DType) -> Result<usize, String> {
    let payload_len = usize::try_from(descriptor.length)
        .map_err(|_| "source range length does not fit usize".to_string())?;
    if payload_len == 0 {
        return Err("source range has zero length".to_string());
    }
    let alignment = quant_block_bytes(dtype);
    if payload_len % alignment != 0 {
        return Err(format!(
            "source range length {payload_len} is not aligned to {alignment}-byte blocks for {dtype:?}"
        ));
    }
    Ok(payload_len)
}

/// Stream `payload_len` bytes of `descriptor` into `dst` starting at
/// `dst_offset`, through one bounded quant-block-aligned staging buffer. The
/// destination is the final tensor (a pooled allocation or an arena span);
/// no model-sized host buffer exists.
fn copy_range_chunks(
    gpu: &mut Gpu,
    descriptor: &SourceRangeDescriptor,
    dtype: DType,
    payload_len: usize,
    dst: &hip_bridge::DeviceBuffer,
    dst_offset: usize,
) -> Result<(), String> {
    let alignment = quant_block_bytes(dtype);
    let mut offset = 0usize;
    let (mut read_ns, mut h2d_ns) = (0u64, 0u64);
    let mut staging = vec![0u8; next_range_chunk(0, payload_len, alignment)];
    while offset < payload_len {
        let chunk_len = next_range_chunk(offset, payload_len, alignment);
        let chunk = &mut staging[..chunk_len];
        let absolute_offset = descriptor
            .offset
            .checked_add(offset as u64)
            .ok_or_else(|| "source range offset overflow".to_string())?;
        let read_started = std::time::Instant::now();
        descriptor
            .read_exact_at(absolute_offset, chunk)
            .map_err(|error| format!("source range read failed: {error}"))?;
        read_ns += rdna_compute::load_phases::ns_since(read_started);
        let h2d_started = std::time::Instant::now();
        gpu.hip
            .memcpy_htod_offset(dst, dst_offset + offset, chunk)
            .map_err(|error| format!("pooled chunk upload failed: {error}"))?;
        h2d_ns += rdna_compute::load_phases::ns_since(h2d_started);
        // Read once per load: on unified memory the cached copy only
        // pushes other pages (the PLE table's) out of the page cache.
        descriptor.release_uploaded(absolute_offset, chunk_len as u64);
        offset += chunk_len;
    }
    rdna_compute::load_phases::add("weights.range_read", read_ns, payload_len as u64);
    rdna_compute::load_phases::add("weights.h2d", h2d_ns, payload_len as u64);
    Ok(())
}

fn upload_range_pooled(
    gpu: &mut Gpu,
    descriptor: &SourceRangeDescriptor,
    dtype: DType,
    logical_shape: &[usize],
    tail_pad: usize,
) -> Result<GpuTensor, String> {
    let payload_len = range_payload_len(descriptor, dtype)?;
    let mut tensor = gpu
        .alloc_tensor(&[payload_len + tail_pad], DType::Raw)
        .map_err(|error| format!("pooled allocation failed: {error}"))?;
    let result = (|| {
        if tail_pad != 0 {
            // A pooled buffer may be recycled: the pad must read as zeros.
            gpu.hip
                .memcpy_htod_offset(&tensor.buf, payload_len, &vec![0u8; tail_pad])
                .map_err(|error| format!("pooled tail pad upload failed: {error}"))?;
        }
        copy_range_chunks(gpu, descriptor, dtype, payload_len, &tensor.buf, 0)
    })();
    if let Err(error) = result {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    tensor.shape = logical_shape.to_vec();
    Ok(tensor)
}

/// Fulfillment-local lookahead for [`upload_range_host_direct`]: the next
/// host-mapped entry's payload (fetched once, consumed by its own iteration)
/// and the error of its early pinned allocation, reported when that entry's
/// turn comes so failures keep their sequential attribution.
#[derive(Default)]
struct HostLookahead<'a> {
    payload: Option<(usize, Result<SourcePayload<'a>, String>)>,
    alloc_error: Option<(usize, String)>,
}

/// Fetch the payload of `weights[next]` when it is a plain host-mapped entry
/// and report `(index, len)` when it is a non-empty source range, so its
/// pinned owner can be allocated while the current read runs.
fn peek_next_host_range<'a, F>(
    weights: &[WeightEntry],
    next: usize,
    ep: Option<EpFulfill>,
    source: &F,
    look: &mut HostLookahead<'a>,
) -> Option<(usize, usize)>
where
    F: Fn(&WeightEntry) -> Result<SourcePayload<'a>, String>,
{
    let entry = weights.get(next)?;
    if entry.residency != WeightResidency::HostMapped
        || matches!(entry.policy, ShardPolicy::Tied { .. })
        || (ep.is_some() && matches!(entry.policy, ShardPolicy::ExpertSharded { .. }))
    {
        return None;
    }
    let payload = source(entry);
    let len = match &payload {
        Ok(SourcePayload::Range(descriptor)) => usize::try_from(descriptor.length)
            .ok()
            .filter(|&len| len != 0),
        _ => None,
    };
    look.payload = Some((next, payload));
    len.map(|len| (next, len))
}

/// Per-tensor read buckets of a direct read. The reader emits no timers.
fn record_direct_stats(stats: &DirectReadStats) {
    use rdna_compute::load_phases::add;
    if stats.bytes_direct != 0 || stats.direct_read_ns != 0 {
        add(
            "weights.direct_read",
            stats.direct_read_ns,
            stats.bytes_direct,
        );
    }
    if stats.bytes_buffered != 0 || stats.buffered_read_ns != 0 {
        add(
            "weights.buffered_read",
            stats.buffered_read_ns,
            stats.bytes_buffered,
        );
    }
    if stats.h2d_ns != 0 {
        add(
            "weights.h2d",
            stats.h2d_ns,
            stats.bytes_direct + stats.bytes_buffered,
        );
    }
}

/// A direct read addresses absolute offsets of the file `direct` was opened
/// on; refuse a descriptor sealed against any other file (or an overlay).
fn check_direct_identity(
    direct: &DirectSource,
    descriptor: &SourceRangeDescriptor,
) -> Result<(), String> {
    match descriptor.source_identity().files.as_slice() {
        [file] if file == direct.file_identity() => Ok(()),
        _ => Err(
            "source range is not sealed against the direct reader's file; refusing direct read"
                .to_string(),
        ),
    }
}

/// Direct-read twin of [`upload_range_pooled`]: same pooled allocation, same
/// zeroed tail pad, but the whole file range streams through the pinned
/// double-buffered reader instead of the descriptor's positional reader.
fn upload_range_direct_device(
    gpu: &mut Gpu,
    direct: &DirectSource,
    descriptor: &SourceRangeDescriptor,
    dtype: DType,
    logical_shape: &[usize],
    tail_pad: usize,
) -> Result<GpuTensor, String> {
    let payload_len = usize::try_from(descriptor.length)
        .map_err(|_| "source range length does not fit usize".to_string())?;
    if payload_len == 0 {
        return Err("source range has zero length".to_string());
    }
    let alignment = quant_block_bytes(dtype);
    if payload_len % alignment != 0 {
        return Err(format!(
            "source range length {payload_len} is not aligned to {alignment}-byte blocks for {dtype:?}"
        ));
    }
    check_direct_identity(direct, descriptor)?;
    let alloc_started = std::time::Instant::now();
    let mut tensor = gpu
        .alloc_tensor(&[payload_len + tail_pad], DType::Raw)
        .map_err(|error| format!("pooled allocation failed: {error}"))?;
    rdna_compute::load_phases::add(
        "weights.device_alloc",
        rdna_compute::load_phases::ns_since(alloc_started),
        (payload_len + tail_pad) as u64,
    );
    let result = (|| {
        if tail_pad != 0 {
            // A pooled buffer may be recycled: the pad must read as zeros.
            gpu.hip
                .memcpy_htod_offset(&tensor.buf, payload_len, &vec![0u8; tail_pad])
                .map_err(|error| format!("pooled tail pad upload failed: {error}"))?;
        }
        let copy = [DeviceRangeCopy {
            file_offset: descriptor.offset,
            len: payload_len,
            dst: &tensor.buf,
            dst_offset: 0,
        }];
        let started = std::time::Instant::now();
        let stats = upload_ranges_to_device(gpu, direct, &copy, HFQ_READER_LANES)
            .map_err(|error| format!("direct source range upload failed: {error}"))?;
        rdna_compute::load_phases::add(
            "weights.device_upload",
            rdna_compute::load_phases::ns_since(started),
            payload_len as u64,
        );
        record_direct_stats(&stats);
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    descriptor.release_uploaded(descriptor.offset, descriptor.length);
    tensor.shape = logical_shape.to_vec();
    Ok(tensor)
}

fn read_host_range(
    direct: &DirectSource,
    file_offset: u64,
    len: usize,
    dst: usize,
) -> Result<DirectReadStats, String> {
    let copy = [HostRangeCopy {
        file_offset,
        len,
        dst: dst as *mut u8,
    }];
    // SAFETY: `dst` is the host pointer of a live pinned allocation of at
    // least `len` bytes that nothing else references until the caller joins
    // this read.
    unsafe { read_ranges_into_host(direct, &copy, HFQ_READER_LANES) }
        .map_err(|error| format!("direct source range read failed: {error}"))
}

/// Fill one host-mapped owner straight from the file. The owner comes from
/// `store.staged_host` when the previous entry staged it, else is allocated
/// now. While this entry's read runs on its own thread, the next host-mapped
/// entry's pinned owner (`next`) is allocated on the calling thread, which is
/// the only one that may touch `gpu`; allocation order stays manifest order.
/// An early allocation failure is deferred to `alloc_error` and surfaces at
/// that entry's own turn.
#[allow(clippy::too_many_arguments)]
fn upload_range_host_direct(
    gpu: &mut Gpu,
    store: &mut WeightStore,
    direct: &DirectSource,
    index: usize,
    descriptor: &SourceRangeDescriptor,
    shape: &[usize],
    next: Option<(usize, usize, &[usize])>,
    alloc_error: &mut Option<(usize, String)>,
) -> Result<GpuTensor, String> {
    let len = usize::try_from(descriptor.length)
        .map_err(|_| "source range length does not fit usize".to_string())?;
    if len == 0 {
        return Err("source range has zero length".to_string());
    }
    check_direct_identity(direct, descriptor)?;
    let (tensor, host) = match store.staged_host.take() {
        Some(staged) if staged.index == index && staged.len == len => {
            (staged.tensor, staged.host as *mut u8)
        }
        stale => {
            if let Some(stale) = stale {
                let _ = gpu.free_tensor(stale.tensor);
            }
            if let Some((failed, message)) = alloc_error.take() {
                if failed == index {
                    return Err(format!("host-mapped allocation failed: {message}"));
                }
            }
            gpu.alloc_raw_host_mapped(len, shape)
                .map_err(|error| format!("host-mapped allocation failed: {error}"))?
        }
    };
    let host_addr = host as usize;
    let file_offset = descriptor.offset;
    let (read, next_alloc) = std::thread::scope(|scope| {
        let reader = scope.spawn(move || read_host_range(direct, file_offset, len, host_addr));
        let next_alloc = next.map(|(next_index, next_len, next_shape)| {
            (
                next_index,
                next_len,
                gpu.alloc_raw_host_mapped(next_len, next_shape),
            )
        });
        let read = reader
            .join()
            .unwrap_or_else(|_| Err("direct reader thread panicked".to_string()));
        (read, next_alloc)
    });
    match next_alloc {
        Some((next_index, next_len, Ok((next_tensor, next_host)))) => {
            store.staged_host = Some(StagedHostAlloc {
                index: next_index,
                len: next_len,
                tensor: next_tensor,
                host: next_host as usize,
            });
        }
        Some((next_index, _, Err(error))) => {
            *alloc_error = Some((next_index, error.to_string()));
        }
        None => {}
    }
    let stats = match read {
        Ok(stats) => stats,
        Err(reason) => {
            // Newest allocation first, mirroring reverse-order rollback.
            if let Some(staged) = store.staged_host.take() {
                let _ = gpu.free_tensor(staged.tensor);
            }
            let _ = gpu.free_tensor(tensor);
            return Err(reason);
        }
    };
    // SAFETY: `host` is a live pinned allocation of `len +
    // HOST_MAPPED_PAD_BYTES` bytes and the read has completed.
    unsafe {
        std::ptr::write_bytes(host.add(len), 0, HOST_MAPPED_PAD_BYTES as usize);
    }
    record_direct_stats(&stats);
    descriptor.release_uploaded(descriptor.offset, descriptor.length);
    Ok(tensor)
}

/// Chunk size of device downloads while hashing a resident tensor.
const WEIGHT_DIGEST_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// Serializes digest appends of concurrent ranks/bands in one process; each
/// batch is also a single `O_APPEND` write, which keeps records whole across
/// processes.
static WEIGHT_DIGEST_APPEND: Mutex<()> = Mutex::new(());

fn append_digest_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let _guard = WEIGHT_DIGEST_APPEND
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(bytes)
}

/// Path of the developer weight digest, from `HIPFIRE_LOAD_WEIGHT_DIGEST`.
pub fn weight_digest_path() -> Option<PathBuf> {
    hipfire_config::memory::load_weight_digest_path()
}

/// Create or empty the digest file so a load starts from a clean record.
pub fn truncate_weight_digest(path: &Path) -> io::Result<()> {
    std::fs::File::create(path).map(|_| ())
}

/// Incremental SHA-256 for callers hashing a logical payload from several
/// spans.
pub struct Sha256Stream(Sha256);

impl Sha256Stream {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    pub fn finish_hex(self) -> String {
        hex_lower(&self.0.finalize())
    }
}

impl Default for Sha256Stream {
    fn default() -> Self {
        Self::new()
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 15)] as char);
    }
    out
}

/// SHA-256 of `bytes`, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// SHA-256 of `len` bytes at `offset` of a device buffer, downloaded in
/// bounded chunks so a whole-VRAM tensor is never staged on the host at once.
pub fn sha256_hex_device(
    gpu: &Gpu,
    buf: &hip_bridge::DeviceBuffer,
    offset: usize,
    len: usize,
) -> Result<String, String> {
    if offset.checked_add(len).is_none_or(|end| end > buf.size()) {
        return Err(format!(
            "digest range {offset}+{len} exceeds device buffer of {} bytes",
            buf.size()
        ));
    }
    gpu.bind_thread().map_err(|error| error.to_string())?;
    let mut stream = Sha256Stream::new();
    let mut chunk = vec![0u8; len.min(WEIGHT_DIGEST_CHUNK_BYTES)];
    let mut done = 0usize;
    while done < len {
        let take = (len - done).min(WEIGHT_DIGEST_CHUNK_BYTES);
        gpu.hip
            .memcpy_dtoh_at(&mut chunk[..take], buf, offset + done)
            .map_err(|error| format!("digest download failed: {error}"))?;
        stream.update(&chunk[..take]);
        done += take;
    }
    Ok(stream.finish_hex())
}

/// One JSONL digest record.
pub struct WeightDigestRow<'a> {
    pub name: &'a str,
    pub layer: Option<usize>,
    pub residency: &'a str,
    pub dtype: &'a str,
    pub logical_bytes: u64,
    pub sha256: &'a str,
}

fn weight_digest_line(row: &WeightDigestRow<'_>) -> String {
    serde_json::json!({
        "name": row.name,
        "layer": row.layer,
        "residency": row.residency,
        "dtype": row.dtype,
        "logical_bytes": row.logical_bytes,
        "sha256": row.sha256,
    })
    .to_string()
}

/// Append one record: whole line serialized first, then one locked
/// `O_APPEND` write.
pub fn append_weight_digest_line(path: &Path, row: &WeightDigestRow<'_>) -> io::Result<()> {
    let mut line = weight_digest_line(row).into_bytes();
    line.push(b'\n');
    append_digest_bytes(path, &line)
}

/// Append one digest record per resident placement of a freshly fulfilled
/// transaction, in allocation order. Hashes logical payload bytes only (the
/// exact stored size, so compact expert shards hash their shard): a
/// host-mapped owner is hashed in place, a device tensor is downloaded in
/// bounded chunks; tail pads are excluded. Aliases and external rows own no
/// bytes and are skipped. The whole batch is serialized first and appended in
/// one locked write, so concurrent ranks never interleave lines. Returns the
/// number of records written.
pub fn write_weight_digest(
    path: &Path,
    transaction: &WeightLoadTransaction,
    gpu: &Gpu,
) -> io::Result<usize> {
    let Some(store) = transaction.store.as_ref() else {
        return Ok(0);
    };
    let mut out = Vec::new();
    let mut rows = 0usize;
    for key in &store.journal {
        let Some(WeightHandle::Resident(tensor)) = store.placements.get(key) else {
            continue;
        };
        let logical = store
            .logical_bytes
            .get(key)
            .copied()
            .unwrap_or_else(|| tensor.buf.size());
        let (residency, sha256) = if tensor.buf.is_host_mapped() {
            let host = gpu.host_bytes(tensor).and_then(|host| host.get(..logical));
            let host = host.ok_or_else(|| {
                io::Error::other(format!("host-mapped digest range missing for {}", key.name))
            })?;
            ("host_mapped", sha256_hex(host))
        } else {
            let hash = sha256_hex_device(gpu, &tensor.buf, 0, logical).map_err(io::Error::other)?;
            ("device", hash)
        };
        let dtype = format!("{:?}", tensor.dtype);
        out.extend_from_slice(
            weight_digest_line(&WeightDigestRow {
                name: &key.name,
                layer: key.layer,
                residency,
                dtype: &dtype,
                logical_bytes: logical as u64,
                sha256: &sha256,
            })
            .as_bytes(),
        );
        out.push(b'\n');
        rows += 1;
    }
    append_digest_bytes(path, &out)?;
    Ok(rows)
}

fn rollback_fulfill_error(
    store: WeightStore,
    gpu: &mut Gpu,
    mut error: FulfillError,
) -> FulfillError {
    if let Err(release_error) = store.rollback(gpu) {
        error
            .reason
            .push_str(&format!("; resident rollback failed: {release_error}"));
    }
    error
}

/// A nonlocal tied endpoint reads the original source identity, but uploads
/// under the tied entry's own placement key. Local sources remain aliases.
fn fulfillment_source(entry: &WeightEntry) -> Cow<'_, WeightEntry> {
    match &entry.policy {
        ShardPolicy::Tied { source } => {
            let mut request = entry.clone();
            request.name = source.clone();
            request.policy = ShardPolicy::Replicate;
            Cow::Owned(request)
        }
        _ => Cow::Borrowed(entry),
    }
}

/// The full logical manifest validates ties against the input declaration.
/// A scoped, explicitly output-pinned materialization may omit that input;
/// validate its source contract without adding a fulfillment/allocation entry.
fn validate_fulfillment_manifest(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
) -> Result<(), String> {
    let mut logical = Cow::Borrowed(weights);
    for entry in weights {
        if let ShardPolicy::Tied { source } = &entry.policy {
            if !logical
                .iter()
                .any(|candidate| candidate.name == *source && candidate.layer == entry.layer)
                && entry.placement
                    == crate::weight_manifest::PlacementHint::Pin(
                        crate::weight_manifest::PinTarget::Output,
                    )
            {
                logical
                    .to_mut()
                    .push(fulfillment_source(entry).into_owned());
            }
        }
    }
    crate::weight_manifest::validate_weight_layers(&logical, n_layers)
        .and_then(|_| crate::weight_manifest::validate_manifest(&logical, mesh))
}

/// Fulfill a manifest for a plain LLaMA Single target.
///
/// The source callback is the architecture-owned namespace seam and returns
/// raw bytes plus the actual source dtype. No file/GGUF/HFQ type crosses this
/// API. On the first source, dtype, or upload failure every earlier resident is
/// explicitly released in reverse allocation order before the error is returned.
///
/// `expected` is the target identity the carrier admitted at plan time (mesh
/// epoch, logical rank, physical device). It is bound **before the first
/// upload**: when the runtime mesh/GPU disagree with the admitted plan
/// identity, fulfillment fails with no resident allocation to roll back.
/// This is still a same-call-site binding — a genuinely independent
/// cross-owner admission authority does not exist in-tree yet, so the
/// post-publication attach check stays as the second gate.
pub fn fulfill_manifest_single<F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
{
    if let Some(error) = target_error(mesh) {
        return Err(error);
    }
    if let Err(reason) = validate_fulfillment_manifest(weights, mesh, n_layers) {
        return Err(FulfillError {
            name: "<manifest>".to_string(),
            layer: None,
            device: 0,
            reason,
        });
    }

    let origin = WeightOrigin::for_single(mesh, gpu);
    if origin != expected {
        return Err(FulfillError {
            name: "<origin>".to_string(),
            layer: None,
            device: 0,
            reason: format!(
                "admitted target identity {expected:?} does not match runtime mesh/GPU {origin:?}; refusing before upload"
            ),
        });
    }
    let placement = BandPlacement::uniform(mesh, n_layers).map_err(|reason| FulfillError {
        name: "<manifest>".to_string(),
        layer: None,
        device: 0,
        reason,
    })?;
    let mut store = WeightStore::with_origin(origin);
    for entry in weights {
        let devices = placement_devices(entry, mesh, &placement);
        if devices.as_slice() != [0] {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: devices.first().copied().unwrap_or(0),
                reason: format!("Single placement resolved to {:?}, expected [0]", devices),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        let key = WeightPlacementKey::new(&entry.name, entry.layer, 0);
        if entry.residency.is_external() {
            return Err(rollback_fulfill_error(
                store,
                gpu,
                fulfill_entry_error(
                    entry,
                    "external rows require fulfill_manifest_from_payloads",
                ),
            ));
        }
        if let ShardPolicy::Tied {
            source: source_name,
        } = &entry.policy
        {
            let source_dtype = match store.get(source_name, entry.layer, 0) {
                Some(WeightHandle::Resident(tensor)) => Some(tensor.dtype),
                Some(WeightHandle::Alias(_)) | None => None,
            };
            if let Some(actual_dtype) = source_dtype {
                if !entry.dtype_constraint.accepts(actual_dtype) {
                    let error = FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: 0,
                        reason: format!(
                            "tied source '{source_name}' actual dtype {actual_dtype:?} is excluded by constraint {:?}",
                            entry.dtype_constraint
                        ),
                    };
                    return Err(rollback_fulfill_error(store, gpu, error));
                }
                let projection = projection_for(entry, 0, 1, actual_dtype);
                if let Err(reason) =
                    store.insert(key, WeightHandle::Alias(source_name.clone()), projection)
                {
                    let error = FulfillError {
                        name: entry.name.clone(),
                        layer: entry.layer,
                        device: 0,
                        reason: reason.to_string(),
                    };
                    return Err(rollback_fulfill_error(store, gpu, error));
                }
                continue;
            }
        }

        let read_entry = fulfillment_source(entry);
        let (bytes, dtype) = match source(&read_entry) {
            Ok(value) => value,
            Err(reason) => {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!("source read failed: {reason}"),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
        };
        if !entry.dtype_constraint.accepts(dtype) {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: format!(
                    "source dtype {dtype:?} violates constraint {:?}",
                    entry.dtype_constraint
                ),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        if let Err(reason) = validate_payload_extent(&entry.logical_shape, dtype, bytes.len()) {
            return Err(rollback_fulfill_error(
                store,
                gpu,
                fulfill_entry_error(entry, reason),
            ));
        }
        let mut tensor = match upload_pooled_bytes(gpu, &bytes, &entry.logical_shape) {
            Ok(tensor) => tensor,
            Err(error) => {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!("pooled upload failed: {error}"),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
        };
        tensor.dtype = dtype;
        let projection = projection_for(entry, 0, 1, dtype);
        if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: reason.to_string(),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
        if let ShardPolicy::Tied { source } = &entry.policy {
            store.aliases.insert(
                WeightPlacementKey::new(&entry.name, entry.layer, 0),
                source.clone(),
            );
        }
        if test_support::record_resident_upload() {
            let error = FulfillError {
                name: entry.name.clone(),
                layer: entry.layer,
                device: 0,
                reason: "test fault injected after resident upload".into(),
            };
            return Err(rollback_fulfill_error(store, gpu, error));
        }
    }
    Ok(WeightLoadTransaction::new(store))
}

fn fulfill_entry_error(entry: &WeightEntry, reason: impl Into<String>) -> FulfillError {
    FulfillError {
        name: entry.name.clone(),
        layer: entry.layer,
        device: 0,
        reason: reason.into(),
    }
}

fn validate_payload_shape(entry: &WeightEntry, shape: &[usize]) -> Result<(), String> {
    if shape != entry.logical_shape.as_slice() {
        return Err(format!(
            "source logical shape {:?} does not match manifest {:?}",
            shape, entry.logical_shape
        ));
    }
    Ok(())
}

/// Validate the source-owned seal for a range payload before it is either
/// uploaded or retained as an external row descriptor.
///
/// HFQ ranges carry the complete indexed source manifest.  External Qwen4
/// rows must be obtained by name from that manifest; accepting a descriptor
/// whose byte extent merely happens to have the right shape would allow a
/// caller to substitute another tensor (or invent offsets in metadata JSON).
/// Non-HFQ ranges retain the generic source-compatible behavior because older
/// source implementations do not expose an indexed manifest.
fn validate_source_range_identity(
    entry: &WeightEntry,
    descriptor: &SourceRangeDescriptor,
    require_named_entry: bool,
) -> Result<(), String> {
    let identity = descriptor.source_identity();
    if identity.format != SourceFormat::Hfq {
        return Ok(());
    }
    let range = identity
        .manifest
        .iter()
        .find(|range| range.name == entry.name);
    let Some(range) = range else {
        if require_named_entry {
            let kind = if entry.residency.is_external() {
                "external entry"
            } else {
                "entry"
            };
            return Err(format!(
                "HFQ source identity has no indexed range for {kind} '{}'",
                entry.name
            ));
        }
        return Ok(());
    };
    if range.offset != descriptor.offset
        || range.length != descriptor.length
        || !range.dtype.eq_ignore_ascii_case(descriptor.dtype())
        || range.logical_shape != descriptor.logical_shape()
    {
        return Err(format!(
            "HFQ source identity range for '{}' does not match descriptor \
             (identity offset={} length={} dtype={} shape={:?}; descriptor \
             offset={} length={} dtype={} shape={:?})",
            entry.name,
            range.offset,
            range.length,
            range.dtype,
            range.logical_shape,
            descriptor.offset,
            descriptor.length,
            descriptor.dtype(),
            descriptor.logical_shape()
        ));
    }
    let file = identity.files.get(range.file_index).ok_or_else(|| {
        format!(
            "HFQ source identity range for '{}' refers to missing file index {}",
            entry.name, range.file_index
        )
    })?;
    let end = descriptor
        .offset
        .checked_add(descriptor.length)
        .ok_or_else(|| format!("HFQ source range for '{}' overflows", entry.name))?;
    if end > file.len {
        return Err(format!(
            "HFQ source range for '{}' ends at {end}, beyond file length {}",
            entry.name, file.len
        ));
    }
    Ok(())
}

fn validate_external_range(
    entry: &WeightEntry,
    dtype: DType,
    descriptor: &SourceRangeDescriptor,
) -> Result<(), String> {
    let WeightResidency::ExternalRows {
        row_bytes,
        valid_rows,
    } = entry.residency
    else {
        return Err("resident entry cannot be fulfilled by an external row range".into());
    };
    let physical_rows = entry.logical_shape.first().copied().unwrap_or(0);
    if valid_rows == 0 || valid_rows > physical_rows {
        return Err(format!(
            "external valid_rows={valid_rows} is outside physical rows 1..={physical_rows}"
        ));
    }
    if !entry.dtype_constraint.accepts(dtype) {
        return Err(format!(
            "external rows dtype {dtype:?} is excluded by constraint {:?}",
            entry.dtype_constraint
        ));
    }
    let row_elements = entry.logical_shape[1..]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| "external row shape overflows usize".to_string())?;
    let expected_row_bytes = external_row_stride(dtype, row_elements).unwrap_or(row_bytes);
    let expected_length = external_range_bytes(dtype, physical_rows, expected_row_bytes)
        .ok_or_else(|| "external range length overflows usize".to_string())?;
    let actual_length = usize::try_from(descriptor.length)
        .map_err(|_| "external range length does not fit usize".to_string())?;
    if actual_length != expected_length {
        return Err(format!(
            "external range has {actual_length} bytes, expected {expected_length} \
             for {physical_rows} physical rows of {dtype:?}"
        ));
    }
    Ok(())
}

/// One rank's share of an expert-parallel fulfillment (see
/// [`fulfill_manifest_from_payloads`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct EpFulfill {
    rank: usize,
    ranks: usize,
}

/// Experts rank `rank` of `ranks` owns under `Stride` assignment
/// (`e % ranks == rank`): `(n_experts + ranks - 1 - rank) / ranks`.
pub fn ep_local_expert_count(n_experts: usize, rank: usize, ranks: usize) -> usize {
    (n_experts + ranks - 1 - rank) / ranks
}

/// The owned experts' bytes of a stacked expert tensor under `Stride`
/// assignment: expert `e`'s extent `[e*B, (e+1)*B)` for every
/// `e % ranks == rank`, concatenated in ascending `e`.
pub fn compact_owned_experts(
    bytes: &[u8],
    n_experts: usize,
    rank: usize,
    ranks: usize,
    assign: ExpertAssign,
) -> Result<Vec<u8>, String> {
    compact_owned_expert_spans(
        bytes.len(),
        n_experts,
        rank,
        ranks,
        assign,
        |offset, dst| {
            dst.copy_from_slice(&bytes[offset..offset + dst.len()]);
            Ok(())
        },
    )
}

/// [`compact_owned_experts`] over a payload of `len` bytes that `read(offset,
/// dst)` fills span by span, so a source range is read only for the owned
/// experts.
fn compact_owned_expert_spans(
    len: usize,
    n_experts: usize,
    rank: usize,
    ranks: usize,
    assign: ExpertAssign,
    mut read: impl FnMut(usize, &mut [u8]) -> Result<(), String>,
) -> Result<Vec<u8>, String> {
    if assign != ExpertAssign::Stride {
        return Err(format!(
            "compact expert upload supports Stride assignment, got {assign:?}"
        ));
    }
    if ranks == 0 || rank >= ranks {
        return Err(format!("expert rank {rank} is outside {ranks} ranks"));
    }
    if n_experts < ranks {
        return Err(format!("{n_experts} experts cannot cover {ranks} ranks"));
    }
    if len == 0 || len % n_experts != 0 {
        return Err(format!(
            "{len} payload bytes do not split into {n_experts} equal experts"
        ));
    }
    let per_expert = len / n_experts;
    let mut compact = vec![0u8; per_expert * ep_local_expert_count(n_experts, rank, ranks)];
    for (slot, expert) in (rank..n_experts).step_by(ranks).enumerate() {
        read(
            expert * per_expert,
            &mut compact[slot * per_expert..(slot + 1) * per_expert],
        )?;
    }
    Ok(compact)
}

/// Validate one expert-stack payload against its manifest entry and upload
/// the rank's owned experts compactly (device-pooled or pinned host-mapped per
/// the entry's residency). Returns the tensor, shaped
/// `[ep_local_expert_count(n_experts, rank, ranks), ..]`,
/// and the source dtype.
fn upload_expert_compact(
    gpu: &mut Gpu,
    entry: &WeightEntry,
    ep: EpFulfill,
    n_experts: usize,
    assign: ExpertAssign,
    payload: SourcePayload<'_>,
) -> Result<(GpuTensor, DType, usize), String> {
    let compact_of =
        |bytes: &[u8]| compact_owned_experts(bytes, n_experts, ep.rank, ep.ranks, assign);
    let (dtype, compact) = match &payload {
        SourcePayload::Borrowed { info, bytes } => (
            checked_expert_payload(entry, &info.dtype, &info.shape, bytes.len())?,
            compact_of(bytes)?,
        ),
        SourcePayload::Owned { info, bytes } => (
            checked_expert_payload(entry, &info.dtype, &info.shape, bytes.len())?,
            compact_of(bytes)?,
        ),
        // Unified memory reads its weights through source ranges: only the
        // owned experts' spans are read, each released from the page cache once read.
        SourcePayload::Range(descriptor) => {
            let len = usize::try_from(descriptor.length)
                .map_err(|_| "source range length does not fit usize".to_string())?;
            let dtype =
                checked_expert_payload(entry, &descriptor.dtype, &descriptor.logical_shape, len)?;
            let read = |offset: usize, dst: &mut [u8]| {
                let absolute = descriptor
                    .offset
                    .checked_add(offset as u64)
                    .ok_or_else(|| "source range offset overflow".to_string())?;
                descriptor
                    .read_exact_at(absolute, dst)
                    .map_err(|error| format!("source range read failed: {error}"))?;
                descriptor.release_uploaded(absolute, dst.len() as u64);
                Ok(())
            };
            let compact =
                compact_owned_expert_spans(len, n_experts, ep.rank, ep.ranks, assign, read)?;
            (dtype, compact)
        }
    };
    let mut shape = entry.logical_shape.clone();
    shape[0] = ep_local_expert_count(n_experts, ep.rank, ep.ranks);
    let mut tensor = upload_resident_bytes(gpu, entry.residency, &compact, &shape)
        .map_err(|error| format!("pooled upload failed: {error}"))?;
    tensor.dtype = dtype;
    Ok((tensor, dtype, compact.len()))
}

fn arena_error(reason: impl Into<String>) -> FulfillError {
    FulfillError {
        name: "<arena>".to_string(),
        layer: None,
        device: 0,
        reason: reason.into(),
    }
}

/// Structural reason `entry` cannot be a view of the weight arena, if any.
/// The plan (not this predicate) names the members; this only refuses an
/// entry whose upload semantics an arena view cannot honor.
fn arena_entry_refusal(entry: &WeightEntry) -> Option<&'static str> {
    if entry.residency != WeightResidency::Resident {
        return Some("is not device-resident (host-mapped or external rows)");
    }
    if matches!(entry.policy, ShardPolicy::Tied { .. }) {
        return Some("is a tied alias");
    }
    if matches!(
        entry.policy,
        ShardPolicy::ExpertSharded { .. } | ShardPolicy::ExpertTensorSharded { .. }
    ) {
        return Some("is an expert-sharded tensor");
    }
    if range_tail_pad(entry) != 0 {
        return Some("carries a zeroed tail pad past its payload");
    }
    None
}

/// Input-plan validation against the manifest, run before any allocation.
/// The layout itself is the plan's ([`ModelWeightArenaPlan::validate`]); this
/// only binds its names to real, eligible, unambiguous manifest entries.
fn validate_weight_arena_manifest(
    plan: &ModelWeightArenaPlan,
    weights: &[WeightEntry],
) -> Result<(), String> {
    plan.validate()?;
    let mut by_name: HashMap<&str, (usize, &WeightEntry)> = HashMap::new();
    for entry in weights {
        by_name
            .entry(entry.name.as_str())
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, entry));
    }
    for member in &plan.weights.members {
        let Some(&(count, entry)) = by_name.get(member.name.as_str()) else {
            return Err(format!(
                "arena member '{}' has no manifest entry",
                member.name
            ));
        };
        if count != 1 {
            return Err(format!(
                "arena member '{}' names {count} manifest entries; names must be unique",
                member.name
            ));
        }
        if let Some(reason) = arena_entry_refusal(entry) {
            return Err(format!(
                "arena member '{}' {reason}; it cannot be an arena view",
                member.name
            ));
        }
    }
    for member in &plan.f16_shadows.members {
        if plan.weights.member(&member.name).is_none() {
            return Err(format!(
                "shadow owner '{}' is not a weight arena member",
                member.name
            ));
        }
        let entry = by_name
            .get(member.name.as_str())
            .map(|&(_, entry)| entry)
            .ok_or_else(|| format!("shadow owner '{}' has no manifest entry", member.name))?;
        let expected = match entry.logical_shape.as_slice() {
            [rows, cols] if entry.dtype == DType::BF16 => rows
                .checked_mul(*cols)
                .and_then(|elements| elements.checked_mul(2)),
            _ => None,
        };
        if expected != Some(member.bytes) {
            return Err(format!(
                "shadow owner '{}' is not a 2-D BF16 weight whose F16 shadow is {} bytes",
                member.name, member.bytes
            ));
        }
    }
    Ok(())
}

/// Exact-allocate every weight root of `plan` into `store`, keeping partial
/// progress in the store so a failure rolls back through the one release
/// path (roots last), then prove each ACTUAL capacity equals its request.
fn acquire_weight_roots(
    store: &mut WeightStore,
    plan: Arc<ModelWeightArenaPlan>,
    mut allocate: impl FnMut(usize) -> hip_bridge::HipResult<GpuTensor>,
) -> Result<(), String> {
    let requests = plan.weights.roots.clone();
    store.arena = Some(WeightArenaRoots {
        plan: Arc::clone(&plan),
        roots: Vec::with_capacity(requests.len()),
    });
    for (index, request) in requests.iter().enumerate() {
        let root = allocate(*request).map_err(|error| {
            format!("weight root {index} ({request} bytes) exact allocation failed: {error}")
        })?;
        ARENA_ROOT_ALLOCATIONS.with(|count| count.set(count.get() + 1));
        store
            .arena
            .as_mut()
            .expect("arena installed above")
            .roots
            .push(root);
    }
    plan.weights
        .validate_capacities(&store.arena_root_capacities(), plan.granule_bytes)
        .map_err(|error| format!("weight root capacity validation failed: {error}"))?;
    Ok(())
}

/// Real resident source pointers of the F16-shadow owners, in the plan's
/// sorted member order. Each must be its planned arena view (borrowed,
/// exactly at root base + planned offset, BF16 2-D), so a binding can never
/// name a pointer that is not the weight the plan sized the shadow for.
fn shadow_source_ptrs(store: &WeightStore, weights: &[WeightEntry]) -> Result<Vec<usize>, String> {
    let Some(arena) = store.arena.as_ref() else {
        return Ok(Vec::new());
    };
    let layers: HashMap<&str, Option<usize>> = weights
        .iter()
        .map(|entry| (entry.name.as_str(), entry.layer))
        .collect();
    let mut ptrs = Vec::with_capacity(arena.plan.f16_shadows.members.len());
    for member in &arena.plan.f16_shadows.members {
        let layer = layers
            .get(member.name.as_str())
            .copied()
            .ok_or_else(|| format!("shadow owner '{}' has no manifest entry", member.name))?;
        let Some(WeightHandle::Resident(tensor)) = store.get(&member.name, layer, 0) else {
            return Err(format!(
                "shadow owner '{}' has no resident handle",
                member.name
            ));
        };
        let span = arena
            .span(&member.name)
            .ok_or_else(|| format!("shadow owner '{}' has no weight arena span", member.name))?;
        let root = arena
            .roots
            .get(span.root)
            .ok_or_else(|| format!("shadow owner '{}' root is not live", member.name))?;
        let expected = (root.buf.as_ptr() as usize).checked_add(span.offset);
        let ptr = tensor.buf.as_ptr() as usize;
        if !tensor.buf.is_borrowed()
            || Some(ptr) != expected
            || tensor.dtype != DType::BF16
            || tensor.shape.len() != 2
        {
            return Err(format!(
                "shadow owner '{}' resident is not its planned BF16 arena view",
                member.name
            ));
        }
        ptrs.push(ptr);
    }
    Ok(ptrs)
}

/// Upload one Borrowed/Owned payload: into the entry's planned arena span
/// (direct copy to the final view, no host concatenation) when the store has
/// an arena naming it, otherwise through the existing per-tensor path.
fn upload_payload_bytes(
    gpu: &mut Gpu,
    store: &WeightStore,
    entry: &WeightEntry,
    dtype: DType,
    bytes: &[u8],
) -> Result<GpuTensor, String> {
    let planned = store
        .arena
        .as_ref()
        .and_then(|arena| arena.span(&entry.name).map(|span| (arena, span)));
    let Some((arena, span)) = planned else {
        return upload_resident_bytes(gpu, entry.residency, bytes, &entry.logical_shape)
            .map(|mut tensor| {
                tensor.dtype = dtype;
                tensor
            })
            .map_err(|error| format!("pooled upload failed: {error}"));
    };
    if bytes.len() != span.bytes {
        return Err(format!(
            "source payload has {} bytes, arena span of '{}' is {} bytes",
            bytes.len(),
            span.name,
            span.bytes
        ));
    }
    let view = arena.view(span, &entry.logical_shape, dtype)?;
    if !bytes.is_empty() {
        gpu.bind_thread()
            .map_err(|error| format!("arena upload failed: {error}"))?;
        gpu.hip
            .memcpy_htod_offset(&arena.roots[span.root].buf, span.offset, bytes)
            .map_err(|error| format!("arena upload failed: {error}"))?;
    }
    Ok(view)
}

/// [`upload_payload_bytes`] for a checked source range: bounded staging
/// straight into the arena span, or the existing pooled range upload.
fn upload_payload_range(
    gpu: &mut Gpu,
    store: &WeightStore,
    entry: &WeightEntry,
    descriptor: &SourceRangeDescriptor,
    dtype: DType,
    range_len: usize,
) -> Result<GpuTensor, String> {
    let planned = store
        .arena
        .as_ref()
        .and_then(|arena| arena.span(&entry.name).map(|span| (arena, span)));
    let Some((arena, span)) = planned else {
        return upload_range_pooled(
            gpu,
            descriptor,
            dtype,
            &entry.logical_shape,
            range_tail_pad(entry),
        )
        .map(|mut tensor| {
            tensor.dtype = dtype;
            tensor
        });
    };
    if range_len != span.bytes {
        return Err(format!(
            "source range has {range_len} bytes, arena span of '{}' is {} bytes",
            span.name, span.bytes
        ));
    }
    let payload_len = range_payload_len(descriptor, dtype)?;
    let view = arena.view(span, &entry.logical_shape, dtype)?;
    gpu.bind_thread()
        .map_err(|error| format!("arena upload failed: {error}"))?;
    copy_range_chunks(
        gpu,
        descriptor,
        dtype,
        payload_len,
        &arena.roots[span.root].buf,
        span.offset,
    )?;
    Ok(view)
}

/// Validate an expert-stack payload's dtype, shape and byte extent against its
/// manifest entry; returns the source dtype.
fn checked_expert_payload(
    entry: &WeightEntry,
    source_dtype_name: &str,
    source_shape: &[usize],
    len: usize,
) -> Result<DType, String> {
    let dtype = source_dtype(source_dtype_name)?;
    validate_payload_shape(entry, source_shape)?;
    if !entry.dtype_constraint.accepts(dtype) {
        return Err(format!(
            "source dtype {dtype:?} violates constraint {:?}",
            entry.dtype_constraint
        ));
    }
    validate_payload_extent(source_shape, dtype, len)?;
    Ok(dtype)
}

/// Fulfill a manifest on one device from the canonical source payload seam.
///
/// `mesh` is the single-device mesh or a pure expert-parallel mesh (an `Ep`
/// axis only), of which this call fulfills logical rank
/// `expected.logical_rank()`. Every EP rank holds the whole manifest, except
/// that each [`ShardPolicy::ExpertSharded`] stack uploads only the rank's
/// owned experts compactly: under `Stride` assignment rank `r` of `ranks`
/// keeps experts `e % ranks == r` in ascending `e` (local index `e / ranks`),
/// so the tensor is `[ep_local_expert_count(n_experts, r, ranks), ..]` and
/// byte-identical per expert to the source; its projection is
/// `ExpertCompact` with `rank`/`ranks`. The store's origin is the rank's
/// identity in `mesh` (epoch, logical rank, physical device), so every rank
/// of one load carries the same mesh epoch.
///
/// Small tensors may be borrowed from a source mmap or owned by the callback.
/// Large tensors use a checked [`SourceRangeDescriptor`], which is uploaded
/// through bounded quant-block-aligned staging. External-row entries are
/// recorded in the transaction census and never become `WeightHandle`s.
pub fn fulfill_manifest_from_payloads<'a, F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<SourcePayload<'a>, String>,
{
    // EP ranks derive from the passed mesh: a pure multi-device `Ep` mesh
    // fulfills this call's logical rank with its compact expert share.
    let ep = (mesh.n_devices() > 1 && mesh.size_of(DimKind::Ep) == mesh.n_devices()).then(|| {
        EpFulfill {
            rank: expected.logical_rank(),
            ranks: mesh.n_devices(),
        }
    });
    fulfill_payloads_impl(
        weights, mesh, n_layers, gpu, expected, ep, None, None, source,
    )
}

/// [`fulfill_manifest_from_payloads`] with an optional [`DirectSource`].
/// With `direct == None` this is exactly the existing path. With
/// `Some(direct)` every [`SourcePayload::Range`] entry (device-resident and
/// host-mapped) is read straight from its absolute file offset in the file
/// `direct` was opened on: host-mapped owners are allocated through
/// `Gpu::alloc_raw_host_mapped`, filled by the parallel reader and tail-zeroed,
/// with the next host-mapped entry's owner allocated while the current read is
/// in flight; device owners use the pooled allocation of the buffered range
/// path. Borrowed and Owned payloads are untouched, and expert-parallel
/// `ExpertSharded` stacks of a single-device direct request still require them.
/// An EP mesh has no direct entry point: every rank of one load fulfills
/// through [`fulfill_manifest_from_payloads`] on the seal's mesh.
#[allow(clippy::too_many_arguments)]
pub fn fulfill_manifest_from_payloads_direct<'a, F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    direct: Option<&DirectSource>,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<SourcePayload<'a>, String>,
{
    fulfill_payloads_impl(
        weights, mesh, n_layers, gpu, expected, None, direct, None, source,
    )
}
/// [`fulfill_manifest_from_payloads`] for the bounded two-family arena: the
/// non-expert weights named by `arenas.weights` are uploaded directly into a
/// few exact-allocated roots and published as borrowed views carrying the
/// same shape/dtype/projection metadata a per-tensor upload would. The roots
/// live in the returned transaction's store (not its placements) and are
/// released last by [`WeightLoadTransaction::rollback`]. The F16-shadow
/// source bindings of `arenas.f16_shadows` are installed on `gpu` before the
/// transaction is returned, so no constructor or priming math can run
/// against an unbound plan. Single-device only: an expert-parallel request
/// is refused before any allocation. One `Arc` plan drives the root request,
/// the capacity proof and every view.
pub fn fulfill_manifest_from_payloads_with_arenas<'a, F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    arenas: Arc<ModelWeightArenaPlan>,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<SourcePayload<'a>, String>,
{
    fulfill_payloads_impl(
        weights,
        mesh,
        n_layers,
        gpu,
        expected,
        None,
        None,
        Some(arenas),
        source,
    )
}

/// The one payload fulfillment implementation. Direct I/O and arena entry
/// points retain their source/ownership behavior through the same transaction.
/// Mesh-derived `ep` cuts only the expert stacks to the rank's share;
/// single-device direct and arena requests pass `None`.
#[allow(clippy::too_many_arguments)]
fn fulfill_payloads_impl<'a, F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    ep: Option<EpFulfill>,
    direct: Option<&DirectSource>,
    arenas: Option<Arc<ModelWeightArenaPlan>>,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<SourcePayload<'a>, String>,
{
    // An EP rank places the manifest as one device does; only the expert
    // stacks are cut to the rank's share.
    let single = DeviceMesh::default();
    let view = if ep.is_some() { &single } else { mesh };
    if let Some(error) = target_error(view) {
        return Err(error);
    }
    if let Err(reason) = validate_fulfillment_manifest(weights, view, n_layers) {
        return Err(FulfillError {
            name: "<manifest>".to_string(),
            layer: None,
            device: 0,
            reason,
        });
    }

    let rank = ep.map_or(0, |ep| ep.rank);
    let origin = WeightOrigin::from_parts(mesh.epoch(), rank, gpu.device_id);
    if origin != expected || ep.is_some_and(|ep| ep.rank >= ep.ranks) {
        return Err(FulfillError {
            name: "<origin>".to_string(),
            layer: None,
            device: 0,
            reason: format!(
                "admitted target identity {expected:?} does not match runtime mesh/GPU {origin:?} \
                 ({} logical devices); refusing before upload",
                mesh.n_devices()
            ),
        });
    }

    let placement = BandPlacement::uniform(view, n_layers).map_err(|reason| FulfillError {
        name: "<manifest>".to_string(),
        layer: None,
        device: 0,
        reason,
    })?;
    if let Some(plan) = &arenas {
        if ep.is_some() {
            return Err(arena_error(
                "weight arenas are single-device only; expert-parallel fulfillment is refused",
            ));
        }
        if let Err(reason) = validate_weight_arena_manifest(plan, weights) {
            return Err(arena_error(reason));
        }
    }
    let mut store = WeightStore::with_origin(origin);
    if let Some(plan) = arenas.clone() {
        if let Err(reason) = acquire_weight_roots(&mut store, plan, |bytes| {
            gpu.alloc_tensor_exact(&[bytes], DType::Raw)
        }) {
            return Err(rollback_fulfill_error(store, gpu, arena_error(reason)));
        }
    }
    let mut look = HostLookahead::default();
    store.capture_logical = weight_digest_path().is_some();
    for (entry_index, entry) in weights.iter().enumerate() {
        let devices = placement_devices(entry, view, &placement);
        if devices.as_slice() != [0] {
            return Err(rollback_fulfill_error(
                store,
                gpu,
                fulfill_entry_error(
                    entry,
                    format!("Single placement resolved to {:?}, expected [0]", devices),
                ),
            ));
        }
        let key = WeightPlacementKey::new(&entry.name, entry.layer, 0);

        if let ShardPolicy::Tied {
            source: source_name,
        } = &entry.policy
        {
            if store
                .external_descriptor(source_name, entry.layer, 0)
                .is_some()
            {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(
                        entry,
                        format!(
                            "tied source '{source_name}' is external rows and cannot be aliased"
                        ),
                    ),
                ));
            }
            let source_dtype = match store.get(source_name, entry.layer, 0) {
                Some(WeightHandle::Resident(tensor)) => Some(tensor.dtype),
                Some(WeightHandle::Alias(_)) | None => None,
            };
            if let Some(actual_dtype) = source_dtype {
                if !entry.dtype_constraint.accepts(actual_dtype) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            format!(
                                "tied source '{source_name}' actual dtype {actual_dtype:?} \
                                 is excluded by constraint {:?}",
                                entry.dtype_constraint
                            ),
                        ),
                    ));
                }
                let projection = projection_for(entry, 0, 1, actual_dtype);
                if let Err(reason) =
                    store.insert(key, WeightHandle::Alias(source_name.clone()), projection)
                {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason.to_string()),
                    ));
                }
                continue;
            }
        }

        if let (Some(ep), ShardPolicy::ExpertSharded { n_experts, assign }) = (ep, &entry.policy) {
            let payload = match source(entry) {
                Ok(payload) => payload,
                Err(reason) => {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, format!("source read failed: {reason}")),
                    ));
                }
            };
            let (tensor, dtype, logical) =
                match upload_expert_compact(gpu, entry, ep, *n_experts, *assign, payload) {
                    Ok(uploaded) => uploaded,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
            store.note_logical_bytes(&key, logical);
            let projection = projection_for(entry, ep.rank, ep.ranks, dtype);
            if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(entry, reason.to_string()),
                ));
            }
            if test_support::record_resident_upload() {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(entry, "test fault injected after resident upload"),
                ));
            }
            continue;
        }
        let read_entry = fulfillment_source(entry);
        let fetched = match look.payload.take() {
            Some((index, prefetched)) if index == entry_index => prefetched,
            _ => source(&read_entry),
        };
        let payload = match fetched {
            Ok(payload) => payload,
            Err(reason) => {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(entry, format!("source read failed: {reason}")),
                ));
            }
        };
        if matches!(entry.policy, ShardPolicy::Tied { .. }) {
            let actual_name = match &payload {
                SourcePayload::Borrowed { info, .. } => Some(info.name.as_str()),
                SourcePayload::Owned { info, .. } => Some(info.name.as_str()),
                SourcePayload::Range(_) => None,
            };
            if actual_name.is_some_and(|name| name != read_entry.name) {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(
                        entry,
                        format!(
                            "tied materialization requested '{}' but received {actual_name:?}",
                            read_entry.name
                        ),
                    ),
                ));
            }
        }
        match payload {
            SourcePayload::Borrowed { info, bytes } => {
                if entry.residency.is_external() {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            "external rows require a checked source range payload",
                        ),
                    ));
                }
                let dtype = match source_dtype(&info.dtype) {
                    Ok(dtype) => dtype,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                if let Err(reason) = validate_payload_shape(entry, &info.shape) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                if !entry.dtype_constraint.accepts(dtype) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            format!(
                                "source dtype {dtype:?} violates constraint {:?}",
                                entry.dtype_constraint
                            ),
                        ),
                    ));
                }
                if let Err(reason) = validate_payload_extent(&info.shape, dtype, bytes.len()) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                let tensor = match upload_payload_bytes(gpu, &store, entry, dtype, bytes) {
                    Ok(tensor) => tensor,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                store.note_logical_bytes(&key, bytes.len());
                let projection = projection_for(entry, 0, 1, dtype);
                if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason.to_string()),
                    ));
                }
                if test_support::record_resident_upload() {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, "test fault injected after resident upload"),
                    ));
                }
            }
            SourcePayload::Owned { info, bytes } => {
                if entry.residency.is_external() {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            "external rows require a checked source range payload",
                        ),
                    ));
                }
                let dtype = match source_dtype(&info.dtype) {
                    Ok(dtype) => dtype,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                if let Err(reason) = validate_payload_shape(entry, &info.shape) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                if !entry.dtype_constraint.accepts(dtype) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            format!(
                                "source dtype {dtype:?} violates constraint {:?}",
                                entry.dtype_constraint
                            ),
                        ),
                    ));
                }
                if let Err(reason) = validate_payload_extent(&info.shape, dtype, bytes.len()) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                let tensor = match upload_payload_bytes(gpu, &store, entry, dtype, &bytes) {
                    Ok(tensor) => tensor,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                store.note_logical_bytes(&key, bytes.len());
                let projection = projection_for(entry, 0, 1, dtype);
                if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason.to_string()),
                    ));
                }
                if test_support::record_resident_upload() {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, "test fault injected after resident upload"),
                    ));
                }
            }
            SourcePayload::Range(descriptor) => {
                let dtype = match source_dtype(descriptor.dtype()) {
                    Ok(dtype) => dtype,
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                if let Err(reason) = validate_source_range_identity(
                    &read_entry,
                    &descriptor,
                    entry.residency.is_external()
                        || matches!(entry.policy, ShardPolicy::Tied { .. }),
                ) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                if let Err(reason) = validate_payload_shape(entry, descriptor.logical_shape()) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                if !entry.dtype_constraint.accepts(dtype) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(
                            entry,
                            format!(
                                "source dtype {dtype:?} violates constraint {:?}",
                                entry.dtype_constraint
                            ),
                        ),
                    ));
                }
                let range_len = match usize::try_from(descriptor.length) {
                    Ok(length) => length,
                    Err(_) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, "source range length does not fit usize"),
                        ));
                    }
                };
                if let Err(reason) =
                    validate_payload_extent(descriptor.logical_shape(), dtype, range_len)
                {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason),
                    ));
                }
                if entry.residency.is_external() {
                    if let Err(reason) = validate_external_range(entry, dtype, &descriptor) {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                    let projection = projection_for(entry, 0, 1, dtype);
                    if let Err(reason) = store.record_external(key, descriptor, projection) {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason.to_string()),
                        ));
                    }
                    continue;
                }
                let uploaded = match (entry.residency, direct) {
                    (WeightResidency::HostMapped, None) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(
                                entry,
                                "host-mapped residency needs a borrowed or owned payload, not a source range",
                            ),
                        ));
                    }
                    (WeightResidency::HostMapped, Some(direct)) => {
                        let next =
                            peek_next_host_range(weights, entry_index + 1, ep, &source, &mut look);
                        upload_range_host_direct(
                            gpu,
                            &mut store,
                            direct,
                            entry_index,
                            &descriptor,
                            &entry.logical_shape,
                            next.map(|(index, len)| {
                                (index, len, weights[index].logical_shape.as_slice())
                            }),
                            &mut look.alloc_error,
                        )
                    }
                    (_, Some(direct)) => upload_range_direct_device(
                        gpu,
                        direct,
                        &descriptor,
                        dtype,
                        &entry.logical_shape,
                        range_tail_pad(entry),
                    ),
                    (_, None) => {
                        upload_payload_range(gpu, &store, entry, &descriptor, dtype, range_len)
                    }
                };
                let tensor = match uploaded {
                    Ok(mut tensor) => {
                        tensor.dtype = dtype;
                        tensor
                    }
                    Err(reason) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, reason),
                        ));
                    }
                };
                store.note_logical_bytes(&key, range_len);
                let projection = projection_for(entry, 0, 1, dtype);
                if let Err(reason) = store.insert(key, WeightHandle::Resident(tensor), projection) {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, reason.to_string()),
                    ));
                }
                if test_support::record_resident_upload() {
                    return Err(rollback_fulfill_error(
                        store,
                        gpu,
                        fulfill_entry_error(entry, "test fault injected after resident upload"),
                    ));
                }
            }
        }
        if let ShardPolicy::Tied { source } = &entry.policy {
            store.aliases.insert(
                WeightPlacementKey::new(&entry.name, entry.layer, 0),
                source.clone(),
            );
        }
    }
    if let Some(staged) = store.staged_host.take() {
        let _ = gpu.free_tensor(staged.tensor);
    }
    if let Some(plan) = &arenas {
        // Bind the shadow sources to their real resident pointers now, after
        // every upload and before the constructors / priming math can run. A
        // failure rolls back through the same release path, which clears the
        // Gpu's bindings before any root pointer can be reused.
        let ptrs = match shadow_source_ptrs(&store, weights) {
            Ok(ptrs) => ptrs,
            Err(reason) => {
                return Err(rollback_fulfill_error(store, gpu, arena_error(reason)));
            }
        };
        if let Err(error) = gpu.install_f16_shadow_arena(Arc::clone(plan), &ptrs) {
            return Err(rollback_fulfill_error(
                store,
                gpu,
                arena_error(format!("F16 shadow binding install failed: {error}")),
            ));
        }
    }
    Ok(WeightLoadTransaction::new(store))
}

/// Canonical name used by the manifest fulfillment seam. The target is
/// deliberately Single-only in this pilot; multi-device fulfillment belongs to
/// the admitted mesh/G5 integration and must not grow a second owner here.
/// `expected` is the plan-time admitted identity; see
/// [`fulfill_manifest_single`].
pub fn fulfill_manifest<F>(
    weights: &[WeightEntry],
    mesh: &DeviceMesh,
    n_layers: usize,
    gpu: &mut Gpu,
    expected: WeightOrigin,
    source: F,
) -> Result<WeightLoadTransaction, FulfillError>
where
    F: Fn(&WeightEntry) -> Result<(Vec<u8>, DType), String>,
{
    fulfill_manifest_single(weights, mesh, n_layers, gpu, expected, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expert `e` of a stack is `per_expert` bytes `(e * 7 + i) as u8`, so a
    /// misplaced or swapped extent changes the bytes.
    fn expert_stack(n_experts: usize, per_expert: usize) -> Vec<u8> {
        (0..n_experts)
            .flat_map(|e| (0..per_expert).map(move |i| (e * 7 + i) as u8))
            .collect()
    }

    #[test]
    fn compact_owned_experts_concatenates_owned_extents_ascending() {
        let (n, b) = (16, 5);
        let stack = expert_stack(n, b);
        for ranks in [2usize, 3, 4, 5, 6] {
            let mut owned_total = 0;
            for rank in 0..ranks {
                let compact =
                    compact_owned_experts(&stack, n, rank, ranks, ExpertAssign::Stride).unwrap();
                let expected: Vec<u8> = (0..n)
                    .filter(|e| e % ranks == rank)
                    .flat_map(|e| stack[e * b..(e + 1) * b].iter().copied())
                    .collect();
                assert_eq!(compact, expected, "ranks={ranks} rank={rank}");
                assert_eq!(compact.len(), ep_local_expert_count(n, rank, ranks) * b);
                // Local index e / ranks holds expert e.
                for e in (rank..n).step_by(ranks) {
                    let local = e / ranks;
                    assert_eq!(
                        &compact[local * b..(local + 1) * b],
                        &stack[e * b..(e + 1) * b]
                    );
                }
                owned_total += compact.len();
            }
            assert_eq!(owned_total, stack.len(), "ranks partition the stack");
        }
    }

    #[test]
    fn compact_owned_experts_refuses_bad_geometry() {
        let stack = expert_stack(8, 4);
        let stride = ExpertAssign::Stride;
        assert!(compact_owned_experts(&stack, 8, 2, 2, stride).is_err());
        assert!(compact_owned_experts(&stack, 8, 0, 3, stride).is_ok());
        assert!(compact_owned_experts(&stack, 8, 0, 9, stride).is_err());
        assert!(compact_owned_experts(&stack, 8, 0, 0, stride).is_err());
        assert!(compact_owned_experts(&stack[..31], 8, 0, 2, stride).is_err());
        assert!(compact_owned_experts(&[], 8, 0, 2, stride).is_err());
        assert!(compact_owned_experts(&stack, 8, 0, 2, ExpertAssign::Contiguous).is_err());
    }
    use crate::device_mesh::DimKind;
    use crate::weight_manifest::{DTypeConstraint, PinTarget, ShardPolicy};

    use crate::model_source::{
        ModelSource, SourceError, SourceFormat, SourceIdentity, SourcePayload,
        SourceRangeDescriptor, SourceReader, SourceReaderImpl, TensorInfo,
    };
    use std::path::PathBuf;
    use std::sync::Arc;

    struct TestRangeReader {
        identity: SourceIdentity,
        bytes: Vec<u8>,
    }

    impl SourceReaderImpl for TestRangeReader {
        fn identity(&self) -> &SourceIdentity {
            &self.identity
        }

        fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            let start = usize::try_from(offset).map_err(|_| SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            let available = self.bytes.len().saturating_sub(start);
            if available < dst.len() {
                if available != 0 {
                    dst[..available].copy_from_slice(&self.bytes[start..]);
                }
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: available,
                });
            }
            dst.copy_from_slice(&self.bytes[start..start + dst.len()]);
            Ok(())
        }
    }

    fn test_descriptor_with_length(
        bytes: Vec<u8>,
        declared_len: u64,
        dtype: &str,
        shape: Vec<usize>,
    ) -> SourceRangeDescriptor {
        let identity = SourceIdentity {
            canonical_path: PathBuf::from("/test/source"),
            format: SourceFormat::Safetensors,
            files: Vec::new(),
            metadata_json: "{}".to_string(),
            manifest: Vec::new(),
        };
        let reader = SourceReader::from_inner(TestRangeReader {
            identity: identity.clone(),
            bytes,
        });
        let identity = Arc::new(identity);
        SourceRangeDescriptor::from_parts(
            identity,
            0,
            declared_len,
            dtype.to_string(),
            shape,
            reader,
        )
        .expect("test descriptor has a checked range")
    }

    fn test_descriptor(bytes: Vec<u8>, dtype: &str, shape: Vec<usize>) -> SourceRangeDescriptor {
        let declared_len = u64::try_from(bytes.len()).expect("test bytes fit u64");
        test_descriptor_with_length(bytes, declared_len, dtype, shape)
    }

    fn owned_info(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            quant_type: 0xFF,
            data_offset: 0,
            data_size: shape.iter().product::<usize>() * 2,
        }
    }

    fn projection(dtype: DType) -> WeightProjection {
        WeightProjection {
            kind: WeightProjectionKind::Static,
            axis: None,
            rank: 0,
            world_size: 1,
            logical_shape: vec![1],
            dtype,
        }
    }

    #[test]
    fn origin_mismatch_is_detected_before_gpu_release() {
        let first = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let second = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let actual = WeightOrigin::from_parts(first.epoch(), 0, 0);
        let expected = WeightOrigin::from_parts(second.epoch(), 0, 0);
        let store = WeightStore::with_origin(actual);
        let error = store.validate_origin_value(expected).unwrap_err();
        assert!(matches!(
            error,
            WeightStoreError::OriginMismatch {
                expected: got_expected,
                actual: got_actual
            } if got_expected == expected && got_actual == actual
        ));
    }

    #[test]
    fn staged_take_moves_handles_but_retains_projection_inventory() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, 0);
        let mut store = WeightStore::with_origin(origin);
        store
            .stage_alias("first", None, 0, "source", projection(DType::F16))
            .unwrap();
        store
            .stage_alias("second", Some(2), 0, "source", projection(DType::F16))
            .unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.inventory_len(), 2);
        let first = store.take_with_projection("first", None, 0).unwrap();
        assert!(matches!(first.0, WeightHandle::Alias(_)));
        assert_eq!(store.len(), 1);
        // The taken cell's projection and alias edge stay behind as the
        // retained publication inventory; only the handle leaves.
        assert!(store.projection("first", None, 0).is_some());
        assert_eq!(store.alias_source("first", None, 0), Some("source"));
        assert_eq!(store.inventory_len(), 2);
        let second = store.take("second", Some(2), 0).unwrap();
        assert!(matches!(second, WeightHandle::Alias(_)));
        assert!(store.is_empty());
        assert_eq!(store.inventory_len(), 2);
    }

    #[test]
    fn allocation_journal_records_insertion_order_for_reverse_rollback() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, 0);
        let mut store = WeightStore::with_origin(origin);
        for name in ["first", "second", "third"] {
            store
                .stage_alias(name, None, 0, "source", projection(DType::F16))
                .unwrap();
        }
        let order: Vec<String> = store.journal.iter().map(|key| key.name.clone()).collect();
        assert_eq!(order, vec!["first", "second", "third"]);
        // Taking a handle leaves its journal slot; rollback walks the journal
        // in reverse and skips keys with no placement, so each resident is
        // freed at most once in exact reverse allocation order.
        let _taken = store.take("second", None, 0).unwrap();
        let order: Vec<String> = store.journal.iter().map(|key| key.name.clone()).collect();
        assert_eq!(order, vec!["first", "second", "third"]);
        let reverse: Vec<String> = store
            .journal
            .iter()
            .rev()
            .map(|key| key.name.clone())
            .collect();
        assert_eq!(reverse, vec!["third", "second", "first"]);
    }

    #[test]
    fn assembly_drop_restores_staged_handles() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source", projection(DType::F16))
            .unwrap();
        {
            let mut assembly = store.begin_assembly();
            assert_eq!(assembly.take("x", None, 0), Some(0));
            let guard = assembly.commit();
            assert!(guard.get(0).is_some());
        }
        assert!(store.contains("x", None, 0));
        assert!(store.projection("x", None, 0).is_some());
    }

    #[test]
    fn repeated_unload_lookup_cannot_reclaim_a_transferred_cell() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source", projection(DType::F16))
            .unwrap();
        let _owned = store.take("x", None, 0).unwrap();
        assert!(store.take("x", None, 0).is_none());
        // The handle is gone exactly once, but its projection and alias edge
        // remain as publication inventory.
        assert!(store.projection("x", None, 0).is_some());
        assert_eq!(store.alias_source("x", None, 0), Some("source"));
        assert!(store.is_empty());
    }

    #[test]
    fn duplicate_projection_is_rejected_without_replacing_identity() {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let mut store = WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0));
        store
            .stage_alias("x", None, 0, "source-a", projection(DType::F16))
            .unwrap();
        let error = store
            .stage_alias("x", None, 0, "source-b", projection(DType::F32))
            .unwrap_err();
        assert!(matches!(error, WeightStoreError::DuplicatePlacement(_)));
        assert!(
            matches!(store.get("x", None, 0), Some(WeightHandle::Alias(source)) if source == "source-a")
        );
        assert_eq!(store.projection("x", None, 0).unwrap().dtype, DType::F16);
    }

    #[test]
    fn single_target_refuses_multi_device_before_source_or_gpu_work() {
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)])
            .expect("small test mesh construction cannot overflow");
        let entry = WeightEntry::model(
            "embed",
            vec![2, 2],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Embed),
        );
        // The target guard is pure and can be checked without constructing a
        // Gpu; the closure would be unreachable on this path.
        assert!(target_error(&mesh).is_some());
        let placement = BandPlacement::uniform(&mesh, 1).unwrap();
        assert_eq!(placement_devices(&entry, &mesh, &placement), vec![0]);
    }

    #[test]
    fn output_tie_validates_and_requests_the_nonlocal_source_identity() {
        let mesh = DeviceMesh::single().unwrap();
        let tied = WeightEntry::model(
            "mtp.embed_tokens",
            vec![8, 8],
            DType::F16,
            ShardPolicy::Tied {
                source: "embed_tokens".to_string(),
            },
        )
        .with_placement(crate::weight_manifest::PlacementHint::Pin(
            PinTarget::Output,
        ));
        // The general manifest still requires a full declaration. Fulfillment
        // alone accepts the explicit scoped output materialization contract.
        assert!(crate::weight_manifest::validate_manifest(&[tied.clone()], &mesh).is_err());
        validate_fulfillment_manifest(&[tied.clone()], &mesh, 48).unwrap();
        let request = fulfillment_source(&tied);
        assert_eq!(request.name, "embed_tokens");
        assert_eq!(request.logical_shape, tied.logical_shape);
        assert_eq!(request.dtype_constraint, tied.dtype_constraint);
        assert_eq!(request.policy, ShardPolicy::Replicate);
        assert!(matches!(tied.policy, ShardPolicy::Tied { .. }));
        let plain = WeightEntry::model("plain", vec![1], DType::F16, ShardPolicy::Replicate);
        assert!(matches!(fulfillment_source(&plain), Cow::Borrowed(_)));
    }

    #[test]
    fn local_tie_and_nonlocal_output_materialization_keep_the_same_provenance() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single().unwrap();
        let source = WeightEntry::model("source", vec![1], DType::F32, ShardPolicy::Replicate);
        let tied = WeightEntry::model(
            "output",
            vec![1],
            DType::F32,
            ShardPolicy::Tied {
                source: "source".to_string(),
            },
        )
        .with_placement(crate::weight_manifest::PlacementHint::Pin(
            PinTarget::Output,
        ));
        for payloads in [false, true] {
            for local in [false, true] {
                let weights = if local {
                    vec![source.clone(), tied.clone()]
                } else {
                    vec![tied.clone()]
                };
                let expected = WeightOrigin::for_single(&mesh, &gpu);
                let calls = std::cell::Cell::new(0);
                let tx = if payloads {
                    fulfill_manifest_from_payloads(
                        &weights,
                        &mesh,
                        1,
                        &mut gpu,
                        expected,
                        |entry| {
                            assert_eq!(entry.name, "source");
                            calls.set(calls.get() + 1);
                            Ok(SourcePayload::Owned {
                                info: crate::model_source::TensorInfo {
                                    name: "source".to_string(),
                                    shape: vec![1],
                                    dtype: "F32".to_string(),
                                    quant_type: 0xFF,
                                    data_offset: 0,
                                    data_size: 4,
                                },
                                bytes: vec![0; 4],
                            })
                        },
                    )
                    .unwrap()
                } else {
                    fulfill_manifest_single(&weights, &mesh, 1, &mut gpu, expected, |entry| {
                        assert_eq!(entry.name, "source");
                        calls.set(calls.get() + 1);
                        Ok((vec![0; 4], DType::F32))
                    })
                    .unwrap()
                };
                assert_eq!(calls.get(), 1);
                assert_eq!(tx.alias_source("output", None, 0), Some("source"));
                assert_eq!(tx.origin(), Some(expected));
                assert_eq!(tx.projection("output", None, 0).unwrap().dtype, DType::F32);
                assert_eq!(
                    matches!(tx.get("output", None, 0), Some(WeightHandle::Alias(_))),
                    local
                );
                assert_eq!(
                    matches!(tx.get("output", None, 0), Some(WeightHandle::Resident(_))),
                    !local
                );
                tx.rollback(&mut gpu).unwrap();
            }
        }
    }

    #[test]
    fn tied_projection_preserves_fulfilled_source_dtype() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let constraint = DTypeConstraint::source_from_sources(vec![DType::F16, DType::F32]);
        let source = WeightEntry::model_with_dtype_constraint(
            "source",
            vec![1],
            DType::F16,
            constraint.clone(),
            ShardPolicy::Replicate,
        );
        let alias = WeightEntry::model_with_dtype_constraint(
            "alias",
            vec![1],
            DType::F16,
            constraint,
            ShardPolicy::Tied {
                source: "source".into(),
            },
        );
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let transaction =
            fulfill_manifest_single(&[source, alias], &mesh, 1, &mut gpu, expected, |_| {
                Ok((vec![0; 4], DType::F32))
            })
            .unwrap();
        assert_eq!(
            transaction.projection("alias", None, 0).unwrap().dtype,
            DType::F32
        );
        assert!(matches!(
            transaction.get("alias", None, 0),
            Some(WeightHandle::Alias(source)) if source == "source"
        ));
        transaction
            .rollback(&mut gpu)
            .expect("resident transaction rollback must return its buffers to the pool");
    }

    #[test]
    fn successful_single_fulfillment_commits_resident_projection() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entry = WeightEntry::model("resident", vec![1], DType::F32, ShardPolicy::Replicate);
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let transaction = fulfill_manifest_single(&[entry], &mesh, 1, &mut gpu, expected, |_| {
            Ok((vec![0; 4], DType::F32))
        })
        .unwrap();
        assert_eq!(transaction.len(), 1);
        assert!(matches!(
            transaction.get("resident", None, 0),
            Some(WeightHandle::Resident(tensor)) if tensor.dtype == DType::F32
        ));
        assert_eq!(
            transaction.projection("resident", None, 0).unwrap().dtype,
            DType::F32
        );
        transaction
            .rollback(&mut gpu)
            .expect("resident transaction rollback must return its buffers to the pool");
    }

    #[test]
    fn full_origin_mismatch_leaves_unpublished_transaction_unchanged() {
        let first = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let second = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let actual = WeightOrigin::from_parts(first.epoch(), 3, 11);
        let expected = WeightOrigin::from_parts(second.epoch(), 4, 12);
        let mut store = WeightStore::with_origin(actual);
        store
            .stage_alias("resident", None, 0, "source", projection(DType::F16))
            .unwrap();
        let transaction = WeightLoadTransaction::new(store);
        let error = transaction.validate_origin_value(expected).unwrap_err();
        assert!(matches!(error, WeightStoreError::OriginMismatch { .. }));
        assert_eq!(transaction.origin(), Some(actual));
        assert!(transaction.contains("resident", None, 0));
        assert!(transaction.projection("resident", None, 0).is_some());
    }

    #[test]
    fn full_origin_mismatch_does_not_free_a_resident_transaction() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entry = WeightEntry::model("resident", vec![1], DType::F32, ShardPolicy::Replicate);
        let admitted = WeightOrigin::for_single(&mesh, &gpu);
        let transaction = fulfill_manifest_single(&[entry], &mesh, 1, &mut gpu, admitted, |_| {
            Ok((vec![0; 4], DType::F32))
        })
        .unwrap();
        let expected = WeightOrigin::from_parts(mesh.epoch(), 1, gpu.device_id);
        let error = transaction.validate_origin_value(expected).unwrap_err();
        assert!(matches!(error, WeightStoreError::OriginMismatch { .. }));
        assert_eq!(transaction.len(), 1);
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            0,
            "origin rejection must not free resident buffers"
        );
        transaction
            .rollback(&mut gpu)
            .expect("resident transaction rollback must return its buffers to the pool");
        assert_eq!(RESIDENT_RELEASES.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn rollback_reports_free_failure_without_counting_release() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        RESIDENT_ALLOCATIONS.with(|count| count.set(1));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let origin = WeightOrigin::from_parts(mesh.epoch(), 0, gpu.device_id);
        let mut store = WeightStore::with_origin(origin);
        let borrowed = GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: DType::F32,
        };
        store
            .insert(
                WeightPlacementKey::new("borrowed", None, 0),
                WeightHandle::Resident(borrowed),
                projection(DType::F32),
            )
            .expect("insert borrowed resident test handle");
        let error = WeightLoadTransaction::new(store)
            .rollback(&mut gpu)
            .expect_err("rollback must surface a failed pool return");
        assert!(error.message.contains("non-owning"));
        assert_eq!(test_support::resident_allocations(), 1);
        assert_eq!(test_support::resident_releases(), 0);
        test_support::reset();
    }

    #[test]
    fn source_failure_after_resident_upload_rolls_back_everything() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::F32, ShardPolicy::Replicate),
        ];
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let error = fulfill_manifest_single(&entries, &mesh, 1, &mut gpu, expected, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Err("injected source failure".into())
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("source read failed"));
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            1,
            "the first resident allocation must be explicitly freed"
        );
    }

    #[test]
    fn dtype_failure_after_resident_upload_rolls_back_everything() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let constraint = DTypeConstraint::source_exact(DType::F32);
        let entries = vec![
            WeightEntry::model_with_dtype_constraint(
                "first",
                vec![1],
                DType::F32,
                constraint.clone(),
                ShardPolicy::Replicate,
            ),
            WeightEntry::model_with_dtype_constraint(
                "second",
                vec![1],
                DType::F32,
                constraint,
                ShardPolicy::Replicate,
            ),
        ];
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let error = fulfill_manifest_single(&entries, &mesh, 1, &mut gpu, expected, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Ok((vec![0; 2], DType::F16))
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("violates constraint"));
        assert_eq!(
            RESIDENT_RELEASES.with(std::cell::Cell::get),
            1,
            "the first resident allocation must be explicitly freed"
        );
    }

    #[test]
    fn malformed_upload_payload_after_resident_allocation_rolls_back() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        RESIDENT_RELEASES.with(|count| count.set(0));
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::F32, ShardPolicy::Replicate),
        ];
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let error = fulfill_manifest_single(&entries, &mesh, 1, &mut gpu, expected, |entry| {
            if entry.name == "first" {
                Ok((vec![0; 4], DType::F32))
            } else {
                Ok((vec![0; 1], DType::F32))
            }
        })
        .unwrap_err();
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("payload"));
        assert_eq!(RESIDENT_RELEASES.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn admitted_origin_mismatch_fails_before_any_upload() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let other = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        // A stale admitted identity (different mesh epoch) must fail before
        // the first source read or upload: nothing is allocated, so there is
        // nothing to roll back.
        let stale = WeightOrigin::from_parts(other.epoch(), 0, gpu.device_id);
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::F32, ShardPolicy::Replicate),
        ];
        let error = fulfill_manifest_single(&entries, &mesh, 1, &mut gpu, stale, |_| {
            panic!("source must not run after an admitted-identity mismatch")
        })
        .unwrap_err();
        assert_eq!(error.name, "<origin>");
        assert!(error.reason.contains("refusing before upload"));
        assert_eq!(test_support::resident_allocations(), 0);
        assert_eq!(test_support::resident_releases(), 0);
        test_support::reset();
    }
    #[test]
    fn expert_parallel_rank_refuses_a_foreign_epoch_or_rank_before_any_upload() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let other = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let entries = vec![WeightEntry::model(
            "w",
            vec![1],
            DType::F32,
            ShardPolicy::Replicate,
        )];
        // Another mesh epoch, then a rank outside the mesh: both refuse before
        // the first source read.
        for expected in [
            WeightOrigin::from_parts(other.epoch(), 1, gpu.device_id),
            WeightOrigin::from_parts(mesh.epoch(), 2, gpu.device_id),
        ] {
            let error =
                fulfill_manifest_from_payloads(&entries, &mesh, 1, &mut gpu, expected, |_| {
                    panic!("source must not run after an admitted-identity mismatch")
                })
                .unwrap_err();
            assert_eq!(error.name, "<origin>", "{expected:?}");
        }
        assert_eq!(test_support::resident_allocations(), 0);
        test_support::reset();
    }
    #[test]
    fn range_chunks_are_bounded_and_quant_block_aligned() {
        for dtype in [DType::BF16, DType::F32, DType::MQ4G256V2, DType::MQ4G128V2] {
            let alignment = quant_block_bytes(dtype);
            let total = RANGE_UPLOAD_CHUNK_BYTES.div_ceil(alignment) * alignment + alignment * 3;
            let mut offset = 0usize;
            let mut chunks = 0usize;
            while offset < total {
                let chunk = next_range_chunk(offset, total, alignment);
                assert!(chunk > 0);
                assert!(chunk <= RANGE_UPLOAD_CHUNK_BYTES);
                assert_eq!(chunk % alignment, 0);
                offset += chunk;
                chunks += 1;
            }
            assert_eq!(offset, total);
            assert!(chunks >= 2);
        }
    }

    #[test]
    fn external_rows_are_census_only_and_do_not_allocate() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entry = WeightEntry::model("ple", vec![2, 1], DType::BF16, ShardPolicy::Replicate)
            .external_rows(2, 1);
        let descriptor = test_descriptor(vec![0; 4], "BF16", vec![2, 1]);
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let transaction =
            fulfill_manifest_from_payloads(&[entry], &mesh, 1, &mut gpu, expected, |_| {
                Ok(SourcePayload::Range(descriptor.clone()))
            })
            .expect("external range should be recorded without GPU work");
        assert_eq!(transaction.len(), 0);
        assert!(transaction.is_empty());
        assert_eq!(transaction.external_rows_len(), 1);
        assert!(transaction.get("ple", None, 0).is_none());
        assert!(transaction.external_descriptor("ple", None, 0).is_some());
        assert_eq!(transaction.inventory_len(), 1);
        assert_eq!(test_support::resident_allocations(), 0);
        assert_eq!(test_support::resident_releases(), 0);
        transaction
            .rollback(&mut gpu)
            .expect("external-only rollback has no resident buffers");
        assert_eq!(test_support::resident_releases(), 0);
        test_support::reset();
    }
    #[test]
    fn compact_qwen4_ple_ranges_bind_external_rows_without_allocating() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        let fixture_dir = tempfile::tempdir().expect("fixture directory");
        let fixture_path = fixture_dir.path().join("qwen4-ple.hfq");
        crate::hfq::hfq_test_fixture::write_compact_qwen4_ple_hfq(&fixture_path)
            .expect("write compact Qwen4 fixture");
        let hfq = crate::hfq::HfqFile::open(&fixture_path).expect("open compact Qwen4 fixture");
        let entries = crate::hfq::hfq_test_fixture::COMPACT_PLE_NAMES
            .into_iter()
            .map(|name| {
                WeightEntry::layer(
                    name,
                    1,
                    vec![
                        crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_COUNT,
                        crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_WIDTH,
                    ],
                    DType::BF16,
                    ShardPolicy::Replicate,
                )
                .external_rows(
                    crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_BYTES,
                    crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_COUNT,
                )
            })
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::single().expect("single-device mesh construction");
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        test_support::reset();
        let transaction =
            fulfill_manifest_from_payloads(&entries, &mesh, 2, &mut gpu, expected, |entry| {
                let source: &dyn ModelSource = &hfq;
                source
                    .tensor_payload(&entry.name)
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| format!("missing fixture tensor '{}'", entry.name))
            })
            .expect("HFQM ranges should bind as external rows");

        assert_eq!(transaction.len(), 0);
        assert_eq!(transaction.external_rows_len(), 2);
        assert_eq!(transaction.inventory_len(), 2);
        assert_eq!(test_support::resident_allocations(), 0);
        let shard_10 = transaction
            .external_descriptor(
                crate::hfq::hfq_test_fixture::COMPACT_PLE_NAMES[0],
                Some(1),
                0,
            )
            .expect("shard_10 external descriptor");
        let shard_2 = transaction
            .external_descriptor(
                crate::hfq::hfq_test_fixture::COMPACT_PLE_NAMES[1],
                Some(1),
                0,
            )
            .expect("shard_2 external descriptor");
        assert_eq!(shard_10.source_identity(), shard_2.source_identity());
        assert_eq!(shard_10.source_identity().format, SourceFormat::Hfq);
        assert_eq!(shard_10.source_identity().manifest.len(), 5);

        let mut last_row = vec![0u8; crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_BYTES];
        shard_2
            .read_exact_at(
                shard_2.offset + crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_BYTES as u64,
                &mut last_row,
            )
            .expect("last row of shard_2");
        assert!(last_row
            .chunks_exact(2)
            .all(|chunk| chunk == 0x21u16.to_le_bytes()));
        let mut first_row = vec![0u8; crate::hfq::hfq_test_fixture::COMPACT_PLE_ROW_BYTES];
        shard_10
            .read_exact_at(shard_10.offset, &mut first_row)
            .expect("first row of shard_10");
        assert!(first_row
            .chunks_exact(2)
            .all(|chunk| chunk == 0x10u16.to_le_bytes()));

        transaction
            .rollback(&mut gpu)
            .expect("external-only transaction rollback");
        assert_eq!(test_support::resident_releases(), 0);
        test_support::reset();
    }

    #[test]
    fn short_range_read_rolls_back_prior_resident_and_partial_tensor() {
        let Ok(mut gpu) = Gpu::init() else {
            return;
        };
        test_support::reset();
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        let entries = vec![
            WeightEntry::model("first", vec![1], DType::BF16, ShardPolicy::Replicate),
            WeightEntry::model("second", vec![1], DType::BF16, ShardPolicy::Replicate),
        ];
        let short = test_descriptor_with_length(vec![0], 2, "BF16", vec![1]);
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let error =
            fulfill_manifest_from_payloads(&entries, &mesh, 1, &mut gpu, expected, |entry| {
                if entry.name == "first" {
                    Ok(SourcePayload::Owned {
                        info: owned_info("first", "BF16", &[1]),
                        bytes: vec![0; 2],
                    })
                } else {
                    Ok(SourcePayload::Range(short.clone()))
                }
            })
            .expect_err("short range read must fail the whole transaction");
        assert_eq!(error.name, "second");
        assert!(error.reason.contains("source range read failed"));
        assert_eq!(
            test_support::resident_releases(),
            1,
            "the earlier resident must be released after the failed range"
        );
        test_support::reset();
    }

    #[test]
    fn external_rows_admit_each_declared_ple_tier() {
        // Qwen4 PLE rows are 160 values wide and ship in two tiers. One
        // declaration must validate against either, because a sealed artifact
        // carries whichever tier it was written with; the stride comes from the
        // shard's own dtype.
        assert_eq!(external_row_stride(DType::BF16, 160), Some(320));
        assert_eq!(external_row_stride(DType::Q8_0, 160), Some(170));
        // A partial Q8 block is refused rather than rounded.
        assert_eq!(external_row_stride(DType::Q8_0, 129), None);
        assert_eq!(external_row_stride(DType::MQ6G256V2, 160), None);

        let mut entry = WeightEntry::layer(
            "model.ple.ngram_embedding.shard_0.weight",
            1,
            vec![4, 160],
            DType::Q8_0,
            ShardPolicy::Replicate,
        )
        .external_rows(170, 4);
        entry.dtype_constraint =
            DTypeConstraint::source_from_sources(vec![DType::Q8_0, DType::BF16]);

        for (dtype, name, stride) in [(DType::BF16, "BF16", 320u64), (DType::Q8_0, "Q8_0", 170u64)]
        {
            let descriptor = test_descriptor_with_length(vec![0], 4 * stride, name, vec![4, 160]);
            validate_external_range(&entry, dtype, &descriptor)
                .unwrap_or_else(|error| panic!("{dtype:?} shard must validate: {error}"));
            let short = test_descriptor_with_length(vec![0], 4 * stride - 1, name, vec![4, 160]);
            assert!(
                validate_external_range(&entry, dtype, &short).is_err(),
                "an extent that does not match {dtype:?} rows must be refused"
            );
        }

        // A tier the declaration excludes is refused even when its own extent
        // is self-consistent.
        let foreign = test_descriptor_with_length(vec![0], 4 * 200, "MQ6G256V2", vec![4, 160]);
        let error = validate_external_range(&entry, DType::MQ6G256V2, &foreign)
            .expect_err("an excluded tier must be refused");
        assert!(error.contains("excluded by constraint"), "{error}");
    }

    #[test]
    fn mq4g128v2_extent_is_row_aware_and_exact() {
        assert_eq!(
            expected_payload_bytes(&[2, 129], DType::MQ4G128V2).unwrap(),
            Some(2 * 2 * 68)
        );
        assert_eq!(
            expected_payload_bytes(&[3, 5, 640], DType::MQ4G128V2).unwrap(),
            Some(15 * 5 * 68)
        );
        assert!(validate_payload_extent(&[3, 5, 640], DType::MQ4G128V2, 15 * 5 * 68,).is_ok());
        assert!(validate_payload_extent(&[2, 129], DType::MQ4G128V2, 2 * 68).is_err());
        assert!(expected_payload_bytes(&[2, 512], DType::MQ4G256V2).is_ok());
        assert!(expected_payload_bytes(&[2, 129], DType::MQ4G256V2).is_err());
    }

    #[test]
    fn hfq4g128_extent_refuses_cross_row_packing() {
        // SmolLM2 q_proj: K=960 is 7.5 groups; each row pads to 8 groups.
        assert_eq!(
            expected_payload_bytes(&[960, 960], DType::HFQ4G128).unwrap(),
            Some(960 * 8 * 72)
        );
        // A pre-`b4846285e` file packs groups across rows: 7.5 groups per row.
        assert!(validate_payload_extent(&[960, 960], DType::HFQ4G128, 960 * 540).is_err());
        assert!(expected_payload_bytes(&[960, 2560], DType::MQ4G256).is_ok());
        assert!(expected_payload_bytes(&[2560, 960], DType::MQ4G256).is_err());
        // The GDN conv1d in `qwen3.6:35b-a3b-mq4r` is a flat-packed MQ4G256
        // `[8192, 1, 4]`; it must keep loading.
        assert_eq!(
            expected_payload_bytes(&[8192, 1, 4], DType::MQ4G256).unwrap(),
            None
        );
        // A flat-packed conv1d `[C, 1, 4]` has no row-padded extent.
        assert_eq!(
            expected_payload_bytes(&[64, 1, 4], DType::HFQ4G128).unwrap(),
            None
        );
    }

    #[test]
    fn floor_stride_extent_refuses_ragged_k() {
        // K=1000 is not a whole number of 64-, 128- or 256-element groups.
        for dtype in [
            DType::HFQ6G256,
            DType::HFQ4G256,
            DType::HFQ2G128,
            DType::HFQ3G128,
            DType::Q4F16G64,
        ] {
            assert!(
                expected_payload_bytes(&[2560, 1000], dtype).is_err(),
                "{dtype:?}"
            );
            // A flat-packed conv1d `[C, 1, 4]` has no row extent.
            assert_eq!(expected_payload_bytes(&[64, 1, 4], dtype).unwrap(), None);
        }
        // The flat SmolLM2 HFQ6 gate_proj [2560, 960]: 9600 groups of 200 B.
        assert!(validate_payload_extent(&[2560, 960], DType::HFQ6G256, 1_920_000).is_err());
        assert_eq!(
            expected_payload_bytes(&[960, 2560], DType::HFQ6G256).unwrap(),
            Some(960 * 10 * 200)
        );
        assert_eq!(
            expected_payload_bytes(&[960, 2560], DType::HFQ2G128).unwrap(),
            Some(960 * 20 * 40)
        );
        assert_eq!(
            expected_payload_bytes(&[960, 2560], DType::HFQ3G128).unwrap(),
            Some(960 * 20 * 56)
        );
        assert_eq!(
            expected_payload_bytes(&[960, 2560], DType::Q4F16G64).unwrap(),
            Some(960 * 40 * 36)
        );
    }

    #[test]
    fn source_dtype_mapping_is_explicit_and_rejects_unknown_formats() {
        assert_eq!(source_dtype("BF16").unwrap(), DType::BF16);
        assert_eq!(source_dtype("mq4g128v2").unwrap(), DType::MQ4G128V2);
        assert!(source_dtype("I64").is_err());
    }

    #[test]
    fn sha256_helpers_match_known_vectors_and_stream_split() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut stream = Sha256Stream::new();
        stream.update(b"a");
        stream.update(b"");
        stream.update(b"bc");
        assert_eq!(stream.finish_hex(), sha256_hex(b"abc"));
    }

    #[test]
    fn digest_rows_append_as_whole_jsonl_lines_and_truncate_clears() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("digest.jsonl");
        append_weight_digest_line(
            &path,
            &WeightDigestRow {
                name: "wq",
                layer: Some(3),
                residency: "device",
                dtype: "Q4F16G64",
                logical_bytes: 4096,
                sha256: &sha256_hex(b"abc"),
            },
        )
        .unwrap();
        append_weight_digest_line(
            &path,
            &WeightDigestRow {
                name: "qwen4.expert_tiers.host",
                layer: None,
                residency: "host_mapped",
                dtype: "expert_payload",
                logical_bytes: 7,
                sha256: &sha256_hex(b""),
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with('\n'));
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["name"], "wq");
        assert_eq!(rows[0]["layer"], 3);
        assert_eq!(rows[0]["residency"], "device");
        assert_eq!(rows[0]["dtype"], "Q4F16G64");
        assert_eq!(rows[0]["logical_bytes"], 4096);
        assert_eq!(rows[0]["sha256"], sha256_hex(b"abc"));
        assert!(rows[1]["layer"].is_null());
        assert_eq!(rows[1]["residency"], "host_mapped");
        assert!(text.lines().next().unwrap().starts_with("{\"name\":\"wq\""));
        truncate_weight_digest(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    }

    #[test]
    fn logical_bytes_are_captured_only_when_digest_enabled() {
        let key = WeightPlacementKey::new("wq", Some(0), 0);
        let mut store = WeightStore::default();
        store.note_logical_bytes(&key, 64);
        assert!(
            store.logical_bytes.is_empty(),
            "off by default: no map growth"
        );
        store.capture_logical = true;
        store.note_logical_bytes(&key, 64);
        assert_eq!(store.logical_bytes.get(&key), Some(&64));
    }

    fn descriptor_sealed_to(
        files: Vec<crate::model_source::SourceFileIdentity>,
    ) -> SourceRangeDescriptor {
        let identity = SourceIdentity {
            canonical_path: PathBuf::from("/test/source"),
            format: SourceFormat::Safetensors,
            files,
            metadata_json: "{}".to_string(),
            manifest: Vec::new(),
        };
        let reader = SourceReader::from_inner(TestRangeReader {
            identity: identity.clone(),
            bytes: vec![0u8; 8],
        });
        SourceRangeDescriptor::from_parts(
            Arc::new(identity),
            0,
            8,
            "Q4K".to_string(),
            vec![8],
            reader,
        )
        .expect("test descriptor has a checked range")
    }

    #[test]
    fn direct_identity_accepts_only_the_readers_own_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.hfq");
        let other_path = dir.path().join("b.hfq");
        std::fs::write(&path, vec![1u8; 4096]).unwrap();
        std::fs::write(&other_path, vec![2u8; 8192]).unwrap();
        let direct = DirectSource::open_path(&path).unwrap();
        let other = DirectSource::open_path(&other_path).unwrap();

        let matching = descriptor_sealed_to(vec![direct.file_identity().clone()]);
        assert!(check_direct_identity(&direct, &matching).is_ok());
        assert!(check_direct_identity(&other, &matching).is_err());

        let none = descriptor_sealed_to(Vec::new());
        assert!(check_direct_identity(&direct, &none).is_err());

        let overlay = descriptor_sealed_to(vec![
            direct.file_identity().clone(),
            other.file_identity().clone(),
        ]);
        assert!(check_direct_identity(&direct, &overlay).is_err());
    }

    /// Offset-base guard for the direct weight path: a sealed HFQ descriptor
    /// carries the ABSOLUTE file offset, and the direct host reader reads it
    /// unchanged, on a real HFQ whose data section starts at a nonzero offset.
    #[test]
    fn direct_host_read_uses_absolute_descriptor_offsets_on_nonzero_data_offset_hfq() {
        use crate::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abs.hfq");
        let tensors: Vec<HfqMemTensor> = (0..3u32)
            .map(|t| HfqMemTensor {
                name: format!("w{t}"),
                quant_type: 2,
                shape: vec![1024 + 4 * t],
                group_size: 0,
                data: (0..(1024 + 4 * t) * 4)
                    .map(|j| (t * 7919 + j).wrapping_mul(2_654_435_761).rotate_right(13) as u8)
                    .collect(),
            })
            .collect();
        write_hfqm_package_mem(&path, 0, "{}", &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        assert!(hfq.data_offset() > 0);
        let direct = DirectSource::open(&hfq).unwrap();
        for tensor in &tensors {
            let descriptor = hfq.tensor_range(&tensor.name).unwrap().unwrap();
            check_direct_identity(&direct, &descriptor).unwrap();
            assert!(
                descriptor.offset >= hfq.data_offset(),
                "descriptor offset is absolute"
            );
            let mut got = vec![0u8; descriptor.length as usize];
            read_host_range(
                &direct,
                descriptor.offset,
                got.len(),
                got.as_mut_ptr() as usize,
            )
            .unwrap();
            assert_eq!(got, tensor.data, "{}", tensor.name);
        }
    }
}

/// CPU tests of the bounded weight arena's ownership: exact roots, borrowed
/// views, and the one release path. Device buffers are fabricated descriptors
/// that are never dereferenced; [`FakeRelease`] records the exact operation
/// order that [`GpuRelease`] performs against a real device.
#[cfg(test)]
mod weight_store_arena {
    use super::*;
    use crate::tp_shard::ExpertAssign;
    use rdna_compute::byte_arena::NamedByteArenaPlan;

    const GRANULE: usize = 512;

    fn fake_buf(ptr: usize, size: usize) -> hip_bridge::DeviceBuffer {
        // SAFETY: never dereferenced; tests only read the descriptor.
        unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, size) }
    }

    /// A fabricated owner-tagged (non-borrowed) buffer, standing in for an
    /// individually allocated resident that must reach the owner free path.
    fn fake_owned(ptr: usize, size: usize) -> GpuTensor {
        GpuTensor {
            // SAFETY: never dereferenced or freed; the fake owner only logs.
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_vmm_owner(ptr as *mut std::ffi::c_void, size)
            },
            shape: vec![size],
            dtype: DType::Raw,
        }
    }

    fn root_tensor(ptr: usize, size: usize) -> GpuTensor {
        GpuTensor {
            buf: fake_buf(ptr, size),
            shape: vec![size],
            dtype: DType::Raw,
        }
    }

    /// Exact (or deliberately oversized / failing) root allocator.
    struct Bump {
        next: usize,
        oversize: usize,
        fail_at: Option<usize>,
        count: usize,
        bases: Vec<usize>,
    }

    impl Bump {
        fn new() -> Self {
            Self {
                next: 0x1000_0000,
                oversize: 0,
                fail_at: None,
                count: 0,
                bases: Vec::new(),
            }
        }

        fn alloc(&mut self, bytes: usize) -> hip_bridge::HipResult<GpuTensor> {
            let index = self.count;
            self.count += 1;
            if self.fail_at == Some(index) {
                return Err(hip_bridge::HipError::new(2, "fake out of memory"));
            }
            let base = self.next;
            self.next += bytes + self.oversize + 0x10000;
            self.bases.push(base);
            Ok(root_tensor(base, bytes + self.oversize))
        }
    }

    #[derive(Default)]
    struct FakeRelease {
        log: Vec<String>,
        fail_quiesce: bool,
        /// Root release call numbers (0-based) that report an error.
        fail_root_calls: Vec<usize>,
        root_calls: usize,
    }

    impl WeightRelease for FakeRelease {
        fn free_resident(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
            self.log
                .push(format!("free_resident:{:#x}", tensor.buf.as_ptr() as usize));
            Ok(())
        }
        fn clear_arena_caches(&mut self) {
            self.log.push("clear".into());
        }
        fn quiesce(&mut self) -> hip_bridge::HipResult<()> {
            self.log.push("quiesce".into());
            if self.fail_quiesce {
                return Err(hip_bridge::HipError::new(9, "fake quiesce failure"));
            }
            Ok(())
        }
        fn release_root(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
            let call = self.root_calls;
            self.root_calls += 1;
            self.log.push(format!(
                "root:{:#x}:{}",
                tensor.buf.as_ptr() as usize,
                tensor.buf.size()
            ));
            if self.fail_root_calls.contains(&call) {
                return Err(hip_bridge::HipError::new(3, "fake root release failure"));
            }
            Ok(())
        }
    }

    /// Weights a(1000) b(1500) c(0) d(600), alignment 256, cap 2048:
    /// root0 = [a]; root1 = [b, c@1536]; root2 = [d]. Shadows: a, b.
    fn plan() -> Arc<ModelWeightArenaPlan> {
        let weights = NamedByteArenaPlan::new(
            vec![
                ("d".to_string(), 600),
                ("a".to_string(), 1000),
                ("c".to_string(), 0),
                ("b".to_string(), 1500),
            ],
            256,
            2048,
        )
        .unwrap();
        let f16_shadows = NamedByteArenaPlan::new(
            vec![("b".to_string(), 1500), ("a".to_string(), 1000)],
            256,
            2048,
        )
        .unwrap();
        let plan = ModelWeightArenaPlan {
            weights,
            f16_shadows,
            granule_bytes: GRANULE,
        };
        plan.validate().unwrap();
        Arc::new(plan)
    }

    fn manifest() -> Vec<WeightEntry> {
        let entry = |name: &str, shape: Vec<usize>, dtype| {
            WeightEntry::model(name, shape, dtype, ShardPolicy::Replicate)
        };
        vec![
            entry("a", vec![25, 20], DType::BF16),
            entry("b", vec![25, 30], DType::BF16),
            entry("c", vec![0], DType::F16),
            entry("d", vec![300], DType::F16),
            // Not planned: individually allocated.
            entry("expert", vec![8], DType::F32),
        ]
    }

    fn projection(dtype: DType) -> WeightProjection {
        WeightProjection {
            kind: WeightProjectionKind::Static,
            axis: None,
            rank: 0,
            world_size: 1,
            logical_shape: vec![1],
            dtype,
        }
    }

    fn empty_store() -> WeightStore {
        let mesh = DeviceMesh::single().expect("single-device mesh construction cannot overflow");
        WeightStore::with_origin(WeightOrigin::from_parts(mesh.epoch(), 0, 0))
    }

    /// A store holding the planned roots and a borrowed view of every planned
    /// member, exactly as a successful arena fulfillment leaves it.
    fn arena_store(bump: &mut Bump) -> WeightStore {
        let mut store = empty_store();
        acquire_weight_roots(&mut store, plan(), |bytes| bump.alloc(bytes)).unwrap();
        let entries = manifest();
        for entry in entries.iter().take(4) {
            let arena = store.arena.as_ref().unwrap();
            let span = arena.span(&entry.name).unwrap().clone();
            let view = arena
                .view(&span, &entry.logical_shape, entry.dtype)
                .unwrap();
            store
                .insert(
                    WeightPlacementKey::new(&entry.name, entry.layer, 0),
                    WeightHandle::Resident(view),
                    projection(entry.dtype),
                )
                .unwrap();
        }
        store
    }

    #[test]
    fn root_request_capacity_per_root_equality() {
        test_support::reset();
        let plan = plan();
        assert_eq!(plan.weights.roots, vec![1000, 1536, 600]);
        let mut bump = Bump::new();
        let mut store = empty_store();
        acquire_weight_roots(&mut store, Arc::clone(&plan), |bytes| bump.alloc(bytes)).unwrap();
        assert_eq!(store.arena_root_requests(), plan.weights.roots);
        assert_eq!(store.arena_root_capacities(), plan.weights.roots);
        assert_eq!(test_support::arena_root_allocations(), 3);
        assert!(store.has_live_weight_roots());
        assert!(Arc::ptr_eq(store.weight_arena_plan().unwrap(), &plan));
        plan.weights
            .validate_capacities(&store.arena_root_capacities(), GRANULE)
            .unwrap();

        // An oversized (pool-reused style) buffer is rejected, not trusted,
        // and every root that was allocated is released through the one path.
        test_support::reset();
        let mut oversized = Bump::new();
        oversized.oversize = 256;
        let mut store = empty_store();
        let error = acquire_weight_roots(&mut store, Arc::clone(&plan), |bytes| {
            oversized.alloc(bytes)
        })
        .unwrap_err();
        assert!(error.contains("capacity validation"), "{error}");
        assert_eq!(store.arena_root_capacities(), vec![1256, 1792, 856]);
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        assert_eq!(test_support::arena_root_releases(), 3);
    }

    #[test]
    fn partial_root_allocation_failure_releases_only_the_roots_made() {
        test_support::reset();
        let mut bump = Bump::new();
        bump.fail_at = Some(1);
        let mut store = empty_store();
        let error =
            acquire_weight_roots(&mut store, plan(), |bytes| bump.alloc(bytes)).unwrap_err();
        assert!(error.contains("root 1"), "{error}");
        assert!(error.contains("exact allocation failed"), "{error}");
        assert_eq!(test_support::arena_root_allocations(), 1);
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        assert_eq!(
            ops.log,
            vec![
                "clear".to_string(),
                "quiesce".to_string(),
                format!("root:{:#x}:1000", bump.bases[0]),
            ]
        );
        assert_eq!(test_support::arena_root_releases(), 1);
    }

    #[test]
    fn views_carry_metadata_order_zero_size_and_tail_bounds() {
        let mut bump = Bump::new();
        let store = arena_store(&mut bump);
        let arena = store.arena.as_ref().unwrap();
        let [r0, r1, r2] = [bump.bases[0], bump.bases[1], bump.bases[2]];
        // Sorted-name order, 256-byte member alignment, next-fit roots.
        let spans: Vec<_> = arena
            .plan
            .weights
            .members
            .iter()
            .map(|span| (span.name.as_str(), span.root, span.offset, span.bytes))
            .collect();
        assert_eq!(
            spans,
            vec![
                ("a", 0, 0, 1000),
                ("b", 1, 0, 1500),
                ("c", 1, 1536, 0),
                ("d", 2, 0, 600)
            ]
        );
        let expected = [
            ("a", r0, 1000),
            ("b", r1, 1500),
            ("c", r1 + 1536, 0),
            ("d", r2, 600),
        ];
        for (name, ptr, bytes) in expected {
            let Some(WeightHandle::Resident(view)) = store.get(name, None, 0) else {
                panic!("{name} has no resident view")
            };
            assert!(view.buf.is_borrowed(), "{name} must be a borrowed view");
            assert_eq!(view.buf.as_ptr() as usize, ptr, "{name} pointer");
            assert_eq!(view.buf.size(), bytes, "{name} bytes");
            assert!(arena.contains_view(view), "{name} lies inside a live root");
        }
        // Shape, dtype and projection survive exactly as a per-tensor upload.
        let Some(WeightHandle::Resident(a)) = store.get("a", None, 0) else {
            panic!()
        };
        assert_eq!((a.shape.clone(), a.dtype), (vec![25, 20], DType::BF16));
        assert_eq!(store.projection("a", None, 0).unwrap().dtype, DType::BF16);
        // The zero-sized member is a real view at its aligned offset, never a
        // fabricated pointer, and the tail of every root stays unviewed.
        let Some(WeightHandle::Resident(c)) = store.get("c", None, 0) else {
            panic!()
        };
        assert_eq!(c.buf.size(), 0);

        // A span past its root, or naming a root that does not exist, is an
        // error and never a view.
        let a_span = arena.span("a").unwrap().clone();
        let past_end = NamedByteArenaSpan {
            bytes: 1001,
            ..a_span.clone()
        };
        assert!(arena
            .view(&past_end, &[1], DType::Raw)
            .err()
            .expect("a span past its root is an error")
            .contains("exceeds root"));
        let bad_root = NamedByteArenaSpan { root: 9, ..a_span };
        assert!(arena
            .view(&bad_root, &[1], DType::Raw)
            .err()
            .expect("a span naming a missing root is an error")
            .contains("names root"));
    }

    #[test]
    fn manifest_validation_refuses_ineligible_or_unknown_members_before_allocation() {
        let plan = plan();
        validate_weight_arena_manifest(&plan, &manifest()).unwrap();

        let mut missing = manifest();
        missing.retain(|entry| entry.name != "d");
        assert!(validate_weight_arena_manifest(&plan, &missing)
            .unwrap_err()
            .contains("no manifest entry"));

        let mut duplicated = manifest();
        duplicated.push(WeightEntry::model(
            "a",
            vec![25, 20],
            DType::BF16,
            ShardPolicy::Replicate,
        ));
        assert!(validate_weight_arena_manifest(&plan, &duplicated)
            .unwrap_err()
            .contains("must be unique"));

        let mut tied = manifest();
        tied[3].policy = ShardPolicy::Tied { source: "a".into() };
        assert!(validate_weight_arena_manifest(&plan, &tied)
            .unwrap_err()
            .contains("tied alias"));

        let mut host_mapped = manifest();
        host_mapped[3].residency = WeightResidency::HostMapped;
        assert!(validate_weight_arena_manifest(&plan, &host_mapped)
            .unwrap_err()
            .contains("device-resident"));

        let mut expert = manifest();
        expert[3].policy = ShardPolicy::ExpertSharded {
            n_experts: 4,
            assign: ExpertAssign::Stride,
        };
        assert!(validate_weight_arena_manifest(&plan, &expert)
            .unwrap_err()
            .contains("expert-sharded"));

        // A shadow owner must be the 2-D BF16 weight its shadow was sized for.
        let mut wrong_dtype = manifest();
        wrong_dtype[0].dtype = DType::F16;
        assert!(validate_weight_arena_manifest(&plan, &wrong_dtype)
            .unwrap_err()
            .contains("2-D BF16"));
    }

    #[test]
    fn partial_upload_rollback_clears_caches_first_and_releases_roots_last() {
        test_support::reset();
        let mut bump = Bump::new();
        let mut store = arena_store(&mut bump);
        // An individually allocated resident that reached the store before the
        // failure (the unplanned `expert`).
        store
            .insert(
                WeightPlacementKey::new("expert", None, 0),
                WeightHandle::Resident(fake_owned(0x7000_0000, 32)),
                projection(DType::F32),
            )
            .unwrap();
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        let [r0, r1, r2] = [bump.bases[0], bump.bases[1], bump.bases[2]];
        assert_eq!(
            ops.log,
            vec![
                "clear".to_string(),
                // Only the owned resident reaches the free path; the four
                // borrowed views own nothing.
                "free_resident:0x70000000".to_string(),
                "quiesce".to_string(),
                // Roots last, once each, reverse allocation order.
                format!("root:{r2:#x}:600"),
                format!("root:{r1:#x}:1536"),
                format!("root:{r0:#x}:1000"),
            ]
        );
        assert_eq!(test_support::arena_root_releases(), 3);
        assert_eq!(test_support::resident_releases(), 1);
    }

    #[test]
    fn constructor_guard_rollback_restores_views_while_roots_stay_live() {
        test_support::reset();
        let mut bump = Bump::new();
        let mut store = arena_store(&mut bump);
        {
            let mut assembly = store.begin_assembly();
            for name in ["a", "b", "c", "d"] {
                assert!(assembly.take(name, None, 0).is_some());
            }
            let guard = assembly.commit();
            assert!(guard.get(0).is_some());
            // Constructor failure: the guard drops without `finalize`.
        }
        assert_eq!(store.len(), 4, "every taken view returned to the store");
        assert!(store.has_live_weight_roots());
        assert_eq!(test_support::arena_root_releases(), 0);
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        assert_eq!(
            ops.log
                .iter()
                .filter(|line| line.starts_with("root:"))
                .count(),
            3
        );
        assert!(
            ops.log
                .iter()
                .all(|line| !line.starts_with("free_resident")),
            "restored views are still borrowed and are never freed: {:?}",
            ops.log
        );
    }

    #[test]
    fn taken_view_lifetime_roots_survive_finalize_and_outlive_the_views() {
        test_support::reset();
        let mut bump = Bump::new();
        let mut store = arena_store(&mut bump);
        let taken = {
            let mut assembly = store.begin_assembly();
            for name in ["a", "b", "c", "d"] {
                assembly.take(name, None, 0).unwrap();
            }
            assembly.commit().finalize()
        };
        assert_eq!(taken.len(), 4);
        // Empty placements with live roots are NOT an empty store.
        assert_eq!(store.len(), 0);
        assert!(!store.is_empty());
        assert!(store.has_live_weight_roots());
        assert_eq!(store.arena_root_capacities(), vec![1000, 1536, 600]);
        assert_eq!(test_support::arena_root_releases(), 0);
        // The typed weights are torn down first (borrowed views skipped)...
        let mut freed = 0;
        for weight in taken {
            if let WeightHandle::Resident(tensor) = weight.handle {
                if !tensor.buf.is_borrowed() {
                    freed += 1;
                }
            }
        }
        assert_eq!(freed, 0);
        assert_eq!(test_support::arena_root_releases(), 0);
        // ...and only then does the store release its roots.
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        assert_eq!(test_support::arena_root_releases(), 3);
        assert!(WeightLoadTransaction::new(empty_store()).is_empty());
    }

    #[test]
    fn rollback_twice_releases_each_root_exactly_once() {
        test_support::reset();
        let mut bump = Bump::new();
        let mut transaction = WeightLoadTransaction::new(arena_store(&mut bump));
        assert!(!transaction.is_empty());
        assert_eq!(
            transaction.weight_arena_root_capacities(),
            vec![1000, 1536, 600]
        );
        assert_eq!(
            transaction.weight_arena_root_requests(),
            vec![1000, 1536, 600]
        );
        let mut ops = FakeRelease::default();
        transaction.rollback_via(&mut ops).unwrap();
        let first = ops.log.clone();
        transaction.rollback_via(&mut ops).unwrap();
        assert_eq!(
            ops.log, first,
            "the second rollback performs no device work"
        );
        assert_eq!(test_support::arena_root_releases(), 3);
        assert!(transaction.weight_arena_plan().is_none());
        assert!(transaction.weight_arena_root_capacities().is_empty());
    }

    #[test]
    fn repeated_loads_clear_bindings_before_each_root_pointer_reuse() {
        test_support::reset();
        let mut ops = FakeRelease::default();
        let mut base_seen = Vec::new();
        for _ in 0..3 {
            // Every load's exact roots reuse the same device addresses.
            let mut bump = Bump::new();
            let store = arena_store(&mut bump);
            base_seen.push(bump.bases[0]);
            let start = ops.log.len();
            store.release_with(&mut ops).unwrap();
            let round = &ops.log[start..];
            let clear = round.iter().position(|line| line == "clear").unwrap();
            let quiesce = round.iter().position(|line| line == "quiesce").unwrap();
            let first_root = round
                .iter()
                .position(|line| line.starts_with("root:"))
                .unwrap();
            assert!(clear < quiesce && quiesce < first_root, "{round:?}");
        }
        assert_eq!(
            base_seen[0], base_seen[2],
            "the address was genuinely reused"
        );
        assert_eq!(
            ops.log.iter().filter(|line| *line == "clear").count(),
            3,
            "one cache clear per teardown"
        );
        assert_eq!(test_support::arena_root_releases(), 9);
    }

    #[test]
    fn failed_quiesce_quarantines_roots_and_never_releases_them() {
        test_support::reset();
        let mut bump = Bump::new();
        let store = arena_store(&mut bump);
        let mut ops = FakeRelease {
            fail_quiesce: true,
            ..FakeRelease::default()
        };
        let error = store.release_with(&mut ops).unwrap_err();
        assert!(error.message.contains("quarantined"), "{}", error.message);
        assert!(error.message.contains("fake quiesce failure"));
        assert!(
            ops.log.iter().all(|line| !line.starts_with("root:")),
            "an unquiesced root must never be released: {:?}",
            ops.log
        );
        assert_eq!(test_support::arena_root_releases(), 0);
        assert_eq!(test_support::arena_root_quarantines(), 3);
    }

    #[test]
    fn a_failed_root_release_is_reported_after_the_remaining_roots_are_released() {
        test_support::reset();
        let mut bump = Bump::new();
        let store = arena_store(&mut bump);
        let mut ops = FakeRelease {
            fail_root_calls: vec![0],
            ..FakeRelease::default()
        };
        let error = store.release_with(&mut ops).unwrap_err();
        assert!(error.message.contains("fake root release failure"));
        assert_eq!(
            ops.log
                .iter()
                .filter(|line| line.starts_with("root:"))
                .count(),
            3,
            "every root is attempted exactly once"
        );
        assert_eq!(
            test_support::arena_root_releases(),
            2,
            "only successes are counted"
        );
    }

    #[test]
    fn shadow_source_pointers_follow_plan_order_and_must_be_planned_views() {
        let mut bump = Bump::new();
        let store = arena_store(&mut bump);
        let entries = manifest();
        let ptrs = shadow_source_ptrs(&store, &entries).unwrap();
        // Plan shadow members are sorted: a then b.
        assert_eq!(ptrs, vec![bump.bases[0], bump.bases[1]]);

        // A shadow owner that is no longer BF16 is refused.
        let mut retagged = arena_store(&mut Bump::new());
        let key = WeightPlacementKey::new("a", None, 0);
        let Some(WeightHandle::Resident(a)) = retagged.placements.get_mut(&key) else {
            panic!()
        };
        a.dtype = DType::F16;
        assert!(shadow_source_ptrs(&retagged, &entries)
            .unwrap_err()
            .contains("not its planned BF16 arena view"));

        // A resident that is not the planned view (an owned copy) is refused.
        let mut copied = arena_store(&mut Bump::new());
        let Some(WeightHandle::Resident(b)) = copied
            .placements
            .get_mut(&WeightPlacementKey::new("b", None, 0))
        else {
            panic!()
        };
        *b = GpuTensor {
            buf: fake_buf(0x7777_0000, 1500),
            shape: vec![25, 30],
            dtype: DType::BF16,
        };
        assert!(shadow_source_ptrs(&copied, &entries)
            .unwrap_err()
            .contains("not its planned BF16 arena view"));

        // A missing resident is refused too.
        let mut absent = arena_store(&mut Bump::new());
        absent.take("a", None, 0).unwrap();
        assert!(shadow_source_ptrs(&absent, &entries)
            .unwrap_err()
            .contains("no resident handle"));
    }

    #[test]
    fn non_arena_store_behaves_exactly_as_before() {
        test_support::reset();
        let mut store = empty_store();
        assert!(store.is_empty());
        assert!(!store.has_live_weight_roots());
        assert!(store.weight_arena_plan().is_none());
        assert!(store.arena_root_capacities().is_empty());
        assert!(store.arena_root_requests().is_empty());
        assert_eq!(
            shadow_source_ptrs(&store, &manifest()).unwrap(),
            Vec::<usize>::new()
        );
        store
            .stage_alias("alias", None, 0, "source", projection(DType::F16))
            .unwrap();
        // A borrowed handle with no arena is NOT silently skipped: it still
        // reaches the owner free path and its refusal.
        store
            .insert(
                WeightPlacementKey::new("borrowed", None, 0),
                WeightHandle::Resident(root_tensor(0x5000_0000, 64)),
                projection(DType::F32),
            )
            .unwrap();
        store
            .insert(
                WeightPlacementKey::new("owned", None, 0),
                WeightHandle::Resident(fake_owned(0x6000_0000, 64)),
                projection(DType::F32),
            )
            .unwrap();
        assert!(!store.is_empty());
        let mut ops = FakeRelease::default();
        store.release_with(&mut ops).unwrap();
        assert_eq!(
            ops.log,
            vec![
                "free_resident:0x60000000".to_string(),
                "free_resident:0x50000000".to_string(),
            ],
            "reverse allocation order, no cache clear, quiesce or root release"
        );
        assert_eq!(test_support::arena_root_releases(), 0);
        assert_eq!(test_support::resident_releases(), 2);
    }
}
