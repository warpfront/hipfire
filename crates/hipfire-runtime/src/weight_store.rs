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
use crate::device_mesh::{DeviceMesh, MeshEpoch};
use crate::model_source::{SourceFormat, SourcePayload, SourceRangeDescriptor};
use crate::weight_manifest::{placement_devices, ShardPolicy, WeightEntry, WeightResidency};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

thread_local! {
    static RESIDENT_ALLOCATIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static RESIDENT_RELEASES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFTER_UPLOAD: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only allocation accounting and deterministic post-upload fault seam.
///
/// The production loader calls the same release path regardless of whether
/// this seam is armed. Callers should use [`reset`] before a scenario and
/// [`clear_faults`] after it so a failed test cannot poison a later one.
#[doc(hidden)]
pub mod test_support {
    use super::{FAIL_AFTER_UPLOAD, RESIDENT_ALLOCATIONS, RESIDENT_RELEASES};

    pub fn reset() {
        RESIDENT_ALLOCATIONS.with(|count| count.set(0));
        RESIDENT_RELEASES.with(|count| count.set(0));
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
    /// Tied-source edges (`lm_head -> token_embd`) recorded at insert.
    /// Retained after assembly takes the alias handle.
    aliases: HashMap<WeightPlacementKey, String>,
    origin: Option<WeightOrigin>,
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

    /// Tied-source edge recorded for an alias identity, if it was staged as
    /// one. Survives assembly like the projection census.
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
        if let Some(store) = self.store.take() {
            store.rollback(gpu)
        } else {
            Ok(())
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
            origin: Some(origin),
        }
    }

    pub fn origin(&self) -> Option<WeightOrigin> {
        self.origin
    }

    pub fn len(&self) -> usize {
        self.placements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
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

    /// Tied-source edge for an alias identity. Retained after assembly.
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
    fn release_unchecked(mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        let mut first_error = None;
        for key in self.journal.drain(..).rev() {
            let handle = match self.placements.remove(&key) {
                Some(handle) => handle,
                None => continue,
            };
            if let WeightHandle::Resident(tensor) = handle {
                // Pool return, not a raw HIP free: residents were uploaded
                // through the pool (`upload_pooled_bytes`), and the
                // steady-state teardown (`WeightTensor::free_all` →
                // `Gpu::free_tensor`) releases to the same pool. A raw
                // `hip.free` here would orphan pool bookkeeping; a pool
                // return keeps the next fulfillment reusing these buffers.
                match gpu.free_tensor(tensor) {
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

fn source_dtype(dtype: &str) -> Result<DType, String> {
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
        _ => None,
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
    dtype
        .row_bytes(k)
        .and_then(|row| rows.checked_mul(row))
        .ok_or_else(|| format!("{dtype:?} payload byte length overflows usize"))
        .map(Some)
}

fn validate_payload_extent(shape: &[usize], dtype: DType, actual: usize) -> Result<(), String> {
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

fn upload_range_pooled(
    gpu: &mut Gpu,
    descriptor: &SourceRangeDescriptor,
    dtype: DType,
    logical_shape: &[usize],
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
    let mut tensor = gpu
        .alloc_tensor(&[payload_len], DType::Raw)
        .map_err(|error| format!("pooled allocation failed: {error}"))?;
    let result = (|| {
        let mut offset = 0usize;
        let mut staging = vec![0u8; next_range_chunk(0, payload_len, alignment)];
        while offset < payload_len {
            let chunk_len = next_range_chunk(offset, payload_len, alignment);
            let chunk = &mut staging[..chunk_len];
            let absolute_offset = descriptor
                .offset
                .checked_add(offset as u64)
                .ok_or_else(|| "source range offset overflow".to_string())?;
            descriptor
                .read_exact_at(absolute_offset, chunk)
                .map_err(|error| format!("source range read failed: {error}"))?;
            gpu.hip
                .memcpy_htod_offset(&tensor.buf, offset, chunk)
                .map_err(|error| format!("pooled chunk upload failed: {error}"))?;
            offset += chunk_len;
        }
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        let _ = gpu.free_tensor(tensor);
        return Err(error);
    }
    tensor.shape = logical_shape.to_vec();
    Ok(tensor)
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
    if let Err(reason) = crate::weight_manifest::validate_weight_layers(weights, n_layers)
        .and_then(|_| crate::weight_manifest::validate_manifest(weights, mesh))
    {
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
    let mut store = WeightStore::with_origin(origin);
    for entry in weights {
        let devices = placement_devices(entry, mesh, n_layers);
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
            let Some(actual_dtype) = source_dtype else {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!(
                        "tied source '{source_name}' is unresolved or has no actual resident dtype"
                    ),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            };
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

        let (bytes, dtype) = match source(entry) {
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
        if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            let expected_bytes = entry
                .logical_shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                .and_then(|elements| elements.checked_mul(dtype.size()));
            if expected_bytes != Some(bytes.len()) {
                let error = FulfillError {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                    reason: format!(
                        "source payload has {} bytes, expected {:?} for {dtype:?} {:?}",
                        bytes.len(),
                        expected_bytes,
                        entry.logical_shape
                    ),
                };
                return Err(rollback_fulfill_error(store, gpu, error));
            }
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
            return Err(format!(
                "HFQ source identity has no indexed range for external entry '{}'",
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
    let expected_length = expected_row_bytes
        .checked_mul(physical_rows)
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

/// Fulfill a Single manifest from the canonical source payload seam.
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
    if let Some(error) = target_error(mesh) {
        return Err(error);
    }
    if let Err(reason) = crate::weight_manifest::validate_weight_layers(weights, n_layers)
        .and_then(|_| crate::weight_manifest::validate_manifest(weights, mesh))
    {
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

    let mut store = WeightStore::with_origin(origin);
    for entry in weights {
        let devices = placement_devices(entry, mesh, n_layers);
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
            let Some(actual_dtype) = source_dtype else {
                return Err(rollback_fulfill_error(
                    store,
                    gpu,
                    fulfill_entry_error(
                        entry,
                        format!(
                            "tied source '{source_name}' is unresolved or has no actual resident dtype"
                        ),
                    ),
                ));
            };
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
                let tensor = match upload_pooled_bytes(gpu, bytes, &entry.logical_shape) {
                    Ok(mut tensor) => {
                        tensor.dtype = dtype;
                        tensor
                    }
                    Err(error) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, format!("pooled upload failed: {error}")),
                        ));
                    }
                };
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
                let tensor = match upload_pooled_bytes(gpu, &bytes, &entry.logical_shape) {
                    Ok(mut tensor) => {
                        tensor.dtype = dtype;
                        tensor
                    }
                    Err(error) => {
                        return Err(rollback_fulfill_error(
                            store,
                            gpu,
                            fulfill_entry_error(entry, format!("pooled upload failed: {error}")),
                        ));
                    }
                };
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
                    entry,
                    &descriptor,
                    entry.residency.is_external(),
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
                let tensor =
                    match upload_range_pooled(gpu, &descriptor, dtype, &entry.logical_shape) {
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
        assert_eq!(placement_devices(&entry, &mesh, 1), vec![0]);
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
    fn source_dtype_mapping_is_explicit_and_rejects_unknown_formats() {
        assert_eq!(source_dtype("BF16").unwrap(), DType::BF16);
        assert_eq!(source_dtype("mq4g128v2").unwrap(), DType::MQ4G128V2);
        assert!(source_dtype("I64").is_err());
    }
}
