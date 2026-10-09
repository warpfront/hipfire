// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Engine-owned cache of prefill snapshots shared by every session on one
//! loaded model.
//!
//! The cache owns keys, planning, eviction, the memory guard, storage
//! placement and every copy. An architecture only describes its live state
//! through [`SessionState`]: where a cold prefill materializes canonical state
//! ([`SessionState::snapshot_boundaries`], a function of the prompt tokens so
//! an architecture can also split at message ends) and which device byte
//! ranges plus host metadata make up that state
//! ([`SessionState::snapshot_parts`]). Drivers call [`SessionCache::begin`]
//! before prefill, split prefill at [`SessionCache::next_boundary`], report
//! each boundary through [`SessionCache::at_boundary`], and
//! [`SessionCache::commit`] once the turn's output reached the client. A turn
//! that continues the live state in place starts with
//! [`SessionCache::begin_live`], which arms boundaries only when asked to
//! capture.
//!
//! Snapshots are deltas: one stores the fixed (overwritten-in-place) state
//! whole, but of each append-only [`RowStream`] only the rows above its
//! parent, the deepest snapshot of the same prefix that was present when it
//! was captured. Restoring walks the chain from the root. Snapshots never
//! depend on what the live state held before; a snapshot that still has
//! children is pinned, so eviction only ever removes leaves.
//!
//! An optional disk tier ([`SessionCache::with_disk`]) keeps snapshots the
//! device tier evicts (and every published snapshot at [`SessionCache::clear`])
//! under its own byte budget, across restarts of the same build; a restore
//! promotes them back. The parent of a device or pending snapshot is always
//! on the device (I1); the parent of a disk snapshot is on the device or on
//! disk (I2), so a chain's disk links are a suffix.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use hip_bridge::DeviceBuffer;
use rdna_compute::tensor_ops::{copy_regions, CopyRegion};
use rdna_compute::Gpu;

use crate::checkpoint_pool::{prefix_fingerprint, CheckpointBlob, CheckpointPool};
use crate::serve_contract::{CacheDomain, CheckpointId};

mod disk;

/// Byte alignment of each part inside a stored snapshot.
const PART_ALIGN: usize = 256;
/// Unified memory: a snapshot is system RAM, and overshoot reaches the global
/// OOM killer, so leave the desktop this much `MemAvailable`.
const UMA_HEADROOM: u64 = 8 << 30;
/// Discrete GPUs: free VRAM left after a snapshot and the state's growth.
const VRAM_HEADROOM: u64 = 256 << 20;

/// Which decode route the snapshot serves. Routes own different state (MTP
/// adds the draft head), so their snapshots never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionRoute {
    Ar,
    Mtp,
}

/// One contiguous device byte range of live state that is overwritten in
/// place; every snapshot copies it whole.
pub struct StatePart<'a> {
    pub buf: &'a DeviceBuffer,
    pub offset: usize,
    pub bytes: usize,
}

/// Append-only rows of live state, from byte 0 of `buf`: `rows` valid rows of
/// `row_bytes` each. A row below a snapshot boundary is never rewritten by a
/// later prefill of the same prefix, so a snapshot shares its parent's rows.
pub struct RowStream<'a> {
    pub buf: &'a DeviceBuffer,
    pub row_bytes: usize,
    pub rows: usize,
}

/// Device ranges that together are the live state, in a fixed order.
pub struct StateLayout<'a> {
    pub fixed: Vec<StatePart<'a>>,
    pub rows: Vec<RowStream<'a>>,
}

/// What an architecture captures at its current position.
pub struct SnapshotParts<'a> {
    pub meta: Vec<u8>,
    pub layout: StateLayout<'a>,
}

/// Implemented by an architecture's model state. It only describes its state;
/// the cache owns keys, planning, placement, eviction and every copy.
pub trait SessionState {
    /// Everything besides the token prefix that decides whether a snapshot is
    /// reusable for `route` (route, prefill chunk, state formats). `None` = not
    /// cacheable now.
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String>;
    /// Ascending, deduplicated positions p of `prompt` with
    /// `after < p <= up_to` (`up_to <= prompt.len()`) where a prefill of this
    /// prompt materializes state: chunk multiples, and whatever token-
    /// dependent split points (message ends) the architecture adds.
    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize>;
    /// Layout + host metadata of the live state, which must be exactly at
    /// `position`.
    fn snapshot_parts(
        &mut self,
        route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String>;
    /// Make the live state ready to receive the snapshot described by `meta`
    /// (map capacity, bump epochs) and return its destination layout, in
    /// capture order with the snapshot's row counts.
    fn restore_parts(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, String>;
    /// Apply host metadata after the device bytes were copied.
    fn finish_restore(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String>;
    /// Bytes the live state may still map or allocate before reaching its
    /// admitted context.
    fn growth_reserve_bytes(&self) -> u64;
    /// Cold-start the live state.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String>;
}

/// Where a snapshot's bytes live: device memory (VRAM, or system RAM on
/// UMA), or one file of the disk tier, `file_bytes` long (header plus
/// payload, what the disk budget counts).
enum SnapshotLocation {
    Device(DeviceBuffer),
    Disk { path: PathBuf, file_bytes: u64 },
}

/// `(scoped domain, boundary, prefix fingerprint)`, the pool's key.
type Key = (CacheDomain, u64, u64);

/// Rows `[from, to)` of one row stream held by a snapshot.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Segment {
    row_bytes: usize,
    from: usize,
    to: usize,
}

impl Segment {
    fn bytes(&self) -> usize {
        (self.to - self.from) * self.row_bytes
    }
}

/// Fixed parts whole, then each stream's segment, packed at [`PART_ALIGN`].
struct StoredSnapshot {
    location: SnapshotLocation,
    /// Holds every stream's rows below `segments[i].from`.
    parent: Option<Key>,
    fixed_bytes: Vec<usize>,
    segments: Vec<Segment>,
    meta: Vec<u8>,
    bytes: u64,
}

impl StoredSnapshot {
    /// Byte offsets of the fixed parts, then of the segments.
    fn offsets(&self) -> Vec<usize> {
        layout(
            self.fixed_bytes
                .iter()
                .copied()
                .chain(self.segments.iter().map(Segment::bytes)),
        )
        .0
    }
}

impl CheckpointBlob for StoredSnapshot {
    fn bytes_len(&self) -> u64 {
        match self.location {
            SnapshotLocation::Device(_) => self.bytes,
            SnapshotLocation::Disk { file_bytes, .. } => file_bytes,
        }
    }
}

struct Turn {
    domain: CacheDomain,
    route: SessionRoute,
    boundaries: VecDeque<usize>,
}

pub struct SessionCache {
    pool: CheckpointPool<StoredSnapshot>,
    domain: CacheDomain,
    /// Captured this turn in ascending boundary order, published by
    /// [`Self::commit`].
    pending: Vec<(Key, StoredSnapshot)>,
    /// Displaced by `commit` (which has no GPU); freed at the next `begin`/`clear`.
    release: Vec<StoredSnapshot>,
    /// Snapshots (published or pending) that name each key as parent. A key
    /// listed here is pinned in the pool.
    children: HashMap<Key, usize>,
    turn: Option<Turn>,
    disk: Option<disk::DiskTier>,
}

/// Offsets of `part_bytes` packed at [`PART_ALIGN`], and the total size.
fn layout(part_bytes: impl IntoIterator<Item = usize>) -> (Vec<usize>, usize) {
    let mut offsets = Vec::new();
    let mut end = 0usize;
    for bytes in part_bytes {
        let offset = end.next_multiple_of(PART_ALIGN);
        offsets.push(offset);
        end = offset + bytes;
    }
    (offsets, end)
}

/// Free a device snapshot's buffer; a disk snapshot is never passed here.
fn free_buffer(gpu: &mut Gpu, snapshot: StoredSnapshot) {
    if let SnapshotLocation::Device(buf) = snapshot.location {
        if let Err(error) = gpu.hip.free(buf) {
            eprintln!("  session cache: freeing a snapshot failed: {error}");
        }
    }
}

/// Whether `need` more bytes leave the guard's headroom; `None` when the
/// free-memory query itself fails.
fn memory_fits(gpu: &mut Gpu, need: u64) -> Option<bool> {
    if gpu.is_uma() {
        let available = rdna_compute::kv_slots::mem_available_bytes()?;
        Some(available >= need + UMA_HEADROOM)
    } else {
        let (free, _) = gpu.hip.get_vram_info().ok()?;
        Some(free as u64 >= need + VRAM_HEADROOM)
    }
}

impl SessionCache {
    pub fn new(domain: CacheDomain, budget_bytes: u64) -> Self {
        Self {
            pool: CheckpointPool::new(budget_bytes),
            domain,
            pending: Vec::new(),
            release: Vec::new(),
            children: HashMap::new(),
            turn: None,
            disk: None,
        }
    }

    /// Attach a disk tier in `dir` with `budget_bytes` (0 = none), bound to
    /// this executable's build.
    pub fn with_disk(self, dir: &Path, budget_bytes: u64) -> Self {
        if budget_bytes == 0 {
            return self;
        }
        match disk::build_digest() {
            Ok(build) => self.with_disk_build(dir, budget_bytes, build),
            Err(reason) => {
                eprintln!("  session cache disk tier off: {reason}");
                self
            }
        }
    }

    fn with_disk_build(mut self, dir: &Path, budget: u64, build: [u8; 32]) -> Self {
        match disk::DiskTier::open(dir, budget, build) {
            Ok(tier) => {
                eprintln!(
                    "  session cache disk tier: {} ({} MiB budget, {} snapshots / {} MiB indexed, {} other-build files)",
                    dir.display(),
                    budget >> 20,
                    tier.pool.len(),
                    tier.pool.total_bytes() >> 20,
                    tier.foreign_files()
                );
                self.disk = Some(tier);
            }
            Err(reason) => eprintln!("  session cache disk tier off: {reason}"),
        }
        self
    }

    /// Published on the device or on disk.
    fn present(&self, key: &Key) -> bool {
        self.pool.contains(&key.0, key.1, key.2)
            || self.disk.as_ref().is_some_and(|disk| disk.contains(key))
    }

    /// Longest cached prefix of `prompt` (always shorter than the prompt, so
    /// prefill computes at least the last token); 0 on a miss.
    pub fn plan(&self, state: &dyn SessionState, prompt: &[u32], route: SessionRoute) -> usize {
        let Some(scope) = state.snapshot_scope(route) else {
            return 0;
        };
        if prompt.len() < 2 {
            return 0;
        }
        let domain = self.domain.scoped(&scope);
        state
            .snapshot_boundaries(prompt, 0, prompt.len() - 1)
            .into_iter()
            .rev()
            .find(|&p| self.present(&(domain.clone(), p as u64, prefix_fingerprint(&prompt[..p]))))
            .unwrap_or(0)
    }

    /// A published or pending snapshot.
    fn entry(&self, key: &Key) -> Option<&StoredSnapshot> {
        self.pool.peek(&key.0, key.1, key.2).or_else(|| {
            self.pending
                .iter()
                .find(|(pending, _)| pending == key)
                .map(|(_, snapshot)| snapshot)
        })
    }

    /// Record a child of `parent`, pinning it against eviction.
    fn link(&mut self, parent: &Key) {
        *self.children.entry(parent.clone()).or_default() += 1;
        self.pool.pin(&parent.0, parent.1, parent.2);
    }

    /// Drop a child of `parent`; its last child unpins it.
    fn unlink(&mut self, parent: &Key) {
        if let Some(count) = self.children.get_mut(parent) {
            *count -= 1;
            if *count == 0 {
                self.children.remove(parent);
                self.pool.unpin(&parent.0, parent.1, parent.2);
            }
        }
    }

    /// Unlink a snapshot that leaves the cache from its parent and free it.
    fn drop_snapshot(&mut self, gpu: &mut Gpu, snapshot: StoredSnapshot) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        free_buffer(gpu, snapshot);
    }

    /// A published device snapshot leaves the device tier: demote it to disk
    /// when there is a disk tier (or drop its disk descendants when that
    /// fails), then free it.
    fn evict_device(&mut self, gpu: &mut Gpu, key: Key, snapshot: StoredSnapshot) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        if let Some(disk) = &mut self.disk {
            if let Err(reason) = disk.demote(gpu, &key, &snapshot) {
                eprintln!(
                    "  session cache: dropped {}-token snapshot instead of demoting it ({reason})",
                    key.1
                );
                disk.drop_descendants(&key);
            }
        }
        free_buffer(gpu, snapshot);
    }

    /// Evict device snapshots until `bytes` more fit the budget (counting
    /// pending snapshots) and the memory guard (counting `growth`).
    fn make_room(&mut self, gpu: &mut Gpu, bytes: u64, growth: u64) -> Result<(), &'static str> {
        let pending_bytes: u64 = self.pending.iter().map(|entry| entry.1.bytes).sum();
        loop {
            let over_budget =
                self.pool.total_bytes() + pending_bytes + bytes > self.pool.max_bytes();
            if !over_budget {
                match memory_fits(gpu, bytes + growth) {
                    Some(true) => return Ok(()),
                    Some(false) => {}
                    // Evicting cannot help a query that fails.
                    None => return Err("memory query failed"),
                }
            }
            match self.pool.pop_lru() {
                Some((key, evicted)) => self.evict_device(gpu, key, evicted),
                None => {
                    return Err(if over_budget {
                        "budget"
                    } else {
                        "memory guard"
                    })
                }
            }
        }
    }

    /// Unlink a snapshot that leaves the cache; free it at the next `begin`.
    fn retire(&mut self, snapshot: StoredSnapshot) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        self.release.push(snapshot);
    }

    /// Start a prefill of `prompt`: restore the `reused`-token snapshot
    /// [`Self::plan`] returned (or cold-start the state for 0) and arm the
    /// boundaries this prefill crosses.
    pub fn begin(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
    ) -> Result<(), String> {
        for snapshot in std::mem::take(&mut self.release) {
            free_buffer(gpu, snapshot);
        }
        for (_, snapshot) in std::mem::take(&mut self.pending) {
            self.drop_snapshot(gpu, snapshot);
        }
        self.turn = None;
        let Some(scope) = state.snapshot_scope(route) else {
            if reused > 0 {
                return Err(format!(
                    "session cache: route {route:?} has no snapshot scope"
                ));
            }
            return state.reset(gpu);
        };
        let domain = self.domain.scoped(&scope);
        if reused == 0 {
            state.reset(gpu)?;
        } else if let Err(error) = self.restore(gpu, state, route, &domain, &prompt[..reused]) {
            let _ = state.reset(gpu);
            return Err(error);
        }
        self.turn = Some(Turn {
            boundaries: state
                .snapshot_boundaries(prompt, reused, prompt.len())
                .into(),
            domain,
            route,
        });
        Ok(())
    }

    /// Start a prefill that continues the live state in place: the caller
    /// keeps every owner as it is (no reset, no restore) and prefills only the
    /// suffix. Does the bookkeeping head of [`Self::begin`] (frees displaced
    /// buffers, drops uncommitted snapshots).
    ///
    /// With `capture == false` it arms no boundary, so
    /// [`Self::next_boundary`] is `None` and nothing is captured this turn: a
    /// live-continued turn's state descends from decode, never from a cold
    /// canonical prefill, so it must not enter the cross-session cache.
    ///
    /// With `capture == true` it arms `snapshot_boundaries(prompt, reused,
    /// prompt.len())` under the route's scoped domain exactly like
    /// [`Self::begin`] (still no reset, no restore), and the captures publish
    /// at [`Self::commit`]. The caller passes `true` only for boundaries the
    /// architecture defines as session-lineage reusable (message ends), whose
    /// scope never aliases cold-exact snapshots.
    pub fn begin_live(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
        capture: bool,
    ) {
        for snapshot in std::mem::take(&mut self.release) {
            free_buffer(gpu, snapshot);
        }
        for (_, snapshot) in std::mem::take(&mut self.pending) {
            self.drop_snapshot(gpu, snapshot);
        }
        self.turn = None;
        if !capture {
            return;
        }
        let Some(scope) = state.snapshot_scope(route) else {
            return;
        };
        self.turn = Some(Turn {
            boundaries: state
                .snapshot_boundaries(prompt, reused, prompt.len())
                .into(),
            domain: self.domain.scoped(&scope),
            route,
        });
    }

    /// Copy the snapshot of `prefix` and its ancestors into the live state,
    /// promoting disk links to the device (or staging them in temporary
    /// buffers when the device tier has no room).
    fn restore(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        route: SessionRoute,
        domain: &CacheDomain,
        prefix: &[u32],
    ) -> Result<(), String> {
        let missing = format!(
            "session cache: planned {}-token snapshot is no longer present",
            prefix.len()
        );
        // Leaf first; each `get` refreshes that link's LRU stamp.
        let mut keys = vec![(
            domain.clone(),
            prefix.len() as u64,
            prefix_fingerprint(prefix),
        )];
        loop {
            let key = keys.last().expect("non-empty chain");
            let parent = match self.pool.get(&key.0, key.1, key.2) {
                Some(snapshot) => snapshot.parent.clone(),
                None => self
                    .disk
                    .as_mut()
                    .and_then(|disk| disk.get(key))
                    .ok_or_else(|| missing.clone())?
                    .parent
                    .clone(),
            };
            match parent {
                Some(parent) => keys.push(parent),
                None => break,
            }
        }
        keys.reverse();
        let growth = state.growth_reserve_bytes();
        if let Some(disk) = &mut self.disk {
            for key in &keys {
                if disk.contains(key) {
                    disk.held.insert(key.clone());
                    disk.pool.pin(&key.0, key.1, key.2);
                }
            }
        }
        let mut temps = Vec::new();
        let result = self.restore_chain(gpu, state, route, &keys, growth, &missing, &mut temps);
        if !temps.is_empty() {
            let _ = gpu.hip.device_synchronize();
            for (_, buf) in temps {
                if let Err(error) = gpu.hip.free(buf) {
                    eprintln!("  session cache: freeing a staging buffer failed: {error}");
                }
            }
        }
        if let Some(disk) = &mut self.disk {
            for key in std::mem::take(&mut disk.held) {
                disk.repin(&key);
            }
        }
        result
    }

    /// Bring the disk links of the root-first chain `keys` to the device
    /// (promoted, or staged into `temps`), then copy the chain into the
    /// live state.
    #[allow(clippy::too_many_arguments)]
    fn restore_chain(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        route: SessionRoute,
        keys: &[Key],
        growth: u64,
        missing: &str,
        temps: &mut Vec<(Key, DeviceBuffer)>,
    ) -> Result<(), String> {
        let first_disk = keys
            .iter()
            .position(|key| !self.pool.contains(&key.0, key.1, key.2))
            .unwrap_or(keys.len());
        let mut promoting = true;
        for (i, key) in keys.iter().enumerate().skip(first_disk) {
            let disk = self
                .disk
                .as_ref()
                .expect("a link off the device is on disk");
            let snapshot = disk.peek(key).ok_or_else(|| missing.to_string())?;
            let (bytes, parent) = (snapshot.bytes, snapshot.parent.clone());
            if promoting {
                if self.promote(gpu, key, bytes, parent.as_ref(), growth)? {
                    continue;
                }
                promoting = false;
            }
            let disk = self
                .disk
                .as_mut()
                .expect("a link off the device is on disk");
            if temps.is_empty() {
                let need: u64 = keys[i..]
                    .iter()
                    .filter_map(|key| disk.peek(key).map(|s| s.bytes))
                    .sum();
                if memory_fits(gpu, need + growth) != Some(true) {
                    return Err("session cache: no memory to stage disk snapshot".into());
                }
            }
            let buf = gpu
                .bind_thread()
                .and_then(|()| gpu.hip.malloc(bytes as usize))
                .map_err(|e| format!("session cache: staging a disk snapshot failed: {e}"))?;
            temps.push((key.clone(), buf));
            let buf = &temps.last().expect("just pushed").1;
            if let Err(error) = disk.load(gpu, key, buf) {
                disk.drop_entry(key);
                return Err(format!(
                    "session cache: reading snapshot from disk failed: {error}"
                ));
            }
        }

        let chain: Vec<(&StoredSnapshot, &DeviceBuffer)> = keys
            .iter()
            .map(|key| {
                if let Some((_, buf)) = temps.iter().find(|(temp, _)| temp == key) {
                    let snapshot = self.disk.as_ref().and_then(|disk| disk.peek(key));
                    return snapshot
                        .map(|s| (s, buf))
                        .ok_or_else(|| missing.to_string());
                }
                match self.pool.peek(&key.0, key.1, key.2) {
                    Some(
                        s @ StoredSnapshot {
                            location: SnapshotLocation::Device(buf),
                            ..
                        },
                    ) => Ok((s, buf)),
                    _ => Err(missing.to_string()),
                }
            })
            .collect::<Result<_, _>>()?;
        let (leaf, leaf_src) = *chain.last().expect("non-empty chain");
        let dst = state.restore_parts(gpu, route, &leaf.meta)?;
        let mismatch = || "session cache: snapshot layout mismatch".to_string();
        if dst.fixed.len() != leaf.fixed_bytes.len()
            || dst
                .fixed
                .iter()
                .zip(&leaf.fixed_bytes)
                .any(|(part, &bytes)| part.bytes != bytes)
            || dst.rows.len() != leaf.segments.len()
        {
            return Err(mismatch());
        }
        let mut regions = Vec::new();
        for (part, src_offset) in dst.fixed.iter().zip(leaf.offsets()) {
            regions.push(CopyRegion {
                dst: part.buf,
                dst_offset: part.offset,
                src: leaf_src,
                src_offset,
                bytes: part.bytes,
            });
        }
        let mut next = vec![0usize; dst.rows.len()];
        for &(snapshot, src) in &chain {
            let offsets = snapshot.offsets();
            if snapshot.segments.len() != dst.rows.len() {
                return Err(mismatch());
            }
            for (i, (segment, stream)) in snapshot.segments.iter().zip(&dst.rows).enumerate() {
                if segment.row_bytes != stream.row_bytes || segment.from != next[i] {
                    return Err(mismatch());
                }
                next[i] = segment.to;
                regions.push(CopyRegion {
                    dst: stream.buf,
                    dst_offset: segment.from * segment.row_bytes,
                    src,
                    src_offset: offsets[snapshot.fixed_bytes.len() + i],
                    bytes: segment.bytes(),
                });
            }
        }
        if next
            .iter()
            .zip(&dst.rows)
            .any(|(&rows, stream)| rows != stream.rows)
        {
            return Err(mismatch());
        }
        copy_regions(gpu, &regions).map_err(|e| e.to_string())?;
        state.finish_restore(gpu, route, &leaf.meta)
    }

    /// Move the held disk snapshot at `key` into the device tier. `Ok(false)`
    /// when the device tier has no room for it (nothing changed); `Err` when
    /// reading it failed (it and its disk descendants are dropped).
    fn promote(
        &mut self,
        gpu: &mut Gpu,
        key: &Key,
        bytes: u64,
        parent: Option<&Key>,
        growth: u64,
    ) -> Result<bool, String> {
        if !self.pool.can_afford(key, bytes) {
            return Ok(false);
        }
        // The parent is on the device (I2 for the first disk link, promoted
        // for the rest); pinned before eviction runs.
        if let Some(parent) = parent {
            self.link(parent);
        }
        let buf = self.make_room(gpu, bytes, growth).ok().and_then(|()| {
            gpu.bind_thread()
                .and_then(|()| gpu.hip.malloc(bytes as usize))
                .ok()
        });
        let Some(buf) = buf else {
            if let Some(parent) = parent {
                self.unlink(parent);
            }
            return Ok(false);
        };
        let disk = self.disk.as_mut().expect("promoting from disk");
        if let Err(error) = disk.load(gpu, key, &buf) {
            disk.drop_entry(key);
            if let Some(parent) = parent {
                self.unlink(parent);
            }
            if let Err(error) = gpu.hip.free(buf) {
                eprintln!("  session cache: freeing a snapshot failed: {error}");
            }
            return Err(format!(
                "session cache: reading snapshot from disk failed: {error}"
            ));
        }
        disk.held.remove(key);
        let stored = disk.peek(key).expect("held on disk");
        let snapshot = StoredSnapshot {
            location: SnapshotLocation::Device(buf),
            parent: stored.parent.clone(),
            fixed_bytes: stored.fixed_bytes.clone(),
            segments: stored.segments.clone(),
            meta: stored.meta.clone(),
            bytes: stored.bytes,
        };
        disk.take(key);
        let (domain, p, fp) = key.clone();
        let (_, displaced) = self.pool.insert_unaligned(domain, p, fp, snapshot);
        for snapshot in displaced {
            self.retire(snapshot);
        }
        if self.children.contains_key(key) {
            self.pool.pin(&key.0, key.1, key.2);
        }
        Ok(true)
    }

    /// The next position at which the prefill must stop and call
    /// [`Self::at_boundary`].
    pub fn next_boundary(&self) -> Option<usize> {
        self.turn.as_ref()?.boundaries.front().copied()
    }

    /// The live state reached `prefix.len()`, the boundary
    /// [`Self::next_boundary`] returned; capture it (pending until commit) as
    /// a delta over the deepest snapshot of a shorter prefix still present.
    pub fn at_boundary(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prefix: &[u32],
    ) -> Result<(), String> {
        let turn = self
            .turn
            .as_mut()
            .ok_or("session cache: no prefill in progress")?;
        if turn.boundaries.front() != Some(&prefix.len()) {
            return Err(format!(
                "session cache: driver reported position {} but next boundary is {:?}",
                prefix.len(),
                turn.boundaries.front()
            ));
        }
        turn.boundaries.pop_front();
        let (domain, route, p) = (turn.domain.clone(), turn.route, prefix.len());
        let fp = prefix_fingerprint(prefix);
        if self.present(&(domain.clone(), p as u64, fp)) {
            return Ok(());
        }
        let growth = state.growth_reserve_bytes();
        let ancestors: Vec<Key> = state
            .snapshot_boundaries(prefix, 0, p - 1)
            .into_iter()
            .rev()
            .map(|b| (domain.clone(), b as u64, prefix_fingerprint(&prefix[..b])))
            .collect();
        let SnapshotParts { meta, layout: live } = state.snapshot_parts(route, p)?;
        // The parent's segment ends are this capture's starts; a parent whose
        // streams do not line up is ignored (rows from 0).
        let parent = ancestors.into_iter().find_map(|key| {
            let segments = &self.entry(&key)?.segments;
            let fits = segments.len() == live.rows.len()
                && segments.iter().zip(&live.rows).all(|(segment, stream)| {
                    segment.row_bytes == stream.row_bytes && segment.to <= stream.rows
                });
            fits.then(|| (key, segments.iter().map(|s| s.to).collect::<Vec<_>>()))
        });
        let segments: Vec<Segment> = live
            .rows
            .iter()
            .enumerate()
            .map(|(i, stream)| Segment {
                row_bytes: stream.row_bytes,
                from: parent.as_ref().map_or(0, |(_, ends)| ends[i]),
                to: stream.rows,
            })
            .collect();
        let parent = parent.map(|(key, _)| key);
        let fixed_bytes: Vec<usize> = live.fixed.iter().map(|part| part.bytes).collect();
        let (offsets, total) = layout(
            fixed_bytes
                .iter()
                .copied()
                .chain(segments.iter().map(Segment::bytes)),
        );
        let bytes = total as u64;
        let skip =
            |reason: &str| eprintln!("  session cache: skipped {p}-token snapshot ({reason})");
        if bytes > self.pool.max_bytes() {
            skip("exceeds budget");
            return Ok(());
        }
        // Pinned before eviction runs, so making room never removes it.
        if let Some(parent) = &parent {
            self.link(parent);
        }
        let fits = self.make_room(gpu, bytes, growth);
        let dst = match fits.and_then(|()| {
            gpu.bind_thread()
                .and_then(|()| gpu.hip.malloc(total))
                .map_err(|_| "allocation")
        }) {
            Ok(dst) => dst,
            Err(reason) => {
                if let Some(parent) = &parent {
                    self.unlink(parent);
                }
                skip(reason);
                return Ok(());
            }
        };
        let fixed = live
            .fixed
            .iter()
            .map(|part| (part.buf, part.offset, part.bytes));
        let rows = live.rows.iter().zip(&segments).map(|(stream, segment)| {
            (
                stream.buf,
                segment.from * segment.row_bytes,
                segment.bytes(),
            )
        });
        let regions: Vec<CopyRegion<'_>> = fixed
            .chain(rows)
            .zip(offsets)
            .map(|((src, src_offset, bytes), dst_offset)| CopyRegion {
                dst: &dst,
                dst_offset,
                src,
                src_offset,
                bytes,
            })
            .collect();
        let copied = copy_regions(gpu, &regions);
        let snapshot = StoredSnapshot {
            location: SnapshotLocation::Device(dst),
            parent,
            fixed_bytes,
            segments,
            meta,
            bytes,
        };
        if let Err(error) = copied {
            self.drop_snapshot(gpu, snapshot);
            return Err(format!("session cache: snapshot copy failed: {error}"));
        }
        self.pending.push(((domain, p as u64, fp), snapshot));
        Ok(())
    }

    /// Publish this turn's snapshots, parents before children. Uncommitted
    /// snapshots are dropped by the next [`Self::begin`], so a failed turn
    /// never publishes state.
    pub fn commit(&mut self) {
        let mut refused = HashSet::new();
        for (key, snapshot) in std::mem::take(&mut self.pending) {
            if snapshot
                .parent
                .as_ref()
                .is_some_and(|parent| refused.contains(parent))
            {
                refused.insert(key);
                self.retire(snapshot);
                continue;
            }
            let (domain, p, fp) = key.clone();
            // `at_boundary` already reserved room for every pending
            // snapshot, so this insert evicts nothing.
            let (id, displaced) = self.pool.insert_unaligned(domain, p, fp, snapshot);
            if id == CheckpointId::NONE {
                refused.insert(key);
            } else if self.children.contains_key(&key) {
                self.pool.pin(&key.0, key.1, key.2);
            }
            for snapshot in displaced {
                self.retire(snapshot);
            }
        }
        self.turn = None;
    }

    /// Bytes of every published snapshot.
    pub fn stored_bytes(&self) -> u64 {
        self.pool.total_bytes()
    }

    /// Free every snapshot; with a disk tier attached, published snapshots
    /// are demoted to disk instead of dropped.
    pub fn clear(&mut self, gpu: &mut Gpu) {
        if self.disk.is_some() {
            for snapshot in std::mem::take(&mut self.release) {
                free_buffer(gpu, snapshot);
            }
            for (_, snapshot) in std::mem::take(&mut self.pending) {
                self.drop_snapshot(gpu, snapshot);
            }
            // Leaves first: each demotion unpins its parent.
            while let Some((key, snapshot)) = self.pool.pop_lru() {
                self.evict_device(gpu, key, snapshot);
            }
        }
        let pooled = self.pool.drain_blobs();
        let pending = std::mem::take(&mut self.pending)
            .into_iter()
            .map(|entry| entry.1);
        for snapshot in pooled
            .into_iter()
            .chain(pending)
            .chain(std::mem::take(&mut self.release))
        {
            free_buffer(gpu, snapshot);
        }
        self.children.clear();
        self.turn = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };
    use rdna_compute::{DType, GpuTensor};

    const FIXED_BYTES: usize = 256;
    const ROW_BYTES: usize = 16;
    const ROW_CAPACITY: usize = 1024;
    const STRIDE: usize = 128;
    /// A token after which the toy also splits, like a message end.
    const MARKER: u32 = 999_999_999;

    /// A fixed part that depends on the whole prefix and one append-only row
    /// per token that depends only on the tokens up to it.
    struct Toy {
        fixed: GpuTensor,
        rows: GpuTensor,
        position: usize,
    }

    impl Toy {
        fn layout(&self, rows: usize) -> StateLayout<'_> {
            StateLayout {
                fixed: vec![StatePart {
                    buf: &self.fixed.buf,
                    offset: 0,
                    bytes: FIXED_BYTES,
                }],
                rows: vec![RowStream {
                    buf: &self.rows.buf,
                    row_bytes: ROW_BYTES,
                    rows,
                }],
            }
        }

        /// Stand-in for a prefill chunk.
        fn advance(&mut self, gpu: &mut Gpu, prefix: &[u32]) {
            self.position = prefix.len();
            gpu.hip
                .memcpy_htod(&self.fixed.buf, &fixed_of(prefix))
                .unwrap();
            gpu.hip
                .memcpy_htod(&self.rows.buf, &rows_of(prefix))
                .unwrap();
        }

        /// Fixed bytes and the valid rows.
        fn bytes(&self, gpu: &mut Gpu) -> (Vec<u8>, Vec<u8>) {
            gpu.hip.device_synchronize().unwrap();
            let mut fixed = vec![0; FIXED_BYTES];
            gpu.hip.memcpy_dtoh(&mut fixed, &self.fixed.buf).unwrap();
            let mut rows = vec![0; ROW_CAPACITY * ROW_BYTES];
            gpu.hip.memcpy_dtoh(&mut rows, &self.rows.buf).unwrap();
            rows.truncate(self.position * ROW_BYTES);
            (fixed, rows)
        }
    }

    impl SessionState for Toy {
        fn snapshot_scope(&self, _route: SessionRoute) -> Option<String> {
            Some("toy".to_string())
        }
        fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
            let mut boundaries: Vec<usize> = (after / STRIDE + 1..=up_to / STRIDE)
                .map(|k| k * STRIDE)
                .collect();
            boundaries.extend((after..up_to).filter(|&i| prompt[i] == MARKER).map(|i| i + 1));
            boundaries.sort_unstable();
            boundaries.dedup();
            boundaries
        }
        fn snapshot_parts(
            &mut self,
            _route: SessionRoute,
            position: usize,
        ) -> Result<SnapshotParts<'_>, String> {
            assert_eq!(self.position, position);
            Ok(SnapshotParts {
                meta: (position as u64).to_le_bytes().to_vec(),
                layout: self.layout(position),
            })
        }
        fn restore_parts(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            meta: &[u8],
        ) -> Result<StateLayout<'_>, String> {
            Ok(self.layout(u64::from_le_bytes(meta.try_into().unwrap()) as usize))
        }
        fn finish_restore(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            meta: &[u8],
        ) -> Result<(), String> {
            self.position = u64::from_le_bytes(meta.try_into().unwrap()) as usize;
            Ok(())
        }
        fn growth_reserve_bytes(&self) -> u64 {
            0
        }
        fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
            self.position = 0;
            gpu.hip
                .memset(&self.fixed.buf, 0, FIXED_BYTES)
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memset(&self.rows.buf, 0, ROW_CAPACITY * ROW_BYTES)
                .map_err(|e| e.to_string())
        }
    }

    fn fixed_of(prefix: &[u32]) -> Vec<u8> {
        let seed = prefix_fingerprint(prefix).to_le_bytes();
        (0..FIXED_BYTES).map(|i| seed[i % 8] ^ i as u8).collect()
    }

    fn rows_of(prefix: &[u32]) -> Vec<u8> {
        (1..=prefix.len())
            .flat_map(|end| {
                let seed = prefix_fingerprint(&prefix[..end]).to_le_bytes();
                (0..ROW_BYTES).map(move |i| seed[i % 8] ^ i as u8)
            })
            .collect()
    }

    fn domain() -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![1],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![2],
                config_digest: vec![3],
            },
            template: TemplateIdentity {
                template_digest: vec![4],
                normalization_tag: "jinja".into(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "toy".into(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![],
                v_stride_bytes: vec![],
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: "gpu-0".into(),
                topology_id: "single".into(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".into(),
            },
        }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 100_000 + i).collect()
    }

    /// One turn: restore what `plan` offers, prefill to the end capturing at
    /// every boundary, optionally commit. Returns the planned reuse.
    fn turn(
        cache: &mut SessionCache,
        gpu: &mut Gpu,
        toy: &mut Toy,
        prompt: &[u32],
        commit: bool,
    ) -> usize {
        let reused = cache.plan(toy, prompt, SessionRoute::Ar);
        cache
            .begin(gpu, toy, prompt, SessionRoute::Ar, reused)
            .unwrap();
        while let Some(b) = cache.next_boundary() {
            toy.advance(gpu, &prompt[..b]);
            cache.at_boundary(gpu, toy, &prompt[..b]).unwrap();
        }
        toy.advance(gpu, prompt);
        if commit {
            cache.commit();
        }
        reused
    }

    /// Restore `prompt`'s planned snapshot and check it equals the state a
    /// cold prefill of that prefix leaves.
    fn assert_restores(
        cache: &mut SessionCache,
        gpu: &mut Gpu,
        toy: &mut Toy,
        prompt: &[u32],
        expected: usize,
    ) {
        let reused = cache.plan(toy, prompt, SessionRoute::Ar);
        assert_eq!(reused, expected);
        cache
            .begin(gpu, toy, prompt, SessionRoute::Ar, reused)
            .unwrap();
        assert_eq!(toy.position, reused);
        assert_eq!(
            toy.bytes(gpu),
            (fixed_of(&prompt[..reused]), rows_of(&prompt[..reused]))
        );
        cache.commit();
    }

    #[test]
    fn delta_snapshots_restore_exactly_share_prefixes_and_evict_leaves() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
        };
        // Every snapshot below holds the fixed part plus one stride of rows.
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 4 * one);
        let a = prompt(1, 3 * STRIDE + 5);

        // Uncommitted captures are invisible and abandoned (with their
        // parent links) by the next begin.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, false), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 0);
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, true), 0);
        assert!(cache.pending.is_empty());

        // A three-link chain, each link one stride of rows.
        assert_eq!(cache.pool.total_bytes(), 3 * one);
        assert_restores(&mut cache, gpu, &mut toy, &a, 3 * STRIDE);

        // A branch shares a's first link.
        let mut b = a[..STRIDE].to_vec();
        b.extend(prompt(2, STRIDE + 7));
        assert_eq!(turn(&mut cache, gpu, &mut toy, &b, true), STRIDE);
        assert_eq!(cache.pool.total_bytes(), 4 * one);
        assert_restores(&mut cache, gpu, &mut toy, &b, 2 * STRIDE);

        // A full budget evicts the least recently used leaf (a's deepest),
        // never a parent that still has children.
        let c = prompt(3, STRIDE + 9);
        turn(&mut cache, gpu, &mut toy, &c, true);
        assert_eq!(cache.plan(&toy, &c, SessionRoute::Ar), STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &a, 2 * STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &b, 2 * STRIDE);
        assert_eq!(cache.children.values().sum::<usize>(), 2);
        cache.clear(gpu);

        // A snapshot larger than the whole budget is skipped, nothing published.
        let mut tiny = SessionCache::new(domain(), one - 1);
        turn(&mut tiny, gpu, &mut toy, &a, true);
        assert_eq!(tiny.plan(&toy, &a, SessionRoute::Ar), 0);
        assert!(tiny.children.is_empty());
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn live_begin_keeps_state_drops_pending_and_captures_nothing() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 4 * one);
        let a = prompt(1, 2 * STRIDE + 5);

        // An uncommitted turn leaves pending snapshots and an armed turn.
        turn(&mut cache, gpu, &mut toy, &a, false);
        assert_eq!(cache.pending.len(), 2);

        // A live begin drops them, arms no boundary and leaves the state alone.
        let before = toy.bytes(gpu);
        cache.begin_live(gpu, &mut toy, &a, SessionRoute::Ar, a.len(), false);
        assert!(cache.pending.is_empty());
        assert!(cache.children.is_empty());
        assert_eq!(cache.next_boundary(), None);
        assert_eq!(toy.position, a.len());
        assert_eq!(toy.bytes(gpu), before);

        // Committing the live turn publishes nothing.
        cache.commit();
        assert_eq!(cache.pool.total_bytes(), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 0);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn prompt_aware_boundaries_plan_arm_and_capture_live() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 64 * one);
        let scoped = domain().scoped("toy");
        let held = |cache: &SessionCache, prompt: &[u32], p: usize| {
            cache
                .pool
                .contains(&scoped, p as u64, prefix_fingerprint(&prompt[..p]))
        };

        // Markers at 40 and 70 sit below one stride (and are not page
        // aligned); 200 sits between strides.
        let mut m = prompt(4, 2 * STRIDE + 10);
        for at in [39, 69, 199] {
            m[at] = MARKER;
        }
        assert_eq!(
            toy.snapshot_boundaries(&m, 0, m.len()),
            [40, 70, STRIDE, 200, 2 * STRIDE]
        );
        assert_eq!(toy.snapshot_boundaries(&m, 40, 200), [70, STRIDE, 200]);

        // begin arms the marker boundaries; the cold turn captures each and
        // commit publishes them although they are not page multiples.
        cache.begin(gpu, &mut toy, &m, SessionRoute::Ar, 0).unwrap();
        assert_eq!(cache.next_boundary(), Some(40));
        cache.commit();
        assert_eq!(turn(&mut cache, gpu, &mut toy, &m, true), 0);
        for p in [40, 70, STRIDE, 200, 2 * STRIDE] {
            assert!(held(&cache, &m, p), "boundary {p} missing");
        }
        assert_eq!(cache.pool.len(), 5);

        // plan finds the marker-boundary snapshot below the first stride.
        let mut early = m[..70].to_vec();
        early.extend(prompt(5, 20));
        assert_eq!(cache.plan(&toy, &early, SessionRoute::Ar), 70);
        assert_restores(&mut cache, gpu, &mut toy, &early, 70);

        // A live continuation that captures: state at the end of m, extended
        // by a message with a marker. Only boundaries above `reused` are
        // armed, and commit publishes them.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &m, true), 2 * STRIDE);
        let mut tail = prompt(6, 30);
        tail[9] = MARKER;
        let mut live = m.clone();
        live.extend(tail);
        let mut probe = live.clone();
        probe.push(1);
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), 2 * STRIDE);
        cache.begin_live(gpu, &mut toy, &live, SessionRoute::Ar, m.len(), true);
        let marker_end = m.len() + 10;
        assert_eq!(cache.next_boundary(), Some(marker_end));
        toy.advance(gpu, &live[..marker_end]);
        cache
            .at_boundary(gpu, &mut toy, &live[..marker_end])
            .unwrap();
        assert_eq!(cache.next_boundary(), None);
        toy.advance(gpu, &live);
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), 2 * STRIDE);
        cache.commit();
        assert!(held(&cache, &live, marker_end));
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), marker_end);
        assert_restores(&mut cache, gpu, &mut toy, &probe, marker_end);
        cache.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn disk_tier_demotes_promotes_and_survives_reopen() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
        };
        let dir = tempfile::tempdir().unwrap();
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let open = |build: u8| {
            SessionCache::new(domain(), 2 * one).with_disk_build(
                dir.path(),
                64 * (one + 8192),
                [build; 32],
            )
        };
        let snaps = || {
            std::fs::read_dir(dir.path())
                .unwrap()
                .filter(|e| e.as_ref().unwrap().path().extension() == Some("snap".as_ref()))
                .count()
        };
        let c = prompt(1, 2 * STRIDE + 5);
        let d = prompt(2, 2 * STRIDE + 5);

        // d's captures demote c's two snapshots (leaf first).
        let mut cache = open(7);
        assert!(cache.disk.is_some());
        turn(&mut cache, gpu, &mut toy, &c, true);
        turn(&mut cache, gpu, &mut toy, &d, true);
        assert_eq!(snaps(), 2);
        assert_eq!(cache.pool.len(), 2);

        // Restoring c promotes both links, demoting d's.
        assert_restores(&mut cache, gpu, &mut toy, &c, 2 * STRIDE);
        assert_eq!(snaps(), 2);
        let scoped = domain().scoped("toy");
        let c2 = prefix_fingerprint(&c[..2 * STRIDE]);
        assert!(cache.pool.contains(&scoped, 2 * STRIDE as u64, c2));

        // clear demotes everything; a second process is locked out.
        cache.clear(gpu);
        assert_eq!(snaps(), 4);
        assert!(open(7).disk.is_none());

        // A reopened cache restores both chains exactly from disk.
        drop(cache);
        let mut cache = open(7);
        assert_restores(&mut cache, gpu, &mut toy, &d, 2 * STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &c, 2 * STRIDE);
        cache.clear(gpu);
        drop(cache);

        // Another build never plans this build's files.
        let other = open(8);
        assert_eq!(other.plan(&toy, &c, SessionRoute::Ar), 0);
        drop(other);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }
}
