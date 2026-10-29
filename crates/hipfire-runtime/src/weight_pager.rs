// SPDX-License-Identifier: MIT
// Copyright (c) 2026 mad-lab-kbando
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight pager — runtime residency management for MoE/dense weights (MAD-93 v0.1).
//!
//! Hipfire today loads all weights to VRAM at startup. For models that exceed
//! VRAM (Qwen3.5-REAP-97B-A10B is the v0.1 target), we need to keep most experts
//! on host (read from the HFQ file) and page them in to VRAM on demand based on
//! routing decisions made by [`crate::cpu_router::CpuRouter`].
//!
//! ## Architecture (foundational)
//!
//! - **CPU is the scheduler authority.** [`CpuRouter`](crate::cpu_router::CpuRouter)
//!   replicates the per-layer router GEMV on CPU so we know top-k expert indices
//!   without a GPU→CPU sync inside the forward path. The pager consumes those
//!   indices and decides what to fetch / evict.
//! - **Transport is abstracted.** Today it's pread + `hipMemcpyAsync`
//!   ([`PreadH2DTransport`]). In a future commit we drop in
//!   `IoUringP2PTransport` for true NVMe→VRAM DMA without changing anything
//!   above this layer.
//! - **Stable per-weight identity.** [`WeightId`] is a small, hashable enum so
//!   the residency map and the (file_offset, byte_len) lookup table can be
//!   keyed identically. This is what an io_uring submission queue would consume
//!   directly.
//! - **Pager owns its VRAM.** All paged weight allocations route through the
//!   pager (not ad-hoc `gpu.alloc_tensor`), so we can later export the slabs as
//!   `dma_buf` for P2P DMA without reorganizing call sites.
//!
//! ## v0.1 scope
//!
//! - [`Transport`] trait + [`PreadH2DTransport`] impl
//! - [`WeightPager`] with residency map and `ensure_resident` (synchronous, no
//!   real eviction yet — assumes VRAM is large enough)
//! - [`WeightId`] schema covering MoE experts, dense attention, norms, embeds
//!
//! Real eviction, async transfer overlap, and predictive prefetch land in
//! follow-up commits — the trait shapes here are the seams those commits plug
//! into without changing the forward path.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::Path;

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

use crate::hfq::HfqFile;

// ---------------------------------------------------------------------------
// Identity: WeightId
// ---------------------------------------------------------------------------

/// Stable identity for a weight that the pager can move between host and VRAM.
///
/// The variants enumerate every kind of weight that participates in paging.
/// For v0.1 only [`WeightId::Expert`] actually pages — the others are listed
/// so the residency map can track them as "always resident" without a special
/// case at the call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightId {
    /// Routed expert weight (one of the 256 experts in Qwen3.5-MoE-A3B).
    /// `role` distinguishes the fused gate_up matrix from the down matrix.
    Expert {
        layer: u16,
        expert: u16,
        role: ExpertRole,
    },
    /// Always-on shared expert (one per layer).
    SharedExpert { layer: u16, role: SharedRole },
    /// Per-layer router weight (small, always-resident in v0.1, but tracked
    /// here so future commits can page it for very large MoE configs).
    Router { layer: u16 },
    /// Dense attention weight (q/k/v/o).
    DenseAttn { layer: u16, role: AttnRole },
    /// RMSNorm gain vector. Tiny but per-layer.
    Norm { layer: u16, kind: NormKind },
    /// Token embedding table (always resident in v0.1).
    Embed,
    /// LM head (often shares storage with Embed; pager tracks separately).
    LmHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertRole {
    /// Fused gate || up: shape `[2 * moe_intermediate, hidden]`.
    GateUp,
    /// Down projection: shape `[hidden, moe_intermediate]`.
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedRole {
    Gate,
    Up,
    Down,
    /// Scalar sigmoid gate on the shared-expert add: `[1, hidden]` row vector.
    SigmoidGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnRole {
    Q,
    K,
    V,
    O,
    /// Fused QKV when the model stores them as one tensor.
    Qkv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Pre-attention norm.
    Attn,
    /// Pre-MoE / pre-FFN norm.
    Ffn,
    /// Final norm before LM head.
    Final,
}

// ---------------------------------------------------------------------------
// Bounded slot residency
// ---------------------------------------------------------------------------
// Adapted from warpfront/hipfire PR #870, a73d9491d3e07f3e821a974a9c5f72e35e34ab93.
// The original per-pool LRU/transaction boundary is extended with generation
// tickets and access-ordered (rather than completion-ordered) recency.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Pending,
    Resident,
}

#[derive(Debug, Clone, Copy)]
struct SlotEntry<P> {
    pool: P,
    slot: usize,
    state: SlotState,
    generation: u64,
}

/// A reservation in one [`SlotCache`] instance. Only the first request for a
/// missing key owns its fill; duplicate pending requests do not acquire it.
/// A caller must withdraw `victim()` from its pointer table before overwriting
/// the slot, and publish the new pointer only after the copy succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotReservation<K, P> {
    pub key: K,
    pub pool: P,
    pub slot: usize,
    generation: u64,
    needs_fill: bool,
    victim: Option<K>,
}

impl<K: Copy, P> SlotReservation<K, P> {
    pub fn needs_fill(&self) -> bool {
        self.needs_fill
    }

    /// Monotonically increasing identity of this fill within its cache.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The formerly resident key replaced by this new fill, if any.
    pub fn victim(&self) -> Option<K> {
        self.victim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotCacheError {
    ZeroSlots,
    PoolMismatch,
    AllSlotsPending,
    GenerationExhausted,
}

impl std::fmt::Display for SlotCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSlots => write!(f, "slot cache requires at least one slot per pool"),
            Self::PoolMismatch => write!(f, "cache key was requested from a different pool"),
            Self::AllSlotsPending => write!(f, "all slots in the pool have pending fills"),
            Self::GenerationExhausted => write!(f, "slot reservation generation exhausted"),
        }
    }
}

impl std::error::Error for SlotCacheError {}

/// Device-independent, fixed-capacity per-pool LRU with transactional fills.
///
/// Logical recency follows successful [`reserve`](Self::reserve) calls,
/// including pending duplicates. Completion order never changes recency.
/// Pending entries are not visible as resident and cannot be evicted. Failed
/// fills are forgotten rather than restoring potentially overwritten victims.
///
/// This is bookkeeping, not a copy/event owner or a consumer lease. Callers
/// must order table withdrawal, copies, publication and consuming launches;
/// before invalidating a pending key they must retire its physical copy. A
/// generation rejects stale completion tickets but cannot stop an old DMA.
/// Tickets belong to the originating cache; do not pass them between instances.
pub struct SlotCache<K, P>
where
    K: Copy + Eq + std::hash::Hash,
    P: Copy + Eq + std::hash::Hash,
{
    slots_per_pool: usize,
    entries: HashMap<K, SlotEntry<P>>,
    occupants: HashMap<(P, usize), K>,
    lru: HashMap<P, VecDeque<usize>>,
    next_slot: HashMap<P, usize>,
    generation: u64,
    hits: u64,
    misses: u64,
}

impl<K, P> SlotCache<K, P>
where
    K: Copy + Eq + std::hash::Hash,
    P: Copy + Eq + std::hash::Hash,
{
    pub fn new(slots_per_pool: usize) -> Result<Self, SlotCacheError> {
        if slots_per_pool == 0 {
            return Err(SlotCacheError::ZeroSlots);
        }
        Ok(Self {
            slots_per_pool,
            entries: HashMap::new(),
            occupants: HashMap::new(),
            lru: HashMap::new(),
            next_slot: HashMap::new(),
            generation: 0,
            hits: 0,
            misses: 0,
        })
    }

    pub fn slots_per_pool(&self) -> usize {
        self.slots_per_pool
    }

    /// Committed hits and newly reserved fills. Pending duplicates count as
    /// neither: they touch recency but schedule no additional fill.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Inspect residency without recording a logical access; use `reserve`
    /// to access a key and update LRU. Pending fills are invisible here.
    pub fn resident_slot(&self, key: K) -> Option<usize> {
        self.entries
            .get(&key)
            .and_then(|entry| (entry.state == SlotState::Resident).then_some(entry.slot))
    }

    pub fn reserve(&mut self, pool: P, key: K) -> Result<SlotReservation<K, P>, SlotCacheError> {
        if let Some(entry) = self.entries.get(&key).copied() {
            if entry.pool != pool {
                return Err(SlotCacheError::PoolMismatch);
            }
            if entry.state == SlotState::Resident {
                self.hits = self.hits.saturating_add(1);
            }
            self.touch_slot(pool, entry.slot);
            return Ok(SlotReservation {
                key,
                pool,
                slot: entry.slot,
                generation: entry.generation,
                needs_fill: false,
                victim: None,
            });
        }

        // Check before changing occupancy or LRU. Never wrap a ticket identity.
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(SlotCacheError::GenerationExhausted)?;
        let next = self.next_slot.get(&pool).copied().unwrap_or(0);
        let slot = if next < self.slots_per_pool {
            self.next_slot.insert(pool, next + 1);
            next
        } else {
            let queue = self.lru.get(&pool).expect("allocated pool has an LRU");
            // Reuse holes left by rollback/invalidate before evicting residents.
            // Searching, instead of rotating pending slots, preserves all other
            // keys' logical recency even when a request cannot obtain a slot.
            queue
                .iter()
                .copied()
                .find(|&slot| !self.occupants.contains_key(&(pool, slot)))
                .or_else(|| {
                    queue.iter().copied().find(|&slot| {
                        self.occupants
                            .get(&(pool, slot))
                            .and_then(|old| self.entries.get(old))
                            .is_some_and(|entry| entry.state == SlotState::Resident)
                    })
                })
                .ok_or(SlotCacheError::AllSlotsPending)?
        };
        let victim = self.occupants.remove(&(pool, slot));
        if let Some(old) = victim {
            self.entries.remove(&old);
        }
        self.generation = generation;
        self.misses = self.misses.saturating_add(1);
        self.entries.insert(
            key,
            SlotEntry {
                pool,
                slot,
                state: SlotState::Pending,
                generation,
            },
        );
        self.occupants.insert((pool, slot), key);
        self.touch_slot(pool, slot);
        Ok(SlotReservation {
            key,
            pool,
            slot,
            generation,
            needs_fill: true,
            victim,
        })
    }

    fn pending_ticket_matches(&self, reservation: &SlotReservation<K, P>) -> bool {
        reservation.needs_fill
            && self.entries.get(&reservation.key).is_some_and(|entry| {
                entry.pool == reservation.pool
                    && entry.slot == reservation.slot
                    && entry.generation == reservation.generation
                    && entry.state == SlotState::Pending
            })
    }

    /// Publish after successful I/O and copy. Duplicate/stale tickets are
    /// rejected. This deliberately does not touch the logical-access LRU.
    pub fn commit(&mut self, reservation: SlotReservation<K, P>) -> bool {
        if !self.pending_ticket_matches(&reservation) {
            return false;
        }
        self.entries.get_mut(&reservation.key).unwrap().state = SlotState::Resident;
        true
    }

    /// Forget a failed fill; never resurrect the victim's overwritten bytes.
    pub fn rollback(&mut self, reservation: SlotReservation<K, P>) -> bool {
        if !self.pending_ticket_matches(&reservation) {
            return false;
        }
        self.invalidate(reservation.key)
    }

    /// Remove a mapping. Caller must retire any pending copy and users of the
    /// old slot before allowing the freed capacity to be reused.
    pub fn invalidate(&mut self, key: K) -> bool {
        let Some(entry) = self.entries.remove(&key) else {
            return false;
        };
        self.occupants.remove(&(entry.pool, entry.slot));
        true
    }

    /// Forget every mapping and restore physical slot allocation order.
    ///
    /// The caller must first retire every pending copy and consumer. Ticket
    /// generations and cumulative statistics deliberately survive the reset,
    /// so a stale completion cannot commit a later use of the same slot.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.occupants.clear();
        self.lru.clear();
        self.next_slot.clear();
    }

    fn touch_slot(&mut self, pool: P, slot: usize) {
        let queue = self.lru.entry(pool).or_default();
        if let Some(pos) = queue.iter().position(|&candidate| candidate == slot) {
            queue.remove(pos);
        }
        queue.push_back(slot);
    }
}

// ---------------------------------------------------------------------------
// Transport: how bytes get from HFQ file to VRAM
// ---------------------------------------------------------------------------

/// Opaque handle for an in-flight or completed transfer. Submit returns one,
/// `wait` consumes a slice of them. v0.1 uses a simple counter; future
/// async-overlap impls will back this with a `hipEvent` or io_uring CQE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferHandle(u64);

/// Abstraction over how the pager moves bytes from host storage to VRAM.
///
/// **This is the migration seam for the NVMe→VRAM DMA future.** Today's impl
/// ([`PreadH2DTransport`]) does `pread` into a host staging buffer then
/// `hipMemcpyAsync` to VRAM. A future `IoUringP2PTransport` reads directly
/// into VRAM via `dma_buf` + io_uring with no host hop. The pager never sees
/// the difference.
///
/// `fetch` is responsible for both the allocation (so io_uring can use a
/// `dma_buf`-exportable slab when needed) and the transfer. `wait` exists for
/// future async overlap; today's pread path completes synchronously inside
/// `fetch` and `wait` is a no-op.
pub trait Transport: Send {
    /// Allocate a `GpuTensor` (with `DType::Raw`, shape `[len]`) and populate
    /// it with `len` bytes from `hfq_offset` in the HFQ file. Returns the
    /// fresh tensor and a handle that can be waited on.
    ///
    /// In v0.1 the transfer is synchronous (handle is informational); in
    /// follow-ups the transport may submit async and have callers `wait`.
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)>;

    /// Block until every handle in `handles` has completed. v0.1 no-op
    /// because `fetch` is synchronous; defined for forward compatibility.
    fn wait(&mut self, handles: &[TransferHandle]) -> HipResult<()>;

    /// Hint: does this transport need pager-allocated VRAM slabs to be
    /// exported as `dma_buf` for P2P DMA? Pager checks this at allocation
    /// time. Default false (host-staged path doesn't care).
    fn requires_dma_buf_alloc(&self) -> bool {
        false
    }

    /// Hint: required alignment for `hfq_offset` in bytes. `O_DIRECT` paths
    /// need 4 KB; pread doesn't care. Pager validates on submit.
    fn alignment(&self) -> usize {
        1
    }
}

/// v0.1 transport: pread the requested byte range from the HFQ file into a
/// reusable host buffer, then upload to VRAM via [`Gpu::upload_raw`]
/// (which internally does `hipMalloc` + `hipMemcpy(H2D)`).
///
/// Synchronous in this commit. A follow-up commit replaces the staging with a
/// pool of pinned (`hipHostMalloc`'d) buffers and uses `hipMemcpyAsync` on a
/// dedicated stream so the next-layer prefetch can overlap with current-layer
/// compute.
pub struct PreadH2DTransport {
    /// Owned file handle for the HFQ file. We open our own (rather than
    /// borrowing `HfqFile`'s) so a future `IoUringP2PTransport` can register
    /// its fd with io_uring + `dma_buf` independently. Path is held alongside
    /// for diagnostics.
    file: File,
    path: std::path::PathBuf,
    /// Reusable host staging buffer. Grows monotonically to the largest
    /// tensor size we've seen.
    staging: Vec<u8>,
    /// Monotonic handle ID. v0.1 fetches complete synchronously, so this is
    /// purely informational; future async impls will key real completion
    /// state on this id.
    next_handle: u64,
}

impl PreadH2DTransport {
    /// Open the HFQ file at `path` for paged reads.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Hint sequential-ish access for the page-cache layer. Tensors don't
        // overlap so reads are effectively sequential within a tensor and
        // random across tensors; the kernel's readahead handles the within
        // case correctly with this advice.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
            }
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            staging: Vec::new(),
            next_handle: 0,
        })
    }

    /// Path the transport was opened with. Useful for diagnostics + the
    /// future io_uring impl which needs to register the same path with
    /// io_uring SQE buffers.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn next_handle(&mut self) -> TransferHandle {
        let h = TransferHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    /// Read `len` bytes at `offset` into `self.staging[..len]`. Linux uses
    /// `pread` (positional read, no seek state); other platforms fall back
    /// to `seek + read_exact` (correct but loses thread-safety on the file
    /// — a non-issue today since the pager is single-threaded).
    fn pread_into_staging(&mut self, offset: usize, len: usize) -> std::io::Result<()> {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file
                .read_exact_at(&mut self.staging[..len], offset as u64)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            self.file.seek(SeekFrom::Start(offset as u64))?;
            self.file.read_exact(&mut self.staging[..len])?;
        }
        Ok(())
    }
}

impl Transport for PreadH2DTransport {
    fn fetch(
        &mut self,
        hfq_offset: usize,
        len: usize,
        gpu: &mut Gpu,
    ) -> HipResult<(GpuTensor, TransferHandle)> {
        // 1. Host: pread the bytes into our staging buffer.
        self.pread_into_staging(hfq_offset, len).map_err(|e| {
            hip_bridge::HipError::new(
                0,
                &format!("pread {} bytes at offset {}: {}", len, hfq_offset, e),
            )
        })?;
        // 2. GPU: alloc + memcpy_htod via the existing rdna-compute helper.
        //    `dtype: Raw` because the pager doesn't care about element layout
        //    — that interpretation belongs to `WeightTensor` at the call site.
        let tensor = gpu.upload_raw(&self.staging[..len], &[len])?;
        Ok((tensor, self.next_handle()))
    }

    fn wait(&mut self, _handles: &[TransferHandle]) -> HipResult<()> {
        // Transfers complete synchronously inside `fetch` in v0.1.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Construction-time config. Keep this small and explicit — runtime flags
/// belong on `Qwen35Config`, not here.
#[derive(Debug, Clone)]
pub struct PagerConfig {
    /// Soft cap on VRAM bytes the pager is allowed to hold for paged weights.
    /// Eviction kicks in when adding a new resident weight would exceed this.
    /// `u64::MAX` means "unlimited" (effectively disables eviction — useful
    /// for testing the routing path without VRAM pressure).
    pub vram_budget_bytes: u64,
    /// If true, the pager prints structured residency events to stderr.
    /// Disabled by default; useful when debugging eviction policy.
    pub trace: bool,
}

impl Default for PagerConfig {
    fn default() -> Self {
        Self {
            vram_budget_bytes: u64::MAX,
            trace: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

/// Tracks which weights are currently resident in VRAM and provides the
/// `ensure_resident` / `evict_lru_until` primitives the forward path uses.
///
/// **This is the GPU-side of the pager.** The CPU-side scheduling
/// (compute router → decide top-k → call ensure_resident) happens in the
/// caller; see [`crate::cpu_router::CpuRouter`].
pub struct WeightPager {
    /// What's currently in VRAM. Maps weight identity to a `Resident` record
    /// holding the buffer + bookkeeping for LRU.
    resident: HashMap<WeightId, Resident>,
    /// Recency queue for LRU eviction. Most-recently-used at the back.
    /// We use VecDeque because mutations are O(n) but n is tiny (top-k
    /// experts × layers, max ~thousands), and VecDeque iteration is
    /// cache-friendly.
    lru: VecDeque<WeightId>,
    /// Bytes currently held by `resident`.
    vram_used_bytes: u64,
    /// Per-weight (file_offset, byte_len) for cold-load via Transport.
    /// Populated at registration time when the model loader walks the HFQ
    /// tensor index. Stable across the run.
    catalog: HashMap<WeightId, ByteRange>,
    /// Transport implementation (v0.1: pread + H2D, future: io_uring + P2P).
    transport: Box<dyn Transport>,
    /// Construction-time config.
    config: PagerConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    pub offset: usize,
    pub len: usize,
}

/// Uniform-shape MoE expert metadata for v0.1.
///
/// Qwen3.5-MoE-A3B has 256 experts that all share the same gate_up and down
/// shape, so we store one set of dimensions per layer instead of per-expert.
/// When we add a heterogeneous-shape MoE arch (Mixtral derivatives, etc.),
/// this generalizes to `Vec<ExpertShape>` indexed by expert index.
#[derive(Debug, Clone, Copy)]
pub struct ExpertShape {
    /// Output rows of the fused gate_up matrix = `2 * moe_intermediate_size`.
    pub gate_up_m: usize,
    /// Input cols of gate_up = `hidden_size`.
    pub gate_up_k: usize,
    /// Output rows of down = `hidden_size`.
    pub down_m: usize,
    /// Input cols of down = `moe_intermediate_size`.
    pub down_k: usize,
}

struct Resident {
    /// The actual VRAM tensor (pager owns its lifecycle). `dtype: Raw` —
    /// callers reinterpret the bytes via their own `WeightTensor` wrapper at
    /// access time. Storing as `GpuTensor` (rather than the lower-level
    /// `DeviceBuffer`) keeps the pager idiomatic with the rest of the
    /// rdna-compute API and lets us free via `gpu.free_tensor`.
    tensor: GpuTensor,
    /// Cached byte length so eviction can update `vram_used_bytes` cheaply.
    bytes: u64,
}

impl WeightPager {
    pub fn new(transport: Box<dyn Transport>, config: PagerConfig) -> Self {
        Self {
            resident: HashMap::new(),
            lru: VecDeque::new(),
            vram_used_bytes: 0,
            catalog: HashMap::new(),
            transport,
            config,
        }
    }

    /// Convenience: open `hfq_path` with the v0.1 pread+H2D transport.
    /// Equivalent to constructing a [`PreadH2DTransport`] manually and passing
    /// it to [`Self::new`].
    pub fn with_pread_transport(hfq_path: &Path, config: PagerConfig) -> std::io::Result<Self> {
        let transport = PreadH2DTransport::open(hfq_path)?;
        Ok(Self::new(Box::new(transport), config))
    }

    /// Register that `id` lives at `range` in the HFQ file. Called by the
    /// loader when it walks the tensor index. Must be called before any
    /// `ensure_resident(id)` for that id.
    pub fn register(&mut self, id: WeightId, range: ByteRange) {
        self.catalog.insert(id, range);
    }

    /// Number of registered weights. Useful for diagnostics.
    pub fn registered_count(&self) -> usize {
        self.catalog.len()
    }

    /// Returns true if `id` is currently in VRAM.
    pub fn is_resident(&self, id: WeightId) -> bool {
        self.resident.contains_key(&id)
    }

    /// Ensure `id` is in VRAM. Synchronous.
    ///
    /// Behavior:
    /// - Already resident → mark recently-used, return Ok.
    /// - Not registered (loader bug) → `NotRegistered` error, no GPU work.
    /// - Cold (registered but not resident) → if adding `id` would exceed
    ///   `config.vram_budget_bytes`, evict LRU residents until enough room.
    ///   Then fetch via transport, populate, track residency.
    pub fn ensure_resident(&mut self, id: WeightId, gpu: &mut Gpu) -> Result<(), WeightPagerError> {
        if self.resident.contains_key(&id) {
            self.touch_lru(id);
            return Ok(());
        }
        let range = *self
            .catalog
            .get(&id)
            .ok_or(WeightPagerError::NotRegistered(id))?;
        let need = range.len as u64;
        // Hard cap: a single weight that exceeds `vram_budget_bytes` can't
        // fit no matter how much we evict. Reject up front rather than
        // dutifully draining the residency map and then fetching anyway —
        // that path used to silently violate the budget because
        // `evict_lru_until` interpreted `need > budget` as "free everything"
        // and stopped, allowing the subsequent fetch to push usage past the
        // cap.
        self.would_fit(need)?;
        // Evict if cold-loading `id` would exceed budget. Skip when budget is
        // u64::MAX (the unlimited / testing default — saves the LRU walk).
        if self.config.vram_budget_bytes != u64::MAX
            && self.vram_used_bytes.saturating_add(need) > self.config.vram_budget_bytes
        {
            self.evict_lru_until(need, gpu)?;
        }
        let (tensor, _handle) = self.transport.fetch(range.offset, range.len, gpu)?;
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(need);
        self.resident.insert(
            id,
            Resident {
                tensor,
                bytes: need,
            },
        );
        self.lru.push_back(id);
        if self.config.trace {
            eprintln!(
                "[weight_pager] cold-load {id:?} ({} bytes) — {} resident, {} bytes used",
                range.len,
                self.resident.len(),
                self.vram_used_bytes
            );
        }
        Ok(())
    }

    /// Patch the device-side `expert_*_ptrs` indirection table so the indexed
    /// MoE GEMV kernels read the currently-resident buffer pointers for the
    /// active experts in `top_indices` for `layer`.
    ///
    /// The ptr tables are laid out as `[num_experts × u64]` (8-byte device
    /// pointers per expert slot). For each `idx` in `top_indices`, we write
    /// the GPU pointer of that expert's resident gate_up buffer into
    /// `gate_up_ptrs.buf[idx * 8 .. idx * 8 + 8]`, same for down_ptrs.
    ///
    /// Caller must have already called `ensure_resident` for both
    /// `WeightId::Expert{layer, expert: idx, role: GateUp}` and
    /// `WeightId::Expert{layer, expert: idx, role: Down}` for every idx in
    /// `top_indices` — this method asserts that and panics on miss (loader bug).
    pub fn patch_expert_ptr_table(
        &self,
        layer: u16,
        top_indices: &[u16],
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        for &idx in top_indices {
            let gate_up_id = WeightId::Expert {
                layer,
                expert: idx,
                role: ExpertRole::GateUp,
            };
            let down_id = WeightId::Expert {
                layer,
                expert: idx,
                role: ExpertRole::Down,
            };
            let gate_up_tensor = self
                .resident
                .get(&gate_up_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {gate_up_id:?} not resident"));
            let down_tensor = self
                .resident
                .get(&down_id)
                .unwrap_or_else(|| panic!("patch_expert_ptr_table: {down_id:?} not resident"));
            // u64 pointer values, written into the device table at expert idx's slot.
            let gate_up_ptr = gate_up_tensor.tensor.buf.as_ptr() as u64;
            let down_ptr = down_tensor.tensor.buf.as_ptr() as u64;
            let offset = (idx as usize) * 8;
            gpu.hip
                .memcpy_htod_offset(&gate_up_ptrs.buf, offset, &gate_up_ptr.to_le_bytes())?;
            gpu.hip
                .memcpy_htod_offset(&down_ptrs.buf, offset, &down_ptr.to_le_bytes())?;
        }
        Ok(())
    }

    /// Evict residents from the LRU front (least-recently-used) until at
    /// least `need_bytes` would fit under the budget. Returns
    /// [`WeightPagerError::BudgetExhausted`] if nothing more can be evicted
    /// but space is still insufficient, OR if `need_bytes > budget` and
    /// the budget is finite (no amount of eviction can fit the requested
    /// weight in that case, and the prior implementation silently drained
    /// the residency map and let the caller violate the cap anyway).
    ///
    /// Frees evicted tensors via `gpu.free_tensor` — the underlying VRAM
    /// returns to the rdna-compute allocator pool, available for the next
    /// `transport.fetch`.
    pub fn evict_lru_until(
        &mut self,
        need_bytes: u64,
        gpu: &mut Gpu,
    ) -> Result<(), WeightPagerError> {
        // Reject up front when the requested weight is alone bigger than
        // the budget. Without this guard, `target_used` saturates to 0 and
        // the loop drains the residency map without erroring.
        self.would_fit(need_bytes)?;
        let budget = self.config.vram_budget_bytes;
        // How much we need to free so that vram_used + need <= budget.
        let target_used = budget.saturating_sub(need_bytes);
        while self.vram_used_bytes > target_used {
            let id = self
                .lru
                .pop_front()
                .ok_or(WeightPagerError::BudgetExhausted {
                    need_bytes,
                    in_use: self.vram_used_bytes,
                    budget,
                })?;
            if let Some(r) = self.resident.remove(&id) {
                self.vram_used_bytes = self.vram_used_bytes.saturating_sub(r.bytes);
                let _ = gpu.free_tensor(r.tensor);
                if self.config.trace {
                    eprintln!(
                        "[weight_pager] evict {id:?} ({} bytes) — {} resident, {} used",
                        r.bytes,
                        self.resident.len(),
                        self.vram_used_bytes
                    );
                }
            } else {
                // LRU and residency map are out of sync — caller pushed onto
                // LRU without inserting into `resident`, or removed without
                // updating LRU. In release this is a silent drop; debug
                // builds catch the invariant violation.
                debug_assert!(
                    false,
                    "weight_pager: LRU contained {id:?} but residency map did not"
                );
            }
        }
        Ok(())
    }

    /// Free all resident tensors back to the GPU pool. Called on model
    /// teardown so VRAM goes back to the system. After this, the pager is
    /// effectively reset (catalog stays, residency map is empty).
    pub fn free_all(&mut self, gpu: &mut Gpu) {
        for (_id, r) in self.resident.drain() {
            let _ = gpu.free_tensor(r.tensor);
        }
        self.lru.clear();
        self.vram_used_bytes = 0;
    }

    /// Get the resident tensor for `id`. Returns `None` if not resident
    /// (caller should `ensure_resident` first). Does not affect LRU.
    pub fn get(&self, id: WeightId) -> Option<&GpuTensor> {
        self.resident.get(&id).map(|r| &r.tensor)
    }

    /// Insert an already-resident weight. Used by the loader for
    /// always-resident weights (token embeds, norms, the router itself in
    /// v0.1) — they live in VRAM from startup but the pager tracks them so
    /// they're visible to `get()` and accounted in `vram_used_bytes`.
    pub fn insert_resident(&mut self, id: WeightId, tensor: GpuTensor, bytes: u64) {
        if let Some(prev) = self.resident.remove(&id) {
            self.vram_used_bytes = self.vram_used_bytes.saturating_sub(prev.bytes);
            self.lru.retain(|x| *x != id);
        }
        self.resident.insert(id, Resident { tensor, bytes });
        self.lru.push_back(id);
        self.vram_used_bytes = self.vram_used_bytes.saturating_add(bytes);
    }

    /// Mark `id` as recently used. No-op if not resident.
    pub fn touch_lru(&mut self, id: WeightId) {
        if let Some(pos) = self.lru.iter().position(|x| *x == id) {
            self.lru.remove(pos);
            self.lru.push_back(id);
        }
    }

    /// Bytes currently held resident. Cheap (cached, not a sum).
    pub fn vram_used_bytes(&self) -> u64 {
        self.vram_used_bytes
    }

    /// Number of currently-resident weights.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Pre-flight budget check. Returns Err if a fetch of `need` bytes
    /// could not fit even with full eviction. Used by `ensure_resident`
    /// and `evict_lru_until`; exposed so unit tests can exercise the
    /// invariant without constructing a real Gpu.
    pub fn would_fit(&self, need: u64) -> Result<(), WeightPagerError> {
        let budget = self.config.vram_budget_bytes;
        if budget != u64::MAX && need > budget {
            return Err(WeightPagerError::BudgetExhausted {
                need_bytes: need,
                in_use: self.vram_used_bytes,
                budget,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum WeightPagerError {
    /// Weight wasn't registered with the pager. Loader bug.
    NotRegistered(WeightId),
    /// Hipfire HIP error (transfer / alloc failed).
    Hip(hip_bridge::HipError),
    /// Eviction couldn't free enough room — budget too small for the
    /// requested weight. User needs to raise `vram_budget_bytes` or
    /// reduce the working set somehow.
    BudgetExhausted {
        need_bytes: u64,
        in_use: u64,
        budget: u64,
    },
    /// Stub for paths still being filled in.
    Unimplemented(&'static str),
}

impl std::fmt::Display for WeightPagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRegistered(id) => write!(f, "weight not registered: {id:?}"),
            Self::Hip(e) => write!(f, "hip error: {e}"),
            Self::BudgetExhausted {
                need_bytes,
                in_use,
                budget,
            } => write!(
                f,
                "weight pager: cannot evict to fit {need_bytes} bytes \
                 (in_use={in_use}, budget={budget}); raise vram_budget_bytes \
                 or reduce paged working set"
            ),
            Self::Unimplemented(why) => write!(f, "weight pager: unimplemented ({why})"),
        }
    }
}

impl std::error::Error for WeightPagerError {}

impl From<hip_bridge::HipError> for WeightPagerError {
    fn from(e: hip_bridge::HipError) -> Self {
        Self::Hip(e)
    }
}

// ---------------------------------------------------------------------------
// Convenience: open an HfqFile by path. The loader uses the existing
// HfqFile::open directly; this re-export keeps the module's surface minimal.
// ---------------------------------------------------------------------------

/// Forwarding helper so callers don't need a separate `use crate::hfq::HfqFile`.
pub fn open_hfq(path: &Path) -> std::io::Result<HfqFile> {
    HfqFile::open(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn slot_cache_clear_restores_slot_order_without_reusing_ticket_identity() {
        let mut cache = SlotCache::new(2).unwrap();
        let first = cache.reserve((), 11usize).unwrap();
        assert!(cache.commit(first));
        let second = cache.reserve((), 22usize).unwrap();
        assert!(cache.commit(second));
        cache.reserve((), 11).unwrap(); // LRU is now slot 1, then slot 0.
        let old_pending = cache.reserve((), 33).unwrap();
        assert_eq!(old_pending.slot, 1);
        let stats = cache.stats();
        // Stand in for the transport's proven DMA retirement before clearing.
        cache.clear();
        assert_eq!(cache.stats(), stats);
        let reset = cache.reserve((), 33).unwrap();
        assert_eq!(reset.slot, 0);
        assert!(reset.generation() > old_pending.generation());
        assert!(!cache.commit(old_pending));
        assert!(!cache.rollback(old_pending));
        assert!(cache.commit(reset));
        assert_eq!(cache.reserve((), 44).unwrap().slot, 1);
    }

    #[test]
    fn slot_cache_generation_rejects_aba_commit_and_rollback() {
        let mut cache = SlotCache::new(1).unwrap();
        let old = cache.reserve(0u8, 7u16).unwrap();
        assert!(cache.invalidate(7));
        let intervening = cache.reserve(0, 8).unwrap();
        assert!(cache.rollback(intervening));
        let current = cache.reserve(0, 7).unwrap();
        assert_eq!(old.slot, current.slot);
        assert!(old.generation() < intervening.generation());
        assert!(intervening.generation() < current.generation());
        assert!(!cache.commit(old));
        assert!(!cache.rollback(old));
        assert_eq!(cache.resident_slot(7), None);
        assert!(cache.commit(current));
        assert!(!cache.rollback(current));
        assert!(!cache.commit(current));
        assert_eq!(cache.resident_slot(7), Some(current.slot));
    }

    #[test]
    fn slot_cache_reverse_completion_preserves_logical_lru() {
        let mut cache = SlotCache::new(3).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        let b = cache.reserve(0, 2).unwrap();
        let c = cache.reserve(0, 3).unwrap();
        assert!(cache.commit(c));
        assert!(cache.commit(b));
        assert!(cache.commit(a));
        let d = cache.reserve(0, 4).unwrap();
        assert_eq!(d.victim(), Some(1));
        assert_eq!(d.slot, a.slot);
    }

    #[test]
    fn slot_cache_pending_duplicate_touches_without_claiming_fill() {
        let mut cache = SlotCache::new(2).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        let b = cache.reserve(0, 2).unwrap();
        let duplicate = cache.reserve(0, 1).unwrap();
        assert_eq!(duplicate.generation(), a.generation());
        assert!(!duplicate.needs_fill());
        assert_eq!(duplicate.victim(), None);
        assert!(!cache.commit(duplicate));
        assert!(!cache.rollback(duplicate));
        assert!(cache.commit(a));
        assert!(cache.commit(b));
        assert_eq!(cache.reserve(0, 3).unwrap().victim(), Some(2));
        assert_eq!(cache.stats(), (0, 3));
    }

    #[test]
    fn slot_cache_partial_rollback_reuses_hole_without_evicting_or_resurrecting() {
        let mut cache = SlotCache::new(2).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        let b = cache.reserve(0, 2).unwrap();
        assert!(cache.commit(a));
        assert!(cache.rollback(b));
        let c = cache.reserve(0, 3).unwrap();
        assert_eq!(c.slot, b.slot);
        assert_eq!(c.victim(), None);
        assert!(cache.commit(c));
        let d = cache.reserve(0, 4).unwrap();
        assert_eq!(d.victim(), Some(1));
        assert!(cache.rollback(d));
        assert_eq!(cache.resident_slot(1), None);
        assert_eq!(cache.resident_slot(3), Some(c.slot));
        assert!(!cache.commit(b));
        assert!(!cache.rollback(b));
    }

    #[test]
    fn slot_cache_all_pending_refusal_preserves_access_order() {
        let mut cache = SlotCache::new(2).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        let b = cache.reserve(0, 2).unwrap();
        assert_eq!(
            cache.reserve(0, 3).unwrap_err(),
            SlotCacheError::AllSlotsPending
        );
        assert_eq!(cache.stats(), (0, 2));
        assert!(cache.commit(b));
        // Skip the older pending A without moving it in the recency queue.
        let c = cache.reserve(0, 3).unwrap();
        assert_eq!(c.victim(), Some(2));
        assert!(cache.commit(c));
        assert!(cache.commit(a));
        assert_eq!(cache.reserve(0, 4).unwrap().victim(), Some(1));
    }

    #[test]
    fn slot_cache_victim_identity_and_pools_are_independent() {
        let mut cache = SlotCache::new(1).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        let b = cache.reserve(1, 2).unwrap();
        assert_eq!((a.slot, b.slot), (0, 0));
        assert!(cache.commit(a));
        assert!(cache.commit(b));
        assert_eq!(
            cache.reserve(1, 1).unwrap_err(),
            SlotCacheError::PoolMismatch
        );
        let hit = cache.reserve(0, 1).unwrap();
        assert!(!hit.needs_fill());
        assert_eq!(hit.victim(), None);
        let c = cache.reserve(0, 3).unwrap();
        assert_eq!(c.victim(), Some(1));
        assert_eq!(cache.resident_slot(1), None);
        assert_eq!(cache.resident_slot(2), Some(0));
        assert_eq!(cache.stats(), (1, 3));
    }

    #[test]
    fn slot_cache_generation_exhaustion_fails_without_eviction() {
        assert!(matches!(
            SlotCache::<u16, u8>::new(0),
            Err(SlotCacheError::ZeroSlots)
        ));
        let mut cache = SlotCache::new(1).unwrap();
        let a = cache.reserve(0u8, 1u16).unwrap();
        assert!(cache.commit(a));
        cache.generation = u64::MAX;
        assert_eq!(
            cache.reserve(0, 2).unwrap_err(),
            SlotCacheError::GenerationExhausted
        );
        assert_eq!(cache.resident_slot(1), Some(a.slot));
        assert_eq!(cache.stats(), (0, 1));
        // Existing accesses need no generation and still succeed.
        assert!(!cache.reserve(0, 1).unwrap().needs_fill());
    }

    #[test]
    fn weight_id_is_hashable() {
        let mut map = HashMap::new();
        let a = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::GateUp,
        };
        let b = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::Down,
        };
        map.insert(a, 1u32);
        map.insert(b, 2u32);
        assert_eq!(map.get(&a), Some(&1));
        assert_eq!(map.get(&b), Some(&2));
    }

    /// Write some bytes to a temp file and verify `PreadH2DTransport::open`
    /// can read arbitrary ranges via the staging buffer. The actual upload
    /// to GPU is exercised in integration tests; here we directly call the
    /// pread helper to keep the unit test device-free.
    #[test]
    fn pread_transport_reads_range() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-test-{}.bin", std::process::id()));
        let payload: Vec<u8> = (0..1024u32).flat_map(|i| (i as u8).to_le_bytes()).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let mut t = PreadH2DTransport::open(&path).unwrap();
        // Read [256..768) — should match payload[256..768].
        t.pread_into_staging(256, 512).unwrap();
        assert_eq!(&t.staging[..512], &payload[256..768]);
        // Read a smaller range; staging must cover it.
        t.pread_into_staging(0, 16).unwrap();
        assert_eq!(&t.staging[..16], &payload[..16]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pager_starts_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-empty-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        assert_eq!(pager.registered_count(), 0);
        assert_eq!(pager.vram_used_bytes(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn register_then_get_returns_none_until_resident() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-reg-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let mut pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        let id = WeightId::Expert {
            layer: 0,
            expert: 0,
            role: ExpertRole::GateUp,
        };
        pager.register(id, ByteRange { offset: 0, len: 1 });
        assert_eq!(pager.registered_count(), 1);
        // Catalog hit, not yet resident → get returns None.
        assert!(!pager.is_resident(id));
        assert!(pager.get(id).is_none());
        // ensure_resident requires a real Gpu — exercised in integration tests.
        let _ = std::fs::remove_file(&path);
    }

    /// Regression: a need bigger than the entire budget must NOT cause
    /// `evict_lru_until` to silently drain the residency map and return Ok.
    /// Prior to this guard, `target_used = budget.saturating_sub(need)` was 0
    /// and the loop exited cleanly even though the subsequent fetch in
    /// `ensure_resident` would push usage past the cap.
    #[test]
    fn would_fit_rejects_need_bigger_than_budget() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-budget-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(
            &path,
            PagerConfig {
                vram_budget_bytes: 100,
                trace: false,
            },
        )
        .unwrap();
        // need <= budget → ok
        assert!(pager.would_fit(50).is_ok());
        assert!(pager.would_fit(100).is_ok());
        // need > budget → BudgetExhausted, even on an empty pager
        match pager.would_fit(1000) {
            Err(WeightPagerError::BudgetExhausted {
                need_bytes,
                in_use,
                budget,
            }) => {
                assert_eq!(need_bytes, 1000);
                assert_eq!(in_use, 0);
                assert_eq!(budget, 100);
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Sanity: with the unlimited budget (default), would_fit accepts
    /// arbitrary sizes — the cap is a no-op in that mode.
    #[test]
    fn would_fit_accepts_anything_when_budget_unlimited() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hipfire-pager-unlim-{}.bin", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let pager = WeightPager::with_pread_transport(&path, PagerConfig::default()).unwrap();
        // Default is u64::MAX; even u64::MAX - 1 fits.
        assert!(pager.would_fit(u64::MAX - 1).is_ok());
        let _ = std::fs::remove_file(&path);
    }
}
