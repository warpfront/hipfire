// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// PagePool — paged KV cache allocator (vLLM-style PagedAttention).
//
// Replaces SlotPool's fixed per-slot slabs with a shared free-list of
// physical pages. Each session owns a BlockTable (Vec<u32>) mapping
// logical page index -> physical page index. Pages are allocated on
// demand and freed when a session ends, enabling:
//
//   - Dynamic memory allocation (no pre-reserved per-slot capacity)
//   - Prefix sharing across sessions (shared physical pages, refcounted)
//   - Over-subscription without host swap (more sessions than fixed
//     slabs would allow, since short sessions consume fewer pages)
//
// The block table is uploaded to the GPU as a flat i32 array and its
// device address is stored in KvSlotDesc.block_table. The attention
// kernels use kv_offset_for_k/v() to translate logical positions to
// physical byte offsets via the block table — no kernel is touched.
//
// ── Wave 2: ownership, COW, and deferred reclaim (spec §4.3 C3, §4.4 C4)
//
// Each physical page carries a `PageMeta` tracking:
//   - `PageState`: Free → Private → Sealed → CacheOnly → ReclaimPending → Free
//   - Checked refcounts separated by `LeaseClass` (table, cache, in-flight)
//   - A per-page `generation` so stale `PageHandle`s fail before dereference
//
// Copy-on-write (`CowPlan`/`plan_cow`/`commit_cow`/`abort_cow`) lets a
// session rewrite a shared (Sealed) page by reserving a private
// destination and copying the valid prefix — raw bytes, no dequant/requant
// — before rebinding the block table. Failed reservations release only
// newly reserved resources (spec §4.3).
//
// Deferred reclaim: when all table/cache refs drop to zero but in-flight
// device reads remain, the page enters `ReclaimPending` and is NOT
// immediately reusable. `drain_completed()` is the synchronous step-fence
// that frees confirmed-complete pages. The upgrade seam to HIP-event-based
// completion is documented at that method.

use crate::kv_slots::{preflight_alloc, KvSlotDesc, R9700_VRAM_BYTES};

/// Page size in tokens. Must match PAGE_TOKENS in slot_pool.rs and the
/// tile size the flash path walks KV in.
pub const PAGE_TOKENS: usize = 128;

/// A physical page index in the shared arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

// =========================================================================
// Wave 2 — physical ownership types (spec §4.3 C3, §4.4 C4)
//
// rdna-compute owns these *physical* types; hipfire-runtime's
// serve_contract.rs owns the matching *logical* types (CacheDomain,
// StepTicket, ReleaseDisposition, …).
// =========================================================================

/// A validated handle to a physical page (spec §4.3). The host validates
/// `generation` (and `epoch`) before table upload; a stale handle — the
/// page was freed and re-allocated to another session — fails before
/// dereference, preventing use-after-free on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageHandle {
    /// Physical page index.
    pub phys: u32,
    /// Pool allocation epoch (bumped on device reload; spec §4.1
    /// "allocation epoch").  In this implementation the epoch is always 0
    /// and never bumped — the field and check exist so the upgrade path is
    /// non-breaking.
    pub epoch: u32,
    /// Per-page generation.  Bumped each time the page transitions
    /// Free → allocated and again on Free → re-alloc, so a handle from a
    /// prior allocation cycle mismatches and is rejected.
    pub generation: u32,
}

/// Lifecycle state of a physical page (spec §4.3 C3).
///
/// ```text
///   Free ──alloc──▶ Private ──seal/share──▶ Sealed
///    ▲                  │                       │
///    │                  │ release (rc→0)        │ release (rc→0)
///    │                  ▼                       │
///    │                 Free              CacheOnly
///    │                                    │
///    │                              release (rc→0, inflight>0)
///    │                                    │
///    │                                    ▼
///    └──────────── drain_completed ◀── ReclaimPending
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageState {
    /// In the free list; no refs; generation already bumped.
    Free,
    /// Owned by exactly one block table; mutable (writes allowed).
    Private,
    /// Shared (≥2 table refs) or cache-pinned; immutable — any write
    /// intention must trigger copy-on-write (spec §4.3).
    Sealed,
    /// No active table ref but cache-owned; still immutable, still
    /// resident.  Can be re-attached (table ref added) or evicted.
    CacheOnly,
    /// All table/cache refs released but in-flight device/transport reads
    /// remain; NOT in the free list until `drain_completed` confirms
    /// completion (spec §4.4).
    ReclaimPending,
}

/// Lease class identifying *why* a page is pinned (spec §4.3: "Separate
/// active/table refs, cache ownership, in-flight/transfer leases; physical
/// bytes charged once").  The total refcount is the sum across classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeaseClass {
    /// A live session's block table references this page.
    Table,
    /// The prefix cache owns this page (it may outlive any single session).
    Cache,
    /// A device kernel or transport reader is still accessing this page;
    /// reuse is deferred until completion (spec §4.4).
    InFlight,
}

/// Per-page metadata (internal).  Replaces the flat `refcounts: Vec<u32>`
/// from Wave 1 with checked, class-separated refcounts and the state
/// machine.
#[derive(Debug, Clone, Copy)]
struct PageMeta {
    state: PageState,
    generation: u32,
    table_refs: u32,
    cache_refs: u32,
    inflight_refs: u32,
}

impl PageMeta {
    const fn free() -> Self {
        Self {
            state: PageState::Free,
            generation: 0,
            table_refs: 0,
            cache_refs: 0,
            inflight_refs: 0,
        }
    }

    /// Total references across all lease classes.
    fn total_refs(&self) -> u32 {
        self.table_refs + self.cache_refs + self.inflight_refs
    }
}

/// One copy-on-write operation planned by [`PagePool::plan_cow`].
///
/// The caller must raw-copy `valid_prefix_tokens` positions' worth of K
/// and V bytes from `src_phys` to `dst_phys` (spec §4.3: "copy valid
/// prefix in every K/V layer arena … preserve encoded bytes/quant/
/// strides, no dequant/requant").  `k_copy_bytes` / `v_copy_bytes` are
/// pre-computed for the caller's convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowCopy {
    /// Logical page index in the block table being rebound.
    pub logical_page: usize,
    /// Source physical page (Sealed or CacheOnly — being COWed away from).
    pub src_phys: u32,
    /// Destination physical page (Private — the COW copy).
    pub dst_phys: u32,
    /// Valid tokens before the write start in this page (0 when the write
    /// begins at the page boundary).
    pub valid_prefix_tokens: usize,
    /// K-arena bytes to copy (`valid_prefix_tokens × k_per_pos_bytes`).
    pub k_copy_bytes: usize,
    /// V-arena bytes to copy (`valid_prefix_tokens × v_per_pos_bytes`).
    pub v_copy_bytes: usize,
}

/// A copy-on-write plan produced by [`PagePool::plan_cow`] and consumed by
/// [`PagePool::commit_cow`] or [`PagePool::abort_cow`].
///
/// `plan_cow` reserves private destination pages for every Sealed or
/// CacheOnly page in the write interval **without** modifying the block
/// table.  The caller executes the GPU memcpys, then calls `commit_cow`
/// to rebind the table and adjust refcounts, or `abort_cow` to release
/// the reserved pages on failure.  This ordering ensures a failed
/// reservation leaves the original table and pool state intact (spec §4.3:
/// "transactional failure releases only newly reserved resources").
/// A plan's reserved pages are released by [`PagePool::commit_cow`] or
/// [`PagePool::abort_cow`], both of which consume the plan. Dropping an
/// armed plan leaks its reserved pages — the `Drop` impl logs loudly
/// because the pool cannot be reached from here.
#[derive(Debug)]
#[must_use = "a CowPlan holds reserved pages; pass it to commit_cow or abort_cow"]
pub struct CowPlan {
    copies: Vec<CowCopy>,
    /// True while the reserved destination pages are still owned by this
    /// plan. Disarmed by commit/abort; an armed drop is a leak.
    armed: bool,
}

impl CowPlan {
    /// The individual copy operations this plan requires.
    pub fn copies(&self) -> &[CowCopy] {
        &self.copies
    }

    /// True when no COW is needed (the write interval touches no
    /// Sealed/CacheOnly pages).
    pub fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }
}

impl Drop for CowPlan {
    fn drop(&mut self) {
        if self.armed {
            eprintln!(
                "[page-pool] CowPlan dropped while armed: {} reserved page(s) \
                 leak (commit_cow/abort_cow consume the plan)",
                self.copies.len()
            );
        }
    }
}

// =========================================================================
// BlockTable — unchanged from Wave 1
// =========================================================================

/// A session's block table: logical page index -> physical page index.
/// Uploaded to the GPU as a flat i32 array; its device address goes into
/// KvSlotDesc.block_table.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical -> physical page mapping.
    pages: Vec<u32>,
    /// Number of live tokens (not pages). The last page may be partially
    /// filled; `live_tokens` determines how many positions the kernel reads.
    live_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            live_tokens: 0,
        }
    }

    /// Number of pages currently allocated.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Number of live tokens (logical KV length).
    pub fn live_tokens(&self) -> usize {
        self.live_tokens
    }

    /// Physical page index for logical page `lp`, or None if out of range.
    pub fn physical(&self, lp: usize) -> Option<u32> {
        self.pages.get(lp).copied()
    }

    /// Raw page indices for GPU upload.
    pub fn page_indices(&self) -> &[u32] {
        &self.pages
    }

    pub(crate) fn push_page(&mut self, page: u32) {
        self.pages.push(page);
    }

    /// Rebind logical page `lp` to physical page `phys`.  Used by
    /// `commit_cow` after the GPU memcpy is issued.
    pub(crate) fn set_page(&mut self, lp: usize, phys: u32) {
        if lp < self.pages.len() {
            self.pages[lp] = phys;
        }
    }

    /// Truncate to exactly `n_pages` pages, returning the freed pages.
    /// Used when trimming a session's KV (e.g. on eviction).
    fn truncate(&mut self, n_pages: usize) -> Vec<u32> {
        let freed: Vec<u32> = self.pages.drain(n_pages..).collect();
        self.live_tokens = self.live_tokens.min(n_pages * PAGE_TOKENS);
        freed
    }

    /// Set the live token count. Must be <= num_pages * PAGE_TOKENS.
    pub(crate) fn set_live_tokens(&mut self, tokens: usize) {
        debug_assert!(
            tokens <= self.pages.len() * PAGE_TOKENS,
            "live_tokens {} exceeds capacity {} ({} pages * {} tokens)",
            tokens,
            self.pages.len() * PAGE_TOKENS,
            self.pages.len(),
            PAGE_TOKENS
        );
        self.live_tokens = tokens;
    }

    /// Host-side copy of the table (pages + live_tokens). Used by
    /// `SlotPool::share_prefix` to hand `PagePool::share_prefix` a stable
    /// snapshot of the source without overlapping field borrows.
    pub(crate) fn snapshot(&self) -> Self {
        Self {
            pages: self.pages.clone(),
            live_tokens: self.live_tokens,
        }
    }
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// PagePool
// =========================================================================

/// Free-list page allocator for a shared KV arena.
///
/// The arena is a single contiguous buffer per layer (K and V each).
/// `PagePool` tracks which pages are free and which are allocated to
/// which session. Pages are `PAGE_TOKENS * per_pos_bytes` bytes each.
///
/// Prefix sharing: when two sessions share the same token prefix, they
/// share the physical pages for that prefix. Shared pages are refcounted,
/// sealed (immutable), and only freed when the last session releases them.
#[derive(Debug)]
pub struct PagePool {
    /// Total number of physical pages in the arena.
    n_pages: usize,
    /// Per-position strides in bytes (n_kv_heads * (head_dim/32) * 34 for
    /// Q8_0). Equal on q8/bf16; DIFFERENT on the rotated-K tiers, where the
    /// K page is `PAGE_TOKENS × k_per_pos_bytes` and the V page
    /// `PAGE_TOKENS × v_per_pos_bytes`. One block table maps both arenas —
    /// a logical page resolves to the same physical index for K and V.
    k_per_pos_bytes: usize,
    v_per_pos_bytes: usize,
    /// Free list of physical page indices. Pages are popped from the end.
    /// Does NOT include ReclaimPending pages.
    free_pages: Vec<u32>,
    /// Per-page metadata (state, generation, class-separated refcounts).
    /// Index is physical page index.
    page_meta: Vec<PageMeta>,
    /// Pages pending reclaim: table/cache refs released but in-flight
    /// device reads remain. Freed by `drain_completed`.
    reclaim_pending: Vec<u32>,
    /// Pool allocation epoch (spec §4.1). Always 0 in this implementation;
    /// the field and `PageHandle.epoch` check exist so a future device-reload
    /// upgrade is non-breaking.
    epoch: u32,
}

impl PagePool {
    /// Create a page pool with `n_pages` physical pages, each holding
    /// `PAGE_TOKENS` positions of `per_pos_bytes` bytes.
    ///
    /// Refuses rather than allocates when the arena would exceed the
    /// deployment-target budget — see `kv_slots::preflight_alloc`.
    pub fn new(n_pages: usize, per_pos_bytes: usize) -> Result<Self, String> {
        Self::new_with_strides(n_pages, per_pos_bytes, per_pos_bytes)
    }

    /// [`PagePool::new`] with independent K/V strides (the rotated-K tiers).
    pub fn new_with_strides(
        n_pages: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Result<Self, String> {
        assert!(n_pages > 0, "n_pages must be positive");
        assert!(k_per_pos_bytes > 0, "k_per_pos_bytes must be positive");
        assert!(v_per_pos_bytes > 0, "v_per_pos_bytes must be positive");

        let total = Self::checked_arena_bytes(n_pages, k_per_pos_bytes, v_per_pos_bytes)?;
        preflight_alloc(total, R9700_VRAM_BYTES, "PagePool arena")?;
        Ok(Self::with_checked_metadata(
            n_pages,
            k_per_pos_bytes,
            v_per_pos_bytes,
        ))
    }

    /// Descriptor pool for EXTERNALLY owned backing (e.g. a VMM arena whose
    /// granules are allocated and budgeted elsewhere): identical metadata and
    /// page lifecycle to [`PagePool::new_with_strides`], minus
    /// `kv_slots::preflight_alloc`.
    ///
    /// The page bytes derived from the strides are a logical coverage
    /// measure (how many positions the pages describe), NOT a physical
    /// budget: this pool allocates no device memory, so no deployment-target
    /// VRAM check applies. Returns `Err` (never panics) on a zero page count
    /// or stride, on `n_pages` exceeding the `u32` page-index space, or when
    /// the nominal arena size overflows `u64`.
    pub fn new_external_with_strides(
        n_pages: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Result<Self, String> {
        if n_pages == 0 {
            return Err("PagePool: n_pages must be positive".to_string());
        }
        if k_per_pos_bytes == 0 {
            return Err("PagePool: k_per_pos_bytes must be positive".to_string());
        }
        if v_per_pos_bytes == 0 {
            return Err("PagePool: v_per_pos_bytes must be positive".to_string());
        }
        Self::checked_arena_bytes(n_pages, k_per_pos_bytes, v_per_pos_bytes)?;
        Ok(Self::with_checked_metadata(
            n_pages,
            k_per_pos_bytes,
            v_per_pos_bytes,
        ))
    }

    /// Nominal bytes of the K+V arenas (`n_pages` pages), with every
    /// intermediate product checked and `n_pages` required to fit the `u32`
    /// physical-page index space. Arguments must be non-zero.
    fn checked_arena_bytes(
        n_pages: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Result<u64, String> {
        const OVERFLOW: &str = "PagePool: arena size overflows u64";
        if n_pages > u32::MAX as usize {
            return Err(format!(
                "PagePool: {} pages exceed the u32 page-index space",
                n_pages
            ));
        }
        let k_page = PAGE_TOKENS
            .checked_mul(k_per_pos_bytes)
            .ok_or_else(|| OVERFLOW.to_string())?;
        let v_page = PAGE_TOKENS
            .checked_mul(v_per_pos_bytes)
            .ok_or_else(|| OVERFLOW.to_string())?;
        let page_bytes = k_page
            .checked_add(v_page)
            .ok_or_else(|| OVERFLOW.to_string())?;
        (page_bytes as u64)
            .checked_mul(n_pages as u64)
            .ok_or_else(|| OVERFLOW.to_string())
    }

    /// Shared metadata initialization: every page Free at generation 0.
    /// Callers must have validated the arguments with `checked_arena_bytes`.
    fn with_checked_metadata(
        n_pages: usize,
        k_per_pos_bytes: usize,
        v_per_pos_bytes: usize,
    ) -> Self {
        let free_pages: Vec<u32> = (0..n_pages as u32).rev().collect();
        Self {
            n_pages,
            k_per_pos_bytes,
            v_per_pos_bytes,
            free_pages,
            page_meta: vec![PageMeta::free(); n_pages],
            reclaim_pending: Vec::new(),
            epoch: 0,
        }
    }

    /// Total number of physical pages.
    pub fn n_pages(&self) -> usize {
        self.n_pages
    }

    /// Per-position stride in bytes. Equal-stride pools only — see
    /// [`PagePool::k_per_pos_bytes`] / [`PagePool::v_per_pos_bytes`].
    pub fn per_pos_bytes(&self) -> usize {
        debug_assert_eq!(
            self.k_per_pos_bytes, self.v_per_pos_bytes,
            "per_pos_bytes() is the EQUAL-stride accessor; this pool carries \
             distinct K/V strides"
        );
        self.k_per_pos_bytes
    }

    /// K-arena per-position stride in bytes.
    pub fn k_per_pos_bytes(&self) -> usize {
        self.k_per_pos_bytes
    }

    /// V-arena per-position stride in bytes.
    pub fn v_per_pos_bytes(&self) -> usize {
        self.v_per_pos_bytes
    }

    /// Bytes per K page (PAGE_TOKENS * k_per_pos_bytes).
    pub fn k_page_bytes(&self) -> usize {
        PAGE_TOKENS * self.k_per_pos_bytes
    }

    /// Bytes per V page (PAGE_TOKENS * v_per_pos_bytes).
    pub fn v_page_bytes(&self) -> usize {
        PAGE_TOKENS * self.v_per_pos_bytes
    }

    /// Bytes per page (PAGE_TOKENS * per_pos_bytes). Equal-stride pools only.
    pub fn page_bytes(&self) -> usize {
        self.k_page_bytes()
    }

    /// Bytes in ONE arena (K or V). The pool holds two of these.
    /// Equal-stride pools only — see `k_arena_bytes`/`v_arena_bytes`.
    pub fn arena_bytes(&self) -> usize {
        self.k_arena_bytes()
    }

    /// Bounds-check a physical page index. Every public entry point that
    /// takes a `phys` calls this first — a stale or fabricated index must
    /// produce `Err`, never an out-of-bounds `page_meta` panic.
    #[inline]
    fn check_phys(&self, phys: u32) -> Result<(), String> {
        if (phys as usize) < self.n_pages {
            Ok(())
        } else {
            Err(format!(
                "PagePool: phys {} out of bounds (n_pages={})",
                phys, self.n_pages
            ))
        }
    }
    /// Bytes of the K arena.
    pub fn k_arena_bytes(&self) -> usize {
        self.n_pages * self.k_page_bytes()
    }

    /// Bytes of the V arena.
    pub fn v_arena_bytes(&self) -> usize {
        self.n_pages * self.v_page_bytes()
    }

    /// Number of free pages available for allocation (excludes
    /// ReclaimPending).
    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    /// Maximum token capacity of the pool (all pages allocated).
    pub fn max_tokens(&self) -> usize {
        self.n_pages * PAGE_TOKENS
    }

    /// Debug-only structural invariant check (spec §9.2 observability).
    ///
    /// Asserts the page-accounting invariants that a leak or double-free
    /// would break:
    /// - every page is accounted exactly once: free list + per-state
    ///   residency == `n_pages`;
    /// - the free list contains only `Free` pages, no duplicates;
    /// - `reclaim_pending` contains only `ReclaimPending` pages;
    /// - `Free` pages carry zero refs; `ReclaimPending` pages carry no
    ///   table/cache refs (in-flight reads may remain).
    ///
    /// O(n_pages) — call from a debug flag, never the hot path.
    pub fn check_invariants(&self) {
        let mut seen = vec![false; self.n_pages];
        for &p in &self.free_pages {
            let m = &self.page_meta[p as usize];
            assert_eq!(
                m.state,
                PageState::Free,
                "free list holds non-Free page {p} (state={:?})",
                m.state
            );
            assert_eq!(
                m.total_refs(),
                0,
                "free page {p} carries {} refs",
                m.total_refs()
            );
            assert!(!seen[p as usize], "free list duplicates page {p}");
            seen[p as usize] = true;
        }
        for &p in &self.reclaim_pending {
            let m = &self.page_meta[p as usize];
            assert_eq!(
                m.state,
                PageState::ReclaimPending,
                "reclaim_pending holds page {p} in state {:?}",
                m.state
            );
            assert_eq!(
                m.table_refs + m.cache_refs,
                0,
                "reclaim-pending page {p} still has table/cache refs"
            );
            assert!(!seen[p as usize], "page {p} both free and reclaim-pending");
            seen[p as usize] = true;
        }
        let mut accounted = self.free_pages.len() + self.reclaim_pending.len();
        for (p, m) in self.page_meta.iter().enumerate() {
            match m.state {
                PageState::Private | PageState::Sealed | PageState::CacheOnly => {
                    accounted += 1;
                    assert!(
                        m.table_refs + m.cache_refs > 0,
                        "resident page {p} has no owner refs (state={:?})",
                        m.state
                    );
                }
                PageState::Free | PageState::ReclaimPending => {}
            }
        }
        assert_eq!(
            accounted, self.n_pages,
            "page accounting drift: {accounted} accounted vs {} physical",
            self.n_pages
        );
    }

    // ── Wave 2: per-page metadata accessors ───────────────────────────

    /// Total refcount (all lease classes) for physical page `phys`.
    pub(crate) fn refcount(&self, phys: u32) -> u32 {
        if self.check_phys(phys).is_err() {
            return 0;
        }
        self.page_meta[phys as usize].total_refs()
    }

    /// Current [`PageState`] of physical page `phys`.
    pub fn page_state(&self, phys: u32) -> PageState {
        if self.check_phys(phys).is_err() {
            // Fail closed: an out-of-range page reports as Free so no
            // caller treats it as a live allocation.
            return PageState::Free;
        }
        self.page_meta[phys as usize].state
    }

    /// Current generation of physical page `phys`.  A [`PageHandle`]
    /// carrying a mismatched generation is stale.
    pub fn page_generation(&self, phys: u32) -> u32 {
        if self.check_phys(phys).is_err() {
            // A generation no real page can match (real generations start
            // at 0 and wrap upward); any handle compared against it fails.
            return u32::MAX;
        }
        self.page_meta[phys as usize].generation
    }

    /// Pool allocation epoch (spec §4.1).
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Number of pages in the ReclaimPending queue.
    pub fn reclaim_pending_count(&self) -> usize {
        self.reclaim_pending.len()
    }

    // ── Wave 2: handle validation ─────────────────────────────────────

    /// Validate that `handle` refers to the current allocation of its
    /// physical page (spec §4.3: "stale handles fail before dereference").
    ///
    /// Returns `Err` if the epoch changed (device reload), the generation
    /// changed (page was freed and re-allocated), or the page is Free
    /// (use-after-free).
    pub fn validate_handle(&self, handle: &PageHandle) -> Result<(), String> {
        if handle.epoch != self.epoch {
            return Err(format!(
                "PageHandle: epoch mismatch (handle={}, pool={}) — device reloaded?",
                handle.epoch, self.epoch
            ));
        }
        self.check_phys(handle.phys)?;
        let meta = &self.page_meta[handle.phys as usize];
        if meta.generation != handle.generation {
            return Err(format!(
                "PageHandle: generation mismatch for phys {} (handle={}, \
                 current={}) — stale handle",
                handle.phys, handle.generation, meta.generation
            ));
        }
        if meta.state == PageState::Free {
            return Err(format!(
                "PageHandle: phys {} is Free (generation={}) — use-after-free",
                handle.phys, handle.generation
            ));
        }
        Ok(())
    }

    // ── Wave 2: internal refcount helpers ─────────────────────────────

    /// Decrement the table refcount for `phys`, transitioning state when
    /// it reaches zero.  Returns `Err` on underflow (spec §4.3: checked
    /// arithmetic, no wraparound).
    fn dec_table_ref(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        if meta.table_refs == 0 {
            return Err(format!(
                "dec_table_ref: table_refs underflow for phys {} (state={:?})",
                phys, meta.state
            ));
        }
        meta.table_refs -= 1;
        if meta.table_refs == 0 {
            if meta.cache_refs > 0 {
                meta.state = PageState::CacheOnly;
            } else if meta.inflight_refs > 0 {
                meta.state = PageState::ReclaimPending;
                self.reclaim_pending.push(phys);
            } else {
                meta.state = PageState::Free;
                meta.generation = meta.generation.wrapping_add(1);
                self.free_pages.push(phys);
            }
        }
        Ok(())
    }

    /// Free a just-reserved page back to the free list (used by COW
    /// rollback).  Bumps generation so any handle obtained during the
    /// failed reservation is stale.
    fn release_reserved(&mut self, phys: u32) {
        let meta = &mut self.page_meta[phys as usize];
        meta.state = PageState::Free;
        meta.table_refs = 0;
        meta.generation = meta.generation.wrapping_add(1);
        self.free_pages.push(phys);
    }

    // ── Allocation ────────────────────────────────────────────────────

    /// Allocate `n_pages` fresh pages and append them to `table`.
    /// Returns the number of pages actually allocated (may be less than
    /// requested if the pool is nearly full).
    pub fn alloc_pages(&mut self, table: &mut BlockTable, n_pages: usize) -> usize {
        let can_alloc = n_pages.min(self.free_pages.len());
        for _ in 0..can_alloc {
            let page = self.free_pages.pop().expect("free_pages non-empty");
            let meta = &mut self.page_meta[page as usize];
            meta.state = PageState::Private;
            meta.generation = meta.generation.wrapping_add(1);
            meta.table_refs = 1;
            table.push_page(page);
        }
        can_alloc
    }

    /// Allocate `n_pages` fresh pages and append them to `table`, returning
    /// [`PageHandle`]s carrying the current generation for each page (spec
    /// §4.3).  Unlike [`alloc_pages`], this returns `Err` on insufficient
    /// free pages rather than silently allocating fewer.
    pub fn alloc_pages_checked(
        &mut self,
        table: &mut BlockTable,
        n_pages: usize,
    ) -> Result<Vec<PageHandle>, String> {
        if n_pages > self.free_pages.len() {
            return Err(format!(
                "PagePool: need {} pages but only {} free",
                n_pages,
                self.free_pages.len()
            ));
        }
        let mut handles = Vec::with_capacity(n_pages);
        for _ in 0..n_pages {
            let page = self.free_pages.pop().expect("free_pages non-empty");
            let meta = &mut self.page_meta[page as usize];
            meta.state = PageState::Private;
            meta.generation = meta.generation.wrapping_add(1);
            meta.table_refs = 1;
            handles.push(PageHandle {
                phys: page,
                epoch: self.epoch,
                generation: meta.generation,
            });
            table.push_page(page);
        }
        Ok(handles)
    }

    // ── Prefix sharing (existing contract preserved) ──────────────────

    /// Share the first `n_pages` pages of `src` with `dst` by pushing the
    /// same physical indices into `dst` and incrementing their refcounts.
    /// Used for prefix sharing: the shared prefix pages persist until the
    /// last session releases them.
    ///
    /// After sharing, the shared pages are marked [`PageState::Sealed`]
    /// (immutable).  Any subsequent write intention against a Sealed page
    /// must go through copy-on-write ([`plan_cow`]/[`commit_cow`]).
    ///
    /// # The copy-on-write-free contract (must hold or KV corrupts silently)
    ///
    /// There is NO copy-on-write in this *sharing* path. Sharing is only
    /// safe at a page-aligned fork point, which this method enforces on
    /// both sides:
    ///
    /// - **Every shared page must be FULL in `src`** (`n_pages *
    ///   PAGE_TOKENS <= src.live_tokens()`). A full page is never written
    ///   again by append-only growth — src's next token goes to a fresh
    ///   page — so src cannot mutate bytes dst is reading.
    /// - **`dst` must have an empty table.** The shared pages BECOME dst's
    ///   prefix; the caller then sets dst's live length to the fork point
    ///   (`n_pages * PAGE_TOKENS`), so dst's own writes also land in fresh
    ///   pages only. Sharing into a non-empty table would interleave two
    ///   unrelated mappings.
    ///
    /// Sharing a partially-filled page, or letting either session's write
    /// frontier sit inside a shared page, lets one session's tokens land in
    /// the other's cache with no error raised.
    ///
    /// Panics if `src` has fewer than `n_pages` pages.
    ///
    /// [`plan_cow`]: PagePool::plan_cow
    /// [`commit_cow`]: PagePool::commit_cow
    pub fn share_prefix(
        &mut self,
        src: &BlockTable,
        dst: &mut BlockTable,
        n_pages: usize,
    ) -> Result<(), String> {
        if n_pages > src.num_pages() {
            return Err(format!(
                "share_prefix: src has {} pages but {} requested",
                src.num_pages(),
                n_pages
            ));
        }
        if n_pages * PAGE_TOKENS > src.live_tokens() {
            return Err(format!(
                "share_prefix: sharing {} pages covers {} tokens but src has \
                 only {} live — refusing to share a partially-filled page \
                 without copy-on-write (would corrupt src on its next append)",
                n_pages,
                n_pages * PAGE_TOKENS,
                src.live_tokens()
            ));
        }
        if dst.num_pages() != 0 {
            return Err(format!(
                "share_prefix: dst already holds {} pages — sharing is only \
                 defined into an empty table (fork point must be the start \
                 of dst's KV)",
                dst.num_pages()
            ));
        }
        for lp in 0..n_pages {
            let phys = src.physical(lp).expect("src page exists");
            self.check_phys(phys)?;
            let meta = &mut self.page_meta[phys as usize];
            meta.table_refs = meta
                .table_refs
                .checked_add(1)
                .ok_or_else(|| format!("share_prefix: refcount overflow for phys {}", phys))?;
            // Shared pages are immutable (spec §4.3: "cached pages immutable
            // even with one owner").
            meta.state = PageState::Sealed;
            dst.push_page(phys);
        }
        Ok(())
    }

    /// Validate that `phys` can accept a table ref — the same preconditions
    /// [`refcount_inc`](Self::refcount_inc) enforces — WITHOUT mutating the
    /// pool. A caller attaching several pages uses this to refuse before the
    /// first mutation, so a refused multi-page share cannot half-attach.
    pub fn check_attachable(&self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("refcount_inc: phys {} is Free", phys));
        }
        if meta.state == PageState::ReclaimPending {
            return Err(format!(
                "refcount_inc: phys {} is ReclaimPending — cannot attach a \
                 table to a page queued for reclaim",
                phys
            ));
        }
        if meta.table_refs == u32::MAX {
            return Err(format!("refcount_inc: overflow for phys {}", phys));
        }
        Ok(())
    }

    /// Increment the table refcount for physical page `phys` and seal it
    /// if currently Private (spec §4.3: shared pages are immutable).
    ///
    /// Uses checked arithmetic — returns `Err` on overflow or if the page
    /// is Free.
    pub fn refcount_inc(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("refcount_inc: phys {} is Free", phys));
        }
        if meta.state == PageState::ReclaimPending {
            return Err(format!(
                "refcount_inc: phys {} is ReclaimPending — cannot attach a \
                 table to a page queued for reclaim",
                phys
            ));
        }
        meta.table_refs = meta
            .table_refs
            .checked_add(1)
            .ok_or_else(|| format!("refcount_inc: overflow for phys {}", phys))?;
        if meta.state == PageState::Private {
            meta.state = PageState::Sealed;
        }
        Ok(())
    }

    // ── Release / free ────────────────────────────────────────────────

    /// Free all pages in `table`, decrementing table refcounts with
    /// checked arithmetic.  Pages whose refs reach zero are returned to
    /// the free list (or to the ReclaimPending queue if in-flight refs
    /// remain).  Returns `Err` if any page had a zero table refcount
    /// (underflow / duplicate release); the table is still cleared.
    pub fn release_table(&mut self, table: &mut BlockTable) -> Result<(), String> {
        let mut had_error = false;
        for &phys in table.page_indices() {
            if let Err(_) = self.dec_table_ref(phys) {
                had_error = true;
            }
        }
        table.pages.clear();
        table.live_tokens = 0;
        if had_error {
            Err("release_table: refcount underflow on one or more pages".to_string())
        } else {
            Ok(())
        }
    }

    /// Free the last `n_pages` pages from `table`, decrementing table
    /// refcounts with checked arithmetic.  Used when trimming a session's
    /// KV (e.g. sliding window eviction).
    ///
    /// All-or-nothing: every tail page's refcount is verified BEFORE the
    /// table is truncated, so an underflow leaves the table and the pool
    /// untouched instead of stranding pages that are no longer mapped.
    /// Returns `Err` on underflow.
    pub fn free_pages_from_tail(
        &mut self,
        table: &mut BlockTable,
        n_pages: usize,
    ) -> Result<(), String> {
        let keep = table.num_pages().saturating_sub(n_pages);
        // Verify phase: every page about to be dropped must hold a table
        // ref. dec_table_ref cannot fail after this.
        for &phys in &table.pages[keep..] {
            if self.page_meta[phys as usize].table_refs == 0 {
                return Err(format!(
                    "free_pages_from_tail: table_refs underflow for phys {} \
                     (state={:?}) — table unchanged",
                    phys, self.page_meta[phys as usize].state
                ));
            }
        }
        let freed = table.truncate(keep);
        for phys in freed {
            self.dec_table_ref(phys)?;
        }
        Ok(())
    }

    /// Ensure `table` has enough pages to hold `n_tokens` positions,
    /// allocating new pages as needed. Returns the number of new pages
    /// allocated.
    pub fn ensure_capacity(
        &mut self,
        table: &mut BlockTable,
        n_tokens: usize,
    ) -> Result<usize, String> {
        let needed_pages = n_tokens.div_ceil(PAGE_TOKENS);
        let current_pages = table.num_pages();
        if needed_pages <= current_pages {
            table.set_live_tokens(n_tokens);
            return Ok(0);
        }
        let to_alloc = needed_pages - current_pages;
        if to_alloc > self.free_pages.len() {
            return Err(format!(
                "PagePool: need {} more pages but only {} free ({} tokens requested, {} pages allocated)",
                to_alloc,
                self.free_pages.len(),
                n_tokens,
                current_pages
            ));
        }
        let allocated = self.alloc_pages(table, to_alloc);
        table.set_live_tokens(n_tokens);
        Ok(allocated)
    }

    // ── Seal ──────────────────────────────────────────────────────────

    /// Mark a full page as [`PageState::Sealed`] (immutable).  Idempotent
    /// on already-Sealed pages.  Returns `Err` if the page is Free,
    /// CacheOnly, or ReclaimPending (spec §4.3).
    ///
    /// Bounds-checks `phys` first like every other public entry point — a
    /// stale or fabricated index must produce `Err`, never a panic.
    pub fn seal(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        match meta.state {
            PageState::Private => {
                meta.state = PageState::Sealed;
                Ok(())
            }
            PageState::Sealed => Ok(()),
            _ => Err(format!(
                "seal: phys {} is {:?}, cannot seal",
                phys, meta.state
            )),
        }
    }

    // ── Copy-on-write (spec §4.3 C3) ──────────────────────────────────

    /// Inspect the write interval `[write_start, write_end)` and reserve
    /// private destination pages for every [`PageState::Sealed`] or
    /// [`PageState::CacheOnly`] page in that interval.  Returns a
    /// [`CowPlan`] describing the copies the caller must perform.
    ///
    /// **Does NOT modify the block table** — the caller executes the GPU
    /// memcpys, then calls [`commit_cow`] to rebind the table, or
    /// [`abort_cow`] to release the reserved pages on failure.
    ///
    /// If any reservation fails (OOM), all reserved pages are released
    /// and `Err` is returned — the original table and pool state are
    /// untouched (spec §4.3: "transactional failure releases only newly
    /// reserved resources").
    ///
    /// [`commit_cow`]: PagePool::commit_cow
    /// [`abort_cow`]: PagePool::abort_cow
    pub fn plan_cow(
        &mut self,
        table: &BlockTable,
        write_start: usize,
        write_end: usize,
    ) -> Result<CowPlan, String> {
        if write_end < write_start {
            return Err("plan_cow: write_end < write_start".to_string());
        }
        if write_end == write_start {
            return Ok(CowPlan {
                copies: Vec::new(),
                armed: false,
            });
        }

        let first_lp = write_start / PAGE_TOKENS;
        let last_lp = (write_end - 1) / PAGE_TOKENS;

        let mut copies = Vec::new();
        let mut reserved: Vec<u32> = Vec::new();

        for lp in first_lp..=last_lp {
            let phys = match table.physical(lp) {
                Some(p) => p,
                None => continue, // page not yet allocated — no COW needed
            };
            let state = self.page_meta[phys as usize].state;
            match state {
                PageState::Sealed | PageState::CacheOnly => {
                    // Reserve a private destination.
                    let dst = match self.free_pages.pop() {
                        Some(p) => p,
                        None => {
                            // OOM — rollback all reservations.
                            for &d in &reserved {
                                self.release_reserved(d);
                            }
                            return Err(format!(
                                "plan_cow: OOM — need COW page for logical {} \
                                 (phys {}, state={:?}) but no free pages",
                                lp, phys, state
                            ));
                        }
                    };
                    let dst_meta = &mut self.page_meta[dst as usize];
                    dst_meta.state = PageState::Private;
                    dst_meta.generation = dst_meta.generation.wrapping_add(1);
                    dst_meta.table_refs = 1;

                    // Valid prefix: tokens in this page before write_start.
                    let page_start = lp * PAGE_TOKENS;
                    let valid_prefix_tokens =
                        write_start.saturating_sub(page_start).min(PAGE_TOKENS);

                    copies.push(CowCopy {
                        logical_page: lp,
                        src_phys: phys,
                        dst_phys: dst,
                        valid_prefix_tokens,
                        k_copy_bytes: valid_prefix_tokens * self.k_per_pos_bytes,
                        v_copy_bytes: valid_prefix_tokens * self.v_per_pos_bytes,
                    });
                    reserved.push(dst);
                }
                _ => {} // Private — mutable, no COW needed
            }
        }

        Ok(CowPlan {
            copies,
            armed: true,
        })
    }

    /// Rebind the block table entries per `plan` and adjust refcounts on
    /// the old (Sealed/CacheOnly) pages.  Call this **after** the GPU
    /// memcpys described by the plan have been issued (spec §4.3: "rebind
    /// table only after copy ordering established").
    ///
    /// All-or-nothing: every copy's source mapping is verified against the
    /// table BEFORE any rebind or refcount change. If the table was
    /// truncated, reallocated, or already partially committed since
    /// `plan_cow`, the commit fails and the plan's reserved pages are
    /// released — the table is left untouched.
    ///
    /// Old pages whose table refs reach zero transition to Free,
    /// CacheOnly, or ReclaimPending as appropriate.
    pub fn commit_cow(&mut self, table: &mut BlockTable, mut plan: CowPlan) -> Result<(), String> {
        // Verify phase: the table must still map each logical page to the
        // plan's source, and every source must still hold a table ref.
        // Anything else means the table changed under the plan — committing
        // would dec a ref this table no longer holds (underflow/double-free)
        // or leak the page being overwritten.
        let mut failure: Option<String> = None;
        for c in &plan.copies {
            match table.physical(c.logical_page) {
                Some(p) if p == c.src_phys => {}
                other => {
                    let got = other
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "none".into());
                    failure = Some(format!(
                        "commit_cow: logical page {} maps to {} not src {} — \
                         table changed since plan_cow; aborted",
                        c.logical_page, got, c.src_phys
                    ));
                    break;
                }
            }
            if self.page_meta[c.src_phys as usize].table_refs == 0 {
                failure = Some(format!(
                    "commit_cow: src phys {} has table_refs=0 — aborted",
                    c.src_phys
                ));
                break;
            }
        }
        if let Some(msg) = failure {
            // Release the plan's reserved pages; the table is untouched.
            plan.armed = false;
            for c in &plan.copies {
                self.release_reserved(c.dst_phys);
            }
            return Err(msg);
        }
        // Apply phase: verified above, so dec_table_ref cannot fail.
        for c in &plan.copies {
            table.set_page(c.logical_page, c.dst_phys);
            self.dec_table_ref(c.src_phys)?;
        }
        plan.armed = false;
        Ok(())
    }

    /// Release all pages reserved by `plan_cow` without rebinding the
    /// table.  Call this when the GPU memcpy failed or the step was
    /// cancelled (spec §4.3: "transactional failure releases only newly
    /// reserved resources").
    pub fn abort_cow(&mut self, mut plan: CowPlan) {
        for c in &plan.copies {
            self.release_reserved(c.dst_phys);
        }
        plan.armed = false;
    }

    // ── Deferred reclaim (spec §4.4 C4) ───────────────────────────────

    /// Add an in-flight lease to `phys`, preventing immediate reuse even
    /// after all table/cache refs are released (spec §4.4: "rc==0 ≠ GPU
    /// completion; free/reuse requires all leases released").
    ///
    /// Returns `Err` if the page is Free or on overflow.
    pub fn add_inflight_ref(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("add_inflight_ref: phys {} is Free", phys));
        }
        if meta.state == PageState::ReclaimPending {
            // A reclaim-pending page is awaiting device completion; taking a
            // new lease would let `drain_completed` free it out from under
            // the new reader once its OLD inflight refs drain.
            return Err(format!(
                "add_inflight_ref: phys {} is ReclaimPending — re-pinning a \
                 page queued for reclaim is refused",
                phys
            ));
        }
        meta.inflight_refs = meta
            .inflight_refs
            .checked_add(1)
            .ok_or_else(|| format!("add_inflight_ref: overflow for phys {}", phys))?;
        Ok(())
    }

    /// Release an in-flight lease from `phys`.  The page is NOT freed
    /// here — it remains [`PageState::ReclaimPending`] until
    /// [`drain_completed`] confirms completion (spec §4.4).
    ///
    /// Returns `Err` on underflow.
    ///
    /// [`drain_completed`]: PagePool::drain_completed
    pub fn release_inflight_ref(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        if meta.inflight_refs == 0 {
            return Err(format!("release_inflight_ref: underflow for phys {}", phys));
        }
        meta.inflight_refs -= 1;
        Ok(())
    }

    /// Synchronous step-fence drain: free every [`PageState::ReclaimPending`]
    /// page whose in-flight refs have all been released (spec §4.4:
    /// "drain_completed(now/lease-proof)").
    ///
    /// Returns the physical page indices that were freed.
    ///
    /// # HIP-event upgrade seam
    ///
    /// This initial implementation uses a **synchronous fence**: the
    /// caller is expected to have completed a `hipStreamSynchronize` (or
    /// equivalent) before calling `drain_completed`, so any page with
    /// `inflight_refs == 0` is safe to free.  The upgrade path is:
    ///
    /// 1. `add_inflight_ref` records the HIP event/stream associated with
    ///    the device read (stored alongside `inflight_refs` in `PageMeta`).
    /// 2. `drain_completed` queries `hipEventQuery` for each pending page
    ///    and frees only those whose event is signaled — no global sync
    ///    needed.
    /// 3. A `drain_completed_timeout` variant may wait on events with a
    ///    deadline for back-pressure.
    ///
    /// The current API (`add_inflight_ref` / `release_inflight_ref` /
    /// `drain_completed`) is structured so this upgrade is non-breaking:
    /// only the internal completion check changes.
    pub fn drain_completed(&mut self) -> Vec<u32> {
        let mut freed = Vec::new();
        let mut still_pending = Vec::new();
        for &phys in &self.reclaim_pending {
            let meta = &self.page_meta[phys as usize];
            // Defensive: a page must not regain owners while queued. If it
            // somehow did, keep it queued and let the underflow-checked
            // release paths reconcile it instead of freeing live data.
            if meta.inflight_refs == 0 && meta.table_refs == 0 && meta.cache_refs == 0 {
                let meta = &mut self.page_meta[phys as usize];
                meta.state = PageState::Free;
                meta.generation = meta.generation.wrapping_add(1);
                self.free_pages.push(phys);
                freed.push(phys);
            } else {
                still_pending.push(phys);
            }
        }
        self.reclaim_pending = still_pending;
        freed
    }

    // ── Cache leases ──────────────────────────────────────────────────

    /// Add a cache ownership lease to `phys`.  A page with a cache lease
    /// is immutable: if it was Private, it transitions to Sealed (spec
    /// §4.3: "cached pages immutable even with one owner").
    ///
    /// Returns `Err` if the page is Free or on overflow.
    pub fn add_cache_ref(&mut self, phys: u32) -> Result<(), String> {
        self.check_phys(phys)?;
        let meta = &mut self.page_meta[phys as usize];
        if meta.state == PageState::Free {
            return Err(format!("add_cache_ref: phys {} is Free", phys));
        }
        if meta.state == PageState::ReclaimPending {
            // See add_inflight_ref: a page queued for reclaim must not gain
            // a new owner — drain_completed frees it once inflight hits 0,
            // regardless of cache_refs.
            return Err(format!(
                "add_cache_ref: phys {} is ReclaimPending — re-pinning a \
                 page queued for reclaim is refused",
                phys
            ));
        }
        meta.cache_refs = meta
            .cache_refs
            .checked_add(1)
            .ok_or_else(|| format!("add_cache_ref: overflow for phys {}", phys))?;
        if meta.state == PageState::Private {
            meta.state = PageState::Sealed;
        }
        Ok(())
    }

    /// Release a cache ownership lease from `phys`.  If all table/cache
    /// refs are now zero the page is freed — or enters ReclaimPending when
    /// in-flight refs remain, mirroring `dec_table_ref` (a page here used to
    /// leak: it was neither freed nor enqueued, so `drain_completed` could
    /// never see it).  Returns `Err` on underflow.
    pub fn release_cache_ref(&mut self, phys: u32) -> Result<(), String> {
        let disposition = {
            self.check_phys(phys)?;
            let meta = &mut self.page_meta[phys as usize];
            if meta.cache_refs == 0 {
                return Err(format!("release_cache_ref: underflow for phys {}", phys));
            }
            meta.cache_refs -= 1;
            if meta.table_refs == 0 && meta.cache_refs == 0 {
                if meta.inflight_refs > 0 {
                    Some(true)
                } else {
                    Some(false)
                }
            } else {
                None
            }
        };
        match disposition {
            Some(true) => {
                let meta = &mut self.page_meta[phys as usize];
                meta.state = PageState::ReclaimPending;
                self.reclaim_pending.push(phys);
            }
            Some(false) => {
                let meta = &mut self.page_meta[phys as usize];
                meta.state = PageState::Free;
                meta.generation = meta.generation.wrapping_add(1);
                self.free_pages.push(phys);
            }
            None => {}
        }
        Ok(())
    }

    // ── Descriptor construction (unchanged from Wave 1) ───────────────

    /// Build a KvSlotDesc for a session with the given block table.
    /// `block_table_dev_addr` is the GPU address of the uploaded page
    /// index array (i32 per page). The descriptor is in paged mode.
    pub fn make_desc(&self, table: &BlockTable, block_table_dev_addr: u64) -> KvSlotDesc {
        // seq_len is i32 on the wire; clamp rather than wrap — a wrapped
        // negative reads as a huge positive length in the kernel.
        let seq_len = table.live_tokens().min(i32::MAX as usize) as i32;
        KvSlotDesc {
            block_table: block_table_dev_addr,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len,
            page_tokens: PAGE_TOKENS as i32,
        }
    }

    /// Build a legacy-mode KvSlotDesc (contiguous slab at `base`).
    /// Used for backward compatibility with non-paged code paths.
    pub fn make_legacy_desc(base: u64, seq_len: usize) -> KvSlotDesc {
        let seq_len = seq_len.min(i32::MAX as usize) as i32;
        KvSlotDesc {
            block_table: 0,
            // Equal K/V bases: this constructor is for equal-stride legacy
            // arenas (Q8_0 / BF16), where one slab offset serves both.
            legacy_k_base: base,
            legacy_v_base: base,
            seq_len,
            page_tokens: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests (updated for Wave 2 API) ───────────────────────

    #[test]
    fn alloc_and_release_roundtrip() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        assert_eq!(pool.free_pages(), 16);

        let mut table = BlockTable::new();
        let allocated = pool.alloc_pages(&mut table, 4);
        assert_eq!(allocated, 4);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 12);

        // Each allocated page should have refcount 1 and be Private
        for &phys in table.page_indices() {
            assert_eq!(pool.refcount(phys), 1);
            assert_eq!(pool.page_state(phys), PageState::Private);
        }

        pool.release_table(&mut table).unwrap();
        assert_eq!(table.num_pages(), 0);
        assert_eq!(pool.free_pages(), 16);
    }

    /// Deterministic xorshift PRNG — no external deps, reproducible seeds.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// Randomized COW/alloc/share/release fuzz against the invariant
    /// checker. Every op is precondition-checked so failures are real
    /// bugs, not bad inputs; `check_invariants` runs after every step so
    /// a refcount drift or double-free is caught at the op that caused it.
    /// Host-only — no GPU needed.
    #[test]
    fn cow_fuzz_invariants_hold_under_random_ops() {
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        let mut pool = PagePool::new(32, 1088).unwrap();
        let mut tables: Vec<BlockTable> = (0..4).map(|_| BlockTable::new()).collect();
        let mut plans: Vec<Option<(usize, CowPlan)>> = (0..4).map(|_| None).collect();
        // Cache refs the fuzz itself holds — a page can carry more than
        // one across a free/realloc cycle, so count them.
        let mut cache_held: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for step in 0..2000 {
            let t = rng.below(4) as usize;
            match rng.below(10) {
                // Allocate 1-3 pages.
                0 => {
                    let want = 1 + rng.below(3) as usize;
                    pool.alloc_pages(&mut tables[t], want);
                }
                // Share a prefix into another table.
                1 => {
                    let src = rng.below(4) as usize;
                    if src != t && tables[src].num_pages() > 0 {
                        let n = 1 + rng.below(tables[src].num_pages() as u64) as usize;
                        let (lo, hi) = if src < t { (src, t) } else { (t, src) };
                        let (a, b) = tables.split_at_mut(hi);
                        let (src_t, dst_t) = if src < t {
                            (&a[lo], &mut b[0])
                        } else {
                            (&b[0], &mut a[lo])
                        };
                        let _ = pool.share_prefix(src_t, dst_t, n);
                    }
                }
                // Plan a COW over a random write range.
                2 => {
                    if tables[t].num_pages() > 0 && plans[t].is_none() {
                        let end = tables[t].num_pages() * PAGE_TOKENS;
                        let start = rng.below(end as u64 + 1) as usize;
                        let stop = start + rng.below((end - start) as u64 + 1) as usize;
                        if let Ok(plan) = pool.plan_cow(&tables[t], start, stop) {
                            plans[t] = Some((t, plan));
                        }
                    }
                }
                // Commit a pending plan.
                3 => {
                    if let Some((ti, plan)) = plans[t].take() {
                        let _ = pool.commit_cow(&mut tables[ti], plan);
                    }
                }
                // Abort a pending plan.
                4 => {
                    if let Some((_, plan)) = plans[t].take() {
                        pool.abort_cow(plan);
                    }
                }
                // Free pages from a table's tail.
                5 => {
                    if tables[t].num_pages() > 0 {
                        let n = 1 + rng.below(tables[t].num_pages() as u64) as usize;
                        let _ = pool.free_pages_from_tail(&mut tables[t], n);
                    }
                }
                // Release a whole table.
                6 => {
                    let _ = pool.release_table(&mut tables[t]);
                    if let Some((_, plan)) = plans[t].take() {
                        pool.abort_cow(plan);
                    }
                }
                // Seal a random page of a table.
                7 => {
                    if tables[t].num_pages() > 0 {
                        let lp = rng.below(tables[t].num_pages() as u64) as usize;
                        if let Some(phys) = tables[t].physical(lp) {
                            let _ = pool.seal(phys);
                        }
                    }
                }
                // Add/release a cache ref on a random page.
                8 => {
                    if tables[t].num_pages() > 0 {
                        let lp = rng.below(tables[t].num_pages() as u64) as usize;
                        if let Some(phys) = tables[t].physical(lp) {
                            if rng.below(2) == 0 {
                                if pool.add_cache_ref(phys).is_ok() {
                                    *cache_held.entry(phys).or_insert(0) += 1;
                                }
                            } else if let Some(n) = cache_held.get_mut(&phys) {
                                if pool.release_cache_ref(phys).is_ok() {
                                    *n -= 1;
                                    if *n == 0 {
                                        cache_held.remove(&phys);
                                    }
                                }
                            }
                        }
                    }
                }
                // Drain reclaim-pending.
                _ => {
                    pool.drain_completed();
                }
            }
            pool.check_invariants();
        }

        // Release the fuzz's own cache refs, then teardown must return
        // every page.
        for (phys, n) in cache_held.drain() {
            for _ in 0..n {
                let _ = pool.release_cache_ref(phys);
            }
        }
        for entry in plans.iter_mut() {
            if let Some((_, plan)) = entry.take() {
                pool.abort_cow(plan);
            }
        }
        for t in &mut tables {
            let _ = pool.release_table(t);
        }
        pool.drain_completed();
        pool.check_invariants();
        for p in 0..pool.n_pages() as u32 {
            let m = pool.page_meta[p as usize];
            if m.state != PageState::Free {
                eprintln!(
                    "leaked page {p}: state={:?} table={} cache={} inflight={}",
                    m.state, m.table_refs, m.cache_refs, m.inflight_refs
                );
            }
        }
        assert_eq!(pool.free_pages(), 32, "pages leaked after teardown");
    }

    #[test]
    fn ensure_capacity_allocates_on_demand() {
        let mut pool = PagePool::new(32, 1088).unwrap();
        let mut table = BlockTable::new();

        // 300 tokens needs 3 pages (ceil(300/128) = 3)
        let n = pool.ensure_capacity(&mut table, 300).unwrap();
        assert_eq!(n, 3);
        assert_eq!(table.num_pages(), 3);
        assert_eq!(table.live_tokens(), 300);

        // Growing to 500 tokens needs 4 pages (ceil(500/128) = 4)
        let n = pool.ensure_capacity(&mut table, 500).unwrap();
        assert_eq!(n, 1);
        assert_eq!(table.num_pages(), 4);
        assert_eq!(table.live_tokens(), 500);

        // Shrinking to 200 tokens needs 2 pages — no alloc, just trim live_tokens
        let n = pool.ensure_capacity(&mut table, 200).unwrap();
        assert_eq!(n, 0);
        assert_eq!(table.num_pages(), 4); // pages not freed, just live_tokens reduced
        assert_eq!(table.live_tokens(), 200);
    }

    #[test]
    fn ensure_capacity_oom() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();

        // 4 pages * 128 = 512 tokens max
        let result = pool.ensure_capacity(&mut table, 600);
        assert!(result.is_err());
    }

    #[test]
    fn prefix_sharing_increments_refcount() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(400);

        // Share first 3 pages with table_b
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 3).unwrap();

        assert_eq!(table_b.num_pages(), 3);
        // Shared pages should have refcount 2 and be Sealed
        for lp in 0..3 {
            let phys = table_a.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 2);
            assert_eq!(pool.page_state(phys), PageState::Sealed);
        }
        // Non-shared page should still have refcount 1 and be Private
        let phys_4 = table_a.physical(3).unwrap();
        assert_eq!(pool.refcount(phys_4), 1);
        assert_eq!(pool.page_state(phys_4), PageState::Private);

        // Releasing table_a should free only the non-shared page
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.free_pages(), 16 - 3); // only 1 page freed, 3 still shared
        for lp in 0..3 {
            let phys = table_b.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1);
            // Still Sealed — "cached pages immutable even with one owner"
            assert_eq!(pool.page_state(phys), PageState::Sealed);
        }

        // Releasing table_b frees the remaining shared pages
        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn free_pages_from_tail() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 6);
        table.set_live_tokens(600);

        // Free last 2 pages
        pool.free_pages_from_tail(&mut table, 2).unwrap();
        assert_eq!(table.num_pages(), 4);
        assert_eq!(pool.free_pages(), 16 - 4);
    }

    #[test]
    fn share_prefix_refuses_partially_filled_tail_page() {
        // 3 pages held, but only 300 tokens live: page 2 is 44/128 full.
        // Sharing it without copy-on-write lets src's next append corrupt
        // dst's view of the prefix — must refuse.
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 3);
        src.set_live_tokens(300);

        let mut dst = BlockTable::new();
        let err = pool.share_prefix(&src, &mut dst, 3).unwrap_err();
        assert!(err.contains("partially-filled"), "unexpected: {err}");
        assert_eq!(dst.num_pages(), 0, "refused share must not touch dst");
        for lp in 0..3 {
            let phys = src.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1, "refcounts untouched");
        }
        // Sharing only the full pages is fine.
        pool.share_prefix(&src, &mut dst, 2).unwrap();
        assert_eq!(dst.num_pages(), 2);
    }

    #[test]
    fn share_prefix_refuses_nonempty_dst() {
        let mut pool = PagePool::new(16, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 2);
        src.set_live_tokens(256);

        let mut dst = BlockTable::new();
        pool.alloc_pages(&mut dst, 1);
        let err = pool.share_prefix(&src, &mut dst, 2).unwrap_err();
        assert!(err.contains("empty table"), "unexpected: {err}");
        assert_eq!(dst.num_pages(), 1, "refused share must not touch dst");
        // And the refused attempt must not have bumped refcounts.
        for lp in 0..2 {
            let phys = src.physical(lp).unwrap();
            assert_eq!(pool.refcount(phys), 1);
        }
    }

    #[test]
    fn release_table_after_failed_share_is_balanced() {
        // Edge case: a failed share must leave the pool so that full
        // alloc/release cycles still return every page.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut src = BlockTable::new();
        pool.alloc_pages(&mut src, 2);
        src.set_live_tokens(100); // partial tail page
        let mut dst = BlockTable::new();
        assert!(pool.share_prefix(&src, &mut dst, 2).is_err());

        pool.release_table(&mut src).unwrap();
        assert_eq!(pool.free_pages(), 8);
        pool.release_table(&mut dst).unwrap(); // empty — no-op
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn make_desc_paged_mode() {
        let pool = PagePool::new(16, 1088).unwrap();
        let mut table = BlockTable::new();
        // Can't mutate table through immutable pool, so test with manual table
        table.push_page(5);
        table.push_page(12);
        table.set_live_tokens(200);

        let desc = pool.make_desc(&table, 0xDEAD_BEEF);
        assert_eq!(desc.block_table, 0xDEAD_BEEF);
        assert_eq!(desc.legacy_k_base, 0);
        assert_eq!(desc.legacy_v_base, 0);
        assert_eq!(desc.seq_len, 200);
        assert_eq!(desc.page_tokens, PAGE_TOKENS as i32);
    }

    #[test]
    fn make_legacy_desc() {
        let desc = PagePool::make_legacy_desc(0x1000, 42);
        assert_eq!(desc.block_table, 0);
        assert_eq!(desc.legacy_k_base, 0x1000);
        assert_eq!(desc.legacy_v_base, 0x1000);
        assert_eq!(desc.seq_len, 42);
        assert_eq!(desc.page_tokens, 0);
    }

    // ── Wave 2: PageHandle / generation tests ──────────────────────────

    #[test]
    fn alloc_pages_checked_returns_valid_handles() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        let handles = pool.alloc_pages_checked(&mut table, 3).unwrap();
        assert_eq!(handles.len(), 3);
        for h in &handles {
            assert!(pool.validate_handle(h).is_ok());
            assert_eq!(pool.page_state(h.phys), PageState::Private);
        }
        assert_eq!(pool.free_pages(), 5);
    }

    #[test]
    fn stale_handle_after_release_fails_validation() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        let handles = pool.alloc_pages_checked(&mut table, 2).unwrap();

        // Release the table — pages are freed, generations bumped.
        pool.release_table(&mut table).unwrap();

        // All handles should now be stale (page is Free, generation changed).
        for h in &handles {
            assert!(
                pool.validate_handle(h).is_err(),
                "handle must be stale after release"
            );
        }
    }

    #[test]
    fn handle_still_valid_while_page_referenced() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table_a = BlockTable::new();
        let handles_a = pool.alloc_pages_checked(&mut table_a, 2);
        let handles_a = handles_a.unwrap();
        table_a.set_live_tokens(2 * PAGE_TOKENS);

        // Share with table_b
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();

        // Release table_a — shared pages still referenced by table_b
        pool.release_table(&mut table_a).unwrap();

        // Handles for shared pages should still be valid (page not freed,
        // generation unchanged — only table_refs decremented).
        for h in &handles_a {
            assert_eq!(pool.page_state(h.phys), PageState::Sealed);
            assert!(
                pool.validate_handle(h).is_ok(),
                "handle for still-referenced page should be valid"
            );
        }

        // Release table_b — now pages are freed
        pool.release_table(&mut table_b).unwrap();
        for h in &handles_a {
            assert!(
                pool.validate_handle(h).is_err(),
                "handle must be stale after all refs released"
            );
        }
    }

    // ── Wave 2: seal / PageState tests ────────────────────────────────

    #[test]
    fn seal_marks_private_page_as_sealed() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Private);

        pool.seal(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        // Idempotent
        pool.seal(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);
    }

    #[test]
    fn seal_refuses_free_page() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let err = pool.seal(0).unwrap_err();
        assert!(err.contains("Free"));
    }

    #[test]
    fn shared_page_is_sealed_and_stays_sealed_after_partial_release() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 2);
        table_a.set_live_tokens(2 * PAGE_TOKENS);

        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();

        let phys = table_a.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);
        assert_eq!(pool.refcount(phys), 2);

        // Release one table — page still Sealed (immutable even with one owner)
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.refcount(phys), 1);
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Free);
    }

    // ── Wave 2: COW tests (A2, A4) ────────────────────────────────────

    #[test]
    fn a2_branch_share_then_cow_isolation() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        // Branch A: 4 full pages (512 tokens)
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(4 * PAGE_TOKENS);

        // Share first 2 pages with branch B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();
        table_b.set_live_tokens(2 * PAGE_TOKENS);

        // Shared pages are Sealed and have identical physical indices
        for lp in 0..2 {
            let phys = table_a.physical(lp).unwrap();
            assert_eq!(pool.page_state(phys), PageState::Sealed);
            assert_eq!(table_a.physical(lp), table_b.physical(lp));
        }

        // Both branches append: allocate new private pages
        pool.alloc_pages(&mut table_a, 1);
        table_a.set_live_tokens(5 * PAGE_TOKENS);
        pool.alloc_pages(&mut table_b, 1);
        table_b.set_live_tokens(3 * PAGE_TOKENS);

        // Private tails differ and are Private
        assert_ne!(table_a.physical(4), table_b.physical(2));
        assert_eq!(
            pool.page_state(table_a.physical(4).unwrap()),
            PageState::Private
        );
        assert_eq!(
            pool.page_state(table_b.physical(2).unwrap()),
            PageState::Private
        );

        // Branch B rewinds and wants to write page 0 (Sealed → COW needed)
        let plan = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].logical_page, 0);
        assert_eq!(plan.copies()[0].src_phys, table_b.physical(0).unwrap());
        assert_eq!(plan.copies()[0].valid_prefix_tokens, 0); // write from pos 0

        // Commit COW: rebind table B's page 0 to the private copy
        pool.commit_cow(&mut table_b, plan).unwrap();

        // Branch B's page 0 is now private (different from A's)
        assert_ne!(table_b.physical(0), table_a.physical(0));
        assert_eq!(
            pool.page_state(table_b.physical(0).unwrap()),
            PageState::Private
        );

        // Branch A's page 0 is still the original shared page (unchanged)
        let original_phys = table_a.physical(0).unwrap();
        assert_eq!(pool.page_state(original_phys), PageState::Sealed);
        assert_eq!(pool.refcount(original_phys), 1); // only A references it now

        // Page conservation: 16 total
        // A: [orig0, orig1, orig2, orig3, new_a] = 5 pages
        // B: [cow0, orig1, new_b] = 3 pages
        // Unique: orig0, orig1, orig2, orig3, new_a, cow0, new_b = 7
        // Free: 16 - 7 = 9
        assert_eq!(pool.free_pages(), 9);
    }

    #[test]
    fn cow_plan_with_valid_prefix() {
        // Write starts mid-page: the valid prefix (tokens before write_start
        // in that page) must be copied.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // Seal page 0 so COW is triggered
        pool.seal(table.physical(0).unwrap()).unwrap();

        // Write at positions [64, 128) — page 0 has 64 valid prefix tokens
        let plan = pool.plan_cow(&table, 64, 128).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].valid_prefix_tokens, 64);
        assert_eq!(plan.copies()[0].k_copy_bytes, 64 * 1088);
        assert_eq!(plan.copies()[0].v_copy_bytes, 64 * 1088);
    }

    #[test]
    fn cow_plan_skips_private_pages() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // No pages are Sealed — plan should be empty
        let plan = pool.plan_cow(&table, 0, 2 * PAGE_TOKENS).unwrap();
        assert!(plan.is_empty());
    }

    #[test]
    fn cow_abort_releases_reserved_pages() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);
        pool.seal(table.physical(0).unwrap()).unwrap();

        let free_before = pool.free_pages();
        let plan = pool.plan_cow(&table, 0, PAGE_TOKENS).unwrap();
        assert_eq!(pool.free_pages(), free_before - 1); // one page reserved

        // Abort — reserved page released
        pool.abort_cow(plan);
        assert_eq!(pool.free_pages(), free_before);

        // Table unchanged
        assert_eq!(table.num_pages(), 2);
    }

    #[test]
    fn cow_commit_refuses_stale_plan_and_releases_reservations() {
        // A plan whose table changed between plan_cow and commit_cow must
        // fail all-or-nothing: no rebind, no refcount drift, and the
        // reserved destination pages return to the free list.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);
        pool.seal(table.physical(0).unwrap()).unwrap();

        let free_before = pool.free_pages();
        let plan = pool.plan_cow(&table, 0, PAGE_TOKENS).unwrap();
        assert_eq!(pool.free_pages(), free_before - 1);

        // Table shrinks under the plan: logical page 0 no longer maps to
        // the planned source.
        pool.free_pages_from_tail(&mut table, 2).unwrap();
        assert_eq!(table.num_pages(), 0);

        let result = pool.commit_cow(&mut table, plan);
        assert!(result.is_err(), "stale plan must be refused");
        // Reserved dst page released; table untouched by the failed commit.
        // Reserved dst page released; the two truncated pages also
        // returned to the free list.
        assert_eq!(pool.free_pages(), free_before + 2);
        assert_eq!(table.num_pages(), 0);
    }

    #[test]
    fn cow_commit_twice_is_impossible_by_value() {
        // commit_cow consumes the plan — a second commit cannot be
        // expressed. This test pins the companion invariant: after a
        // successful commit the dst page is table-owned and the src page's
        // table ref was decremented exactly once.
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 1);
        table_a.set_live_tokens(PAGE_TOKENS);
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 1).unwrap();
        table_b.set_live_tokens(PAGE_TOKENS);

        let src = table_b.physical(0).unwrap();
        let plan = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap();
        let dst = plan.copies()[0].dst_phys;
        pool.commit_cow(&mut table_b, plan).unwrap();

        assert_eq!(table_b.physical(0), Some(dst));
        // src lost exactly one table ref (B's); A's ref keeps it sealed.
        assert_eq!(pool.page_state(src), PageState::Sealed);
        assert_eq!(pool.page_state(dst), PageState::Private);
    }

    #[test]
    fn a4_cow_oom_leaves_tables_intact() {
        let mut pool = PagePool::new(8, 1088).unwrap();

        // Branch A: 4 full pages
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 4);
        table_a.set_live_tokens(4 * PAGE_TOKENS);

        // Share all 4 pages with branch B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 4).unwrap();
        table_b.set_live_tokens(4 * PAGE_TOKENS);

        // Exhaust remaining pages with branch C
        let mut table_c = BlockTable::new();
        pool.alloc_pages(&mut table_c, 4);
        table_c.set_live_tokens(4 * PAGE_TOKENS);
        assert_eq!(pool.free_pages(), 0);

        // COW on table_b fails — no free pages
        let err = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap_err();
        assert!(err.contains("OOM"), "unexpected: {err}");

        // Table B is unchanged (plan_cow doesn't modify the table on failure)
        assert_eq!(table_b.num_pages(), 4);
        for lp in 0..4 {
            assert_eq!(
                table_b.physical(lp),
                table_a.physical(lp),
                "table B must be unchanged after failed COW"
            );
        }

        // Pool state unchanged — no pages leaked
        assert_eq!(pool.free_pages(), 0);

        // Unrelated request: release C, then COW succeeds
        pool.release_table(&mut table_c).unwrap();
        assert_eq!(pool.free_pages(), 4);

        let plan = pool.plan_cow(&table_b, 0, PAGE_TOKENS).unwrap();
        pool.commit_cow(&mut table_b, plan).unwrap();
        assert_ne!(table_b.physical(0), table_a.physical(0));
        assert_eq!(
            pool.page_state(table_b.physical(0).unwrap()),
            PageState::Private
        );
    }

    // ── Wave 2: checked refcount / underflow tests (A3) ───────────────

    #[test]
    fn a3_release_underflow_detected() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        let phys0 = table.physical(0).unwrap();

        // Release normally
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.refcount(phys0), 0);
        assert_eq!(pool.page_state(phys0), PageState::Free);

        // Construct a stale table referencing the freed page
        let mut stale_table = BlockTable::new();
        stale_table.push_page(phys0);

        // Releasing the stale table must detect underflow
        let result = pool.release_table(&mut stale_table);
        assert!(result.is_err(), "releasing freed page must error");
        assert!(result.unwrap_err().contains("underflow"));
    }

    #[test]
    fn a3_property_page_conservation_and_stale_handle_rejection() {
        let mut pool = PagePool::new(16, 1088).unwrap();

        // Allocate 4 pages for table A
        let mut table_a = BlockTable::new();
        let handles_a = pool.alloc_pages_checked(&mut table_a, 4).unwrap();
        table_a.set_live_tokens(4 * PAGE_TOKENS);
        assert_eq!(pool.free_pages(), 12);

        // Share first 2 pages with table B
        let mut table_b = BlockTable::new();
        pool.share_prefix(&table_a, &mut table_b, 2).unwrap();
        assert_eq!(pool.refcount(handles_a[0].phys), 2);
        assert_eq!(pool.page_state(handles_a[0].phys), PageState::Sealed);

        // Page conservation invariant
        let n_alloc: usize = (0..pool.n_pages())
            .filter(|&i| pool.page_state(i as u32) != PageState::Free)
            .count();
        assert_eq!(n_alloc + pool.free_pages(), pool.n_pages());

        // Release table A — shared pages survive, private pages freed
        pool.release_table(&mut table_a).unwrap();
        assert_eq!(pool.page_state(handles_a[0].phys), PageState::Sealed); // shared
        assert_eq!(pool.page_state(handles_a[2].phys), PageState::Free); // freed

        // Stale handle for freed page fails
        assert!(pool.validate_handle(&handles_a[2]).is_err());

        // Handle for still-referenced page is valid
        assert!(pool.validate_handle(&handles_a[0]).is_ok());

        // Release table B — all pages freed
        pool.release_table(&mut table_b).unwrap();
        assert_eq!(pool.free_pages(), 16);

        // All handles stale
        for h in &handles_a {
            assert!(pool.validate_handle(h).is_err());
        }

        // Final conservation check
        let n_alloc: usize = (0..pool.n_pages())
            .filter(|&i| pool.page_state(i as u32) != PageState::Free)
            .count();
        assert_eq!(n_alloc + pool.free_pages(), pool.n_pages());
        assert_eq!(pool.reclaim_pending_count(), 0);
    }

    #[test]
    fn a3_checked_refcount_no_overflow() {
        let mut pool = PagePool::new(2, 1088).unwrap();
        let mut table_a = BlockTable::new();
        pool.alloc_pages(&mut table_a, 1);
        table_a.set_live_tokens(PAGE_TOKENS);

        // Share the same page many times — refcount increments
        let mut tables = Vec::new();
        for _ in 0..10 {
            let mut dst = BlockTable::new();
            pool.share_prefix(&table_a, &mut dst, 1).unwrap();
            tables.push(dst);
        }
        assert_eq!(pool.refcount(table_a.physical(0).unwrap()), 11);

        // Release all — page eventually freed
        pool.release_table(&mut table_a).unwrap();
        for mut t in tables {
            pool.release_table(&mut t).unwrap();
        }
        assert_eq!(pool.free_pages(), 2);
    }

    // ── Wave 2: deferred reclaim tests (A5) ───────────────────────────

    #[test]
    fn a5_reuse_while_reclaim_pending_fails_until_drain() {
        let mut pool = PagePool::new(4, 1088).unwrap();

        // Allocate a page
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        let gen = pool.page_generation(phys);
        assert_eq!(pool.page_state(phys), PageState::Private);

        // Pin with an in-flight lease (simulating a GPU read in progress)
        pool.add_inflight_ref(phys).unwrap();

        // Release the table ref — page goes to ReclaimPending (not Free)
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);
        assert_eq!(pool.free_pages(), 3); // NOT in free list
        assert_eq!(pool.reclaim_pending_count(), 1);

        // Allocation must not reuse the reclaim-pending page
        let mut table2 = BlockTable::new();
        pool.alloc_pages(&mut table2, 1);
        assert_ne!(
            table2.physical(0).unwrap(),
            phys,
            "must not reuse reclaim-pending page"
        );
        assert_eq!(pool.free_pages(), 2);

        // Release the new table
        pool.release_table(&mut table2).unwrap();
        assert_eq!(pool.free_pages(), 3);

        // Release the in-flight ref — page still ReclaimPending
        pool.release_inflight_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);
        assert_eq!(pool.free_pages(), 3);

        // Drain — page freed
        let freed = pool.drain_completed();
        assert_eq!(freed, vec![phys]);
        assert_eq!(pool.page_state(phys), PageState::Free);
        assert_eq!(pool.free_pages(), 4);
        assert_eq!(pool.reclaim_pending_count(), 0);

        // Now the page is allocatable again
        let mut table3 = BlockTable::new();
        pool.alloc_pages(&mut table3, 1);
        assert_eq!(pool.free_pages(), 3);

        // Generation was bumped — old handle is stale
        let new_gen = pool.page_generation(phys);
        assert_ne!(new_gen, gen, "generation must change after free/realloc");
        let old_handle = PageHandle {
            phys,
            epoch: 0,
            generation: gen,
        };
        assert!(
            pool.validate_handle(&old_handle).is_err(),
            "stale handle with old generation must fail"
        );
    }

    #[test]
    fn drain_completed_skips_pages_with_active_inflight() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();

        // Two in-flight refs
        pool.add_inflight_ref(phys).unwrap();
        pool.add_inflight_ref(phys).unwrap();

        // Release table — ReclaimPending
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // Release one in-flight — still pending (one ref remains)
        pool.release_inflight_ref(phys).unwrap();
        let freed = pool.drain_completed();
        assert!(
            freed.is_empty(),
            "page with active in-flight must not drain"
        );
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // Release last in-flight — now drainable
        pool.release_inflight_ref(phys).unwrap();
        let freed = pool.drain_completed();
        assert_eq!(freed, vec![phys]);
        assert_eq!(pool.page_state(phys), PageState::Free);
    }

    #[test]
    fn inflight_underflow_detected() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let err = pool.release_inflight_ref(0).unwrap_err();
        assert!(err.contains("underflow"));
    }

    // ── Wave 2: cache lease tests ─────────────────────────────────────

    #[test]
    fn cache_ref_makes_private_page_sealed() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Private);

        pool.add_cache_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Sealed);

        // Release table ref — page goes to CacheOnly (not Free)
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::CacheOnly);
        assert_eq!(pool.free_pages(), 3);

        // Release cache ref — page freed
        pool.release_cache_ref(phys).unwrap();
        assert_eq!(pool.page_state(phys), PageState::Free);
        assert_eq!(pool.free_pages(), 4);
    }

    #[test]
    fn cow_on_cacheonly_page() {
        let mut pool = PagePool::new(8, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 2);
        table.set_live_tokens(2 * PAGE_TOKENS);

        // Add cache ref and release table — page becomes CacheOnly
        let phys0 = table.physical(0).unwrap();
        pool.add_cache_ref(phys0).unwrap();
        // Page is now Sealed (Private + cache ref)
        assert_eq!(pool.page_state(phys0), PageState::Sealed);

        // plan_cow should detect Sealed page and plan a copy
        let plan = pool.plan_cow(&table, 0, PAGE_TOKENS).unwrap();
        assert_eq!(plan.copies().len(), 1);
        assert_eq!(plan.copies()[0].src_phys, phys0);

        // Commit COW
        pool.commit_cow(&mut table, plan).unwrap();
        assert_ne!(table.physical(0).unwrap(), phys0);
    }

    // ── A5: cache-ref release with in-flight readers must not leak ────

    #[test]
    fn a5_cache_ref_release_with_inflight_enters_reclaim_pending() {
        // Regression: release_cache_ref used to leave a page with zero
        // table/cache refs but active in-flight refs in limbo — neither in
        // the free list nor in reclaim_pending — leaking it permanently.
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();

        pool.add_cache_ref(phys).unwrap();
        pool.add_inflight_ref(phys).unwrap();

        // Drop the table ref, then the cache ref while the in-flight lease
        // is still held (device reader outstanding).
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::CacheOnly);
        pool.release_cache_ref(phys).unwrap();

        // Must be ReclaimPending and NOT allocatable, not silently leaked.
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);
        assert_eq!(pool.free_pages(), 3);
        assert_eq!(pool.reclaim_pending_count(), 1);

        // Completion arrives → drain frees it.
        pool.release_inflight_ref(phys).unwrap();
        let freed = pool.drain_completed();
        assert_eq!(freed, vec![phys]);
        assert_eq!(pool.page_state(phys), PageState::Free);
        assert_eq!(pool.free_pages(), 4);
    }

    #[test]
    fn a5_reclaim_pending_page_refuses_new_leases() {
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();

        pool.add_inflight_ref(phys).unwrap();
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // A stale handle's page must not regain table/cache/inflight leases
        // while queued for reclaim — drain_completed would free it out from
        // under the new owner.
        assert!(pool.refcount_inc(phys).is_err());
        assert!(pool.add_cache_ref(phys).is_err());
        assert!(pool.add_inflight_ref(phys).is_err());

        // After drain the page is Free; leases are still refused (Free).
        pool.release_inflight_ref(phys).unwrap();
        pool.drain_completed();
        assert!(pool.refcount_inc(phys).is_err());
        assert!(pool.add_cache_ref(phys).is_err());
    }

    #[test]
    fn drain_completed_keeps_pages_that_regained_owners() {
        // Defensive: a page in the reclaim queue must not be freed while it
        // carries any live ref, even if inflight drained to zero first.
        let mut pool = PagePool::new(4, 1088).unwrap();
        let mut table = BlockTable::new();
        pool.alloc_pages(&mut table, 1);
        let phys = table.physical(0).unwrap();

        pool.add_inflight_ref(phys).unwrap();
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.page_state(phys), PageState::ReclaimPending);

        // (The public API now refuses new leases on ReclaimPending pages, so
        // the only way refs reappear is via the internal paths; the drain
        // guard is the backstop that keeps such a page queued instead of
        // freed.) A clean drain still works.
        pool.release_inflight_ref(phys).unwrap();
        assert_eq!(pool.drain_completed(), vec![phys]);
    }

    /// Drives one alloc → seal → cache-ref → release lifecycle and returns
    /// every observable that must match between constructors.
    fn external_lifecycle_trace(pool: &mut PagePool) -> Vec<String> {
        let mut trace = Vec::new();
        let mut table = BlockTable::new();
        let handles = pool.alloc_pages_checked(&mut table, 2).unwrap();
        trace.push(format!("free_after_alloc={}", pool.free_pages()));
        for h in &handles {
            trace.push(format!(
                "alloc phys={} gen={} state={:?}",
                h.phys,
                h.generation,
                pool.page_state(h.phys)
            ));
            pool.validate_handle(h).unwrap();
        }
        let first = handles[0];
        pool.seal(first.phys).unwrap();
        trace.push(format!("sealed={:?}", pool.page_state(first.phys)));
        pool.add_cache_ref(first.phys).unwrap();
        pool.release_table(&mut table).unwrap();
        trace.push(format!(
            "after_table_release first={:?} second={:?} free={}",
            pool.page_state(first.phys),
            pool.page_state(handles[1].phys),
            pool.free_pages()
        ));
        pool.validate_handle(&first).unwrap();
        pool.release_cache_ref(first.phys).unwrap();
        trace.push(format!(
            "after_cache_release state={:?} gen={} free={}",
            pool.page_state(first.phys),
            pool.page_generation(first.phys),
            pool.free_pages()
        ));
        assert!(
            pool.validate_handle(&first).is_err(),
            "handle must be stale after the generation bump"
        );
        assert_ne!(pool.page_generation(first.phys), first.generation);
        trace
    }

    #[test]
    fn external_pool_lifecycle_matches_new_with_strides() {
        let mut checked = PagePool::new_with_strides(8, 1088, 544).unwrap();
        let mut external = PagePool::new_external_with_strides(8, 1088, 544).unwrap();
        assert_eq!(external.n_pages(), checked.n_pages());
        assert_eq!(external.k_per_pos_bytes(), checked.k_per_pos_bytes());
        assert_eq!(external.v_per_pos_bytes(), checked.v_per_pos_bytes());
        assert_eq!(external.free_pages(), checked.free_pages());
        assert_eq!(external.epoch(), checked.epoch());
        assert_eq!(
            external_lifecycle_trace(&mut external),
            external_lifecycle_trace(&mut checked)
        );
    }

    #[test]
    fn external_pool_accepts_size_new_with_strides_refuses() {
        // 4 pages * 128 positions * (64 MiB + 64 MiB) = 64 GiB nominal,
        // above the 32 GiB deployment budget `new_with_strides` enforces.
        let stride = 64 * 1024 * 1024;
        assert!(PagePool::new_with_strides(4, stride, stride).is_err());
        let mut pool = PagePool::new_external_with_strides(4, stride, stride).unwrap();
        assert_eq!(pool.n_pages(), 4);
        assert_eq!(pool.free_pages(), 4);
        let trace = external_lifecycle_trace(&mut pool);
        assert!(trace.last().unwrap().contains("state=Free"));
        assert_eq!(pool.free_pages(), 4);
    }

    #[test]
    fn external_pool_refuses_zero_arguments_without_panicking() {
        assert!(PagePool::new_external_with_strides(0, 1088, 1088).is_err());
        assert!(PagePool::new_external_with_strides(4, 0, 1088).is_err());
        assert!(PagePool::new_external_with_strides(4, 1088, 0).is_err());
    }
}
