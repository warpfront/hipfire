// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generation-owned resource API: owned blocks, typestate expert slots,
//! immutable table generations, window read leases and fence-gated
//! reclamation (dynamic expert residency, DESIGN §4 "Reader safety").
//!
//! The module is GPU-free. Device completion enters **only** through three
//! trust roots, each requiring `unsafe`: the `unsafe` traits [`FenceBackend`]
//! and [`PeerReaderFence`], and the `unsafe fn`
//! [`Submission::complete_fenced`](crate::window::Submission::complete_fenced),
//! whose caller vouches for the window queue's quiescence report. Every
//! completion token ([`FillComplete`], [`GenerationRetired`],
//! [`FencedWindow`], [`BlockReclaimed`], [`Quiesced`]) has private fields and
//! is minted from inside this crate after a fence (or, for [`FencedWindow`],
//! the window queue) reported completion on the right backend incarnation.
//! `Quiescence::Proven`, `CompleteWindow` and plain `bool`s are ordinary
//! values and are **never** accepted as a permit.
//!
//! # The three invariant layers (DESIGN §4 table), as implemented here
//!
//! | Layer | What this module provides |
//! |---|---|
//! | **Compile time / safe API** | A slot moves `Free → Filling → Resident → Evicting → Free` by consuming an [`OwnedSlot`] typestate (`start_fill`, `finish_fill`, `unpublish`, `reclaim`); `OwnedSlot`, [`LentBlock`], [`TableBacking`], [`OwnedTableGeneration`], [`FillTicket`] are not `Clone`; the table backing (the only writable table view) is *moved into* a generation at `seal` and only comes back from `retire`; a borrowed [`WindowReadLease`]/[`PendingWindow`] borrows the generation, so the borrow checker refuses `retire` (a move) while it lives; only a `LentBlock<Kv>` has `kv_binding`; the completion tokens have private fields; completion enters only through the `unsafe` traits and the `unsafe fn` `Submission::complete_fenced`. Leaking or forgetting a token leaks capacity (slots never reclaim), never releases early. |
//! | **Runtime checked** | Fence incarnation equals the backend's (stale/foreign fence ⇒ [`ResidencyError::Stale`] or quarantine); `(bank, slot, fill_generation)` of a [`FillComplete`] equals the slot's; per-slot generation reference counts (a slot named by several live generations reclaims only after **all** retired); `Arc` ownership uniqueness at `retire` (no outstanding lease); `GenerationId` strictly increasing per bank; `fill_generation`/epoch wrap fails closed; extents of blocks, slots and table backings pairwise disjoint; quiescence of the bank's issued fills (`Quiesced`) before a scratch loan or close; an `Unknown` fence failure permanently quarantines the affected slots/generation (the bytes are never reused and the pool can never be returned). |
//! | **Host/backend protocol (trusted boundary)** | Everything the type system cannot see: the last-read fence is recorded **after** the window's final consumer was enqueued (`WindowReadLease::enqueued`); the table upload is ordered before the first read of the next generation; a copy fence follows every payload write of its [`FillTicket`]s; the implementor of [`FenceBackend`] reports `Complete` only when the stream really passed the fence, and of [`PeerReaderFence`] only when every peer reader fenced. These are the fault-injection targets of the swap-race tests, **not** claims of this module. |
//!
//! # Lifecycle (runs as a doctest)
//!
//! ```
//! use railgun::sealed_table::generation::*;
//! /// A CPU "device": every fence completes immediately.
//! struct Dev { next: u64 }
//! // SAFETY: nothing is ever enqueued on this fake stream, so every fence is complete.
//! unsafe impl FenceBackend for Dev {
//!     fn incarnation(&self) -> u64 { 1 }
//!     fn record(&mut self) -> Result<u64, FenceFailure> { self.next += 1; Ok(self.next) }
//!     fn poll(&mut self, _f: u64) -> FenceState { FenceState::Complete }
//!     fn wait(&mut self, _f: u64) -> FenceState { FenceState::Complete }
//! }
//! let mut dev = Dev { next: 0 };
//!
//! // One 4 KiB expert block, two 1 KiB slots, one table.
//! let registry = BlockRegistry::new(0);
//! let block = registry.adopt::<Expert>(0x10_0000, 4096, 1).unwrap();
//! let (mut bank, free) = SlotBank::new(
//!     block,
//!     vec![vec![SlotSpan { offset: 0, bytes: 1024 }], vec![SlotSpan { offset: 1024, bytes: 1024 }]],
//!     false,
//! ).map_err(|(_, e)| e).unwrap();
//! let backing = bank.register_table_backing(vec![railgun::Binding { base: 0x20_0000, bytes: 64 }]).unwrap();
//! let mut staging = bank.staging();
//!
//! // Fill slot 0, publish, seal generation {epoch 0}.
//! let mut free = free.into_iter();
//! let (s0, t0) = bank.start_fill(free.next().unwrap(), 7).map_err(|(_, e)| e).unwrap();
//! let pending = bank.issue_fills(vec![t0], &mut dev, &mut |_t| Ok(())).unwrap();
//! let mut done = bank.complete_fills(pending, &mut dev, true).ok().unwrap();
//! let s0 = bank.finish_fill(s0, done.pop().unwrap()).map_err(|(_, _, e)| e).unwrap();
//! bank.publish(&mut staging, &s0).unwrap();
//! let g0 = bank.seal(&staging, backing, GenerationId { epoch: 0, sub: 0 }).map_err(|(_, e)| e).unwrap();
//!
//! // A window reads g0; its last-read fence is recorded after the final consumer.
//! let lease = acquire_window(&g0).unwrap();
//! let pending = lease.enqueued(&mut dev).map_err(|(_, f)| f).unwrap();
//! let _fenced = finish_window(pending, &mut dev).ok().unwrap();
//!
//! // g0 retires; the table backing is writable again and the token says so.
//! let (backing, retired) = bank.retire(g0, None).map_err(|(_, e)| e).unwrap();
//! let _ = backing;
//! // The resident slot is unpublished (evicting), then reclaimed against the token.
//! let e0 = bank.unpublish(&mut staging, s0).map_err(|(_, e)| e).unwrap();
//! let f0 = bank.reclaim(e0, &retired).map_err(|(_, e)| e).unwrap();
//! assert_eq!(f0.fill_generation(), 1);
//! ```

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::window::{AllocationClosure, Quiescence};
use crate::{Binding, ResourceId};

// ------------------------------------------------------------------ errors

/// Why a residency operation was refused. Every fallible consuming operation
/// returns the consumed capability back alongside it (`Err((cap, error))`),
/// so a refusal never loses or releases ownership.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ResidencyError {
    #[error("refused: {0}")]
    Refused(String),
    #[error("stale: {0}")]
    Stale(String),
    #[error("quarantined: {0}")]
    Quarantined(String),
    #[error("overlap: {0}")]
    Overlap(String),
    #[error("exhausted: {0}")]
    Exhausted(String),
}

fn refused(m: impl Into<String>) -> ResidencyError {
    ResidencyError::Refused(m.into())
}
fn stale(m: impl Into<String>) -> ResidencyError {
    ResidencyError::Stale(m.into())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether two nonempty byte ranges intersect (`u128`, so no wrap).
fn overlaps(a: Binding, b: Binding) -> bool {
    if a.bytes == 0 || b.bytes == 0 {
        return false;
    }
    let (a0, a1) = (u128::from(a.base), u128::from(a.base) + u128::from(a.bytes));
    let (b0, b1) = (u128::from(b.base), u128::from(b.base) + u128::from(b.bytes));
    a0 < b1 && b0 < a1
}

// ------------------------------------------------------------ trust boundary

/// A fence's state as reported by a [`FenceBackend`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FenceState {
    Pending,
    Complete,
    Failed(FenceFailure),
}

/// A failed fence/record/wait. `quiescence` says whether the operations
/// fenced are known never to run again (`Proven`) or may still be executing
/// (`Unknown`, which quarantines everything the fence guarded forever).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FenceFailure {
    pub message: String,
    pub quiescence: Quiescence,
}

impl FenceFailure {
    fn unknown(m: impl Into<String>) -> Self {
        Self {
            message: m.into(),
            quiescence: Quiescence::Unknown,
        }
    }
    fn proven(m: impl Into<String>) -> Self {
        Self {
            message: m.into(),
            quiescence: Quiescence::Proven,
        }
    }
}

/// The device side of completion: one stream's fences (HIP events).
///
/// # Safety
///
/// `poll`/`wait` return [`FenceState::Complete`] for fence `f` only when every
/// device operation the owner enqueued on this backend's stream **before
/// `record()` returned `f`** has finished executing, and
/// `Failed { quiescence: Proven }` only when none of it can still execute.
/// `incarnation()` changes whenever the underlying device/stream handle is
/// recreated (a fence of an older incarnation never completes anything).
///
/// Implementing it without `unsafe` does not compile:
///
/// ```compile_fail,E0200
/// use railgun::sealed_table::generation::*;
/// struct Dev;
/// impl FenceBackend for Dev {
///     fn incarnation(&self) -> u64 { 1 }
///     fn record(&mut self) -> Result<u64, FenceFailure> { Ok(1) }
///     fn poll(&mut self, _f: u64) -> FenceState { FenceState::Complete }
///     fn wait(&mut self, _f: u64) -> FenceState { FenceState::Complete }
/// }
/// ```
///
/// and with `unsafe` it does:
///
/// ```
/// use railgun::sealed_table::generation::*;
/// struct Dev;
/// unsafe impl FenceBackend for Dev {
///     fn incarnation(&self) -> u64 { 1 }
///     fn record(&mut self) -> Result<u64, FenceFailure> { Ok(1) }
///     fn poll(&mut self, _f: u64) -> FenceState { FenceState::Complete }
///     fn wait(&mut self, _f: u64) -> FenceState { FenceState::Complete }
/// }
/// ```
pub unsafe trait FenceBackend {
    fn incarnation(&self) -> u64;
    fn record(&mut self) -> Result<u64, FenceFailure>;
    fn poll(&mut self, fence: u64) -> FenceState;
    fn wait(&mut self, fence: u64) -> FenceState;
    /// The fence completed and will not be asked again (an event pool may
    /// recycle it).
    fn recycle(&mut self, _fence: u64) {}
}

/// The peer side of completion (all-rank acknowledgement protocol).
///
/// # Safety
///
/// `released_through(owner, at)` returns `k` only when every **peer** reader
/// that could name a slot of `owner` excluded from generation epoch `<= k`
/// has fenced.
pub unsafe trait PeerReaderFence {
    fn released_through(&self, owner: usize, at: u64) -> u64;
}

/// One recorded fence: private, non-`Clone`. Every consumer compares the
/// incarnation with the backend it is handed.
#[derive(Debug)]
struct Fence {
    incarnation: u64,
    id: u64,
}

fn record_fence(b: &mut dyn FenceBackend) -> Result<Fence, FenceFailure> {
    let before = b.incarnation();
    let id = b.record()?;
    let after = b.incarnation();
    if before != after {
        return Err(FenceFailure::unknown(format!(
            "backend incarnation changed from {before} to {after} while recording a fence"
        )));
    }
    Ok(Fence {
        incarnation: after,
        id,
    })
}

enum Settled {
    Complete,
    Pending,
    Failed(FenceFailure),
    Stale(String),
}

fn settle(fence: &Fence, b: &mut dyn FenceBackend, block: bool) -> Settled {
    let now = b.incarnation();
    if now != fence.incarnation {
        return Settled::Stale(format!(
            "fence {} was recorded on backend incarnation {}, the backend is incarnation {now}",
            fence.id, fence.incarnation
        ));
    }
    let state = if block {
        b.wait(fence.id)
    } else {
        b.poll(fence.id)
    };
    match state {
        FenceState::Complete => {
            b.recycle(fence.id);
            Settled::Complete
        }
        FenceState::Pending => Settled::Pending,
        FenceState::Failed(f) => Settled::Failed(f),
    }
}

// ------------------------------------------------------------------- blocks

mod private {
    pub trait Sealed {}
}

/// Marker of what a [`LentBlock`] is currently lent to (sealed).
pub trait BlockUse: private::Sealed {}

/// The block is lent to the expert slot bank.
#[derive(Clone, Copy, Debug)]
pub struct Expert;
/// The block is KV-owned: only this state has a write capability.
#[derive(Clone, Copy, Debug)]
pub struct Kv;
impl private::Sealed for Expert {}
impl private::Sealed for Kv {}
impl BlockUse for Expert {}
impl BlockUse for Kv {}

struct LiveBlock {
    id: u64,
    bytes: u64,
}

struct RegistryInner {
    rank: u32,
    next_id: u64,
    /// Live (adopted, not yet freed) blocks by base address.
    live: BTreeMap<u64, LiveBlock>,
}

/// The shared block registry of one rank. Not a capability: it only
/// guarantees that live blocks never overlap.
#[derive(Clone)]
pub struct BlockRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

impl std::fmt::Debug for BlockRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = lock(&self.inner);
        f.debug_struct("BlockRegistry")
            .field("rank", &g.rank)
            .field("live_blocks", &g.live.len())
            .finish()
    }
}

impl BlockRegistry {
    pub fn new(rank: u32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                rank,
                next_id: 1,
                live: BTreeMap::new(),
            })),
        }
    }

    /// Number of live (adopted, not freed) blocks, including leaked ones.
    pub fn live_blocks(&self) -> usize {
        lock(&self.inner).live.len()
    }

    /// Adopt one physical allocation `[base, base + bytes)` (allocation
    /// generation `alloc_generation`). Refuses null/empty/overflowing ranges
    /// and any overlap with a live (not yet freed) block of this registry.
    pub fn adopt<U: BlockUse>(
        &self,
        base: u64,
        bytes: u64,
        alloc_generation: u64,
    ) -> Result<LentBlock<U>, ResidencyError> {
        if base == 0 {
            return Err(refused("null block base"));
        }
        if bytes == 0 {
            return Err(refused("empty block"));
        }
        let Some(end) = base.checked_add(bytes) else {
            return Err(refused("block range overflows the address space"));
        };
        let mut g = lock(&self.inner);
        if let Some((&b, live)) = g.live.range(..=base).next_back() {
            if u128::from(b) + u128::from(live.bytes) > u128::from(base) {
                return Err(ResidencyError::Overlap(format!(
                    "block [{base:#x}, {end:#x}) overlaps live block {} at {b:#x}",
                    live.id
                )));
            }
        }
        if let Some((&b, live)) = g.live.range(base..).next() {
            if b < end {
                return Err(ResidencyError::Overlap(format!(
                    "block [{base:#x}, {end:#x}) overlaps live block {} at {b:#x}",
                    live.id
                )));
            }
        }
        let id = g.next_id;
        g.next_id += 1;
        let rank = g.rank;
        g.live.insert(base, LiveBlock { id, bytes });
        drop(g);
        Ok(LentBlock {
            registry: self.clone(),
            id,
            rank,
            base,
            bytes,
            alloc_generation,
            loan: 0,
            _use: PhantomData,
        })
    }

    fn deregister(&self, base: u64, id: u64) {
        let mut g = lock(&self.inner);
        if g.live.get(&base).is_some_and(|b| b.id == id) {
            g.live.remove(&base);
        }
    }
}

/// An owned physical block, lent to experts (`Expert`) or held for KV
/// (`Kv`). Not `Clone`. Dropping it without `free` leaks it in the registry
/// (a capacity leak, never an early release).
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(b: LentBlock<Expert>) { let _b2 = b.clone(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(b: LentBlock<Expert>) { let _b2 = b; }
/// ```
///
/// `kv_binding` (the KV write capability) exists only in the `Kv` state:
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(mut b: LentBlock<Expert>) { let _w = b.kv_binding(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(mut b: LentBlock<Kv>) { let _w = b.kv_binding(); }
/// ```
#[derive(Debug)]
pub struct LentBlock<U> {
    registry: BlockRegistry,
    id: u64,
    rank: u32,
    base: u64,
    bytes: u64,
    alloc_generation: u64,
    /// Bumped on every `Kv -> Expert` lend; binds a [`BlockReclaimed`] token
    /// to one loan.
    loan: u64,
    _use: PhantomData<fn() -> U>,
}

impl<U> LentBlock<U> {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn alloc_generation(&self) -> u64 {
        self.alloc_generation
    }
    /// The whole physical extent (read-only description; not a capability to
    /// write it).
    pub fn extent(&self) -> Binding {
        Binding {
            base: self.base,
            bytes: self.bytes,
        }
    }
    fn retype<V>(self) -> LentBlock<V> {
        LentBlock {
            registry: self.registry,
            id: self.id,
            rank: self.rank,
            base: self.base,
            bytes: self.bytes,
            alloc_generation: self.alloc_generation,
            loan: self.loan,
            _use: PhantomData,
        }
    }
}

impl LentBlock<Kv> {
    /// The KV write capability: exists only while the block is KV-owned.
    pub fn kv_binding(&mut self) -> Binding {
        self.extent()
    }

    /// Lend the block to the expert slot bank (a fresh loan).
    pub fn lend_to_experts(mut self) -> LentBlock<Expert> {
        self.loan += 1;
        self.retype()
    }

    /// Deregister; the caller frees the memory.
    pub fn free(self) -> Binding {
        let b = self.extent();
        self.registry.deregister(self.base, self.id);
        b
    }
}

impl LentBlock<Expert> {
    /// Deregister an expert-state block against the token its bank's `close`
    /// minted for this very loan; the caller frees the memory.
    pub fn free(
        self,
        done: BlockReclaimed,
    ) -> Result<Binding, (Self, BlockReclaimed, ResidencyError)> {
        if let Err(e) = done.check(&self) {
            return Err((self, done, e));
        }
        let b = self.extent();
        self.registry.deregister(self.base, self.id);
        Ok(b)
    }
}

/// Proof that the bank that owned a block's loan was closed with every slot
/// handed back and the device quiesced. Not `Clone`; no public constructor.
///
/// ```compile_fail,E0451
/// use railgun::sealed_table::generation::*;
/// fn f() -> BlockReclaimed { BlockReclaimed { block: 1, loan: 1 } }
/// ```
///
/// Tokens come only from [`SlotBank::close`]:
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: SlotBank, cells: Vec<SlotCell>, q: Quiesced) -> Option<BlockReclaimed> {
///     bank.close(cells, q).ok().map(|(_block, done)| done)
/// }
/// ```
#[derive(Debug)]
pub struct BlockReclaimed {
    block: u64,
    loan: u64,
}

impl BlockReclaimed {
    pub fn block_id(&self) -> u64 {
        self.block
    }
    fn check(&self, b: &LentBlock<Expert>) -> Result<(), ResidencyError> {
        if self.block == b.id && self.loan == b.loan {
            Ok(())
        } else {
            Err(stale(format!(
                "reclaim token is for block {} loan {}, not block {} loan {}",
                self.block, self.loan, b.id, b.loan
            )))
        }
    }
}

/// Return an expert-state block to KV once its bank was closed. The token
/// must be the one minted for this block's current loan.
pub fn return_to_kv(
    block: LentBlock<Expert>,
    done: BlockReclaimed,
) -> Result<LentBlock<Kv>, (LentBlock<Expert>, BlockReclaimed, ResidencyError)> {
    if let Err(e) = done.check(&block) {
        return Err((block, done, e));
    }
    Ok(block.retype())
}

// -------------------------------------------------------------------- slots

/// Slot typestate markers.
#[derive(Clone, Copy, Debug)]
pub struct Free;
#[derive(Clone, Copy, Debug)]
pub struct Filling;
#[derive(Clone, Copy, Debug)]
pub struct Resident;
#[derive(Clone, Copy, Debug)]
pub struct Evicting;

/// The exclusive capability over one slot in state `S`. Not `Clone`; only a
/// [`SlotBank`] mints one.
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(s: OwnedSlot<Free>) { let _s2 = s.clone(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(s: OwnedSlot<Free>) { let _s2 = s; }
/// ```
///
/// `start_fill` takes a `Free` slot only:
///
/// ```compile_fail,E0308
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, s: OwnedSlot<Resident>) { let _ = bank.start_fill(s, 1); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, s: OwnedSlot<Free>) { let _ = bank.start_fill(s, 1); }
/// ```
#[derive(Debug)]
pub struct OwnedSlot<S> {
    bank: u64,
    slot: u32,
    fill_generation: u64,
    occupant: u64,
    _state: PhantomData<fn() -> S>,
}

impl<S> OwnedSlot<S> {
    pub fn slot(&self) -> u32 {
        self.slot
    }
    pub fn fill_generation(&self) -> u64 {
        self.fill_generation
    }
    pub fn occupant(&self) -> u64 {
        self.occupant
    }
    fn retype<T>(self, occupant: u64, fill_generation: u64) -> OwnedSlot<T> {
        OwnedSlot {
            bank: self.bank,
            slot: self.slot,
            fill_generation,
            occupant,
            _state: PhantomData,
        }
    }
}

/// A slot in any state, as handed to [`SlotBank::close`].
#[derive(Debug)]
pub enum SlotCell {
    Free(OwnedSlot<Free>),
    Filling(OwnedSlot<Filling>),
    Resident(OwnedSlot<Resident>),
    Evicting(OwnedSlot<Evicting>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhaseView {
    Free,
    Filling,
    Resident,
    Evicting,
}

impl SlotCell {
    pub fn phase(&self) -> SlotPhaseView {
        match self {
            SlotCell::Free(_) => SlotPhaseView::Free,
            SlotCell::Filling(_) => SlotPhaseView::Filling,
            SlotCell::Resident(_) => SlotPhaseView::Resident,
            SlotCell::Evicting(_) => SlotPhaseView::Evicting,
        }
    }
    pub fn slot(&self) -> u32 {
        match self {
            SlotCell::Free(s) => s.slot,
            SlotCell::Filling(s) => s.slot,
            SlotCell::Resident(s) => s.slot,
            SlotCell::Evicting(s) => s.slot,
        }
    }
    /// `(slot, fill_generation, occupant, bank)`.
    fn ident(&self) -> (u32, u64, u64, u64) {
        match self {
            SlotCell::Free(s) => (s.slot, s.fill_generation, s.occupant, s.bank),
            SlotCell::Filling(s) => (s.slot, s.fill_generation, s.occupant, s.bank),
            SlotCell::Resident(s) => (s.slot, s.fill_generation, s.occupant, s.bank),
            SlotCell::Evicting(s) => (s.slot, s.fill_generation, s.occupant, s.bank),
        }
    }
}

macro_rules! cell_from {
    ($($state:ident),*) => {$(
        impl From<OwnedSlot<$state>> for SlotCell {
            fn from(s: OwnedSlot<$state>) -> Self {
                SlotCell::$state(s)
            }
        }
    )*};
}
cell_from!(Free, Filling, Resident, Evicting);

/// One write extent of a slot, relative to the start of the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotSpan {
    pub offset: u64,
    pub bytes: u64,
}

const MAX_SPANS: usize = 4;

/// Up to [`MAX_SPANS`] absolute extents of one slot, without allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spans {
    n: u8,
    v: [Binding; MAX_SPANS],
}

impl Spans {
    fn as_slice(&self) -> &[Binding] {
        &self.v[..usize::from(self.n)]
    }
}

/// Immutable per-bank data shared with generations (the span table) plus the
/// one mutable field a generation reads: the slot's `fill_generation`, which
/// cannot change while a live generation names the slot.
#[derive(Debug)]
struct BankShared {
    id: u64,
    spans: Box<[Spans]>,
    fill_gen: Box<[AtomicU64]>,
}

static NEXT_BANK: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
struct Rec {
    phase: SlotPhaseView,
    fill_generation: u64,
    occupant: u64,
    /// Live (sealed, not retired) generations naming the slot.
    refs: u32,
    excluded_from: u64,
    issued: bool,
    /// The slot's deferred fill was abandoned (`abandon_fills`).
    abandoned: bool,
    quarantined: bool,
}

/// A fixed-capacity bitset of the slots the next generation will name.
/// Not `Clone`; belongs to one bank.
#[derive(Debug)]
pub struct Staging {
    bank: u64,
    bits: Box<[u64]>,
    len: u32,
}

impl Staging {
    pub fn contains(&self, slot: u32) -> bool {
        let (w, b) = ((slot / 64) as usize, slot % 64);
        self.bits.get(w).is_some_and(|x| x >> b & 1 == 1)
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Identity of one published generation: `epoch` is the retirement index,
/// `sub` distinguishes re-seals inside one epoch. Ordered lexicographically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationId {
    pub epoch: u64,
    pub sub: u32,
}

/// The pointer-table allocations one generation is published through: the
/// only write capability over them. Exists only while no generation holds it
/// (`seal` consumes it, `retire` returns it). Not `Clone`.
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(b: TableBacking) { let _b2 = b.clone(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(b: TableBacking) { let _t = b.tables().len(); }
/// ```
#[derive(Debug)]
pub struct TableBacking {
    bank: u64,
    id: u64,
    tables: Vec<Binding>,
}

impl TableBacking {
    pub fn tables(&self) -> &[Binding] {
        &self.tables
    }
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// A fill authorisation: the extents the copy may write. Not `Clone`.
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(t: FillTicket) { let _t2 = t.clone(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(t: FillTicket) { let _s = t.write_spans().len(); }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct FillTicket {
    bank: u64,
    slot: u32,
    fill_generation: u64,
    occupant: u64,
    spans: Spans,
}

impl FillTicket {
    pub fn slot(&self) -> u32 {
        self.slot
    }
    pub fn fill_generation(&self) -> u64 {
        self.fill_generation
    }
    pub fn occupant(&self) -> u64 {
        self.occupant
    }
    /// Absolute extents the copy may write (and nothing else).
    pub fn write_spans(&self) -> &[Binding] {
        self.spans.as_slice()
    }
}

/// One batch of issued fills sharing a copy fence. Not `Clone`.
#[derive(Debug)]
pub struct PendingFills {
    bank: u64,
    fence: Fence,
    fills: Vec<(u32, u64)>,
}

impl PendingFills {
    /// `(slot, fill_generation)` of every fill of the batch, in issue order.
    pub fn fills(&self) -> &[(u32, u64)] {
        &self.fills
    }
}

/// Why a batch of fills did not complete.
#[derive(Debug)]
pub enum FillsNotDone {
    /// The fence is still pending (poll again / wait).
    Pending(PendingFills),
    /// The fence failed (or belongs to another backend incarnation). With
    /// `Unknown` quiescence the slots of the batch are quarantined forever.
    Failed {
        fills: Vec<(u32, u64)>,
        failure: FenceFailure,
    },
}

/// Proof that one fill's copy fence completed on the right incarnation. Not
/// `Clone`; no public constructor.
///
/// ```compile_fail,E0451
/// use railgun::sealed_table::generation::*;
/// fn f() -> FillComplete { FillComplete { bank: 1, slot: 0, fill_generation: 1 } }
/// ```
///
/// Completions come only from [`SlotBank::complete_fills`]:
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, p: PendingFills, dev: &mut dyn FenceBackend) -> Vec<FillComplete> {
///     bank.complete_fills(p, dev, true).ok().unwrap()
/// }
/// ```
#[derive(Debug)]
pub struct FillComplete {
    bank: u64,
    slot: u32,
    fill_generation: u64,
}

impl FillComplete {
    pub fn slot(&self) -> u32 {
        self.slot
    }
    pub fn fill_generation(&self) -> u64 {
        self.fill_generation
    }
}

/// Proof that the device finished every fill this bank issued before the
/// quiesce fence was recorded (bound to the bank's issue sequence number).
///
/// ```compile_fail,E0451
/// use railgun::sealed_table::generation::*;
/// fn f() -> Quiesced { Quiesced { bank: 1, seq: 0 } }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, dev: &mut dyn FenceBackend) -> Option<Quiesced> {
///     bank.quiesce(dev).ok()
/// }
/// ```
#[derive(Debug)]
pub struct Quiesced {
    bank: u64,
    seq: u64,
}

/// The whole-pool scratch loan: the borrower's write capability over the
/// block. The bank is lent (mutation refused) until `end_scratch`.
#[derive(Debug)]
pub struct ScratchLoan {
    bank: u64,
    loan: u64,
    extent: Binding,
}

impl ScratchLoan {
    /// The borrower's write capability: the whole block.
    pub fn span(&self) -> Binding {
        self.extent
    }
}

/// An issued restore (its fence is waited by `end_scratch`).
#[derive(Debug)]
pub struct PendingRestore {
    bank: u64,
    loan: u64,
    seq: u64,
    fence: Fence,
}

/// The owner of a block lent to experts: slot records, per-slot generation
/// reference counts, table backings, sealing and the scratch loan.
#[derive(Debug)]
pub struct SlotBank {
    shared: Arc<BankShared>,
    block: LentBlock<Expert>,
    recs: Vec<Rec>,
    peer: bool,
    next_backing: u64,
    /// Every registered table extent (pairwise disjoint, disjoint from the block).
    table_ranges: Vec<Binding>,
    last_sealed: Option<GenerationId>,
    /// Sealed-not-retired generations (a leaked generation never decrements).
    live_generations: u64,
    issue_seq: u64,
    lent: Option<u64>,
    loan_counter: u64,
    restore_seq: u64,
}

impl SlotBank {
    /// `slots[i]` = the write extents of slot `i` (1..=4 spans, block
    /// relative). All spans of all slots must be nonempty, pairwise disjoint
    /// and inside the block. `peer`: reclaim also requires peer release.
    /// Returns every slot as `OwnedSlot<Free>` (`fill_generation` 0).
    #[allow(clippy::type_complexity)]
    pub fn new(
        block: LentBlock<Expert>,
        slots: Vec<Vec<SlotSpan>>,
        peer: bool,
    ) -> Result<(SlotBank, Vec<OwnedSlot<Free>>), (LentBlock<Expert>, ResidencyError)> {
        let spans = match Self::validate_spans(&block, &slots) {
            Ok(s) => s,
            Err(e) => return Err((block, e)),
        };
        let id = NEXT_BANK.fetch_add(1, Ordering::Relaxed);
        let n = spans.len();
        let shared = Arc::new(BankShared {
            id,
            spans: spans.into_boxed_slice(),
            fill_gen: (0..n).map(|_| AtomicU64::new(0)).collect(),
        });
        let rec = Rec {
            phase: SlotPhaseView::Free,
            fill_generation: 0,
            occupant: 0,
            refs: 0,
            excluded_from: 0,
            issued: false,
            abandoned: false,
            quarantined: false,
        };
        let bank = SlotBank {
            shared,
            block,
            recs: vec![rec; n],
            peer,
            next_backing: 1,
            table_ranges: Vec::new(),
            last_sealed: None,
            live_generations: 0,
            issue_seq: 0,
            lent: None,
            loan_counter: 0,
            restore_seq: 0,
        };
        let free = (0..n as u32)
            .map(|slot| OwnedSlot {
                bank: id,
                slot,
                fill_generation: 0,
                occupant: 0,
                _state: PhantomData,
            })
            .collect();
        Ok((bank, free))
    }

    fn validate_spans(
        block: &LentBlock<Expert>,
        slots: &[Vec<SlotSpan>],
    ) -> Result<Vec<Spans>, ResidencyError> {
        if slots.is_empty() {
            return Err(refused("a slot bank needs at least one slot"));
        }
        if slots.len() > u32::MAX as usize {
            return Err(refused("too many slots"));
        }
        let mut out = Vec::with_capacity(slots.len());
        let mut all: Vec<(u64, u64, u32)> = Vec::new();
        for (i, s) in slots.iter().enumerate() {
            if s.is_empty() || s.len() > MAX_SPANS {
                return Err(refused(format!(
                    "slot {i} has {} spans, expected 1..={MAX_SPANS}",
                    s.len()
                )));
            }
            let mut sp = Spans {
                n: s.len() as u8,
                v: [Binding { base: 0, bytes: 0 }; MAX_SPANS],
            };
            for (k, span) in s.iter().enumerate() {
                if span.bytes == 0 {
                    return Err(refused(format!("slot {i} span {k} is empty")));
                }
                let end = span.offset.checked_add(span.bytes);
                if end.is_none_or(|e| e > block.bytes) {
                    return Err(refused(format!(
                        "slot {i} span {k} [{}, +{}) leaves the {}-byte block",
                        span.offset, span.bytes, block.bytes
                    )));
                }
                sp.v[k] = Binding {
                    base: block.base + span.offset,
                    bytes: span.bytes,
                };
                all.push((span.offset, span.bytes, i as u32));
            }
            out.push(sp);
        }
        all.sort_unstable();
        for w in all.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a.0 + a.1 > b.0 {
                return Err(ResidencyError::Overlap(format!(
                    "slot {} span [{}, +{}) overlaps slot {} span at {}",
                    a.2, a.0, a.1, b.2, b.0
                )));
            }
        }
        Ok(out)
    }

    pub fn id(&self) -> u64 {
        self.shared.id
    }
    pub fn slots(&self) -> usize {
        self.recs.len()
    }
    pub fn is_peer(&self) -> bool {
        self.peer
    }
    /// The whole block (a description, not a write capability).
    pub fn extent(&self) -> Binding {
        self.block.extent()
    }
    pub fn is_lent(&self) -> bool {
        self.lent.is_some()
    }
    /// Live generations naming `slot` (0 for an out-of-range slot).
    pub fn refs(&self, slot: u32) -> u32 {
        self.recs.get(slot as usize).map_or(0, |r| r.refs)
    }
    /// Sealed-not-retired generations of this bank.
    pub fn live_generations(&self) -> u64 {
        self.live_generations
    }
    pub fn last_sealed(&self) -> Option<GenerationId> {
        self.last_sealed
    }
    pub fn phase(&self, slot: u32) -> Option<SlotPhaseView> {
        self.recs.get(slot as usize).map(|r| r.phase)
    }
    /// The first generation epoch that no longer names an evicting `slot`.
    pub fn excluded_from_epoch(&self, slot: u32) -> Option<u64> {
        self.recs
            .get(slot as usize)
            .filter(|r| r.phase == SlotPhaseView::Evicting)
            .map(|r| r.excluded_from)
    }
    pub fn is_quarantined(&self, slot: u32) -> bool {
        self.recs.get(slot as usize).is_some_and(|r| r.quarantined)
    }

    /// Register the table backing (every pointer-table allocation one
    /// generation is published through). The bindings must be nonempty,
    /// disjoint from each other, from the block and from every earlier
    /// backing. A bank may own several (an image bank).
    pub fn register_table_backing(
        &mut self,
        tables: Vec<Binding>,
    ) -> Result<TableBacking, ResidencyError> {
        if tables.is_empty() {
            return Err(refused("a table backing needs at least one table"));
        }
        let block = self.block.extent();
        for (i, t) in tables.iter().enumerate() {
            if t.base == 0 || t.bytes == 0 || t.base.checked_add(t.bytes).is_none() {
                return Err(refused(format!("table {i} is null, empty or overflows")));
            }
            if overlaps(*t, block) {
                return Err(ResidencyError::Overlap(format!(
                    "table {i} overlaps the slot block"
                )));
            }
            if let Some(o) = self.table_ranges.iter().find(|o| overlaps(*t, **o)) {
                return Err(ResidencyError::Overlap(format!(
                    "table {i} overlaps registered table at {:#x}",
                    o.base
                )));
            }
            if let Some(j) = tables[..i].iter().position(|o| overlaps(*t, *o)) {
                return Err(ResidencyError::Overlap(format!(
                    "table {i} overlaps table {j} of the same backing"
                )));
            }
        }
        self.table_ranges.extend(tables.iter().copied());
        let id = self.next_backing;
        self.next_backing += 1;
        Ok(TableBacking {
            bank: self.shared.id,
            id,
            tables,
        })
    }

    /// An empty membership set for this bank.
    pub fn staging(&self) -> Staging {
        Staging {
            bank: self.shared.id,
            bits: vec![0u64; self.recs.len().div_ceil(64)].into_boxed_slice(),
            len: 0,
        }
    }

    fn quarantine_slot(&mut self, slot: usize) {
        let r = &mut self.recs[slot];
        r.quarantined = true;
    }

    /// Identity check of a typed slot against the bank's record.
    fn check_slot<S>(
        &self,
        s: &OwnedSlot<S>,
        phase: SlotPhaseView,
    ) -> Result<usize, ResidencyError> {
        if s.bank != self.shared.id {
            return Err(stale(format!(
                "slot belongs to bank {}, not bank {}",
                s.bank, self.shared.id
            )));
        }
        let i = s.slot as usize;
        let Some(r) = self.recs.get(i) else {
            return Err(stale(format!("slot {} is out of range", s.slot)));
        };
        if r.phase != phase {
            return Err(stale(format!(
                "slot {} is {:?}, expected {phase:?}",
                s.slot, r.phase
            )));
        }
        if r.fill_generation != s.fill_generation || r.occupant != s.occupant {
            return Err(stale(format!(
                "slot {} handle (generation {}, occupant {}) does not match the bank's (generation {}, occupant {})",
                s.slot, s.fill_generation, s.occupant, r.fill_generation, r.occupant
            )));
        }
        if r.quarantined {
            return Err(ResidencyError::Quarantined(format!(
                "slot {} is quarantined (unknown-quiescence fence failure)",
                s.slot
            )));
        }
        Ok(i)
    }

    fn check_not_lent(&self) -> Result<(), ResidencyError> {
        if self.lent.is_some() {
            Err(refused("the bank is lent as scratch"))
        } else {
            Ok(())
        }
    }

    // --- Free -> Filling

    /// `Free -> Filling`: refused while a live generation still names the
    /// slot, while the bank is lent, or if `fill_generation` would wrap.
    /// Bumps `fill_generation` by one.
    #[allow(clippy::type_complexity)]
    pub fn start_fill(
        &mut self,
        s: OwnedSlot<Free>,
        occupant: u64,
    ) -> Result<(OwnedSlot<Filling>, FillTicket), (OwnedSlot<Free>, ResidencyError)> {
        let i = match self.check_slot(&s, SlotPhaseView::Free).and_then(|i| {
            self.check_not_lent()?;
            Ok(i)
        }) {
            Ok(i) => i,
            Err(e) => return Err((s, e)),
        };
        let r = &mut self.recs[i];
        if r.refs != 0 {
            let m = format!(
                "slot {} is still named by {} live generation(s)",
                s.slot, r.refs
            );
            return Err((s, refused(m)));
        }
        let Some(next) = r.fill_generation.checked_add(1) else {
            let m = format!("slot {} fill_generation would wrap", s.slot);
            return Err((s, ResidencyError::Exhausted(m)));
        };
        r.fill_generation = next;
        r.occupant = occupant;
        r.phase = SlotPhaseView::Filling;
        r.issued = false;
        r.abandoned = false;
        self.shared.fill_gen[i].store(next, Ordering::Release);
        let ticket = FillTicket {
            bank: self.shared.id,
            slot: s.slot,
            fill_generation: next,
            occupant,
            spans: self.shared.spans[i],
        };
        Ok((s.retype(occupant, next), ticket))
    }

    /// Enqueue the copies of `tickets` (`enqueue` runs once per ticket, in
    /// order, and must write only `ticket.write_spans()`), then record ONE
    /// fence on `copy` after the last. A failed `enqueue` leaves an unknown
    /// number of copies enqueued: the batch's slots are quarantined.
    ///
    /// Every error hands the tickets back (`Err((tickets, failure))`), so a
    /// refusal never strands a slot as an unissued `Filling` that blocks
    /// `lend_scratch`/`close`: the caller retries or [`abandon_fills`]s them
    /// (tickets of quarantined slots are inert either way).
    ///
    /// [`abandon_fills`]: SlotBank::abandon_fills
    #[allow(clippy::type_complexity)]
    pub fn issue_fills(
        &mut self,
        tickets: Vec<FillTicket>,
        copy: &mut dyn FenceBackend,
        enqueue: &mut dyn FnMut(&FillTicket) -> Result<(), String>,
    ) -> Result<PendingFills, (Vec<FillTicket>, FenceFailure)> {
        if self.lent.is_some() {
            return Err((
                tickets,
                FenceFailure::proven("the bank is lent as scratch: fills refused"),
            ));
        }
        for t in &tickets {
            if t.bank != self.shared.id {
                let f = FenceFailure::proven(format!(
                    "ticket for slot {} belongs to bank {}",
                    t.slot, t.bank
                ));
                return Err((tickets, f));
            }
            let ok = self.recs.get(t.slot as usize).is_some_and(|r| {
                r.phase == SlotPhaseView::Filling
                    && !r.issued
                    && !r.quarantined
                    && r.fill_generation == t.fill_generation
                    && r.occupant == t.occupant
            });
            if !ok {
                let f = FenceFailure::proven(format!(
                    "stale ticket for slot {} generation {}",
                    t.slot, t.fill_generation
                ));
                return Err((tickets, f));
            }
        }
        // From here copies may be enqueued: any failure that cannot prove the
        // stream idle quarantines the batch.
        self.issue_seq += 1;
        let fills: Vec<(u32, u64)> = tickets
            .iter()
            .map(|t| (t.slot, t.fill_generation))
            .collect();
        for t in &tickets {
            if let Err(m) = enqueue(t) {
                for &(s, _) in &fills {
                    self.quarantine_slot(s as usize);
                }
                let f = FenceFailure::unknown(format!(
                    "enqueue of slot {} failed after earlier copies may be queued: {m}",
                    t.slot
                ));
                return Err((tickets, f));
            }
        }
        match record_fence(copy) {
            Ok(fence) => {
                for &(s, _) in &fills {
                    self.recs[s as usize].issued = true;
                }
                Ok(PendingFills {
                    bank: self.shared.id,
                    fence,
                    fills,
                })
            }
            Err(f) => {
                if f.quiescence == Quiescence::Unknown {
                    for &(s, _) in &fills {
                        self.quarantine_slot(s as usize);
                    }
                }
                Err((tickets, f))
            }
        }
    }

    /// Abandon the deferred fills of `tickets` (a frozen tail dropping its
    /// queued copies): the tickets are consumed, so nothing can ever write
    /// those slots. They stay `Filling`, are not occupied (never restored)
    /// and no longer block `lend_scratch`/`close`. A ticket that is not this
    /// bank's current unissued ticket of its slot is ignored (it cannot
    /// write anything either).
    pub fn abandon_fills(&mut self, tickets: Vec<FillTicket>) {
        for t in tickets {
            if t.bank != self.shared.id {
                continue;
            }
            if let Some(r) = self.recs.get_mut(t.slot as usize) {
                if r.phase == SlotPhaseView::Filling
                    && !r.issued
                    && r.fill_generation == t.fill_generation
                    && r.occupant == t.occupant
                {
                    r.abandoned = true;
                }
            }
        }
    }

    /// Poll (`wait == false`) or wait (`wait == true`) one batch. `Complete`
    /// yields one [`FillComplete`] per fill, `Pending` returns the batch,
    /// `Failed`/stale incarnation fails the batch (with `Unknown` quiescence,
    /// or a fence of another incarnation, its slots are quarantined forever).
    pub fn complete_fills(
        &mut self,
        batch: PendingFills,
        copy: &mut dyn FenceBackend,
        wait: bool,
    ) -> Result<Vec<FillComplete>, FillsNotDone> {
        if batch.bank != self.shared.id {
            return Err(FillsNotDone::Failed {
                failure: FenceFailure::unknown(format!(
                    "batch belongs to bank {}, not bank {}",
                    batch.bank, self.shared.id
                )),
                fills: batch.fills,
            });
        }
        match settle(&batch.fence, copy, wait) {
            Settled::Complete => Ok(batch
                .fills
                .iter()
                .map(|&(slot, fill_generation)| FillComplete {
                    bank: batch.bank,
                    slot,
                    fill_generation,
                })
                .collect()),
            Settled::Pending => Err(FillsNotDone::Pending(batch)),
            Settled::Failed(failure) => {
                if failure.quiescence == Quiescence::Unknown {
                    self.quarantine_fills(&batch.fills);
                }
                Err(FillsNotDone::Failed {
                    fills: batch.fills,
                    failure,
                })
            }
            Settled::Stale(m) => {
                self.quarantine_fills(&batch.fills);
                Err(FillsNotDone::Failed {
                    fills: batch.fills,
                    failure: FenceFailure::unknown(m),
                })
            }
        }
    }

    fn quarantine_fills(&mut self, fills: &[(u32, u64)]) {
        for &(s, g) in fills {
            if self
                .recs
                .get(s as usize)
                .is_some_and(|r| r.fill_generation == g)
            {
                self.quarantine_slot(s as usize);
            }
        }
    }

    // --- Filling -> Resident -> Evicting -> Free

    /// `Filling -> Resident`. `(bank, slot, fill_generation)` of `done` must
    /// equal the slot's, else `Stale` (the slot and token are returned).
    #[allow(clippy::type_complexity)]
    pub fn finish_fill(
        &mut self,
        s: OwnedSlot<Filling>,
        done: FillComplete,
    ) -> Result<OwnedSlot<Resident>, (OwnedSlot<Filling>, FillComplete, ResidencyError)> {
        let i = match self.check_slot(&s, SlotPhaseView::Filling) {
            Ok(i) => i,
            Err(e) => return Err((s, done, e)),
        };
        if done.bank != self.shared.id
            || done.slot != s.slot
            || done.fill_generation != s.fill_generation
        {
            let m = format!(
                "completion (bank {}, slot {}, generation {}) is not slot {} generation {} of bank {}",
                done.bank, done.slot, done.fill_generation, s.slot, s.fill_generation, self.shared.id
            );
            return Err((s, done, stale(m)));
        }
        if !self.recs[i].issued {
            return Err((s, done, refused("the fill was never issued")));
        }
        self.recs[i].phase = SlotPhaseView::Resident;
        let (o, g) = (s.occupant, s.fill_generation);
        Ok(s.retype(o, g))
    }

    fn check_staging(&self, st: &Staging) -> Result<(), ResidencyError> {
        if st.bank == self.shared.id {
            Ok(())
        } else {
            Err(refused(format!(
                "staging belongs to bank {}, not bank {}",
                st.bank, self.shared.id
            )))
        }
    }

    /// Add a resident slot to `staging` (idempotent).
    pub fn publish(
        &mut self,
        staging: &mut Staging,
        s: &OwnedSlot<Resident>,
    ) -> Result<(), ResidencyError> {
        self.check_not_lent()?;
        self.check_staging(staging)?;
        self.check_slot(s, SlotPhaseView::Resident)?;
        let (w, b) = ((s.slot / 64) as usize, s.slot % 64);
        if staging.bits[w] >> b & 1 == 0 {
            staging.bits[w] |= 1 << b;
            staging.len += 1;
        }
        Ok(())
    }

    /// `Resident -> Evicting`, removed from `staging`; records
    /// `excluded_from_epoch` = last sealed epoch + 1 (0 before any seal).
    #[allow(clippy::type_complexity)]
    pub fn unpublish(
        &mut self,
        staging: &mut Staging,
        s: OwnedSlot<Resident>,
    ) -> Result<OwnedSlot<Evicting>, (OwnedSlot<Resident>, ResidencyError)> {
        if let Err(e) = self.check_staging(staging) {
            return Err((s, e));
        }
        let i = match self.check_slot(&s, SlotPhaseView::Resident) {
            Ok(i) => i,
            Err(e) => return Err((s, e)),
        };
        let (w, b) = ((s.slot / 64) as usize, s.slot % 64);
        if staging.bits[w] >> b & 1 == 1 {
            staging.bits[w] &= !(1 << b);
            staging.len -= 1;
        }
        let excluded = self.last_sealed.map_or(0, |g| g.epoch + 1);
        let r = &mut self.recs[i];
        r.phase = SlotPhaseView::Evicting;
        r.excluded_from = excluded;
        let (o, g) = (s.occupant, s.fill_generation);
        Ok(s.retype(o, g))
    }

    /// `Evicting -> Free`. Requires every generation naming the slot retired
    /// (`refs == 0`), and for a peer bank `done.peer_through() >=
    /// excluded_from_epoch`. `done` must come from this bank. Refused while
    /// the bank is lent.
    #[allow(clippy::type_complexity)]
    pub fn reclaim(
        &mut self,
        s: OwnedSlot<Evicting>,
        done: &GenerationRetired,
    ) -> Result<OwnedSlot<Free>, (OwnedSlot<Evicting>, ResidencyError)> {
        let i = match self.check_slot(&s, SlotPhaseView::Evicting).and_then(|i| {
            self.check_not_lent()?;
            Ok(i)
        }) {
            Ok(i) => i,
            Err(e) => return Err((s, e)),
        };
        if done.bank != self.shared.id {
            let m = format!(
                "retirement token is for bank {}, not bank {}",
                done.bank, self.shared.id
            );
            return Err((s, stale(m)));
        }
        let r = &self.recs[i];
        if r.refs != 0 {
            let m = format!(
                "slot {} is still named by {} live generation(s)",
                s.slot, r.refs
            );
            return Err((s, refused(m)));
        }
        if self.peer {
            match done.peer_through {
                Some(k) if k >= r.excluded_from => {}
                Some(k) => {
                    let m = format!(
                        "peer readers released through epoch {k}, slot {} is excluded from epoch {}",
                        s.slot, r.excluded_from
                    );
                    return Err((s, refused(m)));
                }
                None => {
                    return Err((
                        s,
                        refused("a peer bank needs a retirement token carrying peer release"),
                    ));
                }
            }
        }
        let r = &mut self.recs[i];
        r.phase = SlotPhaseView::Free;
        r.occupant = 0;
        r.issued = false;
        r.abandoned = false;
        let g = s.fill_generation;
        Ok(s.retype(0, g))
    }

    // --- generations

    /// Seal `staging` into an immutable generation published through
    /// `backing`. `id` must exceed every earlier id of this bank; every
    /// member must currently be `Resident` in this bank. Each member's
    /// reference count rises by one. Refused while lent.
    pub fn seal(
        &mut self,
        staging: &Staging,
        backing: TableBacking,
        id: GenerationId,
    ) -> Result<OwnedTableGeneration, (TableBacking, ResidencyError)> {
        match self.seal_checks(staging, &backing, id) {
            Ok(()) => {}
            Err(e) => return Err((backing, e)),
        }
        let mut count = 0u32;
        for (w, &word) in staging.bits.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let slot = w * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                self.recs[slot].refs += 1;
                count += 1;
            }
        }
        self.last_sealed = Some(id);
        self.live_generations += 1;
        Ok(OwnedTableGeneration {
            core: Arc::new(GenCore {
                shared: Arc::clone(&self.shared),
                id,
                members: staging.bits.clone(),
                count,
                backing,
                quarantined: AtomicBool::new(false),
            }),
        })
    }

    fn seal_checks(
        &self,
        staging: &Staging,
        backing: &TableBacking,
        id: GenerationId,
    ) -> Result<(), ResidencyError> {
        self.check_not_lent()?;
        self.check_staging(staging)?;
        if backing.bank != self.shared.id {
            return Err(refused(format!(
                "table backing belongs to bank {}, not bank {}",
                backing.bank, self.shared.id
            )));
        }
        if id.epoch == u64::MAX {
            return Err(ResidencyError::Exhausted(
                "generation epoch would wrap".into(),
            ));
        }
        if let Some(last) = self.last_sealed {
            if id <= last {
                return Err(refused(format!(
                    "generation id {id:?} does not exceed the last sealed {last:?}"
                )));
            }
        }
        for (w, &word) in staging.bits.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let slot = w * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                let Some(r) = self.recs.get(slot) else {
                    return Err(refused(format!(
                        "staging names slot {slot} outside the bank"
                    )));
                };
                if r.phase != SlotPhaseView::Resident {
                    return Err(refused(format!(
                        "staging names slot {slot}, which is {:?}, not Resident",
                        r.phase
                    )));
                }
                if r.quarantined {
                    return Err(ResidencyError::Quarantined(format!(
                        "slot {slot} is quarantined"
                    )));
                }
                if r.refs == u32::MAX {
                    return Err(ResidencyError::Exhausted(format!(
                        "slot {slot} reference count would wrap"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Retire a generation: requires sole ownership (no outstanding lease)
    /// and no quarantine. Every member's reference count falls by one; the
    /// backing is returned (writable again) with the token. With `peers`
    /// (`Some((fence, owner, at))`) calls `fence.released_through(owner, at)`
    /// exactly once on success and stores the result in the token.
    #[allow(clippy::type_complexity)]
    pub fn retire(
        &mut self,
        g: OwnedTableGeneration,
        peers: Option<(&dyn PeerReaderFence, usize, u64)>,
    ) -> Result<(TableBacking, GenerationRetired), (OwnedTableGeneration, ResidencyError)> {
        if g.core.shared.id != self.shared.id {
            let m = format!(
                "generation belongs to bank {}, not bank {}",
                g.core.shared.id, self.shared.id
            );
            return Err((g, stale(m)));
        }
        if g.core.quarantined.load(Ordering::Acquire) {
            return Err((
                g,
                ResidencyError::Quarantined(
                    "the generation is quarantined and can never retire".into(),
                ),
            ));
        }
        if Arc::strong_count(&g.core) != 1 || Arc::weak_count(&g.core) != 0 {
            return Err((
                g,
                refused("a window read lease on the generation is still outstanding"),
            ));
        }
        let peer_through = peers.map(|(f, owner, at)| f.released_through(owner, at));
        let core = match Arc::try_unwrap(g.core) {
            Ok(c) => c,
            Err(core) => {
                return Err((
                    OwnedTableGeneration { core },
                    refused("a window read lease on the generation is still outstanding"),
                ));
            }
        };
        for (w, &word) in core.members.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let slot = w * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                self.recs[slot].refs -= 1;
            }
        }
        self.live_generations -= 1;
        Ok((
            core.backing,
            GenerationRetired {
                bank: self.shared.id,
                id: core.id,
                peer_through,
            },
        ))
    }

    // --- scratch loan

    /// Device-wide quiescence of this bank's issued fills: record + wait on a
    /// device-scope backend. Binds the bank's issue sequence number at
    /// record time.
    pub fn quiesce(&mut self, device: &mut dyn FenceBackend) -> Result<Quiesced, FenceFailure> {
        let seq = self.issue_seq;
        let fence = record_fence(device)?;
        match settle(&fence, device, true) {
            Settled::Complete => Ok(Quiesced {
                bank: self.shared.id,
                seq,
            }),
            Settled::Pending => Err(FenceFailure::unknown("the device wait returned pending")),
            Settled::Failed(f) => Err(f),
            Settled::Stale(m) => Err(FenceFailure::unknown(m)),
        }
    }

    fn check_quiesced(&self, q: &Quiesced) -> Result<(), ResidencyError> {
        if q.bank != self.shared.id {
            return Err(stale(format!(
                "quiescence is for bank {}, not bank {}",
                q.bank, self.shared.id
            )));
        }
        if q.seq != self.issue_seq {
            return Err(stale(format!(
                "quiescence covers issue {} but the bank issued fills up to {}",
                q.seq, self.issue_seq
            )));
        }
        Ok(())
    }

    /// Whole-pool scratch loan: refused while any generation is live (sealed
    /// and not retired), while a slot is quarantined or has an unissued
    /// ticket, while already lent, or if `q` is stale. While lent
    /// seal/start_fill/issue_fills/publish/reclaim are refused.
    pub fn lend_scratch(&mut self, q: Quiesced) -> Result<ScratchLoan, ResidencyError> {
        self.check_quiesced(&q)?;
        self.check_not_lent()?;
        if self.live_generations != 0 {
            return Err(refused(format!(
                "{} generation(s) are still live",
                self.live_generations
            )));
        }
        for (i, r) in self.recs.iter().enumerate() {
            if r.refs != 0 {
                return Err(refused(format!(
                    "slot {i} is still named by a live generation"
                )));
            }
            if r.quarantined {
                return Err(ResidencyError::Quarantined(format!(
                    "slot {i} is quarantined: its block can never be lent"
                )));
            }
            if r.phase == SlotPhaseView::Filling && !r.issued && !r.abandoned {
                return Err(refused(format!(
                    "slot {i} has a fill ticket that was never issued"
                )));
            }
        }
        self.loan_counter += 1;
        self.lent = Some(self.loan_counter);
        Ok(ScratchLoan {
            bank: self.shared.id,
            loan: self.loan_counter,
            extent: self.block.extent(),
        })
    }

    fn occupied_slots(&self) -> impl Iterator<Item = (usize, &Rec)> {
        self.recs.iter().enumerate().filter(|(_, r)| match r.phase {
            SlotPhaseView::Free => false,
            SlotPhaseView::Filling => r.issued,
            SlotPhaseView::Resident | SlotPhaseView::Evicting => true,
        })
    }

    /// `(slot, occupant, fill_generation)` of every occupied slot
    /// (Resident, Evicting, or Filling with its copy issued), ascending.
    pub fn occupied(&self) -> Vec<(u32, u64, u64)> {
        self.occupied_slots()
            .map(|(i, r)| (i as u32, r.occupant, r.fill_generation))
            .collect()
    }

    /// Restore after the scratch epoch: `enqueue(slot, occupant, write_spans)`
    /// once per occupied slot, ascending, then ONE fence on `copy`. May be
    /// re-issued (the restore is idempotent); only the latest is accepted by
    /// `end_scratch`.
    pub fn issue_restore(
        &mut self,
        loan: &ScratchLoan,
        copy: &mut dyn FenceBackend,
        enqueue: &mut dyn FnMut(u32, u64, &[Binding]) -> Result<(), String>,
    ) -> Result<PendingRestore, FenceFailure> {
        if loan.bank != self.shared.id || self.lent != Some(loan.loan) {
            return Err(FenceFailure::proven(
                "the scratch loan is not this bank's current loan",
            ));
        }
        for (i, r) in self.recs.iter().enumerate() {
            let occupied = match r.phase {
                SlotPhaseView::Free => false,
                SlotPhaseView::Filling => r.issued,
                SlotPhaseView::Resident | SlotPhaseView::Evicting => true,
            };
            if !occupied {
                continue;
            }
            let spans = self.shared.spans[i];
            if let Err(m) = enqueue(i as u32, r.occupant, spans.as_slice()) {
                return Err(FenceFailure::unknown(format!(
                    "restore enqueue of slot {i} failed after earlier copies may be queued: {m}"
                )));
            }
        }
        let fence = record_fence(copy)?;
        self.restore_seq += 1;
        Ok(PendingRestore {
            bank: self.shared.id,
            loan: loan.loan,
            seq: self.restore_seq,
            fence,
        })
    }

    /// Wait the restore fence and end the loan. On error the loan and the
    /// restore are returned; a fence of another incarnation is `Stale` (issue
    /// the restore again), an unknown-quiescence failure quarantines every
    /// occupied slot.
    #[allow(clippy::type_complexity)]
    pub fn end_scratch(
        &mut self,
        loan: ScratchLoan,
        restore: PendingRestore,
        copy: &mut dyn FenceBackend,
    ) -> Result<(), (ScratchLoan, PendingRestore, ResidencyError)> {
        if loan.bank != self.shared.id || self.lent != Some(loan.loan) {
            return Err((
                loan,
                restore,
                stale("the scratch loan is not this bank's current loan"),
            ));
        }
        if restore.bank != self.shared.id
            || restore.loan != loan.loan
            || restore.seq != self.restore_seq
        {
            return Err((
                loan,
                restore,
                stale("the restore is not this loan's latest restore"),
            ));
        }
        match settle(&restore.fence, copy, true) {
            Settled::Complete => {
                self.lent = None;
                Ok(())
            }
            Settled::Pending => Err((loan, restore, refused("the restore fence is still pending"))),
            Settled::Stale(m) => Err((loan, restore, stale(m))),
            Settled::Failed(f) if f.quiescence == Quiescence::Proven => Err((
                loan,
                restore,
                refused(format!(
                    "the restore failed ({}); issue it again",
                    f.message
                )),
            )),
            Settled::Failed(f) => {
                let occupied: Vec<usize> = self.occupied_slots().map(|(i, _)| i).collect();
                for i in occupied {
                    self.quarantine_slot(i);
                }
                Err((
                    loan,
                    restore,
                    ResidencyError::Quarantined(format!(
                        "the restore failed with unknown quiescence: {}",
                        f.message
                    )),
                ))
            }
        }
    }

    // --- teardown

    /// Teardown / KV return: every cell handed back, no live generation, no
    /// quarantined slot, not lent, `q` current. Yields the block with the
    /// token [`return_to_kv`]/[`LentBlock::free`] need.
    #[allow(clippy::type_complexity)]
    pub fn close(
        self,
        cells: Vec<SlotCell>,
        q: Quiesced,
    ) -> Result<(LentBlock<Expert>, BlockReclaimed), (SlotBank, Vec<SlotCell>, ResidencyError)>
    {
        if let Err(e) = self.close_checks(&cells, &q) {
            return Err((self, cells, e));
        }
        let SlotBank { block, .. } = self;
        let done = BlockReclaimed {
            block: block.id,
            loan: block.loan,
        };
        Ok((block, done))
    }

    fn close_checks(&self, cells: &[SlotCell], q: &Quiesced) -> Result<(), ResidencyError> {
        self.check_quiesced(q)?;
        self.check_not_lent()?;
        if self.live_generations != 0 {
            return Err(refused(format!(
                "{} generation(s) are still live",
                self.live_generations
            )));
        }
        if cells.len() != self.recs.len() {
            return Err(refused(format!(
                "{} cells handed back for {} slots",
                cells.len(),
                self.recs.len()
            )));
        }
        let mut seen = vec![false; self.recs.len()];
        for c in cells {
            let (slot, generation, occupant, bank) = c.ident();
            if bank != self.shared.id {
                return Err(stale(format!(
                    "cell for slot {slot} belongs to bank {bank}"
                )));
            }
            let i = slot as usize;
            let Some(r) = self.recs.get(i) else {
                return Err(stale(format!("cell for slot {slot} is out of range")));
            };
            if std::mem::replace(&mut seen[i], true) {
                return Err(refused(format!("slot {slot} handed back twice")));
            }
            if r.phase != c.phase() || r.fill_generation != generation || r.occupant != occupant {
                return Err(stale(format!(
                    "cell for slot {slot} does not match the bank's record"
                )));
            }
        }
        for (i, r) in self.recs.iter().enumerate() {
            if r.refs != 0 {
                return Err(refused(format!(
                    "slot {i} is still named by a live generation"
                )));
            }
            if r.quarantined {
                return Err(ResidencyError::Quarantined(format!(
                    "slot {i} is quarantined: the block can never be returned"
                )));
            }
            if r.phase == SlotPhaseView::Filling && !r.issued && !r.abandoned {
                return Err(refused(format!(
                    "slot {i} has an outstanding fill ticket (neither issued nor abandoned)"
                )));
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------- generations

/// Proof that a generation retired (every window lease ended, the generation
/// was uniquely owned and not quarantined). Carries the peer release stored
/// at retirement. Not `Clone`; no public constructor.
///
/// ```compile_fail,E0451
/// use railgun::sealed_table::generation::*;
/// fn f() -> GenerationRetired {
///     GenerationRetired { bank: 1, id: GenerationId { epoch: 0, sub: 0 }, peer_through: None }
/// }
/// ```
///
/// Tokens come only from [`SlotBank::retire`]:
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, g: OwnedTableGeneration) -> Option<GenerationRetired> {
///     bank.retire(g, None).ok().map(|(_backing, done)| done)
/// }
/// ```
#[derive(Debug)]
pub struct GenerationRetired {
    bank: u64,
    id: GenerationId,
    peer_through: Option<u64>,
}

impl GenerationRetired {
    pub fn bank(&self) -> u64 {
        self.bank
    }
    pub fn generation(&self) -> GenerationId {
        self.id
    }
    /// `released_through` as observed at retirement (`None` without peers).
    pub fn peer_through(&self) -> Option<u64> {
        self.peer_through
    }
}

#[derive(Debug)]
struct GenCore {
    shared: Arc<BankShared>,
    id: GenerationId,
    members: Box<[u64]>,
    count: u32,
    backing: TableBacking,
    quarantined: AtomicBool,
}

impl GenCore {
    fn quarantine(&self) {
        self.quarantined.store(true, Ordering::Release);
    }
}

/// An immutable published generation: its members, and the table backing it
/// holds. Not `Clone`. Dropping it without [`SlotBank::retire`] leaks its
/// slot references (the slots never reclaim).
///
/// ```compile_fail,E0599
/// use railgun::sealed_table::generation::*;
/// fn f(g: OwnedTableGeneration) { let _g2 = g.clone(); }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(g: OwnedTableGeneration) { let _g2 = g; }
/// ```
///
/// A borrowed lease borrows the generation: retiring (moving) it while the
/// lease is alive does not compile.
///
/// ```compile_fail,E0505
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, g: OwnedTableGeneration, dev: &mut dyn FenceBackend) {
///     let lease = acquire_window(&g).unwrap();
///     let _ = bank.retire(g, None);
///     let pending = lease.enqueued(dev).ok().unwrap();
///     let _ = finish_window(pending, dev);
/// }
/// ```
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(bank: &mut SlotBank, g: OwnedTableGeneration, dev: &mut dyn FenceBackend) {
///     let lease = acquire_window(&g).unwrap();
///     let pending = lease.enqueued(dev).ok().unwrap();
///     let _ = finish_window(pending, dev);
///     let _ = bank.retire(g, None);
/// }
/// ```
#[derive(Debug)]
pub struct OwnedTableGeneration {
    core: Arc<GenCore>,
}

impl OwnedTableGeneration {
    pub fn id(&self) -> GenerationId {
        self.core.id
    }
    pub fn bank(&self) -> u64 {
        self.core.shared.id
    }
    pub fn contains(&self, slot: u32) -> bool {
        let (w, b) = ((slot / 64) as usize, slot % 64);
        self.core.members.get(w).is_some_and(|x| x >> b & 1 == 1)
    }
    /// Member count.
    pub fn len(&self) -> u32 {
        self.core.count
    }
    pub fn is_empty(&self) -> bool {
        self.core.count == 0
    }
    pub fn is_quarantined(&self) -> bool {
        self.core.quarantined.load(Ordering::Acquire)
    }
    /// Test probe of the quarantine flag that outlives the generation.
    #[cfg(test)]
    pub(crate) fn quarantine_probe(&self) -> impl Fn() -> bool {
        let core = Arc::clone(&self.core);
        move || core.quarantined.load(Ordering::Acquire)
    }
    /// The eager closure entrypoint: everything a window reading this
    /// generation reads.
    pub fn closure(&self) -> GenerationClosure<'_> {
        GenerationClosure { core: &self.core }
    }
}

/// A closure item the caller maps to a railgun [`ResourceId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotOrTable {
    /// The `i`-th table of the generation's backing.
    Table(usize),
    /// A member slot (all its spans share one resource: the slot extent).
    Slot(u32),
}

/// One member slot as read through a generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRead<'a> {
    pub bank: u64,
    pub slot: u32,
    pub fill_generation: u64,
    pub spans: &'a [Binding],
}

/// Iterator over a generation's members (no allocation).
#[derive(Debug)]
pub struct SlotReads<'a> {
    core: &'a GenCore,
    word: usize,
    bits: u64,
}

impl<'a> Iterator for SlotReads<'a> {
    type Item = SlotRead<'a>;
    fn next(&mut self) -> Option<SlotRead<'a>> {
        while self.bits == 0 {
            self.word += 1;
            if self.word >= self.core.members.len() {
                return None;
            }
            self.bits = self.core.members[self.word];
        }
        let slot = self.word * 64 + self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1;
        let shared: &'a BankShared = &self.core.shared;
        Some(SlotRead {
            bank: shared.id,
            slot: slot as u32,
            fill_generation: shared.fill_gen[slot].load(Ordering::Acquire),
            spans: shared.spans[slot].as_slice(),
        })
    }
}

/// The reachable-resource closure of one generation: the table backing and
/// every member slot, computed eagerly from the sealed membership (never
/// guessed from predicted top-k).
#[derive(Clone, Copy, Debug)]
pub struct GenerationClosure<'a> {
    core: &'a GenCore,
}

impl<'a> GenerationClosure<'a> {
    pub fn generation(&self) -> GenerationId {
        self.core.id
    }
    /// The generation's table allocations (read by every window).
    pub fn table_reads(&self) -> &'a [Binding] {
        &self.core.backing.tables
    }
    /// Every member slot with its extents, ascending.
    pub fn slot_reads(&self) -> SlotReads<'a> {
        SlotReads {
            core: self.core,
            word: 0,
            bits: self.core.members.first().copied().unwrap_or(0),
        }
    }
    /// Whether a write to `b` overlaps any read of the generation.
    pub fn writes_conflict(&self, b: Binding) -> bool {
        self.table_reads().iter().any(|t| overlaps(*t, b))
            || self
                .slot_reads()
                .any(|s| s.spans.iter().any(|sp| overlaps(*sp, b)))
    }
    /// Add every table and member slot to `c.reads` through `resource_of`.
    /// An item `resource_of` does not map is refused (the closure is never
    /// silently narrowed); `c` is unchanged on error.
    pub fn declare_into(
        &self,
        mut resource_of: impl FnMut(SlotOrTable) -> Option<ResourceId>,
        c: &mut AllocationClosure,
    ) -> Result<(), ResidencyError> {
        let mut found =
            Vec::with_capacity(self.core.backing.tables.len() + self.core.count as usize);
        for i in 0..self.core.backing.tables.len() {
            let item = SlotOrTable::Table(i);
            let Some(r) = resource_of(item) else {
                return Err(refused(format!("no resource declared for {item:?}")));
            };
            found.push(r);
        }
        for s in self.slot_reads() {
            let item = SlotOrTable::Slot(s.slot);
            let Some(r) = resource_of(item) else {
                return Err(refused(format!("no resource declared for {item:?}")));
            };
            found.push(r);
        }
        c.reads.extend(found);
        Ok(())
    }
}

// -------------------------------------------------------------------- leases

/// What a lease holds: a shared core (borrowed form, the borrow is the
/// `'g`) or the generation itself (owned form).
#[derive(Debug)]
enum Form {
    Shared(Arc<GenCore>),
    Owned(OwnedTableGeneration),
}

impl Form {
    fn core(&self) -> &GenCore {
        match self {
            Form::Shared(c) => c,
            Form::Owned(g) => &g.core,
        }
    }
}

/// A read lease on one generation for one window. Dropping it (or its
/// [`PendingWindow`]) without finishing marks the generation quarantined: it
/// can never retire and its slots never reclaim.
#[derive(Debug)]
pub struct WindowReadLease<'g> {
    form: Option<Form>,
    _borrow: PhantomData<&'g OwnedTableGeneration>,
}

/// Borrow `g` for one window. Refused when `g` is quarantined. The borrow
/// keeps the compiler from moving/retiring `g`; the `Arc` count keeps the
/// runtime from retiring a leaked lease.
pub fn acquire_window<'g>(
    g: &'g OwnedTableGeneration,
) -> Result<WindowReadLease<'g>, ResidencyError> {
    if g.is_quarantined() {
        return Err(ResidencyError::Quarantined(
            "the generation is quarantined".into(),
        ));
    }
    Ok(WindowReadLease {
        form: Some(Form::Shared(Arc::clone(&g.core))),
        _borrow: PhantomData,
    })
}

/// Move `g` into a `'static` lease (a window that outlives the borrow, e.g.
/// a `Submission`). Get it back from [`FencedWindow::into_generation`].
pub fn acquire_window_owned(
    g: OwnedTableGeneration,
) -> Result<WindowReadLease<'static>, (OwnedTableGeneration, ResidencyError)> {
    if g.is_quarantined() {
        return Err((
            g,
            ResidencyError::Quarantined("the generation is quarantined".into()),
        ));
    }
    Ok(WindowReadLease {
        form: Some(Form::Owned(g)),
        _borrow: PhantomData,
    })
}

impl<'g> WindowReadLease<'g> {
    fn core(&self) -> &GenCore {
        self.form
            .as_ref()
            .expect("a live lease holds its generation")
            .core()
    }

    pub fn generation(&self) -> GenerationId {
        self.core().id
    }

    pub fn closure(&self) -> GenerationClosure<'_> {
        GenerationClosure { core: self.core() }
    }

    /// Record the last-read fence **after** the window's final consumer was
    /// enqueued on `backend`'s stream. On error the lease is returned
    /// (record again once the failure is handled).
    pub fn enqueued(
        mut self,
        backend: &mut dyn FenceBackend,
    ) -> Result<PendingWindow<'g>, (Self, FenceFailure)> {
        match record_fence(backend) {
            Err(f) => Err((self, f)),
            Ok(fence) => {
                let form = self.form.take();
                Ok(PendingWindow {
                    form,
                    fence,
                    _borrow: PhantomData,
                })
            }
        }
    }

    /// Release a lease that never reached a queue (nothing can read through
    /// it): no quarantine. For `window.rs`'s refusal-before-submit paths.
    pub(crate) fn discard_unused(mut self) {
        self.form = None;
    }

    /// The queue proved quiescent (`Quiescence::Proven` from the existing
    /// submission completion): the window is fenced. For `window.rs` only.
    pub(crate) fn into_fenced(mut self) -> FencedWindow {
        let form = self.form.take().expect("a live lease holds its generation");
        FencedWindow::from_form(form)
    }
}

impl Drop for WindowReadLease<'_> {
    fn drop(&mut self) {
        if let Some(f) = &self.form {
            f.core().quarantine();
        }
    }
}

/// A lease plus its last-read fence. Dropping it unfinished quarantines the
/// generation.
#[derive(Debug)]
pub struct PendingWindow<'g> {
    form: Option<Form>,
    fence: Fence,
    _borrow: PhantomData<&'g OwnedTableGeneration>,
}

impl PendingWindow<'_> {
    pub fn generation(&self) -> GenerationId {
        self.form
            .as_ref()
            .expect("a pending window holds its generation")
            .core()
            .id
    }
}

impl Drop for PendingWindow<'_> {
    fn drop(&mut self) {
        if let Some(f) = &self.form {
            f.core().quarantine();
        }
    }
}

/// Proof that no reader of the window can still run. Not `Clone`/`Copy`;
/// private constructor. The owned form carries the generation back.
///
/// ```compile_fail,E0451
/// use railgun::sealed_table::generation::*;
/// fn f() -> FencedWindow {
///     FencedWindow { id: GenerationId { epoch: 0, sub: 0 }, bank: 1, owned: None }
/// }
/// ```
///
/// Fenced windows come only from [`finish_window`]/[`poll_window`]:
///
/// ```
/// use railgun::sealed_table::generation::*;
/// fn f(p: PendingWindow<'_>, dev: &mut dyn FenceBackend) -> Option<FencedWindow> {
///     finish_window(p, dev).ok()
/// }
/// ```
#[derive(Debug)]
pub struct FencedWindow {
    id: GenerationId,
    bank: u64,
    owned: Option<OwnedTableGeneration>,
}

impl FencedWindow {
    fn from_form(form: Form) -> Self {
        let (id, bank) = (form.core().id, form.core().shared.id);
        let owned = match form {
            Form::Shared(_) => None,
            Form::Owned(g) => Some(g),
        };
        Self { id, bank, owned }
    }
    pub fn generation(&self) -> GenerationId {
        self.id
    }
    pub fn bank(&self) -> u64 {
        self.bank
    }
    /// The generation of an owned-form window (`None` for a borrowed one,
    /// whose owner never gave it up). Dropping it instead keeps its slot
    /// references leaked: never an early release.
    pub fn into_generation(self) -> Option<OwnedTableGeneration> {
        self.owned
    }
}

/// A window whose reader fence failed with unknown quiescence (or belongs to
/// a recreated backend): its generation can never retire.
#[derive(Debug)]
pub struct QuarantinedWindow {
    id: GenerationId,
    bank: u64,
    failure: FenceFailure,
    _keep: Form,
}

impl QuarantinedWindow {
    pub fn generation(&self) -> GenerationId {
        self.id
    }
    pub fn bank(&self) -> u64 {
        self.bank
    }
    pub fn failure(&self) -> &FenceFailure {
        &self.failure
    }
}

#[derive(Debug)]
pub enum PendingOrQuarantined<'g> {
    Pending(PendingWindow<'g>),
    Quarantined(QuarantinedWindow),
}

fn settle_window<'g>(
    mut p: PendingWindow<'g>,
    backend: &mut dyn FenceBackend,
    block: bool,
) -> Result<FencedWindow, PendingOrQuarantined<'g>> {
    let quarantine = |mut p: PendingWindow<'g>, failure: FenceFailure| {
        let form = p
            .form
            .take()
            .expect("a pending window holds its generation");
        form.core().quarantine();
        PendingOrQuarantined::Quarantined(QuarantinedWindow {
            id: form.core().id,
            bank: form.core().shared.id,
            failure,
            _keep: form,
        })
    };
    match settle(&p.fence, backend, block) {
        Settled::Pending => Err(PendingOrQuarantined::Pending(p)),
        Settled::Complete => {
            let form = p
                .form
                .take()
                .expect("a pending window holds its generation");
            Ok(FencedWindow::from_form(form))
        }
        Settled::Failed(f) if f.quiescence == Quiescence::Proven => {
            let form = p
                .form
                .take()
                .expect("a pending window holds its generation");
            Ok(FencedWindow::from_form(form))
        }
        Settled::Failed(f) => Err(quarantine(p, f)),
        Settled::Stale(m) => Err(quarantine(p, FenceFailure::unknown(m))),
    }
}

/// Blocking wait of the window's last-read fence. `Complete` (or a
/// `Proven` failure) fences the window; a pending result returns it;
/// unknown quiescence or a fence of another backend incarnation quarantines
/// the generation.
pub fn finish_window<'g>(
    p: PendingWindow<'g>,
    backend: &mut dyn FenceBackend,
) -> Result<FencedWindow, PendingOrQuarantined<'g>> {
    settle_window(p, backend, true)
}

/// Non-blocking [`finish_window`].
pub fn poll_window<'g>(
    p: PendingWindow<'g>,
    backend: &mut dyn FenceBackend,
) -> Result<FencedWindow, PendingOrQuarantined<'g>> {
    settle_window(p, backend, false)
}

#[cfg(test)]
mod tests;
