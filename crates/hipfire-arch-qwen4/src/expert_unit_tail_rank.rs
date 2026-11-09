// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One EP rank's D2 tail: the device part of [`crate::expert_unit_tail`].
//!
//! Owns, per rank: the tail pool (ONE device allocation holding two blocks laid out like a unit
//! arena, lendable for a borrow epoch: [`UnitTailRank::borrow_suspend`]), a copy
//! stream with its completion events, the per-window usage log, and the rank's
//! DECODE pointer tables' byte images. Only those tables are ever rewritten: an
//! entry is the unit's canonical pointer (pinned host arena or the static core's
//! VRAM arena, exactly what D1 uploaded) or the address of a completed tail slot
//! holding a byte copy of the same unit. Host-side views and the sealed binding never see a
//! slot. With the prefill flag the chunk (prefill) tables follow the publications of the
//! shared owned units, and the long-prefill unit stage reads the live slots at each staged
//! forward's start ([`UnitTailRank::live_slots`]; `gpu_forward::slab_stage::UnitSlotStage`
//! names them in its stage tables and skips their bytes in its H2D copy). It only ever reads
//! the published map: a slot is overwritten only by a fill issued after a retirement's table
//! upload, which no staged forward straddles.
//!
//! # Window protocol (rank thread; the only cross-rank state is the PEER board)
//!
//! * [`UnitTailRank::note_call`] (the row-gather hook, once per MoE layer
//!   call): a stream-ordered D2D of the call's routed ids into the log.
//! * [`UnitTailRank::step_end`] (after the window's forward and its D2H sync): the retirement
//!   cadence ([`crate::expert_unit_tail::RetireCadence`]) decides from the window's routed-id
//!   count (rank-independent) whether this window ends a group; if not it returns at once (no
//!   sync, no D2H) and the next window keeps appending to the log. At a group end:
//!   one D2H of the log, the window's distinct units against the slots live at
//!   its START, then [`crate::expert_unit_tail::TailPolicy::retire_windows`], the touched
//!   tables' upload on the compute stream, and only then the new fills on the
//!   copy stream (it waits for the upload, so no copy can overwrite a slot a
//!   table still names). Completed fills are published at the next retirement.
//! * Fill overlap (`HIPFIRE_EP_UNIT_TAIL_FILL_OVERLAP`, default on): the retirement makes the
//!   copy stream wait for the table upload but does NOT enqueue the fills yet; the first
//!   row-gather call of the next forward does, so the copies run under that forward instead of
//!   inside the lane window's `prepare_commit` device-wide sync that follows the retirement. The
//!   next retirement waits for the last batch's event before it polls, so every fill issued so
//!   far is complete when the policy steps: the policy sees exactly the completions the
//!   device-wide sync gave it. A deferred fill targets a `Filling` slot no table and no staged
//!   forward names (published slots only), so issuing it later changes no read. `=0` issues
//!   the fills at the retirement and polls them (the previous behaviour).
//! * With `HIPFIRE_EP_UNIT_TAIL_PEER` (the protocol and its safety argument are the module docs
//!   of [`crate::expert_unit_tail`]) the retirement also posts this rank's publications to the
//!   shared [`PeerBoard`], takes every delta the peers posted so far, applies it to the
//!   full-table image of the peer-owned entries [`peer_shared`] admits, uploads the touched full
//!   tables with its own table uploads (compute stream, before its own fills) and only then
//!   records the ACK the owners release quarantined slots on. Teardown frees the pools with the
//!   mesh, like the static cores peer tables name.
//!
//! # Generations, leases and fences (the slot bank)
//!
//! The pool is the slot bank's block ([`crate::expert_unit_tail::TailPolicy`] owns the bank):
//! the tail publishes ONE immutable table generation per retirement (epoch = the retirement
//! index, the warm start is epoch 0, a borrow resume re-seals `sub + 1`) through the tables'
//! backing, and holds a window lease on it from its publication until the next retirement's
//! fence. Retirement `j`:
//!
//! 1. fences the window of generation `j - 1` (`finish_window` on a [`RankCompute`] whose `wait`
//!    is the `stream_synchronize(active)` the protocol always did), then the D2H of the log;
//! 2. polls the pending fill batches on the copy stream ([`RankCopy`] over the batch events:
//!    `event_query`, `event_synchronize` under fill overlap, events recycled) into completion
//!    tokens;
//! 3. retires generation `j - 1` (PEER mode: the one `released_through` call of the
//!    retirement), which hands back the table backing: the tables' write capability, held
//!    only until step 5;
//! 4. steps the policy with the retirement token (publish completions, evict, reclaim, start
//!    fills);
//! 5. writes the table images and uploads the changed ranges (the host images are the backing's
//!    host side), posts / takes / applies the peer deltas;
//! 6. seals generation `j` over the tables, leases it, records the PEER ack;
//! 7. records the tables-ready event, makes the copy stream wait for it and issues the fills
//!    (now, or deferred to the next forward under fill overlap): two `memcpy_htod_async` per
//!    ticket, restricted to the ticket's write spans, then ONE event on the copy stream.
//!
//! The backends are borrowed, `!Send` capability structs minted on the rank thread for one call
//! ([`RankCompute`], [`RankCopy`], [`RankDevice`]). A failed or unfinished fence quarantines
//! what it guarded; a borrow epoch lends the whole pool as scratch (a scratch loan, refused
//! while any generation is live) and its close re-copies every occupied slot from the canonical
//! host bytes; release frees the pool only after the bank closed, and leaks it (logged) when it
//! could not.
//!
//! Any device error freezes the tail at its last uploaded table and logs once;
//! it never fails a decode (every table state is valid: canonical, or a live
//! slot with complete bytes). The one exception is the pins lever's own fault (below).
//!
//! # Dynamic pins (`HIPFIRE_EP_UNIT_TAIL_PINS`)
//!
//! With the pins lever the tail runs on PINNED unit layers (D1.9) instead of static owners, and
//! every retirement also steps the rank's [`PinProtocol`] (the protocol, its identity and
//! visibility arguments and its failure path are the module docs of
//! [`crate::expert_unit_tail::pins`]): the rank posts the publications / evictions its
//! retirement made (an empty marker when it made none), applies every rank's posts with stamp
//! `<= r - 2` to its pin planes and uploads the changed plane bytes on its active stream with
//! its table uploads. The lever's step runs at EVERY cadence group end, on a frozen tail too
//! (a freeze never resets the cadence under this lever, so the retirement clock stays the one
//! every rank counts); a frozen tail posts markers only and keeps applying. The plane of a
//! layer is read by `qwen4_ep_owner_assign_pinned` on this rank's stream, enqueued after the
//! upload: a pin never changes between a window's first and last call. A pin FAULT (a missing
//! post, a digest that differs, a failed plane upload: ranks whose planes may differ) is the
//! one thing that does fail decodes: [`UnitTailRank::pins_fault`] makes the owner assignment
//! refuse on every rank, because running a call over diverged planes is not exact.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hip_bridge::{Event, HipRuntime, PinnedHostBuffer, Stream, HIP_EVENT_DISABLE_TIMING};
use railgun::sealed_table::generation::{
    acquire_window_owned, finish_window, BlockRegistry, Expert, FenceBackend, FenceFailure,
    FenceState, FillComplete, FillTicket, FillsNotDone, GenerationId, OwnedTableGeneration,
    PeerReaderFence, PendingFills, PendingOrQuarantined, PendingRestore, ScratchLoan, SlotSpan,
    TableBacking, WindowReadLease,
};
use railgun::window::Quiescence;
use railgun::Binding;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::expert_ep_units::EpUnitPlan;
use crate::expert_unit_tail::{
    occupant_unit, EpUnitTailConfig, FillOp, PeerBoard, PeerDelta, PeerTake, PinDelta, PinPlanes,
    PinProtocol, Retire, RetireCadence, TailPolicy, UnitKey, WindowUse, QUARANTINE_WINDOWS,
};

/// Ids one window's log holds (`calls x slots`); a window past this stops
/// logging (counted, never fatal: it only thins the next admission).
const LOG_IDS: usize = 1 << 18;
/// Print a counter line every this many retirements under the log env (fine enough to
/// attribute the cumulative raw counters to one loadgen cell by their `unix_ms` stamps).
const LOG_EVERY: u64 = 4;

/// One owned unit of a managed layer that a tail slot holds at the last retirement: its
/// canonical decode-table entries (the pinned host arena addresses D1 uploaded) and the two slot
/// addresses that the decode table names instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LiveSlot {
    pub expert: usize,
    /// Canonical `[gate_up, down]` device-visible addresses.
    pub canon: [usize; 2],
    /// Slot `[gate_up, down]` addresses (byte copies of the canonical unit).
    pub slot: [usize; 2],
}

/// The units of one layer whose table `image` (u64 LE per expert, as last uploaded) differs from
/// its canonical entries, restricted to the experts `tail_owned` names (the units the tail may
/// hold): exactly the units a slot serves at the last retirement. Ascending by expert.
fn live_slot_units(
    tail_owned: impl Fn(usize) -> bool,
    canon: (&[u64], &[u64]),
    image: (&[u8], &[u8]),
) -> Vec<LiveSlot> {
    let entry = |bytes: &[u8], expert: usize| -> u64 {
        let at = expert * 8;
        u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte table entry"))
    };
    (0..canon.0.len())
        .filter(|&expert| tail_owned(expert))
        .filter_map(|expert| {
            let slot = [entry(image.0, expert), entry(image.1, expert)];
            (slot != [canon.0[expert], canon.1[expert]]).then_some(LiveSlot {
                expert,
                canon: [canon.0[expert] as usize, canon.1[expert] as usize],
                slot: slot.map(|address| address as usize),
            })
        })
        .collect()
}

/// What the tail needs of one decode MoE layer runtime (from the forward).
pub(crate) struct TailLayerInit {
    /// The layer's label (`layer_idx` of its gather calls).
    pub label: usize,
    /// Non-owning aliases of the layer's two decode pointer tables
    /// (`num_experts` u64 each); the forward owns and outlives them.
    pub gate_up_table: GpuTensor,
    pub down_table: GpuTensor,
    /// The entries the tables were uploaded from (canonical pointers).
    pub gate_up_entries: Vec<usize>,
    pub down_entries: Vec<usize>,
    /// The layer's full (chunk / prefill) table, when `HIPFIRE_EP_UNIT_TAIL_PREFILL` is on.
    pub full: Option<TailFullInit>,
    /// A pinned unit layer's pin plane (`HIPFIRE_EP_UNIT_TAIL_PINS`): a non-owning alias of the
    /// device `u8[512]` the balanced layer owns; the tail rewrites it at retirements. `None`
    /// without the lever and for a layer without a pin plane.
    pub pins: Option<GpuTensor>,
}

/// Whether the forward hands layer `ordinal`'s tables to the tail (`HIPFIRE_EP_UNIT_TAIL`): a
/// static-owner layer over host experts as ever, and with the pins lever also every layer that
/// has a pin plane (pinned unit layers keep the balanced owner kind, their full tables are what
/// the per-call owner assignment reads). Without the lever this is exactly the static-owner rule.
pub(crate) fn tail_layer_selected(
    static_owners: bool,
    host_mapped: bool,
    with_pins: bool,
    has_pin_plane: bool,
) -> bool {
    (static_owners && host_mapped) || (with_pins && has_pin_plane)
}

/// The layer's FULL (chunk / prefill) pointer tables: non-owning aliases and the host entries
/// they were uploaded from. Entries this rank owns that equal the decode table's canonical
/// pointers follow the decode table's slot publications; every other entry is never written.
pub(crate) struct TailFullInit {
    pub gate_up_table: GpuTensor,
    pub down_table: GpuTensor,
    pub gate_up_entries: Vec<usize>,
    pub down_entries: Vec<usize>,
}

/// Bytes of one pointer-table entry (u64 LE per expert).
const ENTRY_BYTES: usize = 8;

/// The inclusive entry range `(lo, hi)` where two byte images of one table differ, `None` when
/// they are equal. The pure definition of what a retirement has to upload of a table: the
/// device holds `old`, the host image is `new`, the bytes of entries `lo..=hi` (and nothing
/// else) make the device table equal to `new`. [`DirtyRange`] tracks it incrementally; the tests
/// hold the two equal.
#[cfg(test)]
fn changed_entry_range(old: &[u8], new: &[u8]) -> Option<(usize, usize)> {
    debug_assert_eq!(old.len(), new.len());
    let entries = old.len().min(new.len()) / ENTRY_BYTES;
    let differs = |entry: usize| {
        old[entry * ENTRY_BYTES..(entry + 1) * ENTRY_BYTES]
            != new[entry * ENTRY_BYTES..(entry + 1) * ENTRY_BYTES]
    };
    let lo = (0..entries).find(|&entry| differs(entry))?;
    let hi = (lo..entries).rev().find(|&entry| differs(entry))?;
    Some((lo, hi))
}

/// The byte span `(offset, len)` of the inclusive entry range `(lo, hi)`.
fn entry_span((lo, hi): (usize, usize)) -> (usize, usize) {
    (lo * ENTRY_BYTES, (hi - lo + 1) * ENTRY_BYTES)
}

/// The entries of one table image rewritten since its last upload, as an inclusive range. Only a
/// write that changes an entry's bytes is noted, so after a retirement the range is exactly
/// [`changed_entry_range`] of the device's table and the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DirtyRange(Option<(usize, usize)>);

impl DirtyRange {
    fn note(&mut self, expert: usize) {
        self.0 = Some(match self.0 {
            None => (expert, expert),
            Some((lo, hi)) => (lo.min(expert), hi.max(expert)),
        });
    }

    /// The range to upload; the image is clean afterwards.
    fn take(&mut self) -> Option<(usize, usize)> {
        self.0.take()
    }
}

/// One hipHostMalloc (portable) allocation that every table image of a rank is carved from.
/// Nothing reads or writes it except through the [`TableImage`] ranges, which are disjoint.
struct PinnedPool {
    _buf: PinnedHostBuffer,
}

// SAFETY: `PinnedPool` only keeps the allocation alive (the buffer has no accessor here); the
// bytes are reached only through `TableImage` ranges, each the unique accessor of its own
// disjoint range, which the tail drives from one rank thread behind `UnitTailRank::inner`'s
// mutex. The HIP library the buffer frees through is thread-agnostic and kept alive by the
// buffer. A device read of an image (an async table upload) is fenced by the window protocol:
// an image is rewritten only after the next retirement's `stream_synchronize`, and `release`
// synchronizes the compute stream before the pool can drop.
unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}

/// One table's host byte image (u64 LE per expert) in pinned host memory, with the entries
/// rewritten since its last upload. Pinned, so an `hipMemcpyAsync` from it is a plain
/// stream-ordered DMA of just the changed entry range instead of a staged copy of the table.
struct TableImage {
    ptr: *mut u8,
    len: usize,
    dirty: DirtyRange,
    _pool: Arc<PinnedPool>,
}

// SAFETY: see `PinnedPool`: this is the unique accessor of `ptr[..len]`, a range of an
// allocation that `_pool` keeps alive, and it moves with its owner (`Inner`, behind a mutex).
unsafe impl Send for TableImage {}

impl std::ops::Deref for TableImage {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr[..len]` is an initialized range of the live pool that only this image
        // accesses (`TableImage`'s Send note).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for TableImage {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `deref`; `&mut self` excludes any other host access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl TableImage {
    /// Point `expert` at `value`; noted dirty only when the entry changed. The table backing is
    /// the write capability: the images are the backing's host side, written only while no
    /// generation is sealed over the tables.
    fn write_entry(&mut self, _backing: &TableBacking, expert: usize, value: u64) {
        let at = expert * ENTRY_BYTES;
        let bytes = value.to_le_bytes();
        if self[at..at + ENTRY_BYTES] != bytes {
            self[at..at + ENTRY_BYTES].copy_from_slice(&bytes);
            self.dirty.note(expert);
        }
    }

    /// Enqueue the upload of the entries rewritten since the last one (nothing when clean) to
    /// `table` on `stream`; returns the bytes enqueued. The image may not be rewritten until the
    /// stream drained (the window protocol's fence of the generation published through it, at
    /// the next retirement). The table backing is the write capability.
    fn upload_dirty(
        &mut self,
        _backing: &TableBacking,
        gpu: &Gpu,
        table: &GpuTensor,
        stream: Option<&Stream>,
    ) -> Result<usize, String> {
        let Some(range) = self.dirty.take() else {
            return Ok(0);
        };
        let (offset, len) = entry_span(range);
        gpu.hip
            .memcpy_htod_async_on(
                &table.buf.byte_view(offset, len),
                &self[offset..offset + len],
                stream,
            )
            .map_err(io)?;
        Ok(len)
    }
}

/// The rank's mirror of one layer's full tables: byte images as last uploaded and which experts
/// follow the decode table.
struct FullTable {
    table_gate_up: GpuTensor,
    table_down: GpuTensor,
    image_gate_up: TableImage,
    image_down: TableImage,
    /// `shared[expert]`: the full table's canonical entries of `expert` are the decode table's
    /// (so a slot copy of the unit is a valid byte copy for both tables).
    shared: Vec<bool>,
    /// The full-table entries as uploaded at install (the canonical pointers): what a peer-owned
    /// entry returns to when its owner unpublishes the slot, and what [`peer_shared`] compares
    /// the owner's canonical entries with.
    base_gate_up: Vec<u64>,
    base_down: Vec<u64>,
}

/// Which experts of a layer the full table shares with the decode table: the rank's OWNED experts
/// (`e % ranks == rank`) whose full-table entries name exactly the decode table's canonical
/// pointers (same unit bytes, so one slot copy serves both tables). Peer-owned entries name the
/// peer's memory and are never touched; an owned entry that differs (a different residency view)
/// stays canonical in the full table.
fn full_shared(
    rank: usize,
    ranks: usize,
    decode: (&[u64], &[u64]),
    full: (&[u64], &[u64]),
) -> Vec<bool> {
    (0..decode.0.len())
        .map(|expert| {
            expert % ranks == rank
                && decode.0[expert] != 0
                && decode.1[expert] != 0
                && decode.0[expert] == full.0[expert]
                && decode.1[expert] == full.1[expert]
        })
        .collect()
}

/// May `rank`'s full table follow `owner`'s slot for `expert`? Only a PEER-owned entry
/// (`owner != rank`, `expert % ranks == owner`, the static owner of the unit) whose install-time
/// full-table entries `base` are exactly the owner's canonical entries `canon` (same unit bytes,
/// so a copy in the owner's slot is a valid byte copy for this table too), both non-null. Any
/// other entry (an owner's differing residency view, a null pointer, an own entry, which
/// [`full_shared`] covers) is never touched.
fn peer_shared(
    rank: usize,
    ranks: usize,
    owner: usize,
    expert: usize,
    canon: [u64; 2],
    base: [u64; 2],
) -> bool {
    owner != rank
        && owner < ranks
        && expert % ranks == owner
        && canon[0] != 0
        && canon[1] != 0
        && canon == base
}

/// The u64 LE table entry of `expert` in a byte image.
fn image_entry(bytes: &[u8], expert: usize) -> u64 {
    let at = expert * 8;
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8-byte table entry"))
}

/// Apply one peer delta (owner `owner`) to one layer's full-table image: a publication points the
/// entry at the owner's slot, an unpublication returns it to `base` (the canonical entry). Only
/// an entry [`peer_shared`] admits is ever written. `true` when the image changed.
fn apply_peer_delta(
    rank: usize,
    ranks: usize,
    owner: usize,
    base: (&[u64], &[u64]),
    image: (&mut [u8], &mut [u8]),
    delta: &PeerDelta,
) -> bool {
    let expert = delta.unit.1 as usize;
    if expert >= base.0.len()
        || expert >= base.1.len()
        || image.0.len() < (expert + 1) * 8
        || image.1.len() < (expert + 1) * 8
    {
        return false;
    }
    let base_entry = [base.0[expert], base.1[expert]];
    if !peer_shared(rank, ranks, owner, expert, delta.canon, base_entry) {
        return false;
    }
    let want = delta.slot.unwrap_or(base_entry);
    if [image_entry(image.0, expert), image_entry(image.1, expert)] == want {
        return false;
    }
    let at = expert * 8;
    image.0[at..at + 8].copy_from_slice(&want[0].to_le_bytes());
    image.1[at..at + 8].copy_from_slice(&want[1].to_le_bytes());
    true
}

impl FullTable {
    /// [`apply_peer_delta`] on this table's image. The table backing is the write capability.
    fn apply_peer(
        &mut self,
        _backing: &TableBacking,
        rank: usize,
        ranks: usize,
        owner: usize,
        delta: &PeerDelta,
    ) -> bool {
        let changed = apply_peer_delta(
            rank,
            ranks,
            owner,
            (&self.base_gate_up, &self.base_down),
            (&mut self.image_gate_up[..], &mut self.image_down[..]),
            delta,
        );
        if changed {
            let expert = delta.unit.1 as usize;
            self.image_gate_up.dirty.note(expert);
            self.image_down.dirty.note(expert);
        }
        changed
    }
}

/// How one `(layer, expert)` read is served, for this rank.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Not held by this rank (or outside the planned layers).
    Foreign,
    /// Static core / a whole-resident layer: VRAM.
    Core,
    /// A non-core unit of a mixed layer: the tail may hold it.
    Tail,
    /// Host-read unit of a layer the tail does not manage (a wholly host layer).
    HostOnly,
}

struct TailLayer {
    table_gate_up: GpuTensor,
    table_down: GpuTensor,
    canon_gate_up: Vec<u64>,
    canon_down: Vec<u64>,
    /// The byte images of the two tables as last uploaded (u64 LE per expert), pinned.
    image_gate_up: TableImage,
    image_down: TableImage,
    /// The layer's full (prefill) tables, mirrored for the experts that share canonical entries.
    full: Option<FullTable>,
}

impl TailLayer {
    /// Point `expert` at `(gate_up, down)` in the decode table and, when its canonical entries
    /// are shared, in the full table. The table backing is the write capability.
    fn set(&mut self, backing: &TableBacking, expert: usize, gate_up: u64, down: u64) {
        self.image_gate_up.write_entry(backing, expert, gate_up);
        self.image_down.write_entry(backing, expert, down);
        if let Some(full) = self.full.as_mut().filter(|full| full.shared[expert]) {
            full.image_gate_up.write_entry(backing, expert, gate_up);
            full.image_down.write_entry(backing, expert, down);
        }
    }
}

/// A failed fence whose stream state cannot be proven: everything the fence guarded stays
/// quarantined.
fn unknown_failure(error: impl std::fmt::Display) -> FenceFailure {
    FenceFailure {
        message: error.to_string(),
        quiescence: Quiescence::Unknown,
    }
}

/// A process-unique backend incarnation: a fence recorded on one backend value is never accepted
/// by another.
fn next_incarnation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The copy stream's completion events: one per batch of fills (and per restore), recycled once
/// the batch completed. Owned by [`Inner`]; every fence method goes through a [`RankCopy`].
struct EventPool {
    /// One copy stream per tail, never recreated: the incarnation is fixed.
    incarnation: u64,
    idle: Vec<Event>,
    /// Recorded and not yet recycled, by fence id (ascending = record order).
    live: BTreeMap<u64, Event>,
    next: u64,
}

impl EventPool {
    fn new() -> Self {
        Self {
            incarnation: next_incarnation(),
            idle: Vec::new(),
            live: BTreeMap::new(),
            next: 1,
        }
    }

    /// Every event the pool holds (teardown destroys them).
    fn drain(&mut self) -> Vec<Event> {
        let mut all = std::mem::take(&mut self.idle);
        all.extend(std::mem::take(&mut self.live).into_values());
        all
    }
}

/// The rank's copy-stream fence capability (DESIGN `RankCopy`): a BORROWED view of the HIP
/// runtime, the copy stream and the event pool, minted on the rank thread for one call and
/// `!Send` / `!Sync` (the handles are not `Sync`; no other thread may fence the stream).
struct RankCopy<'a> {
    hip: &'a HipRuntime,
    stream: &'a Stream,
    pool: &'a mut EventPool,
    _rank_thread: PhantomData<*const ()>,
}

impl<'a> RankCopy<'a> {
    fn new(hip: &'a HipRuntime, stream: &'a Stream, pool: &'a mut EventPool) -> Self {
        Self {
            hip,
            stream,
            pool,
            _rank_thread: PhantomData,
        }
    }

    /// Host-wait the NEWEST recorded fence's event (`hipEventSynchronize`) without completing
    /// anything: fill overlap's wait before the batches are polled (the poll then finds every
    /// batch complete, in stream order).
    fn wait_newest(&mut self) -> Result<(), String> {
        match self.pool.live.values().next_back() {
            Some(event) => self.hip.event_synchronize(event).map_err(io),
            None => Ok(()),
        }
    }

    /// Host-wait every recorded fence's event in record order without completing anything: a
    /// borrow open's drain of the pending batches.
    fn wait_all(&mut self) -> Result<(), String> {
        for event in self.pool.live.values() {
            self.hip.event_synchronize(event).map_err(io)?;
        }
        Ok(())
    }
}

// SAFETY: a fence is one `hipEvent` recorded on the copy stream AFTER every copy the owner
// enqueued there before `record()` returned (the bank records after the last enqueue callback).
// `poll` returns `Complete` only when `hipEventQuery` reports the event recorded and complete,
// and `wait` only when `hipEventSynchronize` returned success: in stream order that means every
// earlier operation of the stream finished. A query / synchronize error is `Failed` with
// `Unknown` quiescence (never `Proven`). The incarnation is the pool's, fixed for the one copy
// stream a tail creates and destroys.
unsafe impl FenceBackend for RankCopy<'_> {
    fn incarnation(&self) -> u64 {
        self.pool.incarnation
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        let event = match self.pool.idle.pop() {
            Some(event) => event,
            None => self
                .hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                .map_err(unknown_failure)?,
        };
        if let Err(error) = self.hip.event_record(&event, Some(self.stream)) {
            self.pool.idle.push(event);
            return Err(unknown_failure(error));
        }
        let id = self.pool.next;
        self.pool.next += 1;
        self.pool.live.insert(id, event);
        Ok(id)
    }

    fn poll(&mut self, fence: u64) -> FenceState {
        match self.pool.live.get(&fence) {
            None => FenceState::Failed(unknown_failure(format!("fence {fence} is not recorded"))),
            Some(event) => match self.hip.event_query(event) {
                Ok(true) => FenceState::Complete,
                Ok(false) => FenceState::Pending,
                Err(error) => FenceState::Failed(unknown_failure(error)),
            },
        }
    }

    fn wait(&mut self, fence: u64) -> FenceState {
        match self.pool.live.get(&fence) {
            None => FenceState::Failed(unknown_failure(format!("fence {fence} is not recorded"))),
            Some(event) => match self.hip.event_synchronize(event) {
                Ok(()) => FenceState::Complete,
                Err(error) => FenceState::Failed(unknown_failure(error)),
            },
        }
    }

    fn recycle(&mut self, fence: u64) {
        if let Some(event) = self.pool.live.remove(&fence) {
            self.pool.idle.push(event);
        }
    }
}

/// The rank's compute-stream fence capability (DESIGN `RankCompute`): the window protocol's
/// last-read fence. Borrowed, minted on the rank thread for one call, `!Send` / `!Sync`. Its
/// `record` only marks (the window's last consumer is already enqueued on the active stream)
/// and its `wait` is the `stream_synchronize(active)` the protocol has always done.
struct RankCompute<'a> {
    hip: &'a HipRuntime,
    stream: Option<&'a Stream>,
    incarnation: u64,
    recorded: u64,
    _rank_thread: PhantomData<*const ()>,
}

impl<'a> RankCompute<'a> {
    fn new(hip: &'a HipRuntime, stream: Option<&'a Stream>) -> Self {
        Self {
            hip,
            stream,
            incarnation: next_incarnation(),
            recorded: 0,
            _rank_thread: PhantomData,
        }
    }
}

// SAFETY: `record` only numbers a mark: the owner enqueued every consumer of the window on this
// stream before calling it. `wait(f)` returns `Complete` only after `hipStreamSynchronize` of
// the rank's active stream (`hipDeviceSynchronize` when there is none: the work then runs on the
// null stream) returned success, which finishes everything enqueued before the call, so
// everything enqueued before `record()` returned `f`. `poll` has no non-blocking probe and
// never reports `Complete`. A synchronize error is `Failed` with `Unknown` quiescence. The
// incarnation is unique per value, so a fence is never accepted by another backend.
unsafe impl FenceBackend for RankCompute<'_> {
    fn incarnation(&self) -> u64 {
        self.incarnation
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        self.recorded += 1;
        Ok(self.recorded)
    }

    fn poll(&mut self, _fence: u64) -> FenceState {
        FenceState::Pending
    }

    fn wait(&mut self, fence: u64) -> FenceState {
        if fence == 0 || fence > self.recorded {
            return FenceState::Failed(unknown_failure(format!(
                "fence {fence} was not recorded by this backend"
            )));
        }
        let synced = match self.stream {
            Some(stream) => self.hip.stream_synchronize(stream),
            None => self.hip.device_synchronize(),
        };
        match synced {
            Ok(()) => FenceState::Complete,
            Err(error) => FenceState::Failed(unknown_failure(error)),
        }
    }
}

/// The rank's device-scope fence capability: the bank's quiescence (`hipDeviceSynchronize`),
/// also the window fence of a borrow open. Borrowed, rank thread, `!Send` / `!Sync`.
struct RankDevice<'a> {
    hip: &'a HipRuntime,
    incarnation: u64,
    recorded: u64,
    _rank_thread: PhantomData<*const ()>,
}

impl<'a> RankDevice<'a> {
    fn new(hip: &'a HipRuntime) -> Self {
        Self {
            hip,
            incarnation: next_incarnation(),
            recorded: 0,
            _rank_thread: PhantomData,
        }
    }
}

// SAFETY: as [`RankCompute`], with `hipDeviceSynchronize`, which finishes everything enqueued on
// every stream of the device before the call returned.
unsafe impl FenceBackend for RankDevice<'_> {
    fn incarnation(&self) -> u64 {
        self.incarnation
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        self.recorded += 1;
        Ok(self.recorded)
    }

    fn poll(&mut self, _fence: u64) -> FenceState {
        FenceState::Pending
    }

    fn wait(&mut self, fence: u64) -> FenceState {
        if fence == 0 || fence > self.recorded {
            return FenceState::Failed(unknown_failure(format!(
                "fence {fence} was not recorded by this backend"
            )));
        }
        match self.hip.device_synchronize() {
            Ok(()) => FenceState::Complete,
            Err(error) => FenceState::Failed(unknown_failure(error)),
        }
    }
}

/// A borrow epoch's hold on the pool: the bank's scratch loan (the borrower's write capability
/// over the whole block) and, once issued, the restore the loan ends with.
struct LentPool {
    loan: ScratchLoan,
    restore: Option<PendingRestore>,
    /// Occupied slots the issued restore re-copies.
    slots: u64,
}

/// The pins lever on one rank: its protocol over the pinned host images of the planes (the
/// protocol's [`PinPlanes`] order, ascending by label) and the non-owning device aliases the
/// retirement uploads them to (the balanced layers own and free the planes).
struct RankPins {
    proto: PinProtocol<TableImage>,
    /// `device[i]` is the device plane of `proto.planes()`'s plane `i`.
    device: Vec<GpuTensor>,
    /// Plane uploads (H2D calls) and bytes.
    upload_calls: u64,
    upload_bytes: u64,
    /// The fault was logged.
    reported: bool,
}

struct Inner {
    rank: usize,
    ranks: usize,
    experts: usize,
    trunk_layers: usize,
    /// `layer label -> index into layers` (managed mixed layers only).
    layer_of: Vec<Option<usize>>,
    layers: Vec<TailLayer>,
    /// `[layer * experts + expert]`.
    kinds: Vec<Kind>,
    gate_up_bytes: usize,
    down_bytes: usize,
    /// The slot pool: ONE device allocation laid out `[gate/up block | pad | down block | pad]`
    /// ([`pool_layout`]). The only pool tensor `release` frees, and the span a borrow epoch lends
    /// ([`UnitTailRank::pool_span`]).
    pool: Option<GpuTensor>,
    /// Bytes of the `pool` allocation as laid out (what [`UnitTailRank::pool_span`] reports).
    pool_bytes: usize,
    /// Non-owning views of the pool's two blocks (`pool_gate_up` at offset 0, `pool_down` at the
    /// 256-aligned down offset): what slot addresses and the fill copies use. Never freed.
    pool_gate_up: Option<GpuTensor>,
    pool_down: Option<GpuTensor>,
    copy_stream: Option<Stream>,
    tables_ready: Option<Event>,
    /// The fills' completion fences: one pending batch per retirement that issued fills.
    batches: VecDeque<PendingFills>,
    /// The copy stream's fence events.
    events: EventPool,
    log: Option<GpuTensor>,
    log_host: Vec<u8>,
    log_used: usize,
    calls: Vec<(u16, u32, u32)>,
    /// Which windows retire ([`RetireCadence`]); fed only by `window_routed`, never by anything
    /// this rank logged.
    cadence: RetireCadence,
    /// Routed ids (`slots`, summed) of the trunk gather calls since the last `step_end`, counted
    /// before any rank-local filter, and whether the window made any such call.
    window_routed: u64,
    window_seen: bool,
    /// Scratch of one retirement: slot reads per `layer * experts + expert` and the indices
    /// touched (so the per-window pass never hashes or clears the whole array).
    reads: Vec<u32>,
    touched: Vec<u32>,
    overflowed: u64,
    policy: TailPolicy,
    /// A device error froze the tail; the message was logged once.
    frozen: Option<String>,
    log_enabled: bool,
    host_only_reads: u64,
    retire_ns: u64,
    log_ns: u64,
    upload_ns: u64,
    fill_ns: u64,
    /// Host time the retirements spent waiting for the previous fills' event (fill overlap).
    fill_wait_ns: u64,
    /// Fill overlap (`HIPFIRE_EP_UNIT_TAIL_FILL_OVERLAP`): the fills of the last retirement wait
    /// here (the copy stream already waits for its table upload) until the next forward's first
    /// row-gather call issues them.
    fill_overlap: bool,
    deferred: Vec<FillTicket>,
    uploads: u64,
    /// H2D calls and bytes of the retirements' table uploads (changed entry ranges only).
    upload_calls: u64,
    upload_bytes: u64,
    /// PEER mode: the board shared by every rank of the mesh.
    peer: Option<Arc<PeerBoard>>,
    /// Peer deltas that changed a full-table image / that did not (an unmanaged layer, an entry
    /// [`peer_shared`] does not admit, or an image already in that state).
    peer_applied: u64,
    peer_skipped: u64,
    /// The pins lever's state, `None` without it.
    pins: Option<RankPins>,
    released: bool,
    /// A borrow epoch is open ([`UnitTailRank::borrow_suspend`]): the pool belongs to another user,
    /// every table this tail owns holds canonical entries, and the policy is frozen in place (no
    /// logging, retirement, fill, publication, peer post/take or pin delta). Separate from
    /// `frozen`: a frozen tail can be suspended and resumed.
    suspended: bool,
    /// The tables' write capability: held from a retirement of the published generation until
    /// the next generation is sealed over the tables (and across a borrow epoch). No table image
    /// is written and no table uploaded without it.
    backing: Option<TableBacking>,
    /// The published generation under its window lease, from its sealing (retirement `j`, or the
    /// warm start / a borrow resume) until the next retirement's window fence.
    lease: Option<WindowReadLease<'static>>,
    /// A generation whose window fenced but whose retirement did not finish (an error between):
    /// kept, never dropped into a lease.
    fenced: Option<OwnedTableGeneration>,
    /// The id of the newest generation sealed (a borrow resume seals `sub + 1`).
    last_gen: GenerationId,
    /// The borrow epoch's hold on the pool, from `borrow_suspend`'s loan until `borrow_restore`.
    lent: Option<LentPool>,
    /// Developer skew census (`HIPFIRE_EP_SKEW_LOG=<dir>`, off by default): every retirement writes
    /// the routed ids of the group's calls with how this rank serves each one. `None` when unset.
    skew: Option<SkewLog>,
}

/// `HIPFIRE_EP_SKEW_LOG=<dir>` (developer diagnostic, off by default): at every D2 retirement each
/// rank appends one text line per logged trunk MoE call of the group to `<dir>/skew-rank<r>.log`:
/// `rank retirement call host_monotonic_ns layer slots` then one token per slot, `<expert><class>`
/// with class `F` (another rank's expert), `C` (static VRAM core), `L` (tail unit served from a
/// live VRAM slot), `T` (tail unit read from pinned host) or `H` (host-only unit). It reads only
/// what the retirement already downloaded (no added sync, no device work) and never changes an
/// output; the host timestamp is taken when the call's copy is enqueued.
pub(crate) const EP_SKEW_LOG_ENV: &str = "HIPFIRE_EP_SKEW_LOG";

struct SkewLog {
    out: std::io::BufWriter<std::fs::File>,
    /// Host CLOCK_MONOTONIC ns of each entry of `Inner::calls`, in order.
    ns: Vec<u64>,
    call: u64,
}

impl SkewLog {
    fn open(rank: usize) -> Option<Self> {
        let dir = hipfire_config::developer_var(EP_SKEW_LOG_ENV).ok()?;
        let dir = std::path::PathBuf::from(dir.trim());
        let opened = std::fs::create_dir_all(&dir).and_then(|()| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(format!("skew-rank{rank}.log")))
        });
        match opened {
            Ok(file) => Some(Self {
                out: std::io::BufWriter::new(file),
                ns: Vec::new(),
                call: 0,
            }),
            Err(error) => {
                eprintln!("qwen4 EP rank {rank}: {EP_SKEW_LOG_ENV}: {error}; skew log off");
                None
            }
        }
    }
}

/// CLOCK_MONOTONIC in ns (the clock rocprofv3 timestamps use).
fn monotonic_ns() -> u64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    // SAFETY: libc `clock_gettime(CLOCK_MONOTONIC = 1, &mut ts)` writes only `ts`.
    if unsafe { clock_gettime(1, &mut ts) } != 0 {
        return 0;
    }
    ts.sec as u64 * 1_000_000_000 + ts.nsec as u64
}

/// The rank's tail (installed in the row-gather hook).
pub(crate) struct UnitTailRank {
    inner: Mutex<Inner>,
}

fn io(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// The pin deltas of one retirement: its publications (`on`, a completed fill the tables now
/// name) in publish order, then its evictions (`off`, back to canonical), the order
/// [`UnitTailRank::install`]'s warm start and the peer deltas use too.
fn pin_deltas(retire: &Retire) -> Vec<PinDelta> {
    retire
        .publish
        .iter()
        .map(|&(unit, _)| PinDelta { unit, on: true })
        .chain(
            retire
                .unpublish
                .iter()
                .map(|&unit| PinDelta { unit, on: false }),
        )
        .collect()
}

impl UnitTailRank {
    /// Build the rank's tail: the pool, the copy stream and log, seed the pool with
    /// the next-hottest units the static core does not hold and publish them.
    /// `layers` are every decode runtime of the rank's forward; only the plan's
    /// MIXED layers are managed. Fails (never falls back) when the plan has no
    /// ranking, no mixed layer or no unit to hold.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install(
        gpu: &mut Gpu,
        rank: usize,
        trunk_layers: usize,
        plan: &EpUnitPlan,
        config: EpUnitTailConfig,
        freeze: bool,
        log_enabled: bool,
        fill_overlap: bool,
        layers: Vec<TailLayerInit>,
        peer: Option<Arc<PeerBoard>>,
        pin_board: Option<Arc<PeerBoard>>,
    ) -> Result<Self, String> {
        use crate::expert_ep_units::EpLayerState;
        let geometry = *plan.geometry();
        let experts = geometry.experts_per_layer;
        let ranks = geometry.ranks;
        let retire_ids = crate::expert_unit_tail::ep_unit_tail_retire_ids()?;
        let ranking = plan
            .ranking_order()
            .ok_or("the unit plan carries no ranking (a whole-layers plan has no tail)")?;
        if geometry.layers != trunk_layers {
            return Err(format!(
                "the unit plan covers {} layers, the model has {trunk_layers}",
                geometry.layers
            ));
        }
        let (gub, dnb) = (
            geometry.gate_up_bytes as usize,
            geometry.down_bytes as usize,
        );

        // Managed layers: this rank's Mixed layers with a decode runtime.
        let mut layer_of: Vec<Option<usize>> = vec![None; geometry.layers];
        let mut managed: Vec<TailLayer> = Vec::new();
        let (mut full_layers, mut shared_entries) = (0usize, 0usize);
        // Every table image of the rank lives in ONE pinned host allocation (2 decode images per
        // managed layer, 2 more with its full table), so a retirement's upload of the changed
        // entries is a plain stream-ordered DMA.
        let images: usize = layers
            .iter()
            .filter(|init| {
                init.label < geometry.layers
                    && matches!(
                        plan.layer_state(rank, init.label),
                        EpLayerState::Mixed { .. }
                    )
            })
            .map(|init| 2 + 2 * usize::from(init.full.is_some()))
            .sum();
        // The pins lever's planes (`experts` bytes each) are carved from the same allocation,
        // after the table images, so their u64 entries stay aligned.
        let pin_count = if pin_board.is_some() {
            layers
                .iter()
                .filter(|init| init.pins.is_some() && init.label < geometry.layers)
                .count()
        } else {
            0
        };
        let pool = if images == 0 && pin_count == 0 {
            None
        } else {
            let mut buf = gpu
                .hip
                .host_buffer(images * experts * ENTRY_BYTES + pin_count * experts)
                .map_err(|e| {
                    format!(
                        "the tail's pinned table images ({images} tables, {pin_count} pin planes): {e}"
                    )
                })?;
            let ptr = buf.as_bytes_mut().as_mut_ptr();
            Some((Arc::new(PinnedPool { _buf: buf }), ptr))
        };
        let mut carved = 0usize;
        let mut image = |entries: &[u64]| -> TableImage {
            let (pool, base) = pool.as_ref().expect("a managed layer has a pinned pool");
            let len = entries.len() * ENTRY_BYTES;
            let mut image = TableImage {
                // SAFETY: the carve offsets are disjoint and `images * experts * 8` bytes were
                // allocated for exactly the images carved here.
                ptr: unsafe { base.add(carved) },
                len,
                dirty: DirtyRange::default(),
                _pool: pool.clone(),
            };
            carved += len;
            for (at, entry) in image.chunks_exact_mut(ENTRY_BYTES).zip(entries) {
                at.copy_from_slice(&entry.to_le_bytes());
            }
            image
        };
        let mut pin_inits: Vec<(usize, GpuTensor)> = Vec::new();
        for mut init in layers {
            let label = init.label;
            if pin_board.is_some() && label < geometry.layers {
                if let Some(alias) = init.pins.take() {
                    pin_inits.push((label, alias));
                }
            }
            if label >= geometry.layers
                || !matches!(plan.layer_state(rank, label), EpLayerState::Mixed { .. })
            {
                continue;
            }
            if init.gate_up_entries.len() != experts || init.down_entries.len() != experts {
                return Err(format!(
                    "layer {label}: the decode table holds {} / {} entries, the model has {experts} experts",
                    init.gate_up_entries.len(),
                    init.down_entries.len()
                ));
            }
            let canon_gate_up: Vec<u64> = init.gate_up_entries.iter().map(|&p| p as u64).collect();
            let canon_down: Vec<u64> = init.down_entries.iter().map(|&p| p as u64).collect();
            let (image_gate_up, image_down) = (image(&canon_gate_up), image(&canon_down));
            if init.gate_up_table.buf.size() < experts * 8
                || init.down_table.buf.size() < experts * 8
            {
                return Err(format!(
                    "layer {label}: a decode pointer table is too small"
                ));
            }
            let full = match init.full {
                None => None,
                Some(full) => {
                    if full.gate_up_entries.len() != experts
                        || full.down_entries.len() != experts
                        || full.gate_up_table.buf.size() < experts * 8
                        || full.down_table.buf.size() < experts * 8
                    {
                        return Err(format!(
                            "layer {label}: the full prefill table does not hold {experts} entries"
                        ));
                    }
                    let full_gate_up: Vec<u64> =
                        full.gate_up_entries.iter().map(|&p| p as u64).collect();
                    let full_down: Vec<u64> = full.down_entries.iter().map(|&p| p as u64).collect();
                    let shared = full_shared(
                        rank,
                        ranks,
                        (&canon_gate_up, &canon_down),
                        (&full_gate_up, &full_down),
                    );
                    shared_entries += shared.iter().filter(|&&s| s).count();
                    full_layers += 1;
                    Some(FullTable {
                        table_gate_up: full.gate_up_table,
                        table_down: full.down_table,
                        image_gate_up: image(&full_gate_up),
                        image_down: image(&full_down),
                        shared,
                        base_gate_up: full_gate_up,
                        base_down: full_down,
                    })
                }
            };
            layer_of[label] = Some(managed.len());
            managed.push(TailLayer {
                table_gate_up: init.gate_up_table,
                table_down: init.down_table,
                canon_gate_up,
                canon_down,
                image_gate_up,
                image_down,
                full,
            });
        }
        if managed.is_empty() {
            return Err("the unit plan has no mixed layer on this rank".into());
        }
        // The pins lever: every layer with a pin plane (all pinned unit layers, whatever this
        // rank holds of them) gets its plane's host image, carved after the table images.
        let pin_state = match pin_board.as_ref() {
            None => None,
            Some(board) => {
                if pin_inits.is_empty() {
                    return Err("the pins lever found no pinned unit layer on this rank".into());
                }
                pin_inits.sort_by_key(|(label, _)| *label);
                let (pool, base) = pool.as_ref().expect("pin planes have a pinned pool");
                let mut planes = Vec::with_capacity(pin_inits.len());
                for (label, alias) in &pin_inits {
                    let statics = crate::expert_ep_units::ep_unit_pin_plane(plan, *label);
                    rdna_compute::moe::qwen4_ep_owner_pins_check(&statics, ranks)
                        .map_err(|why| format!("layer {label} pin plane: {why}"))?;
                    if statics.len() != experts || alias.buf.size() < experts {
                        return Err(format!(
                            "layer {label}: the pin plane holds {} bytes on the device, the plan has {} experts",
                            alias.buf.size(),
                            statics.len()
                        ));
                    }
                    let mut image = TableImage {
                        // SAFETY: `pin_count * experts` bytes were allocated after the table
                        // images for exactly these planes; the offsets are disjoint.
                        ptr: unsafe { base.add(carved) },
                        len: experts,
                        dirty: DirtyRange::default(),
                        _pool: pool.clone(),
                    };
                    carved += experts;
                    image.copy_from_slice(&statics);
                    planes.push((*label, statics, image));
                }
                let planes = PinPlanes::new(ranks, planes)?;
                Some(RankPins {
                    proto: PinProtocol::new(rank, board.clone(), planes),
                    device: pin_inits.into_iter().map(|(_, alias)| alias).collect(),
                    upload_calls: 0,
                    upload_bytes: 0,
                    reported: false,
                })
            }
        };

        // Per-read kinds and the ranking priority of the units the tail may hold.
        let mut kinds = vec![Kind::Foreign; geometry.layers * experts];
        let mut priority: HashMap<UnitKey, u32> = HashMap::new();
        let mut order: Vec<UnitKey> = Vec::new();
        for layer in 0..geometry.layers {
            let managed_layer = layer_of[layer].is_some();
            for expert in (rank..experts).step_by(ranks) {
                kinds[layer * experts + expert] = if plan.is_resident(rank, layer, expert) {
                    Kind::Core
                } else if managed_layer {
                    Kind::Tail
                } else {
                    Kind::HostOnly
                };
            }
        }
        for (pos, &id) in ranking.iter().enumerate() {
            let (layer, expert) = (id as usize / experts, id as usize % experts);
            if layer < geometry.layers && kinds[layer * experts + expert] == Kind::Tail {
                priority.insert((layer as u16, expert as u16), pos as u32);
                order.push((layer as u16, expert as u16));
            }
        }
        if order.is_empty() {
            return Err("no unit of this rank is left for the tail".into());
        }

        // The slot pool: ONE device allocation (one memset): the gate/up block at offset 0, the
        // down block at its 256-aligned offset; `pool_gate_up` / `pool_down` are non-owning views
        // of the two blocks. The allocation is the slot bank's block: every slot's two write
        // extents are block-relative spans of it.
        let slots = config.tail_units;
        let layout = pool_layout(slots, gub, dnb, crate::expert_arena::ARENA_TAIL_PAD_BYTES);
        let pool = gpu.zeros(&[layout.total], DType::Raw).map_err(|e| {
            format!(
                "tail install on rank {rank}: the tail pool ({slots} slots, {} bytes): {e}",
                layout.total
            )
        })?;
        let (pool_view_gate_up, pool_view_down) = (
            pool_view(&pool, 0, layout.gate_up_len),
            pool_view(&pool, layout.down_offset, layout.down_len),
        );
        let block = match BlockRegistry::new(rank as u32).adopt::<Expert>(
            pool.buf.as_ptr() as u64,
            layout.total as u64,
            1,
        ) {
            Ok(block) => block,
            Err(error) => {
                let _ = gpu.free_tensor(pool);
                return Err(format!(
                    "tail install on rank {rank}: the pool block: {error}"
                ));
            }
        };
        let policy = TailPolicy::new(
            config,
            freeze,
            priority,
            peer.is_some(),
            block,
            pool_spans(slots, gub, dnb, &layout),
        );
        let mut policy = match policy {
            Ok(policy) => policy,
            Err((_block, error)) => {
                let _ = gpu.free_tensor(pool);
                return Err(format!(
                    "tail install on rank {rank}: the slot bank: {error}"
                ));
            }
        };
        let seeds = match policy.seed(&order) {
            Ok(seeds) => seeds,
            Err(error) => {
                let _ = gpu.free_tensor(pool);
                return Err(format!(
                    "tail install on rank {rank}: the warm start: {error}"
                ));
            }
        };
        let seed_keys: Vec<(usize, UnitKey)> = seeds.iter().map(|op| (op.slot, op.unit)).collect();

        // Device state. Any failure frees what was made.
        let mut state = Inner {
            rank,
            ranks,
            experts,
            trunk_layers,
            layer_of,
            layers: managed,
            kinds,
            gate_up_bytes: gub,
            down_bytes: dnb,
            pool: Some(pool),
            pool_bytes: layout.total,
            pool_gate_up: Some(pool_view_gate_up),
            pool_down: Some(pool_view_down),
            copy_stream: None,
            tables_ready: None,
            batches: VecDeque::new(),
            events: EventPool::new(),
            log: None,
            log_host: vec![0u8; LOG_IDS * 4],
            log_used: 0,
            calls: Vec::new(),
            cadence: RetireCadence::new(retire_ids, LOG_IDS as u64),
            window_routed: 0,
            window_seen: false,
            reads: vec![0u32; geometry.layers * experts],
            touched: Vec::new(),
            overflowed: 0,
            policy,
            frozen: None,
            log_enabled,
            host_only_reads: 0,
            retire_ns: 0,
            log_ns: 0,
            upload_ns: 0,
            fill_ns: 0,
            fill_wait_ns: 0,
            fill_overlap,
            deferred: Vec::new(),
            uploads: 0,
            upload_calls: 0,
            upload_bytes: 0,
            peer,
            peer_applied: 0,
            peer_skipped: 0,
            pins: pin_state,
            released: false,
            suspended: false,
            backing: None,
            lease: None,
            fenced: None,
            last_gen: GenerationId { epoch: 0, sub: 0 },
            lent: None,
            skew: SkewLog::open(rank),
        };
        let built = state.build_device(gpu, seeds);
        if let Err(error) = built {
            state.release(gpu);
            return Err(format!("tail install on rank {rank}: {error}"));
        }
        if let Some(board) = state.peer.clone() {
            // The warm units are live from the start: post them as retirement 0 (the peers apply
            // them at their first retirement).
            match state.seed_deltas(&seed_keys) {
                Ok(deltas) => board.post(rank, 0, deltas),
                Err(error) => {
                    state.release(gpu);
                    return Err(format!("tail install on rank {rank}: {error}"));
                }
            }
        }
        if let Some(pins) = state.pins.as_mut() {
            // The same warm units, as the pin deltas of retirement 0: every rank applies them at
            // its retirement 2 (the lag), so the planes agree whenever each rank's install ran.
            let deltas: Vec<PinDelta> = seed_keys
                .iter()
                .map(|&(_, unit)| PinDelta { unit, on: true })
                .collect();
            if let Err(error) = pins.proto.post_warm(deltas) {
                state.release(gpu);
                return Err(format!("tail install on rank {rank}: {error}"));
            }
        }
        let owners_note = if state.pins.is_some() {
            format!(
                "on PINNED owners with dynamic pins ({} pinned unit layers; a unit published in its \
                 holder's slot is pinned to its holder two retirements later)",
                state.pins.as_ref().map_or(0, |pins| pins.device.len())
            )
        } else {
            "on static owners (pinned owners are off while the tail is on)".to_string()
        };
        eprintln!(
            "qwen4 EP rank {rank}: D2 tail on: {slots} evictable slots ({} MiB pool) over {} mixed layers \
             {owners_note}, \
             {} warm units published, F={} fills/window{}{}{}; retires every {} routed ids or {} windows",
            ((slots * (gub + dnb)) >> 20),
            state.layers.len(),
            seed_keys.len(),
            config.fill_per_step,
            if config.boundary_every > 1 {
                format!(", boundary every {} windows", config.boundary_every)
            } else {
                String::new()
            },
            if freeze { ", FROZEN (no admission)" } else { "" },
            if fill_overlap {
                ", fills overlap the next forward"
            } else {
                ""
            },
            retire_ids,
            crate::expert_unit_tail::RETIRE_MAX_WINDOWS
        );
        if full_layers > 0 {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail prefill slots on: {full_layers} full (chunk) tables follow the slot \
                 publications of {shared_entries} owned entries (full-table prefill reads a live slot from VRAM, not \
                 the pinned host arena)"
            );
        }
        if state.peer.is_some() {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail PEER slots on: the full (chunk) tables also follow the slot \
                 publications of the {} peer rank(s) for the peer-owned entries equal to the owner's canonical \
                 ones (a full-table prefill reads a peer's live slot over P2P, not the peer's pinned host arena); \
                 an evicted slot is quarantined until every peer's table dropped it (at most {} slots, {} \
                 after a full group of windows)",
                ranks - 1,
                QUARANTINE_WINDOWS * config.fill_per_step,
                QUARANTINE_WINDOWS * config.fill_per_step * crate::expert_unit_tail::FILL_GROUP_CAP
            );
        }
        if let Some(pins) = state.pins.as_ref() {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail PINS on: {} pin planes follow the slot publications of every \
                 rank (a unit published at retirement k is pinned to its holder from retirement k + {}, \
                 an eviction frees it likewise); the planes are identical on every rank by construction, \
                 a missing post or a digest mismatch fails the decode",
                pins.device.len(),
                crate::expert_unit_tail::PIN_LAG
            );
        }
        Ok(Self {
            inner: Mutex::new(state),
        })
    }

    /// Row-gather hook: log this call's routed ids (stream-ordered D2D).
    pub(crate) fn note_call(&self, gpu: &mut Gpu, layer_idx: u16, slots: usize, topk: &GpuTensor) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.note_call(gpu, layer_idx, slots, topk);
    }

    /// End of a decode window (after its forward's D2H sync): retire it.
    pub(crate) fn step_end(&self, gpu: &mut Gpu) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.skip_suspended_window() {
            return;
        }
        if inner.pins.is_some() && !inner.released {
            // The pins lever: the retirement clock runs on a frozen tail too.
            inner.step_end_pinned(gpu);
            return;
        }
        if inner.frozen.is_some() || inner.released {
            inner.calls.clear();
            inner.log_used = 0;
            inner.reset_group();
            return;
        }
        if let Err(error) = inner.step_end(gpu) {
            inner.freeze(&error);
        }
    }

    /// The pins lever's fault, if any rank recorded one: the pinned owner assignment must not
    /// launch (a call over planes that may differ between ranks is not exact), the decode errors
    /// out through the mesh's ordinary error path. `None` without the lever.
    pub(crate) fn pins_fault(&self) -> Option<String> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.pins.as_ref().and_then(|pins| pins.proto.fault())
    }

    /// Free the pool, the log, the stream and the events (forward teardown).
    pub(crate) fn release(&self, gpu: &mut Gpu) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.log_enabled {
            inner.print_line("final");
        }
        inner.release(gpu);
    }
}

impl UnitTailRank {
    /// The units of trunk layer `label` that a tail slot serves as of the LAST RETIREMENT (the
    /// decode table's state: every unit published after its fill completed and not yet evicted),
    /// ascending by expert. Empty for an unmanaged layer or a released tail. The unit stage of a
    /// long prefill forward ([`crate::gpu_forward::slab_stage`]) reads this at the forward's
    /// start; retirements run on the rank thread only after a decode window's forward, never
    /// inside a prefill forward, so the answer holds for the whole forward (the slot lifetime
    /// argument is in `slab_stage`'s `UnitSlotStage` docs).
    pub(crate) fn live_slots(&self, label: usize) -> Vec<LiveSlot> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.live_slots(label)
    }
}

/// The borrow API: lend the slot pool to another user for one epoch (see
/// `local://prefborrow-design.md`). Every call runs on the rank thread of the tail, serialised
/// with the window protocol by `inner`'s mutex.
impl UnitTailRank {
    /// Device span of the slot pool: ONE allocation laid out `[gate/up block | pad | down block |
    /// pad]`, base 256-aligned. `None` once released.
    pub(crate) fn pool_span(&self) -> Option<(usize, usize)> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.pool_span()
    }

    /// Open: drain (issue any deferred fills, host-wait the copy stream and every pending fill
    /// batch), suspend the policy, upload the canonical image of every tail-owned device table
    /// synchronously; host images untouched. Err if released, already suspended, or any HIP call
    /// fails; after an Err past those two checks the tail stays suspended, so the close sequence
    /// (`borrow_restore`, `borrow_resume`) is the way back (the pool was not lent yet).
    pub(crate) fn borrow_suspend(&self, gpu: &mut Gpu) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.borrow_suspend(gpu)
    }

    /// Only while suspended: re-copy every occupied slot (a slot whose fill completed: published
    /// or filled-unpublished) from its unit's canonical pinned host bytes into the pool,
    /// host-wait. Returns bytes copied.
    pub(crate) fn borrow_restore(&self, gpu: &mut Gpu) -> Result<u64, String> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.borrow_restore(gpu)
    }

    /// Only while suspended, after `borrow_restore`: D2H every occupied slot, compare
    /// byte-for-byte with the canonical host bytes. Ok(fnv-1a 64 digest over (slot, layer,
    /// expert, bytes)) or Err naming the first mismatch.
    pub(crate) fn borrow_verify(&self, gpu: &mut Gpu) -> Result<u64, String> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.borrow_verify(gpu)
    }

    /// Close: upload the whole LIVE host image of every tail-owned device table synchronously,
    /// resume the policy.
    pub(crate) fn borrow_resume(&self, gpu: &mut Gpu) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.borrow_resume(gpu)
    }

    pub(crate) fn borrow_suspended(&self) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.suspended && !inner.released
    }

    /// Occupied slots and their bytes (for logs/receipts). Exact once suspended (every issued
    /// fill has completed); before that a fill still in flight counts as occupied too.
    pub(crate) fn borrow_occupied(&self) -> (usize, u64) {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let slots = inner.occupied().len();
        (
            slots,
            slots as u64 * (inner.gate_up_bytes + inner.down_bytes) as u64,
        )
    }
}

impl Inner {
    fn live_slots(&self, label: usize) -> Vec<LiveSlot> {
        if self.released || self.suspended {
            return Vec::new();
        }
        let Some(index) = self.layer_of.get(label).copied().flatten() else {
            return Vec::new();
        };
        let layer = &self.layers[index];
        live_slot_units(
            |expert| self.kind(label, expert) == Kind::Tail,
            (&layer.canon_gate_up, &layer.canon_down),
            (&layer.image_gate_up[..], &layer.image_down[..]),
        )
    }

    fn kind(&self, layer: usize, expert: usize) -> Kind {
        if layer >= self.trunk_layers || expert >= self.experts {
            return Kind::Foreign;
        }
        self.kinds[layer * self.experts + expert]
    }

    fn slot_addresses(&self, slot: usize) -> (u64, u64) {
        let gate_up = self.pool_gate_up.as_ref().expect("pool").buf.as_ptr() as usize
            + slot * self.gate_up_bytes;
        let down =
            self.pool_down.as_ref().expect("pool").buf.as_ptr() as usize + slot * self.down_bytes;
        (gate_up as u64, down as u64)
    }

    /// The device allocations every generation of this tail is published through: the decode
    /// tables and, with the prefill lever, the full tables, `experts` u64 entries each.
    fn table_extents(&self) -> Vec<Binding> {
        let bytes = (self.experts * ENTRY_BYTES) as u64;
        let mut out = Vec::new();
        for layer in &self.layers {
            let mut tables = vec![&layer.table_gate_up, &layer.table_down];
            if let Some(full) = layer.full.as_ref() {
                tables.push(&full.table_gate_up);
                tables.push(&full.table_down);
            }
            out.extend(tables.into_iter().map(|table| Binding {
                base: table.buf.as_ptr() as u64,
                bytes,
            }));
        }
        out
    }

    /// Device state of the install: the usage log, the copy stream and its events, the warm
    /// start (`seeds` copied through the bank's fill API, then published as generation
    /// `{ epoch 0, sub 0 }` over the warm tables) and the first window lease. The pool (the
    /// slot bank's block) already exists.
    fn build_device(&mut self, gpu: &mut Gpu, seeds: Vec<FillOp>) -> Result<(), String> {
        let seed_keys: Vec<(usize, UnitKey)> = seeds.iter().map(|op| (op.slot, op.unit)).collect();
        let tickets: Vec<FillTicket> = seeds.into_iter().map(|op| op.ticket).collect();
        // A device failure before the seed copies are issued must not strand their tickets
        // (an unissued ticket refuses the bank's close, which would leak the pool at release):
        // abandon them, so a failed install frees the pool as before.
        if let Err(error) = self.build_device_streams(gpu) {
            self.policy.abandon_fills(tickets);
            return Err(error);
        }
        self.seed_device(gpu, seed_keys, tickets)
    }

    /// The usage log, the drain of its memset, the copy stream and the tables-ready event.
    fn build_device_streams(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.log = Some(
            gpu.zeros(&[LOG_IDS * 4], DType::Raw)
                .map_err(|e| format!("the usage log: {e}"))?,
        );
        // `Gpu::zeros` is a memset on the active stream, and the seed copies below run on the
        // non-blocking copy stream: without this drain a late memset could zero a seeded slot
        // AFTER its copy, and the slot would be published holding zeros (a rank-local wrong
        // read under the full-table prefill route, wrong decode sums on both ranks).
        if let Some(active) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(active).map_err(io)?;
        } else {
            gpu.hip.device_synchronize().map_err(io)?;
        }
        let stream = gpu.hip.stream_create_non_blocking().map_err(io)?;
        self.copy_stream = Some(stream);
        self.tables_ready = Some(
            gpu.hip
                .event_create_with_flags(HIP_EVENT_DISABLE_TIMING)
                .map_err(io)?,
        );
        Ok(())
    }

    /// Warm start: copy every seed (one copy fence), wait, then publish them all.
    fn seed_device(
        &mut self,
        gpu: &mut Gpu,
        seed_keys: Vec<(usize, UnitKey)>,
        tickets: Vec<FillTicket>,
    ) -> Result<(), String> {
        let pending = {
            let (Some(stream), Some(pool)) = (self.copy_stream.as_ref(), self.pool.as_ref()) else {
                self.policy.abandon_fills(tickets);
                return Err("no copy stream or pool".into());
            };
            let mut copy = RankCopy::new(&gpu.hip, stream, &mut self.events);
            let (layers, layer_of) = (&self.layers, &self.layer_of);
            let bytes = (self.gate_up_bytes, self.down_bytes);
            let hip = &gpu.hip;
            let issued = self.policy.issue_fills(tickets, &mut copy, &mut |ticket| {
                copy_unit_to_spans(
                    hip,
                    stream,
                    pool,
                    layers,
                    layer_of,
                    bytes,
                    occupant_unit(ticket.occupant()),
                    ticket.write_spans(),
                )
            });
            match issued {
                Ok(pending) => pending,
                Err((tickets, failure)) => {
                    self.policy.abandon_fills(tickets);
                    return Err(failure.message);
                }
            }
        };
        let done = {
            let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
            let mut copy = RankCopy::new(&gpu.hip, stream, &mut self.events);
            match self.policy.complete_fills(pending, &mut copy, true) {
                Ok(done) => done,
                Err(FillsNotDone::Pending(_)) => {
                    return Err(
                        "the warm-start fence is still pending after a blocking wait".into(),
                    )
                }
                Err(FillsNotDone::Failed { failure, .. }) => return Err(failure.message),
            }
        };
        // The tables the generations are published through: this backing is the write capability
        // of every table image and upload below, until the warm generation is sealed over it.
        let extents = self.table_extents();
        let backing = self
            .policy
            .register_table_backing(extents)
            .map_err(|error| format!("the tail's table backing: {error}"))?;
        let mut touched: HashSet<usize> = HashSet::new();
        for &(slot, unit) in &seed_keys {
            let (layer, expert) = (unit.0 as usize, unit.1 as usize);
            let (gate_up, down) = self.slot_addresses(slot);
            let index = self.layer_of[layer].expect("seed layers are managed");
            self.layers[index].set(&backing, expert, gate_up, down);
            touched.insert(index);
        }
        // Load time: no kernel reads the tables yet, so a synchronous upload is exact.
        for &index in &touched {
            let layer = &self.layers[index];
            upload_image(&backing, gpu, &layer.table_gate_up, &layer.image_gate_up)?;
            upload_image(&backing, gpu, &layer.table_down, &layer.image_down)?;
            if let Some(full) = layer.full.as_ref() {
                upload_image(&backing, gpu, &full.table_gate_up, &full.image_gate_up)?;
                upload_image(&backing, gpu, &full.table_down, &full.image_down)?;
            }
        }
        // The whole images are on the device: nothing is dirty for the first retirement.
        for &index in &touched {
            let layer = &mut self.layers[index];
            layer.image_gate_up.dirty.take();
            layer.image_down.dirty.take();
            if let Some(full) = layer.full.as_mut() {
                full.image_gate_up.dirty.take();
                full.image_down.dirty.take();
            }
        }
        let generation = self
            .policy
            .finish_seed(done, backing)
            .map_err(|(_backing, error)| format!("the warm generation: {error}"))?;
        self.last_gen = GenerationId { epoch: 0, sub: 0 };
        self.lease = Some(
            acquire_window_owned(generation)
                .map_err(|(_generation, error)| format!("the warm generation's lease: {error}"))?,
        );
        Ok(())
    }

    /// Enqueue `tickets`' copies on the copy stream (which already waits for their retirement's
    /// table upload) and record their batch fence, which the next retirement polls (or waits
    /// for, under fill overlap). The copy callback writes only each ticket's write spans.
    fn issue_batch(&mut self, gpu: &Gpu, tickets: Vec<FillTicket>) -> Result<(), String> {
        let (Some(stream), Some(pool)) = (self.copy_stream.as_ref(), self.pool.as_ref()) else {
            self.policy.abandon_fills(tickets);
            return Err("no copy stream or pool".into());
        };
        let mut copy = RankCopy::new(&gpu.hip, stream, &mut self.events);
        let (layers, layer_of) = (&self.layers, &self.layer_of);
        let bytes = (self.gate_up_bytes, self.down_bytes);
        let hip = &gpu.hip;
        let pending = self.policy.issue_fills(tickets, &mut copy, &mut |ticket| {
            copy_unit_to_spans(
                hip,
                stream,
                pool,
                layers,
                layer_of,
                bytes,
                occupant_unit(ticket.occupant()),
                ticket.write_spans(),
            )
        });
        match pending {
            Ok(pending) => self.batches.push_back(pending),
            Err((tickets, failure)) => {
                // Refused or failed: the tickets come back; they will never be issued.
                self.policy.abandon_fills(tickets);
                return Err(failure.message);
            }
        }
        Ok(())
    }

    /// Fill overlap: issue the last retirement's deferred fills (nothing when none wait). A
    /// frozen or released tail drops them: their slots stay `Filling`, which nothing reads, and
    /// the bank is told ([`TailPolicy::abandon_fills`]) so a scratch loan and the close accept
    /// those slots.
    fn issue_deferred(&mut self, gpu: &Gpu) -> Result<(), String> {
        if self.deferred.is_empty() {
            return Ok(());
        }
        let tickets = std::mem::take(&mut self.deferred);
        if self.frozen.is_some() || self.released {
            self.policy.abandon_fills(tickets);
            return Ok(());
        }
        self.issue_batch(gpu, tickets)
    }

    fn note_call(&mut self, gpu: &mut Gpu, layer_idx: u16, slots: usize, topk: &GpuTensor) {
        if self.suspended {
            // A borrow epoch: no window counting, no deferred issue, no usage-log copy.
            return;
        }
        let layer = layer_idx as usize;
        // The cadence's input: every rank enters this call with the same `slots` (the extent of
        // the cross-rank row exchange), so this count is counted before any rank-local early
        // return below.
        if layer < self.trunk_layers && slots > 0 {
            self.window_routed += slots as u64;
            self.window_seen = true;
        }
        // Fill overlap: the first row-gather call after a retirement starts its fills, so they
        // run under this forward (enqueued after the copy stream's wait for the table upload).
        if let Err(error) = self.issue_deferred(gpu) {
            self.freeze(&format!("deferred fills: {error}"));
        }
        if self.frozen.is_some()
            || self.released
            || layer >= self.trunk_layers
            || slots == 0
            || topk.buf.size() < slots * 4
        {
            return;
        }
        let Some(log) = self.log.as_ref() else {
            return;
        };
        if self.log_used + slots * 4 > LOG_IDS * 4 {
            self.overflowed += 1;
            return;
        }
        let at = self.log_used;
        let dst = log.buf.byte_view(at, slots * 4);
        let result = match gpu.active_stream.as_ref() {
            Some(stream) => gpu
                .hip
                .memcpy_dtod_async_at(&dst, 0, &topk.buf, 0, slots * 4, stream),
            None => gpu
                .hip
                .memcpy_dtod_async_default_at(&dst, 0, &topk.buf, 0, slots * 4),
        };
        match result {
            Ok(()) => {
                self.calls.push((layer_idx, slots as u32, at as u32));
                self.log_used += slots * 4;
                if let Some(skew) = self.skew.as_mut() {
                    skew.ns.push(monotonic_ns());
                }
            }
            Err(error) => self.freeze(&format!("usage log copy: {error}")),
        }
    }

    /// Forget the accumulated group (the tail froze or was released).
    fn reset_group(&mut self) {
        self.window_routed = 0;
        self.window_seen = false;
        self.cadence.clear();
    }

    /// The retirement cadence of one window end: `Some(n)` when this window completes a group of
    /// `n` windows. A window that does not complete a group does nothing on the device (no stream
    /// sync, no D2H); its ids stay in the log for the group's retirement. The decision reads the
    /// routed-id sequence only, which is identical on every rank.
    fn group_end(&mut self) -> Option<u32> {
        if !self.window_seen {
            return None;
        }
        let routed = std::mem::take(&mut self.window_routed);
        self.window_seen = false;
        self.cadence.window_end(routed)
    }

    fn step_end(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let Some(windows) = self.group_end() else {
            return Ok(());
        };
        self.retire_group(gpu, windows).map(|_| ())
    }

    /// The pins lever's window end: the retirement clock of the mesh runs here, whatever the
    /// state of this rank's tail. At a group end the stream is made idle, the tail retires
    /// (unless frozen: a failure freezes it and its posts stop), and the pin step ALWAYS runs
    /// with the deltas of a successful retirement (an empty marker otherwise).
    fn step_end_pinned(&mut self, gpu: &mut Gpu) {
        let Some(windows) = self.group_end() else {
            return;
        };
        // The previous retirement's table and plane uploads read the pinned host images, which
        // this retirement rewrites: the stream is idle before anything is touched.
        let synced = match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.stream_synchronize(stream).map_err(io),
            None => Ok(()),
        };
        let mut deltas = Vec::new();
        match synced {
            Err(error) => self.freeze(&error),
            Ok(()) if self.frozen.is_some() => {
                self.calls.clear();
                self.log_used = 0;
            }
            Ok(()) => match self.retire_group(gpu, windows) {
                Ok(posted) => deltas = posted,
                Err(error) => self.freeze(&error),
            },
        }
        self.pin_retire(gpu, deltas);
    }

    /// The pin step of one retirement ([`PinProtocol::retire`]) and the upload of the plane
    /// bytes it changed, on the active stream (enqueued before any kernel of the next window,
    /// like the table uploads). Any failure is the mesh's pin fault: logged once here, and the
    /// pinned owner assignment refuses from then on ([`UnitTailRank::pins_fault`]).
    fn pin_retire(&mut self, gpu: &mut Gpu, deltas: Vec<PinDelta>) {
        let rank = self.rank;
        let Some(pins) = self.pins.as_mut() else {
            return;
        };
        let mut result = pins.proto.retire(deltas);
        if result.is_ok() {
            let compute = gpu.active_stream.as_ref();
            for index in pins.proto.planes_mut().take_dirty() {
                let label = pins.proto.planes().label(index);
                let bytes = pins.proto.planes().bytes(index);
                let sent = gpu.hip.memcpy_htod_async_on(
                    &pins.device[index].buf.byte_view(0, bytes.len()),
                    bytes,
                    compute,
                );
                match sent {
                    Ok(()) => {
                        pins.upload_calls += 1;
                        pins.upload_bytes += pins.proto.planes().bytes(index).len() as u64;
                    }
                    Err(error) => {
                        result = Err(pins.proto.fail(&format!(
                            "the plane of layer {label} could not be uploaded: {error}"
                        )));
                        break;
                    }
                }
            }
        }
        if let Err(why) = result {
            if !pins.reported {
                pins.reported = true;
                eprintln!(
                    "qwen4 EP rank {rank}: D2 tail PINS FAULT: {why}; the pinned owner assignment refuses \
                     from now on on every rank (the decode fails rather than run over planes that may differ)"
                );
            }
        }
    }

    /// [`EP_SKEW_LOG_ENV`]: one line per call of the group just downloaded into `log_host`, the
    /// class of every slot as the tables served it during the group (the images and kinds are
    /// those of the last retirement, unchanged since). A write error turns the log off.
    fn write_skew_log(&mut self, calls: &[(u16, u32, u32)]) {
        use std::io::Write;
        let Some(mut skew) = self.skew.take() else {
            return;
        };
        let ns = std::mem::take(&mut skew.ns);
        let retirement = self.policy.stats().retirements;
        let mut line = String::new();
        let mut result = Ok(());
        for (i, &(label, slots, at)) in calls.iter().enumerate() {
            let layer = label as usize;
            line.clear();
            line.push_str(&format!(
                "{} {} {} {} {} {}",
                self.rank,
                retirement,
                skew.call,
                if ns.len() == calls.len() { ns[i] } else { 0 },
                layer,
                slots
            ));
            skew.call += 1;
            let ids = &self.log_host[at as usize..at as usize + slots as usize * 4];
            for raw in ids.chunks_exact(4) {
                let id = i32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                let class = match usize::try_from(id) {
                    Err(_) => 'X',
                    Ok(expert) => match self.kind(layer, expert) {
                        Kind::Foreign => 'F',
                        Kind::Core => 'C',
                        Kind::HostOnly => 'H',
                        Kind::Tail => {
                            let live = self.layer_of[layer].is_some_and(|index| {
                                let tail = &self.layers[index];
                                let at = expert * 8;
                                let entry = u64::from_le_bytes(
                                    tail.image_gate_up[at..at + 8]
                                        .try_into()
                                        .expect("8-byte table entry"),
                                );
                                entry != tail.canon_gate_up[expert]
                            });
                            if live {
                                'L'
                            } else {
                                'T'
                            }
                        }
                    },
                };
                line.push_str(&format!(" {id}{class}"));
            }
            line.push('\n');
            result = skew.out.write_all(line.as_bytes());
            if result.is_err() {
                break;
            }
        }
        match result.and_then(|()| skew.out.flush()) {
            Ok(()) => self.skew = Some(skew),
            Err(error) => eprintln!(
                "qwen4 EP rank {}: {EP_SKEW_LOG_ENV}: {error}; skew log off",
                self.rank
            ),
        }
    }

    /// One retirement of the tail: the window fence of the published generation, the log's D2H,
    /// the retirement of that generation (the tables' backing is writable again), the policy
    /// step, the table writes and uploads (through the backing), the sealing of the next
    /// generation and its window lease, the peer protocol and the fills. Returns the pin deltas
    /// of the retirement (its publications `on`, its evictions `off`; empty without the pins
    /// lever and when nothing was retired): only a retirement that completed all its table
    /// uploads returns them, so a pin never claims a publication the tables did not take.
    ///
    /// The generation protocol (module docs): retirement `j` retires generation `j - 1` (the one
    /// the windows since retirement `j - 1` read) and seals generation `j`; every error before
    /// the seal leaves the tail frozen with the generation or the backing still held, never with
    /// a table named by an unfenced reader.
    fn retire_group(&mut self, gpu: &mut Gpu, windows: u32) -> Result<Vec<PinDelta>, String> {
        if self.calls.is_empty() {
            return Ok(Vec::new());
        }
        let t0 = Instant::now();
        // The window's forward ended with a D2H; fence the window of the published generation
        // (`finish_window`: the compute backend's wait is the stream synchronize the protocol
        // has always done), so the stream is idle for certain before reading the log and
        // rewriting tables.
        let lease = self.lease.take().ok_or("no published table generation")?;
        self.fenced = Some(fence_window(gpu, lease, false)?);
        let used = self.log_used;
        {
            let log = self.log.as_ref().ok_or("no log")?;
            gpu.dtoh_stream_ordered(&mut self.log_host[..used], &log.buf.byte_view(0, used))
                .map_err(io)?;
        }
        let calls = std::mem::take(&mut self.calls);
        self.log_used = 0;
        self.write_skew_log(&calls);
        let t_log = Instant::now();

        // The window's units: slot reads per distinct (layer, expert) this rank holds.
        for &(label, slots, at) in &calls {
            let ids = &self.log_host[at as usize..at as usize + slots as usize * 4];
            for raw in ids.chunks_exact(4) {
                let id = i32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                let Ok(expert) = usize::try_from(id) else {
                    continue;
                };
                if self.kind(label as usize, expert) == Kind::Foreign {
                    continue;
                }
                let at = label as usize * self.experts + expert;
                if self.reads[at] == 0 {
                    self.touched.push(at as u32);
                }
                self.reads[at] += 1;
            }
        }
        let mut core_reads = 0u64;
        let mut core_distinct = 0u64;
        let mut host_only = 0u64;
        let mut window: Vec<WindowUse> = Vec::new();
        for &at in &self.touched {
            let at = at as usize;
            let reads = std::mem::take(&mut self.reads[at]);
            let (layer, expert) = (at / self.experts, at % self.experts);
            match self.kinds[at] {
                Kind::Foreign => {}
                Kind::Core => {
                    core_reads += reads as u64;
                    core_distinct += 1;
                }
                Kind::Tail => window.push(WindowUse {
                    unit: (layer as u16, expert as u16),
                    reads,
                }),
                Kind::HostOnly => host_only += reads as u64,
            }
        }
        self.touched.clear();
        self.host_only_reads += host_only;
        window.sort_by_key(|w| w.unit);

        // Fill overlap: every fill issued so far (the last retirement's, started under the
        // forward that just ended) completes before the poll, as the lane window's device-wide
        // prepare sync made them complete before. The wait is the part of the copies the forward
        // did not hide.
        if self.fill_overlap {
            self.issue_deferred(gpu)?;
            if !self.batches.is_empty() {
                let t_wait = Instant::now();
                let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
                RankCopy::new(&gpu.hip, stream, &mut self.events).wait_newest()?;
                self.fill_wait_ns += t_wait.elapsed().as_nanos() as u64;
            }
        }
        // Completed fills (issued in order on one stream: stop at the first pending).
        let completed = self.poll_batches(gpu)?;

        // Retire the generation the finished windows read: its window fenced above; in PEER mode
        // this is the retirement's one `released_through` call. The table backing comes back.
        let generation = self.fenced.take().ok_or("no fenced table generation")?;
        let (rank, at) = (self.rank, self.policy.next_retirement());
        let board = self.peer.clone();
        let peers = board
            .as_deref()
            .map(|board| (board as &dyn PeerReaderFence, rank, at));
        let (backing, retired) =
            self.policy
                .retire_generation(generation, peers)
                .map_err(|(generation, error)| {
                    // Fenced, but the bank refused: keep it (never dropped into a lease).
                    self.fenced = Some(generation);
                    format!("retiring the published generation: {error}")
                })?;
        let mut retire = match board {
            Some(_) => self.policy.retire_peer_windows(
                (core_distinct, core_reads),
                &window,
                completed,
                windows,
                &retired,
            ),
            None => self.policy.retire_windows(
                (core_distinct, core_reads),
                &window,
                completed,
                windows,
                &retired,
            ),
        };
        if let (refused @ 1.., why) = self.policy.refusals() {
            let why = format!(
                "the slot bank refused {refused} transition(s): {}",
                why.unwrap_or("?")
            );
            self.backing = Some(backing);
            self.abandon_retire_fills(&mut retire);
            return Err(why);
        }
        // This retirement's index (the retirement counters of all ranks are one clock: the
        // ranks retire the same groups of windows).
        let j = self.policy.stats().retirements;
        let t_retire = Instant::now();

        // Tables first (publish -> slot, evict -> canonical), written through the backing, then
        // the new generation is sealed over them and leased to the windows that follow, then the
        // copies.
        let peer_take = match self.write_retirement(gpu, &backing, &retire, j) {
            Ok(take) => take,
            Err(error) => {
                self.backing = Some(backing);
                self.abandon_retire_fills(&mut retire);
                return Err(error);
            }
        };
        let id = GenerationId { epoch: j, sub: 0 };
        let generation = match self.policy.seal(backing, id) {
            Ok(generation) => generation,
            Err((backing, error)) => {
                self.backing = Some(backing);
                self.abandon_retire_fills(&mut retire);
                return Err(format!("sealing generation {j}: {error}"));
            }
        };
        match acquire_window_owned(generation) {
            Ok(lease) => self.lease = Some(lease),
            Err((_generation, error)) => {
                self.abandon_retire_fills(&mut retire);
                return Err(format!("leasing generation {j}: {error}"));
            }
        }
        self.last_gen = id;
        // The ACK: every peer delta this retirement applied is now enqueued on the compute
        // stream ahead of every kernel of the next window. A failed upload returned above and
        // records nothing, so the owners keep the slots this rank may still name quarantined.
        if let (Some(board), Some(take)) = (self.peer.as_ref(), peer_take.as_ref()) {
            board.ack(self.rank, j, take);
        }
        let t_upload = Instant::now();
        let fills = std::mem::take(&mut retire.fills);
        if !fills.is_empty() {
            let tickets: Vec<FillTicket> = fills.into_iter().map(|op| op.ticket).collect();
            let gated = (|| -> Result<(), String> {
                let compute = gpu.active_stream.as_ref();
                let ready = self.tables_ready.as_ref().ok_or("no tables event")?;
                gpu.hip.event_record(ready, compute).map_err(io)?;
                let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
                gpu.hip.stream_wait_event(stream, ready).map_err(io)
            })();
            if let Err(error) = gated {
                // The copies were never enqueued: their tickets will never be issued.
                self.policy.abandon_fills(tickets);
                return Err(error);
            }
            if self.fill_overlap {
                // Issued by the next forward's first row-gather call (`issue_deferred`).
                self.deferred = tickets;
            } else {
                self.issue_batch(gpu, tickets)?;
            }
        }
        let t_end = Instant::now();
        self.log_ns += (t_log - t0).as_nanos() as u64;
        self.retire_ns += (t_retire - t_log).as_nanos() as u64;
        self.upload_ns += (t_upload - t_retire).as_nanos() as u64;
        self.fill_ns += (t_end - t_upload).as_nanos() as u64;
        if self.log_enabled && self.policy.stats().retirements % LOG_EVERY == 0 {
            self.print_line("window");
        }
        Ok(if self.pins.is_some() {
            pin_deltas(&retire)
        } else {
            Vec::new()
        })
    }

    /// A retirement that fails before its fills are issued or deferred: the tickets will never
    /// be issued, so the bank is told ([`TailPolicy::abandon_fills`]); dropping them would leave
    /// unissued `Filling` slots that refuse every later scratch loan and the close.
    fn abandon_retire_fills(&mut self, retire: &mut Retire) {
        let fills = std::mem::take(&mut retire.fills);
        self.policy
            .abandon_fills(fills.into_iter().map(|op| op.ticket).collect());
    }

    /// Retirement `j`'s table writes: the publications and evictions into the host images, the
    /// peer post / take / apply into the full-table images, and the upload of every changed
    /// entry range on the compute stream. `backing` is the write capability (no generation is
    /// sealed over the tables while it is held). Returns the peer take the ACK covers.
    fn write_retirement(
        &mut self,
        gpu: &Gpu,
        backing: &TableBacking,
        retire: &Retire,
        j: u64,
    ) -> Result<Option<PeerTake>, String> {
        let mut touched: HashSet<usize> = HashSet::new();
        for &(unit, slot) in &retire.publish {
            let (gate_up, down) = self.slot_addresses(slot);
            let index = self.layer_of[unit.0 as usize].ok_or("published an unmanaged layer")?;
            self.layers[index].set(backing, unit.1 as usize, gate_up, down);
            touched.insert(index);
        }
        for &unit in &retire.unpublish {
            let index = self.layer_of[unit.0 as usize].ok_or("evicted an unmanaged layer")?;
            let layer = &mut self.layers[index];
            let (gate_up, down) = (
                layer.canon_gate_up[unit.1 as usize],
                layer.canon_down[unit.1 as usize],
            );
            layer.set(backing, unit.1 as usize, gate_up, down);
            touched.insert(index);
        }
        // PEER mode: post this retirement's publications, then take and apply what the peers
        // posted so far to the full tables' peer-owned entries.
        let mut peer_touched: HashSet<usize> = HashSet::new();
        let mut peer_take: Option<PeerTake> = None;
        if let Some(board) = self.peer.clone() {
            board.post(self.rank, j, self.publish_deltas(retire)?);
            let take = board.take(self.rank);
            if take.newest().is_some_and(|k| k > j) {
                return Err(format!(
                    "peer posts through retirement {:?} reached rank {} at its retirement {j}: \
                     the ranks' windows are out of step",
                    take.newest(),
                    self.rank
                ));
            }
            peer_touched = self.apply_peer(backing, &take);
            peer_take = Some(take);
        }
        let mut order: Vec<usize> = touched.union(&peer_touched).copied().collect();
        order.sort_unstable();
        let compute = gpu.active_stream.as_ref();
        let (mut calls, mut bytes) = (0u64, 0u64);
        for &index in &order {
            let layer = &mut self.layers[index];
            let mut layer_calls = 0u64;
            let mut upload = |image: &mut TableImage, table: &GpuTensor| -> Result<(), String> {
                let sent = image.upload_dirty(backing, gpu, table, compute)?;
                if sent > 0 {
                    layer_calls += 1;
                    bytes += sent as u64;
                }
                Ok(())
            };
            upload(&mut layer.image_gate_up, &layer.table_gate_up)?;
            upload(&mut layer.image_down, &layer.table_down)?;
            if let Some(full) = layer.full.as_mut() {
                upload(&mut full.image_gate_up, &full.table_gate_up)?;
                upload(&mut full.image_down, &full.table_down)?;
            }
            if layer_calls > 0 {
                calls += layer_calls;
                self.uploads += 1;
            }
        }
        self.upload_calls += calls;
        self.upload_bytes += bytes;
        Ok(peer_take)
    }

    /// Poll the pending fill batches in issue order (one `event_query` each; stop at the first
    /// pending): the completion tokens of the fills whose copy finished.
    fn poll_batches(&mut self, gpu: &Gpu) -> Result<Vec<FillComplete>, String> {
        let mut completed = Vec::new();
        while let Some(batch) = self.batches.pop_front() {
            let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
            let mut copy = RankCopy::new(&gpu.hip, stream, &mut self.events);
            match self.policy.complete_fills(batch, &mut copy, false) {
                Ok(done) => completed.extend(done),
                Err(FillsNotDone::Pending(batch)) => {
                    self.batches.push_front(batch);
                    break;
                }
                Err(FillsNotDone::Failed { failure, .. }) => return Err(failure.message),
            }
        }
        Ok(completed)
    }

    /// The deltas of this retirement's publications (a published fill names its slot, an evicted
    /// unit returns to canonical), for the peers.
    fn publish_deltas(&self, retire: &Retire) -> Result<Vec<PeerDelta>, String> {
        let canon = |unit: UnitKey| -> Result<[u64; 2], String> {
            let index = self.layer_of[unit.0 as usize].ok_or("a delta names an unmanaged layer")?;
            let layer = &self.layers[index];
            Ok([
                layer.canon_gate_up[unit.1 as usize],
                layer.canon_down[unit.1 as usize],
            ])
        };
        let mut deltas = Vec::with_capacity(retire.publish.len() + retire.unpublish.len());
        for &(unit, slot) in &retire.publish {
            let (gate_up, down) = self.slot_addresses(slot);
            deltas.push(PeerDelta {
                unit,
                slot: Some([gate_up, down]),
                canon: canon(unit)?,
            });
        }
        for &unit in &retire.unpublish {
            deltas.push(PeerDelta {
                unit,
                slot: None,
                canon: canon(unit)?,
            });
        }
        Ok(deltas)
    }

    /// The deltas that publish the warm start's `seeds`.
    fn seed_deltas(&self, seeds: &[(usize, UnitKey)]) -> Result<Vec<PeerDelta>, String> {
        self.publish_deltas(&Retire {
            publish: seeds.iter().map(|&(slot, unit)| (unit, slot)).collect(),
            ..Retire::default()
        })
    }

    /// Apply the peers' deltas to this rank's full-table images; the layers whose image changed.
    fn apply_peer(&mut self, backing: &TableBacking, take: &PeerTake) -> HashSet<usize> {
        let (rank, ranks) = (self.rank, self.ranks);
        let mut touched = HashSet::new();
        for (owner, delta) in &take.deltas {
            let index = self.layer_of.get(delta.unit.0 as usize).copied().flatten();
            let changed = index.and_then(|index| {
                self.layers[index]
                    .full
                    .as_mut()
                    .map(|full| (index, full.apply_peer(backing, rank, ranks, *owner, delta)))
            });
            match changed {
                Some((index, true)) => {
                    touched.insert(index);
                    self.peer_applied += 1;
                }
                _ => self.peer_skipped += 1,
            }
        }
        touched
    }

    fn freeze(&mut self, why: &str) {
        if self.frozen.is_none() {
            eprintln!(
                "qwen4 EP rank {}: D2 tail frozen at its last published table: {why}",
                self.rank
            );
            self.frozen = Some(why.to_string());
            if let Some(board) = self.peer.as_ref() {
                // This rank takes nothing more and never acknowledges again: the owners keep
                // every slot it may still name quarantined.
                board.detach(self.rank);
            }
        }
        self.calls.clear();
        self.log_used = 0;
        // The pins lever's retirement clock IS the cadence's: a freeze must never reset the
        // accumulated group (the other ranks keep theirs), nor drop this window's routed count.
        if self.pins.is_none() {
            self.reset_group();
        }
    }

    fn print_line(&self, tag: &str) {
        let s = self.policy.stats();
        let w = s.windows.max(1) as f64;
        let r = s.retirements.max(1) as f64;
        let total = (s.core_reads + s.tail_reads + s.host_reads).max(1) as f64;
        let unit_mb = (self.gate_up_bytes + self.down_bytes) as f64 / 1e6;
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let peer_note = if self.peer.is_some() {
            format!(
                " quarantined {} (now {}) peer-deltas applied {} skipped {}",
                s.quarantined,
                self.policy
                    .count(crate::expert_unit_tail::SlotPhase::Quarantined),
                self.peer_applied,
                self.peer_skipped
            )
        } else {
            String::new()
        };
        let peer_note = match self.pins.as_ref() {
            None => peer_note,
            Some(pins) => {
                let stats = pins.proto.stats();
                format!(
                    "{peer_note} pins clock {} posted {} applied {} verified {} dynamic {} \
                     plane-uploads {} ({} B){}",
                    pins.proto.clock(),
                    stats.posted,
                    stats.applied,
                    stats.verified,
                    pins.proto.planes().dynamic_pins(),
                    pins.upload_calls,
                    pins.upload_bytes,
                    pins.proto
                        .fault()
                        .map_or(String::new(), |why| format!(" PINS FAULT: {why}"))
                )
            }
        };
        eprintln!(
            "[unit-tail] rank {} {tag} windows {} retirements {} (group {:.2} windows, pending {}) reads/window {:.1}: core {:.2}% tail {:.2}% host {:.2}% \
             (slot hit {:.2}%, distinct hit {:.2}%) host distinct/window {:.2} ({:.3} MB) /retirement {:.2} \
             fills {} published {} evictions {}{} skipped {} stale {} live {}/{} late-in-flight {} \
             host-only reads {} log overflow {} | us/window: log {:.0} retire {:.0} upload {:.0} fill {:.0} (us/retirement: {:.0} {:.0} {:.0} {:.0}) ({} layer uploads: {} H2D calls, {} B){} \
             | raw unix_ms={unix_ms} retirements={} unit_bytes={} core_reads={} tail_reads={} host_reads={} core_distinct={} tail_distinct={} host_distinct={} fills_issued={} \
             fill_overlap={} fill_wait_us={:.0}",
            self.rank,
            s.windows,
            s.retirements,
            w / r,
            self.cadence.pending(),
            total / w,
            100.0 * s.core_reads as f64 / total,
            100.0 * s.tail_reads as f64 / total,
            100.0 * s.host_reads as f64 / total,
            100.0 * s.slot_hit(),
            100.0 * s.distinct_hit(),
            s.host_distinct as f64 / w,
            s.host_distinct as f64 / w * unit_mb,
            s.host_distinct as f64 / r,
            s.fills_issued,
            s.fills_published,
            s.evictions,
            peer_note,
            s.skipped_full,
            s.stale_completions,
            self.policy.count(crate::expert_unit_tail::SlotPhase::Live),
            self.policy.slots(),
            self.batches.iter().map(|b| b.fills().len()).sum::<usize>(),
            self.host_only_reads,
            self.overflowed,
            self.log_ns as f64 / w / 1e3,
            self.retire_ns as f64 / w / 1e3,
            self.upload_ns as f64 / w / 1e3,
            self.fill_ns as f64 / w / 1e3,
            self.log_ns as f64 / r / 1e3,
            self.retire_ns as f64 / r / 1e3,
            self.upload_ns as f64 / r / 1e3,
            self.fill_ns as f64 / r / 1e3,
            self.uploads,
            self.upload_calls,
            self.upload_bytes,
            self.frozen
                .as_ref()
                .map_or(String::new(), |why| format!(" FROZEN: {why}")),
            s.retirements,
            self.gate_up_bytes + self.down_bytes,
            s.core_reads,
            s.tail_reads,
            s.host_reads,
            s.core_distinct,
            s.tail_distinct,
            s.host_distinct,
            s.fills_issued,
            self.fill_overlap,
            self.fill_wait_ns as f64 / 1e3,
        );
    }

    fn release(&mut self, gpu: &mut Gpu) {
        if self.released {
            return;
        }
        self.released = true;
        // Fills that never ran: their slots stay `Filling`, which nothing reads; the bank is told
        // (`abandon_fills`) so it can close.
        let dropped = std::mem::take(&mut self.deferred);
        self.policy.abandon_fills(dropped);
        if let Some(board) = self.peer.as_ref() {
            board.detach(self.rank);
        }
        if let Some(pins) = self.pins.as_ref() {
            // The mesh is going down: the other ranks' pin steps stop quietly.
            pins.proto.close();
        }
        // The table images are pinned: no table upload (and no reader of the published
        // generation) may still run when the pool drops. The last-read fence of the leased
        // generation is the compute stream synchronize this always did; the generation then
        // retires. Without a lease (suspended, or a failed retirement) the stream is drained.
        if let Err(error) = self.fence_and_retire(gpu, false) {
            eprintln!(
                "qwen4 EP rank {}: D2 tail release: the published generation did not retire: {error}",
                self.rank
            );
        }
        // A borrow epoch still open (a poisoned close): put the pool's bytes back first, so the
        // bank can be closed.
        if let Some(lent) = self.lent.take() {
            let _ = device_idle(gpu);
            if let Err((_lent, why)) = self.restore_lent(gpu, lent) {
                eprintln!(
                    "qwen4 EP rank {}: D2 tail release: the lent pool was not restored: {why}",
                    self.rank
                );
            }
        }
        // Nothing reads a slot or the log after this: drain the copy stream first.
        if let Some(stream) = self.copy_stream.take() {
            let _ = gpu.hip.stream_synchronize(&stream);
            let _ = gpu.hip.stream_destroy(stream);
        }
        self.batches.clear();
        for event in self.events.drain() {
            let _ = gpu.hip.event_destroy(event);
        }
        if let Some(event) = self.tables_ready.take() {
            let _ = gpu.hip.event_destroy(event);
        }
        // The pool is freed only once the bank closes: every cell handed back, no live
        // generation, the device quiescent. Anything else leaks it (never an early free).
        let closed = self.close_pool(gpu);
        // The two block views alias the pool's allocation: dropping them frees nothing.
        self.pool_gate_up = None;
        self.pool_down = None;
        let pool = self.pool.take();
        if closed {
            if let Some(pool) = pool {
                let _ = gpu.free_tensor(pool);
            }
        }
        if let Some(log) = self.log.take() {
            let _ = gpu.free_tensor(log);
        }
    }

    /// Close the slot bank (device-scope quiescence, then every cell back) and release the pool
    /// block. `false` (logged) leaves the pool allocated: the bank refused, so a reader or a
    /// generation may still name it.
    fn close_pool(&mut self, gpu: &Gpu) -> bool {
        let rank = self.rank;
        let leak = |why: String| {
            eprintln!(
                "qwen4 EP rank {rank}: D2 tail release: the slot pool is NOT freed (it is leaked, never freed early): {why}"
            );
            false
        };
        let mut device = RankDevice::new(&gpu.hip);
        let quiesced = match self.policy.quiesce(&mut device) {
            Ok(quiesced) => quiesced,
            Err(failure) => return leak(failure.message),
        };
        match self.policy.close(quiesced) {
            Ok((block, reclaimed)) => match block.free(reclaimed) {
                Ok(_extent) => true,
                Err((_block, _reclaimed, error)) => leak(error.to_string()),
            },
            Err(error) => leak(error.to_string()),
        }
    }
}

/// Alignment of the pool's down block inside the single allocation (what a separate
/// `hipMalloc` gave it before the pool was one allocation).
const POOL_BLOCK_ALIGN: usize = 256;

/// Layout of the slot pool's single device allocation:
/// `[gate/up: slots * gate_up_bytes + pad | align to 256 | down: slots * down_bytes + pad]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolLayout {
    /// Bytes of the gate/up block (slots plus the over-read pad), at offset 0.
    gate_up_len: usize,
    /// Offset of the down block: `gate_up_len` rounded up to [`POOL_BLOCK_ALIGN`].
    down_offset: usize,
    /// Bytes of the down block (slots plus the over-read pad).
    down_len: usize,
    /// Bytes of the whole allocation.
    total: usize,
}

fn pool_layout(slots: usize, gate_up_bytes: usize, down_bytes: usize, pad: usize) -> PoolLayout {
    let gate_up_len = slots * gate_up_bytes + pad;
    let down_offset = gate_up_len.next_multiple_of(POOL_BLOCK_ALIGN);
    let down_len = slots * down_bytes + pad;
    PoolLayout {
        gate_up_len,
        down_offset,
        down_len,
        total: down_offset + down_len,
    }
}

/// The write extents of every slot of a pool laid out as `layout`: block-relative spans,
/// `[gate_up, down]` per slot (the two copies a fill makes).
fn pool_spans(
    slots: usize,
    gate_up_bytes: usize,
    down_bytes: usize,
    layout: &PoolLayout,
) -> Vec<Vec<SlotSpan>> {
    (0..slots)
        .map(|slot| {
            vec![
                SlotSpan {
                    offset: (slot * gate_up_bytes) as u64,
                    bytes: gate_up_bytes as u64,
                },
                SlotSpan {
                    offset: (layout.down_offset + slot * down_bytes) as u64,
                    bytes: down_bytes as u64,
                },
            ]
        })
        .collect()
}

/// A non-owning raw tensor over bytes `[offset, offset + len)` of the pool allocation `pool`.
/// Like every `byte_view` it must not outlive `pool` nor be freed.
fn pool_view(pool: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: pool.buf.byte_view(offset, len),
        shape: vec![len],
        dtype: DType::Raw,
    }
}

/// The byte image of a device pointer table holding `entries` (u64 LE per expert): the format
/// of [`TableImage`], so uploading it makes the device table equal that canonical entry list.
fn canonical_image(entries: &[u64]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .collect()
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 of `bytes`, continuing from `hash`.
fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The offset of the first byte where `a` and `b` differ (the shorter length when one is a
/// prefix of the other), `None` when equal.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len())),
    )
}

/// Synchronous upload of `bytes` to the start of the pointer table `table`; the table backing is
/// the write capability (it exists only while no generation is sealed through the tables).
fn upload_image(
    _backing: &TableBacking,
    gpu: &Gpu,
    table: &GpuTensor,
    bytes: &[u8],
) -> Result<(), String> {
    upload_plane(gpu, table, bytes)
}

/// Synchronous upload of `bytes` to the start of `table`: a pin plane (not part of any table
/// generation).
fn upload_plane(gpu: &Gpu, table: &GpuTensor, bytes: &[u8]) -> Result<(), String> {
    gpu.hip
        .memcpy_htod(&table.buf.byte_view(0, bytes.len()), bytes)
        .map_err(io)
}

/// Fence the window the leased generation was read in and hand the generation back: the window's
/// last consumer was enqueued before this call, so the fence is recorded now and waited at once
/// (`finish_window`). The compute backend's wait is the stream synchronize the window protocol
/// has always done; `device_scope` waits the whole device instead (a borrow open's drain). A
/// fence that fails or does not complete quarantines the generation (it can never retire).
fn fence_window(
    gpu: &Gpu,
    lease: WindowReadLease<'static>,
    device_scope: bool,
) -> Result<OwnedTableGeneration, String> {
    let mut compute = RankCompute::new(&gpu.hip, gpu.active_stream.as_ref());
    let mut device = RankDevice::new(&gpu.hip);
    let backend: &mut dyn FenceBackend = if device_scope {
        &mut device
    } else {
        &mut compute
    };
    let pending = lease.enqueued(backend).map_err(|(lease, failure)| {
        drop(lease);
        failure.message
    })?;
    let fenced = finish_window(pending, backend).map_err(|outcome| match outcome {
        PendingOrQuarantined::Pending(_) => {
            "the window fence is still pending after a blocking wait".to_string()
        }
        PendingOrQuarantined::Quarantined(window) => {
            format!("the window is quarantined: {}", window.failure().message)
        }
    })?;
    fenced
        .into_generation()
        .ok_or_else(|| "the fenced window does not own its generation".to_string())
}

/// Enqueue the copy of `unit`'s canonical host bytes into the two write extents `spans`
/// (`[gate_up, down]`, absolute device addresses inside `pool`) on `stream`: two
/// `memcpy_htod_async`, exactly the calls a fill has always made, and nothing outside `spans`.
#[allow(clippy::too_many_arguments)]
fn copy_unit_to_spans(
    hip: &HipRuntime,
    stream: &Stream,
    pool: &GpuTensor,
    layers: &[TailLayer],
    layer_of: &[Option<usize>],
    (gate_up_bytes, down_bytes): (usize, usize),
    unit: UnitKey,
    spans: &[Binding],
) -> Result<(), String> {
    let (layer, expert) = (unit.0 as usize, unit.1 as usize);
    let index = layer_of
        .get(layer)
        .copied()
        .flatten()
        .ok_or("a fill names an unmanaged layer")?;
    let canon = &layers[index];
    let (src_gate_up, src_down) = (canon.canon_gate_up[expert], canon.canon_down[expert]);
    if src_gate_up == 0 || src_down == 0 {
        return Err(format!("unit ({layer},{expert}) has no canonical source"));
    }
    let [gate_up_span, down_span] = spans else {
        return Err(format!(
            "a slot has {} write spans, expected 2",
            spans.len()
        ));
    };
    if gate_up_span.bytes != gate_up_bytes as u64 || down_span.bytes != down_bytes as u64 {
        return Err(format!(
            "slot spans of {} / {} bytes, the unit is {gate_up_bytes} / {down_bytes}",
            gate_up_span.bytes, down_span.bytes
        ));
    }
    let pool_base = pool.buf.as_ptr() as u64;
    let offset = |span: &Binding| -> Result<usize, String> {
        span.base
            .checked_sub(pool_base)
            .map(|offset| offset as usize)
            .ok_or_else(|| {
                format!(
                    "slot span {:#x} lies below the pool {pool_base:#x}",
                    span.base
                )
            })
    };
    // SAFETY: a managed unit the static core does not hold lives in its layer's pinned host
    // arena (the canonical entries D1 uploaded name it); that allocation outlives the forward
    // and nothing writes it after load, so the two ranges are valid, immutable host bytes for
    // the copy.
    let (gate_up, down) = unsafe {
        (
            std::slice::from_raw_parts(src_gate_up as usize as *const u8, gate_up_bytes),
            std::slice::from_raw_parts(src_down as usize as *const u8, down_bytes),
        )
    };
    hip.memcpy_htod_async(
        &pool.buf.byte_view(offset(gate_up_span)?, gate_up_bytes),
        gate_up,
        stream,
    )
    .map_err(io)?;
    hip.memcpy_htod_async(
        &pool.buf.byte_view(offset(down_span)?, down_bytes),
        down,
        stream,
    )
    .map_err(io)?;
    Ok(())
}

/// Everything the device finished: the synchronous null-stream copies of a borrow open / close
/// and the compute and copy streams, in one wait.
fn device_idle(gpu: &Gpu) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(io)
}

/// The occupied slots of `policy`'s bank as `(slot, unit, fill generation)`, ascending by slot:
/// `Resident`, `Evicting` and issued-`Filling` slots, each with the unit the bank recorded as its
/// occupant (so an evicted slot, whose bytes a peer's full table may still name, keeps its unit
/// although the policy forgets it). A `Filling` slot counts once its copy was issued; an
/// abandoned fill never does.
fn occupied_units(policy: &TailPolicy) -> Vec<(usize, UnitKey, u64)> {
    policy
        .occupied()
        .into_iter()
        .map(|(slot, occupant, generation)| (slot as usize, occupant_unit(occupant), generation))
        .collect()
}

impl Inner {
    fn pool_span(&self) -> Option<(usize, usize)> {
        if self.released {
            return None;
        }
        self.pool
            .as_ref()
            .map(|pool| (pool.buf.as_ptr() as usize, self.pool_bytes))
    }

    /// A suspended tail does nothing at a window end (no retirement, no pin step, no cadence
    /// clock): drop what the window logged, like a frozen tail, and report that the window end
    /// is skipped.
    fn skip_suspended_window(&mut self) -> bool {
        if !self.suspended {
            return false;
        }
        self.calls.clear();
        self.log_used = 0;
        true
    }

    /// The occupied slots as `(slot, unit, fill generation)`, ascending by slot: the bank's view
    /// (`Resident`, `Evicting` and issued-`Filling` slots) with the unit each holds, the bank's
    /// occupant (so an evicted slot, whose bytes a peer's full table may still name, keeps its
    /// unit). Empty once released. A `Filling` slot counts once its copy was issued, so the
    /// answer is exact after a suspend has waited every batch.
    fn occupied(&self) -> Vec<(usize, UnitKey, u64)> {
        if self.released {
            return Vec::new();
        }
        occupied_units(&self.policy)
    }

    /// The canonical pinned host bytes of `unit`'s `[gate_up, down]`.
    fn canonical_bytes(&self, unit: UnitKey) -> Result<(&[u8], &[u8]), String> {
        let (layer, expert) = (unit.0 as usize, unit.1 as usize);
        let index = self.layer_of[layer].ok_or("a slot names an unmanaged layer")?;
        let canon = &self.layers[index];
        let (src_gate_up, src_down) = (canon.canon_gate_up[expert], canon.canon_down[expert]);
        if src_gate_up == 0 || src_down == 0 {
            return Err(format!("unit ({layer},{expert}) has no canonical source"));
        }
        // SAFETY: as in `copy_unit_to_spans`: a managed unit the static core does not hold lives
        // in its layer's pinned host arena, which outlives the forward and is never written
        // after load.
        Ok(unsafe {
            (
                std::slice::from_raw_parts(src_gate_up as usize as *const u8, self.gate_up_bytes),
                std::slice::from_raw_parts(src_down as usize as *const u8, self.down_bytes),
            )
        })
    }

    fn require_suspended(&self) -> Result<(), String> {
        if self.released {
            return Err("the tail is released".into());
        }
        if !self.suspended {
            return Err("the tail is not suspended".into());
        }
        Ok(())
    }

    /// Fence the window the published generation was read in, retire the generation and keep its
    /// table backing (writable again). The fence is `finish_window` on the compute backend (the
    /// stream synchronize of the window protocol), or with `device_scope` on the device backend
    /// (`device_synchronize`: the drain a borrow open does). A generation a failed retirement
    /// already fenced (`fenced`) retires directly; without any generation (a failed retirement
    /// already retired it) the stream (device) is drained and the backing it left is kept.
    fn fence_and_retire(&mut self, gpu: &Gpu, device_scope: bool) -> Result<(), String> {
        let generation = if let Some(generation) = self.fenced.take() {
            if device_scope {
                device_idle(gpu)?;
            }
            generation
        } else if let Some(lease) = self.lease.take() {
            fence_window(gpu, lease, device_scope)?
        } else {
            if device_scope {
                device_idle(gpu)?;
            } else if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.stream_synchronize(stream).map_err(io)?;
            }
            return if self.backing.is_some() {
                Ok(())
            } else {
                Err(
                    "the tail's published table generation is gone (its window never fenced)"
                        .into(),
                )
            };
        };
        match self.policy.retire_generation(generation, None) {
            Ok((backing, _retired)) => {
                self.backing = Some(backing);
                Ok(())
            }
            Err((generation, error)) => {
                self.fenced = Some(generation);
                Err(error.to_string())
            }
        }
    }

    /// See [`UnitTailRank::borrow_suspend`]. The tail is marked suspended before the first device
    /// call so a failure leaves it in a state the close sequence can leave. Invariant afterwards:
    /// every table the tail owns holds canonical entries (decode tables: `canon_*`; full tables:
    /// their install-time `base_*`, the entries the forward uploaded BEFORE any warm-start
    /// publication (the images are built from the same entries ahead of `build_device`), the same
    /// entries [`full_shared`] / [`peer_shared`] define canonical by), so no table names a slot.
    /// The window log of the open group is dropped (the log's device bytes are not read again);
    /// the retirement cadence is left alone: ranks suspend at the same call, so it stays
    /// rank-identical and the retirement clock resumes where it stopped. Pending fill batches are
    /// kept (all complete), so the next retirement polls them and publishes them as ever.
    ///
    /// Ownership: the published generation's window fences on the device backend and the
    /// generation retires (the table backing comes back), the open's quiescence is the bank's
    /// `quiesce` (the final device drain), and the whole pool is lent as scratch
    /// ([`TailPolicy::lend_scratch`]): refused while any generation is live, a slot is
    /// quarantined or a fill ticket was never issued.
    fn borrow_suspend(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.released {
            return Err("the tail is released".into());
        }
        if self.suspended {
            return Err("the tail is already suspended".into());
        }
        self.suspended = true;
        self.calls.clear();
        self.log_used = 0;
        self.window_routed = 0;
        self.window_seen = false;
        // No table upload of the window protocol is still reading a pinned image, and no reader
        // of the published generation can still run (the device-scope window fence).
        self.fence_and_retire(gpu, true)?;
        // A frozen tail drops its deferred fills (its slots stay `Filling`, which nothing reads).
        self.issue_deferred(gpu)?;
        let stream = self.copy_stream.as_ref().ok_or("no copy stream")?;
        gpu.hip.stream_synchronize(stream).map_err(io)?;
        RankCopy::new(&gpu.hip, stream, &mut self.events).wait_all()?;
        let backing = self.backing.as_ref().ok_or("no table backing")?;
        for layer in &self.layers {
            upload_image(
                backing,
                gpu,
                &layer.table_gate_up,
                &canonical_image(&layer.canon_gate_up),
            )?;
            upload_image(
                backing,
                gpu,
                &layer.table_down,
                &canonical_image(&layer.canon_down),
            )?;
            if let Some(full) = layer.full.as_ref() {
                upload_image(
                    backing,
                    gpu,
                    &full.table_gate_up,
                    &canonical_image(&full.base_gate_up),
                )?;
                upload_image(
                    backing,
                    gpu,
                    &full.table_down,
                    &canonical_image(&full.base_down),
                )?;
            }
        }
        // The pins lever: the pin planes follow the tail's publications, so the epoch runs on
        // their STATIC planes (no unit pinned to a slot holder while the tables are canonical);
        // the live host planes are untouched and re-uploaded by `borrow_resume`.
        if let Some(pins) = self.pins.as_ref() {
            for (index, device) in pins.device.iter().enumerate() {
                let label = pins.proto.planes().label(index);
                let statics = pins
                    .proto
                    .planes()
                    .statics(label)
                    .ok_or_else(|| format!("pin plane of layer {label}: no static plane"))?;
                upload_plane(gpu, device, statics)?;
            }
        }
        // The copies above are null-stream: nothing may run ahead of them on any stream. The
        // bank's quiescence is that final device drain.
        let mut device = RankDevice::new(&gpu.hip);
        let quiesced = self
            .policy
            .quiesce(&mut device)
            .map_err(|failure| failure.message)?;
        let loan = self
            .policy
            .lend_scratch(quiesced)
            .map_err(|error| error.to_string())?;
        self.lent = Some(LentPool {
            loan,
            restore: None,
            slots: 0,
        });
        Ok(())
    }

    /// See [`UnitTailRank::borrow_restore`].
    fn borrow_restore(&mut self, gpu: &Gpu) -> Result<u64, String> {
        self.require_suspended()?;
        // Whatever used the pool (the compute stream) is done before its bytes are overwritten.
        device_idle(gpu)?;
        let Some(lent) = self.lent.take() else {
            // The pool was never lent (the open failed before the loan): nothing wrote it.
            return Ok(0);
        };
        match self.restore_lent(gpu, lent) {
            Ok(bytes) => Ok(bytes),
            Err((lent, why)) => {
                self.lent = Some(lent);
                Err(why)
            }
        }
    }

    /// Re-copy every occupied slot from its unit's canonical host bytes ([`SlotBank::issue_restore`]:
    /// ascending slot, two copies per slot, ONE fence on the copy stream), wait the fence and end
    /// the loan. On error the loan and the issued restore are returned (a retry only waits).
    fn restore_lent(&mut self, gpu: &Gpu, lent: LentPool) -> Result<u64, (LentPool, String)> {
        let (stream, pool) = match (self.copy_stream.as_ref(), self.pool.as_ref()) {
            (Some(stream), Some(pool)) => (stream, pool),
            _ => return Err((lent, "no copy stream or pool".into())),
        };
        let mut copy = RankCopy::new(&gpu.hip, stream, &mut self.events);
        let LentPool {
            loan,
            mut restore,
            mut slots,
        } = lent;
        if restore.is_none() {
            let (layers, layer_of) = (&self.layers, &self.layer_of);
            let bytes = (self.gate_up_bytes, self.down_bytes);
            let mut copied = 0u64;
            let issued =
                self.policy
                    .issue_restore(&loan, &mut copy, &mut |_slot, occupant, spans| {
                        copied += 1;
                        copy_unit_to_spans(
                            &gpu.hip,
                            stream,
                            pool,
                            layers,
                            layer_of,
                            bytes,
                            occupant_unit(occupant),
                            spans,
                        )
                    });
            match issued {
                Ok(pending) => {
                    restore = Some(pending);
                    slots = copied;
                }
                Err(failure) => {
                    return Err((
                        LentPool {
                            loan,
                            restore: None,
                            slots: 0,
                        },
                        failure.message,
                    ))
                }
            }
        }
        let pending = restore.take().expect("a restore was issued");
        match self.policy.end_scratch(loan, pending, &mut copy) {
            Ok(()) => Ok(slots * (self.gate_up_bytes + self.down_bytes) as u64),
            Err((loan, pending, error)) => Err((
                LentPool {
                    loan,
                    restore: Some(pending),
                    slots,
                },
                error.to_string(),
            )),
        }
    }

    /// See [`UnitTailRank::borrow_verify`]. The digest is FNV-1a 64 over, per occupied slot
    /// ascending: slot, layer, expert (u64 LE each), then the gate/up bytes and the down bytes.
    fn borrow_verify(&self, gpu: &Gpu) -> Result<u64, String> {
        self.require_suspended()?;
        let (pool_gate_up, pool_down) = (
            self.pool_gate_up.as_ref().ok_or("no pool")?,
            self.pool_down.as_ref().ok_or("no pool")?,
        );
        let mut device = vec![0u8; self.gate_up_bytes.max(self.down_bytes)];
        let mut digest = FNV_OFFSET;
        for (slot, unit, _generation) in self.occupied() {
            let (layer, expert) = (unit.0 as usize, unit.1 as usize);
            let (host_gate_up, host_down) = self.canonical_bytes(unit)?;
            for (what, pool, at, host) in [
                (
                    "gate/up",
                    pool_gate_up,
                    slot * self.gate_up_bytes,
                    host_gate_up,
                ),
                ("down", pool_down, slot * self.down_bytes, host_down),
            ] {
                let device = &mut device[..host.len()];
                gpu.hip
                    .memcpy_dtoh(device, &pool.buf.byte_view(at, host.len()))
                    .map_err(io)?;
                if let Some(byte) = first_difference(device, host) {
                    return Err(format!(
                        "the tail pool slot {slot} (unit layer {layer} expert {expert}) {what} block \
                         differs from its canonical host bytes at byte {byte}: device {:#04x}, host {:#04x}",
                        device.get(byte).copied().unwrap_or(0),
                        host.get(byte).copied().unwrap_or(0)
                    ));
                }
            }
            for word in [slot as u64, layer as u64, expert as u64] {
                digest = fnv1a64(digest, &word.to_le_bytes());
            }
            digest = fnv1a64(digest, host_gate_up);
            digest = fnv1a64(digest, host_down);
        }
        Ok(digest)
    }

    /// See [`UnitTailRank::borrow_resume`]. Every table gets its whole live host image (a table a
    /// slot publication changed and one that did not alike: the device holds canonical entries
    /// after the open), so no dirty range remains. Fills still in flight at the open were drained
    /// and keep their batch entries: the next retirement publishes them. The staged membership is
    /// sealed again through the backing the open's retirement returned, as generation
    /// `{ epoch, sub + 1 }` of the last sealed one, and its window lease runs until the next
    /// retirement's fence. Refused while the scratch loan is outstanding (restore first).
    fn borrow_resume(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.require_suspended()?;
        if self.lent.is_some() {
            return Err("the tail's pool is still lent: restore it before resuming".into());
        }
        let backing = self.backing.take().ok_or(
            "the tail's table backing is not held (its generation never retired): it cannot resume",
        )?;
        if let Err(error) = self.resume_upload(gpu, &backing) {
            self.backing = Some(backing);
            return Err(error);
        }
        let id = GenerationId {
            epoch: self.last_gen.epoch,
            sub: self.last_gen.sub + 1,
        };
        let generation = match self.policy.seal(backing, id) {
            Ok(generation) => generation,
            Err((backing, error)) => {
                self.backing = Some(backing);
                return Err(error.to_string());
            }
        };
        let lease = acquire_window_owned(generation).map_err(|(_g, error)| error.to_string())?;
        self.last_gen = id;
        self.lease = Some(lease);
        self.suspended = false;
        Ok(())
    }

    /// The live host images (and pin planes) to the device, then the device drain: the upload half
    /// of [`Self::borrow_resume`].
    fn resume_upload(&mut self, gpu: &mut Gpu, backing: &TableBacking) -> Result<(), String> {
        for layer in &mut self.layers {
            upload_image(backing, gpu, &layer.table_gate_up, &layer.image_gate_up)?;
            upload_image(backing, gpu, &layer.table_down, &layer.image_down)?;
            layer.image_gate_up.dirty.take();
            layer.image_down.dirty.take();
            if let Some(full) = layer.full.as_mut() {
                upload_image(backing, gpu, &full.table_gate_up, &full.image_gate_up)?;
                upload_image(backing, gpu, &full.table_down, &full.image_down)?;
                full.image_gate_up.dirty.take();
                full.image_down.dirty.take();
            }
        }
        // The pins lever: the live planes again (what the last retirement left); nothing the
        // epoch did changed the host planes (a suspended window never retires).
        if let Some(pins) = self.pins.as_mut() {
            let _ = pins.proto.planes_mut().take_dirty();
            for (index, device) in pins.device.iter().enumerate() {
                upload_plane(gpu, device, pins.proto.planes().bytes(index))?;
            }
        }
        device_idle(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_peer_delta, canonical_image, changed_entry_range, entry_span, first_difference,
        fnv1a64, full_shared, image_entry, live_slot_units, occupied_units, peer_shared,
        pin_deltas, pool_layout, tail_layer_selected, DirtyRange, EventPool, GenerationId, Inner,
        LiveSlot, ENTRY_BYTES, FNV_OFFSET, LOG_IDS, POOL_BLOCK_ALIGN,
    };
    use crate::expert_unit_tail::rig::{rig_pool, Rig};
    use crate::expert_unit_tail::{
        EpUnitTailConfig, PeerBoard, PeerDelta, PinDelta, Retire, RetireCadence, SlotPhase,
        TailPolicy, UnitKey, WindowUse,
    };
    use std::collections::HashMap;

    fn image(entries: &[u64]) -> Vec<u8> {
        entries.iter().flat_map(|e| e.to_le_bytes()).collect()
    }

    #[test]
    fn live_slots_are_the_tail_owned_units_whose_image_leaves_canonical() {
        let canon_gu = [100u64, 200, 300, 400];
        let canon_dn = [10u64, 20, 30, 40];
        // Expert 1: published (both entries name a slot). Expert 2: evicted back to canonical.
        // Expert 3: a slot, but not a tail unit (core / foreign): never reported.
        let mut img_gu = canon_gu;
        let mut img_dn = canon_dn;
        img_gu[1] = 7000;
        img_dn[1] = 8000;
        img_gu[3] = 7100;
        img_dn[3] = 8100;
        let live = live_slot_units(
            |expert| expert != 3,
            (&canon_gu, &canon_dn),
            (&image(&img_gu), &image(&img_dn)),
        );
        assert_eq!(
            live,
            [LiveSlot {
                expert: 1,
                canon: [200, 20],
                slot: [7000, 8000],
            }]
        );
    }

    #[test]
    fn no_publication_means_no_live_slot() {
        let canon_gu = [1u64, 2];
        let canon_dn = [3u64, 4];
        let live = live_slot_units(
            |_| true,
            (&canon_gu, &canon_dn),
            (&image(&canon_gu), &image(&canon_dn)),
        );
        assert!(live.is_empty());
    }

    #[test]
    fn full_table_shares_only_owned_experts_with_equal_canonical_entries() {
        // 6 experts, 2 ranks, this rank 1: owned = 1, 3, 5.
        let decode_gu = [10u64, 11, 12, 13, 14, 15];
        let decode_dn = [20u64, 21, 22, 23, 24, 25];
        let mut full_gu = decode_gu;
        let mut full_dn = decode_dn;
        // Expert 3: the full table names a different gate/up pointer; expert 5: a different down.
        full_gu[3] = 99;
        full_dn[5] = 99;
        // Expert 0 (peer-owned) is equal in both tables but must never be shared.
        let shared = full_shared(1, 2, (&decode_gu, &decode_dn), (&full_gu, &full_dn));
        assert_eq!(shared, [false, true, false, false, false, false]);
    }

    #[test]
    fn full_table_never_shares_a_null_entry() {
        let decode_gu = [0u64, 7];
        let decode_dn = [5u64, 0];
        let shared = full_shared(0, 1, (&decode_gu, &decode_dn), (&decode_gu, &decode_dn));
        assert_eq!(shared, [false, false]);
    }

    #[test]
    fn peer_shared_admits_only_a_peer_owned_entry_equal_to_the_owner_canonical() {
        // 2 ranks, this rank 1: rank 0 owns the even experts.
        let canon = [100u64, 200];
        assert!(peer_shared(1, 2, 0, 4, canon, canon));
        // The full entry differs from the owner's canonical entries in either pointer.
        assert!(!peer_shared(1, 2, 0, 4, canon, [100, 201]));
        assert!(!peer_shared(1, 2, 0, 4, canon, [101, 200]));
        // Not a peer: this rank's own entries are `full_shared`'s.
        assert!(!peer_shared(1, 2, 1, 5, canon, canon));
        // The expert is not owned by the delta's owner.
        assert!(!peer_shared(1, 2, 0, 5, canon, canon));
        assert!(!peer_shared(1, 2, 2, 4, canon, canon));
        // A null canonical pointer is never followed.
        assert!(!peer_shared(1, 2, 0, 4, [0, 200], [0, 200]));
        assert!(!peer_shared(1, 2, 0, 4, [100, 0], [100, 0]));
    }

    #[test]
    fn a_peer_delta_never_touches_an_entry_that_differs_from_the_owner_canonical() {
        // 6 experts, 2 ranks, this rank 1: rank 0 owns experts 0, 2 and 4.
        let owner_gu = [10u64, 11, 12, 13, 14, 15];
        let owner_dn = [20u64, 21, 22, 23, 24, 25];
        // This rank's full table (install-time entries): expert 2 names something else than the
        // owner's canonical entries (a different residency view); expert 4 equals them.
        let mut full_gu = owner_gu;
        full_gu[2] = 99;
        let full_dn = owner_dn;
        let mut gu = image(&full_gu);
        let mut dn = image(&full_dn);
        let (start_gu, start_dn) = (gu.clone(), dn.clone());
        let apply = |gu: &mut Vec<u8>, dn: &mut Vec<u8>, owner: usize, delta: PeerDelta| {
            apply_peer_delta(
                1,
                2,
                owner,
                (&full_gu, &full_dn),
                (gu.as_mut_slice(), dn.as_mut_slice()),
                &delta,
            )
        };
        let delta = |expert: u16, slot: Option<[u64; 2]>| PeerDelta {
            unit: (0, expert),
            slot,
            canon: [owner_gu[expert as usize], owner_dn[expert as usize]],
        };
        // The differing entry is never written, by a publication or an unpublication.
        assert!(!apply(&mut gu, &mut dn, 0, delta(2, Some([7000, 8000]))));
        assert!(!apply(&mut gu, &mut dn, 0, delta(2, None)));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
        // The equal entry follows its owner: both pointers name the slot, then return to the
        // canonical entries; no neighbour moves.
        assert!(apply(&mut gu, &mut dn, 0, delta(4, Some([7100, 8100]))));
        assert_eq!((image_entry(&gu, 4), image_entry(&dn, 4)), (7100, 8100));
        for expert in [0usize, 1, 2, 3, 5] {
            assert_eq!(image_entry(&gu, expert), image_entry(&start_gu, expert));
            assert_eq!(image_entry(&dn, expert), image_entry(&start_dn, expert));
        }
        assert!(
            !apply(&mut gu, &mut dn, 0, delta(4, Some([7100, 8100]))),
            "idempotent"
        );
        assert!(apply(&mut gu, &mut dn, 0, delta(4, None)));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
        // This rank's own experts (1, 3, 5), a delta that claims this rank as its owner, an
        // expert outside the table and a null canonical entry are all ignored.
        assert!(!apply(&mut gu, &mut dn, 0, delta(3, Some([1, 2]))));
        assert!(!apply(&mut gu, &mut dn, 1, delta(3, Some([1, 2]))));
        let beyond = PeerDelta {
            unit: (0, 6),
            slot: Some([1, 2]),
            canon: [16, 26],
        };
        assert!(!apply(&mut gu, &mut dn, 0, beyond));
        let mut null = delta(4, Some([1, 2]));
        null.canon = [0, 0];
        assert!(!apply(&mut gu, &mut dn, 0, null));
        assert_eq!((&gu, &dn), (&start_gu, &start_dn));
    }

    // ---- table uploads: only the changed entry range of a retirement ----

    #[test]
    fn changed_entry_range_spans_the_first_to_the_last_differing_entry() {
        let old = image(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(changed_entry_range(&old, &old), None);
        assert_eq!(
            changed_entry_range(&old, &image(&[1, 2, 9, 4, 5, 6])),
            Some((2, 2))
        );
        assert_eq!(
            changed_entry_range(&old, &image(&[7, 2, 3, 4, 5, 8])),
            Some((0, 5))
        );
        // Entries in between that did not change are inside the range (one call, not two).
        assert_eq!(
            changed_entry_range(&old, &image(&[1, 9, 3, 4, 9, 6])),
            Some((1, 4))
        );
        // A change in one byte of an entry counts as that entry.
        let mut bytes = old.clone();
        bytes[3 * ENTRY_BYTES + 7] ^= 1;
        assert_eq!(changed_entry_range(&old, &bytes), Some((3, 3)));
        assert_eq!(entry_span((1, 4)), (8, 32));
        assert_eq!(entry_span((0, 0)), (0, 8));
        assert_eq!(changed_entry_range(&[], &[]), None);
    }

    #[test]
    fn the_dirty_range_of_a_retirement_is_the_changed_range_and_uploading_it_syncs_the_device() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let entries = 40usize;
        let mut host = image(&(0..entries as u64).map(|e| 0x1000 + e).collect::<Vec<_>>());
        let mut device = host.clone();
        let mut dirty = DirtyRange::default();
        assert_eq!(dirty.take(), None);
        for retirement in 0..300 {
            // A retirement rewrites a handful of entries (some to their current value).
            let writes = (rand(&mut state) % 5) as usize;
            for _ in 0..writes {
                let expert = (rand(&mut state) % entries as u64) as usize;
                let value = if rand(&mut state) % 4 == 0 {
                    u64::from_le_bytes(
                        host[expert * ENTRY_BYTES..(expert + 1) * ENTRY_BYTES]
                            .try_into()
                            .unwrap(),
                    )
                } else {
                    rand(&mut state)
                };
                // The write the pinned image does: noted only when the bytes change.
                let at = expert * ENTRY_BYTES;
                if host[at..at + ENTRY_BYTES] != value.to_le_bytes() {
                    host[at..at + ENTRY_BYTES].copy_from_slice(&value.to_le_bytes());
                    dirty.note(expert);
                }
            }
            let expected = changed_entry_range(&device, &host);
            let range = dirty.take();
            assert_eq!(range, expected, "retirement {retirement}");
            if let Some(range) = range {
                let (offset, len) = entry_span(range);
                device[offset..offset + len].copy_from_slice(&host[offset..offset + len]);
            }
            assert_eq!(device, host, "retirement {retirement}");
            assert_eq!(dirty.take(), None);
        }
    }

    // ---- two-rank lockstep simulation of the board and both policies ----

    const SIM_RANKS: usize = 2;
    const SIM_LAYERS: u16 = 3;
    const SIM_EXPERTS: u16 = 16;

    /// The owner's canonical `[gate_up, down]` entries (its pinned host arena) of a unit.
    fn sim_canon(unit: UnitKey) -> [u64; 2] {
        let id = unit.0 as u64 * SIM_EXPERTS as u64 + unit.1 as u64;
        [
            0x7000_0000_0000 + id * 0x4000,
            0x7000_0000_2000 + id * 0x4000,
        ]
    }

    const SIM_POOL: u64 = 0x1000_0000_0000;
    const SIM_POOL_SPAN: u64 = 0x1_0000_0000;

    /// The addresses of `owner`'s pool slot (VRAM; one slot is 0x4000 bytes apart).
    fn sim_slot_addr(owner: usize, slot: usize) -> [u64; 2] {
        let base = SIM_POOL + owner as u64 * SIM_POOL_SPAN + slot as u64 * 0x4000;
        [base, base + 0x2000]
    }

    fn sim_slot_of(address: u64) -> Option<(usize, usize)> {
        (0..SIM_RANKS).find_map(|owner| {
            let base = SIM_POOL + owner as u64 * SIM_POOL_SPAN;
            (base..base + SIM_POOL_SPAN)
                .contains(&address)
                .then(|| (owner, ((address - base) / 0x4000) as usize))
        })
    }

    /// The install-time full-table entries `reader` holds for a unit: the owner's canonical
    /// entries, except some peer-owned ones that differ (another residency view).
    fn sim_base(reader: usize, unit: UnitKey) -> [u64; 2] {
        let canon = sim_canon(unit);
        if unit.1 as usize % SIM_RANKS != reader && (unit.0 + unit.1) % 5 == 3 {
            [canon[0] + 8, canon[1] + 8]
        } else {
            canon
        }
    }

    fn rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    struct SimLayer {
        base_gu: Vec<u64>,
        base_dn: Vec<u64>,
        img_gu: Vec<u8>,
        img_dn: Vec<u8>,
    }

    struct SimRank {
        policy: Rig,
        /// This rank's full-table images, one per layer.
        layers: Vec<SimLayer>,
        /// In-flight fills: `(slot, generation, retirements left)`.
        flight: Vec<(usize, u64, u32)>,
        /// `(owner, k)` -> the retirement at which this rank first applied the owner's post `k`.
        applied_at: HashMap<(usize, u64), u64>,
        covered: Option<u64>,
        /// Slots this rank's policy quarantined -> the retirement they were evicted at.
        quarantined_at: HashMap<usize, u64>,
    }

    #[derive(Default, Debug)]
    struct SimTotals {
        evictions: u64,
        quarantined_refills: u64,
        image_changes: u64,
        published: u64,
    }

    struct Sim {
        ranks: Vec<SimRank>,
        board: PeerBoard,
        /// `pool[owner][slot]`: the unit whose bytes the slot holds, or is being overwritten with.
        pool: Vec<Vec<Option<UnitKey>>>,
        rng: u64,
        quarantine: bool,
        totals: SimTotals,
    }

    impl Sim {
        fn new(t: usize, f: usize, b: u32, quarantine: bool, seed: u64) -> Self {
            let board = PeerBoard::new(SIM_RANKS);
            let (mut ranks, mut pool) = (Vec::new(), Vec::new());
            for r in 0..SIM_RANKS {
                let mut priority = HashMap::new();
                let mut owned: Vec<UnitKey> = Vec::new();
                for layer in 0..SIM_LAYERS {
                    for expert in (r as u16..SIM_EXPERTS).step_by(SIM_RANKS) {
                        priority.insert((layer, expert), layer as u32 * 100 + expert as u32);
                        owned.push((layer, expert));
                    }
                }
                let config = EpUnitTailConfig {
                    tail_units: t,
                    fill_per_step: f,
                    boundary_every: b,
                };
                let mut policy = Rig::with_peer(config, false, priority, quarantine);
                let seeds = policy.seed(&owned);
                let mut held = vec![None; t];
                for &(slot, unit, _generation) in &seeds {
                    held[slot] = Some(unit);
                }
                // The warm start posts as retirement 0, like `UnitTailRank::install`.
                board.post(
                    r,
                    0,
                    seeds
                        .iter()
                        .map(|&(slot, unit, _generation)| PeerDelta {
                            unit,
                            slot: Some(sim_slot_addr(r, slot)),
                            canon: sim_canon(unit),
                        })
                        .collect(),
                );
                let layers = (0..SIM_LAYERS)
                    .map(|layer| {
                        let (mut base_gu, mut base_dn) = (Vec::new(), Vec::new());
                        for expert in 0..SIM_EXPERTS {
                            let entry = sim_base(r, (layer, expert));
                            base_gu.push(entry[0]);
                            base_dn.push(entry[1]);
                        }
                        SimLayer {
                            img_gu: image(&base_gu),
                            img_dn: image(&base_dn),
                            base_gu,
                            base_dn,
                        }
                    })
                    .collect();
                ranks.push(SimRank {
                    policy,
                    layers,
                    flight: Vec::new(),
                    applied_at: HashMap::new(),
                    covered: None,
                    quarantined_at: HashMap::new(),
                });
                pool.push(held);
            }
            Self {
                ranks,
                board,
                pool,
                rng: seed | 1,
                quarantine,
                totals: SimTotals::default(),
            }
        }

        /// One rank's retirement of window `w`, in the production order: its policy step, its
        /// post, its fills, then the take / apply / upload / ack of the peers' deltas.
        fn retire(&mut self, r: usize, w: u64, windows: u32) -> Result<(), String> {
            let Sim {
                ranks,
                board,
                pool,
                rng,
                quarantine,
                totals,
            } = self;
            // The window's reads of this rank's own units, skewed to a drifting hot set.
            let owned = SIM_EXPERTS as usize / SIM_RANKS;
            let mut uses: Vec<WindowUse> = Vec::new();
            for _ in 0..8 * windows {
                let layer = (rand(rng) % SIM_LAYERS as u64) as u16;
                let idx = if rand(rng) % 3 != 0 {
                    (rand(rng) as usize % 4 + (w / 40) as usize) % owned
                } else {
                    rand(rng) as usize % owned
                };
                let unit = (layer, (r + idx * SIM_RANKS) as u16);
                let reads = 1 + (rand(rng) % 3) as u32;
                match uses.iter_mut().find(|u| u.unit == unit) {
                    Some(u) => u.reads += reads,
                    None => uses.push(WindowUse { unit, reads }),
                }
            }
            uses.sort_by_key(|u| u.unit);
            let mut completed: Vec<(usize, u64)> = Vec::new();
            ranks[r].flight.retain_mut(|(slot, generation, left)| {
                if *left == 0 {
                    completed.push((*slot, *generation));
                    false
                } else {
                    *left -= 1;
                    true
                }
            });
            if rand(rng) % 17 == 0 {
                completed.push((0, 1 << 40));
            }
            let retire = if *quarantine {
                ranks[r]
                    .policy
                    .retire_peer_windows((0, 0), &uses, &completed, windows, |m| {
                        board.released_through(r, m)
                    })
            } else {
                ranks[r]
                    .policy
                    .retire_windows((0, 0), &uses, &completed, windows)
            };
            let j = ranks[r].policy.stats().retirements;
            if j != w {
                return Err(format!("rank {r}: retirement {j} at window {w}"));
            }
            totals.published += retire.publish.len() as u64;
            totals.evictions = ranks.iter().map(|k| k.policy.stats().evictions).sum();
            for slot in 0..ranks[r].policy.slots() {
                if let Some(k) = ranks[r].policy.slot_evicted_at(slot) {
                    ranks[r].quarantined_at.entry(slot).or_insert(k);
                }
            }
            // Post this retirement's publications.
            let mut deltas = Vec::new();
            for &(unit, slot) in &retire.publish {
                deltas.push(PeerDelta {
                    unit,
                    slot: Some(sim_slot_addr(r, slot)),
                    canon: sim_canon(unit),
                });
            }
            for &unit in &retire.unpublish {
                deltas.push(PeerDelta {
                    unit,
                    slot: None,
                    canon: sim_canon(unit),
                });
            }
            board.post(r, j, deltas);
            // The fills this retirement starts: a slot that was quarantined must have been
            // acknowledged by EVERY peer at a retirement strictly before this one.
            for op in &retire.fills {
                let (op_slot, op_unit, op_generation) = *op;
                if let Some(k) = ranks[r].quarantined_at.remove(&op_slot) {
                    for q in (0..SIM_RANKS).filter(|&q| q != r) {
                        match ranks[q].applied_at.get(&(r, k)) {
                            Some(&applied) if j > applied => {}
                            other => {
                                return Err(format!(
                                    "owner {r}: slot {op_slot} evicted at {k} refilled at {j}, peer {q} \
                                     applied the unpublication at {other:?}"
                                ))
                            }
                        }
                    }
                    totals.quarantined_refills += 1;
                }
                pool[r][op_slot] = Some(op_unit);
                let latency = (rand(rng) % 4) as u32;
                ranks[r].flight.push((op_slot, op_generation, latency));
            }
            // Take every peer delta posted so far, apply it to the full-table images, "upload",
            // acknowledge.
            let take = board.take(r);
            if let Some(top) = take.newest() {
                if top > j {
                    return Err(format!("rank {r}: peer post {top} ahead of retirement {j}"));
                }
                let owner = (r + 1) % SIM_RANKS;
                let from = ranks[r].covered.map_or(0, |covered| covered + 1);
                for k in from..=top {
                    ranks[r].applied_at.insert((owner, k), j);
                }
                ranks[r].covered = Some(top);
            }
            for (owner, delta) in &take.deltas {
                let layer = &mut ranks[r].layers[delta.unit.0 as usize];
                if apply_peer_delta(
                    r,
                    SIM_RANKS,
                    *owner,
                    (&layer.base_gu, &layer.base_dn),
                    (&mut layer.img_gu, &mut layer.img_dn),
                    delta,
                ) {
                    totals.image_changes += 1;
                }
            }
            board.ack(r, j, &take);
            Ok(())
        }

        /// Every peer-owned full-table entry that left its install-time value names a slot of
        /// its owner that holds exactly that unit's bytes right now (published and not being
        /// overwritten), and an entry that differs from the owner's canonical one never moved.
        fn check(&self) -> Result<(), String> {
            for (q, reader) in self.ranks.iter().enumerate() {
                for (layer, table) in reader.layers.iter().enumerate() {
                    for expert in 0..SIM_EXPERTS as usize {
                        let owner = expert % SIM_RANKS;
                        if owner == q {
                            continue;
                        }
                        let unit = (layer as u16, expert as u16);
                        let entry = [
                            image_entry(&table.img_gu, expert),
                            image_entry(&table.img_dn, expert),
                        ];
                        let base = [table.base_gu[expert], table.base_dn[expert]];
                        if entry == base {
                            continue;
                        }
                        if base != sim_canon(unit) {
                            return Err(format!("rank {q}: unshared entry of {unit:?} moved"));
                        }
                        let Some((slot_owner, slot)) = sim_slot_of(entry[0]) else {
                            return Err(format!("rank {q}: {unit:?} names {entry:x?}"));
                        };
                        if slot_owner != owner || entry != sim_slot_addr(owner, slot) {
                            return Err(format!("rank {q}: {unit:?} names a foreign slot"));
                        }
                        if self.pool[owner][slot] != Some(unit) {
                            return Err(format!(
                                "rank {q} names slot {slot} of owner {owner} for {unit:?}, which \
                                 holds {:?}",
                                self.pool[owner][slot]
                            ));
                        }
                        match self.ranks[owner].policy.slot_phase(slot) {
                            SlotPhase::Live | SlotPhase::Quarantined => {}
                            phase => {
                                return Err(format!(
                                    "rank {q} names slot {slot} of owner {owner} in {phase:?}"
                                ))
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    /// The retirement schedule of `windows` decode windows under the cadence: both ranks run
    /// their own [`RetireCadence`] over the same routed-id sequence (the windows' `slots` sums;
    /// NOT what either rank logged) and must retire the same groups.
    fn cadence_schedule(seed: u64, windows: u64, retire_ids: u64) -> Result<Vec<u32>, String> {
        let mut state = seed | 1;
        let mut cadences = [
            RetireCadence::new(retire_ids, 1 << 18),
            RetireCadence::new(retire_ids, 1 << 18),
        ];
        let mut groups = Vec::new();
        for window in 1..=windows {
            // 64 ids (a C1 AR step) .. 3000 (a C8 verify window).
            let routed = 64 + rand(&mut state) % 2900;
            let decided = [
                cadences[0].window_end(routed),
                cadences[1].window_end(routed),
            ];
            if decided[0] != decided[1] {
                return Err(format!("window {window}: ranks decided {decided:?}"));
            }
            groups.extend(decided[0]);
        }
        Ok(groups)
    }

    fn run_sim(
        t: usize,
        f: usize,
        b: u32,
        quarantine: bool,
        seed: u64,
        windows: u64,
    ) -> Result<SimTotals, String> {
        run_sim_groups(t, f, b, quarantine, seed, &vec![1; windows as usize])
    }

    /// `groups[k]`: the windows of the `k`th retirement (both ranks retire the same groups).
    fn run_sim_groups(
        t: usize,
        f: usize,
        b: u32,
        quarantine: bool,
        seed: u64,
        groups: &[u32],
    ) -> Result<SimTotals, String> {
        let mut sim = Sim::new(t, f, b, quarantine, seed);
        sim.check()?;
        for (k, &windows) in groups.iter().enumerate() {
            let w = k as u64 + 1;
            // Both ranks retire every window; which one goes first is arbitrary.
            let order = if rand(&mut sim.rng) & 1 == 0 {
                [0, 1]
            } else {
                [1, 0]
            };
            for r in order {
                sim.retire(r, w, windows)?;
                sim.check()?;
            }
        }
        Ok(sim.totals)
    }

    /// The lever's safety property over random windows, random copy latency, stale completions
    /// and either rank retiring first: no peer full-table entry ever names a slot that is not
    /// live (or quarantined) with exactly that unit's bytes, in particular no slot is refilled
    /// while a peer image names it, a refill follows every peer's ack by at least one
    /// retirement, and an entry that differs from the owner's canonical one never moves.
    #[test]
    fn two_rank_lockstep_never_refills_a_slot_a_peer_image_names() {
        let mut totals = SimTotals::default();
        for (t, f, b) in [
            (4usize, 2usize, 1u32),
            (6, 3, 1),
            (3, 1, 1),
            (5, 2, 2),
            (10, 4, 1),
            (8, 8, 1),
        ] {
            for seed in [
                0x9e37_79b9_7f4a_7c15u64,
                0x2545_f491_4f6c_dd1d,
                0x1234_5678_9abc_def1,
            ] {
                let run = run_sim(t, f, b, true, seed, 500)
                    .unwrap_or_else(|error| panic!("T={t} F={f} b={b} seed={seed:#x}: {error}"));
                totals.evictions += run.evictions;
                totals.quarantined_refills += run.quarantined_refills;
                totals.image_changes += run.image_changes;
                totals.published += run.published;
            }
        }
        // The runs exercised the protocol: units got published, peers' images followed them,
        // slots were evicted into quarantine and refilled after the acks.
        assert!(
            totals.published > 0 && totals.image_changes > 0,
            "{totals:?}"
        );
        assert!(
            totals.evictions > 0 && totals.quarantined_refills > 0,
            "{totals:?}"
        );
    }

    /// Without the pins lever the forward hands the tail exactly the static-owner host layers it
    /// always did (whatever pin planes exist); with it, every layer that has a pin plane too.
    #[test]
    fn the_layer_selector_without_the_pins_lever_is_the_static_owner_rule() {
        for static_owners in [false, true] {
            for host_mapped in [false, true] {
                for has_pin_plane in [false, true] {
                    assert_eq!(
                        tail_layer_selected(static_owners, host_mapped, false, has_pin_plane),
                        static_owners && host_mapped,
                        "so={static_owners} host={host_mapped} plane={has_pin_plane}"
                    );
                }
            }
        }
        // With the lever a pinned unit layer is handed over whatever it holds on this rank (a
        // rank whose layer is wholly resident still applies the pin deltas to its plane) ...
        assert!(tail_layer_selected(false, false, true, true));
        assert!(tail_layer_selected(false, true, true, true));
        // ... a layer without a plane is not, and a static-owner layer is handed over as ever.
        assert!(!tail_layer_selected(false, true, true, false));
        assert!(tail_layer_selected(true, true, true, false));
        assert!(!tail_layer_selected(true, false, true, false));
    }

    #[test]
    fn a_retirement_posts_its_publications_then_its_evictions_as_pin_deltas() {
        let retire = Retire {
            publish: vec![((0, 4), 2), ((1, 6), 0)],
            unpublish: vec![(0, 8), (2, 2)],
            fills: Vec::new(),
        };
        assert_eq!(
            pin_deltas(&retire),
            [
                PinDelta {
                    unit: (0, 4),
                    on: true
                },
                PinDelta {
                    unit: (1, 6),
                    on: true
                },
                PinDelta {
                    unit: (0, 8),
                    on: false
                },
                PinDelta {
                    unit: (2, 2),
                    on: false
                },
            ]
        );
        assert!(pin_deltas(&Retire::default()).is_empty());
    }

    /// The same safety property when the ranks retire GROUPS of windows: the groups come from
    /// two cadences fed the same routed-id sequence (they must agree), each retirement judges a
    /// group of up to 8 windows with a fill budget of up to 4F, and the quarantine bound scales
    /// with that budget.
    #[test]
    fn grouped_retirements_keep_the_peer_quarantine_safe() {
        let mut totals = SimTotals::default();
        let mut grouped = 0usize;
        for (t, f, b) in [
            (4usize, 2usize, 1u32),
            (6, 3, 1),
            (3, 1, 1),
            (5, 2, 2),
            (10, 4, 1),
            (8, 8, 1),
        ] {
            for seed in [
                0x9e37_79b9_7f4a_7c15u64,
                0x2545_f491_4f6c_dd1d,
                0x1234_5678_9abc_def1,
            ] {
                for retire_ids in [4096u64, 12_000, 40_000] {
                    let groups = cadence_schedule(seed, 600, retire_ids).unwrap();
                    grouped += groups.iter().filter(|&&n| n > 1).count();
                    let run =
                        run_sim_groups(t, f, b, true, seed, &groups).unwrap_or_else(|error| {
                            panic!("T={t} F={f} b={b} ids={retire_ids} seed={seed:#x}: {error}")
                        });
                    totals.evictions += run.evictions;
                    totals.quarantined_refills += run.quarantined_refills;
                    totals.image_changes += run.image_changes;
                    totals.published += run.published;
                }
            }
        }
        assert!(grouped > 0, "no retirement covered more than one window");
        assert!(
            totals.published > 0 && totals.image_changes > 0,
            "{totals:?}"
        );
        assert!(
            totals.evictions > 0 && totals.quarantined_refills > 0,
            "{totals:?}"
        );
    }

    /// The same runs with the pre-peer eviction (a victim is refilled in the retirement that
    /// evicts it) trip the invariant: the simulation can see the hazard the quarantine closes.
    #[test]
    fn without_quarantine_a_peer_image_names_a_refilled_slot() {
        let broke = (1..=4u64)
            .any(|seed| run_sim(4, 2, 1, false, seed * 0x9e37_79b9_7f4a_7c15, 300).is_err());
        assert!(broke, "the invariant never fired without the quarantine");
    }

    #[test]
    fn the_canonical_image_is_the_table_image_byte_format() {
        let entries = [0u64, 0x7f00_0000_1000, u64::MAX, 42];
        let bytes = canonical_image(&entries);
        assert_eq!(bytes, image(&entries));
        assert_eq!(bytes.len(), entries.len() * ENTRY_BYTES);
        for (expert, &entry) in entries.iter().enumerate() {
            assert_eq!(image_entry(&bytes, expert), entry);
        }
        assert!(canonical_image(&[]).is_empty());
    }

    #[test]
    fn the_pool_is_one_allocation_with_an_aligned_down_block_after_the_gate_up_block() {
        let pad = 256;
        let layout = pool_layout(3, 1000, 500, pad);
        assert_eq!(layout.gate_up_len, 3 * 1000 + pad);
        assert_eq!(layout.down_offset, 3328);
        assert_eq!(layout.down_len, 3 * 500 + pad);
        assert_eq!(layout.total, layout.down_offset + layout.down_len);
        // Already aligned: no gap.
        let layout = pool_layout(4, 1024, 512, pad);
        assert_eq!(layout.down_offset, layout.gate_up_len);
        for (slots, gate_up, down) in [
            (1usize, 1usize, 1usize),
            (7, 3333, 1111),
            (1536, 2_359_296, 1_179_648),
        ] {
            let layout = pool_layout(slots, gate_up, down, pad);
            assert_eq!(layout.down_offset % POOL_BLOCK_ALIGN, 0);
            assert!(layout.down_offset >= layout.gate_up_len);
            assert!(layout.down_offset - layout.gate_up_len < POOL_BLOCK_ALIGN);
            // Both blocks keep their over-read pad and never overlap.
            assert!(layout.gate_up_len >= slots * gate_up + pad);
            assert!(layout.down_len >= slots * down + pad);
            assert_eq!(layout.total, layout.down_offset + layout.down_len);
        }
    }

    /// An `Inner` with no layer and no device state: enough for the policy-side borrow rules.
    fn bare(policy: TailPolicy, suspended: bool) -> Inner {
        Inner {
            rank: 0,
            ranks: 1,
            experts: 64,
            trunk_layers: 4,
            layer_of: vec![None; 4],
            layers: Vec::new(),
            kinds: Vec::new(),
            gate_up_bytes: 16,
            down_bytes: 8,
            pool: None,
            pool_bytes: 0,
            pool_gate_up: None,
            pool_down: None,
            copy_stream: None,
            tables_ready: None,
            batches: Default::default(),
            events: EventPool::new(),
            log: None,
            log_host: Vec::new(),
            log_used: 0,
            calls: Vec::new(),
            cadence: RetireCadence::new(4096, LOG_IDS as u64),
            window_routed: 0,
            window_seen: false,
            reads: Vec::new(),
            touched: Vec::new(),
            overflowed: 0,
            policy,
            frozen: None,
            log_enabled: false,
            host_only_reads: 0,
            retire_ns: 0,
            log_ns: 0,
            upload_ns: 0,
            fill_ns: 0,
            fill_wait_ns: 0,
            fill_overlap: false,
            deferred: Vec::new(),
            uploads: 0,
            upload_calls: 0,
            upload_bytes: 0,
            peer: None,
            peer_applied: 0,
            peer_skipped: 0,
            pins: None,
            released: false,
            suspended,
            backing: None,
            lease: None,
            fenced: None,
            last_gen: GenerationId { epoch: 0, sub: 0 },
            lent: None,
            skew: None,
        }
    }

    /// A one-slot policy over the rig's pool block (no device).
    fn bare_policy() -> TailPolicy {
        let (block, spans) = rig_pool(1);
        TailPolicy::new(
            EpUnitTailConfig {
                tail_units: 1,
                fill_per_step: 1,
                boundary_every: 1,
            },
            false,
            HashMap::new(),
            false,
            block,
            spans,
        )
        .map_err(|(_, error)| error)
        .expect("a one-slot policy")
    }

    fn occupied_of(rig: &Rig) -> Vec<(usize, UnitKey)> {
        occupied_units(&rig.policy)
            .into_iter()
            .map(|(slot, unit, _generation)| (slot, unit))
            .collect()
    }

    /// The borrow restore's occupancy is the bank's: published and evicted-but-quarantined slots
    /// (the bank remembers the unit of an evicted slot, which the policy forgets), and a
    /// `Filling` slot only once its copy was issued; an abandoned fill (a frozen or released
    /// tail's dropped ticket) stays unoccupied. Replaces the old `slot_is_occupied` phase table.
    #[test]
    fn occupied_slots_are_the_resident_and_evicting_ones_plus_filling_only_once_issued() {
        let mut priority = HashMap::new();
        for expert in 0..16u16 {
            priority.insert((0u16, expert), expert as u32);
        }
        let mut rig = Rig::with_peer(
            EpUnitTailConfig {
                tail_units: 3,
                fill_per_step: 4,
                boundary_every: 1,
            },
            false,
            priority,
            true,
        );
        rig.seed(&[(0, 1), (0, 2)]);
        // Two live seeds, the third slot free.
        assert_eq!(occupied_of(&rig), vec![(0, (0, 1)), (1, (0, 2))]);

        // A miss fills the free slot: Filling with its ticket not issued yet (fill overlap's
        // deferred state): not occupied.
        rig.defer_issue(true);
        let retire = rig.retire_peer((0, 0), &[use_(0, 9)], &[], |_| 0);
        assert_eq!(retire.fills.len(), 1);
        assert_eq!(occupied_of(&rig), vec![(0, (0, 1)), (1, (0, 2))]);

        // A later miss with the pool full: the least recently used live slot is evicted into
        // quarantine (the policy forgets its unit; the bank keeps the occupant).
        let retire = rig.retire_peer((0, 0), &[use_(0, 12)], &[], |_| 0);
        assert_eq!(retire.unpublish.len(), 1);
        let evicted = retire.unpublish[0];
        let quarantined = (0..3)
            .find(|&slot| rig.slot_phase(slot) == SlotPhase::Quarantined)
            .expect("one quarantined slot");
        assert_eq!(rig.slot_unit(quarantined), None);
        let occupied = occupied_of(&rig);
        assert!(occupied.contains(&(quarantined, evicted)), "{occupied:?}");
        assert_eq!(occupied.len(), 2, "{occupied:?}");

        // Issued: the filling slot is occupied too (its copy may have run).
        rig.issue_unissued();
        let occupied = occupied_of(&rig);
        assert_eq!(occupied.len(), 3, "{occupied:?}");
        assert!(occupied.contains(&(2, (0, 9))), "{occupied:?}");

        // An abandoned ticket never becomes occupied.
        let mut frozen = Rig::new(
            EpUnitTailConfig {
                tail_units: 2,
                fill_per_step: 4,
                boundary_every: 1,
            },
            false,
            HashMap::new(),
        );
        frozen.seed(&[(0, 1)]);
        frozen.defer_issue(true);
        assert_eq!(frozen.retire((0, 0), &[use_(0, 9)], &[]).fills.len(), 1);
        frozen.abandon_unissued();
        assert_eq!(occupied_of(&frozen), vec![(0, (0, 1))]);
        assert_eq!(frozen.slot_phase(1), SlotPhase::Filling);

        // Released: nothing to restore.
        let mut inner = bare(bare_policy(), true);
        assert!(inner.occupied().is_empty());
        inner.released = true;
        assert!(inner.occupied().is_empty());
    }

    fn use_(layer: u16, expert: u16) -> WindowUse {
        WindowUse {
            unit: (layer, expert),
            reads: 1,
        }
    }

    #[test]
    fn a_suspended_tail_skips_the_window_end_and_drops_the_window_log() {
        let mut inner = bare(bare_policy(), true);
        inner.calls.push((3, 8, 0));
        inner.log_used = 32;
        assert!(inner.skip_suspended_window());
        assert!(inner.calls.is_empty() && inner.log_used == 0);
        // The cadence clock did not move: no window counted.
        assert_eq!(inner.cadence.pending(), 0);
        assert_eq!(inner.policy.stats().retirements, 0);
        // A live tail leaves its window alone: the real retirement runs.
        inner.suspended = false;
        inner.calls.push((3, 8, 0));
        inner.log_used = 32;
        assert!(!inner.skip_suspended_window());
        assert_eq!((inner.calls.len(), inner.log_used), (1, 32));
        // `released` or not, an Inner without layers has no live slot, suspended or not.
        inner.suspended = true;
        assert!(inner.live_slots(0).is_empty());
    }

    #[test]
    fn the_digest_and_the_mismatch_offset_are_fnv_1a_and_the_first_differing_byte() {
        // FNV-1a 64 reference vectors.
        assert_eq!(fnv1a64(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(FNV_OFFSET, b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(
            fnv1a64(fnv1a64(FNV_OFFSET, b"foo"), b"bar"),
            fnv1a64(FNV_OFFSET, b"foobar")
        );
        assert_eq!(first_difference(b"abcd", b"abcd"), None);
        assert_eq!(first_difference(b"abcd", b"abxd"), Some(2));
        assert_eq!(first_difference(b"abcd", b"abc"), Some(3));
        assert_eq!(first_difference(b"", b"a"), Some(0));
    }
}
