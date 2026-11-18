// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The policy core: pools of [`SlotBank`] slots, the per-window decision rules ported from the
//! Qwen4 D2 `TailPolicy`, the generation retirement protocol and the request-by-request
//! `reserve` / `lookup` / `rollback` surface. See the [module docs](super) for the protocol.
//!
//! # Decision rules (kept from `TailPolicy::retire_inner`, non-peer, per-window mode)
//!
//! * A group of `windows` windows is judged against the slots `Resident` when it starts: a used
//!   key whose slot is `Resident` is a tail hit (its slot is stamped), anything else is a host
//!   read and a candidate. Completed fills are staged AFTER the judging, so a fill that completed
//!   inside the group still counts its uses as host reads.
//! * Candidates are the missed keys that are not already bound to a slot (a pending fill is never
//!   admitted twice), ordered by reads (descending, compared as `f32` like the beta policy), then
//!   ranking (ascending, smaller = hotter), then key.
//! * The fill budget of a pool is `F x clamp(windows, 1, cap)` minus the fills in flight; at most
//!   that many candidates are taken.
//! * A candidate takes the lowest `Free` slot; with none left it takes a victim: the `Resident`
//!   slots stamped before this group, least recently used first, the colder ranking first on a
//!   tie, then the slot index. A key any window of the group used is stamped by the group, so it
//!   is never a victim. A candidate without a free slot or a victim is `skipped_full`.
//!
//! # Where this core differs from `TailPolicy`
//!
//! * A victim's slot cannot be refilled until a generation that named it retired, so the core
//!   splits the beta policy's single step: [`ExpertCachePolicy::plan_group`] withdraws the victim
//!   (`Resident -> Evicting`) and records the candidate as its deferred fill;
//!   [`ExpertCachePolicy::finish_plan`] reclaims the slot against the retirement token and starts
//!   the fill. The decision (which victim, which candidate, which slot, which fill generation) is
//!   the beta policy's; only fills into victim slots are emitted by `finish_plan` (after the
//!   fills into free slots) instead of interleaved in candidate order.
//! * A retirement happens only when something is pending (A2.1): a staged completion or a
//!   withdrawn victim. A group with nothing pending keeps the live generation and its lease.
//! * `windows == 0` is a publication-only plan (no judging, no admission, no counters but the
//!   staged completions): the warm start's seeds and the `reserve` path publish through it.
//! * Peer quarantine, boundary heat mode and pins are not part of the core.

use std::cmp::Reverse;
use std::collections::VecDeque;

use railgun::sealed_table::generation::{
    acquire_window_owned, finish_window, Expert, FenceBackend, FenceFailure, FillComplete,
    FillTicket, FillsNotDone, GenerationId, GenerationRetired, LentBlock, OwnedTableGeneration,
    PendingFills, PendingOrQuarantined, PendingWindow, QuarantinedWindow, ResidencyError, SlotBank,
    SlotCell, SlotPhaseView, SlotSpan, Staging, TableBacking, WindowReadLease,
};
use railgun::window::Quiescence;
use railgun::Binding;

use super::index::{KeyIndex, Loc};
use super::window::{
    CancelOutcome, ConsumerId, RetireCadence, WindowBook, DEFAULT_GROUP_MAX_WINDOWS,
};
use super::{
    CacheError, CacheStats, ExpertKey, PoolId, SlotPhase, WindowConsumerPhase, WindowUse,
    DEFAULT_FILL_GROUP_CAP,
};

/// Capacities and cadence of the whole cache. Every buffer is sized from these at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyConfig {
    /// Routed ids that close a window group (D2's `retire_ids`).
    pub retire_ids: u64,
    /// Most windows in one group (D2's `RETIRE_MAX_WINDOWS`).
    pub max_windows: u32,
    /// The group fill budget is `fills_per_window x min(windows, fill_group_cap)`.
    pub fill_group_cap: usize,
    /// Distinct keys the window use log holds (also the cadence's log bound, in ids).
    pub use_log: usize,
    /// Completion tokens buffered between `deliver_completion` and `plan_group`.
    pub completions: usize,
    /// Withdrawn victims awaiting a retirement.
    pub parked: usize,
    /// Consumers one window registers.
    pub consumers: usize,
    /// Queues consumers register on.
    pub queues: usize,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            retire_ids: 4096,
            max_windows: DEFAULT_GROUP_MAX_WINDOWS,
            fill_group_cap: DEFAULT_FILL_GROUP_CAP,
            use_log: 16384,
            completions: 256,
            parked: 256,
            consumers: 16,
            queues: 2,
        }
    }
}

/// The policy numbers of one pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    /// Slots of the pool (must equal `PoolInit::spans.len()`).
    pub slots: usize,
    /// `F`: fills one window of a group may start.
    pub fills_per_window: usize,
}

/// The physical resources of one pool, handed to the core at construction.
#[derive(Debug)]
pub struct PoolInit {
    /// The pool's allocation (an `Expert` loan).
    pub block: LentBlock<Expert>,
    /// `spans[i]`: the write extents of slot `i`, block relative (1..=4 spans).
    pub spans: Vec<Vec<SlotSpan>>,
    /// The pointer-table allocations every generation of the pool is published through.
    pub tables: Vec<Binding>,
    /// Warm start: `(key, rank)` fills slots `0..` in order (repeats and keys beyond `slots` are
    /// ignored, like the beta policy's `seed`).
    /// The fills are in [`ExpertCachePolicy::take_fills`]; once their completions are delivered a
    /// publication-only plan (`plan_group(0, ..)`) publishes them.
    pub seed: Vec<(ExpertKey, u32)>,
}

/// A copy the device part must start: `key`'s canonical bytes into `slot`. The `ticket` is the
/// bank's non-Clone fill capability; [`ExpertCachePolicy::issue_fills`] consumes it and the copy
/// writes only `ticket.write_spans()`.
pub struct FillOp {
    pub key: ExpertKey,
    pub pool: PoolId,
    pub slot: u32,
    pub fill_generation: u64,
    pub ticket: FillTicket,
}

impl FillOp {
    /// `(key, pool, slot, fill_generation)`: what the policy decided (the ticket is a
    /// capability, not a decision).
    pub fn decision(&self) -> (ExpertKey, PoolId, u32, u64) {
        (self.key, self.pool, self.slot, self.fill_generation)
    }
}

impl PartialEq for FillOp {
    fn eq(&self, other: &Self) -> bool {
        self.decision() == other.decision()
    }
}

impl Eq for FillOp {}

impl std::fmt::Debug for FillOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FillOp")
            .field("key", &self.key)
            .field("pool", &self.pool)
            .field("slot", &self.slot)
            .field("fill_generation", &self.fill_generation)
            .finish_non_exhaustive()
    }
}

/// A key whose slot became `Resident` (a member of the next image): its table entry now names
/// the slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Published {
    pub key: ExpertKey,
    pub pool: PoolId,
    pub slot: u32,
    pub fill_generation: u64,
}

/// A key withdrawn from the next image: its table entry returns to canonical. The slot is
/// reclaimed (and refilled) by `finish_plan` once the old generation retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Withdrawn {
    pub key: ExpertKey,
    pub pool: PoolId,
    pub slot: u32,
}

/// What one `plan_group` decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanSummary {
    /// Windows the group covered (0 = publication only).
    pub windows: u32,
    /// Distinct keys served by a `Resident` slot.
    pub hits: u32,
    /// Distinct keys read from the canonical backing.
    pub misses: u32,
    /// Completions staged (`Filling -> Resident`).
    pub published: u32,
    /// Completions ignored as stale.
    pub stale: u32,
    /// Completions of cancelled fills (their slot was withdrawn, nothing published).
    pub cancelled: u32,
    /// Victims withdrawn.
    pub withdrawn: u32,
    /// Fills started now (into free slots).
    pub fills: u32,
    /// Fills deferred to `finish_plan` (into victim slots).
    pub deferred: u32,
    /// Candidates dropped for lack of a victim.
    pub skipped_full: u32,
    /// Some pool must retire its generation before the next window.
    pub needs_retirement: bool,
}

/// The outcome of a `reserve`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reservation {
    /// The key is published: read `slot`.
    Hit {
        key: ExpertKey,
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
    /// A fill into a free slot was started; its ticket is in `take_fills`. Invisible until the
    /// completion is delivered and published.
    Fill {
        key: ExpertKey,
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
    /// A victim was withdrawn (or a withdrawn slot adopted) for the key: its fill starts in
    /// `finish_plan`, with `fill_generation`.
    Deferred {
        key: ExpertKey,
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
    /// The key is already pending: this caller does not own the fill and it is never a hit.
    Pending {
        key: ExpertKey,
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
}

impl Reservation {
    /// Whether this caller owns a fill (`Fill` or `Deferred`).
    pub fn needs_fill(&self) -> bool {
        matches!(self, Self::Fill { .. } | Self::Deferred { .. })
    }
}

/// What the core knows about a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Absent,
    /// Bound to a slot that is filling or awaiting its deferred fill: invisible to readers.
    Pending {
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
    /// A `Resident` slot (a member of the next image).
    Published {
        pool: PoolId,
        slot: u32,
        fill_generation: u64,
    },
}

/// One slot's bookkeeping beside the bank's typestate cell.
struct SlotRec {
    /// `None` only while a transition holds the cell by value.
    cell: Option<SlotCell>,
    /// The key of a `Filling` (not cancelled) or `Resident` slot.
    key: Option<ExpertKey>,
    /// Stamp of the last use (window group or `reserve`).
    last_used: u64,
    /// Ranking position of the held key (smaller = hotter); the victim tie-break.
    rank: u32,
    /// The fill an `Evicting` slot starts after its reclaim.
    target: Option<(ExpertKey, u32)>,
    /// The key was rolled back while the copy was in flight: the completion withdraws the slot.
    cancelled: bool,
    /// A warm-start fill (published without counting as a fill).
    seeded: bool,
}

/// The generation of one pool: sealed under its lease until a retirement needs it.
enum GenState {
    /// Sealed, under its one lease.
    Live(WindowReadLease<'static>),
    /// The last-read fence is recorded and not yet waited.
    Fencing(PendingWindow<'static>),
    /// Sealed with no lease outstanding (a refused lease, or a refused retirement).
    Idle(OwnedTableGeneration),
    /// Retired: the backing is writable and `retired` holds the token.
    Retired,
    /// An unknown-quiescence fence failure: it can never retire.
    Quarantined(QuarantinedWindow),
    /// A transition holds the state by value.
    Transit,
}

impl GenState {
    fn name(&self) -> &'static str {
        match self {
            Self::Live(_) => "live under its lease",
            Self::Fencing(_) => "waiting for its last-read fence",
            Self::Idle(_) => "sealed without a lease",
            Self::Retired => "retired",
            Self::Quarantined(_) => "quarantined",
            Self::Transit => "in transit",
        }
    }
}

enum Staged {
    Published {
        key: ExpertKey,
        fill_generation: u64,
    },
    Cancelled,
    Stale,
}

/// One pool: a bank, its staged membership, its slots and its generation.
struct Pool {
    id: PoolId,
    bank: SlotBank,
    /// The membership of the NEXT generation: the `Resident` slots.
    staging: Staging,
    slots: Vec<SlotRec>,
    fills_per_window: usize,
    gen: GenState,
    backing: Option<TableBacking>,
    retired: Option<GenerationRetired>,
    next_epoch: u64,
    /// A publication or withdrawal is pending: the generation must retire, the tables be
    /// rewritten and a new generation sealed.
    dirty: bool,
}

impl Pool {
    fn bank_err(&self, error: ResidencyError) -> CacheError {
        CacheError::Bank {
            pool: self.id,
            error,
        }
    }

    fn refused(&self, why: String) -> CacheError {
        self.bank_err(ResidencyError::Refused(why))
    }

    fn phase(&self, slot: usize) -> SlotPhaseView {
        self.slots[slot]
            .cell
            .as_ref()
            .expect("slot cell present")
            .phase()
    }

    fn fill_generation(&self, slot: usize) -> u64 {
        match self.slots[slot].cell.as_ref().expect("slot cell present") {
            SlotCell::Free(s) => s.fill_generation(),
            SlotCell::Filling(s) => s.fill_generation(),
            SlotCell::Resident(s) => s.fill_generation(),
            SlotCell::Evicting(s) => s.fill_generation(),
        }
    }

    fn count(&self, phase: SlotPhaseView) -> usize {
        (0..self.slots.len())
            .filter(|&slot| self.phase(slot) == phase)
            .count()
    }

    fn take_cell(&mut self, slot: usize) -> SlotCell {
        self.slots[slot].cell.take().expect("slot cell present")
    }

    fn put_cell(&mut self, slot: usize, cell: SlotCell) {
        self.slots[slot].cell = Some(cell);
    }

    /// `Free -> Filling` (bumps the fill generation). On refusal the cell stays as it was.
    fn begin_fill(&mut self, slot: usize, key: ExpertKey) -> Result<FillTicket, CacheError> {
        match self.take_cell(slot) {
            SlotCell::Free(free) => match self.bank.start_fill(free, key.occupant()) {
                Ok((filling, ticket)) => {
                    self.put_cell(slot, SlotCell::Filling(filling));
                    Ok(ticket)
                }
                Err((free, error)) => {
                    self.put_cell(slot, SlotCell::Free(free));
                    Err(self.bank_err(error))
                }
            },
            other => {
                self.put_cell(slot, other);
                Err(self.refused(format!("slot {slot} is not free")))
            }
        }
    }

    /// `Filling -> Resident` by the completion token, then staged for the next generation. A
    /// token that names no `Filling` slot at its fill generation (or is another bank's) is
    /// `Stale`; a cancelled fill's slot is withdrawn at once (it never joins the image).
    fn stage(&mut self, done: FillComplete) -> Result<Staged, CacheError> {
        let slot = done.slot() as usize;
        if slot >= self.slots.len() {
            return Ok(Staged::Stale);
        }
        match self.take_cell(slot) {
            SlotCell::Filling(filling) => match self.bank.finish_fill(filling, done) {
                Ok(resident) => {
                    if self.slots[slot].cancelled {
                        return match self.bank.unpublish(&mut self.staging, resident) {
                            Ok(evicting) => {
                                self.put_cell(slot, SlotCell::Evicting(evicting));
                                let rec = &mut self.slots[slot];
                                rec.cancelled = false;
                                rec.seeded = false;
                                Ok(Staged::Cancelled)
                            }
                            Err((resident, error)) => {
                                self.put_cell(slot, SlotCell::Resident(resident));
                                Err(self.bank_err(error))
                            }
                        };
                    }
                    let fill_generation = resident.fill_generation();
                    let staged = self.bank.publish(&mut self.staging, &resident);
                    self.put_cell(slot, SlotCell::Resident(resident));
                    if let Err(error) = staged {
                        return Err(self.bank_err(error));
                    }
                    match self.slots[slot].key {
                        Some(key) => Ok(Staged::Published {
                            key,
                            fill_generation,
                        }),
                        None => Err(self.refused(format!("slot {slot} completed without a key"))),
                    }
                }
                Err((filling, _done, _error)) => {
                    self.put_cell(slot, SlotCell::Filling(filling));
                    Ok(Staged::Stale)
                }
            },
            other => {
                self.put_cell(slot, other);
                Ok(Staged::Stale)
            }
        }
    }

    /// `Resident -> Evicting`, out of the staged membership.
    fn evict(&mut self, slot: usize) -> Result<(), CacheError> {
        match self.take_cell(slot) {
            SlotCell::Resident(resident) => {
                match self.bank.unpublish(&mut self.staging, resident) {
                    Ok(evicting) => {
                        self.put_cell(slot, SlotCell::Evicting(evicting));
                        Ok(())
                    }
                    Err((resident, error)) => {
                        self.put_cell(slot, SlotCell::Resident(resident));
                        Err(self.bank_err(error))
                    }
                }
            }
            other => {
                self.put_cell(slot, other);
                Err(self.refused(format!("slot {slot} is not resident")))
            }
        }
    }

    /// `Evicting -> Free` against the last retirement's token.
    fn reclaim(&mut self, slot: usize) -> Result<(), CacheError> {
        match self.take_cell(slot) {
            SlotCell::Evicting(evicting) => {
                let Some(token) = self.retired.as_ref() else {
                    self.put_cell(slot, SlotCell::Evicting(evicting));
                    return Err(CacheError::RetirementRequired { pool: self.id });
                };
                match self.bank.reclaim(evicting, token) {
                    Ok(free) => {
                        self.put_cell(slot, SlotCell::Free(free));
                        Ok(())
                    }
                    Err((evicting, error)) => {
                        self.put_cell(slot, SlotCell::Evicting(evicting));
                        Err(self.bank_err(error))
                    }
                }
            }
            other => {
                self.put_cell(slot, other);
                Err(self.refused(format!("slot {slot} is not evicting")))
            }
        }
    }

    /// Fence the live generation's last reader and retire it. `Ok(true)` = newly retired.
    fn retire(&mut self, fence: &mut dyn FenceBackend) -> Result<bool, CacheError> {
        let pool = self.id;
        let pending = match std::mem::replace(&mut self.gen, GenState::Transit) {
            GenState::Live(lease) => match lease.enqueued(fence) {
                Ok(pending) => pending,
                Err((lease, failure)) => {
                    self.gen = GenState::Live(lease);
                    return Err(CacheError::Fence { pool, failure });
                }
            },
            GenState::Fencing(pending) => pending,
            GenState::Idle(generation) => return self.retire_sealed(generation),
            GenState::Retired => {
                self.gen = GenState::Retired;
                return Ok(false);
            }
            GenState::Quarantined(q) => {
                self.gen = GenState::Quarantined(q);
                return Err(CacheError::Quarantined { pool });
            }
            GenState::Transit => {
                return Err(CacheError::GenerationState {
                    pool,
                    state: GenState::Transit.name(),
                });
            }
        };
        match finish_window(pending, fence) {
            Ok(fenced) => match fenced.into_generation() {
                Some(generation) => self.retire_sealed(generation),
                None => Err(CacheError::GenerationState {
                    pool,
                    state: "fenced through a borrowed lease",
                }),
            },
            Err(PendingOrQuarantined::Pending(pending)) => {
                self.gen = GenState::Fencing(pending);
                Err(CacheError::WindowFencePending { pool })
            }
            Err(PendingOrQuarantined::Quarantined(q)) => {
                let failure = q.failure().clone();
                self.gen = GenState::Quarantined(q);
                Err(CacheError::Fence { pool, failure })
            }
        }
    }

    fn retire_sealed(&mut self, generation: OwnedTableGeneration) -> Result<bool, CacheError> {
        match self.bank.retire(generation, None) {
            Ok((backing, token)) => {
                self.backing = Some(backing);
                self.retired = Some(token);
                self.gen = GenState::Retired;
                Ok(true)
            }
            Err((generation, error)) => {
                self.gen = GenState::Idle(generation);
                Err(self.bank_err(error))
            }
        }
    }

    /// Seal the staged membership into the next generation under a fresh lease.
    fn seal(&mut self) -> Result<GenerationId, CacheError> {
        let pool = self.id;
        if !matches!(self.gen, GenState::Retired) {
            return Err(CacheError::GenerationState {
                pool,
                state: self.gen.name(),
            });
        }
        let Some(backing) = self.backing.take() else {
            return Err(CacheError::GenerationState {
                pool,
                state: "without a table backing",
            });
        };
        let id = GenerationId {
            epoch: self.next_epoch,
            sub: 0,
        };
        match self.bank.seal(&self.staging, backing, id) {
            Ok(generation) => {
                self.next_epoch += 1;
                self.retired = None;
                match acquire_window_owned(generation) {
                    Ok(lease) => {
                        self.gen = GenState::Live(lease);
                        Ok(id)
                    }
                    Err((generation, _error)) => {
                        self.gen = GenState::Idle(generation);
                        Err(CacheError::Quarantined { pool })
                    }
                }
            }
            Err((backing, error)) => {
                self.backing = Some(backing);
                Err(self.bank_err(error))
            }
        }
    }
}

/// The policy core. See the [module docs](super) for the protocol.
pub struct ExpertCachePolicy {
    config: PolicyConfig,
    pools: Vec<Pool>,
    /// Key -> `(pool, slot)` for every slot that is `Filling`, `Resident` or a deferred target.
    index: KeyIndex,
    book: WindowBook,
    cadence: RetireCadence,
    /// The use log of the group in progress (distinct keys) and its key -> position index.
    log: Vec<WindowUse>,
    log_index: KeyIndex,
    /// Scratch: the group's missed keys, then its candidates.
    cands: Vec<WindowUse>,
    /// Scratch: victim choices of one pool.
    victims: Vec<(u64, Reverse<u32>, u32)>,
    completions: VecDeque<(u16, FillComplete)>,
    /// Withdrawn slots awaiting a retirement: `(pool, slot)`.
    parked: Vec<(u16, u32)>,
    published: Vec<Published>,
    withdrawn: Vec<Withdrawn>,
    fills: Vec<FillOp>,
    /// Monotone stamp source (plan groups, windows, request-path touches).
    clock: u64,
    /// The stamp of the open window (meaningful while a window is open or closing).
    window_stamp: u64,
    stats: CacheStats,
    frozen: bool,
}

impl std::fmt::Debug for ExpertCachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertCachePolicy")
            .field("pools", &self.pools.len())
            .field("keys", &self.index.len())
            .field("clock", &self.clock)
            .field("frozen", &self.frozen)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl ExpertCachePolicy {
    /// Build the pools. Each pool's bank seals an empty first generation (epoch 0) under its
    /// lease, so every later retirement has a generation to retire and a token to reclaim
    /// against. A construction error drops the blocks of the pools already built (a registry
    /// capacity leak, never an early release).
    pub fn new(
        config: PolicyConfig,
        pools: Vec<(PoolConfig, PoolInit)>,
    ) -> Result<Self, CacheError> {
        if pools.is_empty() {
            return Err(CacheError::NoPools);
        }
        if pools.len() > u16::MAX as usize {
            return Err(CacheError::PoolShape {
                pool: PoolId(u16::MAX),
                why: "more pools than a pool id names",
            });
        }
        let mut total = 0usize;
        let mut widest = 0usize;
        for (i, (cfg, init)) in pools.iter().enumerate() {
            let pool = PoolId(i as u16);
            let why = if cfg.slots == 0 {
                Some("a pool needs at least one slot")
            } else if cfg.slots > u32::MAX as usize {
                Some("more slots than a slot index names")
            } else if init.spans.len() != cfg.slots {
                Some("slot extents for another slot count")
            } else {
                None
            };
            if let Some(why) = why {
                return Err(CacheError::PoolShape { pool, why });
            }
            total += cfg.slots;
            widest = widest.max(cfg.slots);
        }
        let mut index = KeyIndex::new(total);
        let mut fills = Vec::with_capacity(total);
        let mut built = Vec::with_capacity(pools.len());
        for (i, (cfg, init)) in pools.into_iter().enumerate() {
            let id = PoolId(i as u16);
            let bank_err = move |error: ResidencyError| CacheError::Bank { pool: id, error };
            let (mut bank, cells) = SlotBank::new(init.block, init.spans, false)
                .map_err(|(_block, error)| bank_err(error))?;
            let backing = bank.register_table_backing(init.tables).map_err(bank_err)?;
            let staging = bank.staging();
            let first = GenerationId { epoch: 0, sub: 0 };
            let generation = bank
                .seal(&staging, backing, first)
                .map_err(|(_backing, error)| bank_err(error))?;
            let lease =
                acquire_window_owned(generation).map_err(|(_generation, error)| bank_err(error))?;
            let slots: Vec<SlotRec> = cells
                .into_iter()
                .map(|free| SlotRec {
                    cell: Some(SlotCell::Free(free)),
                    key: None,
                    last_used: 0,
                    rank: 0,
                    target: None,
                    cancelled: false,
                    seeded: false,
                })
                .collect();
            let mut pool = Pool {
                id,
                bank,
                staging,
                slots,
                fills_per_window: cfg.fills_per_window,
                gen: GenState::Live(lease),
                backing: None,
                retired: None,
                next_epoch: 1,
                dirty: false,
            };
            let mut next = 0usize;
            for &(key, rank) in &init.seed {
                if next >= cfg.slots {
                    break;
                }
                if let Some(bound) = index.get(&key) {
                    if bound.pool != id.0 {
                        return Err(CacheError::PoolMismatch {
                            key,
                            bound: PoolId(bound.pool),
                            requested: id,
                        });
                    }
                    continue;
                }
                let ticket = pool.begin_fill(next, key)?;
                let rec = &mut pool.slots[next];
                rec.key = Some(key);
                rec.rank = rank;
                rec.seeded = true;
                index
                    .insert(
                        key,
                        Loc {
                            pool: id.0,
                            slot: next as u32,
                        },
                    )
                    .map_err(|_| CacheError::IndexFull { capacity: total })?;
                fills.push(FillOp {
                    key,
                    pool: id,
                    slot: next as u32,
                    fill_generation: ticket.fill_generation(),
                    ticket,
                });
                next += 1;
            }
            built.push(pool);
        }
        let cadence = RetireCadence::with_max_windows(
            config.retire_ids,
            config.use_log as u64,
            config.max_windows,
        );
        Ok(Self {
            config,
            pools: built,
            index,
            book: WindowBook::new(config.consumers, config.queues),
            cadence,
            log: Vec::with_capacity(config.use_log),
            log_index: KeyIndex::new(config.use_log),
            cands: Vec::with_capacity(config.use_log),
            victims: Vec::with_capacity(widest),
            completions: VecDeque::with_capacity(config.completions),
            parked: Vec::with_capacity(config.parked),
            published: Vec::with_capacity(total),
            withdrawn: Vec::with_capacity(total),
            fills,
            clock: 0,
            window_stamp: 0,
            stats: CacheStats::default(),
            frozen: false,
        })
    }

    // ------------------------------------------------------------------ accessors

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    pub fn pools(&self) -> usize {
        self.pools.len()
    }

    /// Slots of a pool.
    pub fn slots(&self, pool: PoolId) -> Option<usize> {
        self.pools.get(pool.0 as usize).map(|p| p.slots.len())
    }

    /// The pool's bank (introspection: reference counts, phases, live generations).
    pub fn bank(&self, pool: PoolId) -> Option<&SlotBank> {
        self.pools.get(pool.0 as usize).map(|p| &p.bank)
    }

    /// The window book of the cache.
    pub fn window(&self) -> &WindowBook {
        &self.book
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Keys bound to a slot (filling, resident or a deferred target).
    pub fn keys(&self) -> usize {
        self.index.len()
    }

    /// The public phase of a slot.
    pub fn slot_phase(&self, pool: PoolId, slot: u32) -> Option<SlotPhase> {
        let p = self.pools.get(pool.0 as usize)?;
        if slot as usize >= p.slots.len() {
            return None;
        }
        Some(match p.phase(slot as usize) {
            SlotPhaseView::Free => SlotPhase::Free,
            SlotPhaseView::Filling => SlotPhase::Filling,
            SlotPhaseView::Resident => SlotPhase::Live,
            SlotPhaseView::Evicting => SlotPhase::Withdrawn,
        })
    }

    /// The key a slot holds while `Filling` (not cancelled) or `Live`.
    pub fn slot_key(&self, pool: PoolId, slot: u32) -> Option<ExpertKey> {
        self.pools
            .get(pool.0 as usize)?
            .slots
            .get(slot as usize)?
            .key
    }

    /// The fill generation of a slot (bumped by every fill that reuses it).
    pub fn slot_generation(&self, pool: PoolId, slot: u32) -> Option<u64> {
        let p = self.pools.get(pool.0 as usize)?;
        ((slot as usize) < p.slots.len()).then(|| p.fill_generation(slot as usize))
    }

    /// Slots of a pool in a phase.
    pub fn count(&self, pool: PoolId, phase: SlotPhase) -> usize {
        let Some(p) = self.pools.get(pool.0 as usize) else {
            return 0;
        };
        let view = match phase {
            SlotPhase::Free => SlotPhaseView::Free,
            SlotPhase::Filling => SlotPhaseView::Filling,
            SlotPhase::Live => SlotPhaseView::Resident,
            SlotPhase::Withdrawn => SlotPhaseView::Evicting,
        };
        p.count(view)
    }

    /// What the core knows about `key`.
    pub fn lookup(&self, key: ExpertKey) -> Lookup {
        let Some(loc) = self.index.get(&key) else {
            return Lookup::Absent;
        };
        let p = &self.pools[loc.pool as usize];
        let slot = loc.slot as usize;
        let pool = PoolId(loc.pool);
        match p.phase(slot) {
            SlotPhaseView::Resident => Lookup::Published {
                pool,
                slot: loc.slot,
                fill_generation: p.fill_generation(slot),
            },
            SlotPhaseView::Evicting => Lookup::Pending {
                pool,
                slot: loc.slot,
                fill_generation: p.fill_generation(slot) + 1,
            },
            _ => Lookup::Pending {
                pool,
                slot: loc.slot,
                fill_generation: p.fill_generation(slot),
            },
        }
    }

    /// Keys whose table entry now names their slot, until the pool seals.
    pub fn published(&self) -> &[Published] {
        &self.published
    }

    /// Keys whose table entry returns to canonical, until the pool seals.
    pub fn withdrawn(&self) -> &[Withdrawn] {
        &self.withdrawn
    }

    /// Some pool has a publication or withdrawal pending.
    pub fn needs_retirement(&self) -> bool {
        self.pools.iter().any(|p| p.dirty)
    }

    /// `pool` has a publication or withdrawal pending: it must `retire_generation`, have its
    /// tables rewritten from `published()` / `withdrawn()`, `finish_plan` and `seal`.
    pub fn pool_needs_retirement(&self, pool: PoolId) -> bool {
        self.pools.get(pool.0 as usize).is_some_and(|p| p.dirty)
    }

    /// The generation of a pool in a word (diagnostics).
    pub fn generation_state(&self, pool: PoolId) -> Option<&'static str> {
        self.pools.get(pool.0 as usize).map(|p| p.gen.name())
    }

    /// The warm start of the pool completed: no seeded slot is still `Filling` (a seed fill
    /// whose completion was not delivered and staged is `SeedIncomplete`).
    pub fn check_seeded(&self, pool: PoolId) -> Result<(), CacheError> {
        let Some(p) = self.pools.get(pool.0 as usize) else {
            return Err(CacheError::UnknownPool(pool));
        };
        let missing = (0..p.slots.len())
            .any(|slot| p.slots[slot].seeded && p.phase(slot) == SlotPhaseView::Filling);
        if missing {
            Err(CacheError::SeedIncomplete { pool })
        } else {
            Ok(())
        }
    }

    /// Stop admitting: plans still stage completions, `reserve` refuses. Drops the group in
    /// progress.
    pub fn freeze(&mut self) {
        self.frozen = true;
        self.cadence.clear();
        self.log.clear();
        self.log_index.clear();
    }

    /// Buffer capacities, for the no-growth tests.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> [usize; 10] {
        [
            self.log.capacity(),
            self.cands.capacity(),
            self.victims.capacity(),
            self.parked.capacity(),
            self.published.capacity(),
            self.withdrawn.capacity(),
            self.fills.capacity(),
            self.completions.capacity(),
            self.index.buckets(),
            self.log_index.buckets(),
        ]
    }

    // -------------------------------------------------------------------- windows

    fn pool_idx(&self, pool: PoolId) -> Result<usize, CacheError> {
        if (pool.0 as usize) < self.pools.len() {
            Ok(pool.0 as usize)
        } else {
            Err(CacheError::UnknownPool(pool))
        }
    }

    fn window_active(&self) -> bool {
        self.book.is_open() || self.book.is_closing()
    }

    /// The stamp of a use outside a plan: the open window's, or a fresh one.
    fn touch_stamp(&mut self) -> u64 {
        if self.window_active() {
            self.window_stamp
        } else {
            self.clock += 1;
            self.clock
        }
    }

    /// Slots stamped at or after this are protected from `reserve` victims.
    fn guard(&self) -> u64 {
        if self.window_active() {
            self.window_stamp
        } else {
            self.clock + 1
        }
    }

    /// Open a window (the image the pools' generations name). Refused while a pool still waits
    /// for its retirement, while a window is open, and once the book is poisoned.
    pub fn begin_window(&mut self) -> Result<u64, CacheError> {
        if self.book.is_poisoned() {
            return Err(CacheError::WindowPoisoned);
        }
        if self.needs_retirement() {
            return Err(CacheError::PlanPending);
        }
        let epoch = self.book.begin()?;
        self.clock += 1;
        self.window_stamp = self.clock;
        Ok(epoch)
    }

    /// Register one consumer of the open window on `queue` BEFORE it is enqueued.
    pub fn register_consumer(
        &mut self,
        queue: usize,
        phase: WindowConsumerPhase,
        lane: u32,
    ) -> Result<ConsumerId, CacheError> {
        self.book.register(queue, phase, lane)
    }

    /// The consumer's queue proved it retired.
    pub fn join_consumer(&mut self, id: ConsumerId) -> Result<(), CacheError> {
        self.book.join(id)
    }

    /// Cancel the window (timeout, error, unload). With every queue proven retired the book is
    /// idle again; with unknown completion it is poisoned for good: no later window opens and no
    /// generation retires.
    pub fn cancel_window(&mut self, outcome: CancelOutcome) -> Result<(), CacheError> {
        self.book.cancel(outcome)
    }

    /// Log one distinct key a window read (reads summed). `rank`: ranking position of the key
    /// (smaller = hotter).
    pub fn record_use(&mut self, used: WindowUse) -> Result<(), CacheError> {
        if !self.window_active() {
            return Err(if self.book.is_poisoned() {
                CacheError::WindowPoisoned
            } else {
                CacheError::WindowNotOpen
            });
        }
        self.pool_idx(used.pool)?;
        if let Some(loc) = self.index.get(&used.key) {
            if loc.pool != used.pool.0 {
                return Err(CacheError::PoolMismatch {
                    key: used.key,
                    bound: PoolId(loc.pool),
                    requested: used.pool,
                });
            }
        }
        if let Some(at) = self.log_index.get(&used.key) {
            let logged = &mut self.log[at.slot as usize];
            if logged.pool != used.pool {
                return Err(CacheError::PoolMismatch {
                    key: used.key,
                    bound: logged.pool,
                    requested: used.pool,
                });
            }
            logged.reads = logged.reads.saturating_add(used.reads);
            return Ok(());
        }
        if self.log.len() >= self.config.use_log {
            return Err(CacheError::UseLogFull {
                capacity: self.config.use_log,
            });
        }
        let at = Loc {
            pool: used.pool.0,
            slot: self.log.len() as u32,
        };
        self.log_index
            .insert(used.key, at)
            .map_err(|_| CacheError::UseLogFull {
                capacity: self.config.use_log,
            })?;
        self.log.push(used);
        Ok(())
    }

    /// End the open window: close it to new consumers and finish it (every registered consumer
    /// must have joined; otherwise `ConsumersOutstanding` and the window stays closed, joinable).
    /// `routed`: routed rows x top-k of the window. `Some(n)`: the cadence closes a group of `n`
    /// windows: call [`Self::plan_group`] with it.
    pub fn end_window(&mut self, routed: u64) -> Result<Option<u32>, CacheError> {
        self.book.close()?;
        self.book.finish()?;
        Ok(self.cadence.window_end(routed))
    }

    // ----------------------------------------------------------------------- plan

    /// Judge one retirement group of `windows` windows and decide: stage the completed fills,
    /// withdraw the victims, start the fills into free slots (module docs). `core`:
    /// `(distinct, reads)` the static core served. Needs the book idle and no pool waiting for
    /// its retirement. `windows == 0`: publication only.
    pub fn plan_group(
        &mut self,
        windows: u32,
        core: (u64, u64),
    ) -> Result<PlanSummary, CacheError> {
        if self.book.is_poisoned() {
            return Err(CacheError::WindowPoisoned);
        }
        if !self.book.is_idle() {
            return Err(CacheError::WindowsOpen);
        }
        if self.needs_retirement() {
            return Err(CacheError::PlanPending);
        }
        let judged = windows > 0;
        self.clock += 1;
        let now = self.clock;
        let mut summary = PlanSummary {
            windows,
            ..PlanSummary::default()
        };

        // 1. Judge the group against the slots Resident at its start.
        if judged {
            self.stats.windows += windows as u64;
            self.stats.groups += 1;
            self.stats.core_distinct += core.0;
            self.stats.core_reads += core.1;
            self.cands.clear();
            for &used in &self.log {
                let hit = match self.index.get(&used.key) {
                    Some(loc) if loc.pool != used.pool.0 => {
                        return Err(CacheError::PoolMismatch {
                            key: used.key,
                            bound: PoolId(loc.pool),
                            requested: used.pool,
                        });
                    }
                    Some(loc) => {
                        let pool = &mut self.pools[loc.pool as usize];
                        let slot = loc.slot as usize;
                        let resident = pool.phase(slot) == SlotPhaseView::Resident;
                        if resident {
                            pool.slots[slot].last_used = now;
                        }
                        resident
                    }
                    None => false,
                };
                if hit {
                    self.stats.tail_distinct += 1;
                    self.stats.tail_reads += used.reads as u64;
                    summary.hits += 1;
                } else {
                    self.stats.host_distinct += 1;
                    self.stats.host_reads += used.reads as u64;
                    summary.misses += 1;
                    self.cands.push(used);
                }
            }
        }

        // 2. Stage the fills that completed (they serve the NEXT image).
        self.stage_completions(now, &mut summary)?;

        // 3. Admission.
        if judged && !self.frozen {
            self.admit(now, windows, &mut summary)?;
        }
        if judged {
            self.log.clear();
            self.log_index.clear();
        }
        summary.needs_retirement = self.needs_retirement();
        Ok(summary)
    }

    fn stage_completions(&mut self, now: u64, summary: &mut PlanSummary) -> Result<(), CacheError> {
        while let Some((p, slot)) = self
            .completions
            .front()
            .map(|(p, done)| (*p as usize, done.slot() as usize))
        {
            let cancelled = self.pools[p]
                .slots
                .get(slot)
                .is_some_and(|rec| rec.cancelled);
            if cancelled && self.parked.len() >= self.config.parked {
                return Err(CacheError::ParkedFull {
                    capacity: self.config.parked,
                });
            }
            let Some((_, done)) = self.completions.pop_front() else {
                break;
            };
            let pool = PoolId(p as u16);
            match self.pools[p].stage(done)? {
                Staged::Published {
                    key,
                    fill_generation,
                } => {
                    let rec = &mut self.pools[p].slots[slot];
                    if rec.seeded {
                        rec.seeded = false;
                    } else {
                        rec.last_used = now;
                        self.stats.fills_published += 1;
                    }
                    self.pools[p].dirty = true;
                    self.published.push(Published {
                        key,
                        pool,
                        slot: slot as u32,
                        fill_generation,
                    });
                    summary.published += 1;
                }
                Staged::Cancelled => {
                    self.parked.push((p as u16, slot as u32));
                    self.pools[p].dirty = true;
                    self.stats.cancelled_fills += 1;
                    summary.cancelled += 1;
                }
                Staged::Stale => {
                    self.stats.stale_completions += 1;
                    summary.stale += 1;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn admit(
        &mut self,
        now: u64,
        windows: u32,
        summary: &mut PlanSummary,
    ) -> Result<(), CacheError> {
        // Candidates: the missed keys not bound to a slot, best first.
        let index = &self.index;
        self.cands.retain(|used| index.get(&used.key).is_none());
        self.cands.sort_unstable_by(|a, b| {
            (b.reads as f32)
                .partial_cmp(&(a.reads as f32))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rank.cmp(&b.rank))
                .then_with(|| a.key.cmp(&b.key))
        });
        if self.cands.is_empty() {
            return Ok(());
        }
        let cap = self.config.fill_group_cap.max(1);
        for p in 0..self.pools.len() {
            let total = self.pools[p].fills_per_window * (windows as usize).clamp(1, cap);
            let in_flight = self.pools[p].count(SlotPhaseView::Filling);
            let budget = total.saturating_sub(in_flight);
            if budget == 0 || !self.cands.iter().any(|c| c.pool.0 as usize == p) {
                continue;
            }
            // Victims: Resident slots not used by this group, least recently used first (the
            // colder ranking on a tie). Only `budget` of them can be used.
            self.victims.clear();
            for slot in 0..self.pools[p].slots.len() {
                let rec = &self.pools[p].slots[slot];
                if self.pools[p].phase(slot) == SlotPhaseView::Resident && rec.last_used < now {
                    self.victims
                        .push((rec.last_used, Reverse(rec.rank), slot as u32));
                }
            }
            if self.victims.len() > budget {
                self.victims.select_nth_unstable(budget - 1);
                self.victims.truncate(budget);
            }
            self.victims.sort_unstable();
            let slots = self.pools[p].slots.len();
            let (mut free_at, mut next_victim, mut taken) = (0usize, 0usize, 0usize);
            for ci in 0..self.cands.len() {
                let cand = self.cands[ci];
                if cand.pool.0 as usize != p {
                    continue;
                }
                if taken == budget {
                    break;
                }
                taken += 1;
                let mut free = None;
                while free_at < slots {
                    let slot = free_at;
                    free_at += 1;
                    if self.pools[p].phase(slot) == SlotPhaseView::Free {
                        free = Some(slot);
                        break;
                    }
                }
                if let Some(slot) = free {
                    self.start_fill(p, slot, cand.key, cand.rank, now)?;
                    summary.fills += 1;
                    continue;
                }
                let Some(&(_, _, victim)) = self.victims.get(next_victim) else {
                    self.stats.skipped_full += 1;
                    summary.skipped_full += 1;
                    continue;
                };
                next_victim += 1;
                self.withdraw(p, victim as usize, Some((cand.key, cand.rank)), now)?;
                summary.withdrawn += 1;
                summary.deferred += 1;
            }
        }
        Ok(())
    }

    /// Start `key`'s fill into the `Free` slot (its ticket joins the fills to take).
    fn start_fill(
        &mut self,
        p: usize,
        slot: usize,
        key: ExpertKey,
        rank: u32,
        stamp: u64,
    ) -> Result<u64, CacheError> {
        let ticket = self.pools[p].begin_fill(slot, key)?;
        let fill_generation = ticket.fill_generation();
        let rec = &mut self.pools[p].slots[slot];
        rec.key = Some(key);
        rec.rank = rank;
        rec.last_used = stamp;
        rec.cancelled = false;
        rec.seeded = false;
        let loc = Loc {
            pool: p as u16,
            slot: slot as u32,
        };
        self.index
            .insert(key, loc)
            .map_err(|_| CacheError::IndexFull {
                capacity: self.index.bound(),
            })?;
        self.stats.fills_issued += 1;
        self.fills.push(FillOp {
            key,
            pool: PoolId(p as u16),
            slot: slot as u32,
            fill_generation,
            ticket,
        });
        Ok(fill_generation)
    }

    /// Withdraw the `Resident` slot's key (`Resident -> Evicting`, out of the staged membership)
    /// and park the slot for `finish_plan`; `target` becomes the slot's deferred fill.
    fn withdraw(
        &mut self,
        p: usize,
        slot: usize,
        target: Option<(ExpertKey, u32)>,
        stamp: u64,
    ) -> Result<(), CacheError> {
        if self.parked.len() >= self.config.parked {
            return Err(CacheError::ParkedFull {
                capacity: self.config.parked,
            });
        }
        let pool = PoolId(p as u16);
        let Some(key) = self.pools[p].slots[slot].key else {
            return Err(self.pools[p].refused(format!("resident slot {slot} has no key")));
        };
        self.pools[p].evict(slot)?;
        self.index.remove(&key);
        self.pools[p].slots[slot].key = None;
        self.parked.push((p as u16, slot as u32));
        self.pools[p].dirty = true;
        self.published
            .retain(|entry| !(entry.pool == pool && entry.slot == slot as u32));
        self.withdrawn.push(Withdrawn {
            key,
            pool,
            slot: slot as u32,
        });
        self.stats.evictions += 1;
        if let Some((target_key, rank)) = target {
            self.set_target(p, slot, target_key, rank, stamp)?;
        }
        Ok(())
    }

    fn set_target(
        &mut self,
        p: usize,
        slot: usize,
        key: ExpertKey,
        rank: u32,
        stamp: u64,
    ) -> Result<(), CacheError> {
        let rec = &mut self.pools[p].slots[slot];
        rec.target = Some((key, rank));
        rec.last_used = stamp;
        let loc = Loc {
            pool: p as u16,
            slot: slot as u32,
        };
        self.index
            .insert(key, loc)
            .map_err(|_| CacheError::IndexFull {
                capacity: self.index.bound(),
            })
    }

    // ---------------------------------------------------------------- retirement

    /// Fence the pool's live generation (the last-read fence is recorded on `fence` after the
    /// final consumer was enqueued, then waited) and retire it: its table backing is writable
    /// again. Refused unless something is pending (A2.1: a group with nothing pending never
    /// retires), while a window is open and once the book is poisoned. A pending fence keeps the
    /// state (`WindowFencePending`): call again.
    pub fn retire_generation(
        &mut self,
        pool: PoolId,
        fence: &mut dyn FenceBackend,
    ) -> Result<(), CacheError> {
        let p = self.retirement_gate(pool)?;
        if !self.pools[p].dirty {
            return Err(CacheError::GenerationState {
                pool,
                state: "has nothing pending",
            });
        }
        if self.pools[p].retire(fence)? {
            self.stats.generations_retired += 1;
        }
        Ok(())
    }

    fn retirement_gate(&self, pool: PoolId) -> Result<usize, CacheError> {
        if self.book.is_poisoned() {
            return Err(CacheError::WindowPoisoned);
        }
        if !self.book.is_idle() {
            return Err(CacheError::WindowsOpen);
        }
        self.pool_idx(pool)
    }

    /// After `retire_generation` and the table rewrite: reclaim every withdrawn slot of the pool
    /// (`Evicting -> Free`) and start the fills deferred into them. Returns the fills started.
    pub fn finish_plan(&mut self, pool: PoolId) -> Result<u32, CacheError> {
        let p = self.retirement_gate(pool)?;
        if !self.pools[p].dirty {
            return Ok(0);
        }
        if !matches!(self.pools[p].gen, GenState::Retired) {
            return Err(CacheError::RetirementRequired { pool });
        }
        let mut started = 0u32;
        let mut at = 0usize;
        while at < self.parked.len() {
            let (pp, slot) = self.parked[at];
            if pp as usize != p {
                at += 1;
                continue;
            }
            let slot = slot as usize;
            self.pools[p].reclaim(slot)?;
            self.parked.remove(at);
            let rec = &mut self.pools[p].slots[slot];
            let Some((key, rank)) = rec.target.take() else {
                continue;
            };
            let stamp = rec.last_used;
            if let Err(error) = self.start_fill(p, slot, key, rank, stamp) {
                self.index.remove(&key);
                return Err(error);
            }
            started += 1;
        }
        Ok(started)
    }

    /// Seal the pool's staged membership into its next generation (epoch + 1) under a fresh
    /// lease. The tables must have been rewritten from `published()` / `withdrawn()` and
    /// `finish_plan` must have reclaimed every withdrawn slot. Clears the pool's published and
    /// withdrawn records.
    pub fn seal(&mut self, pool: PoolId) -> Result<GenerationId, CacheError> {
        let p = self.retirement_gate(pool)?;
        if self.parked.iter().any(|&(pp, _)| pp as usize == p) {
            return Err(CacheError::PlanPending);
        }
        let id = self.pools[p].seal()?;
        self.pools[p].dirty = false;
        self.published.retain(|entry| entry.pool != pool);
        self.withdrawn.retain(|entry| entry.pool != pool);
        self.stats.generations_sealed += 1;
        Ok(id)
    }

    // ---------------------------------------------------------------- copy engine

    /// Hand the fills decided so far to the copy engine (their slots are `Filling`).
    pub fn take_fills(&mut self) -> std::vec::Drain<'_, FillOp> {
        self.fills.drain(..)
    }

    /// Fills decided and not yet taken.
    pub fn fills_pending(&self) -> usize {
        self.fills.len()
    }

    /// Enqueue the copies of `tickets` (the callback writes only `ticket.write_spans()`) and
    /// record ONE fence on `copy` after the last. Every error hands the tickets back. A rolled
    /// back fill's ticket is issued like any other (its copy may be skipped): its slot is
    /// withdrawn when the completion arrives.
    #[allow(clippy::type_complexity)]
    pub fn issue_fills(
        &mut self,
        pool: PoolId,
        tickets: Vec<FillTicket>,
        copy: &mut dyn FenceBackend,
        enqueue: &mut dyn FnMut(&FillTicket) -> Result<(), String>,
    ) -> Result<PendingFills, (Vec<FillTicket>, FenceFailure)> {
        match self.pools.get_mut(pool.0 as usize) {
            Some(p) => p.bank.issue_fills(tickets, copy, enqueue),
            None => Err((
                tickets,
                FenceFailure {
                    message: format!("no pool {}", pool.0),
                    quiescence: Quiescence::Proven,
                },
            )),
        }
    }

    /// Poll (`wait == false`) or wait for one batch of issued fills.
    pub fn complete_fills(
        &mut self,
        pool: PoolId,
        batch: PendingFills,
        copy: &mut dyn FenceBackend,
        wait: bool,
    ) -> Result<Vec<FillComplete>, FillsNotDone> {
        match self.pools.get_mut(pool.0 as usize) {
            Some(p) => p.bank.complete_fills(batch, copy, wait),
            None => Err(FillsNotDone::Failed {
                fills: batch.fills().to_vec(),
                failure: FenceFailure {
                    message: format!("no pool {}", pool.0),
                    quiescence: Quiescence::Unknown,
                },
            }),
        }
    }

    /// Tickets that will never be issued: their slots stay `Filling` and are never restored.
    pub fn abandon_fills(&mut self, pool: PoolId, tickets: Vec<FillTicket>) {
        if let Some(p) = self.pools.get_mut(pool.0 as usize) {
            if !tickets.is_empty() {
                p.bank.abandon_fills(tickets);
            }
        }
    }

    /// Deliver a completion token (from `complete_fills`) to the pool it names. It is staged by
    /// the next plan; a token that names no `Filling` slot at its fill generation is ignored
    /// there and counted stale.
    pub fn deliver_completion(
        &mut self,
        pool: PoolId,
        done: FillComplete,
    ) -> Result<(), CacheError> {
        self.pool_idx(pool)?;
        if self.completions.len() >= self.config.completions {
            return Err(CacheError::CompletionBufferFull {
                capacity: self.config.completions,
            });
        }
        self.completions.push_back((pool.0, done));
        Ok(())
    }

    // ------------------------------------------------------- request-by-request path

    /// Reserve `key` in `pool` (the request-by-request surface). A published key is a `Hit`; a
    /// key already pending is `Pending` (never a hit, this caller owns no fill); otherwise a
    /// free slot starts a fill (`Fill`), else a withdrawn slot without a deferred fill is
    /// adopted, else the least recently used slot not used by the open window is withdrawn
    /// (both `Deferred`: the fill starts in `finish_plan`). No slot at all is `NoProgress`.
    pub fn reserve(&mut self, pool: PoolId, key: ExpertKey) -> Result<Reservation, CacheError> {
        if self.frozen {
            return Err(CacheError::Frozen);
        }
        let p = self.pool_idx(pool)?;
        if let Some(loc) = self.index.get(&key) {
            if loc.pool != pool.0 {
                return Err(CacheError::PoolMismatch {
                    key,
                    bound: PoolId(loc.pool),
                    requested: pool,
                });
            }
            let slot = loc.slot as usize;
            let fill_generation = self.pools[p].fill_generation(slot);
            if self.pools[p].phase(slot) == SlotPhaseView::Resident {
                let stamp = self.touch_stamp();
                self.pools[p].slots[slot].last_used = stamp;
                self.stats.reserve_hits += 1;
                return Ok(Reservation::Hit {
                    key,
                    pool,
                    slot: loc.slot,
                    fill_generation,
                });
            }
            self.stats.duplicate_pending += 1;
            let fill_generation = match self.pools[p].phase(slot) {
                SlotPhaseView::Evicting => fill_generation + 1,
                _ => fill_generation,
            };
            return Ok(Reservation::Pending {
                key,
                pool,
                slot: loc.slot,
                fill_generation,
            });
        }

        // A free slot starts the fill now.
        let free = (0..self.pools[p].slots.len())
            .find(|&slot| self.pools[p].phase(slot) == SlotPhaseView::Free);
        if let Some(slot) = free {
            let stamp = self.touch_stamp();
            let fill_generation = self.start_fill(p, slot, key, u32::MAX, stamp)?;
            self.stats.reserve_misses += 1;
            return Ok(Reservation::Fill {
                key,
                pool,
                slot: slot as u32,
                fill_generation,
            });
        }
        // A withdrawn slot nobody wants yet.
        let adopt = self.parked.iter().find_map(|&(pp, slot)| {
            (pp as usize == p && self.pools[p].slots[slot as usize].target.is_none())
                .then_some(slot as usize)
        });
        if let Some(slot) = adopt {
            let stamp = self.touch_stamp();
            self.set_target(p, slot, key, u32::MAX, stamp)?;
            self.stats.reserve_misses += 1;
            return Ok(Reservation::Deferred {
                key,
                pool,
                slot: slot as u32,
                fill_generation: self.pools[p].fill_generation(slot) + 1,
            });
        }
        // The least recently used slot the open window did not use.
        let guard = self.guard();
        let victim = (0..self.pools[p].slots.len())
            .filter(|&slot| {
                self.pools[p].phase(slot) == SlotPhaseView::Resident
                    && self.pools[p].slots[slot].last_used < guard
            })
            .min_by_key(|&slot| {
                let rec = &self.pools[p].slots[slot];
                (rec.last_used, Reverse(rec.rank), slot)
            });
        let Some(slot) = victim else {
            let pl = &self.pools[p];
            let resident = pl.count(SlotPhaseView::Resident);
            return Err(CacheError::NoProgress {
                pool,
                free: 0,
                filling: pl.count(SlotPhaseView::Filling),
                protected: resident,
                withdrawn: pl.count(SlotPhaseView::Evicting),
            });
        };
        let stamp = self.touch_stamp();
        self.withdraw(p, slot, Some((key, u32::MAX)), stamp)?;
        self.stats.reserve_misses += 1;
        Ok(Reservation::Deferred {
            key,
            pool,
            slot: slot as u32,
            fill_generation: self.pools[p].fill_generation(slot) + 1,
        })
    }

    /// Forget a reservation that will never be published (a failed read or copy). Only the
    /// reservation that owns the fill rolls back (`Fill` while its fill is still the slot's
    /// current one, `Deferred` while the slot still wants the key); a hit, a duplicate or a
    /// reservation that no longer matches (the slot moved on to a newer fill generation) is
    /// ignored: `Ok(false)`. A rolled back `Fill` keeps its slot `Filling` until its completion
    /// arrives, which withdraws the slot instead of publishing it.
    pub fn rollback(&mut self, reservation: Reservation) -> Result<bool, CacheError> {
        let (key, pool, slot, fill_generation, deferred) = match reservation {
            Reservation::Fill {
                key,
                pool,
                slot,
                fill_generation,
            } => (key, pool, slot, fill_generation, false),
            Reservation::Deferred {
                key,
                pool,
                slot,
                fill_generation,
            } => (key, pool, slot, fill_generation, true),
            Reservation::Hit { .. } | Reservation::Pending { .. } => return Ok(false),
        };
        let p = self.pool_idx(pool)?;
        if self.index.get(&key) != Some(Loc { pool: pool.0, slot }) {
            return Ok(false);
        }
        let slot = slot as usize;
        let phase = self.pools[p].phase(slot);
        let rec = &self.pools[p].slots[slot];
        let matches = if deferred {
            phase == SlotPhaseView::Evicting
                && rec.target.is_some_and(|(target, _)| target == key)
                && self.pools[p].fill_generation(slot) + 1 == fill_generation
        } else {
            phase == SlotPhaseView::Filling
                && rec.key == Some(key)
                && !rec.cancelled
                && self.pools[p].fill_generation(slot) == fill_generation
        };
        if !matches {
            return Ok(false);
        }
        self.index.remove(&key);
        let rec = &mut self.pools[p].slots[slot];
        if deferred {
            rec.target = None;
        } else {
            rec.key = None;
            rec.cancelled = true;
        }
        self.stats.rollbacks += 1;
        Ok(true)
    }
}
