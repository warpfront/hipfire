// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic dynamic routed-expert cache: the arch-independent POLICY core (slice S1a, CPU only).
//!
//! The core decides which logical expert bundles ([`ExpertKey`]) occupy which physical slots of a
//! pool and when a publication may happen. It owns nothing physical: railgun's
//! [`SlotBank`](railgun::sealed_table::generation::SlotBank) is the sole slot / fill-generation
//! authority. Every slot transition (`Free -> Filling -> Resident -> Evicting -> Free`) goes
//! through the bank's typestate tokens, so a decision made here can never overwrite a slot a live
//! generation still names; the LRU order, the pending-fill record and the use log are metadata
//! beside the bank, never a second commit authority.
//!
//! Descent: this is the arch-independent part of the Qwen4 D2 `TailPolicy`
//! (`hipfire-arch-qwen4::expert_unit_tail`). Its per-window decision rules are kept bit for bit
//! (LRU victims by retirement stamp then colder ranking, candidates by reads then ranking, fill
//! budget `F x min(windows, cap)` minus fills in flight, units used by the retiring group never
//! victims); the differential test pins them against the frozen beta copy. Peer quarantine, the
//! boundary heat mode and the EP `PeerBoard` / pins are NOT part of the core (they stay in the D2
//! adapter); nothing here refuses them, they simply have no state.
//!
//! # Layout
//!
//! * [`ExpertKey`]: `(owner_id, owner_generation, group_id, expert_id)`; a key names a logical
//!   expert BUNDLE (all projection segments), never a single gate or down.
//! * a pool = one `SlotBank` over one compatible block (an exact layout class); keys are bound to
//!   one pool, a request through another pool is [`CacheError::PoolMismatch`].
//! * [`ExpertCachePolicy`]: the pools, a fixed-capacity key index, the window use log, the
//!   [`WindowBook`](window::WindowBook), counters and preallocated output buffers.
//!
//! # Allocation
//!
//! Every capacity is fixed at construction (slots, use log, completions, parked fills, outputs,
//! consumers). Overflow is a named [`CacheError`] variant, never growth. The policy's own steady
//! state (use log, judging, candidate selection, victim selection, fill start, publication,
//! withdrawal, reclaim, window book) performs no heap allocation; railgun's batch tokens
//! (`issue_fills`, `complete_fills`, `seal`) allocate inside railgun and are outside this core.
//!
//! # Protocol (one retirement group)
//!
//! 1. Windows: [`ExpertCachePolicy::begin_window`], `register_consumer` per lane / group / verify
//!    submission, `join_consumer` as each queue proves retirement, `end_window`. Uses are logged
//!    with [`ExpertCachePolicy::record_use`].
//! 2. [`ExpertCachePolicy::plan_group`]: judges the group against the image published at its
//!    start, stages completed fills (`Filling -> Resident`), withdraws victims (`Resident ->
//!    Evicting`, out of the staged membership) and starts fills into slots absent from every
//!    live image.
//! 3. Only if [`ExpertCachePolicy::needs_retirement`] (a completed fill to publish or a victim
//!    to withdraw, A2.1): [`ExpertCachePolicy::retire_generation`] (fence the lease, retire the
//!    generation), write the table images from `published()` / `withdrawn()`, then
//!    [`ExpertCachePolicy::finish_plan`] (reclaim victims, start their fills) and
//!    [`ExpertCachePolicy::seal`]. A group with nothing pending seals, retires and syncs
//!    nothing: the one lease of the live generation simply continues.
//! 4. [`ExpertCachePolicy::take_fills`] hands the fill tickets to the copy engine; their
//!    completion tokens come back through [`ExpertCachePolicy::deliver_completion`].
//!
//! Adapters that judge request by request (DS4-style) use [`ExpertCachePolicy::reserve`],
//! [`ExpertCachePolicy::lookup`] and [`ExpertCachePolicy::rollback`] over the same pools.

mod index;
mod policy;
pub mod window;

#[cfg(test)]
mod tests;

pub use policy::{
    ExpertCachePolicy, FillOp, Lookup, PlanSummary, PolicyConfig, PoolConfig, PoolInit, Published,
    Reservation, Withdrawn,
};
pub use window::{
    CancelOutcome, ConsumerId, RetireCadence, WindowBook, WindowReceipt, DEFAULT_GROUP_MAX_WINDOWS,
};

use railgun::sealed_table::generation::{FenceFailure, ResidencyError};

/// A group's fill budget is `fills_per_window x min(windows, FILL_GROUP_CAP)` by default.
pub const DEFAULT_FILL_GROUP_CAP: usize = 4;

/// Identity of one logical routed-expert bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    /// The model / source owner the bytes belong to.
    pub owner_id: u64,
    /// The owner's allocation generation: a reallocation at the same address is a NEW owner
    /// generation, so a binding made for the old one never hits.
    pub owner_generation: u64,
    /// Trunk layer, MTP head, artifact namespace or rank-local shard.
    pub group_id: u32,
    /// The expert inside its group.
    pub expert_id: u32,
}

impl ExpertKey {
    /// The slot-bank occupant tag of the key (owner identity lives in the policy's side table).
    pub(crate) fn occupant(&self) -> u64 {
        (self.group_id as u64) << 32 | self.expert_id as u64
    }
}

/// A pool of one compatible layout class (index into the policy's pools).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PoolId(pub u16);

/// One distinct key a window group read, with its summed reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowUse {
    pub key: ExpertKey,
    pub pool: PoolId,
    pub reads: u32,
    /// Ranking position of the expert (smaller = hotter): the victim and candidate tie-break.
    pub rank: u32,
}

/// What kind of consumer registered in a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowConsumerPhase {
    Prefill,
    Decode,
    Draft,
    Verify,
    Repair,
}

/// Lifecycle of one slot (the derived public view of the bank's typestate cell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhase {
    /// Holds nothing any table names.
    Free,
    /// A copy into the slot is in flight (or decided); tables still name canonical.
    Filling,
    /// The copy completed: the slot is a member of the staged or the sealed image.
    Live,
    /// Withdrawn from the staged membership: its bytes stay until every generation that named it
    /// retired, then it is reclaimed.
    Withdrawn,
}

/// Counters of the policy (cumulative).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Windows judged (a group judges every window it covers).
    pub windows: u64,
    /// Judged groups (`plan_group` calls).
    pub groups: u64,
    pub core_reads: u64,
    pub core_distinct: u64,
    pub tail_reads: u64,
    pub tail_distinct: u64,
    pub host_reads: u64,
    pub host_distinct: u64,
    pub fills_issued: u64,
    pub fills_published: u64,
    pub evictions: u64,
    /// Candidates dropped for lack of a victim.
    pub skipped_full: u64,
    /// Completions ignored as stale (bank / generation / phase mismatch).
    pub stale_completions: u64,
    /// `reserve` calls that hit a published slot.
    pub reserve_hits: u64,
    /// `reserve` calls that started a fill or withdrew a victim for one.
    pub reserve_misses: u64,
    /// `reserve` calls that found the key already pending (never a hit).
    pub duplicate_pending: u64,
    /// Reservations rolled back (never published).
    pub rollbacks: u64,
    /// Fills whose completion arrived after their key was cancelled (withdrawn, not published).
    pub cancelled_fills: u64,
    /// Generations sealed.
    pub generations_sealed: u64,
    /// Generations retired (a fence wait each).
    pub generations_retired: u64,
}

impl CacheStats {
    /// Slot hit rate: core + tail slot reads over all slot reads.
    pub fn slot_hit(&self) -> f64 {
        let total = self.core_reads + self.tail_reads + self.host_reads;
        if total == 0 {
            1.0
        } else {
            (self.core_reads + self.tail_reads) as f64 / total as f64
        }
    }

    /// Distinct-key hit rate (what the host link is charged by).
    pub fn distinct_hit(&self) -> f64 {
        let total = self.core_distinct + self.tail_distinct + self.host_distinct;
        if total == 0 {
            1.0
        } else {
            (self.core_distinct + self.tail_distinct) as f64 / total as f64
        }
    }
}

/// Every refusal of the core. Capacity overflow, protocol misuse and bank refusals are distinct,
/// named variants; none of them grows a buffer or panics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheError {
    NoPools,
    /// Construction: a pool's shape is unusable (zero slots, slot extents for another slot count).
    PoolShape {
        pool: PoolId,
        why: &'static str,
    },
    /// The slot bank refused a transition.
    Bank {
        pool: PoolId,
        error: ResidencyError,
    },
    UnknownPool(PoolId),
    /// The key is bound to another pool.
    PoolMismatch {
        key: ExpertKey,
        bound: PoolId,
        requested: PoolId,
    },
    /// The key index is full (more keys than slots + reservations the pools can hold).
    IndexFull {
        capacity: usize,
    },
    UseLogFull {
        capacity: usize,
    },
    CompletionBufferFull {
        capacity: usize,
    },
    /// More withdrawn victims awaiting a retirement than the parked buffer holds.
    ParkedFull {
        capacity: usize,
    },
    /// No free slot and no victim: every slot is filling, withdrawn or protected by the current
    /// group. The request is a placement-only skip; nothing changed.
    NoProgress {
        pool: PoolId,
        free: usize,
        filling: usize,
        protected: usize,
        withdrawn: usize,
    },
    /// The pool admits nothing (frozen).
    Frozen,
    /// `plan_group` or `reserve` found withdrawn victims still waiting for `finish_plan`.
    PlanPending,
    /// `finish_plan` needs the pool's generation retired since the plan.
    RetirementRequired {
        pool: PoolId,
    },
    /// The pool's generation is not in the state the call needs.
    GenerationState {
        pool: PoolId,
        state: &'static str,
    },
    /// The pool's window fence cannot finish (a wait that returned pending).
    WindowFencePending {
        pool: PoolId,
    },
    /// The pool's generation can never retire (a quarantined lease or fence).
    Quarantined {
        pool: PoolId,
    },
    Fence {
        pool: PoolId,
        failure: FenceFailure,
    },
    /// The seed left a slot filling (a completion is missing) or named a stale completion.
    SeedIncomplete {
        pool: PoolId,
    },
    WindowBusy,
    WindowNotOpen,
    WindowNotClosed,
    WindowClosed,
    WindowPoisoned,
    /// A plan or retirement needs the window book idle.
    WindowsOpen,
    ConsumerOverflow {
        capacity: usize,
    },
    QueueOutOfRange {
        queue: usize,
        queues: usize,
    },
    UnknownConsumer(u32),
    ConsumerAlreadyJoined(u32),
    ConsumersOutstanding {
        outstanding: usize,
    },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPools => write!(f, "the expert cache needs at least one pool"),
            Self::PoolShape { pool, why } => write!(f, "pool {}: {why}", pool.0),
            Self::Bank { pool, error } => write!(f, "pool {}: slot bank refused: {error}", pool.0),
            Self::UnknownPool(pool) => write!(f, "no pool {}", pool.0),
            Self::PoolMismatch {
                key,
                bound,
                requested,
            } => write!(
                f,
                "key {key:?} is bound to pool {}, requested through pool {}",
                bound.0, requested.0
            ),
            Self::IndexFull { capacity } => write!(f, "key index full ({capacity} keys)"),
            Self::UseLogFull { capacity } => write!(f, "window use log full ({capacity} keys)"),
            Self::CompletionBufferFull { capacity } => {
                write!(f, "completion buffer full ({capacity} tokens)")
            }
            Self::ParkedFull { capacity } => {
                write!(f, "parked-fill buffer full ({capacity} victims)")
            }
            Self::NoProgress {
                pool,
                free,
                filling,
                protected,
                withdrawn,
            } => write!(
                f,
                "pool {}: no slot can be reserved (free {free}, filling {filling}, \
                 protected {protected}, withdrawn {withdrawn})",
                pool.0
            ),
            Self::Frozen => write!(f, "the pool is frozen: it admits nothing"),
            Self::PlanPending => write!(f, "a plan is waiting for `finish_plan`"),
            Self::RetirementRequired { pool } => {
                write!(f, "pool {}: its generation must retire first", pool.0)
            }
            Self::GenerationState { pool, state } => {
                write!(f, "pool {}: generation is {state}", pool.0)
            }
            Self::WindowFencePending { pool } => {
                write!(f, "pool {}: the window fence is still pending", pool.0)
            }
            Self::Quarantined { pool } => write!(f, "pool {}: generation quarantined", pool.0),
            Self::Fence { pool, failure } => {
                write!(f, "pool {}: fence failed: {}", pool.0, failure.message)
            }
            Self::SeedIncomplete { pool } => {
                write!(f, "pool {}: a seeded slot has no completion", pool.0)
            }
            Self::WindowBusy => write!(f, "a window is already open or closing"),
            Self::WindowNotOpen => write!(f, "no window is open"),
            Self::WindowNotClosed => write!(f, "the window is not closed yet"),
            Self::WindowClosed => write!(f, "the window is closed to new consumers"),
            Self::WindowPoisoned => write!(f, "the window book is poisoned (unknown completion)"),
            Self::WindowsOpen => write!(f, "a window is still open"),
            Self::ConsumerOverflow { capacity } => {
                write!(f, "window consumer capacity {capacity} exceeded")
            }
            Self::QueueOutOfRange { queue, queues } => {
                write!(f, "queue {queue} out of range ({queues} queues)")
            }
            Self::UnknownConsumer(id) => write!(f, "no consumer {id} in this window"),
            Self::ConsumerAlreadyJoined(id) => write!(f, "consumer {id} already joined"),
            Self::ConsumersOutstanding { outstanding } => {
                write!(f, "{outstanding} consumer(s) have not joined")
            }
        }
    }
}

impl std::error::Error for CacheError {}
