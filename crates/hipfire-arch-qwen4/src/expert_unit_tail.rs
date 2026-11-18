// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bounded evictable expert TAIL on top of D1's static per-expert core (D2).
//!
//! The D1 plan ([`crate::expert_ep_units`]) keeps a static hot core in VRAM and
//! reads every other unit zero-copy from its pinned host arena. D2 keeps
//! `B - T` units static and turns the other `T` unit slots of the rank into a
//! pool of evictable slots, filled on demand with at most `F` copies per
//! rank per step. A slot only ever holds a byte copy of a unit whose canonical
//! home is the pinned host arena, and only the rank's decode pointer tables
//! are repointed at it, so a table entry is either canonical or a completed
//! slot holding identical bytes: outputs stay exact by construction.
//!
//! This file is the pure part: the flag, the budget charge, the slot policy ([`TailPolicy`]: an
//! adapter over the shared expert-cache core, `hipfire_runtime::expert_cache`, with the D2
//! admission rules as an [`AdmissionStrategy`]) with its tickets and counters, and the
//! cross-rank [`PeerBoard`] of the peer-visibility lever. It touches no device, environment
//! (beyond the parse helpers) or plan; the rank's device part is
//! [`crate::expert_unit_tail_rank`].
//!
//! # Guard
//!
//! One *window* is one trunk forward over every lane's sealed rows (an AR step
//! or a verify window of any row-group count). Hits are judged against the
//! slots published at the START of the window; the policy runs only at a
//! retirement ([`TailPolicy::retire_windows`]), so a window sees ONE table
//! state. A retirement normally covers one window; with the retirement cadence
//! ([`RetireCadence`]) it covers a group of consecutive windows (up to
//! [`RETIRE_MAX_WINDOWS`]) that all saw the one table state of the group's start, judged
//! together ([`TailPolicy::retire_windows`]: a unit any window of the group read is never a
//! victim; the fill budget is `F x windows`, at most `F x `[`FILL_GROUP_CAP`]). A slot is
//! overwritten only after its previous unit was repointed to canonical by the table upload that
//! precedes the copy AND every generation that named it retired, and a fill is published only
//! after its copy completed (the `completed` tokens of [`TailPolicy::retire_windows`]); a stale
//! (other bank, older generation) completion is ignored.
//!
//! # Generations, leases and the slot bank
//!
//! The core owns the rank's [`SlotBank`] (railgun's generation-owned resource API): every slot is
//! a typestate cell (`Free` -> `Filling` -> `Resident` -> `Evicting` -> `Free`;
//! [`SlotPhase`] is the derived view), and a transition that could overwrite bytes a reader may
//! still name needs a token that cannot be forged: [`FillComplete`] (a fill's copy fence
//! completed) to publish, a retirement token (every generation naming the slot retired, and in
//! PEER mode every peer fenced) to reclaim.
//!
//! * One immutable table GENERATION per retirement: the staged membership (the `Resident` slots)
//!   is sealed through the tail's table backing as generation `{ epoch j, sub 0 }` at retirement
//!   `j` (the warm start is epoch 0, sub 1; a borrow resume re-seals `sub + 1`). The core
//!   retires only when something is pending, the tail retires EVERY group, so the peer clock `j`
//!   is [`TailPolicy`]'s own group counter.
//! * One WINDOW LEASE per generation, from its sealing until the next retirement's fence (the
//!   core's `retire_generation_now`): every window and every forward the rank enqueues on its
//!   compute stream in between reads that generation. Retirement `j` fences the lease (the
//!   compute backend's wait is the stream synchronize the protocol always did), retires
//!   generation `j - 1` (PEER mode: the one `released_through` call of the retirement,
//!   [`PeerReaderFence`]) and only then plans the group ([`TailPolicy::retire_windows`]), so a
//!   victim's slot is reclaimable exactly when no generation names it any more.
//! * The table backing (the tables' write capability) exists only between the retirement of
//!   generation `j - 1` and the sealing of generation `j`: table images are written and uploaded
//!   only then, under the [`TablesOpen`] witness ([`TailPolicy::open_tables`]), which the sealing
//!   ([`TailPolicy::finish_retirement`]) consumes.
//! * An unfinished or dropped lease quarantines its generation (it can never retire, its slots
//!   never reclaim); an unknown-quiescence fence failure quarantines the slots it guarded.
//!
//! # Peer visibility
//!
//! A sub-512-row prefill takes the FULL-table route: every rank computes every expert, so a unit
//! the OWNER holds in a tail slot is read by the OTHER ranks from the owner's pinned host arena
//! unless their full tables name the owner's slot. A configured tail on a mesh of two or more
//! ranks shares one [`PeerBoard`] across the ranks (threads of one process) and runs the protocol
//! below. A single-rank mesh has no board and no quarantine, and every decision of
//! [`TailPolicy::retire_windows`] is the pre-peer one.
//!
//! * POST. At each of its retirements `k` (its retirement counter, from 1; the warm start posts
//!   as `k = 0`) the owner posts the deltas of its OWN publications: `(unit, Some(slot
//!   addresses))` for a fill published at `k`, `(unit, None)` for a unit evicted at `k`. Only
//!   completed fills are ever posted: a publication comes out of a `FillComplete` token of
//!   [`TailPolicy::retire_peer_windows`]'s `completed`, i.e. the owner observed the fill's
//!   copy-stream event complete.
//! * APPLY + ACK. At its retirement `j`, after its own policy step and before its own fills, a
//!   rank takes every delta posted so far and applies it to its FULL-table image of a PEER-owned
//!   entry whose full entry equals the owner's canonical entries (`peer_shared` in the rank part;
//!   any other entry is never touched), uploads the touched full tables on its active stream with
//!   its own table uploads, and only then records the ACK `(newest owner retirement it applied,
//!   j)`. A rank therefore names a peer slot only after the owner posted the publication (host
//!   order) and after its own apply.
//! * QUARANTINE. A slot the owner evicts at retirement `k` goes back to canonical in the owner's
//!   tables at once but its BYTES stay: it is `Quarantined` (neither free nor a victim). It
//!   becomes free at an owner retirement `m` only when, for EVERY peer `r`, some ack of `r` has
//!   `applied >= k` and `m >= j + 1` for the `j` it was recorded at
//!   ([`PeerBoard::released_through`]). At most [`QUARANTINE_WINDOWS`]` x ` (this retirement's
//!   fill budget, `F` per window of its group) slots are quarantined at once, so a peer that never
//!   acknowledges (frozen, released) cannot drain the pool; candidates that find no slot are
//!   `skipped`.
//! * RESERVE (per-window mode, saturated demand). An evicted slot is unusable for the one or two
//!   retirements its acknowledgement takes, so a candidate that has to evict waits for it, and
//!   under sustained misses eviction and refill alternate (about `F / 2` fills per window). When
//!   a retirement has at least as many candidates as its fill budget, every candidate it
//!   considers fills a `Free` slot if one exists AND evicts one victim into quarantine while the
//!   reserve (`Free` + `Quarantined` slots) is below [`RESERVE_RETIREMENTS`]` x ` the budget:
//!   the evictions of one retirement become the free slots of a later one, and `F` fills per
//!   window are sustained. Below saturation the candidate-waits rule stays (a candidate that has
//!   to evict is admitted only if it misses again, which keeps one-off misses out when demand is
//!   light, where an extra fill's copy costs more than the host reads it saves), and the
//!   reserve drains into fills. Each candidate still causes at most one eviction, and the
//!   reserve bound is below the quarantine bound. Boundary mode keeps the candidate-waits rule
//!   (its evictions need a challenger's heat).
//!
//! Cadence. With the retirement cadence ([`RetireCadence`]) a retirement ends a GROUP of windows
//! and "window `j + 1`" below means the first window after the group retired at `j`; the
//! argument is otherwise word for word the same, because it only uses (a) that every rank
//! retires the same groups in the same order (the retirement counters are one clock) and (b)
//! that retirement `j` of `r` precedes `r`'s next window's forward and retirement `m` of the owner
//! follows a forward of a LATER window. (a) holds because the decision to retire at a window end
//! is a function of the routed-id sequence of the windows (the sum of the cross-rank exchange's
//! `slots`, [`RetireCadence`]'s rank-independent rule), which every rank sees identically; it
//! never reads a rank-local count (logged ids differ per rank). A window that does not complete
//! a group does nothing on the device, so no table or slot changes inside a group.
//!
//! Why a refill never changes bytes a peer can still read as another unit. A peer image entry
//! names slot `S` for unit `U` only from the peer's apply of the publication of `(U, S)` until its
//! apply of the unpublication of `U`. The owner posted the publication after `S`'s copy completed
//! and `S` holds `U`'s bytes until the owner refills it, so every read through the entry is a
//! read of `U`. The owner refills `S` (evicted at `k`, quarantined) at its retirement `m` only
//! when every peer `r` applied the unpublication at some retirement `j <= m - 1`. `r` enqueued
//! that upload on its active stream at retirement `j`, i.e. BEFORE any kernel of its window
//! `j + 1` (host order = stream order), and every kernel of `r` that was enqueued earlier (every
//! forward that could name `S` through the old image, decode window or full-table chunk forward
//! alike) precedes the upload in the same stream. The owner's retirement `m >= j + 1` follows the
//! owner's `stream_synchronize` after its window-`m` forward, whose cross-rank row exchanges
//! consumed outputs of `r`'s window-`(j + 1)` kernels (all ranks run every decode window in
//! lockstep and retire the same windows in the same order, so the retirement counters are one
//! clock): `r`'s stream has executed through those kernels, hence the upload and every earlier
//! `r` forward that named `S` have executed before the owner issues the fill (the owner's fill is
//! enqueued after that synchronize, after its own table upload). Forwards `r` enqueues after the
//! upload read an image that no longer names `S` for `U`. A full-table chunk forward is preceded by
//! `ep_quiesce` (device synchronize + host barrier on all ranks), which only strengthens this.
//! The condition reads only acks with `j <= m - 1`, all of which were recorded before the owner's
//! retirement `m` (`r`'s retirement `j` precedes `r`'s window `j + 1` forward, which the owner's
//! window `j + 1` forward needs), so the decision never depends on thread timing; a later or
//! missing ack only delays the release.
//!
//! Fail closed: a rank whose retirement finds a post newer than its own counter (the ranks'
//! windows are out of step), or whose upload fails, freezes and records no ack, so the owner never
//! refills a slot that rank may still name; a frozen tail leaves every table state valid
//! (canonical, or a live slot with complete bytes).
//!
//! # Dynamic pins
//!
//! A configured tail keeps the D1.9 pinned owner assignment of the pinned unit layers, and the pin
//! plane follows the tail's publications: a unit published in its holder's slot is pinned to its
//! holder. The protocol, its plane-identity argument, its visibility argument and its failure path
//! are in [`pins`].

pub mod pins;
pub use pins::{PinDelta, PinError, PinPlanes, PinPost, PinProtocol, PinStats, PIN_LAG};

/// Frozen copy of beta's `TailPolicy` for the differential tests.
#[cfg(test)]
mod beta_ref;
/// Differential tests: the migrated policy against the frozen beta copy on seeded random traces.
#[cfg(test)]
mod differential;
/// Race-injection tests: the retirement protocol on a simulated device timeline.
#[cfg(test)]
mod race;
/// Test rig: the policy driven through the retirement protocol over a fake device.
#[cfg(test)]
pub(crate) mod rig;

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};

use hipfire_runtime::expert_cache::{
    AdmissionStrategy, CacheError, Candidate, CandidateSink, EvictMode, ExpertCachePolicy,
    ExpertKey, Lookup, PassContext, PassPlan, PlanContext, PolicyConfig, PoolConfig, PoolId,
    PoolInit, PoolMode, SlotPhase as CorePhase, Victim, WindowUse as CoreUse,
};
use railgun::sealed_table::generation::{
    BlockReclaimed, Expert, FenceBackend, FenceFailure, FillComplete, FillTicket, FillsNotDone,
    GenerationId, LentBlock, PeerReaderFence, PendingFills, PendingRestore, Quiesced,
    ResidencyError, ScratchLoan, SlotBank, SlotSpan,
};
use railgun::Binding;

/// `T,F[,b=N]`: tail slots per rank, fills per rank per window, optional
/// boundary mode (admit only every `N` windows). Unset = the shipped default
/// ([`EP_UNIT_TAIL_DEFAULT_SLOTS`],[`EP_UNIT_TAIL_DEFAULT_FILLS`]) on a load with a per-expert
/// unit plan; empty, `0` or `off` leaves the tail off.
pub const EP_UNIT_TAIL_ENV: &str = "HIPFIRE_EP_UNIT_TAIL";
/// `1`: build the tail (pool, seed, usage log) but never admit or evict: the
/// A/B arm that prices the tail's bookkeeping without any fill traffic.
pub const EP_UNIT_TAIL_FREEZE_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_FREEZE";
/// `1`: print a `[unit-tail]` counter line every 64 windows and at release.
pub const EP_UNIT_TAIL_LOG_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_LOG";
/// `G` (whole units, default [`EP_UNIT_TAIL_STAGE_GAP_DEFAULT`]): with a configured tail, the
/// slot-aware unit stage merges two H2D copy runs across a slot-live gap shorter than `G` units
/// (the gap units are copied too and simply not referenced: their stage-table entries still name
/// the slot). `0` = exactly the complement of the slot-live units.
pub const EP_UNIT_TAIL_STAGE_GAP_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_STAGE_GAP";
/// Default of [`EP_UNIT_TAIL_STAGE_GAP_ENV`].
pub const EP_UNIT_TAIL_STAGE_GAP_DEFAULT: usize = 8;

/// `N` routed ids (default [`EP_UNIT_TAIL_RETIRE_IDS_DEFAULT`]): `step_end` retires the tail's
/// window log only once the windows since the last retirement carry at least `N` routed ids
/// (rows x top-k x MoE calls, see [`RetireCadence`]) or [`RETIRE_MAX_WINDOWS`] windows have
/// accumulated; a window that does not complete a group returns at once (no stream sync, no D2H)
/// and the next one keeps appending to the log. `0` = retire every window.
pub const EP_UNIT_TAIL_RETIRE_IDS_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_RETIRE_IDS";
/// Default of [`EP_UNIT_TAIL_RETIRE_IDS_ENV`].
pub const EP_UNIT_TAIL_RETIRE_IDS_DEFAULT: u64 = 4096;
/// A group never accumulates more windows than this.
pub const RETIRE_MAX_WINDOWS: u32 = 8;
/// A retirement of `n` windows may start `F x min(n, FILL_GROUP_CAP)` fills.
pub const FILL_GROUP_CAP: usize = 4;

/// One MiB.
const MIB: u64 = 1 << 20;

/// What `HIPFIRE_EP_UNIT_TAIL` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpUnitTailConfig {
    /// Evictable slots per rank (`T`).
    pub tail_units: usize,
    /// Fills one rank may start per window (`F`).
    pub fill_per_step: usize,
    /// `1`: admit on every retired window; `N > 1`: boundary mode, admit only on
    /// every `N`th window with a decayed-heat ranking and hysteresis.
    pub boundary_every: u32,
}

/// Parse [`EP_UNIT_TAIL_ENV`]. `Ok(None)` = off.
pub fn parse_ep_unit_tail(raw: Option<&str>) -> Result<Option<EpUnitTailConfig>, String> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(None);
    };
    if raw.is_empty() || raw == "0" || raw.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let bad = |why: &str| format!("{EP_UNIT_TAIL_ENV}={raw}: {why} (expected `T,F[,b=N]`)");
    let mut parts = raw.split(',').map(str::trim);
    let tail_units: usize = parts
        .next()
        .and_then(|part| part.parse().ok())
        .filter(|&t| t > 0)
        .ok_or_else(|| bad("T must be a positive integer"))?;
    let fill_per_step: usize = parts
        .next()
        .and_then(|part| part.parse().ok())
        .filter(|&f| f > 0)
        .ok_or_else(|| bad("F must be a positive integer"))?;
    let boundary_every = match parts.next() {
        None => 1,
        Some(part) => part
            .strip_prefix("b=")
            .or_else(|| part.strip_prefix("boundary="))
            .and_then(|n| n.parse::<u32>().ok())
            .filter(|&n| n >= 1)
            .ok_or_else(|| bad("the third field must be `b=N` with N >= 1"))?,
    };
    if parts.next().is_some() {
        return Err(bad("too many fields"));
    }
    Ok(Some(EpUnitTailConfig {
        tail_units,
        fill_per_step,
        boundary_every,
    }))
}

/// Default tail when [`EP_UNIT_TAIL_ENV`] is unset: `T` slots per rank.
pub const EP_UNIT_TAIL_DEFAULT_SLOTS: usize = 1536;
/// Default tail when [`EP_UNIT_TAIL_ENV`] is unset: `F` fills per rank per window.
pub const EP_UNIT_TAIL_DEFAULT_FILLS: usize = 16;

/// [`parse_ep_unit_tail`] with the shipped default: unset = the default `T,F` (a tail exists only
/// over a per-expert unit plan, see [`ep_unit_tail_explicit`]); empty / `0` / `off` = off.
pub fn resolve_ep_unit_tail(raw: Option<&str>) -> Result<Option<EpUnitTailConfig>, String> {
    match raw {
        None => Ok(Some(EpUnitTailConfig {
            tail_units: EP_UNIT_TAIL_DEFAULT_SLOTS,
            fill_per_step: EP_UNIT_TAIL_DEFAULT_FILLS,
            boundary_every: 1,
        })),
        raw => parse_ep_unit_tail(raw),
    }
}

/// Whether [`EP_UNIT_TAIL_ENV`] was set to a tail (not unset, not off): an explicit tail on a load
/// without a unit plan is a load error; the shipped default silently stays off there.
pub fn ep_unit_tail_explicit() -> bool {
    let raw = rdna_compute::FeatureFlags::process()
        .ep_unit_tail
        .as_deref();
    raw.is_some() && parse_ep_unit_tail(raw).is_ok_and(|config| config.is_some())
}

/// Whether [`EP_UNIT_TAIL_ENV`] asks for a tail in this process. A configured tail's pool is
/// visible to prefill routes (the full chunk tables follow the slot publications, and the unit
/// stage's slot-aware tables skip the copies of slot-live units), and its pinned unit layers
/// follow the tail's publications (the dynamic pins of [`pins`]).
pub fn ep_unit_tail_configured() -> bool {
    resolve_ep_unit_tail(
        rdna_compute::FeatureFlags::process()
            .ep_unit_tail
            .as_deref(),
    )
    .is_ok_and(|config| config.is_some())
}

/// Parse [`EP_UNIT_TAIL_STAGE_GAP_ENV`]: unset or empty = the default, else a non-negative integer.
pub fn parse_stage_gap(raw: Option<&str>) -> Result<usize, String> {
    match raw.map(str::trim).filter(|raw| !raw.is_empty()) {
        None => Ok(EP_UNIT_TAIL_STAGE_GAP_DEFAULT),
        Some(raw) => raw.parse::<usize>().map_err(|_| {
            format!("{EP_UNIT_TAIL_STAGE_GAP_ENV}={raw}: expected a non-negative integer (units)")
        }),
    }
}

/// Parse [`EP_UNIT_TAIL_RETIRE_IDS_ENV`]: unset or empty = the default, else a non-negative
/// integer.
pub fn parse_retire_ids(raw: Option<&str>) -> Result<u64, String> {
    match raw.map(str::trim).filter(|raw| !raw.is_empty()) {
        None => Ok(EP_UNIT_TAIL_RETIRE_IDS_DEFAULT),
        Some(raw) => raw.parse::<u64>().map_err(|_| {
            format!(
                "{EP_UNIT_TAIL_RETIRE_IDS_ENV}={raw}: expected a non-negative integer (routed ids)"
            )
        }),
    }
}

/// The retirement cadence threshold ([`EP_UNIT_TAIL_RETIRE_IDS_ENV`]) of this process.
pub fn ep_unit_tail_retire_ids() -> Result<u64, String> {
    parse_retire_ids(
        hipfire_config::developer_var(EP_UNIT_TAIL_RETIRE_IDS_ENV)
            .ok()
            .as_deref(),
    )
}

/// The retirement cadence: which window ends retire the tail's accumulated log.
///
/// # The rule (rank-independent)
///
/// Every window reports `routed`, the sum of the `slots` argument (routed rows x top-k) of its
/// trunk MoE gather calls, counted in the row-gather hook BEFORE anything rank-local (a frozen
/// tail, a short top-k buffer, a log overflow, the experts a rank holds) can change what the
/// rank logs. `slots` is the extent of the cross-rank row exchange of that call, which every
/// rank enters with the same value, so every rank feeds [`Self::window_end`] the same sequence
/// and decides identically. The group retires at the window end when, after adding the window,
///
/// * its routed ids reach `retire_ids`, or
/// * it holds [`RETIRE_MAX_WINDOWS`] windows, or
/// * the next window could overflow the log: routed ids so far plus the LARGEST window seen
///   exceed `log_ids` (the log holds one id per routed id, so the routed total bounds the logged
///   total from above and the check is conservative on every rank).
///
/// Only retirements move the tail's tables, so every retirement counter of the peer-visibility
/// protocol (module docs) stays one clock across ranks.
#[derive(Clone, Debug)]
pub struct RetireCadence {
    retire_ids: u64,
    log_ids: u64,
    windows: u32,
    routed: u64,
    largest: u64,
}

impl RetireCadence {
    /// `retire_ids`: the threshold of [`EP_UNIT_TAIL_RETIRE_IDS_ENV`]; `log_ids`: the capacity
    /// of one rank's usage log in ids.
    pub fn new(retire_ids: u64, log_ids: u64) -> Self {
        Self {
            retire_ids,
            log_ids,
            windows: 0,
            routed: 0,
            largest: 0,
        }
    }

    /// One window ended with `routed` routed ids. `Some(n)`: retire now, the group is `n`
    /// windows (and the group is empty again); `None`: keep accumulating.
    pub fn window_end(&mut self, routed: u64) -> Option<u32> {
        self.windows += 1;
        self.routed += routed;
        self.largest = self.largest.max(routed);
        let retire = self.routed >= self.retire_ids
            || self.windows >= RETIRE_MAX_WINDOWS
            || self.routed + self.largest > self.log_ids;
        retire.then(|| {
            let windows = self.windows;
            self.windows = 0;
            self.routed = 0;
            windows
        })
    }

    /// Windows accumulated since the last retirement.
    pub fn pending(&self) -> u32 {
        self.windows
    }

    /// Drop the accumulated group (the tail froze or was released).
    pub fn clear(&mut self) {
        self.windows = 0;
        self.routed = 0;
    }
}

/// The unit stage's copy-run merge gap ([`EP_UNIT_TAIL_STAGE_GAP_ENV`]) of this process.
pub fn ep_unit_tail_stage_gap() -> Result<usize, String> {
    parse_stage_gap(
        hipfire_config::developer_var(EP_UNIT_TAIL_STAGE_GAP_ENV)
            .ok()
            .as_deref(),
    )
}

/// VRAM one rank's tail pool takes: `T` units plus the two block tail pads
/// (gate/up block and down block, [`crate::expert_arena::ARENA_TAIL_PAD_BYTES`]
/// each, as in a unit arena).
pub fn ep_unit_tail_pool_bytes(tail_units: usize, unit_bytes: u64) -> u64 {
    tail_units as u64 * unit_bytes + 2 * crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64
}

/// The (negative) MiB delta the tail takes from every rank's unit budget so the
/// core is `B - T` units: the pool's bytes rounded UP to whole MiB, the same
/// unit as the `HIPFIRE_EP_UNIT_BUDGET_DELTA_MIB` lab knob it is added to.
pub fn ep_unit_tail_budget_delta_mib(config: &EpUnitTailConfig, unit_bytes: u64) -> i64 {
    let bytes = ep_unit_tail_pool_bytes(config.tail_units, unit_bytes);
    -(bytes.div_ceil(MIB) as i64)
}

/// A unit: `(trunk layer, expert)`.
pub type UnitKey = (u16, u16);

/// The slot-bank occupant id of a unit: `layer << 32 | expert`, the id the shared cache core
/// gives the key [`unit_key`] builds (`group_id << 32 | expert_id`).
pub const fn unit_occupant(unit: UnitKey) -> u64 {
    (unit.0 as u64) << 32 | unit.1 as u64
}

/// The unit a [`unit_occupant`] id names.
pub const fn occupant_unit(occupant: u64) -> UnitKey {
    ((occupant >> 32) as u16, (occupant & 0xffff_ffff) as u16)
}

/// The key of `unit` in the shared cache core: owner 0, generation 0, the layer as the group.
fn unit_key(unit: UnitKey) -> ExpertKey {
    ExpertKey {
        owner_id: 0,
        owner_generation: 0,
        group_id: unit.0 as u32,
        expert_id: unit.1 as u32,
    }
}

/// The unit a [`unit_key`] names.
fn key_unit(key: &ExpertKey) -> UnitKey {
    (key.group_id as u16, key.expert_id as u16)
}

/// Lifecycle of one tail slot: the derived public view of the slot's typestate cell
/// ([`SlotCell`]: `Free` / `Filling` / `Resident` / `Evicting`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhase {
    /// Holds nothing the tables name (`OwnedSlot<Free>`).
    Free,
    /// A copy into the slot is in flight; the tables still name canonical (`OwnedSlot<Filling>`).
    Filling,
    /// The copy completed and the tables name the slot (`OwnedSlot<Resident>`, a member of the
    /// staged generation).
    Live,
    /// PEER mode: evicted (the owner's tables name canonical again) but a peer's full table may
    /// still name the slot, so its bytes must stay until every peer acknowledged the
    /// unpublication: neither free nor a victim (`OwnedSlot<Evicting>`).
    Quarantined,
}

/// A copy the device part must start: `unit`'s canonical bytes into `slot`. The `ticket` is the
/// bank's non-Clone fill capability: [`TailPolicy::issue_fills`] consumes it, and the copy it
/// enqueues writes only `ticket.write_spans()`. A `FillOp` cannot be built outside railgun's
/// slot bank (the ticket has no public constructor):
///
/// ```compile_fail
/// use hipfire_arch_qwen4::expert_unit_tail::FillOp;
/// let _op = FillOp { slot: 0, unit: (0, 1), generation: 1 };
/// ```
///
/// ```
/// use hipfire_arch_qwen4::expert_unit_tail::FillOp;
/// fn decision(op: &FillOp) -> (usize, (u16, u16), u64) { op.key() }
/// ```
pub struct FillOp {
    pub slot: usize,
    pub unit: UnitKey,
    pub generation: u64,
    pub ticket: FillTicket,
}

impl FillOp {
    /// `(slot, unit, generation)`: what the policy decided (the ticket is a capability, not a
    /// decision).
    pub fn key(&self) -> (usize, UnitKey, u64) {
        (self.slot, self.unit, self.generation)
    }
}

impl PartialEq for FillOp {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for FillOp {}

impl std::fmt::Debug for FillOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FillOp")
            .field("slot", &self.slot)
            .field("unit", &self.unit)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

/// What one retired window changes. The device part uploads the touched layers'
/// tables (publish -> slot address, unpublish -> canonical) BEFORE it starts
/// the fills. The fills move to the device part (the type is not `Clone`: a fill's ticket is an
/// exclusive capability):
///
/// ```compile_fail
/// use hipfire_arch_qwen4::expert_unit_tail::Retire;
/// fn again(r: &Retire) -> Retire { r.clone() }
/// ```
///
/// ```
/// use hipfire_arch_qwen4::expert_unit_tail::Retire;
/// let r = Retire::default();
/// assert!(r.fills.is_empty() && r.publish.is_empty() && r.unpublish.is_empty());
/// ```
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Retire {
    /// Units whose table entry now names their slot.
    pub publish: Vec<(UnitKey, usize)>,
    /// Units (evicted) whose table entry returns to canonical.
    pub unpublish: Vec<UnitKey>,
    /// Copies to start (their slots are `Filling`). They move to the device part.
    pub fills: Vec<FillOp>,
}

/// One distinct non-core unit a window read on this rank, with its slot reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowUse {
    pub unit: UnitKey,
    pub reads: u32,
}

/// Counters of the tail (cumulative).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TailStats {
    /// Decode windows judged (a retirement judges every window of its group).
    pub windows: u64,
    /// Retirements: the policy steps the tables, fills and the peer protocol count in.
    pub retirements: u64,
    /// Slot reads / distinct units served by the static core.
    pub core_reads: u64,
    pub core_distinct: u64,
    /// ... by a live tail slot at window start.
    pub tail_reads: u64,
    pub tail_distinct: u64,
    /// ... read from the pinned host arena (misses).
    pub host_reads: u64,
    pub host_distinct: u64,
    pub fills_issued: u64,
    pub fills_published: u64,
    pub evictions: u64,
    /// Candidates dropped for lack of a victim.
    pub skipped_full: u64,
    /// Completions ignored as stale (generation/phase mismatch).
    pub stale_completions: u64,
    /// Slots put into quarantine (PEER mode): evictions whose refill waits for the peers' acks.
    pub quarantined: u64,
}

impl TailStats {
    /// Slot hit rate: core + tail slot reads over all slot reads.
    pub fn slot_hit(&self) -> f64 {
        let total = self.core_reads + self.tail_reads + self.host_reads;
        if total == 0 {
            1.0
        } else {
            (self.core_reads + self.tail_reads) as f64 / total as f64
        }
    }

    /// Distinct-unit hit rate (what the host link is charged by).
    pub fn distinct_hit(&self) -> f64 {
        let total = self.core_distinct + self.tail_distinct + self.host_distinct;
        if total == 0 {
            1.0
        } else {
            (self.core_distinct + self.tail_distinct) as f64 / total as f64
        }
    }
}

/// Heat decay per boundary (matches the LINKBYTES-D2 boundary design).
const HEAT_DECAY: f32 = 0.8;
/// A challenger must beat the victim's heat by this factor.
const HEAT_HYSTERESIS: f32 = 1.10;
/// PEER mode: at most this many windows' worth of fills (`x F` slots) are quarantined at once.
/// An eviction is acknowledged one or two retirements later, so `F` evictions per window keep at
/// most about `2 x F` slots in quarantine; the third window is jitter headroom, and the bound
/// keeps a silent peer from draining the pool.
pub const QUARANTINE_WINDOWS: usize = 3;
/// PEER per-window mode under saturated demand: the reserve (free + quarantined slots) is kept up
/// to this many times the fill budget, the one or two retirements an eviction's acknowledgement
/// takes (a larger reserve only holds slots idle: in an offline replay of served EP2 code traces,
/// 3x was about 10 MiB/step of host reads worse at code C8 than 2x).
pub const RESERVE_RETIREMENTS: usize = 2;

/// The tail's one pool inside its core.
const POOL: PoolId = PoolId(0);

/// The physical resources of the tail's pool, handed to the core at construction.
pub struct TailPool {
    /// The pool allocation (an `Expert` loan).
    pub block: LentBlock<Expert>,
    /// `spans[i]`: the write extents of slot `i`, block relative.
    pub spans: Vec<Vec<SlotSpan>>,
    /// The pointer-table allocations every generation is published through.
    pub tables: Vec<Binding>,
    /// An upper bound of the distinct units the tail can ever see (the core's use-log capacity).
    pub units: usize,
}

/// The tables' write witness: it exists only while the pool's table backing is writable, from
/// the retirement of the published generation until the next generation is sealed over the
/// tables ([`TailPolicy::open_tables`]). Every table image write and upload takes `&TablesOpen`;
/// sealing ([`TailPolicy::finish_retirement`], [`TailPolicy::reseal`], [`TailPolicy::seal_warm`])
/// consumes it. The backing itself stays in the core ([`ExpertCachePolicy::backing`]).
#[derive(Debug)]
pub struct TablesOpen(());

/// The D2 admission rules over the shared core's seam ([`AdmissionStrategy`]): per-window reads,
/// the boundary heat mode (decayed `f32` heat, the boundary gate, the 10% hysteresis veto), the
/// PEER quarantine (an eviction withdraws the victim without a target, capped at
/// [`QUARANTINE_WINDOWS`] windows of fills) and the reserve of the saturated peer per-window mode.
/// The core keeps every slot invariant; this only decides what is worth admitting.
struct TailAdmission {
    peer: bool,
    boundary_every: u32,
    /// Ranking position of a unit (smaller = hotter): the tie-break and, in boundary mode, the
    /// prior of the heat.
    priority: HashMap<UnitKey, u32>,
    /// Boundary mode: decayed read heat of non-core units.
    heat: HashMap<UnitKey, f32>,
}

impl TailAdmission {
    fn boundary(&self) -> bool {
        self.boundary_every > 1
    }

    fn rank_of(&self, unit: UnitKey) -> u32 {
        self.priority.get(&unit).copied().unwrap_or(u32::MAX)
    }

    fn heat_of(&self, unit: UnitKey) -> f32 {
        self.heat.get(&unit).copied().unwrap_or(0.0)
    }
}

impl AdmissionStrategy for TailAdmission {
    fn observe(&mut self, _ctx: &PlanContext, log: &[CoreUse]) {
        if self.boundary() {
            for used in log {
                *self.heat.entry(key_unit(&used.key)).or_insert(0.0) += used.reads as f32;
            }
        }
    }

    fn open_gate(&mut self, ctx: &PlanContext) -> bool {
        // Boundary mode admits when the group crossed a multiple of N WINDOWS (with one window
        // per retirement: every Nth window).
        !self.boundary() || ctx.crossed(self.boundary_every as u64)
    }

    fn score(&mut self, used: &CoreUse) -> f32 {
        if self.boundary() {
            self.heat_of(key_unit(&used.key))
        } else {
            used.reads as f32
        }
    }

    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        // Units hot over the interval but not missed in the last window. The sink drops the
        // bound and the read ones; the core's candidate order makes the HashMap order moot.
        if self.boundary() {
            for (&unit, &heat) in &self.heat {
                sink.push(unit_key(unit), POOL, self.rank_of(unit), heat)?;
            }
        }
        Ok(())
    }

    fn pass(&mut self, ctx: &PassContext) -> PassPlan {
        if !self.peer {
            return PassPlan::default();
        }
        // PEER per-window mode under SATURATED demand (at least `budget` candidates) keeps the
        // reserve (module docs, RESERVE). Below saturation the candidate that has to evict waits
        // (it is admitted if it misses again).
        PassPlan {
            evict: EvictMode::Quarantine,
            quarantine_cap: QUARANTINE_WINDOWS * ctx.total_budget,
            reserve_target: (!self.boundary() && ctx.candidates >= ctx.budget)
                .then_some(RESERVE_RETIREMENTS * ctx.budget),
        }
    }

    fn veto_victim(&mut self, candidate: &Candidate, victim: &Victim) -> bool {
        self.boundary() && candidate.score < self.heat_of(key_unit(&victim.key)) * HEAT_HYSTERESIS
    }

    fn settled(&mut self, _ctx: &PlanContext, reached: bool) {
        if self.boundary() && reached {
            self.heat.retain(|_, h| {
                *h *= HEAT_DECAY;
                *h > 0.01
            });
        }
    }
}

/// The slot policy of one rank's tail, an adapter over the shared cache core
/// ([`ExpertCachePolicy`], one pool): the core owns the rank's [`SlotBank`] and every decision
/// (LRU eviction, admit-on-miss bounded by `F` fills per window, tickets, counters); this type
/// keeps what is the tail's own: the group counter of the peer clock, the unit <-> key mapping
/// and the D2 admission rules ([`TailAdmission`]). Every slot transition goes through the
/// bank's tokens, so a decision here can never overwrite a slot a live generation still names.
pub struct TailPolicy {
    config: EpUnitTailConfig,
    /// PEER mode: evictions quarantine the slot instead of refilling it in the same retirement.
    peer: bool,
    core: ExpertCachePolicy,
    admission: TailAdmission,
    /// Retirements made so far: the peer clock `j` every rank counts alike (the core retires only
    /// when something is pending, the tail retires every group).
    retirements: u64,
    /// Calls the core refused (a protocol bug: never expected, fail closed). The device part
    /// freezes the tail on the first error it sees.
    refusals: u64,
    first_refusal: Option<String>,
}

impl std::fmt::Debug for TailPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TailPolicy")
            .field("config", &self.config)
            .field("peer", &self.peer)
            .field("slots", &self.slots())
            .field("retirements", &self.retirements)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl TailPolicy {
    /// `priority[unit]`: ranking position of every unit the rank may see (smaller = hotter); an
    /// unknown unit sorts last. `pool` is the rank's pool allocation (adopted as an `Expert`
    /// loan) with the write extents of every slot and the table allocations; `peer` enables PEER
    /// quarantine. Warm start: the first `T` distinct of `seed` fill slots `0..` (fill generation
    /// 1; their fills are in [`Self::take_fills`], the device part copies them and publishes
    /// them all with [`Self::publish_warm`] before the first window), so window 1 sees exactly
    /// what D1's static plan would hold. A `frozen` tail publishes but never admits.
    pub fn new(
        config: EpUnitTailConfig,
        frozen: bool,
        priority: HashMap<UnitKey, u32>,
        peer: bool,
        pool: TailPool,
        seed: &[UnitKey],
    ) -> Result<Self, CacheError> {
        let slots = config.tail_units;
        let admission = TailAdmission {
            peer,
            boundary_every: config.boundary_every,
            priority,
            heat: HashMap::new(),
        };
        let policy = PolicyConfig {
            max_windows: RETIRE_MAX_WINDOWS,
            fill_group_cap: FILL_GROUP_CAP,
            use_log: pool.units.max(1),
            completions: 2 * slots + 16,
            parked: slots.max(1),
            consumers: 1,
            queues: 1,
            ..PolicyConfig::default()
        };
        let init = PoolInit {
            block: pool.block,
            spans: pool.spans,
            tables: pool.tables,
            seed: seed
                .iter()
                .map(|&unit| (unit_key(unit), admission.rank_of(unit)))
                .collect(),
        };
        let mut core = ExpertCachePolicy::new_with_modes(
            policy,
            vec![(
                PoolConfig {
                    slots,
                    fills_per_window: config.fill_per_step,
                },
                init,
            )],
            &[PoolMode { peer }],
        )?;
        if frozen {
            core.freeze();
        }
        Ok(Self {
            config,
            peer,
            core,
            admission,
            retirements: 0,
            refusals: 0,
            first_refusal: None,
        })
    }

    pub fn config(&self) -> &EpUnitTailConfig {
        &self.config
    }

    /// The counters: the core's, with the tail's own retirement count.
    pub fn stats(&self) -> TailStats {
        let s = self.core.stats();
        TailStats {
            windows: s.windows,
            retirements: self.retirements,
            core_reads: s.core_reads,
            core_distinct: s.core_distinct,
            tail_reads: s.tail_reads,
            tail_distinct: s.tail_distinct,
            host_reads: s.host_reads,
            host_distinct: s.host_distinct,
            fills_issued: s.fills_issued,
            fills_published: s.fills_published,
            evictions: s.evictions,
            skipped_full: s.skipped_full,
            stale_completions: s.stale_completions,
            quarantined: s.quarantined,
        }
    }

    pub fn slots(&self) -> usize {
        self.core.slots(POOL).unwrap_or(0)
    }

    /// How many core calls were refused (never expected; the device part freezes the tail on the
    /// first error) and the first refusal's message.
    pub fn refusals(&self) -> (u64, Option<&str>) {
        (self.refusals, self.first_refusal.as_deref())
    }

    /// The retirement index the next retirement will have (`stats().retirements + 1`): the
    /// `at` of the bank's peer fence and the epoch its generation is sealed under.
    pub fn next_retirement(&self) -> u64 {
        self.retirements + 1
    }

    /// Stop admitting: retirements still publish the completed fills ([`Self::new`]'s `frozen`).
    pub fn freeze(&mut self) {
        self.core.freeze();
    }

    /// The slot that serves `unit` now (`Live` only).
    pub fn live_slot(&self, unit: UnitKey) -> Option<usize> {
        match self.core.lookup(unit_key(unit)) {
            Lookup::Published { slot, .. } => Some(slot as usize),
            Lookup::Absent | Lookup::Pending { .. } => None,
        }
    }

    /// Slots in a phase (diagnostics, tests).
    pub fn count(&self, phase: SlotPhase) -> usize {
        let phase = match phase {
            SlotPhase::Free => CorePhase::Free,
            SlotPhase::Filling => CorePhase::Filling,
            SlotPhase::Live => CorePhase::Live,
            SlotPhase::Quarantined => CorePhase::Withdrawn,
        };
        self.core.count(POOL, phase)
    }

    /// The unit a slot holds while it is `Filling` or `Live`. `None` for a `Free` or
    /// `Quarantined` slot (the policy forgets the unit of an evicted slot; the bank still records
    /// it as the occupant, [`Self::occupied`]).
    pub fn slot_unit(&self, slot: usize) -> Option<UnitKey> {
        self.core
            .slot_key(POOL, slot as u32)
            .map(|key| key_unit(&key))
    }

    /// Generation of a slot (bumped by every fill that reuses it): the cell's fill generation.
    pub fn slot_generation(&self, slot: usize) -> u64 {
        self.core
            .slot_generation(POOL, slot as u32)
            .expect("a slot of the tail pool")
    }

    pub fn slot_phase(&self, slot: usize) -> SlotPhase {
        match self
            .core
            .slot_phase(POOL, slot as u32)
            .expect("a slot of the tail pool")
        {
            CorePhase::Free => SlotPhase::Free,
            CorePhase::Filling => SlotPhase::Filling,
            CorePhase::Live => SlotPhase::Live,
            CorePhase::Withdrawn => SlotPhase::Quarantined,
        }
    }

    /// The retirement a `Quarantined` slot was evicted at: the first generation epoch that no
    /// longer names it (the bank's `excluded_from_epoch`, retirement `j` seals epoch `j`).
    pub fn slot_evicted_at(&self, slot: usize) -> Option<u64> {
        self.core.bank(POOL)?.excluded_from_epoch(slot as u32)
    }

    /// The occupied slots as `(slot, occupant, fill_generation)`, ascending by slot: the bank's
    /// view (`Resident`, `Evicting` and issued `Filling` slots).
    pub fn occupied(&self) -> Vec<(u32, u64, u64)> {
        self.core
            .bank(POOL)
            .map_or_else(Vec::new, SlotBank::occupied)
    }

    /// The pool's bank (introspection: reference counts, live generations, quarantine).
    #[cfg(test)]
    pub(crate) fn bank(&self) -> Option<&SlotBank> {
        self.core.bank(POOL)
    }

    /// The core itself (the race tests register a stage reader in its window book).
    #[cfg(test)]
    pub(crate) fn core_mut(&mut self) -> &mut ExpertCachePolicy {
        &mut self.core
    }

    fn refuse(&mut self, error: CacheError) -> CacheError {
        self.refusals += 1;
        if self.first_refusal.is_none() {
            self.first_refusal = Some(error.to_string());
        }
        error
    }

    /// The fills decided and not yet taken, in the core's emission order (fills into free slots
    /// in candidate order, then the fills into victim slots in victim order).
    pub fn take_fills(&mut self) -> Vec<FillOp> {
        self.core
            .take_fills()
            .map(|op| FillOp {
                slot: op.slot as usize,
                unit: key_unit(&op.key),
                generation: op.fill_generation,
                ticket: op.ticket,
            })
            .collect()
    }

    /// A retirement that fails before its fills are issued or deferred: the decided fills that
    /// were not taken yet will never be issued, so the bank is told ([`Self::abandon_fills`]).
    pub fn abandon_decided(&mut self) {
        let tickets: Vec<FillTicket> = self.core.take_fills().map(|op| op.ticket).collect();
        self.abandon_fills(tickets);
    }

    /// The tables' write witness, while the pool's table backing is writable (the published
    /// generation retired, the next not yet sealed).
    pub fn open_tables(&self) -> Result<TablesOpen, CacheError> {
        if self.core.backing(POOL).is_some() {
            Ok(TablesOpen(()))
        } else {
            Err(CacheError::GenerationState {
                pool: POOL,
                state: "without a writable table backing",
            })
        }
    }

    /// Retire the published generation unconditionally, whether or not anything is pending (a
    /// borrow's suspend, a rewrite between windows, the release): the last-read fence is
    /// recorded on `fence` and waited, the table backing is writable again
    /// ([`Self::open_tables`]). `Ok(false)`: it already was retired.
    pub fn retire_now(&mut self, fence: &mut dyn FenceBackend) -> Result<bool, CacheError> {
        self.core
            .retire_generation_now(POOL, fence, None)
            .map_err(|error| self.refuse(error))
    }

    /// Warm start, step 1: retire the empty epoch-0 generation, stage the seeds' completions
    /// (`done`, every seed copy finished) and open the tables for the warm images.
    pub fn publish_warm(
        &mut self,
        fence: &mut dyn FenceBackend,
        done: Vec<FillComplete>,
    ) -> Result<TablesOpen, CacheError> {
        let staged = (|| {
            self.core.retire_generation_now(POOL, fence, None)?;
            for token in done {
                self.core.deliver_completion(POOL, token)?;
            }
            self.core.plan_group(0, (0, 0))?;
            self.core.check_seeded(POOL)
        })();
        staged.map_err(|error| self.refuse(error))?;
        self.open_tables()
    }

    /// Warm start, step 2: seal the staged seeds over the warm tables as generation
    /// `{ epoch 0, sub 1 }` (the core sealed the empty `{ 0, 0 }`), so retirement `j` seals
    /// epoch `j`.
    pub fn seal_warm(&mut self, tables: TablesOpen) -> Result<GenerationId, CacheError> {
        drop(tables);
        self.core
            .seal_as(POOL, GenerationId { epoch: 0, sub: 1 })
            .map_err(|error| self.refuse(error))
    }

    /// Seal the staged membership again as `{ epoch, sub + 1 }` of the last sealed generation
    /// (a borrow resume, a rewrite between windows): the tables were rewritten through
    /// [`TablesOpen`] and the retirement clock does not move.
    pub fn reseal(&mut self, tables: TablesOpen) -> Result<GenerationId, CacheError> {
        drop(tables);
        let Some(last) = self.core.last_sealed(POOL) else {
            let error = CacheError::GenerationState {
                pool: POOL,
                state: "never sealed",
            };
            return Err(self.refuse(error));
        };
        self.core
            .seal_as(
                POOL,
                GenerationId {
                    epoch: last.epoch,
                    sub: last.sub + 1,
                },
            )
            .map_err(|error| self.refuse(error))
    }

    /// Retire one GROUP of `windows` consecutive decode windows (the retirement cadence,
    /// [`RetireCadence`]): the first half of a retirement. The published generation is fenced on
    /// `fence` and retired (PEER mode: `peers = Some((board, owner rank))` asks the board's
    /// `released_through` exactly once, at this retirement's index), then the group is judged:
    /// `window` holds the distinct non-core units any window of the group read (reads summed),
    /// judged against the slots live when the group started; `core` = (distinct, slot reads) the
    /// static core served; `completed` = the completion tokens of the fills whose copy finished
    /// since the last retirement (drained once the retirement succeeded; a token that does not
    /// name a `Filling` slot at its fill generation is ignored and counted stale). A unit read
    /// anywhere in the group is never a victim. The fill budget is `F x windows` capped at
    /// `F x `[`FILL_GROUP_CAP`]. In PEER mode a slot evicted at a retirement `k` every peer
    /// acknowledged by a retirement before this one is free again from this retirement on.
    ///
    /// Returns the table changes (`publish`, `unpublish`); the tables are open
    /// ([`Self::open_tables`]) until [`Self::finish_retirement`] seals the next generation over
    /// them and hands out the fills. An error before the plan leaves `completed` untouched.
    pub fn retire_windows(
        &mut self,
        fence: &mut dyn FenceBackend,
        peers: Option<(&dyn PeerReaderFence, usize)>,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &mut Vec<FillComplete>,
        windows: u32,
    ) -> Result<Retire, CacheError> {
        let at = self.retirements + 1;
        let peers = peers.map(|(fence, owner)| (fence, owner, at));
        self.core
            .retire_generation_now(POOL, fence, peers)
            .map_err(|error| self.refuse(error))?;
        self.retirements = at;
        let planned = (|| {
            for done in completed.drain(..) {
                self.core.deliver_completion(POOL, done)?;
            }
            self.core.begin_window()?;
            for used in window {
                self.core.record_use(CoreUse {
                    key: unit_key(used.unit),
                    pool: POOL,
                    reads: used.reads,
                    rank: self.admission.rank_of(used.unit),
                })?;
            }
            self.core.end_window(0)?;
            self.core
                .plan_group_with(windows.max(1), core, &mut self.admission)
        })();
        planned.map_err(|error| self.refuse(error))?;
        Ok(Retire {
            publish: self
                .core
                .published()
                .iter()
                .map(|p| (key_unit(&p.key), p.slot as usize))
                .collect(),
            unpublish: self
                .core
                .withdrawn()
                .iter()
                .map(|w| key_unit(&w.key))
                .collect(),
            fills: Vec::new(),
        })
    }

    /// The second half of a retirement, after the tables were rewritten and uploaded: reclaim
    /// the slots the retirement frees and start the fills into them, seal the next generation
    /// (epoch = this retirement's index) over the written tables, and hand the decided fills to
    /// `retire.fills`.
    pub fn finish_retirement(
        &mut self,
        tables: TablesOpen,
        retire: &mut Retire,
    ) -> Result<GenerationId, CacheError> {
        drop(tables);
        let sealed = (|| {
            self.core.finish_plan_with(POOL, &mut self.admission)?;
            self.core.seal_as(
                POOL,
                GenerationId {
                    epoch: self.retirements,
                    sub: 0,
                },
            )
        })();
        let id = sealed.map_err(|error| self.refuse(error))?;
        retire.fills = self.take_fills();
        Ok(id)
    }

    /// Enqueue the copies of `tickets` (the callback writes only `ticket.write_spans()`) and
    /// record ONE fence on `copy` after the last. Every error hands the tickets back.
    #[allow(clippy::type_complexity)]
    pub fn issue_fills(
        &mut self,
        tickets: Vec<FillTicket>,
        copy: &mut dyn FenceBackend,
        enqueue: &mut dyn FnMut(&FillTicket) -> Result<(), String>,
    ) -> Result<PendingFills, (Vec<FillTicket>, FenceFailure)> {
        self.core.issue_fills(POOL, tickets, copy, enqueue)
    }

    /// Tickets that will never be issued (a frozen or released tail drops its deferred fills):
    /// their slots stay `Filling` and are never restored, but the bank no longer counts them as
    /// pending, so a scratch loan and the close accept them.
    pub fn abandon_fills(&mut self, tickets: Vec<FillTicket>) {
        self.core.abandon_fills(POOL, tickets);
    }

    /// Poll (`wait == false`) or wait for one batch of issued fills.
    pub fn complete_fills(
        &mut self,
        batch: PendingFills,
        copy: &mut dyn FenceBackend,
        wait: bool,
    ) -> Result<Vec<FillComplete>, FillsNotDone> {
        self.core.complete_fills(POOL, batch, copy, wait)
    }

    /// Device-wide quiescence of the bank's issued fills ([`SlotBank::quiesce`]).
    pub fn quiesce(&mut self, device: &mut dyn FenceBackend) -> Result<Quiesced, FenceFailure> {
        self.core.quiesce(POOL, device)
    }

    /// Lend the whole pool as scratch ([`SlotBank::lend_scratch`]): refused while any generation
    /// is live ([`Self::retire_now`] first).
    pub fn lend_scratch(&mut self, q: Quiesced) -> Result<ScratchLoan, CacheError> {
        self.core.lend_scratch(POOL, q)
    }

    /// Re-copy every occupied slot after a scratch epoch ([`SlotBank::issue_restore`]): the
    /// callback runs once per occupied slot, ascending, then one fence is recorded on `copy`.
    pub fn issue_restore(
        &mut self,
        loan: &ScratchLoan,
        copy: &mut dyn FenceBackend,
        enqueue: &mut dyn FnMut(u32, u64, &[Binding]) -> Result<(), String>,
    ) -> Result<PendingRestore, FenceFailure> {
        self.core.issue_restore(POOL, loan, copy, enqueue)
    }

    /// Wait the restore fence and end the scratch loan ([`SlotBank::end_scratch`]); on error the
    /// loan and the restore come back.
    #[allow(clippy::type_complexity)]
    pub fn end_scratch(
        &mut self,
        loan: ScratchLoan,
        restore: PendingRestore,
        copy: &mut dyn FenceBackend,
    ) -> Result<(), (ScratchLoan, PendingRestore, ResidencyError)> {
        self.core.end_scratch(POOL, loan, restore, copy)
    }

    /// Hand every cell and the bank back: the pool block and its reclaim token, or the refusal
    /// (the pool stays as it was). Needs no live generation ([`Self::retire_now`]), every
    /// decided fill issued or abandoned and a current `q`.
    pub fn close(
        &mut self,
        q: Quiesced,
    ) -> Result<(LentBlock<Expert>, BlockReclaimed), CacheError> {
        self.core.close_pool(POOL, q)
    }
}

/// One change of an owner's slot publications, as the peers apply it to their full-table images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerDelta {
    pub unit: UnitKey,
    /// `Some([gate_up, down])`: the unit's byte copy in the owner's slot is complete and a full
    /// table may name it; `None`: the owner's tables went back to the unit's canonical entries.
    pub slot: Option<[u64; 2]>,
    /// The owner's canonical `[gate_up, down]` entries of the unit (a peer follows the owner only
    /// where its own full-table entries are exactly these).
    pub canon: [u64; 2],
}

/// Every delta a rank took from the board at one retirement, and what it covers.
#[derive(Debug)]
pub struct PeerTake {
    /// By owner, then post order.
    pub deltas: Vec<(usize, PeerDelta)>,
    /// Per owner: the newest owner retirement whose posts this take covers.
    applied: Vec<Option<u64>>,
}

impl PeerTake {
    /// The newest owner retirement any covered post has.
    pub fn newest(&self) -> Option<u64> {
        self.applied.iter().flatten().copied().max()
    }
}

#[derive(Debug)]
struct Post {
    k: u64,
    deltas: Vec<PeerDelta>,
}

/// One reader's acknowledgements of one owner: the cutoff already in force and the entries whose
/// retirement index has not passed yet (both `applied` and `j` strictly increase).
#[derive(Debug, Default)]
struct AckQueue {
    base: u64,
    pending: VecDeque<(u64, u64)>,
}

#[derive(Debug)]
struct BoardState {
    /// Per owner: the newest retirement posted (an empty post counts).
    posted: Vec<Option<u64>>,
    /// Per owner: the posts with deltas that some live reader has not taken yet.
    posts: Vec<VecDeque<Post>>,
    /// `[reader][owner]`: the lowest owner retirement the reader has not taken.
    next: Vec<Vec<u64>>,
    /// `[reader][owner]`.
    acks: Vec<Vec<AckQueue>>,
    /// A frozen or released reader takes nothing more (and never acknowledges again): it no
    /// longer holds posts back.
    detached: Vec<bool>,
}

/// The cross-rank state of the peer board (module docs): what each owner posted, what each
/// rank took, and the acknowledgements the owners release quarantined slots on; plus the pin
/// section of the tail ([`pins`]: the retirement-stamped pin deltas every rank applies under
/// the retirement lag). One board per process and EP mesh, shared by every rank thread; every
/// call is a short critical section.
#[derive(Debug)]
pub struct PeerBoard {
    ranks: usize,
    state: Mutex<BoardState>,
    pins: pins::PinSection,
}

impl PeerBoard {
    pub fn new(ranks: usize) -> Self {
        Self {
            ranks,
            state: Mutex::new(BoardState {
                posted: vec![None; ranks],
                posts: (0..ranks).map(|_| VecDeque::new()).collect(),
                next: vec![vec![0; ranks]; ranks],
                acks: (0..ranks)
                    .map(|_| (0..ranks).map(|_| AckQueue::default()).collect())
                    .collect(),
                detached: vec![false; ranks],
            }),
            pins: pins::PinSection::new(ranks),
        }
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    fn lock(&self) -> MutexGuard<'_, BoardState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Owner `owner` posts the deltas of its retirement `k` (every retirement posts, an empty
    /// one included; `k` strictly increases per owner). The warm start posts as `k = 0`.
    pub fn post(&self, owner: usize, k: u64, deltas: Vec<PeerDelta>) {
        let mut state = self.lock();
        assert!(
            state.posted[owner].is_none_or(|last| k > last),
            "owner {owner} posted retirement {k} after {:?}",
            state.posted[owner]
        );
        state.posted[owner] = Some(k);
        if !deltas.is_empty() {
            state.posts[owner].push_back(Post { k, deltas });
        }
    }

    /// Every delta of every OTHER rank's posts that `reader` has not taken yet, by owner then
    /// post order, and what the take covers (for [`Self::ack`]).
    pub fn take(&self, reader: usize) -> PeerTake {
        let mut guard = self.lock();
        let state = &mut *guard;
        let mut deltas = Vec::new();
        let mut applied = vec![None; self.ranks];
        for owner in (0..self.ranks).filter(|&owner| owner != reader) {
            let Some(top) = state.posted[owner] else {
                continue;
            };
            let from = state.next[reader][owner];
            for post in state.posts[owner].iter().filter(|post| post.k >= from) {
                deltas.extend(post.deltas.iter().map(|delta| (owner, *delta)));
            }
            state.next[reader][owner] = top + 1;
            applied[owner] = Some(top);
            Self::prune(state, owner, self.ranks);
        }
        PeerTake { deltas, applied }
    }

    /// Drop the posts of `owner` that every live reader took.
    fn prune(state: &mut BoardState, owner: usize, ranks: usize) {
        let low = (0..ranks)
            .filter(|&reader| reader != owner && !state.detached[reader])
            .map(|reader| state.next[reader][owner])
            .min()
            .unwrap_or(u64::MAX);
        while state.posts[owner].front().is_some_and(|post| post.k < low) {
            state.posts[owner].pop_front();
        }
    }

    /// `reader` applied `take` and ENQUEUED the table uploads it caused at its retirement `j`.
    /// Records, per owner, `(newest owner retirement the take covers, j)`; a take that covers
    /// nothing newer than the last entry is not recorded (it would only carry a later `j`).
    pub fn ack(&self, reader: usize, j: u64, take: &PeerTake) {
        let mut state = self.lock();
        for owner in 0..self.ranks {
            let Some(top) = take.applied[owner] else {
                continue;
            };
            let queue = &mut state.acks[reader][owner];
            let last = queue
                .pending
                .back()
                .map_or(queue.base, |&(applied, _)| applied);
            if top > last {
                queue.pending.push_back((top, j));
            }
        }
    }

    /// The newest eviction retirement `k` of `owner` that EVERY peer acknowledged by a
    /// retirement `j < m`: a slot `owner` evicted at `k' <= k` may be refilled at its retirement
    /// `m`. `0` while some peer has no such ack (evictions are at `k >= 1`); `u64::MAX` without a
    /// peer. Entries with `j >= m` are ignored (the owner's retirement `m` can only rely on the
    /// uploads enqueued by retirements before `m`); `m` must not decrease between calls.
    pub fn released_through(&self, owner: usize, m: u64) -> u64 {
        let mut state = self.lock();
        let mut through = u64::MAX;
        for reader in (0..self.ranks).filter(|&reader| reader != owner) {
            let queue = &mut state.acks[reader][owner];
            while let Some(&(applied, j)) = queue.pending.front() {
                if j >= m {
                    break;
                }
                queue.base = applied;
                queue.pending.pop_front();
            }
            through = through.min(queue.base);
        }
        through
    }

    /// `reader` took its last take (its tail froze or was released): it no longer holds posts
    /// back. It also never acknowledges again, so owners keep every slot it may still name
    /// quarantined.
    pub fn detach(&self, reader: usize) {
        let mut state = self.lock();
        state.detached[reader] = true;
        for owner in (0..self.ranks).filter(|&owner| owner != reader) {
            Self::prune(&mut state, owner, self.ranks);
        }
    }

    /// Posts of `owner` still held (diagnostics, tests).
    pub fn kept_posts(&self, owner: usize) -> usize {
        self.lock().posts[owner].len()
    }
}

// SAFETY: `released_through(owner, m)` returns `k` only when EVERY peer `r` of `owner` recorded
// an ack `(applied >= k', j)` for every eviction retirement `k' <= k` with `j < m`
// ([`PeerBoard::released_through`]): the peer applied the owner's unpublication posts through
// `k'` and enqueued the table uploads they caused on its compute stream at its retirement
// `j <= m - 1`, ahead of every kernel of its window `j + 1`; the owner's retirement `m` follows
// the cross-rank row exchange that consumed those kernels' outputs (module docs, "Why a refill
// never changes bytes a peer can still read as another unit"). A peer that froze, detached or
// never acknowledged leaves its `base` at the old value, so the minimum over the peers never
// covers an unacknowledged eviction: fail closed.
unsafe impl PeerReaderFence for PeerBoard {
    fn released_through(&self, owner: usize, at: u64) -> u64 {
        PeerBoard::released_through(self, owner, at)
    }
}

#[cfg(test)]
mod tests {
    use super::rig::Rig;
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn retire_ids_parses_a_count_and_defaults_to_4096() {
        assert_eq!(parse_retire_ids(None), Ok(4096));
        assert_eq!(parse_retire_ids(Some(" ")), Ok(4096));
        assert_eq!(parse_retire_ids(Some("0")), Ok(0));
        assert_eq!(parse_retire_ids(Some("12000")), Ok(12000));
        assert!(parse_retire_ids(Some("-5")).is_err());
        assert!(parse_retire_ids(Some("4k")).is_err());
    }

    /// Feed a routed-id sequence to a cadence; the window indices (1-based) it retires at and
    /// the group sizes.
    fn cadence_run(cadence: &mut RetireCadence, routed: &[u64]) -> Vec<(usize, u32)> {
        routed
            .iter()
            .enumerate()
            .filter_map(|(i, &ids)| cadence.window_end(ids).map(|n| (i + 1, n)))
            .collect()
    }

    #[test]
    fn cadence_retires_on_routed_ids_or_eight_windows_or_log_headroom() {
        let log = 1 << 18;
        // 4096 routed ids: 1280 per window (C1 MTP-sized) retires at the 4th window.
        let mut c = RetireCadence::new(4096, log);
        assert_eq!(cadence_run(&mut c, &[1280; 9]), [(4, 4), (8, 4)]);
        assert_eq!(c.pending(), 1);
        // Tiny windows: the 8-window cap.
        let mut c = RetireCadence::new(4096, log);
        assert_eq!(cadence_run(&mut c, &[320; 17]), [(8, 8), (16, 8)]);
        // Big windows retire at once (a prefill or a C8 window of 2560 ids x 2).
        let mut c = RetireCadence::new(4096, log);
        assert_eq!(cadence_run(&mut c, &[5000, 5000]), [(1, 1), (2, 1)]);
        let mut c = RetireCadence::new(4096, log);
        assert_eq!(cadence_run(&mut c, &[2560; 4]), [(2, 2), (4, 2)]);
        // 0 = every window.
        let mut c = RetireCadence::new(0, log);
        assert_eq!(cadence_run(&mut c, &[1, 1, 1]), [(1, 1), (2, 1), (3, 1)]);
        // Log headroom (capacity 450 ids): retire when routed + the largest window seen would
        // pass the capacity: after 4 windows of 100 (400 + 100 > 450).
        let mut c = RetireCadence::new(1_000_000, 450);
        assert_eq!(cadence_run(&mut c, &[100, 100, 100, 100]), [(4, 4)]);
        // The largest window seen bounds the headroom of every later group: after the 150 window
        // a group of 110s retires at 330 + 150 > 450 (it would not with a largest of 110).
        let mut c = RetireCadence::new(1_000_000, 450);
        assert_eq!(
            cadence_run(&mut c, &[150, 100, 100, 110, 110, 110]),
            [(3, 3), (6, 3)]
        );
    }

    #[test]
    fn cadence_decisions_depend_on_the_routed_sequence_only() {
        // Two ranks log different numbers of ids (a frozen / overflowed / foreign-expert rank
        // logs fewer), but the cadence never sees a logged count: both decide identically for
        // every routed sequence.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for retire_ids in [0u64, 1, 700, 4096, 20_000] {
            for log in [300u64, 4096, 1 << 18] {
                let (mut a, mut b) = (
                    RetireCadence::new(retire_ids, log),
                    RetireCadence::new(retire_ids, log),
                );
                for _ in 0..500 {
                    let routed = next() % 3000;
                    assert_eq!(a.window_end(routed), b.window_end(routed));
                    assert_eq!(a.pending(), b.pending());
                }
            }
        }
    }

    #[test]
    fn a_group_never_exceeds_eight_windows_nor_the_log() {
        let log = 5_000u64;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut c = RetireCadence::new(1 << 40, log);
        let (mut total, mut windows, mut largest) = (0u64, 0u32, 0u64);
        for _ in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let routed = state % 900;
            total += routed;
            windows += 1;
            largest = largest.max(routed);
            match c.window_end(routed) {
                Some(n) => {
                    assert_eq!(n, windows);
                    assert!(n <= RETIRE_MAX_WINDOWS);
                    total = 0;
                    windows = 0;
                }
                // The window just kept pending could be followed by the largest window seen
                // without passing the log capacity.
                None => assert!(total + largest <= log, "{total} + {largest}"),
            }
        }
    }

    #[test]
    fn a_group_retirement_never_evicts_a_unit_any_window_of_the_group_used() {
        for windows in [1u32, 2, 4, 8] {
            let mut p = policy(4, 2, 1);
            p.seed(&[(0, 1), (0, 2), (0, 3), (0, 4)]);
            // The merged window of the group: slots (0,1) and (0,2) hit, five misses.
            let uses = [
                use1(0, 1, 2),
                use1(0, 2, 1),
                use1(1, 9, 7),
                use1(1, 8, 6),
                use1(1, 7, 5),
                use1(1, 6, 4),
                use1(1, 5, 3),
            ];
            let r = p.retire_windows((0, 0), &uses, &[], windows);
            assert!(
                !r.unpublish.contains(&(0, 1)) && !r.unpublish.contains(&(0, 2)),
                "windows {windows}: {:?}",
                r.unpublish
            );
            assert_eq!(p.live_slot((0, 1)), Some(0));
            assert_eq!(p.live_slot((0, 2)), Some(1));
            // Only the two unused slots can be victims: at most 2 fills, whatever the budget.
            assert_eq!(
                r.fills.len(),
                2.min(2 * (windows as usize).min(4)),
                "windows {windows}"
            );
            let s = p.stats();
            assert_eq!((s.windows, s.retirements), (windows as u64, 1));
        }
    }

    #[test]
    fn the_fill_budget_is_f_per_window_capped_at_four_f() {
        for (windows, expect) in [(1u32, 2usize), (2, 4), (3, 6), (4, 8), (5, 8), (8, 8)] {
            let mut p = policy(64, 2, 1);
            let uses: Vec<WindowUse> = (0..40u16).map(|e| use1(0, e, 1)).collect();
            let r = p.retire_windows((0, 0), &uses, &[], windows);
            assert_eq!(r.fills.len(), expect, "windows {windows}");
        }
        // In-flight fills count against the group's budget.
        let mut p = policy(64, 2, 1);
        let first: Vec<WindowUse> = (0..40u16).map(|e| use1(0, e, 1)).collect();
        assert_eq!(p.retire_windows((0, 0), &first, &[], 4).fills.len(), 8);
        let second: Vec<WindowUse> = (40..64u16).map(|e| use1(0, e, 1)).collect();
        assert_eq!(p.retire_windows((0, 0), &second, &[], 4).fills.len(), 0);
    }

    #[test]
    fn boundary_mode_admits_when_a_group_crosses_a_multiple_of_n_windows() {
        // b=4: windows 1..=3 admit nothing, the group that reaches window 4..7 admits.
        let mut p = policy(8, 4, 4);
        let uses = [use1(0, 5, 3)];
        assert!(p.retire_windows((0, 0), &uses, &[], 3).fills.is_empty()); // windows 0 -> 3
        assert_eq!(p.retire_windows((0, 0), &uses, &[], 2).fills.len(), 1); // 3 -> 5 crosses 4
        assert!(p.retire_windows((0, 0), &uses, &[], 2).fills.is_empty()); // 5 -> 7
                                                                           // One window per retirement is the old rule: only every 4th window admits.
        let mut q = policy(8, 4, 4);
        let admitted: Vec<bool> = (0..8)
            .map(|_| !q.retire((0, 0), &[use1(0, 5, 3)], &[]).fills.is_empty())
            .collect();
        assert_eq!(
            admitted,
            [false, false, false, true, false, false, false, false]
        );
    }

    #[test]
    fn stage_gap_parses_a_unit_count_and_defaults_to_eight() {
        assert_eq!(parse_stage_gap(None), Ok(8));
        assert_eq!(parse_stage_gap(Some("")), Ok(8));
        assert_eq!(parse_stage_gap(Some(" 0 ")), Ok(0));
        assert_eq!(parse_stage_gap(Some("32")), Ok(32));
        assert!(parse_stage_gap(Some("-1")).is_err());
        assert!(parse_stage_gap(Some("eight")).is_err());
    }

    fn cfg(t: usize, f: usize, b: u32) -> EpUnitTailConfig {
        EpUnitTailConfig {
            tail_units: t,
            fill_per_step: f,
            boundary_every: b,
        }
    }

    fn policy_priority() -> HashMap<UnitKey, u32> {
        // Ranking position = layer * 1000 + expert.
        let mut priority = HashMap::new();
        for layer in 0..4u16 {
            for e in 0..64u16 {
                priority.insert((layer, e), layer as u32 * 1000 + e as u32);
            }
        }
        priority
    }

    /// A policy over rig slots (the pre-bank tests drove `TailPolicy::new` directly; the rig
    /// owns the capabilities the retirement protocol needs, `super::rig`).
    fn policy(t: usize, f: usize, b: u32) -> Rig {
        Rig::new(cfg(t, f, b), false, policy_priority())
    }

    /// [`policy`] with PEER quarantine (the old `enable_peer_quarantine()`).
    fn policy_peer(t: usize, f: usize, b: u32) -> Rig {
        Rig::with_peer(cfg(t, f, b), false, policy_priority(), true)
    }

    fn use1(layer: u16, expert: u16, reads: u32) -> WindowUse {
        WindowUse {
            unit: (layer, expert),
            reads,
        }
    }

    #[test]
    fn resolve_defaults_when_unset() {
        let default = resolve_ep_unit_tail(None).unwrap().unwrap();
        assert_eq!(default.tail_units, EP_UNIT_TAIL_DEFAULT_SLOTS);
        assert_eq!(default.fill_per_step, EP_UNIT_TAIL_DEFAULT_FILLS);
        assert_eq!(default.boundary_every, 1);
        assert_eq!(resolve_ep_unit_tail(Some("")), Ok(None));
        assert_eq!(resolve_ep_unit_tail(Some("0")), Ok(None));
        assert_eq!(resolve_ep_unit_tail(Some("off")), Ok(None));
        assert_eq!(
            resolve_ep_unit_tail(Some("512,8"))
                .unwrap()
                .unwrap()
                .tail_units,
            512
        );
        assert!(resolve_ep_unit_tail(Some("x")).is_err());
    }

    #[test]
    fn parse_forms() {
        assert_eq!(parse_ep_unit_tail(None), Ok(None));
        assert_eq!(parse_ep_unit_tail(Some("")), Ok(None));
        assert_eq!(parse_ep_unit_tail(Some("0")), Ok(None));
        assert_eq!(parse_ep_unit_tail(Some("off")), Ok(None));
        assert_eq!(
            parse_ep_unit_tail(Some("1024,32")),
            Ok(Some(cfg(1024, 32, 1)))
        );
        assert_eq!(
            parse_ep_unit_tail(Some(" 512 , 16 , b=4 ")),
            Ok(Some(cfg(512, 16, 4)))
        );
        assert_eq!(
            parse_ep_unit_tail(Some("512,16,boundary=2")),
            Ok(Some(cfg(512, 16, 2)))
        );
        for bad in [
            "x",
            "512",
            "512,0",
            "0,5",
            "5,5,b=0",
            "5,5,x",
            "5,5,b=2,9",
            "-1,2",
        ] {
            assert!(parse_ep_unit_tail(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn budget_delta_rounds_the_pool_up() {
        let unit = 2_611_200u64;
        let c = cfg(1000, 8, 1);
        let bytes = ep_unit_tail_pool_bytes(1000, unit);
        assert_eq!(
            bytes,
            1000 * unit + 2 * crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64
        );
        let delta = ep_unit_tail_budget_delta_mib(&c, unit);
        assert!(delta < 0 && (-delta) as u64 * MIB >= bytes);
        assert!(((-delta) as u64 - 1) * MIB < bytes);
    }

    #[test]
    fn seed_warms_the_first_t_distinct_units_live() {
        let mut p = policy(3, 2, 1);
        let seeds = p.seed(&[(0, 1), (0, 2), (0, 1), (0, 3), (0, 4)]);
        assert_eq!(seeds.len(), 3);
        assert_eq!(
            seeds.iter().map(|f| f.1).collect::<Vec<_>>(),
            [(0, 1), (0, 2), (0, 3)]
        );
        assert_eq!(p.count(SlotPhase::Live), 3);
        assert_eq!(p.live_slot((0, 2)), Some(1));
        assert_eq!(p.live_slot((0, 4)), None);
    }

    #[test]
    fn a_fill_is_invisible_until_published_and_stale_completions_are_ignored() {
        let mut p = policy(2, 4, 1);
        // window 1: miss (1,5) -> a fill into free slot 0.
        let r = p.retire((0, 0), &[use1(1, 5, 3)], &[]);
        assert_eq!(r.fills, vec![(0, (1, 5), 1)]);
        assert!(r.publish.is_empty() && r.unpublish.is_empty());
        assert_eq!(p.live_slot((1, 5)), None, "Filling is not Live");
        // window 2: still a miss (host read); a wrong-generation completion is stale.
        let r = p.retire((0, 0), &[use1(1, 5, 1)], &[(0, 99)]);
        assert!(r.fills.is_empty(), "a pending fill is not admitted twice");
        assert_eq!(p.stats().stale_completions, 1);
        assert_eq!(p.stats().host_reads, 4);
        // window 3: the right completion publishes it; window 3 itself still missed.
        let r = p.retire((0, 0), &[use1(1, 5, 1)], &[(0, 1)]);
        assert_eq!(r.publish, vec![((1, 5), 0)]);
        assert_eq!(p.live_slot((1, 5)), Some(0));
        // window 4: now a tail hit.
        p.retire((0, 0), &[use1(1, 5, 2)], &[]);
        assert_eq!((p.stats().tail_reads, p.stats().tail_distinct), (2, 1));
    }

    #[test]
    fn lru_evicts_the_oldest_unit_not_used_by_the_retiring_window() {
        let mut p = policy(2, 4, 1);
        p.seed(&[(0, 1), (0, 2)]);
        // window 1 uses (0,1) only and misses (0,9): slot 1 ((0,2), unused) is the victim.
        let r = p.retire((0, 0), &[use1(0, 1, 2), use1(0, 9, 5)], &[]);
        assert_eq!(r.unpublish, vec![(0, 2)]);
        assert_eq!(r.fills, vec![(1, (0, 9), 2)]);
        assert_eq!(p.live_slot((0, 2)), None);
        assert_eq!(p.live_slot((0, 1)), Some(0));
        assert_eq!(p.stats().evictions, 1);
    }

    #[test]
    fn units_used_by_the_retiring_window_are_never_victims() {
        let mut p = policy(2, 8, 1);
        p.seed(&[(0, 1), (0, 2)]);
        // both resident units are read; two misses want slots: no victim exists.
        let r = p.retire(
            (0, 0),
            &[use1(0, 1, 1), use1(0, 2, 1), use1(0, 7, 9), use1(0, 8, 8)],
            &[],
        );
        assert!(r.fills.is_empty() && r.unpublish.is_empty());
        assert_eq!(p.stats().skipped_full, 2);
        assert_eq!(p.count(SlotPhase::Live), 2);
    }

    #[test]
    fn fills_per_window_are_bounded_by_f_minus_in_flight() {
        let mut p = policy(8, 2, 1);
        let misses: Vec<_> = (0..5u16).map(|e| use1(2, e, 1)).collect();
        let r = p.retire((0, 0), &misses, &[]);
        assert_eq!(r.fills.len(), 2, "F = 2");
        // Both still in flight: nothing new.
        let r = p.retire((0, 0), &misses, &[]);
        assert!(r.fills.is_empty());
        // One completes: one more may start.
        let first = p.slot_generation(0);
        let r = p.retire((0, 0), &misses, &[(0, first)]);
        assert_eq!(r.publish.len(), 1);
        assert_eq!(r.fills.len(), 1);
    }

    #[test]
    fn candidates_are_ordered_by_reads_then_ranking() {
        let mut p = policy(8, 3, 1);
        let r = p.retire(
            (0, 0),
            &[use1(3, 4, 1), use1(1, 9, 5), use1(0, 7, 5), use1(2, 2, 7)],
            &[],
        );
        let order: Vec<_> = r.fills.iter().map(|f| f.1).collect();
        assert_eq!(order, vec![(2, 2), (0, 7), (1, 9)]);
    }

    #[test]
    fn a_frozen_policy_publishes_but_never_admits() {
        let mut p = policy(2, 4, 1);
        p.freeze();
        let r = p.retire((1, 4), &[use1(0, 3, 6)], &[]);
        assert!(r.fills.is_empty() && r.unpublish.is_empty());
        assert_eq!(p.stats().host_reads, 6);
        assert_eq!(p.stats().core_reads, 4);
    }

    #[test]
    fn boundary_mode_admits_only_on_the_boundary_with_hysteresis() {
        let mut p = policy(1, 2, 2);
        p.seed(&[(0, 1)]);
        // window 1 (not a boundary): heat accrues, no admission.
        let r = p.retire((0, 0), &[use1(0, 9, 3)], &[]);
        assert!(r.fills.is_empty());
        // window 2 (boundary): (0,9) heat 3+3 = 6 vs the incumbent (0,1) heat 0 -> swap.
        let r = p.retire((0, 0), &[use1(0, 9, 3)], &[]);
        assert_eq!(r.fills.len(), 1);
        assert_eq!(r.unpublish, vec![(0, 1)]);
        // A challenger that does not beat the incumbent's heat by 10% is refused: window 1
        // reads the incumbent (heat 10) and the challenger (5); window 2 (the boundary) reads
        // only the challenger (heat 5 + 5 = 10 < 10 x 1.1).
        let mut q = policy(1, 2, 2);
        q.seed(&[(0, 1)]);
        q.retire((0, 0), &[use1(0, 1, 10), use1(0, 9, 5)], &[]);
        let r = q.retire((0, 0), &[use1(0, 9, 5)], &[]);
        assert!(r.fills.is_empty() && r.unpublish.is_empty());
        assert_eq!(q.stats().skipped_full, 1);
    }

    /// The safety invariant of the whole design, over random windows with random copy
    /// latency: applying each retirement's `publish`/`unpublish` to a model of the decode
    /// table, (1) an entry names a slot exactly while the policy holds the unit LIVE in that
    /// slot, (2) no slot is named by two entries, (3) a slot that gets a fill is named by NO
    /// entry once the retirement's table changes are applied (so the copy that follows
    /// cannot overwrite bytes a kernel could still read), (4) a completion is only honoured
    /// for the generation that was issued.
    #[test]
    fn table_entries_never_name_a_slot_that_is_being_filled() {
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (t, f, b) in [(16usize, 4usize, 1u32), (8, 8, 1), (32, 2, 1), (12, 6, 3)] {
            let mut p = policy(t, f, b);
            let seeds = p.seed(&(0..t as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
            // table model: unit -> slot it names (absent = canonical)
            let mut table: HashMap<UnitKey, usize> = seeds.iter().map(|op| (op.1, op.0)).collect();
            // in-flight copies: (slot, generation, windows until it completes)
            let mut flight: Vec<(usize, u64, u32)> = Vec::new();
            let mut honoured = 0u64;
            for window in 0..600u32 {
                // This window reads ~12 units, skewed to a hot set that drifts.
                let drift = (window / 100) as u16 * 7;
                let mut uses: HashMap<UnitKey, u32> = HashMap::new();
                for _ in 0..12 {
                    let hot = rng() % 4 != 0;
                    let layer = (rng() % 3) as u16;
                    let expert = if hot {
                        ((rng() % 20) as u16 + drift) % 64
                    } else {
                        (rng() % 64) as u16
                    };
                    *uses.entry((layer, expert)).or_insert(0) += 1;
                }
                let used: Vec<WindowUse> = uses
                    .into_iter()
                    .map(|(unit, reads)| WindowUse { unit, reads })
                    .collect();
                // Copies whose latency elapsed complete now (maybe out of date).
                let mut completed: Vec<(usize, u64)> = Vec::new();
                flight.retain_mut(|(slot, generation, left)| {
                    if *left == 0 {
                        completed.push((*slot, *generation));
                        false
                    } else {
                        *left -= 1;
                        true
                    }
                });
                if rng() % 11 == 0 {
                    // A bogus completion for a generation that was never issued.
                    completed.push((0, 1 << 40));
                }
                let before = p.stats().fills_published;
                let r = p.retire((0, 0), &used, &completed);
                honoured += p.stats().fills_published - before;
                for (unit, slot) in &r.publish {
                    assert!(table.insert(*unit, *slot).is_none(), "published twice");
                }
                for unit in &r.unpublish {
                    assert!(
                        table.remove(unit).is_some(),
                        "evicted a unit no entry named"
                    );
                }
                // (1) entries == the policy's LIVE slots.
                for (unit, slot) in &table {
                    assert_eq!(p.live_slot(*unit), Some(*slot));
                }
                assert_eq!(table.len(), p.count(SlotPhase::Live));
                // (2) no slot named twice.
                let named: HashSet<usize> = table.values().copied().collect();
                assert_eq!(named.len(), table.len());
                // (3) every slot that is about to be written is named by no entry.
                for fill in &r.fills {
                    assert!(!named.contains(&fill.0), "fill into a published slot");
                    assert_eq!(p.slot_generation(fill.0), fill.2);
                    assert_eq!(p.slot_unit(fill.0), Some(fill.1));
                    flight.push((fill.0, fill.2, (rng() % 4) as u32));
                }
                assert!(p.count(SlotPhase::Filling) <= f.max(1) + t);
            }
            // Tail never exceeds its capacity and the bogus completions were all rejected.
            assert!(p.count(SlotPhase::Live) + p.count(SlotPhase::Filling) <= t);
            assert!(p.stats().stale_completions > 0);
            assert!(honoured > 0, "T={t} F={f}: no fill ever published");
        }
    }

    #[test]
    fn hit_rates_follow_the_counters() {
        let mut p = policy(2, 4, 1);
        p.seed(&[(0, 1)]);
        p.retire((2, 6), &[use1(0, 1, 2), use1(0, 5, 2)], &[]);
        let s = p.stats();
        assert_eq!((s.core_reads, s.tail_reads, s.host_reads), (6, 2, 2));
        assert!((s.slot_hit() - 0.8).abs() < 1e-12);
        assert!((s.distinct_hit() - 0.75).abs() < 1e-12);
    }

    /// FNV-1a fold of `v` into `h`.
    fn fold(h: &mut u64, v: u64) {
        for byte in v.to_le_bytes() {
            *h ^= byte as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Digest of every decision the policy makes over a deterministic random trace: each
    /// retirement's publish / unpublish / fill lists and the final counters. Pinned for the
    /// no-peer policy (no quarantine path) against the digests recorded on the policy before the
    /// peer board (be067917f3). `peer` runs the same trace through the quarantine path with
    /// every peer acknowledging at once.
    fn trace_digest(
        t: usize,
        f: usize,
        b: u32,
        frozen: bool,
        peer: bool,
        seed: u64,
        windows: u32,
    ) -> u64 {
        let mut state = seed | 1;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut priority = HashMap::new();
        for layer in 0..4u16 {
            for e in 0..64u16 {
                priority.insert((layer, e), layer as u32 * 1000 + e as u32);
            }
        }
        let mut p = Rig::with_peer(cfg(t, f, b), frozen, priority, peer);
        let seeds = p.seed(&(0..(t / 2) as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for op in &seeds {
            fold(&mut h, op.0 as u64);
            fold(&mut h, op.2);
        }
        let mut flight: Vec<(usize, u64, u32)> = Vec::new();
        for window in 0..windows {
            let drift = (window / 50) as u16 * 5;
            let mut uses: Vec<WindowUse> = Vec::new();
            for _ in 0..(6 + rng() % 10) {
                let hot = rng() % 4 != 0;
                let layer = (rng() % 3) as u16;
                let expert = if hot {
                    ((rng() % 14) as u16 + drift) % 48
                } else {
                    (rng() % 48) as u16
                };
                let unit = (layer, expert);
                match uses.iter_mut().find(|u| u.unit == unit) {
                    Some(u) => u.reads += 1 + (rng() % 3) as u32,
                    None => uses.push(WindowUse {
                        unit,
                        reads: 1 + (rng() % 3) as u32,
                    }),
                }
            }
            uses.sort_by_key(|u| u.unit);
            let mut completed: Vec<(usize, u64)> = Vec::new();
            flight.retain_mut(|(slot, generation, left)| {
                if *left == 0 {
                    completed.push((*slot, *generation));
                    false
                } else {
                    *left -= 1;
                    true
                }
            });
            if rng() % 13 == 0 {
                completed.push((0, 1 << 40));
            }
            let core = (rng() % 5, rng() % 9);
            let r = if peer {
                p.retire_peer(core, &uses, &completed, |m| m - 1)
            } else {
                p.retire(core, &uses, &completed)
            };
            for (unit, slot) in &r.publish {
                fold(&mut h, 1);
                fold(&mut h, (unit.0 as u64) << 16 | unit.1 as u64);
                fold(&mut h, *slot as u64);
            }
            for unit in &r.unpublish {
                fold(&mut h, 2);
                fold(&mut h, (unit.0 as u64) << 16 | unit.1 as u64);
            }
            for fill in &r.fills {
                fold(&mut h, 3);
                fold(&mut h, (fill.1 .0 as u64) << 16 | fill.1 .1 as u64);
                fold(&mut h, fill.0 as u64);
                fold(&mut h, fill.2);
                flight.push((fill.0, fill.2, (rng() % 4) as u32));
            }
            fold(&mut h, 4);
        }
        let s = p.stats();
        for v in [
            s.windows,
            s.core_reads,
            s.core_distinct,
            s.tail_reads,
            s.tail_distinct,
            s.host_reads,
            s.host_distinct,
            s.fills_issued,
            s.fills_published,
            s.evictions,
            s.skipped_full,
            s.stale_completions,
            p.count(SlotPhase::Free) as u64,
            p.count(SlotPhase::Filling) as u64,
            p.count(SlotPhase::Live) as u64,
        ] {
            fold(&mut h, v);
        }
        h
    }

    /// `(T, F, boundary, frozen, seed, digest)` of the pinned no-peer traces, recorded on the
    /// pre-peer policy (be067917f3).
    const GOLDEN_CASES: [(usize, usize, u32, bool, u64, u64); 6] = [
        (5, 2, 1, false, 0x1234_5678_9abc_def1, 0x65a2_c5b0_4769_3f07),
        (3, 1, 4, false, 0x0fed_cba9_8765_4321, 0x1da9_b1d2_b640_c96c),
        (8, 3, 1, false, 0x9e37_79b9_7f4a_7c15, 0x42c2_177a_5a9d_dcdd),
        (4, 4, 2, false, 0x2545_f491_4f6c_dd1d, 0x4115_3ce7_93b6_946e),
        (6, 2, 1, true, 0x1357_9bdf_0246_8ace, 0x9acd_277b_24e2_cbbd),
        (
            12,
            6,
            3,
            false,
            0xdead_beef_cafe_f00d,
            0x1918_f493_e57b_f459,
        ),
    ];

    /// Without peers (no quarantine) decides exactly like the pre-peer policy: the digests of
    /// publish / unpublish / fill lists and counters over random traces (evictions, boundary
    /// mode, freeze, stale completions) are the ones recorded before the lever existed.
    #[test]
    fn no_peer_policy_decisions_match_the_pre_peer_golden_traces() {
        for (t, f, b, frozen, seed, want) in GOLDEN_CASES {
            let got = trace_digest(t, f, b, frozen, false, seed, 400);
            assert_eq!(
                got, want,
                "T={t} F={f} b={b} frozen={frozen}: no-peer decisions moved ({got:#018x})"
            );
        }
    }

    /// The quarantine path only differs from the no-peer path when a slot is evicted: a trace
    /// whose free slots never drop below the reserve ([`QUARANTINE_WINDOWS`] x the fill
    /// budget, so no miss needs a victim and no reserve eviction happens) decides identically
    /// in both modes.
    #[test]
    fn peer_mode_decides_like_no_peer_while_no_victim_is_needed() {
        for seed in [0x1234_5678_9abc_def1u64, 0x9e37_79b9_7f4a_7c15] {
            // 300 windows x 3 fills leave more than 3 x 3 of the 1000 slots free throughout.
            let off = trace_digest(1000, 3, 1, false, false, seed, 300);
            let on = trace_digest(1000, 3, 1, false, true, seed, 300);
            assert_eq!(off, on);
        }
    }

    /// RESERVE: under saturated misses (more candidates than the budget) the peer per-window
    /// policy keeps issuing `F` fills per retirement once the reserve is built, with the peers
    /// acknowledging one or two retirements late (without the reserve, eviction and refill
    /// alternate: about `F / 2`). Each candidate evicts at most one victim, the reserve (free +
    /// quarantined) never exceeds `RESERVE_RETIREMENTS x` the budget, and when the misses drop
    /// below saturation the reserve drains into fills (the candidate-waits rule again).
    #[test]
    fn peer_reserve_sustains_f_fills_per_window_under_acknowledgement_lag() {
        for lag in [1u64, 2] {
            let (t, f) = (64usize, 4usize);
            let mut p = policy_peer(t, f, 1);
            p.seed(&(0..t as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
            let mut completed: Vec<(usize, u64)> = Vec::new();
            let mut steady = Vec::new();
            let reserve = |p: &Rig| p.count(SlotPhase::Free) + p.count(SlotPhase::Quarantined);
            for w in 0..60u16 {
                // Every window misses 2F fresh units.
                let uses: Vec<WindowUse> = (0..2 * f as u16)
                    .map(|i| use1(1, w * 2 * f as u16 + i, 1))
                    .collect();
                let before = p.stats();
                let r = p.retire_peer((0, 0), &uses, &completed, |m| m.saturating_sub(lag));
                let s = p.stats();
                assert!(
                    s.evictions - before.evictions <= f as u64,
                    "one eviction per candidate"
                );
                assert!(r.fills.len() <= f);
                let held = reserve(&p);
                assert!(
                    held <= RESERVE_RETIREMENTS * f,
                    "lag {lag} w {w}: reserve {held}"
                );
                completed = r.fills.iter().map(|&(slot, _, g)| (slot, g)).collect();
                if w >= 10 {
                    steady.push(r.fills.len());
                }
            }
            assert!(
                steady.iter().all(|&n| n == f),
                "lag {lag}: fills per retirement {steady:?}"
            );
            // The misses drop to one per window: the reserve drains to the new target.
            for w in 0..20u16 {
                let uses = [use1(2, w, 1)];
                let r = p.retire_peer((0, 0), &uses, &completed, |m| m.saturating_sub(lag));
                completed = r.fills.iter().map(|&(slot, _, g)| (slot, g)).collect();
            }
            assert!(
                reserve(&p) <= RESERVE_RETIREMENTS,
                "lag {lag}: reserve {}",
                reserve(&p)
            );
        }
    }

    #[test]
    fn peer_eviction_quarantines_the_slot_until_every_peer_acknowledged() {
        let mut p = policy_peer(2, 4, 1);
        p.seed(&[(0, 1), (0, 2)]);
        // Window 1: a miss with the pool full of live units this window did not use. The victim
        // is unpublished and quarantined; the candidate gets no slot this window.
        let r = p.retire_peer((0, 0), &[use1(0, 9, 3)], &[], |m| {
            assert_eq!(m, 1, "the closure is asked with this retirement's index");
            0
        });
        assert!(r.fills.is_empty() && r.publish.is_empty());
        assert_eq!(r.unpublish.len(), 1);
        assert_eq!(p.count(SlotPhase::Quarantined), 1);
        assert_eq!(p.count(SlotPhase::Free), 0);
        let first = (0..2)
            .find(|&s| p.slot_phase(s) == SlotPhase::Quarantined)
            .expect("one quarantined slot");
        assert_eq!(p.slot_evicted_at(first), Some(1));
        assert_eq!(p.slot_unit(first), None);
        assert_eq!(p.live_slot(r.unpublish[0]), None);
        let s = p.stats();
        assert_eq!(
            (s.evictions, s.quarantined, s.fills_issued, s.skipped_full),
            (1, 1, 0, 0)
        );
        // Window 2: no acknowledgement yet (nothing released). The quarantined slot is NOT a
        // victim and not free, so the miss quarantines the other live slot instead.
        let r = p.retire_peer((0, 0), &[use1(0, 9, 3)], &[], |_| 0);
        assert!(r.fills.is_empty());
        assert_eq!(p.count(SlotPhase::Quarantined), 2);
        assert_eq!(p.slot_evicted_at(first), Some(1));
        // Window 3: eviction 1 is acknowledged (through 1), eviction 2 is not: the first slot is
        // free again and takes the candidate in the same retirement; the second stays out.
        let r = p.retire_peer((0, 0), &[use1(0, 9, 3)], &[], |m| {
            assert_eq!(m, 3);
            1
        });
        assert_eq!(r.fills, vec![(first, (0, 9), 2)]);
        assert!(r.unpublish.is_empty());
        assert_eq!(p.count(SlotPhase::Quarantined), 1);
        assert_eq!(p.slot_phase(first), SlotPhase::Filling);
        // The release at the SAME retirement it is acknowledged; a later one frees the rest.
        let r = p.retire_peer((0, 0), &[], &[], |_| 2);
        assert!(r.fills.is_empty());
        assert_eq!(p.count(SlotPhase::Quarantined), 0);
        assert_eq!(p.count(SlotPhase::Free), 1);
    }

    /// A peer that never acknowledges (frozen, released) must not drain the pool: at most the
    /// reserve target (`RESERVE_RETIREMENTS x` the candidates a retirement considers, never more
    /// than `QUARANTINE_WINDOWS` windows of fills) is ever quarantined; the rest of the tail keeps
    /// serving and the candidates that find no slot are counted as skipped.
    #[test]
    fn a_silent_peer_quarantines_a_bounded_part_of_the_pool() {
        let (t, f) = (24usize, 2usize);
        // Four misses per window, two considered (the budget): the target is 2 x 2.
        let bound = RESERVE_RETIREMENTS * f;
        let mut p = policy_peer(t, f, 1);
        p.seed(&(0..t as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
        for w in 0..80u16 {
            // Every window misses 4 fresh units.
            let uses: Vec<WindowUse> = (0..4).map(|i| use1(1, (w * 4 + i) % 64, 1)).collect();
            let r = p.retire_peer((0, 0), &uses, &[], |_| 0);
            assert!(r.fills.is_empty(), "nothing is ever released");
            assert!(p.count(SlotPhase::Quarantined) <= bound);
        }
        assert_eq!(p.count(SlotPhase::Quarantined), bound);
        assert_eq!(p.count(SlotPhase::Live), t - bound);
        assert!(p.stats().skipped_full > 0);
        assert_eq!(p.stats().quarantined as usize, bound);
    }

    fn delta(layer: u16, expert: u16, slot: Option<[u64; 2]>) -> PeerDelta {
        PeerDelta {
            unit: (layer, expert),
            slot,
            canon: [1000 + expert as u64, 2000 + expert as u64],
        }
    }

    #[test]
    fn board_hands_each_post_to_a_reader_once_in_post_order_and_prunes() {
        let board = PeerBoard::new(2);
        // Nothing posted: nothing taken, nothing to acknowledge.
        let t = board.take(1);
        assert!(t.deltas.is_empty() && t.newest().is_none());
        board.post(0, 0, vec![delta(0, 2, Some([7, 8]))]);
        board.post(0, 1, vec![]);
        board.post(0, 2, vec![delta(0, 2, None), delta(1, 4, Some([9, 10]))]);
        // A rank never takes its own posts.
        assert!(board.take(0).deltas.is_empty());
        let t = board.take(1);
        assert_eq!(
            t.deltas,
            vec![
                (0, delta(0, 2, Some([7, 8]))),
                (0, delta(0, 2, None)),
                (0, delta(1, 4, Some([9, 10]))),
            ]
        );
        assert_eq!(t.newest(), Some(2));
        // Taken once: the next take sees only later posts.
        board.post(0, 3, vec![delta(0, 6, Some([11, 12]))]);
        let t = board.take(1);
        assert_eq!(t.deltas, vec![(0, delta(0, 6, Some([11, 12])))]);
        assert_eq!(t.newest(), Some(3));
        let t = board.take(1);
        assert!(t.deltas.is_empty());
        assert_eq!(t.newest(), Some(3));
        // Every reader took everything: nothing is kept.
        assert_eq!(board.kept_posts(0), 0);
        // Posts a reader has not taken stay: owner 1's post waits for reader 0.
        board.post(1, 1, vec![delta(0, 3, Some([1, 2]))]);
        assert_eq!(board.kept_posts(1), 1);
        assert_eq!(board.take(0).deltas.len(), 1);
        assert_eq!(board.kept_posts(1), 0);
    }

    /// `released_through(owner, m)` is the newest eviction retirement k acknowledged by an
    /// acknowledgement recorded at a peer retirement `j < m`, minimum over the peers.
    #[test]
    fn board_releases_only_what_was_acknowledged_before_this_retirement() {
        let board = PeerBoard::new(3);
        // Owner 0 posts through retirement 3; peer 1 applies it at its retirement 4, peer 2
        // at its retirement 5.
        for k in 1..=3u64 {
            board.post(0, k, vec![delta(0, 2, None)]);
        }
        let t1 = board.take(1);
        board.ack(1, 4, &t1);
        // Peer 2 has not acknowledged anything: nothing is released at any retirement.
        assert_eq!(board.released_through(0, 5), 0);
        let t2 = board.take(2);
        board.ack(2, 5, &t2);
        // Owner retirement m sees only acknowledgements at j <= m - 1.
        assert_eq!(
            board.released_through(0, 5),
            0,
            "peer 2 acked at j = 5, not < 5"
        );
        assert_eq!(board.released_through(0, 6), 3);
        // A later acknowledgement of newer posts only moves the cutoff once its j has passed.
        board.post(0, 4, vec![delta(0, 2, Some([5, 6]))]);
        board.post(0, 5, vec![delta(0, 2, None)]);
        let t1 = board.take(1);
        board.ack(1, 6, &t1);
        let t2 = board.take(2);
        board.ack(2, 7, &t2);
        assert_eq!(
            board.released_through(0, 7),
            3,
            "peer 2's newer ack is at j = 7"
        );
        assert_eq!(board.released_through(0, 8), 5);
        // An acknowledgement that covers nothing new is not recorded (it would only push the
        // retirement index later).
        let t1 = board.take(1);
        board.ack(1, 9, &t1);
        assert_eq!(board.released_through(0, 10), 5);
        // With no peer at all everything is released.
        assert_eq!(PeerBoard::new(1).released_through(0, 1), u64::MAX);
    }
}
