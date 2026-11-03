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
//! This file is the pure part: the flag, the budget charge, the slot
//! policy ([`TailPolicy`]) with its tickets and counters, and the cross-rank
//! [`PeerBoard`] of the peer-visibility lever. It touches no device,
//! environment (beyond the parse helpers) or plan; the rank's device part is
//! [`crate::expert_unit_tail_rank`].
//!
//! # Guard
//!
//! One *window* is one trunk forward over every lane's sealed rows (an AR step
//! or a verify window of any row-group count). Hits are judged against the
//! slots published at the START of the window; the policy runs only at the
//! window's retirement ([`TailPolicy::retire`]), so a window sees ONE table
//! state. A slot is overwritten only after its previous unit was repointed to
//! canonical by the table upload that precedes the copy, and a fill is
//! published only after its copy completed ([`TailPolicy::retire`]'s
//! `completed`); a stale (older generation) completion is ignored.
//!
//! # Peer visibility (`HIPFIRE_EP_UNIT_TAIL_PEER`)
//!
//! A sub-512-row prefill takes the FULL-table route: every rank computes every expert, so a unit
//! the OWNER holds in a tail slot is read by the OTHER ranks from the owner's pinned host arena
//! unless their full tables name the owner's slot. With the lever the ranks (threads of one
//! process) share one [`PeerBoard`] and the protocol below. With it off there is no board and no
//! quarantine, and every decision of [`TailPolicy::retire`] is the pre-peer one.
//!
//! * POST. At each of its retirements `k` (its retirement counter, from 1; the warm start posts
//!   as `k = 0`) the owner posts the deltas of its OWN publications: `(unit, Some(slot
//!   addresses))` for a fill published at `k`, `(unit, None)` for a unit evicted at `k`. Only
//!   completed fills are ever posted: a publication comes out of `retire`'s `completed`, i.e. the
//!   owner observed the fill's copy-stream event complete.
//! * APPLY + ACK. At its retirement `j`, after its own policy step and before its own fills, a
//!   rank takes every delta posted so far and applies it to its FULL-table image of a PEER-owned
//!   entry whose full entry equals the owner's canonical entries (`peer_shared` in the rank part;
//!   any other entry is never touched), uploads the touched full tables on its active stream with
//!   its own table uploads, and only then records the ACK `(newest owner retirement it applied,
//!   j)`. A rank therefore names a peer slot only after the owner posted the publication (host
//!   order) and after its own apply.
//! * QUARANTINE. A slot the owner evicts at retirement `k` goes back to canonical in the owner's
//!   tables at once but its BYTES stay: it is `Quarantined` (neither free nor a victim; the
//!   candidate that triggered the eviction waits for a released slot). It becomes free at an owner
//!   retirement `m` only when, for EVERY peer `r`, some ack of `r` has `applied >= k` and `m >=
//!   j + 1` for the `j` it was recorded at ([`PeerBoard::released_through`]). At most
//!   [`QUARANTINE_WINDOWS`]` x F` slots are quarantined at once, so a peer that never acknowledges
//!   (frozen, released) cannot drain the pool; candidates that find no victim are `skipped`.
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

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};

/// `T,F[,b=N]`: tail slots per rank, fills per rank per window, optional
/// boundary mode (admit only every `N` windows). Unset, empty, `0` or `off`
/// leaves the tail off.
pub const EP_UNIT_TAIL_ENV: &str = "HIPFIRE_EP_UNIT_TAIL";
/// `1`: build the tail (pool, seed, usage log) but never admit or evict: the
/// A/B arm that prices the tail's bookkeeping without any fill traffic.
pub const EP_UNIT_TAIL_FREEZE_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_FREEZE";
/// `1`: the pool is visible to prefill routes. The rank's FULL (chunk / prefill) pointer tables
/// follow the tail's slot publications for the owned units whose canonical entries they share
/// with the decode tables, so a full-table prefill forward reads a live slot from VRAM instead of
/// the pinned host arena; and the long-prefill UNIT STAGE (the owner-routed tile route's staged
/// forwards) names a slot-live owned unit's slot in its pointer table and skips that unit's bytes
/// in the per-layer H2D copy (the decode tables alone see slots without it). Needs the tail;
/// default off.
pub const EP_UNIT_TAIL_PREFILL_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_PREFILL";
/// `1`: with the tail AND [`EP_UNIT_TAIL_PREFILL_ENV`], every rank's FULL (chunk / prefill)
/// pointer tables also follow the PEER ranks' slot publications for the peer-owned units whose
/// full-table entries equal the owner's canonical ones, so a full-table prefill forward (every
/// rank computes every expert) reads a unit the owner holds in a tail slot from the owner's VRAM
/// over P2P instead of from the owner's pinned host arena. An owner quarantines an evicted slot
/// until every peer's table dropped it (module docs). Default off; ignored (with a note) without
/// the prefill lever.
pub const EP_UNIT_TAIL_PEER_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_PEER";
/// `1`: print a `[unit-tail]` counter line every 64 windows and at release.
pub const EP_UNIT_TAIL_LOG_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_LOG";
/// `G` (whole units, default [`EP_UNIT_TAIL_STAGE_GAP_DEFAULT`]): with the prefill lever, the
/// slot-aware unit stage merges two H2D copy runs across a slot-live gap shorter than `G` units
/// (the gap units are copied too and simply not referenced: their stage-table entries still name
/// the slot). `0` = exactly the complement of the slot-live units.
pub const EP_UNIT_TAIL_STAGE_GAP_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_STAGE_GAP";
/// Default of [`EP_UNIT_TAIL_STAGE_GAP_ENV`].
pub const EP_UNIT_TAIL_STAGE_GAP_DEFAULT: usize = 8;

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

/// Whether [`EP_UNIT_TAIL_ENV`] asks for a tail in this process. A configured tail runs on STATIC
/// `e % ranks` owners: pinned owners (D1.9) deal the non-core experts of a call to either rank, so
/// a rank's tail would have to follow a pin plane both ranks agree on every window (simulated
/// replicated tails lose to D1.9 everywhere; see `SIM-PINNED.md`), and the unit layers' owner kind
/// ([`crate::gpu_forward::ep_unit_layer_owner_kind`]) keeps D1's static rule instead.
pub fn ep_unit_tail_configured() -> bool {
    parse_ep_unit_tail(
        hipfire_config::developer_var(EP_UNIT_TAIL_ENV)
            .ok()
            .as_deref(),
    )
    .is_ok_and(|config| config.is_some())
}

/// Whether the D2 tail's pool is visible to prefill routes: the tail is configured AND
/// [`EP_UNIT_TAIL_PREFILL_ENV`] is on. Gates both the full chunk tables following the slot
/// publications and the unit stage's slot-aware tables / copy runs.
pub fn ep_unit_tail_prefill_requested() -> bool {
    ep_unit_tail_configured() && hipfire_config::developer_bool(EP_UNIT_TAIL_PREFILL_ENV, false)
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

/// Lifecycle of one tail slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhase {
    /// Holds nothing the tables name.
    Free,
    /// A copy into the slot is in flight; the tables still name canonical.
    Filling,
    /// The copy completed and the tables name the slot.
    Live,
    /// PEER mode: evicted (the owner's tables name canonical again) but a peer's full table may
    /// still name the slot, so its bytes must stay until every peer acknowledged the
    /// unpublication: neither free nor a victim.
    Quarantined,
}

#[derive(Clone, Debug)]
struct Slot {
    unit: Option<UnitKey>,
    phase: SlotPhase,
    generation: u64,
    /// The window (counted from 1) that last used the unit the slot holds.
    last_used: u64,
    /// Ranking position of the held unit (smaller = hotter), cached for victim ordering.
    rank: u32,
    /// The retirement the slot was evicted at (meaningful while `Quarantined`).
    evicted_at: u64,
}

/// A copy the device part must start: `unit`'s canonical bytes into `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillOp {
    pub slot: usize,
    pub unit: UnitKey,
    pub generation: u64,
}

/// What one retired window changes. The device part uploads the touched layers'
/// tables (publish -> slot address, unpublish -> canonical) BEFORE it starts
/// the fills.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retire {
    /// Units whose table entry now names their slot.
    pub publish: Vec<(UnitKey, usize)>,
    /// Units (evicted) whose table entry returns to canonical.
    pub unpublish: Vec<UnitKey>,
    /// Copies to start (their slots are `Filling`).
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
    pub windows: u64,
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

/// The slot policy of one rank's tail: LRU eviction, admit-on-miss bounded by
/// `F` fills per window, tickets, counters.
#[derive(Debug)]
pub struct TailPolicy {
    config: EpUnitTailConfig,
    frozen: bool,
    /// PEER mode: evictions quarantine the slot instead of refilling it in the same retirement.
    peer: bool,
    slots: Vec<Slot>,
    /// Unit -> slot for slots that are `Filling` or `Live`.
    index: HashMap<UnitKey, usize>,
    /// Ranking position of a unit (smaller = hotter): the tie-break and, in
    /// boundary mode, the prior of the heat.
    priority: HashMap<UnitKey, u32>,
    /// Boundary mode: decayed read heat of non-core units.
    heat: HashMap<UnitKey, f32>,
    window: u64,
    stats: TailStats,
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

impl TailPolicy {
    /// `priority[unit]`: ranking position of every unit the rank may see
    /// (smaller = hotter); an unknown unit sorts last.
    pub fn new(config: EpUnitTailConfig, frozen: bool, priority: HashMap<UnitKey, u32>) -> Self {
        Self {
            config,
            frozen,
            peer: false,
            slots: vec![
                Slot {
                    unit: None,
                    phase: SlotPhase::Free,
                    generation: 0,
                    last_used: 0,
                    rank: 0,
                    evicted_at: 0,
                };
                config.tail_units
            ],
            index: HashMap::new(),
            priority,
            heat: HashMap::new(),
            window: 0,
            stats: TailStats::default(),
        }
    }

    pub fn config(&self) -> &EpUnitTailConfig {
        &self.config
    }

    pub fn stats(&self) -> &TailStats {
        &self.stats
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The slot that serves `unit` now (`Live` only).
    pub fn live_slot(&self, unit: UnitKey) -> Option<usize> {
        let slot = *self.index.get(&unit)?;
        (self.slots[slot].phase == SlotPhase::Live).then_some(slot)
    }

    /// Slots in a phase (diagnostics, tests).
    pub fn count(&self, phase: SlotPhase) -> usize {
        self.slots.iter().filter(|s| s.phase == phase).count()
    }

    /// The unit a slot holds (any non-free phase).
    pub fn slot_unit(&self, slot: usize) -> Option<UnitKey> {
        self.slots.get(slot).and_then(|s| s.unit)
    }

    /// Generation of a slot (bumped by every fill that reuses it).
    pub fn slot_generation(&self, slot: usize) -> u64 {
        self.slots[slot].generation
    }

    pub fn slot_phase(&self, slot: usize) -> SlotPhase {
        self.slots[slot].phase
    }

    /// The retirement a `Quarantined` slot was evicted at.
    pub fn slot_evicted_at(&self, slot: usize) -> Option<u64> {
        let s = &self.slots[slot];
        (s.phase == SlotPhase::Quarantined).then_some(s.evicted_at)
    }

    /// PEER mode: from now on an eviction quarantines the slot until
    /// [`Self::retire_peer`]'s `released_through` covers it. Must precede the first retirement.
    pub fn enable_peer_quarantine(&mut self) {
        assert_eq!(
            self.window, 0,
            "peer quarantine must precede the first retirement"
        );
        self.peer = true;
    }

    /// Warm start: the first `T` of `units` fill slots `0..` as already-LIVE
    /// (the device part copies and publishes them before the first window), so
    /// window 1 sees exactly what D1's static plan would hold. Returns the
    /// `(slot, unit, generation)` list to copy; units beyond `T` and repeats
    /// are ignored.
    pub fn seed(&mut self, units: &[UnitKey]) -> Vec<FillOp> {
        let mut out = Vec::new();
        for &unit in units {
            if out.len() >= self.slots.len() {
                break;
            }
            if self.index.contains_key(&unit) {
                continue;
            }
            let slot = out.len();
            let rank = self.rank_of(unit);
            let s = &mut self.slots[slot];
            s.unit = Some(unit);
            s.phase = SlotPhase::Live;
            s.generation = 1;
            s.last_used = 0;
            s.rank = rank;
            self.index.insert(unit, slot);
            out.push(FillOp {
                slot,
                unit,
                generation: 1,
            });
        }
        out
    }

    fn rank_of(&self, unit: UnitKey) -> u32 {
        self.priority.get(&unit).copied().unwrap_or(u32::MAX)
    }

    fn fills_in_flight(&self) -> usize {
        self.count(SlotPhase::Filling)
    }

    /// Retire one window. `core` = (distinct, slot reads) the static core served;
    /// `window` = the distinct NON-core units the window read on this rank;
    /// `completed` = `(slot, generation)` of fills whose copy finished since the
    /// last retirement. A policy with peer quarantine retires through
    /// [`Self::retire_peer`].
    pub fn retire(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
    ) -> Retire {
        assert!(
            !self.peer,
            "a policy with peer quarantine retires through `retire_peer`"
        );
        self.retire_inner(core, window, completed, 0)
    }

    /// [`Self::retire`] of a policy with peer quarantine. `released_through(m)` is asked once,
    /// with this retirement's index `m`, before anything else: it is the newest eviction
    /// retirement `k` that EVERY peer acknowledged by a retirement before `m`
    /// ([`PeerBoard::released_through`]); slots evicted at `k' <= k` are free from this
    /// retirement on (this retirement may refill them).
    pub fn retire_peer(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        released_through: impl FnOnce(u64) -> u64,
    ) -> Retire {
        assert!(
            self.peer,
            "`retire_peer` needs a policy with peer quarantine"
        );
        let through = released_through(self.window + 1);
        self.retire_inner(core, window, completed, through)
    }

    fn retire_inner(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        released_through: u64,
    ) -> Retire {
        self.window += 1;
        let now = self.window;
        self.stats.windows += 1;
        self.stats.core_distinct += core.0;
        self.stats.core_reads += core.1;
        let mut out = Retire::default();

        // 1. Judge the window against the slots LIVE at its start.
        let mut misses: Vec<WindowUse> = Vec::new();
        for &used in window {
            match self.index.get(&used.unit).copied() {
                Some(slot) if self.slots[slot].phase == SlotPhase::Live => {
                    self.slots[slot].last_used = now;
                    self.stats.tail_distinct += 1;
                    self.stats.tail_reads += used.reads as u64;
                }
                _ => {
                    self.stats.host_distinct += 1;
                    self.stats.host_reads += used.reads as u64;
                    misses.push(used);
                }
            }
        }
        if self.config.boundary_every > 1 {
            for used in window {
                *self.heat.entry(used.unit).or_insert(0.0) += used.reads as f32;
            }
        }

        // 2. Publish the fills that completed (they serve the NEXT window).
        for &(slot, generation) in completed {
            let valid = self.slots.get(slot).is_some_and(|s| {
                s.phase == SlotPhase::Filling && s.generation == generation && s.unit.is_some()
            });
            if !valid {
                self.stats.stale_completions += 1;
                continue;
            }
            let s = &mut self.slots[slot];
            s.phase = SlotPhase::Live;
            s.last_used = now;
            self.stats.fills_published += 1;
            out.publish.push((s.unit.expect("checked"), slot));
        }

        // 2b. PEER mode: a slot evicted at a retirement every peer has acknowledged is free
        // again (this retirement may refill it).
        if self.peer {
            for s in &mut self.slots {
                if s.phase == SlotPhase::Quarantined && s.evicted_at <= released_through {
                    s.phase = SlotPhase::Free;
                }
            }
        }

        // 3. Admission.
        if self.frozen {
            return out;
        }
        let boundary = self.config.boundary_every > 1;
        if boundary && now % self.config.boundary_every as u64 != 0 {
            return out;
        }
        let budget = self
            .config
            .fill_per_step
            .saturating_sub(self.fills_in_flight());
        if budget == 0 {
            return out;
        }
        let candidates = self.candidates(misses, boundary);
        if candidates.is_empty() {
            if boundary {
                self.decay_heat();
            }
            return out;
        }
        // Victims: free slots first, then Live slots NOT used by the retiring
        // window, least recently used first (colder ranking position on ties).
        let mut free: Vec<usize> = (0..self.slots.len())
            .filter(|&s| self.slots[s].phase == SlotPhase::Free)
            .collect();
        free.reverse(); // pop() takes the lowest slot first
                        // Only `budget` victims can be used: select them in O(T) instead of sorting all T.
        let mut live: Vec<(u64, std::cmp::Reverse<u32>, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.phase == SlotPhase::Live && s.last_used < now)
            .map(|(slot, s)| (s.last_used, std::cmp::Reverse(s.rank), slot))
            .collect();
        if live.len() > budget {
            live.select_nth_unstable(budget - 1);
            live.truncate(budget);
        }
        live.sort_unstable();
        let mut live = live.into_iter().map(|(_, _, slot)| slot);
        let mut quarantined = if self.peer {
            self.count(SlotPhase::Quarantined)
        } else {
            0
        };
        for cand in candidates.into_iter().take(budget) {
            let (slot, evicted) = if let Some(slot) = free.pop() {
                (slot, None)
            } else {
                if self.peer && quarantined >= QUARANTINE_WINDOWS * self.config.fill_per_step {
                    self.stats.skipped_full += 1;
                    continue;
                }
                let Some(slot) = live.next() else {
                    self.stats.skipped_full += 1;
                    continue;
                };
                let victim = self.slots[slot].unit.expect("live slot has a unit");
                if boundary {
                    let challenger = self.heat.get(&cand).copied().unwrap_or(0.0);
                    let incumbent = self.heat.get(&victim).copied().unwrap_or(0.0);
                    if challenger < incumbent * HEAT_HYSTERESIS {
                        self.stats.skipped_full += 1;
                        continue;
                    }
                }
                (slot, Some(victim))
            };
            if let Some(victim) = evicted {
                self.index.remove(&victim);
                out.unpublish.push(victim);
                self.stats.evictions += 1;
                if self.peer {
                    // Quarantine: the peers' full tables may still name the slot, so it takes no
                    // fill until they acknowledged the unpublication; the candidate waits.
                    let s = &mut self.slots[slot];
                    s.phase = SlotPhase::Quarantined;
                    s.unit = None;
                    s.evicted_at = now;
                    self.stats.quarantined += 1;
                    quarantined += 1;
                    continue;
                }
            }
            let rank = self.rank_of(cand);
            let s = &mut self.slots[slot];
            s.generation += 1;
            s.phase = SlotPhase::Filling;
            s.unit = Some(cand);
            s.last_used = now;
            s.rank = rank;
            self.index.insert(cand, slot);
            self.stats.fills_issued += 1;
            out.fills.push(FillOp {
                slot,
                unit: cand,
                generation: s.generation,
            });
        }
        if boundary {
            self.decay_heat();
        }
        out
    }

    /// Candidate misses, best first: per-window mode by reads then ranking;
    /// boundary mode by decayed heat then ranking.
    fn candidates(&self, misses: Vec<WindowUse>, boundary: bool) -> Vec<UnitKey> {
        let mut fresh: Vec<(UnitKey, f32)> = misses
            .into_iter()
            .filter(|m| !self.index.contains_key(&m.unit))
            .map(|m| {
                let score = if boundary {
                    self.heat.get(&m.unit).copied().unwrap_or(0.0)
                } else {
                    m.reads as f32
                };
                (m.unit, score)
            })
            .collect();
        if boundary {
            // Units hot over the interval but not missed in the last window.
            for (&unit, &heat) in &self.heat {
                if !self.index.contains_key(&unit) && !fresh.iter().any(|(u, _)| *u == unit) {
                    fresh.push((unit, heat));
                }
            }
        }
        fresh.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.rank_of(a.0).cmp(&self.rank_of(b.0)))
                .then_with(|| a.0.cmp(&b.0))
        });
        fresh.into_iter().map(|(unit, _)| unit).collect()
    }

    fn decay_heat(&mut self) {
        self.heat.retain(|_, h| {
            *h *= HEAT_DECAY;
            *h > 0.01
        });
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

/// The cross-rank state of the PEER lever (module docs): what each owner posted, what each rank
/// took, and the acknowledgements the owners release quarantined slots on. One board per process
/// and EP mesh, shared by every rank thread; every call is a short critical section.
#[derive(Debug)]
pub struct PeerBoard {
    ranks: usize,
    state: Mutex<BoardState>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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

    fn policy(t: usize, f: usize, b: u32) -> TailPolicy {
        // Ranking position = layer * 1000 + expert.
        let mut priority = HashMap::new();
        for layer in 0..4u16 {
            for e in 0..64u16 {
                priority.insert((layer, e), layer as u32 * 1000 + e as u32);
            }
        }
        TailPolicy::new(cfg(t, f, b), false, priority)
    }

    fn use1(layer: u16, expert: u16, reads: u32) -> WindowUse {
        WindowUse {
            unit: (layer, expert),
            reads,
        }
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
            seeds.iter().map(|f| f.unit).collect::<Vec<_>>(),
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
        assert_eq!(
            r.fills,
            vec![FillOp {
                slot: 0,
                unit: (1, 5),
                generation: 1
            }]
        );
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
        assert_eq!(
            r.fills,
            vec![FillOp {
                slot: 1,
                unit: (0, 9),
                generation: 2
            }]
        );
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
        let order: Vec<_> = r.fills.iter().map(|f| f.unit).collect();
        assert_eq!(order, vec![(2, 2), (0, 7), (1, 9)]);
    }

    #[test]
    fn a_frozen_policy_publishes_but_never_admits() {
        let mut p = policy(2, 4, 1);
        p.frozen = true;
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
            let mut table: HashMap<UnitKey, usize> =
                seeds.iter().map(|op| (op.unit, op.slot)).collect();
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
                    assert!(!named.contains(&fill.slot), "fill into a published slot");
                    assert_eq!(p.slot_generation(fill.slot), fill.generation);
                    assert_eq!(p.slot_unit(fill.slot), Some(fill.unit));
                    flight.push((fill.slot, fill.generation, (rng() % 4) as u32));
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
        let s = *p.stats();
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
    /// flag-off policy (the PEER lever's quarantine is off) against the digests recorded on the
    /// pre-peer policy (be067917f3). `peer` runs the same trace through the quarantine path with
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
        let mut p = TailPolicy::new(cfg(t, f, b), frozen, priority);
        if peer {
            p.enable_peer_quarantine();
        }
        let seeds = p.seed(&(0..(t / 2) as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for op in &seeds {
            fold(&mut h, op.slot as u64);
            fold(&mut h, op.generation);
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
                fold(&mut h, (fill.unit.0 as u64) << 16 | fill.unit.1 as u64);
                fold(&mut h, fill.slot as u64);
                fold(&mut h, fill.generation);
                flight.push((fill.slot, fill.generation, (rng() % 4) as u32));
            }
            fold(&mut h, 4);
        }
        let s = *p.stats();
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

    /// `(T, F, boundary, frozen, seed, digest)` of the pinned flag-off traces, recorded on the
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

    /// Flag off (no quarantine) decides exactly like the pre-peer policy: the digests of
    /// publish / unpublish / fill lists and counters over random traces (evictions, boundary
    /// mode, freeze, stale completions) are the ones recorded before the lever existed.
    #[test]
    fn flag_off_policy_decisions_match_the_pre_peer_golden_traces() {
        for (t, f, b, frozen, seed, want) in GOLDEN_CASES {
            let got = trace_digest(t, f, b, frozen, false, seed, 400);
            assert_eq!(
                got, want,
                "T={t} F={f} b={b} frozen={frozen}: flag-off decisions moved ({got:#018x})"
            );
        }
    }

    /// The quarantine path only differs from flag off when a slot is evicted: a trace that never
    /// needs a victim (every miss finds a free slot) decides identically in both modes.
    #[test]
    fn peer_mode_decides_like_flag_off_while_no_victim_is_needed() {
        for seed in [0x1234_5678_9abc_def1u64, 0x9e37_79b9_7f4a_7c15] {
            let off = trace_digest(200, 3, 1, false, false, seed, 300);
            let on = trace_digest(200, 3, 1, false, true, seed, 300);
            assert_eq!(off, on);
        }
    }

    #[test]
    fn peer_eviction_quarantines_the_slot_until_every_peer_acknowledged() {
        let mut p = policy(2, 4, 1);
        p.enable_peer_quarantine();
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
        let s = *p.stats();
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
        assert_eq!(
            r.fills,
            vec![FillOp {
                slot: first,
                unit: (0, 9),
                generation: 2
            }]
        );
        assert!(r.unpublish.is_empty());
        assert_eq!(p.count(SlotPhase::Quarantined), 1);
        assert_eq!(p.slot_phase(first), SlotPhase::Filling);
        // The release at the SAME retirement it is acknowledged; a later one frees the rest.
        let r = p.retire_peer((0, 0), &[], &[], |_| 2);
        assert!(r.fills.is_empty());
        assert_eq!(p.count(SlotPhase::Quarantined), 0);
        assert_eq!(p.count(SlotPhase::Free), 1);
    }

    /// A peer that never acknowledges (frozen, released) must not drain the pool: at most
    /// `QUARANTINE_WINDOWS` windows of fills are ever quarantined; the rest of the tail keeps
    /// serving and the candidates that find no victim are counted as skipped.
    #[test]
    fn a_silent_peer_quarantines_a_bounded_part_of_the_pool() {
        let (t, f) = (24usize, 2usize);
        let mut p = policy(t, f, 1);
        p.enable_peer_quarantine();
        p.seed(&(0..t as u16).map(|e| (0u16, e)).collect::<Vec<_>>());
        for w in 0..80u16 {
            // Every window misses 4 fresh units.
            let uses: Vec<WindowUse> = (0..4).map(|i| use1(1, (w * 4 + i) % 64, 1)).collect();
            let r = p.retire_peer((0, 0), &uses, &[], |_| 0);
            assert!(r.fills.is_empty(), "nothing is ever released");
            assert!(p.count(SlotPhase::Quarantined) <= QUARANTINE_WINDOWS * f);
        }
        assert_eq!(p.count(SlotPhase::Quarantined), QUARANTINE_WINDOWS * f);
        assert_eq!(p.count(SlotPhase::Live), t - QUARANTINE_WINDOWS * f);
        assert!(p.stats().skipped_full > 0);
        assert_eq!(p.stats().quarantined as usize, QUARANTINE_WINDOWS * f);
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
