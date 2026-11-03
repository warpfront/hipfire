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
//! This file is the pure part: the flag, the budget charge and the slot
//! policy ([`TailPolicy`]) with its tickets and counters. It touches no device,
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

use std::collections::HashMap;

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
/// `1`: print a `[unit-tail]` counter line every 64 windows and at release.
pub const EP_UNIT_TAIL_LOG_ENV: &str = "HIPFIRE_EP_UNIT_TAIL_LOG";

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

impl TailPolicy {
    /// `priority[unit]`: ranking position of every unit the rank may see
    /// (smaller = hotter); an unknown unit sorts last.
    pub fn new(config: EpUnitTailConfig, frozen: bool, priority: HashMap<UnitKey, u32>) -> Self {
        Self {
            config,
            frozen,
            slots: vec![
                Slot {
                    unit: None,
                    phase: SlotPhase::Free,
                    generation: 0,
                    last_used: 0,
                    rank: 0,
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
    /// last retirement.
    pub fn retire(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
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
        for cand in candidates.into_iter().take(budget) {
            let (slot, evicted) = if let Some(slot) = free.pop() {
                (slot, None)
            } else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
}
