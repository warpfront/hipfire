// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FROZEN reference copy of beta's `TailPolicy` (commit 20c7b6de0e, `expert_unit_tail.rs`
//! lines 446-496 and 554-987, verbatim apart from this header and the imports; the types live in
//! this module, so they do not collide with the migrated ones of the parent). The
//! differential tests (`differential`) hold the migrated policy to it: identical
//! publish / unpublish / fills `(slot, unit, generation)` per retirement and identical
//! [`TailStats`](super::TailStats). Never edit the policy code below; change the parent instead.
#![allow(dead_code)]

use super::{EpUnitTailConfig, TailStats, WindowUse, FILL_GROUP_CAP};
use std::collections::HashMap;

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
        self.retire_windows(core, window, completed, 1)
    }

    /// [`Self::retire`] of a GROUP of `windows` consecutive decode windows (the retirement
    /// cadence, [`RetireCadence`]): `window` holds the distinct non-core units any window of
    /// the group read (reads summed), judged against the slots live when the group started. A
    /// unit read anywhere in the group is never a victim. The fill budget is `F x windows`
    /// capped at `F x`[`FILL_GROUP_CAP`]; `windows == 1` is exactly [`Self::retire`].
    pub fn retire_windows(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        windows: u32,
    ) -> Retire {
        assert!(
            !self.peer,
            "a policy with peer quarantine retires through `retire_peer`"
        );
        self.retire_inner(core, window, completed, 0, windows)
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
        self.retire_peer_windows(core, window, completed, 1, released_through)
    }

    /// [`Self::retire_peer`] of a group of `windows` windows ([`Self::retire_windows`]). The
    /// retirement index `m` counts retirements, not windows: the ranks retire the same groups.
    pub fn retire_peer_windows(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        windows: u32,
        released_through: impl FnOnce(u64) -> u64,
    ) -> Retire {
        assert!(
            self.peer,
            "`retire_peer` needs a policy with peer quarantine"
        );
        let through = released_through(self.window + 1);
        self.retire_inner(core, window, completed, through, windows)
    }

    /// Fills one retirement of `windows` windows may have in flight in total.
    fn fill_budget(&self, windows: u32) -> usize {
        self.config.fill_per_step * (windows as usize).clamp(1, FILL_GROUP_CAP)
    }

    fn retire_inner(
        &mut self,
        core: (u64, u64),
        window: &[WindowUse],
        completed: &[(usize, u64)],
        released_through: u64,
        windows: u32,
    ) -> Retire {
        let windows = windows.max(1);
        self.window += 1;
        let now = self.window;
        let windows_before = self.stats.windows;
        self.stats.windows += windows as u64;
        self.stats.retirements += 1;
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
        // Boundary mode admits when the group crossed a multiple of N WINDOWS (with one window
        // per retirement: every Nth window, as before).
        let every = self.config.boundary_every as u64;
        if boundary && (self.stats.windows / every) == (windows_before / every) {
            return out;
        }
        let total_budget = self.fill_budget(windows);
        let budget = total_budget.saturating_sub(self.fills_in_flight());
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
                if self.peer && quarantined >= QUARANTINE_WINDOWS * total_budget {
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
