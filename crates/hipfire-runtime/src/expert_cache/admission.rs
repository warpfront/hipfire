// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The admission seam of the policy core: what an adapter may decide around the core's slot
//! protocol without a second manager.
//!
//! [`ExpertCachePolicy::plan_group_with`](super::ExpertCachePolicy::plan_group_with) and
//! [`ExpertCachePolicy::finish_plan_with`](super::ExpertCachePolicy::finish_plan_with) call an
//! [`AdmissionStrategy`] at fixed points of a plan. The core keeps every invariant (a slot is only
//! reused through the bank's typestate transitions, a victim is never a key the group used, the
//! fill budget is `F x min(windows, cap)` minus the fills in flight, capacities are fixed); the
//! strategy decides only *what is worth admitting* and *how an eviction is staged*:
//!
//! | point | hook | default |
//! |---|---|---|
//! | after judging, with the whole group log (hits and misses) | [`AdmissionStrategy::observe`] | nothing |
//! | before any admission | [`AdmissionStrategy::open_gate`] | open |
//! | per missed key | [`AdmissionStrategy::score`] | `reads as f32` |
//! | before ordering | [`AdmissionStrategy::extra_candidates`] | none |
//! | per pool with budget and candidates | [`AdmissionStrategy::pass`] | [`PassPlan::default`] |
//! | per candidate that would take a victim | [`AdmissionStrategy::veto_victim`] | never |
//! | per withdrawn slot at reclaim time | [`AdmissionStrategy::hold_withdrawn`] | never |
//! | after the admission pass | [`AdmissionStrategy::settled`] | nothing |
//!
//! [`DefaultAdmission`] is the core's own behaviour (the `TailPolicy` per-window rules, non-peer):
//! `plan_group` / `finish_plan` are exactly `plan_group_with` / `finish_plan_with` over it.
//!
//! The scores are exact `f32`: a heat-based strategy never routes its numbers through the `u32`
//! `reads` of [`WindowUse`]. Candidates are ordered by score (descending; a NaN score counts as
//! `-inf`), then by ranking (ascending), then by key, which is the beta policy's order when the
//! score is `reads as f32`.

use super::index::KeyIndex;
use super::{CacheError, ExpertKey, PoolId, WindowUse};

/// What a plan knows about the group it judges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanContext {
    /// Windows the group covers (>= 1 for a judged plan).
    pub windows: u32,
    /// `CacheStats::windows` before this group was added.
    pub windows_before: u64,
    /// `CacheStats::windows` after it (`windows_before + windows`).
    pub windows_after: u64,
    /// The stamp of the plan (the `last_used` of every key the group used).
    pub now: u64,
}

impl PlanContext {
    /// Whether the group crossed a multiple of `every` windows (`every <= 1`: always): the
    /// boundary gate of D2's boundary mode, "every Nth window", counted in windows so a group of
    /// several windows still crosses at most the boundaries it covers.
    pub fn crossed(&self, every: u64) -> bool {
        every <= 1 || self.windows_after / every != self.windows_before / every
    }
}

/// One admission candidate: a missed key, or an extra candidate a strategy pushed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The key and its ranking (`reads` is the group's summed reads; `0` for an extra
    /// candidate).
    pub used: WindowUse,
    /// The strategy's score of the key: the ordering key, descending.
    pub score: f32,
}

/// The victim a candidate would displace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Victim {
    pub key: ExpertKey,
    pub pool: PoolId,
    pub slot: u32,
    /// Ranking position of the held key (smaller = hotter).
    pub rank: u32,
    /// Stamp of the victim's last use.
    pub last_used: u64,
}

/// How an eviction is staged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvictMode {
    /// The victim's slot takes the candidate's fill after the retirement (`withdraw` with a
    /// target; `finish_plan` starts the fill): the core's own mode.
    #[default]
    Deferred,
    /// The victim is withdrawn without a target and its slot waits (it is held by
    /// [`AdmissionStrategy::hold_withdrawn`] and the peer bank's acknowledgement until it can be
    /// reclaimed); the candidate that caused the eviction waits too and is admitted if it misses
    /// again. D2's peer quarantine.
    Quarantine,
}

/// What the core tells a strategy about one pool before its admission pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassContext {
    pub pool: PoolId,
    pub windows: u32,
    /// `F x clamp(windows, 1, cap)`: the fills a group may have in flight in total.
    pub total_budget: usize,
    /// `total_budget` minus the fills in flight: the candidates the pass may take (`> 0`).
    pub budget: usize,
    /// Candidates of this pool (missed keys not bound to a slot, plus the strategy's extras).
    pub candidates: usize,
    /// Slots in each phase when the pass starts (after completions were staged and, with the
    /// pool's generation already retired, the reclaim pass ran).
    pub free: usize,
    pub filling: usize,
    pub resident: usize,
    /// `Evicting` slots with no fill target (quarantined / held).
    pub quarantined: usize,
}

/// How a pool's admission pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassPlan {
    /// How a victim is staged.
    pub evict: EvictMode,
    /// [`EvictMode::Quarantine`]: a candidate that has no free slot is `skipped_full` once this
    /// many slots are quarantined (`usize::MAX`: no cap). D2: `QUARANTINE_WINDOWS x total_budget`.
    pub quarantine_cap: usize,
    /// Keep the reserve: every candidate takes a free slot if one exists and, while
    /// `free + quarantined` is below this target, also evicts one victim into quarantine for a
    /// later plan's fills (a candidate with neither a free slot nor an eviction is `skipped_full`;
    /// one that only evicted waits). `None`: off. Reserve evictions are always quarantined
    /// (withdrawn without a target), skip the veto and ignore `evict` and `quarantine_cap`.
    /// D2: `RESERVE_RETIREMENTS x budget`.
    pub reserve_target: Option<usize>,
}

impl Default for PassPlan {
    fn default() -> Self {
        Self {
            evict: EvictMode::Deferred,
            quarantine_cap: usize::MAX,
            reserve_target: None,
        }
    }
}

/// A withdrawn (`Evicting`) slot the core is about to reclaim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotHold {
    pub pool: PoolId,
    pub slot: u32,
    /// The core stamp of the plan that withdrew the slot.
    pub evicted_at: u64,
    /// The bank's `excluded_from_epoch`: the first generation epoch that no longer names the
    /// slot.
    pub excluded_from: Option<u64>,
    /// `released_through` of the retirement the reclaim is made against (`None` without peers).
    pub peer_through: Option<u64>,
    /// The slot has a deferred fill waiting for the reclaim.
    pub has_target: bool,
}

/// Receives the extra candidates of [`AdmissionStrategy::extra_candidates`].
pub struct CandidateSink<'a> {
    pub(super) index: &'a KeyIndex,
    pub(super) log: &'a KeyIndex,
    pub(super) cands: &'a mut Vec<Cand>,
    pub(super) capacity: usize,
    pub(super) pools: usize,
}

/// One candidate of the plan's scratch.
#[derive(Clone, Copy)]
pub(super) struct Cand {
    pub(super) used: WindowUse,
    pub(super) score: f32,
}

/// A NaN score would break the candidate order's totality: it counts as the worst score.
pub(super) fn clean_score(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

impl CandidateSink<'_> {
    /// Offer `key` (not read by the group) as a candidate with an exact `score`. A key already
    /// bound to a slot or read by the group (it is already a candidate, or a hit) is ignored:
    /// `Ok(false)`. The scratch holds as many candidates as the use log holds keys
    /// ([`CacheError::UseLogFull`] beyond that).
    pub fn push(
        &mut self,
        key: ExpertKey,
        pool: PoolId,
        rank: u32,
        score: f32,
    ) -> Result<bool, CacheError> {
        if pool.0 as usize >= self.pools {
            return Err(CacheError::UnknownPool(pool));
        }
        if self.index.get(&key).is_some() || self.log.get(&key).is_some() {
            return Ok(false);
        }
        if self.cands.len() >= self.capacity {
            return Err(CacheError::UseLogFull {
                capacity: self.capacity,
            });
        }
        self.cands.push(Cand {
            used: WindowUse {
                key,
                pool,
                reads: 0,
                rank,
            },
            score: clean_score(score),
        });
        Ok(true)
    }

    /// Candidates gathered so far (missed keys and extras).
    pub fn len(&self) -> usize {
        self.cands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cands.is_empty()
    }
}

/// The decisions an adapter owns around the core's slot protocol. Every method has the core's
/// default, so a strategy overrides only what it changes. Hooks must not allocate on the hot
/// path (the core calls them inside a plan).
pub trait AdmissionStrategy {
    /// Once per judged group, right after judging: `log` holds every distinct key the group read
    /// (hits and misses alike, reads summed). A heat-based strategy accumulates here.
    fn observe(&mut self, _ctx: &PlanContext, _log: &[WindowUse]) {}

    /// Whether this group may admit at all (completions were staged either way). `false`: no
    /// candidates are gathered, nothing is admitted or counted `skipped_full`, and
    /// [`Self::settled`] sees `reached == false`. D2's boundary mode: [`PlanContext::crossed`].
    fn open_gate(&mut self, _ctx: &PlanContext) -> bool {
        true
    }

    /// The score of a missed key: candidates are ordered by it, descending.
    fn score(&mut self, used: &WindowUse) -> f32 {
        used.reads as f32
    }

    /// Keys the group did not read that are worth admitting anyway (D2's boundary heat). Called
    /// once per pass, only when some pool has budget. Push with [`CandidateSink::push`].
    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        _sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        Ok(())
    }

    /// How the pool's pass runs, asked once per pool that has budget and candidates.
    fn pass(&mut self, _ctx: &PassContext) -> PassPlan {
        PassPlan::default()
    }

    /// `true`: `candidate` does not displace `victim` (hysteresis). The victim is consumed (the
    /// next candidate sees the next victim) and the candidate is `skipped_full`.
    fn veto_victim(&mut self, _candidate: &Candidate, _victim: &Victim) -> bool {
        false
    }

    /// `true`: keep the withdrawn slot `Evicting` for now (quarantine hold-back). Asked for every
    /// slot a reclaim would free, after the peer bank's own acknowledgement test (a peer pool
    /// never offers a slot its token does not cover). A held slot keeps its deferred fill, stays
    /// parked, and is asked again at the next reclaim.
    fn hold_withdrawn(&mut self, _hold: &SlotHold) -> bool {
        false
    }

    /// End of a judged, unfrozen plan's admission. `reached`: the gate was open and some pool had
    /// budget (the candidate stage was reached, even with no candidate). D2 decays its heat on
    /// `reached`.
    fn settled(&mut self, _ctx: &PlanContext, _reached: bool) {}
}

/// The core's own admission: per-window reads, deferred fills, no quarantine, no hysteresis.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAdmission;

impl AdmissionStrategy for DefaultAdmission {}
