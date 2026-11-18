// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tests of the policy core: the differential against a frozen copy of the Qwen4 D2
//! `TailPolicy` decision rules, the #870 `SlotCache` contract rewritten against this API, the
//! A2.1 no-retire-on-idle-window rule, capacity overflow errors and window poisoning.
//!
//! The rig drives the core through the real retirement protocol over a fake in-order device:
//! fills are issued and completed through the bank (`issue_fills` / `complete_fills`), a group's
//! generation is fenced and retired, the tables are "rewritten" (the test reads `published()` /
//! `withdrawn()`) and the next generation sealed.

use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::HashMap;

use railgun::sealed_table::generation::{
    BlockRegistry, Expert, FenceBackend, FenceFailure, FenceState, FillComplete, FillTicket,
    GenerationId, PeerReaderFence, SlotSpan,
};
use railgun::window::Quiescence;
use railgun::Binding;

use super::{
    AdmissionStrategy, CacheError, CacheStats, CancelOutcome, Candidate, CandidateSink, EvictMode,
    ExpertCachePolicy, ExpertKey, FillOp, Lookup, PassContext, PassPlan, PlanContext, PlanSummary,
    PolicyConfig, PoolConfig, PoolId, PoolInit, PoolMode, Published, Reservation, SlotHold,
    SlotPhase, Victim, WindowConsumerPhase, WindowUse, Withdrawn,
};

const BLOCK: u64 = 0x1000_0000;
const TABLE: u64 = 0x7000_0000;
const SLOT_BYTES: u64 = 0x1000;
const P0: PoolId = PoolId(0);
const P1: PoolId = PoolId(1);
const P2: PoolId = PoolId(2);

// ---------------------------------------------------------------------------------------------
// The rig
// ---------------------------------------------------------------------------------------------

/// An in-order fake stream. Fences complete at once unless a test makes them pend or fail.
struct Dev {
    next: u64,
    records: u64,
    hold: bool,
    fail_record: bool,
    fail_wait: Option<Quiescence>,
}

impl Dev {
    fn new() -> Self {
        Self {
            next: 0,
            records: 0,
            hold: false,
            fail_record: false,
            fail_wait: None,
        }
    }

    fn state(&self) -> FenceState {
        if let Some(quiescence) = self.fail_wait {
            FenceState::Failed(FenceFailure {
                message: "the fence failed".into(),
                quiescence,
            })
        } else if self.hold {
            FenceState::Pending
        } else {
            FenceState::Complete
        }
    }
}

// SAFETY: the fake stream executes everything at enqueue time, so every operation enqueued
// before `record()` returned `f` has finished when the fence is `Complete`; a test that makes it
// pend or fail only ever withholds that proof.
unsafe impl FenceBackend for Dev {
    fn incarnation(&self) -> u64 {
        1
    }
    fn record(&mut self) -> Result<u64, FenceFailure> {
        if self.fail_record {
            return Err(FenceFailure {
                message: "record refused".into(),
                quiescence: Quiescence::Proven,
            });
        }
        self.records += 1;
        self.next += 1;
        Ok(self.next)
    }
    fn poll(&mut self, _fence: u64) -> FenceState {
        self.state()
    }
    fn wait(&mut self, _fence: u64) -> FenceState {
        self.state()
    }
}

fn key(group: u32, expert: u32) -> ExpertKey {
    ExpertKey {
        owner_id: 1,
        owner_generation: 1,
        group_id: group,
        expert_id: expert,
    }
}

fn config() -> PolicyConfig {
    PolicyConfig {
        use_log: 512,
        completions: 64,
        parked: 64,
        consumers: 8,
        queues: 2,
        ..PolicyConfig::default()
    }
}

struct Spec {
    slots: usize,
    fills: usize,
    seed: Vec<(ExpertKey, u32)>,
}

fn spec(slots: usize, fills: usize) -> Spec {
    Spec {
        slots,
        fills,
        seed: Vec::new(),
    }
}

fn pool_init(
    registry: &BlockRegistry,
    pool: u64,
    slots: usize,
    seed: &[(ExpertKey, u32)],
) -> PoolInit {
    let block = registry
        .adopt::<Expert>(BLOCK + pool * 0x100_0000, slots as u64 * SLOT_BYTES, 1)
        .expect("the pool block");
    let spans = (0..slots as u64)
        .map(|slot| {
            vec![SlotSpan {
                offset: slot * SLOT_BYTES,
                bytes: SLOT_BYTES,
            }]
        })
        .collect();
    PoolInit {
        block,
        spans,
        tables: vec![Binding {
            base: TABLE + pool * 0x1000,
            bytes: 0x100,
        }],
        seed: seed.to_vec(),
    }
}

fn build(config: PolicyConfig, specs: &[Spec]) -> Result<ExpertCachePolicy, CacheError> {
    let registry = BlockRegistry::new(0);
    let pools = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (
                PoolConfig {
                    slots: s.slots,
                    fills_per_window: s.fills,
                },
                pool_init(&registry, i as u64, s.slots, &s.seed),
            )
        })
        .collect();
    ExpertCachePolicy::new(config, pools)
}

type Decision = (PoolId, u32, ExpertKey, u64);

/// Take every fill the policy decided, issue them per pool and complete them (the copy
/// "finishes" at once); the tokens are returned for the test to deliver when it likes.
fn issue_tokens(
    p: &mut ExpertCachePolicy,
    copy: &mut Dev,
) -> (Vec<Decision>, Vec<(PoolId, FillComplete)>) {
    let ops: Vec<FillOp> = p.take_fills().collect();
    let decisions = ops
        .iter()
        .map(|op| (op.pool, op.slot, op.key, op.fill_generation))
        .collect();
    let mut by_pool: Vec<Vec<FillTicket>> = (0..p.pools()).map(|_| Vec::new()).collect();
    for op in ops {
        by_pool[op.pool.0 as usize].push(op.ticket);
    }
    let mut tokens = Vec::new();
    for (i, tickets) in by_pool.into_iter().enumerate() {
        if tickets.is_empty() {
            continue;
        }
        let pool = PoolId(i as u16);
        let batch = p
            .issue_fills(pool, tickets, copy, &mut |_t: &FillTicket| Ok(()))
            .expect("the fills issue");
        let done = p
            .complete_fills(pool, batch, copy, true)
            .expect("the fills complete");
        tokens.extend(done.into_iter().map(|t| (pool, t)));
    }
    (decisions, tokens)
}

/// What one finished group decided (read before its pools sealed).
#[derive(Debug, Default)]
struct Outcome {
    published: Vec<Published>,
    withdrawn: Vec<Withdrawn>,
    fills: Vec<Decision>,
}

struct Rig {
    p: ExpertCachePolicy,
    copy: Dev,
    compute: Dev,
    /// Completion tokens issued and not yet delivered.
    held: Vec<(PoolId, FillComplete)>,
}

fn rig(config: PolicyConfig, specs: Vec<Spec>) -> Rig {
    Rig {
        p: build(config, &specs).expect("the policy builds"),
        copy: Dev::new(),
        compute: Dev::new(),
        held: Vec::new(),
    }
}

fn rig1(slots: usize, fills: usize) -> Rig {
    rig(config(), vec![spec(slots, fills)])
}

fn used(pool: PoolId, key: ExpertKey, reads: u32, rank: u32) -> WindowUse {
    WindowUse {
        key,
        pool,
        reads,
        rank,
    }
}

impl Rig {
    /// Retire, rewrite (the test reads the records first), reclaim and seal every pool that has
    /// something pending.
    fn retire_cycle(&mut self) {
        for i in 0..self.p.pools() {
            let pool = PoolId(i as u16);
            if !self.p.pool_needs_retirement(pool) {
                continue;
            }
            self.p
                .retire_generation(pool, &mut self.compute)
                .expect("the generation retires");
            self.p.finish_plan(pool).expect("the plan finishes");
            self.p.seal(pool).expect("the next generation seals");
        }
    }

    /// Stage every delivered completion (a publication-only plan) and retire what that made
    /// pending; a pending retirement of an earlier `reserve` goes first.
    fn publish_cycle(&mut self) -> PlanSummary {
        if self.p.needs_retirement() {
            self.retire_cycle();
        }
        let summary = self.p.plan_group(0, (0, 0)).expect("the publication plan");
        if self.p.needs_retirement() {
            self.retire_cycle();
        }
        summary
    }

    fn issue(&mut self) -> Vec<Decision> {
        let (decisions, tokens) = issue_tokens(&mut self.p, &mut self.copy);
        self.held.extend(tokens);
        decisions
    }

    fn deliver_held(&mut self) {
        for (pool, token) in self.held.drain(..) {
            self.p.deliver_completion(pool, token).expect("delivered");
        }
    }

    /// Run the protocol until nothing is in flight: issue, complete, deliver, stage, retire.
    fn settle(&mut self) {
        for _ in 0..16 {
            self.issue();
            let progressed = !self.held.is_empty() || self.p.needs_retirement();
            if !progressed {
                return;
            }
            self.deliver_held();
            self.publish_cycle();
        }
        panic!("settle did not converge");
    }

    fn window_routed(&mut self, uses: &[WindowUse], routed: u64) -> Option<u32> {
        self.p.begin_window().expect("the window opens");
        let id = self
            .p
            .register_consumer(0, WindowConsumerPhase::Decode, 0)
            .expect("the consumer registers");
        for &u in uses {
            self.p.record_use(u).expect("the use logs");
        }
        self.p.join_consumer(id).expect("the consumer joins");
        self.p.end_window(routed).expect("the window ends")
    }

    fn window(&mut self, uses: &[WindowUse]) -> Option<u32> {
        let routed = uses.iter().map(|u| u.reads as u64).sum();
        self.window_routed(uses, routed)
    }

    /// Run `windows` as one group and judge it.
    fn group(&mut self, windows: &[&[WindowUse]], core: (u64, u64)) -> PlanSummary {
        for w in windows {
            self.window(w);
        }
        self.p
            .plan_group(windows.len() as u32, core)
            .expect("the group plans")
    }

    /// Finish a planned group: read the table records, retire if something is pending, issue the
    /// fills.
    fn finish_group(&mut self) -> Outcome {
        let published = self.p.published().to_vec();
        let withdrawn = self.p.withdrawn().to_vec();
        if self.p.needs_retirement() {
            self.retire_cycle();
        }
        let fills = self.issue();
        Outcome {
            published,
            withdrawn,
            fills,
        }
    }

    fn reserve(&mut self, pool: PoolId, key: ExpertKey) -> Reservation {
        self.p.reserve(pool, key).expect("the key reserves")
    }
}

/// Completion tokens of ANOTHER bank (a decoy cache), one per slot, slot `i` at fill generation 1.
fn decoy_tokens(n: usize) -> Vec<FillComplete> {
    let seed: Vec<(ExpertKey, u32)> = (0..n as u32).map(|e| (key(77, e), 0)).collect();
    let mut p = build(
        config(),
        &[Spec {
            slots: n,
            fills: 1,
            seed,
        }],
    )
    .expect("the decoy builds");
    let mut dev = Dev::new();
    let (_, tokens) = issue_tokens(&mut p, &mut dev);
    tokens.into_iter().map(|(_, t)| t).collect()
}

// ---------------------------------------------------------------------------------------------
// The frozen reference: TailPolicy::retire_inner, non-peer, per-window mode, not frozen
// ---------------------------------------------------------------------------------------------

type Unit = (u16, u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RPhase {
    Free,
    Filling,
    Live,
}

#[derive(Clone, Copy, Debug)]
struct RSlot {
    phase: RPhase,
    unit: Unit,
    last_used: u64,
    rank: u32,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RStats {
    windows: u64,
    core_distinct: u64,
    core_reads: u64,
    tail_distinct: u64,
    tail_reads: u64,
    host_distinct: u64,
    host_reads: u64,
    fills_issued: u64,
    fills_published: u64,
    evictions: u64,
    skipped_full: u64,
    stale_completions: u64,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct RRetire {
    publish: Vec<(Unit, usize)>,
    unpublish: Vec<Unit>,
    /// `(slot, unit, generation)`.
    fills: Vec<(usize, Unit, u64)>,
}

struct RefPolicy {
    fill_per_step: usize,
    slots: Vec<RSlot>,
    index: HashMap<Unit, usize>,
    window: u64,
    stats: RStats,
}

const REF_FILL_GROUP_CAP: usize = 4;

impl RefPolicy {
    fn new(slots: usize, fill_per_step: usize) -> Self {
        Self {
            fill_per_step,
            slots: vec![
                RSlot {
                    phase: RPhase::Free,
                    unit: (0, 0),
                    last_used: 0,
                    rank: 0,
                    generation: 0,
                };
                slots
            ],
            index: HashMap::new(),
            window: 0,
            stats: RStats::default(),
        }
    }

    /// `seed` + `finish_seed`: the first `T` distinct units, live at fill generation 1.
    fn seed(&mut self, units: &[(Unit, u32)]) {
        let mut next = 0;
        for &(unit, rank) in units {
            if next >= self.slots.len() {
                break;
            }
            if self.index.contains_key(&unit) {
                continue;
            }
            self.slots[next] = RSlot {
                phase: RPhase::Live,
                unit,
                last_used: 0,
                rank,
                generation: 1,
            };
            self.index.insert(unit, next);
            next += 1;
        }
    }

    fn filling(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.phase == RPhase::Filling)
            .count()
    }

    /// `window`: `(unit, reads, rank)` distinct units the group read; `completed`: the
    /// `(slot, generation)` pairs whose copy finished.
    fn retire(
        &mut self,
        core: (u64, u64),
        window: &[(Unit, u32, u32)],
        completed: &[(usize, u64)],
        windows: u32,
    ) -> RRetire {
        let windows = windows.max(1);
        self.window += 1;
        let now = self.window;
        self.stats.windows += windows as u64;
        self.stats.core_distinct += core.0;
        self.stats.core_reads += core.1;
        let mut out = RRetire::default();

        // 1. Judge against the slots LIVE at the start.
        let mut misses: Vec<(Unit, u32, u32)> = Vec::new();
        for &(unit, reads, rank) in window {
            match self.index.get(&unit).copied() {
                Some(slot) if self.slots[slot].phase == RPhase::Live => {
                    self.slots[slot].last_used = now;
                    self.stats.tail_distinct += 1;
                    self.stats.tail_reads += reads as u64;
                }
                _ => {
                    self.stats.host_distinct += 1;
                    self.stats.host_reads += reads as u64;
                    misses.push((unit, reads, rank));
                }
            }
        }

        // 2. Publish the fills that completed.
        for &(slot, generation) in completed {
            let ok = self
                .slots
                .get(slot)
                .is_some_and(|s| s.phase == RPhase::Filling && s.generation == generation);
            if ok {
                self.slots[slot].phase = RPhase::Live;
                self.slots[slot].last_used = now;
                self.stats.fills_published += 1;
                out.publish.push((self.slots[slot].unit, slot));
            } else {
                self.stats.stale_completions += 1;
            }
        }

        // 3. Admission.
        let total_budget = self.fill_per_step * (windows as usize).clamp(1, REF_FILL_GROUP_CAP);
        let budget = total_budget.saturating_sub(self.filling());
        if budget == 0 {
            return out;
        }
        let mut fresh: Vec<(Unit, f32, u32)> = misses
            .into_iter()
            .filter(|m| !self.index.contains_key(&m.0))
            .map(|m| (m.0, m.1 as f32, m.2))
            .collect();
        fresh.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        if fresh.is_empty() {
            return out;
        }
        let mut free: Vec<usize> = (0..self.slots.len())
            .filter(|&s| self.slots[s].phase == RPhase::Free)
            .collect();
        free.reverse();
        let mut live: Vec<(u64, Reverse<u32>, usize)> = (0..self.slots.len())
            .filter(|&slot| {
                self.slots[slot].phase == RPhase::Live && self.slots[slot].last_used < now
            })
            .map(|slot| {
                let s = &self.slots[slot];
                (s.last_used, Reverse(s.rank), slot)
            })
            .collect();
        if live.len() > budget {
            live.select_nth_unstable(budget - 1);
            live.truncate(budget);
        }
        live.sort_unstable();
        let mut live = live.into_iter().map(|(_, _, slot)| slot);
        for (cand, _, rank) in fresh.into_iter().take(budget) {
            let slot = if let Some(slot) = free.pop() {
                slot
            } else {
                let Some(slot) = live.next() else {
                    self.stats.skipped_full += 1;
                    continue;
                };
                let victim = self.slots[slot].unit;
                self.slots[slot].phase = RPhase::Free;
                self.index.remove(&victim);
                out.unpublish.push(victim);
                self.stats.evictions += 1;
                slot
            };
            let s = &mut self.slots[slot];
            s.phase = RPhase::Filling;
            s.unit = cand;
            s.generation += 1;
            s.last_used = now;
            s.rank = rank;
            self.index.insert(cand, slot);
            self.stats.fills_issued += 1;
            out.fills.push((slot, cand, s.generation));
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// (a) The differential test
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn unit_key(pool: usize, unit: Unit) -> ExpertKey {
    ExpertKey {
        owner_id: pool as u64,
        owner_generation: 1,
        group_id: unit.0 as u32,
        expert_id: unit.1 as u32,
    }
}

fn unit_rank(unit: Unit) -> u32 {
    unit.0 as u32 * 1000 + unit.1 as u32
}

fn core_phase(phase: RPhase) -> SlotPhase {
    match phase {
        RPhase::Free => SlotPhase::Free,
        RPhase::Filling => SlotPhase::Filling,
        RPhase::Live => SlotPhase::Live,
    }
}

/// One route stream: `groups` groups of 1..=4 windows over `pools` pools, with a hot set and a
/// uniform tail, lagged completions and stale (foreign-bank) completions.
fn run_differential(pools: &[(usize, usize, usize)], groups: usize, rng_seed: u64) {
    let mut rng = Rng(rng_seed);
    let specs: Vec<Spec> = pools
        .iter()
        .enumerate()
        .map(|(i, &(slots, fills, seeds))| Spec {
            slots,
            fills,
            seed: (0..seeds as u16)
                .map(|e| (unit_key(i, (0, e)), unit_rank((0, e))))
                .collect(),
        })
        .collect();
    let mut r = rig(
        PolicyConfig {
            retire_ids: u64::MAX,
            max_windows: u32::MAX,
            ..config()
        },
        specs,
    );
    let mut refs: Vec<RefPolicy> = pools
        .iter()
        .map(|&(slots, fills, seeds)| {
            let mut policy = RefPolicy::new(slots, fills);
            let units: Vec<(Unit, u32)> = (0..seeds as u16)
                .map(|e| ((0, e), unit_rank((0, e))))
                .collect();
            policy.seed(&units);
            policy
        })
        .collect();
    let caps = r.p.capacities();

    // The warm start: complete the seeds and publish them.
    r.settle();
    let mut retirements = pools.iter().filter(|p| p.2 > 0).count() as u64;
    for (i, policy) in refs.iter().enumerate() {
        assert_slots_match(&r.p, PoolId(i as u16), policy, "after the warm start");
        assert!(r.p.check_seeded(PoolId(i as u16)).is_ok());
    }
    assert_eq!(
        r.p.stats().fills_published,
        0,
        "seeds are not counted as fills"
    );

    let mut decoys = decoy_tokens(8);
    // `(due group, pool, token)`.
    let mut in_flight: Vec<(usize, PoolId, FillComplete)> = Vec::new();
    for g in 0..groups {
        // Deliver the completions due now, in issue order; refs get the same pairs.
        let mut completed: Vec<Vec<(usize, u64)>> = vec![Vec::new(); pools.len()];
        let mut keep = Vec::new();
        for (due, pool, token) in in_flight.drain(..) {
            if due <= g {
                completed[pool.0 as usize].push((token.slot() as usize, token.fill_generation()));
                r.p.deliver_completion(pool, token).expect("delivered");
            } else {
                keep.push((due, pool, token));
            }
        }
        in_flight = keep;
        if g % 11 == 3 {
            if let Some(token) = decoys.pop() {
                let pool = PoolId((g % pools.len()) as u16);
                completed[pool.0 as usize].push((token.slot() as usize, u64::MAX));
                r.p.deliver_completion(pool, token).expect("delivered");
            }
        }

        // The group's windows: per pool the distinct units with summed reads, first-seen order.
        let n = 1 + rng.below(4) as u32;
        let mut merged: Vec<Vec<(Unit, u32, u32)>> = vec![Vec::new(); pools.len()];
        for _ in 0..n {
            let mut window = Vec::new();
            for (pi, m) in merged.iter_mut().enumerate() {
                for _ in 0..3 + rng.below(6) {
                    let layer = rng.below(3) as u16;
                    let hot = rng.below(10) < 6;
                    let expert = if hot { rng.below(6) } else { rng.below(24) };
                    let unit = (layer, expert as u16);
                    let reads = 1 + rng.below(3) as u32;
                    window.push(used(
                        PoolId(pi as u16),
                        unit_key(pi, unit),
                        reads,
                        unit_rank(unit),
                    ));
                    match m.iter_mut().find(|e| e.0 == unit) {
                        Some(e) => e.1 += reads,
                        None => m.push((unit, reads, unit_rank(unit))),
                    }
                }
            }
            // The core merges repeats inside a window itself; feed it the raw stream.
            r.window(&window);
        }
        let core = (rng.below(5), rng.below(9));
        let summary = r.p.plan_group(n, core).expect("the group plans");
        let outcome = r.finish_group();

        let mut want_active = false;
        for (pi, policy) in refs.iter_mut().enumerate() {
            let pool = PoolId(pi as u16);
            let want = policy.retire(
                if pi == 0 { core } else { (0, 0) },
                &merged[pi],
                &completed[pi],
                n,
            );
            if !want.publish.is_empty() || !want.unpublish.is_empty() {
                want_active = true;
                retirements += 1;
            }

            let published: Vec<(Unit, usize)> = outcome
                .published
                .iter()
                .filter(|e| e.pool == pool)
                .map(|e| {
                    (
                        (e.key.group_id as u16, e.key.expert_id as u16),
                        e.slot as usize,
                    )
                })
                .collect();
            assert_eq!(published, want.publish, "group {g} pool {pi}: publications");
            let withdrawn: Vec<Unit> = outcome
                .withdrawn
                .iter()
                .filter(|e| e.pool == pool)
                .map(|e| (e.key.group_id as u16, e.key.expert_id as u16))
                .collect();
            assert_eq!(withdrawn, want.unpublish, "group {g} pool {pi}: victims");
            let mut fills: Vec<(usize, Unit, u64)> = outcome
                .fills
                .iter()
                .filter(|d| d.0 == pool)
                .map(|d| {
                    (
                        d.1 as usize,
                        (d.2.group_id as u16, d.2.expert_id as u16),
                        d.3,
                    )
                })
                .collect();
            fills.sort();
            let mut want_fills = want.fills.clone();
            want_fills.sort();
            assert_eq!(fills, want_fills, "group {g} pool {pi}: fills");
            assert_slots_match(&r.p, pool, policy, &format!("group {g}"));
        }
        assert_eq!(
            summary.needs_retirement, want_active,
            "group {g}: a retirement happens exactly when something is pending (A2.1)"
        );
        assert_stats_match(&r.p.stats(), &refs, &format!("group {g}"));
        assert_eq!(r.p.stats().generations_retired, retirements, "group {g}");
        assert_eq!(r.p.stats().generations_sealed, retirements, "group {g}");

        // The fills just issued complete after a lag of 0..=2 groups.
        for (pool, token) in r.held.drain(..) {
            let lag = (token.slot() as u64 * 7 + token.fill_generation() * 3 + g as u64) % 3;
            in_flight.push((g + 1 + lag as usize, pool, token));
        }
    }
    for i in 0..pools.len() {
        let bank = r.p.bank(PoolId(i as u16)).expect("the bank");
        assert_eq!(
            bank.live_generations(),
            1,
            "one lease per generation, always"
        );
    }
    assert_eq!(r.p.capacities(), caps, "no buffer grew");
}

fn assert_slots_match(p: &ExpertCachePolicy, pool: PoolId, policy: &RefPolicy, at: &str) {
    for (slot, want) in policy.slots.iter().enumerate() {
        let slot = slot as u32;
        assert_eq!(
            p.slot_phase(pool, slot),
            Some(core_phase(want.phase)),
            "{at}: pool {} slot {slot} phase",
            pool.0
        );
        assert_eq!(
            p.slot_generation(pool, slot),
            Some(want.generation),
            "{at}: pool {} slot {slot} fill generation",
            pool.0
        );
        if want.phase != RPhase::Free {
            assert_eq!(
                p.slot_key(pool, slot),
                Some(unit_key(pool.0 as usize, want.unit)),
                "{at}: pool {} slot {slot} key",
                pool.0
            );
        }
    }
}

fn assert_stats_match(got: &CacheStats, refs: &[RefPolicy], at: &str) {
    let sum = |f: fn(&RStats) -> u64| refs.iter().map(|r| f(&r.stats)).sum::<u64>();
    assert_eq!(got.windows, refs[0].stats.windows, "{at}: windows");
    assert_eq!(got.core_distinct, refs[0].stats.core_distinct, "{at}");
    assert_eq!(got.core_reads, refs[0].stats.core_reads, "{at}");
    assert_eq!(got.tail_distinct, sum(|s| s.tail_distinct), "{at}");
    assert_eq!(got.tail_reads, sum(|s| s.tail_reads), "{at}");
    assert_eq!(got.host_distinct, sum(|s| s.host_distinct), "{at}");
    assert_eq!(got.host_reads, sum(|s| s.host_reads), "{at}");
    assert_eq!(got.fills_issued, sum(|s| s.fills_issued), "{at}");
    assert_eq!(got.fills_published, sum(|s| s.fills_published), "{at}");
    assert_eq!(got.evictions, sum(|s| s.evictions), "{at}");
    assert_eq!(got.skipped_full, sum(|s| s.skipped_full), "{at}");
    assert_eq!(got.stale_completions, sum(|s| s.stale_completions), "{at}");
}

#[test]
fn differential_single_pool_matches_the_beta_policy() {
    run_differential(&[(8, 2, 3)], 400, 0x9e37_79b9_7f4a_7c15);
}

#[test]
fn differential_pressure_pool_matches_the_beta_policy() {
    // Few slots, one fill per window: constant eviction and skipped candidates.
    run_differential(&[(3, 1, 3)], 400, 0x2545_f491_4f6c_dd1d);
}

#[test]
fn differential_wide_pool_matches_the_beta_policy() {
    run_differential(&[(24, 4, 0)], 300, 0xdead_beef_1234_5678);
}

#[test]
fn differential_two_pools_decide_independently() {
    run_differential(&[(6, 2, 2), (4, 1, 4)], 400, 0x0123_4567_89ab_cdef);
}

// Direct ports of the beta policy's own rule tests.

#[test]
fn lru_evicts_the_oldest_key_not_used_by_the_group() {
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 2,
            fills: 4,
            seed: vec![(key(0, 1), 1), (key(0, 2), 2)],
        }],
    );
    r.settle();
    // The group uses (0,1) only and misses (0,9): slot 1 ((0,2), unused) is the victim.
    let summary = r.group(
        &[&[used(P0, key(0, 1), 2, 1), used(P0, key(0, 9), 5, 9)]],
        (0, 0),
    );
    assert_eq!((summary.hits, summary.misses), (1, 1));
    assert_eq!(
        (summary.withdrawn, summary.deferred, summary.fills),
        (1, 1, 0)
    );
    let out = r.finish_group();
    assert_eq!(
        out.withdrawn,
        vec![Withdrawn {
            key: key(0, 2),
            pool: P0,
            slot: 1
        }]
    );
    assert_eq!(out.fills, vec![(P0, 1, key(0, 9), 2)]);
    assert_eq!(r.p.lookup(key(0, 2)), Lookup::Absent);
    assert!(matches!(
        r.p.lookup(key(0, 1)),
        Lookup::Published { slot: 0, .. }
    ));
    assert_eq!(r.p.stats().evictions, 1);
}

#[test]
fn keys_used_by_the_group_are_never_victims() {
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 2,
            fills: 8,
            seed: vec![(key(0, 1), 1), (key(0, 2), 2)],
        }],
    );
    r.settle();
    // Both resident keys are read; two misses want slots: no victim exists.
    let summary = r.group(
        &[&[
            used(P0, key(0, 1), 1, 1),
            used(P0, key(0, 2), 1, 2),
            used(P0, key(0, 7), 9, 7),
            used(P0, key(0, 8), 8, 8),
        ]],
        (0, 0),
    );
    assert_eq!(summary.skipped_full, 2);
    assert!(!summary.needs_retirement);
    let out = r.finish_group();
    assert!(out.fills.is_empty() && out.withdrawn.is_empty());
    assert_eq!(r.p.stats().skipped_full, 2);
    assert_eq!(r.p.count(P0, SlotPhase::Live), 2);
}

#[test]
fn the_fill_budget_is_f_per_window_minus_in_flight_capped_at_four_f() {
    let misses: Vec<WindowUse> = (0..5u32).map(|e| used(P0, key(2, e), 1, e)).collect();
    let mut r = rig1(8, 2);
    r.group(&[misses.as_slice()], (0, 0));
    assert_eq!(r.finish_group().fills.len(), 2, "F = 2");
    // Both still in flight: nothing new.
    r.group(&[misses.as_slice()], (0, 0));
    assert!(r.finish_group().fills.is_empty());
    // One completes: one more may start.
    let first = r.held.remove(0);
    r.p.deliver_completion(first.0, first.1).unwrap();
    let summary = r.group(&[misses.as_slice()], (0, 0));
    assert_eq!(summary.published, 1);
    assert_eq!(summary.fills, 1);
    r.finish_group();

    // The group budget is F x min(windows, 4).
    let many: Vec<WindowUse> = (0..10u32).map(|e| used(P0, key(2, e), 1, e)).collect();
    let mut r = rig1(8, 2);
    let w: Vec<&[WindowUse]> = (0..6).map(|_| many.as_slice()).collect();
    let summary = r.group(&w, (0, 0));
    assert_eq!(summary.fills, 8, "2 x min(6, 4) fills");
}

#[test]
fn candidates_are_ordered_by_reads_then_ranking() {
    let mut r = rig1(8, 3);
    let summary = r.group(
        &[&[
            used(P0, key(3, 4), 1, 3004),
            used(P0, key(1, 9), 5, 1009),
            used(P0, key(0, 7), 5, 7),
            used(P0, key(2, 2), 7, 2002),
        ]],
        (0, 0),
    );
    assert_eq!(summary.fills, 3);
    let order: Vec<ExpertKey> = r.finish_group().fills.iter().map(|d| d.2).collect();
    assert_eq!(order, vec![key(2, 2), key(0, 7), key(1, 9)]);
}

#[test]
fn a_fill_is_invisible_until_published_and_the_group_that_missed_it_still_counts_a_miss() {
    let mut r = rig1(2, 4);
    let a = key(1, 5);
    r.group(&[&[used(P0, a, 3, 5)]], (0, 0));
    let out = r.finish_group();
    assert_eq!(out.fills, vec![(P0, 0, a, 1)]);
    assert!(matches!(r.p.lookup(a), Lookup::Pending { slot: 0, .. }));
    // The copy completes inside the next group: its use is still a host read.
    r.deliver_held();
    let summary = r.group(&[&[used(P0, a, 1, 5)]], (0, 0));
    assert_eq!((summary.hits, summary.misses, summary.published), (0, 1, 1));
    assert_eq!(r.p.stats().host_reads, 4);
    r.finish_group();
    // The group after that hits it.
    let summary = r.group(&[&[used(P0, a, 2, 5)]], (0, 0));
    assert_eq!(summary.hits, 1);
    assert_eq!((r.p.stats().tail_reads, r.p.stats().tail_distinct), (2, 1));
}

// ---------------------------------------------------------------------------------------------
// (b) The #870 SlotCache contract against the reserve / lookup / rollback API
// ---------------------------------------------------------------------------------------------

#[test]
fn a_reservation_publishes_only_after_its_completion_is_staged_and_sealed() {
    let mut r = rig1(2, 2);
    let a = key(0, 10);
    let fill = r.reserve(P0, a);
    assert_eq!(
        fill,
        Reservation::Fill {
            key: a,
            pool: P0,
            slot: 0,
            fill_generation: 1
        }
    );
    assert!(fill.needs_fill());
    assert!(matches!(
        r.p.lookup(a),
        Lookup::Pending {
            slot: 0,
            fill_generation: 1,
            ..
        }
    ));
    // The copy finishing is not publication.
    r.issue();
    r.deliver_held();
    assert!(matches!(r.p.lookup(a), Lookup::Pending { .. }));
    // The next plan stages it.
    let summary = r.p.plan_group(0, (0, 0)).unwrap();
    assert_eq!(summary.published, 1);
    assert!(summary.needs_retirement);
    assert_eq!(
        r.p.published(),
        &[Published {
            key: a,
            pool: P0,
            slot: 0,
            fill_generation: 1
        }]
    );
    assert!(matches!(
        r.p.lookup(a),
        Lookup::Published {
            slot: 0,
            fill_generation: 1,
            ..
        }
    ));
    r.retire_cycle();
    assert!(r.p.published().is_empty(), "the records end with the seal");
    assert_eq!(
        r.p.bank(P0).unwrap().refs(0),
        1,
        "the new generation names the slot"
    );
    let hit = r.reserve(P0, a);
    assert_eq!(
        hit,
        Reservation::Hit {
            key: a,
            pool: P0,
            slot: 0,
            fill_generation: 1
        }
    );
    assert!(!hit.needs_fill());
    let s = r.p.stats();
    assert_eq!(
        (s.reserve_hits, s.reserve_misses, s.fills_published),
        (1, 1, 1)
    );
}

#[test]
fn rollback_makes_a_failed_fill_miss_again_in_another_slot() {
    let mut r = rig1(2, 2);
    let a = key(0, 20);
    let first = r.reserve(P0, a);
    assert_eq!(r.p.rollback(first), Ok(true));
    assert_eq!(r.p.lookup(a), Lookup::Absent);
    // The rolled back copy still owns slot 0 until it completes: the retry takes slot 1.
    let retry = r.reserve(P0, a);
    let Reservation::Fill { slot, .. } = retry else {
        panic!("{retry:?}")
    };
    assert_eq!(slot, 1);
    assert_eq!(
        r.p.rollback(first),
        Ok(false),
        "a stale reservation rolls nothing back"
    );
    // Both copies complete: the retry publishes, the cancelled one withdraws its slot.
    r.issue();
    r.deliver_held();
    let summary = r.publish_cycle();
    assert_eq!((summary.published, summary.cancelled), (1, 1));
    assert!(matches!(r.p.lookup(a), Lookup::Published { slot: 1, .. }));
    assert_eq!(r.p.slot_phase(P0, 0), Some(SlotPhase::Free));
    let s = r.p.stats();
    assert_eq!(
        (s.rollbacks, s.cancelled_fills, s.fills_published),
        (1, 1, 1)
    );
    // The cancelled slot is reusable at a newer fill generation.
    let b = key(0, 21);
    let Reservation::Fill {
        slot,
        fill_generation,
        ..
    } = r.reserve(P0, b)
    else {
        panic!("expected a fill")
    };
    assert_eq!((slot, fill_generation), (0, 2));
}

#[test]
fn a_stale_reservation_never_cancels_a_newer_fill_of_the_reused_slot() {
    // #870's ABA scenario: reserve A, roll back A, reserve A again in the same slot, then the
    // first reservation / completion arrives. The bank's fill generation tells them apart.
    let mut r = rig1(1, 1);
    let a = key(4, 4);
    let first = r.reserve(P0, a);
    assert_eq!(r.p.rollback(first), Ok(true));
    assert_eq!(
        r.p.reserve(P0, a),
        Err(CacheError::NoProgress {
            pool: P0,
            free: 0,
            filling: 1,
            protected: 0,
            withdrawn: 0
        }),
        "the cancelled copy still owns the only slot"
    );
    r.issue();
    r.deliver_held();
    let summary = r.publish_cycle();
    assert_eq!((summary.published, summary.cancelled), (0, 1));
    assert_eq!(
        r.p.lookup(a),
        Lookup::Absent,
        "a cancelled fill never publishes"
    );
    let second = r.reserve(P0, a);
    assert_eq!(
        second,
        Reservation::Fill {
            key: a,
            pool: P0,
            slot: 0,
            fill_generation: 2
        },
        "the same slot, a newer fill generation"
    );
    // The old reservation matches key, pool and slot, but not the fill generation.
    assert_eq!(r.p.rollback(first), Ok(false));
    assert!(matches!(
        r.p.lookup(a),
        Lookup::Pending {
            fill_generation: 2,
            ..
        }
    ));
    // A completion of another bank's slot 0 (an earlier fill generation of ANOTHER pool) is
    // stale and publishes nothing.
    let foreign = decoy_tokens(1).remove(0);
    r.p.deliver_completion(P0, foreign).unwrap();
    r.issue();
    r.deliver_held();
    let summary = r.publish_cycle();
    assert_eq!((summary.stale, summary.published), (1, 1));
    assert_eq!(r.p.stats().stale_completions, 1);
    assert!(matches!(
        r.p.lookup(a),
        Lookup::Published {
            slot: 0,
            fill_generation: 2,
            ..
        }
    ));
}

#[test]
fn a_foreign_completion_never_publishes_a_pending_fill() {
    let mut r = rig1(2, 2);
    let a = key(0, 1);
    r.reserve(P0, a);
    let foreign = decoy_tokens(1).remove(0);
    r.p.deliver_completion(P0, foreign).unwrap();
    let summary = r.p.plan_group(0, (0, 0)).unwrap();
    assert_eq!((summary.stale, summary.published), (1, 0));
    assert!(matches!(r.p.lookup(a), Lookup::Pending { .. }));
    assert!(
        !r.p.needs_retirement(),
        "a stale completion retires nothing"
    );
    // The fill is still issuable and completes normally.
    r.settle();
    assert!(matches!(r.p.lookup(a), Lookup::Published { .. }));
}

#[test]
fn a_pending_fill_is_never_evicted_and_a_published_victim_waits_for_the_retirement() {
    let mut r = rig1(1, 1);
    let (a, b) = (key(0, 1), key(0, 2));
    r.reserve(P0, a);
    assert_eq!(
        r.p.reserve(P0, b),
        Err(CacheError::NoProgress {
            pool: P0,
            free: 0,
            filling: 1,
            protected: 0,
            withdrawn: 0
        }),
        "the only slot is filling"
    );
    r.settle();
    // A published, B takes its slot: the slot is withdrawn, its fill deferred.
    let deferred = r.reserve(P0, b);
    assert_eq!(
        deferred,
        Reservation::Deferred {
            key: b,
            pool: P0,
            slot: 0,
            fill_generation: 2
        }
    );
    assert!(deferred.needs_fill());
    assert_eq!(r.p.lookup(a), Lookup::Absent, "the victim is forgotten");
    assert!(matches!(
        r.p.lookup(b),
        Lookup::Pending {
            slot: 0,
            fill_generation: 2,
            ..
        }
    ));
    assert_eq!(
        r.p.withdrawn(),
        &[Withdrawn {
            key: a,
            pool: P0,
            slot: 0
        }]
    );
    // The live generation still names the slot: no copy may start into it.
    assert_eq!(r.p.bank(P0).unwrap().refs(0), 1);
    assert_eq!(r.p.slot_phase(P0, 0), Some(SlotPhase::Withdrawn));
    assert_eq!(r.p.fills_pending(), 0);
    assert_eq!(
        r.p.finish_plan(P0),
        Err(CacheError::RetirementRequired { pool: P0 })
    );
    // A second duplicate of B is pending, not a hit.
    assert!(matches!(
        r.reserve(P0, b),
        Reservation::Pending {
            fill_generation: 2,
            ..
        }
    ));
    r.retire_cycle();
    assert_eq!(
        r.p.bank(P0).unwrap().refs(0),
        0,
        "the old generation retired"
    );
    assert_eq!(
        r.p.fills_pending(),
        1,
        "the retirement started the deferred fill"
    );
    r.settle();
    assert!(matches!(
        r.p.lookup(b),
        Lookup::Published {
            slot: 0,
            fill_generation: 2,
            ..
        }
    ));
}

#[test]
fn the_least_recently_used_slot_is_the_victim_and_is_forgotten() {
    let mut r = rig1(2, 2);
    let (k1, k2, k3) = (key(0, 1), key(0, 2), key(0, 3));
    r.reserve(P0, k1);
    r.reserve(P0, k2);
    r.settle();
    // A hit on 1 makes 2 the LRU.
    assert!(matches!(
        r.reserve(P0, k1),
        Reservation::Hit { slot: 0, .. }
    ));
    // 3 takes 2's slot.
    assert_eq!(
        r.reserve(P0, k3),
        Reservation::Deferred {
            key: k3,
            pool: P0,
            slot: 1,
            fill_generation: 2
        }
    );
    r.settle();
    assert_eq!(
        r.p.lookup(k2),
        Lookup::Absent,
        "2 is gone: re-requesting it is a miss"
    );
    assert!(matches!(r.p.lookup(k1), Lookup::Published { slot: 0, .. }));
    assert!(matches!(
        r.p.lookup(k3),
        Lookup::Published {
            slot: 1,
            fill_generation: 2,
            ..
        }
    ));
    assert!(r.reserve(P0, k2).needs_fill());
}

#[test]
fn pools_are_independent() {
    let mut r = rig(config(), vec![spec(1, 1), spec(1, 1), spec(1, 1)]);
    let keys = [(P0, key(0, 1)), (P1, key(1, 1)), (P2, key(2, 1))];
    for &(pool, k) in &keys {
        assert!(matches!(
            r.reserve(pool, k),
            Reservation::Fill { slot: 0, .. }
        ));
    }
    r.settle();
    for &(pool, k) in &keys {
        assert!(
            matches!(r.reserve(pool, k), Reservation::Hit { slot: 0, .. }),
            "{pool:?}"
        );
    }
    assert_eq!(r.p.stats().evictions, 0, "none evicted each other");
}

#[test]
fn a_key_belongs_to_one_pool() {
    let mut r = rig(config(), vec![spec(2, 1), spec(2, 1)]);
    let k = key(0, 9);
    r.reserve(P0, k);
    assert_eq!(
        r.p.reserve(P1, k),
        Err(CacheError::PoolMismatch {
            key: k,
            bound: P0,
            requested: P1
        })
    );
    assert_eq!(
        r.p.reserve(PoolId(7), k),
        Err(CacheError::UnknownPool(PoolId(7)))
    );
}

#[test]
fn a_duplicate_of_a_pending_key_is_pending_never_a_hit() {
    let mut r = rig1(2, 2);
    let k = key(0, 3);
    let first = r.reserve(P0, k);
    let second = r.reserve(P0, k);
    assert_eq!(
        second,
        Reservation::Pending {
            key: k,
            pool: P0,
            slot: 0,
            fill_generation: 1
        }
    );
    assert!(first.needs_fill() && !second.needs_fill());
    let s = r.p.stats();
    assert_eq!(
        (s.duplicate_pending, s.reserve_hits, s.reserve_misses),
        (1, 0, 1)
    );
    assert_eq!(r.p.rollback(second), Ok(false), "a duplicate owns no fill");
    assert!(matches!(r.p.lookup(k), Lookup::Pending { .. }));
}

#[test]
fn the_hit_rate_tracks_reuse() {
    let mut r = rig1(4, 2);
    let k = key(0, 7);
    r.reserve(P0, k);
    r.settle();
    r.reserve(P0, k);
    r.reserve(P0, k);
    let s = r.p.stats();
    assert_eq!((s.reserve_hits, s.reserve_misses), (2, 1));
}

#[test]
fn churn_never_hands_out_a_slot_outside_the_pool_nor_grows_a_buffer() {
    let mut r = rig1(3, 3);
    let caps = r.p.capacities();
    for e in 0..64u32 {
        match r.reserve(P0, key(5, e)) {
            Reservation::Fill { slot, .. } | Reservation::Deferred { slot, .. } => {
                assert!(slot < 3, "slot {slot} outside a pool of 3 at expert {e}");
            }
            other => panic!("expert {e}: {other:?}"),
        }
        r.settle();
        assert!(r.p.keys() <= 3, "the key index outgrew the pool");
    }
    assert_eq!(r.p.capacities(), caps);
    assert_eq!(r.p.bank(P0).unwrap().live_generations(), 1);
}

#[test]
fn a_windows_working_set_stays_resident_and_one_more_key_is_refused() {
    let mut r = rig1(6, 2);
    let keys: Vec<ExpertKey> = [11u32, 42, 7, 200, 3, 99]
        .iter()
        .map(|&e| key(2, e))
        .collect();
    for &k in &keys {
        assert!(
            r.reserve(P0, k).needs_fill(),
            "first touch of {k:?} should miss"
        );
    }
    r.settle();
    r.p.begin_window().unwrap();
    for &k in &keys {
        assert!(
            !r.reserve(P0, k).needs_fill(),
            "{k:?} was evicted inside one working set"
        );
    }
    // Every slot is used by the open window: the seventh key has no victim.
    assert_eq!(
        r.p.reserve(P0, key(2, 500)),
        Err(CacheError::NoProgress {
            pool: P0,
            free: 0,
            filling: 0,
            protected: 6,
            withdrawn: 0
        })
    );
    assert_eq!(
        r.p.lookup(key(2, 500)),
        Lookup::Absent,
        "a refusal changes nothing"
    );
    r.p.end_window(6).unwrap();
    // The window ended: the least recently used slot is evictable again.
    assert!(matches!(
        r.reserve(P0, key(2, 500)),
        Reservation::Deferred { slot: 0, .. }
    ));
}

#[test]
fn duplicates_in_one_window_need_one_slot_each() {
    let mut r = rig1(2, 2);
    r.p.begin_window().unwrap();
    let (x, y) = (key(0, 7), key(0, 1));
    assert!(r.reserve(P0, x).needs_fill());
    assert!(!r.reserve(P0, x).needs_fill());
    assert!(!r.reserve(P0, x).needs_fill());
    assert!(r.reserve(P0, y).needs_fill());
    assert!(!r.reserve(P0, y).needs_fill());
    assert_eq!(r.p.fills_pending(), 2);
    assert_eq!(r.p.stats().duplicate_pending, 3);
}

#[test]
fn a_rolled_back_deferred_reservation_leaves_the_slot_withdrawn_and_adoptable() {
    let mut r = rig1(1, 1);
    let (a, b, c) = (key(0, 1), key(0, 2), key(0, 3));
    r.reserve(P0, a);
    r.settle();
    let deferred = r.reserve(P0, b);
    assert_eq!(r.p.rollback(deferred), Ok(true));
    assert_eq!(r.p.lookup(b), Lookup::Absent);
    // The withdrawn slot has no taker: another key adopts it without a second eviction.
    let evictions = r.p.stats().evictions;
    assert_eq!(
        r.reserve(P0, c),
        Reservation::Deferred {
            key: c,
            pool: P0,
            slot: 0,
            fill_generation: 2
        }
    );
    assert_eq!(r.p.stats().evictions, evictions);
    r.settle();
    assert!(matches!(r.p.lookup(c), Lookup::Published { slot: 0, .. }));
}

// ---------------------------------------------------------------------------------------------
// The warm start
// ---------------------------------------------------------------------------------------------

#[test]
fn the_seed_warms_the_first_distinct_keys_and_is_not_counted_as_fills() {
    let seed = vec![
        (key(0, 1), 1),
        (key(0, 2), 2),
        (key(0, 1), 1),
        (key(0, 3), 3),
        (key(0, 4), 4),
    ];
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 3,
            fills: 2,
            seed,
        }],
    );
    assert_eq!(r.p.fills_pending(), 3);
    assert_eq!(
        r.p.check_seeded(P0),
        Err(CacheError::SeedIncomplete { pool: P0 })
    );
    r.settle();
    assert_eq!(r.p.check_seeded(P0), Ok(()));
    assert_eq!(r.p.count(P0, SlotPhase::Live), 3);
    assert!(matches!(
        r.p.lookup(key(0, 2)),
        Lookup::Published { slot: 1, .. }
    ));
    assert_eq!(r.p.lookup(key(0, 4)), Lookup::Absent);
    let s = r.p.stats();
    assert_eq!(
        (s.fills_published, s.fills_issued, s.generations_sealed),
        (0, 0, 1)
    );
}

#[test]
fn a_seed_key_cannot_belong_to_two_pools() {
    let k = key(0, 1);
    let err = build(
        config(),
        &[
            Spec {
                slots: 2,
                fills: 1,
                seed: vec![(k, 0)],
            },
            Spec {
                slots: 2,
                fills: 1,
                seed: vec![(k, 0)],
            },
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        CacheError::PoolMismatch {
            key: k,
            bound: P0,
            requested: P1
        }
    );
}

#[test]
fn construction_names_a_bad_shape() {
    assert_eq!(build(config(), &[]).unwrap_err(), CacheError::NoPools);
    let registry = BlockRegistry::new(0);
    let one_slot = pool_init(&registry, 1, 1, &[]);
    let err = ExpertCachePolicy::new(
        config(),
        vec![(
            PoolConfig {
                slots: 0,
                fills_per_window: 1,
            },
            one_slot,
        )],
    )
    .unwrap_err();
    assert_eq!(
        err,
        CacheError::PoolShape {
            pool: P0,
            why: "a pool needs at least one slot"
        }
    );
    let registry = BlockRegistry::new(0);
    let mut init = pool_init(&registry, 0, 2, &[]);
    init.spans.pop();
    let err = ExpertCachePolicy::new(
        config(),
        vec![(
            PoolConfig {
                slots: 2,
                fills_per_window: 1,
            },
            init,
        )],
    )
    .unwrap_err();
    assert_eq!(
        err,
        CacheError::PoolShape {
            pool: P0,
            why: "slot extents for another slot count"
        }
    );
}

// ---------------------------------------------------------------------------------------------
// (c) A2.1: no retirement without something pending
// ---------------------------------------------------------------------------------------------

fn seeded_rig(slots: usize, seeds: u32) -> Rig {
    let seed = (0..seeds).map(|e| (key(0, e), e)).collect();
    let mut r = rig(
        config(),
        vec![Spec {
            slots,
            fills: 2,
            seed,
        }],
    );
    r.settle();
    r
}

#[test]
fn a_group_with_nothing_pending_neither_seals_nor_retires_nor_syncs() {
    let mut r = seeded_rig(4, 4);
    let before = (r.p.stats(), r.compute.records, r.copy.records);
    assert_eq!(
        before.0.generations_retired, 1,
        "the warm start retired the empty first one"
    );
    assert_eq!(before.0.generations_sealed, 1);
    for g in 0..20u32 {
        // Every window reads resident keys only.
        let summary = r.group(
            &[
                &[
                    used(P0, key(0, g % 4), 2, 0),
                    used(P0, key(0, (g + 1) % 4), 1, 1),
                ],
                &[used(P0, key(0, (g + 2) % 4), 3, 2)],
            ],
            (1, 1),
        );
        assert_eq!((summary.hits, summary.misses), (3, 0));
        assert!(!summary.needs_retirement, "group {g}");
        assert!(!r.p.needs_retirement());
        r.finish_group();
    }
    let s = r.p.stats();
    assert_eq!(s.generations_retired, 1, "no retirement");
    assert_eq!(s.generations_sealed, 1, "no seal");
    assert_eq!(
        r.compute.records, before.1,
        "no last-read fence was recorded"
    );
    assert_eq!(r.copy.records, before.2, "no copy fence either");
    let bank = r.p.bank(P0).unwrap();
    assert_eq!(bank.live_generations(), 1);
    assert_eq!(
        r.p.generation_state(P0),
        Some("live under its lease"),
        "one lease continues"
    );
    // The protocol refuses the steps an idle group has no business with.
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::GenerationState {
            pool: P0,
            state: "has nothing pending"
        })
    );
    assert_eq!(r.p.finish_plan(P0), Ok(0));
    assert!(matches!(
        r.p.seal(P0),
        Err(CacheError::GenerationState { pool: P0, .. })
    ));
    assert_eq!(r.compute.records, before.1);
}

#[test]
fn a_group_that_only_starts_fills_into_free_slots_does_not_retire_either() {
    let mut r = seeded_rig(4, 2);
    let records = r.compute.records;
    let summary = r.group(&[&[used(P0, key(1, 9), 5, 9)]], (0, 0));
    assert_eq!((summary.fills, summary.deferred), (1, 0));
    assert!(!summary.needs_retirement);
    let out = r.finish_group();
    assert_eq!(out.fills.len(), 1);
    assert_eq!(r.compute.records, records);
    assert_eq!(r.p.stats().generations_retired, 1);
}

#[test]
fn a_completed_fill_or_a_victim_makes_the_retirement_mandatory_before_the_next_window() {
    let mut r = seeded_rig(4, 2);
    r.group(&[&[used(P0, key(1, 9), 5, 9)]], (0, 0));
    r.finish_group();
    // The copy completes: staging it makes a retirement pending.
    r.deliver_held();
    let summary = r.p.plan_group(0, (0, 0)).unwrap();
    assert!(summary.needs_retirement && r.p.pool_needs_retirement(P0));
    assert_eq!(r.p.begin_window(), Err(CacheError::PlanPending));
    assert_eq!(r.p.plan_group(1, (0, 0)), Err(CacheError::PlanPending));
    let records = r.compute.records;
    r.retire_cycle();
    assert_eq!(
        r.compute.records,
        records + 1,
        "exactly one fence for the retirement"
    );
    assert_eq!(r.p.stats().generations_retired, 2);
    assert_eq!(r.p.stats().generations_sealed, 2);
    r.p.begin_window().unwrap();
}

#[test]
fn the_retirement_steps_are_ordered() {
    let mut r = seeded_rig(1, 1);
    let b = key(0, 9);
    r.reserve(P0, b); // withdraws the published `a`
    assert_eq!(
        r.p.seal(P0),
        Err(CacheError::PlanPending),
        "a withdrawn slot is not reclaimed"
    );
    assert_eq!(
        r.p.finish_plan(P0),
        Err(CacheError::RetirementRequired { pool: P0 })
    );
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.generation_state(P0), Some("retired"));
    // Retiring again is idempotent.
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.stats().generations_retired, 2);
    assert_eq!(r.p.seal(P0), Err(CacheError::PlanPending));
    assert_eq!(
        r.p.finish_plan(P0),
        Ok(1),
        "the deferred fill of `b` starts"
    );
    assert_eq!(r.p.slot_phase(P0, 0), Some(SlotPhase::Filling));
    r.p.seal(P0).unwrap();
    assert_eq!(r.p.generation_state(P0), Some("live under its lease"));
}

#[test]
fn a_pending_last_read_fence_keeps_the_generation_until_it_completes() {
    let mut r = rig1(2, 2);
    r.reserve(P0, key(0, 1));
    r.issue();
    r.deliver_held();
    r.p.plan_group(0, (0, 0)).unwrap();
    r.compute.hold = true;
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::WindowFencePending { pool: P0 })
    );
    assert_eq!(
        r.p.generation_state(P0),
        Some("waiting for its last-read fence")
    );
    let records = r.compute.records;
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::WindowFencePending { pool: P0 })
    );
    assert_eq!(
        r.compute.records, records,
        "the pending fence is reused, not re-recorded"
    );
    assert_eq!(r.p.bank(P0).unwrap().live_generations(), 1);
    r.compute.hold = false;
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.bank(P0).unwrap().live_generations(), 0);
    r.p.finish_plan(P0).unwrap();
    r.p.seal(P0).unwrap();
    assert_eq!(r.p.stats().generations_retired, 1);
}

#[test]
fn an_unknown_fence_failure_quarantines_the_generation_for_good() {
    let mut r = rig1(2, 2);
    r.reserve(P0, key(0, 1));
    r.issue();
    r.deliver_held();
    r.p.plan_group(0, (0, 0)).unwrap();
    r.compute.fail_wait = Some(Quiescence::Unknown);
    let err = r.p.retire_generation(P0, &mut r.compute).unwrap_err();
    let CacheError::Fence { pool: P0, failure } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(failure.quiescence, Quiescence::Unknown);
    assert_eq!(r.p.generation_state(P0), Some("quarantined"));
    r.compute.fail_wait = None;
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::Quarantined { pool: P0 })
    );
    assert_eq!(
        r.p.finish_plan(P0),
        Err(CacheError::RetirementRequired { pool: P0 })
    );
    assert!(matches!(
        r.p.seal(P0),
        Err(CacheError::GenerationState { pool: P0, .. })
    ));
    assert_eq!(
        r.p.begin_window(),
        Err(CacheError::PlanPending),
        "fail closed: no new window"
    );
    assert_eq!(
        r.p.bank(P0).unwrap().live_generations(),
        1,
        "the slots never reclaim"
    );
}

#[test]
fn a_proven_fence_failure_still_retires_and_a_refused_record_keeps_the_lease() {
    let mut r = rig1(2, 2);
    r.reserve(P0, key(0, 1));
    r.issue();
    r.deliver_held();
    r.p.plan_group(0, (0, 0)).unwrap();
    r.compute.fail_record = true;
    let err = r.p.retire_generation(P0, &mut r.compute).unwrap_err();
    let CacheError::Fence { pool: P0, failure } = &err else {
        panic!("{err:?}")
    };
    assert_eq!(failure.quiescence, Quiescence::Proven);
    assert_eq!(r.p.generation_state(P0), Some("live under its lease"));
    r.compute.fail_record = false;
    r.compute.fail_wait = Some(Quiescence::Proven);
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.generation_state(P0), Some("retired"));
}

#[test]
fn a_frozen_cache_stages_completions_but_admits_nothing() {
    let mut r = rig1(2, 2);
    let a = key(0, 1);
    r.reserve(P0, a);
    r.issue();
    r.deliver_held();
    r.p.freeze();
    assert!(r.p.is_frozen());
    assert_eq!(r.p.reserve(P0, key(0, 2)), Err(CacheError::Frozen));
    let summary = r.group(&[&[used(P0, key(0, 5), 4, 5)]], (0, 0));
    assert_eq!(
        (summary.published, summary.fills),
        (1, 0),
        "published, never admitted"
    );
    assert_eq!(r.p.fills_pending(), 0);
    r.retire_cycle();
    assert!(matches!(r.p.lookup(a), Lookup::Published { .. }));
}

// ---------------------------------------------------------------------------------------------
// (c) Capacity overflow: named errors, never growth
// ---------------------------------------------------------------------------------------------

#[test]
fn the_use_log_overflows_with_a_named_error() {
    let mut r = rig(
        PolicyConfig {
            use_log: 2,
            ..config()
        },
        vec![spec(2, 1)],
    );
    assert_eq!(
        r.p.record_use(used(P0, key(0, 1), 1, 0)),
        Err(CacheError::WindowNotOpen)
    );
    r.p.begin_window().unwrap();
    r.p.record_use(used(P0, key(0, 1), 1, 0)).unwrap();
    r.p.record_use(used(P0, key(0, 2), 1, 0)).unwrap();
    r.p.record_use(used(P0, key(0, 1), 4, 0)).unwrap(); // a repeat sums, it takes no entry
    assert_eq!(
        r.p.record_use(used(P0, key(0, 3), 1, 0)),
        Err(CacheError::UseLogFull { capacity: 2 })
    );
    assert_eq!(
        r.p.record_use(used(PoolId(3), key(0, 1), 1, 0)),
        Err(CacheError::UnknownPool(PoolId(3)))
    );
    r.p.end_window(6).unwrap();
    let summary = r.p.plan_group(1, (0, 0)).unwrap();
    assert_eq!(summary.misses, 2);
    assert_eq!(r.p.stats().host_reads, 6, "the repeat's reads were summed");
}

#[test]
fn the_completion_buffer_overflows_with_a_named_error() {
    let mut r = rig(
        PolicyConfig {
            completions: 1,
            ..config()
        },
        vec![spec(2, 1)],
    );
    r.reserve(P0, key(0, 1));
    r.reserve(P0, key(0, 2));
    r.issue();
    let mut held = std::mem::take(&mut r.held).into_iter();
    let (pool, first) = held.next().unwrap();
    r.p.deliver_completion(pool, first).unwrap();
    let (pool, second) = held.next().unwrap();
    assert_eq!(
        r.p.deliver_completion(pool, second),
        Err(CacheError::CompletionBufferFull { capacity: 1 })
    );
    assert_eq!(
        r.p.deliver_completion(PoolId(9), decoy_tokens(1).remove(0)),
        Err(CacheError::UnknownPool(PoolId(9)))
    );
}

#[test]
fn the_parked_buffer_overflows_with_a_named_error_before_anything_changes() {
    let mut r = rig(
        PolicyConfig {
            parked: 1,
            ..config()
        },
        vec![spec(3, 3)],
    );
    for e in 0..3 {
        r.reserve(P0, key(0, e));
    }
    r.settle();
    // D withdraws one victim: the parked buffer is full.
    assert!(matches!(
        r.reserve(P0, key(0, 10)),
        Reservation::Deferred { .. }
    ));
    let phases: Vec<_> = (0..3).map(|s| r.p.slot_phase(P0, s)).collect();
    assert_eq!(
        r.p.reserve(P0, key(0, 11)),
        Err(CacheError::ParkedFull { capacity: 1 })
    );
    assert_eq!(r.p.lookup(key(0, 11)), Lookup::Absent);
    let after: Vec<_> = (0..3).map(|s| r.p.slot_phase(P0, s)).collect();
    assert_eq!(phases, after, "the refused reserve changed no slot");
    // The retirement drains the buffer.
    r.settle();
    assert!(matches!(
        r.reserve(P0, key(0, 11)),
        Reservation::Deferred { .. }
    ));
}

#[test]
fn the_consumer_table_overflows_with_a_named_error() {
    let mut r = rig(
        PolicyConfig {
            consumers: 1,
            ..config()
        },
        vec![spec(2, 1)],
    );
    assert_eq!(
        r.p.register_consumer(0, WindowConsumerPhase::Decode, 0),
        Err(CacheError::WindowNotOpen)
    );
    r.p.begin_window().unwrap();
    assert_eq!(
        r.p.register_consumer(9, WindowConsumerPhase::Decode, 0),
        Err(CacheError::QueueOutOfRange {
            queue: 9,
            queues: 2
        })
    );
    r.p.register_consumer(1, WindowConsumerPhase::Verify, 0)
        .unwrap();
    assert_eq!(
        r.p.register_consumer(0, WindowConsumerPhase::Draft, 1),
        Err(CacheError::ConsumerOverflow { capacity: 1 })
    );
    assert_eq!(r.p.window().max_consumers(), 1);
}

// ---------------------------------------------------------------------------------------------
// (c) Windows: the join, the cadence and poisoning
// ---------------------------------------------------------------------------------------------

#[test]
fn an_unjoined_consumer_holds_the_window_open() {
    let mut r = rig1(2, 1);
    assert_eq!(r.p.begin_window(), Ok(1));
    assert_eq!(r.p.begin_window(), Err(CacheError::WindowBusy));
    let id =
        r.p.register_consumer(0, WindowConsumerPhase::Decode, 3)
            .unwrap();
    assert_eq!(
        r.p.end_window(4),
        Err(CacheError::ConsumersOutstanding { outstanding: 1 })
    );
    // A closing window admits no new entrant, and nothing retires under it.
    assert_eq!(
        r.p.register_consumer(0, WindowConsumerPhase::Decode, 4),
        Err(CacheError::WindowClosed)
    );
    assert_eq!(r.p.plan_group(1, (0, 0)), Err(CacheError::WindowsOpen));
    assert_eq!(r.p.begin_window(), Err(CacheError::WindowBusy));
    r.p.join_consumer(id).unwrap();
    assert_eq!(
        r.p.join_consumer(id),
        Err(CacheError::ConsumerAlreadyJoined(0))
    );
    assert_eq!(r.p.end_window(4), Ok(None));
    assert_eq!(r.p.begin_window(), Ok(2));
}

#[test]
fn an_unknown_cancel_poisons_the_window_book_and_nothing_retires_again() {
    let mut r = rig1(2, 1);
    let (a, b) = (key(0, 1), key(0, 2));
    r.reserve(P0, a);
    r.settle();
    r.p.begin_window().unwrap();
    r.p.register_consumer(1, WindowConsumerPhase::Prefill, 0)
        .unwrap();
    // A reserve inside the window withdraws a victim: a retirement becomes pending.
    r.reserve(P0, key(0, 3));
    r.reserve(P0, b);
    assert!(r.p.needs_retirement());
    // The consumer never proved retirement: a retirement waits for the window.
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::WindowsOpen)
    );
    r.p.cancel_window(CancelOutcome::Unknown).unwrap();
    assert!(r.p.window().is_poisoned());
    assert_eq!(r.p.begin_window(), Err(CacheError::WindowPoisoned));
    assert_eq!(r.p.plan_group(1, (0, 0)), Err(CacheError::WindowPoisoned));
    assert_eq!(r.p.end_window(1), Err(CacheError::WindowPoisoned));
    assert_eq!(
        r.p.register_consumer(0, WindowConsumerPhase::Decode, 0),
        Err(CacheError::WindowPoisoned)
    );
    assert_eq!(
        r.p.record_use(used(P0, a, 1, 0)),
        Err(CacheError::WindowPoisoned)
    );
    let records = r.compute.records;
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::WindowPoisoned)
    );
    assert_eq!(r.p.finish_plan(P0), Err(CacheError::WindowPoisoned));
    assert_eq!(r.p.seal(P0), Err(CacheError::WindowPoisoned));
    assert_eq!(
        r.compute.records, records,
        "the poisoned image is never fenced or retired"
    );
    assert_eq!(r.p.bank(P0).unwrap().live_generations(), 1);
    assert_eq!(
        r.p.cancel_window(CancelOutcome::Retired),
        Err(CacheError::WindowPoisoned)
    );
}

#[test]
fn a_proven_cancel_closes_the_window_and_the_next_one_opens() {
    let mut r = rig1(2, 1);
    r.p.begin_window().unwrap();
    r.p.register_consumer(0, WindowConsumerPhase::Repair, 0)
        .unwrap();
    r.p.cancel_window(CancelOutcome::Retired).unwrap();
    assert!(r.p.window().is_idle());
    assert_eq!(
        r.p.cancel_window(CancelOutcome::Retired),
        Err(CacheError::WindowNotOpen)
    );
    assert_eq!(r.p.begin_window(), Ok(2));
}

#[test]
fn the_cadence_closes_a_group_on_ids_or_windows() {
    let mut r = rig(
        PolicyConfig {
            retire_ids: 10,
            max_windows: 3,
            ..config()
        },
        vec![spec(2, 1)],
    );
    // 4 + 4 ids: neither 10 ids nor 3 windows yet; the third window reaches both.
    assert_eq!(r.window_routed(&[], 4), None);
    assert_eq!(r.window_routed(&[], 4), None);
    assert_eq!(r.window_routed(&[], 4), Some(3));
    r.p.plan_group(3, (0, 0)).unwrap();
    // One window of 12 ids closes a group of one.
    assert_eq!(r.window_routed(&[], 12), Some(1));
    // A group never exceeds the log: the next window could overflow 512 ids.
    let mut r = rig1(2, 1);
    assert_eq!(r.window_routed(&[], 200), None);
    assert_eq!(r.window_routed(&[], 200), Some(2));
}

// ---------------------------------------------------------------------------------------------
// (d) S1c: pool teardown, backing access, the borrow protocol, peer mode, hold-back and the
//     admission strategy (a D2-style strategy reproduces boundary + hysteresis + reserve)
// ---------------------------------------------------------------------------------------------

/// A rig whose pools are all PEER pools.
fn rig_peer(config: PolicyConfig, specs: Vec<Spec>) -> Rig {
    let registry = BlockRegistry::new(0);
    let pools = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (
                PoolConfig {
                    slots: s.slots,
                    fills_per_window: s.fills,
                },
                pool_init(&registry, i as u64, s.slots, &s.seed),
            )
        })
        .collect();
    let modes = vec![PoolMode { peer: true }; specs.len()];
    Rig {
        p: ExpertCachePolicy::new_with_modes(config, pools, &modes)
            .expect("the peer policy builds"),
        copy: Dev::new(),
        compute: Dev::new(),
        held: Vec::new(),
    }
}

/// A peer board double: `released_through` answers what the test set and counts the asks.
struct Peers {
    through: Cell<u64>,
    asked: Cell<u32>,
}

impl Peers {
    fn new(through: u64) -> Self {
        Self {
            through: Cell::new(through),
            asked: Cell::new(0),
        }
    }
}

// SAFETY: a test double; each test chooses the acknowledgement it reports.
unsafe impl PeerReaderFence for Peers {
    fn released_through(&self, _owner: usize, _at: u64) -> u64 {
        self.asked.set(self.asked.get() + 1);
        self.through.get()
    }
}

fn two_pool_rig() -> Rig {
    let seed = (0..2u32).map(|e| (key(0, e), e)).collect();
    let mut r = rig(
        config(),
        vec![
            Spec {
                slots: 3,
                fills: 2,
                seed,
            },
            spec(2, 1),
        ],
    );
    r.settle();
    r
}

#[test]
fn a_pool_closes_only_when_quiescent_retired_and_without_undelivered_fills() {
    let mut r = two_pool_rig();
    // A live generation still names the seeds: the bank refuses and the pool survives intact.
    let q = r.p.quiesce(P0, &mut r.compute).expect("quiescence");
    assert!(matches!(
        r.p.close_pool(P0, q),
        Err(CacheError::Bank { pool: P0, .. })
    ));
    assert_eq!(r.p.slots(P0), Some(3));
    assert_eq!(r.p.count(P0, SlotPhase::Live), 2);
    assert!(r.p.bank(P0).is_some());
    // Retired, a fill decided and not taken is a named refusal.
    let summary = r.group(&[&[used(P0, key(1, 9), 5, 9)]], (0, 0));
    assert_eq!(summary.fills, 1);
    assert_eq!(r.p.fills_pending(), 1);
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    let q = r.p.quiesce(P0, &mut r.compute).expect("quiescence");
    assert!(matches!(
        r.p.close_pool(P0, q),
        Err(CacheError::FillsPending { pool: P0 })
    ));
    // Issued, the fill is a copy in flight the bank accepts at close; a quiescence from before the
    // issue is stale.
    let stale = r.p.quiesce(P0, &mut r.compute).expect("quiescence");
    r.issue();
    assert!(matches!(
        r.p.close_pool(P0, stale),
        Err(CacheError::Bank { pool: P0, .. })
    ));
    let q = r.p.quiesce(P0, &mut r.copy).expect("quiescence");
    let (block, reclaimed) = r.p.close_pool(P0, q).expect("the pool closes");
    assert!(block.free(reclaimed).is_ok());
    // Closed: nothing is left to ask, every call names the closed pool.
    assert!(r.p.bank(P0).is_none());
    assert_eq!(r.p.slots(P0), Some(0));
    assert_eq!(r.p.keys(), 0);
    assert_eq!(r.p.lookup(key(0, 0)), Lookup::Absent);
    assert_eq!(r.p.reserve(P0, key(7, 7)), Err(CacheError::PoolClosed(P0)));
    assert!(r.p.quiesce(P0, &mut r.compute).is_err());
    assert!(r.p.backing(P0).is_none());
    let (pool, token) = r.held.pop().expect("the fill's completion token");
    assert_eq!(
        r.p.deliver_completion(pool, token),
        Err(CacheError::PoolClosed(P0))
    );
    // The other pool is unaffected.
    let summary = r.group(&[&[used(P1, key(5, 1), 2, 0)]], (0, 0));
    assert_eq!((summary.misses, summary.fills), (1, 1));
    assert_eq!(r.p.fills_pending(), 1);
}

#[test]
fn closing_a_pool_forgets_its_keys_withdrawn_slots_and_deferred_fills() {
    let mut r = seeded_rig(2, 2);
    let b = key(0, 9);
    r.reserve(P0, b); // withdraws a published key; b's fill is deferred
    assert_eq!(r.p.keys(), 2);
    assert!(r.p.needs_retirement());
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    let q = r.p.quiesce(P0, &mut r.compute).expect("quiescence");
    let (block, reclaimed) = r.p.close_pool(P0, q).expect("the pool closes");
    assert!(block.free(reclaimed).is_ok());
    assert_eq!(r.p.keys(), 0);
    assert_eq!(r.p.lookup(b), Lookup::Absent);
    assert!(!r.p.needs_retirement());
    assert!(r.p.withdrawn().is_empty() && r.p.published().is_empty());
    assert_eq!(r.p.quarantined(P0), 0);
}

#[test]
fn the_backing_is_readable_exactly_while_the_generation_is_retired() {
    let mut r = seeded_rig(2, 1);
    assert!(
        r.p.backing(P0).is_none(),
        "a live generation holds the backing"
    );
    // A clean pool: the A2.1 retirement refuses, the unconditional one retires.
    assert_eq!(
        r.p.retire_generation(P0, &mut r.compute),
        Err(CacheError::GenerationState {
            pool: P0,
            state: "has nothing pending"
        })
    );
    let retired = r.p.stats().generations_retired;
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    assert_eq!(r.p.stats().generations_retired, retired + 1);
    let backing = r.p.backing(P0).expect("retired: the backing is writable");
    assert_eq!(backing.tables().len(), 1);
    assert_eq!(backing.tables()[0].base, TABLE);
    assert_eq!(r.p.generation_state(P0), Some("retired"));
    // Idempotent: no second retirement is counted.
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(false)
    );
    assert_eq!(r.p.stats().generations_retired, retired + 1);
    // An id that does not exceed the last sealed one is the bank's refusal; the state is kept.
    let last = r.p.last_sealed(P0).expect("a sealed generation");
    assert_eq!(last, GenerationId { epoch: 1, sub: 0 });
    assert!(matches!(
        r.p.seal_as(P0, last),
        Err(CacheError::Bank { pool: P0, .. })
    ));
    assert!(r.p.backing(P0).is_some());
    assert_eq!(r.p.generation_state(P0), Some("retired"));
    let resumed = GenerationId { epoch: 1, sub: 1 };
    assert_eq!(r.p.seal_as(P0, resumed), Ok(resumed));
    assert!(r.p.backing(P0).is_none());
    assert_eq!(r.p.last_sealed(P0), Some(resumed));
    // The next default seal continues above it.
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    assert_eq!(r.p.seal(P0), Ok(GenerationId { epoch: 2, sub: 0 }));
}

#[test]
fn the_borrow_protocol_lends_restores_and_reseals_the_same_membership() {
    let mut r = seeded_rig(3, 2);
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    let q = r.p.quiesce(P0, &mut r.copy).expect("quiescence");
    let loan = r.p.lend_scratch(P0, q).expect("the pool is lent");
    assert!(r.p.bank(P0).unwrap().is_lent());
    // Nothing plans or reserves while the block is lent.
    assert_eq!(
        r.p.plan_group(1, (0, 0)),
        Err(CacheError::PoolLent { pool: P0 })
    );
    assert_eq!(
        r.p.reserve(P0, key(9, 9)),
        Err(CacheError::PoolLent { pool: P0 })
    );
    let mut visited = Vec::new();
    let pending =
        r.p.issue_restore(P0, &loan, &mut r.copy, &mut |slot, occupant, spans| {
            assert!(!spans.is_empty());
            visited.push((slot, occupant));
            Ok(())
        })
        .expect("the restore issues");
    assert_eq!(
        visited,
        vec![(0, key(0, 0).occupant()), (1, key(0, 1).occupant())],
        "every occupied slot, ascending"
    );
    r.p.end_scratch(P0, loan, pending, &mut r.copy)
        .map_err(|(_loan, _restore, error)| error)
        .expect("the loan ends");
    assert!(!r.p.bank(P0).unwrap().is_lent());
    // Resume: the staged membership is sealed again as `{ epoch, sub + 1 }`.
    let last = r.p.last_sealed(P0).expect("a sealed generation");
    assert_eq!(last, GenerationId { epoch: 1, sub: 0 });
    let id = GenerationId {
        epoch: last.epoch,
        sub: last.sub + 1,
    };
    assert_eq!(r.p.seal_as(P0, id), Ok(id));
    assert_eq!(r.p.generation_state(P0), Some("live under its lease"));
    assert_eq!(r.p.bank(P0).unwrap().refs(0), 1);
    // The cache works again and its own epochs continue above the resumed one.
    r.group(&[&[used(P0, key(1, 9), 5, 9)]], (0, 0));
    assert_eq!(r.finish_group().fills.len(), 1);
    r.settle();
    assert_eq!(r.p.last_sealed(P0), Some(GenerationId { epoch: 2, sub: 0 }));
    assert!(matches!(
        r.p.lookup(key(1, 9)),
        Lookup::Published { slot: 2, .. }
    ));
}

#[test]
fn pool_modes_match_the_pools_and_set_the_bank_to_peer() {
    let registry = BlockRegistry::new(0);
    let pools = vec![(
        PoolConfig {
            slots: 2,
            fills_per_window: 1,
        },
        pool_init(&registry, 0, 2, &[]),
    )];
    assert!(matches!(
        ExpertCachePolicy::new_with_modes(config(), pools, &[]),
        Err(CacheError::PoolShape { .. })
    ));
    assert!(!rig1(2, 1).p.bank(P0).unwrap().is_peer());
    let r = rig_peer(config(), vec![spec(2, 1)]);
    assert!(r.p.bank(P0).unwrap().is_peer());
}

/// Quarantine-only admission: a victim is withdrawn without a target.
struct Quarantine {
    cap: usize,
}

impl AdmissionStrategy for Quarantine {
    fn pass(&mut self, _ctx: &PassContext) -> PassPlan {
        PassPlan {
            evict: EvictMode::Quarantine,
            quarantine_cap: self.cap,
            reserve_target: None,
        }
    }
}

#[test]
fn a_peer_pool_holds_a_quarantined_slot_until_every_peer_acknowledged_it() {
    let seed = (0..2u32).map(|e| (key(0, e), e)).collect();
    let mut r = rig_peer(
        config(),
        vec![Spec {
            slots: 2,
            fills: 1,
            seed,
        }],
    );
    r.settle();
    let peers = Peers::new(0);
    let at2 = Some((&peers as &dyn PeerReaderFence, 0usize, 2u64));
    let mut strategy = Quarantine { cap: usize::MAX };
    let x = key(0, 9);
    r.window(&[used(P0, x, 3, 9)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut strategy).unwrap();
    assert_eq!(
        (
            s.misses,
            s.withdrawn,
            s.quarantined,
            s.deferred,
            s.fills,
            s.skipped_full
        ),
        (1, 1, 1, 0, 0, 0)
    );
    // The colder seed (slot 1) was evicted; the candidate waits for a later plan.
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Withdrawn));
    assert_eq!(r.p.lookup(x), Lookup::Absent);
    assert_eq!(r.p.quarantined(P0), 1);
    assert!(r.p.slot_evicted_at(P0, 1).is_some());
    assert_eq!(r.p.bank(P0).unwrap().excluded_from_epoch(1), Some(2));
    assert_eq!(r.p.stats().quarantined, 1);
    // A pending last-read fence neither asks the peers nor spends the retirement.
    r.compute.hold = true;
    assert_eq!(
        r.p.retire_generation_with(P0, &mut r.compute, at2),
        Err(CacheError::WindowFencePending { pool: P0 })
    );
    assert_eq!(peers.asked.get(), 0);
    r.compute.hold = false;
    r.p.retire_generation_with(P0, &mut r.compute, at2).unwrap();
    assert_eq!(peers.asked.get(), 1, "the peers are asked once");
    // Nobody acknowledged: the slot stays withdrawn, and the seal does not wait for it.
    assert_eq!(r.p.finish_plan_with(P0, &mut strategy), Ok(0));
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Withdrawn));
    assert_eq!(r.p.seal(P0), Ok(GenerationId { epoch: 2, sub: 0 }));
    // The peers acknowledged through epoch 2: a retirement BEFORE the plan reclaims the slot
    // before admission, so the waiting candidate fills it in the same plan.
    peers.through.set(2);
    r.window(&[used(P0, x, 3, 9)]);
    let at3 = Some((&peers as &dyn PeerReaderFence, 0usize, 3u64));
    assert_eq!(r.p.retire_generation_now(P0, &mut r.compute, at3), Ok(true));
    assert_eq!(peers.asked.get(), 2);
    let s = r.p.plan_group_with(1, (0, 0), &mut strategy).unwrap();
    assert_eq!((s.fills, s.withdrawn, s.skipped_full), (1, 0, 0));
    assert!(!s.needs_retirement);
    assert!(matches!(
        r.p.lookup(x),
        Lookup::Pending {
            slot: 1,
            fill_generation: 2,
            ..
        }
    ));
    assert_eq!(r.p.quarantined(P0), 0);
    assert_eq!(r.p.seal(P0), Ok(GenerationId { epoch: 3, sub: 0 }));
}

#[test]
fn a_peer_pool_retired_without_peers_holds_every_withdrawn_slot() {
    let seed = (0..2u32).map(|e| (key(0, e), e)).collect();
    let mut r = rig_peer(
        config(),
        vec![Spec {
            slots: 2,
            fills: 1,
            seed,
        }],
    );
    r.settle();
    let mut strategy = Quarantine { cap: usize::MAX };
    r.window(&[used(P0, key(0, 9), 3, 9)]);
    r.p.plan_group_with(1, (0, 0), &mut strategy).unwrap();
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.finish_plan_with(P0, &mut strategy), Ok(0));
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Withdrawn));
    assert_eq!(r.p.seal(P0), Ok(GenerationId { epoch: 2, sub: 0 }));
}

/// Holds every withdrawn slot while `hold` is set and records what it was asked.
struct Hold {
    hold: bool,
    asked: Vec<SlotHold>,
}

impl AdmissionStrategy for Hold {
    fn hold_withdrawn(&mut self, hold: &SlotHold) -> bool {
        self.asked.push(*hold);
        self.hold
    }
}

#[test]
fn a_strategy_can_hold_a_withdrawn_slot_back_with_its_deferred_fill() {
    let mut r = seeded_rig(2, 2);
    let mut strategy = Hold {
        hold: true,
        asked: Vec::new(),
    };
    let b = key(0, 9);
    r.window(&[used(P0, b, 3, 9)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut strategy).unwrap();
    assert_eq!((s.withdrawn, s.deferred, s.quarantined), (1, 1, 0));
    r.p.retire_generation(P0, &mut r.compute).unwrap();
    assert_eq!(r.p.finish_plan_with(P0, &mut strategy), Ok(0));
    let asked = strategy.asked[0];
    assert_eq!(
        (asked.pool, asked.slot, asked.has_target, asked.peer_through),
        (P0, 1, true, None)
    );
    assert_eq!(asked.excluded_from, Some(2));
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Withdrawn));
    assert!(
        matches!(r.p.lookup(b), Lookup::Pending { slot: 1, .. }),
        "the deferred fill is kept"
    );
    assert_eq!(r.p.fills_pending(), 0);
    r.p.seal(P0).unwrap();
    // Released: the next plan, retired first, reclaims the slot and starts the kept fill.
    strategy.hold = false;
    assert_eq!(
        r.p.retire_generation_now(P0, &mut r.compute, None),
        Ok(true)
    );
    let s = r.p.plan_group_with(0, (0, 0), &mut strategy).unwrap();
    assert_eq!(s.fills, 1);
    assert_eq!(r.p.fills_pending(), 1);
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Filling));
    r.p.seal(P0).unwrap();
}

/// Records every call and answers from the fields: the hook contracts.
#[derive(Default)]
struct Probe {
    closed: bool,
    scores: HashMap<ExpertKey, f32>,
    extras: Vec<(ExpertKey, u32, f32)>,
    pushed: Vec<Result<bool, CacheError>>,
    plan: Option<PassPlan>,
    answers: Vec<bool>,
    observed: Vec<usize>,
    settled: Vec<bool>,
    passes: Vec<PassContext>,
    vetoes: Vec<(Candidate, Victim)>,
}

impl AdmissionStrategy for Probe {
    fn observe(&mut self, _ctx: &PlanContext, log: &[WindowUse]) {
        self.observed.push(log.len());
    }

    fn open_gate(&mut self, _ctx: &PlanContext) -> bool {
        !self.closed
    }

    fn score(&mut self, used: &WindowUse) -> f32 {
        self.scores
            .get(&used.key)
            .copied()
            .unwrap_or(used.reads as f32)
    }

    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        for &(key, rank, score) in &self.extras {
            self.pushed.push(sink.push(key, P0, rank, score));
        }
        Ok(())
    }

    fn pass(&mut self, ctx: &PassContext) -> PassPlan {
        self.passes.push(*ctx);
        self.plan.unwrap_or_default()
    }

    fn veto_victim(&mut self, candidate: &Candidate, victim: &Victim) -> bool {
        self.vetoes.push((*candidate, *victim));
        if self.answers.is_empty() {
            false
        } else {
            self.answers.remove(0)
        }
    }

    fn settled(&mut self, _ctx: &PlanContext, reached: bool) {
        self.settled.push(reached);
    }
}

/// The key the one fill of a plan went to (one free slot, one fill).
fn filled(r: &mut Rig) -> ExpertKey {
    let fills = r.finish_group().fills;
    assert_eq!(fills.len(), 1);
    fills[0].2
}

#[test]
fn scores_are_exact_f32_and_a_nan_score_ranks_last() {
    let (a, b) = (key(0, 1), key(0, 2));
    // `a` has the more reads, `b` the larger score: the score decides (not the u32 reads).
    let mut r = rig1(1, 1);
    let mut probe = Probe::default();
    probe.scores.insert(a, 0.25);
    probe.scores.insert(b, 0.3);
    r.window(&[used(P0, a, 7, 0), used(P0, b, 1, 1)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!((s.misses, s.fills, s.skipped_full), (2, 1, 0));
    assert_eq!(filled(&mut r), b);
    assert_eq!(probe.observed, vec![2]);
    // The default order picks `a` on the same stream.
    let mut r = rig1(1, 1);
    r.window(&[used(P0, a, 7, 0), used(P0, b, 1, 1)]);
    r.p.plan_group(1, (0, 0)).unwrap();
    assert_eq!(filled(&mut r), a);
    // A NaN score is the worst score (and never breaks the ordering).
    let mut r = rig1(1, 1);
    let mut probe = Probe::default();
    probe.scores.insert(a, f32::NAN);
    probe.scores.insert(b, 0.1);
    r.window(&[used(P0, a, 7, 0), used(P0, b, 1, 1)]);
    r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!(filled(&mut r), b);
    // Equal scores fall back to the ranking.
    let mut r = rig1(1, 1);
    let mut probe = Probe::default();
    probe.scores.insert(a, 1.0);
    probe.scores.insert(b, 1.0);
    r.window(&[used(P0, a, 7, 5), used(P0, b, 1, 4)]);
    r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!(filled(&mut r), b);
}

#[test]
fn extra_candidates_join_the_ordering_and_the_sink_ignores_bound_or_read_keys() {
    let seed = vec![(key(0, 0), 0)];
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 3,
            fills: 2,
            seed,
        }],
    );
    r.settle();
    let (a, x) = (key(1, 0), key(1, 1));
    let mut probe = Probe::default();
    probe.extras = vec![(x, 2, 10.0), (a, 3, 99.0), (key(0, 0), 0, 50.0)];
    r.window(&[used(P0, a, 5, 3)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!(
        probe.pushed,
        vec![Ok(true), Ok(false), Ok(false)],
        "a key the group read and a bound key are ignored"
    );
    assert_eq!((s.misses, s.fills), (1, 2));
    // `x` (score 10) goes before `a` (score 5): the lowest free slots, in order.
    let out = r.finish_group();
    assert_eq!(out.fills, vec![(P0, 1, x, 1), (P0, 2, a, 1)]);
}

struct BadExtras(PoolId);

impl AdmissionStrategy for BadExtras {
    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        sink.push(key(9, 9), self.0, 0, 1.0).map(|_| ())
    }
}

struct Flood;

impl AdmissionStrategy for Flood {
    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        for e in 0..600u32 {
            sink.push(key(8, e), P0, e, 1.0)?;
        }
        Ok(())
    }
}

#[test]
fn the_sink_refuses_an_unknown_pool_and_overflows_with_a_named_error() {
    let mut r = rig1(2, 1);
    r.window(&[]);
    assert_eq!(
        r.p.plan_group_with(1, (0, 0), &mut BadExtras(PoolId(9))),
        Err(CacheError::UnknownPool(PoolId(9)))
    );
    let mut r = rig1(2, 1);
    r.window(&[]);
    assert_eq!(
        r.p.plan_group_with(1, (0, 0), &mut Flood),
        Err(CacheError::UseLogFull { capacity: 512 })
    );
}

#[test]
fn a_closed_gate_admits_nothing_and_reports_it() {
    let mut r = rig1(2, 1);
    let mut probe = Probe {
        closed: true,
        ..Probe::default()
    };
    let a = key(0, 1);
    r.window(&[used(P0, a, 3, 0)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!((s.misses, s.fills, s.skipped_full), (1, 0, 0));
    assert_eq!(probe.settled, vec![false]);
    assert!(probe.passes.is_empty() && probe.vetoes.is_empty());
    assert_eq!(r.p.fills_pending(), 0);
    assert_eq!(r.p.stats().skipped_full, 0);
    probe.closed = false;
    r.window(&[used(P0, a, 3, 0)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!(s.fills, 1);
    assert_eq!(probe.settled, vec![false, true]);
}

#[test]
fn the_boundary_gate_counts_windows_not_groups() {
    let ctx = |before: u64, after: u64| PlanContext {
        windows: (after - before) as u32,
        windows_before: before,
        windows_after: after,
        now: 1,
    };
    assert!(!ctx(0, 1).crossed(2));
    assert!(ctx(1, 2).crossed(2));
    assert!(!ctx(2, 3).crossed(2));
    assert!(ctx(3, 4).crossed(2));
    // A group of several windows crosses the boundaries it covers.
    assert!(ctx(3, 5).crossed(4));
    assert!(!ctx(4, 7).crossed(4));
    assert!(ctx(7, 11).crossed(4));
    // N <= 1: every group.
    assert!(ctx(0, 1).crossed(1) && ctx(5, 6).crossed(0));
}

#[test]
fn a_veto_consumes_the_victim_and_counts_the_candidate_skipped() {
    let mut r = seeded_rig(2, 2);
    let (a, b) = (key(1, 0), key(1, 1));
    let mut probe = Probe {
        answers: vec![true, false],
        ..Probe::default()
    };
    r.window(&[used(P0, a, 5, 10), used(P0, b, 4, 11)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!(
        (s.skipped_full, s.withdrawn, s.deferred, s.fills),
        (1, 1, 1, 0)
    );
    assert_eq!(probe.vetoes.len(), 2);
    // The first candidate met the coldest victim (slot 1, rank 1) and was refused ...
    let (candidate, victim) = probe.vetoes[0];
    assert_eq!(
        (
            candidate.used.key,
            candidate.score,
            victim.slot,
            victim.key,
            victim.rank
        ),
        (a, 5.0, 1, key(0, 1), 1)
    );
    // ... the second one met the NEXT victim: the refused one was consumed.
    let (candidate, victim) = probe.vetoes[1];
    assert_eq!((candidate.used.key, victim.slot), (b, 0));
    assert_eq!(r.p.slot_phase(P0, 1), Some(SlotPhase::Live));
    assert_eq!(r.p.slot_phase(P0, 0), Some(SlotPhase::Withdrawn));
    assert!(matches!(r.p.lookup(b), Lookup::Pending { slot: 0, .. }));
    assert_eq!(r.p.stats().skipped_full, 1);
    // The pass context is what the pool looked like before the pass.
    assert_eq!(probe.passes.len(), 1);
    let c = probe.passes[0];
    assert_eq!(
        (
            c.pool,
            c.windows,
            c.total_budget,
            c.budget,
            c.candidates,
            c.free,
            c.filling,
            c.resident,
            c.quarantined
        ),
        (P0, 1, 2, 2, 2, 0, 0, 2, 0)
    );
    assert_eq!(probe.settled, vec![true]);
}

#[test]
fn a_quarantine_pass_withdraws_without_a_target_up_to_its_cap() {
    let mut r = seeded_rig(3, 3);
    let (a, b, c) = (key(1, 0), key(1, 1), key(1, 2));
    let mut probe = Probe {
        plan: Some(PassPlan {
            evict: EvictMode::Quarantine,
            quarantine_cap: 1,
            reserve_target: None,
        }),
        ..Probe::default()
    };
    r.window(&[used(P0, a, 5, 10), used(P0, b, 4, 11), used(P0, c, 3, 12)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    // Budget 2: `a` evicts into quarantine and waits; `b` finds the cap reached; `c` is past
    // the budget and not counted.
    assert_eq!(
        (
            s.withdrawn,
            s.quarantined,
            s.skipped_full,
            s.deferred,
            s.fills
        ),
        (1, 1, 1, 0, 0)
    );
    assert_eq!(r.p.lookup(a), Lookup::Absent);
    assert_eq!(r.p.quarantined(P0), 1);
    assert_eq!(r.p.stats().quarantined, 1);
    assert_eq!(probe.passes[0].candidates, 3);
}

#[test]
fn settled_says_whether_the_candidate_stage_was_reached_and_observe_sees_the_hits_too() {
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 2,
            fills: 1,
            seed: vec![(key(0, 0), 0)],
        }],
    );
    r.settle();
    let mut probe = Probe::default();
    // One hit and one miss: the strategy observes both; the miss takes the free slot.
    r.window(&[used(P0, key(0, 0), 4, 0), used(P0, key(0, 1), 2, 1)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!((s.hits, s.misses, s.fills), (1, 1, 1));
    assert_eq!(probe.observed, vec![2]);
    assert_eq!(probe.settled, vec![true]);
    // The fill is still in flight: no budget, no candidate stage.
    r.window(&[used(P0, key(0, 2), 2, 2)]);
    let s = r.p.plan_group_with(1, (0, 0), &mut probe).unwrap();
    assert_eq!((s.misses, s.fills, s.skipped_full), (1, 0, 0));
    assert_eq!(probe.settled, vec![true, false]);
    // A publication-only plan judges and admits nothing: no hook runs.
    r.p.plan_group_with(0, (0, 0), &mut probe).unwrap();
    assert_eq!((probe.observed.len(), probe.settled.len()), (2, 2));
}

// ------------------------------------------------------------------------------------------
// A D2-style strategy over the core: boundary heat (decayed, f32), the boundary gate, the
// hysteresis veto, the peer quarantine cap and the reserve. The expectations below are worked
// by hand from the `TailPolicy::retire_inner` rules.
// ------------------------------------------------------------------------------------------

struct D2Like {
    peer: bool,
    every: u64,
    heat: HashMap<ExpertKey, f32>,
    ranks: HashMap<ExpertKey, u32>,
    settled: Vec<bool>,
    vetoed: Vec<(ExpertKey, ExpertKey)>,
}

impl D2Like {
    fn new(peer: bool, every: u64, ranks: &[(ExpertKey, u32)]) -> Self {
        Self {
            peer,
            every,
            heat: HashMap::new(),
            ranks: ranks.iter().copied().collect(),
            settled: Vec::new(),
            vetoed: Vec::new(),
        }
    }

    fn boundary(&self) -> bool {
        self.every > 1
    }

    fn heat_of(&self, key: &ExpertKey) -> f32 {
        self.heat.get(key).copied().unwrap_or(0.0)
    }
}

impl AdmissionStrategy for D2Like {
    fn observe(&mut self, _ctx: &PlanContext, log: &[WindowUse]) {
        if self.boundary() {
            for used in log {
                *self.heat.entry(used.key).or_insert(0.0) += used.reads as f32;
            }
        }
    }

    fn open_gate(&mut self, ctx: &PlanContext) -> bool {
        !self.boundary() || ctx.crossed(self.every)
    }

    fn score(&mut self, used: &WindowUse) -> f32 {
        if self.boundary() {
            self.heat_of(&used.key)
        } else {
            used.reads as f32
        }
    }

    fn extra_candidates(
        &mut self,
        _ctx: &PlanContext,
        sink: &mut CandidateSink<'_>,
    ) -> Result<(), CacheError> {
        if self.boundary() {
            for (&key, &heat) in &self.heat {
                let rank = self.ranks.get(&key).copied().unwrap_or(u32::MAX);
                sink.push(key, P0, rank, heat)?;
            }
        }
        Ok(())
    }

    fn pass(&mut self, ctx: &PassContext) -> PassPlan {
        if !self.peer {
            return PassPlan::default();
        }
        PassPlan {
            evict: EvictMode::Quarantine,
            quarantine_cap: 3 * ctx.total_budget,
            reserve_target: (!self.boundary() && ctx.candidates >= ctx.budget)
                .then_some(2 * ctx.budget),
        }
    }

    fn veto_victim(&mut self, candidate: &Candidate, victim: &Victim) -> bool {
        if !self.boundary() {
            return false;
        }
        let veto = candidate.score < self.heat_of(&victim.key) * 1.10;
        if veto {
            self.vetoed.push((candidate.used.key, victim.key));
        }
        veto
    }

    fn settled(&mut self, _ctx: &PlanContext, reached: bool) {
        self.settled.push(reached);
        if self.boundary() && reached {
            self.heat.retain(|_, heat| {
                *heat *= 0.8;
                *heat > 0.01
            });
        }
    }
}

fn near(got: f32, want: f32) -> bool {
    (got - want).abs() < 1e-4
}

#[test]
fn a_d2_style_boundary_strategy_admits_on_the_boundary_with_heat_and_hysteresis() {
    // Two slots hold the seeds s0 (rank 0) and s1 (rank 1); F = 2.
    let seed = (0..2u32).map(|e| (key(0, e), e)).collect();
    let mut r = rig(
        config(),
        vec![Spec {
            slots: 2,
            fills: 2,
            seed,
        }],
    );
    r.settle();
    let (s0, s1) = (key(0, 0), key(0, 1));
    let (x, y, z) = (key(0, 2), key(0, 3), key(0, 4));
    let mut d2 = D2Like::new(false, 2, &[(x, 2), (y, 3), (z, 4)]);

    // Group 1 (windows 0 -> 1): the boundary N = 2 is not crossed. `s0` hits (20 reads), `x`
    // misses; the heat is kept, nothing is admitted, nothing decays.
    r.window(&[used(P0, s0, 20, 0), used(P0, x, 1, 2)]);
    let g1 = r.p.plan_group_with(1, (0, 0), &mut d2).unwrap();
    assert_eq!(
        (g1.hits, g1.misses, g1.fills, g1.withdrawn, g1.skipped_full),
        (1, 1, 0, 0, 0)
    );
    assert_eq!(d2.settled, vec![false]);
    assert!(near(d2.heat_of(&s0), 20.0) && near(d2.heat_of(&x), 1.0));
    assert!(r.finish_group().fills.is_empty());

    // Group 2 (1 -> 2) crosses the boundary: `x` (heat 2.0) is the one candidate; the LRU victim
    // is `s1` (never used, heat 0): 2.0 >= 0 x 1.10, admitted (deferred into the victim's slot).
    r.window(&[used(P0, x, 1, 2)]);
    let g2 = r.p.plan_group_with(1, (0, 0), &mut d2).unwrap();
    assert_eq!(
        (g2.misses, g2.withdrawn, g2.deferred, g2.skipped_full),
        (1, 1, 1, 0)
    );
    assert_eq!(d2.settled, vec![false, true]);
    // The heat decays by 0.8 once per admitting boundary.
    assert!(near(d2.heat_of(&s0), 16.0) && near(d2.heat_of(&x), 1.6));
    let out = r.finish_group();
    assert_eq!(
        out.withdrawn,
        vec![Withdrawn {
            key: s1,
            pool: P0,
            slot: 1
        }]
    );
    assert_eq!(out.fills, vec![(P0, 1, x, 2)]);
    r.settle();
    assert!(matches!(r.p.lookup(x), Lookup::Published { slot: 1, .. }));

    // Group 3 (2 -> 3): no boundary. `y` and `z` only accumulate heat.
    r.window(&[used(P0, y, 2, 3), used(P0, z, 9, 4)]);
    let g3 = r.p.plan_group_with(1, (0, 0), &mut d2).unwrap();
    assert_eq!(
        (g3.misses, g3.fills, g3.withdrawn, g3.skipped_full),
        (2, 0, 0, 0)
    );
    assert_eq!(d2.settled, vec![false, true, false]);
    assert!(r.finish_group().fills.is_empty());

    // Group 4 (3 -> 4) crosses. Only `y` misses (heat 4.0); `z` (heat 9.0, not read this window)
    // is an EXTRA candidate and orders first. The two LRU victims are `s0` (last used in group
    // 1, heat 16) then `x` (published after group 2, heat 1.6):
    //   z vs s0: 9.0 < 16 x 1.10  -> vetoed (the victim `s0` is consumed, skipped_full);
    //   y vs x : 4.0 >= 1.6 x 1.10 -> admitted into x's slot.
    r.window(&[used(P0, y, 2, 3)]);
    let g4 = r.p.plan_group_with(1, (0, 0), &mut d2).unwrap();
    assert_eq!(
        (
            g4.misses,
            g4.withdrawn,
            g4.deferred,
            g4.skipped_full,
            g4.fills
        ),
        (1, 1, 1, 1, 0)
    );
    assert_eq!(d2.vetoed, vec![(z, s0)]);
    assert_eq!(d2.settled, vec![false, true, false, true]);
    assert!(matches!(r.p.lookup(s0), Lookup::Published { slot: 0, .. }));
    assert!(matches!(r.p.lookup(y), Lookup::Pending { slot: 1, .. }));
    assert_eq!(r.p.lookup(z), Lookup::Absent);
    let out = r.finish_group();
    assert_eq!(
        out.withdrawn,
        vec![Withdrawn {
            key: x,
            pool: P0,
            slot: 1
        }]
    );
    assert_eq!(out.fills, vec![(P0, 1, y, 3)]);
    assert_eq!(r.p.stats().skipped_full, 1);
}

/// One D2 peer retirement: complete the earlier fills, retire the generation with the peers'
/// release FIRST, plan (which reclaims what they acknowledged, then admits), rewrite, reclaim
/// what is not held, seal as epoch `j` and start the fills.
fn peer_step(
    r: &mut Rig,
    peers: &Peers,
    j: u64,
    uses: &[WindowUse],
    d2: &mut D2Like,
) -> (PlanSummary, Outcome) {
    r.window(uses);
    r.deliver_held();
    let fence = Some((peers as &dyn PeerReaderFence, 0usize, j));
    r.p.retire_generation_now(P0, &mut r.compute, fence)
        .expect("the generation retires");
    let summary = r.p.plan_group_with(1, (0, 0), d2).expect("the group plans");
    let published = r.p.published().to_vec();
    let withdrawn = r.p.withdrawn().to_vec();
    r.p.finish_plan_with(P0, d2).expect("the plan finishes");
    r.p.seal_as(P0, GenerationId { epoch: j, sub: 0 })
        .expect("the next generation seals");
    let fills = r.issue();
    (
        summary,
        Outcome {
            published,
            withdrawn,
            fills,
        },
    )
}

#[test]
fn a_d2_style_peer_strategy_keeps_the_reserve_under_saturated_demand() {
    // Eight slots s0..s7 (ranks 0..7), F = 2: the fill budget of a one-window group is 2, the
    // reserve target 2 x 2 = 4, the quarantine cap 3 x 2 = 6.
    let seed: Vec<(ExpertKey, u32)> = (0..8u32).map(|e| (key(0, e), e)).collect();
    let mut r = rig_peer(
        config(),
        vec![Spec {
            slots: 8,
            fills: 2,
            seed,
        }],
    );
    r.settle();
    let (a, b, c) = (key(1, 0), key(1, 1), key(1, 2));
    let uses = [used(P0, a, 5, 10), used(P0, b, 4, 11), used(P0, c, 3, 12)];
    let mut d2 = D2Like::new(true, 0, &[]);
    let peers = Peers::new(0);
    let evicted = |slot: u32| Withdrawn {
        key: key(0, slot),
        pool: P0,
        slot,
    };

    // Retirement 2 (nothing acknowledged): three candidates >= the budget of 2 saturate the
    // reserve. Each of the two taken candidates finds no free slot, `free + quarantined` below 4,
    // so it evicts the coldest resident (s7, then s6) into quarantine and waits.
    let (s, out) = peer_step(&mut r, &peers, 2, &uses, &mut d2);
    assert_eq!(
        (
            s.misses,
            s.withdrawn,
            s.quarantined,
            s.fills,
            s.skipped_full
        ),
        (3, 2, 2, 0, 0)
    );
    assert_eq!(out.withdrawn, vec![evicted(7), evicted(6)]);
    assert!(out.fills.is_empty());
    assert_eq!(r.p.quarantined(P0), 2);

    // Retirement 3 (still nothing): two more (s5, s4) reach the reserve target of 4.
    let (s, out) = peer_step(&mut r, &peers, 3, &uses, &mut d2);
    assert_eq!(
        (s.withdrawn, s.quarantined, s.fills, s.skipped_full),
        (2, 2, 0, 0)
    );
    assert_eq!(out.withdrawn, vec![evicted(5), evicted(4)]);
    assert_eq!(r.p.quarantined(P0), 4);

    // Retirement 4: the peers acknowledged through 2, which covers s7 and s6 (evicted at
    // retirement 2) but not s5 and s4 (evicted at 3). The plan reclaims s6 and s7 BEFORE it
    // admits: `a` takes s6 and `b` takes s7, and each also evicts one more victim (s3, s2) to
    // keep `free + quarantined` at 4.
    peers.through.set(2);
    let (s, out) = peer_step(&mut r, &peers, 4, &uses, &mut d2);
    assert_eq!(
        (s.fills, s.withdrawn, s.quarantined, s.skipped_full),
        (2, 2, 2, 0)
    );
    assert_eq!(out.withdrawn, vec![evicted(3), evicted(2)]);
    assert_eq!(out.fills, vec![(P0, 6, a, 2), (P0, 7, b, 2)]);
    assert_eq!(r.p.quarantined(P0), 4);
    assert_eq!(r.p.slot_phase(P0, 6), Some(SlotPhase::Filling));

    // Retirement 5: acknowledged through 3 frees s5 and s4. The copies of `a` and `b` complete
    // and publish; only `c` is a candidate (1 < the budget of 2: no saturation, no reserve), and
    // it takes the lowest free slot, s4.
    peers.through.set(3);
    let (s, out) = peer_step(&mut r, &peers, 5, &uses, &mut d2);
    assert_eq!(
        (s.misses, s.published, s.fills, s.withdrawn, s.skipped_full),
        (3, 2, 1, 0, 0)
    );
    assert_eq!(
        out.published,
        vec![
            Published {
                key: a,
                pool: P0,
                slot: 6,
                fill_generation: 2
            },
            Published {
                key: b,
                pool: P0,
                slot: 7,
                fill_generation: 2
            }
        ]
    );
    assert_eq!(out.fills, vec![(P0, 4, c, 2)]);
    assert_eq!(r.p.quarantined(P0), 2);
    assert_eq!(r.p.stats().quarantined, 6);
    assert_eq!(
        peers.asked.get(),
        4,
        "the peers are asked once per retirement"
    );
}
