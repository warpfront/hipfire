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

use std::cmp::Reverse;
use std::collections::HashMap;

use railgun::sealed_table::generation::{
    BlockRegistry, Expert, FenceBackend, FenceFailure, FenceState, FillComplete, FillTicket,
    SlotSpan,
};
use railgun::window::Quiescence;
use railgun::Binding;

use super::{
    CacheError, CacheStats, CancelOutcome, ExpertCachePolicy, ExpertKey, FillOp, Lookup,
    PlanSummary, PolicyConfig, PoolConfig, PoolId, PoolInit, Published, Reservation, SlotPhase,
    WindowConsumerPhase, WindowUse, Withdrawn,
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
