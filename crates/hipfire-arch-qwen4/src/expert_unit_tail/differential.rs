// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Differential tests: the migrated [`TailPolicy`] (driven through the retirement protocol by
//! [`Rig`]) against the FROZEN beta policy ([`beta_ref::TailPolicy`]) on seeded random traces,
//! in lockstep. Both get the same seed units, the same window uses, the same completion
//! `(slot, generation)` pairs (delayed by a random latency, shuffled, with stale pairs mixed
//! in), the same cadence group sizes and, in PEER mode, the same `released_through` value.
//! Every retirement must decide identically (publish / unpublish / fills) and leave identical
//! counters and per-slot state.

use std::collections::{BTreeMap, HashMap};

use super::beta_ref;
use super::rig::{Rig, RigRetire};
use super::{EpUnitTailConfig, SlotPhase, UnitKey, WindowUse};

/// Retirements per (config, seed).
const RETIREMENTS: u64 = 2_000;
/// Trace seeds per config.
const SEEDS: [u64; 3] = [0x5eed_0001, 0xd1ff_0002, 0xbeef_0003];

/// `(T, F, b)` of every config the axes sweep.
const CONFIGS: [(usize, usize, u32); 8] = [
    (4, 2, 1),
    (8, 3, 1),
    (16, 4, 1),
    (24, 2, 1),
    (12, 6, 3),
    (5, 2, 4),
    (64, 8, 1),
    (3, 1, 2),
];

/// Deterministic splitmix64 generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True with probability `num / den`.
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// The deterministic `released_through` source of PEER mode: a nondecreasing `acked` counter
/// that never exceeds `m - 1`, with long withheld stretches (no ack for many retirements, so
/// quarantine saturates) and bursts that release many retirements at once.
struct Acks {
    acked: u64,
    /// Retirements left in the current withheld stretch.
    withheld: u64,
}

impl Acks {
    fn through(&mut self, rng: &mut Rng, m: u64) -> u64 {
        let cap = m.saturating_sub(1);
        if self.withheld > 0 {
            self.withheld -= 1;
        } else {
            match rng.below(20) {
                0 | 1 => self.withheld = 5 + rng.below(30),
                2 => self.acked = cap,
                3..=14 => self.acked = (self.acked + 1 + rng.below(2)).min(cap),
                _ => {}
            }
        }
        self.acked = self.acked.min(cap);
        self.acked
    }
}

/// Totals of a run, summed over seeds.
#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    retirements: u64,
    fills_issued: u64,
    fills_published: u64,
    evictions: u64,
    quarantined: u64,
    skipped_full: u64,
    stale_completions: u64,
}

impl Totals {
    fn add(&mut self, other: &Totals) {
        self.retirements += other.retirements;
        self.fills_issued += other.fills_issued;
        self.fills_published += other.fills_published;
        self.evictions += other.evictions;
        self.quarantined += other.quarantined;
        self.skipped_full += other.skipped_full;
        self.stale_completions += other.stale_completions;
    }
}

/// A deliberate divergence injected into the BETA side only (the positive control).
#[derive(Clone, Copy, Debug)]
enum Perturb {
    None,
    /// Beta sees a different `core` at this retirement.
    Core(u64),
    /// Beta gets one extra stale completion at this retirement.
    ExtraStale(u64),
}

fn phase_b(phase: beta_ref::SlotPhase) -> u8 {
    match phase {
        beta_ref::SlotPhase::Free => 0,
        beta_ref::SlotPhase::Filling => 1,
        beta_ref::SlotPhase::Live => 2,
        beta_ref::SlotPhase::Quarantined => 3,
    }
}

fn phase_m(phase: SlotPhase) -> u8 {
    match phase {
        SlotPhase::Free => 0,
        SlotPhase::Filling => 1,
        SlotPhase::Live => 2,
        SlotPhase::Quarantined => 3,
    }
}

/// Fail with `what` unless `beta == rig`.
fn same<T: PartialEq + std::fmt::Debug>(
    at: &str,
    what: &str,
    beta: &T,
    rig: &T,
) -> Result<(), String> {
    if beta == rig {
        Ok(())
    } else {
        Err(format!(
            "{at}: {what} differ\n  beta: {beta:?}\n  rig:  {rig:?}"
        ))
    }
}

/// Compare one retirement's decisions, the counters and every slot's state.
fn compare(
    at: &str,
    beta: &beta_ref::TailPolicy,
    rig: &Rig,
    beta_out: &beta_ref::Retire,
    rig_out: &RigRetire,
) -> Result<(), String> {
    same(at, "publish", &beta_out.publish, &rig_out.publish)?;
    same(at, "unpublish", &beta_out.unpublish, &rig_out.unpublish)?;
    let beta_fills: Vec<_> = beta_out
        .fills
        .iter()
        .map(|op| (op.slot, op.unit, op.generation))
        .collect();
    same(at, "fills", &beta_fills, &rig_out.fills)?;
    same(at, "stats", beta.stats(), rig.stats())?;
    for slot in 0..beta.slots() {
        same(
            at,
            &format!("slot {slot} phase"),
            &phase_b(beta.slot_phase(slot)),
            &phase_m(rig.slot_phase(slot)),
        )?;
        same(
            at,
            &format!("slot {slot} generation"),
            &beta.slot_generation(slot),
            &rig.slot_generation(slot),
        )?;
        same(
            at,
            &format!("slot {slot} unit"),
            &beta.slot_unit(slot),
            &rig.slot_unit(slot),
        )?;
    }
    for (b, m) in [
        (beta_ref::SlotPhase::Free, SlotPhase::Free),
        (beta_ref::SlotPhase::Filling, SlotPhase::Filling),
        (beta_ref::SlotPhase::Live, SlotPhase::Live),
        (beta_ref::SlotPhase::Quarantined, SlotPhase::Quarantined),
    ] {
        same(at, &format!("count({m:?})"), &beta.count(b), &rig.count(m))?;
    }
    Ok(())
}

/// One window's distinct non-core units over layers 0..4 x experts 0..64: mostly from a hot set
/// that drifts with the retirement index, sorted by unit, reads 1..=3 (summed over repeats).
fn window_uses(rng: &mut Rng, j: u64, config: &EpUnitTailConfig) -> Vec<WindowUse> {
    let base = (j / 40 * 5 % 64) as u64;
    let hot = (10 + config.tail_units as u64 / 2).min(40);
    let count = 1 + rng.below(4 * config.fill_per_step as u64 + 8);
    let mut uses: BTreeMap<UnitKey, u32> = BTreeMap::new();
    for _ in 0..count {
        let expert = if rng.chance(3, 4) {
            (base + rng.below(hot)) % 64
        } else {
            rng.below(64)
        };
        let unit = (rng.below(4) as u16, expert as u16);
        *uses.entry(unit).or_insert(0) += 1 + rng.below(3) as u32;
    }
    uses.into_iter()
        .map(|(unit, reads)| WindowUse { unit, reads })
        .collect()
}

/// Run one (config, frozen, peer, seed) trace in lockstep; `Err` names the first divergence.
fn run_case(
    (t, f, b): (usize, usize, u32),
    frozen: bool,
    peer: bool,
    seed: u64,
    retirements: u64,
    perturb: Perturb,
) -> Result<Totals, String> {
    let config = EpUnitTailConfig {
        tail_units: t,
        fill_per_step: f,
        boundary_every: b,
    };
    let label = format!("config=({t},{f},{b}) frozen={frozen} peer={peer} seed={seed:#x}");
    let mut rng = Rng(seed);

    // A random ranking priority over half of the unit universe.
    let mut priority: HashMap<UnitKey, u32> = HashMap::new();
    for layer in 0..4u16 {
        for expert in 0..64u16 {
            if rng.chance(1, 2) {
                priority.insert((layer, expert), rng.below(300) as u32);
            }
        }
    }

    let mut beta = beta_ref::TailPolicy::new(config, frozen, priority.clone());
    if peer {
        beta.enable_peer_quarantine();
    }
    let mut rig = Rig::with_peer(config, frozen, priority, peer);

    // Warm start with a random number of units (with repeats, possibly more than T).
    let seeds: Vec<UnitKey> = (0..rng.below(t as u64 + 4))
        .map(|_| (rng.below(2) as u16, rng.below(t as u64 + 2) as u16))
        .collect();
    let beta_seed: Vec<_> = beta
        .seed(&seeds)
        .iter()
        .map(|op| (op.slot, op.unit, op.generation))
        .collect();
    same(&label, "seed fills", &beta_seed, &rig.seed(&seeds))?;

    let mut acks = Acks {
        acked: 0,
        withheld: 0,
    };
    // Issued fills not yet completed: `(first retirement that may report it, slot, generation)`.
    let mut pending: Vec<(u64, usize, u64)> = Vec::new();

    for j in 1..=retirements {
        let at = format!("{label} retirement {j}");
        let windows = 1 + rng.below(8) as u32;
        let core = (rng.below(20), rng.below(200));
        let uses = window_uses(&mut rng, j, &config);

        // Completions due now, shuffled; maybe one stale pair the policy must ignore.
        let mut completed: Vec<(usize, u64)> = Vec::new();
        pending.retain(|&(due, slot, generation)| {
            if due <= j {
                completed.push((slot, generation));
                false
            } else {
                true
            }
        });
        rng.shuffle(&mut completed);
        if rng.chance(1, 9) {
            let slot = rng.below(t as u64) as usize;
            let generation = if rng.chance(1, 2) {
                1 << 40
            } else {
                beta.slot_generation(slot).saturating_sub(1)
            };
            let at_end = rng.below(completed.len() as u64 + 1) as usize;
            completed.insert(at_end, (slot, generation));
        }

        let through = if peer { acks.through(&mut rng, j) } else { 0 };
        let beta_core = match perturb {
            Perturb::Core(n) if n == j => (core.0 + 1, core.1),
            _ => core,
        };
        let mut beta_completed = completed.clone();
        if matches!(perturb, Perturb::ExtraStale(n) if n == j) {
            beta_completed.push((0, 1 << 41));
        }

        let (beta_out, rig_out) = if peer {
            let ask = move |expect: u64| {
                move |m: u64| {
                    assert_eq!(m, expect, "retirement index asked of released_through");
                    through
                }
            };
            (
                beta.retire_peer_windows(beta_core, &uses, &beta_completed, windows, ask(j)),
                rig.retire_peer_windows(core, &uses, &completed, windows, ask(j)),
            )
        } else {
            (
                beta.retire_windows(beta_core, &uses, &beta_completed, windows),
                rig.retire_windows(core, &uses, &completed, windows),
            )
        };
        compare(&at, &beta, &rig, &beta_out, &rig_out)?;

        // Every issued fill completes after a random latency of 0..=5 retirements.
        for &(slot, _, generation) in &rig_out.fills {
            pending.push((j + 1 + rng.below(6), slot, generation));
        }
    }

    let refusals = rig.policy.refusals();
    if refusals.0 != 0 {
        return Err(format!(
            "{label}: the bank refused transitions: {refusals:?}"
        ));
    }
    let stats = rig.stats();
    Ok(Totals {
        retirements: stats.retirements,
        fills_issued: stats.fills_issued,
        fills_published: stats.fills_published,
        evictions: stats.evictions,
        quarantined: stats.quarantined,
        skipped_full: stats.skipped_full,
        stale_completions: stats.stale_completions,
    })
}

/// Every seed of `configs` at one (frozen, peer); prints one summary line per config.
fn sweep(
    configs: &[(usize, usize, u32)],
    frozen: bool,
    peer: bool,
) -> Result<Vec<((usize, usize, u32), Totals)>, String> {
    let mut out = Vec::new();
    for &config in configs {
        let mut total = Totals::default();
        for seed in SEEDS {
            total.add(&run_case(
                config,
                frozen,
                peer,
                seed,
                RETIREMENTS,
                Perturb::None,
            )?);
        }
        eprintln!(
            "differential config=({},{},{}) frozen={frozen} peer={peer} retirements={} \
             fills_issued={} fills_published={} evictions={} quarantined={} skipped_full={} \
             stale_completions={}",
            config.0,
            config.1,
            config.2,
            total.retirements,
            total.fills_issued,
            total.fills_published,
            total.evictions,
            total.quarantined,
            total.skipped_full,
            total.stale_completions,
        );
        assert!(
            total.retirements >= 2_000,
            "{config:?}: too few retirements"
        );
        out.push((config, total));
    }
    Ok(out)
}

fn sum(rows: &[((usize, usize, u32), Totals)], field: impl Fn(&Totals) -> u64) -> u64 {
    rows.iter().map(|(_, t)| field(t)).sum()
}

#[test]
fn differential_peer_off() {
    let rows = sweep(&CONFIGS, false, false).unwrap();
    assert!(sum(&rows, |t| t.evictions) > 0, "no evictions");
    assert!(sum(&rows, |t| t.fills_published) > 0, "no fills published");
    assert!(
        sum(&rows, |t| t.stale_completions) > 0,
        "no stale completions"
    );
    assert_eq!(sum(&rows, |t| t.quarantined), 0, "quarantine without peers");
}

#[test]
fn differential_peer_on() {
    let rows = sweep(&CONFIGS, false, true).unwrap();
    assert!(sum(&rows, |t| t.evictions) > 0, "no evictions");
    assert!(sum(&rows, |t| t.fills_published) > 0, "no fills published");
    assert!(sum(&rows, |t| t.quarantined) > 0, "nothing quarantined");
    assert!(
        sum(&rows, |t| t.skipped_full) > 0,
        "quarantine never saturated"
    );
    assert!(
        sum(&rows, |t| t.stale_completions) > 0,
        "no stale completions"
    );
}

#[test]
fn differential_frozen() {
    // A frozen tail admits nothing new; both modes must still agree on every retirement.
    sweep(&CONFIGS, true, false).unwrap();
    sweep(&CONFIGS, true, true).unwrap();
}

#[test]
fn differential_boundary_mode() {
    // b > 1: admit only every b-th window, decayed-heat ranking with hysteresis.
    let boundary: Vec<_> = CONFIGS.iter().copied().filter(|c| c.2 > 1).collect();
    assert!(!boundary.is_empty());
    for peer in [false, true] {
        let rows = sweep(&boundary, false, peer).unwrap();
        assert!(
            sum(&rows, |t| t.evictions) > 0,
            "no evictions (peer={peer})"
        );
        assert!(
            sum(&rows, |t| t.fills_published) > 0,
            "no fills (peer={peer})"
        );
        assert!(
            sum(&rows, |t| t.stale_completions) > 0,
            "no stale (peer={peer})"
        );
        if peer {
            assert!(sum(&rows, |t| t.quarantined) > 0, "nothing quarantined");
        }
    }
}

/// Positive control: the harness reports a divergence when one decision input of the beta side
/// is perturbed, and agrees on the very same trace unperturbed.
#[test]
fn differential_detects_divergence() {
    let config = (8, 3, 1);
    let seed = SEEDS[0];
    for peer in [false, true] {
        run_case(config, false, peer, seed, 300, Perturb::None).expect("unperturbed run agrees");
        for perturb in [Perturb::Core(150), Perturb::ExtraStale(150)] {
            let error = run_case(config, false, peer, seed, 300, perturb)
                .expect_err("a perturbed beta input must be reported");
            assert!(
                error.contains("retirement 150"),
                "reported at the wrong retirement: {error}"
            );
            assert!(error.contains("stats differ"), "{error}");
        }
    }
}
