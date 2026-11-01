// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only logic of the sampled-MTP **lane suite** (`sampled_mtp_distribution_hw`).
//!
//! `HIPFIRE_SAMPLED_MTP_EXECUTOR=lanes` runs a request cohort through the
//! continuous-batching lane executor instead of the singleton drivers. This
//! module holds everything about that suite that needs no GPU: the
//! executor / topology / lane-count parsers, the declared lane arm set and its
//! per-arm route, the per-arm environment contract, request-seed derivation,
//! shard-file/range parsing, the recorded `lane_suite` config block, the
//! shard **merge key** and the merge planner that refuses mixed or incomplete
//! shard sets, per-lane row splitting and the route-coverage rules over an
//! arm's evidence. Its tests are CPU-only; no test here touches a device.
//!
//! # Lane arms
//!
//! | arm | route | env |
//! |---|---|---|
//! | `lane_ar` | `ar` | no route knob, floor on (target-only lanes; the reference) |
//! | `lane_batched` | `batched` | `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_MTP_AR_FLOOR=0` |
//! | `lane_adaptive` | `adaptive` | no route knob, floor on (calibration, probes, retirement) |
//! | `lane_copy` | `copy` | no route knob, `HIPFIRE_MTP_NGRAM_COPY=on`; explicit one-hot copy proposals |
//! | `lane_cross` | `cross` | as `lane_batched` (`HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_MTP_AR_FLOOR=0`) but `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=3` (the shipped `lane_switch::DEFAULT_MAX_ACTIVE`, pinned so the identity records it) and staggered admission; `ep2`, 8 lanes only |
//!
//! The native arms (`lane_batched`, `lane_adaptive`, `lane_cross`) also pin
//! `HIPFIRE_EP_MTP_LANE_K` to the cohort's lane draft depth, chosen with
//! `HIPFIRE_SAMPLED_MTP_LANE_K` (`2` = the shipped default, or `3`; see
//! [`parse_lane_k`]). The depth is part of every shard's merge key: a K=2 set
//! never merges with a K=3 set. `lane_ar` pins nothing.
//!
//! # Lane counts
//!
//! Every lane count `1..=8` ([`LANE_COUNTS`]) is declared. The fixed-count
//! native arms (`lane_batched`, `lane_adaptive`, `lane_copy`) pin
//! `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE=8` so that `N` concurrent lanes run native
//! windows at every `N`: they are native-window coverage arms, not the shipped
//! switching policy. The shipped policy (native while at most `max_active`
//! lanes decode, AR from `max_active + 2`, hold in between) is covered by
//! `lane_cross` alone: a trial admits its first 3 requests, and the other 5
//! once the first group has finished its prompt prefill ([`admit_schedule`]),
//! so the early lanes' first decode window runs native with 3 active lanes and
//! their later windows run AR with 8 (AR from `max_active + 2` = 5 lanes; a
//! step at 4 lanes is the hold case and is recorded in the occupancy histogram).
//! Its cohort exists at 8 lanes only
//! ([`cohort_arms`]); at any other lane count the `ep2` cohort is
//! [`EP2_FIXED_ARMS`].
//!
//! There is deliberately no `lane_interleaved`: the lane executor has no
//! interleaved route, and an arm of that name is refused rather than aliased to
//! the singleton `interleaved` arm.
//!
//! # Trial rows
//!
//! A lane trial runs `lanes` concurrent requests. An arm's `ids.json` holds one
//! row per request, lane-minor: row `(trial - start) * lanes + lane` of shard
//! `START..END`. A shard therefore holds exactly `(END - START) * lanes` rows.
//! The request's seed is [`lane_request_seed`]: a function of the global trial
//! and the lane identity only, never of the slot the lane happened to occupy.
//!
//! # Merge key
//!
//! A shard's key is `(executor, topology, artifact, lanes, lane K, route,
//! sampling, state format)` plus its `range`. [`plan_lane_merge`] accepts a set only when
//! every shard of one arm shares its key, every arm shares the cross-arm
//! fields (all of the above but `route`), each arm's route is the one its name
//! declares, the file-name range equals the config's range, the ranges tile
//! `0..trials` exactly, row counts match, and every declared arm is present.
//! Anything else is an error that names the first violated field.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const EXECUTOR_ENV: &str = "HIPFIRE_SAMPLED_MTP_EXECUTOR";
pub const LANES_ENV: &str = "HIPFIRE_SAMPLED_MTP_LANES";
pub const TOPOLOGY_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOPOLOGY";
pub const SHARD_ENV: &str = "HIPFIRE_SAMPLED_MTP_SHARD";
pub const STATE_QUANT_ENV: &str = "HIPFIRE_GDN_TEST_STATE_QUANT";
pub const NGRAM_COPY_ENV: &str = "HIPFIRE_MTP_NGRAM_COPY";
/// The cohort's lane draft depth: which depth the native arms pin
/// (`HIPFIRE_EP_MTP_LANE_K`), chosen per run.
pub const LANE_K_ENV: &str = "HIPFIRE_SAMPLED_MTP_LANE_K";
/// `HIPFIRE_EP_MTP_LANE_K`: the runtime's cap on the EP lane draft depth
/// (`ep_lane_spec_k`: the load's K capped at it, default 2).
pub const EP_LANE_K_ENV: &str = "HIPFIRE_EP_MTP_LANE_K";
/// The lane draft depths the suite runs: the shipped default and the older
/// default (the sensitivity run).
pub const LANE_K_VALUES: [usize; 2] = [2, 3];
/// The shipped lane draft depth (`ep_lane_spec_k`'s default cap).
pub const DEFAULT_LANE_K: usize = 2;

/// `HIPFIRE_SAMPLED_MTP_LANE_K`: `2` or `3` (unset: [`DEFAULT_LANE_K`]);
/// anything else is refused; refused under the singleton executor, where it
/// would mislabel the run.
pub fn parse_lane_k(value: Option<&str>, executor: Executor) -> Result<usize, String> {
    match (executor, value) {
        (_, None) => Ok(DEFAULT_LANE_K),
        (Executor::Singleton, Some(v)) => Err(format!(
            "{LANE_K_ENV}={v:?} is set but {EXECUTOR_ENV} is not `lanes`: unset {LANE_K_ENV}"
        )),
        (Executor::Lanes, Some(v)) => {
            let lane_k: usize = v
                .trim()
                .parse()
                .map_err(|e| format!("{LANE_K_ENV}={v:?}: {e}"))?;
            if LANE_K_VALUES.contains(&lane_k) {
                Ok(lane_k)
            } else {
                Err(format!(
                    "{LANE_K_ENV}={v:?}: expected one of {LANE_K_VALUES:?} ({DEFAULT_LANE_K} is the shipped default)"
                ))
            }
        }
    }
}

/// The `HIPFIRE_EP_MTP_LANE_K` value of depth `lane_k` (a [`LANE_K_VALUES`]
/// member; anything else reads as the default).
pub fn lane_k_str(lane_k: usize) -> &'static str {
    match lane_k {
        3 => "3",
        _ => "2",
    }
}

/// Whether `arm` pins [`EP_LANE_K_ENV`]: the native-MTP mesh arms (the AR
/// reference drafts nothing; `lane_copy` has no ep2 executor).
pub fn pins_lane_k(arm: &str) -> bool {
    matches!(arm, "lane_batched" | "lane_adaptive" | "lane_cross")
}

/// The lane counts the suite is declared for: every count `1..=8`.
pub const LANE_COUNTS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// The declared lane arm set, reference first.
pub const LANE_ARMS: [&str; 4] = ["lane_ar", "lane_batched", "lane_adaptive", "lane_copy"];
/// The arm every other lane arm is compared against.
pub const LANE_REFERENCE_ARM: &str = "lane_ar";
/// The native<->AR crossing arm: native-MTP lanes under the PRODUCT switch
/// ([`CROSS_MAX_ACTIVE`]) with staggered admission ([`admit_schedule`]). It
/// runs under `ep2` at [`CROSS_LANES`] lanes only.
pub const CROSS_ARM: &str = "lane_cross";
/// The only lane count of [`CROSS_ARM`].
pub const CROSS_LANES: usize = 8;
/// Requests of a [`CROSS_ARM`] trial admitted at tick 0 (the rest follow once
/// they have finished their prompt prefill).
pub const CROSS_FIRST_GROUP: usize = 3;
/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE` of [`CROSS_ARM`]: the shipped
/// `lane_switch::DEFAULT_MAX_ACTIVE` (a test keeps the two equal), pinned so
/// every shard's identity records it.
pub const CROSS_MAX_ACTIVE: &str = "3";
/// [`CROSS_MAX_ACTIVE`] as a number: steps carrying at most this many lanes
/// run native windows, steps carrying `CROSS_MAX_ACTIVE_N + 2` or more run AR
/// (between the two, the previous step's mode holds).
pub const CROSS_MAX_ACTIVE_N: usize = 3;
/// The arms of an `ep2` cohort at [`CROSS_LANES`] lanes: the copy lane is off
/// by default on EP serve (`HIPFIRE_EP_MTP_LANE_COPY`), so no `ep2` cohort or
/// verdict covers it and `lane_copy` is refused under `ep2`.
pub const EP2_ARMS: [&str; 4] = ["lane_ar", "lane_batched", "lane_adaptive", CROSS_ARM];
/// The arms of an `ep2` cohort at any other lane count ([`CROSS_ARM`] has no
/// such cohort).
pub const EP2_FIXED_ARMS: [&str; 3] = ["lane_ar", "lane_batched", "lane_adaptive"];

/// Every arm a shard of `topology` may legitimately be named after (the union
/// over lane counts); [`cohort_arms`] is the exact set one lane count needs.
pub fn topology_arms(topology: Topology) -> &'static [&'static str] {
    match topology {
        Topology::Ep2 => &EP2_ARMS,
        Topology::Single | Topology::Pp2 => &LANE_ARMS,
    }
}

/// The arms a cohort of `topology` at `lanes` lanes must hold (reference
/// first): the whole declared set, except under `ep2` ([`EP2_ARMS`] at
/// [`CROSS_LANES`] lanes, [`EP2_FIXED_ARMS`] otherwise).
pub fn cohort_arms(topology: Topology, lanes: usize) -> &'static [&'static str] {
    match topology {
        Topology::Ep2 if lanes == CROSS_LANES => &EP2_ARMS,
        Topology::Ep2 => &EP2_FIXED_ARMS,
        Topology::Single | Topology::Pp2 => &LANE_ARMS,
    }
}

/// How many requests of a `lanes`-request trial of `arm` are admitted together,
/// in admission order: every arm admits all at once (`[lanes]`) except
/// [`CROSS_ARM`], which admits [`CROSS_FIRST_GROUP`] at tick 0 and the rest
/// once the previous group has finished its prompt prefill (`[3, 5]` at 8
/// lanes). The groups sum to `lanes`; request `i` is lane `i` of its trial
/// whichever group it joins.
pub fn admit_schedule(arm: &str, lanes: usize) -> Vec<usize> {
    if arm == CROSS_ARM && lanes > CROSS_FIRST_GROUP {
        vec![CROSS_FIRST_GROUP, lanes - CROSS_FIRST_GROUP]
    } else {
        vec![lanes]
    }
}

/// Whether the next admission group is due: every request admitted so far has
/// forwarded its whole `prompt_len`-token prompt (`prefilled` per request).
pub fn next_group_due(prefilled: &[usize], prompt_len: usize) -> bool {
    prefilled.iter().all(|&done| done >= prompt_len)
}

/// The sidecar/config schema version of the `lane_suite` block.
pub const SCHEMA: u64 = 2;

pub fn is_lane_arm(arm: &str) -> bool {
    LANE_ARMS.contains(&arm) || arm == CROSS_ARM
}

// ----------------------------------------------------------------- parsers

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Executor {
    /// The historical singleton drivers (one request per arm process).
    Singleton,
    /// The continuous-batching lane executor.
    Lanes,
}

impl Executor {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("") | Some("singleton") => Ok(Self::Singleton),
            Some("lanes") => Ok(Self::Lanes),
            Some(other) => Err(format!(
                "{EXECUTOR_ENV}={other:?}: expected `singleton` or `lanes`"
            )),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::var(EXECUTOR_ENV).ok().as_deref())
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Singleton => "singleton",
            Self::Lanes => "lanes",
        }
    }
}

/// Where an arm runs. `Single` is one card, `Pp2` a two-card pipeline mesh,
/// `Ep2` a two-rank expert-parallel mesh (`topology.ep = 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Single,
    Pp2,
    Ep2,
}

impl Topology {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("") | Some("single") => Ok(Self::Single),
            Some("pp2") => Ok(Self::Pp2),
            Some("ep2") => Ok(Self::Ep2),
            Some(other) => Err(format!(
                "{TOPOLOGY_ENV}={other:?}: expected `single`, `pp2` or `ep2`"
            )),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::var(TOPOLOGY_ENV).ok().as_deref())
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Pp2 => "pp2",
            Self::Ep2 => "ep2",
        }
    }
}

/// `HIPFIRE_SAMPLED_MTP_LANES`: required (`1` to `8`) under the lane executor
/// so the lane count is always a recorded part of the identity; refused under
/// the singleton executor, where it would mislabel the run.
pub fn parse_lanes(value: Option<&str>, executor: Executor) -> Result<usize, String> {
    match (executor, value) {
        (Executor::Singleton, None) => Ok(1),
        (Executor::Singleton, Some(v)) => Err(format!(
            "{LANES_ENV}={v:?} is set but {EXECUTOR_ENV} is not `lanes`: the singleton arms run one request; unset {LANES_ENV}"
        )),
        (Executor::Lanes, None) => Err(format!(
            "{EXECUTOR_ENV}=lanes needs {LANES_ENV}=1..8 (the lane count is part of every shard's identity)"
        )),
        (Executor::Lanes, Some(v)) => {
            let lanes: usize = v
                .trim()
                .parse()
                .map_err(|e| format!("{LANES_ENV}={v:?}: {e}"))?;
            if LANE_COUNTS.contains(&lanes) {
                Ok(lanes)
            } else {
                Err(format!("{LANES_ENV}={v:?}: expected one of {LANE_COUNTS:?}"))
            }
        }
    }
}

pub fn lanes_from_env(executor: Executor) -> Result<usize, String> {
    parse_lanes(std::env::var(LANES_ENV).ok().as_deref(), executor)
}

/// Executor, topology, lane count and lane draft depth of this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub executor: Executor,
    pub topology: Topology,
    pub lanes: usize,
    /// The native arms' lane draft depth ([`LANE_K_ENV`]).
    pub lane_k: usize,
}

impl Placement {
    pub fn from_env() -> Result<Self, String> {
        let executor = Executor::from_env()?;
        Ok(Self {
            executor,
            topology: Topology::from_env()?,
            lanes: lanes_from_env(executor)?,
            lane_k: parse_lane_k(std::env::var(LANE_K_ENV).ok().as_deref(), executor)?,
        })
    }
}

/// Whether `arm` (run in `mode`) may run under `placement`.
///
/// * Singleton executor: the four short leviathan arms take `single` or `pp2`
///   (`short_arms` is that set); every other arm runs on one card only; `ep2`
///   has no singleton arm.
/// * Lane executor: only the declared lane arms, leviathan mode, any topology;
///   [`CROSS_ARM`] only under `ep2` at [`CROSS_LANES`] lanes.
pub fn validate_arm(
    arm: &str,
    mode: &str,
    placement: Placement,
    short_arms: &[&str],
) -> Result<(), String> {
    match placement.executor {
        Executor::Lanes => {
            if !is_lane_arm(arm) {
                return Err(format!(
                    "{EXECUTOR_ENV}=lanes runs only the lane arms {LANE_ARMS:?} and {CROSS_ARM}, not arm {arm}; unset {EXECUTOR_ENV} for the singleton arms"
                ));
            }
            if mode != "leviathan" {
                return Err(format!(
                    "lane arm {arm} runs the leviathan verifier only (mode {mode})"
                ));
            }
            if let Some(refusal) = cross_placement_failure(arm, placement.topology, placement.lanes)
            {
                return Err(refusal);
            }
            if !cohort_arms(placement.topology, placement.lanes).contains(&arm) {
                return Err(format!(
                    "lane arm {arm} is not part of the {} cohort {:?}: the copy lane is off by default on EP serve and no ep2 verdict covers it",
                    placement.topology.as_str(),
                    cohort_arms(placement.topology, placement.lanes)
                ));
            }
            Ok(())
        }
        Executor::Singleton => {
            if is_lane_arm(arm) {
                return Err(format!(
                    "arm {arm} is a lane arm: set {EXECUTOR_ENV}=lanes and {LANES_ENV}=1..8"
                ));
            }
            let takes_mesh = short_arms.contains(&arm) && mode == "leviathan";
            match placement.topology {
                Topology::Single => Ok(()),
                Topology::Pp2 if takes_mesh => Ok(()),
                Topology::Ep2 => Err(format!(
                    "{TOPOLOGY_ENV}=ep2 runs only under {EXECUTOR_ENV}=lanes: there is no singleton EP2 arm (arm {arm}, mode {mode})"
                )),
                Topology::Pp2 => Err(format!(
                    "{TOPOLOGY_ENV} is set but arm {arm} (mode {mode}) runs on one card only: only the {short_arms:?} leviathan arms take a PP2 topology; unset {TOPOLOGY_ENV}"
                )),
            }
        }
    }
}

/// The refusal of a [`CROSS_ARM`] placement other than `ep2` at [`CROSS_LANES`]
/// lanes; `None` for every other arm and for the one valid placement.
pub fn cross_placement_failure(arm: &str, topology: Topology, lanes: usize) -> Option<String> {
    if arm != CROSS_ARM {
        return None;
    }
    if topology != Topology::Ep2 {
        return Some(format!(
            "{CROSS_ARM} runs under {TOPOLOGY_ENV}=ep2 only (the product switch it covers is the EP mesh step), not {}",
            topology.as_str()
        ));
    }
    (lanes != CROSS_LANES).then(|| {
        format!(
            "{CROSS_ARM} needs {LANES_ENV}={CROSS_LANES} (it admits {CROSS_FIRST_GROUP} requests, then the rest, so 8 lanes cross the max_active={CROSS_MAX_ACTIVE} switch); lanes {lanes} is refused"
        )
    })
}

/// The route label of a lane arm (the `route` merge-key field).
pub fn lane_route(arm: &str) -> Option<&'static str> {
    match arm {
        "lane_ar" => Some("ar"),
        "lane_batched" => Some("batched"),
        "lane_adaptive" => Some("adaptive"),
        "lane_copy" => Some("copy"),
        "lane_cross" => Some("cross"),
        _ => None,
    }
}

/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE`: the largest lane count of a step that
/// still runs native windows (the mesh step switches to AR windows from
/// `max + 2` lanes up).
pub const MAX_ACTIVE_ENV: &str = "HIPFIRE_EP_MTP_LANE_MAX_ACTIVE";
/// The value the fixed-count native-MTP arms pin: at least the largest suite
/// lane count ([`LANE_COUNTS`]), so `N` lanes run native windows at every `N`
/// (a native-window coverage arm, not the shipped switching policy; that is
/// [`CROSS_ARM`], which pins [`CROSS_MAX_ACTIVE`]).
pub const LANE_MAX_ACTIVE: &str = "8";

/// Extra environment a lane arm pins (beyond the route/floor knobs the shared
/// `arm_overrides` derives from the arm name): `lane_copy` always runs the copy
/// lane, every fixed-count native-MTP arm pins [`MAX_ACTIVE_ENV`] to
/// [`LANE_MAX_ACTIVE`], `lane_cross` pins it to [`CROSS_MAX_ACTIVE`], and every
/// native-MTP mesh arm pins [`EP_LANE_K_ENV`] to the cohort's `lane_k` (the AR
/// reference store never switches windows or drafts, so it pins nothing).
pub fn lane_env_overrides(arm: &str, lane_k: usize) -> Vec<(&'static str, Option<&'static str>)> {
    let k = (EP_LANE_K_ENV, Some(lane_k_str(lane_k)));
    match arm {
        "lane_copy" => vec![
            (NGRAM_COPY_ENV, Some("on")),
            (MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE)),
        ],
        "lane_batched" | "lane_adaptive" => vec![(MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE)), k],
        "lane_cross" => vec![(MAX_ACTIVE_ENV, Some(CROSS_MAX_ACTIVE)), k],
        _ => Vec::new(),
    }
}

/// Salt of a lane arm's request seeds, distinct per arm and from every
/// singleton arm salt.
pub fn lane_salt(arm: &str) -> Result<u64, String> {
    match arm {
        "lane_ar" => Ok(0x1A4E_A200),
        "lane_batched" => Ok(0x1A4E_BA7C),
        "lane_adaptive" => Ok(0x1A4E_ADA9),
        "lane_copy" => Ok(0x1A4E_C0B1),
        "lane_cross" => Ok(0x1A4E_C705),
        other => Err(format!("{other} is not a lane arm")),
    }
}

/// The seed of the request that occupies `lane` of global `trial`: the
/// SplitMix64 `trial_seed` of `salt` and the request's global index
/// `trial * lanes + lane`. It depends on the request, never on the slot it was
/// scheduled onto, so different lane scheduling cannot perturb a stream.
pub fn lane_request_seed(
    seed_of: impl Fn(u64, usize) -> u64,
    salt: u64,
    lanes: usize,
    trial: usize,
    lane: usize,
) -> u64 {
    debug_assert!(lane < lanes);
    seed_of(salt, trial * lanes + lane)
}

// ------------------------------------------------------------------ shards

/// `START..END` (half-open) within `0..total`; errors name `env_name`.
pub fn parse_shard(env_name: &str, value: &str, total: usize) -> Result<(usize, usize), String> {
    let (start, end) = value
        .split_once("..")
        .ok_or_else(|| format!("{env_name}={value}: expected START..END"))?;
    let parse = |text: &str| {
        text.trim()
            .parse::<usize>()
            .map_err(|e| format!("{env_name}={value}: {e}"))
    };
    let (start, end) = (parse(start)?, parse(end)?);
    if start >= end || end > total {
        return Err(format!(
            "{env_name}={value}: the range must be non-empty and within 0..{total}"
        ));
    }
    Ok((start, end))
}

/// The `START-END` of `<arm>.<START>-<END>.ids.json`, or `None` for any other
/// file name (whole-arm files, other arms, sidecars).
pub fn shard_of_file(arm: &str, name: &str) -> Option<(usize, usize)> {
    let rest = name
        .strip_prefix(&format!("{arm}."))?
        .strip_suffix(".ids.json")?;
    let (start, end) = rest.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()?))
}

/// Every `<arm>.<START>-<END>.ids.json` in `names`, as `(arm, range)`, for any
/// arm name; whole-arm `<arm>.ids.json` files (merger outputs) are skipped.
pub fn shard_files(names: &[String]) -> Vec<(String, (usize, usize))> {
    let mut found = Vec::new();
    for name in names {
        let Some(stem) = name.strip_suffix(".ids.json") else {
            continue;
        };
        let Some((arm, range)) = stem.rsplit_once('.') else {
            continue;
        };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
            found.push((arm.to_string(), (start, end)));
        }
    }
    found.sort();
    found
}

/// Refuses a directory holding shards of any arm outside the arms of
/// `topology` ([`topology_arms`]: singleton arms, an interleaved lane arm,
/// `lane_copy` under `ep2`, a typo): a lane merge never silently ignores
/// foreign shards. (Whether the arms present are the exact cohort of the
/// recorded lane count is [`plan_lane_merge`]'s check.)
pub fn reject_foreign_shards(names: &[String], topology: Topology) -> Result<(), String> {
    let cohort = topology_arms(topology);
    let foreign: BTreeSet<String> = shard_files(names)
        .into_iter()
        .map(|(arm, _)| arm)
        .filter(|arm| !cohort.contains(&arm.as_str()))
        .collect();
    if foreign.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "mixed shard set: the lane output directory also holds shards of {foreign:?}, outside the {} lane arm set {cohort:?}; use a separate HIPFIRE_SAMPLED_MTP_OUT directory per executor",
            topology.as_str()
        ))
    }
}

/// Uncovered parts of `0..total` given sorted shards; overlap and out-of-range
/// shards are errors (a seed would be counted twice / belongs to no cohort).
pub fn shard_gaps(shards: &[(usize, usize)], total: usize) -> Result<Vec<(usize, usize)>, String> {
    let mut gaps = Vec::new();
    let mut next = 0;
    for &(start, end) in shards {
        if start >= end || end > total {
            return Err(format!("shard {start}-{end} lies outside 0..{total}"));
        }
        if start < next {
            return Err(format!(
                "shard {start}-{end} overlaps an earlier shard ending at {next}: a seed would be counted twice"
            ));
        }
        if start > next {
            gaps.push((next, start));
        }
        next = end;
    }
    if next < total {
        gaps.push((next, total));
    }
    Ok(gaps)
}

// ------------------------------------------------------------------ config

/// The sampling/cohort fields of a distribution config that make two shards
/// the same experiment.
const SAMPLING_FIELDS: [&str; 13] = [
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "repeat_window",
    "presence_penalty",
    "frequency_penalty",
    "fixture",
    "trials",
    "tokens",
    "sampled_mode",
    "mtp_k",
];

/// The `lane_suite` block recorded in a lane arm's config.
pub fn lane_block(
    arm: &str,
    placement: Placement,
    state_format: &str,
    range: (usize, usize),
) -> Value {
    json!({
        "schema": SCHEMA,
        "executor": placement.executor.as_str(),
        "topology": placement.topology.as_str(),
        "lanes": placement.lanes,
        "lane_k": placement.lane_k,
        "route": lane_route(arm),
        "state_format": state_format,
        "range": [range.0, range.1],
        "rows_per_trial": placement.lanes,
    })
}

/// A lane arm's config: the shared distribution `base` (a `dist_config` of the
/// arm) with the `lane_suite` block and the arm's seed salt.
pub fn lane_config(
    base: Value,
    arm: &str,
    placement: Placement,
    state_format: &str,
    range: (usize, usize),
) -> Result<Value, String> {
    let mut config = base;
    config["kind"] = json!("lane-distribution");
    config["lane_suite"] = lane_block(arm, placement, state_format, range);
    config["salt"] = json!(lane_salt(arm)?);
    Ok(config)
}

/// The merge key of one shard (everything but its range).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneShardKey {
    pub arm: String,
    pub executor: String,
    pub topology: String,
    pub artifact: String,
    pub lanes: usize,
    pub lane_k: usize,
    pub route: String,
    pub sampling: String,
    pub state_format: String,
}

impl LaneShardKey {
    /// The key and recorded range of a lane arm's config. Every field is read
    /// from the config and cross-checked against the environment recorded in
    /// its identity, so a config whose `lane_suite` block disagrees with the
    /// environment it claims to have run under is refused.
    pub fn from_config(config: &Value) -> Result<(Self, (usize, usize)), String> {
        let text = |value: &Value, what: &str| -> Result<String, String> {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("config has no {what}"))
        };
        let arm = text(&config["arm"], "arm")?;
        let block = &config["lane_suite"];
        if block.is_null() {
            return Err(format!(
                "{arm}: config has no lane_suite block (not a lane shard)"
            ));
        }
        if block["schema"].as_u64() != Some(SCHEMA) {
            return Err(format!(
                "{arm}: lane_suite schema {} is not {SCHEMA}",
                block["schema"]
            ));
        }
        let executor = text(&block["executor"], "lane_suite.executor")?;
        if executor != Executor::Lanes.as_str() {
            return Err(format!(
                "{arm}: lane_suite.executor is {executor:?}, not \"lanes\""
            ));
        }
        let topology = text(&block["topology"], "lane_suite.topology")?;
        Topology::parse(Some(&topology))?;
        let lanes = block["lanes"]
            .as_u64()
            .map(|lanes| lanes as usize)
            .filter(|lanes| LANE_COUNTS.contains(lanes))
            .ok_or_else(|| {
                format!(
                    "{arm}: lane_suite.lanes {} is not one of {LANE_COUNTS:?}",
                    block["lanes"]
                )
            })?;
        let route = text(&block["route"], "lane_suite.route")?;
        if lane_route(&arm) != Some(route.as_str()) {
            return Err(format!(
                "{arm}: lane_suite.route {route:?} is not the route of this arm ({:?})",
                lane_route(&arm)
            ));
        }
        if let Some(refusal) =
            cross_placement_failure(&arm, Topology::parse(Some(&topology))?, lanes)
        {
            return Err(format!("{arm}: {refusal}"));
        }
        let lane_k = block["lane_k"]
            .as_u64()
            .map(|k| k as usize)
            .filter(|k| LANE_K_VALUES.contains(k))
            .ok_or_else(|| {
                format!(
                    "{arm}: lane_suite.lane_k {} is not one of {LANE_K_VALUES:?}",
                    block["lane_k"]
                )
            })?;
        let state_format = text(&block["state_format"], "lane_suite.state_format")?;
        let range = {
            let pair = block["range"]
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| format!("{arm}: lane_suite.range is not [start, end]"))?;
            let end = |i: usize| {
                pair[i]
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("{arm}: lane_suite.range is not [start, end]"))
            };
            (end(0)?, end(1)?)
        };
        if range.0 >= range.1 {
            return Err(format!("{arm}: lane_suite.range {range:?} is empty"));
        }
        let artifact = text(
            &config["identity"]["model"]["sha256"],
            "identity.model.sha256",
        )
        .map_err(|e| format!("{arm}: {e}"))?;
        // The recorded environment must agree with the block.
        let env = &config["identity"]["env"];
        let env_of = |name: &str| env[name].as_str();
        if Executor::parse(env_of(EXECUTOR_ENV))? != Executor::Lanes {
            return Err(format!(
                "{arm}: identity.env {EXECUTOR_ENV} is {:?}, not lanes",
                env_of(EXECUTOR_ENV)
            ));
        }
        if Topology::parse(env_of(TOPOLOGY_ENV))?.as_str() != topology {
            return Err(format!(
                "{arm}: identity.env {TOPOLOGY_ENV} {:?} disagrees with lane_suite.topology {topology:?}",
                env_of(TOPOLOGY_ENV)
            ));
        }
        if parse_lanes(env_of(LANES_ENV), Executor::Lanes)? != lanes {
            return Err(format!(
                "{arm}: identity.env {LANES_ENV} {:?} disagrees with lane_suite.lanes {lanes}",
                env_of(LANES_ENV)
            ));
        }
        if parse_lane_k(env_of(LANE_K_ENV), Executor::Lanes)? != lane_k {
            return Err(format!(
                "{arm}: identity.env {LANE_K_ENV} {:?} disagrees with lane_suite.lane_k {lane_k}",
                env_of(LANE_K_ENV)
            ));
        }
        if pins_lane_k(&arm) && env_of(EP_LANE_K_ENV) != Some(lane_k_str(lane_k)) {
            return Err(format!(
                "{arm}: identity.env {EP_LANE_K_ENV} {:?} is not the pinned lane depth {lane_k}",
                env_of(EP_LANE_K_ENV)
            ));
        }
        let env_state = env_of(STATE_QUANT_ENV).unwrap_or("q8");
        if env_state != state_format {
            return Err(format!(
                "{arm}: identity.env {STATE_QUANT_ENV} {env_state:?} disagrees with lane_suite.state_format {state_format:?}"
            ));
        }
        let mut sampling = serde_json::Map::new();
        for field in SAMPLING_FIELDS {
            if config.get(field).is_none() {
                return Err(format!("{arm}: config has no sampling field {field}"));
            }
            sampling.insert(field.to_string(), config[field].clone());
        }
        Ok((
            Self {
                arm,
                executor,
                topology,
                artifact,
                lanes,
                lane_k,
                route,
                sampling: Value::Object(sampling).to_string(),
                state_format,
            },
            range,
        ))
    }
}

/// One shard presented to the merge planner.
#[derive(Clone, Debug)]
pub struct LaneShard {
    /// The range in the shard's file name (`<arm>.<START>-<END>.*`).
    pub file_range: (usize, usize),
    pub config: Value,
    /// Rows in the shard's `ids.json`.
    pub rows: usize,
}

/// A validated merge: every arm's ranges in order, plus the one cross-arm key.
#[derive(Debug, PartialEq, Eq)]
pub struct LaneMergePlan {
    pub trials: usize,
    pub lanes: usize,
    pub lane_k: usize,
    pub topology: String,
    pub state_format: String,
    pub artifact: String,
    pub arms: BTreeMap<String, Vec<(usize, usize)>>,
}

/// The first field on which two keys of the same experiment differ.
fn key_difference(a: &LaneShardKey, b: &LaneShardKey, cross_arm: bool) -> Option<&'static str> {
    if a.executor != b.executor {
        return Some("executor");
    }
    if a.topology != b.topology {
        return Some("topology");
    }
    if a.artifact != b.artifact {
        return Some("artifact");
    }
    if a.lanes != b.lanes {
        return Some("lanes");
    }
    if a.lane_k != b.lane_k {
        return Some("lane K");
    }
    if !cross_arm && a.route != b.route {
        return Some("route");
    }
    if a.sampling != b.sampling {
        return Some("sampling");
    }
    if a.state_format != b.state_format {
        return Some("state format");
    }
    None
}

/// The one lane count every shard records (`lane_suite.lanes`), which picks
/// the cohort ([`cohort_arms`]). Shards whose block has no valid count are
/// skipped here and refused by [`LaneShardKey::from_config`]; shards that
/// record different counts are a mixed set.
fn recorded_lanes(shards: &BTreeMap<String, Vec<LaneShard>>) -> Result<usize, String> {
    let counts: BTreeSet<usize> = shards
        .values()
        .flatten()
        .filter_map(|shard| shard.config["lane_suite"]["lanes"].as_u64())
        .map(|lanes| lanes as usize)
        .filter(|lanes| LANE_COUNTS.contains(lanes))
        .collect();
    match counts.len() {
        0 | 1 => Ok(counts.into_iter().next().unwrap_or(0)),
        _ => Err(format!(
            "mixed shard set: the shards record different lanes {counts:?}"
        )),
    }
}

/// Validates `shards` (arm -> its shards) against the lane arm set of the
/// `topology` cohort ([`cohort_arms`]) and the `trials` cohort. See the module
/// docs for the exact rules.
pub fn plan_lane_merge(
    shards: &BTreeMap<String, Vec<LaneShard>>,
    trials: usize,
    topology: Topology,
) -> Result<LaneMergePlan, String> {
    let lanes = recorded_lanes(shards)?;
    let cohort = cohort_arms(topology, lanes);
    for arm in shards.keys() {
        if !cohort.contains(&arm.as_str()) {
            return Err(format!(
                "mixed shard set: arm {arm} is not in the {} lane arm set {cohort:?}",
                topology.as_str()
            ));
        }
    }
    let missing: Vec<&str> = cohort
        .iter()
        .copied()
        .filter(|arm| shards.get(*arm).is_none_or(|parts| parts.is_empty()))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "incomplete arm set: no shards for {missing:?}; the {} lane suite needs all of {cohort:?}",
            topology.as_str()
        ));
    }
    let mut first: Option<LaneShardKey> = None;
    let mut arms = BTreeMap::new();
    for &arm in cohort {
        let parts = &shards[arm];
        let mut arm_key: Option<LaneShardKey> = None;
        let mut ranges = Vec::new();
        for part in parts {
            let (key, range) = LaneShardKey::from_config(&part.config)
                .map_err(|e| format!("{arm} shard {:?}: {e}", part.file_range))?;
            if key.arm != arm {
                return Err(format!(
                    "{arm} shard {:?}: config belongs to arm {}",
                    part.file_range, key.arm
                ));
            }
            if range != part.file_range {
                return Err(format!(
                    "{arm} shard {:?}: config records range {range:?}; the file name and the config disagree",
                    part.file_range
                ));
            }
            let expected_trials = key_trials(&part.config)
                .ok_or_else(|| format!("{arm} shard {range:?}: config has no integer trials"))?;
            if expected_trials != trials {
                return Err(format!(
                    "{arm} shard {range:?}: config declares {expected_trials} trials, merging {trials}"
                ));
            }
            if part.rows != (range.1 - range.0) * key.lanes {
                return Err(format!(
                    "{arm} shard {range:?}: {} rows, expected {} trials x {} lanes = {}",
                    part.rows,
                    range.1 - range.0,
                    key.lanes,
                    (range.1 - range.0) * key.lanes
                ));
            }
            match &arm_key {
                None => arm_key = Some(key),
                Some(reference) => {
                    if let Some(field) = key_difference(reference, &key, false) {
                        return Err(format!(
                            "mixed shard set: {arm} shard {range:?} differs from its siblings in {field}"
                        ));
                    }
                }
            }
            ranges.push(range);
        }
        ranges.sort_unstable();
        let gaps = shard_gaps(&ranges, trials).map_err(|e| format!("{arm}: {e}"))?;
        if !gaps.is_empty() {
            return Err(format!(
                "incomplete shard set: {arm} leaves {gaps:?} of 0..{trials} uncovered"
            ));
        }
        let arm_key = arm_key.expect("an arm with shards has a key");
        match &first {
            None => first = Some(arm_key),
            Some(reference) => {
                if let Some(field) = key_difference(reference, &arm_key, true) {
                    return Err(format!(
                        "mixed shard set: arm {arm} differs from {} in {field}",
                        reference.arm
                    ));
                }
            }
        }
        arms.insert(arm.to_string(), ranges);
    }
    let key = first.expect("the arm set is non-empty");
    Ok(LaneMergePlan {
        trials,
        lanes: key.lanes,
        lane_k: key.lane_k,
        topology: key.topology,
        state_format: key.state_format,
        artifact: key.artifact,
        arms,
    })
}

fn key_trials(config: &Value) -> Option<usize> {
    config["trials"].as_u64().map(|n| n as usize)
}

/// The config a merged whole-arm file carries: the shared config with the
/// range widened to the whole cohort.
pub fn merged_config(shard_config: &Value, trials: usize) -> Value {
    let mut config = shard_config.clone();
    config["lane_suite"]["range"] = json!([0, trials]);
    config
}

// --------------------------------------------------------------- per lane

/// `rows` (lane-minor, `lanes` per trial) split into `lanes` per-lane row
/// lists; a row count that is not a multiple of `lanes` is an error.
pub fn per_lane_rows(rows: &[Vec<u32>], lanes: usize) -> Result<Vec<Vec<Vec<u32>>>, String> {
    if lanes == 0 || rows.len() % lanes != 0 {
        return Err(format!(
            "{} rows do not split into {lanes} lanes",
            rows.len()
        ));
    }
    let mut out = vec![Vec::with_capacity(rows.len() / lanes); lanes];
    for (index, row) in rows.iter().enumerate() {
        out[index % lanes].push(row.clone());
    }
    Ok(out)
}

// --------------------------------------------------------------- evidence

/// Integer evidence counters summed over shards.
const EVIDENCE_SUM_KEYS: [&str; 12] = [
    "trials",
    "requests",
    "mtp_windows",
    "ar_windows",
    "retired_trials",
    "calibration_ok_trials",
    "no_window_trials",
    "copy_windows",
    "copy_point_mass_windows",
    "steps",
    "drafted",
    "accepted",
];
/// Per-request row arrays concatenated in shard order (when present).
const EVIDENCE_CONCAT_KEYS: [&str; 2] = ["per_trial", "retire_steps"];
/// Optional integer counters summed over shards: absent from every shard stays
/// absent (an arm or an older shard that never recorded it), present in some
/// but not all is an error.
const EVIDENCE_OPTIONAL_SUM_KEYS: [&str; 1] = ["crossing_requests"];
/// The integer counters of an evidence `mesh` block, summed over shards (its
/// `steps_by_lanes` histogram is summed bucket by bucket).
const EVIDENCE_MESH_SUM_KEYS: [&str; 8] = [
    "steps",
    "lane_windows",
    "native",
    "copy",
    "ar",
    "offered",
    "accepted",
    "emitted",
];
/// Values that must be identical in every shard (when present).
const EVIDENCE_EQUAL_KEYS: [&str; 2] = ["probe_budget", "lane_k"];

/// Joins the per-shard `evidence.json` of one arm (in range order): the nine
/// coverage counters (see [`lane_coverage`]) plus `steps`/`drafted`/`accepted`
/// are summed and must exist in every shard, `per_trial`/`retire_steps` rows
/// are concatenated, `probe_budget` must agree, and every other field is taken
/// from the first shard.
pub fn merge_evidence(parts: &[Value]) -> Result<Value, String> {
    let Some(first) = parts.first() else {
        return Err("no evidence shards to merge".into());
    };
    let mut merged = first.clone();
    for key in EVIDENCE_SUM_KEYS {
        let mut total = 0u64;
        for (index, part) in parts.iter().enumerate() {
            match part[key].as_u64() {
                Some(value) => total += value,
                None if matches!(key, "steps" | "drafted" | "accepted") && part[key].is_null() => {}
                None => return Err(format!("evidence shard {index} has no integer {key}")),
            }
        }
        merged[key] = json!(total);
    }
    for key in EVIDENCE_OPTIONAL_SUM_KEYS {
        if parts.iter().all(|part| part[key].is_null()) {
            continue;
        }
        let mut total = 0u64;
        for (index, part) in parts.iter().enumerate() {
            total += part[key].as_u64().ok_or_else(|| {
                format!("evidence shard {index} has no integer {key} (other shards do)")
            })?;
        }
        merged[key] = json!(total);
    }
    merge_mesh(parts, &mut merged)?;
    merge_depth(parts, &mut merged)?;
    for key in EVIDENCE_CONCAT_KEYS {
        if first[key].is_null() {
            continue;
        }
        let mut rows = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            rows.extend(
                part[key]
                    .as_array()
                    .ok_or_else(|| format!("evidence shard {index}: {key} is not an array"))?
                    .iter()
                    .cloned(),
            );
        }
        merged[key] = Value::Array(rows);
    }
    for key in EVIDENCE_EQUAL_KEYS {
        if parts.iter().any(|part| part[key] != first[key]) {
            return Err(format!("evidence shards disagree on {key}"));
        }
    }
    Ok(merged)
}

/// Sums the `mesh` blocks of `parts` into `merged` when every shard has one
/// (the counters of [`EVIDENCE_MESH_SUM_KEYS`] and the `steps_by_lanes`
/// histogram); shards without a mesh block leave the first shard's untouched.
fn merge_mesh(parts: &[Value], merged: &mut Value) -> Result<(), String> {
    if parts.iter().any(|part| !part["mesh"].is_object()) {
        return Ok(());
    }
    for key in EVIDENCE_MESH_SUM_KEYS {
        let mut total = 0u64;
        for (index, part) in parts.iter().enumerate() {
            total += part["mesh"][key]
                .as_u64()
                .ok_or_else(|| format!("evidence shard {index} has no integer mesh.{key}"))?;
        }
        merged["mesh"][key] = json!(total);
    }
    let mut histogram: BTreeMap<String, u64> = BTreeMap::new();
    for (index, part) in parts.iter().enumerate() {
        let buckets = part["mesh"]["steps_by_lanes"]
            .as_object()
            .ok_or_else(|| format!("evidence shard {index} has no mesh.steps_by_lanes"))?;
        for (lanes, steps) in buckets {
            *histogram.entry(lanes.clone()).or_default() += steps.as_u64().ok_or_else(|| {
                format!("evidence shard {index}: mesh.steps_by_lanes[{lanes}] is not an integer")
            })?;
        }
    }
    merged["mesh"]["steps_by_lanes"] = json!(histogram);
    Ok(())
}

/// Merges the draft-depth counters of `parts` into `merged`: `max_offered` is
/// the maximum and the `first_window_offered` histogram is summed, when every
/// shard has them (older shards leave the first shard's untouched).
fn merge_depth(parts: &[Value], merged: &mut Value) -> Result<(), String> {
    if parts.iter().all(|part| part["max_offered"].is_u64()) {
        let max = parts
            .iter()
            .filter_map(|part| part["max_offered"].as_u64())
            .max()
            .unwrap_or(0);
        merged["max_offered"] = json!(max);
    }
    if parts
        .iter()
        .all(|part| part["first_window_offered"].is_object())
    {
        let mut histogram: BTreeMap<String, u64> = BTreeMap::new();
        for (index, part) in parts.iter().enumerate() {
            for (offered, requests) in part["first_window_offered"]
                .as_object()
                .into_iter()
                .flatten()
            {
                *histogram.entry(offered.clone()).or_default() +=
                    requests.as_u64().ok_or_else(|| {
                        format!(
                            "evidence shard {index}: first_window_offered[{offered}] is not an integer"
                        )
                    })?;
            }
        }
        merged["first_window_offered"] = json!(histogram);
    }
    Ok(())
}

/// The draft-depth rules of a native arm's evidence under cohort depth
/// `lane_k`: the recorded `lane_k` is the cohort's, no window offered more than
/// `lane_k` drafts (`max_offered`), and the first window of some request
/// offered exactly `lane_k` (`first_window_offered`), so the pinned depth was
/// really exercised. The arms that do not pin a depth have no such rules.
pub fn lane_k_coverage(arm: &str, lane_k: usize, evidence: &Value) -> Vec<String> {
    let mut failures = Vec::new();
    if !pins_lane_k(arm) {
        return failures;
    }
    match evidence["lane_k"].as_u64() {
        Some(recorded) if recorded as usize == lane_k => {}
        other => failures.push(format!(
            "{arm}: evidence records lane_k {other:?}, the cohort runs {lane_k}"
        )),
    }
    match evidence["max_offered"].as_u64() {
        Some(max) if max as usize <= lane_k => {}
        Some(max) => failures.push(format!(
            "{arm}: a window offered {max} drafts, above the pinned lane K={lane_k}"
        )),
        None => failures.push(format!("{arm}: evidence has no integer max_offered")),
    }
    match evidence["first_window_offered"].as_object() {
        Some(histogram) => {
            let full = histogram
                .get(&lane_k.to_string())
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if full == 0 {
                failures.push(format!(
                    "{arm}: no request's first window offered K={lane_k} drafts (first_window_offered {}): the pinned depth was never exercised",
                    Value::Object(histogram.clone())
                ));
            }
        }
        None => failures.push(format!("{arm}: evidence has no first_window_offered")),
    }
    failures
}

// --------------------------------------------------------------- coverage

/// Route coverage of a lane arm from its merged `evidence.json`. Counters are
/// summed over every request of the arm; the schema (all non-negative
/// integers) is the contract the lane executor's evidence writer must meet:
///
/// `trials`, `requests`, `mtp_windows`, `ar_windows`, `retired_trials`,
/// `calibration_ok_trials`, `no_window_trials`, `copy_windows`,
/// `copy_point_mass_windows`.
///
/// Returns one message per coverage failure; empty = the arm exercised what
/// its name claims. A missing counter is a failure, never a zero.
///
/// Under `ep2` the mesh lanes calibrate inline (the per-lane router prices at
/// `ar_us = 1.0`; there is no zero-draft calibration window), so the adaptive
/// arm is required to run native windows only; its floor-on route is checked
/// from the recorded config ([`adaptive_floor_failure`]).
///
/// `lane_cross` additionally needs `crossing_requests` (requests with both a
/// native and an AR window) and the `mesh.steps_by_lanes` occupancy histogram
/// ([`cross_coverage_failures`]).
pub fn lane_coverage(
    topology: Topology,
    arm: &str,
    lanes: usize,
    trials: usize,
    evidence: &Value,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut get = |key: &str| -> u64 {
        match evidence[key].as_u64() {
            Some(value) => value,
            None => {
                failures.push(format!("{arm}: evidence has no integer {key}"));
                0
            }
        }
    };
    let (ev_trials, requests) = (get("trials"), get("requests"));
    let (mtp, ar) = (get("mtp_windows"), get("ar_windows"));
    let (retired, calibrated, no_window) = (
        get("retired_trials"),
        get("calibration_ok_trials"),
        get("no_window_trials"),
    );
    let (copy, point_mass) = (get("copy_windows"), get("copy_point_mass_windows"));
    if !failures.is_empty() {
        return failures;
    }
    if ev_trials as usize != trials {
        failures.push(format!(
            "{arm}: evidence covers {ev_trials} trials, not {trials}"
        ));
    }
    if requests as usize != trials * lanes {
        failures.push(format!(
            "{arm}: evidence covers {requests} requests, not {trials} trials x {lanes} lanes"
        ));
    }
    let mut need = |ok: bool, what: &str| {
        if !ok {
            failures.push(format!(
                "{arm}: {what} (mtp_windows={mtp} ar_windows={ar} retired_trials={retired} calibration_ok_trials={calibrated} no_window_trials={no_window} copy_windows={copy} copy_point_mass_windows={point_mass})"
            ));
        }
    };
    match arm {
        "lane_ar" => {
            need(
                mtp == 0 && copy == 0,
                "the AR reference lanes must run no speculative window",
            );
            need(
                ar > 0,
                "the AR reference lanes must run target-only windows",
            );
        }
        "lane_batched" => {
            need(mtp > 0, "a forced batched arm must run native windows");
            need(
                ar == 0 && retired == 0,
                "a forced floor-off arm must never run an AR window or retire",
            );
        }
        "lane_adaptive" => {
            need(mtp > 0, "the adaptive arm must run native windows");
            if topology != Topology::Ep2 {
                need(
                    calibrated + no_window == requests,
                    "the adaptive arm's first window must be the calibration token in every request",
                );
            }
        }
        "lane_copy" => {
            need(copy > 0, "the copy arm must run copy-proposal windows");
            need(
                point_mass == copy,
                "every copy proposal must be a one-hot (point-mass) q",
            );
        }
        "lane_cross" => {
            need(mtp > 0, "the crossing arm must run native windows");
            need(
                ar > 0,
                "the crossing arm must run AR windows (the product step switch above max_active)",
            );
            need(
                retired == 0,
                "the crossing arm runs floor off: no lane may retire",
            );
        }
        other => failures.push(format!("{other} is not a lane arm")),
    }
    if arm == CROSS_ARM {
        failures.extend(cross_coverage_failures(lanes, evidence));
    }
    failures
}

/// The crossing rules of a [`CROSS_ARM`] evidence: eight lanes, at least one
/// request that ran both a native and an AR window (`crossing_requests`), and
/// mesh steps (`mesh.steps_by_lanes`) at occupancy `<=` [`CROSS_MAX_ACTIVE_N`]
/// (native windows) and at occupancy `>=` [`CROSS_MAX_ACTIVE_N`]` + 2` (AR
/// windows). A run that never crossed cannot produce a verdict.
pub fn cross_coverage_failures(lanes: usize, evidence: &Value) -> Vec<String> {
    let mut failures = Vec::new();
    if lanes != CROSS_LANES {
        failures.push(format!(
            "{CROSS_ARM}: {lanes} lanes, but the crossing arm is defined at {CROSS_LANES}"
        ));
    }
    match evidence["crossing_requests"].as_u64() {
        Some(0) => failures.push(format!(
            "{CROSS_ARM}: crossing_requests=0: no request ran both a native and an AR window, so the run never crossed the product switch and carries no verdict"
        )),
        Some(_) => {}
        None => failures.push(format!(
            "{CROSS_ARM}: evidence has no integer crossing_requests"
        )),
    }
    match step_occupancy(evidence) {
        Ok(histogram) => {
            let steps_where = |keep: &dyn Fn(usize) -> bool| -> u64 {
                histogram
                    .iter()
                    .filter(|(active, _)| keep(**active))
                    .map(|(_, steps)| *steps)
                    .sum()
            };
            if steps_where(&|active| active >= 1 && active <= CROSS_MAX_ACTIVE_N) == 0 {
                failures.push(format!(
                    "{CROSS_ARM}: no mesh step carried <= {CROSS_MAX_ACTIVE_N} active lanes (steps_by_lanes {histogram:?}): the native side of the switch never ran"
                ));
            }
            if steps_where(&|active| active >= CROSS_MAX_ACTIVE_N + 2) == 0 {
                failures.push(format!(
                    "{CROSS_ARM}: no mesh step carried >= {} active lanes (steps_by_lanes {histogram:?}): the AR side of the switch never ran",
                    CROSS_MAX_ACTIVE_N + 2
                ));
            }
        }
        Err(error) => failures.push(format!("{CROSS_ARM}: {error}")),
    }
    failures
}

/// `mesh.steps_by_lanes` of an evidence: mesh steps by the number of active
/// lanes they carried.
fn step_occupancy(evidence: &Value) -> Result<BTreeMap<usize, u64>, String> {
    let buckets = evidence["mesh"]["steps_by_lanes"]
        .as_object()
        .ok_or("evidence has no mesh.steps_by_lanes histogram")?;
    let mut histogram = BTreeMap::new();
    for (lanes, steps) in buckets {
        let lanes: usize = lanes
            .parse()
            .map_err(|e| format!("mesh.steps_by_lanes key {lanes:?}: {e}"))?;
        let steps = steps
            .as_u64()
            .ok_or_else(|| format!("mesh.steps_by_lanes[{lanes}] is not an integer"))?;
        histogram.insert(lanes, steps);
    }
    Ok(histogram)
}

/// The failure of a `lane_adaptive` result whose recorded config is not the
/// floor-on, no-forced-route class (`HIPFIRE_MTP_AR_FLOOR` and
/// `HIPFIRE_MTP_INCREMENTAL` unset): `None` when it is.
pub fn adaptive_floor_failure(config: &Value) -> Option<String> {
    let class = config["identity"]["floor"]["route_class"].as_str();
    (class != Some("adaptive-floor")).then(|| {
        format!(
            "lane_adaptive: the recorded route class is {class:?}, not \"adaptive-floor\" (floor on, no forced route)"
        )
    })
}

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(topology: Topology, lanes: usize) -> Placement {
        Placement {
            executor: Executor::Lanes,
            topology,
            lanes,
            lane_k: DEFAULT_LANE_K,
        }
    }

    const SHORT: [&str; 4] = ["ar", "batched", "interleaved", "adaptive"];

    #[test]
    fn executor_topology_and_lanes_parse() {
        assert_eq!(Executor::parse(None), Ok(Executor::Singleton));
        assert_eq!(Executor::parse(Some("")), Ok(Executor::Singleton));
        assert_eq!(Executor::parse(Some("singleton")), Ok(Executor::Singleton));
        assert_eq!(Executor::parse(Some("lanes")), Ok(Executor::Lanes));
        assert!(Executor::parse(Some("Lanes")).is_err());
        assert_eq!(Topology::parse(None), Ok(Topology::Single));
        assert_eq!(Topology::parse(Some("single")), Ok(Topology::Single));
        assert_eq!(Topology::parse(Some("pp2")), Ok(Topology::Pp2));
        assert_eq!(Topology::parse(Some("ep2")), Ok(Topology::Ep2));
        for bad in ["pp1", "ep4", "EP2", "pp"] {
            assert!(Topology::parse(Some(bad)).is_err(), "{bad}");
        }
        for topology in [Topology::Single, Topology::Pp2, Topology::Ep2] {
            assert_eq!(Topology::parse(Some(topology.as_str())), Ok(topology));
        }
        assert_eq!(parse_lanes(None, Executor::Singleton), Ok(1));
        assert!(parse_lanes(Some("8"), Executor::Singleton).is_err());
        assert!(parse_lanes(None, Executor::Lanes).is_err());
        assert_eq!(parse_lanes(Some("1"), Executor::Lanes), Ok(1));
        assert_eq!(parse_lanes(Some(" 8 "), Executor::Lanes), Ok(8));
        for lanes in 1..=8usize {
            assert_eq!(
                parse_lanes(Some(&lanes.to_string()), Executor::Lanes),
                Ok(lanes)
            );
        }
        for bad in ["0", "9", "16", "-1", "1.5", "x", ""] {
            assert!(parse_lanes(Some(bad), Executor::Lanes).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn arm_validation_separates_executors() {
        for topology in [Topology::Single, Topology::Pp2, Topology::Ep2] {
            for &arm in cohort_arms(topology, 8) {
                validate_arm(arm, "leviathan", placement(topology, 8), &SHORT).unwrap();
                assert!(validate_arm(arm, "naive", placement(topology, 8), &SHORT).is_err());
            }
            for arm in SHORT.iter().chain(&["copy_long_ar", "late_ar"]) {
                assert!(
                    validate_arm(arm, "leviathan", placement(topology, 8), &SHORT).is_err(),
                    "{arm} under lanes"
                );
            }
        }
        // The copy lane is off by default on EP serve: no ep2 cohort covers it.
        let refused = validate_arm(
            "lane_copy",
            "leviathan",
            placement(Topology::Ep2, 8),
            &SHORT,
        )
        .unwrap_err();
        assert!(
            refused.contains("lane_copy") && refused.contains("ep2"),
            "{refused}"
        );
        validate_arm(
            "lane_copy",
            "leviathan",
            placement(Topology::Pp2, 8),
            &SHORT,
        )
        .unwrap();
        assert!(validate_arm(
            "lane_interleaved",
            "leviathan",
            placement(Topology::Single, 1),
            &SHORT
        )
        .is_err());
        let singleton = |topology| Placement {
            executor: Executor::Singleton,
            topology,
            lanes: 1,
            lane_k: DEFAULT_LANE_K,
        };
        for arm in LANE_ARMS {
            assert!(validate_arm(arm, "leviathan", singleton(Topology::Single), &SHORT).is_err());
        }
        // Unchanged singleton rules: pp2 only for the short leviathan arms; ep2 never.
        for arm in SHORT {
            validate_arm(arm, "leviathan", singleton(Topology::Single), &SHORT).unwrap();
            validate_arm(arm, "leviathan", singleton(Topology::Pp2), &SHORT).unwrap();
            assert!(validate_arm(arm, "leviathan", singleton(Topology::Ep2), &SHORT).is_err());
            assert!(validate_arm(arm, "naive", singleton(Topology::Pp2), &SHORT).is_err());
        }
        validate_arm(
            "copy_long_ar",
            "leviathan",
            singleton(Topology::Single),
            &SHORT,
        )
        .unwrap();
        assert!(validate_arm(
            "copy_long_ar",
            "leviathan",
            singleton(Topology::Pp2),
            &SHORT
        )
        .is_err());
    }

    #[test]
    fn every_lane_count_one_to_eight_parses_and_nothing_else() {
        assert_eq!(LANE_COUNTS.to_vec(), (1..=8).collect::<Vec<usize>>());
        for lanes in LANE_COUNTS {
            assert_eq!(
                parse_lanes(Some(&format!(" {lanes} ")), Executor::Lanes),
                Ok(lanes)
            );
            assert!(
                parse_lanes(Some(&lanes.to_string()), Executor::Singleton).is_err(),
                "the singleton executor refuses {LANES_ENV}"
            );
        }
        for bad in ["0", "9"] {
            let error = parse_lanes(Some(bad), Executor::Lanes).unwrap_err();
            assert!(
                error.contains(LANES_ENV) && error.contains("[1, 2,"),
                "{error}"
            );
        }
        let missing = parse_lanes(None, Executor::Lanes).unwrap_err();
        assert!(missing.contains("1..8"), "{missing}");
    }

    #[test]
    fn the_crossing_arm_is_ep2_at_eight_lanes_only() {
        assert!(is_lane_arm(CROSS_ARM));
        assert!(
            !LANE_ARMS.contains(&CROSS_ARM),
            "the single/pp2 cohorts never hold it"
        );
        assert_eq!(
            EP2_ARMS,
            ["lane_ar", "lane_batched", "lane_adaptive", CROSS_ARM]
        );
        assert_eq!(EP2_ARMS[0], LANE_REFERENCE_ARM);
        assert_eq!(lane_route(CROSS_ARM), Some("cross"));
        let salt = lane_salt(CROSS_ARM).unwrap();
        assert!(LANE_ARMS.iter().all(|arm| lane_salt(arm).unwrap() != salt));
        assert_eq!(
            lane_env_overrides(CROSS_ARM, 2),
            vec![(MAX_ACTIVE_ENV, Some("3")), (EP_LANE_K_ENV, Some("2"))],
            "pinned explicitly so the identity records it"
        );
        validate_arm(CROSS_ARM, "leviathan", placement(Topology::Ep2, 8), &SHORT).unwrap();
        assert!(validate_arm(CROSS_ARM, "naive", placement(Topology::Ep2, 8), &SHORT).is_err());
        for lanes in 1..=7 {
            let error = validate_arm(
                CROSS_ARM,
                "leviathan",
                placement(Topology::Ep2, lanes),
                &SHORT,
            )
            .unwrap_err();
            assert!(
                error.contains("lane_cross") && error.contains("=8"),
                "{lanes}: {error}"
            );
        }
        for topology in [Topology::Single, Topology::Pp2] {
            let error =
                validate_arm(CROSS_ARM, "leviathan", placement(topology, 8), &SHORT).unwrap_err();
            assert!(error.contains("ep2 only"), "{error}");
        }
        // Cohorts: the crossing arm exists at 8 ep2 lanes only.
        assert_eq!(cohort_arms(Topology::Ep2, 8), &EP2_ARMS[..]);
        for lanes in 1..=7 {
            assert_eq!(cohort_arms(Topology::Ep2, lanes), &EP2_FIXED_ARMS[..]);
        }
        assert_eq!(cohort_arms(Topology::Pp2, 8), &LANE_ARMS[..]);
        assert_eq!(cohort_arms(Topology::Single, 1), &LANE_ARMS[..]);
        assert_eq!(topology_arms(Topology::Ep2), &EP2_ARMS[..]);
        assert_eq!(topology_arms(Topology::Pp2), &LANE_ARMS[..]);
    }

    #[test]
    fn the_crossing_pin_is_the_shipped_switch_geometry() {
        use hipfire_arch_qwen4::lane_switch::{native_mode, DEFAULT_MAX_ACTIVE};
        // The arm cannot drift from the product default.
        assert_eq!(CROSS_MAX_ACTIVE_N, DEFAULT_MAX_ACTIVE);
        assert_eq!(CROSS_MAX_ACTIVE.parse::<usize>(), Ok(DEFAULT_MAX_ACTIVE));
        // The first group alone runs native windows whatever the previous
        // step's mode was; all eight lanes run AR windows.
        assert!(CROSS_FIRST_GROUP <= CROSS_MAX_ACTIVE_N);
        assert!(native_mode(CROSS_FIRST_GROUP, false, CROSS_MAX_ACTIVE_N));
        assert!(native_mode(CROSS_FIRST_GROUP, true, CROSS_MAX_ACTIVE_N));
        assert!(CROSS_LANES >= CROSS_MAX_ACTIVE_N + 2);
        assert!(!native_mode(CROSS_LANES, true, CROSS_MAX_ACTIVE_N));
        assert!(!native_mode(CROSS_LANES, false, CROSS_MAX_ACTIVE_N));
        // One lane above max_active is the hold case: the previous mode stands.
        assert!(native_mode(
            CROSS_MAX_ACTIVE_N + 1,
            true,
            CROSS_MAX_ACTIVE_N
        ));
        assert!(!native_mode(
            CROSS_MAX_ACTIVE_N + 1,
            false,
            CROSS_MAX_ACTIVE_N
        ));
        // The fixed-count pin (8) never switches at any suite lane count; the
        // crossing pin does at 8.
        let fixed: usize = LANE_MAX_ACTIVE.parse().unwrap();
        for lanes in LANE_COUNTS {
            assert!(native_mode(lanes, false, fixed), "{lanes}");
        }
    }

    #[test]
    fn admission_is_staggered_only_for_the_crossing_arm() {
        assert_eq!(admit_schedule(CROSS_ARM, 8), vec![3, 5]);
        assert_eq!(admit_schedule(CROSS_ARM, 5), vec![3, 2]);
        assert_eq!(admit_schedule(CROSS_ARM, 4), vec![3, 1]);
        assert_eq!(admit_schedule(CROSS_ARM, 3), vec![3]);
        for arm in LANE_ARMS.iter().copied().chain([CROSS_ARM]) {
            for lanes in LANE_COUNTS {
                let groups = admit_schedule(arm, lanes);
                assert_eq!(groups.iter().sum::<usize>(), lanes, "{arm} {lanes}");
                assert!(groups.iter().all(|&group| group > 0), "{arm} {lanes}");
                if arm != CROSS_ARM {
                    assert_eq!(groups, vec![lanes], "{arm} admits all at once");
                }
            }
        }
        // The first group of the crossing arm is the one the shipped switch
        // keeps native on its own; the second brings the step to eight.
        let groups = admit_schedule(CROSS_ARM, CROSS_LANES);
        assert_eq!(
            groups,
            vec![CROSS_FIRST_GROUP, CROSS_LANES - CROSS_FIRST_GROUP]
        );
        assert!(groups[0] <= CROSS_MAX_ACTIVE_N);
        assert!(groups.iter().sum::<usize>() >= CROSS_MAX_ACTIVE_N + 2);
    }

    #[test]
    fn the_next_group_waits_for_every_admitted_prompt() {
        assert!(next_group_due(&[], 33), "nothing admitted yet");
        assert!(!next_group_due(&[0, 0, 0, 0], 33));
        assert!(
            !next_group_due(&[33, 33, 33, 32], 33),
            "one lane still prefilling"
        );
        assert!(next_group_due(&[33, 33, 33, 33], 33));
    }

    #[test]
    fn the_lane_depth_is_two_or_three_and_nothing_else() {
        assert_eq!(DEFAULT_LANE_K, 2, "the shipped ep_lane_spec_k default");
        assert_eq!(LANE_K_VALUES, [2, 3]);
        assert_eq!(parse_lane_k(None, Executor::Lanes), Ok(2));
        assert_eq!(parse_lane_k(Some("2"), Executor::Lanes), Ok(2));
        assert_eq!(parse_lane_k(Some(" 3 "), Executor::Lanes), Ok(3));
        for bad in ["0", "1", "4", "8", "-2", "2.5", "x", ""] {
            let error = parse_lane_k(Some(bad), Executor::Lanes).unwrap_err();
            assert!(error.contains(LANE_K_ENV), "{bad:?}: {error}");
        }
        // The singleton executor has no lane depth: unset is the default, set is refused.
        assert_eq!(parse_lane_k(None, Executor::Singleton), Ok(2));
        assert!(parse_lane_k(Some("3"), Executor::Singleton).is_err());
        assert_eq!(lane_k_str(2), "2");
        assert_eq!(lane_k_str(3), "3");
    }

    #[test]
    fn the_native_arms_pin_the_lane_depth_and_the_reference_pins_nothing() {
        for lane_k in LANE_K_VALUES {
            let pin = (EP_LANE_K_ENV, Some(lane_k_str(lane_k)));
            for arm in ["lane_batched", "lane_adaptive", CROSS_ARM] {
                assert!(pins_lane_k(arm), "{arm}");
                assert!(
                    lane_env_overrides(arm, lane_k).contains(&pin),
                    "{arm} {lane_k}"
                );
            }
            for arm in ["lane_ar", "lane_copy"] {
                assert!(!pins_lane_k(arm), "{arm}");
                assert!(
                    !lane_env_overrides(arm, lane_k)
                        .iter()
                        .any(|(name, _)| *name == EP_LANE_K_ENV),
                    "{arm}"
                );
            }
        }
        assert_eq!(
            lane_env_overrides("lane_batched", 3),
            vec![(MAX_ACTIVE_ENV, Some("8")), (EP_LANE_K_ENV, Some("3"))]
        );
        assert_eq!(
            lane_env_overrides(CROSS_ARM, 3),
            vec![(MAX_ACTIVE_ENV, Some("3")), (EP_LANE_K_ENV, Some("3"))]
        );
    }

    fn k_shard(arm: &str, range: (usize, usize), lanes: usize, lane_k: usize) -> LaneShard {
        let mut base = base_config(arm);
        base["identity"]["env"][TOPOLOGY_ENV] = json!("ep2");
        base["identity"]["env"][LANES_ENV] = json!(lanes.to_string());
        base["identity"]["env"][LANE_K_ENV] = json!(lane_k.to_string());
        if pins_lane_k(arm) {
            base["identity"]["env"][EP_LANE_K_ENV] = json!(lane_k_str(lane_k));
        }
        let placement = Placement {
            lane_k,
            ..placement(Topology::Ep2, lanes)
        };
        LaneShard {
            file_range: range,
            config: lane_config(base, arm, placement, "q8", range).unwrap(),
            rows: (range.1 - range.0) * lanes,
        }
    }

    fn k_set(lanes: usize, lane_k: usize) -> BTreeMap<String, Vec<LaneShard>> {
        cohort_arms(Topology::Ep2, lanes)
            .iter()
            .map(|arm| {
                (
                    arm.to_string(),
                    vec![
                        k_shard(arm, (0, 150), lanes, lane_k),
                        k_shard(arm, (150, 300), lanes, lane_k),
                    ],
                )
            })
            .collect()
    }

    #[test]
    fn the_lane_depth_is_part_of_the_shard_identity_and_the_merge_key() {
        // The block records it; the key carries it.
        let two = k_shard("lane_batched", (0, 100), 5, 2);
        let three = k_shard("lane_batched", (0, 100), 5, 3);
        assert_eq!(two.config["lane_suite"]["lane_k"], json!(2));
        assert_eq!(three.config["lane_suite"]["lane_k"], json!(3));
        let (a, _) = LaneShardKey::from_config(&two.config).unwrap();
        let (b, _) = LaneShardKey::from_config(&three.config).unwrap();
        assert_eq!((a.lane_k, b.lane_k), (2, 3));
        assert_ne!(a, b);
        assert_eq!(key_difference(&a, &b, false), Some("lane K"));
        assert_eq!(
            key_difference(&a, &b, true),
            Some("lane K"),
            "a cross-arm field"
        );
        // Both depths merge as sets of their own; the plan names the depth.
        for lane_k in LANE_K_VALUES {
            for lanes in [5, 8] {
                let plan = plan_lane_merge(&k_set(lanes, lane_k), TRIALS, Topology::Ep2).unwrap();
                assert_eq!((plan.lanes, plan.lane_k), (lanes, lane_k));
            }
        }
        // A K=2 set and a K=3 set do not merge: one shard, one arm, or all arms.
        let mut set = k_set(5, 2);
        set.get_mut("lane_batched").unwrap()[1] = k_shard("lane_batched", (150, 300), 5, 3);
        let error = plan_lane_merge(&set, TRIALS, Topology::Ep2).unwrap_err();
        assert!(
            error.contains("lane_batched") && error.contains("lane K"),
            "{error}"
        );
        let mut set = k_set(5, 2);
        set.insert("lane_adaptive".into(), k_set(5, 3)["lane_adaptive"].clone());
        let error = plan_lane_merge(&set, TRIALS, Topology::Ep2).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("lane K"),
            "{error}"
        );
        // The reference arm records the cohort's depth too: a K=3 reference
        // does not pair with K=2 native arms.
        let mut set = k_set(5, 2);
        set.insert("lane_ar".into(), k_set(5, 3)["lane_ar"].clone());
        assert!(plan_lane_merge(&set, TRIALS, Topology::Ep2)
            .unwrap_err()
            .contains("lane K"));
    }

    #[test]
    fn the_recorded_environment_must_agree_with_the_lane_depth() {
        let ok = k_shard("lane_cross", (0, 100), 8, 3).config;
        LaneShardKey::from_config(&ok).unwrap();
        for (what, mutate) in [
            (
                "block depth",
                (|c: &mut Value| c["lane_suite"]["lane_k"] = json!(2)) as fn(&mut Value),
            ),
            ("cohort env", |c| {
                c["identity"]["env"][LANE_K_ENV] = json!("2")
            }),
            ("pin env", |c| {
                c["identity"]["env"][EP_LANE_K_ENV] = json!("2")
            }),
            ("pin env missing", |c| {
                c["identity"]["env"]
                    .as_object_mut()
                    .unwrap()
                    .remove(EP_LANE_K_ENV);
            }),
            ("block depth 4", |c| c["lane_suite"]["lane_k"] = json!(4)),
            ("block depth missing", |c| {
                c["lane_suite"].as_object_mut().unwrap().remove("lane_k");
            }),
        ] {
            let mut config = ok.clone();
            mutate(&mut config);
            assert!(
                LaneShardKey::from_config(&config).is_err(),
                "{what} accepted"
            );
        }
        // The reference arm pins nothing: its recorded depth is the cohort's only.
        let reference = k_shard("lane_ar", (0, 100), 8, 3).config;
        assert!(reference["identity"]["env"][EP_LANE_K_ENV].is_null());
        LaneShardKey::from_config(&reference).unwrap();
    }

    #[test]
    fn the_depth_coverage_rules() {
        let evidence = |lane_k: u64, max: u64, first: Value| json!({"lane_k": lane_k, "max_offered": max, "first_window_offered": first});
        let good = evidence(2, 2, json!({"1": 4, "2": 10}));
        for arm in ["lane_batched", "lane_adaptive", CROSS_ARM] {
            assert!(lane_k_coverage(arm, 2, &good).is_empty(), "{arm}");
        }
        // The reference arm has no depth rules.
        assert!(lane_k_coverage("lane_ar", 2, &json!({})).is_empty());
        // A window above K.
        let above = lane_k_coverage("lane_batched", 2, &evidence(2, 3, json!({"2": 1}))).concat();
        assert!(
            above.contains("offered 3 drafts, above the pinned lane K=2"),
            "{above}"
        );
        // No first window at K (the pinned depth never ran).
        let never = lane_k_coverage("lane_batched", 3, &evidence(3, 2, json!({"2": 9}))).concat();
        assert!(
            never.contains("never exercised") && never.contains("K=3"),
            "{never}"
        );
        // Zero is not K.
        assert!(!lane_k_coverage("lane_batched", 2, &evidence(2, 2, json!({"2": 0}))).is_empty());
        // Evidence of another cohort depth.
        let other = lane_k_coverage("lane_batched", 3, &good).concat();
        assert!(other.contains("records lane_k"), "{other}");
        // The schema: every field is required.
        let schema = lane_k_coverage("lane_batched", 2, &json!({})).concat();
        for field in ["lane_k", "max_offered", "first_window_offered"] {
            assert!(schema.contains(field), "{field}: {schema}");
        }
    }

    #[test]
    fn lane_arm_table_is_consistent() {
        let mut salts = BTreeSet::new();
        for arm in LANE_ARMS {
            assert!(is_lane_arm(arm));
            assert!(lane_route(arm).is_some(), "{arm}");
            assert!(
                salts.insert(lane_salt(arm).unwrap()),
                "{arm} salt is shared"
            );
        }
        assert_eq!(salts.len(), LANE_ARMS.len());
        assert!(
            salts.insert(lane_salt(CROSS_ARM).unwrap()),
            "cross salt is shared"
        );
        assert_eq!(LANE_ARMS[0], LANE_REFERENCE_ARM);
        assert!(lane_salt("ar").is_err() && lane_route("interleaved").is_none());
        assert_eq!(
            lane_env_overrides("lane_copy", 2),
            vec![
                (NGRAM_COPY_ENV, Some("on")),
                (MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE))
            ]
        );
        for arm in ["lane_batched", "lane_adaptive"] {
            assert_eq!(
                lane_env_overrides(arm, 2),
                vec![
                    (MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE)),
                    (EP_LANE_K_ENV, Some("2"))
                ]
            );
        }
        assert!(lane_env_overrides("lane_ar", 3).is_empty());
    }

    #[test]
    fn the_native_window_pin_covers_every_suite_lane_count() {
        let pin: usize = LANE_MAX_ACTIVE.parse().unwrap();
        assert!(LANE_COUNTS.iter().all(|&lanes| pin >= lanes));
    }

    #[test]
    fn shard_identity_differs_per_lane_count() {
        let key_at = |lanes: usize| {
            LaneShardKey::from_config(&ep2_shard_at("lane_batched", (0, 100), lanes).config)
                .unwrap()
                .0
        };
        for (index, &a) in LANE_COUNTS.iter().enumerate() {
            assert_eq!(key_at(a).lanes, a);
            for &b in &LANE_COUNTS[index + 1..] {
                assert_ne!(key_at(a), key_at(b), "{a} vs {b} lanes");
                assert_eq!(key_difference(&key_at(a), &key_at(b), false), Some("lanes"));
            }
        }
        let config = ep2_shard_at("lane_adaptive", (0, 100), 5).config;
        assert_eq!(config["lane_suite"]["lanes"], json!(5));
        assert_eq!(config["lane_suite"]["rows_per_trial"], json!(5));
        // A config whose environment names another lane count is refused.
        let mut mislabelled = config.clone();
        mislabelled["identity"]["env"][LANES_ENV] = json!("4");
        assert!(LaneShardKey::from_config(&mislabelled)
            .unwrap_err()
            .contains("disagrees"));
    }

    #[test]
    fn request_seeds_ignore_slot_placement_and_never_repeat() {
        // A stand-in mixer: any injective function of (salt, index) suffices here.
        let seed_of =
            |salt: u64, index: usize| salt.wrapping_mul(0x9E37_79B9).wrapping_add(index as u64);
        let salt = lane_salt("lane_batched").unwrap();
        let mut seen = BTreeSet::new();
        for lanes in LANE_COUNTS {
            seen.clear();
            for trial in 0..50 {
                for lane in 0..lanes {
                    let seed = lane_request_seed(seed_of, salt, lanes, trial, lane);
                    assert!(
                        seen.insert(seed),
                        "seed repeated at trial {trial} lane {lane}"
                    );
                    // Same request, same seed: slot placement is not an input.
                    assert_eq!(seed, lane_request_seed(seed_of, salt, lanes, trial, lane));
                }
            }
        }
    }

    #[test]
    fn shard_values_and_file_names_parse() {
        assert_eq!(parse_shard("S", "0..100", 2000), Ok((0, 100)));
        assert_eq!(parse_shard("S", " 1900 .. 2000", 2000), Ok((1900, 2000)));
        for bad in ["5..5", "10..5", "0..2001", "0-100", "a..b", "..", "3..", ""] {
            assert!(parse_shard("S", bad, 2000).is_err(), "{bad:?}");
        }
        assert_eq!(
            parse_shard("S", "7", 2000).unwrap_err(),
            "S=7: expected START..END"
        );
        assert_eq!(
            shard_of_file("lane_ar", "lane_ar.0-100.ids.json"),
            Some((0, 100))
        );
        assert_eq!(
            shard_of_file("ar", "lane_ar.0-100.ids.json"),
            None,
            "prefix must be the whole arm"
        );
        assert_eq!(shard_of_file("lane_ar", "lane_ar.ids.json"), None);
        assert_eq!(
            shard_of_file("lane_ar", "lane_ar.0-100.evidence.json"),
            None
        );
        assert_eq!(shard_of_file("lane_ar", "lane_ar.x-100.ids.json"), None);
        let names: Vec<String> = [
            "lane_ar.0-100.ids.json",
            "lane_ar.100-200.ids.json",
            "lane_ar.ids.json",
            "lane_copy.0-100.ids.json",
            "lane_copy.0-100.ids.config.json",
            "report.txt",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            shard_files(&names),
            vec![
                ("lane_ar".to_string(), (0, 100)),
                ("lane_ar".to_string(), (100, 200)),
                ("lane_copy".to_string(), (0, 100)),
            ]
        );
        reject_foreign_shards(&names, Topology::Pp2).unwrap();
        let mut mixed = names.clone();
        mixed.push("ar.0-100.ids.json".into());
        let error = reject_foreign_shards(&mixed, Topology::Pp2).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("\"ar\""),
            "{error}"
        );
        mixed.push("lane_interleaved.0-5.ids.json".into());
        assert!(reject_foreign_shards(&mixed, Topology::Pp2)
            .unwrap_err()
            .contains("lane_interleaved"));
    }

    #[test]
    fn shard_gaps_tile_exactly_once() {
        assert_eq!(shard_gaps(&[], 10), Ok(vec![(0, 10)]));
        assert_eq!(shard_gaps(&[(0, 4), (4, 10)], 10), Ok(vec![]));
        assert_eq!(
            shard_gaps(&[(2, 4), (6, 8)], 10),
            Ok(vec![(0, 2), (4, 6), (8, 10)])
        );
        assert!(shard_gaps(&[(0, 5), (4, 10)], 10).is_err());
        assert!(shard_gaps(&[(0, 12)], 10).is_err());
    }

    // ---- merger fixtures

    const TRIALS: usize = 300;
    /// The four-arm cohort topology the merger fixtures use.
    const FIXTURE_TOPOLOGY: Topology = Topology::Pp2;

    fn base_config(arm: &str) -> Value {
        let mut config = json!({
            "kind": "distribution",
            "arm": arm,
            "fixture": "prose",
            "repeat_penalty": 1.0,
            "repeat_window": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "mtp_k": 3,
            "temp": 1.0,
            "top_p": 0.95,
            "top_k": null,
            "min_p": 0.0,
            "trials": TRIALS,
            "tokens": 4,
            "sampled_mode": "leviathan",
            "identity": {
                "model": {"size": 1, "sha256": "eafaff94"},
                "env": {
                    EXECUTOR_ENV: "lanes",
                    LANES_ENV: "8",
                    TOPOLOGY_ENV: "pp2",
                },
            },
        });
        // The native arms ran under the default depth pin.
        if pins_lane_k(arm) {
            config["identity"]["env"][EP_LANE_K_ENV] = json!(lane_k_str(DEFAULT_LANE_K));
        }
        config
    }

    fn shard(arm: &str, range: (usize, usize)) -> LaneShard {
        let base = base_config(arm);
        let config = lane_config(base, arm, placement(FIXTURE_TOPOLOGY, 8), "q8", range).unwrap();
        LaneShard {
            file_range: range,
            config,
            rows: (range.1 - range.0) * 8,
        }
    }

    fn full_set() -> BTreeMap<String, Vec<LaneShard>> {
        LANE_ARMS
            .iter()
            .map(|arm| {
                (
                    arm.to_string(),
                    vec![
                        shard(arm, (0, 100)),
                        shard(arm, (100, 200)),
                        shard(arm, (200, 300)),
                    ],
                )
            })
            .collect()
    }

    fn edit(
        set: &mut BTreeMap<String, Vec<LaneShard>>,
        arm: &str,
        index: usize,
        f: impl FnOnce(&mut LaneShard),
    ) {
        f(&mut set.get_mut(arm).unwrap()[index]);
    }

    #[test]
    fn merge_accepts_a_complete_consistent_set() {
        let plan = plan_lane_merge(&full_set(), TRIALS, FIXTURE_TOPOLOGY).unwrap();
        assert_eq!(plan.lanes, 8);
        assert_eq!(plan.topology, "pp2");
        assert_eq!(plan.state_format, "q8");
        assert_eq!(plan.artifact, "eafaff94");
        assert_eq!(plan.arms.len(), 4);
        assert_eq!(
            plan.arms["lane_copy"],
            vec![(0, 100), (100, 200), (200, 300)]
        );
        // Shard order in the input does not matter.
        let mut shuffled = full_set();
        shuffled.get_mut("lane_ar").unwrap().reverse();
        assert_eq!(
            plan_lane_merge(&shuffled, TRIALS, FIXTURE_TOPOLOGY).unwrap(),
            plan
        );
        let whole = merged_config(&full_set()["lane_ar"][1].config, TRIALS);
        assert_eq!(whole["lane_suite"]["range"], json!([0, TRIALS]));
    }

    #[test]
    fn merge_rejects_incomplete_sets() {
        let mut set = full_set();
        set.remove("lane_copy");
        let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
        assert!(
            error.contains("incomplete arm set") && error.contains("lane_copy"),
            "{error}"
        );

        let mut set = full_set();
        set.get_mut("lane_batched").unwrap().remove(1);
        let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
        assert!(
            error.contains("incomplete shard set") && error.contains("(100, 200)"),
            "{error}"
        );

        let mut set = full_set();
        set.get_mut("lane_ar").unwrap().pop();
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("(200, 300)"));

        let mut set = full_set();
        set.get_mut("lane_ar")
            .unwrap()
            .push(shard("lane_ar", (100, 200)));
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("overlaps"));

        assert!(plan_lane_merge(&BTreeMap::new(), TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("incomplete arm set"));
    }

    #[test]
    fn merge_rejects_mixed_sets_on_every_key_field() {
        type Mutate = fn(&mut Value);
        let cases: [(&str, Mutate); 8] = [
            ("executor", |c| {
                c["lane_suite"]["executor"] = json!("singleton");
            }),
            ("topology", |c| {
                c["lane_suite"]["topology"] = json!("single");
                c["identity"]["env"][TOPOLOGY_ENV] = json!("single");
            }),
            ("artifact", |c| {
                c["identity"]["model"]["sha256"] = json!("541cfb55");
            }),
            ("lanes", |c| {
                c["lane_suite"]["lanes"] = json!(1);
                c["identity"]["env"][LANES_ENV] = json!("1");
            }),
            ("sampling", |c| {
                c["top_p"] = json!(0.9);
            }),
            ("state format", |c| {
                c["lane_suite"]["state_format"] = json!("fp32");
                c["identity"]["env"][STATE_QUANT_ENV] = json!("fp32");
            }),
            ("route", |c| {
                c["lane_suite"]["route"] = json!("batched");
            }),
            ("range", |c| {
                c["lane_suite"]["range"] = json!([0, 99]);
            }),
        ];
        for (field, mutate) in cases {
            // Mutate ONE shard of ONE non-first arm: the set must be refused.
            let mut set = full_set();
            edit(&mut set, "lane_adaptive", 1, |s| mutate(&mut s.config));
            let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
            assert!(
                error.contains(field) || error.contains("lane_suite") || error.contains("disagree"),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn merge_names_the_differing_field_for_consistent_but_mixed_shards() {
        // Internally consistent shards from a different experiment than their siblings.
        let mut set = full_set();
        let foreign = {
            let mut config = base_config("lane_ar");
            config["identity"]["model"]["sha256"] = json!("541cfb55");
            lane_config(
                config,
                "lane_ar",
                placement(FIXTURE_TOPOLOGY, 8),
                "q8",
                (100, 200),
            )
            .unwrap()
        };
        edit(&mut set, "lane_ar", 1, |s| s.config = foreign);
        let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("artifact"),
            "{error}"
        );

        let mut set = full_set();
        let foreign = {
            let mut config = base_config("lane_copy");
            config["identity"]["env"][TOPOLOGY_ENV] = json!("single");
            lane_config(
                config,
                "lane_copy",
                placement(Topology::Single, 8),
                "q8",
                (0, 100),
            )
            .unwrap()
        };
        edit(&mut set, "lane_copy", 0, |s| s.config = foreign);
        let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("topology"),
            "{error}"
        );

        // A whole arm from another experiment (consistent within itself).
        let mut set = full_set();
        let other: Vec<LaneShard> = [(0, 100), (100, 200), (200, 300)]
            .iter()
            .map(|&range| {
                let mut config = base_config("lane_batched");
                config["temp"] = json!(0.7);
                LaneShard {
                    file_range: range,
                    config: lane_config(
                        config,
                        "lane_batched",
                        placement(FIXTURE_TOPOLOGY, 8),
                        "q8",
                        range,
                    )
                    .unwrap(),
                    rows: (range.1 - range.0) * 8,
                }
            })
            .collect();
        set.insert("lane_batched".into(), other);
        let error = plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("sampling"),
            "{error}"
        );
    }

    #[test]
    fn merge_rejects_wrong_arms_rows_and_trials() {
        let mut set = full_set();
        set.insert("lane_interleaved".into(), vec![shard("lane_ar", (0, 300))]);
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("lane_interleaved"));

        let mut set = full_set();
        set.insert("ar".into(), vec![]);
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("lane arm set"));

        let mut set = full_set();
        edit(&mut set, "lane_ar", 0, |s| s.rows = 99);
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("rows"));

        let mut set = full_set();
        edit(&mut set, "lane_copy", 2, |s| {
            s.config["trials"] = json!(2000)
        });
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("2000 trials"));

        // A shard filed under the wrong arm.
        let mut set = full_set();
        edit(&mut set, "lane_ar", 0, |s| {
            s.config = shard("lane_copy", (0, 100)).config
        });
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("belongs to arm"));

        // File name and recorded range disagree.
        let mut set = full_set();
        edit(&mut set, "lane_ar", 0, |s| s.file_range = (0, 50));
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("disagree"));

        // A singleton config is not a lane shard.
        let mut set = full_set();
        edit(&mut set, "lane_ar", 0, |s| {
            s.config = base_config("lane_ar")
        });
        assert!(plan_lane_merge(&set, TRIALS, FIXTURE_TOPOLOGY)
            .unwrap_err()
            .contains("not a lane shard"));
    }

    #[test]
    fn config_environment_must_agree_with_the_block() {
        let ok = lane_config(
            base_config("lane_ar"),
            "lane_ar",
            placement(FIXTURE_TOPOLOGY, 8),
            "q8",
            (0, 10),
        )
        .unwrap();
        LaneShardKey::from_config(&ok).unwrap();
        for (what, mutate) in [
            (
                "executor env",
                (|c: &mut Value| c["identity"]["env"][EXECUTOR_ENV] = json!("singleton"))
                    as fn(&mut Value),
            ),
            ("topology env", |c| {
                c["identity"]["env"][TOPOLOGY_ENV] = json!("single")
            }),
            ("lanes env", |c| {
                c["identity"]["env"][LANES_ENV] = json!("1")
            }),
            ("state env", |c| {
                c["identity"]["env"][STATE_QUANT_ENV] = json!("q8_ef")
            }),
            ("schema", |c| c["lane_suite"]["schema"] = json!(3)),
            ("route", |c| c["lane_suite"]["route"] = json!("copy")),
            ("artifact", |c| c["identity"]["model"] = Value::Null),
            ("sampling field", |c| {
                c.as_object_mut().unwrap().remove("min_p");
            }),
        ] {
            let mut config = ok.clone();
            mutate(&mut config);
            assert!(
                LaneShardKey::from_config(&config).is_err(),
                "{what} accepted"
            );
        }
    }

    #[test]
    fn per_lane_rows_split_lane_minor() {
        let rows: Vec<Vec<u32>> = (0..6u32).map(|i| vec![i]).collect();
        let lanes = per_lane_rows(&rows, 3).unwrap();
        assert_eq!(
            lanes,
            vec![
                vec![vec![0], vec![3]],
                vec![vec![1], vec![4]],
                vec![vec![2], vec![5]]
            ]
        );
        assert_eq!(per_lane_rows(&rows, 1).unwrap()[0].len(), 6);
        assert!(per_lane_rows(&rows, 4).is_err());
        assert!(per_lane_rows(&rows, 0).is_err());
    }

    fn evidence(values: [(&str, u64); 9]) -> Value {
        Value::Object(
            values
                .iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect(),
        )
    }

    fn ev(
        trials: u64,
        requests: u64,
        mtp: u64,
        ar: u64,
        retired: u64,
        cal: u64,
        nowin: u64,
        copy: u64,
        pm: u64,
    ) -> Value {
        evidence([
            ("trials", trials),
            ("requests", requests),
            ("mtp_windows", mtp),
            ("ar_windows", ar),
            ("retired_trials", retired),
            ("calibration_ok_trials", cal),
            ("no_window_trials", nowin),
            ("copy_windows", copy),
            ("copy_point_mass_windows", pm),
        ])
    }

    #[test]
    fn evidence_merge_sums_counters_and_concatenates_rows() {
        let part = |trials: u64, rows: &[u64]| {
            let mut value = ev(trials, trials * 8, 10, 2, 1, 7, 0, 3, 3);
            value["steps"] = json!(trials * 4);
            value["per_trial"] = json!(rows);
            value["probe_budget"] = json!(2);
            value["note"] = json!("first");
            value
        };
        let merged = merge_evidence(&[part(100, &[1, 2]), part(200, &[3])]).unwrap();
        assert_eq!(merged["trials"], json!(300));
        assert_eq!(merged["requests"], json!(2400));
        assert_eq!(merged["mtp_windows"], json!(20));
        assert_eq!(merged["copy_point_mass_windows"], json!(6));
        assert_eq!(merged["steps"], json!(1200));
        assert_eq!(merged["per_trial"], json!([1, 2, 3]));
        assert_eq!(merged["probe_budget"], json!(2));
        assert_eq!(
            merged["drafted"],
            json!(0),
            "optional counter absent everywhere sums to zero"
        );
        // A missing required counter, disagreeing budgets and an empty set are errors.
        let mut broken = part(100, &[1]);
        broken.as_object_mut().unwrap().remove("copy_windows");
        assert!(merge_evidence(&[part(100, &[1]), broken]).is_err());
        let mut other = part(100, &[1]);
        other["probe_budget"] = json!(3);
        assert!(merge_evidence(&[part(100, &[1]), other])
            .unwrap_err()
            .contains("probe_budget"));
        assert!(merge_evidence(&[]).is_err());
    }

    fn ep2_shard(arm: &str, range: (usize, usize)) -> LaneShard {
        let mut base = base_config(arm);
        base["identity"]["env"][TOPOLOGY_ENV] = json!("ep2");
        let config = lane_config(base, arm, placement(Topology::Ep2, 8), "q8", range).unwrap();
        LaneShard {
            file_range: range,
            config,
            rows: (range.1 - range.0) * 8,
        }
    }

    fn ep2_set() -> BTreeMap<String, Vec<LaneShard>> {
        EP2_ARMS
            .iter()
            .map(|arm| {
                (
                    arm.to_string(),
                    vec![ep2_shard(arm, (0, 150)), ep2_shard(arm, (150, 300))],
                )
            })
            .collect()
    }

    #[test]
    fn the_ep2_cohort_is_the_mesh_arms() {
        assert_eq!(cohort_arms(Topology::Ep2, 8), &EP2_ARMS[..]);
        assert_eq!(cohort_arms(Topology::Pp2, 8), &LANE_ARMS[..]);
        assert_eq!(cohort_arms(Topology::Single, 8), &LANE_ARMS[..]);
        assert_eq!(EP2_ARMS[0], LANE_REFERENCE_ARM);
        let plan = plan_lane_merge(&ep2_set(), TRIALS, Topology::Ep2).unwrap();
        assert_eq!(plan.topology, "ep2");
        assert_eq!(plan.arms.len(), 4);
        assert!(plan.arms.contains_key(CROSS_ARM));
        assert!(!plan.arms.contains_key("lane_copy"));
        // The same shards are an incomplete set of the four-arm cohort.
        assert!(plan_lane_merge(&ep2_set(), TRIALS, Topology::Pp2)
            .unwrap_err()
            .contains("lane_copy"));
        // A missing ep2 arm (the crossing arm included) and a copy shard under
        // ep2 are all refused.
        for arm in ["lane_adaptive", CROSS_ARM] {
            let mut set = ep2_set();
            set.remove(arm);
            assert!(plan_lane_merge(&set, TRIALS, Topology::Ep2)
                .unwrap_err()
                .contains(arm));
        }
        let mut set = ep2_set();
        set.insert("lane_copy".into(), vec![ep2_shard("lane_copy", (0, 300))]);
        let error = plan_lane_merge(&set, TRIALS, Topology::Ep2).unwrap_err();
        assert!(
            error.contains("lane_copy") && error.contains("ep2"),
            "{error}"
        );
        let names: Vec<String> = vec![
            "lane_ar.0-150.ids.json".into(),
            "lane_copy.0-150.ids.json".into(),
        ];
        assert!(reject_foreign_shards(&names, Topology::Ep2)
            .unwrap_err()
            .contains("lane_copy"));
        reject_foreign_shards(&names, Topology::Pp2).unwrap();
        // The crossing arm's shards are not foreign to ep2, but they are to pp2.
        let names: Vec<String> = vec!["lane_cross.0-150.ids.json".into()];
        reject_foreign_shards(&names, Topology::Ep2).unwrap();
        assert!(reject_foreign_shards(&names, Topology::Pp2)
            .unwrap_err()
            .contains(CROSS_ARM));
    }

    fn ep2_shard_at(arm: &str, range: (usize, usize), lanes: usize) -> LaneShard {
        let mut base = base_config(arm);
        base["identity"]["env"][TOPOLOGY_ENV] = json!("ep2");
        base["identity"]["env"][LANES_ENV] = json!(lanes.to_string());
        let config = lane_config(base, arm, placement(Topology::Ep2, lanes), "q8", range).unwrap();
        LaneShard {
            file_range: range,
            config,
            rows: (range.1 - range.0) * lanes,
        }
    }

    fn ep2_set_at(lanes: usize) -> BTreeMap<String, Vec<LaneShard>> {
        cohort_arms(Topology::Ep2, lanes)
            .iter()
            .map(|arm| {
                (
                    arm.to_string(),
                    vec![
                        ep2_shard_at(arm, (0, 150), lanes),
                        ep2_shard_at(arm, (150, 300), lanes),
                    ],
                )
            })
            .collect()
    }

    #[test]
    fn a_five_lane_evidence_set_merges_and_mixed_lane_counts_do_not() {
        for lanes in LANE_COUNTS {
            let plan = plan_lane_merge(&ep2_set_at(lanes), TRIALS, Topology::Ep2).unwrap();
            assert_eq!(plan.lanes, lanes);
            assert_eq!(
                plan.arms.len(),
                if lanes == CROSS_LANES { 4 } else { 3 },
                "{lanes} lanes"
            );
            assert_eq!(plan.arms.contains_key(CROSS_ARM), lanes == CROSS_LANES);
        }
        let plan = plan_lane_merge(&ep2_set_at(5), TRIALS, Topology::Ep2).unwrap();
        assert_eq!(plan.arms["lane_batched"], vec![(0, 150), (150, 300)]);
        // One arm run at 4 lanes among 5-lane arms: the lane counts differ.
        let mut set = ep2_set_at(5);
        set.insert(
            "lane_adaptive".into(),
            ep2_set_at(4)["lane_adaptive"].clone(),
        );
        let error = plan_lane_merge(&set, TRIALS, Topology::Ep2).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains("lanes"),
            "{error}"
        );
        // Rows must match trials x lanes of the recorded count.
        let mut set = ep2_set_at(5);
        set.get_mut("lane_ar").unwrap()[0].rows = (150) * 8;
        assert!(plan_lane_merge(&set, TRIALS, Topology::Ep2)
            .unwrap_err()
            .contains("rows"));
        // A crossing shard recorded at 5 lanes has no cohort: it is refused,
        // whatever the other arms.
        let mut set = ep2_set_at(5);
        set.insert(CROSS_ARM.into(), vec![ep2_shard_at(CROSS_ARM, (0, 300), 5)]);
        let error = plan_lane_merge(&set, TRIALS, Topology::Ep2).unwrap_err();
        assert!(
            error.contains("mixed") && error.contains(CROSS_ARM),
            "{error}"
        );
        // The eight-lane cohort needs the crossing arm.
        let mut set = ep2_set_at(8);
        set.remove(CROSS_ARM);
        assert!(plan_lane_merge(&set, TRIALS, Topology::Ep2)
            .unwrap_err()
            .contains(CROSS_ARM));
    }

    #[test]
    fn a_crossing_shard_recorded_at_another_lane_count_is_refused() {
        let config = ep2_shard_at(CROSS_ARM, (0, 100), 5).config;
        let error = LaneShardKey::from_config(&config).unwrap_err();
        assert!(error.contains(CROSS_ARM) && error.contains("=8"), "{error}");
        LaneShardKey::from_config(&ep2_shard_at(CROSS_ARM, (0, 100), 8).config).unwrap();
        // And one recorded under pp2.
        let mut base = base_config(CROSS_ARM);
        base["identity"]["env"][TOPOLOGY_ENV] = json!("pp2");
        let config =
            lane_config(base, CROSS_ARM, placement(Topology::Pp2, 8), "q8", (0, 100)).unwrap();
        assert!(LaneShardKey::from_config(&config)
            .unwrap_err()
            .contains("ep2 only"));
    }

    /// An evidence of the crossing arm with the given counters and mesh
    /// occupancy histogram (`None` crossing: the counter is absent).
    fn cross_ev(
        mtp: u64,
        ar: u64,
        crossing: Option<u64>,
        histogram: Option<&[(usize, u64)]>,
    ) -> Value {
        let mut value = ev(10, 80, mtp, ar, 0, 0, 0, 0, 0);
        if let Some(crossing) = crossing {
            value["crossing_requests"] = json!(crossing);
        }
        if let Some(histogram) = histogram {
            value["mesh"] = json!({
                "steps_by_lanes": histogram
                    .iter()
                    .map(|(lanes, steps)| (lanes.to_string(), *steps))
                    .collect::<BTreeMap<_, _>>(),
            });
        }
        value
    }

    fn cross_failures(evidence: &Value) -> String {
        lane_coverage(Topology::Ep2, CROSS_ARM, 8, 10, evidence).concat()
    }

    #[test]
    fn crossing_coverage_needs_requests_that_crossed_and_both_step_occupancies() {
        let healthy = cross_ev(
            300,
            200,
            Some(55),
            Some(&[(3, 10), (4, 3), (5, 5), (8, 40)]),
        );
        assert!(
            cross_failures(&healthy).is_empty(),
            "{}",
            cross_failures(&healthy)
        );
        // The boundary occupancies count: 3 is native, 5 is AR (4 is only the
        // hold case: recorded, never required).
        let edge = cross_ev(300, 200, Some(1), Some(&[(3, 1), (5, 1)]));
        assert!(
            cross_failures(&edge).is_empty(),
            "{}",
            cross_failures(&edge)
        );

        // No request ever had both kinds of window.
        let none = cross_failures(&cross_ev(300, 200, Some(0), Some(&[(3, 10), (8, 40)])));
        assert!(
            none.contains("crossing_requests=0") && none.contains("no verdict"),
            "{none}"
        );
        // The counter is part of the schema.
        let missing = cross_failures(&cross_ev(300, 200, None, Some(&[(3, 10), (8, 40)])));
        assert!(missing.contains("crossing_requests"), "{missing}");
        // Steps never at occupancy <= 3 (4 holds): the native side did not run.
        let no_native = cross_failures(&cross_ev(300, 200, Some(55), Some(&[(4, 10), (8, 40)])));
        assert!(no_native.contains("<= 3"), "{no_native}");
        // Steps never at occupancy >= 5 (4 holds the previous mode).
        let no_ar = cross_failures(&cross_ev(300, 200, Some(55), Some(&[(2, 10), (4, 40)])));
        assert!(no_ar.contains(">= 5"), "{no_ar}");
        // Zero-count buckets are not steps.
        let empty = cross_failures(&cross_ev(300, 200, Some(55), Some(&[(3, 0), (8, 40)])));
        assert!(empty.contains("<= 3"), "{empty}");
        // The histogram is part of the schema.
        let no_mesh = cross_failures(&cross_ev(300, 200, Some(55), None));
        assert!(no_mesh.contains("steps_by_lanes"), "{no_mesh}");
        // Both window kinds, no retirement.
        assert!(!cross_failures(&cross_ev(0, 200, Some(55), Some(&[(3, 10), (8, 40)]))).is_empty());
        assert!(!cross_failures(&cross_ev(300, 0, Some(55), Some(&[(3, 10), (8, 40)]))).is_empty());
        let mut retired = cross_ev(300, 200, Some(55), Some(&[(3, 10), (8, 40)]));
        retired["retired_trials"] = json!(1);
        assert!(!cross_failures(&retired).is_empty());
        // Eight lanes only, and the cohort size is still checked.
        assert!(!lane_coverage(Topology::Ep2, CROSS_ARM, 5, 10, &healthy).is_empty());
        assert!(!lane_coverage(Topology::Ep2, CROSS_ARM, 8, 11, &healthy).is_empty());
        // The other arms never read the crossing fields.
        assert!(lane_coverage(
            Topology::Ep2,
            "lane_batched",
            8,
            10,
            &ev(10, 80, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
    }

    #[test]
    fn crossing_evidence_shards_merge_their_counters_and_occupancy() {
        let part = |crossing: Option<u64>, histogram: &[(usize, u64)]| {
            let mut value = cross_ev(100, 60, crossing, Some(histogram));
            let steps: u64 = histogram.iter().map(|(_, steps)| steps).sum();
            value["mesh"]["steps"] = json!(steps);
            for key in [
                "lane_windows",
                "native",
                "copy",
                "ar",
                "offered",
                "accepted",
                "emitted",
            ] {
                value["mesh"][key] = json!(1);
            }
            value
        };
        let merged = merge_evidence(&[
            part(Some(20), &[(3, 10), (8, 30)]),
            part(Some(35), &[(3, 5), (5, 2), (8, 40)]),
        ])
        .unwrap();
        assert_eq!(merged["crossing_requests"], json!(55));
        assert_eq!(
            merged["mesh"]["steps_by_lanes"],
            json!({"3": 15, "5": 2, "8": 70})
        );
        assert_eq!(merged["mesh"]["steps"], json!(87));
        assert_eq!(merged["mesh"]["native"], json!(2));
        let failures = lane_coverage(Topology::Ep2, CROSS_ARM, 8, 20, &merged);
        assert!(failures.is_empty(), "{failures:?}");
        // Absent from every shard stays absent: the schema failure survives a merge.
        let merged =
            merge_evidence(&[part(None, &[(3, 1), (8, 1)]), part(None, &[(3, 1)])]).unwrap();
        assert!(merged["crossing_requests"].is_null());
        assert!(cross_failures(&merged).contains("crossing_requests"));
        // Present in some shards only is an error.
        assert!(
            merge_evidence(&[part(Some(1), &[(3, 1)]), part(None, &[(3, 1)])])
                .unwrap_err()
                .contains("crossing_requests")
        );
        // Shards without a mesh block keep the first shard's (older evidence).
        let plain = ev(10, 80, 400, 0, 0, 0, 0, 0, 0);
        let merged = merge_evidence(&[plain.clone(), plain]).unwrap();
        assert!(merged["mesh"].is_null());
    }

    #[test]
    fn ep2_adaptive_coverage_needs_native_windows_but_no_calibration_window() {
        let t = 10u64;
        // No request starts with a zero-draft calibration window (the mesh
        // router calibrates inline): met under ep2, a failure on the other
        // topologies.
        let evidence = ev(t, 80, 300, 0, 0, 0, 0, 0, 0);
        assert!(lane_coverage(Topology::Ep2, "lane_adaptive", 8, 10, &evidence).is_empty());
        assert!(!lane_coverage(Topology::Pp2, "lane_adaptive", 8, 10, &evidence).is_empty());
        // Still needs native windows, and the counter schema.
        let silent = ev(t, 80, 0, 0, 0, 0, 0, 0, 0);
        assert!(!lane_coverage(Topology::Ep2, "lane_adaptive", 8, 10, &silent).is_empty());
        assert!(!lane_coverage(Topology::Ep2, "lane_adaptive", 8, 11, &evidence).is_empty());
        // The other ep2 arms keep their rules.
        assert!(lane_coverage(
            Topology::Ep2,
            "lane_batched",
            8,
            10,
            &ev(t, 80, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Ep2,
            "lane_batched",
            8,
            10,
            &ev(t, 80, 400, 1, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(lane_coverage(
            Topology::Ep2,
            "lane_ar",
            8,
            10,
            &ev(t, 80, 0, 500, 0, 0, 80, 0, 0)
        )
        .is_empty());
    }

    #[test]
    fn the_adaptive_arm_must_have_run_floor_on_with_no_forced_route() {
        let config = |class: Value| json!({"identity": {"floor": {"route_class": class}}});
        assert_eq!(
            adaptive_floor_failure(&config(json!("adaptive-floor"))),
            None
        );
        let forced = adaptive_floor_failure(&config(json!("forced-batched-floor-off"))).unwrap();
        assert!(forced.contains("forced-batched-floor-off"), "{forced}");
        assert!(adaptive_floor_failure(&json!({})).is_some());
    }

    #[test]
    fn lane_coverage_rules() {
        let t = 10u64;
        // Healthy evidence per arm (8 lanes => 80 requests).
        assert!(lane_coverage(
            Topology::Pp2,
            "lane_ar",
            8,
            10,
            &ev(t, 80, 0, 500, 0, 0, 80, 0, 0)
        )
        .is_empty());
        assert!(lane_coverage(
            Topology::Pp2,
            "lane_batched",
            8,
            10,
            &ev(t, 80, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(lane_coverage(
            Topology::Pp2,
            "lane_adaptive",
            8,
            10,
            &ev(t, 80, 300, 40, 5, 78, 2, 0, 0)
        )
        .is_empty());
        assert!(lane_coverage(
            Topology::Pp2,
            "lane_copy",
            8,
            10,
            &ev(t, 80, 100, 0, 0, 0, 0, 60, 60)
        )
        .is_empty());
        // Violations.
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_ar",
            8,
            10,
            &ev(t, 80, 3, 500, 0, 0, 80, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_ar",
            8,
            10,
            &ev(t, 80, 0, 0, 0, 0, 80, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_batched",
            8,
            10,
            &ev(t, 80, 400, 1, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_batched",
            8,
            10,
            &ev(t, 80, 0, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_adaptive",
            8,
            10,
            &ev(t, 80, 300, 40, 5, 70, 2, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_copy",
            8,
            10,
            &ev(t, 80, 100, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_copy",
            8,
            10,
            &ev(t, 80, 100, 0, 0, 0, 0, 60, 59)
        )
        .is_empty());
        // Cohort size is checked against trials x lanes.
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_batched",
            8,
            10,
            &ev(t, 10, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(lane_coverage(
            Topology::Pp2,
            "lane_batched",
            1,
            10,
            &ev(t, 10, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_batched",
            1,
            11,
            &ev(t, 10, 400, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
        // A missing counter is a failure, not zero.
        let mut missing = ev(t, 80, 400, 0, 0, 0, 0, 0, 0);
        missing.as_object_mut().unwrap().remove("ar_windows");
        let failures = lane_coverage(Topology::Pp2, "lane_batched", 8, 10, &missing);
        assert!(
            failures.iter().any(|f| f.contains("ar_windows")),
            "{failures:?}"
        );
        assert!(!lane_coverage(
            Topology::Pp2,
            "lane_interleaved",
            8,
            10,
            &ev(t, 80, 1, 0, 0, 0, 0, 0, 0)
        )
        .is_empty());
    }
}
