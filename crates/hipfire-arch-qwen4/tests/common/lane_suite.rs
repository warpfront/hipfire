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
//! A shard's key is `(executor, topology, artifact, lanes, route, sampling,
//! state format)` plus its `range`. [`plan_lane_merge`] accepts a set only when
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

/// The lane counts the suite is declared for.
pub const LANE_COUNTS: [usize; 2] = [1, 8];

/// The declared lane arm set, reference first.
pub const LANE_ARMS: [&str; 4] = ["lane_ar", "lane_batched", "lane_adaptive", "lane_copy"];
/// The arm every other lane arm is compared against.
pub const LANE_REFERENCE_ARM: &str = "lane_ar";
/// The arms of an `ep2` cohort: the copy lane is off by default on EP serve
/// (`HIPFIRE_EP_MTP_LANE_COPY`), so no `ep2` cohort or verdict covers it and
/// `lane_copy` is refused under `ep2`.
pub const EP2_ARMS: [&str; 3] = ["lane_ar", "lane_batched", "lane_adaptive"];

/// The arms a cohort of `topology` must hold (reference first): the whole
/// declared set, except under `ep2` ([`EP2_ARMS`]).
pub fn cohort_arms(topology: Topology) -> &'static [&'static str] {
    match topology {
        Topology::Ep2 => &EP2_ARMS,
        Topology::Single | Topology::Pp2 => &LANE_ARMS,
    }
}

/// The sidecar/config schema version of the `lane_suite` block.
pub const SCHEMA: u64 = 1;

pub fn is_lane_arm(arm: &str) -> bool {
    LANE_ARMS.contains(&arm)
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

/// `HIPFIRE_SAMPLED_MTP_LANES`: required (`1` or `8`) under the lane executor
/// so the lane count is always a recorded part of the identity; refused under
/// the singleton executor, where it would mislabel the run.
pub fn parse_lanes(value: Option<&str>, executor: Executor) -> Result<usize, String> {
    match (executor, value) {
        (Executor::Singleton, None) => Ok(1),
        (Executor::Singleton, Some(v)) => Err(format!(
            "{LANES_ENV}={v:?} is set but {EXECUTOR_ENV} is not `lanes`: the singleton arms run one request; unset {LANES_ENV}"
        )),
        (Executor::Lanes, None) => Err(format!(
            "{EXECUTOR_ENV}=lanes needs {LANES_ENV}=1 or 8 (the lane count is part of every shard's identity)"
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

/// Executor, topology and lane count of this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub executor: Executor,
    pub topology: Topology,
    pub lanes: usize,
}

impl Placement {
    pub fn from_env() -> Result<Self, String> {
        let executor = Executor::from_env()?;
        Ok(Self {
            executor,
            topology: Topology::from_env()?,
            lanes: lanes_from_env(executor)?,
        })
    }
}

/// Whether `arm` (run in `mode`) may run under `placement`.
///
/// * Singleton executor: the four short leviathan arms take `single` or `pp2`
///   (`short_arms` is that set); every other arm runs on one card only; `ep2`
///   has no singleton arm.
/// * Lane executor: only the declared lane arms, leviathan mode, any topology.
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
                    "{EXECUTOR_ENV}=lanes runs only the lane arms {LANE_ARMS:?}, not arm {arm}; unset {EXECUTOR_ENV} for the singleton arms"
                ));
            }
            if mode != "leviathan" {
                return Err(format!(
                    "lane arm {arm} runs the leviathan verifier only (mode {mode})"
                ));
            }
            if !cohort_arms(placement.topology).contains(&arm) {
                return Err(format!(
                    "lane arm {arm} is not part of the {} cohort {:?}: the copy lane is off by default on EP serve and no ep2 verdict covers it",
                    placement.topology.as_str(),
                    cohort_arms(placement.topology)
                ));
            }
            Ok(())
        }
        Executor::Singleton => {
            if is_lane_arm(arm) {
                return Err(format!(
                    "arm {arm} is a lane arm: set {EXECUTOR_ENV}=lanes and {LANES_ENV}=1|8"
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

/// The route label of a lane arm (the `route` merge-key field).
pub fn lane_route(arm: &str) -> Option<&'static str> {
    match arm {
        "lane_ar" => Some("ar"),
        "lane_batched" => Some("batched"),
        "lane_adaptive" => Some("adaptive"),
        "lane_copy" => Some("copy"),
        _ => None,
    }
}

/// `HIPFIRE_EP_MTP_LANE_MAX_ACTIVE`: the largest lane count of a step that
/// still runs native windows (the mesh step switches to AR windows from
/// `max + 2` lanes up).
pub const MAX_ACTIVE_ENV: &str = "HIPFIRE_EP_MTP_LANE_MAX_ACTIVE";
/// The value the native-MTP lane arms pin: at least the largest suite lane
/// count ([`LANE_COUNTS`]), so the 8-lane arms really run native windows.
pub const LANE_MAX_ACTIVE: &str = "8";

/// Extra environment a lane arm pins (beyond the route/floor knobs the shared
/// `arm_overrides` derives from the arm name): `lane_copy` always runs the copy
/// lane, and every native-MTP arm pins [`MAX_ACTIVE_ENV`] to [`LANE_MAX_ACTIVE`]
/// (the AR reference store never switches windows, so it pins nothing).
pub fn lane_env_overrides(arm: &str) -> Vec<(&'static str, Option<&'static str>)> {
    match arm {
        "lane_copy" => vec![
            (NGRAM_COPY_ENV, Some("on")),
            (MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE)),
        ],
        "lane_batched" | "lane_adaptive" => vec![(MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE))],
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

/// Refuses a directory holding shards of any arm outside the cohort of
/// `topology` (singleton arms, an interleaved lane arm, `lane_copy` under
/// `ep2`, a typo): a lane merge never silently ignores foreign shards.
pub fn reject_foreign_shards(names: &[String], topology: Topology) -> Result<(), String> {
    let cohort = cohort_arms(topology);
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

/// Validates `shards` (arm -> its shards) against the lane arm set of the
/// `topology` cohort ([`cohort_arms`]) and the `trials` cohort. See the module
/// docs for the exact rules.
pub fn plan_lane_merge(
    shards: &BTreeMap<String, Vec<LaneShard>>,
    trials: usize,
    topology: Topology,
) -> Result<LaneMergePlan, String> {
    let cohort = cohort_arms(topology);
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
/// Values that must be identical in every shard (when present).
const EVIDENCE_EQUAL_KEYS: [&str; 1] = ["probe_budget"];

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
        other => failures.push(format!("{other} is not a lane arm")),
    }
    failures
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
        for bad in ["0", "2", "4", "16", "x", ""] {
            assert!(parse_lanes(Some(bad), Executor::Lanes).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn arm_validation_separates_executors() {
        for topology in [Topology::Single, Topology::Pp2, Topology::Ep2] {
            for &arm in cohort_arms(topology) {
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
        assert_eq!(LANE_ARMS[0], LANE_REFERENCE_ARM);
        assert!(lane_salt("ar").is_err() && lane_route("interleaved").is_none());
        assert_eq!(
            lane_env_overrides("lane_copy"),
            vec![
                (NGRAM_COPY_ENV, Some("on")),
                (MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE))
            ]
        );
        for arm in ["lane_batched", "lane_adaptive"] {
            assert_eq!(
                lane_env_overrides(arm),
                vec![(MAX_ACTIVE_ENV, Some(LANE_MAX_ACTIVE))]
            );
        }
        assert!(lane_env_overrides("lane_ar").is_empty());
    }

    #[test]
    fn the_native_window_pin_covers_every_suite_lane_count() {
        let pin: usize = LANE_MAX_ACTIVE.parse().unwrap();
        assert!(LANE_COUNTS.iter().all(|&lanes| pin >= lanes));
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
        json!({
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
        })
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
            ("schema", |c| c["lane_suite"]["schema"] = json!(2)),
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
    fn the_ep2_cohort_is_the_three_mesh_arms() {
        assert_eq!(cohort_arms(Topology::Ep2), &EP2_ARMS[..]);
        assert_eq!(cohort_arms(Topology::Pp2), &LANE_ARMS[..]);
        assert_eq!(cohort_arms(Topology::Single), &LANE_ARMS[..]);
        assert_eq!(EP2_ARMS[0], LANE_REFERENCE_ARM);
        let plan = plan_lane_merge(&ep2_set(), TRIALS, Topology::Ep2).unwrap();
        assert_eq!(plan.topology, "ep2");
        assert_eq!(plan.arms.len(), 3);
        assert!(!plan.arms.contains_key("lane_copy"));
        // The same shards are an incomplete set of the four-arm cohort.
        assert!(plan_lane_merge(&ep2_set(), TRIALS, Topology::Pp2)
            .unwrap_err()
            .contains("lane_copy"));
        // A missing ep2 arm and a copy shard under ep2 are both refused.
        let mut set = ep2_set();
        set.remove("lane_adaptive");
        assert!(plan_lane_merge(&set, TRIALS, Topology::Ep2)
            .unwrap_err()
            .contains("lane_adaptive"));
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
