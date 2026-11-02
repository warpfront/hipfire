// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! fn_mesh_lane_oracle: Flash-Next continuous-batching lane exactness oracle (lab).
//!
//! The greedy ids of every concurrently batched request (C1/C2/C4/C8 lanes)
//! must equal that request's serialized singleton run. The model loads once
//! through `hipfire_loader::load_model` exactly as the daemon (and
//! `fn_ep_oracle`) does; expert parallelism comes from `topology.ep` / the S1
//! env var, not from a flag here.
//!
//! 1. Reference: for each prompt, the serialized singleton greedy run
//!    (`reset`, `prefill_final`, then `--tokens` x
//!    `forward_token_or_argmax(None, logits)`), the `fn_ep_oracle` AR loop.
//!    `ids[s-1]` is the token the s-th forward returned (the greedy pick of the
//!    logits the previous forward left); with `--logits` the SHA-256 of the
//!    logits row after the prefill (index 0) and after every forward
//!    (index 1..=T).
//! 2. Lanes: staged once (`max_lanes = max(ks)`, `--row-budget` decode rows,
//!    default `max_lanes`, no spec). For each K the first K prompts run as K
//!    concurrent lanes with fresh epochs. Every step plans one prefill tile
//!    (`exact_prefill_chunk_len`) for every Prefilling lane AND one AR row (the
//!    pending seed) for every Decoding lane, so lanes reach decode at different
//!    steps and prefill tiles share steps with AR rows. A lane's committed ids
//!    are: the pick of its prompt-completing tile, then the pick of each AR
//!    row; the first T are its ids (the T-th AR row's own pick is the T+1-th
//!    commit and is dropped). With `--logits`, the logits row of the
//!    prompt-completing tile (index 0) and of every AR row (1..=T) are hashed.
//! 3. `DIR/report.json` (identity, topology, per-K per-prompt comparison,
//!    timings of the decode-only steps), `DIR/ids.jsonl` (full id lists).
//!
//!    `--policies SPEC[;SPEC...]` repeats step 2 once per lane stage share
//!    policy (`production`, `per-lane`, `per-lane:<stage>=once,...`
//!    ([`LaneStagePolicy::parse`]); `g0` expands to `per-lane`, then
//!    `per-lane:<stage>=once` for each non-stateful stage in `StageName::ALL`
//!    order, then `production`). The reference runs once; each policy is set
//!    on every lane store of the load (`Qwen4Bundle::set_lane_stage_policy`:
//!    single, every EP rank, every PP band) and the whole `--ks` matrix is
//!    compared against the same reference. The first policy runs `--tokens`
//!    per lane, later ones `--tokens-sweep` (default `--tokens`) compared
//!    against the reference prefix. Without `--policies` the store's
//!    construction policy runs alone. Every policy is reported (`policies` in
//!    report.json, a `policy` field in ids.jsonl); a failing policy does not
//!    stop the sweep, and an execution error stops it only when the lane
//!    store is no longer usable.
//!
//! Exit 0: every compared K/prompt equal. 1: mismatch. 2: refusal or error.
//!
//! The lane driver runs behind [`LaneBackend`]; [`LocalLanes`] is the Single/EP
//! implementation (the bundle's lane API, replicated across ranks inside the
//! arch crate under EP) and [`PpLanes`] the pipeline-parallel one
//! (`pp_stage_lanes`, `pp_admit_lane`, `pp_lane_executor` over the model's
//! `Gpus`; the root = output-band store is authoritative for the host reads).
//! The serialized singleton reference runs behind [`Singleton`] the same way
//! (`prefill_final`/`forward_token_or_argmax`, or under `--pp N>1`
//! `pp_prefill_final`/`pp_forward_token_or_argmax`, which reset every band).
//!
//! Usage:
//! ```text
//! fn_mesh_lane_oracle --model M.mq4 --prompt-file a.txt [--prompt-file b.ids ...] \
//!     --max-seq 4096 --tokens 64 --ks 1,2,4,8 --out DIR \
//!     [--pp 1] [--row-budget R] [--skip-model-hash] [--logits]
//!     [--policies 'g0' | 'per-lane;per-lane:moe=once;production'] [--tokens-sweep N]
//!     [--burst I,J,...]   (repeatable; see "Bursts" below)
//! ```
//!
//! Bursts (`--burst I,J,...`, repeatable): after the `--ks` runs (the first K prompts), each
//! `--burst` is one more concurrent run of exactly the listed prompt-file indices in that lane
//! order, so one process covers uniform and mixed-length bursts of 1..max lanes against ONE
//! reference pass per prompt. Per prompt the report gives `prompt_len`, `lane` and
//! `first_divergence` ({row, position, phase, via}: row 0 is the prompt-completing row, row r
//! the r-th AR row; position is the absolute sequence position). Per run `prefill` has the
//! burst's wall, rows, ms/row and one record per prefill pass (tiles, rows, AR rows sharing the
//! step), `prompt_lens`, `rows_total` and `singleton_prefill_ms_sum` (the reference prefills of
//! the same prompts) beside `burst_vs_singleton_sum`. stderr carries one `[oracle-burst]` line
//! per run (read by `d1/scripts/wave-table.py`).
//!
//! Speculative-lane schema (CLI contract; execution is an explicit boundary):
//! ```text
//!     [--spec ar|mtp] [--k K] [--routes floor|cycle|batched]
//!     [--soak-seconds S [--soak-seed N]] [--fault-point POINT]
//! ```
//! `--spec ar` (default) is the oracle above, unchanged. `--spec mtp` selects
//! the native-MTP lane arm: `--k` is the draft depth (1..=7, default 3) and
//! `--routes` the window route policy (`floor` = production floor, `batched` =
//! forced batched windows, `cycle` = deterministic native/copy/AR mix, default
//! `cycle`), as in `fn_lane_oracle`. `--soak-seconds S` runs the bounded
//! mixed-lifecycle soak (lanes 1..=max(ks), cycling native/copy/AR, late
//! arrivals, cancellations and slot reuse from the recorded `--soak-seed`);
//! `--fault-point` injects one all-owner abort at `before-draft`,
//! `head-round`, `target-edge`, `repair` or `before-publish` and expects the
//! bounded join and quarantine. `--k`, `--routes`, `--soak-seconds` and
//! `--fault-point` need `--spec mtp`; soak and a fault point are exclusive.
//! Every `--spec mtp` invocation is parsed, validated and recorded in
//! `manifest.json`, then fails with "not yet implemented by S3/S5" until the
//! EP/PP speculative lane protocol lands: that refusal is the interface
//! boundary, never a pass. `DIR/manifest.json` (all specs) holds the expanded
//! command, the SHA-256 of every prompt fixture and of this binary, the
//! model hash (once computed) and the recorded environment.
//!
//! `--pp N>1` loads the model on a PP mesh (the `pp` argument of `load_model`,
//! as `fn_ep_oracle`); the topology (`pp`, `band_counts`, `output_owner`,
//! `devices`) is recorded in `report.json`.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::ep_mesh_lanes::{EpMeshLanes, MeshLaneAdmission, MeshStepError};
use hipfire_arch_qwen4::gpu_forward_lanes::{LaneStagePolicy, StageName};
use hipfire_arch_qwen4::lane::{LabRoutes, LaneSpecConfig};
use hipfire_arch_qwen4::mesh_lane_protocol::{
    GreedyRule, LaneRequest, SamplingMode, WindowBudgets,
};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const TOOL: &str = "fn_mesh_lane_oracle";

fn e<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ---------------------------------------------------------------- hashing

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> R<String> {
    let mut file =
        std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(e)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// RFC 1321 MD5 (identity tags only, never a security boundary).
fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for chunk in message.chunks_exact(64) {
        let m: Vec<u32> = chunk
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a
                .wrapping_add(f)
                .wrapping_add(k[i])
                .wrapping_add(m[g])
                .rotate_left(S[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = Vec::with_capacity(16);
    for word in [a0, b0, c0, d0] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    hex(&out)
}

fn ids_md5(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    md5_hex(&bytes)
}

// ------------------------------------------------------------------- args

#[derive(Debug, PartialEq)]
struct Args {
    model: String,
    prompt_files: Vec<String>,
    max_seq: usize,
    tokens: usize,
    /// Ascending, unique lane counts.
    ks: Vec<usize>,
    /// `--burst I,J,...` (repeatable): extra concurrent runs of the listed prompt-file indices
    /// (0-based, in lane order, unique within a burst, any lengths) after the `--ks` runs.
    bursts: Vec<Vec<usize>>,
    out: PathBuf,
    /// Pipeline-parallel degree handed to `load_model`; > 1 runs the PP arm.
    pp: usize,
    row_budget: Option<usize>,
    skip_model_hash: bool,
    logits: bool,
    /// Policies to sweep, in order (never empty); default is the store's
    /// construction policy only.
    policies: Vec<PolicyEntry>,
    /// Tokens per lane for every policy after the first (`<= tokens`).
    tokens_sweep: Option<usize>,
    /// Loader-planned AR lane capacity (`--reserve-lanes N`): the admitted
    /// route's per-load VMM batch request. `None` = plain `load_model`.
    reserve_lanes: Option<usize>,
    /// `--spec`; the AR oracle unless `mtp`.
    spec: SpecMode,
    /// Draft depth; `Some` exactly under `--spec mtp` (default [`DEFAULT_K`]).
    k: Option<usize>,
    /// Route policy; `Some` exactly under `--spec mtp` (default cycle).
    routes: Option<RoutesArg>,
    /// Bounded mixed-lifecycle soak, seconds (`--spec mtp` only).
    soak_seconds: Option<u64>,
    /// Lifecycle seed of the soak; `Some` exactly when `soak_seconds` is.
    soak_seed: Option<u64>,
    /// Injected abort point (`--spec mtp` only, exclusive with the soak).
    fault_point: Option<FaultPoint>,
    /// `--copy` (`--spec mtp` only): stage the lanes with the copy lane armed
    /// (default-copy budget: widened rings/rows, head-lag storage).
    copy: bool,
}

/// `--burst I,J,K`: prompt-file indices of one concurrent run, in lane (admission) order.
fn parse_burst(value: &str) -> R<Vec<usize>> {
    let members = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|x| format!("--burst {value:?}: bad prompt index {part:?}: {x}"))
        })
        .collect::<R<Vec<usize>>>()?;
    if members.is_empty() {
        return Err("--burst needs at least one prompt index".into());
    }
    let unique: BTreeSet<usize> = members.iter().copied().collect();
    if unique.len() != members.len() {
        return Err(format!(
            "--burst {value:?}: a prompt index appears twice (one lane per prompt)"
        ));
    }
    Ok(members)
}

/// One concurrent lane run: the prompt-file indices of its lanes, in lane order. A `--ks` run
/// is the first K prompts; a `--burst` run is any list (mixed lengths).
#[derive(Clone, Debug, PartialEq, Eq)]
struct RunSet {
    members: Vec<usize>,
    burst: bool,
}

impl RunSet {
    /// `K=4` for a `--ks` run, `burst=3+1+0` for a `--burst` run.
    fn label(&self) -> String {
        if self.burst {
            format!(
                "burst={}",
                self.members
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("+")
            )
        } else {
            format!("K={}", self.members.len())
        }
    }
}

/// Every run of a policy, in order: the `--ks` runs, then the `--burst` runs.
fn run_sets(args: &Args) -> Vec<RunSet> {
    args.ks
        .iter()
        .map(|&k| RunSet {
            members: (0..k).collect(),
            burst: false,
        })
        .chain(args.bursts.iter().map(|members| RunSet {
            members: members.clone(),
            burst: true,
        }))
        .collect()
}

impl Args {
    /// Lanes to stage: the largest `--ks` entry or `--burst` length.
    fn max_lanes(&self) -> usize {
        let bursts = self.bursts.iter().map(Vec::len).max().unwrap_or(0);
        self.ks.last().copied().unwrap_or(0).max(bursts)
    }
}

fn parse_ks(value: &str) -> R<Vec<usize>> {
    let ks: BTreeSet<usize> = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let k = part
                .trim()
                .parse::<usize>()
                .map_err(|x| format!("--ks: bad number {part:?}: {x}"))?;
            if k == 0 {
                return Err("--ks: lane counts start at 1".to_string());
            }
            Ok(k)
        })
        .collect::<R<_>>()?;
    if ks.is_empty() {
        return Err("--ks needs at least one lane count".into());
    }
    Ok(ks.into_iter().collect())
}

/// One policy of the sweep. `policy: None` is the store's own construction
/// policy (the default run).
#[derive(Debug, Clone, PartialEq, Eq)]
struct PolicyEntry {
    spec: String,
    policy: Option<LaneStagePolicy>,
}

/// Spec of the default entry (no `--policies`).
const DEFAULT_POLICY_SPEC: &str = "default";

/// `g0`: all per-lane, each non-stateful stage alone shared once over all
/// rows (`StageName::ALL` order), then the production policy.
fn g0_specs() -> Vec<String> {
    let mut specs = vec!["per-lane".to_string()];
    specs.extend(
        StageName::ALL
            .iter()
            .filter(|stage| !stage.stateful())
            .map(|stage| format!("per-lane:{}=once", stage.key())),
    );
    specs.push("production".to_string());
    specs
}

/// `SPEC[;SPEC...]`, `g0` expanding in place; every spec is validated.
fn parse_policies(value: &str) -> R<Vec<PolicyEntry>> {
    let mut entries = Vec::new();
    for raw in value
        .split(';')
        .map(str::trim)
        .filter(|spec| !spec.is_empty())
    {
        let specs = if raw == "g0" {
            g0_specs()
        } else {
            vec![raw.to_string()]
        };
        for spec in specs {
            let policy = LaneStagePolicy::parse(&spec)
                .map_err(|x| format!("--policies: bad spec {spec:?}: {x}"))?;
            entries.push(PolicyEntry {
                spec,
                policy: Some(policy),
            });
        }
    }
    if entries.is_empty() {
        return Err("--policies needs at least one spec".into());
    }
    Ok(entries)
}

// ------------------------------------------------------------- spec schema

/// `--spec`: the AR oracle (default) or the native-MTP lane arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpecMode {
    Ar,
    Mtp,
}

impl SpecMode {
    fn parse(value: &str) -> R<Self> {
        match value {
            "ar" => Ok(Self::Ar),
            "mtp" => Ok(Self::Mtp),
            other => Err(format!("--spec must be ar|mtp, got {other:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ar => "ar",
            Self::Mtp => "mtp",
        }
    }
}

/// `--routes`: the window route policy of an MTP run (`fn_lane_oracle`'s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutesArg {
    Floor,
    Cycle,
    Batched,
}

impl RoutesArg {
    fn parse(value: &str) -> R<Self> {
        match value {
            "floor" => Ok(Self::Floor),
            "cycle" => Ok(Self::Cycle),
            "batched" => Ok(Self::Batched),
            other => Err(format!(
                "--routes must be floor|cycle|batched, got {other:?}"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Cycle => "cycle",
            Self::Batched => "batched",
        }
    }
}

/// `--fault-point`: where an injected failure must abort every owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultPoint {
    BeforeDraft,
    HeadRound,
    TargetEdge,
    Repair,
    BeforePublish,
}

impl FaultPoint {
    const ALL: [FaultPoint; 5] = [
        Self::BeforeDraft,
        Self::HeadRound,
        Self::TargetEdge,
        Self::Repair,
        Self::BeforePublish,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::BeforeDraft => "before-draft",
            Self::HeadRound => "head-round",
            Self::TargetEdge => "target-edge",
            Self::Repair => "repair",
            Self::BeforePublish => "before-publish",
        }
    }

    fn parse(value: &str) -> R<Self> {
        Self::ALL
            .into_iter()
            .find(|point| point.name() == value)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|point| point.name()).collect();
                format!(
                    "--fault-point must be one of {}, got {value:?}",
                    names.join("|")
                )
            })
    }
}

/// Default draft depth of `--spec mtp` (`fn_lane_oracle`'s).
const DEFAULT_K: usize = 3;
/// Deepest native draft.
const MAX_K: usize = 7;
/// Soak lifecycle seed when `--soak-seconds` is given without `--soak-seed`.
const DEFAULT_SOAK_SEED: u64 = 0x50A4_5EED;

fn parse_args_from(mut args: impl Iterator<Item = String>) -> R<Args> {
    let mut model = None;
    let mut prompt_files = Vec::new();
    let mut max_seq = None;
    let mut tokens = None;
    let mut ks = None;
    let mut bursts: Vec<Vec<usize>> = Vec::new();
    let mut out = None;
    let mut pp = 1usize;
    let mut row_budget = None;
    let mut skip_model_hash = false;
    let mut logits = false;
    let mut policies = None;
    let mut tokens_sweep = None;
    let mut reserve_lanes = None;
    let mut copy = false;
    let mut spec = SpecMode::Ar;
    let mut k = None;
    let mut routes = None;
    let mut soak_seconds = None;
    let mut soak_seed = None;
    let mut fault_point = None;
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(value("--model")?),
            "--prompt-file" => prompt_files.push(value("--prompt-file")?),
            "--max-seq" => max_seq = Some(value("--max-seq")?.parse::<usize>().map_err(e)?),
            "--tokens" => tokens = Some(value("--tokens")?.parse::<usize>().map_err(e)?),
            "--ks" => ks = Some(parse_ks(&value("--ks")?)?),
            "--burst" => bursts.push(parse_burst(&value("--burst")?)?),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--pp" => pp = value("--pp")?.parse::<usize>().map_err(e)?,
            "--row-budget" => {
                row_budget = Some(value("--row-budget")?.parse::<usize>().map_err(e)?)
            }
            "--skip-model-hash" => skip_model_hash = true,
            "--logits" => logits = true,
            "--policies" => policies = Some(parse_policies(&value("--policies")?)?),
            "--tokens-sweep" => {
                tokens_sweep = Some(value("--tokens-sweep")?.parse::<usize>().map_err(e)?)
            }
            "--reserve-lanes" => {
                reserve_lanes = Some(value("--reserve-lanes")?.parse::<usize>().map_err(e)?)
            }
            "--spec" => spec = SpecMode::parse(&value("--spec")?)?,
            "--copy" => copy = true,
            "--k" => k = Some(value("--k")?.parse::<usize>().map_err(e)?),
            "--routes" => routes = Some(RoutesArg::parse(&value("--routes")?)?),
            "--soak-seconds" => {
                soak_seconds = Some(value("--soak-seconds")?.parse::<u64>().map_err(e)?)
            }
            "--soak-seed" => soak_seed = Some(value("--soak-seed")?.parse::<u64>().map_err(e)?),
            "--fault-point" => fault_point = Some(FaultPoint::parse(&value("--fault-point")?)?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if pp == 0 {
        return Err("--pp must be at least 1".into());
    }
    let ks = ks.ok_or("--ks is required")?;
    let tokens = tokens.ok_or("--tokens is required")?;
    if tokens == 0 {
        return Err("--tokens must be at least 1".into());
    }
    if row_budget == Some(0) {
        return Err("--row-budget must be at least 1".into());
    }
    if let Some(sweep) = tokens_sweep {
        if sweep == 0 || sweep > tokens {
            return Err(format!(
                "--tokens-sweep must be in 1..=--tokens ({tokens}), got {sweep}"
            ));
        }
    }
    for burst in &bursts {
        if let Some(&bad) = burst.iter().find(|&&i| i >= prompt_files.len()) {
            return Err(format!(
                "--burst index {bad} needs at least {} --prompt-file arguments, got {}",
                bad + 1,
                prompt_files.len()
            ));
        }
    }
    let need = *ks.last().expect("ks is non-empty");
    if prompt_files.len() < need {
        return Err(format!(
            "--ks up to {need} needs at least {need} --prompt-file arguments, got {}",
            prompt_files.len()
        ));
    }
    if reserve_lanes == Some(0) {
        return Err("--reserve-lanes must be at least 1".into());
    }
    if spec == SpecMode::Ar {
        for (given, flag) in [
            (k.is_some(), "--k"),
            (routes.is_some(), "--routes"),
            (soak_seconds.is_some(), "--soak-seconds"),
            (soak_seed.is_some(), "--soak-seed"),
            (fault_point.is_some(), "--fault-point"),
        ] {
            if given {
                return Err(format!("{flag} needs --spec mtp"));
            }
        }
    } else {
        let depth = *k.get_or_insert(DEFAULT_K);
        if depth == 0 || depth > MAX_K {
            return Err(format!("--k must be in 1..={MAX_K}, got {depth}"));
        }
        routes.get_or_insert(RoutesArg::Cycle);
    }
    if copy && spec != SpecMode::Mtp {
        return Err("--copy needs --spec mtp".into());
    }
    if soak_seconds == Some(0) {
        return Err("--soak-seconds must be at least 1".into());
    }
    if soak_seed.is_some() && soak_seconds.is_none() {
        return Err("--soak-seed needs --soak-seconds".into());
    }
    if soak_seconds.is_some() && fault_point.is_some() {
        return Err(
            "--soak-seconds and --fault-point are exclusive: run injected failures separately from the uninterrupted soak"
                .into(),
        );
    }
    if soak_seconds.is_some() {
        soak_seed.get_or_insert(DEFAULT_SOAK_SEED);
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt_files,
        max_seq: max_seq.ok_or("--max-seq is required")?,
        tokens,
        ks,
        bursts,
        out: out.ok_or("--out is required")?,
        pp,
        row_budget,
        skip_model_hash,
        logits,
        policies: policies.unwrap_or_else(|| {
            vec![PolicyEntry {
                spec: DEFAULT_POLICY_SPEC.to_string(),
                policy: None,
            }]
        }),
        tokens_sweep,
        reserve_lanes,
        spec,
        k,
        routes,
        soak_seconds,
        soak_seed,
        fault_point,
        copy,
    })
}

const HELP: &str = "fn_mesh_lane_oracle --model M.mq4 --prompt-file P [--prompt-file P ...] \
--max-seq N --tokens T --ks 1,2,4,8 --out DIR [--pp 1] [--row-budget R] [--skip-model-hash] [--logits] \
[--burst I,J,...] [--policies SPEC[;SPEC...]] [--tokens-sweep N] [--reserve-lanes N] \
[--spec ar|mtp] [--k K] [--routes floor|cycle|batched] [--soak-seconds S [--soak-seed N]] \
[--fault-point before-draft|head-round|target-edge|repair|before-publish]\n\
EP comes from topology.ep / the S1 env var as in fn_ep_oracle; --pp N>1 runs the PP lanes arm. \
--policies sweeps lane stage share policies (production | per-lane | per-lane:<stage>=once,... | g0) \
over one reference; --tokens-sweep N sets tokens per lane for every policy after the first. \
--reserve-lanes N loads through the admitted route with the daemon's VMM-batch request (serve.vmm_batch, \
continuous_batch_size N) so a qwen4 EP load reserves N AR lanes before the expert split (planner capacity \
only; default off = plain load_model, N=1 is the same plan). \
--spec mtp (--k 1..=7, default 3; --routes default cycle) selects the native-MTP lane arm; --soak-seconds and \
--fault-point need it and are exclusive. Its execution is not yet implemented by S3/S5: the invocation is \
validated and recorded in DIR/manifest.json (expanded command, prompt fixture hashes, binary sha), then refused. \
Needs at least max(ks) prompt files. \
DIR/manifest.json is written for every run. \
Exit 0 all equal, 1 mismatch, 2 refusal/error. DIR must not exist.";

// ---------------------------------------------------------------- prompts

struct Prompt {
    path: String,
    ids: Vec<u32>,
    file_md5: String,
    ids_md5: String,
    kind: &'static str,
}

/// Raw text (tokenized) or a `.ids` file of whitespace-separated token ids,
/// exactly as `fn_ep_oracle`.
fn load_prompt(path: &str, tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> R<Prompt> {
    let bytes = std::fs::read(path).map_err(|x| format!("read {path}: {x}"))?;
    let file_md5 = md5_hex(&bytes);
    let (ids, kind) = if path.ends_with(".ids") {
        let text = std::str::from_utf8(&bytes).map_err(|x| format!("{path}: {x}"))?;
        let ids = text
            .split_whitespace()
            .map(|word| {
                word.parse::<u32>()
                    .map_err(|x| format!("{path}: bad id {word:?}: {x}"))
            })
            .collect::<R<Vec<u32>>>()?;
        (ids, "ids")
    } else {
        let text = std::str::from_utf8(&bytes).map_err(|x| format!("{path}: not UTF-8: {x}"))?;
        (tokenizer.encode(text), "text")
    };
    if ids.is_empty() {
        return Err(format!("{path}: empty prompt"));
    }
    Ok(Prompt {
        path: path.to_string(),
        file_md5,
        ids_md5: ids_md5(&ids),
        ids,
        kind,
    })
}

// ----------------------------------------------------------------- device

fn tensor_bytes(gpu: &Gpu, tensor: &GpuTensor) -> R<Vec<u8>> {
    gpu.hip.device_synchronize().map_err(e)?;
    gpu.download_raw_bytes(tensor).map_err(e)
}

fn device_info(gpu: &Gpu) -> Value {
    json!({
        "device_id": gpu.device_id,
        "arch": gpu.arch,
        "pci_bus_id": gpu.hip.device_pci_bus_id(gpu.device_id).unwrap_or_else(|x| format!("error: {x}")),
    })
}

/// `elems` f32 elements of `t` starting at element `offset`, as raw bytes.
fn read_view(gpu: &Gpu, t: &GpuTensor, offset: usize, elems: usize) -> R<Vec<u8>> {
    let view = t.sub_offset(offset, elems);
    let bytes = view.byte_size();
    if bytes > view.buf.size() {
        return Err(format!(
            "read of {bytes} bytes exceeds the {}-byte device buffer",
            view.buf.size()
        ));
    }
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view.buf).map_err(e)?;
    Ok(out)
}

// ------------------------------------------------------------- comparison

/// Index of the first differing element; a pure length difference is the
/// shorter length. `None` when equal.
fn first_mismatch<T: PartialEq>(reference: &[T], got: &[T]) -> Option<usize> {
    match reference.iter().zip(got).position(|(a, b)| a != b) {
        Some(index) => Some(index),
        None if reference.len() != got.len() => Some(reference.len().min(got.len())),
        None => None,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Comparison {
    /// ids equal and (when logits were compared) every logits row equal.
    equal: bool,
    first_mismatch: Option<usize>,
    ref_md5: String,
    lane_md5: String,
    /// Digest of every compared logits row's sha (`--logits` only).
    ref_logits_sha: Option<String>,
    lane_logits_sha: Option<String>,
    /// `Some(first differing logits index)` per `--logits`; inner `None` = equal.
    logits_first_mismatch: Option<Option<usize>>,
}

/// One digest of a run's per-row logits shas (rows joined by newlines).
fn rows_digest(rows: &[String]) -> String {
    sha256_hex(rows.join("\n").as_bytes())
}

/// KL(p || q) in nats between the softmax of two raw f32 logits rows (f64 log-softmax).
fn row_kl(reference: &[u8], lane: &[u8]) -> Option<f64> {
    if reference.len() != lane.len() || reference.len() % 4 != 0 || reference.is_empty() {
        return None;
    }
    let f = |b: &[u8]| -> Vec<f64> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()
    };
    let (p, q) = (f(reference), f(lane));
    let lse = |x: &[f64]| -> f64 {
        let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(&p), lse(&q));
    let mut kl = 0.0;
    for i in 0..p.len() {
        let lpi = p[i] - lp;
        kl += lpi.exp() * (lpi - (q[i] - lq));
    }
    Some(kl)
}

/// Per-row KL of one lane vs its singleton reference over the rows computed from an identical
/// context: rows `0..=first id mismatch` (all rows when the ids agree). `(rows, max, mean)`.
fn lane_kl(
    reference_raw: &[Vec<u8>],
    lane_raw: &[Vec<u8>],
    first_id_mismatch: Option<usize>,
) -> Option<(usize, f64, f64)> {
    let rows = lane_raw.len().min(reference_raw.len());
    let upto = first_id_mismatch.map_or(rows, |m| (m + 1).min(rows));
    let kls: Vec<f64> = (0..upto)
        .filter_map(|r| row_kl(&reference_raw[r], &lane_raw[r]))
        .collect();
    if kls.is_empty() {
        return None;
    }
    let max = kls.iter().cloned().fold(0.0, f64::max);
    Some((kls.len(), max, kls.iter().sum::<f64>() / kls.len() as f64))
}

fn compare(
    reference_ids: &[u32],
    lane_ids: &[u32],
    logits: Option<(&[String], &[String])>,
) -> Comparison {
    let ids_mismatch = first_mismatch(reference_ids, lane_ids);
    let logits_first_mismatch = logits.map(|(reference, lane)| first_mismatch(reference, lane));
    Comparison {
        equal: ids_mismatch.is_none() && logits_first_mismatch.is_none_or(|m| m.is_none()),
        first_mismatch: ids_mismatch,
        ref_md5: ids_md5(reference_ids),
        lane_md5: ids_md5(lane_ids),
        ref_logits_sha: logits.map(|(reference, _)| rows_digest(reference)),
        lane_logits_sha: logits.map(|(_, lane)| rows_digest(lane)),
        logits_first_mismatch,
    }
}

fn comparison_json(prompt: usize, c: &Comparison) -> Value {
    let mut record = json!({
        "prompt": prompt,
        "equal": c.equal,
        "ids_equal": c.first_mismatch.is_none(),
        "first_mismatch": c.first_mismatch,
        "ref_md5": c.ref_md5,
        "lane_md5": c.lane_md5,
    });
    if let Some(mismatch) = c.logits_first_mismatch {
        record["logits_equal"] = json!(mismatch.is_none());
        record["first_logits_mismatch"] = json!(mismatch);
        // One digest over every compared logits row's sha: the cross-arm identity key
        // (`wave-table.py`: an exact wave's lane == the serial tile's lane of the same prompt).
        record["ref_logits_sha"] = json!(c.ref_logits_sha);
        record["lane_logits_sha"] = json!(c.lane_logits_sha);
    }
    record
}

/// Where a lane first differs from its singleton: the first differing ids index or logits row
/// (row 0 is the prompt-completing row, row r >= 1 the r-th AR row), the absolute sequence
/// position of the token that row's logits follow (`prompt_len - 1 + row`) and which compare
/// saw it. `None` when ids (and logits, when compared) are all equal.
fn first_divergence(prompt_len: usize, c: &Comparison) -> Option<Value> {
    let ids = c.first_mismatch;
    let logits = c.logits_first_mismatch.flatten();
    let row = match (ids, logits) {
        (None, None) => return None,
        (Some(a), None) | (None, Some(a)) => a,
        (Some(a), Some(b)) => a.min(b),
    };
    let via = match (ids == Some(row), logits == Some(row)) {
        (true, true) => "ids+logits",
        (true, false) => "ids",
        _ => "logits",
    };
    Some(json!({
        "row": row,
        "position": (prompt_len + row).saturating_sub(1),
        "phase": if row == 0 { "prefill-final" } else { "decode" },
        "via": via,
    }))
}

/// [`comparison_json`] plus the lane's place in its run, its prompt length and the first
/// divergence (`null` when equal).
fn compared_json(prompt: usize, prompt_len: usize, lane: usize, c: &Comparison) -> Value {
    let mut record = comparison_json(prompt, c);
    record["lane"] = json!(lane);
    record["prompt_len"] = json!(prompt_len);
    record["first_divergence"] = json!(first_divergence(prompt_len, c));
    record
}

/// The run-level fields of a lane run: its set, lengths, total rows, and the serialized
/// singleton prefills of the same prompts summed (`burst_vs_singleton_sum` = singleton sum /
/// the run's prefill wall: > 1 means the joint prefill beat prefilling the prompts one by one).
fn burst_extras(set: &RunSet, run: &KRun, prompts: &[&[u32]], references: &[Reference]) -> Value {
    let lens: Vec<usize> = set.members.iter().map(|&i| prompts[i].len()).collect();
    let single_ms: f64 = set.members.iter().map(|&i| references[i].prefill_ms).sum();
    json!({
        "set": set.label(),
        "burst": set.burst,
        "members": set.members,
        "prompt_lens": lens,
        "rows_total": lens.iter().sum::<usize>(),
        "singleton_prefill_ms_sum": single_ms,
        "burst_vs_singleton_sum": (run.prefill.wall > 0.0 && single_ms > 0.0)
            .then(|| single_ms / (run.prefill.wall * 1e3)),
    })
}

/// Merge the fields of the object `extra` into the object `record`.
fn with_extras(mut record: Value, extra: Value) -> Value {
    if let (Some(into), Value::Object(from)) = (record.as_object_mut(), extra) {
        into.extend(from);
    }
    record
}

/// The one stderr line per lane run the table scripts read (`wave-table.py`).
fn burst_line(spec: &str, set: &RunSet, run: &KRun, compared: &[Value], extras: &Value) -> String {
    let lens: Vec<u64> = compared
        .iter()
        .filter_map(|c| c["prompt_len"].as_u64())
        .collect();
    let divergences: Vec<(u64, u64, u64)> = compared
        .iter()
        .filter_map(|c| {
            let d = c.get("first_divergence")?;
            Some((
                c["prompt"].as_u64()?,
                d["row"].as_u64()?,
                d["position"].as_u64()?,
            ))
        })
        .collect();
    format!(
        "[oracle-burst] policy={spec} set={} lens={lens:?} prompts_equal={}/{} passes={} rows={} burst_prefill_ms={:.1} admit_ms={:.1} ms_per_row={:.3} singleton_sum_ms={:.1} first_divergence(prompt,row,pos)={divergences:?}",
        set.label(),
        compared.iter().filter(|c| c["equal"] == json!(true)).count(),
        compared.len(),
        run.prefill.steps,
        run.prefill.rows,
        run.prefill.wall * 1e3,
        run.prefill.admit_wall * 1e3,
        if run.prefill.rows > 0 {
            run.prefill.wall * 1e3 / run.prefill.rows as f64
        } else {
            f64::NAN
        },
        extras["singleton_prefill_ms_sum"].as_f64().unwrap_or(f64::NAN),
    )
}

// ------------------------------------------------------------- lane plans

/// One request's rows in a step.
struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    /// Absolute position of the first row.
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot `0..max_lanes` (idle slots
/// empty), requests in slot order, absolute positions.
fn build_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let mut plan = BatchStepPlan {
        batch: SlotBatch::build(&triples),
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: r.kind,
        });
        match r.kind {
            RequestStepKind::Ar => plan.decode_rows += r.tokens.len(),
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            RequestStepKind::Verify { .. } => plan.verify_rows += r.tokens.len(),
            RequestStepKind::Forced => plan.forced_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

/// Where each logits-producing request's row sits in the lane store's logits
/// tensor: the n-th AR request of the plan is row `n`; the i-th Prefill
/// request (counted over every Prefill request) is row `prefill_base + i`,
/// valid only when its tile completes the prompt (its last row has a pick).
fn logits_rows(
    plan: &BatchStepPlan,
    picks: &[u32],
    prefill_base: usize,
) -> Vec<(RequestEpoch, usize)> {
    let (mut ar, mut pre) = (0usize, 0usize);
    let mut rows = Vec::new();
    for request in &plan.requests {
        match request.kind {
            RequestStepKind::Ar => {
                rows.push((request.epoch, ar));
                ar += 1;
            }
            RequestStepKind::Prefill => {
                if picks
                    .get(request.rows.end() - 1)
                    .is_some_and(|&pick| pick != u32::MAX)
                {
                    rows.push((request.epoch, prefill_base + pre));
                }
                pre += 1;
            }
            _ => {}
        }
    }
    rows
}

/// One lane's host-side progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Progress {
    prompt_len: usize,
    prefilled: usize,
    ar_steps: usize,
}

/// The lanes a step plans: `(prefill, ar)` indices. Every Prefilling lane gets
/// its next tile; Decoding lanes (prompt done, fewer than `tokens` AR rows) get
/// one AR row each, at most `row_budget` of them, fewest AR rows first (ties by
/// index) so a small budget rotates fairly.
fn schedule(lanes: &[Progress], tokens: usize, row_budget: usize) -> (Vec<usize>, Vec<usize>) {
    let prefill: Vec<usize> = (0..lanes.len())
        .filter(|&i| lanes[i].prefilled < lanes[i].prompt_len)
        .collect();
    let mut ar: Vec<usize> = (0..lanes.len())
        .filter(|&i| lanes[i].prefilled == lanes[i].prompt_len && lanes[i].ar_steps < tokens)
        .collect();
    if ar.len() > row_budget {
        ar.sort_by_key(|&i| (lanes[i].ar_steps, i));
        ar.truncate(row_budget);
        ar.sort_unstable();
    }
    (prefill, ar)
}

// ----------------------------------------------------------- lane backend

/// What one executed step returns.
struct StepResult {
    advances: Vec<RequestAdvance>,
    /// Raw f32 logits row of every AR row and every prompt-completing prefill
    /// tile (empty unless requested).
    logits: Vec<(RequestEpoch, Vec<u8>)>,
}

/// The lane driver's view of a topology: stage, admit, run steps, release.
/// [`LocalLanes`] is Single/EP, [`PpLanes`] is PP (`pp_stage_lanes` /
/// `pp_admit_lane` / `pp_lane_executor`).
trait LaneBackend {
    /// Stage the lane store (`spec: None`).
    fn stage(&mut self, max_lanes: usize, row_budget: usize) -> R<()>;
    /// Lanes the staged store holds.
    fn lane_capacity(&self) -> R<usize>;
    /// Admit `epoch` with `prompt`; returns the lane slot.
    fn admit(&mut self, epoch: RequestEpoch, prompt: Vec<u32>) -> R<usize>;
    fn release(&mut self, epoch: &RequestEpoch) -> R<()>;
    /// The singleton tile length the lane's next prefill step must use.
    fn prefill_tile_len(&mut self, epoch: &RequestEpoch) -> R<usize>;
    /// provision -> forward -> (logits read) -> commit of one plan; a failure
    /// after provisioning aborts the step.
    fn step(&mut self, plan: &BatchStepPlan, read_logits: bool) -> R<StepResult>;
    /// The policy the staged store was constructed with (rank 0 / root store).
    fn stage_policy(&self) -> R<LaneStagePolicy>;
    /// Set `policy` on every lane store of the load (idempotent; refused
    /// without changes when the stores cannot take it).
    fn set_policy(&mut self, policy: LaneStagePolicy) -> R<()>;
}

/// Single/EP lanes: the bundle's lane API on the primary GPU (under EP the
/// calls replicate across ranks inside the arch crate).
struct LocalLanes<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    vocab: usize,
}

impl LaneBackend for LocalLanes<'_> {
    fn stage(&mut self, max_lanes: usize, row_budget: usize) -> R<()> {
        self.gpu.bind_thread().map_err(e)?;
        let before = self
            .gpu
            .device_mem_info()
            .map(|(free, _)| free)
            .unwrap_or(0);
        self.bundle
            .stage_lanes(self.gpu, max_lanes, row_budget, None)
            .map_err(e)?;
        let after = self
            .gpu
            .device_mem_info()
            .map(|(free, _)| free)
            .unwrap_or(0);
        eprintln!(
            "[{TOOL}] lane stage (AR) consumed {} MiB on rank 0 ({} MiB free after)",
            before.saturating_sub(after) >> 20,
            after >> 20
        );
        Ok(())
    }

    fn stage_policy(&self) -> R<LaneStagePolicy> {
        self.bundle
            .lanes()
            .map(|store| store.stage_policy())
            .ok_or_else(|| "lane store is not staged".to_string())
    }

    fn set_policy(&mut self, policy: LaneStagePolicy) -> R<()> {
        self.bundle.set_lane_stage_policy(None, policy).map_err(e)
    }

    fn lane_capacity(&self) -> R<usize> {
        self.bundle
            .lanes()
            .map(|store| store.max_lanes())
            .ok_or_else(|| "lane store is not staged".to_string())
    }

    fn admit(&mut self, epoch: RequestEpoch, prompt: Vec<u32>) -> R<usize> {
        self.bundle.admit_lane(self.gpu, epoch, prompt).map_err(e)
    }

    fn release(&mut self, epoch: &RequestEpoch) -> R<()> {
        self.bundle.release_lane(epoch).map_err(e)
    }

    fn prefill_tile_len(&mut self, epoch: &RequestEpoch) -> R<usize> {
        self.bundle
            .lane_executor()
            .map_err(e)?
            .exact_prefill_chunk_len(epoch)
    }

    fn step(&mut self, plan: &BatchStepPlan, read_logits: bool) -> R<StepResult> {
        let Self { bundle, gpu, vocab } = self;
        let mut ex = bundle.lane_executor().map_err(e)?;
        ex.provision_step(gpu, plan)
            .map_err(|m| format!("provision_step failed: {m}"))?;
        let out = match ex.forward_step(gpu, plan) {
            Ok(out) => out,
            Err(m) => {
                ex.abort_step(plan);
                return Err(format!("forward_step failed: {m}"));
            }
        };
        let picks = out.target_picks.clone();
        drop(ex);
        let logits = if read_logits {
            let read = (|| -> R<Vec<(RequestEpoch, Vec<u8>)>> {
                gpu.hip.device_synchronize().map_err(e)?;
                let store = bundle.lanes().ok_or("lane store vanished")?;
                let tensor = store
                    .logits()
                    .ok_or("lane store has no step logits (not an output store)")?;
                logits_rows(plan, &picks, store.prefill_logits_row())
                    .into_iter()
                    .map(|(epoch, row)| Ok((epoch, read_view(gpu, tensor, row * *vocab, *vocab)?)))
                    .collect()
            })();
            match read {
                Ok(rows) => rows,
                Err(m) => {
                    if let Ok(mut ex) = bundle.lane_executor() {
                        ex.abort_step(plan);
                    }
                    return Err(format!("logits read failed: {m}"));
                }
            }
        } else {
            Vec::new()
        };
        let mut ex = bundle.lane_executor().map_err(e)?;
        match ex.commit_step(gpu, plan, out) {
            Ok(advances) => Ok(StepResult { advances, logits }),
            Err(m) => {
                ex.abort_step(plan);
                Err(format!("commit_step failed: {m}"))
            }
        }
    }
}

/// PP lanes: the root (output band) bundle plus the caller's `Gpus`. Staging,
/// admission and release replicate over every band inside the arch crate; the
/// root store's step logits live on the output owner's device.
struct PpLanes<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpus: &'a mut Gpus,
    vocab: usize,
}

impl LaneBackend for PpLanes<'_> {
    fn stage(&mut self, max_lanes: usize, row_budget: usize) -> R<()> {
        self.bundle
            .pp_stage_lanes(self.gpus, max_lanes, row_budget, None)
            .map_err(e)
    }

    fn stage_policy(&self) -> R<LaneStagePolicy> {
        self.bundle
            .lanes()
            .map(|store| store.stage_policy())
            .ok_or_else(|| "lane store is not staged".to_string())
    }

    fn set_policy(&mut self, policy: LaneStagePolicy) -> R<()> {
        self.bundle
            .set_lane_stage_policy(Some(&mut *self.gpus), policy)
            .map_err(e)
    }

    fn lane_capacity(&self) -> R<usize> {
        self.bundle
            .lanes()
            .map(|store| store.max_lanes())
            .ok_or_else(|| "lane store is not staged".to_string())
    }

    fn admit(&mut self, epoch: RequestEpoch, prompt: Vec<u32>) -> R<usize> {
        self.bundle
            .pp_admit_lane(self.gpus, epoch, prompt)
            .map_err(e)
    }

    fn release(&mut self, epoch: &RequestEpoch) -> R<()> {
        self.bundle.pp_release_lane(self.gpus, epoch).map_err(e)
    }

    fn prefill_tile_len(&mut self, epoch: &RequestEpoch) -> R<usize> {
        self.bundle
            .pp_lane_executor(self.gpus)
            .map_err(e)?
            .exact_prefill_chunk_len(epoch)
    }

    fn step(&mut self, plan: &BatchStepPlan, read_logits: bool) -> R<StepResult> {
        let Self {
            bundle,
            gpus,
            vocab,
        } = self;
        let mut ex = bundle.pp_lane_executor(gpus).map_err(e)?;
        ex.provision_step(plan)
            .map_err(|m| format!("provision_step failed: {m}"))?;
        let out = match ex.forward_step(plan) {
            Ok(out) => out,
            Err(m) => {
                ex.abort_step(plan);
                return Err(format!("forward_step failed: {m}"));
            }
        };
        let picks = out.target_picks.clone();
        drop(ex);
        let logits = if read_logits {
            let read = (|| -> R<Vec<(RequestEpoch, Vec<u8>)>> {
                let owner = gpus
                    .mesh_output_owner()
                    .ok_or("PP mesh has no output owner")?;
                let gpu = gpus
                    .devices
                    .get_mut(owner)
                    .ok_or("PP output owner is not a mesh device")?;
                gpu.bind_thread().map_err(e)?;
                gpu.hip.device_synchronize().map_err(e)?;
                let store = bundle.lanes().ok_or("lane store vanished")?;
                let tensor = store
                    .logits()
                    .ok_or("the root lane store has no step logits")?;
                logits_rows(plan, &picks, store.prefill_logits_row())
                    .into_iter()
                    .map(|(epoch, row)| Ok((epoch, read_view(gpu, tensor, row * *vocab, *vocab)?)))
                    .collect()
            })();
            match read {
                Ok(rows) => rows,
                Err(m) => {
                    if let Ok(mut ex) = bundle.pp_lane_executor(gpus) {
                        ex.abort_step(plan);
                    }
                    return Err(format!("logits read failed: {m}"));
                }
            }
        } else {
            Vec::new()
        };
        let mut ex = bundle.pp_lane_executor(gpus).map_err(e)?;
        match ex.commit_step(plan, out) {
            Ok(advances) => Ok(StepResult { advances, logits }),
            Err(m) => {
                ex.abort_step(plan);
                Err(format!("commit_step failed: {m}"))
            }
        }
    }
}

// ----------------------------------------------------------------- driver

struct DriveCfg<'a> {
    prompts: &'a [&'a [u32]],
    tokens: usize,
    max_lanes: usize,
    row_budget: usize,
    want_logits: bool,
}

struct LaneRun {
    epoch: RequestEpoch,
    slot: usize,
    progress: Progress,
    /// Prompt-completing pick, then every AR pick.
    committed: Vec<u32>,
    /// SHA-256 of the prompt-completing row, then of every AR row (`--logits`).
    logits_sha: Vec<String>,
    /// The raw f32 logits rows themselves (`--logits`), for the per-row KL check.
    logits_raw: Vec<Vec<u8>>,
    /// AR rows this lane ran in decode-only steps.
    decode_only_rows: usize,
    admitted: bool,
}

/// The steps of a K run that carried at least one prefill tile: how many, the prompt rows their
/// tiles forwarded, and their summed wall seconds (an AR row of an earlier lane may share the
/// step; a K-lane run with equal prompts has one such step). `rows` over `wall` is the lane
/// prefill's cost per prompt row, the number a serve step's `[fn-lane-step] pf_rows`/`fwd` pair
/// reports.
#[derive(Clone, Debug, Default, PartialEq)]
struct PrefillTiming {
    steps: usize,
    rows: usize,
    wall: f64,
    /// Host time spent admitting finished prompts into the mesh decode (MTP arm), seconds;
    /// `wall + admit_wall` is the time to every first token of the burst.
    admit_wall: f64,
    /// One record per prefill step (a pass): how the burst was split across steps.
    passes: Vec<PrefillPass>,
}

/// One step that carried prefill tiles.
#[derive(Clone, Debug, Default, PartialEq)]
struct PrefillPass {
    /// Prefill tiles (lanes) in the pass.
    tiles: usize,
    /// Prompt rows those tiles forwarded.
    rows: usize,
    /// AR rows of earlier-finished lanes that shared the step.
    ar_rows: usize,
    wall: f64,
}

impl PrefillTiming {
    fn record(&mut self, pass: PrefillPass) {
        self.steps += 1;
        self.rows += pass.rows;
        self.wall += pass.wall;
        self.passes.push(pass);
    }

    fn json(&self) -> Value {
        json!({
            "steps": self.steps,
            "rows": self.rows,
            "wall_s": self.wall,
            "ms_per_row": (self.rows > 0).then(|| self.wall * 1e3 / self.rows as f64),
            "admit_wall_s": self.admit_wall,
            "burst_wall_s": self.wall + self.admit_wall,
            "passes": self.passes.iter().map(|p| json!({
                "tiles": p.tiles,
                "rows": p.rows,
                "ar_rows": p.ar_rows,
                "wall_s": p.wall,
                "ms_per_row": (p.rows > 0).then(|| p.wall * 1e3 / p.rows as f64),
            })).collect::<Vec<_>>(),
        })
    }
}

#[derive(Debug)]
struct KRun {
    k: usize,
    /// Per lane: its first `tokens` committed ids.
    ids: Vec<Vec<u32>>,
    /// Per lane: logits SHA-256 rows (`tokens + 1` with `--logits`, else empty).
    logits_sha: Vec<Vec<String>>,
    /// Per lane: the raw logits rows (same rows as `logits_sha`).
    logits_raw: Vec<Vec<Vec<u8>>>,
    steps: usize,
    decode_steps: usize,
    decode_tokens: usize,
    /// Wall seconds of every step / of the decode-only steps.
    steps_wall: f64,
    decode_wall: f64,
    prefill: PrefillTiming,
    lane_decode_rows: Vec<usize>,
}

impl KRun {
    fn aggregate_tok_s(&self) -> Option<f64> {
        (self.decode_wall > 0.0 && self.decode_tokens > 0)
            .then(|| self.decode_tokens as f64 / self.decode_wall)
    }

    fn per_lane_tok_s(&self) -> Vec<Option<f64>> {
        self.lane_decode_rows
            .iter()
            .map(|&rows| (self.decode_wall > 0.0).then(|| rows as f64 / self.decode_wall))
            .collect()
    }
}

/// Run prompts `0..k` as `k` concurrent lanes with fresh epochs until each has
/// `cfg.tokens` ids, releasing every lane (also on error).
fn drive_k(
    backend: &mut dyn LaneBackend,
    cfg: &DriveCfg<'_>,
    k: usize,
    next_tag: &mut u64,
) -> R<KRun> {
    let mut lanes: Vec<LaneRun> = Vec::with_capacity(k);
    let result = drive_inner(backend, cfg, k, next_tag, &mut lanes);
    let mut release_error = None;
    for lane in lanes.iter_mut().filter(|lane| lane.admitted) {
        lane.admitted = false;
        if let Err(x) = backend.release(&lane.epoch) {
            release_error.get_or_insert(format!("release lane {:?}: {x}", lane.epoch));
        }
    }
    match (result, release_error) {
        (Err(x), _) => Err(x),
        (Ok(_), Some(x)) => Err(x),
        (Ok(run), None) => Ok(run),
    }
}

fn drive_inner(
    backend: &mut dyn LaneBackend,
    cfg: &DriveCfg<'_>,
    k: usize,
    next_tag: &mut u64,
    lanes: &mut Vec<LaneRun>,
) -> R<KRun> {
    if k > cfg.max_lanes || k > cfg.prompts.len() {
        return Err(format!(
            "K={k} exceeds the staged lanes ({}) or the prompts ({})",
            cfg.max_lanes,
            cfg.prompts.len()
        ));
    }
    for prompt in &cfg.prompts[..k] {
        *next_tag += 1;
        let epoch = RequestEpoch {
            request_tag: *next_tag,
            owner_generation: 1,
        };
        let slot = backend.admit(epoch, prompt.to_vec())?;
        lanes.push(LaneRun {
            epoch,
            slot,
            progress: Progress {
                prompt_len: prompt.len(),
                prefilled: 0,
                ar_steps: 0,
            },
            committed: Vec::new(),
            logits_sha: Vec::new(),
            logits_raw: Vec::new(),
            decode_only_rows: 0,
            admitted: true,
        });
    }

    let (mut steps, mut decode_steps, mut decode_tokens) = (0usize, 0usize, 0usize);
    let (mut steps_wall, mut decode_wall) = (0f64, 0f64);
    let mut prefill_timing = PrefillTiming::default();
    loop {
        let progress: Vec<Progress> = lanes.iter().map(|lane| lane.progress).collect();
        let (prefill, ar) = schedule(&progress, cfg.tokens, cfg.row_budget);
        if prefill.is_empty() && ar.is_empty() {
            break;
        }
        let mut specs = Vec::with_capacity(prefill.len() + ar.len());
        for &i in &prefill {
            let lane = &lanes[i];
            let len = backend.prefill_tile_len(&lane.epoch)?;
            let p = lane.progress;
            if len == 0 || p.prefilled + len > p.prompt_len {
                return Err(format!(
                    "lane {i}: prefill tile of {len} rows at {}/{}",
                    p.prefilled, p.prompt_len
                ));
            }
            specs.push(RowSpec {
                slot: lane.slot,
                epoch: lane.epoch,
                tokens: cfg.prompts[i][p.prefilled..p.prefilled + len].to_vec(),
                start: p.prefilled,
                kind: RequestStepKind::Prefill,
            });
        }
        for &i in &ar {
            let lane = &lanes[i];
            let seed = *lane
                .committed
                .last()
                .ok_or_else(|| format!("lane {i} decodes without a pending seed"))?;
            specs.push(RowSpec {
                slot: lane.slot,
                epoch: lane.epoch,
                tokens: vec![seed],
                start: lane.progress.prompt_len + lane.progress.ar_steps,
                kind: RequestStepKind::Ar,
            });
        }
        let plan = build_plan(cfg.max_lanes, specs);
        let decode_only = prefill.is_empty();

        let started = Instant::now();
        let result = backend.step(&plan, cfg.want_logits)?;
        let wall = started.elapsed().as_secs_f64();
        steps += 1;
        steps_wall += wall;
        if !decode_only {
            prefill_timing.record(PrefillPass {
                tiles: prefill.len(),
                rows: plan
                    .requests
                    .iter()
                    .filter(|r| r.kind == RequestStepKind::Prefill)
                    .map(|r| r.rows.len)
                    .sum::<usize>(),
                ar_rows: ar.len(),
                wall,
            });
        }
        if decode_only {
            decode_steps += 1;
            decode_wall += wall;
            decode_tokens += ar.len();
        }

        if result.advances.len() != plan.requests.len() {
            return Err(format!(
                "{} advances for {} plan requests",
                result.advances.len(),
                plan.requests.len()
            ));
        }
        for (request, advance) in plan.requests.iter().zip(&result.advances) {
            if advance.epoch != request.epoch {
                return Err("advance order differs from the plan".into());
            }
            let idx = lanes
                .iter()
                .position(|lane| lane.epoch == request.epoch)
                .ok_or("plan epoch is not one of this run's lanes")?;
            let row_sha = || -> R<Option<String>> {
                if !cfg.want_logits {
                    return Ok(None);
                }
                result
                    .logits
                    .iter()
                    .find(|(epoch, _)| *epoch == request.epoch)
                    .map(|(_, bytes)| Some(sha256_hex(bytes)))
                    .ok_or_else(|| format!("lane {idx}: the step returned no logits row"))
            };
            let row_raw = || -> Option<Vec<u8>> {
                if !cfg.want_logits {
                    return None;
                }
                result
                    .logits
                    .iter()
                    .find(|(epoch, _)| *epoch == request.epoch)
                    .map(|(_, bytes)| bytes.to_vec())
            };
            let lane = &mut lanes[idx];
            match request.kind {
                RequestStepKind::Prefill => {
                    lane.progress.prefilled += request.rows.len;
                    if lane.progress.prefilled < lane.progress.prompt_len {
                        if !advance.committed_ids.is_empty() {
                            return Err(format!(
                                "lane {idx}: a non-final prefill tile committed {:?}",
                                advance.committed_ids
                            ));
                        }
                        continue;
                    }
                    let [id] = advance.committed_ids.as_slice() else {
                        return Err(format!(
                            "lane {idx}: prompt-completing tile committed {:?}, expected one id",
                            advance.committed_ids
                        ));
                    };
                    lane.committed.push(*id);
                    lane.logits_sha.extend(row_sha()?);
                    lane.logits_raw.extend(row_raw());
                }
                RequestStepKind::Ar => {
                    lane.progress.ar_steps += 1;
                    let [id] = advance.committed_ids.as_slice() else {
                        return Err(format!(
                            "lane {idx}: AR row committed {:?}, expected one id",
                            advance.committed_ids
                        ));
                    };
                    let expected = lane.progress.prompt_len + lane.progress.ar_steps;
                    if advance.committed_position != expected {
                        return Err(format!(
                            "lane {idx} AR step {}: committed position {} expected {expected}",
                            lane.progress.ar_steps, advance.committed_position
                        ));
                    }
                    lane.committed.push(*id);
                    lane.logits_sha.extend(row_sha()?);
                    lane.logits_raw.extend(row_raw());
                    if decode_only {
                        lane.decode_only_rows += 1;
                    }
                    if lane.progress.ar_steps == cfg.tokens {
                        lane.admitted = false;
                        backend
                            .release(&lane.epoch)
                            .map_err(|x| format!("release lane {idx}: {x}"))?;
                    }
                }
                other => return Err(format!("lane {idx}: unexpected plan row kind {other:?}")),
            }
        }
    }

    for (i, lane) in lanes.iter().enumerate() {
        let p = lane.progress;
        if p.prefilled != p.prompt_len
            || p.ar_steps != cfg.tokens
            || lane.committed.len() != cfg.tokens + 1
        {
            return Err(format!(
                "lane {i} ended at prefill {}/{}, {}/{} AR steps, {} committed ids",
                p.prefilled,
                p.prompt_len,
                p.ar_steps,
                cfg.tokens,
                lane.committed.len()
            ));
        }
    }
    Ok(KRun {
        k,
        ids: lanes
            .iter()
            .map(|lane| lane.committed[..cfg.tokens].to_vec())
            .collect(),
        logits_sha: lanes.iter().map(|lane| lane.logits_sha.clone()).collect(),
        logits_raw: lanes.iter().map(|lane| lane.logits_raw.clone()).collect(),
        steps,
        decode_steps,
        decode_tokens,
        steps_wall,
        decode_wall,
        prefill: prefill_timing,
        lane_decode_rows: lanes.iter().map(|lane| lane.decode_only_rows).collect(),
    })
}

// -------------------------------------------------------------- reference

struct Reference {
    ids: Vec<u32>,
    /// SHA-256 of the logits row after the prefill, then after every forward
    /// (`tokens + 1` rows with `--logits`, else empty).
    logits_sha: Vec<String>,
    /// The raw f32 logits rows (same rows), for the per-row KL check.
    logits_raw: Vec<Vec<u8>>,
    /// Wall milliseconds of the cold prefill (final-logits read included): the singleton the
    /// lane prefill of the same prompt is compared with.
    prefill_ms: f64,
}

/// The serialized singleton run of one topology (`fn_ep_oracle`'s AR loop).
trait Singleton {
    /// Cold prefill of `prompt` (resets the state), leaving the final logits.
    fn prefill(&mut self, prompt: &[u32]) -> R<()>;
    /// One greedy forward of the pending token; returns its pick.
    fn forward(&mut self) -> R<u32>;
    /// The logits row the last prefill/forward left, as raw bytes.
    fn logits_bytes(&mut self) -> R<Vec<u8>>;
}

/// Single/EP singleton on the primary GPU.
struct LocalSingle<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    logits: &'a GpuTensor,
}

impl Singleton for LocalSingle<'_> {
    fn prefill(&mut self, prompt: &[u32]) -> R<()> {
        self.bundle.reset(self.gpu).map_err(e)?;
        self.bundle
            .prefill_final(self.gpu, prompt, 0, self.logits)
            .map_err(e)
    }

    fn forward(&mut self) -> R<u32> {
        self.bundle
            .forward_token_or_argmax(self.gpu, None, self.logits)
            .map_err(e)
    }

    fn logits_bytes(&mut self) -> R<Vec<u8>> {
        self.gpu.bind_thread().map_err(e)?;
        tensor_bytes(self.gpu, self.logits)
    }
}

/// PP singleton: the mesh entries with the model's `Gpus`; `logits` and `gpu`
/// are the daemon's logits device, which is the output owner.
struct PpSingle<'a> {
    bundle: &'a mut Qwen4Bundle,
    gpus: &'a mut Gpus,
    gpu: &'a mut Gpu,
    logits: &'a GpuTensor,
}

impl Singleton for PpSingle<'_> {
    fn prefill(&mut self, prompt: &[u32]) -> R<()> {
        // Cold PP prefill resets every band and uses the canonical tile schedule.
        self.bundle
            .pp_prefill_final(self.gpus, prompt, 0, self.logits)
            .map_err(e)
    }

    fn forward(&mut self) -> R<u32> {
        self.bundle
            .pp_forward_token_or_argmax(self.gpus, None, self.logits)
            .map_err(e)
    }

    fn logits_bytes(&mut self) -> R<Vec<u8>> {
        self.gpu.bind_thread().map_err(e)?;
        tensor_bytes(self.gpu, self.logits)
    }
}

/// The serialized singleton greedy run of `prompt`.
fn reference_run(
    single: &mut dyn Singleton,
    prompt: &[u32],
    tokens: usize,
    want_logits: bool,
) -> R<Reference> {
    // The singleton's cold prefill of this prompt, timed (the final-logits read after it drains
    // the device), to set beside the lane prefill tile of the same rows (`prefill` in
    // report.json, `[fn-lane-tile]` with `HIPFIRE_CB_PHASES=1`): one tile IS this program.
    let clock = Instant::now();
    single.prefill(prompt)?;
    single.logits_bytes()?;
    let prefill_ms = clock.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[oracle-ref-prefill] rows={} ms={prefill_ms:.1}",
        prompt.len(),
    );
    let mut reference = Reference {
        ids: Vec::with_capacity(tokens),
        logits_sha: Vec::new(),
        logits_raw: Vec::new(),
        prefill_ms,
    };
    if want_logits {
        let bytes = single.logits_bytes()?;
        reference.logits_sha.push(sha256_hex(&bytes));
        reference.logits_raw.push(bytes);
    }
    for step in 1..=tokens {
        let id = single
            .forward()
            .map_err(|x| format!("reference step {step}: {x}"))?;
        reference.ids.push(id);
        if want_logits {
            let bytes = single.logits_bytes()?;
            reference.logits_sha.push(sha256_hex(&bytes));
            reference.logits_raw.push(bytes);
        }
    }
    Ok(reference)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PolicyStatus {
    Pass,
    /// Ran to completion with a mismatch.
    Fail,
    /// A refusal or execution error stopped this policy's matrix.
    Error(String),
    /// Not run: the lane store was unusable after an earlier error.
    Skipped(String),
}

impl PolicyStatus {
    fn label(&self) -> &'static str {
        match self {
            PolicyStatus::Pass => "PASS",
            PolicyStatus::Fail => "FAIL",
            PolicyStatus::Error(_) => "ERROR",
            PolicyStatus::Skipped(_) => "SKIPPED",
        }
    }
}

/// One policy's whole `--ks` matrix result.
#[derive(Debug)]
struct PolicyOutcome {
    spec: String,
    status: PolicyStatus,
    /// The `report.json` `policies[]` record.
    record: Value,
    /// Failing `K=.. prompt ..` labels.
    failures: Vec<String>,
}

/// The reference ids / logits SHAs of the first `tokens` tokens (the sweep
/// may run fewer tokens than the reference).
fn reference_prefix(
    reference: &Reference,
    tokens: usize,
    want_logits: bool,
) -> (&[u32], &[String]) {
    (
        &reference.ids[..tokens],
        if want_logits {
            &reference.logits_sha[..tokens + 1]
        } else {
            &[]
        },
    )
}

/// Set `policy`, then run every K of `args.ks` as concurrent lanes and
/// compare with `references`. Never returns an error: a refusal or
/// execution error is the outcome's status (`ids_lines` keeps what ran).
#[allow(clippy::too_many_arguments)]
fn run_policy(
    backend: &mut dyn LaneBackend,
    args: &Args,
    spec: &str,
    policy: LaneStagePolicy,
    tokens: usize,
    prompts: &[&[u32]],
    references: &[Reference],
    max_lanes: usize,
    row_budget: usize,
    next_tag: &mut u64,
    ids_lines: &mut Vec<Value>,
) -> PolicyOutcome {
    let mut runs = Vec::new();
    let mut per_k = Vec::new();
    let mut failures = Vec::new();
    let mut status = PolicyStatus::Pass;
    match backend.set_policy(policy) {
        Err(x) => status = PolicyStatus::Error(format!("set lane stage policy: {x}")),
        Ok(()) => {
            for set in run_sets(args) {
                let label = set.label();
                let k = set.members.len();
                eprintln!(
                    "[{TOOL}] policy {spec}: {label}: {k} concurrent lanes ({tokens} tokens)"
                );
                let selected: Vec<&[u32]> = set.members.iter().map(|&i| prompts[i]).collect();
                let cfg = DriveCfg {
                    prompts: &selected,
                    tokens,
                    max_lanes,
                    row_budget,
                    want_logits: args.logits,
                };
                let run = match drive_k(backend, &cfg, k, next_tag) {
                    Ok(run) => run,
                    Err(x) => {
                        status = PolicyStatus::Error(format!("{label}: {x}"));
                        break;
                    }
                };
                let mut compared = Vec::with_capacity(k);
                for (lane, &index) in set.members.iter().enumerate() {
                    ids_lines.push(json!({
                        "kind": "lane", "policy": spec, "k": k, "prompt": index, "lane": lane,
                        "set": label, "ids": run.ids[lane],
                    }));
                    let (reference_ids, reference_logits) =
                        reference_prefix(&references[index], tokens, args.logits);
                    let comparison = compare(
                        reference_ids,
                        &run.ids[lane],
                        args.logits
                            .then(|| (reference_logits, run.logits_sha[lane].as_slice())),
                    );
                    if args.logits {
                        if let Some((rows, max, mean)) = lane_kl(
                            &references[index].logits_raw[..tokens + 1],
                            &run.logits_raw[lane],
                            comparison.first_mismatch,
                        ) {
                            eprintln!(
                                "[oracle-kl] policy={spec} set={label} lane={lane} prompt={index} rows={rows} max_kl={max:.3e} mean_kl={mean:.3e}"
                            );
                        }
                    }
                    if !comparison.equal {
                        if status == PolicyStatus::Pass {
                            status = PolicyStatus::Fail;
                        }
                        failures.push(format!("{label} prompt {index}"));
                    }
                    compared.push(compared_json(
                        index,
                        prompts[index].len(),
                        lane,
                        &comparison,
                    ));
                }
                // One stderr line per K run: lane == singleton, with the first divergence of
                // the ids / logits rows of every prompt that differs (the report carries them).
                let first = |key: &str| {
                    compared
                        .iter()
                        .filter_map(|c| Some((c["prompt"].as_u64()?, c.get(key)?.as_u64()?)))
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "[oracle-lane-vs-singleton] policy={spec} k={} prompts_equal={}/{} ids_first_mismatch(prompt,index)={:?} logits_first_mismatch(prompt,row)={:?}",
                    run.k,
                    compared.iter().filter(|c| c["equal"] == json!(true)).count(),
                    compared.len(),
                    first("first_mismatch"),
                    first("first_logits_mismatch"),
                );
                let equal = compared.iter().all(|c| c["equal"] == json!(true));
                let extras = burst_extras(&set, &run, prompts, references);
                eprintln!("{}", burst_line(spec, &set, &run, &compared, &extras));
                per_k.push(json!({
                    "k": run.k,
                    "equal": equal,
                    "decode_wall_s": run.decode_wall,
                    "aggregate_tok_s": run.aggregate_tok_s(),
                }));
                runs.push(with_extras(
                    json!({
                        "k": run.k,
                        "equal": equal,
                        "prompts": compared,
                        "steps": run.steps,
                        "decode_steps": run.decode_steps,
                        "decode_tokens": run.decode_tokens,
                        "decode_wall_s": run.decode_wall,
                        "steps_wall_s": run.steps_wall,
                        "prefill": run.prefill.json(),
                        "aggregate_tok_s": run.aggregate_tok_s(),
                        "per_lane_tok_s": run.per_lane_tok_s(),
                        "timing_includes_logits_read": args.logits,
                    }),
                    extras,
                ));
            }
        }
    }
    let mut record = json!({
        "spec": spec,
        "describe": policy.describe(),
        "tokens": tokens,
        "status": status.label(),
        "all_equal": status == PolicyStatus::Pass,
        "failures": failures,
        "per_k": per_k,
        "runs": runs,
    });
    if let PolicyStatus::Error(message) = &status {
        record["error"] = json!(message);
    }
    PolicyOutcome {
        spec: spec.to_string(),
        status,
        record,
        failures,
    }
}

/// Run `args.policies` in order over one staged `backend`: every policy is
/// set on all stores and the whole `--ks` matrix is compared with the same
/// `references`. A failing policy never stops the sweep. After an execution
/// error the next policy runs only if the lane store still accepts a policy
/// (re-setting the failed one is the probe: it needs ready, idle, staged
/// stores); otherwise every remaining policy is `Skipped`.
fn drive_policies(
    backend: &mut dyn LaneBackend,
    args: &Args,
    prompts: &[&[u32]],
    references: &[Reference],
    row_budget: usize,
    ids_lines: &mut Vec<Value>,
) -> R<Vec<PolicyOutcome>> {
    let max_lanes = backend.lane_capacity()?;
    let construction = backend.stage_policy()?;
    let mut next_tag = 0u64;
    let mut outcomes: Vec<PolicyOutcome> = Vec::with_capacity(args.policies.len());
    let mut stopped: Option<String> = None;
    for (index, entry) in args.policies.iter().enumerate() {
        if let Some(reason) = &stopped {
            outcomes.push(PolicyOutcome {
                spec: entry.spec.clone(),
                status: PolicyStatus::Skipped(reason.clone()),
                record: json!({
                    "spec": entry.spec,
                    "describe": entry.policy.map(|policy| policy.describe()),
                    "status": "SKIPPED",
                    "all_equal": false,
                    "skipped": reason,
                    "runs": [],
                    "per_k": [],
                }),
                failures: Vec::new(),
            });
            continue;
        }
        let policy = entry.policy.unwrap_or(construction);
        let tokens = if index == 0 {
            args.tokens
        } else {
            args.tokens_sweep.unwrap_or(args.tokens)
        };
        eprintln!(
            "[{TOOL}] policy {}/{}: {} ({})",
            index + 1,
            args.policies.len(),
            entry.spec,
            policy.describe()
        );
        let outcome = run_policy(
            backend,
            args,
            &entry.spec,
            policy,
            tokens,
            prompts,
            references,
            max_lanes,
            row_budget,
            &mut next_tag,
            ids_lines,
        );
        if let PolicyStatus::Error(message) = &outcome.status {
            eprintln!("[{TOOL}] policy {} errored: {message}", entry.spec);
            if let Err(x) = backend.set_policy(policy) {
                stopped = Some(format!(
                    "lane store unusable after policy {:?} errored: {x}",
                    entry.spec
                ));
            }
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

/// Stage `backend`, then sweep the policies ([`drive_policies`]).
fn drive_all(
    backend: &mut dyn LaneBackend,
    args: &Args,
    prompts: &[&[u32]],
    references: &[Reference],
    max_lanes_requested: usize,
    row_budget: usize,
    ids_lines: &mut Vec<Value>,
) -> R<Vec<PolicyOutcome>> {
    backend.stage(max_lanes_requested, row_budget)?;
    drive_policies(backend, args, prompts, references, row_budget, ids_lines)
}

/// `[i] spec=LABEL` per policy (a failing one lists its K/prompt labels).
fn sweep_summary(outcomes: &[PolicyOutcome]) -> String {
    outcomes
        .iter()
        .enumerate()
        .map(|(index, outcome)| match &outcome.status {
            PolicyStatus::Fail => format!(
                "[{index}] {}=FAIL at {}",
                outcome.spec,
                outcome.failures.join(", ")
            ),
            status => format!("[{index}] {}={}", outcome.spec, status.label()),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// `Pass` only when every policy ran and was equal.
fn sweep_verdict(outcomes: &[PolicyOutcome]) -> Verdict {
    if outcomes
        .iter()
        .all(|outcome| outcome.status == PolicyStatus::Pass)
    {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

// ------------------------------------------------------------ MTP EP arm

/// One lane of the native-MTP EP run: prefill through the ordinary lane
/// executor (identical on every rank), decode through the mesh lane protocol.
struct MtpLane {
    epoch: RequestEpoch,
    slot: usize,
    progress: Progress,
    /// The prompt-completing pick, then every emitted token.
    committed: Vec<u32>,
    admitted: bool,
    in_mesh: bool,
    windows: u64,
    offered: u64,
    accepted: u64,
}

fn lab_routes(routes: RoutesArg) -> LabRoutes {
    match routes {
        RoutesArg::Floor => LabRoutes::Floor,
        RoutesArg::Cycle => LabRoutes::Cycle,
        RoutesArg::Batched => LabRoutes::Batched,
    }
}

fn mesh_step_error(error: MeshStepError) -> String {
    match error {
        MeshStepError::Preflight(v) => format!("mesh preflight rejected: {v:?}"),
        MeshStepError::Aborted(record) => format!("mesh step aborted: {record}"),
        MeshStepError::Protocol(p) => format!("mesh protocol: {p}"),
    }
}

fn merged_json(m: &hipfire_arch_qwen4::mesh_lane_protocol::MergedCostEvidence) -> Value {
    json!({
        "step_seq": m.key.step_seq,
        "ranks": m.ranks,
        "target_exchanges": m.counts.target_exchanges,
        "gather_bytes": m.counts.gather_bytes,
        "head_draft_calls": m.counts.head_draft_calls,
        "head_append_rows": m.counts.head_append_rows,
        "row_groups": m.counts.row_groups,
        "drafts_generated": m.counts.drafts_generated,
        "accepted": m.counts.accepted,
        "consumed_rows": m.counts.consumed_rows,
        "wall_us": m.wall_us,
        "critical_path_us": m.critical_path_us,
        "synchronized": m.synchronized,
        "priced": m.priced,
        "unpriced_reason": m.unpriced_reason,
    })
}

fn median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

struct MtpKRun {
    run: KRun,
    windows: Vec<Value>,
    rank_stats: Vec<Value>,
    summary: Value,
}

/// Run prompts `0..k` as `k` concurrent native-MTP lanes on the EP mesh until
/// each has `cfg.tokens` committed ids; every lane is released (also on
/// error).
#[allow(clippy::too_many_arguments)]
fn drive_mtp_k(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    cfg: &DriveCfg<'_>,
    max_seq: usize,
    depth: usize,
    routes: RoutesArg,
    k: usize,
    next_tag: &mut u64,
    vocab: usize,
) -> R<MtpKRun> {
    let mut lanes: Vec<MtpLane> = Vec::with_capacity(k);
    let mut windows = Vec::new();
    let result = drive_mtp_inner(
        bundle,
        gpu,
        mesh,
        cfg,
        max_seq,
        depth,
        routes,
        k,
        next_tag,
        vocab,
        &mut lanes,
        &mut windows,
    );
    let mut release_error = None;
    for lane in lanes.iter_mut() {
        if lane.in_mesh {
            lane.in_mesh = false;
            if let Err(x) = mesh.release_lane(bundle, gpu, lane.epoch) {
                release_error.get_or_insert(format!("mesh release {:?}: {x}", lane.epoch));
            }
        }
        if lane.admitted {
            lane.admitted = false;
            if let Err(x) = bundle.release_lane(&lane.epoch) {
                release_error.get_or_insert(format!("release {:?}: {x}", lane.epoch));
            }
        }
    }
    let rank_stats = mesh
        .rank_stats(bundle, gpu)
        .map(|all| {
            all.into_iter()
                .map(|s| {
                    json!({
                        "rank": s.rank, "steps": s.steps,
                        "native_windows": s.native_windows,
                        "draft_mismatches": s.draft_mismatches,
                        "pick_mismatches": s.pick_mismatches,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    match (result, release_error) {
        (Err(x), _) => Err(x),
        (Ok(_), Some(x)) => Err(x),
        (Ok((run, summary)), None) => Ok(MtpKRun {
            run,
            windows,
            rank_stats,
            summary,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_mtp_inner(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    cfg: &DriveCfg<'_>,
    max_seq: usize,
    depth: usize,
    routes: RoutesArg,
    k: usize,
    next_tag: &mut u64,
    vocab: usize,
    lanes: &mut Vec<MtpLane>,
    windows: &mut Vec<Value>,
) -> R<(KRun, Value)> {
    if k > cfg.max_lanes || k > cfg.prompts.len() {
        return Err(format!(
            "K={k} exceeds the staged lanes ({}) or the prompts ({})",
            cfg.max_lanes,
            cfg.prompts.len()
        ));
    }
    for prompt in &cfg.prompts[..k] {
        *next_tag += 1;
        let epoch = RequestEpoch {
            request_tag: *next_tag,
            owner_generation: 1,
        };
        let slot = bundle.admit_lane(gpu, epoch, prompt.to_vec()).map_err(e)?;
        lanes.push(MtpLane {
            epoch,
            slot,
            progress: Progress {
                prompt_len: prompt.len(),
                prefilled: 0,
                ar_steps: 0,
            },
            committed: Vec::new(),
            admitted: true,
            in_mesh: false,
            windows: 0,
            offered: 0,
            accepted: 0,
        });
    }
    let budgets: WindowBudgets = mesh.window_budgets(bundle, lab_routes(routes))?;
    let (mut steps, mut decode_steps, mut decode_tokens) = (0usize, 0usize, 0usize);
    let (mut steps_wall, mut decode_wall) = (0f64, 0f64);
    let mut prefill_timing = PrefillTiming::default();
    let (mut ev_wall, mut ev_crit, mut ev_ex, mut ev_head, mut ev_bytes, mut ev_priced) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::<bool>::new(),
    );
    let (mut total_windows, mut total_offered, mut total_accepted, mut total_emitted) =
        (0u64, 0u64, 0u64, 0u64);
    loop {
        // Prefill: one tile per Prefilling lane through the lane executor.
        let prefill: Vec<usize> = (0..lanes.len())
            .filter(|&i| lanes[i].progress.prefilled < lanes[i].progress.prompt_len)
            .collect();
        let mut prefill_ran = false;
        if !prefill.is_empty() {
            let mut specs = Vec::with_capacity(prefill.len());
            for &i in &prefill {
                let lane = &lanes[i];
                let len = bundle
                    .lane_executor()
                    .map_err(e)?
                    .exact_prefill_chunk_len(&lane.epoch)?;
                let p = lane.progress;
                if len == 0 || p.prefilled + len > p.prompt_len {
                    return Err(format!(
                        "lane {i}: prefill tile of {len} rows at {}/{}",
                        p.prefilled, p.prompt_len
                    ));
                }
                specs.push(RowSpec {
                    slot: lane.slot,
                    epoch: lane.epoch,
                    tokens: cfg.prompts[i][p.prefilled..p.prefilled + len].to_vec(),
                    start: p.prefilled,
                    kind: RequestStepKind::Prefill,
                });
            }
            let plan = build_plan(cfg.max_lanes, specs);
            let started = Instant::now();
            let result = LocalLanes {
                bundle: &mut *bundle,
                gpu: &mut *gpu,
                vocab,
            }
            .step(&plan, false)?;
            steps += 1;
            let wall = started.elapsed().as_secs_f64();
            steps_wall += wall;
            prefill_timing.record(PrefillPass {
                tiles: plan.requests.len(),
                rows: plan.total_rows(),
                ar_rows: 0,
                wall,
            });
            prefill_ran = true;
            mesh.note_prefill();
            for (request, advance) in plan.requests.iter().zip(&result.advances) {
                let idx = lanes
                    .iter()
                    .position(|lane| lane.epoch == request.epoch)
                    .ok_or("plan epoch is not one of this run's lanes")?;
                lanes[idx].progress.prefilled += request.rows.len;
                if lanes[idx].progress.prefilled < lanes[idx].progress.prompt_len {
                    continue;
                }
                let [id] = advance.committed_ids.as_slice() else {
                    return Err(format!(
                        "lane {idx}: prompt-completing tile committed {:?}",
                        advance.committed_ids
                    ));
                };
                lanes[idx].committed.push(*id);
                // No EOS handling: the AR reference loop ignores EOS, and the
                // oracle compares exact id streams.
                let admit_clock = Instant::now();
                mesh.admit_lane(
                    bundle,
                    gpu,
                    MeshLaneAdmission {
                        epoch: lanes[idx].epoch,
                        position: lanes[idx].progress.prompt_len,
                        seed: *id,
                        eos: None,
                        rng_seed: lanes[idx].epoch.request_tag,
                        prompt: cfg.prompts[idx].to_vec(),
                        spec_k: Some(depth),
                    },
                )?;
                prefill_timing.admit_wall += admit_clock.elapsed().as_secs_f64();
                lanes[idx].in_mesh = true;
            }
        }

        // Lanes with everything they need leave the mesh.
        for lane in lanes.iter_mut() {
            if lane.in_mesh && lane.committed.len() >= cfg.tokens {
                lane.in_mesh = false;
                mesh.release_lane(bundle, gpu, lane.epoch)?;
                lane.admitted = false;
                bundle.release_lane(&lane.epoch).map_err(e)?;
            }
        }

        // Decode: one mesh step over every lane that still needs tokens.
        let active: Vec<usize> = (0..lanes.len())
            .filter(|&i| lanes[i].in_mesh && lanes[i].committed.len() < cfg.tokens)
            .collect();
        if active.is_empty() {
            if prefill.is_empty() {
                break;
            }
            continue;
        }
        let mut requests = Vec::with_capacity(active.len());
        for &i in &active {
            let lane = &lanes[i];
            let (position, _) = mesh
                .root()
                .lane_position(lane.epoch)
                .ok_or_else(|| format!("lane {i}: the root lost its position"))?;
            requests.push(LaneRequest {
                epoch: lane.epoch,
                remaining: cfg.tokens - lane.committed.len(),
                context_left: max_seq.saturating_sub(position),
                forced: None,
                sampling: SamplingMode::Greedy,
            });
        }
        let started = Instant::now();
        let report = mesh
            .step(bundle, gpu, &requests, &budgets, &mut GreedyRule)
            .map_err(mesh_step_error)?;
        let wall = started.elapsed().as_secs_f64();
        steps += 1;
        steps_wall += wall;
        let mut emitted_step = 0usize;
        for out in &report.output.lanes {
            let idx = lanes
                .iter()
                .position(|lane| lane.epoch == out.key.request_epoch)
                .ok_or("mesh output epoch is not one of this run's lanes")?;
            let lane = &mut lanes[idx];
            lane.committed.extend_from_slice(&out.emitted);
            lane.windows += 1;
            lane.offered += out.source.offered as u64;
            lane.accepted += out.source.accepted as u64;
            emitted_step += out.emitted.len();
            total_windows += 1;
            total_offered += out.source.offered as u64;
            total_accepted += out.source.accepted as u64;
            total_emitted += out.emitted.len() as u64;
            windows.push(json!({
                "k": k, "step": report.key.step_seq, "lane": idx,
                "window_seq": out.key.window_seq,
                "source": format!("{:?}", out.source.source),
                "offered": out.source.offered,
                "accepted": out.source.accepted,
                "consumed_rows": out.consumed_rows,
                "emitted": out.emitted,
                "terminal": out.terminal,
            }));
        }
        if let Some(prior) = mesh.last_evidence() {
            ev_wall.push(prior.wall_us);
            ev_crit.push(prior.critical_path_us);
            ev_ex.push(prior.counts.target_exchanges as f64);
            ev_head.push(prior.counts.head_draft_calls as f64);
            ev_bytes.push(prior.counts.gather_bytes as f64);
            ev_priced.push(prior.priced);
            windows.push(json!({
                "k": k, "kind": "evidence-of-previous-step",
                "prior_priced": report.prior_priced,
                "evidence": merged_json(prior),
            }));
        }
        if !prefill_ran {
            decode_steps += 1;
            decode_wall += wall;
            decode_tokens += emitted_step;
        }
    }
    for (i, lane) in lanes.iter().enumerate() {
        if lane.committed.len() < cfg.tokens {
            return Err(format!(
                "lane {i} ended with {}/{} committed ids",
                lane.committed.len(),
                cfg.tokens
            ));
        }
    }
    let ids: Vec<Vec<u32>> = lanes
        .iter()
        .map(|lane| lane.committed[..cfg.tokens].to_vec())
        .collect();
    let summary = json!({
        "windows": total_windows,
        "drafts_offered": total_offered,
        "drafts_accepted": total_accepted,
        "tokens_emitted": total_emitted,
        // Accepted drafts per window (the wire `tau`) and emitted per window.
        "tau": if total_windows > 0 { total_accepted as f64 / total_windows as f64 } else { 0.0 },
        "yield": if total_windows > 0 { total_emitted as f64 / total_windows as f64 } else { 0.0 },
        // Merged rank evidence of the priced steps (median over steps).
        "evidence_steps": ev_wall.len(),
        "evidence_priced_steps": ev_priced.iter().filter(|p| **p).count(),
        "step_wall_us_median": median(&mut ev_wall.clone()),
        "step_critical_path_us_median": median(&mut ev_crit.clone()),
        "target_exchanges_median": median(&mut ev_ex.clone()),
        "head_draft_calls_median": median(&mut ev_head.clone()),
        "gather_bytes_median": median(&mut ev_bytes.clone()),
        "per_lane": lanes.iter().map(|l| json!({
            "windows": l.windows, "offered": l.offered, "accepted": l.accepted,
        })).collect::<Vec<_>>(),
    });
    Ok((
        KRun {
            k,
            ids,
            logits_sha: vec![Vec::new(); k],
            logits_raw: vec![Vec::new(); k],
            steps,
            decode_steps,
            decode_tokens,
            steps_wall,
            decode_wall,
            prefill: prefill_timing,
            lane_decode_rows: lanes
                .iter()
                .map(|l| l.committed.len().saturating_sub(1))
                .collect(),
        },
        summary,
    ))
}

/// The native-MTP EP arm: stage MTP lanes, then every K of `args.ks` as
/// concurrent lanes, ids compared with the serialized singleton AR reference.
/// One outcome (policy label `mtp:k=..,routes=..`).
#[allow(clippy::too_many_arguments)]
fn drive_mtp_ep(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    args: &Args,
    prompts: &[&[u32]],
    references: &[Reference],
    max_lanes: usize,
    row_budget: usize,
    vocab: usize,
    ids_lines: &mut Vec<Value>,
) -> R<Vec<PolicyOutcome>> {
    let depth = args.k.unwrap_or(DEFAULT_K);
    let routes = args.routes.unwrap_or(RoutesArg::Cycle);
    let label = format!(
        "mtp:k={depth},routes={}{}",
        routes.name(),
        if args.copy { ",copy" } else { "" }
    );
    if args.logits {
        return Err(
            "--logits is not available with --spec mtp (verify rows are not AR rows)".into(),
        );
    }
    gpu.bind_thread().map_err(e)?;
    let copy_cfg = if args.copy {
        let ngram = hipfire_arch_qwen4::mtp_spec::qwen4_ngram_mod_config(
            hipfire_config::ngram_mod_copy_triple(),
        )?;
        Some(hipfire_arch_qwen4::lane::LaneCopyConfig {
            ngram,
            head_lag_rows: 8,
        })
    } else {
        None
    };
    let before = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
    bundle
        .stage_lanes_with_copy(
            gpu,
            max_lanes,
            row_budget,
            Some(LaneSpecConfig {
                k: depth,
                end_of_turn: None,
                spec_lanes: None,
            }),
            copy_cfg,
        )
        .map_err(e)?;
    let after = gpu.device_mem_info().map(|(free, _)| free).unwrap_or(0);
    eprintln!(
        "[{TOOL}] lane stage (MTP k={depth}) consumed {} MiB on rank 0 ({} MiB free after)",
        before.saturating_sub(after) >> 20,
        after >> 20
    );
    let mut mesh = EpMeshLanes::new(bundle, 1, routes == RoutesArg::Floor)?;
    let mut next_tag = 0u64;
    let mut runs = Vec::new();
    let mut per_k = Vec::new();
    let mut failures = Vec::new();
    let mut status = PolicyStatus::Pass;
    for set in run_sets(args) {
        let label_set = set.label();
        let k = set.members.len();
        eprintln!(
            "[{TOOL}] {label}: {label_set}: {k} concurrent MTP lanes ({} tokens)",
            args.tokens
        );
        let selected: Vec<&[u32]> = set.members.iter().map(|&i| prompts[i]).collect();
        let cfg = DriveCfg {
            prompts: &selected,
            tokens: args.tokens,
            max_lanes,
            row_budget,
            want_logits: false,
        };
        let out = match drive_mtp_k(
            bundle,
            gpu,
            &mut mesh,
            &cfg,
            args.max_seq,
            depth,
            routes,
            k,
            &mut next_tag,
            vocab,
        ) {
            Ok(out) => out,
            Err(x) => {
                status = PolicyStatus::Error(format!("{label_set}: {x}"));
                break;
            }
        };
        let run = &out.run;
        let mut compared = Vec::with_capacity(k);
        for (lane, &index) in set.members.iter().enumerate() {
            ids_lines.push(json!({
                "kind": "lane", "policy": label, "k": k, "prompt": index, "lane": lane,
                "set": label_set, "ids": run.ids[lane],
            }));
            let comparison = compare(&references[index].ids[..args.tokens], &run.ids[lane], None);
            if !comparison.equal {
                if status == PolicyStatus::Pass {
                    status = PolicyStatus::Fail;
                }
                failures.push(format!("{label_set} prompt {index}"));
            }
            compared.push(compared_json(
                index,
                prompts[index].len(),
                lane,
                &comparison,
            ));
        }
        for w in &out.windows {
            ids_lines.push(json!({"kind": "window", "policy": label, "window": w}));
        }
        let equal = compared.iter().all(|c| c["equal"] == json!(true));
        let extras = burst_extras(&set, run, prompts, references);
        eprintln!("{}", burst_line(&label, &set, run, &compared, &extras));
        per_k.push(json!({
            "k": run.k, "equal": equal,
            "decode_wall_s": run.decode_wall,
            "aggregate_tok_s": run.aggregate_tok_s(),
        }));
        runs.push(with_extras(
            json!({
                "k": run.k,
                "equal": equal,
                "prompts": compared,
                "steps": run.steps,
                "decode_steps": run.decode_steps,
                "decode_tokens": run.decode_tokens,
                "decode_wall_s": run.decode_wall,
                "steps_wall_s": run.steps_wall,
                "prefill": run.prefill.json(),
                "aggregate_tok_s": run.aggregate_tok_s(),
                "mtp": out.summary,
                "rank_stats": out.rank_stats,
                "timing_includes_logits_read": false,
            }),
            extras,
        ));
    }
    let mut record = json!({
        "spec": label,
        "describe": format!("native-MTP EP lanes depth {depth} routes {}", routes.name()),
        "tokens": args.tokens,
        "status": status.label(),
        "all_equal": status == PolicyStatus::Pass,
        "failures": failures,
        "per_k": per_k,
        "runs": runs,
    });
    if let PolicyStatus::Error(message) = &status {
        record["error"] = json!(message);
    }
    Ok(vec![PolicyOutcome {
        spec: label,
        status,
        record,
        failures,
    }])
}

// ------------------------------------------------------------------- main

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail,
}

fn main() {
    if std::env::args().skip(1).any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return;
    }
    let code = match parse_args_from(std::env::args().skip(1)).and_then(run) {
        Ok(Verdict::Pass) => 0,
        Ok(Verdict::Fail) => 1,
        Err(error) => {
            if error.starts_with(TOOL) {
                eprintln!("{error}");
            } else {
                eprintln!("{TOOL}: {error}");
            }
            2
        }
    };
    std::process::exit(code);
}

fn write_json(path: &Path, value: &Value) -> R<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(e)?)
        .map_err(|x| format!("write {}: {x}", path.display()))
}

fn write_ids(path: &Path, lines: &[Value]) -> R<()> {
    let mut text = String::new();
    for line in lines {
        text.push_str(&line.to_string());
        text.push('\n');
    }
    std::fs::write(path, text).map_err(|x| format!("write {}: {x}", path.display()))
}

// ----------------------------------------------------------------- manifest

const MANIFEST_SCHEMA: u64 = 1;

/// The environment a run records (report and manifest): `HIPFIRE_*` plus the
/// device selectors and `HOME`.
fn recorded_env() -> serde_json::Map<String, Value> {
    std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("HIPFIRE_")
                || matches!(
                    key.as_str(),
                    "ROCR_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES" | "FABRIC_CARDS" | "HOME"
                )
        })
        .map(|(key, value)| (key, Value::String(value)))
        .collect()
}

/// One shell word: bare when it is made only of safe characters, otherwise
/// single-quoted.
fn shell_quote(word: &str) -> String {
    let safe = !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_./=:,+@%".contains(&b));
    if safe {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}

fn absolute_path(path: &str, cwd: &Path) -> String {
    cwd.join(path).display().to_string()
}

/// The invocation with every option spelled out (paths made absolute against
/// `cwd`, defaults filled in). Parsing it yields the same [`Args`] as the
/// original invocation, so the manifest command reproduces the run.
fn expanded_argv(args: &Args, cwd: &Path) -> Vec<String> {
    let mut argv: Vec<String> = Vec::new();
    let mut push = |flag: &str, value: String| {
        argv.push(flag.to_string());
        argv.push(value);
    };
    push("--model", absolute_path(&args.model, cwd));
    for path in &args.prompt_files {
        push("--prompt-file", absolute_path(path, cwd));
    }
    push("--max-seq", args.max_seq.to_string());
    push("--tokens", args.tokens.to_string());
    push(
        "--ks",
        args.ks
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    for burst in &args.bursts {
        push(
            "--burst",
            burst
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    push("--out", absolute_path(&args.out.display().to_string(), cwd));
    push("--pp", args.pp.to_string());
    if let Some(budget) = args.row_budget {
        push("--row-budget", budget.to_string());
    }
    let default_policy = args.policies.len() == 1 && args.policies[0].policy.is_none();
    if !default_policy {
        push(
            "--policies",
            args.policies
                .iter()
                .map(|entry| entry.spec.as_str())
                .collect::<Vec<_>>()
                .join(";"),
        );
    }
    if let Some(sweep) = args.tokens_sweep {
        push("--tokens-sweep", sweep.to_string());
    }
    if let Some(lanes) = args.reserve_lanes {
        push("--reserve-lanes", lanes.to_string());
    }
    push("--spec", args.spec.name().to_string());
    if let Some(k) = args.k {
        push("--k", k.to_string());
    }
    if let Some(routes) = args.routes {
        push("--routes", routes.name().to_string());
    }
    if let Some(seconds) = args.soak_seconds {
        push("--soak-seconds", seconds.to_string());
    }
    if let Some(seed) = args.soak_seed {
        push("--soak-seed", seed.to_string());
    }
    if let Some(point) = args.fault_point {
        push("--fault-point", point.name().to_string());
    }
    if args.copy {
        argv.push("--copy".to_string());
    }
    if args.skip_model_hash {
        argv.push("--skip-model-hash".to_string());
    }
    if args.logits {
        argv.push("--logits".to_string());
    }
    argv
}

/// `exe` and `argv` as one copy-pasteable shell command.
fn expanded_command(exe: &str, argv: &[String]) -> String {
    std::iter::once(exe)
        .chain(argv.iter().map(String::as_str))
        .map(shell_quote)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Size and SHA-256 of every `--prompt-file`, in lane order.
fn fixture_records(args: &Args, cwd: &Path) -> R<Vec<Value>> {
    args.prompt_files
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let absolute = absolute_path(path, cwd);
            let bytes = std::fs::metadata(&absolute)
                .map_err(|x| format!("prompt fixture {path}: {x}"))?
                .len();
            Ok(json!({
                "index": index,
                "path": path,
                "absolute_path": absolute,
                "bytes": bytes,
                "sha256": file_sha256(Path::new(&absolute))?,
            }))
        })
        .collect()
}

/// The speculative-lane selection as recorded in the report and manifest.
fn spec_json(args: &Args) -> Value {
    json!({
        "spec": args.spec.name(),
        "k": args.k,
        "routes": args.routes.map(RoutesArg::name),
        "soak_seconds": args.soak_seconds,
        "soak_seed": args.soak_seed,
        "fault_point": args.fault_point.map(FaultPoint::name),
    })
}

/// `manifest.json` before the model hash is known (`"pending"` until the run
/// computes it; `null` under `--skip-model-hash`).
fn build_manifest(
    args: &Args,
    invoked_argv: &[String],
    exe: &str,
    binary_sha256: &str,
    cwd: &Path,
    fixtures: Vec<Value>,
    env: serde_json::Map<String, Value>,
) -> Value {
    let argv = expanded_argv(args, cwd);
    let distinct: BTreeSet<&str> = fixtures
        .iter()
        .filter_map(|fixture| fixture["sha256"].as_str())
        .collect();
    json!({
        "tool": TOOL,
        "schema": MANIFEST_SCHEMA,
        "command": expanded_command(exe, &argv),
        "expanded_argv": argv,
        "invoked_argv": invoked_argv,
        "cwd": cwd.display().to_string(),
        "binary": {
            "path": exe,
            "sha256": binary_sha256,
            "package_version": env!("CARGO_PKG_VERSION"),
        },
        "model": {
            "path": absolute_path(&args.model, cwd),
            "sha256": if args.skip_model_hash { Value::Null } else { json!("pending") },
        },
        "prompts": fixtures,
        "distinct_prompts": distinct.len(),
        "ks": args.ks,
        "bursts": args.bursts,
        "pp": args.pp,
        "max_seq": args.max_seq,
        "tokens": args.tokens,
        "spec": spec_json(args),
        "env": env,
        "completed": false,
    })
}

/// The interface boundary of `--spec mtp`: the EP/PP speculative lane protocol
/// it would drive (S3/S5) has not landed, so the invocation is refused with an
/// explicit error. Never a pass: no lane window is run and no verdict exists.
fn ensure_spec_supported(args: &Args) -> R<()> {
    match args.spec {
        SpecMode::Ar => Ok(()),
        SpecMode::Mtp => {
            let mut shape = format!(
                "k={} routes={}",
                args.k.unwrap_or(DEFAULT_K),
                args.routes.map_or("cycle", RoutesArg::name)
            );
            if let Some(seconds) = args.soak_seconds {
                shape.push_str(&format!(" soak={seconds}s"));
            }
            if let Some(point) = args.fault_point {
                shape.push_str(&format!(" fault-point={}", point.name()));
            }
            // The EP lane protocol (S3) runs the plain greedy matrix; the PP arm
            // (S5), the soak and the fault points are other owners' boundaries.
            if args.pp > 1 || args.soak_seconds.is_some() || args.fault_point.is_some() {
                return Err(format!(
                    "{TOOL}: --spec mtp ({shape} pp={}) is not yet implemented by S3/S5: only the EP2 greedy matrix (--pp 1, no --soak-seconds/--fault-point) executes. This refusal is the interface boundary, not a pass; the invocation was validated and recorded in manifest.json",
                    args.pp
                ));
            }
            Ok(())
        }
    }
}

/// Writes `manifest.json`, records the speculative selection in `report`, then
/// checks the spec is executable. A fixture that cannot be read fails here.
fn prepare_run(args: &Args, report: &mut Value, manifest: &mut Value) -> R<()> {
    let cwd = std::env::current_dir().map_err(e)?;
    let exe = std::fs::read_link("/proc/self/exe")
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let binary_sha256 = file_sha256(Path::new("/proc/self/exe"))?;
    let invoked: Vec<String> = std::env::args().skip(1).collect();
    *manifest = build_manifest(
        args,
        &invoked,
        &exe,
        &binary_sha256,
        &cwd,
        fixture_records(args, &cwd)?,
        recorded_env(),
    );
    write_json(&args.out.join("manifest.json"), manifest)?;
    report["manifest"] = json!("manifest.json");
    report["spec"] = spec_json(args);
    ensure_spec_supported(args)
}

/// Completes the manifest with the run's outcome (model hash, completion,
/// verdict, error). A manifest that was never built stays absent.
fn finalize_manifest(manifest: &mut Value, report: &Value) {
    if manifest.is_null() {
        return;
    }
    if !report["model_sha256"].is_null() {
        manifest["model"]["sha256"] = report["model_sha256"].clone();
    }
    manifest["completed"] = report["completed"].clone();
    manifest["verdict"] = report["verdict"].clone();
    manifest["error"] = report["error"].clone();
}

/// Refuse an existing DIR, create it, run, and always leave `report.json`
/// (with `error` on a refusal/error), `ids.jsonl` and `manifest.json` behind.
fn run(args: Args) -> R<Verdict> {
    if args.out.exists() {
        return Err(format!(
            "output directory {} already exists (refusing to overwrite)",
            args.out.display()
        ));
    }
    std::fs::create_dir_all(&args.out)
        .map_err(|x| format!("create {}: {x}", args.out.display()))?;
    let mut report = json!({"tool": TOOL, "completed": false});
    let mut manifest = Value::Null;
    let mut ids_lines = Vec::new();
    let result = prepare_run(&args, &mut report, &mut manifest)
        .and_then(|()| run_inner(&args, &mut report, &mut ids_lines));
    if let Err(error) = &result {
        report["error"] = json!(error);
    }
    finalize_manifest(&mut manifest, &report);
    // `HIPFIRE_EP_LAB_UNIT_HITS=1` with a unit plan: the per-rank, per-layer VRAM hit counters.
    let hits = hipfire_arch_qwen4::ep::lab_unit_hits::take();
    if !hits.is_empty() {
        let text = hipfire_arch_qwen4::ep::lab_unit_hits::report(&hits, 2_611_200);
        eprint!("{text}");
        let _ = std::fs::write(args.out.join("unit_hits.txt"), text);
    }
    write_ids(&args.out.join("ids.jsonl"), &ids_lines)?;
    write_json(&args.out.join("report.json"), &report)?;
    if !manifest.is_null() {
        write_json(&args.out.join("manifest.json"), &manifest)?;
    }
    result
}

/// `load_model`'s admitted route (`admit_source_with_options` then
/// `load_admitted_with_gemma4_drafter`, the oracle's defaults throughout)
/// with the per-load VMM batch request set on the admission, so the loader
/// reserves `lanes` AR lanes in the EP expert placement. Planner capacity
/// only: the forward numerics and the oracle's lanes are unchanged.
fn load_with_lane_reservation(
    args: &Args,
    lanes: usize,
    cask: &hipfire_runtime::loader_api::CaskConfig,
    spec: hipfire_runtime::loader_api::SpecLoadCfg,
    gpu: &mut Gpu,
) -> R<hipfire_loader::LoadedModel> {
    use hipfire_loader::admission::{
        admit_source_with_options, legacy_warning, qwen4_kv_adaptive_requested,
        qwen_default_q8_enabled, BatchLoadRequest, KvBackendHints, KvBackendRequest,
        SourceAdmissionOptions,
    };
    let mut admission = admit_source_with_options(
        &args.model,
        1,
        args.pp,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        args.max_seq,
        KvBackendHints {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: qwen_default_q8_enabled(),
            kv_adaptive: None,
            cask: Some(cask),
            deepseek4_heterogeneous: false,
            vmm_runtime_available: gpu.vmm_recommended_granularity().is_ok(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        },
        SourceAdmissionOptions {
            spec,
            kv_adaptive: qwen4_kv_adaptive_requested(None),
            cask: cask.sidecar.is_some(),
            ..Default::default()
        },
    )?;
    // The oracle stages `max(ks)` lanes with `--row-budget` (default max(ks))
    // decode rows; the reserved row budget is that same figure (the loader
    // takes `max(row_budget, lanes)`).
    admission.batch_request = BatchLoadRequest {
        vmm_batch: true,
        continuous_batch_size: lanes,
        row_budget: args.row_budget.unwrap_or_else(|| args.max_lanes()),
        // `--spec mtp` stages spec lanes: the EP plan charges that store.
        spec_k: (args.spec == SpecMode::Mtp).then(|| args.k.unwrap_or(DEFAULT_K)),
        spec_copy: args.spec == SpecMode::Mtp && args.copy,
    };
    let warning = legacy_warning(admission.kv_backend, admission.kv_backend_reason.as_deref());
    let loaded = hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        &args.model,
        args.max_seq,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        None,
        None,
        cask,
        args.pp,
        spec,
        gpu,
    );
    if loaded.is_ok() {
        if let Some(warning) = warning {
            eprintln!("{warning}");
        }
    }
    loaded
}

fn run_inner(args: &Args, report: &mut Value, ids_lines: &mut Vec<Value>) -> R<Verdict> {
    eprintln!("[{TOOL}] loading {}", args.model);
    // Daemon parity: keep host-mapped expert memory out of reclaim before the first HIP call (KFD userptr eviction stalls, ROCm/rocm-systems#12528).
    hipfire_loader::prepare_host_memory_for(Path::new(&args.model));
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    // `--spec mtp` arms the native MTP head on every EP rank (under EP the
    // loader attaches it only on an explicit request); the AR oracle loads as before.
    let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
        mtp: (args.spec == SpecMode::Mtp).then_some(true),
        ..Default::default()
    };
    let mut model = match args.reserve_lanes {
        None => hipfire_loader::load_model(
            &args.model,
            args.max_seq,
            None,
            None,
            None,
            None,
            &cask,
            args.pp,
            spec_cfg,
            &mut gpu,
        )?,
        Some(lanes) => load_with_lane_reservation(args, lanes, &cask, spec_cfg, &mut gpu)?,
    };

    let prompts = {
        let tokenizer = model
            .tokenizer
            .as_ref()
            .ok_or("loaded model has no tokenizer")?;
        args.prompt_files
            .iter()
            .map(|path| load_prompt(path, tokenizer))
            .collect::<R<Vec<Prompt>>>()?
    };
    for prompt in &prompts {
        if prompt.ids.len() + args.tokens > args.max_seq {
            return Err(format!(
                "{}: {} prompt tokens + {} decode tokens exceed --max-seq {}",
                prompt.path,
                prompt.ids.len(),
                args.tokens,
                args.max_seq
            ));
        }
    }

    let (bundle, mut pp_gpus) = if args.pp > 1 {
        let (bundle, gpus) = model
            .qwen4_mesh_mut()
            .ok_or("PP lanes need a Qwen4 bundle with caller-owned pp_gpus")?;
        if !bundle.pp_mesh_active() {
            return Err("PP lanes need an active Qwen4 PP mesh".into());
        }
        (bundle, Some(gpus))
    } else {
        (
            model
                .qwen4_mut()
                .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?,
            None,
        )
    };
    let ranks = pp_gpus
        .as_ref()
        .map_or_else(|| bundle.ranks(), |gpus| gpus.devices.len());
    let ep = bundle.ep_mesh().is_some();
    let vocab = bundle.config.vocab_size;
    gpu.bind_thread().map_err(e)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(e)?;

    let mut devices = Vec::new();
    let mut pp_topology = None;
    if let Some(gpus) = pp_gpus.as_ref() {
        let owner = gpus
            .mesh_output_owner()
            .ok_or("PP mesh has no output owner")?;
        if gpus.devices.get(owner).map(|owner| owner.device_id) != Some(gpu.device_id) {
            return Err("PP output owner is not the daemon logits device".into());
        }
        devices.extend(gpus.devices.iter().enumerate().map(|(rank, gpu)| {
            let mut info = device_info(gpu);
            info["rank"] = json!(rank);
            info
        }));
        let band_counts: Vec<_> = bundle
            .pp_band_states()
            .ok_or("PP mesh has no band states")?
            .map(|state| state.scope_map().band().len())
            .collect();
        pp_topology = Some((band_counts, owner));
    } else {
        devices.push(device_info(&gpu));
        for rank in 1..ranks {
            devices.push(bundle.with_rank_mut(&mut gpu, rank, |_, peer, _| Ok(device_info(peer)))?);
        }
    }

    let model_sha = if args.skip_model_hash {
        Value::Null
    } else {
        eprintln!("[{TOOL}] hashing model (use --skip-model-hash to skip)");
        Value::String(file_sha256(Path::new(&args.model))?)
    };
    let exe = std::fs::read("/proc/self/exe").map_err(e)?;
    let env = recorded_env();
    let max_lanes_requested = args.max_lanes();
    let row_budget = args.row_budget.unwrap_or(max_lanes_requested);
    report["binary"] = json!(std::fs::read_link("/proc/self/exe")
        .map(|p| p.display().to_string())
        .unwrap_or_default());
    report["binary_md5"] = json!(md5_hex(&exe));
    report["model_path"] = json!(args.model);
    report["model_sha256"] = model_sha;
    report["env"] = json!(env);
    let mut topology = json!({"pp": args.pp, "ep": ep, "ranks": ranks, "devices": devices});
    if let Some((band_counts, owner)) = pp_topology {
        topology["band_counts"] = json!(band_counts);
        topology["output_owner"] = json!(owner);
    }
    report["topology"] = topology;
    report["max_seq"] = json!(args.max_seq);
    report["tokens"] = json!(args.tokens);
    report["vocab"] = json!(vocab);
    report["ks"] = json!(args.ks);
    report["bursts"] = json!(args.bursts);
    report["max_lanes"] = json!(max_lanes_requested);
    report["row_budget"] = json!(row_budget);
    report["logits_compared"] = json!(args.logits);
    report["tokens_sweep"] = json!(args.tokens_sweep);
    report["reserve_lanes"] = json!(args.reserve_lanes);
    report["policies_requested"] = json!(args
        .policies
        .iter()
        .map(|entry| entry.spec.as_str())
        .collect::<Vec<_>>());
    report["prompts"] = json!(prompts
        .iter()
        .enumerate()
        .map(|(index, prompt)| json!({
            "prompt": index,
            "path": prompt.path,
            "kind": prompt.kind,
            "file_md5": prompt.file_md5,
            "token_count": prompt.ids.len(),
            "ids_md5": prompt.ids_md5,
        }))
        .collect::<Vec<_>>());

    // Reference: the serialized singleton run of every prompt.
    let mut references = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        eprintln!(
            "[{TOOL}] reference {index}: {} ({} tokens)",
            prompt.path,
            prompt.ids.len()
        );
        let reference = match pp_gpus.as_deref_mut() {
            Some(gpus) => reference_run(
                &mut PpSingle {
                    bundle: &mut *bundle,
                    gpus,
                    gpu: &mut gpu,
                    logits: &logits,
                },
                &prompt.ids,
                args.tokens,
                args.logits,
            ),
            None => reference_run(
                &mut LocalSingle {
                    bundle: &mut *bundle,
                    gpu: &mut gpu,
                    logits: &logits,
                },
                &prompt.ids,
                args.tokens,
                args.logits,
            ),
        }?;
        ids_lines.push(json!({"kind": "reference", "prompt": index, "ids": reference.ids}));
        references.push(reference);
    }
    report["reference"] = json!(references
        .iter()
        .enumerate()
        .map(|(index, r)| json!({
            "prompt": index,
            "ids_md5": ids_md5(&r.ids),
            "prompt_len": prompts[index].ids.len(),
            "prefill_ms": r.prefill_ms,
        }))
        .collect::<Vec<_>>());

    // Lanes: staged once, then every policy's whole K matrix.
    let prompt_ids: Vec<&[u32]> = prompts.iter().map(|prompt| prompt.ids.as_slice()).collect();
    let outcomes = match pp_gpus.as_deref_mut() {
        Some(_) if args.spec == SpecMode::Mtp => {
            return Err("--spec mtp under --pp is the PP arm, not implemented here".into());
        }
        None if args.spec == SpecMode::Mtp => {
            if !ep {
                return Err(
                    "--spec mtp runs the EP2 mesh lane protocol: load with topology.ep (the single-card MTP lane oracle is fn_lane_oracle)"
                        .into(),
                );
            }
            drive_mtp_ep(
                bundle,
                &mut gpu,
                args,
                &prompt_ids,
                &references,
                max_lanes_requested,
                row_budget,
                vocab,
                ids_lines,
            )
        }
        Some(gpus) => drive_all(
            &mut PpLanes {
                bundle,
                gpus,
                vocab,
            },
            args,
            &prompt_ids,
            &references,
            max_lanes_requested,
            row_budget,
            ids_lines,
        ),
        None => {
            gpu.bind_thread().map_err(e)?;
            drive_all(
                &mut LocalLanes {
                    bundle,
                    gpu: &mut gpu,
                    vocab,
                },
                args,
                &prompt_ids,
                &references,
                max_lanes_requested,
                row_budget,
                ids_lines,
            )
        }
    }?;
    report["policies"] = json!(outcomes
        .iter()
        .map(|outcome| outcome.record.clone())
        .collect::<Vec<_>>());
    // `runs` stays the first policy's matrix for existing report readers.
    report["runs"] = outcomes[0].record["runs"].clone();
    let unrun = outcomes
        .iter()
        .filter(|outcome| {
            matches!(
                outcome.status,
                PolicyStatus::Error(_) | PolicyStatus::Skipped(_)
            )
        })
        .count();
    let verdict = sweep_verdict(&outcomes);
    report["completed"] = json!(unrun == 0);
    report["verdict"] = json!(match (unrun, &verdict) {
        (0, Verdict::Pass) => "PASS",
        (0, Verdict::Fail) => "FAIL",
        _ => "ERROR",
    });
    let shape = format!(
        "ks={:?} prompts={} tokens={} pp={} ranks={ranks} ep={ep} logits={}",
        args.ks,
        args.max_lanes(),
        args.tokens,
        args.pp,
        args.logits
    );
    let summary = sweep_summary(&outcomes);
    if unrun > 0 {
        println!(
            "ERROR {TOOL}: {unrun} of {} policies errored or were skipped ({shape}) policies: {summary}",
            outcomes.len()
        );
        return Err(format!(
            "{unrun} of {} policies errored or were skipped: {summary}",
            outcomes.len()
        ));
    }
    match verdict {
        Verdict::Pass => println!(
            "PASS {TOOL}: all lane ids equal their singleton under {} polic{} ({shape}) policies: {summary}",
            outcomes.len(),
            if outcomes.len() == 1 { "y" } else { "ies" }
        ),
        Verdict::Fail => println!(
            "FAIL {TOOL}: lane ids differ from the singleton under {} of {} policies ({shape}) policies: {summary}",
            outcomes
                .iter()
                .filter(|outcome| outcome.status == PolicyStatus::Fail)
                .count(),
            outcomes.len()
        ),
    }
    Ok(verdict)
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn argv(line: &str) -> impl Iterator<Item = String> + '_ {
        line.split_whitespace().map(str::to_owned)
    }

    const BASE: &str = "--model m.mq4 --max-seq 4096 --tokens 8 --out /tmp/x";

    fn files(n: usize) -> String {
        (0..n).map(|i| format!(" --prompt-file p{i}.txt")).collect()
    }

    #[test]
    fn md5_known_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn args_parse_sort_and_dedupe_ks() {
        let line = format!(
            "{BASE} --ks 4,1,2,4 --row-budget 3 --logits --skip-model-hash{}",
            files(4)
        );
        let args = parse_args_from(argv(&line)).unwrap();
        assert_eq!(args.ks, vec![1, 2, 4]);
        assert_eq!(args.row_budget, Some(3));
        assert!(args.logits && args.skip_model_hash);
        assert_eq!((args.pp, args.tokens, args.max_seq), (1, 8, 4096));
        assert_eq!(args.prompt_files.len(), 4);
    }

    #[test]
    fn args_accept_pp_degrees_and_refuse_zero() {
        let line = format!("{BASE} --ks 1 --pp 2{}", files(1));
        assert_eq!(parse_args_from(argv(&line)).unwrap().pp, 2);
        let line = format!("{BASE} --ks 1 --pp 1{}", files(1));
        assert_eq!(parse_args_from(argv(&line)).unwrap().pp, 1);
        let line = format!("{BASE} --ks 1 --pp 0{}", files(1));
        assert_eq!(
            parse_args_from(argv(&line)).unwrap_err(),
            "--pp must be at least 1"
        );
    }

    #[test]
    fn args_reserve_lanes_default_off_and_validated() {
        let line = format!("{BASE} --ks 1{}", files(1));
        assert_eq!(parse_args_from(argv(&line)).unwrap().reserve_lanes, None);
        let line = format!("{BASE} --ks 1 --reserve-lanes 8{}", files(1));
        assert_eq!(parse_args_from(argv(&line)).unwrap().reserve_lanes, Some(8));
        // Planner capacity only: independent of --ks / --row-budget.
        let line = format!("{BASE} --ks 2 --row-budget 2 --reserve-lanes 4{}", files(2));
        let args = parse_args_from(argv(&line)).unwrap();
        assert_eq!(
            (args.reserve_lanes, args.ks, args.row_budget),
            (Some(4), vec![2], Some(2))
        );
        let line = format!("{BASE} --ks 1 --reserve-lanes 0{}", files(1));
        assert_eq!(
            parse_args_from(argv(&line)).unwrap_err(),
            "--reserve-lanes must be at least 1"
        );
        let line = format!("{BASE} --ks 1 --reserve-lanes x{}", files(1));
        assert!(parse_args_from(argv(&line)).is_err());
        let line = format!("{BASE} --ks 1{} --reserve-lanes", files(1));
        assert_eq!(
            parse_args_from(argv(&line)).unwrap_err(),
            "--reserve-lanes needs a value"
        );
    }

    /// A fake singleton for `reference_run`: the logits row is a function of
    /// the position, the pick of the row is the `pick` helper.
    struct FakeSingle {
        prompt: Vec<u32>,
        position: usize,
        prefills: usize,
    }

    impl Singleton for FakeSingle {
        fn prefill(&mut self, prompt: &[u32]) -> R<()> {
            self.prompt = prompt.to_vec();
            self.position = prompt.len() - 1;
            self.prefills += 1;
            Ok(())
        }
        fn forward(&mut self) -> R<u32> {
            let id = pick(&self.prompt, self.position);
            self.position += 1;
            Ok(id)
        }
        fn logits_bytes(&mut self) -> R<Vec<u8>> {
            Ok(pick_bytes(&self.prompt, self.position))
        }
    }

    #[test]
    fn reference_run_matches_the_singleton_loop() {
        let prompt: Vec<u32> = (0..7).collect();
        let mut single = FakeSingle {
            prompt: Vec::new(),
            position: 0,
            prefills: 0,
        };
        let got = reference_run(&mut single, &prompt, 5, true).unwrap();
        let want = reference(&prompt, 5);
        assert_eq!(got.ids, want.ids);
        assert_eq!(got.logits_sha, want.logits_sha);
        assert_eq!(single.prefills, 1);
        let plain = reference_run(&mut single, &prompt, 5, false).unwrap();
        assert!(plain.logits_sha.is_empty());
        assert_eq!(plain.ids, want.ids);
    }

    #[test]
    fn args_refuse_too_few_prompts_and_bad_values() {
        let few = format!("{BASE} --ks 1,2,8{}", files(7));
        assert!(parse_args_from(argv(&few))
            .unwrap_err()
            .contains("at least 8 --prompt-file"));
        for bad in [
            "--ks 0",
            "--ks 1,x",
            "--ks ,",
            "--tokens 0 --ks 1",
            "--row-budget 0 --ks 1",
            "--bogus 1",
        ] {
            let line = format!("{BASE} {bad}{}", files(1));
            // The later duplicate of --tokens (0) wins over BASE's value.
            assert!(parse_args_from(argv(&line)).is_err(), "accepted {bad}");
        }
        let missing = format!("--max-seq 8 --tokens 1 --out /tmp/y --ks 1{}", files(1));
        assert_eq!(
            parse_args_from(argv(&missing)).unwrap_err(),
            "--model is required"
        );
    }

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    #[test]
    fn plan_mixes_prefill_tiles_and_ar_rows_in_slot_order() {
        // Slot 3: prefill tile of 3 rows from position 4; slot 1: AR row at 9;
        // slot 2: prefill tile of 2 rows from 0. Slot 0 idle.
        let plan = build_plan(
            4,
            vec![
                RowSpec {
                    slot: 3,
                    epoch: epoch(30),
                    tokens: vec![7, 8, 9],
                    start: 4,
                    kind: RequestStepKind::Prefill,
                },
                RowSpec {
                    slot: 1,
                    epoch: epoch(10),
                    tokens: vec![55],
                    start: 9,
                    kind: RequestStepKind::Ar,
                },
                RowSpec {
                    slot: 2,
                    epoch: epoch(20),
                    tokens: vec![1, 2],
                    start: 0,
                    kind: RequestStepKind::Prefill,
                },
            ],
        );
        assert_eq!(plan.batch.m_per_slot, vec![0, 1, 2, 3]);
        assert_eq!(plan.batch.tokens, vec![55, 1, 2, 7, 8, 9]);
        assert_eq!(plan.batch.positions, vec![9, 0, 1, 4, 5, 6]);
        assert_eq!(plan.batch.row_slot, vec![1, 2, 2, 3, 3, 3]);
        assert_eq!(
            (
                plan.decode_rows,
                plan.prefill_rows,
                plan.verify_rows,
                plan.forced_rows
            ),
            (1, 5, 0, 0)
        );
        let ranges: Vec<_> = plan
            .requests
            .iter()
            .map(|r| (r.epoch.request_tag, r.rows.begin, r.rows.len, r.kind))
            .collect();
        assert_eq!(
            ranges,
            vec![
                (10, 0, 1, RequestStepKind::Ar),
                (20, 1, 2, RequestStepKind::Prefill),
                (30, 3, 3, RequestStepKind::Prefill),
            ]
        );
    }

    #[test]
    fn logits_rows_follow_the_store_layout() {
        // max_lanes 4: AR rows are 0.., the i-th Prefill request is 4 + i, and
        // only a prompt-completing tile (a pick on its last row) has a row.
        let plan = build_plan(
            4,
            vec![
                RowSpec {
                    slot: 0,
                    epoch: epoch(1),
                    tokens: vec![5],
                    start: 3,
                    kind: RequestStepKind::Ar,
                },
                RowSpec {
                    slot: 1,
                    epoch: epoch(2),
                    tokens: vec![1, 2],
                    start: 0,
                    kind: RequestStepKind::Prefill,
                },
                RowSpec {
                    slot: 2,
                    epoch: epoch(3),
                    tokens: vec![6],
                    start: 8,
                    kind: RequestStepKind::Ar,
                },
                RowSpec {
                    slot: 3,
                    epoch: epoch(4),
                    tokens: vec![3, 4],
                    start: 2,
                    kind: RequestStepKind::Prefill,
                },
            ],
        );
        // Flat rows: AR(0) | P(1,2) | AR(3) | P(4,5). Request 2 does not
        // complete (no pick on row 2); request 4 does (pick on row 5).
        let picks = [10, u32::MAX, u32::MAX, 11, u32::MAX, 12];
        let rows = logits_rows(&plan, &picks, 4);
        assert_eq!(rows, vec![(epoch(1), 0), (epoch(3), 1), (epoch(4), 5)],);
    }

    #[test]
    fn schedule_plans_every_prefill_tile_and_budgeted_ar_rows() {
        let p = |prompt_len, prefilled, ar_steps| Progress {
            prompt_len,
            prefilled,
            ar_steps,
        };
        let lanes = [p(10, 4, 0), p(5, 5, 2), p(5, 5, 0), p(8, 8, 8), p(6, 6, 1)];
        // Lane 3 already has all tokens; lane 0 is prefilling.
        assert_eq!(schedule(&lanes, 8, 8), (vec![0], vec![1, 2, 4]));
        // Budget 2: the fewest AR rows first (lanes 2 then 4), plan order by index.
        assert_eq!(schedule(&lanes, 8, 2), (vec![0], vec![2, 4]));
        assert_eq!(schedule(&lanes, 8, 1), (vec![0], vec![2]));
        let done = [p(5, 5, 3), p(5, 5, 3)];
        assert_eq!(schedule(&done, 3, 4), (vec![], vec![]));
    }

    #[test]
    fn comparison_reports_ids_and_logits() {
        let reference = [1u32, 2, 3, 4];
        let same = compare(&reference, &reference, None);
        assert!(same.equal && same.first_mismatch.is_none());
        assert_eq!(same.ref_md5, same.lane_md5);
        assert_eq!(same.logits_first_mismatch, None);

        let differ = compare(&reference, &[1, 2, 9, 4], None);
        assert!(!differ.equal);
        assert_eq!(differ.first_mismatch, Some(2));
        assert_ne!(differ.ref_md5, differ.lane_md5);

        let short = compare(&reference, &[1, 2], None);
        assert_eq!((short.equal, short.first_mismatch), (false, Some(2)));
        let long = compare(&reference, &[1, 2, 3, 4, 5], None);
        assert_eq!((long.equal, long.first_mismatch), (false, Some(4)));

        // Equal ids with a differing logits row still fails.
        let a: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        let mut b = a.clone();
        let ok = compare(&reference, &reference, Some((&a, &b)));
        assert!(ok.equal && ok.logits_first_mismatch == Some(None));
        b[1] = "x".into();
        let bad = compare(&reference, &reference, Some((&a, &b)));
        assert!(!bad.equal);
        assert_eq!(bad.first_mismatch, None);
        assert_eq!(bad.logits_first_mismatch, Some(Some(1)));
        let record = comparison_json(3, &bad);
        assert_eq!(record["prompt"], json!(3));
        assert_eq!(record["equal"], json!(false));
        assert_eq!(record["ids_equal"], json!(true));
        assert_eq!(record["first_logits_mismatch"], json!(1));
        assert_eq!(record["ref_md5"], record["lane_md5"]);
    }

    // ----- a deterministic fake backend standing in for the lane executor.

    /// The greedy pick after the row at `position` of `prompt`.
    fn pick(prompt: &[u32], position: usize) -> u32 {
        (prompt.iter().sum::<u32>() + 7 * position as u32 + 3) % 1000
    }

    fn pick_bytes(prompt: &[u32], position: usize) -> Vec<u8> {
        pick(prompt, position).to_le_bytes().repeat(3)
    }

    struct FakeLane {
        slot: usize,
        prompt: Vec<u32>,
        prefilled: usize,
        pending: Option<u32>,
        position: usize,
    }

    struct FakeLanes {
        capacity: usize,
        tile: usize,
        lanes: HashMap<u64, FakeLane>,
        /// (prefill requests, AR requests) of every executed step.
        steps: Vec<(usize, usize)>,
        released: Vec<u64>,
        /// The store's current policy (construction: all per lane).
        policy: LaneStagePolicy,
        /// While this policy is set, AR picks are wrong (a stage that is not
        /// byte-exact when shared).
        corrupt: Option<LaneStagePolicy>,
        /// While this policy is set, every step errors.
        fail: Option<LaneStagePolicy>,
        /// A failing step leaves the store unusable (set_policy refuses).
        break_on_fail: bool,
        broken: bool,
    }

    impl FakeLanes {
        fn new(capacity: usize, tile: usize) -> Self {
            Self {
                capacity,
                tile,
                lanes: HashMap::new(),
                steps: Vec::new(),
                released: Vec::new(),
                policy: LaneStagePolicy::all_per_lane(),
                corrupt: None,
                fail: None,
                break_on_fail: false,
                broken: false,
            }
        }
    }

    impl LaneBackend for FakeLanes {
        fn stage(&mut self, _: usize, _: usize) -> R<()> {
            Ok(())
        }
        fn stage_policy(&self) -> R<LaneStagePolicy> {
            Ok(LaneStagePolicy::all_per_lane())
        }
        fn set_policy(&mut self, policy: LaneStagePolicy) -> R<()> {
            if self.broken {
                return Err("lane store is unusable".into());
            }
            self.policy = policy;
            Ok(())
        }
        fn lane_capacity(&self) -> R<usize> {
            Ok(self.capacity)
        }
        fn admit(&mut self, epoch: RequestEpoch, prompt: Vec<u32>) -> R<usize> {
            let slot = (0..self.capacity)
                .find(|s| self.lanes.values().all(|l| l.slot != *s))
                .ok_or("no free lane")?;
            self.lanes.insert(
                epoch.request_tag,
                FakeLane {
                    slot,
                    prompt,
                    prefilled: 0,
                    pending: None,
                    position: 0,
                },
            );
            Ok(slot)
        }
        fn release(&mut self, epoch: &RequestEpoch) -> R<()> {
            self.lanes
                .remove(&epoch.request_tag)
                .ok_or("release of an unknown lane")?;
            self.released.push(epoch.request_tag);
            Ok(())
        }
        fn prefill_tile_len(&mut self, epoch: &RequestEpoch) -> R<usize> {
            let lane = &self.lanes[&epoch.request_tag];
            Ok(self.tile.min(lane.prompt.len() - lane.prefilled))
        }
        fn step(&mut self, plan: &BatchStepPlan, read_logits: bool) -> R<StepResult> {
            if self.fail == Some(self.policy) {
                self.broken |= self.break_on_fail;
                return Err("forward_step failed: injected".into());
            }
            let corrupt = self.corrupt == Some(self.policy);
            let mut advances = Vec::new();
            let mut logits = Vec::new();
            let (mut pre, mut ar) = (0, 0);
            for r in &plan.requests {
                let tokens = &plan.batch.tokens[r.rows.begin..r.rows.end()];
                let lane = self
                    .lanes
                    .get_mut(&r.epoch.request_tag)
                    .ok_or("stale epoch")?;
                for (j, row) in (r.rows.begin..r.rows.end()).enumerate() {
                    if plan.batch.row_slot[row] != lane.slot as i32
                        || plan.batch.positions[row] as usize != lane.position + j
                    {
                        return Err("row slot/position disagree with the lane".into());
                    }
                }
                match r.kind {
                    RequestStepKind::Prefill => {
                        pre += 1;
                        if tokens != &lane.prompt[lane.prefilled..lane.prefilled + tokens.len()]
                            || tokens.len() != self.tile.min(lane.prompt.len() - lane.prefilled)
                        {
                            return Err("bad prefill tile".into());
                        }
                        lane.prefilled += tokens.len();
                        lane.position += tokens.len();
                        let mut committed = Vec::new();
                        if lane.prefilled == lane.prompt.len() {
                            let id = pick(&lane.prompt, lane.position - 1);
                            lane.pending = Some(id);
                            committed.push(id);
                            logits.push((r.epoch, pick_bytes(&lane.prompt, lane.position - 1)));
                        }
                        advances.push(RequestAdvance {
                            epoch: r.epoch,
                            committed_ids: committed,
                            committed_position: lane.position,
                            accepted_drafts: 0,
                            verified_rows: 0,
                            finish: None,
                        });
                    }
                    RequestStepKind::Ar => {
                        ar += 1;
                        if Some(tokens[0]) != lane.pending {
                            return Err("AR token is not the pending seed".into());
                        }
                        let id = pick(&lane.prompt, lane.position) + u32::from(corrupt);
                        logits.push((r.epoch, pick_bytes(&lane.prompt, lane.position)));
                        lane.position += 1;
                        lane.pending = Some(id);
                        advances.push(RequestAdvance {
                            epoch: r.epoch,
                            committed_ids: vec![id],
                            committed_position: lane.position,
                            accepted_drafts: 0,
                            verified_rows: 0,
                            finish: None,
                        });
                    }
                    _ => return Err("unexpected row kind".into()),
                }
            }
            self.steps.push((pre, ar));
            if !read_logits {
                logits.clear();
            }
            Ok(StepResult { advances, logits })
        }
    }

    fn reference(prompt: &[u32], tokens: usize) -> Reference {
        let ids: Vec<u32> = (0..tokens)
            .map(|s| pick(prompt, prompt.len() - 1 + s))
            .collect();
        let logits_sha = (0..=tokens)
            .map(|s| sha256_hex(&pick_bytes(prompt, prompt.len() - 1 + s)))
            .collect();
        Reference {
            ids,
            logits_sha,
            logits_raw: Vec::new(),
            prefill_ms: prompt.len() as f64,
        }
    }

    fn prompts() -> Vec<Vec<u32>> {
        // Different lengths: lanes reach decode at different steps.
        [9usize, 2, 5, 13]
            .iter()
            .enumerate()
            .map(|(i, &len)| (0..len as u32).map(|j| 11 * i as u32 + j).collect())
            .collect()
    }

    #[test]
    fn mixed_lanes_commit_the_singleton_ids_and_release() {
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let mut backend = FakeLanes::new(4, 3);
        let mut tag = 0u64;
        for want_logits in [false, true] {
            for k in [1usize, 2, 4] {
                let cfg = DriveCfg {
                    prompts: &slices,
                    tokens: 6,
                    max_lanes: 4,
                    row_budget: 4,
                    want_logits,
                };
                backend.steps.clear();
                backend.released.clear();
                let run = drive_k(&mut backend, &cfg, k, &mut tag).unwrap();
                assert_eq!(run.k, k);
                for i in 0..k {
                    let want = reference(&prompts[i], 6);
                    assert_eq!(run.ids[i], want.ids, "K={k} lane {i}");
                    if want_logits {
                        assert_eq!(run.logits_sha[i], want.logits_sha, "K={k} lane {i}");
                    } else {
                        assert!(run.logits_sha[i].is_empty());
                    }
                    assert!(compare(&want.ids, &run.ids[i], None).equal);
                }
                assert!(backend.lanes.is_empty(), "every lane is released");
                assert_eq!(backend.released.len(), k);
                assert_eq!(run.steps, backend.steps.len());
                // Decode rows: K lanes x 6 AR rows in total; decode-only rows
                // are what ran in steps without a prefill tile.
                let ar_total: usize = backend.steps.iter().map(|s| s.1).sum();
                assert_eq!(ar_total, k * 6);
                let decode_only: usize =
                    backend.steps.iter().filter(|s| s.0 == 0).map(|s| s.1).sum();
                assert_eq!(run.decode_tokens, decode_only);
                assert_eq!(run.lane_decode_rows.iter().sum::<usize>(), decode_only);
                assert_eq!(
                    run.decode_steps,
                    backend.steps.iter().filter(|s| s.0 == 0).count()
                );
                if k == 4 {
                    // Prefill tiles and AR rows share steps, and lanes with
                    // short prompts decode while long ones still prefill.
                    assert!(backend.steps.iter().any(|s| s.0 > 0 && s.1 > 0));
                    assert!(backend.steps.iter().any(|s| s.0 > 1));
                }
            }
        }
    }

    #[test]
    fn row_budget_limits_ar_rows_per_step_without_starving() {
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let mut backend = FakeLanes::new(4, 4);
        let cfg = DriveCfg {
            prompts: &slices,
            tokens: 5,
            max_lanes: 4,
            row_budget: 2,
            want_logits: true,
        };
        let run = drive_k(&mut backend, &cfg, 4, &mut 0).unwrap();
        assert!(backend.steps.iter().all(|s| s.1 <= 2));
        for (i, ids) in run.ids.iter().enumerate() {
            assert_eq!(*ids, reference(&prompts[i], 5).ids);
        }
    }

    #[test]
    fn a_failing_step_releases_every_lane() {
        struct Failing(FakeLanes);
        impl LaneBackend for Failing {
            fn stage(&mut self, a: usize, b: usize) -> R<()> {
                self.0.stage(a, b)
            }
            fn stage_policy(&self) -> R<LaneStagePolicy> {
                self.0.stage_policy()
            }
            fn set_policy(&mut self, policy: LaneStagePolicy) -> R<()> {
                self.0.set_policy(policy)
            }
            fn lane_capacity(&self) -> R<usize> {
                self.0.lane_capacity()
            }
            fn admit(&mut self, epoch: RequestEpoch, prompt: Vec<u32>) -> R<usize> {
                self.0.admit(epoch, prompt)
            }
            fn release(&mut self, epoch: &RequestEpoch) -> R<()> {
                self.0.release(epoch)
            }
            fn prefill_tile_len(&mut self, epoch: &RequestEpoch) -> R<usize> {
                self.0.prefill_tile_len(epoch)
            }
            fn step(&mut self, plan: &BatchStepPlan, read: bool) -> R<StepResult> {
                if self.0.steps.len() >= 2 {
                    return Err("forward_step failed: injected".into());
                }
                self.0.step(plan, read)
            }
        }
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let mut backend = Failing(FakeLanes::new(4, 2));
        let cfg = DriveCfg {
            prompts: &slices,
            tokens: 4,
            max_lanes: 4,
            row_budget: 4,
            want_logits: false,
        };
        let error = drive_k(&mut backend, &cfg, 3, &mut 0).unwrap_err();
        assert!(error.contains("injected"), "{error}");
        assert!(backend.0.lanes.is_empty());
    }

    #[test]
    fn k_beyond_capacity_is_refused() {
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let mut backend = FakeLanes::new(2, 4);
        let cfg = DriveCfg {
            prompts: &slices,
            tokens: 2,
            max_lanes: 2,
            row_budget: 2,
            want_logits: false,
        };
        assert!(drive_k(&mut backend, &cfg, 3, &mut 0).is_err());
        assert!(backend.lanes.is_empty());
    }

    // ----- policy sweep

    fn policy_of(spec: &str) -> LaneStagePolicy {
        LaneStagePolicy::parse(spec).unwrap()
    }

    #[test]
    fn g0_expands_in_stage_order_between_per_lane_and_production() {
        let entries = parse_policies("g0").unwrap();
        let specs: Vec<&str> = entries.iter().map(|entry| entry.spec.as_str()).collect();
        assert_eq!(
            specs,
            [
                "per-lane",
                "per-lane:embed=once",
                "per-lane:ple_rows=once",
                "per-lane:hc_read=once",
                "per-lane:hc_write=once",
                "per-lane:gdn_in=once",
                "per-lane:gdn_out=once",
                "per-lane:qsa_in=once",
                "per-lane:qsa_out=once",
                "per-lane:moe=once",
                "per-lane:final_hyper=once",
                "per-lane:head=once",
                "per-lane:argmax=once",
                "production",
            ]
        );
        assert_eq!(entries.len(), 14);
        // Stateful stages are never shared.
        assert!(specs.iter().all(|spec| {
            !spec.contains("ple_conv") && !spec.contains("gdn_core") && !spec.contains("qsa_core")
        }));
        for entry in &entries {
            assert_eq!(entry.policy, Some(policy_of(&entry.spec)), "{}", entry.spec);
        }
        assert_eq!(entries[0].policy, Some(LaneStagePolicy::all_per_lane()));
        assert_eq!(entries[13].policy, Some(LaneStagePolicy::production()));
        assert_eq!(
            entries[1].policy,
            Some(
                LaneStagePolicy::all_per_lane()
                    .with(
                        StageName::Embed,
                        hipfire_arch_qwen4::gpu_forward_lanes::StageShare::OnceAllRows
                    )
                    .unwrap()
            )
        );
    }

    #[test]
    fn policies_parse_in_order_with_g0_in_place() {
        let entries = parse_policies(" production ; g0 ;per-lane:moe=once,head=once;;").unwrap();
        assert_eq!(entries.len(), 1 + 14 + 1);
        assert_eq!(entries[0].spec, "production");
        assert_eq!(entries[1].spec, "per-lane");
        assert_eq!(entries[14].spec, "production");
        assert_eq!(entries[15].spec, "per-lane:moe=once,head=once");
        assert_eq!(
            entries[15].policy,
            Some(policy_of("per-lane:moe=once,head=once"))
        );
    }

    #[test]
    fn invalid_policy_specs_are_refused() {
        for bad in [
            "bogus",
            "per-lane:gdn_core=once",
            "per-lane:nope=once",
            "per-lane:moe=sometimes",
            "per-lane:moe",
            "production;bogus",
            "g0;nope",
            "",
            " ; ; ",
        ] {
            let error = parse_policies(bad).unwrap_err();
            assert!(error.starts_with("--policies"), "{bad:?}: {error}");
        }
        // Through the argument parser (a spec with spaces is one argv item).
        for bad in ["bogus", "per-lane:gdn_core=once", "production;bogus"] {
            let line = format!("{BASE} --ks 1 --policies {bad}{}", files(1));
            assert!(parse_args_from(argv(&line)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn args_default_to_the_store_policy_and_parse_sweep_flags() {
        let line = format!("{BASE} --ks 1{}", files(1));
        let args = parse_args_from(argv(&line)).unwrap();
        assert_eq!(
            args.policies,
            vec![PolicyEntry {
                spec: DEFAULT_POLICY_SPEC.to_string(),
                policy: None
            }]
        );
        assert_eq!(args.tokens_sweep, None);

        let line = format!(
            "{BASE} --ks 1 --policies g0;production --tokens-sweep 4{}",
            files(1)
        );
        let args = parse_args_from(argv(&line)).unwrap();
        assert_eq!(args.policies.len(), 15);
        assert_eq!(args.tokens_sweep, Some(4));

        for bad in ["--tokens-sweep 0", "--tokens-sweep 9", "--tokens-sweep x"] {
            let line = format!("{BASE} --ks 1 {bad}{}", files(1));
            assert!(parse_args_from(argv(&line)).is_err(), "accepted {bad}");
        }
    }

    fn sweep_args(extra: &str, ks: &str) -> Args {
        let line = format!("{BASE} --ks {ks} --tokens 6 --logits {extra}{}", files(4));
        parse_args_from(argv(&line)).unwrap()
    }

    #[test]
    fn first_divergence_names_the_first_differing_row_and_position() {
        let equal = compare(&[1, 2, 3], &[1, 2, 3], None);
        assert_eq!(first_divergence(10, &equal), None);
        // ids differ at index 2: row 2 follows the token at position 10 - 1 + 2.
        let ids = compare(&[1, 2, 3], &[1, 2, 9], None);
        assert_eq!(
            first_divergence(10, &ids),
            Some(json!({"row": 2, "position": 11, "phase": "decode", "via": "ids"}))
        );
        let reference = vec!["a".to_string(), "b".into()];
        // Logits differ at the prompt-completing row, ids only later: the logits win.
        let early = vec!["x".to_string(), "b".into()];
        let c = compare(&[1, 2], &[1, 3], Some((&reference, &early)));
        assert_eq!(
            first_divergence(60, &c),
            Some(json!({"row": 0, "position": 59, "phase": "prefill-final", "via": "logits"}))
        );
        // Both at the same row.
        let late = vec!["a".to_string(), "z".into()];
        let c = compare(&[1, 2], &[1, 3], Some((&reference, &late)));
        assert_eq!(
            first_divergence(60, &c),
            Some(json!({"row": 1, "position": 60, "phase": "decode", "via": "ids+logits"}))
        );
        let record = compared_json(7, 60, 2, &c);
        assert_eq!(
            (
                record["prompt"].clone(),
                record["lane"].clone(),
                record["prompt_len"].clone()
            ),
            (json!(7), json!(2), json!(60))
        );
        assert_eq!(record["first_divergence"]["row"], json!(1));
        assert_eq!(
            compared_json(7, 60, 2, &equal)["first_divergence"],
            Value::Null
        );
    }

    #[test]
    fn burst_args_parse_validate_and_drive_the_run_sets() {
        let line = format!("{BASE} --ks 1 --burst 3,1,0 --burst 2{}", files(4));
        let args = parse_args_from(argv(&line)).unwrap();
        assert_eq!(args.bursts, vec![vec![3, 1, 0], vec![2]]);
        assert_eq!(args.max_lanes(), 3);
        let sets = run_sets(&args);
        assert_eq!(
            sets.iter().map(RunSet::label).collect::<Vec<_>>(),
            ["K=1", "burst=3+1+0", "burst=2"]
        );
        assert_eq!(sets[0].members, vec![0]);
        assert_eq!(sets[1].members, vec![3, 1, 0]);
        assert!(!sets[0].burst && sets[1].burst);
        // The manifest command reproduces the bursts.
        let argv_out = expanded_argv(&args, Path::new("/w"));
        let at = argv_out.iter().position(|a| a == "--burst").unwrap();
        assert_eq!(argv_out[at + 1], "3,1,0");
        assert_eq!(argv_out.iter().filter(|a| *a == "--burst").count(), 2);
        // Without bursts nothing changes, and --ks still sets the lanes.
        let line = format!("{BASE} --ks 1,2{}", files(2));
        let args = parse_args_from(argv(&line)).unwrap();
        assert!(args.bursts.is_empty());
        assert_eq!(args.max_lanes(), 2);
        assert_eq!(run_sets(&args).len(), 2);
        // A burst longer than --ks raises the staged lanes.
        let line = format!("{BASE} --ks 1 --burst 0,1,2,3{}", files(4));
        assert_eq!(parse_args_from(argv(&line)).unwrap().max_lanes(), 4);
        for bad in ["--burst 0,0", "--burst x", "--burst ,", "--burst 4"] {
            let line = format!("{BASE} --ks 1 {bad}{}", files(4));
            assert!(parse_args_from(argv(&line)).is_err(), "accepted {bad}");
        }
        let line = format!("{BASE} --ks 1{} --burst", files(4));
        assert_eq!(
            parse_args_from(argv(&line)).unwrap_err(),
            "--burst needs a value"
        );
    }

    #[test]
    fn burst_runs_record_members_lengths_prefill_passes_and_singleton_sums() {
        let args = sweep_args("--burst 3,1,0", "1");
        let mut backend = FakeLanes::new(4, 3);
        let (outcomes, ids_lines) = sweep(&mut backend, &args);
        assert_eq!(outcomes[0].status, PolicyStatus::Pass);
        let runs = outcomes[0].record["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2, "the K=1 run, then the burst");
        assert_eq!(runs[0]["set"], json!("K=1"));
        assert_eq!(runs[0]["burst"], json!(false));
        let burst = &runs[1];
        assert_eq!(burst["set"], json!("burst=3+1+0"));
        assert_eq!(burst["burst"], json!(true));
        assert_eq!(burst["k"], json!(3));
        assert_eq!(burst["members"], json!([3, 1, 0]));
        assert_eq!(burst["prompt_lens"], json!([13, 2, 9]));
        assert_eq!(burst["rows_total"], json!(24));
        // reference() sets prefill_ms = prompt length.
        assert_eq!(burst["singleton_prefill_ms_sum"], json!(24.0));
        let prompts = burst["prompts"].as_array().unwrap();
        let at = |key: &str| prompts.iter().map(|p| p[key].clone()).collect::<Vec<_>>();
        assert_eq!(at("prompt"), [json!(3), json!(1), json!(0)]);
        assert_eq!(at("lane"), [json!(0), json!(1), json!(2)]);
        assert_eq!(at("prompt_len"), [json!(13), json!(2), json!(9)]);
        assert!(prompts.iter().all(|p| p["first_divergence"].is_null()));
        // Tile 3, lanes of 13 / 2 / 9 rows: five passes; the short lane's AR rows share the
        // later passes.
        let prefill = &burst["prefill"];
        assert_eq!(
            (prefill["steps"].clone(), prefill["rows"].clone()),
            (json!(5), json!(24))
        );
        let passes = prefill["passes"].as_array().unwrap();
        let shape = passes
            .iter()
            .map(|p| {
                (
                    p["tiles"].as_u64(),
                    p["rows"].as_u64(),
                    p["ar_rows"].as_u64(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            shape,
            [
                (Some(3), Some(8), Some(0)),
                (Some(2), Some(6), Some(1)),
                (Some(2), Some(6), Some(1)),
                (Some(1), Some(3), Some(2)),
                (Some(1), Some(1), Some(2)),
            ]
        );
        assert!(prefill["burst_wall_s"].as_f64().unwrap() >= prefill["wall_s"].as_f64().unwrap());
        // ids.jsonl names each lane's prompt index and set.
        let burst_lines: Vec<_> = ids_lines
            .iter()
            .filter(|l| l["set"] == json!("burst=3+1+0"))
            .collect();
        assert_eq!(
            burst_lines
                .iter()
                .map(|l| l["prompt"].clone())
                .collect::<Vec<_>>(),
            [json!(3), json!(1), json!(0)]
        );
    }

    #[test]
    fn a_diverging_burst_lane_reports_its_row_position_and_labels() {
        let args = sweep_args("--policies per-lane:moe=once --burst 3,1", "1");
        let mut backend = FakeLanes::new(4, 3);
        backend.corrupt = Some(policy_of("per-lane:moe=once"));
        let (outcomes, _) = sweep(&mut backend, &args);
        assert_eq!(outcomes[0].status, PolicyStatus::Fail);
        assert_eq!(
            outcomes[0].failures,
            ["K=1 prompt 0", "burst=3+1 prompt 3", "burst=3+1 prompt 1"]
        );
        let burst = &outcomes[0].record["runs"][1];
        assert_eq!(burst["equal"], json!(false));
        // The prompt-completing pick is right; the first AR pick (row 1) is wrong.
        let d = &burst["prompts"][0]["first_divergence"];
        assert_eq!(d["row"], json!(1));
        assert_eq!(d["position"], json!(13));
        assert_eq!(d["phase"], json!("decode"));
        assert_eq!(d["via"], json!("ids"));
        assert_eq!(
            burst["prompts"][1]["first_divergence"]["position"],
            json!(2)
        );
    }

    #[test]
    fn the_burst_stderr_line_carries_lengths_passes_and_the_singleton_sum() {
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let mut backend = FakeLanes::new(4, 3);
        let cfg = DriveCfg {
            prompts: &slices[..2],
            tokens: 3,
            max_lanes: 4,
            row_budget: 4,
            want_logits: false,
        };
        let run = drive_k(&mut backend, &cfg, 2, &mut 0).unwrap();
        let references: Vec<Reference> = prompts.iter().map(|p| reference(p, 3)).collect();
        let set = RunSet {
            members: vec![0, 1],
            burst: true,
        };
        let extras = burst_extras(&set, &run, &slices, &references);
        let compared: Vec<Value> = (0..2)
            .map(|i| {
                let ids = &references[i].ids;
                compared_json(i, prompts[i].len(), i, &compare(ids, ids, None))
            })
            .collect();
        let line = burst_line("p", &set, &run, &compared, &extras);
        assert!(
            line.starts_with("[oracle-burst] policy=p set=burst=0+1 lens=[9, 2] prompts_equal=2/2 passes=3 rows=11 "),
            "{line}"
        );
        assert!(line.contains("singleton_sum_ms=11.0"), "{line}");
        assert!(
            line.ends_with("first_divergence(prompt,row,pos)=[]"),
            "{line}"
        );
    }

    fn sweep(backend: &mut dyn LaneBackend, args: &Args) -> (Vec<PolicyOutcome>, Vec<Value>) {
        let prompts = prompts();
        let slices: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let references: Vec<Reference> = prompts.iter().map(|p| reference(p, 6)).collect();
        let mut ids_lines = Vec::new();
        let outcomes =
            drive_policies(backend, args, &slices, &references, 4, &mut ids_lines).unwrap();
        (outcomes, ids_lines)
    }

    #[test]
    fn default_sweep_runs_the_construction_policy_only() {
        let args = sweep_args("", "1,2");
        let mut backend = FakeLanes::new(4, 3);
        let (outcomes, ids_lines) = sweep(&mut backend, &args);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, PolicyStatus::Pass);
        assert_eq!(outcomes[0].spec, DEFAULT_POLICY_SPEC);
        assert_eq!(
            outcomes[0].record["describe"],
            json!(LaneStagePolicy::all_per_lane().describe())
        );
        assert_eq!(outcomes[0].record["runs"].as_array().unwrap().len(), 2);
        assert!(ids_lines
            .iter()
            .all(|line| line["policy"] == json!(DEFAULT_POLICY_SPEC)));
        assert_eq!(sweep_verdict(&outcomes), Verdict::Pass);
    }

    #[test]
    fn sweep_reports_a_failing_policy_among_passing_ones() {
        let args = sweep_args(
            "--policies per-lane;per-lane:moe=once;production --tokens-sweep 4",
            "1,2,4",
        );
        let mut backend = FakeLanes::new(4, 3);
        backend.corrupt = Some(policy_of("per-lane:moe=once"));
        let (outcomes, ids_lines) = sweep(&mut backend, &args);

        // No early exit: every policy ran, the failing one in the middle.
        let statuses: Vec<_> = outcomes.iter().map(|o| o.status.clone()).collect();
        assert_eq!(
            statuses,
            vec![PolicyStatus::Pass, PolicyStatus::Fail, PolicyStatus::Pass]
        );
        assert_eq!(sweep_verdict(&outcomes), Verdict::Fail);
        assert_eq!(backend.policy, LaneStagePolicy::production());
        assert!(backend.lanes.is_empty());

        let failing = &outcomes[1];
        assert_eq!(failing.failures.len(), 1 + 2 + 4);
        assert_eq!(failing.failures[0], "K=1 prompt 0");
        assert_eq!(failing.record["status"], json!("FAIL"));
        assert_eq!(failing.record["all_equal"], json!(false));
        assert_eq!(
            failing.record["describe"],
            json!(policy_of("per-lane:moe=once").describe())
        );
        for (outcome, tokens) in outcomes.iter().zip([6, 4, 4]) {
            assert_eq!(outcome.record["tokens"], json!(tokens));
            let per_k = outcome.record["per_k"].as_array().unwrap();
            assert_eq!(per_k.len(), 3);
            assert!(per_k
                .iter()
                .all(|k| k.get("decode_wall_s").is_some() && k.get("aggregate_tok_s").is_some()));
        }
        assert_eq!(outcomes[0].record["all_equal"], json!(true));
        assert_eq!(outcomes[2].record["all_equal"], json!(true));
        // The passing sweep policy compared its 4 tokens against the reference prefix.
        assert_eq!(
            outcomes[2].record["runs"][2]["prompts"][3]["equal"],
            json!(true)
        );

        // ids.jsonl lines carry their policy and the policy's token count.
        for (spec, tokens) in [("per-lane", 6), ("per-lane:moe=once", 4), ("production", 4)] {
            let lines: Vec<_> = ids_lines
                .iter()
                .filter(|line| line["policy"] == json!(spec))
                .collect();
            assert_eq!(lines.len(), 1 + 2 + 4, "{spec}");
            assert!(lines
                .iter()
                .all(|line| line["ids"].as_array().unwrap().len() == tokens));
        }

        let summary = sweep_summary(&outcomes);
        assert!(summary.contains("[0] per-lane=PASS"), "{summary}");
        assert!(
            summary.contains("[1] per-lane:moe=once=FAIL at K=1 prompt 0"),
            "{summary}"
        );
        assert!(summary.contains("[2] production=PASS"), "{summary}");
    }

    #[test]
    fn an_errored_policy_is_recorded_and_the_sweep_continues_while_the_store_is_usable() {
        let args = sweep_args("--policies per-lane;per-lane:embed=once;production", "1,2");
        let mut backend = FakeLanes::new(4, 3);
        backend.fail = Some(policy_of("per-lane:embed=once"));
        let (outcomes, _) = sweep(&mut backend, &args);
        let labels: Vec<_> = outcomes.iter().map(|o| o.status.label()).collect();
        assert_eq!(labels, ["PASS", "ERROR", "PASS"]);
        let PolicyStatus::Error(message) = &outcomes[1].status else {
            unreachable!()
        };
        assert!(
            message.starts_with("K=1:") && message.contains("injected"),
            "{message}"
        );
        assert_eq!(outcomes[1].record["status"], json!("ERROR"));
        assert_eq!(outcomes[1].record["all_equal"], json!(false));
        assert!(outcomes[1].record["error"]
            .as_str()
            .unwrap()
            .contains("injected"));
        assert!(
            backend.lanes.is_empty(),
            "the failed run released its lanes"
        );
        assert_eq!(sweep_verdict(&outcomes), Verdict::Fail);
    }

    #[test]
    fn an_unusable_store_stops_the_sweep_and_skips_the_rest() {
        let args = sweep_args(
            "--policies per-lane;per-lane:embed=once;per-lane:moe=once;production",
            "1",
        );
        let mut backend = FakeLanes::new(4, 3);
        backend.fail = Some(policy_of("per-lane:embed=once"));
        backend.break_on_fail = true;
        let (outcomes, _) = sweep(&mut backend, &args);
        let labels: Vec<_> = outcomes.iter().map(|o| o.status.label()).collect();
        assert_eq!(labels, ["PASS", "ERROR", "SKIPPED", "SKIPPED"]);
        assert_eq!(outcomes[2].spec, "per-lane:moe=once");
        assert_eq!(outcomes[2].record["status"], json!("SKIPPED"));
        assert!(outcomes[2].record["skipped"]
            .as_str()
            .unwrap()
            .contains("unusable"));
        assert_eq!(sweep_verdict(&outcomes), Verdict::Fail);
        assert!(sweep_summary(&outcomes).contains("[3] production=SKIPPED"));
    }
}

// ----------------------------------------------------- spec schema / manifest

#[cfg(test)]
mod spec_schema_tests {
    use super::*;

    fn argv(line: &str) -> impl Iterator<Item = String> + '_ {
        line.split_whitespace().map(str::to_owned)
    }

    const BASE: &str = "--model /m/m.mq4 --max-seq 4096 --tokens 8 --out /o/x";

    fn files(n: usize) -> String {
        (0..n)
            .map(|i| format!(" --prompt-file /p/p{i}.txt"))
            .collect()
    }

    fn parse(extra: &str, n_files: usize) -> R<Args> {
        parse_args_from(argv(&format!(
            "{BASE} --ks 1{extra}{}",
            files(n_files.max(1))
        )))
    }

    #[test]
    fn ar_is_the_default_and_carries_no_spec_state() {
        let args = parse("", 1).unwrap();
        assert_eq!(args.spec, SpecMode::Ar);
        assert_eq!(
            (args.k, args.routes, args.soak_seconds, args.soak_seed),
            (None, None, None, None)
        );
        assert_eq!(args.fault_point, None);
        assert_eq!(parse(" --spec ar", 1).unwrap(), args);
        // Every pre-existing flag still parses exactly as before.
        let full = parse(
            " --pp 2 --row-budget 3 --skip-model-hash --logits --policies g0 --tokens-sweep 4 --reserve-lanes 8",
            1,
        )
        .unwrap();
        assert_eq!(
            (full.pp, full.row_budget, full.reserve_lanes),
            (2, Some(3), Some(8))
        );
        assert_eq!(full.spec, SpecMode::Ar);
    }

    #[test]
    fn mtp_defaults_and_ranges() {
        let args = parse(" --spec mtp", 1).unwrap();
        assert_eq!(args.spec, SpecMode::Mtp);
        assert_eq!((args.k, args.routes), (Some(3), Some(RoutesArg::Cycle)));
        assert_eq!(
            (args.soak_seconds, args.soak_seed, args.fault_point),
            (None, None, None)
        );
        for k in [1usize, 3, 7] {
            assert_eq!(
                parse(&format!(" --spec mtp --k {k}"), 1).unwrap().k,
                Some(k)
            );
        }
        for bad in ["0", "8", "-1", "x", "3.5"] {
            assert!(
                parse(&format!(" --spec mtp --k {bad}"), 1).is_err(),
                "k {bad}"
            );
        }
        for (text, routes) in [
            ("floor", RoutesArg::Floor),
            ("cycle", RoutesArg::Cycle),
            ("batched", RoutesArg::Batched),
        ] {
            let args = parse(&format!(" --spec mtp --routes {text}"), 1).unwrap();
            assert_eq!(args.routes, Some(routes));
            assert_eq!(routes.name(), text);
        }
        assert!(parse(" --spec mtp --routes interleaved", 1).is_err());
        assert!(parse(" --spec bogus", 1).is_err());
        assert!(parse(" --spec", 1).is_err());
    }

    #[test]
    fn mtp_only_flags_need_spec_mtp() {
        for flag in [
            "--k 3",
            "--routes cycle",
            "--soak-seconds 60",
            "--soak-seed 7",
            "--fault-point repair",
        ] {
            let name = flag.split(' ').next().unwrap();
            let error = parse(&format!(" {flag}"), 1).unwrap_err();
            assert_eq!(error, format!("{name} needs --spec mtp"), "{flag}");
            let error = parse(&format!(" --spec ar {flag}"), 1).unwrap_err();
            assert_eq!(error, format!("{name} needs --spec mtp"), "{flag}");
        }
    }

    #[test]
    fn fault_points_and_soak() {
        let names = [
            ("before-draft", FaultPoint::BeforeDraft),
            ("head-round", FaultPoint::HeadRound),
            ("target-edge", FaultPoint::TargetEdge),
            ("repair", FaultPoint::Repair),
            ("before-publish", FaultPoint::BeforePublish),
        ];
        assert_eq!(names.len(), FaultPoint::ALL.len());
        for (text, point) in names {
            let args = parse(&format!(" --spec mtp --fault-point {text}"), 1).unwrap();
            assert_eq!(args.fault_point, Some(point));
            assert_eq!(point.name(), text);
        }
        for bad in ["before_draft", "publish", "BEFORE-DRAFT", ""] {
            assert!(
                parse_args_from(
                    argv(&format!("{BASE} --ks 1 --spec mtp --fault-point"))
                        .chain([bad.to_string()])
                        .chain(argv(&files(1)))
                )
                .is_err(),
                "accepted fault point {bad:?}"
            );
        }
        let soak = parse(" --spec mtp --k 7 --routes cycle --soak-seconds 1800", 1).unwrap();
        assert_eq!(
            (soak.soak_seconds, soak.soak_seed),
            (Some(1800), Some(DEFAULT_SOAK_SEED))
        );
        let seeded = parse(" --spec mtp --soak-seconds 60 --soak-seed 99", 1).unwrap();
        assert_eq!(seeded.soak_seed, Some(99));
        assert!(parse(" --spec mtp --soak-seconds 0", 1).is_err());
        assert!(parse(" --spec mtp --soak-seed 5", 1)
            .unwrap_err()
            .contains("--soak-seconds"));
        let both = parse(" --spec mtp --soak-seconds 60 --fault-point repair", 1).unwrap_err();
        assert!(both.contains("exclusive"), "{both}");
    }

    #[test]
    fn multiple_prompt_files_are_preserved_in_order() {
        let args = parse_args_from(argv(&format!(
            "{BASE} --ks 1,8 --pp 2 --spec mtp --k 3 --routes cycle{}",
            files(8)
        )))
        .unwrap();
        assert_eq!(args.ks, vec![1, 8]);
        assert_eq!(args.prompt_files.len(), 8);
        assert_eq!(args.prompt_files[0], "/p/p0.txt");
        assert_eq!(args.prompt_files[7], "/p/p7.txt");
        let few = parse_args_from(argv(&format!("{BASE} --ks 1,8 --spec mtp{}", files(4))));
        assert!(few.unwrap_err().contains("at least 8 --prompt-file"));
    }

    #[test]
    fn expanded_argv_round_trips_to_the_same_args() {
        let cwd = Path::new("/work");
        for extra in [
            "",
            " --pp 2 --row-budget 5 --logits --skip-model-hash --tokens-sweep 3 --reserve-lanes 8",
            " --policies g0",
            " --policies per-lane;per-lane:moe=once;production",
            " --spec mtp",
            " --spec mtp --k 7 --routes batched",
            " --spec mtp --k 7 --routes cycle --soak-seconds 1800 --soak-seed 42",
            " --spec mtp --k 1 --routes floor --fault-point head-round",
        ] {
            let args = parse_args_from(
                argv(&format!("{BASE} --ks 1,2{}", files(2))).chain(
                    extra
                        .split(' ')
                        .filter(|w| !w.is_empty())
                        .map(str::to_owned),
                ),
            )
            .unwrap();
            let expanded = expanded_argv(&args, cwd);
            let reparsed = parse_args_from(expanded.iter().cloned()).unwrap();
            assert_eq!(reparsed, args, "{extra}");
        }
    }

    #[test]
    fn expanded_argv_makes_paths_absolute_and_fills_defaults() {
        let args = parse_args_from(argv(
            "--model rel/m.mq4 --max-seq 4096 --tokens 8 --out rel/out --ks 1 --prompt-file rel/a.txt",
        ))
        .unwrap();
        let line = expanded_command("/bin/oracle", &expanded_argv(&args, Path::new("/work")));
        assert!(
            line.starts_with("/bin/oracle --model /work/rel/m.mq4 "),
            "{line}"
        );
        for want in [
            "--prompt-file /work/rel/a.txt",
            "--out /work/rel/out",
            "--pp 1",
            "--spec ar",
            "--ks 1",
        ] {
            assert!(line.contains(want), "{want} not in {line}");
        }
        assert!(
            !line.contains("--policies"),
            "default policy is not spelled out"
        );
    }

    #[test]
    fn shell_words_are_quoted_only_when_needed() {
        assert_eq!(shell_quote("--model"), "--model");
        assert_eq!(shell_quote("/a/b-c_d.e,f=g:h"), "/a/b-c_d.e,f=g:h");
        assert_eq!(shell_quote("per-lane;production"), "'per-lane;production'");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        assert_eq!(shell_quote(""), "''");
    }

    fn fixture(index: usize, sha: &str) -> Value {
        json!({"index": index, "sha256": sha})
    }

    #[test]
    fn manifest_records_command_fixtures_binary_and_spec() {
        let args = parse_args_from(argv(&format!(
            "{BASE} --ks 1,2 --spec mtp --k 7 --routes cycle --soak-seconds 90{}",
            files(2)
        )))
        .unwrap();
        let env = serde_json::Map::from_iter([("HIPFIRE_TOPOLOGY_EP".to_string(), json!("2"))]);
        let manifest = build_manifest(
            &args,
            &["--model".to_string()],
            "/bin/oracle",
            "deadbeef",
            Path::new("/work"),
            vec![fixture(0, "aa"), fixture(1, "bb")],
            env,
        );
        assert_eq!(manifest["tool"], json!(TOOL));
        assert_eq!(manifest["binary"]["sha256"], json!("deadbeef"));
        assert_eq!(manifest["binary"]["path"], json!("/bin/oracle"));
        assert_eq!(manifest["model"]["sha256"], json!("pending"));
        assert_eq!(manifest["prompts"].as_array().unwrap().len(), 2);
        assert_eq!(manifest["distinct_prompts"], json!(2));
        assert_eq!(manifest["spec"]["spec"], json!("mtp"));
        assert_eq!(manifest["spec"]["k"], json!(7));
        assert_eq!(manifest["spec"]["routes"], json!("cycle"));
        assert_eq!(manifest["spec"]["soak_seconds"], json!(90));
        assert_eq!(manifest["spec"]["soak_seed"], json!(DEFAULT_SOAK_SEED));
        assert_eq!(manifest["env"]["HIPFIRE_TOPOLOGY_EP"], json!("2"));
        assert_eq!(manifest["completed"], json!(false));
        let command = manifest["command"].as_str().unwrap();
        for want in ["--spec mtp", "--k 7", "--routes cycle", "--soak-seconds 90"] {
            assert!(command.contains(want), "{want} not in {command}");
        }
        // Skipping the hash records null, never "pending".
        let skipped = parse_args_from(argv(&format!(
            "{BASE} --ks 1 --skip-model-hash{}",
            files(1)
        )))
        .unwrap();
        let manifest = build_manifest(
            &skipped,
            &[],
            "/bin/oracle",
            "x",
            Path::new("/work"),
            vec![fixture(0, "aa")],
            serde_json::Map::new(),
        );
        assert_eq!(manifest["model"]["sha256"], Value::Null);
        assert_eq!(manifest["distinct_prompts"], json!(1));
    }

    #[test]
    fn manifest_finalization_takes_the_runs_outcome() {
        let mut manifest = json!({"model": {"sha256": "pending"}, "completed": false});
        let report = json!({
            "model_sha256": "abc123",
            "completed": true,
            "verdict": "PASS",
        });
        finalize_manifest(&mut manifest, &report);
        assert_eq!(manifest["model"]["sha256"], json!("abc123"));
        assert_eq!(manifest["completed"], json!(true));
        assert_eq!(manifest["verdict"], json!("PASS"));
        // An error run without a model hash keeps the pending marker.
        let mut manifest = json!({"model": {"sha256": "pending"}, "completed": false});
        finalize_manifest(&mut manifest, &json!({"completed": false, "error": "boom"}));
        assert_eq!(manifest["model"]["sha256"], json!("pending"));
        assert_eq!(manifest["error"], json!("boom"));
        let mut absent = Value::Null;
        finalize_manifest(&mut absent, &report);
        assert!(absent.is_null());
    }

    #[test]
    fn mtp_execution_is_an_explicit_boundary_and_ar_is_not() {
        let ar = parse("", 1).unwrap();
        ensure_spec_supported(&ar).unwrap();
        // The EP greedy matrix executes; soak, fault points and PP refuse.
        ensure_spec_supported(&parse(" --spec mtp", 1).unwrap()).unwrap();
        ensure_spec_supported(&parse(" --spec mtp --k 7 --routes batched", 1).unwrap()).unwrap();
        for extra in [
            " --spec mtp --pp 2",
            " --spec mtp --k 7 --routes cycle --soak-seconds 1800",
            " --spec mtp --fault-point target-edge",
        ] {
            let error = ensure_spec_supported(&parse(extra, 1).unwrap()).unwrap_err();
            assert!(error.contains("not yet implemented by S3/S5"), "{error}");
            assert!(error.contains("not a pass"), "{error}");
            assert!(error.starts_with(TOOL), "{error}");
        }
        let soak = ensure_spec_supported(&parse(" --spec mtp --soak-seconds 5", 1).unwrap());
        assert!(soak.unwrap_err().contains("soak=5s"));
        let fault = ensure_spec_supported(&parse(" --spec mtp --fault-point repair", 1).unwrap());
        assert!(fault.unwrap_err().contains("fault-point=repair"));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{TOOL}-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn fixture_records_hash_each_prompt_file() {
        let dir = scratch("fixtures");
        std::fs::write(dir.join("a.txt"), b"abc").unwrap();
        std::fs::write(dir.join("b.txt"), b"").unwrap();
        let line = format!(
            "{BASE} --ks 2 --prompt-file {} --prompt-file {}",
            dir.join("a.txt").display(),
            dir.join("b.txt").display()
        );
        let args = parse_args_from(argv(&line)).unwrap();
        let records = fixture_records(&args, Path::new("/")).unwrap();
        assert_eq!(records[0]["bytes"], json!(3));
        assert_eq!(
            records[0]["sha256"],
            json!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            records[1]["sha256"],
            json!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        let missing = parse_args_from(argv(&format!(
            "{BASE} --ks 1 --prompt-file {}",
            dir.join("nope.txt").display()
        )))
        .unwrap();
        assert!(fixture_records(&missing, Path::new("/"))
            .unwrap_err()
            .contains("nope.txt"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `--spec mtp` through the real `run`: no model is loaded, the refusal is
    /// recorded in report.json, and manifest.json carries the expanded command,
    /// fixture hashes and binary sha.
    #[test]
    fn mtp_run_refuses_without_loading_and_leaves_receipts() {
        let dir = scratch("run");
        std::fs::write(dir.join("p0.txt"), b"abc").unwrap();
        std::fs::write(dir.join("p1.txt"), b"abd").unwrap();
        let out = dir.join("out");
        let line = format!(
            "--model {m} --max-seq 4096 --tokens 8 --ks 2 --pp 2 --spec mtp --k 3 --routes cycle --out {o} --prompt-file {a} --prompt-file {b}",
            m = dir.join("m.mq4").display(),
            o = out.display(),
            a = dir.join("p0.txt").display(),
            b = dir.join("p1.txt").display(),
        );
        let args = parse_args_from(argv(&line)).unwrap();
        let error = match run(args) {
            Err(error) => error,
            Ok(_) => panic!("--spec mtp must not produce a verdict"),
        };
        assert!(error.contains("not yet implemented by S3/S5"), "{error}");
        let read = |name: &str| -> Value {
            serde_json::from_slice(&std::fs::read(out.join(name)).unwrap()).unwrap()
        };
        let report = read("report.json");
        assert_eq!(report["completed"], json!(false));
        assert!(report["error"].as_str().unwrap().contains("S3/S5"));
        assert_eq!(report["manifest"], json!("manifest.json"));
        assert_eq!(report["spec"]["spec"], json!("mtp"));
        assert!(report.get("verdict").is_none(), "a refusal has no verdict");
        assert!(out.join("ids.jsonl").exists());
        let manifest = read("manifest.json");
        let prompts = manifest["prompts"].as_array().unwrap();
        assert_eq!(prompts.len(), 2);
        assert_eq!(
            prompts[0]["sha256"],
            json!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_ne!(prompts[0]["sha256"], prompts[1]["sha256"]);
        assert_eq!(manifest["distinct_prompts"], json!(2));
        assert_eq!(manifest["binary"]["sha256"].as_str().unwrap().len(), 64);
        assert_eq!(manifest["completed"], json!(false));
        assert!(manifest["error"].as_str().unwrap().contains("S3/S5"));
        let command = manifest["command"].as_str().unwrap();
        assert!(
            command.contains("--spec mtp --k 3 --routes cycle"),
            "{command}"
        );
        // The recorded command re-parses to the invocation it records.
        let expanded: Vec<String> = manifest["expanded_argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|word| word.as_str().unwrap().to_string())
            .collect();
        let reparsed = parse_args_from(expanded.into_iter()).unwrap();
        assert_eq!(
            (reparsed.spec, reparsed.k, reparsed.pp),
            (SpecMode::Mtp, Some(3), 2)
        );
        // A second run into the same DIR is refused, as before.
        let again = parse_args_from(argv(&line)).unwrap();
        assert!(run(again).unwrap_err().contains("already exists"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
