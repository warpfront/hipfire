// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#![recursion_limit = "256"]

//! qwen4_wide_verify — real-model oracle for wide MTP verify windows.
//!
//! For each requested window length `n` this runs the PRODUCTION verify
//! transaction (`Qwen4Bundle::verify_block_rows` behind `SpecTarget::verify_block`:
//! snapshot, row-armed `spec_forward_rows` with `prefill_scope = false`) over a
//! block of `n` tokens and compares every row's logits BITWISE with `n`
//! sequential `Qwen4Bundle::forward_token` steps from the same resident state.
//! It then checks the post-commit state (`SpecTarget::commit_prefix`) against
//! the autoregressive state after the same number of tokens, part by part
//! (SHA-256 of every `SessionState::snapshot_parts` part), and times both
//! routes. `Qwen4Bundle::forward_chunk` is NOT the verify route (it sets
//! `prefill_scope = true`) and is not used for the comparison.
//!
//! Usage:
//!   qwen4_wide_verify --model M.hfq (--prompt-file P.txt | --prompt-files a.txt,b.txt)
//!                     --rows 1,2,...,32 --out OUT.json
//!                     [--iters 5] [--n-ctx 8192] [--block greedy|prompt]
//!                     [--accepts ends|all|none] [--arms base,wide16,wide32] [--trace]
//!                     [--hot-budget-mib N] [--hotset SIDECAR]
//!
//! `--rows` is a comma list; `a-b` ranges are accepted (`1-32`).
//! `--prompt-files` runs every prompt in one model load (`--prompt-file` is the
//! single-element alias); each prompt gets its own prefill, S0, block and AR
//! reference. The JSON is per prompt x arm x n.
//! `--arms` (default `base`): each arm sets the rdna-compute runtime verify-rows
//! override (`base` = unset, the arch table / env decides; `wideN` = N) and
//! runs the whole per-n suite in the same process. The override only takes
//! effect on gfx1151/gfx1201; the JSON records the effective verify rows.
//! `--block greedy` (default) is the AR greedy continuation of the prompt
//! (block[0] is the argmax of the prefill's final row, block[i+1] the argmax
//! after block[i]); `--block prompt` is the last max(rows) prompt tokens
//! (copy-like).
//! `--accepts ends` checks accept_len in {0, n-1}; `all` checks 0..n; `none`
//! skips every commit_prefix arm (no re-prefills; use it at deep contexts).
//! `--trace` runs per prompt x arm x n an AR window then a verify window, no
//! digests, each preceded by delimiter launches of `logit_softcap_f32` (never
//! launched by the Qwen4 forward): AR = 2 back-to-back delimiters then n
//! `forward_token`s, verify = 1 delimiter then the verify. `trace_pattern` in
//! the JSON documents it so a rocprofv3 kernel trace can be cut and labelled.
//! Every record carries the execution mode (`replay.is_recording()`,
//! `graphs.capture_mode`) read just before its windows.
//!
//! A fixed `--hot-budget-mib` uses the same frozen expert-tier partition with
//! `HIPFIRE_OFFLOAD_EXEC=cpu` (requires D2 armed) or `pcie` (requires D2 unarmed).
//! Both receipts expose its actual `partition_digest`; `split_digest` remains
//! the real D2 identity, never a substitute PCIe partition identity. Missing or
//! invalid hotsets, zero/overflowing budgets and passback fail the pinned load.
//!
//! Rewind: AR passes and the timed arms rewind with the model-owned snapshot
//! ticket (`Qwen4Bundle::snapshot` / `restore`; a verify is rolled back with its
//! own ticket). Accept arms end in a committed state with no ticket, so those
//! re-prefill the prompt; every re-prefill is checked to reproduce the
//! post-prefill state digest bit for bit, and the first ticket rewind of each
//! kind is checked the same way.
//!
//! Exit status is non-zero if any n <= 8 is not fully bitwise, any state digest
//! mismatches at n <= 8, or the run itself fails. n >= 9 mismatches are
//! reported, never fatal.
//!
//! # Cost-curve profile (`--cost-curve`) and strict qualification (`--strict-qualify`)
//!
//! The default load of every profile is the PRODUCTION serve load
//! (`hipfire_loader::load_qwen4_rank`: split base + arch PLE rows sidecar +
//! trunk overlay + expert hot set / placement + native MTP head) with the
//! artifact identity receipts of `tests/common/hw_identity.rs` in
//! `identity.production`. `--hot-budget-mib` keeps the old manual loader as a
//! DIAGNOSTIC fixed partition (labelled `diagnostic-manual-hfq`, not the
//! shipping profile). `HIPFIRE_STATE_QUANT` is the GDN state request and
//! `HIPFIRE_REPLAY_BACKEND` the launch backend (`hip` = eager).
//!
//! # Preload-plan probe (`--preload-plan-only`)
//!
//! Source-planned allocation probe: opens the HFQ headers/index metadata
//! only, initialises the GPU bare (no model load, no weight fulfillment, no
//! source symmetry scan, no sidecar verification/hash, no worker startup,
//! no capture64, no identity/hash publication) and calls the loader's
//! metadata-only `qwen4_preload_resource_plan` with the ORDINARY loader ask
//! (resolved backend / bases, runtime `speculation.mtp`/`mtp_k`, explicit
//! `--n-ctx` honored, planning default the finite 544 bench; B resolves the
//! effective MTP policy exactly as the single-device load does). Output
//! schema `qwen4_wide_verify.preload_plan.v1` says `planning_only: true`,
//! `source_planned: true`, `unmeasured: true`, `nonqualification: true`,
//! `evidence_kind: source_planned_unmeasured_nonqualification`,
//! `complete: false`, `qualification: nonqualification`: never production
//! RAW, never a fit, never a qualification. Run-shape flags
//! (`--prompt-file(s)`, `--rows`, `--arms`, `--block`, `--accepts`,
//! `--iters`) are accepted but unused in this mode. `--resource-plan-only`
//! is unchanged (post-load RAW via `production::load`).
//! `--cost-curve` (timing / pricing only; `--rows` defaults to 1-16,32,64,
//! `--arms base` = the production verify route, never the 32-row `wideN`
//! clamp): per width `n` (K = n-1 drafts + the seed) the first repeat of
//! every boundary is the warmup and is reported apart. Boundaries: `verify_ms`
//! (snapshot + row-armed forward + device sync), `rollback_ms` (the verify
//! ticket's restore, outside the verify timer), `commit_ms` per accept_len in
//! {0, n-1} (`commit_prefix` + sync; `--commit on`, `--commit-iters N`
//! re-prefills), `ar_ms` (n plain `forward_token`s: a BASELINE only, never the
//! AR-routed window total) and, with `--baseline-prefill on`, the same-S0
//! prefill route (`forward_chunk`). Widths above the arm's qualified row-exact
//! rows are `timing-only-beyond-qualified` (no identity claim); widths inside
//! them keep the hard bitwise gate. `--head-depths LIST|off` (default
//! 1-16,31,63) times exactly `depth` production head steps with their restore;
//! `--prod-windows N --prod-ks LIST` (default K = each requested row minus
//! one, including K0) forces REAL production windows, independently
//! re-prefilled to the same S0. K0 is AR-routed, positive K is native MTP,
//! and OnCopy also requests real offered n-gram K. Lookup, confirmation,
//! verification, acceptance, repair, commit and sync belong to the total.
//! The first repeat is discarded as warmup. `--actual-windows-out PATH`
//! (default OUT.actual_windows.json) writes actual_windows.v1 with actual
//! widths/accepted prefixes, `native_draft_margins` (per actual native
//! draft for Mtp, number where the head-step margin was finite/nonnegative
//! else null, `[]` for other sources; never acceptance/prose, never priced)
//! and explicit unsupported/missing receipts.
//! Isolated `windows` expose `component_subtotal_ms`, never a full price
//! (`window_ms: null`, `complete: false`). `--cost-mode OffPool|OnCopy`
//! asserts that `HIPFIRE_MTP_NGRAM_COPY` resolves to that mode; never mix modes.
//! N-gram `variant: unconfirmed` means established/no confirmation required,
//! not a failed check; `requested_confirmation` is the planned price key.
//! `--deadline-secs N` (<= 1750, loads included) bounds the run; every
//! requested width / component that is not measured lands in `missing`,
//! `complete` is false and the exit status non-zero.
//!
//! `--strict-qualify` (`--rows` defaults to 1-64, `--accepts ends|all`): every
//! requested width is fatal for the largest explicitly requested arm; smaller
//! explicit arms (`wide8` beside `wide64`) diagnose byte/state mismatches only.
//! Source errors and missing coverage remain failures. Both generic and native
//! seed-inclusive commits, including singleton full acceptance, check state and
//! subsequent n-row verify and AR continuations against the same reference.
//! Abort checks restored S0 and both continuations too. Every width records
//! effective bundle caps and capture arming; missing native paths never skip.
//! Strict arithmetic-policy receipts require `--features reference-parity,lab,kernel-timing`.
//! Injected launch failures after snapshot, both before kernel work and after
//! one successful kernel launch, must restore temporary arithmetic/row scopes,
//! S0 and ticket reuse before the same continuation checks. `host_window_ms`
//! is synchronized, uninstrumented end-to-end verify time, including snapshot
//! and uploads; it is not a kernel-only interval. Prefill-route timing is
//! diagnostic; primary arm comparisons use the identical verify transaction.
//!
//! Build: `cargo build --release --features reference-parity,lab,kernel-timing -p
//! hipfire-arch-qwen4 --example qwen4_wide_verify`. `reference-parity` is
//! required to compile; `lab` binds actual arithmetic identity and exact planned
//! device row-capture bytes. `kernel-timing` is required for complete strict
//! evidence, not for the legacy oracle. `gpu_span_ms` is the SUM (never the
//! mean) of the real per-launch eager HIP event spans after the production
//! snapshot, prepared once with `lab_prepare_kernel_timing` and released
//! with `lab_close_kernel_timing` (close failure propagates before any
//! publication); `host_window_ms` is the separate uninstrumented synced
//! end-to-end verify total. Timing uses separate instrumented
//! warmup/repeats; it excludes copies, JIT, downloads, rollback and CPU gaps.
//! PM4, AQL and graph paths are not covered and cannot claim this evidence.

use hip_bridge::{HipError, HipResult};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::wide_verify_oracle as oracle;
use hipfire_config::memory::OffloadExec;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::llama::argmax;
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, PleRowsOrigin, ResolvePolicy};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig, SpecScratch, SpecTarget};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// The identity helpers of the production identity harness (executable /
// source / artifact sha256, GPU arch / UUID / PCI, graph flags, git state).
#[path = "../tests/common/hw_identity.rs"]
mod hw_identity;
// CPU-only cost-curve / strict-qualification logic and the production load.
#[path = "qwen4_wide_cost/production.rs"]
mod production;
#[path = "qwen4_wide_cost/profile.rs"]
mod profile;

use hipfire_arch_qwen4::mtp_spec::{qwen4_ngram_mod_config, Qwen4MtpDrafter};
use profile::{CostMode, Deadline, ProdWindow, Profile, Timed, WidthCost};

const USAGE: &str = "usage: qwen4_wide_verify --model M.hfq (--prompt-file P.txt | --prompt-files a.txt,b.txt) --rows 1,2,...,32 --out OUT.json [--iters 5] [--n-ctx 8192] [--block greedy|prompt] [--accepts ends|all|none] [--arms base,wide16,wide32] [--hot-budget-mib N (diagnostic manual loader, not the shipping profile)] [--hotset SIDECAR] [--row-capture on|off] [--trace] [--cost-curve | --strict-qualify] [--resource-plan-only | --preload-plan-only] [--deadline-secs N] [--max-k N] [--head-depths LIST|off] [--commit on|off] [--commit-iters N] [--baseline-prefill on|off] [--cost-mode OffPool|OnCopy] [--prod-ks LIST] [--prod-windows N] [--actual-windows-out PATH]";

/// The delimiter kernel of `--trace`: a one-block elementwise kernel
/// (`kernels/src/logit_softcap.hip`) launched only by the Gemma/Glimmer
/// logit-softcap paths, never by any Qwen4 forward.
const DELIMITER_KERNEL: &str = "logit_softcap_f32";

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockMode {
    Greedy,
    Prompt,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Accepts {
    /// accept_len in {0, n-1}.
    Ends,
    /// accept_len in 0..n: seed-inclusive consumed prefix lengths 1..=n.
    All,
    /// No commit_prefix arms (no re-prefills).
    None,
}

impl Accepts {
    fn name(self) -> &'static str {
        match self {
            Accepts::Ends => "ends",
            Accepts::All => "all",
            Accepts::None => "none",
        }
    }
}

/// A verify-width arm: `base` leaves the runtime override unset (the arch
/// table / env decides), `wideN` sets it to N (clamped by rdna-compute).
#[derive(Clone)]
struct Arm {
    name: String,
    rows: Option<usize>,
}

impl Arm {
    fn base() -> Self {
        Arm {
            name: "base".to_string(),
            rows: None,
        }
    }
}

fn parse_arms(spec: &str) -> Result<Vec<Arm>, String> {
    let mut arms: Vec<Arm> = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let arm = if part == "base" {
            Arm::base()
        } else if let Some(n) = part
            .strip_prefix("wide")
            .and_then(|n| n.parse::<usize>().ok())
        {
            Arm {
                name: part.to_string(),
                rows: Some(n),
            }
        } else {
            return Err(format!("bad --arms entry '{part}' (base|wideN)"));
        };
        if arms.iter().any(|a| a.name == arm.name) {
            return Err(format!("duplicate --arms entry '{part}'"));
        }
        arms.push(arm);
    }
    if arms.is_empty() {
        return Err("--arms is empty".into());
    }
    Ok(arms)
}

struct Args {
    model: PathBuf,
    prompt_files: Vec<PathBuf>,
    rows: Vec<usize>,
    out: PathBuf,
    iters: usize,
    n_ctx: usize,
    block: BlockMode,
    accepts: Accepts,
    arms: Vec<Arm>,
    trace: bool,
    hot_budget_mib: Option<u64>,
    hotset: Option<PathBuf>,
    row_capture: bool,
    /// `--cost-curve` / `--strict-qualify` (default: the legacy oracle).
    profile: Profile,
    /// Wall-clock budget of the whole process, loads included.
    deadline_secs: u64,
    /// Native MTP reserve of the production load (`Qwen4RankShared::max_k`).
    max_k: usize,
    head_depths: Vec<usize>,
    commit: bool,
    commit_iters: usize,
    baseline_prefill: bool,
    cost_mode: Option<CostMode>,
    prod_ks: Vec<usize>,
    prod_windows: usize,
    actual_windows_out: Option<PathBuf>,
    /// Load-only allocation-plan probe (no prompts, no windows, no timing).
    resource_plan_only: bool,
    /// Source-planned preload probe: metadata-only, before any production
    /// load. Distinct from `resource_plan_only` (post-load RAW).
    preload_plan_only: bool,
    /// Whether `--n-ctx` was explicit (`--preload-plan-only` defaults to the
    /// finite 544 bench planning context when absent).
    n_ctx_given: bool,
    /// Whether `--max-k` was explicit. Post-load planning keeps the resolved
    /// default; `--preload-plan-only` without one plans runtime auto (`None`).
    max_k_given: bool,
}

fn parse_rows(spec: &str) -> Result<Vec<usize>, String> {
    let mut rows = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("bad --rows range '{part}'"))?;
            let b: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("bad --rows range '{part}'"))?;
            if a == 0 || b < a {
                return Err(format!("bad --rows range '{part}'"));
            }
            rows.extend(a..=b);
        } else {
            let n: usize = part
                .parse()
                .map_err(|_| format!("bad --rows entry '{part}'"))?;
            if n == 0 {
                return Err("--rows entries must be >= 1".into());
            }
            rows.insert(n);
        }
    }
    if rows.is_empty() {
        return Err("--rows is empty".into());
    }
    Ok(rows.into_iter().collect())
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    parse_args_from(&argv)
}

fn parse_args_from(argv: &[String]) -> Result<Args, String> {
    let mut model = None;
    let mut prompt_files: Option<Vec<PathBuf>> = None;
    let mut rows = None;
    let mut out = None;
    let mut iters = 5usize;
    let mut n_ctx = 8192usize;
    let mut block = BlockMode::Greedy;
    let mut accepts = Accepts::Ends;
    let mut accepts_given = false;
    let mut arms = vec![Arm::base()];
    let mut trace = false;
    let mut hot_budget_mib = None;
    let mut hotset = None;
    let mut row_capture = true;
    let mut cost_curve = false;
    let mut strict = false;
    let mut resource_plan_only = false;
    let mut preload_plan_only = false;
    let mut n_ctx_given = false;
    let mut max_k_given = false;
    let mut deadline_secs: Option<u64> = None;
    let mut max_k: Option<usize> = None;
    let mut head_depths: Option<Vec<usize>> = None;
    let mut commit: Option<bool> = None;
    let mut commit_iters: Option<usize> = None;
    let mut baseline_prefill: Option<bool> = None;
    let mut cost_mode: Option<CostMode> = None;
    let mut prod_ks: Option<Vec<usize>> = None;
    let mut prod_windows: Option<usize> = None;
    let mut actual_windows_out = None;
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--trace" => {
                trace = true;
                i += 1;
                continue;
            }
            "--cost-curve" => {
                cost_curve = true;
                i += 1;
                continue;
            }
            "--strict-qualify" => {
                strict = true;
                i += 1;
                continue;
            }
            "--resource-plan-only" => {
                resource_plan_only = true;
                i += 1;
                continue;
            }
            "--preload-plan-only" => {
                preload_plan_only = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?
            .as_str();
        match flag {
            "--model" => model = Some(PathBuf::from(value)),
            "--hot-budget-mib" => {
                let mib: u64 = value
                    .parse()
                    .map_err(|_| format!("bad --hot-budget-mib '{value}'"))?;
                if mib == 0 || mib.checked_mul(1 << 20).is_none() {
                    return Err("--hot-budget-mib must be nonzero and fit in bytes".into());
                }
                hot_budget_mib = Some(mib);
            }
            "--hotset" => hotset = Some(PathBuf::from(value)),
            "--row-capture" => {
                row_capture = match value {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--row-capture takes on|off".into()),
                }
            }
            // `--prompt-file` is the single-element alias of `--prompt-files`.
            "--prompt-file" | "--prompt-files" => {
                prompt_files = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(PathBuf::from)
                        .collect(),
                )
            }
            "--rows" => rows = Some(parse_rows(value)?),
            "--out" => out = Some(PathBuf::from(value)),
            "--iters" => {
                iters = value
                    .parse()
                    .map_err(|_| format!("bad --iters '{value}'"))?
            }
            "--n-ctx" => {
                n_ctx = value
                    .parse()
                    .map_err(|_| format!("bad --n-ctx '{value}'"))?;
                n_ctx_given = true;
            }
            "--block" => {
                block = match value {
                    "greedy" => BlockMode::Greedy,
                    "prompt" => BlockMode::Prompt,
                    other => return Err(format!("--block takes greedy|prompt, got '{other}'")),
                }
            }
            "--accepts" => {
                accepts = match value {
                    "ends" => Accepts::Ends,
                    "all" => Accepts::All,
                    "none" => Accepts::None,
                    other => return Err(format!("--accepts takes ends|all|none, got '{other}'")),
                };
                accepts_given = true;
            }
            "--arms" => arms = parse_arms(value)?,
            "--deadline-secs" => deadline_secs = Some(profile::parse_deadline_secs(value)?),
            "--max-k" => {
                let k: usize = value
                    .parse()
                    .map_err(|_| format!("bad --max-k '{value}'"))?;
                if k == 0 {
                    return Err("--max-k must be >= 1".into());
                }
                max_k = Some(k);
                max_k_given = true;
            }
            "--head-depths" => {
                head_depths = Some(if value == "off" {
                    Vec::new()
                } else {
                    profile::parse_list("--head-depths", value, 1)?
                })
            }
            "--commit" => commit = Some(profile::parse_on_off("--commit", value)?),
            "--commit-iters" => {
                let n: usize = value
                    .parse()
                    .map_err(|_| format!("bad --commit-iters '{value}'"))?;
                if n == 0 {
                    return Err("--commit-iters must be >= 1".into());
                }
                commit_iters = Some(n);
            }
            "--baseline-prefill" => {
                baseline_prefill = Some(profile::parse_on_off("--baseline-prefill", value)?)
            }
            "--cost-mode" => cost_mode = Some(CostMode::parse(value)?),
            "--prod-ks" => prod_ks = Some(profile::parse_list("--prod-ks", value, 0)?),
            "--actual-windows-out" => actual_windows_out = Some(PathBuf::from(value)),
            "--prod-windows" => {
                prod_windows = Some(
                    value
                        .parse()
                        .map_err(|_| format!("bad --prod-windows '{value}'"))?,
                )
            }
            other => return Err(format!("unknown argument '{other}'\n{USAGE}")),
        }
        i += 2;
    }
    if iters == 0 {
        return Err("--iters must be >= 1".into());
    }
    if hotset.is_some() && hot_budget_mib.is_none() {
        return Err("--hotset requires a nonzero --hot-budget-mib".into());
    }
    if resource_plan_only && (cost_curve || strict || trace) {
        return Err(
            "--resource-plan-only is load-only: not with --cost-curve / --strict-qualify / --trace"
                .into(),
        );
    }
    if resource_plan_only && hot_budget_mib.is_some() {
        return Err(
            "--resource-plan-only requires the production serve load, not --hot-budget-mib".into(),
        );
    }
    if resource_plan_only
        && (head_depths.is_some()
            || commit.is_some()
            || commit_iters.is_some()
            || cost_mode.is_some()
            || prod_ks.is_some()
            || prod_windows.is_some()
            || actual_windows_out.is_some()
            || baseline_prefill.is_some())
    {
        return Err("--head-depths/--commit/--commit-iters/--cost-mode/--prod-ks/--prod-windows/--actual-windows-out/--baseline-prefill do not apply with --resource-plan-only".into());
    }
    if preload_plan_only && resource_plan_only {
        return Err("--preload-plan-only and --resource-plan-only are exclusive".into());
    }
    if preload_plan_only && (cost_curve || strict || trace) {
        return Err(
            "--preload-plan-only is planning-only: not with --cost-curve / --strict-qualify / --trace"
                .into(),
        );
    }
    if preload_plan_only && (hot_budget_mib.is_some() || hotset.is_some()) {
        return Err(
            "--preload-plan-only uses the ordinary loader config, not --hot-budget-mib / --hotset"
                .into(),
        );
    }
    if preload_plan_only
        && (head_depths.is_some()
            || commit.is_some()
            || commit_iters.is_some()
            || cost_mode.is_some()
            || prod_ks.is_some()
            || prod_windows.is_some()
            || actual_windows_out.is_some()
            || baseline_prefill.is_some())
    {
        return Err("--head-depths/--commit/--commit-iters/--cost-mode/--prod-ks/--prod-windows/--actual-windows-out/--baseline-prefill do not apply with --preload-plan-only".into());
    }
    if cost_curve && strict {
        return Err("--cost-curve and --strict-qualify are exclusive".into());
    }
    let profile = if cost_curve {
        Profile::Cost
    } else if strict {
        Profile::Strict
    } else {
        Profile::Oracle
    };
    if profile != Profile::Oracle && trace {
        return Err(
            "--trace is a legacy oracle mode: not with --cost-curve / --strict-qualify".into(),
        );
    }
    if profile != Profile::Cost
        && (head_depths.is_some()
            || commit.is_some()
            || commit_iters.is_some()
            || cost_mode.is_some()
            || prod_ks.is_some()
            || prod_windows.is_some()
            || actual_windows_out.is_some())
    {
        return Err("--head-depths/--commit/--commit-iters/--cost-mode/--prod-ks/--prod-windows apply only with --cost-curve".into());
    }
    if profile == Profile::Oracle && baseline_prefill.is_some() {
        return Err("--baseline-prefill applies only with --cost-curve / --strict-qualify".into());
    }
    match profile {
        Profile::Cost => {
            if accepts_given && accepts != Accepts::None {
                return Err("--cost-curve measures the commit boundary itself (--commit); --accepts is not used".into());
            }
            accepts = Accepts::None;
        }
        Profile::Strict => {
            if accepts == Accepts::None {
                return Err("--strict-qualify needs --accepts ends|all: none would skip the state comparison".into());
            }
        }
        Profile::Oracle => {}
    }
    if rows.is_none() {
        rows = match profile {
            Profile::Cost => Some(parse_rows(profile::DEFAULT_COST_ROWS)?),
            Profile::Strict => Some(parse_rows("1-64")?),
            Profile::Oracle if resource_plan_only => Some(vec![64]),
            // Run-shape placeholder only, never a capture claim: the planning
            // branch runs before any load and never arms rows.
            Profile::Oracle if preload_plan_only => Some(vec![1]),
            Profile::Oracle => None,
        };
    }
    let max_k = max_k.unwrap_or(if profile == Profile::Oracle {
        profile::ORACLE_MAX_K
    } else {
        profile::PROFILE_MAX_K
    });
    let prod_ks = prod_ks.unwrap_or_else(|| {
        if profile == Profile::Cost {
            rows.as_ref().unwrap().iter().map(|n| n - 1).collect()
        } else {
            vec![3.min(max_k)]
        }
    });
    if let Some(k) = prod_ks.iter().find(|&&k| k > max_k) {
        return Err(format!("--prod-ks {k} exceeds --max-k {max_k}"));
    }
    let head_depths = match head_depths {
        Some(depths) => depths,
        None if profile == Profile::Cost => {
            profile::parse_list("--head-depths", profile::DEFAULT_HEAD_DEPTHS, 1)?
        }
        None => Vec::new(),
    };
    if profile != Profile::Oracle {
        if max_k > profile::PROFILE_MAX_K {
            return Err("--max-k exceeds the physical 63-draft reserve".into());
        }
        if rows
            .as_ref()
            .unwrap()
            .iter()
            .any(|&n| n > WIDE_VERIFY_MAX_ROWS)
        {
            return Err(
                "--cost-curve/--strict-qualify rows exceed the physical verify limit".into(),
            );
        }
        if head_depths.iter().any(|&depth| depth > max_k) {
            return Err("--head-depths exceeds --max-k physical reserve".into());
        }
    }
    let prompt_files = if preload_plan_only || resource_plan_only {
        // Load-only planning has no prompt inputs, even when supplied by a
        // shared command line; do not enter prompt-required validation.
        Vec::new()
    } else {
        let prompt_files =
            prompt_files.ok_or_else(|| format!("--prompt-file(s) is required\n{USAGE}"))?;
        if prompt_files.is_empty() {
            return Err("--prompt-files is empty".into());
        }
        prompt_files
    };
    // The preload probe plans the finite 544 bench unless the caller says
    // otherwise explicitly. This stays an ordinary resolved context: it must
    // never widen into the oracle `n_ctx` default on the production path.
    let n_ctx = if preload_plan_only && !n_ctx_given {
        544usize
    } else {
        n_ctx
    };
    Ok(Args {
        model: model.ok_or_else(|| format!("--model is required\n{USAGE}"))?,
        prompt_files,
        rows: rows.ok_or_else(|| format!("--rows is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        iters,
        n_ctx,
        block,
        accepts,
        arms,
        trace,
        hot_budget_mib,
        hotset,
        row_capture,
        profile,
        deadline_secs: deadline_secs.unwrap_or(if profile == Profile::Oracle {
            u64::from(u32::MAX)
        } else {
            profile::DEFAULT_DEADLINE_SECS
        }),
        max_k,
        head_depths,
        commit: commit.unwrap_or(profile == Profile::Cost),
        commit_iters: commit_iters.unwrap_or(3),
        baseline_prefill: baseline_prefill.unwrap_or(profile == Profile::Strict),
        cost_mode,
        prod_ks,
        prod_windows: prod_windows.unwrap_or(if profile == Profile::Cost { 24 } else { 0 }),
        actual_windows_out,
        resource_plan_only,
        preload_plan_only,
        n_ctx_given,
        max_k_given,
    })
}

/// Admission is separate from GPU setup so CPU-only tests cover both controls.
fn admit_pinned_control(
    exec: OffloadExec,
    uma: bool,
    cpu_experts_enabled: bool,
) -> Result<(), String> {
    if uma {
        return Err("fixed hotset requires a discrete GPU".into());
    }
    match exec {
        OffloadExec::Pcie => Ok(()),
        OffloadExec::Cpu if cpu_experts_enabled => Ok(()),
        OffloadExec::Cpu => Err("fixed CPU hotset requires CPU experts enabled".into()),
        OffloadExec::Passback => {
            Err("fixed hotset requires HIPFIRE_OFFLOAD_EXEC=cpu or pcie (not passback)".into())
        }
    }
}

fn admit_pinned_split(
    exec: OffloadExec,
    has_split_layers: bool,
    has_split_digest: bool,
) -> Result<(), String> {
    match exec {
        OffloadExec::Cpu if has_split_layers && has_split_digest => Ok(()),
        OffloadExec::Cpu => Err("requested D2 hotset split did not arm".into()),
        OffloadExec::Pcie if !has_split_layers && !has_split_digest => Ok(()),
        OffloadExec::Pcie => Err("pinned PCIe control unexpectedly armed a D2 split".into()),
        OffloadExec::Passback => Err("passback is not a pinned control".into()),
    }
}

#[cfg(test)]
mod pinned_control_tests {
    use super::*;

    fn cli(extra: &[&str]) -> Result<Args, String> {
        let argv: Vec<String> = [
            "--model",
            "model.hfq",
            "--prompt-file",
            "prompt.txt",
            "--rows",
            "1,2,4,8",
            "--out",
            "receipt.json",
        ]
        .into_iter()
        .chain(extra.iter().copied())
        .map(str::to_owned)
        .collect();
        parse_args_from(&argv)
    }

    #[test]
    fn pinned_cli_keeps_budget_requirements_fail_closed() {
        assert!(cli(&[]).unwrap().hot_budget_mib.is_none());
        assert!(cli(&["--hotset", "bound.expert-hotset"]).is_err());
        for budget in ["0", "-1", "bad", "17592186044416", "18446744073709551616"] {
            assert!(cli(&["--hot-budget-mib", budget]).is_err(), "{budget}");
        }
        assert!(cli(&["--hot-budget-mib"]).is_err());
        let args = cli(&[
            "--hot-budget-mib",
            "4096",
            "--hotset",
            "bound.expert-hotset",
        ])
        .unwrap();
        assert_eq!(args.hot_budget_mib, Some(4096));
        assert_eq!(
            args.hotset.as_deref(),
            Some(Path::new("bound.expert-hotset"))
        );
    }

    #[test]
    fn pinned_controls_use_the_config_parser_and_require_real_split_status() {
        use hipfire_config::memory::parse_offload_exec;

        let pcie = parse_offload_exec(Some(" PCIE "));
        let cpu = parse_offload_exec(Some(" CPU "));
        assert_eq!(pcie, OffloadExec::Pcie);
        assert_eq!(cpu, OffloadExec::Cpu);
        assert!(admit_pinned_control(pcie, false, false).is_ok());
        assert!(admit_pinned_control(cpu, false, true).is_ok());
        assert!(admit_pinned_control(cpu, false, false).is_err());
        for exec in [pcie, cpu] {
            assert!(admit_pinned_control(exec, true, true).is_err());
            for layers in [false, true] {
                for digest in [false, true] {
                    let expected = match exec {
                        OffloadExec::Cpu => layers && digest,
                        OffloadExec::Pcie => !layers && !digest,
                        OffloadExec::Passback => unreachable!(),
                    };
                    assert_eq!(admit_pinned_split(exec, layers, digest).is_ok(), expected);
                }
            }
        }
        let passback = parse_offload_exec(Some("passback"));
        assert!(admit_pinned_control(passback, false, true).is_err());
        assert!(admit_pinned_split(passback, false, false).is_err());
    }

    #[test]
    fn pinned_hotset_preflight_rejects_disabled_empty_and_missing_paths() {
        use hipfire_arch_qwen4::expert_hotset as hotsets;

        let env = hotsets::HotsetEnv {
            gpu_arch: "gfx1201",
            unified_memory: false,
            expert_parallel: false,
            explicit_vram_layers: false,
            source: hotsets::HotsetSource::Aggregate,
        };
        let missing = std::env::temp_dir()
            .join(format!("qwen4-wide-verify-{}", std::process::id()))
            .join("missing.expert-hotset");
        for policy in [
            hotsets::HotsetPolicy::Off,
            hotsets::HotsetPolicy::Explicit(PathBuf::new()),
            hotsets::HotsetPolicy::Explicit(missing),
        ] {
            assert!(hotsets::preflight(&policy, Path::new("model.hfq"), &env).is_err());
        }
        assert!(hotsets::require_experts(0).is_err());
    }
}

// ---------------------------------------------------------------------------
// Product loader (qwen4_kld `Model::load`, minus the PLE hot store)
// ---------------------------------------------------------------------------

struct LoadInfo {
    arch: String,
    ple_line: String,
    ple_note: Option<String>,
    ple_rows_source: String,
    overlay_applied: bool,
    model_bytes: u64,
    uma: bool,
    expert_vram_layers: Option<usize>,
    offload_exec: OffloadExec,
    partition_digest: Option<String>,
    split_armed: bool,
    loader: &'static str,
    production: Value,
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
    tokenizer: Tokenizer,
    info: LoadInfo,
    /// Loader's cached canonical measured-load identity (`None` on the
    /// diagnostic manual path, which never claims production provenance).
    measured_identity:
        Option<Arc<hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity>>,
    /// Loader's frozen checked allocation-plan snapshot (`None` until the
    /// actual planner census/owners/host-check union lands; never faked).
    measured_snapshot:
        Option<Arc<hipfire_arch_qwen4::measured_resources::Qwen4LoadResourceSnapshot>>,
}

impl Model {
    /// Same range-loaded path as the serve loader (`hipfire-loader` carriers):
    /// PLE rows sidecar (never deriving a missing q4), admission, trunk overlay,
    /// the loader's forward scratch plan, UMA mmap drop, optional
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, `attach_forward`. No MTP head, no
    /// session cache.
    fn load(
        path: &Path,
        n_ctx: usize,
        hot_budget_mib: Option<u64>,
        hotset: Option<&Path>,
    ) -> Result<Self, String> {
        let model_bytes = std::fs::metadata(path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        // The GPU arch drives `memory.ple_format = auto`; initialising it
        // allocates no model memory, so admission still precedes every upload.
        let mut gpu = Gpu::init().map_err(es)?;
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
            ResolvePolicy {
                derive_missing_q4: false,
                progress: None,
            },
        )
        .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
        let mut receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
            .map_err(|e| format!("qwen4 admission: {e}"))?;
        // `<model>.trunk-overlay` (QT44 trunk tensors + the A4 default), bound
        // to this base while it is open and applied by `attach_forward`.
        let trunk_overlay =
            hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        let overlay_applied = trunk_overlay.is_some();
        // A rank-order table is served through its store (no shard
        // descriptors); every other source reads through 128 descriptors.
        let rank_table = receipt.ple_rows.rank_table().is_some();
        let ple_descriptors = if rank_table {
            None
        } else {
            Some(
                receipt
                    .ple_row_descriptors(&hfq)
                    .map_err(|e| format!("qwen4 PLE rows: {e}"))?,
            )
        };
        let (ple_rows_path, ple_rows_bytes) = receipt
            .ple_rows_source(&hfq)
            .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
        let (rows_encoding, rows_total) = match (&receipt.ple_rows, hfq.ple_rows()) {
            (hipfire_arch_qwen4::Qwen4PleRows::Sidecar { encoding, .. }, Some(rows)) => {
                (encoding.dtype_name().to_string(), rows.header().data_bytes)
            }
            _ => {
                let descriptors = ple_descriptors.as_deref().unwrap_or_default();
                (
                    descriptors
                        .first()
                        .map_or_else(String::new, |descriptor| descriptor.dtype.clone()),
                    descriptors.iter().map(|descriptor| descriptor.length).sum(),
                )
            }
        };
        let ple_line = format!(
            "ple rows: {} from {} ({rows_encoding}, {} order, {:.1} GB)",
            choice.format.name(),
            match &choice.origin {
                PleRowsOrigin::Embedded => "embedded model file".to_string(),
                PleRowsOrigin::Sidecar(sidecar) => sidecar.display().to_string(),
            },
            if rank_table { "rank" } else { "row" },
            rows_total as f64 / 1e9
        );
        eprintln!("qwen4_wide_verify: {ple_line}");
        if let Some(note) = &choice.note {
            eprintln!("qwen4_wide_verify: ple rows: {note}");
        }
        // The loader's forward scratch plan, decided from the actual payload
        // while the mapping is readable.
        let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
            &gpu,
            &hfq,
            &receipt.config,
            &receipt.manifest,
        )
        .map_err(es)?;
        let offload_exec = hipfire_config::memory::offload_exec();
        // Use the same bound sidecar, slab plan and tier owners as cpu_split_parity.
        // The fixed budget freezes one partition for all AR/verify/commit passes.
        let tier_lease = if let Some(mib) = hot_budget_mib {
            use hipfire_arch_qwen4::expert_hotset as hotsets;
            admit_pinned_control(
                offload_exec,
                gpu.is_uma(),
                hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled(),
            )?;
            if hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()?.is_some() {
                return Err("fixed hotset conflicts with HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS".into());
            }
            let policy = match hotset {
                Some(path) => hotsets::HotsetPolicy::Explicit(path.to_path_buf()),
                None => hotsets::HotsetPolicy::from_env()?,
            };
            let env = hotsets::HotsetEnv {
                gpu_arch: &gpu.arch,
                unified_memory: false,
                expert_parallel: false,
                explicit_vram_layers: false,
                source: match &receipt.manifest.identity.recipe {
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
                        hotsets::HotsetSource::Aggregate
                    }
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
                        hotsets::HotsetSource::SlabArena
                    }
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
                        hotsets::HotsetSource::StockFp8
                    }
                    _ => {
                        hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into())
                    }
                },
            };
            let sidecar_path = hotsets::preflight(&policy, path, &env).map_err(|skip| {
                skip.log_line(&policy.sidecar_path(path))
                    .unwrap_or_else(|| skip.to_string())
            })?;
            let base = hotsets::HotsetBase::from_hfq(&hfq)?;
            let sidecar = hotsets::HotsetSidecar::load_bound(
                &sidecar_path,
                receipt.config.num_hidden_layers,
                receipt.config.num_experts,
                receipt.config.num_experts_per_tok,
                &base,
            )
            .map_err(|skip| skip.to_string())?;
            receipt.manifest.check_expert_tiers().map_err(es)?;
            let slabs = hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(
                &hfq,
                &receipt.config,
                &receipt.manifest,
            )?;
            let plan = hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs(
                &slabs,
                sidecar.rank(),
                mib * (1 << 20),
            )
            .map_err(es)?;
            hotsets::require_experts(plan.device_experts()).map_err(|skip| skip.to_string())?;
            plan.check_executable(&gpu.arch).map_err(es)?;
            hipfire_arch_qwen4::expert_residency::place_routed_experts(
                &mut receipt.manifest.weights,
                receipt.config.num_hidden_layers,
            );
            receipt.manifest = receipt.manifest.with_expert_tiers().map_err(es)?;
            {
                let keep: BTreeSet<_> = receipt.manifest.resident_keys().collect();
                receipt
                    .placements
                    .retain(|p| keep.contains(&(p.name.as_str(), p.layer)));
            }
            let lease = hipfire_arch_qwen4::expert_tier_owners::materialize(
                &mut gpu,
                std::sync::Arc::new(plan),
                &|offset, len| {
                    let offset = usize::try_from(offset)
                        .map_err(|_| "hotset source offset overflow".to_string())?;
                    let len = usize::try_from(len)
                        .map_err(|_| "hotset source length overflow".to_string())?;
                    hfq.data_range(offset, len).ok_or_else(|| {
                        format!("source span {offset}+{len} is outside the artifact mapping")
                    })
                },
            )
            .map_err(|e| format!("expert tier materialize: {e}"))?;
            lease
                .bind_hotset_sidecar_sha256(sidecar.sha256())
                .map_err(es)?;
            eprintln!(
                "qwen4_wide_verify: hotset {} sha256 {} fixed budget {mib} MiB",
                sidecar_path.display(),
                sidecar.sha256()
            );
            Some(lease)
        } else {
            None
        };
        // The lease retains the actual frozen plan's logical digest in either
        // execution mode; PCIe has no D2 split digest to use as its identity.
        let partition_digest = tier_lease.as_ref().map(|lease| {
            let mut hex = String::with_capacity(64);
            for byte in lease.logical_digest() {
                write!(&mut hex, "{byte:02x}").expect("write digest to String");
            }
            hex
        });
        let uma = gpu.is_uma();
        if uma {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
        let source = HfqModelSource::from_hfq(hfq);
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
        // and host-maps the rest (a slab artifact larger than one card).
        let mut expert_vram_layers = None;
        let weights = {
            let mut weights = receipt.manifest.weights.clone();
            match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
                None => {}
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                    if gpu.is_uma() {
                        return Err("host-mapped experts are only for discrete GPUs".into());
                    }
                    let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                        &mut weights,
                        layers,
                    );
                    expert_vram_layers = Some(layers);
                    eprintln!(
                        "qwen4_wide_verify: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                    );
                }
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                    return Err(
                        "qwen4_wide_verify takes a fixed expert layer count, not auto".into(),
                    );
                }
            }
            weights
        };
        let transaction = fulfill_manifest_from_payloads(
            &weights,
            &mesh,
            receipt.config.num_hidden_layers,
            &mut gpu,
            expected,
            |entry| {
                if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                    &source,
                    &receipt.manifest.identity.recipe,
                    entry,
                ) {
                    return payload;
                }
                // The store rejects a host-mapped entry as a range: borrow it
                // from the resident source bytes, as the product loader does.
                if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped
                {
                    let (info, bytes) = source
                        .tensor_data(&entry.name)
                        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                    return Ok(SourcePayload::Borrowed { info, bytes });
                }
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
        let vocab = receipt.config.vocab_size;
        let state_format = hipfire_arch_qwen4::resolve_state_format(
            &hipfire_runtime::config::get().kv_mode,
            // Qwen3.5's DeltaNet `state_quant` knob: auto (q8) or fp32.
            &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
            &gpu,
            &receipt.config,
        )?;
        let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
        let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
            receipt.config,
            &receipt.manifest.identity.recipe,
            transaction,
            &receipt.placements,
            &mut gpu,
            n_ctx,
            receipt.ple,
            receipt.ple_rows,
            state_format,
            backend,
            tier_lease,
        )
        .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
        bundle.attach_npu_trunk_identity(npu_identity);
        if let Some(overlay) = trunk_overlay {
            bundle
                .attach_trunk_overlay(overlay)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        }
        bundle
            .attach_forward(&mut gpu, n_ctx, scratch_mode)
            .map_err(|e| format!("qwen4 forward setup: {e}"))?;
        let has_split_layers = !bundle.split_layers().is_empty();
        let has_split_digest = bundle.split_digest().is_some();
        let split_armed = has_split_layers && has_split_digest;
        if hot_budget_mib.is_some() {
            admit_pinned_split(offload_exec, has_split_layers, has_split_digest)?;
        }
        eprintln!("qwen4_wide_verify: offload_exec={offload_exec} split_armed={split_armed}");
        if let Some(digest) = &partition_digest {
            eprintln!("qwen4_wide_verify: partition_digest={digest}");
        }
        eprintln!(
            "qwen4_wide_verify: split_digest {:?}",
            bundle.split_digest()
        );
        eprintln!(
            "qwen4_wide_verify: loaded {} ({:.2} GB) on {} (vocab {vocab}), trunk overlay {}",
            path.display(),
            model_bytes as f64 / 1e9,
            gpu.arch,
            if overlay_applied { "applied" } else { "absent" }
        );
        let info = LoadInfo {
            arch: gpu.arch.to_string(),
            ple_line,
            ple_note: choice.note.clone(),
            ple_rows_source: format!("{} ({ple_rows_bytes} bytes)", ple_rows_path.display()),
            overlay_applied,
            model_bytes,
            uma,
            expert_vram_layers,
            offload_exec,
            partition_digest,
            split_armed,
            loader: "diagnostic-manual-hfq (NOT the shipping profile)",
            production: Value::Null,
        };
        Ok(Self {
            gpu,
            bundle,
            vocab,
            tokenizer,
            info,
            measured_identity: None,
            measured_snapshot: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

type StateDigest = Vec<(String, String)>;

/// Keys of the parts whose digests differ (a missing part counts).
fn digest_mismatches(want: &StateDigest, got: &StateDigest) -> Vec<String> {
    let mut keys = Vec::new();
    let want_map: BTreeMap<&str, &str> =
        want.iter().map(|(k, d)| (k.as_str(), d.as_str())).collect();
    let got_map: BTreeMap<&str, &str> = got.iter().map(|(k, d)| (k.as_str(), d.as_str())).collect();
    for (key, _) in want.iter().chain(got.iter()) {
        if want_map.get(key.as_str()) != got_map.get(key.as_str())
            && !keys.iter().any(|k: &String| k == key)
        {
            keys.push(key.clone());
        }
    }
    keys
}

#[derive(Default)]
struct RowCmp {
    bitwise_rows: usize,
    first_mismatch: Option<usize>,
    max_abs_diff: f64,
    mismatching: Vec<Value>,
    argmax_agree: usize,
}

/// Compare `n` rows of `got` against `reference` bit for bit.
fn compare_rows(reference: &[f32], got: &[f32], vocab: usize, n: usize) -> RowCmp {
    let mut cmp = RowCmp::default();
    for r in 0..n {
        let a = &reference[r * vocab..(r + 1) * vocab];
        let g = &got[r * vocab..(r + 1) * vocab];
        let mut differing = 0usize;
        let mut row_max = 0f64;
        for (x, y) in a.iter().zip(g) {
            if x.to_bits() != y.to_bits() {
                differing += 1;
                let d = (f64::from(*x) - f64::from(*y)).abs();
                let d = if d.is_nan() { f64::INFINITY } else { d };
                if d > row_max {
                    row_max = d;
                }
            }
        }
        if differing == 0 {
            cmp.bitwise_rows += 1;
        } else {
            if cmp.first_mismatch.is_none() {
                cmp.first_mismatch = Some(r);
            }
            cmp.mismatching.push(json!({
                "row": r,
                "differing_elements": differing,
                "max_abs_diff": row_max,
            }));
        }
        if row_max > cmp.max_abs_diff {
            cmp.max_abs_diff = row_max;
        }
        if argmax(a) == argmax(g) {
            cmp.argmax_agree += 1;
        }
    }
    cmp
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

struct ArPass {
    tokens: Vec<u32>,
    /// Row-major `count * vocab`: row r after consuming `tokens[..=r]`.
    logits: Vec<f32>,
    /// Live-state digest after `j` tokens, for each requested `j`.
    digests: BTreeMap<usize, StateDigest>,
}

struct NOutcome {
    record: Value,
    /// Fully bitwise, every accept digest equal, no error.
    clean: bool,
    line: String,
    /// Verify median of this width (commit medians are filled by the commit pass).
    cost: Option<WidthCost>,
}

/// How a verify's accepted prefix is committed: the generic
/// `SpecTarget::commit_prefix` or the native retain rollback of the
/// production batched window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CommitPath {
    Generic,
    NativeRetain,
}

impl CommitPath {
    fn name(self) -> &'static str {
        match self {
            CommitPath::Generic => "generic_commit_prefix",
            CommitPath::NativeRetain => "native_retain",
        }
    }

    fn tag(self) -> &'static str {
        match self {
            CommitPath::Generic => "",
            CommitPath::NativeRetain => "r",
        }
    }
}

/// Profile-dependent run configuration of the rig.
#[derive(Clone, Copy)]
struct RunCfg {
    profile: Profile,
    commit: bool,
    commit_iters: usize,
    baseline_prefill: bool,
    deadline: Deadline,
}

/// One arm's results of a prompt, held until the head phase can join them.
struct ArmRun {
    arm: Arm,
    effective: usize,
    caps: Value,
    reprefills: usize,
    checked_kinds: Vec<&'static str>,
    records: Vec<Value>,
    costs: Vec<WidthCost>,
    cells: Vec<(usize, bool)>,
}

struct Rig {
    m: Model,
    prompt: Vec<u32>,
    /// Final-row logits of the product prefill.
    prompt_logits: GpuTensor,
    /// One-row logits of `forward_token`.
    row: GpuTensor,
    seed_logits: Vec<f32>,
    /// Post-prefill state digest (`None` under `--trace`).
    d0: Option<StateDigest>,
    /// The resident state is no longer the post-prefill state.
    dirty: bool,
    p0: usize,
    reprefills: usize,
    rewind_checked: BTreeSet<&'static str>,
    iters: usize,
    cfg: RunCfg,
    /// Qualified row-exact verify rows of the current arm.
    qualified: usize,
    requested_arm_rows: Option<usize>,
    strict_candidate_rows: usize,
    /// Kernels seen loaded so far (route witness).
    kernel_seen: BTreeSet<String>,
    /// Actual bound launches of the current strict width (set by the
    /// collector transaction, consumed by the route fingerprint record).
    pending_launch_witnesses: Vec<StrictLaunchWitness>,
    /// Required cells or evidence that the actual run could not complete.
    missing: Vec<Value>,
}

fn begin_arm_rewind_checks(checked: &mut BTreeSet<&'static str>) {
    // AR was checked before the arm loop; verify rewind is specific to each arm.
    checked.retain(|kind| *kind == "ar_ticket");
}

fn strict_arm_is_diagnostic(requested: Option<usize>, candidate: usize) -> bool {
    requested.is_some_and(|rows| rows < candidate)
}

fn strict_required_rows(requested: Option<usize>, candidate: usize, n: usize) -> usize {
    if strict_arm_is_diagnostic(requested, candidate) {
        requested.unwrap_or(n)
    } else {
        requested.unwrap_or(n).max(n)
    }
}

fn verify_reference_receipt(bundle: &Qwen4Bundle, n: usize) -> Value {
    if n == 1 {
        return json!("singleton-unbound");
    }
    #[cfg(feature = "lab")]
    {
        json!(oracle::dev_last_verify_reference(bundle).map(|mode| format!("{mode:?}")))
    }
    #[cfg(not(feature = "lab"))]
    {
        let _ = bundle;
        Value::Null
    }
}
/// One actual bound launch witnessed inside a strict verify transaction:
/// the qualified symbol, its loaded owner module, the admitted container SHA
/// (via the frozen `lab_kernel_identity` owner proof) and, for in-scope
/// qualified-family launches, the per-arch raw device SHA (via the new
/// `lab_kernel_device_identity` sharing the same owner proof; cached per
/// artifact/arch, hashed once). Out-of-scope helper/AR launches keep the
/// owner/container record truthfully with a null device SHA — extraction is
/// never attempted for them, so an irrelevant compressed/unextractable
/// helper image cannot fail the binding.
#[derive(Clone)]
struct StrictLaunchWitness {
    symbol: String,
    module: String,
    object_sha256: String,
    device_sha256: Option<String>,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
}

/// Immutable G1 raw `.co` image pins: the schema-verified `manifest.json`
/// `artifacts[role == "image"]` entries (`{name, path, sha256}`) of the
/// existing gate-runner contract (`ROW_EXACT_WIDE_RUN_DIR`; GateRunner
/// verifies each file SHA before staging). Digests must be 64-char lowercase
/// hex; missing/unreadable/empty pins fail strict closed, never fake complete.
fn strict_g1_image_pins() -> Result<BTreeMap<String, String>, String> {
    let run = std::env::var("ROW_EXACT_WIDE_RUN_DIR")
        .map_err(|_| "strict rows: ROW_EXACT_WIDE_RUN_DIR is not set".to_string())?;
    let text = std::fs::read_to_string(Path::new(&run).join("manifest.json"))
        .map_err(|e| format!("strict rows: cannot read runner manifest.json ({e})"))?;
    let manifest: Value =
        serde_json::from_str(&text).map_err(|e| format!("strict rows: bad manifest.json ({e})"))?;
    let artifacts = manifest
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or("strict rows: manifest.json has no artifacts array")?;
    let mut pins = BTreeMap::new();
    for entry in artifacts {
        if entry.get("role").and_then(Value::as_str) != Some("image") {
            continue;
        }
        let name = entry.get("name").and_then(Value::as_str).unwrap_or("?");
        let sha = entry
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!(
                "strict rows: image pin {name:?} has no 64-hex digest"
            ));
        }
        // One digest per distinct raw image; the name is a lookup hint only.
        pins.entry(sha.clone()).or_insert_with(|| name.to_string());
    }
    if pins.is_empty() {
        return Err("strict rows: runner manifest.json holds no role=image raw .co pins".into());
    }
    Ok(pins)
}

/// Check every actual in-scope launch witness against the original G1 raw
/// image pins. Classification uses the frozen registry/entrypoint
/// declarations (planner + dense wide families incl. K1 and the real legacy
/// GU/down original + fixed-8 exports), never a module-label-only filter:
/// unknown/helper/AR launches stay truthfully outside wide-binding scope,
/// while a missing expected pin for an actually used qualified family is a
/// strict failure. `requested`/`candidate` are the admitted arm context from
/// the Rig: a diagnostic arm never demands a wide tile at n > 8, while a
/// candidate arm rejects an actually used non-tiled legacy stage.
fn check_strict_binding(
    n: usize,
    gpu: &Gpu,
    witnesses: &[StrictLaunchWitness],
    requested: Option<usize>,
    candidate: usize,
) -> Result<Value, String> {
    let pins = strict_g1_image_pins()?;
    let arch = gpu.arch.clone();
    let cpu_owned = hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled();
    let mut in_scope: Vec<&StrictLaunchWitness> = Vec::new();
    let mut out_of_scope = 0usize;
    for w in witnesses {
        if strict_wide_symbol(&w.symbol) {
            in_scope.push(w);
        } else {
            out_of_scope += 1;
        }
    }
    let mut matched = 0usize;
    let mut mismatched: Vec<Value> = Vec::new();
    for w in &in_scope {
        match w.device_sha256.as_deref() {
            Some(device) => {
                let digest = device.to_ascii_lowercase();
                if digest.len() == 64
                    && digest.bytes().all(|b| b.is_ascii_hexdigit())
                    && pins.contains_key(&digest)
                {
                    matched += 1;
                } else {
                    mismatched.push(json!({
                        "symbol": w.symbol,
                        "module": w.module,
                        "container_sha256": w.object_sha256,
                        "device_sha256": device,
                    }));
                }
            }
            None => {
                mismatched.push(json!({
                    "symbol": w.symbol,
                    "module": w.module,
                    "container_sha256": w.object_sha256,
                    "device_sha256": Value::Null,
                }));
            }
        }
    }
    // Candidate widths above 8 rows must show an actually reached wide
    // witness, not a configured width or load census: no fabricated
    // all_rows64, no claimed-but-unobserved family. The admitted arm context
    // decides, never logical n alone: a diagnostic arm (the wide8 baseline
    // beside a wide64 candidate) stays exempt at n > 8, so the baseline can
    // compare honestly at all logical n 1..=64. The CPU-experts flag only
    // permits an absent GPU K1 (genuine CPU-owned experts doing the MoE
    // work): actually used GPU legacy/dense stages and an applicable
    // wide-tile hit are still required. An entirely empty binding never
    // passes on GPU.
    let diagnostic = strict_arm_is_diagnostic(requested, candidate);
    let wide_reached = in_scope.iter().any(|w| strict_is_wide_tile(&w.symbol));
    let k1_used = in_scope.iter().any(|w| strict_is_k1_symbol(&w.symbol));
    let legacy_gate_up_used = in_scope.iter().any(|w| strict_is_legacy_gate_up(&w.symbol));
    let legacy_down_used = in_scope.iter().any(|w| strict_is_legacy_down(&w.symbol));
    let legacy_original_used = in_scope
        .iter()
        .any(|w| strict_is_legacy_original(&w.symbol));
    let legacy_t8_witnessed = in_scope.iter().any(|w| strict_is_legacy_t8(&w.symbol));
    let dense_used = in_scope
        .iter()
        .any(|w| !strict_is_k1_symbol(&w.symbol) && !w.symbol.contains("slab_union_plan"));
    let gpu_family_used = k1_used || legacy_gate_up_used || legacy_down_used || dense_used;
    let wide_required = n > 8 && !diagnostic && (!cpu_owned || gpu_family_used);
    if !mismatched.is_empty() {
        return Err(format!(
            "strict rows {n}: {} actual wide launch(es) not bound to a G1 raw image pin: {}",
            mismatched.len(),
            mismatched
                .iter()
                .map(|m| m["symbol"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(","),
        ));
    }
    if in_scope.is_empty() {
        return Err(format!(
            "strict rows {n}: no actual in-scope launches witnessed ({out_of_scope} helper/AR out of scope)"
        ));
    }
    if wide_required && !wide_reached {
        return Err(format!(
            "strict rows {n}: no actual wide-tile launch witnessed ({} in-scope, {out_of_scope} helper/AR out of scope)",
            in_scope.len(),
        ));
    }
    // A candidate arm at n > 8 must be actually tiled: an actually used
    // non-tiled legacy GU/down stage (scalar/HIP fallback shape) never
    // completes a wide64 proof, even with the CPU-experts flag set.
    if !diagnostic && n > 8 && legacy_original_used {
        return Err(format!(
            "strict rows {n}: actually used non-tiled legacy GU/down at candidate width ({} in-scope): fallback never completes a wide proof",
            in_scope.len(),
        ));
    }
    if dense_used && in_scope.len() != matched {
        return Err(format!(
            "strict rows {n}: binding incomplete ({} in-scope actual launches, {matched} matched)",
            in_scope.len(),
        ));
    }
    let complete = in_scope.len() == matched && (!wide_required || wide_reached);
    Ok(json!({
        "n": n,
        "arch": arch,
        "cpu_owned_experts": cpu_owned,
        "diagnostic_arm": diagnostic,
        "requested_arm_rows": requested,
        "candidate_rows": candidate,
        "k1_actually_used": k1_used,
        "dense_actually_used": dense_used,
        "legacy_gate_up_actually_used": legacy_gate_up_used,
        "legacy_down_actually_used": legacy_down_used,
        "legacy_t8_witnessed": legacy_t8_witnessed,
        "legacy_original_witnessed": legacy_original_used,
        "expected_image_pins": pins.len(),
        "in_scope_launches": in_scope.len(),
        "out_of_scope_launches": out_of_scope,
        "matched": matched,
        "wide_tile_witnessed": wide_reached,
        "complete": complete,
    }))
}

/// Real legacy GU/down original + fixed-8 exports alongside the planner/K1
/// narrow+rowtile and dense fixed-t8 rowtile families, per the frozen
/// registry/entrypoint declarations. Suffix/prefix matching on the real
/// symbol only; module labels never decide scope. The new `_t8_pm_<arch>`
/// twin names carry neither `rows` nor `moe_gemv`, so they are listed
/// exactly: a broad module filter would wrongly ignore them.
fn strict_wide_symbol(symbol: &str) -> bool {
    // Exact frozen entrypoint names: the K1 narrow symbols (slab_moe.rs
    // MQ_I4_MOE_GEMV_SYMBOLS / CN / C1 / E8P tables), the K1 rowtile symbols
    // (MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS / CN_ROWTILE / gfx1100+1151/E8P _t8
    // spellings), the slab rowtile planner, and the dense fixed-t8 rowtiles
    // (gemv.rs RowTileKind::symbol + lane_rows_pm.rs rowtiles + gfx1100 Q8
    // table). Exact match, or the narrow base with its _c16/_c32 diagnostic
    // suffix (gfx1151 legacy image). No bare `_t8`-contains assumption.
    const IN_SCOPE: [&str; 69] = [
        "mq_i4_moe_gemv_mq3i4_r128_down",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_mq4sym_r128_down",
        "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
        "mq_i4_moe_gemv_cn_mq3i4_r128_down",
        "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_cn_mq4sym_r128_down",
        "mq_i4_moe_gemv_c1_r256_gate_up",
        "mq_i4_moe_gemv_c1_r128_down",
        "e8p_i4_moe_gemv_r256_gate_up",
        "e8p_i4_moe_gemv_r128_down",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1100",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1151",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
        "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100",
        "mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq3i4_r128_down_t8",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq4sym_r128_down_t8",
        "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
        "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_t8",
        "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r256_gate_up_t8",
        "e8p_i4_moe_gemv_r128_down_t8",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151_t8",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1151_t8",
        "qwen4_slab_union_plan",
        "qwen4_slab_union_plan_t8_pm_gfx1151",
        "qwen4_slab_union_plan_t8_pm_gfx1201",
        "qwen4_slab_union_plan_t8_pm_gfx1100",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1151",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1201",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1100",
        "gemv_mq6g256v2_rows_t8_pm_gfx1151",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1151",
        "gemv_mq6g256v2_rows_t8_pm_gfx1201",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201",
        "gemv_mq6g256v2_rows_t8_pm_gfx1100",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1100",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1151",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1151",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1100",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1100",
        "gemv_q8_0_k8_rows_t8_pm_gfx1151",
        "gemv_q8_0_k320_rows_t8_pm_gfx1151",
        "gemv_q8_0_k2560_rows_t8_pm_gfx1151",
        "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1151",
        "gemv_q8_0_wide_rows_t8_pm_gfx1151",
        "gemv_q8_0_k8_rows_t8_pm_gfx1100",
        "gemv_q8_0_k320_rows_t8_pm_gfx1100",
        "gemv_q8_0_k2560_rows_t8_pm_gfx1100",
        "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1100",
        "gemv_q8_0_wide_rows_t8_pm_gfx1100",
        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201",
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151",
    ];
    if IN_SCOPE.contains(&symbol) {
        return true;
    }
    // gfx1151 legacy diagnostic suffixes of the K1 narrow bases on the
    // unchanged reference image (slab_moe.rs: narrow + _c16/_c32).
    const LEGACY_BASES: [&str; 4] = [
        "mq_i4_moe_gemv_mq3i4_r256_gate_up",
        "mq_i4_moe_gemv_mq3i4_r128_down",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_mq4sym_r128_down",
    ];
    for base in LEGACY_BASES {
        if symbol == format!("{base}_c16") || symbol == format!("{base}_c32") {
            return true;
        }
    }
    false
}

/// Whether an in-scope symbol is a real legacy GU/down stage (original
/// aggregate indexed exports or their fixed-eight-token `_t8_pm_<arch>`
/// twins). Exact frozen spellings only.
fn strict_is_legacy_original(symbol: &str) -> bool {
    matches!(
        symbol,
        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched"
            | "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched"
            | "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"
    )
}
/// Whether an in-scope symbol is a legacy fixed-eight-token twin (the
/// candidate fixed-8 proof): exact `_t8_pm_<arch>` export spellings only.
fn strict_is_legacy_t8(symbol: &str) -> bool {
    matches!(
        symbol,
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201"
            | "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151"
            | "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201"
            | "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151"
    )
}
/// Whether an in-scope symbol is a legacy gate/up stage (original or twin).
fn strict_is_legacy_gate_up(symbol: &str) -> bool {
    matches!(
        symbol,
        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched"
            | "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched"
            | "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201"
            | "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151"
    )
}
/// Whether an in-scope symbol is a legacy down stage (original or twin).
fn strict_is_legacy_down(symbol: &str) -> bool {
    matches!(
        symbol,
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"
            | "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201"
            | "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151"
    )
}
/// Source-ground: `slab_moe.rs` K1 tables (`MQ_I4_MOE_GEMV_SYMBOLS`,
/// `MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS`, `MQ_I4_MOE_GEMV_CN_SYMBOLS`,
/// `MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS`, `MQ_I4_MOE_GEMV_C1_SYMBOLS`,
/// `E8P_I4_MOE_GEMV_SYMBOLS` + gfx1100/1151 `_t8` rowtile spellings).
fn strict_is_k1_symbol(symbol: &str) -> bool {
    symbol.contains("mq_i4_moe_gemv")
        || symbol.contains("mq3i4")
        || symbol.contains("mq4sym")
        || symbol.contains("e8p_i4_moe_gemv")
}

/// Whether an in-scope symbol is a fixed-eight-row wide tile (the n>8
/// proof): exact `_t8` entrypoint spellings only.
fn strict_is_wide_tile(symbol: &str) -> bool {
    const WIDE_TILES: [&str; 41] = [
        "mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq3i4_r128_down_t8",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq4sym_r128_down_t8",
        "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
        "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_t8",
        "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r256_gate_up_t8",
        "e8p_i4_moe_gemv_r128_down_t8",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8",
        "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151_t8",
        "e8p_i4_moe_gemv_r128_down_pm_gfx1151_t8",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1151",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1201",
        "gemv_mq4g256v2_rows_x4_t8_pm_gfx1100",
        "gemv_mq6g256v2_rows_t8_pm_gfx1151",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1151",
        "gemv_mq6g256v2_rows_t8_pm_gfx1201",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201",
        "gemv_mq6g256v2_rows_t8_pm_gfx1100",
        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1100",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1151",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1151",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201",
        "gemv_bf16_xf32_rows_x4_t8_pm_gfx1100",
        "gemv_bf16_xf32_k4_rows_t8_pm_gfx1100",
        "gemv_q8_0_k8_rows_t8_pm_gfx1151",
        "gemv_q8_0_k320_rows_t8_pm_gfx1151",
        "gemv_q8_0_k2560_rows_t8_pm_gfx1151",
        "gemv_q8_0_k2560_rows_x2_t8_pm_gfx1151",
        "gemv_q8_0_wide_rows_t8_pm_gfx1151",
        "gemv_q8_0_wide_rows_t8_pm_gfx1100",
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201",
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151",
    ];
    WIDE_TILES.contains(&symbol)
}

impl Rig {
    fn sync(&mut self) -> Result<(), String> {
        self.m.gpu.hip.device_synchronize().map_err(es)
    }

    fn digest(&mut self) -> Result<StateDigest, String> {
        oracle::dev_state_digest(&mut self.m.bundle, &mut self.m.gpu)
    }

    /// Start a prompt: product prefill, S0 digest (unless `--trace`).
    fn begin_prompt(&mut self, prompt: Vec<u32>, want_digest: bool) -> Result<(), String> {
        self.p0 = prompt.len();
        self.prompt = prompt;
        self.d0 = None;
        self.dirty = false;
        self.reprefills = 0;
        self.rewind_checked.clear();
        let started = Instant::now();
        self.prefill()?;
        eprintln!(
            "qwen4_wide_verify: prefilled {} prompt tokens in {:.2} s",
            self.p0,
            started.elapsed().as_secs_f64()
        );
        if want_digest {
            self.d0 = Some(self.digest()?);
        }
        Ok(())
    }

    /// Select a verify-width arm: set the runtime override (`None` = table/env)
    /// and return the verify rows the arch now resolves to.
    fn begin_arm(&mut self, arm: &Arm) -> usize {
        rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(arm.rows);
        begin_arm_rewind_checks(&mut self.rewind_checked);
        self.requested_arm_rows = arm.rows;
        let effective = oracle::dev_effective_row_exact_rows(&self.m.bundle, &self.m.gpu);
        self.qualified = effective;
        effective
    }

    /// Launch mode just before a window: both flags must be false for an eager
    /// run (a recording replay or a hipGraph capture would change the route).
    fn exec_mode(&self) -> Value {
        let recording = self.m.gpu.replay.is_recording();
        let capture = self.m.gpu.graphs.capture_mode;
        json!({
            "replay_recording": recording,
            "graph_capture_mode": capture,
            "eager": !recording && !capture,
        })
    }

    /// The product prefill (`prefill_final`: reset, cold chunk schedule, final
    /// row only), then the seed logits.
    fn prefill(&mut self) -> Result<(), String> {
        self.m
            .bundle
            .prefill_final(&mut self.m.gpu, &self.prompt, 0, &self.prompt_logits)
            .map_err(|e| format!("prefill: {e}"))?;
        self.sync()?;
        self.seed_logits = self.m.gpu.download_f32(&self.prompt_logits).map_err(es)?;
        if self.seed_logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        if oracle::dev_position(&self.m.bundle) != self.p0 {
            return Err(format!(
                "prefill ended at {}, expected {}",
                oracle::dev_position(&self.m.bundle),
                self.p0
            ));
        }
        Ok(())
    }

    /// Back at the post-prefill state S0 by re-prefilling, checked against the
    /// post-prefill digest.
    fn reprefill(&mut self) -> Result<(), String> {
        self.prefill()?;
        self.reprefills += 1;
        if let Some(d0) = self.d0.clone() {
            let now = self.digest()?;
            let bad = digest_mismatches(&d0, &now);
            if !bad.is_empty() {
                return Err(format!(
                    "re-prefill did not reproduce the post-prefill state (parts {bad:?}); rewind by re-prefill is not faithful"
                ));
            }
        }
        Ok(())
    }

    fn ensure_s0(&mut self) -> Result<(), String> {
        if self.dirty {
            self.reprefill()?;
            self.dirty = false;
        }
        Ok(())
    }

    /// After a ticket rewind: the first of each `kind` must reproduce S0.
    fn check_rewind(&mut self, kind: &'static str) -> Result<(), String> {
        if self.d0.is_none() || self.rewind_checked.contains(kind) {
            return Ok(());
        }
        let d0 = self.d0.clone().expect("checked above");
        let now = self.digest()?;
        let bad = digest_mismatches(&d0, &now);
        if !bad.is_empty() {
            self.dirty = true;
            return Err(format!(
                "{kind} rewind did not reproduce the post-prefill state (parts {bad:?})"
            ));
        }
        self.rewind_checked.insert(kind);
        Ok(())
    }

    /// `count` sequential `forward_token` steps from S0 under a snapshot
    /// ticket: fixed tokens, or (`fixed = None`) the greedy continuation from
    /// the prefill's argmax. Rewinds with the ticket.
    fn ar_pass(
        &mut self,
        fixed: Option<&[u32]>,
        count: usize,
        digest_counts: &BTreeSet<usize>,
    ) -> Result<ArPass, String> {
        self.ensure_s0()?;
        let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
        let result = self.ar_pass_steps(fixed, count, digest_counts);
        let restored = self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es);
        match (result, restored) {
            (Ok(pass), Ok(())) => {
                self.check_rewind("ar_ticket")?;
                Ok(pass)
            }
            (Err(error), restored) => {
                self.dirty = true;
                Err(match restored {
                    Ok(()) => error,
                    Err(restore) => format!("{error}; restore failed: {restore}"),
                })
            }
            (Ok(_), Err(restore)) => {
                self.dirty = true;
                Err(format!("AR pass restore failed: {restore}"))
            }
        }
    }

    fn ar_pass_steps(
        &mut self,
        fixed: Option<&[u32]>,
        count: usize,
        digest_counts: &BTreeSet<usize>,
    ) -> Result<ArPass, String> {
        let vocab = self.m.vocab;
        let mut tokens: Vec<u32> = match fixed {
            Some(fixed) => fixed.to_vec(),
            None => vec![argmax(&self.seed_logits)],
        };
        let mut logits = Vec::with_capacity(count * vocab);
        let mut digests = BTreeMap::new();
        for r in 0..count {
            self.m
                .bundle
                .forward_token(&mut self.m.gpu, tokens[r], &self.row, None)
                .map_err(|e| format!("AR forward_token row {r}: {e}"))?;
            let row = self.m.gpu.download_f32(&self.row).map_err(es)?;
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite AR logits at row {r}"));
            }
            if fixed.is_none() && r + 1 < count {
                tokens.push(argmax(&row));
            }
            logits.extend_from_slice(&row);
            if digest_counts.contains(&(r + 1)) {
                digests.insert(r + 1, self.digest()?);
            }
        }
        Ok(ArPass {
            tokens,
            logits,
            digests,
        })
    }

    /// Verify (production transaction), then roll back with its own ticket.
    fn verify_logits_rolled_back(
        &mut self,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let got = oracle::dev_verify_logits(&mut self.m.bundle, &mut self.m.gpu, block, scratch);
        match got {
            Ok(got) => {
                oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                    .inspect_err(|_| self.dirty = true)?;
                self.check_rewind("verify_ticket")?;
                Ok(got)
            }
            Err(error) => {
                self.dirty = true;
                Err(error)
            }
        }
    }

    /// Separate instrumented verifies: never substitute these for host E2E.
    fn verify_kernel_timing(
        &mut self,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
        warmup_launches: u64,
    ) -> Result<(Value, Value), String> {
        #[cfg(feature = "kernel-timing")]
        {
            let capacity = profile::kernel_timing_capacity(warmup_launches)?;
            let measured = (|| -> Result<(Value, Value), String> {
                self.m.gpu.lab_prepare_kernel_timing(capacity).map_err(es)?;
                self.ensure_s0()?;
                let initial_state = self.digest()?;
                let want = self.d0.as_ref().ok_or("kernel timing: missing S0 digest")?;
                if oracle::dev_position(&self.m.bundle) != self.p0
                    || !digest_mismatches(want, &initial_state).is_empty()
                {
                    self.dirty = true;
                    return Err("kernel timing: initial state is not S0".into());
                }
                let mut samples = Vec::new();
                let mut raw = Vec::new();
                for repeat in 0..=self.iters {
                    if self.cfg.deadline.exceeded() {
                        return Err(format!(
                            "deadline before GPU kernel timing repeat {repeat}; incomplete samples"
                        ));
                    }
                    self.ensure_s0()?;
                    self.sync()?;
                    if self.m.gpu.replay.is_recording() || self.m.gpu.graphs.capture_mode {
                        return Err("GPU kernel timing requires eager HIP launches, not replay/graph capture".into());
                    }
                    // This oracle starts the collector only AFTER the real
                    // production snapshot. Errors auto-abort its transaction.
                    let report = oracle::dev_verify_resident_profiled(
                        &mut self.m.bundle,
                        &mut self.m.gpu,
                        block,
                        scratch,
                    )
                    .inspect_err(|_| self.dirty = true)?;
                    // Always consume the successful ticket before validating
                    // the report; even malformed timing must leave no ticket.
                    oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                        .inspect_err(|_| self.dirty = true)?;
                    self.sync()?;
                    let live = self.digest()?;
                    if oracle::dev_position(&self.m.bundle) != self.p0
                        || !digest_mismatches(&initial_state, &live).is_empty()
                    {
                        self.dirty = true;
                        return Err(format!(
                            "GPU kernel timing repeat {repeat} rollback did not restore S0"
                        ));
                    }
                    // SUM of the actual per-launch spans via the shared
                    // validator: exact eager-wrapper coverage, count match,
                    // capacity fit, finite/nonnegative spans. Ticket already
                    // consumed above, so malformed timing leaves no ticket.
                    let span = profile::kernel_report_span_ms(&report, capacity)?;
                    samples.push(span);
                    raw.push(json!({
                        "repeat": repeat,
                        "warmup": repeat == 0,
                        "kernel_count": report.kernel_count,
                        "per_kernel_ms": report.per_kernel_ms,
                        "coverage": "EagerHipLaunchWrappers",
                    }));
                    if self.cfg.deadline.exceeded() {
                        return Err(format!(
                            "deadline after GPU kernel timing repeat {repeat}; incomplete samples"
                        ));
                    }
                }
                let timed = Timed::from_samples(&samples).ok_or("--iters >= 1 is required")?;
                Ok((
                    timed.json(),
                    json!({
                        "available": true,
                        "complete": true,
                        "coverage": "EagerHipLaunchWrappers",
                        "capacity": capacity,
                        "warmup_launches": warmup_launches,
                        "census_boundary": "existing uninstrumented verify warmup, including snapshot launches, excluding ticket rollback",
                        "capacity_policy": "checked 2 * warmup_launches + 32; capacity is not an observed span",
                        "raw": {"warmup": raw[0], "repeats": &raw[1..]},
                    }),
                ))
            })();
            // Explicit release is outside every host/event window and precedes
            // any evidence claim, including on partial collection failure.
            let closed = self.m.gpu.lab_close_kernel_timing().map_err(es);
            match (measured, closed) {
                (Ok((timed, mut receipt)), Ok(())) => {
                    receipt["pool_closed"] = json!(true);
                    Ok((timed, receipt))
                }
                (Err(error), Ok(())) => Err(error),
                (Ok(_), Err(error)) => Err(format!("GPU kernel timing pool close failed: {error}")),
                (Err(error), Err(close)) => Err(format!(
                    "{error}; GPU kernel timing pool close failed: {close}"
                )),
            }
        }
        #[cfg(not(feature = "kernel-timing"))]
        {
            let _ = (block, scratch);
            Ok((
                Value::Null,
                json!({
                    "available": false,
                    "complete": false,
                    "missing_reason": "kernel-timing feature unavailable; build with reference-parity,lab,kernel-timing",
                    "coverage": Value::Null,
                    "capacity": Value::Null,
                    "warmup_launches": warmup_launches,
                    "raw": Value::Null,
                }),
            ))
        }
    }

    fn run_n(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        accepts: Accepts,
    ) -> Result<NOutcome, String> {
        self.ensure_s0()?;
        let mut scratch = self
            .m
            .bundle
            .new_spec_scratch(&mut self.m.gpu, n)
            .map_err(|e| format!("new_spec_scratch({n}): {e}"))?;
        let outcome = self.run_n_body(n, block, ar, accepts, &mut *scratch);
        scratch.free(&mut self.m.gpu);
        outcome
    }

    /// Whether a mismatch at verify width `n` is fatal under the profile.
    fn hard(&self, n: usize) -> bool {
        if self.cfg.profile == Profile::Strict {
            return !strict_arm_is_diagnostic(self.requested_arm_rows, self.strict_candidate_rows);
        }
        self.cfg
            .profile
            .hard_width(n, self.qualified, FEW_ROW_ROUTE_ROWS)
    }

    fn run_n_body(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        accepts: Accepts,
        scratch: &mut dyn SpecScratch,
    ) -> Result<NOutcome, String> {
        let vocab = self.m.vocab;
        let ring_rows = oracle::dev_row_capture_rows(&self.m.bundle);
        let armed = oracle::dev_would_arm(&self.m.bundle, n);
        let qualified = self.qualified;
        let strict = self.cfg.profile == Profile::Strict;
        let caps = oracle::dev_verify_caps(&self.m.bundle, &self.m.gpu);
        let native_applicable = oracle::dev_row_capture_supported(&self.m.bundle, &self.m.gpu);
        let native_supported = oracle::dev_native_retain_supported(&self.m.bundle, n);
        let required_rows =
            strict_required_rows(self.requested_arm_rows, self.strict_candidate_rows, n);
        if strict {
            if accepts == Accepts::None {
                return Err(format!(
                    "strict rows {n}: accepted-prefix coverage was disabled"
                ));
            }
            if caps.row_exact_verify_rows < required_rows
                || caps.spec_verify_rows_cap < required_rows
                || caps.forward_max_chunk < required_rows
            {
                return Err(format!(
                    "strict rows {n}: requested arm {:?} requires {required_rows} actual rows, effective caps are row_exact={} spec={} forward={}",
                    self.requested_arm_rows, caps.row_exact_verify_rows,
                    caps.spec_verify_rows_cap, caps.forward_max_chunk,
                ));
            }
            if !native_supported || (n >= 2 && (!native_applicable || !armed)) {
                return Err(format!(
                    "strict rows {n}: native captured-prefix coverage unavailable (applicable={native_applicable}, supported={native_supported}, armed={armed}, ring={ring_rows})"
                ));
            }
        }

        // 1. Verify logits vs AR, bitwise per row. Strict runs the production
        // transaction under the neutral binding collector and checks every
        // actual in-scope launch against the G1 raw pins (fail closed).
        let mut binding_receipt = Value::Null;
        let (picks, logits) = if strict {
            let ((p, l), receipt) = self.strict_verify_with_binding(n, block, scratch)?;
            binding_receipt = receipt;
            (p, l)
        } else {
            self.verify_logits_rolled_back(block, scratch)?
        };
        let actual_verify_reference = verify_reference_receipt(&self.m.bundle, n);
        if strict && actual_verify_reference.is_null() {
            return Err(format!(
                "strict rows {n}: actual verify reference receipt unavailable (requires lab)"
            ));
        }
        if picks.len() != n || logits.len() != n * vocab {
            return Err(format!(
                "verify returned {} picks / {} logits for {n} rows",
                picks.len(),
                logits.len()
            ));
        }
        let cmp = compare_rows(&ar.logits, &logits, vocab, n);
        let ar_argmax: Vec<u32> = (0..n)
            .map(|r| argmax(&ar.logits[r * vocab..(r + 1) * vocab]))
            .collect();
        let picks_match = picks.iter().zip(&ar_argmax).filter(|(p, a)| p == a).count();
        let bitwise_all = cmp.bitwise_rows == n;

        // Inspect the aborted S0 before a re-prefill could hide rollback damage.
        let (abort_ok, abort_json) = if strict {
            let position = oracle::dev_position(&self.m.bundle);
            let live = self.digest()?;
            let want = self.d0.as_ref().ok_or("strict abort: missing S0 digest")?;
            let bad = digest_mismatches(want, &live);
            let state_ok = bad.is_empty() && position == self.p0;
            self.dirty = true;
            let (next_ok, next_json) = self.next_token_check(ar, 0, n, scratch)?;
            (
                state_ok && next_ok,
                json!({
                    "checked": true,
                    "state_ok": state_ok && next_ok,
                    "state_digest_ok": state_ok,
                    "position_after": position,
                    "expected_position": self.p0,
                    "mismatched_parts": bad,
                    "next_token": next_json,
                }),
            )
        } else {
            (true, Value::Null)
        };
        let (error_injection_ok, error_injection_json) = if strict {
            let (first_ok, mut first) = self.error_injection_check(ar, block, n, scratch, 0)?;
            let (partial_ok, partial) = self.error_injection_check(ar, block, n, scratch, 1)?;
            first["after_partial_work"] = partial;
            (first_ok && partial_ok, first)
        } else {
            (true, Value::Null)
        };

        // 2. Post-commit state vs the AR state after accept_len + 1 tokens.
        // Strict mode runs every accept through BOTH commit paths (the generic
        // `SpecTarget::commit_prefix` and the native retain rollback the
        // production batched window uses) and then checks the next token
        // against the AR reference: none of it may be skipped.
        let accepts: Vec<usize> = match (accepts, n) {
            (Accepts::None, _) => Vec::new(),
            (Accepts::All, _) => (0..n).collect(),
            (Accepts::Ends, 1) => vec![0],
            (Accepts::Ends, _) => vec![0, n - 1],
        };
        let mut paths = vec![CommitPath::Generic];
        if strict {
            paths.push(CommitPath::NativeRetain);
        }
        let mut accept_json = Vec::new();
        let mut state_ok_all: Option<bool> = if accepts.is_empty() { None } else { Some(true) };
        let mut accept_cells = Vec::new();
        let mut verify_repeatable = true;
        let mut exec_verify = Value::Null;
        for &path in &paths {
            for &accept_len in &accepts {
                self.ensure_s0()?;
                let p0 = oracle::dev_position(&self.m.bundle);
                let (again_picks, again) = match oracle::dev_verify_logits(
                    &mut self.m.bundle,
                    &mut self.m.gpu,
                    block,
                    scratch,
                ) {
                    Ok(got) => got,
                    Err(error) => {
                        self.dirty = true;
                        return Err(format!("accept {accept_len}: verify: {error}"));
                    }
                };
                self.dirty = true;
                if again_picks.len() != n || again.len() != n * vocab {
                    return Err(format!(
                        "accept {accept_len}: repeated verify omitted requested rows"
                    ));
                }
                if again_picks != picks
                    || again.len() != logits.len()
                    || again
                        .iter()
                        .zip(&logits)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    verify_repeatable = false;
                }
                match path {
                    CommitPath::Generic => self
                        .m
                        .bundle
                        .commit_prefix(&mut self.m.gpu, block, accept_len, p0, scratch)
                        .map_err(|e| format!("accept {accept_len}: commit_prefix: {e}"))?,
                    CommitPath::NativeRetain => oracle::dev_commit_captured_prefix(
                        &mut self.m.bundle,
                        &mut self.m.gpu,
                        block,
                        accept_len + 1,
                        scratch,
                    )
                    .map_err(|e| format!("accept {accept_len}: native retain commit: {e}"))?,
                }
                let position = oracle::dev_position(&self.m.bundle);
                let live = self.digest()?;
                let want = ar
                    .digests
                    .get(&(accept_len + 1))
                    .ok_or_else(|| format!("no AR digest after {} tokens", accept_len + 1))?;
                let bad = digest_mismatches(want, &live);
                let state_ok = bad.is_empty() && position == p0 + accept_len + 1;
                let (next_ok, next_json) = if strict {
                    self.next_token_check(ar, accept_len + 1, n, scratch)?
                } else {
                    (true, Value::Null)
                };
                let ok = state_ok && next_ok;
                state_ok_all = state_ok_all.map(|all| all && ok);
                accept_cells.push(format!(
                    "{}{accept_len}:{}",
                    path.tag(),
                    if ok { "Y" } else { "N" }
                ));
                accept_json.push(json!({
                    "accept_len": accept_len,
                    "keep": accept_len + 1,
                    "commit_path": path.name(),
                    "state_ok": ok,
                    "state_digest_ok": state_ok,
                    "position_after": position,
                    "expected_position": p0 + accept_len + 1,
                    "mismatched_parts": bad,
                    "next_token": next_json,
                }));
            }
        }

        // 3. Timing: verify (resident, no digest, no pick readback), the
        // rollback of its ticket (its own boundary, outside the verify
        // timer), and the n forward_token steps; the first repeat of each
        // boundary is the warmup and is kept apart.
        self.ensure_s0()?;
        let mut verify_samples = Vec::new();
        let mut rollback_samples = Vec::new();
        let mut ar_samples = Vec::new();
        let mut exec_ar = Value::Null;
        let mut last_kernel = None;
        let mut warmup_launches = 0;
        for it in 0..=self.iters {
            self.sync()?;
            if it == 1 {
                exec_verify = self.exec_mode();
            }
            let launch_count_before = (it == 0).then(hip_bridge::launch_counters::count);
            let start = Instant::now();
            oracle::dev_verify_resident(&mut self.m.bundle, &mut self.m.gpu, block, scratch)
                .inspect_err(|_| self.dirty = true)?;
            self.sync()?;
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            let launch_count_after = (it == 0).then(hip_bridge::launch_counters::count);
            if it == 0 {
                last_kernel = self.m.gpu.last_launched_kernel().map(str::to_owned);
            }
            let rollback_start = Instant::now();
            oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                .inspect_err(|_| self.dirty = true)?;
            self.sync()?;
            rollback_samples.push(rollback_start.elapsed().as_secs_f64() * 1e3);
            verify_samples.push(elapsed);
            if let (Some(before), Some(after)) = (launch_count_before, launch_count_after) {
                warmup_launches = after
                    .checked_sub(before)
                    .ok_or("verify warmup launch census counter overflow")?;
            }
        }
        for it in 0..=self.iters {
            let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
            self.sync()?;
            if it == 1 {
                exec_ar = self.exec_mode();
            }
            let start = Instant::now();
            let mut step_error = None;
            for &token in block {
                if let Err(error) =
                    self.m
                        .bundle
                        .forward_token(&mut self.m.gpu, token, &self.row, None)
                {
                    step_error = Some(error.to_string());
                    break;
                }
            }
            let synced = self.sync();
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es)?;
            if let Some(error) = step_error {
                return Err(format!("timed AR forward_token: {error}"));
            }
            synced?;
            ar_samples.push(elapsed);
        }
        let verify_t = Timed::from_samples(&verify_samples).ok_or("--iters >= 1 is required")?;
        let rollback_t =
            Timed::from_samples(&rollback_samples).ok_or("--iters >= 1 is required")?;
        let ar_t = Timed::from_samples(&ar_samples).ok_or("--iters >= 1 is required")?;
        let verify_med = verify_t.median();
        let ar_med = ar_t.median();
        let witness = self.kernel_witness(last_kernel);
        // The event pool and all instrumented work are separate from the
        // completed uninstrumented host samples above.
        let (gpu_span_ms, gpu_kernel_timing) =
            self.verify_kernel_timing(block, scratch, warmup_launches)?;
        let gpu_timing_available = !gpu_span_ms.is_null();

        // 4. Same-S0 prefill-route baseline (`forward_chunk`, prefill scope).
        let prefill_route = if self.cfg.baseline_prefill {
            Some(self.prefill_route_cost(n, block, ar)?)
        } else {
            None
        };

        let first = cmp
            .first_mismatch
            .map_or_else(|| "-".to_string(), |r| r.to_string());
        let line = format!(
            "rows {n}: bitwise {}/{n} first_mismatch {first} max|diff| {:.3e} argmax {}/{n} state_ok[accept={}] verify_ms {verify_med:.3} rollback_ms {:.3} ar_ms {ar_med:.3}{}",
            cmp.bitwise_rows,
            cmp.max_abs_diff,
            cmp.argmax_agree,
            if accept_cells.is_empty() {
                "none".to_string()
            } else {
                accept_cells.join(",")
            },
            rollback_t.median(),
            prefill_route
                .as_ref()
                .map_or_else(String::new, |p| format!(
                    " prefill_route_ms {:.3}",
                    p["ms"]["median"].as_f64().unwrap_or(f64::NAN)
                )),
        );
        let claim = profile::identity_claim(n, qualified, true, bitwise_all);
        let mut record = json!({
            "n": n,
            "k_drafted": n - 1,
            "executed_width": n,
            "label": profile::width_label(n, qualified),
            "qualified_row_limit": qualified,
            "identity_claim": claim,
            "hard_gate": self.hard(n),
            "strict_role": if strict {
                if self.hard(n) { "candidate" } else { "diagnostic-baseline" }
            } else { "not-strict" },
            "requested_arm_rows": self.requested_arm_rows,
            "effective_caps": bundle_caps(self),
            "route_assertions": {
                "checked": strict,
                "required_row_exact_rows": required_rows,
                "effective_caps_ok": caps.row_exact_verify_rows >= required_rows
                    && caps.spec_verify_rows_cap >= required_rows
                    && caps.forward_max_chunk >= required_rows,
                "native_retain_supported": native_supported,
                "native_capture_applicable": native_applicable,
                "capture_required": n >= 2,
                "capture_route_ok": n == 1 || (native_applicable && armed),
                "actual_route": if n == 1 { "singleton-full-accept" }
                    else if armed { "native-captured-rows" } else { "uncaptured" },
                "ar_reference": "Qwen4Bundle::forward_token",
                "actual_verify_reference": actual_verify_reference,
            },
            "beyond_physical_cap": n > WIDE_VERIFY_MAX_ROWS,
            "row_capture_armed": armed,
            "row_capture_ring_rows": ring_rows,
            "rows_bitwise": cmp.bitwise_rows,
            "all_rows_bitwise": bitwise_all,
            "bitwise": {
                "checked": true,
                "rows_bitwise": cmp.bitwise_rows,
                "all_rows_bitwise": bitwise_all,
            },
            "first_mismatch_row": cmp.first_mismatch,
            "max_abs_diff": cmp.max_abs_diff,
            "mismatching_rows": cmp.mismatching,
            "argmax_agree": cmp.argmax_agree,
            "picks": picks,
            "ar_argmax": ar_argmax,
            "picks_match_ar_argmax": picks_match,
            "verify_repeatable": verify_repeatable,
            "accepts": accept_json,
            "abort": abort_json,
            "error_injection": error_injection_json,
            "state_ok": state_ok_all,
            "execution": {"ar": exec_ar, "verify": exec_verify},
            "verify_route": {
                "kind": if n == 1 { "target-only" } else { "batched" },
                "execution": {"ar": exec_ar, "verify": exec_verify},
                "arithmetic_reference": actual_verify_reference,
            },
            "verify_ms": verify_t.json(),
            "host_window_ms": verify_t.json(),
            "host_window_boundary": {
                "role": "end-to-end-production-verify",
                "instrumented": false,
                "includes": ["snapshot", "token-and-PLE-uploads", "CPU-orchestration", "verify-kernels", "device-sync"],
                "excludes": ["logits-download", "ticket-rollback", "state-digests"],
                "warmup_excluded": true,
                "kernel_only": false,
            },
            "gpu_span_ms": gpu_span_ms,
            "gpu_kernel_timing": gpu_kernel_timing,
            "gpu_span_boundary": {
                "role": "same-production-verify-transaction",
                "instrumented": true,
                "aggregation": "SUM f64 of actual per-launch f32 HIP event spans",
                "collector_starts": "after-production-snapshot",
                "includes": ["eager-HIP-launch-wrapper-kernel-event-spans"],
                "excludes": ["snapshot", "copies", "uploads", "JIT", "downloads", "CPU-orchestration", "CPU-gaps", "ticket-rollback", "state-digests"],
                "coverage": "EagerHipLaunchWrappers only; PM4/AQL/graph refused",
                "separate_profiled_warmup_and_repeats": true,
                "warmup_excluded": true,
                "same_s0_checked_each_repeat": true,
                "prefill_scope": false,
                "primary_comparison": "same verify at wide8 versus wide64; n>8 wide8 may dispatch incumbent prefill kernels",
                "prefill_route_ms_role": "separate diagnostic only",
                "legacy_verify_ms_alias": "host_window_ms, not gpu_span_ms",
            },
            "rollback_ms": rollback_t.json(),
            "ar_ms": ar_t.json(),
            "ar_step_ms": profile::per_row_ms(ar_med, n),
            "prefill_route_ms": prefill_route,
            "per_row_ms": {
                "verify": profile::per_row_ms(verify_med, n),
                "ar_step": profile::per_row_ms(ar_med, n),
            },
        });
        let binding_ok = !strict || binding_receipt.get("complete") == Some(&json!(true));
        {
            let map = record
                .as_object_mut()
                .expect("wide verify record must be a JSON object");
            map.insert("route_fingerprint".to_string(), witness);
            map.insert("code_object_binding".to_string(), binding_receipt);
        }
        Ok(NOutcome {
            record,
            clean: bitwise_all
                && state_ok_all.unwrap_or(true)
                && binding_ok
                && (!strict
                    || (gpu_timing_available
                        && picks_match == n
                        && verify_repeatable
                        && abort_ok
                        && error_injection_ok)),
            line,
            cost: Some(WidthCost {
                n,
                verify_ms: verify_med,
                commit_ms: BTreeMap::new(),
            }),
        })
    }

    /// Compare both n-row continuation routes from the live consumed prefix
    /// (zero means aborted S0). Verify is rolled back before plain AR runs;
    /// neither route can pass with missing reference logits or state.
    fn next_token_check(
        &mut self,
        ar: &ArPass,
        consumed: usize,
        n: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(bool, Value), String> {
        let vocab = self.m.vocab;
        let end = consumed + n;
        if n == 0 || end > ar.tokens.len() || end * vocab > ar.logits.len() {
            return Err(format!(
                "strict continuation: no {n}-row AR window after {consumed} consumed tokens"
            ));
        }
        let want_state = ar
            .digests
            .get(&end)
            .ok_or_else(|| format!("strict continuation: missing AR state after {end} tokens"))?;
        let tokens = &ar.tokens[consumed..end];
        let want_logits = &ar.logits[consumed * vocab..end * vocab];
        let before_position = oracle::dev_position(&self.m.bundle);
        let expected_position = self.p0 + end;
        let before = self.digest()?;
        let (verify_picks, verify_logits) =
            oracle::dev_verify_logits(&mut self.m.bundle, &mut self.m.gpu, tokens, scratch)
                .inspect_err(|_| self.dirty = true)?;
        let actual_reference = verify_reference_receipt(&self.m.bundle, n);
        let verify_position = oracle::dev_position(&self.m.bundle);
        let verify_state = self.digest()?;
        let verify_bad = digest_mismatches(want_state, &verify_state);
        oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
            .inspect_err(|_| self.dirty = true)?;
        if verify_picks.len() != n || verify_logits.len() != n * vocab {
            return Err(format!(
                "strict continuation: verify returned {} picks / {} logits for {n} rows",
                verify_picks.len(),
                verify_logits.len(),
            ));
        }
        if actual_reference.is_null() {
            return Err("strict continuation: actual verify reference receipt unavailable".into());
        }
        let after = self.digest()?;
        let rollback_bad = digest_mismatches(&before, &after);
        let rollback_position = oracle::dev_position(&self.m.bundle);
        let rollback_ok = rollback_bad.is_empty() && rollback_position == before_position;
        let verify_cmp = compare_rows(want_logits, &verify_logits, vocab, n);
        let verify_picks_ok = verify_picks
            .iter()
            .enumerate()
            .all(|(r, &pick)| pick == argmax(&want_logits[r * vocab..(r + 1) * vocab]));
        let verify_state_ok = verify_bad.is_empty() && verify_position == expected_position;
        let verify_ok =
            verify_cmp.bitwise_rows == n && verify_picks_ok && verify_state_ok && rollback_ok;
        let mut ar_bitwise_rows = 0;
        let mut first_bitwise = false;
        let mut first_argmax = false;
        let mut first_max_diff = 0.0;
        for (r, &token) in tokens.iter().enumerate() {
            self.m
                .bundle
                .forward_token(&mut self.m.gpu, token, &self.row, None)
                .map_err(|e| format!("strict continuation AR row {r}: {e}"))?;
            let got = self.m.gpu.download_f32(&self.row).map_err(es)?;
            if got.len() != vocab {
                return Err(format!(
                    "strict continuation AR row {r}: wrong logits length {}",
                    got.len()
                ));
            }
            let cmp = compare_rows(&want_logits[r * vocab..(r + 1) * vocab], &got, vocab, 1);
            ar_bitwise_rows += cmp.bitwise_rows;
            if r == 0 {
                first_bitwise = cmp.bitwise_rows == 1;
                first_argmax = cmp.argmax_agree == 1;
                first_max_diff = cmp.max_abs_diff;
            }
        }
        let ar_position = oracle::dev_position(&self.m.bundle);
        let ar_state = self.digest()?;
        let ar_bad = digest_mismatches(want_state, &ar_state);
        let ar_state_ok = ar_bad.is_empty() && ar_position == expected_position;
        let ok = verify_ok && ar_bitwise_rows == n && ar_state_ok;
        Ok((
            ok,
            json!({
                "checked": true,
                "row": consumed,
                "token": tokens[0],
                "continuation_width": n,
                "bitwise": first_bitwise,
                "argmax_agree": first_argmax,
                "max_abs_diff": first_max_diff,
                "ok": ok,
                "next_verify": {
                    "checked": true,
                    "rows": n,
                    "bitwise": verify_cmp.bitwise_rows == n,
                    "picks_match_ar_argmax": verify_picks_ok,
                    "state_ok": verify_state_ok,
                    "state_digest_ok": verify_bad.is_empty(),
                    "mismatched_parts": verify_bad,
                    "position_after": verify_position,
                    "expected_position": expected_position,
                    "rollback_state_ok": rollback_ok,
                    "rollback_mismatched_parts": rollback_bad,
                    "rollback_position": rollback_position,
                    "expected_rollback_position": before_position,
                    "arithmetic_reference": actual_reference,
                },
                "ar_continuation": {
                    "checked": true,
                    "rows": n,
                    "bitwise": ar_bitwise_rows == n,
                    "state_ok": ar_state_ok,
                    "state_digest_ok": ar_bad.is_empty(),
                    "mismatched_parts": ar_bad,
                    "position_after": ar_position,
                    "expected_position": expected_position,
                },
            }),
        ))
    }

    fn error_injection_check(
        &mut self,
        ar: &ArPass,
        block: &[u32],
        n: usize,
        scratch: &mut dyn SpecScratch,
        successful_launches: usize,
    ) -> Result<(bool, Value), String> {
        #[cfg(not(feature = "lab"))]
        {
            let _ = (ar, block, n, scratch, successful_launches);
            Err("strict scoped-error coverage requires lab".into())
        }
        #[cfg(feature = "lab")]
        {
            self.ensure_s0()?;
            let before_reference = self.m.gpu.qwen4_verify_reference();
            let before_rows = self.m.gpu.set_wide_verify_rows(0);
            self.m.gpu.set_wide_verify_rows(before_rows);
            let outcome = oracle::dev_verify_resident_fault_after_snapshot(
                &mut self.m.bundle,
                &mut self.m.gpu,
                block,
                scratch,
                successful_launches,
            )
            .inspect_err(|_| self.dirty = true)?;
            self.dirty = true;
            let after_reference = self.m.gpu.qwen4_verify_reference();
            let after_rows = self.m.gpu.set_wide_verify_rows(0);
            self.m.gpu.set_wide_verify_rows(after_rows);
            let reference_ok = before_reference == after_reference;
            let rows_ok = before_rows == after_rows;
            if !outcome.receipt.injected
                || outcome.receipt.successful_launches != successful_launches
                || !reference_ok
                || !rows_ok
            {
                return Err(format!(
                    "strict scoped fault rows {n}: injected={}, reference restored={reference_ok}, rows restored={rows_ok}",
                    outcome.receipt.injected,
                ));
            }
            let position = oracle::dev_position(&self.m.bundle);
            let live = self.digest()?;
            let want = self
                .d0
                .as_ref()
                .ok_or("strict scoped fault: missing S0 digest")?;
            let bad = digest_mismatches(want, &live);
            let state_ok = bad.is_empty() && position == self.p0;
            // verify_block_rows refuses an already-active target snapshot.
            // Successfully reusing this scratch proves the failed ticket ended.
            let (next_ok, next_json) = self.next_token_check(ar, 0, n, scratch)?;
            Ok((
                state_ok && next_ok,
                json!({
                    "checked": true,
                    "trigger": if successful_launches == 0 {
                        "next-kernel-launch-after-snapshot"
                    } else {
                        "kernel-launch-after-successful-work"
                    },
                    "requested_successful_launches": successful_launches,
                    "successful_launches": outcome.receipt.successful_launches,
                    "injected": outcome.receipt.injected,
                    "api": "oracle::dev_verify_resident_fault_after_snapshot",
                    "error_observed": true,
                    "error": outcome.verify_error,
                    "reference_scope_restored": reference_ok,
                    "rows_scope_restored": rows_ok,
                    "reference_scope_before": format!("{before_reference:?}"),
                    "reference_scope_after": format!("{after_reference:?}"),
                    "rows_scope_before": before_rows,
                    "rows_scope_after": after_rows,
                    "ticket_released": true,
                    "ticket_release_evidence": "subsequent-production-verify-accepted-same-scratch",
                    "state_ok": state_ok && next_ok,
                    "state_digest_ok": bad.is_empty(),
                    "mismatched_parts": bad,
                    "position_after": position,
                    "expected_position": self.p0,
                    "next_token": next_json,
                }),
            ))
        }
    }

    /// Per-launch code-object binding witnesses of one width: every actual
    /// launch the verify transaction executed (symbol, bound module, admitted
    /// object SHA via the frozen owner-binding proof, grid/block/shared-mem),
    /// plus the cumulative loaded-kernel census. Input module labels are
    /// lookup hints only; the binding digest plus owner proof is the evidence.
    /// Under `lab` the witnesses are collected by the neutral launch hook;
    /// without `lab` no per-launch witness exists (null), never fabricated.
    fn kernel_witness(&mut self, last_kernel: Option<String>) -> Value {
        let names = self.m.gpu.loaded_kernel_names();
        let new: Vec<String> = names
            .iter()
            .filter(|name| !self.kernel_seen.contains(*name))
            .cloned()
            .collect();
        let mut hasher = Sha256::new();
        for name in &names {
            hasher.update(name.as_bytes());
            hasher.update(b"\n");
        }
        self.kernel_seen.extend(names.iter().cloned());
        let launches = std::mem::take(&mut self.pending_launch_witnesses);
        #[cfg(feature = "lab")]
        let launch_witnesses: Vec<Value> = launches
            .iter()
            .map(|w| {
                json!({
                    "symbol": w.symbol,
                    "module": w.module,
                    "container_sha256": w.object_sha256,
                    "device_sha256": w.device_sha256,
                    "grid": w.grid,
                    "block": w.block,
                    "shared_mem": w.shared_mem,
                })
            })
            .collect();
        #[cfg(not(feature = "lab"))]
        let launch_witnesses = Value::Null;
        #[cfg(not(feature = "lab"))]
        let _ = launches;
        json!({
            "loaded_kernel_count": names.len(),
            "loaded_kernels_sha256": format!("{:x}", hasher.finalize()),
            "new_kernels_since_previous_width": new,
            "last_verify_kernel": last_kernel,
            "actual_launches": launch_witnesses,
        })
    }

    /// Run one unpriced strict verify transaction under the neutral
    /// per-launch binding collector, then check every in-scope witness
    /// against the immutable G1 raw image pins. Restores the previous hook
    /// on success, error and rollback paths; priced (G4) transactions and
    /// the normal production path never install this hook.
    #[cfg(feature = "lab")]
    fn strict_verify_with_binding(
        &mut self,
        n: usize,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
    ) -> Result<((Vec<u32>, Vec<f32>), Value), String> {
        // The collector witnesses actual bound launches through the shared
        // eager funnels only. A recording replay or graph capture would
        // relaunch retained/captured blobs that bypass those funnels, so the
        // current-map identity would not name their owner: refuse instead of
        // claiming retained-function ownership.
        if self.m.gpu.replay.is_recording() || self.m.gpu.graphs.capture_mode {
            return Err(
                "strict rows: binding witness requires eager HIP launches, not replay/graph capture"
                    .into(),
            );
        }
        let state = Arc::new(Mutex::new(Vec::<StrictLaunchWitness>::new()));
        let probe = Arc::clone(&state);
        let observer = Arc::new(
            move |gpu: &Gpu, launch: &rdna_compute::LabLaunchWitness<'_>| -> HipResult<()> {
                // Exactly one owner lookup per launch: the device getter's tuple
                // already carries module + container + raw for in-scope symbols;
                // the frozen getter serves out-of-scope symbols only.
                let (module, object_sha256, device_sha256) = if strict_wide_symbol(launch.symbol) {
                    let (module, container, device) =
                    gpu.lab_kernel_device_identity(launch.symbol).map_err(|e| {
                        HipError::new(
                            0,
                            &format!(
                                "strict binding witness: symbol {:?} has no loaded owner or raw device image ({e})",
                                launch.symbol
                            ),
                        )
                    })?;
                    (module, container, Some(device))
                } else {
                    let (module, container) =
                        gpu.lab_kernel_identity(launch.symbol).map_err(|e| {
                            HipError::new(
                                0,
                                &format!(
                                    "strict binding witness: symbol {:?} has no loaded owner ({e})",
                                    launch.symbol
                                ),
                            )
                        })?;
                    (module, container, None)
                };
                let mut guard = probe.lock().unwrap_or_else(|e| e.into_inner());
                guard.push(StrictLaunchWitness {
                    symbol: launch.symbol.to_string(),
                    module,
                    object_sha256,
                    device_sha256,
                    grid: launch.grid,
                    block: launch.block,
                    shared_mem: launch.shared_mem,
                });
                Ok(())
            },
        );
        let old = self.m.gpu.set_lab_launch_observer(Some(observer));
        let outcome = (|| -> Result<(Vec<u32>, Vec<f32>), String> {
            let got =
                oracle::dev_verify_logits(&mut self.m.bundle, &mut self.m.gpu, block, scratch)
                    .inspect_err(|_| self.dirty = true)?;
            oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                .inspect_err(|_| self.dirty = true)?;
            Ok(got)
        })();
        self.m.gpu.set_lab_launch_observer(old);
        let got = outcome?;
        let witnesses = state.lock().unwrap_or_else(|e| e.into_inner()).clone();
        self.pending_launch_witnesses = witnesses;
        let receipt = check_strict_binding(
            n,
            &self.m.gpu,
            &self.pending_launch_witnesses,
            self.requested_arm_rows,
            self.strict_candidate_rows,
        )?;
        Ok((got, receipt))
    }

    #[cfg(not(feature = "lab"))]
    fn strict_verify_with_binding(
        &mut self,
        _n: usize,
        _block: &[u32],
        _scratch: &mut dyn SpecScratch,
    ) -> Result<((Vec<u32>, Vec<f32>), Value), String> {
        Err("strict rows: code-object binding witness requires lab".into())
    }

    /// The same-S0 baseline arm: the block run through the production
    /// PREFILL route (`Qwen4Bundle::forward_chunk`: prefill scope, all-row
    /// logits) from the same post-prefill state, timed with the same
    /// warmup / repeats / device sync as the verify. Informational: it makes
    /// no identity claim (the prefill route is not the verify route).
    fn prefill_route_cost(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
    ) -> Result<Value, String> {
        self.ensure_s0()?;
        let vocab = self.m.vocab;
        let logits = self.m.gpu.zeros(&[n * vocab], DType::F32).map_err(es)?;
        let incumbent = self.cfg.profile == Profile::Strict;
        if incumbent {
            rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(Some(
                FEW_ROW_ROUTE_ROWS,
            ));
        }
        let result = self.prefill_route_samples(n, block, ar, &logits);
        if incumbent {
            rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(
                self.requested_arm_rows,
            );
        }
        let freed = self.m.gpu.free_tensor(logits).map_err(es);
        let value = result?;
        freed?;
        Ok(value)
    }

    fn prefill_route_samples(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        logits: &GpuTensor,
    ) -> Result<Value, String> {
        let vocab = self.m.vocab;
        let mut samples = Vec::new();
        let mut bitwise = None;
        for it in 0..=self.iters {
            let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
            self.sync()?;
            let start = Instant::now();
            let run = self
                .m
                .bundle
                .forward_chunk(&mut self.m.gpu, block, logits, None)
                .map_err(|e| format!("prefill-route forward_chunk: {e}"));
            let synced = self.sync();
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            let mut downloaded = Ok(());
            if it == 0 && run.is_ok() && synced.is_ok() {
                match self.m.gpu.download_f32(logits).map_err(es) {
                    Ok(got) => {
                        let cmp = compare_rows(&ar.logits, &got, vocab, n);
                        bitwise = Some((cmp.bitwise_rows, cmp.argmax_agree));
                    }
                    Err(error) => downloaded = Err(error),
                }
            }
            self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es)?;
            run?;
            synced?;
            downloaded?;
            samples.push(elapsed);
        }
        let timed = Timed::from_samples(&samples).ok_or("--iters >= 1 is required")?;
        Ok(json!({
            "route": "Qwen4Bundle::forward_chunk (prefill_scope=true, all-row logits) from the same S0",
            "comparison_role": if self.cfg.profile == Profile::Strict { "diagnostic-incumbent-prefill" } else { "diagnostic-prefill" },
            "incumbent_arm_rows": if self.cfg.profile == Profile::Strict { Some(FEW_ROW_ROUTE_ROWS) } else { self.requested_arm_rows },
            "executed_width": n,
            "effective_row_exact_rows": oracle::dev_effective_row_exact_rows(&self.m.bundle, &self.m.gpu),
            "identity_claim": false,
            "primary_verify_comparison": false,
            "ms": timed.json(),
            "host_window_ms": timed.json(),
            "host_window_boundary": {
                "role": "diagnostic-prefill-forward",
                "instrumented": false,
                "includes": ["token-and-PLE-uploads", "CPU-orchestration", "prefill-kernels", "device-sync"],
                "excludes": ["snapshot", "logits-download", "ticket-restore", "state-digests"],
                "warmup_excluded": true,
                "kernel_only": false,
            },
            "per_row_ms": profile::per_row_ms(timed.median(), n),
            "rows_bitwise_vs_ar": bitwise.map(|b| b.0),
            "rows_argmax_agree_vs_ar": bitwise.map(|b| b.1),
        }))
    }

    /// The commit boundary of one width (`--cost-curve --commit on`): per
    /// accept_len in `{0, n-1}` the production `commit_prefix` + device sync
    /// after an (untimed) verify. A committed state has no ticket, so every
    /// repeat re-prefills (checked against the post-prefill digest). The
    /// first repeat also compares the post-commit state with the AR state.
    fn run_commit(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
    ) -> Result<(Value, BTreeMap<usize, Option<f64>>, bool), String> {
        self.ensure_s0()?;
        let mut scratch = self
            .m
            .bundle
            .new_spec_scratch(&mut self.m.gpu, n)
            .map_err(|e| format!("new_spec_scratch({n}): {e}"))?;
        let outcome = self.commit_costs(n, block, ar, &mut *scratch);
        scratch.free(&mut self.m.gpu);
        outcome
    }

    fn commit_costs(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(Value, BTreeMap<usize, Option<f64>>, bool), String> {
        let hard = self.hard(n);
        let accepts: Vec<usize> = if n == 1 { vec![0] } else { vec![0, n - 1] };
        let mut out = serde_json::Map::new();
        let mut medians = BTreeMap::new();
        let mut clean = true;
        for &accept_len in &accepts {
            let mut samples = Vec::new();
            let mut state_ok: Option<bool> = None;
            let mut mismatched: Vec<String> = Vec::new();
            for it in 0..=self.cfg.commit_iters {
                if self.cfg.deadline.exceeded() {
                    break;
                }
                self.ensure_s0()?;
                let p0 = oracle::dev_position(&self.m.bundle);
                oracle::dev_verify_resident(&mut self.m.bundle, &mut self.m.gpu, block, scratch)
                    .inspect_err(|_| self.dirty = true)?;
                self.dirty = true;
                self.sync()?;
                let start = Instant::now();
                self.m
                    .bundle
                    .commit_prefix(&mut self.m.gpu, block, accept_len, p0, scratch)
                    .map_err(|e| format!("commit {accept_len}: commit_prefix: {e}"))?;
                self.sync()?;
                samples.push(start.elapsed().as_secs_f64() * 1e3);
                if it == 0 {
                    if let Some(want) = ar.digests.get(&(accept_len + 1)) {
                        let live = self.digest()?;
                        mismatched = digest_mismatches(want, &live);
                        let position = oracle::dev_position(&self.m.bundle);
                        let ok = mismatched.is_empty() && position == p0 + accept_len + 1;
                        state_ok = Some(ok);
                        if !ok && hard {
                            clean = false;
                        }
                    }
                }
            }
            match Timed::from_samples(&samples) {
                Some(timed) => {
                    medians.insert(accept_len, Some(timed.median()));
                    out.insert(
                        accept_len.to_string(),
                        json!({
                            "ms": timed.json(),
                            "state_ok": state_ok,
                            "mismatched_parts": mismatched,
                            "boundary": "SpecTarget::commit_prefix + device sync, after an untimed verify; re-prefill rewinds between repeats",
                        }),
                    );
                }
                None => {
                    medians.insert(accept_len, None);
                    self.missing.push(json!({
                        "component": "commit",
                        "n": n,
                        "accept_len": accept_len,
                        "reason": "deadline exceeded before a measured repeat",
                    }));
                    out.insert(
                        accept_len.to_string(),
                        json!({"skipped": "deadline exceeded before a measured repeat"}),
                    );
                }
            }
        }
        Ok((Value::Object(out), medians, clean))
    }

    /// Head-chain costs and real production windows of one prompt (cost
    /// profile). The target and the head are re-prefilled through
    /// `Qwen4MtpDrafter::mtp_prefill` (the production prefill of a native MTP
    /// request); the rig is marked dirty afterwards.
    fn head_and_production(
        &mut self,
        prompt_index: usize,
        prompt: &[u32],
        args: &Args,
        copy_lane: bool,
    ) -> Result<(Value, BTreeMap<usize, f64>, Vec<ProdWindow>), String> {
        let mut head_ms = BTreeMap::new();
        let mut prod = Vec::new();
        let want_head = !args.head_depths.is_empty();
        let want_prod = args.prod_windows > 0 && !args.prod_ks.is_empty();
        if !want_head && !want_prod {
            return Ok((
                json!({"measured": false, "reason": "--head-depths off and --prod-windows 0"}),
                head_ms,
                prod,
            ));
        }
        if !oracle::dev_head_attached(&self.m.bundle) {
            self.missing.push(json!({
                "component": "head",
                "reason": "the production load attached no native MTP head",
            }));
            return Ok((
                json!({"measured": false, "reason": "the production load attached no native MTP head"}),
                head_ms,
                prod,
            ));
        }
        let mut drafter = Qwen4MtpDrafter::new(args.max_k, args.n_ctx, None);
        let ngram_armed = hipfire_config::mtp_ngram_enabled_for_arch(
            hipfire_arch_qwen4::ARCH_ID,
            &self.m.gpu.arch,
        );
        if ngram_armed {
            let triple = hipfire_config::ngram_mod_triple_for_arch(
                hipfire_arch_qwen4::ARCH_ID,
                &self.m.gpu.arch,
            );
            let config = qwen4_ngram_mod_config(triple)?;
            drafter = drafter.with_ngram(Some(config)).with_copy_lane(copy_lane);
            drafter.configure_request(SpecRequestConfig {
                allow_ngram_modifier: true,
                ..SpecRequestConfig::default()
            });
        }
        let mut depths_json = Vec::new();
        let mut stopped = Value::Null;
        let mut seed_out = Value::Null;
        let mut position_out = Value::Null;
        let body = self.head_and_production_body(
            &mut drafter,
            prompt_index,
            prompt,
            args,
            &mut depths_json,
            &mut head_ms,
            &mut prod,
            &mut stopped,
            &mut seed_out,
            &mut position_out,
        );
        Box::new(drafter).mtp_free(&mut self.m.gpu);
        self.dirty = true;
        body?;
        let head = json!({
            "measured": want_head,
            "seed": seed_out,
            "position": position_out,
            "ngram_armed": ngram_armed,
            "copy_lane": copy_lane,
            "depths": depths_json,
            "boundary": "dev_head_chain: exactly `depth` production batched-route head steps (row 0 reads the pending hidden, later rows chain the previous draft), no margin prune, device-synced; head rollback is its own boundary",
            "production_windows_stopped": stopped,
        });
        Ok((head, head_ms, prod))
    }

    #[allow(clippy::too_many_arguments)]
    fn head_and_production_body(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt_index: usize,
        prompt: &[u32],
        args: &Args,
        depths_json: &mut Vec<Value>,
        head_ms: &mut BTreeMap<usize, f64>,
        prod: &mut Vec<ProdWindow>,
        stopped: &mut Value,
        seed_out: &mut Value,
        position_out: &mut Value,
    ) -> Result<(), String> {
        let deadline = self.cfg.deadline;
        let seed = drafter
            .mtp_prefill(
                &mut self.m.gpu,
                &mut self.m.bundle,
                prompt,
                prompt,
                0,
                false,
                &|| deadline.exceeded(),
            )?
            .into_first_token()?;
        let position = oracle::dev_position(&self.m.bundle);
        let head_position = oracle::dev_head_position(&self.m.bundle)?;
        if head_position != position {
            return Err(format!(
                "MTP prefill left the head at {head_position}, the target at {position}"
            ));
        }
        *seed_out = json!(seed);
        *position_out = json!(position);
        let production_s0 = self.digest()?;
        // 1. Isolated head chains (state restored after every repeat).
        if !args.head_depths.is_empty() {
            let bytes = oracle::dev_head_pending_hidden(drafter, &self.m.gpu)?;
            if bytes.is_empty() || bytes.len() % 4 != 0 {
                return Err(format!("pending hidden is {} bytes", bytes.len()));
            }
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let pending = self
                .m
                .gpu
                .upload_f32(&floats, &[floats.len()])
                .map_err(es)?;
            let chains = self.head_chains(&pending, seed, position, args, depths_json, head_ms);
            let freed = self.m.gpu.free_tensor(pending).map_err(es);
            chains?;
            freed?;
        }
        // Every forced window starts from the SAME real prompt state. Prefill
        // is outside the price; lookup/head/verification/repair/commit and
        // final device synchronization are inside the adapter's total.
        if args.prod_windows > 0 {
            use oracle::{DevForcedWindow, DevWindowSource};
            let eos = self.m.bundle.config.eos_token_id;
            for &k in &args.prod_ks {
                let sources: &[DevWindowSource] = if k == 0 {
                    &[DevWindowSource::ArRouted]
                } else if args.cost_mode.unwrap_or(CostMode::of_copy_lane(
                    hipfire_config::mtp_ngram_copy_enabled_for_arch(
                        hipfire_arch_qwen4::ARCH_ID,
                        &self.m.gpu.arch,
                    ),
                )) == CostMode::OnCopy
                {
                    &[DevWindowSource::Mtp, DevWindowSource::Ngram]
                } else {
                    &[DevWindowSource::Mtp]
                };
                for &source in sources {
                    let source_name = match source {
                        DevWindowSource::ArRouted => "ArRoutedK0",
                        DevWindowSource::Mtp => "Mtp",
                        DevWindowSource::Ngram => "Ngram",
                    };
                    for repeat in 0..=args.prod_windows {
                        let warmup = repeat == 0;
                        let mut receipt = ProdWindow {
                            requested_source: source_name,
                            request_id: None,
                            terminal: None,
                            terminal_reason: None,
                            rejected: None,
                            source: source_name,
                            variant: None,
                            requested_k: k,
                            position,
                            drafts: None,
                            accepted: None,
                            committed: None,
                            wall_ms: None,
                            requested_confirmation: None,
                            head_confirmation: None,
                            terminal_censored: false,
                            native_draft_margins: Vec::new(),
                            complete: false,
                            offer_available: None,
                            missing_reason: None,
                            warmup,
                            verify_route: Value::Null,
                        };
                        let measured = (|| -> Result<(), String> {
                            if self.cfg.deadline.exceeded() {
                                return Err("deadline before production prefill".into());
                            }
                            let deadline = self.cfg.deadline;
                            let fresh_seed = drafter
                                .mtp_prefill(
                                    &mut self.m.gpu,
                                    &mut self.m.bundle,
                                    prompt,
                                    prompt,
                                    0,
                                    false,
                                    &|| deadline.exceeded(),
                                )?
                                .into_first_token()?;
                            let request_epoch_ns = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_err(es)?
                                .as_nanos();
                            receipt.request_id = Some(format!(
                                "pid{}-t{request_epoch_ns}:prompt-{prompt_index}:{source_name}:K{k}:repeat{repeat}",
                                std::process::id(),
                            ));
                            if self.cfg.deadline.exceeded() {
                                return Err("deadline after production prefill".into());
                            }
                            let at = oracle::dev_position(&self.m.bundle);
                            receipt.position = at;
                            if at != position || fresh_seed != seed {
                                return Err(
                                    "same-S0 production prefill changed position or seed".into()
                                );
                            }
                            let fresh_state = self.digest()?;
                            if !digest_mismatches(&production_s0, &fresh_state).is_empty() {
                                return Err(
                                    "same-S0 production prefill changed target state".into()
                                );
                            }
                            let forecast = self.m.bundle.ar_verify_reference(&self.m.gpu);
                            let forecast_eager = forecast.is_some_and(|reference| {
                                matches!(
                                    reference,
                                    rdna_compute::prefill_invariant::Qwen4VerifyReference::Eager
                                )
                            });
                            let mut execution_ar = self.exec_mode();
                            // Forecast decides only the eager leaf: Ready retains
                            // flags false but the actual AR retained. Recording /
                            // capture evidence stays intact for the matcher.
                            execution_ar["eager"] = json!(forecast_eager);
                            receipt.verify_route = json!({
                                "execution": {
                                    "ar": execution_ar,
                                    "verify": self.exec_mode(),
                                },
                            });
                            let measured = oracle::dev_run_forced_window(
                                drafter,
                                &mut self.m.bundle,
                                &mut self.m.gpu,
                                at,
                                fresh_seed,
                                &[fresh_seed],
                                eos,
                                DevForcedWindow {
                                    source,
                                    drafts: k,
                                    // Explicit lab-only cost profile, not a shipping qualification.
                                    timing_only: true,
                                },
                            )?;
                            // Forecast reference for BOTH K0 AR and MTP2+ windows:
                            // no singleton-unbound pricing. The pre-call flags stay
                            // false (verify runs eager); only the AR retained
                            // forecast decides `ar.eager` (Ready retains flags
                            // false but the actual AR retained).
                            receipt.verify_route["arithmetic_reference"] = match forecast {
                                Some(
                                    rdna_compute::prefill_invariant::Qwen4VerifyReference::Eager,
                                ) => json!("Eager"),
                                Some(
                                    rdna_compute::prefill_invariant::Qwen4VerifyReference::Recorded,
                                ) => json!("Recorded"),
                                None => Value::Null,
                            };
                            receipt.verify_route["kind"] = json!(match measured.route {
                                oracle::DevWindowRoute::TargetOnly => "target-only",
                                oracle::DevWindowRoute::Batched => "batched",
                                oracle::DevWindowRoute::Interleaved => "interleaved",
                            });
                            receipt.source = match measured.source {
                                DevWindowSource::ArRouted => "ArRoutedK0",
                                DevWindowSource::Mtp => "Mtp",
                                DevWindowSource::Ngram => "Ngram",
                            };
                            let measured_mode = match measured.cost_mode {
                                hipfire_runtime::spec_economics::DerivedCostMode::OffPool => {
                                    CostMode::OffPool
                                }
                                hipfire_runtime::spec_economics::DerivedCostMode::OnCopy => {
                                    CostMode::OnCopy
                                }
                            };
                            let resolved_mode = CostMode::of_copy_lane(
                                hipfire_config::mtp_ngram_copy_enabled_for_arch(
                                    hipfire_arch_qwen4::ARCH_ID,
                                    &self.m.gpu.arch,
                                ),
                            );
                            if measured_mode != resolved_mode {
                                return Err(
                                    "measured production mode differs from resolved proposal mode"
                                        .into(),
                                );
                            }
                            if !measured.total_us.is_finite() || measured.total_us <= 0.0 {
                                return Err("invalid synchronized production total".into());
                            }
                            receipt.requested_confirmation = Some(measured.requested_confirmation);
                            receipt.head_confirmation = measured.head_confirmation;
                            receipt.variant = if measured.source == DevWindowSource::Ngram {
                                Some(if measured.requested_confirmation {
                                    "confirmed"
                                } else {
                                    "unconfirmed"
                                })
                            } else {
                                None
                            };
                            receipt.drafts = Some(measured.actual_drafts);
                            receipt.accepted = Some(measured.accepted_prefix);
                            receipt.committed = Some(measured.window.committed.len());
                            // Native margin evidence: one entry per actual native draft for
                            // Mtp (`actual_drafted` entries), `[]` otherwise. Length
                            // match is enforced below before any completion claim;
                            // clipped/foreign widths fail there as incomplete.
                            receipt.native_draft_margins =
                                if measured.source == DevWindowSource::Mtp {
                                    measured.native_draft_margins.clone()
                                } else {
                                    if !measured.native_draft_margins.is_empty() {
                                        return Err("non-Mtp window carried native margins".into());
                                    }
                                    Vec::new()
                                };
                            receipt.offer_available =
                                Some(source != DevWindowSource::Ngram || measured.offered >= k);
                            receipt.terminal = Some(measured.terminal);
                            receipt.terminal_reason =
                                measured.terminal.then_some("eos-or-end-of-turn");
                            receipt.rejected = Some(
                                measured.accepted_prefix < measured.actual_drafts
                                    && !measured.terminal,
                            );
                            receipt.terminal_censored = measured.terminal;
                            receipt.wall_ms = Some(measured.total_us / 1000.0);
                            if measured.source != source || measured.requested_drafts != k {
                                return Err("forced-window source/request mismatch".into());
                            }
                            if (k == 0 && measured.route != oracle::DevWindowRoute::TargetOnly)
                                || (k > 0 && measured.route != oracle::DevWindowRoute::Batched)
                            {
                                return Err(
                                    "unsupported actual production route for requested cost width"
                                        .into(),
                                );
                            }
                            if receipt.verify_route["arithmetic_reference"].is_null() {
                                return Err(
                                    "AR forecast reference unavailable before forced window".into(),
                                );
                            }
                            // After the MTP verify, the last actual reference must
                            // match the pre-window forecast; drift is raw
                            // incomplete evidence, never a priced window.
                            if measured.source == DevWindowSource::Mtp {
                                let actual = verify_reference_receipt(
                                    &self.m.bundle,
                                    measured.actual_drafts + 1,
                                );
                                if actual != receipt.verify_route["arithmetic_reference"] {
                                    return Err(format!(
                                        "AR reference drift: forecast {} but last actual {}",
                                        receipt.verify_route["arithmetic_reference"], actual
                                    ));
                                }
                            }
                            // Margin array must match the actual executed width: Mtp
                            // carries exactly `actual_drafted` entries, others none.
                            // A mismatch is raw incomplete evidence, never a pass.
                            let want_margins = if receipt.source == "Mtp" {
                                measured.actual_drafts
                            } else {
                                0
                            };
                            if receipt.native_draft_margins.len() != want_margins {
                                return Err(format!(
                                    "native margin length {} does not match actual drafted {} for {}",
                                    receipt.native_draft_margins.len(), measured.actual_drafts, receipt.source,
                                ));
                            }
                            receipt.complete = true;
                            if measured.actual_drafts != k {
                                return Err(format!(
                                    "requested {k} drafts but actual production executed {}",
                                    measured.actual_drafts,
                                ));
                            }
                            if receipt.terminal_censored {
                                return Err("terminal-censored window: not an uncensored requested-width price".into());
                            }
                            Ok(())
                        })();
                        if let Err(error) = measured {
                            receipt.complete = false;
                            receipt.missing_reason = Some(error.clone());
                            self.missing.push(json!({
                                "component": "production_window", "source": source_name,
                                "requested_k": k, "repeat": repeat, "warmup": warmup,
                                "reason": error,
                            }));
                            *stopped = json!("missing coverage; see individual receipts");
                        }
                        prod.push(receipt);
                    }
                    if source == DevWindowSource::Ngram {
                        // Same-S0 repetitions cannot fabricate an established
                        // pool takeover. Keep that missing variant explicit.
                        for (variant, confirmation) in [("confirmed", true), ("unconfirmed", false)]
                        {
                            if prod.iter().any(|w| {
                                w.source == "Ngram"
                                    && w.requested_k == k
                                    && w.variant == Some(variant)
                                    && !w.warmup
                                    && w.complete
                            }) {
                                continue;
                            }
                            let reason = format!(
                                "no real {variant} n-gram full-window observation at requested K={k}"
                            );
                            prod.push(ProdWindow {
                                requested_source: "Ngram",
                                request_id: None,
                                terminal: None,
                                terminal_reason: None,
                                rejected: None,
                                source: "Ngram",
                                variant: Some(variant),
                                requested_k: k,
                                position,
                                drafts: None,
                                accepted: None,
                                committed: None,
                                wall_ms: None,
                                requested_confirmation: Some(confirmation),
                                head_confirmation: None,
                                terminal_censored: false,
                                native_draft_margins: Vec::new(),
                                complete: false,
                                offer_available: None,
                                missing_reason: Some(reason.clone()),
                                warmup: false,
                                verify_route: Value::Null,
                            });
                            self.missing.push(json!({
                                "component": "production_window_variant",
                                "requested_k": k, "variant": variant, "reason": reason,
                            }));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn head_chains(
        &mut self,
        pending: &GpuTensor,
        seed: u32,
        position: usize,
        args: &Args,
        depths_json: &mut Vec<Value>,
        head_ms: &mut BTreeMap<usize, f64>,
    ) -> Result<(), String> {
        for &depth in &args.head_depths {
            if self.cfg.deadline.exceeded() {
                self.missing.push(json!({
                    "component": "head_chain",
                    "depth": depth,
                    "reason": "deadline",
                }));
                depths_json.push(json!({"depth": depth, "error": "deadline"}));
                continue;
            }
            let mut chain = Vec::new();
            let mut restore = Vec::new();
            let mut failure: Option<String> = None;
            for _ in 0..=self.iters {
                if self.cfg.deadline.exceeded() {
                    failure = Some("deadline during head repeats".into());
                    break;
                }
                let ticket = oracle::dev_head_open(&mut self.m.bundle, &mut self.m.gpu)?;
                self.sync()?;
                let start = Instant::now();
                let run = oracle::dev_head_chain(
                    &mut self.m.bundle,
                    &mut self.m.gpu,
                    seed,
                    pending,
                    position,
                    depth,
                );
                let synced = self.sync();
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                let restore_start = Instant::now();
                let restored =
                    oracle::dev_head_rollback(&mut self.m.bundle, &mut self.m.gpu, ticket);
                let restore_synced = self.sync();
                let restore_ms = restore_start.elapsed().as_secs_f64() * 1e3;
                restored?;
                restore_synced?;
                if let Err(error) = run.map(|_| ()).and(synced) {
                    failure = Some(error);
                    break;
                }
                chain.push(elapsed);
                restore.push(restore_ms);
            }
            if let Some(error) = failure {
                self.missing.push(json!({
                    "component": "head_chain",
                    "depth": depth,
                    "reason": error,
                }));
                depths_json.push(json!({"depth": depth, "error": error}));
                continue;
            }
            let (Some(chain_t), Some(restore_t)) =
                (Timed::from_samples(&chain), Timed::from_samples(&restore))
            else {
                return Err("--iters >= 1 is required".into());
            };
            head_ms.insert(depth, chain_t.median());
            depths_json.push(json!({
                "depth": depth,
                "step_chain_ms": chain_t.json(),
                "per_step_ms": profile::per_row_ms(chain_t.median(), depth),
                "restore_ms": restore_t.json(),
            }));
        }
        Ok(())
    }

    /// `--trace` window pair for one n: the AR reference steps (delimiter
    /// twice back to back, then `n` `forward_token`s) and the verify (one
    /// delimiter, then the production verify). Restore/rollback copies land
    /// between the last forward kernel of a window and the next delimiter.
    fn trace_n(&mut self, n: usize, block: &[u32], delimiter: &GpuTensor) -> Result<Value, String> {
        self.ensure_s0()?;
        let mut scratch = self
            .m
            .bundle
            .new_spec_scratch(&mut self.m.gpu, n)
            .map_err(|e| format!("new_spec_scratch({n}): {e}"))?;
        let armed = oracle::dev_would_arm(&self.m.bundle, n);
        let mut exec_ar = Value::Null;
        let mut exec_verify = Value::Null;
        let result = (|| -> Result<(), String> {
            // AR window.
            let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
            self.sync()?;
            exec_ar = self.exec_mode();
            let mut ar_error = None;
            let launched = (|| -> Result<(), String> {
                for _ in 0..2 {
                    self.m
                        .gpu
                        .logit_softcap_f32(delimiter, delimiter.numel(), 30.0)
                        .map_err(es)?;
                }
                for &token in block {
                    self.m
                        .bundle
                        .forward_token(&mut self.m.gpu, token, &self.row, None)
                        .map_err(es)?;
                }
                Ok(())
            })();
            if let Err(error) = launched {
                ar_error = Some(error);
            }
            let synced = self.sync();
            self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es)?;
            if let Some(error) = ar_error {
                return Err(format!("trace AR window: {error}"));
            }
            synced?;
            // Verify window.
            self.sync()?;
            exec_verify = self.exec_mode();
            self.m
                .gpu
                .logit_softcap_f32(delimiter, delimiter.numel(), 30.0)
                .map_err(es)?;
            oracle::dev_verify_resident(&mut self.m.bundle, &mut self.m.gpu, block, &mut *scratch)?;
            self.sync()?;
            oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, &mut *scratch)?;
            self.sync()
        })();
        scratch.free(&mut self.m.gpu);
        if result.is_err() {
            self.dirty = true;
        }
        result?;
        Ok(json!({
            "n": n,
            "row_capture_armed": armed,
            "execution": {"ar": exec_ar, "verify": exec_verify},
        }))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Block tokens, the AR reference and whether two AR passes agree bitwise.
/// One fixed-token AR pass covers every n (AR row r does not depend on the
/// block length) and is independent of the verify-width arm (AR rows are
/// single-row forwards), so it is computed once per prompt.
fn reference_pass(
    rig: &mut Rig,
    args: &Args,
    max_rows: usize,
    digest_counts: &BTreeSet<usize>,
) -> Result<(Vec<u32>, ArPass, bool), String> {
    match args.block {
        BlockMode::Greedy => {
            let feed = rig.ar_pass(None, max_rows, &BTreeSet::new())?;
            let block = feed.tokens.clone();
            let reference = rig.ar_pass(Some(&block), max_rows, digest_counts)?;
            let repeatable = feed
                .logits
                .iter()
                .zip(&reference.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            Ok((block, reference, repeatable))
        }
        BlockMode::Prompt => {
            let block = block_tokens(rig, args, max_rows)?;
            let reference = rig.ar_pass(Some(&block), max_rows, digest_counts)?;
            let second = rig.ar_pass(Some(&block), max_rows, &BTreeSet::new())?;
            let repeatable = second
                .logits
                .iter()
                .zip(&reference.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            Ok((block, reference, repeatable))
        }
    }
}

/// Warn when a `wideN` arm did not change the verify rows (non-gfx1151/1201).
fn note_arm(arm: &Arm, effective: usize) {
    let asked = arm
        .rows
        .map(|n| n.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS));
    match asked {
        Some(asked) if asked != effective => eprintln!(
            "qwen4_wide_verify: WARNING: arm {} asked for {asked} verify rows but the arch resolves {effective} (override applies on gfx1151/gfx1201 only)",
            arm.name
        ),
        _ => eprintln!(
            "qwen4_wide_verify: arm {}: verify rows {effective}",
            arm.name
        ),
    }
}

fn run() -> Result<bool, String> {
    let started = Instant::now();
    let args = parse_args()?;
    // Source-planned preload probe: metadata-only headers/index open, bare
    // GPU init, ordinary loader config (`production::preload_plan`); no
    // `production::load`, no `hw_identity` binary/source hashes, no capture64
    // arming, no prompts/windows/timing. Runs BEFORE `--resource-plan-only`
    // (post-load RAW, untouched below). A refused OUT path never overwrites
    // an existing file: the honest publication helper claims `--out` first.
    if args.preload_plan_only {
        let raw = production::preload_plan(&args)?;
        if std::fs::symlink_metadata(&args.out).is_ok() {
            return Err(format!(
                "refusing to overwrite existing output {}",
                args.out.display()
            ));
        }
        std::fs::write(&args.out, serde_json::to_string_pretty(&raw).map_err(es)?)
            .map_err(|e| format!("write {}: {e}", args.out.display()))?;
        eprintln!("qwen4_wide_verify: wrote {}", args.out.display());
        return Ok(true);
    }
    // Load-only allocation-plan probe: real production load, then the
    // loader's already-admitted 64-row allocation and typed snapshot verbatim
    // as the exact E raw shape; nothing is armed or grown post-load. Before
    // normal prompt/row/arm loops; normal modes below are unchanged (the
    // balanced cost/strict pool close and rollback paths included). A refused
    // OUT path never overwrites an existing file: the honest publication
    // helper claims `--out` first.
    if args.resource_plan_only {
        let raw = production::resource_plan(&args)?;
        if std::fs::symlink_metadata(&args.out).is_ok() {
            return Err(format!(
                "refusing to overwrite existing output {}",
                args.out.display()
            ));
        }
        std::fs::write(&args.out, serde_json::to_string_pretty(&raw).map_err(es)?)
            .map_err(|e| format!("write {}: {e}", args.out.display()))?;
        eprintln!("qwen4_wide_verify: wrote {}", args.out.display());
        return Ok(true);
    }
    let profile = args.profile;
    let deadline = Deadline::new(started, args.deadline_secs);
    let max_rows = *args.rows.iter().max().expect("rows is non-empty");
    // A full next window after every accepted prefix, plus its next token.
    let ref_rows = if profile == Profile::Strict {
        2 * max_rows + 1
    } else {
        max_rows
    };
    let mut prompt_texts = Vec::new();
    for path in &args.prompt_files {
        prompt_texts.push(
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?,
        );
    }

    // The default load is the production serve load; a fixed hot budget is the
    // diagnostic manual loader and is labelled as such in the JSON.
    let mut m = if args.hot_budget_mib.is_some() {
        Model::load(
            &args.model,
            args.n_ctx,
            args.hot_budget_mib,
            args.hotset.as_deref(),
        )?
    } else {
        production::load(&args)?
    };
    let prompts: Vec<Vec<u32>> = prompt_texts.iter().map(|t| m.tokenizer.encode(t)).collect();
    let head_span = args.head_depths.iter().max().map_or(0, |depth| depth + 1);
    let prod_span = if args.prod_windows > 0 {
        args.prod_ks.iter().map(|k| k + 1).max().unwrap_or(0)
    } else {
        0
    };
    let context_need = ref_rows.max(head_span).max(prod_span) + args.max_k + 1;
    for (path, prompt) in args.prompt_files.iter().zip(&prompts) {
        if prompt.is_empty() {
            return Err(format!("{} encodes to zero tokens", path.display()));
        }
        if prompt.len() + context_need > args.n_ctx {
            return Err(format!(
                "{}: {} prompt tokens + {context_need} rows (verify block, head chain, production windows) exceeds --n-ctx {}",
                path.display(),
                prompt.len(),
                args.n_ctx
            ));
        }
    }
    let copy_lane =
        hipfire_config::mtp_ngram_copy_enabled_for_arch(hipfire_arch_qwen4::ARCH_ID, &m.gpu.arch);
    let cost_mode = CostMode::of_copy_lane(copy_lane);
    if let Some(declared) = args.cost_mode {
        if declared != cost_mode {
            return Err(format!(
                "--cost-mode {} but the n-gram copy lane resolves {} on {} (set HIPFIRE_MTP_NGRAM_COPY accordingly): costs of one mode are never reused for the other",
                declared.key(),
                if copy_lane { "on" } else { "off" },
                m.gpu.arch
            ));
        }
    }
    let vocab = m.vocab;
    let ring_supported = oracle::dev_row_capture_supported(&m.bundle, &m.gpu);
    if args.row_capture {
        oracle::dev_arm_row_capture(&mut m.bundle, &mut m.gpu, max_rows)?;
    }
    let ring_rows = oracle::dev_row_capture_rows(&m.bundle);
    if !args.row_capture && ring_rows != 0 {
        return Err("--row-capture off requested but a capture ring is already allocated".into());
    }
    eprintln!(
        "qwen4_wide_verify: GDN row-capture ring {ring_rows} rows (route supports capture: {ring_supported})"
    );
    // Size the bundle's spec hidden rows once for the widest n, as production
    // MTP does (`new_spec_scratch(max_k + 1)`): without an NPU pipelined trunk
    // the first (n = 1) scratch would otherwise pin the capacity at one row.
    let sizing = m
        .bundle
        .new_spec_scratch(&mut m.gpu, max_rows)
        .map_err(|e| format!("new_spec_scratch({max_rows}): {e}"))?;
    sizing.free(&mut m.gpu);
    let prompt_logits = m.gpu.zeros(&[vocab], DType::F32).map_err(es)?;
    let row = m.gpu.zeros(&[vocab], DType::F32).map_err(es)?;
    let mut rig = Rig {
        m,
        prompt: Vec::new(),
        prompt_logits,
        row,
        seed_logits: Vec::new(),
        d0: None,
        dirty: false,
        p0: 0,
        reprefills: 0,
        rewind_checked: BTreeSet::new(),
        iters: args.iters,
        cfg: RunCfg {
            profile,
            commit: args.commit,
            commit_iters: args.commit_iters,
            baseline_prefill: args.baseline_prefill,
            deadline,
        },
        qualified: 0,
        requested_arm_rows: None,
        strict_candidate_rows: args
            .arms
            .iter()
            .filter_map(|arm| arm.rows)
            .max()
            .unwrap_or(0),
        kernel_seen: BTreeSet::new(),
        pending_launch_witnesses: Vec::new(),
        missing: Vec::new(),
    };

    let hipfire_env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .collect();
    let block_mode = if args.block == BlockMode::Greedy {
        "greedy"
    } else {
        "prompt"
    };
    let model_path = args.model.display().to_string();
    let arm_names: Vec<&str> = args.arms.iter().map(|a| a.name.as_str()).collect();
    let prompt_paths: Vec<String> = args
        .prompt_files
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    eprintln!("qwen4_wide_verify: HIPFIRE_* env: {hipfire_env:?}");
    let mut identity = json!({
        "gpu_arch": rig.m.info.arch,
        "uma": rig.m.info.uma,
        "ple": rig.m.info.ple_line,
        "ple_note": rig.m.info.ple_note,
        "ple_rows_source": rig.m.info.ple_rows_source,
        "trunk_overlay_applied": rig.m.info.overlay_applied,
        "expert_vram_layers": rig.m.info.expert_vram_layers,
        "hot_budget_mib": args.hot_budget_mib,
        "hotset": args.hotset.as_ref().map(|p| p.display().to_string()),
        "offload_exec": rig.m.info.offload_exec.to_string(),
        "partition_digest": rig.m.info.partition_digest,
        "split_armed": rig.m.info.split_armed,
        "split_digest": rig.m.bundle.split_digest(),
        "model_path": model_path,
        "model_bytes": rig.m.info.model_bytes,
        "hipfire_env": hipfire_env,
        "prompt_files": prompt_paths,
        "vocab": vocab,
        "n_ctx": args.n_ctx,
        "iters": args.iters,
        "block_mode": block_mode,
        "accepts": args.accepts.name(),
        "arms": arm_names,
        "rows": args.rows,
        "row_capture_supported": ring_supported,
        "row_capture_ring_rows": ring_rows,
        "row_capture_requested": args.row_capture,
        "accepted_prefix_semantics": "accept_len in 0..n accepted drafts; seed-inclusive consumed prefix lengths 1..=n",
        "verify_route": "Qwen4Bundle::verify_block_rows via SpecTarget (spec_forward_rows, prefill_scope=false); timing uses the resident (no pick readback) variant",
        "loader": rig.m.info.loader,
        "production": rig.m.info.production,
        "profile": profile.name(),
        "deadline_secs": args.deadline_secs,
        "max_k": args.max_k,
        "head_depths": args.head_depths,
        "commit": args.commit,
        "commit_iters": args.commit_iters,
        "baseline_prefill": args.baseline_prefill,
        "prod_ks": args.prod_ks,
        "prod_windows": args.prod_windows,
    });

    let prompt_header = |index: usize, tokens: &[u32]| -> Value {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        json!({
            "prompt_index": index,
            "prompt_file": args.prompt_files[index].display().to_string(),
            "prompt_tokens": tokens.len(),
            "prompt_tokens_sha256": format!("{:x}", Sha256::digest(&bytes)),
            "context": {
                "position": tokens.len(),
                "bucket": profile::context_bucket(tokens.len()),
                "n_ctx": args.n_ctx,
            },
        })
    };

    // -- trace mode ---------------------------------------------------------
    if args.trace {
        let delimiter = rig.m.gpu.zeros(&[256], DType::F32).map_err(es)?;
        eprintln!(
            "qwen4_wide_verify: trace delimiter kernel: {DELIMITER_KERNEL} (256 f32, 1 block, cap 30.0); per prompt x arm x n: AR window = 2 back-to-back delimiters then n forward_token; verify window = 1 delimiter then the verify"
        );
        // Compile the delimiter kernel outside every window.
        rig.m
            .gpu
            .logit_softcap_f32(&delimiter, delimiter.numel(), 30.0)
            .map_err(es)?;
        rig.sync()?;
        let mut windows = Vec::new();
        let mut prompt_records = Vec::new();
        let mut failed = false;
        for (pi, prompt) in prompts.iter().enumerate() {
            rig.begin_prompt(prompt.clone(), false)?;
            let block = block_tokens(&mut rig, &args, max_rows)?;
            let mut header = prompt_header(pi, prompt);
            header["block_tokens"] = json!(block);
            prompt_records.push(header);
            for arm in &args.arms {
                let effective = rig.begin_arm(arm);
                note_arm(arm, effective);
                for &n in &args.rows {
                    let mut label = json!({
                        "prompt_index": pi,
                        "prompt_tokens": prompt.len(),
                        "arm": arm.name,
                        "effective_verify_rows": effective,
                    });
                    match rig.trace_n(n, &block[..n], &delimiter) {
                        Ok(window) => {
                            println!(
                                "trace prompt {pi} arm {} rows {n}: armed {}",
                                arm.name, window["row_capture_armed"]
                            );
                            for (k, v) in window.as_object().expect("window is an object") {
                                label[k] = v.clone();
                            }
                        }
                        Err(error) => {
                            eprintln!(
                                "qwen4_wide_verify: trace prompt {pi} arm {} rows {n}: {error}",
                                arm.name
                            );
                            label["n"] = json!(n);
                            label["error"] = json!(error);
                            failed = true;
                        }
                    }
                    windows.push(label);
                }
            }
        }
        rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(None);
        identity["mode"] = json!("trace");
        identity["delimiter_kernel"] = json!(DELIMITER_KERNEL);
        identity["trace_pattern"] = json!({
            "delimiter_kernel": DELIMITER_KERNEL,
            "order": "for prompt in prompts { for arm in arms { for n in rows { AR window, then verify window } } }; each trace_windows entry is one (prompt, arm, n) in this order",
            "ar_window": "DELIM, DELIM (back to back), then n forward_token calls",
            "verify_window": "DELIM once, then the production verify (resident, no pick readback)",
            "label_rule": "a delimiter run of length 2 starts 'ar n=N'; a run of length 1 starts 'verify n=N'; windows are consecutive: ar, verify, ar, verify, ...",
            "between_windows": "snapshot/restore and rollback copies sit between the last forward kernel of a window and the next delimiter; they belong to no forward",
            "windows_per_entry": 2,
        });
        let report = json!({
            "identity": identity,
            "prompts": prompt_records,
            "trace_windows": windows,
        });
        std::fs::write(
            &args.out,
            serde_json::to_string_pretty(&report).map_err(es)?,
        )
        .map_err(|e| format!("write {}: {e}", args.out.display()))?;
        return Ok(!failed);
    }

    // -- accuracy + timing mode ---------------------------------------------
    let mut digest_counts: BTreeSet<usize> = BTreeSet::new();
    let want_state_refs =
        args.accepts != Accepts::None || (profile == Profile::Cost && args.commit);
    if want_state_refs {
        digest_counts.insert(1);
        for &n in &args.rows {
            if args.accepts == Accepts::All {
                digest_counts.extend(1..=n);
            } else {
                digest_counts.insert(n);
            }
        }
    }
    if profile == Profile::Strict {
        digest_counts.extend(1..=ref_rows);
    }
    let mut ok = true;
    let mut prompt_records = Vec::new();
    let mut actual_windows = Vec::new();
    for (pi, prompt) in prompts.iter().enumerate() {
        if deadline.exceeded() {
            rig.missing.push(json!({
                "prompt_index": pi,
                "component": "prompt",
                "reason": "deadline exceeded before this prompt",
            }));
            ok = false;
            break;
        }
        rig.begin_prompt(prompt.clone(), true)?;
        let (block, ar, ar_repeatable) = reference_pass(&mut rig, &args, ref_rows, &digest_counts)?;
        if !ar_repeatable {
            eprintln!(
                "qwen4_wide_verify: WARNING: two AR passes over the same block differ bitwise"
            );
        }
        eprintln!("qwen4_wide_verify: prompt {pi} block tokens {block:?}");
        let mut arm_runs: Vec<ArmRun> = Vec::new();
        for arm in &args.arms {
            let effective = rig.begin_arm(arm);
            note_arm(arm, effective);
            let caps = bundle_caps(&rig);
            let reprefills_before = rig.reprefills;
            let mut records: Vec<Value> = Vec::new();
            let mut costs: Vec<WidthCost> = Vec::new();
            let mut cells: Vec<(usize, bool)> = Vec::new();
            for &n in &args.rows {
                let tag = format!("[prompt {pi} tokens {} arm {}]", prompt.len(), arm.name);
                if deadline.exceeded() {
                    println!("{tag} rows {n}: ERROR deadline exceeded before this width");
                    rig.missing.push(json!({
                        "prompt_index": pi,
                        "arm": arm.name,
                        "component": "verify_width",
                        "n": n,
                        "reason": "deadline exceeded before this width",
                    }));
                    records.push(json!({
                        "n": n,
                        "arm": arm.name,
                        "prompt_index": pi,
                        "error": "deadline exceeded before this width",
                    }));
                    cells.push((n, false));
                    ok = false;
                    continue;
                }
                match rig.run_n(n, &block[..n], &ar, args.accepts) {
                    Ok(outcome) => {
                        println!("{tag} {}", outcome.line);
                        if rig.hard(n) && !outcome.clean {
                            ok = false;
                        }
                        cells.push((n, outcome.clean));
                        if let Some(cost) = outcome.cost {
                            costs.push(cost);
                        }
                        let mut record = outcome.record;
                        if profile == Profile::Strict && record["gpu_span_ms"].is_null() {
                            rig.missing.push(json!({
                                "prompt_index": pi,
                                "arm": arm.name,
                                "component": "gpu_span_ms",
                                "n": n,
                                "reason": record["gpu_kernel_timing"]["missing_reason"],
                            }));
                            ok = false;
                        }
                        record["arm"] = json!(arm.name);
                        record["prompt_index"] = json!(pi);
                        records.push(record);
                    }
                    Err(error) => {
                        println!("{tag} rows {n}: ERROR {error}");
                        rig.dirty = true;
                        if profile != Profile::Oracle || n <= FEW_ROW_ROUTE_ROWS {
                            ok = false;
                        }
                        if profile != Profile::Oracle {
                            rig.missing.push(json!({
                                "prompt_index": pi,
                                "arm": arm.name,
                                "component": "verify_width",
                                "n": n,
                                "reason": error,
                            }));
                        }
                        cells.push((n, false));
                        records.push(json!({
                            "n": n,
                            "arm": arm.name,
                            "prompt_index": pi,
                            "error": error,
                        }));
                    }
                }
            }
            // The commit boundary after every width's verify / rollback / AR
            // timing, so a deadline never costs a requested width.
            if profile == Profile::Cost && args.commit {
                for index in 0..args.rows.len() {
                    let n = args.rows[index];
                    if records[index].get("error").is_some() {
                        continue;
                    }
                    if deadline.exceeded() {
                        rig.missing.push(json!({
                            "prompt_index": pi,
                            "arm": arm.name,
                            "component": "commit",
                            "n": n,
                            "reason": "deadline exceeded before this width's commit",
                        }));
                        records[index]["commit_ms"] =
                            json!({"skipped": "deadline exceeded before this width's commit"});
                        continue;
                    }
                    match rig.run_commit(n, &block[..n], &ar) {
                        Ok((commit_json, medians, clean)) => {
                            records[index]["commit_ms"] = commit_json;
                            if let Some(cost) = costs.iter_mut().find(|c| c.n == n) {
                                cost.commit_ms = medians;
                            }
                            if rig.hard(n) && !clean {
                                ok = false;
                            }
                        }
                        Err(error) => {
                            rig.dirty = true;
                            rig.missing.push(json!({
                                "prompt_index": pi,
                                "arm": arm.name,
                                "component": "commit",
                                "n": n,
                                "reason": error,
                            }));
                            records[index]["commit_ms"] = json!({"error": error});
                            ok = false;
                        }
                    }
                }
            }
            arm_runs.push(ArmRun {
                arm: arm.clone(),
                effective,
                caps,
                reprefills: rig.reprefills - reprefills_before,
                checked_kinds: rig.rewind_checked.iter().copied().collect(),
                records,
                costs,
                cells,
            });
        }
        // Full production windows use the shipping base route, never whichever
        // diagnostic/strict override happened to run last.
        rig.begin_arm(&Arm::base());
        // Head chains and real production windows (cost profile).
        let (head_json, head_ms, prod) = if profile == Profile::Cost {
            if deadline.exceeded() {
                rig.missing.push(json!({
                    "prompt_index": pi,
                    "component": "head",
                    "reason": "deadline exceeded before the head phase",
                }));
                ok = false;
                (
                    json!({"measured": false, "reason": "deadline"}),
                    BTreeMap::new(),
                    Vec::new(),
                )
            } else {
                match rig.head_and_production(pi, prompt, &args, copy_lane) {
                    Ok(result) => result,
                    Err(error) => {
                        rig.missing.push(json!({
                            "prompt_index": pi,
                            "component": "head",
                            "reason": error,
                        }));
                        ok = false;
                        (
                            json!({"measured": false, "error": error}),
                            BTreeMap::new(),
                            Vec::new(),
                        )
                    }
                }
            }
        } else {
            (Value::Null, BTreeMap::new(), Vec::new())
        };
        let (prod_rows, prod_medians) = profile::aggregate_production_windows(&prod);
        let mut arm_records = Vec::new();
        for run in arm_runs {
            let mut arm_json = json!({
                "arm": run.arm.name,
                "override_rows": run.arm.rows,
                "effective_verify_rows": run.effective,
                "bundle_caps": run.caps,
                "reprefills": run.reprefills,
                "checked_ticket_kinds": run.checked_kinds,
                "records": run.records,
            });
            if profile != Profile::Oracle {
                arm_json["widths"] = arm_json["records"].clone();
            }
            if profile == Profile::Cost {
                arm_json["windows"] = json!(profile::compose_windows(
                    &run.costs,
                    &head_ms,
                    &prod_medians
                ));
            }
            if profile == Profile::Strict {
                arm_json["strict"] = profile::strict_summary(&run.cells);
            }
            arm_records.push(arm_json);
        }
        let mut header = prompt_header(pi, prompt);
        header["block_tokens"] = json!(block);
        header["ar_repeatable_bitwise"] = json!(ar_repeatable);
        header["post_prefill_state_parts"] = json!(rig.d0.as_ref().map_or(0, Vec::len));
        header["ar_reference_digest_counts"] = json!(digest_counts);
        header["reprefill_count"] = json!(rig.reprefills);
        header["arms"] = json!(arm_records);
        if profile == Profile::Cost {
            header["head"] = head_json;
            actual_windows.extend(
                prod.iter()
                    .map(|window| window.actual_json(pi, &header["context"])),
            );
            header["production_windows"] =
                json!(prod.iter().map(ProdWindow::json).collect::<Vec<_>>());
            header["production_by_width"] = json!(prod_rows);
            header["ar_baseline"] = json!({
                "role": "baseline only: n plain Qwen4Bundle::forward_token steps (no MTP head sync / append / controller)",
                "k0_total": "production_windows source ArRoutedK0: forced real AR-routed transaction, not n plain AR forwards",
            });
        }
        prompt_records.push(header);
    }
    rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(None);

    identity["mode"] = json!(if profile == Profile::Oracle {
        "oracle"
    } else {
        profile.name()
    });
    identity["rewind"] = json!({
        "ar_passes_and_timed_arms": "snapshot/restore ticket (Qwen4Bundle::snapshot / restore)",
        "verify_arms": "the verify's own snapshot ticket (rolled back without commit)",
        "accept_arms": "re-prefill (prefill_final); a committed state has no ticket (none with --accepts none)",
        "re_prefill_checked_against_post_prefill_digest": true,
    });
    identity["shared_ar_reference"] = json!("one AR reference per prompt, shared by every arm (AR rows are single-row forwards, independent of the verify-width arm); AR timing is re-measured per arm and n");
    let complete = rig.missing.is_empty();
    let ok = ok && (profile == Profile::Oracle || complete);
    let report = match profile.schema() {
        None => json!({"identity": identity, "prompts": prompt_records, "ok": ok}),
        Some(schema) => json!({
            "schema": schema,
            "mode": profile.name(),
            "complete": complete,
            "ok": ok,
            "missing": rig.missing,
            "deadline": deadline.json(),
            "proposal_mode": {
                "key": cost_mode.key(),
                "copy_lane_resolved": copy_lane,
                "declared": args.cost_mode.map(CostMode::key),
                "card": profile::card_of_arch(&rig.m.info.arch),
                "env": {"HIPFIRE_MTP_NGRAM_COPY": std::env::var("HIPFIRE_MTP_NGRAM_COPY").ok()},
            },
            "caps": {
                "wide_verify_max_rows": WIDE_VERIFY_MAX_ROWS,
                "few_row_route_rows": FEW_ROW_ROUTE_ROWS,
                "requested_widths": args.rows,
                "note": "per-arm `bundle_caps` hold the effective Qwen4Bundle caps and `effective_verify_rows` the qualified row-exact rows; widths above the qualified rows are timing-only unless --strict-qualify proves them",
            },
            "identity": identity,
            "prompts": prompt_records,
        }),
    };
    if profile == Profile::Cost {
        let sidecar_path = args.actual_windows_out.clone().unwrap_or_else(|| {
            let mut path = args.out.as_os_str().to_os_string();
            path.push(".actual_windows.json");
            PathBuf::from(path)
        });
        if sidecar_path == args.out {
            return Err("--actual-windows-out must differ from --out".into());
        }
        let sidecar = json!({
            "schema": profile::ACTUAL_WINDOWS_SCHEMA,
            "identity": report["identity"],
            "proposal_mode": report["proposal_mode"],
            "timing_only": true,
            "qualification_claim": false,
            "complete": complete,
            "missing": report["missing"],
            "windows": actual_windows,
        });
        std::fs::write(
            &sidecar_path,
            serde_json::to_string_pretty(&sidecar).map_err(es)?,
        )
        .map_err(|e| format!("write {}: {e}", sidecar_path.display()))?;
    }
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&report).map_err(es)?,
    )
    .map_err(|e| format!("write {}: {e}", args.out.display()))?;
    eprintln!("qwen4_wide_verify: wrote {}", args.out.display());
    Ok(ok)
}

/// Effective `Qwen4Bundle` verify caps (not just the arch table / override).
fn bundle_caps(rig: &Rig) -> Value {
    let caps = oracle::dev_verify_caps(&rig.m.bundle, &rig.m.gpu);
    json!({
        "spec_verify_rows_cap": caps.spec_verify_rows_cap,
        "forward_max_chunk": caps.forward_max_chunk,
        "row_exact_verify_rows": caps.row_exact_verify_rows,
        "requested_arm_rows": rig.requested_arm_rows,
        "effective_row_exact_rows": oracle::dev_effective_row_exact_rows(&rig.m.bundle, &rig.m.gpu),
    })
}

/// The block tokens: `greedy` = the AR greedy continuation from the prefill
/// (block[0] = argmax of the prefill's final row); `prompt` = the last
/// `max_rows` prompt tokens (cycled when the prompt is shorter).
fn block_tokens(rig: &mut Rig, args: &Args, max_rows: usize) -> Result<Vec<u32>, String> {
    match args.block {
        BlockMode::Greedy => Ok(rig.ar_pass(None, max_rows, &BTreeSet::new())?.tokens),
        BlockMode::Prompt => {
            let len = rig.prompt.len();
            Ok((0..max_rows)
                .map(|i| {
                    if len >= max_rows {
                        rig.prompt[len - max_rows + i]
                    } else {
                        rig.prompt[i % len]
                    }
                })
                .collect())
        }
    }
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("qwen4_wide_verify: FAILED the n <= 8 contract (or a run error)");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("qwen4_wide_verify: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_rewind_checks_preserve_only_completed_reference_checks() {
        let mut checked = BTreeSet::from(["ar_ticket", "verify_ticket"]);
        begin_arm_rewind_checks(&mut checked);
        assert_eq!(checked, BTreeSet::from(["ar_ticket"]));
        let mut unchecked = BTreeSet::from(["verify_ticket"]);
        begin_arm_rewind_checks(&mut unchecked);
        assert!(unchecked.is_empty(), "never invent an AR rewind check");
    }

    #[test]
    fn strict_lower_explicit_arm_is_diagnostic_only_beside_a_larger_arm() {
        assert!(strict_arm_is_diagnostic(Some(8), 64));
        assert!(strict_arm_is_diagnostic(Some(32), 64));
        assert!(!strict_arm_is_diagnostic(Some(64), 64));
        assert!(!strict_arm_is_diagnostic(Some(8), 8));
        assert!(!strict_arm_is_diagnostic(None, 64));
        assert!(!strict_arm_is_diagnostic(None, 0));
    }

    #[test]
    fn strict_candidate_policy_is_independent_of_arm_order() {
        for spec in ["wide8,wide64", "wide64,wide8"] {
            let arms = parse_arms(spec).unwrap();
            let candidate = arms.iter().filter_map(|arm| arm.rows).max().unwrap();
            assert_eq!(candidate, 64);
            for arm in arms {
                assert_eq!(
                    strict_arm_is_diagnostic(arm.rows, candidate),
                    arm.rows == Some(8)
                );
            }
        }
    }

    #[test]
    fn strict_candidate_caps_cannot_silently_shrink_to_the_baseline() {
        for n in 1..=64 {
            assert_eq!(strict_required_rows(Some(8), 64, n), 8);
            assert_eq!(strict_required_rows(Some(64), 64, n), 64);
            assert_eq!(strict_required_rows(None, 64, n), n);
        }
        assert_eq!(strict_required_rows(Some(8), 8, 64), 64);
    }
}

#[cfg(test)]
mod cost_profile_cli_tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Args, String> {
        let argv: Vec<String> = [
            "--model",
            "m.hfq",
            "--prompt-file",
            "p.txt",
            "--out",
            "o.json",
        ]
        .into_iter()
        .chain(extra.iter().copied())
        .map(str::to_owned)
        .collect();
        parse_args_from(&argv)
    }

    #[test]
    fn legacy_oracle_keeps_its_defaults_and_requires_rows() {
        assert!(parse(&[]).is_err());
        let args = parse(&["--rows", "1,2"]).unwrap();
        assert_eq!(args.profile, Profile::Oracle);
        assert_eq!(args.accepts, Accepts::Ends);
        assert_eq!(args.max_k, profile::ORACLE_MAX_K);
        assert!(args.head_depths.is_empty());
        assert_eq!(args.prod_windows, 0);
        assert!(!args.baseline_prefill);
        assert!(args.deadline_secs > 1_000_000);
    }

    #[test]
    fn cost_curve_defaults_cover_the_requested_curve() {
        let args = parse(&["--cost-curve"]).unwrap();
        assert_eq!(args.profile, Profile::Cost);
        let mut rows: Vec<usize> = (1..=16).collect();
        rows.extend([32, 64]);
        assert_eq!(args.rows, rows);
        assert_eq!(args.accepts, Accepts::None);
        assert_eq!(args.arms.len(), 1);
        assert_eq!(args.arms[0].name, "base");
        let mut depths: Vec<usize> = (1..=16).collect();
        depths.extend([31, 63]);
        assert_eq!(args.head_depths, depths);
        assert_eq!(args.max_k, 63);
        assert!(args.commit);
        assert_eq!(args.commit_iters, 3);
        assert_eq!(args.deadline_secs, profile::DEFAULT_DEADLINE_SECS);
        assert_eq!(
            args.prod_ks,
            args.rows.iter().map(|n| n - 1).collect::<Vec<_>>()
        );
        assert_eq!(args.prod_windows, 24);
        assert_eq!(args.cost_mode, None);
    }

    #[test]
    fn cost_curve_accepts_the_fabric_command_line() {
        let args = parse(&[
            "--cost-curve",
            "--cost-mode",
            "OnCopy",
            "--rows",
            "1-16,32,64",
            "--head-depths",
            "1-16,31,63",
            "--commit",
            "on",
            "--baseline-prefill",
            "on",
            "--deadline-secs",
            "1700",
        ])
        .unwrap();
        assert_eq!(args.cost_mode, Some(CostMode::OnCopy));
        assert!(args.baseline_prefill);
        assert_eq!(args.deadline_secs, 1700);
        assert_eq!(*args.rows.last().unwrap(), 64);
        assert!(parse(&["--cost-curve", "--head-depths", "off"])
            .unwrap()
            .head_depths
            .is_empty());
    }

    #[test]
    fn profiles_are_exclusive_and_fail_closed() {
        assert!(parse(&["--cost-curve", "--strict-qualify"]).is_err());
        assert!(parse(&["--cost-curve", "--trace"]).is_err());
        assert!(parse(&["--strict-qualify", "--trace"]).is_err());
        assert!(parse(&["--cost-curve", "--accepts", "ends"]).is_err());
        assert!(parse(&["--cost-curve", "--accepts", "none"]).is_ok());
        assert!(parse(&["--strict-qualify", "--accepts", "none"]).is_err());
        assert!(parse(&["--cost-curve", "--deadline-secs", "1800"]).is_err());
        assert!(parse(&["--cost-curve", "--deadline-secs", "0"]).is_err());
        assert!(parse(&["--cost-curve", "--cost-mode", "copy"]).is_err());
        assert!(parse(&["--cost-curve", "--commit", "maybe"]).is_err());
        assert!(parse(&["--cost-curve", "--max-k", "2", "--prod-ks", "3"]).is_err());
        assert!(parse(&["--cost-curve", "--max-k", "0"]).is_err());
        // Cost-only flags never apply to the other profiles.
        for flag in [
            ["--head-depths", "1-4"],
            ["--commit", "on"],
            ["--commit-iters", "2"],
            ["--cost-mode", "OffPool"],
            ["--prod-ks", "3"],
            ["--prod-windows", "4"],
        ] {
            assert!(
                parse(&["--rows", "1", flag[0], flag[1]]).is_err(),
                "{flag:?}"
            );
            assert!(
                parse(&["--strict-qualify", flag[0], flag[1]]).is_err(),
                "{flag:?}"
            );
        }
        assert!(parse(&["--rows", "1", "--baseline-prefill", "on"]).is_err());
    }

    #[test]
    fn strict_qualify_defaults_to_every_row_with_the_baseline_arm() {
        let args = parse(&["--strict-qualify"]).unwrap();
        assert_eq!(args.profile, Profile::Strict);
        assert_eq!(args.rows, (1..=64).collect::<Vec<_>>());
        assert_eq!(args.accepts, Accepts::Ends);
        assert!(args.baseline_prefill);
        assert_eq!(args.max_k, 63);
        assert!(!args.commit);
        assert_eq!(args.prod_windows, 0);
        let all = parse(&[
            "--strict-qualify",
            "--accepts",
            "all",
            "--baseline-prefill",
            "off",
        ])
        .unwrap();
        assert_eq!(all.accepts, Accepts::All);
        assert!(!all.baseline_prefill);
    }

    #[test]
    fn cost_cli_reserves_physical_heads_and_prices_requested_widths() {
        let args = parse(&[
            "--cost-curve",
            "--rows",
            "1,16,32,64",
            "--prod-ks",
            "0,15,31,63",
            "--prod-windows",
            "1",
            "--actual-windows-out",
            "actual.json",
        ])
        .unwrap();
        assert_eq!(args.prod_ks, vec![0, 15, 31, 63]);
        assert_eq!(args.actual_windows_out, Some(PathBuf::from("actual.json")));
        assert!(parse(&["--cost-curve", "--head-depths", "64"]).is_err());
        assert!(parse(&["--cost-curve", "--max-k", "64"]).is_err());
        assert!(parse(&["--cost-curve", "--max-k", "31"]).is_err());
        assert!(parse(&["--cost-curve", "--rows", "65"]).is_err());
        assert!(parse(&["--rows", "1", "--actual-windows-out", "x"]).is_err());
    }

    #[test]
    fn resource_plan_only_is_load_only_and_keeps_run_shape_unused() {
        let args = parse(&["--resource-plan-only"]).unwrap();
        assert!(args.resource_plan_only);
        assert_eq!(args.rows, vec![64]);
        assert!(args.prompt_files.is_empty());
        assert!(parse(&["--resource-plan-only", "--hot-budget-mib", "1"]).is_err());
        assert!(parse(&["--resource-plan-only", "--cost-curve"]).is_err());
        assert!(parse(&["--resource-plan-only", "--strict-qualify"]).is_err());
        assert!(parse(&["--resource-plan-only", "--trace"]).is_err());
        assert!(parse(&["--resource-plan-only", "--commit", "on"]).is_err());
        assert!(parse(&["--resource-plan-only", "--prod-ks", "1"]).is_err());
    }

    #[test]
    fn preload_plan_only_is_metadata_only_with_finite_544_planning() {
        // Real `usize` bench constant, not a config string: explicit
        // `--n-ctx 544` is honored, absent defaults to the finite 544 bench,
        // never the oracle 8192.
        let planned = parse(&["--preload-plan-only"]).unwrap();
        assert!(planned.preload_plan_only);
        assert!(!planned.resource_plan_only);
        assert!(!planned.n_ctx_given);
        assert_eq!(planned.n_ctx, 544);
        assert_eq!(planned.n_ctx, 512usize.max(128 + 1) + 32);
        assert!(planned.prompt_files.is_empty());
        let explicit = parse(&["--preload-plan-only", "--n-ctx", "544"]).unwrap();
        assert!(explicit.n_ctx_given);
        assert_eq!(explicit.n_ctx, 544);
        // `--max-k` tracks explicitly like `--n-ctx`: absent stays runtime
        // auto for the preload ask, explicit preserves the caller's value.
        assert!(!planned.max_k_given);
        let explicit_k = parse(&["--preload-plan-only", "--max-k", "7"]).unwrap();
        assert!(explicit_k.max_k_given);
        assert_eq!(explicit_k.max_k, 7);
        let wide = parse(&["--preload-plan-only", "--n-ctx", "1024"]).unwrap();
        assert!(wide.n_ctx_given);
        assert_eq!(wide.n_ctx, 1024);
        // Branch isolation lives here: forced oracle lines refuse planning.
        // Pairwise exclusivity with the post-load probe is its own test.
        assert!(parse(&["--preload-plan-only", "--cost-curve"]).is_err());
        assert!(parse(&["--preload-plan-only", "--strict-qualify"]).is_err());
        assert!(parse(&["--preload-plan-only", "--trace"]).is_err());
        assert!(parse(&["--preload-plan-only", "--hot-budget-mib", "1"]).is_err());
        assert!(parse(&["--preload-plan-only", "--commit", "on"]).is_err());
        assert!(parse(&["--preload-plan-only", "--prod-ks", "1"]).is_err());
    }

    #[test]
    fn preload_and_resource_plan_branches_are_exclusive() {
        assert!(parse(&["--preload-plan-only", "--resource-plan-only"]).is_err());
        assert!(parse(&["--resource-plan-only", "--preload-plan-only"]).is_err());
    }

    #[test]
    fn profile_schemas_and_hard_widths() {
        assert_eq!(Profile::Cost.schema(), Some(profile::COST_SCHEMA));
        assert_eq!(Profile::Strict.schema(), Some(profile::STRICT_SCHEMA));
        assert_eq!(Profile::Oracle.schema(), None);
        // Strict: every row is fatal, including those above the shipping cap.
        assert!(Profile::Strict.hard_width(64, FEW_ROW_ROUTE_ROWS, FEW_ROW_ROUTE_ROWS));
        // Cost: only widths inside the qualified rows; beyond them timing only.
        assert!(Profile::Cost.hard_width(8, 8, FEW_ROW_ROUTE_ROWS));
        assert!(!Profile::Cost.hard_width(9, 8, FEW_ROW_ROUTE_ROWS));
    }
}
