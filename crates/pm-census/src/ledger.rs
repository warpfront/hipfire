// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `pm-census ledger`: turns a retained-PM4 dispatch ledger
//! (`HIPFIRE_REDLINE_LEDGER`, written by `rdna-compute::pm4_ledger`) into a
//! [`CensusV1`] cell that `pm-census census matrix` consumes.
//!
//! ## Why a ledger
//!
//! Redline submits retained PM4 decode programs straight to a hardware queue;
//! rocprofv3 never sees those dispatches, so a rocprof census of an AR decode
//! ROI refuses (kernel busy 0.2-3.3 % of the ROI). The ledger is recorded at
//! retain time (one row per retained dispatch: kernel symbol, code-object
//! identity, grid, block) and counts replays per program at replay time.
//!
//! ## Two honest measurement modes
//!
//! * **Dispatch count** ([`MeasurementMode::LedgerDispatchCount`], default
//!   tape, no perturbation). Every retained dispatch executes exactly once per
//!   replay, so the dispatch count per kernel is exact. The cell is
//!   *count-weighted*: `by_origin[*].calls` / `top[*].calls` are executed
//!   dispatches, every `*_ns` per-kernel field is 0, and the ROI wall is 0.
//!   Whole-program GPU time exists (`measurement.program_gpu_span_ns`, from
//!   the tape's bracketing timestamps) but is not apportioned to kernels:
//!   count x per-kernel duration measured elsewhere is not time and is not
//!   produced here. The matrix reports `pm_share_of_dispatches` and leaves the
//!   two time shares unset.
//! * **PM4 timestamps** ([`MeasurementMode::LedgerPm4Timestamps`], ledger
//!   timing run, *perturbing*). The tape carries a GPU-clock write after every
//!   dispatch; span `i` is the time from timestamp `i` to `i+1`, i.e. dispatch
//!   `i` plus the boundary packets (waits, cache acquires) before it. Spans of
//!   one single-queue tape are disjoint, so `exclusive_ns == sum_ns` and the
//!   per-origin sums add up to the retained-program GPU span (`kernel_busy_ns`
//!   == `roi.wall_ns`). Time-weighted, but of a different tape (single queue,
//!   gfx12 pacing off, extra packets) than the product run.
//!
//! Neither mode sees HIP-launched kernels (embedding, sampling, copies) or
//! anything outside retained programs, nor host time between replays; the
//! census says so in `measurement.scope` and `notes`.
//!
//! ## Origin join
//!
//! The ledger carries the SHA-256 of the exact image Redline admitted, which
//! is stronger than the rocprof join by code-object size:
//!
//! 1. image (or device-ELF) SHA-256 equals an inventory PM object and that
//!    object exports the symbol -> `Pm`;
//! 2. image equals a PM object that does not export the symbol -> `Unknown`;
//! 3. image matches no PM object but some PM object exports the symbol (a
//!    different build, or a hipcc kernel sharing a PM export's name) ->
//!    `Unknown`;
//! 4. no match, no PM export, origin `hip_file` (a hipcc cache object) ->
//!    `Hipcc`;
//! 5. otherwise (a native image that is no inventory object) -> `Unknown`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::Deserialize;

use crate::census::{
    contract_index, origin_key, parse_phase, sha256_hex, Bucket, CellKey, CensusV1,
    MeasurementMode, MeasurementV1, RoiV1, TopKernel, Weighting, CENSUS_SCHEMA, MATRIX_ARCHES,
    MATRIX_MODELS,
};
use crate::inventory::{self, Attribution, Inventory, Origin};
use crate::rocprof::TraceSource;

/// The only ledger schema this reader understands.
pub const LEDGER_VERSION: u64 = 1;

/// `by_dispatch` key of every dispatch a ledger cell counts.
pub const DISPATCH_CLASS: &str = "retained_pm4";

/// UNKNOWN share above which a census carries an explicit lower-bound note.
const UNKNOWN_SHARE_LIMIT: f64 = 0.05;

// ───────────────────────────── ledger rows ─────────────────────────────

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Row {
    Program(ProgramRow),
    Dispatch(DispatchRow),
    ReplaySummary(SummaryRow),
}

#[derive(Clone, Debug, Deserialize)]
struct ProgramRow {
    ledger_version: u64,
    program: u64,
    pid: u64,
    #[serde(default)]
    device_ordinal: u64,
    arch: String,
    dispatches: usize,
    #[serde(default)]
    sequence_hash: String,
    #[serde(default)]
    exec_order: Option<Vec<usize>>,
    #[serde(default)]
    timing_mode: bool,
    #[serde(default)]
    span_layout: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct DispatchRow {
    program: u64,
    pid: u64,
    index: usize,
    kernel: String,
    artifact: Artifact,
    grid: [u32; 3],
    block: [u32; 3],
    #[serde(default)]
    shared_mem: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct SummaryRow {
    program: u64,
    pid: u64,
    replays: u64,
    #[serde(default)]
    gpu_span_ns: u64,
    #[serde(default)]
    timing: Option<Timing>,
}

/// Code-object identity of one retained launch.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Artifact {
    /// `native_embedded` | `hip_file`.
    pub origin: String,
    #[serde(default)]
    pub module: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    pub image_sha256: String,
    #[serde(default)]
    pub image_bytes: u64,
}

/// Per-dispatch GPU spans summed over `samples` replays (execution-slot order).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Timing {
    pub samples: u64,
    pub span_ns_sum: Vec<u64>,
    #[serde(default)]
    pub shape_mismatch: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerDispatch {
    pub kernel: String,
    pub artifact: Artifact,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub replays: u64,
    pub gpu_span_ns: u64,
    pub timing: Option<Timing>,
}

/// One retained program of one process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub pid: u64,
    pub id: u64,
    pub arch: String,
    pub device_ordinal: u64,
    pub sequence_hash: String,
    /// The tape carried per-dispatch timestamps (perturbing).
    pub timing_mode: bool,
    pub span_layout: Option<String>,
    /// Execution slot -> recorded index (`None` = recorded order).
    pub exec_order: Option<Vec<usize>>,
    declared: usize,
    pub dispatches: Vec<LedgerDispatch>,
    pub summary: Option<Summary>,
}

impl Program {
    fn label(&self) -> String {
        format!("pid {} program {}", self.pid, self.id)
    }

    fn has_spans(&self) -> bool {
        self.summary.as_ref().is_some_and(|s| {
            s.timing
                .as_ref()
                .is_some_and(|t| t.samples > 0 && !t.span_ns_sum.is_empty())
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ledger {
    pub programs: Vec<Program>,
}

/// Parses a ledger. Every structural defect is an error: a ledger that is
/// incomplete (a program with fewer dispatch rows than it declares, rows of an
/// undeclared program) never yields a census.
pub fn parse(text: &str) -> Result<Ledger, String> {
    let mut programs: Vec<Program> = Vec::new();
    let mut index: HashMap<(u64, u64), usize> = HashMap::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let at = n + 1;
        let row: Row = serde_json::from_str(line)
            .map_err(|e| format!("ledger line {at}: not a ledger row: {e}"))?;
        match row {
            Row::Program(p) => {
                if p.ledger_version != LEDGER_VERSION {
                    return Err(format!(
                        "ledger line {at}: ledger_version {} (this reader understands {LEDGER_VERSION})",
                        p.ledger_version
                    ));
                }
                if index.insert((p.pid, p.program), programs.len()).is_some() {
                    return Err(format!(
                        "ledger line {at}: pid {} program {} declared twice",
                        p.pid, p.program
                    ));
                }
                programs.push(Program {
                    pid: p.pid,
                    id: p.program,
                    arch: p.arch,
                    device_ordinal: p.device_ordinal,
                    sequence_hash: p.sequence_hash,
                    timing_mode: p.timing_mode,
                    span_layout: p.span_layout,
                    exec_order: p.exec_order,
                    declared: p.dispatches,
                    dispatches: Vec::with_capacity(p.dispatches),
                    summary: None,
                });
            }
            Row::Dispatch(d) => {
                let Some(&i) = index.get(&(d.pid, d.program)) else {
                    return Err(format!(
                        "ledger line {at}: dispatch row of undeclared pid {} program {}",
                        d.pid, d.program
                    ));
                };
                let program = &mut programs[i];
                if d.index != program.dispatches.len() {
                    return Err(format!(
                        "ledger line {at}: {} dispatch index {} out of order (expected {})",
                        program.label(),
                        d.index,
                        program.dispatches.len()
                    ));
                }
                if !matches!(d.artifact.origin.as_str(), "native_embedded" | "hip_file") {
                    return Err(format!(
                        "ledger line {at}: unknown artifact origin `{}`",
                        d.artifact.origin
                    ));
                }
                program.dispatches.push(LedgerDispatch {
                    kernel: d.kernel,
                    artifact: d.artifact,
                    grid: d.grid,
                    block: d.block,
                    shared_mem: d.shared_mem,
                });
            }
            Row::ReplaySummary(s) => {
                let Some(&i) = index.get(&(s.pid, s.program)) else {
                    return Err(format!(
                        "ledger line {at}: replay_summary of undeclared pid {} program {}",
                        s.pid, s.program
                    ));
                };
                // Summaries are cumulative: the largest replay count is the latest.
                let keep = programs[i]
                    .summary
                    .as_ref()
                    .is_none_or(|old| s.replays >= old.replays);
                if keep {
                    programs[i].summary = Some(Summary {
                        replays: s.replays,
                        gpu_span_ns: s.gpu_span_ns,
                        timing: s.timing,
                    });
                }
            }
        }
    }
    for p in &programs {
        if p.dispatches.len() != p.declared {
            return Err(format!(
                "{}: header declares {} dispatches but the ledger has {} dispatch rows \
                 (truncated write?)",
                p.label(),
                p.declared,
                p.dispatches.len()
            ));
        }
    }
    Ok(Ledger { programs })
}

// ───────────────────────────── origin join ─────────────────────────────

fn short(sha: &str) -> &str {
    &sha[..sha.len().min(12)]
}

/// Decides who produced the code behind one retained launch (module docs,
/// "Origin join").
pub fn attribute(inv: &Inventory, kernel: &str, art: &Artifact) -> Attribution {
    let at = inv.commit.as_deref().unwrap_or("the working tree");
    let matched: Vec<&inventory::PmObject> = inv
        .pm_objects
        .iter()
        .filter(|o| {
            o.sha256 == art.image_sha256
                || o.device_elfs.iter().any(|e| e.sha256 == art.image_sha256)
        })
        .collect();
    let exports = |o: &inventory::PmObject| {
        o.device_elfs
            .iter()
            .any(|e| e.symbols.iter().any(|s| s == kernel))
    };
    if !matched.is_empty() {
        let exporters: Vec<&&inventory::PmObject> = matched.iter().filter(|o| exports(o)).collect();
        if exporters.is_empty() {
            let paths: BTreeSet<&str> = matched.iter().map(|o| o.path.as_str()).collect();
            return Attribution {
                origin: Origin::Unknown,
                object_path: None,
                object_sha256: None,
                join: format!(
                    "unresolved: image sha256 {} equals {} but no such object exports {kernel}",
                    short(&art.image_sha256),
                    paths.into_iter().collect::<Vec<_>>().join(", ")
                ),
            };
        }
        let paths: BTreeSet<&str> = exporters.iter().map(|o| o.path.as_str()).collect();
        let (object_path, object_sha256) = match paths.iter().next() {
            Some(only) if paths.len() == 1 => (
                Some((*only).to_string()),
                exporters
                    .iter()
                    .find(|o| o.path == *only)
                    .map(|o| o.sha256.clone()),
            ),
            _ => (None, None),
        };
        return Attribution {
            origin: Origin::Pm,
            object_path,
            object_sha256,
            join: format!(
                "image_sha256+symbol: ledger image sha256 {} == {} exporting {kernel}",
                short(&art.image_sha256),
                paths.into_iter().collect::<Vec<_>>().join(", ")
            ),
        };
    }
    let pm_exporters: BTreeSet<&str> = inv
        .pm_objects
        .iter()
        .filter(|o| exports(o))
        .map(|o| o.path.as_str())
        .collect();
    if !pm_exporters.is_empty() {
        return Attribution {
            origin: Origin::Unknown,
            object_path: None,
            object_sha256: None,
            join: format!(
                "unresolved: PM export {kernel} of {}, but the launched image sha256 {} ({}) \
                 matches no inventory object at {at}",
                pm_exporters.into_iter().collect::<Vec<_>>().join(", "),
                short(&art.image_sha256),
                art.origin
            ),
        };
    }
    if art.origin == "hip_file" {
        let how = if inv.hip_kernel_names.contains(kernel) {
            "hip_source_name"
        } else {
            "non_pm_object"
        };
        return Attribution {
            origin: Origin::Hipcc,
            object_path: None,
            object_sha256: None,
            join: format!(
                "ledger_hip_file: {how}: hipcc cache object ({} bytes, sha256 {}) matches no PM \
                 object at {at} and no PM object exports {kernel}",
                art.image_bytes,
                short(&art.image_sha256)
            ),
        };
    }
    Attribution {
        origin: Origin::Unknown,
        object_path: None,
        object_sha256: None,
        join: format!(
            "unresolved: native image {} (sha256 {}) matches no inventory PM object at {at} \
             and no PM object exports {kernel}",
            art.module.as_deref().unwrap_or("?"),
            short(&art.image_sha256)
        ),
    }
}

// ───────────────────────────── census ─────────────────────────────

/// Weighting requested by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightingChoice {
    /// Time when every counted program carries per-dispatch spans, count when
    /// none does; a mixed ledger is an error.
    Auto,
    /// Dispatch counts, even when the ledger also holds spans.
    Count,
    /// Per-dispatch GPU spans; error unless every counted program has them.
    Time,
}

impl WeightingChoice {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "count" => Ok(Self::Count),
            "time" => Ok(Self::Time),
            _ => Err(format!("unknown weighting `{s}` (auto | count | time)")),
        }
    }
}

#[derive(Default)]
struct Agg {
    calls: u64,
    sum_ns: u64,
    symbols: BTreeSet<String>,
}

impl Agg {
    fn add(&mut self, calls: u64, ns: u64, symbol: &str) {
        self.calls += calls;
        self.sum_ns += ns;
        if !self.symbols.contains(symbol) {
            self.symbols.insert(symbol.to_string());
        }
    }

    /// Spans of one single-queue tape are disjoint, so exclusive == sum.
    fn bucket(&self) -> Bucket {
        Bucket {
            calls: self.calls,
            kernels: self.symbols.len() as u64,
            sum_ns: self.sum_ns,
            exclusive_ns: self.sum_ns,
        }
    }
}

/// Execution slot of each recorded index (identity without a reorder).
fn slot_of(program: &Program) -> Result<Vec<usize>, String> {
    let n = program.dispatches.len();
    let Some(order) = &program.exec_order else {
        return Ok((0..n).collect());
    };
    let mut slot = vec![usize::MAX; n];
    if order.len() != n {
        return Err(format!(
            "{}: exec_order has {} entries for {n} dispatches",
            program.label(),
            order.len()
        ));
    }
    for (s, &recorded) in order.iter().enumerate() {
        if recorded >= n || slot[recorded] != usize::MAX {
            return Err(format!(
                "{}: exec_order is not a permutation of 0..{n}",
                program.label()
            ));
        }
        slot[recorded] = s;
    }
    Ok(slot)
}

/// Builds the ledger census of one matrix cell.
pub fn ingest(
    ledger: &Ledger,
    capture: TraceSource,
    inv: &Inventory,
    key: CellKey,
    choice: WeightingChoice,
    top: usize,
) -> Result<CensusV1, String> {
    if ledger.programs.is_empty() {
        return Err(
            "ledger has no program rows (HIPFIRE_REDLINE_LEDGER was not set, or no \
                    retained PM4 program was prepared)"
                .to_string(),
        );
    }
    let unflushed: Vec<String> = ledger
        .programs
        .iter()
        .filter(|p| p.summary.is_none())
        .map(Program::label)
        .collect();
    if !unflushed.is_empty() {
        return Err(format!(
            "no replay_summary for {}: the process exited without flushing the ledger (killed \
             before stdin EOF?) so its replay counts are unknown",
            unflushed.join(", ")
        ));
    }
    let counted: Vec<&Program> = ledger
        .programs
        .iter()
        .filter(|p| p.summary.as_ref().is_some_and(|s| s.replays > 0))
        .collect();
    if counted.is_empty() {
        return Err(format!(
            "ledger records {} retained program(s) but zero replays: nothing executed through them",
            ledger.programs.len()
        ));
    }
    for p in &counted {
        if p.arch != key.arch {
            return Err(format!(
                "{} ran on {} but the cell arch is {}",
                p.label(),
                p.arch,
                key.arch
            ));
        }
    }

    let with_spans = counted.iter().filter(|p| p.has_spans()).count();
    let weighting = match choice {
        WeightingChoice::Count => Weighting::Count,
        WeightingChoice::Time => {
            if with_spans != counted.len() {
                return Err(format!(
                    "--weighting time needs per-dispatch spans for every replayed program \
                     ({with_spans}/{} have them): rerun with HIPFIRE_REDLINE_LEDGER_TIMING=1",
                    counted.len()
                ));
            }
            Weighting::Time
        }
        WeightingChoice::Auto => {
            if with_spans == 0 {
                Weighting::Count
            } else if with_spans == counted.len() {
                Weighting::Time
            } else {
                return Err(format!(
                    "mixed ledger: {with_spans}/{} replayed programs carry per-dispatch spans; \
                     pass --weighting count, or split the ledger by run",
                    counted.len()
                ));
            }
        }
    };
    let time = weighting == Weighting::Time;
    let perturbing = counted.iter().any(|p| p.timing_mode);

    let contracts = contract_index();
    let mut cache: HashMap<(String, String, String), (Attribution, &'static str)> = HashMap::new();
    let mut by_origin: BTreeMap<&'static str, Agg> = BTreeMap::new();
    let mut class = Agg::default();
    let mut by_contract: BTreeMap<String, Agg> = BTreeMap::new();
    type RowKey = (String, &'static str, String);
    let mut rows: BTreeMap<RowKey, (u64, u64)> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let (mut replays, mut executed, mut gpu_span_ns, mut span_sum_ns) = (0u64, 0u64, 0u64, 0u64);
    let mut spanless: Vec<String> = Vec::new();
    let mut shape_mismatch = 0u64;
    let mut zero_replay: Vec<String> = Vec::new();
    for p in &ledger.programs {
        if !counted.iter().any(|c| std::ptr::eq(*c, p)) {
            zero_replay.push(p.label());
        }
    }

    for p in &counted {
        let summary = p.summary.as_ref().expect("counted programs have summaries");
        gpu_span_ns += summary.gpu_span_ns;
        let n = p.dispatches.len();
        let (weight, spans, slots) = if time {
            let t = summary.timing.as_ref().expect("has_spans checked");
            if t.span_ns_sum.len() != n {
                return Err(format!(
                    "{}: {} per-dispatch spans for {n} dispatches (span_layout {:?}): only \
                     one span per dispatch is supported (gfx12 tapes); gfx11 boundary tapes \
                     interleave kernel and boundary spans",
                    p.label(),
                    t.span_ns_sum.len(),
                    p.span_layout
                ));
            }
            if t.samples < summary.replays {
                spanless.push(format!(
                    "{} ({} of {} replays)",
                    p.label(),
                    summary.replays - t.samples,
                    summary.replays
                ));
            }
            shape_mismatch += t.shape_mismatch;
            (t.samples, Some(&t.span_ns_sum), slot_of(p)?)
        } else {
            (summary.replays, None, Vec::new())
        };
        replays += weight;
        executed += weight * n as u64;
        for (i, d) in p.dispatches.iter().enumerate() {
            let cache_key = (
                d.kernel.clone(),
                d.artifact.origin.clone(),
                d.artifact.image_sha256.clone(),
            );
            let (at, origin) = cache.entry(cache_key).or_insert_with(|| {
                let at = attribute(inv, &d.kernel, &d.artifact);
                let origin = origin_key(&at.origin);
                (at, origin)
            });
            let ns = spans.map_or(0, |s| s[slots[i]]);
            span_sum_ns += ns;
            by_origin
                .entry(origin)
                .or_default()
                .add(weight, ns, &d.kernel);
            class.add(weight, ns, &d.kernel);
            if let Some(c) = contracts.get(d.kernel.as_str()) {
                by_contract
                    .entry(c.clone())
                    .or_default()
                    .add(weight, ns, &d.kernel);
            }
            let row = rows
                .entry((d.kernel.clone(), origin, at.join.clone()))
                .or_default();
            row.0 += weight;
            row.1 += ns;
        }
    }
    if shape_mismatch > 0 {
        notes.push(format!(
            "{shape_mismatch} replay(s) returned a span vector of a different length and were \
             not summed"
        ));
    }
    if !spanless.is_empty() {
        notes.push(format!(
            "replays without spans are not counted in this time-weighted cell: {}",
            spanless.join("; ")
        ));
    }
    if !zero_replay.is_empty() {
        notes.push(format!(
            "prepared but never replayed (excluded): {}",
            zero_replay.join(", ")
        ));
    }
    let pids: BTreeSet<u64> = counted.iter().map(|p| p.pid).collect();
    if pids.len() > 1 {
        notes.push(format!(
            "ledger spans {} processes (pids {:?}); programs are summed",
            pids.len(),
            pids
        ));
    }
    let devices: BTreeSet<u64> = counted.iter().map(|p| p.device_ordinal).collect();
    if devices.len() > 1 {
        notes.push(format!(
            "ledger spans device ordinals {devices:?}; `agent` is the first program's ordinal"
        ));
    }

    let scope = "retained-PM4 dispatches of the programs in the ledger only; HIP-launched kernels \
                 (embedding, sampling, copies), host time between replays and any dispatch \
                 outside a retained program are not in this census"
        .to_string();
    let mut total_calls = 0u64;
    let mut dispatch_share_by_origin = BTreeMap::new();
    for a in by_origin.values() {
        total_calls += a.calls;
    }
    for (origin, a) in &by_origin {
        dispatch_share_by_origin.insert(
            (*origin).to_string(),
            a.calls as f64 / total_calls.max(1) as f64,
        );
    }
    if time {
        notes.push(
            "PERTURBING time-weighted ledger cell: the run executed the per-dispatch-timestamp \
             tape (single queue, gfx12 dispatch pacing off, one GPU-clock write after every \
             dispatch), not the product tape. Span i is dispatch i plus the PM4 boundary packets \
             (waits/acquires) that precede it; roi.wall_ns is the summed retained-program span, \
             not request wall time"
                .to_string(),
        );
    } else {
        notes.push(
            "COUNT-WEIGHTED ledger cell: calls are executed dispatches (replays x dispatches); \
             every per-kernel sum_ns/exclusive_ns is 0 and roi.wall_ns is 0 because no per-kernel \
             time was measured. Shares by origin are dispatch-count shares and say nothing \
             about GPU time; the whole-program span is in measurement.program_gpu_span_ns"
                .to_string(),
        );
        if perturbing {
            notes.push(
                "the counted program(s) were prepared on the per-dispatch-timestamp tape \
                 (perturbing); counts are exact but the run was not the product tape"
                    .to_string(),
            );
        }
    }
    if let Some(a) = by_origin.get("unknown") {
        let share = if time {
            a.sum_ns as f64 / span_sum_ns.max(1) as f64
        } else {
            a.calls as f64 / total_calls.max(1) as f64
        };
        if share > UNKNOWN_SHARE_LIMIT {
            notes.push(format!(
                "unknown share {:.1}% of the {}: those dispatches join to no PM object and no \
                 hipcc object, so pm/hipcc shares are lower bounds",
                100.0 * share,
                if time {
                    "retained span"
                } else {
                    "executed dispatches"
                }
            ));
        }
    }

    let wall = if time { span_sum_ns } else { 0 };
    let mut top_rows: Vec<TopKernel> = rows
        .into_iter()
        .map(|((symbol, origin, join), (calls, ns))| {
            let abi = contracts.get(symbol.as_str()).cloned();
            TopKernel {
                symbol,
                origin: crate::census::origin_from_key(origin),
                join,
                dispatch: DISPATCH_CLASS.to_string(),
                abi,
                calls,
                sum_ns: ns,
                exclusive_ns: ns,
                pct_wall: if time {
                    100.0 * ns as f64 / wall.max(1) as f64
                } else {
                    0.0
                },
            }
        })
        .collect();
    if time {
        top_rows.sort_by(|a, b| b.sum_ns.cmp(&a.sum_ns).then(a.symbol.cmp(&b.symbol)));
    } else {
        top_rows.sort_by(|a, b| b.calls.cmp(&a.calls).then(a.symbol.cmp(&b.symbol)));
    }
    top_rows.truncate(top);

    let mut by_dispatch = BTreeMap::new();
    by_dispatch.insert(DISPATCH_CLASS.to_string(), class.bucket());
    Ok(CensusV1 {
        schema: CENSUS_SCHEMA.to_string(),
        cell: key,
        capture,
        inventory_commit: inv.commit.clone(),
        agent: counted[0].device_ordinal,
        roi: RoiV1 {
            start_ns: 0,
            end_ns: wall,
            wall_ns: wall,
            spec: "ledger:process".to_string(),
        },
        kernels_in_roi: executed,
        kernel_sum_ns: wall,
        kernel_busy_ns: wall,
        copy_busy_ns: 0,
        copy_kernel_overlap_ns: 0,
        idle_ns: 0,
        by_origin: by_origin
            .iter()
            .map(|(k, a)| ((*k).to_string(), a.bucket()))
            .collect(),
        by_dispatch,
        by_contract: by_contract
            .iter()
            .map(|(k, a)| (k.clone(), a.bucket()))
            .collect(),
        top: top_rows,
        notes,
        measurement: Some(MeasurementV1 {
            mode: if time {
                MeasurementMode::LedgerPm4Timestamps
            } else {
                MeasurementMode::LedgerDispatchCount
            },
            weighting,
            perturbing: perturbing || time,
            scope,
            programs: counted.len() as u64,
            replays,
            dispatches_executed: executed,
            program_gpu_span_ns: gpu_span_ns,
            dispatch_share_by_origin,
        }),
    })
}

// ───────────────────────────── CLI ─────────────────────────────

const USAGE: &str = "\
pm-census ledger --ledger PATH --model M --arch A --phase P
                 [--weighting auto|count|time] [--repo DIR] [--commit SHA] [--top N]
    Ingests a retained-PM4 dispatch ledger (HIPFIRE_REDLINE_LEDGER, optionally
    HIPFIRE_REDLINE_LEDGER_TIMING=1) and prints a CensusV1 cell for
    `pm-census census matrix`. Origin pm/hipcc comes from the ledger's code-object
    image SHA-256 joined to the object inventory at --commit (or the working tree).
    auto (default): time-weighted when every replayed program carries per-dispatch
    spans (a perturbing timing run), dispatch-count-weighted otherwise. count: always
    dispatch counts. time: spans required. Count-weighted cells carry no per-kernel time.
    model: dense-27b | a3b | flash-next   arch: gfx1201 | gfx1151 | gfx1100   phase: prefill | ar | spec";

fn flag_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

pub fn cli(args: &[String]) -> Result<(), String> {
    let (mut ledger, mut model, mut arch, mut phase) = (None, None, None, None);
    let mut choice = WeightingChoice::Auto;
    let mut repo = ".".to_string();
    let mut commit: Option<String> = None;
    let mut top = 25usize;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--ledger" => ledger = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--model" => model = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--arch" => arch = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--phase" => phase = Some(parse_phase(flag_value(args, &mut i, flag)?)?),
            "--weighting" => choice = WeightingChoice::parse(flag_value(args, &mut i, flag)?)?,
            "--repo" => repo = flag_value(args, &mut i, flag)?.to_string(),
            "--commit" => commit = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--top" => {
                let v = flag_value(args, &mut i, flag)?;
                top = v
                    .parse::<usize>()
                    .map_err(|e| format!("--top `{v}`: {e}"))?;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
        i += 1;
    }
    let need =
        |v: Option<String>, flag: &str| v.ok_or_else(|| format!("{flag} is required\n{USAGE}"));
    let path = need(ledger, "--ledger")?;
    let key = CellKey {
        model: need(model, "--model")?,
        arch: need(arch, "--arch")?,
        phase: phase.ok_or_else(|| format!("--phase is required\n{USAGE}"))?,
    };
    if !MATRIX_MODELS.contains(&key.model.as_str()) || !MATRIX_ARCHES.contains(&key.arch.as_str()) {
        return Err(format!(
            "cell {}:{} is not in the matrix (models {MATRIX_MODELS:?}, arches {MATRIX_ARCHES:?})",
            key.model, key.arch
        ));
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
    let text = String::from_utf8(bytes.clone()).map_err(|_| format!("{path}: not UTF-8"))?;
    let parsed = parse(&text)?;
    let inv = inventory::build(Path::new(&repo), commit.as_deref())?;
    let capture = TraceSource {
        path: path.clone(),
        sha256: sha256_hex(&bytes),
        format: "redline-pm4-ledger-jsonl".to_string(),
    };
    let census = ingest(&parsed, capture, &inv, key, choice, top)?;
    let json =
        serde_json::to_string_pretty(&census).map_err(|e| format!("serialize census: {e}"))?;
    println!("{json}");
    Ok(())
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::{matrix, CellStatus};
    use crate::inventory::{DeviceElf, PmObject};
    use hipfire_atlas::workload::PhaseV1;
    use serde_json::json;

    /// The golden file `rdna-compute`'s serializer test pins byte-for-schema.
    const FIXTURE: &str = include_str!("../../rdna-compute/tests/fixtures/pm4_ledger_v1.jsonl");

    fn key() -> CellKey {
        CellKey {
            model: "a3b".into(),
            arch: "gfx1201".into(),
            phase: PhaseV1::Ar,
        }
    }

    fn source() -> TraceSource {
        TraceSource {
            path: "ledger.jsonl".into(),
            sha256: "0".repeat(64),
            format: "redline-pm4-ledger-jsonl".into(),
        }
    }

    /// PM object `aaaa..` exporting the fixture's PM kernel; hipcc names for the rest.
    fn inv() -> Inventory {
        Inventory {
            repo: "synthetic".into(),
            commit: Some("abc123".into()),
            pm_objects: vec![PmObject {
                path: "kernels/gemm_mq4g256v2_pm_gfx1201.hxaco".into(),
                sha256: "a".repeat(64),
                bytes: 4096,
                declared_at: vec!["crates/rdna-compute/src/gemm.rs:10".into()],
                device_elfs: vec![DeviceElf {
                    arch: "gfx1201".into(),
                    bytes: 3000,
                    sha256: "c".repeat(64),
                    symbols: vec!["gemm_mq4g256v2_gate_up_pm".into()],
                }],
            }],
            hip_kernel_names: ["rmsnorm_f32", "attn_decode_fp8"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn art(origin: &str, sha: &str) -> serde_json::Value {
        json!({"origin": origin, "module": null, "path": null, "image_sha256": sha, "image_bytes": 100})
    }

    fn program_rows(
        id: u64,
        pid: u64,
        timing: bool,
        exec_order: Option<Vec<usize>>,
        kernels: &[(&str, &str, &str)],
    ) -> String {
        let mut out = vec![json!({
            "kind": "program", "ledger_version": 1, "program": id, "pid": pid,
            "device_ordinal": 0, "arch": "gfx1201", "pm4_architecture": "Gfx12",
            "dispatches": kernels.len(), "sequence_hash": "00", "exec_order": exec_order,
            "timing_mode": timing, "perturbing": timing,
            "span_layout": if timing { json!("per_dispatch") } else { json!(null) },
        })];
        for (i, (kernel, origin, sha)) in kernels.iter().enumerate() {
            out.push(json!({
                "kind": "dispatch", "program": id, "pid": pid, "index": i, "kernel": kernel,
                "artifact": art(origin, sha), "grid": [1, 1, 1], "block": [64, 1, 1],
                "shared_mem": 0, "grid_binding": null,
            }));
        }
        out.iter().map(|v| format!("{v}\n")).collect()
    }

    fn summary(
        id: u64,
        pid: u64,
        replays: u64,
        gpu: u64,
        timing: Option<(u64, Vec<u64>)>,
    ) -> String {
        format!(
            "{}\n",
            json!({
                "kind": "replay_summary", "program": id, "pid": pid, "replays": replays,
                "gpu_span_ns": gpu,
                "timing": timing.map(|(s, v)| json!({"samples": s, "span_ns_sum": v, "shape_mismatch": 0})),
            })
        )
    }

    const PM_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn kernels() -> [(&'static str, &'static str, &'static str); 4] {
        [
            ("rmsnorm_f32", "hip_file", "bbbb"),
            ("gemm_mq4g256v2_gate_up_pm", "native_embedded", PM_SHA),
            ("attn_decode_fp8", "hip_file", "bbbb"),
            ("gemm_mq4g256v2_gate_up_pm", "native_embedded", PM_SHA),
        ]
    }

    #[test]
    fn golden_fixture_parses_and_splits_pm_from_hipcc_by_count() {
        let ledger = parse(FIXTURE).unwrap();
        assert_eq!(ledger.programs.len(), 1);
        let p = &ledger.programs[0];
        assert_eq!((p.pid, p.id, p.arch.as_str()), (4242, 1, "gfx1201"));
        assert_eq!(p.dispatches.len(), 3);
        assert_eq!(p.dispatches[1].artifact.origin, "native_embedded");
        assert_eq!(p.dispatches[1].artifact.image_sha256, "a".repeat(64));
        assert_eq!(p.summary.as_ref().unwrap().replays, 128);

        let c = ingest(&ledger, source(), &inv(), key(), WeightingChoice::Auto, 10).unwrap();
        assert_eq!(c.schema, CENSUS_SCHEMA);
        let m = c.measurement.as_ref().unwrap();
        assert_eq!(m.mode, MeasurementMode::LedgerDispatchCount);
        assert_eq!(m.weighting, Weighting::Count);
        assert!(!m.perturbing);
        assert_eq!(
            (m.programs, m.replays, m.dispatches_executed),
            (1, 128, 384)
        );
        assert_eq!(m.program_gpu_span_ns, 9_000_000);
        // pm vs hipcc split, exact dispatch counts
        assert_eq!(c.by_origin["pm"].calls, 128);
        assert_eq!(c.by_origin["hipcc"].calls, 256);
        assert_eq!(c.by_origin["hipcc"].kernels, 2);
        assert!(!c.by_origin.contains_key("unknown"));
        assert!((m.dispatch_share_by_origin["pm"] - 1.0 / 3.0).abs() < 1e-12);
        // count-weighted: no invented time anywhere
        assert_eq!(
            (c.kernel_sum_ns, c.kernel_busy_ns, c.roi.wall_ns),
            (0, 0, 0)
        );
        assert!(c
            .by_origin
            .values()
            .all(|b| b.sum_ns == 0 && b.exclusive_ns == 0));
        assert!(c
            .top
            .iter()
            .all(|t| t.sum_ns == 0 && t.dispatch == DISPATCH_CLASS));
        assert_eq!(c.by_dispatch[DISPATCH_CLASS].calls, 384);
        assert_eq!(c.kernels_in_roi, 384);
        assert_eq!(c.inventory_commit.as_deref(), Some("abc123"));
        assert_eq!(c.top[0].calls, 128);
        assert!(c.notes.iter().any(|n| n.starts_with("COUNT-WEIGHTED")));
        let pm_row = c
            .top
            .iter()
            .find(|t| t.symbol == "gemm_mq4g256v2_gate_up_pm")
            .unwrap();
        assert_eq!(pm_row.origin, Origin::Pm);
        assert!(
            pm_row.join.starts_with("image_sha256+symbol"),
            "{}",
            pm_row.join
        );
    }

    #[test]
    fn count_cell_flows_through_the_matrix_without_a_time_share() {
        let ledger = parse(FIXTURE).unwrap();
        let c = ingest(&ledger, source(), &inv(), key(), WeightingChoice::Auto, 10).unwrap();
        // The census file round-trips as CensusV1 (what `census matrix` reads).
        let bytes = serde_json::to_vec_pretty(&c).unwrap();
        let back: CensusV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.measurement, c.measurement);
        let m = matrix(&[("l.json".into(), "f".repeat(64), back)], &[]);
        let cell = m
            .cells
            .iter()
            .find(|x| x.key.model == "a3b" && x.key.arch == "gfx1201" && x.key.phase == PhaseV1::Ar)
            .unwrap();
        assert!(matches!(cell.status, CellStatus::Measured { .. }));
        assert!((cell.pm_share_of_dispatches.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!(cell.pm_share_of_kernel_busy.is_none() && cell.pm_share_of_wall.is_none());
        let label = cell.measurement.unwrap();
        assert_eq!(label.weighting, Weighting::Count);
        assert_eq!(label.mode, MeasurementMode::LedgerDispatchCount);
        assert!(cell.notes.iter().any(|n| n.contains("NOT of GPU time")));
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"ledger_dispatch_count\"") && j.contains("\"count\""));
    }

    #[test]
    fn timing_ledger_gives_a_labelled_perturbing_time_cell() {
        // slots: rmsnorm 100, pm 700, attn 150, pm 50 ns per replay, 4 replays.
        let mut text = program_rows(1, 7, true, None, &kernels());
        text.push_str(&summary(
            1,
            7,
            4,
            4_000,
            Some((4, vec![400, 2800, 600, 200])),
        ));
        let ledger = parse(&text).unwrap();
        let c = ingest(&ledger, source(), &inv(), key(), WeightingChoice::Auto, 10).unwrap();
        let m = c.measurement.as_ref().unwrap();
        assert_eq!(m.mode, MeasurementMode::LedgerPm4Timestamps);
        assert_eq!(m.weighting, Weighting::Time);
        assert!(m.perturbing);
        assert_eq!((m.replays, m.dispatches_executed), (4, 16));
        assert_eq!(c.by_origin["pm"].sum_ns, 2800 + 200);
        assert_eq!(c.by_origin["pm"].exclusive_ns, 3000);
        assert_eq!(c.by_origin["hipcc"].sum_ns, 400 + 600);
        assert_eq!(c.kernel_busy_ns, 4000);
        assert_eq!(c.roi.wall_ns, 4000);
        assert_eq!(c.idle_ns, 0);
        assert_eq!(c.top[0].symbol, "gemm_mq4g256v2_gate_up_pm");
        assert_eq!(c.top[0].sum_ns, 3000);
        assert!((c.top[0].pct_wall - 75.0).abs() < 1e-9);
        assert!(c.notes.iter().any(|n| n.starts_with("PERTURBING")));
        let cell_m = matrix(&[("l.json".into(), "f".repeat(64), c)], &[]);
        let cell = cell_m
            .cells
            .iter()
            .find(|x| x.key.model == "a3b" && x.key.arch == "gfx1201" && x.key.phase == PhaseV1::Ar)
            .unwrap();
        assert!((cell.pm_share_of_wall.unwrap() - 0.75).abs() < 1e-12);
        assert!((cell.pm_share_of_kernel_busy.unwrap() - 0.75).abs() < 1e-12);
        assert!(cell.pm_share_of_dispatches.is_none());
        assert!(cell.measurement.unwrap().perturbing);
        assert!(cell.notes.iter().any(|n| n.contains("perturbing run")));
    }

    #[test]
    fn timing_ledger_can_still_be_read_by_count() {
        let mut text = program_rows(1, 7, true, None, &kernels());
        text.push_str(&summary(
            1,
            7,
            4,
            4_000,
            Some((4, vec![400, 2800, 600, 200])),
        ));
        let c = ingest(
            &parse(&text).unwrap(),
            source(),
            &inv(),
            key(),
            WeightingChoice::Count,
            10,
        )
        .unwrap();
        let m = c.measurement.unwrap();
        assert_eq!(m.weighting, Weighting::Count);
        assert!(
            m.perturbing,
            "the tape was instrumented even if only counts are used"
        );
        assert_eq!(c.by_origin["pm"].calls, 8);
        assert_eq!(c.by_origin["pm"].sum_ns, 0);
    }

    #[test]
    fn exec_order_maps_spans_back_to_recorded_dispatches() {
        // Executed as recorded [1,0,2,3]: slot 0 ran dispatch 1 (pm), slot 1 dispatch 0 (hip).
        let mut text = program_rows(1, 7, true, Some(vec![1, 0, 2, 3]), &kernels());
        text.push_str(&summary(1, 7, 1, 1_000, Some((1, vec![700, 100, 150, 50]))));
        let c = ingest(
            &parse(&text).unwrap(),
            source(),
            &inv(),
            key(),
            WeightingChoice::Time,
            10,
        )
        .unwrap();
        assert_eq!(c.by_origin["pm"].sum_ns, 700 + 50);
        assert_eq!(c.by_origin["hipcc"].sum_ns, 100 + 150);
    }

    #[test]
    fn join_never_calls_a_mismatched_pm_export_pm() {
        let i = inv();
        // Same symbol, different image: Unknown, never silently Pm or Hipcc.
        let stale = Artifact {
            origin: "native_embedded".into(),
            module: Some("old".into()),
            path: None,
            image_sha256: "d".repeat(64),
            image_bytes: 1,
        };
        let at = attribute(&i, "gemm_mq4g256v2_gate_up_pm", &stale);
        assert_eq!(at.origin, Origin::Unknown);
        assert!(
            at.join.contains("matches no inventory object"),
            "{}",
            at.join
        );
        // hip_file object that shares a PM export's name: also Unknown.
        let hip = Artifact {
            origin: "hip_file".into(),
            ..stale.clone()
        };
        assert_eq!(
            attribute(&i, "gemm_mq4g256v2_gate_up_pm", &hip).origin,
            Origin::Unknown
        );
        // image is a PM object but exports no such symbol.
        let right = Artifact {
            image_sha256: "a".repeat(64),
            ..stale.clone()
        };
        let at = attribute(&i, "not_exported", &right);
        assert_eq!(at.origin, Origin::Unknown);
        assert!(at.join.contains("exports"), "{}", at.join);
        // device-ELF sha also joins.
        let elf = Artifact {
            image_sha256: "c".repeat(64),
            ..stale.clone()
        };
        let at = attribute(&i, "gemm_mq4g256v2_gate_up_pm", &elf);
        assert_eq!(at.origin, Origin::Pm);
        assert_eq!(
            at.object_path.as_deref(),
            Some("kernels/gemm_mq4g256v2_pm_gfx1201.hxaco")
        );
        assert_eq!(at.object_sha256, Some("a".repeat(64)));
        // hipcc cache object, name not a PM export.
        let at = attribute(&i, "some_jit_kernel", &hip);
        assert_eq!(at.origin, Origin::Hipcc);
        assert!(
            at.join.starts_with("ledger_hip_file: non_pm_object"),
            "{}",
            at.join
        );
        let at = attribute(&i, "rmsnorm_f32", &hip);
        assert!(
            at.join.starts_with("ledger_hip_file: hip_source_name"),
            "{}",
            at.join
        );
        // an embedded image that is no inventory object: Unknown.
        let at = attribute(&i, "other", &stale);
        assert_eq!(at.origin, Origin::Unknown);
    }

    #[test]
    fn unjoinable_dispatches_are_counted_unknown_with_a_lower_bound_note() {
        let mut text = program_rows(
            1,
            7,
            false,
            None,
            &[
                ("mystery", "native_embedded", "e"),
                ("rmsnorm_f32", "hip_file", "b"),
            ],
        );
        text.push_str(&summary(1, 7, 10, 1, None));
        let c = ingest(
            &parse(&text).unwrap(),
            source(),
            &inv(),
            key(),
            WeightingChoice::Auto,
            10,
        )
        .unwrap();
        assert_eq!(c.by_origin["unknown"].calls, 10);
        assert!(c.notes.iter().any(|n| n.contains("lower bounds")));
    }

    #[test]
    fn defective_ledgers_are_refused_not_measured() {
        let ok_rows = program_rows(1, 7, false, None, &kernels());
        let census = |text: &str, k: CellKey, w: WeightingChoice| {
            ingest(&parse(text)?, source(), &inv(), k, w, 5).map(|_| ())
        };
        // killed before flush: no summary
        let e = census(&ok_rows, key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("without flushing"), "{e}");
        // zero replays
        let zero = format!("{ok_rows}{}", summary(1, 7, 0, 0, None));
        let e = census(&zero, key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("zero replays"), "{e}");
        // wrong arch
        let fine = format!("{ok_rows}{}", summary(1, 7, 3, 9, None));
        let mut other = key();
        other.arch = "gfx1151".into();
        let e = census(&fine, other, WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("cell arch is gfx1151"), "{e}");
        // time requested without spans
        let e = census(&fine, key(), WeightingChoice::Time).unwrap_err();
        assert!(e.contains("HIPFIRE_REDLINE_LEDGER_TIMING"), "{e}");
        // wrong span count (gfx11 boundary pairs)
        let timed = format!(
            "{}{}",
            program_rows(1, 7, true, None, &kernels()),
            summary(1, 7, 2, 9, Some((2, vec![1; 8])))
        );
        let e = census(&timed, key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("8 per-dispatch spans for 4 dispatches"), "{e}");
        // mixed timing/non-timing programs
        let mixed = format!(
            "{}{}{}{}",
            program_rows(1, 7, true, None, &kernels()),
            summary(1, 7, 1, 9, Some((1, vec![1; 4]))),
            program_rows(2, 7, false, None, &kernels()),
            summary(2, 7, 1, 9, None)
        );
        let e = census(&mixed, key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("mixed ledger"), "{e}");
        // bad permutation
        let bad = format!(
            "{}{}",
            program_rows(1, 7, true, Some(vec![0, 0, 1, 2]), &kernels()),
            summary(1, 7, 1, 9, Some((1, vec![1; 4])))
        );
        let e = census(&bad, key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("not a permutation"), "{e}");
        // empty
        let e = census("", key(), WeightingChoice::Auto).unwrap_err();
        assert!(e.contains("no program rows"), "{e}");
    }

    #[test]
    fn structural_parse_errors() {
        let rows = program_rows(1, 7, false, None, &kernels());
        // truncated: drop the last dispatch row
        let cut: String = rows.lines().take(4).map(|l| format!("{l}\n")).collect();
        assert!(parse(&cut).unwrap_err().contains("declares 4 dispatches"));
        // dispatch of an undeclared program
        let tail = rows.lines().nth(1).unwrap();
        assert!(parse(tail).unwrap_err().contains("undeclared"));
        // out-of-order index
        let swapped: String = {
            let l: Vec<&str> = rows.lines().collect();
            format!("{}\n{}\n{}\n", l[0], l[2], l[1])
        };
        assert!(parse(&swapped).unwrap_err().contains("out of order"));
        // unknown schema version
        let v2 = rows.replacen("\"ledger_version\":1", "\"ledger_version\":2", 1);
        assert!(parse(&v2).unwrap_err().contains("ledger_version 2"));
        // duplicate program header
        let dup = format!("{rows}{}", rows.lines().next().unwrap());
        assert!(parse(&dup).unwrap_err().contains("declared twice"));
        // not JSON
        assert!(parse("garbage").unwrap_err().contains("line 1"));
    }

    #[test]
    fn cumulative_summaries_keep_the_largest_count_and_programs_key_by_pid() {
        let mut text = program_rows(1, 7, false, None, &kernels());
        text.push_str(&summary(1, 7, 5, 50, None));
        text.push_str(&summary(1, 7, 9, 90, None));
        // same program id in another process
        text.push_str(&program_rows(1, 8, false, None, &kernels()[..2]));
        text.push_str(&summary(1, 8, 4, 40, None));
        let ledger = parse(&text).unwrap();
        assert_eq!(ledger.programs.len(), 2);
        assert_eq!(ledger.programs[0].summary.as_ref().unwrap().replays, 9);
        let c = ingest(&ledger, source(), &inv(), key(), WeightingChoice::Auto, 5).unwrap();
        assert_eq!(c.measurement.as_ref().unwrap().replays, 13);
        assert_eq!(
            c.measurement.as_ref().unwrap().dispatches_executed,
            9 * 4 + 4 * 2
        );
        assert!(c.notes.iter().any(|n| n.contains("2 processes")));
    }

    #[test]
    fn zero_replay_programs_are_listed_and_excluded() {
        let mut text = program_rows(1, 7, false, None, &kernels());
        text.push_str(&summary(1, 7, 6, 60, None));
        text.push_str(&program_rows(2, 7, false, None, &kernels()));
        text.push_str(&summary(2, 7, 0, 0, None));
        let c = ingest(
            &parse(&text).unwrap(),
            source(),
            &inv(),
            key(),
            WeightingChoice::Auto,
            5,
        )
        .unwrap();
        assert_eq!(c.measurement.unwrap().programs, 1);
        assert!(c
            .notes
            .iter()
            .any(|n| n.contains("never replayed") && n.contains("program 2")));
    }

    #[test]
    fn rocprof_census_json_is_unchanged_by_the_measurement_field() {
        // A census without `measurement` (every rocprofv3 census) still parses,
        // and a ledger-free serialization does not emit the key.
        let ledger = parse(FIXTURE).unwrap();
        let c = ingest(&ledger, source(), &inv(), key(), WeightingChoice::Auto, 5).unwrap();
        let mut v = serde_json::to_value(&c).unwrap();
        assert!(v.get("measurement").is_some());
        v.as_object_mut().unwrap().remove("measurement");
        let back: CensusV1 = serde_json::from_value(v).unwrap();
        assert!(back.measurement.is_none());
    }

    #[test]
    fn sha_helper_matches_known_digest() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn weighting_choice_parses() {
        assert_eq!(WeightingChoice::parse("auto"), Ok(WeightingChoice::Auto));
        assert_eq!(WeightingChoice::parse("count"), Ok(WeightingChoice::Count));
        assert_eq!(WeightingChoice::parse("time"), Ok(WeightingChoice::Time));
        assert!(WeightingChoice::parse("wall").is_err());
    }
}
