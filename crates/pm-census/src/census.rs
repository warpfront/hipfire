// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Time-weighted PM / hipcc / railgun census of one profiled region of
//! interest (ROI), and the model x arch x phase matrix built from census
//! files.
//!
//! Everything here is measured from a rocprofv3 trace joined to the object
//! inventory. A kernel that never appears in the trace contributes nothing:
//! the census never invents a zero bucket for an absent family, and a matrix
//! cell without a real object-joined profile is `Unmeasured`, not zero.
//!
//! Time accounting (all integer nanoseconds):
//! * `kernel_sum_ns` — sum of dispatch durations (counts concurrent work twice);
//! * `kernel_busy_ns` — length of the union of dispatch intervals;
//! * `exclusive_ns` per dispatch — each elementary segment of the sweep line
//!   over the dispatch intervals is split equally among the dispatches active
//!   in it (the integer remainder goes to the earliest active dispatches), so
//!   the per-dispatch exclusive times add up to `kernel_busy_ns` exactly;
//! * copies are every `MemCopy` touching the agent (source or destination),
//!   clipped to the ROI; `copy_kernel_overlap_ns` is the length of the
//!   intersection of the kernel union and the copy union; `idle_ns` is the
//!   part of the ROI covered by neither.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use hipfire_atlas::workload::PhaseV1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::inventory::{self, Inventory, Origin};
use crate::rocprof::{self, Dispatch, Trace, TraceSource};

pub const CENSUS_SCHEMA: &str = "hipfire.pm_census.v1";
pub const MATRIX_SCHEMA: &str = "hipfire.pm_census_matrix.v1";

/// Models of the matrix, in cell order.
pub const MATRIX_MODELS: [&str; 3] = ["dense-27b", "a3b", "flash-next"];
/// Architectures of the matrix, in cell order.
pub const MATRIX_ARCHES: [&str; 3] = ["gfx1201", "gfx1151", "gfx1100"];

/// Unknown-origin share of `kernel_busy_ns` above which a census (and its
/// matrix cell) carries an explicit reason: PM / hipcc shares are then lower
/// bounds.
const UNKNOWN_SHARE_LIMIT: f64 = 0.05;

/// Prefill-pass rule: the pass prelude is searched this far before the
/// pass's first anchor dispatch.
const PREFILL_PRELUDE_NS: u64 = 5_000_000;

// ───────────────────────────── data model ─────────────────────────────

/// Region of interest of a census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoiSpec {
    /// First dispatch start .. last dispatch end on the agent.
    Full,
    /// Absolute trace timestamps.
    Window { start_ns: u64, end_ns: u64 },
    /// The interval of the single marker with this name.
    Marker { name: String },
    /// One prefill pass isolated by anchor dispatches (the rule of
    /// `fn-prefill-profile-002/analyze.py`): dispatches whose symbol starts
    /// with `anchor_prefix` and whose `grid x * grid y` is at least half the
    /// largest anchor grid are pass-chunk starts; every `chunks_per_pass`
    /// consecutive chunk starts form a pass; the last pass is measured.
    PrefillPass {
        anchor_prefix: String,
        chunks_per_pass: usize,
    },
}

impl RoiSpec {
    /// CLI form: `full`, `window:S:E`, `marker:NAME`, `prefill-pass:PREFIX:CPP`.
    pub fn parse(s: &str) -> Result<Self, String> {
        if s == "full" {
            return Ok(RoiSpec::Full);
        }
        if let Some(rest) = s.strip_prefix("window:") {
            let (a, b) = rest
                .split_once(':')
                .ok_or_else(|| format!("roi `{s}`: expected window:START_NS:END_NS"))?;
            let start_ns = a
                .parse::<u64>()
                .map_err(|e| format!("roi `{s}`: start `{a}`: {e}"))?;
            let end_ns = b
                .parse::<u64>()
                .map_err(|e| format!("roi `{s}`: end `{b}`: {e}"))?;
            return Ok(RoiSpec::Window { start_ns, end_ns });
        }
        if let Some(rest) = s.strip_prefix("marker:") {
            if rest.is_empty() {
                return Err(format!("roi `{s}`: empty marker name"));
            }
            return Ok(RoiSpec::Marker {
                name: rest.to_string(),
            });
        }
        if let Some(rest) = s.strip_prefix("prefill-pass:") {
            let (prefix, cpp) = rest.rsplit_once(':').ok_or_else(|| {
                format!("roi `{s}`: expected prefill-pass:PREFIX:CHUNKS_PER_PASS")
            })?;
            if prefix.is_empty() {
                return Err(format!("roi `{s}`: empty anchor prefix"));
            }
            let chunks_per_pass = cpp
                .parse::<usize>()
                .map_err(|e| format!("roi `{s}`: chunks per pass `{cpp}`: {e}"))?;
            return Ok(RoiSpec::PrefillPass {
                anchor_prefix: prefix.to_string(),
                chunks_per_pass,
            });
        }
        Err(format!(
            "unknown roi `{s}` (full | window:S:E | marker:NAME | prefill-pass:PREFIX:CPP)"
        ))
    }

    /// The string recorded in [`RoiV1::spec`]; round-trips through [`RoiSpec::parse`].
    pub fn label(&self) -> String {
        match self {
            RoiSpec::Full => "full".to_string(),
            RoiSpec::Window { start_ns, end_ns } => format!("window:{start_ns}:{end_ns}"),
            RoiSpec::Marker { name } => format!("marker:{name}"),
            RoiSpec::PrefillPass {
                anchor_prefix,
                chunks_per_pass,
            } => format!("prefill-pass:{anchor_prefix}:{chunks_per_pass}"),
        }
    }
}

/// One cell of the matrix: model x architecture x phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellKey {
    pub model: String,
    pub arch: String,
    pub phase: PhaseV1,
}

/// Per-group totals. `calls` counts dispatches, `kernels` distinct symbols.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bucket {
    pub calls: u64,
    pub kernels: u64,
    pub sum_ns: u64,
    pub exclusive_ns: u64,
}

/// One row of the per-kernel ranking (grouped by symbol, origin, join and
/// dispatch class so a symbol launched through two paths shows as two rows).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopKernel {
    pub symbol: String,
    pub origin: Origin,
    pub join: String,
    /// `hip_graph` | `hip` | `non_hip` | `unknown`.
    pub dispatch: String,
    /// `<contract id>@v<version>` when the symbol is an export of a
    /// `hipfire-kernel-abi` contract.
    pub abi: Option<String>,
    pub calls: u64,
    pub sum_ns: u64,
    pub exclusive_ns: u64,
    /// `100 * sum_ns / roi.wall_ns` (percent, 0..=100 per kernel; may exceed 100 in total under concurrency).
    pub pct_wall: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoiV1 {
    pub start_ns: u64,
    pub end_ns: u64,
    pub wall_ns: u64,
    /// [`RoiSpec::label`].
    pub spec: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CensusV1 {
    pub schema: String,
    pub cell: CellKey,
    pub capture: TraceSource,
    pub inventory_commit: Option<String>,
    pub agent: u64,
    pub roi: RoiV1,
    pub kernels_in_roi: u64,
    pub kernel_sum_ns: u64,
    pub kernel_busy_ns: u64,
    pub copy_busy_ns: u64,
    pub copy_kernel_overlap_ns: u64,
    pub idle_ns: u64,
    /// Keys: `pm` | `hipcc` | `unknown` (only those seen in the ROI).
    pub by_origin: BTreeMap<String, Bucket>,
    /// Keys: `hip_graph` | `hip` | `non_hip` | `unknown` (only those seen).
    pub by_dispatch: BTreeMap<String, Bucket>,
    /// Keys: `<contract id>@v<version>` (only contracts seen).
    pub by_contract: BTreeMap<String, Bucket>,
    pub top: Vec<TopKernel>,
    pub notes: Vec<String>,
    /// How this census was measured. Absent (every rocprofv3 census) =
    /// time-weighted from a rocprofv3 kernel trace. Present for dispatch-ledger
    /// cells (`pm-census ledger`), where it states the weighting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<MeasurementV1>,
}

/// What a ledger census weights its shares by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    /// Shares are of GPU time (nanoseconds).
    Time,
    /// Shares are of executed dispatches. NOT a time share.
    Count,
}

/// Which instrument produced a ledger census.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementMode {
    /// Retained-PM4 dispatch ledger, default tape: exact dispatch counts x
    /// replays; per-kernel time is unknown.
    LedgerDispatchCount,
    /// Retained-PM4 dispatch ledger on the per-dispatch-timestamp tape
    /// (`HIPFIRE_REDLINE_LEDGER_TIMING=1`): per-dispatch GPU spans from a
    /// perturbed run.
    LedgerPm4Timestamps,
}

/// The mode/weighting/perturbation triple a matrix cell repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementLabel {
    pub mode: MeasurementMode,
    pub weighting: Weighting,
    /// The measured run executed a different tape than the product run.
    pub perturbing: bool,
}

/// Measurement record of a ledger census.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementV1 {
    pub mode: MeasurementMode,
    pub weighting: Weighting,
    pub perturbing: bool,
    /// What the census covers and what it cannot see.
    pub scope: String,
    /// Retained programs with at least one counted replay.
    pub programs: u64,
    /// Counted replays summed over programs.
    pub replays: u64,
    /// Dispatches executed = sum over programs of replays x dispatches.
    pub dispatches_executed: u64,
    /// Whole-program GPU span over the counted replays (ns), from the
    /// tape's bracketing timestamps. Time-weighted but whole-program only:
    /// it carries no per-kernel information.
    pub program_gpu_span_ns: u64,
    /// Share of executed dispatches per origin (`pm`/`hipcc`/`unknown`).
    /// A count share in every mode; compare `by_origin[*].exclusive_ns`
    /// for time (non-zero only when `weighting` is `time`).
    pub dispatch_share_by_origin: BTreeMap<String, f64>,
}

impl MeasurementV1 {
    pub fn label(&self) -> MeasurementLabel {
        MeasurementLabel {
            mode: self.mode,
            weighting: self.weighting,
            perturbing: self.perturbing,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CellStatus {
    /// A census file exists for the cell.
    Measured { census: String, sha256: String },
    /// No profile joined to objects exists for the cell.
    Unmeasured { reason: String },
    /// The cell was examined and cannot be measured on purpose.
    Refused { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatrixCell {
    pub key: CellKey,
    pub status: CellStatus,
    /// PM exclusive time / kernel busy time. `None` unless `Measured`.
    pub pm_share_of_kernel_busy: Option<f64>,
    /// PM exclusive time / ROI wall time. `None` unless `Measured`.
    pub pm_share_of_wall: Option<f64>,
    /// PM dispatches / all executed dispatches. Only for count-weighted
    /// ledger cells (where the two time shares above are `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pm_share_of_dispatches: Option<f64>,
    /// How the cell was measured; absent = a rocprofv3 kernel trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<MeasurementLabel>,
    /// Reasons that qualify the shares (e.g. an unknown-origin share above
    /// 5 % makes them lower bounds). Empty for a clean measurement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatrixV1 {
    pub schema: String,
    pub cells: Vec<MatrixCell>,
}

// ───────────────────────── interval arithmetic ─────────────────────────

type Iv = (u64, u64);

/// Sorted, merged (touching intervals merge), empty intervals dropped.
fn union(mut iv: Vec<Iv>) -> Vec<Iv> {
    iv.retain(|&(s, e)| e > s);
    iv.sort_unstable();
    let mut out: Vec<Iv> = Vec::with_capacity(iv.len());
    for (s, e) in iv {
        match out.last_mut() {
            Some(last) if s <= last.1 => {
                if e > last.1 {
                    last.1 = e;
                }
            }
            _ => out.push((s, e)),
        }
    }
    out
}

fn total(u: &[Iv]) -> u64 {
    u.iter().map(|&(s, e)| e - s).sum()
}

/// Length of the intersection of two unions (sorted, disjoint).
fn intersect_len(a: &[Iv], b: &[Iv]) -> u64 {
    let (mut i, mut j, mut acc) = (0, 0, 0u64);
    while i < a.len() && j < b.len() {
        let s = a[i].0.max(b[j].0);
        let e = a[i].1.min(b[j].1);
        if e > s {
            acc += e - s;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    acc
}

/// Sweep-line exclusive time: each elementary segment is split equally among
/// the intervals active in it; the integer remainder of a segment goes to the
/// first (lowest-index) active intervals. Σ result == |union(iv)| exactly.
fn exclusive_shares(iv: &[Iv]) -> Vec<u64> {
    let mut out = vec![0u64; iv.len()];
    let mut ev: Vec<(u64, bool, usize)> = Vec::with_capacity(iv.len() * 2);
    for (i, &(s, e)) in iv.iter().enumerate() {
        if e > s {
            ev.push((s, true, i));
            ev.push((e, false, i));
        }
    }
    ev.sort_unstable();
    let mut active: BTreeSet<usize> = BTreeSet::new();
    let mut prev = 0u64;
    let mut idx = 0;
    while idx < ev.len() {
        let t = ev[idx].0;
        if !active.is_empty() && t > prev {
            let len = t - prev;
            let k = active.len() as u64;
            let base = len / k;
            let rem = (len % k) as usize;
            for (n, &i) in active.iter().enumerate() {
                out[i] += base + u64::from(n < rem);
            }
        }
        while idx < ev.len() && ev[idx].0 == t {
            let (_, is_start, i) = ev[idx];
            if is_start {
                active.insert(i);
            } else {
                active.remove(&i);
            }
            idx += 1;
        }
        prev = t;
    }
    out
}

// ───────────────────────────── helpers ─────────────────────────────

fn phase_str(p: &PhaseV1) -> &'static str {
    match p {
        PhaseV1::Prefill => "prefill",
        PhaseV1::Ar => "ar",
        PhaseV1::Spec => "spec",
    }
}

pub(crate) fn parse_phase(s: &str) -> Result<PhaseV1, String> {
    match s {
        "prefill" => Ok(PhaseV1::Prefill),
        "ar" => Ok(PhaseV1::Ar),
        "spec" => Ok(PhaseV1::Spec),
        _ => Err(format!("unknown phase `{s}` (prefill | ar | spec)")),
    }
}

fn same_key(a: &CellKey, b: &CellKey) -> bool {
    a.model == b.model && a.arch == b.arch && phase_str(&a.phase) == phase_str(&b.phase)
}

fn key_label(k: &CellKey) -> String {
    format!("{}:{}:{}", k.model, k.arch, phase_str(&k.phase))
}

fn in_grid(k: &CellKey) -> bool {
    MATRIX_MODELS.contains(&k.model.as_str()) && MATRIX_ARCHES.contains(&k.arch.as_str())
}

pub(crate) fn origin_key(o: &Origin) -> &'static str {
    match o {
        Origin::Pm => "pm",
        Origin::Hipcc => "hipcc",
        Origin::Unknown => "unknown",
    }
}

pub(crate) fn origin_from_key(k: &str) -> Origin {
    match k {
        "pm" => Origin::Pm,
        "hipcc" => Origin::Hipcc,
        _ => Origin::Unknown,
    }
}

fn dispatch_class(d: &Dispatch) -> &'static str {
    if d.graph_exec_id != 0 {
        "hip_graph"
    } else {
        match d.hip_api {
            Some(true) => "hip",
            Some(false) => "non_hip",
            None => "unknown",
        }
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Export symbol (any arch) -> `<contract id>@v<version>` over every
/// `hipfire-kernel-abi` contract. A symbol exported by several contracts maps
/// to all of them joined by `|` (never silently to one).
pub(crate) fn contract_index() -> HashMap<&'static str, String> {
    let mut by_symbol: HashMap<&'static str, Vec<String>> = HashMap::new();
    for abi in hipfire_kernel_abi::families::all() {
        let id = format!("{}@v{}", abi.id, abi.version);
        for e in abi.exports {
            let v = by_symbol.entry(e.symbol).or_default();
            if !v.contains(&id) {
                v.push(id.clone());
            }
        }
    }
    by_symbol
        .into_iter()
        .map(|(sym, mut ids)| {
            ids.sort();
            (sym, ids.join("|"))
        })
        .collect()
}

#[derive(Default)]
struct Agg {
    calls: u64,
    sum_ns: u64,
    exclusive_ns: u64,
    symbols: BTreeSet<usize>,
}

impl Agg {
    fn add(&mut self, dur: u64, excl: u64, sym: usize) {
        self.calls += 1;
        self.sum_ns += dur;
        self.exclusive_ns += excl;
        self.symbols.insert(sym);
    }
    fn into_bucket(self) -> Bucket {
        Bucket {
            calls: self.calls,
            kernels: self.symbols.len() as u64,
            sum_ns: self.sum_ns,
            exclusive_ns: self.exclusive_ns,
        }
    }
}

fn buckets<K: ToString>(m: BTreeMap<K, Agg>) -> BTreeMap<String, Bucket> {
    m.into_iter()
        .map(|(k, a)| (k.to_string(), a.into_bucket()))
        .collect()
}

/// Per-kernel-id facts resolved once.
struct KInfo {
    symbol: String,
    sym_id: usize,
    origin: &'static str,
    join: String,
    abi: Option<String>,
}

// ───────────────────────────── ROI rules ─────────────────────────────

/// Resolved ROI: dispatches are included iff `start in [start_ns, select_end)`;
/// the ROI end is the larger of `min_end` and the latest end of an included dispatch.
struct Resolved {
    start_ns: u64,
    min_end: u64,
    select_end: u64,
}

fn grid_xy(d: &Dispatch) -> u128 {
    u128::from(d.grid_items[0]) * u128::from(d.grid_items[1])
}

/// The `analyze.py` pass rule. `ds` is sorted by start (stable).
///
/// * anchors = dispatches whose symbol starts with `prefix`;
/// * chunk starts = anchors with `grid x*y >= 0.5 * max anchor grid x*y`;
/// * passes = consecutive groups of `cpp` chunk starts, measured = the last;
/// * start = earliest dispatch starting in `[c1 - 5 ms, c1]` (`c1` = the pass's
///   first chunk start; the anchor itself qualifies);
/// * selection ends at the first anchor of ANY grid size starting after the
///   pass's last chunk (a one-row decode embedding ends the window), else at
///   the end of the last-starting dispatch.
fn resolve_prefill(
    trace: &Trace,
    ds: &[&Dispatch],
    prefix: &str,
    cpp: usize,
) -> Result<Resolved, String> {
    if cpp == 0 {
        return Err("prefill-pass: chunks_per_pass must be >= 1".to_string());
    }
    let anchors: Vec<&Dispatch> = ds
        .iter()
        .copied()
        .filter(|d| {
            trace
                .kernel(d.kernel_id)
                .is_some_and(|k| k.name.starts_with(prefix))
        })
        .collect();
    let gmax =
        anchors.iter().map(|d| grid_xy(d)).max().ok_or_else(|| {
            format!("prefill-pass: no dispatch whose symbol starts with `{prefix}`")
        })?;
    let chunk_starts: Vec<u64> = anchors
        .iter()
        .filter(|d| 2 * grid_xy(d) >= gmax)
        .map(|d| d.start_ns)
        .collect();
    let pass = chunk_starts
        .chunks(cpp)
        .last()
        .ok_or_else(|| "prefill-pass: no chunk-start anchors".to_string())?;
    let c1 = pass[0];
    let pass_last = pass[pass.len() - 1];
    let select_end = match anchors.iter().find(|d| d.start_ns > pass_last) {
        Some(next) => next.start_ns,
        None => ds.last().map(|d| d.end_ns).unwrap_or(c1),
    };
    let lo = c1.saturating_sub(PREFILL_PRELUDE_NS);
    let start_ns = ds
        .iter()
        .filter(|d| d.start_ns >= lo && d.start_ns <= c1)
        .map(|d| d.start_ns)
        .min()
        .unwrap_or(c1);
    Ok(Resolved {
        start_ns,
        min_end: 0,
        select_end,
    })
}

fn resolve_roi(trace: &Trace, ds: &[&Dispatch], spec: &RoiSpec) -> Result<Resolved, String> {
    match spec {
        RoiSpec::Full => Ok(Resolved {
            start_ns: ds[0].start_ns,
            min_end: 0,
            select_end: u64::MAX,
        }),
        RoiSpec::Window { start_ns, end_ns } => {
            if end_ns <= start_ns {
                return Err(format!("window roi: end {end_ns} <= start {start_ns}"));
            }
            Ok(Resolved {
                start_ns: *start_ns,
                min_end: *end_ns,
                select_end: *end_ns,
            })
        }
        RoiSpec::Marker { name } => {
            let found: Vec<_> = trace.markers.iter().filter(|m| &m.name == name).collect();
            match found.as_slice() {
                [] => Err(format!("marker roi: no marker named `{name}` in the trace")),
                [m] => {
                    if m.end_ns <= m.start_ns {
                        return Err(format!(
                            "marker roi: marker `{name}` has an empty interval [{}, {})",
                            m.start_ns, m.end_ns
                        ));
                    }
                    Ok(Resolved {
                        start_ns: m.start_ns,
                        min_end: m.end_ns,
                        select_end: m.end_ns,
                    })
                }
                many => Err(format!(
                    "marker roi: {} markers named `{name}` (ambiguous; use window:S:E)",
                    many.len()
                )),
            }
        }
        RoiSpec::PrefillPass {
            anchor_prefix,
            chunks_per_pass,
        } => resolve_prefill(trace, ds, anchor_prefix, *chunks_per_pass),
    }
}

pub(crate) fn resolve_agent(trace: &Trace, agent: Option<u64>) -> Result<u64, String> {
    let with: BTreeSet<u64> = trace.dispatches.iter().map(|d| d.agent).collect();
    if let Some(a) = agent {
        return if with.contains(&a) {
            Ok(a)
        } else {
            Err(format!("agent {a} has no dispatches in the trace"))
        };
    }
    let gpu: BTreeSet<u64> = trace.gpu_agents().map(|a| a.id).collect();
    let cands: Vec<u64> = if gpu.is_empty() {
        with.iter().copied().collect()
    } else {
        with.iter().copied().filter(|a| gpu.contains(a)).collect()
    };
    match cands.as_slice() {
        [] => Err("trace has no GPU agent with dispatches".to_string()),
        [a] => Ok(*a),
        many => Err(format!(
            "trace has {} GPU agents with dispatches ({many:?}); pass --agent",
            many.len()
        )),
    }
}

// ───────────────────────────── census ─────────────────────────────

/// Census of one ROI of `trace` on one GPU agent, origin-joined through `inv`.
/// `top` is the number of per-kernel rows kept.
pub fn census(
    trace: &Trace,
    inv: &Inventory,
    roi: &RoiSpec,
    agent: Option<u64>,
    key: CellKey,
    top: usize,
) -> Result<CensusV1, String> {
    let agent = resolve_agent(trace, agent)?;
    let mut ds: Vec<&Dispatch> = trace
        .dispatches
        .iter()
        .filter(|d| d.agent == agent)
        .collect();
    ds.sort_by_key(|d| d.start_ns);
    if ds.is_empty() {
        return Err(format!("agent {agent} has no dispatches in the trace"));
    }
    let r = resolve_roi(trace, &ds, roi)?;

    let included: Vec<&Dispatch> = ds
        .iter()
        .copied()
        .filter(|d| d.start_ns >= r.start_ns && d.start_ns < r.select_end)
        .collect();
    if included.is_empty() {
        return Err(format!(
            "roi `{}` selects no dispatches on agent {agent}",
            roi.label()
        ));
    }
    let iv: Vec<Iv> = included
        .iter()
        .map(|d| (d.start_ns, d.start_ns + d.end_ns.saturating_sub(d.start_ns)))
        .collect();
    let max_end = iv.iter().map(|&(_, e)| e).max().unwrap_or(r.start_ns);
    let roi_start = r.start_ns;
    let roi_end = r.min_end.max(max_end);
    let wall = roi_end - roi_start;

    let mut notes: Vec<String> = Vec::new();
    if roi_end > r.min_end && r.min_end != 0 {
        notes.push(format!(
            "roi end extended from {} to {} ns: dispatches started inside the interval finish after it",
            r.min_end, roi_end
        ));
    }
    if let Some(a) = trace.agents.iter().find(|a| a.id == agent) {
        let name = a.name.split(':').next().unwrap_or(&a.name);
        if name != key.arch {
            notes.push(format!(
                "trace agent {agent} reports `{}` but the cell arch is `{}`",
                a.name, key.arch
            ));
        }
    }

    // Kernel and copy unions inside the ROI.
    let ku = union(iv.clone());
    let kernel_busy = total(&ku);
    let kernel_sum: u64 = iv.iter().map(|&(s, e)| e - s).sum();
    let copies: Vec<Iv> = trace
        .copies
        .iter()
        .filter(|c| c.src_agent == agent || c.dst_agent == agent)
        .map(|c| (c.start_ns.max(roi_start), c.end_ns.min(roi_end)))
        .filter(|&(s, e)| e > s)
        .collect();
    let cu = union(copies);
    let copy_busy = total(&cu);
    let overlap = intersect_len(&ku, &cu);
    let all = union(ku.iter().chain(cu.iter()).copied().collect());
    let idle = wall.saturating_sub(total(&all));

    // Exclusive time.
    let excl = exclusive_shares(&iv);
    let excl_sum: u64 = excl.iter().sum();
    if excl_sum != kernel_busy {
        return Err(format!(
            "internal: exclusive time {excl_sum} ns != kernel busy time {kernel_busy} ns"
        ));
    }

    // Per-kernel-id attribution, resolved once per id.
    let contracts = contract_index();
    let mut interned: HashMap<String, usize> = HashMap::new();
    let mut infos: BTreeMap<u64, KInfo> = BTreeMap::new();
    for d in &included {
        if infos.contains_key(&d.kernel_id) {
            continue;
        }
        let (symbol, origin, join) = match trace.kernel(d.kernel_id) {
            Some(k) => {
                let at = inv.attribute(trace, k);
                (k.name.clone(), origin_key(&at.origin), at.join)
            }
            None => (
                format!("<kernel_id {}>", d.kernel_id),
                "unknown",
                "kernel_id has no symbol record in the trace".to_string(),
            ),
        };
        let next = interned.len();
        let sym_id = *interned.entry(symbol.clone()).or_insert(next);
        let abi = contracts.get(symbol.as_str()).cloned();
        infos.insert(
            d.kernel_id,
            KInfo {
                symbol,
                sym_id,
                origin,
                join,
                abi,
            },
        );
    }

    let mut by_origin: BTreeMap<&str, Agg> = BTreeMap::new();
    let mut by_dispatch: BTreeMap<&str, Agg> = BTreeMap::new();
    let mut by_contract: BTreeMap<&str, Agg> = BTreeMap::new();
    let mut per_kernel: BTreeMap<(u64, &'static str), Agg> = BTreeMap::new();
    for (i, d) in included.iter().enumerate() {
        let info = &infos[&d.kernel_id];
        let (dur, ex) = (iv[i].1 - iv[i].0, excl[i]);
        let class = dispatch_class(d);
        by_origin
            .entry(info.origin)
            .or_default()
            .add(dur, ex, info.sym_id);
        by_dispatch
            .entry(class)
            .or_default()
            .add(dur, ex, info.sym_id);
        if let Some(c) = &info.abi {
            by_contract
                .entry(c.as_str())
                .or_default()
                .add(dur, ex, info.sym_id);
        }
        per_kernel
            .entry((d.kernel_id, class))
            .or_default()
            .add(dur, ex, info.sym_id);
    }

    // Per-kernel ranking: fold kernel ids that share symbol/origin/join/class.
    type Row = (u64, u64, u64);
    let mut rows: BTreeMap<(String, &'static str, String, &'static str), Row> = BTreeMap::new();
    for ((kid, class), a) in &per_kernel {
        let info = &infos[kid];
        let row = rows
            .entry((info.symbol.clone(), info.origin, info.join.clone(), *class))
            .or_default();
        row.0 += a.calls;
        row.1 += a.sum_ns;
        row.2 += a.exclusive_ns;
    }
    let mut top_rows: Vec<TopKernel> = rows
        .into_iter()
        .map(
            |((symbol, origin, join, class), (calls, sum_ns, exclusive_ns))| {
                let abi = contracts.get(symbol.as_str()).cloned();
                TopKernel {
                    symbol,
                    origin: origin_from_key(origin),
                    join,
                    dispatch: class.to_string(),
                    abi,
                    calls,
                    sum_ns,
                    exclusive_ns,
                    pct_wall: 100.0 * sum_ns as f64 / wall.max(1) as f64,
                }
            },
        )
        .collect();
    top_rows.sort_by(|a, b| b.sum_ns.cmp(&a.sum_ns));
    top_rows.truncate(top);

    if let Some(a) = by_dispatch.get("non_hip") {
        notes.push(format!(
            "{} dispatches carry no HIP API record (non_hip): submitted outside HIP \
             (Redline/railgun PM4 or direct HSA)",
            a.calls
        ));
    }
    if let Some(a) = by_dispatch.get("unknown") {
        notes.push(format!(
            "{} dispatches: the trace has no HIP API records, so the submission path is unknown",
            a.calls
        ));
    }
    if let Some(a) = by_origin.get("unknown") {
        let share = a.exclusive_ns as f64 / kernel_busy.max(1) as f64;
        if share > UNKNOWN_SHARE_LIMIT {
            notes.push(format!(
                "unknown share {:.1}% of kernel_busy: those kernels join to no PM object and no \
                 hipcc kernel name, so pm/hipcc shares are lower bounds",
                100.0 * share
            ));
        }
    }

    Ok(CensusV1 {
        schema: CENSUS_SCHEMA.to_string(),
        cell: key,
        capture: trace.source.clone(),
        inventory_commit: inv.commit.clone(),
        agent,
        roi: RoiV1 {
            start_ns: roi_start,
            end_ns: roi_end,
            wall_ns: wall,
            spec: roi.label(),
        },
        kernels_in_roi: included.len() as u64,
        kernel_sum_ns: kernel_sum,
        kernel_busy_ns: kernel_busy,
        copy_busy_ns: copy_busy,
        copy_kernel_overlap_ns: overlap,
        idle_ns: idle,
        by_origin: buckets(by_origin),
        by_dispatch: buckets(by_dispatch),
        by_contract: buckets(by_contract),
        top: top_rows,
        notes,
        measurement: None,
    })
}

// ───────────────────────────── matrix ─────────────────────────────

/// The 27-cell matrix (3 models x 3 arches x 3 phases, in that nesting order).
/// `measured` are `(census path, census file sha256, census)`; a cell is
/// `Measured` only from a census whose key matches exactly. A cell with two
/// census files, or with a census and a refusal, is `Refused` with both facts
/// stated — the matrix never silently picks one.
pub fn matrix(measured: &[(String, String, CensusV1)], refused: &[(CellKey, String)]) -> MatrixV1 {
    let mut cells = Vec::with_capacity(MATRIX_MODELS.len() * MATRIX_ARCHES.len() * 3);
    for model in MATRIX_MODELS {
        for arch in MATRIX_ARCHES {
            for phase in [PhaseV1::Prefill, PhaseV1::Ar, PhaseV1::Spec] {
                let key = CellKey {
                    model: model.to_string(),
                    arch: arch.to_string(),
                    phase,
                };
                let found: Vec<&(String, String, CensusV1)> = measured
                    .iter()
                    .filter(|(_, _, c)| same_key(&c.cell, &key))
                    .collect();
                let reasons: Vec<&str> = refused
                    .iter()
                    .filter(|(k, _)| same_key(k, &key))
                    .map(|(_, why)| why.as_str())
                    .collect();
                cells.push(matrix_cell(key, &found, &reasons));
            }
        }
    }
    MatrixV1 {
        schema: MATRIX_SCHEMA.to_string(),
        cells,
    }
}

fn matrix_cell(
    key: CellKey,
    found: &[&(String, String, CensusV1)],
    reasons: &[&str],
) -> MatrixCell {
    let paths = || {
        found
            .iter()
            .map(|(p, _, _)| p.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    fn refused_cell(key: &CellKey, reason: String) -> MatrixCell {
        MatrixCell {
            key: key.clone(),
            status: CellStatus::Refused { reason },
            pm_share_of_kernel_busy: None,
            pm_share_of_wall: None,
            pm_share_of_dispatches: None,
            measurement: None,
            notes: Vec::new(),
        }
    }
    if !reasons.is_empty() {
        let mut reason = reasons.join("; ");
        if !found.is_empty() {
            reason.push_str(&format!(" (census also supplied: {})", paths()));
        }
        return refused_cell(&key, reason);
    }
    match found {
        [] => MatrixCell {
            key,
            status: CellStatus::Unmeasured {
                reason: "no object-joined profile supplied".to_string(),
            },
            pm_share_of_kernel_busy: None,
            pm_share_of_wall: None,
            pm_share_of_dispatches: None,
            measurement: None,
            notes: Vec::new(),
        },
        [(path, sha, c)] => {
            let mut notes = Vec::new();
            let busy = c.kernel_busy_ns;
            let count_weighted = c
                .measurement
                .as_ref()
                .is_some_and(|m| m.weighting == Weighting::Count);
            let mut share_dispatches = None;
            let (share_busy, share_wall) =
                if count_weighted {
                    let total: u64 = c.by_origin.values().map(|b| b.calls).sum();
                    if total == 0 {
                        notes.push("census has no dispatches; shares not computed".to_string());
                    } else {
                        let calls = |origin: &str| c.by_origin.get(origin).map_or(0, |b| b.calls);
                        share_dispatches = Some(calls("pm") as f64 / total as f64);
                        let unknown_share = calls("unknown") as f64 / total as f64;
                        if unknown_share > UNKNOWN_SHARE_LIMIT {
                            notes.push(format!(
                            "unknown share {:.1}% of dispatches exceeds {:.0}%: the share counts \
                             only the joined dispatches (lower bound)",
                            100.0 * unknown_share,
                            100.0 * UNKNOWN_SHARE_LIMIT
                        ));
                        }
                    }
                    notes.push(
                    "count-weighted ledger cell: pm_share_of_dispatches is a share of executed \
                     dispatches, NOT of GPU time; pm_share_of_kernel_busy and pm_share_of_wall \
                     are deliberately not computed"
                        .to_string(),
                );
                    (None, None)
                } else if busy == 0 || c.roi.wall_ns == 0 {
                    notes.push("census has no kernel busy time; shares not computed".to_string());
                    (None, None)
                } else {
                    let pm = c.by_origin.get("pm").map_or(0, |b| b.exclusive_ns) as f64;
                    let unknown = c.by_origin.get("unknown").map_or(0, |b| b.exclusive_ns) as f64;
                    let unknown_share = unknown / busy as f64;
                    if unknown_share > UNKNOWN_SHARE_LIMIT {
                        notes.push(format!(
                        "unknown share {:.1}% of kernel_busy exceeds {:.0}%: shares count only \
                         the joined kernels (lower bound)",
                        100.0 * unknown_share,
                        100.0 * UNKNOWN_SHARE_LIMIT
                    ));
                    }
                    if let Some(m) = &c.measurement {
                        notes.push(format!(
                        "{:?} cell{}: shares are of the summed retained-program GPU span, not of \
                         end-to-end request time",
                        m.mode,
                        if m.perturbing { " (perturbing run)" } else { "" }
                    ));
                    }
                    (Some(pm / busy as f64), Some(pm / c.roi.wall_ns as f64))
                };
            MatrixCell {
                key,
                status: CellStatus::Measured {
                    census: path.clone(),
                    sha256: sha.clone(),
                },
                pm_share_of_kernel_busy: share_busy,
                pm_share_of_wall: share_wall,
                pm_share_of_dispatches: share_dispatches,
                measurement: c.measurement.as_ref().map(MeasurementV1::label),
                notes,
            }
        }
        many => refused_cell(
            &key,
            format!(
                "ambiguous: {} census files for this cell ({})",
                many.len(),
                paths()
            ),
        ),
    }
}

// ───────────────────────────── CLI ─────────────────────────────

const USAGE: &str = "\
pm-census census --trace PATH --model M --arch A --phase P
                 [--roi full|window:S:E|marker:NAME|prefill-pass:PREFIX:CPP]
                 [--agent ID] [--repo DIR] [--commit SHA] [--top N]
    Time-weighted PM/hipcc census of one ROI of a rocprofv3 trace; prints CensusV1 JSON.
    model: dense-27b | a3b | flash-next   arch: gfx1201 | gfx1151 | gfx1100   phase: prefill | ar | spec
pm-census census matrix [--refuse model:arch:phase:reason ...] CENSUS.json...
    Prints the 27-cell MatrixV1 JSON; cells without a census are Unmeasured.";

fn flag_value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

pub fn cli(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None => Err(format!("missing arguments\n{USAGE}")),
        Some("--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("matrix") => cli_matrix(&args[1..]),
        Some(_) => cli_census(args),
    }
}

fn cli_census(args: &[String]) -> Result<(), String> {
    let (mut trace, mut model, mut arch, mut phase) = (None, None, None, None);
    let mut roi = RoiSpec::Full;
    let mut agent: Option<u64> = None;
    let mut repo = ".".to_string();
    let mut commit: Option<String> = None;
    let mut top = 25usize;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--trace" => trace = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--model" => model = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--arch" => arch = Some(flag_value(args, &mut i, flag)?.to_string()),
            "--phase" => phase = Some(parse_phase(flag_value(args, &mut i, flag)?)?),
            "--roi" => roi = RoiSpec::parse(flag_value(args, &mut i, flag)?)?,
            "--agent" => {
                let v = flag_value(args, &mut i, flag)?;
                agent = Some(
                    v.parse::<u64>()
                        .map_err(|e| format!("--agent `{v}`: {e}"))?,
                );
            }
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
    let trace_path = need(trace, "--trace")?;
    let key = CellKey {
        model: need(model, "--model")?,
        arch: need(arch, "--arch")?,
        phase: phase.ok_or_else(|| format!("--phase is required\n{USAGE}"))?,
    };
    let tr = rocprof::load(Path::new(&trace_path))?;
    let inv = inventory::build(Path::new(&repo), commit.as_deref())?;
    let c = census(&tr, &inv, &roi, agent, key, top)?;
    let json = serde_json::to_string_pretty(&c).map_err(|e| format!("serialize census: {e}"))?;
    println!("{json}");
    Ok(())
}

fn cli_matrix(args: &[String]) -> Result<(), String> {
    let mut refused: Vec<(CellKey, String)> = Vec::new();
    let mut paths: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--refuse" => {
                let v = flag_value(args, &mut i, a)?;
                let mut it = v.splitn(4, ':');
                let (Some(m), Some(ar), Some(p), Some(why)) =
                    (it.next(), it.next(), it.next(), it.next())
                else {
                    return Err(format!("--refuse `{v}`: expected model:arch:phase:reason"));
                };
                if why.is_empty() {
                    return Err(format!("--refuse `{v}`: empty reason"));
                }
                let key = CellKey {
                    model: m.to_string(),
                    arch: ar.to_string(),
                    phase: parse_phase(p)?,
                };
                if !in_grid(&key) {
                    return Err(format!(
                        "--refuse `{v}`: cell {} is not in the matrix",
                        key_label(&key)
                    ));
                }
                refused.push((key, why.to_string()));
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            s if s.starts_with("--") => return Err(format!("unknown argument `{s}`\n{USAGE}")),
            s => paths.push(s),
        }
        i += 1;
    }
    let mut measured: Vec<(String, String, CensusV1)> = Vec::with_capacity(paths.len());
    for p in paths {
        let bytes = std::fs::read(p).map_err(|e| format!("read {p}: {e}"))?;
        let c: CensusV1 =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {p} as CensusV1: {e}"))?;
        if c.schema != CENSUS_SCHEMA {
            return Err(format!(
                "{p}: schema `{}`, expected `{CENSUS_SCHEMA}`",
                c.schema
            ));
        }
        if !in_grid(&c.cell) {
            return Err(format!(
                "{p}: cell {} is not in the matrix",
                key_label(&c.cell)
            ));
        }
        measured.push((p.to_string(), sha256_hex(&bytes), c));
    }
    let m = matrix(&measured, &refused);
    let json = serde_json::to_string_pretty(&m).map_err(|e| format!("serialize matrix: {e}"))?;
    println!("{json}");
    Ok(())
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocprof::{Agent, KernelSymbol, Marker, MemCopy};

    const U: u64 = 1_000_000; // 1 ms in ns

    fn key() -> CellKey {
        CellKey {
            model: "flash-next".into(),
            arch: "gfx1201".into(),
            phase: PhaseV1::Prefill,
        }
    }

    fn inv(hip: &[&str]) -> Inventory {
        Inventory {
            repo: "synthetic".into(),
            commit: None,
            pm_objects: Vec::new(),
            hip_kernel_names: hip.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn disp(
        kid: u64,
        agent: u64,
        s: u64,
        e: u64,
        grid: [u32; 3],
        graph: u64,
        api: Option<bool>,
    ) -> Dispatch {
        Dispatch {
            kernel_id: kid,
            agent,
            queue: 1,
            correlation: 0,
            start_ns: s,
            end_ns: e,
            workgroup: [256, 1, 1],
            grid_items: grid,
            group_segment_size: 0,
            private_segment_size: 0,
            graph_exec_id: graph,
            hip_api: api,
        }
    }

    fn cp(s: u64, e: u64, src: u64, dst: u64) -> MemCopy {
        MemCopy {
            start_ns: s,
            end_ns: e,
            bytes: 1024,
            kind: "h2d".into(),
            src_agent: src,
            dst_agent: dst,
        }
    }

    /// Agent 0 = CPU, 1 = the GPU; kernel id = index in `names`.
    fn trace_of(
        names: &[&str],
        dispatches: Vec<Dispatch>,
        copies: Vec<MemCopy>,
        markers: Vec<Marker>,
    ) -> Trace {
        Trace {
            source: TraceSource {
                path: "synthetic".into(),
                sha256: "0".repeat(64),
                format: "rocprofv3-json".into(),
            },
            agents: vec![
                Agent {
                    id: 0,
                    kind: "cpu".into(),
                    name: "cpu".into(),
                    ordinal: None,
                },
                Agent {
                    id: 1,
                    kind: "gpu".into(),
                    name: "gfx1201".into(),
                    ordinal: Some(0),
                },
            ],
            code_objects: Vec::new(),
            kernels: names
                .iter()
                .enumerate()
                .map(|(i, n)| KernelSymbol {
                    kernel_id: i as u64,
                    code_object_id: None,
                    name: n.to_string(),
                    kernarg_segment_size: None,
                    group_segment_size: None,
                    private_segment_size: None,
                    sgpr_count: None,
                    vgpr_count: None,
                })
                .collect(),
            dispatches,
            copies,
            markers,
        }
    }

    #[test]
    fn union_merges_nested_overlapping_touching_and_drops_empty() {
        let u = union(vec![
            (10, 20),
            (12, 15),
            (18, 30),
            (30, 35),
            (50, 60),
            (70, 70),
            (80, 75),
        ]);
        assert_eq!(u, vec![(10, 35), (50, 60)]);
        assert_eq!(total(&u), 35);
        assert!(union(Vec::new()).is_empty());
    }

    #[test]
    fn intersect_len_covers_partial_nested_and_disjoint() {
        let a = union(vec![(0, 10), (20, 30)]);
        let b = union(vec![(5, 25)]);
        assert_eq!(intersect_len(&a, &b), 5 + 5);
        assert_eq!(intersect_len(&a, &union(vec![(40, 50)])), 0);
        assert_eq!(intersect_len(&a, &a), 20);
        let nested = union(vec![(22, 24)]);
        assert_eq!(intersect_len(&a, &nested), 2);
    }

    #[test]
    fn exclusive_nested_overlapping_disjoint() {
        // nested: A[0,10) B[2,4): A alone 0-2 and 4-10, shared 2-4
        assert_eq!(exclusive_shares(&[(0, 10), (2, 4)]), vec![9, 1]);
        // overlapping: A[0,10) B[4,14)
        assert_eq!(exclusive_shares(&[(0, 10), (4, 14)]), vec![7, 7]);
        // disjoint
        assert_eq!(exclusive_shares(&[(0, 3), (5, 8)]), vec![3, 3]);
        // identical, odd length: remainder goes to the earliest, sum exact
        let e = exclusive_shares(&[(0, 5), (0, 5)]);
        assert_eq!(e, vec![3, 2]);
        // empty interval contributes nothing; sum == union length
        let iv = [(0, 9), (3, 12), (6, 6), (7, 20), (30, 31)];
        let e = exclusive_shares(&iv);
        assert_eq!(e[2], 0);
        assert_eq!(e.iter().sum::<u64>(), total(&union(iv.to_vec())));
    }

    #[test]
    fn census_window_clips_copies_and_classifies_dispatches() {
        // K1 [10,30) hip, K2 [20,34) non-hip, K3 starts after the window.
        let names = ["k_a", "k_b"];
        let ds = vec![
            disp(0, 1, 10, 30, [1, 1, 1], 0, Some(true)),
            disp(1, 1, 20, 34, [1, 1, 1], 0, Some(false)),
            disp(0, 1, 60, 70, [1, 1, 1], 0, Some(true)),
        ];
        let copies = vec![
            cp(0, 15, 0, 1),  // clipped to [10,15)
            cp(35, 50, 0, 1), // inside
            cp(45, 55, 1, 0), // clipped to [45,52)
            cp(20, 30, 7, 8), // other agents: ignored
            cp(5, 8, 0, 1),   // entirely before the ROI: dropped
        ];
        let t = trace_of(&names, ds, copies, Vec::new());
        let c = census(
            &t,
            &inv(&[]),
            &RoiSpec::Window {
                start_ns: 10,
                end_ns: 52,
            },
            None,
            key(),
            10,
        )
        .unwrap();
        assert_eq!(c.schema, CENSUS_SCHEMA);
        assert_eq!(c.agent, 1);
        assert_eq!((c.roi.start_ns, c.roi.end_ns, c.roi.wall_ns), (10, 52, 42));
        assert_eq!(c.roi.spec, "window:10:52");
        assert_eq!(c.kernels_in_roi, 2);
        assert_eq!(c.kernel_sum_ns, 20 + 14);
        assert_eq!(c.kernel_busy_ns, 24);
        assert_eq!(c.copy_busy_ns, 5 + 17);
        assert_eq!(c.copy_kernel_overlap_ns, 5);
        // covered: [10,34) + [35,52) = 41 of 42
        assert_eq!(c.idle_ns, 1);
        // exclusive: [10,20) K1 10, [20,30) 5/5, [30,34) K2 4
        let by_class_excl: u64 = c.by_dispatch.values().map(|b| b.exclusive_ns).sum();
        assert_eq!(by_class_excl, c.kernel_busy_ns);
        assert_eq!(c.by_dispatch["hip"].exclusive_ns, 15);
        assert_eq!(c.by_dispatch["non_hip"].exclusive_ns, 9);
        assert!(!c.by_dispatch.contains_key("hip_graph"));
        assert!(!c.by_dispatch.contains_key("unknown"));
        assert!(c
            .notes
            .iter()
            .any(|n| n.contains("non_hip") && n.contains("PM4")));
        // origin buckets partition the busy time and use only the three keys
        assert!(c
            .by_origin
            .keys()
            .all(|k| ["pm", "hipcc", "unknown"].contains(&k.as_str())));
        assert_eq!(
            c.by_origin.values().map(|b| b.exclusive_ns).sum::<u64>(),
            24
        );
        assert_eq!(c.by_origin.values().map(|b| b.calls).sum::<u64>(), 2);
        assert_eq!(c.top.len(), 2);
        assert_eq!(c.top[0].symbol, "k_a");
        assert!((c.top[0].pct_wall - 100.0 * 20.0 / 42.0).abs() < 1e-9);
    }

    #[test]
    fn census_window_end_extends_to_cover_included_dispatches() {
        let t = trace_of(
            &["k"],
            vec![disp(0, 1, 10, 40, [1, 1, 1], 0, Some(true))],
            Vec::new(),
            Vec::new(),
        );
        let c = census(
            &t,
            &inv(&[]),
            &RoiSpec::Window {
                start_ns: 5,
                end_ns: 20,
            },
            None,
            key(),
            5,
        )
        .unwrap();
        assert_eq!((c.roi.start_ns, c.roi.end_ns, c.roi.wall_ns), (5, 40, 35));
        assert_eq!(c.idle_ns, 5);
        assert!(c.notes.iter().any(|n| n.contains("extended")));
    }

    #[test]
    fn census_full_and_graph_dispatch_class() {
        let ds = vec![
            disp(0, 1, 100, 110, [1, 1, 1], 7, Some(false)),
            disp(0, 1, 105, 120, [1, 1, 1], 0, None),
        ];
        let t = trace_of(&["k"], ds, Vec::new(), Vec::new());
        let c = census(&t, &inv(&[]), &RoiSpec::Full, Some(1), key(), 5).unwrap();
        assert_eq!((c.roi.start_ns, c.roi.end_ns), (100, 120));
        assert_eq!(c.by_dispatch["hip_graph"].calls, 1);
        assert_eq!(c.by_dispatch["unknown"].calls, 1);
        assert_eq!(c.kernel_busy_ns, 20);
        assert_eq!(c.kernel_sum_ns, 25);
        assert_eq!(c.idle_ns, 0);
        // one symbol seen via two dispatch classes stays two rows
        assert_eq!(c.top.len(), 2);
        assert!(c.by_origin.values().all(|b| b.kernels == 1));
    }

    #[test]
    fn census_marker_roi_and_errors() {
        let ds = vec![
            disp(0, 1, 10, 20, [1, 1, 1], 0, Some(true)),
            disp(0, 1, 30, 40, [1, 1, 1], 0, Some(true)),
        ];
        let markers = vec![
            Marker {
                name: "pass".into(),
                start_ns: 25,
                end_ns: 45,
            },
            Marker {
                name: "twice".into(),
                start_ns: 0,
                end_ns: 5,
            },
            Marker {
                name: "twice".into(),
                start_ns: 6,
                end_ns: 9,
            },
        ];
        let t = trace_of(&["k"], ds, Vec::new(), markers);
        let c = census(
            &t,
            &inv(&[]),
            &RoiSpec::Marker {
                name: "pass".into(),
            },
            None,
            key(),
            5,
        )
        .unwrap();
        assert_eq!(c.kernels_in_roi, 1);
        assert_eq!((c.roi.start_ns, c.roi.end_ns), (25, 45));
        assert_eq!(c.idle_ns, 10);
        let missing = census(
            &t,
            &inv(&[]),
            &RoiSpec::Marker {
                name: "nope".into(),
            },
            None,
            key(),
            5,
        );
        assert!(missing.unwrap_err().contains("no marker named"));
        let dup = census(
            &t,
            &inv(&[]),
            &RoiSpec::Marker {
                name: "twice".into(),
            },
            None,
            key(),
            5,
        );
        assert!(dup.unwrap_err().contains("ambiguous"));
    }

    #[test]
    fn census_agent_resolution() {
        let mut t = trace_of(
            &["k"],
            vec![
                disp(0, 1, 0, 10, [1, 1, 1], 0, Some(true)),
                disp(0, 2, 0, 10, [1, 1, 1], 0, Some(true)),
            ],
            Vec::new(),
            Vec::new(),
        );
        t.agents.push(Agent {
            id: 2,
            kind: "gpu".into(),
            name: "gfx1100".into(),
            ordinal: Some(1),
        });
        let err = census(&t, &inv(&[]), &RoiSpec::Full, None, key(), 5).unwrap_err();
        assert!(err.contains("--agent"), "{err}");
        let c = census(&t, &inv(&[]), &RoiSpec::Full, Some(2), key(), 5).unwrap();
        assert_eq!(c.agent, 2);
        assert!(
            c.notes.iter().any(|n| n.contains("gfx1100")),
            "arch mismatch is noted"
        );
        assert!(census(&t, &inv(&[]), &RoiSpec::Full, Some(9), key(), 5).is_err());
    }

    /// A synthetic two-pass prefill trace following analyze.py's rule.
    fn prefill_trace() -> Trace {
        let names = ["embedding_q8_batched", "fill_buf", "layer_k", "decode_k"];
        let big = [4096, 2, 1];
        let small = [1, 1, 1];
        let ds = vec![
            disp(0, 1, 10 * U, 11 * U, big, 0, Some(true)), // pass 1 chunk 1
            disp(0, 1, 30 * U, 31 * U, small, 0, Some(true)), // small anchor, not a chunk start
            disp(0, 1, 40 * U, 41 * U, big, 0, Some(true)), // pass 1 chunk 2
            disp(2, 1, 41 * U, 45 * U, small, 0, Some(true)),
            disp(1, 1, 90 * U, 91 * U, small, 0, Some(true)), // before the 5 ms prelude window
            disp(1, 1, 98 * U, 99 * U, small, 0, Some(true)), // prelude of pass 2
            disp(0, 1, 100 * U, 101 * U, big, 0, Some(true)), // pass 2 chunk 1
            disp(2, 1, 101 * U, 110 * U, small, 0, Some(true)),
            disp(0, 1, 120 * U, 121 * U, big, 0, Some(true)), // pass 2 chunk 2
            disp(2, 1, 121 * U, 130 * U, small, 0, Some(true)),
            disp(0, 1, 150 * U, 151 * U, small, 0, Some(true)), // decode embedding ends the window
            disp(3, 1, 151 * U, 152 * U, small, 0, Some(true)),
        ];
        let copies = vec![
            cp(97 * U, 99 * U, 0, 1),   // clipped to [98,99)
            cp(129 * U, 140 * U, 0, 1), // clipped to [129,130)
            cp(140 * U, 141 * U, 0, 1), // outside
            cp(100 * U, 105 * U, 3, 4), // other agents
        ];
        trace_of(&names, ds, copies, Vec::new())
    }

    #[test]
    fn prefill_pass_follows_analyze_rule() {
        let t = prefill_trace();
        let spec = RoiSpec::PrefillPass {
            anchor_prefix: "embedding_q8_batched".into(),
            chunks_per_pass: 2,
        };
        let c = census(&t, &inv(&[]), &spec, None, key(), 10).unwrap();
        // start = the prelude fill at 98 ms (90 ms is outside [95,100]); end = 130 ms
        assert_eq!(c.roi.start_ns, 98 * U);
        assert_eq!(c.roi.end_ns, 130 * U);
        assert_eq!(c.roi.wall_ns, 32 * U);
        assert_eq!(c.roi.spec, "prefill-pass:embedding_q8_batched:2");
        // fill + chunk1 + layer + chunk2 + layer
        assert_eq!(c.kernels_in_roi, 5);
        assert_eq!(c.kernel_sum_ns, U + U + 9 * U + U + 9 * U);
        assert_eq!(c.kernel_busy_ns, c.kernel_sum_ns);
        assert_eq!(c.copy_busy_ns, 2 * U);
        // [98,99) lies inside the fill kernel, [129,130) inside the last layer kernel
        assert_eq!(c.copy_kernel_overlap_ns, 2 * U);
        // kernel union = [98,99) + [100,110) + [120,130); the copies add no uncovered time
        assert_eq!(c.kernel_busy_ns, 21 * U);
        assert_eq!(c.idle_ns, 32 * U - 21 * U);
        // chunks_per_pass = 4 groups all four chunk starts into pass 1 only -> pass 2 absent
        let spec4 = RoiSpec::PrefillPass {
            anchor_prefix: "embedding_q8_batched".into(),
            chunks_per_pass: 4,
        };
        let c4 = census(&t, &inv(&[]), &spec4, None, key(), 10).unwrap();
        assert_eq!(c4.roi.start_ns, 10 * U);
        // chunks_per_pass = 1: the last chunk alone (120 ms), prelude = nothing within 5 ms before except itself
        let spec1 = RoiSpec::PrefillPass {
            anchor_prefix: "embedding_q8_batched".into(),
            chunks_per_pass: 1,
        };
        let c1 = census(&t, &inv(&[]), &spec1, None, key(), 10).unwrap();
        assert_eq!(c1.roi.start_ns, 120 * U);
        assert_eq!(c1.roi.end_ns, 130 * U);
        assert_eq!(c1.kernels_in_roi, 2);
        // errors
        let none = RoiSpec::PrefillPass {
            anchor_prefix: "nothing_like_this".into(),
            chunks_per_pass: 2,
        };
        assert!(census(&t, &inv(&[]), &none, None, key(), 10).is_err());
        let zero = RoiSpec::PrefillPass {
            anchor_prefix: "embedding_q8_batched".into(),
            chunks_per_pass: 0,
        };
        assert!(census(&t, &inv(&[]), &zero, None, key(), 10).is_err());
    }

    #[test]
    fn prefill_pass_without_next_anchor_runs_to_trace_end() {
        let names = ["embedding_q8_batched", "layer_k"];
        let ds = vec![
            disp(0, 1, 10 * U, 11 * U, [4096, 2, 1], 0, Some(true)),
            disp(1, 1, 11 * U, 20 * U, [1, 1, 1], 0, Some(true)),
            disp(1, 1, 20 * U, 25 * U, [1, 1, 1], 0, Some(true)),
        ];
        let t = trace_of(&names, ds, Vec::new(), Vec::new());
        let spec = RoiSpec::PrefillPass {
            anchor_prefix: "embedding".into(),
            chunks_per_pass: 1,
        };
        let c = census(&t, &inv(&[]), &spec, None, key(), 3).unwrap();
        assert_eq!((c.roi.start_ns, c.roi.end_ns), (10 * U, 25 * U));
        assert_eq!(c.kernels_in_roi, 3);
    }

    #[test]
    fn contract_index_keys_are_id_at_version() {
        let idx = contract_index();
        for v in idx.values() {
            assert!(v.contains("@v"), "{v}");
        }
        for abi in hipfire_kernel_abi::families::all() {
            for e in abi.exports {
                assert!(idx.contains_key(e.symbol));
            }
        }
    }

    #[test]
    fn contract_export_gets_by_contract_bucket() {
        let Some(abi) = hipfire_kernel_abi::families::all().next() else {
            return;
        };
        let Some(e) = abi.exports.first() else {
            return;
        };
        let t = trace_of(
            &[e.symbol, "other"],
            vec![
                disp(0, 1, 0, 10, [1, 1, 1], 0, Some(true)),
                disp(1, 1, 10, 15, [1, 1, 1], 0, Some(true)),
            ],
            Vec::new(),
            Vec::new(),
        );
        let c = census(&t, &inv(&[]), &RoiSpec::Full, None, key(), 5).unwrap();
        let id = contract_index()[e.symbol].clone();
        assert_eq!(c.by_contract[&id].calls, 1);
        assert_eq!(c.by_contract[&id].sum_ns, 10);
        assert_eq!(
            c.by_contract.len(),
            1,
            "unseen families and non-contract kernels add no bucket"
        );
        assert_eq!(
            c.top
                .iter()
                .find(|r| r.symbol == e.symbol)
                .unwrap()
                .abi
                .as_deref(),
            Some(id.as_str())
        );
        assert!(c
            .top
            .iter()
            .find(|r| r.symbol == "other")
            .unwrap()
            .abi
            .is_none());
    }

    #[test]
    fn roi_spec_parse_label_roundtrip() {
        for s in [
            "full",
            "window:100:200",
            "marker:prefill pass",
            "prefill-pass:embedding_q8_batched:4",
        ] {
            assert_eq!(RoiSpec::parse(s).unwrap().label(), s);
        }
        assert!(RoiSpec::parse("window:1").is_err());
        assert!(RoiSpec::parse("prefill-pass:x:y").is_err());
        assert!(RoiSpec::parse("bogus").is_err());
        assert!(RoiSpec::parse("marker:").is_err());
    }

    fn fake_census(
        model: &str,
        arch: &str,
        phase: PhaseV1,
        pm_excl: u64,
        unknown_excl: u64,
    ) -> CensusV1 {
        let busy = 1000;
        let mut by_origin = BTreeMap::new();
        by_origin.insert(
            "pm".to_string(),
            Bucket {
                calls: 1,
                kernels: 1,
                sum_ns: pm_excl,
                exclusive_ns: pm_excl,
            },
        );
        if unknown_excl > 0 {
            by_origin.insert(
                "unknown".to_string(),
                Bucket {
                    calls: 1,
                    kernels: 1,
                    sum_ns: unknown_excl,
                    exclusive_ns: unknown_excl,
                },
            );
        }
        CensusV1 {
            schema: CENSUS_SCHEMA.into(),
            cell: CellKey {
                model: model.into(),
                arch: arch.into(),
                phase,
            },
            capture: TraceSource {
                path: "p".into(),
                sha256: "0".repeat(64),
                format: "rocprofv3-json".into(),
            },
            inventory_commit: None,
            agent: 1,
            roi: RoiV1 {
                start_ns: 0,
                end_ns: 2000,
                wall_ns: 2000,
                spec: "full".into(),
            },
            kernels_in_roi: 2,
            kernel_sum_ns: busy,
            kernel_busy_ns: busy,
            copy_busy_ns: 0,
            copy_kernel_overlap_ns: 0,
            idle_ns: 1000,
            by_origin,
            by_dispatch: BTreeMap::new(),
            by_contract: BTreeMap::new(),
            top: Vec::new(),
            notes: Vec::new(),
            measurement: None,
        }
    }

    #[test]
    fn matrix_has_27_cells_all_unmeasured_by_default() {
        let m = matrix(&[], &[]);
        assert_eq!(m.schema, MATRIX_SCHEMA);
        assert_eq!(m.cells.len(), 27);
        for c in &m.cells {
            match &c.status {
                CellStatus::Unmeasured { reason } => {
                    assert_eq!(reason, "no object-joined profile supplied")
                }
                other => panic!("expected Unmeasured, got {other:?}"),
            }
            assert!(c.pm_share_of_kernel_busy.is_none() && c.pm_share_of_wall.is_none());
        }
        let labels: BTreeSet<String> = m.cells.iter().map(|c| key_label(&c.key)).collect();
        assert_eq!(labels.len(), 27);
        assert!(labels.contains("dense-27b:gfx1201:prefill"));
        assert!(labels.contains("flash-next:gfx1100:spec"));
        assert!(labels.contains("a3b:gfx1151:ar"));
        // serializes with the status tag
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"status\":\"unmeasured\""));
    }

    #[test]
    fn matrix_measured_refused_and_conflicts() {
        let measured = vec![
            (
                "a.json".to_string(),
                "sha-a".to_string(),
                fake_census("a3b", "gfx1201", PhaseV1::Ar, 250, 0),
            ),
            (
                "b.json".to_string(),
                "sha-b".to_string(),
                fake_census("dense-27b", "gfx1100", PhaseV1::Spec, 100, 200),
            ),
            (
                "c1.json".to_string(),
                "sha-c1".to_string(),
                fake_census("a3b", "gfx1151", PhaseV1::Prefill, 1, 0),
            ),
            (
                "c2.json".to_string(),
                "sha-c2".to_string(),
                fake_census("a3b", "gfx1151", PhaseV1::Prefill, 2, 0),
            ),
            (
                "d.json".to_string(),
                "sha-d".to_string(),
                fake_census("flash-next", "gfx1100", PhaseV1::Ar, 3, 0),
            ),
        ];
        let refused = vec![
            (
                CellKey {
                    model: "flash-next".into(),
                    arch: "gfx1151".into(),
                    phase: PhaseV1::Spec,
                },
                "no spec path".to_string(),
            ),
            (
                CellKey {
                    model: "flash-next".into(),
                    arch: "gfx1100".into(),
                    phase: PhaseV1::Ar,
                },
                "unsupported".to_string(),
            ),
        ];
        let m = matrix(&measured, &refused);
        assert_eq!(m.cells.len(), 27);
        let find = |model: &str, arch: &str, phase: PhaseV1| {
            m.cells
                .iter()
                .find(|c| {
                    same_key(
                        &c.key,
                        &CellKey {
                            model: model.into(),
                            arch: arch.into(),
                            phase,
                        },
                    )
                })
                .unwrap()
        };
        let a = find("a3b", "gfx1201", PhaseV1::Ar);
        assert!(
            matches!(&a.status, CellStatus::Measured { census, sha256 } if census == "a.json" && sha256 == "sha-a")
        );
        assert_eq!(a.pm_share_of_kernel_busy, Some(0.25));
        assert_eq!(a.pm_share_of_wall, Some(0.125));
        assert!(a.notes.is_empty());
        let b = find("dense-27b", "gfx1100", PhaseV1::Spec);
        assert!(matches!(b.status, CellStatus::Measured { .. }));
        assert_eq!(b.pm_share_of_kernel_busy, Some(0.1));
        assert!(
            b.notes.iter().any(|n| n.contains("unknown share 20.0%")),
            "{:?}",
            b.notes
        );
        let r = find("flash-next", "gfx1151", PhaseV1::Spec);
        assert!(matches!(&r.status, CellStatus::Refused { reason } if reason == "no spec path"));
        assert!(r.pm_share_of_kernel_busy.is_none());
        let dup = find("a3b", "gfx1151", PhaseV1::Prefill);
        assert!(
            matches!(&dup.status, CellStatus::Refused { reason } if reason.contains("ambiguous") && reason.contains("c1.json") && reason.contains("c2.json"))
        );
        let both = find("flash-next", "gfx1100", PhaseV1::Ar);
        assert!(
            matches!(&both.status, CellStatus::Refused { reason } if reason.contains("unsupported") && reason.contains("d.json"))
        );
        let measured_cells = m
            .cells
            .iter()
            .filter(|c| matches!(c.status, CellStatus::Measured { .. }))
            .count();
        assert_eq!(measured_cells, 2);
        let unmeasured = m
            .cells
            .iter()
            .filter(|c| matches!(c.status, CellStatus::Unmeasured { .. }))
            .count();
        assert_eq!(unmeasured, 27 - 2 - 3);
    }

    #[test]
    fn census_round_trips_through_json() {
        let c = fake_census("a3b", "gfx1201", PhaseV1::Ar, 250, 0);
        let j = serde_json::to_vec(&c).unwrap();
        let back: CensusV1 = serde_json::from_slice(&j).unwrap();
        assert_eq!(back.by_origin["pm"], c.by_origin["pm"]);
        assert_eq!(phase_str(&back.cell.phase), "ar");
    }

    // ── historical regression: fn-prefill-profile-002 (REPORT.md §1, §2) ──

    const PROFILE_DIR: &str = "/home/kaden/qcal/release-0.4.3/fn-prefill-profile-002";

    fn within(got_ns: u64, want_ms: f64, what: &str) {
        let got_ms = got_ns as f64 / 1e6;
        let tol = want_ms * 0.0015;
        assert!(
            (got_ms - want_ms).abs() <= tol,
            "{what}: got {got_ms:.3} ms, want {want_ms:.3} ms (+-0.15% = {tol:.3})"
        );
    }

    fn historical(card: &str, arch: &str, cpp: usize) -> Option<CensusV1> {
        let path = Path::new(PROFILE_DIR)
            .join(card)
            .join("prof/trace_results.json");
        if !path.exists() {
            return None;
        }
        let trace = rocprof::load(&path).expect("load historical trace");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = inventory::build(&repo, None).expect("build inventory");
        let spec = RoiSpec::PrefillPass {
            anchor_prefix: "embedding_q8_batched".into(),
            chunks_per_pass: cpp,
        };
        let key = CellKey {
            model: "flash-next".into(),
            arch: arch.into(),
            phase: PhaseV1::Prefill,
        };
        Some(census(&trace, &inventory, &spec, None, key, 15).expect("census"))
    }

    #[test]
    fn historical_r9700_pp16384_prefill_pass() {
        let Some(c) = historical("r9700", "gfx1201", 4) else {
            return;
        };
        // REPORT.md §1 and r9700/analysis.txt
        within(c.roi.wall_ns, 3352.4, "wall");
        within(c.kernel_sum_ns, 3245.6, "kernel sum");
        within(c.kernel_busy_ns, 3243.7, "kernel busy");
        within(c.copy_busy_ns, 3248.0, "copy busy");
        within(c.copy_kernel_overlap_ns, 3147.0, "kernel/copy overlap");
        assert_eq!(c.kernels_in_roi, 9208);
        assert!(
            c.idle_ns.abs_diff(7_611_714) <= 50_000,
            "idle {} ns vs 7.6 ms",
            c.idle_ns
        );
        assert_eq!(
            c.by_origin.values().map(|b| b.calls).sum::<u64>(),
            c.kernels_in_roi
        );
        assert_eq!(
            c.by_origin.values().map(|b| b.exclusive_ns).sum::<u64>(),
            c.kernel_busy_ns
        );
    }

    #[test]
    fn historical_halo_pp16384_prefill_pass() {
        let Some(c) = historical("halo", "gfx1151", 2) else {
            return;
        };
        // REPORT.md §2 and halo/analysis.txt: window == `analyze.py --chunks-per-pass 2`
        within(c.roi.wall_ns, 6631.1, "wall");
        within(c.kernel_sum_ns, 6517.7, "kernel sum");
        within(c.kernel_busy_ns, 6517.7, "kernel busy");
        within(c.copy_busy_ns, 2.506, "copy busy");
        within(c.idle_ns, 110.845, "idle");
        assert_eq!(c.copy_kernel_overlap_ns, 0);
        assert_eq!(c.kernels_in_roi, 3368);
    }
}
