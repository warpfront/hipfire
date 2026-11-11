// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node ingestion: one rocprofv3 kernel trace + the model's config and `.hfq`
//! tensor directory -> a node corpus (`pm-census.nodes.v1`) with decoded
//! shapes, MACs/bytes, row domains, roofline floors and a ranked queue.
//! Spec: `/home/kaden/qcal/release-0.4.3/pmrip/S1-SPEC.md`.
//!
//! Nothing here names a model. Everything model-specific comes from
//! [`hipfire_kernel_abi`] contracts and their [`WorkV1`] sidecars (what a launch
//! computes and how the params a trace cannot carry bind), the config integers
//! the sidecars name by HF-synonym keys, and the weight tensors matched by
//! codec and shape ([`ModelDesc::weight_groups`]).
//!
//! * [`build_corpus`] — the binder: passes, nodes, interval inversion of the
//!   launch grid, [`BindSource`] resolution, leaves, debt rows, roofline.
//! * [`roofline`] — per-token MAC/FLOP by op class and weight group, peak
//!   ceilings, `--what-if` pipe re-charging.
//! * [`queue`] / [`queue_markdown`] — leaves ranked by
//!   `share x (1 - floor / measured)` across corpora.
//! * CLI: [`cli_nodes`] (`pm-census extract nodes`), [`cli_roofline`],
//!   [`cli_queue`].

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use hipfire_kernel_abi::work::{
    op_class, BindSource, Pipe, RowDomainV1, Stacking, WeightAxis, WorkSpecV1, WorkV1, STACK_PARAM,
};
use hipfire_kernel_abi::{families, Arch, ExportV1, Expr, KernelAbiV1};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::extract::const_value;
use crate::hw::{self, Peaks};
use crate::model::{ModelDesc, WeightGroup};
use crate::rocprof;

/// Corpus schema tag.
pub const SCHEMA: &str = "pm-census.nodes.v1";
/// Queue document schema tag.
pub const QUEUE_SCHEMA: &str = "pm-census.node-queue.v1";

/// A contract, its export and (when one is attached) its work spec.
pub type Found = (
    &'static KernelAbiV1,
    &'static ExportV1,
    Option<&'static WorkSpecV1>,
);

/// Contract lookup by `(arch, symbol)`. Injectable so tests bind against
/// small synthetic contracts instead of the real family tables.
pub type Lookup<'a> = dyn Fn(&str, &str) -> Option<Found> + 'a;

/// Production lookup: [`families::find_export`] + [`families::work_for`].
pub fn families_lookup(arch: &str, symbol: &str) -> Option<Found> {
    let arch = Arch::parse(strip_features(arch))?;
    let (abi, export) = families::find_export(arch, symbol)?;
    Some((abi, export, families::work_for(abi.id)))
}

fn strip_features(arch: &str) -> &str {
    arch.split(':').next().unwrap_or(arch)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// intervals

/// Inclusive integer interval a param is known to lie in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Iv {
    pub lo: u64,
    pub hi: u64,
}

impl Iv {
    pub const ALL: Iv = Iv {
        lo: 0,
        hi: u64::MAX,
    };

    pub fn point(v: u64) -> Iv {
        Iv { lo: v, hi: v }
    }

    pub fn is_point(self) -> bool {
        self.lo == self.hi
    }

    pub fn contains(self, v: u64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn intersect(self, o: Iv) -> Option<Iv> {
        let lo = self.lo.max(o.lo);
        let hi = self.hi.min(o.hi);
        (lo <= hi).then_some(Iv { lo, hi })
    }
}

fn narrow(out: &mut BTreeMap<String, Iv>, name: &str, t: Iv) -> bool {
    let cur = out.get(name).copied().unwrap_or(Iv::ALL);
    match cur.intersect(t) {
        Some(i) => {
            out.insert(name.to_string(), i);
            true
        }
        None => false,
    }
}

/// Solve `e` lying in `t` for the launch params `e` names, as intervals.
///
/// Exact inversion where the expression is injective in its single unknown
/// (`Param`, `Mul`/`Add`/`Sub` with constants, `DivExact`); interval inversion
/// through `DivCeil` (`ceil(a/d) = g` <=> `a in [(g-1)d+1, g d]`) and `Div`
/// (`floor(a/d) = g` <=> `a in [g d, g d + d - 1]`). An expression with two
/// unknowns constrains nothing and is accepted. Returns `false` when the
/// target is unreachable (the grid contradicts the contract). Intersects into
/// `out`.
pub fn invert_interval(
    e: &Expr,
    t: Iv,
    abi: &KernelAbiV1,
    export: &ExportV1,
    out: &mut BTreeMap<String, Iv>,
) -> bool {
    if let Some(c) = const_value(e, abi, export) {
        return t.contains(c);
    }
    match e {
        Expr::Const(c) => t.contains(*c),
        Expr::Param(n) => narrow(out, n, t),
        Expr::Mul(xs) => {
            let mut k = 1u64;
            let mut unknown: Option<&Expr> = None;
            let mut many = false;
            for x in xs.iter() {
                match const_value(x, abi, export) {
                    Some(c) => match k.checked_mul(c) {
                        Some(v) => k = v,
                        None => return false,
                    },
                    None => {
                        if unknown.replace(x).is_some() {
                            many = true;
                        }
                    }
                }
            }
            if many {
                return true;
            }
            let Some(x) = unknown else {
                return t.contains(k);
            };
            if k == 0 {
                return t.lo == 0;
            }
            let lo = t.lo.div_ceil(k);
            let hi = t.hi / k;
            lo <= hi && invert_interval(x, Iv { lo, hi }, abi, export, out)
        }
        Expr::Add(xs) => {
            let mut sum = 0u64;
            let mut unknown: Option<&Expr> = None;
            let mut many = false;
            for x in xs.iter() {
                match const_value(x, abi, export) {
                    Some(c) => match sum.checked_add(c) {
                        Some(v) => sum = v,
                        None => return false,
                    },
                    None => {
                        if unknown.replace(x).is_some() {
                            many = true;
                        }
                    }
                }
            }
            if many {
                return true;
            }
            let Some(x) = unknown else {
                return t.contains(sum);
            };
            if t.hi < sum {
                return false;
            }
            let iv = Iv {
                lo: t.lo.saturating_sub(sum),
                hi: t.hi - sum,
            };
            invert_interval(x, iv, abi, export, out)
        }
        Expr::Sub(a, b) => {
            if let Some(bc) = const_value(b, abi, export) {
                let iv = Iv {
                    lo: t.lo.saturating_add(bc),
                    hi: t.hi.saturating_add(bc),
                };
                invert_interval(a, iv, abi, export, out)
            } else if let Some(ac) = const_value(a, abi, export) {
                if ac < t.lo {
                    return false;
                }
                let iv = Iv {
                    lo: ac.saturating_sub(t.hi),
                    hi: ac - t.lo,
                };
                invert_interval(b, iv, abi, export, out)
            } else {
                true
            }
        }
        Expr::DivExact(a, d) => {
            if *d == 0 {
                return false;
            }
            let iv = Iv {
                lo: t.lo.saturating_mul(*d),
                hi: t.hi.saturating_mul(*d),
            };
            invert_interval(a, iv, abi, export, out)
        }
        Expr::DivCeil(a, d) => {
            if *d == 0 {
                return false;
            }
            let lo = if t.lo == 0 {
                0
            } else {
                (t.lo - 1).saturating_mul(*d).saturating_add(1)
            };
            let hi = t.hi.saturating_mul(*d);
            lo <= hi && invert_interval(a, Iv { lo, hi }, abi, export, out)
        }
        Expr::Div(a, d) => {
            if *d == 0 {
                return false;
            }
            let iv = Iv {
                lo: t.lo.saturating_mul(*d),
                hi: t.hi.saturating_mul(*d).saturating_add(*d - 1),
            };
            invert_interval(a, iv, abi, export, out)
        }
    }
}

/// Param intervals the workgroup `grid` implies through the export's grid
/// expressions; `None` when the grid contradicts the contract.
pub fn grid_intervals(
    abi: &KernelAbiV1,
    export: &ExportV1,
    grid: [u32; 3],
) -> Option<BTreeMap<String, Iv>> {
    let mut out = BTreeMap::new();
    for axis in 0..3 {
        if !invert_interval(
            &export.grid[axis],
            Iv::point(u64::from(grid[axis])),
            abi,
            export,
            &mut out,
        ) {
            return None;
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// expression evaluation over named values

#[derive(Debug)]
enum EvErr {
    Unbound(String),
    Math(String),
}

fn eval_vals(e: &Expr, vals: &BTreeMap<String, u64>, export: &ExportV1) -> Result<u64, EvErr> {
    let bin = |a: u64, b: u64, op: &str, r: Option<u64>| {
        r.ok_or_else(|| EvErr::Math(format!("overflow or underflow: {a} {op} {b}")))
    };
    match e {
        Expr::Const(v) => Ok(*v),
        Expr::Param(n) => {
            if let Some(v) = vals.get(*n) {
                return Ok(*v);
            }
            export
                .consts
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, v)| *v)
                .ok_or_else(|| EvErr::Unbound((*n).to_string()))
        }
        Expr::Add(xs) => {
            let mut acc = 0u64;
            for x in xs.iter() {
                let v = eval_vals(x, vals, export)?;
                acc = bin(acc, v, "+", acc.checked_add(v))?;
            }
            Ok(acc)
        }
        Expr::Mul(xs) => {
            let mut acc = 1u64;
            for x in xs.iter() {
                let v = eval_vals(x, vals, export)?;
                acc = bin(acc, v, "*", acc.checked_mul(v))?;
            }
            Ok(acc)
        }
        Expr::Sub(a, b) => {
            let (a, b) = (eval_vals(a, vals, export)?, eval_vals(b, vals, export)?);
            bin(a, b, "-", a.checked_sub(b))
        }
        Expr::DivExact(a, d) => {
            let a = eval_vals(a, vals, export)?;
            if *d == 0 || a % d != 0 {
                return Err(EvErr::Math(format!("inexact division {a} / {d}")));
            }
            Ok(a / d)
        }
        Expr::DivCeil(a, d) => {
            let a = eval_vals(a, vals, export)?;
            if *d == 0 {
                return Err(EvErr::Math(format!("division by zero: ceil({a} / 0)")));
            }
            Ok(a.div_ceil(*d))
        }
        Expr::Div(a, d) => {
            let a = eval_vals(a, vals, export)?;
            if *d == 0 {
                return Err(EvErr::Math(format!("division by zero: {a} / 0")));
            }
            Ok(a / d)
        }
    }
}

// ---------------------------------------------------------------------------
// records

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRef {
    pub path: String,
    pub sha256: String,
    pub format: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfqRef {
    pub path: String,
    pub index_sha256: String,
    pub arch_id: u32,
    pub tensors: usize,
}

/// A weight group a leaf binds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightRef {
    pub members: Vec<String>,
    pub rows: u64,
    pub cols: u64,
    pub codec: String,
    pub stacked: bool,
    pub fused: bool,
    pub layers: u64,
    pub count_per_layer: u64,
}

impl From<&WeightGroup> for WeightRef {
    fn from(g: &WeightGroup) -> Self {
        WeightRef {
            members: g.members.clone(),
            rows: g.rows,
            cols: g.cols,
            codec: g.codec.to_string(),
            stacked: g.stacked,
            fused: g.fused,
            layers: g.layers,
            count_per_layer: g.count_per_layer,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowClass {
    pub min_rows: u64,
    pub max_rows: Option<u64>,
    pub predicate: String,
    pub kill_env: Option<String>,
    /// The bound activation row count (the work spec's `rows` param), if known.
    pub rows: Option<u64>,
    /// Whether `rows` lies in the admitted domain (`None` when `rows` is unknown).
    pub admitted: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Floors {
    pub mem_ns: Option<f64>,
    pub arith_ns: Option<f64>,
    pub floor_ns: Option<f64>,
}

/// One bound node: a `(symbol, grid, block, lds)` of the trace, or one weight
/// split of it. Quantities are per pass (means over the used passes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Leaf {
    pub symbol: String,
    pub abi_id: String,
    pub abi_version: u32,
    pub abi_digest: String,
    pub op_class: String,
    pub codec: String,
    pub pipe: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub lds: u32,
    /// Params bound exactly.
    pub params: BTreeMap<String, u64>,
    /// Params only known to an interval (and not bound).
    pub intervals: BTreeMap<String, Iv>,
    /// Names the MAC/FLOP/byte expressions need that stayed unbound.
    pub unbound: Vec<String>,
    pub weight: Vec<WeightRef>,
    pub calls: f64,
    pub launches_per_chunk: f64,
    pub time_ns: f64,
    /// `time_ns` / pass wall.
    pub share: f64,
    /// How `time_ns` was split when one trace node produced several leaves.
    pub time_basis: String,
    pub macs: Option<f64>,
    pub flops: Option<f64>,
    pub bytes: Option<f64>,
    pub row_class: RowClass,
    pub floors: Floors,
    /// `floor_ns / time_ns`.
    pub efficiency: Option<f64>,
    /// `exact` | `interval_upper` (flops/bytes evaluated at a grid interval's
    /// upper bound because only that was unbound) | none (no floor).
    #[serde(default)]
    pub floor_basis: Option<String>,
    /// `ok` | `partial` | `inconsistent`.
    pub status: String,
    /// `ok` or the binding caveats (`inconsistent: ...`).
    pub binding: String,
    /// MACs/bytes were evaluated per call at each call's causal prefix.
    pub causal: bool,
    pub note: String,
}

/// A node that could not be bound; time and calls are kept.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Debt {
    pub symbol: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub lds: u32,
    pub calls: f64,
    pub time_ns: f64,
    pub share: f64,
    /// `no_contract` | `no_work_spec` | `grid_mismatch` | `block_mismatch`.
    pub status: String,
    pub abi_id: Option<String>,
    pub note: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClassTotal {
    pub leaves: u64,
    pub calls: f64,
    pub time_ns: f64,
    pub share: f64,
    pub macs: f64,
    pub flops: f64,
    pub bytes: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PassStat {
    pub index: usize,
    pub start_ns: u64,
    pub end_ns: u64,
    pub wall_ns: u64,
    pub busy_ns: u64,
    pub idle_ns: u64,
    pub launches: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Corpus {
    pub schema: String,
    pub trace: TraceRef,
    pub config_sha256: String,
    pub hfq: Option<HfqRef>,
    pub arch: String,
    pub tokens: u64,
    pub pass_mode: String,
    pub passes_found: usize,
    /// Indices into `passes` the quantities average over.
    pub passes_used: Vec<usize>,
    pub passes: Vec<PassStat>,
    pub chunk_rows: u64,
    pub chunks_per_pass: u64,
    /// Means over the used passes.
    pub wall_ns: f64,
    pub busy_ns: f64,
    pub idle_ns: f64,
    pub launches_per_pass: f64,
    /// Dispatches before the first pass anchor (or outside `--window`).
    pub outside_pass_dispatches: u64,
    /// `--bench-profile` residual recorded next to the trace idle.
    pub bench_profile: Option<Value>,
    pub leaves: Vec<Leaf>,
    pub debt: Vec<Debt>,
    pub totals_by_op_class: BTreeMap<String, ClassTotal>,
    pub roofline: Value,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassMode {
    Mean,
    Last,
}

impl PassMode {
    pub fn name(self) -> &'static str {
        match self {
            PassMode::Mean => "mean",
            PassMode::Last => "last",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NodesOpts {
    pub arch: String,
    pub tokens: u64,
    pub pass: PassMode,
    /// Trace-time window (ns); required when the trace has no embedding anchor.
    pub window: Option<(u64, u64)>,
    pub agent: Option<u64>,
}

// ---------------------------------------------------------------------------
// passes

type NodeKey = (String, [u32; 3], [u32; 3], u32);

struct Node {
    calls: u64,
    time_ns: u64,
    /// call count per chunk ordinal (over the used passes).
    ordinals: BTreeMap<u32, u64>,
}

struct Pass {
    start: u64,
    end: u64,
    anchors: Vec<u64>,
    /// indices into the sorted dispatch vector
    members: Vec<usize>,
}

fn union_len(mut iv: Vec<(u64, u64)>) -> u64 {
    iv.sort_unstable();
    let mut total = 0u64;
    let mut cur: Option<(u64, u64)> = None;
    for (s, e) in iv {
        match cur {
            Some((cs, ce)) if s <= ce => cur = Some((cs, ce.max(e))),
            Some((cs, ce)) => {
                total += ce - cs;
                cur = Some((s, e));
            }
            None => cur = Some((s, e)),
        }
    }
    if let Some((cs, ce)) = cur {
        total += ce - cs;
    }
    total
}

// ---------------------------------------------------------------------------
// binder

struct BindCtx<'a> {
    abi: &'static KernelAbiV1,
    export: &'static ExportV1,
    work: &'a WorkV1,
    model: &'a ModelDesc,
    groups: &'a [WeightGroup],
    chunk_rows: u64,
    tokens: u64,
    calls_bind: u64,
    calls_per_chunk: Option<u64>,
}

#[derive(Clone)]
struct St {
    vals: BTreeMap<String, u64>,
    ivs: BTreeMap<String, Iv>,
    groups: Vec<WeightGroup>,
    /// Share of the node's calls this leaf takes.
    frac: f64,
    notes: Vec<String>,
    stacked: Option<bool>,
}

impl St {
    fn interval(&self, name: &str) -> Iv {
        if let Some(v) = self.vals.get(name) {
            return Iv::point(*v);
        }
        self.ivs.get(name).copied().unwrap_or(Iv::ALL)
    }

    /// Bind `name = v` when the grid interval admits it; otherwise keep the
    /// interval and say so.
    fn bind(&mut self, name: &str, v: u64, src: &str) {
        let iv = self.interval(name);
        if iv.contains(v) {
            self.vals.insert(name.to_string(), v);
        } else {
            self.notes.push(format!(
                "inconsistent: {src} gives {name}={v} outside the grid interval [{}, {}]",
                iv.lo, iv.hi
            ));
        }
    }
}

fn mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Const(_) => false,
        Expr::Param(n) => *n == name,
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().any(|x| mentions(x, name)),
        Expr::Sub(a, b) => mentions(a, name) || mentions(b, name),
        Expr::DivExact(a, _) | Expr::DivCeil(a, _) | Expr::Div(a, _) => mentions(a, name),
    }
}

/// Whether the work spec's MAC/FLOP/byte expressions (or, when bytes default
/// to the operand extents, those) read the implicit `stack` param.
fn reads_stack(abi: &KernelAbiV1, work: &WorkV1) -> bool {
    let w = [work.macs, work.flops, work.bytes]
        .iter()
        .flatten()
        .any(|e| mentions(e, STACK_PARAM));
    w || (work.bytes.is_none()
        && abi
            .operands
            .iter()
            .filter_map(|o| o.extent.as_ref())
            .any(|e| mentions(e, STACK_PARAM)))
}

fn launches(g: &WeightGroup) -> u64 {
    g.count_per_layer.saturating_mul(g.layers)
}

fn extent(g: &WeightGroup, axis: WeightAxis) -> u64 {
    match axis {
        WeightAxis::Rows => g.rows,
        WeightAxis::Cols => g.cols,
    }
}

fn run_binds(ctx: &BindCtx, init: &St, prefix: u64) -> Vec<St> {
    let mut states = vec![init.clone()];
    for b in ctx.work.bind {
        let mut next = Vec::with_capacity(states.len());
        for mut st in states {
            match b.source {
                BindSource::Config(keys) => {
                    match ctx.model.config_int(keys) {
                        Some(v) => st.bind(b.param, v, "config"),
                        None => st
                            .notes
                            .push(format!("config: none of {keys:?} present for {}", b.param)),
                    }
                    next.push(st);
                }
                BindSource::ChunkRows => {
                    st.bind(b.param, ctx.chunk_rows, "chunk rows");
                    next.push(st);
                }
                BindSource::PassTokens => {
                    st.bind(b.param, ctx.tokens, "pass tokens");
                    next.push(st);
                }
                BindSource::CausalPrefix => {
                    st.vals.insert(b.param.to_string(), prefix);
                    next.push(st);
                }
                BindSource::RoutedRows { topk, dense } => {
                    let stacked = st.stacked == Some(true);
                    if stacked {
                        match ctx.model.config_int(topk) {
                            Some(k) => {
                                st.bind(b.param, ctx.chunk_rows.saturating_mul(k), "routed rows")
                            }
                            None => st
                                .notes
                                .push(format!("config: none of {topk:?} present (top-k)")),
                        }
                    } else {
                        let iv = st.interval(dense);
                        if iv == Iv::ALL {
                            st.notes
                                .push(format!("routed rows: dense param {dense} has no interval"));
                        } else {
                            let v = if iv.contains(ctx.chunk_rows) {
                                ctx.chunk_rows
                            } else {
                                iv.hi.min(ctx.chunk_rows)
                            };
                            st.bind(b.param, v, "routed rows (dense)");
                        }
                    }
                    next.push(st);
                }
                BindSource::RoutedTokens { dense } => {
                    if st.stacked == Some(true) {
                        st.bind(b.param, ctx.chunk_rows, "routed tokens (stack: chunk rows)");
                    } else {
                        let iv = st.interval(dense);
                        if iv == Iv::ALL {
                            st.notes.push(format!(
                                "routed tokens: dense param {dense} has no interval"
                            ));
                        } else {
                            let v = if iv.contains(ctx.chunk_rows) {
                                ctx.chunk_rows
                            } else {
                                iv.hi.min(ctx.chunk_rows)
                            };
                            st.bind(b.param, v, "routed tokens (dense)");
                        }
                    }
                    next.push(st);
                }
                BindSource::Derived(e) => {
                    match eval_vals(&e, &st.vals, ctx.export) {
                        Ok(v) => st.bind(b.param, v, "derived"),
                        Err(EvErr::Unbound(n)) => st
                            .notes
                            .push(format!("derived {}: {n} is unbound", b.param)),
                        Err(EvErr::Math(m)) => st.notes.push(format!("derived {}: {m}", b.param)),
                    }
                    next.push(st);
                }
                BindSource::Weight {
                    by,
                    by_axis,
                    codecs,
                    stacking,
                    fuse_siblings,
                } => {
                    next.extend(bind_weight(
                        ctx,
                        st,
                        b.param,
                        by,
                        by_axis,
                        codecs,
                        stacking,
                        fuse_siblings,
                    ));
                }
            }
        }
        states = next;
    }
    states
}

#[allow(clippy::too_many_arguments)]
fn bind_weight(
    ctx: &BindCtx,
    mut st: St,
    param: &str,
    by: &str,
    by_axis: WeightAxis,
    codecs: &[&str],
    stacking: Stacking,
    fuse_siblings: bool,
) -> Vec<St> {
    let by_iv = st.interval(by);
    let param_iv = st.interval(param);
    let other_axis = match by_axis {
        WeightAxis::Rows => WeightAxis::Cols,
        WeightAxis::Cols => WeightAxis::Rows,
    };
    // A single-weight launch cannot carry more routed slots than the chunk
    // has rows: a dense row interval above the chunk rows means a stack.
    let stack_pref: Option<bool> = ctx.work.bind.iter().find_map(|b| match b.source {
        BindSource::RoutedRows { dense, .. } => Some(st.interval(dense).lo > ctx.chunk_rows),
        _ => None,
    });
    let cands: Vec<&WeightGroup> = ctx
        .groups
        .iter()
        .filter(|g| codecs.is_empty() || codecs.contains(&g.codec))
        .filter(|g| match stacking {
            Stacking::Single => !g.stacked,
            Stacking::Stack => g.stacked,
            Stacking::Either => stack_pref.map_or(true, |want| g.stacked == want),
        })
        .filter(|g| fuse_siblings || !g.fused)
        .filter(|g| by_iv.contains(extent(g, by_axis)))
        .filter(|g| param == by || param_iv.contains(extent(g, other_axis)))
        .collect();
    if cands.is_empty() {
        st.notes.push(format!(
            "no weight tensor with {by_axis:?} in [{}, {}] (codecs {codecs:?})",
            by_iv.lo, by_iv.hi
        ));
        return vec![st];
    }

    // Groups with the same members and shape that differ only in codec/layers
    // (mixed-codec layers) are one weight, not alternative sets: merge them.
    type UnitKey<'a> = (&'a [String], u64, u64, bool, bool);
    let mut units: Vec<(UnitKey, Vec<&WeightGroup>)> = Vec::new();
    for g in &cands {
        let key: UnitKey = (&g.members, g.rows, g.cols, g.stacked, g.fused);
        match units.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(g),
            None => units.push((key, vec![g])),
        }
    }
    let unit_launches = |u: &Vec<&WeightGroup>| -> u64 { u.iter().map(|g| launches(g)).sum() };

    // A set is consistent when its launches match the node's calls per chunk
    // (per-chunk ops) or per pass (per-pass ops such as a last-row head).
    let total_all: u64 = units.iter().map(|(_, u)| unit_launches(u)).sum();
    let mut chosen: Vec<usize> = (0..units.len()).collect();
    let n = units.len();
    let fits_count = |s: u64| s == ctx.calls_bind || ctx.calls_per_chunk == Some(s);
    let mut fit_count: Option<usize> = None;
    let mut selected = false;
    if n <= 20 {
        let mut fits: Vec<u32> = Vec::new();
        for mask in 1u32..(1u32 << n) {
            let s: u64 = (0..n)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| unit_launches(&units[i].1))
                .sum();
            if fits_count(s) {
                fits.push(mask);
            }
        }
        fit_count = Some(fits.len());
        if let [one] = fits.as_slice() {
            chosen = (0..n).filter(|i| one >> i & 1 == 1).collect();
            selected = true;
        }
    }
    if !selected {
        // Every candidate binds the same values: the count is only needed to
        // apportion distinct values.
        let keys: BTreeSet<(u64, u64, u64)> = units
            .iter()
            .flat_map(|(_, u)| u.iter())
            .map(|g| {
                (
                    extent(g, by_axis),
                    if param == by {
                        0
                    } else {
                        extent(g, other_axis)
                    },
                    if reads_stack(ctx.abi, ctx.work) && g.stacked {
                        g.stack_len.max(1)
                    } else {
                        1
                    },
                )
            })
            .collect();
        if keys.len() == 1 {
            selected = true;
            let mut note = format!("unique value across {n} candidates");
            if fit_count != Some(1) {
                note.push_str(&format!(
                    " (calls {} per pass / {:?} per chunk vs {} weight launches)",
                    ctx.calls_bind, ctx.calls_per_chunk, total_all
                ));
            }
            st.notes.push(note);
        }
    }
    if !selected {
        st.notes.push(match fit_count {
            Some(0) => format!(
                "inconsistent: {} calls vs {} weight launches ({} candidate groups)",
                ctx.calls_bind, total_all, n
            ),
            Some(k) => format!(
                "inconsistent: {} calls vs {} weight launches ({} candidate sets fit)",
                ctx.calls_bind, total_all, k
            ),
            None => {
                format!("inconsistent: {n} candidate weight groups (> 20), set selection skipped")
            }
        });
    }

    let total: u64 = chosen.iter().map(|&i| unit_launches(&units[i].1)).sum();
    let mut split: BTreeMap<(u64, u64), Vec<&WeightGroup>> = BTreeMap::new();
    for &i in &chosen {
        for g in &units[i].1 {
            let by_ext = extent(g, by_axis);
            let other = if param == by {
                0
            } else {
                extent(g, other_axis)
            };
            split.entry((by_ext, other)).or_default().push(g);
        }
    }
    let mut out = Vec::with_capacity(split.len());
    for ((by_ext, other), gs) in split {
        let mut s = st.clone();
        s.vals.insert(by.to_string(), by_ext);
        if param != by {
            s.vals.insert(param.to_string(), other);
        }
        let stack = gs
            .iter()
            .map(|g| if g.stacked { g.stack_len.max(1) } else { 1 })
            .max()
            .unwrap_or(1);
        s.vals.insert(STACK_PARAM.to_string(), stack);
        let l: u64 = gs.iter().map(|g| launches(g)).sum();
        s.frac *= if total == 0 {
            1.0
        } else {
            l as f64 / total as f64
        };
        s.stacked = Some(gs.iter().all(|g| g.stacked));
        s.groups.extend(gs.into_iter().cloned());
        out.push(s);
    }
    out
}

/// MAC/FLOP/byte values of one bound state (per call). MACs are strict; flops
/// and bytes fall back to the upper bound of a grid interval when that is the
/// only thing unbound (and the MACs are evaluable): `upper`.
struct Eval {
    macs: Option<u64>,
    flops: Option<u64>,
    bytes: Option<u64>,
    unbound: BTreeSet<String>,
    errors: Vec<String>,
    byte_note: Option<String>,
    upper: bool,
}

fn eval_tol(
    e: &Expr,
    what: &str,
    vals: &BTreeMap<String, u64>,
    tvals: &BTreeMap<String, u64>,
    export: &ExportV1,
    allow_upper: bool,
    ev: &mut Eval,
) -> Option<u64> {
    match eval_vals(e, vals, export) {
        Ok(v) => Some(v),
        Err(EvErr::Unbound(n)) => {
            ev.unbound.insert(n);
            if !allow_upper {
                return None;
            }
            match eval_vals(e, tvals, export) {
                Ok(v) => {
                    ev.upper = true;
                    Some(v)
                }
                Err(EvErr::Unbound(_)) => None,
                Err(EvErr::Math(m)) => {
                    ev.errors.push(format!("{what}: {m}"));
                    None
                }
            }
        }
        Err(EvErr::Math(m)) => {
            ev.errors.push(format!("{what}: {m}"));
            None
        }
    }
}

fn evaluate(
    abi: &KernelAbiV1,
    export: &ExportV1,
    work: &WorkV1,
    vals: &BTreeMap<String, u64>,
    ivs: &BTreeMap<String, Iv>,
) -> Eval {
    let mut ev = Eval {
        macs: None,
        flops: None,
        bytes: None,
        unbound: BTreeSet::new(),
        errors: Vec::new(),
        byte_note: None,
        upper: false,
    };
    let mut tvals = vals.clone();
    for (k, iv) in ivs {
        if iv.hi != u64::MAX {
            tvals.entry(k.clone()).or_insert(iv.hi);
        }
    }
    if let Some(m) = &work.macs {
        ev.macs = eval_tol(m, "macs", vals, &tvals, export, false, &mut ev);
    }
    let allow = work.macs.is_none() || ev.macs.is_some();
    if let Some(f) = &work.flops {
        ev.flops = eval_tol(f, "flops", vals, &tvals, export, allow, &mut ev);
    }
    if let Some(b) = &work.bytes {
        ev.bytes = eval_tol(b, "bytes", vals, &tvals, export, allow, &mut ev);
    } else {
        // The sum of the contract's operand extents.
        let mut sum = 0u64;
        let mut ok = true;
        for op in abi.operands {
            match &op.extent {
                Some(x) => match eval_tol(
                    x,
                    &format!("operand {}", op.name),
                    vals,
                    &tvals,
                    export,
                    allow,
                    &mut ev,
                ) {
                    Some(v) => sum = sum.saturating_add(v),
                    None => ok = false,
                },
                None => {
                    ev.byte_note = Some(format!("bytes: operand {} has no extent", op.name));
                    ok = false;
                }
            }
        }
        if ok {
            ev.bytes = Some(sum);
        }
    }
    ev
}

fn arith_ns(
    hw: &Peaks,
    pipe: Pipe,
    macs: Option<f64>,
    flops: Option<f64>,
) -> Result<Option<f64>, String> {
    if macs.is_none() && flops.is_none() {
        return Ok(None);
    }
    let mut s = 0.0;
    if let Some(m) = macs.filter(|m| *m > 0.0) {
        let p = hw::peak(hw, pipe);
        if p <= 0.0 {
            return Err(format!("{} has no peak for pipe {:?}", hw.arch, pipe));
        }
        s += 2.0 * m / p * 1e9;
    }
    if let Some(f) = flops.filter(|f| *f > 0.0) {
        let p = hw.valu_f32;
        if p <= 0.0 {
            return Err(format!("{} has no VALU F32 peak", hw.arch));
        }
        s += f / p * 1e9;
    }
    Ok(Some(s))
}

fn floors_of(
    hw: &Peaks,
    pipe: Pipe,
    macs: Option<f64>,
    flops: Option<f64>,
    bytes: Option<f64>,
    note: &mut Vec<String>,
) -> Floors {
    let arith = match arith_ns(hw, pipe, macs, flops) {
        Ok(a) => a,
        Err(e) => {
            note.push(e);
            None
        }
    };
    let mem = bytes.map(|b| if hw.bw > 0.0 { b / hw.bw * 1e9 } else { 0.0 });
    let floor = match (arith, mem) {
        (Some(a), Some(m)) => Some(a.max(m)),
        (Some(a), None) => Some(a),
        (None, Some(m)) => Some(m),
        (None, None) => None,
    };
    Floors {
        mem_ns: mem,
        arith_ns: arith,
        floor_ns: floor,
    }
}

fn pipe_name(p: Pipe) -> &'static str {
    match p {
        Pipe::WmmaF16 => "wmma_f16",
        Pipe::WmmaBf16 => "wmma_bf16",
        Pipe::WmmaFp8 => "wmma_fp8",
        Pipe::WmmaIu8 => "wmma_iu8",
        Pipe::WmmaIu4 => "wmma_iu4",
        Pipe::ValuF32 => "valu_f32",
        Pipe::Memory => "memory",
    }
}

/// Parse a pipe name (`wmma_f16`, also the short forms `f16`, `bf16`, `fp8`,
/// `iu8`, `iu4`, `valu`).
pub fn parse_pipe(s: &str) -> Result<Pipe, String> {
    Ok(match s {
        "wmma_f16" | "f16" => Pipe::WmmaF16,
        "wmma_bf16" | "bf16" => Pipe::WmmaBf16,
        "wmma_fp8" | "fp8" => Pipe::WmmaFp8,
        "wmma_iu8" | "iu8" => Pipe::WmmaIu8,
        "wmma_iu4" | "iu4" => Pipe::WmmaIu4,
        "valu_f32" | "valu" | "f32" => Pipe::ValuF32,
        "memory" | "mem" => Pipe::Memory,
        other => {
            return Err(format!(
                "unknown pipe {other:?} (wmma_f16|wmma_bf16|wmma_fp8|wmma_iu8|wmma_iu4|valu_f32|memory)"
            ))
        }
    })
}

fn debt_row(
    key: &NodeKey,
    n: &Node,
    npass: f64,
    wall: f64,
    status: &str,
    abi_id: Option<&str>,
    note: String,
) -> Debt {
    Debt {
        symbol: key.0.clone(),
        grid: key.1,
        block: key.2,
        lds: key.3,
        calls: n.calls as f64 / npass,
        time_ns: n.time_ns as f64 / npass,
        share: if wall > 0.0 {
            n.time_ns as f64 / npass / wall
        } else {
            0.0
        },
        status: status.to_string(),
        abi_id: abi_id.map(str::to_string),
        note,
    }
}

/// Bind every node of `node` into leaves (or a debt row).
#[allow(clippy::too_many_arguments)]
fn bind_node(
    key: &NodeKey,
    n: &Node,
    found: Found,
    model: &ModelDesc,
    groups: &[WeightGroup],
    chunk_rows: u64,
    chunks_per_pass: u64,
    tokens: u64,
    npass: f64,
    wall: f64,
    hw: &Peaks,
) -> Result<Result<Vec<Leaf>, Debt>, String> {
    let (abi, export, spec) = found;
    let Some(spec) = spec else {
        return Ok(Err(debt_row(
            key,
            n,
            npass,
            wall,
            "no_work_spec",
            Some(abi.id),
            "contract found, no WorkSpecV1 attached".to_string(),
        )));
    };
    let work = &spec.work;
    if export.block != key.2 {
        return Ok(Err(debt_row(
            key,
            n,
            npass,
            wall,
            "block_mismatch",
            Some(abi.id),
            format!(
                "trace block {:?} != contract block {:?}",
                key.2, export.block
            ),
        )));
    }
    let Some(ivs) = grid_intervals(abi, export, key.1) else {
        return Ok(Err(debt_row(
            key,
            n,
            npass,
            wall,
            "grid_mismatch",
            Some(abi.id),
            format!(
                "workgroup grid {:?} is unreachable through the contract grid",
                key.1
            ),
        )));
    };

    let calls_mean = n.calls as f64 / npass;
    let calls_bind = calls_mean.round() as u64;
    let calls_per_chunk = (chunks_per_pass > 0 && calls_bind % chunks_per_pass == 0)
        .then(|| calls_bind / chunks_per_pass);
    let ctx = BindCtx {
        abi,
        export,
        work,
        model,
        groups,
        chunk_rows,
        tokens,
        calls_bind,
        calls_per_chunk,
    };

    let mut vals = BTreeMap::new();
    let mut open = BTreeMap::new();
    for (name, iv) in &ivs {
        if iv.is_point() {
            vals.insert(name.clone(), iv.lo);
        } else {
            open.insert(name.clone(), *iv);
        }
    }
    let mut init = St {
        vals,
        ivs: open,
        groups: Vec::new(),
        frac: 1.0,
        notes: Vec::new(),
        stacked: None,
    };
    let mut lds_note = String::new();
    let want_lds = export
        .group_segment_fixed
        .saturating_add(export.dynamic_lds);
    if want_lds != key.3 {
        lds_note = format!("lds: trace {} != contract {want_lds}", key.3);
    }
    if init.vals.is_empty() && init.ivs.is_empty() {
        init.notes.push("grid binds no param".to_string());
    }

    let causal_param = work
        .bind
        .iter()
        .find(|b| matches!(b.source, BindSource::CausalPrefix))
        .map(|b| b.param);
    // Calls per chunk ordinal (causal ops evaluate per call).
    let classes: Vec<(u64 /*prefix*/, u64 /*calls*/)> = match causal_param {
        Some(_) => n
            .ordinals
            .iter()
            .map(|(o, c)| (u64::from(*o) * chunk_rows, *c))
            .collect(),
        None => vec![(0, n.calls)],
    };

    let mut per_class: Vec<(u64, Vec<St>)> = Vec::new();
    for (prefix, count) in &classes {
        per_class.push((*count, run_binds(&ctx, &init, *prefix)));
    }
    let Some((_, first)) = per_class.first() else {
        return Ok(Err(debt_row(
            key,
            n,
            npass,
            wall,
            "grid_mismatch",
            Some(abi.id),
            "no calls".into(),
        )));
    };
    let leaf_count = first.len();
    if per_class.iter().any(|(_, s)| s.len() != leaf_count) {
        return Err(format!(
            "{}: weight split differs between causal classes",
            key.0
        ));
    }

    struct Acc {
        macs: Option<f64>,
        flops: Option<f64>,
        bytes: Option<f64>,
        unbound: BTreeSet<String>,
        errors: Vec<String>,
        byte_note: Option<String>,
        upper: bool,
        calls: f64,
    }
    let mut leaves_raw: Vec<(Acc, &St)> = Vec::with_capacity(leaf_count);
    for i in 0..leaf_count {
        let mut acc = Acc {
            macs: work.macs.map(|_| 0.0),
            flops: work.flops.map(|_| 0.0),
            bytes: Some(0.0),
            unbound: BTreeSet::new(),
            errors: Vec::new(),
            byte_note: None,
            upper: false,
            calls: 0.0,
        };
        let frac = first[i].frac;
        for (count, states) in &per_class {
            let st = &states[i];
            let w = *count as f64 * frac / npass;
            acc.calls += w;
            let ev = evaluate(abi, export, work, &st.vals, &st.ivs);
            acc.unbound.extend(ev.unbound);
            acc.errors.extend(ev.errors);
            acc.upper |= ev.upper;
            if ev.byte_note.is_some() {
                acc.byte_note = ev.byte_note;
            }
            let add = |a: &mut Option<f64>, v: Option<u64>| {
                *a = match (*a, v) {
                    (Some(s), Some(v)) => Some(s + v as f64 * w),
                    _ => None,
                };
            };
            add(&mut acc.macs, ev.macs);
            add(&mut acc.flops, ev.flops);
            add(&mut acc.bytes, ev.bytes);
        }
        leaves_raw.push((acc, &first[i]));
    }

    // Time split between the leaves of one node.
    let node_time = n.time_ns as f64 / npass;
    let macs_w: Vec<Option<f64>> = leaves_raw.iter().map(|(a, _)| a.macs).collect();
    let (weights, basis): (Vec<f64>, &str) = if leaf_count == 1 {
        (vec![1.0], "node")
    } else if macs_w.iter().all(|m| m.is_some_and(|m| m > 0.0)) {
        (
            macs_w.iter().map(|m| m.unwrap()).collect(),
            "apportioned_by_macs",
        )
    } else {
        (
            leaves_raw.iter().map(|(a, _)| a.calls).collect(),
            "apportioned_by_calls",
        )
    };
    let wsum: f64 = weights.iter().sum();

    let mut leaves = Vec::with_capacity(leaf_count);
    for (i, (acc, st)) in leaves_raw.into_iter().enumerate() {
        let time_ns = if wsum > 0.0 {
            node_time * weights[i] / wsum
        } else {
            node_time / leaf_count as f64
        };
        let mut notes = st.notes.clone();
        notes.sort();
        notes.dedup();
        let mut params = st.vals.clone();
        if let Some(p) = causal_param {
            params.remove(p);
        }
        let intervals: BTreeMap<String, Iv> = st
            .ivs
            .iter()
            .filter(|(k, v)| !params.contains_key(*k) && **v != Iv::ALL)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let mut unbound: Vec<String> = acc.unbound.iter().cloned().collect();
        unbound.sort();
        let rows_unbound = work.rows.is_some_and(|r| !params.contains_key(r));
        if let (true, Some(r)) = (rows_unbound, work.rows) {
            if !unbound.iter().any(|u| u == r) {
                unbound.push(r.to_string());
            }
        }
        let inconsistent = notes.iter().any(|s| s.starts_with("inconsistent"));
        let status = if inconsistent {
            "inconsistent"
        } else if !unbound.is_empty() || !acc.errors.is_empty() {
            "partial"
        } else {
            "ok"
        };
        let (macs, flops) = (acc.macs, acc.flops);
        let bytes = acc.bytes.filter(|_| acc.byte_note.is_none());
        let mut floor_notes = Vec::new();
        let floors = floors_of(hw, work.pipe, macs, flops, bytes, &mut floor_notes);
        let floors_have = floors.floor_ns.is_some();
        let efficiency = floors
            .floor_ns
            .filter(|_| time_ns > 0.0)
            .map(|f| f / time_ns);
        let rows = work.rows.and_then(|r| params.get(r).copied());
        let admit: RowDomainV1 = work.admit;
        let row_class = RowClass {
            min_rows: admit.min_rows,
            max_rows: admit.max_rows,
            predicate: admit.predicate.to_string(),
            kill_env: admit.kill_env.map(str::to_string),
            rows,
            admitted: rows.map(|r| r >= admit.min_rows && admit.max_rows.is_none_or(|m| r <= m)),
        };
        let mut note_parts: Vec<String> = Vec::new();
        if !work.note.is_empty() {
            note_parts.push(work.note.to_string());
        }
        if !lds_note.is_empty() {
            note_parts.push(lds_note.clone());
        }
        note_parts.extend(acc.errors.iter().cloned());
        note_parts.extend(acc.byte_note.iter().cloned());
        note_parts.extend(floor_notes);
        leaves.push(Leaf {
            symbol: key.0.clone(),
            abi_id: abi.id.to_string(),
            abi_version: abi.version,
            abi_digest: abi.digest(),
            op_class: work.op_class.to_string(),
            codec: work.codec.to_string(),
            pipe: pipe_name(work.pipe).to_string(),
            grid: key.1,
            block: key.2,
            lds: key.3,
            params,
            intervals,
            unbound,
            weight: st.groups.iter().map(WeightRef::from).collect(),
            calls: acc.calls,
            launches_per_chunk: acc.calls / chunks_per_pass.max(1) as f64,
            time_ns,
            share: if wall > 0.0 { time_ns / wall } else { 0.0 },
            time_basis: basis.to_string(),
            macs,
            flops,
            bytes,
            row_class,
            floors,
            efficiency,
            floor_basis: floors_have
                .then(|| if acc.upper { "interval_upper" } else { "exact" }.to_string()),
            status: status.to_string(),
            binding: if notes.is_empty() {
                "ok".to_string()
            } else {
                notes.join("; ")
            },
            causal: causal_param.is_some(),
            note: note_parts.join("; "),
        });
    }
    Ok(Ok(leaves))
}

// ---------------------------------------------------------------------------
// corpus

/// Mode (weighted by dispatches) of the exact values the grids bind to the
/// work spec's `rows` param; contracts without a rows param are excluded.
fn rows_mode(
    disp: &[&rocprof::Dispatch],
    names: &[String],
    cache: &HashMap<String, Option<Found>>,
) -> Option<u64> {
    let mut seen: BTreeMap<u64, u64> = BTreeMap::new();
    for (i, d) in disp.iter().enumerate() {
        let Some((abi, export, Some(spec))) = cache[&names[i]] else {
            continue;
        };
        let Some(rows) = spec.work.rows else { continue };
        if let Some(iv) = grid_intervals(abi, export, d.grid_workgroups())
            .and_then(|m| m.get(rows).copied())
            .filter(|iv| iv.is_point())
        {
            *seen.entry(iv.lo).or_default() += 1;
        }
    }
    seen.into_iter()
        .max_by_key(|(r, c)| (*c, *r))
        .map(|(r, _)| r)
}

/// Build the node corpus of `trace` against `model` through `lookup`.
///
/// Passes: the dispatches whose work op class is `embedding` are the anchors;
/// chunk rows come from the anchors' grid-bound rows; `tokens / chunk_rows`
/// anchors make one pass. Without anchors, `opts.window` must be given and is
/// taken as one single-chunk pass.
pub fn build_corpus(
    trace: &rocprof::Trace,
    model: &ModelDesc,
    opts: &NodesOpts,
    lookup: &Lookup,
) -> Result<Corpus, String> {
    let src = &trace.source.path;
    let arch = strip_features(&opts.arch).to_string();
    let hw = hw::peaks(&arch).ok_or_else(|| format!("no hardware peak table for arch {arch}"))?;
    if opts.tokens == 0 {
        return Err("--tokens must be positive".to_string());
    }
    let agent =
        crate::census::resolve_agent(trace, opts.agent).map_err(|e| format!("{src}: {e}"))?;
    let mut disp: Vec<&rocprof::Dispatch> = trace
        .dispatches
        .iter()
        .filter(|d| d.agent == agent)
        .filter(|d| match opts.window {
            Some((s, e)) => d.start_ns >= s && d.end_ns <= e,
            None => true,
        })
        .collect();
    if disp.is_empty() {
        return Err(format!(
            "{src}: no dispatch of agent {agent} in the selected window"
        ));
    }
    disp.sort_by_key(|d| (d.start_ns, d.end_ns));
    let outside_window = trace.dispatches.iter().filter(|d| d.agent == agent).count() - disp.len();

    let sym = |d: &rocprof::Dispatch| -> String {
        trace
            .kernel(d.kernel_id)
            .map(|k| k.name.clone())
            .unwrap_or_else(|| format!("kernel_id:{}", d.kernel_id))
    };
    let mut cache: HashMap<String, Option<Found>> = HashMap::new();
    let mut names: Vec<String> = Vec::with_capacity(disp.len());
    for d in &disp {
        let s = sym(d);
        if !cache.contains_key(&s) {
            let f = lookup(&arch, &s);
            cache.insert(s.clone(), f);
        }
        names.push(s);
    }

    // anchors
    let mut anchors: Vec<usize> = Vec::new();
    let mut rows_seen: BTreeMap<u64, u64> = BTreeMap::new();
    for (i, d) in disp.iter().enumerate() {
        let Some((abi, export, Some(spec))) = cache[&names[i]] else {
            continue;
        };
        if spec.work.op_class != op_class::EMBEDDING {
            continue;
        }
        anchors.push(i);
        if let (Some(rows), Some(ivs)) = (
            spec.work.rows,
            grid_intervals(abi, export, d.grid_workgroups()),
        ) {
            if let Some(iv) = ivs.get(rows).filter(|iv| iv.is_point()) {
                *rows_seen.entry(iv.lo).or_default() += 1;
            }
        }
    }

    let mut notes: Vec<String> = Vec::new();
    let (chunk_rows, chunks_per_pass, mut passes) = if anchors.is_empty() {
        let Some((s, e)) = opts.window else {
            return Err(format!(
                "{src}: no embedding anchor (a dispatch whose work op class is `embedding`) in the trace; pass --window START_NS:END_NS"
            ));
        };
        let chunk_rows = rows_mode(&disp, &names, &cache)
            .unwrap_or(opts.tokens)
            .max(1);
        notes.push(format!(
            "no embedding anchor: --window is one pass; chunk rows {chunk_rows} (mode of the grid-bound rows params); chunk ordinals unknown, causal prefixes evaluate at 0"
        ));
        (
            chunk_rows,
            opts.tokens.div_ceil(chunk_rows),
            vec![Pass {
                start: s,
                end: e,
                anchors: vec![s],
                members: (0..disp.len()).collect(),
            }],
        )
    } else {
        let Some((&chunk_rows, _)) = rows_seen.iter().max_by_key(|(r, c)| (**c, **r)) else {
            return Err(format!(
                "{src}: embedding anchors found but none binds its rows exactly through the grid"
            ));
        };
        if let Some(m) = rows_mode(&disp, &names, &cache).filter(|m| *m != chunk_rows) {
            notes.push(format!(
                "chunk rows {chunk_rows} from the anchors differ from the mode {m} of all grid-bound rows params"
            ));
        }
        if rows_seen.len() > 1 {
            notes.push(format!(
                "anchor rows {rows_seen:?}: chunk rows = {chunk_rows} (mode)"
            ));
        }
        if chunk_rows == 0 {
            return Err(format!("{src}: anchor rows bind to zero"));
        }
        let cpp = opts.tokens.div_ceil(chunk_rows);
        if opts.tokens % chunk_rows != 0 {
            notes.push(format!(
                "tokens {} not a multiple of chunk rows {chunk_rows}: {cpp} chunks per pass, last one ragged",
                opts.tokens
            ));
        }
        let full = anchors.len() / cpp as usize;
        if anchors.len() % cpp as usize != 0 {
            notes.push(format!(
                "{} anchors is not a multiple of {cpp} chunks per pass: the trailing {} are not a pass and are excluded",
                anchors.len(),
                anchors.len() % cpp as usize
            ));
        }
        if full == 0 {
            return Err(format!(
                "{src}: {} embedding anchors, fewer than the {cpp} chunks of one {}-token pass",
                anchors.len(),
                opts.tokens
            ));
        }
        let last_end = disp.iter().map(|d| d.end_ns).max().unwrap_or(0);
        let mut passes: Vec<Pass> = Vec::with_capacity(full);
        for p in 0..full {
            let first = anchors[p * cpp as usize];
            let start = disp[first].start_ns;
            let end = match anchors.get((p + 1) * cpp as usize) {
                Some(&next) => disp[next].start_ns,
                None => last_end,
            };
            let a: Vec<u64> = (0..cpp as usize)
                .map(|c| disp[anchors[p * cpp as usize + c]].start_ns)
                .collect();
            passes.push(Pass {
                start,
                end,
                anchors: a,
                members: Vec::new(),
            });
        }
        (chunk_rows, cpp, passes)
    };
    if !anchors.is_empty() {
        for (i, d) in disp.iter().enumerate() {
            if let Some(p) = passes
                .iter_mut()
                .find(|p| d.start_ns >= p.start && d.start_ns < p.end)
            {
                p.members.push(i);
            }
        }
    }
    let in_pass: usize = passes.iter().map(|p| p.members.len()).sum();
    let outside_pass = disp.len() - in_pass + outside_window;

    let stats: Vec<PassStat> = passes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let busy = union_len(
                p.members
                    .iter()
                    .map(|&m| (disp[m].start_ns.max(p.start), disp[m].end_ns.min(p.end)))
                    .filter(|(s, e)| s < e)
                    .collect(),
            );
            let wall = p.end.saturating_sub(p.start);
            PassStat {
                index: i,
                start_ns: p.start,
                end_ns: p.end,
                wall_ns: wall,
                busy_ns: busy,
                idle_ns: wall.saturating_sub(busy),
                launches: p.members.len() as u64,
            }
        })
        .collect();
    let used: Vec<usize> = match opts.pass {
        PassMode::Mean => (0..passes.len()).collect(),
        PassMode::Last => vec![passes.len() - 1],
    };
    let npass = used.len() as f64;
    let mean =
        |f: &dyn Fn(&PassStat) -> f64| used.iter().map(|&i| f(&stats[i])).sum::<f64>() / npass;
    let wall = mean(&|s| s.wall_ns as f64);
    let busy = mean(&|s| s.busy_ns as f64);
    let idle = mean(&|s| s.idle_ns as f64);
    let launches = mean(&|s| s.launches as f64);

    // nodes over the used passes
    let mut nodes: BTreeMap<NodeKey, Node> = BTreeMap::new();
    for &pi in &used {
        let p = &passes[pi];
        for &m in &p.members {
            let d = disp[m];
            let ordinal = p
                .anchors
                .partition_point(|&s| s <= d.start_ns)
                .saturating_sub(1) as u32;
            let key: NodeKey = (
                names[m].clone(),
                d.grid_workgroups(),
                d.workgroup,
                d.group_segment_size,
            );
            let n = nodes.entry(key).or_insert(Node {
                calls: 0,
                time_ns: 0,
                ordinals: BTreeMap::new(),
            });
            n.calls += 1;
            n.time_ns += d.end_ns.saturating_sub(d.start_ns);
            *n.ordinals.entry(ordinal).or_default() += 1;
        }
    }

    let groups = model.weight_groups();
    let mut leaves: Vec<Leaf> = Vec::new();
    let mut debt: Vec<Debt> = Vec::new();
    for (key, n) in &nodes {
        match cache[&key.0] {
            None => debt.push(debt_row(
                key,
                n,
                npass,
                wall,
                "no_contract",
                None,
                format!("no contract exports {} on {arch}", key.0),
            )),
            Some(found) => match bind_node(
                key,
                n,
                found,
                model,
                &groups,
                chunk_rows,
                chunks_per_pass,
                opts.tokens,
                npass,
                wall,
                &hw,
            )? {
                Ok(mut l) => leaves.append(&mut l),
                Err(d) => debt.push(d),
            },
        }
    }
    leaves.sort_by(|a, b| {
        b.time_ns
            .total_cmp(&a.time_ns)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    debt.sort_by(|a, b| {
        b.time_ns
            .total_cmp(&a.time_ns)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });

    let mut totals: BTreeMap<String, ClassTotal> = BTreeMap::new();
    for l in &leaves {
        let t = totals.entry(l.op_class.clone()).or_default();
        t.leaves += 1;
        t.calls += l.calls;
        t.time_ns += l.time_ns;
        t.share += l.share;
        t.macs += l.macs.unwrap_or(0.0);
        t.flops += l.flops.unwrap_or(0.0);
        t.bytes += l.bytes.unwrap_or(0.0);
    }
    if !debt.is_empty() {
        let t = totals.entry("(debt)".to_string()).or_default();
        for d in &debt {
            t.leaves += 1;
            t.calls += d.calls;
            t.time_ns += d.time_ns;
            t.share += d.share;
        }
    }

    let roof = roofline(&leaves, opts.tokens, wall, busy, &hw, &[])?;
    let hfq = model.hfq.as_ref().map(|h| HfqRef {
        path: h.path.clone(),
        index_sha256: h.index_sha256.clone(),
        arch_id: h.arch_id,
        tensors: h.tensors.len(),
    });
    Ok(Corpus {
        schema: SCHEMA.to_string(),
        trace: TraceRef {
            path: trace.source.path.clone(),
            sha256: trace.source.sha256.clone(),
            format: trace.source.format.clone(),
        },
        config_sha256: model.config_sha256.clone(),
        hfq,
        arch,
        tokens: opts.tokens,
        pass_mode: opts.pass.name().to_string(),
        passes_found: passes.len(),
        passes_used: used,
        passes: stats,
        chunk_rows,
        chunks_per_pass,
        wall_ns: wall,
        busy_ns: busy,
        idle_ns: idle,
        launches_per_pass: launches,
        outside_pass_dispatches: outside_pass as u64,
        bench_profile: None,
        leaves,
        debt,
        totals_by_op_class: totals,
        roofline: roof,
        notes: {
            let mut n = notes;
            n.push("roofline ceilings per pass: peak_arith_tok_s = tokens / Σ_leaves(2·macs / peak(leaf pipe) + flops / valu_f32 peak); all_f16_peak_tok_s = tokens / Σ_leaves(2·macs / wmma_f16 peak + flops / valu_f32 peak), the PHASE1 §1 'all-F16 WMMA' route (every MAC charged to F16 WMMA); floor_tok_s additionally takes max(arith, memory) per leaf".to_string());
            n
        },
    })
}

/// Read the section of a bench JSON's `result.profile.sections[]` whose
/// `units` equals `tokens` and record its `residual_ms` next to the trace idle.
/// The trace data is never replaced.
pub fn bench_profile_record(path: &Path, tokens: u64, trace_idle_ns: f64) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: not JSON: {e}", path.display()))?;
    let sections = v
        .pointer("/result/profile/sections")
        .or_else(|| v.pointer("/profile/sections"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{}: no result.profile.sections[]", path.display()))?;
    let sec = sections
        .iter()
        .find(|s| s.get("units").and_then(Value::as_f64) == Some(tokens as f64))
        .ok_or_else(|| {
            format!(
                "{}: no profile section with units == {tokens} (have {:?})",
                path.display(),
                sections
                    .iter()
                    .filter_map(|s| s.get("units").and_then(Value::as_f64))
                    .collect::<Vec<_>>()
            )
        })?;
    let residual = sec
        .get("residual_ms")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{}: section has no residual_ms", path.display()))?;
    Ok(json!({
        "path": path.display().to_string(),
        "sha256": sha256_hex(&bytes),
        "section": sec.get("section").cloned().unwrap_or(Value::Null),
        "units": tokens,
        "residual_ms": residual,
        "wall_unprofiled_ms": sec.get("wall_unprofiled_ms").cloned().unwrap_or(Value::Null),
        "wall_profiled_ms": sec.get("wall_profiled_ms").cloned().unwrap_or(Value::Null),
        "trace_idle_ms": trace_idle_ns / 1e6,
        "note": "residual_ms is the bench's internal-profile launch-gap residual; trace_idle_ms is the rocprof pass wall minus the busy union. Recorded side by side, neither replaces the other.",
    }))
}

// ---------------------------------------------------------------------------
// roofline

/// Per-token MAC/FLOP by op class and weight group and the peak ceilings.
/// `what_if` re-charges leaves whose op class or codec equals the key to another
/// pipe (first match wins).
pub fn roofline(
    leaves: &[Leaf],
    tokens: u64,
    wall_ns: f64,
    busy_ns: f64,
    hw: &Peaks,
    what_if: &[(String, Pipe)],
) -> Result<Value, String> {
    let tok = tokens.max(1) as f64;
    #[derive(Default)]
    struct Agg {
        macs: f64,
        flops: f64,
        bytes: f64,
        arith_ns: f64,
        mem_ns: f64,
        floor_ns: f64,
        measured_ns: f64,
        leaves: u64,
    }
    let mut by_class: BTreeMap<String, Agg> = BTreeMap::new();
    let mut by_group: BTreeMap<String, (Value, Agg)> = BTreeMap::new();
    let (mut sum_arith, mut sum_floor, mut sum_f16) = (0.0f64, 0.0f64, 0.0f64);
    let (mut tot_macs, mut tot_flops, mut tot_bytes) = (0.0f64, 0.0f64, 0.0f64);
    let mut without = 0u64;
    let mut charged: Vec<Value> = Vec::new();
    for l in leaves {
        let pipe = what_if
            .iter()
            .find(|(k, _)| *k == l.op_class || *k == l.codec)
            .map(|(_, p)| *p);
        let own = parse_pipe(&l.pipe)?;
        let eff_pipe = pipe.unwrap_or(own);
        if l.macs.is_none() && l.flops.is_none() && l.bytes.is_none() {
            without += 1;
            continue;
        }
        let arith = arith_ns(hw, eff_pipe, l.macs, l.flops)?;
        let mem = l.floors.mem_ns;
        let floor = match (arith, mem) {
            (Some(a), Some(m)) => a.max(m),
            (Some(a), None) => a,
            (None, Some(m)) => m,
            (None, None) => 0.0,
        };
        let f16 = arith_ns(hw, Pipe::WmmaF16, l.macs, l.flops)?.unwrap_or(0.0);
        if pipe.is_some() {
            charged.push(json!({"op_class": l.op_class, "codec": l.codec, "symbol": l.symbol, "pipe": pipe_name(eff_pipe)}));
        }
        let (m, f, b) = (
            l.macs.unwrap_or(0.0),
            l.flops.unwrap_or(0.0),
            l.bytes.unwrap_or(0.0),
        );
        sum_arith += arith.unwrap_or(0.0);
        sum_floor += floor;
        sum_f16 += f16;
        tot_macs += m;
        tot_flops += f;
        tot_bytes += b;
        let add = |a: &mut Agg| {
            a.macs += m;
            a.flops += f;
            a.bytes += b;
            a.arith_ns += arith.unwrap_or(0.0);
            a.mem_ns += mem.unwrap_or(0.0);
            a.floor_ns += floor;
            a.measured_ns += l.time_ns;
            a.leaves += 1;
        };
        add(by_class.entry(l.op_class.clone()).or_default());
        if !l.weight.is_empty() {
            let members: Vec<&str> = l
                .weight
                .iter()
                .flat_map(|w| w.members.iter().map(String::as_str))
                .collect();
            let key = format!("{}|{}", members.join("+"), l.codec);
            let w0 = &l.weight[0];
            let meta = json!({
                "members": members,
                "op_class": l.op_class,
                "codec": l.codec,
                "rows": w0.rows,
                "cols": w0.cols,
                "stacked": w0.stacked,
                "fused": w0.fused,
            });
            add(&mut by_group
                .entry(key)
                .or_insert_with(|| (meta, Agg::default()))
                .1);
        }
    }
    let agg_json = |a: &Agg| {
        json!({
            "leaves": a.leaves,
            "macs_per_token": a.macs / tok,
            "flops_per_token": a.flops / tok,
            "bytes_per_token": a.bytes / tok,
            "arith_ns": a.arith_ns,
            "mem_ns": a.mem_ns,
            "floor_ns": a.floor_ns,
            "measured_ns": a.measured_ns,
        })
    };
    let classes: BTreeMap<&String, Value> =
        by_class.iter().map(|(k, a)| (k, agg_json(a))).collect();
    let mut groups: Vec<Value> = by_group
        .values()
        .map(|(meta, a)| {
            let mut v = agg_json(a);
            if let (Some(o), Some(m)) = (v.as_object_mut(), meta.as_object()) {
                for (k, x) in m {
                    o.insert(k.clone(), x.clone());
                }
            }
            v
        })
        .collect();
    groups.sort_by(|a, b| {
        b["measured_ns"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["measured_ns"].as_f64().unwrap_or(0.0))
    });
    let ceil = |ns: f64| {
        if ns > 0.0 {
            Some(tok / (ns * 1e-9))
        } else {
            None
        }
    };
    Ok(json!({
        "arch": hw.arch,
        "hw": {"name": hw.name, "source": hw.source},
        "tokens": tokens,
        "what_if": what_if.iter().map(|(k, p)| json!({"match": k, "pipe": pipe_name(*p)})).collect::<Vec<_>>(),
        "what_if_charged": charged,
        "per_token": {
            "macs": tot_macs / tok,
            "flops": tot_flops / tok,
            "gflop": (2.0 * tot_macs + tot_flops) / tok / 1e9,
            "bytes": tot_bytes / tok,
        },
        "by_op_class": classes,
        "by_weight_group": groups,
        "leaves_without_work": without,
        "ceilings": {
            "peak_arith_tok_s": ceil(sum_arith),
            "floor_tok_s": ceil(sum_floor),
            "all_f16_peak_tok_s": ceil(sum_f16),
            "measured_tok_s": ceil(wall_ns),
            "busy_tok_s": ceil(busy_ns),
        },
    }))
}

// ---------------------------------------------------------------------------
// queue

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueItem {
    pub corpus: String,
    pub rank: usize,
    pub symbol: String,
    pub op_class: String,
    pub codec: String,
    pub shape: String,
    pub weight: String,
    pub share: f64,
    pub time_ns: f64,
    pub floor_ns: Option<f64>,
    pub efficiency: Option<f64>,
    pub floor_basis: Option<String>,
    /// `share x (1 - floor / measured)`, 0 when the node already sits at or below its floor.
    pub score: f64,
    pub status: String,
    pub binding: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct QueueCorpus {
    pub label: String,
    pub path: String,
    pub arch: String,
    pub tokens: u64,
    pub pass_mode: String,
    pub wall_ns: f64,
    pub busy_ns: f64,
    pub idle_ns: f64,
    pub passes_used: usize,
    pub queue: Vec<QueueItem>,
    /// Leaves with no floor (partial binding): ordered by share, never scored.
    pub unranked: Vec<QueueItem>,
    pub debt: Vec<Debt>,
}

fn shape_of(l: &Leaf) -> String {
    let mut parts: Vec<String> = l.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
    parts.extend(
        l.intervals
            .iter()
            .map(|(k, v)| format!("{k}∈[{},{}]", v.lo, v.hi)),
    );
    parts.join(" ")
}

fn weight_of(l: &Leaf) -> String {
    l.weight
        .iter()
        .flat_map(|w| w.members.iter().cloned())
        .collect::<Vec<_>>()
        .join(" + ")
}

/// Rank the leaves of `corpora` (`(label, path, corpus)`); one queue per
/// corpus plus the merged top-`top`.
pub fn queue(
    corpora: &[(String, String, Corpus)],
    top: usize,
) -> (Vec<QueueCorpus>, Vec<QueueItem>) {
    let mut out = Vec::new();
    let mut merged: Vec<QueueItem> = Vec::new();
    for (label, path, c) in corpora {
        let mk = |l: &Leaf| QueueItem {
            corpus: label.clone(),
            rank: 0,
            symbol: l.symbol.clone(),
            op_class: l.op_class.clone(),
            codec: l.codec.clone(),
            shape: shape_of(l),
            weight: weight_of(l),
            share: l.share,
            time_ns: l.time_ns,
            floor_ns: l.floors.floor_ns,
            efficiency: l.efficiency,
            score: match l.efficiency {
                Some(e) => l.share * (1.0 - e).max(0.0),
                None => 0.0,
            },
            status: l.status.clone(),
            binding: l.binding.clone(),
            floor_basis: l.floor_basis.clone(),
        };
        let (mut ranked, mut unranked): (Vec<QueueItem>, Vec<QueueItem>) = (Vec::new(), Vec::new());
        for l in &c.leaves {
            let it = mk(l);
            if l.efficiency.is_some() {
                ranked.push(it);
            } else {
                unranked.push(it);
            }
        }
        ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        unranked.sort_by(|a, b| {
            b.share
                .total_cmp(&a.share)
                .then_with(|| a.symbol.cmp(&b.symbol))
        });
        for (i, it) in ranked.iter_mut().enumerate() {
            it.rank = i + 1;
        }
        merged.extend(ranked.iter().cloned());
        out.push(QueueCorpus {
            label: label.clone(),
            path: path.clone(),
            arch: c.arch.clone(),
            tokens: c.tokens,
            pass_mode: c.pass_mode.clone(),
            wall_ns: c.wall_ns,
            busy_ns: c.busy_ns,
            idle_ns: c.idle_ns,
            passes_used: c.passes_used.len(),
            queue: ranked,
            unranked,
            debt: c.debt.clone(),
        });
    }
    merged.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    merged.truncate(top);
    for (i, it) in merged.iter_mut().enumerate() {
        it.rank = i + 1;
    }
    (out, merged)
}

fn md_cell(s: &str) -> String {
    s.replace('|', "\\|")
}

fn md_row(it: &QueueItem, with_corpus: bool) -> String {
    format!(
        "| {} |{} {:.4} | {:.2}% | {:.3} | {} | {} | {} | `{}` | {} | {} |",
        it.rank,
        if with_corpus {
            format!(" {} |", md_cell(&it.corpus))
        } else {
            String::new()
        },
        it.score,
        it.share * 100.0,
        it.time_ns / 1e6,
        it.floor_ns
            .map_or("-".to_string(), |f| format!("{:.3}", f / 1e6)),
        it.efficiency
            .map_or("-".to_string(), |e| format!("{:.1}%", e * 100.0)),
        md_cell(&it.op_class),
        md_cell(&it.symbol),
        md_cell(&it.shape),
        format!(
            "{}{}",
            md_cell(&it.status),
            if it.floor_basis.as_deref() == Some("interval_upper") {
                " (floor at interval upper bound)"
            } else {
                ""
            }
        ),
    )
}

/// The `NODE-QUEUE.md` text: top-`top` per corpus plus (for more than one
/// corpus) the merged top.
pub fn queue_markdown(per: &[QueueCorpus], merged: &[QueueItem], top: usize) -> String {
    let mut s = String::new();
    s.push_str("# NODE-QUEUE\n\n");
    s.push_str("Generated by `pm-census queue`. Rank = share × (1 − floor/measured): the fraction of the pass a node could give back if it reached its roofline floor (max of the arithmetic and memory floor at the arch peaks). A node at or below its floor scores 0. Leaves without a floor (partial binding) are listed unranked; nodes without a contract are debt.\n\n");
    let head = "| # | score | share | ms/pass | floor ms | eff | op | kernel | shape | status |\n|---:|---:|---:|---:|---:|---:|---|---|---|---|\n";
    let head_m = "| # | corpus | score | share | ms/pass | floor ms | eff | op | kernel | shape | status |\n|---:|---|---:|---:|---:|---:|---:|---|---|---|---|\n";
    if per.len() > 1 {
        s.push_str(&format!("## Merged top {top}\n\n{head_m}"));
        for it in merged {
            s.push_str(&md_row(it, true));
            s.push('\n');
        }
        s.push('\n');
    }
    for c in per {
        s.push_str(&format!(
            "## {} — {} pp{} ({}, {} pass{})\n\n",
            c.label,
            c.arch,
            c.tokens,
            c.pass_mode,
            c.passes_used,
            if c.passes_used == 1 { "" } else { "es" }
        ));
        s.push_str(&format!(
            "Pass wall {:.2} ms, busy {:.2} ms, idle {:.2} ms. Source: `{}`.\n\n",
            c.wall_ns / 1e6,
            c.busy_ns / 1e6,
            c.idle_ns / 1e6,
            c.path
        ));
        s.push_str(head);
        for it in c.queue.iter().take(top) {
            s.push_str(&md_row(it, false));
            s.push('\n');
        }
        if c.queue.len() > top {
            s.push_str(&format!(
                "\n{} more ranked leaves not shown.\n",
                c.queue.len() - top
            ));
        }
        if !c.unranked.is_empty() {
            s.push_str("\n**Unranked (no floor: partial binding)**\n\n| share | ms/pass | op | kernel | shape | status | binding |\n|---:|---:|---|---|---|---|---|\n");
            for it in c.unranked.iter().take(top) {
                s.push_str(&format!(
                    "| {:.2}% | {:.3} | {} | `{}` | {} | {} | {} |\n",
                    it.share * 100.0,
                    it.time_ns / 1e6,
                    md_cell(&it.op_class),
                    md_cell(&it.symbol),
                    md_cell(&it.shape),
                    md_cell(&it.status),
                    md_cell(&it.binding),
                ));
            }
        }
        if !c.debt.is_empty() {
            s.push_str("\n**Debt (no contract / unbound)**\n\n| share | ms/pass | calls | kernel | status | note |\n|---:|---:|---:|---|---|---|\n");
            for d in c.debt.iter().take(top.max(15)) {
                s.push_str(&format!(
                    "| {:.2}% | {:.3} | {:.0} | `{}` | {} | {} |\n",
                    d.share * 100.0,
                    d.time_ns / 1e6,
                    d.calls,
                    md_cell(&d.symbol),
                    md_cell(&d.status),
                    md_cell(&d.note),
                ));
            }
            if c.debt.len() > top.max(15) {
                s.push_str(&format!(
                    "\n{} more debt rows not shown.\n",
                    c.debt.len() - top.max(15)
                ));
            }
        }
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------------------
// CLI

const NODES_USAGE: &str = "\
pm-census extract nodes — one rocprofv3 trace + model -> node corpus (pm-census.nodes.v1)

  pm-census extract nodes --trace <rocprof json|kernel_trace.csv|dir> --config <config.json>
        [--hfq <model.hfq>] [--arch gfx1201] --tokens N [--pass mean|last]
        [--window START_NS:END_NS] [--agent ID] [--bench-profile <bench.json>] --out <corpus.json>

  --trace          rocprofv3 results (rocpd .db: convert first with scripts/rocpd_to_csv.py <db> <outdir>)
  --config         model config.json; omit to use the hfq metadata's \"config\" (needs --hfq)
  --hfq            .hfq tensor directory (header only is read)
  --arch           gfx target for the contract lookup and peaks (default: the trace agent's gfx name)
  --tokens         tokens per prefill pass
  --pass           mean (all passes averaged; default) | last
  --window         trace-time window in ns; required when the trace has no embedding anchor
  --agent          GPU agent id when the trace has several
  --bench-profile  bench JSON; records result.profile.sections[units == tokens].residual_ms next to the trace idle
  --out            corpus path; stdout gets a one-document summary";

const ROOFLINE_USAGE: &str = "\
pm-census roofline — per-token MAC/FLOP by op class and weight group, peak ceilings

  pm-census roofline <corpus.json> [--what-if <op_class|codec>=<pipe>,...]

  pipes: wmma_f16 wmma_bf16 wmma_fp8 wmma_iu8 wmma_iu4 valu_f32 memory (short: f16 bf16 fp8 iu8 iu4 valu)
  e.g.   --what-if q8_0=iu8,mq4g256v2=iu4";

const QUEUE_USAGE: &str = "\
pm-census queue — leaves ranked by share x (1 - floor/measured) across corpora

  pm-census queue <corpus.json>... [--top 15] [--md <NODE-QUEUE.md>]";

struct Flags {
    positional: Vec<String>,
    values: BTreeMap<String, String>,
}

fn parse_flags(args: &[String], allowed: &[&str], usage: &str) -> Result<Flags, String> {
    let mut positional = Vec::new();
    let mut values = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let Some(flag) = a.strip_prefix("--") else {
            positional.push(a.clone());
            continue;
        };
        let (name, inline) = match flag.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (flag, None),
        };
        if !allowed.contains(&name) {
            return Err(format!("unknown flag --{name}\n\n{usage}"));
        }
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .cloned()
                .ok_or_else(|| format!("--{name} needs a value"))?,
        };
        if values.insert(name.to_string(), value).is_some() {
            return Err(format!("--{name} given twice"));
        }
    }
    Ok(Flags { positional, values })
}

impl Flags {
    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn req(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("--{name} is required"))
    }

    fn num(&self, name: &str) -> Result<Option<u64>, String> {
        self.get(name)
            .map(|v| {
                v.parse::<u64>()
                    .map_err(|_| format!("--{name}: {v:?} is not an unsigned integer"))
            })
            .transpose()
    }
}

fn is_help(args: &[String]) -> bool {
    args.iter()
        .any(|a| matches!(a.as_str(), "--help" | "-h" | "help"))
}

fn parse_window(s: &str) -> Result<(u64, u64), String> {
    let (a, b) = s
        .split_once(':')
        .ok_or_else(|| format!("--window {s:?}: expected START_NS:END_NS"))?;
    let num = |x: &str| {
        x.parse::<u64>()
            .map_err(|_| format!("--window {s:?}: {x:?} is not an integer"))
    };
    let (a, b) = (num(a)?, num(b)?);
    if a >= b {
        return Err(format!("--window {s:?}: start must be below end"));
    }
    Ok((a, b))
}

fn read_corpus(path: &Path) -> Result<Corpus, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let c: Corpus = serde_json::from_str(&text)
        .map_err(|e| format!("{}: not a corpus: {e}", path.display()))?;
    if c.schema != SCHEMA {
        return Err(format!(
            "{}: schema {:?}, expected {SCHEMA:?}",
            path.display(),
            c.schema
        ));
    }
    Ok(c)
}

/// `pm-census extract nodes ...`
pub fn cli_nodes(args: &[String]) -> Result<(), String> {
    if is_help(args) {
        println!("{NODES_USAGE}");
        return Ok(());
    }
    let f = parse_flags(
        args,
        &[
            "trace",
            "config",
            "hfq",
            "arch",
            "tokens",
            "pass",
            "window",
            "agent",
            "bench-profile",
            "out",
        ],
        NODES_USAGE,
    )?;
    if !f.positional.is_empty() {
        return Err(format!(
            "unexpected argument {:?}\n\n{NODES_USAGE}",
            f.positional[0]
        ));
    }
    let tokens = f
        .num("tokens")?
        .ok_or_else(|| "--tokens is required".to_string())?;
    let pass = match f.get("pass").unwrap_or("mean") {
        "mean" => PassMode::Mean,
        "last" => PassMode::Last,
        other => return Err(format!("--pass {other:?}: expected mean or last")),
    };
    let window = f.get("window").map(parse_window).transpose()?;
    let hfq = f.get("hfq").map(PathBuf::from);
    let model = match (f.get("config"), &hfq) {
        (Some(c), h) => ModelDesc::load(Path::new(c), h.as_deref())?,
        (None, Some(h)) => ModelDesc::from_hfq_only(h)?,
        (None, None) => {
            return Err("--config (or --hfq with an embedded config) is required".to_string())
        }
    };
    let out = PathBuf::from(f.req("out")?);
    let trace = rocprof::load(Path::new(f.req("trace")?))?;
    let agent = f.num("agent")?;
    let arch = match f.get("arch") {
        Some(a) => a.to_string(),
        None => {
            let id = crate::census::resolve_agent(&trace, agent)?;
            trace
                .agent(id)
                .map(|a| a.name.clone())
                .filter(|n| n.starts_with("gfx"))
                .ok_or_else(|| "--arch is required (the trace agent has no gfx name)".to_string())?
        }
    };
    let opts = NodesOpts {
        arch,
        tokens,
        pass,
        window,
        agent,
    };
    let mut corpus = build_corpus(&trace, &model, &opts, &families_lookup)?;
    if let Some(b) = f.get("bench-profile") {
        corpus.bench_profile = Some(bench_profile_record(Path::new(b), tokens, corpus.idle_ns)?);
    }
    let text = serde_json::to_string_pretty(&corpus).map_err(|e| e.to_string())?;
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    let summary = json!({
        "schema": SCHEMA,
        "out": out.display().to_string(),
        "arch": corpus.arch,
        "tokens": corpus.tokens,
        "pass_mode": corpus.pass_mode,
        "passes_found": corpus.passes_found,
        "chunk_rows": corpus.chunk_rows,
        "chunks_per_pass": corpus.chunks_per_pass,
        "wall_ms": corpus.wall_ns / 1e6,
        "idle_ms": corpus.idle_ns / 1e6,
        "leaves": corpus.leaves.len(),
        "leaves_ok": corpus.leaves.iter().filter(|l| l.status == "ok").count(),
        "leaves_partial": corpus.leaves.iter().filter(|l| l.status == "partial").count(),
        "leaves_inconsistent": corpus.leaves.iter().filter(|l| l.status == "inconsistent").count(),
        "debt_rows": corpus.debt.len(),
        "debt_share": corpus.debt.iter().map(|d| d.share).sum::<f64>(),
        "ceilings": corpus.roofline["ceilings"].clone(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// `pm-census roofline <corpus.json> [--what-if ...]`
pub fn cli_roofline(args: &[String]) -> Result<(), String> {
    if is_help(args) {
        println!("{ROOFLINE_USAGE}");
        return Ok(());
    }
    let f = parse_flags(args, &["what-if"], ROOFLINE_USAGE)?;
    let [path] = f.positional.as_slice() else {
        return Err(format!(
            "expected exactly one corpus, got {}\n\n{ROOFLINE_USAGE}",
            f.positional.len()
        ));
    };
    let c = read_corpus(Path::new(path))?;
    let mut what_if: Vec<(String, Pipe)> = Vec::new();
    if let Some(w) = f.get("what-if") {
        for item in w.split(',').filter(|s| !s.is_empty()) {
            let (k, p) = item
                .split_once('=')
                .ok_or_else(|| format!("--what-if {item:?}: expected <op_class|codec>=<pipe>"))?;
            what_if.push((k.to_string(), parse_pipe(p)?));
        }
    }
    let hw =
        hw::peaks(&c.arch).ok_or_else(|| format!("no hardware peak table for arch {}", c.arch))?;
    let r = roofline(&c.leaves, c.tokens, c.wall_ns, c.busy_ns, &hw, &what_if)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// `pm-census queue <corpus.json>... [--top N] [--md PATH]`
pub fn cli_queue(args: &[String]) -> Result<(), String> {
    if is_help(args) {
        println!("{QUEUE_USAGE}");
        return Ok(());
    }
    let f = parse_flags(args, &["top", "md"], QUEUE_USAGE)?;
    if f.positional.is_empty() {
        return Err(format!("queue needs at least one corpus\n\n{QUEUE_USAGE}"));
    }
    let top = f.num("top")?.unwrap_or(15) as usize;
    if top == 0 {
        return Err("--top must be positive".to_string());
    }
    let mut corpora = Vec::new();
    for p in &f.positional {
        let path = Path::new(p);
        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(p)
            .to_string();
        corpora.push((label, p.clone(), read_corpus(path)?));
    }
    let (per, merged) = queue(&corpora, top);
    if let Some(md) = f.get("md") {
        std::fs::write(md, queue_markdown(&per, &merged, top)).map_err(|e| format!("{md}: {e}"))?;
    }
    let doc = json!({
        "schema": QUEUE_SCHEMA,
        "top": top,
        "merged": merged,
        "corpora": per.iter().map(|c| {
            let mut v = serde_json::to_value(c).unwrap_or(Value::Null);
            if let Some(q) = v["queue"].as_array_mut() { q.truncate(top); }
            if let Some(q) = v["unranked"].as_array_mut() { q.truncate(top); }
            v
        }).collect::<Vec<_>>(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{codec_name, HfqIndex, TensorInfo};
    use hipfire_kernel_abi::work::{BindV1, ALL_ROWS};
    use hipfire_kernel_abi::{
        Access, FormatId, NumericContractId, ObjectRefV1, OperandV1, ParamV1, ScalarTy,
    };

    // ---- synthetic contracts -------------------------------------------------

    fn leak<T>(x: T) -> &'static T {
        Box::leak(Box::new(x))
    }
    fn slice<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }
    fn p(n: &'static str) -> Expr {
        Expr::Param(n)
    }
    fn c(v: u64) -> Expr {
        Expr::Const(v)
    }
    fn mul(v: Vec<Expr>) -> Expr {
        Expr::Mul(slice(v))
    }
    fn add(v: Vec<Expr>) -> Expr {
        Expr::Add(slice(v))
    }
    fn dceil(e: Expr, d: u64) -> Expr {
        Expr::DivCeil(leak(e), d)
    }

    fn work(op: &'static str, codec: &'static str, pipe: Pipe) -> WorkV1 {
        WorkV1 {
            op_class: op,
            codec,
            pipe,
            macs: None,
            flops: None,
            bytes: None,
            rows: None,
            admit: ALL_ROWS,
            bind: &[],
            note: "",
        }
    }

    fn contract(
        id: &'static str,
        params: &[&'static str],
        operands: Vec<OperandV1>,
        grid: [Expr; 3],
        block: [u32; 3],
        w: Option<WorkV1>,
    ) -> Found {
        let exports = slice(vec![ExportV1 {
            symbol: id,
            arch: Arch::Gfx1201,
            object: ObjectRefV1::HipccSource {
                module: "t",
                source: "kernels/src/t.hip",
            },
            block,
            dynamic_lds: 0,
            group_segment_fixed: 0,
            grid,
            consts: &[],
        }]);
        let abi: &'static KernelAbiV1 = leak(KernelAbiV1 {
            id,
            version: 1,
            wave: 32,
            numeric: NumericContractId("t"),
            params: slice(
                params
                    .iter()
                    .map(|n| ParamV1 {
                        name: n,
                        ty: ScalarTy::U32,
                        doc: "",
                    })
                    .collect(),
            ),
            operands: slice(operands),
            fields: &[],
            hidden: &[],
            explicit_bytes: 0,
            segment_bytes: 0,
            segment_align: 8,
            domain: &[],
            exports,
        });
        let spec = w.map(|w| &*leak(WorkSpecV1 { abi, work: w }));
        (abi, &abi.exports[0], spec)
    }

    fn operand(name: &'static str, extent: Expr) -> OperandV1 {
        OperandV1 {
            name,
            format: FormatId("t"),
            access: Access::ReadOnly,
            align: 4,
            extent: Some(extent),
            nullable: false,
            note: "",
        }
    }

    const TOPK: &[&str] = &["num_experts_per_tok", "moe_topk"];

    fn table() -> Vec<(&'static str, Found)> {
        let emb = contract(
            "emb",
            &["rows"],
            vec![],
            [mul(vec![p("rows"), c(2)]), c(1), c(1)],
            [64, 1, 1],
            Some(WorkV1 {
                rows: Some("rows"),
                bytes: Some(mul(vec![p("rows"), c(8)])),
                ..work(op_class::EMBEDDING, "emb", Pipe::Memory)
            }),
        );
        // bytes = None: the sum of the operand extents.
        let gemm = contract(
            "gemm",
            &["M", "K", "N"],
            vec![
                operand("A", mul(vec![p("M"), p("K")])),
                operand("X", mul(vec![p("N"), p("K")])),
            ],
            [dceil(p("M"), 16), dceil(p("N"), 16), c(1)],
            [32, 1, 1],
            Some(WorkV1 {
                macs: Some(mul(vec![p("M"), p("N"), p("K")])),
                rows: Some("N"),
                bind: slice(vec![
                    BindV1 {
                        param: "N",
                        source: BindSource::ChunkRows,
                    },
                    BindV1 {
                        param: "K",
                        source: BindSource::Weight {
                            by: "M",
                            by_axis: WeightAxis::Rows,
                            codecs: &["q8_0"],
                            stacking: Stacking::Single,
                            fuse_siblings: false,
                        },
                    },
                ]),
                ..work(op_class::GEMM, "q8_0", Pipe::WmmaF16)
            }),
        );
        let attn = contract(
            "attn",
            &["Q", "P"],
            vec![],
            [dceil(p("Q"), 16), c(1), c(1)],
            [128, 1, 1],
            Some(WorkV1 {
                macs: Some(mul(vec![p("Q"), add(vec![p("P"), p("Q")])])),
                bytes: Some(mul(vec![p("Q"), c(4)])),
                rows: Some("Q"),
                bind: slice(vec![
                    BindV1 {
                        param: "Q",
                        source: BindSource::ChunkRows,
                    },
                    BindV1 {
                        param: "P",
                        source: BindSource::CausalPrefix,
                    },
                ]),
                ..work(op_class::ATTENTION, "f16", Pipe::WmmaF16)
            }),
        );
        let moe_work = |stacking: Stacking, fuse: bool| WorkV1 {
            macs: Some(mul(vec![p("rows"), p("M"), p("K")])),
            bytes: Some(mul(vec![p("stack"), p("M"), p("K")])),
            rows: Some("rows"),
            bind: slice(vec![
                BindV1 {
                    param: "K",
                    source: BindSource::Weight {
                        by: "M",
                        by_axis: WeightAxis::Rows,
                        codecs: &["mq4g256v2"],
                        stacking,
                        fuse_siblings: fuse,
                    },
                },
                BindV1 {
                    param: "rows",
                    source: BindSource::RoutedRows {
                        topk: TOPK,
                        dense: "m_total",
                    },
                },
            ]),
            ..work(op_class::GROUPED_GEMM, "mq4g256v2", Pipe::WmmaF16)
        };
        let moe_grid = || [dceil(p("M"), 16), dceil(p("m_total"), 16), c(1)];
        let moe = contract(
            "moe",
            &["M", "K", "m_total", "rows"],
            vec![],
            moe_grid(),
            [32, 1, 1],
            Some(moe_work(Stacking::Either, true)),
        );
        let moe_dense = contract(
            "moe_dense",
            &["M", "K", "m_total", "rows"],
            vec![],
            moe_grid(),
            [32, 1, 1],
            Some(moe_work(Stacking::Single, false)),
        );
        let nowork = contract(
            "nowork",
            &["n"],
            vec![],
            [p("n"), c(1), c(1)],
            [32, 1, 1],
            None,
        );
        // A memory op whose only param is a grid interval.
        let memop = contract(
            "memop",
            &["n"],
            vec![],
            [dceil(p("n"), 64), c(1), c(1)],
            [64, 1, 1],
            Some(WorkV1 {
                bytes: Some(mul(vec![p("n"), c(4)])),
                ..work(op_class::COPY, "f32", Pipe::Memory)
            }),
        );
        vec![
            ("emb", emb),
            ("gemm", gemm),
            ("attn", attn),
            ("moe", moe),
            ("moe_dense", moe_dense),
            ("nowork", nowork),
            ("memop", memop),
        ]
    }

    fn lookup_fn() -> impl Fn(&str, &str) -> Option<Found> {
        let t = table();
        move |_arch: &str, sym: &str| t.iter().find(|(n, _)| *n == sym).map(|(_, f)| *f)
    }

    // ---- synthetic trace -----------------------------------------------------

    #[derive(Default)]
    struct Tb {
        ids: BTreeMap<String, u64>,
        disp: Vec<rocprof::Dispatch>,
    }

    impl Tb {
        fn push(
            &mut self,
            name: &str,
            grid: [u32; 3],
            block: [u32; 3],
            lds: u32,
            start: u64,
            dur: u64,
        ) {
            let next = self.ids.len() as u64 + 1;
            let id = *self.ids.entry(name.to_string()).or_insert(next);
            self.disp.push(rocprof::Dispatch {
                kernel_id: id,
                agent: 1,
                queue: 1,
                correlation: self.disp.len() as u64 + 1,
                start_ns: start,
                end_ns: start + dur,
                workgroup: block,
                grid_items: [grid[0] * block[0], grid[1] * block[1], grid[2] * block[2]],
                group_segment_size: lds,
                private_segment_size: 0,
                graph_exec_id: 0,
                hip_api: None,
            });
        }

        fn finish(self) -> rocprof::Trace {
            rocprof::Trace {
                source: rocprof::TraceSource {
                    path: "synthetic".into(),
                    sha256: "0".repeat(64),
                    format: "rocprofv3-csv".into(),
                },
                agents: vec![rocprof::Agent {
                    id: 1,
                    kind: "gpu".into(),
                    name: "gfx1201".into(),
                    ordinal: Some(0),
                }],
                code_objects: vec![],
                kernels: self
                    .ids
                    .iter()
                    .map(|(n, id)| rocprof::KernelSymbol {
                        kernel_id: *id,
                        code_object_id: None,
                        name: n.clone(),
                        kernarg_segment_size: None,
                        group_segment_size: None,
                        private_segment_size: None,
                        sgpr_count: None,
                        vgpr_count: None,
                    })
                    .collect(),
                dispatches: self.disp,
                copies: vec![],
                markers: vec![],
            }
        }
    }

    /// 2 passes x 2 chunks of 64 rows (128 tokens). Pass 0 starts at 0, pass 1
    /// at 10_000; chunk c of a pass starts at base + 4_000 c.
    fn fixture(gemm_per_chunk: u64, gemm_dur: [u64; 2]) -> rocprof::Trace {
        let mut tb = Tb::default();
        for pass in 0..2u64 {
            for chunk in 0..2u64 {
                let t = pass * 10_000 + chunk * 4_000;
                tb.push("emb", [128, 1, 1], [64, 1, 1], 0, t, 10);
                for i in 0..gemm_per_chunk {
                    tb.push(
                        "gemm",
                        [3, 4, 1],
                        [32, 1, 1],
                        0,
                        t + 100 + i * 300,
                        gemm_dur[pass as usize],
                    );
                }
                tb.push("attn", [4, 1, 1], [128, 1, 1], 0, t + 1500, 300);
                tb.push("moe", [3, 12, 1], [32, 1, 1], 0, t + 2000, 500);
                tb.push("moe", [3, 12, 1], [32, 1, 1], 0, t + 2600, 500);
                tb.push("moe_dense", [2, 4, 1], [32, 1, 1], 0, t + 3200, 100);
                tb.push("moe_dense", [2, 4, 1], [32, 1, 1], 0, t + 3400, 100);
                tb.push("mystery", [7, 1, 1], [64, 1, 1], 0, t + 3700, 50);
                tb.push("nowork", [9, 1, 1], [32, 1, 1], 0, t + 3800, 20);
            }
        }
        tb.finish()
    }

    // ---- synthetic model -----------------------------------------------------

    fn tensor(name: &str, qt: u8, shape: &[u64]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            quant_type: qt,
            codec: codec_name(qt),
            shape: shape.to_vec(),
        }
    }

    fn tensors(with_k: bool, with_v: bool) -> Vec<TensorInfo> {
        let mut v = vec![tensor("model.embed_tokens.weight", 3, &[1000, 32])];
        for l in 0..3 {
            v.push(tensor(
                &format!("model.layers.{l}.self_attn.q_proj.weight"),
                3,
                &[48, 32],
            ));
        }
        if with_k {
            v.push(tensor(
                "model.layers.0.self_attn.k_proj.weight",
                3,
                &[48, 16],
            ));
        }
        if with_v {
            v.push(tensor(
                "model.layers.0.self_attn.v_proj.weight",
                3,
                &[48, 8],
            ));
        }
        for l in 0..2 {
            for e in 0..4 {
                for n in ["gate_proj", "up_proj"] {
                    v.push(tensor(
                        &format!("model.layers.{l}.mlp.experts.{e}.{n}.weight"),
                        44,
                        &[24, 64],
                    ));
                }
                v.push(tensor(
                    &format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                    44,
                    &[64, 24],
                ));
            }
            v.push(tensor(
                &format!("model.layers.{l}.shared_expert.gate_proj.weight"),
                44,
                &[24, 64],
            ));
        }
        v
    }

    fn model_of(t: Vec<TensorInfo>) -> ModelDesc {
        ModelDesc {
            config: json!({"num_experts_per_tok": 2}),
            config_sha256: "cfg".into(),
            hfq: Some(HfqIndex {
                path: "synthetic.hfq".into(),
                arch_id: 7,
                metadata: Value::Null,
                tensors: t,
                index_sha256: "idx".into(),
                data_offset: 0,
            }),
        }
    }

    fn opts(pass: PassMode) -> NodesOpts {
        NodesOpts {
            arch: "gfx1201".into(),
            tokens: 128,
            pass,
            window: None,
            agent: None,
        }
    }

    fn corpus(model: &ModelDesc, trace: &rocprof::Trace, pass: PassMode) -> Corpus {
        build_corpus(trace, model, &opts(pass), &lookup_fn()).unwrap()
    }

    fn leaves_of<'a>(c: &'a Corpus, sym: &str) -> Vec<&'a Leaf> {
        c.leaves.iter().filter(|l| l.symbol == sym).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * b.abs().max(1.0)
    }

    // ---- tests ---------------------------------------------------------------

    #[test]
    fn interval_inversion_through_divceil_div_add_and_mul() {
        let g = contract(
            "inv",
            &["M", "N", "A", "B"],
            vec![],
            [dceil(p("M"), 16), mul(vec![p("N"), c(4)]), c(1)],
            [32, 1, 1],
            None,
        );
        let (abi, ex, _) = g;
        let m = grid_intervals(abi, ex, [320, 8, 1]).unwrap();
        assert_eq!(m["M"], Iv { lo: 5105, hi: 5120 });
        assert_eq!(m["N"], Iv::point(2));
        // 7 is not a multiple of 4: the contract cannot produce it.
        assert!(grid_intervals(abi, ex, [1, 7, 1]).is_none());
        // Div, Add, Sub and DivExact bounds.
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &Expr::Div(leak(p("A")), 16),
            Iv::point(5),
            abi,
            ex,
            &mut out
        ));
        assert_eq!(out["A"], Iv { lo: 80, hi: 95 });
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &add(vec![p("B"), c(8)]),
            Iv { lo: 10, hi: 20 },
            abi,
            ex,
            &mut out
        ));
        assert_eq!(out["B"], Iv { lo: 2, hi: 12 });
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &Expr::Sub(leak(p("A")), leak(c(3))),
            Iv::point(4),
            abi,
            ex,
            &mut out
        ));
        assert_eq!(out["A"], Iv::point(7));
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &Expr::DivExact(leak(p("A")), 4),
            Iv { lo: 2, hi: 3 },
            abi,
            ex,
            &mut out
        ));
        assert_eq!(out["A"], Iv { lo: 8, hi: 12 });
        // Two unknowns constrain nothing.
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &mul(vec![p("A"), p("B")]),
            Iv::point(6),
            abi,
            ex,
            &mut out
        ));
        assert!(out.is_empty());
        // ceil(a/16) = 0 => a in [0, 0].
        let mut out = BTreeMap::new();
        assert!(invert_interval(
            &dceil(p("A"), 16),
            Iv::point(0),
            abi,
            ex,
            &mut out
        ));
        assert_eq!(out["A"], Iv::point(0));
    }

    #[test]
    fn passes_are_anchored_by_embedding_and_mean_averages_them() {
        let m = model_of(tensors(false, false));
        let tr = fixture(3, [100, 200]);
        let mean = corpus(&m, &tr, PassMode::Mean);
        assert_eq!(mean.schema, SCHEMA);
        assert_eq!(
            (mean.passes_found, mean.chunk_rows, mean.chunks_per_pass),
            (2, 64, 2)
        );
        assert_eq!(mean.passes_used, vec![0, 1]);
        assert_eq!(mean.passes[0].start_ns, 0);
        assert_eq!(mean.passes[0].wall_ns, 10_000);
        assert_eq!(mean.passes[1].start_ns, 10_000);
        assert_eq!(mean.passes[1].wall_ns, 7_820);
        assert!(close(mean.wall_ns, 8_910.0));
        assert!(close(mean.busy_ns + mean.idle_ns, mean.wall_ns));
        assert!(mean.idle_ns > 0.0);
        assert_eq!(mean.outside_pass_dispatches, 0);
        let g = leaves_of(&mean, "gemm");
        assert_eq!(g.len(), 1);
        // 3 calls x 2 chunks per pass; (100 + 200) / 2 ns each.
        assert!(close(g[0].calls, 6.0));
        assert!(close(g[0].time_ns, 6.0 * 150.0));
        assert!(close(g[0].share, 900.0 / mean.wall_ns));

        let last = corpus(&m, &tr, PassMode::Last);
        assert_eq!(last.passes_used, vec![1]);
        assert!(close(last.wall_ns, 7_820.0));
        assert!(close(leaves_of(&last, "gemm")[0].time_ns, 6.0 * 200.0));
    }

    #[test]
    fn weight_set_selected_by_calls_per_chunk_single_set() {
        let m = model_of(tensors(true, false));
        // 3 calls per chunk: q_proj alone (3 layers) is the only set that sums to 3.
        let c3 = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        let g = leaves_of(&c3, "gemm");
        assert_eq!(g.len(), 1);
        let l = g[0];
        assert_eq!(l.params["M"], 48);
        assert_eq!(l.params["K"], 32);
        assert_eq!(l.params["N"], 64);
        assert_eq!(l.status, "ok");
        assert_eq!(l.binding, "ok");
        assert_eq!(l.weight.len(), 1);
        assert_eq!(
            l.weight[0].members,
            vec!["model.layers.*.self_attn.q_proj.weight".to_string()]
        );
        assert!(close(l.macs.unwrap(), 6.0 * 48.0 * 64.0 * 32.0));
        // bytes = sum of operand extents = M*K + N*K per call.
        assert!(close(l.bytes.unwrap(), 6.0 * (48.0 * 32.0 + 64.0 * 32.0)));
        let hw = hw::peaks("gfx1201").unwrap();
        assert!(close(
            l.floors.arith_ns.unwrap(),
            2.0 * l.macs.unwrap() / hw::peak(&hw, Pipe::WmmaF16) * 1e9
        ));
        assert!(l.efficiency.unwrap() > 0.0 && l.efficiency.unwrap() < 1.0);
        assert_eq!(l.row_class.rows, Some(64));
        assert_eq!(l.row_class.admitted, Some(true));
    }

    #[test]
    fn distinct_k_values_split_into_leaves_with_calls_by_tensor_count() {
        let m = model_of(tensors(true, false));
        // 4 calls per chunk = q_proj (3 layers) + k_proj (1): the only set.
        let c4 = corpus(&m, &fixture(4, [100, 100]), PassMode::Mean);
        let mut g = leaves_of(&c4, "gemm");
        g.sort_by_key(|l| l.params["K"]);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].params["K"], g[1].params["K"]), (16, 32));
        assert!(close(g[0].calls, 2.0));
        assert!(close(g[1].calls, 6.0));
        assert!(g.iter().all(|l| l.status == "ok" && l.binding == "ok"));
        // Time (8 calls x 100 ns) is split by MACs: 2*16 : 6*32.
        assert!(g.iter().all(|l| l.time_basis == "apportioned_by_macs"));
        assert!(close(g[0].time_ns + g[1].time_ns, 800.0));
        assert!(close(g[1].time_ns / g[0].time_ns, 6.0));
    }

    #[test]
    fn ambiguous_or_missing_set_takes_all_matches_and_says_inconsistent() {
        // k_proj and v_proj both launch once: {q,k} and {q,v} both sum to 4.
        let m = model_of(tensors(true, true));
        let amb = corpus(&m, &fixture(4, [100, 100]), PassMode::Mean);
        let g = leaves_of(&amb, "gemm");
        assert_eq!(g.len(), 3);
        assert!(g
            .iter()
            .all(|l| l.binding.starts_with("inconsistent") && l.status == "inconsistent"));
        // 5 calls per chunk fits no set of {q:3} alone.
        let m = model_of(tensors(false, false));
        let none = corpus(&m, &fixture(5, [100, 100]), PassMode::Mean);
        let g = leaves_of(&none, "gemm");
        assert_eq!(g.len(), 1);
        // One candidate: its value binds even though 5 calls/chunk match no launch count.
        assert_eq!(g[0].status, "ok");
        assert!(
            g[0].binding.starts_with("unique value across 1 candidates"),
            "{}",
            g[0].binding
        );
        assert_eq!(g[0].params["K"], 32);
        // Three candidate sets, all with different values and none fitting: inconsistent.
        let m3 = model_of(tensors(true, true));
        let none3 = corpus(&m3, &fixture(7, [100, 100]), PassMode::Mean);
        assert!(leaves_of(&none3, "gemm")
            .iter()
            .all(|l| l.binding.starts_with("inconsistent")));
        // No hfq at all: K stays unbound, the leaf is partial with its interval.
        let bare = ModelDesc {
            config: json!({"num_experts_per_tok": 2}),
            config_sha256: "c".into(),
            hfq: None,
        };
        let c = corpus(&bare, &fixture(3, [100, 100]), PassMode::Mean);
        let g = leaves_of(&c, "gemm");
        assert_eq!(g[0].status, "partial");
        assert_eq!(g[0].unbound, vec!["K".to_string(), "M".to_string()]);
        assert_eq!(g[0].intervals["M"], Iv { lo: 33, hi: 48 });
        assert!(g[0].macs.is_none());
        assert!(g[0].floors.floor_ns.is_none());
    }

    #[test]
    fn routed_rows_use_topk_for_stacks_and_the_dense_interval_otherwise() {
        let m = model_of(tensors(false, false));
        let c = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        // Stacked fused gate+up (48 rows, 64 cols): 2 layers per chunk.
        let g = leaves_of(&c, "moe");
        assert_eq!(g.len(), 1);
        let l = g[0];
        assert_eq!((l.params["M"], l.params["K"]), (48, 64));
        assert_eq!(l.params["rows"], 64 * 2, "chunk rows x topk");
        assert_eq!(l.params["stack"], 4, "experts per stack");
        assert!(l.weight[0].stacked && l.weight[0].fused);
        assert_eq!(l.status, "ok");
        assert!(close(l.macs.unwrap(), 4.0 * 128.0 * 48.0 * 64.0));
        assert!(close(l.bytes.unwrap(), 4.0 * 4.0 * 48.0 * 64.0));
        // Unstacked shared gate: dense m_total in [49, 64] contains the chunk rows.
        let d = leaves_of(&c, "moe_dense");
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].params["M"], d[0].params["K"]), (24, 64));
        assert_eq!(d[0].params["rows"], 64);
        assert_eq!(d[0].params["stack"], 1);
        assert!(!d[0].weight[0].stacked);
        assert!(close(d[0].macs.unwrap(), 4.0 * 64.0 * 24.0 * 64.0));
    }

    #[test]
    fn causal_prefix_is_evaluated_per_call() {
        let m = model_of(tensors(false, false));
        let c = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        let a = leaves_of(&c, "attn");
        assert_eq!(a.len(), 1);
        assert!(a[0].causal);
        assert!(!a[0].params.contains_key("P"));
        // chunk 0: 64*(0+64); chunk 1: 64*(64+64).
        assert!(close(a[0].macs.unwrap(), 4096.0 + 8192.0));
        assert!(close(a[0].calls, 2.0));
        let last = corpus(&m, &fixture(3, [100, 100]), PassMode::Last);
        assert!(close(
            leaves_of(&last, "attn")[0].macs.unwrap(),
            4096.0 + 8192.0
        ));
    }

    #[test]
    fn symbols_without_contract_or_work_are_debt_rows_with_time_kept() {
        let m = model_of(tensors(false, false));
        let c = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        let by = |s: &str| c.debt.iter().find(|d| d.symbol == s).unwrap();
        let my = by("mystery");
        assert_eq!(my.status, "no_contract");
        assert!(close(my.calls, 2.0));
        assert!(close(my.time_ns, 100.0));
        assert!(close(my.share, 100.0 / c.wall_ns));
        assert_eq!(by("nowork").status, "no_work_spec");
        assert_eq!(by("nowork").abi_id.as_deref(), Some("nowork"));
        assert!(c
            .leaves
            .iter()
            .all(|l| l.symbol != "mystery" && l.symbol != "nowork"));
        // Debt shows in the class totals; nothing is dropped.
        assert_eq!(c.totals_by_op_class["(debt)"].leaves, 2);
        let n: f64 = c.leaves.iter().map(|l| l.calls).sum::<f64>()
            + c.debt.iter().map(|d| d.calls).sum::<f64>();
        assert!(close(n, c.launches_per_pass));
    }

    #[test]
    fn no_anchor_needs_a_window() {
        let m = model_of(tensors(false, false));
        let mut tb = Tb::default();
        for i in 0..3 {
            tb.push("gemm", [3, 4, 1], [32, 1, 1], 0, 100 + i * 300, 100);
        }
        let tr = tb.finish();
        let e = build_corpus(&tr, &m, &opts(PassMode::Mean), &lookup_fn()).unwrap_err();
        assert!(
            e.contains("no embedding anchor") && e.contains("--window"),
            "{e}"
        );
        let mut o = opts(PassMode::Mean);
        o.window = Some((0, 2_000));
        let c = build_corpus(&tr, &m, &o, &lookup_fn()).unwrap();
        assert_eq!(c.passes_found, 1);
        assert_eq!(c.passes[0].wall_ns, 2_000);
        assert_eq!(c.leaves.len(), 1);
        assert!(c.notes.iter().any(|n| n.contains("no embedding anchor")));
    }

    #[test]
    fn bench_profile_residual_is_recorded_next_to_trace_idle() {
        let dir = tmp("bench");
        let f = dir.join("bench.json");
        std::fs::write(
            &f,
            r#"{"result":{"profile":{"sections":[
                {"section":"pp4096","units":4096.0,"residual_ms":107.2},
                {"section":"pp128","units":128.0,"residual_ms":1.5,"wall_profiled_ms":9.0}]}}}"#,
        )
        .unwrap();
        let r = bench_profile_record(&f, 128, 2_500_000.0).unwrap();
        assert_eq!(r["section"], "pp128");
        assert_eq!(r["residual_ms"], 1.5);
        assert_eq!(r["trace_idle_ms"], 2.5);
        assert_eq!(r["wall_profiled_ms"], 9.0);
        assert!(bench_profile_record(&f, 999, 0.0)
            .unwrap_err()
            .contains("units == 999"));
    }

    #[test]
    fn roofline_per_token_and_what_if_recharge() {
        let m = model_of(tensors(false, false));
        let c = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        let hw = hw::peaks("gfx1201").unwrap();
        let r = roofline(&c.leaves, c.tokens, c.wall_ns, c.busy_ns, &hw, &[]).unwrap();
        let gemm = &r["by_op_class"]["gemm"];
        assert!(close(
            gemm["macs_per_token"].as_f64().unwrap(),
            6.0 * 48.0 * 64.0 * 32.0 / 128.0
        ));
        assert!(close(
            gemm["arith_ns"].as_f64().unwrap(),
            2.0 * 6.0 * 48.0 * 64.0 * 32.0 / hw.wmma_f16 * 1e9
        ));
        assert!(r["ceilings"]["peak_arith_tok_s"].as_f64().unwrap() > 0.0);
        assert!(r["by_weight_group"].as_array().unwrap().len() >= 2);
        let w = roofline(
            &c.leaves,
            c.tokens,
            c.wall_ns,
            c.busy_ns,
            &hw,
            &[("q8_0".into(), Pipe::WmmaIu8)],
        )
        .unwrap();
        assert!(close(
            w["by_op_class"]["gemm"]["arith_ns"].as_f64().unwrap(),
            2.0 * 6.0 * 48.0 * 64.0 * 32.0 / hw.wmma_iu8 * 1e9
        ));
        // The attention leaf (codec f16) is not touched.
        assert_eq!(w["by_op_class"]["attention"], r["by_op_class"]["attention"]);
        assert_eq!(w["what_if_charged"].as_array().unwrap().len(), 1);
        assert!(parse_pipe("nope").is_err());
    }

    #[test]
    fn queue_ranks_by_share_times_headroom_and_writes_markdown() {
        let m = model_of(tensors(true, false));
        let c = corpus(&m, &fixture(4, [100, 100]), PassMode::Mean);
        let (per, merged) = queue(&[("raon".into(), "raon.json".into(), c.clone())], 3);
        assert_eq!(per.len(), 1);
        let q = &per[0].queue;
        assert!(q.windows(2).all(|w| w[0].score >= w[1].score));
        assert_eq!(q[0].rank, 1);
        for it in q {
            let l = c
                .leaves
                .iter()
                .find(|l| l.symbol == it.symbol && shape_of(l) == it.shape)
                .unwrap();
            assert!(close(
                it.score,
                l.share * (1.0 - l.efficiency.unwrap()).max(0.0)
            ));
        }
        assert!(merged.len() <= 3);
        let md = queue_markdown(&per, &merged, 3);
        assert!(md.starts_with("# NODE-QUEUE"));
        assert!(md.contains("## raon"));
        assert!(md.contains("`mystery`") && md.contains("no_contract"));
        // Leaves without a floor are listed unranked, never scored.
        let bare = ModelDesc {
            config: json!({}),
            config_sha256: "c".into(),
            hfq: None,
        };
        let cb = corpus(&bare, &fixture(3, [100, 100]), PassMode::Mean);
        let (per, _) = queue(&[("bare".into(), "b.json".into(), cb)], 5);
        assert!(per[0].unranked.iter().any(|u| u.symbol == "gemm"));
        assert!(per[0].queue.iter().all(|u| u.symbol != "gemm"));
    }

    // ---- file-based pipeline -------------------------------------------------

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pm-census-ingest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_csv(tr: &rocprof::Trace, path: &Path) {
        let mut s = String::from("Agent_Id,Queue_Id,Kernel_Id,Kernel_Name,Correlation_Id,Start_Timestamp,End_Timestamp,LDS_Block_Size,Scratch_Size,VGPR_Count,SGPR_Count,Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z,Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n");
        for d in &tr.dispatches {
            let name = &tr.kernel(d.kernel_id).unwrap().name;
            s.push_str(&format!(
                "Agent {},{},{},{},{},{},{},{},0,32,32,{},{},{},{},{},{}\n",
                d.agent,
                d.queue,
                d.kernel_id,
                name,
                d.correlation,
                d.start_ns,
                d.end_ns,
                d.group_segment_size,
                d.workgroup[0],
                d.workgroup[1],
                d.workgroup[2],
                d.grid_items[0],
                d.grid_items[1],
                d.grid_items[2],
            ));
        }
        std::fs::write(path, s).unwrap();
    }

    /// A standalone hfq header region (no payload needed: only the header is read).
    fn hfq_bytes(metadata: &str, tensors: &[TensorInfo]) -> Vec<u8> {
        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for t in tensors {
            index.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
            index.extend_from_slice(t.name.as_bytes());
            index.push(t.quant_type);
            index.push(t.shape.len() as u8);
            for d in &t.shape {
                index.extend_from_slice(&(*d as u32).to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes());
            index.extend_from_slice(&16u64.to_le_bytes());
        }
        let data_offset = 32 + metadata.len() + index.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"HFQM");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        out.extend_from_slice(&32u64.to_le_bytes());
        out.extend_from_slice(&(data_offset as u64).to_le_bytes());
        out.extend_from_slice(metadata.as_bytes());
        out.extend_from_slice(&index);
        out.resize(data_offset + 16 * tensors.len(), 0);
        out
    }

    #[test]
    fn csv_trace_and_hfq_header_bind_like_the_in_memory_fixture() {
        let dir = tmp("csv");
        let mem = fixture(4, [100, 200]);
        write_csv(&mem, &dir.join("kernel_trace.csv"));
        let ts = tensors(true, false);
        std::fs::write(dir.join("m.hfq"), hfq_bytes("{}", &ts)).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"text_config":{"num_experts_per_tok":2}}"#,
        )
        .unwrap();
        let tr = rocprof::load(&dir.join("kernel_trace.csv")).unwrap();
        assert_eq!(tr.dispatches.len(), mem.dispatches.len());
        let model = ModelDesc::load(&dir.join("config.json"), Some(&dir.join("m.hfq"))).unwrap();
        assert_eq!(model.hfq.as_ref().unwrap().tensors.len(), ts.len());
        let from_files = build_corpus(&tr, &model, &opts(PassMode::Mean), &lookup_fn()).unwrap();
        let from_mem = corpus(&model_of(ts), &mem, PassMode::Mean);
        assert_eq!(from_files.leaves.len(), from_mem.leaves.len());
        assert_eq!(from_files.trace.format, "rocprofv3-csv");
        assert_eq!(from_files.config_sha256.len(), 64);
        assert_eq!(from_files.hfq.as_ref().unwrap().index_sha256.len(), 64);
        for (a, b) in from_files.leaves.iter().zip(&from_mem.leaves) {
            assert_eq!(
                (a.symbol.as_str(), &a.params, a.status.as_str()),
                (b.symbol.as_str(), &b.params, b.status.as_str())
            );
            assert!(close(a.time_ns, b.time_ns));
            assert_eq!(a.macs.map(|m| m.round()), b.macs.map(|m| m.round()));
        }
        // The corpus survives a JSON round trip (roofline/queue read it back).
        let path = dir.join("corpus.json");
        std::fs::write(&path, serde_json::to_string_pretty(&from_files).unwrap()).unwrap();
        let back = read_corpus(&path).unwrap();
        assert_eq!(back.leaves.len(), from_files.leaves.len());
        let (a, b) = (
            &back.roofline["per_token"],
            &from_files.roofline["per_token"],
        );
        assert!(close(
            a["macs"].as_f64().unwrap(),
            b["macs"].as_f64().unwrap()
        ));
        assert_eq!(
            back.roofline["by_op_class"].as_object().unwrap().len(),
            from_files.roofline["by_op_class"]
                .as_object()
                .unwrap()
                .len()
        );
    }

    #[test]
    fn cli_flags_parse() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(cli_nodes(&s(&["--help"])).is_ok());
        assert!(cli_roofline(&s(&["--help"])).is_ok());
        assert!(cli_queue(&s(&["--help"])).is_ok());
        assert!(cli_nodes(&s(&["--bogus", "1"]))
            .unwrap_err()
            .contains("unknown flag --bogus"));
        assert!(cli_nodes(&s(&["--trace", "x", "--out", "y"]))
            .unwrap_err()
            .contains("--tokens"));
        assert!(cli_roofline(&s(&["a.json", "--what-if", "q8_0"])).is_err());
        assert!(cli_queue(&s(&[]))
            .unwrap_err()
            .contains("at least one corpus"));
        assert_eq!(parse_window("5:9").unwrap(), (5, 9));
        assert!(parse_window("9:5").is_err());
    }

    fn anchor_chunks(tb: &mut Tb) {
        for pass in 0..2u64 {
            for chunk in 0..2u64 {
                tb.push(
                    "emb",
                    [128, 1, 1],
                    [64, 1, 1],
                    0,
                    pass * 10_000 + chunk * 4_000,
                    10,
                );
            }
        }
    }

    #[test]
    fn either_stacking_follows_the_dense_row_interval() {
        let m = model_of(tensors(false, false));
        // y = 12 => m_total in [177, 192] > 64 chunk rows: only the stack (gate+up fused, 48 rows).
        // y = 4 => m_total in [49, 64] <= 64: only single weights. M in [17, 32] has both a
        // stacked expert gate and the shared gate (24 rows, 2 launches each): the filter picks
        // the single one instead of reporting two fitting sets.
        let mut tb = Tb::default();
        anchor_chunks(&mut tb);
        for pass in 0..2u64 {
            for chunk in 0..2u64 {
                let t = pass * 10_000 + chunk * 4_000;
                tb.push("moe", [2, 4, 1], [32, 1, 1], 0, t + 100, 100);
                tb.push("moe", [2, 4, 1], [32, 1, 1], 0, t + 300, 100);
                tb.push("moe", [3, 12, 1], [32, 1, 1], 0, t + 500, 100);
                tb.push("moe", [3, 12, 1], [32, 1, 1], 0, t + 700, 100);
            }
        }
        let c = corpus(&m, &tb.finish(), PassMode::Mean);
        let mut g = leaves_of(&c, "moe");
        g.sort_by_key(|l| l.params["M"]);
        assert_eq!(g.len(), 2);
        assert_eq!(
            (g[0].params["M"], g[0].params["rows"], g[0].params["stack"]),
            (24, 64, 1)
        );
        assert!(!g[0].weight[0].stacked);
        assert_eq!(g[0].binding, "ok");
        assert_eq!(
            (g[1].params["M"], g[1].params["rows"], g[1].params["stack"]),
            (48, 128, 4)
        );
        assert!(g[1].weight[0].stacked);
        assert!(g.iter().all(|l| l.status == "ok"));
    }

    #[test]
    fn per_pass_ops_match_launches_per_pass() {
        // One gemm call per pass: k_proj (1 launch) is the set; q_proj (3) is not.
        let m = model_of(tensors(true, false));
        let mut tb = Tb::default();
        anchor_chunks(&mut tb);
        for pass in 0..2u64 {
            tb.push("gemm", [3, 4, 1], [32, 1, 1], 0, pass * 10_000 + 100, 100);
        }
        let c = corpus(&m, &tb.finish(), PassMode::Mean);
        let g = leaves_of(&c, "gemm");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].params["K"], 16);
        assert_eq!(g[0].status, "ok");
        assert_eq!(g[0].binding, "ok");
    }

    #[test]
    fn unique_value_binds_without_a_count_match() {
        // q_proj [48,32] in 3 layers and a second pattern with the same shape: 4 launches
        // per chunk match nothing, but every candidate gives K = 32.
        let mut t = tensors(false, false);
        t.push(tensor(
            "model.layers.0.self_attn.o_proj.weight",
            3,
            &[48, 32],
        ));
        let m = model_of(t);
        let c = corpus(&m, &fixture(7, [100, 100]), PassMode::Mean);
        let g = leaves_of(&c, "gemm");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].params["K"], 32);
        assert_eq!(g[0].status, "ok");
        assert!(
            g[0].binding.starts_with("unique value across 2 candidates"),
            "{}",
            g[0].binding
        );
    }

    #[test]
    fn interval_only_unknowns_give_an_upper_bound_floor_and_stay_partial() {
        let m = model_of(tensors(false, false));
        let mut tb = Tb::default();
        anchor_chunks(&mut tb);
        for pass in 0..2u64 {
            for chunk in 0..2u64 {
                tb.push(
                    "memop",
                    [8, 1, 1],
                    [64, 1, 1],
                    0,
                    pass * 10_000 + chunk * 4_000 + 100,
                    50,
                );
            }
        }
        let c = corpus(&m, &tb.finish(), PassMode::Mean);
        let l = &leaves_of(&c, "memop")[0];
        assert_eq!(l.status, "partial");
        assert_eq!(l.intervals["n"], Iv { lo: 449, hi: 512 });
        assert_eq!(l.floor_basis.as_deref(), Some("interval_upper"));
        assert!(close(l.bytes.unwrap(), 2.0 * 512.0 * 4.0));
        assert!(l.floors.floor_ns.is_some());
        let (per, _) = queue(&[("m".into(), "m.json".into(), c.clone())], 5);
        assert!(per[0]
            .queue
            .iter()
            .any(|q| q.symbol == "memop" && q.floor_basis.as_deref() == Some("interval_upper")));
        assert!(queue_markdown(&per, &[], 5).contains("floor at interval upper bound"));
    }

    #[test]
    fn all_f16_ceiling_charges_every_mac_to_wmma_f16() {
        let m = model_of(tensors(false, false));
        let c = corpus(&m, &fixture(3, [100, 100]), PassMode::Mean);
        let hw = hw::peaks("gfx1201").unwrap();
        let r = roofline(
            &c.leaves,
            c.tokens,
            c.wall_ns,
            c.busy_ns,
            &hw,
            &[("q8_0".into(), Pipe::WmmaIu8)],
        )
        .unwrap();
        let macs: f64 = c.leaves.iter().filter_map(|l| l.macs).sum();
        let flops: f64 = c.leaves.iter().filter_map(|l| l.flops).sum();
        let ns = (2.0 * macs / hw.wmma_f16 + flops / hw.valu_f32) * 1e9;
        // Independent of what-if: the all-F16 route is a fixed definition.
        assert!(close(
            r["ceilings"]["all_f16_peak_tok_s"].as_f64().unwrap(),
            c.tokens as f64 / (ns * 1e-9)
        ));
        assert!(c.notes.iter().any(|n| n.contains("all_f16_peak_tok_s")));
    }
}
