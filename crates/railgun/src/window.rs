// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed shape binding for the certified DFlash window (design A, slice A1:
//! `docs/design/2026-10-25-dflash-railgun-window.md` §2 "Frozen implementation interfaces" and §3
//! "Shapes: finite topology, explicit patch expressions").
//!
//! A [`WindowPlan`] is one railgun [`KernelProgram`] plus the *declared*
//! per-window changes (§3: [`PatchKind::Scalar32`], [`PatchKind::Interior64`],
//! [`PatchKind::GridAxis`], and the existing `GdnFrame` word role), typed
//! copy/fill steps, the read/write allocation closure, the shape-family key,
//! the proposed host bookkeeping delta and the completion surfaces. A
//! [`PreparedWindow`] binds a plan to one [`WindowBindingKey`]: [`bind`]
//! validates *every* patch (overflow, domain, `u32` fit, allocation extent,
//! alias, generation) without touching the queue, then re-encodes each
//! kernarg image in retired storage through
//! [`KernelProgram::encode_kernarg_into`] — the only full-image encoder.
//! [`submit`] consumes the bound generation; [`Submission::complete`] proves
//! queue quiescence. There is no in-flight patching: a prepared window is
//! only writable while retired (§2: "No in-flight patching").
//!
//! The GPU-facing half (a queue that writes kernarg buffers and launches the
//! IB) is the [`WindowQueue`] trait; `rdna-compute` implements it next to the
//! extracted owned lowerer. Nothing here depends on Redline or HIP.
//!
//! [`bind`]: PreparedWindow::bind
//! [`submit`]: PreparedWindow::submit

use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::index_bind::{
    abi as writer_abi, BindInputs, IndexBindError, IndexedBindWitness, SealedPositions,
    WriterExtent,
};
use crate::kernel::{Arg, Expr, ExprError, ImageOverrides, KernelProgram, PatchKind, WordRole};
use crate::plan::Pm4Plan;
use crate::sealed_table::{
    SealedTableError, SealedTableRegistry, SealedTableWitness, TableBindInputs, TableContent,
    TableLease,
};
use crate::{Binding, ResourceId, WordId};

pub use crate::watchdog::{
    run_watchdog, FailureKind, PersistentWatchdog, PoisonCell, ResolvedWatchdog, Teardown,
};

/// Prepared topology skeletons retained per owner (§2: "one bounded cache of
/// **32 prepared topology skeletons** per owner").
pub const WINDOW_CACHE_CAPACITY: usize = 32;

/// Why a window plan or bind was refused. A refusal is always raised before
/// the queue is touched (§6 "Refusal before submission: ordinary HIP route,
/// no changes").
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("malformed window plan: {0}")]
    Malformed(String),
    #[error("node {node} ({symbol}) is not PM4-lowered: {reason}")]
    NotPm4 {
        node: usize,
        symbol: String,
        reason: String,
    },
    #[error("{site}: unknown window input {name:?}")]
    UnknownInput { site: String, name: String },
    #[error("{site}: checked arithmetic overflow")]
    Overflow { site: String },
    #[error("{site}: value {value} lies outside its domain {start}..={end}")]
    OutOfDomain {
        site: String,
        value: u64,
        start: u64,
        end: u64,
    },
    #[error("{site}: value {value} does not fit u32")]
    DoesNotFit { site: String, value: u64 },
    #[error("{site}: bytes [{offset}, {end}) exceed resource {resource} ({bytes} bytes)")]
    ExtentExceeded {
        site: String,
        resource: u32,
        offset: u64,
        end: u64,
        bytes: u64,
    },
    #[error("{site}: {detail}")]
    BadStep { site: String, detail: String },
    #[error("resources {a} and {b} overlap and one is written")]
    Alias { a: u32, b: u32 },
    #[error("bad allocation binding for resource {resource}: {detail}")]
    BadBinding { resource: u32, detail: String },
    #[error("window binding key changed: {0}")]
    BindingKeyMismatch(String),
    #[error("stale generation: window generation {got} is not after the last submitted {last}")]
    StaleGeneration { got: u64, last: u64 },
    #[error("bound window is stale (superseded, foreign or already submitted)")]
    StaleBound,
    #[error("the prepared window is in flight: patching is refused")]
    InFlight,
    #[error("the prepared window is quarantined (unknown-quiescence failure, or its queue was torn down)")]
    Quarantined,
    #[error("invalid window shape: {0}")]
    Shape(String),
    #[error("invalid window inputs: {0}")]
    Inputs(String),
    #[error("kernarg encode failed: {0}")]
    Encode(String),
    #[error("submit failed ({quiescence:?}, {kind:?}): {message}")]
    SubmitFailed {
        message: String,
        quiescence: Quiescence,
        kind: FailureKind,
    },
    #[error("all 32 prepared skeletons are in flight or quarantined")]
    CacheFull,
    #[error("indexed bind: {0}")]
    Indexed(IndexBindError),
    #[error("node {node} ({symbol}) is conditional: submit needs an indexed bind witness")]
    IndexedWitnessMissing { node: usize, symbol: String },
    #[error("sealed table: {0}")]
    SealedTable(SealedTableError),
    #[error("node {node} ({symbol}) is conditional: submit needs a sealed_table witness")]
    SealedTableWitnessMissing { node: usize, symbol: String },
}

/// What is known about the queue after a failure (§2: "failure distinguishes
/// proven-quiescent from unknown/in-flight").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quiescence {
    Proven,
    Unknown,
}

// ---------------------------------------------------------------- shapes

/// `WindowShape` (§2): the checked-`u64` logical geometry of one window. The
/// kernel-facing `i32`/`u32` conversions are separate checked steps in the
/// patch sites. `draft_rows = verify_rows = 16` for `Whole16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowShape {
    pub position: u64,
    pub logical_ctx_rows: u64,
    pub projected_rows: u64,
    pub full_cached_rows: u64,
    pub compact_offset: u64,
    pub ring_head: u64,
    pub swa_w: u64,
    pub full_w: u64,
    pub max_position: u64,
    pub max_tiles: u64,
    pub verify_rows: u64,
    pub draft_rows: u64,
    /// Monotonic window generation: a bind refuses a generation that is not
    /// after the last submitted one.
    pub generation: u64,
}

impl WindowShape {
    /// The value of the named input (the field's own name).
    pub fn get(&self, name: &str) -> Option<u64> {
        Some(match name {
            "position" => self.position,
            "logical_ctx_rows" => self.logical_ctx_rows,
            "projected_rows" => self.projected_rows,
            "full_cached_rows" => self.full_cached_rows,
            "compact_offset" => self.compact_offset,
            "ring_head" => self.ring_head,
            "swa_w" => self.swa_w,
            "full_w" => self.full_w,
            "max_position" => self.max_position,
            "max_tiles" => self.max_tiles,
            "verify_rows" => self.verify_rows,
            "draft_rows" => self.draft_rows,
            "generation" => self.generation,
            _ => return None,
        })
    }

    fn is_name(name: &str) -> bool {
        WindowShape::default().get(name).is_some()
    }
}

/// One allocation of a binding key: identity, extent and generation (§2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationIdentity {
    pub resource: ResourceId,
    pub base: u64,
    pub bytes: u64,
    /// The allocation registry's generation (a VMM mapping generation for
    /// mapped KV): equal-size reuse at another generation is a different key.
    pub generation: u64,
}

/// `WindowBindingKey` (§2): everything a prepared executable is valid for.
/// Any change invalidates it **before mutation** ([`PreparedWindow::bind`]
/// compares the whole key).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WindowBindingKey {
    pub model_id: String,
    pub draft_id: String,
    /// Exact loaded object digests, in node order.
    pub object_digests: Vec<String>,
    pub toolchain_digest: String,
    pub device: String,
    /// Numerical-route policy flags (§2 "Authoring, not capture").
    pub route_flags: u64,
    /// One entry per program resource, in `ResourceId` order.
    pub allocations: Vec<AllocationIdentity>,
    pub kv_mapping_generation: u64,
    pub scratch_capacities: Vec<u64>,
    pub context_mode: String,
    /// GDN quant / EF / requant mode.
    pub gdn_mode: String,
    pub extraction_layers: Vec<u32>,
}

impl WindowBindingKey {
    /// The first field in which `self` differs from `other`, for refusals.
    fn first_difference(&self, other: &Self) -> Option<String> {
        macro_rules! field {
            ($f:ident) => {
                if self.$f != other.$f {
                    return Some(stringify!($f).to_owned());
                }
            };
        }
        field!(model_id);
        field!(draft_id);
        field!(object_digests);
        field!(toolchain_digest);
        field!(device);
        field!(route_flags);
        if self.allocations.len() != other.allocations.len() {
            return Some("allocations.len".to_owned());
        }
        for (i, (a, b)) in self.allocations.iter().zip(&other.allocations).enumerate() {
            if a.generation != b.generation {
                return Some(format!("allocations[{i}].generation"));
            }
            if a != b {
                return Some(format!("allocations[{i}]"));
            }
        }
        field!(kv_mapping_generation);
        field!(scratch_capacities);
        field!(context_mode);
        field!(gdn_mode);
        field!(extraction_layers);
        None
    }
}

/// The live bindings a window is bound against (§2 `WindowBindings`): the
/// key as the owner sees it now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowBindings {
    pub key: WindowBindingKey,
}

/// `WindowInputs` (§2): the host-known bytes of one window. The owner packs
/// them into persistent pinned storage; [`WindowInputs::pack`] is the byte
/// oracle of that upload. `q_positions` / `k_position_suffix` /
/// `verify_positions` are the *live* absolute positions
/// (`thlog.abs_positions()`, compaction offsets included): nothing here
/// synthesizes `0..L`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowInputs {
    pub seed: u32,
    pub mask_id: u32,
    pub q_positions: Vec<u32>,
    pub k_position_suffix: Vec<u32>,
    pub verify_positions: Vec<u32>,
    /// Named scalar inputs patch expressions may read (e.g. the projection
    /// segment row counts). They may not shadow a [`WindowShape`] field.
    pub scalars: BTreeMap<String, u64>,
    /// The `GdnFrame` value of every such word, reserved by the owner in the
    /// same order and increments as HIP before bind.
    pub gdn_frames: BTreeMap<WordId, u32>,
}

impl WindowInputs {
    /// The upload bytes: little-endian `seed`, `mask_id`, `q_positions`,
    /// `k_position_suffix`, `verify_positions`, in that order.
    pub fn pack(&self) -> Vec<u8> {
        let words =
            2 + self.q_positions.len() + self.k_position_suffix.len() + self.verify_positions.len();
        let mut out = Vec::with_capacity(words * 4);
        for w in [self.seed, self.mask_id]
            .iter()
            .chain(&self.q_positions)
            .chain(&self.k_position_suffix)
            .chain(&self.verify_positions)
        {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn validate(&self, shape: &WindowShape) -> Result<(), Refusal> {
        let rows = usize::try_from(shape.verify_rows)
            .map_err(|_| Refusal::Inputs("verify_rows exceeds usize".into()))?;
        if self.q_positions.len() != rows || self.verify_positions.len() != rows {
            return Err(Refusal::Inputs(format!(
                "{} Q and {} verify positions for {rows} verify rows",
                self.q_positions.len(),
                self.verify_positions.len()
            )));
        }
        if self.k_position_suffix.len() as u64 > shape.logical_ctx_rows {
            return Err(Refusal::Inputs(format!(
                "K-position suffix of {} rows exceeds logical_ctx_rows {}",
                self.k_position_suffix.len(),
                shape.logical_ctx_rows
            )));
        }
        for (what, list) in [
            ("q_positions", &self.q_positions),
            ("k_position_suffix", &self.k_position_suffix),
            ("verify_positions", &self.verify_positions),
        ] {
            if let Some(p) = list.iter().find(|p| u64::from(**p) > shape.max_position) {
                return Err(Refusal::Inputs(format!(
                    "{what} holds {p}, beyond max_position {}",
                    shape.max_position
                )));
            }
        }
        for name in self.scalars.keys() {
            if WindowShape::is_name(name) {
                return Err(Refusal::Inputs(format!(
                    "scalar {name:?} shadows a WindowShape field"
                )));
            }
        }
        Ok(())
    }

    fn lookup(&self, shape: &WindowShape, name: &str) -> Option<u64> {
        shape.get(name).or_else(|| self.scalars.get(name).copied())
    }
}

/// A proposed `from → to` change of one host watermark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub from: u64,
    pub to: u64,
}

/// `WindowHostDelta` (§2): proposed updates to the draft projection / upload
/// watermarks and the hidden ring head/written. Planning does not publish
/// them: the owner commits them once, after successful GPU completion and
/// before the accepted-prefix scatter (§6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowHostDelta {
    pub projection_watermark: Option<Transition>,
    pub upload_watermark: Option<Transition>,
    pub ring_head: Option<Transition>,
    pub ring_written: Option<Transition>,
}

/// `from`/`to` expressions of one [`Transition`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionSpec {
    pub from: Expr,
    pub to: Expr,
}

/// The expressions [`WindowHostDelta`] is evaluated from at bind time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostDeltaSpec {
    pub projection_watermark: Option<TransitionSpec>,
    pub upload_watermark: Option<TransitionSpec>,
    pub ring_head: Option<TransitionSpec>,
    pub ring_written: Option<TransitionSpec>,
}

// ------------------------------------------------------------------ plan

/// §2 `WindowPlan` execution shape: one IB, or the draft segment followed by
/// the host proposal splice (each segment is its own single-segment plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowExecution {
    Whole16,
    HostSplice,
}

/// A byte address inside a resource: `base(resource) + byte_offset`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub resource: ResourceId,
    pub byte_offset: Expr,
}

/// A typed D2D copy (§2 "Route all D2D copies ... through typed
/// operations"): `bytes` from `src` to `dst`, ranges disjoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyStep {
    pub src: Region,
    pub dst: Region,
    pub bytes: Expr,
}

/// A typed zero/constant fill (§2 `railgun_fill_u32`): `words` u32 words of
/// `value` at a 4-byte-aligned `dst`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FillStep {
    pub dst: Region,
    pub value: u32,
    pub words: Expr,
}

/// One ordered descriptor of the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowStep {
    /// Dispatch of program node `node`; kernel steps appear in node order.
    Kernel {
        node: usize,
    },
    Copy(CopyStep),
    Fill(FillStep),
}

/// A copy/fill step with its addresses resolved and bounds proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedStep {
    Copy { src: u64, dst: u64, bytes: u64 },
    Fill { dst: u64, value: u32, words: u64 },
}

/// The shape family (§3): ordered kernel identities and projection segment
/// row counts — not the changing addresses or position.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeFamily {
    pub kernels: Vec<String>,
    pub projection_rows: Vec<u32>,
}

/// The read/write allocation closure of a plan (§2).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllocationClosure {
    pub reads: BTreeSet<ResourceId>,
    pub writes: BTreeSet<ResourceId>,
}

/// Everything a [`WindowPlan`] is built from.
#[derive(Clone, Debug)]
pub struct WindowPlanParts {
    pub program: KernelProgram,
    pub steps: Vec<WindowStep>,
    pub patches: Vec<PatchKind>,
    pub host_delta: HostDeltaSpec,
    /// Resources downloaded or read after completion (must be written).
    pub completion_surfaces: Vec<ResourceId>,
    /// Projection segment row counts (part of the shape family).
    pub projection_rows: Vec<u32>,
    pub execution: WindowExecution,
    /// True only for a certified exact route (§2 `WindowVerifyReceipt`).
    pub hip_graph_certified: bool,
    pub tape_populated: bool,
}

/// `WindowPlan` (§2): ordered typed descriptors, bounded patch sites,
/// read/write allocation closure, shape-family key, host delta, completion
/// surfaces and the execution shape — validated at construction.
#[derive(Clone, Debug)]
pub struct WindowPlan {
    program: KernelProgram,
    steps: Vec<WindowStep>,
    patches: Vec<PatchKind>,
    /// Per patch: the `WordId` index of a `Scalar32`.
    scalar_words: Vec<Option<usize>>,
    node_has_grid_patch: Vec<bool>,
    host_delta: HostDeltaSpec,
    completion_surfaces: Vec<ResourceId>,
    closure: AllocationClosure,
    family: ShapeFamily,
    execution: WindowExecution,
    hip_graph_certified: bool,
    tape_populated: bool,
    /// The one sealed-positions resource every conditional node indexes
    /// (`None` when the plan has no conditional node).
    indexed_positions: Option<ResourceId>,
    /// The persistent completion budget ([`Self::with_persistent_watchdog`]);
    /// `None` keeps today's non-persistent semantics.
    persistent: Option<PersistentWatchdog>,
}

impl WindowPlan {
    /// Validate and assemble a plan. `pm4` is railgun's boundary plan of
    /// `parts.program`: the window must be one segment with every node
    /// PM4-lowered (§7 A1: "One segment/no untyped holes"), and every
    /// declared patch must name a real typed field of the program.
    pub fn new(parts: WindowPlanParts, pm4: &Pm4Plan) -> Result<Self, Refusal> {
        let WindowPlanParts {
            program,
            steps,
            patches,
            host_delta,
            completion_surfaces,
            projection_rows,
            execution,
            hip_graph_certified,
            tape_populated,
        } = parts;
        let malformed = |m: String| Err(Refusal::Malformed(m));
        let nodes = program.nodes.len();
        if nodes == 0 {
            return malformed("empty program".into());
        }
        for (node, n) in program.nodes.iter().enumerate() {
            if !n.lowering.is_pm4() {
                return Err(Refusal::NotPm4 {
                    node,
                    symbol: n.symbol.clone(),
                    reason: format!("{:?}", n.lowering),
                });
            }
        }
        if !pm4.is_single_segment(nodes) {
            return malformed(format!(
                "{} segments over {nodes} nodes: a window plan is one segment",
                pm4.segments.len()
            ));
        }
        // Steps: kernel nodes exactly once, in order.
        let mut next_node = 0usize;
        let resources = program.resources.len();
        for (i, step) in steps.iter().enumerate() {
            match step {
                WindowStep::Kernel { node } => {
                    if *node != next_node {
                        return malformed(format!(
                            "step {i} dispatches node {node}, expected node {next_node}"
                        ));
                    }
                    next_node += 1;
                }
                WindowStep::Copy(c) => {
                    for r in [&c.src, &c.dst] {
                        if r.resource.0 as usize >= resources {
                            return malformed(format!(
                                "step {i}: unknown resource {}",
                                r.resource.0
                            ));
                        }
                    }
                }
                WindowStep::Fill(f) => {
                    if f.dst.resource.0 as usize >= resources {
                        return malformed(format!(
                            "step {i}: unknown resource {}",
                            f.dst.resource.0
                        ));
                    }
                }
            }
        }
        if next_node != nodes {
            return malformed(format!("steps dispatch {next_node} of {nodes} nodes"));
        }
        // Patches: typed, unique, and aimed at real fields.
        let mut scalar_words = Vec::with_capacity(patches.len());
        let mut node_has_grid_patch = vec![false; nodes];
        let mut seen_scalar = BTreeSet::new();
        let mut seen_interior = BTreeSet::new();
        let mut seen_axis = BTreeSet::new();
        for (i, patch) in patches.iter().enumerate() {
            let mut word_index = None;
            match patch {
                PatchKind::Scalar32 { field, domain, .. } => {
                    if domain.start() > domain.end() {
                        return malformed(format!("patch {i}: empty domain"));
                    }
                    let Some(index) = program
                        .words
                        .iter()
                        .position(|w| w.node == field.node && w.offset == field.offset)
                    else {
                        return malformed(format!(
                            "patch {i}: node {} +{} is not a declared word",
                            field.node, field.offset
                        ));
                    };
                    if program.words[index].role != WordRole::Window {
                        return malformed(format!(
                            "patch {i}: node {} +{} is not a Window word",
                            field.node, field.offset
                        ));
                    }
                    if !seen_scalar.insert(index) {
                        return malformed(format!("patch {i}: duplicate Scalar32 site"));
                    }
                    word_index = Some(index);
                }
                PatchKind::Interior64 {
                    node,
                    arg_offset,
                    resource,
                    ..
                } => {
                    let Some(n) = program.nodes.get(*node) else {
                        return malformed(format!("patch {i}: no node {node}"));
                    };
                    let ok = n.args.iter().any(|(offset, arg)| {
                        offset == arg_offset
                            && matches!(arg, Arg::Resource { resource: r, .. } if r == resource)
                    });
                    if !ok {
                        return malformed(format!(
                            "patch {i}: node {node} +{arg_offset} is not a pointer into resource {}",
                            resource.0
                        ));
                    }
                    if !seen_interior.insert((*node, *arg_offset)) {
                        return malformed(format!("patch {i}: duplicate Interior64 site"));
                    }
                }
                PatchKind::GridAxis {
                    node, axis, domain, ..
                } => {
                    let Some(n) = program.nodes.get(*node) else {
                        return malformed(format!("patch {i}: no node {node}"));
                    };
                    if n.segment_bytes.is_none() {
                        return malformed(format!(
                            "patch {i}: node {node} has no certified segment"
                        ));
                    }
                    if *domain.start() == 0 || domain.start() > domain.end() {
                        return malformed(format!(
                            "patch {i}: a grid domain must be a nonempty range of nonzero counts"
                        ));
                    }
                    if !seen_axis.insert((*node, axis.index())) {
                        return malformed(format!("patch {i}: duplicate GridAxis site"));
                    }
                    node_has_grid_patch[*node] = true;
                }
            }
            scalar_words.push(word_index);
        }
        // No untyped holes: every Window word has its Scalar32.
        for (index, word) in program.words.iter().enumerate() {
            if word.role == WordRole::Window && !seen_scalar.contains(&index) {
                return malformed(format!(
                    "Window word node {} +{} has no Scalar32 patch",
                    word.node, word.offset
                ));
            }
        }
        // Closure.
        let mut closure = AllocationClosure::default();
        for n in &program.nodes {
            for e in &n.effects {
                if e.reads() {
                    closure.reads.insert(e.resource);
                }
                if e.writes() {
                    closure.writes.insert(e.resource);
                }
            }
        }
        for step in &steps {
            match step {
                WindowStep::Kernel { .. } => {}
                WindowStep::Copy(c) => {
                    closure.reads.insert(c.src.resource);
                    closure.writes.insert(c.dst.resource);
                }
                WindowStep::Fill(f) => {
                    closure.writes.insert(f.dst.resource);
                }
            }
        }
        for r in &completion_surfaces {
            if !closure.writes.contains(r) {
                return malformed(format!(
                    "completion surface resource {} is not written by the window",
                    r.0
                ));
            }
        }
        // Conditional (loaded-index) nodes: all must index one positions
        // resource, which is the resource the sealed positions occupy.
        let mut indexed_positions: Option<ResourceId> = None;
        for (node, n) in program.nodes.iter().enumerate() {
            let Some(case) = &n.indexed else { continue };
            let source = case
                .effects
                .iter()
                .flat_map(|e| e.index_loads.iter().map(|l| l.source_arg_offset))
                .collect::<BTreeSet<_>>();
            if source != BTreeSet::from([writer_abi::POSITIONS]) {
                return malformed(format!(
                    "node {node}: index loads read arguments {source:?}, the sealed-positions case reads +{}",
                    writer_abi::POSITIONS
                ));
            }
            let Some(resource) = n.args.iter().find_map(|(offset, arg)| match arg {
                Arg::Resource { resource, .. } if *offset == writer_abi::POSITIONS => {
                    Some(*resource)
                }
                _ => None,
            }) else {
                return malformed(format!("node {node}: positions argument is not a resource"));
            };
            if indexed_positions.is_some_and(|r| r != resource) {
                return malformed(format!(
                    "node {node}: indexes positions resource {}, another conditional node indexes {}",
                    resource.0,
                    indexed_positions.expect("checked").0
                ));
            }
            indexed_positions = Some(resource);
        }
        let kernels = program
            .nodes
            .iter()
            .map(|n| match &n.kref {
                Some(k) => format!("{}:{}", k.artifact_sha256, n.symbol),
                None => n.symbol.clone(),
            })
            .collect();
        Ok(Self {
            program,
            steps,
            patches,
            scalar_words,
            node_has_grid_patch,
            host_delta,
            completion_surfaces,
            closure,
            family: ShapeFamily {
                kernels,
                projection_rows,
            },
            execution,
            hip_graph_certified,
            tape_populated,
            indexed_positions,
            persistent: None,
        })
    }

    pub fn program(&self) -> &KernelProgram {
        &self.program
    }

    pub fn steps(&self) -> &[WindowStep] {
        &self.steps
    }

    pub fn patches(&self) -> &[PatchKind] {
        &self.patches
    }

    pub fn closure(&self) -> &AllocationClosure {
        &self.closure
    }

    pub fn family(&self) -> &ShapeFamily {
        &self.family
    }

    pub fn execution(&self) -> WindowExecution {
        self.execution
    }

    pub fn completion_surfaces(&self) -> &[ResourceId] {
        &self.completion_surfaces
    }

    /// True only for a certified exact route: what the window's
    /// [`WindowVerifyReceipt::same_as_hip_graph`] reports on completion.
    pub fn hip_graph_certified(&self) -> bool {
        self.hip_graph_certified
    }

    /// Whether the verify portion populates the GDN tape (its receipt's
    /// `tape_populated`).
    pub fn tape_populated(&self) -> bool {
        self.tape_populated
    }

    /// The sealed-positions resource of the plan's conditional nodes.
    pub fn indexed_positions(&self) -> Option<ResourceId> {
        self.indexed_positions
    }

    /// Declare this plan persistent: its completion lease runs `watchdog`
    /// (wait, poison, tear down — see [`crate::watchdog`]) instead of the
    /// queue's global wait. The poison cell must be a 4-byte-aligned cell of
    /// a resource of the program declared `EventCells` (set-only for the
    /// device by the event contract); it is validated against the program
    /// here and resolved to an address at [`PreparedWindow::new`], which also
    /// refuses a queue that cannot cancel.
    pub fn with_persistent_watchdog(
        mut self,
        watchdog: PersistentWatchdog,
    ) -> Result<Self, Refusal> {
        watchdog
            .validate(&self.program)
            .map_err(Refusal::Malformed)?;
        self.persistent = Some(watchdog);
        Ok(self)
    }

    /// The persistent completion budget, when the plan declares one.
    pub fn persistent_watchdog(&self) -> Option<&PersistentWatchdog> {
        self.persistent.as_ref()
    }
}

// ----------------------------------------------------------------- queue

/// A failure of the queue, with what is known of its state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueFailure {
    pub message: String,
    pub quiescence: Quiescence,
    /// What the failure means, typed ([`FailureKind::Queue`] for every
    /// failure that is not a persistent-cancel or quarantine outcome).
    pub kind: FailureKind,
}

impl QueueFailure {
    /// An ordinary ([`FailureKind::Queue`]) failure.
    pub fn new(message: impl Into<String>, quiescence: Quiescence) -> Self {
        Self {
            message: message.into(),
            quiescence,
            kind: FailureKind::Queue,
        }
    }

    pub fn with_kind(mut self, kind: FailureKind) -> Self {
        self.kind = kind;
        self
    }
}

/// The result of [`WindowQueue::wait_timeout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TimedWait {
    /// The work finished (or failed) within the timeout: exactly what
    /// [`WindowQueue::wait`] would return.
    Done(Result<(), QueueFailure>),
    /// The timeout elapsed with the work still in flight. Nothing was torn
    /// down; the queue state is untouched.
    Pending,
}

/// One node's bound inputs, handed to the queue at submit.
#[derive(Clone, Copy, Debug)]
pub struct NodeImage<'a> {
    pub kernarg: &'a [u8],
    /// The HIP grid (workgroups) the node dispatches.
    pub grid: [u32; 3],
}

/// The GPU-facing half of a prepared window. Implemented by `rdna-compute`
/// over the owned lowerer; its kernarg buffers, loaded kernels and IB outlive
/// completion and are only written while the window is retired.
pub trait WindowQueue: Send {
    /// Write every node's kernarg image and grid into the queue's retired
    /// storage, run the typed copy/fill steps' emission inputs, and launch.
    /// `Ok` means the work was launched; `Err` carries what is known. A
    /// queue that blocks here has proven quiescence by the time it returns
    /// (its [`Self::wait`] only reports the outcome); see
    /// [`Self::submit_nowait`] for one that has not.
    fn submit(
        &mut self,
        nodes: &[NodeImage<'_>],
        steps: &[ResolvedStep],
    ) -> Result<(), QueueFailure>;

    /// [`Self::submit`] that does not wait for the launched work: `Ok` means
    /// *in flight* and nothing more, and quiescence is proven only by
    /// [`Self::wait`], whose result is this submit's outcome. The queue's
    /// kernarg and grid storage belongs to the GPU until `wait` returns, so a
    /// queue refuses a `submit` or `submit_nowait` issued while one is
    /// outstanding (a [`Quiescence::Proven`] failure, "submit without
    /// completion", before anything is written).
    ///
    /// The default is `submit`: a queue that blocks in `submit` is trivially
    /// submitted when it returns, and its `wait` reports the stored result.
    fn submit_nowait(
        &mut self,
        nodes: &[NodeImage<'_>],
        steps: &[ResolvedStep],
    ) -> Result<(), QueueFailure> {
        self.submit(nodes, steps)
    }

    /// Whether this queue implements the persistent-cancel contract
    /// ([`Self::wait_timeout`], [`Self::raise_poison`], [`Self::teardown`]).
    /// A persistent window (or a watched raw submit) is refused on a queue
    /// that does not.
    fn supports_persistent_cancel(&self) -> bool {
        false
    }

    /// Wait up to `timeout` for the in-flight work to retire. It MUST NOT
    /// tear anything down on timeout: it returns [`TimedWait::Pending`] with
    /// the work in flight and the queue state untouched. `Done` is exactly
    /// what [`Self::wait`] would return, and a call with nothing in flight
    /// is `Done` at once. The default has no bounded wait and reports an
    /// unknown-quiescence failure.
    fn wait_timeout(&mut self, timeout: Duration) -> TimedWait {
        let _ = timeout;
        TimedWait::Done(Err(QueueFailure::new(
            "queue has no bounded wait",
            Quiescence::Unknown,
        )))
    }

    /// Store the non-zero `value` at device `address` (a persistent
    /// window's poison cell). It MUST NOT wait on, or submit to, the stuck
    /// queue; `Err` means the poison was not delivered. The default cannot
    /// write.
    fn raise_poison(&mut self, address: u64, value: u32) -> Result<(), QueueFailure> {
        let _ = (address, value);
        Err(QueueFailure::new(
            "queue has no queue-independent poison write",
            Quiescence::Proven,
        ))
    }

    /// Cancel the in-flight work (queue inactivation or destruction), then
    /// observe up to `observe` for retirement evidence, and return what is
    /// *proven*. After a teardown the queue refuses every further submit
    /// ([`FailureKind::TornDown`]). Idempotent. The default proves nothing.
    fn teardown(&mut self, observe: Duration) -> Teardown {
        let _ = observe;
        Teardown {
            quiescence: Quiescence::Unknown,
            message: "queue has no teardown".into(),
        }
    }

    /// Prove queue quiescence (blocking).
    fn wait(&mut self) -> Result<(), QueueFailure>;
}

/// A [`WindowQueue::submit_nowait`] in flight on a queue driven directly
/// (without a [`PreparedWindow`]), from [`submit_queue_nowait`]. It holds the
/// queue's `&mut` borrow, so a second submit is impossible while it lives.
///
/// [`Self::wait`] proves quiescence and returns the queue's outcome. Dropping
/// the handle without waiting waits too (the result is discarded), so a
/// dropped handle never leaves the queue in flight unobserved; a failure
/// stays stored in the queue for its next submit (an `Unknown` one refuses
/// it).
#[must_use = "dropping the handle blocks until the queue is quiescent; call wait() for its result"]
pub struct QueueCompletion<'q, Q: WindowQueue + ?Sized> {
    queue: &'q mut Q,
    waited: bool,
    /// `Some` for a persistent submit ([`submit_queue_nowait_watched`]).
    watchdog: Option<ResolvedWatchdog>,
}

impl<Q: WindowQueue + ?Sized> QueueCompletion<'_, Q> {
    /// Prove queue quiescence (blocking) and return the submit's outcome. A
    /// watched (persistent) completion runs [`run_watchdog`]: it returns only
    /// when the program completed, was poisoned and retired, or was torn down
    /// — never claiming quiescence it has not proven.
    pub fn wait(mut self) -> Result<(), QueueFailure> {
        self.waited = true;
        self.finish()
    }

    fn finish(&mut self) -> Result<(), QueueFailure> {
        match &self.watchdog {
            Some(w) => run_watchdog(&mut *self.queue, w),
            None => self.queue.wait(),
        }
    }
}

impl<Q: WindowQueue + ?Sized> Drop for QueueCompletion<'_, Q> {
    fn drop(&mut self) {
        if !self.waited {
            let _ = self.finish();
        }
    }
}

/// [`WindowQueue::submit_nowait`] on a raw queue: `Ok` is the in-flight
/// handle, `Err` is a refused or failed submit (nothing in flight; the
/// queue's `wait` still reports a stored failure).
pub fn submit_queue_nowait<'q, Q: WindowQueue + ?Sized>(
    queue: &'q mut Q,
    nodes: &[NodeImage<'_>],
    steps: &[ResolvedStep],
) -> Result<QueueCompletion<'q, Q>, QueueFailure> {
    queue.submit_nowait(nodes, steps)?;
    Ok(QueueCompletion {
        queue,
        waited: false,
        watchdog: None,
    })
}

/// [`submit_queue_nowait`] for a persistent program: the returned handle's
/// [`QueueCompletion::wait`] (and its drop) run the persistent watchdog
/// instead of the queue's global wait. Refused (`Proven`, nothing written)
/// on a queue that does not implement the persistent-cancel contract.
pub fn submit_queue_nowait_watched<'q, Q: WindowQueue + ?Sized>(
    queue: &'q mut Q,
    nodes: &[NodeImage<'_>],
    steps: &[ResolvedStep],
    watchdog: &ResolvedWatchdog,
) -> Result<QueueCompletion<'q, Q>, QueueFailure> {
    if !queue.supports_persistent_cancel() {
        return Err(QueueFailure::new(
            "queue does not implement persistent cancel (wait_timeout/raise_poison/teardown)",
            Quiescence::Proven,
        ));
    }
    queue.submit_nowait(nodes, steps)?;
    Ok(QueueCompletion {
        queue,
        waited: false,
        watchdog: Some(*watchdog),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Retired,
    Bound,
    InFlight,
    Quarantined,
}

struct Shared {
    phase: Phase,
    /// Incremented by every bind attempt: a [`BoundWindow`] of an older
    /// epoch is stale.
    epoch: u64,
    last_submitted_generation: Option<u64>,
    queue: Box<dyn WindowQueue>,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

static NEXT_PREPARED_ID: AtomicU64 = AtomicU64::new(1);

// -------------------------------------------------------------- prepared

/// A plan bound to one [`WindowBindingKey`] with its kernarg storage.
pub struct PreparedWindow {
    id: u64,
    plan: Arc<WindowPlan>,
    key: WindowBindingKey,
    bindings: Vec<Binding>,
    shared: Arc<Mutex<Shared>>,
    /// Retired-storage kernarg images, patched in place per bind.
    images: Vec<Vec<u8>>,
    words: Vec<u32>,
    grids: Vec<[u32; 3]>,
    interiors: Vec<Vec<(u32, u64)>>,
    resolved: Vec<ResolvedStep>,
    /// The persistent completion budget with its poison cell resolved to this
    /// window's allocations; `None` for a non-persistent plan.
    watchdog: Option<ResolvedWatchdog>,
}

/// `BoundWindow` (§2): the proof that one generation's patches validated.
/// Consumed by [`PreparedWindow::submit`].
#[derive(Debug)]
pub struct BoundWindow {
    prepared: u64,
    epoch: u64,
    generation: u64,
    host_delta: WindowHostDelta,
    /// The consumed-at-submit bind witness of a plan with conditional nodes.
    indexed: Option<(IndexedBindWitness, SealedPositions)>,
    /// The consumed-at-submit sealed-table witness of a plan with
    /// sealed-table nodes, with the registry it was bound against.
    tables: Option<(SealedTableWitness, SealedTableRegistry)>,
    /// A certified exact route for **this** window: the manifest admitted the
    /// plan with this bind's witnesses ([`PreparedWindow::certify_exact`]).
    exact: bool,
}

impl BoundWindow {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn host_delta(&self) -> &WindowHostDelta {
        &self.host_delta
    }

    /// The attached bind witness, when the plan has conditional nodes.
    pub fn indexed_witness(&self) -> Option<&IndexedBindWitness> {
        self.indexed.as_ref().map(|(w, _)| w)
    }

    /// The attached sealed-table witness, when the plan has sealed-table
    /// nodes.
    pub fn sealed_table_witness(&self) -> Option<&SealedTableWitness> {
        self.tables.as_ref().map(|(w, _)| w)
    }
}

/// `WindowVerifyReceipt` (§2). `same_as_hip_graph` is true only for a
/// certified exact route, never merely for a successful submit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowVerifyReceipt {
    pub same_as_hip_graph: bool,
    pub tape_populated: bool,
    pub generation: u64,
}

/// `CompleteWindow` (§2): queue quiescence is proven; the host delta may now
/// be committed once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompleteWindow {
    pub generation: u64,
    pub execution: WindowExecution,
    pub receipt: WindowVerifyReceipt,
    pub host_delta: WindowHostDelta,
}

/// A completion that did not prove success; `quiescence` says whether the
/// queue is known idle (storage reusable) or unknown (quarantined).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("window generation {generation} failed to complete ({quiescence:?}, {kind:?}): {message}")]
pub struct CompletionFailure {
    pub generation: u64,
    pub quiescence: Quiescence,
    /// [`FailureKind::Poisoned`] / [`FailureKind::TornDown`] for a persistent
    /// window's watchdog outcomes; [`FailureKind::Queue`] otherwise.
    pub kind: FailureKind,
    pub message: String,
}

/// `Submission` (§2): the work is queued; every bound resource, kernarg
/// buffer and loaded kernel outlives [`Submission::complete`]. Dropping it
/// without completing leaves the window in flight (never reusable). After
/// [`PreparedWindow::submit_nowait`] the queue may still be running:
/// `complete` is then what waits for it, and nothing it reads may be written
/// or freed before it returns.
pub struct Submission {
    shared: Arc<Mutex<Shared>>,
    generation: u64,
    execution: WindowExecution,
    receipt_hip: bool,
    receipt_tape: bool,
    host_delta: WindowHostDelta,
    sealed: Option<SealedPositions>,
    tables: Option<TableLease>,
    /// `Some` for a persistent window: its completion runs the watchdog.
    watchdog: Option<ResolvedWatchdog>,
}

impl Submission {
    /// Prove queue quiescence. Success retires the prepared window and
    /// returns the host delta for its single commit; a failure with
    /// [`Quiescence::Proven`] retires it, [`Quiescence::Unknown`] quarantines
    /// it (§6). A persistent window runs its watchdog here
    /// ([`run_watchdog`]): a [`FailureKind::Poisoned`] failure is `Proven`
    /// (retired, outputs invalid); a [`FailureKind::TornDown`] one always
    /// quarantines the window (its queue is gone) and carries the
    /// quiescence the teardown proved.
    pub fn complete(mut self) -> Result<CompleteWindow, CompletionFailure> {
        let mut shared = lock(&self.shared);
        let waited = match &self.watchdog {
            Some(w) => run_watchdog(&mut *shared.queue, w),
            None => shared.queue.wait(),
        };
        match waited {
            Ok(()) => {
                shared.phase = Phase::Retired;
                if let Some(s) = &self.sealed {
                    s.finish(Quiescence::Proven);
                }
                if let Some(l) = self.tables.take() {
                    l.release(Quiescence::Proven);
                }
                Ok(CompleteWindow {
                    generation: self.generation,
                    execution: self.execution,
                    receipt: WindowVerifyReceipt {
                        same_as_hip_graph: self.receipt_hip,
                        tape_populated: self.receipt_tape,
                        generation: self.generation,
                    },
                    host_delta: self.host_delta,
                })
            }
            Err(f) => {
                shared.phase = match (f.kind, f.quiescence) {
                    (FailureKind::TornDown, _) => Phase::Quarantined,
                    (_, Quiescence::Proven) => Phase::Retired,
                    (_, Quiescence::Unknown) => Phase::Quarantined,
                };
                if let Some(s) = &self.sealed {
                    s.finish(f.quiescence);
                }
                if let Some(l) = self.tables.take() {
                    l.release(f.quiescence);
                }
                Err(CompletionFailure {
                    generation: self.generation,
                    quiescence: f.quiescence,
                    kind: f.kind,
                    message: f.message,
                })
            }
        }
    }
}

fn expr_refusal(error: ExprError, site: &str) -> Refusal {
    match error {
        ExprError::UnknownInput(name) => Refusal::UnknownInput {
            site: site.to_owned(),
            name,
        },
        ExprError::Overflow => Refusal::Overflow {
            site: site.to_owned(),
        },
    }
}

fn domain_check(site: &str, value: u64, domain: &RangeInclusive<u64>) -> Result<u32, Refusal> {
    if !domain.contains(&value) {
        return Err(Refusal::OutOfDomain {
            site: site.to_owned(),
            value,
            start: *domain.start(),
            end: *domain.end(),
        });
    }
    u32::try_from(value).map_err(|_| Refusal::DoesNotFit {
        site: site.to_owned(),
        value,
    })
}

impl PreparedWindow {
    /// Prepare `plan` for `key`: proves the allocation set (one per
    /// resource, nonzero extent, no address overflow, no overlap with a
    /// written resource) and that every unpatched pointer lies inside its
    /// allocation. Nothing is submitted.
    pub fn new(
        plan: Arc<WindowPlan>,
        key: WindowBindingKey,
        queue: Box<dyn WindowQueue>,
    ) -> Result<Self, Refusal> {
        let program = &plan.program;
        if key.allocations.len() != program.resources.len() {
            return Err(Refusal::BadBinding {
                resource: 0,
                detail: format!(
                    "{} allocations for {} resources",
                    key.allocations.len(),
                    program.resources.len()
                ),
            });
        }
        let mut bindings = Vec::with_capacity(key.allocations.len());
        for (i, a) in key.allocations.iter().enumerate() {
            let resource = i as u32;
            if a.resource.0 != resource {
                return Err(Refusal::BadBinding {
                    resource,
                    detail: format!("allocation {i} names resource {}", a.resource.0),
                });
            }
            if a.base == 0 || a.bytes == 0 {
                return Err(Refusal::BadBinding {
                    resource,
                    detail: "null base or zero capacity".into(),
                });
            }
            if a.base.checked_add(a.bytes).is_none() {
                return Err(Refusal::BadBinding {
                    resource,
                    detail: "base + capacity overflows the address space".into(),
                });
            }
            bindings.push(Binding {
                base: a.base,
                bytes: a.bytes,
            });
        }
        for i in 0..bindings.len() {
            for j in i + 1..bindings.len() {
                let (a, b) = (bindings[i], bindings[j]);
                let overlap = a.base < b.base + b.bytes && b.base < a.base + a.bytes;
                let written = plan.closure.writes.contains(&ResourceId(i as u32))
                    || plan.closure.writes.contains(&ResourceId(j as u32));
                if overlap && written {
                    return Err(Refusal::Alias {
                        a: i as u32,
                        b: j as u32,
                    });
                }
            }
        }
        // Unpatched interiors must lie inside their allocation.
        for (node, n) in program.nodes.iter().enumerate() {
            for (offset, arg) in &n.args {
                let Arg::Resource { resource, interior } = arg else {
                    continue;
                };
                let patched = plan.patches.iter().any(|p| {
                    matches!(p, PatchKind::Interior64 { node: pn, arg_offset, .. }
                        if *pn == node && arg_offset == offset)
                });
                let bytes = bindings[resource.0 as usize].bytes;
                if !patched && *interior >= bytes {
                    return Err(Refusal::ExtentExceeded {
                        site: format!("node {node} +{offset}"),
                        resource: resource.0,
                        offset: *interior,
                        end: *interior,
                        bytes,
                    });
                }
            }
        }
        // A persistent plan needs a queue that can cancel, and its poison
        // cell resolved to this key's allocation.
        let watchdog = match &plan.persistent {
            Some(w) => {
                if !queue.supports_persistent_cancel() {
                    return Err(Refusal::Malformed(
                        "persistent window on a queue without persistent cancel \
                         (wait_timeout/raise_poison/teardown)"
                            .into(),
                    ));
                }
                Some(w.resolve(&bindings).map_err(Refusal::Malformed)?)
            }
            None => None,
        };
        let images = program
            .nodes
            .iter()
            .map(|n| vec![0u8; n.segment_bytes.unwrap_or(0) as usize])
            .collect();
        let interiors = vec![Vec::new(); program.nodes.len()];
        let grids = program.nodes.iter().map(|n| n.grid).collect();
        let words = program.authored_words();
        Ok(Self {
            id: NEXT_PREPARED_ID.fetch_add(1, Ordering::Relaxed),
            plan: plan.clone(),
            key,
            bindings,
            shared: Arc::new(Mutex::new(Shared {
                phase: Phase::Retired,
                epoch: 0,
                last_submitted_generation: None,
                queue,
            })),
            images,
            words,
            grids,
            interiors,
            resolved: Vec::new(),
            watchdog,
        })
    }

    pub fn plan(&self) -> &WindowPlan {
        &self.plan
    }

    pub fn key(&self) -> &WindowBindingKey {
        &self.key
    }

    /// True when nothing of this window can be in flight or quarantined:
    /// the only state in which its storage may be patched or evicted.
    pub fn is_retired(&self) -> bool {
        matches!(lock(&self.shared).phase, Phase::Retired | Phase::Bound)
    }

    /// The kernarg image encoded by the last successful bind.
    pub fn kernarg_image(&self, node: usize) -> Option<&[u8]> {
        self.images.get(node).map(Vec::as_slice)
    }

    /// The grid bound by the last successful bind.
    pub fn grid(&self, node: usize) -> Option<[u32; 3]> {
        self.grids.get(node).copied()
    }

    /// The copy/fill steps resolved by the last successful bind.
    pub fn resolved_steps(&self) -> &[ResolvedStep] {
        &self.resolved
    }

    /// `PreparedWindow::bind` (§2): validate every patch without submitting,
    /// then re-encode the kernarg images in place. Any refusal leaves the
    /// queue untouched and invalidates earlier un-submitted [`BoundWindow`]s.
    pub fn bind(
        &mut self,
        shape: &WindowShape,
        bindings: &WindowBindings,
        inputs: &WindowInputs,
    ) -> Result<BoundWindow, Refusal> {
        let (epoch, last) = {
            let mut shared = lock(&self.shared);
            match shared.phase {
                Phase::InFlight => return Err(Refusal::InFlight),
                Phase::Quarantined => return Err(Refusal::Quarantined),
                Phase::Retired | Phase::Bound => {}
            }
            shared.epoch += 1;
            shared.phase = Phase::Retired;
            (shared.epoch, shared.last_submitted_generation)
        };
        if let Some(field) = self.key.first_difference(&bindings.key) {
            return Err(Refusal::BindingKeyMismatch(field));
        }
        if let Some(last) = last {
            if shape.generation <= last {
                return Err(Refusal::StaleGeneration {
                    got: shape.generation,
                    last,
                });
            }
        }
        if shape.verify_rows == 0 {
            return Err(Refusal::Shape("verify_rows is zero".into()));
        }
        if self.plan.execution == WindowExecution::Whole16
            && (shape.verify_rows != 16 || shape.draft_rows != 16)
        {
            return Err(Refusal::Shape(format!(
                "Whole16 needs draft_rows = verify_rows = 16, got {} and {}",
                shape.draft_rows, shape.verify_rows
            )));
        }
        if shape.position > shape.max_position {
            return Err(Refusal::Shape(format!(
                "position {} exceeds max_position {}",
                shape.position, shape.max_position
            )));
        }
        inputs.validate(shape)?;

        let plan = Arc::clone(&self.plan);
        let program = &plan.program;
        let eval = |expr: &Expr, site: &str| -> Result<u64, Refusal> {
            expr.eval(&|name| inputs.lookup(shape, name))
                .map_err(|e| expr_refusal(e, site))
        };

        // Stage every value; nothing below writes an image.
        self.words.clear();
        self.words.extend(program.authored_words());
        for (grid, n) in self.grids.iter_mut().zip(&program.nodes) {
            *grid = n.grid;
        }
        for list in &mut self.interiors {
            list.clear();
        }
        self.resolved.clear();

        for (index, word) in program.words.iter().enumerate() {
            if let WordRole::GdnFrame { .. } = word.role {
                let value = inputs
                    .gdn_frames
                    .get(&WordId(index as u32))
                    .ok_or_else(|| {
                        Refusal::Inputs(format!(
                            "no GdnFrame value for word {index} (node {} +{})",
                            word.node, word.offset
                        ))
                    })?;
                self.words[index] = *value;
            }
        }
        for id in inputs.gdn_frames.keys() {
            let ok = program
                .words
                .get(id.0 as usize)
                .is_some_and(|w| matches!(w.role, WordRole::GdnFrame { .. }));
            if !ok {
                return Err(Refusal::Inputs(format!(
                    "GdnFrame value for word {} which is not a GdnFrame word",
                    id.0
                )));
            }
        }

        for (patch, word) in plan.patches.iter().zip(&plan.scalar_words) {
            match patch {
                PatchKind::Scalar32 {
                    field,
                    expr,
                    domain,
                } => {
                    let site = format!("Scalar32 node {} +{}", field.node, field.offset);
                    let value = eval(expr, &site)?;
                    let value = domain_check(&site, value, domain)?;
                    // `word` is resolved at plan construction.
                    self.words[word.expect("Scalar32 word resolved by WindowPlan::new")] = value;
                }
                PatchKind::GridAxis {
                    node,
                    axis,
                    expr,
                    domain,
                } => {
                    let site = format!("GridAxis node {node} axis {}", axis.index());
                    let value = eval(expr, &site)?;
                    self.grids[*node][axis.index()] = domain_check(&site, value, domain)?;
                }
                PatchKind::Interior64 {
                    node,
                    arg_offset,
                    resource,
                    byte_offset,
                    required_extent,
                } => {
                    let site = format!("Interior64 node {node} +{arg_offset}");
                    let offset = eval(byte_offset, &site)?;
                    let extent = eval(required_extent, &site)?;
                    let end = offset
                        .checked_add(extent)
                        .ok_or_else(|| Refusal::Overflow { site: site.clone() })?;
                    let bytes = self.bindings[resource.0 as usize].bytes;
                    if offset >= bytes || end > bytes {
                        return Err(Refusal::ExtentExceeded {
                            site,
                            resource: resource.0,
                            offset,
                            end,
                            bytes,
                        });
                    }
                    self.interiors[*node].push((*arg_offset, offset));
                }
            }
        }

        let region = |region: &Region, len: u64, site: &str| -> Result<(u64, u64), Refusal> {
            let offset = eval(&region.byte_offset, site)?;
            let end = offset.checked_add(len).ok_or_else(|| Refusal::Overflow {
                site: site.to_owned(),
            })?;
            let binding = self.bindings[region.resource.0 as usize];
            if end > binding.bytes {
                return Err(Refusal::ExtentExceeded {
                    site: site.to_owned(),
                    resource: region.resource.0,
                    offset,
                    end,
                    bytes: binding.bytes,
                });
            }
            Ok((binding.base + offset, binding.base + end))
        };
        let mut resolved = Vec::new();
        for (i, step) in plan.steps.iter().enumerate() {
            match step {
                WindowStep::Kernel { .. } => {}
                WindowStep::Copy(c) => {
                    let site = format!("Copy step {i}");
                    let bytes = eval(&c.bytes, &site)?;
                    let (src, src_end) = region(&c.src, bytes, &site)?;
                    let (dst, dst_end) = region(&c.dst, bytes, &site)?;
                    if bytes != 0 && src < dst_end && dst < src_end {
                        return Err(Refusal::BadStep {
                            site,
                            detail: "source and destination ranges overlap".into(),
                        });
                    }
                    resolved.push(ResolvedStep::Copy { src, dst, bytes });
                }
                WindowStep::Fill(f) => {
                    let site = format!("Fill step {i}");
                    let words = eval(&f.words, &site)?;
                    let bytes = words
                        .checked_mul(4)
                        .ok_or_else(|| Refusal::Overflow { site: site.clone() })?;
                    let (dst, _) = region(&f.dst, bytes, &site)?;
                    if dst % 4 != 0 {
                        return Err(Refusal::BadStep {
                            site,
                            detail: format!("fill destination {dst:#x} is not 4-byte aligned"),
                        });
                    }
                    resolved.push(ResolvedStep::Fill {
                        dst,
                        value: f.value,
                        words,
                    });
                }
            }
        }

        let delta_of =
            |spec: &Option<TransitionSpec>, what: &str| -> Result<Option<Transition>, Refusal> {
                spec.as_ref()
                    .map(|s| {
                        let site = format!("host delta {what}");
                        Ok(Transition {
                            from: eval(&s.from, &site)?,
                            to: eval(&s.to, &site)?,
                        })
                    })
                    .transpose()
            };
        let host_delta = WindowHostDelta {
            projection_watermark: delta_of(&plan.host_delta.projection_watermark, "projection")?,
            upload_watermark: delta_of(&plan.host_delta.upload_watermark, "upload")?,
            ring_head: delta_of(&plan.host_delta.ring_head, "ring_head")?,
            ring_written: delta_of(&plan.host_delta.ring_written, "ring_written")?,
        };

        // Everything validated: encode in place.
        for (node, image) in self.images.iter_mut().enumerate() {
            let overrides = ImageOverrides {
                grid: plan.node_has_grid_patch[node].then_some(self.grids[node]),
                resources: Some(&self.bindings),
                interiors: &self.interiors[node],
            };
            program
                .encode_kernarg_into(node, &self.words, &overrides, image)
                .map_err(Refusal::Encode)?;
        }
        self.resolved = resolved;

        let mut shared = lock(&self.shared);
        shared.phase = Phase::Bound;
        Ok(BoundWindow {
            prepared: self.id,
            epoch,
            generation: shape.generation,
            host_delta,
            indexed: None,
            tables: None,
            exact: false,
        })
    }

    /// Bind the sealed positions to this window's bound state (§A9.4): the
    /// per-writer and full-program proofs run against the **encoded kernarg
    /// images** and the window's allocation identities, and the resulting
    /// [`IndexedBindWitness`] is held by `bound` until [`Self::submit`]
    /// consumes it. A stale `bound` is refused before any proof runs.
    pub fn attach_indexed_witness(
        &self,
        bound: &mut BoundWindow,
        sealed: &SealedPositions,
        writers: &[WriterExtent],
    ) -> Result<(), Refusal> {
        {
            let shared = lock(&self.shared);
            if bound.prepared != self.id
                || bound.epoch != shared.epoch
                || shared.phase != Phase::Bound
            {
                return Err(Refusal::StaleBound);
            }
        }
        let Some(positions_resource) = self.plan.indexed_positions else {
            return Err(Refusal::Indexed(IndexBindError::Closure(
                "the plan has no conditional node".into(),
            )));
        };
        let witness = sealed
            .bind_writers(&BindInputs {
                window: self.id,
                shape_generation: bound.generation,
                program: &self.plan.program,
                steps: &self.plan.steps,
                bindings: &self.bindings,
                allocations: &self.key.allocations,
                images: &self.images,
                grids: &self.grids,
                positions_resource,
                writers,
            })
            .map_err(Refusal::Indexed)?;
        bound.indexed = Some((witness, sealed.clone()));
        Ok(())
    }

    /// Bind the sealed tables to this window's bound state: the per-node and
    /// full-program proofs of [`SealedTableRegistry::bind_sealed_table`] run
    /// against the **encoded kernarg images** and the window's allocation
    /// identities, and the resulting [`SealedTableWitness`] is held by
    /// `bound` until [`Self::submit`] consumes it. `content` is the owner's
    /// view of the live table bytes. A stale `bound` is refused before any
    /// proof runs.
    pub fn attach_sealed_table_witness(
        &self,
        bound: &mut BoundWindow,
        registry: &SealedTableRegistry,
        content: &dyn TableContent,
    ) -> Result<(), Refusal> {
        {
            let shared = lock(&self.shared);
            if bound.prepared != self.id
                || bound.epoch != shared.epoch
                || shared.phase != Phase::Bound
            {
                return Err(Refusal::StaleBound);
            }
        }
        if self.plan.program.sealed_table_nodes().is_empty() {
            return Err(Refusal::SealedTable(SealedTableError::Closure(
                "the plan has no sealed-table node".into(),
            )));
        }
        let witness = registry
            .bind_sealed_table(
                &TableBindInputs {
                    window: self.id,
                    shape_generation: bound.generation,
                    program: &self.plan.program,
                    steps: &self.plan.steps,
                    bindings: &self.bindings,
                    allocations: &self.key.allocations,
                    images: &self.images,
                    grids: &self.grids,
                },
                content,
            )
            .map_err(Refusal::SealedTable)?;
        bound.tables = Some((witness, registry.clone()));
        Ok(())
    }

    /// Record that the window manifest admitted this plan **together with
    /// this bind's witnesses** (a plan with conditional nodes is not
    /// certified exact by its plan alone). The owner calls it after
    /// [`Self::attach_indexed_witness`] and a witnessed manifest admission;
    /// the completion receipt's `same_as_hip_graph` then reports it.
    pub fn certify_exact(&self, bound: &mut BoundWindow) -> Result<(), Refusal> {
        {
            let shared = lock(&self.shared);
            if bound.prepared != self.id
                || bound.epoch != shared.epoch
                || shared.phase != Phase::Bound
            {
                return Err(Refusal::StaleBound);
            }
        }
        if self.plan.indexed_positions.is_some() && bound.indexed.is_none() {
            return Err(Refusal::IndexedWitnessMissing {
                node: self
                    .plan
                    .program
                    .conditional_nodes()
                    .first()
                    .copied()
                    .unwrap_or(0),
                symbol: "certify_exact".into(),
            });
        }
        if bound.tables.is_none() {
            if let Some(&node) = self.plan.program.sealed_table_nodes().first() {
                return Err(Refusal::SealedTableWitnessMissing {
                    node,
                    symbol: self.plan.program.nodes[node].symbol.clone(),
                });
            }
        }
        bound.exact = true;
        Ok(())
    }

    /// `PreparedWindow::submit` (§2): consume the bound generation and queue
    /// it, blocking where the queue blocks (see [`WindowQueue::submit`]). A
    /// stale, foreign or already-consumed [`BoundWindow`] is refused before
    /// the queue is touched.
    pub fn submit(&mut self, bound: BoundWindow) -> Result<Submission, Refusal> {
        self.submit_with(bound, false)
    }

    /// [`Self::submit`] over [`WindowQueue::submit_nowait`]: the same checks,
    /// witnesses and failure handling, but `Ok` only means the window is in
    /// flight. The returned [`Submission`]'s [`complete`] proves quiescence
    /// (and reports the failure of a launch that did not finish); the window
    /// stays in flight, and `bind` refuses, until then. On a queue that
    /// blocks in `submit` this is exactly `submit`.
    ///
    /// [`complete`]: Submission::complete
    pub fn submit_nowait(&mut self, bound: BoundWindow) -> Result<Submission, Refusal> {
        self.submit_with(bound, true)
    }

    fn submit_with(&mut self, bound: BoundWindow, nowait: bool) -> Result<Submission, Refusal> {
        let mut shared = lock(&self.shared);
        if bound.prepared != self.id || bound.epoch != shared.epoch {
            return Err(Refusal::StaleBound);
        }
        match shared.phase {
            Phase::Bound => {}
            Phase::InFlight => return Err(Refusal::InFlight),
            Phase::Quarantined => return Err(Refusal::Quarantined),
            Phase::Retired => return Err(Refusal::StaleBound),
        }
        // A plan with sealed-table nodes is PM4-ineligible until every one of
        // them has this window's matching sealed_table witness; the pure
        // checks run before anything transitions.
        let table_nodes = self.plan.program.sealed_table_nodes();
        let table_witness = if table_nodes.is_empty() {
            None
        } else {
            let Some((witness, registry)) = &bound.tables else {
                return Err(Refusal::SealedTableWitnessMissing {
                    node: table_nodes[0],
                    symbol: self.plan.program.nodes[table_nodes[0]].symbol.clone(),
                });
            };
            witness
                .matches_window(self.id, bound.generation, &self.images, &self.grids)
                .map_err(Refusal::SealedTable)?;
            let covered: Vec<usize> = witness.nodes().iter().map(|n| n.node).collect();
            if covered != table_nodes {
                return Err(Refusal::SealedTable(SealedTableError::Stale(format!(
                    "witness covers nodes {covered:?}, the plan's sealed-table nodes are {table_nodes:?}"
                ))));
            }
            registry
                .check_witness(witness)
                .map_err(Refusal::SealedTable)?;
            Some((witness, registry))
        };
        // A plan with conditional nodes is PM4-ineligible until every one of
        // them has this window's matching witness (§A9.2).
        let sealed = if self.plan.indexed_positions.is_some() {
            let Some((witness, sealed)) = &bound.indexed else {
                let node = self
                    .plan
                    .program
                    .conditional_nodes()
                    .first()
                    .copied()
                    .unwrap_or(0);
                return Err(Refusal::IndexedWitnessMissing {
                    node,
                    symbol: self.plan.program.nodes[node].symbol.clone(),
                });
            };
            witness
                .matches_window(self.id, bound.generation, &self.images, &self.grids)
                .map_err(Refusal::Indexed)?;
            // The witness must cover exactly the plan's conditional nodes.
            let covered: Vec<usize> = witness.writers().iter().map(|w| w.node).collect();
            if covered != self.plan.program.conditional_nodes() {
                return Err(Refusal::Indexed(IndexBindError::Stale(format!(
                    "witness covers nodes {covered:?}, the plan's conditional nodes are {:?}",
                    self.plan.program.conditional_nodes()
                ))));
            }
            sealed.check_witness(witness).map_err(Refusal::Indexed)?;
            sealed.mark_submitted(witness).map_err(Refusal::Indexed)?;
            Some(sealed.clone())
        } else {
            None
        };
        // Mark the sealed tables as used by a window in flight; nothing is
        // queued yet, so a refusal here retires the sealed positions again.
        let lease = match table_witness {
            Some((witness, registry)) => match registry.lease(witness) {
                Ok(lease) => Some(lease),
                Err(e) => {
                    if let Some(s) = &sealed {
                        s.submit_failed(Quiescence::Proven);
                    }
                    return Err(Refusal::SealedTable(e));
                }
            },
            None => None,
        };
        let nodes: Vec<NodeImage<'_>> = self
            .images
            .iter()
            .zip(&self.grids)
            .map(|(kernarg, grid)| NodeImage {
                kernarg,
                grid: *grid,
            })
            .collect();
        // A persistent window never uses the queue's blocking `submit`: that
        // would wait under the queue's own global timeout, outside the
        // watchdog. Its completion lease is `Submission::complete`.
        let queued = if nowait || self.watchdog.is_some() {
            shared.queue.submit_nowait(&nodes, &self.resolved)
        } else {
            shared.queue.submit(&nodes, &self.resolved)
        };
        match queued {
            Ok(()) => {
                shared.phase = Phase::InFlight;
                shared.last_submitted_generation = Some(bound.generation);
                Ok(Submission {
                    shared: Arc::clone(&self.shared),
                    generation: bound.generation,
                    execution: self.plan.execution,
                    receipt_hip: self.plan.hip_graph_certified || bound.exact,
                    receipt_tape: self.plan.tape_populated,
                    host_delta: bound.host_delta,
                    sealed,
                    tables: lease,
                    watchdog: self.watchdog,
                })
            }
            Err(f) => {
                shared.phase = match f.quiescence {
                    Quiescence::Proven => Phase::Retired,
                    Quiescence::Unknown => Phase::Quarantined,
                };
                if let Some(s) = &sealed {
                    s.submit_failed(f.quiescence);
                }
                if let Some(l) = lease {
                    l.release(f.quiescence);
                }
                Err(Refusal::SubmitFailed {
                    message: f.message,
                    quiescence: f.quiescence,
                    kind: f.kind,
                })
            }
        }
    }
}

// ----------------------------------------------------------------- cache

/// Cache key of one prepared skeleton: the binding key and the shape family.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TopologyKey {
    pub binding: WindowBindingKey,
    pub family: ShapeFamily,
}

struct CacheEntry {
    key: TopologyKey,
    prepared: PreparedWindow,
    last_used: u64,
}

/// The bounded per-owner cache of prepared topology skeletons (§2: 32
/// entries, evict only retired ones, patch retired storage in place).
#[derive(Default)]
pub struct WindowCache {
    entries: Vec<CacheEntry>,
    tick: u64,
}

impl WindowCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The prepared skeleton for `key`, building it with `build` on a miss.
    /// A miss at capacity evicts the least recently used **retired** entry;
    /// when none is retired the lookup is refused with
    /// [`Refusal::CacheFull`] *before* `build` runs.
    pub fn get_or_prepare(
        &mut self,
        key: &TopologyKey,
        build: impl FnOnce() -> Result<PreparedWindow, Refusal>,
    ) -> Result<&mut PreparedWindow, Refusal> {
        self.tick += 1;
        let tick = self.tick;
        if let Some(index) = self.entries.iter().position(|e| &e.key == key) {
            let entry = &mut self.entries[index];
            entry.last_used = tick;
            return Ok(&mut entry.prepared);
        }
        if self.entries.len() >= WINDOW_CACHE_CAPACITY {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.prepared.is_retired())
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .ok_or(Refusal::CacheFull)?;
            self.entries.swap_remove(victim);
        }
        let prepared = build()?;
        self.entries.push(CacheEntry {
            key: key.clone(),
            prepared,
            last_used: tick,
        });
        Ok(&mut self.entries.last_mut().expect("entry just pushed").prepared)
    }

    /// Drop every retired entry whose binding key is not `live` (an
    /// allocator or mapping change invalidated it); in-flight entries stay
    /// until they retire. Returns how many were dropped.
    pub fn evict_stale(&mut self, live: &WindowBindingKey) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|e| &e.key.binding == live || !e.prepared.is_retired());
        before - self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::tests::{launch, ptr};
    use crate::kernel::{
        author, ArgFact, ArgRole, Axis, DeclaredWord, HiddenArg, HiddenKind, MemClass, ScalarField,
    };
    use crate::plan::{plan, CacheTable, RowPolicy};

    const BASE_Q: u64 = 0x7000_0000;
    const BASE_OUT: u64 = 0x7001_0000;
    const EXTENT: u64 = 0x10000;
    /// Explicit end 24 + hidden (block_count_x/y, grid_dims) = 88.
    const SEGMENT: u32 = 88;

    fn hidden(offset: u32, size: u32, kind: HiddenKind, name: &str) -> HiddenArg {
        HiddenArg {
            offset,
            size,
            kind,
            value_kind: name.into(),
            loaded: false,
        }
    }

    fn integer(offset: u32) -> ArgFact {
        ArgFact {
            offset,
            size: 4,
            role: ArgRole::Integer,
        }
    }

    /// One node: q (+0, read), out (+8, written, authored interior 0x200),
    /// rows (+16) and tiles (+20) as Window words; hidden grid fields.
    fn program() -> KernelProgram {
        let mut l = launch(
            &[(0, BASE_Q + 0x100), (8, BASE_OUT + 0x200)],
            24,
            vec![
                ptr(0, &[MemClass::VmemLoad]),
                ptr(8, &[MemClass::VmemStore]),
                integer(16),
                integer(20),
            ],
        );
        l.grid = [2, 1, 1];
        l.block = [256, 1, 1];
        let mut f = (*l.facts.clone().unwrap()).clone();
        f.facts.segment_bytes = SEGMENT;
        f.facts.hidden = vec![
            hidden(24, 4, HiddenKind::BlockCount(0), "hidden_block_count_x"),
            hidden(28, 4, HiddenKind::BlockCount(1), "hidden_block_count_y"),
            hidden(32, 2, HiddenKind::GridDims, "hidden_grid_dims"),
        ];
        l.facts = Ok(Arc::new(f));
        l.declared = vec![
            DeclaredWord {
                offset: 16,
                role: WordRole::Window,
            },
            DeclaredWord {
                offset: 20,
                role: WordRole::Window,
            },
        ];
        author("gfx1201", &[l])
    }

    fn input(name: &str) -> Expr {
        Expr::input(name)
    }

    /// tiles = ceil((position + 1) / 128)
    fn tiles() -> Expr {
        Expr::ceil_div(Expr::add(input("position"), Expr::Const(1)), 128).unwrap()
    }

    fn patches() -> Vec<PatchKind> {
        vec![
            PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 16,
                },
                expr: input("projected_rows"),
                domain: 1..=16,
            },
            PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 20,
                },
                expr: tiles(),
                domain: 1..=1024,
            },
            PatchKind::Interior64 {
                node: 0,
                arg_offset: 8,
                resource: ResourceId(1),
                byte_offset: Expr::mul(input("position"), Expr::Const(4)),
                required_extent: Expr::Const(64),
            },
            PatchKind::GridAxis {
                node: 0,
                axis: Axis::X,
                expr: tiles(),
                domain: 1..=64,
            },
        ]
    }

    fn parts_with(patches: Vec<PatchKind>) -> WindowPlanParts {
        WindowPlanParts {
            program: program(),
            steps: vec![WindowStep::Kernel { node: 0 }],
            patches,
            host_delta: HostDeltaSpec {
                ring_head: Some(TransitionSpec {
                    from: input("ring_head"),
                    to: Expr::rem(Expr::add(input("ring_head"), input("verify_rows")), 64).unwrap(),
                }),
                ..HostDeltaSpec::default()
            },
            completion_surfaces: vec![ResourceId(1)],
            projection_rows: vec![3, 1],
            execution: WindowExecution::Whole16,
            hip_graph_certified: true,
            tape_populated: true,
        }
    }

    fn make_plan(parts: WindowPlanParts) -> Result<Arc<WindowPlan>, Refusal> {
        let p = parts;
        let table = CacheTable::for_arch("gfx1201").expect("gfx1201 table");
        let pm4 = plan(&p.program, &table, RowPolicy::UnreceiptedSystem);
        WindowPlan::new(p, &pm4).map(Arc::new)
    }

    fn key() -> WindowBindingKey {
        WindowBindingKey {
            model_id: "m".into(),
            draft_id: "d".into(),
            object_digests: vec!["aa".into()],
            toolchain_digest: "tc".into(),
            device: "gfx1201".into(),
            route_flags: 0,
            allocations: vec![
                AllocationIdentity {
                    resource: ResourceId(0),
                    base: BASE_Q,
                    bytes: EXTENT,
                    generation: 1,
                },
                AllocationIdentity {
                    resource: ResourceId(1),
                    base: BASE_OUT,
                    bytes: EXTENT,
                    generation: 1,
                },
            ],
            kv_mapping_generation: 7,
            scratch_capacities: vec![4096],
            context_mode: "sliding".into(),
            gdn_mode: "ef".into(),
            extraction_layers: vec![1, 2],
        }
    }

    #[derive(Default)]
    struct Log {
        submits: usize,
        waits: usize,
        last_kernarg: Vec<u8>,
        last_grid: [u32; 3],
        wait_failure: Option<Quiescence>,
        submit_failure: Option<Quiescence>,
    }

    struct MockQueue(Arc<Mutex<Log>>);

    impl WindowQueue for MockQueue {
        fn submit(
            &mut self,
            nodes: &[NodeImage<'_>],
            _steps: &[ResolvedStep],
        ) -> Result<(), QueueFailure> {
            let mut log = self.0.lock().unwrap();
            if let Some(q) = log.submit_failure {
                return Err(QueueFailure::new("submit", q));
            }
            log.submits += 1;
            log.last_kernarg = nodes[0].kernarg.to_vec();
            log.last_grid = nodes[0].grid;
            Ok(())
        }

        fn wait(&mut self) -> Result<(), QueueFailure> {
            let mut log = self.0.lock().unwrap();
            log.waits += 1;
            match log.wait_failure {
                Some(q) => Err(QueueFailure::new("wait", q)),
                None => Ok(()),
            }
        }
    }

    fn prepared_with(patches: Vec<PatchKind>) -> (PreparedWindow, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        let plan = make_plan(parts_with(patches)).expect("plan");
        let prepared =
            PreparedWindow::new(plan, key(), Box::new(MockQueue(log.clone()))).expect("prepared");
        (prepared, log)
    }

    fn prepared() -> (PreparedWindow, Arc<Mutex<Log>>) {
        prepared_with(patches())
    }

    #[derive(Default)]
    struct AsyncLog {
        blocking_submits: usize,
        nowait_submits: usize,
        waits: usize,
        in_flight: bool,
        wait_failure: Option<Quiescence>,
    }

    /// A queue whose `submit_nowait` returns with the work in flight and whose
    /// `wait` alone settles it; like the real queue it refuses a submit while
    /// one is outstanding. Its `submit` blocks (never in flight afterwards).
    struct AsyncQueue(Arc<Mutex<AsyncLog>>);

    impl AsyncQueue {
        fn refuse_in_flight(log: &AsyncLog) -> Result<(), QueueFailure> {
            if log.in_flight {
                return Err(QueueFailure::new(
                    "submit without completion",
                    Quiescence::Proven,
                ));
            }
            Ok(())
        }
    }

    impl WindowQueue for AsyncQueue {
        fn submit(
            &mut self,
            _nodes: &[NodeImage<'_>],
            _steps: &[ResolvedStep],
        ) -> Result<(), QueueFailure> {
            let mut log = self.0.lock().unwrap();
            Self::refuse_in_flight(&log)?;
            log.blocking_submits += 1;
            Ok(())
        }

        fn submit_nowait(
            &mut self,
            _nodes: &[NodeImage<'_>],
            _steps: &[ResolvedStep],
        ) -> Result<(), QueueFailure> {
            let mut log = self.0.lock().unwrap();
            Self::refuse_in_flight(&log)?;
            log.nowait_submits += 1;
            log.in_flight = true;
            Ok(())
        }

        fn wait(&mut self) -> Result<(), QueueFailure> {
            let mut log = self.0.lock().unwrap();
            log.waits += 1;
            log.in_flight = false;
            match log.wait_failure {
                Some(q) => Err(QueueFailure::new("wait", q)),
                None => Ok(()),
            }
        }
    }

    fn prepared_async() -> (PreparedWindow, Arc<Mutex<AsyncLog>>) {
        let log = Arc::new(Mutex::new(AsyncLog::default()));
        let plan = make_plan(parts_with(patches())).expect("plan");
        let prepared =
            PreparedWindow::new(plan, key(), Box::new(AsyncQueue(log.clone()))).expect("prepared");
        (prepared, log)
    }

    fn shape(position: u64, generation: u64) -> WindowShape {
        WindowShape {
            position,
            logical_ctx_rows: 2048,
            projected_rows: 3,
            full_cached_rows: 0,
            compact_offset: 0,
            ring_head: 60,
            swa_w: 2048,
            full_w: 0,
            max_position: 8191,
            max_tiles: 64,
            verify_rows: 16,
            draft_rows: 16,
            generation,
        }
    }

    fn inputs() -> WindowInputs {
        WindowInputs {
            seed: 11,
            mask_id: 2,
            q_positions: (0..16).map(|i| 300 + i).collect(),
            k_position_suffix: vec![298, 299],
            verify_positions: (0..16).map(|i| 300 + i).collect(),
            scalars: BTreeMap::new(),
            gdn_frames: BTreeMap::new(),
        }
    }

    fn bindings() -> WindowBindings {
        WindowBindings { key: key() }
    }

    /// The hand-built image of node 0 for `rows`, `tiles`, interior offset.
    fn expected_image(rows: u32, tiles: u32, interior: u64) -> Vec<u8> {
        let mut image = vec![0u8; SEGMENT as usize];
        image[0..8].copy_from_slice(&(BASE_Q + 0x100).to_le_bytes());
        image[8..16].copy_from_slice(&(BASE_OUT + interior).to_le_bytes());
        image[16..20].copy_from_slice(&rows.to_le_bytes());
        image[20..24].copy_from_slice(&tiles.to_le_bytes());
        image[24..28].copy_from_slice(&tiles.to_le_bytes()); // hidden_block_count_x
        image[28..32].copy_from_slice(&1u32.to_le_bytes()); // hidden_block_count_y
        image[32..34].copy_from_slice(&1u16.to_le_bytes()); // hidden_grid_dims
        image
    }

    #[test]
    fn kernarg_image_reconstructs_bytes_including_hidden_grid_fields() {
        let (mut w, _) = prepared();
        for (position, generation, tiles) in
            [(0u64, 1u64, 1u32), (127, 2, 1), (128, 3, 2), (300, 4, 3)]
        {
            let bound = w
                .bind(&shape(position, generation), &bindings(), &inputs())
                .unwrap();
            assert_eq!(bound.generation(), generation);
            assert_eq!(
                w.kernarg_image(0).unwrap(),
                expected_image(3, tiles, position * 4).as_slice(),
                "position {position}"
            );
            assert_eq!(w.grid(0), Some([tiles, 1, 1]));
            // consume the generation so the next bind's generation is fresh
            w.submit(bound).unwrap().complete().unwrap();
        }
        // The authored program, without overrides, is unchanged by patching.
        let authored = w
            .plan()
            .program()
            .kernarg_image(0, SEGMENT as usize, &w.plan().program().authored_words())
            .unwrap();
        assert_eq!(
            &authored[24..28],
            &2u32.to_le_bytes(),
            "authored hidden grid word"
        );
    }

    #[test]
    fn the_default_encoder_is_the_overrides_encoder_with_no_overrides() {
        let p = program();
        let words = p.authored_words();
        let mut into = vec![0xAAu8; SEGMENT as usize];
        p.encode_kernarg_into(0, &words, &ImageOverrides::default(), &mut into)
            .unwrap();
        assert_eq!(into, p.kernarg_image(0, SEGMENT as usize, &words).unwrap());
        assert!(p
            .encode_kernarg_into(0, &words, &ImageOverrides::default(), &mut [0u8; 8])
            .is_err());
        let zero_grid = ImageOverrides {
            grid: Some([0, 1, 1]),
            ..ImageOverrides::default()
        };
        assert!(p
            .encode_kernarg_into(0, &words, &zero_grid, &mut into)
            .is_err());
    }

    #[test]
    fn overflow_is_refused_before_work() {
        let (mut w, log) = prepared();
        let mut i = inputs();
        // projected_rows is a shape field; overflow through a hand patch.
        let (mut w2, _) = prepared_with({
            let mut p = patches();
            p[0] = PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 16,
                },
                expr: Expr::mul(input("big"), Expr::Const(4)),
                domain: 0..=u64::MAX,
            };
            p
        });
        i.scalars.insert("big".into(), u64::MAX / 2);
        let err = w2.bind(&shape(10, 1), &bindings(), &i).unwrap_err();
        assert!(matches!(err, Refusal::Overflow { .. }), "{err}");
        // A value that fits u64 but not u32.
        i.scalars.insert("big".into(), u64::from(u32::MAX));
        let err = w2.bind(&shape(10, 1), &bindings(), &i).unwrap_err();
        assert!(matches!(err, Refusal::DoesNotFit { .. }), "{err}");
        // The Interior64 sum overflows.
        let mut s = shape(10, 1);
        s.position = u64::MAX / 4 + 1;
        s.max_position = u64::MAX;
        let err = w
            .bind(
                &s,
                &bindings(),
                &WindowInputs {
                    q_positions: vec![0; 16],
                    verify_positions: vec![0; 16],
                    ..inputs()
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, Refusal::Overflow { .. } | Refusal::OutOfDomain { .. }),
            "{err}"
        );
        assert_eq!(log.lock().unwrap().submits, 0, "nothing was submitted");
    }

    #[test]
    fn out_of_domain_values_are_refused() {
        let (mut w, log) = prepared();
        let mut s = shape(300, 1);
        s.projected_rows = 17;
        let err = w.bind(&s, &bindings(), &inputs()).unwrap_err();
        assert!(
            matches!(err, Refusal::OutOfDomain { value: 17, .. }),
            "{err}"
        );
        s.projected_rows = 0;
        assert!(matches!(
            w.bind(&s, &bindings(), &inputs()).unwrap_err(),
            Refusal::OutOfDomain { value: 0, .. }
        ));
        // Grid beyond its domain: ceil((8191+1)/128) = 64 is fine, 65 is not.
        let mut s = shape(8191, 1);
        assert!(w
            .bind(
                &s,
                &bindings(),
                &WindowInputs {
                    q_positions: vec![8191; 16],
                    verify_positions: vec![8191; 16],
                    k_position_suffix: vec![],
                    ..inputs()
                }
            )
            .is_ok());
        s.position = 8192;
        s.max_position = 9000;
        let err = w
            .bind(
                &s,
                &bindings(),
                &WindowInputs {
                    q_positions: vec![8192; 16],
                    verify_positions: vec![8192; 16],
                    k_position_suffix: vec![],
                    ..inputs()
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, Refusal::OutOfDomain { value: 65, .. }),
            "{err}"
        );
        // Unknown input names refuse.
        let (mut u, _) = prepared_with({
            let mut p = patches();
            p[0] = PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 16,
                },
                expr: input("nonexistent"),
                domain: 0..=16,
            };
            p
        });
        assert!(matches!(
            u.bind(&shape(1, 1), &bindings(), &inputs()).unwrap_err(),
            Refusal::UnknownInput { .. }
        ));
        // Position beyond max_position, wrong row counts and Whole16 rows.
        let mut s = shape(300, 1);
        s.position = 9000;
        assert!(matches!(
            w.bind(&s, &bindings(), &inputs()).unwrap_err(),
            Refusal::Shape(_)
        ));
        let mut s = shape(300, 1);
        s.draft_rows = 15;
        assert!(matches!(
            w.bind(&s, &bindings(), &inputs()).unwrap_err(),
            Refusal::Shape(_)
        ));
        let mut i = inputs();
        i.verify_positions.pop();
        assert!(matches!(
            w.bind(&shape(300, 1), &bindings(), &i).unwrap_err(),
            Refusal::Inputs(_)
        ));
        let mut i = inputs();
        i.q_positions[3] = 9000;
        assert!(matches!(
            w.bind(&shape(300, 1), &bindings(), &i).unwrap_err(),
            Refusal::Inputs(_)
        ));
        assert_eq!(log.lock().unwrap().submits, 0);
    }

    #[test]
    fn interior_beyond_the_allocation_is_a_capacity_refusal() {
        let (mut w, _) = prepared();
        let mut s = shape(300, 1);
        // 0x10000 / 4 = 16384 positions fit; with the 64-byte extent the last
        // legal offset is EXTENT - 64.
        s.max_position = u64::MAX / 8;
        s.position = (EXTENT - 64) / 4;
        let mut i = inputs();
        i.q_positions = vec![0; 16];
        i.verify_positions = vec![0; 16];
        i.k_position_suffix.clear();
        // The tile count would exceed the grid domain first; widen the grid.
        let (mut w2, _) = prepared_with({
            let mut p = patches();
            p.pop();
            p
        });
        assert!(w2.bind(&s, &bindings(), &i).is_ok(), "exact fit");
        s.position += 1;
        s.generation = 2;
        let err = w2.bind(&s, &bindings(), &i).unwrap_err();
        assert!(matches!(err, Refusal::ExtentExceeded { .. }), "{err}");
        let _ = &mut w;
    }

    #[test]
    fn alias_and_capacity_mistakes_are_refused_at_prepare() {
        let plan = make_plan(parts_with(patches())).unwrap();
        let build = |mutate: &dyn Fn(&mut WindowBindingKey)| {
            let mut k = key();
            mutate(&mut k);
            let log = Arc::new(Mutex::new(Log::default()));
            PreparedWindow::new(plan.clone(), k, Box::new(MockQueue(log)))
                .err()
                .expect("refused")
        };
        // The output resource is written: overlapping it with the input aliases.
        assert!(matches!(
            build(&|k| k.allocations[1].base = BASE_Q + 0x1000),
            Refusal::Alias { a: 0, b: 1 }
        ));
        assert!(matches!(
            build(&|k| k.allocations[0].bytes = 0),
            Refusal::BadBinding { resource: 0, .. }
        ));
        assert!(matches!(
            build(&|k| k.allocations[1].base = u64::MAX - 8),
            Refusal::BadBinding { resource: 1, .. }
        ));
        assert!(matches!(
            build(&|k| {
                k.allocations.pop();
            }),
            Refusal::BadBinding { .. }
        ));
        // The unpatched input pointer (+0x100) must lie inside its allocation.
        assert!(matches!(
            build(&|k| k.allocations[0].bytes = 0x80),
            Refusal::ExtentExceeded { resource: 0, .. }
        ));
    }

    #[test]
    fn stale_generations_and_changed_bindings_are_refused() {
        let (mut w, log) = prepared();
        // Allocator reuse: same size, same base, new generation => new key.
        let mut reused = key();
        reused.allocations[1].generation = 2;
        let err = w
            .bind(&shape(300, 1), &WindowBindings { key: reused }, &inputs())
            .unwrap_err();
        assert_eq!(
            err,
            Refusal::BindingKeyMismatch("allocations[1].generation".into())
        );
        let mut moved = key();
        moved.allocations[0].base += 0x100000;
        assert!(matches!(
            w.bind(&shape(300, 1), &WindowBindings { key: moved }, &inputs())
                .unwrap_err(),
            Refusal::BindingKeyMismatch(_)
        ));
        let mut digest = key();
        digest.object_digests[0] = "bb".into();
        assert_eq!(
            w.bind(&shape(300, 1), &WindowBindings { key: digest }, &inputs())
                .unwrap_err(),
            Refusal::BindingKeyMismatch("object_digests".into())
        );
        let mut kv = key();
        kv.kv_mapping_generation += 1;
        assert_eq!(
            w.bind(&shape(300, 1), &WindowBindings { key: kv }, &inputs())
                .unwrap_err(),
            Refusal::BindingKeyMismatch("kv_mapping_generation".into())
        );
        // A submitted generation is consumed; the same or an older one is stale.
        let bound = w.bind(&shape(300, 5), &bindings(), &inputs()).unwrap();
        w.submit(bound).unwrap().complete().unwrap();
        for g in [5, 4] {
            assert_eq!(
                w.bind(&shape(300, g), &bindings(), &inputs()).unwrap_err(),
                Refusal::StaleGeneration { got: g, last: 5 }
            );
        }
        // A superseded BoundWindow cannot be submitted.
        let first = w.bind(&shape(300, 6), &bindings(), &inputs()).unwrap();
        let second = w.bind(&shape(300, 7), &bindings(), &inputs()).unwrap();
        assert_eq!(w.submit(first).err(), Some(Refusal::StaleBound));
        // A failed bind also invalidates the earlier bound window.
        assert!(w.bind(&shape(300, 1), &bindings(), &inputs()).is_err());
        assert_eq!(w.submit(second).err(), Some(Refusal::StaleBound));
        assert_eq!(log.lock().unwrap().submits, 1);
    }

    #[test]
    fn a_bound_window_of_another_prepared_window_is_refused() {
        let (mut a, _) = prepared();
        let (mut b, log_b) = prepared();
        let bound = a.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        assert_eq!(b.submit(bound).err(), Some(Refusal::StaleBound));
        assert_eq!(log_b.lock().unwrap().submits, 0);
    }

    #[test]
    fn patching_an_in_flight_window_is_refused() {
        let (mut w, log) = prepared();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit(bound).unwrap();
        assert!(!w.is_retired());
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::InFlight
        );
        assert_eq!(log.lock().unwrap().submits, 1);
        let kernarg = log.lock().unwrap().last_kernarg.clone();
        let done = submission.complete().unwrap();
        assert_eq!(
            done.host_delta.ring_head,
            Some(Transition {
                from: 60,
                to: (60 + 16) % 64
            })
        );
        assert_eq!(done.generation, 1);
        assert_eq!(
            done.receipt,
            WindowVerifyReceipt {
                same_as_hip_graph: true,
                tape_populated: true,
                generation: 1
            }
        );
        assert!(w.is_retired());
        // Retired: patched in place, and the earlier submit saw the old bytes.
        let bound = w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap();
        assert_eq!(kernarg, expected_image(3, 3, 1200));
        w.submit(bound).unwrap().complete().unwrap();
        assert_eq!(log.lock().unwrap().waits, 2);
        assert_eq!(log.lock().unwrap().last_grid, [3, 1, 1]);
    }

    #[test]
    fn unknown_quiescence_quarantines_and_proven_failure_retires() {
        let (mut w, log) = prepared();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let s = w.submit(bound).unwrap();
        log.lock().unwrap().wait_failure = Some(Quiescence::Unknown);
        let err = s.complete().unwrap_err();
        assert_eq!(err.quiescence, Quiescence::Unknown);
        assert!(!w.is_retired());
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::Quarantined
        );

        let (mut w, log) = prepared();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let s = w.submit(bound).unwrap();
        log.lock().unwrap().wait_failure = Some(Quiescence::Proven);
        assert_eq!(s.complete().unwrap_err().quiescence, Quiescence::Proven);
        assert!(w.is_retired());

        // A submit failure with unknown state quarantines too.
        let (mut w, log) = prepared();
        log.lock().unwrap().submit_failure = Some(Quiescence::Unknown);
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        assert!(matches!(
            w.submit(bound).err(),
            Some(Refusal::SubmitFailed {
                quiescence: Quiescence::Unknown,
                ..
            })
        ));
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::Quarantined
        );
    }

    #[test]
    fn submit_nowait_then_complete_retires_the_window() {
        // A queue that blocks in `submit`: `submit_nowait` is its `submit`.
        let (mut w, log) = prepared();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit_nowait(bound).unwrap();
        assert_eq!(log.lock().unwrap().submits, 1);
        assert!(!w.is_retired());
        let done = submission.complete().unwrap();
        assert_eq!(done.generation, 1);
        assert!(w.is_retired());
        assert_eq!(log.lock().unwrap().waits, 1);

        // A queue with its own `submit_nowait`: in flight until `complete`.
        let (mut w, log) = prepared_async();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit_nowait(bound).unwrap();
        {
            let l = log.lock().unwrap();
            assert_eq!((l.nowait_submits, l.blocking_submits, l.waits), (1, 0, 0));
            assert!(l.in_flight);
        }
        assert!(!w.is_retired());
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::InFlight
        );
        let done = submission.complete().unwrap();
        assert_eq!(done.generation, 1);
        assert!(w.is_retired());
        {
            let l = log.lock().unwrap();
            assert_eq!(l.waits, 1);
            assert!(!l.in_flight);
        }
        // Retired again: the blocking path still works on the same window.
        let bound = w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap();
        w.submit(bound).unwrap().complete().unwrap();
        let l = log.lock().unwrap();
        assert_eq!((l.nowait_submits, l.blocking_submits, l.waits), (1, 1, 2));
    }

    #[test]
    fn submit_without_completion_is_refused_before_the_queue_is_touched() {
        let (mut w, log) = prepared_async();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        // `bind` refuses while in flight, so the only way to hold a second
        // bound window of the live epoch is to build one here.
        let forge = |generation: u64| BoundWindow {
            prepared: bound.prepared,
            epoch: bound.epoch,
            generation,
            host_delta: bound.host_delta,
            indexed: None,
            tables: None,
            exact: false,
        };
        let (second_nowait, second_blocking) = (forge(2), forge(3));
        let submission = w.submit_nowait(bound).unwrap();
        assert_eq!(
            w.submit_nowait(second_nowait).err(),
            Some(Refusal::InFlight)
        );
        assert_eq!(w.submit(second_blocking).err(), Some(Refusal::InFlight));
        {
            let l = log.lock().unwrap();
            assert_eq!((l.nowait_submits, l.blocking_submits, l.waits), (1, 0, 0));
            assert!(l.in_flight);
        }
        // The refusals changed nothing: the first submission completes.
        assert_eq!(submission.complete().unwrap().generation, 1);
        assert!(w.is_retired());
    }

    #[test]
    fn async_wait_failure_decides_quarantine_at_complete() {
        let (mut w, log) = prepared_async();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit_nowait(bound).unwrap();
        // The queue was still running: its wait, not the submit, fails.
        log.lock().unwrap().wait_failure = Some(Quiescence::Unknown);
        assert_eq!(
            submission.complete().unwrap_err(),
            CompletionFailure {
                generation: 1,
                quiescence: Quiescence::Unknown,
                kind: FailureKind::Queue,
                message: "wait".into(),
            }
        );
        assert!(!w.is_retired());
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::Quarantined
        );

        let (mut w, log) = prepared_async();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit_nowait(bound).unwrap();
        log.lock().unwrap().wait_failure = Some(Quiescence::Proven);
        assert_eq!(
            submission.complete().unwrap_err().quiescence,
            Quiescence::Proven
        );
        assert!(w.is_retired());
    }

    #[test]
    fn queue_completion_waits_exactly_once_on_wait_or_drop() {
        let log = Arc::new(Mutex::new(AsyncLog::default()));
        let mut queue = AsyncQueue(log.clone());

        // `wait` returns the queue's result; the drop after it does not wait
        // again.
        let completion = submit_queue_nowait(&mut queue, &[], &[]).unwrap();
        {
            let l = log.lock().unwrap();
            assert!(l.in_flight);
            assert_eq!(l.waits, 0);
        }
        completion.wait().unwrap();
        {
            let l = log.lock().unwrap();
            assert!(!l.in_flight);
            assert_eq!(l.waits, 1);
        }
        log.lock().unwrap().wait_failure = Some(Quiescence::Unknown);
        let completion = submit_queue_nowait(&mut queue, &[], &[]).unwrap();
        assert_eq!(
            completion.wait().unwrap_err(),
            QueueFailure::new("wait", Quiescence::Unknown)
        );
        assert_eq!(log.lock().unwrap().waits, 2);

        // Dropped without `wait`: the drop waits, once, and the queue is no
        // longer in flight.
        log.lock().unwrap().wait_failure = None;
        let completion = submit_queue_nowait(&mut queue, &[], &[]).unwrap();
        assert!(log.lock().unwrap().in_flight);
        drop(completion);
        {
            let l = log.lock().unwrap();
            assert!(!l.in_flight);
            assert_eq!(l.waits, 3);
        }

        // A refused submit hands back no handle, so nothing waits.
        log.lock().unwrap().in_flight = true;
        let refused = submit_queue_nowait(&mut queue, &[], &[]).err().unwrap();
        assert_eq!(refused.quiescence, Quiescence::Proven);
        assert!(refused.message.contains("submit without completion"));
        assert_eq!(log.lock().unwrap().waits, 3);

        // Works through a trait object too.
        log.lock().unwrap().in_flight = false;
        let mut boxed: Box<dyn WindowQueue> = Box::new(AsyncQueue(log.clone()));
        submit_queue_nowait(&mut *boxed, &[], &[])
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(log.lock().unwrap().waits, 4);
    }

    #[test]
    fn plan_construction_leaves_no_untyped_holes() {
        // A Window word without its Scalar32.
        let mut p = patches();
        p.remove(1);
        assert!(
            matches!(make_plan(parts_with(p)), Err(Refusal::Malformed(m)) if m.contains("no Scalar32"))
        );
        // Scalar32 aimed at a pointer.
        let mut p = patches();
        p[0] = PatchKind::Scalar32 {
            field: ScalarField { node: 0, offset: 8 },
            expr: Expr::Const(1),
            domain: 0..=1,
        };
        assert!(matches!(
            make_plan(parts_with(p)),
            Err(Refusal::Malformed(_))
        ));
        // Interior64 aimed at the wrong resource.
        let mut p = patches();
        p[2] = PatchKind::Interior64 {
            node: 0,
            arg_offset: 8,
            resource: ResourceId(0),
            byte_offset: Expr::Const(0),
            required_extent: Expr::Const(1),
        };
        assert!(matches!(
            make_plan(parts_with(p)),
            Err(Refusal::Malformed(_))
        ));
        // Duplicate sites and a zero-based grid domain.
        let mut p = patches();
        p.push(p[3].clone());
        assert!(matches!(
            make_plan(parts_with(p)),
            Err(Refusal::Malformed(_))
        ));
        let mut p = patches();
        p[3] = PatchKind::GridAxis {
            node: 0,
            axis: Axis::X,
            expr: tiles(),
            domain: 0..=4,
        };
        assert!(matches!(
            make_plan(parts_with(p)),
            Err(Refusal::Malformed(_))
        ));
        // Steps must dispatch every node in order; completion surfaces must be written.
        let mut parts = parts_with(patches());
        parts.steps.clear();
        assert!(matches!(make_plan(parts), Err(Refusal::Malformed(_))));
        let mut parts = parts_with(patches());
        parts.completion_surfaces = vec![ResourceId(0)];
        assert!(matches!(make_plan(parts), Err(Refusal::Malformed(_))));
        // A HipDirect node is not admissible.
        let mut parts = parts_with(patches());
        parts.program.nodes[0].lowering = crate::kernel::Lowering::HipDirect { reason: "x".into() };
        assert!(matches!(make_plan(parts), Err(Refusal::NotPm4 { .. })));
        // A lab-pending node is not PM4 either: the product window refuses it.
        let mut parts = parts_with(patches());
        parts.program.nodes[0].lowering = crate::kernel::Lowering::LabPending {
            pending: "A1 Unknown (x)".into(),
        };
        assert!(matches!(make_plan(parts), Err(Refusal::NotPm4 { .. })));
        // The plan's closure and family.
        let plan = make_plan(parts_with(patches())).unwrap();
        assert!(plan.closure().reads.contains(&ResourceId(0)));
        assert!(plan.closure().writes.contains(&ResourceId(1)));
        assert_eq!(plan.family().projection_rows, vec![3, 1]);
        assert_eq!(plan.family().kernels.len(), 1);
    }

    #[test]
    fn gdn_frames_must_be_supplied_for_every_gdn_word() {
        let mut l = launch(
            &[(0, BASE_Q + 0x100)],
            12,
            vec![ptr(0, &[MemClass::VmemLoad]), integer(8)],
        );
        l.declared = vec![DeclaredWord {
            offset: 8,
            role: WordRole::GdnFrame { frames: 1 },
        }];
        let program = author("gfx1201", &[l]);
        let table = CacheTable::for_arch("gfx1201").unwrap();
        let pm4 = plan(&program, &table, RowPolicy::UnreceiptedSystem);
        let parts = WindowPlanParts {
            program,
            steps: vec![WindowStep::Kernel { node: 0 }],
            patches: vec![],
            host_delta: HostDeltaSpec::default(),
            completion_surfaces: vec![],
            projection_rows: vec![],
            execution: WindowExecution::HostSplice,
            hip_graph_certified: false,
            tape_populated: false,
        };
        let plan = Arc::new(WindowPlan::new(parts, &pm4).unwrap());
        let mut k = key();
        k.allocations.truncate(1);
        let log = Arc::new(Mutex::new(Log::default()));
        let mut w = PreparedWindow::new(plan, k.clone(), Box::new(MockQueue(log))).unwrap();
        let b = WindowBindings { key: k };
        let mut s = shape(1, 1);
        s.verify_rows = 5;
        let mut i = WindowInputs {
            q_positions: vec![0; 5],
            verify_positions: vec![0; 5],
            ..WindowInputs::default()
        };
        assert!(
            matches!(w.bind(&s, &b, &i).unwrap_err(), Refusal::Inputs(m) if m.contains("GdnFrame"))
        );
        i.gdn_frames.insert(WordId(0), 41);
        w.bind(&s, &b, &i).unwrap();
        assert_eq!(&w.kernarg_image(0).unwrap()[8..12], &41u32.to_le_bytes());
        i.gdn_frames.insert(WordId(9), 1);
        s.generation = 2;
        assert!(matches!(
            w.bind(&s, &b, &i).unwrap_err(),
            Refusal::Inputs(_)
        ));
    }

    #[test]
    fn copy_and_fill_steps_are_bounded_aligned_and_disjoint() {
        let mut parts = parts_with(patches());
        parts.steps.push(WindowStep::Copy(CopyStep {
            src: Region {
                resource: ResourceId(0),
                byte_offset: Expr::Const(0),
            },
            dst: Region {
                resource: ResourceId(1),
                byte_offset: input("compact_offset"),
            },
            bytes: Expr::mul(input("projected_rows"), Expr::Const(16)),
        }));
        parts.steps.push(WindowStep::Fill(FillStep {
            dst: Region {
                resource: ResourceId(1),
                byte_offset: Expr::Const(0x8000),
            },
            value: 0,
            words: Expr::Const(16),
        }));
        let log = Arc::new(Mutex::new(Log::default()));
        let plan = make_plan(parts).unwrap();
        let mut w = PreparedWindow::new(plan, key(), Box::new(MockQueue(log))).unwrap();
        let mut s = shape(300, 1);
        s.compact_offset = 0x40;
        w.bind(&s, &bindings(), &inputs()).unwrap();
        assert_eq!(
            w.resolved_steps(),
            &[
                ResolvedStep::Copy {
                    src: BASE_Q,
                    dst: BASE_OUT + 0x40,
                    bytes: 48
                },
                ResolvedStep::Fill {
                    dst: BASE_OUT + 0x8000,
                    value: 0,
                    words: 16
                },
            ]
        );
        s.compact_offset = EXTENT - 47; // 48 bytes no longer fit
        assert!(matches!(
            w.bind(&s, &bindings(), &inputs()).unwrap_err(),
            Refusal::ExtentExceeded { .. }
        ));
        // Same-resource overlap is refused (source and destination in resource 0).
        let mut parts = parts_with(patches());
        parts.steps.push(WindowStep::Copy(CopyStep {
            src: Region {
                resource: ResourceId(0),
                byte_offset: Expr::Const(0),
            },
            dst: Region {
                resource: ResourceId(0),
                byte_offset: Expr::Const(8),
            },
            bytes: Expr::Const(16),
        }));
        let log = Arc::new(Mutex::new(Log::default()));
        let mut w = PreparedWindow::new(make_plan(parts).unwrap(), key(), Box::new(MockQueue(log)))
            .unwrap();
        assert!(matches!(
            w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap_err(),
            Refusal::BadStep { .. }
        ));
    }

    #[test]
    fn the_cache_holds_32_and_evicts_only_retired_entries() {
        let mut cache = WindowCache::new();
        let plan = make_plan(parts_with(patches())).unwrap();
        let topo = |n: u64| {
            let mut k = key();
            k.route_flags = n;
            TopologyKey {
                binding: k,
                family: plan.family().clone(),
            }
        };
        let build = |t: &TopologyKey| {
            PreparedWindow::new(
                plan.clone(),
                t.binding.clone(),
                Box::new(MockQueue(Arc::new(Mutex::new(Log::default())))),
            )
        };
        // Fill the cache; keep every window in flight.
        let mut subs = Vec::new();
        for n in 0..WINDOW_CACHE_CAPACITY as u64 {
            let t = topo(n);
            let w = cache.get_or_prepare(&t, || build(&t)).unwrap();
            let b = w
                .bind(
                    &shape(300, 1),
                    &WindowBindings {
                        key: t.binding.clone(),
                    },
                    &inputs(),
                )
                .unwrap();
            subs.push(w.submit(b).unwrap());
        }
        assert_eq!(cache.len(), WINDOW_CACHE_CAPACITY);
        let extra = topo(1000);
        let mut built = false;
        let err = cache
            .get_or_prepare(&extra, || {
                built = true;
                build(&extra)
            })
            .err()
            .unwrap();
        assert_eq!(err, Refusal::CacheFull);
        assert!(!built, "refused before building a plan");
        // A hit on an in-flight entry still resolves.
        assert!(cache.get_or_prepare(&topo(3), || unreachable!()).is_ok());
        // Retire entry 5: it is the only evictable one.
        subs.remove(5).complete().unwrap();
        cache.get_or_prepare(&extra, || build(&extra)).unwrap();
        assert_eq!(cache.len(), WINDOW_CACHE_CAPACITY);
        assert_eq!(cache.len(), WINDOW_CACHE_CAPACITY);
        // Stale eviction only drops retired entries of another binding key.
        let live = topo(0).binding;
        assert_eq!(
            cache.evict_stale(&live),
            1,
            "only the never-bound extra is retired"
        );
        for s in subs {
            s.complete().unwrap();
        }
        assert_eq!(cache.evict_stale(&live), WINDOW_CACHE_CAPACITY - 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn expressions_follow_the_section_3_arithmetic() {
        let env = |name: &str| match name {
            "a" => Some(10u64),
            "b" => Some(3),
            "max" => Some(u64::MAX),
            _ => None,
        };
        let e = |x: Expr| x.eval(&env);
        assert_eq!(e(Expr::sub_sat(input("b"), input("a"))), Ok(0));
        assert_eq!(e(Expr::sub_sat(input("a"), input("b"))), Ok(7));
        assert_eq!(e(Expr::min(input("a"), input("b"))), Ok(3));
        assert_eq!(e(Expr::max(input("a"), input("b"))), Ok(10));
        assert_eq!(e(Expr::ceil_div(input("a"), 3).unwrap()), Ok(4));
        assert_eq!(
            e(Expr::ceil_div(input("max"), 2).unwrap()),
            Ok(u64::MAX / 2 + 1),
            "no a+c-1 wrap"
        );
        assert_eq!(e(Expr::div(input("a"), 4).unwrap()), Ok(2));
        assert_eq!(e(Expr::rem(input("a"), 4).unwrap()), Ok(2));
        assert_eq!(Expr::div(input("a"), 0), None);
        assert_eq!(Expr::rem(input("a"), 0), None);
        assert_eq!(Expr::ceil_div(input("a"), 0), None);
        assert_eq!(
            e(Expr::add(input("max"), Expr::Const(1))),
            Err(ExprError::Overflow)
        );
        assert_eq!(
            e(Expr::mul(input("max"), Expr::Const(2))),
            Err(ExprError::Overflow)
        );
        assert_eq!(e(input("zzz")), Err(ExprError::UnknownInput("zzz".into())));
        let mut names = Vec::new();
        let expr = Expr::add(input("a"), Expr::mul(input("b"), input("a")));
        expr.inputs(&mut names);
        assert_eq!(names, ["a", "b"]);
    }

    #[test]
    fn inputs_pack_the_live_positions_in_order() {
        let i = WindowInputs {
            seed: 1,
            mask_id: 2,
            q_positions: vec![3, 4],
            k_position_suffix: vec![5],
            verify_positions: vec![6, 7],
            ..WindowInputs::default()
        };
        let words: Vec<u32> = i
            .pack()
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words, [1, 2, 3, 4, 5, 6, 7]);
        let mut shadow = WindowInputs {
            q_positions: vec![0; 16],
            verify_positions: vec![0; 16],
            ..inputs()
        };
        shadow.scalars.insert("position".into(), 1);
        let (mut w, _) = prepared();
        assert!(matches!(
            w.bind(&shape(300, 1), &bindings(), &shadow).unwrap_err(),
            Refusal::Inputs(m) if m.contains("shadows")
        ));
    }

    // ---- §A9.4 sealed positions through the window (tests 7 and 8) ----------

    use crate::index_bind::tests as ib;
    use crate::index_bind::{IndexBindError, PositionsState, SealedPositions};

    fn indexed_parts(fx: &ib::Fixture) -> WindowPlanParts {
        WindowPlanParts {
            program: fx.program.clone(),
            steps: fx.steps.clone(),
            patches: Vec::new(),
            host_delta: HostDeltaSpec::default(),
            completion_surfaces: vec![ResourceId(0), ResourceId(3)],
            projection_rows: vec![],
            execution: WindowExecution::Whole16,
            hip_graph_certified: false,
            tape_populated: false,
        }
    }

    fn indexed_key(fx: &ib::Fixture) -> WindowBindingKey {
        WindowBindingKey {
            allocations: fx.allocations.clone(),
            object_digests: vec!["aa".into(); fx.program.nodes.len()],
            ..key()
        }
    }

    fn indexed_window() -> (ib::Fixture, PreparedWindow, Arc<Mutex<Log>>) {
        let fx = ib::Fixture::new(Vec::new());
        let log = Arc::new(Mutex::new(Log::default()));
        let plan =
            make_plan(indexed_parts(&fx)).expect("a conditional plan under assumed witnesses");
        assert_eq!(plan.indexed_positions(), Some(fx.p_res));
        let w = PreparedWindow::new(plan, indexed_key(&fx), Box::new(MockQueue(log.clone())))
            .expect("prepared");
        (fx, w, log)
    }

    fn indexed_bind(w: &mut PreparedWindow, fx: &ib::Fixture, generation: u64) -> BoundWindow {
        let bindings = WindowBindings {
            key: indexed_key(fx),
        };
        w.bind(&shape(300, generation), &bindings, &inputs())
            .expect("bind")
    }

    #[test]
    fn a_conditional_node_is_not_pm4_without_its_witness_and_submit_refuses() {
        // The cached program (nodes still `Conditional`) never plans as PM4.
        let fx = ib::Fixture::new(Vec::new());
        let mut parts = indexed_parts(&fx);
        parts.program = {
            let mut p = fx.program.clone();
            for n in &mut p.nodes {
                n.lowering = crate::kernel::Lowering::Conditional { reason: "w".into() };
            }
            p
        };
        assert!(matches!(
            make_plan(parts),
            Err(Refusal::NotPm4 { node: 0, .. })
        ));

        let (fx, mut w, log) = indexed_window();
        let bound = indexed_bind(&mut w, &fx, 1);
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::IndexedWitnessMissing { node: 0, .. })
        ));
        assert_eq!(
            log.lock().unwrap().submits,
            0,
            "the queue was never touched"
        );
    }

    #[test]
    fn bind_attach_submit_complete_walks_the_sealed_lifecycle() {
        let (fx, mut w, log) = indexed_window();
        let sealed = ib::ready(&ib::contiguous(300));
        let mut bound = indexed_bind(&mut w, &fx, 1);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap();
        assert_eq!(sealed.state(), PositionsState::SealedReady);
        let submission = w.submit(bound).unwrap();
        assert_eq!(sealed.state(), PositionsState::SubmittedReadOnly);
        assert_eq!(log.lock().unwrap().submits, 1);
        // While in flight nothing may touch P.
        assert!(sealed.overwrite().is_err() && sealed.recycle().is_err());
        submission.complete().unwrap();
        assert_eq!(sealed.state(), PositionsState::Retired);

        // The next window cannot reuse the previous witness or sealed bytes:
        // it needs its own check, upload and bind.
        let mut bound = indexed_bind(&mut w, &fx, 2);
        assert!(matches!(
            w.attach_indexed_witness(&mut bound, &sealed, &fx.extents),
            Err(Refusal::Indexed(IndexBindError::State {
                found: PositionsState::Retired,
                ..
            }))
        ));
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::IndexedWitnessMissing { .. })
        ));
    }

    #[test]
    fn a_post_bind_kernarg_or_grid_change_and_a_superseded_bind_are_refused_at_submit() {
        let (fx, mut w, log) = indexed_window();
        let sealed = ib::ready(&ib::contiguous(300));
        let mut bound = indexed_bind(&mut w, &fx, 1);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap();
        // Patch the bound kernarg image of the V writer after the bind.
        w.images[1][24] ^= 1;
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::Indexed(IndexBindError::Writer { node: 1, .. }))
        ));
        assert_eq!(
            sealed.state(),
            PositionsState::SealedReady,
            "nothing was submitted"
        );
        assert_eq!(log.lock().unwrap().submits, 0);

        // A shape/grid change after bind.
        let mut bound = indexed_bind(&mut w, &fx, 2);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap();
        w.grids[0] = [ib::H, 16, 2];
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::Indexed(IndexBindError::Writer { node: 0, .. }))
        ));

        // A later bind supersedes the attached one.
        let mut stale = indexed_bind(&mut w, &fx, 3);
        w.attach_indexed_witness(&mut stale, &sealed, &fx.extents)
            .unwrap();
        let _fresh = indexed_bind(&mut w, &fx, 4);
        assert!(matches!(w.submit(stale), Err(Refusal::StaleBound)));
        sealed.release_unsubmitted().unwrap();

        // A witness bound to another payload/generation: re-seal then submit.
        let mut bound = indexed_bind(&mut w, &fx, 5);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap_err();
        ib::seal(&sealed, &ib::contiguous(300));
        let mut bound = indexed_bind(&mut w, &fx, 6);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap();
        w.submit(bound).unwrap().complete().unwrap();
    }

    #[test]
    fn unknown_quiescence_after_submit_quarantines_the_sealed_positions() {
        let (fx, mut w, log) = indexed_window();
        let sealed: SealedPositions = ib::ready(&ib::contiguous(300));
        let mut bound = indexed_bind(&mut w, &fx, 1);
        w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
            .unwrap();
        let submission = w.submit(bound).unwrap();
        log.lock().unwrap().wait_failure = Some(Quiescence::Unknown);
        assert!(submission.complete().is_err());
        assert_eq!(sealed.state(), PositionsState::Quarantined);
        assert!(sealed.recycle().is_err() && sealed.overwrite().is_err());
        assert!(matches!(
            w.bind(
                &shape(300, 2),
                &WindowBindings {
                    key: indexed_key(&fx)
                },
                &inputs()
            ),
            Err(Refusal::Quarantined)
        ));

        // A submit failure with unknown quiescence quarantines too; a proven
        // one retires the positions (they must be re-checked and re-uploaded).
        for (q, state) in [
            (Quiescence::Unknown, PositionsState::Quarantined),
            (Quiescence::Proven, PositionsState::Retired),
        ] {
            let (fx, mut w, log) = indexed_window();
            let sealed = ib::ready(&ib::contiguous(300));
            let mut bound = indexed_bind(&mut w, &fx, 1);
            w.attach_indexed_witness(&mut bound, &sealed, &fx.extents)
                .unwrap();
            log.lock().unwrap().submit_failure = Some(q);
            assert!(matches!(w.submit(bound), Err(Refusal::SubmitFailed { .. })));
            assert_eq!(sealed.state(), state);
        }
    }

    #[test]
    fn a_conditional_plan_whose_nodes_index_different_positions_is_malformed() {
        let mut fx = ib::Fixture::new(Vec::new());
        // Node 1 indexes the V resource as its "positions".
        let alt = fx.program.nodes[1]
            .args
            .iter()
            .find_map(|(o, a)| match a {
                Arg::Resource { resource, .. } if *o == 0 => Some(*resource),
                _ => None,
            })
            .unwrap();
        for (o, a) in &mut fx.program.nodes[1].args {
            if *o == 16 {
                *a = Arg::Resource {
                    resource: alt,
                    interior: 0,
                };
            }
        }
        assert!(
            matches!(make_plan(indexed_parts(&fx)), Err(Refusal::Malformed(m)) if m.contains("another conditional node"))
        );
    }

    // ---- sealed pointer tables through the window ---------------------------

    use crate::kernel::LaunchRecord;
    use crate::sealed_table::tests as st;

    fn sealed_parts(fx: &st::Fixture) -> WindowPlanParts {
        WindowPlanParts {
            program: fx.program.clone(),
            steps: fx.steps.clone(),
            patches: Vec::new(),
            host_delta: HostDeltaSpec::default(),
            completion_surfaces: vec![fx.out_res],
            projection_rows: vec![],
            execution: WindowExecution::Whole16,
            hip_graph_certified: false,
            tape_populated: false,
        }
    }

    fn sealed_key(fx: &st::Fixture) -> WindowBindingKey {
        WindowBindingKey {
            allocations: fx.allocations.clone(),
            object_digests: vec!["aa".into(); fx.program.nodes.len()],
            ..key()
        }
    }

    fn sealed_window(extra: Vec<LaunchRecord>) -> (st::Fixture, PreparedWindow, Arc<Mutex<Log>>) {
        sealed_window_of(st::Fixture::new(extra))
    }

    fn sealed_window_of(fx: st::Fixture) -> (st::Fixture, PreparedWindow, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        let plan =
            make_plan(sealed_parts(&fx)).expect("a sealed-table plan under the assumed witness");
        let w = PreparedWindow::new(plan, sealed_key(&fx), Box::new(MockQueue(log.clone())))
            .expect("prepared");
        (fx, w, log)
    }

    fn sealed_bind(w: &mut PreparedWindow, fx: &st::Fixture, generation: u64) -> BoundWindow {
        let bindings = WindowBindings {
            key: sealed_key(fx),
        };
        w.bind(&shape(300, generation), &bindings, &inputs())
            .expect("bind")
    }

    fn attach(
        w: &PreparedWindow,
        bound: &mut BoundWindow,
        fx: &st::Fixture,
    ) -> Result<(), Refusal> {
        w.attach_sealed_table_witness(bound, &fx.registry, &fx.content)
    }

    #[test]
    fn a_sealed_table_node_is_not_pm4_without_its_witness_and_submit_refuses() {
        // The cached program (the node still `Conditional`) never plans as PM4.
        let fx = st::Fixture::new(Vec::new());
        let mut parts = sealed_parts(&fx);
        parts.program = fx.authored.clone();
        assert!(matches!(
            make_plan(parts),
            Err(Refusal::NotPm4 { node: 0, reason, .. }) if reason.contains("sealed_table witness not consumed")
        ));

        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut bound = sealed_bind(&mut w, &fx, 1);
        assert!(matches!(
            w.certify_exact(&mut bound),
            Err(Refusal::SealedTableWitnessMissing { node: 0, .. })
        ));
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTableWitnessMissing { node: 0, .. })
        ));
        assert_eq!(
            log.lock().unwrap().submits,
            0,
            "the queue was never touched"
        );

        // A plan with no sealed-table node has nothing to attach.
        let (ifx, mut iw, _) = indexed_window();
        let mut bound = indexed_bind(&mut iw, &ifx, 1);
        assert!(matches!(
            iw.attach_sealed_table_witness(&mut bound, &fx.registry, &fx.content),
            Err(Refusal::SealedTable(SealedTableError::Closure(_)))
        ));
    }

    #[test]
    fn bind_attach_submit_complete_walks_the_table_lease() {
        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        let witness = bound.sealed_table_witness().expect("attached");
        assert_eq!(
            witness.discharged_cases(),
            vec![(0, fx.case_digest.as_str(), "sealed_table")]
        );
        w.certify_exact(&mut bound).unwrap();
        let submission = w.submit(bound).unwrap();
        assert_eq!(log.lock().unwrap().submits, 1);
        // While the window is in flight the table cannot be retired.
        assert!(matches!(
            fx.registry.retire(st::TABLE),
            Err(SealedTableError::Retire(m)) if m.contains("in flight")
        ));
        let done = submission.complete().unwrap();
        assert!(done.receipt.same_as_hip_graph, "certified with the witness");
        // Released: the next window binds anew and the table can then be retired.
        let mut next = sealed_bind(&mut w, &fx, 2);
        attach(&w, &mut next, &fx).unwrap();
        w.submit(next).unwrap().complete().unwrap();
        assert_eq!(fx.registry.retire(st::TABLE), Ok(st::GEN + 1));
    }

    /// The owner flow for a table that holds null entries (an absent expert),
    /// end to end: seal with `NullEntries::Allow` (the receipt's `null_guarded`),
    /// bind, consume the witness, certify, submit. The same table under a
    /// certificate without a null guard never gets a witness.
    #[test]
    fn a_null_holding_table_is_consumed_only_under_a_null_guarded_certificate() {
        let (fx, mut w, log) = sealed_window_of(st::Fixture::nullable(&st::guarded_record()));
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).expect("null-guarded certificate");
        assert_eq!(
            bound
                .sealed_table_witness()
                .expect("attached")
                .discharged_cases(),
            vec![(0, fx.case_digest.as_str(), "sealed_table")]
        );
        w.certify_exact(&mut bound).unwrap();
        w.submit(bound).unwrap().complete().unwrap();
        assert_eq!(log.lock().unwrap().submits, 1);

        let (fx, mut w, log) = sealed_window_of(st::Fixture::nullable(&st::record()));
        let mut bound = sealed_bind(&mut w, &fx, 1);
        assert!(matches!(
            attach(&w, &mut bound, &fx),
            Err(Refusal::SealedTable(
                SealedTableError::NullEntryUnguarded { .. }
            ))
        ));
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTableWitnessMissing { node: 0, .. })
        ));
        assert_eq!(log.lock().unwrap().submits, 0);
    }

    /// A real table node and a zero-bound (vector mode) node in one window:
    /// both are discharged, and raising the vacuous node's bound after the bind
    /// voids the witness at submit.
    #[test]
    fn a_vacuous_sealed_node_is_discharged_and_its_bound_is_rechecked_at_submit() {
        let k = st::record();
        let (fx, mut w, log) = sealed_window(vec![st::vector_mode_launch(&k)]);
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        let cases = bound
            .sealed_table_witness()
            .expect("attached")
            .discharged_cases();
        assert_eq!(cases.iter().map(|c| c.0).collect::<Vec<_>>(), vec![0, 1]);
        w.certify_exact(&mut bound).unwrap();
        w.submit(bound).unwrap().complete().unwrap();

        let (fx, mut w, log2) = sealed_window(vec![st::vector_mode_launch(&k)]);
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        w.images[1][16..20].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTable(SealedTableError::Node { node: 1, .. }))
        ));
        assert_eq!(log2.lock().unwrap().submits, 0);
        assert_eq!(log.lock().unwrap().submits, 1);
    }

    #[test]
    fn unknown_quiescence_quarantines_the_table_and_a_proven_failure_releases_it() {
        // Unknown after submit: the window may still read the table.
        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        let submission = w.submit(bound).unwrap();
        log.lock().unwrap().wait_failure = Some(Quiescence::Unknown);
        assert!(submission.complete().is_err());
        assert!(matches!(
            fx.registry.retire(st::TABLE),
            Err(SealedTableError::Retire(m)) if m.contains("quarantined")
        ));

        for (q, retirable) in [(Quiescence::Proven, true), (Quiescence::Unknown, false)] {
            let (fx, mut w, log) = sealed_window(Vec::new());
            let mut bound = sealed_bind(&mut w, &fx, 1);
            attach(&w, &mut bound, &fx).unwrap();
            log.lock().unwrap().submit_failure = Some(q);
            assert!(matches!(w.submit(bound), Err(Refusal::SubmitFailed { .. })));
            assert_eq!(fx.registry.retire(st::TABLE).is_ok(), retirable, "{q:?}");
        }
    }

    #[test]
    fn a_modified_table_a_stale_generation_and_a_writer_node_are_refused() {
        // The live table differs from the sealed digest: no witness.
        let (fx, mut w, log) = sealed_window(Vec::new());
        fx.content.bytes.lock().unwrap()[0] ^= 1;
        let mut bound = sealed_bind(&mut w, &fx, 1);
        assert!(matches!(
            attach(&w, &mut bound, &fx),
            Err(Refusal::SealedTable(SealedTableError::Modified { .. }))
        ));
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTableWitnessMissing { .. })
        ));
        assert_eq!(log.lock().unwrap().submits, 0);

        // Retired and re-sealed between attach and submit: the witness is
        // stale and nothing is queued.
        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        fx.registry.retire(st::TABLE).unwrap();
        fx.registry
            .seal(
                st::table_binding(),
                st::ENTRIES,
                &st::table_bytes(&st::entry_ptrs()),
                &st::pointees(),
                st::GEN + 1,
            )
            .unwrap();
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTable(SealedTableError::Stale(_)))
        ));
        assert_eq!(log.lock().unwrap().submits, 0);

        // A kernarg image or grid changed after the bind.
        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut bound = sealed_bind(&mut w, &fx, 1);
        attach(&w, &mut bound, &fx).unwrap();
        w.images[0][16] ^= 1;
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTable(SealedTableError::Node { node: 0, .. }))
        ));
        let mut bound = sealed_bind(&mut w, &fx, 2);
        attach(&w, &mut bound, &fx).unwrap();
        w.grids[0] = [9, 1, 1];
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTable(SealedTableError::Node { node: 0, .. }))
        ));
        assert_eq!(log.lock().unwrap().submits, 0);

        // A program that can write a pointee cannot obtain a witness, so the
        // node is never submitted.
        let (fx, mut w, log) = sealed_window(vec![ib::extra_launch(
            "writer",
            st::W1 + 0x40,
            st::w1(),
            &[MemClass::VmemStore],
        )]);
        let mut bound = sealed_bind(&mut w, &fx, 1);
        assert!(matches!(
            attach(&w, &mut bound, &fx),
            Err(Refusal::SealedTable(SealedTableError::Closure(m))) if m.contains("pointee")
        ));
        assert!(matches!(
            w.submit(bound),
            Err(Refusal::SealedTableWitnessMissing { node: 0, .. })
        ));
        assert_eq!(log.lock().unwrap().submits, 0);
    }

    #[test]
    fn a_superseded_bind_refuses_the_table_attach_and_submit() {
        let (fx, mut w, log) = sealed_window(Vec::new());
        let mut first = sealed_bind(&mut w, &fx, 1);
        let mut second = sealed_bind(&mut w, &fx, 2);
        assert!(matches!(
            attach(&w, &mut first, &fx),
            Err(Refusal::StaleBound)
        ));
        attach(&w, &mut second, &fx).unwrap();
        let mut third = sealed_bind(&mut w, &fx, 3);
        attach(&w, &mut third, &fx).unwrap();
        assert!(matches!(w.submit(second), Err(Refusal::StaleBound)));
        assert_eq!(log.lock().unwrap().submits, 0);
        w.submit(third).unwrap().complete().unwrap();
    }

    // ------------------------------------------------ persistent cancel

    type FakeDevice = Arc<Mutex<Option<String>>>;

    #[derive(Default)]
    struct CancelLog {
        blocking_submits: usize,
        nowait_submits: usize,
        waits: usize,
        in_flight: bool,
        timeouts: Vec<Duration>,
        poison_writes: Vec<(u64, u32)>,
        teardowns: Vec<Duration>,
        poisoned: bool,
        torn_down: bool,
        /// The program retires on its own at the first bounded wait.
        completes_on_own: bool,
        /// A worker that polls the poison cell and exits.
        exits_on_poison: bool,
        /// The poison write itself fails.
        poison_fails: bool,
        /// What the teardown proves (default `Unknown`).
        teardown_proves: Option<Quiescence>,
    }

    /// A persistent program that never completes on its own: only the poison
    /// (when `exits_on_poison`) or a teardown ends it.
    struct CancelQueue {
        log: Arc<Mutex<CancelLog>>,
        device: FakeDevice,
    }

    impl CancelQueue {
        fn refuse(&self, log: &CancelLog) -> Result<(), QueueFailure> {
            if let Some(reason) = self.device.lock().unwrap().clone() {
                return Err(QueueFailure::new(reason, Quiescence::Proven)
                    .with_kind(FailureKind::DeviceQuarantined));
            }
            if log.torn_down {
                return Err(QueueFailure::new("queue torn down", Quiescence::Proven)
                    .with_kind(FailureKind::TornDown));
            }
            if log.in_flight {
                return Err(QueueFailure::new(
                    "submit without completion",
                    Quiescence::Proven,
                ));
            }
            Ok(())
        }
    }

    impl WindowQueue for CancelQueue {
        fn submit(
            &mut self,
            _nodes: &[NodeImage<'_>],
            _steps: &[ResolvedStep],
        ) -> Result<(), QueueFailure> {
            let mut log = self.log.lock().unwrap();
            self.refuse(&log)?;
            log.blocking_submits += 1;
            Ok(())
        }

        fn submit_nowait(
            &mut self,
            _nodes: &[NodeImage<'_>],
            _steps: &[ResolvedStep],
        ) -> Result<(), QueueFailure> {
            let mut log = self.log.lock().unwrap();
            self.refuse(&log)?;
            log.nowait_submits += 1;
            log.in_flight = true;
            log.poisoned = false;
            Ok(())
        }

        fn wait(&mut self) -> Result<(), QueueFailure> {
            let mut log = self.log.lock().unwrap();
            log.waits += 1;
            log.in_flight = false;
            Ok(())
        }

        fn supports_persistent_cancel(&self) -> bool {
            true
        }

        fn wait_timeout(&mut self, timeout: Duration) -> TimedWait {
            let mut log = self.log.lock().unwrap();
            log.timeouts.push(timeout);
            if !log.in_flight {
                return TimedWait::Done(Ok(()));
            }
            if log.completes_on_own || (log.poisoned && log.exits_on_poison) {
                log.in_flight = false;
                return TimedWait::Done(Ok(()));
            }
            TimedWait::Pending
        }

        fn raise_poison(&mut self, address: u64, value: u32) -> Result<(), QueueFailure> {
            let mut log = self.log.lock().unwrap();
            if log.poison_fails {
                return Err(QueueFailure::new(
                    "poison write refused",
                    Quiescence::Proven,
                ));
            }
            log.poison_writes.push((address, value));
            log.poisoned = true;
            Ok(())
        }

        fn teardown(&mut self, observe: Duration) -> Teardown {
            let mut log = self.log.lock().unwrap();
            log.teardowns.push(observe);
            log.torn_down = true;
            log.in_flight = false;
            let quiescence = log.teardown_proves.unwrap_or(Quiescence::Unknown);
            if quiescence == Quiescence::Unknown {
                // The device-level reaction of a queue that could not prove
                // retirement (rdna-compute's registry, modelled).
                *self.device.lock().unwrap() =
                    Some("teardown could not prove every workgroup retired".into());
            }
            Teardown {
                quiescence,
                message: format!("fake teardown proves {quiescence:?}"),
            }
        }
    }

    const BUDGET: Duration = Duration::from_millis(50);
    const GRACE: Duration = Duration::from_millis(20);
    const ESCALATION: Duration = Duration::from_millis(30);

    fn watchdog() -> PersistentWatchdog {
        PersistentWatchdog {
            budget: BUDGET,
            grace: GRACE,
            escalation_grace: ESCALATION,
            poison: PoisonCell {
                resource: ResourceId(0),
                offset: 4,
                value: 1,
            },
        }
    }

    fn persistent_parts() -> WindowPlanParts {
        let mut parts = parts_with(patches());
        parts
            .program
            .kinds
            .insert(ResourceId(0), crate::ResourceKind::EventCells { cells: 2 });
        parts
    }

    fn persistent_plan(w: PersistentWatchdog) -> Result<Arc<WindowPlan>, Refusal> {
        let parts = persistent_parts();
        let table = CacheTable::for_arch("gfx1201").expect("gfx1201 table");
        let pm4 = plan(&parts.program, &table, RowPolicy::UnreceiptedSystem);
        WindowPlan::new(parts, &pm4)?
            .with_persistent_watchdog(w)
            .map(Arc::new)
    }

    fn cancel_window_on(device: &FakeDevice) -> (PreparedWindow, Arc<Mutex<CancelLog>>) {
        let log = Arc::new(Mutex::new(CancelLog::default()));
        let queue = CancelQueue {
            log: log.clone(),
            device: device.clone(),
        };
        let w = PreparedWindow::new(
            persistent_plan(watchdog()).expect("persistent plan"),
            key(),
            Box::new(queue),
        )
        .expect("prepared");
        (w, log)
    }

    fn cancel_window() -> (PreparedWindow, Arc<Mutex<CancelLog>>) {
        cancel_window_on(&FakeDevice::default())
    }

    fn submit_persistent(w: &mut PreparedWindow, generation: u64) -> Submission {
        let bound = w
            .bind(&shape(300, generation), &bindings(), &inputs())
            .unwrap();
        w.submit(bound).unwrap()
    }

    #[test]
    fn persistent_window_that_exits_on_poison_is_poisoned_and_proven() {
        let (mut w, log) = cancel_window();
        log.lock().unwrap().exits_on_poison = true;
        let submission = submit_persistent(&mut w, 1);
        let failure = submission.complete().unwrap_err();
        assert_eq!(failure.kind, FailureKind::Poisoned);
        assert_eq!(failure.quiescence, Quiescence::Proven);
        assert_eq!(failure.generation, 1);
        assert!(w.is_retired(), "a poisoned, retired window is reusable");
        {
            let l = log.lock().unwrap();
            // The poison cell is the declared cell of resource 0: base + 4.
            assert_eq!(l.poison_writes, vec![(BASE_Q + 4, 1)]);
            assert_eq!(l.timeouts, vec![BUDGET, GRACE]);
            assert!(
                l.teardowns.is_empty(),
                "no escalation when poison retired it"
            );
            // The queue's global wait (and a blocking submit) are never used.
            assert_eq!((l.waits, l.blocking_submits, l.nowait_submits), (0, 0, 1));
        }
        // Retired: the same window binds and submits again.
        log.lock().unwrap().exits_on_poison = false;
        log.lock().unwrap().completes_on_own = true;
        let submission = submit_persistent(&mut w, 2);
        assert_eq!(submission.complete().unwrap().generation, 2);
    }

    #[test]
    fn persistent_window_completing_within_budget_is_never_poisoned() {
        let (mut w, log) = cancel_window();
        log.lock().unwrap().completes_on_own = true;
        let done = submit_persistent(&mut w, 1).complete().unwrap();
        assert_eq!(done.generation, 1);
        assert!(w.is_retired());
        let l = log.lock().unwrap();
        assert_eq!(l.timeouts, vec![BUDGET]);
        assert!(l.poison_writes.is_empty() && l.teardowns.is_empty());
    }

    #[test]
    fn persistent_window_that_ignores_poison_escalates_to_teardown() {
        // The program never completes on its own and ignores the poison; the
        // teardown observes retirement (Proven).
        let (mut w, log) = cancel_window();
        log.lock().unwrap().teardown_proves = Some(Quiescence::Proven);
        let failure = submit_persistent(&mut w, 1).complete().unwrap_err();
        assert_eq!(failure.kind, FailureKind::TornDown);
        assert_eq!(failure.quiescence, Quiescence::Proven);
        {
            let l = log.lock().unwrap();
            assert_eq!(l.poison_writes, vec![(BASE_Q + 4, 1)]);
            assert_eq!(l.timeouts, vec![BUDGET, GRACE]);
            assert_eq!(l.teardowns, vec![ESCALATION]);
            assert_eq!(l.waits, 0);
        }
        // The queue is gone: the window is never reusable, even though the
        // teardown proved quiescence (its resources are released).
        assert!(!w.is_retired());
        assert_eq!(
            w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap_err(),
            Refusal::Quarantined
        );
    }

    #[test]
    fn failed_teardown_is_unknown_and_quarantines_the_device_for_every_submit() {
        let device = FakeDevice::default();
        let (mut w, log) = cancel_window_on(&device);
        // The default teardown proves nothing.
        let failure = submit_persistent(&mut w, 1).complete().unwrap_err();
        assert_eq!(failure.kind, FailureKind::TornDown);
        assert_eq!(failure.quiescence, Quiescence::Unknown);
        assert!(failure.message.contains("fake teardown proves Unknown"));
        assert!(!w.is_retired());
        let reason = device.lock().unwrap().clone().expect("device quarantined");

        // Another window on the same device: its submit refuses with the
        // recorded reason, before the queue launches anything.
        let (mut other, other_log) = cancel_window_on(&device);
        let bound = other.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        match other.submit(bound).err() {
            Some(Refusal::SubmitFailed {
                message,
                quiescence: Quiescence::Proven,
                kind: FailureKind::DeviceQuarantined,
            }) => assert_eq!(message, reason),
            other => panic!("expected a quarantine refusal, got {other:?}"),
        }
        assert_eq!(other_log.lock().unwrap().nowait_submits, 0);

        // The raw-queue watched path refuses the same way.
        let mut raw = CancelQueue {
            log: Arc::new(Mutex::new(CancelLog::default())),
            device: device.clone(),
        };
        let wd = watchdog().resolve(&key_bindings()).unwrap();
        let refused = submit_queue_nowait_watched(&mut raw, &[], &[], &wd)
            .err()
            .expect("refused");
        assert_eq!(refused.kind, FailureKind::DeviceQuarantined);
        assert_eq!(refused.message, reason);
        // So does the torn-down window's own queue.
        assert_eq!(log.lock().unwrap().nowait_submits, 1);
    }

    #[test]
    fn a_failed_poison_write_escalates_without_waiting_the_grace() {
        let (mut w, log) = cancel_window();
        log.lock().unwrap().poison_fails = true;
        let failure = submit_persistent(&mut w, 1).complete().unwrap_err();
        assert_eq!(failure.kind, FailureKind::TornDown);
        assert_eq!(failure.quiescence, Quiescence::Unknown);
        assert!(failure.message.contains("could not be raised"));
        let l = log.lock().unwrap();
        assert_eq!(
            l.timeouts,
            vec![BUDGET],
            "no grace wait after a failed write"
        );
        assert!(l.poison_writes.is_empty());
        assert_eq!(l.teardowns, vec![ESCALATION]);
    }

    #[test]
    fn a_blocking_submit_of_a_persistent_window_is_routed_through_nowait() {
        let (mut w, log) = cancel_window();
        log.lock().unwrap().completes_on_own = true;
        submit_persistent(&mut w, 1).complete().unwrap();
        let l = log.lock().unwrap();
        assert_eq!((l.blocking_submits, l.nowait_submits), (0, 1));
    }

    #[test]
    fn raw_watched_completion_poisons_and_escalates() {
        let log = Arc::new(Mutex::new(CancelLog::default()));
        let mut queue = CancelQueue {
            log: log.clone(),
            device: FakeDevice::default(),
        };
        let wd = watchdog().resolve(&key_bindings()).unwrap();
        assert_eq!(wd.poison_address, BASE_Q + 4);

        log.lock().unwrap().exits_on_poison = true;
        let failure = submit_queue_nowait_watched(&mut queue, &[], &[], &wd)
            .unwrap()
            .wait()
            .unwrap_err();
        assert_eq!(failure.kind, FailureKind::Poisoned);
        assert_eq!(failure.quiescence, Quiescence::Proven);
        assert_eq!(log.lock().unwrap().waits, 0);

        // A dropped handle runs the watchdog too (and reports nothing).
        {
            let mut l = log.lock().unwrap();
            l.exits_on_poison = false;
            l.teardown_proves = Some(Quiescence::Proven);
        }
        let completion = submit_queue_nowait_watched(&mut queue, &[], &[], &wd).unwrap();
        drop(completion);
        let l = log.lock().unwrap();
        assert!(
            !l.in_flight,
            "a dropped watched handle never leaves it in flight"
        );
        assert_eq!(l.teardowns, vec![ESCALATION]);
    }

    #[test]
    fn watched_submit_is_refused_on_a_queue_without_persistent_cancel() {
        let (_, log) = prepared();
        let mut queue = MockQueue(log.clone());
        let wd = watchdog().resolve(&key_bindings()).unwrap();
        let refused = submit_queue_nowait_watched(&mut queue, &[], &[], &wd)
            .err()
            .expect("refused");
        assert_eq!(refused.quiescence, Quiescence::Proven);
        assert_eq!(refused.kind, FailureKind::Queue);
        assert_eq!(log.lock().unwrap().submits, 0, "nothing was launched");

        // A persistent plan on such a queue is refused at prepare time.
        let err = PreparedWindow::new(
            persistent_plan(watchdog()).unwrap(),
            key(),
            Box::new(MockQueue(log)),
        )
        .err()
        .expect("refused");
        assert!(matches!(err, Refusal::Malformed(m) if m.contains("persistent cancel")));
    }

    #[test]
    fn non_persistent_window_keeps_todays_completion_semantics() {
        // No watchdog: the lease is the queue's own `wait`, never a bounded
        // wait, and its failures stay ordinary queue failures.
        let log = Arc::new(Mutex::new(CancelLog::default()));
        let plan = make_plan(parts_with(patches())).expect("plan");
        assert!(plan.persistent_watchdog().is_none());
        let queue = CancelQueue {
            log: log.clone(),
            device: FakeDevice::default(),
        };
        let mut w = PreparedWindow::new(plan, key(), Box::new(queue)).unwrap();
        let bound = w.bind(&shape(300, 1), &bindings(), &inputs()).unwrap();
        let submission = w.submit_nowait(bound).unwrap();
        assert_eq!(submission.complete().unwrap().generation, 1);
        {
            let l = log.lock().unwrap();
            assert_eq!(
                (l.waits, l.timeouts.len(), l.poison_writes.len()),
                (1, 0, 0)
            );
            assert_eq!(l.blocking_submits + l.nowait_submits, 1);
        }
        // A blocking submit stays a blocking submit.
        let bound = w.bind(&shape(300, 2), &bindings(), &inputs()).unwrap();
        w.submit(bound).unwrap().complete().unwrap();
        let l = log.lock().unwrap();
        assert_eq!((l.blocking_submits, l.nowait_submits), (1, 1));
        assert!(l.teardowns.is_empty());
    }

    fn key_bindings() -> Vec<Binding> {
        key()
            .allocations
            .iter()
            .map(|a| Binding {
                base: a.base,
                bytes: a.bytes,
            })
            .collect()
    }

    #[test]
    fn persistent_declarations_are_validated() {
        let with = |f: &dyn Fn(&mut PersistentWatchdog)| {
            let mut w = watchdog();
            f(&mut w);
            persistent_plan(w).err().map(|e| e.to_string())
        };
        assert_eq!(with(&|_| {}), None);
        // Not an EventCells resource.
        let e = with(&|w| w.poison.resource = ResourceId(1)).unwrap();
        assert!(e.contains("not declared EventCells"), "{e}");
        let e = with(&|w| w.poison.offset = 2).unwrap();
        assert!(e.contains("not 4-byte aligned"), "{e}");
        // Two cells hold 32 bytes: the word at 32 is outside them.
        let e = with(&|w| w.poison.offset = 32).unwrap();
        assert!(e.contains("outside the 2 declared event cells"), "{e}");
        let e = with(&|w| w.poison.value = 0).unwrap();
        assert!(e.contains("poison value is zero"), "{e}");
        for field in ["budget", "grace", "escalation_grace"] {
            let e = with(&|w| match field {
                "budget" => w.budget = Duration::ZERO,
                "grace" => w.grace = Duration::ZERO,
                _ => w.escalation_grace = Duration::ZERO,
            })
            .unwrap();
            assert!(e.contains(&format!("{field} is zero")), "{e}");
        }
        // No EventCells declaration at all.
        let parts = parts_with(patches());
        let table = CacheTable::for_arch("gfx1201").unwrap();
        let pm4 = plan(&parts.program, &table, RowPolicy::UnreceiptedSystem);
        let e = WindowPlan::new(parts, &pm4)
            .unwrap()
            .with_persistent_watchdog(watchdog())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("not declared EventCells"), "{e}");

        // Resolution is checked against the live allocation.
        let mut small = key_bindings();
        small[0].bytes = 6;
        assert!(watchdog()
            .resolve(&small)
            .unwrap_err()
            .contains("exceeds the 6-byte allocation"));
        assert!(ResolvedWatchdog::new(BUDGET, GRACE, ESCALATION, 0x7000_0002, 1).is_err());
        assert!(ResolvedWatchdog::new(BUDGET, GRACE, ESCALATION, 0, 1).is_err());
    }
}
