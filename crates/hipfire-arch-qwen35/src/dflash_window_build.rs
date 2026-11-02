// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun design A slice A8: the pure (GPU-free) half of the window
//! builder. [`build_window`] turns the A4 draft emission and the A5 verify
//! plan of one window into [`WindowPlanParts`] — one railgun
//! [`KernelProgram`](railgun::kernel::KernelProgram) with a declared patch
//! table — and refuses the whole window when any node lacks exact-artifact
//! facts (design §5).
//!
//! * **Authoring.** Every planned kernel is authored with the certified facts
//!   of its exact code object (`WindowFactsSource::facts`, keyed by the
//!   planned `code_object_digest`) through [`railgun::kernel::author`]; every
//!   pointer argument is bound to the live allocation behind it
//!   ([`build_window`]'s `probe`). `Scalar32` patch sites become declared
//!   [`WordRole::Window`] words, the verify plan's declared `GdnFrame` words
//!   stay `GdnFrame`.
//! * **Copy / fill.** `lower_owned` dispatches program nodes only, so every
//!   typed `PlannedOp::Copy` becomes a `railgun_copy` node and every `Fill` a
//!   `railgun_fill_u32` node, in plan order (hazard edges are derived by
//!   `railgun::plan` like for any node). Their source/destination pointers are
//!   `Interior64` patch sites over the program resource that holds them, with
//!   `required_extent` = the byte count, so [`PreparedWindow::bind`]
//!   (`railgun::window`) proves every copy/fill extent inside its allocation
//!   before work is queued; a shape-dependent byte count is a `Scalar32` plus
//!   a `GridAxis` patch of the copy grid. A zero-byte copy, a fill that is not
//!   word-aligned or a copy whose two ranges can overlap inside one allocation
//!   is refused at build.
//! * **Rebasing.** Emitter patches name the *kernel ordinal* of their own
//!   emission and the emission's own resource ids. In the combined program
//!   every planned op (prefix copies, then draft ops, then verify ops) is one
//!   node, so a kernel's node index is the index of its op in the combined
//!   list; draft resource ids are re-mapped onto the program's resources by
//!   allocation containment, and interior offsets/extents are re-based so the
//!   bound proof is exactly the emitter's buffer, not the (possibly larger)
//!   backing allocation.
//!
//! [`PreparedWindow::bind`]: railgun::window::PreparedWindow::bind

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use railgun::copy as rcopy;
use railgun::kernel::{
    author, Arg, Axis, DeclaredWord, Expr, FactStatus, KernelProgram, LaunchRecord, NodeFacts,
    PatchKind, ScalarField, WordRole,
};
use railgun::window::{
    CopyStep, HostDeltaSpec, TransitionSpec, WindowExecution, WindowPlanParts, WindowStep,
};
use railgun::{Binding, ResourceId, WordId};
use rdna_compute::code_object::CodeObjectId;
use rdna_compute::plan_sink::{PlannedKernel, PlannedOp};

use crate::dflash_draft_emit::DraftEmission;
use crate::dflash_verify_emit::DflashVerifyPlan;

/// Window scalar the owner supplies: the hidden ring's `written` counter.
pub const SCALAR_RING_WRITTEN: &str = "ring_written";

/// Rows of the fixed chain window.
const WINDOW_ROWS: usize = 16;

/// Kernarg offsets of the transport kernels (`railgun::copy` launch builders).
const COPY_OFF_BYTES: u32 = 16;

/// Object digests of the loaded railgun transport kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportDigests {
    pub copy: [u8; 32],
    pub fill: [u8; 32],
}

/// Certified-facts source: the receipt corpus (GPU side) or a test double.
pub trait WindowFactsSource {
    /// Facts of `symbol` in the exact code object `artifact_sha256_hex`.
    fn facts(&mut self, artifact_sha256_hex: &str, symbol: &str) -> Result<Arc<NodeFacts>, String>;
    /// Digests of the loaded `railgun_copy` / `railgun_fill_u32` objects.
    fn transport(&mut self) -> Result<TransportDigests, String>;
}

/// A window the builder refused, with the node it stopped at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildRefusal {
    /// Combined node/op index, when the refusal is about one node.
    pub node: Option<usize>,
    pub symbol: String,
    pub reason: String,
}

impl BuildRefusal {
    fn whole(reason: impl Into<String>) -> Self {
        Self {
            node: None,
            symbol: String::new(),
            reason: reason.into(),
        }
    }

    fn at(node: usize, symbol: &str, reason: impl Into<String>) -> Self {
        Self {
            node: Some(node),
            symbol: symbol.to_owned(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for BuildRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node {
            Some(n) => write!(f, "node {n} {}: {}", self.symbol, self.reason),
            None => write!(f, "{}", self.reason),
        }
    }
}

impl std::error::Error for BuildRefusal {}

/// A transport copy the builder prepends (absolute addresses, constant size):
/// staging → verify tokens / positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransportCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
}

/// What produced a program node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeOrigin {
    /// `prefix[i]`.
    Prefix(usize),
    /// Op `i` of the draft emission.
    Draft(usize),
    /// Op `i` of the verify plan.
    Verify(usize),
}

/// §A9.4: the dedicated device-only sealed-positions allocation the
/// conditional (loaded-index) KV writers read instead of `pbs.positions`.
/// The 16 bytes are the same `[i32; 16]` the window's prefix copy writes to
/// `pbs.positions`, so a writer is retargeted only if it already read
/// `pbs.positions` at offset 0 (`from`); any other conditional writer is a
/// build refusal, never silently rewritten.
#[derive(Clone, Copy, Debug)]
pub struct SealedRetarget {
    /// Base address of `pbs.positions`.
    pub from: u64,
    /// Base address of the dedicated positions allocation.
    pub to: u64,
    /// That allocation as the program binds it.
    pub binding: Binding,
}

pub struct BuildInput<'a> {
    pub arch: &'a str,
    pub execution: WindowExecution,
    /// Typed copies that run first (`Whole16` only: the verify inputs).
    pub prefix: &'a [TransportCopy],
    pub draft: &'a DraftEmission,
    /// Required for `Whole16`, must be `None` for `HostSplice` (the verify
    /// segment is incumbent until a verify segment is certified at its B).
    pub verify: Option<&'a DflashVerifyPlan>,
    /// `HiddenStateRingBuffer::max_positions` (modulus of the ring head).
    pub ring_capacity: u64,
    /// Device addresses the owner reads after completion (selector block and
    /// status, verify argmax): they must lie in a window-written allocation.
    pub completion_addrs: &'a [u64],
    /// True only for a manifest-admitted exact route.
    pub certified: bool,
    pub tape_populated: bool,
    /// Present when the owner holds a sealed-positions allocation: every
    /// `Conditional` loaded-index writer is pointed at it (see
    /// [`SealedRetarget`]). Without it a conditional node refuses the build.
    pub sealed_positions: Option<SealedRetarget>,
}

/// The authored window.
#[derive(Clone, Debug)]
pub struct BuiltWindow {
    pub parts: WindowPlanParts,
    /// Per program node: the op it came from.
    pub origins: Vec<NodeOrigin>,
    /// `GdnFrame` words in node order: the owner supplies each word's value at
    /// bind (the builder uses the plan's authored frame, as the HIP-graph
    /// replay bakes it).
    pub gdn_frames: Vec<(WordId, u32)>,
    /// Window scalars the owner must supply in `WindowInputs::scalars`.
    pub scalars_required: Vec<String>,
    /// Per node: phase, symbol and loaded module — the JSONL
    /// `railgun-cert whole16-manifest --sequence` consumes.
    pub sequence: Vec<SequenceNode>,
}

/// One node of the window sequence (`railgun-cert whole16-manifest` input).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceNode {
    pub phase: String,
    pub symbol: String,
    pub module: String,
}

// ----------------------------------------------------------------- slots

#[derive(Clone, Copy, PartialEq, Eq)]
enum Emission {
    Draft,
    Verify,
}

/// Where a copy/fill endpoint lives.
#[derive(Clone)]
enum Addr {
    /// Absolute device address at plan time (no shape dependence): the
    /// authored pointer argument is the whole truth.
    Abs,
    /// `draft resource + byte offset expression`.
    Draft { resource: ResourceId, offset: Expr },
}

enum Slot<'a> {
    Kernel {
        kernel: &'a PlannedKernel,
        emission: Emission,
        /// Ordinal among the `Kernel` ops of its emission.
        ordinal: usize,
        gdn: Vec<DeclaredWord>,
    },
    Copy {
        src: Addr,
        dst: Addr,
        bytes: Expr,
        planned: (u64, u64, u64),
    },
    Fill {
        dst: Addr,
        value: u32,
        words: Expr,
        planned: (u64, u32),
    },
}

fn hex(digest: &[u8; 32]) -> String {
    CodeObjectId(*digest).to_hex()
}

fn fold_add(e: &Expr, c: u64) -> Expr {
    match e {
        Expr::Const(v) => Expr::Const(v + c),
        _ if c == 0 => e.clone(),
        _ => Expr::add(e.clone(), Expr::Const(c)),
    }
}

fn expr_const(e: &Expr) -> Option<u64> {
    match e {
        Expr::Const(c) => Some(*c),
        _ => None,
    }
}

/// Allocation lookup with a containment cache over the caller's probe.
struct Probe<'a> {
    probe: &'a mut dyn FnMut(u64) -> Result<Binding, String>,
    known: BTreeMap<u64, Binding>,
}

impl Probe<'_> {
    fn get(&mut self, addr: u64) -> Result<Binding, String> {
        if let Some((_, b)) = self.known.range(..=addr).next_back() {
            if addr - b.base < b.bytes.max(1) {
                return Ok(*b);
            }
        }
        let b = (self.probe)(addr)?;
        if addr < b.base || addr - b.base >= b.bytes.max(1) {
            return Err(format!(
                "address {addr:#x} lies outside its reported allocation {:#x}+{}",
                b.base, b.bytes
            ));
        }
        self.known.insert(b.base, b);
        Ok(b)
    }
}

fn le_u64(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
}

/// Plan-time value of a draft expression.
fn eval_at_plan(e: &Expr, draft: &DraftEmission) -> Result<u64, String> {
    let scalars = draft.scalars(0);
    e.eval(&|name| draft.shape.get(name).or_else(|| scalars.get(name).copied()))
        .map_err(|x| x.to_string())
}

// ----------------------------------------------------------------- build

/// Author `input`'s window. See the module docs.
pub fn build_window(
    input: &BuildInput<'_>,
    facts: &mut dyn WindowFactsSource,
    probe: &mut dyn FnMut(u64) -> Result<Binding, String>,
) -> Result<BuiltWindow, BuildRefusal> {
    let (slots, origins, whole16) = collect_slots(input)?;
    let draft = input.draft;
    let mut probe = Probe {
        probe,
        known: BTreeMap::new(),
    };

    // ---- launch records ---------------------------------------------------
    let transport = facts
        .transport()
        .map_err(|e| BuildRefusal::whole(format!("railgun transport kernels unavailable: {e}")));
    let has_transport = slots
        .iter()
        .any(|s| matches!(s, Slot::Copy { .. } | Slot::Fill { .. }));
    let transport = if has_transport {
        Some(transport?)
    } else {
        None
    };

    let mut launches: Vec<LaunchRecord> = Vec::with_capacity(slots.len());
    // Per kernel slot: the Scalar32 field offsets its patches declare.
    for (node, slot) in slots.iter().enumerate() {
        let launch = match slot {
            Slot::Kernel {
                kernel,
                ordinal,
                gdn,
                ..
            } => {
                let mut declared: Vec<DeclaredWord> = gdn.clone();
                for p in &kernel.patches {
                    if let PatchKind::Scalar32 { field, .. } = p {
                        if field.node != *ordinal {
                            return Err(BuildRefusal::at(
                                node,
                                &kernel.symbol,
                                format!(
                                    "Scalar32 patch names kernel ordinal {} but sits on ordinal {ordinal}",
                                    field.node
                                ),
                            ));
                        }
                        if declared.iter().any(|d| d.offset == field.offset) {
                            return Err(BuildRefusal::at(
                                node,
                                &kernel.symbol,
                                format!("word +{} is declared twice", field.offset),
                            ));
                        }
                        declared.push(DeclaredWord {
                            offset: field.offset,
                            role: WordRole::Window,
                        });
                    }
                }
                let digest = hex(&kernel.code_object_digest);
                let node_facts = facts.facts(&digest, &kernel.symbol).map_err(|e| {
                    BuildRefusal::at(
                        node,
                        &kernel.symbol,
                        format!("no exact-artifact facts ({}): {e}", &digest[..16]),
                    )
                })?;
                let mut kernarg = kernel.kernarg.clone();
                let mut retargeted = None;
                if matches!(node_facts.facts.status, FactStatus::Conditional(_)) {
                    let at = railgun::index_bind::abi::POSITIONS;
                    let Some(rt) = input.sealed_positions else {
                        return Err(BuildRefusal::at(
                            node,
                            &kernel.symbol,
                            "conditional loaded-index writer but no sealed positions allocation",
                        ));
                    };
                    let value = le_u64(&kernel.kernarg, at as usize);
                    if value != Some(rt.from) {
                        return Err(BuildRefusal::at(
                            node,
                            &kernel.symbol,
                            format!(
                                "conditional writer reads positions {value:#x?}, not pbs.positions {:#x}: not retargeted",
                                rt.from
                            ),
                        ));
                    }
                    kernarg[at as usize..at as usize + 8].copy_from_slice(&rt.to.to_le_bytes());
                    retargeted = Some((at, rt.binding));
                }
                let mut allocations = Vec::new();
                for offset in node_facts.facts.pointer_offsets() {
                    let Some(value) = le_u64(&kernarg, offset as usize) else {
                        continue;
                    };
                    if let Some((at, binding)) = retargeted {
                        if offset == at {
                            allocations.push((offset, Ok(binding)));
                            continue;
                        }
                    }
                    if value != 0 {
                        allocations.push((offset, probe.get(value)));
                    }
                }
                LaunchRecord {
                    symbol: kernel.symbol.clone(),
                    grid: kernel.grid,
                    block: kernel.block,
                    dynamic_lds: kernel.dynamic_lds,
                    kernarg,
                    declared,
                    facts: Ok(node_facts),
                    allocations,
                }
            }
            Slot::Copy { bytes, planned, .. } => {
                let (src, dst, n) = *planned;
                if n == 0 || n > rcopy::MAX_COPY_BYTES {
                    return Err(BuildRefusal::at(
                        node,
                        rcopy::COPY_SYMBOL,
                        format!("copy of {n} bytes outside 1..={}", rcopy::MAX_COPY_BYTES),
                    ));
                }
                let launch = rcopy::copy_launch(src, dst, n as u32);
                let digest = hex(&transport.expect("transport resolved").copy);
                let node_facts = facts.facts(&digest, rcopy::COPY_SYMBOL).map_err(|e| {
                    BuildRefusal::at(
                        node,
                        rcopy::COPY_SYMBOL,
                        format!("no exact-artifact facts ({}): {e}", &digest[..16]),
                    )
                })?;
                let mut declared = Vec::new();
                if expr_const(bytes).is_none() {
                    declared.push(DeclaredWord {
                        offset: COPY_OFF_BYTES,
                        role: WordRole::Window,
                    });
                }
                LaunchRecord {
                    symbol: launch.symbol.to_owned(),
                    grid: launch.grid,
                    block: launch.block,
                    dynamic_lds: 0,
                    kernarg: launch.kernarg,
                    declared,
                    facts: Ok(node_facts),
                    allocations: vec![(0, probe.get(src)), (8, probe.get(dst))],
                }
            }
            Slot::Fill {
                words,
                planned,
                value,
                ..
            } => {
                let (dst, n) = *planned;
                if expr_const(words).is_none() {
                    return Err(BuildRefusal::at(
                        node,
                        rcopy::FILL_SYMBOL,
                        "a shape-dependent fill size is not supported",
                    ));
                }
                if let Err(e) = rcopy::check_fill(0, u64::from(n), u64::MAX) {
                    return Err(BuildRefusal::at(node, rcopy::FILL_SYMBOL, e));
                }
                if dst % 4 != 0 {
                    return Err(BuildRefusal::at(
                        node,
                        rcopy::FILL_SYMBOL,
                        format!("fill destination {dst:#x} is not 4-byte aligned"),
                    ));
                }
                let launch = rcopy::fill_launch(dst, *value, n);
                let digest = hex(&transport.expect("transport resolved").fill);
                let node_facts = facts.facts(&digest, rcopy::FILL_SYMBOL).map_err(|e| {
                    BuildRefusal::at(
                        node,
                        rcopy::FILL_SYMBOL,
                        format!("no exact-artifact facts ({}): {e}", &digest[..16]),
                    )
                })?;
                LaunchRecord {
                    symbol: launch.symbol.to_owned(),
                    grid: launch.grid,
                    block: launch.block,
                    dynamic_lds: 0,
                    kernarg: launch.kernarg,
                    declared: Vec::new(),
                    facts: Ok(node_facts),
                    allocations: vec![(0, probe.get(dst))],
                }
            }
        };
        launches.push(launch);
    }

    // ---- author -----------------------------------------------------------
    // Conditional (indexed) nodes are laid out as PM4 against their bind
    // witness (amendment §A9.4); the plan keeps each node's `indexed` case and
    // `PreparedWindow` refuses to submit until `seal_positions` attaches a
    // witness covering every one of them.
    let mut program = author(input.arch, &launches).assume_indexed_witness();
    for (node, n) in program.nodes.iter().enumerate() {
        if !n.lowering.is_pm4() {
            return Err(BuildRefusal::at(
                node,
                &n.symbol,
                format!("not a railgun PM4 node: {:?}", n.lowering),
            ));
        }
    }

    // ---- resources and rebasing ------------------------------------------
    let mut draft_map: BTreeMap<ResourceId, Mapped> = BTreeMap::new();
    for r in &draft.resources {
        let id = resource_of_addr(&mut program, &mut probe, r.base)
            .map_err(|e| BuildRefusal::whole(format!("draft resource {}: {e}", r.name)))?;
        let p = program.resources[id.0 as usize];
        let (d_end, p_end) = (r.base + r.bytes, p.base + p.bytes);
        if d_end > p_end {
            return Err(BuildRefusal::whole(format!(
                "draft resource {} ({:#x}+{}) straddles its allocation {:#x}+{}",
                r.name, r.base, r.bytes, p.base, p.bytes
            )));
        }
        draft_map.insert(
            r.id,
            Mapped {
                program: id,
                delta: r.base - p.base,
                slack: p_end - d_end,
            },
        );
    }

    let mut patches: Vec<PatchKind> = Vec::new();
    for (node, slot) in slots.iter().enumerate() {
        match slot {
            Slot::Kernel {
                kernel,
                emission,
                ordinal,
                ..
            } => {
                for p in &kernel.patches {
                    match p {
                        PatchKind::Scalar32 {
                            field,
                            expr,
                            domain,
                        } => {
                            patches.push(PatchKind::Scalar32 {
                                field: ScalarField {
                                    node,
                                    offset: field.offset,
                                },
                                expr: expr.clone(),
                                domain: domain.clone(),
                            });
                        }
                        PatchKind::GridAxis {
                            node: pn,
                            axis,
                            expr,
                            domain,
                        } => {
                            if *pn != *ordinal {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!("GridAxis patch names kernel ordinal {pn}, sits on {ordinal}"),
                                ));
                            }
                            patches.push(PatchKind::GridAxis {
                                node,
                                axis: *axis,
                                expr: expr.clone(),
                                domain: domain.clone(),
                            });
                        }
                        PatchKind::Interior64 {
                            node: pn,
                            arg_offset,
                            resource,
                            byte_offset,
                            required_extent,
                        } => {
                            if *pn != *ordinal {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!("Interior64 patch names kernel ordinal {pn}, sits on {ordinal}"),
                                ));
                            }
                            if *emission != Emission::Draft {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    "an Interior64 patch of the verify plan has no resource table to rebase against",
                                ));
                            }
                            let Some(m) = draft_map.get(resource) else {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!(
                                        "Interior64 names unknown draft resource {}",
                                        resource.0
                                    ),
                                ));
                            };
                            let Some((arg_res, _)) = arg_resource(&program, node, *arg_offset)
                            else {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!("+{arg_offset} is not a pointer argument of the authored node"),
                                ));
                            };
                            if arg_res != m.program {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!(
                                        "+{arg_offset} points into program resource {} but the patch names {}",
                                        arg_res.0, m.program.0
                                    ),
                                ));
                            }
                            // The patch must reproduce the planned pointer.
                            let planned = le_u64(&kernel.kernarg, *arg_offset as usize);
                            let at = eval_at_plan(byte_offset, draft).map_err(|e| {
                                BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!("cannot evaluate Interior64 at the planned shape: {e}"),
                                )
                            })?;
                            let b = program.resources[m.program.0 as usize];
                            if planned != Some(b.base + m.delta + at) {
                                return Err(BuildRefusal::at(
                                    node,
                                    &kernel.symbol,
                                    format!(
                                        "Interior64 at +{arg_offset} evaluates to {:#x}, the planned pointer is {planned:x?}",
                                        b.base + m.delta + at
                                    ),
                                ));
                            }
                            patches.push(PatchKind::Interior64 {
                                node,
                                arg_offset: *arg_offset,
                                resource: m.program,
                                byte_offset: fold_add(byte_offset, m.delta),
                                required_extent: fold_add(required_extent, m.slack),
                            });
                        }
                    }
                }
            }
            Slot::Copy {
                src,
                dst,
                bytes,
                planned,
            } => {
                let (ps, pd, pn) = *planned;
                let sym = rcopy::COPY_SYMBOL;
                if let Some(c) = expr_const(bytes) {
                    if c != pn {
                        return Err(BuildRefusal::at(
                            node,
                            sym,
                            format!("typed copy size {c} != recorded {pn}"),
                        ));
                    }
                } else {
                    let at = eval_at_plan(bytes, draft).map_err(|e| {
                        BuildRefusal::at(node, sym, format!("cannot evaluate the copy size: {e}"))
                    })?;
                    if at != pn {
                        return Err(BuildRefusal::at(
                            node,
                            sym,
                            format!("typed copy size evaluates to {at}, recorded {pn}"),
                        ));
                    }
                }
                let s = endpoint(&program, node, 0, src, ps, draft, &draft_map)
                    .map_err(|e| BuildRefusal::at(node, sym, format!("source: {e}")))?;
                let d = endpoint(&program, node, 8, dst, pd, draft, &draft_map)
                    .map_err(|e| BuildRefusal::at(node, sym, format!("destination: {e}")))?;
                if s.resource == d.resource {
                    let (Some(so), Some(dof), Some(len)) = (
                        expr_const(&s.offset),
                        expr_const(&d.offset),
                        expr_const(bytes),
                    ) else {
                        return Err(BuildRefusal::at(
                            node,
                            sym,
                            "a copy inside one allocation with shape-dependent offsets may overlap",
                        ));
                    };
                    if so < dof + len && dof < so + len {
                        return Err(BuildRefusal::at(
                            node,
                            sym,
                            "source and destination ranges overlap",
                        ));
                    }
                }
                patches.push(PatchKind::Interior64 {
                    node,
                    arg_offset: 0,
                    resource: s.resource,
                    byte_offset: s.offset,
                    required_extent: fold_add(bytes, s.slack),
                });
                patches.push(PatchKind::Interior64 {
                    node,
                    arg_offset: 8,
                    resource: d.resource,
                    byte_offset: d.offset,
                    required_extent: fold_add(bytes, d.slack),
                });
                if expr_const(bytes).is_none() {
                    patches.push(PatchKind::Scalar32 {
                        field: ScalarField {
                            node,
                            offset: COPY_OFF_BYTES,
                        },
                        expr: bytes.clone(),
                        domain: 1..=rcopy::MAX_COPY_BYTES,
                    });
                    let chunks = Expr::ceil_div(bytes.clone(), rcopy::CHUNK_BYTES)
                        .expect("chunk size is nonzero");
                    patches.push(PatchKind::GridAxis {
                        node,
                        axis: Axis::X,
                        expr: Expr::min(
                            Expr::max(chunks, Expr::Const(1)),
                            Expr::Const(u64::from(rcopy::MAX_GRID_X)),
                        ),
                        domain: 1..=u64::from(rcopy::MAX_GRID_X),
                    });
                }
            }
            Slot::Fill {
                dst,
                words,
                planned,
                ..
            } => {
                let sym = rcopy::FILL_SYMBOL;
                let (pd, pn) = *planned;
                let n = expr_const(words).expect("checked while authoring");
                if n != u64::from(pn) {
                    return Err(BuildRefusal::at(
                        node,
                        sym,
                        format!("typed fill size {n} != recorded {pn}"),
                    ));
                }
                let bytes = Expr::Const(n * 4);
                let d = endpoint(&program, node, 0, dst, pd, draft, &draft_map)
                    .map_err(|e| BuildRefusal::at(node, sym, format!("destination: {e}")))?;
                let Some(offset) = expr_const(&d.offset) else {
                    return Err(BuildRefusal::at(
                        node,
                        sym,
                        "a fill needs a constant destination offset",
                    ));
                };
                if let Err(e) = rcopy::check_fill(offset, n, u64::MAX) {
                    return Err(BuildRefusal::at(node, sym, e));
                }
                patches.push(PatchKind::Interior64 {
                    node,
                    arg_offset: 0,
                    resource: d.resource,
                    byte_offset: d.offset,
                    required_extent: fold_add(&bytes, d.slack),
                });
            }
        }
    }

    // ---- completion surfaces ---------------------------------------------
    let mut completion_surfaces: Vec<ResourceId> = Vec::new();
    for id in &draft.completion_surfaces {
        let m = draft_map.get(id).ok_or_else(|| {
            BuildRefusal::whole(format!(
                "completion surface names unknown draft resource {}",
                id.0
            ))
        })?;
        if !completion_surfaces.contains(&m.program) {
            completion_surfaces.push(m.program);
        }
    }
    for addr in input.completion_addrs {
        let id = resource_of_addr(&mut program, &mut probe, *addr)
            .map_err(|e| BuildRefusal::whole(format!("completion address {addr:#x}: {e}")))?;
        if !completion_surfaces.contains(&id) {
            completion_surfaces.push(id);
        }
    }

    // ---- host delta --------------------------------------------------------
    let mut host_delta = HostDeltaSpec {
        projection_watermark: draft.host_delta.projection_watermark.clone(),
        upload_watermark: draft.host_delta.upload_watermark.clone(),
        ring_head: draft.host_delta.ring_head.clone(),
        ring_written: draft.host_delta.ring_written.clone(),
    };
    let mut scalars_required: Vec<String> = draft
        .scalars_required
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    if let Some(v) = input.verify {
        let (Some(head), Some(written)) = (v.host_delta.ring_head, v.host_delta.ring_written)
        else {
            return Err(BuildRefusal::whole(
                "the verify plan carries no hidden-ring head/written transition",
            ));
        };
        let rows = v.rows as u64;
        let cap = input.ring_capacity;
        if cap == 0 || head.to != (head.from + rows) % cap || written.to != written.from + rows {
            return Err(BuildRefusal::whole(format!(
                "ring transition head {}->{} written {}->{} is not +{rows} modulo {cap}",
                head.from, head.to, written.from, written.to
            )));
        }
        if host_delta.ring_head.is_some() || host_delta.ring_written.is_some() {
            return Err(BuildRefusal::whole(
                "draft and verify both propose a hidden-ring transition",
            ));
        }
        host_delta.ring_head = Some(TransitionSpec {
            from: Expr::input("ring_head"),
            to: Expr::rem(Expr::add(Expr::input("ring_head"), Expr::Const(rows)), cap)
                .expect("ring capacity is nonzero"),
        });
        host_delta.ring_written = Some(TransitionSpec {
            from: Expr::input(SCALAR_RING_WRITTEN),
            to: Expr::add(Expr::input(SCALAR_RING_WRITTEN), Expr::Const(rows)),
        });
        scalars_required.push(SCALAR_RING_WRITTEN.to_owned());
    }

    // ---- GdnFrame words -----------------------------------------------------
    let gdn_frames: Vec<(WordId, u32)> = program
        .words
        .iter()
        .enumerate()
        .filter_map(|(i, w)| match w.role {
            WordRole::GdnFrame { frames } => Some((WordId(i as u32), frames)),
            WordRole::Window => None,
        })
        .collect();
    let sequence = sequence_of(&slots);
    let steps = (0..program.nodes.len())
        .map(|node| WindowStep::Kernel { node })
        .collect();
    Ok(BuiltWindow {
        parts: WindowPlanParts {
            program,
            steps,
            patches,
            host_delta,
            completion_surfaces,
            projection_rows: draft.family.projection_rows.clone(),
            execution: input.execution,
            hip_graph_certified: input.certified && whole16,
            tape_populated: input.tape_populated,
        },
        origins,
        gdn_frames,
        scalars_required,
        sequence,
    })
}

/// The node sequence of a window plan, in program order: what
/// `railgun-cert whole16-manifest --sequence` and `railgun-jit-corpus
/// --window-sequence` read. It needs no facts, so a window whose objects have
/// no corpus record yet can still dump it.
pub fn window_sequence(input: &BuildInput<'_>) -> Result<Vec<SequenceNode>, BuildRefusal> {
    let (slots, _, _) = collect_slots(input)?;
    Ok(sequence_of(&slots))
}

fn sequence_of(slots: &[Slot<'_>]) -> Vec<SequenceNode> {
    slots
        .iter()
        .map(|slot| match slot {
            Slot::Kernel {
                kernel, emission, ..
            } => SequenceNode {
                phase: if kernel.label.is_empty() {
                    match emission {
                        Emission::Draft => "draft".to_owned(),
                        Emission::Verify => "verify".to_owned(),
                    }
                } else {
                    kernel.label.to_owned()
                },
                symbol: kernel.symbol.clone(),
                module: kernel.module.clone(),
            },
            Slot::Copy { .. } => SequenceNode {
                phase: "transport".to_owned(),
                symbol: rcopy::COPY_SYMBOL.to_owned(),
                module: rcopy::MODULE.to_owned(),
            },
            Slot::Fill { .. } => SequenceNode {
                phase: "transport".to_owned(),
                symbol: rcopy::FILL_SYMBOL.to_owned(),
                module: rcopy::FILL_MODULE.to_owned(),
            },
        })
        .collect()
}

/// The ordered slots (one per program node) of `input`, their origins and
/// whether the window is `Whole16`; refuses an ill-formed window.
#[allow(clippy::type_complexity)]
fn collect_slots<'a>(
    input: &BuildInput<'a>,
) -> Result<(Vec<Slot<'a>>, Vec<NodeOrigin>, bool), BuildRefusal> {
    let whole16 = match input.execution {
        WindowExecution::Whole16 => {
            if input.verify.is_none() {
                return Err(BuildRefusal::whole("a Whole16 window needs a verify plan"));
            }
            true
        }
        WindowExecution::HostSplice => {
            if input.verify.is_some() || !input.prefix.is_empty() {
                return Err(BuildRefusal::whole(
                    "a HostSplice window is the draft segment only (the verify segment is incumbent)",
                ));
            }
            false
        }
    };
    let draft = input.draft;
    if draft.ops.len() != draft.steps.len() {
        return Err(BuildRefusal::whole("draft ops and steps differ in length"));
    }
    // ---- slots -----------------------------------------------------------
    let mut slots: Vec<Slot<'_>> = Vec::new();
    let mut origins: Vec<NodeOrigin> = Vec::new();
    for (i, c) in input.prefix.iter().enumerate() {
        slots.push(Slot::Copy {
            src: Addr::Abs,
            dst: Addr::Abs,
            bytes: Expr::Const(c.bytes),
            planned: (c.src, c.dst, c.bytes),
        });
        origins.push(NodeOrigin::Prefix(i));
    }
    let mut draft_ordinal = 0usize;
    for (i, (op, step)) in draft.ops.iter().zip(&draft.steps).enumerate() {
        let node = slots.len();
        match (op, step) {
            (PlannedOp::Kernel(kernel), WindowStep::Kernel { node: ord }) => {
                if *ord != draft_ordinal {
                    return Err(BuildRefusal::at(
                        node,
                        &kernel.symbol,
                        format!("draft step names kernel ordinal {ord}, expected {draft_ordinal}"),
                    ));
                }
                slots.push(Slot::Kernel {
                    kernel,
                    emission: Emission::Draft,
                    ordinal: draft_ordinal,
                    gdn: Vec::new(),
                });
                draft_ordinal += 1;
            }
            (
                PlannedOp::Copy { dst, src, bytes },
                WindowStep::Copy(CopyStep {
                    src: s,
                    dst: d,
                    bytes: b,
                }),
            ) => {
                slots.push(Slot::Copy {
                    src: Addr::Draft {
                        resource: s.resource,
                        offset: s.byte_offset.clone(),
                    },
                    dst: Addr::Draft {
                        resource: d.resource,
                        offset: d.byte_offset.clone(),
                    },
                    bytes: b.clone(),
                    planned: (*src, *dst, *bytes),
                });
            }
            (PlannedOp::Fill { dst, value, words }, WindowStep::Fill(f)) => {
                slots.push(Slot::Fill {
                    dst: Addr::Draft {
                        resource: f.dst.resource,
                        offset: f.dst.byte_offset.clone(),
                    },
                    value: *value,
                    words: f.words.clone(),
                    planned: (*dst, *words),
                });
            }
            _ => {
                return Err(BuildRefusal::at(
                    node,
                    "<draft>",
                    format!("draft op {i} and its typed step are out of step"),
                ))
            }
        }
        origins.push(NodeOrigin::Draft(i));
    }
    if let Some(v) = input.verify {
        if v.rows != WINDOW_ROWS {
            return Err(BuildRefusal::whole(format!(
                "verify plan has {} rows, a Whole16 window is {WINDOW_ROWS}",
                v.rows
            )));
        }
        let declared: HashMap<usize, &Vec<DeclaredWord>> =
            v.declared_words.iter().map(|(i, w)| (*i, w)).collect();
        for (i, w) in &declared {
            if !matches!(v.ops.get(*i), Some(PlannedOp::Kernel(_))) {
                return Err(BuildRefusal::whole(format!(
                    "declared words name verify op {i}, which is not a kernel ({} words)",
                    w.len()
                )));
            }
        }
        let mut ordinal = 0usize;
        for (i, op) in v.ops.iter().enumerate() {
            match op {
                PlannedOp::Kernel(kernel) => {
                    slots.push(Slot::Kernel {
                        kernel,
                        emission: Emission::Verify,
                        ordinal,
                        gdn: declared.get(&i).map(|w| (*w).clone()).unwrap_or_default(),
                    });
                    ordinal += 1;
                }
                PlannedOp::Copy { dst, src, bytes } => slots.push(Slot::Copy {
                    src: Addr::Abs,
                    dst: Addr::Abs,
                    bytes: Expr::Const(*bytes),
                    planned: (*src, *dst, *bytes),
                }),
                PlannedOp::Fill { dst, value, words } => slots.push(Slot::Fill {
                    dst: Addr::Abs,
                    value: *value,
                    words: Expr::Const(u64::from(*words)),
                    planned: (*dst, *words),
                }),
            }
            origins.push(NodeOrigin::Verify(i));
        }
    }
    debug_assert_eq!(slots.len(), origins.len());
    if slots.is_empty() {
        return Err(BuildRefusal::whole("empty window"));
    }
    Ok((slots, origins, whole16))
}

fn arg_resource(program: &KernelProgram, node: usize, offset: u32) -> Option<(ResourceId, u64)> {
    program
        .nodes
        .get(node)?
        .args
        .iter()
        .find_map(|(o, a)| match a {
            Arg::Resource { resource, interior } if *o == offset => Some((*resource, *interior)),
            _ => None,
        })
}
/// A resolved copy/fill endpoint: the program resource of the authored
/// pointer argument, the (re-based) byte-offset expression and the slack that
/// keeps the bound proof exactly the emitter's extent.
struct Endpoint {
    resource: ResourceId,
    offset: Expr,
    slack: u64,
}

fn endpoint(
    program: &KernelProgram,
    node: usize,
    arg_offset: u32,
    addr: &Addr,
    planned_addr: u64,
    draft: &DraftEmission,
    draft_map: &BTreeMap<ResourceId, Mapped>,
) -> Result<Endpoint, String> {
    let (resource, interior) = arg_resource(program, node, arg_offset)
        .ok_or_else(|| format!("+{arg_offset} is not an authored pointer argument"))?;
    let base = program.resources[resource.0 as usize].base;
    if base + interior != planned_addr {
        return Err(format!(
            "authored pointer {:#x} != planned address {planned_addr:#x}",
            base + interior
        ));
    }
    match addr {
        Addr::Abs => Ok(Endpoint {
            resource,
            offset: Expr::Const(interior),
            slack: 0,
        }),
        Addr::Draft {
            resource: d,
            offset,
        } => {
            let m = draft_map
                .get(d)
                .ok_or_else(|| format!("unknown draft resource {}", d.0))?;
            if m.program != resource {
                return Err(format!(
                    "typed region is in program resource {}, the authored pointer in {}",
                    m.program.0, resource.0
                ));
            }
            let at = eval_at_plan(offset, draft)?;
            if base + m.delta + at != planned_addr {
                return Err(format!(
                    "typed region evaluates to {:#x}, recorded address {planned_addr:#x}",
                    base + m.delta + at
                ));
            }
            Ok(Endpoint {
                resource,
                offset: fold_add(offset, m.delta),
                slack: m.slack,
            })
        }
    }
}

/// A draft emission resource placed in the program: its program resource,
/// its offset inside it (`delta`) and the bytes of the program resource
/// behind it (`slack`).
struct Mapped {
    program: ResourceId,
    delta: u64,
    slack: u64,
}

/// The program resource holding `addr`; a new one is appended (probed) when
/// no authored pointer reached that allocation.
fn resource_of_addr(
    program: &mut KernelProgram,
    probe: &mut Probe<'_>,
    addr: u64,
) -> Result<ResourceId, String> {
    if let Some(i) = program
        .resources
        .iter()
        .position(|b| addr >= b.base && addr - b.base < b.bytes.max(1))
    {
        return Ok(ResourceId(i as u32));
    }
    let b = probe.get(addr)?;
    program.resources.push(b);
    Ok(ResourceId(program.resources.len() as u32 - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dflash_draft_emit::DraftResource;
    use crate::dflash_verify_emit::DflashVerifyPlan;
    use railgun::kernel::{
        ArgFact, ArgRole, FactStatus, KernelFacts, KernelRef, MemClass, ReadCache, WordRole,
    };
    use railgun::plan::{plan, CacheTable, RowPolicy};
    use railgun::window::{
        AllocationIdentity, BoundWindow, HostDeltaSpec, NodeImage, PreparedWindow, QueueFailure,
        Refusal, ResolvedStep, ShapeFamily, Transition, WindowBindingKey, WindowBindings,
        WindowHostDelta, WindowInputs, WindowPlan, WindowQueue, WindowShape,
    };
    use rdna_compute::plan_sink::PlannedKernel;
    use std::collections::BTreeMap;

    const A_STAGING: (u64, u64) = (0x1000_0000, 0x1000);
    const A_TOKENS: (u64, u64) = (0x1001_0000, 0x100);
    const A_POS: (u64, u64) = (0x1002_0000, 0x100);
    const A_X: (u64, u64) = (0x2000_0000, 0x20000);
    const A_OUT: (u64, u64) = (0x3000_0000, 0x1000);
    const A_RING: (u64, u64) = (0x4000_0000, 0x8000);
    const A_VOUT: (u64, u64) = (0x5000_0000, 0x1000);
    /// The draft registers the middle of `A_X` as its own resource.
    const X_DRAFT: (u64, u64) = (0x2000_1000, 0x10000);
    const ALLOCATIONS: [(u64, u64); 7] = [A_STAGING, A_TOKENS, A_POS, A_X, A_OUT, A_RING, A_VOUT];

    fn probe(addr: u64) -> Result<Binding, String> {
        ALLOCATIONS
            .iter()
            .find(|(b, n)| addr >= *b && addr < b + n)
            .map(|(base, bytes)| Binding {
                base: *base,
                bytes: *bytes,
            })
            .ok_or_else(|| format!("{addr:#x} is in no allocation"))
    }

    fn ptr_fact(offset: u32, classes: &[MemClass]) -> ArgFact {
        ArgFact {
            offset,
            size: 8,
            role: ArgRole::Pointer {
                classes: classes.to_vec(),
                read_cache: ReadCache::VmemOnly,
            },
        }
    }

    /// `(ptr read +0, ptr write +8, u32 +16)`, explicit end 20.
    fn facts(symbol: &str) -> Arc<NodeFacts> {
        Arc::new(NodeFacts {
            kref: KernelRef {
                module: "m".into(),
                symbol: symbol.into(),
                artifact_sha256: "00".repeat(32),
                kd_sha256: String::new(),
                receipt_sha256: String::new(),
                tier: "D".into(),
                obligations: vec!["InBoundsNoAlias".into()],
            },
            facts: KernelFacts {
                status: FactStatus::Proven,
                explicit_bytes: 20,
                segment_bytes: 20,
                read_cache: ReadCache::VmemOnly,
                args: vec![
                    ptr_fact(0, &[MemClass::VmemLoad]),
                    ptr_fact(8, &[MemClass::VmemStore]),
                    ArgFact {
                        offset: 16,
                        size: 4,
                        role: ArgRole::Integer,
                    },
                ],
                hidden: Vec::new(),
                hidden_grid_live: Vec::new(),
                differential: "pass".into(),
                guard_flag: None,
            },
        })
    }

    /// The fill kernel: `(ptr write +0, u32 +8, u32 +12)`, explicit end 16.
    fn fill_facts() -> Arc<NodeFacts> {
        let mut f = (*facts(rcopy::FILL_SYMBOL)).clone();
        f.facts.explicit_bytes = 16;
        f.facts.segment_bytes = 16;
        f.facts.args = vec![
            ptr_fact(0, &[MemClass::VmemStore]),
            ArgFact {
                offset: 8,
                size: 4,
                role: ArgRole::Integer,
            },
            ArgFact {
                offset: 12,
                size: 4,
                role: ArgRole::Integer,
            },
        ];
        Arc::new(f)
    }

    struct Source {
        missing: Vec<&'static str>,
        transport: bool,
    }

    impl Source {
        fn ok() -> Self {
            Self {
                missing: Vec::new(),
                transport: true,
            }
        }
    }

    impl WindowFactsSource for Source {
        fn facts(&mut self, _sha: &str, symbol: &str) -> Result<Arc<NodeFacts>, String> {
            if self.missing.contains(&symbol) {
                return Err(format!("no record for {symbol}"));
            }
            Ok(if symbol == rcopy::FILL_SYMBOL {
                fill_facts()
            } else {
                facts(symbol)
            })
        }

        fn transport(&mut self) -> Result<TransportDigests, String> {
            if self.transport {
                Ok(TransportDigests {
                    copy: [0xc0; 32],
                    fill: [0xf1; 32],
                })
            } else {
                Err("railgun_copy is not loaded".into())
            }
        }
    }

    fn kernarg(p0: u64, p8: u64, word: u32) -> Vec<u8> {
        let mut k = vec![0u8; 32];
        k[0..8].copy_from_slice(&p0.to_le_bytes());
        k[8..16].copy_from_slice(&p8.to_le_bytes());
        k[16..20].copy_from_slice(&word.to_le_bytes());
        k
    }

    fn kernel(symbol: &str, kernarg: Vec<u8>, patches: Vec<PatchKind>) -> PlannedKernel {
        PlannedKernel {
            module: "m".into(),
            symbol: symbol.into(),
            code_object_digest: [1; 32],
            grid: [4, 1, 1],
            block: [64, 1, 1],
            dynamic_lds: 0,
            kernarg,
            patches,
            label: "t",
        }
    }

    fn input(name: &str) -> Expr {
        Expr::input(name)
    }

    fn shape() -> WindowShape {
        WindowShape {
            position: 100,
            logical_ctx_rows: 16,
            projected_rows: 4,
            full_cached_rows: 4,
            compact_offset: 0,
            ring_head: 5,
            swa_w: 2048,
            full_w: 2048,
            max_position: 4096,
            max_tiles: 1,
            verify_rows: 16,
            draft_rows: 16,
            generation: 1,
        }
    }

    /// Draft ops 0..4: kernel, dynamic-size copy, constant fill, kernel.
    fn draft() -> DraftEmission {
        let x = |off: u64| X_DRAFT.0 + off;
        let k0 = kernel(
            "k_a",
            kernarg(x(0x100), A_OUT.0, 7),
            vec![
                PatchKind::Interior64 {
                    node: 0,
                    arg_offset: 0,
                    resource: ResourceId(0),
                    byte_offset: Expr::Const(0x100),
                    required_extent: Expr::Const(64),
                },
                PatchKind::Scalar32 {
                    field: ScalarField {
                        node: 0,
                        offset: 16,
                    },
                    expr: input("logical_ctx_rows"),
                    domain: 1..=4096,
                },
            ],
        );
        let k1 = kernel(
            "k_a",
            kernarg(x(0), A_OUT.0 + 0x100, 9),
            vec![PatchKind::Scalar32 {
                field: ScalarField {
                    node: 1,
                    offset: 16,
                },
                expr: Expr::add(input("logical_ctx_rows"), Expr::Const(1)),
                domain: 1..=4096,
            }],
        );
        let bytes = Expr::mul(input("logical_ctx_rows"), Expr::Const(4));
        let ops = vec![
            PlannedOp::Kernel(k0),
            PlannedOp::Copy {
                dst: A_OUT.0 + 0x400,
                src: x(0x40),
                bytes: 64,
            },
            PlannedOp::Fill {
                dst: A_OUT.0 + 0x800,
                value: 0,
                words: 16,
            },
            PlannedOp::Kernel(k1),
        ];
        let region = |r: u32, off: u64| railgun::window::Region {
            resource: ResourceId(r),
            byte_offset: Expr::Const(off),
        };
        let steps = vec![
            WindowStep::Kernel { node: 0 },
            WindowStep::Copy(CopyStep {
                src: region(0, 0x40),
                dst: region(1, 0x400),
                bytes,
            }),
            WindowStep::Fill(railgun::window::FillStep {
                dst: region(1, 0x800),
                value: 0,
                words: Expr::Const(16),
            }),
            WindowStep::Kernel { node: 1 },
        ];
        DraftEmission {
            ops,
            steps,
            resources: vec![
                DraftResource {
                    id: ResourceId(0),
                    name: "x".into(),
                    base: X_DRAFT.0,
                    bytes: X_DRAFT.1,
                    dynamic: true,
                },
                DraftResource {
                    id: ResourceId(1),
                    name: "out".into(),
                    base: A_OUT.0,
                    bytes: A_OUT.1,
                    dynamic: false,
                },
            ],
            host_delta: HostDeltaSpec {
                projection_watermark: Some(railgun::window::TransitionSpec {
                    from: input("projected_rows"),
                    to: input("logical_ctx_rows"),
                }),
                ..HostDeltaSpec::default()
            },
            family: ShapeFamily {
                kernels: vec!["k_a".into(), "k_a".into()],
                projection_rows: vec![12],
            },
            concat_segments: 1,
            l_domain: 1..=100,
            attention_lds_saturated: false,
            scalars_required: Vec::new(),
            completion_surfaces: vec![ResourceId(1)],
            shape: shape(),
            mask_token: 0,
            swa_w: 2048,
        }
    }

    fn verify() -> DflashVerifyPlan {
        let v0 = kernel(
            "k_v0",
            kernarg(A_RING.0, A_VOUT.0, 5),
            vec![PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 16,
                },
                expr: input("ring_head"),
                domain: 0..=63,
            }],
        );
        let v1 = kernel("k_v1", kernarg(A_VOUT.0, A_VOUT.0 + 0x100, 3), Vec::new());
        DflashVerifyPlan {
            ops: vec![PlannedOp::Kernel(v0), PlannedOp::Kernel(v1)],
            forward: 0..1,
            commit: 1..1,
            head: 1..2,
            host_delta: WindowHostDelta {
                ring_head: Some(Transition { from: 5, to: 21 }),
                ring_written: Some(Transition { from: 100, to: 116 }),
                ..WindowHostDelta::default()
            },
            gdn_frame_base: 0,
            gdn_frames_reserved: 1,
            declared_words: vec![(
                1,
                vec![DeclaredWord {
                    offset: 16,
                    role: WordRole::GdnFrame { frames: 1 },
                }],
            )],
            rows: 16,
            start_pos: 100,
            ring_head: 5,
        }
    }

    const PREFIX: [TransportCopy; 2] = [
        TransportCopy {
            src: A_STAGING.0,
            dst: A_TOKENS.0,
            bytes: 4,
        },
        TransportCopy {
            src: A_STAGING.0 + 0x20,
            dst: A_POS.0,
            bytes: 64,
        },
    ];

    fn build(
        draft: &DraftEmission,
        verify: Option<&DflashVerifyPlan>,
        prefix: &[TransportCopy],
        execution: WindowExecution,
        source: &mut Source,
    ) -> Result<BuiltWindow, BuildRefusal> {
        build_window(
            &BuildInput {
                arch: "gfx1201",
                execution,
                prefix,
                draft,
                verify,
                ring_capacity: 64,
                completion_addrs: if verify.is_some() { &[A_VOUT.0] } else { &[] },
                certified: true,
                tape_populated: true,
                sealed_positions: None,
            },
            source,
            &mut probe,
        )
    }

    fn whole16() -> BuiltWindow {
        build(
            &draft(),
            Some(&verify()),
            &PREFIX,
            WindowExecution::Whole16,
            &mut Source::ok(),
        )
        .expect("whole16 builds")
    }

    fn resource_of(program: &KernelProgram, alloc: (u64, u64)) -> ResourceId {
        ResourceId(
            program
                .resources
                .iter()
                .position(|b| b.base == alloc.0 && b.bytes == alloc.1)
                .expect("allocation is a program resource") as u32,
        )
    }

    fn symbols(b: &BuiltWindow) -> Vec<&str> {
        b.parts
            .program
            .nodes
            .iter()
            .map(|n| n.symbol.as_str())
            .collect()
    }

    #[test]
    fn ops_map_to_nodes_in_plan_order() {
        let b = whole16();
        assert_eq!(
            symbols(&b),
            [
                "railgun_copy",
                "railgun_copy",
                "k_a",
                "railgun_copy",
                "railgun_fill_u32",
                "k_a",
                "k_v0",
                "k_v1"
            ]
        );
        assert_eq!(
            b.origins,
            [
                NodeOrigin::Prefix(0),
                NodeOrigin::Prefix(1),
                NodeOrigin::Draft(0),
                NodeOrigin::Draft(1),
                NodeOrigin::Draft(2),
                NodeOrigin::Draft(3),
                NodeOrigin::Verify(0),
                NodeOrigin::Verify(1),
            ]
        );
        let want: Vec<WindowStep> = (0..8).map(|node| WindowStep::Kernel { node }).collect();
        assert_eq!(b.parts.steps, want);
        assert!(b.parts.program.nodes.iter().all(|n| n.lowering.is_pm4()));
        // The fill node writes only its destination; the copy reads/writes.
        let fill = &b.parts.program.nodes[4];
        assert_eq!(fill.effects.len(), 1);
        assert!(fill.effects[0].writes() && !fill.effects[0].reads());
        assert_eq!(b.parts.execution, WindowExecution::Whole16);
        assert!(b.parts.hip_graph_certified && b.parts.tape_populated);
        assert_eq!(b.scalars_required, [SCALAR_RING_WRITTEN]);
        // The node sequence `railgun-cert whole16-manifest` consumes (the test
        // kernels carry the label "t"; an unlabeled kernel is `draft`/`verify`).
        let phases: Vec<(&str, &str)> = b
            .sequence
            .iter()
            .map(|n| (n.phase.as_str(), n.symbol.as_str()))
            .collect();
        assert_eq!(
            phases,
            [
                ("transport", "railgun_copy"),
                ("transport", "railgun_copy"),
                ("t", "k_a"),
                ("transport", "railgun_copy"),
                ("transport", "railgun_fill_u32"),
                ("t", "k_a"),
                ("t", "k_v0"),
                ("t", "k_v1"),
            ]
        );
        assert_eq!(b.sequence[0].module, rcopy::MODULE);
        assert_eq!(b.sequence[4].module, rcopy::FILL_MODULE);
        assert_eq!(b.sequence[2].module, "m");
    }

    #[test]
    fn the_sequence_needs_no_facts() {
        let d = draft();
        let v = verify();
        let input = BuildInput {
            arch: "gfx1201",
            execution: WindowExecution::Whole16,
            prefix: &PREFIX,
            draft: &d,
            verify: Some(&v),
            ring_capacity: 64,
            completion_addrs: &[A_VOUT.0],
            certified: false,
            tape_populated: true,
            sealed_positions: None,
        };
        let seq = window_sequence(&input).expect("sequence");
        assert_eq!(seq, whole16().sequence);
        // Even a window no object of which has a record yet.
        let mut none = Source {
            missing: vec!["k_a", "k_v0", "k_v1", "railgun_copy", "railgun_fill_u32"],
            transport: true,
        };
        assert!(build(&d, Some(&v), &PREFIX, WindowExecution::Whole16, &mut none).is_err());
        assert_eq!(window_sequence(&input).unwrap(), seq);
    }

    #[test]
    fn patch_ordinals_are_rebased_into_the_combined_program() {
        let b = whole16();
        let scalar_nodes: Vec<(usize, u32)> = b
            .parts
            .patches
            .iter()
            .filter_map(|p| match p {
                PatchKind::Scalar32 { field, .. } => Some((field.node, field.offset)),
                _ => None,
            })
            .collect();
        // draft kernel ordinals 0 and 1 -> nodes 2 and 5; the dynamic copy
        // (node 3) carries its byte count; the verify commit word -> node 6.
        assert_eq!(scalar_nodes, [(2, 16), (3, 16), (5, 16), (6, 16)]);
        // Interior64 of draft ordinal 0 now sits on node 2 and names the
        // *program* resource of the allocation that contains the draft
        // buffer, with offset and extent re-based so the bound is the
        // draft buffer's, not the backing allocation's.
        let x = resource_of(&b.parts.program, A_X);
        let delta = X_DRAFT.0 - A_X.0;
        let slack = (A_X.0 + A_X.1) - (X_DRAFT.0 + X_DRAFT.1);
        let interior = b
            .parts
            .patches
            .iter()
            .find_map(|p| match p {
                PatchKind::Interior64 {
                    node: 2,
                    arg_offset: 0,
                    resource,
                    byte_offset,
                    required_extent,
                } => Some((*resource, byte_offset.clone(), required_extent.clone())),
                _ => None,
            })
            .expect("rebased interior patch");
        assert_eq!(
            interior,
            (x, Expr::Const(0x100 + delta), Expr::Const(64 + slack))
        );
        // GdnFrame word of verify ordinal 1 (node 7) stays a GdnFrame word.
        assert_eq!(b.gdn_frames.len(), 1);
        let (id, frames) = b.gdn_frames[0];
        let word = &b.parts.program.words[id.0 as usize];
        assert_eq!((word.node, word.offset, frames), (7, 16, 1));
        // Every Scalar32 patch is a declared Window word, none is untyped.
        assert_eq!(
            b.parts
                .program
                .words
                .iter()
                .filter(|w| w.role == WordRole::Window)
                .count(),
            4
        );
    }

    #[test]
    fn copy_and_fill_nodes_carry_bound_proving_patches() {
        let b = whole16();
        let program = &b.parts.program;
        let x = resource_of(program, A_X);
        let out = resource_of(program, A_OUT);
        let delta = X_DRAFT.0 - A_X.0;
        let slack = (A_X.0 + A_X.1) - (X_DRAFT.0 + X_DRAFT.1);
        let on = |node: usize| -> Vec<&PatchKind> {
            b.parts
                .patches
                .iter()
                .filter(|p| match p {
                    PatchKind::Scalar32 { field, .. } => field.node == node,
                    PatchKind::Interior64 { node: n, .. } | PatchKind::GridAxis { node: n, .. } => {
                        *n == node
                    }
                })
                .collect()
        };
        // Node 3: the dynamic-size draft copy.
        let bytes = Expr::mul(input("logical_ctx_rows"), Expr::Const(4));
        let copy = on(3);
        assert_eq!(copy.len(), 4, "src + dst + byte count + grid");
        assert!(copy.contains(&&PatchKind::Interior64 {
            node: 3,
            arg_offset: 0,
            resource: x,
            byte_offset: Expr::Const(0x40 + delta),
            required_extent: Expr::add(bytes.clone(), Expr::Const(slack)),
        }));
        assert!(copy.contains(&&PatchKind::Interior64 {
            node: 3,
            arg_offset: 8,
            resource: out,
            byte_offset: Expr::Const(0x400),
            required_extent: bytes.clone(),
        }));
        assert!(copy.contains(&&PatchKind::Scalar32 {
            field: ScalarField {
                node: 3,
                offset: 16
            },
            expr: bytes.clone(),
            domain: 1..=rcopy::MAX_COPY_BYTES,
        }));
        assert!(copy.iter().any(|p| matches!(
            p,
            PatchKind::GridAxis { axis: Axis::X, domain, .. } if *domain == (1..=u64::from(rcopy::MAX_GRID_X))
        )));
        // Node 4: the constant fill: destination bound only.
        let fill = on(4);
        assert_eq!(
            fill,
            vec![&PatchKind::Interior64 {
                node: 4,
                arg_offset: 0,
                resource: out,
                byte_offset: Expr::Const(0x800),
                required_extent: Expr::Const(64),
            }]
        );
        // Prefix copies (nodes 0, 1): constant sizes, constant bounds, no
        // word, no grid patch, and the baked grid is the launch builder's.
        assert_eq!(on(0).len(), 2);
        assert_eq!(on(1).len(), 2);
        assert_eq!(program.nodes[1].grid, rcopy::copy_launch(0, 0, 64).grid);
        assert_eq!(program.nodes[1].block, [rcopy::BLOCK, 1, 1]);
        // Planned dynamic copy grid is grid_x(64 B).
        assert_eq!(program.nodes[3].grid, [rcopy::grid_x(64), 1, 1]);
    }

    struct Mock;

    impl WindowQueue for Mock {
        fn submit(&mut self, _: &[NodeImage<'_>], _: &[ResolvedStep]) -> Result<(), QueueFailure> {
            Ok(())
        }
        fn wait(&mut self) -> Result<(), QueueFailure> {
            Ok(())
        }
    }

    fn prepared(b: &BuiltWindow) -> (PreparedWindow, WindowBindingKey) {
        let program = &b.parts.program;
        let pm4 = plan(
            program,
            &CacheTable::for_arch("gfx1201").expect("gfx1201 table"),
            RowPolicy::UnreceiptedSystem,
        );
        assert!(pm4.is_single_segment(program.nodes.len()), "one segment");
        let plan = Arc::new(WindowPlan::new(b.parts.clone(), &pm4).expect("window plan"));
        let key = WindowBindingKey {
            model_id: "m".into(),
            draft_id: "d".into(),
            object_digests: program
                .nodes
                .iter()
                .map(|n| n.kref.as_ref().expect("kref").artifact_sha256.clone())
                .collect(),
            toolchain_digest: "t".into(),
            device: "gfx1201".into(),
            route_flags: 0,
            allocations: program
                .resources
                .iter()
                .enumerate()
                .map(|(i, r)| AllocationIdentity {
                    resource: ResourceId(i as u32),
                    base: r.base,
                    bytes: r.bytes,
                    generation: 1,
                })
                .collect(),
            kv_mapping_generation: 1,
            scratch_capacities: vec![],
            context_mode: "sliding".into(),
            gdn_mode: "ef".into(),
            extraction_layers: vec![],
        };
        let prepared = PreparedWindow::new(plan, key.clone(), Box::new(Mock)).expect("prepared");
        (prepared, key)
    }

    fn inputs(b: &BuiltWindow) -> WindowInputs {
        WindowInputs {
            seed: 7,
            mask_id: 0,
            q_positions: (100..116).collect(),
            k_position_suffix: (84..100).collect(),
            verify_positions: (100..116).collect(),
            scalars: BTreeMap::from([(SCALAR_RING_WRITTEN.to_owned(), 100u64)]),
            gdn_frames: b.gdn_frames.iter().map(|(id, _)| (*id, 77)).collect(),
        }
    }

    fn bind(
        p: &mut PreparedWindow,
        key: &WindowBindingKey,
        shape: &WindowShape,
        inputs: &WindowInputs,
    ) -> Result<BoundWindow, Refusal> {
        p.bind(shape, &WindowBindings { key: key.clone() }, inputs)
    }

    #[test]
    fn copy_extents_are_proven_at_bind() {
        let b = whole16();
        let (mut p, key) = prepared(&b);
        let inputs = inputs(&b);
        let bound = bind(&mut p, &key, &shape(), &inputs).expect("planned shape binds");
        // The dynamic copy's byte word and the draft Scalar32 land in the image.
        let image = p.kernarg_image(3).expect("copy image");
        assert_eq!(u32::from_le_bytes(image[16..20].try_into().unwrap()), 64);
        assert_eq!(p.grid(3), Some([rcopy::grid_x(64), 1, 1]));
        assert_eq!(
            bound.host_delta().ring_head,
            Some(Transition { from: 5, to: 21 })
        );
        assert_eq!(
            bound.host_delta().ring_written,
            Some(Transition { from: 100, to: 116 })
        );
        assert_eq!(
            bound.host_delta().projection_watermark,
            Some(Transition { from: 4, to: 16 })
        );
        // A longer context grows the copy past its destination allocation
        // (0x400 + 4*2000 > 0x1000): refused before work.
        let mut long = shape();
        long.logical_ctx_rows = 2000;
        long.generation = 2;
        let mut long_inputs = inputs.clone();
        long_inputs.k_position_suffix = vec![];
        match bind(&mut p, &key, &long, &long_inputs) {
            Err(Refusal::ExtentExceeded { site, resource, .. }) => {
                assert!(site.contains("node 3 +8"), "{site}");
                assert_eq!(ResourceId(resource), resource_of(&b.parts.program, A_OUT));
            }
            other => panic!("expected an extent refusal, got {other:?}"),
        }
        // A zero-row context drives the byte word out of its 1.. domain.
        let mut zero = shape();
        zero.logical_ctx_rows = 0;
        zero.generation = 3;
        let mut zero_inputs = inputs.clone();
        zero_inputs.k_position_suffix = vec![];
        assert!(matches!(
            bind(&mut p, &key, &zero, &zero_inputs),
            Err(Refusal::OutOfDomain { .. })
        ));
    }

    #[test]
    fn a_node_without_exact_artifact_facts_refuses_the_whole_plan() {
        for (missing, node, symbol) in [
            ("k_v1", 7, "k_v1"),
            ("k_a", 2, "k_a"),
            ("railgun_copy", 0, "railgun_copy"),
            ("railgun_fill_u32", 4, "railgun_fill_u32"),
        ] {
            let mut source = Source {
                missing: vec![missing],
                transport: true,
            };
            let err = build(
                &draft(),
                Some(&verify()),
                &PREFIX,
                WindowExecution::Whole16,
                &mut source,
            )
            .expect_err("a node without facts must refuse");
            assert_eq!(err.node, Some(node), "{err}");
            assert_eq!(err.symbol, symbol);
            assert!(err.reason.contains("no exact-artifact facts"), "{err}");
        }
        // Missing transport kernels refuse a window that needs copies.
        let mut source = Source {
            missing: vec![],
            transport: false,
        };
        let err = build(
            &draft(),
            Some(&verify()),
            &PREFIX,
            WindowExecution::Whole16,
            &mut source,
        )
        .expect_err("no transport");
        assert!(err.reason.contains("transport"), "{err}");
    }

    #[test]
    fn an_unprobed_or_unknown_pointer_does_not_become_a_node() {
        // A kernel pointing outside every allocation cannot be authored.
        let mut d = draft();
        if let PlannedOp::Kernel(k) = &mut d.ops[3] {
            k.kernarg[8..16].copy_from_slice(&0xdead_0000u64.to_le_bytes());
        }
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .expect_err("pointer outside an allocation");
        assert_eq!(err.node, Some(3));
        assert!(err.reason.contains("not a railgun PM4 node"), "{err}");
    }

    #[test]
    fn bad_copies_and_fills_are_refused_at_build() {
        // Zero-byte copy.
        let mut d = draft();
        d.ops[1] = PlannedOp::Copy {
            dst: A_OUT.0 + 0x400,
            src: X_DRAFT.0 + 0x40,
            bytes: 0,
        };
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert_eq!(err.node, Some(1));
        assert!(err.reason.contains("outside 1.."), "{err}");
        // Typed copy size disagrees with the recorded op.
        let mut d = draft();
        d.ops[1] = PlannedOp::Copy {
            dst: A_OUT.0 + 0x400,
            src: X_DRAFT.0 + 0x40,
            bytes: 128,
        };
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert!(err.reason.contains("typed copy size"), "{err}");
        // Unaligned fill.
        let mut d = draft();
        d.ops[2] = PlannedOp::Fill {
            dst: A_OUT.0 + 0x802,
            value: 0,
            words: 16,
        };
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert!(err.reason.contains("not 4-byte aligned"), "{err}");
        // Empty fill.
        let mut d = draft();
        d.ops[2] = PlannedOp::Fill {
            dst: A_OUT.0 + 0x800,
            value: 0,
            words: 0,
        };
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert_eq!(err.symbol, rcopy::FILL_SYMBOL);
        // A constant copy whose two ranges overlap inside one allocation.
        let overlap = [TransportCopy {
            src: A_STAGING.0,
            dst: A_STAGING.0 + 8,
            bytes: 16,
        }];
        let err = build(
            &draft(),
            Some(&verify()),
            &overlap,
            WindowExecution::Whole16,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert!(err.reason.contains("overlap"), "{err}");
    }

    #[test]
    fn an_interior_patch_that_does_not_reproduce_the_planned_pointer_is_refused() {
        let mut d = draft();
        if let PlannedOp::Kernel(k) = &mut d.ops[0] {
            k.patches[0] = PatchKind::Interior64 {
                node: 0,
                arg_offset: 0,
                resource: ResourceId(0),
                byte_offset: Expr::Const(0x108),
                required_extent: Expr::Const(64),
            };
        }
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert_eq!(err.node, Some(0));
        assert!(err.reason.contains("planned pointer"), "{err}");
        // A patch naming another kernel ordinal is refused too.
        let mut d = draft();
        if let PlannedOp::Kernel(k) = &mut d.ops[3] {
            k.patches[0] = PatchKind::Scalar32 {
                field: ScalarField {
                    node: 0,
                    offset: 16,
                },
                expr: input("logical_ctx_rows"),
                domain: 1..=4096,
            };
        }
        let err = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert!(err.reason.contains("kernel ordinal"), "{err}");
    }

    #[test]
    fn host_splice_is_the_draft_segment_only() {
        let d = draft();
        let b = build(
            &d,
            None,
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok(),
        )
        .expect("host splice draft segment");
        assert_eq!(
            symbols(&b),
            ["k_a", "railgun_copy", "railgun_fill_u32", "k_a"],
            "no prefix copies and no verify nodes"
        );
        assert_eq!(b.parts.execution, WindowExecution::HostSplice);
        assert!(
            !b.parts.hip_graph_certified,
            "only a Whole16 route is certified"
        );
        assert!(
            b.parts.host_delta.ring_head.is_none() && b.parts.host_delta.ring_written.is_none()
        );
        assert!(b.parts.host_delta.projection_watermark.is_some());
        assert!(b.scalars_required.is_empty());
        // It lowers as one segment and binds.
        let (mut p, key) = prepared(&b);
        let mut i = inputs(&b);
        i.scalars.clear();
        i.gdn_frames.clear();
        bind(&mut p, &key, &shape(), &i).expect("binds");
        // A verify plan or prefix copies make no sense for it.
        let v = verify();
        assert!(build(
            &d,
            Some(&v),
            &[],
            WindowExecution::HostSplice,
            &mut Source::ok()
        )
        .is_err());
        assert!(build(
            &d,
            None,
            &PREFIX,
            WindowExecution::HostSplice,
            &mut Source::ok()
        )
        .is_err());
        assert!(build(&d, None, &[], WindowExecution::Whole16, &mut Source::ok()).is_err());
    }

    fn note() -> crate::dflash_railgun_builder::PrepNote {
        crate::dflash_railgun_builder::PrepNote {
            hit: false,
            key_hash: 0xabcd,
            fc_rows: vec![12],
            concat_segments: 1,
            l_domain: 1..=100,
        }
    }

    /// The twin runs exactly what the PM4 queue would: the bound node images,
    /// grids and the copy / fill parameters decoded from the bound bytes.
    #[test]
    fn the_twin_executes_the_bound_images_grids_and_transport_parameters() {
        use crate::dflash_railgun_builder::twin_ops;
        let b = whole16();
        let (mut p, key) = prepared(&b);
        let i = inputs(&b);
        bind(&mut p, &key, &shape(), &i).expect("binds");
        let program = p.plan().program();
        let images: Vec<&[u8]> = (0..program.nodes.len())
            .map(|n| p.kernarg_image(n).unwrap())
            .collect();
        let grids: Vec<[u32; 3]> = (0..program.nodes.len())
            .map(|n| p.grid(n).unwrap())
            .collect();
        let ops = twin_ops(program, &images, &grids).expect("twin ops");
        assert_eq!(ops.len(), 8);
        // Prefix copies: the bound absolute addresses and byte counts.
        for (n, c) in PREFIX.iter().enumerate() {
            assert_eq!(
                ops[n],
                PlannedOp::Copy {
                    src: c.src,
                    dst: c.dst,
                    bytes: c.bytes
                },
                "prefix copy {n}"
            );
        }
        // The dynamic draft copy carries the BOUND byte count (4 * ctx rows),
        // not the planned one.
        assert_eq!(
            ops[3],
            PlannedOp::Copy {
                src: X_DRAFT.0 + 0x40,
                dst: A_OUT.0 + 0x400,
                bytes: 64,
            }
        );
        assert_eq!(
            ops[4],
            PlannedOp::Fill {
                dst: A_OUT.0 + 0x800,
                value: 0,
                words: 16
            }
        );
        // A kernel launches with its bound explicit bytes, block and grid.
        let PlannedOp::Kernel(k) = &ops[2] else {
            panic!("node 2 is a kernel")
        };
        assert_eq!(k.symbol, "k_a");
        assert_eq!(k.kernarg.len(), 32, "explicit 20 bytes padded to 16");
        assert_eq!(&k.kernarg[..20], &images[2][..20]);
        assert!(k.kernarg[20..].iter().all(|b| *b == 0));
        assert_eq!((k.grid, k.block), (grids[2], [64, 1, 1]));
        // The bound scalar (logical_ctx_rows = 16) is in the launch.
        assert_eq!(
            u32::from_le_bytes(k.kernarg[16..20].try_into().unwrap()),
            16
        );
        // A shorter image is refused, not silently padded.
        assert!(twin_ops(program, &images[..7], &grids[..7]).is_err());
        let short = [&images[2][..8]];
        let mut one = images.clone();
        one[2] = short[0];
        assert!(twin_ops(program, &one, &grids).is_err());
    }

    /// The debug line names the shape, the skeleton and its cache outcome, the
    /// patched words and the bound pointer range per resource.
    #[test]
    fn the_debug_line_describes_the_bound_window() {
        use crate::dflash_railgun_builder::trace_line;
        let b = whole16();
        let (mut p, key) = prepared(&b);
        let i = inputs(&b);
        let mut s = shape();
        s.generation = 7;
        bind(&mut p, &key, &s, &i).expect("binds");
        let line = trace_line(&p, &s, &note(), true, false);
        for want in [
            "gen=7",
            "pos=100",
            "delta=12", // L=16, projected=4, W=2048: fc_start = max(4, 0)
            "ctx=16",
            "projected=4",
            "ring_head=5",
            "family=fc[12]/cat1",
            "l_domain=1..=100",
            "key=000000000000abcd",
            "cache=miss",
            "nodes=8",
            "manifest=on",
            "certified=false",
        ] {
            assert!(line.contains(want), "{want:?} missing from {line}");
        }
        assert!(line.contains("patched_words="), "{line}");
        assert!(!line.contains("OOB"), "{line}");
        // Every program resource appears with its bound range.
        let program = p.plan().program();
        assert!(
            line.contains(&format!("r{}[", resource_of(program, A_OUT).0)),
            "{line}"
        );
        // The same image digest twice is deterministic; a rebind to another
        // shape changes it.
        assert_eq!(line, trace_line(&p, &s, &note(), true, false));
        let mut s2 = s;
        s2.generation = 8;
        s2.logical_ctx_rows = 32;
        let mut i2 = i.clone();
        i2.k_position_suffix = (68..100).collect();
        bind(&mut p, &key, &s2, &i2).expect("binds at 32 rows");
        assert_ne!(
            trace_line(&p, &s2, &note(), true, false)
                .split("image_digest=")
                .nth(1)
                .unwrap()[..16]
                .to_owned(),
            line.split("image_digest=").nth(1).unwrap()[..16].to_owned()
        );
    }

    #[test]
    fn the_manifest_is_shown_the_cases_a_bind_will_discharge() {
        use crate::dflash_railgun_builder::assumed_cases;
        use railgun::kernel::IndexedNode;
        let mut program = whole16().parts.program;
        assert!(
            assumed_cases(&program).is_empty(),
            "no conditional node, no case"
        );
        program.nodes[3].indexed = Some(IndexedNode {
            args: Vec::new(),
            effects: Vec::new(),
            guards: Vec::new(),
            witnesses: vec!["indexed_bind".into()],
            case_digest: "ab".repeat(32),
        });
        program.nodes[6].indexed = Some(IndexedNode {
            args: Vec::new(),
            effects: Vec::new(),
            guards: Vec::new(),
            witnesses: vec!["indexed_bind".into()],
            case_digest: "cd".repeat(32),
        });
        assert_eq!(
            assumed_cases(&program),
            [(3, "ab".repeat(32)), (6, "cd".repeat(32))],
            "node order, exactly the receipt case digests"
        );
    }

    #[test]
    fn whole16_lowers_to_one_segment_and_binds_with_gdn_frames() {
        let b = whole16();
        let (mut p, key) = prepared(&b);
        let i = inputs(&b);
        bind(&mut p, &key, &shape(), &i).expect("binds");
        let (id, _) = b.gdn_frames[0];
        let node = b.parts.program.words[id.0 as usize].node;
        let image = p.kernarg_image(node).expect("gdn node image");
        assert_eq!(u32::from_le_bytes(image[16..20].try_into().unwrap()), 77);
        let head = p.kernarg_image(6).expect("commit node image");
        assert_eq!(u32::from_le_bytes(head[16..20].try_into().unwrap()), 5);
        // A missing GdnFrame value is refused by bind (the owner reserves it).
        let mut missing = i.clone();
        missing.gdn_frames.clear();
        let mut s = shape();
        s.generation = 2;
        assert!(matches!(
            bind(&mut p, &key, &s, &missing),
            Err(Refusal::Inputs(_))
        ));
        // The stale ring transition proposed by the plan is rejected by the
        // builder, not silently re-based.
        let mut v = verify();
        v.host_delta.ring_head = Some(Transition { from: 5, to: 22 });
        let err = build(
            &draft(),
            Some(&v),
            &PREFIX,
            WindowExecution::Whole16,
            &mut Source::ok(),
        )
        .unwrap_err();
        assert!(err.reason.contains("modulo 64"), "{err}");
    }
}
