// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure mesh splitter: `StepGraph` + `ManifestPlan` + `DeviceMesh` -> `SplitPlan`.
//!
//! Contract: `release-0.4.3/mesh-splitter/CONTRACT.md` §2/§3. No `Gpu`, no HIP,
//! no file I/O, no env reads. The admission pre-pass (epoch, `Ep` axis, aliases,
//! geometry) performs no allocation; every refusal there is a `Copy` error built
//! from ids and numbers only.
//!
//! Interpretations fixed where the frozen contract is silent:
//! * **Layer -> stage.** `ManifestPlan` does not retain `n_layers`, so the
//!   stage of a layer op is recovered from `plan.band_xfers`, which holds one
//!   `(boundary_layer, BandXfer{src,dst})` per real PP boundary: the stage is the
//!   `src` of the earliest boundary with `boundary_layer >= layer`, else the
//!   `dst` of the last boundary, else stage 0 (no PP, or no boundary).
//! * **`Geometry.align`.** For `RowShard` it is the weight resource's packed
//!   `group` (divisibility is checked against `parts * align`); `extent` is the
//!   split-axis extent from `WeightPlacement.logical_shape`. For all other
//!   geometry refusals `align == 1`. Head refusals carry the offending head
//!   count as `extent`; a `HeadLocal` refusal names the op's first read (else
//!   first write, else resource 0) as the representative resource.
//! * **`BandXfer` group** is the source stage devices (ascending) followed by
//!   the destination stage devices (ascending); the pair for TP coordinate `j`
//!   is `group[j] -> group[n + j]`.
//! * **Tp == 1** is not collapsed outside the N=1 identity: a multi-device PP
//!   mesh still emits `Split{Tp,1}`/`Partial{Tp}` layouts and every
//!   Project-used TP schedule entry is fulfilled exactly once (by a one-device
//!   `AllReduceAdd`/`AllReduce`).
//! * A resource consumed by `AllReduceAdd` (fused `dst += Σ src`) exits `Whole`
//!   once its schedule entry is discharged (so no stale `Partial` survives to the
//!   end-of-program check), but its memory is still per-rank addends: any later
//!   read of it is refused with `UnreducedPartial`, and a second reduction of
//!   the same schedule entry is a `ScheduleMismatch`. A later write revives it.
//! * **N=1** (`mesh.n_devices() == 1`): the full admission pre-pass (epoch, `Ep`,
//!   aliases, geometry) runs first even for N=1, then the identity plan is
//!   returned without walking the graph.
//! * **`GatedMlp`** splits the intermediate only when gate/up are
//!   `ColumnShard { axis: 0 }` and down is `RowShard { axis: 1 }`, on more
//!   than one TP part: the output is then `Partial` and binds the down
//!   weight's schedule entry. With one TP part (a PP stage) the op is local
//!   and its output Whole, so no one-device reduction is emitted.

use crate::device_mesh::{CollectiveHint, DeviceMesh, DimKind, MeshEpoch};
use crate::weight_manifest::{ManifestPlan, ShardPolicy, StateKind, WeightPlacement};
use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpId, OpKind, ResourceClass, ResourceId, SplitRule, StateClass, StepGraph,
};
use std::collections::HashSet;
use std::fmt;

/// Layout of one resource across the devices of an op's group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShardSpec {
    /// Complete copy on every device of the op's group.
    Whole,
    /// Contiguous even blocks in group rank order along the resource's split
    /// dim (last dim / heads).
    Split { kind: DimKind, parts: usize },
    /// Full-shape addend per rank; true value = rank-ordered sum.
    Partial { kind: DimKind },
    /// Only on one global device (pinned output stage).
    Owner { device: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CollectiveId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollectiveKind {
    /// Fused `dst += Σ_rank-order operand` (replaces an `Accumulate` op on every group rank).
    AllReduceAdd { into: ResourceId, replaces: OpId },
    /// `operand := Σ_rank-order operand` before its first Whole-requiring consumer.
    AllReduce,
    /// Pipeline boundary copy of `operand` from stage `src` to `dst` (same TP/EP coordinate).
    BandXfer { src: usize, dst: usize },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PlannedCollective {
    pub id: CollectiveId,
    pub kind: CollectiveKind,
    pub operand: ResourceId,
    /// Last writer of `operand`.
    pub producer: OpId,
    /// Global devices; reduction order == this order.
    pub group: Vec<usize>,
    /// Which manifest entry this collective discharges.
    pub fulfils: ScheduleRef,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScheduleRef {
    /// `ManifestPlan.collective_schedule` index.
    Collective(usize),
    /// `ManifestPlan.band_xfers` index.
    Band(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RankItem {
    Op(OpId),
    Collective(CollectiveId),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RankProgram {
    pub device: usize,
    pub items: Vec<RankItem>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResourceShard {
    pub entry: ShardSpec,
    pub exit: ShardSpec,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SplitPlan {
    pub mesh_epoch: MeshEpoch,
    /// Index == global device id, len == `mesh.n_devices()`.
    pub ranks: Vec<RankProgram>,
    /// Issue order; `CollectiveId` == index.
    pub collectives: Vec<PlannedCollective>,
    /// Index == `ResourceId`.
    pub resources: Vec<ResourceShard>,
}

/// Splitter refusal. `Copy`: ids and numbers only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitError {
    EpochMismatch {
        plan: MeshEpoch,
        mesh: MeshEpoch,
    },
    UnsupportedAxis(DimKind),
    Geometry {
        resource: ResourceId,
        extent: usize,
        parts: usize,
        align: usize,
    },
    Alias {
        resource: ResourceId,
    },
    UnknownWeight {
        op: OpId,
        resource: ResourceId,
    },
    UnknownState {
        resource: ResourceId,
    },
    UnsupportedPolicy {
        op: OpId,
        resource: ResourceId,
    },
    UnsupportedOp {
        op: OpId,
        kind: OpKind,
    },
    /// The op's weights are sharded but its kernels cannot run on a shard.
    UnshardableOp {
        op: OpId,
        kind: OpKind,
    },
    LayoutMismatch {
        op: OpId,
        resource: ResourceId,
    },
    NeedsGather {
        op: OpId,
        resource: ResourceId,
    },
    UnreducedPartial {
        resource: ResourceId,
    },
    ScheduleMismatch {
        op: OpId,
    },
    UnfulfilledSchedule(ScheduleRef),
    PlacementMismatch {
        op: OpId,
    },
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EpochMismatch { plan, mesh } => write!(
                f,
                "manifest plan epoch {} does not match mesh epoch {}",
                plan.as_u64(),
                mesh.as_u64()
            ),
            Self::UnsupportedAxis(kind) => {
                write!(f, "mesh axis {kind:?} is not supported by the splitter")
            }
            Self::Geometry {
                resource,
                extent,
                parts,
                align,
            } => write!(
                f,
                "resource {} extent {extent} is not divisible by parts {parts} x align {align}",
                resource.0
            ),
            Self::Alias { resource } => write!(f, "resource {} has an invalid alias", resource.0),
            Self::UnknownWeight { op, resource } => write!(
                f,
                "op {} projects weight resource {} that is not in the manifest plan",
                op.0, resource.0
            ),
            Self::UnknownState { resource } => write!(
                f,
                "state resource {} has no matching manifest state entry",
                resource.0
            ),
            Self::UnsupportedPolicy { op, resource } => write!(
                f,
                "op {} uses weight resource {} with a policy that cannot be projected",
                op.0, resource.0
            ),
            Self::UnsupportedOp { op, kind } => write!(
                f,
                "op {} ({kind:?}) has no declared split point and a non-whole operand",
                op.0
            ),
            Self::UnshardableOp { op, kind } => write!(
                f,
                "op {} ({kind:?}) has sharded weights but its kernel ladder cannot run on a shard",
                op.0
            ),
            Self::LayoutMismatch { op, resource } => write!(
                f,
                "op {} sees an incompatible layout on resource {}",
                op.0, resource.0
            ),
            Self::NeedsGather { op, resource } => write!(
                f,
                "op {} needs resource {} whole but it is split (no all-gather in this slice)",
                op.0, resource.0
            ),
            Self::UnreducedPartial { resource } => write!(
                f,
                "resource {} is still a partial sum at end of program",
                resource.0
            ),
            Self::ScheduleMismatch { op } => write!(
                f,
                "op {} has a reduction or transfer that does not bind to exactly one manifest schedule entry",
                op.0
            ),
            Self::UnfulfilledSchedule(schedule) => {
                write!(f, "manifest schedule entry {schedule:?} was never fulfilled")
            }
            Self::PlacementMismatch { op } => write!(
                f,
                "op {} operands are placed on devices that do not match its group",
                op.0
            ),
        }
    }
}

impl std::error::Error for SplitError {}

// ── Admission pre-pass (allocation-free) ──────────────────────────────────

fn numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn find_weight<'a>(plan: &'a ManifestPlan, class: &ResourceClass) -> Option<&'a WeightPlacement> {
    match class {
        ResourceClass::Weight { name, layer, .. } => plan
            .weights
            .iter()
            .find(|w| w.name == *name && w.layer == *layer),
        _ => None,
    }
}

fn splits_target(policy: &ShardPolicy) -> bool {
    matches!(
        policy,
        ShardPolicy::ColumnShard { .. }
            | ShardPolicy::RowShard { .. }
            | ShardPolicy::VocabShard { .. }
            | ShardPolicy::FusedQkv { .. }
            | ShardPolicy::HeadSharded { .. }
    )
}

fn weight_geometry(
    graph: &StepGraph,
    plan: &ManifestPlan,
    parts: usize,
    weight: ResourceId,
) -> Result<(), SplitError> {
    let Some(decl) = graph.resources().get(weight.0 as usize) else {
        return Ok(());
    };
    let Some(w) = find_weight(plan, &decl.class) else {
        return Ok(());
    };
    let group = match decl.class {
        ResourceClass::Weight { group, .. } => group,
        _ => 1,
    };
    let bad = |extent: usize, align: usize| SplitError::Geometry {
        resource: weight,
        extent,
        parts,
        align,
    };
    match &w.policy {
        ShardPolicy::ColumnShard { axis } | ShardPolicy::VocabShard { axis } => {
            let Some(&extent) = w.logical_shape.get(*axis) else {
                return Err(bad(0, 1));
            };
            if extent % parts != 0 {
                return Err(bad(extent, 1));
            }
        }
        ShardPolicy::RowShard { axis } => {
            let Some(&extent) = w.logical_shape.get(*axis) else {
                return Err(bad(0, group));
            };
            let Some(unit) = parts.checked_mul(group).filter(|&u| u != 0) else {
                return Err(bad(extent, group));
            };
            if extent % unit != 0 {
                return Err(bad(extent, group));
            }
        }
        ShardPolicy::FusedQkv {
            q_heads, kv_heads, ..
        } => {
            if q_heads % parts != 0 {
                return Err(bad(*q_heads, 1));
            }
            if kv_heads % parts != 0 {
                return Err(bad(*kv_heads, 1));
            }
        }
        ShardPolicy::HeadSharded { n_heads, .. } if n_heads % parts != 0 => {
            return Err(bad(*n_heads, 1));
        }
        _ => {}
    }
    Ok(())
}

/// `GatedMlp` admission: the three weights split the intermediate together
/// (gate/up `ColumnShard { axis: 0 }`, down `RowShard { axis: 1 }`) or not at
/// all (all `Replicate`), over one intermediate extent, on a shard-safe
/// ladder, and every split weight passes [`weight_geometry`]
/// (`intermediate % (parts * group)` for down). A weight missing from the plan
/// is left to the walk (`UnknownWeight`).
fn gated_mlp_admission(
    graph: &StepGraph,
    plan: &ManifestPlan,
    parts: usize,
    op: OpId,
    weights: [ResourceId; 3],
    ladder: hipfire_dispatch::pipeline::gated_mlp::MlpLadder,
) -> Result<(), SplitError> {
    let placement = |w: ResourceId| {
        graph
            .resources()
            .get(w.0 as usize)
            .and_then(|decl| find_weight(plan, &decl.class))
    };
    let [Some(gate), Some(up), Some(down)] = weights.map(placement) else {
        return Ok(());
    };
    let placed = [gate, up, down];
    if placed.iter().all(|w| w.policy == ShardPolicy::Replicate) {
        return Ok(());
    }
    let split = [
        ShardPolicy::ColumnShard { axis: 0 },
        ShardPolicy::ColumnShard { axis: 0 },
        ShardPolicy::RowShard { axis: 1 },
    ];
    for ((w, placement), want) in weights.iter().zip(placed).zip(&split) {
        if placement.policy != *want {
            return Err(SplitError::UnsupportedPolicy { op, resource: *w });
        }
    }
    let intermediate = gate.logical_shape.first();
    if up.logical_shape.first() != intermediate {
        return Err(SplitError::LayoutMismatch {
            op,
            resource: weights[1],
        });
    }
    if down.logical_shape.get(1) != intermediate {
        return Err(SplitError::LayoutMismatch {
            op,
            resource: weights[2],
        });
    }
    if parts > 1 && !ladder.tp_shardable() {
        return Err(SplitError::UnshardableOp {
            op,
            kind: OpKind::GatedMlp,
        });
    }
    for w in weights {
        weight_geometry(graph, plan, parts, w)?;
    }
    Ok(())
}

/// `HeadMixer` admission: either every placed weight of the op is
/// `Replicate` (the op stays whole), or `out_proj` is `RowShard { axis: 1 }`
/// and every other placed weight of the op is a head split
/// (`HeadSharded`, `FusedQkv`, `ColumnShard { axis: 0 }`); then `heads` and
/// `kv_heads` must split evenly over `parts` and every split weight passes
/// [`weight_geometry`]. Every weight the op reads (including `out_proj`) must
/// be planned, else `UnknownWeight`.
#[allow(clippy::too_many_arguments)]
fn head_mixer_admission(
    graph: &StepGraph,
    plan: &ManifestPlan,
    parts: usize,
    op: OpId,
    decl: &OpDecl,
    out_proj: ResourceId,
    heads: usize,
    kv_heads: usize,
) -> Result<(), SplitError> {
    let resource = |w: ResourceId| graph.resources().get(w.0 as usize);
    let placement = |w: ResourceId| resource(w).and_then(|r| find_weight(plan, &r.class));
    // A split rank would otherwise run a head projection it holds no slice of.
    for &w in decl.reads.iter().chain([&out_proj]) {
        let weight = resource(w).is_some_and(|r| matches!(r.class, ResourceClass::Weight { .. }));
        if weight && placement(w).is_none() {
            return Err(SplitError::UnknownWeight { op, resource: w });
        }
    }
    let Some(out) = placement(out_proj) else {
        return Ok(());
    };
    let weights = || {
        decl.reads
            .iter()
            .copied()
            .filter(|&w| w != out_proj)
            .filter_map(|w| placement(w).map(|p| (w, p)))
    };
    if out.policy == ShardPolicy::Replicate {
        return match weights().find(|(_, p)| p.policy != ShardPolicy::Replicate) {
            Some((resource, _)) => Err(SplitError::UnsupportedPolicy { op, resource }),
            None => Ok(()),
        };
    }
    if out.policy != (ShardPolicy::RowShard { axis: 1 }) {
        return Err(SplitError::UnsupportedPolicy {
            op,
            resource: out_proj,
        });
    }
    for (resource, p) in weights() {
        let head_split = matches!(
            p.policy,
            ShardPolicy::HeadSharded { .. }
                | ShardPolicy::FusedQkv { .. }
                | ShardPolicy::ColumnShard { axis: 0 }
        );
        if !head_split {
            return Err(SplitError::UnsupportedPolicy { op, resource });
        }
    }
    for extent in [heads, kv_heads] {
        if extent == 0 || extent % parts != 0 {
            return Err(SplitError::Geometry {
                resource: out_proj,
                extent,
                parts,
                align: 1,
            });
        }
    }
    weight_geometry(graph, plan, parts, out_proj)?;
    for (resource, _) in weights() {
        weight_geometry(graph, plan, parts, resource)?;
    }
    Ok(())
}

fn admit(graph: &StepGraph, plan: &ManifestPlan, mesh: &DeviceMesh) -> Result<(), SplitError> {
    if plan.mesh_epoch != mesh.epoch() {
        return Err(SplitError::EpochMismatch {
            plan: plan.mesh_epoch,
            mesh: mesh.epoch(),
        });
    }
    if mesh
        .axes()
        .iter()
        .any(|axis| axis.kind == DimKind::Ep && axis.size > 1)
    {
        return Err(SplitError::UnsupportedAxis(DimKind::Ep));
    }

    // Static alias checks first, for every alias, before any geometry.
    let resources = graph.resources();
    for (index, decl) in resources.iter().enumerate() {
        let Some(alias) = decl.alias else { continue };
        let resource = ResourceId(index as u32);
        let of = alias.of.0 as usize;
        if of >= index {
            return Err(SplitError::Alias { resource });
        }
        let (Some(alias_n), Some(target_n)) = (numel(&decl.shape), numel(&resources[of].shape))
        else {
            return Err(SplitError::Alias { resource });
        };
        match alias.offset.checked_add(alias_n) {
            Some(end) if end <= target_n => {}
            _ => return Err(SplitError::Alias { resource }),
        }
    }

    let parts = mesh.size_of(DimKind::Tp);
    for (index, op) in graph.ops().iter().enumerate() {
        match op.rule {
            SplitRule::Project { weight, .. } => weight_geometry(graph, plan, parts, weight)?,
            SplitRule::GatedMlp {
                gate,
                up,
                down,
                ladder,
                ..
            } => gated_mlp_admission(
                graph,
                plan,
                parts,
                OpId(index as u32),
                [gate, up, down],
                ladder,
            )?,
            SplitRule::HeadMixer {
                out_proj,
                heads,
                kv_heads,
                ..
            } => head_mixer_admission(
                graph,
                plan,
                parts,
                OpId(index as u32),
                op,
                out_proj,
                heads,
                kv_heads,
            )?,
            SplitRule::HeadLocal { heads, kv_heads } => {
                let resource = op
                    .reads
                    .first()
                    .or_else(|| op.writes.first())
                    .copied()
                    .unwrap_or(ResourceId(0));
                if heads % parts != 0 {
                    return Err(SplitError::Geometry {
                        resource,
                        extent: heads,
                        parts,
                        align: 1,
                    });
                }
                if kv_heads % parts != 0 {
                    return Err(SplitError::Geometry {
                        resource,
                        extent: kv_heads,
                        parts,
                        align: 1,
                    });
                }
            }
            _ => {}
        }
    }

    // Declared split geometry: an alias view of a split weight must stay
    // inside one per-rank block (contiguous even blocks of the target).
    for (index, decl) in resources.iter().enumerate() {
        let Some(alias) = decl.alias else { continue };
        let target = &resources[alias.of.0 as usize];
        let Some(w) = find_weight(plan, &target.class) else {
            continue;
        };
        if !splits_target(&w.policy) {
            continue;
        }
        let (Some(target_n), Some(alias_n)) = (numel(&target.shape), numel(&decl.shape)) else {
            continue;
        };
        if target_n % parts != 0 || alias_n == 0 {
            continue;
        }
        let block = target_n / parts;
        if block == 0 {
            continue;
        }
        if alias.offset / block != (alias.offset + alias_n - 1) / block {
            return Err(SplitError::Alias {
                resource: ResourceId(index as u32),
            });
        }
    }
    Ok(())
}

// ── Layout helpers ────────────────────────────────────────────────────────

fn tp_split(tp: usize) -> ShardSpec {
    ShardSpec::Split {
        kind: DimKind::Tp,
        parts: tp,
    }
}

fn is_pinned(policy: &ShardPolicy) -> bool {
    matches!(policy, ShardPolicy::Pin(_) | ShardPolicy::Tied { .. })
}

fn weight_layout(policy: &ShardPolicy, devices: &[usize], tp: usize) -> ShardSpec {
    match policy {
        ShardPolicy::ColumnShard { .. }
        | ShardPolicy::RowShard { .. }
        | ShardPolicy::VocabShard { .. }
        | ShardPolicy::FusedQkv { .. }
        | ShardPolicy::HeadSharded { .. }
        | ShardPolicy::ExpertTensorSharded { .. } => tp_split(tp),
        ShardPolicy::Pin(_) | ShardPolicy::Tied { .. } => match devices {
            [device] if tp > 1 => ShardSpec::Owner { device: *device },
            _ => ShardSpec::Whole,
        },
        ShardPolicy::Replicate | ShardPolicy::ExpertSharded { .. } => ShardSpec::Whole,
    }
}

fn state_matches(kind: &StateKind, class: StateClass) -> bool {
    matches!(
        (kind, class),
        (StateKind::Kv { .. }, StateClass::Kv)
            | (StateKind::Recurrent, StateClass::Recurrent)
            | (StateKind::Conv, StateClass::Conv)
    )
}

fn reads_of(op: &OpDecl) -> impl Iterator<Item = ResourceId> + '_ {
    let extra: [Option<ResourceId>; 2] = match op.rule {
        SplitRule::Accumulate { dst, src } => [Some(dst), Some(src)],
        SplitRule::Project { input, .. }
        | SplitRule::GatedMlp { input, .. }
        | SplitRule::HeadMixer { input, .. } => [Some(input), None],
        _ => [None, None],
    };
    op.reads
        .iter()
        .copied()
        .chain(IntoIterator::into_iter(extra).flatten())
}

fn operands_of(op: &OpDecl) -> impl Iterator<Item = ResourceId> + '_ {
    op.reads.iter().chain(op.writes.iter()).copied()
}

// ── Walk ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Run {
    Group,
    One(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Plain,
    State,
    Input,
    Weight,
}

#[derive(Clone, Copy)]
struct PartialInfo {
    sched: Option<usize>,
}

fn out_whole(run: Run) -> ShardSpec {
    match run {
        Run::One(device) => ShardSpec::Owner { device },
        Run::Group => ShardSpec::Whole,
    }
}

struct Walk<'a> {
    graph: &'a StepGraph,
    plan: &'a ManifestPlan,
    tp: usize,
    pp: usize,
    groups: Vec<Vec<usize>>,
    dev_stage: Vec<usize>,
    cur: Vec<Option<ShardSpec>>,
    entry: Vec<Option<ShardSpec>>,
    last_write: Vec<Option<(usize, OpId)>>,
    partial: Vec<Option<PartialInfo>>,
    ranks: Vec<RankProgram>,
    collectives: Vec<PlannedCollective>,
    sched_used: Vec<bool>,
    sched_done: Vec<bool>,
    band_done: Vec<bool>,
    band_crossed: Vec<bool>,
    xfer_done: HashSet<(u32, u32, usize)>,
    /// Partial sources already consumed by a fused `AllReduceAdd`: their
    /// memory is per-rank addends, so any later read is refused.
    spent: Vec<bool>,
}

impl<'a> Walk<'a> {
    fn new(graph: &'a StepGraph, plan: &'a ManifestPlan, mesh: &DeviceMesh) -> Self {
        let tp = mesh.size_of(DimKind::Tp);
        let pp = mesh.size_of(DimKind::Pp);
        let n_devices = mesh.n_devices();
        let stride: usize = match mesh.axes().iter().position(|a| a.kind == DimKind::Pp) {
            Some(index) => mesh.axes()[index + 1..].iter().map(|a| a.size).product(),
            None => 1,
        };
        let dev_stage: Vec<usize> = (0..n_devices)
            .map(|d| if pp <= 1 { 0 } else { (d / stride) % pp })
            .collect();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); pp.max(1)];
        for (device, &stage) in dev_stage.iter().enumerate() {
            groups[stage].push(device);
        }

        let mut cur = Vec::with_capacity(graph.resources().len());
        for decl in graph.resources() {
            let initial = match &decl.class {
                ResourceClass::Weight { .. } => Some(match find_weight(plan, &decl.class) {
                    Some(w) => weight_layout(&w.policy, &w.devices, tp),
                    None => ShardSpec::Whole,
                }),
                ResourceClass::State { .. } => None,
                _ => Some(ShardSpec::Whole),
            };
            cur.push(initial);
        }
        let n_resources = cur.len();
        Self {
            graph,
            plan,
            tp,
            pp,
            groups,
            dev_stage,
            entry: cur.clone(),
            cur,
            last_write: vec![None; n_resources],
            partial: vec![None; n_resources],
            ranks: (0..n_devices)
                .map(|device| RankProgram {
                    device,
                    items: Vec::new(),
                })
                .collect(),
            collectives: Vec::new(),
            sched_used: vec![false; plan.collective_schedule.len()],
            sched_done: vec![false; plan.collective_schedule.len()],
            band_done: vec![false; plan.band_xfers.len()],
            band_crossed: vec![false; plan.band_xfers.len()],
            xfer_done: HashSet::new(),
            spent: vec![false; n_resources],
        }
    }

    /// Stage of `layer`, recovered from the PP boundaries recorded in the plan.
    fn layer_stage(&self, layer: usize) -> usize {
        if self.pp <= 1 {
            return 0;
        }
        for (boundary, hint) in &self.plan.band_xfers {
            if *boundary >= layer {
                if let CollectiveHint::BandXfer { src, .. } = hint {
                    return *src;
                }
            }
        }
        for (_, hint) in self.plan.band_xfers.iter().rev() {
            if let CollectiveHint::BandXfer { dst, .. } = hint {
                return *dst;
            }
        }
        0
    }

    fn weight_of(&self, resource: ResourceId) -> Option<&'a WeightPlacement> {
        let plan: &'a ManifestPlan = self.plan;
        let graph: &'a StepGraph = self.graph;
        find_weight(plan, &graph.resources().get(resource.0 as usize)?.class)
    }

    fn kind_of(&self, resource: ResourceId) -> Kind {
        match self.graph.resources()[resource.0 as usize].class {
            ResourceClass::Activation | ResourceClass::Output => Kind::Plain,
            ResourceClass::State { .. } => Kind::State,
            ResourceClass::Input => Kind::Input,
            ResourceClass::Weight { .. } => Kind::Weight,
        }
    }

    fn sched_entry(&self, w: &WeightPlacement) -> Option<usize> {
        let layer = w.layer?;
        self.plan.collective_schedule.iter().position(|e| {
            e.name == w.name
                && e.layer == layer
                && e.hint == CollectiveHint::AllReduce { kind: DimKind::Tp }
        })
    }

    fn run(mut self) -> Result<SplitPlan, SplitError> {
        let graph: &'a StepGraph = self.graph;
        let mut prev_stage = 0usize;
        for (index, op) in graph.ops().iter().enumerate() {
            let opid = OpId(index as u32);
            let stage = self.op_stage(opid, op, prev_stage)?;
            let run = self.resolve_run(opid, op, stage)?;
            self.check_states(opid, op, stage)?;
            if index > 0 && stage != prev_stage {
                self.mark_crossed(prev_stage, stage);
            }
            for resource in reads_of(op) {
                if self.spent[resource.0 as usize] {
                    return Err(SplitError::UnreducedPartial { resource });
                }
            }
            self.cross_stage_reads(opid, op, stage)?;
            self.reduce_partials(opid, op)?;
            self.apply_rule(opid, op, stage, run)?;
            prev_stage = stage;
        }
        self.finish()
    }

    fn op_stage(&self, opid: OpId, op: &OpDecl, prev: usize) -> Result<usize, SplitError> {
        if let Some(layer) = op.layer {
            return Ok(self.layer_stage(layer));
        }
        for resource in operands_of(op) {
            if let Some(w) = self.weight_of(resource) {
                let device = *w
                    .devices
                    .first()
                    .ok_or(SplitError::PlacementMismatch { op: opid })?;
                return self
                    .dev_stage
                    .get(device)
                    .copied()
                    .ok_or(SplitError::PlacementMismatch { op: opid });
            }
        }
        Ok(prev)
    }

    fn resolve_run(&self, opid: OpId, op: &OpDecl, stage: usize) -> Result<Run, SplitError> {
        let mismatch = SplitError::PlacementMismatch { op: opid };
        let group = &self.groups[stage];
        let mut pinned: Option<usize> = None;
        for resource in operands_of(op) {
            let Some(w) = self.weight_of(resource) else {
                continue;
            };
            if is_pinned(&w.policy) {
                let [device] = w.devices.as_slice() else {
                    return Err(mismatch);
                };
                if !group.contains(device) {
                    return Err(mismatch);
                }
                if pinned.is_some_and(|p| p != *device) {
                    return Err(mismatch);
                }
                pinned = Some(*device);
            } else if w.devices != *group {
                return Err(mismatch);
            }
        }
        Ok(match pinned {
            Some(device) if group.len() > 1 => Run::One(device),
            _ => Run::Group,
        })
    }

    fn check_states(&self, opid: OpId, op: &OpDecl, stage: usize) -> Result<(), SplitError> {
        for resource in operands_of(op) {
            let ResourceClass::State { class, layer } =
                self.graph.resources()[resource.0 as usize].class
            else {
                continue;
            };
            let Some((_, devices)) = self.plan.state.iter().find(|(entry, _)| {
                entry.layer() == Some(layer) && state_matches(&entry.kind, class)
            }) else {
                return Err(SplitError::UnknownState { resource });
            };
            if *devices != self.groups[stage] {
                return Err(SplitError::PlacementMismatch { op: opid });
            }
        }
        Ok(())
    }

    fn mark_crossed(&mut self, src: usize, dst: usize) {
        if self.pp <= 1 {
            return;
        }
        let hint = CollectiveHint::BandXfer { src, dst };
        for (index, (_, h)) in self.plan.band_xfers.iter().enumerate() {
            if *h == hint {
                self.band_crossed[index] = true;
            }
        }
    }

    fn push_collective(
        &mut self,
        kind: CollectiveKind,
        operand: ResourceId,
        producer: OpId,
        group: Vec<usize>,
        fulfils: ScheduleRef,
    ) {
        let id = CollectiveId(self.collectives.len() as u32);
        for &device in &group {
            self.ranks[device].items.push(RankItem::Collective(id));
        }
        self.collectives.push(PlannedCollective {
            id,
            kind,
            operand,
            producer,
            group,
            fulfils,
        });
    }

    fn emit(&mut self, opid: OpId, stage: usize, run: Run) {
        match run {
            Run::One(device) => self.ranks[device].items.push(RankItem::Op(opid)),
            Run::Group => {
                for i in 0..self.groups[stage].len() {
                    let device = self.groups[stage][i];
                    self.ranks[device].items.push(RankItem::Op(opid));
                }
            }
        }
    }

    /// Insert one `AllReduce` for a Partial resource, bound to the schedule
    /// entry of the RowShard weight that produced it.
    fn reduce(&mut self, opid: OpId, resource: ResourceId) -> Result<(), SplitError> {
        let mismatch = SplitError::ScheduleMismatch { op: opid };
        let ri = resource.0 as usize;
        let entry = self.partial[ri].and_then(|p| p.sched).ok_or(mismatch)?;
        if self.sched_done[entry] {
            return Err(mismatch);
        }
        self.sched_done[entry] = true;
        let (stage, producer) = self.last_write[ri].ok_or(mismatch)?;
        let group = self.groups[stage].clone();
        self.push_collective(
            CollectiveKind::AllReduce,
            resource,
            producer,
            group,
            ScheduleRef::Collective(entry),
        );
        self.cur[ri] = Some(ShardSpec::Whole);
        self.partial[ri] = None;
        Ok(())
    }

    fn cross_stage_reads(
        &mut self,
        opid: OpId,
        op: &OpDecl,
        stage: usize,
    ) -> Result<(), SplitError> {
        for resource in reads_of(op) {
            if self.kind_of(resource) != Kind::Plain {
                continue;
            }
            let ri = resource.0 as usize;
            let Some((src, producer)) = self.last_write[ri] else {
                continue;
            };
            if src == stage {
                continue;
            }
            match self.cur[ri] {
                Some(ShardSpec::Owner { .. }) => {
                    return Err(SplitError::LayoutMismatch { op: opid, resource })
                }
                Some(ShardSpec::Partial { .. }) => self.reduce(opid, resource)?,
                _ => {}
            }
            if !self.xfer_done.insert((resource.0, producer.0, stage)) {
                continue;
            }
            let hint = CollectiveHint::BandXfer { src, dst: stage };
            let band = self
                .plan
                .band_xfers
                .iter()
                .position(|(_, h)| *h == hint)
                .ok_or(SplitError::ScheduleMismatch { op: opid })?;
            if self.band_done[band] {
                return Err(SplitError::ScheduleMismatch { op: opid });
            }
            self.band_done[band] = true;
            self.band_crossed[band] = true;
            let mut group = self.groups[src].clone();
            group.extend_from_slice(&self.groups[stage]);
            self.push_collective(
                CollectiveKind::BandXfer { src, dst: stage },
                resource,
                producer,
                group,
                ScheduleRef::Band(band),
            );
        }
        Ok(())
    }

    fn reduce_partials(&mut self, opid: OpId, op: &OpDecl) -> Result<(), SplitError> {
        let skip = match op.rule {
            SplitRule::Accumulate { src, .. } => Some(src),
            _ => None,
        };
        for resource in reads_of(op) {
            if self.kind_of(resource) != Kind::Plain || Some(resource) == skip {
                continue;
            }
            if matches!(
                self.cur[resource.0 as usize],
                Some(ShardSpec::Partial { .. })
            ) {
                self.reduce(opid, resource)?;
            }
        }
        Ok(())
    }

    fn adopt(&mut self, resource: ResourceId, spec: ShardSpec) {
        let ri = resource.0 as usize;
        if self.cur[ri].is_none() {
            self.cur[ri] = Some(spec);
            self.entry[ri] = Some(spec);
        }
    }

    fn require_whole(
        &mut self,
        opid: OpId,
        resource: ResourceId,
        run: Run,
    ) -> Result<(), SplitError> {
        match self.cur[resource.0 as usize] {
            None => {
                self.adopt(resource, ShardSpec::Whole);
                Ok(())
            }
            Some(ShardSpec::Whole) => Ok(()),
            Some(ShardSpec::Owner { device }) if run == Run::One(device) => Ok(()),
            Some(ShardSpec::Owner { .. }) => Err(SplitError::LayoutMismatch { op: opid, resource }),
            Some(ShardSpec::Split { .. }) => Err(SplitError::NeedsGather { op: opid, resource }),
            Some(ShardSpec::Partial { .. }) => Err(SplitError::UnreducedPartial { resource }),
        }
    }

    fn require_input_whole(
        &mut self,
        opid: OpId,
        resource: ResourceId,
        run: Run,
    ) -> Result<(), SplitError> {
        if self.kind_of(resource) == Kind::Input {
            return Ok(());
        }
        self.require_whole(opid, resource, run)
    }

    fn require_input_split(&mut self, opid: OpId, resource: ResourceId) -> Result<(), SplitError> {
        if self.kind_of(resource) == Kind::Input {
            return Ok(());
        }
        let want = tp_split(self.tp);
        match self.cur[resource.0 as usize] {
            None => {
                self.adopt(resource, want);
                Ok(())
            }
            Some(spec) if spec == want => Ok(()),
            Some(ShardSpec::Partial { .. }) => Err(SplitError::UnreducedPartial { resource }),
            Some(_) => Err(SplitError::LayoutMismatch { op: opid, resource }),
        }
    }

    fn write_res(
        &mut self,
        opid: OpId,
        resource: ResourceId,
        spec: ShardSpec,
        stage: usize,
    ) -> Result<(), SplitError> {
        let ri = resource.0 as usize;
        if self.partial[ri].is_some() {
            return Err(SplitError::UnreducedPartial { resource });
        }
        self.spent[ri] = false;
        if self.kind_of(resource) == Kind::State {
            match self.cur[ri] {
                None => {
                    self.cur[ri] = Some(spec);
                    self.entry[ri] = Some(spec);
                }
                Some(c) if c == spec => {}
                Some(_) => return Err(SplitError::LayoutMismatch { op: opid, resource }),
            }
        } else {
            self.cur[ri] = Some(spec);
        }
        self.partial[ri] = None;
        self.last_write[ri] = Some((stage, opid));
        Ok(())
    }

    fn apply_rule(
        &mut self,
        opid: OpId,
        op: &OpDecl,
        stage: usize,
        run: Run,
    ) -> Result<(), SplitError> {
        match op.rule {
            SplitRule::Project {
                weight,
                input,
                output,
            } => {
                let w = self.weight_of(weight).ok_or(SplitError::UnknownWeight {
                    op: opid,
                    resource: weight,
                })?;
                let out = match &w.policy {
                    ShardPolicy::Replicate | ShardPolicy::Pin(_) => {
                        self.require_input_whole(opid, input, run)?;
                        out_whole(run)
                    }
                    ShardPolicy::ColumnShard { .. }
                    | ShardPolicy::FusedQkv { .. }
                    | ShardPolicy::HeadSharded { .. }
                    | ShardPolicy::VocabShard { .. } => {
                        self.require_input_whole(opid, input, run)?;
                        tp_split(self.tp)
                    }
                    ShardPolicy::RowShard { .. } => {
                        if let Some(entry) = self.sched_entry(w) {
                            self.sched_used[entry] = true;
                        }
                        self.require_input_split(opid, input)?;
                        ShardSpec::Partial { kind: DimKind::Tp }
                    }
                    _ => {
                        return Err(SplitError::UnsupportedPolicy {
                            op: opid,
                            resource: weight,
                        })
                    }
                };
                self.write_res(opid, output, out, stage)?;
                if matches!(out, ShardSpec::Partial { .. }) {
                    let sched = self.sched_entry(w);
                    self.partial[output.0 as usize] = Some(PartialInfo { sched });
                }
                self.emit(opid, stage, run);
            }
            SplitRule::Accumulate { dst, src } => {
                self.require_whole(opid, dst, run)?;
                let si = src.0 as usize;
                if matches!(self.cur[si], Some(ShardSpec::Partial { .. })) {
                    let mismatch = SplitError::ScheduleMismatch { op: opid };
                    let entry = self.partial[si].and_then(|p| p.sched).ok_or(mismatch)?;
                    if self.sched_done[entry] {
                        return Err(mismatch);
                    }
                    self.sched_done[entry] = true;
                    let producer = self.last_write[si].map_or(opid, |(_, p)| p);
                    let group = self.groups[stage].clone();
                    self.push_collective(
                        CollectiveKind::AllReduceAdd {
                            into: dst,
                            replaces: opid,
                        },
                        src,
                        producer,
                        group,
                        ScheduleRef::Collective(entry),
                    );
                    self.cur[si] = Some(ShardSpec::Whole);
                    self.partial[si] = None;
                    self.spent[si] = true;
                } else {
                    self.require_whole(opid, src, run)?;
                    self.emit(opid, stage, run);
                }
                self.last_write[dst.0 as usize] = Some((stage, opid));
            }
            SplitRule::Pointwise => {
                let mut layout: Option<ShardSpec> = None;
                for &resource in &op.reads {
                    if !matches!(self.kind_of(resource), Kind::Plain | Kind::State) {
                        continue;
                    }
                    if let Some(c) = self.cur[resource.0 as usize] {
                        match layout {
                            None => layout = Some(c),
                            Some(l) if l != c => {
                                return Err(SplitError::LayoutMismatch { op: opid, resource })
                            }
                            _ => {}
                        }
                    }
                }
                let layout = layout.unwrap_or_else(|| out_whole(run));
                for &resource in &op.reads {
                    if self.kind_of(resource) == Kind::State {
                        self.adopt(resource, layout);
                    }
                }
                for &resource in &op.writes {
                    if matches!(self.kind_of(resource), Kind::Plain | Kind::State) {
                        self.write_res(opid, resource, layout, stage)?;
                    }
                }
                self.emit(opid, stage, run);
            }
            SplitRule::RowComplete => {
                for &resource in &op.reads {
                    if matches!(self.kind_of(resource), Kind::Plain | Kind::State) {
                        self.require_whole(opid, resource, run)?;
                    }
                }
                for &resource in &op.writes {
                    if matches!(self.kind_of(resource), Kind::Plain | Kind::State) {
                        self.write_res(opid, resource, out_whole(run), stage)?;
                    }
                }
                self.emit(opid, stage, run);
            }
            SplitRule::HeadLocal { .. } => {
                let head = tp_split(self.tp);
                let mut layout: Option<ShardSpec> = None;
                for &resource in &op.reads {
                    if self.kind_of(resource) != Kind::Plain {
                        continue;
                    }
                    let Some(c) = self.cur[resource.0 as usize] else {
                        continue;
                    };
                    if c != ShardSpec::Whole && c != head {
                        return Err(SplitError::LayoutMismatch { op: opid, resource });
                    }
                    match layout {
                        None => layout = Some(c),
                        Some(l) if l != c => {
                            return Err(SplitError::LayoutMismatch { op: opid, resource })
                        }
                        _ => {}
                    }
                }
                let layout = layout.unwrap_or(ShardSpec::Whole);
                for resource in operands_of(op) {
                    if self.kind_of(resource) != Kind::State {
                        continue;
                    }
                    match self.cur[resource.0 as usize] {
                        None => self.adopt(resource, layout),
                        Some(c) if c != layout => {
                            return Err(SplitError::LayoutMismatch { op: opid, resource })
                        }
                        Some(_) => {}
                    }
                }
                for &resource in &op.writes {
                    if self.kind_of(resource) == Kind::Plain {
                        self.write_res(opid, resource, layout, stage)?;
                    }
                }
                self.emit(opid, stage, run);
            }
            SplitRule::GatedMlp {
                input,
                gate,
                up,
                down,
                output,
                ..
            } => {
                let placed = |weight: ResourceId| {
                    self.weight_of(weight).ok_or(SplitError::UnknownWeight {
                        op: opid,
                        resource: weight,
                    })
                };
                placed(gate)?;
                placed(up)?;
                let down = placed(down)?;
                // The pre-norm (or the gate/up projections) read complete rows.
                self.require_input_whole(opid, input, run)?;
                if self.tp > 1 && matches!(down.policy, ShardPolicy::RowShard { .. }) {
                    let sched = self.sched_entry(down);
                    if let Some(entry) = sched {
                        self.sched_used[entry] = true;
                    }
                    self.write_res(
                        opid,
                        output,
                        ShardSpec::Partial { kind: DimKind::Tp },
                        stage,
                    )?;
                    self.partial[output.0 as usize] = Some(PartialInfo { sched });
                } else {
                    self.write_res(opid, output, out_whole(run), stage)?;
                }
                self.emit(opid, stage, run);
            }
            SplitRule::HeadMixer {
                input,
                out_proj,
                output,
                ..
            } => {
                let out = self.weight_of(out_proj).ok_or(SplitError::UnknownWeight {
                    op: opid,
                    resource: out_proj,
                })?;
                // The pre-norm reads complete rows.
                self.require_input_whole(opid, input, run)?;
                let split = self.tp > 1 && matches!(out.policy, ShardPolicy::RowShard { .. });
                // Each rank keeps the state of exactly its heads.
                let state = if split {
                    tp_split(self.tp)
                } else {
                    ShardSpec::Whole
                };
                for resource in operands_of(op) {
                    if self.kind_of(resource) != Kind::State {
                        continue;
                    }
                    match self.cur[resource.0 as usize] {
                        None => self.adopt(resource, state),
                        Some(c) if c != state => {
                            return Err(SplitError::LayoutMismatch { op: opid, resource })
                        }
                        Some(_) => {}
                    }
                }
                if split {
                    let sched = self.sched_entry(out);
                    if let Some(entry) = sched {
                        self.sched_used[entry] = true;
                    }
                    let partial = ShardSpec::Partial { kind: DimKind::Tp };
                    self.write_res(opid, output, partial, stage)?;
                    self.partial[output.0 as usize] = Some(PartialInfo { sched });
                } else {
                    self.write_res(opid, output, out_whole(run), stage)?;
                }
                self.emit(opid, stage, run);
            }
            SplitRule::Opaque => {
                for resource in operands_of(op) {
                    if self.kind_of(resource) == Kind::Input {
                        continue;
                    }
                    let spec = match self.cur[resource.0 as usize] {
                        Some(spec) => spec,
                        None => {
                            self.adopt(resource, ShardSpec::Whole);
                            ShardSpec::Whole
                        }
                    };
                    let ok = match spec {
                        ShardSpec::Whole => true,
                        ShardSpec::Owner { device } => run == Run::One(device),
                        _ => false,
                    };
                    if !ok {
                        return Err(SplitError::UnsupportedOp {
                            op: opid,
                            kind: op.kind,
                        });
                    }
                }
                for &resource in &op.writes {
                    if self.kind_of(resource) == Kind::Plain {
                        self.write_res(opid, resource, out_whole(run), stage)?;
                    }
                }
                self.emit(opid, stage, run);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<SplitPlan, SplitError> {
        for (index, info) in self.partial.iter().enumerate() {
            if info.is_some() {
                return Err(SplitError::UnreducedPartial {
                    resource: ResourceId(index as u32),
                });
            }
        }
        for (index, &used) in self.sched_used.iter().enumerate() {
            if used && !self.sched_done[index] {
                return Err(SplitError::UnfulfilledSchedule(ScheduleRef::Collective(
                    index,
                )));
            }
        }
        for (index, &crossed) in self.band_crossed.iter().enumerate() {
            if crossed && !self.band_done[index] {
                return Err(SplitError::UnfulfilledSchedule(ScheduleRef::Band(index)));
            }
        }
        let resources = self
            .entry
            .iter()
            .zip(self.cur.iter())
            .map(|(entry, exit)| ResourceShard {
                entry: entry.unwrap_or(ShardSpec::Whole),
                exit: exit.unwrap_or(ShardSpec::Whole),
            })
            .collect();
        Ok(SplitPlan {
            mesh_epoch: self.plan.mesh_epoch,
            ranks: self.ranks,
            collectives: self.collectives,
            resources,
        })
    }
}

fn identity(graph: &StepGraph, plan: &ManifestPlan) -> SplitPlan {
    SplitPlan {
        mesh_epoch: plan.mesh_epoch,
        ranks: vec![RankProgram {
            device: 0,
            items: (0..graph.ops().len())
                .map(|index| RankItem::Op(OpId(index as u32)))
                .collect(),
        }],
        collectives: Vec::new(),
        resources: vec![
            ResourceShard {
                entry: ShardSpec::Whole,
                exit: ShardSpec::Whole,
            };
            graph.resources().len()
        ],
    }
}

/// Split a step graph across `mesh` using the declared policies in `plan`.
///
/// Runs the zero-allocation admission pre-pass first (epoch, `Ep` axis,
/// aliases, geometry), then the single-device identity fast path, then the
/// program-order layout walk.
pub fn split_program(
    graph: &StepGraph,
    plan: &ManifestPlan,
    mesh: &DeviceMesh,
) -> Result<SplitPlan, SplitError> {
    admit(graph, plan, mesh)?;
    if mesh.n_devices() == 1 {
        return Ok(identity(graph, plan));
    }
    Walk::new(graph, plan, mesh).run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_placement::BandPlacement;
    use crate::weight_manifest::{
        plan_manifest, FusedQkvLayout, PinTarget, PlacementHint, StateEntry, WeightEntry,
    };
    use hipfire_dispatch::pipeline::gated_mlp::MlpLadder;
    use hipfire_dispatch::pipeline::step_graph::{Alias, ResourceDecl};
    use rdna_compute::DType;

    const DIM: usize = 64;
    const QDIM: usize = 32;
    const QKV_OUT: usize = 64;
    const FFN: usize = 128;
    const VOCAB: usize = 32;
    const LAYERS: usize = 2;

    struct B {
        g: StepGraph,
        ws: Vec<WeightEntry>,
        st: Vec<StateEntry>,
    }

    impl B {
        fn new() -> Self {
            Self {
                g: StepGraph::new(),
                ws: Vec::new(),
                st: Vec::new(),
            }
        }

        fn res(&mut self, class: ResourceClass, shape: Vec<usize>) -> ResourceId {
            self.g
                .add_resource(ResourceDecl {
                    class,
                    shape,
                    alias: None,
                })
                .unwrap()
        }

        fn act(&mut self, n: usize) -> ResourceId {
            self.res(ResourceClass::Activation, vec![n])
        }

        fn entry(&mut self, e: WeightEntry) -> ResourceId {
            let id = self.res(
                ResourceClass::Weight {
                    name: e.name.clone(),
                    layer: e.layer,
                    group: 1,
                },
                e.logical_shape.clone(),
            );
            self.ws.push(e);
            id
        }

        fn weight(
            &mut self,
            name: &str,
            layer: usize,
            shape: Vec<usize>,
            policy: ShardPolicy,
        ) -> ResourceId {
            self.entry(WeightEntry::layer(name, layer, shape, DType::F16, policy))
        }

        fn ghost(&mut self, name: &str, layer: usize) -> ResourceId {
            self.res(
                ResourceClass::Weight {
                    name: name.to_string(),
                    layer: Some(layer),
                    group: 1,
                },
                vec![8, 4],
            )
        }

        fn op(
            &mut self,
            kind: OpKind,
            layer: Option<usize>,
            rule: SplitRule,
            reads: &[ResourceId],
            writes: &[ResourceId],
        ) -> OpId {
            self.g
                .add_op(OpDecl {
                    kind,
                    layer,
                    rule,
                    reads: reads.to_vec(),
                    writes: writes.to_vec(),
                })
                .unwrap()
        }

        fn project(
            &mut self,
            layer: Option<usize>,
            w: ResourceId,
            input: ResourceId,
            output: ResourceId,
        ) -> OpId {
            self.op(
                OpKind::Project,
                layer,
                SplitRule::Project {
                    weight: w,
                    input,
                    output,
                },
                &[w, input],
                &[output],
            )
        }

        fn accumulate(&mut self, layer: Option<usize>, dst: ResourceId, src: ResourceId) -> OpId {
            self.op(
                OpKind::ResidualAdd,
                layer,
                SplitRule::Accumulate { dst, src },
                &[dst, src],
                &[dst],
            )
        }

        fn row_complete(
            &mut self,
            layer: Option<usize>,
            input: ResourceId,
            output: ResourceId,
        ) -> OpId {
            self.op(
                OpKind::RmsnormAutomatic,
                layer,
                SplitRule::RowComplete,
                &[input],
                &[output],
            )
        }

        fn plan(&self, mesh: &DeviceMesh, n_layers: usize) -> ManifestPlan {
            let placement = BandPlacement::uniform(mesh, n_layers).unwrap();
            self.plan_with(mesh, n_layers, &placement)
        }

        fn plan_with(
            &self,
            mesh: &DeviceMesh,
            n_layers: usize,
            placement: &BandPlacement,
        ) -> ManifestPlan {
            plan_manifest(&self.ws, &self.st, mesh, n_layers, placement).unwrap()
        }
    }

    struct Fix {
        b: B,
        plan: ManifestPlan,
        x: ResourceId,
        logits: ResourceId,
        kv: Vec<ResourceId>,
    }

    fn qwen(mesh: &DeviceMesh) -> Fix {
        let mut b = B::new();
        let token = b.res(ResourceClass::Input, vec![1]);
        let x = b.act(DIM);
        let embed = b.entry(
            WeightEntry::model(
                "embed",
                vec![VOCAB, DIM],
                DType::F16,
                ShardPolicy::Replicate,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        b.op(
            OpKind::Embed,
            None,
            SplitRule::Project {
                weight: embed,
                input: token,
                output: x,
            },
            &[embed, token],
            &[x],
        );
        let mut kv = Vec::new();
        for l in 0..LAYERS {
            let an = b.weight("attn_norm", l, vec![DIM], ShardPolicy::Replicate);
            let qkv_w = b.weight(
                "qkv",
                l,
                vec![QKV_OUT, DIM],
                ShardPolicy::FusedQkv {
                    q_heads: 4,
                    kv_heads: 2,
                    head_dim: 8,
                    layout: FusedQkvLayout::Qkv,
                },
            );
            let o_w = b.weight("o", l, vec![DIM, QDIM], ShardPolicy::RowShard { axis: 1 });
            let fn_w = b.weight("ffn_norm", l, vec![DIM], ShardPolicy::Replicate);
            let gate_w = b.weight(
                "gate",
                l,
                vec![FFN, DIM],
                ShardPolicy::ColumnShard { axis: 0 },
            );
            let up_w = b.weight(
                "up",
                l,
                vec![FFN, DIM],
                ShardPolicy::ColumnShard { axis: 0 },
            );
            let down_w = b.weight("down", l, vec![DIM, FFN], ShardPolicy::RowShard { axis: 1 });
            let state = b.res(
                ResourceClass::State {
                    class: StateClass::Kv,
                    layer: l,
                },
                vec![16],
            );
            b.st.push(StateEntry::new(
                StateKind::Kv {
                    quant: "q8".to_string(),
                },
                l,
            ));
            kv.push(state);

            let xn = b.act(DIM);
            let qkv = b.act(QKV_OUT);
            let attn = b.act(QDIM);
            let o_out = b.act(DIM);
            let xn2 = b.act(DIM);
            let gate = b.act(FFN);
            let up = b.act(FFN);
            let act = b.act(FFN);
            let ffn_out = b.act(DIM);

            b.op(
                OpKind::RmsnormAutomatic,
                Some(l),
                SplitRule::RowComplete,
                &[x, an],
                &[xn],
            );
            b.project(Some(l), qkv_w, xn, qkv);
            b.op(
                OpKind::Attend,
                Some(l),
                SplitRule::HeadLocal {
                    heads: 4,
                    kv_heads: 2,
                },
                &[qkv, state],
                &[attn, state],
            );
            b.project(Some(l), o_w, attn, o_out);
            b.accumulate(Some(l), x, o_out);
            b.op(
                OpKind::RmsnormAutomatic,
                Some(l),
                SplitRule::RowComplete,
                &[x, fn_w],
                &[xn2],
            );
            b.project(Some(l), gate_w, xn2, gate);
            b.project(Some(l), up_w, xn2, up);
            b.op(
                OpKind::SiluMul,
                Some(l),
                SplitRule::Pointwise,
                &[gate, up],
                &[act],
            );
            b.project(Some(l), down_w, act, ffn_out);
            b.accumulate(Some(l), x, ffn_out);
        }
        let on_w = b.entry(WeightEntry::model(
            "output_norm",
            vec![DIM],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Output),
        ));
        let lm_w = b.entry(WeightEntry::model(
            "lm_head",
            vec![VOCAB, DIM],
            DType::F16,
            ShardPolicy::Pin(PinTarget::Output),
        ));
        let xo = b.act(DIM);
        let logits = b.res(ResourceClass::Output, vec![VOCAB]);
        b.op(
            OpKind::RmsnormAutomatic,
            None,
            SplitRule::RowComplete,
            &[x, on_w],
            &[xo],
        );
        b.project(None, lm_w, xo, logits);
        let plan = b.plan(mesh, LAYERS);
        Fix {
            b,
            plan,
            x,
            logits,
            kv,
        }
    }

    fn tp2() -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap()
    }

    #[test]
    fn n1_identity() {
        let mesh = DeviceMesh::single().unwrap();
        let fix = qwen(&mesh);
        let sp = split_program(&fix.b.g, &fix.plan, &mesh).unwrap();
        let n = fix.b.g.ops().len();
        assert_eq!(sp.ranks.len(), 1);
        assert_eq!(sp.ranks[0].device, 0);
        assert_eq!(
            sp.ranks[0].items,
            (0..n)
                .map(|i| RankItem::Op(OpId(i as u32)))
                .collect::<Vec<_>>()
        );
        assert!(sp.collectives.is_empty());
        assert_eq!(sp.resources.len(), fix.b.g.resources().len());
        assert!(sp
            .resources
            .iter()
            .all(|r| r.entry == ShardSpec::Whole && r.exit == ShardSpec::Whole));
    }

    #[test]
    fn epoch_mismatch_is_refused() {
        let mesh = tp2();
        let other = tp2();
        let fix = qwen(&mesh);
        assert_eq!(
            split_program(&fix.b.g, &fix.plan, &other).unwrap_err(),
            SplitError::EpochMismatch {
                plan: mesh.epoch(),
                mesh: other.epoch()
            }
        );
    }

    #[test]
    fn tp2_dense_exactly_once() {
        let mesh = tp2();
        let fix = qwen(&mesh);
        let sp = split_program(&fix.b.g, &fix.plan, &mesh).unwrap();
        assert_eq!(sp.ranks.len(), 2);
        assert_eq!(sp.collectives.len(), 4);
        let mut seen = HashSet::new();
        for (i, c) in sp.collectives.iter().enumerate() {
            assert_eq!(c.id, CollectiveId(i as u32));
            assert!(matches!(c.kind, CollectiveKind::AllReduceAdd { .. }));
            assert_eq!(c.group, vec![0, 1]);
            let ScheduleRef::Collective(e) = c.fulfils else {
                panic!("expected collective schedule ref");
            };
            assert!(seen.insert(e), "schedule entry {e} bound twice");
        }
        assert_eq!(seen.len(), fix.plan.collective_schedule.len());
        assert_eq!(seen.len(), 4);

        let n = fix.b.g.ops().len() as u32;
        let pinned_dev = fix
            .plan
            .weights
            .iter()
            .find(|w| w.name == "lm_head")
            .unwrap()
            .devices[0];
        let other_dev = 1 - pinned_dev;
        let pinned_items = &sp.ranks[pinned_dev].items;
        let other_items = &sp.ranks[other_dev].items;
        let stripped: Vec<RankItem> = pinned_items
            .iter()
            .copied()
            .filter(|item| !matches!(item, RankItem::Op(op) if op.0 >= n - 2))
            .collect();
        assert_eq!(&stripped, other_items);
        assert!(pinned_items.contains(&RankItem::Op(OpId(n - 1))));
        for c in &sp.collectives {
            if let CollectiveKind::AllReduceAdd { replaces, .. } = c.kind {
                assert!(!pinned_items.contains(&RankItem::Op(replaces)));
                assert!(!other_items.contains(&RankItem::Op(replaces)));
            }
        }
        assert_eq!(
            sp.resources[fix.logits.0 as usize].exit,
            ShardSpec::Owner { device: pinned_dev }
        );
        let split = ShardSpec::Split {
            kind: DimKind::Tp,
            parts: 2,
        };
        for kv in &fix.kv {
            let shard = sp.resources[kv.0 as usize];
            assert_eq!(shard.entry, split);
            assert_eq!(shard.exit, split);
        }
    }

    /// `x -> col -> a(Split)`; returns `(x, a)`.
    fn split_prelude(b: &mut B) -> (ResourceId, ResourceId) {
        let x = b.act(4);
        let c = b.weight("c", 0, vec![8, 4], ShardPolicy::ColumnShard { axis: 0 });
        let a = b.act(8);
        b.project(Some(0), c, x, a);
        (x, a)
    }

    fn row_policy() -> ShardPolicy {
        ShardPolicy::RowShard { axis: 1 }
    }

    #[test]
    fn two_row_projections_keep_two_reductions() {
        let mesh = tp2();
        let mut b = B::new();
        let (x, a) = split_prelude(&mut b);
        let r1 = b.weight("r1", 0, vec![4, 8], row_policy());
        let r2 = b.weight("r2", 0, vec![4, 8], row_policy());
        let p1 = b.act(4);
        let p2 = b.act(4);
        b.project(Some(0), r1, a, p1);
        b.accumulate(Some(0), x, p1);
        b.project(Some(0), r2, a, p2);
        b.accumulate(Some(0), x, p2);
        let plan = b.plan(&mesh, 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert_eq!(sp.collectives.len(), 2);
        let mut entries = HashSet::new();
        for c in &sp.collectives {
            assert!(matches!(c.kind, CollectiveKind::AllReduceAdd { .. }));
            let ScheduleRef::Collective(e) = c.fulfils else {
                panic!("expected collective schedule ref");
            };
            entries.insert(e);
        }
        assert_eq!(entries.len(), 2);
        assert_eq!(plan.collective_schedule.len(), 2);
    }

    #[test]
    fn partial_reduced_once_for_two_consumers() {
        let mesh = tp2();
        let mut b = B::new();
        let (_, a) = split_prelude(&mut b);
        let r = b.weight("r", 0, vec![4, 8], row_policy());
        let p = b.act(4);
        let n1 = b.act(4);
        let n2 = b.act(4);
        b.project(Some(0), r, a, p);
        b.row_complete(Some(0), p, n1);
        b.row_complete(Some(0), p, n2);
        let plan = b.plan(&mesh, 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert_eq!(sp.collectives.len(), 1);
        let c = &sp.collectives[0];
        assert_eq!(c.kind, CollectiveKind::AllReduce);
        assert_eq!(c.operand, p);
        assert_eq!(c.producer, OpId(1));
        assert_eq!(c.fulfils, ScheduleRef::Collective(0));
        let expect = vec![
            RankItem::Op(OpId(0)),
            RankItem::Op(OpId(1)),
            RankItem::Collective(CollectiveId(0)),
            RankItem::Op(OpId(2)),
            RankItem::Op(OpId(3)),
        ];
        assert_eq!(sp.ranks[0].items, expect);
        assert_eq!(sp.ranks[1].items, expect);
    }

    #[test]
    fn declared_unused_row_weight_requires_no_schedule() {
        let mesh = tp2();
        let mut b = B::new();
        let _ = split_prelude(&mut b);
        // Declared in the manifest (so it owns a schedule entry) but no Project uses it.
        b.ws.push(WeightEntry::layer(
            "unused",
            0,
            vec![4, 8],
            DType::F16,
            row_policy(),
        ));
        let plan = b.plan(&mesh, 1);
        assert_eq!(plan.collective_schedule.len(), 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert!(sp.collectives.is_empty());
    }

    #[test]
    fn pp2_band_xfer_once() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let fix = qwen(&mesh);
        assert_eq!(
            fix.plan.band_xfers[0].1,
            CollectiveHint::BandXfer { src: 0, dst: 1 }
        );
        let sp = split_program(&fix.b.g, &fix.plan, &mesh).unwrap();
        let bands: Vec<_> = sp
            .collectives
            .iter()
            .filter(|c| matches!(c.kind, CollectiveKind::BandXfer { .. }))
            .collect();
        assert_eq!(bands.len(), 1);
        let c = bands[0];
        assert_eq!(c.kind, CollectiveKind::BandXfer { src: 0, dst: 1 });
        assert_eq!(c.operand, fix.x);
        assert_eq!(c.fulfils, ScheduleRef::Band(0));
        assert_eq!(c.group, vec![0, 1]);
        // Size-one TP groups still reduce: 2 row projections x 2 layers, one device each.
        let reductions: Vec<_> = sp
            .collectives
            .iter()
            .filter(|c| matches!(c.kind, CollectiveKind::AllReduceAdd { .. }))
            .collect();
        assert_eq!(reductions.len(), 4);
        assert_eq!(reductions.iter().filter(|c| c.group == vec![0]).count(), 2);
        assert_eq!(reductions.iter().filter(|c| c.group == vec![1]).count(), 2);
        assert_eq!(sp.collectives.len(), 5);
        assert!(sp.ranks[0].items.contains(&RankItem::Collective(c.id)));
        assert!(sp.ranks[1].items.contains(&RankItem::Collective(c.id)));
        assert!(sp.ranks[0].items.contains(&RankItem::Op(OpId(0))));
        assert!(!sp.ranks[1].items.contains(&RankItem::Op(OpId(0))));
    }

    #[test]
    fn pp2_tp2_composed() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Tp, 2)]).unwrap();
        let fix = qwen(&mesh);
        let sp = split_program(&fix.b.g, &fix.plan, &mesh).unwrap();
        let bands: Vec<_> = sp
            .collectives
            .iter()
            .filter(|c| matches!(c.kind, CollectiveKind::BandXfer { .. }))
            .collect();
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].kind, CollectiveKind::BandXfer { src: 0, dst: 1 });
        assert_eq!(bands[0].operand, fix.x);
        assert_eq!(bands[0].fulfils, ScheduleRef::Band(0));
        assert_eq!(bands[0].group, vec![0, 1, 2, 3]);
        let reductions: Vec<_> = sp
            .collectives
            .iter()
            .filter(|c| matches!(c.kind, CollectiveKind::AllReduceAdd { .. }))
            .collect();
        assert_eq!(reductions.len(), 4);
        assert_eq!(
            reductions.iter().filter(|c| c.group == vec![0, 1]).count(),
            2
        );
        assert_eq!(
            reductions.iter().filter(|c| c.group == vec![2, 3]).count(),
            2
        );
        assert_eq!(sp.collectives.len(), 5);
        let pinned_dev = fix
            .plan
            .weights
            .iter()
            .find(|w| w.name == "lm_head")
            .unwrap()
            .devices[0];
        assert_eq!(pinned_dev, 2);
        assert_eq!(
            sp.resources[fix.logits.0 as usize].exit,
            ShardSpec::Owner { device: pinned_dev }
        );
    }

    #[test]
    fn refuses_opaque_with_sharded_weight() {
        let mesh = tp2();
        let mut b = B::new();
        let x = b.act(4);
        let w = b.weight("gate", 0, vec![8, 4], ShardPolicy::ColumnShard { axis: 0 });
        let y = b.act(4);
        let op = b.op(OpKind::GatedMlp, Some(0), SplitRule::Opaque, &[x, w], &[y]);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnsupportedOp {
                op,
                kind: OpKind::GatedMlp
            }
        );
    }

    #[test]
    fn refuses_row_complete_on_split() {
        let mesh = tp2();
        let mut b = B::new();
        let (_, a) = split_prelude(&mut b);
        let n = b.act(8);
        let op = b.row_complete(Some(0), a, n);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::NeedsGather { op, resource: a }
        );
    }

    #[test]
    fn refuses_unreduced_partial() {
        let mesh = tp2();
        let mut b = B::new();
        let (_, a) = split_prelude(&mut b);
        let r = b.weight("r", 0, vec![4, 8], row_policy());
        let p = b.act(4);
        b.project(Some(0), r, a, p);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnreducedPartial { resource: p }
        );
    }

    #[test]
    fn refuses_unknown_weight() {
        let mesh = tp2();
        let mut b = B::new();
        let x = b.act(4);
        let ghost = b.ghost("ghost", 0);
        let y = b.act(8);
        let op = b.project(Some(0), ghost, x, y);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnknownWeight {
                op,
                resource: ghost
            }
        );
    }

    #[test]
    fn refuses_reused_row_weight() {
        let mesh = tp2();
        let mut b = B::new();
        let (x, a) = split_prelude(&mut b);
        let r = b.weight("r", 0, vec![4, 8], row_policy());
        let p1 = b.act(4);
        let p2 = b.act(4);
        b.project(Some(0), r, a, p1);
        b.accumulate(Some(0), x, p1);
        b.project(Some(0), r, a, p2);
        let second = b.accumulate(Some(0), x, p2);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::ScheduleMismatch { op: second }
        );
    }

    #[test]
    fn refuses_ep_axis() {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let mut b = B::new();
        let x = b.act(4);
        let w = b.weight("w", 0, vec![8, 4], ShardPolicy::Replicate);
        let y = b.act(8);
        b.project(Some(0), w, x, y);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnsupportedAxis(DimKind::Ep)
        );
    }

    #[test]
    fn refuses_unfulfilled_band_schedule() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let mut b = B::new();
        let x0 = b.act(4);
        let a = b.act(4);
        let x1 = b.act(4);
        let c = b.act(4);
        b.op(OpKind::SiluMul, Some(0), SplitRule::Pointwise, &[x0], &[a]);
        b.op(OpKind::SiluMul, Some(1), SplitRule::Pointwise, &[x1], &[c]);
        let plan = b.plan(&mesh, 2);
        assert_eq!(plan.band_xfers.len(), 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnfulfilledSchedule(ScheduleRef::Band(0))
        );
    }

    #[test]
    fn refuses_placement_mismatch() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let mut b = B::new();
        let x = b.act(4);
        // Layer-1 weight lives on stage 1; a layer-0 op runs on stage 0.
        let w = b.weight("w", 1, vec![8, 4], ShardPolicy::Replicate);
        let y = b.act(8);
        let op = b.project(Some(0), w, x, y);
        let plan = b.plan(&mesh, 2);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::PlacementMismatch { op }
        );
    }

    #[test]
    fn layer_stage_follows_custom_band_counts() {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let mut b = B::new();
        let x = b.act(4);
        // Layer-1 weight: stage 0 under uniform [2,2], stage 1 under [1,3].
        let w = b.weight("w", 1, vec![8, 4], ShardPolicy::Replicate);
        let y = b.act(8);
        let op = b.project(Some(0), w, x, y);
        let custom = BandPlacement::from_counts(&mesh, 4, &[1, 3]).unwrap();
        let plan = b.plan_with(&mesh, 4, &custom);
        assert_eq!(
            plan.band_xfers,
            vec![(0, CollectiveHint::BandXfer { src: 0, dst: 1 })]
        );
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::PlacementMismatch { op }
        );
        let uniform = b.plan(&mesh, 4);
        assert_ne!(
            split_program(&b.g, &uniform, &mesh).err(),
            Some(SplitError::PlacementMismatch { op })
        );
    }

    #[test]
    fn refuses_unknown_state() {
        let mesh = tp2();
        let mut b = B::new();
        let q = b.act(8);
        let state = b.res(
            ResourceClass::State {
                class: StateClass::Kv,
                layer: 0,
            },
            vec![16],
        );
        let o = b.act(8);
        b.op(
            OpKind::Attend,
            Some(0),
            SplitRule::HeadLocal {
                heads: 2,
                kv_heads: 2,
            },
            &[q, state],
            &[o, state],
        );
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnknownState { resource: state }
        );
    }

    #[test]
    fn refuses_alias_straddling_split_block() {
        let mesh = tp2();
        let mut b = B::new();
        let w = b.weight("w", 0, vec![8, 4], ShardPolicy::ColumnShard { axis: 0 });
        // 32 elements, 2 ranks -> 16-element blocks; [12, 20) straddles the boundary.
        let view =
            b.g.add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![8],
                alias: Some(Alias { of: w, offset: 12 }),
            })
            .unwrap();
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::Alias { resource: view }
        );

        let mut ok = B::new();
        let w = ok.weight("w", 0, vec![8, 4], ShardPolicy::ColumnShard { axis: 0 });
        ok.g.add_resource(ResourceDecl {
            class: ResourceClass::Activation,
            shape: vec![8],
            alias: Some(Alias { of: w, offset: 16 }),
        })
        .unwrap();
        let plan = ok.plan(&mesh, 1);
        assert!(split_program(&ok.g, &plan, &mesh).is_ok());
    }

    #[test]
    fn n1_geometry_refused_before_identity() {
        let mesh = DeviceMesh::single().unwrap();
        let mut b = B::new();
        let x = b.act(4);
        b.ws.push(WeightEntry::layer(
            "r",
            0,
            vec![4, 257],
            DType::F16,
            row_policy(),
        ));
        let w = b.res(
            ResourceClass::Weight {
                name: "r".to_string(),
                layer: Some(0),
                group: 256,
            },
            vec![4, 257],
        );
        let y = b.act(4);
        b.project(Some(0), w, x, y);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::Geometry {
                resource: w,
                extent: 257,
                parts: 1,
                align: 256
            }
        );
    }

    #[test]
    fn all_reduce_add_source_is_whole_and_spent() {
        let mesh = tp2();
        let mut b = B::new();
        let (x, a) = split_prelude(&mut b);
        let r = b.weight("r", 0, vec![4, 8], row_policy());
        let p = b.act(4);
        let n = b.act(4);
        b.project(Some(0), r, a, p);
        b.accumulate(Some(0), x, p);
        let plan = b.plan(&mesh, 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert_eq!(sp.resources[p.0 as usize].exit, ShardSpec::Whole);

        // Reading the fused-reduction source afterwards is refused.
        let op = b.row_complete(Some(0), p, n);
        let _ = op;
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnreducedPartial { resource: p }
        );
    }

    const MLP_H: usize = 16;
    const MLP_I: usize = 32;

    fn mlp_split() -> [ShardPolicy; 3] {
        [
            ShardPolicy::ColumnShard { axis: 0 },
            ShardPolicy::ColumnShard { axis: 0 },
            row_policy(),
        ]
    }

    /// `out = GatedMlp(input)` in `layer` over weights of `policies`, gate/up
    /// `[inter, MLP_H]` and down `[MLP_H, down_inter]`; returns the op, its
    /// output and the down weight.
    fn gated_mlp(
        b: &mut B,
        layer: usize,
        input: ResourceId,
        policies: [ShardPolicy; 3],
        (inter, down_inter): (usize, usize),
        ladder: MlpLadder,
    ) -> (OpId, ResourceId, ResourceId) {
        let [gp, upp, dp] = policies;
        let gate = b.weight("gate", layer, vec![inter, MLP_H], gp);
        let up = b.weight("up", layer, vec![inter, MLP_H], upp);
        let down = b.weight("down", layer, vec![MLP_H, down_inter], dp);
        let output = b.act(MLP_H);
        let op = b.op(
            OpKind::GatedMlp,
            Some(layer),
            SplitRule::GatedMlp {
                input,
                gate,
                up,
                down,
                output,
                ladder,
            },
            &[input, gate, up, down],
            &[output],
        );
        (op, output, down)
    }

    /// Schedule entry of layer `layer`'s down weight.
    fn down_entry(plan: &ManifestPlan, layer: usize) -> ScheduleRef {
        ScheduleRef::Collective(
            plan.collective_schedule
                .iter()
                .position(|e| e.name == "down" && e.layer == layer)
                .unwrap(),
        )
    }

    const MLP_INTER: (usize, usize) = (MLP_I, MLP_I);

    #[test]
    fn gated_mlp_partial_fuses_into_one_all_reduce_add() {
        for ranks in [2, 4] {
            let mesh = DeviceMesh::rect(&[(DimKind::Tp, ranks)]).unwrap();
            let mut b = B::new();
            let x = b.act(MLP_H);
            let (mlp, p, _) = gated_mlp(&mut b, 0, x, mlp_split(), MLP_INTER, MlpLadder::Trunk);
            let add = b.accumulate(Some(0), x, p);
            let plan = b.plan(&mesh, 1);
            let sp = split_program(&b.g, &plan, &mesh).unwrap();
            assert_eq!(
                sp.collectives,
                vec![PlannedCollective {
                    id: CollectiveId(0),
                    kind: CollectiveKind::AllReduceAdd {
                        into: x,
                        replaces: add
                    },
                    operand: p,
                    producer: mlp,
                    group: (0..ranks).collect(),
                    fulfils: down_entry(&plan, 0),
                }]
            );
            for rank in &sp.ranks {
                assert_eq!(
                    rank.items,
                    [RankItem::Op(mlp), RankItem::Collective(CollectiveId(0))]
                );
            }
            assert_eq!(sp.resources[x.0 as usize].exit, ShardSpec::Whole);
        }
    }

    /// Every ladder's partial is reduced before a whole reader: for the
    /// sandwich ladders that reader is the post-norm residual epilogue.
    #[test]
    fn gated_mlp_partial_reduces_before_a_whole_reader() {
        let mesh = tp2();
        for ladder in [
            MlpLadder::Trunk,
            MlpLadder::SandwichWide,
            MlpLadder::SandwichSteps,
        ] {
            let mut b = B::new();
            let x = b.act(MLP_H);
            let (mlp, p, _) = gated_mlp(&mut b, 0, x, mlp_split(), MLP_INTER, ladder);
            let n = b.act(MLP_H);
            let read = b.row_complete(Some(0), p, n);
            let plan = b.plan(&mesh, 1);
            let sp = split_program(&b.g, &plan, &mesh).unwrap();
            assert_eq!(sp.collectives.len(), 1);
            let c = &sp.collectives[0];
            assert_eq!(
                (c.kind, c.operand, c.producer, c.fulfils),
                (CollectiveKind::AllReduce, p, mlp, down_entry(&plan, 0))
            );
            for rank in &sp.ranks {
                assert_eq!(
                    rank.items,
                    [
                        RankItem::Op(mlp),
                        RankItem::Collective(CollectiveId(0)),
                        RankItem::Op(read)
                    ]
                );
            }
        }
    }

    #[test]
    fn gated_mlp_stays_whole_replicated_or_on_one_tp_part() {
        // Replicated weights on TP2: every rank runs the whole op, any ladder.
        let mesh = tp2();
        let mut b = B::new();
        let x = b.act(MLP_H);
        let replicate = [(); 3].map(|_| ShardPolicy::Replicate);
        let ladder = MlpLadder::SandwichWide;
        let (_, p, _) = gated_mlp(&mut b, 0, x, replicate, MLP_INTER, ladder);
        let n = b.act(MLP_H);
        b.row_complete(Some(0), p, n);
        let plan = b.plan(&mesh, 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert!(sp.collectives.is_empty());
        assert_eq!(sp.resources[p.0 as usize].exit, ShardSpec::Whole);

        // A PP stage is one TP part: split policies run locally, so only the
        // band transfer is planned (no one-device reduction).
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]).unwrap();
        let mut b = B::new();
        let x = b.act(MLP_H);
        for layer in 0..2 {
            let ladder = MlpLadder::Trunk;
            let (_, p, _) = gated_mlp(&mut b, layer, x, mlp_split(), MLP_INTER, ladder);
            b.accumulate(Some(layer), x, p);
        }
        let plan = b.plan(&mesh, 2);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert_eq!(
            sp.collectives.iter().map(|c| c.kind).collect::<Vec<_>>(),
            [CollectiveKind::BandXfer { src: 0, dst: 1 }]
        );
    }

    #[test]
    fn gated_mlp_refusals() {
        let mesh = tp2();
        let trunk = MlpLadder::Trunk;
        // gate and up split, down replicated.
        let mut b = B::new();
        let x = b.act(MLP_H);
        let mut mixed = mlp_split();
        mixed[2] = ShardPolicy::Replicate;
        let (op, _, down) = gated_mlp(&mut b, 0, x, mixed, MLP_INTER, trunk);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnsupportedPolicy { op, resource: down }
        );
        // down splits a different intermediate than gate/up.
        let mut b = B::new();
        let x = b.act(MLP_H);
        let skew = (MLP_I, 2 * MLP_I);
        let (op, _, down) = gated_mlp(&mut b, 0, x, mlp_split(), skew, trunk);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::LayoutMismatch { op, resource: down }
        );
        // The pre-norm needs whole rows: a split input is refused.
        let mut b = B::new();
        let (_, a) = split_prelude(&mut b);
        let (op, _, _) = gated_mlp(&mut b, 0, a, mlp_split(), MLP_INTER, trunk);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::NeedsGather { op, resource: a }
        );
        // A weight the manifest does not plan.
        let mut b = B::new();
        let x = b.act(MLP_H);
        let (op, _, down) = gated_mlp(&mut b, 0, x, mlp_split(), MLP_INTER, trunk);
        b.ws.retain(|entry| entry.name != "down");
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnknownWeight { op, resource: down }
        );
    }

    const MIX_HEADS: usize = 4;
    const MIX_KV: usize = 2;
    const MIX_HD: usize = 8;

    /// `out = HeadMixer(input)` in `layer`: a fused qkv projection with
    /// `qkv_policy` and an output projection with `out_policy`, over
    /// `MIX_HEADS` query / `kv` KV heads; returns the op, its output and the
    /// two weights.
    fn head_mixer(
        b: &mut B,
        layer: usize,
        input: ResourceId,
        qkv_policy: ShardPolicy,
        out_policy: ShardPolicy,
        kv: usize,
    ) -> (OpId, ResourceId, ResourceId, ResourceId) {
        let rows = (MIX_HEADS + 2 * kv) * MIX_HD;
        let qkv = b.weight("qkv", layer, vec![rows, MLP_H], qkv_policy);
        let wo = b.weight("wo", layer, vec![MLP_H, MIX_HEADS * MIX_HD], out_policy);
        let output = b.act(MLP_H);
        let op = b.op(
            OpKind::GatedAttention,
            Some(layer),
            SplitRule::HeadMixer {
                input,
                out_proj: wo,
                output,
                heads: MIX_HEADS,
                kv_heads: kv,
            },
            &[input, qkv, wo],
            &[output],
        );
        (op, output, qkv, wo)
    }

    fn qkv_heads(kv: usize) -> ShardPolicy {
        ShardPolicy::FusedQkv {
            q_heads: MIX_HEADS,
            kv_heads: kv,
            head_dim: MIX_HD,
            layout: crate::weight_manifest::FusedQkvLayout::Qkv,
        }
    }

    #[test]
    fn head_mixer_partial_fuses_into_one_all_reduce_add() {
        for ranks in [2, 4] {
            let mesh = DeviceMesh::rect(&[(DimKind::Tp, ranks)]).unwrap();
            let mut b = B::new();
            let x = b.act(MLP_H);
            let (mix, p, _, wo) =
                head_mixer(&mut b, 0, x, qkv_heads(MIX_HEADS), row_policy(), MIX_HEADS);
            let add = b.accumulate(Some(0), x, p);
            let plan = b.plan(&mesh, 1);
            let sp = split_program(&b.g, &plan, &mesh).unwrap();
            let entry = plan
                .collective_schedule
                .iter()
                .position(|e| e.name == "wo" && e.layer == 0)
                .unwrap();
            assert_eq!(
                sp.collectives,
                vec![PlannedCollective {
                    id: CollectiveId(0),
                    kind: CollectiveKind::AllReduceAdd {
                        into: x,
                        replaces: add
                    },
                    operand: p,
                    producer: mix,
                    group: (0..ranks).collect(),
                    fulfils: ScheduleRef::Collective(entry),
                }]
            );
            assert_eq!(sp.resources[wo.0 as usize].exit, tp_split(ranks));
            assert_eq!(sp.resources[x.0 as usize].exit, ShardSpec::Whole);
        }
    }

    #[test]
    fn head_mixer_stays_whole_replicated() {
        let mesh = tp2();
        let mut b = B::new();
        let x = b.act(MLP_H);
        let rep = ShardPolicy::Replicate;
        let (_, p, _, _) = head_mixer(&mut b, 0, x, rep.clone(), rep, MIX_KV);
        b.accumulate(Some(0), x, p);
        let plan = b.plan(&mesh, 1);
        let sp = split_program(&b.g, &plan, &mesh).unwrap();
        assert!(sp.collectives.is_empty());
        assert_eq!(sp.resources[p.0 as usize].exit, ShardSpec::Whole);
    }

    #[test]
    fn head_mixer_refusals() {
        let mesh = tp2();
        // Heads split on the input side only, the output projection
        // replicated: the split head projection is refused.
        let mut b = B::new();
        let x = b.act(MLP_H);
        let (op, _, qkv, _) = head_mixer(
            &mut b,
            0,
            x,
            qkv_heads(MIX_KV),
            ShardPolicy::Replicate,
            MIX_KV,
        );
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnsupportedPolicy { op, resource: qkv }
        );
        // Output projection split, input projection replicated.
        let mut b = B::new();
        let x = b.act(MLP_H);
        let (op, _, qkv, _) =
            head_mixer(&mut b, 0, x, ShardPolicy::Replicate, row_policy(), MIX_KV);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::UnsupportedPolicy { op, resource: qkv }
        );
        // One KV head cannot split two ways (no KV replication in the plan).
        let mut b = B::new();
        let x = b.act(MLP_H);
        let one_kv = ShardPolicy::HeadSharded {
            n_heads: MIX_HEADS + 2,
            head_dim: MIX_HD,
        };
        let (_, _, _, wo) = head_mixer(&mut b, 0, x, one_kv, row_policy(), 1);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::Geometry {
                resource: wo,
                extent: 1,
                parts: 2,
                align: 1
            }
        );
        // The pre-norm needs whole rows.
        let mut b = B::new();
        let (_, a) = split_prelude(&mut b);
        let (op, _, _, _) = head_mixer(&mut b, 0, a, qkv_heads(MIX_KV), row_policy(), MIX_KV);
        let plan = b.plan(&mesh, 1);
        assert_eq!(
            split_program(&b.g, &plan, &mesh).unwrap_err(),
            SplitError::NeedsGather { op, resource: a }
        );
        // Every weight the op reads must be planned, split or replicated.
        for missing in ["qkv", "wo"] {
            for split in [true, false] {
                let (qkv_policy, out_policy) = if split {
                    (qkv_heads(MIX_KV), row_policy())
                } else {
                    (ShardPolicy::Replicate, ShardPolicy::Replicate)
                };
                let mut b = B::new();
                let x = b.act(MLP_H);
                let (op, _, qkv, wo) = head_mixer(&mut b, 0, x, qkv_policy, out_policy, MIX_KV);
                b.ws.retain(|entry| entry.name != missing);
                let plan = b.plan(&mesh, 1);
                let resource = if missing == "qkv" { qkv } else { wo };
                assert_eq!(
                    split_program(&b.g, &plan, &mesh).unwrap_err(),
                    SplitError::UnknownWeight { op, resource },
                    "{missing} missing, split {split}"
                );
            }
        }
    }
}
