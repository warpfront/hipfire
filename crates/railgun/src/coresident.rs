// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Co-resident groups (design §8.1/§8.3): PM4 nodes that must run
//! *concurrently* — persistent worker pools that wait on each other's event
//! cells — so the planner may not put a queue-draining boundary between them.
//!
//! The ordinary plan derives a hazard edge for every RAW/WAR/WAW between
//! nodes and answers it with a `WaitIdle` boundary: between two pools that
//! deadlocks the window. A group changes the plan in exactly two ways and
//! admits nothing for free:
//!
//! - **Outside edges** are unchanged and checked up front. Before the group's
//!   first member the plan emits the join of *every* member's edges against
//!   the nodes that precede the group, so nothing outside the group needs a
//!   boundary inside it; later members get an empty visibility.
//! - **Inside edges** (member against member) are never answered by a
//!   boundary. Each is classified; it is admitted only when its resource is
//!   event cells (atomics and acquire loads only), or an [`OrderedEdge`]
//!   covers it, and every other one is a refusal that blocks
//!   [`Pm4Plan::group_admission`].
//!
//! Concurrent progress is a separate obligation, discharged by
//! [`Pm4Plan::admit_residency`] from the device's occupancy.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::Serialize;

use crate::kernel::{KernelProgram, MemClass};
use crate::plan::{Edge, Hazard, Pm4Plan};
use crate::{ResourceId, ResourceKind};

/// Nodes `nodes` of a program run concurrently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoResidentGroup {
    pub nodes: Range<usize>,
    pub ordering: GroupOrdering,
}

/// The data edges of a group that a certificate orders through event cells.
///
/// The fields are private and the only constructors are
/// [`GroupOrdering::none`], a `pub(crate)` one and the lab-only
/// `GroupOrdering::lab_pending`: an ordering must come from railgun's
/// task-graph certificate (`crate::task_graph`, which checks event
/// generation, producer cardinality and publication dominance against the
/// emitted objects), never from host code asserting that two pools are
/// ordered. The lab one marks the ordering uncertified
/// ([`GroupOrdering::is_pending`]); the plan reports it in
/// [`Pm4Plan::admission`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupOrdering {
    edges: Vec<OrderedEdge>,
    /// The edges are not from a certificate (`lab`: differential-verified,
    /// certificate pending).
    pending: bool,
}

impl GroupOrdering {
    /// No data edge is ordered: the group's members may share only event
    /// cells and read-only data.
    pub fn none() -> Self {
        Self {
            edges: Vec::new(),
            pending: false,
        }
    }

    /// The ordering a task-graph certificate produced.
    #[allow(dead_code)] // the producer, `crate::task_graph`, does not exist yet; tests build one
    pub(crate) fn from_certificate(edges: Vec<OrderedEdge>) -> Self {
        Self {
            edges,
            pending: false,
        }
    }

    /// Lab only: an ordering whose certificate is pending. Its edges get
    /// exactly the validation a certificate's do
    /// ([`KernelProgram::add_coresident_group`]); the group is reported
    /// [`crate::admission::Admission::LabPending`].
    #[cfg(feature = "lab")]
    pub fn lab_pending(edges: Vec<OrderedEdge>) -> Self {
        Self {
            edges,
            pending: true,
        }
    }

    pub fn edges(&self) -> &[OrderedEdge] {
        &self.edges
    }

    /// The ordering is awaiting its certificate (lab builds only).
    pub fn is_pending(&self) -> bool {
        self.pending
    }
}

/// `producer`'s writes to `resource` happen-before `consumer`'s accesses,
/// published by a release on event cell `event` (an `AtomicRelease` effect of
/// `producer`) and acquired by `consumer` (a `LoadAcquire` effect).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct OrderedEdge {
    pub producer: usize,
    pub consumer: usize,
    pub resource: ResourceId,
    pub event: ResourceId,
}

/// Why a [`Pm4Plan`] admitted a member-member edge.
pub const WHY_EVENT_CELLS: &str = "event cells: atomics and acquire loads only";
pub const WHY_ORDERED_EDGE: &str = "ordered edge: release on its event, acquired by the consumer";

/// What the plan decided about one [`CoResidentGroup`].
#[derive(Clone, Debug, Serialize)]
pub struct GroupPlan {
    pub nodes: Range<usize>,
    /// Member-member edges the plan admitted without a boundary, with why.
    /// (Read-read pairs have no hazard and no edge.)
    pub admitted: Vec<(Edge, &'static str)>,
    /// Member-member conflicts nothing orders; any entry blocks
    /// [`Pm4Plan::group_admission`].
    pub refusals: Vec<String>,
    /// Why the group is not certified: its `LabPending` members and an
    /// uncertified ordering (empty in a product build).
    pub pending: Vec<String>,
    /// The progress proof ([`Pm4Plan::admit_residency`]); `None` until given.
    pub residency: Option<ResidencyProof>,
}

/// Evidence that every member of a group can be resident at once.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResidencyProof {
    pub compute_units: u32,
    /// Per member: workgroups of its emitted descriptor that fit one CU.
    pub blocks_per_cu: Vec<u32>,
    /// Per member: its grid (workgroups per axis).
    pub grids: Vec<[u32; 3]>,
    /// The admission rule that held: `"A"` or `"B"` (see
    /// [`Pm4Plan::admit_residency`]).
    pub rule: &'static str,
}

fn member_classes(program: &KernelProgram, node: usize) -> BTreeMap<ResourceId, Vec<MemClass>> {
    let mut out: BTreeMap<ResourceId, Vec<MemClass>> = BTreeMap::new();
    for e in &program.nodes[node].effects {
        let classes = out.entry(e.resource).or_default();
        for c in &e.classes {
            if !classes.contains(c) {
                classes.push(*c);
            }
        }
    }
    out
}

/// An [`OrderedEdge`] whose publication is really there: `event` is declared
/// event cells, `producer` releases on it, `consumer` acquires from it, and
/// `producer` writes `resource`. `group` is the group's node range.
pub(crate) fn check_ordered_edge(
    program: &KernelProgram,
    group: &Range<usize>,
    e: &OrderedEdge,
) -> Result<(), String> {
    if !group.contains(&e.producer) || !group.contains(&e.consumer) || e.producer == e.consumer {
        return Err(format!(
            "ordered edge {}->{}: endpoints must be distinct members of group {group:?}",
            e.producer, e.consumer
        ));
    }
    if program.resources.get(e.resource.0 as usize).is_none() {
        return Err(format!(
            "ordered edge {}->{}: unknown resource {}",
            e.producer, e.consumer, e.resource.0
        ));
    }
    if !matches!(
        program.kinds.get(&e.event),
        Some(ResourceKind::EventCells { .. })
    ) {
        return Err(format!(
            "ordered edge {}->{}: event resource {} is not declared EventCells",
            e.producer, e.consumer, e.event.0
        ));
    }
    let has = |node: usize, resource: ResourceId, class: MemClass| {
        program.nodes[node]
            .effects
            .iter()
            .any(|x| x.resource == resource && x.has(class))
    };
    if !has(e.producer, e.event, MemClass::AtomicRelease) {
        return Err(format!(
            "ordered edge {}->{}: missing release: node {} has no atomic_release on event {}",
            e.producer, e.consumer, e.producer, e.event.0
        ));
    }
    if !has(e.consumer, e.event, MemClass::LoadAcquire) {
        return Err(format!(
            "ordered edge {}->{}: missing acquire: node {} has no load_acquire on event {}",
            e.producer, e.consumer, e.consumer, e.event.0
        ));
    }
    if !program.nodes[e.producer]
        .effects
        .iter()
        .any(|x| x.resource == e.resource && x.writes())
    {
        return Err(format!(
            "ordered edge {}->{}: producer {} does not write resource {}",
            e.producer, e.consumer, e.producer, e.resource.0
        ));
    }
    Ok(())
}

/// Classify every member-member conflict of `group`: the edges the plan may
/// admit without a boundary, and the refusals.
pub(crate) fn classify(
    program: &KernelProgram,
    group: &CoResidentGroup,
) -> (Vec<(Edge, &'static str)>, Vec<String>) {
    let mut admitted = Vec::new();
    let mut refusals = Vec::new();
    let members: Vec<usize> = group.nodes.clone().collect();
    for &m in &members {
        if !program.nodes[m].lowering.is_pm4_lowerable() {
            refusals.push(format!(
                "member {m} ({}) is not PM4",
                program.nodes[m].symbol
            ));
        }
    }
    let classes: Vec<BTreeMap<ResourceId, Vec<MemClass>>> =
        members.iter().map(|&m| member_classes(program, m)).collect();
    // Event cells admit only event classes, from every member.
    let mut resources: Vec<ResourceId> = classes.iter().flat_map(|c| c.keys().copied()).collect();
    resources.sort_unstable();
    resources.dedup();
    let published: Vec<&OrderedEdge> = group
        .ordering
        .edges()
        .iter()
        .filter(|e| check_ordered_edge(program, &group.nodes, e).is_ok())
        .collect();
    for &resource in &resources {
        let event_cells = matches!(
            program.kinds.get(&resource),
            Some(ResourceKind::EventCells { .. })
        );
        let event_only = classes
            .iter()
            .filter_map(|c| c.get(&resource))
            .all(|cs| cs.iter().all(|c| c.is_event_class()));
        if event_cells && !event_only {
            refusals.push(format!(
                "resource {}: plain store into event cells by a member",
                resource.0
            ));
            continue;
        }
        for (ai, &a) in members.iter().enumerate() {
            let Some(ca) = classes[ai].get(&resource) else {
                continue;
            };
            for (bi, &b) in members.iter().enumerate().skip(ai + 1) {
                let Some(cb) = classes[bi].get(&resource) else {
                    continue;
                };
                let (a_reads, a_writes) = (
                    ca.iter().any(|c| c.reads()),
                    ca.iter().any(|c| c.writes()),
                );
                let (b_reads, b_writes) = (
                    cb.iter().any(|c| c.reads()),
                    cb.iter().any(|c| c.writes()),
                );
                if !(a_writes || b_writes) {
                    continue; // both sides only read: no hazard
                }
                let mut push = |producer, consumer, hazard, why| {
                    admitted.push((
                        Edge {
                            producer,
                            consumer,
                            hazard,
                            resource: resource.0,
                        },
                        why,
                    ));
                };
                if event_cells {
                    if a_writes && b_reads {
                        push(a, b, Hazard::Raw, WHY_EVENT_CELLS);
                    }
                    if a_reads && b_writes {
                        push(a, b, Hazard::War, WHY_EVENT_CELLS);
                    }
                    if a_writes && b_writes {
                        push(a, b, Hazard::Waw, WHY_EVENT_CELLS);
                    }
                    continue;
                }
                // The edge's direction is the producer's: it writes `resource`
                // (checked), and the consumer's accesses follow.
                let ordered = published.iter().find(|e| {
                    e.resource == resource
                        && ((e.producer == a && e.consumer == b)
                            || (e.producer == b && e.consumer == a))
                });
                match ordered {
                    Some(e) => {
                        let consumer_reads = if e.consumer == a { a_reads } else { b_reads };
                        let hazard = if consumer_reads {
                            Hazard::Raw
                        } else {
                            Hazard::Waw
                        };
                        push(e.producer, e.consumer, hazard, WHY_ORDERED_EDGE);
                    }
                    None => {
                        let mut hazards = Vec::new();
                        if a_writes && b_reads {
                            hazards.push("RAW");
                        }
                        if a_reads && b_writes {
                            hazards.push("WAR");
                        }
                        if a_writes && b_writes {
                            hazards.push("WAW");
                        }
                        refusals.push(format!(
                            "members {a} and {b} conflict on resource {} ({}) with no event-cell publication or ordered edge",
                            resource.0,
                            hazards.join("+")
                        ));
                    }
                }
            }
        }
    }
    admitted.sort_by_key(|(e, _)| (e.producer, e.consumer, e.resource, e.hazard));
    admitted.dedup();
    (admitted, refusals)
}

impl KernelProgram {
    /// Declare nodes `nodes` a co-resident group: they run concurrently, so
    /// [`crate::plan::plan`] puts no boundary between them (see the module
    /// docs). Refused (the program is unchanged) unless there are at least
    /// two members, in bounds, every one PM4, disjoint from every other group
    /// and from every conditional body, and every [`OrderedEdge`] of
    /// `ordering` is real: distinct members, its `event` declared
    /// `EventCells`, the producer holds an `AtomicRelease` on the event
    /// ("missing release" otherwise), the consumer a `LoadAcquire`
    /// ("missing acquire"), and the producer writes the resource.
    pub fn add_coresident_group(
        &mut self,
        nodes: Range<usize>,
        ordering: GroupOrdering,
    ) -> Result<(), String> {
        if nodes.len() < 2 {
            return Err(format!(
                "co-resident group {nodes:?} needs at least two nodes"
            ));
        }
        if nodes.end > self.nodes.len() {
            return Err(format!(
                "co-resident group {nodes:?} is past {} nodes",
                self.nodes.len()
            ));
        }
        for j in nodes.clone() {
            if !self.nodes[j].lowering.is_pm4_lowerable() {
                return Err(format!(
                    "node {j} ({}) in a co-resident group is not PM4",
                    self.nodes[j].symbol
                ));
            }
        }
        if let Some(other) = self
            .groups
            .iter()
            .find(|g| g.nodes.start < nodes.end && nodes.start < g.nodes.end)
        {
            return Err(format!(
                "co-resident group {nodes:?} overlaps group {:?}",
                other.nodes
            ));
        }
        if let Some(other) = self
            .conditionals
            .iter()
            .find(|c| c.body.start < nodes.end && nodes.start < c.body.end)
        {
            return Err(format!(
                "co-resident group {nodes:?} overlaps conditional body {:?}",
                other.body
            ));
        }
        for e in ordering.edges() {
            check_ordered_edge(self, &nodes, e)?;
        }
        let at = self.groups.partition_point(|g| g.nodes.start < nodes.start);
        self.groups.insert(at, CoResidentGroup { nodes, ordering });
        Ok(())
    }
}

impl Pm4Plan {
    /// Admit the residency of group `group` (an index into
    /// [`Pm4Plan::groups`]): all its members can be resident at once, so a
    /// member waiting on another's event cell never waits for a slot.
    ///
    /// `compute_units` is the device's CU count and `blocks_per_cu[m]` the
    /// number of workgroups of member `m` one CU holds, from the device's
    /// occupancy calculator over the *loaded object's emitted descriptor*
    /// (registers, LDS, workgroup size). With `W_m` the product of member
    /// `m`'s grid dimensions and `B_m = blocks_per_cu[m]` (zero is refused),
    /// the group is admitted iff, in exact integer arithmetic,
    ///
    /// - **(A)** `Σ W_m ≤ CU · min_m B_m`, or
    /// - **(B)** `Σ_m W_m / B_m ≤ CU · (1 − 1 / min_m B_m)`.
    ///
    /// Soundness. A workgroup of member `m` uses at most `1/B_m` of every
    /// per-CU resource (registers, LDS) and of the workgroup-slot limit, and
    /// `B_m ≥ min B`, so a CU accepts a member workgroup whenever the
    /// workgroups resident on it use at most `1 − 1/B_m` (hence at most
    /// `1 − 1/min B`) of each resource. Suppose a workgroup `w` of member `m'`
    /// is waiting while every CU refuses it. Under (A): a CU holding fewer
    /// than `min B` workgroups uses less than `(min B − 1)/min B` of each
    /// resource and would accept `w`, so every CU holds at least `min B`
    /// workgroups and `Σ W − 1 ≥ CU · min B`, contradicting (A). Under (B):
    /// a refusing CU's resident workgroups use more than `1 − 1/B_m'` ≥
    /// `1 − 1/min B` of some resource, hence their weights `1/B_m` sum above
    /// that; summed over the CUs, the other workgroups' weights exceed
    /// `CU · (1 − 1/min B)`, but they total at most `Σ W_m/B_m − 1/B_m'`,
    /// contradicting (B). So every workgroup of every member is placed
    /// without any member having to finish.
    ///
    /// Obligation (not checked here): the queue and device run nothing else
    /// while the group does (an exclusive device), and `blocks_per_cu` is the
    /// calculator's answer for the object the group will actually launch.
    pub fn admit_residency(
        &mut self,
        program: &KernelProgram,
        group: usize,
        compute_units: u32,
        blocks_per_cu: &[u32],
    ) -> Result<(), String> {
        let plan = self
            .groups
            .get(group)
            .ok_or_else(|| format!("no co-resident group {group}"))?;
        let nodes = plan.nodes.clone();
        if nodes.end > program.nodes.len() {
            return Err(format!(
                "co-resident group {group} {nodes:?} is past the program's {} nodes",
                program.nodes.len()
            ));
        }
        if blocks_per_cu.len() != nodes.len() {
            return Err(format!(
                "co-resident group {group} has {} members, {} blocks-per-CU values given",
                nodes.len(),
                blocks_per_cu.len()
            ));
        }
        if compute_units == 0 {
            return Err("co-resident group: zero compute units".to_owned());
        }
        if let Some(m) = blocks_per_cu.iter().position(|&b| b == 0) {
            return Err(format!(
                "co-resident group exceeds occupancy: member {} fits no workgroup on a CU",
                nodes.start + m
            ));
        }
        let grids: Vec<[u32; 3]> = nodes.clone().map(|j| program.nodes[j].grid).collect();
        let overflow = || "co-resident group: occupancy arithmetic overflows".to_owned();
        let widths: Vec<u128> = grids
            .iter()
            .map(|g| g.iter().map(|&d| u128::from(d)).product())
            .collect();
        let cu = u128::from(compute_units);
        let min_b = u128::from(*blocks_per_cu.iter().min().expect("non-empty group"));
        let sum_w = widths
            .iter()
            .try_fold(0u128, |acc, &w| acc.checked_add(w))
            .ok_or_else(overflow)?;
        let rule = if sum_w <= cu * min_b {
            "A"
        } else {
            // Σ W_m / B_m ≤ CU (min B − 1) / min B over the common
            // denominator D = lcm(B_m): Σ W_m (D / B_m) · min B ≤ CU (min B − 1) D.
            let lcm = blocks_per_cu.iter().try_fold(1u128, |acc, &b| {
                let b = u128::from(b);
                acc.checked_mul(b / gcd(acc, b))
            });
            let lcm = lcm.ok_or_else(overflow)?;
            let demand = widths
                .iter()
                .zip(blocks_per_cu)
                .try_fold(0u128, |acc, (&w, &b)| {
                    w.checked_mul(lcm / u128::from(b))
                        .and_then(|x| acc.checked_add(x))
                })
                .ok_or_else(overflow)?;
            let lhs = demand.checked_mul(min_b).ok_or_else(overflow)?;
            let rhs = cu
                .checked_mul(min_b - 1)
                .and_then(|x| x.checked_mul(lcm))
                .ok_or_else(overflow)?;
            if lhs <= rhs {
                "B"
            } else {
                return Err(format!(
                    "co-resident group exceeds occupancy: Σ W = {sum_w} workgroups over {nodes:?} \
                     (blocks per CU {blocks_per_cu:?}) fit neither rule A (≤ {} = {compute_units} CUs × min {min_b}) \
                     nor rule B (Σ W/B ≤ {compute_units} × (1 − 1/{min_b}))",
                    cu * min_b
                ));
            }
        };
        self.groups[group].residency = Some(ResidencyProof {
            compute_units,
            blocks_per_cu: blocks_per_cu.to_vec(),
            grids,
            rule,
        });
        Ok(())
    }

    /// Every co-resident group is admitted: no refused member-member
    /// conflict and a residency proof. The lowering calls this before it
    /// emits a plan with groups.
    pub fn group_admission(&self) -> Result<(), String> {
        for (i, g) in self.groups.iter().enumerate() {
            if !g.refusals.is_empty() {
                return Err(format!(
                    "co-resident group {i} {:?} refused: {}",
                    g.nodes,
                    g.refusals.join("; ")
                ));
            }
            if g.residency.is_none() {
                return Err(format!(
                    "co-resident group {i} {:?} has no residency proof",
                    g.nodes
                ));
            }
        }
        Ok(())
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::tests::{launch, ptr};
    use crate::kernel::{author, Lowering, MemClass::*};
    use crate::plan::{plan, CacheTable, Rung, RowPolicy, Visibility};

    const X: u64 = 0x7000_0000;
    const E: u64 = 0x7100_0000;
    const D1: u64 = 0x7200_0000;
    const D2: u64 = 0x7300_0000;

    fn plan_of(p: &KernelProgram) -> Pm4Plan {
        plan(p, &CacheTable::for_arch("gfx1201").unwrap(), RowPolicy::TierD)
    }

    /// n0 stores X; n1 (member) releases on E, stores D1; n2 (member)
    /// acquires E, loads X. Resources: X = 0, E = 1, D1 = 2.
    fn event_program() -> KernelProgram {
        let launches = vec![
            launch(&[(0, X)], 8, vec![ptr(0, &[VmemStore])]),
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
            ),
            launch(
                &[(0, E), (8, X)],
                16,
                vec![ptr(0, &[LoadAcquire]), ptr(8, &[VmemLoad])],
            ),
        ];
        let mut p = author("gfx1201", &launches);
        p.declare_event_cells(ResourceId(1), 8).unwrap();
        p
    }

    #[test]
    fn members_sharing_event_cells_get_no_boundary_and_the_first_carries_pre_group_edges() {
        let mut p = event_program();
        // Without a group the pre-group RAW (n0 -> n2) lands before n2 and the
        // event-cell RAW (n1 -> n2) also forces a wait.
        let loose = plan_of(&p);
        assert!(loose.boundaries[2].visibility.wait_idle);

        p.add_coresident_group(1..3, GroupOrdering::none()).unwrap();
        let g = plan_of(&p);
        // The first member's boundary is the join of every member's pre-group
        // edges: n2 reads X that n0 wrote.
        let first = &g.boundaries[1];
        assert!(first.visibility.wait_idle);
        assert_eq!(first.visibility.acquire, Rung::InterNode);
        assert_eq!(first.edges.len(), 1);
        assert_eq!(
            (first.edges[0].producer, first.edges[0].consumer),
            (0, 2),
            "an edge of the second member sits on the first member's boundary"
        );
        assert_eq!(first.edges[0].hazard, Hazard::Raw);
        // The later member has an empty visibility and no edges.
        assert_eq!(g.boundaries[2].visibility, Visibility::default());
        assert!(g.boundaries[2].edges.is_empty());
        // One segment, and no Boundary op between the members.
        assert!(g.is_single_segment(3));
        let ops = g.segment_ops(&g.segments[0]);
        let dispatches: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(o, crate::plan::Pm4Op::Dispatch(n) if *n > 0))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(dispatches[1], dispatches[0] + 1);
        // The event cells conflict is admitted, with a reason.
        assert_eq!(g.groups.len(), 1);
        assert!(g.groups[0].refusals.is_empty(), "{:?}", g.groups[0].refusals);
        assert_eq!(g.groups[0].admitted.len(), 1);
        assert_eq!(g.groups[0].admitted[0].0.resource, 1);
        assert_eq!(g.groups[0].admitted[0].1, WHY_EVENT_CELLS);
    }

    #[test]
    fn a_group_at_the_start_of_a_segment_still_gets_empty_member_boundaries() {
        let launches = vec![
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
            ),
            launch(
                &[(0, E), (8, D2)],
                16,
                vec![ptr(0, &[LoadAcquire]), ptr(8, &[VmemStore])],
            ),
        ];
        let mut p = author("gfx1201", &launches);
        p.declare_event_cells(ResourceId(0), 4).unwrap();
        p.add_coresident_group(0..2, GroupOrdering::none()).unwrap();
        let g = plan_of(&p);
        assert_eq!(g.boundaries[1].visibility, Visibility::default());
        assert!(g.groups[0].refusals.is_empty());
    }

    #[test]
    fn later_nodes_see_every_member_as_an_open_producer() {
        let mut p = event_program();
        // n3 stores X: a WAR/WAW against member n2 (reads X) and n0.
        let extra = launch(&[(0, X)], 8, vec![ptr(0, &[VmemStore])]);
        let mut launches: Vec<_> = Vec::new();
        launches.push(extra);
        let tail = author("gfx1201", &launches);
        p.nodes.push(tail.nodes[0].clone());
        // The tail's pointer resolves to the same X allocation as resource 0.
        p.add_coresident_group(1..3, GroupOrdering::none()).unwrap();
        let g = plan_of(&p);
        assert!(g.boundaries[3].visibility.wait_idle);
        assert!(g.boundaries[3]
            .edges
            .iter()
            .any(|e| e.producer == 2 && e.hazard == Hazard::War));
    }

    #[test]
    fn members_sharing_a_written_data_resource_without_ordering_are_refused() {
        let launches = vec![
            launch(&[(0, D1)], 8, vec![ptr(0, &[VmemStore])]),
            launch(&[(0, D1)], 8, vec![ptr(0, &[VmemStore])]),
        ];
        let mut p = author("gfx1201", &launches);
        p.add_coresident_group(0..2, GroupOrdering::none()).unwrap();
        let g = plan_of(&p);
        assert_eq!(g.boundaries[1].visibility, Visibility::default());
        assert_eq!(g.groups[0].refusals.len(), 1);
        assert!(
            g.groups[0].refusals[0].contains("resource 0"),
            "{:?}",
            g.groups[0].refusals
        );
        let err = g.group_admission().unwrap_err();
        assert!(err.contains("refused") && err.contains("resource 0"), "{err}");
    }

    #[test]
    fn read_only_sharing_needs_no_ordering() {
        let launches = vec![
            launch(&[(0, D1)], 8, vec![ptr(0, &[VmemLoad])]),
            launch(&[(0, D1)], 8, vec![ptr(0, &[SmemLoad])]),
        ];
        let mut p = author("gfx1201", &launches);
        p.add_coresident_group(0..2, GroupOrdering::none()).unwrap();
        let g = plan_of(&p);
        assert!(g.groups[0].refusals.is_empty());
        assert!(g.groups[0].admitted.is_empty());
    }

    /// n0 releases on E and stores D1; n1 acquires E and loads D1.
    /// Resources: E = 0, D1 = 1.
    fn ordered_program() -> KernelProgram {
        let launches = vec![
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
            ),
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[LoadAcquire]), ptr(8, &[VmemLoad])],
            ),
        ];
        let mut p = author("gfx1201", &launches);
        p.declare_event_cells(ResourceId(0), 4).unwrap();
        p
    }

    fn edge(producer: usize, consumer: usize) -> OrderedEdge {
        OrderedEdge {
            producer,
            consumer,
            resource: ResourceId(1),
            event: ResourceId(0),
        }
    }

    #[test]
    fn an_ordered_edge_admits_the_data_conflict() {
        let mut p = ordered_program();
        p.add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(0, 1)]))
            .unwrap();
        let g = plan_of(&p);
        assert!(g.groups[0].refusals.is_empty(), "{:?}", g.groups[0].refusals);
        let data: Vec<_> = g.groups[0]
            .admitted
            .iter()
            .filter(|(e, _)| e.resource == 1)
            .collect();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].0.hazard, Hazard::Raw);
        assert_eq!(data[0].1, WHY_ORDERED_EDGE);
    }

    #[test]
    fn an_ordered_edge_needs_its_release_acquire_and_a_writer() {
        let mut p = ordered_program();
        let before = p.clone();
        // Producer is the reader: no release on the event (and it never writes D1).
        let err = p
            .add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(1, 0)]))
            .unwrap_err();
        assert!(err.contains("missing release"), "{err}");
        assert_eq!(p, before, "an Err leaves the program unchanged");

        // A producer with a release but a consumer that never acquires.
        let launches = vec![
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
            ),
            launch(
                &[(0, E), (8, D1)],
                16,
                vec![ptr(0, &[VmemLoad]), ptr(8, &[VmemLoad])],
            ),
        ];
        let mut q = author("gfx1201", &launches);
        q.declare_event_cells(ResourceId(0), 4).unwrap();
        let err = q
            .add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(0, 1)]))
            .unwrap_err();
        assert!(err.contains("missing acquire"), "{err}");

        // An event that is not event cells.
        let mut undeclared = author("gfx1201", &launches);
        let err = undeclared
            .add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(0, 1)]))
            .unwrap_err();
        assert!(err.contains("not declared EventCells"), "{err}");

        // Endpoints outside the group / equal.
        let err = p
            .add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(0, 0)]))
            .unwrap_err();
        assert!(err.contains("distinct members"), "{err}");
    }

    #[test]
    fn group_shape_is_validated() {
        let mut p = ordered_program();
        assert!(p.add_coresident_group(0..1, GroupOrdering::none()).is_err());
        assert!(p.add_coresident_group(0..5, GroupOrdering::none()).is_err());
        p.nodes[1].lowering = Lowering::HipDirect {
            reason: "x".into(),
        };
        assert!(p.add_coresident_group(0..2, GroupOrdering::none()).is_err());
        p.nodes[1].lowering = Lowering::Pm4;
        p.add_coresident_group(0..2, GroupOrdering::none()).unwrap();
        assert!(
            p.add_coresident_group(0..2, GroupOrdering::none()).is_err(),
            "disjoint from other groups"
        );
        assert_eq!(p.groups.len(), 1);

        let mut q = ordered_program();
        q.conditionals.push(crate::kernel::Conditional {
            flag: ResourceId(0),
            body: 1..2,
            lowering: crate::kernel::CondLowering::KernelGuard,
        });
        assert!(
            q.add_coresident_group(0..2, GroupOrdering::none()).is_err(),
            "disjoint from conditional bodies"
        );
    }

    fn residency_plan(grids: [[u32; 3]; 2]) -> (KernelProgram, Pm4Plan) {
        let mut p = ordered_program();
        p.nodes[0].grid = grids[0];
        p.nodes[1].grid = grids[1];
        p.add_coresident_group(0..2, GroupOrdering::from_certificate(vec![edge(0, 1)]))
            .unwrap();
        let plan = plan_of(&p);
        (p, plan)
    }

    #[test]
    fn residency_is_admitted_under_rule_a() {
        // CU = 10, B = [4, 8]: 4 + 4 workgroups <= 10 * 4.
        let (p, mut plan) = residency_plan([[4, 1, 1], [2, 2, 1]]);
        plan.admit_residency(&p, 0, 10, &[4, 8]).unwrap();
        let proof = plan.groups[0].residency.as_ref().unwrap();
        assert_eq!(proof.rule, "A");
        assert_eq!(proof.grids, vec![[4, 1, 1], [2, 2, 1]]);
        assert_eq!(proof.blocks_per_cu, vec![4, 8]);
        plan.group_admission().unwrap();
    }

    #[test]
    fn residency_is_admitted_under_rule_b_but_not_a() {
        // CU = 10, B = [4, 8]: A needs sum <= 40, here 6 + 48 = 54; B needs
        // 6/4 + 48/8 = 7.5 <= 10 * (1 - 1/4) = 7.5.
        let (p, mut plan) = residency_plan([[6, 1, 1], [48, 1, 1]]);
        plan.admit_residency(&p, 0, 10, &[4, 8]).unwrap();
        assert_eq!(plan.groups[0].residency.as_ref().unwrap().rule, "B");
    }

    #[test]
    fn residency_exceeding_occupancy_is_refused() {
        // 10/4 + 48/8 = 8.5 > 7.5 and 58 > 40.
        let (p, mut plan) = residency_plan([[10, 1, 1], [48, 1, 1]]);
        let err = plan.admit_residency(&p, 0, 10, &[4, 8]).unwrap_err();
        assert!(
            err.contains("co-resident group exceeds occupancy"),
            "{err}"
        );
        assert!(plan.groups[0].residency.is_none());
        // One more workgroup over the rule-B boundary.
        let (p, mut plan) = residency_plan([[7, 1, 1], [48, 1, 1]]);
        assert!(plan.admit_residency(&p, 0, 10, &[4, 8]).is_err());
    }

    #[test]
    fn a_member_that_fits_no_workgroup_is_refused() {
        let (p, mut plan) = residency_plan([[1, 1, 1], [1, 1, 1]]);
        let err = plan.admit_residency(&p, 0, 10, &[4, 0]).unwrap_err();
        assert!(err.contains("exceeds occupancy"), "{err}");
        assert!(plan.admit_residency(&p, 0, 0, &[4, 8]).is_err());
        assert!(plan.admit_residency(&p, 0, 10, &[4]).is_err());
        assert!(plan.admit_residency(&p, 3, 10, &[4, 8]).is_err());
        // min B = 1: rule B can never hold (nothing is spare); rule A can.
        let (p, mut plan) = residency_plan([[5, 1, 1], [5, 1, 1]]);
        plan.admit_residency(&p, 0, 10, &[1, 1]).unwrap();
        assert_eq!(plan.groups[0].residency.as_ref().unwrap().rule, "A");
        let (p, mut plan) = residency_plan([[6, 1, 1], [5, 1, 1]]);
        assert!(plan.admit_residency(&p, 0, 10, &[1, 1]).is_err());
    }

    #[test]
    fn group_admission_needs_a_residency_proof() {
        let (p, mut plan) = residency_plan([[4, 1, 1], [4, 1, 1]]);
        let err = plan.group_admission().unwrap_err();
        assert!(err.contains("no residency proof"), "{err}");
        plan.admit_residency(&p, 0, 10, &[4, 8]).unwrap();
        plan.group_admission().unwrap();
        // A plan with no group is trivially admitted.
        let plain = plan_of(&author(
            "gfx1201",
            &[launch(&[(0, D1)], 8, vec![ptr(0, &[VmemLoad])])],
        ));
        plain.group_admission().unwrap();
    }
}
