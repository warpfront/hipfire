// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The task-graph proof (design §8.2): what railgun admits about a
//! co-resident group of persistent worker pools from the *immutable task
//! headers* the pools execute.
//!
//! A pool kernel is one dispatcher loop: claim `idx = ticket++`, read
//! `tasks[idx]`, wait (acquire) until `events[pred] >= expected`, run the body
//! of the header's template on the record at `ka_off`, then release-increment
//! `events[done]`. The certificate side (`railgun-cert`, the receipt's
//! [`TaskProtocolRecord`](railgun_corpus::receipt::TaskProtocolRecord))
//! proves that dispatcher on the emitted object; this module proves, from the
//! headers the harness uploads, everything that follows from it. Nothing here
//! takes the host's word: the per-task effects come from the receipts'
//! arena facts joined with the program's bound binding records, and the
//! protocol from the receipt.
//!
//! # What [`TaskGraph::check`] proves
//!
//! 1. **Well-formed headers.** Every event id is below `event_cells` and
//!    `events` is declared `EventCells`; every `ka_off` is the base of a
//!    binding record of `arena` whose template is the header's; the pool's
//!    receipt knows that template.
//! 2. **Exact producer cardinality.** For every task `b` with a `pred`, the
//!    number of tasks (all pools) whose `done` is that event is exactly
//!    `b.expected`. More would let `b` pass on a *subset* of its producers
//!    (the counter reaches `expected` early); fewer never reaches it
//!    (deadlock). With exactly `expected` increments and counters that start
//!    at zero, `events[pred] >= expected` is true exactly when *every*
//!    producer's release has happened, and each producer's release follows
//!    its whole body (protocol), so the acquire makes all producers' writes
//!    visible: the producers happen-before `b`.
//! 3. **Progress.** Let the dependency graph have an edge from every producer
//!    to each consumer of its event, and from task `k` to task `k+1` of the
//!    same pool (claim order). It is acyclic. Argument: take a topologically
//!    minimal unfinished task `t`. Every earlier task of its pool has
//!    finished, so the ticket has handed out at most `t`'s index and `t` is
//!    claimed by the next free worker (a worker never parks while a task of
//!    its pool is unclaimed); every producer of `t` finished (it precedes `t`
//!    in the order), so `events[pred] >= expected` holds and the spin ends.
//!    Hence some unfinished task always makes progress, and all tasks finish.
//!    This needs at least one resident worker per pool, which is the group's
//!    residency proof (`Pm4Plan::admit_residency`), and the claim bound
//!    `n_tasks == tasks.len()` (checked against the kernarg).
//! 4. **No races.** For every pair of distinct tasks (same pool included)
//!    whose effects share a resource with at least one write, one
//!    happens-before the other in the transitive closure of the producer ->
//!    consumer relation of (2). A task's effects are those of its template's
//!    arena pointer loads, resolved through its record's fields; event cells
//!    are the protocol and are excluded. Claim order is *not* a
//!    happens-before edge (workers run concurrently). The granularity is the
//!    resource: a binding field carries no extent, so two tasks writing
//!    disjoint ranges of one allocation are conservatively refused unless
//!    ordered.
//! 5. **Nothing outside the protocol.** Resources a pool reaches through its
//!    explicit arguments (not through the arena) are not covered by the
//!    task headers: such an access that conflicts with any other member's is
//!    refused, so the group's ordering never rests on the headers alone for
//!    them. The tasks' arena and header buffers are never written, the claim
//!    tickets of the pools are distinct cells outside the event ids, and
//!    each pool's `events` argument is the declared event cells.
//!
//! The proof records what it cannot establish from the program as
//! `obligations` (counters and tickets zero at every submit; header bytes
//! equal to those parsed). [`TaskGraphProof::ordering`] turns it into the
//! [`GroupOrdering`] `add_coresident_group` consumes.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use railgun_corpus::receipt::{ArgRecord, TaskProtocolRecord, TASK_PROTO_REQUIRED};

use crate::coresident::{GroupOrdering, OrderedEdge};
use crate::kernel::{
    ArenaPointerLoad, ArenaSealedBinding, Arg, KernelNode, KernelProgram, MemClass, RegionEffect,
};
use crate::{BindingRecord, ResourceId, ResourceKind};

/// The "no event" value of the M0 task ABI.
pub const NONE: u32 = u32::MAX;

/// Bytes a claim ticket occupies (a `u32` counter; 8 reserved).
const TICKET_BYTES: u64 = 8;

/// The byte layout of a task header and of the event cells, as the
/// dispatcher reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskLayout {
    pub desc_stride: u32,
    pub template_offset: u32,
    pub ka_off_offset: u32,
    pub pred_event_offset: u32,
    pub expected_offset: u32,
    pub done_event_offset: u32,
    pub none_event: u32,
    pub event_stride: u32,
}

impl TaskLayout {
    /// MegaM0's `M0TaskDesc`: 128-byte records; template @0, kernarg offset
    /// @28, predecessor event @32, expected count @36, completion event @40;
    /// `u32::MAX` is "no event"; event `i` is the `u32` at `events + 4 * i`.
    pub const M0: TaskLayout = TaskLayout {
        desc_stride: 128,
        template_offset: 0,
        ka_off_offset: 28,
        pred_event_offset: 32,
        expected_offset: 36,
        done_event_offset: 40,
        none_event: NONE,
        event_stride: 4,
    };
}

/// The dispatcher protocol a pool kernel's receipt proves (a copy of
/// [`TaskProtocolRecord`], validated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskProtocolFact {
    pub layout: TaskLayout,
    pub tasks_arg_offset: u32,
    pub ticket_arg_offset: u32,
    pub events_arg_offset: u32,
    pub n_tasks_arg_offset: u32,
    /// The proved properties (`TASK_PROTO_*`).
    pub proved: Vec<String>,
}

impl TaskProtocolFact {
    /// Parse a receipt record of kernel `symbol` whose arguments are `args`:
    /// the three pointers are 8-byte arguments and the task count a 4-byte one.
    pub fn from_record(
        record: &TaskProtocolRecord,
        args: &[ArgRecord],
        symbol: &str,
    ) -> Result<Self, String> {
        record.validate(symbol)?;
        let arg = |offset: u32, size: u32, what: &str| {
            args.iter()
                .any(|a| a.offset == offset && a.size == size)
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "{symbol}: task protocol {what} argument +{offset} is not a {size}-byte argument"
                    )
                })
        };
        arg(record.tasks_arg_offset, 8, "tasks")?;
        arg(record.ticket_arg_offset, 8, "ticket")?;
        arg(record.events_arg_offset, 8, "events")?;
        arg(record.n_tasks_arg_offset, 4, "n_tasks")?;
        Ok(Self {
            layout: TaskLayout {
                desc_stride: record.desc_stride,
                template_offset: record.template_offset,
                ka_off_offset: record.ka_off_offset,
                pred_event_offset: record.pred_event_offset,
                expected_offset: record.expected_offset,
                done_event_offset: record.done_event_offset,
                none_event: record.none_event,
                event_stride: record.event_stride,
            },
            tasks_arg_offset: record.tasks_arg_offset,
            ticket_arg_offset: record.ticket_arg_offset,
            events_arg_offset: record.events_arg_offset,
            n_tasks_arg_offset: record.n_tasks_arg_offset,
            proved: record.proved.clone(),
        })
    }

    /// Every property a task-graph proof needs is proved.
    pub fn is_proven(&self) -> Result<(), String> {
        match TASK_PROTO_REQUIRED
            .iter()
            .find(|p| !self.proved.iter().any(|q| q == *p))
        {
            None => Ok(()),
            Some(missing) => Err(format!("property {missing:?} is not proved")),
        }
    }

    /// The M0 pool ABI (`M0Params`: tasks @0, ticket @8, events @16, task
    /// count @56) with **nothing proved**: the pending protocol of a lab
    /// build whose certificate has not been produced.
    #[cfg(feature = "lab")]
    fn m0_pending() -> Self {
        Self {
            layout: TaskLayout::M0,
            tasks_arg_offset: 0,
            ticket_arg_offset: 8,
            events_arg_offset: 16,
            n_tasks_arg_offset: 56,
            proved: Vec::new(),
        }
    }
}

/// The dispatcher-protocol facts of every member of a group: what the
/// certificate proved per pool kernel. The only constructors read the
/// receipts' facts out of the program
/// ([`TaskProtocol::from_program`]) or, in a lab build, mark the missing ones
/// pending; the host cannot assert a protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskProtocol {
    nodes: BTreeMap<usize, TaskProtocolFact>,
    /// Lab: some member's protocol is not proven (its certificate is
    /// pending); [`TaskGraph::check`] refuses such a protocol.
    pending: bool,
}

impl TaskProtocol {
    /// The protocol facts the receipts of `group`'s members carry. A member
    /// without a task protocol is an error.
    pub fn from_program(program: &KernelProgram, group: Range<usize>) -> Result<Self, String> {
        let mut nodes = BTreeMap::new();
        for n in group {
            let node = program
                .nodes
                .get(n)
                .ok_or_else(|| format!("group member {n} is past {} nodes", program.nodes.len()))?;
            let fact = node
                .arena
                .as_ref()
                .and_then(|a| a.task_protocol.clone())
                .ok_or_else(|| {
                    format!(
                        "node {n} ({}) has no certified task protocol (its receipt carries none)",
                        node.symbol
                    )
                })?;
            nodes.insert(n, fact);
        }
        Ok(Self {
            nodes,
            pending: false,
        })
    }

    /// Lab only: a member's receipt facts when it has them, the M0 pool ABI
    /// with nothing proved otherwise; always marked pending.
    #[cfg(feature = "lab")]
    fn lab_pending(program: &KernelProgram, group: Range<usize>) -> Self {
        let nodes = group
            .filter_map(|n| program.nodes.get(n).map(|node| (n, node)))
            .map(|(n, node)| {
                let fact = node
                    .arena
                    .as_ref()
                    .and_then(|a| a.task_protocol.clone())
                    .unwrap_or_else(TaskProtocolFact::m0_pending);
                (n, fact)
            })
            .collect();
        Self {
            nodes,
            pending: true,
        }
    }

    pub fn is_pending(&self) -> bool {
        self.pending
    }

    pub fn node(&self, node: usize) -> Option<&TaskProtocolFact> {
        self.nodes.get(&node)
    }
}

/// One task header (`M0TaskDesc`'s scheduling fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskHeader {
    /// The body template (receipt arena template id).
    pub template: u32,
    /// Byte offset in the binding-record arena of the task's record.
    pub ka_off: u32,
    /// The event the task waits on (`NONE`: none), and the counter value that
    /// releases it.
    pub pred_event: u32,
    pub expected: u32,
    /// The event the task increments on completion (`NONE`: none).
    pub done_event: u32,
}

/// The tasks of one pool, in claim order: task `k` is the one the pool's
/// `k`-th ticket value claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolTasks {
    /// The pool's kernel node.
    pub node: usize,
    pub tasks: Vec<TaskHeader>,
}

impl PoolTasks {
    /// Parse the exact bytes the harness uploads as the pool's task arena:
    /// whole `layout.desc_stride` records, the five scheduling fields read
    /// little-endian at the layout's offsets.
    pub fn parse(node: usize, bytes: &[u8], layout: &TaskLayout) -> Result<Self, String> {
        let stride = layout.desc_stride as usize;
        if stride == 0 || bytes.len() % stride != 0 {
            return Err(format!(
                "pool node {node}: {} task bytes are not whole {stride}-byte records",
                bytes.len()
            ));
        }
        let word = |rec: &[u8], at: u32| {
            let at = at as usize;
            rec.get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
                .ok_or_else(|| format!("pool node {node}: header field +{at} is past the record"))
        };
        let tasks = bytes
            .chunks_exact(stride)
            .map(|rec| {
                Ok(TaskHeader {
                    template: word(rec, layout.template_offset)?,
                    ka_off: word(rec, layout.ka_off_offset)?,
                    pred_event: word(rec, layout.pred_event_offset)?,
                    expected: word(rec, layout.expected_offset)?,
                    done_event: word(rec, layout.done_event_offset)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { node, tasks })
    }
}

/// The task graph of a co-resident group, from its immutable headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskGraph {
    /// One entry per member node, any order.
    pub pools: Vec<PoolTasks>,
    /// The declared event cells every pool's `events` argument points at.
    pub events: ResourceId,
    /// Event ids used by the headers are below this (and below the declared
    /// capacity of `events`).
    pub event_cells: u32,
    /// The bound binding-record arena the headers' `ka_off` index.
    pub arena: ResourceId,
    pub layout: TaskLayout,
}

/// What [`TaskGraph::check`] established. Only `check` (and the lab
/// `check_pending`) construct one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskGraphProof {
    group: Range<usize>,
    edges: Vec<OrderedEdge>,
    /// Tasks proved, over all pools.
    pub tasks: usize,
    /// Distinct events with at least one producer or consumer.
    pub events_used: usize,
    /// Conflicting task pairs (a write on a shared resource) proved ordered.
    pub ordered_pairs: u64,
    /// Obligations the program cannot discharge and the harness must hold at
    /// every submit.
    pub obligations: Vec<String>,
}

impl TaskGraphProof {
    /// The member range the proof covers.
    pub fn group(&self) -> &Range<usize> {
        &self.group
    }

    /// One edge per (producer node, consumer node, resource) conflict.
    pub fn edges(&self) -> &[OrderedEdge] {
        &self.edges
    }

    /// The ordering `add_coresident_group` consumes: the edges, marked as
    /// covered by a task-graph proof of this group.
    pub fn ordering(&self) -> GroupOrdering {
        GroupOrdering::from_certificate(self.edges.clone()).with_task_graph(self.group.clone())
    }
}

/// How per-template effects are obtained.
#[derive(Clone, Copy)]
struct Mode<'a> {
    /// The dispatcher protocol and per-template effects are not certified
    /// (lab): a node without bound receipt arena facts takes its templates
    /// from `templates`, else reads and writes every field of its records.
    pending: bool,
    templates: Option<&'a BTreeMap<u32, Vec<ArenaPointerLoad>>>,
}

impl Mode<'_> {
    const CERTIFIED: Mode<'static> = Mode {
        pending: false,
        templates: None,
    };
}

/// One task's access to a resource.
#[derive(Clone, Copy)]
struct Access {
    resource: ResourceId,
    read: bool,
    write: bool,
}

/// Where a pool's per-template pointer loads come from.
enum PoolEffects {
    Templates {
        loads: BTreeMap<u32, Vec<ArenaPointerLoad>>,
        /// The sealed tables bound for the pool node's templates.
        sealed: Vec<ArenaSealedBinding>,
    },
    /// Pending without facts: every field of the record is read and written.
    Conservative,
}

/// A fixed-size bitset.
#[derive(Clone)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(bits: usize) -> Self {
        Bits(vec![0; bits.div_ceil(64)])
    }

    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }

    fn get(&self, i: usize) -> bool {
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    fn or(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= *b;
        }
    }
}

/// A claim: the global index of a task and its place.
struct Index {
    /// First global index of each pool (plus the total at the end).
    base: Vec<usize>,
}

impl Index {
    fn new(graph: &TaskGraph) -> Self {
        let mut base = Vec::with_capacity(graph.pools.len() + 1);
        let mut at = 0;
        for p in &graph.pools {
            base.push(at);
            at += p.tasks.len();
        }
        base.push(at);
        Index { base }
    }

    fn total(&self) -> usize {
        *self.base.last().expect("base has the total")
    }

    fn pool_of(&self, gid: usize) -> usize {
        self.base.partition_point(|&b| b <= gid) - 1
    }
}

impl TaskGraph {
    /// A graph over the M0 task ABI ([`TaskLayout::M0`]).
    pub fn new(
        pools: Vec<PoolTasks>,
        events: ResourceId,
        event_cells: u32,
        arena: ResourceId,
    ) -> Self {
        Self {
            pools,
            events,
            event_cells,
            arena,
            layout: TaskLayout::M0,
        }
    }

    /// Prove the group `group` of `program` from its task headers and the
    /// dispatcher `protocol` its receipts carry (see the module docs).
    /// `Err` names the first violation. `Ok` only with a fully proven
    /// protocol for every member.
    pub fn check(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        protocol: &TaskProtocol,
    ) -> Result<TaskGraphProof, String> {
        if protocol.pending {
            return Err("task protocol is pending: no certificate proves the dispatcher".into());
        }
        self.prove(program, group, protocol, Mode::CERTIFIED)
    }

    /// Lab only: every host check of [`TaskGraph::check`] with the dispatcher
    /// protocol and (where a node has no receipt arena facts) the per-template
    /// effects pending: such a template is assumed to read and write every
    /// field of its records. The ordering is
    /// [`GroupOrdering::is_pending`], and `add_coresident_group` accepts it
    /// with the same edge validation as a certificate's.
    #[cfg(feature = "lab")]
    pub fn check_pending(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
    ) -> Result<GroupOrdering, String> {
        self.check_pending_with(program, group, None)
    }

    /// [`TaskGraph::check_pending`] with the pointer loads (and classes) of
    /// each template taken from `templates` for nodes that have no arena
    /// facts: for example the access classes of the standalone body kernels'
    /// own receipts. Still pending: nothing certifies them against the
    /// pool object.
    #[cfg(feature = "lab")]
    pub fn check_pending_with(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        templates: Option<&BTreeMap<u32, Vec<ArenaPointerLoad>>>,
    ) -> Result<GroupOrdering, String> {
        let protocol = TaskProtocol::lab_pending(program, group.clone());
        let mode = Mode {
            pending: true,
            templates,
        };
        let proof = self.prove(program, group.clone(), &protocol, mode)?;
        Ok(GroupOrdering::lab_pending(proof.edges).with_task_graph(group))
    }

    fn prove(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        protocol: &TaskProtocol,
        mode: Mode<'_>,
    ) -> Result<TaskGraphProof, String> {
        let pending = mode.pending;
        let layout = &self.layout;

        // ---- Shape: the pools are exactly the group's members.
        if group.end > program.nodes.len() || group.is_empty() {
            return Err(format!(
                "group {group:?} is empty or past {} nodes",
                program.nodes.len()
            ));
        }
        let mut seen = BTreeSet::new();
        for p in &self.pools {
            if !group.contains(&p.node) {
                return Err(format!("pool node {} is not in group {group:?}", p.node));
            }
            if !seen.insert(p.node) {
                return Err(format!("pool node {} is listed twice", p.node));
            }
        }
        if let Some(n) = group.clone().find(|n| !seen.contains(n)) {
            return Err(format!(
                "group member {n} ({}) has no task pool",
                program.nodes[n].symbol
            ));
        }
        if self.pools.iter().map(|p| p.tasks.len()).sum::<usize>() >= u32::MAX as usize {
            return Err("more tasks than a counter can count".into());
        }
        if !matches!(
            program.kinds.get(&self.events),
            Some(ResourceKind::EventCells { .. })
        ) {
            return Err(format!(
                "events resource {} is not declared EventCells",
                self.events.0
            ));
        }
        // Event id `i` is the `u32` at `events + i * stride`: the ids in use
        // must lie inside the declared event cells.
        let events_bytes = program.resources[self.events.0 as usize].bytes;
        let region = u64::from(self.event_cells) * u64::from(layout.event_stride);
        if self.event_cells == 0 || region > events_bytes {
            return Err(format!(
                "event ids 0..{} at stride {} need {region} bytes, the event cells hold {events_bytes}",
                self.event_cells, layout.event_stride
            ));
        }
        let Some(ResourceKind::BindingRecords(records)) = program.kinds.get(&self.arena) else {
            return Err(format!(
                "arena resource {} is not a bound binding-record arena",
                self.arena.0
            ));
        };
        let by_base: BTreeMap<u64, &BindingRecord> = records.iter().map(|r| (r.base, r)).collect();
        // Immutable header carriers: no node writes the binding records, nor
        // (certified) the task headers.
        let writes = |resource: ResourceId| {
            program.nodes.iter().position(|n| {
                n.effects
                    .iter()
                    .any(|e| e.resource == resource && e.writes())
            })
        };
        if let Some(n) = writes(self.arena) {
            return Err(format!(
                "node {n} writes binding-record arena {}",
                self.arena.0
            ));
        }

        // ---- Protocol and argument binding, per pool.
        let mut task_buffers: BTreeSet<ResourceId> = BTreeSet::new();
        let mut tickets: Vec<(ResourceId, u64, usize)> = Vec::new();
        let mut pool_effects: Vec<PoolEffects> = Vec::with_capacity(self.pools.len());
        for (pi, pool) in self.pools.iter().enumerate() {
            let n = pool.node;
            let node = &program.nodes[n];
            let who = format!("pool {pi} (node {n}, {})", node.symbol);
            let fact = protocol
                .node(n)
                .ok_or_else(|| format!("{who}: no proven task protocol"))?;
            if !pending {
                fact.is_proven()
                    .map_err(|why| format!("{who}: task protocol: {why}"))?;
            }
            if fact.layout != self.layout {
                return Err(format!(
                    "{who}: the dispatcher's task layout {:?} is not the layout the headers were parsed with {:?}",
                    fact.layout, self.layout
                ));
            }
            let arg = |offset: u32| node.args.iter().find(|(o, _)| *o == offset).map(|(_, a)| a);
            match arg(fact.events_arg_offset) {
                Some(Arg::Resource {
                    resource,
                    interior: 0,
                }) if *resource == self.events => {}
                other => {
                    return Err(format!(
                    "{who}: events argument +{} is {other:?}, not the declared event cells {} at 0",
                    fact.events_arg_offset, self.events.0
                ))
                }
            }
            let tasks_res = match arg(fact.tasks_arg_offset) {
                Some(Arg::Resource { resource, interior: 0 })
                    if program.kinds.get(resource) == Some(&ResourceKind::TaskArena) =>
                {
                    *resource
                }
                other => {
                    return Err(format!(
                        "{who}: tasks argument +{} is {other:?}, not the start of a declared task arena",
                        fact.tasks_arg_offset
                    ))
                }
            };
            if !task_buffers.insert(tasks_res) {
                return Err(format!(
                    "{who}: task arena {} is shared with another pool",
                    tasks_res.0
                ));
            }
            let need = pool.tasks.len() as u64 * u64::from(layout.desc_stride);
            if program.resources[tasks_res.0 as usize].bytes < need {
                return Err(format!(
                    "{who}: {} tasks need {need} bytes, task arena {} holds {}",
                    pool.tasks.len(),
                    tasks_res.0,
                    program.resources[tasks_res.0 as usize].bytes
                ));
            }
            if !pending {
                if let Some(w) = writes(tasks_res) {
                    return Err(format!(
                        "{who}: node {w} writes task arena {}: headers are immutable during replay",
                        tasks_res.0
                    ));
                }
            }
            match arg(fact.ticket_arg_offset) {
                Some(Arg::Resource { resource, interior })
                    if matches!(
                        program.kinds.get(resource),
                        Some(ResourceKind::EventCells { .. })
                    ) =>
                {
                    let end = interior + TICKET_BYTES;
                    if program.resources[resource.0 as usize].bytes < end {
                        return Err(format!("{who}: ticket is past its resource"));
                    }
                    if *resource == self.events
                        && *interior < u64::from(self.event_cells) * u64::from(layout.event_stride)
                    {
                        return Err(format!(
                            "{who}: ticket @{interior} lies in event ids 0..{}",
                            self.event_cells
                        ));
                    }
                    if let Some(&(_, _, other)) = tickets
                        .iter()
                        .find(|(r, i, _)| r == resource && *i < end && *interior < i + TICKET_BYTES)
                    {
                        return Err(format!(
                            "{who}: claim ticket is shared with pool {other}: tasks would be split across pools"
                        ));
                    }
                    tickets.push((*resource, *interior, pi));
                }
                other => {
                    return Err(format!(
                        "{who}: ticket argument +{} is {other:?}, not declared event cells",
                        fact.ticket_arg_offset
                    ))
                }
            }
            let at = fact.n_tasks_arg_offset as usize;
            let n_tasks = node.args.iter().find_map(|(o, a)| match a {
                Arg::Const(bytes) => {
                    let o = *o as usize;
                    bytes
                        .get(at.checked_sub(o)?..at.checked_sub(o)? + 4)
                        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
                }
                _ => None,
            });
            if n_tasks != Some(pool.tasks.len() as u32) {
                return Err(format!(
                    "{who}: n_tasks argument +{at} is {n_tasks:?}, the pool has {} tasks",
                    pool.tasks.len()
                ));
            }
            pool_effects.push(self.pool_effects(node, &who, mode)?);
        }

        // ---- Header checks and per-task effects.
        let index = Index::new(self);
        let total = index.total();
        let mut accesses: Vec<Vec<Access>> = Vec::with_capacity(total);
        let mut memo: BTreeMap<(usize, u32, u32), Vec<Access>> = BTreeMap::new();
        for (pi, pool) in self.pools.iter().enumerate() {
            for (k, t) in pool.tasks.iter().enumerate() {
                let what = format!("task {k} of pool {pi} (node {})", pool.node);
                for (name, ev) in [("pred", t.pred_event), ("done", t.done_event)] {
                    if ev != layout.none_event && ev >= self.event_cells {
                        return Err(format!(
                            "{what}: {name} event {ev} is past the {} event cells",
                            self.event_cells
                        ));
                    }
                }
                let Some(record) = by_base
                    .get(&u64::from(t.ka_off))
                    .filter(|r| r.template == t.template)
                else {
                    return Err(format!(
                        "{what} names no record of its template: {} is not the base of a template-{} record of arena {}",
                        t.ka_off, t.template, self.arena.0
                    ));
                };
                let effects = match memo.get(&(pi, t.template, t.ka_off)) {
                    Some(e) => e.clone(),
                    None => {
                        let e = self.task_effects(program, &pool_effects[pi], record, &what)?;
                        memo.insert((pi, t.template, t.ka_off), e.clone());
                        e
                    }
                };
                accesses.push(effects);
            }
        }

        // ---- Exact producer cardinality.
        let flat: Vec<&TaskHeader> = self.pools.iter().flat_map(|p| p.tasks.iter()).collect();
        let place = |gid: usize| {
            let pi = index.pool_of(gid);
            (pi, gid - index.base[pi])
        };
        let mut producers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        let mut consumers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (gid, t) in flat.iter().enumerate() {
            if t.done_event != layout.none_event {
                producers.entry(t.done_event).or_default().push(gid);
            }
            if t.pred_event != layout.none_event {
                consumers.entry(t.pred_event).or_default().push(gid);
            }
        }
        for (gid, t) in flat.iter().enumerate() {
            if t.pred_event == layout.none_event {
                continue;
            }
            let count = producers.get(&t.pred_event).map_or(0, Vec::len);
            if count != t.expected as usize {
                let (pi, k) = place(gid);
                let why = if count > t.expected as usize {
                    "the counter reaches its target before every producer finished: the consumer could pass on a subset"
                } else {
                    "the counter never reaches its target: the consumer deadlocks"
                };
                return Err(format!(
                    "task {k} of pool {pi} waits for event {} to reach {}, but {count} tasks release it: {why}",
                    t.pred_event, t.expected
                ));
            }
        }

        // ---- Progress: acyclic over (producer -> event -> consumer) and claim order.
        let event_ids: Vec<u32> = producers
            .keys()
            .chain(consumers.keys())
            .copied()
            .collect::<BTreeSet<u32>>()
            .into_iter()
            .collect();
        let event_ix = |id: u32| event_ids.binary_search(&id).expect("a used event");
        let nodes_total = total + event_ids.len();
        let mut indegree = vec![0u32; nodes_total];
        for (gid, t) in flat.iter().enumerate() {
            let (_, k) = place(gid);
            indegree[gid] = u32::from(t.pred_event != layout.none_event) + u32::from(k > 0);
        }
        for (e, id) in event_ids.iter().enumerate() {
            indegree[total + e] = producers.get(id).map_or(0, Vec::len) as u32;
        }
        let pred_ix: Vec<Option<usize>> = flat
            .iter()
            .map(|t| (t.pred_event != layout.none_event).then(|| event_ix(t.pred_event)))
            .collect();
        let consumed: Vec<bool> = event_ids
            .iter()
            .map(|id| consumers.contains_key(id))
            .collect();
        let words = total.div_ceil(64);
        let held = consumed.iter().filter(|c| **c).count() as u64 * words as u64 * 8;
        if held > 1 << 30 {
            return Err(format!(
                "{total} tasks over {} events exceed the closure budget",
                event_ids.len()
            ));
        }
        let mut before: Vec<Option<Bits>> = vec![None; event_ids.len()];
        let mut stack: Vec<usize> = (0..nodes_total).filter(|&v| indegree[v] == 0).collect();
        let mut done = vec![false; nodes_total];
        let mut processed = 0usize;
        while let Some(v) = stack.pop() {
            done[v] = true;
            processed += 1;
            if v < total {
                let t = flat[v];
                let (pi, k) = place(v);
                if t.done_event != layout.none_event {
                    let e = total + event_ix(t.done_event);
                    indegree[e] -= 1;
                    if indegree[e] == 0 {
                        stack.push(e);
                    }
                }
                if k + 1 < self.pools[pi].tasks.len() {
                    indegree[v + 1] -= 1;
                    if indegree[v + 1] == 0 {
                        stack.push(v + 1);
                    }
                }
            } else {
                let e = v - total;
                if consumed[e] {
                    // Every task that happens-before a consumer of the event: each
                    // producer and what happened-before it (a producer's own wait
                    // is an event processed earlier in this order).
                    let mut set = Bits::new(total);
                    for &a in &producers[&event_ids[e]] {
                        set.set(a);
                        if let Some(pe) = pred_ix[a] {
                            set.or(before[pe].as_ref().expect("an earlier event"));
                        }
                    }
                    before[e] = Some(set);
                    for &b in &consumers[&event_ids[e]] {
                        indegree[b] -= 1;
                        if indegree[b] == 0 {
                            stack.push(b);
                        }
                    }
                }
            }
        }
        if processed != nodes_total {
            return Err(format!(
                "no progress: {}",
                self.describe_cycle(&flat, &index, &done, &producers)
            ));
        }
        let hb = |a: usize, b: usize| {
            pred_ix[b].is_some_and(|e| before[e].as_ref().expect("a used event").get(a))
        };
        // ---- Sealed tables are immutable: no task and no member's node-level
        // effect may write a table or a pointee a pool's records bind.
        let mut protected: BTreeMap<ResourceId, &'static str> = BTreeMap::new();
        for source in &pool_effects {
            if let PoolEffects::Templates { sealed, .. } = source {
                for b in sealed {
                    protected.insert(b.resource, "sealed table");
                    for p in &b.pointees {
                        protected.entry(*p).or_insert("sealed pointee allocation");
                    }
                }
            }
        }
        for (gid, acc) in accesses.iter().enumerate() {
            if let Some(a) = acc
                .iter()
                .find(|a| a.write && protected.contains_key(&a.resource))
            {
                let (pi, k) = place(gid);
                return Err(format!(
                    "task {k} of pool {pi} (node {}) writes resource {}, a {}: sealed content is immutable",
                    self.pools[pi].node, a.resource.0, protected[&a.resource]
                ));
            }
        }
        for n in group.clone() {
            if let Some(e) = program.nodes[n]
                .effects
                .iter()
                .find(|e| e.writes() && protected.contains_key(&e.resource))
            {
                return Err(format!(
                    "node {n} ({}) writes resource {}, a {}: sealed content is immutable",
                    program.nodes[n].symbol, e.resource.0, protected[&e.resource]
                ));
            }
        }
        // ---- Races, and the node-level edges they give.
        let mut by_resource: BTreeMap<ResourceId, Vec<(usize, bool)>> = BTreeMap::new();
        let mut node_access: BTreeMap<(usize, ResourceId), (bool, bool)> = BTreeMap::new();
        for (gid, acc) in accesses.iter().enumerate() {
            let node = self.pools[index.pool_of(gid)].node;
            for a in acc {
                by_resource
                    .entry(a.resource)
                    .or_default()
                    .push((gid, a.write));
                let n = node_access.entry((node, a.resource)).or_default();
                n.0 |= a.read;
                n.1 |= a.write;
            }
        }
        // The node-level arena effects (what `bind_records` derived from the
        // receipt facts over *all* records) bound what its tasks touch; the
        // plan's group classification sees exactly these, so the edges are
        // computed over them too.
        for pool in &self.pools {
            let node = &program.nodes[pool.node];
            for e in node.effects.iter().filter(|e| self.through_arena(node, e)) {
                if matches!(
                    program.kinds.get(&e.resource),
                    Some(ResourceKind::EventCells { .. })
                ) {
                    continue;
                }
                let n = node_access.entry((pool.node, e.resource)).or_default();
                n.0 |= e.reads();
                n.1 |= e.writes();
            }
        }
        let p = self.pools.len();
        let mut ordered_pairs = 0u64;
        let mut directed: BTreeSet<(usize, usize, ResourceId)> = BTreeSet::new();
        for (&resource, list) in &by_resource {
            let mut dir = vec![false; p * p];
            for (i, &(a, a_writes)) in list.iter().enumerate() {
                for &(b, b_writes) in &list[i + 1..] {
                    if !(a_writes || b_writes) {
                        continue;
                    }
                    let (from, to) = if hb(a, b) {
                        (a, b)
                    } else if hb(b, a) {
                        (b, a)
                    } else {
                        let (pa, ka) = place(a);
                        let (pb, kb) = place(b);
                        return Err(format!(
                            "race on resource {}: task {ka} of pool {pa} (node {}) and task {kb} of pool {pb} (node {}) {} it and neither happens-before the other",
                            resource.0,
                            self.pools[pa].node,
                            self.pools[pb].node,
                            match (a_writes, b_writes) {
                                (true, true) => "both write",
                                (true, false) => "write/read",
                                _ => "read/write",
                            },
                        ));
                    };
                    ordered_pairs += 1;
                    let (pf, pt) = (index.pool_of(from), index.pool_of(to));
                    if pf != pt {
                        dir[pf * p + pt] = true;
                    }
                }
            }
            for pf in 0..p {
                for pt in 0..p {
                    if dir[pf * p + pt] {
                        directed.insert((self.pools[pf].node, self.pools[pt].node, resource));
                    }
                }
            }
        }

        // ---- Resources reached outside the arena: not covered by the headers.
        self.check_direct(program, &group, &node_access)?;

        // ---- Edges: one per node pair and conflicting resource.
        let mut edges: Vec<OrderedEdge> = Vec::new();
        let mut node_ids: Vec<usize> = self.pools.iter().map(|p| p.node).collect();
        node_ids.sort_unstable();
        for (i, &na) in node_ids.iter().enumerate() {
            for &nb in &node_ids[i + 1..] {
                let resources: BTreeSet<ResourceId> = node_access
                    .keys()
                    .filter(|(n, _)| *n == na || *n == nb)
                    .map(|(_, r)| *r)
                    .collect();
                for r in resources {
                    let (Some(&(_, wa)), Some(&(_, wb))) =
                        (node_access.get(&(na, r)), node_access.get(&(nb, r)))
                    else {
                        continue;
                    };
                    if !(wa || wb) {
                        continue;
                    }
                    let mut any = false;
                    for (producer, consumer) in [(na, nb), (nb, na)] {
                        if directed.contains(&(producer, consumer, r)) {
                            any = true;
                            edges.push(OrderedEdge {
                                producer,
                                consumer,
                                resource: r,
                                event: self.events,
                            });
                        }
                    }
                    if !any {
                        // No task pair of the two pools conflicts on `r` (the
                        // node-level effects are a superset of what the records
                        // in use touch): the ordering is vacuous, oriented
                        // writer first.
                        let (producer, consumer) = if wa { (na, nb) } else { (nb, na) };
                        edges.push(OrderedEdge {
                            producer,
                            consumer,
                            resource: r,
                            event: self.events,
                        });
                    }
                }
            }
        }
        edges.sort_by_key(|e| (e.producer, e.consumer, e.resource));

        let mut obligations = vec![
            format!(
                "event cells {} (ids 0..{}) are zero at every submit: counters start at 0",
                self.events.0, self.event_cells
            ),
            "claim tickets are zero at every submit".to_owned(),
            "task headers hold exactly the parsed bytes and are not rewritten during the submit"
                .to_owned(),
            "every pool has at least one resident workgroup (the group's residency proof)"
                .to_owned(),
        ];
        if pending {
            obligations
                .push("lab: dispatcher protocol and per-template effects are pending".into());
        }
        Ok(TaskGraphProof {
            group,
            edges,
            tasks: total,
            events_used: event_ids.len(),
            ordered_pairs,
            obligations,
        })
    }

    /// The per-template pointer loads of a pool node's task records.
    fn pool_effects(
        &self,
        node: &KernelNode,
        who: &str,
        mode: Mode<'_>,
    ) -> Result<PoolEffects, String> {
        // Bound arena facts whose arena argument is the graph's arena.
        let mut templates: BTreeMap<u32, Vec<ArenaPointerLoad>> = BTreeMap::new();
        let mut any = false;
        let mut sealed: Vec<ArenaSealedBinding> = Vec::new();
        if let Some(arena) = &node.arena {
            for (k, (fact, bound)) in arena.arenas.iter().zip(&arena.bound).enumerate() {
                let is_arena = node.args.iter().any(|(o, a)| {
                    *o == fact.arena_arg_offset
                        && matches!(a, Arg::Resource { resource, interior: 0 } if *resource == self.arena)
                });
                if !is_arena {
                    continue;
                }
                if !bound {
                    return Err(format!(
                        "{who}: the arena's records are not bound (bind_records) for the node's receipt facts"
                    ));
                }
                any = true;
                sealed.extend(arena.sealed.iter().filter(|b| b.arena == k).cloned());
                for t in &fact.templates {
                    templates
                        .entry(t.template)
                        .or_default()
                        .extend(t.pointer_loads.iter().cloned());
                }
            }
        }
        if any {
            return Ok(PoolEffects::Templates {
                loads: templates,
                sealed,
            });
        }
        if !mode.pending {
            return Err(format!(
                "{who}: pool node has no certified per-template effects (no bound arena facts over arena {})",
                self.arena.0
            ));
        }
        Ok(match mode.templates {
            Some(map) => PoolEffects::Templates {
                loads: map.clone(),
                sealed: Vec::new(),
            },
            None => PoolEffects::Conservative,
        })
    }

    /// The accesses of one task: its template's pointer loads resolved
    /// through the record's fields, event cells excluded.
    fn task_effects(
        &self,
        program: &KernelProgram,
        source: &PoolEffects,
        record: &BindingRecord,
        what: &str,
    ) -> Result<Vec<Access>, String> {
        let mut merged: BTreeMap<ResourceId, (bool, bool)> = BTreeMap::new();
        let mut add = |resource: ResourceId, read: bool, write: bool| {
            if matches!(
                program.kinds.get(&resource),
                Some(ResourceKind::EventCells { .. })
            ) {
                return;
            }
            let e = merged.entry(resource).or_default();
            e.0 |= read;
            e.1 |= write;
        };
        match source {
            PoolEffects::Conservative => {
                for f in &record.fields {
                    add(f.resource, true, true);
                }
            }
            PoolEffects::Templates { loads: map, sealed } => {
                let loads = map.get(&record.template).ok_or_else(|| {
                    format!(
                        "{what}: template {} is not in the pool node's certified templates",
                        record.template
                    )
                })?;
                for load in loads {
                    let field = record
                        .fields
                        .iter()
                        .find(|f| f.offset == load.record_offset)
                        .ok_or_else(|| {
                            format!(
                                "{what}: record @{} has no field at +{} (undeclared arena entry)",
                                record.base, load.record_offset
                            )
                        })?;
                    let classes: &[MemClass] = &load.classes;
                    if classes.is_empty() {
                        continue;
                    }
                    add(
                        field.resource,
                        classes.iter().any(|c| c.reads()),
                        classes.iter().any(|c| c.writes()),
                    );
                }
                // The sealed tables `bind_records` discharged: the task reads
                // the table its record names and every pointee of that table
                // (read-only, so these only conflict with a writer, which
                // `prove` refuses outright).
                for b in sealed.iter().filter(|b| b.template == record.template) {
                    let names_table = record
                        .fields
                        .iter()
                        .any(|f| f.offset == b.table_arg_offset && f.resource == b.resource);
                    if !names_table {
                        continue;
                    }
                    add(b.resource, true, false);
                    for p in &b.pointees {
                        add(*p, true, false);
                    }
                }
            }
        }
        Ok(merged
            .into_iter()
            .map(|(resource, (read, write))| Access {
                resource,
                read,
                write,
            })
            .collect())
    }

    /// Whether node effect `e` is reached through the binding-record arena
    /// (an arena pointer load, so covered by the task headers) rather than a
    /// pointer argument of its own.
    fn through_arena(&self, node: &KernelNode, e: &RegionEffect) -> bool {
        e.resource != self.arena && node.args.iter().any(|(o, a)| {
            *o == e.arg_offset
                && matches!(a, Arg::Resource { resource, interior: 0 } if *resource == self.arena)
        })
    }

    /// Accesses a pool makes outside the arena (through its explicit
    /// pointer arguments) are not covered by the headers: refuse any that
    /// conflicts with an access of another member, or with an arena access.
    fn check_direct(
        &self,
        program: &KernelProgram,
        group: &Range<usize>,
        arena_access: &BTreeMap<(usize, ResourceId), (bool, bool)>,
    ) -> Result<(), String> {
        let mut direct: BTreeMap<ResourceId, Vec<(usize, bool, bool)>> = BTreeMap::new();
        for n in group.clone() {
            let node = &program.nodes[n];
            for e in &node.effects {
                if self.through_arena(node, e) {
                    continue; // covered by the headers
                }
                if matches!(
                    program.kinds.get(&e.resource),
                    Some(
                        ResourceKind::EventCells { .. }
                            | ResourceKind::TaskArena
                            | ResourceKind::BindingRecords(_)
                    )
                ) {
                    continue; // the protocol's own resources (immutability checked)
                }
                direct
                    .entry(e.resource)
                    .or_default()
                    .push((n, e.reads(), e.writes()));
            }
        }
        for (resource, list) in direct {
            for (i, &(n, r, w)) in list.iter().enumerate() {
                for &(m, r2, w2) in &list[i + 1..] {
                    if n != m && (w || w2) {
                        return Err(format!(
                            "resource {} is accessed outside the task protocol by nodes {n} and {m} (read {r}/{r2}, write {w}/{w2}): the task headers do not order it",
                            resource.0
                        ));
                    }
                }
                for (&(m, res), &(ar, aw)) in arena_access.iter() {
                    if res == resource && (w || aw) {
                        return Err(format!(
                            "resource {} is accessed outside the task protocol by node {n} and through the arena by node {m} (read {r}/{ar}, write {w}/{aw}): the task headers do not order it",
                            resource.0
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// A readable dependency cycle among the unfinished nodes of the
    /// progress search.
    fn describe_cycle(
        &self,
        flat: &[&TaskHeader],
        index: &Index,
        done: &[bool],
        producers: &BTreeMap<u32, Vec<usize>>,
    ) -> String {
        let total = flat.len();
        let place = |gid: usize| {
            let pi = index.pool_of(gid);
            (pi, gid - index.base[pi])
        };
        let Some(start) = (0..total).find(|&g| !done[g]) else {
            return "an event waits on tasks that never run".into();
        };
        // Each unfinished task has an unfinished predecessor: a producer of
        // its event or its claim predecessor. Walk predecessors until a task
        // repeats; `path[i].1` says how `path[i]` depends on `path[i + 1]`.
        let mut path: Vec<(usize, String)> = Vec::new();
        let mut cur = start;
        loop {
            if let Some(at) = path.iter().position(|(g, _)| *g == cur) {
                let step = |g: usize| {
                    let (pi, k) = place(g);
                    format!("task {k} of pool {pi}")
                };
                let mut out = String::from("dependency cycle: ");
                for (g, how) in &path[at..] {
                    out.push_str(&format!("{} {how} ", step(*g)));
                }
                out.push_str(&step(cur));
                return out;
            }
            let (pi, k) = place(cur);
            let pred = flat[cur].pred_event;
            let producer = (pred != self.layout.none_event)
                .then(|| {
                    producers
                        .get(&pred)
                        .and_then(|ps| ps.iter().copied().find(|&a| !done[a]))
                })
                .flatten();
            let (next, how) = match producer {
                Some(a) => (a, format!("waits for event {pred}, released by")),
                None if k > 0 && !done[cur - 1] => (cur - 1, "is claimed after".to_owned()),
                None => return format!("task {k} of pool {pi} never becomes runnable"),
            };
            path.push((cur, how));
            cur = next;
        }
    }
}

#[cfg(test)]
mod tests;
