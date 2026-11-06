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
//! 6. **Spans (GPU-generated routed tasks, `DYNAMIC.md`).** A header with
//!    flag bit 0 is a *span task*: the dispatcher, not the header, picks the
//!    workgroup (entry `k` of a producer-written live list, validated at
//!    consume time). The `K` span tasks of one list are one launch
//!    instance; see [`SpanDescriptor`] for the table they name. Because the
//!    live coordinates are unknown on the host, the runtime claim bitmap
//!    (certificate property `span_entry_validated`) replaces the `W`-set
//!    disjointness of (4), and a same-instance conflict is admitted only
//!    if its template separates **every** pair of distinct workgroups of
//!    the grid. The live list and count cell are written only by the
//!    producer, the bitmap only by the pool nodes. The result is a safety
//!    proof, **not** a completeness proof (the admission receipt says so,
//!    see `Admission::receipt_with_completeness`).
//!
//! The proof records what it cannot establish from the program as
//! `obligations` (counters and tickets zero at every submit; header bytes
//! equal to those parsed). [`TaskGraphProof::ordering`] turns it into the
//! [`GroupOrdering`] `add_coresident_group` consumes.
//!
//! # Complexity
//!
//! `n` tasks, `E` events some task waits on, `w = ceil(n / 64)`. Header
//! checks, per-task effects, producer cardinality and launch instances are
//! `O(n log n)`; progress and the happens-before closure are `O(n + E·w)`
//! (one `w`-word row `before[e]` per awaited event, refused above 1 GiB).
//! The race pass is **not** the all-pairs scan of (4), which is `O(n^2)` per
//! resource (one MoE launch of 38,400 same-instance tasks took ~80 s):
//! per resource with `m` accessing tasks it is `O(m log m + G·w)`, `G` the
//! number of distinct `(event awaited, writes)` among them, plus `O(m·w)`
//! only to *name* a race once one is known to exist (a refusal). Launch
//! instances whose members are not uniform, or are ordered among themselves
//! (neither happens in one eager launch), are checked pairwise.
//!
//! ## Why the race pass agrees with all pairs
//!
//! The reference semantics (`Races::check_pairwise`, the test oracle; the
//! tests compare verdict, message and the whole [`TaskGraphProof`] on random
//! graphs) visits every pair `a < b` of tasks accessing a resource with a
//! write, resources in order: `a -> b` or `b -> a` (happens-before) is
//! ordered, an unordered pair of one launch instance goes to the instance
//! rule, any other unordered pair is a race, and the first refusal in
//! `(resource, a, b)` order is the error. Happens-before `->` is a strict
//! partial order (the progress search proved the dependency graph acyclic),
//! so a conflicting pair is ordered in exactly one direction or unordered.
//!
//! * *Ordered pairs and edges.* `a -> b` iff `a` is in `before[pred(b)]`, and
//!   `a` conflicts with `b` iff `b` writes (any accessor `a`) or `a` writes.
//!   So the ordered pairs ending in `b` number
//!   `|before[pred(b)] ∩ (accessors if b writes else writers)|`, the same
//!   for every task of one `(pred, writes)`: one popcount per group.
//!   `ordered_pairs` is their sum. The same intersection, cut at the pool
//!   boundaries (a pool's tasks are one index range), says which pool pairs
//!   have an ordered pair: the `directed` set the node edges come from.
//! * *Unordered pairs are counted, not visited.* Conflicting pairs are
//!   `C(m, 2) - C(readers, 2)`; unordered ones are those minus the ordered
//!   ones. Inside one instance the same formula over its members counts its
//!   unordered pairs. The unordered pairs outside every instance are the
//!   difference of the two totals: zero iff there is no race.
//! * *Instances.* Members of one instance run on one `(template, ka_off)`
//!   record, so they share the header grid, the record and the footprint
//!   assumptions: whether a pair is admitted depends only on the template's
//!   rule for `a` (commutative, unproven, or a footprint) and, for a
//!   footprint whose assumptions hold, on coordinates. If the members are
//!   uniform (same write flag, same certified concurrency) and unordered
//!   among themselves, every pair is conflicting and unordered, the first
//!   pair's rule is every pair's, and a footprint over axes `A` admits a
//!   pair iff it differs on an axis of `A`: `vy` / `vz` unequal, x-intervals
//!   disjoint. The *refused* pairs are therefore those agreeing on `vy` (if
//!   `y` in `A`) and `vz` (if `z` in `A`) and, if `x` in `A`, overlapping
//!   in x: found by sorting on `(vy, vz)` (`O(m log m)`), where without `x`
//!   the first two members of a key group are the lowest pair and with it a
//!   sweep over a set of merged x-intervals finds the lowest task that
//!   overlaps a later one. Per instance this is the lowest refused pair of
//!   the all-pairs scan, and the lowest over instances is the instance
//!   candidate.
//! * *Naming a race.* Process the tasks of the resource in the order the
//!   progress search finished them: every `->` edge goes forward in it, so
//!   for `y` before `x` the pair is unordered iff `y` is not in
//!   `before[pred(x)]`. The unordered conflicting `y` outside `x`'s instance
//!   form a bitset of `w` words (running sets of the earlier accessors and
//!   writers, minus the row, minus the instance's members); its lowest bit
//!   `y` gives the lowest pair `x` is in (`(y, x)` if `y < x`, else
//!   `(x, y)`, all its bits being above `x`). The lowest over all `x` is
//!   the first race.
//! * *The error.* The sweep reports the lowest of the race and instance
//!   candidates of the first resource that has one, which is the first
//!   refusal of the all-pairs scan; the messages are built by the same
//!   code from the same pair. Nothing is counted on a refused resource.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use railgun_corpus::receipt::{
    ArgRecord, RequireOp, TaskGridOffsets, TaskProtocolRecord, TaskSpanOffsets,
    TASK_PROTO_REGION_WG, TASK_PROTO_REGION_WG_LIVE, TASK_PROTO_REQUIRED, TASK_PROTO_SPAN_ENTRY,
};

use crate::coresident::{GroupOrdering, OrderedEdge};
use crate::kernel::{
    ArenaPointerLoad, ArenaSealedBinding, Arg, KernelNode, KernelProgram, MemClass, RegionEffect,
    WgFootprint,
};
use crate::{BindingField, BindingRecord, ResourceId, ResourceKind, LIVE_ENTRY_BYTES};

/// The "no event" value of the M0 task ABI.
pub const NONE: u32 = u32::MAX;

/// Header flag bit 0 (`M0TaskDesc::flags`): the task is a span task. Any
/// other bit set makes the dispatcher poison, so a header with one is refused.
pub const TASK_FLAG_SPAN: u32 = 1;

/// Kernarg offset of `M0Params::spans`, the read-only pointer to the span
/// table (zero when no task is in span mode).
pub const M0_SPANS_ARG_OFFSET: u32 = 64;

/// Template ids of the span table's binding records. The table is an
/// ordinary [`ResourceKind::BindingRecords`] arena railgun writes at prepare:
/// a header record at byte 0 (template [`SPAN_HEADER_TEMPLATE`], no pointer
/// loads) carrying the scalar `n_spans` at +0, then one descriptor record
/// (template [`SPAN_DESC_TEMPLATE`] = the receipt's
/// `TaskSpanOffsets::nop_template`) per list at `32 + 32 * i`
/// ([`span_table_records`]). The pool's arena fact at the `spans` argument
/// names both templates.
pub const SPAN_HEADER_TEMPLATE: u32 = 0xFFFF_FFFD;
pub const SPAN_DESC_TEMPLATE: u32 = TaskSpanOffsets::M0.nop_template;
/// Bytes of the span table's header and of each descriptor.
pub const SPAN_TABLE_HEADER_BYTES: u64 = 32;
pub const SPAN_DESC_BYTES: u64 = 32;

/// One span list of the span table (descriptor `i` at `spans + 32 + 32 * i`:
/// `{ u64 live_list, u64 count_cell, u64 bitmap, u32 K, u32 0 }`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanDescriptor {
    /// The [`ResourceKind::LiveList`] the producer writes (`cap >= K`).
    pub live_list: ResourceId,
    /// The `u32` count cell of valid entries (declared `EventCells`),
    /// written only by the producer.
    pub count: ResourceId,
    /// The claim bitmap, `ceil(gx * gy * gz / 32)` `u32` words (declared
    /// `EventCells`), written only by the dispatcher (atomic OR).
    pub bitmap: ResourceId,
    /// The host upper bound on live entries, and the number of span tasks.
    pub k: u32,
}

/// The binding records of a span table over `spans`, to pass to
/// [`crate::kernel::KernelProgram::bind_records`] (railgun writes the
/// pointers and scalars with [`crate::kernel::KernelProgram::binding_record_image`]
/// like any record arena). The arena needs [`span_table_bytes`] bytes.
pub fn span_table_records(spans: &[SpanDescriptor]) -> Result<Vec<BindingRecord>, String> {
    let n = u32::try_from(spans.len()).map_err(|_| "more spans than a u32 counts".to_owned())?;
    let mut records = vec![BindingRecord {
        base: 0,
        template: SPAN_HEADER_TEMPLATE,
        fields: Vec::new(),
        scalars: vec![(0, n)],
    }];
    for (i, s) in spans.iter().enumerate() {
        let field = |offset, resource| BindingField {
            offset,
            resource,
            interior: 0,
        };
        records.push(BindingRecord {
            base: SPAN_TABLE_HEADER_BYTES + SPAN_DESC_BYTES * i as u64,
            template: SPAN_DESC_TEMPLATE,
            fields: vec![
                field(0, s.live_list),
                field(8, s.count),
                field(16, s.bitmap),
            ],
            scalars: vec![(24, s.k), (28, 0)],
        });
    }
    Ok(records)
}

/// Bytes a span table of `n_spans` lists occupies.
pub fn span_table_bytes(n_spans: usize) -> u64 {
    SPAN_TABLE_HEADER_BYTES + SPAN_DESC_BYTES * n_spans as u64
}

/// The span table of binding-record arena `arena`: the exact inverse of
/// [`span_table_records`] (anything else is refused), inside the arena's
/// declared size.
fn read_span_table(
    program: &KernelProgram,
    arena: ResourceId,
) -> Result<Vec<SpanDescriptor>, String> {
    let Some(ResourceKind::BindingRecords(records)) = program.kinds.get(&arena) else {
        return Err(format!(
            "span table resource {} is not a bound binding-record arena",
            arena.0
        ));
    };
    let bad = |what: String| Err(format!("span table {}: {what}", arena.0));
    let by_base: BTreeMap<u64, &BindingRecord> = records.iter().map(|r| (r.base, r)).collect();
    let Some(header) = by_base.get(&0) else {
        return bad("no header record at +0".into());
    };
    if header.template != SPAN_HEADER_TEMPLATE
        || !header.fields.is_empty()
        || header.scalars.len() != 1
        || header.scalars[0].0 != 0
    {
        return bad(format!(
            "the header record must be template {SPAN_HEADER_TEMPLATE:#x} with the one scalar n_spans at +0"
        ));
    }
    let n = header.scalars[0].1 as usize;
    if records.len() != n + 1 {
        return bad(format!(
            "n_spans is {n}, but the table has {} descriptor records",
            records.len() - 1
        ));
    }
    let need = span_table_bytes(n);
    let have = program.resources[arena.0 as usize].bytes;
    if have < need {
        return bad(format!(
            "{n} spans need {need} bytes, the arena holds {have}"
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = SPAN_TABLE_HEADER_BYTES + SPAN_DESC_BYTES * i as u64;
        let Some(rec) = by_base.get(&base) else {
            return bad(format!("no descriptor record at +{base}"));
        };
        let mut fields: Vec<(u32, ResourceId, u64)> = rec
            .fields
            .iter()
            .map(|f| (f.offset, f.resource, f.interior))
            .collect();
        fields.sort_unstable_by_key(|f| f.0);
        let mut scalars = rec.scalars.clone();
        scalars.sort_unstable();
        let ok = rec.template == SPAN_DESC_TEMPLATE
            && fields.len() == 3
            && fields.iter().map(|f| f.0).eq([0, 8, 16])
            && fields.iter().all(|f| f.2 == 0)
            && scalars.len() == 2
            && scalars[0].0 == 24
            && scalars[1] == (28, 0);
        if !ok {
            return bad(format!(
                "descriptor record @{base} must be template {SPAN_DESC_TEMPLATE:#x} with pointers at +0/+8/+16 (interior 0), the scalar K at +24 and 0 at +28"
            ));
        }
        out.push(SpanDescriptor {
            live_list: fields[0].1,
            count: fields[1].1,
            bitmap: fields[2].1,
            k: scalars[0].1,
        });
    }
    Ok(out)
}

/// What [`TaskGraph::check`] established about one span list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanListProof {
    pub list_id: u32,
    /// The consumer template and its record (one launch instance).
    pub template: u32,
    pub ka_off: u32,
    /// `K`: the number of span tasks, and the bound the dispatcher checks
    /// the count cell against; the virtual grid every entry is checked in.
    pub k: u32,
    pub grid: (u32, u32, u32),
    /// The producer: `(pool index, task index in the pool)`, the only
    /// releaser of the list's pred event, with `expected = 1`.
    pub producer: (usize, usize),
    pub live_list: ResourceId,
    pub count: ResourceId,
    pub bitmap: ResourceId,
}

/// Who may write a resource of a span list.
enum SpanRole {
    /// The live list or the count cell: the producer's task (global index)
    /// and node only.
    Producer { gid: usize, node: usize },
    /// The claim bitmap: the pool nodes only (every dispatcher carries the
    /// span code, whose atomic OR is the one writer), no task.
    Bitmap { nodes: BTreeSet<usize> },
}

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
    /// Header offsets of the span-mode words: the flags (bit 0 = span mode),
    /// the span-relative index `k` and the span-table list id.
    pub flags_offset: u32,
    pub k_offset: u32,
    pub list_id_offset: u32,
    /// Offsets of the virtual-grid fields (`vx, vy, vz, gx, gy, gz, vcount`).
    pub grid: TaskGridOffsets,
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
        grid: TaskGridOffsets {
            vx_offset: 4,
            vy_offset: 8,
            vz_offset: 12,
            gx_offset: 16,
            gy_offset: 20,
            gz_offset: 24,
            vcount_offset: 44,
        },
        flags_offset: 48,
        k_offset: 52,
        list_id_offset: 56,
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
        if let Some(span) = &record.span {
            // The host writes the span table and reads the span words at the
            // M0 layout only; another certified ABI is not one it implements.
            if *span != TaskSpanOffsets::M0 {
                return Err(format!(
                    "{symbol}: task protocol span layout {span:?} is not the M0 span ABI"
                ));
            }
        }
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
                // A receipt that does not carry the offsets reads the M0 ones;
                // `region_wg_from_header` is only honoured with them proved.
                flags_offset: TaskLayout::M0.flags_offset,
                k_offset: TaskLayout::M0.k_offset,
                list_id_offset: TaskLayout::M0.list_id_offset,
                grid: record.grid.unwrap_or(TaskLayout::M0.grid),
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

    /// The certificate proved the region runs at virtual workgroup
    /// `(vx + step, vyz)` for `step < max(vcount, 1)` from the header fields.
    pub fn proves_region_wg(&self) -> bool {
        self.proved.iter().any(|p| p == TASK_PROTO_REGION_WG)
    }

    /// The certificate proved the dispatcher's span mode: the entry
    /// validation (`span_entry_validated`) and the region's workgroup taken
    /// from the validated live entry (`region_wg_from_live_entry`).
    pub fn proves_span(&self) -> bool {
        [TASK_PROTO_SPAN_ENTRY, TASK_PROTO_REGION_WG_LIVE]
            .iter()
            .all(|p| self.proved.iter().any(|q| q == p))
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// The virtual workgroup the task's region starts at, the record's
    /// virtual grid, and the number of consecutive x workgroups claimed
    /// (`0` and `1` both mean one).
    pub vx: u32,
    pub vy: u32,
    pub vz: u32,
    pub gx: u32,
    pub gy: u32,
    pub gz: u32,
    pub vcount: u32,
    /// Span mode: header flags (bit 0 = span task, any other bit is refused),
    /// the span-relative index `k` and the span-table list id. All zero for
    /// an ordinary task.
    pub flags: u32,
    pub k: u32,
    pub list_id: u32,
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
                    vx: word(rec, layout.grid.vx_offset)?,
                    vy: word(rec, layout.grid.vy_offset)?,
                    vz: word(rec, layout.grid.vz_offset)?,
                    gx: word(rec, layout.grid.gx_offset)?,
                    gy: word(rec, layout.grid.gy_offset)?,
                    gz: word(rec, layout.grid.gz_offset)?,
                    vcount: word(rec, layout.grid.vcount_offset)?,
                    flags: word(rec, layout.flags_offset)?,
                    k: word(rec, layout.k_offset)?,
                    list_id: word(rec, layout.list_id_offset)?,
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
    /// The binding-record arena holding the span table the pools' `spans`
    /// argument points at ([`TaskGraph::with_spans`]); `None` when no task
    /// is in span mode.
    pub spans: Option<ResourceId>,
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
    /// Lab only: named reasons the proof is uncertified beyond the pending
    /// protocol (`inter-WG disjointness unproven: template <t>, resource
    /// <r>`, one per pair, sorted); empty for a certified proof.
    pub pending: Vec<String>,
    /// The span lists proved (empty without span tasks).
    pub spans: Vec<SpanListProof>,
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
        GroupOrdering::from_certificate(self.edges.clone())
            .with_task_graph(self.group.clone())
            .with_spans(self.spans.len())
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
#[derive(Clone)]
struct Access {
    resource: ResourceId,
    read: bool,
    write: bool,
    /// What the template certifies about this resource across the workgroups
    /// of its own grid.
    conc: Conc,
}

/// How a template's accesses to one resource behave across two workgroups of
/// the same launch.
#[derive(Clone, PartialEq, Eq)]
enum Conc {
    /// Nothing certified: concurrent workgroups may conflict.
    None,
    /// Every access is a commutative atomic (`AtomicCommutative` /
    /// `AtomicRelease` only): concurrent workgroups commute.
    Commutative,
    /// One write-only pointer field with a certified `WgDisjoint` footprint
    /// (axes mask: bit 0 x, 1 y, 2 z) and its proof assumptions.
    Footprint(WgFootprint),
}

/// Where a pool's per-template pointer loads come from.
enum PoolEffects {
    Templates {
        loads: BTreeMap<u32, Vec<ArenaPointerLoad>>,
        /// Certified per-workgroup write footprints by template.
        footprints: BTreeMap<u32, Vec<WgFootprint>>,
        /// The sealed tables bound for the pool node's templates.
        sealed: Vec<ArenaSealedBinding>,
        /// Hidden grid arguments the template bodies load, by template.
        hidden: BTreeMap<u32, Vec<(u32, String)>>,
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

/// The tasks that are workgroups of one eager launch.
///
/// Two tasks `a != b` are the same launch instance iff they have the same
/// template and the same `ka_off` (the same binding record). Preconditions,
/// refused when violated: (i) each task's virtual workgroup set
/// `W(t) = {(vx + s, vy, vz) : s in [0, max(vcount, 1))}` lies inside the
/// record's virtual grid `[0, gx) x [0, gy) x [0, gz)` and (ii) the `W` sets
/// of the instance's tasks are pairwise disjoint. They need the certificate
/// to prove `region_wg_from_header` (the dispatcher runs the region at
/// `(vx + step, vyz)` for `step < max(vcount, 1)` from those header fields),
/// so `W(t)` is what actually runs; without it the instance rule is off and
/// every conflicting pair needs happens-before.
///
/// Given them, the tasks of one instance are exactly the concurrent
/// workgroups of one eager launch of the certified standalone grid kernel.
/// That kernel's certificate says nothing about workgroups racing each other
/// on memory, so a conflicting same-instance pair on a resource is admitted
/// only by the template's own facts: (A) every access of the template to it
/// is a commutative atomic, or (B) its one write-only pointer field has a
/// certified `WgDisjoint` footprint over axes such that every workgroup of
/// `a` and every workgroup of `b` differ on one of them
/// ([`Races::same_instance_pair`]). Otherwise the pair is a race.
/// Across instances, and between an instance's tasks and any other task,
/// every conflicting pair needs happens-before.
struct Instances(Vec<Option<(u32, u32)>>);

impl Instances {
    /// The all-pairs oracle's test: both tasks run in one instance.
    #[cfg(test)]
    fn same(&self, a: usize, b: usize) -> bool {
        matches!((self.0[a], self.0[b]), (Some(x), Some(y)) if x == y)
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
            spans: None,
        }
    }

    /// This graph's span tasks index the span table in binding-record arena
    /// `arena` ([`span_table_records`]).
    pub fn with_spans(mut self, arena: ResourceId) -> Self {
        self.spans = Some(arena);
        self
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
        let reasons = proof.pending.clone();
        let lists = proof.spans.len();
        Ok(GroupOrdering::lab_pending(proof.edges)
            .with_pending_reasons(reasons)
            .with_task_graph(group)
            .with_spans(lists))
    }

    fn prove(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        protocol: &TaskProtocol,
        mode: Mode<'_>,
    ) -> Result<TaskGraphProof, String> {
        self.prove_with(program, group, protocol, mode, RacePass::Sweep)
    }

    /// [`TaskGraph::check`] with the all-pairs race pass: the oracle the
    /// sweep must agree with, proof and message, on every input.
    #[cfg(test)]
    fn check_pairwise(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        protocol: &TaskProtocol,
    ) -> Result<TaskGraphProof, String> {
        if protocol.pending {
            return Err("task protocol is pending: no certificate proves the dispatcher".into());
        }
        self.prove_with(
            program,
            group,
            protocol,
            Mode::CERTIFIED,
            RacePass::Pairwise,
        )
    }

    /// [`TaskGraph::check_pending`]'s proof, with the sweep or the all-pairs
    /// race pass.
    #[cfg(all(test, feature = "lab"))]
    fn prove_pending(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        pairwise: bool,
    ) -> Result<TaskGraphProof, String> {
        let protocol = TaskProtocol::lab_pending(program, group.clone());
        let mode = Mode {
            pending: true,
            templates: None,
        };
        let pass = if pairwise {
            RacePass::Pairwise
        } else {
            RacePass::Sweep
        };
        self.prove_with(program, group, &protocol, mode, pass)
    }

    fn prove_with(
        &self,
        program: &KernelProgram,
        group: Range<usize>,
        protocol: &TaskProtocol,
        mode: Mode<'_>,
        pass: RacePass,
    ) -> Result<TaskGraphProof, String> {
        let pending = mode.pending;
        let mut pending_reasons: Option<BTreeSet<String>> = pending.then(BTreeSet::new);
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
                self.grid_tie(&pool_effects[pi], record, t, &what)?;
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
        // ---- Span lists: header validation, table, producer and writers.
        let spans = self.check_spans(
            program,
            protocol,
            &pool_effects,
            &by_base,
            &flat,
            &index,
            (&producers, &consumers),
            pending_reasons.as_mut(),
        )?;
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
        let mut topo = vec![0usize; total];
        let mut processed = 0usize;
        while let Some(v) = stack.pop() {
            done[v] = true;
            if v < total {
                topo[v] = processed;
            }
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
        // ---- Same-launch instances: tasks of one record (same template and
        // `ka_off`) whose virtual workgroup sets are inside the grid and
        // pairwise disjoint are the intra-grid concurrency of one launch, not
        // a hazard between launches (see `Instances`).
        let wg_ok: Vec<bool> = self
            .pools
            .iter()
            .map(|p| {
                pending
                    || protocol
                        .node(p.node)
                        .is_some_and(TaskProtocolFact::proves_region_wg)
            })
            .collect();
        let instances = self.instances(&flat, &index, &wg_ok)?;
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
        let races = Races {
            graph: self,
            flat: &flat,
            index: &index,
            accesses: &accesses,
            by_base: &by_base,
            instances: &instances,
            pred_ix: &pred_ix,
            before: &before,
            topo: &topo,
        };
        let totals = match pass {
            RacePass::Sweep => races.check(&by_resource, pending_reasons.as_mut())?,
            #[cfg(test)]
            RacePass::Pairwise => races.check_pairwise(&by_resource, pending_reasons.as_mut())?,
        };
        let RaceTotals {
            ordered_pairs,
            directed,
        } = totals;

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
        if !spans.is_empty() {
            let cells: Vec<String> = spans
                .iter()
                .flat_map(|s| [s.count.0, s.bitmap.0])
                .map(|r| r.to_string())
                .collect();
            obligations.push(format!(
                "span count cells and claim bitmaps (resources {}) are zero at every submit",
                cells.join(", ")
            ));
            obligations.push(
                "span completeness is not certified: the live lists are exactly the eager live set only by functional evidence (EXPAND_LIVE), which the admission receipt must carry"
                    .to_owned(),
            );
        }
        Ok(TaskGraphProof {
            group,
            edges,
            tasks: total,
            events_used: event_ids.len(),
            ordered_pairs,
            spans,
            obligations,
            pending: pending_reasons
                .map(|s| s.into_iter().collect())
                .unwrap_or_default(),
        })
    }

    /// Group tasks by launch instance and validate each qualifying instance.
    fn instances(
        &self,
        flat: &[&TaskHeader],
        index: &Index,
        wg_ok: &[bool],
    ) -> Result<Instances, String> {
        let mut by_key: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (gid, t) in flat.iter().enumerate() {
            by_key.entry((t.template, t.ka_off)).or_default().push(gid);
        }
        let place = |gid: usize| {
            let pi = index.pool_of(gid);
            (pi, gid - index.base[pi])
        };
        let who = |gid: usize| {
            let (pi, k) = place(gid);
            format!("task {k} of pool {pi} (node {})", self.pools[pi].node)
        };
        let mut of = vec![None; flat.len()];
        for (key, gids) in by_key {
            // A span list is one instance by construction (`check_spans`):
            // the dispatcher picks each task's workgroup and the claim bitmap
            // keeps them distinct, so its header coordinates are not `W` sets.
            let span = flat[gids[0]].flags == TASK_FLAG_SPAN;
            // One task is no concurrency; without the proof the rule is off.
            if gids.len() < 2 || (!span && gids.iter().any(|&g| !wg_ok[index.pool_of(g)])) {
                continue;
            }
            if span {
                for &g in &gids {
                    of[g] = Some(key);
                }
                continue;
            }
            let first = flat[gids[0]];
            let mut spans: Vec<(u32, u32, u64, u64, usize)> = Vec::with_capacity(gids.len());
            for &g in &gids {
                let t = flat[g];
                if (t.gx, t.gy, t.gz) != (first.gx, first.gy, first.gz) {
                    return Err(format!(
                        "{}: virtual grid ({}, {}, {}) differs from {}'s ({}, {}, {}) in the same launch instance (template {}, ka_off {})",
                        who(g), t.gx, t.gy, t.gz, who(gids[0]), first.gx, first.gy, first.gz, key.0, key.1
                    ));
                }
                let n = u64::from(t.vcount.max(1));
                let end = u64::from(t.vx) + n;
                if t.vy >= t.gy || t.vz >= t.gz || end > u64::from(t.gx) {
                    return Err(format!(
                        "{}: virtual workgroups x {}..{end} at (y {}, z {}) lie outside the record's virtual grid ({}, {}, {})",
                        who(g), t.vx, t.vy, t.vz, t.gx, t.gy, t.gz
                    ));
                }
                // The certified dispatcher packs wg_yz = vy | vz << 16: exact only for
                // y and z extents of at most 65536.
                if t.gy > 1 << 16 || t.gz > 1 << 16 {
                    return Err(format!(
                        "{}: virtual grid ({}, {}, {}) exceeds the dispatcher's 16-bit y/z workgroup packing",
                        who(g), t.gx, t.gy, t.gz
                    ));
                }
                spans.push((t.vy, t.vz, u64::from(t.vx), end, g));
            }
            spans.sort_unstable();
            for w in spans.windows(2) {
                let (a, b) = (w[0], w[1]);
                if (a.0, a.1) == (b.0, b.1) && b.2 < a.3 {
                    return Err(format!(
                        "{} and {} both run virtual workgroup x {} at (y {}, z {}) of the same launch instance (template {}, ka_off {}): one workgroup would run twice concurrently",
                        who(a.4), who(b.4), b.2, a.0, a.1, key.0, key.1
                    ));
                }
            }
            let id = u32::try_from(key.1).ok().map(|_| key);
            for &g in &gids {
                of[g] = id;
            }
        }
        Ok(Instances(of))
    }

    /// Span mode (see the module docs, item 6): the headers, the span table,
    /// the producer's cardinality and the writers of the live lists, count
    /// cells and claim bitmaps. `Ok(vec![])` when no task is in span mode.
    /// In a lab build (`pending`) a pool whose certificate lacks the span
    /// properties is a named pending reason, not a refusal.
    #[allow(clippy::too_many_arguments)]
    fn check_spans(
        &self,
        program: &KernelProgram,
        protocol: &TaskProtocol,
        pool_effects: &[PoolEffects],
        by_base: &BTreeMap<u64, &BindingRecord>,
        flat: &[&TaskHeader],
        index: &Index,
        (producers, consumers): (&BTreeMap<u32, Vec<usize>>, &BTreeMap<u32, Vec<usize>>),
        mut pending: Option<&mut BTreeSet<String>>,
    ) -> Result<Vec<SpanListProof>, String> {
        let layout = &self.layout;
        let place = |gid: usize| {
            let pi = index.pool_of(gid);
            (pi, gid - index.base[pi])
        };
        let who = |gid: usize| {
            let (pi, k) = place(gid);
            format!("task {k} of pool {pi} (node {})", self.pools[pi].node)
        };

        // ---- Flags: 0 or exactly the span bit.
        let mut lists: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (gid, t) in flat.iter().enumerate() {
            match t.flags {
                0 => {}
                TASK_FLAG_SPAN => lists.entry(t.list_id).or_default().push(gid),
                f => {
                    return Err(format!(
                        "{}: header flags {f:#x}: only bit 0 (span mode) is defined, the dispatcher poisons any other bit",
                        who(gid)
                    ))
                }
            }
        }
        let Some(first_span) = lists.values().flatten().next().copied() else {
            return Ok(Vec::new());
        };

        // ---- The span table.
        let Some(arena) = self.spans else {
            return Err(format!(
                "{}: span-mode task, but the graph names no span table (TaskGraph::with_spans)",
                who(first_span)
            ));
        };
        if arena == self.arena || arena == self.events {
            return Err(format!(
                "span table {} is the graph's own arena or events",
                arena.0
            ));
        }
        let table = read_span_table(program, arena)?;
        if let Some(n) = program
            .nodes
            .iter()
            .position(|n| n.effects.iter().any(|e| e.resource == arena && e.writes()))
        {
            return Err(format!("node {n} writes span table {}", arena.0));
        }

        // ---- The pools hosting span tasks: protocol properties and `spans` argument.
        let hosting: BTreeSet<usize> = lists
            .values()
            .flatten()
            .map(|&g| index.pool_of(g))
            .collect();
        let mut tickets: BTreeSet<ResourceId> = BTreeSet::new();
        for pool in &self.pools {
            let node = &program.nodes[pool.node];
            if let Some(fact) = protocol.node(pool.node) {
                if let Some((_, Arg::Resource { resource, .. })) =
                    node.args.iter().find(|(o, _)| *o == fact.ticket_arg_offset)
                {
                    tickets.insert(*resource);
                }
            }
        }
        for &pi in &hosting {
            let pool = &self.pools[pi];
            let node = &program.nodes[pool.node];
            let who_pool = format!("pool {pi} (node {}, {})", pool.node, node.symbol);
            let fact = protocol
                .node(pool.node)
                .ok_or_else(|| format!("{who_pool}: no proven task protocol"))?;
            for prop in [TASK_PROTO_SPAN_ENTRY, TASK_PROTO_REGION_WG_LIVE] {
                if fact.proved.iter().any(|p| p == prop) {
                    continue;
                }
                match pending.as_deref_mut() {
                    Some(set) => {
                        set.insert(format!(
                            "span protocol pending: node {} ({}): {prop} not proven",
                            pool.node, node.symbol
                        ));
                    }
                    None => {
                        return Err(format!(
                            "{who_pool}: span-mode tasks need protocol property {prop:?}, which its certificate does not prove"
                        ))
                    }
                }
            }
            match node
                .args
                .iter()
                .find(|(o, _)| *o == M0_SPANS_ARG_OFFSET)
                .map(|(_, a)| a)
            {
                Some(Arg::Resource { resource, interior: 0 }) if *resource == arena => {}
                other => {
                    return Err(format!(
                        "{who_pool}: spans argument +{M0_SPANS_ARG_OFFSET} is {other:?}, not the span table {} at 0",
                        arena.0
                    ))
                }
            }
        }

        // ---- Each list.
        let mut span_keys: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut seen_res: BTreeMap<ResourceId, (u32, &str)> = BTreeMap::new();
        let mut roles: BTreeMap<ResourceId, (u32, &str, SpanRole)> = BTreeMap::new();
        let mut proofs = Vec::with_capacity(lists.len());
        for (&list_id, gids) in &lists {
            let first = flat[gids[0]];
            let what = format!("span list {list_id}");
            let Some(d) = table.get(list_id as usize) else {
                return Err(format!(
                    "{}: list_id {list_id} is not below n_spans {} of the span table",
                    who(gids[0]),
                    table.len()
                ));
            };
            for &g in gids {
                let t = flat[g];
                let uniform = (
                    t.template,
                    t.ka_off,
                    t.gx,
                    t.gy,
                    t.gz,
                    t.pred_event,
                    t.expected,
                    t.done_event,
                ) == (
                    first.template,
                    first.ka_off,
                    first.gx,
                    first.gy,
                    first.gz,
                    first.pred_event,
                    first.expected,
                    first.done_event,
                );
                if !uniform {
                    return Err(format!(
                        "{}: differs from {} in template, ka_off, virtual grid, pred or done: the tasks of {what} are one launch instance",
                        who(g),
                        who(gids[0])
                    ));
                }
                if t.vcount != 1 {
                    return Err(format!(
                        "{}: vcount {} in span mode: a span task runs exactly one workgroup (vcount 1)",
                        who(g),
                        t.vcount
                    ));
                }
            }
            if let Some(prev) = span_keys.insert((first.template, first.ka_off), list_id) {
                return Err(format!(
                    "{what} and span list {prev} share template {} record @{}: one record is one instance",
                    first.template, first.ka_off
                ));
            }
            let k_total = d.k;
            let grid = u128::from(first.gx) * u128::from(first.gy) * u128::from(first.gz);
            if u128::from(k_total) > grid {
                return Err(format!(
                    "{what}: K = {k_total} exceeds the {grid} workgroups of the ({}, {}, {}) grid of template {}",
                    first.gx, first.gy, first.gz, first.template
                ));
            }
            // The certified dispatcher packs wg_yz = vy | vz << 16 and indexes the
            // claim bitmap with a 32-bit bit = vx + gx*(vy + gy*vz): both are exact
            // only when every in-grid y/z fits 16 bits and the grid fits 2^32.
            if first.gy > 1 << 16 || first.gz > 1 << 16 || grid > 1u128 << 32 {
                return Err(format!(
                    "{what}: grid ({}, {}, {}) exceeds what the dispatcher encodes exactly (y, z <= 65536 for the wg_yz packing, gx*gy*gz <= 2^32 for the claim bitmap index)",
                    first.gx, first.gy, first.gz
                ));
            }
            if gids.len() != k_total as usize {
                return Err(format!(
                    "{what}: the span table says K = {k_total}, but {} tasks carry the list: exactly K tasks with k = 0..K-1 are needed (NOP tasks still release done)",
                    gids.len()
                ));
            }
            let mut seen = vec![false; k_total as usize];
            for &g in gids {
                let k = flat[g].k;
                match seen.get_mut(k as usize) {
                    None => {
                        return Err(format!(
                            "{}: span index k = {k} is not below K = {k_total} of {what}",
                            who(g)
                        ))
                    }
                    Some(s) if *s => {
                        return Err(format!(
                            "{}: span index k = {k} of {what} is carried by two tasks, so some index below K = {k_total} has none",
                            who(g)
                        ))
                    }
                    Some(s) => *s = true,
                }
            }

            // The resources the descriptor names.
            for (role, r) in [
                ("live list", d.live_list),
                ("count cell", d.count),
                ("claim bitmap", d.bitmap),
            ] {
                if program.resources.get(r.0 as usize).is_none() {
                    return Err(format!("{what}: {role} resource {} does not exist", r.0));
                }
                if r == self.events || r == self.arena || r == arena || tickets.contains(&r) {
                    return Err(format!(
                        "{what}: {role} resource {} is a protocol resource (events, arena, span table or ticket)",
                        r.0
                    ));
                }
                if let Some((other, theirs)) = seen_res.insert(r, (list_id, role)) {
                    return Err(format!(
                        "{what}: {role} resource {} is also the {theirs} of span list {other}",
                        r.0
                    ));
                }
            }
            match program.kinds.get(&d.live_list) {
                Some(ResourceKind::LiveList { cap, stride })
                    if u64::from(*stride) == LIVE_ENTRY_BYTES && *cap >= k_total => {}
                other => {
                    return Err(format!(
                        "{what}: live list resource {} is {other:?}, not a LiveList of stride {LIVE_ENTRY_BYTES} with capacity >= K = {k_total}",
                        d.live_list.0
                    ))
                }
            }
            let bitmap_bytes = grid.div_ceil(32) * 4;
            for (role, r, need) in [
                ("count cell", d.count, 4u128),
                ("claim bitmap", d.bitmap, bitmap_bytes),
            ] {
                let have = u128::from(program.resources[r.0 as usize].bytes);
                if !matches!(program.kinds.get(&r), Some(ResourceKind::EventCells { .. }))
                    || have < need
                {
                    return Err(format!(
                        "{what}: {role} resource {} must be declared EventCells and hold {need} bytes (holds {have})",
                        r.0
                    ));
                }
            }

            // The producer and the consumers.
            if first.pred_event == layout.none_event || first.expected != 1 {
                return Err(format!(
                    "{what}: span tasks wait for the producer's done event with expected 1, not event {} expected {}",
                    first.pred_event, first.expected
                ));
            }
            let released = producers
                .get(&first.pred_event)
                .map_or(&[][..], Vec::as_slice);
            let [producer] = released else {
                return Err(format!(
                    "{what}: pred event {} is released by {} tasks: the producer must be its only releaser",
                    first.pred_event,
                    released.len()
                ));
            };
            let producer = *producer;
            if first.done_event != layout.none_event {
                if let Some(waiting) = consumers.get(&first.done_event) {
                    for &c in waiting {
                        if u64::from(flat[c].expected) != u64::from(k_total) {
                            return Err(format!(
                                "{} waits for event {} to reach {}, but {what} releases it K = {k_total} times (NOP tasks release too): expected must equal K exactly",
                                who(c),
                                first.done_event,
                                flat[c].expected
                            ));
                        }
                    }
                    let all = producers.get(&first.done_event).map_or(0, Vec::len);
                    if all != k_total as usize {
                        return Err(format!(
                            "event {} is released by {all} tasks, {what} accounts for K = {k_total}: the span's done event has other releasers",
                            first.done_event
                        ));
                    }
                }
            }

            let nodes: BTreeSet<usize> = self.pools.iter().map(|p| p.node).collect();
            let (pp, pk) = place(producer);
            for (name, r) in [("live list", d.live_list), ("count cell", d.count)] {
                roles.insert(
                    r,
                    (
                        list_id,
                        name,
                        SpanRole::Producer {
                            gid: producer,
                            node: self.pools[pp].node,
                        },
                    ),
                );
            }
            roles.insert(
                d.bitmap,
                (list_id, "claim bitmap", SpanRole::Bitmap { nodes }),
            );
            proofs.push(SpanListProof {
                list_id,
                template: first.template,
                ka_off: first.ka_off,
                k: k_total,
                grid: (first.gx, first.gy, first.gz),
                producer: (pp, pk),
                live_list: d.live_list,
                count: d.count,
                bitmap: d.bitmap,
            });
        }
        // An instance is span or not: no ordinary task shares a span record.
        for (gid, t) in flat.iter().enumerate() {
            if t.flags == 0 {
                if let Some(list) = span_keys.get(&(t.template, t.ka_off)) {
                    return Err(format!(
                        "{}: ordinary task on template {} record @{}, which span list {list} owns",
                        who(gid),
                        t.template,
                        t.ka_off
                    ));
                }
            }
        }

        // ---- Writers: the live list and count only by the producer, the
        // bitmap only by the pool nodes. A pool node's effects through its
        // arena records are the union over every record of the templates it
        // knows (every pool of a megakernel knows the producer's), so for
        // them the per-task pass below is the precise check; what a node
        // reaches through an argument of its own is checked here.
        let pool_node = |n: usize| self.pools.iter().any(|p| p.node == n);
        for (n, node) in program.nodes.iter().enumerate() {
            for e in node.effects.iter().filter(|e| e.writes()) {
                if pool_node(n) && self.through_arena(node, e) {
                    continue;
                }
                let Some((list, name, role)) = roles.get(&e.resource) else {
                    continue;
                };
                let (allowed, who_may) = match role {
                    SpanRole::Producer { node: pn, .. } => (n == *pn, "the producer's node"),
                    SpanRole::Bitmap { nodes } => (nodes.contains(&n), "the pool nodes"),
                };
                if !allowed {
                    return Err(format!(
                        "node {n} ({}) writes the {name} (resource {}) of span list {list}: only {who_may} may",
                        node.symbol, e.resource.0
                    ));
                }
            }
        }
        let mut memo: BTreeMap<(usize, u32, u32), Vec<ResourceId>> = BTreeMap::new();
        for (gid, t) in flat.iter().enumerate() {
            let pi = index.pool_of(gid);
            let written = memo.entry((pi, t.template, t.ka_off)).or_insert_with(|| {
                by_base
                    .get(&u64::from(t.ka_off))
                    .map(|rec| Self::raw_writes(&pool_effects[pi], rec))
                    .unwrap_or_default()
            });
            for r in written.iter() {
                let Some((list, name, role)) = roles.get(r) else {
                    continue;
                };
                if !matches!(role, SpanRole::Producer { gid: p, .. } if *p == gid) {
                    return Err(format!(
                        "{} writes the {name} (resource {}) of span list {list}: only the producer task writes it, the claim bitmap only the dispatcher",
                        who(gid),
                        r.0
                    ));
                }
            }
        }
        Ok(proofs)
    }

    /// The resources a task of `record` may write through its pointer
    /// fields, event cells included (unlike [`TaskGraph::task_effects`]).
    fn raw_writes(source: &PoolEffects, record: &BindingRecord) -> Vec<ResourceId> {
        match source {
            PoolEffects::Conservative => record.fields.iter().map(|f| f.resource).collect(),
            PoolEffects::Templates { loads, .. } => loads
                .get(&record.template)
                .into_iter()
                .flatten()
                .filter(|l| l.classes.iter().any(|c| c.writes()))
                .filter_map(|l| record.fields.iter().find(|f| f.offset == l.record_offset))
                .map(|f| f.resource)
                .collect(),
        }
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
        let mut footprints: BTreeMap<u32, Vec<WgFootprint>> = BTreeMap::new();
        let mut any = false;
        let mut sealed: Vec<ArenaSealedBinding> = Vec::new();
        let mut hidden: BTreeMap<u32, Vec<(u32, String)>> = BTreeMap::new();
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
                    footprints
                        .entry(t.template)
                        .or_default()
                        .extend(t.wg_footprints.iter().cloned());
                    hidden
                        .entry(t.template)
                        .or_default()
                        .extend(t.hidden_grid.iter().cloned());
                }
            }
        }
        if any {
            return Ok(PoolEffects::Templates {
                loads: templates,
                footprints,
                sealed,
                hidden,
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
                footprints: BTreeMap::new(),
                sealed: Vec::new(),
                hidden: BTreeMap::new(),
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
        let mut merged: BTreeMap<ResourceId, (bool, bool, Conc)> = BTreeMap::new();
        let mut add = |resource: ResourceId, read: bool, write: bool, conc: Conc| {
            if matches!(
                program.kinds.get(&resource),
                Some(ResourceKind::EventCells { .. })
            ) {
                return;
            }
            match merged.get_mut(&resource) {
                None => {
                    merged.insert(resource, (read, write, conc));
                }
                Some(e) => {
                    e.0 |= read;
                    e.1 |= write;
                    // Two accesses (fields) to one resource only stay certified
                    // when both are commutative atomics: a footprint speaks
                    // for one pointer, and another field may alias it.
                    if !(e.2 == Conc::Commutative && conc == Conc::Commutative) {
                        e.2 = Conc::None;
                    }
                }
            }
        };
        match source {
            PoolEffects::Conservative => {
                for f in &record.fields {
                    add(f.resource, true, true, Conc::None);
                }
            }
            PoolEffects::Templates {
                loads: map,
                footprints,
                sealed,
                ..
            } => {
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
                    let conc = if classes
                        .iter()
                        .all(|c| matches!(c, MemClass::AtomicCommutative | MemClass::AtomicRelease))
                    {
                        Conc::Commutative
                    } else if classes.iter().any(|c| c.writes())
                        && !classes.iter().any(|c| c.reads())
                    {
                        footprints
                            .get(&record.template)
                            .and_then(|fs| {
                                fs.iter().find(|f| f.record_offset == load.record_offset)
                            })
                            .map_or(Conc::None, |f| Conc::Footprint(f.clone()))
                    } else {
                        Conc::None
                    };
                    add(
                        field.resource,
                        classes.iter().any(|c| c.reads()),
                        classes.iter().any(|c| c.writes()),
                        conc,
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
                    add(b.resource, true, false, Conc::None);
                    for p in &b.pointees {
                        add(*p, true, false, Conc::None);
                    }
                }
            }
        }
        Ok(merged
            .into_iter()
            .map(|(resource, (read, write, conc))| Access {
                resource,
                read,
                write,
                conc,
            })
            .collect())
    }

    /// The host grid tie of a task: every hidden grid argument the template's
    /// body loads (receipt `hidden_grid`) is a scalar the record declares,
    /// and its `block_count_*` equals the task header's `gx/gy/gz` (the grid
    /// the same-instance rule and the workgroup-inside-grid check use). A
    /// pool without receipt facts for the template has nothing to tie.
    fn grid_tie(
        &self,
        source: &PoolEffects,
        record: &BindingRecord,
        t: &TaskHeader,
        what: &str,
    ) -> Result<(), String> {
        let PoolEffects::Templates { hidden, .. } = source else {
            return Ok(());
        };
        let Some(list) = hidden.get(&record.template) else {
            return Ok(());
        };
        for (offset, kind) in list {
            // A 2-byte hidden value (group size, remainder, grid dims) lives in
            // the declared 4-aligned scalar dword that contains it.
            let dword = offset & !3;
            let Some(&(_, value)) = record.scalars.iter().find(|(o, _)| *o == dword) else {
                return Err(format!(
                    "{what}: template {} loads hidden grid argument {kind} at record +{offset}, but record @{} does not declare that offset (its containing dword) as a railgun scalar",
                    record.template, record.base
                ));
            };
            let header = match kind.as_str() {
                "block_count_x" => Some(("x", t.gx)),
                "block_count_y" => Some(("y", t.gy)),
                "block_count_z" => Some(("z", t.gz)),
                _ => None,
            };
            if let Some((axis, g)) = header {
                if value != g {
                    return Err(format!(
                        "{what}: header grid {axis} extent {g} differs from the record's hidden {kind} {value} (+{offset}, record @{}): the body would see another grid than the one the headers were proved over",
                        record.base
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether node effect `e` is reached through the binding-record arena
    /// (an arena pointer load, so covered by the task headers) rather than a
    /// pointer argument of its own. The span table's records are the second
    /// such arena: the dispatcher reaches the live lists, count cells and
    /// claim bitmaps through it, which [`TaskGraph::check_spans`] covers.
    fn through_arena(&self, node: &KernelNode, e: &RegionEffect) -> bool {
        e.resource != self.arena
            && Some(e.resource) != self.spans
            && node.args.iter().any(|(o, a)| {
                *o == e.arg_offset
                    && matches!(a, Arg::Resource { resource, interior: 0 }
                        if *resource == self.arena || Some(*resource) == self.spans)
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

/// Which race pass [`TaskGraph::prove`] runs: the sweep, or (tests only) the
/// all-pairs scan it must agree with on every input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RacePass {
    Sweep,
    #[cfg(test)]
    Pairwise,
}

/// A refusal on one resource: the violating task pair (global indices,
/// `a < b`) and the message.
type Fault = (usize, usize, String);

/// Keep the violating pair the all-pairs scan reaches first (`a` ascending,
/// then `b`).
fn keep_first(best: &mut Option<Fault>, fault: Fault) {
    if best
        .as_ref()
        .map_or(true, |b| (fault.0, fault.1) < (b.0, b.1))
    {
        *best = Some(fault);
    }
}

/// Unordered pairs among `n` items.
fn pairs(n: u64) -> u64 {
    n * n.saturating_sub(1) / 2
}

/// The number of set bits of `a & b`.
fn popcount_and(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| u64::from((x & y).count_ones()))
        .sum()
}

/// Whether `a & b` has a set bit among the bits `lo..hi`.
fn any_and_in(a: &[u64], b: &[u64], lo: usize, hi: usize) -> bool {
    if lo >= hi {
        return false;
    }
    let (first, last) = (lo / 64, (hi - 1) / 64);
    (first..=last).any(|w| {
        let mut x = a[w] & b[w];
        if w == first {
            x &= !0u64 << (lo % 64);
        }
        if w == last {
            x &= !0u64 >> (63 - (hi - 1) % 64);
        }
        x != 0
    })
}

/// The lowest set bit of `words`.
fn lowest_bit(words: &[u64]) -> Option<usize> {
    let i = words.iter().position(|w| *w != 0)?;
    Some(i * 64 + words[i].trailing_zeros() as usize)
}

/// What the race pass proved over every resource.
#[derive(Default)]
struct RaceTotals {
    /// Conflicting task pairs proved ordered.
    ordered_pairs: u64,
    /// `(producer node, consumer node, resource)` of every ordered
    /// conflicting pair across two pools.
    directed: BTreeSet<(usize, usize, ResourceId)>,
}

/// What admits a same-instance conflict, once everything that does not
/// depend on the pair has held.
enum PairRule {
    /// Commutative atomics, or an unproven access of a lab build: every
    /// pair is admitted.
    Admit,
    /// A footprint whose assumptions hold: a pair is admitted iff its
    /// workgroups differ on one of these axes (bit 0 x, 1 y, 2 z).
    Axes(u8),
}

/// The race pass of [`TaskGraph::prove`]: on every resource, every
/// conflicting task pair is ordered by happens-before or, inside one launch
/// instance, admitted by the instance rule
/// ([`Races::same_instance_pair`]); the first refusal is reported. It is not
/// a scan of all pairs: the module docs ("Complexity") give the argument
/// that it agrees with one on every input.
struct Races<'a> {
    graph: &'a TaskGraph,
    flat: &'a [&'a TaskHeader],
    index: &'a Index,
    accesses: &'a [Vec<Access>],
    by_base: &'a BTreeMap<u64, &'a BindingRecord>,
    instances: &'a Instances,
    /// The event index each task waits on.
    pred_ix: &'a [Option<usize>],
    /// `before[e]`: every task that happens-before a consumer of event `e`.
    before: &'a [Option<Bits>],
    /// The position of each task in the order the progress search finished
    /// them: every happens-before edge goes forward in it.
    topo: &'a [usize],
}

impl Races<'_> {
    fn place(&self, gid: usize) -> (usize, usize) {
        let pi = self.index.pool_of(gid);
        (pi, gid - self.index.base[pi])
    }

    fn row(&self, event: usize) -> &Bits {
        self.before[event].as_ref().expect("a used event")
    }

    /// Task `a` happens-before task `b`.
    fn hb(&self, a: usize, b: usize) -> bool {
        self.pred_ix[b].is_some_and(|e| self.row(e).get(a))
    }

    /// Every resource of `by_resource` (accessing tasks in index order,
    /// with whether each writes), in resource order; the first refusal.
    fn check(
        &self,
        by_resource: &BTreeMap<ResourceId, Vec<(usize, bool)>>,
        mut pending: Option<&mut BTreeSet<String>>,
    ) -> Result<RaceTotals, String> {
        let mut totals = RaceTotals::default();
        for (&resource, list) in by_resource {
            self.check_resource(resource, list, pending.as_deref_mut(), &mut totals)?;
        }
        Ok(totals)
    }

    /// One resource. `list` is its accessing tasks in index order.
    ///
    /// Three independent facts are computed, none of them by visiting pairs:
    /// the number of ordered conflicting pairs (popcounts of the
    /// happens-before rows), the verdict of each launch instance, and, only
    /// when the counts show an unordered pair outside every instance, which
    /// pair that is.
    fn check_resource(
        &self,
        resource: ResourceId,
        list: &[(usize, bool)],
        mut pending: Option<&mut BTreeSet<String>>,
        totals: &mut RaceTotals,
    ) -> Result<(), String> {
        let total = self.index.total();
        let mut all = Bits::new(total);
        let mut writers = Bits::new(total);
        let mut readers = 0u64;
        for &(g, w) in list {
            all.set(g);
            if w {
                writers.set(g);
            } else {
                readers += 1;
            }
        }
        // Pairs with a write: a pair of two readers never conflicts.
        let conflicting = pairs(list.len() as u64) - pairs(readers);
        let (ordered, directed) = self.ordered(list, &all, &writers);

        let mut by_instance: BTreeMap<(u32, u32), Vec<(usize, bool)>> = BTreeMap::new();
        for &(g, w) in list {
            if let Some(key) = self.instances.0[g] {
                by_instance.entry(key).or_default().push((g, w));
            }
        }
        let mut best: Option<Fault> = None;
        // Conflicting pairs inside an instance that no happens-before orders.
        let mut inside = 0u64;
        for members in by_instance.values().filter(|m| m.len() >= 2) {
            let m_readers = members.iter().filter(|(_, w)| !w).count() as u64;
            let m_conflicting = pairs(members.len() as u64) - pairs(m_readers);
            let m_ordered = self.ordered_within(members);
            inside += m_conflicting - m_ordered;
            // Nothing to decide: every conflicting pair is ordered, or an
            // earlier instance already refused at a smaller first task.
            if m_conflicting == m_ordered || best.as_ref().is_some_and(|b| b.0 < members[0].0) {
                continue;
            }
            let fault = if m_ordered == 0 && self.uniform(resource, members) {
                self.uniform_instance(resource, members, pending.as_deref_mut())
            } else {
                self.pairwise_instance(resource, members, pending.as_deref_mut())
            };
            if let Some(fault) = fault {
                keep_first(&mut best, fault);
            }
        }
        // Every conflicting pair is ordered in exactly one direction or in
        // none (the graph is acyclic), so `conflicting - ordered` is the
        // number of unordered ones; the rest of them, outside every
        // instance, are races.
        if conflicting - ordered != inside {
            match self.first_race(resource, list, &by_instance) {
                Some(fault) => keep_first(&mut best, fault),
                None => {
                    return Err(format!(
                        "race on resource {}: {} unordered conflicting pairs, {inside} inside launch instances, and no pair found (inconsistent happens-before rows)",
                        resource.0,
                        conflicting - ordered
                    ))
                }
            }
        }
        if let Some((_, _, why)) = best {
            return Err(why);
        }
        totals.ordered_pairs += ordered;
        for (pf, pt) in directed {
            totals.directed.insert((
                self.graph.pools[pf].node,
                self.graph.pools[pt].node,
                resource,
            ));
        }
        Ok(())
    }

    /// The conflicting pairs `(a, b)` of `list` with `a` happens-before `b`,
    /// and the pools `(producer, consumer)` they run in across two pools.
    ///
    /// `a` happens-before `b` iff `a` is in `before` of `b`'s event, and a
    /// pair conflicts iff `b` writes (`a` any accessor) or `a` writes (`b`
    /// any accessor): so for every `b` the ordered pairs ending in it are
    /// `|before ∩ (all if b writes else writers)|`, the same for every task
    /// of one `(event, writes)`.
    fn ordered(
        &self,
        list: &[(usize, bool)],
        all: &Bits,
        writers: &Bits,
    ) -> (u64, BTreeSet<(usize, usize)>) {
        let mut groups: BTreeMap<(usize, bool), (u64, BTreeSet<usize>)> = BTreeMap::new();
        for &(g, w) in list {
            if let Some(e) = self.pred_ix[g] {
                let entry = groups.entry((e, w)).or_default();
                entry.0 += 1;
                entry.1.insert(self.index.pool_of(g));
            }
        }
        let mut ordered = 0u64;
        let mut directed = BTreeSet::new();
        for (&(e, w), (count, consumers)) in &groups {
            let row = &self.row(e).0;
            let mask = &(if w { all } else { writers }).0;
            let hits = popcount_and(row, mask);
            if hits == 0 {
                continue;
            }
            ordered += count * hits;
            for pf in 0..self.graph.pools.len() {
                if any_and_in(row, mask, self.index.base[pf], self.index.base[pf + 1]) {
                    for &pt in consumers {
                        if pt != pf {
                            directed.insert((pf, pt));
                        }
                    }
                }
            }
        }
        (ordered, directed)
    }

    /// [`Races::ordered`] over the pairs of one instance's members.
    fn ordered_within(&self, members: &[(usize, bool)]) -> u64 {
        let mut events: Vec<(usize, bool)> = members
            .iter()
            .filter_map(|&(g, w)| self.pred_ix[g].map(|e| (e, w)))
            .collect();
        if events.is_empty() {
            return 0;
        }
        let words = self.index.total().div_ceil(64);
        let (mut all, mut writers) = (vec![0u64; words], vec![0u64; words]);
        for &(g, w) in members {
            all[g / 64] |= 1 << (g % 64);
            if w {
                writers[g / 64] |= 1 << (g % 64);
            }
        }
        events.sort_unstable();
        events
            .chunk_by(|a, b| a == b)
            .map(|run| {
                let (e, w) = run[0];
                let mask = if w { &all } else { &writers };
                run.len() as u64 * popcount_and(&self.row(e).0, mask)
            })
            .sum()
    }

    /// Every member accesses the resource the same way: the same write flag
    /// and (members of one pool share their accesses) the same certified
    /// concurrency, so one pair's rule is every pair's.
    fn uniform(&self, resource: ResourceId, members: &[(usize, bool)]) -> bool {
        let conc = |g: usize| {
            self.accesses[g]
                .iter()
                .find(|x| x.resource == resource)
                .map(|x| &x.conc)
        };
        let (first, writes) = members[0];
        let pool = self.index.pool_of(first);
        let first_conc = conc(first);
        members
            .iter()
            .all(|&(g, w)| w == writes && (self.index.pool_of(g) == pool || conc(g) == first_conc))
    }

    /// The first refusal inside one instance whose members are
    /// [`Races::uniform`] writers and have no happens-before among them:
    /// every pair is conflicting and unordered, so the rule of the first
    /// pair is the rule of all, and a footprint separates them by
    /// coordinates alone ([`Races::first_agreeing`]).
    fn uniform_instance(
        &self,
        resource: ResourceId,
        members: &[(usize, bool)],
        pending: Option<&mut BTreeSet<String>>,
    ) -> Option<Fault> {
        let (a, b) = (members[0].0, members[1].0);
        let axes = match self.pair_rule(resource, a, b, pending) {
            Err(why) => return Some((a, b, why)),
            Ok(PairRule::Admit) => return None,
            Ok(PairRule::Axes(axes)) => axes,
        };
        let (x, y) = self.first_agreeing(members, axes)?;
        let why = self.separation(resource, x, y, axes).err().unwrap_or_else(|| {
            format!(
                "race on resource {}: internal: tasks {x} and {y} agree on every footprint axis but the pair is separated",
                resource.0
            )
        });
        Some((x, y, why))
    }

    /// The lowest pair `(a, b)`, `a < b`, of `members` (in index order) no
    /// footprint axis in `axes` separates: equal `vy` if `y` is an axis,
    /// equal `vz` if `z` is, overlapping x-intervals if `x` is. Tiles of one
    /// instance on equal `(vy, vz)` have disjoint x-intervals, so they
    /// never overlap.
    fn first_agreeing(&self, members: &[(usize, bool)], axes: u8) -> Option<(usize, usize)> {
        let key = |g: usize| {
            let t = self.flat[g];
            (
                if axes & 2 != 0 { t.vy } else { 0 },
                if axes & 4 != 0 { t.vz } else { 0 },
            )
        };
        let mut order: Vec<((u32, u32), usize)> =
            members.iter().map(|&(g, _)| (key(g), g)).collect();
        order.sort_unstable();
        let mut best: Option<(usize, usize)> = None;
        for group in order.chunk_by(|a, b| a.0 == b.0) {
            let gids: Vec<usize> = group.iter().map(|&(_, g)| g).collect();
            let found = match gids.len() {
                0 | 1 => None,
                _ if axes & 1 == 0 => Some((gids[0], gids[1])),
                _ => self.first_x_overlap(&gids),
            };
            if let Some(pair) = found {
                if best.map_or(true, |b| pair < b) {
                    best = Some(pair);
                }
            }
        }
        best
    }

    /// The lowest pair `(a, b)`, `a < b`, of `gids` (ascending) whose
    /// x-intervals `[vx, vx + max(vcount, 1))` overlap: the lowest `a` that
    /// overlaps a later task, found by inserting the tasks from the last in
    /// a set of disjoint merged intervals, then its lowest later partner.
    fn first_x_overlap(&self, gids: &[usize]) -> Option<(usize, usize)> {
        let span = |g: usize| {
            let t = self.flat[g];
            let start = u64::from(t.vx);
            (start, start + u64::from(t.vcount.max(1)))
        };
        // start -> end of disjoint, non-touching segments of the union of the
        // x-intervals of the tasks after the one being inserted.
        let mut union: BTreeMap<u64, u64> = BTreeMap::new();
        let mut first = None;
        for &g in gids.iter().rev() {
            let (start, end) = span(g);
            if union
                .range(..end)
                .next_back()
                .is_some_and(|(_, &seg_end)| seg_end > start)
            {
                first = Some(g);
            }
            let (mut lo, mut hi) = (start, end);
            if let Some((&seg_start, &seg_end)) = union.range(..=lo).next_back() {
                if seg_end >= lo {
                    lo = seg_start;
                    hi = hi.max(seg_end);
                    union.remove(&seg_start);
                }
            }
            while let Some((&seg_start, &seg_end)) = union.range(lo..=hi).next() {
                hi = hi.max(seg_end);
                union.remove(&seg_start);
            }
            union.insert(lo, hi);
        }
        let a = first?;
        let (a0, a1) = span(a);
        let b = gids.iter().copied().find(|&g| {
            let (b0, b1) = span(g);
            g > a && a0 < b1 && b0 < a1
        })?;
        Some((a, b))
    }

    /// The first refusal among the pairs of one instance's members: the
    /// all-pairs scan, for instances whose members are not uniform or are
    /// ordered among themselves (neither happens in one eager launch).
    fn pairwise_instance(
        &self,
        resource: ResourceId,
        members: &[(usize, bool)],
        mut pending: Option<&mut BTreeSet<String>>,
    ) -> Option<Fault> {
        for (i, &(a, a_writes)) in members.iter().enumerate() {
            for &(b, b_writes) in &members[i + 1..] {
                if !(a_writes || b_writes) || self.hb(a, b) || self.hb(b, a) {
                    continue;
                }
                if let Err(why) = self.same_instance_pair(resource, a, b, pending.as_deref_mut()) {
                    return Some((a, b, why));
                }
            }
        }
        None
    }

    /// The lowest unordered conflicting pair of tasks of `list` that are not
    /// of one instance.
    ///
    /// Process the tasks in the order the progress search finished them.
    /// Every happens-before edge goes forward in it, so a pair `(y, x)` with
    /// `y` earlier is unordered iff `y` is not in `before` of `x`'s event.
    /// For each task `x` the earlier tasks that conflict with it and are
    /// outside `x`'s instance form a bitset; its lowest bit `y` gives the
    /// lowest pair this `x` is in (`(y, x)` if `y < x`, else `(x, y)`, as
    /// all its bits are above `x`). The lowest over all `x` is the lowest
    /// pair.
    fn first_race(
        &self,
        resource: ResourceId,
        list: &[(usize, bool)],
        by_instance: &BTreeMap<(u32, u32), Vec<(usize, bool)>>,
    ) -> Option<Fault> {
        let words = self.index.total().div_ceil(64);
        let mut order = list.to_vec();
        order.sort_unstable_by_key(|&(g, _)| self.topo[g]);
        let (mut seen, mut seen_writers) = (vec![0u64; words], vec![0u64; words]);
        let mut unordered = vec![0u64; words];
        // Dense masks of the instances larger than a mask.
        let mut dense: BTreeMap<(u32, u32), Vec<u64>> = BTreeMap::new();
        let mut best: Option<(usize, usize)> = None;
        for &(x, writes) in &order {
            let conflicting = if writes { &seen } else { &seen_writers };
            match self.pred_ix[x] {
                Some(e) => {
                    for ((u, c), r) in unordered.iter_mut().zip(conflicting).zip(&self.row(e).0) {
                        *u = c & !r;
                    }
                }
                None => unordered.copy_from_slice(conflicting),
            }
            if let Some(key) = self.instances.0[x] {
                let members = &by_instance[&key];
                if members.len() > words {
                    let mask = dense.entry(key).or_insert_with(|| {
                        let mut mask = vec![0u64; words];
                        for &(g, _) in members {
                            mask[g / 64] |= 1 << (g % 64);
                        }
                        mask
                    });
                    for (u, m) in unordered.iter_mut().zip(mask.iter()) {
                        *u &= !m;
                    }
                } else {
                    for &(g, _) in members {
                        unordered[g / 64] &= !(1u64 << (g % 64));
                    }
                }
            }
            if let Some(y) = lowest_bit(&unordered) {
                let pair = (x.min(y), x.max(y));
                if best.map_or(true, |b| pair < b) {
                    best = Some(pair);
                }
            }
            seen[x / 64] |= 1 << (x % 64);
            if writes {
                seen_writers[x / 64] |= 1 << (x % 64);
            }
        }
        let (a, b) = best?;
        let writes = |g: usize| {
            list.binary_search_by_key(&g, |p| p.0)
                .map(|i| list[i].1)
                .expect("a task of the list")
        };
        Some((
            a,
            b,
            self.race_message(resource, a, b, writes(a), writes(b)),
        ))
    }

    /// The refusal of an unordered conflicting pair of different instances.
    fn race_message(
        &self,
        resource: ResourceId,
        a: usize,
        b: usize,
        a_writes: bool,
        b_writes: bool,
    ) -> String {
        let (pa, ka) = self.place(a);
        let (pb, kb) = self.place(b);
        format!(
            "race on resource {}: task {ka} of pool {pa} (node {}) and task {kb} of pool {pb} (node {}) {} it and neither happens-before the other",
            resource.0,
            self.graph.pools[pa].node,
            self.graph.pools[pb].node,
            match (a_writes, b_writes) {
                (true, true) => "both write",
                (true, false) => "write/read",
                _ => "read/write",
            },
        )
    }

    /// The reference semantics of the race pass, kept as the oracle the
    /// sweep is tested against: every conflicting pair of every resource in
    /// index order, `O(n^2)` per resource.
    #[cfg(test)]
    fn check_pairwise(
        &self,
        by_resource: &BTreeMap<ResourceId, Vec<(usize, bool)>>,
        mut pending: Option<&mut BTreeSet<String>>,
    ) -> Result<RaceTotals, String> {
        let p = self.graph.pools.len();
        let mut totals = RaceTotals::default();
        for (&resource, list) in by_resource {
            let mut dir = vec![false; p * p];
            for (i, &(a, a_writes)) in list.iter().enumerate() {
                for &(b, b_writes) in &list[i + 1..] {
                    if !(a_writes || b_writes) {
                        continue;
                    }
                    let (from, to) = if self.hb(a, b) {
                        (a, b)
                    } else if self.hb(b, a) {
                        (b, a)
                    } else if self.instances.same(a, b) {
                        self.same_instance_pair(resource, a, b, pending.as_deref_mut())?;
                        continue;
                    } else {
                        return Err(self.race_message(resource, a, b, a_writes, b_writes));
                    };
                    totals.ordered_pairs += 1;
                    let (pf, pt) = (self.index.pool_of(from), self.index.pool_of(to));
                    if pf != pt {
                        dir[pf * p + pt] = true;
                    }
                }
            }
            for pf in 0..p {
                for pt in 0..p {
                    if dir[pf * p + pt] {
                        totals.directed.insert((
                            self.graph.pools[pf].node,
                            self.graph.pools[pt].node,
                            resource,
                        ));
                    }
                }
            }
        }
        Ok(totals)
    }

    /// `"task k of pool p (node n) and task k' of pool p' (node n')"`.
    fn pair_label(&self, a: usize, b: usize) -> String {
        let (pa, ka) = self.place(a);
        let (pb, kb) = self.place(b);
        format!(
            "task {ka} of pool {pa} (node {}) and task {kb} of pool {pb} (node {})",
            self.graph.pools[pa].node, self.graph.pools[pb].node
        )
    }

    /// A conflicting pair of tasks `a < b` of one launch instance on
    /// `resource` (see [`Instances`]): admitted only when the template's own
    /// certificate says concurrent workgroups of its grid do not conflict
    /// there. A commutative-atomic resource always; a write-only field with a
    /// `WgDisjoint` footprint iff every workgroup of `a` and every workgroup
    /// of `b` differ on some footprint axis; otherwise refused.
    fn same_instance_pair(
        &self,
        resource: ResourceId,
        a: usize,
        b: usize,
        pending: Option<&mut BTreeSet<String>>,
    ) -> Result<(), String> {
        match self.pair_rule(resource, a, b, pending)? {
            PairRule::Admit => Ok(()),
            PairRule::Axes(axes) => self.separation(resource, a, b, axes),
        }
    }

    /// The part of [`Races::same_instance_pair`] that does not depend on
    /// which two tasks of the instance (they share template, record and
    /// virtual grid): the rule `a`'s access carries, and for a footprint the
    /// checks of its assumptions against the headers and the record.
    fn pair_rule(
        &self,
        resource: ResourceId,
        a: usize,
        b: usize,
        pending: Option<&mut BTreeSet<String>>,
    ) -> Result<PairRule, String> {
        let (ta, tb) = (self.flat[a], self.flat[b]);
        let fp = match self.accesses[a]
            .iter()
            .find(|x| x.resource == resource)
            .map(|x| &x.conc)
        {
            Some(Conc::Commutative) => return Ok(PairRule::Admit),
            Some(Conc::Footprint(fp)) => fp,
            Some(Conc::None) | None => {
                if let Some(set) = pending {
                    set.insert(format!(
                        "inter-WG disjointness unproven: template {}, resource {}",
                        ta.template, resource.0
                    ));
                    return Ok(PairRule::Admit);
                }
                return Err(format!(
                    "race on resource {}: {} are workgroups of one launch instance of template {} (ka_off {}) and conflict on it; standalone receipt proves no inter-WG write disjointness (not all commutative atomics, and no write-only per-workgroup footprint)",
                    resource.0,
                    self.pair_label(a, b),
                    ta.template,
                    ta.ka_off
                ));
            }
        };
        // The proof's assumptions, against the headers and the record's
        // bound scalars.
        for (i, name) in ["x", "y", "z"].iter().enumerate() {
            let Some(bound) = fp.bounds[i] else { continue };
            for t in [ta, tb] {
                let extent = [t.gx, t.gy, t.gz][i];
                if u64::from(extent) > bound {
                    return Err(format!(
                        "{} (template {}, ka_off {}): virtual grid extent {extent} on {name} exceeds the footprint's certified tile-id bound {bound} on resource {}",
                        self.pair_label(a, b),
                        ta.template,
                        ta.ka_off,
                        resource.0
                    ));
                }
            }
        }
        for req in &fp.requires {
            let got = self
                .by_base
                .get(&u64::from(ta.ka_off))
                .and_then(|r| r.scalars.iter().find(|(o, _)| *o == req.offset))
                .map(|(_, v)| *v);
            if !got.is_some_and(|v| req.holds(v, [ta.gx, ta.gy, ta.gz])) {
                let (offset, mask, value) = (req.offset, req.mask, req.value);
                let rel = match req.op {
                    RequireOp::MaskEq => format!("& {mask:#x} == {value:#x}"),
                    RequireOp::LeU => format!("<= {value:#x}"),
                    RequireOp::GeU => format!(">= {value:#x}"),
                    RequireOp::GridLe => format!(
                        ">= {value} * grid extent {} (virtual grid {}, {}, {})",
                        mask, ta.gx, ta.gy, ta.gz
                    ),
                };
                return Err(format!(
                    "{} (template {}, ka_off {}): the footprint on resource {} requires record dword +{offset} {rel}, the record's scalar is {got:?}",
                    self.pair_label(a, b),
                    ta.template,
                    ta.ka_off,
                    resource.0
                ));
            }
        }
        if ta.flags == TASK_FLAG_SPAN {
            return self.span_rule(resource, a, fp.axes, pending);
        }
        Ok(PairRule::Axes(fp.axes))
    }

    /// A footprint over `axes` for a pair of span tasks. The live
    /// coordinates are unknown on the host (the dispatcher's claim bitmap
    /// only keeps them distinct), so the footprint must separate every pair
    /// of distinct workgroups of the grid: its axes cover every axis the grid
    /// has extent > 1 on. Then any pair is admitted; otherwise refused, or
    /// (lab) a named pending reason.
    fn span_rule(
        &self,
        resource: ResourceId,
        a: usize,
        axes: u8,
        pending: Option<&mut BTreeSet<String>>,
    ) -> Result<PairRule, String> {
        let t = self.flat[a];
        let required = u8::from(t.gx > 1) | (u8::from(t.gy > 1) << 1) | (u8::from(t.gz > 1) << 2);
        if axes & required == required {
            return Ok(PairRule::Admit);
        }
        if let Some(set) = pending {
            set.insert(format!(
                "inter-WG disjointness unproven: template {}, resource {}",
                t.template, resource.0
            ));
            return Ok(PairRule::Admit);
        }
        Err(format!(
            "race on resource {}: inter-WG disjointness unproven: template {}, resource {}: the live workgroups of span list {} are unknown on the host, so the footprint (axes mask {axes:#05b}) must separate every pair of workgroups of the ({}, {}, {}) grid, which needs axes mask {required:#05b}",
            resource.0, t.template, resource.0, t.list_id, t.gx, t.gy, t.gz
        ))
    }

    /// A footprint over `axes` admits the pair `a < b` iff it differs on an
    /// axis: `vy` / `vz` unequal, or (x) disjoint x-intervals.
    fn separation(&self, resource: ResourceId, a: usize, b: usize, axes: u8) -> Result<(), String> {
        let (ta, tb) = (self.flat[a], self.flat[b]);
        let (ax0, ax1) = (
            u64::from(ta.vx),
            u64::from(ta.vx) + u64::from(ta.vcount.max(1)),
        );
        let (bx0, bx1) = (
            u64::from(tb.vx),
            u64::from(tb.vx) + u64::from(tb.vcount.max(1)),
        );
        let x_overlap = ax0 < bx1 && bx0 < ax1;
        let differs = (axes & 2 != 0 && ta.vy != tb.vy)
            || (axes & 4 != 0 && ta.vz != tb.vz)
            || (axes & 1 != 0 && !x_overlap);
        if differs {
            return Ok(());
        }
        let common = ax0.max(bx0);
        let (wx, wx2) = if axes & 1 != 0 {
            (common, common)
        } else {
            (ax0, bx0)
        };
        let tasks = self.pair_label(a, b);
        Err(format!(
            "race on resource {}: {tasks} (template {}, ka_off {}) run workgroups ({wx}, {}, {}) and ({wx2}, {}, {}) concurrently, which agree on every footprint axis (mask {axes:#05b}); the certified footprint only separates workgroups differing on those axes",
            resource.0, ta.template, ta.ka_off, ta.vy, ta.vz, tb.vy, tb.vz
        ))
    }
}

#[cfg(test)]
mod tests;
