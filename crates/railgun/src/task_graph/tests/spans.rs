// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Span mode (GPU-generated routed task spans, `DYNAMIC.md`): the host side
//! of the proof. Pool 0 runs the producer (`T_PROD`, event 0); pool 1 runs
//! the `K` span tasks of list 0 (wait for event 0, release event 1) and a
//! reader of their output (waits for event 1 to reach `K`).

use super::*;
use crate::admission::{Admission, CompletenessEvidence, COMPLETENESS_FUNCTIONAL};
use crate::kernel::{ArenaPointerLoad, FactStatus};
use railgun_corpus::receipt::{TASK_PROTO_REGION_WG_LIVE, TASK_PROTO_SPAN_ENTRY};
use std::time::Instant;

const LIVE: u64 = 0x7700_0000;
const COUNT: u64 = 0x7800_0000;
const BITMAP: u64 = 0x7900_0000;
const SPANS: u64 = 0x7A00_0000;
/// A second list's resources.
const LIVE2: u64 = 0x7B00_0000;
const COUNT2: u64 = 0x7C00_0000;
const BITMAP2: u64 = 0x7D00_0000;

fn span_protocol() -> TaskProtocolFact {
    let mut p = region_wg_protocol();
    p.proved.push(TASK_PROTO_SPAN_ENTRY.to_owned());
    p.proved.push(TASK_PROTO_REGION_WG_LIVE.to_owned());
    p
}

/// The pool kernel's facts plus the arena at argument 64 (the span table):
/// descriptors load the live list, the count cell (plain loads) and the
/// claim bitmap (a commutative atomic).
fn span_facts(known: &[u32], protocol: Option<TaskProtocolFact>) -> Arc<NodeFacts> {
    let mut f = (*conditional_facts(known, protocol)).clone();
    f.facts.args.push(ptr(64, &[SmemLoad]));
    f.facts.explicit_bytes = 72;
    f.facts.segment_bytes = 80;
    let FactStatus::Conditional(c) = &mut f.facts.status else {
        unreachable!("conditional facts")
    };
    let load = |record_offset, class| ArenaPointerLoad {
        record_offset,
        classes: vec![class],
    };
    let template = |template, pointer_loads| ArenaTemplate {
        template,
        pointer_loads,
        sealed: Vec::new(),
        wg_footprints: Vec::new(),
        hidden_grid: Vec::new(),
    };
    c.arenas.push(ArenaFact {
        arena_arg_offset: 64,
        templates: vec![
            template(
                SPAN_DESC_TEMPLATE,
                vec![
                    load(0, VmemLoad),
                    load(8, VmemLoad),
                    load(16, AtomicCommutative),
                ],
            ),
            template(SPAN_HEADER_TEMPLATE, Vec::new()),
        ],
    });
    Arc::new(f)
}

fn span_pool_launch(pool: usize, n_tasks: u32, facts: Arc<NodeFacts>) -> LaunchRecord {
    let mut args = pool_args();
    args.push(ptr(64, &[SmemLoad]));
    let mut l = launch(
        &[
            (0, TASKS + pool as u64 * 0x1_0000),
            (8, CTL + 8 * pool as u64),
            (16, EVENTS),
            (24, ARENA),
            (64, SPANS),
        ],
        72,
        args,
    );
    l.kernarg[56..60].copy_from_slice(&n_tasks.to_le_bytes());
    l.facts = Ok(facts);
    l
}

/// A kernel that only reads the given resources: makes them program
/// resources.
fn reads(bases: &[u64]) -> LaunchRecord {
    let pointers: Vec<(u32, u64)> = bases
        .iter()
        .enumerate()
        .map(|(i, &b)| (8 * i as u32, b))
        .collect();
    let args = pointers.iter().map(|&(o, _)| ptr(o, &[VmemLoad])).collect();
    launch(&pointers, 8 * bases.len() as u32, args)
}

fn writes(base: u64, class: crate::kernel::MemClass) -> LaunchRecord {
    launch(&[(0, base)], 8, vec![ptr(0, &[class])])
}

struct Spec {
    pools: Vec<Vec<TaskHeader>>,
    records: Vec<(u32, Vec<(u32, u64)>)>,
    known: Vec<Vec<u32>>,
    protocol: Option<TaskProtocolFact>,
    /// `(live, count, bitmap, K)` per span list.
    descs: Vec<(u64, u64, u64, u32)>,
    extra: Vec<LaunchRecord>,
}

impl Spec {
    /// List 0 of `k` tasks over a `(gx, gy, 1)` grid; span tasks write X
    /// (footprint over x and y), the reader reads X.
    fn std(k: u32, gx: u32, gy: u32) -> Spec {
        let mut span: Vec<TaskHeader> = (0..k)
            .map(|i| TaskHeader {
                flags: TASK_FLAG_SPAN,
                k: i,
                list_id: 0,
                vcount: 1,
                gx,
                gy,
                pred_event: 0,
                expected: 1,
                done_event: 1,
                ..t(T_TILE_XY, 1, 0, 0, 0)
            })
            .collect();
        span.push(t(T_READ, 2, 1, k, N));
        Spec {
            pools: vec![vec![t(T_PROD, 0, N, 0, 0)], span],
            records: vec![
                (T_PROD, vec![(0, LIVE), (8, COUNT)]),
                (T_TILE_XY, vec![(8, X)]),
                (T_READ, vec![(0, X)]),
            ],
            known: vec![vec![T_PROD], vec![T_TILE_XY, T_READ]],
            protocol: Some(span_protocol()),
            descs: vec![(LIVE, COUNT, BITMAP, k)],
            extra: vec![reads(&[LIVE, COUNT, BITMAP])],
        }
    }

    /// The span tasks (and their record) run `template` instead.
    fn template(mut self, template: u32) -> Spec {
        for task in self.pools[1].iter_mut().filter(|t| t.flags != 0) {
            task.template = template;
        }
        self.records[1].0 = template;
        self.known[1] = vec![template, T_READ];
        self
    }

    fn build(self) -> Result<Fixture, String> {
        let mut launches = vec![launch(
            &[(0, X), (8, Y), (16, Z)],
            24,
            vec![
                ptr(0, &[VmemLoad]),
                ptr(8, &[VmemLoad]),
                ptr(16, &[VmemLoad]),
            ],
        )];
        for (p, tasks) in self.pools.iter().enumerate() {
            let f = span_facts(&self.known[p], self.protocol.clone());
            launches.push(span_pool_launch(p, tasks.len() as u32, f));
        }
        launches.extend(self.extra);
        let mut program = author("gfx1201", &launches);
        let big = 1u64 << 24;
        let grow = |program: &mut KernelProgram, base: u64| {
            let r = rid(program, base);
            program.resources[r.0 as usize].bytes = big;
            r
        };
        let cells = (big / EVENT_CELL_BYTES) as u32;
        let events = grow(&mut program, EVENTS);
        let ctl = grow(&mut program, CTL);
        program.declare_event_cells(events, cells)?;
        program.declare_event_cells(ctl, cells)?;
        for p in 0..self.pools.len() {
            let r = grow(&mut program, TASKS + p as u64 * 0x1_0000);
            program.declare_task_arena(r)?;
        }
        for &(live, count, bitmap, k) in &self.descs {
            let l = grow(&mut program, live);
            if !program.kinds.contains_key(&l) {
                program.declare_live_list(l, k.max(1))?;
            }
            for base in [count, bitmap] {
                let r = grow(&mut program, base);
                if !program.kinds.contains_key(&r) {
                    program.declare_event_cells(r, cells)?;
                }
            }
        }
        let arena = rid(&program, ARENA);
        let records: Vec<BindingRecord> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, (template, fields))| BindingRecord {
                base: 64 * i as u64,
                template: *template,
                fields: fields
                    .iter()
                    .map(|&(offset, base)| BindingField {
                        offset,
                        resource: rid(&program, base),
                        interior: 0,
                    })
                    .collect(),
                scalars: Vec::new(),
            })
            .collect();
        let sealed = SealedTableRegistry::new();
        program.bind_records(arena, records, &sealed)?;
        let spans = rid(&program, SPANS);
        let descs: Vec<SpanDescriptor> = self
            .descs
            .iter()
            .map(|&(l, c, b, k)| SpanDescriptor {
                live_list: rid(&program, l),
                count: rid(&program, c),
                bitmap: rid(&program, b),
                k,
            })
            .collect();
        program.bind_records(spans, span_table_records(&descs)?, &sealed)?;
        let group = 1..1 + self.pools.len();
        let graph = TaskGraph::new(
            self.pools
                .into_iter()
                .enumerate()
                .map(|(p, tasks)| PoolTasks { node: 1 + p, tasks })
                .collect(),
            events,
            20_000,
            arena,
        )
        .with_spans(spans);
        Ok(Fixture {
            program,
            group,
            graph,
        })
    }
}

fn refused(spec: Spec) -> String {
    spec.build().unwrap().check().unwrap_err()
}

#[test]
fn a_valid_span_is_admitted_with_its_properties_and_counts_in_the_ordering() {
    let fx = Spec::std(12, 4, 3).build().unwrap();
    let proof = fx.check().unwrap();
    assert_eq!(proof.tasks, 1 + 12 + 1);
    let [s] = &proof.spans[..] else {
        panic!("one span list: {:?}", proof.spans)
    };
    assert_eq!(
        (s.list_id, s.template, s.ka_off, s.k, s.grid, s.producer),
        (0, T_TILE_XY, 64, 12, (4, 3, 1), (0, 0))
    );
    assert_eq!(
        (s.live_list, s.count, s.bitmap),
        (fx.data(LIVE), fx.data(COUNT), fx.data(BITMAP))
    );
    assert!(proof
        .obligations
        .iter()
        .any(|o| o.contains("claim bitmaps") && o.contains("zero at every submit")));
    assert!(proof
        .obligations
        .iter()
        .any(|o| o.contains("span completeness is not certified")));
    // The all-pairs oracle agrees on the whole proof.
    let oracle = fx
        .graph
        .check_pairwise(&fx.program, fx.group.clone(), &fx.protocol())
        .unwrap();
    assert_eq!(proof, oracle);
    // The group is admitted and its admission says completeness is open.
    let ordering = proof.ordering();
    assert_eq!(ordering.spans(), 1);
    let mut program = fx.program.clone();
    program.add_coresident_group(1..3, ordering).unwrap();
    assert_eq!(program.admission().spans(), 1);
    let plan = plan(
        &program,
        &CacheTable::for_arch("gfx1201").unwrap(),
        RowPolicy::TierD,
    );
    assert!(
        plan.groups[0].refusals.is_empty(),
        "{:?}",
        plan.groups[0].refusals
    );
    assert_eq!(plan.groups[0].spans, 1);
    assert_eq!(plan.admission().spans(), 1);
    assert_eq!(plan.groups[0].admission().spans(), 1);
}

#[test]
fn every_span_pool_needs_both_protocol_properties() {
    for missing in [TASK_PROTO_SPAN_ENTRY, TASK_PROTO_REGION_WG_LIVE] {
        let mut p = span_protocol();
        p.proved.retain(|q| q != missing);
        let mut spec = Spec::std(4, 2, 2);
        spec.protocol = Some(p);
        let err = refused(spec);
        assert!(
            err.contains("span-mode tasks need protocol property") && err.contains(missing),
            "{err}"
        );
    }
    let mut spec = Spec::std(4, 2, 2);
    spec.protocol = Some(region_wg_protocol());
    let err = refused(spec);
    assert!(
        err.contains("span-mode tasks need protocol property"),
        "{err}"
    );
}

#[test]
fn header_flags_with_any_other_bit_are_refused() {
    for flags in [3, 2, 0x8000_0001] {
        let mut spec = Spec::std(4, 2, 2);
        spec.pools[1][0].flags = flags;
        let err = refused(spec);
        assert!(
            err.contains(&format!("header flags {flags:#x}")) && err.contains("poisons"),
            "{err}"
        );
    }
    // Also on an ordinary task.
    let mut spec = Spec::std(4, 2, 2);
    spec.pools[1][4].flags = 4;
    assert!(refused(spec).contains("header flags 0x4"));
}

#[test]
fn span_indices_are_exactly_zero_to_k_minus_one_once() {
    let mut dup = Spec::std(4, 2, 2);
    dup.pools[1][1].k = 0;
    let err = refused(dup);
    assert!(
        err.contains("span index k = 0") && err.contains("two tasks"),
        "{err}"
    );
    let mut high = Spec::std(4, 2, 2);
    high.pools[1][3].k = 4;
    assert!(refused(high).contains("not below K = 4"));
    let mut missing = Spec::std(4, 2, 2);
    missing.pools[1].remove(2);
    let err = refused(missing);
    assert!(
        err.contains("exactly K tasks") && err.contains("K = 4"),
        "{err}"
    );
    // Too many tasks for K.
    let mut extra = Spec::std(4, 2, 2);
    extra.descs[0].3 = 3;
    assert!(refused(extra).contains("exactly K tasks"));
}

#[test]
fn k_may_not_exceed_the_grid() {
    let err = refused(Spec::std(5, 2, 2));
    assert!(err.contains("K = 5 exceeds the 4 workgroups"), "{err}");
    // K equal to the grid is fine.
    Spec::std(4, 2, 2).build().unwrap().check().unwrap();
}

#[test]
fn list_headers_must_agree_and_name_a_table_entry() {
    let mut list = Spec::std(4, 2, 2);
    for t in list.pools[1].iter_mut().filter(|t| t.flags != 0) {
        t.list_id = 1;
    }
    assert!(refused(list).contains("is not below n_spans 1"));
    let mut vcount = Spec::std(4, 2, 2);
    vcount.pools[1][0].vcount = 2;
    assert!(refused(vcount).contains("vcount 2 in span mode"));
    let mut grid = Spec::std(4, 2, 2);
    grid.pools[1][2].gx = 3;
    assert!(refused(grid).contains("one launch instance"));
    let mut done = Spec::std(4, 2, 2);
    done.pools[1][2].done_event = 7;
    assert!(refused(done).contains("one launch instance"));
    let mut pred = Spec::std(4, 2, 2);
    for t in pred.pools[1].iter_mut().filter(|t| t.flags != 0) {
        t.expected = 2;
    }
    let err = refused(pred);
    assert!(err.contains("expected 1, not event 0 expected 2"), "{err}");
    let mut nopred = Spec::std(4, 2, 2);
    for t in nopred.pools[1].iter_mut().filter(|t| t.flags != 0) {
        t.pred_event = N;
        t.expected = 0;
    }
    assert!(refused(nopred).contains("expected 1"));
}

#[test]
fn the_done_event_expects_exactly_k_and_the_producer_is_the_only_releaser() {
    for wrong in [3, 5] {
        let mut spec = Spec::std(4, 2, 2);
        spec.pools[1][4].expected = wrong;
        let err = refused(spec);
        assert!(
            err.contains("expected must equal K exactly") && err.contains("K = 4"),
            "{err}"
        );
    }
    // Another task also releases D: the counter reaches K before the span did.
    let mut other = Spec::std(4, 2, 2);
    other.pools[0].push(t(T_PROD, 0, N, 0, 1));
    let err = refused(other);
    assert!(err.contains("other releasers"), "{err}");
    // A second releaser of the producer's event.
    let mut two = Spec::std(4, 2, 2);
    two.pools[0].push(t(T_PROD, 0, N, 0, 0));
    let err = refused(two);
    assert!(err.contains("producer must be its only releaser"), "{err}");
}

#[test]
fn the_live_list_and_count_have_one_writer_and_the_bitmap_none() {
    // A second *task* writing the live list.
    let mut task = Spec::std(4, 2, 2);
    task.records.push((T_WRITE, vec![(8, LIVE)]));
    task.known[0] = vec![T_PROD, T_WRITE];
    task.pools[0].push(t(T_WRITE, 3, N, 0, N));
    let err = refused(task);
    assert!(err.contains("only the producer task writes it"), "{err}");
    // ... the count cell.
    let mut count = Spec::std(4, 2, 2);
    count.records.push((T_ATOM, vec![(8, COUNT)]));
    count.known[0] = vec![T_PROD, T_ATOM];
    count.pools[0].push(t(T_ATOM, 3, N, 0, N));
    assert!(refused(count).contains("only the producer task writes it"));
    // ... the bitmap: no task at all.
    let mut bitmap = Spec::std(4, 2, 2);
    bitmap.records.push((T_ATOM, vec![(8, BITMAP)]));
    bitmap.known[0] = vec![T_PROD, T_ATOM];
    bitmap.pools[0].push(t(T_ATOM, 3, N, 0, N));
    let err = refused(bitmap);
    assert!(
        err.contains("claim bitmap") && err.contains("only the dispatcher"),
        "{err}"
    );
    // A kernel outside the pools writing the live list through an argument.
    let mut node = Spec::std(4, 2, 2);
    node.extra.push(writes(LIVE, VmemStore));
    let err = refused(node);
    assert!(err.contains("only the producer's node may"), "{err}");
    // ... or the bitmap.
    let mut node = Spec::std(4, 2, 2);
    node.extra.push(writes(BITMAP, AtomicCommutative));
    let err = refused(node);
    assert!(
        err.contains("claim bitmap") && err.contains("only the pool nodes may"),
        "{err}"
    );
    // Reading them is not a write.
    let mut read = Spec::std(4, 2, 2);
    read.extra.push(reads(&[LIVE, COUNT, BITMAP]));
    read.build().unwrap().check().unwrap();
}

#[test]
fn the_footprint_must_separate_every_pair_of_workgroups_of_the_grid() {
    // Axes {x} over a grid that is 3 tall: two live tiles in one column
    // would collide, and the host cannot know they do not.
    let err = refused(Spec::std(12, 4, 3).template(T_TILE_X));
    assert!(
        err.contains("inter-WG disjointness unproven: template 6, resource"),
        "{err}"
    );
    // The same footprint over a one-row grid covers every axis with extent > 1.
    Spec::std(4, 4, 1)
        .template(T_TILE_X)
        .build()
        .unwrap()
        .check()
        .unwrap();
    // {x, y} over a 2-D grid, and over a 3-D one it does not.
    Spec::std(12, 4, 3)
        .template(T_TILE_XY)
        .build()
        .unwrap()
        .check()
        .unwrap();
    let mut deep = Spec::std(4, 2, 2).template(T_TILE_XY);
    for t in deep.pools[1].iter_mut().filter(|t| t.flags != 0) {
        t.gz = 2;
    }
    assert!(refused(deep).contains("inter-WG disjointness unproven"));
    // Commutative atomics need no footprint; a plain writer is refused.
    Spec::std(12, 4, 3)
        .template(T_ATOM)
        .build()
        .unwrap()
        .check()
        .unwrap();
    let err = refused(Spec::std(12, 4, 3).template(T_WRITE));
    assert!(
        err.contains("proves no inter-WG write disjointness"),
        "{err}"
    );
}

#[test]
fn the_table_the_graph_and_the_resources_must_be_the_declared_ones() {
    let none = Spec::std(4, 2, 2).build().unwrap();
    let mut graph = none.graph.clone();
    graph.spans = None;
    let err = graph
        .check(&none.program, none.group.clone(), &none.protocol())
        .unwrap_err();
    assert!(err.contains("names no span table"), "{err}");
    let mut graph = none.graph.clone();
    graph.spans = Some(none.data(Y));
    let err = graph
        .check(&none.program, none.group.clone(), &none.protocol())
        .unwrap_err();
    assert!(err.contains("not a bound binding-record arena"), "{err}");

    // The live list: a LiveList with room for K.
    let mut fx = Spec::std(4, 2, 2).build().unwrap();
    let live = fx.data(LIVE);
    fx.program
        .kinds
        .insert(live, ResourceKind::LiveList { cap: 3, stride: 16 });
    assert!(fx.check().unwrap_err().contains("capacity >= K = 4"));
    fx.program.kinds.remove(&live);
    assert!(fx.check().unwrap_err().contains("not a LiveList"));
    // The count cell and bitmap: event cells with room for their content.
    let mut fx = Spec::std(4, 8, 8).build().unwrap();
    let bitmap = fx.data(BITMAP);
    fx.program.resources[bitmap.0 as usize].bytes = 4;
    let err = fx.check().unwrap_err();
    assert!(err.contains("hold 8 bytes (holds 4)"), "{err}");
    let mut fx = Spec::std(4, 2, 2).build().unwrap();
    let count = fx.data(COUNT);
    fx.program.kinds.remove(&count);
    assert!(fx
        .check()
        .unwrap_err()
        .contains("must be declared EventCells"));

    // An ordinary task on a span record.
    let mut mixed = Spec::std(4, 2, 2);
    mixed.known[0].push(T_TILE_XY);
    mixed.pools[0].push(t(T_TILE_XY, 1, N, 0, N));
    let err = refused(mixed);
    assert!(err.contains("ordinary task on template"), "{err}");
}

#[test]
fn the_span_table_is_written_by_railgun_through_the_binding_record_writer() {
    let fx = Spec::std(4, 2, 2).build().unwrap();
    let spans = fx.data(SPANS);
    let descs = [SpanDescriptor {
        live_list: fx.data(LIVE),
        count: fx.data(COUNT),
        bitmap: fx.data(BITMAP),
        k: 4,
    }];
    assert_eq!(read_span_table(&fx.program, spans).unwrap(), descs);
    let bindings: Vec<Binding> = fx.program.resources.clone();
    let image = fx.program.binding_record_image(spans, &bindings).unwrap();
    let base = |r: ResourceId| bindings[r.0 as usize].base;
    assert_eq!(
        image.pointers,
        vec![
            (32, base(descs[0].live_list)),
            (40, base(descs[0].count)),
            (48, base(descs[0].bitmap)),
        ]
    );
    // n_spans at +0, K at descriptor +24, zero at +28.
    assert_eq!(image.scalars, vec![(0, 1), (56, 4), (60, 0)]);
    assert_eq!(span_table_bytes(1), 64);
    // The layout is the receipt's M0 span ABI.
    let s = TaskSpanOffsets::M0;
    assert_eq!(
        (
            s.flags_offset,
            s.k_offset,
            s.list_id_offset,
            s.spans_arg_offset
        ),
        (
            TaskLayout::M0.flags_offset,
            TaskLayout::M0.k_offset,
            TaskLayout::M0.list_id_offset,
            M0_SPANS_ARG_OFFSET
        )
    );
    assert_eq!(
        (s.desc_base, s.desc_stride, s.nop_template),
        (32, 32, SPAN_DESC_TEMPLATE)
    );
    // A tampered table is refused (the exact inverse of the writer).
    let mut fx = fx;
    let Some(ResourceKind::BindingRecords(records)) = fx.program.kinds.get_mut(&spans) else {
        panic!("span table")
    };
    records[0].scalars[0].1 = 2;
    assert!(fx.check().unwrap_err().contains("n_spans is 2"));
}

#[test]
fn two_lists_are_two_instances_and_may_not_share_resources() {
    let two = || {
        let mut spec = Spec::std(4, 2, 2);
        // List 1: 2 tasks of a (2, 1, 1) grid writing Y, produced by a
        // second producer (record 5) and read by a Y reader (record 4).
        spec.records.push((T_TILE_XY, vec![(8, Y)]));
        spec.records.push((T_READ, vec![(0, Y)]));
        spec.records.push((T_PROD, vec![(0, LIVE2), (8, COUNT2)]));
        spec.pools[0].push(t(T_PROD, 5, N, 0, 2));
        for i in 0..2 {
            spec.pools[1].push(TaskHeader {
                flags: TASK_FLAG_SPAN,
                k: i,
                list_id: 1,
                vcount: 1,
                gx: 2,
                gy: 1,
                pred_event: 2,
                expected: 1,
                done_event: 3,
                ..t(T_TILE_XY, 3, 0, 0, 0)
            });
        }
        spec.pools[1].push(t(T_READ, 4, 3, 2, N));
        spec.known[0] = vec![T_PROD];
        spec.known[1] = vec![T_TILE_XY, T_READ];
        spec.descs.push((LIVE2, COUNT2, BITMAP2, 2));
        spec.extra.push(reads(&[LIVE2, COUNT2, BITMAP2]));
        spec
    };
    let fx = two().build().unwrap();
    let proof = fx.check().unwrap();
    let lists: Vec<(u32, u32)> = proof.spans.iter().map(|s| (s.list_id, s.k)).collect();
    assert_eq!(lists, [(0, 4), (1, 2)]);
    let oracle = fx
        .graph
        .check_pairwise(&fx.program, fx.group.clone(), &fx.protocol())
        .unwrap();
    assert_eq!(proof, oracle);
    assert_eq!(proof.ordering().spans(), 2);
    let mut shared = two();
    shared.descs[1].2 = BITMAP;
    let err = refused(shared);
    assert!(
        err.contains("claim bitmap resource")
            && err.contains("also the claim bitmap of span list 0"),
        "{err}"
    );
    // One record cannot be two lists.
    let mut same = two();
    for t in same.pools[1].iter_mut().filter(|t| t.list_id == 1) {
        t.ka_off = 64;
    }
    let err = refused(same);
    assert!(
        err.contains("share template") || err.contains("ordinary task"),
        "{err}"
    );
}

#[test]
fn a_program_with_spans_renders_a_receipt_only_with_completeness_evidence() {
    let fx = Spec::std(4, 2, 2).build().unwrap();
    let mut program = fx.program.clone();
    program
        .add_coresident_group(1..3, fx.check().unwrap().ordering())
        .unwrap();
    let admission = program.admission();
    assert_eq!(admission.spans(), 1);
    assert!(admission.is_certified(), "safety rests on certificates");
    assert!(matches!(
        &admission,
        Admission::Spans { base, lists: 1 } if **base == Admission::Certified
    ));
    // No evidence: refused, however the receipt is asked for.
    let err = admission.receipt(None).unwrap_err();
    assert!(err.contains(COMPLETENESS_FUNCTIONAL), "{err}");
    assert!(admission.receipt_with_completeness(None, None).is_err());
    let evidence = CompletenessEvidence {
        producer: "expand_live t30 obj sha256:ab12".into(),
        oracle: "job j8316 sha256:cd34".into(),
        p0_vs_eager: "job j8320 byte-identical".into(),
    };
    for blank in 0..3 {
        let mut e = evidence.clone();
        [&mut e.producer, &mut e.oracle, &mut e.p0_vs_eager][blank].clear();
        let err = admission
            .receipt_with_completeness(None, Some(&e))
            .unwrap_err();
        assert!(err.contains("is empty"), "{err}");
    }
    let receipt = admission
        .receipt_with_completeness(None, Some(&evidence))
        .unwrap();
    assert_eq!(receipt.admission, "proven");
    assert_eq!(receipt.completeness, Some(COMPLETENESS_FUNCTIONAL));
    assert_eq!(receipt.completeness_evidence.as_ref(), Some(&evidence));
    let json = serde_json::to_value(&receipt).unwrap();
    assert_eq!(
        json["completeness"],
        "functional (EXPAND_LIVE), not certified"
    );
    assert_eq!(
        json["completeness_evidence"]["oracle"],
        "job j8316 sha256:cd34"
    );
    // Evidence for a program without spans is a claim about nothing.
    let err = Admission::Certified
        .receipt_with_completeness(None, Some(&evidence))
        .unwrap_err();
    assert!(err.contains("without GPU-generated spans"), "{err}");
    let plain = Admission::Certified.receipt(None).unwrap();
    assert!(serde_json::to_value(&plain)
        .unwrap()
        .get("completeness")
        .is_none());
}

#[test]
fn four_thousand_eight_hundred_entries_check_fast() {
    let fx = Spec::std(4_800, 80, 60).build().unwrap();
    let start = Instant::now();
    let proof = fx.check().unwrap();
    let took = start.elapsed();
    assert_eq!(proof.tasks, 4_802);
    assert_eq!(proof.spans[0].k, 4_800);
    assert!(took.as_secs_f64() < 5.0, "a 4,800-entry span took {took:?}");
    // The refusals are as quick: the first duplicate index of a big list.
    let mut dup = Spec::std(4_800, 80, 60);
    dup.pools[1][4_799].k = 0;
    let start = Instant::now();
    let err = dup.build().unwrap().check().unwrap_err();
    assert!(err.contains("two tasks"), "{err}");
    assert!(start.elapsed().as_secs_f64() < 5.0);
}

#[cfg(feature = "lab")]
mod lab {
    use super::*;

    fn pending(fx: &Fixture) -> Result<crate::coresident::GroupOrdering, String> {
        fx.graph.check_pending(&fx.program, fx.group.clone())
    }

    #[test]
    fn lab_assumes_the_span_properties_pending_with_a_named_reason() {
        let mut spec = Spec::std(4, 2, 2);
        spec.protocol = None;
        let fx = spec.build().unwrap();
        // Certified `check` has no protocol to read; lab names what is missing.
        let ordering = pending(&fx).unwrap();
        assert!(ordering.is_pending());
        assert_eq!(ordering.spans(), 1);
        let reasons = ordering.pending_reasons();
        for property in [TASK_PROTO_SPAN_ENTRY, TASK_PROTO_REGION_WG_LIVE] {
            let want = format!("span protocol pending: node 2 (k): {property} not proven");
            assert!(reasons.contains(&want), "{reasons:?}");
        }
        // With the properties proven the span reasons are gone.
        let fx = Spec::std(4, 2, 2).build().unwrap();
        let ordering = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap();
        assert!(
            !ordering
                .pending_reasons()
                .iter()
                .any(|r| r.contains("span protocol pending")),
            "{:?}",
            ordering.pending_reasons()
        );
    }

    #[test]
    fn lab_names_an_uncovered_footprint_and_still_refuses_the_rest() {
        let fx = Spec::std(12, 4, 3).template(T_TILE_X).build().unwrap();
        let ordering = pending(&fx).unwrap();
        let x = fx.data(X).0;
        let reason = format!("inter-WG disjointness unproven: template {T_TILE_X}, resource {x}");
        assert!(
            ordering.pending_reasons().contains(&reason),
            "{:?}",
            ordering.pending_reasons()
        );
        // The structural checks hold in lab too.
        let mut spec = Spec::std(4, 2, 2);
        spec.pools[1][0].flags = 3;
        let err = pending(&spec.build().unwrap()).unwrap_err();
        assert!(err.contains("header flags 0x3"), "{err}");
        let mut spec = Spec::std(5, 2, 2);
        spec.protocol = None;
        assert!(pending(&spec.build().unwrap())
            .unwrap_err()
            .contains("exceeds"));
        let mut spec = Spec::std(4, 2, 2);
        spec.pools[1][4].expected = 3;
        assert!(pending(&spec.build().unwrap())
            .unwrap_err()
            .contains("expected must equal K exactly"));
        let mut spec = Spec::std(4, 2, 2);
        spec.extra.push(writes(LIVE, VmemStore));
        assert!(pending(&spec.build().unwrap())
            .unwrap_err()
            .contains("only the producer's node may"));
    }

    #[test]
    fn a_lab_span_program_needs_the_witness_and_the_completeness_evidence() {
        use crate::admission::DifferentialWitness;
        let mut spec = Spec::std(4, 2, 2);
        spec.protocol = None;
        let fx = spec.build().unwrap();
        let ordering = pending(&fx).unwrap();
        let mut program = fx.program.clone();
        program.add_coresident_group(1..3, ordering).unwrap();
        let admission = program.admission();
        let Admission::Spans { base, lists: 1 } = &admission else {
            panic!("{admission:?}")
        };
        assert!(matches!(**base, Admission::LabPending { .. }));
        let witness = DifferentialWitness::compare(b"f", b"a", b"a").unwrap();
        let evidence = CompletenessEvidence {
            producer: "p".into(),
            oracle: "o".into(),
            p0_vs_eager: "e".into(),
        };
        assert!(admission.receipt(Some(&witness)).is_err(), "no evidence");
        assert!(
            admission
                .receipt_with_completeness(None, Some(&evidence))
                .unwrap_err()
                .contains("witness"),
            "no witness"
        );
        let receipt = admission
            .receipt_with_completeness(Some(&witness), Some(&evidence))
            .unwrap();
        assert_eq!(receipt.certified, Some(false));
        assert_eq!(receipt.completeness, Some(COMPLETENESS_FUNCTIONAL));
    }
}
