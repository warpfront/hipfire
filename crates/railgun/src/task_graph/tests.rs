// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::sync::Arc;

use super::*;
use crate::kernel::tests::{facts, launch, ptr};
use crate::kernel::{
    author, ArenaFact, ArenaTemplate, ArgFact, ArgRole, ConditionalFacts, FactStatus, LaunchRecord,
    MemClass::*, NodeFacts, RegionEffect, SealedTableFact, TableAccess, TableBound, TableEntryLoad,
};
use crate::plan::{plan, CacheTable, RowPolicy};
use crate::sealed_table::tests as st;
use crate::sealed_table::{Pointee, SealedTableRegistry};
use crate::{Binding, BindingField, BindingRecord, EVENT_CELL_BYTES};

const TASKS: u64 = 0x7000_0000;
const EVENTS: u64 = 0x7100_0000;
const ARENA: u64 = 0x7200_0000;
const CTL: u64 = 0x7300_0000;
const X: u64 = 0x7400_0000;
const Y: u64 = 0x7500_0000;
const Z: u64 = 0x7600_0000;

/// Body templates: 1 reads field 0 and writes field 8; 2 reads field 0;
/// 3 touches nothing; 4 writes field 8.
const T_COPY: u32 = 1;
const T_READ: u32 = 2;
const T_NOP: u32 = 3;
const T_WRITE: u32 = 4;
/// Write-only field 8 with a certified workgroup footprint over axes x;
/// over x and y; a commutative atomic on field 8.
const T_TILE_X: u32 = 6;
const T_TILE_XY: u32 = 7;
const T_ATOM: u32 = 8;
/// Footprint over x with a tile-id bound of 4 on x; with the requirement
/// `dword@16 & 0xffff == 0x80` (met by the record) / `== 0x81` (not met).
const T_TILE_B: u32 = 9;
const T_TILE_R: u32 = 10;
const T_TILE_Q: u32 = 11;

const N: u32 = NONE;
const ALL: [u32; 4] = [T_COPY, T_READ, T_NOP, T_WRITE];
/// Reads through the sealed pointer table at field +0 (bound: the declared
/// `u32` scalar at record +8); every pointee is read-only.
const T_SEALED: u32 = 5;
const SEALED_BOUND_AT: u32 = 8;
const SEALED_N: u32 = st::ENTRIES;

fn t(template: u32, record: u32, pred: u32, expected: u32, done: u32) -> TaskHeader {
    TaskHeader {
        template,
        ka_off: record * 64,
        pred_event: pred,
        expected,
        done_event: done,
        vx: 0,
        vy: 0,
        vz: 0,
        gx: 1,
        gy: 1,
        gz: 1,
        vcount: 0,
    }
}

fn template(id: u32) -> ArenaTemplate {
    use crate::kernel::ArenaPointerLoad as L;
    let loads = match id {
        T_COPY => vec![
            L {
                record_offset: 0,
                classes: vec![VmemLoad],
            },
            L {
                record_offset: 8,
                classes: vec![VmemStore],
            },
        ],
        T_READ => vec![L {
            record_offset: 0,
            classes: vec![VmemLoad],
        }],
        T_NOP => vec![],
        T_WRITE => vec![L {
            record_offset: 8,
            classes: vec![VmemStore],
        }],
        T_TILE_X | T_TILE_XY | T_TILE_B | T_TILE_R | T_TILE_Q => vec![L {
            record_offset: 8,
            classes: vec![VmemStore],
        }],
        T_ATOM => vec![L {
            record_offset: 8,
            classes: vec![AtomicCommutative],
        }],
        T_SEALED => vec![L {
            record_offset: 0,
            classes: vec![SmemLoad],
        }],
        other => panic!("no template {other}"),
    };
    ArenaTemplate {
        template: id,
        pointer_loads: loads,
        sealed: if id == T_SEALED {
            vec![SealedTableFact {
                table_arg_offset: 0,
                bound: TableBound::Kernarg(SEALED_BOUND_AT),
                entry_bytes: 8,
                entries: vec![TableEntryLoad {
                    pc: 0x10,
                    class: SmemLoad,
                    path: "scalar".into(),
                    cache: "th=0".into(),
                    width: 8,
                    index_expr: "load@0x8".into(),
                    guard_pcs: vec![0x20],
                }],
                accesses: vec![TableAccess {
                    pc: 0x30,
                    class: VmemLoad,
                    path: "vector".into(),
                    cache: "th=0".into(),
                    width: 16,
                    entry_pcs: vec![0x10],
                }],
                null_guarded: false,
            }]
        } else {
            Vec::new()
        },
        wg_footprints: match id {
            T_TILE_X => vec![crate::kernel::WgFootprint::plain(8, 1)],
            T_TILE_XY => vec![crate::kernel::WgFootprint::plain(8, 3)],
            T_TILE_B => vec![crate::kernel::WgFootprint {
                bounds: [Some(4), None, None],
                ..crate::kernel::WgFootprint::plain(8, 1)
            }],
            T_TILE_R | T_TILE_Q => vec![crate::kernel::WgFootprint {
                requires: vec![(16, 0xffff, if id == T_TILE_R { 0x80 } else { 0x81 })],
                ..crate::kernel::WgFootprint::plain(8, 1)
            }],
            _ => Vec::new(),
        },
    }
}

fn proven_protocol() -> TaskProtocolFact {
    TaskProtocolFact {
        layout: TaskLayout::M0,
        tasks_arg_offset: 0,
        ticket_arg_offset: 8,
        events_arg_offset: 16,
        n_tasks_arg_offset: 56,
        proved: TASK_PROTO_REQUIRED
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
    }
}

fn region_wg_protocol() -> TaskProtocolFact {
    let mut p = proven_protocol();
    p.proved
        .push(railgun_corpus::receipt::TASK_PROTO_REGION_WG.to_owned());
    p
}

/// A tile claim: workgroups `vx..vx+vcount` at `(vy, vz = 0)` of a
/// `(gx, gy, 1)` virtual grid, of record `record`.
fn tile(template: u32, record: u32, vx: u32, vy: u32, vcount: u32, gx: u32, gy: u32) -> TaskHeader {
    TaskHeader {
        vx,
        vy,
        gx,
        gy,
        vcount,
        ..t(template, record, N, 0, N)
    }
}

fn tiles(
    template: u32,
    tasks: Vec<TaskHeader>,
    protocol: TaskProtocolFact,
) -> Result<TaskGraphProof, String> {
    build(
        vec![tasks],
        &[(template, &[(8, X)])],
        vec![vec![template]],
        Some(protocol),
        false,
    )
    .check()
}

#[test]
fn same_instance_tiles_with_a_footprint_are_admitted_only_with_the_property() {
    let two = || {
        vec![
            tile(T_TILE_X, 0, 0, 0, 2, 4, 1),
            tile(T_TILE_X, 0, 2, 0, 2, 4, 1),
        ]
    };
    let proof = tiles(T_TILE_X, two(), region_wg_protocol()).unwrap();
    assert_eq!(proof.tasks, 2);
    assert_eq!(proof.ordered_pairs, 0);
    let err = tiles(T_TILE_X, two(), proven_protocol()).unwrap_err();
    assert!(
        err.contains("race on resource") && err.contains("both write"),
        "{err}"
    );
}

#[test]
fn same_instance_tiles_with_overlapping_workgroups_are_refused() {
    let err = tiles(
        T_TILE_X,
        vec![
            tile(T_TILE_X, 0, 0, 0, 3, 4, 1),
            tile(T_TILE_X, 0, 2, 0, 2, 4, 1),
        ],
        region_wg_protocol(),
    )
    .unwrap_err();
    assert!(err.contains("run twice concurrently"), "{err}");
}

#[test]
fn same_instance_tiles_outside_the_virtual_grid_are_refused() {
    for bad in [
        tile(T_TILE_X, 0, 2, 0, 3, 4, 1), // x 2..5 > gx 4
        tile(T_TILE_X, 0, 2, 1, 2, 4, 1), // vy 1 >= gy 1
    ] {
        let err = tiles(
            T_TILE_X,
            vec![tile(T_TILE_X, 0, 0, 0, 2, 4, 1), bad],
            region_wg_protocol(),
        )
        .unwrap_err();
        assert!(err.contains("outside the record's virtual grid"), "{err}");
    }
}

#[test]
fn a_different_ka_off_of_one_template_still_needs_happens_before() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_TILE_X, &[(8, X)]), (T_TILE_X, &[(8, X)])];
    let run = |second: TaskHeader| {
        build(
            vec![vec![tile(T_TILE_X, 0, 0, 0, 2, 4, 1), second]],
            records,
            vec![vec![T_TILE_X]],
            Some(region_wg_protocol()),
            false,
        )
        .check()
    };
    let err = run(tile(T_TILE_X, 1, 2, 0, 2, 4, 1)).unwrap_err();
    assert!(err.contains("neither happens-before"), "{err}");
    // Ordered by an event, the same pair is admitted.
    let fx = build(
        vec![vec![t(T_TILE_X, 0, N, 0, 0), t(T_TILE_X, 1, 0, 1, N)]],
        records,
        vec![vec![T_TILE_X]],
        Some(region_wg_protocol()),
        false,
    );
    assert_eq!(fx.check().unwrap().ordered_pairs, 1);
}

#[test]
fn the_same_ka_off_cannot_name_two_templates() {
    // A record has one template: a task naming another is no instance of it.
    let err = tiles(
        T_TILE_X,
        vec![
            tile(T_TILE_X, 0, 0, 0, 2, 4, 1),
            tile(T_TILE_XY, 0, 2, 0, 2, 4, 1),
        ],
        region_wg_protocol(),
    )
    .unwrap_err();
    assert!(err.contains("names no record of its template"), "{err}");
}

#[test]
fn footprint_axes_must_separate_every_workgroup_pair() {
    let by_y = || {
        vec![
            tile(T_TILE_X, 0, 0, 0, 2, 4, 2),
            tile(T_TILE_X, 0, 0, 1, 2, 4, 2),
        ]
    };
    // Axes {x}: tiles that differ only in y overlap in x: refused.
    let err = tiles(T_TILE_X, by_y(), region_wg_protocol()).unwrap_err();
    assert!(
        err.contains("agree on every footprint axis") && err.contains("(0, 0, 0) and (0, 1, 0)"),
        "{err}"
    );
    // Axes {x, y}: the same tiles differ on y: admitted.
    let by_y_xy = vec![
        tile(T_TILE_XY, 0, 0, 0, 2, 4, 2),
        tile(T_TILE_XY, 0, 0, 1, 2, 4, 2),
    ];
    tiles(T_TILE_XY, by_y_xy, region_wg_protocol()).unwrap();
    // Axes {x}: disjoint x ranges at different y: admitted.
    let diag = vec![
        tile(T_TILE_X, 0, 0, 0, 2, 4, 2),
        tile(T_TILE_X, 0, 2, 1, 2, 4, 2),
    ];
    tiles(T_TILE_X, diag, region_wg_protocol()).unwrap();
}

#[test]
fn same_instance_writers_without_a_footprint_or_atomics_are_refused() {
    let two = |tpl| vec![tile(tpl, 0, 0, 0, 2, 4, 1), tile(tpl, 0, 2, 0, 2, 4, 1)];
    let err = tiles(T_WRITE, two(T_WRITE), region_wg_protocol()).unwrap_err();
    assert!(
        err.contains("proves no inter-WG write disjointness"),
        "{err}"
    );
    // Commutative atomics commute across workgroups.
    tiles(T_ATOM, two(T_ATOM), region_wg_protocol()).unwrap();
}

#[test]
fn footprint_assumptions_are_checked_against_the_headers_and_record() {
    let two = |tpl, gx| vec![tile(tpl, 0, 0, 0, 2, gx, 1), tile(tpl, 0, 2, 0, 2, gx, 1)];
    tiles(T_TILE_B, two(T_TILE_B, 4), region_wg_protocol()).unwrap();
    let err = tiles(T_TILE_B, two(T_TILE_B, 8), region_wg_protocol()).unwrap_err();
    assert!(err.contains("tile-id bound 4"), "{err}");
    tiles(T_TILE_R, two(T_TILE_R, 4), region_wg_protocol()).unwrap();
    let err = tiles(T_TILE_Q, two(T_TILE_Q, 4), region_wg_protocol()).unwrap_err();
    assert!(err.contains("requires record dword +16"), "{err}");
}

fn pool_args() -> Vec<ArgFact> {
    vec![
        ptr(0, &[SmemLoad]),
        ptr(8, &[AtomicCommutative]),
        ptr(16, &[AtomicRelease, LoadAcquire]),
        ptr(24, &[SmemLoad]),
        ArgFact {
            offset: 56,
            size: 4,
            role: ArgRole::Integer,
        },
    ]
}

fn conditional_facts(known: &[u32], protocol: Option<TaskProtocolFact>) -> Arc<NodeFacts> {
    let mut f = (*facts(pool_args(), 64)).clone();
    f.facts.status = FactStatus::Conditional(ConditionalFacts {
        guards: Vec::new(),
        indexed: Vec::new(),
        tables: Vec::new(),
        arenas: vec![ArenaFact {
            arena_arg_offset: 24,
            templates: known.iter().map(|&id| template(id)).collect(),
        }],
        task_protocol: protocol,
        witnesses: if known.contains(&T_SEALED) {
            vec!["descriptor_arena".into(), "sealed_table".into()]
        } else {
            vec!["descriptor_arena".into()]
        },
        case_digest: "ab".repeat(32),
    });
    Arc::new(f)
}

fn unknown_facts() -> Arc<NodeFacts> {
    let mut f = (*facts(pool_args(), 64)).clone();
    f.facts.status = FactStatus::Unknown(vec!["indirect-pointer-load".into()]);
    Arc::new(f)
}

fn pool_launch(pool: usize, n_tasks: u32, facts: Arc<NodeFacts>) -> LaunchRecord {
    let mut l = launch(
        &[
            (0, TASKS + pool as u64 * 0x1_0000),
            (8, CTL + 8 * pool as u64),
            (16, EVENTS),
            (24, ARENA),
        ],
        60,
        pool_args(),
    );
    l.kernarg[56..60].copy_from_slice(&n_tasks.to_le_bytes());
    l.facts = Ok(facts);
    l
}

fn rid(program: &KernelProgram, base: u64) -> ResourceId {
    ResourceId(
        program
            .resources
            .iter()
            .position(|b| b.base == base)
            .unwrap_or_else(|| panic!("no resource at {base:#x}")) as u32,
    )
}

struct Fixture {
    program: KernelProgram,
    group: Range<usize>,
    graph: TaskGraph,
}

impl Fixture {
    fn events(&self) -> ResourceId {
        rid(&self.program, EVENTS)
    }

    fn data(&self, base: u64) -> ResourceId {
        rid(&self.program, base)
    }

    fn protocol(&self) -> TaskProtocol {
        TaskProtocol::from_program(&self.program, self.group.clone()).unwrap()
    }

    fn check(&self) -> Result<TaskGraphProof, String> {
        self.graph
            .check(&self.program, self.group.clone(), &self.protocol())
    }
}

/// Node 0 touches the data resources; nodes `1..=pools` are the pool nodes.
/// Record `i` is at arena offset `64 * i`. `known[p]` are the templates pool
/// `p`'s receipt facts carry.
fn build(
    pools: Vec<Vec<TaskHeader>>,
    records: &[(u32, &[(u32, u64)])],
    known: Vec<Vec<u32>>,
    protocol: Option<TaskProtocolFact>,
    unknown: bool,
) -> Fixture {
    build_with(
        pools,
        records,
        known,
        protocol,
        unknown,
        &SealedTableRegistry::new(),
        Vec::new(),
    )
    .unwrap()
}

/// [`build`] over a sealed-table registry, with `extra` launches (after the
/// pools, so the group is unchanged) that make further allocations program
/// resources; `Err` is a refused `bind_records`.
fn build_with(
    pools: Vec<Vec<TaskHeader>>,
    records: &[(u32, &[(u32, u64)])],
    known: Vec<Vec<u32>>,
    protocol: Option<TaskProtocolFact>,
    unknown: bool,
    sealed: &SealedTableRegistry,
    extra: Vec<LaunchRecord>,
) -> Result<Fixture, String> {
    let mut launches = vec![launch(
        &[(0, X), (8, Y), (16, Z)],
        24,
        vec![
            ptr(0, &[VmemLoad]),
            ptr(8, &[VmemLoad]),
            ptr(16, &[VmemLoad]),
        ],
    )];
    for (p, tasks) in pools.iter().enumerate() {
        let f = if unknown {
            unknown_facts()
        } else {
            conditional_facts(&known[p], protocol.clone())
        };
        launches.push(pool_launch(p, tasks.len() as u32, f));
    }
    launches.extend(extra);
    let mut program = author("gfx1201", &launches);
    let big = 1 << 22;
    for base in [EVENTS, CTL] {
        let r = rid(&program, base);
        program.resources[r.0 as usize].bytes = big;
    }
    for p in 0..pools.len() {
        let r = rid(&program, TASKS + p as u64 * 0x1_0000);
        program.resources[r.0 as usize].bytes = big;
    }
    let events = rid(&program, EVENTS);
    let ctl = rid(&program, CTL);
    let arena = rid(&program, ARENA);
    let cells = (big / EVENT_CELL_BYTES) as u32;
    program.declare_event_cells(events, cells).unwrap();
    program.declare_event_cells(ctl, cells).unwrap();
    for p in 0..pools.len() {
        let r = rid(&program, TASKS + p as u64 * 0x1_0000);
        program.declare_task_arena(r).unwrap();
    }
    let records: Vec<BindingRecord> = records
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
            scalars: if *template == T_SEALED {
                vec![(SEALED_BOUND_AT, SEALED_N)]
            } else if matches!(*template, T_TILE_R | T_TILE_Q) {
                vec![(16, 0x1_0080)]
            } else {
                Vec::new()
            },
        })
        .collect();
    program.bind_records(arena, records, sealed)?;
    let group = 1..1 + pools.len();
    let graph = TaskGraph::new(
        pools
            .into_iter()
            .enumerate()
            .map(|(p, tasks)| PoolTasks { node: 1 + p, tasks })
            .collect(),
        events,
        20_000,
        arena,
    );
    Ok(Fixture {
        program,
        group,
        graph,
    })
}

/// Every pool knows every template.
fn fixture(pools: Vec<Vec<TaskHeader>>, records: &[(u32, &[(u32, u64)])]) -> Fixture {
    let known = vec![ALL.to_vec(); pools.len()];
    build(pools, records, known, Some(proven_protocol()), false)
}

#[test]
fn an_ordered_producer_consumer_is_admitted_and_orders_its_edge() {
    let fx = build(
        vec![vec![t(T_COPY, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 1, N)]],
        &[(T_COPY, &[(0, X), (8, Y)]), (T_READ, &[(0, Y)])],
        vec![vec![T_COPY], vec![T_READ]],
        Some(proven_protocol()),
        false,
    );
    let proof = fx.check().unwrap();
    assert_eq!(proof.tasks, 2);
    assert_eq!(proof.ordered_pairs, 1);
    assert_eq!(
        proof.edges(),
        &[OrderedEdge {
            producer: 1,
            consumer: 2,
            resource: fx.data(Y),
            event: fx.events(),
        }]
    );
    assert!(proof
        .obligations
        .iter()
        .any(|o| o.contains("zero at every submit")));
    let ordering = proof.ordering();
    assert_eq!(ordering.edges(), proof.edges());
    assert_eq!(ordering.task_graph(), Some(&(1..3)));
    assert!(!ordering.is_pending());
    let mut program = fx.program.clone();
    program.add_coresident_group(1..3, ordering).unwrap();
}

#[test]
fn a_war_order_is_an_edge_with_the_reader_first() {
    // pool 0 reads Y, then pool 1 (after it) writes Y.
    let fx = build(
        vec![vec![t(T_READ, 0, N, 0, 0)], vec![t(T_WRITE, 1, 0, 1, N)]],
        &[(T_READ, &[(0, Y)]), (T_WRITE, &[(8, Y)])],
        vec![vec![T_READ], vec![T_WRITE]],
        Some(proven_protocol()),
        false,
    );
    let proof = fx.check().unwrap();
    assert_eq!(
        proof.edges(),
        &[OrderedEdge {
            producer: 1,
            consumer: 2,
            resource: fx.data(Y),
            event: fx.events(),
        }]
    );
    let mut program = fx.program.clone();
    program
        .add_coresident_group(1..3, proof.ordering())
        .expect("a write-after-read order passes the edge validation");
}

#[test]
fn node_level_conflicts_no_task_pair_has_get_a_vacuous_edge() {
    // Both pools' receipts write Z through template 2 of a record, but no
    // task uses that record: the node effects over-approximate the tasks.
    let fx = build(
        vec![vec![t(T_COPY, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 1, N)]],
        &[
            (T_COPY, &[(0, X), (8, Y)]),
            (T_READ, &[(0, Y)]),
            (T_WRITE, &[(8, Z)]),
        ],
        vec![vec![T_COPY, T_WRITE], vec![T_READ, T_WRITE]],
        Some(proven_protocol()),
        false,
    );
    let proof = fx.check().unwrap();
    let z = fx.data(Z);
    assert!(
        proof.edges().iter().any(|e| e.resource == z),
        "{:?}",
        proof.edges()
    );
    let mut program = fx.program.clone();
    program
        .add_coresident_group(1..3, proof.ordering())
        .unwrap();
    let g = plan(
        &program,
        &CacheTable::for_arch("gfx1201").unwrap(),
        RowPolicy::TierD,
    );
    assert!(
        g.groups[0].refusals.is_empty(),
        "{:?}",
        g.groups[0].refusals
    );
}

#[test]
fn the_plan_refuses_a_task_protocol_member_without_a_task_graph_proof() {
    let fx = build(
        vec![vec![t(T_COPY, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 1, N)]],
        &[(T_COPY, &[(0, X), (8, Y)]), (T_READ, &[(0, Y)])],
        vec![vec![T_COPY], vec![T_READ]],
        Some(proven_protocol()),
        false,
    );
    let mut program = fx.program.clone();
    program
        .add_coresident_group(1..3, GroupOrdering::none())
        .unwrap();
    let g = plan(
        &program,
        &CacheTable::for_arch("gfx1201").unwrap(),
        RowPolicy::TierD,
    );
    assert!(
        g.groups[0]
            .refusals
            .iter()
            .any(|r| r.contains("task protocol but no task-graph proof")),
        "{:?}",
        g.groups[0].refusals
    );
    // A proof of another range does not cover the group.
    let other = GroupOrdering::none().with_task_graph(0..2);
    assert!(fx
        .program
        .clone()
        .add_coresident_group(1..3, other)
        .is_err());
}

#[test]
fn unordered_write_write_race_between_pools_is_refused_and_ordered_is_admitted() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_WRITE, &[(8, Y)]), (T_WRITE, &[(8, Y)])];
    let fx = fixture(
        vec![vec![t(T_WRITE, 0, N, 0, N)], vec![t(T_WRITE, 1, N, 0, N)]],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("race on resource"), "{err}");
    assert!(err.contains("both write"), "{err}");
    assert!(err.contains("neither happens-before"), "{err}");

    let fx = fixture(
        vec![vec![t(T_WRITE, 0, N, 0, 0)], vec![t(T_WRITE, 1, 0, 1, N)]],
        records,
    );
    let proof = fx.check().unwrap();
    assert_eq!(proof.ordered_pairs, 1);
    assert_eq!(proof.edges().len(), 1);
    assert_eq!(
        (proof.edges()[0].producer, proof.edges()[0].consumer),
        (1, 2)
    );
}

#[test]
fn a_read_write_race_is_refused_in_either_direction() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_READ, &[(0, Y)]), (T_WRITE, &[(8, Y)])];
    let fx = fixture(
        vec![vec![t(T_READ, 0, N, 0, N)], vec![t(T_WRITE, 1, N, 0, N)]],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("read/write"), "{err}");
    // Two readers never race.
    let fx = fixture(
        vec![vec![t(T_READ, 0, N, 0, N)], vec![t(T_READ, 0, N, 0, N)]],
        &[(T_READ, &[(0, Y)])],
    );
    fx.check().unwrap();
}

#[test]
fn a_same_pool_race_is_refused_and_an_event_chain_admits_it() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_WRITE, &[(8, Y)])];
    // Claim order is not an order: two workers run task 0 and task 1 at once.
    let fx = fixture(
        vec![vec![t(T_WRITE, 0, N, 0, N), t(T_WRITE, 0, N, 0, N)], vec![]],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("task 0 of pool 0"), "{err}");
    assert!(err.contains("task 1 of pool 0"), "{err}");

    let fx = fixture(
        vec![vec![t(T_WRITE, 0, N, 0, 0), t(T_WRITE, 0, 0, 1, N)], vec![]],
        records,
    );
    fx.check().unwrap();
}

#[test]
fn ordering_is_transitive_through_a_chain_of_events() {
    // pool0: a(done 0) ; pool1: b(pred 0, done 1) ; pool0: c(pred 1).
    // a and c write Y; the only order between them is a -> b -> c.
    let records: &[(u32, &[(u32, u64)])] = &[(T_WRITE, &[(8, Y)]), (T_NOP, &[])];
    let fx = fixture(
        vec![
            vec![t(T_WRITE, 0, N, 0, 0), t(T_WRITE, 0, 1, 1, N)],
            vec![t(T_NOP, 1, 0, 1, 1)],
        ],
        records,
    );
    let proof = fx.check().unwrap();
    assert_eq!(proof.ordered_pairs, 1);
    // Break the chain's middle: c waits on an event nobody that follows a
    // releases.
    let fx = fixture(
        vec![
            vec![t(T_WRITE, 0, N, 0, 0), t(T_WRITE, 0, 2, 1, N)],
            vec![t(T_NOP, 1, 0, 1, 1), t(T_NOP, 1, N, 0, 2)],
        ],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("race on resource"), "{err}");
}

#[test]
fn producer_cardinality_more_and_fewer_are_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    // Two producers, the consumer expects one: it could pass on a subset.
    let fx = fixture(
        vec![
            vec![t(T_NOP, 0, N, 0, 0), t(T_NOP, 0, N, 0, 0)],
            vec![t(T_NOP, 0, 0, 1, N)],
        ],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("2 tasks release it"), "{err}");
    assert!(err.contains("subset"), "{err}");
    // One producer, the consumer expects two: it never starts.
    let fx = fixture(
        vec![vec![t(T_NOP, 0, N, 0, 0)], vec![t(T_NOP, 0, 0, 2, N)]],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("1 tasks release it"), "{err}");
    assert!(err.contains("deadlocks"), "{err}");
    // Nobody releases an awaited event.
    let fx = fixture(vec![vec![t(T_NOP, 0, 3, 1, N)], vec![]], records);
    assert!(fx.check().unwrap_err().contains("0 tasks release it"));
    // Exactly `expected` producers: admitted.
    let fx = fixture(
        vec![
            vec![t(T_NOP, 0, N, 0, 0), t(T_NOP, 0, N, 0, 0)],
            vec![t(T_NOP, 0, 0, 2, N)],
        ],
        records,
    );
    fx.check().unwrap();
}

#[test]
fn a_dependency_cycle_across_two_pools_is_refused() {
    // Pool P = [p1, p2], pool Q = [q1, q2]: p1 waits on q2, q1 waits on p2.
    // Claim order makes p1 -> p2 and q1 -> q2; with p2 -> q1 and q2 -> p1 the
    // four tasks wait on each other although no single pool cycles.
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    let fx = fixture(
        vec![
            vec![t(T_NOP, 0, 1, 1, 2), t(T_NOP, 0, N, 0, 3)],
            vec![t(T_NOP, 0, 3, 1, 4), t(T_NOP, 0, N, 0, 1)],
        ],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("no progress"), "{err}");
    assert!(err.contains("dependency cycle"), "{err}");
    assert!(err.contains("is claimed after"), "{err}");
    // Reordering the pools' tasks (producers first) breaks the cycle.
    let fx = fixture(
        vec![
            vec![t(T_NOP, 0, N, 0, 3), t(T_NOP, 0, 1, 1, 2)],
            vec![t(T_NOP, 0, N, 0, 1), t(T_NOP, 0, 3, 1, 4)],
        ],
        records,
    );
    fx.check().unwrap();
}

#[test]
fn a_task_waiting_on_a_later_task_of_its_own_pool_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    let fx = fixture(
        vec![vec![t(T_NOP, 0, 1, 1, N), t(T_NOP, 0, N, 0, 1)], vec![]],
        records,
    );
    let err = fx.check().unwrap_err();
    assert!(err.contains("no progress"), "{err}");
    assert!(err.contains("task 0 of pool 0"), "{err}");
    // A task waiting on itself.
    let fx = fixture(vec![vec![t(T_NOP, 0, 0, 1, 0)], vec![]], records);
    assert!(fx.check().unwrap_err().contains("no progress"));
}

#[test]
fn a_task_naming_a_missing_record_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_COPY, &[(0, X), (8, Y)])];
    // No record at 5 * 64.
    let fx = fixture(vec![vec![t(T_COPY, 5, N, 0, N)], vec![]], records);
    let err = fx.check().unwrap_err();
    assert!(err.contains("task 0 of pool 0"), "{err}");
    assert!(err.contains("names no record of its template"), "{err}");
    // A record exists, but of another template.
    let fx = fixture(vec![vec![t(T_READ, 0, N, 0, N)], vec![]], records);
    assert!(fx
        .check()
        .unwrap_err()
        .contains("names no record of its template"));
}

#[test]
fn a_template_the_pools_receipt_does_not_know_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_COPY, &[(0, X), (8, Y)]), (T_READ, &[(0, Y)])];
    let fx = build(
        vec![vec![t(T_COPY, 0, N, 0, N)], vec![t(T_READ, 1, N, 0, N)]],
        records,
        vec![vec![T_READ], vec![T_COPY, T_READ]],
        Some(proven_protocol()),
        false,
    );
    let err = fx.check().unwrap_err();
    assert!(
        err.contains("not in the pool node's certified templates"),
        "{err}"
    );
}

#[test]
fn a_missing_or_partial_protocol_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    let tasks = || vec![vec![t(T_NOP, 0, N, 0, N)], vec![]];
    // No protocol in the receipt facts.
    let fx = build(tasks(), records, vec![ALL.to_vec(); 2], None, false);
    let err = TaskProtocol::from_program(&fx.program, fx.group.clone()).unwrap_err();
    assert!(err.contains("no certified task protocol"), "{err}");
    // Only one member has one.
    let mut fx = fixture(tasks(), records);
    fx.program.nodes[2].arena.as_mut().unwrap().task_protocol = None;
    assert!(TaskProtocol::from_program(&fx.program, fx.group.clone()).is_err());
    // A protocol missing a property.
    for missing in TASK_PROTO_REQUIRED {
        let mut p = proven_protocol();
        p.proved.retain(|q| q != missing);
        let fx = build(tasks(), records, vec![ALL.to_vec(); 2], Some(p), false);
        let err = fx.check().unwrap_err();
        assert!(err.contains(missing), "{err}");
        assert!(err.contains("not proved"), "{err}");
    }
    // A layout the headers were not parsed with.
    let mut p = proven_protocol();
    p.layout.desc_stride = 64;
    let fx = build(tasks(), records, vec![ALL.to_vec(); 2], Some(p), false);
    assert!(fx.check().unwrap_err().contains("task layout"));
}

#[test]
fn a_pool_without_certified_per_template_effects_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    let mut fx = fixture(vec![vec![t(T_NOP, 0, N, 0, N)], vec![]], records);
    let protocol = fx.protocol();
    fx.program.nodes[1].arena = None;
    let err = fx
        .graph
        .check(&fx.program, fx.group.clone(), &protocol)
        .unwrap_err();
    assert!(err.contains("no certified per-template effects"), "{err}");
}

#[test]
fn malformed_headers_and_pool_bindings_are_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_NOP, &[])];
    let base = || {
        fixture(
            vec![vec![t(T_NOP, 0, N, 0, N)], vec![t(T_NOP, 0, N, 0, N)]],
            records,
        )
    };

    // Event id past the cells.
    let mut fx = base();
    fx.graph.pools[0].tasks[0].done_event = 20_000;
    assert!(fx
        .check()
        .unwrap_err()
        .contains("past the 20000 event cells"));
    let mut fx = base();
    fx.graph.pools[0].tasks[0].pred_event = 20_000;
    assert!(fx.check().unwrap_err().contains("pred event 20000"));

    // More event ids than the event cells hold.
    let mut fx = base();
    fx.graph.event_cells = 1 << 30;
    assert!(fx.check().unwrap_err().contains("the event cells hold"));

    // The events resource is not event cells.
    let mut fx = base();
    fx.graph.events = fx.data(X);
    assert!(fx.check().unwrap_err().contains("not declared EventCells"));

    // A pool whose events argument is some other event cells.
    let mut fx = base();
    fx.graph.events = fx.data(CTL);
    assert!(fx.check().unwrap_err().contains("events argument"));

    // A task count that is not the pool's.
    let mut fx = base();
    fx.graph.pools[0].tasks.push(t(T_NOP, 0, N, 0, N));
    assert!(fx.check().unwrap_err().contains("n_tasks"));

    // Two pools sharing a claim ticket.
    let mut fx = base();
    let ticket = fx.program.nodes[1]
        .args
        .iter()
        .find(|(o, _)| *o == 8)
        .unwrap()
        .1
        .clone();
    for (o, a) in &mut fx.program.nodes[2].args {
        if *o == 8 {
            *a = ticket.clone();
        }
    }
    assert!(fx.check().unwrap_err().contains("ticket is shared"));

    // A ticket inside the event ids.
    let mut fx = base();
    let events = fx.events();
    for (o, a) in &mut fx.program.nodes[1].args {
        if *o == 8 {
            *a = Arg::Resource {
                resource: events,
                interior: 16,
            };
        }
    }
    assert!(fx.check().unwrap_err().contains("lies in event ids"));

    // A pool that is not in the graph / a graph pool outside the group.
    let mut fx = base();
    fx.graph.pools.pop();
    assert!(fx.check().unwrap_err().contains("has no task pool"));
    let mut fx = base();
    fx.graph.pools[1].node = 5;
    assert!(fx.check().unwrap_err().contains("not in group"));

    // Task headers somebody writes.
    let mut fx = base();
    let tasks = rid(&fx.program, TASKS);
    fx.program.nodes[0].effects.push(RegionEffect {
        arg_offset: 0,
        resource: tasks,
        classes: vec![VmemStore],
    });
    assert!(fx.check().unwrap_err().contains("immutable during replay"));

    // More tasks than the arena holds.
    let mut fx = base();
    let tasks = rid(&fx.program, TASKS);
    fx.program.resources[tasks.0 as usize].bytes = 64;
    assert!(fx.check().unwrap_err().contains("task arena"));
}

#[test]
fn access_outside_the_protocol_is_refused() {
    let records: &[(u32, &[(u32, u64)])] = &[(T_WRITE, &[(8, Y)])];
    let mut fx = fixture(vec![vec![t(T_WRITE, 0, N, 0, N)], vec![]], records);
    // Pool 1 also stores into Y through an explicit pointer argument.
    let y = fx.data(Y);
    fx.program.nodes[2].effects.push(RegionEffect {
        arg_offset: 32,
        resource: y,
        classes: vec![VmemStore],
    });
    let err = fx.check().unwrap_err();
    assert!(err.contains("outside the task protocol"), "{err}");
    // Reaching a resource no task touches is fine.
    let mut fx = fixture(vec![vec![t(T_WRITE, 0, N, 0, N)], vec![]], records);
    let z = fx.data(Z);
    fx.program.nodes[2].effects.push(RegionEffect {
        arg_offset: 32,
        resource: z,
        classes: vec![VmemStore],
    });
    fx.check().unwrap();
}

#[test]
fn headers_parse_from_the_uploaded_bytes() {
    let mut bytes = vec![0u8; 256];
    let mut put = |rec: usize, at: u32, v: u32| {
        let at = rec * 128 + at as usize;
        bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    let l = TaskLayout::M0;
    // Task 0: template 7 at record offset 64, waits 3 for 2, releases 9.
    put(0, l.template_offset, 7);
    put(0, l.ka_off_offset, 64);
    put(0, l.pred_event_offset, 3);
    put(0, l.expected_offset, 2);
    put(0, l.done_event_offset, 9);
    // Task 1: no events.
    put(1, l.template_offset, 1);
    put(1, l.pred_event_offset, NONE);
    put(1, l.done_event_offset, NONE);
    let pool = PoolTasks::parse(4, &bytes, &l).unwrap();
    assert_eq!(pool.node, 4);
    assert_eq!(
        pool.tasks,
        vec![
            TaskHeader {
                vx: 0,
                vy: 0,
                vz: 0,
                gx: 0,
                gy: 0,
                gz: 0,
                vcount: 0,
                template: 7,
                ka_off: 64,
                pred_event: 3,
                expected: 2,
                done_event: 9
            },
            TaskHeader {
                vx: 0,
                vy: 0,
                vz: 0,
                gx: 0,
                gy: 0,
                gz: 0,
                vcount: 0,
                template: 1,
                ka_off: 0,
                pred_event: NONE,
                expected: 0,
                done_event: NONE
            },
        ]
    );
    assert!(PoolTasks::parse(4, &bytes[..200], &l).is_err());
    assert_eq!(PoolTasks::parse(4, &[], &l).unwrap().tasks, vec![]);
}

#[test]
fn a_ten_thousand_task_chain_is_proved() {
    // Alternating pools; task i waits for task i-1 and every task writes Y:
    // the whole of the 10^4 x 10^4 pair space is ordered through the closure.
    let n = 10_000u32;
    let mut pools = vec![Vec::new(), Vec::new()];
    for i in 0..n {
        let pred = if i == 0 { N } else { i - 1 };
        let expected = u32::from(i != 0);
        pools[(i % 2) as usize].push(t(T_WRITE, 0, pred, expected, i));
    }
    let fx = fixture(pools, &[(T_WRITE, &[(8, Y)])]);
    let proof = fx.check().unwrap();
    assert_eq!(proof.tasks, n as usize);
    assert_eq!(proof.ordered_pairs, u64::from(n) * u64::from(n - 1) / 2);
    assert_eq!(proof.events_used, n as usize);
}

#[cfg(feature = "lab")]
mod lab {
    use super::*;

    #[test]
    fn check_pending_runs_the_host_checks_and_yields_a_pending_ordering() {
        // The receipts carry arena facts but no task protocol yet.
        let tasks = || vec![vec![t(T_COPY, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 1, N)]];
        let records: &[(u32, &[(u32, u64)])] = &[(T_COPY, &[(0, X), (8, Y)]), (T_READ, &[(0, Y)])];
        let fx = build(
            tasks(),
            records,
            vec![vec![T_COPY], vec![T_READ]],
            None,
            false,
        );
        assert!(
            TaskProtocol::from_program(&fx.program, fx.group.clone()).is_err(),
            "check needs a proven protocol"
        );
        let ordering = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap();
        assert!(ordering.is_pending());
        assert_eq!(ordering.task_graph(), Some(&(1..3)));
        assert_eq!(ordering.edges().len(), 1);
        assert_eq!(
            (ordering.edges()[0].producer, ordering.edges()[0].consumer),
            (1, 2)
        );
        let mut program = fx.program.clone();
        program.add_coresident_group(1..3, ordering).unwrap();

        // The checks are the same ones: a race is refused while pending too.
        let fx = build(
            vec![vec![t(T_COPY, 0, N, 0, N)], vec![t(T_READ, 1, N, 0, N)]],
            records,
            vec![vec![T_COPY], vec![T_READ]],
            None,
            false,
        );
        let err = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap_err();
        assert!(err.contains("race on resource"), "{err}");
        // And so is a header cardinality error.
        let fx = build(
            vec![vec![t(T_COPY, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 2, N)]],
            records,
            vec![vec![T_COPY], vec![T_READ]],
            None,
            false,
        );
        let err = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap_err();
        assert!(err.contains("deadlocks"), "{err}");
    }

    /// The lab flow of today's pool receipts: `A1 Unknown`, no arena facts.
    /// Every record field is conservatively read and written, so only an
    /// ordered graph passes.
    #[test]
    fn unknown_receipts_are_checked_conservatively() {
        let records: &[(u32, &[(u32, u64)])] = &[(T_COPY, &[(0, X), (8, Y)]), (T_READ, &[(0, Y)])];
        let ordered = |pred: u32, expected: u32| {
            let mut fx = build(
                vec![
                    vec![t(T_COPY, 0, N, 0, 0)],
                    vec![t(T_READ, 1, pred, expected, N)],
                ],
                records,
                vec![vec![], vec![]],
                None,
                true,
            );
            let arena = fx.graph.arena;
            for node in fx.group.clone() {
                fx.program.admit_lab_pending(node, Some(arena)).unwrap();
            }
            fx
        };
        let fx = ordered(0, 1);
        let ordering = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap();
        assert!(ordering.is_pending());
        let mut program = fx.program.clone();
        program.add_coresident_group(1..3, ordering).unwrap();

        // Unordered: the conservative effects of X (read+written by the copy
        // task) and Y race.
        let fx = ordered(N, 0);
        let err = fx
            .graph
            .check_pending(&fx.program, fx.group.clone())
            .unwrap_err();
        assert!(err.contains("race on resource"), "{err}");

        // The caller may supply the body kernels' own classes instead.
        let mut templates = BTreeMap::new();
        templates.insert(T_COPY, template(T_COPY).pointer_loads);
        templates.insert(T_READ, template(T_READ).pointer_loads);
        let fx = ordered(N, 0);
        let err = fx
            .graph
            .check_pending_with(&fx.program, fx.group.clone(), Some(&templates))
            .unwrap_err();
        assert!(err.contains("race on resource"), "{err}");
    }
}

// ---- sealed tables of arena templates (design D.2) ------------------------

// The default seal's addresses moved clear of the fixture's `X`/`Y`/`Z`.
const TBL: u64 = 0x7800_0000;
const PW0: u64 = 0x7900_0000;
const PW1: u64 = 0x7a00_0000;

fn tbl() -> Binding {
    Binding {
        base: TBL,
        bytes: st::TABLE_BYTES,
    }
}

fn pw0() -> Binding {
    Binding {
        base: PW0,
        bytes: st::W_BYTES,
    }
}

fn pw1() -> Binding {
    Binding {
        base: PW1,
        bytes: st::W_BYTES,
    }
}

fn pointees() -> Vec<Pointee> {
    vec![Pointee::immutable(pw0()), Pointee::immutable(pw1())]
}

fn entry_ptrs() -> [u64; 4] {
    [PW0, PW0 + 0x100, PW1 + 8, PW1 + st::W_BYTES - 8]
}

/// The table and both pointee allocations of the default seal as program
/// resources (read by plain nodes after the pools, so the group is unchanged).
fn sealed_extras(more: Vec<LaunchRecord>) -> Vec<LaunchRecord> {
    use crate::index_bind::tests as ib;
    let mut v = vec![
        ib::extra_launch("table", TBL, tbl(), &[SmemLoad]),
        ib::extra_launch("w0", PW0, pw0(), &[VmemLoad]),
        ib::extra_launch("w1", PW1, pw1(), &[VmemLoad]),
    ];
    v.extend(more);
    v
}

fn sealed_registry() -> SealedTableRegistry {
    let registry = SealedTableRegistry::new();
    registry
        .seal(
            tbl(),
            st::ENTRIES,
            &st::table_bytes(&entry_ptrs()),
            &pointees(),
            st::GEN,
        )
        .unwrap();
    registry
}

/// Pool 0 runs one sealed-table task (record 0: the table field at +0, the
/// declared entry count at +8); pool 1 runs a task reading `W0` (a pointee)
/// through an ordinary field, ordered after it.
fn sealed_fixture(
    registry: &SealedTableRegistry,
    more: Vec<LaunchRecord>,
) -> Result<Fixture, String> {
    build_with(
        vec![vec![t(T_SEALED, 0, N, 0, 0)], vec![t(T_READ, 1, 0, 1, N)]],
        &[(T_SEALED, &[(0, TBL)]), (T_READ, &[(0, PW0)])],
        vec![vec![T_SEALED], vec![T_READ]],
        Some(proven_protocol()),
        false,
        registry,
        sealed_extras(more),
    )
}

#[test]
fn a_sealed_table_task_reads_the_table_and_every_pointee() {
    let fx = sealed_fixture(&sealed_registry(), Vec::new()).unwrap();
    let (table, w0, w1) = (fx.data(TBL), fx.data(PW0), fx.data(PW1));
    let node = &fx.program.nodes[1];
    let arena = node.arena.as_ref().unwrap();
    assert_eq!(arena.sealed.len(), 1);
    assert_eq!(arena.sealed[0].pointees, vec![w0, w1]);

    // The per-task accesses: table and both pointees, read-only.
    let mode = Mode {
        pending: false,
        templates: None,
    };
    let source = fx.graph.pool_effects(node, "pool 0", mode).unwrap();
    let ResourceKind::BindingRecords(records) = &fx.program.kinds[&fx.graph.arena] else {
        panic!("bound arena")
    };
    let access = fx
        .graph
        .task_effects(&fx.program, &source, &records[0], "task 0")
        .unwrap();
    let got: BTreeSet<(ResourceId, bool, bool)> = access
        .iter()
        .map(|a| (a.resource, a.read, a.write))
        .collect();
    assert_eq!(
        got,
        BTreeSet::from([(table, true, false), (w0, true, false), (w1, true, false)])
    );

    // Reading the same pointee from the other pool is no conflict: the proof
    // is admitted and orders nothing on the weights.
    let proof = fx.check().unwrap();
    assert!(
        proof
            .edges()
            .iter()
            .all(|e| e.resource != w0 && e.resource != w1),
        "{:?}",
        proof.edges()
    );
}

#[test]
fn a_writer_of_a_sealed_pointee_or_table_in_the_group_refuses() {
    for (what, target) in [("pointee", PW1), ("table", TBL)] {
        let mut fx = sealed_fixture(&sealed_registry(), Vec::new()).unwrap();
        let r = fx.data(target);
        // A pool member (node 2) whose node effects write sealed content.
        fx.program.nodes[2].effects.push(RegionEffect {
            arg_offset: 0,
            resource: r,
            classes: vec![VmemStore],
        });
        let err = fx.check().unwrap_err();
        assert!(
            err.contains("sealed content is immutable") && err.contains(what),
            "{what}: {err}"
        );
    }
}

#[test]
fn a_sealed_table_bind_is_refused_for_writers_missing_seals_and_long_bounds() {
    use crate::index_bind::tests as ib;
    // A node outside the group writes an expert-weight allocation.
    let err = sealed_fixture(
        &sealed_registry(),
        vec![ib::extra_launch(
            "weight_writer",
            PW0 + 0x100,
            pw0(),
            &[VmemStore],
        )],
    )
    .err()
    .expect("a written pointee refuses");
    assert!(
        err.contains("writes the sealed pointee allocation"),
        "{err}"
    );

    // A record of another template in the same bind writes one through its
    // fields.
    let err = build_with(
        vec![vec![t(T_SEALED, 0, N, 0, N)]],
        &[(T_SEALED, &[(0, TBL)]), (T_WRITE, &[(8, PW1)])],
        vec![vec![T_SEALED, T_WRITE]],
        Some(proven_protocol()),
        false,
        &sealed_registry(),
        sealed_extras(Vec::new()),
    )
    .err()
    .expect("a record-field writer refuses");
    assert!(
        err.contains("writes the sealed pointee allocation"),
        "{err}"
    );

    // No seal: no witness.
    let err = sealed_fixture(&SealedTableRegistry::new(), Vec::new())
        .err()
        .expect("an unsealed table refuses");
    assert!(err.contains("no sealed table"), "{err}");

    // The declared bound (4) is past a table sealed with two entries.
    let short = SealedTableRegistry::new();
    short
        .seal(
            tbl(),
            2,
            &st::table_bytes(&entry_ptrs()[..2]),
            &pointees(),
            st::GEN,
        )
        .unwrap();
    let err = sealed_fixture(&short, Vec::new())
        .err()
        .expect("a bound past the table refuses");
    assert!(err.contains("exceeds the 2 sealed entries"), "{err}");
}
