// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only tests of the sealed tables of a descriptor-arena template
//! (design D.2: bounded indices into a sealed immutable source bank): a pool
//! kernel whose records carry a pointer to a loader-sealed pointer table and
//! a railgun-declared entry-count scalar. `bind_records` discharges the
//! program-level obligations, the window bind (`bind_sealed_table`) the
//! generation, content and no-writer closure. Hand-built facts only.

use std::sync::Arc;

use super::tests as st;
use super::*;
use crate::index_bind::tests as ib;
use crate::kernel::tests::{facts, launch, ptr};
use crate::kernel::{
    author, ArenaFact, ArenaPointerLoad, ArenaTemplate, ConditionalFacts, FactStatus, LaunchRecord,
    MemClass::*, NodeFacts, SealedTableFact, TableAccess, TableBound, TableEntryLoad,
};
use crate::{BindingField, BindingRecord, ResourceKind};

const ARENA: u64 = 0x7000_0000;
const TEMPLATE: u32 = 5;
/// The record offset of the entry-count scalar.
const BOUND_AT: u32 = 8;

fn sealed_fact(bound: TableBound) -> SealedTableFact {
    SealedTableFact {
        table_arg_offset: 0,
        bound,
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
    }
}

fn pool_facts(bound: TableBound) -> Arc<NodeFacts> {
    let mut f = (*facts(vec![ptr(0, &[SmemLoad])], 8)).clone();
    f.facts.status = FactStatus::Conditional(ConditionalFacts {
        guards: Vec::new(),
        indexed: Vec::new(),
        tables: Vec::new(),
        arenas: vec![ArenaFact {
            arena_arg_offset: 0,
            templates: vec![ArenaTemplate {
                template: TEMPLATE,
                pointer_loads: vec![ArenaPointerLoad {
                    record_offset: 0,
                    classes: vec![SmemLoad],
                }],
                sealed: vec![sealed_fact(bound)],
                wg_footprints: Vec::new(),
                hidden_grid: Vec::new(),
            }],
        }],
        task_protocol: None,
        witnesses: vec!["descriptor_arena".into(), "sealed_table".into()],
        case_digest: "ab".repeat(32),
    });
    Arc::new(f)
}

/// Node 0 is the pool node (arena at resource 0); the table (1) and the two
/// pointee allocations (2, 3) are program resources touched by plain readers;
/// `extra` follows.
fn program(bound: TableBound, extra: Vec<LaunchRecord>) -> KernelProgram {
    let mut pool = launch(&[(0, ARENA)], 8, vec![ptr(0, &[SmemLoad])]);
    pool.facts = Ok(pool_facts(bound));
    let mut launches = vec![
        pool,
        ib::extra_launch("table", st::TABLE, st::table_binding(), &[SmemLoad]),
        ib::extra_launch("w0", st::W0, st::w0(), &[VmemLoad]),
        ib::extra_launch("w1", st::W1, st::w1(), &[VmemLoad]),
    ];
    launches.extend(extra);
    author("gfx1201", &launches)
}

fn res(p: &KernelProgram, base: u64) -> ResourceId {
    ResourceId(p.resources.iter().position(|b| b.base == base).unwrap() as u32)
}

fn record(p: &KernelProgram, scalars: Vec<(u32, u32)>) -> BindingRecord {
    BindingRecord {
        base: 0,
        template: TEMPLATE,
        fields: vec![BindingField {
            offset: 0,
            resource: res(p, st::TABLE),
            interior: 0,
        }],
        scalars,
    }
}

fn sealed_registry() -> SealedTableRegistry {
    let registry = SealedTableRegistry::new();
    st::seal_default(&registry).unwrap();
    registry
}

fn bind(
    p: &mut KernelProgram,
    scalars: Vec<(u32, u32)>,
    registry: &SealedTableRegistry,
) -> Result<(), String> {
    let r = record(p, scalars);
    p.bind_records(ResourceId(0), vec![r], registry)
}

struct Window {
    program: KernelProgram,
    allocations: Vec<AllocationIdentity>,
    images: Vec<Vec<u8>>,
    steps: Vec<WindowStep>,
}

/// The program a plan lays out (assuming the `sealed_table` witness) and the
/// bound state of a window over it.
fn window(bound: &KernelProgram, table_generation: u64) -> Window {
    let program = bound.assume_sealed_table_witness();
    let words = program.authored_words();
    let images = (0..program.nodes.len())
        .map(|i| {
            program.nodes[i].segment_bytes.map_or(Vec::new(), |s| {
                program.kernarg_image(i, s as usize, &words).unwrap()
            })
        })
        .collect();
    let allocations = program
        .resources
        .iter()
        .enumerate()
        .map(|(i, b)| AllocationIdentity {
            resource: ResourceId(i as u32),
            base: b.base,
            bytes: b.bytes,
            generation: if b.base == st::TABLE {
                table_generation
            } else {
                1
            },
        })
        .collect();
    let steps = (0..program.nodes.len())
        .map(|node| WindowStep::Kernel { node })
        .collect();
    Window {
        program,
        allocations,
        images,
        steps,
    }
}

fn window_bind(
    w: &Window,
    registry: &SealedTableRegistry,
    content: &st::MockContent,
) -> Result<SealedTableWitness, SealedTableError> {
    let grids: Vec<[u32; 3]> = w.program.nodes.iter().map(|n| n.grid).collect();
    registry.bind_sealed_table(
        &TableBindInputs {
            window: 1,
            shape_generation: 7,
            program: &w.program,
            steps: &w.steps,
            bindings: &w.program.resources,
            allocations: &w.allocations,
            images: &w.images,
            grids: &grids,
        },
        content,
    )
}

#[test]
fn a_bound_template_with_a_sealed_table_is_discharged_end_to_end() {
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let registry = sealed_registry();
    bind(&mut p, vec![(BOUND_AT, st::ENTRIES)], &registry).unwrap();
    let (table, w0, w1) = (res(&p, st::TABLE), res(&p, st::W0), res(&p, st::W1));

    let node = &p.nodes[0];
    // Stays conditional on the window's sealed_table witness, not on the bind.
    assert!(
        matches!(&node.lowering, Lowering::Conditional { reason } if reason == "sealed_table witness not consumed"),
        "{:?}",
        node.lowering
    );
    assert_eq!(p.sealed_table_nodes(), vec![0]);
    let arena = node.arena.as_ref().unwrap();
    assert!(arena.all_bound());
    assert_eq!(arena.sealed.len(), 1);
    let b = &arena.sealed[0];
    assert_eq!(
        (b.resource, b.bound, b.entries, b.generation),
        (table, st::ENTRIES, st::ENTRIES, st::GEN)
    );
    assert_eq!(b.pointees, vec![w0, w1]);
    // The effects: the table read and a read of every pointee, nothing else.
    let effect = |r: ResourceId| {
        node.effects
            .iter()
            .find(|e| e.resource == r && e.arg_offset == 0 && r != ResourceId(0))
            .unwrap_or_else(|| panic!("no effect on {r:?}"))
    };
    assert!(effect(table).has(SmemLoad) && !effect(table).writes());
    assert!(effect(w0).has(VmemLoad) && !effect(w0).writes());
    assert!(effect(w1).has(VmemLoad) && !effect(w1).writes());
    assert!(node.effects.iter().all(|e| !e.writes()));
    // Without the witness a plan never sees it as PM4; with it, it does.
    assert_eq!(
        bound_lowering_with_witness(&p),
        Lowering::Pm4,
        "assumed witness"
    );

    // The scalar is written by railgun, like the pointer.
    let image = p.binding_record_image(ResourceId(0), &p.resources).unwrap();
    assert_eq!(image.pointers, vec![(0, st::TABLE)]);
    assert_eq!(image.scalars, vec![(u64::from(BOUND_AT), st::ENTRIES)]);
    assert_eq!(image.len(), 2);
    let writes = image.writes();
    assert_eq!(writes[0], (0, st::TABLE.to_le_bytes().to_vec()));
    assert_eq!(
        writes[1],
        (u64::from(BOUND_AT), st::ENTRIES.to_le_bytes().to_vec())
    );
    assert!(matches!(
        p.kinds[&ResourceId(0)],
        ResourceKind::BindingRecords(_)
    ));

    // The window bind re-verifies and issues the witness the lease holds.
    let w = window(&p, st::GEN);
    let content = st::MockContent::new();
    let witness = window_bind(&w, &registry, &content).unwrap();
    assert_eq!(witness.nodes().len(), 1);
    let n = &witness.nodes()[0];
    assert_eq!(n.node, 0);
    assert_eq!(n.case_digest, "ab".repeat(32));
    assert_eq!(n.tables.len(), 1);
    assert_eq!(n.tables[0].bound, st::ENTRIES);
    assert_eq!(n.tables[0].allocation, st::table_binding());
    assert_eq!(
        witness.discharged_cases(),
        vec![(0, "ab".repeat(32).as_str(), "sealed_table")]
    );
    let lease = registry.lease(&witness).unwrap();
    assert!(matches!(
        registry.retire(st::TABLE),
        Err(SealedTableError::Retire(m)) if m.contains("in flight")
    ));
    lease.release(Quiescence::Proven);
    assert_eq!(registry.retire(st::TABLE), Ok(st::GEN + 1));
}

fn bound_lowering_with_witness(p: &KernelProgram) -> Lowering {
    p.assume_sealed_table_witness().nodes[0].lowering.clone()
}

#[test]
fn a_zero_bound_is_vacuous_and_needs_no_seal() {
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    bind(&mut p, vec![(BOUND_AT, 0)], &SealedTableRegistry::new()).unwrap();
    assert!(p.nodes[0].arena.as_ref().unwrap().sealed.is_empty());
    assert!(p.nodes[0].lowering.is_pm4(), "{:?}", p.nodes[0].lowering);
    assert!(p.sealed_table_nodes().is_empty());
}

#[test]
fn a_bound_past_the_table_is_refused() {
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let before = p.clone();
    let err = bind(
        &mut p,
        vec![(BOUND_AT, st::ENTRIES + 1)],
        &sealed_registry(),
    )
    .unwrap_err();
    assert!(err.contains("exceeds the 4 sealed entries"), "{err}");
    assert_eq!(p, before);
    // A constant bound is checked the same way.
    let mut c = program(TableBound::Const(st::ENTRIES + 1), Vec::new());
    let err = bind(&mut c, Vec::new(), &sealed_registry()).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
}

#[test]
fn a_missing_or_foreign_seal_is_refused() {
    // No seal at all.
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let before = p.clone();
    let err = bind(&mut p, vec![(BOUND_AT, 1)], &SealedTableRegistry::new()).unwrap_err();
    assert!(err.contains("no sealed table"), "{err}");
    assert_eq!(p, before);

    // Sealed, but over another allocation at that base (the resource is not
    // the sealed allocation exactly).
    let registry = SealedTableRegistry::new();
    let bigger = Binding {
        base: st::TABLE,
        bytes: st::TABLE_BYTES * 2,
    };
    registry
        .seal(
            bigger,
            st::ENTRIES,
            &st::table_bytes(&st::entry_ptrs()),
            &st::pointees(),
            st::GEN,
        )
        .unwrap();
    let err = bind(&mut p, vec![(BOUND_AT, 1)], &registry).unwrap_err();
    assert!(err.contains("sealed at that base"), "{err}");

    // Sealed at the right place, but the table points into a pointee the
    // program partially overlaps: not a table this program may read.
    let mut q = program(
        TableBound::Kernarg(BOUND_AT),
        vec![ib::extra_launch(
            "overlap",
            st::W0 + 0x8000,
            Binding {
                base: st::W0 + 0x8000,
                bytes: 0x1_0000,
            },
            &[VmemLoad],
        )],
    );
    let err = bind(&mut q, vec![(BOUND_AT, 1)], &sealed_registry()).unwrap_err();
    assert!(err.contains("partially overlaps"), "{err}");
}

#[test]
fn an_undeclared_bound_or_table_field_is_refused() {
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let registry = sealed_registry();
    // The bound byte is not a declared scalar of the record.
    let err = bind(&mut p, Vec::new(), &registry).unwrap_err();
    assert!(err.contains("not a declared scalar"), "{err}");
    // A different declared scalar is not the bound either.
    let err = bind(&mut p, vec![(12, 2)], &registry).unwrap_err();
    assert!(err.contains("not a declared scalar"), "{err}");
    // The table pointer is not a declared field.
    let mut r = record(&p, vec![(BOUND_AT, 2)]);
    r.fields.clear();
    let err = p
        .bind_records(ResourceId(0), vec![r], &registry)
        .unwrap_err();
    assert!(err.contains("undeclared arena entry"), "{err}");
    // The table field must point at the table's base.
    let mut r = record(&p, vec![(BOUND_AT, 2)]);
    r.fields[0].interior = 8;
    let err = p
        .bind_records(ResourceId(0), vec![r], &registry)
        .unwrap_err();
    assert!(err.contains("not at its base"), "{err}");
}

#[test]
fn a_scalar_overlapping_a_pointer_field_or_misdeclared_is_refused() {
    let registry = sealed_registry();
    for scalars in [
        // inside the 8-byte pointer field at +0
        vec![(4, 2), (BOUND_AT, 2)],
        vec![(0, 2), (BOUND_AT, 2)],
        // misaligned
        vec![(BOUND_AT + 2, 2), (BOUND_AT, 2)],
        // declared twice
        vec![(BOUND_AT, 2), (BOUND_AT, 3)],
    ] {
        let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
        let before = p.clone();
        let err = bind(&mut p, scalars.clone(), &registry).unwrap_err();
        assert!(
            err.contains("overlaps a pointer field")
                || err.contains("not 4-byte aligned")
                || err.contains("declared twice"),
            "{scalars:?}: {err}"
        );
        assert_eq!(p, before);
    }
    // Past the arena.
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let end = p.resources[0].bytes as u32;
    let err = bind(&mut p, vec![(BOUND_AT, 2), (end, 1)], &registry).unwrap_err();
    assert!(err.contains("out-of-bounds descriptor"), "{err}");
}

#[test]
fn a_written_table_or_pointee_refuses() {
    let registry = sealed_registry();
    // A node writes the table allocation.
    let mut p = program(
        TableBound::Kernarg(BOUND_AT),
        vec![ib::extra_launch(
            "table_writer",
            st::TABLE,
            st::table_binding(),
            &[VmemStore],
        )],
    );
    let before = p.clone();
    let err = bind(&mut p, vec![(BOUND_AT, 2)], &registry).unwrap_err();
    assert!(err.contains("writes the sealed table allocation"), "{err}");
    assert_eq!(p, before);

    // A node writes a pointee allocation (an expert-weight bank).
    let mut p = program(
        TableBound::Kernarg(BOUND_AT),
        vec![ib::extra_launch(
            "weight_writer",
            st::W1 + 0x40,
            st::w1(),
            &[VmemStore],
        )],
    );
    let err = bind(&mut p, vec![(BOUND_AT, 2)], &registry).unwrap_err();
    assert!(
        err.contains("writes the sealed pointee allocation"),
        "{err}"
    );

    // An atomic is a write too.
    let mut p = program(
        TableBound::Kernarg(BOUND_AT),
        vec![ib::extra_launch(
            "weight_atomic",
            st::W0,
            st::w0(),
            &[Atomic],
        )],
    );
    assert!(bind(&mut p, vec![(BOUND_AT, 2)], &registry).is_err());
}

#[test]
fn a_writer_added_by_the_same_or_a_later_bind_refuses() {
    // The pool node's own second template writes a pointee through a record
    // field: planned in the same bind, invisible to the per-record check.
    let registry = sealed_registry();
    let mut f = (*pool_facts(TableBound::Kernarg(BOUND_AT))).clone();
    let FactStatus::Conditional(c) = &mut f.facts.status else {
        unreachable!()
    };
    c.arenas[0].templates.push(ArenaTemplate {
        template: 6,
        pointer_loads: vec![ArenaPointerLoad {
            record_offset: 0,
            classes: vec![VmemStore],
        }],
        sealed: Vec::new(),
        wg_footprints: Vec::new(),
        hidden_grid: Vec::new(),
    });
    let mut pool = launch(&[(0, ARENA)], 8, vec![ptr(0, &[SmemLoad])]);
    pool.facts = Ok(Arc::new(f));
    let mut p = author(
        "gfx1201",
        &[
            pool,
            ib::extra_launch("table", st::TABLE, st::table_binding(), &[SmemLoad]),
            ib::extra_launch("w0", st::W0, st::w0(), &[VmemLoad]),
            ib::extra_launch("w1", st::W1, st::w1(), &[VmemLoad]),
        ],
    );
    let before = p.clone();
    let sealed = record(&p, vec![(BOUND_AT, 2)]);
    let writer = BindingRecord {
        base: 64,
        template: 6,
        fields: vec![BindingField {
            offset: 0,
            resource: res(&p, st::W1),
            interior: 0,
        }],
        scalars: Vec::new(),
    };
    let err = p
        .bind_records(ResourceId(0), vec![sealed, writer], &registry)
        .unwrap_err();
    assert!(
        err.contains("writes the sealed pointee allocation") && err.contains("arena records"),
        "{err}"
    );
    assert_eq!(p, before);
}

#[test]
fn the_window_bind_refuses_a_changed_generation_content_or_reseal() {
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let registry = sealed_registry();
    bind(&mut p, vec![(BOUND_AT, 2)], &registry).unwrap();

    // The bound allocation is at another generation than the seal.
    let w = window(&p, st::GEN + 1);
    let err = window_bind(&w, &registry, &st::MockContent::new()).unwrap_err();
    assert!(matches!(err, SealedTableError::Stale(m) if m.contains("generation")));

    // The live table content differs from the sealed digest.
    let w = window(&p, st::GEN);
    let content = st::MockContent::new();
    content.bytes.lock().unwrap()[0] ^= 1;
    assert!(matches!(
        window_bind(&w, &registry, &content),
        Err(SealedTableError::Modified { .. })
    ));

    // A node writes the table only after the records were bound: the
    // full-program closure still refuses.
    let mut late = p.clone();
    let t = res(&late, st::TABLE);
    late.nodes[1].effects.push(crate::kernel::RegionEffect {
        arg_offset: 0,
        resource: t,
        classes: vec![VmemStore],
    });
    let w = window(&late, st::GEN);
    assert!(matches!(
        window_bind(&w, &registry, &st::MockContent::new()),
        Err(SealedTableError::Closure(m)) if m.contains("sealed table allocation")
    ));
    let mut late = p.clone();
    let w1 = res(&late, st::W1);
    late.nodes[3].effects.push(crate::kernel::RegionEffect {
        arg_offset: 0,
        resource: w1,
        classes: vec![VmemStore],
    });
    let w = window(&late, st::GEN);
    assert!(matches!(
        window_bind(&w, &registry, &st::MockContent::new()),
        Err(SealedTableError::Closure(m)) if m.contains("pointee")
    ));

    // Retired and re-sealed (same bytes, next generation): the records were
    // discharged against the old seal.
    let next = registry.retire(st::TABLE).unwrap();
    registry
        .seal(
            st::table_binding(),
            st::ENTRIES,
            &st::table_bytes(&st::entry_ptrs()),
            &st::pointees(),
            next,
        )
        .unwrap();
    let w = window(&p, next);
    let err = window_bind(&w, &registry, &st::MockContent::new()).unwrap_err();
    assert!(
        matches!(&err, SealedTableError::Stale(m) if m.contains("rebind the records")),
        "{err:?}"
    );
    // A foreign registry (nothing sealed) has no witness to give.
    assert!(matches!(
        window_bind(
            &window(&p, st::GEN),
            &SealedTableRegistry::new(),
            &st::MockContent::new()
        ),
        Err(SealedTableError::NotSealed(_))
    ));
}

#[test]
fn a_receipt_that_does_not_require_the_witness_is_not_discharged() {
    let mut f = (*pool_facts(TableBound::Kernarg(BOUND_AT))).clone();
    let FactStatus::Conditional(c) = &mut f.facts.status else {
        unreachable!()
    };
    c.witnesses = vec!["descriptor_arena".into()];
    let mut pool = launch(&[(0, ARENA)], 8, vec![ptr(0, &[SmemLoad])]);
    pool.facts = Ok(Arc::new(f));
    let mut p = author(
        "gfx1201",
        &[
            pool,
            ib::extra_launch("table", st::TABLE, st::table_binding(), &[SmemLoad]),
            ib::extra_launch("w0", st::W0, st::w0(), &[VmemLoad]),
            ib::extra_launch("w1", st::W1, st::w1(), &[VmemLoad]),
        ],
    );
    let registry = sealed_registry();
    bind(&mut p, vec![(BOUND_AT, 2)], &registry).unwrap();
    let w = window(&p, st::GEN);
    let err = window_bind(&w, &registry, &st::MockContent::new()).unwrap_err();
    assert!(
        matches!(&err, SealedTableError::Node { detail, .. } if detail.contains("does not require")),
        "{err:?}"
    );
}

/// The entry-index analysis accepts a sign-extended index under an unsigned
/// guard, sound only for `bound <= i32::MAX`: a larger bound is refused by the
/// record bind and by the window bind (a kernarg bound and an arena bound).
#[test]
fn a_bound_past_i32_max_is_refused_everywhere() {
    let too_big = MAX_BOUND + 1;
    // The record bind refuses it before any seal is consulted.
    for registry in [SealedTableRegistry::new(), sealed_registry()] {
        let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
        let before = p.clone();
        let err = bind(&mut p, vec![(BOUND_AT, too_big)], &registry).unwrap_err();
        assert!(err.contains("largest sealed index bound"), "{err}");
        assert_eq!(p, before);
    }
    let mut c = program(TableBound::Const(u32::MAX), Vec::new());
    let err = bind(&mut c, Vec::new(), &sealed_registry()).unwrap_err();
    assert!(err.contains("largest sealed index bound"), "{err}");
    // i32::MAX itself is within the cap: it is then just past the entries.
    let mut p = program(TableBound::Kernarg(BOUND_AT), Vec::new());
    let err = bind(&mut p, vec![(BOUND_AT, MAX_BOUND)], &sealed_registry()).unwrap_err();
    assert!(err.contains("exceeds the 4 sealed entries"), "{err}");

    // The table argument case (kernarg bound in the encoded image).
    let fx = st::Fixture::with(&st::record(), too_big, Vec::new(), true);
    assert!(matches!(
        fx.bind(),
        Err(SealedTableError::BoundTooLarge { bound, .. }) if bound == too_big
    ));
    let fx = st::Fixture::with(&st::record(), MAX_BOUND, Vec::new(), true);
    assert!(matches!(
        fx.bind(),
        Err(SealedTableError::BoundExceedsEntries { .. })
    ));
}
