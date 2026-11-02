// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! GPU-free tests of the Raon railgun decode owner: admission, key, record
//! construction, authoring + sealed-table bind, and the execution/quarantine
//! state machine over a mock queue. A GPU oracle is a separate slice.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use railgun::kernel::{FactStatus, KernelFacts, KernelRef, NodeFacts};
use railgun::window::{QueueFailure, ResolvedStep};
use railgun_corpus::receipt::{
    self, ArgRecord, Differential, GuardRecord, ImplicitRecord, IndexedEffectRecord, KernelRecord,
    SealedAccessRecord, SealedEntryRecord, SealedTableRecord, SideRecord, WITNESS_INDEXED_BIND,
};
use rdna_compute::plan_sink::PlannedKernel;

use super::*;

// ------------------------------------------------------------------ fixtures

const TABLE: u64 = 0x7400_0000;
const TABLE_BYTES: u64 = 0x1000;
const OUT: u64 = 0x7600_0000;
const OUT_BYTES: u64 = 0x1000;
const W0: u64 = 0x7500_0000;
const W1: u64 = 0x7510_0000;
const W_BYTES: u64 = 0x1_0000;
const SEGMENT: u32 = 24;
const OBJECT: &str = "abababababababababababababababababababababababababababababababab";
const ELF: &str = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
const CACHE: &str = "th=0 scope=0 glc=0 slc=0 dlc=0 nv=0";

fn bind(base: u64, bytes: u64) -> Binding {
    Binding { base, bytes }
}

fn entry_ptrs() -> [u64; 4] {
    [W0, W0 + 0x100, W1 + 8, W1 + W_BYTES - 8]
}

fn table_bytes(ptrs: &[u64]) -> Vec<u8> {
    ptrs.iter().flat_map(|p| p.to_le_bytes()).collect()
}

fn arg_record(
    offset: u32,
    size: u32,
    role: &str,
    mode: Option<&str>,
    classes: &[&str],
) -> ArgRecord {
    ArgRecord {
        offset,
        size,
        index: None,
        name: format!("a{offset}"),
        value_kind: if role == "pointer" {
            "global_buffer".into()
        } else {
            "by_value".into()
        },
        role: role.into(),
        mode: mode.map(str::to_owned),
        bound: None,
        classes: classes.iter().map(|c| (*c).to_owned()).collect(),
        paths: Vec::new(),
        cache_bits: Vec::new(),
        accesses: 0,
        shared: false,
        read_cache: "scalar_and_vmem".into(),
        sinks: Vec::new(),
        loaded: true,
    }
}

fn sealed_record(null_guarded: bool) -> SealedTableRecord {
    SealedTableRecord {
        table_arg_offset: 0,
        bound_kind: receipt::SEALED_BOUND_KERNARG.into(),
        bound_value: 16,
        entry_bytes: 8,
        entries: vec![SealedEntryRecord {
            pc: 0x10,
            class: "smem_load".into(),
            path: "smem".into(),
            cache: CACHE.into(),
            width: 8,
            index_expr: "karg32:20".into(),
            guard_pcs: vec![0x8],
        }],
        accesses: vec![SealedAccessRecord {
            pc: 0x20,
            class: "vmem_load".into(),
            path: "vmem".into(),
            cache: CACHE.into(),
            width: 4,
            entry_pcs: vec![0x10],
        }],
        null_guarded,
    }
}

/// `table_gather(table +0, out +8, n +16, pad +20)`, a sealed-only conditional.
fn gather_record(null_guarded: bool) -> KernelRecord {
    let mut k = KernelRecord {
        symbol: "table_gather".into(),
        status: receipt::STATUS_CONDITIONAL.into(),
        unresolved: Vec::new(),
        kernarg_size: SEGMENT,
        read_cache: "scalar_and_vmem".into(),
        dynamic_kernarg_reads: false,
        taint_overflow: false,
        side: SideRecord::default(),
        implicit: ImplicitRecord::default(),
        args: vec![
            arg_record(0, 8, "pointer", Some("read"), &["smem_load"]),
            arg_record(8, 8, "pointer", Some("write"), &["vmem_store"]),
            arg_record(16, 4, "integer", None, &[]),
            arg_record(20, 4, "integer", None, &[]),
        ],
        dwords: Vec::new(),
        differential: Differential {
            source: "none".into(),
            status: "pass".into(),
            args: Vec::new(),
            radiowave_mutable_read_cache: None,
            guards: Vec::new(),
            guarded: true,
        },
        pm_obligations: BTreeMap::new(),
        pm_analyze_error: None,
        unguarded_unresolved: Vec::new(),
        guards: Vec::new(),
        indexed_effects: Vec::new(),
        sealed_tables: vec![sealed_record(null_guarded)],
        required_witnesses: vec![receipt::WITNESS_SEALED_TABLE.into()],
        case_digest: String::new(),
        guard_flag: None,
    };
    k.case_digest = receipt::case_digest(OBJECT, ELF, "gfx1201", &k);
    k
}

/// The gather kernel that also needs an indexed-bind witness this owner never
/// holds.
fn mixed_record() -> KernelRecord {
    let mut k = gather_record(false);
    k.guards = vec![GuardRecord {
        kind: "kernarg_eq".into(),
        arg_offset: 20,
        size: 4,
        value: "0".into(),
    }];
    k.differential.guards = k.guards.clone();
    k.indexed_effects = vec![IndexedEffectRecord {
        pc: 0x40,
        base_arg_offset: 8,
        mode: "write".into(),
        class: "vmem_store".into(),
        path: "vmem".into(),
        cache: CACHE.into(),
        width: 4,
        address_expr: "out + 4*lane".into(),
        index_loads: Vec::new(),
        guards: vec!["kernarg_eq@20".into()],
    }];
    k.required_witnesses = vec![
        WITNESS_INDEXED_BIND.into(),
        receipt::WITNESS_SEALED_TABLE.into(),
    ];
    k.case_digest = receipt::case_digest(OBJECT, ELF, "gfx1201", &k);
    k
}

fn node_facts(k: &KernelRecord, tier: &str) -> Arc<NodeFacts> {
    Arc::new(NodeFacts {
        kref: KernelRef {
            module: "m".into(),
            symbol: k.symbol.clone(),
            artifact_sha256: OBJECT.into(),
            kd_sha256: String::new(),
            receipt_sha256: "ef".repeat(32),
            tier: tier.into(),
            obligations: vec!["InBoundsNoAlias".into()],
        },
        facts: KernelFacts::from_record(k).expect("fixture facts"),
    })
}

fn gather_launch(facts: Result<Arc<NodeFacts>, String>, table: u64, n: u32) -> LaunchRecord {
    let mut kernarg = vec![0u8; SEGMENT as usize];
    kernarg[0..8].copy_from_slice(&table.to_le_bytes());
    kernarg[8..16].copy_from_slice(&OUT.to_le_bytes());
    kernarg[16..20].copy_from_slice(&n.to_le_bytes());
    LaunchRecord {
        symbol: "table_gather".into(),
        grid: [4, 1, 1],
        block: [64, 1, 1],
        dynamic_lds: 0,
        kernarg,
        declared: Vec::new(),
        facts,
        allocations: vec![
            (0, Ok(bind(TABLE, TABLE_BYTES))),
            (8, Ok(bind(OUT, OUT_BYTES))),
        ],
    }
}

/// A fake device: a table allocation whose content is `bytes`, two weight
/// allocations, an output. `reads` counts table reads.
struct FakeDevice {
    bytes: Vec<u8>,
    fail_read: Option<String>,
    reads: AtomicUsize,
}

impl FakeDevice {
    fn new(ptrs: &[u64]) -> Self {
        Self {
            bytes: table_bytes(ptrs),
            fail_read: None,
            reads: AtomicUsize::new(0),
        }
    }
}

impl DeviceView for FakeDevice {
    fn probe(&self, addr: u64) -> Result<Binding, String> {
        [
            bind(W0, W_BYTES),
            bind(W1, W_BYTES),
            bind(TABLE, TABLE_BYTES),
            bind(OUT, OUT_BYTES),
        ]
        .into_iter()
        .find(|b| addr >= b.base && addr < b.base + b.bytes)
        .ok_or_else(|| format!("{addr:#x} is not in a live allocation"))
    }
    fn read(&self, allocation: Binding, len: usize) -> Result<Vec<u8>, String> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if let Some(e) = &self.fail_read {
            return Err(e.clone());
        }
        assert_eq!(allocation.base, TABLE);
        Ok(self.bytes[..len].to_vec())
    }
}

fn table() -> CacheTable {
    CacheTable::for_arch(ARCH).expect("gfx1201 cache table")
}

// ----------------------------------------------------------------- admission

fn ok_facts() -> Admission<'static> {
    Admission {
        arch: ARCH,
        whole_band: true,
        band: "whole",
        graph_forward_disabled: false,
        graph_mode: true,
        retained_replay: false,
        host_weights: false,
        tap_armed: false,
        dump_armed: false,
        absorbed_route: false,
        warmed: true,
        corpus_dir: Some("corpus"),
    }
}

#[test]
fn a_clean_context_is_admitted() {
    assert_eq!(check_admission(&ok_facts()), Ok(()));
}

#[test]
fn every_cheap_admission_check_declines_with_its_exact_reason() {
    use RaonRailgunRefusal as R;
    type Mutate = fn(&mut Admission<'static>);
    let cases: Vec<(&str, Mutate, R)> = vec![
        (
            "arch",
            |a| a.arch = "gfx1100",
            R::WrongArch {
                arch: "gfx1100".into(),
            },
        ),
        (
            "band",
            |a| {
                a.whole_band = false;
                a.band = "pp 0/2";
            },
            R::PartitionedBand {
                band: "pp 0/2".into(),
            },
        ),
        (
            "graph=0",
            |a| a.graph_forward_disabled = true,
            R::GraphDisabled,
        ),
        ("replay", |a| a.retained_replay = true, R::RetainedReplay),
        ("host weights", |a| a.host_weights = true, R::HostWeights),
        ("eager route", |a| a.graph_mode = false, R::NotGraphMode),
        ("census tap", |a| a.tap_armed = true, R::TapArmed),
        ("dumps", |a| a.dump_armed = true, R::DumpArmed),
        ("absorbed", |a| a.absorbed_route = true, R::AbsorbedRoute),
        ("cold graph", |a| a.warmed = false, R::NotWarm),
        ("no corpus", |a| a.corpus_dir = None, R::CorpusUnset),
        ("empty corpus", |a| a.corpus_dir = Some(""), R::CorpusUnset),
    ];
    for (name, mutate, want) in cases {
        let mut a = ok_facts();
        mutate(&mut a);
        assert_eq!(check_admission(&a), Err(want), "{name}");
    }
}

#[test]
fn the_first_failing_admission_check_wins() {
    let mut a = ok_facts();
    a.arch = "gfx1100";
    a.tap_armed = true;
    a.warmed = false;
    assert_eq!(
        check_admission(&a),
        Err(RaonRailgunRefusal::WrongArch {
            arch: "gfx1100".into()
        })
    );
}

#[test]
fn every_refusal_has_a_distinct_kind_and_a_message() {
    use RaonRailgunRefusal as R;
    let all = vec![
        R::Disabled,
        R::WrongArch { arch: "x".into() },
        R::PartitionedBand { band: "x".into() },
        R::GraphDisabled,
        R::NotGraphMode,
        R::RetainedReplay,
        R::HostWeights,
        R::TapArmed,
        R::DumpArmed,
        R::AbsorbedRoute,
        R::NotWarm,
        R::CorpusUnset,
        R::Corpus("x".into()),
        R::Plan("x".into()),
        R::NoReceipt {
            node: 0,
            symbol: "s".into(),
            why: "w".into(),
        },
        R::NotTierD {
            node: 0,
            symbol: "s".into(),
            tier: "STUB".into(),
        },
        R::DeclaredWords { node: 3 },
        R::UntypedCopy {
            node: 1,
            what: "w".into(),
        },
        R::NonPm4 {
            node: 0,
            symbol: "s".into(),
            lowering: "l".into(),
        },
        R::SealedTable("x".into()),
        R::CacheTable("x".into()),
        R::Segments(2),
        R::Lower("x".into()),
        R::SelfCheck("x".into()),
    ];
    let kinds: BTreeSet<_> = all.iter().map(R::kind).collect();
    assert_eq!(kinds.len(), all.len());
    assert!(all.iter().all(|r| !r.to_string().is_empty()));
}

// ----------------------------------------------------------------------- key

fn base_key() -> BaseKey {
    BaseKey {
        binding: 0xfeed,
        mapped_rows: 4096,
        mirror_rows: 2048,
        absorbed: false,
        scratch_rows: 64,
        scratch_epoch: 7,
        model_digest: "model".into(),
        flags_digest: "flags".into(),
    }
}

fn key() -> RaonRailgunKey {
    RaonRailgunKey {
        base: base_key(),
        object_digests: vec![("a".into(), "11".into()), ("b".into(), "22".into())],
        table_generations: vec![(TABLE, 1, "cc".into())],
    }
}

#[test]
fn an_identical_key_has_no_difference() {
    assert_eq!(key(), key());
    assert_eq!(key().first_difference(&key()), None);
}

#[test]
fn every_key_field_invalidates_and_is_named() {
    type Mutate = fn(&mut RaonRailgunKey);
    let cases: Vec<(&str, Mutate)> = vec![
        ("binding", |k| k.base.binding ^= 1),
        ("mapped_rows", |k| k.base.mapped_rows += 1),
        ("mirror_rows", |k| k.base.mirror_rows += 1),
        ("route", |k| k.base.absorbed = true),
        ("scratch_rows", |k| k.base.scratch_rows += 1),
        ("scratch_epoch", |k| k.base.scratch_epoch += 1),
        ("model_digest", |k| k.base.model_digest.push('x')),
        ("flags_digest", |k| k.base.flags_digest.push('x')),
        ("object_digests", |k| k.object_digests[1].1 = "23".into()),
        ("object_digests", |k| {
            k.object_digests.push(("c".into(), "33".into()))
        }),
        ("table_generations", |k| k.table_generations[0].1 = 2),
        ("table_generations", |k| {
            k.table_generations[0].2 = "cd".into()
        }),
        ("table_generations", |k| k.table_generations.clear()),
    ];
    for (field, mutate) in cases {
        let mut k = key();
        mutate(&mut k);
        assert_ne!(k, key(), "{field}");
        assert_eq!(key().first_difference(&k), Some(field), "{field}");
        assert_eq!(k.first_difference(&key()), Some(field), "{field} (sym)");
    }
}

// ------------------------------------------------------------ launch records

fn planned_kernel(symbol: &str) -> PlannedKernel {
    PlannedKernel {
        module: "m".into(),
        symbol: symbol.into(),
        code_object_digest: [0x5a; 32],
        grid: [2, 1, 1],
        block: [64, 1, 1],
        dynamic_lds: 0,
        kernarg: {
            let mut k = vec![0u8; 24];
            k[0..8].copy_from_slice(&TABLE.to_le_bytes());
            k[8..16].copy_from_slice(&OUT.to_le_bytes());
            k
        },
        patches: Vec::new(),
        label: "test",
    }
}

fn records(
    ops: &[PlannedOp],
    declared: &[usize],
    dev: &FakeDevice,
) -> Result<Vec<LaunchRecord>, RaonRailgunRefusal> {
    launch_records(
        ops,
        declared,
        "copysha",
        "fillsha",
        &mut |_, _| Err("no corpus in this test".into()),
        &|a| dev.probe(a),
    )
}

#[test]
fn declared_words_decline_the_program() {
    let dev = FakeDevice::new(&entry_ptrs());
    let ops = [PlannedOp::Kernel(planned_kernel("k"))];
    assert_eq!(
        records(&ops, &[0], &dev).err(),
        Some(RaonRailgunRefusal::DeclaredWords { node: 0 })
    );
}

#[test]
fn copies_and_fills_the_typed_nodes_cannot_express_decline() {
    let dev = FakeDevice::new(&entry_ptrs());
    let untyped = |op: PlannedOp| match records(&[op], &[], &dev) {
        Err(RaonRailgunRefusal::UntypedCopy { node: 0, .. }) => true,
        other => panic!("expected UntypedCopy, got {:?}", other.map(|r| r.len())),
    };
    assert!(untyped(PlannedOp::Copy {
        dst: OUT,
        src: W0,
        bytes: 0
    }));
    assert!(untyped(PlannedOp::Copy {
        dst: OUT,
        src: W0,
        bytes: u64::from(u32::MAX) + 1
    }));
    assert!(untyped(PlannedOp::Fill {
        dst: OUT + 2,
        value: 0,
        words: 4
    }));
    assert!(untyped(PlannedOp::Fill {
        dst: OUT,
        value: 0,
        words: 0
    }));
}

#[test]
fn typed_copy_and_fill_become_transport_launch_records() {
    let dev = FakeDevice::new(&entry_ptrs());
    let seen = Cell::new(Vec::new());
    let r = launch_records(
        &[
            PlannedOp::Copy {
                dst: OUT,
                src: W0,
                bytes: 64,
            },
            PlannedOp::Fill {
                dst: OUT,
                value: 7,
                words: 4,
            },
        ],
        &[],
        "copysha",
        "fillsha",
        &mut |sha, sym| {
            let mut v = seen.take();
            v.push((sha.to_owned(), sym.to_owned()));
            seen.set(v);
            Err("none".into())
        },
        &|a| dev.probe(a),
    )
    .expect("records");
    assert_eq!(r[0].symbol, railgun::copy::COPY_SYMBOL);
    assert_eq!(r[1].symbol, railgun::copy::FILL_SYMBOL);
    assert_eq!(
        seen.take(),
        vec![
            ("copysha".to_owned(), railgun::copy::COPY_SYMBOL.to_owned()),
            ("fillsha".to_owned(), railgun::copy::FILL_SYMBOL.to_owned()),
        ]
    );
    assert_eq!(r[0].allocations.len(), 2);
    assert_eq!(r[1].allocations.len(), 1);
}

#[test]
fn a_kernel_without_facts_keeps_the_lookup_error_for_the_refusal() {
    let dev = FakeDevice::new(&entry_ptrs());
    let r = records(&[PlannedOp::Kernel(planned_kernel("k"))], &[], &dev).expect("records");
    assert_eq!(
        r[0].facts.as_ref().err().map(String::as_str),
        Some("no corpus in this test")
    );
    assert!(r[0].allocations.is_empty());
}

#[test]
fn a_kernel_with_facts_probes_every_nonzero_pointer_argument() {
    let dev = FakeDevice::new(&entry_ptrs());
    let facts = node_facts(&gather_record(false), "D");
    let r = launch_records(
        &[PlannedOp::Kernel(planned_kernel("table_gather"))],
        &[],
        "c",
        "f",
        &mut |_, _| Ok(facts.clone()),
        &|a| dev.probe(a),
    )
    .expect("records");
    assert_eq!(
        r[0].allocations,
        vec![
            (0, Ok(bind(TABLE, TABLE_BYTES))),
            (8, Ok(bind(OUT, OUT_BYTES)))
        ]
    );
}

// ------------------------------------------------------------------ authoring

fn authored(launches: &[LaunchRecord], dev: &FakeDevice) -> Result<Authored, RaonRailgunRefusal> {
    author_and_plan(launches, &table(), dev)
}

#[test]
fn a_node_without_a_receipt_declines_with_the_lookup_error() {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = gather_launch(Err("receipt_missing: no record".into()), TABLE, 4);
    assert_eq!(
        authored(&[l], &dev).err(),
        Some(RaonRailgunRefusal::NoReceipt {
            node: 0,
            symbol: "table_gather".into(),
            why: "receipt_missing: no record".into()
        })
    );
}

#[test]
fn a_receipt_below_tier_d_declines() {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = gather_launch(Ok(node_facts(&gather_record(false), "STUB")), TABLE, 4);
    assert_eq!(
        authored(&[l], &dev).err(),
        Some(RaonRailgunRefusal::NotTierD {
            node: 0,
            symbol: "table_gather".into(),
            tier: "STUB".into()
        })
    );
}

#[test]
fn a_sealed_table_node_is_sealed_bound_witnessed_and_planned() {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 4);
    let a = authored(&[l], &dev).expect("authored");
    assert_eq!(a.program.nodes.len(), 1);
    assert!(a.program.nodes[0].lowering.is_pm4());
    assert_eq!(a.discharged.len(), 1);
    assert_eq!(a.table_bases, vec![TABLE]);
    assert_eq!(a.registry.table_at(TABLE).expect("sealed").generation, 1);
    assert_eq!(a.images.len(), 1);
    assert_eq!(a.grids, vec![[4, 1, 1]]);
    assert_eq!(dev.reads.load(Ordering::SeqCst) > 0, true);
}

#[test]
fn a_zero_bound_table_is_vacuous_and_reads_nothing() {
    let mut dev = FakeDevice::new(&entry_ptrs());
    dev.fail_read = Some("a vacuous bound must not read the table".into());
    let l = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 0);
    let a = authored(&[l], &dev).expect("vacuous bind");
    assert!(a.table_bases.is_empty());
    assert!(a.registry.is_empty());
    assert_eq!(a.discharged.len(), 1);
    assert_eq!(dev.reads.load(Ordering::SeqCst), 0);
}

#[test]
fn a_sealed_bind_failure_declines_and_leaves_nothing_behind() {
    let mut dev = FakeDevice::new(&entry_ptrs());
    dev.fail_read = Some("device lost".into());
    let l = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 4);
    match authored(&[l.clone()], &dev) {
        Err(RaonRailgunRefusal::SealedTable(e)) => assert!(e.contains("device lost"), "{e}"),
        other => panic!(
            "expected a sealed-table refusal, got {:?}",
            other.map(|_| ())
        ),
    }
    // The records are untouched and the same records author fine once the
    // table reads back: the refusal mutated nothing.
    dev.fail_read = None;
    assert!(authored(&[l], &dev).is_ok());
}

#[test]
fn null_entries_need_a_proved_null_guard() {
    let nullable = [W0, 0, W1 + 8, 0];
    let dev = FakeDevice::new(&nullable);
    let unguarded = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 4);
    match authored(&[unguarded], &dev) {
        Err(RaonRailgunRefusal::SealedTable(e)) => assert!(e.contains("null"), "{e}"),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
    let guarded = gather_launch(Ok(node_facts(&gather_record(true), "D")), TABLE, 4);
    assert!(authored(&[guarded], &dev).is_ok());
}

#[test]
fn an_entry_outside_any_live_allocation_declines() {
    let dev = FakeDevice::new(&[W0, 0xdead_0000, W1, W1 + 8]);
    let l = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 4);
    assert!(matches!(
        authored(&[l], &dev),
        Err(RaonRailgunRefusal::SealedTable(_))
    ));
}

#[test]
fn a_table_argument_inside_its_allocation_declines() {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE + 8, 4);
    match authored(&[l], &dev) {
        Err(RaonRailgunRefusal::SealedTable(e)) => assert!(e.contains("bytes into"), "{e}"),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn a_node_that_still_needs_a_witness_declines_as_non_pm4() {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = gather_launch(Ok(node_facts(&mixed_record(), "D")), TABLE, 4);
    match authored(&[l], &dev) {
        Err(RaonRailgunRefusal::NonPm4 {
            node: 0, symbol, ..
        }) => {
            assert_eq!(symbol, "table_gather")
        }
        other => panic!("expected NonPm4, got {:?}", other.map(|_| ())),
    }
    // Declined before the device was asked anything.
    assert_eq!(dev.reads.load(Ordering::SeqCst), 0);
}

#[test]
fn a_node_with_unknown_facts_declines_as_non_pm4() {
    let dev = FakeDevice::new(&entry_ptrs());
    let mut k = gather_record(false);
    k.status = receipt::STATUS_UNKNOWN.into();
    k.required_witnesses.clear();
    k.sealed_tables.clear();
    k.case_digest = String::new();
    let facts = KernelFacts::from_record(&k).expect("facts");
    assert!(!matches!(facts.status, FactStatus::Proven));
    let l = gather_launch(
        Ok(Arc::new(NodeFacts {
            kref: node_facts(&gather_record(false), "D").kref.clone(),
            facts,
        })),
        TABLE,
        4,
    );
    assert!(matches!(
        authored(&[l], &dev),
        Err(RaonRailgunRefusal::NonPm4 { .. })
    ));
}

// -------------------------------------------------------------------- state

#[derive(Default)]
struct Log {
    submits: usize,
    waits: usize,
    dropped: usize,
    submit_failure: Option<(&'static str, Quiescence)>,
    wait_failure: Option<(&'static str, Quiescence)>,
}

struct MockQueue(Arc<Mutex<Log>>);

impl Drop for MockQueue {
    fn drop(&mut self) {
        self.0.lock().unwrap().dropped += 1;
    }
}

impl WindowQueue for MockQueue {
    fn submit(
        &mut self,
        nodes: &[NodeImage<'_>],
        steps: &[ResolvedStep],
    ) -> Result<(), QueueFailure> {
        assert!(steps.is_empty(), "the owner submits no typed steps");
        assert_eq!(nodes.len(), 2);
        let mut log = self.0.lock().unwrap();
        if let Some((m, q)) = log.submit_failure {
            return Err(QueueFailure {
                message: m.into(),
                quiescence: q,
            });
        }
        log.submits += 1;
        Ok(())
    }

    fn wait(&mut self) -> Result<(), QueueFailure> {
        let mut log = self.0.lock().unwrap();
        log.waits += 1;
        match log.wait_failure {
            Some((m, q)) => Err(QueueFailure {
                message: m.into(),
                quiescence: q,
            }),
            None => Ok(()),
        }
    }
}

fn built(log: &Arc<Mutex<Log>>, key: RaonRailgunKey) -> Built<MockQueue> {
    Built {
        key,
        queue: MockQueue(log.clone()),
        images: vec![vec![1; 8], vec![2; 8]],
        grids: vec![[1, 1, 1], [2, 1, 1]],
        symbols: vec!["a".into(), "b".into()],
        registry: SealedTableRegistry::new(),
        table_bases: vec![TABLE],
        nodes: 2,
        ib_dwords: 10,
    }
}

fn live_extra(
    key: &RaonRailgunKey,
) -> impl FnMut(&Built<MockQueue>) -> (Vec<(String, String)>, Vec<(u64, u64, String)>) + '_ {
    |_| (key.object_digests.clone(), key.table_generations.clone())
}

#[test]
fn an_empty_slot_misses_and_a_matching_key_hits() {
    let log = Arc::new(Mutex::new(Log::default()));
    let mut exec = Exec::<MockQueue>::default();
    let k = key();
    assert_eq!(
        exec.lookup(&k.base, &mut live_extra(&k)),
        Lookup::Miss(None)
    );
    exec.install(built(&log, k.clone()));
    assert_eq!(exec.lookup(&k.base, &mut live_extra(&k)), Lookup::Hit);
}

#[test]
fn any_live_change_rebuilds_and_names_the_cause() {
    let log = Arc::new(Mutex::new(Log::default()));
    let mut exec = Exec::<MockQueue>::default();
    let k = key();
    exec.install(built(&log, k.clone()));

    let mut grown = k.clone();
    grown.base.mapped_rows *= 2;
    assert_eq!(
        exec.lookup(&grown.base, &mut live_extra(&k)),
        Lookup::Miss(Some("mapped_rows"))
    );
    let mut regrown = k.clone();
    regrown.base.scratch_epoch += 1;
    assert_eq!(
        exec.lookup(&regrown.base, &mut live_extra(&k)),
        Lookup::Miss(Some("scratch_epoch"))
    );
    let mut swapped = k.clone();
    swapped.object_digests[0].1 = "99".into();
    assert_eq!(
        exec.lookup(&k.base, &mut live_extra(&swapped)),
        Lookup::Miss(Some("object_digests"))
    );
    let mut resealed = k.clone();
    resealed.table_generations[0].1 += 1;
    assert_eq!(
        exec.lookup(&k.base, &mut live_extra(&resealed)),
        Lookup::Miss(Some("table_generations"))
    );
}

#[test]
fn a_refused_binding_is_remembered_until_the_binding_changes() {
    let mut exec = Exec::<MockQueue>::default();
    let k = key();
    exec.decline(k.base.clone(), RaonRailgunRefusal::NotWarm);
    assert_eq!(
        exec.lookup(&k.base, &mut live_extra(&k)),
        Lookup::Declined(RaonRailgunRefusal::NotWarm)
    );
    let mut moved = k.base.clone();
    moved.binding += 1;
    assert_eq!(exec.lookup(&moved, &mut live_extra(&k)), Lookup::Miss(None));
}

#[test]
fn a_completed_submit_reports_completed_and_keeps_the_program() {
    let log = Arc::new(Mutex::new(Log::default()));
    let mut exec = Exec::<MockQueue>::default();
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Completed);
    assert_eq!(exec.submit(), StepOutcome::Completed);
    let l = log.lock().unwrap();
    assert_eq!((l.submits, l.waits, l.dropped), (2, 2, 0));
    assert!(matches!(exec.slot, Slot::Ready(_)));
}

#[test]
fn submitting_with_no_program_fails_without_quarantine() {
    let mut exec = Exec::<MockQueue>::default();
    assert!(matches!(exec.submit(), StepOutcome::Failed(_)));
    assert!(!exec.is_quarantined());
}

#[test]
fn a_proven_failure_aborts_the_request_and_drops_the_program() {
    let log = Arc::new(Mutex::new(Log::default()));
    log.lock().unwrap().submit_failure = Some(("device fault", Quiescence::Proven));
    let mut exec = Exec::<MockQueue>::default();
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Failed("device fault".into()));
    assert!(!exec.is_quarantined());
    assert!(matches!(exec.slot, Slot::Empty));
    assert_eq!(
        log.lock().unwrap().dropped,
        1,
        "a quiescent program is freed"
    );
}

#[test]
fn a_failure_reported_by_wait_is_classified_the_same_way() {
    let log = Arc::new(Mutex::new(Log::default()));
    log.lock().unwrap().wait_failure = Some(("hang", Quiescence::Unknown));
    let mut exec = Exec::<MockQueue>::default();
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Quarantined("hang".into()));
}

#[test]
fn a_lowering_self_check_failure_is_the_one_decline_after_submit() {
    let log = Arc::new(Mutex::new(Log::default()));
    log.lock().unwrap().submit_failure = Some((
        "lowering self-check: node 3 grid mismatch",
        Quiescence::Proven,
    ));
    let mut exec = Exec::<MockQueue>::default();
    let k = key();
    exec.install(built(&log, k.clone()));
    let want = RaonRailgunRefusal::SelfCheck("lowering self-check: node 3 grid mismatch".into());
    assert_eq!(exec.submit(), StepOutcome::Declined(want.clone()));
    assert!(!exec.is_quarantined());
    // Cached for the binding: not rebuilt and resubmitted per token.
    assert_eq!(
        exec.lookup(&k.base, &mut live_extra(&k)),
        Lookup::Declined(want)
    );
    assert_eq!(log.lock().unwrap().submits, 0);
}

#[test]
fn unknown_quiescence_quarantines_and_never_frees_the_program() {
    let log = Arc::new(Mutex::new(Log::default()));
    log.lock().unwrap().submit_failure = Some(("timeout", Quiescence::Unknown));
    let mut exec = Exec::<MockQueue>::default();
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Quarantined("timeout".into()));
    assert!(exec.is_quarantined());
    assert_eq!(exec.quarantine_reason(), Some("timeout"));
    // Every later submit is refused without touching the queue.
    log.lock().unwrap().submit_failure = None;
    match exec.submit() {
        StepOutcome::Failed(m) => assert!(m.contains("quarantined"), "{m}"),
        other => panic!("expected Failed, got {other:?}"),
    }
    assert_eq!(log.lock().unwrap().submits, 0);
    // Dropping the machine leaks the possibly-executing queue.
    drop(exec);
    assert_eq!(log.lock().unwrap().dropped, 0);
}

#[test]
fn replacing_a_quiescent_program_drops_the_old_one() {
    let log = Arc::new(Mutex::new(Log::default()));
    let mut exec = Exec::<MockQueue>::default();
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Completed);
    exec.clear();
    assert_eq!(log.lock().unwrap().dropped, 1);
    exec.install(built(&log, key()));
    assert_eq!(exec.submit(), StepOutcome::Completed);
}
