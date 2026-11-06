// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only tests of the sealed-table certificate runtime: the receipt parse,
//! `author`, the registry seal/retire, the bind and the manifest admission of
//! its witness. The window-side tests (consume, submit, lease) are in
//! `window::tests`, next to the indexed-witness ones. The fixture shape (a
//! kernel that reads `table[i]` and only reads through it) is model-agnostic.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use railgun_corpus::receipt::{
    self, ArgRecord, Differential, GuardRecord, ImplicitRecord, IndexedEffectRecord, KernelRecord,
    SealedAccessRecord, SealedEntryRecord, SealedTableRecord, SideRecord, WITNESS_INDEXED_BIND,
};
use railgun_corpus::window::{
    sequence_digest, ConditionalWitness, NodeRoute, RuntimeNode, WindowManifest, WindowNodeRecord,
    WHOLE16_SHAPE, WINDOW_MANIFEST_SCHEMA, WINDOW_MANIFEST_VERSION,
};

use super::*;
use crate::index_bind::tests as ib;
use crate::kernel::{
    author, ConditionalFacts, Expr, FactStatus, KernelFacts, KernelRef, LaunchRecord, MemClass,
    NodeFacts,
};
use crate::plan::{plan, CacheTable, RowPolicy};
use crate::window::{CopyStep, FillStep, Region};
use crate::ResourceId;

pub(crate) const TABLE: u64 = 0x7400_0000;
pub(crate) const TABLE_BYTES: u64 = 0x1000;
pub(crate) const OUT: u64 = 0x7600_0000;
pub(crate) const OUT_BYTES: u64 = 0x1000;
pub(crate) const W0: u64 = 0x7500_0000;
pub(crate) const W1: u64 = 0x7510_0000;
pub(crate) const W_BYTES: u64 = 0x1_0000;
pub(crate) const ENTRIES: u32 = 4;
pub(crate) const GEN: u64 = 3;
pub(crate) const SEGMENT: u32 = 24;
const OBJECT: &str = "abababababababababababababababababababababababababababababababab";
const ELF: &str = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
const CACHE: &str = "th=0 scope=0 glc=0 slc=0 dlc=0 nv=0";

pub(crate) fn entry_ptrs() -> [u64; 4] {
    [W0, W0 + 0x100, W1 + 8, W1 + W_BYTES - 8]
}

pub(crate) fn table_bytes(ptrs: &[u64]) -> Vec<u8> {
    ptrs.iter().flat_map(|p| p.to_le_bytes()).collect()
}

pub(crate) fn w0() -> Binding {
    Binding {
        base: W0,
        bytes: W_BYTES,
    }
}

pub(crate) fn w1() -> Binding {
    Binding {
        base: W1,
        bytes: W_BYTES,
    }
}

pub(crate) fn table_binding() -> Binding {
    Binding {
        base: TABLE,
        bytes: TABLE_BYTES,
    }
}

pub(crate) fn pointees() -> Vec<Pointee> {
    vec![Pointee::immutable(w0()), Pointee::immutable(w1())]
}

pub(crate) fn seal_default(
    registry: &SealedTableRegistry,
) -> Result<SealedTable, SealedTableError> {
    registry.seal(
        table_binding(),
        ENTRIES,
        &table_bytes(&entry_ptrs()),
        &pointees(),
        GEN,
    )
}

// ------------------------------------------------------------------ records

pub(crate) fn arg_record(
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

pub(crate) fn sealed_record(bound_kind: &str, bound_value: u64) -> SealedTableRecord {
    SealedTableRecord {
        table_arg_offset: 0,
        bound_kind: bound_kind.into(),
        bound_value,
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
        null_guarded: false,
    }
}

fn digest_of(mut k: KernelRecord) -> KernelRecord {
    k.case_digest = receipt::case_digest(OBJECT, ELF, "gfx1201", &k);
    k
}

/// A sealed-only conditional record: `table_gather(table +0, out +8, n +16,
/// pad +20)`; guards empty, `required_witnesses = [sealed_table]`.
pub(crate) fn record_with(tables: Vec<SealedTableRecord>) -> KernelRecord {
    digest_of(KernelRecord {
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
        sealed_tables: tables,
        descriptor_arenas: Vec::new(),
        required_witnesses: vec![receipt::WITNESS_SEALED_TABLE.into()],
        case_digest: String::new(),
        guard_flag: None,
        task_protocol: None,
    })
}

pub(crate) fn record() -> KernelRecord {
    record_with(vec![sealed_record(receipt::SEALED_BOUND_KERNARG, 16)])
}

/// [`record`] whose table certificate proves a null guard.
pub(crate) fn guarded_record() -> KernelRecord {
    let mut t = sealed_record(receipt::SEALED_BOUND_KERNARG, 16);
    t.null_guarded = true;
    record_with(vec![t])
}

/// A table whose entries 1 and 3 are null (an absent expert).
pub(crate) fn nullable_ptrs() -> [u64; 4] {
    [W0, 0, W1 + 8, 0]
}

/// The same kernel with an `indexed_bind` case as well (a guarded write of
/// `out`), so a node needs both witnesses.
pub(crate) fn mixed_record() -> KernelRecord {
    let mut k = record();
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
    digest_of(k)
}

pub(crate) fn node_facts(k: &KernelRecord) -> Result<Arc<NodeFacts>, String> {
    Ok(Arc::new(NodeFacts {
        kref: KernelRef {
            module: "m".into(),
            symbol: k.symbol.clone(),
            artifact_sha256: OBJECT.into(),
            kd_sha256: String::new(),
            receipt_sha256: "ef".repeat(32),
            tier: "D".into(),
            obligations: vec!["InBoundsNoAlias".into()],
        },
        facts: KernelFacts::from_record(k)?,
    }))
}

pub(crate) fn gather_launch(facts: Arc<NodeFacts>, table: u64, n: u32) -> LaunchRecord {
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
        facts: Ok(facts),
        allocations: vec![(0, Ok(table_binding())), (8, Ok(out_binding()))],
    }
}

pub(crate) fn out_binding() -> Binding {
    Binding {
        base: OUT,
        bytes: OUT_BYTES,
    }
}

// -------------------------------------------------------------------- bind

/// The owner's view of the live table bytes.
pub(crate) struct MockContent {
    pub bytes: Mutex<Vec<u8>>,
    pub fail: Mutex<Option<String>>,
    pub reads: AtomicUsize,
}

impl MockContent {
    pub fn new() -> Self {
        Self {
            bytes: Mutex::new(table_bytes(&entry_ptrs())),
            fail: Mutex::new(None),
            reads: AtomicUsize::new(0),
        }
    }
}

impl TableContent for MockContent {
    fn read_table(&self, allocation: Binding, entries: usize) -> Result<Vec<u8>, String> {
        assert_eq!(allocation, table_binding(), "the sealed allocation is read");
        self.reads.fetch_add(1, Ordering::Relaxed);
        if let Some(why) = self.fail.lock().unwrap().clone() {
            return Err(why);
        }
        let b = self.bytes.lock().unwrap();
        Ok(b[..(entries * 8).min(b.len())].to_vec())
    }
}

pub(crate) struct Fixture {
    /// As authored: the sealed node is still `Lowering::Conditional`.
    pub authored: KernelProgram,
    /// The program a plan is laid out against, assuming the witness.
    pub program: KernelProgram,
    pub bindings: Vec<Binding>,
    pub allocations: Vec<AllocationIdentity>,
    pub images: Vec<Vec<u8>>,
    pub grids: Vec<[u32; 3]>,
    pub steps: Vec<WindowStep>,
    pub table_res: ResourceId,
    pub out_res: ResourceId,
    pub registry: SealedTableRegistry,
    pub content: MockContent,
    pub case_digest: String,
}

impl Fixture {
    /// The gather node (node 0, `n = 4`) with the default table sealed, then
    /// `extra` launches.
    pub fn new(extra: Vec<LaunchRecord>) -> Self {
        Self::with(&record(), 4, extra, true)
    }

    pub fn with(k: &KernelRecord, n: u32, extra: Vec<LaunchRecord>, seal: bool) -> Self {
        let mut launches = vec![gather_launch(node_facts(k).unwrap(), TABLE, n)];
        launches.extend(extra);
        let authored = author("gfx1201", &launches);
        assert!(
            matches!(&authored.nodes[0].lowering, Lowering::Conditional { reason } if reason.contains("sealed_table")),
            "{:?}",
            authored.nodes[0].lowering
        );
        let program = authored.assume_sealed_table_witness();
        let words = program.authored_words();
        let images = (0..program.nodes.len())
            .map(|i| {
                program.nodes[i].segment_bytes.map_or(Vec::new(), |s| {
                    program.kernarg_image(i, s as usize, &words).unwrap()
                })
            })
            .collect();
        let grids = program.nodes.iter().map(|n| n.grid).collect();
        let bindings = program.resources.clone();
        let allocations = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| AllocationIdentity {
                resource: ResourceId(i as u32),
                base: b.base,
                bytes: b.bytes,
                generation: if b.base == TABLE { GEN } else { 1 },
            })
            .collect();
        let steps = (0..program.nodes.len())
            .map(|node| WindowStep::Kernel { node })
            .collect();
        let res = |base| {
            ResourceId(
                program
                    .resources
                    .iter()
                    .position(|b| b.base == base)
                    .unwrap() as u32,
            )
        };
        let registry = SealedTableRegistry::new();
        if seal {
            seal_default(&registry).unwrap();
        }
        Self {
            table_res: res(TABLE),
            out_res: res(OUT),
            authored,
            program,
            bindings,
            allocations,
            images,
            grids,
            steps,
            registry,
            content: MockContent::new(),
            case_digest: k.case_digest.clone(),
        }
    }

    pub fn bind(&self) -> Result<SealedTableWitness, SealedTableError> {
        self.registry.bind_sealed_table(
            &TableBindInputs {
                window: 1,
                shape_generation: 7,
                program: &self.program,
                steps: &self.steps,
                bindings: &self.bindings,
                allocations: &self.allocations,
                images: &self.images,
                grids: &self.grids,
            },
            &self.content,
        )
    }

    /// The gather node (`n = 4`) over a table holding null entries, sealed
    /// with [`NullEntries::Allow`] (what an owner does for a null-guarded
    /// certificate).
    pub fn nullable(k: &KernelRecord) -> Self {
        let fx = Self::with(k, 4, Vec::new(), false);
        fx.registry
            .seal_with(
                table_binding(),
                ENTRIES,
                &table_bytes(&nullable_ptrs()),
                &pointees(),
                GEN,
                NullEntries::Allow,
            )
            .expect("null entries allowed");
        *fx.content.bytes.lock().unwrap() = table_bytes(&nullable_ptrs());
        fx
    }
}

/// A certified node of `classes` on `binding`, through `target` inside it.
fn toucher(symbol: &str, target: u64, binding: Binding, classes: &[MemClass]) -> LaunchRecord {
    ib::extra_launch(symbol, target, binding, classes)
}

fn closure_reason(e: SealedTableError) -> String {
    match e {
        SealedTableError::Closure(m) => m,
        other => panic!("expected a closure refusal, got {other:?}"),
    }
}

// ------------------------------------------------------------------ the API

/// The API a model owner calls: seal the loader-uploaded pointer table with
/// its immutable pointee allocations, then bind the window's sealed-table
/// nodes against the sealed registry.
#[test]
fn owner_seals_a_pointer_table_and_binds() {
    // Load time: the owner uploaded these bytes into the table allocation
    // (`TABLE`) and the weights into `W0`/`W1`.
    let registry = SealedTableRegistry::new();
    let sealed = registry
        .seal(
            Binding {
                base: TABLE,
                bytes: TABLE_BYTES,
            },
            ENTRIES,
            &table_bytes(&entry_ptrs()),
            &[
                Pointee::immutable(Binding {
                    base: W0,
                    bytes: W_BYTES,
                }),
                Pointee::immutable(Binding {
                    base: W1,
                    bytes: W_BYTES,
                }),
            ],
            GEN,
        )
        .expect("every entry lies inside exactly one registered pointee");
    assert_eq!(
        (sealed.entries, sealed.entry_width, sealed.generation),
        (ENTRIES, 8, GEN)
    );
    assert_eq!(sealed.content_sha256, digest(&table_bytes(&entry_ptrs())));
    assert_eq!(sealed.pointees.len(), 2);
    assert_eq!(registry.table_at(TABLE), Some(sealed.clone()));

    // Window time: the receipt's node stays conditional (non-PM4) until the
    // bind issues the witness, and the bound window then plans as PM4.
    let mut fx = Fixture::new(Vec::new());
    fx.registry = registry;
    assert!(matches!(
        &fx.authored.nodes[0].lowering,
        Lowering::Conditional { reason } if reason == "sealed_table witness not consumed"
    ));
    assert!(fx.authored.nodes[0].indexed.is_none());
    assert_eq!(fx.authored.sealed_table_nodes(), vec![0]);
    assert_eq!(fx.authored.conditional_nodes(), Vec::<usize>::new());
    assert!(fx.program.nodes[0].lowering.is_pm4());

    let w = fx.bind().expect("a sealed table, n <= entries, no writer");
    assert_eq!(
        w.discharged_cases(),
        vec![(0, fx.case_digest.as_str(), "sealed_table")]
    );
    let n = &w.nodes()[0];
    assert_eq!(n.symbol, "table_gather");
    assert_eq!(n.case_digest, fx.case_digest);
    let t = &n.tables[0];
    assert_eq!(
        (t.allocation, t.entries, t.bound, t.generation),
        (table_binding(), ENTRIES, 4, GEN)
    );
    assert_eq!(t.content_sha256, sealed.content_sha256);
    assert_eq!((w.window(), w.shape_generation()), (1, 7));
    assert_eq!(fx.content.reads.load(Ordering::Relaxed), 1);
    fx.registry.check_witness(&w).expect("current");

    // Consumed: the plan lays out as one PM4 segment (the table is an
    // ordinary read of its resource; pointees add no edge).
    let table = CacheTable::for_arch("gfx1201").unwrap();
    let pm4 = plan(&fx.program, &table, RowPolicy::TierD);
    assert!(pm4.is_single_segment(1));
    // The same node unconsumed is a barrier, not PM4.
    let unconsumed = plan(&fx.authored, &table, RowPolicy::TierD);
    assert!(!fx.authored.nodes[0].lowering.is_pm4());
    assert!(!unconsumed.is_single_segment(1));
}

#[test]
fn a_table_bound_equal_to_the_entry_count_is_in_range_and_a_constant_bound_is_checked() {
    for (n, ok) in [(0, true), (3, true), (4, true), (5, false)] {
        let fx = Fixture::with(&record(), n, Vec::new(), true);
        match (fx.bind(), ok) {
            // A zero bound is vacuous: discharged without a bound table.
            (Ok(w), true) if n == 0 => {
                assert!(w.nodes()[0].tables.is_empty());
                assert_eq!(w.nodes()[0].vacuous, vec![0]);
            }
            (Ok(w), true) => assert_eq!(w.nodes()[0].tables[0].bound, n),
            (
                Err(SealedTableError::BoundExceedsEntries {
                    node: 0,
                    table_arg: 0,
                    bound,
                    entries: 4,
                }),
                false,
            ) => assert_eq!(bound, n),
            (other, _) => panic!("n={n}: {other:?}"),
        }
    }
    // A constant bound ignores the launch's scalar.
    for (c, ok) in [(4, true), (5, false)] {
        let k = record_with(vec![sealed_record(receipt::SEALED_BOUND_CONST, c)]);
        let fx = Fixture::with(&k, 999, Vec::new(), true);
        assert_eq!(fx.bind().is_ok(), ok, "const {c}");
    }
}

// --------------------------------------------------------------------- seal

fn seal_err(r: Result<SealedTable, SealedTableError>) -> String {
    match r {
        Err(SealedTableError::Seal(m)) => m,
        other => panic!("expected a seal refusal, got {other:?}"),
    }
}

#[test]
fn seal_refuses_a_pointee_overlapping_the_table_allocation() {
    let r = SealedTableRegistry::new();
    // Wholly inside the table allocation, and straddling its start.
    for bad in [
        Binding {
            base: TABLE + 0x100,
            bytes: 0x100,
        },
        Binding {
            base: TABLE - 0x80,
            bytes: 0x100,
        },
    ] {
        let mut p = pointees();
        p.push(Pointee::immutable(bad));
        let m = seal_err(r.seal(
            table_binding(),
            ENTRIES,
            &table_bytes(&entry_ptrs()),
            &p,
            GEN,
        ));
        assert!(m.contains("overlaps the table allocation"), "{m}");
    }
    assert!(r.is_empty());
}

#[test]
fn seal_refuses_an_entry_outside_every_pointee_and_a_null_entry() {
    let r = SealedTableRegistry::new();
    for (entry, needle) in [
        (0x9000_0000u64, "lies outside every registered pointee"),
        // One past the end of W0, in the gap before W1.
        (W0 + W_BYTES, "lies outside every registered pointee"),
        (W0 - 1, "lies outside every registered pointee"),
        (0, "is null"),
    ] {
        let mut ptrs = entry_ptrs();
        ptrs[2] = entry;
        let m = seal_err(r.seal(
            table_binding(),
            ENTRIES,
            &table_bytes(&ptrs),
            &pointees(),
            GEN,
        ));
        assert!(
            m.contains("entry 2") && m.contains(needle),
            "{entry:#x}: {m}"
        );
    }
    assert!(r.is_empty(), "a refused seal leaves nothing sealed");
}

#[test]
fn seal_refuses_malformed_tables_and_pointee_sets() {
    let r = SealedTableRegistry::new();
    let good = table_bytes(&entry_ptrs());
    let seal = |alloc: Binding, entries: u32, bytes: &[u8], p: &[Pointee], g: u64| {
        seal_err(r.seal(alloc, entries, bytes, p, g))
    };
    assert!(seal(table_binding(), 0, &[], &pointees(), GEN).contains("at least one entry"));
    assert!(seal(table_binding(), ENTRIES, &good[..24], &pointees(), GEN).contains("bytes are 24"));
    assert!(seal(
        Binding {
            base: TABLE,
            bytes: 16
        },
        ENTRIES,
        &good,
        &pointees(),
        GEN
    )
    .contains("smaller than 4 entries"));
    assert!(seal(
        Binding {
            base: TABLE + 4,
            bytes: TABLE_BYTES
        },
        ENTRIES,
        &good,
        &pointees(),
        GEN
    )
    .contains("not 8-byte aligned"));
    assert!(seal(table_binding(), ENTRIES, &good, &[], GEN).contains("no pointee"));
    let mut mutable = pointees();
    mutable[1].immutable = false;
    assert!(seal(table_binding(), ENTRIES, &good, &mutable, GEN).contains("not immutable"));
    let mut overlapping = pointees();
    overlapping.push(Pointee::immutable(Binding {
        base: W0 + 0x8000,
        bytes: W_BYTES,
    }));
    assert!(seal(table_binding(), ENTRIES, &good, &overlapping, GEN).contains("overlap"));
    let mut empty = pointees();
    empty.push(Pointee::immutable(Binding {
        base: 0x7700_0000,
        bytes: 0,
    }));
    assert!(seal(table_binding(), ENTRIES, &good, &empty, GEN).contains("empty"));
    assert!(r.is_empty());
}

#[test]
fn seal_refuses_a_duplicate_or_overlapping_seal_and_retire_raises_the_generation() {
    let r = SealedTableRegistry::new();
    seal_default(&r).unwrap();
    assert!(seal_err(seal_default(&r)).contains("already sealed"));
    let overlap = Binding {
        base: TABLE + 0x800,
        bytes: 0x1000,
    };
    let m = seal_err(r.seal(
        overlap,
        ENTRIES,
        &table_bytes(&entry_ptrs()),
        &pointees(),
        GEN,
    ));
    assert!(m.contains("overlaps the sealed table"), "{m}");

    // Retire bumps the generation: a re-seal at the old one is refused.
    assert_eq!(r.retire(TABLE), Ok(GEN + 1));
    assert!(matches!(
        r.retire(TABLE),
        Err(SealedTableError::NotSealed(_))
    ));
    assert!(r.is_empty());
    let m = seal_err(seal_default(&r));
    assert!(m.contains("below") && m.contains("retired seal"), "{m}");
    let again = r
        .seal(
            table_binding(),
            ENTRIES,
            &table_bytes(&entry_ptrs()),
            &pointees(),
            GEN + 1,
        )
        .unwrap();
    assert_eq!(again.generation, GEN + 1);
}

// -------------------------------------------------------------------- bind

#[test]
fn an_unsealed_table_or_another_extent_is_refused() {
    // Nothing sealed at all.
    let fx = Fixture::with(&record(), 4, Vec::new(), false);
    match fx.bind() {
        Err(SealedTableError::NotSealed(m)) => assert!(m.contains("no sealed table"), "{m}"),
        other => panic!("{other:?}"),
    }
    // Sealed at the same base but a different extent: not this allocation.
    let r = SealedTableRegistry::new();
    r.seal(
        Binding {
            base: TABLE,
            bytes: 0x800,
        },
        ENTRIES,
        &table_bytes(&entry_ptrs()),
        &pointees(),
        GEN,
    )
    .unwrap();
    let mut fx = Fixture::with(&record(), 4, Vec::new(), false);
    fx.registry = r;
    match fx.bind() {
        Err(SealedTableError::WrongAllocation(m)) => {
            assert!(m.contains("sealed at that base"), "{m}")
        }
        other => panic!("{other:?}"),
    }
    // A table sealed elsewhere never matches by address overlap alone.
    let r = SealedTableRegistry::new();
    r.seal(
        Binding {
            base: TABLE + 0x2000,
            bytes: TABLE_BYTES,
        },
        ENTRIES,
        &table_bytes(&entry_ptrs()),
        &pointees(),
        GEN,
    )
    .unwrap();
    let mut fx = Fixture::with(&record(), 4, Vec::new(), false);
    fx.registry = r;
    assert!(matches!(fx.bind(), Err(SealedTableError::NotSealed(_))));
}

#[test]
fn an_interior_table_pointer_in_the_bound_image_is_refused() {
    let mut fx = Fixture::new(Vec::new());
    fx.images[0][0..8].copy_from_slice(&(TABLE + 8).to_le_bytes());
    match fx.bind() {
        Err(SealedTableError::WrongAllocation(m)) => {
            assert!(m.contains("not the sealed allocation base"), "{m}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_modified_unreadable_or_short_table_is_refused() {
    let fx = Fixture::new(Vec::new());
    fx.content.bytes.lock().unwrap()[9] ^= 1;
    assert_eq!(
        fx.bind().unwrap_err(),
        SealedTableError::Modified { base: TABLE }
    );

    let fx = Fixture::new(Vec::new());
    *fx.content.fail.lock().unwrap() = Some("download failed".into());
    assert_eq!(
        fx.bind().unwrap_err(),
        SealedTableError::Content("download failed".into())
    );

    let fx = Fixture::new(Vec::new());
    fx.content.bytes.lock().unwrap().truncate(24);
    assert!(matches!(fx.bind(), Err(SealedTableError::Content(m)) if m.contains("24 bytes")));
}

#[test]
fn a_stale_allocation_generation_is_refused() {
    let mut fx = Fixture::new(Vec::new());
    let i = fx.table_res.0 as usize;
    fx.allocations[i].generation = GEN + 1;
    match fx.bind() {
        Err(SealedTableError::Stale(m)) => assert!(m.contains("generation"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_node_that_writes_the_table_allocation_cannot_obtain_a_witness() {
    for classes in [
        &[MemClass::VmemStore][..],
        &[MemClass::Atomic][..],
        &[MemClass::AtomicCommutative][..],
        &[MemClass::VmemLoad, MemClass::VmemStore][..],
    ] {
        let fx = Fixture::new(vec![toucher("w", TABLE + 0x40, table_binding(), classes)]);
        let m = closure_reason(fx.bind().unwrap_err());
        assert!(
            m.contains("node 1") && m.contains("sealed table allocation"),
            "{classes:?}: {m}"
        );
    }
    // A partial overlap (the tail of the table allocation and past it).
    let fx = Fixture::new(vec![toucher(
        "w",
        TABLE + 0x800,
        Binding {
            base: TABLE + 0x800,
            bytes: 0x1000,
        },
        &[MemClass::VmemStore],
    )]);
    assert!(closure_reason(fx.bind().unwrap_err()).contains("sealed table allocation"));
}

#[test]
fn a_node_that_writes_a_pointee_allocation_cannot_obtain_a_witness() {
    for classes in [
        &[MemClass::VmemStore][..],
        &[MemClass::Atomic][..],
        &[MemClass::AtomicCommutative][..],
    ] {
        let fx = Fixture::new(vec![toucher("w", W1 + 0x40, w1(), classes)]);
        let m = closure_reason(fx.bind().unwrap_err());
        assert!(
            m.contains("node 1") && m.contains("sealed pointee allocation"),
            "{classes:?}: {m}"
        );
    }
    // A partial overlap: the tail of W0.
    let fx = Fixture::new(vec![toucher(
        "w",
        W0 + 0x8000,
        Binding {
            base: W0 + 0x8000,
            bytes: 0x1_0000,
        },
        &[MemClass::VmemStore],
    )]);
    assert!(closure_reason(fx.bind().unwrap_err()).contains("sealed pointee allocation"));
    // A write wholly inside one pointee.
    let fx = Fixture::new(vec![toucher(
        "w",
        W0 + 0x4000,
        Binding {
            base: W0 + 0x4000,
            bytes: 0x100,
        },
        &[MemClass::VmemStore],
    )]);
    assert!(closure_reason(fx.bind().unwrap_err()).contains("sealed pointee allocation"));
}

#[test]
fn readers_of_the_table_and_pointees_do_not_block_the_witness() {
    let fx = Fixture::new(vec![
        toucher("r0", W0, w0(), &[MemClass::VmemLoad]),
        toucher("r1", TABLE, table_binding(), &[MemClass::SmemLoad]),
        toucher("w", OUT + 8, out_binding(), &[MemClass::VmemStore]),
    ]);
    fx.bind()
        .expect("reads of the protected allocations are fine");
}

#[test]
fn copy_and_fill_destinations_and_unknown_effect_nodes_are_closed_over() {
    // A read-only node makes the pointee a program resource a step can name.
    let mut fx = Fixture::new(vec![toucher("r", W0, w0(), &[MemClass::VmemLoad])]);
    let w0_res = ResourceId(
        fx.program
            .resources
            .iter()
            .position(|b| b.base == W0)
            .unwrap() as u32,
    );
    fx.steps.push(WindowStep::Fill(FillStep {
        dst: Region {
            resource: w0_res,
            byte_offset: Expr::Const(0x40),
        },
        value: 0,
        words: Expr::Const(4),
    }));
    let m = closure_reason(fx.bind().unwrap_err());
    assert!(
        m.contains("step 2") && m.contains("sealed pointee allocation"),
        "{m}"
    );

    fx.steps.pop();
    fx.steps.push(WindowStep::Copy(CopyStep {
        src: Region {
            resource: fx.out_res,
            byte_offset: Expr::Const(0),
        },
        dst: Region {
            resource: fx.table_res,
            byte_offset: Expr::Const(0),
        },
        bytes: Expr::Const(8),
    }));
    let m = closure_reason(fx.bind().unwrap_err());
    assert!(m.contains("sealed table allocation"), "{m}");

    // A copy *from* a protected allocation reads it: fine.
    fx.steps.pop();
    fx.steps.push(WindowStep::Copy(CopyStep {
        src: Region {
            resource: fx.table_res,
            byte_offset: Expr::Const(0),
        },
        dst: Region {
            resource: fx.out_res,
            byte_offset: Expr::Const(0),
        },
        bytes: Expr::Const(8),
    }));
    fx.bind().expect("a copy source is a read");

    // A node with no certified facts has unknown effects.
    let mut missing = toucher("h", W0, w0(), &[MemClass::VmemLoad]);
    missing.facts = Err("no record".into());
    let fx = Fixture::new(vec![missing]);
    let m = closure_reason(fx.bind().unwrap_err());
    assert!(m.contains("node 1") && m.contains("unknown effects"), "{m}");
}

// -------------------------------------------------------------- generation

#[test]
fn a_witness_is_stale_after_retire_reseal_or_a_changed_seal() {
    let fx = Fixture::new(Vec::new());
    let w = fx.bind().unwrap();
    fx.registry.check_witness(&w).unwrap();

    assert_eq!(fx.registry.retire(TABLE), Ok(GEN + 1));
    match fx.registry.check_witness(&w) {
        Err(SealedTableError::Stale(m)) => assert!(m.contains("retired"), "{m}"),
        other => panic!("{other:?}"),
    }
    // Re-sealed (identical bytes) at a higher generation: still another seal.
    fx.registry
        .seal(
            table_binding(),
            ENTRIES,
            &table_bytes(&entry_ptrs()),
            &pointees(),
            GEN + 1,
        )
        .unwrap();
    match fx.registry.check_witness(&w) {
        Err(SealedTableError::Stale(m)) => assert!(m.contains("generation"), "{m}"),
        other => panic!("{other:?}"),
    }
    // And a new bind needs the allocation at the new generation.
    assert!(matches!(fx.bind(), Err(SealedTableError::Stale(_))));
}

#[test]
fn a_leased_table_cannot_be_retired_until_released() {
    let fx = Fixture::new(Vec::new());
    let w = fx.bind().unwrap();
    let lease = fx.registry.lease(&w).unwrap();
    assert!(matches!(
        fx.registry.retire(TABLE),
        Err(SealedTableError::Retire(m)) if m.contains("in flight")
    ));
    lease.release(Quiescence::Proven);
    assert_eq!(fx.registry.retire(TABLE), Ok(GEN + 1));

    let fx = Fixture::new(Vec::new());
    let w = fx.bind().unwrap();
    fx.registry.lease(&w).unwrap().release(Quiescence::Unknown);
    assert!(matches!(
        fx.registry.retire(TABLE),
        Err(SealedTableError::Retire(m)) if m.contains("quarantined")
    ));
}

// ------------------------------------------------------ manifest admission

fn node_record(f: &Fixture) -> WindowNodeRecord {
    WindowNodeRecord {
        index: 0,
        phase: "verify_body".into(),
        class: String::new(),
        symbol: "table_gather".into(),
        artifact_sha256: OBJECT.into(),
        device_elf_sha256: ELF.into(),
        text_sha256: "t".into(),
        kd_sha256: "k".into(),
        metadata_sha256: "m".into(),
        receipt_sha256: "r".into(),
        a1_status: "conditional".into(),
        case_sha256: f.case_digest.clone(),
        required_witnesses: vec!["sealed_table".into()],
        kernarg_size: SEGMENT,
        explicit_end: SEGMENT,
        hidden_loaded: Vec::new(),
    }
}

fn manifest(f: &Fixture) -> WindowManifest {
    let nodes = vec![node_record(f)];
    WindowManifest {
        schema: WINDOW_MANIFEST_SCHEMA.into(),
        version: WINDOW_MANIFEST_VERSION,
        shape: WHOLE16_SHAPE.into(),
        arch: "gfx1201".into(),
        toolchain_pin: "pin".into(),
        sequence_sha256: sequence_digest(&nodes),
        nodes,
        abi: Vec::new(),
    }
}

fn admit(m: &WindowManifest, witnesses: &[ConditionalWitness]) -> Result<(), String> {
    let plan = [RuntimeNode {
        symbol: "table_gather",
        artifact_sha256: OBJECT,
        route: NodeRoute::Railgun,
    }];
    m.admit_with("gfx1201", "pin", &plan, witnesses)
        .map_err(|e| {
            e.iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        })
}

#[test]
fn the_manifest_admits_only_the_exact_node_case_and_kind_the_bind_discharged() {
    let fx = Fixture::new(Vec::new());
    let m = manifest(&fx);
    assert!(m.validate().is_empty(), "sealed_table is a known kind");
    let w = fx.bind().unwrap();

    let held: Vec<ConditionalWitness> = w
        .discharged_cases()
        .into_iter()
        .map(|(index, case_sha256, kind)| ConditionalWitness {
            index: index as u32,
            case_sha256,
            kind,
        })
        .collect();
    admit(&m, &held).expect("the discharged case is admitted");

    // No witness, the indexed kind, another node, another case: all refused.
    let e = admit(&m, &[]).unwrap_err();
    assert!(
        e.contains("conditional, witness required: no sealed_table witness"),
        "{e}"
    );
    for wrong in [
        ConditionalWitness {
            index: 0,
            case_sha256: &fx.case_digest,
            kind: WITNESS_INDEXED_BIND,
        },
        ConditionalWitness {
            index: 1,
            case_sha256: &fx.case_digest,
            kind: WITNESS_SEALED_TABLE,
        },
        ConditionalWitness {
            index: 0,
            case_sha256: "00",
            kind: WITNESS_SEALED_TABLE,
        },
    ] {
        assert!(admit(&m, &[wrong]).is_err(), "{wrong:?}");
    }
    // A manifest naming an unknown kind is invalid on its own.
    let mut bad = m.clone();
    bad.nodes[0].required_witnesses = vec!["sealed_tabel".into()];
    bad.sequence_sha256 = sequence_digest(&bad.nodes);
    assert!(admit(&bad, &held)
        .unwrap_err()
        .contains("unknown witness kind"));
}

// ------------------------------------------------- receipt parse / authoring

#[test]
fn the_fixture_records_satisfy_the_corpus_validate_case() {
    for k in [record(), mixed_record()] {
        k.validate_case(OBJECT, ELF, "gfx1201")
            .expect("a valid case");
    }
}

#[test]
fn a_sealed_only_record_parses_into_separate_table_facts() {
    let facts = KernelFacts::from_record(&record()).unwrap();
    let FactStatus::Conditional(c) = facts.status else {
        panic!("conditional")
    };
    assert!(c.guards.is_empty() && c.indexed.is_empty());
    assert_eq!(c.witnesses, vec!["sealed_table".to_owned()]);
    let [t] = c.tables.as_slice() else {
        panic!("one table")
    };
    assert_eq!(
        (t.table_arg_offset, t.bound, t.entry_bytes),
        (0, TableBound::Kernarg(16), 8)
    );
    assert_eq!(t.entries[0].class, MemClass::SmemLoad);
    assert_eq!(t.accesses[0].class, MemClass::VmemLoad);
    assert_eq!(t.accesses[0].entry_pcs, vec![0x10]);

    let k = record_with(vec![sealed_record(receipt::SEALED_BOUND_CONST, 7)]);
    let FactStatus::Conditional(c) = KernelFacts::from_record(&k).unwrap().status else {
        panic!()
    };
    assert_eq!(c.tables[0].bound, TableBound::Const(7));
}

fn parse_err(k: &KernelRecord) -> String {
    KernelFacts::from_record(k).expect_err("rejected")
}

#[test]
fn a_sealed_only_record_with_the_indexed_bind_kind_is_rejected() {
    let mut k = record();
    k.required_witnesses = vec!["indexed_bind".into()];
    let k = digest_of(k);
    assert!(
        parse_err(&k).contains("not the kinds the case implies"),
        "{}",
        parse_err(&k)
    );
    // ... and so is the converse (indexed case claiming only sealed_table).
    let mut k = mixed_record();
    k.required_witnesses = vec!["sealed_table".into()];
    assert!(parse_err(&digest_of(k)).contains("not the kinds the case implies"));
    // A mixed record must name both.
    let mut k = mixed_record();
    k.required_witnesses = vec!["indexed_bind".into()];
    assert!(parse_err(&digest_of(k)).contains("not the kinds the case implies"));
}

#[test]
fn an_unknown_witness_kind_is_rejected() {
    for kinds in [
        vec!["bogus".to_owned()],
        vec!["bogus".to_owned(), "sealed_table".to_owned()],
        vec!["sealed_table".to_owned(), "bogus".to_owned()],
    ] {
        let mut k = record();
        k.required_witnesses = kinds;
        let m = parse_err(&digest_of(k));
        assert!(m.contains("unknown required witness kind"), "{m}");
    }
}

#[test]
fn a_conditional_record_must_follow_the_validate_case_shape() {
    // Sealed-only records carry no guards.
    let mut k = record();
    k.guards = mixed_record().guards;
    assert!(parse_err(&digest_of(k)).contains("carries guards"));
    // An indexed case needs its guards.
    let mut k = mixed_record();
    k.guards.clear();
    assert!(parse_err(&digest_of(k)).contains("lacks guards"));
    // No effects, no witnesses, no digest.
    let k = record_with(Vec::new());
    assert!(parse_err(&k).contains("lacks effects"));
    let mut k = record();
    k.required_witnesses.clear();
    assert!(parse_err(&digest_of(k)).contains("lacks effects"));
    let mut k = record();
    k.case_digest.clear();
    assert!(parse_err(&k).contains("lacks effects"));
    // The table argument is a pointer of 8 bytes; the bound lies in the kernarg.
    let mut k = record();
    k.sealed_tables[0].table_arg_offset = 16;
    assert!(parse_err(&digest_of(k)).contains("not an 8-byte pointer"));
    let mut k = record();
    k.sealed_tables[0].bound_value = 24;
    assert!(parse_err(&digest_of(k)).contains("lies outside"));
    // Duplicates and structurally malformed tables.
    let mut k = record();
    let twin = k.sealed_tables[0].clone();
    k.sealed_tables.push(twin);
    assert!(parse_err(&digest_of(k)).contains("duplicate sealed table"));
    let mut k = record();
    k.sealed_tables[0].entry_bytes = 4;
    assert!(parse_err(&digest_of(k)).contains("entry width"));
    let mut k = record();
    k.sealed_tables[0].accesses[0].class = "vmem_store".into();
    assert!(parse_err(&digest_of(k)).contains("not a read"));
}

#[test]
fn the_table_argument_gets_a_read_effect_and_never_a_write() {
    // The receipt's own arg classes omit the table read: the entry loads
    // still make it an ordinary read of the table's resource.
    let mut k = record();
    k.args[0].classes.clear();
    k.args[0].mode = None;
    let fx = Fixture::with(&digest_of(k), 4, Vec::new(), true);
    let node = &fx.authored.nodes[0];
    let e = node
        .effects
        .iter()
        .find(|e| e.arg_offset == 0)
        .expect("the table argument has an effect");
    assert_eq!(e.resource, fx.table_res);
    assert_eq!(e.classes, vec![MemClass::SmemLoad]);
    assert!(e.reads() && !e.writes());
    // The output argument keeps its store and nothing else gains one.
    assert!(node
        .effects
        .iter()
        .filter(|e| e.writes())
        .all(|e| e.arg_offset == 8));
    assert_eq!(node.table.as_ref().unwrap().case_digest, fx.case_digest);
}

fn authored_reason(l: LaunchRecord) -> Lowering {
    author("gfx1201", &[l]).nodes[0].lowering.clone()
}

#[test]
fn a_table_with_write_effects_a_null_pointer_or_a_missing_allocation_is_hip_direct() {
    // A write class on the table argument (here the receipt reports one).
    let mut k = record();
    k.args[0].classes.push("vmem_store".into());
    let l = gather_launch(node_facts(&digest_of(k)).unwrap(), TABLE, 4);
    assert!(
        matches!(authored_reason(l), Lowering::HipDirect { reason } if reason.contains("has write effects"))
    );
    // A null table pointer is a constant, not a bound allocation.
    let l = gather_launch(node_facts(&record()).unwrap(), 0, 4);
    assert!(
        matches!(authored_reason(l), Lowering::HipDirect { reason } if reason.contains("not a bound allocation"))
    );
    // A table pointer with no probed allocation.
    let mut l = gather_launch(node_facts(&record()).unwrap(), TABLE, 4);
    let _probe = l.allocations.remove(0);
    assert!(
        matches!(authored_reason(l), Lowering::HipDirect { reason } if reason.contains("not probed"))
    );
}

#[test]
fn assume_witnesses_lowers_only_nodes_whose_every_kind_is_assumed() {
    let sealed = Fixture::new(Vec::new());
    let a = &sealed.authored;
    assert!(matches!(a.nodes[0].lowering, Lowering::Conditional { .. }));
    // The indexed assumption does not cover a sealed-only node.
    assert!(matches!(
        a.assume_indexed_witness().nodes[0].lowering,
        Lowering::Conditional { .. }
    ));
    assert!(a.assume_sealed_table_witness().nodes[0].lowering.is_pm4());
    assert!(a
        .assume_witnesses(&[WITNESS_INDEXED_BIND, WITNESS_SEALED_TABLE])
        .nodes[0]
        .lowering
        .is_pm4());
    // The cached program is never promoted.
    assert!(matches!(a.nodes[0].lowering, Lowering::Conditional { .. }));

    // A node needing both kinds is PM4 only when both are assumed.
    let mixed = Fixture::with(&mixed_record(), 4, Vec::new(), true);
    let m = &mixed.authored;
    assert!(
        matches!(&m.nodes[0].lowering, Lowering::Conditional { reason } if reason.contains("indexed_bind and sealed_table"))
    );
    assert!(m.nodes[0].indexed.is_some() && m.nodes[0].table.is_some());
    assert_eq!(m.conditional_nodes(), vec![0]);
    assert_eq!(m.sealed_table_nodes(), vec![0]);
    for one in [m.assume_indexed_witness(), m.assume_sealed_table_witness()] {
        assert!(matches!(
            one.nodes[0].lowering,
            Lowering::Conditional { .. }
        ));
    }
    assert!(m
        .assume_witnesses(&[WITNESS_INDEXED_BIND, WITNESS_SEALED_TABLE])
        .nodes[0]
        .lowering
        .is_pm4());
    // The mixed record's facts keep the two families apart.
    let FactStatus::Conditional(ConditionalFacts {
        indexed, tables, ..
    }) = KernelFacts::from_record(&mixed_record()).unwrap().status
    else {
        panic!()
    };
    assert_eq!((indexed.len(), tables.len()), (1, 1));
}

#[test]
fn a_failed_conditional_guard_still_refuses_a_mixed_node_and_sealed_only_has_none() {
    // The mixed record's guard `kernarg[20] == 0`; a launch with another value
    // is HipDirect, as before.
    let mut l = gather_launch(node_facts(&mixed_record()).unwrap(), TABLE, 4);
    l.kernarg[20] = 9;
    assert!(
        matches!(authored_reason(l), Lowering::HipDirect { reason } if reason.contains("conditional guard"))
    );
}

// ---------------------------------------------------------- null entries

/// Null entries are the owner's decision per table, taken from the receipt's
/// `null_guarded`: only a certificate that proves every read through an entry
/// happens for a nonzero entry may bind a table that holds one.
#[test]
fn null_entries_need_the_allow_policy_and_a_null_guarded_certificate() {
    // Strict seal refuses a null entry (an unguarded certificate's table).
    let registry = SealedTableRegistry::new();
    let nullable = table_bytes(&nullable_ptrs());
    let m = seal_err(registry.seal(table_binding(), ENTRIES, &nullable, &pointees(), GEN));
    assert!(m.contains("entry 1 is null"), "{m}");
    // `Allow` seals it, still checking every other entry against the pointees,
    // and records that the table holds nulls.
    let sealed = registry
        .seal_with(
            table_binding(),
            ENTRIES,
            &nullable,
            &pointees(),
            GEN,
            NullEntries::Allow,
        )
        .expect("null entries allowed");
    assert!(sealed.has_null_entries);
    let mut stray = table_bytes(&nullable_ptrs());
    stray[16..24].copy_from_slice(&0x9000_0000u64.to_le_bytes());
    let m = seal_err(SealedTableRegistry::new().seal_with(
        table_binding(),
        ENTRIES,
        &stray,
        &pointees(),
        GEN,
        NullEntries::Allow,
    ));
    assert!(m.contains("entry 2") && m.contains("outside"), "{m}");

    // The bind: a null-holding table with a certificate that records no null
    // guard gets no witness; with one it does.
    let fx = Fixture::nullable(&record());
    assert_eq!(
        fx.bind().unwrap_err(),
        SealedTableError::NullEntryUnguarded {
            node: 0,
            table_arg: 0
        }
    );
    let fx = Fixture::nullable(&guarded_record());
    let w = fx.bind().expect("null-guarded certificate");
    assert_eq!(
        w.discharged_cases(),
        vec![(0, fx.case_digest.as_str(), "sealed_table")]
    );
    // A table sealed with `Allow` that happens to hold no null binds with an
    // unguarded certificate: the policy is about content, not intent.
    let fx = Fixture::with(&record(), 4, Vec::new(), false);
    let t = fx
        .registry
        .seal_with(
            table_binding(),
            ENTRIES,
            &table_bytes(&entry_ptrs()),
            &pointees(),
            GEN,
            NullEntries::Allow,
        )
        .unwrap();
    assert!(!t.has_null_entries);
    fx.bind().expect("no null entry");
    // A null-guarded certificate is no reason to relax the other checks: a
    // table modified after sealing is still refused.
    let fx = Fixture::nullable(&guarded_record());
    fx.content.bytes.lock().unwrap()[8] = 1;
    assert!(matches!(fx.bind(), Err(SealedTableError::Modified { .. })));
}

#[test]
fn a_null_guarded_receipt_record_parses_into_the_table_fact() {
    for (k, want) in [(guarded_record(), true), (record(), false)] {
        let FactStatus::Conditional(c) = KernelFacts::from_record(&k).unwrap().status else {
            panic!("conditional")
        };
        assert_eq!(c.tables[0].null_guarded, want);
    }
}

// ------------------------------------------------------------ vacuous cases

const VEC: u64 = 0x7800_0000;
const VEC_BYTES: u64 = 0x20_0000;
const VEC_AT: u64 = 0x18_8000;

/// The same kernel in its "plain vector" mode: the table argument points
/// `VEC_AT` bytes into a larger allocation and `n = 0`, so the certified guard
/// can never pass.
pub(crate) fn vector_mode_launch(k: &KernelRecord) -> LaunchRecord {
    let mut l = gather_launch(node_facts(k).unwrap(), VEC + VEC_AT, 0);
    l.allocations = vec![
        (
            0,
            Ok(Binding {
                base: VEC,
                bytes: VEC_BYTES,
            }),
        ),
        (8, Ok(out_binding())),
    ];
    l
}

fn discharged(w: &SealedTableWitness) -> Vec<usize> {
    w.discharged_cases().iter().map(|c| c.0).collect()
}

/// The kernel in a mode that takes no table at all: the table argument is a null optional pointer and
/// `n = 0`.
pub(crate) fn null_table_launch(k: &KernelRecord) -> LaunchRecord {
    let mut l = gather_launch(node_facts(k).unwrap(), 0, 0);
    l.allocations = vec![(8, Ok(out_binding()))];
    l
}

#[test]
fn a_null_table_pointer_with_a_zero_bound_is_authored_and_bound_as_vacuous() {
    let k = record();
    // Authored: not HipDirect (the table is never read), the argument stays a constant.
    let l = null_table_launch(&k);
    let authored = author("gfx1201", &[l]);
    assert!(
        matches!(&authored.nodes[0].lowering, Lowering::Conditional { reason } if reason.contains("sealed_table")),
        "{:?}",
        authored.nodes[0].lowering
    );
    assert!(authored.nodes[0].kref.is_some());
    // Bound beside a real table node: discharged vacuous, nothing sealed for it.
    let fx = Fixture::with(&k, 4, vec![null_table_launch(&k)], true);
    let w = fx.bind().expect("both nodes are discharged");
    assert_eq!(discharged(&w), vec![0, 1]);
    assert!(w.nodes()[1].tables.is_empty());
    assert_eq!(w.nodes()[1].vacuous, vec![0]);
    // A null pointer with a nonzero bound is still refused.
    let l = gather_launch(node_facts(&k).unwrap(), 0, 4);
    assert!(
        matches!(authored_reason(l), Lowering::HipDirect { reason } if reason.contains("not a bound allocation"))
    );
}

#[test]
fn a_zero_bound_node_is_vacuous_beside_a_real_table_node() {
    // Node 0: a real 4-entry table (mode 2). Node 1: the same kernel with the
    // table argument an interior pointer into a plain vector and n = 0.
    let k = record();
    let fx = Fixture::with(&k, 4, vec![vector_mode_launch(&k)], true);
    let w = fx.bind().expect("both nodes are discharged");
    assert_eq!(discharged(&w), vec![0, 1]);
    assert_eq!(w.nodes()[0].tables.len(), 1);
    assert!(w.nodes()[0].vacuous.is_empty());
    assert!(w.nodes()[1].tables.is_empty());
    assert_eq!(w.nodes()[1].vacuous, vec![0]);
    // Both are still bound under the receipt's case digest.
    assert!(w.discharged_cases().iter().all(|c| c.1 == fx.case_digest));
    // The real node's content is still verified once.
    assert_eq!(fx.content.reads.load(Ordering::Relaxed), 1);
    // A vacuous node alone needs no seal and no content read.
    let fx = Fixture::with(&k, 0, Vec::new(), false);
    let w = fx.bind().expect("vacuous: nothing to seal");
    assert_eq!(discharged(&w), vec![0]);
    assert_eq!(fx.content.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn a_nonzero_bound_on_an_interior_pointer_is_still_refused() {
    let k = record();
    let mut fx = Fixture::with(&k, 4, vec![vector_mode_launch(&k)], true);
    // Node 1's bound becomes 1: the interior vector is no sealed table.
    let at = 16;
    fx.images[1][at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        fx.bind(),
        Err(SealedTableError::NotSealed(_) | SealedTableError::WrongAllocation(_))
    ));
}

#[test]
fn a_constant_zero_bound_is_vacuous_whatever_the_launch_scalar() {
    let k = record_with(vec![sealed_record(receipt::SEALED_BOUND_CONST, 0)]);
    let fx = Fixture::with(&k, 999, vec![vector_mode_launch(&k)], false);
    let w = fx.bind().expect("const 0 bound");
    assert_eq!(discharged(&w), vec![0, 1]);
    assert!(w.nodes().iter().all(|n| n.tables.is_empty()));
    // A nonzero constant still needs the seal.
    let k = record_with(vec![sealed_record(receipt::SEALED_BOUND_CONST, 1)]);
    let fx = Fixture::with(&k, 0, Vec::new(), false);
    assert!(matches!(fx.bind(), Err(SealedTableError::NotSealed(_))));
}

#[test]
fn a_vacuous_witness_is_void_once_its_bound_image_changes() {
    let k = record();
    let fx = Fixture::with(&k, 4, vec![vector_mode_launch(&k)], true);
    let w = fx.bind().unwrap();
    let mut images = fx.images.clone();
    assert!(w.matches_window(1, 7, &images, &fx.grids).is_ok());
    images[1][16..20].copy_from_slice(&1u32.to_le_bytes());
    assert!(matches!(
        w.matches_window(1, 7, &images, &fx.grids),
        Err(SealedTableError::Node { node: 1, .. })
    ));
}
