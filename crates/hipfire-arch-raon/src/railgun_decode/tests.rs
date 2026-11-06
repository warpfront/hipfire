// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! GPU-free tests of the Raon railgun decode owner: admission, key, record
//! construction, authoring + sealed-table bind, and the execution/quarantine
//! state machine over a mock queue. A GPU oracle is a separate slice.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
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
        null_guarded: false,
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
        descriptor_arenas: Vec::new(),
        required_witnesses: vec![receipt::WITNESS_SEALED_TABLE.into()],
        case_digest: String::new(),
        guard_flag: None,
        task_protocol: None,
        wg_footprints: Vec::new(),
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

const GFX1201: &str = "gfx1201";
const GFX1100: &str = "gfx1100";

fn table_for(arch: &str) -> CacheTable {
    CacheTable::for_arch(arch).unwrap_or_else(|| panic!("{arch} cache table"))
}

fn table() -> CacheTable {
    table_for(GFX1201)
}

// ----------------------------------------------------------------- admission

type Facts = (Structural<'static>, StepFacts<'static>);

fn ok_facts() -> Facts {
    (
        Structural {
            arch: GFX1201,
            device_identity: true,
            whole_band: true,
            band: "whole",
            graph_forward_disabled: false,
            graph_mode: true,
            retained_replay: false,
            host_weights: false,
        },
        StepFacts {
            tap_armed: false,
            dump_armed: false,
            absorbed_route: false,
            warmed: true,
            corpus: CorpusState::Ready("corpus"),
        },
    )
}

fn admit(f: &Facts) -> Result<(), RaonRailgunRefusal> {
    check_admission(&f.0, &f.1)
}

#[test]
fn a_clean_context_is_admitted() {
    assert_eq!(admit(&ok_facts()), Ok(()));
}

#[test]
fn every_cheap_admission_check_declines_with_its_exact_reason() {
    use RaonRailgunRefusal as R;
    type Mutate = fn(&mut Facts);
    let cases: Vec<(&str, Mutate, R)> = vec![
        (
            "arch",
            |a| a.0.arch = "gfx1151",
            R::WrongArch {
                arch: "gfx1151".into(),
            },
        ),
        (
            "identity",
            |a| {
                a.0.arch = GFX1100;
                a.0.device_identity = false;
            },
            R::DeviceIdentity {
                arch: GFX1100.into(),
            },
        ),
        (
            "band",
            |a| {
                a.0.whole_band = false;
                a.0.band = "pp 0/2";
            },
            R::PartitionedBand {
                band: "pp 0/2".into(),
            },
        ),
        (
            "graph=0",
            |a| a.0.graph_forward_disabled = true,
            R::GraphDisabled,
        ),
        ("replay", |a| a.0.retained_replay = true, R::RetainedReplay),
        ("host weights", |a| a.0.host_weights = true, R::HostWeights),
        ("eager route", |a| a.0.graph_mode = false, R::NotGraphMode),
        ("census tap", |a| a.1.tap_armed = true, R::TapArmed),
        ("dumps", |a| a.1.dump_armed = true, R::DumpArmed),
        ("absorbed", |a| a.1.absorbed_route = true, R::AbsorbedRoute),
        ("cold graph", |a| a.1.warmed = false, R::NotWarm),
        (
            "no corpus",
            |a| a.1.corpus = CorpusState::Unset,
            R::CorpusUnset,
        ),
        (
            "empty corpus",
            |a| a.1.corpus = CorpusState::Ready(""),
            R::CorpusUnset,
        ),
    ];
    for (name, mutate, want) in cases {
        let mut a = ok_facts();
        mutate(&mut a);
        assert_eq!(admit(&a), Err(want), "{name}");
    }
}

#[test]
fn the_first_failing_admission_check_wins() {
    let mut a = ok_facts();
    a.0.arch = "gfx1101";
    a.0.device_identity = false;
    a.1.tap_armed = true;
    a.1.warmed = false;
    assert_eq!(
        admit(&a),
        Err(RaonRailgunRefusal::WrongArch {
            arch: "gfx1101".into()
        })
    );
    // The identity check precedes every other structural check.
    a.0.arch = GFX1100;
    a.0.whole_band = false;
    assert_eq!(
        admit(&a),
        Err(RaonRailgunRefusal::DeviceIdentity {
            arch: GFX1100.into()
        })
    );
}

#[test]
fn both_admitted_arches_pass_the_same_checks_in_the_same_order() {
    for arch in [GFX1201, GFX1100] {
        let mut a = ok_facts();
        a.0.arch = arch;
        assert_eq!(admit(&a), Ok(()), "{arch}");
        a.1.tap_armed = true;
        a.1.warmed = false;
        assert_eq!(admit(&a), Err(RaonRailgunRefusal::TapArmed), "{arch}");
    }
}

#[test]
fn only_the_exact_arches_are_admitted() {
    for arch in [
        "gfx1200", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1030",
        "gfx942", "GFX1100", "gfx1100 ", "gfx11", "",
    ] {
        assert!(arch_policy(arch).is_none(), "{arch:?}");
        let mut a = ok_facts();
        a.0.arch = arch;
        assert_eq!(
            admit(&a),
            Err(RaonRailgunRefusal::WrongArch {
                arch: arch.to_owned()
            }),
            "{arch:?}"
        );
    }
    let arches: Vec<_> = ARCH_POLICIES.iter().map(|p| p.arch).collect();
    assert_eq!(arches, [GFX1201, GFX1100]);
    let text = RaonRailgunRefusal::WrongArch {
        arch: "gfx1151".into(),
    }
    .to_string();
    assert_eq!(text, "arch gfx1151 is not one of gfx1201, gfx1100");
}

#[test]
fn only_gfx1100_needs_a_pci_identity() {
    assert!(!arch_policy(GFX1201).expect("gfx1201 policy").pci_identity);
    assert!(arch_policy(GFX1100).expect("gfx1100 policy").pci_identity);
}

#[test]
fn every_policy_variable_is_arch_qualified_before_the_generic_one() {
    for p in ARCH_POLICIES {
        let suffix = p.arch.to_ascii_uppercase();
        assert_eq!(
            p.corpus_vars,
            [
                format!("HIPFIRE_RAON_RAILGUN_CORPUS_{suffix}").as_str(),
                ENV_CORPUS
            ]
        );
        // The generic table variable is the one the DFlash window and the M2
        // shadow read: unchanged for gfx1201.
        assert_eq!(
            p.cache_table_vars,
            [
                format!("HIPFIRE_RAILGUN_CACHE_TABLE_{suffix}").as_str(),
                "HIPFIRE_RAILGUN_CACHE_TABLE"
            ]
        );
    }
}

#[test]
fn the_lowering_parameters_are_per_arch() {
    assert_eq!(
        arch_policy(GFX1201).unwrap().lowering().post_dispatch_nop,
        rp::GFX1201_POST_DISPATCH_NOP
    );
    assert_eq!(
        arch_policy(GFX1100).unwrap().lowering().post_dispatch_nop,
        0
    );
}

#[test]
fn the_cache_table_is_the_arch_table() {
    for p in ARCH_POLICIES {
        let t = cache_table(p).expect("table");
        assert_eq!(t.arch, p.arch);
        assert!(t.rows.iter().all(|r| r.g4_receipt.is_none()));
    }
}

fn g4_json(arch: &str, stale: u64, trials: u64) -> Vec<u8> {
    let row = |name: &str, rung: &str| {
        format!(
            r#"{{"row":"{name}","weakest_safe_rung":"{rung}","trials":[{{"rung":"{rung}","trials":{trials},"stale_trials":{stale}}}]}}"#
        )
    };
    format!(
        r#"{{"identity":{{"arch":"{arch}"}},"rows":[{},{},{},{},{}]}}"#,
        row("raw_vmem", "wait_inter_node"),
        row("raw_smem", "wait_inter_node"),
        row("war", "wait_inter_node"),
        row("waw", "wait_inter_node"),
        row("war_pre_writer", "wait_system"),
    )
    .into_bytes()
}

#[test]
fn a_g4_table_attaches_only_to_its_own_arch() {
    for p in ARCH_POLICIES {
        let mut t = CacheTable::for_arch(p.arch).unwrap();
        let other = ARCH_POLICIES.iter().find(|q| q.arch != p.arch).unwrap();
        let err = attach_g4(&mut t, "other.json", &g4_json(other.arch, 0, 1 << 20)).unwrap_err();
        assert!(
            err.contains(other.arch) && err.contains(p.arch),
            "{}: {err}",
            p.arch
        );
        assert!(t.rows.iter().all(|r| r.g4_receipt.is_none()));
    }
}

#[test]
fn a_stale_g4_row_fails_and_too_few_trials_back_nothing() {
    for p in ARCH_POLICIES {
        let mut t = CacheTable::for_arch(p.arch).unwrap();
        let stale = attach_g4(&mut t, "g4.json", &g4_json(p.arch, 3, 1 << 20)).unwrap_err();
        assert!(stale.contains("stale"), "{}: {stale}", p.arch);
        let mut t = CacheTable::for_arch(p.arch).unwrap();
        let few = attach_g4(&mut t, "g4.json", &g4_json(p.arch, 0, 1000)).unwrap();
        assert_eq!(few, 0, "{}", p.arch);
        assert!(t.rows.iter().all(|r| r.g4_receipt.is_none()));
        let err = attach_g4(&mut t, "g4.json", b"{not json").unwrap_err();
        assert!(err.starts_with("g4.json:"), "{err}");
    }
}

// ----------------------------------------------------------------- tri-state

#[test]
fn the_variable_parses_as_auto_1_or_0_and_defaults_to_auto() {
    use RailgunMode::{Auto, Force, Off};
    for (raw, want) in [
        (None, Auto),
        (Some(""), Auto),
        (Some("auto"), Auto),
        (Some(" AUTO "), Auto),
        (Some("1"), Force),
        (Some("true"), Force),
        (Some("On"), Force),
        (Some("0"), Off),
        (Some("false"), Off),
        (Some("OFF"), Off),
    ] {
        assert_eq!(parse_mode(raw), Ok(want), "{raw:?}");
    }
    for bad in ["2", "yes", "enable", "autoo", "-1"] {
        let e = parse_mode(Some(bad)).expect_err(bad);
        assert!(e.contains("auto, 1 or 0") && e.contains(bad), "{e}");
    }
}

/// The explicit and pack candidates only: no per-user directory, no build.
fn c<'a>(env: Option<&'a str>, pack: Option<&'a str>) -> Candidates<'a> {
    Candidates {
        env,
        pack,
        local: None,
        may_provision: false,
    }
}

fn has_index(dirs: &'static [&'static str]) -> impl Fn(&str) -> bool {
    move |d| dirs.contains(&d)
}

fn structural() -> Structural<'static> {
    ok_facts().0
}

#[test]
fn zero_is_a_hard_kill_that_looks_at_nothing() {
    let looked = Cell::new(0);
    let probe = |_: &str| {
        looked.set(looked.get() + 1);
        true
    };
    let mut s = structural();
    for arch in [GFX1201, GFX1100] {
        s.arch = arch;
        assert_eq!(
            decide_owner(RailgunMode::Off, &s, &c(Some("env"), Some("pack")), &probe),
            Err(NoOwner::Killed)
        );
    }
    assert_eq!(
        looked.get(),
        0,
        "the kill switch must not touch the file system"
    );
}

#[test]
fn auto_without_a_corpus_creates_no_owner() {
    let s = structural();
    let none = has_index(&[]);
    // No pack at all (objects are JIT-compiled), a pack without a corpus, and
    // an explicit directory without one: no owner in each case.
    for (env, pack) in [(None, None), (None, Some("pack")), (Some("env"), None)] {
        match decide_owner(RailgunMode::Auto, &s, &c(env, pack), &none) {
            Err(NoOwner::NoCorpus(why)) => assert!(why.contains(ENV_CORPUS), "{why}"),
            other => panic!("expected NoCorpus, got {other:?}"),
        }
    }
    // An empty explicit directory is the same as unset.
    assert!(matches!(
        decide_owner(RailgunMode::Auto, &s, &c(Some(""), None), &none),
        Err(NoOwner::NoCorpus(_))
    ));
}

#[test]
fn auto_creates_the_owner_for_an_explicit_or_pack_corpus_quietly() {
    let s = structural();
    let both = has_index(&["env", "pack"]);
    let plan = |env, pack| decide_owner(RailgunMode::Auto, &s, &c(env, pack), &both);
    let want = |dir: &str| {
        Ok(OwnerPlan {
            corpus_dir: Some(dir.into()),
            loud: false,
            provision: false,
        })
    };
    assert_eq!(plan(Some("env"), Some("pack")), want("env"));
    assert_eq!(plan(None, Some("pack")), want("pack"));
    assert_eq!(plan(Some("env"), None), want("env"));
    // An explicit directory wins even over a good pack corpus, and is not
    // silently replaced by it when it holds nothing.
    let only_pack = has_index(&["pack"]);
    assert!(matches!(
        decide_owner(RailgunMode::Auto, &s, &c(Some("env"), Some("pack")), &only_pack),
        Err(NoOwner::NoCorpus(w)) if w.contains("env")
    ));
}

#[test]
fn auto_refuses_a_structurally_unsuitable_context_before_looking_for_a_corpus() {
    let looked = Cell::new(0);
    let probe = |_: &str| {
        looked.set(looked.get() + 1);
        true
    };
    type Mutate = fn(&mut Structural<'static>);
    let cases: Vec<(Mutate, RaonRailgunRefusal)> = vec![
        (
            |s| s.arch = "gfx1151",
            RaonRailgunRefusal::WrongArch {
                arch: "gfx1151".into(),
            },
        ),
        (
            |s| s.whole_band = false,
            RaonRailgunRefusal::PartitionedBand {
                band: "whole".into(),
            },
        ),
        (
            |s| s.graph_forward_disabled = true,
            RaonRailgunRefusal::GraphDisabled,
        ),
        (
            |s| s.retained_replay = true,
            RaonRailgunRefusal::RetainedReplay,
        ),
        (|s| s.host_weights = true, RaonRailgunRefusal::HostWeights),
        (|s| s.graph_mode = false, RaonRailgunRefusal::NotGraphMode),
    ];
    for (mutate, want) in cases {
        let mut s = structural();
        mutate(&mut s);
        assert_eq!(
            decide_owner(RailgunMode::Auto, &s, &c(Some("env"), Some("pack")), &probe),
            Err(NoOwner::Refused(want))
        );
    }
    assert_eq!(looked.get(), 0);
}

#[test]
fn one_forces_the_owner_loudly_whatever_the_context() {
    let mut s = structural();
    s.arch = "gfx1100";
    s.graph_forward_disabled = true;
    let pack_only = has_index(&["pack"]);
    let plan = |env, pack| decide_owner(RailgunMode::Force, &s, &c(env, pack), &pack_only);
    let loud = |dir: Option<&str>| {
        Ok(OwnerPlan {
            corpus_dir: dir.map(str::to_owned),
            loud: true,
            provision: false,
        })
    };
    assert_eq!(plan(Some("env"), Some("pack")), loud(Some("env")));
    assert_eq!(plan(None, Some("pack")), loud(Some("pack")));
    // No corpus anywhere: the owner still exists and declines CorpusUnset on
    // every step (loudly), instead of the route silently not existing.
    assert_eq!(plan(None, Some("elsewhere")), loud(None));
    assert_eq!(plan(None, None), loud(None));
}

fn owner(plan: OwnerPlan) -> RaonRailgunOwner {
    RaonRailgunOwner::new(plan, GFX1201, None, "model".into())
}

#[test]
fn an_owner_of_an_unadmitted_arch_has_no_policy_and_no_cache_table() {
    let plan = |loud| OwnerPlan {
        corpus_dir: Some("c".into()),
        loud,
        provision: false,
    };
    let mut o = RaonRailgunOwner::new(plan(false), "gfx1151", None, "m".into());
    assert!(o.policy.is_none());
    assert_eq!(
        o.cache_table().err(),
        Some(RaonRailgunRefusal::WrongArch {
            arch: "gfx1151".into()
        })
    );
    let o = RaonRailgunOwner::new(plan(false), GFX1100, None, "m".into());
    assert_eq!(o.policy, arch_policy(GFX1100));
}

#[test]
fn the_forced_owner_logs_each_kind_once_and_counts_every_decline() {
    let mut o = owner(OwnerPlan {
        corpus_dir: None,
        loud: true,
        provision: false,
    });
    assert!(o.is_forced());
    for r in [
        RaonRailgunRefusal::NotWarm,
        RaonRailgunRefusal::NotWarm,
        RaonRailgunRefusal::TapArmed,
        RaonRailgunRefusal::NotWarm,
    ] {
        o.decline(r);
    }
    assert_eq!(
        o.logged_kinds.iter().copied().collect::<Vec<_>>(),
        vec!["not_warm", "tap_armed"]
    );
    let s = o.stats();
    assert_eq!(s.declined, 4);
    assert_eq!(s.declined_by_kind["not_warm"], 3);
    assert_eq!(
        s.last_refusal.as_deref(),
        Some("the decode graph warm-up step has not run")
    );
}

#[test]
fn the_auto_owner_stays_quiet_for_the_routine_warm_up_and_logs_one_line() {
    let mut o = owner(OwnerPlan {
        corpus_dir: Some("corpus".into()),
        loud: false,
        provision: false,
    });
    assert!(!o.is_forced());
    o.decline(RaonRailgunRefusal::NotWarm);
    assert!(!o.auto_logged, "the warm-up step is routine");
    o.decline(RaonRailgunRefusal::AbsorbedRoute);
    assert!(o.auto_logged);
    o.decline(RaonRailgunRefusal::TapArmed);
    assert!(o.logged_kinds.is_empty(), "auto keeps no per-kind log");
    assert_eq!(o.stats().declined, 3);
}

#[test]
fn every_refusal_has_a_distinct_kind_and_a_message() {
    use RaonRailgunRefusal as R;
    let all = vec![
        R::Disabled,
        R::WrongArch { arch: "x".into() },
        R::DeviceIdentity { arch: "x".into() },
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
        R::CorpusProvision("x".into()),
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
        ("model_digest", |k| k.base.model_digest = "modelx".into()),
        ("flags_digest", |k| k.base.flags_digest = "flagsx".into()),
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

/// Two gathers writing the same output: a WAW boundary before node 1.
fn authored_pair_for(table: &CacheTable) -> Authored {
    let dev = FakeDevice::new(&entry_ptrs());
    let l = || gather_launch(Ok(node_facts(&gather_record(false), "D")), TABLE, 4);
    author_and_plan(&[l(), l()], table, &dev).expect("authored")
}

#[test]
fn the_program_is_authored_for_the_cache_table_arch() {
    for arch in [GFX1201, GFX1100] {
        let a = authored_pair_for(&table_for(arch));
        assert_eq!(a.program.arch, arch);
        assert!(a.plan.is_single_segment(a.program.nodes.len()), "{arch}");
        assert_eq!(a.plan.policy, RowPolicy::UnreceiptedSystem);
        assert!(
            a.program.nodes.iter().all(|n| n.lowering.is_pm4()),
            "{arch}"
        );
    }
}

#[test]
fn a_dependent_boundary_is_system_scope_until_a_g4_receipt_backs_the_arch_row() {
    use railgun::plan::Rung;
    for arch in [GFX1201, GFX1100] {
        let unreceipted = authored_pair_for(&table_for(arch));
        let v = unreceipted.plan.boundaries[1].visibility;
        assert!(v.wait_idle, "{arch}");
        assert_eq!(
            v.acquire,
            Rung::System,
            "{arch}: strict row 3 without a receipt"
        );

        let mut t = table_for(arch);
        attach_g4(&mut t, "g4.json", &g4_json(arch, 0, 1 << 20)).expect("attach");
        let receipted = authored_pair_for(&t);
        let v = receipted.plan.boundaries[1].visibility;
        assert!(v.wait_idle, "{arch}");
        assert_eq!(v.acquire, Rung::InterNode, "{arch}: receipted tier-D rung");
    }
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
            return Err(QueueFailure::new(m, q));
        }
        log.submits += 1;
        Ok(())
    }

    fn wait(&mut self) -> Result<(), QueueFailure> {
        let mut log = self.0.lock().unwrap();
        log.waits += 1;
        match log.wait_failure {
            Some((m, q)) => Err(QueueFailure::new(m, q)),
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
        Lookup::Declined(Arc::new(RaonRailgunRefusal::NotWarm))
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
        Lookup::Declined(Arc::new(want))
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

// ----------------------------------------------------- first-load provisioning

fn cands<'a>(local: Option<&'a str>, may_provision: bool) -> Candidates<'a> {
    Candidates {
        env: None,
        pack: Some("pack"),
        local,
        may_provision,
    }
}

#[test]
fn auto_without_any_corpus_waits_to_build_the_per_user_one() {
    let s = structural();
    let none = has_index(&[]);
    assert_eq!(
        decide_owner(RailgunMode::Auto, &s, &cands(Some("local"), true), &none),
        Ok(OwnerPlan {
            corpus_dir: None,
            loud: false,
            provision: true,
        })
    );
}

#[test]
fn the_consent_variable_off_means_no_build_and_no_owner() {
    let s = structural();
    let none = has_index(&[]);
    match decide_owner(RailgunMode::Auto, &s, &cands(Some("local"), false), &none) {
        Err(NoOwner::NoCorpus(why)) => assert!(why.contains("local"), "{why}"),
        other => panic!("expected NoCorpus, got {other:?}"),
    }
    // Forced: the owner exists, declines loudly, builds nothing.
    assert_eq!(
        decide_owner(RailgunMode::Force, &s, &cands(Some("local"), false), &none),
        Ok(OwnerPlan {
            corpus_dir: None,
            loud: true,
            provision: false,
        })
    );
}

#[test]
fn an_existing_corpus_is_used_and_never_rebuilt() {
    let s = structural();
    let local_only = has_index(&["local"]);
    let want = |dir: &str, loud| {
        Ok(OwnerPlan {
            corpus_dir: Some(dir.into()),
            loud,
            provision: false,
        })
    };
    // The pack's corpus wins over the per-user one; the per-user one is found.
    let both = has_index(&["pack", "local"]);
    for mode in [RailgunMode::Auto, RailgunMode::Force] {
        let loud = mode == RailgunMode::Force;
        assert_eq!(
            decide_owner(mode, &s, &cands(Some("local"), true), &both),
            want("pack", loud)
        );
        assert_eq!(
            decide_owner(mode, &s, &cands(Some("local"), true), &local_only),
            want("local", loud)
        );
    }
}

#[test]
fn an_explicit_directory_without_a_corpus_is_never_replaced_by_a_build() {
    let s = structural();
    let none = has_index(&[]);
    let explicit = Candidates {
        env: Some("env"),
        ..cands(Some("local"), true)
    };
    assert!(matches!(
        decide_owner(RailgunMode::Auto, &s, &explicit, &none),
        Err(NoOwner::NoCorpus(w)) if w.contains("env")
    ));
}

#[test]
fn a_structurally_unsuitable_context_never_builds_a_corpus() {
    let mut s = structural();
    s.arch = "gfx1151";
    let none = has_index(&[]);
    assert_eq!(
        decide_owner(RailgunMode::Auto, &s, &cands(Some("local"), true), &none),
        Err(NoOwner::Refused(RaonRailgunRefusal::WrongArch {
            arch: "gfx1151".into()
        }))
    );
}

#[test]
fn a_refused_build_declines_the_step_with_the_exact_reason() {
    let (s, mut step) = ok_facts();
    step.corpus = CorpusState::Refused("railgun-jit-corpus not found");
    assert_eq!(
        check_admission(&s, &step),
        Err(RaonRailgunRefusal::CorpusProvision(
            "railgun-jit-corpus not found".into()
        ))
    );
}

/// A spec whose output parent cannot be created (it sits under a regular
/// file), so a run is refused before anything is spawned, whoever runs it.
fn blocked_spec(tag: &str) -> (ProvisionSpec, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/railgun-owner-tests")
        .join(format!("{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let hot = root.join("cache/gfx1201");
    std::fs::create_dir_all(&hot).expect("hot dir");
    std::fs::write(root.join("blocker"), b"file").expect("blocker");
    let spec = ProvisionSpec {
        arch: "gfx1201".into(),
        program: "raon-ar-decode-gfx1201".into(),
        toolchain_id: "t".into(),
        hot_dir: hot,
        pack_dir: None,
        out_dir: root.join("blocker/railgun-corpus/gfx1201"),
    };
    (spec, root)
}

#[test]
fn the_build_is_pending_until_run_and_runs_only_once_per_process() {
    let plan = OwnerPlan {
        corpus_dir: None,
        loud: false,
        provision: true,
    };
    let (spec, root) = blocked_spec("once");
    let mut o = RaonRailgunOwner::new(plan, GFX1201, Some(spec), "model".into());
    assert_eq!(o.provision_status(), "pending");
    // The run is refused (the corpus parent cannot be created): that is the
    // owner's final answer for this process.
    o.run_provision(&mut || Ok(()), &|| None);
    let first = o.provision_status();
    assert!(first.starts_with("refused:"), "{first}");
    assert!(o.corpus_dir.is_none());
    // A second call has nothing pending: it neither re-runs nor changes state.
    o.run_provision(&mut || Ok(()), &|| None);
    assert_eq!(o.provision_status(), first);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn an_owner_with_a_corpus_never_builds() {
    let o = owner(OwnerPlan {
        corpus_dir: Some("corpus".into()),
        loud: false,
        provision: false,
    });
    assert_eq!(o.provision_status(), "off");
}

// ------------------------------------------------- hook order, O(1) declines

/// The `from-cache` fake: refuses with a missing-entry report unless the
/// transport module is in the hot cache directory it is given.
#[cfg(unix)]
const NEEDS_TRANSPORT: &str = "#!/bin/sh\n\
    OUT=\"\"; CACHE=\"\"\n\
    while [ $# -gt 0 ]; do [ \"$1\" = --out ] && OUT=\"$2\"; [ \"$1\" = --cache ] && CACHE=\"$2\"; shift; done\n\
    if [ -f \"$CACHE/railgun_copy.hsaco\" ]; then mkdir -p \"$OUT\" && echo '{}' > \"$OUT/index.json\"; \
    echo '{\"status\":\"ok\"}'; \
    else echo '{\"status\":\"refused\",\"reason\":\"no hot cache entry for railgun_copy\",\"missing\":[{\"module\":\"railgun_copy\"}]}'; exit 2; fi\n";

#[cfg(unix)]
fn fake_binary(root: &Path, text: &str) -> PathBuf {
    let path = root.join("from-cache-fake");
    crate::railgun_provision::tests::write_executable(&path, text);
    path
}

fn ordered_spec(tag: &str) -> (ProvisionSpec, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/railgun-owner-tests")
        .join(format!("{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let hot = root.join("cache/gfx1201");
    std::fs::create_dir_all(&hot).expect("hot dir");
    let spec = ProvisionSpec {
        arch: "gfx1201".into(),
        program: "raon-ar-decode-gfx1201".into(),
        toolchain_id: "t".into(),
        hot_dir: hot,
        pack_dir: None,
        out_dir: root.join("cache/railgun-corpus/gfx1201"),
    };
    (spec, root)
}

fn pending_owner(spec: ProvisionSpec) -> RaonRailgunOwner {
    RaonRailgunOwner::new(
        OwnerPlan {
            corpus_dir: None,
            loud: false,
            provision: true,
        },
        GFX1201,
        Some(spec),
        "model".into(),
    )
}

#[cfg(unix)]
#[test]
fn the_transport_kernels_are_loaded_before_the_subcommand_is_spawned() {
    let (spec, root) = ordered_spec("order");
    let hot = spec.hot_dir.clone();
    let script = fake_binary(&root, NEEDS_TRANSPORT);
    // The subcommand is spawned under the crate-wide spawn lock (see
    // `railgun_provision::tests::spawn_lock`): a fork elsewhere in this test
    // binary must not hold a write descriptor of any script while it execs.
    let _spawn = crate::railgun_provision::tests::spawn_lock();
    let mut o = pending_owner(spec);
    let prepared = Cell::new(0);
    // The load puts the module in the hot cache, as `railgun_transport_digests`
    // does; the subcommand only succeeds if it is there when it runs.
    o.run_provision(
        &mut || {
            prepared.set(prepared.get() + 1);
            std::fs::write(hot.join("railgun_copy.hsaco"), b"obj").map_err(|e| e.to_string())
        },
        &|| Some(script.clone()),
    );
    assert_eq!(prepared.get(), 1);
    let status = o.provision_status();
    assert!(status.starts_with("done:"), "{status}");
    assert!(o.corpus_dir.is_some() && o.permanent.is_none());
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn without_the_pre_spawn_load_the_missing_entry_refusal_is_not_persisted() {
    let (spec, root) = ordered_spec("missing");
    let marker = spec.hot_dir.join(crate::railgun_provision::MARKER);
    let script = fake_binary(&root, NEEDS_TRANSPORT);
    let _spawn = crate::railgun_provision::tests::spawn_lock();
    let mut o = pending_owner(spec.clone());
    o.run_provision(&mut || Ok(()), &|| Some(script.clone()));
    assert!(
        o.provision_status().contains("no hot cache entry"),
        "{}",
        o.provision_status()
    );
    assert!(
        !marker.exists(),
        "a missing-entry refusal must not be recorded"
    );
    // A fresh process (a fresh owner) tries again, and now succeeds.
    std::fs::write(spec.hot_dir.join("railgun_copy.hsaco"), b"obj").unwrap();
    let mut o2 = pending_owner(spec);
    o2.run_provision(&mut || Ok(()), &|| Some(script.clone()));
    assert!(o2.provision_status().starts_with("done:"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_failing_pre_spawn_load_fails_the_build_without_spawning_or_recording() {
    let (spec, root) = ordered_spec("prepfail");
    let marker = spec.hot_dir.join(crate::railgun_provision::MARKER);
    let mut o = pending_owner(spec);
    let spawned = Cell::new(false);
    o.run_provision(&mut || Err("hipcc failed".into()), &|| {
        spawned.set(true);
        None
    });
    let s = o.provision_status();
    assert!(
        s.starts_with("refused:") && s.contains("hipcc failed"),
        "{s}"
    );
    assert!(!spawned.get() && !marker.exists());
    assert!(o.permanent.is_some());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_failed_build_latches_and_every_later_decline_is_the_same_allocation() {
    let (spec, root) = ordered_spec("latch");
    let mut o = pending_owner(spec);
    o.run_provision(&mut || Ok(()), &|| None);
    let latched = o.permanent.clone().expect("the refusal latches");
    assert_eq!(latched.kind(), "corpus_provision");
    // Replaying the latched refusal (what every later step does) allocates
    // nothing: the stored decline stays the very same `Arc`.
    o.note_decline(Arc::clone(&latched));
    let first = o.last_decline.clone().unwrap();
    for _ in 0..1000 {
        let again = Arc::clone(o.permanent.as_ref().unwrap());
        o.note_decline(again);
    }
    assert!(Arc::ptr_eq(&first, o.last_decline.as_ref().unwrap()));
    let s = o.stats();
    assert_eq!(s.declined_by_kind["corpus_provision"], 1001);
    assert_eq!(s.declined, 1001);
    assert_eq!(
        s.last_refusal.as_deref(),
        Some(latched.to_string().as_str())
    );
    // A repeated freshly computed unit refusal reuses its allocation too.
    o.decline(RaonRailgunRefusal::AbsorbedRoute);
    let a = o.last_decline.clone().unwrap();
    o.decline(RaonRailgunRefusal::AbsorbedRoute);
    assert!(Arc::ptr_eq(&a, o.last_decline.as_ref().unwrap()));
    let _ = std::fs::remove_dir_all(root);
}
