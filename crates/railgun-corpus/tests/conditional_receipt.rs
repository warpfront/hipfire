// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Design A §A9.5 fixture 5, runtime side: a schema-8 conditional receipt
//! round-trips with widths, PCs, guards, index source and status intact;
//! changing or removing any of them, the object/case digest or the schema
//! fails admission; and a conditional manifest node is never admitted as
//! proven.
use std::collections::BTreeMap;
use std::path::PathBuf;

use railgun_corpus::receipt::*;
use railgun_corpus::window::*;
use railgun_corpus::{
    sha256_hex, ArtifactKey, BuildHost, CorpusIndex, CorpusRecord, JitCorpus, KernelVerdict,
    Lookup, ObjectOrigin, ToolchainPin, CORPUS_INDEX, CORPUS_SCHEMA, CORPUS_VERSION,
};

const SYMBOL: &str = "kv_cache_write_fp8_e4m3_batched";

fn guard(kind: &str, off: u32, size: u32, value: &str) -> GuardRecord {
    GuardRecord {
        kind: kind.into(),
        arg_offset: off,
        size,
        value: value.into(),
    }
}

fn index_load() -> IndexLoadRecord {
    IndexLoadRecord {
        pc: 0x44,
        source_arg_offset: 16,
        element: "i32".into(),
        offset_expr: "4*workgroup_id_y".into(),
        extension: "sext32".into(),
        class: "smem_load".into(),
        path: "smem".into(),
        cache: "th=0 scope=0 glc=0 slc=0 dlc=0 nv=0".into(),
        width: 4,
    }
}

fn effect() -> IndexedEffectRecord {
    IndexedEffectRecord {
        pc: 0x1c0,
        base_arg_offset: 0,
        mode: "write".into(),
        class: "vmem_store".into(),
        path: "global".into(),
        cache: "th=0 scope=0 glc=0 slc=0 dlc=0 nv=0".into(),
        width: 4,
        address_expr: "arg0 + sext64(load_i32(arg16 + 4*wg_y))*(arg24*258) + wg_x*256 + lane*8"
            .into(),
        index_loads: vec![index_load()],
        guards: vec!["scalar_eq@28".into()],
    }
}

fn arg(offset: u32, size: u32, role: &str, mode: Option<&str>) -> ArgRecord {
    ArgRecord {
        offset,
        size,
        index: None,
        name: format!("a{offset}"),
        value_kind: "global_buffer".into(),
        role: role.into(),
        mode: mode.map(str::to_owned),
        bound: None,
        classes: vec![],
        paths: vec![],
        cache_bits: vec![],
        accesses: 0,
        shared: false,
        read_cache: "not_read".into(),
        sinks: vec![],
        loaded: true,
        null_guarded: false,
    }
}

fn conditional(object: &str, elf: &str) -> KernelRecord {
    let guards = vec![
        guard("scalar_eq", 28, 4, "256"),
        guard("scalar_eq", 32, 4, "16"),
        guard("ptr_null", 40, 8, "null"),
        guard("ptr_null", 48, 8, "null"),
        guard("grid", 0, 0, "H,16,1"),
        guard("block", 0, 0, "32,1,1"),
    ];
    let mut k = KernelRecord {
        symbol: SYMBOL.into(),
        status: STATUS_CONDITIONAL.into(),
        unresolved: vec![],
        kernarg_size: 56,
        read_cache: "scalar_and_vmem".into(),
        dynamic_kernarg_reads: false,
        taint_overflow: false,
        side: SideRecord::default(),
        implicit: ImplicitRecord::default(),
        args: vec![
            arg(0, 8, "pointer", Some("write")),
            arg(16, 8, "pointer", Some("read")),
        ],
        dwords: vec![],
        unguarded_unresolved: vec![],
        differential: Differential {
            source: "object-metadata".into(),
            status: "pass".into(),
            args: vec![],
            radiowave_mutable_read_cache: None,
            guards: guards.clone(),
            guarded: true,
        },
        pm_obligations: BTreeMap::new(),
        pm_analyze_error: None,
        guards,
        indexed_effects: vec![effect()],
        sealed_tables: vec![],
        descriptor_arenas: vec![],
        required_witnesses: vec!["indexed_bind".into()],
        case_digest: String::new(),
        guard_flag: None,
        task_protocol: None,
        wg_footprints: Vec::new(),
    };
    k.case_digest = case_digest(object, elf, "gfx1201", &k);
    k
}

fn receipt(object: &str, elf: &str, k: KernelRecord) -> Receipt {
    Receipt {
        schema: RECEIPT_SCHEMA.into(),
        schema_version: RECEIPT_SCHEMA_VERSION,
        object_sha256: object.into(),
        device_elf_sha256: elf.into(),
        arch: "gfx1201".into(),
        strict: false,
        tier: "D".into(),
        lifter: Tool {
            name: "l".into(),
            crate_version: "0".into(),
            git_sha: "g".into(),
        },
        certifier: Tool {
            name: "c".into(),
            crate_version: "0".into(),
            git_sha: "g".into(),
        },
        round_trip_proof_sha256: String::new(),
        kernels: vec![],
        checks: vec![],
        obligations: vec![],
        railgun: RailgunSection {
            section_version: RAILGUN_SECTION_VERSION,
            analysis: "test".into(),
            assumptions: vec![],
            kernels: vec![k],
        },
    }
}

struct Fixture {
    root: PathBuf,
    object: String,
    pin: String,
}

impl Fixture {
    /// Write a one-record corpus whose receipt is `bytes`. The index carries
    /// the receipt's real digest, so only receipt content can fail admission.
    fn new(tag: &str, bytes: &[u8], object: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("railgun-corpus-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("r.json"), bytes).unwrap();
        let id = "hipcc 1".to_owned();
        let pin = sha256_hex(id.as_bytes());
        let index = CorpusIndex {
            schema: CORPUS_SCHEMA.into(),
            version: CORPUS_VERSION,
            source_commit: "c".into(),
            build_host: BuildHost {
                key: "k".into(),
                hostname: "h".into(),
                hipcc: "h".into(),
                rocm_env_root: None,
                clang_sha256: "c".into(),
                header_set_sha256: "h".into(),
                build_dir: "/b".into(),
            },
            toolchain: ToolchainPin {
                pin: pin.clone(),
                id,
                version_sha256: "v".into(),
            },
            records: vec![CorpusRecord {
                arch: "gfx1201".into(),
                artifact_sha256: object.into(),
                module: "m".into(),
                origin: ObjectOrigin::Hipcc,
                symbols: vec![SYMBOL.into()],
                source_sha256: "s".into(),
                extra_flags: String::new(),
                flags: vec![],
                scheduler_profile: String::new(),
                packaging_key: "p".into(),
                jit_cache_key: "j".into(),
                argv: vec![],
                object: "o.hsaco".into(),
                receipt: "r.json".into(),
                receipt_sha256: sha256_hex(bytes),
                kernels: vec![KernelVerdict {
                    symbol: SYMBOL.into(),
                    status: "conditional".into(),
                    differential: "pass".into(),
                }],
            }],
            rejected: vec![],
            programs: vec![],
        };
        std::fs::write(root.join(CORPUS_INDEX), serde_json::to_vec(&index).unwrap()).unwrap();
        Self {
            root,
            object: object.into(),
            pin,
        }
    }

    fn lookup(&self) -> Result<Lookup, railgun_corpus::CorpusError> {
        JitCorpus::open(&self.root).unwrap().lookup(&ArtifactKey {
            arch: "gfx1201",
            toolchain_pin: &self.pin,
            artifact_sha256: &self.object,
        })
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn sha(c: char) -> String {
    std::iter::repeat(c).take(64).collect()
}

fn admits(tag: &str, r: &Receipt) -> bool {
    let f = Fixture::new(tag, &serde_json::to_vec(r).unwrap(), &r.object_sha256);
    matches!(f.lookup(), Ok(Lookup::Hit(_)))
}

#[test]
fn conditional_receipt_round_trips_and_is_admitted() {
    let (object, elf) = (sha('a'), sha('b'));
    let r = receipt(&object, &elf, conditional(&object, &elf));
    let bytes = serde_json::to_vec(&r).unwrap();
    let f = Fixture::new("roundtrip", &bytes, &object);
    let Ok(Lookup::Hit(hit)) = f.lookup() else {
        panic!("admits")
    };
    let k = &hit.receipt.railgun.kernels[0];
    let want = conditional(&object, &elf);
    assert_eq!(k.status, STATUS_CONDITIONAL);
    assert_eq!(k.guards, want.guards);
    assert_eq!(k.indexed_effects, want.indexed_effects);
    assert_eq!(k.indexed_effects[0].width, 4);
    assert_eq!(k.indexed_effects[0].pc, 0x1c0);
    assert_eq!(k.indexed_effects[0].index_loads[0].source_arg_offset, 16);
    assert_eq!(k.indexed_effects[0].index_loads[0].pc, 0x44);
    assert_eq!(k.required_witnesses, want.required_witnesses);
    assert_eq!(k.case_digest, want.case_digest);
    assert!(k.differential.guarded);
    assert_eq!(k.differential.guards, want.guards);
}

#[test]
fn tampering_any_recorded_fact_fails_admission() {
    let (object, elf) = (sha('a'), sha('b'));
    let edits: Vec<(&str, Box<dyn Fn(&mut KernelRecord)>)> = vec![
        ("width", Box::new(|k| k.indexed_effects[0].width = 8)),
        ("pc", Box::new(|k| k.indexed_effects[0].pc += 4)),
        (
            "mode",
            Box::new(|k| k.indexed_effects[0].mode = "read".into()),
        ),
        (
            "cache",
            Box::new(|k| k.indexed_effects[0].cache = "th=1 scope=0 glc=0 slc=0 dlc=0 nv=0".into()),
        ),
        (
            "expr",
            Box::new(|k| k.indexed_effects[0].address_expr.push_str("+4")),
        ),
        (
            "base",
            Box::new(|k| k.indexed_effects[0].base_arg_offset = 16),
        ),
        (
            "load-pc",
            Box::new(|k| k.indexed_effects[0].index_loads[0].pc += 4),
        ),
        (
            "load-src",
            Box::new(|k| k.indexed_effects[0].index_loads[0].source_arg_offset = 0),
        ),
        (
            "load-expr",
            Box::new(|k| {
                k.indexed_effects[0].index_loads[0].offset_expr = "4*workgroup_id_x".into()
            }),
        ),
        (
            "load-ext",
            Box::new(|k| k.indexed_effects[0].index_loads[0].extension = "zext32".into()),
        ),
        (
            "guard-value",
            Box::new(|k| k.guards[0].value = "128".into()),
        ),
        (
            "guard-removed",
            Box::new(|k| {
                k.guards.remove(2);
            }),
        ),
        ("effects-removed", Box::new(|k| k.indexed_effects.clear())),
        (
            "witness-removed",
            Box::new(|k| k.required_witnesses.clear()),
        ),
        (
            "status-proven",
            Box::new(|k| k.status = STATUS_PROVEN.into()),
        ),
        (
            "status-unknown",
            Box::new(|k| k.status = STATUS_UNKNOWN.into()),
        ),
        (
            "status-other",
            Box::new(|k| k.status = "conditional2".into()),
        ),
        ("case-digest", Box::new(|k| k.case_digest = sha('0'))),
        ("case-digest-empty", Box::new(|k| k.case_digest.clear())),
        (
            "differential-guard",
            Box::new(|k| k.differential.guards.clear()),
        ),
        (
            "differential-unguarded",
            Box::new(|k| k.differential.guarded = false),
        ),
    ];
    // The untouched record admits, so each refusal below is the edit's.
    assert!(admits(
        "base",
        &receipt(&object, &elf, conditional(&object, &elf))
    ));
    for (name, edit) in &edits {
        let mut k = conditional(&object, &elf);
        edit(&mut k); // digest deliberately not recomputed
        assert!(
            !admits(&format!("t-{name}"), &receipt(&object, &elf, k)),
            "{name}"
        );
    }
}

#[test]
fn object_digest_device_digest_arch_and_schema_fail_admission() {
    let (object, elf) = (sha('a'), sha('b'));
    // A case digest computed for another object is refused under this one.
    let k = conditional(&sha('c'), &elf);
    assert!(!admits("other-object", &receipt(&object, &elf, k)));
    let k = conditional(&object, &sha('c'));
    assert!(!admits("other-elf", &receipt(&object, &elf, k)));
    let mut r = receipt(&object, &elf, conditional(&object, &elf));
    r.arch = "gfx1100".into();
    assert!(!admits("arch", &r));
    for (name, v) in [("v7", 7), ("v8", 8), ("v10", 10)] {
        let mut r = receipt(&object, &elf, conditional(&object, &elf));
        r.schema_version = v;
        assert!(!admits(name, &r), "{name}");
    }
    for (name, v) in [("s1", 1), ("s2", 2), ("s4", 4)] {
        let mut r = receipt(&object, &elf, conditional(&object, &elf));
        r.railgun.section_version = v;
        assert!(!admits(name, &r), "{name}");
    }
    let mut r = receipt(&object, &elf, conditional(&object, &elf));
    r.schema = "peacemaker-receipt-x".into();
    assert!(!admits("schema-name", &r));
    // The receipt object digest must be the looked-up artifact.
    let r = receipt(&object, &elf, conditional(&object, &elf));
    let f = Fixture::new("wrong-key", &serde_json::to_vec(&r).unwrap(), &sha('d'));
    assert!(f.lookup().is_err());
}

#[test]
fn missing_fields_and_v1_rows_fail_closed() {
    let (object, elf) = (sha('a'), sha('b'));
    let r = receipt(&object, &elf, conditional(&object, &elf));
    let json = serde_json::to_value(&r).unwrap();
    for field in [
        "guards",
        "indexed_effects",
        "required_witnesses",
        "case_digest",
    ] {
        let mut v = json.clone();
        v["railgun"]["kernels"][0]
            .as_object_mut()
            .unwrap()
            .remove(field);
        let f = Fixture::new(
            &format!("missing-{field}"),
            &serde_json::to_vec(&v).unwrap(),
            &object,
        );
        assert!(f.lookup().is_err(), "{field}");
    }
    let mut v = json.clone();
    v["railgun"]["kernels"][0]["differential"]
        .as_object_mut()
        .unwrap()
        .remove("guards");
    let f = Fixture::new(
        "missing-diff-guards",
        &serde_json::to_vec(&v).unwrap(),
        &object,
    );
    assert!(f.lookup().is_err());
    // A proven row may not carry guarded fields, and a v1-shaped conditional
    // row (no case) is never promoted.
    let mut k = conditional(&object, &elf);
    k.status = STATUS_PROVEN.into();
    k.case_digest.clear();
    assert!(!admits("proven-with-effects", &receipt(&object, &elf, k)));
    let mut k = conditional(&object, &elf);
    k.guards.clear();
    k.indexed_effects.clear();
    k.required_witnesses.clear();
    k.case_digest.clear();
    k.differential.guards.clear();
    k.differential.guarded = false;
    assert!(!admits(
        "conditional-without-case",
        &receipt(&object, &elf, k.clone())
    ));
    k.status = STATUS_PROVEN.into();
    assert!(admits("plain-proven", &receipt(&object, &elf, k)));
}

fn node(i: u32, status: &str, case: &str, witnesses: &[&str]) -> WindowNodeRecord {
    WindowNodeRecord {
        index: i,
        phase: "verify_body".into(),
        class: String::new(),
        symbol: format!("n{i}"),
        artifact_sha256: format!("a{i}"),
        device_elf_sha256: "d".into(),
        text_sha256: "t".into(),
        kd_sha256: "k".into(),
        metadata_sha256: "m".into(),
        receipt_sha256: "r".into(),
        a1_status: status.into(),
        case_sha256: case.into(),
        required_witnesses: witnesses.iter().map(|s| (*s).to_owned()).collect(),
        kernarg_size: 16,
        explicit_end: 16,
        hidden_loaded: vec![],
    }
}

fn manifest(nodes: Vec<WindowNodeRecord>) -> WindowManifest {
    WindowManifest {
        schema: WINDOW_MANIFEST_SCHEMA.into(),
        version: WINDOW_MANIFEST_VERSION,
        shape: WHOLE16_SHAPE.into(),
        arch: "gfx1201".into(),
        toolchain_pin: "pin".into(),
        sequence_sha256: sequence_digest(&nodes),
        nodes,
        abi: vec![],
    }
}

fn plan() -> Vec<RuntimeNode<'static>> {
    (0..2)
        .map(|i| RuntimeNode {
            symbol: ["n0", "n1"][i],
            artifact_sha256: ["a0", "a1"][i],
            route: NodeRoute::Railgun,
        })
        .collect()
}

#[test]
fn conditional_node_is_not_proven_without_a_matching_witness() {
    let m = manifest(vec![
        node(0, "proven", "", &[]),
        node(1, "conditional", "case1", &["indexed_bind"]),
    ]);
    // The legacy admit path never admits a conditional node.
    let e = m.admit("gfx1201", "pin", &plan()).expect_err("refused");
    assert!(e
        .iter()
        .any(|r| r.reason.contains("conditional, witness required")));
    let w = |index, case, kind| ConditionalWitness {
        index,
        case_sha256: case,
        kind,
    };
    for bad in [
        vec![],
        vec![w(0, "case1", "indexed_bind")],
        vec![w(1, "case2", "indexed_bind")],
        vec![w(1, "case1", "other")],
    ] {
        assert!(m.admit_with("gfx1201", "pin", &plan(), &bad).is_err());
    }
    m.admit_with("gfx1201", "pin", &plan(), &[w(1, "case1", "indexed_bind")])
        .expect("matching witness admits");
    // A conditional node needs its case and witnesses recorded; tampering the
    // status of a node breaks the sequence digest.
    let mut bad = m.clone();
    bad.nodes[1].a1_status = "proven".into();
    assert!(bad.admit("gfx1201", "pin", &plan()).is_err());
    let mut bad = m.clone();
    bad.nodes[1].case_sha256.clear();
    bad.sequence_sha256 = sequence_digest(&bad.nodes);
    assert!(bad
        .admit_with("gfx1201", "pin", &plan(), &[w(1, "case1", "indexed_bind")])
        .is_err());
}

// ---------------------------------------------------------------------------
// Sealed-table cases (schema 9 / section 3).
// ---------------------------------------------------------------------------

const CACHE0: &str = "th=0 scope=0 glc=0 slc=0 dlc=0 nv=0";

fn sealed_table() -> SealedTableRecord {
    SealedTableRecord {
        table_arg_offset: 0,
        bound_kind: SEALED_BOUND_KERNARG.into(),
        bound_value: 52,
        entry_bytes: 8,
        entries: vec![SealedEntryRecord {
            pc: 70,
            class: "smem_load".into(),
            path: "smem".into(),
            cache: CACHE0.into(),
            width: 8,
            index_expr: "load@16".into(),
            guard_pcs: vec![20],
        }],
        accesses: vec![SealedAccessRecord {
            pc: 112,
            class: "vmem_load".into(),
            path: "flat".into(),
            cache: CACHE0.into(),
            width: 4,
            entry_pcs: vec![70],
        }],
        null_guarded: false,
    }
}

fn sealed(object: &str, elf: &str) -> KernelRecord {
    let mut k = conditional(object, elf);
    k.guards.clear();
    k.differential.guards.clear();
    k.indexed_effects.clear();
    k.sealed_tables = vec![sealed_table()];
    k.required_witnesses = vec![WITNESS_SEALED_TABLE.into()];
    k.case_digest = case_digest(object, elf, "gfx1201", &k);
    k
}

#[test]
fn sealed_receipt_round_trips_and_is_admitted() {
    let (object, elf) = (sha('a'), sha('b'));
    let r = receipt(&object, &elf, sealed(&object, &elf));
    let f = Fixture::new(
        "sealed-roundtrip",
        &serde_json::to_vec(&r).unwrap(),
        &object,
    );
    let Ok(Lookup::Hit(hit)) = f.lookup() else {
        panic!("admits")
    };
    let k = &hit.receipt.railgun.kernels[0];
    assert_eq!(k.status, STATUS_CONDITIONAL);
    assert_eq!(k.sealed_tables, vec![sealed_table()]);
    assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE]);
    assert!(k.guards.is_empty() && k.indexed_effects.is_empty());
    // A kernel without sealed tables serialises without the field at all.
    let plain = serde_json::to_value(conditional(&object, &elf)).unwrap();
    assert!(plain.get("sealed_tables").is_none());
}

#[test]
fn every_sealed_field_the_digest_binds_fails_admission() {
    let (object, elf) = (sha('a'), sha('b'));
    let edits: Vec<(&str, Box<dyn Fn(&mut KernelRecord)>)> = vec![
        (
            "table arg",
            Box::new(|k| k.sealed_tables[0].table_arg_offset = 8),
        ),
        ("bound", Box::new(|k| k.sealed_tables[0].bound_value = 56)),
        (
            "bound kind",
            Box::new(|k| k.sealed_tables[0].bound_kind = SEALED_BOUND_CONST.into()),
        ),
        (
            "bound kind unknown",
            Box::new(|k| k.sealed_tables[0].bound_kind = "x".into()),
        ),
        (
            "bound misaligned",
            Box::new(|k| k.sealed_tables[0].bound_value = 53),
        ),
        (
            "entry width",
            Box::new(|k| k.sealed_tables[0].entry_bytes = 4),
        ),
        (
            "entry pc",
            Box::new(|k| k.sealed_tables[0].entries[0].pc = 71),
        ),
        (
            "entry guard",
            Box::new(|k| k.sealed_tables[0].entries[0].guard_pcs[0] = 21),
        ),
        (
            "entry unguarded",
            Box::new(|k| k.sealed_tables[0].entries[0].guard_pcs.clear()),
        ),
        ("no entry", Box::new(|k| k.sealed_tables[0].entries.clear())),
        (
            "access class store",
            Box::new(|k| k.sealed_tables[0].accesses[0].class = "vmem_store".into()),
        ),
        (
            "access entry",
            Box::new(|k| k.sealed_tables[0].accesses[0].entry_pcs = vec![71]),
        ),
        (
            "access no entry",
            Box::new(|k| k.sealed_tables[0].accesses[0].entry_pcs.clear()),
        ),
        ("table dropped", Box::new(|k| k.sealed_tables.clear())),
        (
            "witness",
            Box::new(|k| k.required_witnesses = vec![WITNESS_INDEXED_BIND.into()]),
        ),
        (
            "witness extra",
            Box::new(|k| k.required_witnesses.push(WITNESS_INDEXED_BIND.into())),
        ),
        (
            "guards added",
            Box::new(|k| k.guards = vec![guard("scalar_eq", 28, 4, "1")]),
        ),
        (
            "duplicate table",
            Box::new(|k| {
                let t = k.sealed_tables[0].clone();
                k.sealed_tables.push(t);
            }),
        ),
    ];
    for (name, edit) in &edits {
        // Re-sealing the digest after the edit must not rescue a malformed
        // case; an un-resealed edit fails on the digest.
        let mut stale = sealed(&object, &elf);
        edit(&mut stale);
        assert!(
            !admits(&format!("sealed-{name}"), &receipt(&object, &elf, stale)),
            "{name}"
        );
        let mut resealed = sealed(&object, &elf);
        edit(&mut resealed);
        resealed.case_digest = case_digest(&object, &elf, "gfx1201", &resealed);
        let structurally_valid = resealed.validate_case(&object, &elf, "gfx1201").is_ok();
        assert!(
            !structurally_valid
                || matches!(
                    *name,
                    "entry pc" | "bound" | "bound kind" | "table arg" | "entry guard"
                ),
            "{name}: re-sealed malformed case must be rejected"
        );
    }
    // The digest names the object and device ELF.
    let k = sealed(&object, &elf);
    assert!(!admits(
        "sealed-other-elf",
        &receipt(&object, &sha('c'), k.clone())
    ));
    assert!(!admits("sealed-other-object", &receipt(&sha('d'), &elf, k)));
}

#[test]
fn an_older_schema_receipt_is_never_promoted_to_sealed() {
    let (object, elf) = (sha('a'), sha('b'));
    // A v8 / section 2 receipt, even carrying a perfectly valid sealed case.
    let mut r = receipt(&object, &elf, sealed(&object, &elf));
    r.schema_version = 8;
    assert!(!admits("sealed-v8", &r));
    let mut r = receipt(&object, &elf, sealed(&object, &elf));
    r.railgun.section_version = 2;
    assert!(!admits("sealed-s2", &r));
    // A v8 receipt of an indexed kernel is refused too: its digest scheme
    // (`railgun-case-v1`) is not this one's.
    let mut r = receipt(&object, &elf, conditional(&object, &elf));
    r.schema_version = 8;
    r.railgun.section_version = 2;
    assert!(!admits("indexed-v8", &r));
    // A conditional row whose only effect field is dropped is not a
    // sealed-table case and is not promoted into one.
    let mut k = sealed(&object, &elf);
    k.sealed_tables.clear();
    assert!(!admits("sealed-without-tables", &receipt(&object, &elf, k)));
    // `unknown`/`proven` rows carry no sealed tables.
    for status in [STATUS_PROVEN, STATUS_UNKNOWN] {
        let mut k = sealed(&object, &elf);
        k.status = status.into();
        assert!(!admits(
            &format!("sealed-as-{status}"),
            &receipt(&object, &elf, k)
        ));
    }
}

#[test]
fn window_nodes_name_only_known_witness_kinds() {
    let m = manifest(vec![
        node(0, "proven", "", &[]),
        node(1, "conditional", "case1", &[WITNESS_SEALED_TABLE]),
    ]);
    let w = |kind| ConditionalWitness {
        index: 1,
        case_sha256: "case1",
        kind,
    };
    assert!(m.admit_with("gfx1201", "pin", &plan(), &[]).is_err());
    assert!(m
        .admit_with("gfx1201", "pin", &plan(), &[w(WITNESS_INDEXED_BIND)])
        .is_err());
    m.admit_with("gfx1201", "pin", &plan(), &[w(WITNESS_SEALED_TABLE)])
        .expect("a sealed_table witness for the case admits");
    // Both kinds may be required together; an unknown or repeated kind is
    // not a manifest anyone generated.
    let both = manifest(vec![
        node(0, "proven", "", &[]),
        node(
            1,
            "conditional",
            "case1",
            &[WITNESS_INDEXED_BIND, WITNESS_SEALED_TABLE],
        ),
    ]);
    assert!(both.validate().is_empty());
    for kinds in [
        &["trust_me"][..],
        &[WITNESS_SEALED_TABLE, WITNESS_SEALED_TABLE],
    ] {
        let bad = manifest(vec![
            node(0, "proven", "", &[]),
            node(1, "conditional", "case1", kinds),
        ]);
        assert!(!bad.validate().is_empty(), "{kinds:?}");
    }
}
