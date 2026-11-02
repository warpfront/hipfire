// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Design A §A9.5 fixture 5 on the actual gfx1201 object
//! `kv_cache_write_fp8_e4m3_batched` (`kernels/src/kv_cache_write_*_batched.hip`,
//! unbundled from the Whole16 corpus pack): the receipt is Conditional with
//! guards and indexed effects, round-trips with every width/PC/guard/index
//! source intact, fails admission when any of them changes, and its Whole16
//! manifest node is `conditional, witness required`, never proven.
use railgun_cert::whole16::{build_with, ClassSpec, SequenceEntry};
use railgun_cert::{
    case_digest, certify, sha256_hex, KernelRecord, Receipt, STATUS_CONDITIONAL, STATUS_PROVEN,
};
use railgun_corpus::window::{ConditionalWitness, NodeRoute, RuntimeNode, NODE_CONDITIONAL};

const OBJECT: &[u8] = include_bytes!("fixtures/kv_cache_write_fp8_e4m3_batched.gfx1201.hsaco");
const SYMBOL: &str = "kv_cache_write_fp8_e4m3_batched";
const PIN: &str = "0000000000000000000000000000000000000000000000000000000000000001";

fn receipt() -> Receipt {
    certify(OBJECT, None).expect("certifies")
}
fn writer(r: &Receipt) -> &KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == SYMBOL)
        .expect("writer present")
}
/// Would runtime admission accept this (mutated) record of this object?
fn admits(r: &Receipt, k: &KernelRecord) -> bool {
    k.validate_case(&r.object_sha256, &r.device_elf_sha256, &r.arch)
        .is_ok()
}

#[test]
fn writer_is_conditional_with_guards_effects_and_a_witness_requirement() {
    let r = receipt();
    let k = writer(&r);
    assert_eq!(
        k.status, STATUS_CONDITIONAL,
        "unresolved: {:?}",
        k.unguarded_unresolved
    );
    assert_ne!(k.status, STATUS_PROVEN);
    assert!(!k.guards.is_empty() && !k.indexed_effects.is_empty());
    assert_eq!(k.required_witnesses, ["indexed_bind"]);
    assert!(k.unresolved.is_empty());
    assert!(
        !k.unguarded_unresolved.is_empty(),
        "unguarded refusals stay as diagnostics"
    );
    assert_eq!(r.schema_version, 9);
    assert_eq!(r.railgun.section_version, 3);
    // The object's actual stores into dst (+0): one 8-byte code store
    // (global_store_b64, the compiler merges the 4+4 code words) and one
    // 2-byte scale store; the index load is a 4-byte i32 of positions (+16).
    let widths: Vec<u32> = k.indexed_effects.iter().map(|e| e.width).collect();
    assert_eq!(widths, [8, 2]);
    for e in &k.indexed_effects {
        assert_eq!(e.base_arg_offset, 0);
        assert!(e
            .index_loads
            .iter()
            .all(|l| l.source_arg_offset == 16 && l.element == "i32"));
        assert!(!e.index_loads.is_empty());
    }
    // G8 reports the guarded summary and its guards.
    assert!(k.differential.guarded);
    assert_eq!(k.differential.guards, k.guards);
    assert_ne!(k.differential.status, "fail", "{:?}", k.differential.args);
    assert!(admits(&r, k));
}

#[test]
fn receipt_round_trip_keeps_every_fact_and_tampering_fails_admission() {
    let r = receipt();
    let back: Receipt = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    let (k, b) = (writer(&r), writer(&back));
    assert_eq!(b.status, k.status);
    assert_eq!(b.guards, k.guards);
    assert_eq!(b.indexed_effects, k.indexed_effects);
    assert_eq!(b.required_witnesses, k.required_witnesses);
    assert_eq!(b.case_digest, k.case_digest);
    assert!(admits(&back, b));

    let edits: Vec<(&str, Box<dyn Fn(&mut KernelRecord)>)> = vec![
        ("width", Box::new(|k| k.indexed_effects[0].width += 1)),
        ("pc", Box::new(|k| k.indexed_effects[0].pc += 4)),
        (
            "mode",
            Box::new(|k| k.indexed_effects[0].mode = "read".into()),
        ),
        ("cache", Box::new(|k| k.indexed_effects[0].cache.push('x'))),
        (
            "expr",
            Box::new(|k| k.indexed_effects[0].address_expr.push('x')),
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
            Box::new(|k| k.indexed_effects[0].index_loads[0].offset_expr.push('x')),
        ),
        (
            "load-ext",
            Box::new(|k| k.indexed_effects[0].index_loads[0].extension.push('x')),
        ),
        ("guard", Box::new(|k| k.guards[0].value = "1".into())),
        (
            "guard-removed",
            Box::new(|k| {
                k.guards.pop();
            }),
        ),
        (
            "effect-removed",
            Box::new(|k| {
                k.indexed_effects.pop();
            }),
        ),
        ("witness", Box::new(|k| k.required_witnesses.clear())),
        ("status", Box::new(|k| k.status = STATUS_PROVEN.into())),
        ("digest", Box::new(|k| k.case_digest = "0".repeat(64))),
    ];
    for (name, edit) in &edits {
        let mut m = k.clone();
        edit(&mut m);
        assert!(!admits(&r, &m), "{name}");
    }
    // Digest of this case under another object or device ELF is not this one's.
    let mut other = r.clone();
    other.object_sha256 = sha256_hex(b"other object");
    assert!(!admits(&other, k));
    let mut other = r.clone();
    other.device_elf_sha256 = sha256_hex(b"other elf");
    assert!(!admits(&other, k));
    assert_ne!(
        k.case_digest,
        case_digest(&r.object_sha256, &r.device_elf_sha256, "gfx1100", k)
    );
}

#[test]
fn whole16_node_is_conditional_witness_required_and_never_proven() {
    let classes = [ClassSpec {
        id: "writer",
        any_of: &[SYMBOL],
        exact: true,
    }];
    let entries = [SequenceEntry {
        phase: "verify_body".into(),
        symbol: SYMBOL.into(),
        route: NodeRoute::Railgun,
        object: Some(OBJECT.to_vec()),
    }];
    let m = build_with(&entries, "gfx1201", PIN, &classes)
        .expect("manifest accepts the conditional node");
    let n = &m.nodes[0];
    assert_eq!(n.a1_status, NODE_CONDITIONAL);
    assert_eq!(n.case_sha256, writer(&receipt()).case_digest);
    assert_eq!(n.required_witnesses, ["indexed_bind"]);
    let plan = [RuntimeNode {
        symbol: SYMBOL,
        artifact_sha256: &sha256_hex(OBJECT),
        route: NodeRoute::Railgun,
    }];
    // Legacy admission cannot treat Conditional as Proven.
    let e = m
        .admit("gfx1201", PIN, &plan)
        .expect_err("legacy admit refuses");
    assert!(e
        .iter()
        .any(|r| r.reason.contains("conditional, witness required")));
    let w = ConditionalWitness {
        index: 0,
        case_sha256: &n.case_sha256,
        kind: "indexed_bind",
    };
    m.admit_with("gfx1201", PIN, &plan, &[w])
        .expect("matching witness admits");
    let wrong = ConditionalWitness {
        index: 0,
        case_sha256: "00",
        kind: "indexed_bind",
    };
    assert!(m.admit_with("gfx1201", PIN, &plan, &[wrong]).is_err());
}
