// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The sealed-table certificate on the actual gfx1201 objects of two kernels
//! that load a 64-bit pointer from a table (`expert_ptrs[expert_id]`, guarded
//! by `expert_id < n_experts`) and then only read through it: each receipt is
//! Conditional with a `sealed_table` witness requirement and a case digest that
//! binds object, kernel, table argument, bound, entry loads and derived reads;
//! it round-trips, fails admission when any of them changes, is never proven,
//! and its Whole16 node is `conditional, witness required`.
use railgun_cert::whole16::{build_with, ClassSpec, SequenceEntry};
use railgun_cert::{
    case_digest, certify, sha256_hex, KernelRecord, Receipt, SEALED_BOUND_CONST,
    SEALED_BOUND_KERNARG, STATUS_CONDITIONAL, STATUS_PROVEN, WITNESS_INDEXED_BIND,
    WITNESS_SEALED_TABLE,
};
use railgun_corpus::window::{ConditionalWitness, NodeRoute, RuntimeNode, NODE_CONDITIONAL};

const GATE_UP: &[u8] = include_bytes!("fixtures/moe_gate_up_k8_indexed_batched.gfx1201.hsaco");
const DOWN: &[u8] = include_bytes!("fixtures/moe_down_k8_indexed_batched_expanded.gfx1201.hsaco");
const GATE_UP_SYMBOL: &str = "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched";
const DOWN_SYMBOL: &str = "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded";
const PIN: &str = "0000000000000000000000000000000000000000000000000000000000000001";

/// (object, symbol, bound kernarg offset of `n_experts`, derived reads)
const CASES: &[(&[u8], &str, u64, usize)] = &[
    (GATE_UP, GATE_UP_SYMBOL, 52, 14),
    (DOWN, DOWN_SYMBOL, 44, 24),
];

fn kernel<'r>(r: &'r Receipt, symbol: &str) -> &'r KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .expect("kernel present")
}
fn admits(r: &Receipt, k: &KernelRecord) -> bool {
    k.validate_case(&r.object_sha256, &r.device_elf_sha256, &r.arch)
        .is_ok()
}

#[test]
fn moe_kernels_are_conditional_sealed_tables_with_a_valid_case_digest() {
    for &(object, symbol, bound, reads) in CASES {
        let r = certify(object, None).expect("certifies");
        let k = kernel(&r, symbol);
        assert_eq!(k.status, STATUS_CONDITIONAL, "{symbol}: {:?}", k.unresolved);
        assert_ne!(k.status, STATUS_PROVEN);
        assert!(k.unresolved.is_empty(), "{symbol}");
        assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE], "{symbol}");
        assert!(k.guards.is_empty() && k.indexed_effects.is_empty());
        assert_eq!((r.schema_version, r.railgun.section_version), (9, 3));
        assert!(admits(&r, k), "{symbol}");
        let [t] = &k.sealed_tables[..] else {
            panic!("{symbol}: one table, got {:?}", k.sealed_tables)
        };
        assert_eq!(t.table_arg_offset, 0);
        assert_eq!(
            (t.bound_kind.as_str(), t.bound_value, t.entry_bytes),
            (SEALED_BOUND_KERNARG, bound, 8)
        );
        let [e] = &t.entries[..] else { panic!() };
        assert_eq!((e.class.as_str(), e.width), ("smem_load", 8));
        assert!(e.index_expr.starts_with("load@") && !e.guard_pcs.is_empty());
        // Every access the unguarded analysis refused is now a read through
        // the table's entry; none writes.
        assert_eq!(t.accesses.len(), reads, "{symbol}");
        assert_eq!(k.unguarded_unresolved.len(), reads);
        assert!(k
            .unguarded_unresolved
            .iter()
            .all(|u| u.rule == "base-memory-derived"));
        assert!(t
            .accesses
            .iter()
            .all(|a| a.class == "vmem_load" && a.entry_pcs == [e.pc]));
        // The table argument is read; only the output arguments are written.
        let table_arg = k.args.iter().find(|a| a.offset == 0).expect("arg 0");
        assert_eq!(table_arg.mode.as_deref(), Some("read"));
        assert!(r.obligations.iter().any(|o| o == "SealedTableWitness"));
        assert!(!r.obligations.iter().any(|o| o == "IndexedBindWitness"));
        assert!(r
            .railgun
            .assumptions
            .iter()
            .any(|a| a.starts_with("SealedTable:")));
        // Serialisation round trip keeps everything and still admits.
        let back: Receipt =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).expect("round trip");
        let b = kernel(&back, symbol);
        assert_eq!(b.sealed_tables, k.sealed_tables);
        assert_eq!(b.case_digest, k.case_digest);
        assert!(admits(&back, b));
    }
}

#[test]
fn any_change_to_the_sealed_case_fails_admission() {
    for &(object, symbol, _, _) in CASES {
        let r = certify(object, None).expect("certifies");
        let k = kernel(&r, symbol);
        let edits: Vec<(&str, Box<dyn Fn(&mut KernelRecord)>)> = vec![
            (
                "table arg",
                Box::new(|k| k.sealed_tables[0].table_arg_offset = 8),
            ),
            (
                "bound value",
                Box::new(|k| k.sealed_tables[0].bound_value += 4),
            ),
            (
                "bound kind",
                Box::new(|k| {
                    k.sealed_tables[0].bound_kind = SEALED_BOUND_CONST.into();
                    k.sealed_tables[0].bound_value = 8;
                }),
            ),
            (
                "bound kind unknown",
                Box::new(|k| k.sealed_tables[0].bound_kind = "i64".into()),
            ),
            (
                "entry width",
                Box::new(|k| k.sealed_tables[0].entry_bytes = 4),
            ),
            (
                "entry pc",
                Box::new(|k| k.sealed_tables[0].entries[0].pc += 1),
            ),
            (
                "entry guard pc",
                Box::new(|k| k.sealed_tables[0].entries[0].guard_pcs[0] += 1),
            ),
            (
                "entry guard dropped",
                Box::new(|k| k.sealed_tables[0].entries[0].guard_pcs.clear()),
            ),
            (
                "entry index",
                Box::new(|k| k.sealed_tables[0].entries[0].index_expr.push('x')),
            ),
            (
                "entry class",
                Box::new(|k| k.sealed_tables[0].entries[0].class = "vmem_load".into()),
            ),
            (
                "entry cache",
                Box::new(|k| k.sealed_tables[0].entries[0].cache.push('x')),
            ),
            (
                "access pc",
                Box::new(|k| k.sealed_tables[0].accesses[0].pc += 1),
            ),
            (
                "access width",
                Box::new(|k| k.sealed_tables[0].accesses[0].width += 1),
            ),
            (
                "access made a store",
                Box::new(|k| k.sealed_tables[0].accesses[0].class = "vmem_store".into()),
            ),
            (
                "access atomic",
                Box::new(|k| k.sealed_tables[0].accesses[0].class = "atomic".into()),
            ),
            (
                "access entry",
                Box::new(|k| k.sealed_tables[0].accesses[0].entry_pcs = vec![0]),
            ),
            (
                "access dropped",
                Box::new(|k| {
                    k.sealed_tables[0].accesses.pop();
                }),
            ),
            ("table dropped", Box::new(|k| k.sealed_tables.clear())),
            (
                "witness cleared",
                Box::new(|k| k.required_witnesses.clear()),
            ),
            (
                "witness swapped",
                Box::new(|k| k.required_witnesses = vec![WITNESS_INDEXED_BIND.into()]),
            ),
            (
                "witness added",
                Box::new(|k| k.required_witnesses.push(WITNESS_INDEXED_BIND.into())),
            ),
            (
                "witness unknown",
                Box::new(|k| k.required_witnesses = vec!["trust_me".into()]),
            ),
            ("status", Box::new(|k| k.status = STATUS_PROVEN.into())),
            ("digest", Box::new(|k| k.case_digest = "0".repeat(64))),
            ("symbol", Box::new(|k| k.symbol.push('x'))),
            ("kernarg size", Box::new(|k| k.kernarg_size += 8)),
        ];
        for (name, edit) in &edits {
            let mut m = k.clone();
            edit(&mut m);
            assert!(!admits(&r, &m), "{symbol}: {name} must fail admission");
        }
        // The digest binds the exact object, device ELF and arch.
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
        // One kernel's case is not the other kernel's.
        let (other_object, other_symbol) = if symbol == DOWN_SYMBOL {
            (GATE_UP, GATE_UP_SYMBOL)
        } else {
            (DOWN, DOWN_SYMBOL)
        };
        let o = certify(other_object, None).expect("certifies");
        assert_ne!(kernel(&o, other_symbol).case_digest, k.case_digest);
        assert!(!admits(&o, k));
    }
}

#[test]
fn a_sealed_case_cannot_be_dressed_as_proven_or_unknown() {
    let r = certify(GATE_UP, None).unwrap();
    let k = kernel(&r, GATE_UP_SYMBOL);
    for status in [STATUS_PROVEN, "unknown"] {
        let mut m = k.clone();
        m.status = status.into();
        assert!(!admits(&r, &m), "{status} carries a sealed case");
        m.sealed_tables.clear();
        m.required_witnesses.clear();
        m.case_digest.clear();
        assert!(admits(&r, &m), "{status} without case fields is ordinary");
    }
}

#[test]
fn whole16_node_is_conditional_with_a_sealed_table_witness_requirement() {
    for &(object, symbol, _, _) in CASES {
        let any_of: &'static [&'static str] = if symbol == DOWN_SYMBOL {
            &[DOWN_SYMBOL]
        } else {
            &[GATE_UP_SYMBOL]
        };
        let classes = [ClassSpec {
            id: "moe",
            any_of,
            exact: true,
        }];
        let entries = [SequenceEntry {
            phase: "verify_body".into(),
            symbol: symbol.into(),
            route: NodeRoute::Railgun,
            object: Some(object.to_vec()),
        }];
        let m = build_with(&entries, "gfx1201", PIN, &classes)
            .expect("manifest accepts the conditional node");
        let n = &m.nodes[0];
        assert_eq!(n.a1_status, NODE_CONDITIONAL);
        let r = certify(object, None).unwrap();
        assert_eq!(n.case_sha256, kernel(&r, symbol).case_digest);
        assert_eq!(n.required_witnesses, [WITNESS_SEALED_TABLE]);
        let plan = [RuntimeNode {
            symbol,
            artifact_sha256: &sha256_hex(object),
            route: NodeRoute::Railgun,
        }];
        let e = m
            .admit("gfx1201", PIN, &plan)
            .expect_err("legacy admit refuses");
        assert!(e
            .iter()
            .any(|r| r.reason.contains("conditional, witness required")));
        let w = ConditionalWitness {
            index: 0,
            case_sha256: &n.case_sha256,
            kind: WITNESS_SEALED_TABLE,
        };
        m.admit_with("gfx1201", PIN, &plan, &[w])
            .expect("matching witness admits");
        // The wrong kind, or another case digest, does not discharge it.
        let wrong_kind = ConditionalWitness {
            kind: WITNESS_INDEXED_BIND,
            ..w
        };
        assert!(m.admit_with("gfx1201", PIN, &plan, &[wrong_kind]).is_err());
        let wrong_case = ConditionalWitness {
            case_sha256: "00",
            ..w
        };
        assert!(m.admit_with("gfx1201", PIN, &plan, &[wrong_case]).is_err());
    }
}

const SITU: &[u8] = include_bytes!("fixtures/ffn_situ_rows_f32.gfx1201.hsaco");
const SITU_SYMBOL: &str = "ffn_situ_rows_f32";

/// `ffn_situ_rows_f32` (awq mode 2): a per-row expert index `e = row_expert[row]`
/// guarded by the signed pair `e >= 0 && e < n_experts` (compiled to two compares
/// into masks, an `or`, `exec & mask` and `vccnz`), a pointer-table entry
/// `table[e]`, and a null check on the entry before `scale[col]` is read. The
/// pointer argument `awq_a` is the table in mode 2 and a plain vector in mode 1,
/// so the read's base is an argument or an entry.
#[test]
fn awq_table_kernel_is_conditional_with_a_signed_guard_and_a_null_guard() {
    let r = certify(SITU, None).expect("certifies");
    let k = kernel(&r, SITU_SYMBOL);
    assert_eq!(k.status, STATUS_CONDITIONAL, "{:?}", k.unresolved);
    assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE]);
    assert!(admits(&r, k));
    let [t] = &k.sealed_tables[..] else {
        panic!("one table: {:?}", k.sealed_tables)
    };
    // awq_a +24, n_experts +64.
    assert_eq!(
        (t.table_arg_offset, t.bound_kind.as_str(), t.bound_value),
        (24, SEALED_BOUND_KERNARG, 64)
    );
    assert!(t.null_guarded, "the read happens only for a nonzero entry");
    let [e] = &t.entries[..] else { panic!() };
    assert_eq!(e.guard_pcs.len(), 2, "both signed compares");
    let [a] = &t.accesses[..] else { panic!() };
    assert_eq!((a.class.as_str(), a.path.as_str()), ("vmem_load", "flat"));
    assert_eq!(a.entry_pcs, [e.pc]);
    assert!(k
        .unguarded_unresolved
        .iter()
        .all(|u| u.rule == "base-memory-derived"));
    // The awq_a argument is read (mode 1 reads it directly).
    let arg = k.args.iter().find(|a| a.offset == 24).expect("awq_a");
    assert_eq!(arg.mode.as_deref(), Some("read"));
    // Dropping or forging the null guard changes the case.
    for flip in [false, true] {
        let mut m = k.clone();
        m.sealed_tables[0].null_guarded = flip;
        assert_eq!(admits(&r, &m), flip, "null_guarded={flip}");
    }
}
