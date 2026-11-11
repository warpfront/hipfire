// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The sealed-table certificate on the real standalone gfx1201 objects of the
//! two MoE consumers that P0 pool b128 hosts as regions: the grouped gate/up
//! GEMM (`07782644…`, four o4 kernels) and the top-10 down GEMV (`8c49fcc0…`,
//! two dni kernels). Each loads an expert id from a device-written id table,
//! guards it with `if (e < 0 || e >= n_experts) return;` (a signed pair, two
//! compares into masks, `or`, `exec & mask`, `vccnz`, no EXECZ branch before
//! it), zero-extends it with a separate `s_mov_b32 sN, 0`, loads the expert's
//! weight pointer from the loader-written pointer table (kernarg 0) and only
//! reads through it. Every kernel is `conditional` with one sealed table at
//! arg 0 bounded by the `n_experts` kernarg.
use std::path::PathBuf;

use railgun_cert::{
    certify, KernelRecord, Receipt, SEALED_BOUND_KERNARG, STATUS_CONDITIONAL, STATUS_PROVEN,
    WITNESS_SEALED_TABLE,
};

const GATE_UP: &str = "07782644d68283b422e3b042e64785281bdaa256d3e488b90cf50419d6d53e50";
const DOWN: &str = "8c49fcc03c571de42932eb3a40a51f0c2f71dc2ef199a850f328e903e52d5809";

/// (symbol, `n_experts` kernarg offset, entry load pc, index load, guard
/// compare pcs, derived reads)
type Case = (&'static str, u64, u32, &'static str, [u32; 2], usize);

const GATE_UP_CASES: &[Case] = &[
    (
        "gemm_mq4g256v2p_moe_grouped_top10_o4_r8_x4",
        56,
        297,
        "load@45",
        [67, 69],
        72,
    ),
    (
        "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4",
        56,
        125,
        "load@18",
        [30, 32],
        72,
    ),
    (
        "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_inverse",
        64,
        142,
        "load@27",
        [40, 42],
        72,
    ),
    (
        "gemm_mq4g256v2p_moe_grouped_top10_o4_r2_x4_inverse_k2560",
        64,
        141,
        "load@26",
        [38, 40],
        56,
    ),
];
const DOWN_CASES: &[Case] = &[
    (
        "gemv_mq4g128v2_moe_down_top10_dni_k640",
        44,
        64,
        "load@49",
        [51, 53],
        42,
    ),
    (
        "gemv_mq4g128v2p_moe_down_top10_dni_k640",
        44,
        64,
        "load@49",
        [51, 53],
        42,
    ),
];

fn object(digest: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/task_graph/standalone")
        .join(format!("{digest}.co"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn kernel<'r>(r: &'r Receipt, symbol: &str) -> &'r KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .unwrap_or_else(|| panic!("{symbol} present"))
}

fn check(digest: &str, cases: &[Case]) {
    let r = certify(&object(digest), None).expect("certifies");
    assert_eq!(
        r.railgun.kernels.len(),
        cases.len(),
        "exactly the pinned kernels"
    );
    for &(symbol, bound, entry_pc, index, guard_pcs, reads) in cases {
        let k = kernel(&r, symbol);
        assert_eq!(k.status, STATUS_CONDITIONAL, "{symbol}: {:?}", k.unresolved);
        assert_ne!(k.status, STATUS_PROVEN, "{symbol}");
        assert!(k.unresolved.is_empty(), "{symbol}: {:?}", k.unresolved);
        assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE], "{symbol}");
        assert!(
            k.guards.is_empty() && k.indexed_effects.is_empty(),
            "{symbol}"
        );
        assert!(
            k.validate_case(&r.object_sha256, &r.device_elf_sha256, &r.arch)
                .is_ok(),
            "{symbol}: case digest"
        );

        let [t] = &k.sealed_tables[..] else {
            panic!("{symbol}: one sealed table, got {:?}", k.sealed_tables)
        };
        assert_eq!(t.table_arg_offset, 0, "{symbol}");
        assert_eq!(
            (t.bound_kind.as_str(), t.bound_value, t.entry_bytes),
            (SEALED_BOUND_KERNARG, bound, 8),
            "{symbol}: bound is the n_experts kernarg"
        );
        let [e] = &t.entries[..] else {
            panic!("{symbol}: one entry load, got {:?}", t.entries)
        };
        assert_eq!(
            (e.class.as_str(), e.width, e.pc, e.index_expr.as_str()),
            ("smem_load", 8, entry_pc, index),
            "{symbol}"
        );
        // Both halves of the signed pair, in order.
        assert_eq!(e.guard_pcs, guard_pcs, "{symbol}");

        // Every read the unguarded analysis refused as memory-derived is now a
        // plain load through that one entry; nothing writes.
        assert_eq!(t.accesses.len(), reads, "{symbol}");
        assert_eq!(k.unguarded_unresolved.len(), reads, "{symbol}");
        assert!(
            k.unguarded_unresolved
                .iter()
                .all(|u| u.rule == "base-memory-derived"),
            "{symbol}"
        );
        assert!(
            t.accesses
                .iter()
                .all(|a| a.class == "vmem_load" && a.entry_pcs == [e.pc]),
            "{symbol}"
        );
        let table_arg = k.args.iter().find(|a| a.offset == 0).expect("arg 0");
        assert_eq!(table_arg.mode.as_deref(), Some("read"), "{symbol}");
    }
    assert!(r.obligations.iter().any(|o| o == "SealedTableWitness"));
}

#[test]
fn gate_up_o4_kernels_are_conditional_sealed_tables_bounded_by_n_experts() {
    check(GATE_UP, GATE_UP_CASES);
}

#[test]
fn down_dni_kernels_are_conditional_sealed_tables_bounded_by_n_experts() {
    check(DOWN, DOWN_CASES);
}
