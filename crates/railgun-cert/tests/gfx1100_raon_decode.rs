// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certification of the exact gfx1100 (7900 XTX) Raon plain-AR decode objects that `railgun-cert coverage` first
//! rejected or left conditional: the fixtures are the code objects (and radiowave sidecars) the `raon_oracle
//! railgun-probe --phase inventory` run loaded on the XTX, byte for byte (the object SHA-256 is pinned below).
//!
//! * Four objects (seven kernels) stopped the lifter with `undefined Gfx1100 opcode/form` (`v_cmp_ge_i64_e64`,
//!   `v_ashrrev_i64`, `v_cmpx_gt_u64_e32`); with the table rows they lift and every kernel is Proven with a passing
//!   sidecar differential.
//! * The two MoE pointer-table kernels and `ffn_situ_rows_f32` are Conditional with exactly the `sealed_table`
//!   witness on the gfx11 encodings (`s_load_b64` entry, `flat_load_b32`/`buffer_load_b32` reads), the same
//!   certificates the gfx1201 objects have, and nothing else: no guards, no indexed effects, nothing unresolved.
use railgun_cert::{
    certify, KernelRecord, Receipt, SEALED_BOUND_KERNARG, STATUS_CONDITIONAL, STATUS_PROVEN,
    WITNESS_SEALED_TABLE,
};

macro_rules! fixture {
    ($name:literal) => {
        (
            &include_bytes!(concat!("fixtures/", $name, ".gfx1100.hsaco"))[..],
            include_str!(concat!("fixtures/", $name, ".gfx1100.radiowave.json")),
        )
    };
}

fn kernel<'r>(r: &'r Receipt, symbol: &str) -> &'r KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .unwrap_or_else(|| panic!("{symbol} missing"))
}

fn admits(r: &Receipt, k: &KernelRecord) -> bool {
    k.validate_case(&r.object_sha256, &r.device_elf_sha256, &r.arch)
        .is_ok()
}

/// (object, sidecar, loaded-object SHA-256, kernels the Raon decode launches from it)
#[allow(clippy::type_complexity)]
fn lifted_objects() -> [(
    (&'static [u8], &'static str),
    &'static str,
    &'static [&'static str],
); 3] {
    [
        (
            fixture!("raon_latent_attention_decompress"),
            "07e7b06adabc1c44d214bbb16b684ad8dbdea2456c42995d989a6845f1b1de49",
            &[
                "latent_attention_gather_latent_f32",
                "latent_attention_scatter_expanded_q8",
            ],
        ),
        (
            fixture!("raon_ffn_moe_combine_policy"),
            "106d451be5e67c26d0e20bbb3262f07a916faa9a1df38fc6ba01abd11ea9415c",
            &["ffn_moe_combine_f32", "ffn_round_bf16_f32"],
        ),
        (
            fixture!("raon_latent_attention_decode"),
            "69f7e3b32362903ff5c2c80d5499f5d35ce52d89518d933e9da4ddbd8d6f3694",
            &[
                "latent_attention_decode_partial_q8",
                "latent_attention_decode_merge",
            ],
        ),
    ]
}

#[test]
fn the_objects_that_hit_decode_gaps_lift_and_are_proven() {
    for ((object, sidecar), sha, symbols) in lifted_objects() {
        let r = certify(object, Some(sidecar)).unwrap_or_else(|e| panic!("{sha}: {e}"));
        assert_eq!(r.arch, "gfx1100");
        assert_eq!(r.object_sha256, sha, "the fixture is the loaded object");
        for &symbol in symbols {
            let k = kernel(&r, symbol);
            assert_eq!(k.status, STATUS_PROVEN, "{symbol}: {:?}", k.unresolved);
            assert!(k.unresolved.is_empty(), "{symbol}: {:?}", k.unresolved);
            assert_eq!(k.differential.source, "sidecar", "{symbol}");
            assert_eq!(k.differential.status, "pass", "{symbol}");
            assert!(k.required_witnesses.is_empty() && k.sealed_tables.is_empty());
            assert!(admits(&r, k), "{symbol}");
        }
    }
}

#[test]
fn the_moe_pointer_table_kernels_are_sealed_table_conditional_on_gfx11_encodings() {
    // (object, sidecar, object SHA-256, symbol, table arg offset, n_experts offset, derived reads, read path)
    let cases = [
        (
            fixture!("raon_moe_gate_up_k8_indexed_batched"),
            "d1ff25a2c1ea5d7fd991f9762f78d53fcd63749d5e46c16a8aeb8bc6dbebc6eb",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched",
            52u64,
            14usize,
            "flat",
        ),
        (
            fixture!("raon_moe_down_k8_indexed_batched_expanded"),
            "003852ebde67e98909d04ee7862911ce1b86edf37bc47aae0f782b22de7e382b",
            "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",
            44,
            24,
            "buffer",
        ),
    ];
    for ((object, sidecar), sha, symbol, bound, reads, path) in cases {
        let r = certify(object, Some(sidecar)).expect("certifies");
        assert_eq!(
            (r.arch.as_str(), r.object_sha256.as_str()),
            ("gfx1100", sha)
        );
        let k = kernel(&r, symbol);
        assert_eq!(k.status, STATUS_CONDITIONAL, "{symbol}: {:?}", k.unresolved);
        assert!(k.unresolved.is_empty(), "{symbol}");
        assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE], "{symbol}");
        assert!(
            k.guards.is_empty() && k.indexed_effects.is_empty(),
            "{symbol}"
        );
        assert_eq!(k.differential.status, "pass", "{symbol}");
        assert!(admits(&r, k), "{symbol}");
        let [t] = &k.sealed_tables[..] else {
            panic!("{symbol}: one table, got {:?}", k.sealed_tables)
        };
        assert_eq!(t.table_arg_offset, 0, "{symbol}");
        assert_eq!(
            (t.bound_kind.as_str(), t.bound_value, t.entry_bytes),
            (SEALED_BOUND_KERNARG, bound, 8),
            "{symbol}"
        );
        assert!(
            !t.null_guarded,
            "{symbol}: reads happen inside the expert loop"
        );
        let [e] = &t.entries[..] else {
            panic!("{symbol}")
        };
        assert_eq!(
            (e.class.as_str(), e.path.as_str(), e.width),
            ("smem_load", "smem", 8)
        );
        assert!(e.index_expr.starts_with("load@") && !e.guard_pcs.is_empty());
        assert_eq!(t.accesses.len(), reads, "{symbol}");
        assert!(
            t.accesses
                .iter()
                .all(|a| a.class == "vmem_load" && a.path == path && a.entry_pcs == [e.pc]),
            "{symbol}: every derived access is a plain {path} load through the entry"
        );
        assert_eq!(k.unguarded_unresolved.len(), reads, "{symbol}");
        assert!(k
            .unguarded_unresolved
            .iter()
            .all(|u| u.rule == "base-memory-derived"));
        // Forging the case (status, bound, dropped witness, dropped access) fails admission.
        for edit in [
            (|k: &mut KernelRecord| k.status = STATUS_PROVEN.into()) as fn(&mut KernelRecord),
            |k| k.sealed_tables[0].bound_value += 4,
            |k| k.required_witnesses.clear(),
            |k| {
                k.sealed_tables[0].accesses.pop();
            },
        ] {
            let mut m = k.clone();
            edit(&mut m);
            assert!(
                !admits(&r, &m),
                "{symbol}: a forged case must fail admission"
            );
        }
    }
}

/// `ffn_situ_rows_f32` (awq mode 2) on gfx11: the signed pair `e >= 0 && e < n_experts` (two compares into lane
/// masks, `or`, `exec & mask`, `vccnz`), the pointer-table entry `table[e]` and the null check before `scale[col]`
/// is read; the same table, bound, guards and null guard as the gfx1201 object.
#[test]
fn ffn_situ_is_conditional_with_a_signed_guard_and_a_null_guard_on_gfx11() {
    let (object, sidecar) = fixture!("raon_ffn_situ_rows_f32");
    let r = certify(object, Some(sidecar)).expect("certifies");
    assert_eq!(r.arch, "gfx1100");
    assert_eq!(
        r.object_sha256,
        "d7c374102d3a42396fceb2f8137746c216bbf2c01c1b29fead4817cf5f32a580"
    );
    let k = kernel(&r, "ffn_situ_rows_f32");
    assert_eq!(k.status, STATUS_CONDITIONAL, "{:?}", k.unresolved);
    assert!(k.unresolved.is_empty());
    assert_eq!(k.required_witnesses, [WITNESS_SEALED_TABLE]);
    assert!(k.guards.is_empty() && k.indexed_effects.is_empty());
    assert_eq!(k.differential.status, "pass");
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
    let arg = k.args.iter().find(|a| a.offset == 24).expect("awq_a");
    assert_eq!(arg.mode.as_deref(), Some("read"));
    for flip in [false, true] {
        let mut m = k.clone();
        m.sealed_tables[0].null_guarded = flip;
        assert_eq!(admits(&r, &m), flip, "null_guarded={flip}");
    }
}
