// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The FP8 verify attention tile/reduce explicit-ABI table of the DFlash
//! window design (§5) against the **actual** gfx1201 code object
//! `attention_verify_gqa_gfx1201` (`kernels/src/attention_verify_gqa.gfx1201.hip`),
//! through the certifier, and the Whole16 manifest rules that consume it.
//!
//! Fixture: the object the runtime JIT loads for that module on gfx1201.
use railgun_cert::whole16::{
    build_with, check_abi, verify_with, AbiEffect, AbiSpec, ClassSpec, SequenceEntry, FP8_ABI,
};
use railgun_cert::{certify, sha256_hex, KernelRecord, Receipt};
use railgun_corpus::window::NodeRoute;

const OBJECT: &[u8] = include_bytes!("fixtures/attention_verify_gqa_gfx1201.hsaco");
const PIN: &str = "0000000000000000000000000000000000000000000000000000000000000001";
const TILE: &str = "attention_verify_gqa_fp8_gfx1201";
const GQA8: &str = "attention_verify_gqa_fp8_gqa8_gfx1201";
const REDUCE: &str = "attention_verify_reduce_gfx1201";
const CLASSES: &[ClassSpec] = &[
    ClassSpec {
        id: "tile",
        any_of: &[TILE],
        exact: true,
    },
    ClassSpec {
        id: "reduce",
        any_of: &[REDUCE],
        exact: true,
    },
];

fn receipt() -> Receipt {
    certify(OBJECT, None).expect("certifies")
}

fn kernel<'r>(r: &'r Receipt, symbol: &str) -> &'r KernelRecord {
    r.railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .unwrap_or_else(|| panic!("no {symbol}"))
}

fn entry(symbol: &str) -> SequenceEntry {
    SequenceEntry {
        phase: "verify_body".into(),
        symbol: symbol.into(),
        route: NodeRoute::Railgun,
        object: Some(OBJECT.to_vec()),
    }
}

#[test]
fn explicit_abi_table_matches_the_object_metadata_and_facts() {
    let r = receipt();
    for spec in FP8_ABI {
        let k = kernel(&r, spec.symbol);
        assert_eq!(k.status, "proven", "{}: {:?}", spec.symbol, k.unresolved);
        let abi = check_abi(k, spec).unwrap_or_else(|e| panic!("{}: {e:?}", spec.symbol));
        // The explicit end comes from the table and the object agrees; the
        // loader segment is whatever the object declares (hidden block
        // included), never a constant of ours.
        assert_eq!(abi.explicit_end, spec.explicit_end(), "{}", spec.symbol);
        assert!(k.kernarg_size >= abi.explicit_end, "{}", spec.symbol);
        assert_eq!(abi.kernarg_size, k.kernarg_size);
    }
    assert_eq!(FP8_ABI[0].explicit_end(), 60, "tile: scale_attn f32 at 56");
    assert_eq!(FP8_ABI[2].explicit_end(), 36, "reduce: batch_offset at +32");
    // Hidden arguments start at or after the explicit end.
    for symbol in [TILE, REDUCE] {
        let k = kernel(&r, symbol);
        let end = FP8_ABI
            .iter()
            .find(|s| s.symbol == symbol)
            .unwrap()
            .explicit_end();
        for a in k
            .args
            .iter()
            .filter(|a| a.value_kind.starts_with("hidden_"))
        {
            assert!(
                a.offset >= end,
                "{symbol}: {} at {}",
                a.value_kind,
                a.offset
            );
        }
    }
}

#[test]
fn a_changed_table_refuses_against_the_real_object() {
    let r = receipt();
    let tile = kernel(&r, TILE);
    // q read, but claimed written.
    const WRONG_EFFECT: &[(u32, u32, AbiEffect)] = &[
        (0, 8, AbiEffect::Write),
        (8, 8, AbiEffect::Read),
        (16, 8, AbiEffect::Read),
        (24, 8, AbiEffect::Write),
        (32, 8, AbiEffect::Read),
        (40, 4, AbiEffect::Scalar),
        (44, 4, AbiEffect::Scalar),
        (48, 4, AbiEffect::Scalar),
        (52, 4, AbiEffect::Scalar),
        (56, 4, AbiEffect::Scalar),
    ];
    // Scalar claimed at 36 instead of 40: no such explicit argument.
    const SHIFTED: &[(u32, u32, AbiEffect)] = &[
        (0, 8, AbiEffect::Read),
        (8, 8, AbiEffect::Read),
        (16, 8, AbiEffect::Read),
        (24, 8, AbiEffect::Write),
        (32, 8, AbiEffect::Read),
        (36, 4, AbiEffect::Scalar),
        (44, 4, AbiEffect::Scalar),
        (48, 4, AbiEffect::Scalar),
        (52, 4, AbiEffect::Scalar),
        (56, 4, AbiEffect::Scalar),
    ];
    for args in [WRONG_EFFECT, SHIFTED] {
        let spec = AbiSpec {
            symbol: TILE,
            admitted: true,
            args,
        };
        assert!(check_abi(tile, &spec).is_err());
    }
}

#[test]
fn manifest_generates_the_gqa8_receipt_without_admitting_it() {
    let seq = vec![entry(TILE), entry(REDUCE)];
    let m = build_with(&seq, "gfx1201", PIN, CLASSES).expect("builds");
    let abi: Vec<(&str, bool)> = m
        .abi
        .iter()
        .map(|a| (a.symbol.as_str(), a.admitted))
        .collect();
    for want in [(TILE, true), (GQA8, false), (REDUCE, true)] {
        assert!(abi.contains(&want), "{abi:?}");
    }
    assert!(m
        .abi
        .iter()
        .all(|a| a.artifact_sha256 == sha256_hex(OBJECT)));
    verify_with(&m, &seq, "gfx1201", PIN, CLASSES).expect("verifies");

    // The GQA8 object is analysed; a plan naming it is still refused.
    let seq = vec![entry(TILE), entry(GQA8), entry(REDUCE)];
    let e = build_with(&seq, "gfx1201", PIN, CLASSES).expect_err("refuses");
    assert!(e
        .iter()
        .any(|r| r.symbol == GQA8 && r.reason.contains("not admitted")));

    // Missing reduce node: the required class is absent.
    let e = build_with(&[entry(TILE)], "gfx1201", PIN, CLASSES).expect_err("refuses");
    assert!(e.iter().any(|r| r.reason.contains("reduce")));
}
