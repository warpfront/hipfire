// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Registration of an externally emitted object by pinned SHA-256
//! (`railgun_corpus::PinnedReceipt`, `NodeFacts::from_pinned`): the receipt is
//! `railgun-cert certify`'s output for exactly the pinned bytes, verified by the
//! same `parse_receipt` a corpus lookup uses.

use railgun::kernel::{ArgRole, FactStatus, NodeFacts};
use railgun_corpus::{sha256_hex, PinnedReceipt};

const OBJECT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../railgun-cert/tests/fixtures/railgun_copy.gfx1201.hsaco"
);
const SYMBOL: &str = "railgun_copy";

fn object() -> Vec<u8> {
    std::fs::read(OBJECT).expect("fixture object")
}

fn receipt_json(bytes: &[u8]) -> Vec<u8> {
    let receipt = railgun_cert::certify(bytes, None).expect("certify");
    serde_json::to_vec_pretty(&receipt).unwrap()
}

#[test]
fn pinned_receipt_yields_the_receipt_facts() {
    let bytes = object();
    let sha = sha256_hex(&bytes);
    let json = receipt_json(&bytes);
    let pinned = PinnedReceipt::admit("copy", "gfx1201", &bytes, &sha, &json).unwrap();
    assert_eq!(pinned.receipt_sha256, sha256_hex(&json));
    let facts = NodeFacts::from_pinned(&pinned, SYMBOL).unwrap();
    assert_eq!(facts.kref.artifact_sha256, sha);
    assert_eq!(facts.kref.module, "copy");
    assert_eq!(facts.kref.receipt_sha256, pinned.receipt_sha256);
    assert_eq!(facts.facts.status, FactStatus::Proven);
    assert!(facts
        .facts
        .args
        .iter()
        .any(|a| matches!(a.role, ArgRole::Pointer { .. })));
}

#[test]
fn wrong_pin_is_refused() {
    let bytes = object();
    let json = receipt_json(&bytes);
    let other = sha256_hex(b"not the object");
    let err = PinnedReceipt::admit("copy", "gfx1201", &bytes, &other, &json).unwrap_err();
    assert!(err.contains("is not the pinned"), "{err}");
    let err = PinnedReceipt::admit("copy", "gfx1201", &bytes, "ABC", &json).unwrap_err();
    assert!(err.contains("not a lowercase SHA-256"), "{err}");
}

#[test]
fn altered_object_is_refused() {
    let bytes = object();
    let sha = sha256_hex(&bytes);
    let json = receipt_json(&bytes);
    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    let err = PinnedReceipt::admit("copy", "gfx1201", &tampered, &sha, &json).unwrap_err();
    assert!(err.contains("is not the pinned"), "{err}");
}

#[test]
fn receipt_of_another_object_is_refused() {
    let bytes = object();
    let other = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../railgun-cert/tests/fixtures/railgun_copy_ptrtable.gfx1201.hsaco"
    ))
    .unwrap();
    let sha = sha256_hex(&bytes);
    let err =
        PinnedReceipt::admit("copy", "gfx1201", &bytes, &sha, &receipt_json(&other)).unwrap_err();
    assert!(err.contains("receipt certifies"), "{err}");
    // ... and a receipt for the right bytes on another arch.
    let err =
        PinnedReceipt::admit("copy", "gfx1100", &bytes, &sha, &receipt_json(&bytes)).unwrap_err();
    assert!(err.contains("receipt certifies"), "{err}");
}

#[test]
fn receipt_schema_drift_is_refused() {
    let bytes = object();
    let sha = sha256_hex(&bytes);
    let mut receipt: serde_json::Value = serde_json::from_slice(&receipt_json(&bytes)).unwrap();
    receipt["railgun"]["section_version"] = serde_json::json!(0);
    let json = serde_json::to_vec(&receipt).unwrap();
    let err = PinnedReceipt::admit("copy", "gfx1201", &bytes, &sha, &json).unwrap_err();
    assert!(err.contains("section v0"), "{err}");
}

#[test]
fn symbol_without_a_record_is_refused() {
    let bytes = object();
    let sha = sha256_hex(&bytes);
    let pinned =
        PinnedReceipt::admit("copy", "gfx1201", &bytes, &sha, &receipt_json(&bytes)).unwrap();
    let err = NodeFacts::from_pinned(&pinned, "no_such_kernel").unwrap_err();
    assert!(err.contains("no railgun record"), "{err}");
}
