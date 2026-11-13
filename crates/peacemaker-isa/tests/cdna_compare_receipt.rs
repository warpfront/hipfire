// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Source-bound gfx942 compare evidence: all predicates, both encodings, and CMPX EXEC.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const RECEIPT: &str = include_str!("fixtures/cdna/gfx942_float_compare_receipt.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_float_compare_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_float_compare_oracle.txt");
const PREDS: [&str; 16] = [
    "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt",
    "tru",
];
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_sha(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}
fn zero_mismatches(value: &Value) {
    for field in [
        "bad_records_hardware_vs_interpreter",
        "bad_records_hardware_vs_truth",
        "bad_records_interpreter_vs_truth",
    ] {
        assert_eq!(value[field], 0, "{field}: {value}");
    }
}

#[test]
fn compare_evidence_is_bound_to_the_exercised_model_and_binary() {
    let receipt: Value = serde_json::from_str(RECEIPT).unwrap();
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["raw_oracle_sha256"], sha(ORACLE.as_bytes()));
    assert_eq!(receipt["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(receipt["ledger"]["exit"], 0);
    assert!(receipt["ledger"]["cmd"]
        .as_str()
        .unwrap()
        .contains("run-compiler-oracles"));
    assert!(is_sha(&receipt["binary_sha256"]));
    for (field, bytes) in [
        (
            "oracle_source_sha256",
            &include_bytes!("../examples/cdna_compare_hw_oracle.rs")[..],
        ),
        (
            "interpreter_valu_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/sim/valu.rs")[..],
        ),
        (
            "interpreter_mod_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/sim/mod.rs")[..],
        ),
        (
            "codec_vop_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/enc/vop.rs")[..],
        ),
        (
            "codec_vop3_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/enc/vop3.rs")[..],
        ),
        (
            "codec_inst_source_sha256",
            &include_bytes!("../../peacemaker-cdna/src/inst.rs")[..],
        ),
    ] {
        assert_eq!(oracle[field], sha(bytes), "{field}");
    }
    assert_eq!(oracle["status"], "PASS");
    assert_eq!(oracle["oracle"], "cdna_compare_hw_oracle");
    assert_eq!(oracle["arch"], "gfx942");
    assert_eq!(oracle["complete"], true);
    assert!(oracle["error"].is_null());
    assert!(oracle["mismatch_samples"].as_array().unwrap().is_empty());
    zero_mismatches(&oracle);
    for field in [
        "hip_source_sha256",
        "code_object_sha256",
        "variant_encodings_sha256",
        "inputs_sha256",
        "hardware_outputs_sha256",
    ] {
        assert!(is_sha(&oracle[field]), "{field}");
    }
    assert_eq!(oracle["operations"], 65);
    assert_eq!(oracle["rows"], 769);
    assert_eq!(oracle["cases"], 184576);
    assert_eq!(oracle["records"], 7834368);
    assert_eq!(oracle["lane_compares"], 501399552u64);
    assert_eq!(oracle["launches"], 1538);
    assert_eq!(oracle["controls_detected"], 390);
    assert_eq!(oracle["variants_total"], 2723);
    assert_eq!(oracle["poison_bytes"], serde_json::json!(["0x5a", "0xa5"]));
    assert!(LOG.contains("PASS cdna_compare_hw_oracle: 65 operations, 769 rows, 184576 cases, 7834368 records, 501399552 lane compares, 390 controls detected"));
}

#[test]
fn every_predicate_encoding_mask_and_denorm_mode_has_hardware_rows() {
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    let ops = oracle["operation_table"].as_array().unwrap();
    assert_eq!(ops.len(), 65);
    let mut expected = BTreeSet::new();
    for width in [16, 32] {
        for cmpx in [false, true] {
            for (pred, token) in PREDS.iter().enumerate() {
                let name = format!("v_cmp{}_{}_f{width}", if cmpx { "x" } else { "" }, token);
                expected.insert(name.clone());
                let op = ops.iter().find(|o| o["op"] == name).unwrap();
                assert_eq!(op["predicate_nibble"], pred);
                assert_eq!(op["predicate_token"], *token);
                assert_eq!(op["cmpx"], cmpx);
                assert_eq!(
                    op["opcode"],
                    (if width == 16 { 32 } else { 64 }) + if cmpx { 16 } else { 0 } + pred
                );
                assert_eq!(op["kind"], format!("F{width}"));
                assert_eq!(op["modes"], 12);
                assert_eq!(op["cases"], if pred == 0 || pred == 15 { 128 } else { 256 });
                assert_eq!(
                    op["variants"].as_array().unwrap().len(),
                    if width == 16 { 46 } else { 39 }
                );
                assert_eq!(op["vop3_clamp0_variants"], 17);
                assert_eq!(op["vop3_clamp1_variants"], 17);
                assert_eq!(op["case_kinds"]["edge_product"], 16);
                assert_eq!(op["case_kinds"]["random"], 48);
                assert_eq!(op["case_kinds"]["exec_sweep"], 64);
                if pred != 0 && pred != 15 {
                    assert_eq!(op["case_kinds"]["lane_walk"], 128);
                }
            }
        }
    }
    expected.insert("v_cmp_ge_u64".to_owned());
    let actual: BTreeSet<_> = ops
        .iter()
        .map(|o| o["op"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(actual, expected);
    let ge = ops.iter().find(|o| o["op"] == "v_cmp_ge_u64").unwrap();
    assert_eq!(ge["opcode"], 238);
    assert_eq!(ge["modes"], 1);
    assert_eq!(ge["cases"], 256);
    assert_eq!(ge["variants"].as_array().unwrap().len(), 3);
    for op in ops {
        zero_mismatches(op);
        let variants = op["variants"].as_array().unwrap();
        let labels: BTreeSet<_> = variants
            .iter()
            .map(|v| v["label"].as_str().unwrap())
            .collect();
        assert_eq!(labels.len(), variants.len());
        for label in ["e32", "e64_vcc", "e64_s20"] {
            assert!(labels.contains(label));
        }
        for variant in variants {
            assert!(!variant["words"].as_array().unwrap().is_empty());
            assert!(variant["text"]
                .as_str()
                .unwrap()
                .starts_with(op["op"].as_str().unwrap()));
        }
        let controls = op["controls"].as_array().unwrap();
        assert_eq!(controls.len(), 6);
        for control in controls {
            assert_eq!(control["detected_vs_interpreter"], true);
            assert_eq!(control["detected_vs_truth"], true);
        }
    }
    let mut rows_per_op = BTreeMap::<&str, usize>::new();
    let mut denorms = BTreeMap::<&str, BTreeSet<u64>>::new();
    let mut modes = BTreeSet::new();
    let mut records = 0;
    let mut cases = 0;
    let rows = oracle["mode_rows"].as_array().unwrap();
    assert_eq!(rows.len(), 769);
    for row in rows {
        let name = row["op"].as_str().unwrap();
        let op = ops.iter().find(|o| o["op"] == name).unwrap();
        assert!(modes.insert((name, row["mode"]["bits"].as_str().unwrap())));
        *rows_per_op.entry(name).or_default() += 1;
        let field = if name.ends_with("f16") {
            "fp_denorm16_64"
        } else {
            "fp_denorm32"
        };
        denorms
            .entry(name)
            .or_default()
            .insert(row["mode"][field].as_u64().unwrap());
        assert_eq!(row["cases"], op["cases"]);
        assert_eq!(row["variants"], op["variants"].as_array().unwrap().len());
        let n = row["cases"].as_u64().unwrap();
        let nv = row["variants"].as_u64().unwrap();
        assert_eq!(row["records"], n * nv);
        records += n * nv;
        cases += n;
        zero_mismatches(row);
        assert_eq!(row["guards_ok"], true);
        assert_eq!(row["mode_readbacks_ok"], true);
        assert_eq!(row["poison_dependent_words"], 0);
        assert!(is_sha(&row["hardware_sha256"]));
        assert_eq!(row["hardware_sha256"], row["interpreter_sha256"]);
        assert_eq!(row["hardware_sha256"], row["truth_sha256"]);
    }
    assert_eq!(records, oracle["records"].as_u64().unwrap());
    assert_eq!(cases, oracle["cases"].as_u64().unwrap());
    for name in expected {
        assert_eq!(
            rows_per_op[&*name],
            if name == "v_cmp_ge_u64" { 1 } else { 12 }
        );
        if name != "v_cmp_ge_u64" {
            assert_eq!(denorms[&*name], BTreeSet::from([0, 1, 2, 3]));
        }
    }
}
