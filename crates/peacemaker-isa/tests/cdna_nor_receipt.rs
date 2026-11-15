// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A real gfx942 integer-SALU receipt, never substituted by a CPU-only run.
use serde_json::Value;
use sha2::{Digest, Sha256};

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

#[test]
fn nor_b64_receipt_binds_sources_raw_hardware_output_and_guarded_alias_runs() {
    let raw = include_bytes!("fixtures/cdna/gfx942_nor_b64_oracle.json");
    let log = include_bytes!("fixtures/cdna/gfx942_nor_b64_oracle.txt");
    let receipt: Value =
        serde_json::from_slice(include_bytes!("fixtures/cdna/gfx942_nor_b64_receipt.json"))
            .unwrap();
    let oracle: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["raw_receipt_sha256"], sha(raw));
    assert_eq!(receipt["raw_log_sha256"], sha(log));
    assert_eq!(receipt["ledger"]["owner"], "TwinWide942d");
    assert_eq!(receipt["ledger"]["class"], "oracle");
    assert_eq!(receipt["ledger"]["exit"], 0);
    assert!(receipt["ledger"]["cmd"]
        .as_str()
        .unwrap()
        .ends_with("/run-nor.sh"));
    let log = std::str::from_utf8(log).unwrap();
    assert!(log.contains(receipt["binary_sha256"].as_str().unwrap()));
    let logged: Value =
        serde_json::from_str(log.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    assert_eq!(logged, oracle);
    assert_eq!(oracle["oracle"], "cdna_nor_hw_oracle");
    assert_eq!(oracle["status"], "PASS");
    assert_eq!(oracle["arch"], "gfx942");
    assert_eq!(oracle["cases"], 24576);
    assert_eq!(oracle["controls_detected"], 12);
    assert_eq!(
        oracle["oracle_source_sha256"],
        sha(include_bytes!("../examples/cdna_nor_hw_oracle.rs"))
    );
    assert_eq!(
        oracle["interpreter_source_sha256"],
        sha(include_bytes!("../../peacemaker-cdna/src/sim/salu.rs"))
    );
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 6);
    for name in ["nor_separate", "nor_alias0", "nor_alias1"] {
        for poison in [0x5a, 0xa5] {
            let row = rows
                .iter()
                .find(|r| r["kernel"] == name && r["poison"] == poison)
                .unwrap();
            assert_eq!(row["cases"], 4096);
            assert_eq!(row["guards_ok"], true);
            assert_eq!(row["bad_bytes"], 0);
            assert_eq!(row["hardware_sha256"], row["interpreter_sha256"]);
            assert_eq!(
                row["hardware_sha256"],
                "8dec15b5cec5b4919f83cf7475cf615e404e1c6d67880d5ae9f679cb991c8fd0"
            );
        }
    }
}
