// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Real MI300X writeback ordering evidence; no GPU is opened by this fixture test.
use serde_json::Value;
use sha2::{Digest, Sha256};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn ordered_vmem_waw_receipt_binds_exact_sources_binary_and_guarded_results() {
    let raw = include_bytes!("fixtures/cdna/gfx942_vmem_waw_oracle.json");
    let log = include_bytes!("fixtures/cdna/gfx942_vmem_waw_oracle.txt");
    let receipt: Value =
        serde_json::from_slice(include_bytes!("fixtures/cdna/gfx942_vmem_waw_receipt.json"))
            .unwrap();
    let oracle: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["raw_receipt_sha256"], sha(raw));
    assert_eq!(receipt["raw_log_sha256"], sha(log));
    assert_eq!(
        receipt["llvm_revision"],
        "d6f6cb691863e9258034ab48c1dfce21689d5606"
    );
    assert_eq!(receipt["ledger"]["owner"], "PmLedgerWaw942.order");
    assert_eq!(receipt["ledger"]["class"], "oracle");
    assert_eq!(receipt["ledger"]["exit"], 0);
    assert_eq!(
        receipt["ledger"]["cmd"],
        "bash /scratch/quant/fn-fp8/beat/WC8/waw942/run-waw.sh"
    );
    let log = std::str::from_utf8(log).unwrap();
    assert!(log.contains(receipt["binary_sha256"].as_str().unwrap()));
    assert!(log.contains(receipt["source_commit"].as_str().unwrap()));
    let logged: Value =
        serde_json::from_str(log.lines().find(|l| l.starts_with('{')).unwrap()).unwrap();
    let mut summary = oracle.clone();
    summary.as_object_mut().unwrap().remove("rows");
    assert_eq!(logged, summary);
    assert_eq!(oracle["status"], "PASS");
    assert_eq!(oracle["arch"], "gfx942");
    assert_eq!(oracle["oracle"], "cdna_vmem_waw_hw_oracle");
    assert_eq!(
        oracle["device_elf_sha256"],
        "0d6562946e52e6d1e7d923036181768c4d12087ab85eba9c3f6c712cfaa72200"
    );
    for (field, source) in [
        (
            "oracle_source_sha256",
            include_bytes!("../examples/cdna_vmem_waw_hw_oracle.rs").as_slice(),
        ),
        (
            "lift_source_sha256",
            include_bytes!("../../peacemaker-lift/src/cdna.rs").as_slice(),
        ),
        (
            "bundle_codec_source_sha256",
            include_bytes!("../../peacemaker-lift/src/bundle.rs").as_slice(),
        ),
        (
            "flat_codec_source_sha256",
            include_bytes!("../../peacemaker-cdna/src/enc/flat.rs").as_slice(),
        ),
        (
            "mubuf_codec_source_sha256",
            include_bytes!("../../peacemaker-cdna/src/enc/mubuf.rs").as_slice(),
        ),
    ] {
        assert_eq!(oracle[field], sha(source), "{field}");
    }
    assert_eq!(oracle["kernels"], 519);
    assert_eq!(oracle["launches"], 1042);
    assert_eq!(oracle["blocks_per_launch"], 512);
    assert_eq!(oracle["scenarios"], 256);
    assert_eq!(oracle["queue_distances"], serde_json::json!([0, 8, 32]));
    assert_eq!(oracle["failing_rows"], 0);
    assert_eq!(oracle["controls_detected"], 3552);
    assert_eq!(oracle["controls_detected"], oracle["controls_expected"]);
    assert_eq!(oracle["latency_calibration_ok"], true);
    let cal = &oracle["latency_calibration"];
    assert_eq!(cal["ok"], true);
    assert!(cal["cold_over_hit"].as_f64().unwrap() >= 1.25);
    assert!(cal["bypass_over_hit"].as_f64().unwrap() >= 1.25);
    for row in cal["rows"].as_array().unwrap() {
        assert_eq!(row["trials"], 256);
        assert!(row["median"].as_u64().unwrap() > 0);
    }
    assert_eq!(oracle["cold_region_bytes"], 248 * 1024 * 1024);
    assert_eq!(oracle["cold_random_window_bytes"], 64 * 1024 * 1024);
    assert_eq!(oracle["race_cold_older_primed_newer_lane_pairs"], 1_478_754);
    assert!(oracle["cold_lines_max_per_launch"].as_u64().unwrap() > 2000);
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 519);
    let qualified: Vec<_> = rows.iter().filter(|r| r["gating"] == true).collect();
    assert_eq!(qualified.len(), 447);
    assert_eq!(qualified.len() * 2 * 512 * 64, 29_294_592);
    let mut controls = 0;
    for row in &qualified {
        assert_eq!(row["row_ok"], true, "{}", row["kernel"]);
        assert!(!row["class"].as_str().unwrap().contains("flat"));
        let passes = row["passes"].as_array().unwrap();
        assert_eq!(passes.len(), 2);
        for pass in passes {
            assert_eq!(pass["guards_ok"], true);
            assert_eq!(pass["bad_bytes"], 0);
            assert_eq!(pass["output_sha256"], pass["expected_sha256"]);
            assert_eq!(pass["controls_detected"], pass["controls_expected"]);
            controls += pass["controls_detected"].as_u64().unwrap();
            if row["overlap"] == true {
                assert!(pass["discriminating_lanes"].as_u64().unwrap() > 0);
                assert_ne!(pass["expected_sha256"], pass["reversed_order_model_sha256"]);
            }
        }
        if row["queue_distance"] == 0 {
            let window = row["pair_window"].as_array().unwrap();
            assert!(window
                .iter()
                .all(|i| !i.as_str().unwrap().starts_with("s_wait")));
        }
    }
    assert_eq!(controls, 3552);
    for class in [
        "global_global",
        "buffer_buffer",
        "global_buffer",
        "buffer_global",
    ] {
        assert!(qualified.iter().any(|r| r["class"] == class));
    }
    for policy in ["dd", "bd", "db", "nn"] {
        assert!(qualified.iter().any(|r| r["policy"] == policy));
    }
    assert!(qualified.iter().any(|r| r["older"] == "global_load_dword"
        && r["newer"] == "global_load_dwordx2"
        && r["overlap"] == true));
    // FLAT global-address-only observations do not promote FLAT, scratch, image or atomics.
    assert!(rows
        .iter()
        .filter(|r| r["class"].as_str().unwrap().contains("flat"))
        .all(|r| r["gating"] == false));
    assert_eq!(oracle["not_covered"].as_array().unwrap().len(), 4);
}
