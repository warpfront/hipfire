// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen gfx942 moe_wide Route-A twin and exact-image MI300X evidence.
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE: &[u8] = include_bytes!("../../../kernels/src/qwen4_g942_moe_wide.gfx942.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_g942_moe_wide_hipcc_gfx942.co");
const TWIN: &[u8] = include_bytes!("../../../kernels/qwen4_g942_moe_wide_pm_gfx942.hxaco");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_g942_moe_wide_pm_gfx942.twin.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_moe_wide_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_moe_wide_oracle.txt");
const SYMBOLS: [&str; 4] = [
    "qwen4_g942_moe_wide_router",
    "qwen4_g942_moe_wide_group_scatter_desc",
    "qwen4_g942_moe_wide_unscatter_act",
    "qwen4_g942_moe_wide_combine_shared_add",
];
const ELF_SHA: &str = "ffcd2dc8ac4c35eb5bb42a2b0e36259d12f142db897e62e9e8af06377662cf51";
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn manifest() -> Value {
    serde_json::from_str(MANIFEST).unwrap()
}

#[test]
fn frozen_inputs_and_artifacts_are_exact() {
    let m = manifest();
    assert_eq!(
        sha(SOURCE),
        "7806d4fcdd2aef9d74ad99301d8ae1199c0a71b58ef4e1f8afdbe82746154e2c"
    );
    assert_eq!(
        sha(ANCHOR),
        "892b48a01c9a3f3a59a756609c9bde02e9a1de1eb8e4f4fb9e3f7dad8d2bc6a5"
    );
    assert_eq!(sha(SOURCE), m["source"]["sha256"]);
    assert_eq!(sha(ANCHOR), m["anchor"]["sha256"]);
    assert_eq!(sha(TWIN), m["hxaco"]["sha256"]);
    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect();
    let sealed = seal_many(ANCHOR, "gfx942".parse().unwrap(), &specs).unwrap();
    assert_eq!(sealed.len(), SYMBOLS.len());
    for entry in sealed {
        assert_eq!(entry.hxaco, TWIN);
    }
}

#[test]
fn every_entry_reverifies_exact_reports_and_strict_ledger() {
    let m = manifest();
    let reports = verify_hxaco_many(TWIN, "gfx942".parse().unwrap(), &SYMBOLS).unwrap();
    assert_eq!(reports.len(), SYMBOLS.len());
    for (i, report) in reports.iter().enumerate() {
        assert_eq!(
            pinned_report(&report.report),
            pinned_report(&m["seal"]["reports"][i])
        );
        assert_eq!(report.report["elf_sha256"], ELF_SHA);
        let h = &report.report["hazards"];
        assert_eq!(h["checked"], true);
        for field in [
            "blocking",
            "findings",
            "bypasses",
            "unreachable_instructions",
        ] {
            assert_eq!(h[field], 0);
        }
        for field in ["inserted_nops", "rewritten_instructions", "violations"] {
            assert_eq!(h["hazard"][field], 0);
        }
    }
}

#[test]
fn malformed_images_and_missing_symbols_refuse() {
    let arch = "gfx942".parse().unwrap();
    assert!(verify_hxaco_many(&TWIN[..TWIN.len() / 2], arch, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(TWIN, arch, &["missing_entry"]).is_err());
    assert!(verify_hxaco_many(TWIN, "gfx1201".parse().unwrap(), &SYMBOLS).is_err());
}

const SOURCE_ORACLE: &str = include_str!("fixtures/cdna/gfx942_moe_wide_source_oracle.json");
const SOURCE_LOG: &str = include_str!("fixtures/cdna/gfx942_moe_wide_source_oracle.txt");

#[test]
fn exact_image_hardware_receipt_is_hash_bound() {
    let m = manifest();
    let r = &m["gpu_oracle"];
    assert_eq!(r["status"], "PASS");
    assert_eq!(r["input_hxaco_sha256"], sha(TWIN));
    assert_eq!(r["hipcc_image_sha256"], sha(ANCHOR));
    assert_eq!(r["pm_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["hipcc_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(r["raw_receipt_sha256"], sha(ORACLE.as_bytes()));
    assert_eq!(r["ledger"]["exit"], 0);
    assert_eq!(r["ledger"]["owner"], "TwinWide942f.twins");
    assert_eq!(
        r["binary_sha256"],
        "2b029d0dcf7e864df5620acb91cf7dc8d1f6aa99c7b8454fbaebeda88f23c317"
    );
    assert_eq!(
        r["oracle_source_sha256"],
        sha(include_bytes!(
            "../../../crates/rdna-compute/examples/qwen4_g942_moe_wide_oracle.rs"
        ))
    );
    let source = &r["source_oracle"];
    assert_eq!(source["raw_log_sha256"], sha(SOURCE_LOG.as_bytes()));
    assert_eq!(source["raw_receipt_sha256"], sha(SOURCE_ORACLE.as_bytes()));
    assert_eq!(source["ledger"]["exit"], 0);
    for (raw, log, twin) in [(ORACLE, LOG, true), (SOURCE_ORACLE, SOURCE_LOG, false)] {
        let oracle: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(oracle["oracle"], "qwen4_g942_moe_wide_oracle");
        if twin {
            assert_eq!(oracle["image"], r["device_image_path"]);
        } else {
            assert!(oracle["image"].is_null());
        }
        assert_eq!(oracle["quick"], false);
        assert_eq!(oracle["cases_pass"], 582);
        assert_eq!(oracle["cases_fail"], 0);
        assert_eq!(oracle["controls_detected"], 28);
        assert_eq!(oracle["controls_missed"], 0);
        let mut controls = 0;
        let mut positives = 0;
        for row in oracle["rows"].as_array().unwrap() {
            if row.get("control").is_some() {
                assert_eq!(row["detected"], true);
                assert!(row["differing_bytes"].as_u64().unwrap() > 0 || row["guards_ok"] == false);
                controls += 1;
            } else {
                assert_eq!(row["ok"], true);
                assert_eq!(row["guards_ok"], true);
                assert_eq!(row["module"], 0, "{row}");
                assert_eq!(row["product"], 0, "{row}");
                assert_eq!(row["reference_repeat"], 0, "{row}");
                assert_eq!(row["fault_module"], 0, "{row}");
                assert_eq!(row["fault_product"], 0, "{row}");
                assert_eq!(row["fault_reference"], 0, "{row}");
                assert_eq!(row["fault_reference_repeat"], 0, "{row}");
                positives += 1;
            }
        }
        assert_eq!(controls, 28);
        assert_eq!(positives, 24);
        assert!(log.contains("RESULT cases PASS 582 FAIL 0 | controls DETECTED 28 MISSED 0"));
        assert!(log.ends_with("PASS\n"));
    }
}

#[test]
fn shared_oracle_plumbing_is_source_bound() {
    assert_eq!(
        manifest()["gpu_oracle"]["oracle_common_sha256"],
        sha(include_bytes!(
            "../../rdna-compute/examples/g942_oracle_common/mod.rs"
        ))
    );
}
