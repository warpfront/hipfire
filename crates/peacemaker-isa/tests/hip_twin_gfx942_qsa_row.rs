// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen owner dc33131250: two-entry parallel QSA row twin, exact-image MI300X evidence.
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE: &[u8] =
    include_bytes!("../../../kernels/src/qwen4_g942_qsa_row_attention.gfx942.hip");
const ANCHOR: &[u8] =
    include_bytes!("../../../kernels/qwen4_g942_qsa_row_attention_hipcc_gfx942.co");
const TWIN: &[u8] = include_bytes!("../../../kernels/qwen4_g942_qsa_row_attention_pm_gfx942.hxaco");
const MANIFEST: &str =
    include_str!("../../../kernels/qwen4_g942_qsa_row_attention_pm_gfx942.twin.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_qsa_row_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_qsa_row_oracle.txt");
const SYMBOLS: [&str; 2] = ["qwen4_g942_qsa_row_f32", "qwen4_g942_qsa_row_fp8_fnuz"];
const ELF_SHA: &str = "1caa4bd326be42faeab2f09fc43ff7a5b46a63e3a29cbe0f552bd44a88443b0a";
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
        "e95e554a7f939feaa2d9a80393edb9a4d1df0596149afb3b4cc54e9599a06b6e"
    );
    assert_eq!(
        sha(ANCHOR),
        "bbc9fbcfe6a2cd5ebc830fc82ffc9a6a360d7efe3c15c5824d93eda5513e4d9c"
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

#[test]
fn exact_image_hardware_receipt_is_hash_bound() {
    let m = manifest();
    let r = &m["gpu_oracle"];
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(r["status"], "PASS");
    assert_eq!(r["input_hxaco_sha256"], sha(TWIN));
    assert_eq!(r["hipcc_image_sha256"], sha(ANCHOR));
    assert_eq!(r["pm_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["hipcc_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(r["raw_receipt_sha256"], sha(ORACLE.as_bytes()));
    assert_eq!(r["ledger"]["exit"], 0);
    assert_eq!(oracle["oracle"], "qwen4_g942_qsa_row_oracle");
    assert_eq!(oracle["image"], r["device_image_path"]);
    assert_eq!(oracle["quick"], false);
    assert_eq!(oracle["cases_pass"], 1200);
    assert_eq!(oracle["cases_fail"], 0);
    assert_eq!(oracle["controls_detected"], 12);
    assert_eq!(oracle["controls_missed"], 0);
    let rows = oracle["rows"].as_array().unwrap();
    let mut controls = 0;
    let mut positive_samples = 0;
    for row in rows {
        if row.get("control").is_some() {
            assert_eq!(row["detected"], true);
            assert!(row["differing_bytes"].as_u64().unwrap() > 0);
            controls += 1;
        } else {
            assert_eq!(row["ok"], true);
            assert_eq!(row["module_vs_reference"], 0);
            assert_eq!(row["product_vs_reference"], 0);
            assert_eq!(row["reference_repeat"], 0);
            assert_eq!(row["guards_ok"], true);
            positive_samples += 1;
        }
    }
    assert_eq!(controls, 12);
    assert_eq!(positive_samples, 8);
    assert!(LOG.contains("module arm: code object "));
    assert!(LOG.contains("RESULT cases PASS 1200 FAIL 0 | controls DETECTED 12 MISSED 0"));
    assert!(LOG.ends_with("PASS\n"));
}
