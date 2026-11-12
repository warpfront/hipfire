// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen owner bb2f24ad74: four-entry MQ2 xbatch twin, exact-image MI300X evidence.
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE: &[u8] =
    include_bytes!("../../../kernels/src/qwen4_g942_gemv_mq2g256v2_xbatch.gfx942.hip");
const ANCHOR: &[u8] =
    include_bytes!("../../../kernels/qwen4_g942_gemv_mq2g256v2_xbatch_gfx942_hipcc_gfx942.co");
const TWIN: &[u8] =
    include_bytes!("../../../kernels/qwen4_g942_gemv_mq2g256v2_xbatch_pm_gfx942.hxaco");
const MANIFEST: &str =
    include_str!("../../../kernels/qwen4_g942_gemv_mq2g256v2_xbatch_pm_gfx942.twin.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_mq2_xbatch_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_mq2_xbatch_oracle.txt");
const SYMBOLS: [&str; 4] = [
    "qwen4_g942_gemv_mq2g256v2_xbatch4_k2048",
    "qwen4_g942_gemv_mq2g256v2_xbatch4_k2560",
    "qwen4_g942_gemv_mq2g256v2_xbatch8_k2048",
    "qwen4_g942_gemv_mq2g256v2_xbatch8_k2560",
];
const ELF_SHA: &str = "7d76c7bfc75adb5592766301474996cf364938cb15b6142c758c8fab3da0a055";
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
        "2bd7f03ff9027f4ad38145e0b34b665ffa8eee3fd63000be9cf828267f78bec5"
    );
    assert_eq!(
        sha(ANCHOR),
        "ceef88237047ecc125aa95a9903b48f02e4c19f83e37366028459174f955090e"
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
    assert_eq!(oracle["oracle"], "qwen4_g942_mq2_xbatch");
    assert_eq!(oracle["image_hook"], r["device_image_path"]);
    assert_eq!(oracle["cases"], 864);
    assert_eq!(oracle["failing_cases"], 0);
    let rows = oracle["detail"].as_array().unwrap();
    assert_eq!(rows.len(), 864);
    let mut direct = 0;
    for row in rows {
        assert_eq!(row["wrapper_bad_bytes"], 0);
        assert_eq!(row["wrapper_guard_clobbered"], false);
        if row["b"].as_u64().unwrap() <= 8 {
            assert_eq!(row["kernel_bad_bytes"], 0);
            assert_eq!(row["kernel_guard_clobbered"], false);
            direct += 1;
        } else {
            assert!(row["kernel_bad_bytes"].is_null());
            assert!(row["kernel_guard_clobbered"].is_null());
        }
    }
    assert_eq!(direct, 432);
    assert!(LOG.contains("qwen4_g942_mq2_xbatch_oracle: 864 cases, 0 failing"));
}
