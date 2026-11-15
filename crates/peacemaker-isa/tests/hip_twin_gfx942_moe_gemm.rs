// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen gfx942 moe_gemm Route-A twin and exact-image MI300X evidence.
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE: &[u8] = include_bytes!("../../../kernels/src/qwen4_g942_moe_gemm.gfx942.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_g942_moe_gemm_hipcc_gfx942.co");
const TWIN: &[u8] = include_bytes!("../../../kernels/qwen4_g942_moe_gemm_pm_gfx942.hxaco");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_g942_moe_gemm_pm_gfx942.twin.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_moe_gemm_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_moe_gemm_oracle.txt");
const SYMBOLS: [&str; 2] = ["qwen4_g942_moe_gemm_f32", "qwen4_g942_moe_gemm_xbf16"];
const ELF_SHA: &str = "116e454a721b9264a3331e5bb7e3a220517826c8c295b00827872b2c1a0dc1b2";
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
        "ecad36922a715ce9bd5803831bd7b55ecdf3a40fa5aebc68c7b4893337f73b37"
    );
    assert_eq!(
        sha(ANCHOR),
        "740fddb9e0e03ff9c7dc6fa782b20d44e06f5fe9758f4fee741e0d9c0c8598e4"
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

const SOURCE_ORACLE: &str = include_str!("fixtures/cdna/gfx942_moe_gemm_source_oracle.json");
const SOURCE_LOG: &str = include_str!("fixtures/cdna/gfx942_moe_gemm_source_oracle.txt");

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
        "37a43c6790fb717c0748f891497a463455e5e117deffacf98dfd155279406b2a"
    );
    assert_eq!(
        r["oracle_source_sha256"],
        sha(include_bytes!(
            "../../../crates/rdna-compute/examples/qwen4_g942_moe_gemm_oracle.rs"
        ))
    );
    let source = &r["source_oracle"];
    assert_eq!(source["raw_log_sha256"], sha(SOURCE_LOG.as_bytes()));
    assert_eq!(source["raw_receipt_sha256"], sha(SOURCE_ORACLE.as_bytes()));
    assert_eq!(source["ledger"]["exit"], 0);
    for (raw, log, twin) in [(ORACLE, LOG, true), (SOURCE_ORACLE, SOURCE_LOG, false)] {
        let oracle: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(oracle["oracle"], "qwen4_g942_moe_gemm_oracle");
        if twin {
            assert_eq!(oracle["image"], r["device_image_path"]);
        } else {
            assert!(oracle["image"].is_null());
        }
        assert_eq!(oracle["quick"], false);
        assert_eq!(oracle["cases_pass"], 276);
        assert_eq!(oracle["cases_fail"], 0);
        assert_eq!(oracle["controls_detected"], 18);
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
                assert_eq!(row["baseline_p1"], 0, "{row}");
                positives += 1;
            }
        }
        assert_eq!(controls, 18);
        assert_eq!(positives, 24);
        assert!(log.contains("RESULT cases PASS 276 FAIL 0 | controls DETECTED 18 MISSED 0"));
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
