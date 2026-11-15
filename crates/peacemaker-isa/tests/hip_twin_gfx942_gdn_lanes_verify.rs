// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen owner cefeacaffa: exact three-entry GDN Core lanes twin and MI300X receipt.
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] =
    include_bytes!("../../../kernels/qwen4_g942_gdn_lanes_verify_hipcc_gfx942.co");
const TWIN: &[u8] = include_bytes!("../../../kernels/qwen4_g942_gdn_lanes_verify_pm_gfx942.hxaco");
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/qwen4_g942_gdn_lanes_verify.gfx942.hip");
const MANIFEST: &str =
    include_str!("../../../kernels/qwen4_g942_gdn_lanes_verify_pm_gfx942.twin.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_gdn_lanes_verify_oracle.txt");
const SYMBOLS: [&str; 3] = [
    "qwen4_g942_gdn_lanes_conv_params_k4",
    "qwen4_g942_gdn_lanes_step_f32",
    "qwen4_g942_gdn_lanes_gate_batched",
];
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
        "f3f43131e579f55daf57adecf9ca3aa2b14b97a14984d10312e5489e6a760ab5"
    );
    assert_eq!(
        sha(ANCHOR),
        "12792ae10d021ac8597cf771896b7eed0262618c9bb45cb26b38c8452102c0ab"
    );
    assert_eq!(sha(TWIN), m["hxaco"]["sha256"]);
    assert_eq!(sha(SOURCE), m["source"]["sha256"]);
    assert_eq!(sha(ANCHOR), m["anchor"]["sha256"]);
    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect();
    let sealed = seal_many(ANCHOR, "gfx942".parse().unwrap(), &specs).unwrap();
    for entry in sealed {
        assert_eq!(entry.hxaco, TWIN);
    }
}

#[test]
fn every_entry_reverifies_exact_reports_and_strict_ledger() {
    let m = manifest();
    let reports = verify_hxaco_many(TWIN, "gfx942".parse().unwrap(), &SYMBOLS).unwrap();
    assert_eq!(reports.len(), 3);
    for (i, report) in reports.iter().enumerate() {
        assert_eq!(
            pinned_report(&report.report),
            pinned_report(&m["seal"]["reports"][i])
        );
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
        assert_eq!(
            report.report["elf_sha256"],
            "18d6455ab98b6292f74364ebfdea70d93986f9506173557aaf96b00e3a8344f5"
        );
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
    assert_eq!(r["status"], "PASS");
    assert_eq!(r["input_hxaco_sha256"], sha(TWIN));
    assert_eq!(r["hipcc_image_sha256"], sha(ANCHOR));
    assert_eq!(r["pm_device_elf_sha256"], m["anchor"]["device_elf_sha256"]);
    assert_eq!(r["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(r["cases"], 7);
    assert_eq!(r["steps"], 3);
    assert_eq!(r["controls"], 23);
    assert_eq!(r["ledger"]["exit"], 0);
    assert_eq!(LOG.matches("lanes route vs per-lane chains:").count(), 21);
    assert_eq!(LOG.matches("control:").count(), 23);
    assert!(!LOG.contains("=> FAIL"));
    assert!(LOG.contains("gdn_lanes_verify_f32_gfx942_oracle: PASS"));
    assert!(LOG.contains("candidate route: HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE="));
}
