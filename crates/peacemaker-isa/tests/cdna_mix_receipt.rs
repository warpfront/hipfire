// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hash-bind the codec's alias promotion to the actual strict MI300X interpreter receipt.
use peacemaker_cdna::V_MAD_MIX_F32_AS_FMA_MIX_F32;
use serde_json::Value;
use sha2::{Digest, Sha256};

const RECEIPT: &str = include_str!("fixtures/cdna/gfx942_fma_mix_f32_receipt.json");
const ORACLE: &str = include_str!("fixtures/cdna/gfx942_fma_mix_f32_oracle.json");
const LOG: &str = include_str!("fixtures/cdna/gfx942_fma_mix_f32_oracle.txt");
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_sha(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

#[test]
fn mixed_fma_promotion_is_bound_to_strict_hardware_evidence() {
    let alias = &V_MAD_MIX_F32_AS_FMA_MIX_F32;
    assert_eq!(
        alias.oracle_receipt,
        "crates/peacemaker-isa/tests/fixtures/cdna/gfx942_fma_mix_f32_receipt.json"
    );
    assert_eq!(alias.oracle_sha256, sha(RECEIPT.as_bytes()));
    let receipt: Value = serde_json::from_str(RECEIPT).unwrap();
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["raw_oracle_sha256"], sha(ORACLE.as_bytes()));
    assert_eq!(receipt["raw_log_sha256"], sha(LOG.as_bytes()));
    assert_eq!(receipt["ledger"]["exit"], 0);
    assert!(receipt["ledger"]["cmd"]
        .as_str()
        .unwrap()
        .contains("run-mix"));
    assert_eq!(oracle["schema"], "cdna_mix_hw_oracle/1");
    assert_eq!(oracle["hardware_evidence"], true);
    assert_eq!(oracle["mode"], "hardware");
    assert_eq!(oracle["device"]["device_arch"], "gfx942");
    assert_eq!(oracle["complete"], true);
    assert_eq!(oracle["exit_ok"], true);
    assert_eq!(oracle["exit_status"], 0);
    assert_eq!(
        oracle["verdict"],
        "PASS_HARDWARE_EQUALS_INTERPRETER_STRICT_BITS"
    );
    assert!(oracle["errors"].as_array().unwrap().is_empty());
    assert!(oracle["failures"].as_array().unwrap().is_empty());
    assert_eq!(oracle["scope"]["variants_total"], 13312);
    assert_eq!(oracle["scope"]["operand_shapes"], 13);
    assert_eq!(oracle["scope"]["neg_hi"], 0);
    assert_eq!(oracle["inputs"]["patterns"], 16640);
    let prefix = &oracle["inputs"]["preserved_r1_prefix"];
    assert_eq!(prefix["patterns"], 16384);
    assert_eq!(
        prefix["in0_sha256"],
        "0369929c7b7cddf4dccbb91caac275a9ac02fd93fa83ee3cd8e45c769bab160c"
    );
    assert_eq!(
        prefix["in1_sha256"],
        "4713adc0d612e4f704417dffc714feb256b39459e98a0f7e967e2243c9af9a95"
    );
    assert_eq!(
        prefix["in2_sha256"],
        "37854516df0c0449165ca23137a4d063a0fda2e5c4f7084edb7314ed58e538ff"
    );
    assert_eq!(oracle["inputs"]["modes"], 32);
    assert_eq!(oracle["inputs"]["fma_discriminator_rows_unresolved"], 0);
    assert_eq!(oracle["summary"]["modes_run"], 32);
    assert_eq!(oracle["summary"]["rows_compared"], 32 * 13312 * 16640u64);
    assert_eq!(oracle["summary"]["strict_mismatches"], 0);
    assert_eq!(oracle["summary"]["controls"], 8);
    assert_eq!(oracle["summary"]["controls_detected"], 8);
    for field in [
        "in0_sha256",
        "in1_sha256",
        "in2_sha256",
        "in0_in1_in2_sha256",
    ] {
        assert!(is_sha(&oracle["inputs"][field]));
    }
    for (object, field) in [
        ("hip_source", "sha256"),
        ("code_object", "sha256"),
        ("implementation", "oracle_source_sha256"),
        ("implementation", "interpreter_mfma_source_sha256"),
    ] {
        assert!(is_sha(&oracle[object][field]));
    }
    let modes = oracle["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 32);
    for mode in modes {
        assert_eq!(mode["variants"], 13312);
        assert_eq!(mode["inputs_per_variant"], 16640);
        assert_eq!(mode["compare"]["rows_compared"], 13312 * 16640u64);
        assert_eq!(mode["compare"]["mismatches"], 0);
        assert_eq!(mode["hw_output_sha256"], mode["sim_output_sha256"]);
        assert!(is_sha(&mode["hw_output_sha256"]));
        for field in [
            "mode_readbacks_not_equal_requested",
            "poison_dependent_words",
            "guard_canaries_corrupted_launches",
        ] {
            assert_eq!(mode["hw"][field], 0);
        }
        assert!(mode["hw"]["mode_readbacks_checked"].as_u64().unwrap() > 0);
        assert_eq!(mode["hw"]["launches"], 1664);
    }
    let controls = oracle["controls"].as_array().unwrap();
    assert_eq!(controls.len(), 8);
    for control in controls {
        assert_eq!(control["reference"], "hardware");
        assert_eq!(control["detected"], true);
        assert!(control["mismatches"].as_u64().unwrap() > 0);
    }
    let corruption = controls
        .iter()
        .find(|c| c["control"] == "ref_word_flip_bit0")
        .unwrap();
    assert_eq!(corruption["mismatches"], 1);
    let fused = &oracle["fused_discrimination"];
    assert!(
        fused["rows_where_unfused_differs_from_interpreter"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(fused["hw_equals_interpreter"], fused["rows"]);
    assert_eq!(fused["hw_equals_unfused_on_discriminating_rows"], 0);
    assert!(LOG.contains("SUMMARY: 32 modes, 0 errors, 0 failures"));
}
