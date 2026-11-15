// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The owner's replacement GDN verification hipcc anchor (owner commit `1b0d830afa`, rollback
//! wrapper on the global address space), byte-pinned, must certify **all four** entries.
//!
//! History (do not lose it): the original anchor, kept untouched as the negative fixture
//! `gfx942_gdn_verify_open.{co,hip}` (`cdna_gdn_verify_open.rs`), was refused for its rollback
//! entry (FLAT segment, instruction 60) and for the convolution entry,
//! `wait ledger replay: pc 674 global_load_dword v1, v[6:7], off (after label L648): needs Vm
//! wait 0 (Waw { reg: "v1" })`. The replacement fixes the rollback entry in the owner source; the
//! convolution code is byte-identical and is discharged by the verifier: the witness crossed
//! `s_branch L648` (pc 665) after a pending `global_load_ushort v1`, whose predecessor stores 0
//! in `s[14:15]`, so the join's `s_and_b64 s[14:15], s[14:15], exec; s_cselect_b32 s14, 1, 0;
//! s_cmp_lg_u32 s14, 1; s_cbranch_scc1` (pc 669..=672) takes its branch over the load for every
//! EXEC (`pm/LEDGER-PATH-PROOF.md` section 2, extended from a block to one CFG edge by
//! `cdna::lifted::entry_proofs`; unit test
//! `gdn_conv_params_pc674_is_not_reachable_over_the_s_branch_into_l648`). No wait cap, obligation
//! allowlist, suppression, anchor instruction or scheduler is changed.
//!
//! This is an acceptance test of the reviewed verifier change; a different refusal for any of
//! the four symbols is a failure to be investigated, never to be suppressed here.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use sha2::{Digest, Sha256};

const ANCHOR: &[u8] = include_bytes!("fixtures/cdna/gfx942_gdn_verify_global.co");
const SOURCE: &[u8] = include_bytes!("fixtures/cdna/gfx942_gdn_verify_global.hip");
const SYMBOLS: [&str; 4] = [
    "qwen4_g942_gdn_verify_capture_f32",
    "qwen4_g942_gdn_verify_rollback_f32",
    "qwen4_g942_gdn_verify_conv_params_k4",
    "qwen4_g942_gdn_verify_gate_batched",
];
// Owner commit 1b0d830afa (module source 976f1d71a4).
const SOURCE_SHA256: &str = "cc4aff6a4571b5f70ea7c0630b513b0ea99ea6d5ce8352f6601295ea368dc76e";
const BUNDLE_SHA256: &str = "98b7f4176fb329bb0ec5b42a4e55434e46a24a0350967a17a85abe60a843b80c";
const DEVICE_ELF_SHA256: &str = "42b8525e4d92ec90349d04e8226d8bbcaec4aa9f9538528ca176a6aa1fe5162f";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn specs() -> Vec<KernelSpec<'static>> {
    SYMBOLS
        .iter()
        .map(|symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect()
}

#[test]
fn replacement_anchor_is_byte_exact_and_distinct_from_the_original_negative_fixture() {
    assert_eq!(sha(SOURCE), SOURCE_SHA256);
    assert_eq!(sha(ANCHOR), BUNDLE_SHA256);
    // The original negative fixture is a different, untouched pair.
    let original = include_bytes!("fixtures/cdna/gfx942_gdn_verify_open.co");
    assert_eq!(
        sha(original),
        "a9c68299ce5bf448d5172a5dba6f46d101ffb86b620acd4772c73fae8272f355"
    );
    assert_ne!(sha(ANCHOR), sha(original));
    let (bundle, payloads) = Bundle::read(ANCHOR).expect("pinned offload bundle");
    let entries: Vec<_> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
        .collect();
    assert_eq!(entries.len(), 1);
    let elf = payloads[entries[0].0];
    assert_eq!(sha(elf), DEVICE_ELF_SHA256);
    let lifted = peacemaker_lift::cdna::lift_object(elf).expect("fully typed GDN image");
    assert_eq!(peacemaker_lift::cdna::checked(&lifted).unwrap(), elf);
    let mut names: Vec<_> = lifted
        .kernels
        .iter()
        .map(|k| k.image.name.as_str())
        .collect();
    names.sort_unstable();
    let mut expected = SYMBOLS;
    expected.sort_unstable();
    assert_eq!(names, expected);
}

#[test]
fn replacement_anchor_certifies_all_four_entries() {
    let sealed = seal_many(ANCHOR, Arch::Gfx942, &specs()).unwrap_or_else(|e| {
        panic!("the pinned replacement GDN anchor must certify all four entries, got: {e}")
    });
    assert_eq!(sealed.len(), SYMBOLS.len());
    for (symbol, s) in SYMBOLS.iter().zip(&sealed) {
        let h = &s.report["hazards"];
        assert_eq!(s.report["kernel"], *symbol, "{symbol}");
        assert_eq!(h["checked"], true, "{symbol}: {h}");
        assert_eq!(h["blocking"], 0, "{symbol}: {h}");
        assert_eq!(h["findings"], 0, "{symbol}: {h}");
        assert_eq!(h["bypasses"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["violations"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["inserted_nops"], 0, "{symbol}: {h}");
        assert_eq!(h["hazard"]["rewritten_instructions"], 0, "{symbol}: {h}");
        // The device image is unchanged by certification.
        assert_eq!(s.elf_sha256, DEVICE_ELF_SHA256, "{symbol}");
    }
    // One shared bundle; every symbol re-verifies the exact same bytes.
    let hxaco = &sealed[0].hxaco;
    assert!(sealed.iter().all(|s| &s.hxaco == hxaco));
    let verified = verify_hxaco_many(hxaco, Arch::Gfx942, &SYMBOLS)
        .unwrap_or_else(|e| panic!("sealed GDN bundle must re-verify all four entries: {e}"));
    assert_eq!(verified.len(), SYMBOLS.len());
    for (a, b) in sealed.iter().zip(&verified) {
        assert_eq!(a.hxaco_sha256, b.hxaco_sha256);
        assert_eq!(a.elf_sha256, b.elf_sha256);
        assert_eq!(a.text_sha256, b.text_sha256);
        assert_eq!(b.report["hazards"]["blocking"], 0);
        assert_eq!(b.report["hazards"]["findings"], 0);
        assert_eq!(b.report["hazards"]["hazard"]["rewritten_instructions"], 0);
    }
}

#[test]
fn committed_twin_reproduces_and_binds_the_exact_hardware_receipt() {
    let hxaco = include_bytes!("../../../kernels/qwen4_g942_gdn_verify_pm_gfx942.hxaco");
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../kernels/qwen4_g942_gdn_verify_pm_gfx942.twin.json"
    ))
    .unwrap();
    let sha_file = include_str!("../../../kernels/qwen4_g942_gdn_verify_pm_gfx942.hxaco.sha256");
    let log = include_str!("fixtures/cdna/gfx942_gdn_verify_oracle.txt");
    let sealed = seal_many(ANCHOR, Arch::Gfx942, &specs()).unwrap();
    assert_eq!(sealed[0].hxaco.as_slice(), hxaco.as_slice());
    assert_eq!(sha_file.split_whitespace().next().unwrap(), sha(hxaco));
    assert_eq!(manifest["anchor"]["sha256"], BUNDLE_SHA256);
    assert_eq!(manifest["source"]["sha256"], SOURCE_SHA256);
    assert_eq!(manifest["hxaco"]["sha256"], sha(hxaco));
    assert_eq!(
        manifest["seal"]["reports"].as_array().unwrap().len(),
        SYMBOLS.len()
    );
    for (report, symbol) in manifest["seal"]["reports"]
        .as_array()
        .unwrap()
        .iter()
        .zip(SYMBOLS)
    {
        assert_eq!(report["kernel"], symbol);
        assert_eq!(report["elf_sha256"], DEVICE_ELF_SHA256);
        assert_eq!(report["hazards"]["checked"], true);
        for field in ["findings", "blocking", "bypasses"] {
            assert_eq!(report["hazards"][field], 0);
        }
    }
    let receipt = &manifest["gpu_oracle"];
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], "gfx942");
    assert_eq!(receipt["input_hxaco_sha256"], sha(hxaco));
    assert_eq!(receipt["hipcc_image_sha256"], BUNDLE_SHA256);
    assert_eq!(receipt["pm_device_elf_sha256"], DEVICE_ELF_SHA256);
    assert_eq!(receipt["hipcc_device_elf_sha256"], DEVICE_ELF_SHA256);
    assert_eq!(receipt["raw_log_sha256"], sha(log.as_bytes()));
    assert_eq!(
        receipt["oracle_binary_sha256"],
        "2ecc14cfb1dd105cdac8e6b77b298b764c7d14e5472862f7c3316b2330c4f24c"
    );
    assert_eq!(receipt["case_count"], 280);
    assert_eq!(
        log.lines().filter(|line| line.contains("=> ok")).count(),
        280
    );
    assert_eq!(
        log.lines()
            .filter(|line| line.starts_with("[conv_params"))
            .count(),
        84
    );
    assert_eq!(
        log.lines()
            .filter(|line| line.starts_with("[gate_batched"))
            .count(),
        12
    );
    assert!(log.contains("gdn_capture_f32_gfx942_oracle: PASS"));
    assert!(log.contains(
        "GDN_VERIFY_IMAGE=/scratch/quant/fn-fp8/beat/WC8/twin/gdn/gdn-verify-twin-device.elf"
    ));
    assert!(log.contains("GDN_LANES_CAPTURE_IMAGE=/scratch/quant/fn-fp8/beat/WC8/twin/gdn/gdn-capture-twin-device.elf"));
    assert_eq!(receipt["failed_cases"], 0);
    assert_eq!(receipt["exit_code"], 0);
    assert_eq!(receipt["ledger"]["row"], 248);
    assert_eq!(receipt["ledger"]["exit"], 0);
}
