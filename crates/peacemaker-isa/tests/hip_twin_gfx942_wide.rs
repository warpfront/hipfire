// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Committed P1b wide BF16 twin: all three entries, immutable owner anchor,
//! strict typed certification, exact kernarg ABI and an image-bound Tier C receipt.
//! Regenerate explicitly with `scripts/pm-gfx942-wide-twin.py`.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SYMBOLS: [&str; 3] = [
    "gemm_bf16_mfma_gfx942_wide64",
    "gemm_bf16_mfma_gfx942_wide64_m32",
    "gemm_bf16_mfma_gfx942_wide64_m64",
];
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/gemm_bf16_mfma_wide.gfx942.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/gemm_bf16_mfma_wide_gfx942_hipcc_gfx942.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/gemm_bf16_mfma_wide_gfx942_pm_gfx942.hxaco");
const SHA_FILE: &str =
    include_str!("../../../kernels/gemm_bf16_mfma_wide_gfx942_pm_gfx942.hxaco.sha256");
const MANIFEST: &str =
    include_str!("../../../kernels/gemm_bf16_mfma_wide_gfx942_pm_gfx942.twin.json");
const SOURCE_SHA: &str = "49f356151cf2fac0c31e6e965bf7cf7e5a9968e2c8970e90f691e4acac512567";
const ANCHOR_SHA: &str = "5ecf282d0a3129cc9cfc0d4268677bc1b67e11454931818ba808a33a30061bcc";
const ELF_SHA: &str = "141276895e3dd53cb51746e7fb8f761d2e29929dc36147c719d0fe8355dec1e6";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("manifest")
}

fn elf(bytes: &[u8]) -> &[u8] {
    let (bundle, payloads) = Bundle::read(bytes).expect("bundle");
    let entries: Vec<_> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.triple == "hipv4-amdgcn-amd-amdhsa--gfx942")
        .collect();
    assert_eq!(entries.len(), 1);
    payloads[entries[0].0]
}

#[test]
fn immutable_owner_inputs_and_recipe_are_pinned() {
    let m = manifest();
    assert_eq!(sha(SOURCE), SOURCE_SHA);
    assert_eq!(sha(ANCHOR), ANCHOR_SHA);
    assert_eq!(sha(elf(ANCHOR)), ELF_SHA);
    assert_eq!(sha(elf(HXACO)), ELF_SHA);
    assert_eq!(m["arch"], "gfx942");
    assert_eq!(m["module"], "gemm_bf16_mfma_wide_gfx942_pm_gfx942");
    assert_eq!(m["symbols"], serde_json::json!(SYMBOLS));
    assert_eq!(m["source"]["owner_commit"], "17183b9b89");
    assert_eq!(
        m["source"]["path"],
        "kernels/src/gemm_bf16_mfma_wide.gfx942.hip"
    );
    assert_eq!(m["source"]["sha256"], SOURCE_SHA);
    assert_eq!(m["source"]["bytes"], SOURCE.len());
    assert_eq!(
        m["anchor"]["file"],
        "gemm_bf16_mfma_wide_gfx942_hipcc_gfx942.co"
    );
    assert_eq!(m["anchor"]["sha256"], ANCHOR_SHA);
    assert_eq!(m["anchor"]["device_elf_sha256"], ELF_SHA);
    assert_eq!(m["anchor"]["bytes"], ANCHOR.len());
    assert_eq!(
        m["hxaco"]["file"],
        "gemm_bf16_mfma_wide_gfx942_pm_gfx942.hxaco"
    );
    assert_eq!(m["hxaco"]["sha256"], sha(HXACO));
    assert_eq!(m["hxaco"]["bytes"], HXACO.len());
    assert_eq!(SHA_FILE.trim(), sha(HXACO));
    assert_eq!(
        m["compiler"]["runtime_source_basename"],
        "gemm_bf16_mfma_wide_gfx942.hip"
    );
    assert_eq!(
        m["compiler"]["flags"],
        serde_json::json!([
            "--genco",
            "--offload-arch=gfx942",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6",
            "--rocm-path=<rocm>",
            "--hip-path=<rocm>",
            "-I<rocm>/include",
            "-fmacro-prefix-map=<source_dir>/="
        ])
    );
}

#[test]
fn all_three_entries_reproduce_the_full_image_reports_and_rows16_abi() {
    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect();
    let sealed = seal_many(ANCHOR, Arch::Gfx942, &specs).expect("seal every entry");
    let verified = verify_hxaco_many(HXACO, Arch::Gfx942, &SYMBOLS).expect("verify every entry");
    assert_eq!(sealed.len(), 3);
    assert_eq!(verified.len(), 3);
    let m = manifest();
    let reports = m["seal"]["reports"].as_array().expect("reports");
    assert_eq!(reports.len(), 3);
    let expected_args: Vec<_> = (0..9)
        .map(|i| (i * 8, 8, true))
        .chain((0..8).map(|i| (72 + i * 4, 4, false)))
        .collect();
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let (a, v, r) = (&sealed[i], &verified[i], &reports[i]);
        assert_eq!(a.hxaco, HXACO, "{symbol}");
        assert_eq!(a.elf_sha256, ELF_SHA, "{symbol}");
        assert_eq!(a.text_sha256, v.text_sha256, "{symbol}");
        assert_eq!(
            pinned_report(&a.report),
            pinned_report(r),
            "{symbol}: fresh seal evidence must not drift"
        );
        assert_eq!(
            pinned_report(&v.report),
            pinned_report(r),
            "{symbol}: fresh verification evidence must not drift"
        );
        assert_eq!(r["kernel"], *symbol);
        assert_eq!(r["wavefront_size"], 64);
        assert_eq!(r["lifted_kernels"], 3);
        assert_eq!(r["scratch"], 0);
        assert_eq!(r["lds"], 0);
        assert_eq!(
            r["lift_identity"],
            "emit::checked ok; emitted bytes == input ELF"
        );
        assert_eq!(r["hazards"]["checked"], true);
        for field in ["findings", "blocking", "bypasses", "advisory"] {
            assert_eq!(r["hazards"][field], 0, "{symbol}: {field}");
        }
        for field in ["inserted_nops", "rewritten_instructions", "violations"] {
            assert_eq!(r["hazards"]["hazard"][field], 0, "{symbol}: {field}");
        }
        assert_eq!(
            r["hazards"]["ledger"]["verifier"],
            "peacemaker_isa::ledger_cfg::verify_with_branch_proofs"
        );
        assert_eq!(r["abi"]["kernarg_segment_size"], 104);
        let args: Vec<_> = r["abi"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
            .map(|a| {
                (
                    a["offset"].as_u64().unwrap(),
                    a["size"].as_u64().unwrap(),
                    a["pointer"].as_bool().unwrap(),
                )
            })
            .collect();
        assert_eq!(args, expected_args, "{symbol}");
    }
    for bytes in [ANCHOR, HXACO] {
        let native = elf(bytes);
        let lifted = peacemaker_lift::cdna::lift_object(native).expect("typed lift");
        assert_eq!(peacemaker_lift::cdna::checked(&lifted).unwrap(), native);
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
}

#[test]
fn explicit_anchor_abi_comparison_roundtrips() {
    let specs: Vec<_> = SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: Some((ANCHOR, symbol)),
        })
        .collect();
    let sealed = seal_many(ANCHOR, Arch::Gfx942, &specs).expect("anchor ABI comparison");
    for (s, symbol) in sealed.iter().zip(SYMBOLS) {
        assert_eq!(s.hxaco, HXACO);
        assert_eq!(s.report["abi"]["compared_to_incumbent"], true);
        assert_eq!(s.report["abi"]["incumbent_symbol"], symbol);
        assert_eq!(s.report["abi"]["incumbent_sha256"], ELF_SHA);
    }
}

#[test]
fn exact_image_tier_c_receipt_is_bound_to_all_three_entries() {
    let m = manifest();
    let r = &m["gpu_oracle"];
    assert_eq!(r["status"], "PASS");
    assert_eq!(r["target"], "gfx942");
    assert_eq!(r["full_matrix"], true);
    assert_eq!(r["input_hxaco_sha256"], sha(HXACO));
    assert_eq!(r["hipcc_image_sha256"], ANCHOR_SHA);
    assert_eq!(r["pm_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["hipcc_device_elf_sha256"], ELF_SHA);
    assert_eq!(r["symbols"], serde_json::json!(SYMBOLS));
    assert_eq!(r["bitwise_checks"], 4008);
    assert_eq!(r["declined_untouched"], 12);
    assert_eq!(r["failed_cases"], 0);
    assert_eq!(r["exit_code"], 0);
    assert_eq!(r["qa_pass"], true);
    assert_eq!(r["supervisor_pass"], true);
    assert!(r["ledger"]["row"].as_u64().unwrap() > 0);
    for field in ["stdout_sha256", "oracle_binary_sha256"] {
        let hash = r[field].as_str().expect("receipt hash");
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()));
    }
}

#[test]
fn malformed_images_unknown_symbols_and_wrong_architectures_refuse() {
    assert!(verify_hxaco_many(HXACO, Arch::Gfx942, &["absent"]).is_err());
    assert!(verify_hxaco_many(&HXACO[..HXACO.len() / 2], Arch::Gfx942, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1201, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(&[], Arch::Gfx942, &SYMBOLS).is_err());
    let mut corrupted = HXACO.to_vec();
    corrupted[0] ^= 0xff;
    assert!(verify_hxaco_many(&corrupted, Arch::Gfx942, &SYMBOLS).is_err());
}
