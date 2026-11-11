// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 BF16 MFMA rows16 Route-A twin.
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-gfx942-rows16-twin.py`.
//!
//! One bundle holds `gemm_bf16_mfma_gfx942_rows16` and `gemm_bf16_mfma_gfx942_rows16_x2`.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, verify_hxaco_many, KernelSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 2] = [
    "gemm_bf16_mfma_gfx942_rows16",
    "gemm_bf16_mfma_gfx942_rows16_x2",
];
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/gemm_bf16_mfma.gfx942.hip");
const ANCHOR: &[u8] =
    include_bytes!("../../../kernels/gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co");
const HXACO: &[u8] =
    include_bytes!("../../../kernels/gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco");
const SHA_FILE: &str =
    include_str!("../../../kernels/gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco.sha256");
const MANIFEST: &str =
    include_str!("../../../kernels/gemm_bf16_mfma_gfx942_rows16_pm_gfx942.twin.json");

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).unwrap_or_else(|e| panic!("manifest: {e}"))
}

fn arch() -> Arch {
    ARCH.parse::<Arch>()
        .unwrap_or_else(|e| panic!("{ARCH} is not a known Arch: {e}"))
}

fn s<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    let mut cur = v;
    for k in path {
        cur = &cur[*k];
    }
    cur.as_str()
        .unwrap_or_else(|| panic!("manifest field {path:?} missing or not a string"))
}

fn n(v: &Value, path: &[&str]) -> u64 {
    let mut cur = v;
    for k in path {
        cur = &cur[*k];
    }
    cur.as_u64()
        .unwrap_or_else(|| panic!("manifest field {path:?} missing or not a number"))
}

fn specs() -> Vec<KernelSpec<'static>> {
    SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: None,
        })
        .collect()
}

#[test]
fn manifest_records_source_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(s(&m, &["arch"]), ARCH);
    assert_eq!(s(&m, &["module"]), "gemm_bf16_mfma_gfx942_rows16_pm_gfx942");
    let symbols: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|v| v.as_str().expect("symbol string"))
        .collect();
    assert_eq!(symbols, SYMBOLS);
    assert_eq!(
        s(&m, &["source", "path"]),
        "kernels/src/gemm_bf16_mfma.gfx942.hip"
    );
    assert_eq!(
        s(&m, &["source", "sha256"]),
        hex(SOURCE),
        "gemm_bf16_mfma.gfx942.hip changed since the twin was generated; rerun scripts/pm-gfx942-rows16-twin.py"
    );
    assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), hex(ANCHOR));
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco"
    );
    assert_eq!(s(&m, &["hxaco", "sha256"]), hex(HXACO));
    assert_eq!(n(&m, &["hxaco", "bytes"]), HXACO.len() as u64);
    assert_eq!(SHA_FILE.trim(), hex(HXACO), ".sha256 file");
}

#[test]
fn source_defines_both_rows16_entries() {
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    assert!(src.contains("R16_ENTRY(gemm_bf16_mfma_gfx942_rows16, 1)"));
    assert!(src.contains("R16_ENTRY(gemm_bf16_mfma_gfx942_rows16_x2, 2)"));
}

#[test]
fn manifest_flags_are_the_portable_core_recipe() {
    let m = manifest();
    let flags: Vec<&str> = m["compiler"]["flags"]
        .as_array()
        .expect("compiler.flags")
        .iter()
        .map(|f| f.as_str().expect("flag string"))
        .collect();
    let want = [
        "--genco",
        "--offload-arch=gfx942",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
    ];
    assert_eq!(flags, want);
    assert!(
        flags.iter().all(|f| !f.contains('/')),
        "flags must carry no paths"
    );
    assert!(
        !m["compiler"]["version"]
            .as_array()
            .expect("version")
            .is_empty(),
        "compiler provenance missing"
    );
}

#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    let m = manifest();
    let from_anchor =
        seal_many(ANCHOR, arch(), &specs()).unwrap_or_else(|e| panic!("seal_many: {e}"));
    let from_twin = verify_hxaco_many(HXACO, arch(), &SYMBOLS)
        .unwrap_or_else(|e| panic!("verify_hxaco_many: {e}"));
    assert_eq!(from_anchor.len(), SYMBOLS.len());
    assert_eq!(from_twin.len(), SYMBOLS.len());
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    assert_eq!(reports.len(), SYMBOLS.len());
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let (a, t, r) = (&from_anchor[i], &from_twin[i], &reports[i]);
        assert_eq!(
            a.hxaco, HXACO,
            "{symbol}: sealing the anchor must reproduce the hxaco"
        );
        assert_eq!(a.elf_sha256, t.elf_sha256, "{symbol}");
        assert_eq!(a.text_sha256, t.text_sha256, "{symbol}");
        assert_eq!(t.hxaco_sha256, hex(HXACO), "{symbol}");
        // The manifest's seal report is the generated fact for these exact bytes.
        assert_eq!(r["kernel"], *symbol);
        assert_eq!(r["arch"], ARCH);
        assert_eq!(r["elf_sha256"], t.elf_sha256, "{symbol}");
        assert_eq!(r["text_sha256"], t.text_sha256, "{symbol}");
        assert_eq!(r["hxaco_sha256"], hex(HXACO), "{symbol}");
        assert_eq!(r["vgprs"], t.vgprs, "{symbol}");
        assert_eq!(r["sgprs"], t.sgprs, "{symbol}");
        assert_eq!(r["lds"], t.lds, "{symbol}");
        assert_eq!(r["scratch"], t.scratch, "{symbol}");
        assert_eq!(r["wavefront_size"], 64, "{symbol}");
        assert_eq!(
            r["lift_identity"], "emit::checked ok; emitted bytes == input ELF",
            "{symbol}"
        );
    }
}

/// The checked evidence of the gfx942 path: the generated reports and a fresh seal of the
/// committed anchor must agree, and the hazard ledger must be run (`checked`) with nothing
/// blocking, for each sealed symbol.
#[test]
fn gfx942_checked_evidence_is_recorded_and_reproduced() {
    let m = manifest();
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let r = &reports[i];
        assert_eq!(
            r["hazards"]["checked"], true,
            "{symbol}: hazards not checked"
        );
        assert_eq!(r["hazards"]["blocking"], 0, "{symbol}");
        assert_eq!(fresh[i].report["hazards"], r["hazards"], "{symbol}");
        assert_eq!(
            fresh[i].report["lift_identity"], r["lift_identity"],
            "{symbol}"
        );
        assert_eq!(fresh[i].report["abi"], r["abi"], "{symbol}");
    }
}

#[test]
fn gpu_oracle_receipt_is_full_and_bound_to_these_images() {
    let m = manifest();
    let receipt = &m["gpu_oracle"];
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["full_matrix"], true);
    assert_eq!(receipt["input_hxaco_sha256"], hex(HXACO));
    assert_eq!(receipt["hipcc_image_sha256"], hex(ANCHOR));
    assert_eq!(receipt["failed_cases"], 0);
    assert_eq!(receipt["results"]["raw_outputs_pm_equals_hipcc"], true);
    assert_eq!(receipt["results"]["raw_outputs_hipcc_equals_tiled"], true);
    assert_eq!(receipt["results"]["raw_outputs_pm_equals_tiled"], true);
    assert_eq!(receipt["results"]["all_checks_pass"], true);
    assert_eq!(
        receipt["tiled_comparison"]["kernel"],
        "gemm_bf16_mfma_gfx942"
    );
    assert_eq!(receipt["tiled_comparison"]["cases_hipcc_ne_tiled"], 0);
    assert_eq!(receipt["tiled_comparison"]["cases_pm_ne_tiled"], 0);
    assert!(receipt["case_count"].as_u64().expect("case_count") > 0);
    assert_eq!(receipt["rows"], serde_json::json!([1, 7, 8, 9, 16]));
    for symbol in SYMBOLS {
        assert!(
            receipt["cases_by_kernel"][symbol]
                .as_u64()
                .expect("kernel cases")
                > 0
        );
    }
    let seal = &m["seal"]["reports"][0];
    assert_eq!(receipt["pm_device_elf_sha256"], seal["elf_sha256"]);
    assert_eq!(receipt["hipcc_device_elf_sha256"], seal["elf_sha256"]);
}

/// Exact explicit kernarg layout shared by both entries (`R16_ENTRY`):
/// `X, A0..A3, D0..D3` (nine pointers, offsets 0..72) then
/// `M0..M3, K, batch, x_f32, d_bf16_mask` (eight i32, offsets 72..104).
#[test]
fn kernarg_abi_is_the_seventeen_explicit_args_of_both_entries() {
    let mut want: Vec<(u64, u64, bool)> = (0..9).map(|i| (i * 8, 8, true)).collect();
    want.extend((0..8).map(|i| (72 + i * 4, 4, false)));
    let m = manifest();
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        for (label, report) in [("fresh", &fresh[i].report), ("manifest", &reports[i])] {
            let args = report["abi"]["args"].as_array().expect("abi.args");
            let explicit: Vec<(u64, u64, bool)> = args
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
            assert_eq!(explicit, want, "{symbol} ({label})");
        }
    }
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    let specs: Vec<KernelSpec<'_>> = SYMBOLS
        .iter()
        .map(|&symbol| KernelSpec {
            symbol,
            abi_ref: Some((ANCHOR, symbol)),
        })
        .collect();
    let sealed = seal_many(ANCHOR, arch(), &specs).unwrap_or_else(|e| panic!("abi_ref seal: {e}"));
    for (sealed, symbol) in sealed.iter().zip(SYMBOLS) {
        assert_eq!(sealed.hxaco, HXACO, "{symbol}");
        assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true);
        assert_eq!(sealed.report["abi"]["incumbent_symbol"], symbol);
        assert_eq!(sealed.report["abi"]["incumbent_sha256"], sealed.elf_sha256);
    }
    // Same bytes, wrong incumbent symbol: refused.
    let bad = [KernelSpec {
        symbol: SYMBOLS[0],
        abi_ref: Some((ANCHOR, "no_such_kernel")),
    }];
    assert!(seal_many(ANCHOR, arch(), &bad).is_err());
}

#[test]
fn malformed_artifacts_and_unknown_symbols_are_rejected() {
    assert!(verify_hxaco_many(HXACO, arch(), &["no_such_kernel"]).is_err());
    assert!(verify_hxaco_many(&HXACO[..HXACO.len() / 2], arch(), &SYMBOLS).is_err());
    assert!(verify_hxaco_many(&[], arch(), &SYMBOLS).is_err());
    assert!(seal_many(&ANCHOR[..ANCHOR.len() / 2], arch(), &specs()).is_err());
    // A non-gfx942 request for the gfx942 object must not seal.
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1201, &SYMBOLS).is_err());
    // A corrupted bundle magic is structurally invalid. Arbitrary valid instruction edits
    // need not be rejected by re-emission; the manifest's SHA pin catches those.
    let mut flipped = HXACO.to_vec();
    flipped[0] ^= 0xff;
    assert!(verify_hxaco_many(&flipped, arch(), &SYMBOLS).is_err());
}
