// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 FP8 grouped MoE Route-A twin.
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-gfx942-moe-grouped-twin.py`.
//!
//! One bundle holds `gemm_fp8b128_moe_grouped_bf16mfma`,
//! `gemm_fp8b128_moe_grouped_bf16mfma_xbf16` and `gemm_fp8b128_moe_tile_desc_pad16`: every
//! kernel of the production object (the diagnostic `_reference` entries exist only under
//! `FP8MOE_TEST_REFERENCE`, which the production build never defines).

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 3] = [
    "gemm_fp8b128_moe_grouped_bf16mfma",
    "gemm_fp8b128_moe_grouped_bf16mfma_xbf16",
    "gemm_fp8b128_moe_tile_desc_pad16",
];
/// The two grouped GEMM entries share one kernarg layout; the descriptor adapter has its own.
const GEMM_SYMBOLS: [&str; 2] = [SYMBOLS[0], SYMBOLS[1]];
const SOURCE: &[u8] =
    include_bytes!("../../../kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip");
const ANCHOR: &[u8] =
    include_bytes!("../../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co");
const HXACO: &[u8] =
    include_bytes!("../../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco");
const SHA_FILE: &str =
    include_str!("../../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco.sha256");
const MANIFEST: &str =
    include_str!("../../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.twin.json");

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

/// The single native `hipv4-amdgcn-amd-amdhsa--gfx942` device entry of an offload bundle.
fn device_elf(bundle_bytes: &[u8]) -> &[u8] {
    let (bundle, payloads) = Bundle::read(bundle_bytes).expect("readable bundle");
    let triple = format!(
        "{}{ARCH}",
        peacemaker_lift::bundle::HIP_DEVICE_TRIPLE_PREFIX
    );
    let hits: Vec<usize> = bundle
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.triple == triple
                || e.triple
                    .strip_prefix(&triple)
                    .is_some_and(|rest| rest.starts_with(':'))
        })
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "exactly one gfx942 device entry");
    payloads[hits[0]]
}

#[test]
fn manifest_records_source_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(s(&m, &["arch"]), ARCH);
    assert_eq!(
        s(&m, &["module"]),
        "gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942"
    );
    let symbols: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|v| v.as_str().expect("symbol string"))
        .collect();
    assert_eq!(symbols, SYMBOLS);
    assert_eq!(
        s(&m, &["source", "path"]),
        "kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip"
    );
    assert_eq!(
        s(&m, &["source", "sha256"]),
        hex(SOURCE),
        "gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip changed since the twin was generated; rerun scripts/pm-gfx942-moe-grouped-twin.py"
    );
    assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), hex(ANCHOR));
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco"
    );
    assert_eq!(s(&m, &["hxaco", "sha256"]), hex(HXACO));
    assert_eq!(n(&m, &["hxaco", "bytes"]), HXACO.len() as u64);
    assert_eq!(SHA_FILE.trim(), hex(HXACO), ".sha256 file");
}

/// The three production entries are the source's only unconditional kernels; the reference
/// variants sit behind `FP8MOE_TEST_REFERENCE` and so are in neither committed image.
#[test]
fn source_defines_the_production_entries_and_gates_the_reference_variants() {
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    for symbol in SYMBOLS {
        assert!(
            src.contains(&format!("void {symbol}(")),
            "{symbol} is not defined by the source"
        );
    }
    let guard = src
        .find("#if defined(FP8MOE_TEST_REFERENCE)")
        .expect("reference entries must stay behind FP8MOE_TEST_REFERENCE");
    let reference = src
        .find("void gemm_fp8b128_moe_grouped_bf16mfma_reference(")
        .expect("reference entry");
    let reference_xbf16 = src
        .find("void gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16(")
        .expect("reference x-bf16 entry");
    let endif = guard + src[guard..].find("#endif").expect("reference guard #endif");
    assert!(guard < reference && reference_xbf16 < endif, "guard span");
    for reference in [
        "gemm_fp8b128_moe_grouped_bf16mfma_reference",
        "gemm_fp8b128_moe_grouped_bf16mfma_reference_xbf16",
    ] {
        assert!(
            verify_hxaco_many(HXACO, arch(), &[reference]).is_err(),
            "{reference} must not be in the production twin"
        );
        assert!(
            seal_many(
                ANCHOR,
                arch(),
                &[KernelSpec {
                    symbol: reference,
                    abi_ref: None
                }]
            )
            .is_err(),
            "{reference} must not be in the production anchor"
        );
    }
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
        assert_eq!(r["lifted_kernels"], SYMBOLS.len(), "{symbol}");
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
        assert_eq!(r["hazards"]["findings"], 0, "{symbol}");
        assert_eq!(
            pinned_report(&fresh[i].report)["hazards"],
            pinned_report(r)["hazards"],
            "{symbol}"
        );
        assert_eq!(
            fresh[i].report["lift_identity"], r["lift_identity"],
            "{symbol}"
        );
        assert_eq!(fresh[i].report["abi"], r["abi"], "{symbol}");
    }
}

/// Typed coverage of the kernel classes this object exercises, on both the hipcc anchor and
/// the sealed twin: every kernel lifts to typed CDNA instructions and certifies (hazards plus
/// the typed CFG wait ledger); the grouped GEMMs contain the FP8 FNUZ pair convert, the LDS
/// stage/read and the BF16 MFMA; nothing in the object is an atomic (fixed accumulation order,
/// no split-K). The image bytes are those the lift re-emits exactly.
#[test]
fn typed_lift_certifies_every_kernel_and_covers_the_fp8_convert_lds_and_mfma_classes() {
    for (label, bytes) in [("anchor", ANCHOR), ("twin", HXACO)] {
        let elf = device_elf(bytes);
        let lifted = peacemaker_lift::cdna::lift_object(elf)
            .unwrap_or_else(|e| panic!("{label}: cdna lift: {e}"));
        assert_eq!(
            peacemaker_lift::cdna::checked(&lifted)
                .unwrap_or_else(|e| panic!("{label}: cdna checked: {e}")),
            elf,
            "{label}: typed re-emission must be byte-identical"
        );
        let mut names: Vec<String> = lifted
            .kernels
            .iter()
            .map(|k| k.image.name.clone())
            .collect();
        names.sort();
        let mut want: Vec<String> = SYMBOLS.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(names, want, "{label}: kernel symbol set");
        for kernel in &lifted.kernels {
            let name = &kernel.image.name;
            peacemaker_isa::cdna::lifted::certify(kernel)
                .unwrap_or_else(|e| panic!("{label}: certify {name}: {e}"));
            let text: Vec<String> = kernel
                .instructions
                .iter()
                .map(|i| i.inst.to_string())
                .collect();
            let has = |needle: &str| text.iter().any(|l| l.contains(needle));
            assert!(
                !has("atomic")
                    && !text.iter().any(|l| [
                        "ds_add",
                        "ds_sub",
                        "ds_inc",
                        "ds_dec",
                        "ds_min",
                        "ds_max",
                        "ds_and",
                        "ds_or",
                        "ds_xor",
                        "ds_cmpst",
                        "ds_wrxchg",
                    ]
                    .iter()
                    .any(|p| l.starts_with(p))),
                "{label}: {name} contains an atomic"
            );
            if GEMM_SYMBOLS.contains(&name.as_str()) {
                assert!(
                    has("v_cvt_pk_f32_fp8"),
                    "{label}: {name}: FNUZ pair convert"
                );
                assert!(
                    has("v_mfma_f32_16x16x16_bf16"),
                    "{label}: {name}: BF16 MFMA"
                );
                assert!(has("ds_write_b128"), "{label}: {name}: LDS stage");
                assert!(has("ds_read2_b64"), "{label}: {name}: LDS read");
                assert!(has("s_barrier"), "{label}: {name}: barrier");
            }
        }
    }
}

/// Full hardware byte-oracle receipt bound to both exact input images.
#[test]
fn gpu_oracle_receipt_is_full_and_bound_to_these_images() {
    let m = manifest();
    let receipt = m.get("gpu_oracle").expect("hardware byte-oracle receipt");
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["full_matrix"], true);
    assert_eq!(receipt["real_fixture"]["used"], true);
    assert_eq!(receipt["input_hxaco_sha256"], hex(HXACO));
    assert_eq!(receipt["hipcc_image_sha256"], hex(ANCHOR));
    assert_eq!(receipt["failed_cases"], 0);
    assert!(receipt["case_count"].as_u64().expect("case_count") > 0);
    for symbol in SYMBOLS {
        assert!(
            receipt["cases_by_kernel"][symbol]
                .as_u64()
                .expect("kernel cases")
                > 0,
            "{symbol}"
        );
    }
    let seal = &m["seal"]["reports"][0];
    assert_eq!(receipt["pm_device_elf_sha256"], seal["elf_sha256"]);
    assert_eq!(receipt["hipcc_device_elf_sha256"], seal["elf_sha256"]);
}

/// Exact explicit kernarg layouts (`W, expert_stride, tile_desc, sorted_rows, X, Y` then
/// `R, K, x_row_div, n_exp`: 64 bytes; the descriptor adapter takes `tile_ids, tile_desc,
/// n_tiles`: 20 bytes), from the fresh lift and the recorded manifest alike.
#[test]
fn kernarg_abi_is_the_declared_layout_of_each_entry() {
    let gemm: Vec<(u64, u64, bool)> = vec![
        (0, 8, true),
        (8, 8, false),
        (16, 8, true),
        (24, 8, true),
        (32, 8, true),
        (40, 8, true),
        (48, 4, false),
        (52, 4, false),
        (56, 4, false),
        (60, 4, false),
    ];
    let tile_desc: Vec<(u64, u64, bool)> = vec![(0, 8, true), (8, 8, true), (16, 4, false)];
    let m = manifest();
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let (want, segment) = if GEMM_SYMBOLS.contains(symbol) {
            (&gemm, 64)
        } else {
            (&tile_desc, 20)
        };
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
            assert_eq!(&explicit, want, "{symbol} ({label})");
            let kernarg = report["abi"]["kernarg_segment_size"].as_u64().unwrap();
            assert_eq!(kernarg, segment, "{symbol} ({label}) kernarg segment");
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
