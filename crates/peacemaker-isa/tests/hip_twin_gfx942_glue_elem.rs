// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 C-fold glue-elem Route-A twin.
//! Every artifact input is a committed file (`include_bytes!`); nothing is compiled or shelled
//! out. Regenerate with `scripts/pm-gfx942-glue-elem-twin.py`.
//!
//! The kernel source is owned by another change and is deliberately NOT embedded here: the
//! manifest's source sha256 is compared with the pin below instead.
//!
//! One bundle holds `qwen4_g942_gemm_bf16_rows16_hc_act` and
//! `qwen4_g942_gemm_bf16_rows16_hc_act_x2`: every kernel of the object.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{
    ledger_diagnostics, pinned_report, seal_many, verify_hxaco_many, KernelSpec,
};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 2] = [
    "qwen4_g942_gemm_bf16_rows16_hc_act",
    "qwen4_g942_gemm_bf16_rows16_hc_act_x2",
];
const SOURCE_PATH: &str = "kernels/src/qwen4_g942_glue_elem.hip";
const SOURCE_SHA256: &str = "2fac7e00d2f8021bea81a48d85b16f75f9b842c06a4a7abbd00f17b382c1ff4f";
const ANCHOR_SHA256: &str = "06f54962af6c84457bb46c6603f8dd12ed64b3ad25a0f1c78dfdb8d426e213ca";
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_g942_glue_elem_hipcc_gfx942.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/qwen4_g942_glue_elem_pm_gfx942.hxaco");
const SHA_FILE: &str = include_str!("../../../kernels/qwen4_g942_glue_elem_pm_gfx942.hxaco.sha256");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_g942_glue_elem_pm_gfx942.twin.json");

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
fn manifest_records_pinned_source_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(s(&m, &["arch"]), ARCH);
    assert_eq!(s(&m, &["module"]), "qwen4_g942_glue_elem_pm_gfx942");
    let symbols: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|v| v.as_str().expect("symbol string"))
        .collect();
    assert_eq!(symbols, SYMBOLS);
    // Portable: a repo-relative path, never a host path.
    assert_eq!(s(&m, &["source", "path"]), SOURCE_PATH);
    assert_eq!(
        s(&m, &["source", "sha256"]),
        SOURCE_SHA256,
        "twin was generated from a different source than the pinned one"
    );
    assert!(n(&m, &["source", "bytes"]) > 0);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "qwen4_g942_glue_elem_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), ANCHOR_SHA256);
    assert_eq!(hex(ANCHOR), ANCHOR_SHA256, "committed anchor bytes");
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "qwen4_g942_glue_elem_pm_gfx942.hxaco"
    );
    assert_eq!(s(&m, &["hxaco", "sha256"]), hex(HXACO));
    assert_eq!(n(&m, &["hxaco", "bytes"]), HXACO.len() as u64);
    assert_eq!(SHA_FILE.trim(), hex(HXACO), ".sha256 file");
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
    for line in m["compiler"]["version"].as_array().unwrap() {
        assert!(
            !line.as_str().unwrap().contains("/home/"),
            "compiler provenance carries a host path"
        );
    }
}

/// Whole-device ELF identity and text equality between anchor and twin, and reseal
/// reproduction of the committed bundle, for both symbols.
#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    let m = manifest();
    let from_anchor =
        seal_many(ANCHOR, arch(), &specs()).unwrap_or_else(|e| panic!("seal_many: {e}"));
    let from_twin = verify_hxaco_many(HXACO, arch(), &SYMBOLS)
        .unwrap_or_else(|e| panic!("verify_hxaco_many: {e}"));
    assert_eq!(from_anchor.len(), SYMBOLS.len());
    assert_eq!(from_twin.len(), SYMBOLS.len());
    let anchor_elf = device_elf(ANCHOR);
    let twin_elf = device_elf(HXACO);
    assert_eq!(
        anchor_elf, twin_elf,
        "anchor and twin device ELF must be byte-identical"
    );
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
        assert_eq!(t.elf_sha256, hex(twin_elf), "{symbol}");
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

/// Recorded reports and a fresh verify of the committed twin agree; the hazard ledger is run
/// with no findings, bypasses, blocking hazards or violations, for each sealed symbol.
#[test]
fn gfx942_checked_evidence_is_recorded_reproduced_and_clean() {
    let m = manifest();
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let r = &reports[i];
        let h = &r["hazards"];
        assert_eq!(h["checked"], true, "{symbol}: hazards not checked");
        assert_eq!(h["blocking"], 0, "{symbol}");
        assert_eq!(h["findings"], 0, "{symbol}");
        assert_eq!(h["bypasses"], 0, "{symbol}");
        assert_eq!(h["hazard"]["violations"], 0, "{symbol}");
        assert_eq!(h["unreachable_instructions"], 0, "{symbol}");
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

/// The ledger's exploration counters (`steps`, `label_states`) are recorded diagnostics, not
/// pinned claims: the committed legacy reports still carry them, they may drift without
/// breaking the pin, while hazards, findings, structural counts and the other claims may not.
#[test]
fn exploration_counters_are_diagnostics_but_hazards_findings_and_claims_stay_pinned() {
    let m = manifest();
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    for (i, symbol) in SYMBOLS.iter().enumerate() {
        let r = &reports[i];
        let fresh = &fresh[i].report;
        assert!(
            ledger_diagnostics(r).is_some(),
            "{symbol}: the legacy report records its exploration counters"
        );
        assert_eq!(pinned_report(fresh), pinned_report(r), "{symbol}");

        let mut drifted = r.clone();
        drifted["hazards"]["ledger"]["steps"] = serde_json::json!(1);
        drifted["hazards"]["ledger"]["label_states"] = serde_json::json!(1);
        assert_ne!(&drifted, r, "{symbol}");
        assert_eq!(pinned_report(fresh), pinned_report(&drifted), "{symbol}");

        let mutations: [(&str, &[&str], Value); 7] = [
            ("findings", &["hazards", "findings"], 1.into()),
            ("blocking", &["hazards", "blocking"], 1.into()),
            ("violations", &["hazards", "hazard", "violations"], 1.into()),
            (
                "branches",
                &["hazards", "ledger", "branches"],
                10_000.into(),
            ),
            ("waitcnt", &["hazards", "ledger", "waitcnt"], 10_000.into()),
            ("text_sha256", &["text_sha256"], "00".into()),
            ("lift_identity", &["lift_identity"], "none".into()),
        ];
        for (name, path, value) in mutations {
            let mut changed = drifted.clone();
            let mut at = &mut changed;
            for key in path {
                at = &mut at[*key];
            }
            *at = value;
            assert_ne!(
                pinned_report(fresh),
                pinned_report(&changed),
                "{symbol}: {name} must stay pinned"
            );
        }
    }
}

/// Every kernel lifts to typed CDNA instructions and certifies on both the anchor and twin,
/// re-emits byte-identically, and the object holds exactly the two entries, no atomics.
#[test]
fn typed_lift_certifies_every_kernel_with_no_atomics() {
    for (label, elf) in [("anchor", device_elf(ANCHOR)), ("twin", device_elf(HXACO))] {
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
            assert!(
                !text.iter().any(|l| l.contains("atomic")
                    || [
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
                        "ds_wrxchg"
                    ]
                    .iter()
                    .any(|p| l.starts_with(p))),
                "{label}: {name} contains an atomic"
            );
        }
    }
}

/// Hardware byte-oracle receipt bound to both exact input images and device ELFs. Main fills a
/// simple schema; this test only accepts a PASS with zero failed cases that names these hashes.
#[test]
fn gpu_oracle_receipt_is_pass_and_bound_to_these_images() {
    let m = manifest();
    let receipt = m.get("gpu_oracle").expect("hardware byte-oracle receipt");
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["input_hxaco_sha256"], hex(HXACO));
    assert_eq!(receipt["hipcc_image_sha256"], hex(ANCHOR));
    assert_eq!(receipt["failed_cases"], 0);
    assert!(receipt["case_count"].as_u64().expect("case_count") > 0);
    let seal = &m["seal"]["reports"][0];
    assert_eq!(receipt["pm_device_elf_sha256"], seal["elf_sha256"]);
    assert_eq!(receipt["hipcc_device_elf_sha256"], seal["elf_sha256"]);
}

/// Exact explicit kernarg layout of both entries: three pointers at 0/8/16 then five 32-bit
/// scalars at 24..40; 44 bytes. From the fresh lift and the manifest alike.
#[test]
fn kernarg_abi_is_the_declared_layout_of_each_entry() {
    let want: Vec<(u64, u64, bool)> = vec![
        (0, 8, true),
        (8, 8, true),
        (16, 8, true),
        (24, 4, false),
        (28, 4, false),
        (32, 4, false),
        (36, 4, false),
        (40, 4, false),
    ];
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
            assert_eq!(
                report["abi"]["kernarg_segment_size"].as_u64().unwrap(),
                44,
                "{symbol} ({label}) kernarg segment"
            );
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
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1201, &SYMBOLS).is_err());
    let mut flipped = HXACO.to_vec();
    flipped[0] ^= 0xff;
    assert!(verify_hxaco_many(&flipped, arch(), &SYMBOLS).is_err());
}
