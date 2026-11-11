// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 Qwen4 MoE glue Route-A twin.
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-gfx942-glue-moe-twin.py`.
//!
//! One bundle holds `qwen4_g942_moe_group_scatter_desc`, `qwen4_g942_moe_unscatter_act` and
//! `qwen4_g942_moe_combine_shared_add`. The kernel source is owned by another change and is
//! not included here; the manifest's source pin is compared to the pinned constant instead.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 3] = [
    "qwen4_g942_moe_group_scatter_desc",
    "qwen4_g942_moe_unscatter_act",
    "qwen4_g942_moe_combine_shared_add",
];
const SOURCE_SHA256: &str = "0f2e37e06e386ceaf9c95defeda3e7cd7d73355bf4367fb96fb39108ca0eb2bc";
const ANCHOR_SHA256: &str = "e3304e667978ca5c8b32c9cd6b2d8a1580a1d6121e9327bd1bfd36afe39ff4c1";
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_g942_glue_moe_hipcc_gfx942.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/qwen4_g942_glue_moe_pm_gfx942.hxaco");
const SHA_FILE: &str = include_str!("../../../kernels/qwen4_g942_glue_moe_pm_gfx942.hxaco.sha256");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_g942_glue_moe_pm_gfx942.twin.json");

/// `(offset, size, pointer)` of the explicit kernargs, in source order (natural alignment).
type Layout = Vec<(u64, u64, bool)>;

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
    assert_eq!(s(&m, &["module"]), "qwen4_g942_glue_moe_pm_gfx942");
    let symbols: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|v| v.as_str().expect("symbol string"))
        .collect();
    assert_eq!(symbols, SYMBOLS);
    // Portable: the repo-relative logical path, never a host path.
    assert_eq!(
        s(&m, &["source", "path"]),
        "kernels/src/qwen4_g942_glue_moe.hip"
    );
    assert_eq!(
        s(&m, &["source", "sha256"]),
        SOURCE_SHA256,
        "source changed since the twin was generated; rerun scripts/pm-gfx942-glue-moe-twin.py"
    );
    assert!(n(&m, &["source", "bytes"]) > 0);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "qwen4_g942_glue_moe_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), ANCHOR_SHA256);
    assert_eq!(hex(ANCHOR), ANCHOR_SHA256, "committed anchor bytes");
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "qwen4_g942_glue_moe_pm_gfx942.hxaco"
    );
    assert_eq!(s(&m, &["hxaco", "sha256"]), hex(HXACO));
    assert_eq!(n(&m, &["hxaco", "bytes"]), HXACO.len() as u64);
    assert_eq!(SHA_FILE.trim(), hex(HXACO), ".sha256 file");
    assert!(
        !MANIFEST.contains("/home/") && !MANIFEST.contains("/tmp/"),
        "manifest must carry no host paths"
    );
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

/// Whole-device ELF identity and per-symbol text/ABI/reseal equality for all three entries.
#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    let m = manifest();
    let from_anchor =
        seal_many(ANCHOR, arch(), &specs()).unwrap_or_else(|e| panic!("seal_many: {e}"));
    let from_twin = verify_hxaco_many(HXACO, arch(), &SYMBOLS)
        .unwrap_or_else(|e| panic!("verify_hxaco_many: {e}"));
    assert_eq!(from_anchor.len(), SYMBOLS.len());
    assert_eq!(from_twin.len(), SYMBOLS.len());
    // The full device ELF of the twin bundle is exactly the hipcc anchor object.
    assert_eq!(
        device_elf(HXACO),
        device_elf(ANCHOR),
        "twin device ELF != anchor device ELF"
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
        assert_eq!(a.report["abi"], t.report["abi"], "{symbol}: abi");
        assert_eq!(t.hxaco_sha256, hex(HXACO), "{symbol}");
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
    // One whole-module ELF: every symbol reports the same device-ELF hash.
    assert!(from_twin
        .iter()
        .all(|t| t.elf_sha256 == from_twin[0].elf_sha256));
}

/// The generated reports and a fresh verify of the committed twin must agree; the hazard
/// ledger must be run (`checked`) with no blocking entries, findings or bypasses.
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
        if let Some(b) = r["hazards"].get("bypasses") {
            assert!(
                b == 0 || b.as_array().is_some_and(|a| a.is_empty()),
                "{symbol}: hazard bypasses {b}"
            );
        }
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

/// Typed coverage on both the hipcc anchor and the sealed twin: every kernel lifts to typed
/// CDNA instructions, certifies (hazards plus the typed CFG wait ledger) and re-emits the
/// image bytes exactly; the symbol set is exactly the three glue entries; no atomics (the
/// glue kernels are deterministic, exchanging only through LDS behind barriers).
#[test]
fn typed_lift_certifies_every_kernel_and_has_no_atomics() {
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
            assert!(!text.is_empty(), "{label}: {name}: no instructions");
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
                        "ds_wrxchg",
                    ]
                    .iter()
                    .any(|p| l.starts_with(p))),
                "{label}: {name} contains an atomic"
            );
        }
    }
}

/// Hardware byte-oracle receipt bound to both exact input images and the shared device ELF.
/// Absent or unbound receipt fails closed.
#[test]
fn gpu_oracle_receipt_is_bound_to_these_images() {
    let m = manifest();
    let receipt = m.get("gpu_oracle").expect("hardware byte-oracle receipt");
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["input_hxaco_sha256"], hex(HXACO));
    assert_eq!(receipt["hipcc_image_sha256"], hex(ANCHOR));
    assert_eq!(receipt["failed_cases"], 0);
    assert!(receipt["case_count"].as_u64().expect("case_count") > 0);
    let elf = hex(device_elf(HXACO));
    assert_eq!(elf, hex(device_elf(ANCHOR)));
    assert_eq!(receipt["pm_device_elf_sha256"], elf);
    assert_eq!(receipt["hipcc_device_elf_sha256"], elf);
    for r in m["seal"]["reports"].as_array().expect("seal.reports") {
        assert_eq!(receipt["pm_device_elf_sha256"], r["elf_sha256"]);
    }
}

/// Exact explicit kernarg layouts (natural alignment) of each entry, from the fresh lift and
/// the recorded manifest alike.
#[test]
fn kernarg_abi_is_the_declared_layout_of_each_entry() {
    let p = |o| (o, 8, true);
    let i = |o| (o, 4, false);
    // topk, counts, offsets, sorted, tiles, inverse, tile_desc, L, E, Pmax
    let scatter: Layout = vec![
        p(0),
        p(8),
        p(16),
        p(24),
        p(32),
        p(40),
        p(48),
        i(56),
        i(60),
        i(64),
    ];
    // Y_grouped, sorted_slot_index, activation, mi, n_sorted, slots, sh_gate, sh_up, sh_act,
    // sh_selector, sh_rows, sh_mi, act_blocks
    let unscatter: Layout = vec![
        p(0),
        p(8),
        p(16),
        i(24),
        i(28),
        i(32),
        p(40),
        p(48),
        p(56),
        p(64),
        i(72),
        i(76),
        i(80),
    ];
    // grouped_down, inverse_perm, topk_indices, topk_weights, residual, hidden, grouped_rows,
    // tokens, n_experts, shared_out, scalar
    let combine: Layout = vec![
        p(0),
        p(8),
        p(16),
        p(24),
        p(32),
        i(40),
        i(44),
        i(48),
        i(52),
        p(56),
        p(64),
    ];
    // Full metadata segment sizes include hidden launch arguments where present.
    let wants = [(&scatter, 68u64), (&unscatter, 344), (&combine, 328)];
    let m = manifest();
    let fresh = verify_hxaco_many(HXACO, arch(), &SYMBOLS).expect("verify");
    let reports = m["seal"]["reports"].as_array().expect("seal.reports");
    for (idx, symbol) in SYMBOLS.iter().enumerate() {
        let (want, segment_size) = wants[idx];
        for (label, report) in [("fresh", &fresh[idx].report), ("manifest", &reports[idx])] {
            let args = report["abi"]["args"].as_array().expect("abi.args");
            let explicit: Layout = args
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
            assert_eq!(kernarg, segment_size, "{symbol} ({label}) kernarg segment");
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
