// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 HC-gates Route-A twin.
//! Every artifact input is a committed file (`include_bytes!`); nothing is compiled or shelled
//! out. Regenerate with `scripts/pm-gfx942-hc-gates-twin.py`.
//!
//! The kernel source is owned by another change and is deliberately NOT embedded here: the
//! manifest's source sha256 is compared with the pin below instead.
//!
//! One bundle holds `qwen4_g942_hc_gates`: the full device object.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 7] = [
    "gemv_bf16_xf32_k4_rows_tiled_r2",
    "gemv_bf16_xf32_k4_rows_tiled_r3",
    "gemv_bf16_xf32_k4_rows_tiled_r4",
    "gemv_bf16_xf32_k4_rows_tiled_r5",
    "gemv_bf16_xf32_k4_rows_tiled_r6",
    "gemv_bf16_xf32_k4_rows_tiled_r7",
    "gemv_bf16_xf32_k4_rows_tiled_r8",
];
const OWNER_COMMIT: &str = "28942e230da94be36a5cfc20cc3e93be5a8f5f98";
const ANCHOR_COMMIT: &str = "135c6a5c7496793335c1cc9ce8683fa4fecf7a04";
const SOURCE_PATH: &str = "kernels/src/qwen4_g942_hc_gates.gfx942.hip";
const SOURCE_SHA256: &str = "4f94fe4e929fcbb22a8ec115220b4e21d154093403129f7aeea045d975416740";
const ANCHOR_SHA256: &str = "00be6bc34e992331dd241f520e7659ebfd0ff28515895f83839bebe0aefe9287";
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_g942_hc_gates_hipcc_gfx942.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/qwen4_g942_hc_gates_pm_gfx942.hxaco");
const SHA_FILE: &str = include_str!("../../../kernels/qwen4_g942_hc_gates_pm_gfx942.hxaco.sha256");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_g942_hc_gates_pm_gfx942.twin.json");

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
    assert_eq!(s(&m, &["module"]), "%s_pm_gfx942");
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
        "qwen4_g942_hc_gates_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), ANCHOR_SHA256);
    assert_eq!(hex(ANCHOR), ANCHOR_SHA256, "committed anchor bytes");
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "qwen4_g942_hc_gates_pm_gfx942.hxaco"
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
    assert_eq!(s(&m, &["source", "owner_commit"]), OWNER_COMMIT);
    assert_eq!(s(&m, &["source", "anchor_commit"]), ANCHOR_COMMIT);
    let provenance = s(&m, &["compiler", "provenance"]);
    assert!(provenance.contains(ANCHOR_COMMIT));
    assert!(provenance.contains("does not recompile"));
    assert!(
        m["compiler"].get("version").is_none(),
        "no recompile version array"
    );
    assert_eq!(
        s(&m, &["compiler", "runtime_source_basename"]),
        "qwen4_g942_hc_gates.gfx942.hip"
    );
    assert_eq!(
        s(&m, &["anchor", "device_elf_sha256"]),
        s(&m["seal"]["reports"][0], &["elf_sha256"])
    );
}

/// Whole-device ELF identity and text equality between anchor and twin, and reseal
/// reproduction of the committed bundle, for the symbol.
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
        assert_eq!(fresh[i].report["hazards"], *h, "{symbol}");
        assert_eq!(
            fresh[i].report["lift_identity"], r["lift_identity"],
            "{symbol}"
        );
        assert_eq!(fresh[i].report["abi"], r["abi"], "{symbol}");
    }
}

/// Every kernel lifts to typed CDNA instructions and certifies on both the anchor and twin,
/// re-emits byte-identically, and the object holds exactly the one entry, no atomics.
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

/// Hardware byte-oracle receipt bound to both exact images, the preserved raw log, and
/// the MI300X FIFO ledger row.
#[test]
fn gpu_oracle_receipt_is_pass_and_bound_to_these_images() {
    let m = manifest();
    let receipt = m
        .get("gpu_oracle")
        .expect("hardware byte-oracle receipt absent: fail closed until the parent binds it");
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["input_hxaco_sha256"], hex(HXACO));
    assert_eq!(receipt["hipcc_image_sha256"], hex(ANCHOR));
    assert_eq!(receipt["failed_cases"], 0);
    assert_eq!(receipt["exit_code"], 0);
    let seal = &m["seal"]["reports"][0];
    assert_eq!(receipt["pm_device_elf_sha256"], seal["elf_sha256"]);
    assert_eq!(receipt["hipcc_device_elf_sha256"], seal["elf_sha256"]);
    let log = include_str!("fixtures/cdna/gfx942_hc_gates_oracle.txt");
    assert_eq!(receipt["raw_log_sha256"], hex(log.as_bytes()));
    assert_eq!(receipt["case_count"], 192);
    assert_eq!(receipt["controls_passed"], 264);
    assert_eq!(receipt["controls_total"], 264);
    assert!(log.contains(
        "gate entries from image /scratch/quant/fn-fp8/beat/WC8/twin/hc/hc-twin-device.elf"
    ));
    assert!(log.contains("ORACLE PASS cases=192 mismatches=0 census_failures=0 controls=264/264"));
    assert_eq!(
        receipt["oracle_binary_sha256"],
        "4e291bf81a81f2ac48c7a288c09c71cef81a7b926bc57e37e81e7f40832dc756"
    );
    let ledger = &receipt["ledger"];
    assert_eq!(ledger["row"], 246);
    assert_eq!(ledger["owner"], "TwinWide942b");
    assert_eq!(ledger["run_s"], 26);
    assert_eq!(ledger["exit"], 0);
}

/// Exactly one native device entry in each image, and the manifest sizes/hashes match it.
#[test]
fn single_native_entry_and_full_elf_identity() {
    let m = manifest();
    assert_eq!(SYMBOLS.len(), 7);
    assert_eq!(
        m["seal"]["reports"].as_array().expect("reports").len(),
        SYMBOLS.len()
    );
    let (bundle, payloads) = Bundle::read(HXACO).expect("readable hxaco");
    let native = payloads
        .iter()
        .zip(bundle.entries.iter())
        .filter(|(_, e)| e.triple.contains(ARCH))
        .count();
    assert_eq!(native, 1, "exactly one native gfx942 entry");
    let elf = device_elf(HXACO);
    for r in m["seal"]["reports"].as_array().expect("reports") {
        assert_eq!(r["elf_bytes"].as_u64().unwrap(), elf.len() as u64);
        assert_eq!(r["elf_sha256"], hex(elf));
        assert_eq!(r["hxaco_bytes"].as_u64().unwrap(), HXACO.len() as u64);
    }
}

/// Exact HC-gates ABI as sealed for every r2..r8 entry: three pointers (W, x, y), then M, K,
/// scale and hc_act as 32-bit scalars, 40-byte segment.
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
                40,
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
