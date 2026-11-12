// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx942 QSA lanes Route-A twin.
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//!
//! The module holds seven entries; exactly FIVE f32 entries are sealed and certified. The two
//! FP8 (`*_fp8_fnuz_gfx942`) entries stay OPEN: they ship in the same ELF but are never
//! launched, are not in the sealed symbol set, and sealing them against the current anchor
//! must keep being refused (strict negative seals below). Nothing here claims seven.

use peacemaker_ir::envelope::Bundle;
use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{pinned_report, seal_many, verify_hxaco_many, KernelSpec};
use peacemaker_lift::bundle::BundleCodec;
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx942";
const SYMBOLS: [&str; 5] = [
    "qwen4_qsa_lanes_prologue_f32_gfx942",
    "qwen4_qsa_lanes_pool_f32_gfx942",
    "qwen4_qsa_lanes_scores_rows8_f32_gfx942",
    "qwen4_qsa_lanes_select_from_scores_gfx942",
    "qwen4_qsa_lanes_attention_f32_gfx942",
];
/// Present in the full ELF, NOT sealed, NOT launched: stay OPEN. `(symbol, exact refusal)`.
const OPEN_FP8: [(&str, &str); 2] = [
    (
        "qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942",
        "CDNA certification of `qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942`: qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942: dword 1209: generated instruction v_cmp_eq_f16 (ENC_VOP3 opcode 34, gfx942) has no hand-written model",
    ),
    (
        "qwen4_qsa_lanes_attention_fp8_fnuz_gfx942",
        "CDNA certification of `qwen4_qsa_lanes_attention_fp8_fnuz_gfx942`: v_fma_mix_f32 v4, v10, 1.0, v10 op_sel_hi:[1,1,1]: unverified encoding: unverified(xml-only: XML mnemonic V_MAD_MIX_F32 decoded as v_fma_mix_f32 v0, v0, v0, v0 op_sel_hi:[1,1,1])",
    ),
];
const SOURCE_SHA256: &str = "552ea9cb5af4f1bcb6efd2fabef98b9b99b51f10d3522c57beea5d845287550f";
const ANCHOR_SHA256: &str = "7d2fdcae51bc1f235692d3d171f80025607257ad79d2107428e019fa9a816f1f";
const HXACO_SHA256: &str = "a7f134b50b0360ca395f4e689c5cc081111b7163dde77ef9892ca23c3fee4578";
const ELF_SHA256: &str = "142d236f6b1f03146d052f7576723d1b5eb11f7237a4385bbe65e1bc3bb7015e";
const ELF_BYTES: usize = 75272;
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/qwen4_qsa_lanes_f32.gfx942.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_qsa_lanes_f32_gfx942_hipcc_gfx942.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/qwen4_qsa_lanes_f32_gfx942_pm_gfx942.hxaco");
const SHA_FILE: &str =
    include_str!("../../../kernels/qwen4_qsa_lanes_f32_gfx942_pm_gfx942.hxaco.sha256");
const MANIFEST: &str =
    include_str!("../../../kernels/qwen4_qsa_lanes_f32_gfx942_pm_gfx942.twin.json");

/// Per-symbol `(text_sha256, text_bytes, vgprs, sgprs, lds)` from the seal report.
const SEAL: [(&str, u64, u32, u32, u32); 5] = [
    (
        "7b44dfdf7b432786b854eae364b0f7127a8e5cb5c299e7bacd41d7263e3bd0ed",
        11224,
        12,
        60,
        1028,
    ),
    (
        "2318d3f0749f8e2d8ecd2432b411357e860d04e5759bdd4ffb205474aa325fd3",
        4640,
        12,
        32,
        1024,
    ),
    (
        "10480ec5edb27345ad86124d20d802a351a3ac855261b39cef84ab6cdbbea8f5",
        8012,
        52,
        44,
        16384,
    ),
    (
        "78c1eec46021b1b1c2dbbfb7c633731656db3bb35a21e5818569469f8b51d9b4",
        6104,
        44,
        59,
        12360,
    ),
    (
        "0ce5fb5785c16fc0d0a703610d40ad52945bcb02ef962f501d9f462ffd65cbde",
        7300,
        60,
        100,
        32,
    ),
];

/// `(offset, size, pointer)` of the explicit (non-hidden) kernargs.
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
fn immutable_source_anchor_and_twin_hashes_are_pinned() {
    assert_eq!(hex(SOURCE), SOURCE_SHA256, "frozen source");
    assert_eq!(hex(ANCHOR), ANCHOR_SHA256, "frozen anchor");
    assert_eq!(hex(HXACO), HXACO_SHA256, "twin hxaco");
    assert_eq!(SHA_FILE.split_whitespace().next(), Some(HXACO_SHA256));
    // Full ELF identity: the twin's device ELF is exactly the hipcc anchor's device ELF.
    let (twin, anchor) = (device_elf(HXACO), device_elf(ANCHOR));
    assert_eq!(twin, anchor, "twin device ELF != anchor device ELF");
    assert_eq!(twin.len(), ELF_BYTES);
    assert_eq!(hex(twin), ELF_SHA256);
}

#[test]
fn manifest_records_source_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(s(&m, &["arch"]), ARCH);
    assert_eq!(s(&m, &["module"]), "qwen4_qsa_lanes_f32_gfx942_pm_gfx942");
    let symbols: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|v| v.as_str().expect("symbol string"))
        .collect();
    assert_eq!(symbols, SYMBOLS, "only the five f32 entries are certified");
    for (open, _) in OPEN_FP8 {
        assert!(!symbols.contains(&open), "{open} must not be certified");
    }
    assert_eq!(
        s(&m, &["source", "path"]),
        "kernels/src/qwen4_qsa_lanes_f32.gfx942.hip"
    );
    assert_eq!(s(&m, &["source", "sha256"]), SOURCE_SHA256);
    assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "qwen4_qsa_lanes_f32_gfx942_hipcc_gfx942.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), ANCHOR_SHA256);
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(
        s(&m, &["hxaco", "file"]),
        "qwen4_qsa_lanes_f32_gfx942_pm_gfx942.hxaco"
    );
    assert_eq!(s(&m, &["hxaco", "sha256"]), HXACO_SHA256);
    assert_eq!(n(&m, &["hxaco", "bytes"]), HXACO.len() as u64);
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
    assert_eq!(
        m["source"]["owner_commit"],
        "c06ab5d626165652c21fee9f131b0dd1352109d4"
    );
    assert!(s(&m, &["compiler", "provenance"]).contains("does not recompile"));
}

/// Seal/verify evidence for the five certified entries, anchored to the full 7-symbol ELF.
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
        let (text_sha, text_bytes, vgprs, sgprs, lds) = SEAL[i];
        assert_eq!(a.hxaco, HXACO, "{symbol}: reseal must reproduce the hxaco");
        assert_eq!(a.elf_sha256, ELF_SHA256, "{symbol}");
        assert_eq!(t.elf_sha256, ELF_SHA256, "{symbol}");
        assert_eq!(a.text_sha256, t.text_sha256, "{symbol}");
        assert_eq!(t.text_sha256, text_sha, "{symbol}");
        assert_eq!(t.hxaco_sha256, HXACO_SHA256, "{symbol}");
        assert_eq!(a.report["abi"], t.report["abi"], "{symbol}: abi");
        assert_eq!(t.report["text_bytes"], text_bytes, "{symbol}");
        assert_eq!(t.vgprs, vgprs, "{symbol}");
        assert_eq!(t.sgprs, sgprs, "{symbol}");
        assert_eq!(t.lds, lds, "{symbol}");
        assert_eq!(t.scratch, 0, "{symbol}");
        assert_eq!(r["kernel"], *symbol);
        assert_eq!(r["arch"], ARCH);
        assert_eq!(r["elf_sha256"], ELF_SHA256, "{symbol}");
        assert_eq!(r["text_sha256"], text_sha, "{symbol}");
        assert_eq!(r["hxaco_sha256"], HXACO_SHA256, "{symbol}");
        assert_eq!(r["vgprs"], vgprs, "{symbol}");
        assert_eq!(r["sgprs"], sgprs, "{symbol}");
        assert_eq!(r["lds"], lds, "{symbol}");
        assert_eq!(r["scratch"], 0, "{symbol}");
        assert_eq!(r["wavefront_size"], 64, "{symbol}");
        // The module lifts all SEVEN entries; only five are sealed.
        assert_eq!(r["lifted_kernels"], 7, "{symbol}");
        assert_eq!(
            r["lift_identity"], "emit::checked ok; emitted bytes == input ELF",
            "{symbol}"
        );
    }
}

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
        assert_eq!(r["hazards"]["bypasses"], 0, "{symbol}");
        assert_eq!(
            r["hazards"]["ledger"]["memory"]["vmem_atomic"], 0,
            "{symbol}"
        );
        assert_eq!(
            pinned_report(&fresh[i].report)["hazards"],
            pinned_report(r)["hazards"],
            "{symbol}"
        );
        assert_eq!(fresh[i].report["abi"], r["abi"], "{symbol}");
        assert_eq!(
            fresh[i].report["lift_identity"], r["lift_identity"],
            "{symbol}"
        );
    }
}

/// Typed lift of anchor and twin: the module holds seven kernels; the five sealed ones
/// certify and re-emit exactly and carry no global atomics. The select kernel uses integer
/// LDS atomicAdd for its radix histogram, as pinned by source and exact text identity.
#[test]
fn typed_lift_certifies_the_five_sealed_kernels_and_has_no_global_atomics() {
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
        let mut want: Vec<String> = SYMBOLS
            .iter()
            .copied()
            .chain(OPEN_FP8.iter().map(|(s, _)| *s))
            .map(|s| s.to_string())
            .collect();
        want.sort();
        assert_eq!(names.len(), 7, "{label}: full module has seven symbols");
        assert_eq!(names, want, "{label}: kernel symbol set");
        for kernel in lifted
            .kernels
            .iter()
            .filter(|k| SYMBOLS.contains(&k.image.name.as_str()))
        {
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
                !text.iter().any(|l| l.contains("atomic")),
                "{label}: {name} contains a global atomic"
            );
        }
    }
}

/// The FP8 entries are never certified: sealing each against the current anchor is refused
/// with the exact observed error; the twin must not verify seven symbols.
#[test]
fn fp8_entries_stay_open_and_refuse_sealing() {
    for (symbol, expected) in OPEN_FP8 {
        let error = match seal_many(
            ANCHOR,
            arch(),
            &[KernelSpec {
                symbol,
                abi_ref: None,
            }],
        ) {
            Ok(_) => panic!("{symbol}: OPEN obligation unexpectedly discharged; review pin"),
            Err(error) => error,
        };
        assert_eq!(error, expected, "{symbol}");
    }
    // Mixing an OPEN entry into an otherwise valid five-symbol seal refuses the whole seal.
    let mut mixed = specs();
    mixed.push(KernelSpec {
        symbol: OPEN_FP8[0].0,
        abi_ref: None,
    });
    assert!(seal_many(ANCHOR, arch(), &mixed).is_err());
    let mut all: Vec<&str> = SYMBOLS.to_vec();
    all.extend(OPEN_FP8.iter().map(|(s, _)| *s));
    assert!(
        verify_hxaco_many(HXACO, arch(), &all).is_err(),
        "twin must not certify seven symbols"
    );
}

/// Hardware byte-oracle receipt bound to both exact input images and the shared device ELF.
/// Absent or unbound receipt fails closed.
#[test]
fn gpu_oracle_receipt_is_bound_to_these_images() {
    let m = manifest();
    let receipt = m.get("gpu_oracle").expect("hardware byte-oracle receipt");
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["target"], ARCH);
    assert_eq!(receipt["input_hxaco_sha256"], HXACO_SHA256);
    assert_eq!(receipt["hipcc_image_sha256"], ANCHOR_SHA256);
    assert_eq!(receipt["pm_device_elf_sha256"], ELF_SHA256);
    assert_eq!(receipt["hipcc_device_elf_sha256"], ELF_SHA256);
    assert_eq!(receipt["symbols"], serde_json::json!(SYMBOLS));
    assert_eq!(receipt["failed_cases"], 0);
    assert_eq!(receipt["exit_code"], 0);
    assert_eq!(receipt["case_count"], 46);
    assert_eq!(receipt["certified_f32_case_count"], 23);
    assert_eq!(receipt["steps"], 2);
    assert_eq!(receipt["controls_passed"], 58);
    let dispatch = &receipt["dispatch"];
    assert_eq!(dispatch["status"], "PASS");
    assert_eq!(dispatch["case_count"], 8);
    assert_eq!(dispatch["certified_f32_case_count"], 4);
    assert_eq!(dispatch["steps"], 3);
    assert_eq!(dispatch["lanes"], serde_json::json!([2, 8, 33, 64]));
    assert_eq!(dispatch["controls_passed"], 16);
    assert_eq!(dispatch["state_diff_bytes"], 0);
    assert_eq!(dispatch["sentinel_words_left"], 0);
    assert_eq!(dispatch["exit_code"], 0);
    assert!(dispatch["ledger"]["row"].as_u64().expect("dispatch ledger") > 0);
    let stdout = include_str!("fixtures/cdna/gfx942_qsa_lanes_oracle.txt");
    assert_eq!(receipt["stdout_sha256"], hex(stdout.as_bytes()));
    assert!(stdout.contains("qsa_lanes_gfx942_oracle: PASS"));
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("ok   f32/"))
            .count(),
        23
    );
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("ok     control:"))
            .count(),
        58
    );
    assert!(receipt["ledger"]["row"].as_u64().expect("ledger.row") > 0);
    for field in ["stdout_sha256", "oracle_binary_sha256"] {
        let hash = receipt[field].as_str().expect("receipt hash");
        assert_eq!(hash.len(), 64, "{field}");
        assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()), "{field}");
    }
    for r in m["seal"]["reports"].as_array().expect("seal.reports") {
        assert_eq!(receipt["pm_device_elf_sha256"], r["elf_sha256"]);
    }
}

/// Exact explicit kernarg layouts of each certified entry (hidden args excluded) and total
/// kernarg segment sizes, from the fresh lift and the recorded manifest alike.
#[test]
fn kernarg_abi_is_the_declared_layout_of_each_entry() {
    let p = |o| (o, 8, true);
    let i = |o| (o, 4, false);
    let v = |o, z| (o, z, false);
    let prologue: Layout = vec![
        p(0),
        p(8),
        p(16),
        p(24),
        i(32),
        i(36),
        i(40),
        i(44),
        i(48),
        i(52),
        v(56, 16),
        v(72, 1024),
    ];
    let pool: Layout = vec![p(0), p(8), i(16), i(20), v(24, 16), v(40, 1024)];
    let rows8: Layout = vec![p(0), p(8), i(16), i(20), i(24), i(28), i(32), v(36, 1024)];
    let select: Layout = rows8.clone();
    let attention: Layout = vec![
        p(0),
        i(8),
        i(12),
        i(16),
        i(20),
        i(24),
        i(28),
        i(32),
        i(36),
        v(40, 1024),
    ];
    let wants = [
        (&prologue, 1096u64),
        (&pool, 1320),
        (&rows8, 1060),
        (&select, 1060),
        (&attention, 1320),
    ];
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
            assert_eq!(
                report["abi"]["kernarg_segment_size"].as_u64().unwrap(),
                segment_size,
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
        assert_eq!(sealed.report["abi"]["incumbent_sha256"], ELF_SHA256);
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
