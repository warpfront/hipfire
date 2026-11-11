// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed F4 `gemm_mq6sym_i8_set_*_x4_gfx1201` Route-A twin.
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-gemm-mq6sym-i8-twin.py`.
//!
//! The twin is proof only (dispatch keeps the registry HIP source). The sealed image is the
//! whole `GEMM_MQ6SYM_I8_GFX1201_SRC` device ELF (legacy entries included), so the four x4 entries
//! are sealed from one byte-identical object.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco};
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: &str = "gfx1201";
const MODULE: &str = "gemm_mq6sym_i8_pm_gfx1201";
const SYMBOLS: [&str; 4] = [
    "gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
    "gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
    "gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
];
const PRELUDE_DEFINE: &str = "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n";
const PRELUDE: &[u8] = include_bytes!("../../../kernels/src/block_i8_128_quant.hip");
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/gemm_mq6sym_i8.gfx1201.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/gemm_mq6sym_i8_hipcc_gfx1201.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco");
const SHA_FILE: &str = include_str!("../../../kernels/gemm_mq6sym_i8_pm_gfx1201.hxaco.sha256");
const MANIFEST: &str = include_str!("../../../kernels/gemm_mq6sym_i8_pm_gfx1201.twin.json");

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn arch() -> Arch {
    ARCH.parse::<Arch>().expect("gfx1201 is a known Arch")
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("manifest json")
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

fn report_for<'a>(m: &'a Value, sym: &str) -> &'a Value {
    m["seal"]["reports"]
        .as_array()
        .expect("seal.reports")
        .iter()
        .find(|r| r["kernel"] == sym)
        .unwrap_or_else(|| panic!("no seal report for {sym}"))
}

#[test]
fn manifest_records_source_prelude_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(s(&m, &["arch"]), ARCH);
    assert_eq!(s(&m, &["module"]), MODULE);
    assert_eq!(m["symbols"], serde_json::json!(SYMBOLS), "manifest symbols");
    assert_eq!(
        s(&m, &["source", "path"]),
        "kernels/src/gemm_mq6sym_i8.gfx1201.hip"
    );
    assert_eq!(
        s(&m, &["source", "sha256"]),
        hex(SOURCE),
        "gemm_mq6sym_i8.gfx1201.hip changed since the twin was generated; rerun scripts/pm-gemm-mq6sym-i8-twin.py"
    );
    assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
    assert_eq!(
        s(&m, &["prelude", "sha256"]),
        hex(PRELUDE),
        "block_i8_128_quant.hip changed since the twin was generated; rerun scripts/pm-gemm-mq6sym-i8-twin.py"
    );
    assert_eq!(n(&m, &["prelude", "bytes"]), PRELUDE.len() as u64);
    assert_eq!(
        s(&m, &["prelude", "define"]),
        PRELUDE_DEFINE.trim_end_matches('\n')
    );
    let mut combined = PRELUDE_DEFINE.as_bytes().to_vec();
    combined.extend_from_slice(PRELUDE);
    combined.extend_from_slice(SOURCE);
    assert_eq!(s(&m, &["prelude", "combined_sha256"]), hex(&combined));
    assert_eq!(n(&m, &["prelude", "combined_bytes"]), combined.len() as u64);
    assert_eq!(
        s(&m, &["anchor", "file"]),
        "gemm_mq6sym_i8_hipcc_gfx1201.co"
    );
    assert_eq!(s(&m, &["anchor", "sha256"]), hex(ANCHOR));
    assert_eq!(n(&m, &["anchor", "bytes"]), ANCHOR.len() as u64);
    assert_eq!(s(&m, &["hxaco", "file"]), format!("{MODULE}.hxaco"));
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
        "--offload-arch=gfx1201",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
    ];
    assert_eq!(flags, want);
    assert!(flags.iter().all(|f| !f.contains('/')));
    assert!(!m["compiler"]["version"]
        .as_array()
        .expect("version")
        .is_empty());
}

#[test]
fn source_has_the_four_x4_entries() {
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    for sym in SYMBOLS {
        assert!(
            src.contains(&format!("SYM_X4({sym},")),
            "{sym} not instantiated in source"
        );
    }
}

#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    let m = manifest();
    for sym in SYMBOLS {
        let from_anchor =
            seal(ANCHOR, arch(), sym, None).unwrap_or_else(|e| panic!("{sym}: seal(anchor): {e}"));
        let from_twin =
            verify_hxaco(HXACO, arch(), sym).unwrap_or_else(|e| panic!("{sym}: verify_hxaco: {e}"));
        assert_eq!(from_anchor.elf_sha256, from_twin.elf_sha256, "{sym}");
        assert_eq!(from_anchor.text_sha256, from_twin.text_sha256, "{sym}");
        assert_eq!(from_twin.hxaco_sha256, hex(HXACO), "{sym}");
        let r = report_for(&m, sym);
        assert_eq!(r["arch"], ARCH);
        assert_eq!(r["elf_sha256"], from_twin.elf_sha256, "{sym}");
        assert_eq!(r["text_sha256"], from_twin.text_sha256, "{sym}");
        assert_eq!(r["hxaco_sha256"], hex(HXACO), "{sym}");
        assert_eq!(r["vgprs"], from_twin.vgprs, "{sym}");
        assert_eq!(r["sgprs"], from_twin.sgprs, "{sym}");
        assert_eq!(r["lds"], from_twin.lds, "{sym}");
        assert_eq!(r["scratch"], from_twin.scratch, "{sym}");
        assert_eq!(
            r["lift_identity"], "emit::checked ok; emitted bytes == input ELF",
            "{sym}"
        );
        assert_eq!(r["hazards"]["blocking"], 0, "{sym}");
    }
    // One image holds all four entries: they share the single sealed ELF.
    let elfs: Vec<String> = SYMBOLS
        .iter()
        .map(|s| verify_hxaco(HXACO, arch(), s).unwrap().elf_sha256)
        .collect();
    assert!(elfs.windows(2).all(|w| w[0] == w[1]));
}

/// Exact explicit kernarg layout of every x4 entry (96 bytes, no hidden args):
/// `W0..W3 Y0..Y3 Xq` (nine pointers, offsets 0..72) then `M0..M3 K N` (six u32, 72..96).
#[test]
fn kernarg_abi_is_the_96_byte_explicit_layout() {
    let mut want: Vec<(u64, u64, bool)> = (0..9).map(|i| (i * 8, 8, true)).collect();
    want.extend((0..6).map(|i| (72 + i * 4, 4, false)));
    let m = manifest();
    for sym in SYMBOLS {
        let sealed = verify_hxaco(HXACO, arch(), sym).expect("verify");
        for (label, report) in [("fresh", &sealed.report), ("manifest", report_for(&m, sym))] {
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
            assert_eq!(explicit, want, "{sym} ({label})");
            assert!(
                args.iter()
                    .all(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_")),
                "{sym} ({label}): x4 entries carry no hidden kernargs"
            );
            // Only the generation-time seal (--abi-ref) compares against the incumbent.
            assert_eq!(
                report["abi"]["compared_to_incumbent"],
                label == "manifest",
                "{sym} ({label})"
            );
        }
    }
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    for sym in SYMBOLS {
        let sealed = seal(ANCHOR, arch(), sym, Some((ANCHOR, sym)))
            .unwrap_or_else(|e| panic!("{sym}: abi_ref seal: {e}"));
        assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true);
        assert_eq!(sealed.report["abi"]["incumbent_symbol"], sym);
        assert_eq!(sealed.report["abi"]["incumbent_sha256"], sealed.elf_sha256);
        assert!(seal(ANCHOR, arch(), sym, Some((ANCHOR, "no_such_kernel"))).is_err());
    }
}

#[test]
fn malformed_artifacts_wrong_symbol_and_wrong_arch_are_rejected() {
    let sym = SYMBOLS[0];
    assert!(verify_hxaco(&HXACO[..HXACO.len() / 2], arch(), sym).is_err());
    assert!(verify_hxaco(&[], arch(), sym).is_err());
    assert!(verify_hxaco(b"not an offload bundle", arch(), sym).is_err());
    let mut bad = HXACO.to_vec();
    let elf = bad
        .windows(4)
        .position(|w| w == b"\x7fELF")
        .expect("device ELF");
    bad[elf] = 0;
    assert!(verify_hxaco(&bad, arch(), sym).is_err());
    assert!(verify_hxaco(HXACO, arch(), "no_such_kernel").is_err());
    // Wrong architecture bundles are refused.
    for other in ["gfx1100", "gfx1151", "gfx1200"] {
        let a = other.parse::<Arch>().expect("known arch");
        assert!(verify_hxaco(HXACO, a, sym).is_err(), "as {other}");
    }
    let cdna = "gfx942".parse::<Arch>().expect("known arch");
    assert!(seal(ANCHOR, cdna, sym, None).is_err());
    assert!("gfx9999".parse::<Arch>().is_err());
}

/// Proof boundary: byte identity and zero blocking hip_twin hazards are proven; the strict
/// ledger DS-store-source WAR obligations of the original LLVM code are not.
#[test]
fn strict_cfg_ledger_boundary_stays_open() {
    let m = manifest();
    let b = &m["strict_cfg_ledger"];
    assert_eq!(b["status"], "OPEN");
    assert_eq!(b["proven"], false);
    assert!(s(&m, &["strict_cfg_ledger", "scope"]).contains("UNPROVEN"));
    assert_eq!(
        s(&m, &["strict_cfg_ledger", "evidence"]),
        "crates/peacemaker-isa/tests/ledger_cfg_pins/allowlist.rs"
    );
}
