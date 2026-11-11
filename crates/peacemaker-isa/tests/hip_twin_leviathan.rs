// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed `lane_leviathan_verify` Route-A twins (gfx1200, gfx1201).
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-leviathan-twin.py`.
//!
//! One symbol holds the complete kernel: mode 0 (full row), mode 1 (select) and mode 2 (merged)
//! are `mode` kernarg branches of the same text, so sealing the symbol covers all three.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SYMBOL: &str = "lane_leviathan_verify";
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/lane_leviathan.hip");

struct Twin {
    arch: &'static str,
    anchor: &'static [u8],
    hxaco: &'static [u8],
    sha_file: &'static str,
    manifest: &'static str,
}

const TWINS: [Twin; 2] = [
    Twin {
        arch: "gfx1200",
        anchor: include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1200.co"),
        hxaco: include_bytes!("../../../kernels/lane_leviathan_pm_gfx1200.hxaco"),
        sha_file: include_str!("../../../kernels/lane_leviathan_pm_gfx1200.hxaco.sha256"),
        manifest: include_str!("../../../kernels/lane_leviathan_pm_gfx1200.twin.json"),
    },
    Twin {
        arch: "gfx1201",
        anchor: include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1201.co"),
        hxaco: include_bytes!("../../../kernels/lane_leviathan_pm_gfx1201.hxaco"),
        sha_file: include_str!("../../../kernels/lane_leviathan_pm_gfx1201.hxaco.sha256"),
        manifest: include_str!("../../../kernels/lane_leviathan_pm_gfx1201.twin.json"),
    },
];

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn arch_of(t: &Twin) -> Arch {
    t.arch
        .parse::<Arch>()
        .unwrap_or_else(|e| panic!("{} is not a known Arch: {e}", t.arch))
}

fn manifest(t: &Twin) -> Value {
    serde_json::from_str(t.manifest).unwrap_or_else(|e| panic!("{} manifest: {e}", t.arch))
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

#[test]
fn manifest_records_source_and_artifact_hashes() {
    for t in &TWINS {
        let m = manifest(t);
        assert_eq!(s(&m, &["arch"]), t.arch);
        assert_eq!(s(&m, &["symbol"]), SYMBOL);
        assert_eq!(s(&m, &["module"]), format!("lane_leviathan_pm_{}", t.arch));
        assert_eq!(s(&m, &["source", "path"]), "kernels/src/lane_leviathan.hip");
        assert_eq!(
            s(&m, &["source", "sha256"]),
            hex(SOURCE),
            "{}: lane_leviathan.hip changed since the twin was generated; rerun scripts/pm-leviathan-twin.py",
            t.arch
        );
        assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
        assert_eq!(s(&m, &["anchor", "sha256"]), hex(t.anchor), "{}", t.arch);
        assert_eq!(n(&m, &["anchor", "bytes"]), t.anchor.len() as u64);
        assert_eq!(s(&m, &["hxaco", "sha256"]), hex(t.hxaco), "{}", t.arch);
        assert_eq!(n(&m, &["hxaco", "bytes"]), t.hxaco.len() as u64);
        assert_eq!(t.sha_file.trim(), hex(t.hxaco), "{}: .sha256 file", t.arch);
    }
}

#[test]
fn manifest_flags_are_the_portable_core_recipe() {
    for t in &TWINS {
        let m = manifest(t);
        let flags: Vec<&str> = m["compiler"]["flags"]
            .as_array()
            .expect("compiler.flags")
            .iter()
            .map(|f| f.as_str().expect("flag string"))
            .collect();
        let want = [
            "--genco".to_string(),
            format!("--offload-arch={}", t.arch),
            "-O3".to_string(),
            "--no-offload-compress".to_string(),
            "-fuse-cuid=none".to_string(),
            "-mcode-object-version=6".to_string(),
        ];
        assert_eq!(flags, want.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(
            flags.iter().all(|f| !f.contains('/')),
            "{}: flags must carry no paths",
            t.arch
        );
        assert!(
            !m["compiler"]["version"]
                .as_array()
                .expect("version")
                .is_empty(),
            "{}: compiler provenance missing",
            t.arch
        );
    }
}

#[test]
fn math_paths_are_recorded_without_a_local_toolchain_requirement() {
    for t in &TWINS {
        let m = manifest(t);
        assert_eq!(m["math_evidence"]["status"], "recorded", "{}", t.arch);
        for name in ["expf", "powf"] {
            let f = &m["math_evidence"]["functions"][name];
            assert_eq!(f["found"], true, "{}: {name}", t.arch);
            assert!(!f["callees"].as_array().expect("callees").is_empty());
            assert!(!f["extract"].as_array().expect("IR extract").is_empty());
        }
        // This compiler's observed paths, not a requirement on the machine running
        // the test: the complete lowered instruction stream is already committed.
        assert_eq!(
            m["math_evidence"]["functions"]["expf"]["callees"],
            serde_json::json!(["llvm.exp.f32"])
        );
        assert_eq!(
            m["math_evidence"]["functions"]["powf"]["callees"],
            serde_json::json!(["__ocml_pow_f32"])
        );
    }
}

#[test]
fn source_has_the_seven_arg_kernel_with_all_three_modes() {
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    assert!(src.contains(&format!("void {SYMBOL}(")));
    assert!(src.contains("#define LLV_MODE_FULL 0"));
    assert!(src.contains("#define LLV_MODE_SELECT 1"));
    assert!(src.contains("#define LLV_MODE_MERGED 2"));
    assert!(src.contains("mode == LLV_MODE_SELECT"));
    assert!(src.contains("mode == LLV_MODE_MERGED"));
}

#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    for t in &TWINS {
        let arch = arch_of(t);
        let m = manifest(t);
        let from_anchor = seal(t.anchor, arch, SYMBOL, None)
            .unwrap_or_else(|e| panic!("{}: seal(anchor): {e}", t.arch));
        let from_twin = verify_hxaco(t.hxaco, arch, SYMBOL)
            .unwrap_or_else(|e| panic!("{}: verify_hxaco: {e}", t.arch));
        assert_eq!(
            from_anchor.hxaco, t.hxaco,
            "{}: sealing the anchor must reproduce the hxaco",
            t.arch
        );
        assert_eq!(from_anchor.elf_sha256, from_twin.elf_sha256);
        assert_eq!(from_anchor.text_sha256, from_twin.text_sha256);
        assert_eq!(from_twin.hxaco_sha256, hex(t.hxaco));
        // The manifest's seal report is the generated fact for these exact bytes.
        let r = &m["seal"]["report"];
        assert_eq!(r["kernel"], SYMBOL);
        assert_eq!(r["arch"], t.arch);
        assert_eq!(r["elf_sha256"], from_twin.elf_sha256);
        assert_eq!(r["text_sha256"], from_twin.text_sha256);
        assert_eq!(r["hxaco_sha256"], hex(t.hxaco));
        assert_eq!(r["vgprs"], from_twin.vgprs);
        assert_eq!(r["sgprs"], from_twin.sgprs);
        assert_eq!(r["lds"], from_twin.lds);
        assert_eq!(r["scratch"], from_twin.scratch);
        assert_eq!(
            r["lift_identity"],
            "emit::checked ok; emitted bytes == input ELF"
        );
        assert_eq!(r["hazards"]["blocking"], 0);
        assert_eq!(r["helpers"], from_twin.report["helpers"]);
        let helpers = r["helpers"]
            .as_array()
            .expect("out-of-line helper receipts");
        assert!(
            helpers
                .iter()
                .any(|h| h["symbol"].as_str().unwrap().contains("llv_tail")),
            "{}: probability exponential helper was not codec-checked",
            t.arch
        );
        for helper in helpers {
            assert_eq!(helper["coverage"], "typed instruction re-encode");
            assert!(helper["instructions"].as_u64().unwrap() > 0);
            assert!(helper["bytes"].as_u64().unwrap() > 0);
            assert_eq!(helper["stream_sha256"].as_str().unwrap().len(), 64);
        }
    }
}

/// Exact explicit kernarg layout of `lane_leviathan_verify`:
/// `region, lanes_off, rows_off, ptab, mode, cand_local, cand_peer`.
#[test]
fn kernarg_abi_is_the_seven_explicit_args() {
    // (offset, size, pointer)
    let want: [(u64, u64, bool); 7] = [
        (0, 8, true),
        (8, 4, false),
        (12, 4, false),
        (16, 8, true),
        (24, 4, false),
        (32, 8, true),
        (40, 8, true),
    ];
    for t in &TWINS {
        let sealed = verify_hxaco(t.hxaco, arch_of(t), SYMBOL).expect("verify");
        for (label, report) in [
            ("fresh", &sealed.report),
            ("manifest", &manifest(t)["seal"]["report"]),
        ] {
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
            assert_eq!(explicit, want, "{} ({label})", t.arch);
        }
    }
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    for t in &TWINS {
        let arch = arch_of(t);
        let sealed = seal(t.anchor, arch, SYMBOL, Some((t.anchor, SYMBOL)))
            .unwrap_or_else(|e| panic!("{}: abi_ref seal: {e}", t.arch));
        assert_eq!(sealed.hxaco, t.hxaco);
        assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true);
        assert_eq!(sealed.report["abi"]["incumbent_symbol"], SYMBOL);
        assert_eq!(sealed.report["abi"]["incumbent_sha256"], sealed.elf_sha256);
        // Same bytes, wrong incumbent symbol: refused.
        assert!(seal(t.anchor, arch, SYMBOL, Some((t.anchor, "no_such_kernel"))).is_err());
    }
}

#[test]
fn malformed_artifacts_and_wrong_arch_are_rejected() {
    for t in &TWINS {
        let arch = arch_of(t);
        // Truncated bundle.
        assert!(verify_hxaco(&t.hxaco[..t.hxaco.len() / 2], arch, SYMBOL).is_err());
        // Empty and non-bundle input.
        assert!(verify_hxaco(&[], arch, SYMBOL).is_err());
        assert!(verify_hxaco(b"not an offload bundle", arch, SYMBOL).is_err());
        // Corrupt the device ELF magic, rather than assuming an arbitrary bit flip
        // must be invalid (a different, valid instruction stream can also seal).
        let mut bad = t.hxaco.to_vec();
        let elf = bad
            .windows(4)
            .position(|w| w == b"\x7fELF")
            .expect("device ELF");
        bad[elf] = 0;
        assert!(verify_hxaco(&bad, arch, SYMBOL).is_err());
        // Unknown symbol.
        assert!(verify_hxaco(t.hxaco, arch, "no_such_kernel").is_err());
        // The other arches' sealing of this object must fail.
        for other in ["gfx1100", "gfx1151"] {
            let a = other.parse::<Arch>().expect("known arch");
            assert!(
                verify_hxaco(t.hxaco, a, SYMBOL).is_err(),
                "{} as {other}",
                t.arch
            );
        }
        // CDNA is not liftable.
        let cdna = "gfx942".parse::<Arch>().expect("known arch");
        assert!(seal(t.anchor, cdna, SYMBOL, None).is_err());
    }
    // The two gfx12 objects are not interchangeable.
    let (a, b) = (&TWINS[0], &TWINS[1]);
    assert!(verify_hxaco(a.hxaco, arch_of(b), SYMBOL).is_err());
    assert!(verify_hxaco(b.hxaco, arch_of(a), SYMBOL).is_err());
    // Unknown arch names do not parse.
    assert!("gfx9999".parse::<Arch>().is_err());
    assert!("".parse::<Arch>().is_err());
}

/// Audits the supplemental receipts in the manifests: they must be bound to the committed
/// object, keep M7 explicitly OPEN, and (gfx1201 only) record the real-GPU oracle result.
/// Regenerating a changed image drops stale receipts, which makes this test fail closed.
#[test]
fn supplemental_receipts_are_bound_to_the_committed_object() {
    for t in &TWINS {
        let m = manifest(t);
        let h = hex(t.hxaco);
        assert_eq!(
            s(&m, &["hxaco", "sha256"]),
            h,
            "{} manifest hxaco hash",
            t.arch
        );

        assert_eq!(
            s(&m, &["m7", "status"]),
            "OPEN",
            "{} m7 must stay OPEN",
            t.arch
        );
        assert_eq!(
            s(&m, &["m7", "input_hxaco_sha256"]),
            h,
            "{} m7 binding",
            t.arch
        );
        assert!(s(&m, &["m7", "command"]).contains("--ignored m7_reports_open_findings"));
        let sm = &m["m7"]["summary"];
        assert_eq!(sm["table"], t.arch);
        assert_eq!(sm["lift"], "byte-exact");
        assert_eq!(n(&m, &["m7", "summary", "instructions"]), 6502);
        let ob = &sm["obligations"];
        assert_eq!(ob["Definedness/definedness-entry"], 4);
        assert_eq!(ob["SrcReadTiming(DsStore)/wait-war-ds-store"], 160);
        assert_eq!(ob["SrcReadTiming(VmemStore)/wait-war-vmem-store"], 13);
        assert_eq!(ob["Unknown/scratch-in-use"], 1);
        assert_eq!(
            ob.as_object().map(|o| o.len()),
            Some(4),
            "{} unexpected obligation classes",
            t.arch
        );
        assert_eq!(n(&m, &["m7", "scratch_bytes_per_item"]), 272);
        let sites: Vec<(String, Vec<String>)> = m["m7"]["definedness_sites"]
            .as_array()
            .expect("definedness_sites array")
            .iter()
            .map(|x| {
                (
                    x["offset"].as_str().expect("offset").to_string(),
                    x["registers"]
                        .as_array()
                        .expect("registers")
                        .iter()
                        .map(|r| r.as_str().expect("reg").to_string())
                        .collect(),
                )
            })
            .collect();
        let want: [(&str, &[&str]); 4] = [
            ("0x6790", &["v7", "v8"]),
            ("0x6b8c", &["s24"]),
            ("0x6bac", &["s24"]),
            ("0x6bc4", &["s22"]),
        ];
        assert_eq!(sites.len(), want.len());
        for (got, (off, regs)) in sites.iter().zip(want) {
            assert_eq!(got.0, off);
            assert_eq!(got.1, regs);
        }

        if t.arch == "gfx1201" {
            assert_eq!(s(&m, &["gpu_oracle", "status"]), "PASS");
            assert_eq!(s(&m, &["gpu_oracle", "target"]), "gfx1201");
            assert_eq!(s(&m, &["gpu_oracle", "input_hxaco_sha256"]), h);
            assert_eq!(n(&m, &["gpu_oracle", "case_count"]), 360);
            assert_eq!(m["gpu_oracle"]["modes"], serde_json::json!([0, 1, 2]));
            assert_eq!(m["gpu_oracle"]["confounded"], false);
            assert_eq!(n(&m, &["gpu_oracle", "tests", "passed"]), 2);
        } else {
            assert!(
                m.get("gpu_oracle").is_none(),
                "{} has no hardware run",
                t.arch
            );
        }
    }
}

/// M7 on the metadata kernel in each committed bundle; the separate helper receipts
/// cover llv_tail's inlined probability body. Run with `--features toolchain`.
#[cfg(feature = "toolchain")]
#[test]
#[ignore = "diagnostic: original hipcc M7 obligations remain OPEN, not a twin-equivalence gate"]
fn m7_reports_open_findings_without_suppressing_them() {
    let mut refused = Vec::new();
    for t in &TWINS {
        let arch = arch_of(t);
        let path = std::path::PathBuf::from(format!(
            "{}/../../kernels/lane_leviathan_pm_{}.hxaco",
            env!("CARGO_MANIFEST_DIR"),
            t.arch
        ));
        match peacemaker_isa::pm_check::m7(&path, arch.name(), SYMBOL) {
            Ok(receipt) => eprintln!("M7 {}: {receipt}", t.arch),
            Err(error) => {
                eprintln!("M7 OPEN {}: {error}", t.arch);
                refused.push(t.arch);
            }
        }
    }
    assert!(
        refused.is_empty(),
        "M7 remains OPEN for {refused:?}; full twin ELF identity is a separate proof"
    );
}
