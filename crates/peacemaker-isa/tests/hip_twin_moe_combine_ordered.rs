// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed `moe_combine_ordered` Route-A twins (gfx1200, gfx1201).
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate the artifacts with `scripts/pm-moe-combine-ordered-twin.py`.
//!
//! One `.hxaco` per arch holds BOTH entries of `kernels/src/moe_combine_ordered.hip`:
//! `moe_down_combine_grouped_top10_bf16in_ordered` and its `_zinit` sibling.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{
    seal, seal_many, verify_hxaco, verify_hxaco_many, KernelSpec, Sealed,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SYMBOLS: [&str; 2] = [
    "moe_down_combine_grouped_top10_bf16in_ordered",
    "moe_down_combine_grouped_top10_bf16in_ordered_zinit",
];
const SOURCE: &[u8] = include_bytes!("../../../kernels/src/moe_combine_ordered.hip");

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
        anchor: include_bytes!("../../../kernels/moe_combine_ordered_hipcc_gfx1200.co"),
        hxaco: include_bytes!("../../../kernels/moe_combine_ordered_pm_gfx1200.hxaco"),
        sha_file: include_str!("../../../kernels/moe_combine_ordered_pm_gfx1200.hxaco.sha256"),
        manifest: include_str!("../../../kernels/moe_combine_ordered_pm_gfx1200.twin.json"),
    },
    Twin {
        arch: "gfx1201",
        anchor: include_bytes!("../../../kernels/moe_combine_ordered_hipcc_gfx1201.co"),
        hxaco: include_bytes!("../../../kernels/moe_combine_ordered_pm_gfx1201.hxaco"),
        sha_file: include_str!("../../../kernels/moe_combine_ordered_pm_gfx1201.hxaco.sha256"),
        manifest: include_str!("../../../kernels/moe_combine_ordered_pm_gfx1201.twin.json"),
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

/// Explicit (non-hidden) kernargs of a seal report, in metadata order: (offset, size, pointer).
fn explicit_of(report: &Value) -> Vec<(u64, u64, bool)> {
    report["abi"]["args"]
        .as_array()
        .expect("abi.args")
        .iter()
        .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
        .map(|a| {
            (
                a["offset"].as_u64().unwrap(),
                a["size"].as_u64().unwrap(),
                a["pointer"].as_bool().unwrap(),
            )
        })
        .collect()
}

fn explicit(s: &Sealed) -> Vec<(u64, u64, bool)> {
    explicit_of(&s.report)
}

#[test]
fn manifest_records_source_and_artifact_hashes() {
    for t in &TWINS {
        let m = manifest(t);
        assert_eq!(s(&m, &["arch"]), t.arch);
        assert_eq!(
            m["symbols"],
            serde_json::json!(SYMBOLS),
            "{}: symbol list",
            t.arch
        );
        assert_eq!(
            s(&m, &["module"]),
            format!("moe_combine_ordered_pm_{}", t.arch)
        );
        assert_eq!(
            s(&m, &["source", "path"]),
            "kernels/src/moe_combine_ordered.hip"
        );
        assert_eq!(
            s(&m, &["source", "sha256"]),
            hex(SOURCE),
            "{}: moe_combine_ordered.hip changed since the twin was generated; \
             rerun scripts/pm-moe-combine-ordered-twin.py",
            t.arch
        );
        assert_eq!(n(&m, &["source", "bytes"]), SOURCE.len() as u64);
        assert_eq!(s(&m, &["anchor", "sha256"]), hex(t.anchor), "{}", t.arch);
        assert_eq!(n(&m, &["anchor", "bytes"]), t.anchor.len() as u64);
        assert_eq!(s(&m, &["hxaco", "sha256"]), hex(t.hxaco), "{}", t.arch);
        assert_eq!(n(&m, &["hxaco", "bytes"]), t.hxaco.len() as u64);
        assert_eq!(t.sha_file.trim(), hex(t.hxaco), "{}: .sha256 file", t.arch);
        assert_eq!(t.sha_file.trim().len(), 64, "{}: bare sha file", t.arch);
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

/// The source defines exactly the two sealed entries, each `extern "C"` with block 64.
#[test]
fn source_has_both_entries_with_block_64() {
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    for sym in SYMBOLS {
        assert!(
            src.contains(&format!("void {sym}(")) || src.contains(&format!("void\n{sym}(")),
            "source lacks entry {sym}"
        );
    }
    assert_eq!(
        src.matches("extern \"C\" __global__").count(),
        SYMBOLS.len(),
        "unexpected number of kernel entries"
    );
    assert_eq!(
        src.matches("__launch_bounds__(64)").count(),
        SYMBOLS.len(),
        "every entry must be launch_bounds(64)"
    );
}

#[test]
fn anchor_and_sealed_twin_are_the_same_elf_and_text() {
    for t in &TWINS {
        let arch = arch_of(t);
        let m = manifest(t);
        let reports = m["seal"]["reports"].as_array().expect("seal.reports");
        assert_eq!(reports.len(), SYMBOLS.len(), "{}", t.arch);
        let specs: Vec<KernelSpec<'_>> = SYMBOLS
            .iter()
            .map(|sym| KernelSpec {
                symbol: sym,
                abi_ref: None,
            })
            .collect();
        let many = seal_many(t.anchor, arch, &specs)
            .unwrap_or_else(|e| panic!("{}: seal_many(anchor): {e}", t.arch));
        let verified = verify_hxaco_many(t.hxaco, arch, &SYMBOLS)
            .unwrap_or_else(|e| panic!("{}: verify_hxaco_many: {e}", t.arch));
        assert_eq!(many.len(), SYMBOLS.len());
        assert_eq!(verified.len(), SYMBOLS.len());
        let mut texts = Vec::new();
        for (i, sym) in SYMBOLS.iter().enumerate() {
            let (a, v, r) = (&many[i], &verified[i], &reports[i]);
            assert_eq!(
                a.hxaco, t.hxaco,
                "{} {sym}: sealing the anchor must reproduce the hxaco",
                t.arch
            );
            assert_eq!(a.elf_sha256, v.elf_sha256, "{} {sym}", t.arch);
            assert_eq!(a.text_sha256, v.text_sha256, "{} {sym}", t.arch);
            assert_eq!(v.hxaco_sha256, hex(t.hxaco), "{} {sym}", t.arch);
            assert_eq!(v.scratch, 0, "{} {sym}: scratch", t.arch);
            // The manifest's seal report is the generated fact for these exact bytes.
            assert_eq!(r["kernel"], *sym);
            assert_eq!(r["arch"], t.arch);
            assert_eq!(r["elf_sha256"], v.elf_sha256);
            assert_eq!(r["text_sha256"], v.text_sha256);
            assert_eq!(r["hxaco_sha256"], hex(t.hxaco));
            assert_eq!(r["vgprs"], v.vgprs);
            assert_eq!(r["sgprs"], v.sgprs);
            assert_eq!(r["lds"], v.lds);
            assert_eq!(r["scratch"], v.scratch);
            assert_eq!(
                r["lift_identity"],
                "emit::checked ok; emitted bytes == input ELF"
            );
            assert_eq!(r["hazards"]["blocking"], 0, "{} {sym}", t.arch);
            assert_eq!(v.report["hazards"]["checked"], true);
            assert_eq!(v.report["hazards"]["blocking"], 0);
            assert_eq!(r["wavefront_size"], 32, "{} {sym}", t.arch);
            assert_eq!(v.report["wavefront_size"], 32, "{} {sym}", t.arch);
            texts.push(v.text_sha256.clone());
        }
        // The `_zinit` entry is a different instruction stream (read-modify-write vs write-only).
        assert_ne!(texts[0], texts[1], "{}: entries must differ", t.arch);
    }
}

/// Explicit kernarg layout of both entries (identical signature):
/// `grouped_down, inverse_perm, topk_indices, topk_weights, residual, owner` (6 pointers, 0..48)
/// then `hidden, grouped_rows, tokens, n_experts, rank` (5 x i32, 48..68): 68 explicit bytes.
#[test]
fn kernarg_abi_is_the_68_byte_explicit_layout() {
    let want: [(u64, u64, bool); 11] = [
        (0, 8, true),
        (8, 8, true),
        (16, 8, true),
        (24, 8, true),
        (32, 8, true),
        (40, 8, true),
        (48, 4, false),
        (52, 4, false),
        (56, 4, false),
        (60, 4, false),
        (64, 4, false),
    ];
    for t in &TWINS {
        let m = manifest(t);
        for (i, sym) in SYMBOLS.iter().enumerate() {
            let sealed = verify_hxaco(t.hxaco, arch_of(t), sym).expect("verify");
            assert_eq!(explicit(&sealed), want, "{} {sym} (fresh)", t.arch);
            assert_eq!(
                explicit_of(&m["seal"]["reports"][i]),
                want,
                "{} {sym} (manifest)",
                t.arch
            );
            let last = want[want.len() - 1];
            assert_eq!(last.0 + last.1, 68, "explicit ABI is 68 bytes");
        }
    }
}

/// Block size 64 (`__launch_bounds__(64)`) from the anchor's code-object metadata, recorded by
/// the generator for each sealed symbol; wave32 so two waves per workgroup.
#[test]
fn workgroup_size_is_64_for_both_entries() {
    for t in &TWINS {
        let m = manifest(t);
        for sym in SYMBOLS {
            assert_eq!(
                n(
                    &m,
                    &["code_object_metadata", sym, "max_flat_workgroup_size"]
                ),
                64,
                "{} {sym}",
                t.arch
            );
        }
    }
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    for t in &TWINS {
        let arch = arch_of(t);
        for sym in SYMBOLS {
            let sealed = seal(t.anchor, arch, sym, Some((t.anchor, sym)))
                .unwrap_or_else(|e| panic!("{} {sym}: abi_ref seal: {e}", t.arch));
            assert_eq!(sealed.hxaco, t.hxaco, "{} {sym}", t.arch);
            assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true);
            assert_eq!(sealed.report["abi"]["incumbent_symbol"], sym);
            assert_eq!(sealed.report["abi"]["incumbent_sha256"], sealed.elf_sha256);
            // Same bytes, wrong incumbent symbol: refused.
            assert!(seal(t.anchor, arch, sym, Some((t.anchor, "no_such_kernel"))).is_err());
        }
        // Each entry's kernarg layout equals its sibling's (the launch is interchangeable).
        assert!(seal(t.anchor, arch, SYMBOLS[1], Some((t.anchor, SYMBOLS[0]))).is_ok());
        assert!(seal(t.anchor, arch, SYMBOLS[0], Some((t.anchor, SYMBOLS[1]))).is_ok());
    }
}

#[test]
fn malformed_artifacts_wrong_arch_and_missing_symbols_are_rejected() {
    for t in &TWINS {
        let arch = arch_of(t);
        for sym in SYMBOLS {
            // Truncated bundle.
            assert!(verify_hxaco(&t.hxaco[..t.hxaco.len() / 2], arch, sym).is_err());
            // Empty and non-bundle input.
            assert!(verify_hxaco(&[], arch, sym).is_err());
            assert!(verify_hxaco(b"not an offload bundle", arch, sym).is_err());
            // Corrupt the device ELF magic.
            let mut bad = t.hxaco.to_vec();
            let elf = bad
                .windows(4)
                .position(|w| w == b"\x7fELF")
                .expect("device ELF");
            bad[elf] = 0;
            assert!(verify_hxaco(&bad, arch, sym).is_err());
            // The other arches' sealing of this object must fail.
            for other in ["gfx1100", "gfx1151"] {
                let a = other.parse::<Arch>().expect("known arch");
                assert!(
                    verify_hxaco(t.hxaco, a, sym).is_err(),
                    "{} {sym} as {other}",
                    t.arch
                );
            }
            // CDNA is not liftable.
            let cdna = "gfx942".parse::<Arch>().expect("known arch");
            assert!(seal(t.anchor, cdna, sym, None).is_err());
        }
        // Unknown symbols, and a symbol prefix of a real entry, are not entries.
        assert!(verify_hxaco(t.hxaco, arch, "no_such_kernel").is_err());
        assert!(verify_hxaco(t.hxaco, arch, "moe_down_combine_grouped_top10_bf16in").is_err());
        // Duplicate symbols and an empty list are refused by seal_many.
        let dup = [
            KernelSpec {
                symbol: SYMBOLS[0],
                abi_ref: None,
            },
            KernelSpec {
                symbol: SYMBOLS[0],
                abi_ref: None,
            },
        ];
        assert!(seal_many(t.anchor, arch, &dup).is_err());
        assert!(seal_many(t.anchor, arch, &[]).is_err());
    }
    // The two gfx12 objects are not interchangeable.
    let (a, b) = (&TWINS[0], &TWINS[1]);
    for sym in SYMBOLS {
        assert!(verify_hxaco(a.hxaco, arch_of(b), sym).is_err());
        assert!(verify_hxaco(b.hxaco, arch_of(a), sym).is_err());
    }
    assert_ne!(hex(a.hxaco), hex(b.hxaco));
    // Unknown arch names do not parse.
    assert!("gfx9999".parse::<Arch>().is_err());
    assert!("".parse::<Arch>().is_err());
}
