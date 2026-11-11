// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx1201 Route-A twin of the four
//! `lwmma_gemm_bf16_t2_m{1,2,3,4}_n1_multi` kernels (one module, one anchor object).
//! Every input is a committed file (`include_bytes!`); nothing is compiled or shelled out.
//! Regenerate with `scripts/pm-leviathan-twin.py --family lwmma-multi --arch gfx1201`.
//! Proof only: the runtime registry keeps loading the HIP source. The sealed image is the whole
//! `qwen4_lane_wmma.gfx1201.hip` device ELF (plain lane-WMMA entries included).

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco_many};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE: &[u8] = include_bytes!("../../../kernels/src/qwen4_lane_wmma.gfx1201.hip");
const ANCHOR: &[u8] = include_bytes!("../../../kernels/qwen4_lane_wmma_multi_hipcc_gfx1201.co");
const HXACO: &[u8] = include_bytes!("../../../kernels/qwen4_lane_wmma_multi_pm_gfx1201.hxaco");
const SHA_FILE: &str =
    include_str!("../../../kernels/qwen4_lane_wmma_multi_pm_gfx1201.hxaco.sha256");
const MANIFEST: &str = include_str!("../../../kernels/qwen4_lane_wmma_multi_pm_gfx1201.twin.json");

const SYMBOLS: [&str; 4] = [
    "lwmma_gemm_bf16_t2_m1_n1_multi",
    "lwmma_gemm_bf16_t2_m2_n1_multi",
    "lwmma_gemm_bf16_t2_m3_n1_multi",
    "lwmma_gemm_bf16_t2_m4_n1_multi",
];

const ARCH: Arch = Arch::Gfx1201;

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("manifest json")
}

fn reports(m: &Value) -> &Vec<Value> {
    m["seal"]["reports"].as_array().expect("seal.reports")
}

/// Explicit (non-hidden) kernargs `(offset, size, pointer)` of a seal report, in metadata order.
fn explicit(report: &Value) -> Vec<(u64, u64, bool)> {
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

#[test]
fn manifest_records_source_and_artifact_hashes() {
    let m = manifest();
    assert_eq!(m["arch"], "gfx1201");
    assert_eq!(m["module"], "qwen4_lane_wmma_multi_pm_gfx1201");
    assert_eq!(
        m["source"]["path"],
        "kernels/src/qwen4_lane_wmma.gfx1201.hip"
    );
    assert_eq!(
        m["source"]["sha256"],
        hex(SOURCE),
        "qwen4_lane_wmma.gfx1201.hip changed since the twin was generated; \
         rerun scripts/pm-leviathan-twin.py --family lwmma-multi"
    );
    assert_eq!(m["source"]["bytes"], SOURCE.len() as u64);
    assert_eq!(m["anchor"]["sha256"], hex(ANCHOR));
    assert_eq!(m["anchor"]["bytes"], ANCHOR.len() as u64);
    assert_eq!(m["hxaco"]["sha256"], hex(HXACO));
    assert_eq!(m["hxaco"]["bytes"], HXACO.len() as u64);
    assert_eq!(SHA_FILE.trim(), hex(HXACO), ".sha256 file");
    assert_eq!(
        SHA_FILE.trim().len(),
        64,
        "sha file must be bare lowercase hex"
    );
    assert!(
        m.get("math_evidence").is_none(),
        "no math evidence for this family"
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
    assert_eq!(
        flags,
        [
            "--genco",
            "--offload-arch=gfx1201",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6",
        ]
    );
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
fn manifest_symbols_and_reports_are_exactly_the_four() {
    let m = manifest();
    let syms: Vec<&str> = m["symbols"]
        .as_array()
        .expect("symbols")
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    assert_eq!(syms, SYMBOLS);
    let kernels: Vec<&str> = reports(&m)
        .iter()
        .map(|r| r["kernel"].as_str().unwrap())
        .collect();
    assert_eq!(kernels, SYMBOLS);
}

#[test]
fn every_symbol_seal_reproduces_the_hxaco_text_and_elf() {
    let m = manifest();
    let verified = verify_hxaco_many(HXACO, ARCH, &SYMBOLS).expect("verify_hxaco_many");
    assert_eq!(verified.len(), SYMBOLS.len());
    for ((sym, v), rec) in SYMBOLS.iter().zip(&verified).zip(reports(&m)) {
        let from_anchor =
            seal(ANCHOR, ARCH, sym, None).unwrap_or_else(|e| panic!("{sym}: seal(anchor): {e}"));
        assert_eq!(
            from_anchor.hxaco, HXACO,
            "{sym}: sealing the anchor must reproduce the hxaco"
        );
        assert_eq!(v.hxaco, HXACO, "{sym}");
        assert_eq!(from_anchor.elf_sha256, v.elf_sha256, "{sym}");
        assert_eq!(from_anchor.text_sha256, v.text_sha256, "{sym}");
        assert_eq!(v.hxaco_sha256, hex(HXACO), "{sym}");
        assert_eq!(v.scratch, 0, "{sym}: scratch");

        assert_eq!(rec["kernel"], *sym);
        assert_eq!(rec["arch"], "gfx1201", "{sym}");
        assert_eq!(rec["elf_sha256"], v.elf_sha256, "{sym}");
        assert_eq!(rec["text_sha256"], v.text_sha256, "{sym}");
        assert_eq!(rec["hxaco_sha256"], hex(HXACO), "{sym}");
        assert_eq!(rec["vgprs"], v.vgprs, "{sym}");
        assert_eq!(rec["sgprs"], v.sgprs, "{sym}");
        assert_eq!(rec["lds"], v.lds, "{sym}");
        assert_eq!(rec["scratch"], v.scratch, "{sym}");
        assert_eq!(
            rec["lift_identity"], "emit::checked ok; emitted bytes == input ELF",
            "{sym}"
        );
        assert_eq!(rec["hazards"]["blocking"], 0, "{sym}");
        assert_eq!(v.report["hazards"]["blocking"], 0, "{sym}");
        assert_eq!(rec["helpers"], v.report["helpers"], "{sym}");
    }
}

/// `W0..W3, Y0..Y3, X, partial, counters` (eleven u64 pointers at 0..88) then
/// `M0..M3, S0..S3, K, rows` (ten i32 at 88..128); see `LWMMA_GEMM_MULTI_ENTRY` in the source.
#[test]
fn kernarg_abi_is_twenty_one_explicit_args() {
    let want: Vec<(u64, u64, bool)> = (0..11)
        .map(|i| (8 * i, 8, true))
        .chain((0..10).map(|i| (88 + 4 * i, 4, false)))
        .collect();
    assert_eq!(want.len(), 21);
    assert_eq!(want[10], (80, 8, true));
    assert_eq!(want[11], (88, 4, false));
    assert_eq!(want[20], (124, 4, false));
    let m = manifest();
    let verified = verify_hxaco_many(HXACO, ARCH, &SYMBOLS).expect("verify");
    for ((sym, v), rec) in SYMBOLS.iter().zip(&verified).zip(reports(&m)) {
        assert_eq!(explicit(&v.report), want, "{sym} (fresh)");
        assert_eq!(explicit(rec), want, "{sym} (manifest)");
    }
    let src = std::str::from_utf8(SOURCE).expect("utf8 source");
    assert!(
        src.contains("int M3, int S0, int S1, int S2, int S3, int K, int rows) {"),
        "source kernel signature no longer matches the pinned ABI"
    );
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    for sym in SYMBOLS {
        let sealed = seal(ANCHOR, ARCH, sym, Some((ANCHOR, sym)))
            .unwrap_or_else(|e| panic!("{sym}: abi_ref seal: {e}"));
        assert_eq!(sealed.hxaco, HXACO, "{sym}");
        assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true, "{sym}");
        assert_eq!(sealed.report["abi"]["incumbent_symbol"], sym);
        assert_eq!(
            sealed.report["abi"]["incumbent_sha256"], sealed.elf_sha256,
            "{sym}"
        );
        assert!(seal(ANCHOR, ARCH, sym, Some((ANCHOR, "no_such_kernel"))).is_err());
    }
}

#[test]
fn module_is_refused_as_another_arch_or_symbol() {
    assert!(verify_hxaco_many(HXACO, Arch::Gfx1200, &SYMBOLS).is_err());
    assert!(verify_hxaco_many(HXACO, ARCH, &["lwmma_gemm_bf16_t2_m5_n1_multi"]).is_err());
}
