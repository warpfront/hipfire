// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only checks of the committed gfx1201 Route-A twins of the X8 wide-verify MoE glue entries:
//! `moe_scatter_fused_top10_pmask` + `moe_route_scatter_top10_wide_f32` (module `moe_route_scatter_wide`) and
//! `qwen4_moe_ep_a8_gate_up_rot_planar_nt{1,2}` (module `qwen4_moe_ep_a8_rot_gfx12`). Every input is a
//! committed file (`include_bytes!`); nothing is compiled or shelled out. Regenerate with
//! `scripts/pm-leviathan-twin.py --family moe-route-scatter-wide|qwen4-moe-ep-a8-rot --arch gfx1201`.
//! Proof only: the runtime registry keeps loading the HIP source. Each sealed image is the whole device
//! ELF of the exact concatenated runtime module source (`module_source.parts`).

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{seal, verify_hxaco_many};
use serde_json::Value;
use sha2::{Digest, Sha256};

const ARCH: Arch = Arch::Gfx1201;
const DEFINE: &str = "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n";

struct Twin {
    module: &'static str,
    prefix: &'static str,
    family: &'static str,
    symbols: &'static [&'static str],
    define: &'static str,
    parts: &'static [(&'static str, &'static [u8])],
    anchor: &'static [u8],
    hxaco: &'static [u8],
    sha_file: &'static str,
    manifest: &'static str,
}

const WIDE_SYMBOLS: [&str; 2] = [
    "moe_scatter_fused_top10_pmask",
    "moe_route_scatter_top10_wide_f32",
];
const ROT_SYMBOLS: [&str; 2] = [
    "qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
    "qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
];

const WIDE_PARTS: [(&str, &[u8]); 3] = [
    (
        "kernels/src/moe_router_softmax_top10_f32_fast.hip",
        include_bytes!("../../../kernels/src/moe_router_softmax_top10_f32_fast.hip"),
    ),
    (
        "kernels/src/qwen4_moe_scatter_fused_k8.hip",
        include_bytes!("../../../kernels/src/qwen4_moe_scatter_fused_k8.hip"),
    ),
    (
        "kernels/src/moe_route_scatter_wide.gfx1201.hip",
        include_bytes!("../../../kernels/src/moe_route_scatter_wide.gfx1201.hip"),
    ),
];
const ROT_PARTS: [(&str, &[u8]); 3] = [
    (
        "kernels/src/block_i8_128_quant.hip",
        include_bytes!("../../../kernels/src/block_i8_128_quant.hip"),
    ),
    (
        "kernels/src/qwen4_moe_ep_a8.gfx12.hip",
        include_bytes!("../../../kernels/src/qwen4_moe_ep_a8.gfx12.hip"),
    ),
    (
        "kernels/src/qwen4_moe_ep_a8_rot.gfx12.hip",
        include_bytes!("../../../kernels/src/qwen4_moe_ep_a8_rot.gfx12.hip"),
    ),
];

fn twins() -> [Twin; 2] {
    [
        Twin {
            module: "moe_route_scatter_wide_pm_gfx1201",
            prefix: "moe_route_scatter_wide",
            family: "moe-route-scatter-wide",
            symbols: &WIDE_SYMBOLS,
            define: "",
            parts: &WIDE_PARTS,
            anchor: include_bytes!("../../../kernels/moe_route_scatter_wide_hipcc_gfx1201.co"),
            hxaco: include_bytes!("../../../kernels/moe_route_scatter_wide_pm_gfx1201.hxaco"),
            sha_file: include_str!(
                "../../../kernels/moe_route_scatter_wide_pm_gfx1201.hxaco.sha256"
            ),
            manifest: include_str!("../../../kernels/moe_route_scatter_wide_pm_gfx1201.twin.json"),
        },
        Twin {
            module: "qwen4_moe_ep_a8_rot_pm_gfx1201",
            prefix: "qwen4_moe_ep_a8_rot",
            family: "qwen4-moe-ep-a8-rot",
            symbols: &ROT_SYMBOLS,
            define: DEFINE,
            parts: &ROT_PARTS,
            anchor: include_bytes!("../../../kernels/qwen4_moe_ep_a8_rot_hipcc_gfx1201.co"),
            hxaco: include_bytes!("../../../kernels/qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco"),
            sha_file: include_str!("../../../kernels/qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco.sha256"),
            manifest: include_str!("../../../kernels/qwen4_moe_ep_a8_rot_pm_gfx1201.twin.json"),
        },
    ]
}

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

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
fn manifests_record_sources_and_artifact_hashes() {
    for t in twins() {
        let m: Value = serde_json::from_str(t.manifest).expect("manifest json");
        let id = t.module;
        assert_eq!(m["arch"], "gfx1201", "{id}");
        assert_eq!(m["module"], id);
        let (last_path, last) = t.parts[t.parts.len() - 1];
        assert_eq!(m["source"]["path"], last_path, "{id}");
        assert_eq!(m["source"]["sha256"], hex(last), "{id}");
        let mut combined = t.define.as_bytes().to_vec();
        for (_, b) in t.parts {
            combined.extend_from_slice(b);
        }
        let ms = &m["module_source"];
        assert_eq!(ms["define"], t.define.trim_end(), "{id}");
        assert_eq!(
            ms["combined_sha256"],
            hex(&combined),
            "{id}: a source part changed since the twin was generated; \
             rerun scripts/pm-leviathan-twin.py --family {}",
            t.family
        );
        assert_eq!(ms["combined_bytes"], combined.len() as u64, "{id}");
        let parts = ms["parts"].as_array().expect("parts");
        assert_eq!(parts.len(), t.parts.len(), "{id}");
        for (rec, (path, b)) in parts.iter().zip(t.parts) {
            assert_eq!(rec["path"], *path, "{id}");
            assert_eq!(rec["sha256"], hex(b), "{id}: {path}");
            assert_eq!(rec["bytes"], b.len() as u64, "{id}: {path}");
        }
        assert_eq!(
            m["anchor"]["file"],
            format!("{}_hipcc_gfx1201.co", t.prefix)
        );
        assert_eq!(m["anchor"]["sha256"], hex(t.anchor), "{id}");
        assert_eq!(m["anchor"]["bytes"], t.anchor.len() as u64, "{id}");
        assert_eq!(m["hxaco"]["sha256"], hex(t.hxaco), "{id}");
        assert_eq!(m["hxaco"]["bytes"], t.hxaco.len() as u64, "{id}");
        assert_eq!(t.sha_file.trim(), hex(t.hxaco), "{id}: .sha256 file");
        assert!(m.get("math_evidence").is_none(), "{id}");
    }
}

#[test]
fn manifest_flags_are_the_portable_core_recipe() {
    for t in twins() {
        let m: Value = serde_json::from_str(t.manifest).unwrap();
        let flags: Vec<&str> = m["compiler"]["flags"]
            .as_array()
            .expect("compiler.flags")
            .iter()
            .map(|f| f.as_str().unwrap())
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
            ],
            "{}",
            t.module
        );
        let text = t.manifest;
        assert!(
            !text.contains("/home/") && !text.contains("/opt/") && !text.contains("/tmp"),
            "{}: manifest carries a host path",
            t.module
        );
        assert!(!m["compiler"]["version"].as_array().unwrap().is_empty());
    }
}

#[test]
fn manifest_symbols_and_reports_are_exactly_the_requested_entries() {
    for t in twins() {
        let m: Value = serde_json::from_str(t.manifest).unwrap();
        let syms: Vec<&str> = m["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert_eq!(syms, t.symbols, "{}", t.module);
        let kernels: Vec<&str> = m["seal"]["reports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["kernel"].as_str().unwrap())
            .collect();
        assert_eq!(kernels, t.symbols, "{}", t.module);
    }
}

#[test]
fn every_symbol_seal_reproduces_the_hxaco_text_and_elf() {
    for t in twins() {
        let m: Value = serde_json::from_str(t.manifest).unwrap();
        let reports = m["seal"]["reports"].as_array().unwrap();
        let verified = verify_hxaco_many(t.hxaco, ARCH, t.symbols).expect("verify_hxaco_many");
        assert_eq!(verified.len(), t.symbols.len());
        for ((sym, v), rec) in t.symbols.iter().zip(&verified).zip(reports) {
            let from_anchor = seal(t.anchor, ARCH, sym, None)
                .unwrap_or_else(|e| panic!("{sym}: seal(anchor): {e}"));
            assert_eq!(
                from_anchor.hxaco, t.hxaco,
                "{sym}: sealing the anchor must reproduce the hxaco"
            );
            assert_eq!(v.hxaco, t.hxaco, "{sym}");
            assert_eq!(from_anchor.elf_sha256, v.elf_sha256, "{sym}");
            assert_eq!(from_anchor.text_sha256, v.text_sha256, "{sym}");
            assert_eq!(v.hxaco_sha256, hex(t.hxaco), "{sym}");
            assert_eq!(v.scratch, 0, "{sym}: scratch");
            assert_eq!(rec["kernel"], *sym);
            assert_eq!(rec["arch"], "gfx1201", "{sym}");
            assert_eq!(rec["elf_sha256"], v.elf_sha256, "{sym}");
            assert_eq!(rec["text_sha256"], v.text_sha256, "{sym}");
            assert_eq!(rec["hxaco_sha256"], hex(t.hxaco), "{sym}");
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
            let args = explicit(&v.report);
            assert!(!args.is_empty(), "{sym}: no explicit kernargs");
            assert_eq!(explicit(rec), args, "{sym}: manifest vs fresh kernarg ABI");
        }
    }
}

#[test]
fn abi_ref_roundtrip_against_the_anchor() {
    for t in twins() {
        for sym in t.symbols {
            let sealed = seal(t.anchor, ARCH, sym, Some((t.anchor, sym)))
                .unwrap_or_else(|e| panic!("{sym}: abi_ref seal: {e}"));
            assert_eq!(sealed.hxaco, t.hxaco, "{sym}");
            assert_eq!(sealed.report["abi"]["compared_to_incumbent"], true, "{sym}");
            assert_eq!(sealed.report["abi"]["incumbent_symbol"], *sym);
            assert!(seal(t.anchor, ARCH, sym, Some((t.anchor, "no_such_kernel"))).is_err());
        }
    }
}

#[test]
fn module_is_refused_as_another_arch_or_symbol() {
    for t in twins() {
        assert!(verify_hxaco_many(t.hxaco, Arch::Gfx1200, t.symbols).is_err());
        assert!(verify_hxaco_many(t.hxaco, ARCH, &["no_such_kernel"]).is_err());
    }
}
