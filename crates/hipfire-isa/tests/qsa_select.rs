// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QSA selector pair builder module (gfx1151): rows16 F32 score + top-k select.
use hipfire_isa::Arch;
use hipfire_isa::kernels::{qsa_score, qsa_select::{self, Kind, Spec}, qsa_topk};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;

fn symbol(kind: Kind) -> String {
    match kind { Kind::Score => qsa_score::symbol(ARCH), Kind::ScoreBf16 => unreachable!("F32 module test"), Kind::Select => qsa_topk::symbol(ARCH) }
}

fn tag(kind: Kind) -> &'static str {
    match kind { Kind::Score => "score", Kind::ScoreBf16 => unreachable!("F32 module test"), Kind::Select => "select" }
}

#[test]
fn entries_are_deterministic_and_target_only_gfx1151() {
    for kind in Kind::ALL {
        let spec = Spec { arch: ARCH, kind };
        let (a, b) = (qsa_select::emit(spec).unwrap(), qsa_select::emit(spec).unwrap());
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        // The score kernel has no LDS and so no barrier.
        if kind == Kind::Score { assert_eq!(a.shape.barriers, 0) } else { assert!(a.shape.barriers > 0) }
        assert!(a.s_text.contains(&format!("{}:", symbol(kind))));
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1201] {
        for kind in Kind::ALL { assert!(qsa_select::emit(Spec { arch, kind }).is_err(), "{arch:?} {kind:?}") }
        assert!(qsa_score::emit(arch).is_err() && qsa_topk::emit(arch).is_err());
        assert!(qsa_select::emit_module(arch).is_err());
    }
    let (emitted, text, _) = qsa_select::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 2);
    let (s, t) = (text.find(&format!("{}:", symbol(Kind::Score))).unwrap(), text.find(&format!("{}:", symbol(Kind::Select))).unwrap());
    assert!(s < t, "module order is score then select");
    assert_eq!(qsa_select::module(ARCH), "qsa_select_pm_gfx1151");
}

/// Kernarg ABI: the hipcc pair's pointer/scalar offsets, 52 / 56 meaningful bytes.
#[test]
fn kernarg_layouts_match_the_hipcc_pair() {
    let text = qsa_select::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let args = |l: &str| l.trim().strip_prefix(".offset:").map(|v| v.trim().parse::<u32>().unwrap());
    let offsets: Vec<u32> = metadata.lines().filter_map(args).collect();
    let score = [0, 8, 16, 24, 28, 32, 36, 40, 44, 48];
    let select = [0, 8, 16, 24, 28, 32, 36, 40, 44, 48];
    assert_eq!(offsets, [&score[..], &select[..]].concat());
    for (kind, size) in [(Kind::Score, 52), (Kind::Select, 56)] {
        let e = qsa_select::emit(Spec { arch: ARCH, kind }).unwrap();
        assert!(e.s_text.contains(&format!(".amdhsa_kernarg_size {size}\n")), "{kind:?}: kernarg size");
    }
    for (name, size) in ["selected", "mirror"].iter().zip([8, 8]) {
        let at = metadata.find(&format!(".name: {name}\n")).unwrap();
        assert!(metadata[at..].lines().nth(2).unwrap().trim() == format!(".size: {size}"), "{name}");
    }
}
#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = qsa_select::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = qsa_select::emit_module(ARCH).unwrap().1;
    assert!(text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed), "{module}: fresh emission differs from the committed bundle");
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-qsa-select-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (dir.join(format!("{stem}.s")), dir.join(format!("{stem}.o")), dir.join(format!("{stem}.co")));
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj"]).arg(&s).arg("-o").arg(&o).status().unwrap().success());
    assert!(Command::new(linker).arg("-shared").arg(&o).arg("-o").arg(&co).status().unwrap().success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum).find_map(|i| {
        let name = &e[names + u32_(sh(i))..];
        name.starts_with(b".text\0").then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
    }).unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, toolchain::{build, certify, Toolchain}};

    /// Both symbols certify: parse-back, independent wait replay, the
    /// committed shape contract, the static LDS bound, and M7's lift identity
    /// plus wait/hazard/barrier/window analyses of the linked ELF,
    /// obligation-free.
    #[test]
    fn module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = qsa_select::emit_module(ARCH).unwrap();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-qsa-select-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for kind in Kind::ALL {
            let symbol = symbol(kind);
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}: {m7}");
            let path = format!("{}/kernels/qsa_select.{}.{}.contract.json", env!("CARGO_MANIFEST_DIR"), ARCH.name(), tag(kind));
            let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
            certify(&toolchain, &build, &s, ARCH.name(), &dir.join(format!("{}.manifest.json", tag(kind))), Some(&contract), "test", "test")
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }

    /// Every select LDS access ends inside the 10304 static bytes; score has no LDS.
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        let e = qsa_select::emit(Spec { arch: ARCH, kind: Kind::Select }).unwrap();
        let end = pm_check::lds_bounds(&e.s_text, &symbol(Kind::Select), 8, qsa_topk::STATIC_LDS).unwrap();
        assert!(end <= qsa_topk::STATIC_LDS, "static end {end}");
        let e = qsa_select::emit(Spec { arch: ARCH, kind: Kind::Score }).unwrap();
        assert!(!e.s_text.lines().any(|l| l.trim_start().starts_with("ds_")), "score uses no LDS");
    }
}
