// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 symmetric IU4 MoE builder family (gfx1151 + gfx1201).
use hipfire_isa::Arch;
use hipfire_isa::kernels::qwen4_moe_sym::{self, Kind, Spec};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

fn specs(arch: Arch) -> impl Iterator<Item = Spec> {
    qwen4_moe_sym::module_specs(arch).into_iter()
}

#[test]
fn entries_are_deterministic_and_target_only_their_arches() {
    for arch in ARCHES {
        for spec in specs(arch) {
            let (a, b) = (qwen4_moe_sym::emit(spec).unwrap(), qwen4_moe_sym::emit(spec).unwrap());
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
            // Only the expert-run gate/up hands its gate rows over LDS: one
            // barrier on each of the gate and up paths.
            let handoff = spec.kind == Kind::GateUp && spec.nt > 1;
            assert_eq!(a.shape.barriers, if handoff { 2 } else { 0 }, "{spec:?}");
            assert_eq!(a.s_text.contains("ds_store") || a.s_text.contains("ds_load"), handoff, "{spec:?}");
        }
    }
    assert!(qwen4_moe_sym::emit(Spec { arch: Arch::Gfx1100, kind: Kind::Down, nt: 1, rr: 1 }).is_err());
    // gfx11 NT8 does not fit the VGPR file.
    assert!(qwen4_moe_sym::emit(Spec { arch: Arch::Gfx1151, kind: Kind::GateUp, nt: 8, rr: 1 }).is_err());
}

/// Row repeats exist only as 2 or 4 blocks of the gfx1151 down NT4 entry.
#[test]
fn row_repeat_shapes_are_validated() {
    let spec = |arch, kind, nt, rr| Spec { arch, kind, nt, rr };
    for rr in [0, 3, 8] { assert!(qwen4_moe_sym::emit(spec(Arch::Gfx1151, Kind::Down, 4, rr)).is_err(), "rr {rr}"); }
    for bad in [
        spec(Arch::Gfx1151, Kind::GateUp, 4, 2),
        spec(Arch::Gfx1151, Kind::Down, 1, 2),
        spec(Arch::Gfx1151, Kind::Down, 2, 2),
        spec(Arch::Gfx1201, Kind::Down, 4, 2),
        spec(Arch::Gfx1201, Kind::Down, 8, 4),
    ] {
        assert!(qwen4_moe_sym::emit(bad).is_err(), "{bad:?}");
    }
}

/// The runtime embeds one certified module per arch: it must be exactly
/// what the builder emits today.
#[test]
fn committed_bundles_equal_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for arch in ARCHES {
        let module = Spec::module(arch);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = qwen4_moe_sym::emit_module(arch).unwrap().1;
        assert!(text_section(&link(&text, arch.name(), &module)) == text_section(&committed), "{module}: fresh emission differs from the committed bundle");
    }
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-q4s-{}", std::process::id()));
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

    /// Every symbol of both modules certifies: parse-back, independent wait
    /// replay, its committed shape contract, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn modules_pass_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) = qwen4_moe_sym::emit_module(arch).unwrap();
            for e in &emitted { ledger_replay::replay_waits(&e.s_text, arch).unwrap(); }
            let dir = std::env::temp_dir().join(format!("hipfire-isa-q4s-cert-{}-{}", arch.name(), std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            for spec in specs(arch) {
                let symbol = spec.symbol();
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
                assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
                let tag = spec.variant();
                let path = format!("{}/kernels/qwen4_moe_sym.{}.{tag}.contract.json", env!("CARGO_MANIFEST_DIR"), arch.name());
                let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
                certify(&toolchain, &build, &s, arch.name(), &dir.join(format!("{tag}.manifest.json")), Some(&contract), "test", "test")
                    .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            }
        }
    }
}
