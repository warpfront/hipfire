// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 MQ6 trunk GEMM builder family (gfx1201 W4 + W8 entries).
use hipfire_isa::Arch;
use hipfire_isa::kernels::qwen4_mq6_x4::{self, Spec};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;

#[test]
fn entries_are_deterministic_and_target_only_gfx1201() {
    for wr in Spec::ALL {
        let spec = Spec { arch: ARCH, wr };
        let (a, b) = (qwen4_mq6_x4::emit(spec).unwrap(), qwen4_mq6_x4::emit(spec).unwrap());
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        // One staging barrier per 32-K chunk of a group (eight) and one for the prologue stage.
        assert_eq!((a.shape.barriers, a.shape.wmma), (9, 256), "{spec:?}");
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(qwen4_mq6_x4::emit(Spec { arch, wr: 4 }).is_err());
    }
    for wr in [0, 2, 6, 16] {
        assert!(qwen4_mq6_x4::emit(Spec { arch: ARCH, wr }).is_err(), "wr {wr}");
    }
}

/// The runtime embeds one certified module per arch: it must be exactly
/// what the builder emits today.
#[test]
fn committed_bundles_equal_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = Spec::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = qwen4_mq6_x4::emit_module(ARCH).unwrap().1;
    assert!(text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed), "{module}: fresh emission differs from the committed bundle");
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-mq6x4-{}", std::process::id()));
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

    /// Every symbol of the module certifies: parse-back, independent wait
    /// replay, its committed shape contract (counts, registers, 40960 fixed
    /// and zero dynamic LDS), the all-lane static LDS bound, and M7's lift
    /// identity plus wait/hazard/barrier/window analyses of the linked ELF,
    /// obligation-free.
    #[test]
    fn modules_pass_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = qwen4_mq6_x4::emit_module(ARCH).unwrap();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-mq6x4-cert-{}-{}", ARCH.name(), std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for wr in Spec::ALL {
            let symbol = Spec { arch: ARCH, wr }.symbol();
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
            let path = format!("{}/kernels/qwen4_mq6_x4.{}.w{wr}.contract.json", env!("CARGO_MANIFEST_DIR"), ARCH.name());
            let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
            certify(&toolchain, &build, &s, ARCH.name(), &dir.join(format!("w{wr}.manifest.json")), Some(&contract), "test", "test")
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }

    /// Both entries use only the static allocation: every DS access of every
    /// lane of every wave ends inside the two 20480-byte X slots, and the
    /// bound is tight (one byte less is refused).
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        for wr in Spec::ALL {
            let e = qwen4_mq6_x4::emit(Spec { arch: ARCH, wr }).unwrap();
            let symbol = Spec { arch: ARCH, wr }.symbol();
            let waves = u32::from(wr);
            let (end, hosted) = pm_check::lds_bounds_host(&e.s_text, &symbol, waves, qwen4_mq6_x4::GROUP_BYTES, &[]).unwrap();
            assert!(end > qwen4_mq6_x4::SLOT_BYTES && end <= qwen4_mq6_x4::GROUP_BYTES && hosted == 0, "{symbol}: static end {end}, {hosted} host-bounded accesses");
            assert!(pm_check::lds_bounds_host(&e.s_text, &symbol, waves, end - 1, &[]).is_err(), "{symbol}: bound is not tight");
        }
    }
}
