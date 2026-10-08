// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 MQ6 trunk GEMM builder family, gfx1151 (W4 + W8 plain and BF16-out,
//! W8 regions and W8 HC-write): exact twins of hipcc's `u3_b8_r8` entries.
use hipfire_isa::Arch;
use hipfire_isa::kernels::qwen4_mq6_x4_gfx11::{self, Kind, Spec};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;

/// `(wr, kind, symbol, kernarg size, kernarg (offset, size) list)` of the six
/// entries. The offsets are hipcc's (`llvm-readelf --notes` of the U3 object):
/// plain / bf16out `A X Y M K N`; regions `A0 A1 A2 X Y0 Y1 Y2 M0 M1 M2 K N`;
/// hcw `A X Y M K N hc_streams hc_gates hc_bf16`.
type Entry = (u8, Kind, &'static str, u32, &'static [(u32, u32)]);
const PLAIN_ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)];
const REGIONS_ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 8), (40, 8), (48, 8), (56, 4), (60, 4), (64, 4), (68, 4), (72, 4)];
const HCW_ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4), (40, 8), (48, 8), (56, 4)];
const ENTRIES: [Entry; 6] = [
    (8, Kind::Plain, "qwen4_mq6_x4_pm_gfx1151_w8", 36, PLAIN_ARGS),
    (4, Kind::Plain, "qwen4_mq6_x4_pm_gfx1151_w4", 36, PLAIN_ARGS),
    (8, Kind::Bf16, "qwen4_mq6_x4_pm_gfx1151_w8_bf16out", 36, PLAIN_ARGS),
    (4, Kind::Bf16, "qwen4_mq6_x4_pm_gfx1151_w4_bf16out", 36, PLAIN_ARGS),
    (8, Kind::Regions, "qwen4_mq6_x4_pm_gfx1151_w8_regions", 76, REGIONS_ARGS),
    (8, Kind::Hcw, "qwen4_mq6_x4_pm_gfx1151_w8_hcw", 60, HCW_ARGS),
];

#[test]
fn spec_all_is_the_six_pm_entries() {
    let mut got: Vec<_> = Spec::ALL.into_iter().map(|(wr, kind)| Spec { arch: ARCH, wr, kind }.symbol()).collect();
    let mut want: Vec<_> = ENTRIES.iter().map(|e| e.2.to_string()).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(Spec::module(ARCH), "qwen4_mq6_x4_pm_gfx1151");
}

#[test]
fn entries_are_deterministic_and_target_only_gfx1151() {
    for (wr, kind) in Spec::ALL {
        let spec = Spec { arch: ARCH, wr, kind };
        let (a, b) = (qwen4_mq6_x4_gfx11::emit(spec).unwrap(), qwen4_mq6_x4_gfx11::emit(spec).unwrap());
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        // One 16x16 tile chain per K-step: 16 tiles x 16 K-steps of a 256-K group.
        assert_eq!(a.shape.wmma, 256, "{spec:?}");
    }
    let (_, ta, pa) = qwen4_mq6_x4_gfx11::emit_module(ARCH).unwrap();
    let (_, tb, pb) = qwen4_mq6_x4_gfx11::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.module), (tb, pb.module));
    // Every other arch (the gfx1201 twin has its own builder) refuses every entry and the module.
    for arch in [Arch::Gfx1201, Arch::Gfx1100] {
        for (wr, kind) in Spec::ALL {
            assert!(qwen4_mq6_x4_gfx11::emit(Spec { arch, wr, kind }).is_err(), "{arch:?} w{wr} {kind:?}");
        }
        assert!(qwen4_mq6_x4_gfx11::emit_module(arch).is_err(), "{arch:?} module");
    }
    // Only the six PM entries exist.
    for wr in [0, 2, 6, 16] {
        for kind in [Kind::Plain, Kind::Bf16, Kind::Regions, Kind::Hcw] {
            assert!(qwen4_mq6_x4_gfx11::emit(Spec { arch: ARCH, wr, kind }).is_err(), "wr {wr} {kind:?}");
        }
    }
    for kind in [Kind::Regions, Kind::Hcw] {
        assert!(qwen4_mq6_x4_gfx11::emit(Spec { arch: ARCH, wr: 4, kind }).is_err(), "w4 {kind:?}");
    }
}

/// The kernarg block is hipcc's: size and every (offset, size) of the metadata.
#[test]
fn kernarg_abi_matches_hipcc_metadata() {
    for (wr, kind, symbol, size, args) in ENTRIES {
        let e = qwen4_mq6_x4_gfx11::emit(Spec { arch: ARCH, wr, kind }).unwrap();
        assert!(e.s_text.contains(&format!(".kernarg_segment_size: {size}\n")), "{symbol}: kernarg size");
        let field = |key: &str| -> Vec<u32> {
            e.s_text.lines().filter_map(|l| l.trim().strip_prefix(key)).map(|v| v.trim().parse().unwrap()).collect()
        };
        let want: Vec<u32> = args.iter().map(|a| a.0).collect();
        assert_eq!(field(".offset:"), want, "{symbol}: kernarg offsets");
        let want: Vec<u32> = args.iter().map(|a| a.1).collect();
        assert_eq!(field(".size:"), want, "{symbol}: kernarg sizes");
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
    let text = qwen4_mq6_x4_gfx11::emit_module(ARCH).unwrap().1;
    assert!(text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed), "{module}: fresh emission differs from the committed bundle");
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-mq6x4g11-{}", std::process::id()));
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

/// Committed per-entry shape contract path of a symbol.
fn contract_path(symbol: &str) -> String {
    let entry = symbol.strip_prefix("qwen4_mq6_x4_pm_gfx1151_").unwrap();
    format!("{}/kernels/qwen4_mq6_x4.gfx1151.{entry}.contract.json", env!("CARGO_MANIFEST_DIR"))
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
        let (emitted, text, _) = qwen4_mq6_x4_gfx11::emit_module(ARCH).unwrap();
        assert_eq!(emitted.len(), ENTRIES.len());
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-mq6x4g11-cert-{}-{}", ARCH.name(), std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for (wr, kind) in Spec::ALL {
            let symbol = Spec { arch: ARCH, wr, kind }.symbol();
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
            let path = contract_path(&symbol);
            let contract = serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap();
            certify(&toolchain, &build, &s, ARCH.name(), &dir.join(format!("{symbol}.manifest.json")), Some(&contract), "test", "test")
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }

    /// Every entry uses only the static allocation: every DS access of every
    /// lane of every wave ends inside the two 20480-byte X slots (the HC-write
    /// parking reuses them) and inside 40960 bytes, and the bound is tight
    /// (one byte less is refused).
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        for (wr, kind) in Spec::ALL {
            let e = qwen4_mq6_x4_gfx11::emit(Spec { arch: ARCH, wr, kind }).unwrap();
            let symbol = Spec { arch: ARCH, wr, kind }.symbol();
            let waves = u32::from(wr);
            let (end, hosted) = pm_check::lds_bounds_host(&e.s_text, &symbol, waves, qwen4_mq6_x4_gfx11::GROUP_BYTES, &[]).unwrap();
            assert!(end > qwen4_mq6_x4_gfx11::SLOT_BYTES && end <= qwen4_mq6_x4_gfx11::GROUP_BYTES && hosted == 0, "{symbol}: static end {end}, {hosted} host-bounded accesses");
            assert!(qwen4_mq6_x4_gfx11::GROUP_BYTES <= 40960, "{symbol}");
            assert!(pm_check::lds_bounds_host(&e.s_text, &symbol, waves, end - 1, &[]).is_err(), "{symbol}: bound is not tight");
        }
    }
}
