// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampled-DFlash device accept builder module (gfx1201): the PM twin of
//! `kernels/src/dflash_sampled_accept_sparse_q.hip`.
use peacemaker_isa::kernels::dflash_accept;
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;

#[test]
fn entry_is_deterministic_and_target_only_gfx1201() {
    let (a, b) = (
        dflash_accept::emit(ARCH).unwrap(),
        dflash_accept::emit(ARCH).unwrap(),
    );
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(a.shape.barriers, 4);
    assert!(a
        .s_text
        .contains(&format!("{}:", dflash_accept::symbol(ARCH))));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(dflash_accept::emit(arch).is_err(), "{arch:?}");
        assert!(dflash_accept::emit_module(arch).is_err(), "{arch:?}");
    }
    let (emitted, _, _) = dflash_accept::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!(
        dflash_accept::module(ARCH),
        "dflash_sampled_accept_pm_gfx1201"
    );
}

/// Kernarg ABI of the hipcc oracle: probs, tau, z, payload, out, n, vocab (48 bytes).
#[test]
fn kernarg_layout_matches_the_hipcc_oracle() {
    let text = dflash_accept::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let offsets: Vec<u32> = metadata
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse().unwrap())
        })
        .collect();
    assert_eq!(offsets, [0, 8, 16, 24, 32, 40, 44]);
    assert!(text.contains(".amdhsa_kernarg_size 48\n"));
    assert!(text.contains(&format!(
        ".amdhsa_group_segment_fixed_size {}\n",
        dflash_accept::STATIC_LDS
    )));
}

/// No hipcc fallback exists inside the module: only builder instructions, no
/// private segment, no atomics and no flat/global access.
#[test]
fn module_has_no_scratch_atomics_or_global_access() {
    let e = dflash_accept::emit(ARCH).unwrap();
    for l in e.s_text.lines().map(str::trim) {
        for banned in [
            "scratch_",
            "flat_",
            "global_",
            "buffer_atomic",
            "ds_add",
            "ds_max",
            "ds_min",
            "ds_bpermute",
            "v_writelane",
        ] {
            assert!(!l.starts_with(banned), "{l}");
        }
    }
    assert!(e.s_text.contains("s_barrier_signal"));
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = dflash_accept::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = dflash_accept::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle"
    );
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-dflash-accept-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
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
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The symbol certifies: parse-back, independent wait replay, the committed
    /// shape contract, the static LDS bound, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = dflash_accept::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-dflash-accept-cert-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let symbol = dflash_accept::symbol(ARCH);
        let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol)
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{symbol}: {m7}"
        );
        let path = format!(
            "{}/kernels/dflash_sampled_accept.{}.sparse_q.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name()
        );
        let contract =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join("sparse_q.manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{symbol}: {e}"));
    }

    /// Every LDS access ends inside the 6208 static bytes.
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        let e = dflash_accept::emit(ARCH).unwrap();
        let end = pm_check::lds_bounds(
            &e.s_text,
            &dflash_accept::symbol(ARCH),
            8,
            dflash_accept::STATIC_LDS,
        )
        .unwrap();
        assert!(end <= dflash_accept::STATIC_LDS, "static end {end}");
    }
}
