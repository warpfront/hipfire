// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Canonical parallel CPU-AR target-row sampler builder module (gfx1151, gfx1201).
use peacemaker_isa::kernels::spec_sampling_rows_pm as rows;
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

#[test]
fn emission_is_deterministic_and_targets_gfx1151_and_gfx1201_only() {
    for arch in ARCHES {
        let (a, b) = (rows::emit(arch).unwrap(), rows::emit(arch).unwrap());
        assert_eq!(a.s_text, b.s_text, "{arch:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        assert!(a.s_text.contains(&format!("{}:", rows::symbol(arch))));
        assert!(a.shape.barriers > 0);
        assert!(
            a.shape.next_free_vgpr <= 224 && a.shape.next_free_sgpr <= 104,
            "{:?}",
            a.shape
        );
        let (emitted, text, _) = rows::emit_module(arch).unwrap();
        assert_eq!(emitted.len(), 1);
        assert!(text.contains(&format!("{}:", rows::symbol(arch))));
        assert_eq!(
            rows::module(arch),
            format!("spec_sampling_rows_parallel_pm_{}", arch.name())
        );
        assert_eq!(
            rows::symbol(arch),
            format!("spec_cpu_ar_rows_parallel_f32_pm_{}", arch.name())
        );
    }
    assert!(rows::emit(Arch::Gfx1100).is_err());
    assert!(rows::emit_module(Arch::Gfx1100).is_err());
}

/// Kernarg ABI: the ten-argument hipcc kernel, 56 bytes, no atomics, 64 bytes of LDS.
#[test]
fn kernarg_layout_matches_the_hipcc_kernel() {
    for arch in ARCHES {
        let text = rows::emit_module(arch).unwrap().1;
        let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
        let offsets: Vec<u32> = metadata
            .lines()
            .filter_map(|l| {
                l.trim()
                    .strip_prefix(".offset:")
                    .map(|v| v.trim().parse::<u32>().unwrap())
            })
            .collect();
        assert_eq!(offsets, [0, 8, 16, 24, 32, 36, 40, 44, 48, 52], "{arch:?}");
        assert!(
            text.contains(".amdhsa_kernarg_size 56\n"),
            "{arch:?}: kernarg size"
        );
        assert!(
            text.contains(&format!(
                ".amdhsa_group_segment_fixed_size {}\n",
                rows::STATIC_LDS
            )),
            "{arch:?}: LDS"
        );
        assert!(!text.contains("atomic"), "{arch:?}: no atomics");
        assert!(
            !text
                .lines()
                .any(|l| l.trim_start().starts_with("scratch_")
                    || l.trim_start().starts_with("flat_")),
            "{arch:?}"
        );
    }
}

#[test]
fn committed_bundles_equal_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for arch in ARCHES {
        let module = rows::module(arch);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = rows::emit_module(arch).unwrap().1;
        assert!(
            text_section(&link(&text, arch.name(), &module)) == text_section(&committed),
            "{module}: fresh emission differs from the committed bundle"
        );
    }
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-spec-rows-{}", std::process::id()));
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

    /// The symbol certifies on both arches: parse-back, independent wait replay, the
    /// committed shape contract, the static LDS bound, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_on_both_arches() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) = rows::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-spec-rows-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            let symbol = rows::symbol(arch);
            let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}: {m7}"
            );
            let path = format!(
                "{}/kernels/spec_sampling_rows_pm.{}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                arch.name()
            );
            let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(json, rows::contract(arch, &emitted[0]), "{path} is stale");
            let contract = serde_json::from_slice(&raw).unwrap();
            certify(
                &toolchain,
                &build,
                &s,
                arch.name(),
                &dir.join("manifest.json"),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }

    /// Every LDS access ends inside the 64 static bytes, for all eight waves.
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        for arch in ARCHES {
            let e = rows::emit(arch).unwrap();
            let end =
                pm_check::lds_bounds(&e.s_text, &rows::symbol(arch), 8, rows::STATIC_LDS).unwrap();
            assert!(end <= rows::STATIC_LDS, "{arch:?}: static end {end}");
        }
    }
}
