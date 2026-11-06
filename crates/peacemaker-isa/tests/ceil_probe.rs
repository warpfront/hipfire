// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `v_ceil_f32` through the Builder text path (`kernels/ceil_probe.rs`): the instruction is
//! emitted, parsed back through the shared tables, waits replayed, assembled natively and (with
//! ROCm) cross-checked against llvm-mc and M7 (lift identity + hazard/wait analyses).
use peacemaker_isa::kernels::ceil_probe;
use peacemaker_isa::{native, Arch};
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];

#[test]
fn emits_deterministically_and_contains_the_ceil() {
    for arch in ARCHES {
        let (a, b) = (
            ceil_probe::emit(arch).unwrap(),
            ceil_probe::emit(arch).unwrap(),
        );
        assert_eq!(a.s_text, b.s_text, "{arch:?}");
        assert!(
            a.s_text.contains("v_ceil_f32_e32 v3, v2"),
            "{arch:?}\n{}",
            a.s_text
        );
        assert!(a.s_text.contains(&format!("{}:", ceil_probe::symbol(arch))));
        assert_eq!(a.shape.barriers, 0);
    }
    assert!(ceil_probe::emit(Arch::Gfx942).is_err());
}

/// The native assembler encodes the row exactly as llvm-mc (`7E064502` is `v_ceil_f32_e32 v3, v2`).
#[test]
fn native_code_object_holds_the_llvm_mc_ceil_word() {
    for arch in ARCHES {
        let e = ceil_probe::emit(arch).unwrap();
        let co = native::assemble(&e.s_text, arch).unwrap();
        let word = 0x7e06_4502u32.to_le_bytes();
        assert!(
            co.windows(4).any(|w| w == word),
            "{arch:?}: v_ceil_f32_e32 v3, v2 (7E064502) missing from the native code object"
        );
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, Toolchain},
    };

    /// Parse-back through llvm-mc, wait replay, and M7: byte-exact lift of the linked ELF and no obligations.
    #[test]
    fn certifies_on_every_arch() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-ceil-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for arch in ARCHES {
            let e = ceil_probe::emit(arch).unwrap();
            ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            let s = dir.join(format!("{}.s", arch.name()));
            std::fs::write(&s, &e.s_text).unwrap();
            let toolchain = Toolchain::oracle();
            let built = build(
                &toolchain,
                &s,
                &dir.join(format!("{}.hsaco", arch.name())),
                arch.name(),
            )
            .unwrap();
            let symbol = ceil_probe::symbol(arch);
            let m7 = pm_check::m7(&built.elf, arch.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}: {m7}"
            );
        }
    }
}
