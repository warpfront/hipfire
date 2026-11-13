// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The libm receipt kernels emit through the checked builder and assemble on gfx1100 / gfx1151 / gfx1201.
use peacemaker_isa::kernels::libm_exact::{emit, emit_elf, symbol, Fun};
use peacemaker_isa::Arch;

const ARCHES: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];
const FUNS: [Fun; 4] = [Fun::Sinf, Fun::Cosf, Fun::Sincosf, Fun::Powf];

#[test]
fn emits_everywhere() {
    for arch in ARCHES {
        for fun in FUNS {
            for masked in [false, true] {
                let e = emit(arch, fun, masked).unwrap_or_else(|e| panic!("{arch:?} {fun:?}: {e}"));
                assert!(e.s_text.contains(&symbol(arch, fun, masked)));
                let large = e.s_text.contains("s_cbranch_execz");
                assert_eq!(
                    large,
                    !masked && fun != Fun::Powf,
                    "{arch:?} {fun:?} masked={masked}"
                );
                assert!(!e.s_text.contains("vcc_lo"), "{arch:?} {fun:?}: vcc leaked");
                if arch.gfx12() {
                    assert!(e.s_text.contains("v_mad_co_u64_u32") == (fun != Fun::Powf));
                }
            }
        }
    }
}

#[test]
fn assembles_everywhere() {
    for arch in ARCHES {
        for fun in FUNS {
            let elf =
                emit_elf(arch, fun, false).unwrap_or_else(|e| panic!("{arch:?} {fun:?}: {e}"));
            assert!(elf.starts_with(b"\x7fELF"));
        }
    }
}
