// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The script candidate on the CPU emulator: `kernels/iu4_gemm_fine.rip`, whose
//! dense fragment loads are the script body `rip_fragments`, runs the whole
//! launch grid through the `iu4_gemm` emulator harness of `peacemaker-isa` and
//! is compared bitwise with the exact CPU fold (`reference`), not only with the
//! Rust kernel. A fragment offset that is statically legal but wrong builds,
//! and is caught by the twin (the text differs from Rust) and by the numeric
//! differential (Y differs from the reference); M7 certifies legal machine
//! code, not arithmetic.
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Epi, RingDepth, Spec, Tune};
use peacemaker_isa::Arch;

#[macro_use]
#[path = "../../peacemaker-isa/tests/support/rocm.rs"]
mod rocm;
#[path = "../../peacemaker-isa/tests/support/iu4_emu.rs"]
mod iu4_emu;

const FINE: &str = include_str!("../../peacemaker-isa/src/kernels/iu4_gemm_fine.rip");

/// The script names the shipped variant; its depth is the source's `RING`.
fn build(src: &str, spec: Spec) -> Result<peacemaker_isa::Emitted, String> {
    let shipped = Spec {
        tune: Tune::default(),
        ..spec
    };
    peacemaker_rip::compile_named(src, "fine.rip", Arch::Gfx1201, &shipped.variant_name())
}

fn mismatches(got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn with_ring3(src: &str) -> String {
    let from = "const RING: usize = 2;";
    assert!(src.contains(from), "the fine twin no longer has `{from}`");
    src.replacen(from, "const RING: usize = 3;", 1)
}

/// SET (token and slab activations), ADD, and SET at ring depth 3: the script
/// candidate's Y equals the CPU fold bitwise, and the Rust kernel's.
#[test]
fn the_script_candidate_matches_the_cpu_fold_bitwise() {
    require_rocm_tool!("llvm-mc");
    // M = 136 / N = 40: partial row and token tiles; K = 768: two rolled trips
    // and the peeled group at depth 2, the entry branch at depth 3.
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    let ring3 = Tune {
        ring: RingDepth::Three,
        ..Tune::default()
    };
    for (spec, src) in [
        (Spec::control(Epi::Set), FINE.to_string()),
        (
            Spec {
                act: ALayout::Slab,
                ..Spec::control(Epi::Set)
            },
            FINE.to_string(),
        ),
        (Spec::control(Epi::Add), FINE.to_string()),
        (
            Spec {
                tune: ring3,
                ..Spec::control(Epi::Set)
            },
            with_ring3(FINE),
        ),
    ] {
        let name = spec.variant_name();
        let cand = build(&src, spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let got = run_emitted(spec, &cand, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = reference(&p, spec.epi == Epi::Add);
        assert_eq!(
            mismatches(&got, &want),
            0,
            "{name} {:?}: vs the CPU fold",
            spec.tune.ring
        );
        let rust = run_emitted(spec, &iu4_gemm::emit(spec).unwrap(), &p).unwrap();
        assert_eq!(mismatches(&got, &rust), 0, "{name}: vs the Rust kernel");
    }
}

/// The A loads of a sub-block with their halves swapped (columns 2-3 into the
/// registers of columns 0-1): every address stays inside the current slab, so
/// the build, the typed core and the register checks accept it; the twin and
/// the numeric differential refuse it.
#[test]
fn a_legal_but_wrong_fragment_offset_is_caught_by_twin_and_numerics() {
    require_rocm_tool!("llvm-mc");
    let from = "frag_load(f + 4 + 4 * half, fr_a, a + 2 * half)";
    assert!(
        FINE.contains(from),
        "the fine twin no longer contains `{from}`"
    );
    let src = FINE.replacen(
        from,
        "frag_load(f + 4 + 4 * half, fr_a, a + 2 * (1 - half))",
        1,
    );
    let spec = Spec::control(Epi::Set);
    let cand = build(&src, spec).expect("a statically legal body builds");
    let rust = iu4_gemm::emit(spec).unwrap();
    assert!(
        cand.s_text != rust.s_text,
        "the twin must see the changed offsets"
    );
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    let got = run_emitted(spec, &cand, &p).unwrap();
    let want = reference(&p, false);
    assert!(
        mismatches(&got, &want) > 0,
        "the numeric differential must refuse it"
    );
}
