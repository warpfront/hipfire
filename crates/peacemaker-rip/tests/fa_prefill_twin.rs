// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twins of `fa_prefill::emit` / `emit_preconvert` (`kernels/fa_prefill.rip`, the Q8_0
//! module, and `kernels/fa_prefill_fp8.rip`, the native-fp8 direct export; both composed from the `fa::*`
//! natives and `Builder` statements) re-emit the production kernels' bytes, the coarse natives return the
//! Rust functions' values, and edits of the composition that break a builder rule are refused by the builder.

use peacemaker_isa::kernels::fa_prefill::{self as fa, FaKvCodec, FaPrefillSpec};
use peacemaker_isa::Arch;

const Q8: &str = include_str!("../../peacemaker-isa/src/kernels/fa_prefill.rip");
const FP8: &str = include_str!("../../peacemaker-isa/src/kernels/fa_prefill_fp8.rip");

fn spec(kv: FaKvCodec) -> FaPrefillSpec {
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv,
    }
}

fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: rust `{x}` vs rip `{y}`", i + 1);
        }
    }
    format!(
        "lengths {} vs {} lines",
        a.lines().count(),
        b.lines().count()
    )
}

fn build(src: &str, variant: &str) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(src, "variation.rip", Arch::Gfx1201, variant)
}

fn same(name: &str, rust: &peacemaker_isa::Emitted, rip: &peacemaker_isa::Emitted) {
    assert!(
        rust.s_text == rip.s_text,
        "{name}: {}",
        first_difference(&rust.s_text, &rip.s_text)
    );
    assert_eq!(
        serde_json::to_value(&rust.proof).unwrap(),
        serde_json::to_value(&rip.proof).unwrap(),
        "{name}: proof"
    );
}

/// `s_text` and the proof (the builder's facts about the program: waits, barriers, loop fixpoints,
/// hazards) are identical for every export; a difference would mean the composition emitted another program.
#[test]
fn fa_prefill_twin_matches_the_rust_emit() {
    let q8 = spec(FaKvCodec::Q8_0);
    for (variant, rust) in [
        ("direct", fa::emit(Arch::Gfx1201, &q8).unwrap()),
        (
            "preconvert",
            fa::emit_preconvert(Arch::Gfx1201, &q8).unwrap(),
        ),
    ] {
        let rip = peacemaker_rip::compile_named(Q8, "fa_prefill.rip", Arch::Gfx1201, variant)
            .unwrap_or_else(|e| panic!("{variant}: {e}"));
        same(variant, &rust, &rip);
    }
}

#[test]
fn fa_prefill_fp8_twin_matches_the_rust_emit() {
    let rust = fa::emit(Arch::Gfx1201, &spec(FaKvCodec::Fp8E4M3)).unwrap();
    let rip = peacemaker_rip::compile_named(FP8, "fa_prefill_fp8.rip", Arch::Gfx1201, "direct")
        .unwrap_or_else(|e| panic!("fp8 direct: {e}"));
    same("fp8 direct", &rust, &rip);
}

#[test]
fn the_twins_declare_exactly_their_variants() {
    let mut q8 = peacemaker_rip::variants(Q8, "fa_prefill.rip").unwrap();
    q8.sort();
    assert_eq!(q8, ["direct", "preconvert"]);
    assert_eq!(
        peacemaker_rip::variants(FP8, "fa_prefill_fp8.rip").unwrap(),
        ["direct"]
    );
}

#[test]
fn an_unknown_variant_is_refused_with_its_grammar() {
    let e = build(Q8, "split").expect_err("unknown variant");
    assert!(e.contains("expected direct or preconvert"), "{e}");
    let e = build(FP8, "preconvert").expect_err("fp8 has no preconvert");
    assert!(e.contains("expected direct"), "{e}");
}

#[test]
fn the_twins_build_on_gfx1201_only() {
    for (src, variant) in [(Q8, "direct"), (Q8, "preconvert"), (FP8, "direct")] {
        let e = peacemaker_rip::compile_named(src, "fa.rip", Arch::Gfx1151, variant)
            .expect_err("gfx1151 must be refused");
        assert!(e.contains("fa_prefill: built for gfx1201 only"), "{e}");
    }
}

// ---- coarse natives ---------------------------------------------------------

const COARSE: &str = r#"
fn spec_q8() -> FaSpec { FaSpec { heads: 16, kv_heads: 2, head_dim: 256, kv: FaKv::Q8_0 } }
fn spec_fp8() -> FaSpec { FaSpec { heads: 16, kv_heads: 2, head_dim: 256, kv: FaKv::Fp8E4M3 } }
fn emit(spec: Spec) -> Result<Emitted, String> {
    match spec.variant.as_str() {
        "direct" => FaPrefill::emit(spec.arch, spec_q8()),
        "preconvert" => FaPrefill::emit_preconvert(spec.arch, spec_q8()),
        "fp8" => FaPrefill::emit(spec.arch, spec_fp8()),
        "bad" => FaPrefill::emit(spec.arch, FaSpec { heads: 8, kv_heads: 2, head_dim: 256, kv: FaKv::Q8_0 }),
        "short" => FaPrefill::emit(spec.arch, FaSpec { heads: 16, kv_heads: 2, head_dim: 256 }),
        v => Err(format!("variant {v}")),
    }
}
fn module_count() -> usize { let (a, _t, _p) = FaPrefill::emit_module(Arch::Gfx1201, spec_q8()).unwrap(); a.len() }
fn module_text() -> String { let (_a, t, _p) = FaPrefill::emit_module(Arch::Gfx1201, spec_q8()).unwrap(); t }
fn module_proof() -> String { let (_a, _t, p) = FaPrefill::emit_module(Arch::Gfx1201, spec_q8()).unwrap(); p }
fn fp8_module_text() -> String { let (_a, t, _p) = FaPrefill::emit_fp8_module(Arch::Gfx1201, spec_fp8()).unwrap(); t }
fn wrong_module() -> String { let (_a, t, _p) = FaPrefill::emit_module(Arch::Gfx1201, spec_fp8()).unwrap(); t }
"#;

#[test]
fn fa_prefill_emit_natives_return_the_rust_emitted() {
    let q8 = spec(FaKvCodec::Q8_0);
    for (variant, want) in [
        ("direct", fa::emit(Arch::Gfx1201, &q8).unwrap()),
        (
            "preconvert",
            fa::emit_preconvert(Arch::Gfx1201, &q8).unwrap(),
        ),
        (
            "fp8",
            fa::emit(Arch::Gfx1201, &spec(FaKvCodec::Fp8E4M3)).unwrap(),
        ),
    ] {
        let got = peacemaker_rip::compile(COARSE, Arch::Gfx1201, variant).unwrap();
        assert_eq!(want.s_text, got.s_text, "{variant}");
    }
    // The Rust gate stays in force.
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1201, "bad").expect_err("8 heads");
    assert!(e.contains("only heads 16, kv_heads 2, head_dim 256"), "{e}");
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1151, "direct").expect_err("gfx1151");
    assert!(e.contains("built for gfx1201 only"), "{e}");
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1201, "short").expect_err("no kv");
    assert!(e.contains("missing the field `kv`"), "{e}");
}

#[test]
fn fa_prefill_module_natives_return_the_rust_modules() {
    let (emitted, text, proof) = fa::emit_module(Arch::Gfx1201, &spec(FaKvCodec::Q8_0)).unwrap();
    assert_eq!(
        peacemaker_rip::eval(COARSE, "module_count").unwrap(),
        emitted.len().to_string()
    );
    assert_eq!(
        peacemaker_rip::eval(COARSE, "module_text").unwrap(),
        format!("{text:?}")
    );
    let json = serde_json::to_string(&proof).unwrap();
    assert_eq!(
        peacemaker_rip::eval(COARSE, "module_proof").unwrap(),
        format!("{json:?}")
    );
    let (_, text8, _) = fa::emit_fp8_module(Arch::Gfx1201, &spec(FaKvCodec::Fp8E4M3)).unwrap();
    assert_eq!(
        peacemaker_rip::eval(COARSE, "fp8_module_text").unwrap(),
        format!("{text8:?}")
    );
    // emit_module is the Q8_0 module.
    let e = peacemaker_rip::eval(COARSE, "wrong_module").expect_err("fp8 spec in the Q8 module");
    assert!(e.contains("emit_module is the Q8_0 module"), "{e}");
}

// ---- compositions the script can change -------------------------------------------

fn edit(src: &str, from: &str, to: &str) -> String {
    assert!(src.contains(from), "the twin no longer contains `{from}`");
    src.replacen(from, to, 1)
}

/// Variations that are different programs but break no builder rule build and differ from the
/// production bytes: the order of the four softmax/PV subtiles, and an extra DS wait after QK.
#[test]
fn reordered_subtiles_and_an_extra_wait_build_and_differ() {
    for (src, variant, qk_tail) in [
        (Q8, "direct", "    fa::qk(b, g)?;\n"),
        (FP8, "direct", "    fa::qk(b, g)?;\n"),
    ] {
        let base = build(src, variant).unwrap();
        let reversed = edit(
            src,
            "for sub in 0..4 {\n        fa::softmax_sub(b, g, sub)?;",
            "for k in 0..4 {\n        let sub = 3 - k;\n        fa::softmax_sub(b, g, sub)?;",
        );
        let reversed = build(&reversed, variant).unwrap();
        assert!(reversed.s_text != base.s_text, "{variant}: reversed");
        let waited = edit(
            src,
            qk_tail,
            &format!("{qk_tail}    b.wait(Counter::Ds, 0)?;\n"),
        );
        let waited = build(&waited, variant).unwrap();
        assert!(waited.s_text != base.s_text, "{variant}: wait");
    }
}

/// A barrier over a slot the kernel never stores is refused by the slot machine.
#[test]
fn publishing_a_slot_that_does_not_exist_is_refused() {
    let q8 = edit(Q8, "[ready(0), ready(1)]", "[ready(0), ready(1), ready(2)]");
    let e = build(&q8, "direct").expect_err("slot 2");
    assert!(e.contains("slot") || e.contains("LDS"), "{e}");
    let fp8 = edit(FP8, "b.barrier_with_scope([ready(0)], \"lds_only\")?;\n    fa::label(b, fa::KREADY)?;", "b.barrier_with_scope([ready(0), ready(1)], \"lds_only\")?;\n    fa::label(b, fa::KREADY)?;");
    let e = build(&fp8, "direct").expect_err("slot 1 in the single-stage kernel");
    assert!(e.contains("slot") || e.contains("LDS"), "{e}");
}

/// The fp8 stage is reused: dropping the retire between QK and the V fill leaves it Published/Reading,
/// and the V fill's store into it is refused.
#[test]
fn dropping_the_stage_retire_before_the_v_fill_is_refused() {
    let src = edit(
        FP8,
        "b.barrier_with_scope([retire(0)], \"lds_only\")?;\n    fa::fill_v(b, g)?;",
        "fa::fill_v(b, g)?;",
    );
    let e = build(&src, "direct").expect_err("V fill into a Reading stage");
    assert!(e.contains("stage"), "{e}");
}

/// Dropping the Q8 retire of the V plane leaves it published at the loop's back edge.
#[test]
fn dropping_a_plane_retire_is_refused() {
    let src = edit(
        Q8,
        "b.barrier_with_scope([retire(0), retire(1)], \"lds_only\")?;",
        "b.barrier_with_scope([retire(0)], \"lds_only\")?;",
    );
    let e = build(&src, "direct").expect_err("v_plane stays published");
    assert!(
        e.contains("v_plane") || e.contains("LDS") || e.contains("loop"),
        "{e}"
    );
}

/// The natives that exist for one codec refuse the other.
#[test]
fn the_codec_specific_fills_refuse_the_other_codec() {
    let e = build(
        &edit(Q8, "fa::fill(b, g)?;", "fa::fill_k(b, g)?;"),
        "direct",
    )
    .expect_err("fp8 fill in the Q8 kernel");
    assert!(
        e.contains("only the fp8 kernel fills K and V separately"),
        "{e}"
    );
    let e = build(
        &edit(FP8, "fa::fill_k(b, g)?;", "fa::fill(b, g)?;"),
        "direct",
    )
    .expect_err("Q8 fill in the fp8 kernel");
    assert!(
        e.contains("only the Q8_0 kernel fills K and V jointly"),
        "{e}"
    );
}

#[test]
fn the_subtile_index_is_checked() {
    let e = build(
        &edit(Q8, "for sub in 0..4 {", "for sub in 0..5 {"),
        "direct",
    )
    .expect_err("subtile 4");
    assert!(e.contains("softmax_sub: subtile 4 is outside 0..4"), "{e}");
}
