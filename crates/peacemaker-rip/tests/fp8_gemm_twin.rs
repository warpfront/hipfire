// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of `fp8_gemm::emit` (`kernels/fp8_gemm.rip`, composed from the `fp8::*`
//! natives and `Builder` statements) re-emits the production kernel's bytes for every valid `Spec`
//! on gfx1201, the coarse natives return the Rust functions' values, and edits of the composition
//! that break a builder rule are refused by the builder.

use peacemaker_isa::kernels::fp8_gemm::{self as fp8, ActScale, Epi, Spec};
use peacemaker_isa::Arch;

const TWIN: &str = include_str!("../../peacemaker-isa/src/kernels/fp8_gemm.rip");

const EPIS: [Epi; 7] = [
    Epi::Set,
    Epi::Add,
    Epi::GateUpSilu,
    Epi::GateUpSiluBf16,
    Epi::Qkv,
    Epi::Qkvza,
    Epi::QkvzaGdn,
];

fn every_spec() -> Vec<Spec> {
    let mut out = Vec::new();
    for act_scale in [ActScale::Row, ActScale::K128] {
        for epi in EPIS {
            out.push(Spec {
                arch: Arch::Gfx1201,
                act_scale,
                epi,
            });
        }
    }
    out
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

/// `s_text` is byte-identical for every valid spec, and so is the proof (it records the builder's
/// facts about the program: waits, barriers, loop fixpoints, hazards); a difference would mean the
/// composition emitted another program.
#[test]
fn fp8_gemm_twin_matches_the_rust_emit() {
    let mut valid = 0;
    for spec in every_spec() {
        let name = spec.variant();
        if spec.validate().is_err() {
            // The script refuses what the Rust generator refuses, with its message.
            let e = peacemaker_rip::compile_named(TWIN, "fp8_gemm.rip", spec.arch, &name)
                .expect_err(&format!("{name}: the twin built an invalid spec"));
            let want = spec.validate().unwrap_err();
            assert!(e.contains(&want), "{name}: {e} does not carry `{want}`");
            continue;
        }
        valid += 1;
        let rust = fp8::emit(spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = peacemaker_rip::compile_named(TWIN, "fp8_gemm.rip", spec.arch, &name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
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
    assert_eq!(
        valid, 12,
        "the number of valid specs changed: update VARIANTS"
    );
}

#[test]
fn the_twin_declares_exactly_the_valid_variants() {
    let mut declared = peacemaker_rip::variants(TWIN, "fp8_gemm.rip").unwrap();
    let mut valid: Vec<String> = every_spec()
        .into_iter()
        .filter(|s| s.validate().is_ok())
        .map(|s| s.variant())
        .collect();
    valid.sort();
    declared.sort();
    assert_eq!(declared, valid);
}

#[test]
fn a_malformed_variant_is_refused_with_its_grammar() {
    for (v, want) in [
        ("row.set", "expected ratio-256x128x8-SCALE.EPI"),
        ("ratio-256x128x8-row", "expected ratio-256x128x8-SCALE.EPI"),
        ("ratio-256x128x8-col.set", "unknown scale `col`"),
        ("ratio-256x128x8-row.mul", "unknown epilogue `mul`"),
    ] {
        let e = peacemaker_rip::compile_named(TWIN, "fp8_gemm.rip", Arch::Gfx1201, v).expect_err(v);
        assert!(e.contains(want), "{v}: {e}");
    }
}

#[test]
fn the_twin_builds_on_gfx1201_only() {
    let e = peacemaker_rip::compile_named(
        TWIN,
        "fp8_gemm.rip",
        Arch::Gfx1151,
        "ratio-256x128x8-row.set",
    )
    .expect_err("gfx1151 must be refused");
    assert!(e.contains("fp8 GEMM requires gfx1201"), "{e}");
}

// ---- coarse natives ---------------------------------------------------------

const COARSE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Fp8Gemm::emit(Fp8Spec { arch: spec.arch, scale: Fp8Scale::K128, epi: Fp8Epi::Qkvza })
}
"#;

#[test]
fn fp8_gemm_emit_native_returns_the_rust_emitted() {
    let want = fp8::emit(Spec {
        arch: Arch::Gfx1201,
        act_scale: ActScale::K128,
        epi: Epi::Qkvza,
    })
    .unwrap();
    let got = peacemaker_rip::compile(COARSE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(want.s_text, got.s_text);
    // Spec::validate stays the gate.
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1151, "default").expect_err("gfx1151");
    assert!(e.contains("fp8 GEMM requires gfx1201"), "{e}");
}

const COARSE_FIELDS: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Fp8Gemm::emit(Fp8Spec { arch: spec.arch, epi: Fp8Epi::Set })
}
"#;

#[test]
fn fp8_spec_requires_every_field() {
    let e = peacemaker_rip::compile(COARSE_FIELDS, Arch::Gfx1201, "default").expect_err("no scale");
    assert!(e.contains("missing the field `scale`"), "{e}");
}

const MODULE: &str = r#"
fn module() -> (Vec<Emitted>, String, String) {
    Fp8Gemm::emit_module(Arch::Gfx1201, Fp8Scale::Row, [Fp8Epi::Set, Fp8Epi::Add, Fp8Epi::GateUpSilu, Fp8Epi::Qkv, Fp8Epi::Qkvza]).unwrap()
}
fn count() -> usize { let (a, _t, _p) = module(); a.len() }
fn text() -> String { let (_a, t, _p) = module(); t }
fn proof() -> String { let (_a, _t, p) = module(); p }
fn emit(spec: Spec) -> Result<Emitted, String> {
    let (all, text, proof) = Fp8Gemm::emit_module(spec.arch, Fp8Scale::Row, [Fp8Epi::Set, Fp8Epi::Add])?;
    assert!(text.contains(".amdgpu_metadata"));
    assert!(proof.contains("s_text_sha256"));
    Ok(all[1])
}
"#;

#[test]
fn fp8_gemm_emit_module_returns_the_rust_module() {
    let (emitted, text, proof) = fp8::emit_module(
        Arch::Gfx1201,
        ActScale::Row,
        &[Epi::Set, Epi::Add, Epi::GateUpSilu, Epi::Qkv, Epi::Qkvza],
    )
    .unwrap();
    let got = peacemaker_rip::compile(MODULE, Arch::Gfx1201, "default").unwrap();
    let two = fp8::emit_module(Arch::Gfx1201, ActScale::Row, &[Epi::Set, Epi::Add]).unwrap();
    assert_eq!(two.0[1].s_text, got.s_text);
    assert_eq!(
        peacemaker_rip::eval(MODULE, "count").unwrap(),
        emitted.len().to_string()
    );
    assert_eq!(
        peacemaker_rip::eval(MODULE, "text").unwrap(),
        format!("{text:?}")
    );
    let json = serde_json::to_string(&proof).unwrap();
    assert_eq!(
        peacemaker_rip::eval(MODULE, "proof").unwrap(),
        format!("{json:?}")
    );
}

// ---- compositions the script can change -------------------------------------------

const ROW_SET: &str = "ratio-256x128x8-row.set";
const K128_SET: &str = "ratio-256x128x8-k128.set";

fn build(src: &str, variant: &str) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(src, "variation.rip", Arch::Gfx1201, variant)
}

fn edit(from: &str, to: &str) -> String {
    assert!(TWIN.contains(from), "the twin no longer contains `{from}`");
    TWIN.replacen(from, to, 1)
}

/// A timing experiment: start the final block's weight refill a unit later is not expressible, but
/// dropping the delay-ALU hints is, and changes the bytes.
#[test]
fn a_variation_without_delay_alu_builds_and_differs() {
    let src = edit("    b.enable_delay_alu();\n", "");
    let base = build(TWIN, ROW_SET).unwrap();
    let varied = build(&src, ROW_SET).unwrap();
    assert!(base.s_text.contains("s_delay_alu"));
    assert!(varied.s_text.lines().count() < base.s_text.lines().count());
    assert!(!varied.s_text.contains("s_delay_alu"));
}

/// Moving a ring statement is checked by the builder: reading slot 1 for the second slab before
/// the barrier that publishes it is refused by the slot machine.
#[test]
fn reading_a_slot_before_it_is_published_is_refused() {
    let src = edit(
        "    for u in 0..2 {\n        for st in 0..2 {\n            step(b, g, u, st, first && u == 0 && st == 0, 0)?;",
        "    for u in 0..2 {\n        for st in 0..2 {\n            step(b, g, u, st, first && u == 0 && st == 0, 1)?;",
    );
    let e = build(&src, ROW_SET).expect_err("slab 0 read from the unpublished slot");
    assert!(e.contains("A1"), "{e}");
}

/// A barrier over a slot the block never stored is refused.
#[test]
fn publishing_an_unwritten_slot_is_refused() {
    let src = edit(
        "    let mut b1 = vec![retire(0), ready(1)];",
        "    let mut b1 = vec![retire(0), ready(1), ready(4)];",
    );
    let e = build(&src, ROW_SET).expect_err("D0 was never stored");
    assert!(e.contains("invalid LDS transition Ready(4)"), "{e}");
}

/// The split barrier's ring state: dropping the retire of A1 leaves the slot Published, so the next
/// block's publish into A1 is refused by the slot machine (`A1 cannot be stored while Reading`).
#[test]
fn dropping_the_retire_of_a_slot_is_refused_at_its_next_store() {
    let src = edit(
        "        let mut b2 = vec![retire(1), retire(2 + p), ready(0)];",
        "        let mut b2 = vec![retire(2 + p), ready(0)];",
    );
    let e = build(&src, ROW_SET).expect_err("A1 stays published");
    assert!(e.contains("A1 cannot be stored while Reading"), "{e}");
}

#[test]
fn the_k128_only_natives_refuse_a_row_spec() {
    let src = edit(
        "    let k128 = g.spec().scale == Fp8Scale::K128;\n    let p = if first { 0 } else { 1 };",
        "    let k128 = true;\n    let p = if first { 0 } else { 1 };",
    );
    let e = build(&src, ROW_SET).expect_err("a D ratio in the row kernel");
    assert!(
        e.contains("only the K128-scaled kernel has a D ratio"),
        "{e}"
    );
    build(&src, K128_SET).unwrap();
}

#[test]
fn block_indices_are_checked() {
    let src = edit(
        "fp8::mma(b, g, unit, st, first, i)?;",
        "fp8::mma(b, g, unit, st, first, i + 16)?;",
    );
    let e = build(&src, ROW_SET).expect_err("wmma index 16");
    assert!(e.contains("mma_insn"), "{e}");
}
