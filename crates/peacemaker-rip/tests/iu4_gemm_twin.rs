// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of `iu4_gemm::emit` (`kernels/iu4_gemm.rip`,
//! composed from the `iu4::*` block natives) re-emits the production kernel's
//! bytes for every valid `Spec` on gfx1201. It is the oracle that keeps the
//! `.rip` composition honest; the coarse natives (`Iu4Gemm::emit`,
//! `MoeSym::emit`) return the Rust functions' values.

use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Cacc, Epi, Fold, Mma, Spec, Tile};
use peacemaker_isa::kernels::qwen4_moe_sym as moe;
use peacemaker_isa::Arch;

const TWIN: &str = include_str!("../../peacemaker-isa/src/kernels/iu4_gemm.rip");

fn every_spec() -> Vec<Spec> {
    let mut out = Vec::new();
    for fold in [Fold::K128, Fold::K256Shared, Fold::K256Pow2] {
        for tile in [Tile::T128x128x8, Tile::T256x128x16] {
            for cacc in [Cacc::One, Cacc::Two] {
                for epi in [
                    Epi::Set,
                    Epi::Add,
                    Epi::AddSigmoid,
                    Epi::GateUpSilu,
                    Epi::GateUpSiluBf16,
                    Epi::QkvzaGdn,
                ] {
                    for act in [ALayout::Token, ALayout::Slab] {
                        for mma in [Mma::Dense, Mma::S24X64, Mma::S24X32] {
                            out.push(Spec {
                                fold,
                                tile,
                                cacc,
                                epi,
                                act,
                                arch: Arch::Gfx1201,
                                mma,
                                tune: Default::default(),
                            });
                        }
                    }
                }
            }
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

/// `s_text` is byte-identical for every valid spec. The proof is compared too:
/// it records the builder's facts about the program (variant, symbol, resource
/// and ledger digests), which a `.rip` composition of the same calls
/// reproduces; a difference would mean the composition emitted another program.
#[test]
fn iu4_gemm_twin_matches_the_rust_emit() {
    let mut valid = 0;
    for spec in every_spec() {
        let name = spec.variant_name();
        if spec.validate().is_err() {
            // The script refuses what the Rust generator refuses, with its message.
            let e = peacemaker_rip::compile_named(TWIN, "iu4_gemm.rip", spec.arch, &name)
                .err()
                .unwrap_or_else(|| panic!("{name}: the twin built an invalid spec"));
            let want = spec.validate().unwrap_err();
            assert!(e.contains(&want), "{name}: {e} does not carry `{want}`");
            continue;
        }
        valid += 1;
        let rust = iu4_gemm::emit(spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = peacemaker_rip::compile_named(TWIN, "iu4_gemm.rip", spec.arch, &name)
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
        valid, 18,
        "the number of valid specs changed: update VARIANTS"
    );
}

#[test]
fn the_twin_declares_exactly_the_valid_variants() {
    let declared = peacemaker_rip::variants(TWIN, "iu4_gemm.rip").unwrap();
    let mut valid: Vec<String> = every_spec()
        .into_iter()
        .filter(|s| s.validate().is_ok())
        .map(|s| s.variant_name())
        .collect();
    let mut declared = declared;
    valid.sort();
    declared.sort();
    assert_eq!(declared, valid);
}

#[test]
fn a_malformed_variant_is_refused_with_its_grammar() {
    for (v, want) in [
        ("k128", "expected FOLD-TILE-CACC[-slab]-EPI"),
        ("k64-128x128x8-cacc1-set", "unknown fold `k64`"),
        ("k128-64x64x4-cacc1-set", "unknown tile `64x64x4`"),
        (
            "k128-128x128x8-cacc3-set",
            "unknown accumulator count `cacc3`",
        ),
        ("k128-128x128x8-cacc1-slab", "missing the epilogue"),
        ("k128-128x128x8-cacc1-mul", "unknown epilogue `mul`"),
    ] {
        let e = peacemaker_rip::compile_named(TWIN, "iu4_gemm.rip", Arch::Gfx1201, v)
            .err()
            .unwrap_or_else(|| panic!("{v} built"));
        assert!(e.contains(want), "{v}: {e}");
    }
}

#[test]
fn the_twin_builds_on_gfx1201_only() {
    let e = peacemaker_rip::compile_named(
        TWIN,
        "iu4_gemm.rip",
        Arch::Gfx1151,
        "k128-128x128x8-cacc1-set",
    )
    .expect_err("gfx1151 must be refused");
    assert!(e.contains("iu4_gemm targets gfx1201 only"), "{e}");
}

// ---- coarse natives ---------------------------------------------------------

const COARSE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    let s = Iu4Spec {
        fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::GateUpSilu, act: ALayout::Slab, arch: spec.arch, mma: Mma::Dense,
    };
    Iu4Gemm::emit(s)
}
"#;

#[test]
fn iu4_gemm_emit_native_returns_the_rust_emitted() {
    let want = iu4_gemm::emit(Spec {
        fold: Fold::K128,
        tile: Tile::T128x128x8,
        cacc: Cacc::One,
        epi: Epi::GateUpSilu,
        act: ALayout::Slab,
        arch: Arch::Gfx1201,
        mma: Mma::Dense,
        tune: Default::default(),
    })
    .unwrap();
    let got = peacemaker_rip::compile(COARSE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(want.s_text, got.s_text);
    // Spec::validate stays the gate.
    let e = peacemaker_rip::compile(COARSE, Arch::Gfx1151, "default")
        .err()
        .unwrap();
    assert!(e.contains("iu4_gemm targets gfx1201 only"), "{e}");
}

const COARSE_FIELDS: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Iu4Gemm::emit(Iu4Spec { fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, arch: spec.arch, mma: Mma::Dense })
}
"#;

#[test]
fn iu4_spec_requires_every_field() {
    let e = peacemaker_rip::compile(COARSE_FIELDS, Arch::Gfx1201, "default")
        .err()
        .unwrap();
    assert!(e.contains("missing the field `act`"), "{e}");
}

const MODULE: &str = r#"
fn module() -> (Vec<Emitted>, String, String) {
    Iu4Gemm::emit_module(Fold::K128, Tile::T128x128x8, Cacc::One, ALayout::Token, Arch::Gfx1201).unwrap()
}
fn emit(spec: Spec) -> Result<Emitted, String> {
    let (all, text, proof) = Iu4Gemm::emit_module(Fold::K128, Tile::T128x128x8, Cacc::One, ALayout::Token, spec.arch)?;
    assert!(text.contains(".amdgpu_metadata"));
    assert!(proof.contains("s_text_sha256"));
    Ok(all[0])
}
fn count() -> usize { let (a, _t, _p) = module(); a.len() }
fn text() -> String { let (_a, t, _p) = module(); t }
fn proof() -> String { let (_a, _t, p) = module(); p }
"#;

#[test]
fn iu4_gemm_emit_module_returns_the_rust_module() {
    let (emitted, text, proof) = iu4_gemm::emit_module(
        Fold::K128,
        Tile::T128x128x8,
        Cacc::One,
        ALayout::Token,
        Arch::Gfx1201,
    )
    .unwrap();
    let got = peacemaker_rip::compile(MODULE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(emitted[0].s_text, got.s_text);
    assert_eq!(
        peacemaker_rip::eval(MODULE, "count").unwrap(),
        emitted.len().to_string()
    );
    // The module text and its proof (as JSON) come back unchanged.
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

const MOE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    MoeSym::emit(MoeSymSpec { arch: spec.arch, kind: MoeKind::GateUp, nt: 1, rr: 1, codec: WeightCodec::Mq4Sym })
}
"#;

#[test]
fn moe_sym_emit_native_returns_the_rust_emitted() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        let want = moe::emit(moe::Spec {
            arch,
            kind: moe::Kind::GateUp,
            nt: 1,
            rr: 1,
            codec: moe::WeightCodec::Mq4Sym,
        })
        .unwrap();
        let got = peacemaker_rip::compile(MOE, arch, "default").unwrap();
        assert_eq!(want.s_text, got.s_text);
    }
}

const MOE_MODULE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    let (all, _text, _proof) = MoeSym::emit_module(spec.arch)?;
    let (codec_all, _t, _p) = MoeSym::emit_module_codec(spec.arch, WeightCodec::Mq4Sym)?;
    assert_eq!(all.len(), codec_all.len());
    Ok(all[0])
}
"#;

#[test]
fn moe_sym_emit_module_natives_return_the_rust_modules() {
    let (emitted, _, _) = moe::emit_module(Arch::Gfx1201).unwrap();
    let got = peacemaker_rip::compile(MOE_MODULE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(emitted[0].s_text, got.s_text);
}

// ---- compositions the script can change -------------------------------------------

const SET: &str = "k128-128x128x8-cacc1-set";

fn build(src: &str, variant: &str) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(src, "variation.rip", Arch::Gfx1201, variant)
}

fn edit(from: &str, to: &str) -> String {
    assert!(TWIN.contains(from), "the twin no longer contains `{from}`");
    TWIN.replacen(from, to, 1)
}

/// A timing experiment: run the K-loop at the default wave priority.
#[test]
fn a_variation_without_the_k_loop_priority_builds_and_differs() {
    let src = edit("    op(wg.isa(), \"s_setprio 1\", vec![], vec![])?;\n", "");
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert_eq!(base.s_text.matches("s_setprio 1").count(), 1);
    assert_eq!(varied.s_text.matches("s_setprio 1").count(), 0);
    assert_eq!(varied.s_text.matches("s_setprio 0").count(), 1);
}

/// A different trip structure: a `K == 256` specialization drops the trip
/// compare, the skip and the rolled loop and runs only the peeled pair.
#[test]
fn a_k256_specialization_drops_the_rolled_loop() {
    let k256 = r#"
fn kloop_k256<T: Target>(wg: &mut Wg<T>, g: &Iu4Gen, lds: Lds) -> Result<Lds, String> {
    wg.label(iu4::K_BEGIN)?;
    let lds = iu4::block(wg, g, "t0", 0, true, iu4::label("t1"), true, lds)?;
    iu4::block(wg, g, "t1", 1, false, iu4::EPI, true, lds)
}
"#;
    let src = format!(
        "{}{k256}",
        edit(
            "let lds = kloop(wg, g, lds)?;",
            "let lds = kloop_k256(wg, g, lds)?;"
        )
    );
    let base = build(TWIN, SET).unwrap();
    let varied = build(&src, SET).unwrap();
    assert!(base.s_text.contains(".Liu4_k_loop:"));
    assert!(!varied.s_text.contains(".Liu4_k_loop:"));
    assert!(varied.s_text.lines().count() < base.s_text.lines().count());
}

// ---- the typed bridge refuses what the driver refuses -------------------------------

#[test]
fn a_ring_of_the_wrong_kind_fails_the_tag_check() {
    // The weight ring where the activation ring is expected.
    let src = edit(
        "    let lds = kloop(wg, g, lds)?;\n",
        "    let lds = [lds[1], lds[0], lds[2], lds[3]];\n    let lds = kloop(wg, g, lds)?;\n",
    );
    let e = build(&src, SET).expect_err("the swapped rings must be refused");
    assert!(e.contains("different tag"), "{e}");
}

#[test]
fn a_block_under_wave_uniform_control_is_refused() {
    let src = edit(
        "    iu4::epilogue(wg, g, end)?;\n",
        "    let t = g.reg(\"trips\").unwrap();\n    let c = wg.scmp(Instruction::new(format!(\"s_cmp_lg_u32 s{t}, 0\"), vec![], vec![s(t)]))?;\n    wg.skip_if(c, \".Lwave\", |w| iu4::trip_end(w, g))?;\n    iu4::epilogue(wg, g, end)?;\n",
    );
    let e = build(&src, SET).expect_err("a wave-scope block must be refused");
    assert!(e.contains("typed is workgroup scope"), "{e}");
}

#[test]
fn a_ring_the_script_made_cannot_enter_a_block() {
    let src = edit(
        "    let lds = kloop(wg, g, lds)?;\n",
        "    let r0 = wg.lds(\"R0\", 40000, 64)?;\n    let r1 = wg.lds(\"R1\", 40064, 64)?;\n    let own = Ring::new(r0, r1);\n    let lds = kloop(wg, g, [own, lds[1], lds[2], lds[3]])?;\n",
    );
    let e = build(&src, SET).expect_err("a script-made ring must be refused");
    assert!(e.contains("was not given by a typed body"), "{e}");
}

#[test]
fn trips_cmp_takes_only_scalar_compares() {
    let src = edit("\"s_cmp_eq_u32\"", "\"s_mov_b32\"");
    let e = build(&src, SET).expect_err("a non-compare must be refused");
    assert!(e.contains("is not a 32-bit scalar compare"), "{e}");
}

#[test]
fn a_block_must_run_on_the_plane_the_rings_are_on() {
    // The odd plane first: the rings are out of phase with it.
    let src = edit("\"m0\", 0, true", "\"m0\", 1, true");
    let e = build(&src, SET).expect_err("an out-of-phase block must be refused");
    assert!(e.contains("rings out of phase with plane 1"), "{e}");
}

// ---- the sparse blocks -------------------------------------------------------------

const SPARSE_SET: &str = "k128-128x128x8-cacc1-set-s24x64";

/// The block family follows the generator's spec: the dense `iu4::block` on a sparse generator, and the sparse block
/// on a dense one, are refused with the family named.
#[test]
fn a_block_family_that_does_not_match_the_spec_is_refused() {
    let dense_on_sparse = edit(
        "    if g.spec().mma == Mma::Dense {\n        iu4::block(wg, g, tag, h, fetch_next, end_label, entry_label, lds)",
        "    if true {\n        iu4::block(wg, g, tag, h, fetch_next, end_label, entry_label, lds)",
    );
    let e = build(&dense_on_sparse, SPARSE_SET).expect_err("a dense block on a sparse spec");
    assert!(e.contains("use the sparse blocks"), "{e}");
    build(&dense_on_sparse, SET).unwrap();

    let sparse_on_dense = edit(
        "    if g.spec().mma == Mma::Dense {\n        iu4::block(wg, g, tag, h, fetch_next, end_label, entry_label, lds)",
        "    if false {\n        iu4::block(wg, g, tag, h, fetch_next, end_label, entry_label, lds)",
    );
    let e = build(&sparse_on_dense, SET).expect_err("a sparse block on a dense spec");
    assert!(e.contains("use the dense blocks"), "{e}");
    build(&sparse_on_dense, SPARSE_SET).unwrap();
}

#[test]
fn a_sparse_variant_the_generator_does_not_build_is_refused_with_its_axis() {
    for (v, want) in [
        ("k128-256x128x16-cacc1-set-s24x64", "needs Tile::T128x128x8"),
        (
            "k128-128x128x8-cacc1-slab-set-s24x32",
            "needs ALayout::Token",
        ),
        (
            "k128-128x128x8-cacc1-silu-s24x64",
            "supports Epi::Set and Epi::Add only",
        ),
        ("k256s-128x128x8-cacc1-set-s24x32", "needs Fold::K128"),
    ] {
        let e = build(TWIN, v).expect_err(v);
        assert!(e.contains(want), "{v}: {e}");
    }
}

const SPARSE_FIELDS: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Iu4Gemm::emit(Iu4Spec { fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Set, act: ALayout::Token, arch: spec.arch })
}
"#;

#[test]
fn iu4_spec_needs_mma() {
    let e = peacemaker_rip::compile(SPARSE_FIELDS, Arch::Gfx1201, "default").expect_err("no mma");
    assert!(e.contains("missing the field `mma`"), "{e}");
}

const SPARSE_COARSE: &str = r#"
fn emit(spec: Spec) -> Result<Emitted, String> {
    Iu4Gemm::emit(Iu4Spec { fold: Fold::K128, tile: Tile::T128x128x8, cacc: Cacc::One,
        epi: Epi::Add, act: ALayout::Token, arch: spec.arch, mma: Mma::S24X32 })
}
"#;

#[test]
fn iu4_gemm_emit_native_takes_the_sparse_mma() {
    let want = iu4_gemm::emit(Spec {
        epi: Epi::Add,
        mma: Mma::S24X32,
        ..Spec::control(Epi::Add)
    })
    .unwrap();
    let got = peacemaker_rip::compile(SPARSE_COARSE, Arch::Gfx1201, "default").unwrap();
    assert_eq!(want.s_text, got.s_text);
    assert!(got.s_text.contains("v_swmmac_i32_16x16x32_iu4"));
}
