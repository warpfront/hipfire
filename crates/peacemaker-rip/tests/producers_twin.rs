// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twins of the PM producers (`iu4_c8_producer.rip`, `moe_rotate_i4_pm.rip`,
//! `gdn_chunk_prep.rip`, composed from the `c8::*` / `rot::*` / `gdn::*` natives over the public
//! `blocks` of the Rust generators) re-emit the generators' bytes: `s_text` and the builder proof for
//! every export, and the module text assembled from the twin's exports.

use peacemaker_isa::kernels::iu4_gemm;
use peacemaker_isa::kernels::{gdn_chunk_prep, iu4_c8_producer, moe_rotate_i4_pm};
use peacemaker_isa::{Arch, Emitted};

const C8: &str = include_str!("../../peacemaker-isa/src/kernels/iu4_c8_producer.rip");
const ROT: &str = include_str!("../../peacemaker-isa/src/kernels/moe_rotate_i4_pm.rip");
const GDN: &str = include_str!("../../peacemaker-isa/src/kernels/gdn_chunk_prep.rip");
const ARCH: Arch = Arch::Gfx1201;

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

fn same(name: &str, rust: &Emitted, rip: &Emitted) {
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

fn rip(src: &str, file: &str, variant: &str) -> Emitted {
    peacemaker_rip::compile_named(src, file, ARCH, variant)
        .unwrap_or_else(|e| panic!("{variant}: {e}"))
}

#[test]
fn c8_twin_matches_the_rust_emit() {
    let mut twin = Vec::new();
    for x in iu4_c8_producer::module_exports() {
        let r = iu4_c8_producer::emit(ARCH, x).unwrap();
        let t = rip(C8, "iu4_c8_producer.rip", x.tag());
        same(x.tag(), &r, &t);
        twin.push(t);
    }
    let (_, text, _) = iu4_c8_producer::emit_module(ARCH).unwrap();
    let (rip_text, _) = iu4_gemm::module(&twin, iu4_c8_producer::MODULE).unwrap();
    assert_eq!(text, rip_text, "module text");
}

#[test]
fn rotate_twin_matches_the_rust_emit() {
    let spec = moe_rotate_i4_pm::Spec { arch: ARCH };
    same(
        "rows",
        &moe_rotate_i4_pm::emit(spec).unwrap(),
        &rip(ROT, "moe_rotate_i4_pm.rip", "rows"),
    );
    same(
        "gp",
        &moe_rotate_i4_pm::emit_gp(spec).unwrap(),
        &rip(ROT, "moe_rotate_i4_pm.rip", "gp"),
    );
}

#[test]
fn gdn_twin_matches_the_rust_emit() {
    let a3b = gdn_chunk_prep::GdnChunkPrepSpec::A3B;
    let r = gdn_chunk_prep::emit(ARCH, &a3b).unwrap();
    let t = rip(GDN, "gdn_chunk_prep.rip", "chunk_prep");
    same("chunk_prep", &r, &t);
    let (_, text, _) = gdn_chunk_prep::emit_module(ARCH, &a3b).unwrap();
    let (rip_text, _) = iu4_gemm::module(&[t], gdn_chunk_prep::MODULE).unwrap();
    assert_eq!(text, rip_text, "module text");
}

#[test]
fn each_twin_declares_exactly_its_variants() {
    for (src, file, want) in [
        (
            C8,
            "iu4_c8_producer.rip",
            vec!["norm", "norm_out", "rotate", "sigmoid_rotate"],
        ),
        (ROT, "moe_rotate_i4_pm.rip", vec!["gp", "rows"]),
        (GDN, "gdn_chunk_prep.rip", vec!["chunk_prep"]),
    ] {
        let mut got = peacemaker_rip::variants(src, file).unwrap();
        got.sort();
        assert_eq!(got, want, "{file}");
    }
}

#[test]
fn malformed_variants_and_other_arches_are_refused() {
    for (src, file, v, want) in [
        (
            C8,
            "iu4_c8_producer.rip",
            "rotate2",
            "expected rotate, norm",
        ),
        (ROT, "moe_rotate_i4_pm.rip", "flat", "expected rows or gp"),
        (GDN, "gdn_chunk_prep.rip", "x", "expected chunk_prep"),
    ] {
        let e = peacemaker_rip::compile_named(src, file, ARCH, v)
            .err()
            .unwrap_or_else(|| panic!("{v} built"));
        assert!(e.contains(want), "{v}: {e}");
    }
    for (src, file, v) in [
        (C8, "iu4_c8_producer.rip", "rotate"),
        (ROT, "moe_rotate_i4_pm.rip", "rows"),
        (GDN, "gdn_chunk_prep.rip", "chunk_prep"),
    ] {
        let e = peacemaker_rip::compile_named(src, file, Arch::Gfx1151, v)
            .err()
            .unwrap_or_else(|| panic!("{file} built on gfx1151"));
        assert!(e.contains("gfx1201"), "{file}: {e}");
    }
}

/// The cookbook edit: issue the norm's phase-2 prefetch after the sum of squares instead of before it
/// (a pure scheduling change: every result bit is the same). The builder either accepts the new order
/// and emits a different program, or its wait ledger refuses it; it never emits the same bytes.
#[test]
fn moving_the_prefetch_behind_the_sum_of_squares_changes_the_program() {
    let edited = C8.replace(
        "    c8::norm_prefetch(b)?;\n    c8::norm_sumsq(b)?;\n",
        "    c8::norm_sumsq(b)?;\n    c8::norm_prefetch(b)?;\n",
    );
    assert_ne!(edited, C8, "the norm phases must exist in the twin");
    for tag in ["norm", "norm_out"] {
        let base = rip(C8, "iu4_c8_producer.rip", tag);
        let moved = peacemaker_rip::compile_named(&edited, "edited.rip", ARCH, tag);
        match moved {
            Ok(m) => assert_ne!(base.s_text, m.s_text, "{tag}"),
            Err(e) => panic!("{tag}: the edit was refused: {e}"),
        }
    }
}

/// The norm's LDS exchange is the script's: reading the partials before the barrier publishes them is
/// refused by the builder's slot machine.
#[test]
fn reading_the_norm_partials_before_the_barrier_is_refused() {
    let edited = C8.replace(
        "    b.barrier_with_scope([ready(0)], \"lds_only\")?;\n    c8::norm_reduce_addr(b, g)?;\n    for insn in c8::norm_load_insns() {\n        b.ds_load(0, insn)?;\n    }\n",
        "    c8::norm_reduce_addr(b, g)?;\n    for insn in c8::norm_load_insns() {\n        b.ds_load(0, insn)?;\n    }\n    b.barrier_with_scope([ready(0)], \"lds_only\")?;\n",
    );
    assert_ne!(edited, C8);
    let e = peacemaker_rip::compile_named(&edited, "edited.rip", ARCH, "norm")
        .err()
        .expect("the early read must be refused");
    assert!(!e.is_empty());
}
