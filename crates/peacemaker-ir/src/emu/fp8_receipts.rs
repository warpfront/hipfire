// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Regression tests of the gfx1201 FP8 attention capabilities (B2) against recorded hardware words: the E4M3 encoder
//! of `v_cvt_pk_fp8_f32` and the dense `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` arithmetic. The register-level
//! checks live next to their handlers (`wmma.rs`, `memory.rs`, `alu.rs`). The `#[ignore]`d replays re-check a whole
//! receipt directory written by `isa/emu/numerics-probe/fp8_probe.hip`:
//! `PM_FP8_RECEIPT_DIR=<receipt dir> cargo test --release -p peacemaker-ir --lib <name> -- --ignored`.

use super::{
    convert::f32_to_fp8,
    fp8_fixtures as fx,
    mma::{self, Fmt},
};
use crate::Arch;

pub(super) fn kind_formats(kind: u8) -> (Fmt, Fmt) {
    match kind {
        0 => (Fmt::Fp8, Fmt::Fp8),
        1 => (Fmt::Fp8, Fmt::Bf8),
        2 => (Fmt::Bf8, Fmt::Fp8),
        3 => (Fmt::Bf8, Fmt::Bf8),
        _ => panic!("unknown FP8/BF8 WMMA kind {kind}"),
    }
}

/// Hardware byte of input `x` according to the recorded run table.
fn table_byte(x: u32) -> u8 {
    let i = fx::CVT_RUNS.partition_point(|&(start, _)| start <= x) - 1;
    fx::CVT_RUNS[i].1
}

#[test]
fn f32_to_fp8_matches_the_hardware_run_table() {
    let mut checked = 0u64;
    for (i, &(start, byte)) in fx::CVT_RUNS.iter().enumerate() {
        let end = fx::CVT_RUNS
            .get(i + 1)
            .map_or(u64::MAX >> 32, |r| u64::from(r.0) - 1) as u32;
        // Both ends of every run, the neighbours across the boundary, and the middle.
        for x in [
            start,
            end,
            start.wrapping_sub(1),
            end.wrapping_add(1),
            start + (end - start) / 2,
        ] {
            assert_eq!(f32_to_fp8(x), table_byte(x), "x = {x:#010x}");
            checked += 1;
        }
    }
    // A fixed pseudo-random walk over all 2^32 patterns (xorshift32).
    let mut x = 0x2545_f491u32;
    for _ in 0..4_000_000 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        assert_eq!(f32_to_fp8(x), table_byte(x), "x = {x:#010x}");
        checked += 1;
    }
    assert!(checked > 4_000_000);
}

#[test]
fn f32_to_fp8_named_edges() {
    let bits = |f: f32| f.to_bits();
    for (x, want) in [
        (bits(0.0), 0x00),
        (bits(-0.0), 0x80),
        (bits(1.0), 0x38),
        (bits(-1.0), 0xb8),
        (bits(448.0), 0x7e),
        // 464 = midpoint of 448 and the NaN code 0x7f: ties to even (448); anything above is NaN, not saturation.
        (bits(464.0), 0x7e),
        (464.0f32.next_up().to_bits(), 0x7f),
        (bits(-464.0), 0xfe),
        (bits(1e30), 0x7f),
        (bits(-1e30), 0xff),
        (bits(f32::INFINITY), 0x7f),
        (bits(f32::NEG_INFINITY), 0xff),
        // Every f32 NaN gives 0xff, whatever its sign or payload.
        (0x7fc0_0000, 0xff),
        (0x7f80_0001, 0xff),
        (0xffc0_1234, 0xff),
        // Smallest denormal 2^-9 and its rounding boundary 2^-10 (tie to even: zero).
        (bits(2f32.powi(-9)), 0x01),
        (bits(2f32.powi(-10)), 0x00),
        (2f32.powi(-10).next_up().to_bits(), 0x01),
        (bits(-(2f32.powi(-10))), 0x80),
        // An f32 denormal is far below half the smallest FP8 denormal.
        (0x0000_0001, 0x00),
        (0x807f_ffff, 0x80),
    ] {
        assert_eq!(f32_to_fp8(x), want, "x = {x:#010x}");
    }
}

#[test]
fn wmma_f32_f8_matches_hardware_samples() {
    let mut tags = std::collections::BTreeSet::new();
    for c in &fx::WMMA_CASES {
        let (fa, fb) = kind_formats(c.kind);
        assert_eq!(
            mma::wmma_f32_f8(Arch::Gfx1201, &c.a, &c.b, c.c, fa, fb),
            c.d,
            "kind {} tag {} a {:02x?} b {:02x?} c {:#010x}",
            c.kind,
            c.tag,
            c.a,
            c.b,
            c.c
        );
        tags.insert((c.kind, c.tag));
    }
    // Every operand-format kind carries cancellation, subnormal, wide-range, NaN, infinity and single-product
    // (Cartesian: 0 * inf, inf * finite, NaN codes, extreme finite codes) samples; quad2/step2/xstep2 are the
    // two-product placements (same quad but different pairs / same step / different steps) that separate the
    // pairwise, quad-in-8-step and per-step structures.
    for kind in 0..4 {
        for tag in [
            "cancel0", "cart", "dense", "nan", "quad2", "step2", "sub", "wide", "xstep2",
        ] {
            assert!(tags.contains(&(kind, tag)), "kind {kind} tag {tag}");
        }
    }
}

/// All 2^32 inputs against the four recorded run tables (`cvt.s0_lo`, `cvt.s1_lo`, `cvt.s0_hi`, `cvt.s1_hi`), the
/// byte tables of the four sweeps agreeing with each other, with the seven other MODE settings and the two cross
/// sweeps (random pairs / random old VDST / random half select) reporting zero mismatches.
#[test]
#[ignore = "needs a hardware receipt directory (PM_FP8_RECEIPT_DIR)"]
fn f32_to_fp8_receipt_replay() {
    let dir = std::env::var("PM_FP8_RECEIPT_DIR").expect("PM_FP8_RECEIPT_DIR");
    let load = |name: &str| -> Vec<(u32, u32)> {
        let b = std::fs::read(format!("{dir}/{name}.bin")).unwrap();
        b.chunks_exact(8)
            .map(|c| {
                (
                    u32::from_le_bytes(c[0..4].try_into().unwrap()),
                    u32::from_le_bytes(c[4..8].try_into().unwrap()),
                )
            })
            .collect()
    };
    // Merged (start, byte) table of one sweep; `shift` selects the byte of the 32-bit vdst word.
    let bytes = |t: &[(u32, u32)], shift: u32| -> Vec<(u32, u8)> {
        let mut out: Vec<(u32, u8)> = Vec::new();
        for &(s, w) in t {
            let b = (w >> shift) as u8;
            if out.last().is_none_or(|l| l.1 != b) {
                out.push((s, b));
            }
        }
        out
    };
    let s0_lo = bytes(&load("cvt.s0_lo"), 0);
    // Word layout of every sweep: the other operand is 1.0 (0x38) and the unwritten half is zero.
    assert_eq!(
        s0_lo,
        bytes(&load("cvt.s1_lo"), 8),
        "src1 == src0 (low half)"
    );
    assert_eq!(
        s0_lo,
        bytes(&load("cvt.s0_hi"), 16),
        "src0 high half == low half"
    );
    assert_eq!(
        s0_lo,
        bytes(&load("cvt.s1_hi"), 24),
        "src1 high half == low half"
    );
    for (name, shift, want_other) in [
        ("cvt.s0_lo", 8, 0x38u32),
        ("cvt.s1_lo", 0, 0x38),
        ("cvt.s0_hi", 24, 0x38),
        ("cvt.s1_hi", 16, 0x38),
    ] {
        for (_, w) in load(name) {
            assert_eq!(
                (w >> shift) & 0xff,
                want_other,
                "{name}: other operand byte"
            );
            let mask = if name.ends_with("lo") {
                0xffff_0000u32
            } else {
                0x0000_ffff
            };
            assert_eq!(w & mask, 0, "{name}: untouched half");
        }
    }
    let mut checked = 0u64;
    let mut bad = 0u64;
    for (i, &(start, byte)) in s0_lo.iter().enumerate() {
        let end = s0_lo.get(i + 1).map_or(1u64 << 32, |r| u64::from(r.0));
        for x in u64::from(start)..end {
            checked += 1;
            if f32_to_fp8(x as u32) != byte {
                bad += 1;
            }
        }
    }
    assert_eq!(checked, 1 << 32);
    assert_eq!(bad, 0, "model != hardware over all 2^32 inputs");
    let base = std::fs::read(format!("{dir}/cvt.s0_lo.bin")).unwrap();
    for m in 1..8 {
        assert_eq!(
            std::fs::read(format!("{dir}/cvt.s0_lo.m{m}.bin")).unwrap(),
            base,
            "MODE index {m} changes the conversion"
        );
    }
    for r in 0..2 {
        let j = std::fs::read_to_string(format!("{dir}/cvt.cross.r{r}.json")).unwrap();
        assert!(
            j.contains("\"pairs\":4294967296") && j.contains("\"mismatches\":0,"),
            "{j}"
        );
    }
    println!("f32_to_fp8 replay: {checked} inputs, 0 mismatches; 4 sweeps, 7 MODE tables and 2 x 2^32 cross pairs identical");
}
