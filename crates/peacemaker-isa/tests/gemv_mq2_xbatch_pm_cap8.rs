// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ2 x-batch PM at capacity 8: layout closure of the register/ring plan, rejection of the tuples that do not
//! close, and a CPU byte oracle on the PeaceMaker emulator.
//!
//! Emulator domain: the emulator runs only hardware-qualified semantics (no `nan_meets_nan` relaxation). The f16
//! header words (scale / zero point, through `v_fma_mix_f32`) are therefore finite only: random normals plus signed
//! zeros, subnormals and extreme exponents, never half NaN/Inf (those are compared against the singleton on hardware
//! by the GPU oracle). Activations additionally carry f32 NaN payloads, Inf, zeros and denormals through the qualified
//! f32 mul/fmac/add paths, in columns 1, 4 and 7 only (one NaN makes its whole column NaN).
//!
//! Tuple subset: NOT every emitted tuple is executed. Per shape, either one tuple per (rows, split, ilp) at unroll 1
//! (`family_pick`) or every rows x unroll at anchor schedule / native loads (`loop_pick`, for the quad main and
//! remainder loops) is lifted and run; layout/legality is asserted for the whole grid separately.
//!
//! Oracle: for each executed tuple, `B = 1..=8` columns of one launch must equal, byte for byte, the single-column
//! (`B = 1`) launch over the same column (x pointer advanced by `c*K*4`), which in turn must equal the shipped
//! `r4_b8_u1_native_anchor` `B = 1` launch; no host reimplementation of the arithmetic. `B > 8` clamps to 8;
//! `B <= 0`, columns past `B`, rows past `M` and the guard words past `y` are never written.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201::{
    emit_bundle, emit_elf, emit_text, vgpr_top, Params, SHIPPED, SYMBOL, XBATCH_MAX,
};
use std::collections::BTreeMap;

const KARG: u64 = 0x1_0000;
const A_BASE: u64 = 0x100_0000;
const X_BASE: u64 = 0x200_0000;
const Y_BASE: u64 = 0x300_0000;
const SENTINEL: u32 = 0xdead_beef;
/// Guard words after `y` (`8*M` result words), checked untouched.
const GUARD: usize = 16;
const GROUP_BYTES: usize = 72;

fn cap8(rows: u8, unroll: u8, split: bool, ilp: bool) -> Params {
    Params {
        rows,
        cap: 8,
        unroll,
        split,
        ilp,
    }
}

/// Every grid point at `cap 8`, canonical order.
fn cap8_grid() -> Vec<Params> {
    let mut v = Vec::new();
    for rows in [1u8, 2, 4] {
        for unroll in [1u8, 2, 4] {
            for split in [false, true] {
                for ilp in [false, true] {
                    v.push(cap8(rows, unroll, split, ilp));
                }
            }
        }
    }
    v
}

/// The tuples the emitter's ring/register check must accept: all but `rows 4` with `quad_ILP`.
fn legal(p: &Params) -> bool {
    !(p.rows == 4 && p.ilp)
}

// ------------------------------------------------------------------------------------------------ layout

#[test]
fn xbatch_max_is_8_and_cap_grid_is_1_2_4_8() {
    assert_eq!(XBATCH_MAX, 8);
    for cap in [1u8, 2, 4, 8] {
        let p = Params { cap, ..SHIPPED };
        p.validate().unwrap();
    }
    for cap in [0u8, 3, 5, 6, 7, 9, 16] {
        let p = Params { cap, ..SHIPPED };
        assert!(p.validate().is_err(), "cap {cap}");
    }
}

#[test]
fn r4_b8_anchor_closes_with_six_ring_slots() {
    // r4: weights end at v80 (ring base), 6 ring slots (v80..v127), the 8-column accumulators fill v128..v255.
    let p = cap8(4, 1, false, false);
    assert_eq!(p.tag(), "r4_b8_u1_native_anchor");
    assert_eq!(vgpr_top(&p), 256);
    let text = emit_text(&p).expect("r4_b8 anchor must close");
    assert!(text.contains("v_mov_b32_e32 v128, 0"), "first accumulator v128");
    assert!(text.contains("v_mov_b32_e32 v255, 0"), "last accumulator v255");
    assert!(emit_bundle(&p).is_ok());
}

#[test]
fn r4_b8_ilp_one_ring_slot_is_rejected() {
    for unroll in [1u8, 2, 4] {
        for split in [false, true] {
            let p = cap8(4, unroll, split, true);
            let e = emit_text(&p).expect_err(&p.tag());
            assert!(
                e.contains("ring slots cannot hold a 4-item step"),
                "{}: {e}",
                p.tag()
            );
            assert!(emit_elf(&p).is_err() && emit_bundle(&p).is_err());
        }
    }
}

#[test]
fn cap8_legality_matches_the_ring_plan() {
    let mut ok = 0;
    for p in cap8_grid() {
        let r = emit_bundle(&p);
        assert_eq!(r.is_ok(), legal(&p), "{}: {:?}", p.tag(), r.as_ref().err());
        if r.is_ok() {
            ok += 1;
            assert!(vgpr_top(&p) <= 256, "{}", p.tag());
        }
    }
    assert_eq!(ok, 30);
}

#[test]
fn smaller_caps_are_unchanged_prefixes_of_the_grid() {
    // cap 8 adds columns 4..8 only: every pre-existing tuple still emits and the shipped tuple's top is unchanged.
    for cap in [1u8, 2, 4] {
        for rows in [1u8, 2, 4] {
            let p = Params {
                rows,
                cap,
                unroll: 1,
                split: false,
                ilp: false,
            };
            assert!(emit_bundle(&p).is_ok(), "{}", p.tag());
        }
    }
    assert!(vgpr_top(&SHIPPED) <= 256);
}

// ------------------------------------------------------------------------------------------------ oracle

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

const EDGE_F32: [u32; 13] = [
    0x7fc0_0000, // qNaN
    0xffc0_0000, // -qNaN
    0x7f80_0001, // sNaN
    0x7f80_0000, // +Inf
    0xff80_0000, // -Inf
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0x0000_0001, // denormal
    0x007f_ffff, // largest denormal
    0x7f7f_ffff, // max finite
    0xff7f_ffff,
    0x7fc1_2345, // NaN payload
];

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    /// 256-element groups of K
    g: usize,
    /// one element in `edge` of x / headers is an edge bit pattern (0: none)
    edge: u64,
    seed: u64,
}

struct Problem {
    shape: Shape,
    a: Vec<u8>,
    /// `[8][K]` f32 words
    x: Vec<u8>,
}

fn f16_word(rng: &mut Rng, edge: u64) -> u16 {
    let r = rng.next();
    let sign = (r & 1) as u16;
    let mant = (r >> 20 & 1023) as u16;
    let exp = if edge != 0 && r >> 8 & 0xff < 0x100 / edge {
        [0u16, 0, 1, 30, 30, 0][(r >> 40) as usize % 6] // signed zero / subnormal, extremes; never Inf/NaN
    } else {
        (8 + (r >> 8) % 14) as u16
    };
    sign << 15 | exp << 10 | mant
}

fn problem(shape: Shape) -> Problem {
    let mut rng = Rng(shape.seed | 1);
    let k = shape.g * 256;
    let mut a = Vec::with_capacity(shape.m * shape.g * GROUP_BYTES + 64);
    for _ in 0..shape.m * shape.g {
        for _ in 0..4 {
            a.extend(f16_word(&mut rng, shape.edge * 8).to_le_bytes());
        }
        for _ in 0..64 {
            a.push(rng.next() as u8);
        }
    }
    a.resize(a.len() + 64, 0xa5); // the last lane's dword spills 2 bytes past its group
    let mut x = Vec::with_capacity(8 * k * 4);
    for i in 0..8 * k {
        let r = rng.next();
        // Only columns 1, 4, 7 carry edge words: one NaN/Inf makes every row of its column NaN/Inf.
        let col = i / k;
        let bits = if shape.edge != 0 && col % 3 == 1 && r % shape.edge == 0 {
            EDGE_F32[(r >> 16) as usize % EDGE_F32.len()]
        } else {
            let e = 120 + (r >> 8) % 12;
            ((r & 1) << 31 | e << 23 | (r >> 20 & 0x7f_ffff)) as u32
        };
        x.extend(bits.to_le_bytes());
    }
    Problem { shape, a, x }
}

/// Lift one tuple's object and run workgroups on a fresh memory image.
struct Kernel {
    program: peacemaker_ir::Program,
    rows: usize,
}

fn compile(p: &Params) -> Kernel {
    let elf = emit_elf(p).unwrap_or_else(|e| panic!("{}: {e}", p.tag()));
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("{} lift: {e:?}", p.tag()))
    .program;
    Kernel {
        program,
        rows: usize::from(p.rows),
    }
}

impl Kernel {
    /// `y` words (`8*M`) after a launch of `b` columns over columns `col0..` of `x`.
    fn run(&self, pr: &Problem, col0: usize, b: i32, tag: &str) -> Vec<u32> {
        let Shape { m, g, .. } = pr.shape;
        let k = g * 256;
        let emu = Emulator::new(&self.program, SYMBOL)
            .unwrap_or_else(|e| panic!("{tag}: {e}"));
        let x_ptr = X_BASE + (col0 * k * 4) as u64;
        let mut karg = Vec::new();
        for p in [A_BASE, x_ptr, Y_BASE] {
            karg.extend(p.to_le_bytes());
        }
        for v in [m as i32, k as i32, b] {
            karg.extend(v.to_le_bytes());
        }
        karg.resize(40, 0);
        let y_words = 8 * m + GUARD;
        let y: Vec<u8> = (0..y_words).flat_map(|_| SENTINEL.to_le_bytes()).collect();
        let mut mem = Memory::default();
        mem.map("karg".into(), KARG, karg).unwrap();
        mem.map("a".into(), A_BASE, pr.a.clone()).unwrap();
        mem.map("x".into(), X_BASE, pr.x.clone()).unwrap();
        mem.map("y".into(), Y_BASE, y).unwrap();
        for wg in 0..m.div_ceil(self.rows) {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [wg as u32, 0, 0],
                    block: [32, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{tag} B={b} col0={col0} wg={wg}: {e}"));
        }
        mem.read(Y_BASE, 4 * y_words)
            .unwrap()
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }
}

/// The per-column reference: the shipped tuple's `B = 1` launch over each of the 8 columns.
fn reference(pr: &Problem) -> Vec<Vec<u32>> {
    let shipped = compile(&SHIPPED);
    let m = pr.shape.m;
    (0..8)
        .map(|c| {
            let y = shipped.run(pr, c, 1, "shipped");
            assert!(
                y[m..].iter().all(|&w| w == SENTINEL),
                "shipped B=1 wrote outside column 0"
            );
            y[..m].to_vec()
        })
        .collect()
}

fn check_tuple(p: &Params, pr: &Problem, refs: &[Vec<u32>], bs: &[i32]) {
    let tag = p.tag();
    let m = pr.shape.m;
    let k = compile(p);
    // B = 1 over every column equals the reference column.
    for (c, want) in refs.iter().enumerate() {
        let y = k.run(pr, c, 1, &tag);
        assert_eq!(&y[..m], &want[..], "{tag} m={m} g={} B=1 col {c}", pr.shape.g);
        assert!(y[m..].iter().all(|&w| w == SENTINEL), "{tag} B=1 col {c}: stray write");
    }
    for &b in bs {
        let y = k.run(pr, 0, b, &tag);
        let cols = b.clamp(0, 8) as usize;
        for c in 0..cols {
            assert_eq!(
                &y[c * m..(c + 1) * m],
                &refs[c][..],
                "{tag} m={m} g={} B={b} col {c}",
                pr.shape.g
            );
        }
        assert!(
            y[cols * m..].iter().all(|&w| w == SENTINEL),
            "{tag} m={m} g={} B={b}: wrote past column {cols} / row M / guard",
            pr.shape.g
        );
    }
}

fn sweep(shape: Shape, bs: &[i32], pick: fn(&Params) -> bool) {
    let pr = problem(shape);
    let refs = reference(&pr);
    // Columns must actually differ (a column-indexing slip cannot hide behind equal data).
    let distinct: std::collections::BTreeSet<&Vec<u32>> = refs.iter().collect();
    assert!(distinct.len() >= 3, "columns too alike: {}", distinct.len());
    for p in cap8_grid().into_iter().filter(|p| legal(p) && pick(p)) {
        check_tuple(&p, &pr, &refs, bs);
    }
}

/// One tuple per (rows, split, ilp), unroll 1.
fn family_pick(p: &Params) -> bool {
    p.unroll == 1
}

/// Every rows x unroll at the anchor schedule/native loads: exercises the main/remainder quad loops.
fn loop_pick(p: &Params) -> bool {
    !p.split && !p.ilp
}

const ALL_B: [i32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 0, -3];

/// M = 7 (row residue 3 at rows 4, 1 at rows 2), one quad plus two tail groups; edge bit patterns throughout.
#[test]
fn cap8_oracle_all_b_edge_patterns() {
    sweep(
        Shape {
            m: 7,
            g: 6,
            edge: 24,
            seed: 0x51,
        },
        &ALL_B,
        family_pick,
    );
}

/// Tail groups only (no quad loop), finite data.
#[test]
fn cap8_oracle_tail_only() {
    sweep(
        Shape {
            m: 5,
            g: 3,
            edge: 0,
            seed: 0x52,
        },
        &[8, 5, 1, 9],
        family_pick,
    );
}

/// Two quads plus three tail groups: unroll 2 main loop only, unroll 4 remainder loop only; sparse NaN/Inf.
#[test]
fn cap8_oracle_two_quads_three_tail() {
    sweep(
        Shape {
            m: 3,
            g: 11,
            edge: 97,
            seed: 0x53,
        },
        &[8, 3, 2],
        loop_pick,
    );
}

/// Five quads plus one tail group: unroll 4 main trip plus remainder, unroll 2 two trips plus remainder.
#[test]
fn cap8_oracle_five_quads_one_tail() {
    sweep(
        Shape {
            m: 2,
            g: 21,
            edge: 0,
            seed: 0x54,
        },
        &[8, 7, 4],
        loop_pick,
    );
}

/// Aligned quads (no tail), M one row past a multiple of 4.
#[test]
fn cap8_oracle_aligned_quads() {
    sweep(
        Shape {
            m: 9,
            g: 4,
            edge: 0,
            seed: 0x55,
        },
        &[8, 6, 3],
        family_pick,
    );
}

#[test]
fn byte_oracle_covers_every_legal_tuple_once() {
    let mut tags: BTreeMap<String, usize> = BTreeMap::new();
    for p in cap8_grid().into_iter().filter(legal) {
        *tags.entry(p.tag()).or_default() += 1;
    }
    assert_eq!(tags.len(), 30);
    assert!(tags.values().all(|&n| n == 1));
}
