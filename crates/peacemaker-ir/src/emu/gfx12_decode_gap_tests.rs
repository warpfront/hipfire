// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 decode-table gaps found by lifting hipcc/clang code objects with `railgun-cert` (every instruction below
//! made the lifter stop with `undefined Gfx1201 opcode/form`): `v_add_f32_e64` (VOP3 form with source modifiers),
//! `s_min_num_f32`/`s_max_num_f32` (SOP2 float min/max), `v_ashrrev_i64` (VOP3 64-bit arithmetic shift) and the
//! VOP3 64-bit integer compares into a lane mask (`v_cmp_{lt,le,eq,ne,gt,ge}_{i64,u64}_e64`).
//!
//! Every word is what `llvm-mc --arch=amdgcn --mcpu=gfx1201 --show-encoding` assembles from the text in its comment.
//! Each is decoded with the production gfx1201 decoder, re-encoded byte-identically (also rebuilt from typed fields
//! alone), and executed through the emulator's dispatch entry. Every expected value is derived by hand from the
//! RDNA4 ISA pseudocode rules (never from the implementation under test); each test carries the edge lanes a wrong
//! semantic fails: signed zero, infinity, overflow, denormal, INT64_MIN/MAX, 64-bit high-word-only differences,
//! shift counts 0/31/63/64+ and execution-mask holes.
use super::{alu, Result, State};
use crate::{
    codec::gfx12,
    effects::ImplicitSet,
    inst::Wave,
    operand::Omod,
    provenance::Provenance,
    reg::{Kind, RegRef},
    Arch, Inst,
};

fn state() -> State {
    State {
        s: [0; 106],
        t: [0; 16],
        v: Box::new([[0; 64]; 256]),
        lanes: 32,
        nan_loose: false,
        exec: 1,
        vcc: 0,
        scc: false,
        m0: 0,
        pc: 0,
        ended: false,
        barrier: None,
        signals: 0,
        scratch: Vec::new(),
    }
}

/// The instruction and its table name, decoded from machine words by the production gfx1201 decoder.
fn dec(words: &[u32]) -> (Inst, &'static str) {
    let (inst, n) = gfx12::decode(words).unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"));
    assert_eq!(n, words.len(), "{words:08x?} decoded to {n} words");
    let name = crate::isa::lookup(Arch::Gfx1201, inst.op, inst.form)
        .unwrap_or_else(|| panic!("{words:08x?} has no gfx1201 row"))
        .name;
    (inst, name)
}

/// Decode, check the table name, and prove both byte-identical re-encodings (decoded provenance and typed fields).
fn roundtrip(words: &[u32], want_name: &str) -> Inst {
    let (inst, name) = dec(words);
    assert_eq!(name, want_name, "{words:08x?}");
    assert_eq!(
        gfx12::encode(&inst).unwrap().as_slice(),
        words,
        "{want_name}: re-encode differs"
    );
    let rebuilt = Inst::from_parts(
        Arch::Gfx1201,
        inst.op,
        inst.form,
        inst.fields.clone(),
        inst.operands.clone(),
        inst.mods.clone(),
        inst.literal,
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(
        gfx12::encode(&rebuilt).unwrap().as_slice(),
        words,
        "{want_name}: typed rebuild differs"
    );
    assert_eq!(gfx12::decode(words).unwrap().0, inst);
    inst
}

/// Run an ALU instruction through the emulator's dispatch; its name must have a semantics row of class `alu`.
fn run(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("alu"), "{name}");
    alu::execute(Arch::Gfx1201, name, &inst, st)
}

fn pair64(st: &mut State, base: usize, lane: usize, x: u64) {
    st.v[base][lane] = x as u32;
    st.v[base + 1][lane] = (x >> 32) as u32;
}

fn get64(st: &State, base: usize, lane: usize) -> u64 {
    u64::from(st.v[base][lane]) | u64::from(st.v[base + 1][lane]) << 32
}

fn spair(st: &mut State, base: usize, x: u64) {
    st.s[base] = x as u32;
    st.s[base + 1] = (x >> 32) as u32;
}

fn reg(kind: Kind, base: u16, len: u8) -> RegRef {
    RegRef { kind, base, len }
}

// ---- v_add_f32_e64 ---------------------------------------------------------------------------------------------

/// `v_add_f32_e64 v4, |v3|, |v3|` (the Raon situ kernel word).
const ADD_ABS_ABS: &[u32] = &[0xd503_0304, 0x0202_0703];
/// `v_add_f32_e64 v4, -|v3|, v5`
const ADD_NEGABS_V: &[u32] = &[0xd503_0104, 0x2202_0b03];
/// `v_add_f32_e64 v4, v3, -|v5|`
const ADD_V_NEGABS: &[u32] = &[0xd503_0204, 0x4202_0b03];
/// `v_add_f32_e64 v4, -v3, v5`
const ADD_NEG_V: &[u32] = &[0xd503_0004, 0x2202_0b03];
/// `v_add_f32_e64 v4, v3, v5 clamp`
const ADD_CLAMP: &[u32] = &[0xd503_8004, 0x0202_0b03];
/// `v_add_f32_e64 v4, v3, v5 mul:2`
const ADD_MUL2: &[u32] = &[0xd503_0004, 0x0a02_0b03];

#[test]
fn v_add_f32_e64_decodes_with_exact_operands_and_roundtrips() {
    for words in [
        ADD_ABS_ABS,
        ADD_NEGABS_V,
        ADD_V_NEGABS,
        ADD_NEG_V,
        ADD_CLAMP,
        ADD_MUL2,
    ] {
        roundtrip(words, "v_add_f32_e64");
    }
    let inst = roundtrip(ADD_ABS_ABS, "v_add_f32_e64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::V, 4, 1)]);
    assert_eq!(
        inst.effects.uses.as_slice(),
        &[reg(Kind::V, 3, 1), reg(Kind::V, 3, 1)]
    );
    // abs on both sources (VOP3 word0 bits 8..=9), no neg, no clamp, no omod: the modifiers are decoded, not dropped.
    assert_eq!((inst.mods.abs, inst.mods.neg), (0b11, 0));
    assert!(!inst.mods.clamp);
    assert_eq!(inst.mods.omod, Omod::None);
    let inst = roundtrip(ADD_NEGABS_V, "v_add_f32_e64");
    assert_eq!((inst.mods.abs, inst.mods.neg), (0b01, 0b01));
    let inst = roundtrip(ADD_V_NEGABS, "v_add_f32_e64");
    assert_eq!((inst.mods.abs, inst.mods.neg), (0b10, 0b10));
    let inst = roundtrip(ADD_NEG_V, "v_add_f32_e64");
    assert_eq!((inst.mods.abs, inst.mods.neg), (0, 0b01));
    assert!(roundtrip(ADD_CLAMP, "v_add_f32_e64").mods.clamp);
    assert_eq!(roundtrip(ADD_MUL2, "v_add_f32_e64").mods.omod, Omod::Mul2);
}

/// `D = |S0| + |S1|` with both sources `v3`: 2*|x|. Hand values: `|-0| + |-0| = +0`, `|-1.5| * 2 = 3.0`,
/// `|±inf| * 2 = +inf`, `|-max| * 2` overflows to +inf (RNE), `|-denorm_min| * 2 = 2 * denorm_min`.
#[test]
fn v_add_f32_e64_abs_sources() {
    let cases: [(u32, u32); 8] = [
        (0x8000_0000, 0x0000_0000), // -0 -> +0 (abs clears the sign of zero)
        (0x0000_0000, 0x0000_0000), // +0
        (0xbfc0_0000, 0x4040_0000), // -1.5 -> 3.0
        (0x7f80_0000, 0x7f80_0000), // +inf
        (0xff80_0000, 0x7f80_0000), // -inf -> +inf
        (0xff7f_ffff, 0x7f80_0000), // -FLT_MAX: 2 * FLT_MAX > FLT_MAX + half ulp -> +inf
        (0x8000_0001, 0x0000_0002), // -denorm_min -> +2*denorm_min (denormals preserved)
        (0x3f80_0000, 0x4000_0000), // 1.0 -> 2.0
    ];
    let mut st = state();
    st.exec = 0xff;
    for (lane, (x, _)) in cases.into_iter().enumerate() {
        st.v[3][lane] = x;
    }
    st.v[4][8] = 0x1234_5678; // inactive lane keeps its register
    run(ADD_ABS_ABS, &mut st).unwrap();
    for (lane, (x, want)) in cases.into_iter().enumerate() {
        assert_eq!(
            st.v[4][lane], want,
            "lane {lane}: |{x:#010x}| + |{x:#010x}|"
        );
    }
    assert_eq!(st.v[4][8], 0x1234_5678);
    // A NaN operand under `abs` is refused: the effect of the modifier on NaN sign/payload is not measured.
    let mut st = state();
    st.v[3][0] = 0x7fc0_0000;
    let e = run(ADD_ABS_ABS, &mut st).unwrap_err();
    assert!(e.contains("NaN operand"), "{e}");
    st.v[3][0] = 0xffc0_0001;
    assert!(run(ADD_ABS_ABS, &mut st).is_err());
}

/// `-|v3| + v5`, `v3 + -|v5|`, `-v3 + v5`: modifier order is abs first, neg after, per source.
#[test]
fn v_add_f32_e64_neg_abs_combinations() {
    let mut st = state();
    st.exec = 0x3f;
    // (v3, v5, -|v3| + v5)
    let negabs: [(u32, u32, u32); 6] = [
        (0xc000_0000, 0x40a0_0000, 0x4040_0000), // -|-2| + 5 = 3
        (0x4000_0000, 0x40a0_0000, 0x4040_0000), // -|2| + 5 = 3 (abs makes the sign irrelevant)
        (0x0000_0000, 0x8000_0000, 0x8000_0000), // -|+0| + -0 = -0 + -0 = -0
        (0x8000_0000, 0x0000_0000, 0x0000_0000), // -|-0| + +0 = -0 + +0 = +0 (RNE)
        (0x7f80_0000, 0x7f80_0000, 0xffc0_0000), // -inf + inf: invalid, the measured default NaN
        (0x3f80_0000, 0x3f80_0000, 0x0000_0000), // -1 + 1 = +0
    ];
    for (lane, (a, b, _)) in negabs.into_iter().enumerate() {
        st.v[3][lane] = a;
        st.v[5][lane] = b;
    }
    run(ADD_NEGABS_V, &mut st).unwrap();
    for (lane, (a, b, want)) in negabs.into_iter().enumerate() {
        assert_eq!(st.v[4][lane], want, "lane {lane}: -|{a:#x}| + {b:#x}");
    }
    // v3 + -|v5|
    run(ADD_V_NEGABS, &mut st).unwrap();
    // lane 0: -2 + -|5| = ... v3=-2.0, v5=5.0 -> -2 - 5 = -7
    assert_eq!(st.v[4][0], 0xc0e0_0000);
    // lane 1: 2 + -|5| = -3
    assert_eq!(st.v[4][1], 0xc040_0000);
    // lane 5: v3 = 1, v5 = 1 -> 1 + -1 = +0
    assert_eq!(st.v[4][5], 0x0000_0000);
    // -v3 + v5 (neg without abs): lane 2: -(+0) + -0 = -0; lane 3: -(-0) + +0 = +0 + +0 = +0
    run(ADD_NEG_V, &mut st).unwrap();
    assert_eq!(st.v[4][0], 0x40e0_0000); // -(-2) + 5 = 7
    assert_eq!(st.v[4][2], 0x8000_0000);
    assert_eq!(st.v[4][3], 0x0000_0000);
}

/// CLAMP saturates the result to [0, 1] (a negative result and -0 give +0); OMOD `mul:2` doubles the result.
#[test]
fn v_add_f32_e64_clamp_and_omod_are_honored() {
    let mut st = state();
    st.exec = 0b1111;
    let cases: [(u32, u32, u32); 4] = [
        (0x3f40_0000, 0x3f40_0000, 0x3f80_0000), // 0.75 + 0.75 = 1.5 -> 1.0
        (0xbf80_0000, 0x3f00_0000, 0x0000_0000), // -1 + 0.5 = -0.5 -> +0
        (0x3e80_0000, 0x3e80_0000, 0x3f00_0000), // 0.25 + 0.25 = 0.5 (inside the range)
        (0x8000_0000, 0x8000_0000, 0x0000_0000), // -0 + -0 = -0 -> +0
    ];
    for (lane, (a, b, _)) in cases.into_iter().enumerate() {
        st.v[3][lane] = a;
        st.v[5][lane] = b;
    }
    run(ADD_CLAMP, &mut st).unwrap();
    for (lane, (_, _, want)) in cases.into_iter().enumerate() {
        assert_eq!(st.v[4][lane], want, "clamp lane {lane}");
    }
    // 1.0 + 1.5 = 2.5; mul:2 -> 5.0
    st.v[3][0] = 0x3f80_0000;
    st.v[5][0] = 0x3fc0_0000;
    run(ADD_MUL2, &mut st).unwrap();
    assert_eq!(st.v[4][0], 0x40a0_0000);
}

// ---- s_min_num_f32 / s_max_num_f32 ------------------------------------------------------------------------------

/// `s_min_num_f32 s16, s16, 0` (the Raon RDA prepare word)
const S_MIN_S16_0: &[u32] = &[0xa110_8010];
/// `s_max_num_f32 s16, s16, 0`
const S_MAX_S16_0: &[u32] = &[0xa190_8010];
/// `s_min_num_f32 s0, s1, 1.0`
const S_MIN_S1_ONE: &[u32] = &[0xa100_f201];
/// `s_max_num_f32 s0, s1, -2.0`
const S_MAX_S1_NEG2: &[u32] = &[0xa180_f501];
/// `s_min_num_f32 s0, s1, 0x40490fdb`
const S_MIN_S1_LIT: &[u32] = &[0xa100_ff01, 0x4049_0fdb];

#[test]
fn s_min_max_num_f32_decode_exact_operands_and_roundtrip() {
    for (words, name) in [
        (S_MIN_S16_0, "s_min_num_f32"),
        (S_MAX_S16_0, "s_max_num_f32"),
        (S_MIN_S1_ONE, "s_min_num_f32"),
        (S_MAX_S1_NEG2, "s_max_num_f32"),
        (S_MIN_S1_LIT, "s_min_num_f32"),
    ] {
        roundtrip(words, name);
    }
    let inst = roundtrip(S_MIN_S16_0, "s_min_num_f32");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::S, 16, 1)]);
    assert_eq!(inst.effects.uses.as_slice(), &[reg(Kind::S, 16, 1)]);
    // SOP2 float min/max do not touch SCC.
    assert_eq!(inst.effects.implicit, ImplicitSet::default());
}

/// Hand values: minNum/maxNum of `x` and `+0.0` with the ISA's `-0 < +0` rule; one operand is returned bit for bit.
#[test]
fn s_min_max_num_f32_ordering_and_signed_zero() {
    // (s16, min(s16, 0), max(s16, 0))
    let cases: [(u32, u32, u32); 10] = [
        (0x3f80_0000, 0x0000_0000, 0x3f80_0000), // 1.0
        (0xbf80_0000, 0xbf80_0000, 0x0000_0000), // -1.0
        (0x0000_0000, 0x0000_0000, 0x0000_0000), // +0 vs +0
        (0x8000_0000, 0x8000_0000, 0x0000_0000), // -0 vs +0: min = -0, max = +0
        (0x7f80_0000, 0x0000_0000, 0x7f80_0000), // +inf
        (0xff80_0000, 0xff80_0000, 0x0000_0000), // -inf
        (0x0000_0001, 0x0000_0000, 0x0000_0001), // +denorm_min stays unflushed
        (0x8000_0001, 0x8000_0001, 0x0000_0000), // -denorm_min stays unflushed
        (0x7f7f_ffff, 0x0000_0000, 0x7f7f_ffff), // +FLT_MAX
        (0xff7f_ffff, 0xff7f_ffff, 0x0000_0000), // -FLT_MAX
    ];
    for scc in [false, true] {
        for (x, min, max) in cases {
            let mut st = state();
            st.scc = scc;
            st.s[16] = x;
            run(S_MIN_S16_0, &mut st).unwrap();
            assert_eq!(st.s[16], min, "s_min_num_f32 {x:#010x}, 0");
            assert_eq!(st.scc, scc, "s_min_num_f32 must not touch SCC");
            st.s[16] = x;
            run(S_MAX_S16_0, &mut st).unwrap();
            assert_eq!(st.s[16], max, "s_max_num_f32 {x:#010x}, 0");
            assert_eq!(st.scc, scc, "s_max_num_f32 must not touch SCC");
        }
    }
}

#[test]
fn s_min_max_num_f32_inline_float_and_literal_sources() {
    let mut st = state();
    // min(2.0, 1.0) = 1.0, min(0.5, 1.0) = 0.5
    st.s[1] = 0x4000_0000;
    run(S_MIN_S1_ONE, &mut st).unwrap();
    assert_eq!(st.s[0], 0x3f80_0000);
    st.s[1] = 0x3f00_0000;
    run(S_MIN_S1_ONE, &mut st).unwrap();
    assert_eq!(st.s[0], 0x3f00_0000);
    // max(-3.0, -2.0) = -2.0, max(-1.0, -2.0) = -1.0
    st.s[1] = 0xc040_0000;
    run(S_MAX_S1_NEG2, &mut st).unwrap();
    assert_eq!(st.s[0], 0xc000_0000);
    st.s[1] = 0xbf80_0000;
    run(S_MAX_S1_NEG2, &mut st).unwrap();
    assert_eq!(st.s[0], 0xbf80_0000);
    // min(4.0, pi_literal) = pi_literal, min(3.0, pi_literal) = 3.0
    st.s[1] = 0x4080_0000;
    run(S_MIN_S1_LIT, &mut st).unwrap();
    assert_eq!(st.s[0], 0x4049_0fdb);
    st.s[1] = 0x4040_0000;
    run(S_MIN_S1_LIT, &mut st).unwrap();
    assert_eq!(st.s[0], 0x4040_0000);
}

/// The scalar unit's NaN selection, quieting and payload are not measured: any NaN operand (quiet, signalling,
/// negative, in either position) is refused, never guessed.
#[test]
fn s_min_max_num_f32_refuse_nan_operands() {
    for nan in [
        0x7fc0_0000u32,
        0xffc0_0000,
        0x7fa0_0000,
        0x7f80_0001,
        0xffff_ffff,
    ] {
        let mut st = state();
        st.s[16] = nan;
        for words in [S_MIN_S16_0, S_MAX_S16_0] {
            let e = run(words, &mut st).unwrap_err();
            assert!(e.contains("NaN operand"), "{nan:#x}: {e}");
            assert_eq!(st.s[16], nan, "a refused instruction writes nothing");
        }
        // NaN as the second source (literal).
        let mut st = state();
        st.s[1] = 0x3f80_0000;
        let words = [S_MIN_S1_LIT[0], nan];
        assert!(run(&words, &mut st).is_err(), "{nan:#x} as S1");
    }
}

// ---- v_ashrrev_i64 -----------------------------------------------------------------------------------------------

/// `v_ashrrev_i64 v[2:3], 31, v[2:3]` (the Raon gather word)
const ASHR_V23_31: &[u32] = &[0xd73e_0002, 0x0202_049f];
/// `v_ashrrev_i64 v[0:1], v4, v[2:3]`
const ASHR_V01_V4: &[u32] = &[0xd73e_0000, 0x0202_0504];
/// `v_ashrrev_i64 v[0:1], 63, -1`
const ASHR_V01_63_M1: &[u32] = &[0xd73e_0000, 0x0201_82bf];

#[test]
fn v_ashrrev_i64_decodes_with_pair_operands_and_roundtrips() {
    for words in [ASHR_V23_31, ASHR_V01_V4, ASHR_V01_63_M1] {
        roundtrip(words, "v_ashrrev_i64");
    }
    let inst = roundtrip(ASHR_V23_31, "v_ashrrev_i64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::V, 2, 2)]);
    assert_eq!(inst.effects.uses.as_slice(), &[reg(Kind::V, 2, 2)]);
    let inst = roundtrip(ASHR_V01_V4, "v_ashrrev_i64");
    assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::V, 0, 2)]);
    assert_eq!(
        inst.effects.uses.as_slice(),
        &[reg(Kind::V, 4, 1), reg(Kind::V, 2, 2)]
    );
}

/// D.i64 = S1.i64 >> S0[5:0] (arithmetic): count 0, 31, 63, and 64+ (bits above 5:0 ignored).
#[test]
fn v_ashrrev_i64_masks_the_count_and_fills_with_the_sign() {
    const MIN: u64 = 0x8000_0000_0000_0000;
    const MAX: u64 = 0x7fff_ffff_ffff_ffff;
    // (count in v4, data, hand-computed result)
    let cases: [(u32, u64, u64); 12] = [
        (0, MIN | 1, MIN | 1),            // count 0: unchanged
        (31, MIN, 0xffff_ffff_0000_0000), // INT64_MIN >> 31 = -2^32
        (63, MIN, u64::MAX),              // sign fill: -1
        (63, MAX, 0),                     // positive: 0
        (64, MIN | 1, MIN | 1),           // 64 & 63 = 0
        (65, MIN, 0xc000_0000_0000_0000), // 65 & 63 = 1
        (127, MIN, u64::MAX),             // 127 & 63 = 63
        (0xffff_ffc0, 0x8000_0000_dead_beef, 0x8000_0000_dead_beef), // bits above 5:0 ignored
        (4, 0x0123_4567_89ab_cdef, 0x0012_3456_789a_bcde), // positive, plain shift
        (4, 0xf123_4567_89ab_cdef, 0xff12_3456_789a_bcde), // negative: top nibble filled with ones
        (32, 0x0000_0001_0000_0000, 1),   // crosses the dword boundary exactly
        (1, u64::MAX, u64::MAX),          // -1 stays -1
    ];
    let mut st = state();
    st.exec = 0xfff;
    for (lane, (count, data, _)) in cases.into_iter().enumerate() {
        st.v[4][lane] = count;
        pair64(&mut st, 2, lane, data);
    }
    pair64(&mut st, 0, 12, 0x1111_2222_3333_4444); // inactive lane
    run(ASHR_V01_V4, &mut st).unwrap();
    for (lane, (count, data, want)) in cases.into_iter().enumerate() {
        assert_eq!(
            get64(&st, 0, lane),
            want,
            "lane {lane}: ashr({data:#x}, {count})"
        );
    }
    assert_eq!(get64(&st, 0, 12), 0x1111_2222_3333_4444);
}

/// The in-place form (destination aliases the source pair) with the inline count 31, plus the inline-constant
/// source: the inline integer -1 is sign-extended to 64 bits before the shift.
#[test]
fn v_ashrrev_i64_in_place_and_inline_constant_source() {
    let cases: [(u64, u64); 5] = [
        (0x8000_0000_0000_0000, 0xffff_ffff_0000_0000), // MIN >> 31
        (0x0000_0001_8000_0000, 3),                     // (2^32 + 2^31) >> 31 = 3
        (u64::MAX, u64::MAX),                           // -1
        (0x7fff_ffff_ffff_ffff, 0x0000_0000_ffff_ffff), // MAX >> 31 = 2^32 - 1
        (0xffff_ffff_8000_0000, 0xffff_ffff_ffff_ffff), // -2^31 >> 31 = -1
    ];
    let mut st = state();
    st.exec = 0x1f;
    for (lane, (data, _)) in cases.into_iter().enumerate() {
        pair64(&mut st, 2, lane, data);
    }
    run(ASHR_V23_31, &mut st).unwrap();
    for (lane, (data, want)) in cases.into_iter().enumerate() {
        assert_eq!(get64(&st, 2, lane), want, "lane {lane}: {data:#x} >> 31");
    }
    let mut st = state();
    st.exec = 1;
    run(ASHR_V01_63_M1, &mut st).unwrap();
    assert_eq!(get64(&st, 0, 0), u64::MAX);
}

#[test]
fn v_ashrrev_i64_refuses_float_modifiers() {
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.abs = 1;
    let mut st = state();
    assert!(alu::execute(Arch::Gfx1201, name, &inst, &mut st).is_err());
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.neg = 2;
    assert!(alu::execute(Arch::Gfx1201, name, &inst, &mut st).is_err());
    let (mut inst, name) = dec(ASHR_V01_V4);
    inst.mods.clamp = true;
    assert!(alu::execute(Arch::Gfx1201, name, &inst, &mut st).is_err());
}

// ---- VOP3 64-bit integer compares into a lane mask ---------------------------------------------------------------

/// `v_cmp_ge_i64_e64 s26, s[20:21], s[28:29]`
const CMP_GE_I64_S26: &[u32] = &[0xd456_001a, 0x0200_3814];
/// `v_cmp_ge_i64_e64 s2, s[40:41], s[24:25]`
const CMP_GE_I64_S2: &[u32] = &[0xd456_0002, 0x0200_3028];

#[test]
fn v_cmp_i64_e64_decodes_with_mask_dest_and_pair_sources() {
    for (words, dst, a, b) in [
        (CMP_GE_I64_S26, 26u16, 20u16, 28u16),
        (CMP_GE_I64_S2, 2, 40, 24),
    ] {
        let inst = roundtrip(words, "v_cmp_ge_i64_e64");
        // The table keeps the 64-bit mask slot; a wave32 consumer narrows it to one SGPR, wave64 keeps the pair.
        assert_eq!(inst.effects.defs.as_slice(), &[reg(Kind::S, dst, 2)]);
        assert_eq!(
            inst.effects.uses.as_slice(),
            &[reg(Kind::S, a, 2), reg(Kind::S, b, 2)]
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1201, Wave::Wave32, &inst, 0),
            Some(reg(Kind::S, dst, 1))
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1201, Wave::Wave64, &inst, 0),
            Some(reg(Kind::S, dst, 2))
        );
        assert_eq!(
            gfx12::operand_register(Arch::Gfx1201, Wave::Wave32, &inst, 1),
            Some(reg(Kind::S, a, 2))
        );
    }
}

/// Signed 64-bit `>=` around INT64_MIN/MAX and for differences confined to the high dword. The scalar-pair
/// sources are uniform, so every active lane gets the same bit; inactive lanes write zero.
#[test]
fn v_cmp_ge_i64_e64_signed_edges_on_scalar_pairs() {
    const MIN: u64 = 0x8000_0000_0000_0000;
    const MAX: u64 = 0x7fff_ffff_ffff_ffff;
    // (S0 = s[20:21], S1 = s[28:29], S0 >= S1 as signed)
    let cases: [(u64, u64, bool); 10] = [
        (MIN, MIN, true),
        (MIN, MAX, false),
        (MAX, MIN, true),
        (MAX, MAX, true),
        (u64::MAX, 0, false), // -1 >= 0: false (unsigned would say true)
        (0, u64::MAX, true),  // 0 >= -1
        (0x0000_0001_0000_0000, 0x0000_0000_ffff_ffff, true), // high dword decides
        (0x0000_0000_ffff_ffff, 0x0000_0001_0000_0000, false),
        (MIN | 1, MIN, true),
        (MIN, MIN | 1, false),
    ];
    for (a, b, want) in cases {
        let mut st = state();
        st.exec = 0b1011; // lane 2 inactive
        st.s[26] = 0xffff_ffff; // stale bits must be replaced
        spair(&mut st, 20, a);
        spair(&mut st, 28, b);
        run(CMP_GE_I64_S26, &mut st).unwrap();
        assert_eq!(
            st.s[26],
            if want { 0b1011 } else { 0 },
            "{a:#x} >= {b:#x} (signed)"
        );
    }
    // The sibling word with other registers: s2 = (s[40:41] >= s[24:24+1]).
    let mut st = state();
    st.exec = 0xffff_ffff;
    spair(&mut st, 40, MIN);
    spair(&mut st, 24, MAX);
    run(CMP_GE_I64_S2, &mut st).unwrap();
    assert_eq!(st.s[2], 0);
    spair(&mut st, 40, MAX);
    spair(&mut st, 24, MIN);
    run(CMP_GE_I64_S2, &mut st).unwrap();
    assert_eq!(st.s[2], 0xffff_ffff);
    // Nothing else is written.
    assert_eq!((st.s[3], st.s[26]), (0, 0));
}

#[test]
fn v_cmp_ge_i64_e64_wave64_writes_the_sgpr_pair() {
    let mut st = state();
    st.lanes = 64;
    st.exec = u64::MAX;
    spair(&mut st, 20, 5);
    spair(&mut st, 28, 5);
    run(CMP_GE_I64_S26, &mut st).unwrap();
    assert_eq!((st.s[26], st.s[27]), (0xffff_ffff, 0xffff_ffff));
    st.exec = 0x8000_0000_0000_0001;
    run(CMP_GE_I64_S26, &mut st).unwrap();
    assert_eq!((st.s[26], st.s[27]), (1, 0x8000_0000));
    spair(&mut st, 28, 6);
    run(CMP_GE_I64_S26, &mut st).unwrap();
    assert_eq!((st.s[26], st.s[27]), (0, 0));
}

/// `v_cmp_<p>_<ty>_e64 s0, v[0:1], v[2:3]`: opcode (VOP3 word0 bits 16..=25) and the shared word1 `0x02020500`.
const CMP_I64: [(&str, u32); 6] = [
    ("lt", 0x51),
    ("eq", 0x52),
    ("le", 0x53),
    ("gt", 0x54),
    ("ne", 0x55),
    ("ge", 0x56),
];
const CMP_U64: [(&str, u32); 6] = [
    ("lt", 0x59),
    ("eq", 0x5a),
    ("le", 0x5b),
    ("gt", 0x5c),
    ("ne", 0x5d),
    ("ge", 0x5e),
];

/// Lane operand pairs `(S0, S1)`: equal, ordered small values, INT64_MIN vs MAX, -1 vs 0, a high-dword-only win,
/// a high-dword win against a larger low dword, equal high / differing low dwords, equal INT64_MIN.
const PAIRS: [(u64, u64); 8] = [
    (0, 0),
    (1, 2),
    (0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff),
    (0xffff_ffff_ffff_ffff, 0),
    (0x0000_0001_0000_0000, 0x0000_0000_ffff_ffff),
    (0x0000_0001_0000_0005, 0x0000_0002_0000_0001),
    (0x7fff_ffff_0000_0000, 0x7fff_ffff_0000_0001),
    (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
];

/// Hand-derived result masks, bit `l` = lane `l` (pair `PAIRS[l]`).
/// Signed: lt {1,2,3,5,6}, eq {0,7}, le = lt+eq, gt {4}, ne = !eq, ge = gt+eq.
/// Unsigned: lt {1,5,6}, eq {0,7}, le = lt+eq, gt {2,3,4}, ne = !eq, ge = gt+eq.
fn want_mask(signed: bool, pred: &str) -> u64 {
    let eq = 0b1000_0001u64;
    let (lt, gt) = if signed {
        (0b0110_1110u64, 0b0001_0000u64)
    } else {
        (0b0110_0010u64, 0b0001_1100u64)
    };
    match pred {
        "lt" => lt,
        "le" => lt | eq,
        "eq" => eq,
        "ne" => !eq & 0xff,
        "gt" => gt,
        "ge" => gt | eq,
        _ => unreachable!(),
    }
}

#[test]
fn v_cmp_i64_u64_e64_family_per_lane_masks() {
    for (signed, ty, ops) in [(true, "i64", CMP_I64), (false, "u64", CMP_U64)] {
        for (pred, opcode) in ops {
            let words = [0xd400_0000 | opcode << 16, 0x0202_0500];
            let name = format!("v_cmp_{pred}_{ty}_e64");
            let inst = roundtrip(&words, &name);
            assert_eq!(
                inst.effects.defs.as_slice(),
                &[reg(Kind::S, 0, 2)],
                "{name}"
            );
            assert_eq!(
                inst.effects.uses.as_slice(),
                &[reg(Kind::V, 0, 2), reg(Kind::V, 2, 2)],
                "{name}"
            );
            let mut st = state();
            st.exec = 0xff;
            st.s[0] = 0xffff_ffff;
            for (lane, (a, b)) in PAIRS.into_iter().enumerate() {
                pair64(&mut st, 0, lane, a);
                pair64(&mut st, 2, lane, b);
            }
            // Lanes 8.. are inactive even though their operands would satisfy `ne`.
            for lane in 8..32 {
                pair64(&mut st, 0, lane, 1);
                pair64(&mut st, 2, lane, 2);
            }
            run(&words, &mut st).unwrap();
            assert_eq!(
                u64::from(st.s[0]),
                want_mask(signed, pred),
                "{name}: got {:#010b}",
                st.s[0]
            );
        }
    }
}

/// The five new compare rows exist only in the gfx1201 table (and agree with the llvm-mc text pinned there).
#[test]
fn new_i64_compare_rows_pin_their_llvm_text() {
    let rows: [(&str, &str, &[u32]); 8] = [
        (
            "v_cmp_eq_i64_e64",
            "v_cmp_eq_i64_e64 s0, s[2:3], s[4:5]",
            &[0xd452_0000u32, 0x0200_0802],
        ),
        (
            "v_cmp_le_i64_e64",
            "v_cmp_le_i64_e64 s0, s[2:3], s[4:5]",
            &[0xd453_0000, 0x0200_0802],
        ),
        (
            "v_cmp_ne_i64_e64",
            "v_cmp_ne_i64_e64 s0, s[2:3], s[4:5]",
            &[0xd455_0000, 0x0200_0802],
        ),
        (
            "v_cmp_le_u64_e64",
            "v_cmp_le_u64_e64 s0, s[2:3], s[4:5]",
            &[0xd45b_0000, 0x0200_0802],
        ),
        (
            "v_cmp_ge_u64_e64",
            "v_cmp_ge_u64_e64 s0, s[2:3], s[4:5]",
            &[0xd45e_0000, 0x0200_0802],
        ),
        ("v_add_f32_e64", "v_add_f32_e64 v4, |v3|, |v3|", ADD_ABS_ABS),
        ("s_min_num_f32", "s_min_num_f32 s16, s16, 0", S_MIN_S16_0),
        (
            "v_ashrrev_i64",
            "v_ashrrev_i64 v[2:3], 31, v[2:3]",
            ASHR_V23_31,
        ),
    ];
    for (name, text, words) in rows {
        let inst = roundtrip(words, name);
        let row = crate::isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
        let table: Vec<u32> = row
            .encoding
            .split_whitespace()
            .map(|w| u32::from_str_radix(w, 16).unwrap())
            .collect();
        assert_eq!((row.sample, table.as_slice()), (text, words), "{name}");
        // v_add_f32_e64 predates this work (gfx1100 has it at the same encoding); the others are gfx1201-table only.
        if name == "v_add_f32_e64" {
            continue;
        }
        for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1030] {
            let e = gfx12::decode_for(arch, words).unwrap_err().to_string();
            assert!(e.contains("undefined"), "{arch:?} {name}: {e}");
        }
    }
}
