// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Unit tests for the gfx1100 opcodes the pre-card emulator gate (`hip_loop --emu`) added so the hipcc
//! incumbents (MQ6 residual GEMM, HC 64x64 GEMM, QSA hg4 attention) and the round-1 candidates execute:
//! scalar `s_abs_i32`, `s_and_not1_*`, `s_lshlN_add_u32`; vector `v_or3_b32`, `v_mul_i32_i24`, `v_mov_b16`,
//! `v_cvt_f16_u16`, `v_fma_f16`, `v_fmac_f16`, `v_dual_max_f32`; wide `v_mad_u64_u32`, `v_mad_i64_i32`; the memory
//! rows `ds_load_2addr_b64`, `ds_store_2addr_b64`, `ds_load_i8`, `ds_load_u16_d16`, `global_load_d16_{hi_b16,i8,u8}`,
//! `buffer_load_i8`; and the no-op `wait` rows. Every expected value is derived from the ISA pseudocode by hand
//! (not from the implementation under test).
use super::{alu, convert, memory, Memory, Result, State};
use crate::{
    operand::{Half, ImmField, Modifiers, Operand, Special, VmemToken},
    reg::{Kind, RegRef},
    Arch, FormFields, Inst,
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
fn reg(kind: Kind, base: u16, len: u8) -> Operand {
    Operand::Reg(RegRef { kind, base, len })
}
fn v(base: u16) -> Operand {
    reg(Kind::V, base, 1)
}
fn s(base: u16) -> Operand {
    reg(Kind::S, base, 1)
}
fn half(base: u16, h: Half) -> Operand {
    Operand::Half(
        RegRef {
            kind: Kind::V,
            base,
            len: 1,
        },
        h,
    )
}
fn insn_with(name: &str, ops: Vec<Operand>, mods: Modifiers, fields: FormFields) -> Inst {
    let row = crate::isa::gfx1100()
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("{name} missing from the gfx1100 table"));
    Inst::from_parts(
        Arch::Gfx1100,
        row.op,
        row.form,
        fields,
        ops.into_iter().collect(),
        mods,
        None,
        Default::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: {e:?}"))
}
fn insn(name: &str, ops: Vec<Operand>) -> Inst {
    insn_with(name, ops, Modifiers::default(), FormFields::default())
}
fn run(name: &str, ops: Vec<Operand>, st: &mut State) -> Result<()> {
    alu::execute(Arch::Gfx1100, name, &insn(name, ops), st)
}
fn mem_run(
    name: &str,
    ops: Vec<Operand>,
    st: &mut State,
    mem: &mut Memory,
    lds: &mut [u8],
) -> Result<()> {
    memory::execute(name, &insn(name, ops), st, mem, lds)
}

#[test]
fn every_new_row_has_the_semantic_class_the_dispatch_expects() {
    for (name, class) in [
        ("buffer_gl0_inv", "wait"),
        ("s_sendmsg", "wait"),
        ("s_set_inst_prefetch_distance", "wait"),
        ("s_abs_i32", "alu"),
        ("s_and_not1_b32", "alu"),
        ("s_and_not1_saveexec_b32", "alu"),
        ("s_lshl2_add_u32", "alu"),
        ("s_lshl4_add_u32", "alu"),
        ("v_or3_b32", "alu"),
        ("v_mul_i32_i24_e32", "alu"),
        ("v_mov_b16_e32", "alu"),
        ("v_mov_b16_e64", "alu"),
        ("v_cvt_f16_u16_e32", "alu"),
        ("v_cvt_f16_f32_e64", "alu"),
        ("v_fma_f16", "alu"),
        ("v_fmac_f16_e32", "alu"),
        ("v_fmac_f16_e64", "alu"),
        ("v_dual_max_f32", "alu"),
        ("v_lshlrev_b64_e64", "alu"),
        ("v_mad_u64_u32", "alu"),
        ("v_mad_i64_i32", "alu"),
        ("v_cmp_lt_i32_e32", "alu"),
        ("v_cmpx_lt_f32_e32", "alu"),
        ("ds_load_2addr_b64", "memory"),
        ("ds_store_2addr_b64", "memory"),
        ("ds_load_i8", "memory"),
        ("ds_load_u16_d16", "memory"),
        ("global_load_d16_hi_b16", "memory"),
        ("global_load_d16_i8", "memory"),
        ("global_load_d16_u8", "memory"),
        ("buffer_load_i8", "memory"),
    ] {
        assert_eq!(super::semantic_class(name), Some(class), "{name}");
    }
}

#[test]
fn scalar_abs_and_not_and_shift_add() {
    let mut st = state();
    st.s[1] = (-7i32) as u32;
    run("s_abs_i32", vec![s(0), s(1)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (7, true));
    st.s[1] = 0;
    run("s_abs_i32", vec![s(0), s(1)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (0, false));
    st.s[1] = i32::MIN as u32; // wraps to itself
    run("s_abs_i32", vec![s(0), s(1)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (0x8000_0000, true));
    // D = S0 & ~S1, SCC = D != 0.
    st.s[1] = 0b1100;
    st.s[2] = 0b0110;
    run("s_and_not1_b32", vec![s(0), s(1), s(2)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (0b1000, true));
    st.s[2] = 0b1111;
    run("s_and_not1_b32", vec![s(0), s(1), s(2)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (0, false));
    // tmp = (S0 << N) + S1 in 64 bits; SCC = tmp >= 2^32.
    st.s[1] = 3;
    st.s[2] = 5;
    run("s_lshl4_add_u32", vec![s(0), s(1), s(2)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (3 * 16 + 5, false));
    st.s[1] = 0x1000_0000;
    st.s[2] = 7;
    run("s_lshl4_add_u32", vec![s(0), s(1), s(2)], &mut st).unwrap();
    assert_eq!((st.s[0], st.scc), (7, true), "2^32 + 7 wraps with carry");
    st.s[1] = 0x4000_0001;
    st.s[2] = 0xffff_fffe;
    run("s_lshl2_add_u32", vec![s(0), s(1), s(2)], &mut st).unwrap();
    // (0x4000_0001 << 2) = 0x1_0000_0004; + 0xffff_fffe = 0x2_0000_0002.
    assert_eq!((st.s[0], st.scc), (2, true));
}

#[test]
fn and_not1_saveexec_saves_exec_and_sets_exec_to_s0_and_not_exec() {
    let mut st = state();
    st.exec = 0b1010;
    st.s[1] = 0b0111;
    run("s_and_not1_saveexec_b32", vec![s(0), s(1)], &mut st).unwrap();
    assert_eq!(st.s[0], 0b1010, "old exec saved");
    assert_eq!(st.exec, 0b0101, "S0 & ~EXEC");
    assert!(st.scc);
    st.exec = 0b0111;
    st.s[1] = 0b0111;
    run("s_and_not1_saveexec_b32", vec![s(0), s(1)], &mut st).unwrap();
    assert_eq!((st.exec, st.scc), (0, false));
}

#[test]
fn or3_and_mul_i24() {
    let mut st = state();
    st.exec = 0b11;
    for (lane, (a, b, c)) in [(0x1u32, 0x20, 0x300), (0xf000_0000, 0x0f00_0000, 0)]
        .into_iter()
        .enumerate()
    {
        st.v[1][lane] = a;
        st.v[2][lane] = b;
        st.v[3][lane] = c;
    }
    run("v_or3_b32", vec![v(0), v(1), v(2), v(3)], &mut st).unwrap();
    assert_eq!([st.v[0][0], st.v[0][1]], [0x321, 0xff00_0000]);
    // S0.i24 * S1.i24, low 32 bits: only bits 23:0 of each source count, sign-extended from bit 23.
    st.v[1][0] = 0x00ff_ffff; // -1
    st.v[2][0] = 5;
    st.v[1][1] = 0xff00_0003; // 3 (bits above 23 ignored)
    st.v[2][1] = 0xab00_0005; // 5
    run("v_mul_i32_i24_e32", vec![v(0), v(1), v(2)], &mut st).unwrap();
    assert_eq!([st.v[0][0], st.v[0][1]], [(-5i32) as u32, 15]);
    // |-8388608 * -8388608| = 2^46: low 32 bits are 0.
    st.v[1][0] = 0x0080_0000;
    st.v[2][0] = 0x0080_0000;
    run("v_mul_i32_i24_e32", vec![v(0), v(1), v(2)], &mut st).unwrap();
    assert_eq!(st.v[0][0], 0);
}

#[test]
fn true16_mov_and_cvt_f16_u16_preserve_the_other_destination_half() {
    let mut st = state();
    st.v[1][0] = 0x1234_5678;
    st.v[0][0] = 0xabcd_ef01;
    run(
        "v_mov_b16_e32",
        vec![half(0, Half::Lo), half(1, Half::Hi)],
        &mut st,
    )
    .unwrap();
    assert_eq!(st.v[0][0], 0xabcd_1234);
    run(
        "v_mov_b16_e32",
        vec![half(0, Half::Hi), half(1, Half::Lo)],
        &mut st,
    )
    .unwrap();
    assert_eq!(st.v[0][0], 0x5678_1234);
    // u16 -> f16, one rounding to nearest even: 1 -> 0x3c00, 2049 -> tie between 2048 and 2050 -> 2048 (0x6800),
    // 2051 -> 2052 (0x6802), 65504 exact (0x7bff), 65520 and up round to infinity (0x7c00).
    for (x, h) in [
        (0u32, 0u16),
        (1, 0x3c00),
        (2049, 0x6800),
        (2051, 0x6802),
        (65504, 0x7bff),
        (65519, 0x7bff),
        (65520, 0x7c00),
        (65535, 0x7c00),
    ] {
        st.v[1][0] = x | 0xdead_0000;
        st.v[0][0] = 0xabcd_0000;
        run(
            "v_cvt_f16_u16_e32",
            vec![half(0, Half::Lo), half(1, Half::Lo)],
            &mut st,
        )
        .unwrap();
        assert_eq!(st.v[0][0], 0xabcd_0000 | u32::from(h), "u16 {x}");
    }
    // High source half, high destination half.
    st.v[1][0] = (7u32 << 16) | 0xffff;
    st.v[0][0] = 0x0000_beef;
    run(
        "v_cvt_f16_u16_e32",
        vec![half(0, Half::Hi), half(1, Half::Hi)],
        &mut st,
    )
    .unwrap();
    assert_eq!(
        st.v[0][0],
        (u32::from(convert::f32_to_f16(7f32.to_bits())) << 16) | 0xbeef
    );
}

#[test]
fn fma_f16_rounds_once_and_honours_op_sel() {
    let h = |x: f32| convert::f32_to_f16(x.to_bits());
    let mut st = state();
    // a = b = 1 + 2^-10 (0x3c01), c = -(1 + 2^-9) (0xbc02): the exact product is 1 + 2^-9 + 2^-20, so the fused
    // result is 2^-20 (0x0010, a subnormal half); rounding the product to f16 first would give 0.
    let (a, b, c) = (0x3c01u32, 0x3c01u32, 0xbc02u32);
    st.v[1][0] = 0xdead_0000 | a;
    st.v[2][0] = 0xbeef_0000 | b;
    st.v[3][0] = 0xf00d_0000 | c;
    st.v[0][0] = 0xabcd_0000;
    run(
        "v_fma_f16",
        vec![half(0, Half::Lo), v(1), v(2), v(3)],
        &mut st,
    )
    .unwrap();
    assert_eq!(
        st.v[0][0], 0xabcd_0010,
        "fused, upper destination half preserved"
    );
    // op_sel picks the high half of src1 and src2, and op_sel[3] writes the high destination half.
    st.v[1][0] = 0x0000_0000 | a;
    st.v[2][0] = (b << 16) | 0xffff; // src1 high
    st.v[3][0] = (c << 16) | 0xffff; // src2 high
    st.v[0][0] = 0x0000_1234;
    let mods = Modifiers {
        op_sel: 0b0110,
        ..Default::default()
    };
    let i = insn_with(
        "v_fma_f16",
        vec![half(0, Half::Hi), v(1), v(2), v(3)],
        mods,
        FormFields::default(),
    );
    alu::execute(Arch::Gfx1100, "v_fma_f16", &i, &mut st).unwrap();
    assert_eq!(st.v[0][0], 0x0010_1234);
    // Plain case: 2 * 3 + 1 = 7.
    st.v[1][0] = u32::from(h(2.0));
    st.v[2][0] = u32::from(h(3.0));
    st.v[3][0] = u32::from(h(1.0));
    run(
        "v_fma_f16",
        vec![half(0, Half::Lo), v(1), v(2), v(3)],
        &mut st,
    )
    .unwrap();
    assert_eq!(st.v[0][0] & 0xffff, u32::from(h(7.0)));
    // An infinite operand is not modeled.
    st.v[1][0] = 0x7c00;
    assert!(run(
        "v_fma_f16",
        vec![half(0, Half::Lo), v(1), v(2), v(3)],
        &mut st
    )
    .is_err());
}

#[test]
fn fmac_f16_accumulates_into_the_destination_half() {
    let h = |x: f32| u32::from(convert::f32_to_f16(x.to_bits()));
    let mut st = state();
    st.v[1][0] = h(2.0); // S0.l
    st.v[2][0] = h(3.0) << 16; // S1.h
    st.v[0][0] = (h(1.0) << 16) | 0x4242; // accumulator in the destination's high half
    run(
        "v_fmac_f16_e32",
        vec![half(0, Half::Hi), half(1, Half::Lo), half(2, Half::Hi)],
        &mut st,
    )
    .unwrap();
    assert_eq!(
        st.v[0][0],
        (h(7.0) << 16) | 0x4242,
        "D = S0 * S1 + D, other half kept"
    );
}

#[test]
fn mad_u64_u32_carries_and_mad_i64_i32_wraps() {
    let mut st = state();
    st.exec = 0b11;
    // lane 0: 0xffffffff * 0xffffffff + 0xffffffffffffffff overflows 64 bits (carry 1);
    // lane 1: 3 * 4 + 5 = 17, no carry.
    st.v[2][0] = u32::MAX;
    st.v[3][0] = u32::MAX;
    st.v[4][0] = u32::MAX;
    st.v[5][0] = u32::MAX; // S2 = v[4:5]
    st.v[2][1] = 3;
    st.v[3][1] = 4;
    st.v[4][1] = 5;
    st.v[5][1] = 0;
    run(
        "v_mad_u64_u32",
        vec![
            reg(Kind::V, 0, 2),
            reg(Kind::S, 10, 1),
            v(2),
            v(3),
            reg(Kind::V, 4, 2),
        ],
        &mut st,
    )
    .unwrap();
    // 0xffff_ffff^2 = 0xffff_fffe_0000_0001; + 0xffff_ffff_ffff_ffff = 0x1_ffff_fffe_0000_0000.
    assert_eq!([st.v[0][0], st.v[1][0]], [0, 0xffff_fffe]);
    assert_eq!([st.v[0][1], st.v[1][1]], [17, 0]);
    assert_eq!(st.s[10], 0b01);
    // Null SDST is accepted.
    run(
        "v_mad_u64_u32",
        vec![
            reg(Kind::V, 0, 2),
            Operand::Special(Special::Null),
            v(2),
            v(3),
            reg(Kind::V, 4, 2),
        ],
        &mut st,
    )
    .unwrap();
    // Signed: -3 * 4 + 5 = -7.
    st.v[2][0] = (-3i32) as u32;
    st.v[3][0] = 4;
    st.v[4][0] = 5;
    st.v[5][0] = 0;
    run(
        "v_mad_i64_i32",
        vec![
            reg(Kind::V, 0, 2),
            Operand::Special(Special::Null),
            v(2),
            v(3),
            reg(Kind::V, 4, 2),
        ],
        &mut st,
    )
    .unwrap();
    assert_eq!(
        [st.v[0][0], st.v[1][0]],
        [(-7i64) as u64 as u32, ((-7i64) as u64 >> 32) as u32]
    );
    // The signed form's carry output is unspecified: a real SDST is refused.
    assert!(run(
        "v_mad_i64_i32",
        vec![
            reg(Kind::V, 0, 2),
            reg(Kind::S, 10, 1),
            v(2),
            v(3),
            reg(Kind::V, 4, 2)
        ],
        &mut st
    )
    .is_err());
}

#[test]
fn vopd_dual_max_selects_the_larger_and_prefers_positive_zero() {
    let row = |n: &str| {
        crate::isa::gfx1100()
            .iter()
            .find(|r| r.name == n)
            .unwrap()
            .op
    };
    let i = insn_with(
        "v_dual_max_f32",
        vec![v(1), v(2), v(3), v(4), v(5)],
        Modifiers::default(),
        FormFields::Vopd {
            y_op: row("v_dual_mov_b32"),
            x_operands: 3,
        },
    );
    let mut st = state();
    st.exec = 0b111;
    let vals = [(-1.0f32, 2.0f32), (-0.0, 0.0), (0.0, -0.0)];
    for (lane, (a, b)) in vals.into_iter().enumerate() {
        st.v[2][lane] = a.to_bits();
        st.v[3][lane] = b.to_bits();
        st.v[5][lane] = 0x1111_0000 + lane as u32;
    }
    alu::execute(Arch::Gfx1100, "v_dual_max_f32", &i, &mut st).unwrap();
    assert_eq!(st.v[1][0], 2.0f32.to_bits());
    assert_eq!(st.v[1][1], 0.0f32.to_bits(), "max(-0, +0) = +0");
    assert_eq!(st.v[1][2], 0.0f32.to_bits(), "max(+0, -0) = +0");
    assert_eq!(
        [st.v[4][0], st.v[4][1], st.v[4][2]],
        [0x1111_0000, 0x1111_0001, 0x1111_0002],
        "Y half moves too"
    );
}

#[test]
fn lds_two_address_b64_and_sub_dword_loads() {
    let mut lds = vec![0u8; 128];
    let mut st = state();
    st.v[2][0] = 8; // ADDR
    for w in 0..2 {
        st.v[4 + w][0] = 0x1111_1111 * (w as u32 + 1); // DATA0 = v[4:5]
        st.v[8 + w][0] = 0x1111_1111 * (w as u32 + 3); // DATA1 = v[8:9]
    }
    mem_run(
        "ds_store_2addr_b64",
        vec![
            v(2),
            reg(Kind::V, 4, 2),
            reg(Kind::V, 8, 2),
            Operand::Imm(ImmField::DsOffset0(1)),
            Operand::Imm(ImmField::DsOffset1(3)),
        ],
        &mut st,
        &mut Memory::default(),
        &mut lds,
    )
    .unwrap();
    // Element offsets scale by the 8-byte element: bytes 8 + 8 and 8 + 24.
    let word = |at: usize| u32::from_le_bytes(lds[at..at + 4].try_into().unwrap());
    assert_eq!(
        [word(16), word(20), word(32), word(36)],
        [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444]
    );
    assert_eq!(
        lds.iter().filter(|&&b| b != 0).count(),
        16,
        "only the two elements are written"
    );
    mem_run(
        "ds_load_2addr_b64",
        vec![
            reg(Kind::V, 12, 4),
            v(2),
            Operand::Imm(ImmField::DsOffset0(1)),
            Operand::Imm(ImmField::DsOffset1(3)),
        ],
        &mut st,
        &mut Memory::default(),
        &mut lds,
    )
    .unwrap();
    assert_eq!(
        [st.v[12][0], st.v[13][0], st.v[14][0], st.v[15][0]],
        [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444]
    );
    // ds_load_i8 sign-extends; ds_load_u16_d16 merges into the low half and keeps the high half.
    lds[40] = 0x80;
    lds[41] = 0x7f;
    st.v[1][0] = 40;
    mem_run(
        "ds_load_i8",
        vec![v(0), v(1)],
        &mut st,
        &mut Memory::default(),
        &mut lds,
    )
    .unwrap();
    assert_eq!(st.v[0][0], 0xffff_ff80);
    st.v[0][0] = 0xabcd_0000;
    mem_run(
        "ds_load_u16_d16",
        vec![v(0), v(1)],
        &mut st,
        &mut Memory::default(),
        &mut lds,
    )
    .unwrap();
    assert_eq!(st.v[0][0], 0xabcd_7f80);
}

#[test]
fn global_d16_loads_merge_halves_and_buffer_i8_zero_fills_out_of_range() {
    let bytes: Vec<u8> = (0..16).map(|n| 0x80 + n).collect();
    let mut mem = Memory::default();
    mem.map("m".into(), 0x1000, bytes).unwrap();
    let mut st = state();
    st.v[2][0] = 0x1000;
    st.v[3][0] = 0;
    for (name, init, want) in [
        ("global_load_d16_u8", 0xabcd_1234u32, 0xabcd_0080u32),
        ("global_load_d16_i8", 0xabcd_1234, 0xabcd_ff80),
        ("global_load_d16_hi_b16", 0xabcd_1234, 0x8180_1234),
    ] {
        st.v[0][0] = init;
        mem_run(
            name,
            vec![v(0), reg(Kind::V, 2, 2)],
            &mut st,
            &mut mem,
            &mut [],
        )
        .unwrap();
        assert_eq!(st.v[0][0], want, "{name}");
    }
    // buffer_load_i8 with a raw SRD (base 0x1000, 8 records): in range sign-extends, past NUM_RECORDS reads 0.
    let mut st = state();
    st.s[4] = 0x1000;
    st.s[6] = 8;
    st.s[7] = 0x3100_4000;
    let load = |st: &mut State, mem: &mut Memory, off: u32| {
        st.v[2][0] = off;
        st.v[0][0] = 0xdead_beef;
        mem_run(
            "buffer_load_i8",
            vec![
                v(0),
                v(2),
                reg(Kind::S, 4, 4),
                Operand::Special(Special::Null),
                Operand::Vmem(VmemToken::Offen),
            ],
            st,
            mem,
            &mut [],
        )
    };
    load(&mut st, &mut mem, 1).unwrap();
    assert_eq!(st.v[0][0], 0xffff_ff81);
    load(&mut st, &mut mem, 8).unwrap();
    assert_eq!(st.v[0][0], 0, "an out-of-range raw buffer load returns 0");
}

#[test]
fn hint_and_invalidate_rows_are_no_ops_that_never_fault() {
    // `wait`-class rows are skipped by the interpreter loop; the class is what keeps them from "no semantics".
    for name in [
        "buffer_gl0_inv",
        "s_sendmsg",
        "s_set_inst_prefetch_distance",
    ] {
        assert_eq!(super::semantic_class(name), Some("wait"), "{name}");
    }
}
