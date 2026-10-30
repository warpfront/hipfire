// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Unit tests for the opcodes the production gfx1100 decode kernels need and the emulator did not yet model
//! (PM-KRSI-1100 IsaGaps; the opcode/encoding/semantics-source log is `decode-audit/ISA-GAPS.md`).
//!
//! Every instruction is the exact word sequence `llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding` assembles from the
//! text in the doc comment above its constant, decoded by the production gfx11 decoder (so the operand shapes are the
//! lifter's). Every expected value is derived by hand from the RDNA3 ISA pseudocode (`isarefs` RDNA3 PDF, the page of
//! the opcode named in each test), never from the implementation under test; each test includes lanes that a wrong
//! semantic fails (sign, NaN, signed zero, 64-bit high word, exec masking, half selection).
use super::super::{memory, Memory, Result, State};
use crate::{codec::gfx11, Arch, Inst};

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

/// The instruction and its table name, decoded from machine words.
fn dec(words: &[u32]) -> (Inst, &'static str) {
    let (inst, n) =
        gfx11::decode(Arch::Gfx1100, words).unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"));
    assert_eq!(n, words.len(), "{words:08x?} decoded to {n} words");
    let name = crate::isa::lookup(Arch::Gfx1100, inst.op, inst.form)
        .unwrap_or_else(|| panic!("{words:08x?} has no gfx1100 row"))
        .name;
    (inst, name)
}

/// Run an ALU instruction; its name must have a semantics row of class `alu`.
fn run(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("alu"), "{name}");
    super::execute(Arch::Gfx1100, name, &inst, st)
}

/// Run a memory instruction; its name must have a semantics row of class `memory`.
fn mem_run(words: &[u32], st: &mut State, mem: &mut Memory, lds: &mut [u8]) -> Result<()> {
    let (inst, name) = dec(words);
    assert_eq!(crate::emu::semantic_class(name), Some("memory"), "{name}");
    memory::execute(name, &inst, st, mem, lds)
}

fn setv(st: &mut State, reg: usize, vals: &[u32]) {
    for (lane, v) in vals.iter().enumerate() {
        st.v[reg][lane] = *v;
    }
}

fn getv(st: &State, reg: usize, n: usize) -> Vec<u32> {
    st.v[reg][..n].to_vec()
}

/// `s_or_saveexec_b32 s6, s3`
const S_OR_SAVEEXEC: &[u32] = &[0xbe862203];
/// `s_lshr_b64 s[4:5], s[36:37], 21`
const S_LSHR_B64: &[u32] = &[0x85849524];
/// `s_xor_b64 s[8:9], s[8:9], s[6:7]`
const S_XOR_B64: &[u32] = &[0x8d880608];
/// `v_alignbyte_b32 v1, v2, v3, 2`
const V_ALIGNBYTE_IMM: &[u32] = &[0xd6170001, 0x020a0702];
/// `v_alignbyte_b32 v1, v2, v3, s4`
const V_ALIGNBYTE_SGPR: &[u32] = &[0xd6170001, 0x00120702];
/// `v_alignbyte_b32 v1, v2, v3, v26.h`
const V_ALIGNBYTE_HI: &[u32] = &[0xd6172001, 0x046a0702];
/// `v_bfrev_b32_e32 v0, v1`
const V_BFREV: &[u32] = &[0x7e007101];
/// `v_bfrev_b32_e32 v0, -2`
const V_BFREV_INL: &[u32] = &[0x7e0070c2];
/// `v_not_b32_e32 v15, v16`
const V_NOT: &[u32] = &[0x7e1e6f10];
/// `v_clz_i32_u32_e32 v14, v13`
const V_CLZ: &[u32] = &[0x7e1c730d];
/// `v_xor3_b32 v5, v6, v8, v7`
const V_XOR3: &[u32] = &[0xd6400005, 0x041e1106];
/// `v_bcnt_u32_b32 v16, v1, v2`
const V_BCNT: &[u32] = &[0xd71e0010, 0x02020501];
/// `v_min3_i32 v1, s4, s5, v1`
const V_MIN3_I32: &[u32] = &[0xd61a0001, 0x04040a04];
/// `v_med3_i32 v1, s9, 1, 0x100`
const V_MED3_I32: &[u32] = &[0xd6200001, 0x03fd0209, 0x00000100];
/// `v_movrels_b32_e32 v4, v9`
const V_MOVRELS: &[u32] = &[0x7e088709];
/// `v_movreld_b32_e32 v4, v9`
const V_MOVRELD: &[u32] = &[0x7e088509];
/// `v_lshlrev_b16 v8.l, 5, v8.l`
const V_LSHLREV_B16_IMM: &[u32] = &[0xd7380008, 0x02021085];
/// `v_lshlrev_b16 v8.h, v2.l, v3.h`
const V_LSHLREV_B16_HI: &[u32] = &[0xd7385008, 0x02020702];
/// `v_or_b16 v0.h, v0.h, v0.l`
const V_OR_B16: &[u32] = &[0xd7634800, 0x02020100];
/// `v_cndmask_b16 v9.l, v9.l, v13.l, vcc_lo`
const V_CNDMASK_B16: &[u32] = &[0xd65d0009, 0x01aa1b09];
/// `v_cvt_f32_f16_e32 v9, s0`
const V_CVT_F32_F16_E32: &[u32] = &[0x7e121600];
/// `v_cvt_f16_u16_e64 v229.l, v164.h`
const V_CVT_F16_U16_E64: &[u32] = &[0xd5d008e5, 0x020101a4];
/// `v_cvt_u32_u16_e64 v143, v2.l`
const V_CVT_U32_U16_E64_LO: &[u32] = &[0xd5eb008f, 0x02010102];
/// `v_cvt_u32_u16_e64 v143, v2.h`
const V_CVT_U32_U16_E64_HI: &[u32] = &[0xd5eb088f, 0x02010102];
/// `v_pk_mul_f16 v33, v12, v33`
const V_PK_MUL_F16: &[u32] = &[0xcc104021, 0x1a02430c];
/// `v_sub_f32_e64 v7, s11, s10`
const V_SUB_F32_E64: &[u32] = &[0xd5040007, 0x0200140b];
/// `v_fmac_f32_e64 v2, s15, s15`
const V_FMAC_F32_E64: &[u32] = &[0xd52b0002, 0x02001e0f];
/// `v_trunc_f32_e32 v5, v5`
const V_TRUNC: &[u32] = &[0x7e0a4305];
/// `v_floor_f32_e32 v67, v67`
const V_FLOOR: &[u32] = &[0x7e864943];
/// `v_med3_f32 v67, v67, s17, 0x42fe0000`
const V_MED3_F32: &[u32] = &[0xd61f0043, 0x03fc2343, 0x42fe0000];
/// `v_frexp_exp_i32_f32_e32 v7, v5`
const V_FREXP_EXP: &[u32] = &[0x7e0e7f05];
/// `v_frexp_mant_f32_e32 v6, v5`
const V_FREXP_MANT: &[u32] = &[0x7e0c8105];
/// `v_frexp_mant_f32_e64 v7, |v6|`
const V_FREXP_MANT_ABS: &[u32] = &[0xd5c00107, 0x02010106];
/// `v_add_co_ci_u32_e32 v4, vcc_lo, v5, v7, vcc_lo`
const V_ADD_CO_CI_E32: &[u32] = &[0x40080f05];
/// `v_sub_co_u32 v4, vcc_lo, v13, v4`
const V_SUB_CO: &[u32] = &[0xd7016a04, 0x0202090d];
/// `v_sub_co_ci_u32_e64 v5, null, v12, v5, vcc_lo`
const V_SUB_CO_CI_NULL: &[u32] = &[0xd5217c05, 0x01aa0b0c];
/// `v_sub_co_ci_u32_e64 v5, s2, v12, v5, s3`
const V_SUB_CO_CI_S: &[u32] = &[0xd5210205, 0x000e0b0c];
/// `v_subrev_co_ci_u32_e64 v10, null, s9, v10, vcc_lo`
const V_SUBREV_CO_CI_NULL: &[u32] = &[0xd5227c0a, 0x01aa1409];
/// `v_subrev_co_ci_u32_e64 v10, s6, s9, v10, s4`
const V_SUBREV_CO_CI_S: &[u32] = &[0xd522060a, 0x00121409];
/// `v_add_nc_u32_e64 v74, 0x7800, 0`
const V_ADD_NC_E64: &[u32] = &[0xd525004a, 0x020100ff, 0x00007800];
/// `v_cmp_eq_f32_e32 vcc_lo, v3, v8`
const V_CMP_EQ_F32: &[u32] = &[0x7c241103];
/// `v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v68`
const V_CMP_NEQ_F32: &[u32] = &[0x7c3a88ff, 0xff800000];
/// `v_cmp_o_f32_e32 vcc_lo, v10, v10`
const V_CMP_O_F32: &[u32] = &[0x7c2e150a];
/// `v_cmp_le_f32_e32 vcc_lo, v1, v2`
const V_CMP_LE_F32: &[u32] = &[0x7c260501];
/// `v_cmp_ge_i32_e32 vcc_lo, s9, v1`
const V_CMP_GE_I32: &[u32] = &[0x7c8c0209];
/// `v_cmp_ge_u32_e32 vcc_lo, v2, v5`
const V_CMP_GE_U32: &[u32] = &[0x7c9c0b02];
/// `v_cmp_lt_i64_e64 s11, s[26:27], v[11:12]`
const V_CMP_LT_I64_E64: &[u32] = &[0xd451000b, 0x0202161a];
/// `v_cmp_ne_u64_e64 s0, 0, v[27:28]`
const V_CMP_NE_U64_E64: &[u32] = &[0xd45d0000, 0x02023680];
/// `v_cmpx_ne_u32_e32 0x7f800000, v68`
const V_CMPX_NE_U32: &[u32] = &[0x7d9a88ff, 0x7f800000];
/// `v_cmpx_le_i32_e32 s1, v0`
const V_CMPX_LE_I32: &[u32] = &[0x7d860001];
/// `v_cmpx_le_u32_e32 v6, v2`
const V_CMPX_LE_U32: &[u32] = &[0x7d960506];
/// `v_cmpx_lt_u32_e32 1, v64`
const V_CMPX_LT_U32: &[u32] = &[0x7d928081];
/// `v_cmpx_eq_f32_e32 v3, v6`
const V_CMPX_EQ_F32: &[u32] = &[0x7d240d03];
/// `v_cmpx_nlt_f32_e32 0, v5`
const V_CMPX_NLT_F32: &[u32] = &[0x7d3c0a80];
/// `v_cmpx_ngt_f32_e64 0x3f200000, |v54|`
const V_CMPX_NGT_F32_E64: &[u32] = &[0xd49b027e, 0x02026cff, 0x3f200000];
/// `v_cmpx_le_f32_e64 0x48000000, |v6|`
const V_CMPX_LE_F32_E64: &[u32] = &[0xd493027e, 0x02020cff, 0x48000000];
/// `v_cmpx_gt_i64_e32 s[6:7], v[2:3]`
const V_CMPX_GT_I64_E32: &[u32] = &[0x7da80406];
/// `v_min_i32_dpp v0, v0, v0 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:1`
const V_MIN_I32_DPP: &[u32] = &[0x220000fa, 0xff096100];
/// `v_or_b32_dpp v5, v5, v5 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1`
const V_OR_B32_DPP: &[u32] = &[0x380a0afa, 0xff090105];
/// `ds_store_2addr_b32 v9, v5, v6 offset1:1`
const DS_STORE_2ADDR_B32: &[u32] = &[0xd8380100, 0x00060509];
/// `ds_load_2addr_stride64_b32 v[4:5], v3 offset1:2`
const DS_LOAD_2ADDR_STRIDE64_B32: &[u32] = &[0xd8e00200, 0x04000003];
/// `ds_load_b96 v[49:51], v34 offset:1266`
const DS_LOAD_B96: &[u32] = &[0xdbf804f2, 0x31000022];
/// `global_store_b8 v[0:1], v4, off`
const GLOBAL_STORE_B8: &[u32] = &[0xdc620000, 0x007c0400];
/// `global_store_d16_hi_b16 v[69:70], v68, off`
const GLOBAL_STORE_D16_HI_B16: &[u32] = &[0xdc960000, 0x007c4445];
/// `global_load_b96 v[40:42], v[36:37], off`
const GLOBAL_LOAD_B96: &[u32] = &[0xdc5a0000, 0x287c0024];
/// `global_atomic_add_u32 v2, v2, v3, s[4:5] glc`
const GLOBAL_ATOMIC_ADD_RTN: &[u32] = &[0xdcd64000, 0x02040302];
/// `global_atomic_add_u32 v[0:1], v2, off`
const GLOBAL_ATOMIC_ADD: &[u32] = &[0xdcd60000, 0x007c0200];
/// `flat_load_b64 v[36:37], v[9:10]`
const FLAT_LOAD_B64: &[u32] = &[0xdc540000, 0x247c0009];
/// `flat_load_b128 v[33:36], v[22:23]`
const FLAT_LOAD_B128: &[u32] = &[0xdc5c0000, 0x217c0016];
/// `buffer_gl1_inv`
const BUFFER_GL1_INV: &[u32] = &[0xe0b00000, 0x00000000];

// ---- scalar -------------------------------------------------------------------------------------------------------

/// RDNA3 ISA S_OR_SAVEEXEC_B32: `saveexec = EXEC; EXEC = S0 | EXEC; D0 = saveexec; SCC = EXEC != 0`.
#[test]
fn s_or_saveexec_b32_saves_old_exec_ors_and_sets_scc() {
    let mut st = state();
    st.exec = 0b0101;
    st.s[3] = 0b0011;
    run(S_OR_SAVEEXEC, &mut st).unwrap();
    assert_eq!((st.s[6], st.exec, st.scc), (0b0101, 0b0111, true));
    // Empty EXEC and empty S0: EXEC stays empty and SCC is clear (a stale SCC is overwritten).
    st.exec = 0;
    st.s[3] = 0;
    st.scc = true;
    run(S_OR_SAVEEXEC, &mut st).unwrap();
    assert_eq!((st.s[6], st.exec, st.scc), (0, 0, false));
    // Only S0 sets lanes: EXEC = 0x80000001 | 0xf0, the saved value is the OLD exec.
    st.exec = 0x8000_0001;
    st.s[3] = 0xf0;
    run(S_OR_SAVEEXEC, &mut st).unwrap();
    assert_eq!((st.s[6], st.exec, st.scc), (0x8000_0001, 0x8000_00f1, true));
}

/// RDNA3 ISA S_LSHR_B64: `D0.u64 = S0.u64 >> S1[5:0]` (logical), SCC = result nonzero.
#[test]
fn s_lshr_b64_shifts_the_whole_pair_logically() {
    let mut st = state();
    st.s[36] = 0x9abc_def0;
    st.s[37] = 0x1234_5678;
    run(S_LSHR_B64, &mut st).unwrap(); // >> 21
                                       // 0x123456789abcdef0 >> 21 = 0x91a2b3c4d5
    assert_eq!((st.s[4], st.s[5], st.scc), (0xa2b3_c4d5, 0x91, true));
    // The high word's top bit is a logical, not arithmetic, shift: 0x8000000000000000 >> 21 = 2^42.
    st.s[36] = 0;
    st.s[37] = 0x8000_0000;
    run(S_LSHR_B64, &mut st).unwrap();
    assert_eq!((st.s[4], st.s[5], st.scc), (0, 0x400, true));
    // Everything shifted out: zero and SCC clear.
    st.s[36] = 1;
    st.s[37] = 0;
    run(S_LSHR_B64, &mut st).unwrap();
    assert_eq!((st.s[4], st.s[5], st.scc), (0, 0, false));
}

/// RDNA3 ISA S_XOR_B64: `D0.u64 = S0.u64 ^ S1.u64`, SCC = result nonzero.
#[test]
fn s_xor_b64_xors_both_words_and_sets_scc() {
    let mut st = state();
    st.s[8] = 1;
    st.s[9] = 0xffff_ffff;
    st.s[6] = 1;
    st.s[7] = 1;
    run(S_XOR_B64, &mut st).unwrap();
    assert_eq!(
        (st.s[8], st.s[9], st.scc),
        (0, 0xffff_fffe, true),
        "the high word differs only"
    );
    st.s[6] = 0;
    st.s[7] = 0xffff_fffe; // s[8:9] is now {0, 0xfffffffe}
    run(S_XOR_B64, &mut st).unwrap();
    assert_eq!((st.s[8], st.s[9], st.scc), (0, 0, false));
}

// ---- integer / bit VALU -------------------------------------------------------------------------------------------

/// RDNA3 ISA V_ALIGNBYTE_B32: `D0 = ({S0, S1} >> (S2[1:0] * 8))[31:0]`, S0 the MSBs.
#[test]
fn v_alignbyte_b32_funnels_by_bytes() {
    let mut st = state();
    st.exec = 0b11;
    setv(&mut st, 2, &[0x1122_3344, 0xaabb_ccdd]);
    setv(&mut st, 3, &[0x5566_7788, 0x0102_0304]);
    run(V_ALIGNBYTE_IMM, &mut st).unwrap(); // S2 = 2: >> 16
                                            // 0x1122334455667788 >> 16 -> 0x33445566; 0xaabbccdd01020304 >> 16 -> 0xccdd0102
    assert_eq!(getv(&st, 1, 2), vec![0x3344_5566, 0xccdd_0102]);
    // S2 comes from an SGPR and only bits [1:0] count: 5 -> byte 1, 4 -> byte 0, 7 -> byte 3.
    for (s4, want0) in [
        (5u32, 0x4455_6677u32),
        (4, 0x5566_7788),
        (7, 0x2233_4455),
        (0xffff_fffc, 0x5566_7788),
    ] {
        st.s[4] = s4;
        run(V_ALIGNBYTE_SGPR, &mut st).unwrap();
        assert_eq!(st.v[1][0], want0, "S2 = {s4:#x}");
    }
    // Inactive lanes keep their old VGPR.
    st.exec = 0b01;
    st.v[1][1] = 0xdead_beef;
    st.s[4] = 1;
    run(V_ALIGNBYTE_SGPR, &mut st).unwrap();
    assert_eq!(st.v[1][1], 0xdead_beef);
    // OPSEL[2] (the high half of the 16-bit S2) is not modeled: refused, not guessed.
    assert!(run(V_ALIGNBYTE_HI, &mut st).is_err());
}

/// RDNA3 ISA V_BFREV_B32: `D0.u[31:0] = S0.u[0:31]`.
#[test]
fn v_bfrev_b32_reverses_bit_order() {
    let mut st = state();
    st.exec = 0b111;
    setv(&mut st, 1, &[1, 0x8000_0000, 0x1234_5678]);
    run(V_BFREV, &mut st).unwrap();
    // 0x12345678 = 0001 0010 0011 0100 0101 0110 0111 1000 -> reversed 0001 1110 0110 1010 0010 1100 0100 1000
    assert_eq!(getv(&st, 0, 3), vec![0x8000_0000, 1, 0x1e6a_2c48]);
    // Inline constant -2 = 0xfffffffe -> 0x7fffffff.
    run(V_BFREV_INL, &mut st).unwrap();
    assert_eq!(st.v[0][0], 0x7fff_ffff);
}

/// RDNA3 ISA V_NOT_B32: `D0.u = ~S0.u`.
#[test]
fn v_not_b32_complements() {
    let mut st = state();
    st.exec = 0b11;
    setv(&mut st, 16, &[0x0f0f_0f0f, 0]);
    run(V_NOT, &mut st).unwrap();
    assert_eq!(getv(&st, 15, 2), vec![0xf0f0_f0f0, 0xffff_ffff]);
}

/// RDNA3 ISA V_CLZ_I32_U32: first set bit counted from the MSB, -1 when S0 == 0.
#[test]
fn v_clz_i32_u32_counts_leading_zeros_and_minus_one_for_zero() {
    let mut st = state();
    st.exec = 0b1_1111;
    setv(&mut st, 13, &[0, 1, 0x8000_0000, 0x0001_0000, 0xffff_ffff]);
    run(V_CLZ, &mut st).unwrap();
    assert_eq!(getv(&st, 14, 5), vec![0xffff_ffff, 31, 0, 15, 0]);
}

/// RDNA3 ISA V_XOR3_B32: `D0.u = S0.u ^ S1.u ^ S2.u`.
#[test]
fn v_xor3_b32_xors_three_sources() {
    let mut st = state();
    setv(&mut st, 6, &[0xf0f0_f0f0]);
    setv(&mut st, 8, &[0x0ff0_0ff0]);
    setv(&mut st, 7, &[0x1234_5678]);
    run(V_XOR3, &mut st).unwrap();
    // f0f0f0f0 ^ 0ff00ff0 = ff00ff00; ^ 12345678 = ed34a978
    assert_eq!(st.v[5][0], 0xed34_a978);
}

/// RDNA3 ISA V_BCNT_U32_B32: `D0.u = S1.u + popcount(S0)` (wrapping).
#[test]
fn v_bcnt_u32_b32_adds_the_population_count() {
    let mut st = state();
    st.exec = 0b111;
    setv(&mut st, 1, &[0xf0f0_f0f0, 0, 0xffff_ffff]);
    setv(&mut st, 2, &[100, 0xffff_ffff, 0xffff_ffff]);
    run(V_BCNT, &mut st).unwrap();
    // 16 + 100; 0 + 0xffffffff; 32 + 0xffffffff wraps to 31.
    assert_eq!(getv(&st, 16, 3), vec![116, 0xffff_ffff, 31]);
}

/// RDNA3 ISA V_MIN3_I32: signed minimum of three.
#[test]
fn v_min3_i32_is_a_signed_three_way_minimum() {
    let mut st = state();
    st.exec = 0b11;
    st.s[4] = 5;
    st.s[5] = (-3i32) as u32;
    setv(&mut st, 1, &[7, (-10i32) as u32]);
    run(V_MIN3_I32, &mut st).unwrap();
    assert_eq!(getv(&st, 1, 2), vec![(-3i32) as u32, (-10i32) as u32]);
    // i32::MIN is the minimum, not a large unsigned: min(0x80000000, 3, 0x7fffffff).
    st.s[4] = 0x8000_0000;
    st.s[5] = 3;
    setv(&mut st, 1, &[0x7fff_ffff, 0x7fff_ffff]);
    run(V_MIN3_I32, &mut st).unwrap();
    assert_eq!(getv(&st, 1, 2), vec![0x8000_0000, 0x8000_0000]);
}

/// RDNA3 ISA V_MED3_I32: the signed median (max3 chain of the pseudocode).
#[test]
fn v_med3_i32_is_the_signed_median() {
    let mut st = state();
    st.exec = 1;
    // median(S0, 1, 0x100)
    for (s9, want) in [
        (5i32, 5i32),
        (-16, 1),
        (1000, 0x100),
        (1, 1),
        (0x100, 0x100),
        (i32::MIN, 1),
    ] {
        st.s[9] = s9 as u32;
        run(V_MED3_I32, &mut st).unwrap();
        assert_eq!(st.v[1][0] as i32, want, "median({s9}, 1, 256)");
    }
}

/// RDNA3 ISA V_MOVRELS_B32: `D0 = VGPR[laneId][SRC0 + M0]`.
#[test]
fn v_movrels_b32_reads_the_vgpr_offset_by_m0() {
    let mut st = state();
    st.exec = 0b11;
    setv(&mut st, 9, &[0x9, 0x99]);
    setv(&mut st, 12, &[0xaaaa, 0xbbbb]);
    st.m0 = 3;
    run(V_MOVRELS, &mut st).unwrap();
    assert_eq!(
        getv(&st, 4, 2),
        vec![0xaaaa, 0xbbbb],
        "v9 + M0=3 = v12, per lane"
    );
    st.m0 = 0;
    run(V_MOVRELS, &mut st).unwrap();
    assert_eq!(getv(&st, 4, 2), vec![0x9, 0x99]);
    // Past the end of the VGPR file: refused.
    st.m0 = 250;
    assert!(run(V_MOVRELS, &mut st).is_err());
}

/// RDNA3 ISA V_MOVRELD_B32: `VGPR[laneId][DST + M0] = SRC0`.
#[test]
fn v_movreld_b32_writes_the_vgpr_offset_by_m0() {
    let mut st = state();
    st.exec = 0b01;
    setv(&mut st, 9, &[0x9, 0x99]);
    setv(&mut st, 7, &[0x1, 0x2]);
    st.m0 = 3;
    run(V_MOVRELD, &mut st).unwrap();
    assert_eq!(
        getv(&st, 7, 2),
        vec![0x9, 0x2],
        "v4 + M0=3 = v7, active lane 0 only"
    );
    assert_eq!(getv(&st, 4, 1), vec![0], "v4 itself is untouched");
    st.m0 = 252;
    assert!(run(V_MOVRELD, &mut st).is_err());
}

/// RDNA3 ISA V_ADD_NC_U32 through its VOP3 spelling (literal + inline zero).
#[test]
fn v_add_nc_u32_e64_adds_a_literal() {
    let mut st = state();
    run(V_ADD_NC_E64, &mut st).unwrap();
    assert_eq!(st.v[74][0], 0x7800);
}

// ---- carry arithmetic ---------------------------------------------------------------------------------------------

/// RDNA3 ISA V_ADD_CO_CI_U32 (VOP2): `tmp = S0 + S1 + VCC[lane]; VCC[lane] = tmp >= 2^32; D0 = tmp[31:0]`.
#[test]
fn v_add_co_ci_u32_e32_carries_through_vcc() {
    let mut st = state();
    st.exec = 0b1111;
    setv(
        &mut st,
        5,
        &[0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0x7fff_ffff],
    );
    setv(&mut st, 7, &[0, 1, 0xffff_ffff, 0x8000_0000]);
    st.vcc = 0b0101; // carry-in on lanes 0 and 2
    run(V_ADD_CO_CI_E32, &mut st).unwrap();
    // lane0 ffffffff+0+1 = 2^32 -> 0 carry; lane1 ffffffff+1+0 = 2^32 -> 0 carry; lane2 ffffffff+ffffffff+1 = 2^33-1 ->
    // ffffffff carry; lane3 7fffffff+80000000+0 = ffffffff no carry.
    assert_eq!(getv(&st, 4, 4), vec![0, 0, 0xffff_ffff, 0xffff_ffff]);
    assert_eq!(st.vcc, 0b0111);
}

/// RDNA3 ISA V_SUB_CO_U32: `tmp = S0 - S1; VCC[lane] = S1 > S0; D0 = tmp` (the carry-out lands in SDST).
#[test]
fn v_sub_co_u32_borrows_unsigned() {
    let mut st = state();
    st.exec = 0b1111;
    setv(&mut st, 13, &[5, 3, 0, 0]); // S0
    setv(&mut st, 4, &[3, 5, 0, 0xffff_ffff]); // S1
    run(V_SUB_CO, &mut st).unwrap();
    assert_eq!(getv(&st, 4, 4), vec![2, 0xffff_fffe, 0, 1]);
    assert_eq!(st.vcc, 0b1010);
}

/// RDNA3 ISA V_SUB_CO_CI_U32 (VOP3): `tmp = S0 - S1 - cin; carry = S1 + cin > S0` with a 64-bit sum.
#[test]
fn v_sub_co_ci_u32_e64_subtracts_with_borrow_in() {
    let mut st = state();
    st.exec = 0b1111;
    setv(&mut st, 12, &[10, 3, 0, 0]); // S0
    setv(&mut st, 5, &[3, 3, 0, 0xffff_ffff]); // S1
    st.vcc = 0b0101; // carry-in lanes 0 and 2; SDST is null: VCC must stay untouched
    run(V_SUB_CO_CI_NULL, &mut st).unwrap();
    // lane0 10-3-1 = 6 (4 > 10 no); lane1 3-3-0 = 0; lane2 0-0-1 = ffffffff (borrow); lane3 0-ffffffff = 1 (borrow)
    assert_eq!(getv(&st, 5, 4), vec![6, 0, 0xffff_ffff, 1]);
    assert_eq!(st.vcc, 0b0101);
    // Named carry-in (s3) and carry-out (s2): the borrow mask of the same four lanes is lanes 2 and 3.
    setv(&mut st, 12, &[10, 3, 0, 0]);
    setv(&mut st, 5, &[3, 3, 0, 0xffff_ffff]);
    st.s[3] = 0b0101;
    st.s[2] = 0xdead_beef;
    run(V_SUB_CO_CI_S, &mut st).unwrap();
    assert_eq!(getv(&st, 5, 4), vec![6, 0, 0xffff_ffff, 1]);
    assert_eq!(st.s[2], 0b1100);
}

/// RDNA3 ISA V_SUBREV_CO_CI_U32: `tmp = S1 - S0 - cin`; the borrow is `S0 + cin > S1` (the RDNA4 pseudocode; the RDNA3
/// page prints the V_SUB_CO_CI formula under this opcode).
#[test]
fn v_subrev_co_ci_u32_e64_reverses_operands_and_borrows_correctly() {
    let mut st = state();
    st.exec = 0b1111;
    st.s[9] = 5; // S0
    setv(&mut st, 10, &[10, 4, 0xffff_ffff, 5]); // S1
    st.vcc = 0b0101; // cin lanes 0, 2
    run(V_SUBREV_CO_CI_NULL, &mut st).unwrap();
    // lane0 10-5-1 = 4; lane1 4-5 = ffffffff; lane2 ffffffff-5-1 = fffffff9; lane3 5-5 = 0
    assert_eq!(getv(&st, 10, 4), vec![4, 0xffff_ffff, 0xffff_fff9, 0]);
    // Borrow lanes (S0 + cin > S1): lane0 6>10 no; lane1 5>4 yes; lane2 6>ffffffff no; lane3 5>5 no.
    st.s[9] = 5;
    setv(&mut st, 10, &[10, 4, 0xffff_ffff, 5]);
    st.s[4] = 0b0101;
    st.s[6] = 0;
    run(V_SUBREV_CO_CI_S, &mut st).unwrap();
    assert_eq!(st.s[6], 0b0010);
    // 32-bit wrap of the sum must not hide a borrow: S0 = ffffffff, cin = 1, S1 = ffffffff -> D = ffffffff, borrow.
    st.exec = 1;
    st.s[9] = 0xffff_ffff;
    st.v[10][0] = 0xffff_ffff;
    st.s[4] = 1;
    run(V_SUBREV_CO_CI_S, &mut st).unwrap();
    assert_eq!((st.v[10][0], st.s[6]), (0xffff_ffff, 1));
}

// ---- compares ------------------------------------------------------------------------------------------------------

const NAN: u32 = 0x7fc0_0000;
const INF: u32 = 0x7f80_0000;
const NINF: u32 = 0xff80_0000;
const ONE: u32 = 0x3f80_0000;
const NEG_ONE: u32 = 0xbf80_0000;
const TWO: u32 = 0x4000_0000;
const NEG_ZERO: u32 = 0x8000_0000;

#[test]
fn f32_compares_follow_ieee_ordering() {
    let mut st = state();
    st.exec = 0b1111;
    // v_cmp_eq_f32 vcc, v3, v8: 1==1, 0==-0 (equal), 1==2, NaN==NaN (unordered: false).
    setv(&mut st, 3, &[ONE, 0, ONE, NAN]);
    setv(&mut st, 8, &[ONE, NEG_ZERO, TWO, NAN]);
    run(V_CMP_EQ_F32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b0011);
    // v_cmp_neq_f32 vcc, -inf, v68 (S0 != S1 or unordered): -inf, 1.0, NaN.
    st.exec = 0b111;
    setv(&mut st, 68, &[NINF, ONE, NAN]);
    run(V_CMP_NEQ_F32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b110);
    // v_cmp_o_f32 vcc, v10, v10 (both ordered): 1.0, NaN, +inf.
    setv(&mut st, 10, &[ONE, NAN, INF]);
    run(V_CMP_O_F32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b101);
    // v_cmp_le_f32 vcc, v1, v2: (1,2) true, (2,1) false, (-0,0) true, NaN false.
    st.exec = 0b1111;
    setv(&mut st, 1, &[ONE, TWO, NEG_ZERO, NAN]);
    setv(&mut st, 2, &[TWO, ONE, 0, ONE]);
    run(V_CMP_LE_F32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b0101);
}

#[test]
fn integer_compares_respect_signedness_and_width() {
    let mut st = state();
    st.exec = 0b111;
    // v_cmp_ge_i32 vcc, s9, v1 with s9 = 0: 0 >= -1 (signed), 0 >= 0, 0 >= 1.
    st.s[9] = 0;
    setv(&mut st, 1, &[0xffff_ffff, 0, 1]);
    run(V_CMP_GE_I32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b011);
    // v_cmp_ge_u32 vcc, v2, v5 (unsigned): ffffffff>=0 true, 0>=ffffffff false, equal true.
    setv(&mut st, 2, &[0xffff_ffff, 0, 7]);
    setv(&mut st, 5, &[0, 0xffff_ffff, 7]);
    run(V_CMP_GE_U32, &mut st).unwrap();
    assert_eq!(st.vcc, 0b101);
    // v_cmp_lt_i64 s11, s[26:27], v[11:12]: 2^32 < 2^32+1 true; 2^32 < 2^32-1 false; 2^32 < -2^32 false (signed).
    st.s[26] = 0;
    st.s[27] = 1;
    setv(&mut st, 11, &[1, 0xffff_ffff, 0]);
    setv(&mut st, 12, &[1, 0, 0xffff_ffff]);
    run(V_CMP_LT_I64_E64, &mut st).unwrap();
    assert_eq!(st.s[11], 0b001);
    // v_cmp_ne_u64 s0, 0, v[27:28]: 0 != 0 false; (lo 0, hi 1) true (the high word counts); (lo 1, hi 0) true.
    setv(&mut st, 27, &[0, 0, 1]);
    setv(&mut st, 28, &[0, 1, 0]);
    run(V_CMP_NE_U64_E64, &mut st).unwrap();
    assert_eq!(st.s[0], 0b110);
}

#[test]
fn cmpx_writes_exec_and_only_active_lanes_can_stay_active() {
    let mut st = state();
    // v_cmpx_ne_u32 0x7f800000, v68; lane 1 is inactive and must stay out although it would compare true.
    st.exec = 0b101;
    setv(&mut st, 68, &[0x7f80_0000, 0x7f80_0001, 0]);
    run(V_CMPX_NE_U32, &mut st).unwrap();
    assert_eq!(st.exec, 0b100);
    // v_cmpx_le_i32 s1, v0 with s1 = -2: -3 false, -2 true, 0 true.
    st.exec = 0b111;
    st.s[1] = (-2i32) as u32;
    setv(&mut st, 0, &[(-3i32) as u32, (-2i32) as u32, 0]);
    run(V_CMPX_LE_I32, &mut st).unwrap();
    assert_eq!(st.exec, 0b110);
    // v_cmpx_le_u32 v6, v2 (unsigned): (ffffffff,0) false, (1,1) true, (0,ffffffff) true.
    st.exec = 0b111;
    setv(&mut st, 6, &[0xffff_ffff, 1, 0]);
    setv(&mut st, 2, &[0, 1, 0xffff_ffff]);
    run(V_CMPX_LE_U32, &mut st).unwrap();
    assert_eq!(st.exec, 0b110);
    // v_cmpx_lt_u32 1, v64: 1<0 f, 1<1 f, 1<2 t, 1<ffffffff t.
    st.exec = 0b1111;
    setv(&mut st, 64, &[0, 1, 2, 0xffff_ffff]);
    run(V_CMPX_LT_U32, &mut st).unwrap();
    assert_eq!(st.exec, 0b1100);
    // v_cmpx_eq_f32 v3, v6: (1,1) t, (NaN,NaN) f, (0,-0) t.
    st.exec = 0b111;
    setv(&mut st, 3, &[ONE, NAN, 0]);
    setv(&mut st, 6, &[ONE, NAN, NEG_ZERO]);
    run(V_CMPX_EQ_F32, &mut st).unwrap();
    assert_eq!(st.exec, 0b101);
    // v_cmpx_nlt_f32 0, v5 = !(0 < v5): 1.0 -> f, -1.0 -> t, NaN -> t (unordered), 0 -> t.
    st.exec = 0b1111;
    setv(&mut st, 5, &[ONE, NEG_ONE, NAN, 0]);
    run(V_CMPX_NLT_F32, &mut st).unwrap();
    assert_eq!(st.exec, 0b1110);
    // v_cmpx_ngt_f32_e64 0.625, |v54| = !(0.625 > |v54|): 0.5 -> f; -0.75 -> |.| = 0.75 -> t; NaN -> t.
    st.exec = 0b111;
    setv(&mut st, 54, &[0x3f00_0000, 0xbf40_0000, NAN]);
    run(V_CMPX_NGT_F32_E64, &mut st).unwrap();
    assert_eq!(st.exec, 0b110);
    // v_cmpx_le_f32_e64 131072.0, |v6| : |-200000| t; |-100000| f; 131072 t (equal).
    st.exec = 0b111;
    setv(&mut st, 6, &[0xc843_5000, 0xc7c3_5000, 0x4800_0000]);
    run(V_CMPX_LE_F32_E64, &mut st).unwrap();
    assert_eq!(st.exec, 0b101);
    // v_cmpx_gt_i64 s[6:7], v[2:3] with s[6:7] = 0: 0 > -1 t; 0 > 1 f; 0 > -2^63 t (signed; unsigned would give f, f, f).
    st.exec = 0b111;
    st.s[6] = 0;
    st.s[7] = 0;
    setv(&mut st, 2, &[0xffff_ffff, 1, 0]);
    setv(&mut st, 3, &[0xffff_ffff, 0, 0x8000_0000]);
    run(V_CMPX_GT_I64_E32, &mut st).unwrap();
    assert_eq!(st.exec, 0b101);
}

// ---- DPP ------------------------------------------------------------------------------------------------------------

/// `v_min_i32_dpp v0, v0, v0 row_xmask:1`: src0 of lane l is lane l^1 (inside its row), D = min(src0, v0[l]) signed.
#[test]
fn v_min_i32_dpp_row_xmask_takes_the_partner_lane() {
    let mut st = state();
    st.exec = 0xffff_ffff;
    for lane in 0..32 {
        st.v[0][lane] = 1000 + lane as u32;
    }
    setv(&mut st, 0, &[5, (-3i32) as u32, 7, 100]);
    run(V_MIN_I32_DPP, &mut st).unwrap();
    // pairs (0,1): min(5,-3) = -3 both; (2,3): min(7,100) = 7 both; (16,17): 1000 both.
    assert_eq!(st.v[0][0] as i32, -3);
    assert_eq!(st.v[0][1] as i32, -3);
    assert_eq!((st.v[0][2], st.v[0][3]), (7, 7));
    assert_eq!((st.v[0][16], st.v[0][17]), (1016, 1016));
    assert_eq!((st.v[0][30], st.v[0][31]), (1030, 1030));
}

/// `v_or_b32_dpp v5, v5, v5 row_shl:1 bound_ctrl:1`: lane l reads lane l+1 of its row; the last lane of a row reads 0.
#[test]
fn v_or_b32_dpp_row_shl_reads_the_next_lane_and_zero_at_the_row_end() {
    let mut st = state();
    st.exec = 0xffff_ffff;
    for lane in 0..32 {
        st.v[5][lane] = 1 << lane;
    }
    run(V_OR_B32_DPP, &mut st).unwrap();
    assert_eq!(st.v[5][0], 0b11);
    assert_eq!(st.v[5][7], (1 << 7) | (1 << 8));
    assert_eq!(
        st.v[5][15],
        1 << 15,
        "row end: src0 out of range reads 0 (bound_ctrl)"
    );
    assert_eq!(st.v[5][16], (1 << 16) | (1 << 17));
    assert_eq!(st.v[5][31], 1 << 31);
}

// ---- 16-bit and conversions --------------------------------------------------------------------------------------

/// RDNA3 ISA V_LSHLREV_B16: `D0.u[15:0] = S1.u[15:0] << S0[3:0]`; the other destination half is preserved (true16).
#[test]
fn v_lshlrev_b16_shifts_a_half_and_preserves_the_other() {
    let mut st = state();
    st.v[8][0] = 0xabcd_00f1;
    run(V_LSHLREV_B16_IMM, &mut st).unwrap(); // v8.l = v8.l << 5
    assert_eq!(
        st.v[8][0], 0xabcd_1e20,
        "0x00f1 << 5 = 0x1e20, high half untouched"
    );
    st.v[8][0] = 0xabcd_ffff;
    run(V_LSHLREV_B16_IMM, &mut st).unwrap();
    assert_eq!(
        st.v[8][0], 0xabcd_ffe0,
        "bits shifted out of the half are dropped, not carried into the other half"
    );
    // v8.h = v3.h << (v2.l & 15): count 0x13 & 15 = 3.
    st.v[8][0] = 0x1111_2222;
    st.v[2][0] = 0x0000_0013;
    st.v[3][0] = 0x0003_0000;
    run(V_LSHLREV_B16_HI, &mut st).unwrap();
    assert_eq!(st.v[8][0], 0x0018_2222);
}

/// RDNA3 ISA V_OR_B16: `D0.u16 = S0.u16 | S1.u16`.
#[test]
fn v_or_b16_ors_the_selected_halves() {
    let mut st = state();
    st.v[0][0] = 0x1200_00ff;
    run(V_OR_B16, &mut st).unwrap(); // v0.h = v0.h | v0.l
    assert_eq!(st.v[0][0], 0x12ff_00ff);
}

/// RDNA3 ISA V_CNDMASK_B16: `D0.u16 = VCC[lane] ? S1.u16 : S0.u16`.
#[test]
fn v_cndmask_b16_selects_per_lane() {
    let mut st = state();
    st.exec = 0b11;
    setv(&mut st, 9, &[0x1111_aaaa, 0x2222_bbbb]);
    setv(&mut st, 13, &[0x3333_cccc, 0x4444_dddd]);
    st.vcc = 0b01;
    run(V_CNDMASK_B16, &mut st).unwrap();
    // lane0 (mask set) takes S1 = v13.l = cccc; lane1 keeps S0 = v9.l = bbbb; high halves untouched.
    assert_eq!(getv(&st, 9, 2), vec![0x1111_cccc, 0x2222_bbbb]);
}

/// RDNA3 ISA V_CVT_F32_F16: the selected half widens exactly to f32.
#[test]
fn v_cvt_f32_f16_e32_widens_the_low_half_of_a_scalar() {
    let mut st = state();
    st.s[0] = 0xffff_3c00; // upper half is ignored
    run(V_CVT_F32_F16_E32, &mut st).unwrap();
    assert_eq!(st.v[9][0], ONE);
    st.s[0] = 0x0000_c000; // -2.0
    run(V_CVT_F32_F16_E32, &mut st).unwrap();
    assert_eq!(st.v[9][0], 0xc000_0000);
}

/// RDNA3 ISA V_CVT_F16_U16 (VOP3, high source half): `D0.f16 = u16_to_f16(S0.u16)`, one rounding.
#[test]
fn v_cvt_f16_u16_e64_converts_the_selected_half() {
    let mut st = state();
    st.v[164][0] = 0x0007_0000;
    st.v[229][0] = 0x5555_0000;
    run(V_CVT_F16_U16_E64, &mut st).unwrap();
    assert_eq!(
        st.v[229][0], 0x5555_4700,
        "7 -> 0x4700, destination high half preserved"
    );
    st.v[164][0] = 2049 << 16; // exactly halfway between 2048 and 2050: ties to even = 2048 (0x6800)
    run(V_CVT_F16_U16_E64, &mut st).unwrap();
    assert_eq!(st.v[229][0] & 0xffff, 0x6800);
    st.v[164][0] = 0xffff_0000; // 65535 > 65520: rounds to infinity
    run(V_CVT_F16_U16_E64, &mut st).unwrap();
    assert_eq!(st.v[229][0] & 0xffff, 0x7c00);
}

/// RDNA3 ISA V_CVT_U32_U16: `D0 = {16'0, S0.u16}`.
#[test]
fn v_cvt_u32_u16_e64_zero_extends_the_selected_half() {
    let mut st = state();
    st.v[2][0] = 0xabcd_1234;
    run(V_CVT_U32_U16_E64_LO, &mut st).unwrap();
    assert_eq!(st.v[143][0], 0x1234);
    run(V_CVT_U32_U16_E64_HI, &mut st).unwrap();
    assert_eq!(
        st.v[143][0], 0xabcd,
        "the high half is zero-extended, not sign-extended"
    );
}

/// RDNA3 ISA V_PK_MUL_F16: `D0[31:16] = S0[31:16] * S1[31:16]; D0[15:0] = S0[15:0] * S1[15:0]`.
#[test]
fn v_pk_mul_f16_multiplies_both_halves_with_one_rounding() {
    let mut st = state();
    // hi: 2.0 * 0.5 = 1.0 (0x3c00); lo: 3.0 * -1.5 = -4.5 (0xc480)
    st.v[12][0] = 0x4000_4200;
    st.v[33][0] = 0x3800_be00;
    run(V_PK_MUL_F16, &mut st).unwrap();
    assert_eq!(st.v[33][0], 0x3c00_c480);
    // Signed zeros: lo -0 * 2.0 = -0; hi +0 * -1.0 = -0.
    st.v[12][0] = 0x0000_8000;
    st.v[33][0] = 0xbc00_4000;
    run(V_PK_MUL_F16, &mut st).unwrap();
    assert_eq!(st.v[33][0], 0x8000_8000);
    // (1 + 2^-10)^2 = 1 + 2^-9 + 2^-20 rounds once to 1 + 2^-9 = 0x3c02 (the 2^-20 is far below half an ulp).
    st.v[12][0] = 0x3c01_3c01;
    st.v[33][0] = 0x3c01_3c01;
    run(V_PK_MUL_F16, &mut st).unwrap();
    assert_eq!(st.v[33][0], 0x3c02_3c02);
    // 256 * 256 = 65536 overflows to +inf; 2^-14 (min normal) * 0.5 = 2^-15 is the subnormal 0x0200.
    st.v[12][0] = 0x0400_5c00;
    st.v[33][0] = 0x3800_5c00;
    run(V_PK_MUL_F16, &mut st).unwrap();
    assert_eq!(st.v[33][0], 0x0200_7c00);
    // A non-finite operand is refused, not approximated.
    st.v[12][0] = 0x7c00_3c00;
    assert!(run(V_PK_MUL_F16, &mut st).is_err());
}

/// The gfx1201 instruction and its table name, decoded from machine words (gfx12 codec).
fn dec12(words: &[u32]) -> (Inst, &'static str) {
    let (inst, n) = crate::codec::gfx12::decode_for(Arch::Gfx1201, words)
        .unwrap_or_else(|e| panic!("decode {words:08x?}: {e}"));
    assert_eq!(n, words.len(), "{words:08x?} decoded to {n} words");
    let name = crate::isa::lookup(Arch::Gfx1201, inst.op, inst.form)
        .unwrap_or_else(|| panic!("{words:08x?} has no gfx1201 row"))
        .name;
    (inst, name)
}

fn run12(words: &[u32], st: &mut State) -> Result<()> {
    let (inst, name) = dec12(words);
    assert_eq!(crate::emu::semantic_class(name), Some("alu"), "{name}");
    super::execute(Arch::Gfx1201, name, &inst, st)
}

/// `v_mul_hi_u32_u24_e32 v2, 36, v12` (gfx1201, VOP2 0xc).
const V_MUL_HI_U32_U24: &[u32] = &[0x180418a4];
/// `v_mul_hi_u32_u24_e32 v2, v1, v12`
const V_MUL_HI_U32_U24_VV: &[u32] = &[0x18041901];
/// `v_pk_lshrrev_b16 v20, 8, v20 op_sel:[1,0] op_sel_hi:[0,0]` (gfx1201, VOP3P 0x5).
const V_PK_LSHRREV_B16: &[u32] = &[0xcc054814, 0x02022888];

/// RDNA ISA V_MUL_HI_U32_U24: `D0.u = ((S0.u & 0xffffff) * (S1.u & 0xffffff)) >> 32`.
#[test]
fn v_mul_hi_u32_u24_decoded_masks_and_takes_the_high_word() {
    let mut st = state();
    // Inline 36 * (0xff00_0001 & 0xffffff = 1) = 36 -> 0; an unmasked S1 would give (0xff00_0001*36)>>32 = 35.
    st.v[12][0] = 0xff00_0001;
    run12(V_MUL_HI_U32_U24, &mut st).unwrap();
    assert_eq!(st.v[2][0], 0);
    // 0xffffff * 0xffffff = 2^48 - 2^25 + 1 -> bits 47:32 = 0xffff; the bits above 24 of both sources are ignored.
    st.v[1][0] = 0xabff_ffff;
    st.v[12][0] = 0x01ff_ffff;
    run12(V_MUL_HI_U32_U24_VV, &mut st).unwrap();
    assert_eq!(st.v[2][0], 0xffff);
}

/// The real encoding `CC054814 02022888`: op_sel = 0b001 (src0 high half feeds the LOW result), op_sel_hi = 0b000
/// (both sources' low halves feed the HIGH result). The inline constant 8 is the word 8 (high half 0), so
/// lo = v20.lo >> (0 & 15) = v20.lo and hi = v20.lo >> 8. Expected values are from the ISA formula, not kernel output.
#[test]
fn v_pk_lshrrev_b16_decoded_encoding_executes_with_op_sel() {
    let (inst, name) = dec12(V_PK_LSHRREV_B16);
    assert_eq!(name, "v_pk_lshrrev_b16");
    assert_eq!((inst.mods.op_sel, inst.mods.op_sel_hi & 0b011), (0b001, 0)); // bit 2 of op_sel_hi is the unused src2
    let mut st = state();
    st.v[20][0] = 0xab12_cd34;
    super::execute(Arch::Gfx1201, name, &inst, &mut st).unwrap();
    assert_eq!(st.v[20][0], 0x00cd_cd34);
}

/// `v_mul_hi_u32_u24_e32 v2, 36, v9` (gfx1100, VOP2 0xc: RDNA3 ISA XML sha256 6eee5f87...), the word hipcc emits for the
/// `kbx * 36` address product of the ggml mmvq objects.
const V_MUL_HI_U32_U24_G11: &[u32] = &[0x180412a4];
/// `v_mul_hi_u32_u24_e32 v2, v1, v9`
const V_MUL_HI_U32_U24_VV_G11: &[u32] = &[0x18041301];
/// `v_pk_lshrrev_b16 v20, 8, v20 op_sel:[1,0] op_sel_hi:[0,0]` (gfx1100, VOP3P 0x5), same words as gfx1201.
const V_PK_LSHRREV_B16_G11: &[u32] = &[0xcc054814, 0x02022888];

/// RDNA3 ISA V_MUL_HI_U32_U24: `D0.u = ((S0.u & 0xffffff) * (S1.u & 0xffffff)) >> 32`.
#[test]
fn v_mul_hi_u32_u24_gfx1100_masks_and_takes_the_high_word() {
    let mut st = state();
    // Inline 36 * (0xff00_0001 & 0xffffff = 1) = 36 -> 0; an unmasked S1 would give (0xff00_0001*36)>>32 = 35.
    st.v[9][0] = 0xff00_0001;
    run(V_MUL_HI_U32_U24_G11, &mut st).unwrap();
    assert_eq!(st.v[2][0], 0);
    // 0xffffff * 0xffffff = 2^48 - 2^25 + 1 -> bits 47:32 = 0xffff; the bits above 24 of both sources are ignored.
    st.v[1][0] = 0xabff_ffff;
    st.v[9][0] = 0x01ff_ffff;
    run(V_MUL_HI_U32_U24_VV_G11, &mut st).unwrap();
    assert_eq!(st.v[2][0], 0xffff);
}

/// The real encoding `CC054814 02022888` decoded by the gfx11 codec: op_sel = 0b001 (src0's high half feeds the LOW
/// result), op_sel_hi = 0b000. The inline constant 8 is the word 8 (high half 0): lo = v20.lo >> 0, hi = v20.lo >> 8.
#[test]
fn v_pk_lshrrev_b16_gfx1100_decoded_encoding_executes_with_op_sel() {
    let (inst, name) = dec(V_PK_LSHRREV_B16_G11);
    assert_eq!(name, "v_pk_lshrrev_b16");
    assert_eq!((inst.mods.op_sel, inst.mods.op_sel_hi & 0b011), (0b001, 0));
    let mut st = state();
    st.v[20][0] = 0xab12_cd34;
    super::execute(Arch::Gfx1100, name, &inst, &mut st).unwrap();
    assert_eq!(st.v[20][0], 0x00cd_cd34);
}

// ---- f32 VALU -----------------------------------------------------------------------------------------------------

#[test]
fn v_sub_f32_e64_and_v_fmac_f32_e64_use_the_vop3_operand_order() {
    let mut st = state();
    st.s[11] = 0x40a0_0000; // 5.0
    st.s[10] = 0x3fc0_0000; // 1.5
    run(V_SUB_F32_E64, &mut st).unwrap();
    assert_eq!(st.v[7][0], 0x4060_0000, "5.0 - 1.5 = 3.5");
    // v2 = s15 * s15 + v2 (fused): 3.0 * 3.0 + 1.0 = 10.0.
    st.s[15] = 0x4040_0000;
    st.v[2][0] = ONE;
    run(V_FMAC_F32_E64, &mut st).unwrap();
    assert_eq!(st.v[2][0], 0x4120_0000);
}

/// RDNA3 ISA V_TRUNC_F32: `D0.f = trunc(S0.f)`.
#[test]
fn v_trunc_f32_rounds_toward_zero() {
    let mut st = state();
    st.exec = 0b1111_1111;
    setv(
        &mut st,
        5,
        &[
            0x4030_0000,
            0xc030_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x5015_02f9,
            INF,
            0x0000_0001,
            NEG_ZERO,
        ],
    );
    run(V_TRUNC, &mut st).unwrap();
    assert_eq!(
        getv(&st, 5, 8),
        // 2.75 -> 2; -2.75 -> -2; 0.5 -> +0; -0.5 -> -0 (IEEE round-toward-zero keeps the sign); 1e10 is an integer already;
        // +inf; the smallest denormal -> +0; -0 -> -0.
        vec![
            0x4000_0000,
            0xc000_0000,
            0,
            NEG_ZERO,
            0x5015_02f9,
            INF,
            0,
            NEG_ZERO
        ]
    );
    // A NaN operand is refused (propagation is not specified for this opcode).
    st.exec = 1;
    st.v[5][0] = NAN;
    assert!(run(V_TRUNC, &mut st).is_err());
}

/// RDNA3 ISA V_FLOOR_F32: `D0 = trunc(S0); if S0 < 0 && S0 != D0 then D0 += -1.0`.
#[test]
fn v_floor_f32_rounds_toward_negative_infinity() {
    let mut st = state();
    st.exec = 0b1_1111;
    setv(
        &mut st,
        67,
        &[
            0x4030_0000,
            0xc030_0000,
            0xbf00_0000,
            0x3f00_0000,
            0xc040_0000,
        ],
    );
    run(V_FLOOR, &mut st).unwrap();
    // 2.75 -> 2.0; -2.75 -> -3.0; -0.5 -> -1.0; 0.5 -> +0; -3.0 -> -3.0.
    assert_eq!(
        getv(&st, 67, 5),
        vec![0x4000_0000, 0xc040_0000, 0xbf80_0000, 0, 0xc040_0000]
    );
    st.exec = 1;
    st.v[67][0] = NEG_ZERO;
    run(V_FLOOR, &mut st).unwrap();
    assert_eq!(
        st.v[67][0], NEG_ZERO,
        "S0 = -0 is not < 0: trunc(-0) = -0 is kept"
    );
    st.v[67][0] = NAN;
    assert!(run(V_FLOOR, &mut st).is_err());
}

/// RDNA3 ISA V_MED3_F32: the median of three (the ISA's max3/max chain), used as a clamp `med3(x, lo, hi)`.
#[test]
fn v_med3_f32_is_a_clamp_when_the_bounds_are_ordered() {
    let mut st = state();
    st.exec = 0b1111;
    st.s[17] = 0xc2fe_0000; // -127.0; the third source is the literal 127.0
    setv(
        &mut st,
        67,
        &[0x4348_0000, 0xc396_0000, 0x40b0_0000, 0x42fe_0000],
    ); // 200, -300, 5.5, 127
    run(V_MED3_F32, &mut st).unwrap();
    assert_eq!(
        getv(&st, 67, 4),
        vec![0x42fe_0000, 0xc2fe_0000, 0x40b0_0000, 0x42fe_0000]
    );
    // Infinities order normally.
    st.exec = 0b11;
    setv(&mut st, 67, &[INF, NINF]);
    run(V_MED3_F32, &mut st).unwrap();
    assert_eq!(getv(&st, 67, 2), vec![0x42fe_0000, 0xc2fe_0000]);
    // A NaN operand, and a median zero whose operand zeros differ in sign, are refused.
    st.exec = 1;
    st.v[67][0] = NAN;
    assert!(run(V_MED3_F32, &mut st).is_err());
    st.s[17] = 0;
    st.v[67][0] = NEG_ZERO;
    assert!(run(V_MED3_F32, &mut st).is_err());
}

/// RDNA3 ISA V_FREXP_EXP_I32_F32: `D0.i = exponent(S0.f) - 127 + 1`, 0 for infinities and NaN.
#[test]
fn v_frexp_exp_i32_f32_returns_the_binary_exponent_of_a_half_open_significand() {
    let mut st = state();
    st.exec = 0b1_1111_1;
    setv(
        &mut st,
        5,
        &[0x4100_0000, ONE, 0x3f00_0000, 0xbf40_0000, INF, 0x7f7f_ffff],
    );
    run(V_FREXP_EXP, &mut st).unwrap();
    // 8 = 0.5 * 2^4; 1 = 0.5 * 2^1; 0.5 = 0.5 * 2^0; -0.75 = -0.75 * 2^0; inf -> 0; f32::MAX < 2^128 -> 128.
    assert_eq!(getv(&st, 7, 6), vec![4, 1, 0, 0, 0, 128]);
    st.exec = 1;
    st.v[5][0] = NAN;
    run(V_FREXP_EXP, &mut st).unwrap();
    assert_eq!(st.v[7][0], 0, "the ISA returns 0 for NaN");
    // Zero and denormals: the pseudocode and frexp() disagree, so they are refused.
    st.v[5][0] = 0;
    assert!(run(V_FREXP_EXP, &mut st).is_err());
    st.v[5][0] = 1;
    assert!(run(V_FREXP_EXP, &mut st).is_err());
}

/// RDNA3 ISA V_FREXP_MANT_F32: the significand in [0.5, 1) with the input's sign; +-inf passes through.
#[test]
fn v_frexp_mant_f32_returns_the_significand() {
    let mut st = state();
    st.exec = 0b1111;
    setv(&mut st, 5, &[0x4100_0000, 0xc040_0000, ONE, NINF]);
    run(V_FREXP_MANT, &mut st).unwrap();
    // 8 -> 0.5; -3 = -1.5 * 2 -> -0.75; 1 -> 0.5; -inf -> -inf.
    assert_eq!(
        getv(&st, 6, 4),
        vec![0x3f00_0000, 0xbf40_0000, 0x3f00_0000, NINF]
    );
    // With |src|: -3.0 -> 3.0 -> 0.75.
    st.exec = 1;
    st.v[6][0] = 0xc040_0000;
    run(V_FREXP_MANT_ABS, &mut st).unwrap();
    assert_eq!(st.v[7][0], 0x3f40_0000);
    st.v[6][0] = NAN;
    assert!(run(V_FREXP_MANT_ABS, &mut st).is_err());
    st.v[6][0] = 0;
    assert!(run(V_FREXP_MANT_ABS, &mut st).is_err());
}

// ---- memory -------------------------------------------------------------------------------------------------------

#[test]
fn ds_2addr_b32_and_stride64_and_b96_address_the_documented_dwords() {
    let mut st = state();
    st.exec = 0b11;
    let mut mem = Memory::default();
    let mut lds = vec![0u8; 2048];
    // ds_store_2addr_b32 v9, v5, v6 offset1:1: DATA0 at addr + 0*4, DATA1 at addr + 1*4.
    setv(&mut st, 9, &[16, 64]);
    setv(&mut st, 5, &[0x1111_1111, 0xaabb_ccdd]);
    setv(&mut st, 6, &[0x2222_2222, 0x0102_0304]);
    mem_run(DS_STORE_2ADDR_B32, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        lds[16..24],
        [0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22]
    );
    assert_eq!(
        lds[64..72],
        [0xdd, 0xcc, 0xbb, 0xaa, 0x04, 0x03, 0x02, 0x01]
    );
    assert_eq!(lds[24], 0, "nothing past the second dword");
    // ds_load_2addr_stride64_b32 v[4:5], v3 offset1:2: elements at addr + 0*64*4 and addr + 2*64*4 = addr + 512.
    lds[100..104].copy_from_slice(&0x0a0b_0c0du32.to_le_bytes());
    lds[612..616].copy_from_slice(&0x1e1f_2021u32.to_le_bytes());
    setv(&mut st, 3, &[100, 0]);
    st.exec = 1;
    mem_run(DS_LOAD_2ADDR_STRIDE64_B32, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!((st.v[4][0], st.v[5][0]), (0x0a0b_0c0d, 0x1e1f_2021));
    // ds_load_b96 v[49:51], v34 offset:1266: three consecutive dwords from v34 + 1266.
    for (k, b) in lds[1276..1288].iter_mut().enumerate() {
        *b = 0x40 + k as u8;
    }
    st.v[34][0] = 10;
    mem_run(DS_LOAD_B96, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[49][0], st.v[50][0], st.v[51][0]),
        (0x4342_4140, 0x4746_4544, 0x4b4a_4948)
    );
}

#[test]
fn global_sub_dword_stores_and_wide_loads() {
    let mut st = state();
    st.exec = 1;
    let mut mem = Memory::default();
    mem.map("buf".into(), 0x1000, vec![0xaa; 64]).unwrap();
    let mut lds = [];
    // global_store_b8 v[0:1], v4, off: one byte.
    st.v[0][0] = 0x1003;
    st.v[1][0] = 0;
    st.v[4][0] = 0xdead_be42;
    mem_run(GLOBAL_STORE_B8, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        mem.read(0x1000, 8).unwrap(),
        [0xaa, 0xaa, 0xaa, 0x42, 0xaa, 0xaa, 0xaa, 0xaa]
    );
    // global_store_d16_hi_b16 v[69:70], v68, off: v68[31:16] as two bytes.
    st.v[69][0] = 0x1010;
    st.v[70][0] = 0;
    st.v[68][0] = 0x1234_5678;
    mem_run(GLOBAL_STORE_D16_HI_B16, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        mem.read(0x100e, 6).unwrap(),
        [0xaa, 0xaa, 0x34, 0x12, 0xaa, 0xaa]
    );
    // global_load_b96 v[40:42], v[36:37], off.
    mem.write(0x1020, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        .unwrap();
    st.v[36][0] = 0x1020;
    st.v[37][0] = 0;
    mem_run(GLOBAL_LOAD_B96, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[40][0], st.v[41][0], st.v[42][0]),
        (0x0403_0201, 0x0807_0605, 0x0c0b_0a09)
    );
    // flat_load_b64 v[36:37], v[9:10] and flat_load_b128 v[33:36], v[22:23].
    st.v[9][0] = 0x1020;
    st.v[10][0] = 0;
    mem_run(FLAT_LOAD_B64, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!((st.v[36][0], st.v[37][0]), (0x0403_0201, 0x0807_0605));
    mem.write(0x1030, &(0u8..16).collect::<Vec<_>>()).unwrap();
    st.v[22][0] = 0x1030;
    st.v[23][0] = 0;
    mem_run(FLAT_LOAD_B128, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[33][0], st.v[34][0], st.v[35][0], st.v[36][0]),
        (0x0302_0100, 0x0706_0504, 0x0b0a_0908, 0x0f0e_0d0c)
    );
}

/// RDNA3 ISA GLOBAL_ATOMIC_ADD_U32: `tmp = MEM[ADDR]; MEM[ADDR] += DATA; RETURN_DATA = tmp`.
#[test]
fn global_atomic_add_u32_adds_in_memory_and_returns_the_old_value() {
    let mut st = state();
    let mut mem = Memory::default();
    let mut lds = [];
    mem.map(
        "buf".into(),
        0x2000,
        [100u32, 200, 10]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    )
    .unwrap();
    // Returning, SADDR form (`v2, v2, v3, s[4:5] glc`): address = s[4:5] + v2 (32-bit offset), data v3, old value into v2.
    st.exec = 0b11;
    st.s[4] = 0x2000;
    st.s[5] = 0;
    setv(&mut st, 2, &[0, 4]);
    setv(&mut st, 3, &[5, 7]);
    mem_run(GLOBAL_ATOMIC_ADD_RTN, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(getv(&st, 2, 2), vec![100, 200], "returned old values");
    assert_eq!(mem.read(0x2000, 8).unwrap(), [105, 0, 0, 0, 207, 0, 0, 0]);
    // Two active lanes returning from one dword would observe different intermediate values: refused.
    setv(&mut st, 2, &[8, 8]);
    assert!(mem_run(GLOBAL_ATOMIC_ADD_RTN, &mut st, &mut mem, &mut lds).is_err());
    // Non-returning, 64-bit address form (`v[0:1], v2, off`): the adds of two lanes on one dword commute: 10 + 3 + 4.
    setv(&mut st, 0, &[0x2008, 0x2008]);
    setv(&mut st, 1, &[0, 0]);
    setv(&mut st, 2, &[3, 4]);
    mem_run(GLOBAL_ATOMIC_ADD, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(mem.read(0x2008, 4).unwrap(), [17, 0, 0, 0]);
    assert_eq!(getv(&st, 2, 2), vec![3, 4], "no return data is written");
    // Wrapping add.
    mem.write(0x2008, &0xffff_ffffu32.to_le_bytes()).unwrap();
    st.exec = 1;
    setv(&mut st, 2, &[2]);
    mem_run(GLOBAL_ATOMIC_ADD, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(mem.read(0x2008, 4).unwrap(), [1, 0, 0, 0]);
}

#[test]
fn buffer_gl1_inv_is_a_wait_class_no_op_and_every_new_row_is_in_the_table() {
    assert_eq!(
        crate::emu::semantic_class(dec(BUFFER_GL1_INV).1),
        Some("wait")
    );
    // Every opcode of this file decodes to a row the semantics table knows.
    for words in [
        S_OR_SAVEEXEC,
        S_LSHR_B64,
        S_XOR_B64,
        V_ALIGNBYTE_IMM,
        V_BFREV,
        V_NOT,
        V_CLZ,
        V_XOR3,
        V_BCNT,
        V_MIN3_I32,
        V_MED3_I32,
        V_MOVRELS,
        V_MOVRELD,
        V_LSHLREV_B16_IMM,
        V_OR_B16,
        V_CNDMASK_B16,
        V_CVT_F32_F16_E32,
        V_CVT_F16_U16_E64,
        V_CVT_U32_U16_E64_LO,
        V_PK_MUL_F16,
        V_SUB_F32_E64,
        V_FMAC_F32_E64,
        V_TRUNC,
        V_FLOOR,
        V_MED3_F32,
        V_FREXP_EXP,
        V_FREXP_MANT,
        V_FREXP_MANT_ABS,
        V_ADD_CO_CI_E32,
        V_SUB_CO,
        V_SUB_CO_CI_NULL,
        V_SUBREV_CO_CI_NULL,
        V_ADD_NC_E64,
        V_CMP_EQ_F32,
        V_CMP_NEQ_F32,
        V_CMP_O_F32,
        V_CMP_LE_F32,
        V_CMP_GE_I32,
        V_CMP_GE_U32,
        V_CMP_LT_I64_E64,
        V_CMP_NE_U64_E64,
        V_CMPX_NE_U32,
        V_CMPX_LE_I32,
        V_CMPX_LE_U32,
        V_CMPX_LT_U32,
        V_CMPX_EQ_F32,
        V_CMPX_NLT_F32,
        V_CMPX_NGT_F32_E64,
        V_CMPX_LE_F32_E64,
        V_CMPX_GT_I64_E32,
        V_MIN_I32_DPP,
        V_OR_B32_DPP,
        DS_STORE_2ADDR_B32,
        DS_LOAD_2ADDR_STRIDE64_B32,
        DS_LOAD_B96,
        GLOBAL_STORE_B8,
        GLOBAL_STORE_D16_HI_B16,
        GLOBAL_LOAD_B96,
        GLOBAL_ATOMIC_ADD_RTN,
        GLOBAL_ATOMIC_ADD,
        FLAT_LOAD_B64,
        FLAT_LOAD_B128,
    ] {
        let (_, name) = dec(words);
        assert!(
            crate::emu::semantic_class(name).is_some(),
            "{name} has no semantics row"
        );
    }
}

/// Handler-coverage ratchet: every gfx1100 table row whose semantics class is `alu` decodes from its pinned sample
/// words and reaches a handler (the dispatch never answers "unsupported ... opcode"). A semantics row alone only
/// lets `Emulator::new` load a program; this is what keeps a row from being a promise with no code behind it. The
/// instruction runs on a zeroed state with one active lane: any other error (NaN refusal, modifier refusal, ...)
/// is a modeled refusal, not a missing handler.
#[test]
fn every_gfx1100_alu_row_reaches_a_handler() {
    let mut missing = Vec::new();
    let mut checked = 0;
    for row in crate::isa::gfx1100() {
        if crate::emu::semantic_class(row.name) != Some("alu") {
            continue;
        }
        let words: Vec<u32> = row
            .encoding
            .split_whitespace()
            .map(|w| u32::from_str_radix(w, 16).unwrap())
            .collect();
        let Ok((inst, n)) = gfx11::decode(Arch::Gfx1100, &words) else {
            continue; // a sample the decoder rejects is the table gate's finding, not this test's
        };
        if n != words.len() {
            continue;
        }
        checked += 1;
        let mut st = state();
        if let Err(e) = super::execute(Arch::Gfx1100, row.name, &inst, &mut st) {
            if e.contains("unsupported vector opcode")
                || e.contains("unsupported scalar opcode")
                || e.contains("unsupported wide opcode")
                || e.contains("unsupported dot opcode")
                || e.contains("unsupported compare")
                || e.contains("unsupported scalar compare")
            {
                missing.push(format!("{}: {e}", row.name));
            }
        }
    }
    assert!(checked > 200, "only {checked} rows were exercised");
    assert!(
        missing.is_empty(),
        "semantics rows without a handler:\n{}",
        missing.join("\n")
    );
}

// ---- scratch (per-lane private memory) -----------------------------------------------------------------------------

/// `scratch_store_b32 off, v11, s14`
const SCRATCH_STORE_B32_S: &[u32] = &[0xdc690000, 0x000e0b00];
/// `scratch_load_b32 v90, off, off offset:528`
const SCRATCH_LOAD_B32_OFF: &[u32] = &[0xdc510210, 0x5a7c0000];
/// `scratch_store_b64 off, v[13:14], off offset:116`
const SCRATCH_STORE_B64_OFF: &[u32] = &[0xdc6d0074, 0x007c0d00];
/// `scratch_load_b64 v[0:1], v5, off offset:100`
const SCRATCH_LOAD_B64_V: &[u32] = &[0xdc550064, 0x00fc0005];
/// `scratch_store_b128 off, v[72:75], off offset:608`
const SCRATCH_STORE_B128_OFF: &[u32] = &[0xdc750260, 0x007c4800];
/// `scratch_load_b128 v[0:3], off, off offset:608`
const SCRATCH_LOAD_B128_OFF: &[u32] = &[0xdc5d0260, 0x007c0000];
/// `scratch_load_d16_b16 v4, v2, off`
const SCRATCH_LOAD_D16_B16_V: &[u32] = &[0xdc810000, 0x04fc0002];
/// `scratch_load_b96 v[0:2], off, off offset:164`
const SCRATCH_LOAD_B96_OFF: &[u32] = &[0xdc5900a4, 0x007c0000];
/// `scratch_store_b96 off, v[39:41], off offset:164`
const SCRATCH_STORE_B96_OFF: &[u32] = &[0xdc7100a4, 0x007c2700];

fn scratch_state(bytes_per_lane: usize) -> State {
    let mut st = state();
    st.scratch = vec![vec![0; bytes_per_lane]; 32];
    st
}

/// RDNA3 ISA ch. 11 (Scratch): the private byte offset is `SGPR + VGPR (SVE) + OFFSET`, and each lane owns its own
/// private bytes.
#[test]
fn scratch_access_is_lane_private_and_addressed_by_sgpr_vgpr_and_offset() {
    let mut st = scratch_state(1200);
    let mut mem = Memory::default();
    let mut lds = [];
    st.exec = 0b11;
    // scratch_store_b32 off, v11, s14: private offset = s14 = 8, one byte image per lane.
    st.s[14] = 8;
    setv(&mut st, 11, &[0x1122_3344, 0xaabb_ccdd]);
    mem_run(SCRATCH_STORE_B32_S, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(st.scratch[0][8..12], [0x44, 0x33, 0x22, 0x11]);
    assert_eq!(
        st.scratch[1][8..12],
        [0xdd, 0xcc, 0xbb, 0xaa],
        "lane 1 wrote its own private bytes"
    );
    assert_eq!(st.scratch[0][12..16], [0, 0, 0, 0]);
    assert_eq!(
        st.scratch[2][8..12],
        [0, 0, 0, 0],
        "an inactive lane's private memory is untouched"
    );
    // scratch_load_b32 v90, off, off offset:528: only the immediate offset.
    st.scratch[0][528..532].copy_from_slice(&0x0102_0304u32.to_le_bytes());
    st.scratch[1][528..532].copy_from_slice(&0x0506_0708u32.to_le_bytes());
    mem_run(SCRATCH_LOAD_B32_OFF, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(getv(&st, 90, 2), vec![0x0102_0304, 0x0506_0708]);
    // scratch_store_b64 off, v[13:14], off offset:116, read back through a VGPR offset: v5 + 100 = 116.
    setv(&mut st, 13, &[0x10, 0x20]);
    setv(&mut st, 14, &[0x30, 0x40]);
    mem_run(SCRATCH_STORE_B64_OFF, &mut st, &mut mem, &mut lds).unwrap();
    setv(&mut st, 5, &[16, 16]);
    mem_run(SCRATCH_LOAD_B64_V, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[0][0], st.v[1][0], st.v[0][1], st.v[1][1]),
        (0x10, 0x30, 0x20, 0x40)
    );
    // 128- and 96-bit round trips keep dword order.
    for (r, vals) in [(72, [1, 5]), (73, [2, 6]), (74, [3, 7]), (75, [4, 8])] {
        setv(&mut st, r, &vals);
    }
    mem_run(SCRATCH_STORE_B128_OFF, &mut st, &mut mem, &mut lds).unwrap();
    mem_run(SCRATCH_LOAD_B128_OFF, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[0][0], st.v[1][0], st.v[2][0], st.v[3][0]),
        (1, 2, 3, 4)
    );
    assert_eq!(
        (st.v[0][1], st.v[1][1], st.v[2][1], st.v[3][1]),
        (5, 6, 7, 8)
    );
    for (r, vals) in [(39, [0xa, 0xd]), (40, [0xb, 0xe]), (41, [0xc, 0xf])] {
        setv(&mut st, r, &vals);
    }
    mem_run(SCRATCH_STORE_B96_OFF, &mut st, &mut mem, &mut lds).unwrap();
    mem_run(SCRATCH_LOAD_B96_OFF, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!((st.v[0][0], st.v[1][0], st.v[2][0]), (0xa, 0xb, 0xc));
    assert_eq!((st.v[0][1], st.v[1][1], st.v[2][1]), (0xd, 0xe, 0xf));
    // scratch_load_d16_b16 v4, v2, off: the low half is loaded, the high half preserved.
    st.scratch[0][4..6].copy_from_slice(&0xbeefu16.to_le_bytes());
    st.scratch[1][6..8].copy_from_slice(&0xcafeu16.to_le_bytes());
    setv(&mut st, 2, &[4, 6]);
    setv(&mut st, 4, &[0x1234_ffff, 0x5678_ffff]);
    mem_run(SCRATCH_LOAD_D16_B16_V, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(getv(&st, 4, 2), vec![0x1234_beef, 0x5678_cafe]);
    // Outside the private segment (and with no private segment at all): refused.
    st.s[14] = 1199;
    assert!(mem_run(SCRATCH_STORE_B32_S, &mut st, &mut mem, &mut lds).is_err());
    st.s[14] = 1196;
    mem_run(SCRATCH_STORE_B32_S, &mut st, &mut mem, &mut lds).unwrap();
    assert!(mem_run(SCRATCH_STORE_B32_S, &mut state(), &mut mem, &mut lds).is_err());
}

/// A `memory` row that is neither a load nor a store nor a modeled special (atomic, swizzle, permute) would run through
/// the generic load/store path and silently do the wrong thing: every such row must be a named, dedicated handler.
#[test]
fn every_gfx1100_memory_row_is_a_plain_access_or_a_dedicated_handler() {
    const DEDICATED: [&str; 7] = [
        "ds_swizzle_b32",
        "ds_bpermute_b32",
        "ds_add_u32",
        "ds_add_rtn_u32",
        "ds_or_b32",
        "ds_add_f32",
        "global_atomic_add_u32",
    ];
    let mut bad = Vec::new();
    let mut checked = 0;
    for row in crate::isa::gfx1100() {
        if crate::emu::semantic_class(row.name) != Some("memory") {
            continue;
        }
        checked += 1;
        let prefixed = ["s_load_", "ds_", "buffer_", "global_", "flat_", "scratch_"]
            .iter()
            .any(|p| row.name.starts_with(p));
        let plain = row.name.contains("load") || row.name.contains("store");
        if !prefixed || !(plain || DEDICATED.contains(&row.name)) {
            bad.push(row.name);
        }
    }
    assert!(checked > 40, "only {checked} memory rows were seen");
    assert!(
        bad.is_empty(),
        "memory rows with no plain or dedicated handler: {bad:?}"
    );
}

// ---- flat / generic addressing: the shared (LDS) aperture ---------------------------------------------------------

/// `s_mov_b64 s[2:3], src_shared_base`
const S_MOV_B64_SHARED: &[u32] = &[0xbe8201eb];
/// `s_mov_b64 s[2:3], src_private_base`
const S_MOV_B64_PRIVATE: &[u32] = &[0xbe8201ed];
/// `flat_store_b32 v[0:1], v2 offset:8`
const FLAT_STORE_B32: &[u32] = &[0xdc680008, 0x007c0200];
/// `flat_load_b32 v3, v[4:5]`
const FLAT_LOAD_B32: &[u32] = &[0xdc500000, 0x037c0004];
/// `flat_load_b128 v[8:11], v[4:5]`
const FLAT_LOAD_B128_G: &[u32] = &[0xdc5c0000, 0x087c0004];
/// `ds_load_b32 v6, v7`
const DS_LOAD_B32: &[u32] = &[0xd8d80000, 0x06000007];

/// The aperture operands read as `{aperture_hi, 0}`: hi dword 2 (shared) and 3 (private), low dword 0.
#[test]
fn src_shared_base_and_src_private_base_are_64_bit_aperture_constants() {
    let mut st = state();
    run(S_MOV_B64_SHARED, &mut st).unwrap();
    assert_eq!((st.s[2], st.s[3]), (0, 2));
    run(S_MOV_B64_PRIVATE, &mut st).unwrap();
    assert_eq!((st.s[2], st.s[3]), (0, 3));
}

/// A flat access whose generic address has the shared aperture's high dword is an LDS access at the low 32 bits
/// (offset added first): a flat store is seen by `ds_load`, defined-byte tracking and race logging see it as a
/// `ds_*` access, a global flat access is unchanged, and an out-of-range aperture access or an unmapped address errors.
#[test]
fn flat_access_to_the_shared_aperture_is_an_lds_access() {
    use super::super::{memory::LdsTouch, WriteSet};
    let mut st = state();
    st.exec = 0b11;
    let mut mem = Memory::default();
    mem.map("g".into(), 0x1000, vec![0xaa; 64]).unwrap();
    let mut lds = vec![0u8; 2048];
    let mut defined = WriteSet::new(2048);
    let mut log: Vec<LdsTouch> = Vec::new();
    let (store, store_name) = dec(FLAT_STORE_B32);
    // lane 0 -> aperture {hi 2, lo 0x100} + offset 8 = LDS 0x108; lane 1 -> {hi 2, lo 0x200} + 8 = LDS 0x208.
    setv(&mut st, 0, &[0x100, 0x200]);
    setv(&mut st, 1, &[2, 2]);
    setv(&mut st, 2, &[0x1122_3344, 0x5566_7788]);
    memory::execute_tracked(
        store_name,
        &store,
        &mut st,
        &mut mem,
        &mut lds,
        Some(&mut defined),
        Some(&mut log),
    )
    .unwrap();
    assert_eq!(lds[0x108..0x10c], [0x44, 0x33, 0x22, 0x11]);
    assert_eq!(lds[0x208..0x20c], [0x88, 0x77, 0x66, 0x55]);
    assert_eq!(
        mem.read(0x1000, 64).unwrap(),
        [0xaa; 64],
        "global memory untouched"
    );
    assert!(
        defined.first_unwritten(0x108, 4).is_none() && defined.first_unwritten(0x208, 4).is_none()
    );
    let touched: Vec<(usize, usize, bool)> = log.iter().map(|t| (t.lo, t.len, t.write)).collect();
    assert_eq!(touched, vec![(0x108, 4, true), (0x208, 4, true)]);
    // ds_load sees it.
    st.exec = 0b01;
    setv(&mut st, 7, &[0x108]);
    mem_run(DS_LOAD_B32, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(st.v[6][0], 0x1122_3344);
    // flat_load_b32 through the aperture reads it back (offset-free address {2, 0x208}); b128 reads 4 dwords at 0x100.
    let (load, load_name) = dec(FLAT_LOAD_B32);
    setv(&mut st, 4, &[0x208]);
    setv(&mut st, 5, &[2]);
    log.clear();
    memory::execute_tracked(
        load_name,
        &load,
        &mut st,
        &mut mem,
        &mut lds,
        Some(&mut defined),
        Some(&mut log),
    )
    .unwrap();
    assert_eq!(st.v[3][0], 0x5566_7788);
    assert_eq!(
        log.iter()
            .map(|t| (t.lo, t.len, t.write))
            .collect::<Vec<_>>(),
        vec![(0x208, 4, false)]
    );
    // A flat load of LDS bytes nobody wrote is the same strict-LDS error as ds_load.
    setv(&mut st, 4, &[0x300]);
    let strict = memory::execute_tracked(
        load_name,
        &load,
        &mut st,
        &mut mem,
        &mut lds,
        Some(&mut defined),
        None,
    );
    assert!(strict.unwrap_err().contains("never written"));
    // Out of the LDS allocation through the aperture: error. The 64-bit address {2, 0xfffffff0} is not global either.
    setv(&mut st, 4, &[0xffff_fff0]);
    assert!(mem_run(FLAT_LOAD_B32, &mut st, &mut mem, &mut lds).is_err());
    // Global flat addresses (hi dword 0) are unchanged: load 16 bytes from the mapped region; hi dword 4 is unmapped.
    mem.write(0x1010, &(0u8..16).collect::<Vec<_>>()).unwrap();
    setv(&mut st, 4, &[0x1010]);
    setv(&mut st, 5, &[0]);
    mem_run(FLAT_LOAD_B128_G, &mut st, &mut mem, &mut lds).unwrap();
    assert_eq!(
        (st.v[8][0], st.v[9][0], st.v[10][0], st.v[11][0]),
        (0x0302_0100, 0x0706_0504, 0x0b0a_0908, 0x0f0e_0d0c)
    );
    setv(&mut st, 5, &[4]);
    assert!(mem_run(FLAT_LOAD_B128_G, &mut st, &mut mem, &mut lds).is_err());
    // The private aperture (hi 3) is not modeled: refused, not mapped onto anything.
    setv(&mut st, 5, &[3]);
    assert!(mem_run(FLAT_LOAD_B128_G, &mut st, &mut mem, &mut lds).is_err());
}
