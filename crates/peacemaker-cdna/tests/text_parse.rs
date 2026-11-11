// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `text::parse_line`: llvm-mc ground truth (the encodings below are what
//! `llvm-mc -triple=amdgcn -mcpu=gfx942 -show-encoding` printed for each
//! spelling), `Display` round trips over every opcode table, structured
//! modifier sweeps, a decode-driven round trip, and the named parse errors.
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::enc::mubuf::{Mubuf, MubufOp};
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::enc::sop::{
    waitcnt_imm, Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp,
};
use peacemaker_cdna::enc::vop::{
    Dpp, DppCtrl, Sdwa, SdwaSel, SdwaUnused, Vop1, Vop1Op, Vop2, Vop2K, Vop2KOp, Vop2Op, VopExt,
    Vopc, VopcOp,
};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp, Vop3p, Vop3pOp};
use peacemaker_cdna::text::parse_line;
use peacemaker_cdna::{Arch, Format, Inst, Src};

const ARCH: Arch = Arch::Gfx942;

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse(text: &str) -> Inst {
    match parse_line(text, ARCH) {
        Ok(i) => i,
        Err(e) => panic!("`{text}` did not parse: {e}"),
    }
}

fn parse_err(text: &str) -> String {
    match parse_line(text, ARCH) {
        Ok(i) => panic!("`{text}` parsed as {i:?}, expected an error"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------------
// llvm-mc ground truth.

/// `(assembly text, llvm-mc gfx942 encoding bytes)`.
const LLVM: &[(&str, &[u8])] = &[
    // MFMA, with the pre-gfx940 spellings llvm still accepts.
    ("v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[0:3]", &[0x00, 0x80, 0xe1, 0xd3, 0x00, 0x05, 0x02, 0x04]),
    ("v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]", &[0x00, 0x80, 0xe1, 0xd3, 0x00, 0x05, 0x02, 0x04]),
    ("v_mfma_f32_32x32x8bf16_1k a[0:15], v[0:1], v[2:3], a[0:15]", &[0x00, 0x80, 0xe0, 0xd3, 0x00, 0x05, 0x02, 0x04]),
    ("v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3] cbsz:1 abid:2 blgp:3", &[0x00, 0x91, 0xe1, 0xd3, 0x00, 0x05, 0x02, 0x64]),
    ("v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]", &[0x00, 0x80, 0xcd, 0xd3, 0x00, 0x05, 0x02, 0x04]),
    ("v_mfma_f32_16x16x4f32 a[0:3], v0, v1, a[0:3]", &[0x00, 0x80, 0xc5, 0xd3, 0x00, 0x03, 0x02, 0x04]),
    ("v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:5], v[6:7], v[0:3]", &[0x00, 0x00, 0xf3, 0xd3, 0x04, 0x0d, 0x02, 0x04]),
    ("v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:5], v[6:7], 0", &[0x00, 0x00, 0xf3, 0xd3, 0x04, 0x0d, 0x02, 0x02]),
    ("v_mfma_f64_16x16x4f64 v[0:7], v[8:9], v[10:11], v[0:7]", &[0x00, 0x00, 0xee, 0xd3, 0x08, 0x15, 0x02, 0x04]),
    ("v_mfma_f64_16x16x4_f64 v[0:7], v[8:9], v[10:11], v[0:7] neg:[1,0,1]", &[0x00, 0x00, 0xee, 0xd3, 0x08, 0x15, 0x02, 0xa4]),
    ("v_mfma_i32_16x16x32i8 a[0:3], v[0:1], v[2:3], a[0:3]", &[0x00, 0x80, 0xd7, 0xd3, 0x00, 0x05, 0x02, 0x04]),
    ("v_mfma_f32_16x16x16_bf16 v[0:3], a[0:1], v[2:3], v[0:3]", &[0x00, 0x00, 0xe1, 0xd3, 0x00, 0x05, 0x02, 0x0c]),
    ("v_accvgpr_read_b32 v0, a1", &[0x00, 0x40, 0xd8, 0xd3, 0x01, 0x01, 0x00, 0x18]),
    ("v_accvgpr_write_b32 a0, 1", &[0x00, 0x40, 0xd9, 0xd3, 0x81, 0x00, 0x00, 0x18]),
    ("v_accvgpr_write_b32 a0, v1", &[0x00, 0x40, 0xd9, 0xd3, 0x01, 0x01, 0x00, 0x18]),
    // FLAT / GLOBAL / SCRATCH / MUBUF / SMEM.
    ("global_load_dword v0, v[2:3], off sc0 sc1 nt", &[0x00, 0x80, 0x53, 0xde, 0x02, 0x00, 0x7f, 0x00]),
    ("global_load_dword v0, v1, s[2:3] offset:16", &[0x10, 0x80, 0x50, 0xdc, 0x01, 0x00, 0x02, 0x00]),
    ("scratch_load_dword v0, off, off", &[0x00, 0x40, 0x50, 0xdc, 0x00, 0x00, 0x7f, 0x00]),
    ("scratch_load_dword v0, v1, s5", &[0x00, 0x60, 0x50, 0xdc, 0x01, 0x00, 0x05, 0x00]),
    ("scratch_load_dword v0, off, s5 offset:8", &[0x08, 0x40, 0x50, 0xdc, 0x00, 0x00, 0x05, 0x00]),
    ("scratch_store_dword off, v1, off", &[0x00, 0x40, 0x70, 0xdc, 0x00, 0x01, 0x7f, 0x00]),
    ("flat_load_dword v0, v[2:3]", &[0x00, 0x00, 0x50, 0xdc, 0x02, 0x00, 0x00, 0x00]),
    ("flat_store_dword v[2:3], v1 offset:4", &[0x04, 0x00, 0x70, 0xdc, 0x02, 0x01, 0x00, 0x00]),
    ("global_atomic_add v0, v[2:3], v1, off sc0", &[0x00, 0x80, 0x09, 0xdd, 0x02, 0x01, 0x7f, 0x00]),
    ("global_atomic_add_f32 v[2:3], v1, off", &[0x00, 0x80, 0x34, 0xdd, 0x02, 0x01, 0x7f, 0x00]),
    ("global_store_dword v[2:3], v1, off sc0 sc1 nt", &[0x00, 0x80, 0x73, 0xde, 0x02, 0x01, 0x7f, 0x00]),
    ("buffer_load_dword v0, off, s[4:7], 0", &[0x00, 0x00, 0x50, 0xe0, 0x00, 0x00, 0x01, 0x80]),
    ("buffer_load_dword v0, v[2:3], s[4:7], s8 idxen offen offset:4095", &[0xff, 0x3f, 0x50, 0xe0, 0x02, 0x00, 0x01, 0x08]),
    ("buffer_load_dword a0, v1, s[4:7], 0 offen", &[0x00, 0x10, 0x50, 0xe0, 0x01, 0x00, 0x81, 0x80]),
    ("s_load_dword s0, s[2:3], 0x10", &[0x01, 0x00, 0x02, 0xc0, 0x10, 0x00, 0x00, 0x00]),
    ("s_load_dword s0, s[2:3], 16", &[0x01, 0x00, 0x02, 0xc0, 0x10, 0x00, 0x00, 0x00]),
    ("s_load_dword s0, s[2:3], s4 offset:0x10", &[0x01, 0x40, 0x02, 0xc0, 0x10, 0x00, 0x00, 0x08]),
    ("s_load_dword s0, s[2:3], s4 glc", &[0x01, 0x00, 0x01, 0xc0, 0x04, 0x00, 0x00, 0x00]),
    // SALU / SOPP.
    ("s_waitcnt vmcnt(0) & lgkmcnt(0)", &[0x70, 0x00, 0x8c, 0xbf]),
    ("s_waitcnt vmcnt(0), lgkmcnt(0)", &[0x70, 0x00, 0x8c, 0xbf]),
    ("s_waitcnt vmcnt(0) lgkmcnt(0)", &[0x70, 0x00, 0x8c, 0xbf]),
    ("s_waitcnt 0", &[0x00, 0x00, 0x8c, 0xbf]),
    ("s_waitcnt vmcnt(1) lgkmcnt(2) expcnt(3)", &[0x31, 0x02, 0x8c, 0xbf]),
    ("s_waitcnt lgkmcnt(0)", &[0x7f, 0xc0, 0x8c, 0xbf]),
    ("s_waitcnt vmcnt(0)", &[0x70, 0x0f, 0x8c, 0xbf]),
    ("s_endpgm", &[0x00, 0x00, 0x81, 0xbf]),
    ("s_endpgm 0", &[0x00, 0x00, 0x81, 0xbf]),
    ("s_nop 1", &[0x01, 0x00, 0x80, 0xbf]),
    ("s_branch 5", &[0x05, 0x00, 0x82, 0xbf]),
    ("s_cbranch_execz 5", &[0x05, 0x00, 0x88, 0xbf]),
    ("s_sendmsg sendmsg(MSG_INTERRUPT)", &[0x01, 0x00, 0x90, 0xbf]),
    ("s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)", &[0x22, 0x00, 0x90, 0xbf]),
    ("s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)", &[0x03, 0x00, 0x90, 0xbf]),
    ("s_sendmsg sendmsg(MSG_SYSMSG, SYSMSG_OP_REG_RD)", &[0x2f, 0x00, 0x90, 0xbf]),
    ("s_getreg_b32 s0, hwreg(HW_REG_MODE, 4, 2)", &[0x01, 0x09, 0x80, 0xb8]),
    ("s_getreg_b32 s0, hwreg(2)", &[0x02, 0xf8, 0x80, 0xb8]),
    ("s_getreg_b32 s0, hwreg(HW_REG_XCC_ID)", &[0x14, 0xf8, 0x80, 0xb8]),
    ("s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), s1", &[0x01, 0x18, 0x01, 0xb9]),
    ("s_setreg_b32 hwreg(HW_REG_MODE), s1", &[0x01, 0xf8, 0x01, 0xb9]),
    ("s_trap 2", &[0x02, 0x00, 0x92, 0xbf]),
    ("s_sleep 1", &[0x01, 0x00, 0x8e, 0xbf]),
    ("s_setprio 3", &[0x03, 0x00, 0x8f, 0xbf]),
    ("s_cmp_eq_u32 s0, 5", &[0x00, 0x85, 0x06, 0xbf]),
    ("s_bitcmp1_b32 s0, 5", &[0x00, 0x85, 0x0d, 0xbf]),
    ("s_movk_i32 s0, -1", &[0xff, 0xff, 0x00, 0xb0]),
    ("s_movk_i32 s0, 0xffff", &[0xff, 0xff, 0x00, 0xb0]),
    ("s_mov_b32 s0, 0x12345678", &[0xff, 0x00, 0x80, 0xbe, 0x78, 0x56, 0x34, 0x12]),
    ("s_mov_b64 s[0:1], 1.0", &[0xf2, 0x01, 0x80, 0xbe]),
    ("s_mov_b64 s[0:1], 0x1234", &[0xff, 0x01, 0x80, 0xbe, 0x34, 0x12, 0x00, 0x00]),
    ("s_mov_b64 s[0:1], -17", &[0xff, 0x01, 0x80, 0xbe, 0xef, 0xff, 0xff, 0xff]),
    ("s_mov_b64 exec, -1", &[0xc1, 0x01, 0xfe, 0xbe]),
    ("s_and_saveexec_b64 s[0:1], vcc", &[0x6a, 0x20, 0x80, 0xbe]),
    ("s_lshl_b64 s[0:1], s[2:3], 4", &[0x02, 0x84, 0x80, 0x8e]),
    // VOP1 / VOP2 / VOPC, 32-bit forms and constants.
    ("v_add_f32 v0, v1, v2", &[0x01, 0x05, 0x00, 0x02]),
    ("v_add_f32_e32 v0, v1, v2", &[0x01, 0x05, 0x00, 0x02]),
    ("v_mov_b32 v0, 0x3f800000", &[0xf2, 0x02, 0x00, 0x7e]),
    ("v_mov_b32 v0, 1.0", &[0xf2, 0x02, 0x00, 0x7e]),
    ("v_mov_b32 v0, 0.1", &[0xff, 0x02, 0x00, 0x7e, 0xcd, 0xcc, 0xcc, 0x3d]),
    ("v_mov_b32 v0, -1", &[0xc1, 0x02, 0x00, 0x7e]),
    ("v_mov_b32 v0, 0xfffffff0", &[0xd0, 0x02, 0x00, 0x7e]),
    ("v_mov_b32 v0, 65", &[0xff, 0x02, 0x00, 0x7e, 0x41, 0x00, 0x00, 0x00]),
    ("v_mov_b32 v0, lit(65)", &[0xff, 0x02, 0x00, 0x7e, 0x41, 0x00, 0x00, 0x00]),
    ("v_mov_b32 v0, s1", &[0x01, 0x02, 0x00, 0x7e]),
    ("v_mov_b64_e32 v[0:1], 1", &[0x81, 0x70, 0x00, 0x7e]),
    ("v_mov_b64 v[0:1], 0x1234", &[0xff, 0x70, 0x00, 0x7e, 0x34, 0x12, 0x00, 0x00]),
    ("v_add_f16 v0, 0.5, v1", &[0xf0, 0x02, 0x00, 0x3e]),
    ("v_add_f16 v0, 1.0, v1", &[0xf2, 0x02, 0x00, 0x3e]),
    ("v_cvt_f32_f64 v0, v[2:3]", &[0x02, 0x1f, 0x00, 0x7e]),
    ("v_cvt_f32_f64 v0, 1.0", &[0xf2, 0x1e, 0x00, 0x7e]),
    ("v_add_co_u32 v0, vcc, v1, v2", &[0x01, 0x05, 0x00, 0x32]),
    ("v_cndmask_b32 v0, v1, v2, vcc", &[0x01, 0x05, 0x00, 0x00]),
    ("v_readfirstlane_b32 s0, v1", &[0x01, 0x05, 0x00, 0x7e]),
    ("v_nop", &[0x00, 0x00, 0x00, 0x7e]),
    ("v_nop_e64", &[0x00, 0x00, 0x40, 0xd1, 0x00, 0x00, 0x00, 0x00]),
    ("v_cmpx_lt_u32_e32 vcc, v0, v1", &[0x00, 0x03, 0xb2, 0x7d]),
    ("v_fmaak_f32 v0, v1, v2, 1.0", &[0x01, 0x05, 0x00, 0x30, 0x00, 0x00, 0x80, 0x3f]),
    ("v_fmamk_f32 v0, v1, 0x3f800000, v2", &[0x01, 0x05, 0x00, 0x2e, 0x00, 0x00, 0x80, 0x3f]),
    // DPP / SDWA.
    ("v_mov_b32_dpp v0, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf bound_ctrl:0", &[0xfa, 0x02, 0x00, 0x7e, 0x01, 0xe4, 0x08, 0xff]),
    ("v_mov_b32 v0, v1 row_shl:1", &[0xfa, 0x02, 0x00, 0x7e, 0x01, 0x01, 0x01, 0xff]),
    ("v_add_f32_dpp v0, -v1, |v2| row_shl:1 row_mask:0xf bank_mask:0xf", &[0xfa, 0x04, 0x00, 0x02, 0x01, 0x01, 0x91, 0xff]),
    ("v_mov_b32_sdwa v0, v1 dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:WORD_1", &[0xf9, 0x02, 0x00, 0x7e, 0x01, 0x00, 0x05, 0x00]),
    ("v_mov_b32 v0, v1 dst_sel:BYTE_0", &[0xf9, 0x02, 0x00, 0x7e, 0x01, 0x10, 0x06, 0x00]),
    ("v_mov_b32_sdwa v0, v1 src0_sel:WORD_1", &[0xf9, 0x02, 0x00, 0x7e, 0x01, 0x16, 0x05, 0x00]),
    ("v_add_f32_sdwa v0, v1, v2", &[0xf9, 0x04, 0x00, 0x02, 0x01, 0x16, 0x06, 0x06]),
    ("v_add_f32_sdwa v0, -v1, |v2| clamp mul:2 dst_sel:BYTE_1 dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:BYTE_3", &[0xf9, 0x04, 0x00, 0x02, 0x01, 0x69, 0x14, 0x23]),
    ("v_add_u32_sdwa v0, sext(v1), v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD", &[0xf9, 0x04, 0x00, 0x68, 0x01, 0x06, 0x08, 0x06]),
    ("v_cvt_f32_fp8_sdwa v0, v1 src0_sel:BYTE_1", &[0xf9, 0xa8, 0x00, 0x7e, 0x01, 0x06, 0x01, 0x00]),
    ("v_cvt_pk_f32_fp8_sdwa v[0:1], v1 src0_sel:WORD_1", &[0xf9, 0xac, 0x00, 0x7e, 0x01, 0x06, 0x05, 0x00]),
    ("v_cmp_lt_f32_sdwa vcc, v0, v1 src0_sel:WORD_1 src1_sel:DWORD", &[0xf9, 0x02, 0x82, 0x7c, 0x00, 0x00, 0x05, 0x06]),
    ("v_cmp_lt_f32_sdwa s[0:1], v0, v1 src0_sel:WORD_1 src1_sel:DWORD", &[0xf9, 0x02, 0x82, 0x7c, 0x00, 0x80, 0x05, 0x06]),
    ("v_cmp_lt_f32_sdwa vcc, v0, v1", &[0xf9, 0x02, 0x82, 0x7c, 0x00, 0x00, 0x06, 0x06]),
    // VOP3 and its modifier spellings.
    ("v_add_f32_e64 v0, v1, v2", &[0x00, 0x00, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x00]),
    ("v_add_f32_e64 v0, -v1, |v2| clamp mul:2", &[0x00, 0x82, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x28]),
    ("v_add_f32 v0, -|v1|, abs(v2)", &[0x00, 0x03, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x20]),
    ("v_add_f32_e64 v0, neg(v1), abs(v2)", &[0x00, 0x02, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x20]),
    ("v_add_f32_e64 v0, neg(1.0), v2", &[0x00, 0x00, 0x01, 0xd1, 0xf2, 0x04, 0x02, 0x20]),
    ("v_add_f32_e64 v0, -1.0, v2", &[0x00, 0x00, 0x01, 0xd1, 0xf3, 0x04, 0x02, 0x00]),
    ("v_add_f32_e64 v0, v1, v2 mul:2", &[0x00, 0x00, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x08]),
    ("v_add_f32_e64 v0, v1, v2 div:2", &[0x00, 0x00, 0x01, 0xd1, 0x01, 0x05, 0x02, 0x18]),
    ("v_add_co_u32 v0, s[0:1], v1, v2", &[0x00, 0x00, 0x19, 0xd1, 0x01, 0x05, 0x02, 0x00]),
    ("v_cndmask_b32 v0, v1, v2, s[0:1]", &[0x00, 0x00, 0x00, 0xd1, 0x01, 0x05, 0x02, 0x00]),
    ("v_cvt_pk_fp8_f32 v0, v1, v2 op_sel:[0,0,1]", &[0x00, 0x40, 0xa2, 0xd2, 0x01, 0x05, 0x02, 0x00]),
    ("v_perm_b32 v0, v1, v2, v3", &[0x00, 0x00, 0xed, 0xd1, 0x01, 0x05, 0x0e, 0x04]),
    ("v_lshl_add_u64 v[0:1], v[2:3], 2, v[4:5]", &[0x00, 0x00, 0x08, 0xd2, 0x02, 0x05, 0x11, 0x04]),
    ("v_readlane_b32 s0, v1, 5", &[0x00, 0x00, 0x89, 0xd2, 0x01, 0x0b, 0x01, 0x00]),
    ("v_writelane_b32 v0, s1, m0", &[0x00, 0x00, 0x8a, 0xd2, 0x01, 0xf8, 0x00, 0x00]),
    ("v_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[6:7]", &[0x00, 0x02, 0xe8, 0xd1, 0x04, 0x0b, 0x1a, 0x04]),
    ("v_div_scale_f32 v0, vcc, v1, v2, v3", &[0x00, 0x6a, 0xe0, 0xd1, 0x01, 0x05, 0x0e, 0x04]),
    ("v_cmp_class_f32_e64 s[0:1], v0, 3", &[0x00, 0x00, 0x10, 0xd0, 0x00, 0x07, 0x01, 0x00]),
    ("v_lshlrev_b64 v[0:1], 2, v[2:3]", &[0x00, 0x00, 0x8f, 0xd2, 0x82, 0x04, 0x02, 0x00]),
    ("v_cmp_eq_u64_e64 s[0:1], v[2:3], 0.15915494309189532", &[0x00, 0x00, 0xea, 0xd0, 0x02, 0xf1, 0x01, 0x00]),
    // VOP3P.
    ("v_pk_fma_f16 v0, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,1,0] neg_lo:[1,0,0] neg_hi:[0,0,1] clamp", &[0x00, 0x94, 0x8e, 0xd3, 0x01, 0x05, 0x0e, 0x3c]),
    ("v_fma_mix_f32 v0, -v1, v2, neg(1.0) op_sel:[1,0,0] op_sel_hi:[1,0,1]", &[0x00, 0x48, 0xa0, 0xd3, 0x01, 0x05, 0xca, 0xab]),
    ("v_pk_mul_f32 v[0:1], v[2:3], v[4:5] op_sel:[1,0] op_sel_hi:[0,1]", &[0x00, 0x48, 0xb1, 0xd3, 0x02, 0x09, 0x02, 0x10]),
    ("v_pk_mul_f32 v[0:1], v[2:3], 1.0", &[0x00, 0x40, 0xb1, 0xd3, 0x02, 0xe5, 0x01, 0x18]),
    ("v_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7] neg_lo:[1,0,1]", &[0x00, 0x40, 0xb0, 0xd3, 0x02, 0x09, 0x1a, 0xbc]),
    // DS.
    ("ds_swizzle_b32 v0, v1 offset:swizzle(QUAD_PERM,0,1,2,3)", &[0xe4, 0x80, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(BITMASK_PERM, \"01pi0\")", &[0x06, 0x09, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(SWAP,16)", &[0x1f, 0x40, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(REVERSE,8)", &[0x1f, 0x1c, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(BROADCAST,8,3)", &[0x78, 0x00, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(ROTATE,1,5)", &[0xa0, 0xc4, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_swizzle_b32 v0, v1 offset:swizzle(FFT,3)", &[0x03, 0xe0, 0x7a, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_read_b32 v0, v1 offset:0x10", &[0x10, 0x00, 0x6c, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_read_b32 v0, v1 offset:65535", &[0xff, 0xff, 0x6c, 0xd8, 0x01, 0x00, 0x00, 0x00]),
    ("ds_write2_b32 v1, v2, v3 offset0:1 offset1:2", &[0x01, 0x02, 0x1c, 0xd8, 0x01, 0x02, 0x03, 0x00]),
    ("ds_read2_b64 v[0:3], v4 offset0:1", &[0x01, 0x00, 0xee, 0xd8, 0x04, 0x00, 0x00, 0x00]),
    ("ds_write_b128 v1, v[2:5]", &[0x00, 0x00, 0xbe, 0xd9, 0x01, 0x02, 0x00, 0x00]),
];

#[test]
fn llvm_mc_spellings_encode_to_llvm_bytes() {
    let mut failures = Vec::new();
    for &(text, bytes) in LLVM {
        match parse_line(text, ARCH).and_then(|inst| inst.words(ARCH)) {
            Ok(w) if w == words_of(bytes) => {}
            Ok(w) => failures.push(format!(
                "`{text}`: parsed to {w:08x?}, llvm-mc has {:08x?}",
                words_of(bytes)
            )),
            Err(e) => failures.push(format!("`{text}`: {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn comments_whitespace_and_case() {
    let a = parse("s_mov_b32 s0, 5");
    assert_eq!(
        parse("  s_mov_b32   s0 ,  5  // 0000000000000000: BE800085"),
        a
    );
    assert_eq!(parse("s_mov_b32 s0, 5 ; trailing"), a);
    assert_eq!(parse("S_MOV_B32 s0, 5"), a);
    assert_eq!(parse("v_mov_b32 v[ 0 : 0 ], 1"), parse("v_mov_b32 v0, 1"),);
}

#[test]
fn e32_is_tried_first_then_e64() {
    assert_eq!(parse("v_add_f32 v0, v1, v2").format(), Format::Vop2);
    assert_eq!(parse("v_add_f32 v0, s[0:0], v2").format(), Format::Vop2);
    assert_eq!(parse("v_add_f32 v0, v1, s2").format(), Format::Vop3);
    assert_eq!(parse("v_add_f32 v0, v1, v2 clamp").format(), Format::Vop3);
    assert_eq!(parse("v_add_f32_e64 v0, v1, v2").format(), Format::Vop3);
    assert_eq!(parse("v_cmp_lt_f32 vcc, v0, v1").format(), Format::Vopc);
    assert_eq!(parse("v_cmp_lt_f32 s[0:1], v0, v1").format(), Format::Vop3);
    assert_eq!(parse("v_fma_f32 v0, v1, v2, v3").format(), Format::Vop3);
    assert_eq!(parse("v_fma_f32_e64 v0, v1, v2, v3").format(), Format::Vop3);
}

#[test]
fn aliases_parse_to_the_same_instruction() {
    for (a, b) in [
        (
            "v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], a[0:3]",
            "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]",
        ),
        (
            "v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]",
            "v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]",
        ),
        (
            "v_mfma_f32_32x32x2f32 a[0:15], v0, v1, a[0:15]",
            "v_mfma_f32_32x32x2_f32 a[0:15], v0, v1, a[0:15]",
        ),
        (
            "s_waitcnt vmcnt(0) & lgkmcnt(0)",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
        ),
        ("v_add_f32 v0, v1, v2", "v_add_f32_e32 v0, v1, v2"),
        (
            "ds_read_b32 v0, v1 offset:0x10",
            "ds_read_b32 v0, v1 offset:16",
        ),
        (
            "v_add_f32 v0, -|v1|, v2",
            "v_add_f32_e64 v0, neg(abs(v1)), v2",
        ),
    ] {
        assert_eq!(parse(a), parse(b), "{a} / {b}");
    }
}

// ---------------------------------------------------------------------------
// Display round trips.

fn sreg(n: u8, dwords: u8) -> Option<Src> {
    (dwords > 0).then_some(Src::Sgpr(n))
}
fn vreg(n: u8, dwords: u8) -> Option<Src> {
    (dwords > 0).then_some(Src::Vgpr(n))
}

fn vop3_inst(op: Vop3Op) -> Inst {
    let (wd, w0, w1, w2) = op.widths();
    let ws = [w0, w1, w2];
    let mut src: [Option<Src>; 3] = [None; 3];
    for j in 0..3 {
        if ws[j] > 0 {
            src[j] = Some(Src::Vgpr(12 + 4 * j as u8));
        }
    }
    let mut dst = if wd == 0 { Src::Vgpr(0) } else { Src::Vgpr(10) };
    match op {
        Vop3Op::C(_) => dst = Src::Sgpr(20),
        Vop3Op::Only(Vop3OnlyOp::ReadlaneB32) => {
            dst = Src::Sgpr(10);
            src[1] = Some(Src::Int(3));
        }
        Vop3Op::Only(Vop3OnlyOp::WritelaneB32) => {
            src[0] = Some(Src::Sgpr(13));
            src[1] = Some(Src::Int(3));
        }
        Vop3Op::V2(
            Vop2Op::CndmaskB32 | Vop2Op::AddcCoU32 | Vop2Op::SubbCoU32 | Vop2Op::SubbrevCoU32,
        ) => src[2] = Some(Src::Sgpr(22)),
        _ => {}
    }
    Inst::Vop3(Vop3 {
        op,
        dst,
        sdst: op.is_vop3b().then_some(Src::Sgpr(20)),
        src,
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    })
}

/// Every opcode of every table with canonical operands (the emulator
/// coverage table's universe, plus every promoted VOP3 opcode).
fn universe() -> Vec<(String, Inst)> {
    let mut out: Vec<(String, Inst)> = Vec::new();
    let mut add = |name: &str, inst: Inst| out.push((name.to_string(), inst));
    for &op in Sop2Op::ALL {
        let (_, a, b) = op.widths();
        add(
            op.name(),
            Inst::Sop2(Sop2 {
                op,
                sdst: Src::Sgpr(10),
                ssrc0: Src::Sgpr(if a > 1 { 12 } else { 13 }),
                ssrc1: Src::Sgpr(if b > 1 { 14 } else { 15 }),
            }),
        );
    }
    for &op in Sop1Op::ALL {
        let (d, a) = op.widths();
        add(
            op.name(),
            Inst::Sop1(Sop1 {
                op,
                sdst: sreg(10, d),
                ssrc0: sreg(12, a),
            }),
        );
    }
    for &op in SopkOp::ALL {
        let simm16 = if matches!(op, SopkOp::GetregB32 | SopkOp::SetregB32) {
            (3 << 11) | 1
        } else {
            5
        };
        add(
            op.name(),
            Inst::Sopk(Sopk {
                op,
                sdst: Src::Sgpr(10),
                simm16,
            }),
        );
    }
    for &op in SopcOp::ALL {
        add(
            op.name(),
            Inst::Sopc(Sopc {
                op,
                ssrc0: Src::Sgpr(12),
                ssrc1: Src::Sgpr(14),
            }),
        );
    }
    for &op in SoppOp::ALL {
        add(op.name(), Inst::Sopp(Sopp { op, simm16: 0 }));
    }
    for &op in SmemOp::ALL {
        let (sdata, sbase) = match op {
            SmemOp::DcacheInv => (None, None),
            SmemOp::Memtime | SmemOp::Memrealtime => (Some(Src::Sgpr(20)), None),
            _ if op.buffer() => (Some(Src::Sgpr(20)), Some(Src::Sgpr(4))),
            _ => (Some(Src::Sgpr(20)), Some(Src::Sgpr(2))),
        };
        add(
            op.name(),
            Inst::Smem(Smem {
                op,
                sdata,
                sbase,
                offset: SmemOffset::Imm(0),
                glc: false,
            }),
        );
    }
    for &op in Vop1Op::ALL {
        let (d, a) = op.widths();
        let (vdst, src0) = match op {
            Vop1Op::ReadfirstlaneB32 => (Src::Sgpr(10), Src::Vgpr(12)),
            Vop1Op::AccvgprMovB32 => (Src::Agpr(10), Src::Agpr(12)),
            Vop1Op::Nop => (Src::Vgpr(0), Src::Sgpr(0)),
            _ => (
                vreg(10, d).unwrap_or(Src::Vgpr(0)),
                vreg(12, a).unwrap_or(Src::Sgpr(0)),
            ),
        };
        add(
            op.name(),
            Inst::Vop1(Vop1 {
                op,
                vdst,
                src0,
                ext: VopExt::None,
            }),
        );
    }
    for &op in Vop2Op::ALL {
        add(
            op.name(),
            Inst::Vop2(Vop2 {
                op,
                vdst: Src::Vgpr(10),
                src0: Src::Vgpr(12),
                src1: Src::Vgpr(14),
                ext: VopExt::None,
            }),
        );
    }
    for &op in Vop2KOp::ALL {
        add(
            op.name(),
            Inst::Vop2K(Vop2K {
                op,
                vdst: Src::Vgpr(10),
                src0: Src::Vgpr(12),
                src1: Src::Vgpr(14),
                literal: 0x4048_f5c3,
            }),
        );
    }
    for &op in VopcOp::ALL {
        add(
            op.name(),
            Inst::Vopc(Vopc {
                op,
                src0: Src::Vgpr(12),
                src1: Src::Vgpr(14),
                ext: VopExt::None,
            }),
        );
    }
    for op in Vop3Op::all(ARCH) {
        add(op.name(), vop3_inst(op));
    }
    for &op in Vop3pOp::ALL {
        let (_, a, b, c) = op.widths();
        add(
            op.name(),
            Inst::Vop3p(Vop3p {
                op,
                dst: Src::Vgpr(10),
                src: [vreg(12, a), vreg(14, b), vreg(16, c)],
                op_sel: 0,
                op_sel_hi: 7,
                neg_lo: 0,
                neg_hi: 0,
                clamp: false,
            }),
        );
    }
    for &op in MaiOp::ALL {
        let inst = match op {
            MaiOp::AccvgprReadB32 => Mai {
                op,
                dst: Src::Vgpr(10),
                src_a: Src::Agpr(12),
                src_b: None,
                src_c: None,
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
            MaiOp::AccvgprWriteB32 => Mai {
                op,
                dst: Src::Agpr(10),
                src_a: Src::Vgpr(12),
                src_b: None,
                src_c: None,
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
            _ => Mai {
                op,
                dst: Src::Agpr(32),
                src_a: Src::Vgpr(12),
                src_b: Some(Src::Vgpr(14)),
                src_c: Some(Src::Agpr(32)),
                cbsz: 0,
                abid: 0,
                blgp: 0,
            },
        };
        add(op.name(), Inst::Mai(inst));
    }
    for &op in DsOp::ALL {
        let (d, a, b) = op.widths();
        add(
            op.name(),
            Inst::Ds(Ds {
                op,
                vdst: vreg(20, d),
                addr: (op != DsOp::Nop).then_some(Src::Vgpr(12)),
                data0: vreg(14, a),
                data1: vreg(16, b),
                offset0: 0,
                offset1: 0,
                gds: false,
            }),
        );
    }
    for seg in [FlatSeg::Global, FlatSeg::Flat, FlatSeg::Scratch] {
        for &op in FlatOp::ALL {
            let Some(name) = op.name(seg) else { continue };
            let (d, data) = op.widths();
            add(
                name,
                Inst::Flat(Flat {
                    op,
                    seg,
                    vdst: vreg(20, d),
                    addr: Some(Src::Vgpr(12)),
                    data: vreg(14, data),
                    saddr: match seg {
                        FlatSeg::Flat => None,
                        FlatSeg::Global => Some(Src::Sgpr(2)),
                        FlatSeg::Scratch => Some(Src::Sgpr(5)),
                    },
                    offset: 0,
                    sc0: d > 0,
                    sc1: false,
                    nt: false,
                }),
            );
        }
    }
    for &op in MubufOp::ALL {
        add(
            op.name(),
            Inst::Mubuf(Mubuf {
                op,
                vdata: Src::Vgpr(20),
                vaddr: Some(Src::Vgpr(12)),
                srsrc: Src::Sgpr(4),
                soffset: Src::Int(0),
                offset: 0,
                offen: true,
                idxen: false,
                sc0: false,
                sc1: false,
                nt: false,
            }),
        );
    }
    out
}

/// Assert lossless assembler text parses back to the exact typed instruction.
fn roundtrip(label: &str, inst: &Inst) {
    let text = inst.assembly().to_string();
    match parse_line(&text, ARCH) {
        Ok(back) => assert_eq!(&back, inst, "{label}: `{text}`"),
        Err(e) => panic!("{label}: `{text}` did not parse: {e}"),
    }
}

#[test]
fn display_of_every_opcode_round_trips() {
    let mut total = 0;
    let mut checked = 0;
    let mut skipped = Vec::new();
    for (name, inst) in universe() {
        total += 1;
        // Operand combinations an encoder rejects have no instruction to print.
        if inst.words(ARCH).is_err() {
            skipped.push(name);
            continue;
        }
        roundtrip(&name, &inst);
        checked += 1;
    }
    assert!(
        checked * 10 >= total * 9,
        "only {checked}/{total} universe instructions encode; skipped {skipped:?}"
    );
}

#[test]
fn every_table_mnemonic_resolves() {
    // Each opcode's name alone (no operands) must reach the operand parser,
    // never `unknown mnemonic`.
    for (name, _) in universe() {
        let err = match parse_line(&name, ARCH) {
            Ok(_) => continue,
            Err(e) => e,
        };
        assert!(!err.contains("unknown mnemonic"), "{name}: {err}");
    }
}

fn dpp(ctrl: DppCtrl) -> Dpp {
    Dpp {
        ctrl,
        row_mask: 0xf,
        bank_mask: 0xf,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    }
}

fn sdwa() -> Sdwa {
    Sdwa {
        dst_sel: SdwaSel::Dword,
        dst_unused: SdwaUnused::Pad,
        clamp: false,
        omod: 0,
        src0_sel: SdwaSel::Dword,
        src0_sext: false,
        src0_neg: false,
        src0_abs: false,
        src1_sel: SdwaSel::Dword,
        src1_sext: false,
        src1_neg: false,
        src1_abs: false,
        sdst: None,
    }
}

#[test]
fn dpp_and_sdwa_forms_round_trip() {
    let mut insts: Vec<Inst> = Vec::new();
    let ctrls = [
        DppCtrl::QuadPerm([3, 2, 1, 0]),
        DppCtrl::RowShl(1),
        DppCtrl::RowShr(15),
        DppCtrl::RowRor(7),
        DppCtrl::WaveShl1,
        DppCtrl::WaveRol1,
        DppCtrl::WaveShr1,
        DppCtrl::WaveRor1,
        DppCtrl::RowMirror,
        DppCtrl::RowHalfMirror,
        DppCtrl::RowBcast15,
        DppCtrl::RowBcast31,
    ];
    for ctrl in ctrls {
        let mut d = dpp(ctrl);
        insts.push(Inst::Vop1(Vop1 {
            op: Vop1Op::MovB32,
            vdst: Src::Vgpr(1),
            src0: Src::Vgpr(2),
            ext: VopExt::Dpp(d),
        }));
        d.row_mask = 0x5;
        d.bank_mask = 0xa;
        d.bound_ctrl = true;
        d.src0_neg = true;
        d.src1_abs = true;
        insts.push(Inst::Vop2(Vop2 {
            op: Vop2Op::AddF32,
            vdst: Src::Vgpr(1),
            src0: Src::Vgpr(2),
            src1: Src::Vgpr(3),
            ext: VopExt::Dpp(d),
        }));
    }
    for sel in [
        SdwaSel::Byte0,
        SdwaSel::Byte3,
        SdwaSel::Word0,
        SdwaSel::Word1,
        SdwaSel::Dword,
    ] {
        for unused in [SdwaUnused::Pad, SdwaUnused::Sext, SdwaUnused::Preserve] {
            let mut s = sdwa();
            s.dst_sel = sel;
            s.dst_unused = unused;
            s.src0_sel = sel;
            s.src1_sel = SdwaSel::Byte2;
            s.clamp = true;
            s.omod = 2;
            s.src0_neg = true;
            s.src1_abs = true;
            insts.push(Inst::Vop2(Vop2 {
                op: Vop2Op::MulF32,
                vdst: Src::Vgpr(1),
                src0: Src::Vgpr(2),
                src1: Src::Sgpr(3),
                ext: VopExt::Sdwa(s),
            }));
            s.clamp = false;
            s.omod = 0;
            s.src0_neg = false;
            s.src1_abs = false;
            s.src0_sext = true;
            s.src1_sext = true;
            insts.push(Inst::Vop2(Vop2 {
                op: Vop2Op::AddU32,
                vdst: Src::Vgpr(1),
                src0: Src::Vgpr(2),
                src1: Src::Vgpr(3),
                ext: VopExt::Sdwa(s),
            }));
            let mut s = sdwa();
            s.src0_sel = sel;
            insts.push(Inst::Vop1(Vop1 {
                op: Vop1Op::CvtF32Fp8,
                vdst: Src::Vgpr(1),
                src0: Src::Vgpr(2),
                ext: VopExt::Sdwa(s),
            }));
        }
    }
    let mut s = sdwa();
    s.src0_sel = SdwaSel::Word1;
    s.src1_sel = SdwaSel::Byte1;
    insts.push(Inst::Vopc(Vopc {
        op: VopcOp::CmpLtU32,
        src0: Src::Vgpr(2),
        src1: Src::Vgpr(3),
        ext: VopExt::Sdwa(s),
    }));
    s.sdst = Some(Src::Sgpr(4));
    insts.push(Inst::Vopc(Vopc {
        op: VopcOp::CmpLtU32,
        src0: Src::Vgpr(2),
        src1: Src::Vgpr(3),
        ext: VopExt::Sdwa(s),
    }));
    let mut encoded = 0;
    for inst in &insts {
        if inst.words(ARCH).is_ok() {
            roundtrip("dpp/sdwa", inst);
            encoded += 1;
        }
    }
    assert!(
        encoded * 10 >= insts.len() * 8,
        "only {encoded}/{} dpp/sdwa forms encode",
        insts.len()
    );
}

#[test]
fn modifier_sweeps_round_trip() {
    let mut insts: Vec<Inst> = Vec::new();
    // VOP3: every modifier the opcode honours, plus constants.
    for op in Vop3Op::all(ARCH) {
        let Inst::Vop3(mut v) = vop3_inst(op) else {
            unreachable!()
        };
        let m = op.mods();
        v.neg = m.neg;
        v.abs = m.abs;
        v.clamp = m.clamp;
        v.omod = if m.omod { 2 } else { 0 };
        v.op_sel = m.op_sel;
        insts.push(Inst::Vop3(v.clone()));
        v.neg = 0;
        v.abs = 0;
        v.clamp = false;
        v.omod = 0;
        v.op_sel = 0;
        let (_, w0, w1, _) = op.widths();
        if w0 == 1 && !matches!(op, Vop3Op::Only(Vop3OnlyOp::WritelaneB32)) {
            v.src[0] = Some(Src::Int(-5));
            insts.push(Inst::Vop3(v.clone()));
            v.src[0] = Some(Src::Float(peacemaker_cdna::operand::InlineFloat::NegHalf));
            insts.push(Inst::Vop3(v.clone()));
        }
        if w1 == 1
            && !matches!(
                op,
                Vop3Op::Only(Vop3OnlyOp::ReadlaneB32 | Vop3OnlyOp::WritelaneB32)
            )
        {
            v.src[1] = Some(Src::Int(64));
            insts.push(Inst::Vop3(v));
        }
    }
    // VOP3P: every mask.
    for &op in Vop3pOp::ALL {
        let (_, a, b, c) = op.widths();
        for (op_sel, op_sel_hi, neg_lo, neg_hi) in [
            (0u8, 7u8, 0u8, 0u8),
            (3, 7, 3, 0),
            (1, 4, 0, 3),
            (2, 0, 1, 2),
            (7, 0, 7, 7),
        ] {
            for clamp in [false, true] {
                insts.push(Inst::Vop3p(Vop3p {
                    op,
                    dst: Src::Vgpr(10),
                    src: [vreg(12, a), vreg(14, b), vreg(16, c)],
                    op_sel,
                    op_sel_hi,
                    neg_lo,
                    neg_hi,
                    clamp,
                }));
            }
        }
    }
    // MFMA modifiers.
    for &op in MaiOp::ALL {
        if op.shape().is_none() {
            continue;
        }
        for (cbsz, abid, blgp) in [(0u8, 0u8, 0u8), (1, 2, 3), (4, 15, 7)] {
            let s = op.shape().unwrap();
            insts.push(Inst::Mai(Mai {
                op,
                dst: Src::Agpr(32),
                src_a: Src::Vgpr(12),
                src_b: Some(Src::Vgpr(14)),
                src_c: Some(Src::Agpr(32)),
                cbsz,
                abid,
                blgp,
            }));
            insts.push(Inst::Mai(Mai {
                op,
                dst: Src::Vgpr(32),
                src_a: Src::Agpr(12),
                src_b: Some(Src::Agpr(14)),
                src_c: Some(Src::Int(0)),
                cbsz,
                abid,
                blgp,
            }));
            let _ = s;
        }
    }
    // SMEM offsets.
    for &op in SmemOp::ALL {
        if !op.has_address() {
            continue;
        }
        let base = if op.buffer() {
            Src::Sgpr(4)
        } else {
            Src::Sgpr(2)
        };
        for offset in [
            SmemOffset::Imm(0x10),
            SmemOffset::Imm(0xf_ffff),
            SmemOffset::Imm(-0x40),
            SmemOffset::Sgpr(Src::Sgpr(8)),
            SmemOffset::Sgpr(Src::M0),
            SmemOffset::SgprImm(Src::Sgpr(8), 0x20),
        ] {
            for glc in [false, true] {
                insts.push(Inst::Smem(Smem {
                    op,
                    sdata: Some(Src::Sgpr(20)),
                    sbase: Some(base),
                    offset,
                    glc,
                }));
            }
        }
    }
    // MUBUF, FLAT / GLOBAL / SCRATCH cache bits and offsets.
    for &op in MubufOp::ALL {
        for (offen, idxen, vaddr) in [
            (true, false, Some(Src::Vgpr(12))),
            (false, true, Some(Src::Vgpr(12))),
            (true, true, Some(Src::Vgpr(12))),
            (false, false, None),
        ] {
            for (sc0, sc1, nt) in [
                (false, false, false),
                (true, false, true),
                (true, true, true),
            ] {
                insts.push(Inst::Mubuf(Mubuf {
                    op,
                    vdata: Src::Vgpr(20),
                    vaddr,
                    srsrc: Src::Sgpr(8),
                    soffset: Src::Sgpr(5),
                    offset: 4095,
                    offen,
                    idxen,
                    sc0,
                    sc1,
                    nt,
                }));
            }
        }
        insts.push(Inst::Mubuf(Mubuf {
            op,
            vdata: Src::Agpr(20),
            vaddr: None,
            srsrc: Src::Ttmp(4),
            soffset: Src::Int(-3),
            offset: 7,
            offen: false,
            idxen: false,
            sc0: false,
            sc1: false,
            nt: false,
        }));
    }
    for seg in [FlatSeg::Global, FlatSeg::Flat, FlatSeg::Scratch] {
        for &op in FlatOp::ALL {
            if op.name(seg).is_none() {
                continue;
            }
            let (d, data) = op.widths();
            for (sc0, sc1, nt) in [
                (false, false, false),
                (true, false, true),
                (true, true, true),
            ] {
                for (addr, saddr, offset) in [
                    (Some(Src::Vgpr(12)), None, -8i16),
                    (Some(Src::Vgpr(12)), Some(Src::Sgpr(2)), 4095),
                    (None, Some(Src::Sgpr(6)), 16),
                    (None, None, 0),
                ] {
                    let returns = d > 0 && data > 0;
                    insts.push(Inst::Flat(Flat {
                        op,
                        seg,
                        vdst: vreg(20, d).filter(|_| data == 0 || sc0),
                        addr,
                        data: vreg(14, data),
                        saddr,
                        offset,
                        sc0: if returns { sc0 } else { sc0 && !returns },
                        sc1,
                        nt,
                    }));
                }
            }
        }
    }
    // DS offsets, including every swizzle shape.
    for &op in DsOp::ALL {
        let (d, a, b) = op.widths();
        let offsets: Vec<(u8, u8)> = if op == DsOp::SwizzleB32 {
            [
                0x8000u16 | 0xe4,
                0x0006 | 0x0900,
                0x401f,
                0x1c1f,
                0x0078,
                0xc4a0,
                0xe003,
                0x0104,
            ]
            .iter()
            .map(|&o| ((o & 0xff) as u8, (o >> 8) as u8))
            .collect()
        } else if op.two_addr() {
            vec![(1, 0), (0, 2), (255, 255)]
        } else {
            vec![(0, 0), (0x10, 0), (0xff, 0xff)]
        };
        for (offset0, offset1) in offsets {
            insts.push(Inst::Ds(Ds {
                op,
                vdst: vreg(20, d),
                addr: (op != DsOp::Nop).then_some(Src::Vgpr(12)),
                data0: vreg(14, a),
                data1: vreg(16, b),
                offset0,
                offset1,
                gds: false,
            }));
        }
    }
    // SALU immediates.
    for imm in [0u16, 1, 5, 64, 65, 0x7fff, 0x8000, 0xffff] {
        for &op in SopkOp::ALL {
            if matches!(op, SopkOp::GetregB32 | SopkOp::SetregB32) {
                continue;
            }
            insts.push(Inst::Sopk(Sopk {
                op,
                sdst: Src::Sgpr(7),
                simm16: imm,
            }));
        }
        for &op in SoppOp::ALL {
            insts.push(Inst::Sopp(Sopp { op, simm16: imm }));
        }
    }
    for hw in 1u16..=24 {
        for (off, size) in [(0u16, 32u16), (4, 2), (31, 1), (0, 4)] {
            let simm16 = hw | off << 6 | (size - 1) << 11;
            insts.push(Inst::Sopk(Sopk {
                op: SopkOp::GetregB32,
                sdst: Src::Sgpr(3),
                simm16,
            }));
            insts.push(Inst::Sopk(Sopk {
                op: SopkOp::SetregB32,
                sdst: Src::Sgpr(3),
                simm16,
            }));
        }
    }
    for vm in [0u8, 1, 15, 16, 63] {
        for exp in [0u8, 3, 7] {
            for lgkm in [0u8, 7, 15] {
                insts.push(Inst::Sopp(Sopp {
                    op: SoppOp::Waitcnt,
                    simm16: waitcnt_imm(ARCH, vm, exp, lgkm).unwrap(),
                }));
            }
        }
    }
    for simm16 in [
        0x1u16, 0x2, 0x12, 0x22, 0x32, 0x102, 0x3, 0x13, 0x23, 0x33, 0xf, 0x1f, 0x2f, 0x4f, 0x5,
        0x6, 0x7, 0x8, 0x9, 0xa, 0x3ff, 0x4321,
    ] {
        insts.push(Inst::Sopp(Sopp {
            op: SoppOp::Sendmsg,
            simm16,
        }));
    }
    let mut encoded = 0;
    for inst in &insts {
        if inst.words(ARCH).is_ok() {
            roundtrip("sweep", inst);
            encoded += 1;
        }
    }
    assert!(
        encoded * 2 >= insts.len(),
        "only {encoded}/{} sweep instructions encode",
        insts.len()
    );
}

// ---------------------------------------------------------------------------
// Decode-driven round trip: words -> Inst -> text -> Inst.

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

#[test]
fn decoded_instructions_print_text_that_parses_back() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // (format label, base first dword, opcode shift, opcode values, mask of random bits)
    type Gen = (&'static str, u32, u32, Vec<u32>, u32);
    let ops = |it: &mut dyn Iterator<Item = Option<u32>>| -> Vec<u32> { it.flatten().collect() };
    let gens: Vec<Gen> = vec![
        (
            "SOP2",
            0x8000_0000,
            23,
            ops(&mut Sop2Op::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x007f_ffff,
        ),
        (
            "SOPK",
            0xb000_0000,
            23,
            ops(&mut SopkOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x007f_ffff,
        ),
        (
            "SOP1",
            0xbe80_0000,
            8,
            ops(&mut Sop1Op::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x007f_00ff,
        ),
        (
            "SOPC",
            0xbf00_0000,
            16,
            ops(&mut SopcOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0000_ffff,
        ),
        (
            "SOPP",
            0xbf80_0000,
            16,
            ops(&mut SoppOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0000_ffff,
        ),
        (
            "SMEM",
            0xc000_0000,
            18,
            ops(&mut SmemOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0003_ffff,
        ),
        (
            "VOP1",
            0x7e00_0000,
            9,
            ops(&mut Vop1Op::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x01fe_01ff,
        ),
        (
            "VOP2",
            0x0000_0000,
            25,
            ops(&mut Vop2Op::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x01ff_ffff,
        ),
        (
            "VOPC",
            0x7c00_0000,
            17,
            ops(&mut VopcOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0001_ffff,
        ),
        (
            "VOP3",
            0xd000_0000,
            16,
            Vop3Op::all(ARCH)
                .iter()
                .filter_map(|o| o.opcode(ARCH).map(u32::from))
                .collect(),
            0x0000_ffff,
        ),
        (
            "VOP3P",
            0xd380_0000,
            16,
            ops(&mut Vop3pOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0000_ffff,
        ),
        (
            "MAI",
            0xd380_0000,
            16,
            ops(&mut MaiOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0000_ffff,
        ),
        (
            "DS",
            0xd800_0000,
            17,
            ops(&mut DsOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0001_ffff,
        ),
        (
            "FLAT",
            0xdc00_0000,
            18,
            ops(&mut FlatOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0003_ffff,
        ),
        (
            "MUBUF",
            0xe000_0000,
            18,
            ops(&mut MubufOp::ALL.iter().map(|o| o.opcode(ARCH).map(u32::from))),
            0x0003_ffff,
        ),
    ];
    for (label, base, shift, opcodes, mask) in gens {
        let mut ok = 0usize;
        for &opcode in &opcodes {
            for t in 0..600usize {
                // Dense, half-dense and sparse random fields: strict decoders
                // reject reserved / unused bits, so sparse words decode more often.
                let mut rnd = || match t % 3 {
                    0 => rng.next(),
                    1 => rng.next() & rng.next(),
                    _ => rng.next() & rng.next() & rng.next(),
                };
                let w0 = base | opcode << shift | (rnd() & mask);
                let words = [w0, rnd(), rnd()];
                let Ok((inst, _)) = Inst::decode(ARCH, &words) else {
                    continue;
                };
                // This is the typed assembler's roundtrip contract. Generated
                // forms preserve bytes but deliberately have no typed semantics.
                if let Inst::Xml(x) = &inst {
                    assert!(x.unsupported().contains(x.form().mnemonic));
                    continue;
                }
                let text = inst.assembly().to_string();
                let back = parse_line(&text, ARCH).unwrap_or_else(|e| {
                    panic!("{label}: `{text}` ({words:08x?}) did not parse: {e}")
                });
                assert_eq!(back, inst, "{label}: `{text}` ({words:08x?})");
                assert_eq!(
                    back.words(ARCH).unwrap(),
                    inst.words(ARCH).unwrap(),
                    "{label}: `{text}`"
                );
                ok += 1;
            }
        }
        assert!(
            ok >= 20,
            "{label}: only {ok} random words decoded; the sweep proves nothing"
        );
    }
}

// ---------------------------------------------------------------------------
// Negative cases: every failure is named.

#[test]
fn unknown_and_malformed_lines() {
    let cases: &[(&str, &str)] = &[
        ("", "empty instruction"),
        ("   // only a comment", "empty instruction"),
        ("loop:", "not an instruction"),
        (".amdgcn_target \"x\"", "not an instruction"),
        ("v_frobnicate_f32 v0, v1", "unknown mnemonic"),
        ("s_mov_b32_e64 s0, 1", "unknown mnemonic"),
        (
            "v_mfma_f32_4x4x4_16b_f16 a[0:3], v[0:1], v[2:3], a[0:3]",
            "unknown mnemonic",
        ),
        ("s_mov_b32 s0", "operand count"),
        ("s_mov_b32 s0, s1, s2", "operand count"),
        ("s_mov_b32 s0,, s2", "operand count"),
        ("v_add_f32 v0, v1", "operand count"),
        ("s_barrier 1", "operand count"),
        ("s_cbranch_execz some_label", "label operand"),
        ("s_branch .Lend", "label operand"),
        ("s_branch 70000", "bad operand"),
        ("s_mov_b32 s0, null", "unsupported operand"),
        ("s_mov_b32 s200, 1", "bad operand"),
        ("s_mov_b32 s[4:2], 1", "bad operand"),
        ("v_mov_b32 v256, 1", "bad operand"),
        ("s_mov_b64 s[0:1], s2", "bad operand"),
        ("s_mov_b32 s[0:1], s2", "bad operand"),
        ("s_mov_b32 s0, 0x1ffffffff", "bad operand"),
        ("s_mov_b32 s0, banana", "bad operand"),
        ("s_mov_b32 s0, neg(neg(1))", "bad operand"),
        ("s_mov_b32 -s0, 1", "bad operand"),
        ("s_mov_b64 s[1:2], 1", "encode"),
        (
            "s_load_dword s0, s[2:3], 0x10 offset:0x10",
            "unsupported modifier",
        ),
        ("s_load_dword s0, s[2:3], s4 slc", "unsupported modifier"),
        ("s_waitcnt vmcnt(64)", "vmcnt"),
        ("s_waitcnt lgkmcnt(16)", "lgkmcnt"),
        ("s_waitcnt vmcnt(0) vmcnt(1)", "twice"),
        ("s_waitcnt foo(1)", "unknown wait counter"),
        ("s_getreg_b32 s0, hwreg(HW_REG_BOGUS)", "bad operand"),
        ("s_getreg_b32 s0, hwreg(HW_REG_MODE, 32, 1)", "bad operand"),
        ("s_sendmsg sendmsg(MSG_BOGUS)", "bad operand"),
        ("s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 4)", "bad operand"),
        ("v_add_f32 v0, v1, v2 clamp clamp", "duplicate modifier"),
        ("v_add_f32_e64 v0, v1, v2 mul:3", "unsupported modifier"),
        (
            "v_add_f32_e64 v0, v1, v2 mul:2 div:2",
            "unsupported modifier",
        ),
        ("v_add_f32_e32 v0, v1, v2 clamp", "unsupported modifier"),
        ("v_add_f32_e64 v0, v1, v2 row_shl:1", "unsupported modifier"),
        ("v_add_f32_e64 v0, sext(v1), v2", "bad operand"),
        ("v_add_f32_dpp v0, v1, v2", "unsupported modifier"),
        (
            "v_add_f32_dpp v0, v1, v2 row_shl:16",
            "unsupported modifier",
        ),
        (
            "v_add_f32_dpp v0, v1, v2 row_shl:1 row_mirror",
            "unsupported modifier",
        ),
        (
            "v_add_f32_dpp v0, v1, v2 row_newbcast:1",
            "unsupported modifier",
        ),
        (
            "v_add_f32_dpp v0, v1, v2 quad_perm:[0,1,2] row_mask:0xf",
            "unsupported modifier",
        ),
        (
            "v_add_f32_dpp v0, v1, v2 row_shl:1 dst_sel:BYTE_0",
            "unsupported modifier",
        ),
        (
            "v_add_f32_sdwa v0, v1, v2 src0_sel:BYTE_9",
            "unsupported modifier",
        ),
        (
            "v_add_f32_sdwa v0, v1, v2 dst_unused:BOGUS",
            "unsupported modifier",
        ),
        (
            "v_cvt_f32_fp8_sdwa v0, v1 dst_sel:BYTE_1",
            "unsupported modifier",
        ),
        ("v_mov_b32_e64 v0, 0x12345678", "gfx9 VOP3 has no literal"),
        ("v_add_f32_e32 v0, v1, 5", "encode"),
        ("v_add_f32_e32 v0, s1, s2", "encode"),
        ("v_add_f32 v0, s1, s2", "constant-bus"),
        ("v_cndmask_b32_e32 v0, v1, v2, s[0:1]", "vcc"),
        ("v_add_co_u32_e32 v0, s[0:1], v1, v2", "vcc"),
        ("v_cmp_lt_f32_e32 s[0:1], v0, v1", "vcc"),
        ("v_pk_add_f32 v[0:1], -v[2:3], v[4:5]", "neg_lo"),
        (
            "v_pk_add_f32 v[0:1], v[2:3], v[4:5] op_sel:[1,0,0]",
            "unsupported modifier",
        ),
        ("v_fma_mix_f32 v0, |v1|, v2, v3", "bad operand"),
        (
            "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], v[0:3]",
            "encode",
        ),
        (
            "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3] cbsz:8",
            "unsupported modifier",
        ),
        (
            "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3] neg:[1,0,0]",
            "unsupported modifier",
        ),
        (
            "v_mfma_f32_16x16x16_bf16 a[0:3], v0, v[2:3], a[0:3]",
            "bad operand",
        ),
        (
            "v_mfma_f32_16x16x16_bf16 a[0:3], 1, v[2:3], a[0:3]",
            "bad operand",
        ),
        ("ds_read_b32 v0, v1 gds", "GDS"),
        ("ds_read_b32 v0, v1 offset:65536", "unsupported modifier"),
        ("ds_read_b32 v0, v1 offset0:1", "unsupported modifier"),
        ("ds_read2_b32 v[0:1], v1 offset:4", "unsupported modifier"),
        ("ds_read_b32 v0, v1, v2", "operand count"),
        ("ds_read_b64 v[1:2], v3", "encode"),
        (
            "ds_swizzle_b32 v0, v1 offset:swizzle(BOGUS,1)",
            "unsupported modifier",
        ),
        (
            "ds_swizzle_b32 v0, v1 offset:swizzle(QUAD_PERM,0,1,2)",
            "unsupported modifier",
        ),
        (
            "ds_swizzle_b32 v0, v1 offset:swizzle(BITMASK_PERM,\"01x00\")",
            "unsupported modifier",
        ),
        (
            "global_load_dword v0, v[2:3], off glc",
            "unsupported modifier",
        ),
        (
            "global_load_dword v0, v[2:3], off slc",
            "unsupported modifier",
        ),
        ("global_load_dword v0, v1, off", "bad operand"),
        ("global_load_dword v0, v[2:3], s[2:3]", "bad operand"),
        ("global_load_dword v0, v[2:3], s5", "bad operand"),
        ("global_load_dword v0, v[2:3], off offset:4096", "encode"),
        ("global_load_dword v0, v[2:3]", "operand count"),
        ("global_atomic_add v[2:3], v1, off sc0", "operand count"),
        ("scratch_load_dwordx2 v[1:2], off, off", "encode"),
        (
            "buffer_load_dword v0, v1, s[4:7], s8 lds",
            "unsupported modifier",
        ),
        (
            "buffer_load_dword v0, v1, s[4:7], s8 offset:4096",
            "unsupported modifier",
        ),
        ("buffer_load_dword v0, v1, s[4:7], s8", "encode"),
        (
            "buffer_load_dword v0, v1, s[4:7], s8 idxen offen",
            "bad operand",
        ),
        ("buffer_load_dword v0, v1, s[3:6], s8 offen", "encode"),
        (
            "buffer_load_dword v[0:1], v1, s[4:7], s8 offen",
            "bad operand",
        ),
        ("v_fmaak_f32 v0, v1, v2, v3", "bad operand"),
        ("v_fmaak_f32 v0, s1, v2, 1.0", "encode"),
        ("v_fmaak_f32_e64 v0, v1, v2, 1.0", "unknown mnemonic"),
        ("s_nop", "operand count"),
        ("s_nop 70000", "bad operand"),
        ("s_movk_i32 s0, 0x10000", "bad operand"),
        ("s_cmp_eq_u32 s0, v1", "encode"),
        ("s_load_dword s0, s[3:4], 0", "encode"),
    ];
    let mut wrong = Vec::new();
    for &(text, needle) in cases {
        match parse_line(text, ARCH) {
            Ok(inst) => wrong.push(format!("`{text}` parsed as {inst:?}")),
            Err(e) if e.contains(needle) => {}
            Err(e) => wrong.push(format!("`{text}`: error `{e}` lacks `{needle}`")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn ambiguity_lists_every_attempt() {
    // v_add_f32 is VOP2 and VOP3: a line neither accepts names both.
    let e = parse_err("v_add_f32 v0, v1, v2, v3");
    assert!(e.contains("VOP2 e32") && e.contains("VOP3 e64"), "{e}");
}
