// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA3 hazard pass (`peacemaker_isa::cdna::hazard`): one test per rule
//! (detected by `check`, fixed by `resolve` with the minimal wait states, no
//! NOP when enough independent instructions intervene), branch/loop
//! soundness, and LLVM evidence: hipcc-compiled gfx942 MFMA kernels decode
//! to a list `check` finds hazard-free.
#[path = "support/rocm.rs"]
mod rocm;

use peacemaker_cdna::enc::{
    ds::{Ds, DsOp},
    flat::{Flat, FlatOp, FlatSeg},
    mubuf::{Mubuf, MubufOp},
    sop::{Sop1, Sop1Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp},
    vop::{Dpp, DppCtrl, Vop1, Vop1Op, Vop2, Vop2Op, VopExt, Vopc, VopcOp},
    vop3::{Vop3, Vop3OnlyOp, Vop3Op},
    vop3p::{Mai, MaiOp, Vop3p, Vop3pOp},
};
use peacemaker_cdna::{decode_all, oracle, Arch, Inst, Src};
use peacemaker_isa::cdna::hazard::{check, resolve};
use peacemaker_isa::cdna::Item;
use std::collections::HashMap;

const A: Arch = Arch::Gfx942;
use Src::{Agpr as a, Sgpr as s, Vgpr as v};

// ---- instruction constructors ----

fn filler() -> Inst {
    Inst::Sop1(Sop1 {
        op: Sop1Op::MovB32,
        sdst: Some(s(100)),
        ssrc0: Some(Src::Int(0)),
    })
}
fn s_mov(dst: Src, src: Src) -> Inst {
    Inst::Sop1(Sop1 {
        op: Sop1Op::MovB32,
        sdst: Some(dst),
        ssrc0: Some(src),
    })
}
fn s_cmp() -> Inst {
    Inst::Sopc(Sopc {
        op: SopcOp::CmpEqU32,
        ssrc0: s(0),
        ssrc1: s(1),
    })
}
fn endpgm() -> Inst {
    Inst::Sopp(Sopp {
        op: SoppOp::Endpgm,
        simm16: 0,
    })
}
fn hwreg(id: u16, offset: u16, size: u16) -> u16 {
    id | offset << 6 | (size - 1) << 11
}
fn setreg(simm16: u16) -> Inst {
    Inst::Sopk(Sopk {
        op: SopkOp::SetregB32,
        sdst: s(0),
        simm16,
    })
}
fn getreg(simm16: u16) -> Inst {
    Inst::Sopk(Sopk {
        op: SopkOp::GetregB32,
        sdst: s(1),
        simm16,
    })
}
fn vop1(op: Vop1Op, vdst: Src, src0: Src) -> Inst {
    Inst::Vop1(Vop1 {
        op,
        vdst,
        src0,
        ext: VopExt::None,
    })
}
fn v_mov(vdst: Src, src0: Src) -> Inst {
    vop1(Vop1Op::MovB32, vdst, src0)
}
fn dpp() -> Dpp {
    Dpp {
        ctrl: DppCtrl::QuadPerm([1, 0, 3, 2]),
        row_mask: 0xf,
        bank_mask: 0xf,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    }
}
fn v_mov_dpp(vdst: Src, src0: Src) -> Inst {
    Inst::Vop1(Vop1 {
        op: Vop1Op::MovB32,
        vdst,
        src0,
        ext: VopExt::Dpp(dpp()),
    })
}
fn vop2(op: Vop2Op, vdst: Src, src0: Src, src1: Src) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst,
        src0,
        src1,
        ext: VopExt::None,
    })
}
fn vopc(op: VopcOp) -> Inst {
    Inst::Vopc(Vopc {
        op,
        src0: v(0),
        src1: v(1),
        ext: VopExt::None,
    })
}
fn vop3(op: Vop3Op, dst: Src, src: [Option<Src>; 3], op_sel: u8) -> Inst {
    Inst::Vop3(Vop3 {
        op,
        dst,
        sdst: None,
        src,
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel,
    })
}
fn only(op: Vop3OnlyOp, dst: Src, src: [Option<Src>; 3], op_sel: u8) -> Inst {
    vop3(Vop3Op::Only(op), dst, src, op_sel)
}
fn readlane(dst: Src, vsrc: Src, lane: Src) -> Inst {
    only(
        Vop3OnlyOp::ReadlaneB32,
        dst,
        [Some(vsrc), Some(lane), None],
        0,
    )
}
fn readfirstlane(dst: Src, vsrc: Src) -> Inst {
    vop1(Vop1Op::ReadfirstlaneB32, dst, vsrc)
}
fn cvt_pk_fp8(dst: Src, x: Src, y: Src, op_sel: u8) -> Inst {
    only(
        Vop3OnlyOp::CvtPkFp8F32,
        dst,
        [Some(x), Some(y), None],
        op_sel,
    )
}
fn dot2(dst: Src, x: Src, y: Src, c: Src) -> Inst {
    Inst::Vop3p(Vop3p {
        op: Vop3pOp::Dot2F32F16,
        dst,
        src: [Some(x), Some(y), Some(c)],
        op_sel: 0,
        op_sel_hi: 7,
        neg_lo: 0,
        neg_hi: 0,
        clamp: false,
    })
}
fn mfma(op: MaiOp, dst: Src, x: Src, y: Src, c: Src) -> Inst {
    Inst::Mai(Mai {
        op,
        dst,
        src_a: x,
        src_b: Some(y),
        src_c: Some(c),
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
/// XDL, 4 passes (Table 28: 16 cycles).
fn bf16_16(dst: Src, x: Src, y: Src, c: Src) -> Inst {
    mfma(MaiOp::MfmaF32_16x16x16Bf16, dst, x, y, c)
}
/// XDL, 8 passes (32 cycles).
fn bf16_32(dst: Src, x: Src, y: Src, c: Src) -> Inst {
    mfma(MaiOp::MfmaF32_32x32x8Bf16, dst, x, y, c)
}
fn acc_read(dst: Src, src: Src) -> Inst {
    Inst::Mai(Mai {
        op: MaiOp::AccvgprReadB32,
        dst,
        src_a: src,
        src_b: None,
        src_c: None,
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
fn acc_write(dst: Src, src: Src) -> Inst {
    Inst::Mai(Mai {
        op: MaiOp::AccvgprWriteB32,
        dst,
        src_a: src,
        src_b: None,
        src_c: None,
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
fn global(op: FlatOp, vdst: Option<Src>, addr: Src, data: Option<Src>, saddr: Option<Src>) -> Inst {
    Inst::Flat(Flat {
        op,
        seg: FlatSeg::Global,
        vdst,
        addr: Some(addr),
        data,
        saddr,
        offset: 0,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
fn store_x4(data: Src) -> Inst {
    global(FlatOp::StoreDwordx4, None, v(0), Some(data), None)
}
fn buffer_store_x4(data: Src, soffset: Src) -> Inst {
    Inst::Mubuf(Mubuf {
        op: MubufOp::StoreDwordx4,
        vdata: data,
        vaddr: None,
        srsrc: s(0),
        soffset,
        offset: 0,
        offen: false,
        idxen: false,
        sc0: false,
        sc1: false,
        nt: false,
    })
}

// ---- harness ----

fn items(insts: &[Inst]) -> Vec<Item> {
    insts.iter().cloned().map(Item::Inst).collect()
}

/// The instructions of resolved items, branches lowered to dword offsets.
fn lower(items: &[Item]) -> Vec<Inst> {
    let size = |i: &Item| match i {
        Item::Inst(inst) => inst.words(A).expect("encodable test instruction").len(),
        Item::Branch { .. } => 1,
        Item::Label(_) => 0,
    };
    let mut at = 0usize;
    let mut labels = HashMap::new();
    for i in items {
        if let Item::Label(l) = i {
            labels.insert(l.clone(), at);
        }
        at += size(i);
    }
    let mut out = Vec::new();
    let mut at = 0usize;
    for i in items {
        match i {
            Item::Inst(inst) => out.push(inst.clone()),
            Item::Branch { op, target } => {
                let off = labels[target] as i64 - (at as i64 + 1);
                out.push(Inst::Sopp(Sopp {
                    op: *op,
                    simm16: off as i16 as u16,
                }));
            }
            Item::Label(_) => {}
        }
        at += size(i);
    }
    out
}

fn nop_waits(items: &[Item]) -> u32 {
    items
        .iter()
        .map(|i| match i {
            Item::Inst(Inst::Sopp(p)) if p.op == SoppOp::Nop => u32::from(p.simm16 & 0xf) + 1,
            _ => 0,
        })
        .sum()
}

/// One rule: `prod` then `cons` violates `rule` (a substring of the cited
/// row) and needs exactly `waits` wait states; `k` independent instructions
/// in between leave `waits - k`.
fn case(rule: &str, prod: Inst, cons: Inst, waits: u32) {
    let direct = [prod.clone(), cons.clone()];
    let found = check(A, &direct);
    assert!(
        found
            .iter()
            .any(|f| f.starts_with("1: ") && f.contains(rule)),
        "{rule}: check missed it: {found:#?}"
    );
    for k in 0..=waits + 1 {
        let mut seq = vec![prod.clone()];
        seq.extend(std::iter::repeat_with(filler).take(k as usize));
        seq.push(cons.clone());
        let want = waits.saturating_sub(k);
        let (out, proofs) =
            resolve(A, items(&seq)).unwrap_or_else(|e| panic!("{rule}: resolve: {e}"));
        assert_eq!(
            nop_waits(&out),
            want,
            "{rule}: {k} independent instructions: {out:#?}"
        );
        assert_eq!(
            proofs.len() as u32,
            want.div_ceil(8),
            "{rule}: proofs {proofs:#?}"
        );
        for p in &proofs {
            assert!(p.rule.contains(rule), "{rule}: proof cites {}", p.rule);
            assert!(
                matches!(&out[p.pc_index], Item::Inst(Inst::Sopp(n)) if n.op == SoppOp::Nop),
                "{rule}: pc_index {} is not the s_nop",
                p.pc_index
            );
            assert!(p.insn.starts_with("s_nop "));
        }
        assert!(
            check(A, &lower(&out)).is_empty(),
            "{rule}: resolved list still violates: {:#?}",
            check(A, &lower(&out))
        );
        let raw = check(A, &seq);
        assert_eq!(
            raw.iter().any(|f| f.contains(rule)),
            k < waits,
            "{rule}: {k} independent instructions: {raw:#?}"
        );
    }
}

fn clean(name: &str, seq: &[Inst]) {
    let found = check(A, seq);
    assert!(found.is_empty(), "{name}: {found:#?}");
    let (out, proofs) = resolve(A, items(seq)).unwrap();
    assert_eq!(nop_waits(&out), 0, "{name}: {proofs:#?}");
}

// ---- CDNA3 ISA §4.5 Table 11 ----

#[test]
fn setreg_then_getreg_same_reg() {
    case(
        "S_SETREG -> S_GETREG same reg",
        setreg(hwreg(1, 0, 4)),
        getreg(hwreg(1, 0, 32)),
        2,
    );
    clean(
        "different hwreg",
        &[setreg(hwreg(1, 0, 4)), getreg(hwreg(3, 0, 32))],
    );
}

#[test]
fn setreg_then_setreg_same_reg() {
    case(
        "S_SETREG -> S_SETREG same reg",
        setreg(hwreg(1, 0, 4)),
        setreg(hwreg(1, 4, 4)),
        2,
    )
}

#[test]
fn setreg_vskip_then_vector_op() {
    case("MODE.vskip", setreg(hwreg(1, 28, 1)), v_mov(v(0), v(1)), 2);
    clean(
        "MODE write not covering bit 28",
        &[setreg(hwreg(1, 0, 4)), v_mov(v(0), v(1))],
    );
}

#[test]
fn valu_vcc_then_vccz_source() {
    case(
        "uses EXECZ or VCCZ",
        vopc(VopcOp::CmpEqU32),
        v_mov(v(2), Src::Vccz),
        5,
    )
}

#[test]
fn valu_sgpr_then_lane_select() {
    case(
        "using that SGPR/VCC as the lane select",
        readfirstlane(s(4), v(1)),
        readlane(s(5), v(2), s(4)),
        4,
    );
    case(
        "using that SGPR/VCC as the lane select",
        readfirstlane(s(4), v(1)),
        only(
            Vop3OnlyOp::WritelaneB32,
            v(3),
            [Some(Src::Int(7)), Some(s(4)), None],
            0,
        ),
        4,
    );
}

#[test]
fn valu_vcc_then_div_fmas() {
    case(
        "V_DIV_FMAS",
        vopc(VopcOp::CmpEqU32),
        only(
            Vop3OnlyOp::DivFmasF32,
            v(0),
            [Some(v(1)), Some(v(2)), Some(v(3))],
            0,
        ),
        4,
    );
}

#[test]
fn wide_store_data_overwritten() {
    case(
        "VALU writes VGPRs holding writedata (2)",
        store_x4(v(4)),
        v_mov(v(5), v(9)),
        2,
    );
    case(
        "write VGPRs holding writedata (1)",
        store_x4(v(4)),
        global(FlatOp::LoadDword, Some(v(6)), v(10), None, None),
        1,
    );
    // BUFFER_STORE with an inline-constant SOFFSET: the ISA rows apply.
    case(
        "VALU writes VGPRs holding writedata (2)",
        buffer_store_x4(v(4), Src::Int(0)),
        v_mov(v(7), v(9)),
        2,
    );
    // With an SGPR SOFFSET: ISA 0, LLVM gfx940 1.
    case(
        "SGPR SOFFSET",
        buffer_store_x4(v(4), s(8)),
        v_mov(v(4), v(9)),
        1,
    );
    clean(
        "64-bit store",
        &[
            global(FlatOp::StoreDwordx2, None, v(0), Some(v(4)), None),
            v_mov(v(5), v(9)),
        ],
    );
}

#[test]
fn valu_sgpr_then_vmem_reads_it() {
    case(
        "VMEM reads that SGPR",
        readfirstlane(s(8), v(0)),
        global(FlatOp::LoadDword, Some(v(1)), v(2), None, Some(s(8))),
        5,
    );
    case(
        "VMEM reads that SGPR",
        readfirstlane(s(9), v(0)),
        buffer_store_x4(v(4), s(9)),
        5,
    );
}

#[test]
fn salu_m0_then_sendmsg() {
    case(
        "SALU writes M0 -> GDS, S_SENDMSG",
        s_mov(Src::M0, s(0)),
        Inst::Sopp(Sopp {
            op: SoppOp::Sendmsg,
            simm16: 1,
        }),
        1,
    )
}

#[test]
fn valu_vgpr_then_dpp() {
    case(
        "VALU DPP reads that VGPR",
        vop2(Vop2Op::AddU32, v(1), v(2), v(3)),
        v_mov_dpp(v(0), v(1)),
        2,
    );
    clean(
        "DPP of another VGPR",
        &[
            vop2(Vop2Op::AddU32, v(1), v(2), v(3)),
            v_mov_dpp(v(0), v(4)),
        ],
    );
}

#[test]
fn valu_exec_then_dpp() {
    case(
        "VALU writes EXEC -> VALU DPP op",
        vopc(VopcOp::CmpxEqU32),
        v_mov_dpp(v(4), v(5)),
        5,
    )
}

#[test]
fn valu_sgpr_then_valu_constant() {
    case(
        "VALU reads SGPR as constant",
        readfirstlane(s(4), v(0)),
        vop2(Vop2Op::AddU32, v(1), s(4), v(2)),
        2,
    );
    // Carry-in / VCC select: ISA 0, LLVM gfx940 2.
    case(
        "VALU reads SGPR as constant",
        vopc(VopcOp::CmpEqU32),
        vop2(Vop2Op::CndmaskB32, v(3), v(1), v(2)),
        2,
    );
    case(
        "VALU reads SGPR as constant",
        vop3(
            Vop3Op::C(VopcOp::CmpEqU32),
            s(4),
            [Some(v(0)), Some(v(1)), None],
            0,
        ),
        vop3(
            Vop3Op::V2(Vop2Op::CndmaskB32),
            v(3),
            [Some(v(1)), Some(v(2)), Some(s(4))],
            0,
        ),
        2,
    );
}

#[test]
fn cmpx_then_exec_constant() {
    case(
        "VALU reads EXEC as constant",
        vopc(VopcOp::CmpxEqU32),
        v_mov(v(2), Src::ExecLo),
        2,
    )
}

#[test]
fn cmpx_then_readlane() {
    case(
        "v_readlane, v_readfirstlane, v_writelane",
        vopc(VopcOp::CmpxEqU32),
        readfirstlane(s(0), v(3)),
        4,
    );
    case(
        "v_readlane, v_readfirstlane, v_writelane",
        vopc(VopcOp::CmpxEqU32),
        readlane(s(0), v(3), Src::Int(1)),
        4,
    );
}

#[test]
fn valu_vgpr_then_readlane_src0() {
    case(
        "v_readlane vsrc0 reads VGPRn",
        vop2(Vop2Op::AddU32, v(1), v(2), v(3)),
        readlane(s(0), v(1), Src::Int(0)),
        1,
    );
    case(
        "v_readlane vsrc0 reads VGPRn",
        vop2(Vop2Op::AddU32, v(1), v(2), v(3)),
        readfirstlane(s(0), v(1)),
        1,
    );
}

#[test]
fn opsel_dst_then_consumer() {
    case(
        "OPSEL or SDWA",
        cvt_pk_fp8(v(1), v(2), v(3), 0b1000),
        v_mov(v(0), v(1)),
        1,
    );
    case(
        "OPSEL or SDWA",
        only(
            Vop3OnlyOp::CvtSrFp8F32,
            v(1),
            [Some(v(2)), Some(v(3)), None],
            0b0100,
        ),
        vop2(Vop2Op::AddU32, v(0), v(1), v(4)),
        1,
    );
    clean(
        "low-half write consumed",
        &[cvt_pk_fp8(v(1), v(2), v(3), 0), v_mov(v(0), v(1))],
    );
}

#[test]
fn trans_then_non_trans() {
    case(
        "VALU Trans op",
        vop1(Vop1Op::ExpF32, v(1), v(2)),
        vop2(Vop2Op::AddF32, v(0), v(1), v(3)),
        1,
    );
    clean(
        "trans feeding trans",
        &[
            vop1(Vop1Op::ExpF32, v(1), v(2)),
            vop1(Vop1Op::LogF32, v(0), v(1)),
        ],
    );
}

// ---- CDNA3 ISA §7.2 p.51 ----

#[test]
fn fp8_converts_writing_one_vgpr() {
    case(
        "FP8/BF8 conversions writing halves/bytes",
        cvt_pk_fp8(v(1), v(2), v(3), 0),
        cvt_pk_fp8(v(1), v(4), v(5), 0b1000),
        1,
    );
    clean(
        "different destinations",
        &[
            cvt_pk_fp8(v(1), v(2), v(3), 0),
            cvt_pk_fp8(v(6), v(4), v(5), 0b1000),
        ],
    );
}

// ---- CDNA3 ISA §7.5 Table 37 ----

#[test]
fn valu_write_then_mfma_read() {
    case(
        "non-DLops VALU write VGPR -> V_MFMA read VGPR",
        acc_write(a(0), v(0)),
        bf16_16(a(0), v(2), v(4), a(0)),
        2,
    );
    case(
        "non-DLops VALU write VGPR -> V_MFMA read VGPR",
        v_mov(v(4), v(9)),
        bf16_16(a(0), v(2), v(4), a(0)),
        2,
    );
}

#[test]
fn valu_exec_then_mfma() {
    case(
        "write EXEC -> V_MFMA*",
        vopc(VopcOp::CmpxEqU32),
        bf16_16(a(0), v(2), v(4), a(0)),
        4,
    )
}

#[test]
fn dlops_rules() {
    case(
        "same DLops opcode reads it as SrcA/B",
        dot2(v(0), v(1), v(2), v(3)),
        dot2(v(4), v(0), v(5), v(6)),
        3,
    );
    case(
        "any other opcode reads/writes it",
        dot2(v(0), v(1), v(2), v(3)),
        vop2(Vop2Op::AddF32, v(7), v(0), v(8)),
        3,
    );
    case(
        "any other opcode reads/writes it",
        dot2(v(0), v(1), v(2), v(3)),
        v_mov(v(0), v(8)),
        3,
    );
    case(
        "any other opcode reads/writes it",
        dot2(v(4), v(1), v(2), v(3)),
        bf16_16(a(0), v(4), v(6), a(0)),
        3,
    );
    clean(
        "same-opcode SrcC accumulation",
        &[dot2(v(0), v(1), v(2), v(0)), dot2(v(0), v(4), v(5), v(0))],
    );
}

#[test]
fn mfma_srcc_exact_and_overlapped() {
    clean(
        "same XDL opcode, exact SrcC",
        &[
            bf16_16(a(0), v(2), v(4), a(0)),
            bf16_16(a(0), v(6), v(8), a(0)),
        ],
    );
    clean(
        "same XDL opcode, exact SrcC (8 passes)",
        &[
            bf16_32(a(0), v(2), v(4), a(0)),
            bf16_32(a(0), v(6), v(8), a(0)),
        ],
    );
    // Different opcode, same passes, same vDst: LLVM isSameMFMA → overlapped (4 + 1).
    case(
        "Source C, same vDst, different opcode",
        bf16_16(a(0), v(2), v(4), a(0)),
        mfma(MaiOp::MfmaF32_16x16x16F16, a(0), v(6), v(8), a(0)),
        5,
    );
    case(
        "Xdl (8 passes) write VGPR -> Xdl read VGPR as Source C, overlapped",
        bf16_32(a(0), v(2), v(4), a(0)),
        bf16_16(a(4), v(6), v(8), a(4)),
        9,
    );
    case(
        "Xdl (4 passes) write VGPR -> Sgemm read VGPR as Source C, overlapped",
        bf16_16(a(0), v(2), v(4), a(0)),
        mfma(MaiOp::MfmaF32_16x16x4F32, a(2), v(6), v(7), a(2)),
        5,
    );
    // SGEMM writer: ISA 8 for an overlapped XDL SrcC (LLVM 0), 0 when exact.
    case(
        "Sgemm (8 passes) write VGPR -> Xdl read VGPR as Source C, overlapped",
        mfma(MaiOp::MfmaF32_16x16x4F32, a(0), v(2), v(3), a(0)),
        bf16_32(a(0), v(6), v(8), a(0)),
        8,
    );
    clean(
        "SGEMM -> XDL exact SrcC",
        &[
            mfma(MaiOp::MfmaF32_16x16x4F32, a(0), v(2), v(3), a(0)),
            bf16_16(a(0), v(6), v(8), a(0)),
        ],
    );
    case(
        "Sgemm (16 passes) write VGPR -> Sgemm read VGPR as Source C, overlapped",
        mfma(MaiOp::MfmaF32_32x32x2F32, a(0), v(2), v(3), a(0)),
        mfma(MaiOp::MfmaF32_16x16x4F32, a(4), v(6), v(7), a(4)),
        16,
    );
    // DGEMM 16x16x4: exact 0, S/DGEMM overlapped 9, XDL overlapped 0.
    let dgemm = |d: Src, c: Src| mfma(MaiOp::MfmaF64_16x16x4F64, d, v(20), v(22), c);
    clean("DGEMM exact SrcC", &[dgemm(v(0), v(0)), dgemm(v(0), v(0))]);
    case(
        "Dgemm (8 passes) write VGPR -> Sgemm read VGPR as Source C",
        dgemm(v(0), v(0)),
        mfma(MaiOp::MfmaF32_16x16x4F32, v(4), v(24), v(25), v(4)),
        9,
    );
    clean(
        "DGEMM -> XDL overlapped SrcC",
        &[dgemm(v(0), v(0)), bf16_16(v(4), v(24), v(26), v(4))],
    );
}

#[test]
fn mfma_result_then_srcab() {
    case(
        "Xdl (4 passes) write VGPR -> V_MFMA read VGPR as SrcA or SrcB",
        bf16_16(v(0), v(4), v(6), v(0)),
        bf16_16(v(8), v(0), v(6), v(8)),
        7,
    );
    case(
        "Xdl (8 passes) write VGPR -> V_MFMA read VGPR as SrcA or SrcB",
        bf16_32(v(0), v(16), v(18), v(0)),
        bf16_16(v(20), v(16), v(14), v(20)),
        11,
    );
    case(
        "Sgemm (16 passes) write VGPR -> V_MFMA read VGPR as SrcA or SrcB",
        mfma(MaiOp::MfmaF32_32x32x2F32, v(0), v(16), v(17), v(0)),
        mfma(MaiOp::MfmaF32_16x16x4F32, v(20), v(3), v(17), v(20)),
        18,
    );
    case(
        "Dgemm (8 passes) write VGPR -> V_MFMA read VGPR as SrcA or SrcB",
        mfma(MaiOp::MfmaF64_16x16x4F64, v(0), v(20), v(22), v(0)),
        mfma(MaiOp::MfmaF64_16x16x4F64, v(10), v(2), v(22), v(10)),
        11,
    );
}

#[test]
fn mfma_result_then_valu() {
    case(
        "Xdl (8 passes) write VGPR -> VALU read/write",
        bf16_32(a(0), v(2), v(4), a(0)),
        acc_read(v(0), a(3)),
        11,
    );
    // Inline SrcC, so only the WAW row applies (no SrcC WAR).
    case(
        "Xdl (8 passes) write VGPR -> VALU read/write",
        bf16_32(a(0), v(2), v(4), Src::Int(0)),
        acc_write(a(15), v(0)),
        11,
    );
    case(
        "Xdl (4 passes) write VGPR -> VALU read/write",
        bf16_16(v(0), v(4), v(6), v(0)),
        vop2(Vop2Op::AddF32, v(9), v(1), v(9)),
        7,
    );
    case(
        "Sgemm (8 passes) write VGPR -> VALU read/write",
        mfma(MaiOp::MfmaF32_16x16x4F32, a(0), v(2), v(3), a(0)),
        acc_read(v(0), a(1)),
        10,
    );
    case(
        "Dgemm (8 passes) write VGPR -> VALU read/write",
        mfma(MaiOp::MfmaF64_16x16x4F64, v(0), v(20), v(22), v(0)),
        v_mov(v(30), v(7)),
        11,
    );
    clean(
        "accvgpr read of another AGPR",
        &[bf16_16(a(0), v(2), v(4), a(0)), acc_read(v(0), a(4))],
    );
}

#[test]
fn mfma_result_then_memory() {
    case(
        "Xdl (4 passes) write VGPR -> VM, L/GDS, FLAT read",
        bf16_16(v(0), v(4), v(6), v(0)),
        global(FlatOp::StoreDword, None, v(10), Some(v(2)), None),
        7,
    );
    case(
        "Xdl (4 passes) write VGPR -> VM, L/GDS, FLAT read",
        bf16_16(v(0), v(4), v(6), v(0)),
        Inst::Ds(Ds {
            op: DsOp::WriteB32,
            vdst: None,
            addr: Some(v(12)),
            data0: Some(v(3)),
            data1: None,
            offset0: 0,
            offset1: 0,
            gds: false,
        }),
        7,
    );
    case(
        "Dgemm (8 passes) write VGPR -> VM, L/GDS, FLAT read",
        mfma(MaiOp::MfmaF64_16x16x4F64, v(0), v(20), v(22), v(0)),
        global(FlatOp::StoreDword, None, v(10), Some(v(7)), None),
        18,
    );
    case(
        "VMEM/DS write of the same VGPR",
        bf16_16(v(0), v(4), v(6), v(0)),
        global(FlatOp::LoadDword, Some(v(1)), v(10), None, None),
        7,
    );
}

#[test]
fn xdl_srcc_then_valu_write() {
    case(
        "XDL (8 passes) read VGPR SrcC",
        bf16_32(a(16), v(2), v(4), a(0)),
        acc_write(a(0), v(0)),
        7,
    );
    case(
        "XDL (4 passes) read VGPR SrcC",
        bf16_16(a(16), v(2), v(4), a(0)),
        acc_write(a(3), v(0)),
        3,
    );
    clean(
        "SGEMM SrcC WAR (XDL only on gfx940)",
        &[
            mfma(MaiOp::MfmaF32_16x16x4F32, a(16), v(2), v(3), a(0)),
            acc_write(a(0), v(0)),
        ],
    );
}

#[test]
fn dgemm_then_fma_f64() {
    case(
        "DGEMM -> v_fma_f64",
        mfma(MaiOp::MfmaF64_16x16x4F64, v(0), v(20), v(22), v(0)),
        only(
            Vop3OnlyOp::FmaF64,
            v(30),
            [Some(v(40)), Some(v(42)), Some(v(44))],
            0,
        ),
        2,
    );
}

#[test]
fn proofs_split_long_waits_into_s_nop_7() {
    let (out, proofs) = resolve(
        A,
        items(&[
            mfma(MaiOp::MfmaF32_32x32x2F32, v(0), v(16), v(17), v(0)),
            mfma(MaiOp::MfmaF32_16x16x4F32, v(20), v(3), v(17), v(20)),
        ]),
    )
    .unwrap();
    let nops: Vec<&str> = proofs.iter().map(|p| p.insn.as_str()).collect();
    assert_eq!(nops, ["s_nop 7", "s_nop 7", "s_nop 1"]);
    assert_eq!(
        proofs.iter().map(|p| p.pc_index).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(out.len(), 5);
}

// ---- control flow ----

#[test]
fn branch_join_takes_the_shortest_path() {
    // The taken s_cbranch_execz reaches the DPP 1 wait state after v_cmpx;
    // the fallthrough 7. Table 11: VALU writes EXEC -> DPP needs 5.
    let mut prog = vec![
        Item::Inst(vopc(VopcOp::CmpxEqU32)),
        Item::Branch {
            op: SoppOp::CbranchExecz,
            target: "skip".into(),
        },
    ];
    prog.extend(std::iter::repeat_with(|| Item::Inst(filler())).take(6));
    prog.push(Item::Label("skip".into()));
    prog.push(Item::Inst(v_mov_dpp(v(2), v(3))));
    prog.push(Item::Inst(endpgm()));
    let raw = lower(&prog);
    let found = check(A, &raw);
    assert!(
        found
            .iter()
            .any(|f| f.starts_with("8: ") && f.contains("VALU writes EXEC -> VALU DPP op")),
        "{found:#?}"
    );
    let (out, proofs) = resolve(A, prog.clone()).unwrap();
    assert_eq!(nop_waits(&out), 4);
    assert_eq!(proofs.len(), 1);
    let label = out
        .iter()
        .position(|i| matches!(i, Item::Label(_)))
        .unwrap();
    assert_eq!(
        proofs[0].pc_index,
        label + 1,
        "NOP goes after the label, on the joined path"
    );
    assert!(check(A, &lower(&out)).is_empty());
    // Without the branch the fillers suffice.
    let straight: Vec<Item> = prog
        .into_iter()
        .filter(|i| !matches!(i, Item::Branch { .. }))
        .collect();
    assert_eq!(nop_waits(&resolve(A, straight).unwrap().0), 0);
}

#[test]
fn loop_back_edge_reaches_a_fixpoint() {
    // L: v_accvgpr_read v0, a0 reads the previous iteration's 8-pass XDL
    // result: back edge = s_cmp + s_cbranch = 2 wait states, needs 11.
    let body = |pad: usize| {
        let mut p = vec![
            Item::Label("L".into()),
            Item::Inst(acc_read(v(0), a(0))),
            Item::Inst(bf16_32(a(0), v(2), v(4), a(0))),
        ];
        p.extend(std::iter::repeat_with(|| Item::Inst(filler())).take(pad));
        p.extend([
            Item::Inst(s_cmp()),
            Item::Branch {
                op: SoppOp::CbranchScc1,
                target: "L".into(),
            },
            Item::Inst(endpgm()),
        ]);
        p
    };
    let prog = body(0);
    let raw = lower(&prog);
    let found = check(A, &raw);
    assert!(
        found
            .iter()
            .any(|f| f.starts_with("0: ")
                && f.contains("Xdl (8 passes) write VGPR -> VALU read/write")),
        "{found:#?}"
    );
    let (out, proofs) = resolve(A, prog).unwrap();
    assert_eq!(nop_waits(&out), 9);
    assert_eq!(
        proofs
            .iter()
            .map(|p| (p.pc_index, p.insn.as_str()))
            .collect::<Vec<_>>(),
        [(1, "s_nop 7"), (2, "s_nop 0")]
    );
    assert!(
        check(A, &lower(&out)).is_empty(),
        "{:#?}",
        check(A, &lower(&out))
    );
    // Nine independent instructions in the body cover the back edge.
    let padded = body(9);
    assert!(check(A, &lower(&padded)).is_empty());
    assert_eq!(nop_waits(&resolve(A, padded).unwrap().0), 0);
    // Eight leave one.
    assert_eq!(nop_waits(&resolve(A, body(8)).unwrap().0), 1);
}

#[test]
fn resolve_rejects_malformed_control_flow() {
    assert!(resolve(
        A,
        vec![Item::Branch {
            op: SoppOp::Branch,
            target: "nowhere".into()
        }]
    )
    .is_err());
    assert!(resolve(A, vec![Item::Label("x".into()), Item::Label("x".into())]).is_err());
    assert!(resolve(
        A,
        vec![Item::Inst(Inst::Sopp(Sopp {
            op: SoppOp::Branch,
            simm16: 0
        }))]
    )
    .is_err());
}

// ---- LLVM evidence ----

/// `(symbol, code dwords)` of every sized STT_FUNC symbol of an ELF64 code object.
fn functions(elf: &[u8]) -> Vec<(String, Vec<u32>)> {
    let u16_at = |o: usize| u16::from_le_bytes(elf[o..o + 2].try_into().unwrap()) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
    assert_eq!(&elf[..4], b"\x7fELF");
    let (shoff, shentsize, shnum) = (u64_at(0x28), u16_at(0x3a), u16_at(0x3c));
    let sh = |i: usize| shoff + i * shentsize;
    let mut out = Vec::new();
    for i in 0..shnum {
        if u32_at(sh(i) + 4) != 2 {
            continue;
        } // SHT_SYMTAB
        let (off, size, entsize) = (u64_at(sh(i) + 24), u64_at(sh(i) + 32), u64_at(sh(i) + 56));
        let strtab = u64_at(sh(u32_at(sh(i) + 40)) + 24);
        for k in 0..size / entsize {
            let e = off + k * entsize;
            let (info, shndx, value, sz) =
                (elf[e + 4], u16_at(e + 6), u64_at(e + 8), u64_at(e + 16));
            if info & 0xf != 2 || sz == 0 {
                continue;
            }
            let name_at = strtab + u32_at(e);
            let name = String::from_utf8_lossy(
                &elf[name_at..name_at + elf[name_at..].iter().position(|&b| b == 0).unwrap()],
            )
            .into_owned();
            let (addr, file) = (u64_at(sh(shndx) + 16), u64_at(sh(shndx) + 24));
            let start = file + (value - addr);
            out.push((
                name,
                elf[start..start + sz]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ));
        }
    }
    out
}

#[test]
fn llvm_compiled_mfma_kernels_are_hazard_free() {
    if oracle::disabled() {
        eprintln!("skipping: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src");
    let mut checked = 0;
    for kernel in ["gemm_f16_x_f16_mfma", "gemm_bf16_mfma"] {
        let src = root.join(format!("{kernel}.gfx942.hip"));
        let obj =
            std::env::temp_dir().join(format!("pm-cdna-hazard-{}-{kernel}.co", std::process::id()));
        let status = std::process::Command::new(rocm::hipcc())
            .args([
                "--offload-arch=gfx942",
                "-O3",
                "--offload-device-only",
                "--no-gpu-bundle-output",
                "-c",
                "-o",
            ])
            .arg(&obj)
            .arg(&src)
            .status()
            .expect("running hipcc");
        assert!(status.success(), "hipcc failed on {}", src.display());
        let elf = std::fs::read(&obj).unwrap();
        let _ = std::fs::remove_file(&obj);
        let funcs = functions(&elf);
        assert!(!funcs.is_empty(), "{kernel}: no kernel code found");
        for (name, words) in funcs {
            let insts = match decode_all(A, &words) {
                Ok(d) => d.into_iter().map(|(_, i)| i).collect::<Vec<_>>(),
                Err(e) => {
                    eprintln!("skipping {kernel}::{name}: contains an instruction outside the PeaceMaker CDNA table ({e})");
                    continue;
                }
            };
            assert!(
                insts.iter().any(|i| matches!(i, Inst::Mai(_))),
                "{kernel}::{name}: no MFMA"
            );
            let found = check(A, &insts);
            assert!(
                found.is_empty(),
                "{kernel}::{name}: LLVM-scheduled code violates:\n{}",
                found.join("\n")
            );
            // Non-vacuous: with LLVM's wait-state NOPs cut to `s_nop 0`
            // (sizes and branch offsets unchanged) the rules must fire.
            let mut cut = insts.clone();
            let mut load_bearing = false;
            for i in &mut cut {
                if let Inst::Sopp(p) = i {
                    if p.op == SoppOp::Nop && p.simm16 & 0xf != 0 {
                        p.simm16 = 0;
                        load_bearing = true
                    }
                }
            }
            if load_bearing {
                let fired = check(A, &cut);
                assert!(
                    !fired.is_empty(),
                    "{kernel}::{name}: LLVM's s_nops were not needed by any rule"
                );
                eprintln!("{kernel}::{name}: with LLVM's s_nop N cut to s_nop 0, check reports {} violations, e.g. {}", fired.len(), fired[0]);
            }
            eprintln!(
                "{kernel}::{name}: {} instructions, hazard-free",
                insts.len()
            );
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("no MFMA kernel decoded fully; LLVM evidence not exercised")
    }
}
