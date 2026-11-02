// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cdna::ops` constructors: the e32/e64 selection table (structural, no
//! oracle) and the llvm-mc round trip (every helper's output encodes and its
//! `Display` re-assembles to the same bytes; the round trip needs the
//! `peacemaker-cdna` encoders and `Display` impls).
use peacemaker_cdna::enc::{
    ds::DsOp,
    flat::{FlatOp, FlatSeg},
    mubuf::MubufOp,
    smem::{SmemOffset, SmemOp},
    sop::{Sop1Op, Sop2Op, SopcOp, SopkOp, SoppOp},
    vop::{DppCtrl, SdwaSel, SdwaUnused, Vop1Op, Vop2Op, VopExt, VopcOp},
    vop3::{Vop3OnlyOp, Vop3Op},
    vop3p::{MaiOp, Vop3pOp},
};
use peacemaker_cdna::{oracle, Arch, Inst, Src};
use peacemaker_isa::cdna::ops::*;

const ARCH: Arch = Arch::Gfx942;

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn a(n: u8) -> Src {
    Src::Agpr(n)
}
fn s(n: u8) -> Src {
    Src::Sgpr(n)
}
const VCC: Src = Src::VccLo;

fn is_e32(i: &Inst) -> bool {
    matches!(i, Inst::Vop1(_) | Inst::Vop2(_) | Inst::Vopc(_))
}
fn vop3(i: &Inst) -> &peacemaker_cdna::enc::vop3::Vop3 {
    match i {
        Inst::Vop3(x) => x,
        other => panic!("expected VOP3, got {other:?}"),
    }
}

// ---------------------------------------------------------------- selection

#[test]
fn v1_selection() {
    assert!(is_e32(&v1(Vop1Op::MovB32, v(0), v(1))));
    assert!(is_e32(&v1(Vop1Op::MovB32, v(0), s(1))));
    assert!(is_e32(&v1(Vop1Op::MovB32, v(0), Src::Lit(0x1234_5678))));
    assert!(is_e32(&v1(Vop1Op::MovB32, v(0), Src::Int(-3))));
    assert!(is_e32(&v1(Vop1Op::ReadfirstlaneB32, s(0), v(1))));
    assert!(is_e32(&v1(Vop1Op::AccvgprMovB32, a(0), a(1))));
    // A non-VGPR destination (and an AGPR source of an ordinary op) cannot be
    // a 32-bit form: promote to VOP3 and let `encode` reject it.
    let i = v1(Vop1Op::MovB32, s(0), v(1));
    assert!(!is_e32(&i));
    assert_eq!(vop3(&i).op, Vop3Op::V1(Vop1Op::MovB32));
    assert!(!is_e32(&v1(Vop1Op::MovB32, v(0), a(1))));
    assert!(!is_e32(&v1(Vop1Op::ReadfirstlaneB32, v(0), v(1))));
}

#[test]
fn v2_selection() {
    let e32 = v2(Vop2Op::AddF32, v(0), v(1), v(2));
    assert!(
        matches!(&e32, Inst::Vop2(x) if x.src0 == v(1) && x.src1 == v(2) && x.ext == VopExt::None)
    );
    assert!(is_e32(&v2(Vop2Op::AddF32, v(0), s(1), v(2))));
    assert!(is_e32(&v2(
        Vop2Op::AddF32,
        v(0),
        Src::Float(peacemaker_cdna::operand::InlineFloat::One),
        v(2)
    )));
    // literal in src0 stays 32-bit
    assert!(is_e32(&v2(Vop2Op::AddU32, v(0), Src::Lit(0x1_0000), v(2))));
    // src1 not a VGPR -> VOP3
    for b in [s(2), Src::Int(4), VCC] {
        let i = v2(Vop2Op::AddF32, v(0), v(1), b);
        let x = vop3(&i);
        assert_eq!(
            (x.op, x.dst, x.sdst, x.src),
            (
                Vop3Op::V2(Vop2Op::AddF32),
                v(0),
                None,
                [Some(v(1)), Some(b), None]
            )
        );
    }
    // A literal that would need VOP3 (src1) is impossible on gfx9: the VOP3 form
    // is returned with the literal intact and `encode` rejects it.
    let i = v2(Vop2Op::AddU32, v(0), v(1), Src::Lit(7));
    assert_eq!(vop3(&i).src[1], Some(Src::Lit(7)));
    // two literals: VOP3 as well
    assert!(!is_e32(&v2(Vop2Op::AddU32, v(0), Src::Lit(1), Src::Lit(2))));
    // non-VGPR destination
    assert!(!is_e32(&v2(Vop2Op::AddU32, s(0), v(1), v(2))));
}

#[test]
fn v2c_selection() {
    let i = v2c(Vop2Op::AddCoU32, v(0), VCC, v(1), v(2));
    assert!(
        matches!(&i, Inst::Vop2(x) if x.op == Vop2Op::AddCoU32 && x.vdst == v(0) && x.src0 == v(1) && x.src1 == v(2))
    );
    assert!(is_e32(&v2c(Vop2Op::SubCoU32, v(0), VCC, Src::Lit(5), v(2))));
    // SGPR-pair carry-out -> VOP3B
    let i = v2c(Vop2Op::AddCoU32, v(0), s(2), v(1), v(2));
    let x = vop3(&i);
    assert_eq!(
        (x.op, x.dst, x.sdst, x.src),
        (
            Vop3Op::V2(Vop2Op::AddCoU32),
            v(0),
            Some(s(2)),
            [Some(v(1)), Some(v(2)), None]
        )
    );
    // src1 not a VGPR with VCC carry-out -> VOP3B with sdst = vcc
    let i = v2c(Vop2Op::AddCoU32, v(0), VCC, v(1), s(2));
    assert_eq!(vop3(&i).sdst, Some(VCC));
}

#[test]
fn v2ci_selection() {
    let i = v2ci(Vop2Op::AddcCoU32, v(0), Some(VCC), v(1), v(2), VCC);
    assert!(matches!(&i, Inst::Vop2(x) if x.op == Vop2Op::AddcCoU32));
    let i = v2ci(Vop2Op::AddcCoU32, v(0), None, v(1), v(2), VCC);
    assert!(is_e32(&i));
    assert!(is_e32(&v2ci(
        Vop2Op::CndmaskB32,
        v(0),
        None,
        v(1),
        v(2),
        VCC
    )));
    // mask in an SGPR pair -> VOP3A with src2 = mask
    let i = v2ci(Vop2Op::CndmaskB32, v(0), None, v(1), v(2), s(4));
    let x = vop3(&i);
    assert_eq!(
        (x.op, x.sdst, x.src),
        (
            Vop3Op::V2(Vop2Op::CndmaskB32),
            None,
            [Some(v(1)), Some(v(2)), Some(s(4))]
        )
    );
    // carry chain through SGPR pairs -> VOP3B
    let i = v2ci(Vop2Op::AddcCoU32, v(0), Some(s(2)), v(1), v(2), s(4));
    let x = vop3(&i);
    assert_eq!(
        (x.sdst, x.src),
        (Some(s(2)), [Some(v(1)), Some(v(2)), Some(s(4))])
    );
    // VCC carry-in but SGPR carry-out -> VOP3B
    assert!(!is_e32(&v2ci(
        Vop2Op::AddcCoU32,
        v(0),
        Some(s(2)),
        v(1),
        v(2),
        VCC
    )));
    // src1 an SGPR -> VOP3
    assert!(!is_e32(&v2ci(
        Vop2Op::CndmaskB32,
        v(0),
        None,
        v(1),
        s(2),
        VCC
    )));
}

#[test]
fn vc_selection() {
    let i = vc(VopcOp::CmpLtU32, VCC, v(1), v(2));
    assert!(
        matches!(&i, Inst::Vopc(x) if x.op == VopcOp::CmpLtU32 && x.src0 == v(1) && x.src1 == v(2))
    );
    assert!(is_e32(&vc(
        VopcOp::CmpLtF32,
        VCC,
        Src::Lit(0x3f80_0000),
        v(2)
    )));
    assert!(is_e32(&vc(VopcOp::CmpEqU64, VCC, v(2), v(4))));
    let i = vc(VopcOp::CmpLtU32, s(2), v(1), v(2));
    let x = vop3(&i);
    assert_eq!(
        (x.op, x.dst, x.sdst, x.src),
        (
            Vop3Op::C(VopcOp::CmpLtU32),
            s(2),
            None,
            [Some(v(1)), Some(v(2)), None]
        )
    );
    assert!(!is_e32(&vc(VopcOp::CmpLtU32, VCC, v(1), s(2))));
    assert!(!is_e32(&vc(VopcOp::CmpLtU32, VCC, v(1), Src::Int(5))));
}

#[test]
fn vop3_helpers() {
    let i = v3(Vop3OnlyOp::FmaF32, v(0), &[v(1), v(2), v(3)]);
    let x = vop3(&i);
    assert_eq!(
        (x.op, x.sdst, x.abs, x.neg, x.clamp, x.omod, x.op_sel),
        (Vop3Op::Only(Vop3OnlyOp::FmaF32), None, 0, 0, false, 0, 0)
    );
    assert_eq!(x.src, [Some(v(1)), Some(v(2)), Some(v(3))]);
    let i = v3(Vop3OnlyOp::CvtPkFp8F32, v(0), &[v(1), v(2)]);
    assert_eq!(vop3(&i).src, [Some(v(1)), Some(v(2)), None]);
    let i = v3b(Vop3OnlyOp::DivScaleF32, v(0), VCC, &[v(1), v(2), v(3)]);
    assert_eq!(vop3(&i).sdst, Some(VCC));
    let m = Mods {
        abs: 0b001,
        neg: 0b010,
        clamp: true,
        omod: 2,
        op_sel: 0b1000,
    };
    let i = v3m(Vop3Op::V1(Vop1Op::CvtF32Fp8), v(0), &[v(1)], m);
    let x = vop3(&i);
    assert_eq!(
        (x.op, x.abs, x.neg, x.clamp, x.omod, x.op_sel, x.sdst),
        (Vop3Op::V1(Vop1Op::CvtF32Fp8), 1, 2, true, 2, 8, None)
    );
    let i = vop3_of(Vop3Op::V2(Vop2Op::MulF32), v(0), &[v(1), v(2)]);
    assert!(
        matches!(&i, Inst::Vop3(x) if x.abs == 0 && x.neg == 0 && !x.clamp && x.omod == 0 && x.op_sel == 0)
    );
}

#[test]
fn dpp_and_sdwa() {
    let i = dpp1(
        Vop1Op::MovB32,
        v(0),
        v(1),
        DppCtrl::RowShr(1),
        0xf,
        0xf,
        true,
    );
    match &i {
        Inst::Vop1(x) => {
            assert_eq!((x.vdst, x.src0), (v(0), v(1)));
            match x.ext {
                VopExt::Dpp(d) => {
                    assert_eq!(
                        (d.ctrl, d.row_mask, d.bank_mask, d.bound_ctrl),
                        (DppCtrl::RowShr(1), 0xf, 0xf, true)
                    );
                    assert!(!(d.src0_neg || d.src0_abs || d.src1_neg || d.src1_abs));
                }
                other => panic!("{other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }
    let i = dpp2(
        Vop2Op::AddU32,
        v(0),
        v(1),
        v(2),
        DppCtrl::QuadPerm([1, 0, 3, 2]),
        0xf,
        0xf,
        false,
    );
    match &i {
        Inst::Vop2(x) => {
            assert_eq!((x.vdst, x.src0, x.src1), (v(0), v(1), v(2)));
            assert!(
                matches!(x.ext, VopExt::Dpp(d) if d.ctrl == DppCtrl::QuadPerm([1, 0, 3, 2]) && !d.bound_ctrl)
            );
        }
        other => panic!("{other:?}"),
    }
    let i = sdwa1(Vop1Op::CvtF32Fp8, v(0), v(1), SdwaSel::Byte2);
    match &i {
        Inst::Vop1(x) => match x.ext {
            VopExt::Sdwa(w) => {
                assert_eq!(
                    (w.src0_sel, w.dst_sel, w.dst_unused),
                    (SdwaSel::Byte2, SdwaSel::Dword, SdwaUnused::Pad)
                );
                assert!(
                    !(w.clamp || w.src0_sext || w.src0_neg || w.src0_abs)
                        && w.omod == 0
                        && w.sdst.is_none()
                );
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
    let i = sdwa1(Vop1Op::MovB32, v(0), v(1), SdwaSel::Word1);
    assert!(
        matches!(&i, Inst::Vop1(x) if matches!(x.ext, VopExt::Sdwa(w) if w.dst_unused == SdwaUnused::Preserve))
    );
}

#[test]
fn vp_canonicalizes_absent_source_op_sel_hi() {
    let i = vp(Vop3pOp::PkAddF16, v(0), &[v(1), v(2)], 0, 0b011);
    assert!(
        matches!(&i, Inst::Vop3p(x) if x.op_sel_hi == 0b111 && x.op_sel == 0 && x.src[2].is_none())
    );
    let i = vp(Vop3pOp::PkFmaF16, v(0), &[v(1), v(2), v(3)], 0b100, 0b001);
    assert!(matches!(&i, Inst::Vop3p(x) if x.op_sel_hi == 0b001 && x.op_sel == 0b100));
}

#[test]
fn mfma_and_acc_moves() {
    let i = mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, a(0), v(4), v(6), Src::Int(0));
    match &i {
        Inst::Mai(m) => {
            assert_eq!(
                (m.dst, m.src_a, m.src_b, m.src_c),
                (a(0), v(4), Some(v(6)), Some(Src::Int(0)))
            );
            assert_eq!((m.cbsz, m.abid, m.blgp), (0, 0, 0));
        }
        other => panic!("{other:?}"),
    }
    let i = mfma(MaiOp::MfmaF32_32x32x8F16, v(0), a(8), a(10), v(0));
    assert!(
        matches!(&i, Inst::Mai(m) if m.dst == v(0) && m.src_a == a(8) && m.src_b == Some(a(10)) && m.src_c == Some(v(0)))
    );
    let i = acc_read(v(3), a(2));
    assert!(
        matches!(&i, Inst::Mai(m) if m.op == MaiOp::AccvgprReadB32 && m.dst == v(3) && m.src_a == a(2) && m.src_b.is_none() && m.src_c.is_none())
    );
    let i = acc_write(a(2), Src::Int(0));
    assert!(
        matches!(&i, Inst::Mai(m) if m.op == MaiOp::AccvgprWriteB32 && m.dst == a(2) && m.src_a == Src::Int(0))
    );
}

#[test]
fn memory_helpers_use_neutral_bits() {
    let i = global_load(FlatOp::LoadDwordx4, v(0), v(4), Some(s(2)), 16);
    match &i {
        Inst::Flat(f) => {
            assert_eq!(
                (f.seg, f.vdst, f.addr, f.data, f.saddr, f.offset),
                (
                    FlatSeg::Global,
                    Some(v(0)),
                    Some(v(4)),
                    None,
                    Some(s(2)),
                    16
                )
            );
            assert!(!(f.sc0 || f.sc1 || f.nt));
        }
        other => panic!("{other:?}"),
    }
    let i = global_store(FlatOp::StoreDword, v(4), v(1), None, -8);
    assert!(
        matches!(&i, Inst::Flat(f) if f.vdst.is_none() && f.data == Some(v(1)) && f.saddr.is_none() && f.offset == -8 && !(f.sc0 || f.sc1 || f.nt))
    );

    let i = buffer_load(
        MubufOp::LoadDword,
        v(0),
        Some(v(1)),
        s(4),
        Src::Int(0),
        12,
        true,
    );
    assert!(
        matches!(&i, Inst::Mubuf(m) if m.offen && !m.idxen && m.vaddr == Some(v(1)) && m.offset == 12 && !(m.sc0 || m.sc1 || m.nt))
    );
    let i = buffer_load(MubufOp::LoadDword, v(0), Some(v(1)), s(4), s(8), 0, false);
    assert!(matches!(&i, Inst::Mubuf(m) if !m.offen && m.idxen));
    let i = buffer_store(MubufOp::StoreDwordx2, v(2), None, s(4), s(8), 4, false);
    assert!(
        matches!(&i, Inst::Mubuf(m) if !m.offen && !m.idxen && m.vaddr.is_none() && m.vdata == v(2))
    );

    let i = s_load(SmemOp::LoadDwordx4, s(4), s(0), 16);
    assert!(
        matches!(&i, Inst::Smem(m) if m.sdata == Some(s(4)) && m.sbase == Some(s(0)) && m.offset == SmemOffset::Imm(16) && !m.glc)
    );
}

#[test]
fn ds_helpers() {
    let i = ds_read(DsOp::ReadB32, v(0), v(1), 0x1234);
    assert!(
        matches!(&i, Inst::Ds(d) if d.vdst == Some(v(0)) && d.addr == Some(v(1)) && d.data0.is_none() && d.offset() == 0x1234 && !d.gds)
    );
    let i = ds_write(DsOp::WriteB64, v(1), v(2), 8);
    assert!(
        matches!(&i, Inst::Ds(d) if d.vdst.is_none() && d.data0 == Some(v(2)) && d.data1.is_none() && d.offset() == 8)
    );
    let i = ds_read2(DsOp::Read2B32, v(0), v(2), 1, 2);
    assert!(matches!(&i, Inst::Ds(d) if (d.offset0, d.offset1) == (1, 2) && d.vdst == Some(v(0))));
    let i = ds_write2(DsOp::Write2B32, v(2), v(3), v(4), 3, 5);
    assert!(
        matches!(&i, Inst::Ds(d) if (d.offset0, d.offset1) == (3, 5) && d.data0 == Some(v(3)) && d.data1 == Some(v(4)))
    );
    let i = ds_xlane(DsOp::BpermuteB32, v(0), v(1), v(2), 0);
    assert!(matches!(&i, Inst::Ds(d) if d.addr == Some(v(1)) && d.data0 == Some(v(2))));
    let i = ds_xlane(DsOp::SwizzleB32, v(0), v(9), v(2), 0x041f);
    assert!(
        matches!(&i, Inst::Ds(d) if d.addr == Some(v(2)) && d.data0.is_none() && d.offset() == 0x041f)
    );
}

#[test]
fn scalar_helpers() {
    let i = s1(Sop1Op::GetpcB64, s(0), s(0));
    assert!(matches!(&i, Inst::Sop1(x) if x.sdst == Some(s(0)) && x.ssrc0.is_none()));
    let i = s1(Sop1Op::SetpcB64, s(0), s(2));
    assert!(matches!(&i, Inst::Sop1(x) if x.sdst.is_none() && x.ssrc0 == Some(s(2))));
    let i = s1(Sop1Op::MovB32, s(0), Src::Int(5));
    assert!(matches!(&i, Inst::Sop1(x) if x.sdst == Some(s(0)) && x.ssrc0 == Some(Src::Int(5))));
    assert!(
        matches!(s2(Sop2Op::AddU32, s(0), s(1), s(2)), Inst::Sop2(x) if (x.sdst, x.ssrc0, x.ssrc1) == (s(0), s(1), s(2)))
    );
    assert!(matches!(sk(SopkOp::MovkI32, s(0), 5), Inst::Sopk(x) if x.simm16 == 5));
    assert!(
        matches!(sc(SopcOp::CmpEqU32, s(0), s(1)), Inst::Sopc(x) if (x.ssrc0, x.ssrc1) == (s(0), s(1)))
    );
    assert!(matches!(sp(SoppOp::Sleep, 3), Inst::Sopp(x) if x.simm16 == 3));
}

// --------------------------------------------------------- llvm-mc round trip

fn corpus() -> Vec<Inst> {
    use peacemaker_cdna::operand::InlineFloat as F;
    let lit = Src::Lit(0x1234_5678);
    vec![
        // SALU
        s1(Sop1Op::MovB32, s(0), Src::Int(5)),
        s1(Sop1Op::MovB32, s(0), lit),
        s1(Sop1Op::MovB64, s(2), s(4)),
        s1(Sop1Op::AndSaveexecB64, s(4), VCC),
        s1(Sop1Op::GetpcB64, s(0), s(0)),
        s1(Sop1Op::SetpcB64, s(0), s(2)),
        s2(Sop2Op::AddU32, s(0), s(1), s(2)),
        s2(Sop2Op::AddU32, s(0), s(1), lit),
        s2(Sop2Op::LshlB64, s(2), s(4), Src::Int(3)),
        sk(SopkOp::MovkI32, s(0), 5),
        sc(SopcOp::CmpEqU32, s(0), s(1)),
        sp(SoppOp::Nop, 1),
        sp(SoppOp::Sleep, 2),
        sp(SoppOp::Setprio, 1),
        s_load(SmemOp::LoadDwordx4, s(4), s(0), 16),
        s_load(SmemOp::BufferLoadDword, s(4), s(8), 0),
        // VOP1 / VOP2 / VOPC e32 and e64
        v1(Vop1Op::MovB32, v(0), v(1)),
        v1(Vop1Op::MovB32, v(0), s(1)),
        v1(Vop1Op::MovB32, v(0), lit),
        v1(Vop1Op::CvtF32F16, v(0), v(1)),
        v1(Vop1Op::ReadfirstlaneB32, s(0), v(1)),
        v1(Vop1Op::AccvgprMovB32, a(0), a(1)),
        v2(Vop2Op::AddF32, v(0), v(1), v(2)),
        v2(Vop2Op::AddF32, v(0), s(1), v(2)),
        v2(Vop2Op::AddF32, v(0), v(1), s(2)),
        v2(Vop2Op::AddF32, v(0), v(1), Src::Float(F::Two)),
        v2(Vop2Op::AddU32, v(0), lit, v(2)),
        v2(Vop2Op::LshlrevB32, v(0), Src::Int(2), v(3)),
        v2c(Vop2Op::AddCoU32, v(0), VCC, v(1), v(2)),
        v2c(Vop2Op::AddCoU32, v(0), s(2), v(1), v(2)),
        v2c(Vop2Op::SubCoU32, v(0), VCC, v(1), s(6)),
        v2ci(Vop2Op::AddcCoU32, v(0), Some(VCC), v(1), v(2), VCC),
        v2ci(Vop2Op::AddcCoU32, v(0), Some(s(2)), v(1), v(2), s(4)),
        v2ci(Vop2Op::CndmaskB32, v(0), None, v(1), v(2), VCC),
        v2ci(Vop2Op::CndmaskB32, v(0), None, v(1), v(2), s(4)),
        vc(VopcOp::CmpLtU32, VCC, v(1), v(2)),
        vc(VopcOp::CmpLtU32, s(2), v(1), v(2)),
        vc(VopcOp::CmpLtF32, VCC, Src::Lit(0x3f80_0000), v(2)),
        vc(VopcOp::CmpEqU64, VCC, v(2), v(4)),
        vc(VopcOp::CmpGtI32, s(2), v(1), Src::Int(7)),
        // VOP3
        v3(Vop3OnlyOp::FmaF32, v(0), &[v(1), v(2), v(3)]),
        v3(Vop3OnlyOp::BfeU32, v(0), &[v(1), Src::Int(8), Src::Int(4)]),
        v3(Vop3OnlyOp::PermB32, v(0), &[v(1), v(2), s(3)]),
        v3(Vop3OnlyOp::ReadlaneB32, s(0), &[v(1), s(2)]),
        v3(Vop3OnlyOp::CvtPkFp8F32, v(0), &[v(1), v(2)]),
        v3(Vop3OnlyOp::AddF64, v(0), &[v(2), v(4)]),
        v3b(Vop3OnlyOp::DivScaleF32, v(0), VCC, &[v(1), v(2), v(3)]),
        v3b(Vop3OnlyOp::MadU64U32, v(0), s(2), &[v(4), v(5), v(6)]),
        v3m(
            Vop3Op::V2(Vop2Op::AddF32),
            v(0),
            &[v(1), v(2)],
            Mods {
                abs: 1,
                neg: 2,
                clamp: true,
                omod: 1,
                op_sel: 0,
            },
        ),
        v3m(
            Vop3Op::Only(Vop3OnlyOp::FmaF32),
            v(0),
            &[v(1), v(2), v(3)],
            Mods {
                abs: 0,
                neg: 0b101,
                clamp: false,
                omod: 0,
                op_sel: 0,
            },
        ),
        vop3_of(Vop3Op::V1(Vop1Op::MovB32), v(0), &[v(1)]),
        vop3_of(Vop3Op::V2(Vop2Op::MulF32), v(0), &[v(1), v(2)]),
        vop3_of(Vop3Op::C(VopcOp::CmpLtU32), VCC, &[v(1), v(2)]),
        // DPP / SDWA
        dpp1(
            Vop1Op::MovB32,
            v(0),
            v(1),
            DppCtrl::RowShr(1),
            0xf,
            0xf,
            true,
        ),
        dpp1(
            Vop1Op::MovB32,
            v(0),
            v(1),
            DppCtrl::RowMirror,
            0xf,
            0xf,
            false,
        ),
        dpp2(
            Vop2Op::AddU32,
            v(0),
            v(1),
            v(2),
            DppCtrl::QuadPerm([1, 0, 3, 2]),
            0xf,
            0xf,
            false,
        ),
        dpp2(
            Vop2Op::AddF32,
            v(0),
            v(1),
            v(2),
            DppCtrl::RowShl(3),
            0xa,
            0x5,
            true,
        ),
        sdwa1(Vop1Op::CvtF32Fp8, v(0), v(1), SdwaSel::Byte1),
        sdwa1(Vop1Op::CvtPkF32Fp8, v(0), v(2), SdwaSel::Word1),
        sdwa1(Vop1Op::CvtF32Bf8, v(0), v(1), SdwaSel::Byte3),
        sdwa1(Vop1Op::MovB32, v(0), v(1), SdwaSel::Byte2),
        // VOP3P
        vp(Vop3pOp::PkFmaF16, v(0), &[v(1), v(2), v(3)], 0, 0b111),
        vp(Vop3pOp::PkAddF16, v(0), &[v(1), v(2)], 0, 0b011),
        vp(Vop3pOp::PkMulF16, v(0), &[v(1), v(2)], 0b001, 0b011),
        // MFMA / accvgpr
        mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, a(0), v(4), v(6), Src::Int(0)),
        mfma(MaiOp::MfmaF32_16x16x32Fp8Fp8, a(0), v(4), v(6), a(0)),
        mfma(MaiOp::MfmaF32_32x32x8F16, v(0), a(8), a(10), v(0)),
        mfma(MaiOp::MfmaF32_16x16x16Bf16, a(4), v(8), v(10), a(4)),
        acc_read(v(3), a(2)),
        acc_write(a(2), v(3)),
        acc_write(a(2), Src::Int(0)),
        // memory
        global_load(FlatOp::LoadDwordx4, v(0), v(4), None, 0),
        global_load(FlatOp::LoadDword, v(0), v(4), Some(s(2)), 16),
        global_load(FlatOp::LoadUbyte, v(0), v(4), Some(s(2)), -8),
        global_store(FlatOp::StoreDword, v(4), v(1), None, 4),
        global_store(FlatOp::StoreDwordx4, v(4), v(0), Some(s(2)), 0),
        buffer_load(
            MubufOp::LoadDword,
            v(0),
            Some(v(1)),
            s(4),
            Src::Int(0),
            12,
            true,
        ),
        buffer_load(MubufOp::LoadDwordx4, v(0), None, s(4), s(8), 0, false),
        buffer_load(MubufOp::LoadDword, v(0), Some(v(1)), s(4), s(8), 4, false),
        buffer_store(MubufOp::StoreDwordx2, v(2), Some(v(1)), s(4), s(8), 4, true),
        ds_read(DsOp::ReadB32, v(0), v(1), 0),
        ds_read(DsOp::ReadB128, v(0), v(4), 0x1234),
        ds_write(DsOp::WriteB64, v(1), v(2), 8),
        ds_read2(DsOp::Read2B32, v(0), v(2), 1, 2),
        ds_write2(DsOp::Write2B32, v(2), v(3), v(4), 3, 5),
        ds_xlane(DsOp::BpermuteB32, v(0), v(1), v(2), 0),
        ds_xlane(DsOp::PermuteB32, v(0), v(1), v(2), 0),
        ds_xlane(DsOp::SwizzleB32, v(0), v(1), v(1), 0x041f),
    ]
}

#[test]
fn every_helper_round_trips_through_llvm_mc() {
    if oracle::disabled() {
        eprintln!("cdna_ops: llvm-mc oracle disabled; skipping");
        return;
    }
    let cases = corpus();
    let mut words = Vec::new();
    let mut failures = Vec::new();
    let mut lines = Vec::new();
    for (i, inst) in cases.iter().enumerate() {
        let mut w = Vec::new();
        match inst.encode(ARCH, &mut w) {
            Ok(()) => {
                lines.push(inst.to_string());
                words.push((i, w));
            }
            Err(e) => failures.push(format!("#{i} {inst:?}: encode: {e}")),
        }
    }
    let assembled = oracle::mc_encode(ARCH, &lines)
        .unwrap_or_else(|e| panic!("llvm-mc rejected the printed corpus:\n{e}"));
    for ((i, w), (text, bytes)) in words.iter().zip(&assembled) {
        let ours = oracle::bytes(w);
        if &ours != bytes {
            failures.push(format!(
                "#{i} {:?}\n  display `{}` llvm `{text}`\n  ours {ours:02x?}\n  llvm {bytes:02x?}",
                cases[*i], cases[*i]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} helper outputs fail the oracle:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn literal_that_needs_vop3_is_rejected_by_encode() {
    // gfx9 VOP3 has no literal dword: a literal in src1 cannot be encoded.
    // (Needs the VOP3 encoder; the structural form is checked in `v2_selection`.)
    let i = v2(Vop2Op::AddU32, v(0), v(1), Src::Lit(7));
    let mut out = Vec::new();
    assert!(
        i.encode(ARCH, &mut out).is_err(),
        "VOP3 with a literal must not encode on gfx9"
    );
}
