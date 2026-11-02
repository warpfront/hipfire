// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Instruction constructors. VALU helpers choose the 32-bit encoding when
//! the operands allow it (src1 a VGPR, no modifiers, VCC as the implicit
//! scalar operand) and the VOP3 form otherwise, as the assembler does.
//!
//! Selection rules (gfx9 / CDNA3):
//! - The 32-bit form is chosen when the destination is a VGPR (an SGPR for
//!   `v_readfirstlane_b32`, an AGPR for `v_accvgpr_mov_b32`), `src1` is a VGPR,
//!   no operand is an AGPR, the carry-out / carry-in / compare scalar operand is
//!   VCC, and at most one literal is present (in `src0`; a literal in any other
//!   position cannot be a 32-bit-form operand).
//! - Otherwise the promoted `Vop3` is built (`Vop3Op::V1/V2/C`). gfx9 VOP3 has
//!   no literal dword (CDNA3 ISA §13.3.5): when a literal forces VOP3 the
//!   helper still returns the VOP3 value and `Inst::encode` rejects it. The
//!   builder therefore never silently rewrites a literal into a register.
//! - `v3`, `v3b`, `v3m` always build VOP3; `v3m` takes no scalar destination,
//!   so VOP3B opcodes go through `v3b` / `v2c` / `v2ci`.
//! - DPP/SDWA helpers use neutral modifiers (no neg/abs/sext/clamp/omod, full
//!   selects). The SDWA `dst_unused` is what llvm-mc assembles when the text
//!   omits it: `Pad` for the FP8/BF8 converts (whose assembler syntax has no
//!   `dst_unused`), `Preserve` otherwise.
//! - Memory helpers set neutral cache bits (`sc0 = sc1 = nt = false`, `glc`
//!   false, `gds` false). `buffer_*` with a `vaddr` and `offen == false` selects
//!   the index (`idxen`) addressing mode.
use peacemaker_cdna::enc::{
    ds::{Ds, DsOp},
    flat::{Flat, FlatOp, FlatSeg},
    mubuf::{Mubuf, MubufOp},
    smem::{Smem, SmemOffset, SmemOp},
    sop::{Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp},
    vop::{
        Dpp, DppCtrl, Sdwa, SdwaSel, SdwaUnused, Vop1, Vop1Op, Vop2, Vop2Op, VopExt, Vopc, VopcOp,
    },
    vop3::{Vop3, Vop3OnlyOp, Vop3Op},
    vop3p::{Mai, MaiOp, Vop3p, Vop3pOp},
};
use peacemaker_cdna::{Inst, Src};

/// VOP3 modifiers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods {
    pub abs: u8,
    pub neg: u8,
    pub clamp: bool,
    pub omod: u8,
    pub op_sel: u8,
}

fn src3(srcs: &[Src]) -> [Option<Src>; 3] {
    assert!(
        srcs.len() <= 3,
        "a VOP3 instruction has at most three sources, got {}",
        srcs.len()
    );
    let mut out = [None; 3];
    for (slot, s) in out.iter_mut().zip(srcs) {
        *slot = Some(*s);
    }
    out
}

/// `src0` of a 32-bit form: anything but an AGPR (a literal is the one dword the
/// form carries); `src1`: a VGPR, which also rules out a second literal.
fn e32_srcs_ok(a: Src, b: Option<Src>) -> bool {
    !a.is_agpr() && b.map_or(true, |b| b.is_vgpr())
}

fn vop3(op: Vop3Op, dst: Src, sdst: Option<Src>, srcs: &[Src], m: Mods) -> Inst {
    Inst::Vop3(Vop3 {
        op,
        dst,
        sdst,
        src: src3(srcs),
        abs: m.abs,
        neg: m.neg,
        clamp: m.clamp,
        omod: m.omod,
        op_sel: m.op_sel,
    })
}

pub fn s1(op: Sop1Op, dst: Src, src: Src) -> Inst {
    match op {
        Sop1Op::GetpcB64 => Inst::Sop1(Sop1 {
            op,
            sdst: Some(dst),
            ssrc0: None,
        }),
        Sop1Op::SetpcB64 => Inst::Sop1(Sop1 {
            op,
            sdst: None,
            ssrc0: Some(src),
        }),
        _ => Inst::Sop1(Sop1 {
            op,
            sdst: Some(dst),
            ssrc0: Some(src),
        }),
    }
}
pub fn s2(op: Sop2Op, dst: Src, a: Src, b: Src) -> Inst {
    Inst::Sop2(Sop2 {
        op,
        sdst: dst,
        ssrc0: a,
        ssrc1: b,
    })
}
pub fn sk(op: SopkOp, dst: Src, imm: u16) -> Inst {
    Inst::Sopk(Sopk {
        op,
        sdst: dst,
        simm16: imm,
    })
}
pub fn sc(op: SopcOp, a: Src, b: Src) -> Inst {
    Inst::Sopc(Sopc {
        op,
        ssrc0: a,
        ssrc1: b,
    })
}
/// SOPP other than branches (`s_nop`, `s_sleep`, `s_setprio`, ...).
pub fn sp(op: SoppOp, imm: u16) -> Inst {
    Inst::Sopp(Sopp { op, simm16: imm })
}
/// `s_load_dword*` / `s_buffer_load_dword*` with an immediate byte offset.
pub fn s_load(op: SmemOp, dst: Src, base: Src, offset: i32) -> Inst {
    Inst::Smem(Smem {
        op,
        sdata: Some(dst),
        sbase: Some(base),
        offset: SmemOffset::Imm(offset),
        glc: false,
    })
}

/// Is the 32-bit VOP1 destination legal for `op`?
fn vop1_e32_dst_ok(op: Vop1Op, dst: Src) -> bool {
    match op {
        Vop1Op::ReadfirstlaneB32 => dst.is_sreg(),
        Vop1Op::AccvgprMovB32 => dst.is_agpr(),
        _ => dst.is_vgpr(),
    }
}

/// VOP1: the 32-bit form unless `dst` / `src` need the promoted encoding.
/// `v_accvgpr_mov_b32` keeps its AGPR operands in the 32-bit form.
pub fn v1(op: Vop1Op, dst: Src, src: Src) -> Inst {
    let src_ok = if op == Vop1Op::AccvgprMovB32 {
        src.is_agpr()
    } else {
        e32_srcs_ok(src, None)
    };
    if vop1_e32_dst_ok(op, dst) && src_ok {
        Inst::Vop1(Vop1 {
            op,
            vdst: dst,
            src0: src,
            ext: VopExt::None,
        })
    } else {
        vop3(Vop3Op::V1(op), dst, None, &[src], Mods::default())
    }
}

pub fn v2(op: Vop2Op, dst: Src, a: Src, b: Src) -> Inst {
    if dst.is_vgpr() && e32_srcs_ok(a, Some(b)) {
        Inst::Vop2(Vop2 {
            op,
            vdst: dst,
            src0: a,
            src1: b,
            ext: VopExt::None,
        })
    } else {
        vop3(Vop3Op::V2(op), dst, None, &[a, b], Mods::default())
    }
}

/// Carry-out VOP2 (`v_add_co_u32` ...): `sdst` VCC → e32 when legal, else VOP3B.
pub fn v2c(op: Vop2Op, dst: Src, sdst: Src, a: Src, b: Src) -> Inst {
    if sdst == Src::VccLo && dst.is_vgpr() && e32_srcs_ok(a, Some(b)) {
        Inst::Vop2(Vop2 {
            op,
            vdst: dst,
            src0: a,
            src1: b,
            ext: VopExt::None,
        })
    } else {
        vop3(Vop3Op::V2(op), dst, Some(sdst), &[a, b], Mods::default())
    }
}

/// Carry-in VOP2 (`v_addc_co_u32`, `v_cndmask_b32`): `cin` VCC → e32 when legal.
/// `sdst` is the carry-out of `v_addc_co_u32` (`None` for `v_cndmask_b32`); the
/// e32 form needs it to be VCC (or absent).
pub fn v2ci(op: Vop2Op, dst: Src, sdst: Option<Src>, a: Src, b: Src, cin: Src) -> Inst {
    let sdst_vcc = sdst.map_or(true, |s| s == Src::VccLo);
    if cin == Src::VccLo && sdst_vcc && dst.is_vgpr() && e32_srcs_ok(a, Some(b)) {
        Inst::Vop2(Vop2 {
            op,
            vdst: dst,
            src0: a,
            src1: b,
            ext: VopExt::None,
        })
    } else {
        vop3(Vop3Op::V2(op), dst, sdst, &[a, b, cin], Mods::default())
    }
}

/// Compare into `sdst` (VCC → e32 when legal).
pub fn vc(op: VopcOp, sdst: Src, a: Src, b: Src) -> Inst {
    if sdst == Src::VccLo && e32_srcs_ok(a, Some(b)) {
        Inst::Vopc(Vopc {
            op,
            src0: a,
            src1: b,
            ext: VopExt::None,
        })
    } else {
        vop3(Vop3Op::C(op), sdst, None, &[a, b], Mods::default())
    }
}

/// VOP3-only op without modifiers.
pub fn v3(op: Vop3OnlyOp, dst: Src, srcs: &[Src]) -> Inst {
    vop3(Vop3Op::Only(op), dst, None, srcs, Mods::default())
}
/// VOP3B op (`v_div_scale_f32`, `v_mad_u64_u32`, ...).
pub fn v3b(op: Vop3OnlyOp, dst: Src, sdst: Src, srcs: &[Src]) -> Inst {
    vop3(Vop3Op::Only(op), dst, Some(sdst), srcs, Mods::default())
}
/// Any VOP3 op with modifiers.
pub fn v3m(op: Vop3Op, dst: Src, srcs: &[Src], mods: Mods) -> Inst {
    vop3(op, dst, None, srcs, mods)
}

fn dpp(ctrl: DppCtrl, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> VopExt {
    VopExt::Dpp(Dpp {
        ctrl,
        row_mask,
        bank_mask,
        bound_ctrl,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    })
}

/// VOP1 with DPP on `src` (a VGPR).
pub fn dpp1(
    op: Vop1Op,
    dst: Src,
    src: Src,
    ctrl: DppCtrl,
    row_mask: u8,
    bank_mask: u8,
    bound_ctrl: bool,
) -> Inst {
    Inst::Vop1(Vop1 {
        op,
        vdst: dst,
        src0: src,
        ext: dpp(ctrl, row_mask, bank_mask, bound_ctrl),
    })
}
/// VOP2 with DPP on `a` (a VGPR).
pub fn dpp2(
    op: Vop2Op,
    dst: Src,
    a: Src,
    b: Src,
    ctrl: DppCtrl,
    row_mask: u8,
    bank_mask: u8,
    bound_ctrl: bool,
) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst: dst,
        src0: a,
        src1: b,
        ext: dpp(ctrl, row_mask, bank_mask, bound_ctrl),
    })
}
/// VOP1 with SDWA source select (`v_cvt_f32_fp8` byte, `v_cvt_pk_f32_fp8` word).
pub fn sdwa1(op: Vop1Op, dst: Src, src: Src, src_sel: SdwaSel) -> Inst {
    let dst_unused = match op {
        Vop1Op::CvtF32Fp8 | Vop1Op::CvtF32Bf8 | Vop1Op::CvtPkF32Fp8 | Vop1Op::CvtPkF32Bf8 => {
            SdwaUnused::Pad
        }
        _ => SdwaUnused::Preserve,
    };
    let ext = VopExt::Sdwa(Sdwa {
        dst_sel: SdwaSel::Dword,
        dst_unused,
        clamp: false,
        omod: 0,
        src0_sel: src_sel,
        src0_sext: false,
        src0_neg: false,
        src0_abs: false,
        src1_sel: SdwaSel::Dword,
        src1_sext: false,
        src1_neg: false,
        src1_abs: false,
        sdst: None,
    });
    Inst::Vop1(Vop1 {
        op,
        vdst: dst,
        src0: src,
        ext,
    })
}

/// Packed math with no NEG_LO/NEG_HI/clamp. The OP_SEL_HI bit of an absent
/// source is forced to 1: that is the assembler's default and the only
/// canonical value the encoder accepts (the bit selects nothing there).
pub fn vp(op: Vop3pOp, dst: Src, srcs: &[Src], op_sel: u8, op_sel_hi: u8) -> Inst {
    let src = src3(srcs);
    let absent: u8 = src
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_none())
        .map(|(i, _)| 1u8 << i)
        .fold(0, |m, b| m | b);
    Inst::Vop3p(Vop3p {
        op,
        dst,
        src,
        op_sel,
        op_sel_hi: op_sel_hi | absent,
        neg_lo: 0,
        neg_hi: 0,
        clamp: false,
    })
}
/// MFMA `dst = c + a·b` (dst/c AGPRs or VGPRs, `c` may be inline 0).
///
/// The AGPR/VGPR choice (ACC_CD, ACC bits of A/B) is carried by the operand
/// kinds: `Src::Agpr` sets the bit, `Src::Vgpr` clears it.
pub fn mfma(op: MaiOp, dst: Src, a: Src, b: Src, c: Src) -> Inst {
    Inst::Mai(Mai {
        op,
        dst,
        src_a: a,
        src_b: Some(b),
        src_c: Some(c),
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}
pub fn acc_read(dst: Src, src: Src) -> Inst {
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
pub fn acc_write(dst: Src, src: Src) -> Inst {
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

/// `global_load_*`: `saddr` `None` → 64-bit VGPR address in `vaddr`.
pub fn global_load(op: FlatOp, dst: Src, vaddr: Src, saddr: Option<Src>, offset: i16) -> Inst {
    Inst::Flat(Flat {
        op,
        seg: FlatSeg::Global,
        vdst: Some(dst),
        addr: Some(vaddr),
        data: None,
        saddr,
        offset,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
pub fn global_store(op: FlatOp, vaddr: Src, data: Src, saddr: Option<Src>, offset: i16) -> Inst {
    Inst::Flat(Flat {
        op,
        seg: FlatSeg::Global,
        vdst: None,
        addr: Some(vaddr),
        data: Some(data),
        saddr,
        offset,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
pub fn buffer_load(
    op: MubufOp,
    dst: Src,
    vaddr: Option<Src>,
    srsrc: Src,
    soffset: Src,
    offset: u16,
    offen: bool,
) -> Inst {
    Inst::Mubuf(Mubuf {
        op,
        vdata: dst,
        vaddr,
        srsrc,
        soffset,
        offset,
        offen,
        idxen: vaddr.is_some() && !offen,
        sc0: false,
        sc1: false,
        nt: false,
    })
}
pub fn buffer_store(
    op: MubufOp,
    data: Src,
    vaddr: Option<Src>,
    srsrc: Src,
    soffset: Src,
    offset: u16,
    offen: bool,
) -> Inst {
    Inst::Mubuf(Mubuf {
        op,
        vdata: data,
        vaddr,
        srsrc,
        soffset,
        offset,
        offen,
        idxen: vaddr.is_some() && !offen,
        sc0: false,
        sc1: false,
        nt: false,
    })
}

fn ds(
    op: DsOp,
    vdst: Option<Src>,
    addr: Option<Src>,
    data0: Option<Src>,
    data1: Option<Src>,
    offset0: u8,
    offset1: u8,
) -> Inst {
    Inst::Ds(Ds {
        op,
        vdst,
        addr,
        data0,
        data1,
        offset0,
        offset1,
        gds: false,
    })
}

pub fn ds_read(op: DsOp, dst: Src, addr: Src, offset: u16) -> Inst {
    ds(
        op,
        Some(dst),
        Some(addr),
        None,
        None,
        offset as u8,
        (offset >> 8) as u8,
    )
}
pub fn ds_write(op: DsOp, addr: Src, data: Src, offset: u16) -> Inst {
    ds(
        op,
        None,
        Some(addr),
        Some(data),
        None,
        offset as u8,
        (offset >> 8) as u8,
    )
}
pub fn ds_read2(op: DsOp, dst: Src, addr: Src, offset0: u8, offset1: u8) -> Inst {
    ds(op, Some(dst), Some(addr), None, None, offset0, offset1)
}
pub fn ds_write2(op: DsOp, addr: Src, data0: Src, data1: Src, offset0: u8, offset1: u8) -> Inst {
    ds(
        op,
        None,
        Some(addr),
        Some(data0),
        Some(data1),
        offset0,
        offset1,
    )
}
/// `ds_swizzle_b32` (`addr` ignored: pass the data VGPR in `data`; the VGPR
/// being swizzled is the instruction's VADDR field, `data0` is absent),
/// `ds_permute_b32`, `ds_bpermute_b32` (VADDR = lane address, `data0` = data).
pub fn ds_xlane(op: DsOp, dst: Src, addr: Src, data: Src, offset: u16) -> Inst {
    let (lo, hi) = (offset as u8, (offset >> 8) as u8);
    match op {
        DsOp::SwizzleB32 => ds(op, Some(dst), Some(data), None, None, lo, hi),
        _ => ds(op, Some(dst), Some(addr), Some(data), None, lo, hi),
    }
}
/// Raw promoted-VOP3 form of a 32-bit op (for tests and for forms `v1`/`v2` would not pick).
pub fn vop3_of(op: Vop3Op, dst: Src, srcs: &[Src]) -> Inst {
    v3m(op, dst, srcs, Mods::default())
}
