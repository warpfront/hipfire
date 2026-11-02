// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 32-bit vector formats VOP1, VOP2, VOPC and their DPP / SDWA extension
//! dwords (CDNA3 ISA §6.2 p.32-33; VOP2 §13.3.1 p.522, VOP1 §13.3.2 p.524,
//! VOPC §13.3.3 p.526, SDWA §13.3.7 p.542-543, SDWAB §13.3.8 p.543, DPP
//! §13.3.9 p.544-545).
//!
//! Field layout: VOP1 `[8:0]` SRC0, `[16:9]` OP, `[24:17]` VDST, `[31:25]`
//! `0111111`; VOP2 `[8:0]` SRC0, `[16:9]` VSRC1, `[24:17]` VDST, `[30:25]` OP,
//! `[31]` 0; VOPC `[8:0]` SRC0, `[16:9]` VSRC1, `[24:17]` OP, `[31:25]`
//! `0111110`. SRC0 = 249 selects an SDWA dword, 250 a DPP dword, 255 a
//! literal dword; at most one of the three follows.
//!
//! Every form is checked against what llvm's gfx942 assembler accepts, so the
//! printed text (llvm's disassembler spelling) reassembles to the same bytes:
//! - constant bus (§6.2.1 p.32): at most one scalar value (SGPR, VCC, EXEC,
//!   M0, TTMP, `src_*`, or the literal) per instruction, and the implicit VCC
//!   of `v_cndmask_b32` / `v_addc_co_u32` / `v_subb*_co_u32` counts as one;
//! - 64-bit operands start at an even SGPR / TTMP / VGPR (gfx90a+ VGPR tuple
//!   alignment); a 16-bit operand's literal fits 16 bits;
//! - a literal whose value is an inline constant for the operand type prints
//!   as `lit(0x…)` (llvm's spelling that forces the literal encoding);
//!   a 64-bit-float literal `0..=64` is rejected: llvm prints it as `0x…`,
//!   which reassembles as the inline integer;
//! - DPP: src0/src1/vdst are VGPRs, NEG/ABS only on float sources; no VOPC
//!   DPP (absent from the gfx942 XML and llvm, although §13.3.9 lists VOPC
//!   [UNVERIFIED: confirm on MI300X]); no 64-bit-operand DPP (gfx942 DP DPP
//!   only takes `row_newbcast`, which [`DppCtrl`] does not model);
//! - SDWA: no literal; float sources take NEG/ABS, integer sources SEXT, the
//!   FP8/BF8 converts neither; OMOD only on float results. The ISA's SDWA
//!   table reserves DST_SEL/SRC_SEL 0-3, yet llvm (and every compiler) uses
//!   them as `BYTE_0..BYTE_3` [UNVERIFIED: confirm on MI300X].
//!
//! Canonical forms (decode rejects anything else, so `decode(encode(x)) == x`
//! and `encode(decode(bytes)) == bytes`): a VOP1 SDWA carries `src1_sel =
//! Dword` with no src1 modifiers (the src1 bits are zero); a VOPC SDWA carries
//! `dst_sel = Dword`, `dst_unused = Pad`, no clamp/omod, and `sdst: None` for
//! VCC (`Some(VccLo)` is rejected); the FP8/BF8 converts' SDWA forms carry
//! `dst_sel = Dword`, `dst_unused = Pad` (llvm has no syntax for others).
use crate::arch::Arch;
use crate::inst::{Footprint, RegFile};
use crate::operand::{InlineFloat, Src, LITERAL};
use crate::Result;
use std::fmt;

op_table!(Vop1Op {
    Nop,
    MovB32,
    ReadfirstlaneB32,
    CvtF32I32,
    CvtF32U32,
    CvtU32F32,
    CvtI32F32,
    CvtF16F32,
    CvtF32F16,
    CvtRpiI32F32,
    CvtFlrI32F32,
    CvtF32Ubyte0,
    CvtF32Ubyte1,
    CvtF32Ubyte2,
    CvtF32Ubyte3,
    FractF32,
    TruncF32,
    CeilF32,
    RndneF32,
    FloorF32,
    ExpF32,
    LogF32,
    RcpF32,
    RcpIflagF32,
    RsqF32,
    SqrtF32,
    NotB32,
    BfrevB32,
    FfbhU32,
    FfblB32,
    FfbhI32,
    FrexpExpI32F32,
    FrexpMantF32,
    MovB64,
    AccvgprMovB32,
    CvtF32Fp8,
    CvtF32Bf8,
    CvtPkF32Fp8,
    CvtPkF32Bf8,
    SwapB32,
    CvtF64F32,
    CvtF32F64,
});

op_table!(Vop2Op {
    CndmaskB32,
    AddF32,
    SubF32,
    SubrevF32,
    MulF32,
    MulI32I24,
    MulHiI32I24,
    MulU32U24,
    MulHiU32U24,
    MinF32,
    MaxF32,
    MinI32,
    MaxI32,
    MinU32,
    MaxU32,
    LshrrevB32,
    AshrrevI32,
    LshlrevB32,
    AndB32,
    OrB32,
    XorB32,
    XnorB32,
    FmacF32,
    AddCoU32,
    SubCoU32,
    SubrevCoU32,
    AddcCoU32,
    SubbCoU32,
    SubbrevCoU32,
    AddU32,
    SubU32,
    SubrevU32,
    AddF16,
    SubF16,
    MulF16,
    MaxF16,
    MinF16,
    AddU16,
    SubU16,
    LshlrevB16,
    MulLoU16,
});

op_table!(VopcOp {
    CmpClassF32,
    CmpLtF32,
    CmpEqF32,
    CmpLeF32,
    CmpGtF32,
    CmpLgF32,
    CmpGeF32,
    CmpOF32,
    CmpUF32,
    CmpNeqF32,
    CmpLtI32,
    CmpEqI32,
    CmpLeI32,
    CmpGtI32,
    CmpNeI32,
    CmpGeI32,
    CmpLtU32,
    CmpEqU32,
    CmpLeU32,
    CmpGtU32,
    CmpNeU32,
    CmpGeU32,
    CmpxEqU32,
    CmpxNeU32,
    CmpxGtU32,
    CmpxLtU32,
    CmpxGtI32,
    CmpxLtI32,
    CmpxGtF32,
    CmpxLtF32,
    CmpEqU64,
    CmpNeU64,
});

impl Vop1Op {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Dword widths of (vdst, src0); 0 where absent (`v_nop`).
    pub fn widths(self) -> (u8, u8) {
        match self {
            Self::Nop => (0, 0),
            Self::MovB64 => (2, 2),
            Self::CvtPkF32Fp8 | Self::CvtPkF32Bf8 | Self::CvtF64F32 => (2, 1),
            Self::CvtF32F64 => (1, 2),
            _ => (1, 1),
        }
    }
}
impl Vop2Op {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Reads VCC as carry-in / select (`v_cndmask_b32`, `v_addc_co_u32`, ...).
    pub fn reads_vcc(self) -> bool {
        matches!(
            self,
            Self::CndmaskB32 | Self::AddcCoU32 | Self::SubbCoU32 | Self::SubbrevCoU32
        )
    }
    /// Writes VCC as carry-out in the e32 form (`v_add_co_u32`, ...)
    /// (CDNA3 ISA §6.2.2 p.33: "write their result to VCC when used in the
    /// VOP2 form"); the DPP and SDWA forms of gfx9 write VCC as well.
    pub fn writes_vcc(self) -> bool {
        matches!(
            self,
            Self::AddCoU32
                | Self::SubCoU32
                | Self::SubrevCoU32
                | Self::AddcCoU32
                | Self::SubbCoU32
                | Self::SubbrevCoU32
        )
    }
}
impl VopcOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Source dword width (2 for the `_u64` compares).
    pub fn src_dwords(self) -> u8 {
        if matches!(self, Self::CmpEqU64 | Self::CmpNeU64) {
            2
        } else {
            1
        }
    }
    /// `v_cmpx_*`: writes EXEC as well as VCC (e32) or the SDWA SDST
    /// (CDNA3 ISA §6.2.2 p.33: "All V_CMPX instructions write the result of
    /// their comparison ... to both an SGPR (or VCC) and the EXEC mask";
    /// Table 25 p.36 "Write VCC and exec").
    pub fn is_cmpx(self) -> bool {
        matches!(
            self,
            Self::CmpxEqU32
                | Self::CmpxNeU32
                | Self::CmpxGtU32
                | Self::CmpxLtU32
                | Self::CmpxGtI32
                | Self::CmpxLtI32
                | Self::CmpxGtF32
                | Self::CmpxLtF32
        )
    }
}

/// DPP_CTRL (CDNA3 ISA §13.3.9 Table 93 "DPP_CTRL Enumeration" p.544-545).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DppCtrl {
    /// `quad_perm:[a,b,c,d]`, each 0..=3.
    QuadPerm([u8; 4]),
    /// `row_shl:n`, `row_shr:n`, `row_ror:n`, n in 1..=15.
    RowShl(u8),
    RowShr(u8),
    RowRor(u8),
    WaveShl1,
    WaveRol1,
    WaveShr1,
    WaveRor1,
    RowMirror,
    RowHalfMirror,
    RowBcast15,
    RowBcast31,
}

/// The DPP dword: `src0` of the owning instruction is the DPP source VGPR
/// (the src0 field holds 250 = DPP).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dpp {
    pub ctrl: DppCtrl,
    pub row_mask: u8,
    pub bank_mask: u8,
    /// BOUND_CTRL bit 51 (gfx942 llvm prints `bound_ctrl:1` when set and
    /// accepts `bound_ctrl:0` as a legacy spelling of the same bit).
    pub bound_ctrl: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
}

/// SDWA operand / destination select.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaSel {
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Word0,
    Word1,
    Dword,
}
/// SDWA DST_UNUSED.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaUnused {
    Pad,
    Sext,
    Preserve,
}

/// The SDWA dword (src0 field holds 249 = SDWA). `src0`/`src1` of the owning
/// instruction may be SGPRs (the S0/S1 bits). VOPC: `sdst` is the SDWA SD
/// destination (`None` = VCC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sdwa {
    pub dst_sel: SdwaSel,
    pub dst_unused: SdwaUnused,
    pub clamp: bool,
    pub omod: u8,
    pub src0_sel: SdwaSel,
    pub src0_sext: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_sel: SdwaSel,
    pub src1_sext: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
    pub sdst: Option<Src>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VopExt {
    None,
    Dpp(Dpp),
    Sdwa(Sdwa),
}

/// `vdst = op(src0)`. `v_readfirstlane_b32`: `vdst` is an SGPR. `v_nop`:
/// `vdst = Vgpr(0)`, `src0 = Sgpr(0)` (zero fields).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop1 {
    pub op: Vop1Op,
    pub vdst: Src,
    pub src0: Src,
    pub ext: VopExt,
}
/// `vdst = op(src0, src1)`; `src1` is a VGPR except under SDWA (S1 bit).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop2 {
    pub op: Vop2Op,
    pub vdst: Src,
    pub src0: Src,
    pub src1: Src,
    pub ext: VopExt,
}
/// `VCC = op(src0, src1)` (EXEC too for `v_cmpx_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct Vopc {
    pub op: VopcOp,
    pub src0: Src,
    pub src1: Src,
    pub ext: VopExt,
}

/// SRC0 field values that select the second dword (§13.3.1-13.3.3).
const SRC_SDWA: u32 = 249;
const SRC_DPP: u32 = 250;

// ---------------------------------------------------------------------------
// Operand classes. Each table below is what llvm's gfx942 assembler accepts
// and prints, swept per opcode with `llvm-mc -mcpu=gfx942` (ROCm LLVM).

/// How llvm types a source operand: which literal values it folds to inline
/// constants (and therefore prints as `lit(…)` when they are encoded as a
/// literal), and how wide the literal may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LitTy {
    /// Any 32-bit operand: llvm folds 0..=64, -16..=-1 and the f32
    /// inline-float patterns, for integer opcodes as well.
    B32,
    /// 16-bit float: 0..=64, 0xfff0..=0xffff (-16..=-1) and the f16 patterns.
    F16,
    /// 16-bit integer: 0..=64 only.
    I16,
    /// 64-bit integer: 0..=64 only (the literal is extended, §6.2.1 p.32).
    I64,
    /// 64-bit float: the literal is the high dword (§6.2.1 p.32); folds the
    /// f64 inline patterns whose low dword is zero.
    F64,
}

impl LitTy {
    fn inline_equivalent(self, v: u32) -> bool {
        let int = v <= 64;
        match self {
            Self::B32 => {
                int || v >= 0xffff_fff0 || InlineFloat::ALL.iter().any(|f| f.f32_bits() == v)
            }
            Self::F16 => {
                int || (0xfff0..=0xffff).contains(&v)
                    || InlineFloat::ALL
                        .iter()
                        .any(|f| u32::from(f.f16_bits()) == v)
            }
            Self::I16 | Self::I64 => int,
            Self::F64 => InlineFloat::ALL
                .iter()
                .any(|f| f.f64_bits() as u32 == 0 && (f.f64_bits() >> 32) as u32 == v),
        }
    }
    fn check(self, name: &str, v: u32) -> Result<()> {
        match self {
            Self::F16 | Self::I16 if v > 0xffff => Err(format!("{name}: literal {v:#x} does not fit the 16-bit operand")),
            Self::F64 if v <= 64 => Err(format!(
                "{name}: 64-bit-float literal {v:#x} (high dword) has no llvm spelling: llvm prints it as {v:#x}, which assembles as the inline integer {v}"
            )),
            _ => Ok(()),
        }
    }
    fn text(self, v: u32) -> String {
        if self.inline_equivalent(v) {
            format!("lit({v:#x})")
        } else {
            format!("{v:#x}")
        }
    }
}

/// Source modifiers a DPP / SDWA source takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mods {
    /// Float source: NEG/ABS.
    Float,
    /// Integer source: SEXT (SDWA only; DPP has no integer modifier).
    Int,
    /// No source modifier (the FP8/BF8 converts).
    None,
}

impl Vop1Op {
    fn lit_ty(self) -> LitTy {
        match self {
            Self::MovB64 => LitTy::I64,
            Self::CvtF32F64 => LitTy::F64,
            Self::CvtF32F16 => LitTy::F16,
            _ => LitTy::B32,
        }
    }
    fn mods(self) -> Mods {
        match self {
            Self::CvtU32F32
            | Self::CvtI32F32
            | Self::CvtF16F32
            | Self::CvtF32F16
            | Self::CvtRpiI32F32
            | Self::CvtFlrI32F32
            | Self::FractF32
            | Self::TruncF32
            | Self::CeilF32
            | Self::RndneF32
            | Self::FloorF32
            | Self::ExpF32
            | Self::LogF32
            | Self::RcpF32
            | Self::RcpIflagF32
            | Self::RsqF32
            | Self::SqrtF32
            | Self::FrexpExpI32F32
            | Self::FrexpMantF32
            | Self::CvtF64F32
            | Self::CvtF32F64 => Mods::Float,
            Self::CvtF32Fp8 | Self::CvtF32Bf8 | Self::CvtPkF32Fp8 | Self::CvtPkF32Bf8 => Mods::None,
            _ => Mods::Int,
        }
    }
    /// The FP8/BF8 converts: their SDWA form takes only `src0_sel` (a byte for
    /// `v_cvt_f32_*8`, a word for `v_cvt_pk_f32_*8`; CDNA3 ISA §12.8
    /// V_CVT_F32_FP8 / V_CVT_PK_F32_FP8 p.203-204).
    fn fp8(self) -> bool {
        matches!(
            self,
            Self::CvtF32Fp8 | Self::CvtF32Bf8 | Self::CvtPkF32Fp8 | Self::CvtPkF32Bf8
        )
    }
    /// SDWA form exists. `v_accvgpr_mov_b32` is listed with DPP/SDWA in the
    /// machine-readable ISA but llvm has neither; rejected
    /// [UNVERIFIED: confirm on MI300X].
    fn has_sdwa(self) -> bool {
        !matches!(
            self,
            Self::Nop
                | Self::ReadfirstlaneB32
                | Self::MovB64
                | Self::AccvgprMovB32
                | Self::SwapB32
                | Self::CvtF64F32
                | Self::CvtF32F64
        )
    }
    /// DPP form exists (64-bit operands would need `row_newbcast`).
    fn has_dpp(self) -> bool {
        self.has_sdwa() && !matches!(self, Self::CvtPkF32Fp8 | Self::CvtPkF32Bf8)
    }
    /// SDWA OMOD allowed (float result).
    fn has_omod(self) -> bool {
        self.fp8()
            || matches!(
                self,
                Self::CvtF32I32
                    | Self::CvtF32U32
                    | Self::CvtF16F32
                    | Self::CvtF32F16
                    | Self::CvtF32Ubyte0
                    | Self::CvtF32Ubyte1
                    | Self::CvtF32Ubyte2
                    | Self::CvtF32Ubyte3
                    | Self::FractF32
                    | Self::TruncF32
                    | Self::CeilF32
                    | Self::RndneF32
                    | Self::FloorF32
                    | Self::ExpF32
                    | Self::LogF32
                    | Self::RcpF32
                    | Self::RcpIflagF32
                    | Self::RsqF32
                    | Self::SqrtF32
                    | Self::FrexpMantF32
            )
    }
    /// llvm prints the e32 form without `_e32`.
    fn bare(self) -> bool {
        matches!(
            self,
            Self::Nop | Self::ReadfirstlaneB32 | Self::AccvgprMovB32 | Self::SwapB32
        )
    }
}

impl Vop2Op {
    fn f16(self) -> bool {
        matches!(
            self,
            Self::AddF16 | Self::SubF16 | Self::MulF16 | Self::MaxF16 | Self::MinF16
        )
    }
    fn lit_ty(self) -> LitTy {
        if self.f16() {
            LitTy::F16
        } else if matches!(
            self,
            Self::AddU16 | Self::SubU16 | Self::LshlrevB16 | Self::MulLoU16
        ) {
            LitTy::I16
        } else {
            LitTy::B32
        }
    }
    fn mods(self) -> Mods {
        let float = matches!(
            self,
            Self::CndmaskB32
                | Self::AddF32
                | Self::SubF32
                | Self::SubrevF32
                | Self::MulF32
                | Self::MinF32
                | Self::MaxF32
                | Self::FmacF32
        );
        if float || self.f16() {
            Mods::Float
        } else {
            Mods::Int
        }
    }
    fn has_omod(self) -> bool {
        self.f16()
            || matches!(
                self,
                Self::AddF32
                    | Self::SubF32
                    | Self::SubrevF32
                    | Self::MulF32
                    | Self::MinF32
                    | Self::MaxF32
            )
    }
    /// `v_fmac_f32` has no SDWA form (machine-readable ISA and llvm).
    fn has_sdwa(self) -> bool {
        self != Self::FmacF32
    }
}

impl VopcOp {
    fn lit_ty(self) -> LitTy {
        if self.src_dwords() == 2 {
            LitTy::I64
        } else {
            LitTy::B32
        }
    }
    /// (src0, src1) modifier classes; `v_cmp_class_f32`'s src1 is the class mask.
    fn mods(self) -> (Mods, Mods) {
        match self {
            Self::CmpClassF32 => (Mods::Float, Mods::Int),
            Self::CmpLtF32
            | Self::CmpEqF32
            | Self::CmpLeF32
            | Self::CmpGtF32
            | Self::CmpLgF32
            | Self::CmpGeF32
            | Self::CmpOF32
            | Self::CmpUF32
            | Self::CmpNeqF32
            | Self::CmpxGtF32
            | Self::CmpxLtF32 => (Mods::Float, Mods::Float),
            _ => (Mods::Int, Mods::Int),
        }
    }
    /// The 64-bit compares have no SDWA form.
    fn has_sdwa(self) -> bool {
        self.src_dwords() == 1
    }
}

// ---------------------------------------------------------------------------
// Operand checks.

/// Ranges the operand type does not enforce itself.
fn in_range(name: &str, s: Src) -> Result<()> {
    match s {
        Src::Sgpr(n) if n > 101 => Err(format!("{name}: s{n} is beyond s101")),
        Src::Ttmp(n) if n > 15 => Err(format!("{name}: ttmp{n} is beyond ttmp15")),
        Src::Int(i) if !(-16..=64).contains(&i) => {
            Err(format!("{name}: {i} is not an inline integer (-16..=64)"))
        }
        _ => Ok(()),
    }
}

/// `s` may start a `dwords`-wide operand: 64-bit operands start at an even
/// SGPR / TTMP / VGPR / AGPR or a `*_lo` special register, never M0.
fn check_width(name: &str, what: &str, s: Src, dwords: u8) -> Result<()> {
    in_range(name, s)?;
    if dwords < 2 {
        return Ok(());
    }
    let bad = match s {
        Src::Sgpr(n) => n % 2 != 0 || n > 100,
        Src::Ttmp(n) | Src::Vgpr(n) | Src::Agpr(n) => n % 2 != 0,
        Src::VccHi | Src::ExecHi | Src::FlatScratchHi | Src::XnackMaskHi | Src::M0 => true,
        _ => false,
    };
    if bad {
        Err(format!("{name}: {what} {s:?} cannot start a {dwords}-dword operand (even register or a *_lo pair)"))
    } else {
        Ok(())
    }
}

fn vgpr(name: &str, what: &str, s: Src, dwords: u8) -> Result<u32> {
    match s {
        Src::Vgpr(n) => check_width(name, what, s, dwords).map(|()| u32::from(n)),
        other => Err(format!("{name}: {what} must be a VGPR, got {other:?}")),
    }
}

fn agpr(name: &str, what: &str, s: Src) -> Result<u32> {
    match s {
        Src::Agpr(n) => Ok(u32::from(n)),
        other => Err(format!("{name}: {what} must be an AGPR, got {other:?}")),
    }
}

/// The e32 SRC0 field and the literal dword it implies.
fn e32_src0(name: &str, s: Src, dwords: u8, lit: LitTy) -> Result<(u32, Option<u32>)> {
    if s.is_agpr() {
        return Err(format!("{name}: src0 cannot be an AGPR"));
    }
    check_width(name, "src0", s, dwords)?;
    match s {
        Src::Lit(v) => lit.check(name, v).map(|()| (u32::from(LITERAL), Some(v))),
        s => Ok((u32::from(s.code()), None)),
    }
}

/// The operand reads a scalar value over the constant bus (§6.2.1 p.32);
/// inline constants do not.
fn on_bus(s: Src) -> bool {
    !matches!(s, Src::Vgpr(_) | Src::Agpr(_) | Src::Int(_) | Src::Float(_))
}

/// At most one scalar value per instruction (§6.2.1 p.32: one SGPR, usable
/// by several operands; ADDC/SUBB/CNDMASK already read VCC).
fn bus_check(name: &str, src0: Src, src1: Src, implicit_vcc: bool) -> Result<()> {
    let distinct = match (on_bus(src0), on_bus(src1)) {
        (true, true) if src0 != src1 => 2,
        (false, false) => 0,
        _ => 1,
    } + usize::from(implicit_vcc);
    if distinct > 1 {
        Err(format!("{name}: {src0:?}, {src1:?}{} exceed the one scalar value per instruction (CDNA3 ISA §6.2.1 p.32)", if implicit_vcc { " and the implicit VCC" } else { "" }))
    } else {
        Ok(())
    }
}

fn check_mods(name: &str, what: &str, kind: Mods, sext: bool, neg: bool, abs: bool) -> Result<()> {
    let (ok, allowed) = match kind {
        Mods::Float => (!sext, "neg/abs only"),
        Mods::Int => (!neg && !abs, "sext only (SDWA), none under DPP"),
        Mods::None => (!sext && !neg && !abs, "no modifier"),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{name}: {what} takes {allowed}"))
    }
}

/// Check the decoded instruction re-encodes to exactly the input words.
fn canonical(enc: Result<([u32; 2], usize)>, words: &[u32], name: &str) -> Result<usize> {
    let (enc, n) = enc?;
    if words.get(..n) == Some(&enc[..n]) {
        Ok(n)
    } else {
        Err(format!(
            "{name}: non-canonical encoding {:#010x?} (canonical {:#010x?})",
            &words[..n.min(words.len())],
            &enc[..n]
        ))
    }
}

fn second(words: &[u32], name: &str) -> Result<u32> {
    words
        .get(1)
        .copied()
        .ok_or_else(|| format!("{name}: missing second dword"))
}

// ---------------------------------------------------------------------------
// SDWA / SDWAB (§13.3.7-13.3.8 p.542-543).

const SELS: [SdwaSel; 7] = [
    SdwaSel::Byte0,
    SdwaSel::Byte1,
    SdwaSel::Byte2,
    SdwaSel::Byte3,
    SdwaSel::Word0,
    SdwaSel::Word1,
    SdwaSel::Dword,
];
const UNUSED: [SdwaUnused; 3] = [SdwaUnused::Pad, SdwaUnused::Sext, SdwaUnused::Preserve];

impl SdwaSel {
    fn bits(self) -> u32 {
        self as u32
    }
    fn from_bits(b: u32) -> Result<Self> {
        SELS.get(b as usize)
            .copied()
            .ok_or_else(|| format!("reserved SDWA select {b}"))
    }
    fn text(self) -> &'static str {
        [
            "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD",
        ][self as usize]
    }
}

impl SdwaUnused {
    fn bits(self) -> u32 {
        self as u32
    }
    fn from_bits(b: u32) -> Result<Self> {
        UNUSED
            .get(b as usize)
            .copied()
            .ok_or_else(|| format!("reserved SDWA DST_UNUSED {b}"))
    }
    fn text(self) -> &'static str {
        ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"][self as usize]
    }
}

/// An SDWA source: 8-bit field and S bit (VGPR number, or scalar code with S).
fn sdwa_src(name: &str, what: &str, s: Src) -> Result<(u32, bool)> {
    in_range(name, s)?;
    match s {
        Src::Vgpr(n) => Ok((u32::from(n), false)),
        Src::Lit(_) => Err(format!("{name}: SDWA {what} cannot be a literal")),
        Src::Agpr(_) => Err(format!("{name}: SDWA {what} cannot be an AGPR")),
        s => Ok((u32::from(s.code()), true)),
    }
}

fn sdwa_src_decode((field, scalar): (u32, bool)) -> Result<Src> {
    if scalar {
        Src::from_code(field as u16, None)
    } else {
        Ok(Src::Vgpr(field as u8))
    }
}

/// VOP1/VOP2 SDWA bits [15:8]: DST_SEL, DST_U, CLMP, OMOD.
fn sdwa_dst_byte(s: &Sdwa) -> u32 {
    s.dst_sel.bits() | s.dst_unused.bits() << 3 | u32::from(s.clamp) << 5 | u32::from(s.omod) << 6
}

fn sdwa_word(s: &Sdwa, src0: (u32, bool), src1: Option<(u32, bool)>, dst_byte: u32) -> u32 {
    let mut w = src0.0
        | dst_byte << 8
        | s.src0_sel.bits() << 16
        | u32::from(s.src0_sext) << 19
        | u32::from(s.src0_neg) << 20
        | u32::from(s.src0_abs) << 21
        | u32::from(src0.1) << 23;
    if let Some((_, scalar)) = src1 {
        w |= s.src1_sel.bits() << 24
            | u32::from(s.src1_sext) << 27
            | u32::from(s.src1_neg) << 28
            | u32::from(s.src1_abs) << 29
            | u32::from(scalar) << 31;
    }
    w
}

/// Parse an SDWA dword: the fields, the src0 (field, S) and the S1 bit. VOPC
/// reads SDST/SD from [15:8] (SDWAB); VOP1 has no src1 fields.
fn sdwa_parse(w: u32, vopc: bool, vop1: bool) -> Result<(Sdwa, (u32, bool), bool)> {
    let bit = |n: u32| w >> n & 1 != 0;
    let (dst_sel, dst_unused, clamp, omod, sdst) = if vopc {
        let sdst = if bit(15) {
            Some(Src::from_code((w >> 8 & 0x7f) as u16, None)?)
        } else {
            None
        };
        (SdwaSel::Dword, SdwaUnused::Pad, false, 0, sdst)
    } else {
        (
            SdwaSel::from_bits(w >> 8 & 7)?,
            SdwaUnused::from_bits(w >> 11 & 3)?,
            bit(13),
            (w >> 14 & 3) as u8,
            None,
        )
    };
    let (src1_sel, src1_sext, src1_neg, src1_abs) = if vop1 {
        (SdwaSel::Dword, false, false, false)
    } else {
        (SdwaSel::from_bits(w >> 24 & 7)?, bit(27), bit(28), bit(29))
    };
    let s = Sdwa {
        dst_sel,
        dst_unused,
        clamp,
        omod,
        src0_sel: SdwaSel::from_bits(w >> 16 & 7)?,
        src0_sext: bit(19),
        src0_neg: bit(20),
        src0_abs: bit(21),
        src1_sel,
        src1_sext,
        src1_neg,
        src1_abs,
        sdst,
    };
    Ok((s, (w & 0xff, bit(23)), bit(31)))
}

/// VOP1/VOP2 output-modifier checks.
fn sdwa_out_check(name: &str, s: &Sdwa, omod_ok: bool) -> Result<()> {
    if s.omod > 3 {
        return Err(format!("{name}: SDWA omod {} outside 0..=3", s.omod));
    }
    if s.omod != 0 && !omod_ok {
        return Err(format!("{name}: no SDWA output modifier (integer result)"));
    }
    if s.sdst.is_some() {
        return Err(format!("{name}: sdst exists only in the VOPC SDWA form"));
    }
    Ok(())
}

/// The bits of `v` that read the old destination (DST_UNUSED = preserve).
fn preserves(s: &Sdwa) -> bool {
    s.dst_unused == SdwaUnused::Preserve && s.dst_sel != SdwaSel::Dword
}

fn out_mods(s: &Sdwa) -> &'static str {
    match (s.clamp, s.omod) {
        (false, 0) => "",
        (false, 1) => " mul:2",
        (false, 2) => " mul:4",
        (false, _) => " div:2",
        (true, 0) => " clamp",
        (true, 1) => " clamp mul:2",
        (true, 2) => " clamp mul:4",
        (true, _) => " clamp div:2",
    }
}

// ---------------------------------------------------------------------------
// DPP (§13.3.9 p.544-545).

impl DppCtrl {
    /// The 9-bit DPP_CTRL value. `wave_*` and `row_bcast` exist only where
    /// the variant's caps say so (gfx9 DPP).
    fn bits(self, name: &str, arch: Arch) -> Result<u32> {
        let row = |base: u32, n: u8, what: &str| {
            if (1..=15).contains(&n) {
                Ok(base | u32::from(n))
            } else {
                Err(format!("{name}: {what}:{n} outside 1..=15"))
            }
        };
        let wave = |b: u32| {
            if arch.caps().dpp_wave_ops {
                Ok(b)
            } else {
                Err(format!("{name}: DPP {self:?} does not exist on {arch}"))
            }
        };
        match self {
            Self::QuadPerm(p) => {
                if p.iter().any(|&x| x > 3) {
                    return Err(format!("{name}: quad_perm {p:?} element > 3"));
                }
                Ok(u32::from(p[0])
                    | u32::from(p[1]) << 2
                    | u32::from(p[2]) << 4
                    | u32::from(p[3]) << 6)
            }
            Self::RowShl(n) => row(0x100, n, "row_shl"),
            Self::RowShr(n) => row(0x110, n, "row_shr"),
            Self::RowRor(n) => row(0x120, n, "row_ror"),
            Self::WaveShl1 => wave(0x130),
            Self::WaveRol1 => wave(0x134),
            Self::WaveShr1 => wave(0x138),
            Self::WaveRor1 => wave(0x13c),
            Self::RowMirror => Ok(0x140),
            Self::RowHalfMirror => Ok(0x141),
            Self::RowBcast15 => wave(0x142),
            Self::RowBcast31 => wave(0x143),
        }
    }
    fn from_bits(b: u32) -> Result<Self> {
        Ok(match b {
            0..=0xff => Self::QuadPerm([
                (b & 3) as u8,
                (b >> 2 & 3) as u8,
                (b >> 4 & 3) as u8,
                (b >> 6 & 3) as u8,
            ]),
            0x101..=0x10f => Self::RowShl((b & 0xf) as u8),
            0x111..=0x11f => Self::RowShr((b & 0xf) as u8),
            0x121..=0x12f => Self::RowRor((b & 0xf) as u8),
            0x130 => Self::WaveShl1,
            0x134 => Self::WaveRol1,
            0x138 => Self::WaveShr1,
            0x13c => Self::WaveRor1,
            0x140 => Self::RowMirror,
            0x141 => Self::RowHalfMirror,
            0x142 => Self::RowBcast15,
            0x143 => Self::RowBcast31,
            0x150..=0x15f => {
                return Err(format!(
                    "DPP_CTRL {b:#x} (row_newbcast, 64-bit DPP) is not modelled"
                ))
            }
            _ => return Err(format!("reserved DPP_CTRL {b:#x}")),
        })
    }
    fn text(self) -> String {
        match self {
            Self::QuadPerm(p) => format!("quad_perm:[{},{},{},{}]", p[0], p[1], p[2], p[3]),
            Self::RowShl(n) => format!("row_shl:{n}"),
            Self::RowShr(n) => format!("row_shr:{n}"),
            Self::RowRor(n) => format!("row_ror:{n}"),
            Self::WaveShl1 => "wave_shl:1".into(),
            Self::WaveRol1 => "wave_rol:1".into(),
            Self::WaveShr1 => "wave_shr:1".into(),
            Self::WaveRor1 => "wave_ror:1".into(),
            Self::RowMirror => "row_mirror".into(),
            Self::RowHalfMirror => "row_half_mirror".into(),
            Self::RowBcast15 => "row_bcast:15".into(),
            Self::RowBcast31 => "row_bcast:31".into(),
        }
    }
}

fn dpp_word(name: &str, arch: Arch, d: &Dpp, src0: u32) -> Result<u32> {
    if d.row_mask > 0xf || d.bank_mask > 0xf {
        return Err(format!(
            "{name}: DPP row_mask {:#x} / bank_mask {:#x} exceed 4 bits",
            d.row_mask, d.bank_mask
        ));
    }
    Ok(src0
        | d.ctrl.bits(name, arch)? << 8
        | u32::from(d.bound_ctrl) << 19
        | u32::from(d.src0_neg) << 20
        | u32::from(d.src0_abs) << 21
        | u32::from(d.src1_neg) << 22
        | u32::from(d.src1_abs) << 23
        | u32::from(d.bank_mask) << 24
        | u32::from(d.row_mask) << 28)
}

/// Parse a DPP dword into its fields and the src0 VGPR number. Reserved bits
/// [50:49] are left to the canonical re-encode check.
fn dpp_parse(w: u32) -> Result<(Dpp, u8)> {
    let bit = |n: u32| w >> n & 1 != 0;
    let d = Dpp {
        ctrl: DppCtrl::from_bits(w >> 8 & 0x1ff)?,
        row_mask: (w >> 28) as u8,
        bank_mask: (w >> 24 & 0xf) as u8,
        bound_ctrl: bit(19),
        src0_neg: bit(20),
        src0_abs: bit(21),
        src1_neg: bit(22),
        src1_abs: bit(23),
    };
    Ok((d, (w & 0xff) as u8))
}

fn dpp_text(d: &Dpp) -> String {
    format!(
        " {} row_mask:{:#x} bank_mask:{:#x}{}",
        d.ctrl.text(),
        d.row_mask,
        d.bank_mask,
        if d.bound_ctrl { " bound_ctrl:1" } else { "" }
    )
}

// ---------------------------------------------------------------------------
// Operand text.

/// An e32 source: `lit(…)` for a literal llvm would otherwise fold.
fn e32_text(s: Src, dwords: u8, lit: LitTy) -> String {
    match s {
        Src::Lit(v) => lit.text(v),
        s => s.fmt(dwords),
    }
}

/// A DPP/SDWA source with its modifiers, as llvm prints it: `sext(x)`,
/// `|x|`, `-x`, `-|x|`, and `neg(c)` for a negated constant without ABS.
fn modded(s: Src, sext: bool, neg: bool, abs: bool) -> String {
    let t = s.fmt(1);
    if sext {
        return format!("sext({t})");
    }
    let t = if abs { format!("|{t}|") } else { t };
    if !neg {
        t
    } else if !abs && matches!(s, Src::Int(_) | Src::Float(_) | Src::Lit(_)) {
        format!("neg({t})")
    } else {
        format!("-{t}")
    }
}

// ---------------------------------------------------------------------------
// VOP1 (§13.3.2 p.524).

impl Vop1 {
    /// The instruction's dwords (`[word, second]`, count) after every legality check.
    fn words(&self, arch: Arch) -> Result<([u32; 2], usize)> {
        let op = self.op;
        let name = op.name();
        let opcode = op
            .opcode(arch)
            .ok_or_else(|| format!("{name} does not exist on {arch}"))?;
        let base =
            |vdst: u32, src0: u32| 0b011_1111 << 25 | vdst << 17 | u32::from(opcode) << 9 | src0;
        let plain = || {
            if self.ext == VopExt::None {
                Ok(())
            } else {
                Err(format!("{name} has no DPP/SDWA form on {arch}"))
            }
        };
        match op {
            Vop1Op::Nop => {
                plain()?;
                if self.vdst != Src::Vgpr(0) || self.src0 != Src::Sgpr(0) {
                    return Err("v_nop takes no operands (vdst = Vgpr(0), src0 = Sgpr(0))".into());
                }
                return Ok(([base(0, 0), 0], 1));
            }
            // VDST is an SGPR code; llvm's SReg_32_XM0 class excludes M0
            // (the ISA operand type is OPR_SREG_NOVCC, yet llvm accepts VCC)
            // [UNVERIFIED: confirm on MI300X]. SRC0 is a VGPR (CDNA3 ISA
            // §12.8 V_READFIRSTLANE_B32 p.179).
            Vop1Op::ReadfirstlaneB32 => {
                plain()?;
                in_range(name, self.vdst)?;
                if !self.vdst.is_sreg() || self.vdst == Src::M0 {
                    return Err(format!(
                        "{name}: vdst must be a scalar register other than M0, got {:?}",
                        self.vdst
                    ));
                }
                let src0 = vgpr(name, "src0", self.src0, 1)?;
                return Ok(([base(u32::from(self.vdst.code()), 256 + src0), 0], 1));
            }
            // AccVGPRs sit in the VGPR fields; the opcode implies the AGPR
            // file (CDNA3 ISA §12.8 V_ACCVGPR_MOV_B32 p.203; llvm encodes
            // `a1, a2` as VDST 1, SRC0 258).
            Vop1Op::AccvgprMovB32 => {
                plain()?;
                let (vdst, src0) = (
                    agpr(name, "vdst", self.vdst)?,
                    agpr(name, "src0", self.src0)?,
                );
                return Ok(([base(vdst, 256 + src0), 0], 1));
            }
            Vop1Op::SwapB32 => {
                plain()?;
                let (vdst, src0) = (
                    vgpr(name, "vdst", self.vdst, 1)?,
                    vgpr(name, "src0", self.src0, 1)?,
                );
                return Ok(([base(vdst, 256 + src0), 0], 1));
            }
            _ => {}
        }
        let (dw, sw) = op.widths();
        let vdst = vgpr(name, "vdst", self.vdst, dw)?;
        match self.ext {
            VopExt::None => {
                let (src0, lit) = e32_src0(name, self.src0, sw, op.lit_ty())?;
                Ok((
                    [base(vdst, src0), lit.unwrap_or(0)],
                    1 + usize::from(lit.is_some()),
                ))
            }
            VopExt::Dpp(d) => {
                if !op.has_dpp() {
                    return Err(format!("{name} has no DPP form on {arch}"));
                }
                let src0 = vgpr(name, "DPP src0", self.src0, 1)?;
                check_mods(name, "src0", op.mods(), false, d.src0_neg, d.src0_abs)?;
                if d.src1_neg || d.src1_abs {
                    return Err(format!("{name}: VOP1 has no src1 modifiers"));
                }
                Ok(([base(vdst, SRC_DPP), dpp_word(name, arch, &d, src0)?], 2))
            }
            VopExt::Sdwa(s) => {
                if !op.has_sdwa() {
                    return Err(format!("{name} has no SDWA form on {arch}"));
                }
                let src0 = sdwa_src(name, "src0", self.src0)?;
                check_mods(name, "src0", op.mods(), s.src0_sext, s.src0_neg, s.src0_abs)?;
                if (s.src1_sel, s.src1_sext, s.src1_neg, s.src1_abs)
                    != (SdwaSel::Dword, false, false, false)
                {
                    return Err(format!(
                        "{name}: VOP1 SDWA has no src1 (src1_sel must be Dword, no src1 modifiers)"
                    ));
                }
                if op.fp8() && (s.dst_sel, s.dst_unused) != (SdwaSel::Dword, SdwaUnused::Pad) {
                    return Err(format!("{name}: the FP8/BF8 convert SDWA form selects src0 only (dst_sel Dword, dst_unused Pad)"));
                }
                sdwa_out_check(name, &s, op.has_omod())?;
                Ok((
                    [
                        base(vdst, SRC_SDWA),
                        sdwa_word(&s, src0, None, sdwa_dst_byte(&s)),
                    ],
                    2,
                ))
            }
        }
    }

    /// Append the instruction's dwords (the literal / DPP / SDWA dword included).
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let (w, n) = self.words(arch)?;
        out.extend_from_slice(&w[..n]);
        Ok(())
    }

    /// Decode the VOP1 instruction at `words[0]`; non-canonical encodings
    /// (reserved fields set, forms the encoder rejects) are errors.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let w = *words
            .first()
            .ok_or("VOP1 decode past the end of the code")?;
        if w >> 25 != 0b011_1111 {
            return Err(format!("{w:#010x} is not a VOP1 encoding"));
        }
        let opcode = (w >> 9 & 0xff) as u8;
        let op = Vop1Op::from_opcode(arch, opcode).ok_or_else(|| {
            format!("VOP1 opcode {opcode} is outside the PeaceMaker CDNA table on {arch}")
        })?;
        let name = op.name();
        let (vdst, src0) = (w >> 17 & 0xff, w & 0x1ff);
        let inst = match op {
            Vop1Op::Nop => Vop1 {
                op,
                vdst: Src::Vgpr(0),
                src0: Src::Sgpr(0),
                ext: VopExt::None,
            },
            Vop1Op::ReadfirstlaneB32 => Vop1 {
                op,
                vdst: Src::from_code(vdst as u16, None)?,
                src0: Src::from_code(src0 as u16, None)?,
                ext: VopExt::None,
            },
            Vop1Op::AccvgprMovB32 => {
                if src0 < 256 {
                    return Err(format!("{name}: src0 code {src0} is not an AccVGPR"));
                }
                Vop1 {
                    op,
                    vdst: Src::Agpr(vdst as u8),
                    src0: Src::Agpr((src0 - 256) as u8),
                    ext: VopExt::None,
                }
            }
            _ => {
                let vdst = Src::Vgpr(vdst as u8);
                match src0 {
                    SRC_SDWA => {
                        let (s, s0, _) = sdwa_parse(second(words, name)?, false, true)?;
                        Vop1 {
                            op,
                            vdst,
                            src0: sdwa_src_decode(s0)?,
                            ext: VopExt::Sdwa(s),
                        }
                    }
                    SRC_DPP => {
                        let (d, s0) = dpp_parse(second(words, name)?)?;
                        Vop1 {
                            op,
                            vdst,
                            src0: Src::Vgpr(s0),
                            ext: VopExt::Dpp(d),
                        }
                    }
                    c => {
                        let lit = if c == u32::from(LITERAL) {
                            Some(second(words, name)?)
                        } else {
                            None
                        };
                        Vop1 {
                            op,
                            vdst,
                            src0: Src::from_code(c as u16, lit)?,
                            ext: VopExt::None,
                        }
                    }
                }
            }
        };
        let n = canonical(inst.words(arch), words, name)?;
        Ok((inst, n))
    }

    /// EXEC, src0, vdst; `v_swap_b32` reads and writes both registers; an
    /// SDWA `UNUSED_PRESERVE` sub-dword write also reads vdst. `v_nop`
    /// touches nothing.
    pub fn footprint(&self) -> Footprint {
        let mut f = Footprint::default();
        if self.op == Vop1Op::Nop {
            return f;
        }
        let (dw, sw) = self.op.widths();
        f.read_file(RegFile::Exec, 0, 2);
        f.read(self.src0, sw);
        f.write(self.vdst, dw);
        if self.op == Vop1Op::SwapB32 {
            f.read(self.vdst, 1);
            f.write(self.src0, 1);
        }
        if let VopExt::Sdwa(s) = &self.ext {
            if preserves(s) {
                f.read(self.vdst, dw)
            }
        }
        f
    }
}

impl fmt::Display for Vop1 {
    /// `llvm-objdump --mcpu=gfx942` text: `_e32` / `_dpp` / `_sdwa` suffix as llvm prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = self.op;
        let name = op.name();
        if op == Vop1Op::Nop {
            return f.write_str(name);
        }
        let (dw, sw) = op.widths();
        let vdst = self.vdst.fmt(dw);
        match &self.ext {
            VopExt::None => {
                let suffix = if op.bare() { "" } else { "_e32" };
                write!(
                    f,
                    "{name}{suffix} {vdst}, {}",
                    e32_text(self.src0, sw, op.lit_ty())
                )
            }
            VopExt::Dpp(d) => write!(
                f,
                "{name}_dpp {vdst}, {}{}",
                modded(self.src0, false, d.src0_neg, d.src0_abs),
                dpp_text(d)
            ),
            VopExt::Sdwa(s) => {
                write!(
                    f,
                    "{name}_sdwa {vdst}, {}{}",
                    modded(self.src0, s.src0_sext, s.src0_neg, s.src0_abs),
                    out_mods(s)
                )?;
                if !op.fp8() {
                    write!(
                        f,
                        " dst_sel:{} dst_unused:{}",
                        s.dst_sel.text(),
                        s.dst_unused.text()
                    )?
                }
                write!(f, " src0_sel:{}", s.src0_sel.text())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VOP2 (§13.3.1 p.522).

impl Vop2 {
    fn words(&self, arch: Arch) -> Result<([u32; 2], usize)> {
        let op = self.op;
        let name = op.name();
        let opcode = op
            .opcode(arch)
            .ok_or_else(|| format!("{name} does not exist on {arch}"))?;
        // Opcodes 62/63 would read as VOPC/VOP1 (`inst::classify`).
        if opcode >= 0b11_1110 {
            return Err(format!(
                "{name}: VOP2 opcode {opcode} collides with the VOPC/VOP1 encodings"
            ));
        }
        let base = |vdst: u32, src1: u32, src0: u32| {
            u32::from(opcode) << 25 | vdst << 17 | src1 << 9 | src0
        };
        let vdst = vgpr(name, "vdst", self.vdst, 1)?;
        match self.ext {
            VopExt::None => {
                let src1 = vgpr(name, "src1 (VSRC1)", self.src1, 1)?;
                let (src0, lit) = e32_src0(name, self.src0, 1, op.lit_ty())?;
                if op.reads_vcc() && on_bus(self.src0) {
                    return Err(format!("{name}: src0 {:?} and the implicit VCC exceed the one scalar value per instruction (CDNA3 ISA §6.2.1 p.32)", self.src0));
                }
                Ok((
                    [base(vdst, src1, src0), lit.unwrap_or(0)],
                    1 + usize::from(lit.is_some()),
                ))
            }
            VopExt::Dpp(d) => {
                let (src0, src1) = (
                    vgpr(name, "DPP src0", self.src0, 1)?,
                    vgpr(name, "src1", self.src1, 1)?,
                );
                check_mods(name, "src0", op.mods(), false, d.src0_neg, d.src0_abs)?;
                check_mods(name, "src1", op.mods(), false, d.src1_neg, d.src1_abs)?;
                Ok((
                    [base(vdst, src1, SRC_DPP), dpp_word(name, arch, &d, src0)?],
                    2,
                ))
            }
            VopExt::Sdwa(s) => {
                if !op.has_sdwa() {
                    return Err(format!("{name} has no SDWA form on {arch}"));
                }
                let (src0, src1) = (
                    sdwa_src(name, "src0", self.src0)?,
                    sdwa_src(name, "src1", self.src1)?,
                );
                check_mods(name, "src0", op.mods(), s.src0_sext, s.src0_neg, s.src0_abs)?;
                check_mods(name, "src1", op.mods(), s.src1_sext, s.src1_neg, s.src1_abs)?;
                bus_check(name, self.src0, self.src1, op.reads_vcc())?;
                sdwa_out_check(name, &s, op.has_omod())?;
                Ok((
                    [
                        base(vdst, src1.0, SRC_SDWA),
                        sdwa_word(&s, src0, Some(src1), sdwa_dst_byte(&s)),
                    ],
                    2,
                ))
            }
        }
    }

    /// Append the instruction's dwords (the literal / DPP / SDWA dword included).
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let (w, n) = self.words(arch)?;
        out.extend_from_slice(&w[..n]);
        Ok(())
    }

    /// Decode the VOP2 instruction at `words[0]`; non-canonical encodings are errors.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let w = *words
            .first()
            .ok_or("VOP2 decode past the end of the code")?;
        if w >> 31 != 0 || w >> 25 >= 0b011_1110 {
            return Err(format!("{w:#010x} is not a VOP2 encoding"));
        }
        let opcode = (w >> 25) as u8;
        let op = Vop2Op::from_opcode(arch, opcode).ok_or_else(|| {
            format!("VOP2 opcode {opcode} is outside the PeaceMaker CDNA table on {arch}")
        })?;
        let name = op.name();
        let (vdst, src1, src0) = (Src::Vgpr((w >> 17 & 0xff) as u8), w >> 9 & 0xff, w & 0x1ff);
        let inst = match src0 {
            SRC_SDWA => {
                let (s, s0, s1) = sdwa_parse(second(words, name)?, false, false)?;
                Vop2 {
                    op,
                    vdst,
                    src0: sdwa_src_decode(s0)?,
                    src1: sdwa_src_decode((src1, s1))?,
                    ext: VopExt::Sdwa(s),
                }
            }
            SRC_DPP => {
                let (d, s0) = dpp_parse(second(words, name)?)?;
                Vop2 {
                    op,
                    vdst,
                    src0: Src::Vgpr(s0),
                    src1: Src::Vgpr(src1 as u8),
                    ext: VopExt::Dpp(d),
                }
            }
            c => {
                let lit = if c == u32::from(LITERAL) {
                    Some(second(words, name)?)
                } else {
                    None
                };
                Vop2 {
                    op,
                    vdst,
                    src0: Src::from_code(c as u16, lit)?,
                    src1: Src::Vgpr(src1 as u8),
                    ext: VopExt::None,
                }
            }
        };
        let n = canonical(inst.words(arch), words, name)?;
        Ok((inst, n))
    }

    /// EXEC, both sources, vdst; VCC per [`Vop2Op::reads_vcc`] /
    /// [`Vop2Op::writes_vcc`] (CDNA3 ISA §6.2.1 p.32, §6.2.2 p.33);
    /// `v_fmac_f32` reads vdst as the addend; an SDWA `UNUSED_PRESERVE`
    /// sub-dword write reads vdst.
    pub fn footprint(&self) -> Footprint {
        let mut f = Footprint::default();
        f.read_file(RegFile::Exec, 0, 2);
        f.read(self.src0, 1);
        f.read(self.src1, 1);
        f.write(self.vdst, 1);
        if self.op == Vop2Op::FmacF32 {
            f.read(self.vdst, 1)
        }
        if let VopExt::Sdwa(s) = &self.ext {
            if preserves(s) {
                f.read(self.vdst, 1)
            }
        }
        if self.op.reads_vcc() {
            f.read_file(RegFile::Vcc, 0, 2)
        }
        if self.op.writes_vcc() {
            f.write_file(RegFile::Vcc, 0, 2)
        }
        f
    }
}

impl fmt::Display for Vop2 {
    /// `llvm-objdump --mcpu=gfx942` text: `_e32` / `_dpp` / `_sdwa` suffix as llvm prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = self.op;
        let (src0, src1, suffix, tail) = match &self.ext {
            VopExt::None => (
                e32_text(self.src0, 1, op.lit_ty()),
                self.src1.fmt(1),
                "_e32",
                String::new(),
            ),
            VopExt::Dpp(d) => (
                modded(self.src0, false, d.src0_neg, d.src0_abs),
                modded(self.src1, false, d.src1_neg, d.src1_abs),
                "_dpp",
                dpp_text(d),
            ),
            VopExt::Sdwa(s) => (
                modded(self.src0, s.src0_sext, s.src0_neg, s.src0_abs),
                modded(self.src1, s.src1_sext, s.src1_neg, s.src1_abs),
                "_sdwa",
                format!(
                    "{} dst_sel:{} dst_unused:{} src0_sel:{} src1_sel:{}",
                    out_mods(s),
                    s.dst_sel.text(),
                    s.dst_unused.text(),
                    s.src0_sel.text(),
                    s.src1_sel.text()
                ),
            ),
        };
        let vdst = self.vdst.fmt(1);
        write!(f, "{}{suffix} {vdst}, ", op.name())?;
        if op == Vop2Op::CndmaskB32 {
            write!(f, "{src0}, {src1}, vcc")?;
        } else if op.reads_vcc() {
            write!(f, "vcc, {src0}, {src1}, vcc")?;
        } else if op.writes_vcc() {
            write!(f, "vcc, {src0}, {src1}")?;
        } else {
            write!(f, "{src0}, {src1}")?;
        }
        f.write_str(&tail)
    }
}

// ---------------------------------------------------------------------------
// VOPC (§13.3.3 p.526; SDWAB §13.3.8 p.543).

impl Vopc {
    fn words(&self, arch: Arch) -> Result<([u32; 2], usize)> {
        let op = self.op;
        let name = op.name();
        let opcode = op
            .opcode(arch)
            .ok_or_else(|| format!("{name} does not exist on {arch}"))?;
        let base =
            |src1: u32, src0: u32| 0b011_1110 << 25 | u32::from(opcode) << 17 | src1 << 9 | src0;
        let dw = op.src_dwords();
        match self.ext {
            VopExt::None => {
                let src1 = vgpr(name, "src1 (VSRC1)", self.src1, dw)?;
                let (src0, lit) = e32_src0(name, self.src0, dw, op.lit_ty())?;
                Ok((
                    [base(src1, src0), lit.unwrap_or(0)],
                    1 + usize::from(lit.is_some()),
                ))
            }
            // §13.3.9 p.544 lists DPP after VOPC, but the CDNA3 machine-readable
            // ISA has no VOPC DPP encoding and llvm rejects it for gfx942.
            // [UNVERIFIED: confirm on MI300X]
            VopExt::Dpp(_) => Err(format!("{name}: VOPC has no DPP form on {arch}")),
            VopExt::Sdwa(s) => {
                if !op.has_sdwa() {
                    return Err(format!("{name} has no SDWA form on {arch}"));
                }
                let (m0, m1) = op.mods();
                let (src0, src1) = (
                    sdwa_src(name, "src0", self.src0)?,
                    sdwa_src(name, "src1", self.src1)?,
                );
                check_mods(name, "src0", m0, s.src0_sext, s.src0_neg, s.src0_abs)?;
                check_mods(name, "src1", m1, s.src1_sext, s.src1_neg, s.src1_abs)?;
                bus_check(name, self.src0, self.src1, false)?;
                if (s.dst_sel, s.dst_unused, s.clamp, s.omod)
                    != (SdwaSel::Dword, SdwaUnused::Pad, false, 0)
                {
                    return Err(format!(
                        "{name}: SDWAB has no dst_sel/dst_unused/clamp/omod (Dword, Pad, false, 0)"
                    ));
                }
                let sd = match s.sdst {
                    None => 0,
                    Some(Src::VccLo) => {
                        return Err(format!("{name}: SDWA sdst VCC is `None` (SD = 0)"))
                    }
                    Some(
                        d @ (Src::Sgpr(_)
                        | Src::Ttmp(_)
                        | Src::ExecLo
                        | Src::FlatScratchLo
                        | Src::XnackMaskLo),
                    ) => {
                        check_width(name, "sdst", d, 2)?;
                        0x80 | u32::from(d.code())
                    }
                    Some(d) => {
                        return Err(format!(
                            "{name}: SDWA sdst must be a 64-bit scalar register pair, got {d:?}"
                        ))
                    }
                };
                Ok((
                    [base(src1.0, SRC_SDWA), sdwa_word(&s, src0, Some(src1), sd)],
                    2,
                ))
            }
        }
    }

    /// Append the instruction's dwords (the literal / SDWA dword included).
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let (w, n) = self.words(arch)?;
        out.extend_from_slice(&w[..n]);
        Ok(())
    }

    /// Decode the VOPC instruction at `words[0]`; non-canonical encodings are errors.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let w = *words
            .first()
            .ok_or("VOPC decode past the end of the code")?;
        if w >> 25 != 0b011_1110 {
            return Err(format!("{w:#010x} is not a VOPC encoding"));
        }
        let opcode = (w >> 17 & 0xff) as u8;
        let op = VopcOp::from_opcode(arch, opcode).ok_or_else(|| {
            format!("VOPC opcode {opcode} is outside the PeaceMaker CDNA table on {arch}")
        })?;
        let name = op.name();
        let (src1, src0) = (w >> 9 & 0xff, w & 0x1ff);
        let inst = match src0 {
            SRC_SDWA => {
                let (s, s0, s1) = sdwa_parse(second(words, name)?, true, false)?;
                Vopc {
                    op,
                    src0: sdwa_src_decode(s0)?,
                    src1: sdwa_src_decode((src1, s1))?,
                    ext: VopExt::Sdwa(s),
                }
            }
            SRC_DPP => return Err(format!("{name}: VOPC DPP is not a gfx942 encoding")),
            c => {
                let lit = if c == u32::from(LITERAL) {
                    Some(second(words, name)?)
                } else {
                    None
                };
                Vopc {
                    op,
                    src0: Src::from_code(c as u16, lit)?,
                    src1: Src::Vgpr(src1 as u8),
                    ext: VopExt::None,
                }
            }
        };
        let n = canonical(inst.words(arch), words, name)?;
        Ok((inst, n))
    }

    /// EXEC and both sources read; VCC (e32) or the SDWA SDST written;
    /// `v_cmpx_*` also writes EXEC (CDNA3 ISA §6.2.2 p.33).
    pub fn footprint(&self) -> Footprint {
        let mut f = Footprint::default();
        let dw = self.op.src_dwords();
        f.read_file(RegFile::Exec, 0, 2);
        f.read(self.src0, dw);
        f.read(self.src1, dw);
        match &self.ext {
            VopExt::Sdwa(Sdwa { sdst: Some(d), .. }) => f.write(*d, 2),
            _ => f.write_file(RegFile::Vcc, 0, 2),
        }
        if self.op.is_cmpx() {
            f.write_file(RegFile::Exec, 0, 2)
        }
        f
    }
}

impl fmt::Display for Vopc {
    /// `llvm-objdump --mcpu=gfx942` text: `_e32` / `_sdwa` suffix as llvm prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = self.op;
        let name = op.name();
        let dw = op.src_dwords();
        match &self.ext {
            VopExt::None => write!(
                f,
                "{name}_e32 vcc, {}, {}",
                e32_text(self.src0, dw, op.lit_ty()),
                self.src1.fmt(dw)
            ),
            VopExt::Dpp(d) => write!(
                f,
                "{name}_dpp vcc, {}, {}{}",
                modded(self.src0, false, d.src0_neg, d.src0_abs),
                modded(self.src1, false, d.src1_neg, d.src1_abs),
                dpp_text(d)
            ),
            VopExt::Sdwa(s) => write!(
                f,
                "{name}_sdwa {}, {}, {} src0_sel:{} src1_sel:{}",
                s.sdst.map_or_else(|| "vcc".to_string(), |d| d.fmt(2)),
                modded(self.src0, s.src0_sext, s.src0_neg, s.src0_abs),
                modded(self.src1, s.src1_sext, s.src1_neg, s.src1_abs),
                s.src0_sel.text(),
                s.src1_sel.text()
            ),
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP1); proven by the
// llvm-mc round-trip tests.
impl Vop1Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::Nop => 0,
                Self::MovB32 => 1,
                Self::ReadfirstlaneB32 => 2,
                Self::CvtF32I32 => 5,
                Self::CvtF32U32 => 6,
                Self::CvtU32F32 => 7,
                Self::CvtI32F32 => 8,
                Self::CvtF16F32 => 10,
                Self::CvtF32F16 => 11,
                Self::CvtRpiI32F32 => 12,
                Self::CvtFlrI32F32 => 13,
                Self::CvtF32Ubyte0 => 17,
                Self::CvtF32Ubyte1 => 18,
                Self::CvtF32Ubyte2 => 19,
                Self::CvtF32Ubyte3 => 20,
                Self::FractF32 => 27,
                Self::TruncF32 => 28,
                Self::CeilF32 => 29,
                Self::RndneF32 => 30,
                Self::FloorF32 => 31,
                Self::ExpF32 => 32,
                Self::LogF32 => 33,
                Self::RcpF32 => 34,
                Self::RcpIflagF32 => 35,
                Self::RsqF32 => 36,
                Self::SqrtF32 => 39,
                Self::NotB32 => 43,
                Self::BfrevB32 => 44,
                Self::FfbhU32 => 45,
                Self::FfblB32 => 46,
                Self::FfbhI32 => 47,
                Self::FrexpExpI32F32 => 51,
                Self::FrexpMantF32 => 52,
                Self::MovB64 => 56,
                Self::AccvgprMovB32 => 82,
                Self::CvtF32Fp8 => 84,
                Self::CvtF32Bf8 => 85,
                Self::CvtPkF32Fp8 => 86,
                Self::CvtPkF32Bf8 => 87,
                Self::SwapB32 => 81,
                Self::CvtF64F32 => 16,
                Self::CvtF32F64 => 15,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::Nop => "v_nop",
            Self::MovB32 => "v_mov_b32",
            Self::ReadfirstlaneB32 => "v_readfirstlane_b32",
            Self::CvtF32I32 => "v_cvt_f32_i32",
            Self::CvtF32U32 => "v_cvt_f32_u32",
            Self::CvtU32F32 => "v_cvt_u32_f32",
            Self::CvtI32F32 => "v_cvt_i32_f32",
            Self::CvtF16F32 => "v_cvt_f16_f32",
            Self::CvtF32F16 => "v_cvt_f32_f16",
            Self::CvtRpiI32F32 => "v_cvt_rpi_i32_f32",
            Self::CvtFlrI32F32 => "v_cvt_flr_i32_f32",
            Self::CvtF32Ubyte0 => "v_cvt_f32_ubyte0",
            Self::CvtF32Ubyte1 => "v_cvt_f32_ubyte1",
            Self::CvtF32Ubyte2 => "v_cvt_f32_ubyte2",
            Self::CvtF32Ubyte3 => "v_cvt_f32_ubyte3",
            Self::FractF32 => "v_fract_f32",
            Self::TruncF32 => "v_trunc_f32",
            Self::CeilF32 => "v_ceil_f32",
            Self::RndneF32 => "v_rndne_f32",
            Self::FloorF32 => "v_floor_f32",
            Self::ExpF32 => "v_exp_f32",
            Self::LogF32 => "v_log_f32",
            Self::RcpF32 => "v_rcp_f32",
            Self::RcpIflagF32 => "v_rcp_iflag_f32",
            Self::RsqF32 => "v_rsq_f32",
            Self::SqrtF32 => "v_sqrt_f32",
            Self::NotB32 => "v_not_b32",
            Self::BfrevB32 => "v_bfrev_b32",
            Self::FfbhU32 => "v_ffbh_u32",
            Self::FfblB32 => "v_ffbl_b32",
            Self::FfbhI32 => "v_ffbh_i32",
            Self::FrexpExpI32F32 => "v_frexp_exp_i32_f32",
            Self::FrexpMantF32 => "v_frexp_mant_f32",
            Self::MovB64 => "v_mov_b64",
            Self::AccvgprMovB32 => "v_accvgpr_mov_b32",
            Self::CvtF32Fp8 => "v_cvt_f32_fp8",
            Self::CvtF32Bf8 => "v_cvt_f32_bf8",
            Self::CvtPkF32Fp8 => "v_cvt_pk_f32_fp8",
            Self::CvtPkF32Bf8 => "v_cvt_pk_f32_bf8",
            Self::SwapB32 => "v_swap_b32",
            Self::CvtF64F32 => "v_cvt_f64_f32",
            Self::CvtF32F64 => "v_cvt_f32_f64",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP2); proven by the
// llvm-mc round-trip tests.
impl Vop2Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CndmaskB32 => 0,
                Self::AddF32 => 1,
                Self::SubF32 => 2,
                Self::SubrevF32 => 3,
                Self::MulF32 => 5,
                Self::MulI32I24 => 6,
                Self::MulHiI32I24 => 7,
                Self::MulU32U24 => 8,
                Self::MulHiU32U24 => 9,
                Self::MinF32 => 10,
                Self::MaxF32 => 11,
                Self::MinI32 => 12,
                Self::MaxI32 => 13,
                Self::MinU32 => 14,
                Self::MaxU32 => 15,
                Self::LshrrevB32 => 16,
                Self::AshrrevI32 => 17,
                Self::LshlrevB32 => 18,
                Self::AndB32 => 19,
                Self::OrB32 => 20,
                Self::XorB32 => 21,
                Self::XnorB32 => 61,
                Self::FmacF32 => 59,
                Self::AddCoU32 => 25,
                Self::SubCoU32 => 26,
                Self::SubrevCoU32 => 27,
                Self::AddcCoU32 => 28,
                Self::SubbCoU32 => 29,
                Self::SubbrevCoU32 => 30,
                Self::AddU32 => 52,
                Self::SubU32 => 53,
                Self::SubrevU32 => 54,
                Self::AddF16 => 31,
                Self::SubF16 => 32,
                Self::MulF16 => 34,
                Self::MaxF16 => 45,
                Self::MinF16 => 46,
                Self::AddU16 => 38,
                Self::SubU16 => 39,
                Self::LshlrevB16 => 42,
                Self::MulLoU16 => 41,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CndmaskB32 => "v_cndmask_b32",
            Self::AddF32 => "v_add_f32",
            Self::SubF32 => "v_sub_f32",
            Self::SubrevF32 => "v_subrev_f32",
            Self::MulF32 => "v_mul_f32",
            Self::MulI32I24 => "v_mul_i32_i24",
            Self::MulHiI32I24 => "v_mul_hi_i32_i24",
            Self::MulU32U24 => "v_mul_u32_u24",
            Self::MulHiU32U24 => "v_mul_hi_u32_u24",
            Self::MinF32 => "v_min_f32",
            Self::MaxF32 => "v_max_f32",
            Self::MinI32 => "v_min_i32",
            Self::MaxI32 => "v_max_i32",
            Self::MinU32 => "v_min_u32",
            Self::MaxU32 => "v_max_u32",
            Self::LshrrevB32 => "v_lshrrev_b32",
            Self::AshrrevI32 => "v_ashrrev_i32",
            Self::LshlrevB32 => "v_lshlrev_b32",
            Self::AndB32 => "v_and_b32",
            Self::OrB32 => "v_or_b32",
            Self::XorB32 => "v_xor_b32",
            Self::XnorB32 => "v_xnor_b32",
            Self::FmacF32 => "v_fmac_f32",
            Self::AddCoU32 => "v_add_co_u32",
            Self::SubCoU32 => "v_sub_co_u32",
            Self::SubrevCoU32 => "v_subrev_co_u32",
            Self::AddcCoU32 => "v_addc_co_u32",
            Self::SubbCoU32 => "v_subb_co_u32",
            Self::SubbrevCoU32 => "v_subbrev_co_u32",
            Self::AddU32 => "v_add_u32",
            Self::SubU32 => "v_sub_u32",
            Self::SubrevU32 => "v_subrev_u32",
            Self::AddF16 => "v_add_f16",
            Self::SubF16 => "v_sub_f16",
            Self::MulF16 => "v_mul_f16",
            Self::MaxF16 => "v_max_f16",
            Self::MinF16 => "v_min_f16",
            Self::AddU16 => "v_add_u16",
            Self::SubU16 => "v_sub_u16",
            Self::LshlrevB16 => "v_lshlrev_b16",
            Self::MulLoU16 => "v_mul_lo_u16",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOPC); proven by the
// llvm-mc round-trip tests.
impl VopcOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CmpClassF32 => 16,
                Self::CmpLtF32 => 65,
                Self::CmpEqF32 => 66,
                Self::CmpLeF32 => 67,
                Self::CmpGtF32 => 68,
                Self::CmpLgF32 => 69,
                Self::CmpGeF32 => 70,
                Self::CmpOF32 => 71,
                Self::CmpUF32 => 72,
                Self::CmpNeqF32 => 77,
                Self::CmpLtI32 => 193,
                Self::CmpEqI32 => 194,
                Self::CmpLeI32 => 195,
                Self::CmpGtI32 => 196,
                Self::CmpNeI32 => 197,
                Self::CmpGeI32 => 198,
                Self::CmpLtU32 => 201,
                Self::CmpEqU32 => 202,
                Self::CmpLeU32 => 203,
                Self::CmpGtU32 => 204,
                Self::CmpNeU32 => 205,
                Self::CmpGeU32 => 206,
                Self::CmpxEqU32 => 218,
                Self::CmpxNeU32 => 221,
                Self::CmpxGtU32 => 220,
                Self::CmpxLtU32 => 217,
                Self::CmpxGtI32 => 212,
                Self::CmpxLtI32 => 209,
                Self::CmpxGtF32 => 84,
                Self::CmpxLtF32 => 81,
                Self::CmpEqU64 => 234,
                Self::CmpNeU64 => 237,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CmpClassF32 => "v_cmp_class_f32",
            Self::CmpLtF32 => "v_cmp_lt_f32",
            Self::CmpEqF32 => "v_cmp_eq_f32",
            Self::CmpLeF32 => "v_cmp_le_f32",
            Self::CmpGtF32 => "v_cmp_gt_f32",
            Self::CmpLgF32 => "v_cmp_lg_f32",
            Self::CmpGeF32 => "v_cmp_ge_f32",
            Self::CmpOF32 => "v_cmp_o_f32",
            Self::CmpUF32 => "v_cmp_u_f32",
            Self::CmpNeqF32 => "v_cmp_neq_f32",
            Self::CmpLtI32 => "v_cmp_lt_i32",
            Self::CmpEqI32 => "v_cmp_eq_i32",
            Self::CmpLeI32 => "v_cmp_le_i32",
            Self::CmpGtI32 => "v_cmp_gt_i32",
            Self::CmpNeI32 => "v_cmp_ne_i32",
            Self::CmpGeI32 => "v_cmp_ge_i32",
            Self::CmpLtU32 => "v_cmp_lt_u32",
            Self::CmpEqU32 => "v_cmp_eq_u32",
            Self::CmpLeU32 => "v_cmp_le_u32",
            Self::CmpGtU32 => "v_cmp_gt_u32",
            Self::CmpNeU32 => "v_cmp_ne_u32",
            Self::CmpGeU32 => "v_cmp_ge_u32",
            Self::CmpxEqU32 => "v_cmpx_eq_u32",
            Self::CmpxNeU32 => "v_cmpx_ne_u32",
            Self::CmpxGtU32 => "v_cmpx_gt_u32",
            Self::CmpxLtU32 => "v_cmpx_lt_u32",
            Self::CmpxGtI32 => "v_cmpx_gt_i32",
            Self::CmpxLtI32 => "v_cmpx_lt_i32",
            Self::CmpxGtF32 => "v_cmpx_gt_f32",
            Self::CmpxLtF32 => "v_cmpx_lt_f32",
            Self::CmpEqU64 => "v_cmp_eq_u64",
            Self::CmpNeU64 => "v_cmp_ne_u64",
        }
    }
}
