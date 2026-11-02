// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 64-bit vector formats VOP3A / VOP3B (CDNA3 ISA §13.3.5-13.3.6, §12.7-12.10).
//!
//! On gfx9 the VOP3 opcode space embeds the 32-bit formats: VOPC at
//! `0x000 + op`, VOP2 at `0x100 + op`, VOP1 at `0x140 + op`, VOP3-only ops at
//! their own numbers (CDNA3 ISA §12.7.1 "VOP2 using VOP3 encoding" and
//! siblings). VOP3 on gfx9 has no literal constant.
//!
//! Layout (two dwords): `[31:26] = 110100`, `[25:16]` OP, `[15]` CLAMP,
//! VOP3A `[14:11]` OP_SEL and `[10:8]` ABS, VOP3B `[14:8]` SDST, `[7:0]` VDST
//! (a VGPR number, or an SGPR/VCC code for compares and `v_readlane_b32`);
//! second dword `[8:0]` SRC0, `[17:9]` SRC1, `[26:18]` SRC2, `[28:27]` OMOD,
//! `[31:29]` NEG.
//!
//! The encoder is exactly as permissive as `llvm-mc --mcpu=gfx942`, with these
//! deliberate differences (each proven or checked by `tests/enc_vop3.rs`):
//!
//! - opcodes llvm cannot assemble in the 64-bit form (`v_readfirstlane_b32`,
//!   `v_accvgpr_mov_b32`, `v_swap_b32`, which the machine-readable ISA lists
//!   as ENC_VOP3) are not part of [`Vop3Op::all`];
//! - modifier bits an opcode does not honour ([`Vop3Op::mods`]: per-source
//!   NEG/ABS, CLAMP, OMOD, OP_SEL) are rejected by `encode` and by `decode`
//!   (llvm's disassembler silently ignores unused OP_SEL and source fields);
//!   VOP3B has no ABS or OP_SEL field;
//! - fields that an opcode does not use must be zero when decoding, so every
//!   decoded instruction re-encodes to the same words;
//! - 64-bit operands must be even-aligned (VGPR, SGPR and TTMP pairs; gfx90a+
//!   requires aligned VGPR tuples), SGPRs are `s0..=s101`, inline integers
//!   `-16..=64`, AGPRs and literals are not VOP3 operands, destinations that
//!   llvm decodes but that are not writable (`src_*` codes) are rejected;
//! - gfx9 allows one constant-bus read per VOP3 (distinct SGPR / VCC / EXEC /
//!   M0 / `src_*` operand; inline constants do not count; `v_div_fmas_f32`
//!   spends it on its implicit VCC; the lane select of `v_writelane_b32` may
//!   be M0 for free);
//! - `v_cvt_pk_*_f32` and `v_cvt_sr_*_f32` take VGPR sources only (CDNA3 ISA
//!   §7.2 p.51), and `v_lshl_add_u64` rejects an inline shift count above 4
//!   (CDNA3 ISA §12.11: counts above 4 are unsupported).
//!
//! FP8/BF8 converts: `v_cvt_pk_{fp8,bf8}_f32` use `OP_SEL[3]` as the
//! destination half and `v_cvt_sr_{fp8,bf8}_f32` use `OP_SEL[3:2]` as the
//! destination byte (CDNA3 ISA §7.2 Table 31 p.50-51); both preserve the rest
//! of the destination, so [`Vop3::footprint`] lists the destination as read as
//! well as written. Both ignore CLAMP and OMOD (not encodable here).
use crate::arch::Arch;
use crate::enc::vop::{Vop1Op, Vop2Op, VopcOp};
use crate::inst::{Footprint, RegFile};
use crate::operand::{InlineFloat, Src};
use crate::Result;
use std::fmt;

op_table!(Vop3OnlyOp {
    MadI32I24,
    MadU32U24,
    FmaF32,
    BfeU32,
    BfeI32,
    BfiB32,
    AlignbitB32,
    AlignbyteB32,
    Min3F32,
    Min3I32,
    Min3U32,
    Max3F32,
    Max3I32,
    Max3U32,
    Med3F32,
    Med3I32,
    Med3U32,
    DivScaleF32,
    DivFmasF32,
    DivFixupF32,
    MadU64U32,
    MadI64I32,
    LshlAddU32,
    AddLshlU32,
    Add3U32,
    LshlOrB32,
    AndOrB32,
    Or3B32,
    XadU32,
    PermB32,
    MulLoU32,
    MulHiU32,
    MulHiI32,
    LdexpF32,
    CvtPkFp8F32,
    CvtPkBf8F32,
    CvtSrFp8F32,
    CvtSrBf8F32,
    CvtPkrtzF16F32,
    CvtPkU16U32,
    CvtPkI16I32,
    LshlrevB64,
    LshrrevB64,
    AshrrevI64,
    ReadlaneB32,
    WritelaneB32,
    BcntU32B32,
    MbcntLoU32B32,
    MbcntHiU32B32,
    PackB32F16,
    BfmB32,
    AddF64,
    MulF64,
    FmaF64,
    LshlAddU64,
});

/// A VOP3-encoded opcode: a promoted 32-bit opcode or a VOP3-only one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Vop3Op {
    C(VopcOp),
    V2(Vop2Op),
    V1(Vop1Op),
    Only(Vop3OnlyOp),
}

/// The modifier fields an opcode's VOP3 encoding honours. `neg`/`abs` are
/// per-source masks (bit `n` = `src[n]`), `op_sel` a mask over the 4-bit
/// OP_SEL field. Modifier bits outside these masks are rejected by
/// [`Vop3::encode`] and [`Vop3::decode`]. Derived from `llvm-mc` (assembler
/// and disassembler) for gfx942; VOP3B opcodes never have `abs` or `op_sel`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vop3Mods {
    pub neg: u8,
    pub abs: u8,
    pub clamp: bool,
    pub omod: bool,
    pub op_sel: u8,
}

/// Operand class of a source slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// VGPR, SGPR, special register or inline constant.
    Any,
    /// VGPR only.
    Vgpr,
    /// Scalar source (no VGPR): `v_writelane_b32` value.
    Scalar,
    /// Lane select: scalar source (no VGPR).
    Lane,
    /// 64-bit lane mask: SGPR/TTMP pair, VCC, EXEC, FLAT_SCRATCH, XNACK_MASK, `src_*`.
    Mask,
}

/// Operand class of the destination field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DstClass {
    Vgpr,
    Pair,
    Sreg,
    Nop,
}

impl Vop3Op {
    /// Opcodes llvm-mc rejects in their 64-bit form on gfx942.
    fn has_e64(self) -> bool {
        !matches!(
            self,
            Self::V1(Vop1Op::ReadfirstlaneB32 | Vop1Op::AccvgprMovB32 | Vop1Op::SwapB32)
        )
    }

    /// Every encodable VOP3 opcode: promoted VOPC, VOP2, VOP1 (except those
    /// with no VOP3 form on `arch`: `v_readfirstlane_b32`, `v_accvgpr_mov_b32`
    /// and `v_swap_b32` are rejected by llvm in the 64-bit encoding), then
    /// VOP3-only.
    pub fn all(arch: Arch) -> Vec<Vop3Op> {
        let mut ops: Vec<Vop3Op> = Vec::new();
        ops.extend(VopcOp::ALL.iter().map(|&op| Self::C(op)));
        ops.extend(Vop2Op::ALL.iter().map(|&op| Self::V2(op)));
        ops.extend(Vop1Op::ALL.iter().map(|&op| Self::V1(op)));
        ops.extend(Vop3OnlyOp::ALL.iter().map(|&op| Self::Only(op)));
        ops.retain(|op| op.opcode(arch).is_some());
        ops
    }

    /// The 10-bit OP field: VOPC `op`, VOP2 `0x100 + op`, VOP1 `0x140 + op`,
    /// else the VOP3-only number. `None` if `arch` lacks the instruction or
    /// it has no 64-bit form.
    pub fn opcode(self, arch: Arch) -> Option<u16> {
        if !self.has_e64() {
            return None;
        }
        match self {
            Self::C(op) => op.opcode(arch).map(u16::from),
            Self::V2(op) => op.opcode(arch).map(|o| 0x100 + u16::from(o)),
            Self::V1(op) => op.opcode(arch).map(|o| 0x140 + u16::from(o)),
            Self::Only(op) => op.opcode(arch),
        }
    }

    pub fn from_opcode(arch: Arch, opcode: u16) -> Option<Self> {
        let op = match opcode {
            0..=0xff => Self::C(VopcOp::from_opcode(arch, opcode as u8)?),
            0x100..=0x13f => Self::V2(Vop2Op::from_opcode(arch, (opcode - 0x100) as u8)?),
            0x140..=0x1bf => Self::V1(Vop1Op::from_opcode(arch, (opcode - 0x140) as u8)?),
            _ => Self::Only(
                Vop3OnlyOp::ALL
                    .iter()
                    .copied()
                    .find(|op| op.opcode(arch) == Some(opcode))?,
            ),
        };
        op.has_e64().then_some(op)
    }

    /// llvm-mc mnemonic without the `_e64` suffix.
    pub fn name(self) -> &'static str {
        match self {
            Self::C(op) => op.name(),
            Self::V2(op) => op.name(),
            Self::V1(op) => op.name(),
            Self::Only(op) => op.name(),
        }
    }

    /// llvm prints `_e64` after the mnemonic of every promoted opcode (the
    /// 32-bit forms exist) except `v_nop`, and never after VOP3-only ones.
    fn suffix(self) -> &'static str {
        match self {
            Self::V1(Vop1Op::Nop) | Self::Only(_) => "",
            _ => "_e64",
        }
    }

    /// VOP3B (SDST field in bits 14:8 instead of ABS/OP_SEL): carry ops,
    /// `v_div_scale_f32`, `v_mad_u64_u32`, `v_mad_i64_i32`.
    pub fn is_vop3b(self) -> bool {
        matches!(
            self,
            Self::V2(
                Vop2Op::AddCoU32
                    | Vop2Op::SubCoU32
                    | Vop2Op::SubrevCoU32
                    | Vop2Op::AddcCoU32
                    | Vop2Op::SubbCoU32
                    | Vop2Op::SubbrevCoU32
            ) | Self::Only(Vop3OnlyOp::DivScaleF32 | Vop3OnlyOp::MadU64U32 | Vop3OnlyOp::MadI64I32)
        )
    }

    /// Dword widths of (dst, src0, src1, src2); 0 where absent. The VOP3B
    /// SDST is always two dwords and is not part of this tuple.
    pub fn widths(self) -> (u8, u8, u8, u8) {
        use crate::enc::vop::{Vop1Op as P, Vop2Op as B, VopcOp as K};
        use Vop3OnlyOp as O;
        use Vop3Op::{Only, C, V1, V2};
        match self {
            C(K::CmpClassF32) | C(K::CmpLtF32) | C(K::CmpEqF32) | C(K::CmpLeF32)
            | C(K::CmpGtF32) | C(K::CmpLgF32) | C(K::CmpGeF32) | C(K::CmpOF32) | C(K::CmpUF32)
            | C(K::CmpNeqF32) | C(K::CmpLtI32) | C(K::CmpEqI32) | C(K::CmpLeI32)
            | C(K::CmpGtI32) | C(K::CmpNeI32) | C(K::CmpGeI32) | C(K::CmpLtU32)
            | C(K::CmpEqU32) | C(K::CmpLeU32) | C(K::CmpGtU32) | C(K::CmpNeU32)
            | C(K::CmpGeU32) | C(K::CmpxEqU32) | C(K::CmpxNeU32) | C(K::CmpxGtU32)
            | C(K::CmpxLtU32) | C(K::CmpxGtI32) | C(K::CmpxLtI32) | C(K::CmpxGtF32)
            | C(K::CmpxLtF32) => (2, 1, 1, 0),
            C(K::CmpEqU64) | C(K::CmpNeU64) | Only(O::AddF64) | Only(O::MulF64) => (2, 2, 2, 0),
            V2(B::CndmaskB32) | V2(B::AddcCoU32) | V2(B::SubbCoU32) | V2(B::SubbrevCoU32) => {
                (1, 1, 1, 2)
            }
            V1(P::Nop) => (0, 0, 0, 0),
            V1(P::MovB32)
            | V1(P::CvtF32I32)
            | V1(P::CvtF32U32)
            | V1(P::CvtU32F32)
            | V1(P::CvtI32F32)
            | V1(P::CvtF16F32)
            | V1(P::CvtF32F16)
            | V1(P::CvtRpiI32F32)
            | V1(P::CvtFlrI32F32)
            | V1(P::CvtF32Ubyte0)
            | V1(P::CvtF32Ubyte1)
            | V1(P::CvtF32Ubyte2)
            | V1(P::CvtF32Ubyte3)
            | V1(P::FractF32)
            | V1(P::TruncF32)
            | V1(P::CeilF32)
            | V1(P::RndneF32)
            | V1(P::FloorF32)
            | V1(P::ExpF32)
            | V1(P::LogF32)
            | V1(P::RcpF32)
            | V1(P::RcpIflagF32)
            | V1(P::RsqF32)
            | V1(P::SqrtF32)
            | V1(P::NotB32)
            | V1(P::BfrevB32)
            | V1(P::FfbhU32)
            | V1(P::FfblB32)
            | V1(P::FfbhI32)
            | V1(P::FrexpExpI32F32)
            | V1(P::FrexpMantF32)
            | V1(P::CvtF32Fp8)
            | V1(P::CvtF32Bf8)
            | V1(P::ReadfirstlaneB32)
            | V1(P::AccvgprMovB32)
            | V1(P::SwapB32) => (1, 1, 0, 0),
            V1(P::MovB64) => (2, 2, 0, 0),
            V1(P::CvtPkF32Fp8) | V1(P::CvtPkF32Bf8) | V1(P::CvtF64F32) => (2, 1, 0, 0),
            V1(P::CvtF32F64) => (1, 2, 0, 0),
            Only(O::MadI32I24)
            | Only(O::MadU32U24)
            | Only(O::FmaF32)
            | Only(O::BfeU32)
            | Only(O::BfeI32)
            | Only(O::BfiB32)
            | Only(O::AlignbitB32)
            | Only(O::AlignbyteB32)
            | Only(O::Min3F32)
            | Only(O::Min3I32)
            | Only(O::Min3U32)
            | Only(O::Max3F32)
            | Only(O::Max3I32)
            | Only(O::Max3U32)
            | Only(O::Med3F32)
            | Only(O::Med3I32)
            | Only(O::Med3U32)
            | Only(O::DivScaleF32)
            | Only(O::DivFmasF32)
            | Only(O::DivFixupF32)
            | Only(O::LshlAddU32)
            | Only(O::AddLshlU32)
            | Only(O::Add3U32)
            | Only(O::LshlOrB32)
            | Only(O::AndOrB32)
            | Only(O::Or3B32)
            | Only(O::XadU32)
            | Only(O::PermB32) => (1, 1, 1, 1),
            Only(O::MadU64U32) | Only(O::MadI64I32) => (2, 1, 1, 2),
            Only(O::LshlrevB64) | Only(O::LshrrevB64) | Only(O::AshrrevI64) => (2, 1, 2, 0),
            Only(O::FmaF64) => (2, 2, 2, 2),
            Only(O::LshlAddU64) => (2, 2, 1, 2),
            _ => (1, 1, 1, 0),
        }
    }

    /// The modifier fields this opcode's 64-bit encoding honours; see
    /// [`Vop3Mods`].
    pub fn mods(self) -> Vop3Mods {
        use crate::enc::vop::{Vop1Op as P, Vop2Op as B, VopcOp as K};
        use Vop3OnlyOp as O;
        use Vop3Op::{Only, C, V1, V2};
        match self {
            V1(P::CvtU32F32) | V1(P::CvtI32F32) | V1(P::CvtF16F32) | V1(P::CvtF32F16)
            | V1(P::FractF32) | V1(P::TruncF32) | V1(P::CeilF32) | V1(P::RndneF32)
            | V1(P::FloorF32) | V1(P::ExpF32) | V1(P::LogF32) | V1(P::RcpF32)
            | V1(P::RcpIflagF32) | V1(P::RsqF32) | V1(P::SqrtF32) | V1(P::FrexpMantF32)
            | V1(P::CvtF64F32) | V1(P::CvtF32F64) | Only(O::LdexpF32) => Vop3Mods {
                neg: 0b001,
                abs: 0b001,
                clamp: true,
                omod: true,
                op_sel: 0b0000,
            },
            V2(B::MulI32I24) | V2(B::MulU32U24) | V2(B::AddCoU32) | V2(B::SubCoU32)
            | V2(B::SubrevCoU32) | V2(B::AddcCoU32) | V2(B::SubbCoU32) | V2(B::SubbrevCoU32)
            | V2(B::AddU32) | V2(B::SubU32) | V2(B::SubrevU32) | V2(B::AddU16) | V2(B::SubU16)
            | Only(O::MadI32I24) | Only(O::MadU32U24) | Only(O::MadU64U32) | Only(O::MadI64I32) => {
                Vop3Mods {
                    neg: 0b000,
                    abs: 0b000,
                    clamp: true,
                    omod: false,
                    op_sel: 0b0000,
                }
            }
            V2(B::AddF32)
            | V2(B::SubF32)
            | V2(B::SubrevF32)
            | V2(B::MulF32)
            | V2(B::MinF32)
            | V2(B::MaxF32)
            | V2(B::FmacF32)
            | V2(B::AddF16)
            | V2(B::SubF16)
            | V2(B::MulF16)
            | V2(B::MaxF16)
            | V2(B::MinF16)
            | Only(O::CvtPkrtzF16F32)
            | Only(O::AddF64)
            | Only(O::MulF64) => Vop3Mods {
                neg: 0b011,
                abs: 0b011,
                clamp: true,
                omod: true,
                op_sel: 0b0000,
            },
            C(K::CmpLtF32) | C(K::CmpEqF32) | C(K::CmpLeF32) | C(K::CmpGtF32) | C(K::CmpLgF32)
            | C(K::CmpGeF32) | C(K::CmpOF32) | C(K::CmpUF32) | C(K::CmpNeqF32)
            | C(K::CmpxGtF32) | C(K::CmpxLtF32) => Vop3Mods {
                neg: 0b011,
                abs: 0b011,
                clamp: true,
                omod: false,
                op_sel: 0b0000,
            },
            V1(P::CvtF32I32) | V1(P::CvtF32U32) | V1(P::CvtF32Ubyte0) | V1(P::CvtF32Ubyte1)
            | V1(P::CvtF32Ubyte2) | V1(P::CvtF32Ubyte3) | V1(P::CvtF32Fp8) | V1(P::CvtF32Bf8)
            | V1(P::CvtPkF32Fp8) | V1(P::CvtPkF32Bf8) => Vop3Mods {
                neg: 0b000,
                abs: 0b000,
                clamp: true,
                omod: true,
                op_sel: 0b0000,
            },
            Only(O::FmaF32) | Only(O::Min3F32) | Only(O::Max3F32) | Only(O::Med3F32)
            | Only(O::DivFmasF32) | Only(O::DivFixupF32) | Only(O::FmaF64) => Vop3Mods {
                neg: 0b111,
                abs: 0b111,
                clamp: true,
                omod: true,
                op_sel: 0b0000,
            },
            V1(P::CvtRpiI32F32) | V1(P::CvtFlrI32F32) | V1(P::FrexpExpI32F32) => Vop3Mods {
                neg: 0b001,
                abs: 0b001,
                clamp: true,
                omod: false,
                op_sel: 0b0000,
            },
            Only(O::AlignbitB32) | Only(O::AlignbyteB32) => Vop3Mods {
                neg: 0b000,
                abs: 0b000,
                clamp: true,
                omod: false,
                op_sel: 0b1111,
            },
            Only(O::CvtPkFp8F32) | Only(O::CvtPkBf8F32) => Vop3Mods {
                neg: 0b011,
                abs: 0b011,
                clamp: false,
                omod: false,
                op_sel: 0b1011,
            },
            Only(O::CvtSrFp8F32) | Only(O::CvtSrBf8F32) => Vop3Mods {
                neg: 0b001,
                abs: 0b001,
                clamp: false,
                omod: false,
                op_sel: 0b1111,
            },
            C(K::CmpClassF32) => Vop3Mods {
                neg: 0b001,
                abs: 0b001,
                clamp: false,
                omod: false,
                op_sel: 0b0000,
            },
            V2(B::CndmaskB32) => Vop3Mods {
                neg: 0b011,
                abs: 0b011,
                clamp: false,
                omod: false,
                op_sel: 0b0000,
            },
            Only(O::DivScaleF32) => Vop3Mods {
                neg: 0b111,
                abs: 0b000,
                clamp: true,
                omod: true,
                op_sel: 0b0000,
            },
            Only(O::PackB32F16) => Vop3Mods {
                neg: 0b011,
                abs: 0b011,
                clamp: true,
                omod: false,
                op_sel: 0b1011,
            },
            _ => Vop3Mods::default(),
        }
    }

    fn slots(self) -> [Slot; 3] {
        match self {
            Self::Only(Vop3OnlyOp::ReadlaneB32) => [Slot::Vgpr, Slot::Lane, Slot::Any],
            Self::Only(Vop3OnlyOp::WritelaneB32) => [Slot::Scalar, Slot::Lane, Slot::Any],
            Self::V2(
                Vop2Op::CndmaskB32 | Vop2Op::AddcCoU32 | Vop2Op::SubbCoU32 | Vop2Op::SubbrevCoU32,
            ) => [Slot::Any, Slot::Any, Slot::Mask],
            Self::Only(
                Vop3OnlyOp::CvtPkFp8F32
                | Vop3OnlyOp::CvtPkBf8F32
                | Vop3OnlyOp::CvtSrFp8F32
                | Vop3OnlyOp::CvtSrBf8F32,
            ) => [Slot::Vgpr, Slot::Vgpr, Slot::Any],
            _ => [Slot::Any; 3],
        }
    }

    fn dst_class(self) -> DstClass {
        match self {
            Self::C(_) => DstClass::Pair,
            Self::Only(Vop3OnlyOp::ReadlaneB32) => DstClass::Sreg,
            Self::V1(Vop1Op::Nop) => DstClass::Nop,
            _ => DstClass::Vgpr,
        }
    }

    /// The destination keeps bits the instruction does not write (a half, a
    /// byte, 31 lanes), so the old value is an input.
    fn reads_dst(self) -> bool {
        matches!(
            self,
            Self::Only(
                Vop3OnlyOp::CvtPkFp8F32
                    | Vop3OnlyOp::CvtPkBf8F32
                    | Vop3OnlyOp::CvtSrFp8F32
                    | Vop3OnlyOp::CvtSrBf8F32
                    | Vop3OnlyOp::WritelaneB32,
            )
        )
    }

    /// Lane-masked VALU op. `v_readlane_b32` / `v_writelane_b32` address one
    /// lane regardless of EXEC. [UNVERIFIED: confirm on MI300X]
    fn reads_exec(self) -> bool {
        !matches!(
            self,
            Self::V1(Vop1Op::Nop) | Self::Only(Vop3OnlyOp::ReadlaneB32 | Vop3OnlyOp::WritelaneB32)
        )
    }

    /// `v_cmpx_*` in the 64-bit form writes EXEC as well as SDST.
    fn writes_exec(self) -> bool {
        matches!(
            self,
            Self::C(
                VopcOp::CmpxEqU32
                    | VopcOp::CmpxNeU32
                    | VopcOp::CmpxGtU32
                    | VopcOp::CmpxLtU32
                    | VopcOp::CmpxGtI32
                    | VopcOp::CmpxLtI32
                    | VopcOp::CmpxGtF32
                    | VopcOp::CmpxLtF32,
            )
        )
    }

    /// `OP_SEL` as llvm lists it: two-source ops with a destination select
    /// print `[src0, src1, dst]` (bits 0, 1, 3), the others bits 0..=3.
    fn op_sel_bits(self) -> &'static [u8] {
        match self {
            Self::Only(
                Vop3OnlyOp::CvtPkFp8F32 | Vop3OnlyOp::CvtPkBf8F32 | Vop3OnlyOp::PackB32F16,
            ) => &[0, 1, 3],
            _ => &[0, 1, 2, 3],
        }
    }
}

/// `dst = op(src0, src1, src2)` in the 64-bit encoding. `dst` is a VGPR, or
/// an SGPR/VCC for promoted VOPC compares and `v_readlane_b32`. `sdst` is
/// the VOP3B scalar destination (carry-out / `v_div_scale` VCC), `None` for
/// VOP3A. `abs`/`neg` are 3-bit per-source masks, `op_sel` the 4-bit OP_SEL
/// field (bit 3 = destination half for the FP8 converts, CDNA3 ISA §7.2 Table 31).
/// Absent operands (`Vop3Op::widths` is 0) are `None`; `v_nop` is
/// `dst = Vgpr(0)` with no sources.
#[derive(Clone, Debug, PartialEq)]
pub struct Vop3 {
    pub op: Vop3Op,
    pub dst: Src,
    pub sdst: Option<Src>,
    pub src: [Option<Src>; 3],
    pub abs: u8,
    pub neg: u8,
    pub clamp: bool,
    pub omod: u8,
    pub op_sel: u8,
}

/// Register-class and alignment rules shared by every operand.
fn check_reg(s: Src, w: u8) -> std::result::Result<(), String> {
    match s {
        Src::Lit(_) => Err("gfx9 VOP3 has no literal constant".into()),
        Src::Agpr(_) => Err("an AccVGPR is not a VOP3 operand".into()),
        Src::Sgpr(n) if n > 101 => Err(format!("s{n} does not exist (s0..=s101)")),
        Src::Sgpr(n) if w == 2 && n % 2 == 1 => Err(format!(
            "64-bit operand s[{n}:{}] is not even-aligned",
            u16::from(n) + 1
        )),
        Src::Ttmp(n) if n > 15 => Err(format!("ttmp{n} does not exist (ttmp0..=ttmp15)")),
        Src::Ttmp(n) if w == 2 && n % 2 == 1 => Err(format!(
            "64-bit operand ttmp[{n}:{}] is not even-aligned",
            u16::from(n) + 1
        )),
        Src::Vgpr(n) if w == 2 && n % 2 == 1 => Err(format!(
            "64-bit operand v[{n}:{}] is not even-aligned",
            u16::from(n) + 1
        )),
        Src::VccHi | Src::ExecHi | Src::FlatScratchHi | Src::XnackMaskHi | Src::M0 if w == 2 => {
            Err(format!("{} cannot start a 64-bit operand", s.fmt(1)))
        }
        Src::Int(i) if !(-16..=64).contains(&i) => {
            Err(format!("inline integer {i} is outside -16..=64"))
        }
        _ => Ok(()),
    }
}

fn check_slot(kind: Slot, s: Src, w: u8) -> std::result::Result<(), String> {
    match kind {
        Slot::Any => {}
        Slot::Vgpr => {
            if !s.is_vgpr() {
                return Err(format!("{} must be a VGPR", s.fmt(w)));
            }
        }
        Slot::Scalar | Slot::Lane => {
            if s.is_vgpr() {
                return Err(format!("{} must be a scalar operand", s.fmt(w)));
            }
        }
        Slot::Mask => {
            let ok = matches!(
                s,
                Src::Sgpr(_)
                    | Src::Ttmp(_)
                    | Src::VccLo
                    | Src::ExecLo
                    | Src::FlatScratchLo
                    | Src::XnackMaskLo
                    | Src::SharedBase
                    | Src::SharedLimit
                    | Src::PrivateBase
                    | Src::PrivateLimit
                    | Src::Vccz
                    | Src::Execz
                    | Src::Scc
            );
            if !ok {
                return Err(format!("{} is not a 64-bit lane-mask operand", s.fmt(w)));
            }
        }
    }
    check_reg(s, w)
}

fn check_dst(class: DstClass, s: Src, w: u8) -> std::result::Result<(), String> {
    match class {
        DstClass::Vgpr => {
            if !s.is_vgpr() {
                return Err(format!("destination {} must be a VGPR", s.fmt(w)));
            }
            check_reg(s, w)
        }
        DstClass::Pair => {
            let ok = matches!(
                s,
                Src::Sgpr(_)
                    | Src::Ttmp(_)
                    | Src::VccLo
                    | Src::ExecLo
                    | Src::FlatScratchLo
                    | Src::XnackMaskLo
            );
            if !ok {
                return Err(format!(
                    "{} is not a writable 64-bit scalar destination",
                    s.fmt(2)
                ));
            }
            check_reg(s, 2)
        }
        DstClass::Sreg => {
            let ok = matches!(
                s,
                Src::Sgpr(_)
                    | Src::Ttmp(_)
                    | Src::VccLo
                    | Src::VccHi
                    | Src::ExecLo
                    | Src::ExecHi
                    | Src::FlatScratchLo
                    | Src::FlatScratchHi
                    | Src::XnackMaskLo
                    | Src::XnackMaskHi
            );
            if !ok {
                return Err(format!(
                    "{} is not a writable 32-bit scalar destination",
                    s.fmt(1)
                ));
            }
            check_reg(s, 1)
        }
        DstClass::Nop => {
            if s != Src::Vgpr(0) {
                return Err("v_nop has no destination (dst must be Vgpr(0))".into());
            }
            Ok(())
        }
    }
}

/// The constant-bus identity of a source: scalar registers and `src_*`
/// values count, VGPRs and inline constants do not. Width distinguishes
/// `s2` from `s[2:3]` (llvm counts them separately).
fn bus_key(s: Src, w: u8) -> Option<(u16, u8)> {
    match s {
        Src::Int(_) | Src::Float(_) | Src::Lit(_) | Src::Vgpr(_) | Src::Agpr(_) => None,
        _ => Some((s.code(), w)),
    }
}

impl Vop3 {
    /// Check every rule listed in the module docs and return the OP field.
    fn validate(&self, arch: Arch) -> Result<u16> {
        let op = self.op;
        let name = op.name();
        let opcode = op
            .opcode(arch)
            .ok_or_else(|| format!("{name}: no VOP3 encoding on {arch}"))?;
        let (wd, w0, w1, w2) = op.widths();
        let ws = [w0, w1, w2];
        let mods = op.mods();

        if op.is_vop3b() != self.sdst.is_some() {
            return Err(if op.is_vop3b() {
                format!("{name}: VOP3B form requires an SDST operand")
            } else {
                format!("{name}: VOP3A form has no SDST operand")
            });
        }
        for (j, &w) in ws.iter().enumerate() {
            if (w > 0) != self.src[j].is_some() {
                return Err(format!(
                    "{name}: src{j} must be {}",
                    if w > 0 { "present" } else { "absent" }
                ));
            }
        }
        if self.neg > 7 || self.abs > 7 || self.omod > 3 || self.op_sel > 15 {
            return Err(format!(
                "{name}: modifier field out of range (neg {:#x}, abs {:#x}, omod {}, op_sel {:#x})",
                self.neg, self.abs, self.omod, self.op_sel
            ));
        }
        if self.neg & !mods.neg != 0 {
            return Err(format!(
                "{name}: neg {:#05b} not supported (allowed {:#05b})",
                self.neg, mods.neg
            ));
        }
        if self.abs & !mods.abs != 0 {
            return Err(format!(
                "{name}: abs {:#05b} not supported (allowed {:#05b})",
                self.abs, mods.abs
            ));
        }
        if self.clamp && !mods.clamp {
            return Err(format!("{name}: clamp not supported"));
        }
        if self.omod != 0 && !mods.omod {
            return Err(format!("{name}: omod not supported"));
        }
        if self.op_sel & !mods.op_sel != 0 {
            return Err(format!(
                "{name}: op_sel {:#06b} not supported (allowed {:#06b})",
                self.op_sel, mods.op_sel
            ));
        }

        check_dst(op.dst_class(), self.dst, wd).map_err(|e| format!("{name}: {e}"))?;
        if let Some(s) = self.sdst {
            check_dst(DstClass::Pair, s, 2).map_err(|e| format!("{name}: sdst: {e}"))?
        }
        let slots = op.slots();
        for j in 0..3 {
            if let Some(s) = self.src[j] {
                check_slot(slots[j], s, ws[j]).map_err(|e| format!("{name}: src{j}: {e}"))?
            }
        }
        if matches!(op, Vop3Op::Only(Vop3OnlyOp::LshlAddU64)) {
            if let Some(Src::Int(i)) = self.src[1] {
                if !(0..=4).contains(&i) {
                    return Err(format!("{name}: shift count {i} is unsupported (0..=4)"));
                }
            }
        }

        // Constant bus: gfx9 VOP3 reads at most one distinct scalar value.
        let mut bus: Vec<(u16, u8)> = Vec::with_capacity(4);
        for j in 0..3 {
            let Some(s) = self.src[j] else { continue };
            if matches!(op, Vop3Op::Only(Vop3OnlyOp::WritelaneB32)) && j == 1 && s == Src::M0 {
                continue;
            }
            if let Some(key) = bus_key(s, ws[j]) {
                if !bus.contains(&key) {
                    bus.push(key)
                }
            }
        }
        if matches!(op, Vop3Op::Only(Vop3OnlyOp::DivFmasF32)) && !bus.contains(&(106, 2)) {
            bus.push((106, 2))
        }
        if bus.len() > 1 {
            return Err(format!("{name}: violates the gfx9 constant-bus limit (one scalar source; implicit VCC counts)"));
        }
        Ok(opcode)
    }

    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let opcode = self.validate(arch)?;
        let vdst = match self.dst {
            Src::Vgpr(n) => u32::from(n),
            s => u32::from(s.code()),
        };
        let mut lo =
            (0b11_0100u32 << 26) | (u32::from(opcode) << 16) | vdst | (u32::from(self.clamp) << 15);
        lo |= match self.sdst {
            Some(s) => u32::from(s.code()) << 8,
            None => (u32::from(self.abs) << 8) | (u32::from(self.op_sel) << 11),
        };
        let field = |j: usize| self.src[j].map_or(0u32, |s| u32::from(s.code()));
        let hi = field(0)
            | (field(1) << 9)
            | (field(2) << 18)
            | (u32::from(self.omod) << 27)
            | (u32::from(self.neg) << 29);
        out.push(lo);
        out.push(hi);
        Ok(())
    }

    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let (lo, hi) = match words {
            [lo, hi, ..] => (*lo, *hi),
            _ => return Err("VOP3 needs two dwords".to_string()),
        };
        if lo >> 26 != 0b11_0100 {
            return Err(format!("{lo:#010x} is not a VOP3 first dword"));
        }
        let opcode = ((lo >> 16) & 0x3ff) as u16;
        let op = Vop3Op::from_opcode(arch, opcode)
            .ok_or_else(|| format!("no VOP3 instruction with opcode {opcode:#x} on {arch}"))?;
        let (_, w0, w1, w2) = op.widths();
        let vdst = lo & 0xff;
        let dst = match op.dst_class() {
            DstClass::Vgpr | DstClass::Nop => Src::Vgpr(vdst as u8),
            DstClass::Pair | DstClass::Sreg => Src::from_code(vdst as u16, None)?,
        };
        let (sdst, abs, op_sel) = if op.is_vop3b() {
            (Some(Src::from_code(((lo >> 8) & 0x7f) as u16, None)?), 0, 0)
        } else {
            (None, ((lo >> 8) & 0x7) as u8, ((lo >> 11) & 0xf) as u8)
        };
        let mut src: [Option<Src>; 3] = [None; 3];
        for (j, &(shift, w)) in [(0u32, w0), (9, w1), (18, w2)].iter().enumerate() {
            let code = (hi >> shift) & 0x1ff;
            if w > 0 {
                src[j] = Some(Src::from_code(code as u16, None)?);
            } else if code != 0 {
                return Err(format!(
                    "{}: unused src{j} field is {code:#x}, must be 0",
                    op.name()
                ));
            }
        }
        let inst = Vop3 {
            op,
            dst,
            sdst,
            src,
            abs,
            neg: (hi >> 29) as u8,
            clamp: (lo >> 15) & 1 != 0,
            omod: ((hi >> 27) & 0x3) as u8,
            op_sel,
        };
        inst.validate(arch)?;
        Ok((inst, 2))
    }

    /// Registers read and written. Explicit: `dst` (and the VOP3B `sdst`)
    /// written, sources read, `dst` also read where the instruction keeps
    /// part of it (FP8 packed/stochastic converts, `v_writelane_b32`).
    /// Implicit: EXEC read by every lane-masked op, EXEC written by
    /// `v_cmpx_*`, VCC read by `v_div_fmas_f32`.
    pub fn footprint(&self) -> Footprint {
        let (wd, w0, w1, w2) = self.op.widths();
        let mut fp = Footprint::default();
        if wd > 0 {
            fp.write(self.dst, wd);
            if self.op.reads_dst() {
                fp.read(self.dst, wd)
            }
        }
        if let Some(s) = self.sdst {
            fp.write(s, 2)
        }
        for (src, w) in self.src.iter().zip([w0, w1, w2]) {
            if let Some(s) = *src {
                fp.read(s, w)
            }
        }
        if self.op.reads_exec() {
            fp.read_file(RegFile::Exec, 0, 2)
        }
        if self.op.writes_exec() {
            fp.write_file(RegFile::Exec, 0, 2)
        }
        if matches!(self.op, Vop3Op::Only(Vop3OnlyOp::DivFmasF32)) {
            fp.read_file(RegFile::Vcc, 0, 2)
        }
        fp
    }
}

/// A source with its NEG/ABS modifiers as llvm prints them: `-v1`, `|v1|`,
/// `-|v1|`, and `neg(1.0)` for a negated inline constant (`-` would read as
/// part of the number). `1/(2*pi)` in a 64-bit slot prints as the f64.
fn src_text(s: Src, w: u8, neg: bool, abs: bool) -> String {
    let mut t = if w == 2 && s == Src::Float(InlineFloat::InvTwoPi) {
        "0.15915494309189532".to_string()
    } else {
        s.fmt(w)
    };
    if abs {
        t = format!("|{t}|")
    }
    if neg {
        t = if s.is_inline() && !abs {
            format!("neg({t})")
        } else {
            format!("-{t}")
        }
    }
    t
}

impl fmt::Display for Vop3 {
    /// `llvm-objdump --mcpu=gfx942` text, `_e64` suffix where llvm prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wd, w0, w1, w2) = self.op.widths();
        let mut operands: Vec<String> = Vec::with_capacity(5);
        if wd > 0 {
            operands.push(self.dst.fmt(wd))
        }
        if let Some(s) = self.sdst {
            operands.push(s.fmt(2))
        }
        for (j, w) in [w0, w1, w2].into_iter().enumerate() {
            if let Some(s) = self.src[j] {
                operands.push(src_text(
                    s,
                    w,
                    (self.neg >> j) & 1 != 0,
                    (self.abs >> j) & 1 != 0,
                ));
            }
        }
        write!(f, "{}{}", self.op.name(), self.op.suffix())?;
        if !operands.is_empty() {
            write!(f, " {}", operands.join(", "))?
        }
        if self.op_sel != 0 {
            let bits: Vec<String> = self
                .op
                .op_sel_bits()
                .iter()
                .map(|&b| ((self.op_sel >> b) & 1).to_string())
                .collect();
            write!(f, " op_sel:[{}]", bits.join(","))?;
        }
        if self.clamp {
            f.write_str(" clamp")?
        }
        match self.omod {
            1 => f.write_str(" mul:2")?,
            2 => f.write_str(" mul:4")?,
            3 => f.write_str(" div:2")?,
            _ => {}
        }
        Ok(())
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP3 / VOP3_SDST_ENC); proven by the
// llvm-mc round-trip tests.
impl Vop3OnlyOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u16> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MadI32I24 => 450,
                Self::MadU32U24 => 451,
                Self::FmaF32 => 459,
                Self::BfeU32 => 456,
                Self::BfeI32 => 457,
                Self::BfiB32 => 458,
                Self::AlignbitB32 => 462,
                Self::AlignbyteB32 => 463,
                Self::Min3F32 => 464,
                Self::Min3I32 => 465,
                Self::Min3U32 => 466,
                Self::Max3F32 => 467,
                Self::Max3I32 => 468,
                Self::Max3U32 => 469,
                Self::Med3F32 => 470,
                Self::Med3I32 => 471,
                Self::Med3U32 => 472,
                Self::DivScaleF32 => 480,
                Self::DivFmasF32 => 482,
                Self::DivFixupF32 => 478,
                Self::MadU64U32 => 488,
                Self::MadI64I32 => 489,
                Self::LshlAddU32 => 509,
                Self::AddLshlU32 => 510,
                Self::Add3U32 => 511,
                Self::LshlOrB32 => 512,
                Self::AndOrB32 => 513,
                Self::Or3B32 => 514,
                Self::XadU32 => 499,
                Self::PermB32 => 493,
                Self::MulLoU32 => 645,
                Self::MulHiU32 => 646,
                Self::MulHiI32 => 647,
                Self::LdexpF32 => 648,
                Self::CvtPkFp8F32 => 674,
                Self::CvtPkBf8F32 => 675,
                Self::CvtSrFp8F32 => 676,
                Self::CvtSrBf8F32 => 677,
                Self::CvtPkrtzF16F32 => 662,
                Self::CvtPkU16U32 => 663,
                Self::CvtPkI16I32 => 664,
                Self::LshlrevB64 => 655,
                Self::LshrrevB64 => 656,
                Self::AshrrevI64 => 657,
                Self::ReadlaneB32 => 649,
                Self::WritelaneB32 => 650,
                Self::BcntU32B32 => 651,
                Self::MbcntLoU32B32 => 652,
                Self::MbcntHiU32B32 => 653,
                Self::PackB32F16 => 672,
                Self::BfmB32 => 659,
                Self::AddF64 => 640,
                Self::MulF64 => 641,
                Self::FmaF64 => 460,
                Self::LshlAddU64 => 520,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MadI32I24 => "v_mad_i32_i24",
            Self::MadU32U24 => "v_mad_u32_u24",
            Self::FmaF32 => "v_fma_f32",
            Self::BfeU32 => "v_bfe_u32",
            Self::BfeI32 => "v_bfe_i32",
            Self::BfiB32 => "v_bfi_b32",
            Self::AlignbitB32 => "v_alignbit_b32",
            Self::AlignbyteB32 => "v_alignbyte_b32",
            Self::Min3F32 => "v_min3_f32",
            Self::Min3I32 => "v_min3_i32",
            Self::Min3U32 => "v_min3_u32",
            Self::Max3F32 => "v_max3_f32",
            Self::Max3I32 => "v_max3_i32",
            Self::Max3U32 => "v_max3_u32",
            Self::Med3F32 => "v_med3_f32",
            Self::Med3I32 => "v_med3_i32",
            Self::Med3U32 => "v_med3_u32",
            Self::DivScaleF32 => "v_div_scale_f32",
            Self::DivFmasF32 => "v_div_fmas_f32",
            Self::DivFixupF32 => "v_div_fixup_f32",
            Self::MadU64U32 => "v_mad_u64_u32",
            Self::MadI64I32 => "v_mad_i64_i32",
            Self::LshlAddU32 => "v_lshl_add_u32",
            Self::AddLshlU32 => "v_add_lshl_u32",
            Self::Add3U32 => "v_add3_u32",
            Self::LshlOrB32 => "v_lshl_or_b32",
            Self::AndOrB32 => "v_and_or_b32",
            Self::Or3B32 => "v_or3_b32",
            Self::XadU32 => "v_xad_u32",
            Self::PermB32 => "v_perm_b32",
            Self::MulLoU32 => "v_mul_lo_u32",
            Self::MulHiU32 => "v_mul_hi_u32",
            Self::MulHiI32 => "v_mul_hi_i32",
            Self::LdexpF32 => "v_ldexp_f32",
            Self::CvtPkFp8F32 => "v_cvt_pk_fp8_f32",
            Self::CvtPkBf8F32 => "v_cvt_pk_bf8_f32",
            Self::CvtSrFp8F32 => "v_cvt_sr_fp8_f32",
            Self::CvtSrBf8F32 => "v_cvt_sr_bf8_f32",
            Self::CvtPkrtzF16F32 => "v_cvt_pkrtz_f16_f32",
            Self::CvtPkU16U32 => "v_cvt_pk_u16_u32",
            Self::CvtPkI16I32 => "v_cvt_pk_i16_i32",
            Self::LshlrevB64 => "v_lshlrev_b64",
            Self::LshrrevB64 => "v_lshrrev_b64",
            Self::AshrrevI64 => "v_ashrrev_i64",
            Self::ReadlaneB32 => "v_readlane_b32",
            Self::WritelaneB32 => "v_writelane_b32",
            Self::BcntU32B32 => "v_bcnt_u32_b32",
            Self::MbcntLoU32B32 => "v_mbcnt_lo_u32_b32",
            Self::MbcntHiU32B32 => "v_mbcnt_hi_u32_b32",
            Self::PackB32F16 => "v_pack_b32_f16",
            Self::BfmB32 => "v_bfm_b32",
            Self::AddF64 => "v_add_f64",
            Self::MulF64 => "v_mul_f64",
            Self::FmaF64 => "v_fma_f64",
            Self::LshlAddU64 => "v_lshl_add_u64",
        }
    }
}
