// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP3P packed math and the VOP3P-MAI matrix format (MFMA, accvgpr moves)
//! (CDNA3 ISA §6.7 p.39, §7, §13.3.6-13.3.6.1 p.539-541).
//!
use crate::arch::Arch;
use crate::inst::{Footprint, RegFile};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(Vop3pOp {
    PkFmaF16,
    PkAddF16,
    PkMulF16,
    PkMinF16,
    PkMaxF16,
    PkAddU16,
    PkSubU16,
    PkMulLoU16,
    PkLshlrevB16,
    PkLshrrevB16,
    PkMaxI16,
    PkMinI16,
    FmaMixF32,
    FmaMixloF16,
    FmaMixhiF16,
    Dot2F32F16,
    Dot2I32I16,
    Dot2U32U16,
    Dot4I32I8,
    Dot4U32U8,
    Dot8I32I4,
    Dot8U32U4,
    PkFmaF32,
    PkMulF32,
    PkAddF32,
    PkMovB32,
});

op_table!(MaiOp {
    AccvgprReadB32,
    AccvgprWriteB32,
    MfmaF32_16x16x16Bf16,
    MfmaF32_32x32x8Bf16,
    MfmaF32_16x16x16F16,
    MfmaF32_32x32x8F16,
    MfmaF32_16x16x32Fp8Fp8,
    MfmaF32_16x16x32Fp8Bf8,
    MfmaF32_16x16x32Bf8Fp8,
    MfmaF32_16x16x32Bf8Bf8,
    MfmaF32_32x32x16Fp8Fp8,
    MfmaF32_32x32x16Fp8Bf8,
    MfmaF32_32x32x16Bf8Fp8,
    MfmaF32_32x32x16Bf8Bf8,
    MfmaI32_16x16x32I8,
    MfmaI32_32x32x16I8,
    MfmaF32_16x16x4F32,
    MfmaF32_32x32x2F32,
    MfmaF64_16x16x4F64,
});

impl Vop3pOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Dword widths of (dst, src0, src1, src2); 0 where absent.
    pub fn widths(self) -> (u8, u8, u8, u8) {
        use Vop3pOp::*;
        match self {
            PkFmaF32 => (2, 2, 2, 2),
            PkMulF32 | PkAddF32 | PkMovB32 => (2, 2, 2, 0),
            PkFmaF16 | FmaMixF32 | FmaMixloF16 | FmaMixhiF16 | Dot2F32F16 | Dot2I32I16
            | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4 | Dot8U32U4 => (1, 1, 1, 1),
            _ => (1, 1, 1, 0),
        }
    }
    fn mix(self) -> bool {
        matches!(
            self,
            Self::FmaMixF32 | Self::FmaMixloF16 | Self::FmaMixhiF16
        )
    }
    fn dot(self) -> bool {
        self.name().starts_with("v_dot")
    }
}

/// Element type of an MFMA operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MfmaType {
    F32,
    F16,
    Bf16,
    Fp8,
    Bf8,
    I8,
    I32,
    F64,
}

/// `D[b,i,j] = C[b,i,j] + Σ_k A[b,i,k]·B[b,k,j]` over `blocks` blocks of
/// M×N×K (CDNA3 ISA §7.1.4 p.46-48), with the per-lane dword counts of the
/// A, B and C/D register ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MfmaShape {
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub blocks: u8,
    pub a: MfmaType,
    pub b: MfmaType,
    pub cd: MfmaType,
    pub a_dwords: u8,
    pub b_dwords: u8,
    pub cd_dwords: u8,
}

impl MaiOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// The matrix shape; `None` for the accvgpr moves.
    pub fn shape(self) -> Option<MfmaShape> {
        use MfmaType::*;
        let s = |m, n, k, blocks, a, b, cd, a_dwords, cd_dwords| {
            Some(MfmaShape {
                m,
                n,
                k,
                blocks,
                a,
                b,
                cd,
                a_dwords,
                b_dwords: a_dwords,
                cd_dwords,
            })
        };
        match self {
            Self::AccvgprReadB32 | Self::AccvgprWriteB32 => None,
            Self::MfmaF32_16x16x16Bf16 => s(16, 16, 16, 1, Bf16, Bf16, F32, 2, 4),
            Self::MfmaF32_32x32x8Bf16 => s(32, 32, 8, 1, Bf16, Bf16, F32, 2, 16),
            Self::MfmaF32_16x16x16F16 => s(16, 16, 16, 1, F16, F16, F32, 2, 4),
            Self::MfmaF32_32x32x8F16 => s(32, 32, 8, 1, F16, F16, F32, 2, 16),
            Self::MfmaF32_16x16x32Fp8Fp8 => s(16, 16, 32, 1, Fp8, Fp8, F32, 2, 4),
            Self::MfmaF32_16x16x32Fp8Bf8 => s(16, 16, 32, 1, Fp8, Bf8, F32, 2, 4),
            Self::MfmaF32_16x16x32Bf8Fp8 => s(16, 16, 32, 1, Bf8, Fp8, F32, 2, 4),
            Self::MfmaF32_16x16x32Bf8Bf8 => s(16, 16, 32, 1, Bf8, Bf8, F32, 2, 4),
            Self::MfmaF32_32x32x16Fp8Fp8 => s(32, 32, 16, 1, Fp8, Fp8, F32, 2, 16),
            Self::MfmaF32_32x32x16Fp8Bf8 => s(32, 32, 16, 1, Fp8, Bf8, F32, 2, 16),
            Self::MfmaF32_32x32x16Bf8Fp8 => s(32, 32, 16, 1, Bf8, Fp8, F32, 2, 16),
            Self::MfmaF32_32x32x16Bf8Bf8 => s(32, 32, 16, 1, Bf8, Bf8, F32, 2, 16),
            Self::MfmaI32_16x16x32I8 => s(16, 16, 32, 1, I8, I8, I32, 2, 4),
            Self::MfmaI32_32x32x16I8 => s(32, 32, 16, 1, I8, I8, I32, 2, 16),
            Self::MfmaF32_16x16x4F32 => s(16, 16, 4, 1, F32, F32, F32, 1, 4),
            Self::MfmaF32_32x32x2F32 => s(32, 32, 2, 1, F32, F32, F32, 1, 16),
            Self::MfmaF64_16x16x4F64 => s(16, 16, 4, 1, F64, F64, F64, 2, 8),
        }
    }
}

/// Packed math: `dst = op(src0, src1, src2)` with per-half OP_SEL /
/// OP_SEL_HI / NEG_LO / NEG_HI masks (3 bits each, one per source).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop3p {
    pub op: Vop3pOp,
    pub dst: Src,
    pub src: [Option<Src>; 3],
    pub op_sel: u8,
    pub op_sel_hi: u8,
    pub neg_lo: u8,
    pub neg_hi: u8,
    pub clamp: bool,
}

/// VOP3P-MAI. `dst`/`src_c` are `Agpr` when ACC_CD is set (they must agree;
/// `src_c` may also be an inline constant); `src_a`/`src_b` are `Vgpr` or
/// `Agpr` (the ACC bits). `v_accvgpr_read_b32 vD, aS`: `dst = Vgpr`,
/// `src_a = Agpr`. `v_accvgpr_write_b32 aD, src`: `dst = Agpr`,
/// `src_a` = VGPR, scalar source or inline constant. `src_b`/`src_c` are `None` for the moves.
///
/// Encodes the complete llvm-mc modifier field range, including broadcasts
/// that the ISA does not define for one-block shapes (§7.1.5.1 p.49).
/// Their execution is [UNVERIFIED: confirm on MI300X]; encoding support is
/// not a claim that these forms have defined arithmetic semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct Mai {
    pub op: MaiOp,
    pub dst: Src,
    pub src_a: Src,
    pub src_b: Option<Src>,
    pub src_c: Option<Src>,
    pub cbsz: u8,
    pub abid: u8,
    /// Lane-group pattern; for F64 MFMA these bits negate A/B/C instead
    /// (CDNA3 ISA §13.3.6.1 p.541), printed as `neg:[a,b,c]`.
    pub blgp: u8,
}

/// Register tuples follow CDNA3 ISA §6.7 p.39 and §7.1.4 p.46-48.
fn tuple(src: Src, width: u8, acc: bool, scalar: bool) -> Result<()> {
    match src {
        Src::Vgpr(n) | Src::Agpr(n) => {
            if src.is_agpr() && !acc {
                return Err("AGPR is not legal in packed math".into());
            }
            if (width > 1 && n % 2 != 0) || u16::from(n) + u16::from(width) > 256 {
                return Err(format!(
                    "misaligned or overflowing register tuple {src:?} width {width}"
                ));
            }
        }
        Src::Int(i) if (-16..=64).contains(&i) => {}
        Src::Float(_) => {}
        _ if scalar && !src.is_literal() && Src::from_code(src.code(), None)? == src => {
            if let Some(r) = crate::inst::RegSpan::of(src, width) {
                let limit = match r.file {
                    RegFile::Sgpr => 102,
                    RegFile::Ttmp => 16,
                    RegFile::M0 | RegFile::Scc => 1,
                    _ => 2,
                };
                if u16::from(r.base) + u16::from(width) > limit
                    || (width > 1 && r.base % width != 0)
                {
                    return Err("misaligned or overflowing scalar tuple".into());
                }
            }
        }
        _ => return Err(format!("illegal source {src:?}")),
    }
    Ok(())
}

fn mask_fmt(f: &mut fmt::Formatter<'_>, name: &str, mask: u8, count: usize) -> fmt::Result {
    write!(f, " {name}:[")?;
    for i in 0..count {
        if i != 0 {
            f.write_str(",")?
        }
        write!(f, "{}", (mask >> i) & 1)?;
    }
    f.write_str("]")
}

impl Vop3p {
    /// CDNA3 ISA §13.3.6 p.539-540. OPSEL_HI's actual source ordering is
    /// verified against gfx942 llvm-mc (Table 87 reverses the split fields).
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        out.extend_from_slice(&self.encoded(arch)?);
        Ok(())
    }
    fn encoded(&self, arch: Arch) -> Result<[u32; 2]> {
        let op = self.op.opcode(arch).ok_or("packed opcode unavailable")?;
        let (d, a, b, c) = self.op.widths();
        if !self.dst.is_vgpr() {
            return Err("packed destination must be a VGPR".into());
        }
        tuple(self.dst, d, false, false)?;
        if [self.op_sel, self.op_sel_hi, self.neg_lo, self.neg_hi]
            .iter()
            .any(|&m| m > 7)
        {
            return Err("packed modifier exceeds three bits".into());
        }
        let active = if c == 0 { 3 } else { 7 };
        if (self.op_sel | self.neg_lo | self.neg_hi) & !active != 0
            || (c == 0 && self.op_sel_hi & 4 == 0)
        {
            return Err("noncanonical modifier on absent source".into());
        }
        if self.op.mix() && self.neg_hi != 0 {
            return Err("mix has no NEG_HI modifier".into());
        }
        if self.op.dot() && (self.op_sel != 0 || self.op_sel_hi != 7) {
            return Err("dot opcode does not support these packed modifiers".into());
        }
        let mut w1 = (u32::from(self.op_sel_hi & 3) << 27) | (u32::from(self.neg_lo) << 29);
        let mut scalar = None;
        for (i, width) in [a, b, c].into_iter().enumerate() {
            match (width, self.src[i]) {
                (0, None) => {}
                (0, Some(_)) | (_, None) => return Err("wrong packed source count".into()),
                (_, Some(src)) => {
                    tuple(src, width, false, true)?;
                    if !src.is_vgpr() && !src.is_inline() {
                        if scalar.is_some_and(|s| s != src) {
                            return Err(
                                "packed constant bus permits only one scalar register source"
                                    .into(),
                            );
                        }
                        scalar = Some(src);
                    }
                    w1 |= u32::from(src.code()) << (9 * i);
                }
            }
        }
        Ok([
            0xd3800000
                | (u32::from(op) << 16)
                | (u32::from(self.dst.code()) & 255)
                | (u32::from(self.neg_hi) << 8)
                | (u32::from(self.op_sel) << 11)
                | (u32::from(self.op_sel_hi & 4) << 12)
                | (u32::from(self.clamp) << 15),
            w1,
        ])
    }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        if words.len() < 2 || words[0] >> 23 != 0x1a7 {
            return Err("invalid or truncated VOP3P".into());
        }
        let (w0, w1) = (words[0], words[1]);
        let op =
            Vop3pOp::from_opcode(arch, ((w0 >> 16) & 127) as u8).ok_or("unknown packed opcode")?;
        let (_, a, b, c) = op.widths();
        let mut src = [None; 3];
        for (i, width) in [a, b, c].into_iter().enumerate() {
            if width != 0 {
                src[i] = Some(Src::from_code(((w1 >> (9 * i)) & 511) as u16, None)?);
            }
        }
        let result = Self {
            op,
            dst: Src::Vgpr(w0 as u8),
            src,
            op_sel: ((w0 >> 11) & 7) as u8,
            op_sel_hi: (((w0 >> 12) & 4) | ((w1 >> 27) & 3)) as u8,
            neg_lo: (w1 >> 29) as u8,
            neg_hi: ((w0 >> 8) & 7) as u8,
            clamp: w0 & 0x8000 != 0,
        };
        if result.encoded(arch)? != words[..2] {
            return Err("noncanonical packed encoding".into());
        }
        Ok((result, 2))
    }
    pub fn footprint(&self) -> Footprint {
        let (d, a, b, c) = self.op.widths();
        let mut fp = Footprint::default();
        fp.write(self.dst, d);
        for (s, n) in self.src.iter().zip([a, b, c]) {
            if let Some(s) = s {
                fp.read(*s, n);
            }
        }
        fp.read_file(RegFile::Exec, 0, 2);
        fp
    }
}
impl fmt::Display for Vop3p {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (d, a, b, c) = self.op.widths();
        write!(f, "{} {}", self.op.name(), self.dst.fmt(d))?;
        for (i, width) in [a, b, c].into_iter().enumerate() {
            if let Some(s) = self.src[i] {
                if self.op.mix() && self.neg_lo & (1 << i) != 0 {
                    if s.is_inline() {
                        write!(f, ", neg({})", s.fmt(width))?;
                    } else {
                        write!(f, ", -{}", s.fmt(width))?;
                    }
                } else {
                    write!(f, ", {}", s.fmt(width))?;
                }
            }
        }
        let n = if c == 0 { 2 } else { 3 };
        if self.op_sel != 0 {
            mask_fmt(f, "op_sel", self.op_sel, n)?;
        }
        let default_hi = if self.op.mix() { 0 } else { 7 };
        let active = (1 << n) - 1;
        if self.op_sel_hi & active != default_hi & active {
            mask_fmt(f, "op_sel_hi", self.op_sel_hi, n)?;
        }
        if !self.op.mix() && self.neg_lo != 0 {
            mask_fmt(f, "neg_lo", self.neg_lo, n)?;
        }
        if self.neg_hi != 0 {
            mask_fmt(f, "neg_hi", self.neg_hi, n)?;
        }
        if self.clamp {
            f.write_str(" clamp")?;
        }
        Ok(())
    }
}

impl Mai {
    /// CDNA3 ISA §13.3.6.1 p.540-541: ACC A/B are bits 59/60,
    /// ACC_CD is bit 15. AccVGPR moves use the packed default selector bits.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        out.extend_from_slice(&self.encoded(arch)?);
        Ok(())
    }
    fn encoded(&self, arch: Arch) -> Result<[u32; 2]> {
        let op = self.op.opcode(arch).ok_or("MAI opcode unavailable")?;
        if self.cbsz > 7 || self.abid > 15 || self.blgp > 7 {
            return Err("MAI modifier exceeds field width".into());
        }
        let mut w0 = 0xd3800000 | (u32::from(op) << 16) | (u32::from(self.dst.code()) & 255);
        let w1;
        if let Some(s) = self.op.shape() {
            if !self.dst.is_vgpr() && !self.dst.is_agpr() {
                return Err("MFMA destination must be VGPR or AGPR".into());
            }
            tuple(self.dst, s.cd_dwords, true, false)?;
            let b = self.src_b.ok_or("MFMA missing B")?;
            let c = self.src_c.ok_or("MFMA missing C")?;
            if (!self.src_a.is_vgpr() && !self.src_a.is_agpr()) || (!b.is_vgpr() && !b.is_agpr()) {
                return Err("MFMA A/B must be VGPR or AGPR tuples".into());
            }
            tuple(self.src_a, s.a_dwords, true, false)?;
            tuple(b, s.b_dwords, true, false)?;
            tuple(c, s.cd_dwords, true, false)?;
            if self.op == MaiOp::MfmaF64_16x16x4F64
                && c == Src::Float(crate::operand::InlineFloat::InvTwoPi)
            {
                return Err("F64 MFMA does not support inline 1/(2*pi)".into());
            }
            if !c.is_inline() && c.is_agpr() != self.dst.is_agpr() {
                return Err("MFMA C/D register files differ".into());
            }
            w0 |= (u32::from(self.cbsz) << 8)
                | (u32::from(self.abid) << 11)
                | (u32::from(self.dst.is_agpr()) << 15);
            w1 = u32::from(self.src_a.code())
                | (u32::from(b.code()) << 9)
                | (u32::from(c.code()) << 18)
                | (u32::from(self.src_a.is_agpr()) << 27)
                | (u32::from(b.is_agpr()) << 28)
                | (u32::from(self.blgp) << 29);
        } else {
            if self.src_b.is_some()
                || self.src_c.is_some()
                || self.cbsz != 0
                || self.abid != 0
                || self.blgp != 0
            {
                return Err("AccVGPR move has extra operands/modifiers".into());
            }
            let valid = match self.op {
                MaiOp::AccvgprReadB32 => self.dst.is_vgpr() && self.src_a.is_agpr(),
                MaiOp::AccvgprWriteB32 => {
                    self.dst.is_agpr() && !self.src_a.is_agpr() && !self.src_a.is_literal()
                }
                _ => false,
            };
            if !valid {
                return Err("illegal AccVGPR move register files".into());
            }
            tuple(self.src_a, 1, true, self.op == MaiOp::AccvgprWriteB32)?;
            w0 |= 0x4000;
            w1 = 0x18000000 | u32::from(self.src_a.code());
        }
        Ok([w0, w1])
    }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        if words.len() < 2 || words[0] >> 23 != 0x1a7 {
            return Err("invalid or truncated MAI".into());
        }
        let (w0, w1) = (words[0], words[1]);
        let op = MaiOp::from_opcode(arch, ((w0 >> 16) & 127) as u8).ok_or("unknown MAI opcode")?;
        let convert = |code: u16, acc: bool| -> Result<Src> {
            let s = Src::from_code(code, None)?;
            Ok(if let Src::Vgpr(n) = s {
                if acc {
                    Src::Agpr(n)
                } else {
                    s
                }
            } else {
                s
            })
        };
        let matrix = op.shape().is_some();
        let result = Self {
            op,
            dst: if (matrix && w0 & 0x8000 != 0) || op == MaiOp::AccvgprWriteB32 {
                Src::Agpr(w0 as u8)
            } else {
                Src::Vgpr(w0 as u8)
            },
            src_a: convert(
                (w1 & 511) as u16,
                if matrix {
                    w1 & (1 << 27) != 0
                } else {
                    op == MaiOp::AccvgprReadB32
                },
            )?,
            src_b: if matrix {
                Some(convert(((w1 >> 9) & 511) as u16, w1 & (1 << 28) != 0)?)
            } else {
                None
            },
            src_c: if matrix {
                Some(convert(((w1 >> 18) & 511) as u16, w0 & 0x8000 != 0)?)
            } else {
                None
            },
            cbsz: if matrix { ((w0 >> 8) & 7) as u8 } else { 0 },
            abid: if matrix { ((w0 >> 11) & 15) as u8 } else { 0 },
            blgp: if matrix { (w1 >> 29) as u8 } else { 0 },
        };
        if result.encoded(arch)? != words[..2] {
            return Err("noncanonical MAI encoding".into());
        }
        Ok((result, 2))
    }
    pub fn footprint(&self) -> Footprint {
        let mut fp = Footprint::default();
        let (a, b, cd) = self
            .op
            .shape()
            .map_or((1, 0, 1), |s| (s.a_dwords, s.b_dwords, s.cd_dwords));
        fp.write(self.dst, cd);
        fp.read(self.src_a, a);
        if let Some(s) = self.src_b {
            fp.read(s, b);
        }
        if let Some(s) = self.src_c {
            fp.read(s, cd);
        }
        fp.read_file(RegFile::Exec, 0, 2);
        fp
    }
}
impl fmt::Display for Mai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b, cd) = self
            .op
            .shape()
            .map_or((1, 0, 1), |s| (s.a_dwords, s.b_dwords, s.cd_dwords));
        let name = if self.op == MaiOp::MfmaF64_16x16x4F64 {
            "v_mfma_f64_16x16x4_f64"
        } else {
            self.op.name()
        };
        write!(f, "{name} {}, {}", self.dst.fmt(cd), self.src_a.fmt(a))?;
        if let Some(s) = self.src_b {
            write!(f, ", {}", s.fmt(b))?;
        }
        if let Some(s) = self.src_c {
            write!(f, ", {}", s.fmt(cd))?;
        }
        if self.cbsz != 0 {
            write!(f, " cbsz:{}", self.cbsz)?;
        }
        if self.abid != 0 {
            write!(f, " abid:{}", self.abid)?;
        }
        if self.blgp != 0 {
            if self.op == MaiOp::MfmaF64_16x16x4F64 {
                mask_fmt(f, "neg", self.blgp, 3)?;
            } else {
                write!(f, " blgp:{}", self.blgp)?;
            }
        }
        Ok(())
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP3P); proven by the
// llvm-mc round-trip tests.
impl Vop3pOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::PkFmaF16 => 14,
                Self::PkAddF16 => 15,
                Self::PkMulF16 => 16,
                Self::PkMinF16 => 17,
                Self::PkMaxF16 => 18,
                Self::PkAddU16 => 10,
                Self::PkSubU16 => 11,
                Self::PkMulLoU16 => 1,
                Self::PkLshlrevB16 => 4,
                Self::PkLshrrevB16 => 5,
                Self::PkMaxI16 => 7,
                Self::PkMinI16 => 8,
                Self::FmaMixF32 => 32,
                Self::FmaMixloF16 => 33,
                Self::FmaMixhiF16 => 34,
                Self::Dot2F32F16 => 35,
                Self::Dot2I32I16 => 38,
                Self::Dot2U32U16 => 39,
                Self::Dot4I32I8 => 40,
                Self::Dot4U32U8 => 41,
                Self::Dot8I32I4 => 42,
                Self::Dot8U32U4 => 43,
                Self::PkFmaF32 => 48,
                Self::PkMulF32 => 49,
                Self::PkAddF32 => 50,
                Self::PkMovB32 => 51,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::PkFmaF16 => "v_pk_fma_f16",
            Self::PkAddF16 => "v_pk_add_f16",
            Self::PkMulF16 => "v_pk_mul_f16",
            Self::PkMinF16 => "v_pk_min_f16",
            Self::PkMaxF16 => "v_pk_max_f16",
            Self::PkAddU16 => "v_pk_add_u16",
            Self::PkSubU16 => "v_pk_sub_u16",
            Self::PkMulLoU16 => "v_pk_mul_lo_u16",
            Self::PkLshlrevB16 => "v_pk_lshlrev_b16",
            Self::PkLshrrevB16 => "v_pk_lshrrev_b16",
            Self::PkMaxI16 => "v_pk_max_i16",
            Self::PkMinI16 => "v_pk_min_i16",
            Self::FmaMixF32 => "v_fma_mix_f32",
            Self::FmaMixloF16 => "v_fma_mixlo_f16",
            Self::FmaMixhiF16 => "v_fma_mixhi_f16",
            Self::Dot2F32F16 => "v_dot2_f32_f16",
            Self::Dot2I32I16 => "v_dot2_i32_i16",
            Self::Dot2U32U16 => "v_dot2_u32_u16",
            Self::Dot4I32I8 => "v_dot4_i32_i8",
            Self::Dot4U32U8 => "v_dot4_u32_u8",
            Self::Dot8I32I4 => "v_dot8_i32_i4",
            Self::Dot8U32U4 => "v_dot8_u32_u4",
            Self::PkFmaF32 => "v_pk_fma_f32",
            Self::PkMulF32 => "v_pk_mul_f32",
            Self::PkAddF32 => "v_pk_add_f32",
            Self::PkMovB32 => "v_pk_mov_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding VOP3P_MFMA / ENC_VOP3P); proven by the
// llvm-mc round-trip tests.
impl MaiOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::AccvgprReadB32 => 88,
                Self::AccvgprWriteB32 => 89,
                Self::MfmaF32_16x16x16Bf16 => 97,
                Self::MfmaF32_32x32x8Bf16 => 96,
                Self::MfmaF32_16x16x16F16 => 77,
                Self::MfmaF32_32x32x8F16 => 76,
                Self::MfmaF32_16x16x32Fp8Fp8 => 115,
                Self::MfmaF32_16x16x32Fp8Bf8 => 114,
                Self::MfmaF32_16x16x32Bf8Fp8 => 113,
                Self::MfmaF32_16x16x32Bf8Bf8 => 112,
                Self::MfmaF32_32x32x16Fp8Fp8 => 119,
                Self::MfmaF32_32x32x16Fp8Bf8 => 118,
                Self::MfmaF32_32x32x16Bf8Fp8 => 117,
                Self::MfmaF32_32x32x16Bf8Bf8 => 116,
                Self::MfmaI32_16x16x32I8 => 87,
                Self::MfmaI32_32x32x16I8 => 86,
                Self::MfmaF32_16x16x4F32 => 69,
                Self::MfmaF32_32x32x2F32 => 68,
                Self::MfmaF64_16x16x4F64 => 110,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::AccvgprReadB32 => "v_accvgpr_read_b32",
            Self::AccvgprWriteB32 => "v_accvgpr_write_b32",
            Self::MfmaF32_16x16x16Bf16 => "v_mfma_f32_16x16x16_bf16",
            Self::MfmaF32_32x32x8Bf16 => "v_mfma_f32_32x32x8_bf16",
            Self::MfmaF32_16x16x16F16 => "v_mfma_f32_16x16x16_f16",
            Self::MfmaF32_32x32x8F16 => "v_mfma_f32_32x32x8_f16",
            Self::MfmaF32_16x16x32Fp8Fp8 => "v_mfma_f32_16x16x32_fp8_fp8",
            Self::MfmaF32_16x16x32Fp8Bf8 => "v_mfma_f32_16x16x32_fp8_bf8",
            Self::MfmaF32_16x16x32Bf8Fp8 => "v_mfma_f32_16x16x32_bf8_fp8",
            Self::MfmaF32_16x16x32Bf8Bf8 => "v_mfma_f32_16x16x32_bf8_bf8",
            Self::MfmaF32_32x32x16Fp8Fp8 => "v_mfma_f32_32x32x16_fp8_fp8",
            Self::MfmaF32_32x32x16Fp8Bf8 => "v_mfma_f32_32x32x16_fp8_bf8",
            Self::MfmaF32_32x32x16Bf8Fp8 => "v_mfma_f32_32x32x16_bf8_fp8",
            Self::MfmaF32_32x32x16Bf8Bf8 => "v_mfma_f32_32x32x16_bf8_bf8",
            Self::MfmaI32_16x16x32I8 => "v_mfma_i32_16x16x32_i8",
            Self::MfmaI32_32x32x16I8 => "v_mfma_i32_32x32x16_i8",
            Self::MfmaF32_16x16x4F32 => "v_mfma_f32_16x16x4_f32",
            Self::MfmaF32_32x32x2F32 => "v_mfma_f32_32x32x2_f32",
            Self::MfmaF64_16x16x4F64 => "v_mfma_f64_16x16x4_f64",
        }
    }
}
