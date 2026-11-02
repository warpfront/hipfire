// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The CDNA3 operand space.
//!
//! Scalar sources and destinations share the 8-bit SSRC / 7-bit SDST code
//! space of CDNA3 ISA §5.2 Table 13 (p.25-26); vector sources extend it to 9
//! bits with VGPRs at 256..=511 (CDNA3 ISA §6.2, §13 VOP formats). AccVGPRs
//! are not in that code space: the MAI encoding (and the memory formats' ACC
//! bit, unused here) select them with a separate bit, so [`Src::Agpr`] encodes
//! as `256 + n` and the format sets its ACC bit.
//!
//! A multi-dword operand is named by its first register; the width comes from
//! the opcode, so printing takes it as an argument ([`Src::fmt`]).
use crate::Result;

/// The nine inline floating-point constants, codes 240..=248. Their bit
/// pattern depends on the operand type of the instruction that reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InlineFloat {
    Half,
    NegHalf,
    One,
    NegOne,
    Two,
    NegTwo,
    Four,
    NegFour,
    InvTwoPi,
}

impl InlineFloat {
    pub const ALL: [InlineFloat; 9] = [
        Self::Half,
        Self::NegHalf,
        Self::One,
        Self::NegOne,
        Self::Two,
        Self::NegTwo,
        Self::Four,
        Self::NegFour,
        Self::InvTwoPi,
    ];
    pub fn code(self) -> u16 {
        240 + self as u16
    }
    pub fn from_code(code: u16) -> Option<Self> {
        Self::ALL.get(usize::from(code.checked_sub(240)?)).copied()
    }
    /// Bits when the instruction reads a 32-bit float operand (and the bit
    /// pattern an integer or untyped 32-bit operand receives).
    pub fn f32_bits(self) -> u32 {
        [
            0x3f00_0000,
            0xbf00_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x4000_0000,
            0xc000_0000,
            0x4080_0000,
            0xc080_0000,
            0x3e22_f983,
        ][self as usize]
    }
    /// Bits when the instruction reads a 16-bit float operand.
    pub fn f16_bits(self) -> u16 {
        [
            0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4400, 0xc400, 0x3118,
        ][self as usize]
    }
    /// Bits when the instruction reads a 64-bit float operand.
    pub fn f64_bits(self) -> u64 {
        [
            0x3fe0_0000_0000_0000,
            0xbfe0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xc000_0000_0000_0000,
            0x4010_0000_0000_0000,
            0xc010_0000_0000_0000,
            0x3fc4_5f30_6dc9_c882,
        ][self as usize]
    }
    /// llvm-mc spelling.
    pub fn text(self) -> &'static str {
        [
            "0.5",
            "-0.5",
            "1.0",
            "-1.0",
            "2.0",
            "-2.0",
            "4.0",
            "-4.0",
            "0.15915494",
        ][self as usize]
    }
}

/// One operand field value. Destinations use the same type (a VGPR
/// destination is `Vgpr(n)`, an SGPR destination `Sgpr(n)`, `VccLo`, ...);
/// each format's encoder rejects kinds its field cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
    /// s0..=s101, codes 0..=101.
    Sgpr(u8),
    FlatScratchLo,
    FlatScratchHi,
    XnackMaskLo,
    XnackMaskHi,
    /// Codes 106/107; a 64-bit operand starting at `VccLo` is `vcc`.
    VccLo,
    VccHi,
    /// ttmp0..=ttmp15, codes 108..=123.
    Ttmp(u8),
    /// Code 124.
    M0,
    /// Codes 126/127; a 64-bit operand starting at `ExecLo` is `exec`.
    ExecLo,
    ExecHi,
    /// Inline integer -16..=64 (codes 128..=208).
    Int(i8),
    SharedBase,
    SharedLimit,
    PrivateBase,
    PrivateLimit,
    /// Inline float, codes 240..=248.
    Float(InlineFloat),
    /// `src_vccz`, `src_execz`, `src_scc`: codes 251..=253.
    Vccz,
    Execz,
    Scc,
    /// Code 255: a 32-bit literal dword follows the instruction.
    Lit(u32),
    /// v0..=v255, codes 256..=511.
    Vgpr(u8),
    /// a0..=a255: code `256 + n` with the format's ACC bit set.
    Agpr(u8),
}

/// The literal-dword code.
pub const LITERAL: u16 = 255;

impl Src {
    /// The field code (9-bit space; SDST fields use the low 7 bits of a
    /// scalar register code). `Agpr(n)` returns `256 + n`.
    pub fn code(self) -> u16 {
        match self {
            Self::Sgpr(n) => u16::from(n),
            Self::FlatScratchLo => 102,
            Self::FlatScratchHi => 103,
            Self::XnackMaskLo => 104,
            Self::XnackMaskHi => 105,
            Self::VccLo => 106,
            Self::VccHi => 107,
            Self::Ttmp(n) => 108 + u16::from(n),
            Self::M0 => 124,
            Self::ExecLo => 126,
            Self::ExecHi => 127,
            Self::Int(i) if i >= 0 => 128 + i as u16,
            Self::Int(i) => 192 + i.unsigned_abs() as u16,
            Self::SharedBase => 235,
            Self::SharedLimit => 236,
            Self::PrivateBase => 237,
            Self::PrivateLimit => 238,
            Self::Float(f) => f.code(),
            Self::Vccz => 251,
            Self::Execz => 252,
            Self::Scc => 253,
            Self::Lit(_) => LITERAL,
            Self::Vgpr(n) | Self::Agpr(n) => 256 + u16::from(n),
        }
    }

    /// Decode a field code. `literal` is the dword after the instruction
    /// when the format carries one (`None` makes code 255 an error). Codes
    /// 256..=511 decode as `Vgpr`; formats with an ACC bit convert to `Agpr`.
    pub fn from_code(code: u16, literal: Option<u32>) -> Result<Self> {
        Ok(match code {
            0..=101 => Self::Sgpr(code as u8),
            102 => Self::FlatScratchLo,
            103 => Self::FlatScratchHi,
            104 => Self::XnackMaskLo,
            105 => Self::XnackMaskHi,
            106 => Self::VccLo,
            107 => Self::VccHi,
            108..=123 => Self::Ttmp((code - 108) as u8),
            124 => Self::M0,
            126 => Self::ExecLo,
            127 => Self::ExecHi,
            128..=192 => Self::Int((code - 128) as i8),
            193..=208 => Self::Int(-((code - 192) as i8)),
            235 => Self::SharedBase,
            236 => Self::SharedLimit,
            237 => Self::PrivateBase,
            238 => Self::PrivateLimit,
            240..=248 => Self::Float(InlineFloat::from_code(code).expect("240..=248")),
            251 => Self::Vccz,
            252 => Self::Execz,
            253 => Self::Scc,
            LITERAL => Self::Lit(literal.ok_or("literal operand without a literal dword")?),
            256..=511 => Self::Vgpr((code - 256) as u8),
            _ => return Err(format!("reserved operand code {code}")),
        })
    }

    pub fn is_vgpr(self) -> bool {
        matches!(self, Self::Vgpr(_))
    }
    pub fn is_agpr(self) -> bool {
        matches!(self, Self::Agpr(_))
    }
    pub fn is_literal(self) -> bool {
        matches!(self, Self::Lit(_))
    }
    pub fn is_inline(self) -> bool {
        matches!(self, Self::Int(_) | Self::Float(_))
    }
    /// A scalar register (codes 0..=127): legal in a 7-bit SDST field.
    pub fn is_sreg(self) -> bool {
        !self.is_vgpr() && !self.is_agpr() && self.code() <= 127
    }
    /// The literal dword this operand appends, if any.
    pub fn literal(self) -> Option<u32> {
        if let Self::Lit(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// The operand as llvm-mc/llvm-objdump print it for a `dwords`-wide
    /// operand (1 = 32-bit, 2 = 64-bit pair, 4 = quad, ...). Literals print
    /// as `0x…` hex of the dword; inline constants ignore `dwords`.
    pub fn fmt(self, dwords: u8) -> String {
        let range = |p: &str, n: u16| {
            if dwords <= 1 {
                format!("{p}{n}")
            } else {
                format!("{p}[{n}:{}]", n + u16::from(dwords) - 1)
            }
        };
        match (self, dwords) {
            (Self::Sgpr(n), _) => range("s", u16::from(n)),
            (Self::Vgpr(n), _) => range("v", u16::from(n)),
            (Self::Agpr(n), _) => range("a", u16::from(n)),
            (Self::Ttmp(n), _) => range("ttmp", u16::from(n)),
            (Self::VccLo, 2) => "vcc".into(),
            (Self::ExecLo, 2) => "exec".into(),
            (Self::FlatScratchLo, 2) => "flat_scratch".into(),
            (Self::XnackMaskLo, 2) => "xnack_mask".into(),
            (Self::VccLo, _) => "vcc_lo".into(),
            (Self::VccHi, _) => "vcc_hi".into(),
            (Self::ExecLo, _) => "exec_lo".into(),
            (Self::ExecHi, _) => "exec_hi".into(),
            (Self::FlatScratchLo, _) => "flat_scratch_lo".into(),
            (Self::FlatScratchHi, _) => "flat_scratch_hi".into(),
            (Self::XnackMaskLo, _) => "xnack_mask_lo".into(),
            (Self::XnackMaskHi, _) => "xnack_mask_hi".into(),
            (Self::M0, _) => "m0".into(),
            (Self::Int(i), _) => i.to_string(),
            (Self::SharedBase, _) => "src_shared_base".into(),
            (Self::SharedLimit, _) => "src_shared_limit".into(),
            (Self::PrivateBase, _) => "src_private_base".into(),
            (Self::PrivateLimit, _) => "src_private_limit".into(),
            (Self::Float(f), _) => f.text().into(),
            (Self::Vccz, _) => "src_vccz".into(),
            (Self::Execz, _) => "src_execz".into(),
            (Self::Scc, _) => "src_scc".into(),
            (Self::Lit(v), _) => format!("{v:#x}"),
        }
    }
}

/// The 32-bit value an integer/untyped operand reads, for operands whose
/// value does not depend on wave state (inline constants and literals).
pub fn constant_bits(src: Src) -> Option<u32> {
    match src {
        Src::Int(i) => Some(i32::from(i) as u32),
        Src::Float(f) => Some(f.f32_bits()),
        Src::Lit(v) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_code_round_trips() {
        for code in 0u16..512 {
            if let Ok(src) = Src::from_code(code, Some(0x1234_5678)) {
                assert_eq!(src.code(), code, "{src:?}");
            }
        }
        assert_eq!(Src::from_code(193, None).unwrap(), Src::Int(-1));
        assert_eq!(Src::from_code(208, None).unwrap(), Src::Int(-16));
        assert_eq!(Src::from_code(192, None).unwrap(), Src::Int(64));
        assert!(Src::from_code(125, None).is_err());
        assert!(Src::from_code(255, None).is_err());
    }
    #[test]
    fn printing() {
        assert_eq!(Src::Sgpr(4).fmt(2), "s[4:5]");
        assert_eq!(Src::VccLo.fmt(2), "vcc");
        assert_eq!(Src::Vgpr(10).fmt(4), "v[10:13]");
        assert_eq!(Src::Agpr(0).fmt(1), "a0");
        assert_eq!(Src::Lit(0x3e80_0000).fmt(1), "0x3e800000");
    }
}
