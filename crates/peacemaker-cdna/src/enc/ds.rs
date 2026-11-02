// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Data share format DS (CDNA3 ISA §11, §13.4.1 Table 94 p.545). gfx9
//! spellings: `ds_read_*` / `ds_write_*`.
//!
//! Layout (CDNA3 ISA §13.4.1 Table 94 p.545), two dwords: `OFFSET0` `[7:0]`,
//! `OFFSET1` `[15:8]`, `GDS` `[16]`, `OP` `[24:17]`, `ACC` `[25]`, `ENCODING`
//! `[31:26]` = `110110`; `ADDR` `[39:32]`, `DATA0` `[47:40]`, `DATA1`
//! `[55:48]`, `VDST` `[63:56]`.
//!
//! Decisions, each proven against `llvm-mc` (gfx942) by `tests/enc_ds.rs`:
//! - `GDS` must be 0: llvm-mc rejects the `gds` modifier on gfx942 and the
//!   gfx942 disassembler prints a set GDS bit as nothing at all, so no text
//!   round-trips it. `GDS=1` means GWS (Table 94), and no GWS opcode is in
//!   [`DsOp`]. [`Ds::encode`] and [`Ds::decode`] return `Err` for it.
//! - `ACC` (gfx90a+: VDST/DATA0/DATA1 name AccVGPRs) is left 0; [`Src::Agpr`]
//!   operands are rejected and a set ACC bit is a decode error.
//! - Fields an opcode does not use must be 0 (the disassembler rejects the
//!   bytes otherwise), so `decode` followed by `encode` is the identity.
//! - 64/96/128-bit VGPR tuples (`vdst`, `data0`, `data1` of width >= 2) must
//!   start at an even VGPR on gfx90a+ (llvm-mc: register class
//!   `AV_*_Align2`) and end at or below v255.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind, RegFile};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(DsOp {
    ReadB32,
    ReadB64,
    ReadB96,
    ReadB128,
    Read2B32,
    Read2B64,
    Read2st64B32,
    Read2st64B64,
    ReadU8,
    ReadI8,
    ReadU16,
    ReadI16,
    WriteB8,
    WriteB16,
    WriteB32,
    WriteB64,
    WriteB96,
    WriteB128,
    Write2B32,
    Write2B64,
    Write2st64B32,
    Write2st64B64,
    AddU32,
    AddRtnU32,
    MaxU32,
    MaxI32,
    MaxF32,
    MinF32,
    AddF32,
    SwizzleB32,
    PermuteB32,
    BpermuteB32,
    Nop,
});

/// `ENCODING` field `[31:26]` = `110110` (CDNA3 ISA §13.4.1 Table 94 p.545).
const ENC_DS: u32 = 0b11_0110 << 26;

impl DsOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Dword widths of (vdst, data0, data1); 0 where absent. Sub-dword ops
    /// (`u8`/`i8`/`u16`/`i16`/`b8`/`b16`) occupy one VGPR; `read2*` return
    /// two elements (`vdst` = 2 dwords for `b32`, 4 for `b64`); `write2*`
    /// take two data operands of one element each.
    pub fn widths(self) -> (u8, u8, u8) {
        match self {
            Self::ReadB32 | Self::ReadU8 | Self::ReadI8 | Self::ReadU16 | Self::ReadI16 => {
                (1, 0, 0)
            }
            Self::ReadB64 | Self::Read2B32 | Self::Read2st64B32 => (2, 0, 0),
            Self::ReadB96 => (3, 0, 0),
            Self::ReadB128 | Self::Read2B64 | Self::Read2st64B64 => (4, 0, 0),
            Self::WriteB8 | Self::WriteB16 | Self::WriteB32 => (0, 1, 0),
            Self::WriteB64 => (0, 2, 0),
            Self::WriteB96 => (0, 3, 0),
            Self::WriteB128 => (0, 4, 0),
            Self::Write2B32 | Self::Write2st64B32 => (0, 1, 1),
            Self::Write2B64 | Self::Write2st64B64 => (0, 2, 2),
            Self::AddU32
            | Self::MaxU32
            | Self::MaxI32
            | Self::MaxF32
            | Self::MinF32
            | Self::AddF32 => (0, 1, 0),
            Self::AddRtnU32 => (1, 1, 0),
            Self::SwizzleB32 => (1, 0, 0),
            Self::PermuteB32 | Self::BpermuteB32 => (1, 1, 0),
            Self::Nop => (0, 0, 0),
        }
    }
    /// Two-address forms (`ds_read2*`, `ds_write2*`): `offset0`/`offset1`
    /// are separate 8-bit offsets in units of the element size (×64 for
    /// `st64`); otherwise one 16-bit byte offset `offset1:offset0`.
    /// (CDNA3 ISA §11.3.1 Table 56 p.86.)
    pub fn two_addr(self) -> bool {
        matches!(
            self,
            Self::Read2B32
                | Self::Read2B64
                | Self::Read2st64B32
                | Self::Read2st64B64
                | Self::Write2B32
                | Self::Write2B64
                | Self::Write2st64B32
                | Self::Write2st64B64
        )
    }
    /// Whether the op takes an `ADDR` VGPR (everything but `ds_nop`).
    pub fn has_addr(self) -> bool {
        self != Self::Nop
    }
    /// Whether the op reads or writes LDS memory. `ds_swizzle_b32` "does not
    /// read or write the DS memory banks", `ds_permute_b32` /
    /// `ds_bpermute_b32` "do not access LDS memory" (CDNA3 ISA §12.12
    /// p.424-427); `ds_nop` does nothing (p.415).
    pub fn accesses_lds(self) -> bool {
        !matches!(
            self,
            Self::SwizzleB32 | Self::PermuteB32 | Self::BpermuteB32 | Self::Nop
        )
    }
    /// The wait-counter behaviour (all DS ops count on LGKM_CNT, CDNA3 ISA
    /// §4.4 p.20). `ds_nop` is classed [`MemKind::DsStore`]: it is a DS
    /// instruction that returns no data. [UNVERIFIED: confirm on MI300X that
    /// `ds_nop` moves LGKM_CNT like other DS instructions.]
    pub fn mem_kind(self) -> MemKind {
        match self {
            Self::ReadB32
            | Self::ReadB64
            | Self::ReadB96
            | Self::ReadB128
            | Self::Read2B32
            | Self::Read2B64
            | Self::Read2st64B32
            | Self::Read2st64B64
            | Self::ReadU8
            | Self::ReadI8
            | Self::ReadU16
            | Self::ReadI16 => MemKind::DsLoad,
            Self::WriteB8
            | Self::WriteB16
            | Self::WriteB32
            | Self::WriteB64
            | Self::WriteB96
            | Self::WriteB128
            | Self::Write2B32
            | Self::Write2B64
            | Self::Write2st64B32
            | Self::Write2st64B64
            | Self::Nop => MemKind::DsStore,
            Self::AddU32
            | Self::MaxU32
            | Self::MaxI32
            | Self::MaxF32
            | Self::MinF32
            | Self::AddF32 => MemKind::DsAtomic,
            Self::AddRtnU32 => MemKind::DsAtomicRtn,
            Self::SwizzleB32 | Self::PermuteB32 | Self::BpermuteB32 => MemKind::DsXlane,
        }
    }
}

/// `addr` is the VGPR address (`None` for `ds_nop`), `data0`/`data1` the
/// data VGPRs, `vdst` the returned VGPRs.
#[derive(Clone, Debug, PartialEq)]
pub struct Ds {
    pub op: DsOp,
    pub vdst: Option<Src>,
    pub addr: Option<Src>,
    pub data0: Option<Src>,
    pub data1: Option<Src>,
    pub offset0: u8,
    pub offset1: u8,
    pub gds: bool,
}

impl Ds {
    /// The 16-bit byte offset of single-address forms.
    pub fn offset(&self) -> u16 {
        u16::from(self.offset1) << 8 | u16::from(self.offset0)
    }

    /// `(offset0, offset1)` of a single-address op's 16-bit byte offset;
    /// `Err` above 0xffff (the field is `{OFFSET1, OFFSET0}`, CDNA3 ISA
    /// §11.3.1 Table 56 p.86).
    pub fn split_offset(offset: u32) -> Result<(u8, u8)> {
        let o = u16::try_from(offset)
            .map_err(|_| format!("DS byte offset {offset} exceeds the 16-bit field (max 65535)"))?;
        Ok((o as u8, (o >> 8) as u8))
    }

    /// `(offset0, offset1)` of a two-address op; each is an 8-bit unsigned
    /// offset in units of the element size (CDNA3 ISA §11.3.1 p.86); `Err`
    /// above 255.
    pub fn pair_offsets(offset0: u32, offset1: u32) -> Result<(u8, u8)> {
        let a = u8::try_from(offset0)
            .map_err(|_| format!("DS offset0 {offset0} exceeds the 8-bit field (max 255)"))?;
        let b = u8::try_from(offset1)
            .map_err(|_| format!("DS offset1 {offset1} exceeds the 8-bit field (max 255)"))?;
        Ok((a, b))
    }

    /// Validate the fields against `arch` and the op's operand shape; return
    /// the opcode.
    fn check(&self, arch: Arch) -> Result<u8> {
        let name = self.op.name();
        let opcode = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{name} does not exist on {arch:?}"))?;
        if self.gds {
            return Err(format!("{name}: the GDS bit (GWS) is not supported on {arch:?}; llvm-mc rejects the `gds` modifier there"));
        }
        let (wv, w0, w1) = self.op.widths();
        let wa = u8::from(self.op.has_addr());
        for (role, field, width) in [
            ("vdst", self.vdst, wv),
            ("addr", self.addr, wa),
            ("data0", self.data0, w0),
            ("data1", self.data1, w1),
        ] {
            match (field, width) {
                (None, 0) => {}
                (Some(_), 0) => return Err(format!("{name}: takes no {role} operand")),
                (None, _) => return Err(format!("{name}: missing {role} operand")),
                (Some(Src::Vgpr(n)), w) => {
                    if w >= 2 && n % 2 != 0 {
                        return Err(format!("{name}: {role} v{n} must be even-aligned for a {w}-dword tuple (gfx90a+)"));
                    }
                    if u16::from(n) + u16::from(w) > 256 {
                        return Err(format!(
                            "{name}: {role} v[{n}:{}] runs past v255",
                            u16::from(n) + u16::from(w) - 1
                        ));
                    }
                }
                (Some(Src::Agpr(_)), _) => {
                    return Err(format!(
                        "{name}: {role} AccVGPR operands (ACC bit) are not supported"
                    ))
                }
                (Some(other), _) => {
                    return Err(format!("{name}: {role} must be a VGPR, got {other:?}"))
                }
            }
        }
        if self.op == DsOp::Nop && (self.offset0 | self.offset1) != 0 {
            return Err("ds_nop takes no offset".into());
        }
        Ok(opcode)
    }

    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let opcode = self.check(arch)?;
        let field = |s: Option<Src>| s.map_or(0, |s| u32::from(s.code() as u8));
        out.push(
            ENC_DS
                | u32::from(opcode) << 17
                | u32::from(self.offset1) << 8
                | u32::from(self.offset0),
        );
        out.push(
            field(self.addr)
                | field(self.data0) << 8
                | field(self.data1) << 16
                | field(self.vdst) << 24,
        );
        Ok(())
    }

    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let &[w0, w1, ..] = words else {
            return Err("DS instruction needs two dwords".into());
        };
        if w0 & 0xfc00_0000 != ENC_DS {
            return Err(format!("{w0:#010x} is not a DS encoding"));
        }
        if w0 >> 25 & 1 != 0 {
            return Err(format!(
                "{w0:#010x}: DS ACC bit set (AccVGPR operands unsupported)"
            ));
        }
        let opcode = (w0 >> 17 & 0xff) as u8;
        let op = DsOp::from_opcode(arch, opcode).ok_or_else(|| {
            format!("DS opcode {opcode} is not in the PeaceMaker CDNA table for {arch:?}")
        })?;
        let (wv, d0, d1) = op.widths();
        let (addr, data0, data1, vdst) = (w1 & 0xff, w1 >> 8 & 0xff, w1 >> 16 & 0xff, w1 >> 24);
        let pick = |value: u32, width: u8, role: &str| -> Result<Option<Src>> {
            match (width, value) {
                (0, 0) => Ok(None),
                (0, v) => Err(format!(
                    "{}: unused {role} field is {v}, must be 0",
                    op.name()
                )),
                (_, v) => Ok(Some(Src::Vgpr(v as u8))),
            }
        };
        let ds = Self {
            op,
            vdst: pick(vdst, wv, "vdst")?,
            addr: pick(addr, u8::from(op.has_addr()), "addr")?,
            data0: pick(data0, d0, "data0")?,
            data1: pick(data1, d1, "data1")?,
            offset0: w0 as u8,
            offset1: (w0 >> 8) as u8,
            gds: w0 >> 16 & 1 != 0,
        };
        ds.check(arch)?;
        Ok((ds, 2))
    }

    /// EXEC (every DS op except `ds_nop` is lane-masked), the `addr` and
    /// data VGPRs read, the `vdst` VGPRs written, and M0 for the ops that
    /// touch LDS memory: "All LDS operations require that M0 be initialized
    /// prior to use. M0 contains a size value that can be used to restrict
    /// access to a subset of the allocated LDS range" (CDNA3 ISA §11.3.1
    /// Table 56 p.86, "Implied use of M0"). `ds_swizzle_b32`,
    /// `ds_permute_b32`, `ds_bpermute_b32` access no LDS memory (p.424-427)
    /// and `ds_nop` does nothing (p.415), so they record no M0 use.
    /// [UNVERIFIED: confirm on MI300X whether gfx9-family hardware really
    /// consumes M0 for plain LDS ops; LLVM does not initialise it for them.]
    pub fn footprint(&self) -> Footprint {
        let mut fp = Footprint::default();
        if self.op == DsOp::Nop {
            return fp;
        }
        let (wv, w0, w1) = self.op.widths();
        fp.read_file(RegFile::Exec, 0, 2);
        if self.op.accesses_lds() {
            fp.read_file(RegFile::M0, 0, 1);
        }
        if let Some(a) = self.addr {
            fp.read(a, 1)
        }
        if let Some(d) = self.data0 {
            fp.read(d, w0)
        }
        if let Some(d) = self.data1 {
            fp.read(d, w1)
        }
        if let Some(d) = self.vdst {
            fp.write(d, wv)
        }
        fp
    }

    pub fn mem_kind(&self) -> MemKind {
        self.op.mem_kind()
    }
}

/// `offset:swizzle(...)` as llvm's `AMDGPUInstPrinter::printSwizzle` prints
/// the 16-bit `ds_swizzle_b32` offset (`AMDGPU::Swizzle` constants:
/// rotate/FFT modes from 0xc000/0xe000, quad-perm `0x8000 | lanes`, bitmask
/// perm `AND[4:0] OR[9:5] XOR[14:10]`); empty for offset 0.
fn swizzle_offset(imm: u16) -> String {
    if imm == 0 {
        return String::new();
    }
    if imm >= 0xc000 {
        return if imm >= 0xe000 {
            format!(" offset:swizzle(FFT,{})", imm & 0x1f)
        } else {
            format!(
                " offset:swizzle(ROTATE,{},{})",
                imm >> 10 & 1,
                imm >> 5 & 0x1f
            )
        };
    }
    if imm & 0xff00 == 0x8000 {
        let lane = |i: u32| imm >> (2 * i) & 3;
        return format!(
            " offset:swizzle(QUAD_PERM,{},{},{},{})",
            lane(0),
            lane(1),
            lane(2),
            lane(3)
        );
    }
    if imm & 0x8000 == 0 {
        let (and, or, xor) = (imm & 0x1f, imm >> 5 & 0x1f, imm >> 10 & 0x1f);
        if and == 0x1f && or == 0 && xor.count_ones() == 1 {
            return format!(" offset:swizzle(SWAP,{xor})");
        }
        if and == 0x1f && or == 0 && xor > 0 && (xor + 1).is_power_of_two() {
            return format!(" offset:swizzle(REVERSE,{})", xor + 1);
        }
        let group = 0x1f - and + 1;
        if group > 1 && group.is_power_of_two() && or < group && xor == 0 {
            return format!(" offset:swizzle(BROADCAST,{group},{or})");
        }
        let probe0 = or ^ xor;
        let probe1 = (0x1f & and | or) ^ xor;
        let mask: String = (0..5)
            .rev()
            .map(|b| {
                let (p0, p1) = (probe0 >> b & 1, probe1 >> b & 1);
                match (p0 == p1, p0) {
                    (true, 0) => '0',
                    (true, _) => '1',
                    (false, 0) => 'p',
                    (false, _) => 'i',
                }
            })
            .collect();
        return format!(" offset:swizzle(BITMASK_PERM,\"{mask}\")");
    }
    format!(" offset:{imm}")
}

impl fmt::Display for Ds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wv, w0, w1) = self.op.widths();
        f.write_str(self.op.name())?;
        let mut sep = " ";
        for (src, width) in [
            (self.vdst, wv),
            (self.addr, 1),
            (self.data0, w0),
            (self.data1, w1),
        ] {
            if let Some(s) = src {
                write!(f, "{sep}{}", s.fmt(width))?;
                sep = ", ";
            }
        }
        if self.op.two_addr() {
            if self.offset0 != 0 {
                write!(f, " offset0:{}", self.offset0)?
            }
            if self.offset1 != 0 {
                write!(f, " offset1:{}", self.offset1)?
            }
        } else if self.op == DsOp::SwizzleB32 {
            f.write_str(&swizzle_offset(self.offset()))?;
        } else if self.offset() != 0 {
            write!(f, " offset:{}", self.offset())?;
        }
        if self.gds {
            f.write_str(" gds")?;
        }
        Ok(())
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_DS); proven by the
// llvm-mc round-trip tests.
impl DsOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::ReadB32 => 54,
                Self::ReadB64 => 118,
                Self::ReadB96 => 254,
                Self::ReadB128 => 255,
                Self::Read2B32 => 55,
                Self::Read2B64 => 119,
                Self::Read2st64B32 => 56,
                Self::Read2st64B64 => 120,
                Self::ReadU8 => 58,
                Self::ReadI8 => 57,
                Self::ReadU16 => 60,
                Self::ReadI16 => 59,
                Self::WriteB8 => 30,
                Self::WriteB16 => 31,
                Self::WriteB32 => 13,
                Self::WriteB64 => 77,
                Self::WriteB96 => 222,
                Self::WriteB128 => 223,
                Self::Write2B32 => 14,
                Self::Write2B64 => 78,
                Self::Write2st64B32 => 15,
                Self::Write2st64B64 => 79,
                Self::AddU32 => 0,
                Self::AddRtnU32 => 32,
                Self::MaxU32 => 8,
                Self::MaxI32 => 6,
                Self::MaxF32 => 19,
                Self::MinF32 => 18,
                Self::AddF32 => 21,
                Self::SwizzleB32 => 61,
                Self::PermuteB32 => 62,
                Self::BpermuteB32 => 63,
                Self::Nop => 20,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadB32 => "ds_read_b32",
            Self::ReadB64 => "ds_read_b64",
            Self::ReadB96 => "ds_read_b96",
            Self::ReadB128 => "ds_read_b128",
            Self::Read2B32 => "ds_read2_b32",
            Self::Read2B64 => "ds_read2_b64",
            Self::Read2st64B32 => "ds_read2st64_b32",
            Self::Read2st64B64 => "ds_read2st64_b64",
            Self::ReadU8 => "ds_read_u8",
            Self::ReadI8 => "ds_read_i8",
            Self::ReadU16 => "ds_read_u16",
            Self::ReadI16 => "ds_read_i16",
            Self::WriteB8 => "ds_write_b8",
            Self::WriteB16 => "ds_write_b16",
            Self::WriteB32 => "ds_write_b32",
            Self::WriteB64 => "ds_write_b64",
            Self::WriteB96 => "ds_write_b96",
            Self::WriteB128 => "ds_write_b128",
            Self::Write2B32 => "ds_write2_b32",
            Self::Write2B64 => "ds_write2_b64",
            Self::Write2st64B32 => "ds_write2st64_b32",
            Self::Write2st64B64 => "ds_write2st64_b64",
            Self::AddU32 => "ds_add_u32",
            Self::AddRtnU32 => "ds_add_rtn_u32",
            Self::MaxU32 => "ds_max_u32",
            Self::MaxI32 => "ds_max_i32",
            Self::MaxF32 => "ds_max_f32",
            Self::MinF32 => "ds_min_f32",
            Self::AddF32 => "ds_add_f32",
            Self::SwizzleB32 => "ds_swizzle_b32",
            Self::PermuteB32 => "ds_permute_b32",
            Self::BpermuteB32 => "ds_bpermute_b32",
            Self::Nop => "ds_nop",
        }
    }
}
