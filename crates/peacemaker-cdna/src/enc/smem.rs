// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar memory format SMEM (CDNA3 ISA §8, §13.2).
//!
//! Layout (CDNA3 ISA §13.2.1 Table 74 p.521; §8.1 Table 38 p.59): dword 0 is
//! `SBASE[5:0]` (register pair / quad index, implied LSB zero), `SDATA[12:6]`,
//! `SOE[14]`, `NV[15]`, `GLC[16]`, `IMM[17]`, `OP[25:18]`, `110000[31:26]`;
//! dword 1 is `OFFSET[20:0]` and `SOFFSET[31:25]`. The address is selected by
//! `IMM`/`SOE` (§8.2.1.1 p.60): `1,0` immediate; `1,1` immediate plus the
//! `SOFFSET` register; `0,0` the register in `OFFSET[6:0]`. The encoder emits
//! the canonical `0,0` form for [`SmemOffset::Sgpr`]; the redundant `0,1`
//! form (register in `SOFFSET`) is rejected by the decoder because it does
//! not re-encode to the same bytes. `s_memtime`, `s_memrealtime` and
//! `s_dcache_inv` have no address operands: their address fields must be
//! zero, the offset is [`SmemOffset::Imm`]`(0)` and `glc` is false. The `NV`
//! bit is not modelled and a set bit is a decode error.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(SmemOp {
    LoadDword,
    LoadDwordx2,
    LoadDwordx4,
    LoadDwordx8,
    LoadDwordx16,
    BufferLoadDword,
    BufferLoadDwordx2,
    BufferLoadDwordx4,
    BufferLoadDwordx8,
    BufferLoadDwordx16,
    Memtime,
    Memrealtime,
    DcacheInv,
});

impl SmemOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Dwords written to `sdata` (0 for `s_dcache_inv`).
    pub fn data_dwords(self) -> u8 {
        match self {
            Self::LoadDword | Self::BufferLoadDword => 1,
            Self::LoadDwordx2 | Self::BufferLoadDwordx2 | Self::Memtime | Self::Memrealtime => 2,
            Self::LoadDwordx4 | Self::BufferLoadDwordx4 => 4,
            Self::LoadDwordx8 | Self::BufferLoadDwordx8 => 8,
            Self::LoadDwordx16 | Self::BufferLoadDwordx16 => 16,
            Self::DcacheInv => 0,
        }
    }
    /// The op reads memory through an address (`s_load_*`, `s_buffer_load_*`);
    /// `s_memtime`, `s_memrealtime` and `s_dcache_inv` do not.
    pub fn has_address(self) -> bool {
        !matches!(self, Self::Memtime | Self::Memrealtime | Self::DcacheInv)
    }
    /// `s_buffer_load_*` reads a 4-SGPR buffer resource at `sbase`; the
    /// others a 2-SGPR address.
    pub fn buffer(self) -> bool {
        matches!(
            self,
            Self::BufferLoadDword
                | Self::BufferLoadDwordx2
                | Self::BufferLoadDwordx4
                | Self::BufferLoadDwordx8
                | Self::BufferLoadDwordx16
        )
    }
}

/// The address offset: a 21-bit signed byte immediate, an SGPR, or both
/// (`soffset_en`; CDNA3 ISA §13.2 SMEM fields IMM / SOFFSET_EN / OFFSET / SOFFSET).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmemOffset {
    Imm(i32),
    Sgpr(Src),
    SgprImm(Src, i32),
}

/// `sdata = load(sbase + offset)`. `sdata` and `sbase` are `None` for the
/// opcodes without them (`s_dcache_inv`; `s_memtime` has `sdata` only).
#[derive(Clone, Debug, PartialEq)]
pub struct Smem {
    pub op: SmemOp,
    pub sdata: Option<Src>,
    pub sbase: Option<Src>,
    pub offset: SmemOffset,
    pub glc: bool,
}

/// Whether a `dwords`-wide register range starting at `src` is a legal SMEM
/// register operand: inside one register file and aligned (pairs even,
/// quads and wider a multiple of 4; CDNA3 ISA §8.1 Table 38 p.59).
fn span_ok(src: Src, dwords: u8) -> bool {
    let w = u16::from(dwords);
    let align = match dwords {
        0 | 1 => 1,
        2 => 2,
        _ => 4,
    };
    match src {
        Src::Sgpr(n) => u16::from(n) % align == 0 && u16::from(n) + w <= 102,
        Src::Ttmp(n) => u16::from(n) % align == 0 && u16::from(n) + w <= 16,
        Src::VccLo | Src::FlatScratchLo | Src::XnackMaskLo => w <= 2,
        Src::VccHi | Src::FlatScratchHi | Src::XnackMaskHi => w <= 1,
        _ => false,
    }
}

/// SDATA: an SGPR, TTMP, VCC, FLAT_SCRATCH or XNACK_MASK range; "Not: exec or m0".
fn sdata_ok(src: Src, dwords: u8) -> bool {
    span_ok(src, dwords)
}

/// SBASE: a pair (quad for `s_buffer_*`) that is a plain SGPR/TTMP range, or
/// (non-buffer only) VCC / FLAT_SCRATCH / XNACK_MASK / EXEC as llvm-mc accepts.
fn sbase_ok(src: Src, buffer: bool) -> bool {
    if buffer {
        matches!(src, Src::Sgpr(_) | Src::Ttmp(_)) && span_ok(src, 4)
    } else {
        src == Src::ExecLo || span_ok(src, 2)
    }
}

/// SOFFSET / OFFSET[6:0] register: SGPR, TTMP, VCC, FLAT_SCRATCH, XNACK_MASK, M0 or EXEC halves.
fn soffset_ok(src: Src) -> bool {
    match src {
        Src::Sgpr(n) => n <= 101,
        Src::Ttmp(n) => n <= 15,
        Src::VccLo
        | Src::VccHi
        | Src::FlatScratchLo
        | Src::FlatScratchHi
        | Src::XnackMaskLo
        | Src::XnackMaskHi
        | Src::M0
        | Src::ExecLo
        | Src::ExecHi => true,
        _ => false,
    }
}

/// `s_load_*` takes a signed 21-bit byte offset (CDNA3 ISA §8.2.1.1 p.60,
/// §13.2.1 Table 74 p.521: "Signed offsets only work with S_LOAD/STORE");
/// `s_buffer_load_*` an unsigned 20-bit one, the range llvm-mc enforces
/// (`expected a 20-bit unsigned offset`). [UNVERIFIED: confirm on MI300X that
/// bit 20 is ignored rather than a sign for buffer loads.]
fn imm_ok(imm: i32, buffer: bool) -> bool {
    if buffer {
        (0..=0xf_ffff).contains(&imm)
    } else {
        (-0x10_0000..=0xf_ffff).contains(&imm)
    }
}

fn fmt_imm(imm: i32) -> String {
    if imm < 0 {
        format!("-{:#x}", imm.unsigned_abs())
    } else {
        format!("{imm:#x}")
    }
}

impl Smem {
    /// Encode to two dwords. Errors: opcode absent on `arch`, operands missing
    /// or present that the op does not take, unaligned or unsupported
    /// `sdata`/`sbase`/`soffset` registers, offset outside the signed 21-bit
    /// range, `glc` on an op without a cache policy.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let name = self.op.name();
        let opcode = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{name} does not exist on {}", arch.name()))?;
        let mut w0 = 0b11_0000u32 << 26 | u32::from(opcode) << 18;
        let mut w1 = 0u32;
        if self.op.has_address() {
            if self.glc {
                w0 |= 1 << 16
            }
            let sbase = self.sbase.ok_or_else(|| format!("{name} needs an sbase"))?;
            if !sbase_ok(sbase, self.op.buffer()) {
                return Err(format!("{name}: unsupported or misaligned sbase {sbase:?}"));
            }
            w0 |= u32::from(sbase.code() >> 1);
            match self.offset {
                SmemOffset::Imm(i) => {
                    if !imm_ok(i, self.op.buffer()) {
                        return Err(format!(
                            "{name}: offset {i} outside the {} range",
                            if self.op.buffer() {
                                "unsigned 20-bit"
                            } else {
                                "signed 21-bit"
                            }
                        ));
                    }
                    w0 |= 1 << 17;
                    w1 = i as u32 & 0x1f_ffff;
                }
                SmemOffset::Sgpr(s) => {
                    if !soffset_ok(s) {
                        return Err(format!("{name}: {s:?} cannot be an SMEM soffset register"));
                    }
                    w1 = u32::from(s.code());
                }
                SmemOffset::SgprImm(s, i) => {
                    if !soffset_ok(s) {
                        return Err(format!("{name}: {s:?} cannot be an SMEM soffset register"));
                    }
                    if !imm_ok(i, self.op.buffer()) {
                        return Err(format!(
                            "{name}: offset {i} outside the {} range",
                            if self.op.buffer() {
                                "unsigned 20-bit"
                            } else {
                                "signed 21-bit"
                            }
                        ));
                    }
                    w0 |= 1 << 17 | 1 << 14;
                    w1 = (i as u32 & 0x1f_ffff) | u32::from(s.code()) << 25;
                }
            }
        } else {
            if self.glc {
                return Err(format!("{name} has no glc"));
            }
            if self.sbase.is_some() {
                return Err(format!("{name} has no sbase"));
            }
            if self.offset != SmemOffset::Imm(0) {
                return Err(format!("{name} has no address offset"));
            }
        }
        match (self.op.data_dwords(), self.sdata) {
            (0, None) => {}
            (0, Some(_)) => return Err(format!("{name} has no sdata")),
            (_, None) => return Err(format!("{name} needs an sdata")),
            (n, Some(d)) => {
                if !sdata_ok(d, n) {
                    return Err(format!(
                        "{name}: unsupported or misaligned sdata {d:?} for {n} dwords"
                    ));
                }
                w0 |= u32::from(d.code()) << 6;
            }
        }
        out.push(w0);
        out.push(w1);
        Ok(())
    }

    /// Decode the instruction at `words[0..2]`; returns it and the dwords consumed (2).
    /// Errors on anything [`Smem::encode`] would not produce: unknown opcode,
    /// `NV` set, the redundant `IMM=0,SOE=1` form, nonzero reserved or
    /// ignored fields, unsupported register codes.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let (&w0, &w1) = match words {
            [a, b, ..] => (a, b),
            _ => return Err("SMEM needs two dwords".into()),
        };
        if w0 >> 26 != 0b11_0000 {
            return Err(format!("{w0:#010x} is not an SMEM instruction"));
        }
        let opcode = (w0 >> 18 & 0xff) as u8;
        let op = SmemOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SMEM opcode {opcode} is not modelled on {}", arch.name()))?;
        let name = op.name();
        if w0 >> 15 & 1 != 0 {
            return Err(format!("{name}: NV bit set is not modelled"));
        }
        let (glc, imm, soe) = (w0 >> 16 & 1 != 0, w0 >> 17 & 1 != 0, w0 >> 14 & 1 != 0);
        let sdata_field = (w0 >> 6 & 0x7f) as u16;
        let sbase_field = (w0 & 0x3f) as u16;
        let (off, soff) = (w1 & 0x1f_ffff, (w1 >> 25) as u16);
        if w1 >> 21 & 0xf != 0 {
            return Err(format!("{name}: reserved SMEM bits 24:21 set"));
        }
        let sdata = match op.data_dwords() {
            0 => {
                if sdata_field != 0 {
                    return Err(format!("{name}: sdata field must be zero"));
                }
                None
            }
            n => {
                let d =
                    Src::from_code(sdata_field, None).map_err(|e| format!("{name}: sdata: {e}"))?;
                if !sdata_ok(d, n) {
                    return Err(format!(
                        "{name}: unsupported or misaligned sdata {d:?} for {n} dwords"
                    ));
                }
                Some(d)
            }
        };
        if !op.has_address() {
            if glc || imm || soe || sbase_field != 0 || off != 0 || soff != 0 {
                return Err(format!(
                    "{name}: address, glc and offset fields must be zero"
                ));
            }
            return Ok((
                Self {
                    op,
                    sdata,
                    sbase: None,
                    offset: SmemOffset::Imm(0),
                    glc: false,
                },
                2,
            ));
        }
        let sbase =
            Src::from_code(sbase_field * 2, None).map_err(|e| format!("{name}: sbase: {e}"))?;
        if !sbase_ok(sbase, op.buffer()) {
            return Err(format!("{name}: unsupported or misaligned sbase {sbase:?}"));
        }
        let soffset = |code: u16| -> Result<Src> {
            let s = Src::from_code(code, None).map_err(|e| format!("{name}: soffset: {e}"))?;
            if soffset_ok(s) {
                Ok(s)
            } else {
                Err(format!("{name}: {s:?} cannot be an SMEM soffset register"))
            }
        };
        let simm = ((off << 11) as i32) >> 11;
        if imm && !imm_ok(simm, op.buffer()) {
            return Err(format!(
                "{name}: offset {simm} outside the unsigned 20-bit range"
            ));
        }
        let offset = match (imm, soe) {
            (true, false) => {
                if soff != 0 { return Err(format!("{name}: soffset field set without SOE")) }
                SmemOffset::Imm(simm)
            }
            (true, true) => SmemOffset::SgprImm(soffset(soff)?, simm),
            (false, false) => {
                if off >> 7 != 0 || soff != 0 { return Err(format!("{name}: register offset form has bits set beyond OFFSET[6:0]")) }
                SmemOffset::Sgpr(soffset(off as u16)?)
            }
            (false, true) => return Err(format!("{name}: IMM=0 with SOE=1 (register in SOFFSET) is not the canonical register-offset encoding")),
        };
        Ok((
            Self {
                op,
                sdata,
                sbase: Some(sbase),
                offset,
                glc,
            },
            2,
        ))
    }

    /// `sdata` is defined; `sbase` (2 SGPRs, 4 for `s_buffer_*`) and the
    /// offset register are used.
    pub fn footprint(&self) -> Footprint {
        let mut fp = Footprint::default();
        if let Some(d) = self.sdata {
            fp.write(d, self.op.data_dwords())
        }
        if let Some(b) = self.sbase {
            fp.read(b, if self.op.buffer() { 4 } else { 2 })
        }
        match self.offset {
            SmemOffset::Imm(_) => {}
            SmemOffset::Sgpr(s) | SmemOffset::SgprImm(s, _) => fp.read(s, 1),
        }
        fp
    }
    /// `SmemLoad` for every opcode here (all count on LGKM_CNT).
    pub fn mem_kind(&self) -> Option<MemKind> {
        Some(MemKind::SmemLoad)
    }
}

impl fmt::Display for Smem {
    /// `s_load_dword s0, s[2:3], 0x10`, `s_load_dword s0, s[2:3], s4 offset:-0x10 glc`,
    /// `s_memtime s[0:1]`, `s_dcache_inv` (llvm-mc / llvm-objdump spelling).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op.name())?;
        let mut sep = " ";
        if let Some(d) = self.sdata {
            write!(f, "{sep}{}", d.fmt(self.op.data_dwords()))?;
            sep = ", ";
        }
        if let Some(b) = self.sbase {
            write!(f, "{sep}{}", b.fmt(if self.op.buffer() { 4 } else { 2 }))?;
            sep = ", ";
        }
        if self.op.has_address() {
            match self.offset {
                SmemOffset::Imm(i) => write!(f, "{sep}{}", fmt_imm(i))?,
                SmemOffset::Sgpr(s) => write!(f, "{sep}{}", s.fmt(1))?,
                SmemOffset::SgprImm(s, i) => write!(f, "{sep}{} offset:{}", s.fmt(1), fmt_imm(i))?,
            }
        }
        if self.glc {
            write!(f, " glc")?;
        }
        Ok(())
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SMEM); proven by the
// llvm-mc round-trip tests.
impl SmemOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::LoadDword => 0,
                Self::LoadDwordx2 => 1,
                Self::LoadDwordx4 => 2,
                Self::LoadDwordx8 => 3,
                Self::LoadDwordx16 => 4,
                Self::BufferLoadDword => 8,
                Self::BufferLoadDwordx2 => 9,
                Self::BufferLoadDwordx4 => 10,
                Self::BufferLoadDwordx8 => 11,
                Self::BufferLoadDwordx16 => 12,
                Self::Memtime => 36,
                Self::Memrealtime => 37,
                Self::DcacheInv => 32,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::LoadDword => "s_load_dword",
            Self::LoadDwordx2 => "s_load_dwordx2",
            Self::LoadDwordx4 => "s_load_dwordx4",
            Self::LoadDwordx8 => "s_load_dwordx8",
            Self::LoadDwordx16 => "s_load_dwordx16",
            Self::BufferLoadDword => "s_buffer_load_dword",
            Self::BufferLoadDwordx2 => "s_buffer_load_dwordx2",
            Self::BufferLoadDwordx4 => "s_buffer_load_dwordx4",
            Self::BufferLoadDwordx8 => "s_buffer_load_dwordx8",
            Self::BufferLoadDwordx16 => "s_buffer_load_dwordx16",
            Self::Memtime => "s_memtime",
            Self::Memrealtime => "s_memrealtime",
            Self::DcacheInv => "s_dcache_inv",
        }
    }
}
