// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Untyped buffer format MUBUF (CDNA3 ISA §9.1 p.64-65, §13.5.2 Tables 98-99 p.549-550).
//!
//! Two dwords. Dword 0: `OFFSET[11:0]`, `OFFEN[12]`, `IDXEN[13]`, `SC0[14]`
//! (`glc` on earlier parts), `SC1[15]`, `LDS[16]`, `NT[17]` (`slc` on earlier
//! parts), `OP[24:18]`, encoding `111000` in `[31:26]`. Dword 1: `VADDR[7:0]`,
//! `VDATA[15:8]`, `SRSRC[20:16]` (first SGPR of the 4-dword resource / 4),
//! `ACC[23]` (VDATA is an AccVGPR, gfx90a+), `SOFFSET[31:24]` (an SGPR or
//! inline constant, never a literal).
//!
//! The model covers the register-data forms. `LDS = 1` (`buffer_load_* ...
//! lds`, which has no VDATA) and the reserved bits `[25]` of dword 0 and
//! `[22:21]` of dword 1 are rejected on decode rather than silently dropped,
//! so decode followed by encode is the identity. `SOFFSET = 125` (`null`) is
//! rejected on decode: [`Src`] has no `null` value.
//!
//! Register-tuple legality follows the llvm-mc gfx942 assembler: a VDATA
//! tuple wider than one dword (VGPR or AGPR) and the 64-bit `idxen offen`
//! VADDR pair must start on an even register, and `SRSRC` names `s0..s96` or
//! `ttmp0..ttmp12` in steps of four. [UNVERIFIED: confirm on MI300X] what the
//! hardware does with an odd-aligned tuple the assembler rejects.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind, RegFile};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(MubufOp {
    LoadUbyte,
    LoadSbyte,
    LoadUshort,
    LoadSshort,
    LoadDword,
    LoadDwordx2,
    LoadDwordx3,
    LoadDwordx4,
    StoreByte,
    StoreShort,
    StoreDword,
    StoreDwordx2,
    StoreDwordx3,
    StoreDwordx4,
});

/// First-dword `[31:26]` of every MUBUF instruction.
const ENCODING: u32 = 0b111000;

impl MubufOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }

    /// Dwords of VDATA moved per lane: 1 for the byte/short/dword forms,
    /// 2..=4 for `dwordx2..x4` (CDNA3 ISA §9.1.2 Table 39 p.64, §9.1.3 p.65).
    pub fn data_dwords(self) -> u8 {
        match self {
            Self::LoadUbyte
            | Self::LoadSbyte
            | Self::LoadUshort
            | Self::LoadSshort
            | Self::LoadDword
            | Self::StoreByte
            | Self::StoreShort
            | Self::StoreDword => 1,
            Self::LoadDwordx2 | Self::StoreDwordx2 => 2,
            Self::LoadDwordx3 | Self::StoreDwordx3 => 3,
            Self::LoadDwordx4 | Self::StoreDwordx4 => 4,
        }
    }

    /// `buffer_store_*`: VDATA is read; otherwise it is written.
    pub fn is_store(self) -> bool {
        matches!(
            self,
            Self::StoreByte
                | Self::StoreShort
                | Self::StoreDword
                | Self::StoreDwordx2
                | Self::StoreDwordx3
                | Self::StoreDwordx4
        )
    }
}

/// `vdata` is the loaded/stored VGPRs; `vaddr` the VGPR(s) for `offen`
/// and/or `idxen` (`None` when neither); `srsrc` the first of 4 SGPRs of the
/// buffer resource (multiple of 4); `soffset` an SGPR or inline constant;
/// `offset` the unsigned 12-bit immediate.
#[derive(Clone, Debug, PartialEq)]
pub struct Mubuf {
    pub op: MubufOp,
    pub vdata: Src,
    pub vaddr: Option<Src>,
    pub srsrc: Src,
    pub soffset: Src,
    pub offset: u16,
    pub offen: bool,
    pub idxen: bool,
    pub sc0: bool,
    pub sc1: bool,
    pub nt: bool,
}

/// SRSRC field value of a resource-descriptor base register.
fn srsrc_field(srsrc: Src) -> Result<u32> {
    match srsrc {
        Src::Sgpr(n) if n % 4 == 0 && n <= 96 => Ok(u32::from(n / 4)),
        Src::Ttmp(n) if n % 4 == 0 && n <= 12 => Ok((108 + u32::from(n)) / 4),
        other => Err(format!(
            "buffer resource must start at a multiple-of-4 SGPR s0..s96 or ttmp0/4/8/12, got {}",
            other.fmt(4)
        )),
    }
}

fn srsrc_from_field(field: u32) -> Result<Src> {
    match field {
        0..=24 => Ok(Src::Sgpr((field * 4) as u8)),
        27..=30 => Ok(Src::Ttmp(((field - 27) * 4) as u8)),
        other => Err(format!(
            "reserved SRSRC field {other} (s[100:103], xnack_mask/vcc range)"
        )),
    }
}

/// A scalar or inline-constant SOFFSET: the field code must name exactly
/// this operand (rejects literals, VGPRs, `s102+`, out-of-range inline ints).
fn soffset_field(soffset: Src) -> Result<u32> {
    let code = soffset.code();
    if code > 253 || Src::from_code(code, None).ok() != Some(soffset) {
        return Err(format!(
            "soffset must be an SGPR, special scalar register or inline constant, got {soffset:?}"
        ));
    }
    Ok(u32::from(code))
}

impl Mubuf {
    /// Dwords of VADDR: a 64-bit `{index, offset}` pair with both modes.
    fn vaddr_dwords(&self) -> u8 {
        if self.offen && self.idxen {
            2
        } else {
            1
        }
    }

    /// Every encodability rule, shared by `encode` and `decode`.
    fn check(&self, arch: Arch) -> Result<()> {
        let name = self.op.name();
        if self.op.opcode(arch).is_none() {
            return Err(format!("{name} does not exist on {arch}"));
        }
        if self.offset > 0xfff {
            return Err(format!(
                "{name}: offset {} exceeds the 12-bit field",
                self.offset
            ));
        }
        match (self.offen || self.idxen, self.vaddr) {
            (false, None) => {}
            (false, Some(v)) => {
                return Err(format!(
                    "{name}: vaddr {v:?} given but neither offen nor idxen is set"
                ))
            }
            (true, None) => return Err(format!("{name}: offen/idxen require a vaddr")),
            (true, Some(Src::Vgpr(n))) => {
                let w = self.vaddr_dwords();
                if usize::from(n) + usize::from(w) > 256 {
                    return Err(format!("{name}: vaddr v{n} (+{w} dwords) runs past v255"));
                }
                if w == 2 && n % 2 != 0 {
                    return Err(format!(
                        "{name}: idxen+offen vaddr pair v[{n}:{}] must be even-aligned",
                        n + 1
                    ));
                }
            }
            (true, Some(other)) => {
                return Err(format!("{name}: vaddr must be a VGPR, got {other:?}"))
            }
        }
        let w = self.op.data_dwords();
        match self.vdata {
            Src::Vgpr(n) | Src::Agpr(n) => {
                if usize::from(n) + usize::from(w) > 256 {
                    return Err(format!(
                        "{name}: vdata {} runs past register 255",
                        self.vdata.fmt(w)
                    ));
                }
                if w > 1 && n % 2 != 0 {
                    return Err(format!(
                        "{name}: {w}-dword vdata tuple {} must be even-aligned",
                        self.vdata.fmt(w)
                    ));
                }
            }
            other => {
                return Err(format!(
                    "{name}: vdata must be a VGPR or AccVGPR, got {other:?}"
                ))
            }
        }
        srsrc_field(self.srsrc).map_err(|e| format!("{name}: {e}"))?;
        soffset_field(self.soffset).map_err(|e| format!("{name}: {e}"))?;
        Ok(())
    }

    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        self.check(arch)?;
        let opcode = u32::from(self.op.opcode(arch).expect("checked"));
        let vaddr = match self.vaddr {
            Some(Src::Vgpr(n)) => u32::from(n),
            _ => 0,
        };
        let (vdata, acc) = match self.vdata {
            Src::Agpr(n) => (u32::from(n), 1),
            Src::Vgpr(n) => (u32::from(n), 0),
            _ => unreachable!("checked"),
        };
        let w0 = u32::from(self.offset)
            | u32::from(self.offen) << 12
            | u32::from(self.idxen) << 13
            | u32::from(self.sc0) << 14
            | u32::from(self.sc1) << 15
            | u32::from(self.nt) << 17
            | opcode << 18
            | ENCODING << 26;
        let w1 = vaddr
            | vdata << 8
            | srsrc_field(self.srsrc)? << 16
            | acc << 23
            | soffset_field(self.soffset)? << 24;
        out.push(w0);
        out.push(w1);
        Ok(())
    }

    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let [w0, w1] = match words {
            [a, b, ..] => [*a, *b],
            _ => return Err("MUBUF needs two dwords".into()),
        };
        if w0 >> 26 != ENCODING {
            return Err(format!("{w0:#010x} is not a MUBUF instruction"));
        }
        if w0 >> 25 & 1 != 0 {
            return Err(format!("MUBUF {w0:#010x}: reserved bit 25 set"));
        }
        if w0 >> 16 & 1 != 0 {
            return Err(format!("MUBUF {w0:#010x}: LDS forms are not modeled"));
        }
        if w1 >> 21 & 3 != 0 {
            return Err(format!("MUBUF {w1:#010x}: reserved bits [22:21] set"));
        }
        let opcode = (w0 >> 18 & 0x7f) as u8;
        let op = MubufOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("MUBUF opcode {opcode} is not in the {arch} table"))?;
        let (offen, idxen) = (w0 >> 12 & 1 != 0, w0 >> 13 & 1 != 0);
        let vaddr_field = (w1 & 0xff) as u8;
        let vaddr = if offen || idxen {
            Some(Src::Vgpr(vaddr_field))
        } else if vaddr_field == 0 {
            None
        } else {
            return Err(format!(
                "MUBUF {w1:#010x}: VADDR {vaddr_field} set without offen/idxen"
            ));
        };
        let vdata_field = (w1 >> 8 & 0xff) as u8;
        let vdata = if w1 >> 23 & 1 != 0 {
            Src::Agpr(vdata_field)
        } else {
            Src::Vgpr(vdata_field)
        };
        let inst = Self {
            op,
            vdata,
            vaddr,
            srsrc: srsrc_from_field(w1 >> 16 & 0x1f)?,
            soffset: Src::from_code((w1 >> 24) as u16, None)
                .map_err(|e| format!("MUBUF soffset: {e}"))?,
            offset: (w0 & 0xfff) as u16,
            offen,
            idxen,
            sc0: w0 >> 14 & 1 != 0,
            sc1: w0 >> 15 & 1 != 0,
            nt: w0 >> 17 & 1 != 0,
        };
        inst.check(arch)?;
        Ok((inst, 2))
    }

    /// Reads EXEC, the 4 resource SGPRs, SOFFSET and VADDR; VDATA is read by
    /// stores and written by loads.
    pub fn footprint(&self) -> Footprint {
        let mut fp = Footprint::default();
        fp.read_file(RegFile::Exec, 0, 2);
        fp.read(self.srsrc, 4);
        fp.read(self.soffset, 1);
        if let Some(v) = self.vaddr {
            fp.read(v, self.vaddr_dwords())
        }
        if self.op.is_store() {
            fp.read(self.vdata, self.op.data_dwords())
        } else {
            fp.write(self.vdata, self.op.data_dwords())
        }
        fp
    }

    pub fn mem_kind(&self) -> MemKind {
        if self.op.is_store() {
            MemKind::VmemStore
        } else {
            MemKind::VmemLoad
        }
    }
}

impl fmt::Display for Mubuf {
    /// `llvm-objdump --mcpu=gfx942` text: `name vdata, vaddr|off, s[a:d], soffset
    /// [idxen] [offen] [offset:N] [sc0] [nt] [sc1]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}, ",
            self.op.name(),
            self.vdata.fmt(self.op.data_dwords())
        )?;
        match self.vaddr {
            Some(v) => write!(f, "{}", v.fmt(self.vaddr_dwords()))?,
            None => f.write_str("off")?,
        }
        write!(f, ", {}, {}", self.srsrc.fmt(4), self.soffset.fmt(1))?;
        if self.idxen {
            f.write_str(" idxen")?
        }
        if self.offen {
            f.write_str(" offen")?
        }
        if self.offset != 0 {
            write!(f, " offset:{}", self.offset)?
        }
        if self.sc0 {
            f.write_str(" sc0")?
        }
        if self.nt {
            f.write_str(" nt")?
        }
        if self.sc1 {
            f.write_str(" sc1")?
        }
        Ok(())
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_MUBUF); proven by the
// llvm-mc round-trip tests.
impl MubufOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::LoadUbyte => 16,
                Self::LoadSbyte => 17,
                Self::LoadUshort => 18,
                Self::LoadSshort => 19,
                Self::LoadDword => 20,
                Self::LoadDwordx2 => 21,
                Self::LoadDwordx3 => 22,
                Self::LoadDwordx4 => 23,
                Self::StoreByte => 24,
                Self::StoreShort => 26,
                Self::StoreDword => 28,
                Self::StoreDwordx2 => 29,
                Self::StoreDwordx3 => 30,
                Self::StoreDwordx4 => 31,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::LoadUbyte => "buffer_load_ubyte",
            Self::LoadSbyte => "buffer_load_sbyte",
            Self::LoadUshort => "buffer_load_ushort",
            Self::LoadSshort => "buffer_load_sshort",
            Self::LoadDword => "buffer_load_dword",
            Self::LoadDwordx2 => "buffer_load_dwordx2",
            Self::LoadDwordx3 => "buffer_load_dwordx3",
            Self::LoadDwordx4 => "buffer_load_dwordx4",
            Self::StoreByte => "buffer_store_byte",
            Self::StoreShort => "buffer_store_short",
            Self::StoreDword => "buffer_store_dword",
            Self::StoreDwordx2 => "buffer_store_dwordx2",
            Self::StoreDwordx3 => "buffer_store_dwordx3",
            Self::StoreDwordx4 => "buffer_store_dwordx4",
        }
    }
}
