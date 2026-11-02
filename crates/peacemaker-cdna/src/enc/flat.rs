// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLAT / GLOBAL / SCRATCH format (CDNA3 ISA §10, §13.6). gfx940+ cache
//! policy bits are `sc0`, `sc1`, `nt` (replacing glc/slc).
//!
//! Operands (CDNA3 ISA §10.1 Table 53-55 p.79-80, encoding §13.6): `saddr`
//! `None` is the 64-bit VGPR address `v[n:n+1]` with `off`; `Some` is a 32-bit
//! VGPR offset from the SGPR-pair base (GLOBAL) or an SGPR offset (SCRATCH).
//! SCRATCH `addr: None` clears SVE. Atomics return into `vdst` iff `sc0`.
//! Wire layout: dword 0 = OFFSET[12:0], SVE[13] (the LDS bit is not modeled),
//! SEG[15:14], SC0[16], NT[17], OP[24:18], SC1[25], `110111`[31:26]; dword 1 =
//! ADDR[7:0], DATA[15:8], SADDR[22:16] (0x7f = off), ACC[23], VDST[31:24].
//! Decoding rejects any input that does not re-encode to itself.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind, RegFile};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(FlatOp {
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
    AtomicAdd,
    AtomicAddF32,
});

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlatSeg {
    Flat,
    Scratch,
    Global,
}

impl FlatOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|op| op.opcode(arch) == Some(opcode))
    }
    /// Dwords of (vdst, data): loads `(n, 0)`, stores `(0, n)`, atomics `(1, 1)`
    /// (`vdst` only when `sc0` returns the old value).
    pub fn widths(self) -> (u8, u8) {
        match self {
            Self::LoadUbyte
            | Self::LoadSbyte
            | Self::LoadUshort
            | Self::LoadSshort
            | Self::LoadDword => (1, 0),
            Self::LoadDwordx2 => (2, 0),
            Self::LoadDwordx3 => (3, 0),
            Self::LoadDwordx4 => (4, 0),
            Self::StoreByte | Self::StoreShort | Self::StoreDword => (0, 1),
            Self::StoreDwordx2 => (0, 2),
            Self::StoreDwordx3 => (0, 3),
            Self::StoreDwordx4 => (0, 4),
            Self::AtomicAdd | Self::AtomicAddF32 => (1, 1),
        }
    }

    fn class(self) -> Class {
        match self.widths() {
            (_, 0) => Class::Load,
            (0, _) => Class::Store,
            _ => Class::Atomic,
        }
    }
}

/// `addr` is a 64-bit VGPR address when `saddr` is `None` (`off`), else a
/// 32-bit VGPR offset added to the SGPR-pair base `saddr`. `offset` is the
/// signed 13-bit immediate (global/scratch) or unsigned 12-bit (flat). Atomics
/// return the old value into `vdst` when `sc0` is set.
#[derive(Clone, Debug, PartialEq)]
pub struct Flat {
    pub op: FlatOp,
    pub seg: FlatSeg,
    pub vdst: Option<Src>,
    pub addr: Option<Src>,
    pub data: Option<Src>,
    pub saddr: Option<Src>,
    pub offset: i16,
    pub sc0: bool,
    pub sc1: bool,
    pub nt: bool,
}

/// The three operation shapes of the table: a load defines `vdst` from memory,
/// a store reads `data`, an atomic reads `data` and (with `sc0`) defines `vdst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Load,
    Store,
    Atomic,
}

const ENC_FLAT: u32 = 0b11_0111;

/// ACC (dword 1 bit 23): the DATA/VDST registers are AccVGPRs (CDNA3 ISA §10.1
/// Table 53 p.79, field ACC).
const ACC_BIT: u32 = 1 << 23;

impl Flat {
    /// Dwords of the VGPR address operand: a 64-bit pair for FLAT and for
    /// GLOBAL without `saddr`; a single 32-bit offset otherwise (SCRATCH is
    /// always 32-bit) (CDNA3 ISA §10.1 Table 53/55 p.79-80).
    fn addr_dwords(&self) -> u8 {
        match self.seg {
            FlatSeg::Flat => 2,
            FlatSeg::Global => {
                if self.saddr.is_some() {
                    1
                } else {
                    2
                }
            }
            FlatSeg::Scratch => 1,
        }
    }

    /// Dwords of the SGPR base operand: a pair for GLOBAL, one SGPR for SCRATCH.
    fn saddr_dwords(&self) -> u8 {
        if self.seg == FlatSeg::Global {
            2
        } else {
            1
        }
    }

    /// A VGPR/AccVGPR operand `width` dwords wide: returns `(first register, acc)`.
    /// Tuples of two or more dwords are even-aligned on gfx90a+/gfx942 (llvm
    /// register class `VReg_*_Align2`).
    fn vreg(src: Option<Src>, width: u8, what: &str, allow_agpr: bool) -> Result<(u8, bool)> {
        let src = src.ok_or_else(|| format!("{what} is required"))?;
        let (n, acc) = match src {
            Src::Vgpr(n) => (n, false),
            Src::Agpr(n) if allow_agpr => (n, true),
            other => {
                return Err(format!(
                    "{what} must be a {}, got {other:?}",
                    if allow_agpr {
                        "VGPR or AccVGPR"
                    } else {
                        "VGPR"
                    }
                ))
            }
        };
        if width >= 2 && n % 2 != 0 {
            return Err(format!("{what} {} is not even-aligned", src.fmt(width)));
        }
        if usize::from(n) + usize::from(width) > 256 {
            return Err(format!("{what} {} runs past register 255", src.fmt(width)));
        }
        Ok((n, acc))
    }

    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let opcode = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{:?} does not exist on {arch:?}", self.op))?;
        self.op
            .name(self.seg)
            .ok_or_else(|| format!("{:?} has no {:?} form", self.op, self.seg))?;
        let class = self.op.class();
        let (vw, dw) = self.op.widths();

        // Offset: 13-bit signed (global/scratch) or 12-bit unsigned (flat).
        let (lo, hi) = if self.seg == FlatSeg::Flat {
            (0, 4095)
        } else {
            (-4096, 4095)
        };
        if !(lo..=hi).contains(&self.offset) {
            return Err(format!(
                "offset {} outside {lo}..={hi} for {:?}",
                self.offset, self.seg
            ));
        }

        // vdst / data / ACC.
        let (vdst, data, acc) = match class {
            Class::Load => {
                if self.data.is_some() {
                    return Err("a load takes no data operand".into());
                }
                let (n, acc) = Self::vreg(self.vdst, vw, "vdst", true)?;
                (u32::from(n), 0, acc)
            }
            Class::Store => {
                if self.vdst.is_some() {
                    return Err("a store has no vdst".into());
                }
                let (n, acc) = Self::vreg(self.data, dw, "data", true)?;
                (0, u32::from(n), acc)
            }
            Class::Atomic => {
                if self.sc0 != self.vdst.is_some() {
                    return Err(
                        "an atomic returns its old value into vdst exactly when sc0 is set".into(),
                    );
                }
                let (n, acc) = Self::vreg(self.data, dw, "data", true)?;
                let mut vdst = 0;
                if self.vdst.is_some() {
                    let (d, vacc) = Self::vreg(self.vdst, vw, "vdst", true)?;
                    if vacc != acc {
                        return Err("vdst and data of an atomic must be in the same register file (one ACC bit)".into());
                    }
                    vdst = u32::from(d);
                }
                // llvm-mc has no AccVGPR form of the returning f32 atomic
                // (`global_atomic_add_f32 a1, v[2:3], a1, off sc0` is rejected).
                if acc && self.sc0 && self.op == FlatOp::AtomicAddF32 {
                    return Err("returning atomic_add_f32 takes VGPRs only".into());
                }
                (vdst, u32::from(n), acc)
            }
        };

        // addr / saddr / SVE.
        let mut addr = 0u32;
        let mut sve = false;
        match (self.seg, self.addr) {
            (FlatSeg::Scratch, None) => {}
            (_, a) => {
                let (n, _) = Self::vreg(a, self.addr_dwords(), "addr", false)?;
                addr = u32::from(n);
                sve = self.seg == FlatSeg::Scratch;
            }
        }
        let saddr = match (self.seg, self.saddr) {
            (FlatSeg::Flat, None) => 0,
            (FlatSeg::Flat, Some(_)) => return Err("FLAT has no saddr".into()),
            (_, None) => 0x7f,
            (seg, Some(s)) => {
                let code = s.code();
                if !s.is_sreg() {
                    return Err(format!("saddr must be a scalar register, got {s:?}"));
                }
                if seg == FlatSeg::Global {
                    // 64-bit SGPR base: even first register, below m0/exec (llvm
                    // rejects `exec`; m0 is not a pair).
                    if code % 2 != 0 || code > 123 {
                        return Err(format!(
                            "global saddr {} is not a valid SGPR pair",
                            s.fmt(2)
                        ));
                    }
                } else if code > 126 {
                    return Err(format!("scratch saddr {s:?} is not a valid SGPR"));
                }
                u32::from(code)
            }
        };

        let off13 = (self.offset as u16 as u32) & 0x1fff;
        let seg = match self.seg {
            FlatSeg::Flat => 0,
            FlatSeg::Scratch => 1,
            FlatSeg::Global => 2,
        };
        let w0 = off13
            | u32::from(sve) << 13
            | seg << 14
            | u32::from(self.sc0) << 16
            | u32::from(self.nt) << 17
            | u32::from(opcode) << 18
            | u32::from(self.sc1) << 25
            | ENC_FLAT << 26;
        let w1 = addr | data << 8 | saddr << 16 | (if acc { ACC_BIT } else { 0 }) | vdst << 24;
        out.extend_from_slice(&[w0, w1]);
        Ok(())
    }

    /// Decode two dwords. Reserved or ignored fields that llvm drops from the
    /// text (junk `data` on a load, `vdst` on a store, LDS bit, `saddr` on
    /// FLAT, ...) are rejected: the decoded instruction is re-encoded and
    /// must reproduce the input words exactly.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let (&w0, &w1) = match words {
            [a, b, ..] => (a, b),
            _ => return Err("FLAT needs two dwords".into()),
        };
        if w0 >> 26 != ENC_FLAT {
            return Err(format!("{w0:#010x} is not a FLAT-family encoding"));
        }
        let seg = match (w0 >> 14) & 3 {
            0 => FlatSeg::Flat,
            1 => FlatSeg::Scratch,
            2 => FlatSeg::Global,
            _ => return Err("reserved FLAT segment 3".into()),
        };
        let opcode = ((w0 >> 18) & 0x7f) as u8;
        let op = FlatOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("unknown FLAT opcode {opcode:#x}"))?;
        let sve = (w0 >> 13) & 1 != 0;
        let off13 = (w0 & 0x1fff) as i16;
        let offset = if seg == FlatSeg::Flat {
            off13
        } else {
            (off13 << 3) >> 3
        };
        let (sc0, nt, sc1) = (
            (w0 >> 16) & 1 != 0,
            (w0 >> 17) & 1 != 0,
            (w0 >> 25) & 1 != 0,
        );
        let acc = w1 & ACC_BIT != 0;
        let vreg = |n: u32| {
            if acc {
                Src::Agpr(n as u8)
            } else {
                Src::Vgpr(n as u8)
            }
        };
        let (vdst, data) = match op.class() {
            Class::Load => (Some(vreg(w1 >> 24)), None),
            Class::Store => (None, Some(vreg((w1 >> 8) & 0xff))),
            Class::Atomic => (sc0.then(|| vreg(w1 >> 24)), Some(vreg((w1 >> 8) & 0xff))),
        };
        let addr = match seg {
            FlatSeg::Scratch if !sve => None,
            _ => Some(Src::Vgpr((w1 & 0xff) as u8)),
        };
        let saddr_code = (w1 >> 16) & 0x7f;
        let saddr = match seg {
            FlatSeg::Flat => None,
            _ if saddr_code == 0x7f => None,
            _ => Some(Src::from_code(saddr_code as u16, None)?),
        };
        let inst = Self {
            op,
            seg,
            vdst,
            addr,
            data,
            saddr,
            offset,
            sc0,
            sc1,
            nt,
        };
        let mut again = Vec::with_capacity(2);
        inst.encode(arch, &mut again)?;
        if again != [w0, w1] {
            return Err(format!(
                "{w0:#010x} {w1:#010x} has reserved/ignored bits set (canonical form {again:08x?})"
            ));
        }
        Ok((inst, 2))
    }

    /// Reads EXEC, the address VGPR(s), `saddr` and `data`; writes `vdst`.
    /// SCRATCH also reads the FLAT_SCRATCH base, which the hardware attaches to
    /// every scratch request (CDNA3 ISA §10.1 p.80, Table 55).
    pub fn footprint(&self) -> Footprint {
        let (vw, dw) = self.op.widths();
        let mut fp = Footprint::default();
        fp.read_file(RegFile::Exec, 0, 2);
        if let Some(a) = self.addr {
            fp.read(a, self.addr_dwords())
        }
        if let Some(s) = self.saddr {
            fp.read(s, self.saddr_dwords())
        }
        if let Some(d) = self.data {
            fp.read(d, dw)
        }
        if self.seg == FlatSeg::Scratch {
            fp.read_file(RegFile::FlatScratch, 0, 2)
        }
        if let Some(v) = self.vdst {
            fp.write(v, vw)
        }
        fp
    }

    /// Global/scratch count on VM_CNT only; plain FLAT on VM_CNT and LGKM_CNT
    /// (CDNA3 ISA §4.4 p.19): the kind reports the VMEM side. An atomic is
    /// `VmemAtomicRtn` exactly when `sc0` asks for the pre-op value.
    pub fn mem_kind(&self) -> MemKind {
        match self.op.class() {
            Class::Load => MemKind::VmemLoad,
            Class::Store => MemKind::VmemStore,
            Class::Atomic if self.sc0 => MemKind::VmemAtomicRtn,
            Class::Atomic => MemKind::VmemAtomic,
        }
    }
}

impl fmt::Display for Flat {
    /// llvm-mc spelling: operands in assembler order, then `offset:N` (omitted
    /// when 0), then `sc0 nt sc1`. A missing `addr`/`saddr` prints `off`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(name) = self.op.name(self.seg) else {
            return write!(f, "<{:?} has no {:?} form>", self.op, self.seg);
        };
        let (vw, dw) = self.op.widths();
        let operand =
            |src: Option<Src>, width: u8| src.map_or_else(|| "off".to_string(), |s| s.fmt(width));
        write!(f, "{name}")?;
        let mut sep = " ";
        if self.vdst.is_some() || self.op.class() == Class::Load {
            write!(f, "{sep}{}", operand(self.vdst, vw))?;
            sep = ", ";
        }
        write!(f, "{sep}{}", operand(self.addr, self.addr_dwords()))?;
        if self.op.class() != Class::Load {
            write!(f, ", {}", operand(self.data, dw))?
        }
        if self.seg != FlatSeg::Flat {
            write!(f, ", {}", operand(self.saddr, self.saddr_dwords()))?
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
// machine-readable ISA zip sha256 82404f11…, encoding ENC_FLAT_GLBL); proven by the
// llvm-mc round-trip tests.
impl FlatOp {
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
                Self::AtomicAdd => 66,
                Self::AtomicAddF32 => 77,
            }),
        }
    }
    /// Mnemonic for `seg`; `None` where the segment lacks the instruction (scratch atomics).
    pub fn name(self, seg: FlatSeg) -> Option<&'static str> {
        Some(match (seg, self) {
            (FlatSeg::Global, Self::LoadUbyte) => "global_load_ubyte",
            (FlatSeg::Flat, Self::LoadUbyte) => "flat_load_ubyte",
            (FlatSeg::Scratch, Self::LoadUbyte) => "scratch_load_ubyte",
            (FlatSeg::Global, Self::LoadSbyte) => "global_load_sbyte",
            (FlatSeg::Flat, Self::LoadSbyte) => "flat_load_sbyte",
            (FlatSeg::Scratch, Self::LoadSbyte) => "scratch_load_sbyte",
            (FlatSeg::Global, Self::LoadUshort) => "global_load_ushort",
            (FlatSeg::Flat, Self::LoadUshort) => "flat_load_ushort",
            (FlatSeg::Scratch, Self::LoadUshort) => "scratch_load_ushort",
            (FlatSeg::Global, Self::LoadSshort) => "global_load_sshort",
            (FlatSeg::Flat, Self::LoadSshort) => "flat_load_sshort",
            (FlatSeg::Scratch, Self::LoadSshort) => "scratch_load_sshort",
            (FlatSeg::Global, Self::LoadDword) => "global_load_dword",
            (FlatSeg::Flat, Self::LoadDword) => "flat_load_dword",
            (FlatSeg::Scratch, Self::LoadDword) => "scratch_load_dword",
            (FlatSeg::Global, Self::LoadDwordx2) => "global_load_dwordx2",
            (FlatSeg::Flat, Self::LoadDwordx2) => "flat_load_dwordx2",
            (FlatSeg::Scratch, Self::LoadDwordx2) => "scratch_load_dwordx2",
            (FlatSeg::Global, Self::LoadDwordx3) => "global_load_dwordx3",
            (FlatSeg::Flat, Self::LoadDwordx3) => "flat_load_dwordx3",
            (FlatSeg::Scratch, Self::LoadDwordx3) => "scratch_load_dwordx3",
            (FlatSeg::Global, Self::LoadDwordx4) => "global_load_dwordx4",
            (FlatSeg::Flat, Self::LoadDwordx4) => "flat_load_dwordx4",
            (FlatSeg::Scratch, Self::LoadDwordx4) => "scratch_load_dwordx4",
            (FlatSeg::Global, Self::StoreByte) => "global_store_byte",
            (FlatSeg::Flat, Self::StoreByte) => "flat_store_byte",
            (FlatSeg::Scratch, Self::StoreByte) => "scratch_store_byte",
            (FlatSeg::Global, Self::StoreShort) => "global_store_short",
            (FlatSeg::Flat, Self::StoreShort) => "flat_store_short",
            (FlatSeg::Scratch, Self::StoreShort) => "scratch_store_short",
            (FlatSeg::Global, Self::StoreDword) => "global_store_dword",
            (FlatSeg::Flat, Self::StoreDword) => "flat_store_dword",
            (FlatSeg::Scratch, Self::StoreDword) => "scratch_store_dword",
            (FlatSeg::Global, Self::StoreDwordx2) => "global_store_dwordx2",
            (FlatSeg::Flat, Self::StoreDwordx2) => "flat_store_dwordx2",
            (FlatSeg::Scratch, Self::StoreDwordx2) => "scratch_store_dwordx2",
            (FlatSeg::Global, Self::StoreDwordx3) => "global_store_dwordx3",
            (FlatSeg::Flat, Self::StoreDwordx3) => "flat_store_dwordx3",
            (FlatSeg::Scratch, Self::StoreDwordx3) => "scratch_store_dwordx3",
            (FlatSeg::Global, Self::StoreDwordx4) => "global_store_dwordx4",
            (FlatSeg::Flat, Self::StoreDwordx4) => "flat_store_dwordx4",
            (FlatSeg::Scratch, Self::StoreDwordx4) => "scratch_store_dwordx4",
            (FlatSeg::Global, Self::AtomicAdd) => "global_atomic_add",
            (FlatSeg::Flat, Self::AtomicAdd) => "flat_atomic_add",
            (FlatSeg::Global, Self::AtomicAddF32) => "global_atomic_add_f32",
            (FlatSeg::Flat, Self::AtomicAddF32) => "flat_atomic_add_f32",
            _ => return None,
        })
    }
}
