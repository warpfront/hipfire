// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! [`Inst`]: one typed CDNA instruction, a variant per microcode format.
//!
//! Encoding and decoding take the [`Arch`] variant: an opcode a variant lacks
//! is an error, and a later variant may renumber or add opcodes per table.
//!
//! Format selection from a first dword follows the encoding field values of
//! CDNA3 ISA §13 (p.205 ff.): SOP2 `[31:30]=10`, SOPK `[31:28]=1011`, SOP1
//! `[31:23]=1_0111_1101`, SOPC `1_0111_1110`, SOPP `1_0111_1111`, SMEM
//! `[31:26]=110000`, VOP2 `[31]=0`, VOP1 `[31:25]=011_1111`, VOPC
//! `011_1110`, VOP3 `[31:26]=110100`, VOP3P `[31:23]=1_1010_0111` (MAI when
//! the 7-bit opcode is >= 0x40), DS `110110`, FLAT/GLOBAL/SCRATCH `110111`,
//! MUBUF `111000`, MTBUF `111010`, MIMG `111100`, EXP `110001`.
use crate::arch::Arch;
use crate::enc::{
    ds::Ds,
    flat::Flat,
    mubuf::Mubuf,
    smem::Smem,
    sop::{Sop1, Sop2, Sopc, Sopk, Sopp},
    vop::{Vop1, Vop2, Vop2K, Vop2KOp, Vopc},
    vop3::Vop3,
    vop3p::{Mai, Vop3p},
};
use crate::operand::Src;
use crate::xml_inst::XmlInst;
use crate::Result;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Inst {
    Sop1(Sop1),
    Sop2(Sop2),
    Sopk(Sopk),
    Sopc(Sopc),
    Sopp(Sopp),
    Smem(Smem),
    Vop1(Vop1),
    Vop2(Vop2),
    Vop2K(Vop2K),
    Vopc(Vopc),
    Vop3(Vop3),
    Vop3p(Vop3p),
    Mai(Mai),
    Ds(Ds),
    Flat(Flat),
    Mubuf(Mubuf),
    /// An instruction only the generated tables describe ([`XmlInst`]).
    Xml(XmlInst),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Sop1,
    Sop2,
    Sopk,
    Sopc,
    Sopp,
    Smem,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vop3p,
    Mai,
    Ds,
    Flat,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
}

/// The format of the instruction whose first dword is `word`.
pub fn classify(word: u32) -> Option<Format> {
    if word >> 31 == 0 {
        return Some(match word >> 25 {
            0b011_1111 => Format::Vop1,
            0b011_1110 => Format::Vopc,
            _ => Format::Vop2,
        });
    }
    if word >> 30 == 0b10 {
        return Some(match word >> 23 {
            0b1_0111_1101 => Format::Sop1,
            0b1_0111_1110 => Format::Sopc,
            0b1_0111_1111 => Format::Sopp,
            _ if word >> 28 == 0b1011 => Format::Sopk,
            _ => Format::Sop2,
        });
    }
    if word >> 23 == 0b1_1010_0111 {
        return Some(if (word >> 16) & 0x7f >= 0x40 {
            Format::Mai
        } else {
            Format::Vop3p
        });
    }
    Some(match word >> 26 {
        0b11_0000 => Format::Smem,
        0b11_0100 => Format::Vop3,
        0b11_0110 => Format::Ds,
        0b11_0111 => Format::Flat,
        0b11_1000 => Format::Mubuf,
        0b11_1010 => Format::Mtbuf,
        0b11_1100 => Format::Mimg,
        0b11_0001 => Format::Exp,
        _ => return None,
    })
}

/// Register files as the builder, ledger and hazard pass see them. VCC,
/// EXEC, FLAT_SCRATCH and XNACK_MASK are two dwords each (`base` 0 = lo,
/// 1 = hi); SCC and M0 are one unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegFile {
    Sgpr,
    Vgpr,
    Agpr,
    Ttmp,
    Vcc,
    Exec,
    M0,
    Scc,
    FlatScratch,
    XnackMask,
}

/// `len` consecutive 32-bit registers of one file starting at `base`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegSpan {
    pub file: RegFile,
    pub base: u8,
    pub len: u8,
}

impl RegSpan {
    pub fn new(file: RegFile, base: u8, len: u8) -> Self {
        Self { file, base, len }
    }
    /// The registers a `dwords`-wide operand starting at `src` names; `None`
    /// for constants, literals and the read-only `src_*` values (which are
    /// still state reads, see [`Footprint`] users: `Scc`, `Vccz` and `Execz`
    /// map to their underlying file).
    pub fn of(src: Src, dwords: u8) -> Option<Self> {
        let d = dwords.max(1);
        Some(match src {
            Src::Sgpr(n) => Self::new(RegFile::Sgpr, n, d),
            Src::Vgpr(n) => Self::new(RegFile::Vgpr, n, d),
            Src::Agpr(n) => Self::new(RegFile::Agpr, n, d),
            Src::Ttmp(n) => Self::new(RegFile::Ttmp, n, d),
            Src::VccLo => Self::new(RegFile::Vcc, 0, d.min(2)),
            Src::VccHi => Self::new(RegFile::Vcc, 1, 1),
            Src::ExecLo => Self::new(RegFile::Exec, 0, d.min(2)),
            Src::ExecHi => Self::new(RegFile::Exec, 1, 1),
            Src::FlatScratchLo => Self::new(RegFile::FlatScratch, 0, d.min(2)),
            Src::FlatScratchHi => Self::new(RegFile::FlatScratch, 1, 1),
            Src::XnackMaskLo => Self::new(RegFile::XnackMask, 0, d.min(2)),
            Src::XnackMaskHi => Self::new(RegFile::XnackMask, 1, 1),
            Src::M0 => Self::new(RegFile::M0, 0, 1),
            Src::Scc => Self::new(RegFile::Scc, 0, 1),
            Src::Vccz => Self::new(RegFile::Vcc, 0, 2),
            Src::Execz => Self::new(RegFile::Exec, 0, 2),
            _ => return None,
        })
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.file == other.file
            && u16::from(self.base) < u16::from(other.base) + u16::from(other.len)
            && u16::from(other.base) < u16::from(self.base) + u16::from(self.len)
    }
}

/// Every register an instruction reads (`uses`) and writes (`defs`),
/// explicit and implicit: EXEC for every VALU/VMEM/DS lane-masked op, VCC
/// for VOP2 carry ops / `v_cndmask_b32_e32` / VOPC, SCC for SOPC and the
/// SCC-writing SALU ops, M0 where the ISA reads it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    pub defs: Vec<RegSpan>,
    pub uses: Vec<RegSpan>,
}

impl Footprint {
    /// Record `src` (`dwords` wide) as a use; constants record nothing.
    pub fn read(&mut self, src: Src, dwords: u8) {
        if let Some(r) = RegSpan::of(src, dwords) {
            self.uses.push(r)
        }
    }
    /// Record `dst` (`dwords` wide) as a def.
    pub fn write(&mut self, dst: Src, dwords: u8) {
        if let Some(r) = RegSpan::of(dst, dwords) {
            self.defs.push(r)
        }
    }
    pub fn read_file(&mut self, file: RegFile, base: u8, len: u8) {
        self.uses.push(RegSpan::new(file, base, len))
    }
    pub fn write_file(&mut self, file: RegFile, base: u8, len: u8) {
        self.defs.push(RegSpan::new(file, base, len))
    }
}

/// What a memory instruction does to the wait counters (CDNA3 ISA §4.4
/// p.19-20): VMEM (MUBUF/global) counts on VM_CNT, DS and SMEM on LGKM_CNT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemKind {
    VmemLoad,
    VmemStore,
    VmemAtomic,
    VmemAtomicRtn,
    DsLoad,
    DsStore,
    DsAtomic,
    DsAtomicRtn,
    /// `ds_swizzle_b32`, `ds_permute_b32`, `ds_bpermute_b32`: cross-lane
    /// moves through the LDS crossbar that touch no LDS memory but return
    /// data under LGKM_CNT.
    DsXlane,
    SmemLoad,
    /// `s_store_*` / `s_scratch_store_*`: counts on LGKM_CNT, returns nothing.
    SmemStore,
    /// `s_atomic_*` / `s_buffer_atomic_*` without return (`GLC` = 0).
    SmemAtomic,
    /// `s_atomic_*` / `s_buffer_atomic_*` returning the pre-op value (`GLC` = 1).
    SmemAtomicRtn,
}

/// Whether the generated-table row of an instruction is confirmed by the
/// LLVM cross-check ([`Inst::verification`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowStatus {
    /// Cross-checked: the row assembles and disassembles identically in
    /// `llvm-mc`.
    Verified,
    /// Recognized but unconfirmed: the reason. Never emit, never seal.
    Unverified(&'static str),
    /// No XML row identifies the encoding (the generated decode refused the
    /// words, or the instruction does not encode): the reason.
    Unknown(String),
}

impl Inst {
    pub fn format(&self) -> Format {
        match self {
            Self::Sop1(_) => Format::Sop1,
            Self::Sop2(_) => Format::Sop2,
            Self::Sopk(_) => Format::Sopk,
            Self::Sopc(_) => Format::Sopc,
            Self::Sopp(_) => Format::Sopp,
            Self::Smem(_) => Format::Smem,
            Self::Vop1(_) => Format::Vop1,
            Self::Vop2(_) | Self::Vop2K(_) => Format::Vop2,
            Self::Vopc(_) => Format::Vopc,
            Self::Vop3(_) => Format::Vop3,
            Self::Vop3p(_) => Format::Vop3p,
            Self::Mai(_) => Format::Mai,
            Self::Ds(_) => Format::Ds,
            Self::Flat(_) => Format::Flat,
            Self::Mubuf(_) => Format::Mubuf,
            Self::Xml(i) => i.format(),
        }
    }

    /// The generated instruction, if this is one.
    pub fn xml(&self) -> Option<&XmlInst> {
        match self {
            Self::Xml(x) => Some(x),
            _ => None,
        }
    }

    /// The by-name refusal for an instruction without a hand-written model
    /// (every generated instruction); `None` for the typed instructions.
    pub fn unsupported(&self) -> Option<String> {
        self.xml().map(XmlInst::unsupported)
    }

    /// Whether the XML row this instruction encodes to is confirmed by the
    /// LLVM cross-check. Typed instructions are encoded and matched against
    /// the generated tables; a generated one reports its own row. A row the
    /// cross-check could not confirm decodes and encodes through the typed
    /// model unchanged, but emission (the builder) and seals must refuse it.
    pub fn verification(&self, arch: Arch) -> RowStatus {
        let status = |row: Result<(), &'static str>| match row {
            Ok(()) => RowStatus::Verified,
            Err(why) => RowStatus::Unverified(why),
        };
        if let Self::Xml(x) = self {
            return status(x.verification());
        }
        let words = match self.words(arch) {
            Ok(w) => w,
            Err(e) => return RowStatus::Unknown(e),
        };
        match crate::xml::decode(crate::xml_inst::isa(arch), &words) {
            Ok((d, _)) => status(XmlInst::verification_of(&d)),
            Err(e) => RowStatus::Unknown(format!("no XML row for the encoding: {e}")),
        }
    }

    /// Append the instruction's dwords (including a literal) to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        if !arch.encodes() {
            return Err(format!(
                "{} is decode-only: instructions cannot be encoded for it",
                arch.name()
            ));
        }
        match self {
            Self::Sop1(i) => i.encode(arch, out),
            Self::Sop2(i) => i.encode(arch, out),
            Self::Sopk(i) => i.encode(arch, out),
            Self::Sopc(i) => i.encode(arch, out),
            Self::Sopp(i) => i.encode(arch, out),
            Self::Smem(i) => i.encode(arch, out),
            Self::Vop1(i) => i.encode(arch, out),
            Self::Vop2(i) => i.encode(arch, out),
            Self::Vop2K(i) => i.encode(arch, out),
            Self::Vopc(i) => i.encode(arch, out),
            Self::Vop3(i) => i.encode(arch, out),
            Self::Vop3p(i) => i.encode(arch, out),
            Self::Mai(i) => i.encode(arch, out),
            Self::Ds(i) => i.encode(arch, out),
            Self::Flat(i) => i.encode(arch, out),
            Self::Mubuf(i) => i.encode(arch, out),
            Self::Xml(i) => i.encode(arch, out),
        }
    }

    pub fn words(&self, arch: Arch) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(3);
        self.encode(arch, &mut out)?;
        Ok(out)
    }

    /// Decode the instruction starting at `words[0]`; returns it and its
    /// dword count.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let first = *words.first().ok_or("decode past the end of the code")?;
        // gfx950 has no typed instructions: the generated tables decode it.
        if !arch.encodes() {
            return Self::decode_xml(arch, words, None);
        }
        let Some(format) = classify(first) else {
            return Self::decode_xml(arch, words, Some(format!("unknown encoding {first:#010x}")));
        };
        match Self::decode_typed(arch, words, format) {
            Ok(decoded) => Ok(decoded),
            // A generated decode preserves forms the typed model cannot
            // represent, without granting them execution or seal semantics.
            Err(e) => Self::decode_xml(arch, words, Some(e)),
        }
    }

    fn decode_xml(arch: Arch, words: &[u32], typed_error: Option<String>) -> Result<(Self, usize)> {
        match XmlInst::decode(arch, words) {
            Ok((x, n)) => Ok((Self::Xml(x), n)),
            Err(e) => Err(match typed_error {
                Some(t) => format!("{t}; generated tables: {e}"),
                None => e,
            }),
        }
    }

    fn decode_typed(arch: Arch, words: &[u32], format: Format) -> Result<(Self, usize)> {
        let first = words[0];
        Ok(match format {
            Format::Sop1 => {
                let (i, n) = Sop1::decode(arch, words)?;
                (Self::Sop1(i), n)
            }
            Format::Sop2 => {
                let (i, n) = Sop2::decode(arch, words)?;
                (Self::Sop2(i), n)
            }
            Format::Sopk => {
                let (i, n) = Sopk::decode(arch, words)?;
                (Self::Sopk(i), n)
            }
            Format::Sopc => {
                let (i, n) = Sopc::decode(arch, words)?;
                (Self::Sopc(i), n)
            }
            Format::Sopp => {
                let (i, n) = Sopp::decode(arch, words)?;
                (Self::Sopp(i), n)
            }
            Format::Smem => {
                let (i, n) = Smem::decode(arch, words)?;
                (Self::Smem(i), n)
            }
            Format::Vop1 => {
                let (i, n) = Vop1::decode(arch, words)?;
                (Self::Vop1(i), n)
            }
            Format::Vop2 => {
                if Vop2KOp::from_opcode(arch, (first >> 25) as u8).is_some() {
                    let (i, n) = Vop2K::decode(arch, words)?;
                    (Self::Vop2K(i), n)
                } else {
                    let (i, n) = Vop2::decode(arch, words)?;
                    (Self::Vop2(i), n)
                }
            }
            Format::Vopc => {
                let (i, n) = Vopc::decode(arch, words)?;
                (Self::Vopc(i), n)
            }
            Format::Vop3 => {
                let (i, n) = Vop3::decode(arch, words)?;
                (Self::Vop3(i), n)
            }
            Format::Vop3p => {
                let (i, n) = Vop3p::decode(arch, words)?;
                (Self::Vop3p(i), n)
            }
            Format::Mai => {
                let (i, n) = Mai::decode(arch, words)?;
                (Self::Mai(i), n)
            }
            Format::Ds => {
                let (i, n) = Ds::decode(arch, words)?;
                (Self::Ds(i), n)
            }
            Format::Flat => {
                let (i, n) = Flat::decode(arch, words)?;
                (Self::Flat(i), n)
            }
            Format::Mubuf => {
                let (i, n) = Mubuf::decode(arch, words)?;
                (Self::Mubuf(i), n)
            }
            other => {
                return Err(format!(
                "{other:?} encoding {first:#010x} is outside the PeaceMaker CDNA instruction table"
            ))
            }
        })
    }

    pub fn footprint(&self) -> Footprint {
        match self {
            Self::Sop1(i) => i.footprint(),
            Self::Sop2(i) => i.footprint(),
            Self::Sopk(i) => i.footprint(),
            Self::Sopc(i) => i.footprint(),
            Self::Sopp(i) => i.footprint(),
            Self::Smem(i) => i.footprint(),
            Self::Vop1(i) => i.footprint(),
            Self::Vop2(i) => i.footprint(),
            Self::Vop2K(i) => i.footprint(),
            Self::Vopc(i) => i.footprint(),
            Self::Vop3(i) => i.footprint(),
            Self::Vop3p(i) => i.footprint(),
            Self::Mai(i) => i.footprint(),
            Self::Ds(i) => i.footprint(),
            Self::Flat(i) => i.footprint(),
            Self::Mubuf(i) => i.footprint(),
            Self::Xml(i) => i.footprint(),
        }
    }

    pub fn memory(&self) -> Option<MemKind> {
        match self {
            Self::Smem(i) => i.mem_kind(),
            Self::Ds(i) => Some(i.mem_kind()),
            Self::Flat(i) => Some(i.mem_kind()),
            Self::Mubuf(i) => Some(i.mem_kind()),
            Self::Xml(i) => i.memory(),
            _ => None,
        }
    }

    /// `s_branch` / `s_cbranch_*`: the signed dword offset from the next
    /// instruction (`SIMM16`, CDNA3 ISA §4.2 p.18: PC = PC + 4 + SIMM16 * 4).
    pub fn branch_offset(&self) -> Option<i16> {
        match self {
            Self::Sopp(p) => p.branch_offset(),
            Self::Xml(x) => x.branch_offset(),
            _ => None,
        }
    }
    /// Lossless assembler spelling. Unlike LLVM's display spelling, this
    /// preserves the VOP3 NOP form and every DS swizzle offset bit.
    pub fn assembly(&self) -> impl fmt::Display + '_ {
        Assembly(self)
    }
}

struct Assembly<'a>(&'a Inst);

impl fmt::Display for Assembly<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Inst::Vop3(i) if i.op == crate::enc::vop3::Vop3Op::V1(crate::enc::vop::Vop1Op::Nop) => {
                f.write_str("v_nop_e64")
            }
            Inst::Ds(i) => i.fmt_assembly(f),
            inst => fmt::Display::fmt(inst, f),
        }
    }
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sop1(i) => i.fmt(f),
            Self::Sop2(i) => i.fmt(f),
            Self::Sopk(i) => i.fmt(f),
            Self::Sopc(i) => i.fmt(f),
            Self::Sopp(i) => i.fmt(f),
            Self::Smem(i) => i.fmt(f),
            Self::Vop1(i) => i.fmt(f),
            Self::Vop2(i) => i.fmt(f),
            Self::Vop2K(i) => i.fmt(f),
            Self::Vopc(i) => i.fmt(f),
            Self::Vop3(i) => i.fmt(f),
            Self::Vop3p(i) => i.fmt(f),
            Self::Mai(i) => i.fmt(f),
            Self::Ds(i) => i.fmt(f),
            Self::Flat(i) => i.fmt(f),
            Self::Mubuf(i) => i.fmt(f),
            Self::Xml(i) => fmt::Display::fmt(i, f),
        }
    }
}

/// Encode a straight-line instruction list.
pub fn encode_all(arch: Arch, insts: &[Inst]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(insts.len() * 2);
    for (index, inst) in insts.iter().enumerate() {
        inst.encode(arch, &mut out)
            .map_err(|e| format!("instruction {index} ({inst:?}): {e}"))?
    }
    Ok(out)
}

/// Decode a whole code range into `(dword offset, instruction)` pairs.
pub fn decode_all(arch: Arch, words: &[u32]) -> Result<Vec<(usize, Inst)>> {
    let mut at = 0;
    let mut out = Vec::new();
    while at < words.len() {
        let (inst, n) = Inst::decode(arch, &words[at..]).map_err(|e| format!("dword {at}: {e}"))?;
        out.push((at, inst));
        at += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classify_llvm_mc_words() {
        // `llvm-mc -mcpu=gfx942 -show-encoding` (ROCm 10.1, LLVM 24).
        assert_eq!(classify(0xd3f3_0000), Some(Format::Mai)); // v_mfma_f32_16x16x32_fp8_fp8
        assert_eq!(classify(0xd3d8_4001), Some(Format::Mai)); // v_accvgpr_read_b32 v1, a2
        assert_eq!(classify(0xd2a2_0001), Some(Format::Vop3)); // v_cvt_pk_fp8_f32 v1, v2, v3
        assert_eq!(classify(0xbf8c_0070), Some(Format::Sopp)); // s_waitcnt vmcnt(0) lgkmcnt(0)
        assert_eq!(classify(0xdc5c_8010), Some(Format::Flat)); // global_load_dwordx4
        assert_eq!(classify(0x7e02_02fa), Some(Format::Vop1)); // v_mov_b32_dpp
    }
}
