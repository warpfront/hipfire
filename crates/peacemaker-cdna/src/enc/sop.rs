// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar ALU formats SOP1, SOP2, SOPK, SOPC, SOPP (CDNA3 ISA §13.1 p.513-520,
//! §12.1-12.5 p.88-140).
//!
//! Field layouts (CDNA3 ISA §13.1): SOP2 `[31:30]=10 OP[29:23] SDST[22:16]
//! SSRC1[15:8] SSRC0[7:0]`; SOPK `[31:28]=1011 OP[27:23] SDST[22:16]
//! SIMM16[15:0]`; SOP1 `[31:23]=1_0111_1101 SDST[22:16] OP[15:8] SSRC0[7:0]`;
//! SOPC `[31:23]=1_0111_1110 OP[22:16] SSRC1[15:8] SSRC0[7:0]`; SOPP
//! `[31:23]=1_0111_1111 OP[22:16] SIMM16[15:0]`. SOP1/SOP2/SOPC sources may
//! name the literal code 255: one 32-bit literal dword follows the
//! instruction (both SOP2/SOPC sources may name it only with the same value).
//! SOPK and SOPP never carry a literal.
//!
//! Opcode numbers come from the CDNA3 machine-readable XML and are proven
//! against `llvm-mc -arch=amdgcn -mcpu=gfx942 -show-encoding`. `Display` is
//! exactly the text llvm's gfx942 disassembler prints (`llvm-mc
//! -disassemble`): literals whose value an inline constant could carry print
//! as `lit(0x…)`, branch offsets print as the unsigned SIMM16, `s_waitcnt`
//! omits counters at their maximum, `hwreg(...)`/`sendmsg(...)` use llvm's
//! symbolic names. Encoders and decoders accept only what that text
//! reassembles to bit-exactly (e.g. `s_waitcnt` with its unused SIMM16 bits
//! set, or `s_barrier` with a nonzero SIMM16, is rejected).
use crate::arch::Arch;
use crate::inst::{classify, Footprint, Format, RegFile};
use crate::operand::{InlineFloat, Src, LITERAL};
use crate::Result;
use std::fmt;

op_table!(Sop2Op {
    AddU32,
    SubU32,
    AddI32,
    SubI32,
    AddcU32,
    SubbU32,
    MinI32,
    MinU32,
    MaxI32,
    MaxU32,
    CselectB32,
    CselectB64,
    AndB32,
    AndB64,
    OrB32,
    OrB64,
    XorB32,
    XorB64,
    Andn2B32,
    Andn2B64,
    Orn2B32,
    Orn2B64,
    LshlB32,
    LshlB64,
    LshrB32,
    LshrB64,
    AshrI32,
    AshrI64,
    BfmB32,
    MulI32,
    BfeU32,
    BfeI32,
    MulHiU32,
    MulHiI32,
    Lshl1AddU32,
    Lshl2AddU32,
    Lshl3AddU32,
    Lshl4AddU32,
    PackLlB32B16,
    PackLhB32B16,
    PackHhB32B16,
    AbsdiffI32,
});

op_table!(Sop1Op {
    MovB32,
    MovB64,
    CmovB32,
    CmovB64,
    NotB32,
    NotB64,
    BrevB32,
    Bcnt1I32B32,
    Bcnt1I32B64,
    Ff1I32B32,
    Ff1I32B64,
    FlbitI32B32,
    SextI32I8,
    SextI32I16,
    AndSaveexecB64,
    OrSaveexecB64,
    XorSaveexecB64,
    Andn2SaveexecB64,
    GetpcB64,
    SetpcB64,
    AbsI32,
    Bitset0B32,
    Bitset1B32,
});

op_table!(SopkOp {
    MovkI32,
    CmovkI32,
    CmpkEqI32,
    CmpkLgI32,
    CmpkGtI32,
    CmpkGeI32,
    CmpkLtI32,
    CmpkLeI32,
    CmpkEqU32,
    CmpkLgU32,
    CmpkGtU32,
    CmpkGeU32,
    CmpkLtU32,
    CmpkLeU32,
    AddkI32,
    MulkI32,
    GetregB32,
    SetregB32,
});

op_table!(SopcOp {
    CmpEqI32,
    CmpLgI32,
    CmpGtI32,
    CmpGeI32,
    CmpLtI32,
    CmpLeI32,
    CmpEqU32,
    CmpLgU32,
    CmpGtU32,
    CmpGeU32,
    CmpLtU32,
    CmpLeU32,
    Bitcmp0B32,
    Bitcmp1B32,
    CmpEqU64,
    CmpLgU64,
});

op_table!(SoppOp {
    Nop,
    Endpgm,
    Branch,
    CbranchScc0,
    CbranchScc1,
    CbranchVccz,
    CbranchVccnz,
    CbranchExecz,
    CbranchExecnz,
    Barrier,
    Waitcnt,
    Sleep,
    Setprio,
    Sendmsg,
    Trap,
    IcacheInv,
});

macro_rules! op_meta {
    ($t:ty, $w:ty) => {
        impl $t {
            /// Inverse of [`Self::opcode`].
            pub fn from_opcode(arch: Arch, opcode: $w) -> Option<Self> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|op| op.opcode(arch) == Some(opcode))
            }
        }
    };
}
op_meta!(Sop2Op, u8);
op_meta!(Sop1Op, u8);
op_meta!(SopkOp, u8);
op_meta!(SopcOp, u8);
op_meta!(SoppOp, u8);

impl Sop2Op {
    /// Dword widths of (sdst, ssrc0, ssrc1). The 64-bit shifts take a 32-bit
    /// shift count in SSRC1 (CDNA3 ISA §12.1 p.94-95).
    pub fn widths(self) -> (u8, u8, u8) {
        use Sop2Op::*;
        match self {
            CselectB64 | AndB64 | OrB64 | XorB64 | Andn2B64 | Orn2B64 => (2, 2, 2),
            LshlB64 | LshrB64 | AshrI64 => (2, 2, 1),
            _ => (1, 1, 1),
        }
    }
}
impl Sop1Op {
    /// Dword widths of (sdst, ssrc0); 0 where the operand is absent
    /// (`s_getpc_b64` has no source, `s_setpc_b64` no destination).
    pub fn widths(self) -> (u8, u8) {
        use Sop1Op::*;
        match self {
            MovB64 | CmovB64 | NotB64 | AndSaveexecB64 | OrSaveexecB64 | XorSaveexecB64
            | Andn2SaveexecB64 => (2, 2),
            Bcnt1I32B64 | Ff1I32B64 => (1, 2),
            GetpcB64 => (2, 0),
            SetpcB64 => (0, 2),
            _ => (1, 1),
        }
    }
}
impl SopcOp {
    /// Dword width of both sources.
    pub fn width(self) -> u8 {
        if matches!(self, SopcOp::CmpEqU64 | SopcOp::CmpLgU64) {
            2
        } else {
            1
        }
    }
}

/// `sdst = op(ssrc0, ssrc1)`; at most one source may be a literal.
#[derive(Clone, Debug, PartialEq)]
pub struct Sop2 {
    pub op: Sop2Op,
    pub sdst: Src,
    pub ssrc0: Src,
    pub ssrc1: Src,
}
/// `sdst = op(ssrc0)`; `None` where the opcode has no such operand (the
/// field encodes as 0 and decodes back to `None`).
#[derive(Clone, Debug, PartialEq)]
pub struct Sop1 {
    pub op: Sop1Op,
    pub sdst: Option<Src>,
    pub ssrc0: Option<Src>,
}
/// 16-bit immediate forms. `s_getreg_b32`/`s_setreg_b32`: `sdst` is the SGPR
/// read or written and `simm16` the hwreg(id, offset, size) field.
#[derive(Clone, Debug, PartialEq)]
pub struct Sopk {
    pub op: SopkOp,
    pub sdst: Src,
    pub simm16: u16,
}
/// `SCC = op(ssrc0, ssrc1)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sopc {
    pub op: SopcOp,
    pub ssrc0: Src,
    pub ssrc1: Src,
}
/// Program control. Branches carry the signed dword offset from the next
/// instruction in `simm16`; `s_waitcnt` carries [`waitcnt_imm`].
#[derive(Clone, Debug, PartialEq)]
pub struct Sopp {
    pub op: SoppOp,
    pub simm16: u16,
}

impl Sopp {
    /// `Some(simm16 as i16)` for `s_branch` / `s_cbranch_*`.
    pub fn branch_offset(&self) -> Option<i16> {
        matches!(
            self.op,
            SoppOp::Branch
                | SoppOp::CbranchScc0
                | SoppOp::CbranchScc1
                | SoppOp::CbranchVccz
                | SoppOp::CbranchVccnz
                | SoppOp::CbranchExecz
                | SoppOp::CbranchExecnz
        )
        .then_some(self.simm16 as i16)
    }
}

/// `s_waitcnt` SIMM16 bits no counter field uses on gfx9 (7, 12, 13). The
/// hardware ignores them, but llvm's text drops them, so a word with any of
/// them set has no exact text form and is rejected.
const WAITCNT_UNUSED: u16 = 0x3080;

/// The `s_waitcnt` SIMM16 for the given counter thresholds (gfx9 layout:
/// VM_CNT[3:0] in bits 3:0 and VM_CNT[5:4] in bits 15:14, EXP_CNT bits 6:4,
/// LGKM_CNT bits 11:8; CDNA3 ISA §12.5 S_WAITCNT p.133). Errors past the field
/// maxima of `arch` ([`crate::Caps`]).
pub fn waitcnt_imm(arch: Arch, vm: u8, exp: u8, lgkm: u8) -> Result<u16> {
    let caps = arch.caps();
    if vm > caps.vmcnt_max {
        return Err(format!(
            "s_waitcnt vmcnt({vm}) exceeds the {} maximum {}",
            caps.name, caps.vmcnt_max
        ));
    }
    if exp > caps.expcnt_max {
        return Err(format!(
            "s_waitcnt expcnt({exp}) exceeds the {} maximum {}",
            caps.name, caps.expcnt_max
        ));
    }
    if lgkm > caps.lgkmcnt_max {
        return Err(format!(
            "s_waitcnt lgkmcnt({lgkm}) exceeds the {} maximum {}",
            caps.name, caps.lgkmcnt_max
        ));
    }
    let (vm, exp, lgkm) = (u16::from(vm), u16::from(exp), u16::from(lgkm));
    Ok(match arch {
        Arch::Gfx942 => (vm & 0xf) | (vm >> 4) << 14 | exp << 4 | lgkm << 8,
    })
}
/// Inverse of [`waitcnt_imm`]: `(vm, exp, lgkm)`. Unused SIMM16 bits are
/// ignored, as the hardware ignores them.
pub fn waitcnt_fields(arch: Arch, simm16: u16) -> (u8, u8, u8) {
    match arch {
        Arch::Gfx942 => (
            ((simm16 & 0xf) | ((simm16 >> 14) & 0x3) << 4) as u8,
            ((simm16 >> 4) & 0x7) as u8,
            ((simm16 >> 8) & 0xf) as u8,
        ),
    }
}

/// Check that `src` can occupy a `dwords`-wide scalar operand field (`dst`:
/// a 7-bit SDST field). 64-bit register operands start at an even SGPR or
/// TTMP, or are `vcc`/`exec`/`flat_scratch`/`xnack_mask` (CDNA3 ISA §5.2
/// p.25-26; llvm rejects misaligned pairs).
fn check_operand(arch: Arch, src: Src, dwords: u8, dst: bool, what: &str) -> Result<()> {
    let bad = |why: &str| Err(format!("{what} {src:?}: {why}"));
    if dst && !src.is_sreg() {
        return bad("not a scalar register destination");
    }
    let wide = dwords == 2;
    match src {
        Src::Vgpr(_) | Src::Agpr(_) => bad("vector registers are not SALU operands"),
        Src::Sgpr(n) if u16::from(n) + u16::from(dwords.max(1)) > arch.caps().sgprs => {
            bad("past the last addressable SGPR")
        }
        Src::Ttmp(n) if u16::from(n) + u16::from(dwords.max(1)) > 16 => bad("past ttmp15"),
        Src::Sgpr(n) | Src::Ttmp(n) if wide && n % 2 != 0 => {
            bad("a 64-bit register pair must start at an even register")
        }
        Src::VccHi | Src::ExecHi | Src::FlatScratchHi | Src::XnackMaskHi | Src::M0 if wide => {
            bad("a 32-bit register cannot be a 64-bit operand")
        }
        Src::Int(i) if !(-16..=64).contains(&i) => {
            bad("inline integers are -16..=64 (use a literal)")
        }
        _ => Ok(()),
    }
}

/// The literal dword the sources name, if any; two literal sources must
/// carry the same value (they share the one literal dword).
fn shared_literal(what: &str, srcs: &[Src]) -> Result<Option<u32>> {
    let mut lit = None;
    for value in srcs.iter().filter_map(|s| s.literal()) {
        match lit {
            Some(prev) if prev != value => return Err(format!("{what}: two different literals {prev:#x} and {value:#x}; the format carries one literal dword")),
            _ => lit = Some(value),
        }
    }
    Ok(lit)
}

/// The literal dword following `words[0]`, when a source field holds code 255.
fn literal_after(what: &str, words: &[u32], codes: &[u16]) -> Result<Option<u32>> {
    if !codes.contains(&LITERAL) {
        return Ok(None);
    }
    words
        .get(1)
        .copied()
        .map(Some)
        .ok_or_else(|| format!("{what}: literal operand without a literal dword"))
}

/// A literal value an inline constant of the operand's type could also
/// carry; llvm's disassembler prints such a literal as `lit(0x…)` so that it
/// reassembles as a literal. 32-bit operands: integers -16..=64 and the
/// nine inline floats' f32 patterns. 64-bit operands: llvm treats the 32-bit
/// literal as zero-extended (`0xfffffff0` stays a literal, not -16), so only
/// 0..=64 collides (proven by the enc_sop llvm-mc tests). How the SALU
/// extends a 32-bit literal for a 64-bit operand is the emulator's concern.
/// [UNVERIFIED: confirm on MI300X] zero- vs sign-extension of a SALU 64-bit
/// literal (§6.2.1 p.32 states the rule only for VALU: unsigned zero-extend,
/// signed sign-extend).
fn literal_is_inlinable(value: u32, dwords: u8) -> bool {
    if dwords == 2 {
        return value <= 64;
    }
    (-16..=64).contains(&(value as i32)) || InlineFloat::ALL.iter().any(|f| f.f32_bits() == value)
}

/// One source/destination as llvm prints it in a `dwords`-wide field.
fn operand_text(src: Src, dwords: u8) -> String {
    match src {
        Src::Lit(v) if literal_is_inlinable(v, dwords) => format!("lit({v:#x})"),
        // 64-bit operands read the f64 1/(2*pi) pattern and llvm prints its
        // shortest round-trip decimal.
        Src::Float(InlineFloat::InvTwoPi) if dwords == 2 => "0.15915494309189532".into(),
        _ => src.fmt(dwords),
    }
}

/// Fetch and classify the first dword of a scalar format.
fn first_word(words: &[u32], format: Format) -> Result<u32> {
    let word = *words
        .first()
        .ok_or_else(|| format!("{format:?}: decode past the end of the code"))?;
    if classify(word) != Some(format) {
        return Err(format!("{word:#010x} is not a {format:?} encoding"));
    }
    Ok(word)
}

impl Sop2 {
    /// Operand legality: register kinds, pair alignment, one literal dword.
    fn check(&self, arch: Arch) -> Result<Option<u32>> {
        let (wd, w0, w1) = self.op.widths();
        let what = self.op.name();
        check_operand(arch, self.sdst, wd, true, what)?;
        check_operand(arch, self.ssrc0, w0, false, what)?;
        check_operand(arch, self.ssrc1, w1, false, what)?;
        shared_literal(what, &[self.ssrc0, self.ssrc1])
    }

    /// Append this instruction's dwords (and literal) to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let op = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{} does not exist on {arch}", self.op.name()))?;
        let lit = self.check(arch)?;
        out.push(
            0b10 << 30
                | u32::from(op) << 23
                | u32::from(self.sdst.code()) << 16
                | u32::from(self.ssrc1.code()) << 8
                | u32::from(self.ssrc0.code()),
        );
        out.extend(lit);
        Ok(())
    }
    /// Decode from `words[0..]`; returns the dword count consumed.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let word = first_word(words, Format::Sop2)?;
        let opcode = ((word >> 23) & 0x7f) as u8;
        let op = Sop2Op::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SOP2 opcode {opcode} is not in the {arch} table"))?;
        let (c0, c1) = ((word & 0xff) as u16, ((word >> 8) & 0xff) as u16);
        let lit = literal_after(op.name(), words, &[c0, c1])?;
        let inst = Self {
            op,
            sdst: Src::from_code(((word >> 16) & 0x7f) as u16, None)?,
            ssrc0: Src::from_code(c0, lit)?,
            ssrc1: Src::from_code(c1, lit)?,
        };
        inst.check(arch)?;
        Ok((inst, 1 + usize::from(lit.is_some())))
    }
    /// Registers read and written, including SCC implicits (CDNA3 ISA §12.1
    /// p.88-100: carry/borrow ops read and write SCC, `s_cselect_*` reads it,
    /// arithmetic/logic/shift/BFE/min/max/absdiff/lshlN_add write it;
    /// `s_bfm`, `s_mul_*`, `s_pack_*` leave it).
    pub fn footprint(&self) -> Footprint {
        use Sop2Op::*;
        let (wd, w0, w1) = self.op.widths();
        let mut f = Footprint::default();
        f.read(self.ssrc0, w0);
        f.read(self.ssrc1, w1);
        if matches!(self.op, AddcU32 | SubbU32 | CselectB32 | CselectB64) {
            f.read_file(RegFile::Scc, 0, 1)
        }
        f.write(self.sdst, wd);
        if !matches!(
            self.op,
            CselectB32
                | CselectB64
                | BfmB32
                | MulI32
                | MulHiU32
                | MulHiI32
                | PackLlB32B16
                | PackLhB32B16
                | PackHhB32B16
        ) {
            f.write_file(RegFile::Scc, 0, 1)
        }
        f
    }
}
impl fmt::Display for Sop2 {
    /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wd, w0, w1) = self.op.widths();
        write!(
            f,
            "{} {}, {}, {}",
            self.op.name(),
            operand_text(self.sdst, wd),
            operand_text(self.ssrc0, w0),
            operand_text(self.ssrc1, w1)
        )
    }
}

impl Sop1 {
    fn check(&self, arch: Arch) -> Result<Option<u32>> {
        let (wd, w0) = self.op.widths();
        let what = self.op.name();
        match (wd, self.sdst) {
            (0, Some(d)) => return Err(format!("{what} has no destination (got {d:?})")),
            (0, None) => {}
            (_, None) => return Err(format!("{what} needs a destination")),
            (_, Some(d)) => check_operand(arch, d, wd, true, what)?,
        }
        match (w0, self.ssrc0) {
            (0, Some(s)) => Err(format!("{what} has no source (got {s:?})")),
            (0, None) => Ok(None),
            (_, None) => Err(format!("{what} needs a source")),
            // s_setpc_b64 takes a register pair only (llvm's SReg_64 operand:
            // a constant source has no assembler text).
            (_, Some(s)) if self.op == Sop1Op::SetpcB64 && !s.is_sreg() => Err(format!(
                "{what} needs a scalar register pair source (got {s:?})"
            )),
            (_, Some(s)) => {
                check_operand(arch, s, w0, false, what)?;
                Ok(s.literal())
            }
        }
    }

    /// Append this instruction's dwords (and literal) to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let op = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{} does not exist on {arch}", self.op.name()))?;
        let lit = self.check(arch)?;
        let sdst = self.sdst.map_or(0, |d| u32::from(d.code()));
        let ssrc0 = self.ssrc0.map_or(0, |s| u32::from(s.code()));
        out.push(0b1_0111_1101 << 23 | sdst << 16 | u32::from(op) << 8 | ssrc0);
        out.extend(lit);
        Ok(())
    }
    /// Decode from `words[0..]`; returns the dword count consumed. A field
    /// the opcode does not use must be 0.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let word = first_word(words, Format::Sop1)?;
        let opcode = ((word >> 8) & 0xff) as u8;
        let op = Sop1Op::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SOP1 opcode {opcode} is not in the {arch} table"))?;
        let (wd, w0) = op.widths();
        let (cd, c0) = (((word >> 16) & 0x7f) as u16, (word & 0xff) as u16);
        if wd == 0 && cd != 0 {
            return Err(format!("{}: unused SDST field is {cd}, not 0", op.name()));
        }
        if w0 == 0 && c0 != 0 {
            return Err(format!("{}: unused SSRC0 field is {c0}, not 0", op.name()));
        }
        let lit = if w0 == 0 {
            None
        } else {
            literal_after(op.name(), words, &[c0])?
        };
        let sdst = if wd == 0 {
            None
        } else {
            Some(Src::from_code(cd, None)?)
        };
        let ssrc0 = if w0 == 0 {
            None
        } else {
            Some(Src::from_code(c0, lit)?)
        };
        let inst = Self { op, sdst, ssrc0 };
        inst.check(arch)?;
        Ok((inst, 1 + usize::from(lit.is_some())))
    }
    /// Registers read and written (CDNA3 ISA §12.3 p.107-124): `s_cmov_*`
    /// read SCC and keep the old destination when it is 0 (a destination
    /// use); `s_bitset*` modify one bit of the destination (a use);
    /// `s_not_*`/`s_bcnt1_*`/`s_abs_i32` write SCC; `s_*_saveexec_b64`
    /// read and write EXEC and write SCC (p.115-117).
    pub fn footprint(&self) -> Footprint {
        use Sop1Op::*;
        let (wd, w0) = self.op.widths();
        let mut f = Footprint::default();
        if let Some(s) = self.ssrc0 {
            f.read(s, w0)
        }
        let conditional_or_rmw = matches!(self.op, CmovB32 | CmovB64 | Bitset0B32 | Bitset1B32);
        if let Some(d) = self.sdst.filter(|_| conditional_or_rmw) {
            f.read(d, wd)
        }
        if matches!(self.op, CmovB32 | CmovB64) {
            f.read_file(RegFile::Scc, 0, 1)
        }
        let saveexec = matches!(
            self.op,
            AndSaveexecB64 | OrSaveexecB64 | XorSaveexecB64 | Andn2SaveexecB64
        );
        if saveexec {
            f.read_file(RegFile::Exec, 0, 2)
        }
        if let Some(d) = self.sdst {
            f.write(d, wd)
        }
        if saveexec {
            f.write_file(RegFile::Exec, 0, 2)
        }
        if saveexec
            || matches!(
                self.op,
                NotB32 | NotB64 | Bcnt1I32B32 | Bcnt1I32B64 | AbsI32
            )
        {
            f.write_file(RegFile::Scc, 0, 1)
        }
        f
    }
}
impl fmt::Display for Sop1 {
    /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (wd, w0) = self.op.widths();
        let ops: Vec<String> = [(self.sdst, wd), (self.ssrc0, w0)]
            .into_iter()
            .filter_map(|(s, w)| s.map(|s| operand_text(s, w)))
            .collect();
        write!(f, "{} {}", self.op.name(), ops.join(", "))
    }
}

/// llvm's symbolic `HW_REG_*` names on gfx942 (CDNA3 ISA §5.8 Table 19 p.29;
/// spellings as llvm prints them, checked exhaustively by the tests).
fn hwreg_name(id: u16) -> Option<&'static str> {
    Some(match id {
        1 => "HW_REG_MODE",
        2 => "HW_REG_STATUS",
        3 => "HW_REG_TRAPSTS",
        4 => "HW_REG_HW_ID",
        5 => "HW_REG_GPR_ALLOC",
        6 => "HW_REG_LDS_ALLOC",
        7 => "HW_REG_IB_STS",
        15 => "HW_REG_SH_MEM_BASES",
        16 => "HW_REG_TBA_LO",
        17 => "HW_REG_TBA_HI",
        18 => "HW_REG_TMA_LO",
        19 => "HW_REG_TMA_HI",
        20 => "HW_REG_XCC_ID",
        21 => "HW_REG_SQ_PERF_SNAPSHOT_DATA",
        22 => "HW_REG_SQ_PERF_SNAPSHOT_DATA1",
        23 => "HW_REG_SQ_PERF_SNAPSHOT_PC_LO",
        24 => "HW_REG_SQ_PERF_SNAPSHOT_PC_HI",
        _ => return None,
    })
}

/// `hwreg(id[, offset, size])` for the S_GETREG/S_SETREG SIMM16
/// `{size-1[15:11], offset[10:6], id[5:0]}` (CDNA3 ISA §12.2 p.103-104);
/// llvm omits offset and size when they are 0 and 32.
fn hwreg_text(simm16: u16) -> String {
    let id = simm16 & 0x3f;
    let offset = (simm16 >> 6) & 0x1f;
    let size = (simm16 >> 11) + 1;
    let name = hwreg_name(id).map_or_else(|| id.to_string(), str::to_string);
    if offset == 0 && size == 32 {
        format!("hwreg({name})")
    } else {
        format!("hwreg({name}, {offset}, {size})")
    }
}

impl Sopk {
    fn check(&self, arch: Arch) -> Result<()> {
        check_operand(arch, self.sdst, 1, true, self.op.name())
    }

    /// Append this instruction's dword to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let op = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{} does not exist on {arch}", self.op.name()))?;
        self.check(arch)?;
        out.push(
            0b1011 << 28
                | u32::from(op) << 23
                | u32::from(self.sdst.code()) << 16
                | u32::from(self.simm16),
        );
        Ok(())
    }
    /// Decode from `words[0..]`; returns the dword count consumed (always 1).
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let word = first_word(words, Format::Sopk)?;
        let opcode = ((word >> 23) & 0x1f) as u8;
        let op = SopkOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SOPK opcode {opcode} is not in the {arch} table"))?;
        let inst = Self {
            op,
            sdst: Src::from_code(((word >> 16) & 0x7f) as u16, None)?,
            simm16: word as u16,
        };
        inst.check(arch)?;
        Ok((inst, 1))
    }
    /// Registers read and written (CDNA3 ISA §12.2 p.100-104): `s_cmpk_*`
    /// read SDST and write SCC; `s_addk_i32` reads and writes SDST and
    /// writes SCC; `s_mulk_i32` reads and writes SDST; `s_cmovk_i32` reads
    /// SCC and keeps the old SDST when it is 0 (a use); `s_setreg_b32` reads
    /// SDST, `s_getreg_b32` writes it. `s_getreg_b32` of HW_REG_STATUS reads
    /// the SCC/EXECZ/VCCZ bits (§3.4 Table 3 p.9-10), recorded as uses of SCC,
    /// EXEC and VCC whatever the offset/size. The hardware registers
    /// themselves (MODE, TRAPSTS, ...) are not register files here.
    pub fn footprint(&self) -> Footprint {
        use SopkOp::*;
        let mut f = Footprint::default();
        match self.op {
            MovkI32 => f.write(self.sdst, 1),
            CmovkI32 => {
                f.read_file(RegFile::Scc, 0, 1);
                f.read(self.sdst, 1);
                f.write(self.sdst, 1)
            }
            CmpkEqI32 | CmpkLgI32 | CmpkGtI32 | CmpkGeI32 | CmpkLtI32 | CmpkLeI32 | CmpkEqU32
            | CmpkLgU32 | CmpkGtU32 | CmpkGeU32 | CmpkLtU32 | CmpkLeU32 => {
                f.read(self.sdst, 1);
                f.write_file(RegFile::Scc, 0, 1)
            }
            AddkI32 => {
                f.read(self.sdst, 1);
                f.write(self.sdst, 1);
                f.write_file(RegFile::Scc, 0, 1)
            }
            MulkI32 => {
                f.read(self.sdst, 1);
                f.write(self.sdst, 1)
            }
            GetregB32 => {
                if self.simm16 & 0x3f == 2 {
                    f.read_file(RegFile::Scc, 0, 1);
                    f.read_file(RegFile::Exec, 0, 2);
                    f.read_file(RegFile::Vcc, 0, 2);
                }
                f.write(self.sdst, 1)
            }
            SetregB32 => f.read(self.sdst, 1),
        }
        f
    }
}
impl fmt::Display for Sopk {
    /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.op.name();
        let reg = operand_text(self.sdst, 1);
        match self.op {
            SopkOp::GetregB32 => write!(f, "{name} {reg}, {}", hwreg_text(self.simm16)),
            SopkOp::SetregB32 => write!(f, "{name} {}, {reg}", hwreg_text(self.simm16)),
            _ => write!(f, "{name} {reg}, {:#x}", self.simm16),
        }
    }
}

impl Sopc {
    fn check(&self, arch: Arch) -> Result<Option<u32>> {
        let w = self.op.width();
        let what = self.op.name();
        check_operand(arch, self.ssrc0, w, false, what)?;
        check_operand(arch, self.ssrc1, w, false, what)?;
        shared_literal(what, &[self.ssrc0, self.ssrc1])
    }

    /// Append this instruction's dwords (and literal) to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let op = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{} does not exist on {arch}", self.op.name()))?;
        let lit = self.check(arch)?;
        out.push(
            0b1_0111_1110 << 23
                | u32::from(op) << 16
                | u32::from(self.ssrc1.code()) << 8
                | u32::from(self.ssrc0.code()),
        );
        out.extend(lit);
        Ok(())
    }
    /// Decode from `words[0..]`; returns the dword count consumed.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let word = first_word(words, Format::Sopc)?;
        let opcode = ((word >> 16) & 0x7f) as u8;
        let op = SopcOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SOPC opcode {opcode} is not in the {arch} table"))?;
        let (c0, c1) = ((word & 0xff) as u16, ((word >> 8) & 0xff) as u16);
        let lit = literal_after(op.name(), words, &[c0, c1])?;
        let inst = Self {
            op,
            ssrc0: Src::from_code(c0, lit)?,
            ssrc1: Src::from_code(c1, lit)?,
        };
        inst.check(arch)?;
        Ok((inst, 1 + usize::from(lit.is_some())))
    }
    /// Both sources read, SCC written (CDNA3 ISA §12.4 p.125-129).
    pub fn footprint(&self) -> Footprint {
        let w = self.op.width();
        let mut f = Footprint::default();
        f.read(self.ssrc0, w);
        f.read(self.ssrc1, w);
        f.write_file(RegFile::Scc, 0, 1);
        f
    }
}
impl fmt::Display for Sopc {
    /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = self.op.width();
        write!(
            f,
            "{} {}, {}",
            self.op.name(),
            operand_text(self.ssrc0, w),
            operand_text(self.ssrc1, w)
        )
    }
}

/// llvm's `sendmsg(...)` text for an S_SENDMSG SIMM16 (gfx9 layout: message
/// id [3:0], operation [6:4], GS stream [9:8]; CDNA3 ISA §12.5 p.134 gives
/// only "SIMM16[9:0] contains the message type", the names are llvm's).
/// Named when the id, operation and stream are a valid combination; the
/// numeric triple when the SIMM16 is exactly that triple; else the decimal
/// SIMM16. The second value is false when the named form drops SIMM16 bits
/// (and so does not reassemble to the same word).
fn sendmsg_text(simm16: u16) -> (String, bool) {
    let (id, op, stream) = (simm16 & 0xf, (simm16 >> 4) & 0x7, (simm16 >> 8) & 0x3);
    let exact = id | op << 4 | stream << 8 == simm16;
    let name = match id {
        1 => Some("MSG_INTERRUPT"),
        2 => Some("MSG_GS"),
        3 => Some("MSG_GS_DONE"),
        4 => Some("MSG_SAVEWAVE"),
        5 => Some("MSG_STALL_WAVE_GEN"),
        6 => Some("MSG_HALT_WAVES"),
        7 => Some("MSG_ORDERED_PS_DONE"),
        8 => Some("MSG_EARLY_PRIM_DEALLOC"),
        9 => Some("MSG_GS_ALLOC_REQ"),
        10 => Some("MSG_GET_DOORBELL"),
        15 => Some("MSG_SYSMSG"),
        _ => None,
    };
    let gs_op = ["GS_OP_NOP", "GS_OP_CUT", "GS_OP_EMIT", "GS_OP_EMIT_CUT"];
    let named = match (id, name) {
        // MSG_GS needs CUT/EMIT/EMIT_CUT plus a stream; MSG_GS_DONE also takes NOP (stream 0, unprinted).
        (2 | 3, Some(n)) if (1..=3).contains(&op) => Some(format!(
            "sendmsg({n}, {}, {stream})",
            gs_op[usize::from(op)]
        )),
        (3, Some(n)) if op == 0 && stream == 0 => Some(format!("sendmsg({n}, {})", gs_op[0])),
        (15, Some(n)) if stream == 0 => match op {
            1 => Some(format!("sendmsg({n}, SYSMSG_OP_ECC_ERR_INTERRUPT)")),
            2 => Some(format!("sendmsg({n}, SYSMSG_OP_REG_RD)")),
            4 => Some(format!("sendmsg({n}, SYSMSG_OP_TTRACE_PC)")),
            _ => None,
        },
        (2 | 3 | 15, _) => None,
        (_, Some(n)) if op == 0 && stream == 0 => Some(format!("sendmsg({n})")),
        _ => None,
    };
    match named {
        Some(text) => (text, exact),
        None if exact => (format!("sendmsg({id}, {op}, {stream})"), true),
        None => (simm16.to_string(), true),
    }
}

/// llvm's `s_waitcnt` operand text: each counter below its maximum, or all
/// three when every counter is at its maximum.
fn waitcnt_text(arch: Arch, simm16: u16) -> String {
    let caps = arch.caps();
    let (vm, exp, lgkm) = waitcnt_fields(arch, simm16);
    let all = vm == caps.vmcnt_max && exp == caps.expcnt_max && lgkm == caps.lgkmcnt_max;
    let mut parts = Vec::with_capacity(3);
    if all || vm != caps.vmcnt_max {
        parts.push(format!("vmcnt({vm})"))
    }
    if all || exp != caps.expcnt_max {
        parts.push(format!("expcnt({exp})"))
    }
    if all || lgkm != caps.lgkmcnt_max {
        parts.push(format!("lgkmcnt({lgkm})"))
    }
    parts.join(" ")
}

impl Sopp {
    /// SIMM16 values with an exact text form: `s_barrier`/`s_icache_inv`
    /// take none (llvm rejects a nonzero SIMM16 as an invalid encoding),
    /// `s_waitcnt` must leave its unused bits clear, `s_sendmsg` must be
    /// printable without dropping bits.
    fn check(&self) -> Result<()> {
        let name = self.op.name();
        match self.op {
            SoppOp::Barrier | SoppOp::IcacheInv if self.simm16 != 0 => {
                Err(format!("{name} takes no SIMM16 (got {:#x})", self.simm16))
            }
            SoppOp::Waitcnt if self.simm16 & WAITCNT_UNUSED != 0 => Err(format!(
                "s_waitcnt SIMM16 {:#06x} sets unused bits {WAITCNT_UNUSED:#06x}",
                self.simm16
            )),
            SoppOp::Sendmsg if !sendmsg_text(self.simm16).1 => Err(format!(
                "s_sendmsg SIMM16 {:#06x} sets bits outside the message id/op/stream fields",
                self.simm16
            )),
            _ => Ok(()),
        }
    }

    /// Append this instruction's dword to `out`.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        let op = self
            .op
            .opcode(arch)
            .ok_or_else(|| format!("{} does not exist on {arch}", self.op.name()))?;
        self.check()?;
        out.push(0b1_0111_1111 << 23 | u32::from(op) << 16 | u32::from(self.simm16));
        Ok(())
    }
    /// Decode from `words[0..]`; returns the dword count consumed (always 1).
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let word = first_word(words, Format::Sopp)?;
        let opcode = ((word >> 16) & 0x7f) as u8;
        let op = SoppOp::from_opcode(arch, opcode)
            .ok_or_else(|| format!("SOPP opcode {opcode} is not in the {arch} table"))?;
        let inst = Self {
            op,
            simm16: word as u16,
        };
        inst.check()?;
        Ok((inst, 1))
    }
    /// Registers read and written (CDNA3 ISA §12.5 p.130-135):
    /// `s_cbranch_scc*` read SCC, `s_cbranch_vcc*` VCC (through VCCZ),
    /// `s_cbranch_exec*` EXEC (through EXECZ); `s_sendmsg` reads M0 (§4.5
    /// Table 11 p.21: "SALU writes M0 → S_SENDMSG", 1 wait state); `s_trap`
    /// writes {TTMP1, TTMP0} (p.134-135). Wait counters, priority and the
    /// barrier are not register state.
    /// With no trap handler (STATUS.TRAP_EN = 0) `s_trap` is a NOP (§3.10
    /// p.14) and writes nothing; the TTMP def is then conservative.
    pub fn footprint(&self) -> Footprint {
        let mut f = Footprint::default();
        match self.op {
            SoppOp::CbranchScc0 | SoppOp::CbranchScc1 => f.read_file(RegFile::Scc, 0, 1),
            SoppOp::CbranchVccz | SoppOp::CbranchVccnz => f.read_file(RegFile::Vcc, 0, 2),
            SoppOp::CbranchExecz | SoppOp::CbranchExecnz => f.read_file(RegFile::Exec, 0, 2),
            SoppOp::Sendmsg => f.read_file(RegFile::M0, 0, 1),
            SoppOp::Trap => f.write_file(RegFile::Ttmp, 0, 2),
            SoppOp::Nop
            | SoppOp::Endpgm
            | SoppOp::Branch
            | SoppOp::Barrier
            | SoppOp::Waitcnt
            | SoppOp::Sleep
            | SoppOp::Setprio
            | SoppOp::IcacheInv => {}
        }
        f
    }
}
impl fmt::Display for Sopp {
    /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces). `Display`
    /// takes no arch: `s_waitcnt` uses the gfx942 counter maxima, which every
    /// CDNA variant shares.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.op.name();
        let imm = self.simm16;
        match self.op {
            SoppOp::Endpgm if imm == 0 => f.write_str(name),
            SoppOp::Barrier | SoppOp::IcacheInv => f.write_str(name),
            SoppOp::Waitcnt => write!(f, "{name} {}", waitcnt_text(Arch::Gfx942, imm)),
            SoppOp::Sendmsg => write!(f, "{name} {}", sendmsg_text(imm).0),
            // llvm prints these immediates as inline-constant-style values.
            SoppOp::Nop | SoppOp::Sleep | SoppOp::Setprio | SoppOp::Trap if imm > 64 => {
                write!(f, "{name} {imm:#x}")
            }
            // Endpgm, branches (unsigned SIMM16, not the signed offset), small immediates.
            _ => write!(f, "{name} {imm}"),
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOP2); proven by the
// llvm-mc round-trip tests.
impl Sop2Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::AddU32 => 0,
                Self::SubU32 => 1,
                Self::AddI32 => 2,
                Self::SubI32 => 3,
                Self::AddcU32 => 4,
                Self::SubbU32 => 5,
                Self::MinI32 => 6,
                Self::MinU32 => 7,
                Self::MaxI32 => 8,
                Self::MaxU32 => 9,
                Self::CselectB32 => 10,
                Self::CselectB64 => 11,
                Self::AndB32 => 12,
                Self::AndB64 => 13,
                Self::OrB32 => 14,
                Self::OrB64 => 15,
                Self::XorB32 => 16,
                Self::XorB64 => 17,
                Self::Andn2B32 => 18,
                Self::Andn2B64 => 19,
                Self::Orn2B32 => 20,
                Self::Orn2B64 => 21,
                Self::LshlB32 => 28,
                Self::LshlB64 => 29,
                Self::LshrB32 => 30,
                Self::LshrB64 => 31,
                Self::AshrI32 => 32,
                Self::AshrI64 => 33,
                Self::BfmB32 => 34,
                Self::MulI32 => 36,
                Self::BfeU32 => 37,
                Self::BfeI32 => 38,
                Self::MulHiU32 => 44,
                Self::MulHiI32 => 45,
                Self::Lshl1AddU32 => 46,
                Self::Lshl2AddU32 => 47,
                Self::Lshl3AddU32 => 48,
                Self::Lshl4AddU32 => 49,
                Self::PackLlB32B16 => 50,
                Self::PackLhB32B16 => 51,
                Self::PackHhB32B16 => 52,
                Self::AbsdiffI32 => 42,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::AddU32 => "s_add_u32",
            Self::SubU32 => "s_sub_u32",
            Self::AddI32 => "s_add_i32",
            Self::SubI32 => "s_sub_i32",
            Self::AddcU32 => "s_addc_u32",
            Self::SubbU32 => "s_subb_u32",
            Self::MinI32 => "s_min_i32",
            Self::MinU32 => "s_min_u32",
            Self::MaxI32 => "s_max_i32",
            Self::MaxU32 => "s_max_u32",
            Self::CselectB32 => "s_cselect_b32",
            Self::CselectB64 => "s_cselect_b64",
            Self::AndB32 => "s_and_b32",
            Self::AndB64 => "s_and_b64",
            Self::OrB32 => "s_or_b32",
            Self::OrB64 => "s_or_b64",
            Self::XorB32 => "s_xor_b32",
            Self::XorB64 => "s_xor_b64",
            Self::Andn2B32 => "s_andn2_b32",
            Self::Andn2B64 => "s_andn2_b64",
            Self::Orn2B32 => "s_orn2_b32",
            Self::Orn2B64 => "s_orn2_b64",
            Self::LshlB32 => "s_lshl_b32",
            Self::LshlB64 => "s_lshl_b64",
            Self::LshrB32 => "s_lshr_b32",
            Self::LshrB64 => "s_lshr_b64",
            Self::AshrI32 => "s_ashr_i32",
            Self::AshrI64 => "s_ashr_i64",
            Self::BfmB32 => "s_bfm_b32",
            Self::MulI32 => "s_mul_i32",
            Self::BfeU32 => "s_bfe_u32",
            Self::BfeI32 => "s_bfe_i32",
            Self::MulHiU32 => "s_mul_hi_u32",
            Self::MulHiI32 => "s_mul_hi_i32",
            Self::Lshl1AddU32 => "s_lshl1_add_u32",
            Self::Lshl2AddU32 => "s_lshl2_add_u32",
            Self::Lshl3AddU32 => "s_lshl3_add_u32",
            Self::Lshl4AddU32 => "s_lshl4_add_u32",
            Self::PackLlB32B16 => "s_pack_ll_b32_b16",
            Self::PackLhB32B16 => "s_pack_lh_b32_b16",
            Self::PackHhB32B16 => "s_pack_hh_b32_b16",
            Self::AbsdiffI32 => "s_absdiff_i32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOP1); proven by the
// llvm-mc round-trip tests.
impl Sop1Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MovB32 => 0,
                Self::MovB64 => 1,
                Self::CmovB32 => 2,
                Self::CmovB64 => 3,
                Self::NotB32 => 4,
                Self::NotB64 => 5,
                Self::BrevB32 => 8,
                Self::Bcnt1I32B32 => 12,
                Self::Bcnt1I32B64 => 13,
                Self::Ff1I32B32 => 16,
                Self::Ff1I32B64 => 17,
                Self::FlbitI32B32 => 18,
                Self::SextI32I8 => 22,
                Self::SextI32I16 => 23,
                Self::AndSaveexecB64 => 32,
                Self::OrSaveexecB64 => 33,
                Self::XorSaveexecB64 => 34,
                Self::Andn2SaveexecB64 => 35,
                Self::GetpcB64 => 28,
                Self::SetpcB64 => 29,
                Self::AbsI32 => 48,
                Self::Bitset0B32 => 24,
                Self::Bitset1B32 => 26,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MovB32 => "s_mov_b32",
            Self::MovB64 => "s_mov_b64",
            Self::CmovB32 => "s_cmov_b32",
            Self::CmovB64 => "s_cmov_b64",
            Self::NotB32 => "s_not_b32",
            Self::NotB64 => "s_not_b64",
            Self::BrevB32 => "s_brev_b32",
            Self::Bcnt1I32B32 => "s_bcnt1_i32_b32",
            Self::Bcnt1I32B64 => "s_bcnt1_i32_b64",
            Self::Ff1I32B32 => "s_ff1_i32_b32",
            Self::Ff1I32B64 => "s_ff1_i32_b64",
            Self::FlbitI32B32 => "s_flbit_i32_b32",
            Self::SextI32I8 => "s_sext_i32_i8",
            Self::SextI32I16 => "s_sext_i32_i16",
            Self::AndSaveexecB64 => "s_and_saveexec_b64",
            Self::OrSaveexecB64 => "s_or_saveexec_b64",
            Self::XorSaveexecB64 => "s_xor_saveexec_b64",
            Self::Andn2SaveexecB64 => "s_andn2_saveexec_b64",
            Self::GetpcB64 => "s_getpc_b64",
            Self::SetpcB64 => "s_setpc_b64",
            Self::AbsI32 => "s_abs_i32",
            Self::Bitset0B32 => "s_bitset0_b32",
            Self::Bitset1B32 => "s_bitset1_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPK); proven by the
// llvm-mc round-trip tests.
impl SopkOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MovkI32 => 0,
                Self::CmovkI32 => 1,
                Self::CmpkEqI32 => 2,
                Self::CmpkLgI32 => 3,
                Self::CmpkGtI32 => 4,
                Self::CmpkGeI32 => 5,
                Self::CmpkLtI32 => 6,
                Self::CmpkLeI32 => 7,
                Self::CmpkEqU32 => 8,
                Self::CmpkLgU32 => 9,
                Self::CmpkGtU32 => 10,
                Self::CmpkGeU32 => 11,
                Self::CmpkLtU32 => 12,
                Self::CmpkLeU32 => 13,
                Self::AddkI32 => 14,
                Self::MulkI32 => 15,
                Self::GetregB32 => 17,
                Self::SetregB32 => 18,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MovkI32 => "s_movk_i32",
            Self::CmovkI32 => "s_cmovk_i32",
            Self::CmpkEqI32 => "s_cmpk_eq_i32",
            Self::CmpkLgI32 => "s_cmpk_lg_i32",
            Self::CmpkGtI32 => "s_cmpk_gt_i32",
            Self::CmpkGeI32 => "s_cmpk_ge_i32",
            Self::CmpkLtI32 => "s_cmpk_lt_i32",
            Self::CmpkLeI32 => "s_cmpk_le_i32",
            Self::CmpkEqU32 => "s_cmpk_eq_u32",
            Self::CmpkLgU32 => "s_cmpk_lg_u32",
            Self::CmpkGtU32 => "s_cmpk_gt_u32",
            Self::CmpkGeU32 => "s_cmpk_ge_u32",
            Self::CmpkLtU32 => "s_cmpk_lt_u32",
            Self::CmpkLeU32 => "s_cmpk_le_u32",
            Self::AddkI32 => "s_addk_i32",
            Self::MulkI32 => "s_mulk_i32",
            Self::GetregB32 => "s_getreg_b32",
            Self::SetregB32 => "s_setreg_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPC); proven by the
// llvm-mc round-trip tests.
impl SopcOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CmpEqI32 => 0,
                Self::CmpLgI32 => 1,
                Self::CmpGtI32 => 2,
                Self::CmpGeI32 => 3,
                Self::CmpLtI32 => 4,
                Self::CmpLeI32 => 5,
                Self::CmpEqU32 => 6,
                Self::CmpLgU32 => 7,
                Self::CmpGtU32 => 8,
                Self::CmpGeU32 => 9,
                Self::CmpLtU32 => 10,
                Self::CmpLeU32 => 11,
                Self::Bitcmp0B32 => 12,
                Self::Bitcmp1B32 => 13,
                Self::CmpEqU64 => 18,
                Self::CmpLgU64 => 19,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CmpEqI32 => "s_cmp_eq_i32",
            Self::CmpLgI32 => "s_cmp_lg_i32",
            Self::CmpGtI32 => "s_cmp_gt_i32",
            Self::CmpGeI32 => "s_cmp_ge_i32",
            Self::CmpLtI32 => "s_cmp_lt_i32",
            Self::CmpLeI32 => "s_cmp_le_i32",
            Self::CmpEqU32 => "s_cmp_eq_u32",
            Self::CmpLgU32 => "s_cmp_lg_u32",
            Self::CmpGtU32 => "s_cmp_gt_u32",
            Self::CmpGeU32 => "s_cmp_ge_u32",
            Self::CmpLtU32 => "s_cmp_lt_u32",
            Self::CmpLeU32 => "s_cmp_le_u32",
            Self::Bitcmp0B32 => "s_bitcmp0_b32",
            Self::Bitcmp1B32 => "s_bitcmp1_b32",
            Self::CmpEqU64 => "s_cmp_eq_u64",
            Self::CmpLgU64 => "s_cmp_lg_u64",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPP); proven by the
// llvm-mc round-trip tests.
impl SoppOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::Nop => 0,
                Self::Endpgm => 1,
                Self::Branch => 2,
                Self::CbranchScc0 => 4,
                Self::CbranchScc1 => 5,
                Self::CbranchVccz => 6,
                Self::CbranchVccnz => 7,
                Self::CbranchExecz => 8,
                Self::CbranchExecnz => 9,
                Self::Barrier => 10,
                Self::Waitcnt => 12,
                Self::Sleep => 14,
                Self::Setprio => 15,
                Self::Sendmsg => 16,
                Self::Trap => 18,
                Self::IcacheInv => 19,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::Nop => "s_nop",
            Self::Endpgm => "s_endpgm",
            Self::Branch => "s_branch",
            Self::CbranchScc0 => "s_cbranch_scc0",
            Self::CbranchScc1 => "s_cbranch_scc1",
            Self::CbranchVccz => "s_cbranch_vccz",
            Self::CbranchVccnz => "s_cbranch_vccnz",
            Self::CbranchExecz => "s_cbranch_execz",
            Self::CbranchExecnz => "s_cbranch_execnz",
            Self::Barrier => "s_barrier",
            Self::Waitcnt => "s_waitcnt",
            Self::Sleep => "s_sleep",
            Self::Setprio => "s_setprio",
            Self::Sendmsg => "s_sendmsg",
            Self::Trap => "s_trap",
            Self::IcacheInv => "s_icache_inv",
        }
    }
}
