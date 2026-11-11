// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! [`XmlInst`]: an instruction decoded from the machine-readable AMD ISA
//! (`amdgpu_isa_cdna3.xml` for gfx942, `amdgpu_isa_cdna4.xml` for gfx950)
//! through the generated tables of [`crate::xml`], for every instruction the
//! typed model ([`crate::enc`]) has no entry for.
//!
//! The typed instructions keep priority: [`crate::Inst::decode`] decodes
//! through them first and only an opcode the typed table does not know falls
//! back here. What the typed model does not describe is derived from the
//! XML row and nothing else:
//!
//! - **Footprint**: every XML operand that has a field is resolved from the
//!   decoded field value (register class, register count from the operand
//!   size, ACC selection from the encoding's ACC fields, `off`/`SVE`/`SADDR`
//!   rules of the memory encodings); every implicit operand the row lists
//!   (VCC, EXEC, M0, SCC, FLAT_SCRATCH) is a use or def per its Input/Output
//!   flags. EXEC is read by every lane-masked instruction (VALU, VMEM, DS
//!   except `ds_nop`) and M0 by LDS-touching DS operations and `lds_direct`
//!   operands, exactly as the typed footprints do. An operand the model
//!   cannot resolve makes the decode fail ([`XmlInst::decode`]): the facts
//!   are computed once, at decode, and never defaulted.
//! - **Memory class**: from the memory operands of the row (`DSMEM` /
//!   `GPUMEM` input/output), the ATOMIC functional subgroup and the return
//!   bit (`SC0` / `GLC`).
//! - **Branch offset**: from the row's `IsBranch` flag and its `LABEL`
//!   operand.
//!
//! The emulator never executes an [`XmlInst`] (no hand-written semantics) and
//! refuses it by name ([`XmlInst::unsupported`]); a row the LLVM cross-check
//! could not confirm is *unverified* ([`XmlInst::verification`]): it decodes
//! but [`XmlInst::encode`] refuses it. A [`FieldReceipt`] promotes only the
//! instances inside one hardware-proven field combination of such a row
//! ([`XmlInst::verification_of`]); the row itself stays unverified. gfx950 is
//! decode-only.
use crate::arch::Arch;
use crate::inst::{Footprint, Format, MemKind, RegFile};
use crate::operand::Src;
use crate::xml::{self, Decoded, Form, Isa, Operand, OperandKind as K, Verification};
use crate::Result;
use std::fmt;

/// One decoded generated instruction with the facts derived at decode.
#[derive(Clone, Debug, PartialEq)]
pub struct XmlInst {
    arch: Arch,
    decoded: Decoded,
    format: Format,
    footprint: Footprint,
    memory: Option<MemKind>,
    branch: Option<i16>,
    text: String,
}

pub(crate) fn isa(arch: Arch) -> Isa {
    match arch {
        Arch::Gfx942 => Isa::Cdna3,
        Arch::Gfx950 => Isa::Cdna4,
    }
}

/// A hardware receipt that confirms one field combination of an XML row the
/// LLVM cross-check left unverified. It never confirms the row: an instance
/// is covered only when every field of its encoding, except the `free`
/// register-number fields, equals the value in `fixed` (a field named in
/// neither is not covered).
#[derive(Debug)]
pub struct FieldReceipt {
    pub isa: Isa,
    /// XML `InstructionName` of the row.
    pub name: &'static str,
    /// XML encoding name of the row.
    pub encoding: &'static str,
    /// Proven value of every non-register field.
    pub fixed: &'static [(&'static str, u64)],
    /// Register-number fields that may take any value.
    pub free: &'static [&'static str],
    /// The committed twin receipt carrying the `gpu_oracle` block.
    pub twin_receipt: &'static str,
    /// The hardware oracle receipt (`moe-twin-oracle.json`, on `mi300x`:
    /// `/scratch/quant/fn-fp8/beat/moe-twin-oracle.json`) and its sha256.
    pub oracle_receipt: &'static str,
    pub oracle_sha256: &'static str,
    /// The anchor (hipcc image) and the twin `.hxaco` it was compared with.
    pub anchor_sha256: &'static str,
    pub twin_sha256: &'static str,
    /// What the receipt proved, and where the combination occurs.
    pub evidence: &'static str,
}

impl FieldReceipt {
    /// Whether `d` is exactly inside the proven combination.
    pub fn covers(&self, d: &Decoded) -> bool {
        d.form.isa == self.isa
            && d.form.name == self.name
            && d.form.encoding == self.encoding
            && d.form.fields.iter().all(|fl| {
                self.free.contains(&fl.name)
                    || self
                        .fixed
                        .iter()
                        .find(|(n, _)| *n == fl.name)
                        .is_some_and(|&(n, v)| d.field(n) == Some(v))
            })
    }
}

/// Every hardware receipt that promotes an instance of an unverified row.
pub static FIELD_RECEIPTS: &[&FieldReceipt] = &[&V_CVT_PK_F32_FP8_SDWA_WORD1];

/// `v_cvt_pk_f32_fp8_sdwa vdst[2], vsrc0 src0_sel:WORD_1` (dst_sel DWORD,
/// dst_unused PAD, no clamp/omod/sext/neg/abs, src1 fields zero), the 32 SDWA
/// instructions (words `0x7e??acf9, 0x000506??`) of the gfx942 FP8 grouped MoE
/// anchor `gemm_fp8b128_moe_grouped_bf16mfma`. The twin `.hxaco` carries the
/// anchor's device ELF byte for byte (sha256 8359a935...), and the oracle
/// executed both through the full `test_fp8_moe_gfx942` matrix.
pub static V_CVT_PK_F32_FP8_SDWA_WORD1: FieldReceipt = FieldReceipt {
    isa: Isa::Cdna3,
    name: "V_CVT_PK_F32_FP8",
    encoding: "VOP1_VOP_SDWA",
    fixed: &[
        ("ENCODING", 0x3f),
        ("OP", 86),
        ("SRC0", 0xf9),
        ("S0", 0),
        ("S1", 0),
        ("DST_SEL", 6),
        ("DST_UNUSED", 0),
        ("CLAMP", 0),
        ("OMOD", 0),
        ("SRC0_SEL", 5),
        ("SRC0_SEXT", 0),
        ("SRC0_NEG", 0),
        ("SRC0_ABS", 0),
        ("SRC1_SEL", 0),
        ("SRC1_SEXT", 0),
        ("SRC1_NEG", 0),
        ("SRC1_ABS", 0),
    ],
    free: &["VDST", "VSRC0"],
    twin_receipt: "kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.twin.json#gpu_oracle",
    oracle_receipt: "mi300x/beat/Twin/moe-twin-oracle.json",
    oracle_sha256: "2b6842e984b63540443e9f11efdf15744c8f90cbf69c2fad0879f06a2878456c",
    anchor_sha256: "08ebe4036da4c71b2c5d685af77a6a3692b975a2a28839c3683200dca2de10c0",
    twin_sha256: "fc16c27637517e54f078c25f3f9166d2663df10a840badda4f83ecc69008ea44",
    evidence: "gfx942 device 0: 109/109 cases PASS, 12828 launches each of \
gemm_fp8b128_moe_grouped_bf16mfma and _xbf16 bitwise equal anchor vs twin \
(device ELF sha256 8359a935d4dc453a76f4856d8dbba5d2230e4957c3a966de982c6308b549ea65 \
identical); 32 SDWA instances in the device ELF, all WORD_1",
};

impl XmlInst {
    /// Decode the instruction at `words[0]` for `arch` from the generated
    /// tables; returns it and its dword count. Fails when the encoding is not
    /// in the table, a reserved bit pattern is set, or a field value names an
    /// operand the footprint model cannot resolve.
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> {
        let (decoded, len) = xml::decode(isa(arch), words)?;
        let form = decoded.form;
        let format = format_of(form.encoding)
            .ok_or_else(|| format!("{}: encoding {} has no Format", form.name, form.encoding))?;
        let cx = Cx { d: &decoded, form };
        let facts = cx
            .facts()
            .map_err(|e| format!("{} ({}): {e}", form.mnemonic(), form.encoding))?;
        Ok((
            Self {
                arch,
                decoded,
                format,
                footprint: facts.footprint,
                memory: facts.memory,
                branch: facts.branch,
                text: facts.text,
            },
            len,
        ))
    }

    /// Append the instruction's dwords to `out`. Refused for gfx950
    /// (decode-only), for an `arch` other than the one it was decoded for, and
    /// for a row the LLVM cross-check marked unverified.
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> {
        if !arch.encodes() {
            return Err(format!(
                "{} is decode-only: {} cannot be encoded",
                arch.name(),
                self.name()
            ));
        }
        if arch != self.arch {
            return Err(format!(
                "{} was decoded for {} and cannot be encoded for {}",
                self.name(),
                self.arch.name(),
                arch.name()
            ));
        }
        self.verification()
            .map_err(|why| format!("unverified encoding: {}: {why}", self.name()))?;
        // `xml::encode` refuses every row the LLVM cross-check left
        // unverified; the per-instance gate above is the finer one.
        let (words, n) = self.decoded.assemble()?;
        out.extend_from_slice(&words[..n]);
        Ok(())
    }

    /// The decoded row and field values.
    pub fn decoded(&self) -> &Decoded {
        &self.decoded
    }
    pub fn form(&self) -> &'static Form {
        self.decoded.form
    }
    /// Lowercase llvm mnemonic of the XML instruction name (no `_e32`/`_dpp`
    /// suffix), e.g. `ds_add_u32`.
    pub fn name(&self) -> &'static str {
        self.decoded.form.mnemonic()
    }
    /// XML encoding name, e.g. `ENC_DS`, `VOP2_VOP_SDWA`.
    pub fn encoding(&self) -> &'static str {
        self.decoded.form.encoding
    }
    pub fn opcode(&self) -> u32 {
        self.decoded.form.opcode
    }
    pub fn arch(&self) -> Arch {
        self.arch
    }
    pub fn format(&self) -> Format {
        self.format
    }
    /// Raw field value by XML field name.
    pub fn field(&self, name: &str) -> Option<u64> {
        self.decoded.field(name)
    }
    /// `Ok` for an instruction the LLVM cross-check or a field-specific
    /// hardware receipt confirmed; `Err(reason)` for an unverified one
    /// (recognized by decode, refused by encode and by seals). Per instance:
    /// see [`XmlInst::verification_of`].
    pub fn verification(&self) -> std::result::Result<(), &'static str> {
        Self::verification_of(&self.decoded)
    }
    pub fn verified(&self) -> bool {
        self.verification().is_ok()
    }
    /// Per-instance verification of a decoded XML value (also the one the
    /// words of a typed instruction decode to through [`xml::decode`]). A row
    /// the LLVM cross-check confirmed is `Ok` for every instance. An
    /// unverified row is `Ok` only for an instance whose every field outside
    /// a [`FieldReceipt`]'s free (register-number) fields equals the
    /// combination that receipt proved on hardware; every other instance
    /// keeps the row's refusal.
    pub fn verification_of(d: &Decoded) -> std::result::Result<(), &'static str> {
        match d.form.verification {
            Verification::Verified => Ok(()),
            Verification::Unverified(why) => match Self::receipt_of(d) {
                Some(_) => Ok(()),
                None => Err(why),
            },
        }
    }
    /// The hardware receipt that promotes this instruction, if one does.
    pub fn receipt(&self) -> Option<&'static FieldReceipt> {
        Self::receipt_of(&self.decoded)
    }
    /// [`XmlInst::receipt`] for a decoded value: `Some` only when `d`'s row is
    /// unverified and `d` is inside a receipt's proven field combination.
    pub fn receipt_of(d: &Decoded) -> Option<&'static FieldReceipt> {
        if d.form.verification.is_verified() {
            return None;
        }
        FIELD_RECEIPTS.iter().copied().find(|r| r.covers(d))
    }
    /// Registers read and written, explicit and implicit, from the XML row.
    pub fn footprint(&self) -> Footprint {
        self.footprint.clone()
    }
    /// The wait-counter class of a memory instruction; `None` for the rest.
    pub fn memory(&self) -> Option<MemKind> {
        self.memory
    }
    /// Signed dword offset of a relative branch (`SIMM16` of a row flagged
    /// `IsBranch` with a `LABEL` operand).
    pub fn branch_offset(&self) -> Option<i16> {
        self.branch
    }
    /// The refusal every consumer without a hand-written model gives, by name.
    pub fn unsupported(&self) -> String {
        let mut text = format!(
            "generated instruction {} ({} opcode {}, {}) has no hand-written model",
            self.name(),
            self.encoding(),
            self.opcode(),
            self.arch.name()
        );
        if let Err(why) = self.verification() {
            text.push_str(&format!("; unverified encoding: {why}"));
        }
        text
    }
}

impl fmt::Display for XmlInst {
    /// `mnemonic operand, operand, ... modifier ...` in llvm spelling where
    /// the field has one and `field:value` otherwise (diagnostic text: the
    /// typed text assembler does not parse generated instructions).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// XML encoding name to the microcode [`Format`] it belongs to.
fn format_of(encoding: &str) -> Option<Format> {
    Some(match encoding {
        "ENC_SOP1" | "SOP1_INST_LITERAL" => Format::Sop1,
        "ENC_SOP2" | "SOP2_INST_LITERAL" => Format::Sop2,
        "ENC_SOPK" | "SOPK_INST_LITERAL" => Format::Sopk,
        "ENC_SOPC" | "SOPC_INST_LITERAL" => Format::Sopc,
        "ENC_SOPP" => Format::Sopp,
        "ENC_SMEM" => Format::Smem,
        "ENC_VOP1" | "VOP1_INST_LITERAL" | "VOP1_VOP_DPP" | "VOP1_VOP_SDWA" => Format::Vop1,
        "ENC_VOP2"
        | "VOP2_INST_LITERAL"
        | "VOP2_VOP_DPP"
        | "VOP2_VOP_SDWA"
        | "VOP2_VOP_SDWA_SDST_ENC" => Format::Vop2,
        "ENC_VOPC" | "VOPC_INST_LITERAL" | "VOPC_VOP_SDWA_SDST_ENC" => Format::Vopc,
        "ENC_VOP3" | "VOP3_SDST_ENC" => Format::Vop3,
        "ENC_VOP3P" | "ENC_VOP3PX2" => Format::Vop3p,
        "VOP3P_MFMA" => Format::Mai,
        "ENC_DS" => Format::Ds,
        "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => Format::Flat,
        "ENC_MUBUF" => Format::Mubuf,
        "ENC_MTBUF" => Format::Mtbuf,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Operand resolution.

/// An operand's resolved value.
enum Val {
    /// Registers (or constants): `(first register / constant, dwords)`.
    Srcs(Vec<(Src, u8)>),
    /// An immediate field value.
    Imm(u64),
    /// SMEM offset: SGPR offsets and the signed immediate, if any.
    SmemOffset(Vec<(Src, u8)>, Option<i32>),
    /// `lds_direct` (source code 254): an LDS read through M0.
    LdsDirect,
    /// A literal dword.
    Literal(u32),
    /// `off`: the operand is absent in this encoding.
    Off,
    /// Memory / program counter: no register.
    None,
}

struct Facts {
    footprint: Footprint,
    memory: Option<MemKind>,
    branch: Option<i16>,
    text: String,
}

struct Cx<'a> {
    d: &'a Decoded,
    form: &'static Form,
}

fn dwords(size_bits: u16) -> u8 {
    (size_bits.div_ceil(32)).clamp(1, 255) as u8
}

impl Cx<'_> {
    fn field(&self, name: &str) -> Option<u64> {
        self.d.field(name)
    }
    fn flag(&self, name: &str) -> bool {
        self.field(name).is_some_and(|v| v != 0)
    }
    /// Total bits of a field of this encoding.
    fn width(&self, name: &str) -> u32 {
        self.form
            .fields
            .iter()
            .find(|f| f.name == name)
            .map_or(0, |f| f.ranges.iter().map(|r| u32::from(r.1)).sum())
    }
    fn is(&self, encoding: &str) -> bool {
        self.form.encoding == encoding
    }
    fn is_flat(&self) -> bool {
        matches!(
            self.form.encoding,
            "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH"
        )
    }
    fn is_mfma(&self) -> bool {
        matches!(self.form.encoding, "VOP3P_MFMA" | "ENC_VOP3PX2")
    }

    /// Whether operand field `name` names an AccVGPR: the encoding's ACC bit
    /// (DS, FLAT, MUBUF, MTBUF: VDST/DATA*/VDATA), or for MFMA ACC bit 0 =
    /// srcA, bit 1 = srcB and ACC_CD = srcC and vdst.
    fn acc(&self, name: &str) -> bool {
        if self.is_mfma() {
            let acc = self.field("ACC").unwrap_or(0);
            match name {
                "SRC0" => acc & 1 != 0,
                "SRC1" => acc & 2 != 0,
                _ => self.flag("ACC_CD"),
            }
        } else {
            self.field("ACC").is_some_and(|v| v & 1 != 0)
        }
    }

    fn src_val(&self, src: Src, n: u8) -> Val {
        Val::Srcs(vec![(src, n)])
    }

    fn vgpr(&self, code: u64, n: u8, acc: bool) -> Result<Val> {
        let r = u8::try_from(code).map_err(|_| format!("VGPR field value {code} exceeds 255"))?;
        Ok(self.src_val(if acc { Src::Agpr(r) } else { Src::Vgpr(r) }, n))
    }

    /// Resolve a source-class code (9-bit source space; scalar codes below
    /// 256, VGPRs from 256).
    fn source(&self, kind: K, code: u64, n: u8, acc: bool) -> Result<Val> {
        let lds_ok = matches!(kind, K::Src | K::SrcSimple | K::VgprOrLds);
        if code == 254 && lds_ok {
            return Ok(Val::LdsDirect);
        }
        let code16 =
            u16::try_from(code).map_err(|_| format!("operand code {code} out of range"))?;
        let mut src = Src::from_code(code16, self.d.literal())?;
        if acc {
            if let Src::Vgpr(r) = src {
                src = Src::Agpr(r);
            }
        }
        Ok(self.src_val(src, n))
    }

    fn operand_value(&self, op: &Operand) -> Result<Val> {
        let n = dwords(op.size_bits);
        let Some(name) = op.field else {
            return self.implicit(op);
        };
        if name == "LITERAL" {
            let lit = self
                .d
                .literal()
                .ok_or("literal operand without a literal dword")?;
            return Ok(Val::Literal(lit));
        }
        let code = self
            .field(name)
            .ok_or_else(|| format!("operand field {name} is absent from the decoded row"))?;
        match op.kind {
            K::Simm16
            | K::Simm32
            | K::Simm4
            | K::Simm8
            | K::Label
            | K::Hwreg
            | K::Sendmsg
            | K::Waitcnt => Ok(Val::Imm(code)),
            K::SmemOffset => self.smem_offset(),
            K::Vgpr => {
                if self.is("ENC_MUBUF") || self.is("ENC_MTBUF") {
                    if name == "VADDR" {
                        return match (self.flag("OFFEN"), self.flag("IDXEN")) {
                            (false, false) => Ok(Val::Off),
                            (true, true) => self.vgpr(code, 2, false),
                            _ => self.vgpr(code, 1, false),
                        };
                    }
                } else if self.is_flat() && name == "ADDR" {
                    if self.is("ENC_FLAT_SCRATCH") {
                        return if self.flag("SVE") {
                            self.vgpr(code, 1, false)
                        } else {
                            Ok(Val::Off)
                        };
                    }
                    if self.is("ENC_FLAT_GLBL") && self.field("SADDR") != Some(0x7f) {
                        return self.vgpr(code, 1, false);
                    }
                }
                self.vgpr(code, n, false)
            }
            K::VgprOrAccvgpr => self.vgpr(code, n, self.acc(name)),
            K::Accvgpr => self.vgpr(code, n, true),
            K::SrcAccvgpr => {
                if code < 256 {
                    return Err(format!("{name}={code} is not an AccVGPR source"));
                }
                self.vgpr(code - 256, n, true)
            }
            K::Sreg | K::SregNovcc | K::Sdst => {
                if self.is_flat() && name == "SADDR" && code == 0x7f {
                    return Ok(Val::Off);
                }
                let scale = match self.width(name) {
                    5 => 4,
                    6 => 2,
                    _ => 1,
                };
                let code = code * scale;
                if code > 127 {
                    return Err(format!("{name}={code} is not a scalar register code"));
                }
                self.source(op.kind, code, n, false)
            }
            K::SrcVgprOrAccvgpr | K::SrcVgprOrAccvgprOrConst => {
                self.source(op.kind, code, n, self.acc(name))
            }
            K::Vcc
            | K::SdstExec
            | K::SdstM0
            | K::SsrcSpecialScc
            | K::FlatScratch
            | K::Pc
            | K::Dsmem
            | K::Gpumem => Err(format!(
                "operand {name} of kind {:?} carries a field but is an implicit register",
                op.kind
            )),
            K::Src
            | K::SrcSimple
            | K::SrcNolit
            | K::SrcNolds
            | K::Ssrc
            | K::SsrcNolit
            | K::SsrcLanesel
            | K::SrcVgpr
            | K::VgprOrLds => {
                // SDWA / DPP: VSRC0 / VSRC1 are 8-bit; S0 / S1 pick scalar
                // (code below 256) or VGPR.
                let code = match name {
                    "VSRC0" if !self.flag("S0") => 256 + code,
                    "VSRC1" if !self.flag("S1") => 256 + code,
                    _ => code,
                };
                self.source(op.kind, code, n, false)
            }
        }
    }

    /// SMEM offset: IMM=1 is a signed 21-bit immediate (plus the SOFFSET
    /// SGPR when SOFFSET_EN); IMM=0 names the offset SGPR in OFFSET[6:0].
    fn smem_offset(&self) -> Result<Val> {
        let imm = self.flag("IMM");
        let off = self.field("OFFSET").unwrap_or(0);
        let mut regs = Vec::new();
        if !imm {
            regs.push((Src::from_code((off & 0x7f) as u16, None)?, 1));
        }
        if self.flag("SOFFSET_EN") {
            let code = self.field("SOFFSET").unwrap_or(0);
            regs.push((Src::from_code(code as u16, None)?, 1));
        }
        let simm = imm.then(|| ((off as u32) << 11) as i32 >> 11);
        Ok(Val::SmemOffset(regs, simm))
    }

    /// An operand with no field: a state register of the row, or memory.
    fn implicit(&self, op: &Operand) -> Result<Val> {
        Ok(match op.kind {
            K::Vcc => self.src_val(Src::VccLo, 2),
            K::SdstExec => self.src_val(Src::ExecLo, 2),
            K::SdstM0 => self.src_val(Src::M0, 1),
            K::SsrcSpecialScc => self.src_val(Src::Scc, 1),
            K::FlatScratch => self.src_val(Src::FlatScratchLo, 2),
            K::Pc | K::Dsmem | K::Gpumem => Val::None,
            other => {
                return Err(format!(
                    "implicit operand of kind {other:?} has no register model"
                ))
            }
        })
    }

    // -----------------------------------------------------------------------
    // Facts.

    fn operand_has(&self, kind: K, input: bool, output: bool) -> bool {
        self.form
            .operands
            .iter()
            .any(|o| o.kind == kind && (!input || o.input) && (!output || o.output))
    }

    fn memory_kind(&self, lds_direct: bool) -> Option<MemKind> {
        let ret_vdata = self.form.operands.iter().any(|o| {
            o.output
                && o.field
                    .is_some_and(|f| matches!(f, "VDST" | "VDATA" | "SDATA"))
        });
        let atomic = self.form.subgroups.contains(&"ATOMIC");
        match self.form.encoding {
            "ENC_DS" => {
                let (i, o) = (
                    self.operand_has(K::Dsmem, true, false),
                    self.operand_has(K::Dsmem, false, true),
                );
                Some(match (i, o) {
                    (true, true) if ret_vdata => MemKind::DsAtomicRtn,
                    (true, true) => MemKind::DsAtomic,
                    (true, false) => MemKind::DsLoad,
                    (false, true) => MemKind::DsStore,
                    (false, false) if ret_vdata => MemKind::DsXlane,
                    (false, false) => MemKind::DsStore,
                })
            }
            "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" | "ENC_MUBUF" | "ENC_MTBUF" => {
                let (i, o) = (
                    self.operand_has(K::Gpumem, true, false),
                    self.operand_has(K::Gpumem, false, true),
                );
                Some(match (i, o) {
                    (true, true) if self.flag("SC0") => MemKind::VmemAtomicRtn,
                    (true, true) => MemKind::VmemAtomic,
                    (true, false) => MemKind::VmemLoad,
                    (false, true) => MemKind::VmemStore,
                    (false, false) if ret_vdata => MemKind::VmemLoad,
                    (false, false) => MemKind::VmemStore,
                })
            }
            "ENC_SMEM" => {
                let (i, o) = (
                    self.operand_has(K::Gpumem, true, false),
                    self.operand_has(K::Gpumem, false, true),
                );
                Some(match (i, o) {
                    _ if atomic && self.flag("GLC") => MemKind::SmemAtomicRtn,
                    _ if atomic => MemKind::SmemAtomic,
                    (false, true) => MemKind::SmemStore,
                    _ => MemKind::SmemLoad,
                })
            }
            _ if lds_direct => Some(MemKind::DsLoad),
            _ => None,
        }
    }

    fn facts(&self) -> Result<Facts> {
        let form = self.form;
        let mut fp = Footprint::default();
        let mut lds_direct = false;
        let mut vals = Vec::with_capacity(form.operands.len());
        let atomic_vmem = !self.is("ENC_SMEM")
            && self.operand_has(K::Gpumem, true, false)
            && self.operand_has(K::Gpumem, false, true);
        let smem_atomic = self.is("ENC_SMEM") && form.subgroups.contains(&"ATOMIC");
        let preserve = self.field("DST_UNUSED") == Some(2);
        for op in form.operands {
            let val = self.operand_value(op)?;
            let (mut input, mut output) = (op.input, op.output);
            let is = |names: &[&str]| op.field.is_some_and(|f| names.contains(&f));
            if atomic_vmem && output && is(&["VDST", "VDATA"]) && !self.flag("SC0") {
                output = false;
            }
            if smem_atomic && is(&["SDATA"]) {
                input = true;
                output = self.flag("GLC");
            }
            if preserve && output && is(&["VDST"]) {
                input = true;
            }
            match &val {
                Val::Srcs(list) | Val::SmemOffset(list, _) => {
                    for &(src, n) in list {
                        if input {
                            fp.read(src, n);
                        }
                        if output {
                            fp.write(src, n);
                        }
                    }
                }
                Val::LdsDirect => lds_direct = true,
                _ => {}
            }
            vals.push(val);
        }
        let memory = self.memory_kind(lds_direct);
        let lane_masked = match form.group {
            "VALU" => form.name != "V_NOP",
            "VMEM" => form.name != "DS_NOP",
            _ => false,
        };
        if lane_masked {
            fp.read_file(RegFile::Exec, 0, 2);
        }
        let touches_lds = matches!(
            memory,
            Some(MemKind::DsLoad | MemKind::DsStore | MemKind::DsAtomic | MemKind::DsAtomicRtn)
        ) && form.name != "DS_NOP";
        if touches_lds || lds_direct {
            fp.read_file(RegFile::M0, 0, 1);
        }
        let branch = if form.is_branch {
            form.operands
                .iter()
                .find(|o| o.kind == K::Label)
                .and_then(|o| o.field)
                .and_then(|f| self.field(f))
                .map(|v| v as u16 as i16)
        } else {
            None
        };
        self.legal(&vals)?;
        let text = self.text(&vals)?;
        Ok(Facts {
            footprint: fp,
            memory,
            branch,
            text,
        })
    }

    // -----------------------------------------------------------------------
    // Operand legality: the refusals the typed encoders make, derived from the
    // operand rules rather than per-word.

    fn legal(&self, vals: &[Val]) -> Result<()> {
        crate::xml_legality::selectors(self.d)?;
        self.unused_operand_fields()?;
        for (op, val) in self.form.operands.iter().zip(vals) {
            if op.field.is_none() {
                continue;
            }
            if let Val::Srcs(list) | Val::SmemOffset(list, _) = val {
                for &(src, n) in list {
                    Self::check_register_span(src, n)
                        .map_err(|e| format!("{}: {e}", op.field.unwrap_or("operand")))?;
                }
            }
        }
        self.constant_bus(vals)
    }

    /// An operand field the form does not use must be zero in the scalar and
    /// VOP3 encodings (an unused SSRC1 / SRC2 carries no meaning, so a
    /// nonzero value is a different, unmodelled encoding).
    fn unused_operand_fields(&self) -> Result<()> {
        let names: &[&str] = match self.form.encoding {
            "ENC_SOP1" | "SOP1_INST_LITERAL" | "ENC_SOP2" | "SOP2_INST_LITERAL" | "ENC_SOPC"
            | "SOPC_INST_LITERAL" | "ENC_SOPK" | "SOPK_INST_LITERAL" => &["SDST", "SSRC0", "SSRC1"],
            "ENC_VOP3" | "VOP3_SDST_ENC" | "ENC_VOP3P" => &["SRC0", "SRC1", "SRC2"],
            "ENC_SOPP" => &["SIMM16"],
            _ => return Ok(()),
        };
        for f in self.form.fields {
            if !names.contains(&f.name)
                || self.form.operands.iter().any(|o| o.field == Some(f.name))
            {
                continue;
            }
            let marker = self.form.encoding.ends_with("_INST_LITERAL");
            if let Some(v) = self
                .field(f.name)
                .filter(|&v| v != 0 && !(marker && v == 255))
            {
                return Err(format!("unused field {}={v} must be zero", f.name));
            }
        }
        Ok(())
    }

    /// A multi-dword scalar register operand starts at a register aligned to
    /// min(dwords, 4) and lies inside its file; a 64-bit special-register
    /// operand must start at the low half.
    fn check_register_span(src: Src, n: u8) -> Result<()> {
        if n < 2 {
            return Ok(());
        }
        let align = u32::from(n.min(4));
        match src {
            Src::Sgpr(r) => {
                let r = u32::from(r);
                if r % align != 0 || r + u32::from(n) > 102 {
                    return Err(format!("s{r} is not a valid {n}-dword SGPR start"));
                }
            }
            Src::Ttmp(r) => {
                let r = u32::from(r);
                if r % align != 0 || r + u32::from(n) > 16 {
                    return Err(format!("ttmp{r} is not a valid {n}-dword ttmp start"));
                }
            }
            Src::FlatScratchHi | Src::XnackMaskHi | Src::VccHi | Src::ExecHi | Src::M0 => {
                return Err(format!("{src:?} cannot start a {n}-dword operand"));
            }
            _ => {}
        }
        Ok(())
    }

    /// CDNA VALU instructions read at most one scalar register, special
    /// register or literal through the constant bus (inline constants are
    /// free; one register read twice is one use).
    fn constant_bus(&self, vals: &[Val]) -> Result<()> {
        if self.form.group != "VALU" || self.is_mfma() {
            return Ok(());
        }
        // (kind, value): 0 = register code, 1 = literal source, 2 = instruction literal,
        // 3 = lds_direct.
        let mut uses: Vec<(u8, u32)> = Vec::new();
        for (op, val) in self.form.operands.iter().zip(vals) {
            if op.field.is_none() || !op.input {
                continue;
            }
            let key = match val {
                Val::Srcs(list) => match list.first() {
                    Some(&(src, _)) => match src {
                        Src::Vgpr(_) | Src::Agpr(_) | Src::Int(_) | Src::Float(_) => continue,
                        Src::Lit(v) => (1, v),
                        other => (0, u32::from(other.code())),
                    },
                    None => continue,
                },
                Val::Literal(v) => (if op.kind == K::Simm32 { 2 } else { 1 }, *v),
                Val::LdsDirect => (3, 0),
                _ => continue,
            };
            if !uses.contains(&key) {
                uses.push(key);
            }
        }
        if uses.len() > 1 {
            return Err(
                "more than one constant-bus operand (scalar register, special \
                register or literal)"
                    .into(),
            );
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Text.

    fn mnemonic(&self) -> String {
        let base = self.form.mnemonic();
        let suffix = match self.form.encoding {
            "ENC_VOP1" | "VOP1_INST_LITERAL" | "ENC_VOP2" | "VOP2_INST_LITERAL" | "ENC_VOPC"
            | "VOPC_INST_LITERAL" => "_e32",
            "VOP1_VOP_DPP" | "VOP2_VOP_DPP" => "_dpp",
            "VOP1_VOP_SDWA"
            | "VOP2_VOP_SDWA"
            | "VOP2_VOP_SDWA_SDST_ENC"
            | "VOPC_VOP_SDWA_SDST_ENC" => "_sdwa",
            "ENC_VOP3" | "VOP3_SDST_ENC" if self.form.opcode < 0x1c0 => "_e64",
            _ => "",
        };
        format!("{base}{suffix}")
    }

    fn operand_text(&self, op: &Operand, val: &Val) -> String {
        match val {
            Val::Srcs(list) => list
                .iter()
                .map(|&(s, n)| s.fmt(n))
                .collect::<Vec<_>>()
                .join(", "),
            Val::Literal(v) => format!("{v:#x}"),
            Val::Imm(code) => match op.kind {
                K::Label => format!("{}", *code as u16 as i16),
                K::Hwreg => format!(
                    "hwreg({}, {}, {})",
                    code & 0x3f,
                    (code >> 6) & 0x1f,
                    ((code >> 11) & 0x1f) + 1
                ),
                K::Waitcnt => {
                    let (vm, exp, lgkm) =
                        crate::enc::sop::waitcnt_fields(self.arch(), *code as u16);
                    format!("vmcnt({vm}) expcnt({exp}) lgkmcnt({lgkm})")
                }
                _ => format!("{code:#x}"),
            },
            Val::SmemOffset(regs, imm) => {
                let mut t = regs
                    .iter()
                    .map(|&(s, n)| s.fmt(n))
                    .collect::<Vec<_>>()
                    .join(", ");
                if let Some(i) = imm {
                    if t.is_empty() {
                        t = format!("{i:#x}");
                    } else {
                        t.push_str(&format!(" offset:{i:#x}"));
                    }
                }
                t
            }
            Val::LdsDirect => "lds_direct".to_string(),
            Val::Off => "off".to_string(),
            Val::None => String::new(),
        }
    }

    fn arch(&self) -> Arch {
        match self.d.isa() {
            Isa::Cdna3 => Arch::Gfx942,
            Isa::Cdna4 => Arch::Gfx950,
        }
    }

    fn text(&self, vals: &[Val]) -> Result<String> {
        let mut out = self.mnemonic();
        let mut sep = " ";
        let mut consumed: Vec<&str> = vec!["ENCODING", "X2ENCODING", "OP", "LITERAL"];
        for (op, val) in self.form.operands.iter().zip(vals) {
            if let Some(f) = op.field {
                consumed.push(f);
            }
            if op.implicit {
                continue;
            }
            if matches!(val, Val::None) {
                continue;
            }
            out.push_str(sep);
            out.push_str(&self.operand_text(op, val));
            sep = ", ";
        }
        if self.is("ENC_SMEM") {
            consumed.extend(["IMM", "SOFFSET_EN", "OFFSET", "SOFFSET", "NV"]);
        }
        if self.is_mfma() {
            consumed.extend(["ACC", "ACC_CD"]);
        } else {
            consumed.push("ACC");
        }
        consumed.extend(["SEG", "SVE", "S0", "S1", "SD"]);
        let two_addr = ["READ2", "WRITE2", "WRXCHG2"]
            .iter()
            .any(|p| self.form.name.contains(p));
        for field in self.form.fields {
            let name = field.name;
            if consumed.contains(&name) {
                continue;
            }
            let Some(v) = self.field(name) else { continue };
            if let Some(m) = self.modifier(name, v, two_addr) {
                out.push(' ');
                out.push_str(&m);
            }
        }
        Ok(out)
    }

    fn modifier(&self, name: &str, v: u64, two_addr: bool) -> Option<String> {
        const SEL: [&str; 7] = [
            "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD",
        ];
        const UNUSED: [&str; 3] = ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"];
        let lower = name.to_ascii_lowercase();
        Some(match name {
            "OFFSET" => {
                if v == 0 {
                    return None;
                }
                let w = self.width(name);
                let signed = self.is("ENC_FLAT_GLBL") || self.is("ENC_FLAT_SCRATCH");
                if signed && v >> (w - 1) & 1 != 0 {
                    format!("offset:{}", v as i64 - (1i64 << w))
                } else {
                    format!("offset:{v}")
                }
            }
            "OFFSET0" | "OFFSET1" if !two_addr => {
                if name == "OFFSET1" {
                    return None;
                }
                let full = v | self.field("OFFSET1").unwrap_or(0) << 8;
                if full == 0 {
                    return None;
                }
                format!("offset:{full}")
            }
            "OFFSET0" | "OFFSET1" => {
                if v == 0 {
                    return None;
                }
                format!("{lower}:{v}")
            }
            "OMOD" => match v {
                0 => return None,
                1 => "mul:2".to_string(),
                2 => "mul:4".to_string(),
                _ => "div:2".to_string(),
            },
            "DST_SEL" | "SRC0_SEL" | "SRC1_SEL" => {
                if v == 6 {
                    return None;
                }
                format!(
                    "{lower}:{}",
                    SEL.get(v as usize).copied().unwrap_or("RESERVED")
                )
            }
            "DST_UNUSED" => {
                if v == 0 {
                    return None;
                }
                format!(
                    "dst_unused:{}",
                    UNUSED.get(v as usize).copied().unwrap_or("RESERVED")
                )
            }
            "ROW_MASK" | "BANK_MASK" => {
                if v == 0xf {
                    return None;
                }
                format!("{lower}:{v:#x}")
            }
            "DPP_CTRL" => format!("dpp_ctrl:{v:#x}"),
            "OP_SEL" | "OP_SEL_HI" | "NEG" | "NEG_HI" | "ABS" => {
                if v == 0 && name != "OP_SEL_HI" {
                    return None;
                }
                let n = if name == "OP_SEL" && self.is("ENC_VOP3") {
                    4
                } else {
                    3
                };
                let bits: Vec<String> = (0..n).map(|i| (v >> i & 1).to_string()).collect();
                format!("{lower}:[{}]", bits.join(","))
            }
            "SC0" => return (v != 0).then(|| "sc0".to_string()),
            "GLC" => return (v != 0).then(|| "glc".to_string()),
            "SC1" | "NT" | "GDS" | "LDS" | "OFFEN" | "IDXEN" | "CLAMP" => {
                return (v != 0).then_some(lower)
            }
            _ => {
                if v == 0 {
                    return None;
                }
                format!("{lower}:{v}")
            }
        })
    }
}
