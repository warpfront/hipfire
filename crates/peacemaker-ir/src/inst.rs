use std::fmt::Write as _;
use smallvec::SmallVec;
use thiserror::Error;

use crate::{cfg::{Body, Terminator}, descriptor::KernelDescriptor, effects::Effects, envelope::Source, metadata::HsaKernelMetadata, operand::{Modifiers, Operand}, provenance::Provenance};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Arch { Gfx1010, Gfx1030, Gfx1100, Gfx1151, Gfx1201 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Setting { Any, On, Off }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target { pub arch: Arch, pub xnack: Setting, pub sramecc: Setting, pub abi_version: u8 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wave { Wave32, Wave64 }
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymbolId(pub String);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Family { Sop1, Sop2, Sopc, Sopk, Sopp, Smem, Vop1, Vop2, Vopc, Vop3, Vop3p, Vopd, Vinterp, Ds, Vmem, Export }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Opcode { pub family: Family, pub id: u16 }
impl Opcode {
    pub fn name(self, arch: Arch) -> Option<&'static str> { crate::isa::lookup_opcode(arch, self).map(|row| row.name) }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VmemForm { Global, Scratch, Flat, Buffer, Image }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Form { Sop1, Sop2, Sopc, Sopk, Sopp, Smem, Vop1, Vop1Dpp, Vop2, Vop2Dpp, Vopc, Vop3, Vop3p, Vopd, Vinterp, Ds, Vmem(VmemForm), Export }

/// Values not represented by semantic operands/modifiers. An honored field is never
/// canonicalized; an ignored field is validated against the table's benign set.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub enum FormFields {
    #[default] None,
    Vop3b { src2_unused: u16 },
    /// VOPD is one instruction with two independently typed opcode halves.
    /// The flattened operand list contains X then Y operands.
    Vopd { y_op: Opcode, x_operands: u8 },
    Bits { ignored: SmallVec<[NamedField; 4]>, honored: SmallVec<[NamedField; 4]> },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedField { pub name: &'static str, pub value: u32 }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inst {
    pub op: Opcode,
    pub form: Form,
    pub fields: FormFields,
    pub operands: SmallVec<[Operand; 6]>,
    pub mods: Modifiers,
    pub literal: Option<u32>,
    pub effects: Effects,
    pub prov: Provenance,
}
impl Inst {
    pub fn text(&self, arch: Arch) -> Result<String, ValidateError> {
        let row = crate::isa::lookup(arch, self.op, self.form)
            .ok_or(ValidateError::UnknownOpcode { op: self.op, form: self.form })?;
        let mut text = row.name.to_owned();
        let mut previous_operand = false;
        for (index, operand) in self.operands.iter().enumerate() {
            if let FormFields::Vopd { y_op, x_operands } = self.fields {
                if index == usize::from(x_operands) {
                    let y_name = y_op.name(arch).ok_or(ValidateError::UnknownOpcode { op: y_op, form: Form::Vopd })?;
                    text.push_str(" :: ");
                    text.push_str(y_name);
                    previous_operand = false;
                }
            }
            if previous_operand {
                if matches!(operand, Operand::Imm(crate::operand::ImmField::DsOffset(_) | crate::operand::ImmField::DsOffset0(_) | crate::operand::ImmField::DsOffset1(_) | crate::operand::ImmField::VmemOffset(_) | crate::operand::ImmField::SmemDisplacement(_)) | Operand::Vmem(crate::operand::VmemToken::Offen)) {
                    text.push(' ');
                } else { text.push_str(", "); }
            } else { text.push(' '); }
            write!(&mut text, "{operand}").expect("String write");
            previous_operand = true;
        }
        self.mods.append_text_suffix(&mut text);
        Ok(text)
    }
    pub fn from_parts(
        arch: Arch, op: Opcode, form: Form, fields: FormFields,
        operands: SmallVec<[Operand; 6]>, mods: Modifiers, literal: Option<u32>,
        prov: Provenance,
    ) -> Result<Self, ValidateError> {
        let effects = if let FormFields::Vopd { y_op, x_operands } = &fields {
            let split = usize::from(*x_operands);
            if split == 0 || split >= operands.len() {
                return Err(ValidateError::Operand("VOPD operand split is outside the packet".into()));
            }
            let mut x = Effects::from_table(arch, op, form, &operands[..split], &mods.cpol)?;
            let y = Effects::from_table(arch, *y_op, Form::Vopd, &operands[split..], &mods.cpol)?;
            x.defs.extend(y.defs);
            x.uses.extend(y.uses);
            x.implicit.reads |= y.implicit.reads;
            x.implicit.writes |= y.implicit.writes;
            x
        } else { Effects::from_table(arch, op, form, &operands, &mods.cpol)? };
        let inst = Self { op, form, fields, operands, mods, literal, effects, prov };
        inst.validate(arch)?;
        Ok(inst)
    }
    pub fn validate(&self, arch: Arch) -> Result<(), ValidateError> {
        let row = crate::isa::lookup(arch, self.op, self.form)
            .ok_or(ValidateError::UnknownOpcode { op: self.op, form: self.form })?;
        for operand in &self.operands {
            operand.validate().map_err(|reason| ValidateError::Operand(reason))?;
        }
        if let FormFields::Vopd { y_op, x_operands } = self.fields {
            if self.form != Form::Vopd || usize::from(x_operands) >= self.operands.len()
                || crate::isa::lookup(arch, y_op, Form::Vopd).is_none() {
                return Err(ValidateError::Operand("VOPD must have both typed opcode halves and operands".into()));
            }
        }
        if let FormFields::Vop3b { src2_unused } = self.fields {
            if src2_unused > 0x1ff || src2_unused == 0xff {
                return Err(ValidateError::DangerousFill { field: "src2_unused", value: src2_unused as u32 });
            }
            if !row.benign_src2.iter().any(|&x| x == src2_unused) {
                return Err(ValidateError::UnknownDontCare { field: "src2_unused", value: src2_unused as u32 });
            }
        }
        for field in self.fields.ignored() {
            if field.name == "src2_unused" && field.value == 0xff {
                return Err(ValidateError::DangerousFill { field: "src2_unused", value: field.value });
            }
            let rule = row.fields.iter().find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnknownDontCare { field: field.name, value: field.value })?;
            if rule.class != crate::isa::FieldClass::Ignored || !rule.allowed.contains(&field.value) {
                return Err(ValidateError::UnknownDontCare { field: field.name, value: field.value });
            }
        }
        for field in self.fields.honored() {
            let rule = row.fields.iter().find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnmodeledField { field: field.name })?;
            if rule.class != crate::isa::FieldClass::Honored || field.value & !rule.mask != 0 {
                return Err(ValidateError::UnmodeledField { field: field.name });
            }
            if !rule.allowed.is_empty() && !rule.allowed.contains(&field.value) {
                return Err(ValidateError::UnsupportedFieldValue { field: field.name, value: field.value });
            }
        }
        if matches!(self.form, Form::Smem) {
            if let Some(Operand::Reg(reg)) = self.operands.first() {
                if reg.kind == crate::reg::Kind::S && reg.len > 1
                    && reg.base % u16::from(reg.len.next_power_of_two().min(4)) != 0 {
                    return Err(ValidateError::MisalignedSmemSdata { base: reg.base, len: reg.len });
                }
            }
        }
        Ok(())
    }
}
impl FormFields {
    pub fn ignored(&self) -> &[NamedField] { match self { Self::Bits { ignored, .. } => ignored, _ => &[] } }
    pub fn honored(&self) -> &[NamedField] { match self { Self::Bits { honored, .. } => honored, _ => &[] } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program { pub target: Target, pub kernels: Vec<Kernel>, pub source: Option<Source> }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel { pub symbol: SymbolId, pub wave: Wave, pub abi: Abi, pub body: Body, pub origin: KernelOrigin }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Abi {
    Hsa { descriptor: KernelDescriptor, metadata: HsaKernelMetadata },
    Raw { user_sgprs: Vec<UserSgprRole>, wave: Wave, lds_bytes: u32, sidecar_sha256: [u8; 32] },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UserSgprRole { PrivateSegmentBuffer, DispatchPtr, QueuePtr, KernargSegmentPtr, DispatchId, FlatScratchInit, PrivateSegmentSize }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelOrigin {
    Frontend { kind: Frontend, object_sha256: [u8; 32], entry_va: u64, size: u64 },
    Authored { builder_crate: String, version: String, git: String },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frontend { Hipcc, Triton, Aco, Builder, Ctor }
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ValidateError {
    #[error("unknown opcode {op:?} / encoding form {form:?}")]
    UnknownOpcode { op: Opcode, form: Form },
    #[error("invalid operand: {0}")]
    Operand(String),
    #[error("unknown don't-care value {value:#x} for {field}")]
    UnknownDontCare { field: &'static str, value: u32 },
    #[error("field {field} has no known encoding rule")]
    UnmodeledField { field: &'static str },
    #[error("semantic field {field} value {value:#x} is outside the modeled form")]
    UnsupportedFieldValue { field: &'static str, value: u32 },
    #[error("unused field {field} value {value:#x} is the literal selector or out of range")]
    DangerousFill { field: &'static str, value: u32 },
    #[error("SMEM SDATA s{base} length {len} must be aligned")]
    MisalignedSmemSdata { base: u16, len: u8 },
    #[error("invalid program layout: {0}")]
    Layout(String),
    #[error("kernel {0} overlaps another kernel in the source image")]
    KernelOverlap(String),
}
impl Program {
    pub fn validate(&self) -> Result<(), ValidateError> {
        if let Some(source) = &self.source {
            if source.elf.kernels.len() != self.kernels.len() {
                return Err(ValidateError::Layout("source ELF kernel slots disagree with program kernels".into()));
            }
        }
        for (index, kernel) in self.kernels.iter().enumerate() {
            if self.kernels[..index].iter().any(|previous| previous.symbol == kernel.symbol) {
                return Err(ValidateError::Layout("two kernels share a symbol".into()));
            }
            if let Some(source) = &self.source {
                let slot = &source.elf.kernels[index];
                if slot.name != kernel.symbol.0 {
                    return Err(ValidateError::Layout("source ELF kernel slot symbol disagrees with program kernel".into()));
                }
                if let KernelOrigin::Frontend { entry_va, size, .. } = &kernel.origin {
                    if slot.entry_va != *entry_va || slot.size != *size {
                        return Err(ValidateError::Layout("source ELF kernel slot range disagrees with origin".into()));
                    }
                }
            }
            match &kernel.abi {
                Abi::Hsa { descriptor, metadata } => {
                    if metadata.parsed.symbol != format!("{}.kd", kernel.symbol.0) {
                        return Err(ValidateError::Layout("HSA metadata symbol does not match descriptor symbol".into()));
                    }
                    if descriptor.kernel_code_properties.wave32() != (kernel.wave == Wave::Wave32) {
                        return Err(ValidateError::Layout("kernel wave disagrees with descriptor".into()));
                    }
                }
                Abi::Raw { wave, .. } if *wave != kernel.wave =>
                    return Err(ValidateError::Layout("raw ABI wave disagrees with kernel".into())),
                Abi::Raw { .. } => {}
            }
            let body = &kernel.body;
            if body.layout.is_empty() || body.blocks.is_empty() {
                return Err(ValidateError::Layout("kernel body must contain instructions and a CFG entry".into()));
            }
            let mut seen = vec![false; body.insts.len()];
            for &id in &body.layout {
                let Some(slot) = seen.get_mut(id.0) else { return Err(ValidateError::Layout("layout refers to missing slot".into())); };
                if *slot { return Err(ValidateError::Layout("layout repeats instruction".into())); }
                *slot = true;
                body.insts.get(id).ok_or_else(|| ValidateError::Layout("layout refers to a tombstone".into()))?
                    .validate(self.target.arch)?;
            }
            if body.insts.iter().any(|(id, _)| !seen[id.0]) { return Err(ValidateError::Layout("live instruction omitted from layout".into())); }
            let mut cursor = 0;
            for block in &body.blocks {
                if block.range.0 != cursor || block.range.1 <= cursor || block.range.1 > body.layout.len() {
                    return Err(ValidateError::Layout("blocks must partition layout into nonempty contiguous runs".into()));
                }
                if block.id.0 >= body.blocks.len() || body.blocks[block.id.0].id != block.id {
                    return Err(ValidateError::Layout("block ids must match block indices".into()));
                }
                for &succ in &block.succs {
                    if body.blocks.get(succ.0).is_none() { return Err(ValidateError::Layout("edge leaves CFG".into())); }
                }
                let last = body.insts.get(body.layout[block.range.1 - 1])
                    .expect("layout instructions were checked above");
                let name = last.op.name(self.target.arch)
                    .ok_or(ValidateError::UnknownOpcode { op: last.op, form: last.form })?;
                let matches_term = match &block.term {
                    Terminator::EndPgm => name == "s_endpgm",
                    Terminator::Jump(_) => name == "s_branch",
                    Terminator::Branch { .. } => name.starts_with("s_cbranch_"),
                    Terminator::FallThrough => name != "s_endpgm" && name != "s_branch" && !name.starts_with("s_cbranch_"),
                    Terminator::Unreachable => true,
                };
                if !matches_term {
                    return Err(ValidateError::Layout("block terminator disagrees with encoded last instruction".into()));
                }
                cursor = block.range.1;
            }
            if cursor != body.layout.len() { return Err(ValidateError::Layout("blocks do not cover layout".into())); }
            if !body.blocks.is_empty() {
                let mut reached = vec![false; body.blocks.len()];
                let mut frontier = Vec::with_capacity(body.blocks.len());
                frontier.push(crate::cfg::BlockId(0));
                let mut has_exit = false;
                while let Some(id) = frontier.pop() {
                    if std::mem::replace(&mut reached[id.0], true) { continue; }
                    let block = &body.blocks[id.0];
                    let mut expected: SmallVec<[crate::cfg::BlockId; 2]> = match &block.term {
                        Terminator::FallThrough if id.0 + 1 < body.blocks.len() =>
                            smallvec::smallvec![crate::cfg::BlockId(id.0 + 1)],
                        Terminator::FallThrough => SmallVec::new(),
                        Terminator::Jump(to) => smallvec::smallvec![*to],
                        Terminator::Branch { taken, fallthrough, .. } => smallvec::smallvec![*taken, *fallthrough],
                        Terminator::EndPgm => { has_exit = true; SmallVec::new() },
                        Terminator::Unreachable => return Err(ValidateError::Layout("reachable unreachable-block terminator".into())),
                    };
                    // A neutral conditional branch (taken == fall-through) has one successor edge.
                    expected.sort_unstable_by_key(|b| b.0);
                    expected.dedup();
                    let mut actual = block.succs.clone();
                    actual.sort_unstable_by_key(|b| b.0);
                    actual.dedup();
                    if actual != expected {
                        return Err(ValidateError::Layout("terminator and successor edges disagree".into()));
                    }
                    if expected.is_empty() && block.term != Terminator::EndPgm {
                        return Err(ValidateError::Layout("reachable path has no s_endpgm".into()));
                    }
                    for next in expected {
                        if !body.blocks[next.0].preds.contains(&id) {
                            return Err(ValidateError::Layout("predecessor and successor edges disagree".into()));
                        }
                        frontier.push(next);
                    }
                }
                if !has_exit { return Err(ValidateError::Layout("no reachable s_endpgm".into())); }
            }
            if let KernelOrigin::Frontend { entry_va, size, .. } = &kernel.origin {
                let end = entry_va.checked_add(*size).ok_or_else(|| ValidateError::KernelOverlap(kernel.symbol.0.clone()))?;
                if self.kernels[..index].iter().any(|previous| {
                    if let KernelOrigin::Frontend { entry_va: lo, size: previous_size, .. } = &previous.origin {
                        lo.checked_add(*previous_size).is_none_or(|hi| *entry_va < hi && *lo < end)
                    } else { false }
                }) {
                    return Err(ValidateError::KernelOverlap(kernel.symbol.0.clone()));
                }
            }
        }
        Ok(())
    }
}
