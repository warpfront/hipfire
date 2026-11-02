// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use smallvec::SmallVec;
use std::fmt::Write as _;
use thiserror::Error;

use crate::{
    cfg::{Body, Terminator},
    descriptor::KernelDescriptor,
    effects::Effects,
    envelope::Source,
    metadata::HsaKernelMetadata,
    operand::{Modifiers, Operand, Special},
    provenance::Provenance,
    reg::RegRef,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Arch {
    Gfx1010,
    Gfx1030,
    Gfx1100,
    Gfx1151,
    Gfx1201,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Setting {
    Any,
    On,
    Off,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target {
    pub arch: Arch,
    pub xnack: Setting,
    pub sramecc: Setting,
    pub abi_version: u8,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wave {
    Wave32,
    Wave64,
}
impl Wave {
    /// Work-items per wavefront.
    pub fn lanes(self) -> usize {
        match self {
            Wave::Wave32 => 32,
            Wave::Wave64 => 64,
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymbolId(pub String);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
/// One family per VMEM segment: BUFFER/GLOBAL/SCRATCH/FLAT reuse opcode ids with different mnemonics,
/// so the segment is part of an opcode's identity (`Opcode::name` is form-blind).
pub enum Family {
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    Smem,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vop3p,
    Vopd,
    Vinterp,
    Ds,
    VmemGlobal,
    VmemScratch,
    VmemFlat,
    VmemBuffer,
    VmemImage,
    Export,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Opcode {
    pub family: Family,
    pub id: u16,
}
impl Opcode {
    pub fn name(self, arch: Arch) -> Option<&'static str> {
        crate::isa::lookup_opcode(arch, self).map(|row| row.name)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VmemForm {
    Global,
    Scratch,
    Flat,
    Buffer,
    Image,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Form {
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    Smem,
    Vop1,
    Vop1Dpp,
    Vop2,
    Vop2Dpp,
    Vopc,
    Vop3,
    Vop3p,
    Vopd,
    Vinterp,
    Ds,
    Vmem(VmemForm),
    Export,
}

/// Values not represented by semantic operands/modifiers. An honored field is never
/// canonicalized; an ignored field is validated against the table's benign set.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub enum FormFields {
    #[default]
    None,
    Vop3b {
        src2_unused: u16,
    },
    /// VOPD is one instruction with two independently typed opcode halves.
    /// The flattened operand list contains X then Y operands.
    Vopd {
        y_op: Opcode,
        x_operands: u8,
    },
    Bits {
        ignored: SmallVec<[NamedField; 4]>,
        honored: SmallVec<[NamedField; 4]>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedField {
    pub name: &'static str,
    pub value: u32,
}
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
        let row =
            crate::isa::lookup(arch, self.op, self.form).ok_or(ValidateError::UnknownOpcode {
                op: self.op,
                form: self.form,
            })?;
        let mut text = row.name.to_owned();
        let mut previous_operand = false;
        for (index, operand) in self.operands.iter().enumerate() {
            if let FormFields::Vopd { y_op, x_operands } = self.fields {
                if index == usize::from(x_operands) {
                    let y_name = y_op.name(arch).ok_or(ValidateError::UnknownOpcode {
                        op: y_op,
                        form: Form::Vopd,
                    })?;
                    text.push_str(" :: ");
                    text.push_str(y_name);
                    previous_operand = false;
                }
            }
            if previous_operand {
                if matches!(
                    operand,
                    Operand::Imm(
                        crate::operand::ImmField::DsOffset(_)
                            | crate::operand::ImmField::DsOffset0(_)
                            | crate::operand::ImmField::DsOffset1(_)
                            | crate::operand::ImmField::VmemOffset(_)
                            | crate::operand::ImmField::SmemDisplacement(_)
                    ) | Operand::Vmem(crate::operand::VmemToken::Offen)
                ) {
                    text.push(' ');
                } else {
                    text.push_str(", ");
                }
            } else {
                text.push(' ');
            }
            write!(&mut text, "{operand}").expect("String write");
            previous_operand = true;
        }
        self.mods.append_text_suffix(&mut text, row.name);
        Ok(text)
    }
    pub fn from_parts(
        arch: Arch,
        op: Opcode,
        form: Form,
        fields: FormFields,
        operands: SmallVec<[Operand; 6]>,
        mods: Modifiers,
        literal: Option<u32>,
        prov: Provenance,
    ) -> Result<Self, ValidateError> {
        let effects = if let FormFields::Vopd { y_op, x_operands } = &fields {
            let split = usize::from(*x_operands);
            if split == 0 || split >= operands.len() {
                return Err(ValidateError::Operand(
                    "VOPD operand split is outside the packet".into(),
                ));
            }
            let mut x = Effects::from_table(arch, op, form, &operands[..split], &mods.cpol)?;
            let y = Effects::from_table(arch, *y_op, Form::Vopd, &operands[split..], &mods.cpol)?;
            x.defs.extend(y.defs);
            x.uses.extend(y.uses);
            x.implicit.reads |= y.implicit.reads;
            x.implicit.writes |= y.implicit.writes;
            x
        } else {
            Effects::from_table(arch, op, form, &operands, &mods.cpol)?
        };
        let inst = Self {
            op,
            form,
            fields,
            operands,
            mods,
            literal,
            effects,
            prov,
        };
        inst.validate(arch)?;
        Ok(inst)
    }
    pub fn validate(&self, arch: Arch) -> Result<(), ValidateError> {
        let row =
            crate::isa::lookup(arch, self.op, self.form).ok_or(ValidateError::UnknownOpcode {
                op: self.op,
                form: self.form,
            })?;
        for operand in &self.operands {
            operand
                .validate()
                .map_err(|reason| ValidateError::Operand(reason))?;
        }
        if let FormFields::Vopd { y_op, x_operands } = self.fields {
            if self.form != Form::Vopd
                || usize::from(x_operands) >= self.operands.len()
                || crate::isa::lookup(arch, y_op, Form::Vopd).is_none()
            {
                return Err(ValidateError::Operand(
                    "VOPD must have both typed opcode halves and operands".into(),
                ));
            }
        }
        if let FormFields::Vop3b { src2_unused } = self.fields {
            if src2_unused > 0x1ff || src2_unused == 0xff {
                return Err(ValidateError::DangerousFill {
                    field: "src2_unused",
                    value: src2_unused as u32,
                });
            }
            if !row.benign_src2.iter().any(|&x| x == src2_unused) {
                return Err(ValidateError::UnknownDontCare {
                    field: "src2_unused",
                    value: src2_unused as u32,
                });
            }
        }
        for field in self.fields.ignored() {
            if field.name == "src2_unused" && field.value == 0xff {
                return Err(ValidateError::DangerousFill {
                    field: "src2_unused",
                    value: field.value,
                });
            }
            let rule = row
                .fields
                .iter()
                .find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnknownDontCare {
                    field: field.name,
                    value: field.value,
                })?;
            if rule.class != crate::isa::FieldClass::Ignored || !rule.allowed.contains(&field.value)
            {
                return Err(ValidateError::UnknownDontCare {
                    field: field.name,
                    value: field.value,
                });
            }
        }
        for field in self.fields.honored() {
            let rule = row
                .fields
                .iter()
                .find(|rule| rule.name == field.name)
                .ok_or(ValidateError::UnmodeledField { field: field.name })?;
            if rule.class != crate::isa::FieldClass::Honored || field.value & !rule.mask != 0 {
                return Err(ValidateError::UnmodeledField { field: field.name });
            }
            if !rule.allowed.is_empty() && !rule.allowed.contains(&field.value) {
                return Err(ValidateError::UnsupportedFieldValue {
                    field: field.name,
                    value: field.value,
                });
            }
        }
        if matches!(self.form, Form::Smem) {
            if let Some(Operand::Reg(reg)) = self.operands.first() {
                if reg.kind == crate::reg::Kind::S
                    && reg.len > 1
                    && reg.base % u16::from(reg.len.next_power_of_two().min(4)) != 0
                {
                    return Err(ValidateError::MisalignedSmemSdata {
                        base: reg.base,
                        len: reg.len,
                    });
                }
            }
        }
        Ok(())
    }
}
impl FormFields {
    pub fn ignored(&self) -> &[NamedField] {
        match self {
            Self::Bits { ignored, .. } => ignored,
            _ => &[],
        }
    }
    pub fn honored(&self) -> &[NamedField] {
        match self {
            Self::Bits { honored, .. } => honored,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub target: Target,
    pub kernels: Vec<Kernel>,
    pub source: Option<Source>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    pub symbol: SymbolId,
    pub wave: Wave,
    pub abi: Abi,
    pub body: Body,
    pub origin: KernelOrigin,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Abi {
    Hsa {
        descriptor: KernelDescriptor,
        metadata: HsaKernelMetadata,
    },
    /// A foreign (non-HSA) executable stream such as a Mesa/ACO shader. It carries the
    /// producer's declared entry state and resources; nothing is inferred from HSA.
    Raw(RawAbi),
}

/// Workgroup/local-ID axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// One local-ID component packed into a VGPR dword: `(v >> shift) & (2^bits - 1)`, known to
/// be at most `max` for every launch the contract admits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalIdField {
    pub axis: Axis,
    pub shift: u8,
    pub bits: u8,
    pub max: u32,
}

/// A physical location at kernel entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntryLoc {
    /// `len` consecutive registers of one file.
    Reg(RegRef),
    /// Architectural special register. `Exec`/`Vcc` cover the lanes of the kernel's wave.
    Special(Special),
    /// The `s_getreg` hardware-register state.
    Hwreg,
}

/// What is known about the dword(s) of a location at entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EntryValue {
    /// Defined with a value the contract does not constrain (descriptor pointers, push
    /// constants, the wave's initial EXEC).
    Defined,
    /// A work-group id of the dispatch (single dword, unconstrained).
    WorkgroupId(Axis),
    /// Exactly this value (single dword).
    Exact(u32),
    /// Within `min..=max` inclusive (single dword, unsigned).
    Bounded { min: u32, max: u32 },
    /// Local work-item ids packed into one VGPR; bits outside the fields are zero.
    LocalId(Vec<LocalIdField>),
}

/// A location defined at entry. A location with no fact is undefined at entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EntryFact {
    pub loc: EntryLoc,
    pub value: EntryValue,
}

/// Declared resources. Actual and allocated register counts are distinct, and every
/// register an entry fact names must be inside the allocation.
///
/// Count basis: SGPR counts may be ordinary (`s0..s105`, at most 106) or the driver's physical
/// count, where `VCC_LO` = 106 and `VCC_HI` = 107 follow the ordinary file (at most 108); the
/// producer's value is kept verbatim, never truncated. Only the first
/// [`RawResources::ORDINARY_SGPRS`] are ordinary: code and entry facts are bound by
/// [`RawResources::ordinary_sgprs_actual`] / [`RawResources::ordinary_sgprs_allocated`], never by
/// the total, and VCC is a `Special` that needs its own entry fact.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawResources {
    pub lds_bytes: u32,
    pub scratch_bytes_per_lane: u32,
    pub vgprs_actual: u16,
    pub vgprs_allocated: u16,
    pub sgprs_actual: u16,
    pub sgprs_allocated: u16,
}
impl RawResources {
    /// Addressable ordinary SGPRs `s0..s105`.
    pub const ORDINARY_SGPRS: u16 = 106;
    /// Physical SGPR count limit including `VCC_LO`/`VCC_HI` (106, 107).
    pub const SGPR_COUNT_LIMIT: u16 = 108;
    pub fn ordinary_sgprs_actual(&self) -> u16 {
        self.sgprs_actual.min(Self::ORDINARY_SGPRS)
    }
    pub fn ordinary_sgprs_allocated(&self) -> u16 {
        self.sgprs_allocated.min(Self::ORDINARY_SGPRS)
    }
}

/// Floating-point and execution mode state the shader config declares (RSRC1/RSRC2 fields,
/// no defaults). The MODE register is defined at entry because this state is declared.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FpState {
    pub round32: u8,
    pub round16_64: u8,
    pub denorm32: u8,
    pub denorm16_64: u8,
    pub dx10_clamp: bool,
    pub ieee_mode: bool,
    pub fp16_overflow: bool,
    pub wgp_mode: bool,
    pub mem_ordered: bool,
    pub forward_progress: bool,
    /// gfx12 RSRC1 bit 21 `WG_RR_EN` (`.amdhsa_round_robin_scheduling`). Bit 21 is
    /// `DX10_CLAMP` on gfx11, so it must be false there.
    pub wg_round_robin: bool,
    /// RSRC2 `EXCP_EN` bits.
    pub exceptions: u16,
}

/// A relocation against a dword of the executable range. Its dword must be an instruction
/// literal; the stored value is the unrelocated one. `emit::module` and any adapter must
/// refuse a program with relocations they do not resolve.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Relocation {
    pub offset: u64,
    pub r_type: u32,
    pub symbol: String,
    pub addend: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawAbi {
    pub wave: Wave,
    pub resources: RawResources,
    pub fp: FpState,
    pub relocations: Vec<Relocation>,
    pub entry: Vec<EntryFact>,
    pub sidecar_sha256: [u8; 32],
}

/// The byte range of the executable stream inside the binary handed to `lift_raw`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecExtent {
    pub offset: u64,
    pub len: u64,
}

/// Everything a producer must state about one raw kernel. Owned by the core so the lifter,
/// the adapter and the certifier share one definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawKernelSpec {
    pub target: Target,
    pub symbol: SymbolId,
    pub wave: Wave,
    pub extent: ExecExtent,
    /// SHA-256 of the whole binary and of the executable range, as the producer recorded.
    pub binary_sha256: [u8; 32],
    pub exec_sha256: [u8; 32],
    pub resources: RawResources,
    pub fp: FpState,
    pub relocations: Vec<Relocation>,
    pub entry: Vec<EntryFact>,
    pub sidecar_sha256: [u8; 32],
}

/// A single architectural storage cell named by an entry fact (VGPR/SGPR/TTMP dword,
/// VCC/EXEC half, ...). Wave32 has no `VccHi`/`ExecHi`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum EntryCell {
    Vgpr(u16),
    Sgpr(u16),
    Ttmp(u8),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    Scc,
    M0,
    FlatScratch,
    SrcSharedBase,
    SrcSharedLimit,
    SrcPrivateBase,
    SrcPrivateLimit,
    Pc,
    Hwreg,
}

impl EntryLoc {
    /// The cells the location covers under `wave`; an error names an unrepresentable location.
    pub fn cells(self, wave: Wave) -> Result<SmallVec<[EntryCell; 4]>, String> {
        let mut out: SmallVec<[EntryCell; 4]> = SmallVec::new();
        let wide = wave == Wave::Wave64;
        match self {
            EntryLoc::Reg(reg) => {
                reg.validate()?;
                for n in reg.base..reg.base + u16::from(reg.len) {
                    out.push(match reg.kind {
                        crate::reg::Kind::V => EntryCell::Vgpr(n),
                        crate::reg::Kind::S => EntryCell::Sgpr(n),
                        crate::reg::Kind::Ttmp => EntryCell::Ttmp(n as u8),
                    });
                }
            }
            EntryLoc::Special(special) => match special {
                Special::Vcc => {
                    out.push(EntryCell::VccLo);
                    if wide {
                        out.push(EntryCell::VccHi);
                    }
                }
                Special::VccLo => out.push(EntryCell::VccLo),
                Special::VccHi if wide => out.push(EntryCell::VccHi),
                Special::Exec => {
                    out.push(EntryCell::ExecLo);
                    if wide {
                        out.push(EntryCell::ExecHi);
                    }
                }
                Special::ExecLo => out.push(EntryCell::ExecLo),
                Special::ExecHi if wide => out.push(EntryCell::ExecHi),
                Special::VccHi | Special::ExecHi => return Err("high half of a wave32 mask".into()),
                Special::Scc => out.push(EntryCell::Scc),
                Special::M0 => out.push(EntryCell::M0),
                Special::Null => return Err("null is not a storage location".into()),
                Special::Ttmp(n) if n < 16 => out.push(EntryCell::Ttmp(n)),
                Special::Ttmp(n) => return Err(format!("ttmp{n} is out of range")),
                Special::FlatScratch => out.push(EntryCell::FlatScratch),
                Special::SrcSharedBase => out.push(EntryCell::SrcSharedBase),
                Special::SrcSharedLimit => out.push(EntryCell::SrcSharedLimit),
                Special::SrcPrivateBase => out.push(EntryCell::SrcPrivateBase),
                Special::SrcPrivateLimit => out.push(EntryCell::SrcPrivateLimit),
                Special::Pc => out.push(EntryCell::Pc),
            },
            EntryLoc::Hwreg => out.push(EntryCell::Hwreg),
        }
        Ok(out)
    }
}

impl FpState {
    /// Target-dependent rules: RSRC1 bit 21 is `WG_RR_EN` on gfx12 and `DX10_CLAMP` on gfx11,
    /// so `wg_round_robin` is only meaningful on gfx12.
    pub fn validate_for(&self, arch: Arch) -> Result<(), String> {
        if self.wg_round_robin && arch != Arch::Gfx1201 {
            return Err(format!(
                "wg_round_robin (RSRC1 bit 21) is a gfx12 field; it is DX10_CLAMP on {arch:?}"
            ));
        }
        Ok(())
    }
    fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("round32", self.round32),
            ("round16_64", self.round16_64),
            ("denorm32", self.denorm32),
            ("denorm16_64", self.denorm16_64),
        ] {
            if v > 3 {
                return Err(format!("{name} {v} does not fit two bits"));
            }
        }
        if self.exceptions > 0x1ff {
            return Err(format!(
                "exception enables {:#x} exceed nine bits",
                self.exceptions
            ));
        }
        Ok(())
    }
}

impl RawAbi {
    /// Entry cells of every fact, each with its fact; errors on malformed or overlapping facts.
    pub fn entry_cells(&self) -> Result<Vec<(EntryCell, &EntryFact)>, String> {
        let mut seen: Vec<(EntryCell, &EntryFact)> = Vec::new();
        for fact in &self.entry {
            let cells = fact.loc.cells(self.wave)?;
            let single = cells.len() == 1;
            match &fact.value {
                EntryValue::Defined => {}
                EntryValue::WorkgroupId(_) | EntryValue::Exact(_) | EntryValue::Bounded { .. } if !single =>
                    return Err(format!("{:?}: exact, bounded and work-group-id values need a single-dword location", fact.loc)),
                EntryValue::Bounded { min, max } if min > max => return Err(format!("{:?}: empty bound {min}..={max}", fact.loc)),
                EntryValue::WorkgroupId(_) if !matches!(fact.loc, EntryLoc::Reg(RegRef { kind: crate::reg::Kind::S | crate::reg::Kind::Ttmp, .. }) | EntryLoc::Special(Special::Ttmp(_))) =>
                    return Err(format!("{:?}: a work-group id lives in an SGPR or TTMP", fact.loc)),
                EntryValue::LocalId(fields) => {
                    if !matches!(fact.loc, EntryLoc::Reg(RegRef { kind: crate::reg::Kind::V, len: 1, .. })) {
                        return Err(format!("{:?}: packed local ids live in one VGPR", fact.loc));
                    }
                    if fields.is_empty() { return Err("local-id packing has no fields".into()); }
                    let mut used = 0u32;
                    let mut axes = 0u8;
                    for f in fields {
                        if f.bits == 0 || u32::from(f.shift) + u32::from(f.bits) > 32 {
                            return Err(format!("local-id field shift {} bits {} is outside a dword", f.shift, f.bits));
                        }
                        let mask = (u64::MAX >> (64 - u32::from(f.bits))) as u32;
                        let field = mask << f.shift;
                        if f.max > mask { return Err(format!("local-id bound {} does not fit {} bits", f.max, f.bits)); }
                        if used & field != 0 { return Err("local-id fields overlap".into()); }
                        used |= field;
                        let bit = 1u8 << (f.axis as u8);
                        if axes & bit != 0 { return Err(format!("local-id axis {:?} is packed twice", f.axis)); }
                        axes |= bit;
                    }
                }
                _ => {}
            }
            for cell in cells {
                if let Some((_, other)) = seen.iter().find(|(c, _)| *c == cell) {
                    return Err(format!(
                        "entry facts {:?} and {:?} overlap at {cell:?}",
                        other.loc, fact.loc
                    ));
                }
                seen.push((cell, fact));
            }
        }
        Ok(seen)
    }

    pub fn validate(&self) -> Result<(), ValidateError> {
        let bad = ValidateError::RawAbi;
        let r = &self.resources;
        if r.vgprs_actual > r.vgprs_allocated || r.vgprs_allocated > 256 {
            return Err(bad(format!(
                "VGPR counts actual {} / allocated {} are inconsistent",
                r.vgprs_actual, r.vgprs_allocated
            )));
        }
        if r.sgprs_actual > r.sgprs_allocated || r.sgprs_allocated > RawResources::SGPR_COUNT_LIMIT
        {
            return Err(bad(format!("SGPR counts actual {} / allocated {} are inconsistent (physical basis, at most {})",
                r.sgprs_actual, r.sgprs_allocated, RawResources::SGPR_COUNT_LIMIT)));
        }
        self.fp.validate().map_err(bad)?;
        for (cell, _) in self.entry_cells().map_err(bad)? {
            match cell {
                EntryCell::Vgpr(n) if n >= r.vgprs_allocated => {
                    return Err(bad(format!(
                        "entry v{n} is outside the {} allocated VGPRs",
                        r.vgprs_allocated
                    )))
                }
                EntryCell::Sgpr(n) if n >= r.ordinary_sgprs_allocated() => {
                    return Err(bad(format!(
                        "entry s{n} is outside the {} allocated ordinary SGPRs",
                        r.ordinary_sgprs_allocated()
                    )))
                }
                _ => {}
            }
        }
        if let Some(rel) = self.relocations.iter().find(|rel| rel.offset % 4 != 0) {
            return Err(bad(format!(
                "relocation offset {:#x} is not dword aligned",
                rel.offset
            )));
        }
        Ok(())
    }
}

impl RawKernelSpec {
    /// The ABI a lifted kernel carries.
    pub fn abi(&self) -> RawAbi {
        RawAbi {
            wave: self.wave,
            resources: self.resources,
            fp: self.fp,
            relocations: self.relocations.clone(),
            entry: self.entry.clone(),
            sidecar_sha256: self.sidecar_sha256,
        }
    }
    /// Self-consistency of the declaration (not of any binary).
    pub fn validate(&self) -> Result<(), ValidateError> {
        self.abi().validate()?;
        self.fp
            .validate_for(self.target.arch)
            .map_err(ValidateError::RawAbi)?;
        if self.extent.len == 0 || self.extent.len % 4 != 0 || self.extent.offset % 4 != 0 {
            return Err(ValidateError::RawAbi(format!(
                "executable extent {:#x}+{:#x} is empty or not dword aligned",
                self.extent.offset, self.extent.len
            )));
        }
        if self.extent.offset.checked_add(self.extent.len).is_none() {
            return Err(ValidateError::RawAbi("executable extent overflows".into()));
        }
        if let Some(rel) = self
            .relocations
            .iter()
            .find(|rel| rel.offset >= self.extent.len)
        {
            return Err(ValidateError::RawAbi(format!(
                "relocation offset {:#x} is outside the executable range",
                rel.offset
            )));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelOrigin {
    Frontend {
        kind: Frontend,
        object_sha256: [u8; 32],
        entry_va: u64,
        size: u64,
    },
    Authored {
        builder_crate: String,
        version: String,
        git: String,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frontend {
    Hipcc,
    Triton,
    Aco,
    Builder,
    Ctor,
}
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
    #[error("invalid raw kernel contract: {0}")]
    RawAbi(String),
    #[error("kernel {0} overlaps another kernel in the source image")]
    KernelOverlap(String),
}
impl Program {
    pub fn validate(&self) -> Result<(), ValidateError> {
        if let Some(source) = &self.source {
            if source.elf.kernels.len() != self.kernels.len() {
                return Err(ValidateError::Layout(
                    "source ELF kernel slots disagree with program kernels".into(),
                ));
            }
        }
        for (index, kernel) in self.kernels.iter().enumerate() {
            if self.kernels[..index]
                .iter()
                .any(|previous| previous.symbol == kernel.symbol)
            {
                return Err(ValidateError::Layout("two kernels share a symbol".into()));
            }
            if let Some(source) = &self.source {
                let slot = &source.elf.kernels[index];
                if slot.name != kernel.symbol.0 {
                    return Err(ValidateError::Layout(
                        "source ELF kernel slot symbol disagrees with program kernel".into(),
                    ));
                }
                if let KernelOrigin::Frontend { entry_va, size, .. } = &kernel.origin {
                    if slot.entry_va != *entry_va || slot.size != *size {
                        return Err(ValidateError::Layout(
                            "source ELF kernel slot range disagrees with origin".into(),
                        ));
                    }
                }
            }
            match &kernel.abi {
                Abi::Hsa {
                    descriptor,
                    metadata,
                } => {
                    if metadata.parsed.symbol != format!("{}.kd", kernel.symbol.0) {
                        return Err(ValidateError::Layout(
                            "HSA metadata symbol does not match descriptor symbol".into(),
                        ));
                    }
                    if descriptor.kernel_code_properties.wave32() != (kernel.wave == Wave::Wave32) {
                        return Err(ValidateError::Layout(
                            "kernel wave disagrees with descriptor".into(),
                        ));
                    }
                }
                Abi::Raw(raw) if raw.wave != kernel.wave => {
                    return Err(ValidateError::Layout(
                        "raw ABI wave disagrees with kernel".into(),
                    ))
                }
                Abi::Raw(raw) => {
                    raw.validate()?;
                    raw.fp
                        .validate_for(self.target.arch)
                        .map_err(ValidateError::RawAbi)?;
                }
            }
            let body = &kernel.body;
            if body.layout.is_empty() || body.blocks.is_empty() {
                return Err(ValidateError::Layout(
                    "kernel body must contain instructions and a CFG entry".into(),
                ));
            }
            let mut seen = vec![false; body.insts.len()];
            for &id in &body.layout {
                let Some(slot) = seen.get_mut(id.0) else {
                    return Err(ValidateError::Layout(
                        "layout refers to missing slot".into(),
                    ));
                };
                if *slot {
                    return Err(ValidateError::Layout("layout repeats instruction".into()));
                }
                *slot = true;
                body.insts
                    .get(id)
                    .ok_or_else(|| ValidateError::Layout("layout refers to a tombstone".into()))?
                    .validate(self.target.arch)?;
            }
            if body.insts.iter().any(|(id, _)| !seen[id.0]) {
                return Err(ValidateError::Layout(
                    "live instruction omitted from layout".into(),
                ));
            }
            let mut cursor = 0;
            for block in &body.blocks {
                if block.range.0 != cursor
                    || block.range.1 <= cursor
                    || block.range.1 > body.layout.len()
                {
                    return Err(ValidateError::Layout(
                        "blocks must partition layout into nonempty contiguous runs".into(),
                    ));
                }
                if block.id.0 >= body.blocks.len() || body.blocks[block.id.0].id != block.id {
                    return Err(ValidateError::Layout(
                        "block ids must match block indices".into(),
                    ));
                }
                for &succ in &block.succs {
                    if body.blocks.get(succ.0).is_none() {
                        return Err(ValidateError::Layout("edge leaves CFG".into()));
                    }
                }
                let last = body
                    .insts
                    .get(body.layout[block.range.1 - 1])
                    .expect("layout instructions were checked above");
                let name = last
                    .op
                    .name(self.target.arch)
                    .ok_or(ValidateError::UnknownOpcode {
                        op: last.op,
                        form: last.form,
                    })?;
                let matches_term = match &block.term {
                    Terminator::EndPgm => name == "s_endpgm",
                    Terminator::Jump(_) => name == "s_branch",
                    Terminator::Branch { .. } => name.starts_with("s_cbranch_"),
                    Terminator::FallThrough => {
                        name != "s_endpgm" && name != "s_branch" && !name.starts_with("s_cbranch_")
                    }
                    Terminator::Unreachable => true,
                };
                if !matches_term {
                    return Err(ValidateError::Layout(
                        "block terminator disagrees with encoded last instruction".into(),
                    ));
                }
                cursor = block.range.1;
            }
            if cursor != body.layout.len() {
                return Err(ValidateError::Layout("blocks do not cover layout".into()));
            }
            if !body.blocks.is_empty() {
                let mut reached = vec![false; body.blocks.len()];
                let mut frontier = Vec::with_capacity(body.blocks.len());
                frontier.push(crate::cfg::BlockId(0));
                let mut has_exit = false;
                while let Some(id) = frontier.pop() {
                    if std::mem::replace(&mut reached[id.0], true) {
                        continue;
                    }
                    let block = &body.blocks[id.0];
                    let mut expected: SmallVec<[crate::cfg::BlockId; 2]> = match &block.term {
                        Terminator::FallThrough if id.0 + 1 < body.blocks.len() => {
                            smallvec::smallvec![crate::cfg::BlockId(id.0 + 1)]
                        }
                        Terminator::FallThrough => SmallVec::new(),
                        Terminator::Jump(to) => smallvec::smallvec![*to],
                        Terminator::Branch {
                            taken, fallthrough, ..
                        } => smallvec::smallvec![*taken, *fallthrough],
                        Terminator::EndPgm => {
                            has_exit = true;
                            SmallVec::new()
                        }
                        Terminator::Unreachable => {
                            return Err(ValidateError::Layout(
                                "reachable unreachable-block terminator".into(),
                            ))
                        }
                    };
                    // A neutral conditional branch (taken == fall-through) has one successor edge.
                    expected.sort_unstable_by_key(|b| b.0);
                    expected.dedup();
                    let mut actual = block.succs.clone();
                    actual.sort_unstable_by_key(|b| b.0);
                    actual.dedup();
                    if actual != expected {
                        return Err(ValidateError::Layout(
                            "terminator and successor edges disagree".into(),
                        ));
                    }
                    if expected.is_empty() && block.term != Terminator::EndPgm {
                        return Err(ValidateError::Layout(
                            "reachable path has no s_endpgm".into(),
                        ));
                    }
                    for next in expected {
                        if !body.blocks[next.0].preds.contains(&id) {
                            return Err(ValidateError::Layout(
                                "predecessor and successor edges disagree".into(),
                            ));
                        }
                        frontier.push(next);
                    }
                }
                if !has_exit {
                    return Err(ValidateError::Layout("no reachable s_endpgm".into()));
                }
            }
            if let KernelOrigin::Frontend { entry_va, size, .. } = &kernel.origin {
                let end = entry_va
                    .checked_add(*size)
                    .ok_or_else(|| ValidateError::KernelOverlap(kernel.symbol.0.clone()))?;
                if self.kernels[..index].iter().any(|previous| {
                    if let KernelOrigin::Frontend {
                        entry_va: lo,
                        size: previous_size,
                        ..
                    } = &previous.origin
                    {
                        lo.checked_add(*previous_size)
                            .is_none_or(|hi| *entry_va < hi && *lo < end)
                    } else {
                        false
                    }
                }) {
                    return Err(ValidateError::KernelOverlap(kernel.symbol.0.clone()));
                }
            }
        }
        Ok(())
    }
}
