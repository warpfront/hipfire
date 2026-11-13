// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-target opcode/encoding rows. Tables are backed by AMD XML and pinned LLVM samples.
use crate::inst::{Arch, Family, Form, Opcode, VmemForm};
use crate::operand::CachePolicy;
use std::sync::LazyLock;
pub mod xml;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldClass {
    Ignored,
    Honored,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldRule {
    pub name: &'static str,
    pub class: FieldClass,
    pub mask: u32,
    pub allowed: Vec<u32>,
}
/// Instruction-local encoding identity, including conditional and extension forms.
/// `None` on an instruction denotes the unique census encoding, not a wildcard.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EncodingVariant {
    pub instruction: &'static str,
    pub encoding: &'static str,
    pub condition: &'static str,
    pub ordinal: u16,
}
#[derive(Clone, Debug)]
pub struct OpRow {
    pub op: Opcode,
    pub form: Form,
    pub variant: Option<EncodingVariant>,
    pub name: &'static str,
    pub grammar: &'static str,
    pub defs: &'static str,
    pub uses: &'static str,
    pub implicit: &'static str,
    pub counter: &'static str,
    pub sample: &'static str,
    pub encoding: &'static str,
    pub benign_src2: Vec<u16>,
    pub fields: Vec<FieldRule>,
}
impl OpRow {
    /// Printed operand slots `(field, bits)` in grammar order. A slot tagged
    /// `@rtn` (a gfx11 VMEM atomic's `VDST`) exists only in the returning form;
    /// untyped markers (`literal@last`, `-`) are not slots.
    pub fn slots(&self, returns: bool) -> impl Iterator<Item = (&'static str, u16)> {
        let grammar: &'static str = self.grammar;
        grammar.split(',').filter_map(move |part| {
            let (name, bits) = part.split_once(':')?;
            let bits = bits.parse().ok()?;
            match name.strip_suffix("@rtn") {
                Some(name) => returns.then_some((name, bits)),
                None => Some((name, bits)),
            }
        })
    }
    /// Counter rules `(counter, units)`. A rule tagged `@rtn` / `@nortn` applies
    /// only to the returning / non-returning form of a VMEM atomic.
    pub fn counter_rules(
        &self,
        returns: bool,
    ) -> impl Iterator<Item = (&'static str, &'static str)> {
        let counter: &'static str = self.counter;
        counter.split(',').filter_map(move |pair| {
            let (counter, units) = pair.split_once(':')?;
            match units.split_once('@') {
                Some((units, "rtn")) => returns.then_some((counter, units)),
                Some((units, "nortn")) => (!returns).then_some((counter, units)),
                _ => Some((counter, units)),
            }
        })
    }
}
/// Whether a VMEM atomic returns its pre-op value: `glc` on gfx11, `th` bit 0
/// (`TH_ATOMIC_RETURN`) on gfx12. Only rows with `@rtn`/`@nortn` tags consult it.
pub fn atomic_returns(arch: Arch, cpol: &CachePolicy) -> bool {
    if arch.is_gfx12() {
        cpol.th & 1 != 0
    } else {
        cpol.glc
    }
}
#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("line {line}: malformed opcode row: {reason}")]
    Malformed { line: usize, reason: String },
    #[error("line {line}: duplicate opcode/form")]
    Duplicate { line: usize },
}
fn form(value: &str) -> Option<Form> {
    Some(match value {
        "sop1" => Form::Sop1,
        "sop2" => Form::Sop2,
        "sopc" => Form::Sopc,
        "sopk" => Form::Sopk,
        "sopp" => Form::Sopp,
        "smem" => Form::Smem,
        "vop1" => Form::Vop1,
        "vop1_dpp" => Form::Vop1Dpp,
        "vop1_dpp8" => Form::Vop1Dpp8,
        "vop2" => Form::Vop2,
        "vop2_dpp" => Form::Vop2Dpp,
        "vopc" => Form::Vopc,
        "vop3" => Form::Vop3,
        "vop3p" => Form::Vop3p,
        "vopd" => Form::Vopd,
        "vinterp" => Form::Vinterp,
        "ds" => Form::Ds,
        "global" => Form::Vmem(VmemForm::Global),
        "scratch" => Form::Vmem(VmemForm::Scratch),
        "flat" => Form::Vmem(VmemForm::Flat),
        "buffer" => Form::Vmem(VmemForm::Buffer),
        "image" => Form::Vmem(VmemForm::Image),
        "export" => Form::Export,
        _ => return None,
    })
}
fn family(form: Form) -> Family {
    match form {
        Form::Sop1 => Family::Sop1,
        Form::Sop2 => Family::Sop2,
        Form::Sopc => Family::Sopc,
        Form::Sopk => Family::Sopk,
        Form::Sopp => Family::Sopp,
        Form::Smem => Family::Smem,
        Form::Vop1 | Form::Vop1Dpp | Form::Vop1Dpp8 => Family::Vop1,
        Form::Vop2 | Form::Vop2Dpp => Family::Vop2,
        Form::Vopc => Family::Vopc,
        Form::Vop3 => Family::Vop3,
        Form::Vop3p => Family::Vop3p,
        Form::Vopd => Family::Vopd,
        Form::Vinterp => Family::Vinterp,
        Form::Ds => Family::Ds,
        Form::Vmem(VmemForm::Global) => Family::VmemGlobal,
        Form::Vmem(VmemForm::Scratch) => Family::VmemScratch,
        Form::Vmem(VmemForm::Flat) => Family::VmemFlat,
        Form::Vmem(VmemForm::Buffer) => Family::VmemBuffer,
        Form::Vmem(VmemForm::Image) => Family::VmemImage,
        Form::Export => Family::Export,
    }
}
/// Tab-separated: `name form opcode grammar defs uses implicit counter sample words field-rules`.
/// A VMEM atomic whose return is a cache-policy bit (gfx11 `glc`) is one row: its
/// destination slot is tagged `VDST@rtn:N` and its counter rules `@rtn` / `@nortn`.
pub fn parse_gfx12() -> Result<Vec<OpRow>, TableError> {
    parse(include_str!("../isa/gfx12.tbl"))
}
fn parse(contents: &'static str) -> Result<Vec<OpRow>, TableError> {
    let mut rows = Vec::new();
    for (index, line) in contents.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        let fail = |reason: &str| TableError::Malformed {
            line: index + 1,
            reason: reason.into(),
        };
        if fields.len() != 11 && fields.len() != 14 {
            return Err(fail(
                "expected 11 census or 14 variant-keyed tab-delimited columns",
            ));
        }
        let kind = form(fields[1]).ok_or_else(|| fail("unknown instruction form"))?;
        let id = u16::from_str_radix(fields[2].trim_start_matches("0x"), 16)
            .map_err(|_| fail("bad opcode id"))?;
        let mut rule_fields = Vec::new();
        let mut benign_src2 = Vec::new();
        for field in fields[10].split(',').filter(|s| !s.is_empty() && *s != "-") {
            let parts: Vec<_> = field.split(':').collect();
            if parts.len() < 3 {
                return Err(fail("malformed field rule"));
            }
            let class = match parts[1] {
                "ignored" => FieldClass::Ignored,
                "honored" => FieldClass::Honored,
                _ => return Err(fail("unknown field class")),
            };
            let mask = u32::from_str_radix(parts[2].trim_start_matches("0x"), 16)
                .map_err(|_| fail("bad field mask"))?;
            let allowed = parts
                .get(3)
                .unwrap_or(&"")
                .split('/')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    u32::from_str_radix(s.trim_start_matches("0x"), 16)
                        .map_err(|_| fail("bad benign field value"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if parts[0] == "src2_unused" {
                benign_src2 = allowed.iter().map(|&v| v as u16).collect();
            }
            rule_fields.push(FieldRule {
                name: parts[0],
                class,
                mask,
                allowed,
            });
        }
        let tagged_slot = fields[3]
            .split(',')
            .filter(|part| part.contains('@') && *part != "literal@last")
            .map(|part| part.split(':').next().unwrap_or("").ends_with("@rtn"))
            .reduce(|a, b| a && b);
        let tagged_counter = fields[7]
            .split(',')
            .filter_map(|pair| pair.split_once('@'))
            .map(|(_, tag)| matches!(tag, "rtn" | "nortn"))
            .reduce(|a, b| a && b);
        if tagged_slot == Some(false) || tagged_counter == Some(false) {
            return Err(fail("unknown return-form tag"));
        }
        if (tagged_slot.is_some() || tagged_counter.is_some()) && !matches!(kind, Form::Vmem(_)) {
            return Err(fail("return-form tags apply to VMEM atomics only"));
        }
        let row = OpRow {
            op: Opcode {
                family: family(kind),
                id,
            },
            form: kind,
            variant: if fields.len() == 14 {
                Some(EncodingVariant {
                    instruction: fields[0],
                    encoding: fields[11],
                    condition: fields[12],
                    ordinal: fields[13]
                        .parse()
                        .map_err(|_| fail("bad encoding ordinal"))?,
                })
            } else {
                None
            },
            name: fields[0],
            grammar: fields[3],
            defs: fields[4],
            uses: fields[5],
            implicit: fields[6],
            counter: fields[7],
            sample: fields[8],
            encoding: fields[9],
            benign_src2,
            fields: rule_fields,
        };
        if rows
            .iter()
            .any(|r: &OpRow| r.op == row.op && r.form == row.form && r.variant == row.variant)
        {
            return Err(TableError::Duplicate { line: index + 1 });
        }
        rows.push(row);
    }
    Ok(rows)
}
pub fn gfx12() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> =
        LazyLock::new(|| parse_gfx12().expect("shipped gfx12 table must parse"));
    &TABLE
}
pub fn gfx1100() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        parse(include_str!("../isa/gfx1100.tbl")).expect("shipped gfx1100 table must parse")
    });
    &TABLE
}
pub fn gfx1151() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        let mut rows = gfx1100().to_vec();
        for delta in
            parse(include_str!("../isa/gfx1151.tbl")).expect("shipped gfx1151 deltas must parse")
        {
            if let Some(old) = rows
                .iter_mut()
                .find(|row| row.op == delta.op && row.form == delta.form)
            {
                *old = delta;
            } else {
                rows.push(delta);
            }
        }
        rows
    });
    &TABLE
}
pub fn table(arch: Arch) -> &'static [OpRow] {
    match arch {
        Arch::Gfx1100 => gfx1100(),
        Arch::Gfx1151 => gfx1151(),
        Arch::Gfx1200 | Arch::Gfx1201 => gfx12(),
        Arch::Gfx1010 | Arch::Gfx1030 => &[],
    }
}
/// Complete variant-keyed XML rows, separate from the frozen census regression set.
pub fn runtime_table(arch: Arch) -> &'static [OpRow] {
    static GFX1100: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        parse(include_str!("../isa/gfx1100.runtime.tbl")).expect("gfx1100 runtime table")
    });
    static GFX1151: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        parse(include_str!("../isa/gfx1151.runtime.tbl")).expect("gfx1151 runtime table")
    });
    static GFX12: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        parse(include_str!("../isa/gfx12.runtime.tbl")).expect("gfx12 runtime table")
    });
    match arch {
        Arch::Gfx1100 => &GFX1100,
        Arch::Gfx1151 => &GFX1151,
        Arch::Gfx1200 | Arch::Gfx1201 => &GFX12,
        _ => &[],
    }
}
fn variant_index(arch: Arch) -> Option<&'static std::collections::HashMap<EncodingVariant, usize>> {
    fn build(arch: Arch) -> std::collections::HashMap<EncodingVariant, usize> {
        let mut index = std::collections::HashMap::new();
        for (i, row) in runtime_table(arch).iter().enumerate() {
            assert!(
                index.insert(row.variant.expect("XML key"), i).is_none(),
                "duplicate XML encoding identity"
            );
        }
        index
    }
    static GFX1100: LazyLock<std::collections::HashMap<EncodingVariant, usize>> =
        LazyLock::new(|| build(Arch::Gfx1100));
    static GFX1151: LazyLock<std::collections::HashMap<EncodingVariant, usize>> =
        LazyLock::new(|| build(Arch::Gfx1151));
    static GFX12: LazyLock<std::collections::HashMap<EncodingVariant, usize>> =
        LazyLock::new(|| build(Arch::Gfx1201));
    match arch {
        Arch::Gfx1100 => Some(&GFX1100),
        Arch::Gfx1151 => Some(&GFX1151),
        Arch::Gfx1200 | Arch::Gfx1201 => Some(&GFX12),
        _ => None,
    }
}
pub fn lookup_xml(arch: Arch, key: EncodingVariant) -> Option<&'static OpRow> {
    runtime_table(arch).get(*variant_index(arch)?.get(&key)?)
}
/// Look up an explicitly selected encoding. `None` selects a census row.
pub fn lookup_variant(
    arch: Arch,
    opcode: Opcode,
    form: Form,
    variant: Option<EncodingVariant>,
) -> Option<&'static OpRow> {
    if let Some(key) = variant {
        return lookup_xml(arch, key).filter(|row| row.op == opcode && row.form == form);
    }
    let rows = table(arch);
    let mut matches = rows
        .iter()
        .filter(|row| row.op == opcode && row.form == form && row.variant == variant);
    let row = matches.next()?;
    matches.next().is_none().then_some(row)
}
pub fn lookup(arch: Arch, opcode: Opcode, form: Form) -> Option<&'static OpRow> {
    let mut matches = table(arch)
        .iter()
        .chain(runtime_table(arch).iter())
        .filter(|row| row.op == opcode && row.form == form);
    let row = matches.next()?;
    matches.next().is_none().then_some(row)
}
/// Select the exact encoding named by an instruction; never choose the first variant.
pub fn lookup_inst(arch: Arch, inst: &crate::inst::Inst) -> Result<&'static OpRow, String> {
    lookup_variant(arch, inst.op, inst.form, inst.variant).ok_or_else(|| {
        let name = table(arch)
            .iter()
            .chain(runtime_table(arch))
            .find(|row| row.op == inst.op && row.form == inst.form)
            .map_or("unknown instruction", |row| row.name);
        format!(
            "{name}: unavailable or ambiguous encoding {:?}",
            inst.variant
        )
    })
}
pub fn lookup_opcode(arch: Arch, opcode: Opcode) -> Option<&'static OpRow> {
    table(arch).iter().find(|row| row.op == opcode)
}

#[cfg(test)]
mod variant_tests {
    use super::*;

    #[test]
    fn full_key_preserves_conditional_variants() {
        let rows = parse(concat!(
            "s_mov_b32\tsop1\t0\tSDST:32,SSRC0:32\tSDST\tSSRC0\t-\t-\ts_mov_b32 s0, s0\tbe800000\t-\tENC_SOP1\tno_literal\t0\n",
            "s_mov_b32\tsop1\t0\tSDST:32,LITERAL:32\tSDST\tLITERAL\t-\t-\ts_mov_b32 s0, 0x41234567\tbe8000ff 41234567\t-\tSOP1_INST_LITERAL\tliteral\t1\n",
        )).unwrap();
        assert_eq!(rows.len(), 2);
        assert_ne!(rows[0].variant, rows[1].variant);
    }

    #[test]
    fn instruction_lookup_requires_the_exact_variant() {
        let arch = Arch::Gfx1100;
        let row = runtime_table(arch)
            .iter()
            .find(|row| row.name == "s_mov_b32" && row.variant.unwrap().encoding == "ENC_SOP1")
            .unwrap();
        let mut inst = crate::codec::gfx12::decode_for(arch, &[0xbe80_0002])
            .unwrap()
            .0;
        assert!(lookup(arch, row.op, row.form).is_none());
        inst.variant = row.variant;
        assert_eq!(lookup_inst(arch, &inst).unwrap().variant, row.variant);
        inst.variant.as_mut().unwrap().ordinal = u16::MAX;
        let error = lookup_inst(arch, &inst).unwrap_err();
        assert!(error.contains("s_mov_b32"), "{error}");
    }

    #[test]
    fn opcode_form_lookup_accepts_only_unambiguous_rows() {
        let arch = Arch::Gfx1201;
        let rows: Vec<_> = table(arch).iter().chain(runtime_table(arch)).collect();
        let unique = rows
            .iter()
            .find(|candidate| {
                rows.iter()
                    .filter(|row| row.op == candidate.op && row.form == candidate.form)
                    .count()
                    == 1
            })
            .expect("the complete XML contains unique opcode/form keys");
        assert_eq!(
            lookup(arch, unique.op, unique.form).unwrap().variant,
            unique.variant
        );
    }
}
