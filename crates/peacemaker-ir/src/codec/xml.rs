// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! XML-layout compute codec. Every encoded bit belongs to a named ISA field;
//! instruction provenance and oracle specimen bytes are never consulted.
use super::gfx12::{encoded_operand, operand, DecodeError};
use crate::{
    inst::{Arch, Form, FormFields, Inst, NamedField},
    isa::{
        self,
        xml::{self, Catalog, Encoding, Row, Selection},
        OpRow,
    },
    operand::{ImmField, Modifiers, Omod, Operand},
    provenance::Provenance,
    reg::{Kind, RegRef},
};
use smallvec::SmallVec;

fn error(reason: impl ToString) -> DecodeError {
    DecodeError::Rejected {
        offset: 0,
        reason: reason.to_string(),
    }
}
fn runtime_row(arch: Arch, row: &Row) -> Result<&'static OpRow, DecodeError> {
    isa::lookup_xml(
        arch,
        isa::EncodingVariant {
            instruction: row.name,
            encoding: row.encoding,
            condition: row.condition,
            ordinal: row
                .variant
                .try_into()
                .map_err(|_| error("XML encoding ordinal exceeds u16"))?,
        },
    )
    .filter(|r| r.op.id == row.opcode as u16)
    .ok_or_else(|| error(format!("{}: missing full encoding key", row.name)))
}
fn metadata(
    arch: Arch,
    row: &OpRow,
) -> Result<(&'static Catalog, &'static Row, &'static Encoding), DecodeError> {
    let cat = xml::catalog(arch).ok_or_else(|| error("target has no XML catalog"))?;
    let key = row
        .variant
        .ok_or_else(|| error(format!("{}: no XML variant key", row.name)))?;
    let source = cat
        .row_by_key(
            row.name,
            key.encoding,
            key.condition,
            row.op.id.into(),
            key.ordinal.into(),
        )
        .ok_or_else(|| error(format!("{}: XML variant is unavailable", row.name)))?;
    Ok((cat, source, cat.encoding_of(source)))
}
fn field(enc: &Encoding, word: u128, name: &str) -> u32 {
    enc.field(name).map_or(0, |f| f.extract(word) as u32)
}

pub(crate) fn field_effects(
    arch: Arch,
    row: &OpRow,
    fields: &FormFields,
    operands: &[Operand],
    mods: &Modifiers,
) -> Result<crate::effects::Effects, DecodeError> {
    let (cat, source, base) = metadata(arch, row)?;
    let (enc, word) = packet_fields(cat, base, fields)?;
    let words = [
        word as u32,
        (word >> 32) as u32,
        (word >> 64) as u32,
        (word >> 96) as u32,
    ];
    let selection = cat
        .select(&words[..enc.bits as usize / 32])
        .map_err(error)?;
    let (x, y) = match selection {
        Selection::Single(x) => (x, None),
        Selection::Vopd(pair) => (pair.x, Some(pair.y)),
    };
    if x.key() != source.key() {
        return Err(error(format!(
            "{}: named fields select a different XML row",
            row.name
        )));
    }
    let split = explicit(x, enc, word).count();
    if operands.len() != split + y.map_or(0, |y| explicit(y, enc, word).count()) {
        return Err(error(format!(
            "{}: XML operand count differs from fields",
            row.name
        )));
    }
    let mut effects = crate::effects::Effects::from_row(arch, row, &operands[..split], &mods.cpol)
        .map_err(error)?;
    if let Some(y) = y {
        let ys = crate::effects::Effects::from_row(
            arch,
            runtime_row(arch, y)?,
            &operands[split..],
            &mods.cpol,
        )
        .map_err(error)?;
        effects.defs.extend(ys.defs);
        effects.uses.extend(ys.uses);
        effects.implicit.reads |= ys.implicit.reads;
        effects.implicit.writes |= ys.implicit.writes;
    }
    for source in std::iter::once(x).chain(y) {
        for spec in explicit(source, enc, word) {
            match field(enc, word, spec.field_name.unwrap()) {
                251 => effects.implicit.reads |= crate::effects::ImplicitSet::VCC,
                252 => effects.implicit.reads |= crate::effects::ImplicitSet::EXEC,
                _ => {}
            }
        }
    }
    Ok(effects)
}

pub(crate) fn validate_packet_fields(
    arch: Arch,
    row: &OpRow,
    fields: &FormFields,
) -> Result<(), DecodeError> {
    let (cat, _, base) = metadata(arch, row)?;
    let (enc, word) = packet_fields(cat, base, fields)?;
    supported_selectors(row.name, enc, word)
}

fn packet_fields<'a>(
    cat: &'a Catalog,
    base: &'a Encoding,
    fields: &FormFields,
) -> Result<(&'a Encoding, u128), DecodeError> {
    let FormFields::Bits { ignored, honored } = fields else {
        return Err(error("XML codec requires named encoding fields"));
    };
    if !ignored.is_empty() {
        return Err(error("XML fields may not be silently ignored"));
    }
    let enc = cat
        .encodings()
        .iter()
        .find(|enc| {
            (enc.name == base.name || (enc.is_pair() && base.is_pair()))
                && enc.fields.len() == honored.len()
                && base.fields.iter().all(|f| enc.field(f.name) == Some(f))
                && enc
                    .fields
                    .iter()
                    .all(|f| honored.iter().any(|v| v.name == f.name))
        })
        .ok_or_else(|| error("no compatible XML packet layout for named fields"))?;
    let mut word = 0;
    for f in honored {
        word = enc
            .field(f.name)
            .unwrap()
            .insert(word, f.value.into())
            .map_err(error)?;
    }
    Ok((enc, word))
}

fn supported_selectors(name: &str, enc: &Encoding, word: u128) -> Result<(), DecodeError> {
    if enc.field("LANE_SEL_0").is_some() && field(enc, word, "SRC0") == 234 {
        return Err(error(format!(
            "{name}: DPP8FI selector 0xea has no LLVM-verified XML evidence"
        )));
    }
    Ok(())
}
fn explicit<'a>(
    row: &'a Row,
    enc: &'a Encoding,
    word: u128,
) -> impl Iterator<Item = &'a xml::Operand> {
    let atomic = row.memory.iter().any(|o| o.input) && row.memory.iter().any(|o| o.output);
    let returns = if enc.field("TH").is_some() {
        field(enc, word, "TH") & 1 != 0
    } else {
        field(enc, word, "GLC") != 0
    };
    let optional_destination = atomic && !matches!(row.encoding, "ENC_DS" | "ENC_VDS");
    row.operands.iter().filter(move |o| {
        !o.implicit
            && o.field_name.is_some_and(|n| enc.field(n).is_some())
            && (!optional_destination || !o.output || returns)
    })
}
fn decode_operand(
    arch: Arch,
    source: &Row,
    row: &OpRow,
    enc: &Encoding,
    word: u128,
    spec: &xml::Operand,
    words: &[u32],
) -> Result<Operand, DecodeError> {
    let name = spec.field_name.expect("explicit field");
    let code = field(enc, word, name);
    let bits = spec.size as u16;
    if name == "LITERAL" {
        return Ok(Operand::Literal(code));
    }
    if spec.operand_type == "OPR_VGPR" {
        if name == "VDSTY" {
            let base = code * 2 + ((field(enc, word, "VDSTX") & 1) ^ 1);
            return Ok(Operand::Reg(RegRef {
                kind: Kind::V,
                base: base as u16,
                len: ((spec.size + 31) / 32).max(1) as u8,
            }));
        }
        // DPP moves the first VGPR source into VSRC0. Its true16 e32
        // register/half packing is the same as the ordinary VSRC1 field.
        let selector = if name == "VSRC0"
            && bits == 16
            && matches!(row.form, Form::Vop1 | Form::Vop2 | Form::Vopc)
        {
            "VSRC1"
        } else {
            name
        };
        return operand(arch, selector, bits, code, None, row, words);
    }
    if name == "SRSRC" {
        return Ok(Operand::Reg(RegRef {
            kind: Kind::S,
            base: (code * 4) as u16,
            len: ((spec.size + 31) / 32).max(1) as u8,
        }));
    }
    if matches!(
        spec.operand_type,
        "OPR_SRC"
            | "OPR_SRC_VGPR"
            | "OPR_SRC_VGPR_OR_INLINE"
            | "OPR_SSRC"
            | "OPR_SSRC_LANESEL"
            | "OPR_SREG_M0_INL"
    ) {
        if (108..=123).contains(&code) {
            return Ok(Operand::Reg(RegRef {
                kind: Kind::Ttmp,
                base: (code - 108) as u16,
                len: ((spec.size + 31) / 32).max(1) as u8,
            }));
        }
        // These special selectors have architectural flag/aperture meanings, not
        // integer-immediate meanings. They are retained as typed selector values;
        // XML variants are refused by the numeric emulator until qualified.
        if matches!(code, 239 | 251 | 252 | 254) {
            return Ok(Operand::Imm(ImmField::Unsigned(code)));
        }
        let literal = enc.field("LITERAL").map(|f| f.extract(word) as u32);
        return operand(arch, name, bits, code, literal, row, words);
    }
    if matches!(
        spec.operand_type,
        "OPR_SDST"
            | "OPR_SREG"
            | "OPR_SREG_M0"
            | "OPR_SDST_EXEC"
            | "OPR_SDST_M0"
            | "OPR_SDST_NULL"
            | "OPR_SMEM_OFFSET"
            | "OPR_SMEM_OFFSET_NOK"
            | "OPR_SREG_LITERAL"
    ) {
        let canonical = if name == "SRSRC" { "RSRC" } else { name };
        return operand(arch, canonical, bits, code, None, row, words);
    }
    if matches!(spec.operand_type, "OPR_LABEL" | "OPR_SIMM16") {
        return Ok(Operand::Imm(if row.form == Form::Sopk {
            ImmField::Sopk(code as i16)
        } else {
            ImmField::Sopp(code as i16)
        }));
    }
    let _ = source;
    Ok(Operand::Imm(ImmField::Unsigned(code)))
}
fn encode_operand(
    arch: Arch,
    row: &OpRow,
    spec: &xml::Operand,
    value: &Operand,
    enc: &Encoding,
    word: u128,
) -> Result<u32, DecodeError> {
    let name = spec.field_name.expect("explicit field");
    if name == "LITERAL" {
        return match value {
            Operand::Literal(n) => Ok(*n),
            _ => Err(error(format!(
                "{}: LITERAL requires literal bits",
                row.name
            ))),
        };
    }
    if let Operand::Imm(ImmField::Unsigned(n)) = value {
        return Ok(*n);
    }
    if let Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) = value {
        return Ok(u32::from(*n as u16));
    }
    if let Operand::Reg(r) | Operand::Half(r, _) = value {
        if spec.operand_type == "OPR_VGPR" {
            if r.kind != Kind::V {
                return Err(error(format!("{}: {name} requires a VGPR", row.name)));
            }
            if name == "VDSTY" {
                if (u32::from(r.base) ^ field(enc, word, "VDSTX")) & 1 == 0 {
                    return Err(error(format!(
                        "{}: VOPD destinations require opposite parity",
                        row.name
                    )));
                }
                return Ok(u32::from(r.base / 2));
            }
            let selector = if name == "VSRC0"
                && spec.size == 16
                && matches!(row.form, Form::Vop1 | Form::Vop2 | Form::Vopc)
            {
                "VSRC1"
            } else {
                name
            };
            return encoded_operand(arch, selector, spec.size as u16, value, row.form);
        }
        if name == "SRSRC" || name == "SBASE" {
            let scale = if name == "SBASE" { 2 } else { 4 };
            if r.kind != Kind::S || r.base % scale != 0 {
                return Err(error(format!(
                    "{}: {name} is not an aligned SGPR base",
                    row.name
                )));
            }
            return Ok(u32::from(r.base / scale));
        }
    }
    encoded_operand(arch, name, spec.size as u16, value, row.form)
}
fn modifiers(enc: &Encoding, word: u128) -> Modifiers {
    let mut m = Modifiers::default();
    m.neg = field(enc, word, "NEG") as u8;
    m.abs = field(enc, word, "ABS") as u8;
    m.clamp = field(enc, word, "CLAMP") != 0;
    m.op_sel = field(enc, word, "OPSEL") as u8;
    m.op_sel_hi = field(enc, word, "OPSEL_HI") as u8;
    m.neg_hi = field(enc, word, "NEG_HI") as u8;
    m.omod = match field(enc, word, "OMOD") {
        1 => Omod::Mul2,
        2 => Omod::Mul4,
        3 => Omod::Div2,
        _ => Omod::None,
    };
    m.cpol.th = field(enc, word, "TH") as u8;
    m.cpol.scope = field(enc, word, "SCOPE") as u8;
    m.cpol.nv = field(enc, word, "NV") != 0;
    m.cpol.glc = field(enc, word, "GLC") != 0;
    m.cpol.slc = field(enc, word, "SLC") != 0;
    m.cpol.dlc = field(enc, word, "DLC") != 0;
    m.ds.gds = field(enc, word, "GDS") != 0;
    if enc.field("DPP_CTRL").is_some() {
        m.dpp = Some(crate::operand::Dpp {
            ctrl: field(enc, word, "DPP_CTRL") as u16,
            row_mask: field(enc, word, "ROW_MASK") as u8,
            bank_mask: field(enc, word, "BANK_MASK") as u8,
            bound_ctrl: field(enc, word, "BOUND_CTRL") != 0,
        });
    }
    m
}
fn modifier_value(m: &Modifiers, name: &str) -> Option<u32> {
    Some(match name {
        "NEG" => m.neg.into(),
        "ABS" => m.abs.into(),
        "CLAMP" => m.clamp.into(),
        "OPSEL" => m.op_sel.into(),
        "OPSEL_HI" => m.op_sel_hi.into(),
        "NEG_HI" => m.neg_hi.into(),
        "OMOD" => match m.omod {
            Omod::None => 0,
            Omod::Mul2 => 1,
            Omod::Mul4 => 2,
            Omod::Div2 => 3,
        },
        "TH" => m.cpol.th.into(),
        "SCOPE" => m.cpol.scope.into(),
        "NV" => m.cpol.nv.into(),
        "GLC" => m.cpol.glc.into(),
        "SLC" => m.cpol.slc.into(),
        "DLC" => m.cpol.dlc.into(),
        "GDS" => m.ds.gds.into(),
        "DPP_CTRL" => m.dpp?.ctrl.into(),
        "ROW_MASK" => m.dpp?.row_mask.into(),
        "BANK_MASK" => m.dpp?.bank_mask.into(),
        "BOUND_CTRL" => m.dpp?.bound_ctrl.into(),
        _ => return None,
    })
}

pub fn decode(arch: Arch, words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    let cat = xml::catalog(arch).ok_or_else(|| error("target has no XML catalog"))?;
    let selection = cat.select(&words[..words.len().min(4)]).map_err(error)?;
    let (source, y, enc) = match selection {
        Selection::Single(row) => (row, None, cat.encoding_of(row)),
        Selection::Vopd(pair) => (pair.x, Some(pair.y), pair.encoding),
    };
    supported_selectors(source.name, enc, xml::pack_words(words))?;
    let count = (enc.bits / 32) as usize;
    if words.len() < count {
        return Err(error(format!(
            "{}: truncated {}",
            source.name, source.encoding
        )));
    }
    let words = &words[..count];
    let word = xml::pack_words(words);
    // Reserved bits are not an opaque escape hatch. A valid encoding must be
    // composed entirely of XML fields, including fixed identifier/opcode bits.
    let mut covered = 0u128;
    for f in &enc.fields {
        covered = f.insert(covered, (1u128 << f.width()) - 1).map_err(error)?;
    }
    if word & !covered != 0 {
        return Err(error(format!("{}: nonzero reserved XML bits", source.name)));
    }
    let row = runtime_row(arch, source)?;
    let mut operands = SmallVec::new();
    for spec in explicit(source, enc, word) {
        operands.push(decode_operand(arch, source, row, enc, word, spec, words)?);
    }
    if let Some(y) = y {
        let yrow = runtime_row(arch, y)?;
        for spec in explicit(y, enc, word) {
            operands.push(decode_operand(arch, y, yrow, enc, word, spec, words)?);
        }
    }
    let fields = FormFields::Bits {
        ignored: SmallVec::new(),
        honored: enc
            .fields
            .iter()
            .map(|field| NamedField {
                name: field.name,
                value: field.extract(word) as u32,
            })
            .collect(),
    };
    let mods = modifiers(enc, word);
    let literal = enc.field("LITERAL").map(|f| f.extract(word) as u32);
    let inst = Inst::from_row(
        arch,
        row,
        fields,
        operands,
        mods,
        literal,
        Provenance::default(),
    )
    .map_err(error)?;
    Ok((inst, count))
}

pub fn encode(arch: Arch, inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    let row = isa::lookup_inst(arch, inst).map_err(error)?;
    let (cat, source, base) = metadata(arch, row)?;
    let (enc, packet) = packet_fields(cat, base, &inst.fields)?;
    supported_selectors(row.name, enc, packet)?;
    let FormFields::Bits { ignored, honored } = &inst.fields else {
        return Err(error(format!(
            "{}: XML codec requires named encoding fields",
            row.name
        )));
    };
    if !ignored.is_empty() {
        return Err(error(format!(
            "{}: XML fields may not be silently ignored",
            row.name
        )));
    }
    let mut values: SmallVec<[(&str, u128); 32]> = honored
        .iter()
        .map(|v| {
            (
                v.name,
                u128::from(modifier_value(&inst.mods, v.name).unwrap_or(v.value)),
            )
        })
        .collect();
    let mut word = 0;
    for &(name, value) in &values {
        word = enc
            .field(name)
            .unwrap()
            .insert(word, value)
            .map_err(error)?;
    }
    let selector_words: SmallVec<[u32; 3]> = (0..enc.bits / 32)
        .map(|i| (word >> (i * 32)) as u32)
        .collect();
    let selection = cat.select(&selector_words).map_err(error)?;
    let y = match selection {
        Selection::Single(_) => None,
        Selection::Vopd(pair) => Some(pair.y),
    };
    let mut index = 0;
    let mut assigned: SmallVec<[(&str, u32); 12]> = SmallVec::new();
    for source in std::iter::once(source).chain(y) {
        let source_row = runtime_row(arch, source)?;
        for spec in explicit(source, enc, word) {
            let value = inst
                .operands
                .get(index)
                .ok_or_else(|| error(format!("{}: missing XML operand {index}", row.name)))?;
            let name = spec.field_name.unwrap();
            let encoded = encode_operand(arch, source_row, spec, value, enc, word)?;
            if let Some((_, previous)) = assigned.iter().find(|(field, _)| *field == name) {
                if *previous != encoded {
                    return Err(error(format!(
                        "{}: shared field {name} has conflicting operand values",
                        row.name
                    )));
                }
            } else {
                assigned.push((name, encoded));
            }
            let slot = values
                .iter_mut()
                .find(|v| v.0 == name)
                .ok_or_else(|| error(format!("{}: missing named field {name}", row.name)))?;
            slot.1 = encoded.into();
            word = enc
                .field(name)
                .unwrap()
                .insert(word, encoded.into())
                .map_err(error)?;
            if let Operand::Imm(ImmField::SmemOffset(offset)) = value {
                if name == "SOFFSET" {
                    let displacement = enc
                        .field("IOFFSET")
                        .or_else(|| enc.field("OFFSET"))
                        .ok_or_else(|| {
                            error(format!("{}: SMEM displacement field absent", row.name))
                        })?;
                    let mask = (1u128 << displacement.width()) - 1;
                    let bits = (*offset as u32 as u128) & mask;
                    let displacement_name = if enc.field("IOFFSET").is_some() {
                        "IOFFSET"
                    } else {
                        "OFFSET"
                    };
                    let slot = values
                        .iter_mut()
                        .find(|v| v.0 == displacement_name)
                        .ok_or_else(|| {
                            error(format!("{}: missing SMEM displacement value", row.name))
                        })?;
                    slot.1 = bits;
                    word = displacement.insert(word, bits).map_err(error)?;
                }
            }
            index += 1;
        }
    }
    if index != inst.operands.len() {
        return Err(error(format!("{}: extra XML operands", row.name)));
    }
    // Validate each half in its own XML layout; the physical packet can be
    // longer than either half's declared layout when its peer needs a literal.
    for source in std::iter::once(source).chain(y) {
        let half = cat.encoding_of(source);
        let fields: SmallVec<[(&str, u128); 32]> = values
            .iter()
            .copied()
            .filter(|(name, _)| half.field(name).is_some())
            .collect();
        cat.encode_row(source, &fields).map_err(error)?;
    }
    let mut word = 0;
    for &(name, value) in &values {
        word = enc
            .field(name)
            .unwrap()
            .insert(word, value)
            .map_err(error)?;
    }
    Ok((0..enc.bits / 32)
        .map(|i| (word >> (32 * i)) as u32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(hex: &str) -> Vec<u32> {
        let bytes: Vec<u8> = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn every_llvm_verified_compute_specimen_reencodes_from_fields() {
        let mut failures = Vec::new();
        for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
            let cat = xml::catalog(arch).unwrap();
            for row in cat.rows() {
                if row.oracle.status != "verified"
                    || row.status == xml::CodecStatus::UnsupportedGraphics
                {
                    continue;
                }
                let data = words(row.oracle.candidate.expect("verified specimen"));
                let result = decode(arch, &data).and_then(|(mut inst, count)| {
                    if count != data.len() {
                        return Err(error("specimen width differs from XML"));
                    }
                    inst.prov = Provenance::default();
                    let rebuilt = Inst::from_encoding(
                        arch,
                        inst.op,
                        inst.form,
                        inst.variant,
                        inst.fields.clone(),
                        inst.operands.clone(),
                        inst.mods.clone(),
                        inst.literal,
                        Provenance::default(),
                    )
                    .map_err(error)?;
                    assert_eq!(rebuilt.effects, inst.effects, "{arch:?} {}", row.name);
                    assert_eq!(encode(arch, &rebuilt)?, encode(arch, &inst)?);
                    if let Selection::Vopd(pair) = cat.select(&data).map_err(error)? {
                        let split = explicit(pair.x, pair.encoding, xml::pack_words(&data)).count();
                        let x = crate::effects::Effects::from_row(
                            arch,
                            runtime_row(arch, pair.x)?,
                            &inst.operands[..split],
                            &inst.mods.cpol,
                        )
                        .map_err(error)?;
                        let y = crate::effects::Effects::from_row(
                            arch,
                            runtime_row(arch, pair.y)?,
                            &inst.operands[split..],
                            &inst.mods.cpol,
                        )
                        .map_err(error)?;
                        assert_eq!(inst.effects.defs.len(), x.defs.len() + y.defs.len());
                        assert!(x
                            .defs
                            .iter()
                            .chain(&y.defs)
                            .all(|r| inst.effects.defs.contains(r)));
                        assert!(x
                            .uses
                            .iter()
                            .chain(&y.uses)
                            .all(|r| inst.effects.uses.contains(r)));
                    }
                    let output = encode(arch, &inst)?;
                    if output.as_slice() != data {
                        return Err(error(format!(
                            "field reencode {output:08x?} != {data:08x?}"
                        )));
                    }
                    Ok(())
                });
                if let Err(error) = result {
                    failures.push(format!(
                        "{arch:?} {} {} {} #{}: {error}",
                        row.name, row.encoding, row.condition, row.variant
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn true16_dpp_footprints_match_llvm_half_register_oracles() {
        use crate::operand::Half;
        // llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=<each arch> -show-encoding:
        // v_mov_b16 v3.h, v5.h dpp8:[0,1,2,3,4,5,6,7]
        // v_mov_b16 v3.h, v5.h quad_perm:[0,1,2,3]
        for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
            for extension in [0xfac6_8885, 0xff00_e485] {
                let data = [
                    if extension == 0xfac6_8885 {
                        0x7f06_38e9
                    } else {
                        0x7f06_38fa
                    },
                    extension,
                ];
                let (inst, count) = decode(arch, &data).unwrap();
                assert_eq!(count, 2);
                let dst = RegRef {
                    kind: Kind::V,
                    base: 3,
                    len: 1,
                };
                let src = RegRef {
                    kind: Kind::V,
                    base: 5,
                    len: 1,
                };
                assert_eq!(
                    inst.operands.as_slice(),
                    [Operand::Half(dst, Half::Hi), Operand::Half(src, Half::Hi)]
                );
                assert_eq!(inst.effects.defs.as_slice(), [dst]);
                assert!(inst.effects.uses.contains(&src));
                assert!(
                    inst.effects.uses.contains(&dst),
                    "partial write preserves the other half"
                );
                assert_eq!(encode(arch, &inst).unwrap().as_slice(), data);
            }
        }
    }

    #[test]
    fn literal_bits_are_not_numeric_conversions_or_provenance() {
        let arch = Arch::Gfx1201;
        let edges = [
            0,
            1,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0x7f80_0001,
            0xffff_ffff,
            0x4123_4567,
        ];
        let random = (0..64).scan(0x7a31_c695u32, |state, _| {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            Some(*state)
        });
        for bits in edges.into_iter().chain(random) {
            let data = [0xbe80_00ff, bits];
            let (mut inst, count) = decode(arch, &data).unwrap();
            assert_eq!(count, 2);
            inst.prov.bytes = None;
            assert_eq!(encode(arch, &inst).unwrap().as_slice(), data);
        }
    }

    #[test]
    fn setpc_proven_register_domain_is_not_an_inline_or_literal_domain() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
            let (mut inst, _) = super::super::gfx12::decode_for(arch, &[0xbe80_4802]).unwrap();
            assert_eq!(
                super::super::gfx12::encode_for(arch, &inst)
                    .unwrap()
                    .as_slice(),
                [0xbe80_4802]
            );
            for code in [128u32, 255] {
                let mut data = vec![0xbe80_4800 | code];
                if code == 255 {
                    data.push(0x4123_4567);
                }
                let reason = super::super::gfx12::decode_for(arch, &data)
                    .unwrap_err()
                    .to_string();
                assert!(reason.contains("s_setpc_b64"), "{arch:?}: {reason}");
                assert!(decode(arch, &data)
                    .unwrap_err()
                    .to_string()
                    .contains("s_setpc_b64"));
            }
            inst.operands[0] = Operand::Inline(crate::operand::InlineConst::Integer(0));
            let reason = super::super::gfx12::encode_for(arch, &inst)
                .unwrap_err()
                .to_string();
            assert!(reason.contains("s_setpc_b64"), "{arch:?}: {reason}");
        }
    }
}
