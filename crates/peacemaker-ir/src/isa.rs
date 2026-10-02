//! Per-target opcode/encoding rows. Tables are backed by AMD XML and pinned LLVM samples.
use std::sync::LazyLock;
use crate::inst::{Arch, Family, Form, Opcode, VmemForm};
use crate::operand::CachePolicy;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldClass { Ignored, Honored }
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldRule { pub name: &'static str, pub class: FieldClass, pub mask: u32, pub allowed: Vec<u32> }
#[derive(Clone, Debug)]
pub struct OpRow {
    pub op: Opcode, pub form: Form, pub name: &'static str,
    pub grammar: &'static str, pub defs: &'static str, pub uses: &'static str,
    pub implicit: &'static str, pub counter: &'static str,
    pub sample: &'static str, pub encoding: &'static str,
    pub benign_src2: Vec<u16>, pub fields: Vec<FieldRule>,
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
    pub fn counter_rules(&self, returns: bool) -> impl Iterator<Item = (&'static str, &'static str)> {
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
    if arch == Arch::Gfx1201 { cpol.th & 1 != 0 } else { cpol.glc }
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
        "sop1" => Form::Sop1, "sop2" => Form::Sop2, "sopc" => Form::Sopc, "sopk" => Form::Sopk,
        "sopp" => Form::Sopp, "smem" => Form::Smem, "vop1" => Form::Vop1,
        "vop1_dpp" => Form::Vop1Dpp, "vop2" => Form::Vop2, "vop2_dpp" => Form::Vop2Dpp, "vopc" => Form::Vopc, "vop3" => Form::Vop3,
        "vop3p" => Form::Vop3p, "vopd" => Form::Vopd, "vinterp" => Form::Vinterp,
        "ds" => Form::Ds, "global" => Form::Vmem(VmemForm::Global),
        "scratch" => Form::Vmem(VmemForm::Scratch), "flat" => Form::Vmem(VmemForm::Flat),
        "buffer" => Form::Vmem(VmemForm::Buffer), "image" => Form::Vmem(VmemForm::Image),
        "export" => Form::Export, _ => return None,
    })
}
fn family(form: Form) -> Family {
    match form {
        Form::Sop1 => Family::Sop1, Form::Sop2 => Family::Sop2, Form::Sopc => Family::Sopc,
        Form::Sopk => Family::Sopk, Form::Sopp => Family::Sopp, Form::Smem => Family::Smem,
        Form::Vop1 | Form::Vop1Dpp => Family::Vop1, Form::Vop2 | Form::Vop2Dpp => Family::Vop2, Form::Vopc => Family::Vopc,
        Form::Vop3 => Family::Vop3, Form::Vop3p => Family::Vop3p,
        Form::Vopd => Family::Vopd, Form::Vinterp => Family::Vinterp, Form::Ds => Family::Ds,
        Form::Vmem(_) => Family::Vmem, Form::Export => Family::Export,
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
        if line.is_empty() || line.starts_with('#') { continue; }
        let fields: Vec<_> = line.split('\t').collect();
        let fail = |reason: &str| TableError::Malformed { line: index + 1, reason: reason.into() };
        if fields.len() != 11 { return Err(fail("expected 11 tab-delimited columns")); }
        let kind = form(fields[1]).ok_or_else(|| fail("unknown instruction form"))?;
        let id = u16::from_str_radix(fields[2].trim_start_matches("0x"), 16).map_err(|_| fail("bad opcode id"))?;
        let mut rule_fields = Vec::new();
        let mut benign_src2 = Vec::new();
        for field in fields[10].split(',').filter(|s| !s.is_empty() && *s != "-") {
            let parts: Vec<_> = field.split(':').collect();
            if parts.len() < 3 { return Err(fail("malformed field rule")); }
            let class = match parts[1] { "ignored" => FieldClass::Ignored, "honored" => FieldClass::Honored, _ => return Err(fail("unknown field class")) };
            let mask = u32::from_str_radix(parts[2].trim_start_matches("0x"), 16).map_err(|_| fail("bad field mask"))?;
            let allowed = parts.get(3).unwrap_or(&"").split('/').filter(|s| !s.is_empty())
                .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|_| fail("bad benign field value")))
                .collect::<Result<Vec<_>, _>>()?;
            if parts[0] == "src2_unused" { benign_src2 = allowed.iter().map(|&v| v as u16).collect(); }
            rule_fields.push(FieldRule { name: parts[0], class, mask, allowed });
        }
        let tagged_slot = fields[3].split(',').filter(|part| part.contains('@') && *part != "literal@last")
            .map(|part| part.split(':').next().unwrap_or("").ends_with("@rtn")).reduce(|a, b| a && b);
        let tagged_counter = fields[7].split(',').filter_map(|pair| pair.split_once('@'))
            .map(|(_, tag)| matches!(tag, "rtn" | "nortn")).reduce(|a, b| a && b);
        if tagged_slot == Some(false) || tagged_counter == Some(false) { return Err(fail("unknown return-form tag")); }
        if (tagged_slot.is_some() || tagged_counter.is_some()) && !matches!(kind, Form::Vmem(_)) {
            return Err(fail("return-form tags apply to VMEM atomics only"));
        }
        let row = OpRow { op: Opcode { family: family(kind), id }, form: kind, name: fields[0],
            grammar: fields[3], defs: fields[4], uses: fields[5], implicit: fields[6], counter: fields[7],
            sample: fields[8], encoding: fields[9], benign_src2, fields: rule_fields };
        if rows.iter().any(|r: &OpRow| r.op == row.op && r.form == row.form) { return Err(TableError::Duplicate { line: index + 1 }); }
        rows.push(row);
    }
    Ok(rows)
}
pub fn gfx12() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| parse_gfx12().expect("shipped gfx12 table must parse"));
    &TABLE
}
pub fn gfx1100() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| parse(include_str!("../isa/gfx1100.tbl")).expect("shipped gfx1100 table must parse"));
    &TABLE
}
pub fn gfx1151() -> &'static [OpRow] {
    static TABLE: LazyLock<Vec<OpRow>> = LazyLock::new(|| {
        let mut rows = gfx1100().to_vec();
        for delta in parse(include_str!("../isa/gfx1151.tbl")).expect("shipped gfx1151 deltas must parse") {
            if let Some(old) = rows.iter_mut().find(|row| row.op == delta.op && row.form == delta.form) {
                *old = delta;
            } else { rows.push(delta); }
        }
        rows
    });
    &TABLE
}
pub fn table(arch: Arch) -> &'static [OpRow] {
    match arch {
        Arch::Gfx1100 => gfx1100(), Arch::Gfx1151 => gfx1151(),
        Arch::Gfx1201 => gfx12(), Arch::Gfx1010 | Arch::Gfx1030 => &[],
    }
}
pub fn lookup(arch: Arch, opcode: Opcode, form: Form) -> Option<&'static OpRow> {
    table(arch).iter().find(|row| row.op == opcode && row.form == form)
}
pub fn lookup_opcode(arch: Arch, opcode: Opcode) -> Option<&'static OpRow> {
    table(arch).iter().find(|row| row.op == opcode)
}
