// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C3b: canonical objdump spelling for typed instructions.
//!
//! [`canonical`] renders the exact text the pinned `llvm-objdump` prints for
//! an instruction: same mnemonic, same operand spellings, same modifier
//! suffixes. Every rule below was reverse-engineered from the pinned tools
//! (probed where KT48 gives no example) and is pinned by `t3_*` over all
//! 1,696 KT48 instructions:
//!
//! * SOPK immediates print hex (`0x6180`); SOPC immediates follow their
//!   operand (inline decimal, literal hex); SMEM offsets print hex
//!   (`0x30`, `-0x4`); waits/clause print hex (`0x0`); branches print the
//!   unsigned 16-bit value (`65525` for -11); `s_barrier_signal` prints
//!   signed decimal (`-1`); `s_barrier_wait` prints decimal up to 64, hex
//!   above (`0xffff`).
//! * `s_delay_alu` prints symbolic fields (`instid0(SALU_CYCLE_1) | …`,
//!   zero fields omitted, all-zero prints `0`); `s_wait_alu` prints the
//!   non-max depctr counters in fixed order (`depctr_sa_sdst(0)`; exact
//!   `0xffff` prints `0xffff`, reserved bits set print hex, and the
//!   all-max `0xff9f` prints the full list).
//! * VOP3 `neg` prints as a `-` prefix on registers and as `neg(…)` on
//!   constants, `abs` wraps in `|…|`, `omod` prints `mul:2/4`/`div:2`,
//!   `clamp` prints `clamp`; `op_sel` prints one entry per source plus the
//!   destination bit (`op_sel:[1,0]`) for rows with 16-bit operands only.
//! * VOP3 scalar pairs print single (`s2`, never `s[2:3]`), except the two
//!   `v_mad_co_*` rows whose wide source prints as a range; `v_cmpx*`
//!   omits its implicit exec destination; VOP2 `v_cndmask_b32_e32` appends
//!   `, vcc_lo`; VOPC compares print their `vcc_lo` destination.
//! * Inline floats print LLVM-style (`1.0`, `0.5`, `0.15915494`); literals
//!   print hex (`0xff800000`).
//!
//! Anything the syntax cannot spell (don't-care fills, omitted modifiers)
//! is classified by the T9 harness, never normalised here.

use peacemaker_ir::{
    inst::{Arch, Form, FormFields, Inst, Opcode},
    isa::{self, OpRow},
    operand::{ImmField, Modifiers, Omod, Operand, Special},
    reg::Kind,
};

#[derive(Debug, thiserror::Error)]
pub enum PrintError {
    #[error("unknown opcode {op:?} / form {form:?}")]
    UnknownOpcode { op: Opcode, form: Form },
    #[error("cannot spell in objdump canonical text: {0}")]
    Unspellable(String),
}

/// Canonical objdump text for one instruction.
pub fn canonical(inst: &Inst, arch: Arch) -> Result<String, PrintError> {
    let row = isa::lookup(arch, inst.op, inst.form).ok_or(PrintError::UnknownOpcode {
        op: inst.op,
        form: inst.form,
    })?;
    if inst.form == Form::Sopp && row.name == "s_delay_alu" {
        return delay_text(&inst.mods);
    }
    if inst.form == Form::Sopp && row.name == "s_wait_alu" {
        return wait_alu_text(inst);
    }
    if inst.form == Form::Vopd {
        return vopd_text(inst, row);
    }
    let slots = grammar_slots(row);
    let skew = if inst.form == Form::Vopc && row.name.starts_with("v_cmp_") {
        1
    } else {
        0
    };
    // `v_cmpx*` VOP3 rows carry a hidden exec VDST (fixed encoding 0x7e)
    // that canonical text omits; the two visible operands map to the
    // SRC0/SRC1 slots.
    let cmpx_offset = usize::from(row.name.starts_with("v_cmpx") && inst.form == Form::Vop3);
    if cmpx_offset == 1 && inst.operands.len() + 1 != slots.len() {
        return Err(PrintError::Unspellable(format!(
            "{} needs two visible operands, have {}",
            row.name,
            inst.operands.len()
        )));
    }
    let mut parts: Vec<String> = Vec::new();
    // VOPC compares carry a synthetic vcc destination that objdump prints.
    if skew == 1 {
        let first = inst
            .operands
            .first()
            .ok_or_else(|| PrintError::Unspellable(format!("{} is missing vcc_lo", row.name)))?;
        if *first != Operand::Special(Special::VccLo) {
            return Err(PrintError::Unspellable(format!(
                "{} leads with {first:?}, not vcc_lo",
                row.name
            )));
        }
        parts.push(operand_text(first, None, row, inst)?);
    }
    for (slot_index, (slot_name, _)) in slots.iter().enumerate() {
        if slot_index < cmpx_offset {
            continue;
        }
        let op_index = slot_index + skew - cmpx_offset;
        let operand = inst.operands.get(op_index).ok_or_else(|| {
            PrintError::Unspellable(format!("{} is missing its {slot_name} operand", row.name))
        })?;
        let mut text = operand_text(operand, Some(slot_name), row, inst)?;
        if matches!(inst.form, Form::Vop3 | Form::Vop1Dpp | Form::Vop2Dpp) {
            text = apply_vop3_src_mods(&text, operand, slot_name, &inst.mods)?;
        }
        parts.push(text);
    }
    // `s_fmamk_f32 D, S0, K, S1` / `s_fmaak_f32 D, S0, S1, K`: the K constant is
    // the instruction's literal word, spelled as the fourth operand.
    if matches!(row.name, "s_fmamk_f32" | "s_fmaak_f32") {
        let literal = inst.literal.ok_or_else(|| {
            PrintError::Unspellable(format!("{} is missing its literal word", row.name))
        })?;
        let at = if row.name == "s_fmamk_f32" {
            2
        } else {
            parts.len()
        };
        parts.insert(at.min(parts.len()), format!("{literal:#x}"));
    }
    // Trailing operands the grammar does not declare (DS offsets, VMEM
    // offset/offen/scope) join with a space, exactly as objdump does.
    let mut tail: Vec<String> = Vec::new();
    for operand in inst.operands.iter().skip(slots.len() + skew - cmpx_offset) {
        tail.push(operand_text(operand, None, row, inst)?);
    }
    let mut text = row.name.to_owned();
    if inst.form == Form::Sopp && row.name == "s_endpgm" {
        return Ok(text);
    }
    let mut first = true;
    for part in parts.iter().chain(tail.iter()) {
        if first {
            text.push(' ');
            first = false;
        } else if part.starts_with("offset")
            || *part == "offen"
            || part.starts_with("scope:")
            || part.starts_with("th:")
        {
            text.push(' ');
        } else {
            text.push_str(", ");
        }
        text.push_str(part);
    }
    if inst.form == Form::Vop2 && row.name == "v_cndmask_b32_e32" {
        text.push_str(", vcc_lo");
    }
    text.push_str(&modifier_suffix(inst, row)?);
    Ok(text)
}

/// One canonical line per instruction, in order (assembler-parity input).
pub fn kernel_lines(insts: &[Inst], arch: Arch) -> Result<Vec<String>, PrintError> {
    insts.iter().map(|inst| canonical(inst, arch)).collect()
}

fn grammar_slots(row: &OpRow) -> Vec<(&str, u16)> {
    row.grammar
        .split(',')
        .filter_map(|part| {
            if part == "literal@last" {
                return None;
            }
            let (name, bits) = part.split_once(':')?;
            Some((name, bits.parse().ok()?))
        })
        .collect()
}

fn operand_text(
    operand: &Operand,
    _slot: Option<&str>,
    row: &OpRow,
    inst: &Inst,
) -> Result<String, PrintError> {
    match operand {
        Operand::Reg(r) => {
            if inst.form == Form::Vop3
                && r.kind == Kind::S
                && r.len == 2
                && !MAD_PAIR_ROWS.contains(&row.name)
            {
                return Ok(format!("s{}", r.base));
            }
            Ok(format!("{r}"))
        }
        Operand::Half(r, part) => Ok(format!(
            "{r}.{}",
            match part {
                peacemaker_ir::operand::Half::Hi => "h",
                peacemaker_ir::operand::Half::Lo => "l",
            }
        )),
        Operand::Special(_) | Operand::Hwreg(_) | Operand::SendMsg(_) => Ok(format!("{operand}")),
        Operand::Inline(peacemaker_ir::operand::InlineConst::Integer(n)) => Ok(format!("{n}")),
        Operand::Inline(peacemaker_ir::operand::InlineConst::FloatBits(b)) => Ok(llvm_float(*b)),
        Operand::Inline(peacemaker_ir::operand::InlineConst::InvTwoPi) => {
            Ok("0.15915494".to_owned())
        }
        Operand::Literal(n) => Ok(format!("{n:#x}")),
        Operand::Imm(ImmField::Sopp(n)) | Operand::Imm(ImmField::Sopk(n)) => {
            Ok(sopp_imm_text(row, *n))
        }
        Operand::Imm(ImmField::SmemOffset(n)) => Ok(if *n >= 0 {
            format!("0x{n:x}")
        } else {
            format!("-0x{:x}", n.unsigned_abs())
        }),
        Operand::Imm(ImmField::SmemDisplacement(_))
        | Operand::Imm(ImmField::VmemOffset(_))
        | Operand::Imm(ImmField::DsOffset(_))
        | Operand::Imm(ImmField::DsOffset0(_))
        | Operand::Imm(ImmField::DsOffset1(_)) => Ok(format!("{operand}")),
        Operand::Imm(ImmField::Unsigned(n)) => Ok(format!("{n:#x}")),
        Operand::Label(id) => Ok(format!(".LBB{}", id.0)),
        Operand::CacheTh(th) if row.name.starts_with("buffer_") => {
            let hints = if row.name.contains("_load") {
                ["RT", "NT", "HT", "LU", "NT_RT", "RT_NT", "NT_HT", "BYPASS"]
            } else {
                [
                    "RT", "NT", "HT", "BYPASS", "NT_RT", "RT_NT", "NT_HT", "NT_WB",
                ]
            };
            Ok(format!(
                "th:TH_{}_{}",
                if row.name.contains("_load") {
                    "LOAD"
                } else {
                    "STORE"
                },
                hints[usize::from(*th)]
            ))
        }
        Operand::Scope(_) | Operand::CacheTh(_) | Operand::Vmem(_) => Ok(format!("{operand}")),
        Operand::Depctr(_) => Err(PrintError::Unspellable(
            "depctr operand (encode via s_wait_alu immediate)".into(),
        )),
    }
}

/// VOP3 rows whose SGPR-wide source prints as a range, not single.
const MAD_PAIR_ROWS: [&str; 2] = ["v_mad_co_u64_u32", "v_mad_co_i64_i32"];

/// LLVM float-inline spelling: `.0` suffix on integral values.
fn llvm_float(bits: u32) -> String {
    let value = f32::from_bits(bits);
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

fn sopp_imm_text(row: &OpRow, n: i16) -> String {
    if row.form == Form::Sopk {
        return format!("0x{:x}", n as u16);
    }
    match row.name {
        name if name.starts_with("s_branch") || name.starts_with("s_cbranch") => {
            format!("{}", n as u16)
        }
        name if name.starts_with("s_wait_") || name == "s_clause" => {
            format!("0x{:x}", n as u16)
        }
        "s_barrier_wait" => {
            if (0..=64).contains(&n) {
                format!("{n}")
            } else {
                format!("0x{:x}", n as u16)
            }
        }
        _ => format!("{n}"),
    }
}

fn delay_text(mods: &Modifiers) -> Result<String, PrintError> {
    const IDS: [&str; 12] = [
        "",
        "VALU_DEP_1",
        "VALU_DEP_2",
        "VALU_DEP_3",
        "VALU_DEP_4",
        "TRANS32_DEP_1",
        "TRANS32_DEP_2",
        "TRANS32_DEP_3",
        "FMA_ACCUM_CYCLE_1",
        "SALU_CYCLE_1",
        "SALU_CYCLE_2",
        "SALU_CYCLE_3",
    ];
    const SKIPS: [&str; 6] = ["", "NEXT", "SKIP_1", "SKIP_2", "SKIP_3", "SKIP_4"];
    let Some(delay) = &mods.delay else {
        return Err(PrintError::Unspellable("s_delay_alu without hint".into()));
    };
    let id0 = usize::from(delay.instid0);
    let skip = usize::from(delay.instskip);
    let id1 = usize::from(delay.instid1);
    if id0 >= IDS.len() || skip >= SKIPS.len() || id1 >= IDS.len() {
        return Err(PrintError::Unspellable(format!(
            "invalid s_delay_alu hint {delay:?}"
        )));
    }
    // instid1 is 9 bits; only the named values have a spelling.
    if id1 > 11 {
        return Err(PrintError::Unspellable(format!(
            "invalid s_delay_alu instid1 {id1}"
        )));
    }
    let mut parts = Vec::new();
    if id0 != 0 {
        parts.push(format!("instid0({})", IDS[id0]));
    }
    if skip != 0 {
        parts.push(format!("instskip({})", SKIPS[skip]));
    }
    if id1 != 0 {
        parts.push(format!("instid1({})", IDS[id1]));
    }
    if parts.is_empty() {
        return Ok("s_delay_alu 0".to_owned());
    }
    Ok(format!("s_delay_alu {}", parts.join(" | ")))
}

fn wait_alu_text(inst: &Inst) -> Result<String, PrintError> {
    let raw = match inst.operands.first() {
        Some(Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n))) => *n as u16,
        other => {
            return Err(PrintError::Unspellable(format!(
                "s_wait_alu without raw immediate (have {other:?})"
            )));
        }
    };
    // (table name, shift, max). Composition base is 0xff9f (reserved bits
    // 5-6 clear, counters max); 0xffff is the assembler's all-ones
    // spelling, not a counter state. Reserved bits set print hex.
    if raw == 0xffff {
        return Ok("s_wait_alu 0xffff".to_owned());
    }
    if raw & 0x60 != 0 {
        return Ok(format!("s_wait_alu 0x{raw:x}"));
    }
    const FIELDS: [(&str, u8, u16); 7] = [
        ("depctr_hold_cnt", 7, 1),
        ("depctr_sa_sdst", 0, 1),
        ("depctr_va_vdst", 12, 15),
        ("depctr_va_sdst", 9, 7),
        ("depctr_va_ssrc", 8, 1),
        ("depctr_va_vcc", 1, 1),
        ("depctr_vm_vsrc", 2, 7),
    ];
    let mut parts = Vec::new();
    for (name, shift, max) in FIELDS {
        let value = (raw >> shift) & max;
        if value != max {
            parts.push(format!("{name}({value})"));
        }
    }
    if parts.is_empty() {
        // 0xff9f (every counter maxed, reserved clear) prints the full
        // list; only exact 0xffff prints as hex (handled above).
        parts = FIELDS
            .iter()
            .map(|(name, _, max)| format!("{name}({max})"))
            .collect();
    }
    Ok(format!("s_wait_alu {}", parts.join(" ")))
}

fn apply_vop3_src_mods(
    text: &str,
    operand: &Operand,
    slot: &str,
    mods: &Modifiers,
) -> Result<String, PrintError> {
    let src = match slot {
        "SRC0" => 0,
        "SRC1" => 1,
        "SRC2" => 2,
        _ => return Ok(text.to_owned()),
    };
    let mut out = text.to_owned();
    if mods.abs >> src & 1 != 0 {
        out = format!("|{out}|");
    }
    if mods.neg >> src & 1 != 0 {
        out = match operand {
            Operand::Reg(_) | Operand::Half(_, _) | Operand::Special(_) => {
                format!("-{out}")
            }
            _ => format!("neg({out})"),
        };
    }
    Ok(out)
}

fn modifier_suffix(inst: &Inst, row: &OpRow) -> Result<String, PrintError> {
    let mods = &inst.mods;
    let mut out = String::new();
    match inst.form {
        Form::Vop1Dpp | Form::Vop2Dpp => {
            let dpp = mods
                .dpp
                .ok_or_else(|| PrintError::Unspellable("DPP modifiers missing".into()))?;
            let control =
                if let Some(shift) = dpp.ctrl.checked_sub(0x100).filter(|v| (1..=15).contains(v)) {
                    format!("row_shl:{shift}")
                } else if let Some(lane) = dpp.ctrl.checked_sub(0x150).filter(|v| *v <= 15) {
                    format!("row_share:{lane}")
                } else {
                    return Err(PrintError::Unspellable(format!(
                        "DPP control {:#x}",
                        dpp.ctrl
                    )));
                };
            out.push_str(&format!(
                " {control} row_mask:{:#x} bank_mask:{:#x}",
                dpp.row_mask, dpp.bank_mask
            ));
            if dpp.bound_ctrl {
                out.push_str(" bound_ctrl:1");
            }
        }
        Form::Vop3 => {
            if mods.omod != Omod::None {
                out.push_str(match mods.omod {
                    Omod::Mul2 => " mul:2",
                    Omod::Mul4 => " mul:4",
                    Omod::Div2 => " div:2",
                    Omod::None => unreachable!(),
                });
            }
            if mods.clamp {
                out.push_str(" clamp");
            }
            out.push_str(&vop3_op_sel_suffix(inst, row)?);
        }
        Form::Vop3p => {
            // SWMMAC carries its sparse-index half select in OP_SEL bit 0.
            if row.name.starts_with("v_swmmac_") && mods.op_sel & 1 != 0 {
                out.push_str(" index_key:1");
            }
            if mods.neg_lo != 0 {
                out.push_str(&format!(
                    " neg_lo:[{},{},{}]",
                    mods.neg_lo & 1,
                    mods.neg_lo >> 1 & 1,
                    mods.neg_lo >> 2 & 1
                ));
            }
            if mods.neg_hi != 0 {
                out.push_str(&format!(
                    " neg_hi:[{},{},{}]",
                    mods.neg_hi & 1,
                    mods.neg_hi >> 1 & 1,
                    mods.neg_hi >> 2 & 1
                ));
            }
            // No KT48 VOP3P instruction prints op_sel: omitted for rows
            // without 16-bit operands (WMMA, duals), same rule as VOP3.
            // Entry layout for a hypothetical 16-bit VOP3P row is
            // unprobed; the source-bit entries below are best-effort.
            if has_w16_operand(row) && mods.op_sel != 0 {
                let nsrcs = src_slot_count(row);
                let mut entries: Vec<String> = (0..nsrcs)
                    .map(|i| ((mods.op_sel >> i) & 1).to_string())
                    .collect();
                entries.push(((mods.op_sel >> 3) & 1).to_string());
                out.push_str(&format!(" op_sel:[{}]", entries.join(",")));
            }
            if has_w16_operand(row) && mods.op_sel_hi != 0 {
                out.push_str(&format!(
                    " op_sel_hi:[{},{},{}]",
                    mods.op_sel_hi & 1,
                    mods.op_sel_hi >> 1 & 1,
                    mods.op_sel_hi >> 2 & 1
                ));
            }
            if mods.clamp {
                out.push_str(" clamp");
            }
        }
        _ => {}
    }
    Ok(out)
}

fn has_w16_operand(row: &OpRow) -> bool {
    grammar_slots(row).iter().any(|(_, bits)| *bits == 16)
}

fn src_slot_count(row: &OpRow) -> u32 {
    grammar_slots(row)
        .iter()
        .filter(|(name, _)| *name == "SRC0" || *name == "SRC1" || *name == "SRC2")
        .count() as u32
}

fn vop3_op_sel_suffix(inst: &Inst, row: &OpRow) -> Result<String, PrintError> {
    if !has_w16_operand(row) || inst.mods.op_sel == 0 {
        return Ok(String::new());
    }
    let nsrcs = src_slot_count(row);
    // LLVM only decodes the used bits (sources plus the bit-3 destination
    // select); anything else is unspellable, not normalisable.
    let allowed = if nsrcs >= 3 {
        0xf
    } else {
        ((1 << nsrcs) - 1) | 0x8
    };
    if u32::from(inst.mods.op_sel) & !allowed != 0 {
        return Err(PrintError::Unspellable(format!(
            "{} has op_sel {:#x} outside its spellable bits",
            row.name, inst.mods.op_sel
        )));
    }
    let mut entries: Vec<String> = (0..nsrcs)
        .map(|i| ((inst.mods.op_sel >> i) & 1).to_string())
        .collect();
    entries.push(((inst.mods.op_sel >> 3) & 1).to_string());
    Ok(format!(" op_sel:[{}]", entries.join(",")))
}

fn vopd_text(inst: &Inst, row: &OpRow) -> Result<String, PrintError> {
    let (y_op, x_count) = match inst.fields {
        FormFields::Vopd { y_op, x_operands } => (y_op, usize::from(x_operands)),
        _ => {
            return Err(PrintError::Unspellable(format!(
                "{} without VOPD halves",
                row.name
            )));
        }
    };
    // Y-dest note: the codec must supply the true destination (7-bit field
    // plus implied opposite parity); the printer renders what it is given
    // and the T3 test fails closed on any disagreement with objdump.
    let x_row = row;
    let arch = Arch::Gfx1201;
    let y_row = isa::lookup(arch, y_op, Form::Vopd).ok_or(PrintError::UnknownOpcode {
        op: y_op,
        form: Form::Vopd,
    })?;
    if inst.operands.len() < x_count {
        return Err(PrintError::Unspellable(format!(
            "{} has fewer operands than its X half",
            row.name
        )));
    }
    let x_slots = grammar_slots(x_row);
    let y_slots = grammar_slots(y_row);
    if x_slots.len() != x_count || inst.operands.len() != x_count + y_slots.len() {
        return Err(PrintError::Unspellable(format!(
            "{} operand split disagrees with its halves",
            row.name
        )));
    }
    let mut x = x_row.name.to_owned();
    for (i, (slot, _)) in x_slots.iter().enumerate() {
        x.push(if i == 0 { ' ' } else { ',' });
        if i != 0 {
            x.push(' ');
        }
        x.push_str(&operand_text(&inst.operands[i], Some(slot), x_row, inst)?);
    }
    let mut y = y_row.name.to_owned();
    for (i, (slot, _)) in y_slots.iter().enumerate() {
        y.push(if i == 0 { ' ' } else { ',' });
        if i != 0 {
            y.push(' ');
        }
        y.push_str(&operand_text(
            &inst.operands[x_count + i],
            Some(slot),
            y_row,
            inst,
        )?);
    }
    Ok(format!("{x} :: {y}"))
}

#[cfg(test)]
#[path = "print_tests.rs"]
mod tests;
