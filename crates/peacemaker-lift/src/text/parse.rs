//! C3b: parser for canonical objdump text and `.s` sources.
//!
//! [`parse_line`] reads one canonical instruction line (as [`canonical`]
//! prints it, optionally with its `//` address comment) back into a typed
//! [`Inst`]. Don't-care fills the syntax cannot spell take canonical
//! defaults (`src2_unused = 0x80`, ignored zeros, honored values re-derived
//! from the parsed operands/modifiers); anything else is a [`ParseError`],
//! never a guess. [`parse_source`] parses `.s` label/directive structure for
//! [`lift_text`](crate::lift_text): labels map to instruction ordinals,
//! directives pass through verbatim to the assembler.
//!
//! [`canonical`]: crate::text::canonical
//! [`Inst`]: peacemaker_ir::inst::Inst

use smallvec::SmallVec;
use peacemaker_ir::{
    cfg::BlockId,
    inst::{Arch, Form, FormFields, Inst, NamedField},
    isa::{self, FieldClass, OpRow},
    operand::{
        CacheScope, DelayAluHint, Half, HwReg, ImmField, InlineConst, Modifiers,
        Msg, Omod, Operand, Special, VmemToken,
    },
    provenance::Provenance,
    reg::{Kind, RegRef},
    wait::{Counter, WaitImm},
};

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unknown mnemonic {0:?}")]
    UnknownMnemonic(String),
    #[error("bad operand {operand:?} for {name}: {reason}")]
    BadOperand { name: String, operand: String, reason: String },
    #[error("bad modifier {modifier:?} for {name}: {reason}")]
    BadModifier { name: String, modifier: String, reason: String },
    #[error("operand count mismatch for {name}: have {have}, need {need}")]
    Count { name: String, have: usize, need: usize },
    #[error("bad source line {line:?}: {reason}")]
    BadSource { line: String, reason: String },
    #[error("typed construction failed for {name}: {reason}")]
    Invalid { name: String, reason: String },
}

fn bad_operand(
    name: &str,
    operand: &str,
    reason: impl Into<String>,
) -> ParseError {
    ParseError::BadOperand {
        name: name.into(),
        operand: operand.into(),
        reason: reason.into(),
    }
}

/// Parse an instruction line for gfx1100, gfx1151, or gfx1201.
///
/// Accepts canonical disassembly and PM builder syntax, including gfx11
/// wait/cache modifiers, true16 halves, VOPD packets, and numeric branch
/// word offsets. Labels belong to [`parse_source`]; unsupported mnemonics
/// and operand/modifier spellings return [`ParseError`].
pub fn parse_line(line: &str, arch: Arch) -> Result<Inst, ParseError> {
    let line = strip_comment(line).trim();
    if line.is_empty() {
        return Err(ParseError::BadSource {
            line: line.into(),
            reason: "empty instruction line".into(),
        });
    }
    if let Some((x, y)) = line.split_once("::") {
        return parse_vopd(x, y, arch);
    }
    let (name, rest) = split_name(line)?;
    let row = find_row(name, arch)?;
    if row.form == Form::Sopp && row.name == "s_delay_alu" {
        return parse_delay(row, rest, arch);
    }
    if row.form == Form::Sopp && matches!(row.name, "s_wait_alu" | "s_waitcnt_depctr") {
        return parse_wait_alu(row, rest, arch);
    }
    if row.name == "s_waitcnt" {
        return parse_waitcnt(row, rest, arch);
    }
    parse_single(row, rest, arch)
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => line[..i].trim_end(),
        None => line,
    }
}

fn split_name(line: &str) -> Result<(&str, &str), ParseError> {
    match line.find(char::is_whitespace) {
        Some(i) => Ok((line[..i].trim_end(), line[i..].trim())),
        None => Ok((line, "")),
    }
}

fn find_row(name: &str, arch: Arch) -> Result<&'static OpRow, ParseError> {
    isa::table(arch)
        .iter()
        .find(|row| row.name == name)
        .ok_or_else(|| ParseError::UnknownMnemonic(name.into()))
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

/// Split operands on top-level commas (bracket/paren aware).
fn split_operands(rest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in rest.char_indices() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(rest[start..i].trim().to_owned());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = rest[start..].trim();
    if !(out.is_empty() && last.is_empty()) {
        out.push(last.to_owned());
    }
    out
}

fn is_suffix_token(token: &str) -> bool {
    token.starts_with("op_sel:")
        || token.starts_with("op_sel_hi:")
        || token.starts_with("neg_lo:")
        || token.starts_with("neg_hi:")
        || token == "clamp"
        || token == "mul:2"
        || token == "mul:4"
        || token == "div:2"
        || token.starts_with("row_shl:")
        || token.starts_with("row_share:")
        || token.starts_with("row_xmask:")
        || token.starts_with("row_mask:")
        || token.starts_with("bank_mask:")
        || token.starts_with("bound_ctrl:")
}

fn is_extra_token(token: &str) -> bool {
    token.starts_with("offset")
        || token == "offen"
        || token.starts_with("scope:") || token.starts_with("th:")
}

/// The last comma segment may carry space-separated suffix modifiers (VOP3)
/// or trailing memory operands (DS/VMEM). Returns (operand, suffixes,
/// extras); unknown trailing words are reported as suffixes so the caller
/// fails them as bad modifiers rather than misparsing an operand.
fn split_tail(segment: &str) -> (String, Vec<String>, Vec<String>) {
    let mut depth = 0usize;
    let mut words = segment.split(|ch: char| {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            _ => {}
        }
        ch.is_whitespace() && depth == 0
    }).filter(|word| !word.is_empty());
    let operand = words.next().unwrap_or("").to_owned();
    let mut suffixes = Vec::new();
    let mut extras = Vec::new();
    for word in words {
        if is_suffix_token(word) {
            suffixes.push(word.to_owned());
        } else if is_extra_token(word) {
            extras.push(word.to_owned());
        } else {
            suffixes.push(word.to_owned());
        }
    }
    (operand, suffixes, extras)
}

struct ParsedOperands {
    arch: Arch,
    operands: Vec<Operand>,
    mods: Modifiers,
    literal: Option<u32>,
}

fn parse_single(row: &'static OpRow, rest: &str, arch: Arch) -> Result<Inst, ParseError> {
    let slots = grammar_slots(row);
    let segments = if rest.is_empty() { Vec::new() } else { split_operands(rest) };
    let mut operand_texts: Vec<String> = Vec::new();
    let mut suffixes: Vec<String> = Vec::new();
    let mut extras: Vec<String> = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        if i + 1 == segments.len() {
            let (op, suf, ext) = split_tail(segment);
            if op.is_empty() {
                // Suffix-only tail (cannot happen canonically).
            } else if is_extra_token(&op) {
                // A trailing memory operand standing alone
                // (`global_inv scope:SCOPE_SE`).
                extras.push(op);
            } else {
                operand_texts.push(op);
            }
            suffixes.extend(suf);
            extras.extend(ext);
        } else {
            operand_texts.push(segment.clone());
        }
    }
    let mut parsed = ParsedOperands {
        arch,
        operands: Vec::new(),
        mods: Modifiers::default(),
        literal: None,
    };
    if matches!(row.name, "v_fmamk_f32" | "v_fmaak_f32") && operand_texts.len() == 4 {
        let index = if row.name == "v_fmamk_f32" { 2 } else { 3 };
        let text = operand_texts.remove(index);
        let value = text.strip_prefix("0x")
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
            .ok_or_else(|| bad_operand(row.name, &text, "embedded literal needs a 32-bit hex value"))?;
        parsed.literal = Some(value);
    }
    if row.form == Form::Vop2 && row.name == "v_cndmask_b32_e32" {
        if operand_texts.last().is_some_and(|t| t == "vcc_lo") {
            operand_texts.pop();
        } else {
            return Err(bad_operand(
                row.name,
                rest,
                "v_cndmask_b32_e32 needs its trailing vcc_lo",
            ));
        }
    }
    // `v_cmpx*` VOP3 rows carry a hidden exec VDST (fixed encoding 0x7e)
    // that canonical text omits; the visible operands map to SRC0/SRC1.
    let mut slot_offset = 0usize;
    if row.form == Form::Vop3 && row.name.starts_with("v_cmpx") {
        if operand_texts.len() + 1 != slots.len() || slots.is_empty() {
            return Err(ParseError::Count {
                name: row.name.into(),
                have: operand_texts.len(),
                need: slots.len().saturating_sub(1),
            });
        }
        slot_offset = 1;
    }
    // VOPC compares carry a synthetic vcc destination that objdump prints.
    let mut text_offset = 0usize;
    if row.form == Form::Vopc && row.name.starts_with("v_cmp_") {
        if operand_texts.first().is_some_and(|t| t == "vcc_lo") {
            text_offset = 1;
        } else {
            return Err(bad_operand(
                row.name,
                rest,
                "VOPC compare needs its vcc_lo destination",
            ));
        }
    }
    let need = slots.len() + text_offset - slot_offset;
    if operand_texts.len() != need {
        return Err(ParseError::Count {
            name: row.name.into(),
            have: operand_texts.len(),
            need,
        });
    }
    if !extras.is_empty()
        && !(row.form == Form::Ds || row.form == Form::Smem || matches!(row.form, Form::Vmem(_)))
    {
        return Err(bad_operand(
            row.name,
            rest,
            "trailing operands only exist on DS/VMEM rows",
        ));
    }
    for (i, text) in operand_texts.iter().enumerate() {
        if text_offset == 1 && i == 0 {
            parsed.operands.push(Operand::Special(Special::VccLo));
            continue;
        }
        let slot_index = i + slot_offset - text_offset;
        let (slot, bits) = slots[slot_index];
        let operand = if matches!(row.form, Form::Vop3 | Form::Vop3p | Form::Vop1Dpp | Form::Vop2Dpp) {
            parse_vop3_operand(row, slot, bits, text, &mut parsed.mods, arch)?
        } else {
            parse_operand(row, slot, bits, text, arch)?
        };
        parsed.operands.push(operand);
    }
    // Match the codec's canonical order: offset, offen, TH, scope.
    extras.sort_by_key(|text| if text.starts_with("offset") { 0 } else if text == "offen" { 1 } else if text.starts_with("th:") { 2 } else { 3 });
    for text in &extras {
        if parsed.arch != Arch::Gfx1201 && (text.starts_with("th:") || text.starts_with("scope:")) {
            return Err(bad_operand(row.name, text, "th:/scope: cache policy is gfx12 syntax"));
        }
        let operand = parse_extra(row, text)?;
        // Only VBUFFER carries TH as an operand; GLOBAL keeps it in the cache policy.
        if let Operand::CacheTh(th) = operand {
            if row.form == Form::Vmem(peacemaker_ir::inst::VmemForm::Global) { parsed.mods.cpol.th = th; continue }
        }
        if !matches!(operand, Operand::Imm(ImmField::SmemDisplacement(0) | ImmField::VmemOffset(0) | ImmField::DsOffset(0) | ImmField::DsOffset0(0) | ImmField::DsOffset1(0)) | Operand::Scope(CacheScope::Cu) | Operand::CacheTh(0)) {
            parsed.operands.push(operand);
        }
    }
    if row.form == Form::Vmem(peacemaker_ir::inst::VmemForm::Buffer)
        && parsed.operands.iter().any(|op| matches!(op, Operand::Vmem(VmemToken::Offen)))
    {
        if let Some((index, _)) = slots.iter().enumerate().find(|(_, (name, _))| *name == "VADDR") {
            if let Some(Operand::Reg(addr)) = parsed.operands.get_mut(index) { addr.len = 1; }
        }
    }
    if row.form == Form::Vmem(peacemaker_ir::inst::VmemForm::Global) {
        let scalar = slots.iter().position(|(name, _)| *name == "SADDR");
        if scalar.is_some_and(|index| parsed.operands.get(index) != Some(&Operand::Vmem(VmemToken::Off))) {
            if let Some(index) = slots.iter().position(|(name, _)| matches!(*name, "ADDR" | "VADDR")) {
                if let Some(Operand::Reg(addr)) = parsed.operands.get_mut(index) { addr.len = 1; }
            }
        }
    }
    parse_suffixes(row, &suffixes, &mut parsed.mods)?;
    imply_op_sel(row, &suffixes, &mut parsed.mods, &parsed.operands)?;
    // Packed arithmetic and WMMA default high-half selects to all-ones;
    // mixed-precision FMA defaults them to zero. Explicit spellings win.
    if row.form == Form::Vop3p && !row.name.starts_with("v_fma_mix")
        && !suffixes.iter().any(|s| s.starts_with("op_sel_hi:["))
        && parsed.mods.op_sel_hi == 0
    {
        parsed.mods.op_sel_hi = 7;
    }
    parse_wait_clause(row, &mut parsed)?;
    finish(row, parsed)
}

/// Strip a VOP3 neg/abs affix; returns the bare operand. A leading `-`
/// followed by a digit is an inline constant (`-16`), not negation.
fn strip_affix(text: &str) -> (String, bool, bool) {
    let mut text = text;
    let mut neg = false;
    if let Some(inner) = text.strip_prefix('-') {
        if inner.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
            return (text.to_owned(), false, false);
        }
        neg = true;
        text = inner;
    } else if let Some(inner) =
        text.strip_prefix("neg(").and_then(|s| s.strip_suffix(')'))
    {
        neg = true;
        text = inner;
    }
    let mut abs = false;
    if let Some(inner) =
        text.strip_prefix('|').and_then(|s| s.strip_suffix('|'))
    {
        abs = true;
        text = inner;
    }
    (text.to_owned(), neg, abs)
}

/// Parse a VOP3 operand: strip the neg/abs affix into `mods`, then parse
/// the bare operand against the slot width.
fn parse_vop3_operand(
    row: &OpRow,
    slot: &str,
    bits: u16,
    text: &str,
    mods: &mut Modifiers,
    arch: Arch,
) -> Result<Operand, ParseError> {
    let src: u8 = match slot {
        "SRC0" => 0,
        "SRC1" => 1,
        "SRC2" => 2,
        _ => {
            let (bare, neg, abs) = strip_affix(text);
            if neg || abs || bare != text {
                return Err(bad_operand(
                    row.name,
                    text,
                    "affix on a non-source operand",
                ));
            }
            return parse_operand(row, slot, bits, text, arch);
        }
    };
    let (bare, neg, abs) = strip_affix(text);
    if neg {
        if row.form == Form::Vop3p { mods.neg_lo |= 1 << src; } else { mods.neg |= 1 << src; }
    }
    if abs {
        if row.form == Form::Vop3p { return Err(bad_operand(row.name, text, "packed sources do not support abs")); }
        mods.abs |= 1 << src;
    }
    parse_operand(row, slot, bits, &bare, arch)
}

fn parse_waitcnt(row: &'static OpRow, rest: &str, arch: Arch) -> Result<Inst, ParseError> {
    let raw = if rest.contains('(') {
        let mut raw = 0xfff7u16;
        for token in rest.split_whitespace() {
            let (name, value) = token.split_once('(').and_then(|(name, value)| value.strip_suffix(')').map(|value| (name, value)))
                .ok_or_else(|| bad_operand(row.name, token, "expected counter(value)"))?;
            let (shift, max) = match name {
                "vmcnt" => (10, 63),
                "expcnt" => (0, 7),
                "lgkmcnt" => (4, 63),
                _ => return Err(bad_operand(row.name, token, "unknown wait counter")),
            };
            let value: u16 = value.parse().map_err(|_| bad_operand(row.name, token, "bad wait count"))?;
            if value > max { return Err(bad_operand(row.name, token, "wait count out of range")); }
            raw = (raw & !(max << shift)) | (value << shift);
        }
        raw
    } else { parse_number(rest, row.name, rest)? };
    let mut parsed = ParsedOperands { arch, operands: vec![Operand::Imm(ImmField::Sopp(raw as i16))], mods: Modifiers::default(), literal: None };
    parse_wait_clause(row, &mut parsed)?;
    finish(row, parsed)
}

/// Delay-hint lines: symbolic fields or a bare number.
fn parse_delay(
    row: &'static OpRow,
    rest: &str,
    arch: Arch,
) -> Result<Inst, ParseError> {
    const IDS: [(&str, u8); 12] = [
        ("NO_DEP", 0),
        ("VALU_DEP_1", 1),
        ("VALU_DEP_2", 2),
        ("VALU_DEP_3", 3),
        ("VALU_DEP_4", 4),
        ("TRANS32_DEP_1", 5),
        ("TRANS32_DEP_2", 6),
        ("TRANS32_DEP_3", 7),
        ("FMA_ACCUM_CYCLE_1", 8),
        ("SALU_CYCLE_1", 9),
        ("SALU_CYCLE_2", 10),
        ("SALU_CYCLE_3", 11),
    ];
    const SKIPS: [(&str, u8); 6] = [
        ("", 0),
        ("NEXT", 1),
        ("SKIP_1", 2),
        ("SKIP_2", 3),
        ("SKIP_3", 4),
        ("SKIP_4", 5),
    ];
    let raw: u16 = if rest.starts_with("instid") {
        let (mut id0, mut skip, mut id1) = (0u8, 0u8, 0u8);
        for part in rest.split('|') {
            let part = part.trim();
            let (kind, name) =
                part.split_once('(').and_then(|(k, v)| {
                    v.strip_suffix(')').map(|v| (k.trim(), v.trim()))
                }).ok_or_else(|| {
                    bad_operand(row.name, rest, "expected instid0(..) | …")
                })?;
            let value = match kind {
                "instid0" | "instid1" => IDS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| *v)
                    .ok_or_else(|| {
                        bad_operand(row.name, rest, "unknown delay id")
                    })?,
                "instskip" => SKIPS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| *v)
                    .ok_or_else(|| {
                        bad_operand(row.name, rest, "unknown delay skip")
                    })?,
                _ => {
                    return Err(bad_operand(
                        row.name,
                        rest,
                        "expected instid0/instskip/instid1",
                    ));
                }
            };
            match kind {
                "instid0" => id0 = value,
                "instskip" => skip = value,
                _ => id1 = value,
            }
        }
        u16::from(id0) | u16::from(skip) << 4 | u16::from(id1) << 7
    } else {
        parse_number(rest, row.name, rest)?
    };
    let mut parsed = ParsedOperands {
        arch,
        operands: vec![Operand::Imm(ImmField::Sopp(raw as i16))],
        mods: Modifiers::default(),
        literal: None,
    };
    parsed.mods.delay = Some(DelayAluHint {
        instid0: (raw & 15) as u8,
        instskip: ((raw >> 4) & 7) as u8,
        instid1: ((raw >> 7) & 0x1ff) as u8,
    });
    finish(row, parsed)
}

/// `s_wait_alu`: depctr symbols (space-separated) or a bare number.
fn parse_wait_alu(
    row: &'static OpRow,
    rest: &str,
    arch: Arch,
) -> Result<Inst, ParseError> {
    // (table name, shift, max)
    const FIELDS: [(&str, u8, u16); 7] = [
        ("depctr_hold_cnt", 7, 1),
        ("depctr_sa_sdst", 0, 1),
        ("depctr_va_vdst", 12, 15),
        ("depctr_va_sdst", 9, 7),
        ("depctr_va_ssrc", 8, 1),
        ("depctr_va_vcc", 1, 1),
        ("depctr_vm_vsrc", 2, 7),
    ];
    let raw: u16 = if rest.starts_with("depctr") {
        // Unmentioned counters stay at max (no wait), as the assembler
        // does; the composition base is 0xff9f (reserved bits clear).
        let mut raw = 0xff9fu16;
        for token in rest.split_whitespace() {
            let (name, value) =
                token.split_once('(').and_then(|(n, v)| {
                    v.strip_suffix(')').map(|v| (n, v))
                }).ok_or_else(|| {
                    bad_operand(row.name, rest, "expected depctr_name(n)")
                })?;
            let (shift, max) = FIELDS
                .iter()
                .find(|(n, _, _)| *n == name)
                .map(|(_, s, m)| (*s, *m))
                .ok_or_else(|| {
                    bad_operand(row.name, rest, "unknown depctr counter")
                })?;
            let value: u16 = value.parse().map_err(|_| {
                bad_operand(row.name, rest, "bad depctr count")
            })?;
            if value > max {
                return Err(bad_operand(
                    row.name,
                    rest,
                    "depctr count out of range",
                ));
            }
            raw = raw & !(max << shift) | value << shift;
        }
        raw
    } else {
        parse_number(rest, row.name, rest)?
    };
    finish(
        row,
        ParsedOperands {
            arch,
            operands: vec![Operand::Imm(ImmField::Sopp(raw as i16))],
            mods: Modifiers { wait: (row.name == "s_waitcnt_depctr").then(WaitImm::default), ..Modifiers::default() },
            literal: None,
        },
    )
}

fn parse_vopd(x: &str, y: &str, arch: Arch) -> Result<Inst, ParseError> {
    let (x_name, x_rest) = split_name(x.trim())?;
    let (y_name, y_rest) = split_name(y.trim())?;
    let x_row = find_row(x_name, arch)?;
    let y_row = find_row(y_name, arch)?;
    if x_row.form != Form::Vopd || y_row.form != Form::Vopd {
        return Err(ParseError::BadSource {
            line: format!("{x} :: {y}"),
            reason: "VOPD halves must both be VOPD opcodes".into(),
        });
    }
    let x_slots = grammar_slots(x_row);
    let y_slots = grammar_slots(y_row);
    let x_texts = split_operands(x_rest);
    let y_texts = split_operands(y_rest);
    if x_texts.len() != x_slots.len() || y_texts.len() != y_slots.len() {
        return Err(ParseError::Count {
            name: format!("{} :: {}", x_row.name, y_row.name),
            have: x_texts.len() + y_texts.len(),
            need: x_slots.len() + y_slots.len(),
        });
    }
    let mut parsed = ParsedOperands {
        arch,
        operands: Vec::new(),
        mods: Modifiers::default(),
        literal: None,
    };
    for (text, (slot, bits)) in x_texts.iter().zip(x_slots.iter()) {
        parsed.operands.push(parse_operand(x_row, slot, *bits, text, arch)?);
    }
    for (text, (slot, bits)) in y_texts.iter().zip(y_slots.iter()) {
        parsed.operands.push(parse_operand(y_row, slot, *bits, text, arch)?);
    }
    // One shared literal at most (builder's VOPD rule).
    let mut literal = None;
    for operand in &parsed.operands {
        if let Operand::Literal(value) = operand {
            if literal.is_some_and(|l| l != *value) {
                return Err(bad_operand(
                    x_row.name,
                    x_rest,
                    "VOPD supports only one shared literal",
                ));
            }
            literal = Some(*value);
        }
    }
    parsed.literal = literal;
    let fields = FormFields::Vopd {
        y_op: y_row.op,
        x_operands: x_slots.len() as u8,
    };
    Inst::from_parts(
        arch,
        x_row.op,
        Form::Vopd,
        fields,
        SmallVec::from_vec(parsed.operands),
        parsed.mods,
        parsed.literal,
        Provenance::default(),
    )
    .map_err(|e| ParseError::Invalid {
        name: x_row.name.into(),
        reason: e.to_string(),
    })
}

fn parse_operand(
    row: &OpRow,
    slot: &str,
    bits: u16,
    text: &str,
    arch: Arch,
) -> Result<Operand, ParseError> {
    let width = (bits / 32).max(1) as u8;
    if let Some(ttmp) = parse_ttmp(text)? {
        if width != 1 {
            return Err(bad_operand(
                row.name,
                text,
                "ttmp needs a 32-bit slot",
            ));
        }
        return Ok(ttmp);
    }
    if let Some(mut reg) = parse_reg(text)? {
        // Selector registers take the slot width, exactly as the codec's
        // `select` does (this is what turns canonical `s2` back into the
        // decoded pair).
        if reg.len == 1 {
            reg.len = width;
        } else if reg.len != width {
            return Err(bad_operand(
                row.name,
                text,
                format!("register range needs slot width {width}"),
            ));
        }
        return Ok(Operand::Reg(reg));
    }
    if let Some((kind, base, half)) = parse_half(text)? {
        if width != 1 {
            return Err(bad_operand(
                row.name,
                text,
                "half operand needs a 16-bit slot",
            ));
        }
        return Ok(Operand::Half(
            RegRef { kind, base, len: 1 },
            half,
        ));
    }
    if let Some(special) = parse_special(text) {
        if special == Special::Null && slot == "SOFFSET" && row.form == Form::Smem {
            return Ok(Operand::Imm(ImmField::SmemOffset(0)));
        }
        return Ok(Operand::Special(special));
    }
    if text == "off" {
        if slot == "VADDR" && row.form == Form::Vmem(peacemaker_ir::inst::VmemForm::Buffer) {
            return Ok(Operand::Reg(RegRef { kind: Kind::V, base: 0, len: width }));
        }
        return Ok(Operand::Vmem(VmemToken::Off));
    }
    if text == "offen" {
        return Ok(Operand::Vmem(VmemToken::Offen));
    }
    if let Some(scope) = parse_scope(text) {
        return Ok(Operand::Scope(scope));
    }
    if let Some(msg) = parse_sendmsg(text)? {
        return Ok(Operand::SendMsg(msg));
    }
    if let Some(hw) = parse_hwreg(text, arch)? {
        if slot != "SIMM16" || row.form != Form::Sopk { return Err(bad_operand(row.name, text, "hwreg needs SOPK immediate")); }
        let raw = u16::from(hw.id) | u16::from(hw.offset) << 6 | u16::from(hw.size - 1) << 11;
        return Ok(Operand::Imm(ImmField::Sopk(raw as i16)));
    }
    if let Some(rest) = text.strip_prefix("0x") {
        let value = u32::from_str_radix(rest, 16)
            .map_err(|_| bad_operand(row.name, text, "bad hex immediate"))?;
        return hex_operand(row, slot, text, value);
    }
    if let Some(rest) = text.strip_prefix("-0x") {
        let value = u32::from_str_radix(rest, 16)
            .map_err(|_| bad_operand(row.name, text, "bad hex immediate"))?;
        if slot == "SOFFSET" && row.form == Form::Smem {
            return Ok(Operand::Imm(ImmField::SmemOffset(-(value as i32))));
        }
        return Err(bad_operand(
            row.name,
            text,
            "negative hex outside SMEM offsets",
        ));
    }
    if text.contains('.') {
        if text == "0.15915494" {
            return Ok(Operand::Inline(InlineConst::InvTwoPi));
        }
        let value: f32 = text.parse().map_err(|_| {
            bad_operand(row.name, text, "bad float immediate")
        })?;
        return Ok(Operand::Inline(InlineConst::FloatBits(value.to_bits())));
    }
    if let Ok(n) = text.parse::<i32>() {
        return decimal_operand(row, slot, text, n);
    }
    Err(bad_operand(row.name, text, "unrecognised operand"))
}

fn hex_operand(
    row: &OpRow,
    slot: &str,
    text: &str,
    value: u32,
) -> Result<Operand, ParseError> {
    if slot == "SIMM16" {
        if value > 0xffff {
            return Err(bad_operand(
                row.name,
                text,
                "SIMM16 immediate exceeds 16 bits",
            ));
        }
        let raw = value as u16 as i16;
        return Ok(Operand::Imm(if row.form == Form::Sopk {
            ImmField::Sopk(raw)
        } else {
            ImmField::Sopp(raw)
        }));
    }
    if slot == "SOFFSET" && row.form == Form::Smem {
        if value > 0x00ff_ffff {
            return Err(bad_operand(
                row.name,
                text,
                "SMEM offset exceeds 21 bits",
            ));
        }
        return Ok(Operand::Imm(ImmField::SmemOffset(value as i32)));
    }
    Ok(Operand::Literal(value))
}

fn decimal_operand(
    row: &OpRow,
    slot: &str,
    text: &str,
    n: i32,
) -> Result<Operand, ParseError> {
    if slot == "SIMM16" {
        if row.form == Form::Sopk {
            let raw = i16::try_from(n).map_err(|_| bad_operand(row.name, text, "SIMM16 immediate out of range"))?;
            return Ok(Operand::Imm(ImmField::Sopk(raw)));
        }
        if (row.name == "s_branch" || row.name.starts_with("s_cbranch"))
            && !(-32768..=65535).contains(&n)
        {
            return Err(bad_operand(row.name, text, "branch offset exceeds 16 bits"));
        }
        // Unsigned-printing rows (branches, waits, clause, barrier_wait,
        // delay/wait_alu raw) accept the wrapped value, as mc does.
        let raw = if row.name.starts_with("s_branch")
            || row.name.starts_with("s_cbranch")
            || row.name.starts_with("s_wait_")
            || row.name == "s_clause"
            || row.name == "s_barrier_wait"
            || row.name == "s_delay_alu"
            || row.name == "s_wait_alu"
        {
            (n as u32 & 0xffff) as u16 as i16
        } else {
            i16::try_from(n).map_err(|_| {
                bad_operand(row.name, text, "SIMM16 immediate out of range")
            })?
        };
        return Ok(Operand::Imm(ImmField::Sopp(raw)));
    }
    if slot == "SOFFSET" && row.form == Form::Smem {
        return Ok(Operand::Imm(ImmField::SmemOffset(n)));
    }
    let value = i8::try_from(n)
        .map_err(|_| bad_operand(row.name, text, "not an inline constant"))?;
    if !(-16..=64).contains(&value) {
        return Err(bad_operand(
            row.name,
            text,
            "decimal outside the inline range; literals print hex",
        ));
    }
    Ok(Operand::Inline(InlineConst::Integer(value)))
}

/// `ttmp0/1` are specials (as the codec decodes them); the rest are Ttmp
/// registers.
fn parse_ttmp(text: &str) -> Result<Option<Operand>, ParseError> {
    let Some(num) = text.strip_prefix("ttmp") else { return Ok(None) };
    if !num.bytes().all(|b| b.is_ascii_digit()) || num.is_empty() {
        return Ok(None);
    }
    let n: u8 = num.parse().map_err(|_| ParseError::BadSource {
        line: text.into(),
        reason: "bad ttmp register".into(),
    })?;
    if n > 15 {
        return Err(ParseError::BadSource {
            line: text.into(),
            reason: "ttmp register out of range".into(),
        });
    }
    if n <= 1 {
        return Ok(Some(Operand::Special(Special::Ttmp(n))));
    }
    Ok(Some(Operand::Reg(RegRef { kind: Kind::Ttmp, base: u16::from(n), len: 1 })))
}

fn parse_reg(text: &str) -> Result<Option<RegRef>, ParseError> {
    let (kind, rest) = match text.as_bytes().first() {
        Some(b'v') => (Kind::V, &text[1..]),
        Some(b's') => (Kind::S, &text[1..]),
        _ => return Ok(None),
    };
    if rest.starts_with('[') && rest.ends_with(']') {
        let inner = &rest[1..rest.len() - 1];
        let Some((lo, hi)) = inner.split_once(':') else {
            return Err(ParseError::BadSource {
                line: text.into(),
                reason: "bad register range".into(),
            });
        };
        let (lo, hi): (u16, u16) = (lo.parse().map_err(|_| {
            ParseError::BadSource {
                line: text.into(),
                reason: "bad register range".into(),
            }
        })?, hi.parse().map_err(|_| {
            ParseError::BadSource {
                line: text.into(),
                reason: "bad register range".into(),
            }
        })?);
        if hi < lo {
            return Err(ParseError::BadSource {
                line: text.into(),
                reason: "inverted register range".into(),
            });
        }
        let len = hi - lo + 1;
        if len > 255 {
            return Err(ParseError::BadSource {
                line: text.into(),
                reason: "register range too wide".into(),
            });
        }
        return Ok(Some(RegRef { kind, base: lo, len: len as u8 }));
    }
    if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
        let base: u16 = rest.parse().map_err(|_| ParseError::BadSource {
            line: text.into(),
            reason: "bad register number".into(),
        })?;
        return Ok(Some(RegRef { kind, base, len: 1 }));
    }
    Ok(None)
}

fn parse_half(text: &str) -> Result<Option<(Kind, u16, Half)>, ParseError> {
    let (reg, half) = match text.rsplit_once('.') {
        Some((r, "h")) => (r, Half::Hi),
        Some((r, "l")) => (r, Half::Lo),
        _ => return Ok(None),
    };
    let (kind, num) = match reg.as_bytes().first() {
        Some(b'v') => (Kind::V, &reg[1..]),
        Some(b's') => (Kind::S, &reg[1..]),
        _ => return Ok(None),
    };
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(None);
    }
    let base: u16 = num.parse().map_err(|_| ParseError::BadSource {
        line: text.into(),
        reason: "bad half register".into(),
    })?;
    Ok(Some((kind, base, half)))
}

fn parse_special(text: &str) -> Option<Special> {
    Some(match text {
        "vcc" => Special::Vcc,
        "vcc_lo" => Special::VccLo,
        "vcc_hi" => Special::VccHi,
        "exec" => Special::Exec,
        "exec_lo" => Special::ExecLo,
        "exec_hi" => Special::ExecHi,
        "scc" => Special::Scc,
        "m0" => Special::M0,
        "null" => Special::Null,
        "flat_scratch" => Special::FlatScratch,
        "src_shared_base" => Special::SrcSharedBase,
        "src_shared_limit" => Special::SrcSharedLimit,
        "src_private_base" => Special::SrcPrivateBase,
        "src_private_limit" => Special::SrcPrivateLimit,
        "pc" => Special::Pc,
        _ => return None,
    })
}

fn parse_scope(text: &str) -> Option<CacheScope> {
    match text {
        "scope:SCOPE_CU" => Some(CacheScope::Cu),
        "scope:SCOPE_SE" => Some(CacheScope::Se),
        "scope:SCOPE_DEV" => Some(CacheScope::Dev),
        "scope:SCOPE_SYS" => Some(CacheScope::Sys),
        _ => None,
    }
}

fn parse_sendmsg(text: &str) -> Result<Option<Msg>, ParseError> {
    if text == "sendmsg(MSG_DEALLOC_VGPRS)" {
        return Ok(Some(Msg { id: 3, op: 0 }));
    }
    let Some(inner) = text
        .strip_prefix("sendmsg(")
        .and_then(|s| s.strip_suffix(')'))
    else {
        return Ok(None);
    };
    if inner == "MSG_RTN_GET_REALTIME" {
        return Ok(Some(Msg { id: 131, op: 0 }));
    }
    let (id, op) = inner.split_once(',').ok_or_else(|| {
        bad_operand("s_sendmsg", text, "expected sendmsg(id, op)")
    })?;
    Ok(Some(Msg {
        id: id.trim().parse().map_err(|_| {
            bad_operand("s_sendmsg", text, "bad message id")
        })?,
        op: op.trim().parse().map_err(|_| {
            bad_operand("s_sendmsg", text, "bad message op")
        })?,
    }))
}

/// `hwreg(NAME[, offset, size])` with the target's symbolic register names
/// (the ones `llvm-mc` accepts for it), or a numeric id.
fn parse_hwreg(text: &str, arch: Arch) -> Result<Option<HwReg>, ParseError> {
    let Some(inner) = text
        .strip_prefix("hwreg(")
        .and_then(|s| s.strip_suffix(')'))
    else {
        return Ok(None);
    };
    let mut parts = inner.split(',').map(str::trim);
    let name = parts.next().ok_or_else(|| bad_operand("hwreg", text, "missing id"))?;
    let named = if arch == Arch::Gfx1201 {
        match name {
            // llvm-mc also takes the gfx11 names on gfx12.
            "HW_REG_WAVE_HW_ID1" | "HW_REG_HW_ID1" => Some(23),
            "HW_REG_WAVE_HW_ID2" | "HW_REG_HW_ID2" => Some(24),
            "HW_REG_SHADER_CYCLES_LO" => Some(29),
            "HW_REG_SHADER_CYCLES_HI" => Some(30),
            _ => None,
        }
    } else {
        match name {
            "HW_REG_HW_ID1" => Some(23),
            "HW_REG_HW_ID2" => Some(24),
            "HW_REG_SHADER_CYCLES" => Some(29),
            _ => None,
        }
    };
    let id = match named {
        Some(id) => id,
        None => name.parse().map_err(|_| bad_operand("hwreg", text, "unknown hardware register for this target"))?,
    };
    let offset = parts.next().map(|value| value.parse()).transpose()
        .map_err(|_| bad_operand("hwreg", text, "bad offset"))?.unwrap_or(0);
    let size = parts.next().map(|value| value.parse()).transpose()
        .map_err(|_| bad_operand("hwreg", text, "bad size"))?.unwrap_or(32);
    if parts.next().is_some() || id > 63 || offset > 31 || size == 0 || size > 32 || u16::from(offset) + u16::from(size) > 32 {
        return Err(bad_operand("hwreg", text, "hardware register field out of range"));
    }
    Ok(Some(HwReg { id, offset, size }))
}

fn parse_extra(row: &OpRow, text: &str) -> Result<Operand, ParseError> {
    if text == "offen" {
        return Ok(Operand::Vmem(VmemToken::Offen));
    }
    if let Some(scope) = parse_scope(text) {
        return Ok(Operand::Scope(scope));
    }
    if let Some(th) = text.strip_prefix("th:") {
        let global = row.form == Form::Vmem(peacemaker_ir::inst::VmemForm::Global);
        let load = row.name.starts_with("buffer_load") || global && row.name.starts_with("global_load");
        let store = row.name.starts_with("buffer_store") || global && row.name.starts_with("global_store");
        let names = if load {
            ["RT", "NT", "HT", "LU", "NT_RT", "RT_NT", "NT_HT", "BYPASS"]
        } else if store {
            ["RT", "NT", "HT", "BYPASS", "NT_RT", "RT_NT", "NT_HT", "NT_WB"]
        } else {
            return Err(bad_operand(row.name, text, "cache TH only modeled for VBUFFER and GLOBAL loads/stores"));
        };
        let prefix = if load { "TH_LOAD_" } else { "TH_STORE_" };
        let value = th.strip_prefix(prefix).and_then(|word| names.iter().position(|name| *name == word))
            .ok_or_else(|| bad_operand(row.name, text, "bad cache TH"))?;
        return Ok(Operand::CacheTh(value as u8));
    }
    if let Some(inner) = text.strip_prefix("offset:swizzle(BROADCAST,").and_then(|s| s.strip_suffix(')')) {
        if row.name != "ds_swizzle_b32" { return Err(bad_operand(row.name, text, "swizzle on non-swizzle opcode")); }
        let (group, lane) = inner.split_once(',').ok_or_else(|| bad_operand(row.name, text, "expected broadcast group and lane"))?;
        let group: u16 = group.parse().map_err(|_| bad_operand(row.name, text, "bad broadcast group"))?;
        let lane: u16 = lane.parse().map_err(|_| bad_operand(row.name, text, "bad broadcast lane"))?;
        if group > 32 || !group.is_power_of_two() || lane >= group { return Err(bad_operand(row.name, text, "invalid broadcast group or lane")); }
        return Ok(Operand::Imm(ImmField::DsOffset((32 - group) | lane << 5)));
    }
    if let Some((kind, value)) = text.split_once(':') {
        // VMEM offsets are signed 24-bit (`offset:-48` occurs in hipcc's
        // own `.s`); DS offsets are unsigned.
        let signed: Option<i32> = if let Some(hex) = value.strip_prefix("0x")
        {
            u32::from_str_radix(hex, 16).ok().map(|v| v as i32)
        } else if let Some(hex) = value.strip_prefix("-0x") {
            u32::from_str_radix(hex, 16).ok().map(|v| -(v as i32))
        } else {
            value.parse().ok()
        };
        let unsigned: Option<u32> = signed.and_then(|v| u32::try_from(v).ok());
        let bad = || bad_operand(row.name, text, "bad offset");
        match kind {
            "offset0" => {
                return Ok(Operand::Imm(ImmField::DsOffset0(
                    unsigned.ok_or_else(bad)? as u8,
                )));
            }
            "offset1" => {
                return Ok(Operand::Imm(ImmField::DsOffset1(
                    unsigned.ok_or_else(bad)? as u8,
                )));
            }
            "offset" => {
                if row.form == Form::Smem {
                    return Ok(Operand::Imm(ImmField::SmemDisplacement(
                        signed.ok_or_else(bad)?,
                    )));
                }
                if matches!(row.form, Form::Vmem(_)) {
                    return Ok(Operand::Imm(ImmField::VmemOffset(
                        signed.ok_or_else(bad)?,
                    )));
                }
                return Ok(Operand::Imm(ImmField::DsOffset(
                    unsigned.ok_or_else(bad)? as u16,
                )));
            }
            _ => {}
        }
    }
    Err(bad_operand(row.name, text, "unexpected trailing operand"))
}

fn parse_suffixes(
    row: &OpRow,
    suffixes: &[String],
    mods: &mut Modifiers,
) -> Result<(), ParseError> {
    if matches!(row.form, Form::Vop1Dpp | Form::Vop2Dpp) {
        mods.dpp = Some(peacemaker_ir::operand::Dpp { ctrl: 0x100, row_mask: 15, bank_mask: 15, bound_ctrl: false });
    }
    for suffix in suffixes {
        if suffix == "clamp" {
            mods.clamp = true;
        } else if matches!(suffix.as_str(), "glc" | "slc" | "dlc") && (row.form == Form::Smem || matches!(row.form, Form::Vmem(_))) {
            match suffix.as_str() {
                "glc" => mods.cpol.glc = true,
                "slc" => mods.cpol.slc = true,
                _ => mods.cpol.dlc = true,
            }
        } else if suffix == "mul:2" {
            mods.omod = Omod::Mul2;
        } else if suffix == "mul:4" {
            mods.omod = Omod::Mul4;
        } else if suffix == "div:2" {
            mods.omod = Omod::Div2;
        } else if let Some(list) = suffix.strip_prefix("op_sel:[") {
            parse_op_sel(row, suffix, list, false, mods)?;
        } else if let Some(list) = suffix.strip_prefix("op_sel_hi:[") {
            parse_op_sel(row, suffix, list, true, mods)?;
        } else if let Some(list) = suffix.strip_prefix("neg_lo:[") {
            mods.neg_lo = parse_trits(row, suffix, list)?;
        } else if let Some(list) = suffix.strip_prefix("neg_hi:[") {
            mods.neg_hi = parse_trits(row, suffix, list)?;
        } else if let Some(value) = suffix.strip_prefix("row_shl:") {
            let shift: u16 = value.parse().map_err(|_| bad_operand(row.name, suffix, "invalid DPP row shift"))?;
            if !(1..=15).contains(&shift) { return Err(bad_operand(row.name, suffix, "DPP shift outside 1..=15")); }
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP control on non-DPP row"))?.ctrl = 0x100 + shift;
        } else if let Some(value) = suffix.strip_prefix("row_share:") {
            let lane: u16 = value.parse().map_err(|_| bad_operand(row.name, suffix, "invalid DPP row-share lane"))?;
            if lane > 15 { return Err(bad_operand(row.name, suffix, "DPP row-share lane outside 0..=15")); }
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP control on non-DPP row"))?.ctrl = 0x150 + lane;
        } else if let Some(value) = suffix.strip_prefix("row_xmask:") {
            let mask = parse_number(value, row.name, suffix)?;
            if mask > 15 { return Err(bad_operand(row.name, suffix, "DPP xor mask exceeds four bits")); }
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP control on non-DPP row"))?.ctrl = 0x160 + mask;
        } else if let Some(value) = suffix.strip_prefix("row_mask:") {
            let mask = parse_number(value, row.name, suffix)?;
            if mask > 15 { return Err(bad_operand(row.name, suffix, "DPP row mask exceeds four bits")); }
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP mask on non-DPP row"))?.row_mask = mask as u8;
        } else if let Some(value) = suffix.strip_prefix("bank_mask:") {
            let mask = parse_number(value, row.name, suffix)?;
            if mask > 15 { return Err(bad_operand(row.name, suffix, "DPP bank mask exceeds four bits")); }
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP mask on non-DPP row"))?.bank_mask = mask as u8;
        } else if let Some(value) = suffix.strip_prefix("bound_ctrl:") {
            mods.dpp.as_mut().ok_or_else(|| bad_operand(row.name, suffix, "DPP bound control on non-DPP row"))?.bound_ctrl = match value {
                "0" => false,
                "1" => true,
                _ => return Err(bad_operand(row.name, suffix, "DPP bound control must be 0 or 1")),
            };
        } else {
            return Err(ParseError::BadModifier {
                name: row.name.into(),
                modifier: suffix.into(),
                reason: "unknown suffix".into(),
            });
        }
    }
    Ok(())
}

/// VOP3 `op_sel` ends in a bit-3 destination select; packed VOP3P
/// `op_sel` and `op_sel_hi` each contain three source selects only.
fn parse_op_sel(
    row: &OpRow,
    suffix: &str,
    list: &str,
    hi: bool,
    mods: &mut Modifiers,
) -> Result<(), ParseError> {
    let bad = |reason: &str| ParseError::BadModifier {
        name: row.name.into(),
        modifier: suffix.into(),
        reason: reason.into(),
    };
    let list = list.strip_suffix(']').ok_or_else(|| bad("unterminated list"))?;
    let entries: Vec<u8> = list
        .split(',')
        .map(|e| {
            e.trim().parse::<u8>().map_err(|_| bad("entries are 0/1"))
        })
        .collect::<Result<_, _>>()?;
    if entries.iter().any(|&e| e > 1) {
        return Err(bad("entries are 0/1"));
    }
    if row.form == Form::Vop3p {
        if entries.len() != 3 { return Err(bad("packed op_sel needs 3 entries")); }
        let value = entries[0] | entries[1] << 1 | entries[2] << 2;
        if hi { mods.op_sel_hi = value; } else { mods.op_sel = value; }
        return Ok(());
    }
    if hi {
        if entries.len() != 3 {
            return Err(bad("op_sel_hi needs 3 entries"));
        }
        mods.op_sel_hi = entries[0] | entries[1] << 1 | entries[2] << 2;
        return Ok(());
    }
    if entries.len() < 2 || entries.len() > 4 {
        return Err(bad("op_sel needs 2-4 entries"));
    }
    let mut value = 0u8;
    for (i, entry) in entries[..entries.len() - 1].iter().enumerate() {
        value |= entry << i;
    }
    value |= entries[entries.len() - 1] << 3;
    mods.op_sel = value;
    Ok(())
}

fn parse_trits(
    row: &OpRow,
    suffix: &str,
    list: &str,
) -> Result<u8, ParseError> {
    let bad = |reason: &str| ParseError::BadModifier {
        name: row.name.into(),
        modifier: suffix.into(),
        reason: reason.into(),
    };
    let list = list.strip_suffix(']').ok_or_else(|| bad("unterminated list"))?;
    let mut value = 0u8;
    for (i, entry) in list.split(',').enumerate() {
        if i >= 3 {
            return Err(bad("at most 3 entries"));
        }
        match entry.trim() {
            "0" => {}
            "1" => value |= 1 << i,
            _ => return Err(bad("entries are 0/1")),
        }
    }
    Ok(value)
}

/// Wait/clause immediates live in `mods`, decoded per counter layout.
fn parse_wait_clause(
    row: &OpRow,
    parsed: &mut ParsedOperands,
) -> Result<(), ParseError> {
    if parsed.arch != Arch::Gfx1201 && row.name.starts_with("s_waitcnt") {
        let raw = parsed.operands.iter().find_map(|operand| match operand {
            Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)) => Some(*n as u16),
            _ => None,
        }).ok_or_else(|| bad_operand(row.name, "", "missing wait immediate"))?;
        let mut wait = WaitImm::default();
        if row.name == "s_waitcnt" {
            for (counter, value, max) in [(Counter::Vm, (raw >> 10) as u8 & 63, 63), (Counter::Exp, raw as u8 & 7, 7), (Counter::Lgkm, (raw >> 4) as u8 & 63, 63)] {
                if value != max { wait.per_counter[counter as usize] = Some(value); }
            }
        } else {
            let counter = match row.name {
                "s_waitcnt_vscnt" => Some(Counter::Vs),
                "s_waitcnt_vmcnt" => Some(Counter::Vm),
                "s_waitcnt_lgkmcnt" => Some(Counter::Lgkm),
                _ => None,
            };
            if let Some(counter) = counter { wait.per_counter[counter as usize] = Some(raw as u8 & 63); }
        }
        parsed.mods.wait = Some(wait);
        return Ok(());
    }
    let raw = match parsed.operands.first() {
        Some(Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n)))
            if row.form == Form::Sopp =>
        {
            *n as u16
        }
        _ => return Ok(()),
    };
    if row.name == "s_delay_alu" || row.name == "s_wait_alu" {
        return Ok(());
    }
    if row.name == "s_clause" {
        if raw > 0xff {
            return Err(bad_operand(
                row.name,
                "clause",
                "clause window exceeds 8 bits",
            ));
        }
        parsed.mods.clause = Some(raw as u8);
        return Ok(());
    }
    if let Some(name) = row.name.strip_prefix("s_wait_") {
        let mut wait = WaitImm::default();
        let set = |wait: &mut WaitImm, counter: Counter, value: u8| {
            wait.per_counter[counter as usize] = Some(value);
        };
        if name == "loadcnt_dscnt" {
            set(&mut wait, Counter::Load, (raw >> 8) as u8 & 63);
            set(&mut wait, Counter::Ds, raw as u8 & 63);
        } else if name == "storecnt_dscnt" {
            set(&mut wait, Counter::Store, (raw >> 8) as u8 & 63);
            set(&mut wait, Counter::Ds, raw as u8 & 63);
        } else {
            let counter = match name {
                "loadcnt" => Counter::Load,
                "storecnt" => Counter::Store,
                "dscnt" => Counter::Ds,
                "kmcnt" => Counter::Km,
                "samplecnt" => Counter::Sample,
                "bvhcnt" => Counter::Bvh,
                _ => Counter::Exp,
            };
            set(&mut wait, counter, raw as u8);
        }
        parsed.mods.wait = Some(wait);
    }
    Ok(())
}

/// Bare numbers for delay/wait_alu lines: hex, negative hex, or decimal.
fn parse_number(text: &str, name: &str, operand: &str) -> Result<u16, ParseError> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x") {
        return u16::from_str_radix(hex, 16)
            .map_err(|_| bad_operand(name, operand, "bad hex immediate"));
    }
    if let Some(hex) = text.strip_prefix("-0x") {
        let value = u16::from_str_radix(hex, 16)
            .map_err(|_| bad_operand(name, operand, "bad hex immediate"))?;
        return Ok((-(value as i32)) as i16 as u16);
    }
    if let Some(magnitude) = text.strip_prefix('-') {
        let magnitude: i32 = magnitude
            .parse()
            .map_err(|_| bad_operand(name, operand, "bad decimal"))?;
        return Ok((-magnitude) as i16 as u16);
    }
    text.parse::<u32>()
        .map(|v| (v & 0xffff) as u16)
        .map_err(|_| bad_operand(name, operand, "bad decimal"))
}

/// Canonical don't-care defaults plus honored values re-derived from the
/// parsed operands/modifiers, mirroring the codec's own derivation.
fn derive_fields(row: &OpRow, parsed: &ParsedOperands) -> Result<FormFields, ParseError> {
    if row.form == Form::Vop3
        && !row.grammar.contains("SRC2:")
        && !row.benign_src2.is_empty()
        && row.fields.len() == 1
    {
        return Ok(FormFields::Vop3b { src2_unused: 0x80 });
    }
    if row.fields.is_empty() {
        return Ok(FormFields::None);
    }
    let mut ignored: SmallVec<[NamedField; 4]> = SmallVec::new();
    let mut honored: SmallVec<[NamedField; 4]> = SmallVec::new();
    // The codec emits src2_unused first for VOP3 rows without a SRC2
    // slot, then the remaining rules in table order; mirror that order
    // exactly (field lists compare order-sensitively).
    let vop3_no_src2 = row.form == Form::Vop3
        && !row.grammar.contains("SRC2:")
        && !row.benign_src2.is_empty();
    if vop3_no_src2 {
        let rule = row.fields.iter().find(|r| r.name == "src2_unused").ok_or_else(|| {
            bad_operand(row.name, "src2_unused", "table lacks src2 rule")
        })?;
        push_rule(row, parsed, rule, &mut ignored, &mut honored)?;
    }
    for rule in &row.fields {
        if vop3_no_src2 && rule.name == "src2_unused" {
            continue;
        }
        push_rule(row, parsed, rule, &mut ignored, &mut honored)?;
    }
    // A lone src2 don't-care keeps the compact Vop3b shape.
    if ignored.len() == 1
        && honored.is_empty()
        && ignored[0].name == "src2_unused"
    {
        return Ok(FormFields::Vop3b {
            src2_unused: ignored[0].value as u16,
        });
    }
    if ignored.is_empty() && honored.is_empty() {
        return Ok(FormFields::None);
    }
    Ok(FormFields::Bits { ignored, honored })
}
/// Compute one table field rule's canonical value and file it by class.
fn push_rule(
    row: &OpRow,
    parsed: &ParsedOperands,
    rule: &peacemaker_ir::isa::FieldRule,
    ignored: &mut SmallVec<[NamedField; 4]>,
    honored: &mut SmallVec<[NamedField; 4]>,
) -> Result<(), ParseError> {
    let value = match rule.name {
        "src2_unused" | "src1_unused" => 0x80,
        "w0_extra" | "w1_extra" | "wait_unused" | "vsrc_unused" => 0,
        "neg" => u32::from(parsed.mods.neg),
        "abs" => u32::from(parsed.mods.abs),
        "omod" => match parsed.mods.omod {
            Omod::None => 0,
            Omod::Mul2 => 1,
            Omod::Mul4 => 2,
            Omod::Div2 => 3,
        },
        "op_sel" => u32::from(parsed.mods.op_sel),
        "op_sel_hi" => u32::from(parsed.mods.op_sel_hi),
        "global_scope" => u32::from(parsed.mods.cpol.scope) << 18,
        "dpp_fi" => 0,
        "global_th" => u32::from(parsed.mods.cpol.th) << 20,
        "global_nv" | "global_sve" => 0,
        "vbuffer_format" => 0x800000,
        "vbuffer_offen" => {
            if parsed.operands.iter().any(|op| {
                matches!(op, Operand::Vmem(VmemToken::Offen))
            }) {
                0x40000000
            } else {
                0
            }
        }
        "vbuffer_scope" => u32::from(parsed.mods.cpol.scope) << 18,
        "vbuffer_th" => u32::from(parsed.mods.cpol.th) << 20,
        "vbuffer_idxen" | "vbuffer_nv" | "vbuffer_tfe" => 0,
        _ => {
            return Err(bad_operand(
                row.name,
                rule.name,
                "unmodelled table field",
            ));
        }
    };
    let field = NamedField { name: rule.name, value };
    if rule.class == FieldClass::Honored {
        honored.push(field);
    } else {
        ignored.push(field);
    }
    Ok(())
}

fn finish(
    row: &'static OpRow,
    mut parsed: ParsedOperands,
) -> Result<Inst, ParseError> {
    // Literals ride along as the instruction's literal word.
    for operand in &parsed.operands {
        if let Operand::Literal(value) = operand {
            if parsed.literal.is_some_and(|l| l != *value) {
                return Err(bad_operand(
                    row.name,
                    &format!("{value:#x}"),
                    "conflicting literals",
                ));
            }
            parsed.literal = Some(*value);
        }
    }
    // Cache policy for memory rows comes from scope operands.
    for operand in &parsed.operands {
        if let Operand::Scope(scope) = operand {
            parsed.mods.cpol.scope = match scope {
                CacheScope::Cu => 0,
                CacheScope::Se => 1,
                CacheScope::Dev => 2,
                CacheScope::Sys => 3,
            };
        }
        if let Operand::CacheTh(th) = operand { parsed.mods.cpol.th = *th; }
    }
    let fields = derive_fields(row, &parsed)?;
    Inst::from_parts(
        parsed.arch,
        row.op,
        row.form,
        fields,
        SmallVec::from_vec(parsed.operands),
        parsed.mods,
        parsed.literal,
        Provenance::default(),
    )
    .map_err(|e| ParseError::Invalid {
        name: row.name.into(),
        reason: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// `.s` source structure for `lift_text`
// ---------------------------------------------------------------------------

/// One classified source line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLine {
    /// 1-based source line number.
    pub line_no: usize,
    pub kind: SourceKind,
}

/// Classified source content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceKind {
    /// `name:` — maps to the following instruction ordinal.
    Label(String),
    /// `.directive …` — passed through verbatim to the assembler.
    Directive(String),
    /// One instruction line (comments stripped).
    Inst(String),
}

/// A parsed assembly source: labels resolve to instruction ordinals.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct SourceFile {
    pub lines: Vec<SourceLine>,
    /// `(label, instruction ordinal)` in source order.
    pub labels: Vec<(String, usize)>,
}

impl SourceFile {
    /// Number of instruction lines.
    pub fn inst_len(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| matches!(l.kind, SourceKind::Inst(_)))
            .count()
    }

    /// Instruction texts in order.
    pub fn insts(&self) -> Vec<&str> {
        self.lines
            .iter()
            .filter_map(|l| match &l.kind {
                SourceKind::Inst(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}

fn is_label_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'_' || byte == b'$'
}
/// Strip a trailing `//`, `#` or `;` comment, ignoring markers inside
/// double-quoted strings (e.g. `.ident` version strings holding URLs).
fn strip_source_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                in_string = !in_string;
                i += 1;
            }
            b'\\' if in_string && i + 1 < bytes.len() => i += 2,
            b'/' if !in_string
                && i + 1 < bytes.len()
                && bytes[i + 1] == b'/' =>
            {
                return line[..i].trim_end();
            }
            b'#' | b';' if !in_string => return line[..i].trim_end(),
            _ => i += 1,
        }
    }
    line
}

/// Parse `.s` structure: labels, directives, instruction lines.
pub fn parse_source(src: &str) -> Result<SourceFile, ParseError> {
    let mut file = SourceFile::default();
    let mut insts = 0usize;
    let mut in_metadata = false;
    for (index, raw) in src.lines().enumerate() {
        let line_no = index + 1;
        let mut line = raw.trim();
        // `.amdgpu_metadata` … `.end_amdgpu_metadata` is a YAML region,
        // not code: keep every line verbatim as a directive (its `#`
        // comments and `- key:` lines must not be reclassified).
        if line == ".amdgpu_metadata" {
            in_metadata = true;
        }
        if in_metadata {
            if !line.is_empty() {
                file.lines.push(SourceLine {
                    line_no,
                    kind: SourceKind::Directive(line.to_owned()),
                });
            }
            if line == ".end_amdgpu_metadata" {
                in_metadata = false;
            }
            continue;
        }
        // Comments (`//`, `#`, `;`) end the line, unless quoted
        // (e.g. `.ident "… (https://…)"`).
        line = strip_source_comment(line);
        if line.is_empty() {
            continue;
        }
        if line.ends_with(':') {
            let name = line[..line.len() - 1].trim().to_owned();
            if name.is_empty() || !name.bytes().all(is_label_char) {
                return Err(ParseError::BadSource {
                    line: raw.into(),
                    reason: "bad label".into(),
                });
            }
            if file.labels.iter().any(|(l, _)| *l == name) {
                return Err(ParseError::BadSource {
                    line: raw.into(),
                    reason: "duplicate label".into(),
                });
            }
            file.labels.push((name.clone(), insts));
            file.lines.push(SourceLine {
                line_no,
                kind: SourceKind::Label(name),
            });
        } else if line.starts_with('.') {
            file.lines.push(SourceLine {
                line_no,
                kind: SourceKind::Directive(line.to_owned()),
            });
        } else {
            file.lines.push(SourceLine {
                line_no,
                kind: SourceKind::Inst(line.to_owned()),
            });
            insts += 1;
        }
    }
    Ok(file)
}

/// Block identifiers for parsed labels (source order).
pub fn label_block_ids(file: &SourceFile) -> Vec<(String, BlockId)> {
    file.labels
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (name.clone(), BlockId(i)))
        .collect()
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;

/// True16 `op_sel` is redundant with the `.h`/`.l` halves: hipcc's own
/// `.s` omits the suffix while objdump prints it, and both assemble to
/// the same bytes (core.md §0). When the suffix is absent, derive it
/// from the halves (sources to bits 0.., destination half to bit 3);
/// when present, it must agree with them.
fn imply_op_sel(
    row: &OpRow,
    suffixes: &[String],
    mods: &mut Modifiers,
    operands: &[Operand],
) -> Result<(), ParseError> {
    if !matches!(row.form, Form::Vop3 | Form::Vop3p) {
        return Ok(());
    }
    let slots = grammar_slots(row);
    if !slots.iter().any(|(_, bits)| *bits == 16) {
        return Ok(());
    }
    if operands.len() != slots.len() {
        return Ok(());
    }
    let mut implied = 0u8;
    for ((slot, _), operand) in slots.iter().zip(operands.iter()) {
        let half_hi = matches!(operand, Operand::Half(_, Half::Hi));
        match *slot {
            "SRC0" => implied |= u8::from(half_hi),
            "SRC1" => implied |= u8::from(half_hi) << 1,
            "SRC2" => implied |= u8::from(half_hi) << 2,
            "VDST" => implied |= u8::from(half_hi) << 3,
            _ => {}
        }
    }
    if suffixes.iter().any(|s| s.starts_with("op_sel:[")) {
        if mods.op_sel != implied {
            return Err(ParseError::BadModifier {
                name: row.name.into(),
                modifier: format!("op_sel vs halves {implied:#x}"),
                reason: "op_sel suffix contradicts the .h/.l halves".into(),
            });
        }
        return Ok(());
    }
    mods.op_sel = implied;
    Ok(())
}
